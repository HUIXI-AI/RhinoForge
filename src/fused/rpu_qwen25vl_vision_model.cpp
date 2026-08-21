// rpu_qwen25vl_vision_model.cpp — Qwen2.5-VL Vision Encoder
//
// 32-block ViT encoder for Qwen2.5-VL-3B (depth=32, hidden=1280,
// intermediate=3420→padded 3456, num_heads=16, head_dim=80). Differences from
// Qwen3-VL ViT:
//   - Norm: RMSNorm (single weight, no bias) — Qwen3-VL used LayerNorm (γ+β).
//   - MLP : SwiGLU `down(silu(gate(x)) * up(x))` with bias — Qwen3-VL used
//           fc1 → GELU(tanh) → fc2. gate/up/down all have bias.
//   - intermediate 3420 is not swizzle-representable (3420 % 128 = 92); the
//     Python adapter zero-pads gate/up/down (+biases) to 3456 and passes
//     intermediate_size=3456 → local_inter=432. silu(0)=0 keeps the pad inert.
//   - head_dim 80 (Qwen3-VL 64). 2D RoPE table is [max_hw, head_dim/4=20].
//   - Window attention uses fullatt_block_indexes; other layers receive a
//     dense block-diagonal mask.
//   - No DeepStack (that was Qwen3-VL specific).
//
// Patch embedding + window reorder + merger live in the Python adapter (CPU).
// The fused subsystem expects pre-embedded and pre-reordered input
// `[1, num_patches, hidden]`.

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"
#include "rpu_spm_allocator.h"
#include "rpu_spm_pipeline.h"
#include "graph/graph_runtime.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

constexpr int64_t QWEN25VL_VISION_MAX_KEEPALIVE_SEQ = 4096;
constexpr int64_t QWEN25VL_VISION_SDPA_QUERY_CHUNK = 144;
constexpr int64_t QWEN25VL_VISION_DEBUG_PHASE_LAYER = 17;
constexpr int64_t QWEN25VL_VISION_DEBUG_PHASE_COUNT = 3;
constexpr int64_t QWEN25VL_VISION_MIXED_BOUNDARY_LAYER = 17;

// First physical window-merger canary only. Keep this envelope intentionally
// narrow until the permutation route has proven BUILD -> stable REPLAY.
constexpr int64_t QWEN25VL_Z1_NUM_PATCHES = 256;
constexpr int64_t QWEN25VL_Z1_VISION_HIDDEN = 1280;
constexpr int64_t QWEN25VL_Z1_MERGED_ROWS = 64;
constexpr int64_t QWEN25VL_Z1_TEXT_HIDDEN = 2048;

// RPU_WALL_OSS_VISION_ROPE_SPM — default OFF in C++ (only this qwen25vl vision
// encoder reads it; wall_oss opts in default-on via build_wall_oss_vla). When
// on, the 2D-RoPE cos/sin tables are broadcast into SPM once at preload and the
// SPM-table kernel variant is used, so the per-token cos/sin gather hits SPM
// instead of DDR. Numerically identical to the DDR variant (same kernel math).
static bool vision_rope_spm_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_WALL_OSS_VISION_ROPE_SPM");
        cached = (e && (std::string(e) == "1" || std::string(e) == "true")) ? 1 : 0;
    }
    return cached != 0;
}

// RPU_WALL_OSS_VISION_FUSED_MERGER — default OFF. When on (and merger weights
// are registered) the post-encoder merger (ln_q RMSNorm -> m0 GEMM -> GELU ->
// m2 GEMM -> all-reduce) runs inside the qwen25vl_vision_forward graph via a
// post_fn. Exact MIMC runs the merger image-locally for its three chunks.
static bool vision_fused_merger_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_WALL_OSS_VISION_FUSED_MERGER");
        cached = (e && std::string(e) != "0" && std::string(e) != "false") ? 1 : 0;
    }
    return cached != 0;
}

// Fixed-shape mixed-precision mode, default OFF. Blocks 0..16 stay FP16;
// block 17's down projection transitions the residual stream to BF16, and
// blocks 18..31 keep only the residual in BF16 while every Linear/RMSNorm
// output consumed by the next projection remains FP16. The Linear weights may
// be all FP16 or exact hybrid W8A16 with block-17 down_proj retained as FP16.
static bool vision_mixed_precision_enabled() {
    static bool cached = []() -> bool {
        const char* e = std::getenv("RPU_WALL_OSS_VISION_MIXED_PRECISION");
        if (!e || !*e || std::strcmp(e, "0") == 0) return false;
        TORCH_CHECK(
            std::strcmp(e, "1") == 0,
            "RPU_WALL_OSS_VISION_MIXED_PRECISION accepts only 0 or 1, got ", e);
        return true;
    }();
    return cached;
}

// Optional layer-group schedule, default OFF. Value 32 processes all layers while
// traversing independent equal-size image chunks:
//   layer-group -> image-chunk -> layer.
// Each parallel_linear invocation still streams its weight from DDR.
static bool vision_layer_group_enabled() {
    static bool cached = []() -> bool {
        const char* e = std::getenv("RPU_WALL_OSS_VISION_LAYER_GROUP");
        if (!e || !*e || std::strcmp(e, "0") == 0) return false;
        TORCH_CHECK(std::strcmp(e, "32") == 0,
                    "RPU_WALL_OSS_VISION_LAYER_GROUP accepts only 0 or 32, got ", e);
        return true;
    }();
    return cached;
}

// Native default OFF; exact RTC W8 builders default it ON. Exact 3x576 only.
// The three independent images
// advance layer-major and every projection uses the current shape-selected
// auto-tile family. Attention and row-partition reductions remain image-local.
// The compatibility environment name selects the shape-based auto-tile family.
static bool vision_exact_mimc_enabled_from_legacy_env() {
    static bool cached = []() -> bool {
        const char* e = std::getenv("RPU_WALL_OSS_VISION_WEIGHT_OUTER");
        return e && *e && std::strcmp(e, "0") != 0
            && std::strcmp(e, "false") != 0
            && std::strcmp(e, "False") != 0;
    }();
    return cached;
}

namespace v3 {

struct ExactMimcPlan {
    static constexpr int64_t kImages = 3;
    static constexpr int64_t kQkvRows = 1728;
    static constexpr int64_t kAttentionRows = 576;
    static constexpr int64_t kMlpRows = 576;
    static constexpr int64_t kDownRows = 288;
    static constexpr int64_t kHiddenSize = 1280;
    static constexpr int64_t kCores = 8;
    static constexpr int64_t kResidualShardElements =
        kAttentionRows * kHiddenSize / kCores;
    static constexpr int64_t kResidualShardBytes =
        kResidualShardElements * DWIDTH;
    static constexpr int64_t kCompactResidualBytesPerCore =
        kImages * kResidualShardBytes;
    static constexpr int64_t kValidatedTempPeakBytes = 7'704'576;
    static constexpr int64_t kFullVlaSuperPersistentBytes = 591'360;
    static constexpr int64_t kEffectiveSpmBudgetBytes =
        static_cast<int64_t>(SpmAllocator::SPM_PLANNING_BUDGET &
                             ~(SpmAllocator::ALIGN - 1));
};
static_assert(ExactMimcPlan::kQkvRows
                  == ExactMimcPlan::kImages
                      * ExactMimcPlan::kAttentionRows
              && ExactMimcPlan::kAttentionRows == ExactMimcPlan::kMlpRows
              && ExactMimcPlan::kAttentionRows
                  == 2 * ExactMimcPlan::kDownRows);
static_assert(ExactMimcPlan::kResidualShardElements == 92'160
              && ExactMimcPlan::kResidualShardBytes == 184'320
              && ExactMimcPlan::kCompactResidualBytesPerCore == 552'960);
static_assert(ExactMimcPlan::kValidatedTempPeakBytes
                  + ExactMimcPlan::kFullVlaSuperPersistentBytes
              <= ExactMimcPlan::kEffectiveSpmBudgetBytes);

enum class VisionSchedule : int64_t {
    SISC,
    MISC,
    MIMC,
};

enum class VisionImplementation : int64_t {
    Generic,
    LegacyLayerGroup,
    ExactByMha,
};

struct VisionDispatch {
    VisionSchedule schedule;
    VisionImplementation implementation;
};

static VisionDispatch resolve_vision_dispatch(int64_t image_batch_count) {
    const bool legacy_layer_group = vision_layer_group_enabled();
    const bool exact_mimc = vision_exact_mimc_enabled_from_legacy_env();
    TORCH_CHECK(!(legacy_layer_group && exact_mimc),
                "RPU_WALL_OSS_VISION_WEIGHT_OUTER and "
                "RPU_WALL_OSS_VISION_LAYER_GROUP are mutually exclusive");
    if (exact_mimc) {
        // The total 3x576 input does not fit one encoder chunk and its packed
        // QKV/attention/MLP stages use different row chunking.
        return {VisionSchedule::MIMC, VisionImplementation::ExactByMha};
    }
    if (legacy_layer_group) {
        return {VisionSchedule::MIMC,
                VisionImplementation::LegacyLayerGroup};
    }
    return {image_batch_count == 1 ? VisionSchedule::SISC
                                   : VisionSchedule::MISC,
            VisionImplementation::Generic};
}

class Qwen25VLVisionModel : public FusedModelBase {
public:
    struct LayerWeights {
        // All [out, in] DDR, swizzled per partition (fp16 or W8A16 int8).
        at::Tensor q_w, k_w, v_w, o_w;        // attention (col/col/col/row partition)
        at::Tensor gate_w, up_w, down_w;       // SwiGLU MLP (col/col/row partition)
        at::Tensor q_ws, k_ws, v_ws, o_ws;      // optional fp16 per-output scales
        at::Tensor gate_ws, up_ws, down_ws;
    };

    struct LayerBiasNorm {
        at::Tensor norm1_w, norm2_w;           // RMSNorm (single weight, no bias)
        at::Tensor q_b, k_b, v_b, o_b;
        at::Tensor gate_b, up_b, down_b;
    };

    Qwen25VLVisionModel() = default;

    void set_configured_chunk_size(int64_t chunk_size) {
        TORCH_CHECK(chunk_size == 0
                        || (chunk_size >= 16 && chunk_size % 16 == 0),
                    "qwen25vl vision chunk_size must be 0 (auto) or a positive "
                    "multiple of 16, got ", chunk_size);
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "qwen25vl vision chunk_size must be set before first forward");
        configured_chunk_size_ = chunk_size;
        invalidate_model_state();
    }

    // ========================================================================
    // set_weights — per-layer weight + bias storage.
    //
    // QKV bias present (Qwen2.5-VL vision attn `qkv` Linear bias=True; the
    // adapter pre-splits the fused [3*dim, dim] weight into q/k/v [dim, dim] and
    // the [3*dim] bias into q_b/k_b/v_b). SwiGLU gate/up/down all have bias;
    // gate/up come from the fused `gate_up_proj` split + zero-pad to
    // intermediate_size (3456).
    // ========================================================================
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList gate_w_list, at::TensorList up_w_list, at::TensorList down_w_list,
        at::TensorList norm1_w_list, at::TensorList norm2_w_list,
        at::TensorList q_b_list, at::TensorList k_b_list,
        at::TensorList v_b_list, at::TensorList o_b_list,
        at::TensorList gate_b_list, at::TensorList up_b_list, at::TensorList down_b_list,
        int64_t num_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps, at::IntArrayRef fullatt_block_indexes = {},
        at::TensorList q_w_scale_list = {},
        at::TensorList k_w_scale_list = {},
        at::TensorList v_w_scale_list = {},
        at::TensorList o_w_scale_list = {},
        at::TensorList gate_scale_list = {},
        at::TensorList up_scale_list = {},
        at::TensorList down_scale_list = {})
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "qwen25vl_vision_set_weights: empty weight lists");
        TORCH_CHECK(num_heads > 0 && head_dim > 0 && hidden_size > 0
                    && intermediate_size > 0,
                    "qwen25vl_vision_set_weights: dim params must be positive");

        auto check_list = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "qwen25vl_vision_set_weights: ", n, ".size()=", l.size(),
                        " != num_layers=", N);
        };
        check_list(k_w_list,    "k_w_list");
        check_list(v_w_list,    "v_w_list");
        check_list(o_w_list,    "o_w_list");
        check_list(gate_w_list, "gate_w_list");
        check_list(up_w_list,   "up_w_list");
        check_list(down_w_list, "down_w_list");
        check_list(norm1_w_list, "norm1_w_list");
        check_list(norm2_w_list, "norm2_w_list");
        check_list(q_b_list,    "q_b_list");
        check_list(k_b_list,    "k_b_list");
        check_list(v_b_list,    "v_b_list");
        check_list(o_b_list,    "o_b_list");
        check_list(gate_b_list, "gate_b_list");
        check_list(up_b_list,   "up_b_list");
        check_list(down_b_list, "down_b_list");

        const bool has_scale = !q_w_scale_list.empty();
        auto check_optional_scale_list = [&](const at::TensorList& list,
                                             const char* name) {
            if (has_scale) {
                check_list(list, name);
            } else {
                TORCH_CHECK(list.empty(),
                            "qwen25vl_vision_set_weights: ", name,
                            " must be empty unless all W8A16 scale lists are provided");
            }
        };
        check_optional_scale_list(k_w_scale_list,  "k_w_scale_list");
        check_optional_scale_list(v_w_scale_list,  "v_w_scale_list");
        check_optional_scale_list(o_w_scale_list,  "o_w_scale_list");
        check_optional_scale_list(gate_scale_list, "gate_scale_list");
        check_optional_scale_list(up_scale_list,   "up_scale_list");
        check_optional_scale_list(down_scale_list, "down_scale_list");
        if (has_scale) {
            check_list(q_w_scale_list, "q_w_scale_list");
        }

        auto check_rank = [&](const at::TensorList& list, const char* name, int64_t r) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "qwen25vl_vision_set_weights: ", name, "[", i, "] undefined");
                TORCH_CHECK(list[i].dim() == r,
                            "qwen25vl_vision_set_weights: ", name, "[", i, "] must be ",
                            r, "D, got ", list[i].dim(), "D");
            }
        };
        check_rank(q_w_list,    "q_w_list",    2);
        check_rank(k_w_list,    "k_w_list",    2);
        check_rank(v_w_list,    "v_w_list",    2);
        check_rank(o_w_list,    "o_w_list",    2);
        check_rank(gate_w_list, "gate_w_list", 2);
        check_rank(up_w_list,   "up_w_list",   2);
        check_rank(down_w_list, "down_w_list", 2);
        check_rank(norm1_w_list, "norm1_w_list", 1);
        check_rank(norm2_w_list, "norm2_w_list", 1);
        check_rank(q_b_list,    "q_b_list",    1);
        check_rank(k_b_list,    "k_b_list",    1);
        check_rank(v_b_list,    "v_b_list",    1);
        check_rank(o_b_list,    "o_b_list",    1);
        check_rank(gate_b_list, "gate_b_list", 1);
        check_rank(up_b_list,   "up_b_list",   1);
        check_rank(down_b_list, "down_b_list", 1);
        if (has_scale) {
            check_rank(q_w_scale_list,  "q_w_scale_list",  1);
            check_rank(k_w_scale_list,  "k_w_scale_list",  1);
            check_rank(v_w_scale_list,  "v_w_scale_list",  1);
            check_rank(o_w_scale_list,  "o_w_scale_list",  1);
            check_rank(gate_scale_list, "gate_scale_list", 1);
            check_rank(up_scale_list,   "up_scale_list",   1);
            check_rank(down_scale_list, "down_scale_list", 1);
            auto check_half = [&](const at::TensorList& list, const char* name) {
                for (int64_t i = 0; i < N; i++) {
                    TORCH_CHECK(list[i].scalar_type() == at::kHalf,
                                "qwen25vl_vision_set_weights: ", name, "[", i,
                                "] must be fp16, got ", list[i].scalar_type());
                }
            };
            check_half(q_w_scale_list,  "q_w_scale_list");
            check_half(k_w_scale_list,  "k_w_scale_list");
            check_half(v_w_scale_list,  "v_w_scale_list");
            check_half(o_w_scale_list,  "o_w_scale_list");
            check_half(gate_scale_list, "gate_scale_list");
            check_half(up_scale_list,   "up_scale_list");
            check_half(down_scale_list, "down_scale_list");
        }

        // Vision encoder is MHA (no GQA): num_kv_heads == num_q_heads.
        set_model_params(num_heads, num_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;
        orig_head_dim_ = head_dim;  // exact (80) — no pad.

        fullatt_block_indexes_.assign(fullatt_block_indexes.begin(),
                                      fullatt_block_indexes.end());

        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                gate_w_list[i], up_w_list[i], down_w_list[i],
                has_scale ? q_w_scale_list[i]  : at::Tensor(),
                has_scale ? k_w_scale_list[i]  : at::Tensor(),
                has_scale ? v_w_scale_list[i]  : at::Tensor(),
                has_scale ? o_w_scale_list[i]  : at::Tensor(),
                has_scale ? gate_scale_list[i] : at::Tensor(),
                has_scale ? up_scale_list[i]   : at::Tensor(),
                has_scale ? down_scale_list[i] : at::Tensor(),
            });
        }

        layer_bias_norm_.clear();
        layer_bias_norm_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_bias_norm_.push_back({
                norm1_w_list[i], norm2_w_list[i],
                q_b_list[i], k_b_list[i], v_b_list[i], o_b_list[i],
                gate_b_list[i], up_b_list[i], down_b_list[i],
            });
        }

        invalidate_model_state();
    }

    void set_per_window_sdpa(bool enabled) {
        TORCH_CHECK(!z1_adopted_ && !z1_bound_ &&
                        z1_prepared_num_patches_ == 0,
                    "qwen25vl per-window SDPA cannot be enabled on a Z1 handle");
        if (enabled) {
            TORCH_CHECK(num_layers() > 0,
                        "qwen25vl per-window SDPA must be set after weights");
            TORCH_CHECK(fullatt_block_indexes_ ==
                            std::vector<int64_t>({7, 15, 23, 31}),
                        "qwen25vl per-window SDPA requires the controlled "
                        "32-layer window profile [7, 15, 23, 31]");
            TORCH_CHECK(
                        !vision_mixed_precision_enabled() &&
                            resolve_vision_dispatch(/*image_batch_count=*/1)
                                    .implementation
                                == VisionImplementation::Generic,
                        "qwen25vl per-window SDPA is incompatible with mixed "
                        "precision, layer grouping, and weight-outer scheduling");
        }
        per_window_sdpa_enabled_ = enabled;
        invalidate_model_state();
    }

    // ========================================================================
    // set_merger_weights — post-encoder merger weight storage (plumb
    // only, no compute). Merger: ln_q RMSNorm(HID) -> reshape[seq/4, 4*HID] ->
    // mlp.0 GEMM -> exact-erf GELU -> mlp.2 GEMM. Weights DDR-resident,
    // col-swizzled by the Python adapter.
    // ========================================================================
    void set_merger_weights(const at::Tensor& ln_q_w, const at::Tensor& m0_w,
                            const at::Tensor& m0_b, const at::Tensor& m2_w,
                            const at::Tensor& m2_b, int64_t out_hidden) {
        TORCH_CHECK(ln_q_w.dim()==1 && ln_q_w.size(0)==hidden_size(),
                    "merger ln_q_w must be [HID]");
        TORCH_CHECK(m0_w.dim()==2 && m0_w.size(0)==m0_w.size(1) && (m0_w.size(0)%128)==0,
                    "merger m0_w must be [merge_hidden, merge_hidden], merge_hidden%128==0");
        TORCH_CHECK(m2_w.dim()==2 && m2_w.size(0)==out_hidden && m2_w.size(1)==m0_w.size(0),
                    "merger m2_w must be [out_hidden, merge_hidden]");
        TORCH_CHECK(m0_b.dim()==1 && m0_b.size(0)==m0_w.size(0), "merger m0_b must be [merge_hidden]");
        TORCH_CHECK(m2_b.dim()==1 && m2_b.size(0)==out_hidden, "merger m2_b must be [out_hidden]");
        merger_ln_q_w_ = ln_q_w.contiguous();
        merger_m0_w_   = m0_w.contiguous();   merger_m0_b_ = m0_b.contiguous();
        merger_m2_w_   = m2_w.contiguous();   merger_m2_b_ = m2_b.contiguous();
        merge_hidden_  = m0_w.size(0);
        merger_out_hidden_ = out_hidden;
        // Affects static_config / declare_buffers / emit_preload_weights / post_fn — force a
        // graph rebuild if a forward already ran (mirrors qwen3vl set_merger_weights).
        invalidate_model_state();
    }

    // ========================================================================
    // set_rope_tables — register FreqCos / FreqSin static tables
    // [max_hw, head_dim/4] FP16 DDR. Also pre-allocs position_idx keepalive.
    // ========================================================================
    void set_rope_tables(const at::Tensor& freq_cos, const at::Tensor& freq_sin)
    {
        TORCH_CHECK(freq_cos.defined() && freq_sin.defined(),
                    "qwen25vl_vision_set_rope: freq_cos / freq_sin must be defined");
        TORCH_CHECK(freq_cos.dim() == 2 && freq_sin.dim() == 2,
                    "qwen25vl_vision_set_rope: freq_cos/sin must be 2D, got ",
                    freq_cos.dim(), "/", freq_sin.dim(), "D");
        TORCH_CHECK(freq_cos.sizes() == freq_sin.sizes(),
                    "qwen25vl_vision_set_rope: freq_cos / freq_sin shape mismatch");
        TORCH_CHECK(freq_cos.scalar_type() == at::kHalf && freq_sin.scalar_type() == at::kHalf,
                    "qwen25vl_vision_set_rope: freq_cos / freq_sin must be fp16");
        TORCH_CHECK(freq_cos.device().type() == at::kPrivateUse1
                    && freq_sin.device().type() == at::kPrivateUse1,
                    "qwen25vl_vision_set_rope: freq_cos / freq_sin must be on RPU device");
        TORCH_CHECK(freq_cos.is_contiguous() && freq_sin.is_contiguous(),
                    "qwen25vl_vision_set_rope: freq_cos / freq_sin must be contiguous");

        freq_cos_ = freq_cos;
        freq_sin_ = freq_sin;
        max_hw_ = freq_cos.size(0);

        if (!position_idx_keepalive_.defined()) {
            position_idx_keepalive_ = at::empty(
                {QWEN25VL_VISION_MAX_KEEPALIVE_SEQ, 2},
                at::TensorOptions().dtype(at::kShort).device(at::kPrivateUse1));
            position_idx_keepalive_.zero_();
        }

        has_rope_ = true;
        invalidate_model_state();
    }

    int64_t max_keepalive_seq() const { return QWEN25VL_VISION_MAX_KEEPALIVE_SEQ; }
    at::Tensor& position_idx_keepalive() { return position_idx_keepalive_; }

    // ========================================================================
    // forward — drive the 32-block encoder. Input/output [1, num_patches, hidden].
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches_in,
        std::optional<at::Tensor> window_mask = std::nullopt,
        int64_t image_batch_count = 1,
        std::optional<at::Tensor> cu_window_seqlens = std::nullopt)
    {
        TORCH_CHECK(!z1_adopted_,
                    "Qwen25VLVisionModel::forward: physical Z1 lease is active; "
                    "use the coordinator-owned Z1 dispatch");
        return forward_impl(input, k_caches, v_caches, num_patches_in,
                            std::move(window_mask), image_batch_count,
                            std::move(cu_window_seqlens),
                            /*z1_dispatch=*/false);
    }

    at::Tensor forward_impl(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches_in,
        std::optional<at::Tensor> window_mask,
        int64_t image_batch_count,
        std::optional<at::Tensor> cu_window_seqlens,
        bool z1_dispatch)
    {
        TORCH_CHECK(z1_dispatch == z1_bound_,
                    "Qwen25VLVisionModel::forward_impl: Z1 dispatch/binding mismatch");
        TORCH_CHECK(num_layers() > 0,
                    "Qwen25VLVisionModel::forward called before set_weights");
        TORCH_CHECK(has_rope_,
                    "Qwen25VLVisionModel::forward called before set_rope_tables");
        TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                    "Qwen25VLVisionModel::forward: input must be on RPU device");
        TORCH_CHECK(input.is_contiguous(),
                    "Qwen25VLVisionModel::forward: input must be contiguous");
        TORCH_CHECK(input.dim() == 3,
                    "Qwen25VLVisionModel::forward: input must be 3D [1,N,H], got ",
                    input.dim(), "D");
        TORCH_CHECK(num_patches_in > 0 && num_patches_in <= QWEN25VL_VISION_MAX_KEEPALIVE_SEQ,
                    "Qwen25VLVisionModel::forward: num_patches=", num_patches_in,
                    " must be in (0, MAX_KEEPALIVE_SEQ=", QWEN25VL_VISION_MAX_KEEPALIVE_SEQ, "]");
        TORCH_CHECK(input.size(0) == 1,
                    "Qwen25VLVisionModel::forward: batch must be 1, got ", input.size(0));
        TORCH_CHECK(input.size(1) == num_patches_in,
                    "Qwen25VLVisionModel::forward: input.size(1)=", input.size(1),
                    " must match num_patches=", num_patches_in);
        if (z1_dispatch) {
            TORCH_CHECK(RpuKernelGraph::has_active(),
                        "Qwen25VLVisionModel Z1 forward requires one active outer Graph");
            TORCH_CHECK(num_patches_in == QWEN25VL_Z1_NUM_PATCHES &&
                            image_batch_count == 1 &&
                            input.size(2) == QWEN25VL_Z1_VISION_HIDDEN &&
                            input.scalar_type() == at::kHalf,
                        "Qwen25VLVisionModel Z1 forward requires contiguous fp16 "
                        "[1, 256, 1280] input and image_batch_count=1");
            TORCH_CHECK(window_mask.has_value() && window_mask->defined(),
                        "Qwen25VLVisionModel Z1 forward requires the window-attention mask");
            TORCH_CHECK(z1_inputs_primed_ &&
                            z1_primed_window_mask_ref_.defined(),
                        "Qwen25VLVisionModel Z1 inputs must be primed outside Graph "
                        "before forward");
            TORCH_CHECK(window_mask->device().is_cpu() &&
                            window_mask->scalar_type() == at::kHalf &&
                            window_mask->is_contiguous() &&
                            window_mask->dim() == 2 &&
                            window_mask->size(0) == QWEN25VL_Z1_NUM_PATCHES &&
                            window_mask->size(1) == QWEN25VL_Z1_NUM_PATCHES,
                        "Qwen25VLVisionModel Z1 window mask must remain contiguous "
                        "CPU fp16 [256, 256]");
            TORCH_CHECK(window_mask->data_ptr() ==
                            z1_primed_window_mask_ptr_ &&
                            window_mask->sizes() ==
                                z1_primed_window_mask_ref_.sizes() &&
                            last_window_mask_ptr_ ==
                                z1_primed_window_mask_ptr_ &&
                            last_mask_seq_ == QWEN25VL_Z1_NUM_PATCHES,
                        "Qwen25VLVisionModel Z1 window mask address/shape changed "
                        "after Graph-external priming");
            TORCH_CHECK(!get_debug_export(),
                        "Qwen25VLVisionModel Z1 forward rejects debug export");
        }

        current_num_patches_ = num_patches_in;
        TORCH_CHECK(image_batch_count >= 1 && num_patches_in % image_batch_count == 0,
                    "Qwen25VLVisionModel::forward: image_batch_count=", image_batch_count,
                    " must be >=1 and divide num_patches=", num_patches_in);
        image_batch_count_ = image_batch_count;   // set before declare_buffers (chunk plan)
        // Resolve semantic topology separately from the concrete implementation.
        // Legacy env names remain startup-only compatibility selectors.
        const VisionDispatch dispatch =
            resolve_vision_dispatch(image_batch_count_);
        schedule_ = dispatch.schedule;
        implementation_ = dispatch.implementation;
        mixed_precision_ = vision_mixed_precision_enabled();

        const bool has_window_boundaries =
            cu_window_seqlens.has_value() && cu_window_seqlens->defined();
        TORCH_CHECK(has_window_boundaries == per_window_sdpa_enabled_,
                    per_window_sdpa_enabled_
                        ? "qwen25vl per-window SDPA requires cu_window_seqlens"
                        : "cu_window_seqlens requires a controlled per-window SDPA handle");
        cu_window_.clear();
        if (per_window_sdpa_enabled_) {
            TORCH_CHECK(!z1_dispatch && image_batch_count_ == 1,
                        "qwen25vl per-window SDPA supports only ordinary "
                        "single-image dispatch, not Z1 or multi-image");
            TORCH_CHECK(!mixed_precision_ && is_generic() &&
                            schedule_ == VisionSchedule::SISC,
                        "qwen25vl per-window SDPA is incompatible with mixed "
                        "precision, layer grouping, and weight-outer scheduling");
            TORCH_CHECK(!(window_mask.has_value() && window_mask->defined()),
                        "qwen25vl per-window SDPA must not receive a dense window mask");
            const at::Tensor& cu = *cu_window_seqlens;
            TORCH_CHECK(cu.device().is_cpu() && cu.scalar_type() == at::kLong &&
                            cu.is_contiguous() && cu.dim() == 1 && cu.numel() >= 2,
                        "qwen25vl cu_window_seqlens must be contiguous CPU int64 "
                        "with at least two boundaries");
            const int64_t* boundaries = cu.data_ptr<int64_t>();
            TORCH_CHECK(boundaries[0] == 0 &&
                            boundaries[cu.numel() - 1] == num_patches_in,
                        "qwen25vl cu_window_seqlens must start at 0 and end at "
                        "num_patches");
            for (int64_t i = 1; i < cu.numel(); ++i) {
                TORCH_CHECK(boundaries[i] > boundaries[i - 1] &&
                                boundaries[i] % 16 == 0,
                            "qwen25vl cu_window_seqlens must be strictly "
                            "increasing and 16-aligned");
            }
            cu_window_.assign(boundaries, boundaries + cu.numel());
        }
        if (mixed_precision_) {
            TORCH_CHECK(
                !is_legacy_layer_group(),
                "qwen25vl Vision mixed precision is incompatible with layer "
                "grouping");
            TORCH_CHECK(
                vision_fused_merger_enabled() && merger_out_hidden_ > 0,
                "qwen25vl Vision mixed precision requires the fused merger so "
                "the final BF16 residual is consumed before returning to Python");
            TORCH_CHECK(
                (schedule_ == VisionSchedule::SISC
                    && image_batch_count_ == 1 && num_patches_in == 576)
                    || (is_exact_mimc()
                        && image_batch_count_ ==
                            ExactMimcPlan::kImages
                        && num_patches_in ==
                            ExactMimcPlan::kQkvRows),
                "qwen25vl Vision mixed precision is validated only for one "
                "576-patch image or exact MIMC 3x576, got image_batch_count=",
                image_batch_count_, " num_patches=", num_patches_in);
            const bool pure_fp16 = intermediate_size() == 3456
                && !layer_weights_.front().q_ws.defined()
                && std::all_of(
                    layer_weights_.begin(), layer_weights_.end(),
                    [](const LayerWeights& lw) {
                        return lw.q_w.scalar_type() == at::kHalf
                            && lw.k_w.scalar_type() == at::kHalf
                            && lw.v_w.scalar_type() == at::kHalf
                            && lw.o_w.scalar_type() == at::kHalf
                            && lw.gate_w.scalar_type() == at::kHalf
                            && lw.up_w.scalar_type() == at::kHalf
                            && lw.down_w.scalar_type() == at::kHalf;
                    });
            const bool hybrid_w8a16 = intermediate_size() == 3584
                && layer_weights_.front().q_ws.defined() && [&] {
                    for (size_t i = 0; i < layer_weights_.size(); ++i) {
                        const auto& lw = layer_weights_[i];
                        if (lw.q_w.scalar_type() != at::kChar
                            || lw.k_w.scalar_type() != at::kChar
                            || lw.v_w.scalar_type() != at::kChar
                            || lw.o_w.scalar_type() != at::kChar
                            || lw.gate_w.scalar_type() != at::kChar
                            || lw.up_w.scalar_type() != at::kChar
                            || lw.down_w.scalar_type()
                                != (i == QWEN25VL_VISION_MIXED_BOUNDARY_LAYER
                                    ? at::kHalf : at::kChar)) {
                            return false;
                        }
                    }
                    return true;
                }();
            TORCH_CHECK(
                num_layers() == 32 && hidden_size() == 1280
                    && num_q_heads() == 16 && head_dim() == 80
                    && (pure_fp16 || hybrid_w8a16),
                "qwen25vl Vision mixed precision requires the Wall-OSS "
                "32xH1280/heads16/head80 profile with either I3456 all-FP16 "
                "weights or I3584 exact W8A16 weights with only block-17 "
                "down_proj FP16");
            TORCH_CHECK(
                !is_exact_mimc() || hybrid_w8a16,
                "qwen25vl Vision mixed exact MIMC supports only the exact "
                "I3584 hybrid W8A16 profile");
            TORCH_CHECK(
                !get_debug_export(),
                "qwen25vl Vision debug export tags raw BF16 residual taps as "
                "FP16 and is disabled in mixed-precision mode");
        }
        if (is_exact_mimc()) {
            TORCH_CHECK(
                image_batch_count_ == ExactMimcPlan::kImages
                    && num_patches_in == ExactMimcPlan::kQkvRows,
                "qwen25vl vision exact MIMC is validated only for 3x576 "
                "patches, got image_batch_count=", image_batch_count_,
                " num_patches=", num_patches_in);
            if (!mixed_precision_) {
                for (const auto& lw : layer_weights_) {
                    TORCH_CHECK(
                        lw.q_w.scalar_type() == at::kHalf
                            && lw.k_w.scalar_type() == at::kHalf
                            && lw.v_w.scalar_type() == at::kHalf
                            && lw.o_w.scalar_type() == at::kHalf
                            && lw.gate_w.scalar_type() == at::kHalf
                            && lw.up_w.scalar_type() == at::kHalf
                            && lw.down_w.scalar_type() == at::kHalf,
                        "qwen25vl vision non-mixed exact MIMC supports "
                        "fp16 weights only");
                }
            }
        }
        if (is_legacy_layer_group()) {
            TORCH_CHECK(image_batch_count_ == 3 && num_patches_in == 3 * 576,
                        "experimental qwen25vl vision layer grouping is validated "
                        "only for 3x576 patches, got image_batch_count=",
                        image_batch_count_, " num_patches=", num_patches_in);
        }

        // Validation-only per-layer tap. Keep one stable DDR allocation for the
        // lifetime of this handle because its address is baked into the captured
        // graph's SPM->DDR nodes. The Gate-A diagnostic deliberately runs one
        // 576-patch image at a time (face BUILD, wrist REPLAY); reject shape or
        // schedule changes instead of leaving an old cached graph with a stale
        // destination pointer.
        if (get_debug_export()) {
            TORCH_CHECK(
                image_batch_count_ == 1 && is_generic()
                    && schedule_ == VisionSchedule::SISC,
                "qwen25vl vision debug layer tap supports only the normal "
                "single-image schedule");
            const int64_t N = num_layers();
            const int64_t st = num_patches_in;
            const int64_t h = hidden_size();
            if (!per_layer_debug_buf_.defined()) {
                per_layer_debug_buf_ = at::empty(
                    {N, 1, st, h},
                    at::TensorOptions()
                        .dtype(at::kHalf)
                        .device(at::kPrivateUse1));
            } else {
                TORCH_CHECK(
                    per_layer_debug_buf_.dim() == 4
                        && per_layer_debug_buf_.size(0) == N
                        && per_layer_debug_buf_.size(1) == 1
                        && per_layer_debug_buf_.size(2) == st
                        && per_layer_debug_buf_.size(3) == h,
                    "qwen25vl vision debug layer tap is fixed-shape per handle; "
                    "start a fresh process for a different shape");
            }
            if (!phase_debug_buf_.defined()) {
                phase_debug_buf_ = at::empty(
                    {QWEN25VL_VISION_DEBUG_PHASE_COUNT, 1, st, h},
                    at::TensorOptions()
                        .dtype(at::kHalf)
                        .device(at::kPrivateUse1));
            } else {
                TORCH_CHECK(
                    phase_debug_buf_.dim() == 4
                        && phase_debug_buf_.size(0)
                            == QWEN25VL_VISION_DEBUG_PHASE_COUNT
                        && phase_debug_buf_.size(1) == 1
                        && phase_debug_buf_.size(2) == st
                        && phase_debug_buf_.size(3) == h,
                    "qwen25vl vision debug phase tap is fixed-shape per handle; "
                    "start a fresh process for a different shape");
            }
        }

        // Prepare the dense block-diagonal window mask once
        // per forward (stable DDR slot, Route B). Z1 already primed that slot
        // outside the outer Graph and only validates its stable identity here.
        // Upload to SPM still happens per window-layer because sdpa_mask aliases
        // temporary storage. Without a mask every layer is full attention.
        window_mask_present_ = window_mask.has_value() && window_mask->defined();
        if (window_mask_present_ && !z1_dispatch) {
            // Batched (image_batch_count_>1): window layers run per-image SDPA at
            // per_image_ctx, so the prepared mask is [per_image_ctx, per_image_ctx].
            int64_t mask_seq = (image_batch_count_ > 1)
                ? num_patches_in / image_batch_count_ : num_patches_in;
            // Skip the re-prepare (CPU→RPU upload + stable-slot copy) when the
            // caller passes the same (memoized) mask at the same shape — the slot
            // still holds it (no other same-shape user touches it between forwards).
            const void* wm_ptr = window_mask->data_ptr();
            if (wm_ptr != last_window_mask_ptr_ || mask_seq != last_mask_seq_) {
                prepared_window_mask_ = sdpa_prepare_mask(
                    window_mask, /*is_causal=*/false, mask_seq, mask_seq,
                    sdpa_stable_mask_cache());
                last_window_mask_ptr_ = wm_ptr;
                last_mask_seq_ = mask_seq;
            }
        } else if (z1_dispatch) {
            TORCH_CHECK(prepared_window_mask_.mask_type == 4 &&
                            prepared_window_mask_.ddr_tensor.defined(),
                        "Qwen25VLVisionModel Z1 prepared window mask is unavailable");
        }

        // Ordinary forward flushes after the adapter's per-forward copy. Z1 did
        // the same flush during Graph-external priming; repeating it here would
        // put host preparation into BUILD/REPLAY.
        if (!z1_dispatch) {
            rpu_ddr_flush_force_sized(
                position_idx_keepalive_.data_ptr<int16_t>(),
                static_cast<size_t>(num_patches_in * 2 * sizeof(int16_t)));
        }

        // Generic SISC/MISC stays single-chunk. Specialized implementations
        // expose one framework chunk per image; exact MIMC then applies its
        // stage-specific packed row chunking inside the layer builder.
        const int64_t logical_chunk = is_generic()
            ? num_patches_in : num_patches_in / image_batch_count_;
        const int64_t required_chunk = ((logical_chunk + 15) / 16) * 16;
        TORCH_CHECK(configured_chunk_size_ == 0
                        || configured_chunk_size_ == required_chunk,
                    "qwen25vl vision exact chunk_size must equal the native "
                    "single-chunk capacity: configured=", configured_chunk_size_,
                    " logical=", logical_chunk, " required=", required_chunk);
        set_chunk_size_override(is_generic()
            ? configured_chunk_size_ : required_chunk);

        at::Tensor result = run_all_layers(
            input, k_caches, v_caches,
            /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false);

        if (get_debug_export() && per_layer_debug_buf_.defined()) {
            // Publish the stable live buffer as one tensor. The validation
            // harness snapshots it after each forward. get_debug_tensor()
            // performs the post-capture RPU-to-CPU synchronization.
            g_debug_tensors["qwen25vl_vision_layer_outputs"] =
                per_layer_debug_buf_;
            g_debug_tensors["qwen25vl_vision_layer17_phases"] =
                phase_debug_buf_;
        }
        return result;
    }

    SpmPipelineComponentLayout prepare_z1_layout(int64_t num_patches) {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen25VLVisionModel Z1 prepare must run outside Graph capture");
        TORCH_CHECK(!per_window_sdpa_enabled_,
                    "Qwen25VLVisionModel Z1 rejects per-window SDPA handles");
        TORCH_CHECK(!z1_adopted_,
                    "Qwen25VLVisionModel Z1 cannot prepare while a lease is active");
        TORCH_CHECK(num_patches == QWEN25VL_Z1_NUM_PATCHES,
                    "Qwen25VLVisionModel Z1 canary admits exactly ",
                    QWEN25VL_Z1_NUM_PATCHES, " patches, got ", num_patches);
        TORCH_CHECK(num_layers() == 32 && num_q_heads() == 16 &&
                        head_dim() == 80 &&
                        hidden_size() == QWEN25VL_Z1_VISION_HIDDEN &&
                        (intermediate_size() == 3456 ||
                         intermediate_size() == 3584),
                    "Qwen25VLVisionModel Z1 requires the exact Wall-OSS "
                    "32-layer Vision profile");
        TORCH_CHECK(fullatt_block_indexes_ ==
                        std::vector<int64_t>({7, 15, 23, 31}),
                    "Qwen25VLVisionModel Z1 requires the validated window path "
                    "with full-attention blocks [7, 15, 23, 31]");
        TORCH_CHECK(merger_active() && merge_hidden_ == 5120 &&
                        merger_out_hidden_ == QWEN25VL_Z1_TEXT_HIDDEN,
                    "Qwen25VLVisionModel Z1 requires the fused [1280 -> 2048] merger");
        TORCH_CHECK(!get_debug_export(),
                    "Qwen25VLVisionModel Z1 prepare rejects debug export");
        TORCH_CHECK(
                    resolve_vision_dispatch(/*image_batch_count=*/1)
                            .implementation
                        == VisionImplementation::Generic,
                    "Qwen25VLVisionModel Z1 requires the normal one-image schedule");

        z1_inputs_primed_ = false;
        z1_primed_window_mask_ref_ = at::Tensor();
        z1_primed_window_mask_ptr_ = nullptr;

        // These values participate in static/dynamic config and declare_buffers.
        // Seal the same exact state that forward_z1 will re-establish.
        current_num_patches_ = num_patches;
        image_batch_count_ = 1;
        schedule_ = VisionSchedule::SISC;
        implementation_ = VisionImplementation::Generic;
        window_mask_present_ = true;
        set_chunk_size_override(0);

        const int64_t resolved = resolve_chunk_size_for_shape(
            num_patches, /*position=*/0, /*attention_mask=*/std::nullopt,
            /*is_causal=*/false);
        TORCH_CHECK(resolved == num_patches,
                    "Qwen25VLVisionModel Z1 requires one full Vision chunk; "
                    "planner resolved ", resolved, " for ", num_patches,
                    " patches");

        LayoutContext layout;
        layout.chunk_size = resolved;
        layout.max_kv_seq_len = num_patches;
        layout.num_layers = num_layers();
        layout.use_attn_mask = false;
        layout.is_causal = false;
        auto prepared = prepare_spm_pipeline_component(layout);
        z1_prepared_num_patches_ = num_patches;
        return prepared;
    }

    void prime_z1_inputs(const at::Tensor& window_mask,
                         int64_t num_patches) {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen25VLVisionModel Z1 input priming must run outside Graph capture");
        TORCH_CHECK(num_patches == QWEN25VL_Z1_NUM_PATCHES &&
                        z1_prepared_num_patches_ == num_patches,
                    "Qwen25VLVisionModel Z1 input priming requires the prepared "
                    "256-patch profile");
        TORCH_CHECK(window_mask.defined() && window_mask.device().is_cpu() &&
                        window_mask.scalar_type() == at::kHalf &&
                        window_mask.is_contiguous() &&
                        window_mask.dim() == 2 &&
                        window_mask.size(0) == num_patches &&
                        window_mask.size(1) == num_patches,
                    "Qwen25VLVisionModel Z1 input priming requires contiguous "
                    "CPU fp16 [256, 256] window mask");
        TORCH_CHECK(position_idx_keepalive_.defined() &&
                        position_idx_keepalive_.device().type() ==
                            at::kPrivateUse1 &&
                        position_idx_keepalive_.scalar_type() == at::kShort &&
                        position_idx_keepalive_.dim() == 2 &&
                        position_idx_keepalive_.size(0) >= num_patches &&
                        position_idx_keepalive_.size(1) == 2,
                    "Qwen25VLVisionModel Z1 input priming requires the Vision "
                    "position keepalive [>=256, 2]");

        prepared_window_mask_ = sdpa_prepare_mask(
            c10::optional<at::Tensor>(window_mask),
            /*is_causal=*/false, num_patches, num_patches,
            sdpa_stable_mask_cache());
        TORCH_CHECK(prepared_window_mask_.mask_type == 4 &&
                        prepared_window_mask_.ddr_tensor.defined(),
                    "Qwen25VLVisionModel Z1 failed to prepare its window mask");
        last_window_mask_ptr_ = window_mask.data_ptr();
        last_mask_seq_ = num_patches;
        z1_primed_window_mask_ref_ = window_mask;
        z1_primed_window_mask_ptr_ = window_mask.data_ptr();

        // The adapter populated the fixed-grid position table before priming.
        // Flush it here so neither BUILD nor REPLAY performs host preparation.
        rpu_ddr_flush_force_sized(
            position_idx_keepalive_.data_ptr<int16_t>(),
            static_cast<size_t>(num_patches * 2 * sizeof(int16_t)));
        z1_inputs_primed_ = true;
    }

    void rollback_z1_inputs() {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen25VLVisionModel Z1 input rollback must run outside Graph capture");
        TORCH_CHECK(!z1_adopted_ && !z1_bound_,
                    "Qwen25VLVisionModel Z1 input rollback requires a primed-only component");
        z1_inputs_primed_ = false;
        z1_primed_window_mask_ref_ = at::Tensor{};
        z1_primed_window_mask_ptr_ = nullptr;
        prepared_window_mask_ = PreparedMask{};
        last_window_mask_ptr_ = nullptr;
        last_mask_seq_ = -1;
    }

    void unprepare_z1_layout() {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen25VLVisionModel Z1 unprepare must run outside Graph capture");
        TORCH_CHECK(!z1_adopted_ && !z1_bound_,
                    "Qwen25VLVisionModel Z1 unprepare requires no active component lease");
        rollback_z1_inputs();
        z1_prepared_num_patches_ = 0;
    }

    SpmDense2DSpec z1_source_spec(int64_t num_patches) const {
        TORCH_CHECK(num_patches == QWEN25VL_Z1_NUM_PATCHES,
                    "Qwen25VLVisionModel Z1 source admits exactly ",
                    QWEN25VL_Z1_NUM_PATCHES, " patches, got ", num_patches);
        TORCH_CHECK(hidden_size() == QWEN25VL_Z1_VISION_HIDDEN &&
                        merge_hidden_ == 5120 &&
                        merger_out_hidden_ == QWEN25VL_Z1_TEXT_HIDDEN,
                    "Qwen25VLVisionModel Z1 source requires the exact merger profile");
        SpmDense2DSpec spec;
        spec.rows = QWEN25VL_Z1_MERGED_ROWS;
        spec.cols = QWEN25VL_Z1_TEXT_HIDDEN;
        spec.validate();
        return spec;
    }

    void adopt_z1_layout(const SpmPipelineLease& lease,
                         const SpmTensorView& scratch) {
        TORCH_CHECK(z1_prepared_num_patches_ == QWEN25VL_Z1_NUM_PATCHES,
                    "Qwen25VLVisionModel Z1 prepare must precede adopt");
        TORCH_CHECK(!z1_adopted_ && !z1_bound_,
                    "Qwen25VLVisionModel Z1 lease is already active");
        TORCH_CHECK(z1_inputs_primed_,
                    "Qwen25VLVisionModel Z1 inputs must be primed before lease adoption");
        adopt_spm_pipeline_component(lease, scratch);
        z1_adopted_ = true;
        z1_epoch_ = lease.epoch();
        z1_plan_hash_ = lease.plan_hash();
    }

    void bind_z1_source(const SpmPipelineLease& lease,
                        const SpmPortView& source) {
        TORCH_CHECK(z1_adopted_ && !z1_bound_ &&
                        z1_epoch_ == lease.epoch() &&
                        z1_plan_hash_ == lease.plan_hash(),
                    "Qwen25VLVisionModel Z1 bind requires the adopted live lease");
        const auto expected = z1_source_spec(z1_prepared_num_patches_);
        TORCH_CHECK(source.spec() == expected &&
                        source.size_bytes() == expected.storage_bytes(),
                    "Qwen25VLVisionModel Z1 source port must be replicated fp16 "
                    "[64, 2048]");
        z1_source_addr_ = source.resolve_physical_addr(/*core=*/0, lease);
        z1_bound_ = true;
    }

    void validate_z1_layout(const SpmPipelineLease& lease) const {
        TORCH_CHECK(z1_adopted_ && z1_bound_ && z1_source_addr_ != 0,
                    "Qwen25VLVisionModel Z1 has no active source binding");
        TORCH_CHECK(z1_epoch_ == lease.epoch() &&
                        z1_plan_hash_ == lease.plan_hash(),
                    "Qwen25VLVisionModel Z1 has a stale lease binding");
        validate_spm_pipeline_component(lease);
    }

    void clear_z1_layout(uint64_t epoch, uint64_t plan_hash) {
        TORCH_CHECK(z1_adopted_ && z1_epoch_ == epoch &&
                        z1_plan_hash_ == plan_hash,
                    "Qwen25VLVisionModel Z1 clear received a stale lease token");
        release_spm_pipeline_component(epoch, plan_hash);
        z1_adopted_ = false;
        z1_bound_ = false;
        z1_source_addr_ = 0;
        z1_epoch_ = 0;
        z1_plan_hash_ = 0;
        z1_inputs_primed_ = false;
        z1_primed_window_mask_ref_ = at::Tensor();
        z1_primed_window_mask_ptr_ = nullptr;
    }

    void forward_z1(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches,
        std::optional<at::Tensor> window_mask,
        uint64_t epoch,
        uint64_t plan_hash) {
        RECORD_FUNCTION("qwen25vl_vision_forward_z1", {});
        TORCH_CHECK(z1_adopted_ && z1_bound_ &&
                        z1_epoch_ == epoch && z1_plan_hash_ == plan_hash,
                    "Qwen25VLVisionModel Z1 forward received a stale lease token");
        (void)forward_impl(input, k_caches, v_caches, num_patches,
                           std::move(window_mask), /*image_batch_count=*/1,
                           /*cu_window_seqlens=*/std::nullopt,
                           /*z1_dispatch=*/true);
    }

    void check_z1_destroy_allowed() const {
        TORCH_CHECK(!z1_adopted_ && !z1_bound_ && !z1_inputs_primed_ &&
                        z1_prepared_num_patches_ == 0,
                    "cannot destroy the Qwen25VL Vision handle while its Z1 "
                    "layout is prepared, primed, or active; clear the outer "
                    "GraphCache and unprepare Z1 first");
    }

protected:
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers = num_layers();
        cfg.preload_fn = static_cast<void(FusedModelBase::*)()>(
                             &Qwen25VLVisionModel::emit_preload_weights);
        cfg.cross_layer_batch_size = num_layers();
        // Fused merger (single- and batched-image): run the merger inside the
        // same capture after the encoder loop, returning [seq/4, oh] from the C++
        // op (batched: g per-image blocks of per_image_rows). static_config()
        // runs per-forward (run_all_layers step 1), AFTER forward() set
        // current_num_patches_/image_batch_count_, so the post_output_shape
        // (= seq/4 total merged rows) reflects this forward's seq.
        if (merger_active()) {
            cfg.post_fn = static_cast<void(FusedModelBase::*)()>(
                              &Qwen25VLVisionModel::merger_post_fn);
            cfg.post_output_shape = { current_num_patches_ / 4, merger_out_hidden_ };
        }
        return cfg;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        if (is_exact_mimc()) {
            TORCH_CHECK(
                plan.chunk_size == ExactMimcPlan::kAttentionRows
                    && plan.num_chunks == ExactMimcPlan::kImages,
                "qwen25vl vision exact MIMC plan must be 3x576 chunks: "
                "chunk_size=", plan.chunk_size, " num_chunks=", plan.num_chunks);
            ModelDynamicConfig cfg;
            cfg.chunk_mode = ChunkMode::SEQUENTIAL;
            // Layer-major traversal lets one weight launch serve all images.
            // Packed hidden and the compact per-core O-residual shards stay in
            // SPM across all encoder layers.
            cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
            return cfg;
        }
        if (is_legacy_layer_group()) {
            const int64_t per_image = current_num_patches_ / image_batch_count_;
            TORCH_CHECK(plan.chunk_size == per_image
                        && plan.num_chunks == image_batch_count_,
                        "qwen25vl_vision layer-group plan must be one chunk/image: "
                        "chunk_size=", plan.chunk_size, " num_chunks=", plan.num_chunks,
                        " expected ", per_image, " x ", image_batch_count_);
            ModelDynamicConfig cfg;
            cfg.chunk_mode = ChunkMode::SEQUENTIAL;
            cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
            cfg.chunk_outer_within_group = true;
            return cfg;
        }
        TORCH_CHECK(schedule_ != VisionSchedule::MIMC,
                    "qwen25vl generic DDR MIMC is not implemented");
        // This vision encoder is hard-wired single-chunk: build_layer_subgraph uses GLOBAL
        // RoPE pos_offset/KV-insert position but a CHUNK-LOCAL SDPA kv_seq_len, so a >1-chunk
        // plan would silently degrade full/window attention to per-chunk self-attention. Fail
        // loudly instead. (If a real input ever exceeds the single-chunk SPM budget, implement
        // true multi-chunk: per-chunk.offset KV/positions + full-sequence kv_seq_len + mask.)
        TORCH_CHECK(plan.num_chunks == 1,
            "qwen25vl_vision requires a single chunk (got ", plan.num_chunks,
            "); multi-chunk would silently break cross-patch attention");
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
        return cfg;
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        int64_t cs = ctx.chunk_size;
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();   // padded (3456)

        int64_t local_q_dim = (nq / NUM_CORES) * hd;
        int64_t local_inter = is_ / NUM_CORES;   // 3456/8 = 432
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        int64_t res  = A(cs * h * DWIDTH);
        int64_t qkv  = A(cs * local_q_dim * DWIDTH);
        int64_t inter = A(cs * local_inter * DWIDTH);

        SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                            hd, /*nq*/nq, /*nkv*/nq,
                            /*cores*/NUM_CORES, /*mask*/0};
        // Size FLASH_ATTN scratch for the largest query call, not the packed/full
        // sequence. Batched vision calls SDPA per image; a long single image uses
        // 144-query strips below while retaining full-sequence K/V. This saves
        // ~320 KiB/core at seq=576 without changing attention or buffer lifetimes.
        const bool packed_batch = is_generic()
            && schedule_ == VisionSchedule::MISC;
        int64_t sdpa_seq = packed_batch
            ? (cs / image_batch_count_)
            : (cs > 256 ? std::min(cs, QWEN25VL_VISION_SDPA_QUERY_CHUNK) : cs);
        SdpaTiling t = sdpa_compute_tiling(sdpa_cfg, sdpa_seq);
        int64_t nkv_per_core = CeilDiv(nq, (int64_t)NUM_CORES);
        int64_t sdpa_tmp = A(t.tile_n_v16 * t.tile_k * nkv_per_core * CeilDiv(sdpa_seq, t.tile_m) * 32);

        auto dma_safe = [&](int64_t elems) -> int64_t {
            int64_t dma_elems = ((elems + 255) / 256) * 256;
            return A(dma_elems * DWIDTH);
        };
        int64_t norm_w_sz     = dma_safe(h);
        int64_t q_bias_sz     = dma_safe(local_q_dim);
        int64_t inter_bias_sz = dma_safe(local_inter);   // gate/up per-core 432
        int64_t full_bias_sz  = dma_safe(h);             // o/down full width 1280

        // Dense block-diagonal mask [seq_q, seq_k_v16*16] fp16, broadcast to
        // all cores. seq_k = seq_q = cs (single-chunk vision). Same formula as
        // Gemma (rpu_gemma_model.cpp). Only declared when windowing.
        // Batched: window SDPA is per-image (per_image_ctx queries), so the mask is
        // [per_image_ctx, per_image_ctx], not [cs, cs] — much smaller in SPM.
        int64_t mask_cs = packed_batch ? (cs / image_batch_count_) : cs;
        int64_t mask_sz = A(mask_cs * CeilDiv(mask_cs, (int64_t)16) * 32);

        int nl = static_cast<int>(num_layers());

        if (is_exact_mimc()) {
            const int64_t packed_seq = ExactMimcPlan::kQkvRows;
            const int64_t packed_res = A(packed_seq * h * DWIDTH);
            const int64_t packed_qkv =
                A(packed_seq * local_q_dim * DWIDTH);
            const int64_t mlp_rows = ExactMimcPlan::kMlpRows;
            const int64_t mlp_res = A(mlp_rows * h * DWIDTH);
            const int64_t mlp_inter =
                A(mlp_rows * local_inter * DWIDTH);
            const int64_t down_half_res =
                A(ExactMimcPlan::kDownRows * h * DWIDTH);
            TORCH_CHECK(
                packed_qkv == ExactMimcPlan::kCompactResidualBytesPerCore,
                "exact MIMC compact residual requires the fixed Wall "
                "H1280/heads16/head80 geometry; got packed_qkv=",
                packed_qkv);
            TORCH_CHECK(
                (is_ == 3456 && local_inter == 432)
                    || (is_ == 3584 && local_inter == 448),
                "exact MIMC compact residual requires the fixed Wall "
                "I3456 FP16 or I3584 W8A16 geometry; got intermediate_size=",
                is_, " local_inter=", local_inter);

            // One packed full-hidden slab is reserved for the whole layer and
            // changes roles in-place:
            //   norm1 output -> O partial/norm2 output -> down partial.
            // Reserving it through phase 9 makes the aliases visible to the
            // allocator throughout their live ranges.
            // The image-local MLP reuses separate residual/down-partial slabs,
            // so no extra full-hidden SPM copy is needed between layers.
            std::vector<BufferDecl> decls = {
                {"input_norm",   packed_res,   1,
                 merger_active() ? 15 : 9,
                 StorageClass::Temp, 0, nullptr},
                {"compact_residual",
                                  packed_qkv,   1, 9,
                                  StorageClass::Temp, 0, nullptr},
                {"oproj",        packed_res,   4, 7, StorageClass::Temp, 0, "input_norm"},

                // Stage 3 stays image-local at M576. M864 fits standalone
                // Vision, but not the full VLA's shared super-persistent floor.
                {"o_result",      mlp_res,      5, 9, StorageClass::Temp, 0, nullptr},
                {"down_partial",  down_half_res, 8, 9, StorageClass::Temp, 0, nullptr},

                {"q",            packed_qkv,   2, 4, StorageClass::Temp, 0, nullptr},
                {"k",            packed_qkv,   2, 3, StorageClass::Temp, 0, nullptr},
                {"v",            packed_qkv,   2, 3, StorageClass::Temp, 0, nullptr},

                // These must be independent roots: first-fit legitimately
                // reuses dead Q/K for live o_result, while compact residual
                // shards for later images remain live through the MLP loop.
                {"gate",         mlp_inter,    7, 8, StorageClass::Temp, 0, nullptr},
                {"up",           mlp_inter,    7, 7, StorageClass::Temp, 0, nullptr},

                // One current-layer parameter set. The layer-major graph
                // stages these addresses at the start of each layer, avoiding
                // the 32-copy persistent footprint of the normal schedule.
                {"norm1_w",   norm_w_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"norm2_w",   norm_w_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"q_bias",    q_bias_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"k_bias",    q_bias_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"v_bias",    q_bias_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"gate_bias", inter_bias_sz, 0, 0, StorageClass::Persistent, 0, nullptr},
                {"up_bias",   inter_bias_sz, 0, 0, StorageClass::Persistent, 0, nullptr},
                {"o_bias",    full_bias_sz,  0, 0, StorageClass::Persistent, 0, nullptr},
                {"down_bias", full_bias_sz,  0, 0, StorageClass::Persistent, 0, nullptr},
            };
            if (window_mask_present_) {
                decls.push_back(
                    {"sdpa_mask", mask_sz, 3, 3, StorageClass::Temp, 0, nullptr});
            }
            if (vision_rope_spm_enabled() && max_hw_ > 0) {
                int64_t rope_tbl_sz = A(max_hw_ * (hd / 4) * DWIDTH);
                decls.push_back(
                    {"rope_cos", rope_tbl_sz, 0, 0,
                     StorageClass::Persistent, 0, nullptr});
                decls.push_back(
                    {"rope_sin", rope_tbl_sz, 0, 0,
                     StorageClass::Persistent, 0, nullptr});
            }
            if (merger_active()) {
                const int64_t mrows = cs / 4;
                const int64_t local_mh = merge_hidden_ / NUM_CORES;
                const int64_t oh = merger_out_hidden_;
                const int64_t normed_sz = A(cs * h * DWIDTH);
                const int64_t mid_sz = A(mrows * local_mh * DWIDTH);
                const int64_t out_sz = A(mrows * oh * DWIDTH);

                // The merger runs after phase 9. Declare real overlapping
                // lifetimes (rather than aliases to encoder names) so its four
                // simultaneously-live operands are guaranteed distinct even
                // when first-fit reused encoder scratch aggressively.
                decls.push_back(
                    {"merger_normed", normed_sz, 10, 14,
                     StorageClass::Temp, 0, nullptr});
                decls.push_back(
                    {"merger_mid", mid_sz, 11, 13,
                     StorageClass::Temp, 0, nullptr});
                decls.push_back(
                    {"merger_zero", out_sz, 14, 14,
                     StorageClass::Temp, 0, nullptr});
                decls.push_back(
                    {"merger_out", out_sz, 14, 15,
                     StorageClass::Temp, 0, nullptr});
                decls.push_back(
                    {"merger_ln_q_w", dma_safe(h), 0, 0,
                     StorageClass::Persistent, 0, nullptr});
                decls.push_back(
                    {"merger_m0_bias", dma_safe(local_mh), 0, 0,
                     StorageClass::Persistent, 0, nullptr});
                decls.push_back(
                    {"merger_m2_bias", dma_safe(oh), 0, 0,
                     StorageClass::Persistent, 0, nullptr});
            }
            const int64_t temp_peak =
                detail::estimate_temporary_total(decls);
            TORCH_CHECK(
                temp_peak <= ExactMimcPlan::kValidatedTempPeakBytes &&
                    temp_peak
                        + ExactMimcPlan::kFullVlaSuperPersistentBytes
                    <= ExactMimcPlan::kEffectiveSpmBudgetBytes,
                "exact MIMC compact-residual layout exceeds its validated "
                "full-VLA SPM envelope: temp_peak=", temp_peak,
                " super_persistent=",
                ExactMimcPlan::kFullVlaSuperPersistentBytes,
                " budget=", ExactMimcPlan::kEffectiveSpmBudgetBytes);
            return decls;
        }

        std::vector<BufferDecl> decls = {
            {"residual1",  res,    1, 6, StorageClass::Temp, 0, nullptr},
            {"input_norm", res,    1, 6, StorageClass::Temp, 0, nullptr},
            {"oproj",      res,    4, 6, StorageClass::Temp, 0, nullptr},

            {"q",          qkv,    2, 3, StorageClass::Temp, 0, nullptr},
            {"k",          qkv,    2, 3, StorageClass::Temp, 0, nullptr},
            {"v",          qkv,    2, 3, StorageClass::Temp, 0, nullptr},
            {"sdpa_out",   qkv,    3, 4, StorageClass::Temp, 0, nullptr},
            {"sdpa_tmp",   sdpa_tmp, 3, 3, StorageClass::Temp, 0, nullptr},

            // SwiGLU: gate + up live across phases 5..6 (gate also holds the
            // silu*up product fed to down).
            {"gate",       inter,  5, 6, StorageClass::Temp, 0, nullptr},
            // `up` is live only in phase 5, when residual1 is dead, so the two
            // buffers safely share one slot.
            {"up",         inter,  5, 5, StorageClass::Temp, 0, "residual1"},

            {"norm1_w",    norm_w_sz,     0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"norm2_w",    norm_w_sz,     0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"q_bias",     q_bias_sz,     0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"k_bias",     q_bias_sz,     0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"v_bias",     q_bias_sz,     0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"gate_bias",  inter_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"up_bias",    inter_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"o_bias",     full_bias_sz,  0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"down_bias",  full_bias_sz,  0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        };
        if (window_mask_present_) {
            decls.push_back({"sdpa_mask", mask_sz, 3, 4, StorageClass::Temp, 0, nullptr});
        }
        // SPM-resident 2D-RoPE cos/sin tables [max_hw, head_dim/4] fp16 (shared
        // across all layers; broadcast once in emit_preload_weights). Tiny
        // (max_hw=128, hd/4=20 → ~5 KB each). See vision_rope_spm_enabled().
        if (vision_rope_spm_enabled() && max_hw_ > 0) {
            int64_t rope_tbl_sz = A(max_hw_ * (hd / 4) * DWIDTH);
            decls.push_back({"rope_cos", rope_tbl_sz, 0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"rope_sin", rope_tbl_sz, 0, 0, StorageClass::Persistent, 0, nullptr});
        }
        // Post-encoder merger (single- AND batched-image, UNIFIED — no per-g
        // branch). The merger is row-independent (RMSNorm per-row, GEMMs per-row,
        // GELU per-elem, all-reduce per-row), so one full-batch pass over the
        // packed residual1[seq,h] (seq = current_num_patches_ = g*n_i) has no
        // cross-iteration buffer reuse.
        //   RMSNorm(residual1[seq,h]) -> m0 GEMM col [mrows, merge_hidden] -> GELU
        //   -> m2 GEMM row partial [mrows, oh] -> all-reduce [mrows, oh] -> SPM->DDR.
        //
        // All 4 activation buffers ALIAS dead encoder Temps, so the merger adds 0
        // Fixed / 0 Temp peak AND the persistent set is g-INDEPENDENT (no merger
        // buffer is Persistent). g-independence is required because forward()
        // routes g=1 -> _forward_one and g>1 -> _forward_group on ONE shared
        // handle, and the framework hard-aborts on any persistent-set change after
        // the first forward (persistent layout hash lock).
        //
        // Lifetime (all reuses are sequentially safe — the graph executes encoder
        // nodes then post_fn nodes in emission order):
        //   merger_normed (->oproj):     RMSNorm out (1->2), then m2 PARTIAL (4->5).
        //   merger_mid    (->gate):      m0 col / GELU (2->4).
        //   merger_zero   (->input_norm):all-reduce ZERO residual (5).
        //   merger_out    (->residual1): all-reduce out (5->6).
        // WHY each alias target is DEAD at every merger reuse:
        //   - oproj / gate / input_norm are encoder Temps, written/read only inside
        //     the per-layer subgraph; the merger runs in post_fn, strictly AFTER the
        //     encoder loop, so they hold no live value.
        //   - residual1 is read in FULL by merger STEP 1 (RMSNorm input) and never
        //     touched by steps 2..4; merger_out (=residual1) is written at STEP 5,
        //     strictly after step 1 fully consumed it. The single SPM->DDR (step 6)
        //     reads merger_out and nothing afterwards overwrites it (no loop) -> no
        //     WAR race. The 3 all-reduce operands stay DISTINCT slots:
        //     partial=merger_normed(oproj), zero=merger_zero(input_norm),
        //     out=merger_out(residual1).
        // Sizes fit both g (cs=256 single big image, cs=768 batched g=3):
        //   normed = cs*h*DWIDTH == oproj(res) exactly;
        //   mid = mrows*local_mh*DWIDTH (80K@256/245K@768) <= gate(inter, 216K/663K);
        //   zero/out = mrows*oh*DWIDTH (256K@256/786K@768) <= input_norm/residual1
        //     (res, 640K@256/1.97M@768). All hold.
        if (merger_active()) {
            const int64_t mrows    = cs / 4;
            const int64_t local_mh = merge_hidden_ / NUM_CORES;   // 640
            const int64_t oh       = merger_out_hidden_;          // 2048
            const int64_t normed_sz = A(cs * h * DWIDTH);                 // == oproj (res)
            const int64_t mid_sz    = A(mrows * local_mh * DWIDTH);
            const int64_t out_sz    = A(mrows * oh * DWIDTH);            // zero & out
            decls.push_back({"merger_normed", normed_sz, 0, 0, StorageClass::Temp, 0, "oproj"});
            decls.push_back({"merger_mid",    mid_sz,    0, 0, StorageClass::Temp, 0, "gate"});
            decls.push_back({"merger_zero",   out_sz,    0, 0, StorageClass::Temp, 0, "input_norm"});
            decls.push_back({"merger_out",    out_sz,    0, 0, StorageClass::Temp, 0, "residual1"});
            decls.push_back({"merger_ln_q_w",  dma_safe(h),              0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"merger_m0_bias", dma_safe(local_mh),       0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"merger_m2_bias", dma_safe(oh),             0, 0, StorageClass::Persistent, 0, nullptr});
        }
        return decls;
    }

    // Two of these are re-set on EVERY forward and both reach non-aliased Temp
    // sizes, so the params hash alone could not tell two layouts apart:
    //   image_batch_count_   -> packed_batch -> sdpa_seq / mask_cs -> the
    //                           "sdpa_tmp" and "sdpa_mask" sizes. ONE handle
    //                           serves both single- and multi-image forwards
    //                           (adapters/wall_oss/vision.py routes g=1 and g>1
    //                           through the same handle), so 1x512 patches and
    //                           2x256 can resolve the same chunk_size and hash
    //                           identically while wanting different temps.
    //   window_mask_present_ -> whether "sdpa_mask" is declared at all, which
    //                           repacks the whole temp arena.
    // The resolved schedule and implementation select different graph topology
    // and declarations even when the outer tensor dimensions match.
    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(0, image_batch_count_);
        h = detail::layout_mix(h, window_mask_present_ ? 1 : 0);
        h = detail::layout_mix(h, static_cast<int64_t>(schedule_));
        h = detail::layout_mix(h, static_cast<int64_t>(implementation_));
        h = detail::layout_mix(h, mixed_precision_ ? 1 : 0);
        h = detail::layout_mix(h, per_window_sdpa_enabled_ ? 1 : 0);
        return h;
    }

    void emit_preload_weights() {
        int64_t h = hidden_size();
        int64_t local_q_dim = (num_q_heads() / NUM_CORES) * head_dim();
        int64_t local_inter = intermediate_size() / NUM_CORES;

        if (!is_exact_mimc()) {
            // Normal schedule: keep all 32 layers' small parameters resident.
            // Exact MIMC instead stages one shared set at each layer boundary
            // in emit_exact_mimc_layer_params().
            for (int64_t L = 0; L < num_layers(); ++L) {
                auto& bn = layer_bias_norm_[L];

                rpu_launch_memset_spm_multicore(layer_addr(L, 0, "o_bias"), h);
                rpu_launch_memset_spm_multicore(layer_addr(L, 0, "down_bias"), h);

                rpu_launch_ddr_broadcast_spm_dma(
                    bn.norm1_w.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "norm1_w"));
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.norm2_w.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "norm2_w"));

                rpu_launch_ddr_scatter_spm_dma(
                    bn.q_b.data_ptr<c10::Half>(),
                    local_q_dim, local_q_dim * DWIDTH,
                    layer_addr(L, 0, "q_bias"), /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.k_b.data_ptr<c10::Half>(),
                    local_q_dim, local_q_dim * DWIDTH,
                    layer_addr(L, 0, "k_bias"), /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.v_b.data_ptr<c10::Half>(),
                    local_q_dim, local_q_dim * DWIDTH,
                    layer_addr(L, 0, "v_bias"), /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.gate_b.data_ptr<c10::Half>(),
                    local_inter, local_inter * DWIDTH,
                    layer_addr(L, 0, "gate_bias"), /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.up_b.data_ptr<c10::Half>(),
                    local_inter, local_inter * DWIDTH,
                    layer_addr(L, 0, "up_bias"), /*num_cores=*/NUM_CORES);
            }

            // Row-partition biases live on core 0 only so all-reduce adds them
            // exactly once.
            for (int64_t L = 0; L < num_layers(); ++L) {
                auto& bn = layer_bias_norm_[L];
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.o_b.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "o_bias"), /*num_cores=*/1);
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.down_b.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "down_bias"), /*num_cores=*/1);
            }
        }

        // Global: broadcast the 2D-RoPE cos/sin tables into SPM once (SPM-rope
        // variant). Flush first — same coherency care the DDR rope launcher
        // takes per-forward for these exact tensors.
        if (vision_rope_spm_enabled() && max_hw_ > 0) {
            int64_t tbl_elems = max_hw_ * (head_dim() / 4);
            rpu_ddr_flush(freq_cos_.data_ptr<c10::Half>());
            rpu_ddr_flush(freq_sin_.data_ptr<c10::Half>());
            rpu_launch_ddr_broadcast_spm_dma(
                freq_cos_.data_ptr<c10::Half>(), tbl_elems, addr(0, "rope_cos"));
            rpu_launch_ddr_broadcast_spm_dma(
                freq_sin_.data_ptr<c10::Half>(), tbl_elems, addr(0, "rope_sin"));
        }

        // Merger ln_q weight + biases (single-image fused path). ln_q broadcast;
        // m0 bias col-scatter (merge_hidden/8 per core, matches the col GEMM);
        // m2 bias is row-partition => zero all cores then broadcast full to core
        // 0 only (mirror down_bias).
        if (merger_active()) {
            rpu_launch_ddr_broadcast_spm_dma(
                merger_ln_q_w_.data_ptr<c10::Half>(), hidden_size(),
                addr(0, "merger_ln_q_w"));
            rpu_launch_ddr_scatter_spm_dma(
                merger_m0_b_.data_ptr<c10::Half>(),
                merge_hidden_ / NUM_CORES, (merge_hidden_ / NUM_CORES) * DWIDTH,
                addr(0, "merger_m0_bias"), NUM_CORES);
            rpu_launch_memset_spm_multicore(addr(0, "merger_m2_bias"), merger_out_hidden_);
            rpu_launch_ddr_broadcast_spm_dma(
                merger_m2_b_.data_ptr<c10::Half>(), merger_out_hidden_,
                addr(0, "merger_m2_bias"), 1);
        }
    }

    void emit_exact_mimc_layer_params(int layer_idx) {
        const int64_t h = hidden_size();
        const int64_t local_q_dim =
            (num_q_heads() / NUM_CORES) * head_dim();
        const int64_t local_inter = intermediate_size() / NUM_CORES;
        auto& bn = layer_bias_norm_[layer_idx];

        rpu_launch_memset_spm_multicore(addr(0, "o_bias"), h);
        rpu_launch_memset_spm_multicore(addr(0, "down_bias"), h);
        rpu_launch_ddr_broadcast_spm_dma(
            bn.norm1_w.data_ptr<c10::Half>(), h, addr(0, "norm1_w"));
        rpu_launch_ddr_broadcast_spm_dma(
            bn.norm2_w.data_ptr<c10::Half>(), h, addr(0, "norm2_w"));
        rpu_launch_ddr_scatter_spm_dma(
            bn.q_b.data_ptr<c10::Half>(),
            local_q_dim, local_q_dim * DWIDTH,
            addr(0, "q_bias"), NUM_CORES);
        rpu_launch_ddr_scatter_spm_dma(
            bn.k_b.data_ptr<c10::Half>(),
            local_q_dim, local_q_dim * DWIDTH,
            addr(0, "k_bias"), NUM_CORES);
        rpu_launch_ddr_scatter_spm_dma(
            bn.v_b.data_ptr<c10::Half>(),
            local_q_dim, local_q_dim * DWIDTH,
            addr(0, "v_bias"), NUM_CORES);
        rpu_launch_ddr_scatter_spm_dma(
            bn.gate_b.data_ptr<c10::Half>(),
            local_inter, local_inter * DWIDTH,
            addr(0, "gate_bias"), NUM_CORES);
        rpu_launch_ddr_scatter_spm_dma(
            bn.up_b.data_ptr<c10::Half>(),
            local_inter, local_inter * DWIDTH,
            addr(0, "up_bias"), NUM_CORES);

        // Row-partition bias is applied by core 0 exactly once before reduce.
        rpu_launch_ddr_broadcast_spm_dma(
            bn.o_b.data_ptr<c10::Half>(), h, addr(0, "o_bias"), 1);
        rpu_launch_ddr_broadcast_spm_dma(
            bn.down_b.data_ptr<c10::Half>(), h, addr(0, "down_bias"), 1);
    }

    void build_exact_mimc_layer_subgraph(int layer_idx, int64_t per_image_seq) {
        TORCH_CHECK(
            per_image_seq == ExactMimcPlan::kAttentionRows
                && image_batch_count_ == ExactMimcPlan::kImages,
            "exact MIMC layer requires 3x576, got images=", image_batch_count_,
            " per_image_seq=", per_image_seq);

        const auto& lw = layer_weights_[layer_idx];
        const int64_t images = ExactMimcPlan::kImages;
        const int64_t packed_seq = ExactMimcPlan::kQkvRows;
        const int64_t h = hidden_size();
        const int64_t nq = num_q_heads();
        const int64_t hd = head_dim();
        const int64_t is_ = intermediate_size();
        const int64_t local_q_dim = (nq / NUM_CORES) * hd;
        const int64_t local_inter = is_ / NUM_CORES;
        const bool residual_is_bf16 =
            mixed_precision_
            && layer_idx > QWEN25VL_VISION_MIXED_BOUNDARY_LAYER;
        const bool boundary_down_is_bf16 =
            mixed_precision_
            && layer_idx == QWEN25VL_VISION_MIXED_BOUNDARY_LAYER;
        TORCH_CHECK(per_image_seq % 2 == 0,
                    "exact MIMC MLP requires an even per-image sequence");
        TORCH_CHECK(packed_seq % 2 == 0,
                    "exact MIMC MLP requires an even packed sequence");
        const int64_t mlp_rows = ExactMimcPlan::kMlpRows;
        const int64_t down_rows = ExactMimcPlan::kDownRows;
        const uint32_t h_image_bytes =
            static_cast<uint32_t>(per_image_seq * h * DWIDTH);
        const uint32_t q_image_bytes =
            static_cast<uint32_t>(per_image_seq * local_q_dim * DWIDTH);
        const uint32_t packed_h_bytes =
            static_cast<uint32_t>(packed_seq * h * DWIDTH);
        const uint32_t packed_q_bytes =
            static_cast<uint32_t>(packed_seq * local_q_dim * DWIDTH);
        auto image_chunk = [&](int64_t image_idx) {
            return ChunkInfo{
                static_cast<int>(image_idx),
                image_idx * per_image_seq,
                per_image_seq,
                per_image_seq,
            };
        };
        auto packed_linear = [&](uint32_t input_addr,
                                 const at::Tensor& weight,
                                 uint32_t output_addr,
                                 int64_t N, int64_t K,
                                 int partition, uint32_t bias_addr,
                                 const at::Tensor& scale) {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                input_addr, weight, output_addr,
                packed_seq, N, K, partition, NUM_CORES, bias_addr,
                /*force_gemm=*/false, scale);
        };

        // Stage one layer's small parameters into the shared persistent slots.
        emit_exact_mimc_layer_params(layer_idx);

        // Phase 1: layer 0 packs caller DDR directly into the full-hidden slab;
        // later layers consume the previous block output retained there. Each
        // core then extracts its own residual shard from its replicated input.
        for (int64_t i = 0; i < images; ++i) {
            const ChunkInfo chunk = image_chunk(i);
            const uint32_t norm_out = addr(0, "input_norm")
                + static_cast<uint32_t>(i) * h_image_bytes;
            if (layer_idx == 0) {
                emit_layer_input_dma(
                    layer_idx, chunk, "input_norm",
                    i * per_image_seq);
            }
            const uint32_t compact_residual = addr(0, "compact_residual")
                + static_cast<uint32_t>(i)
                    * ExactMimcPlan::kResidualShardBytes;
            rpu_launch_spm_local_shard_copy_dma(
                norm_out, ExactMimcPlan::kResidualShardElements,
                ExactMimcPlan::kResidualShardBytes,
                compact_residual, NUM_CORES);
            if (residual_is_bf16) {
                rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                    norm_out, norm_out, addr(0, "norm1_w"),
                    per_image_seq, h, eps_);
            } else {
                rpu_launch_rmsnorm_spm_kernel(
                    norm_out, norm_out, addr(0, "norm1_w"),
                    per_image_seq, h, eps_);
            }
        }

        // Phase 2: V first, then transpose it in SPM for the SISC by-MHA
        // layout. Reuse q as the row-major V staging slot before producing K/Q.
        packed_linear(
            addr(0, "input_norm"), lw.v_w, addr(0, "q"),
            nq * hd, h, /*col=*/1, addr(0, "v_bias"), lw.v_ws);
        rpu_launch_v_transpose_spm_batch3_576(
            addr(0, "q"), addr(0, "v"));
        packed_linear(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            nq * hd, h, /*col=*/1, addr(0, "k_bias"), lw.k_ws);
        packed_linear(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            nq * hd, h, /*col=*/1, addr(0, "q_bias"), lw.q_ws);

        // Phase 3: RoPE remains image-local because each image owns its 2-D
        // positions. The by-MHA kernel then consumes all three independent
        // batches in one launch; K/V never leave SPM or enter the KV cache.
        const int64_t local_heads = nq / NUM_CORES;
        const int64_t head_dim_pad = hd;
        const double attn_scale =
            1.0 / std::sqrt(static_cast<double>(orig_head_dim_));
        const bool is_full = !window_mask_present_ ||
            std::find(fullatt_block_indexes_.begin(),
                      fullatt_block_indexes_.end(),
                      static_cast<int64_t>(layer_idx))
                != fullatt_block_indexes_.end();
        const int mask_type = is_full ? 0 : prepared_window_mask_.mask_type;
        uint32_t mask_addr = 0;
        if (!is_full) {
            mask_addr = addr(0, "sdpa_mask");
            sdpa_dma_mask_to_spm(
                prepared_window_mask_, addr_offset("sdpa_mask").value,
                per_image_seq, per_image_seq, NUM_CORES);
        }

        for (int64_t i = 0; i < images; ++i) {
            const uint32_t q_abs =
                addr(0, "q") + static_cast<uint32_t>(i) * q_image_bytes;
            const uint32_t k_abs =
                addr(0, "k") + static_cast<uint32_t>(i) * q_image_bytes;
            const int64_t pos_offset = i * per_image_seq;
            if (vision_rope_spm_enabled()) {
                rpu_launch_rope_2d_spm_kernel(
                    q_abs, q_abs, addr(0, "rope_cos"), addr(0, "rope_sin"),
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    pos_offset, per_image_seq, local_heads,
                    hd, head_dim_pad, NUM_CORES);
                rpu_launch_rope_2d_spm_kernel(
                    k_abs, k_abs, addr(0, "rope_cos"), addr(0, "rope_sin"),
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    pos_offset, per_image_seq, local_heads,
                    hd, head_dim_pad, NUM_CORES);
            } else {
                rpu_launch_rope_2d_ddr_kernel(
                    q_abs, q_abs,
                    freq_cos_.data_ptr<c10::Half>(),
                    freq_sin_.data_ptr<c10::Half>(),
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    pos_offset, per_image_seq, local_heads,
                    hd, head_dim_pad, NUM_CORES);
                rpu_launch_rope_2d_ddr_kernel(
                    k_abs, k_abs,
                    freq_cos_.data_ptr<c10::Half>(),
                    freq_sin_.data_ptr<c10::Half>(),
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    pos_offset, per_image_seq, local_heads,
                    hd, head_dim_pad, NUM_CORES);
            }

        }
        // Q/output aliasing is not safe for this multi-block kernel. Its
        // normalized input is dead after Q/K/V, so use the tail of that larger
        // slab as scratch, then copy the completed attention tensor back to Q.
        const uint32_t attn_scratch = addr(0, "input_norm")
            + packed_h_bytes - packed_q_bytes;
        rpu_launch_sdpa_by_mha_spm_batch3_576(
            addr(0, "q"), addr(0, "k"), addr(0, "v"),
            attn_scratch, mask_addr, mask_type, attn_scale);
        rpu_launch_spm_local_copy_dma(
            attn_scratch, packed_seq * local_q_dim,
            addr(0, "q"), NUM_CORES);

        // Phase 4: packed O projection. Stage 3 is row-independent after the
        // attention residual, so run one no-bridge M576 chunk per image while
        // keeping norm/gate/up/down entirely in SPM.
        packed_linear(
            addr(0, "q"), lw.o_w, addr(0, "oproj"),
            h, nq * hd, /*row=*/0, addr(0, "o_bias"), lw.o_ws);
        for (int64_t mlp_begin = 0; mlp_begin < packed_seq;
             mlp_begin += mlp_rows) {
            const uint32_t packed_h_addr = addr(0, "input_norm")
                + static_cast<uint32_t>(mlp_begin * h * DWIDTH);

            const uint32_t partial_addr = addr(0, "oproj")
                + static_cast<uint32_t>(mlp_begin * h * DWIDTH);
            const uint32_t residual_local = addr(0, "compact_residual")
                + static_cast<uint32_t>(mlp_begin / mlp_rows)
                    * ExactMimcPlan::kResidualShardBytes;
            if (residual_is_bf16) {
                rpu_launch_all_reduce_fp16_partial_bf16_residual_local_spm_kernel(
                    partial_addr, residual_local, addr(0, "o_result"),
                    mlp_rows, h);
            } else {
                rpu_launch_all_reduce_sum_residual_local_spm_kernel(
                    partial_addr, residual_local, addr(0, "o_result"),
                    mlp_rows, h, NUM_CORES, NUM_CORES);
            }

            if (residual_is_bf16) {
                rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                    addr(0, "o_result"), packed_h_addr,
                    addr(0, "norm2_w"), mlp_rows, h, eps_);
            } else {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "o_result"), packed_h_addr,
                    addr(0, "norm2_w"), mlp_rows, h, eps_);
            }
            // Finish this image before the next O-ring overwrites o_result.
            // Independent gate/up roots leave o_result and the later images'
            // compact residual shards untouched.
            // Phases 7/8: gate/up share this larger row chunk. Down remains
            // 288-row tiled so its full-hidden partial can reuse one fixed slab.
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                packed_h_addr, lw.gate_w, addr(0, "gate"),
                mlp_rows, is_, h, /*col=*/1, NUM_CORES,
                addr(0, "gate_bias"), /*force_gemm=*/false, lw.gate_ws);
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "gate"), addr(0, "gate"),
                mlp_rows * local_inter, ValuOpType::SILU);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                packed_h_addr, lw.up_w, addr(0, "up"),
                mlp_rows, is_, h, /*col=*/1, NUM_CORES,
                addr(0, "up_bias"), /*force_gemm=*/false, lw.up_ws);
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
                mlp_rows * local_inter, ValuOpType::MUL,
                c10::Half(1.0));
            for (int64_t row = 0; row < mlp_rows; row += down_rows) {
                const uint32_t gate_row_addr = addr(0, "gate")
                    + static_cast<uint32_t>(row * local_inter * DWIDTH);
                const uint32_t residual_row_addr = addr(0, "o_result")
                    + static_cast<uint32_t>(row * h * DWIDTH);
                const uint32_t output_row_addr = packed_h_addr
                    + static_cast<uint32_t>(row * h * DWIDTH);
                if (boundary_down_is_bf16) {
                    rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(
                        gate_row_addr, lw.down_w, addr(0, "down_partial"),
                        down_rows, h, is_, /*row=*/0, NUM_CORES,
                        addr(0, "down_bias"));
                    rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(
                        addr(0, "down_partial"), residual_row_addr,
                        output_row_addr, down_rows, h);
                } else {
                    rpu_launch_linear_spm_to_spm_acc16_kernel(
                        gate_row_addr, lw.down_w, addr(0, "down_partial"),
                        down_rows, h, is_, /*row=*/0, NUM_CORES,
                        addr(0, "down_bias"), /*force_gemm=*/false, lw.down_ws);
                    if (residual_is_bf16) {
                        rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(
                            addr(0, "down_partial"), residual_row_addr,
                            output_row_addr, down_rows, h);
                    } else {
                        rpu_launch_all_reduce_sum_residual_kernel(
                            addr(0, "down_partial"), residual_row_addr,
                            output_row_addr, down_rows, h,
                            NUM_CORES, NUM_CORES);
                    }
                }
            }
        }

        // Inner layers stay entirely in SPM. Only a standalone encoder result
        // needs a final DDR publication; the fused merger consumes input_norm.
        if (layer_idx + 1 == num_layers() && !merger_active()) {
            for (int64_t i = 0; i < images; ++i) {
                const ChunkInfo chunk = image_chunk(i);
                emit_layer_output_dma(
                    layer_idx, chunk, "input_norm",
                    static_cast<uint32_t>(i) * h_image_bytes);
            }
        }
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        if (is_exact_mimc()) {
            // FusedModelBase traverses layer -> chunk in SPM_RESIDENT mode.
            // Chunk 0 emits one subgraph that consumes all three fixed chunks;
            // chunk 1/2 are intentionally empty.
            if (chunk.idx == 0) {
                build_exact_mimc_layer_subgraph(layer_idx, chunk.len);
            }
            return;
        }
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();        // padded (3456)
        int64_t local_inter = is_ / NUM_CORES;     // 432
        const bool residual_is_bf16 =
            mixed_precision_
            && layer_idx > QWEN25VL_VISION_MIXED_BOUNDARY_LAYER;
        auto emit_debug_phase = [&](int64_t phase, const char* source) {
            if (!get_debug_export()
                || !phase_debug_buf_.defined()
                || layer_idx != QWEN25VL_VISION_DEBUG_PHASE_LAYER) {
                return;
            }
            c10::Half* phase_base =
                phase_debug_buf_.data_ptr<c10::Half>()
                + phase * phase_debug_buf_.size(1)
                    * phase_debug_buf_.size(2) * h
                + chunk.offset * h;
            rpu_launch_spm_copy_ddr_dma(
                addr(0, source), phase_base, chunk.len * h);
        };

        // Phase 1: DDR→SPM input DMA + RMSNorm1 (residual1 → input_norm).
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }
        if (residual_is_bf16) {
            rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                addr(0, "residual1"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "norm1_w"), seq_len, h, eps_);
        } else {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual1"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "norm1_w"), seq_len, h, eps_);
        }
        emit_debug_phase(0, "input_norm");

        // Phase 2: Q / K / V Linear with bias (col-partition).
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "q_bias"),
            /*force_gemm=*/false, lw.q_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "k_bias"),
            /*force_gemm=*/false, lw.k_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "v_bias"),
            /*force_gemm=*/false, lw.v_ws);

        // Phase 2.5: 2D RoPE on Q and K (in-place). Each core holds nq/8=2 heads.
        const int64_t local_heads = nq / NUM_CORES;
        const int64_t head_dim_pad = hd;  // head_dim=80, no padding.
        if (vision_rope_spm_enabled()) {
            // SPM-resident cos/sin tables (broadcast once in emit_preload_weights).
            const uint32_t cos_a = addr(0, "rope_cos");
            const uint32_t sin_a = addr(0, "rope_sin");
            rpu_launch_rope_2d_spm_kernel(
                addr(0, "q"), addr(0, "q"), cos_a, sin_a,
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset, seq_len,
                local_heads, hd, head_dim_pad, NUM_CORES);
            rpu_launch_rope_2d_spm_kernel(
                addr(0, "k"), addr(0, "k"), cos_a, sin_a,
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset, seq_len,
                local_heads, hd, head_dim_pad, NUM_CORES);
        } else {
            rpu_launch_rope_2d_ddr_kernel(
                addr(0, "q"), addr(0, "q"),
                freq_cos_.data_ptr<c10::Half>(), freq_sin_.data_ptr<c10::Half>(),
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset, seq_len,
                local_heads, hd, head_dim_pad, NUM_CORES);
            rpu_launch_rope_2d_ddr_kernel(
                addr(0, "k"), addr(0, "k"),
                freq_cos_.data_ptr<c10::Half>(), freq_sin_.data_ptr<c10::Half>(),
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset, seq_len,
                local_heads, hd, head_dim_pad, NUM_CORES);
        }

        // Phase 3: KV cache insert (position=0) + bidirectional SDPA (MASK_NONE).
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        rpu_launch_insert_kcache_spm_unified(
            k_cache, 0 /*position*/, addr_offset("k").value,
            seq_len, nq, hd, NUM_CORES);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, 0 /*position*/, addr_offset("v").value,
            seq_len, nq, hd, NUM_CORES);

        double attn_scale = 1.0 / std::sqrt(static_cast<double>(orig_head_dim_));

        // Window layers get a dense block-diagonal MASK_2D; full-attention layers
        // and the no-window path stay MASK_NONE. The mask MUST be
        // re-uploaded before every window-layer SDPA — the sdpa_mask Temp slot is
        // phase-aliased and gets clobbered by the next layer's buffers.
        bool is_full = !(window_mask_present_ || per_window_sdpa_enabled_) ||
            std::find(fullatt_block_indexes_.begin(), fullatt_block_indexes_.end(),
                      static_cast<int64_t>(layer_idx)) != fullatt_block_indexes_.end();
        int mask_type = (is_full || per_window_sdpa_enabled_)
            ? 0 : prepared_window_mask_.mask_type;  // 0 / 4

        if (is_generic() && schedule_ == VisionSchedule::MISC) {
            // Batched multi-image: g per-image SDPA calls, each at the proven ≤256
            // config (full → MASK_NONE, window → shared [pic,pic] MASK_2D). Keeping
            // every SDPA at per_image_ctx (a) dodges the documented ≥512 unified
            // numeric failure and (b) lets sdpa_tmp be sized for 256 not the packed
            // 768 — the smaller scratch is what lets 3×256 fit a single chunk. image
            // i's K/V live at cache rows [i*pic:(i+1)*pic] (contiguous insert at 0).
            int64_t per_image_ctx = seq_len / image_batch_count_;
            uint32_t mask_off = 0;
            if (!is_full) {
                mask_off = addr_offset("sdpa_mask").value;
                sdpa_dma_mask_to_spm(prepared_window_mask_, mask_off,
                                     per_image_ctx, per_image_ctx, NUM_CORES);
            }
            int64_t chunks = per_image_ctx / 16;          // seq-chunks per image
            int64_t qd = (nq / NUM_CORES) * hd;           // per-core q stride (elems)
            for (int64_t i = 0; i < image_batch_count_; ++i) {
                auto k_slice = k_cache.narrow(1, i * chunks, chunks);
                auto v_slice = v_cache.narrow(1, i * chunks, chunks);
                TORCH_CHECK(k_slice.is_contiguous() && v_slice.is_contiguous(),
                            "per-image KV slice not contiguous; need an offset launcher");
                uint32_t q_off   = addr_offset("q").value
                                   + (uint32_t)(i * per_image_ctx * qd * DWIDTH);
                uint32_t out_off = addr_offset("sdpa_out").value
                                   + (uint32_t)(i * per_image_ctx * qd * DWIDTH);
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    k_slice, v_slice, mask_type, attn_scale,
                    q_off, out_off, addr_offset("sdpa_tmp").value, mask_off,
                    per_image_ctx, nq, nq, hd,
                    per_image_ctx, NUM_CORES, NUM_CORES);
            }
        } else if (!is_full && per_window_sdpa_enabled_) {
            // Controlled single-image path: window-reordered patches make each
            // [begin,end) range contiguous in Q/output and KV-cache storage. Run
            // MASK_NONE inside each window, avoiding the dense [seq,seq] mask.
            const int64_t qd = (nq / NUM_CORES) * hd;
            for (size_t wi = 0; wi + 1 < cu_window_.size(); ++wi) {
                const int64_t window_begin = cu_window_[wi];
                const int64_t window_len = cu_window_[wi + 1] - window_begin;
                auto k_slice = k_cache.narrow(
                    1, window_begin / 16, window_len / 16);
                auto v_slice = v_cache.narrow(
                    1, window_begin / 16, window_len / 16);
                TORCH_CHECK(k_slice.is_contiguous() && v_slice.is_contiguous(),
                            "per-window KV slice is not contiguous");
                const int64_t query_chunk = seq_len > 256
                    ? QWEN25VL_VISION_SDPA_QUERY_CHUNK : seq_len;
                for (int64_t q_rel = 0; q_rel < window_len;
                     q_rel += query_chunk) {
                    const int64_t query_len =
                        std::min(query_chunk, window_len - q_rel);
                    const int64_t query_begin = window_begin + q_rel;
                    const uint32_t q_off = addr_offset("q").value
                        + static_cast<uint32_t>(query_begin * qd * DWIDTH);
                    const uint32_t out_off = addr_offset("sdpa_out").value
                        + static_cast<uint32_t>(query_begin * qd * DWIDTH);
                    rpu_launch_sdpa_spm_unified_kernel_v2(
                        k_slice, v_slice, /*mask_type=*/0, attn_scale,
                        q_off, out_off, addr_offset("sdpa_tmp").value,
                        /*mask_off=*/0, query_len, nq, nq, hd,
                        window_len, NUM_CORES, NUM_CORES);
                }
            }
        } else {
            uint32_t mask_off = 0;
            if (!is_full) {
                mask_off = addr_offset("sdpa_mask").value;
                sdpa_dma_mask_to_spm(prepared_window_mask_, mask_off,
                                     seq_len, seq_len, NUM_CORES);
            }

            // The unified fp16 kernel is numerically unsafe for a 576-query
            // launch. Keep the full K/V cache (cross-patch attention remains
            // global) and only strip the Q/output rows, mirroring Qwen3-VL's
            // proven 144-query KV_FIRST compute phase. MASK_2D is row-major
            // [seq_q, aligned_seq_k], so each strip advances the mask base by
            // q_begin rows; MASK_NONE ignores the address.
            const int64_t qd = (nq / NUM_CORES) * hd;
            const int64_t mask_row_elems = Align(seq_len, 16);
            const int64_t query_chunk = (seq_len > 256)
                ? QWEN25VL_VISION_SDPA_QUERY_CHUNK : seq_len;
            for (int64_t q_begin = 0; q_begin < seq_len; q_begin += query_chunk) {
                const int64_t query_len = std::min(query_chunk, seq_len - q_begin);
                const uint32_t q_off = addr_offset("q").value
                    + static_cast<uint32_t>(q_begin * qd * DWIDTH);
                const uint32_t out_off = addr_offset("sdpa_out").value
                    + static_cast<uint32_t>(q_begin * qd * DWIDTH);
                const uint32_t query_mask_off = is_full ? 0
                    : mask_off + static_cast<uint32_t>(
                        q_begin * mask_row_elems * DWIDTH);
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    k_cache, v_cache,
                    mask_type, attn_scale,
                    q_off, out_off,
                    addr_offset("sdpa_tmp").value, query_mask_off,
                    query_len, nq, nq, hd,
                    seq_len, NUM_CORES, NUM_CORES);
            }
        }

        // Phase 4: O_proj (row-partition, with bias) + AllReduce + Residual.
        // input_norm := reduce(oproj) + residual1 = post-attn hidden.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, NUM_CORES, layer_addr(layer_idx, 0, "o_bias"),
            /*force_gemm=*/false, lw.o_ws);
        if (residual_is_bf16) {
            rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(
                addr(0, "oproj"), addr(0, "residual1"),
                addr(0, "input_norm"), seq_len, h);
        } else {
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual1"),
                addr(0, "input_norm"), seq_len, h, NUM_CORES, NUM_CORES);
        }
        emit_debug_phase(1, "input_norm");

        // Phase 5: RMSNorm2 (input_norm → oproj) + biased SwiGLU.
        if (residual_is_bf16) {
            rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                addr(0, "input_norm"), addr(0, "oproj"),
                layer_addr(layer_idx, 0, "norm2_w"), seq_len, h, eps_);
        } else {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "input_norm"), addr(0, "oproj"),
                layer_addr(layer_idx, 0, "norm2_w"), seq_len, h, eps_);
        }
        emit_debug_phase(2, "oproj");

        const int64_t elems = seq_len * local_inter;
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "oproj"), lw.gate_w, addr(0, "gate"),
            seq_len, is_, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "gate_bias"),
            /*force_gemm=*/false, lw.gate_ws);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "gate"), addr(0, "gate"), elems, ValuOpType::SILU);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "oproj"), lw.up_w, addr(0, "up"),
            seq_len, is_, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "up_bias"),
            /*force_gemm=*/false, lw.up_ws);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
            elems, ValuOpType::MUL, c10::Half(1.0));

        // Phase 6: down_proj (row-partition, with bias) + AllReduce + Residual.
        // residual1 := reduce(oproj) + input_norm = block output (SPM_RESIDENT).
        if (mixed_precision_
            && layer_idx == QWEN25VL_VISION_MIXED_BOUNDARY_LAYER) {
            rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(
                addr(0, "gate"), lw.down_w, addr(0, "oproj"),
                seq_len, h, is_, /*row=*/0, NUM_CORES,
                layer_addr(layer_idx, 0, "down_bias"));
            rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(
                addr(0, "oproj"), addr(0, "input_norm"),
                addr(0, "residual1"), seq_len, h);
        } else {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "gate"), lw.down_w, addr(0, "oproj"),
                seq_len, h, is_, 0, NUM_CORES,
                layer_addr(layer_idx, 0, "down_bias"),
                /*force_gemm=*/false, lw.down_ws);
            if (residual_is_bf16) {
                rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(
                    addr(0, "oproj"), addr(0, "input_norm"),
                    addr(0, "residual1"), seq_len, h);
            } else {
                rpu_launch_all_reduce_sum_residual_kernel(
                    addr(0, "oproj"), addr(0, "input_norm"),
                    addr(0, "residual1"), seq_len, h,
                    NUM_CORES, NUM_CORES);
            }
        }

        if (get_debug_export() && per_layer_debug_buf_.defined()) {
            c10::Half* layer_base =
                per_layer_debug_buf_.data_ptr<c10::Half>()
                + layer_idx * per_layer_debug_buf_.size(1)
                    * per_layer_debug_buf_.size(2) * h
                + chunk.offset * h;
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "residual1"), layer_base, chunk.len * h);
        }

        const bool z1_final = z1_bound_ &&
                              layer_idx == num_layers() - 1 &&
                              !ctx().output_to_spm;
        if (z1_final) {
            TORCH_CHECK(chunk.idx == 0 && chunk.offset == 0 &&
                            chunk.len == z1_prepared_num_patches_,
                        "Qwen25VLVisionModel Z1 requires one full-sequence final chunk");
        }

        // Ordinary final output lands in DDR. Z1 keeps the replicated
        // [256,1280] final hidden in residual1 for the immediately following
        // merger, eliminating the 655,360-byte bridge store.
        if (!ctx().output_to_spm && !z1_final) {
            emit_layer_output_dma(layer_idx, chunk, "residual1");
        }
    }

    // ========================================================================
    // merger_post_fn — runs once AFTER the 32-block encoder, INSIDE the same
    // GraphCache capture through post_fn. Computes the Qwen2.5-VL vision merger:
    //   merged = m2( gelu( m0( rmsnorm_lnq(residual1[seq,h]).reshape(seq/4, 5120) ) ) )
    // output [seq/4, 2048] (window-order; Python applies the reverse-gather).
    // Mirrors the ViT SwiGLU dataflow: m0 = col GEMM (like gate), m2 = row GEMM
    // (like down) + all-reduce.
    //
    // Normal single/packed vision runs one full-sequence pass from residual1.
    // The experimental layer-group path ended in DDR ping-pong, so it reloads
    // and merges one independent image chunk at a time.
    // ========================================================================
    void emit_merger_chunk(
        int64_t seq, int64_t dst_offset_elems, uint32_t input_addr,
        uint32_t output_addr = 0) {
        const int64_t h        = hidden_size();           // 1280
        const int64_t mh       = merge_hidden_;           // 5120
        const int64_t oh       = merger_out_hidden_;      // 2048
        const int64_t mrows    = seq / 4;                 // merged rows (SMS*SMS=4)
        const int64_t local_mh = mh / NUM_CORES;          // 640

        // 1. RMSNorm(ln_q) over h: input_addr[seq,h] (full per core) ->
        //    merger_normed.
        if (mixed_precision_) {
            rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                input_addr, addr(0, "merger_normed"),
                addr(0, "merger_ln_q_w"), seq, h, eps_);
        } else {
            rpu_launch_rmsnorm_spm_kernel(
                input_addr, addr(0, "merger_normed"),
                addr(0, "merger_ln_q_w"), seq, h, eps_);
        }

        // 2. m0 GEMM (col): merger_normed reinterpreted [mrows, mh] -> merger_mid
        //    col [mrows, mh] (each core holds [mrows, local_mh]) + col bias.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "merger_normed"), merger_m0_w_, addr(0, "merger_mid"),
            mrows, mh, mh, /*col*/1, NUM_CORES, addr(0, "merger_m0_bias"),
            /*force_gemm=*/false, at::Tensor{});

        // 3. GELU, per-core element count (col slice).
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "merger_mid"), addr(0, "merger_mid"),
            mrows * local_mh, ValuOpType::ADD,
            GeluMode::ERF, NUM_CORES);

        // 4. m2 GEMM (row): merger_mid [mrows, mh] (col slice = K-slice) ->
        //    merger_normed REUSED as PARTIAL [mrows, oh] per core + bias (core 0).
        //    merger_normed's RMSNorm content is dead after step 2 (m0 read it),
        //    so overwriting it here is safe.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "merger_mid"), merger_m2_w_, addr(0, "merger_normed"),
            mrows, oh, mh, /*row*/0, NUM_CORES, addr(0, "merger_m2_bias"),
            /*force_gemm=*/false, at::Tensor{});

        // 5. all-reduce(partial=merger_normed) + zero residual(merger_zero) ->
        //    merger_out [mrows, oh]. Three DISTINCT live slots (oproj/input_norm/
        //    residual1 offsets): partial=merger_normed, zero=merger_zero,
        //    out=merger_out. (no non-residual reduce variant; zero first.)
        rpu_launch_memset_spm_multicore(addr(0, "merger_zero"), mrows * oh);
        const uint32_t merger_out_addr = z1_bound_
            ? z1_source_addr_
            : (output_addr != 0 ? output_addr : addr(0, "merger_out"));
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "merger_normed"), addr(0, "merger_zero"), merger_out_addr,
            mrows, oh, NUM_CORES, NUM_CORES);

        if (!z1_bound_) {
            // 6. SPM(core 0) -> the fresh post_output_tensor_. Its DDR address
            //    drifts across REPLAYs, so only the base is mutable; each image
            //    has a fixed byte offset within that live allocation.
            rpu_launch_spm_copy_ddr_dma_mutable(
                merger_out_addr, &merger_out_dst_base_,
                dst_offset_elems * static_cast<int64_t>(DWIDTH), mrows * oh);
        }
    }

    void merger_post_fn() {
        const int64_t total_seq = current_num_patches_;
        const int64_t h = hidden_size();
        const int64_t oh = merger_out_hidden_;

        if (z1_bound_) {
            TORCH_CHECK(total_seq == QWEN25VL_Z1_NUM_PATCHES &&
                            image_batch_count_ == 1 &&
                            is_generic() &&
                            schedule_ == VisionSchedule::SISC &&
                            z1_source_addr_ != 0,
                        "Qwen25VLVisionModel Z1 merger requires the exact "
                        "single-image 256-patch source binding");
        } else {
            // The framework allocates this tensor fresh on every forward.
            merger_out_dst_base_ =
                ::rhino_lkn::RpuGetDevAddr(post_output_tensor().data_ptr());
            rpu_ddr_flush_force_sized(
                post_output_tensor().data_ptr<c10::Half>(),
                post_output_tensor().nbytes());
        }

        if (is_generic()) {
            emit_merger_chunk(
                total_seq, /*dst_offset_elems=*/0, addr(0, "residual1"));
            return;
        }

        const int64_t seq = total_seq / image_batch_count_;
        const int64_t merged_elems = (seq / 4) * oh;
        for (int64_t i = 0; i < image_batch_count_; ++i) {
            uint32_t merger_input_addr = 0;
            if (is_exact_mimc()) {
                // Exact MIMC leaves the final packed BF16 hidden in input_norm.
                // Its lifetime spans all three image-local merger passes, so no
                // final encoder store/reload bridge is needed.
                merger_input_addr = addr(0, "input_norm")
                    + static_cast<uint32_t>(i * seq * h * DWIDTH);
            } else {
                // The layer-group schedule ends in DDR ping-pong.
                merger_input_addr = addr(0, "residual1");
                rpu_launch_ddr_broadcast_spm_dma(
                    output_ptr() + i * seq * h, seq * h,
                    merger_input_addr, NUM_CORES);
            }
            // Exact MIMC gives each image a distinct output slot in its now-dead
            // encoder input slice. This prevents the next image's all-reduce
            // from racing the preceding image's asynchronous SPM->DDR DMA.
            const uint32_t merger_output_addr =
                is_exact_mimc() ? merger_input_addr : 0;
            emit_merger_chunk(
                seq, i * merged_elems, merger_input_addr,
                merger_output_addr);
        }
    }

private:
    bool is_generic() const {
        return implementation_ == VisionImplementation::Generic;
    }

    bool is_legacy_layer_group() const {
        return implementation_ == VisionImplementation::LegacyLayerGroup;
    }

    bool is_exact_mimc() const {
        return implementation_ == VisionImplementation::ExactByMha;
    }

    // Fused merger is active when the weights are registered and the env flag
    // is on. Normal packed vision is one pass; experimental layer grouping uses
    // the chunked path above.
    bool merger_active() const {
        return vision_fused_merger_enabled() && merger_out_hidden_ > 0;
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerBiasNorm> layer_bias_norm_;

    // Merger (post-encoder): ln_q RMSNorm(HID) -> reshape[seq/4, 4*HID] -> mlp.0 GEMM
    // -> exact-erf GELU -> mlp.2 GEMM. Weights DDR-resident (52MB+21MB), col-swizzled.
    at::Tensor merger_ln_q_w_;              // [HID] fp16
    at::Tensor merger_m0_w_, merger_m0_b_;  // [merge_hidden, merge_hidden], [merge_hidden]
    at::Tensor merger_m2_w_, merger_m2_b_;  // [OUT_HID, merge_hidden], [OUT_HID]
    int64_t merger_out_hidden_ = 0;         // OUT_HID (2048); 0 => merger not configured
    int64_t merge_hidden_ = 0;              // HID * sms * sms (5120)
    uint64_t merger_out_dst_base_ = 0;      // live RPU addr of post_output_tensor_ (mutable DMA)

    // 2D RoPE state
    at::Tensor freq_cos_;
    at::Tensor freq_sin_;
    at::Tensor position_idx_keepalive_;
    int64_t max_hw_ = 0;
    bool has_rope_ = false;
    int64_t current_num_patches_ = 0;
    int64_t image_batch_count_ = 1;   // >1 => batched multi-image (per-image SDPA isolation)
    int64_t configured_chunk_size_ = 0; // 0=auto; otherwise exact physical capacity
    VisionSchedule schedule_ = VisionSchedule::SISC;
    VisionImplementation implementation_ = VisionImplementation::Generic;
    bool mixed_precision_ = false;     // block-17 BF16 residual transition (RTC only)
    at::Tensor per_layer_debug_buf_;   // [layers, 1, seq, hidden], debug-only
    at::Tensor phase_debug_buf_;       // [3, 1, seq, hidden], layer-17 debug-only
    // Model config
    double eps_ = 1e-6;
    int64_t orig_head_dim_ = 0;

    // Window attention: per-layer dense block-diagonal MASK_2D. Layers in
    // fullatt_block_indexes_ stay MASK_NONE; the rest use prepared_window_mask_.
    std::vector<int64_t> fullatt_block_indexes_;
    bool window_mask_present_ = false;
    PreparedMask prepared_window_mask_;
    bool per_window_sdpa_enabled_ = false;
    std::vector<int64_t> cu_window_;
    // Cache the prepared window mask across forwards: the window_mask is
    // frame-invariant for a fixed camera (adapter memoizes it in _group_layout),
    // so re-running sdpa_prepare_mask (a [ctx,ctx] CPU→RPU upload + stable-slot
    // copy) every forward is redundant. Skip when the caller's mask ptr + shape
    // are unchanged — the [256,256] vision slot has no other same-shape user, so
    // the stable slot stays valid across frames.
    const void* last_window_mask_ptr_ = nullptr;
    int64_t last_mask_seq_ = -1;

    // Physical Z1 canary state. The coordinator owns the lease and route; the
    // producer retains only the checked identity and resolved replicated source
    // address. No SPM address crosses the C++ boundary.
    bool z1_adopted_ = false;
    bool z1_bound_ = false;
    int64_t z1_prepared_num_patches_ = 0;
    uint32_t z1_source_addr_ = 0;
    uint64_t z1_epoch_ = 0;
    uint64_t z1_plan_hash_ = 0;
    bool z1_inputs_primed_ = false;
    at::Tensor z1_primed_window_mask_ref_;
    const void* z1_primed_window_mask_ptr_ = nullptr;
};

}  // namespace v3

using Qwen25VLVisionRegistry = ModelHandleRegistry<v3::Qwen25VLVisionModel>;

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers
// =============================================================================

int64_t rpu_qwen25vl_vision_create() {
    return Qwen25VLVisionRegistry::create();
}

void rpu_qwen25vl_vision_destroy(int64_t handle) {
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision_destroy")
        ->check_z1_destroy_allowed();
    Qwen25VLVisionRegistry::destroy(handle, "rpu_qwen25vl_vision_destroy");
}

void rpu_qwen25vl_vision_set_per_window_sdpa(int64_t handle, bool enabled) {
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->set_per_window_sdpa(enabled);
}

void rpu_qwen25vl_vision_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_w_list, at::TensorList up_w_list, at::TensorList down_w_list,
    at::TensorList norm1_w_list, at::TensorList norm2_w_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList gate_b_list, at::TensorList up_b_list, at::TensorList down_b_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, at::IntArrayRef fullatt_block_indexes)
{
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_w_list, up_w_list, down_w_list,
        norm1_w_list, norm2_w_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        gate_b_list, up_b_list, down_b_list,
        num_heads, head_dim, hidden_size, intermediate_size, eps,
        fullatt_block_indexes);
}

void rpu_qwen25vl_vision_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_w_list, at::TensorList up_w_list, at::TensorList down_w_list,
    at::TensorList norm1_w_list, at::TensorList norm2_w_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList gate_b_list, at::TensorList up_b_list, at::TensorList down_b_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, at::IntArrayRef fullatt_block_indexes,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list)
{
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_w_list, up_w_list, down_w_list,
        norm1_w_list, norm2_w_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        gate_b_list, up_b_list, down_b_list,
        num_heads, head_dim, hidden_size, intermediate_size, eps,
        fullatt_block_indexes,
        q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list);
}

void rpu_qwen25vl_vision_set_merger_weights(
    int64_t handle,
    const at::Tensor& ln_q_w, const at::Tensor& m0_w, const at::Tensor& m0_b,
    const at::Tensor& m2_w, const at::Tensor& m2_b, int64_t out_hidden)
{
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->set_merger_weights(ln_q_w, m0_w, m0_b, m2_w, m2_b, out_hidden);
}

void rpu_qwen25vl_vision_set_rope(
    int64_t handle, const at::Tensor& freq_cos, const at::Tensor& freq_sin)
{
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->set_rope_tables(freq_cos, freq_sin);
}

void rpu_qwen25vl_vision_set_chunk_size(
    int64_t handle, int64_t chunk_size)
{
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->set_configured_chunk_size(chunk_size);
}

at::Tensor rpu_qwen25vl_vision_position_idx_keepalive(int64_t handle) {
    return Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->position_idx_keepalive();
}

at::Tensor rpu_qwen25vl_vision_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t num_patches,
    std::optional<at::Tensor> window_mask,
    int64_t image_batch_count,
    std::optional<at::Tensor> cu_window_seqlens)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->forward(input, k_caches, v_caches, num_patches, window_mask,
                  image_batch_count, cu_window_seqlens);
}

namespace v3::wall_oss_z1_internal {

SpmPipelineComponentLayout prepare_vision(int64_t handle,
                                          int64_t num_patches) {
    return Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_prepare_vision")
        ->prepare_z1_layout(num_patches);
}

void prime_vision_inputs(int64_t handle,
                         const at::Tensor& window_mask,
                         int64_t num_patches) {
    Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_prime_vision_inputs")
        ->prime_z1_inputs(window_mask, num_patches);
}

void rollback_vision_inputs(int64_t handle) {
    Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_rollback_vision_inputs")
        ->rollback_z1_inputs();
}

void unprepare_vision(int64_t handle) {
    Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_unprepare_vision")
        ->unprepare_z1_layout();
}

SpmDense2DSpec vision_source_spec(int64_t handle,
                                  int64_t num_patches) {
    return Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_vision_source_spec")
        ->z1_source_spec(num_patches);
}

void adopt_vision(int64_t handle,
                  const SpmPipelineLease& lease,
                  const SpmTensorView& scratch) {
    Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_adopt_vision")
        ->adopt_z1_layout(lease, scratch);
}

void bind_vision_source(int64_t handle,
                        const SpmPipelineLease& lease,
                        const SpmPortView& source) {
    Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_bind_vision_source")
        ->bind_z1_source(lease, source);
}

void validate_vision(int64_t handle,
                     const SpmPipelineLease& lease) {
    Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_validate_vision")
        ->validate_z1_layout(lease);
}

void clear_vision(int64_t handle,
                  uint64_t epoch,
                  uint64_t plan_hash) {
    Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_clear_vision")
        ->clear_z1_layout(epoch, plan_hash);
}

void forward_vision_z1(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches,
    at::TensorList v_caches,
    int64_t num_patches,
    std::optional<at::Tensor> window_mask,
    uint64_t epoch,
    uint64_t plan_hash) {
    std::vector<at::Tensor> k_cache_vec(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> v_cache_vec(v_caches.begin(), v_caches.end());
    Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_forward_vision")
        ->forward_z1(input, k_cache_vec, v_cache_vec, num_patches,
                     std::move(window_mask), epoch, plan_hash);
}

void check_vision_destroy_allowed(int64_t handle) {
    Qwen25VLVisionRegistry::get(handle, "wall_oss_z1_check_vision_destroy")
        ->check_z1_destroy_allowed();
}

}  // namespace v3::wall_oss_z1_internal
