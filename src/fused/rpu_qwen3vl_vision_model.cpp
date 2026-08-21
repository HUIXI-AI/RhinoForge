// rpu_qwen3vl_vision_model.cpp — Qwen3-VL Vision Encoder
//
// 24-block ViT encoder for Qwen3-VL (2B/4B variants, depth=24, hidden=1024,
// intermediate=4096, num_heads=16, head_dim=64). Architecture is identical to
// SigLIP-So400m EXCEPT:
//   - 2D RoPE applied to Q/K between fused-QKV+bias and bidir SDPA.
//   - No post-LN + projector (merger moved to Python after C++ forward
//     because the merger requires `spatial_merge_size² × hidden` reshape, see
//     so it remains a single-layer SPM op chain outside this graph).
//   - DeepStack snapshots at layers {5, 11, 17} → DMA to Python-side mergers
//     Reuses spm_copy_ddr_dma the same way SigLIP's post-LN-to-DDR
//     transfer works (see siglip's `temp_ddr_slots_` pattern).
//
// This class does not inherit SigLIPModel: Conv3d, 2D RoPE, fused QKV, and the
// merger reshape give the vision tower a distinct execution contract.
//
// Patch embedding + pos_emb interpolate are performed OUTSIDE the fused graph:
// Python adapter runs Conv3d→Conv2d folded GEMM via existing rpu_conv2d /
// rpu_linear, then adds `fast_pos_embed_interpolate` output via rpu_add. The
// fused subsystem expects pre-embedded input `[num_patches, hidden]`.
//
// 2D RoPE uses the `rope_2d_ddr` kernel; FreqCos /
// FreqSin are static tables `[max_hw, head_dim/4]` set once via set_rope.
// position_idx is a `[max_seq, 2] int16` keepalive using the absolute-index
// contract extended from text-domain M-RoPE to vision.

#include "fused_model_base.h"
#include "core/rpu_lingbot2_ring_collective.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"

#include <c10/util/ScopeExit.h>
#include "rpu_spm_allocator.h"
#include "rpu_bounded_shape_map.h"
#include "rpu_spm_pipeline.h"
#include "graph/graph_runtime.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

// Position-table keepalive cap. 4096 patches covers up to 64×64 grid (typical
// single-image is 28×28 = 784; pre-merger grid for 4096 is overkill but cheap
// at 4096 × 2 × 2 = 16 KB DDR. Aligned to allow 256B-boundary writes).
constexpr int64_t QWEN3VL_VISION_MAX_KEEPALIVE_SEQ = 4096;

constexpr int64_t QWEN3VL_VISION_DEFAULT_CHUNK_SIZE = 144;

// Qwen3-VL pooler-only physical handoff profile. Keep the envelope exact:
// one 2B-profile image, one full 256-patch encoder chunk, and a replicated
// [64, 2048] FP16 pooler result written directly by the merger all-reduce.
constexpr int64_t QWEN3VL_POOLER_Z1_NUM_PATCHES = 256;
constexpr int64_t QWEN3VL_POOLER_Z1_VISION_HIDDEN = 1024;
constexpr int64_t QWEN3VL_POOLER_Z1_MERGE_HIDDEN = 4096;
constexpr int64_t QWEN3VL_POOLER_Z1_MERGED_ROWS = 64;
constexpr int64_t QWEN3VL_POOLER_Z1_TEXT_HIDDEN = 2048;
constexpr int64_t QWEN3VL_POOLER_Z1_DEEPSTACK1_TAP = 5;
constexpr std::array<int64_t, 3> QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS = {
    QWEN3VL_POOLER_Z1_DEEPSTACK1_TAP, 11, 17};
constexpr size_t QWEN3VL_POOLER_Z1_SCRATCH_BYTES = 1966080;
constexpr int64_t QWEN3VL_MULTIVIEW_TEXT_HIDDEN = 2560;
// Exact closed-lifetime greedy first-fit extent.  The phase-5 live-size lower
// bound is 1,835,008 bytes, but q_comp/sdpa_out leave 131,072 bytes of placement
// fragmentation before oproj/fc1.
constexpr size_t QWEN3VL_MULTIVIEW_VISION_SCRATCH_BYTES = 1966080;
constexpr uint64_t QWEN3VL_MULTIVIEW_FP16_PROFILE_VERSION =
    UINT64_C(0x51564d564e323536);  // QVMVN256
constexpr uint64_t QWEN3VL_MULTIVIEW_W8A16_PROFILE_VERSION =
    UINT64_C(0x51564d5657384131);  // QVMVW8A1
constexpr uint64_t QWEN3VL_MULTIVIEW_W8A16_TWOSTAGE_PROFILE_VERSION =
    UINT64_C(0x51564d5657385432);  // QVMVW8T2
constexpr uint64_t QWEN3VL_MULTIVIEW_YIELD_POLICY =
    UINT64_C(0x51564d56594c4401);  // QVMVYLD1
constexpr uint64_t QWEN3VL_MULTIVIEW_OCCURRENCE_POLICY =
    UINT64_C(0x51564d5633494e01);  // QVMV3IN1
// RPU_QWEN3VL_VISION_ROPE_SPM — structural default OFF; model facades including
// GR00T and ordinary LingBot2 opt in through scoped profiles. Runs the 2D-RoPE
// from SPM-resident cos/sin tables (broadcast once in emit_preload_weights)
// instead of re-streaming the DDR tables per layer. Bit-identical math. Mirrors the qwen25vl
// RPU_WALL_OSS_VISION_ROPE_SPM path.
static bool read_vision_rope_spm_enabled() {
    const char* e = std::getenv("RPU_QWEN3VL_VISION_ROPE_SPM");
    return e && (e[0] == '1' || e[0] == 't' || e[0] == 'T');
}

// RPU_QWEN3VL_VISION_FUSED_MERGER — default OFF. When on (and merger weights are
// registered) the post-encoder patch merger (LayerNorm -> m0 GEMM -> GELU -> m2 GEMM
// -> all-reduce) runs INSIDE the qwen3vl_vision_forward graph via a post_fn (single
// image per forward). gr00t opts in; plain qwen3_vl stays byte-identical.
static bool read_vision_fused_merger_enabled() {
    const char* e = std::getenv("RPU_QWEN3VL_VISION_FUSED_MERGER");
    return e && std::string(e) != "0" && std::string(e) != "false";
}

// W8A16 per-projection scale validation. Same four checks as the already-strong
// sites in rpu_siglip_model.cpp and rpu_pi05_denoise_step_model.cpp: list
// length, int8 weight dtype, 1D contiguous fp16 RPU scale, and
// scale.numel() == output channels. Checking only the list length is not enough —
// a wrong-length or wrong-dtype scale reaches PARALLEL_LINEAR_W8A16 as garbage
// per-channel multipliers, i.e. a silent numeric corruption rather than a crash.
// The `else` leg is the reverse hole: an int8 weight with no scale list is read
// back as fp16 by the kernel.
static void check_qwen3vl_vision_w8a16_scale_list(int64_t n_layers,
                                                  const at::TensorList& weights,
                                                  const at::TensorList& scales,
                                                  const char* name) {
    const char* ctx = "qwen3vl_vision_set_weights";
    const bool quantized = !scales.empty();
    if (quantized) {
        TORCH_CHECK(static_cast<int64_t>(scales.size()) == n_layers,
                    ctx, ": ", name, "_scale.size()=", scales.size(),
                    " != num_layers=", n_layers);
    }
    for (int64_t i = 0; i < n_layers; ++i) {
        const at::Tensor& w = weights[i];
        if (quantized) {
            const at::Tensor& s = scales[i];
            TORCH_CHECK(w.scalar_type() == at::kChar,
                        ctx, ": W8A16 mode requires int8 ", name, "[", i,
                        "], got ", w.scalar_type());
            TORCH_CHECK(s.defined() && s.dim() == 1
                        && s.scalar_type() == at::kHalf
                        && s.device().type() == at::kPrivateUse1
                        && s.is_contiguous(),
                        ctx, ": ", name, "_scale[", i,
                        "] must be 1D contiguous fp16 RPU tensor");
            TORCH_CHECK(s.numel() == w.size(0),
                        ctx, ": ", name, "_scale[", i, "].numel()=", s.numel(),
                        " != output dim=", w.size(0));
        } else {
            TORCH_CHECK(w.scalar_type() != at::kChar,
                        ctx, ": int8 ", name, "[", i,
                        "] requires a non-empty scale list for this projection");
        }
    }
}

namespace v3 {

enum class MultiviewLinearProfile : uint8_t {
    DenseFp16 = 1,
    W8A16 = 2,
};

class Qwen3VLVisionModel : public FusedModelBase {
public:
    struct LayerWeights {
        // All [out, in] fp16 DDR, swizzled per partition (Python adapter handles).
        at::Tensor q_w, k_w, v_w, o_w;       // attention (col/col/col/row partition)
        at::Tensor fc1_w, fc2_w;              // MLP (col/row partition)
        // W8A16: per-output-channel fp16 scales for the 6 quantized GEMMs (undefined = fp16).
        at::Tensor q_ws, k_ws, v_ws, o_ws, fc1_ws, fc2_ws;
    };

    struct LayerBiasNorm {
        at::Tensor ln1_w, ln1_b, ln2_w, ln2_b;
        at::Tensor q_b, k_b, v_b, o_b;
        at::Tensor fc1_b, fc2_b;
    };

    Qwen3VLVisionModel() = default;

    // ========================================================================
    // set_weights — per-layer weight + bias storage.
    //
    // QKV bias is present (Qwen3VLVisionAttention sets `bias=True` for the
    // fused qkv Linear; the Python adapter pre-splits the fused [3*dim, dim]
    // weight into q/k/v [dim, dim] and the [3*dim] bias into q_b/k_b/v_b).
    // ========================================================================
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList fc1_w_list, at::TensorList fc2_w_list,
        at::TensorList ln1_w_list, at::TensorList ln1_b_list,
        at::TensorList ln2_w_list, at::TensorList ln2_b_list,
        at::TensorList q_b_list, at::TensorList k_b_list,
        at::TensorList v_b_list, at::TensorList o_b_list,
        at::TensorList fc1_b_list, at::TensorList fc2_b_list,
        int64_t num_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps, at::IntArrayRef deepstack_visual_indexes = {},
        at::TensorList q_scale_list = {}, at::TensorList k_scale_list = {},
        at::TensorList v_scale_list = {}, at::TensorList o_scale_list = {},
        at::TensorList fc1_scale_list = {}, at::TensorList fc2_scale_list = {})
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "qwen3vl_vision_set_weights: empty weight lists");
        TORCH_CHECK(num_heads > 0 && head_dim > 0 && hidden_size > 0
                    && intermediate_size > 0,
                    "qwen3vl_vision_set_weights: dim params must be positive");

        auto check_list = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "qwen3vl_vision_set_weights: ", n, ".size()=", l.size(),
                        " != num_layers=", N);
        };
        check_list(k_w_list,   "k_w_list");
        check_list(v_w_list,   "v_w_list");
        check_list(o_w_list,   "o_w_list");
        check_list(fc1_w_list, "fc1_w_list");
        check_list(fc2_w_list, "fc2_w_list");
        check_list(ln1_w_list, "ln1_w_list");
        check_list(ln1_b_list, "ln1_b_list");
        check_list(ln2_w_list, "ln2_w_list");
        check_list(ln2_b_list, "ln2_b_list");
        check_list(q_b_list,   "q_b_list");
        check_list(k_b_list,   "k_b_list");
        check_list(v_b_list,   "v_b_list");
        check_list(o_b_list,   "o_b_list");
        check_list(fc1_b_list, "fc1_b_list");
        check_list(fc2_b_list, "fc2_b_list");

        auto check_rank = [&](const at::TensorList& list, const char* name, int64_t r) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "qwen3vl_vision_set_weights: ", name, "[", i, "] undefined");
                TORCH_CHECK(list[i].dim() == r,
                            "qwen3vl_vision_set_weights: ", name, "[", i, "] must be ",
                            r, "D, got ", list[i].dim(), "D");
            }
        };
        check_rank(q_w_list,   "q_w_list",   2);
        check_rank(k_w_list,   "k_w_list",   2);
        check_rank(v_w_list,   "v_w_list",   2);
        check_rank(o_w_list,   "o_w_list",   2);
        check_rank(fc1_w_list, "fc1_w_list", 2);
        check_rank(fc2_w_list, "fc2_w_list", 2);
        check_rank(ln1_w_list, "ln1_w_list", 1);
        check_rank(ln1_b_list, "ln1_b_list", 1);
        check_rank(ln2_w_list, "ln2_w_list", 1);
        check_rank(ln2_b_list, "ln2_b_list", 1);
        check_rank(q_b_list,   "q_b_list",   1);
        check_rank(k_b_list,   "k_b_list",   1);
        check_rank(v_b_list,   "v_b_list",   1);
        check_rank(o_b_list,   "o_b_list",   1);
        check_rank(fc1_b_list, "fc1_b_list", 1);
        check_rank(fc2_b_list, "fc2_b_list", 1);

        TORCH_CHECK(hidden_size % num_heads == 0,
                    "qwen3vl_vision_set_weights: hidden_size must be divisible "
                    "by num_heads");
        const int64_t logical_head_dim = hidden_size / num_heads;
        const bool padded_32b =
            N == 27 && num_heads == 16 && hidden_size == 1152 &&
            logical_head_dim == 72 && head_dim == 80 &&
            intermediate_size == 4352;
        TORCH_CHECK(head_dim == logical_head_dim || padded_32b,
                    "qwen3vl_vision_set_weights: physical head_dim=", head_dim,
                    " must equal logical hidden_size/num_heads=", logical_head_dim,
                    " except for the exact controlled 32B 72->80 profile");
        TORCH_CHECK(head_dim % 16 == 0,
                    "qwen3vl_vision_set_weights: physical head_dim must be a "
                    "multiple of 16, got ", head_dim);

        const int64_t attention_width = num_heads * head_dim;
        auto check_matrix_shape = [&](const at::TensorList& list,
                                      const char* name,
                                      int64_t rows, int64_t cols) {
            for (int64_t i = 0; i < N; ++i) {
                TORCH_CHECK(list[i].size(0) == rows && list[i].size(1) == cols,
                            "qwen3vl_vision_set_weights: ", name, "[", i,
                            "] must be [", rows, ",", cols, "], got ",
                            list[i].sizes());
            }
        };
        auto check_vector_shape = [&](const at::TensorList& list,
                                      const char* name, int64_t length) {
            for (int64_t i = 0; i < N; ++i) {
                TORCH_CHECK(list[i].size(0) == length,
                            "qwen3vl_vision_set_weights: ", name, "[", i,
                            "] must be [", length, "], got ", list[i].sizes());
            }
        };
        check_matrix_shape(q_w_list, "q_w_list", attention_width, hidden_size);
        check_matrix_shape(k_w_list, "k_w_list", attention_width, hidden_size);
        check_matrix_shape(v_w_list, "v_w_list", attention_width, hidden_size);
        check_matrix_shape(o_w_list, "o_w_list", hidden_size, attention_width);
        check_matrix_shape(fc1_w_list, "fc1_w_list", intermediate_size, hidden_size);
        check_matrix_shape(fc2_w_list, "fc2_w_list", hidden_size, intermediate_size);
        check_vector_shape(ln1_w_list, "ln1_w_list", hidden_size);
        check_vector_shape(ln1_b_list, "ln1_b_list", hidden_size);
        check_vector_shape(ln2_w_list, "ln2_w_list", hidden_size);
        check_vector_shape(ln2_b_list, "ln2_b_list", hidden_size);
        check_vector_shape(q_b_list, "q_b_list", attention_width);
        check_vector_shape(k_b_list, "k_b_list", attention_width);
        check_vector_shape(v_b_list, "v_b_list", attention_width);
        check_vector_shape(o_b_list, "o_b_list", hidden_size);
        check_vector_shape(fc1_b_list, "fc1_b_list", intermediate_size);
        check_vector_shape(fc2_b_list, "fc2_b_list", hidden_size);

        // Vision encoder is MHA (no GQA): num_kv_heads == num_q_heads.
        set_model_params(num_heads, num_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;

        // Keep logical attention math separate from the physical RPU stride.
        // The controlled 32B tower pads each head 72->80 and its MLP
        // 4304->4352 at the Python conversion boundary; 2B/4B remain exact.
        orig_head_dim_ = logical_head_dim;

        // DeepStack snapshot indices. Empty list → no snapshots emitted.
        deepstack_visual_indexes_.clear();
        for (auto idx : deepstack_visual_indexes) {
            TORCH_CHECK(idx >= 0 && idx < N,
                        "qwen3vl_vision_set_weights: deepstack_visual_indexes entry ",
                        idx, " out of range [0,", N, ")");
            deepstack_visual_indexes_.push_back(static_cast<int64_t>(idx));
        }
        // Distinct check.
        {
            std::set<int64_t> seen(deepstack_visual_indexes_.begin(),
                                   deepstack_visual_indexes_.end());
            TORCH_CHECK(seen.size() == deepstack_visual_indexes_.size(),
                        "qwen3vl_vision_set_weights: deepstack_visual_indexes must be distinct");
        }
        if (padded_32b) {
            TORCH_CHECK(deepstack_visual_indexes_ ==
                            std::vector<int64_t>({8, 16, 24}),
                        "qwen3vl_vision_set_weights: the controlled 32B profile "
                        "requires DeepStack indexes [8,16,24]");
        }

        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                fc1_w_list[i], fc2_w_list[i],
            });
        }

        // W8A16: attach per-output-channel scales for the 6 quantized GEMMs (empty = fp16).
        check_qwen3vl_vision_w8a16_scale_list(N, q_w_list,   q_scale_list,   "q_w");
        check_qwen3vl_vision_w8a16_scale_list(N, k_w_list,   k_scale_list,   "k_w");
        check_qwen3vl_vision_w8a16_scale_list(N, v_w_list,   v_scale_list,   "v_w");
        check_qwen3vl_vision_w8a16_scale_list(N, o_w_list,   o_scale_list,   "o_w");
        check_qwen3vl_vision_w8a16_scale_list(N, fc1_w_list, fc1_scale_list, "fc1_w");
        check_qwen3vl_vision_w8a16_scale_list(N, fc2_w_list, fc2_scale_list, "fc2_w");
        // W8A16: attach per-output-channel scales for all 6 quantized GEMMs
        // (all empty = fp16).  Reject a partial inventory before indexing any
        // TensorList; the multiview profile validator checks tensor details.
        const bool any_scale =
            !q_scale_list.empty() || !k_scale_list.empty() ||
            !v_scale_list.empty() || !o_scale_list.empty() ||
            !fc1_scale_list.empty() || !fc2_scale_list.empty();
        TORCH_CHECK(!padded_32b || !any_scale,
                    "qwen3vl_vision_set_weights: the controlled 32B vision "
                    "tower must remain FP16; W8A16 applies only to text");
        if (any_scale) {
            check_list(q_scale_list, "q_scale_list");
            check_list(k_scale_list, "k_scale_list");
            check_list(v_scale_list, "v_scale_list");
            check_list(o_scale_list, "o_scale_list");
            check_list(fc1_scale_list, "fc1_scale_list");
            check_list(fc2_scale_list, "fc2_scale_list");
            for (int64_t i = 0; i < N; i++) {
                layer_weights_[i].q_ws   = q_scale_list[i];
                layer_weights_[i].k_ws   = k_scale_list[i];
                layer_weights_[i].v_ws   = v_scale_list[i];
                layer_weights_[i].o_ws   = o_scale_list[i];
                layer_weights_[i].fc1_ws = fc1_scale_list[i];
                layer_weights_[i].fc2_ws = fc2_scale_list[i];
            }
        }

        layer_bias_norm_.clear();
        layer_bias_norm_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_bias_norm_.push_back({
                ln1_w_list[i], ln1_b_list[i],
                ln2_w_list[i], ln2_b_list[i],
                q_b_list[i], k_b_list[i], v_b_list[i], o_b_list[i],
                fc1_b_list[i], fc2_b_list[i],
            });
        }

        invalidate_model_state();
    }

    // ========================================================================
    // set_rope_tables — register FreqCos / FreqSin static tables.
    //
    // Tables shape: [max_hw, head_dim/4] FP16 DDR. Same table is indexed by
    // BOTH row_idx and col_idx (per Qwen3VLVisionRotaryEmbedding which uses
    // head_dim/2 dim → freqs (max_hw, head_dim/4)). 256B alignment required
    // by the rope_2d kernel — the Python adapter is responsible for padded
    // allocation if the natural tensor size doesn't align.
    //
    // Also pre-allocates position_idx keepalive `[MAX_KEEPALIVE_SEQ, 2] int16`.
    // ========================================================================
    void set_rope_tables(const at::Tensor& freq_cos, const at::Tensor& freq_sin)
    {
        TORCH_CHECK(freq_cos.defined() && freq_sin.defined(),
                    "qwen3vl_vision_set_rope: freq_cos / freq_sin must be defined");
        TORCH_CHECK(freq_cos.dim() == 2 && freq_sin.dim() == 2,
                    "qwen3vl_vision_set_rope: freq_cos/sin must be 2D, got ",
                    freq_cos.dim(), "/", freq_sin.dim(), "D");
        TORCH_CHECK(freq_cos.sizes() == freq_sin.sizes(),
                    "qwen3vl_vision_set_rope: freq_cos / freq_sin shape mismatch");
        TORCH_CHECK(freq_cos.scalar_type() == at::kHalf && freq_sin.scalar_type() == at::kHalf,
                    "qwen3vl_vision_set_rope: freq_cos / freq_sin must be fp16");
        TORCH_CHECK(freq_cos.device().type() == at::kPrivateUse1
                    && freq_sin.device().type() == at::kPrivateUse1,
                    "qwen3vl_vision_set_rope: freq_cos / freq_sin must be on RPU device");
        TORCH_CHECK(freq_cos.is_contiguous() && freq_sin.is_contiguous(),
                    "qwen3vl_vision_set_rope: freq_cos / freq_sin must be contiguous");
        TORCH_CHECK(orig_head_dim_ > 0 && orig_head_dim_ % 4 == 0,
                    "qwen3vl_vision_set_rope: logical head dim must be a "
                    "positive multiple of 4");
        TORCH_CHECK(freq_cos.size(1) == orig_head_dim_ / 4,
                    "qwen3vl_vision_set_rope: table columns must use logical "
                    "head_dim/4=", orig_head_dim_ / 4, ", got ",
                    freq_cos.size(1));

        freq_cos_ = freq_cos;
        freq_sin_ = freq_sin;
        max_hw_ = freq_cos.size(0);
        rope_spm_enabled_ = read_vision_rope_spm_enabled();

        // Lazy-alloc on first call; reuse on subsequent calls (sizes never change).
        if (!position_idx_keepalive_.defined()) {
            position_idx_keepalive_ = at::empty(
                {QWEN3VL_VISION_MAX_KEEPALIVE_SEQ, 2},
                at::TensorOptions().dtype(at::kShort).device(at::kPrivateUse1));
            position_idx_keepalive_.zero_();
        }

        has_rope_ = true;
        invalidate_model_state();
    }

    // ========================================================================
    // set_merger_weights — register the post-encoder patch-merger weights for
    // the in-graph fold. LayerNorm gamma/beta [HID] + m0
    // [merge_hidden, merge_hidden] (col-swizzled) + m2 [OUT_HID, merge_hidden]
    // (row-swizzled), all fp16 DDR-resident (swizzled by the Python adapter).
    // ========================================================================
    void set_merger_weights(
        const at::Tensor& ln_q_w, const at::Tensor& ln_q_b,
        const at::Tensor& m0_w, const at::Tensor& m0_b,
        const at::Tensor& m2_w, const at::Tensor& m2_b,
        int64_t out_hidden, double ln_eps)
    {
        TORCH_CHECK(ln_q_w.dim() == 1 && ln_q_w.size(0) == hidden_size(),
                    "merger ln_q_w must be [HID]");
        TORCH_CHECK(ln_q_b.dim() == 1 && ln_q_b.size(0) == hidden_size(),
                    "merger ln_q_b must be [HID]");
        TORCH_CHECK(m0_w.dim() == 2 && m0_w.size(0) == m0_w.size(1) && (m0_w.size(0) % 128) == 0,
                    "merger m0_w must be [merge_hidden, merge_hidden], merge_hidden%128==0");
        TORCH_CHECK(m2_w.dim() == 2 && m2_w.size(0) == out_hidden && m2_w.size(1) == m0_w.size(0),
                    "merger m2_w must be [out_hidden, merge_hidden]");
        TORCH_CHECK(m0_b.dim() == 1 && m0_b.size(0) == m0_w.size(0), "merger m0_b must be [merge_hidden]");
        TORCH_CHECK(m2_b.dim() == 1 && m2_b.size(0) == out_hidden, "merger m2_b must be [out_hidden]");
        merger_ln_q_w_ = ln_q_w.contiguous();
        merger_ln_q_b_ = ln_q_b.contiguous();
        merger_m0_w_   = m0_w.contiguous();   merger_m0_b_ = m0_b.contiguous();
        merger_m2_w_   = m2_w.contiguous();   merger_m2_b_ = m2_b.contiguous();
        merge_hidden_      = m0_w.size(0);
        merger_out_hidden_ = out_hidden;
        merger_ln_eps_     = ln_eps;
        fused_merger_enabled_ = read_vision_fused_merger_enabled();
        invalidate_model_state();
    }

    // Register the N deepstack mergers' weights (one per deepstack_visual_index). Same
    // shapes as the patch merger; m0 col-swizzled, m2 row-swizzled, ln_q raw [HID]. Folded in
    // merger_post_fn (each reads its layer's snapshot, writes a stable merged slot).
    void set_deepstack_merger_weights(
        at::TensorList ln_q_w, at::TensorList ln_q_b,
        at::TensorList m0_w, at::TensorList m0_b,
        at::TensorList m2_w, at::TensorList m2_b)
    {
        TORCH_CHECK(merger_out_hidden_ > 0,
                    "set_deepstack_merger_weights: call set_merger_weights first");
        const int64_t n = static_cast<int64_t>(m0_w.size());
        TORCH_CHECK(n == static_cast<int64_t>(deepstack_visual_indexes_.size()),
                    "set_deepstack_merger_weights: count ", n, " != n_deepstack ",
                    deepstack_visual_indexes_.size());
        ds_merger_ln_q_w_.clear(); ds_merger_ln_q_b_.clear();
        ds_merger_m0_w_.clear();   ds_merger_m0_b_.clear();
        ds_merger_m2_w_.clear();   ds_merger_m2_b_.clear();
        for (int64_t k = 0; k < n; ++k) {
            TORCH_CHECK(m0_w[k].dim() == 2 && m0_w[k].size(0) == merge_hidden_
                        && m0_w[k].size(1) == merge_hidden_,
                        "set_deepstack_merger_weights: m0_w[", k, "] must be [merge_hidden, merge_hidden]");
            TORCH_CHECK(m2_w[k].dim() == 2 && m2_w[k].size(0) == merger_out_hidden_
                        && m2_w[k].size(1) == merge_hidden_,
                        "set_deepstack_merger_weights: m2_w[", k, "] must be [out_hidden, merge_hidden]");
            ds_merger_ln_q_w_.push_back(ln_q_w[k].contiguous());
            ds_merger_ln_q_b_.push_back(ln_q_b[k].contiguous());
            ds_merger_m0_w_.push_back(m0_w[k].contiguous());
            ds_merger_m0_b_.push_back(m0_b[k].contiguous());
            ds_merger_m2_w_.push_back(m2_w[k].contiguous());
            ds_merger_m2_b_.push_back(m2_b[k].contiguous());
        }
        invalidate_model_state();
    }

    int64_t max_keepalive_seq() const { return QWEN3VL_VISION_MAX_KEEPALIVE_SEQ; }

    void set_configured_chunk_size(int64_t chunk_size) {
        TORCH_CHECK(chunk_size == 0
                        || (chunk_size >= 16 && chunk_size % 16 == 0),
                    "Qwen3-VL vision chunk size must be 0 (auto) or a "
                    "positive multiple of 16, got ", chunk_size);
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "Qwen3-VL vision chunk size must be configured before "
                    "the first forward");
        configured_chunk_size_ = chunk_size;
        invalidate_model_state();
    }

    int64_t resolved_chunk_size() const {
        return get_last_resolved_chunk_size();
    }

    // Expose the keepalive base for adapter-side copy_in.
    at::Tensor& position_idx_keepalive() { return position_idx_keepalive_; }

    // ========================================================================
    // forward — drive the 24-block encoder.
    //
    // Input  : [num_patches, hidden_size] fp16 RPU (Python adapter must have
    //          already added pos_embed and folded patch_embed).
    // Output : [num_patches, hidden_size] fp16 RPU (final hidden state).
    //          DeepStack snapshots are accessible via `pop_deepstack_snapshots()`
    //          after forward returns — Python adapter pulls them out for the
    //          merger pass.
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches_in,
        int64_t image_batch_count)
    {
        TORCH_CHECK(!multiview_dry_prepared_ &&
                        !multiview_composite_prepared_ &&
                        !multiview_composite_dispatch_ &&
                        z1_prepared_num_patches_ == 0 && !z1_adopted_ &&
                        !z1_bound_ && !z1_dispatch_,
                    "Qwen3VLVisionModel::forward: pooler Z1 or multiview dry "
                    "layout is prepared or active; use the coordinator-owned "
                    "Z1 dispatch or cancel the dry probe");
        return forward_impl(input, k_caches, v_caches, num_patches_in,
                            image_batch_count, /*z1_dispatch=*/false);
    }

    at::Tensor forward_impl(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches_in,
        int64_t image_batch_count,
        bool z1_dispatch)
    {
        const bool pipeline_dispatch = pipeline_dispatch_active();
        TORCH_CHECK(z1_dispatch == z1_dispatch_ &&
                        !(z1_dispatch_ &&
                          multiview_composite_dispatch_) &&
                        (!z1_dispatch ||
                         (z1_adopted_ && z1_bound_ &&
                          retained_deepstack_bindings_complete())) &&
                        (!multiview_composite_dispatch_ ||
                         multiview_composite_prepared_),
                    "Qwen3VLVisionModel::forward_impl: pipeline "
                    "dispatch/binding mismatch");
        TORCH_CHECK(num_layers() > 0,
                    "Qwen3VLVisionModel::forward called before set_weights");
        TORCH_CHECK(has_rope_,
                    "Qwen3VLVisionModel::forward called before set_rope_tables");
        TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                    "Qwen3VLVisionModel::forward: input must be on RPU device");
        TORCH_CHECK(input.is_contiguous(),
                    "Qwen3VLVisionModel::forward: input must be contiguous");
        TORCH_CHECK(input.dim() == 3,
                    "Qwen3VLVisionModel::forward: input must be 3D [1,N,H] (adapter "
                    "unsqueezes batch axis), got ", input.dim(), "D");
        TORCH_CHECK(num_patches_in > 0 && num_patches_in <= QWEN3VL_VISION_MAX_KEEPALIVE_SEQ,
                    "Qwen3VLVisionModel::forward: num_patches=", num_patches_in,
                    " must be in (0, MAX_KEEPALIVE_SEQ=", QWEN3VL_VISION_MAX_KEEPALIVE_SEQ, "]");
        TORCH_CHECK(input.size(0) == 1,
                    "Qwen3VLVisionModel::forward: batch must be 1, got ", input.size(0));
        TORCH_CHECK(input.size(1) == num_patches_in,
                    "Qwen3VLVisionModel::forward: input.size(1)=", input.size(1),
                    " must match num_patches=", num_patches_in);
        if (pipeline_dispatch) {
            TORCH_CHECK(RpuKernelGraph::has_active(),
                        "Qwen3VLVisionModel pipeline forward requires one "
                        "active outer Graph");
            if (z1_dispatch) {
                validate_pooler_z1_contract();
            } else {
                validate_multiview_composite_profile();
            }
            TORCH_CHECK(num_patches_in == QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                            image_batch_count == 1 &&
                            input.size(2) == QWEN3VL_POOLER_Z1_VISION_HIDDEN &&
                            input.scalar_type() == at::kHalf,
                        "Qwen3VLVisionModel pipeline forward requires contiguous "
                        "FP16 [1,256,1024] input and image_batch_count=1");
            if (z1_dispatch) {
                TORCH_CHECK(z1_inputs_primed_ &&
                                z1_primed_position_ptr_ ==
                                    position_idx_keepalive_.data_ptr<int16_t>(),
                            "Qwen3VLVisionModel pooler Z1 position input was "
                            "not primed outside Graph or its address drifted");
            } else {
                TORCH_CHECK(
                    multiview_primed_position_ptr_ ==
                            position_idx_keepalive_.data_ptr<int16_t>() &&
                        multiview_primed_position_owner_ ==
                            position_idx_keepalive_.unsafeGetTensorImpl(),
                    "Qwen3VLVisionModel multiview position input was not "
                    "primed outside Graph or its owner drifted");
            }
        }

        TORCH_CHECK(!k_caches.empty(),
                    "Qwen3VLVisionModel::forward: k_caches must not be empty");
        const at::Tensor& k0 = k_caches.front();
        TORCH_CHECK(k0.dim() == 7,
                    "Qwen3VLVisionModel::forward: k_cache must use 7-D RPU layout, got ",
                    k0.dim(), "D");
        const int64_t cache_max_seq = k0.size(1) * k0.size(5);
        TORCH_CHECK(num_patches_in <= cache_max_seq,
                    "Qwen3VLVisionModel::forward: num_patches=", num_patches_in,
                    " exceeds k_cache max_seq_len=", cache_max_seq);

        // Multi-image BATCH (minibatch SDPA): num_patches_in packs `image_batch_count`
        // equal-size images of `num_patches_in / image_batch_count` patches each. =1 is
        // the legacy single-image path (byte-identical, unified_v2 SDPA).
        TORCH_CHECK(image_batch_count >= 1,
                    "Qwen3VLVisionModel::forward: image_batch_count must be >= 1, got ",
                    image_batch_count);
        TORCH_CHECK(num_patches_in % image_batch_count == 0,
                    "Qwen3VLVisionModel::forward: num_patches=", num_patches_in,
                    " must be divisible by image_batch_count=", image_batch_count);
        image_batch_count_ = image_batch_count;

        // Cache the num_patches for build_layer_subgraph to consume (Q/K rope
        // launcher needs the actual token count, which is also the chunk len).
        current_num_patches_ = num_patches_in;

        // Pre-allocate DeepStack snapshot DDR slots for this shape BEFORE
        // run_all_layers — `at::empty` inside build_layer_subgraph competes
        // with graph queue/instruction buffers which have already
        // consumed most of the high-mem heap (304 MB BufferPool + 64 MB
        // instr buffer + 8 MB cmd buffer). Pre-allocating here keeps the
        // allocations on the PyTorch caching allocator's slow path before
        // any batch resources lock the high-mem heap.
        if (!pipeline_dispatch) {
            for (int64_t layer_idx : deepstack_visual_indexes_) {
                auto key = std::make_pair(layer_idx, num_patches_in);
                auto& slot = deepstack_ddr_slots_.touch(key);
                if (!slot.defined()) {
                    slot = at::empty({num_patches_in, hidden_size()},
                        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
                }
                // Stable merged-output slot [num_patches/4, oh] for the in-graph deepstack
                // merger (fixed DMA → deterministic). Pop via pop_deepstack_merged().
                if (deepstack_merger_active()) {
                    auto& mslot = deepstack_merged_ddr_slots_.touch(key);
                    if (!mslot.defined()) {
                        mslot = at::empty({num_patches_in / 4, merger_out_hidden_},
                            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
                    }
                }
            }
            // Patch merger: stable pooler_merged slot [num_patches/4, oh] (fixed DMA → deterministic,
            // replaces the mutable post_output_tensor). Pop via pop_pooler_merged().
            if (merger_active()) {
                auto& ps = pooler_merged_ddr_slots_.touch(num_patches_in);
                if (!ps.defined()) {
                    ps = at::empty({num_patches_in / 4, merger_out_hidden_},
                        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
                }
            }
        }

        // Flush the position_idx keepalive — adapter just wrote per-forward
        // (row, col) entries into it via Python `.copy_()`. Without an explicit
        // flush, the CPU write may sit in the caching allocator's write buffer
        // and the RPU DMA reads stale (zero or prior-forward) data, silently
        // collapsing 2D RoPE to identity. Same pattern as Qwen3 M-RoPE in
        // CausalDecoderModel::forward (Phase 1 keepalive flush).
        if (!pipeline_dispatch) {
            rpu_ddr_flush_force(position_idx_keepalive_.data_ptr<int16_t>());
        }

        // Fixed-DMA Q staging must keep a stable address for every captured shape.
        // A grow-only tensor would invalidate an older shape's captured address after
        // reallocating for a larger image, so retain one slot per (patches, width).
        const int64_t local_q_dim = (num_q_heads() / NUM_CORES) * head_dim();
        const std::pair<int64_t, int64_t> q_key{num_patches_in, local_q_dim};
        if (!pipeline_dispatch &&
            q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
            TORCH_CHECK(q_ddr_slots_.size() < 128,
                        "Qwen3VLVisionModel: q_ddr_slots_ exceeded 128 distinct "
                        "(num_patches,width) shapes on one handle");
            q_ddr_slots_.emplace(q_key, at::empty(
                {NUM_CORES, num_patches_in, local_q_dim},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1)));
        }
        TORCH_CHECK(q_ddr_slots_.count(q_key) == 1,
                    "Qwen3VLVisionModel pooler Z1 requires its fixed Q staging "
                    "slot to be created during Graph-external prepare");

        // Packed-image minibatch SDPA and the fused merger both consume the full
        // sequence from SPM, so they require one chunk. FusedModelBase aligns a
        // chunk override down to 16; round up here so a non-aligned sequence is
        // not accidentally split into a trailing partial chunk.
        const int64_t full_sequence_chunk = ((num_patches_in + 15) / 16) * 16;
        set_chunk_size_override(
            configured_chunk_size_ > 0
                ? configured_chunk_size_
                : ((merger_active() || image_batch_count_ > 1)
                       ? full_sequence_chunk
                       : QWEN3VL_VISION_DEFAULT_CHUNK_SIZE));

        at::Tensor result = run_all_layers(
            input, k_caches, v_caches,
            /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false);

        return result;
    }

    SpmPipelineComponentLayout prepare_pooler_z1_layout(
        int64_t num_patches,
        int64_t retained_count) {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen3VLVisionModel pooler Z1 prepare must run outside "
                    "Graph capture");
        TORCH_CHECK(!multiview_dry_prepared_ &&
                        !multiview_composite_prepared_ &&
                        !multiview_composite_dispatch_ &&
                        z1_prepared_num_patches_ == 0 && !z1_inputs_primed_ &&
                        !z1_adopted_ && !z1_bound_ && !z1_dispatch_ &&
                        z1_retained_deepstack_count_ == 0 &&
                        z1_retained_deepstack_bound_count_ == 0 &&
                        retained_deepstack_addresses_empty(),
                    "Qwen3VLVisionModel pooler Z1 prepare requires a fresh "
                    "component lifecycle");
        TORCH_CHECK(retained_count == 0 || retained_count == 1 ||
                        retained_count == 3,
                    "Qwen3VLVisionModel pooler Z1 retained DeepStack count "
                    "must be exactly 0, 1, or 3, got ", retained_count);
        TORCH_CHECK(num_patches == QWEN3VL_POOLER_Z1_NUM_PATCHES,
                    "Qwen3VLVisionModel pooler Z1 canary admits exactly ",
                    QWEN3VL_POOLER_Z1_NUM_PATCHES, " patches, got ",
                    num_patches);
        validate_pooler_z1_profile();
        if (retained_count != 0) {
            validate_pooler_z1_retained_deepstack_profile(retained_count);
        }

        // Fixed Q staging uses fixed-address DMA in both KV_FIRST phases. Create
        // its exact backing slot before the coordinator opens the outer Graph.
        const int64_t local_q_dim =
            (num_q_heads() / NUM_CORES) * head_dim();
        const std::pair<int64_t, int64_t> q_key{num_patches, local_q_dim};
        if (q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
            TORCH_CHECK(q_ddr_slots_.size() < 128,
                        "Qwen3VLVisionModel pooler Z1 cannot create its Q "
                        "staging slot: per-shape slot cap reached");
            q_ddr_slots_.emplace(
                q_key,
                at::empty(
                    {NUM_CORES, num_patches, local_q_dim},
                    at::TensorOptions()
                        .dtype(at::kHalf)
                        .device(at::kPrivateUse1)));
        }
        const at::Tensor& q_slot = q_ddr_slots_.at(q_key);
        TORCH_CHECK(q_slot.scalar_type() == at::kHalf &&
                        q_slot.device().type() == at::kPrivateUse1 &&
                        q_slot.is_contiguous() &&
                        q_slot.dim() == 3 &&
                        q_slot.size(0) == NUM_CORES &&
                        q_slot.size(1) == num_patches &&
                        q_slot.size(2) == local_q_dim,
                    "Qwen3VLVisionModel pooler Z1 Q staging slot drifted");

        current_num_patches_ = num_patches;
        image_batch_count_ = 1;
        set_chunk_size_override(QWEN3VL_POOLER_Z1_NUM_PATCHES);
        const int64_t resolved = resolve_chunk_size_for_shape(
            num_patches, /*position=*/0, /*attention_mask=*/std::nullopt,
            /*is_causal=*/false);
        TORCH_CHECK(resolved == num_patches,
                    "Qwen3VLVisionModel pooler Z1 requires one full Vision "
                    "chunk; planner resolved ", resolved, " for ", num_patches,
                    " patches");

        LayoutContext layout;
        layout.chunk_size = resolved;
        layout.max_kv_seq_len = num_patches;
        layout.num_layers = num_layers();
        layout.use_attn_mask = false;
        layout.is_causal = false;
        auto prepared = prepare_spm_pipeline_component(layout);
        TORCH_CHECK(prepared.temporary_bytes ==
                        QWEN3VL_POOLER_Z1_SCRATCH_BYTES,
                    "Qwen3VLVisionModel pooler Z1 scratch layout drifted: "
                    "expected ", QWEN3VL_POOLER_Z1_SCRATCH_BYTES,
                    " bytes/core, got ", prepared.temporary_bytes);
        z1_retained_deepstack_count_ = retained_count;
        z1_prepared_num_patches_ = num_patches;
        return prepared;
    }

    SpmPipelineComponentLayout prepare_multiview_dry_layout(
        int64_t num_patches) {
        TORCH_CHECK(
            !RpuKernelGraph::has_active(),
            "Qwen3VLVisionModel multiview dry prepare must run outside Graph "
            "capture");
        TORCH_CHECK(
            !multiview_dry_prepared_ &&
                !multiview_composite_prepared_ &&
                !multiview_composite_dispatch_ &&
                z1_prepared_num_patches_ == 0 &&
                !z1_inputs_primed_ && !z1_adopted_ && !z1_bound_ &&
                !z1_dispatch_ && z1_retained_deepstack_count_ == 0 &&
                z1_retained_deepstack_bound_count_ == 0 &&
                retained_deepstack_addresses_empty() &&
                z1_source_addr_ == 0 && z1_epoch_ == 0 &&
                z1_plan_hash_ == 0 && z1_primed_position_ptr_ == nullptr &&
                z1_primed_position_owner_ == nullptr,
            "Qwen3VLVisionModel multiview dry prepare requires a fresh "
            "lifecycle without a physical Z1 profile");
        TORCH_CHECK(
            num_patches == 256 && num_layers() == 24 &&
                num_q_heads() == 16 && head_dim() == 64 &&
                hidden_size() == 1024 && intermediate_size() == 4096,
            "Qwen3VLVisionModel multiview dry prepare accepts only the exact "
            "N256/L24/H1024/I4096/Q16/D64 Vision profile");
        TORCH_CHECK(
            has_rope_ && rope_spm_enabled_ && max_hw_ == 48 &&
                merger_active() &&
                merge_hidden_ == 4096 && merger_out_hidden_ == 2560 &&
                deepstack_visual_indexes_.size() == 3 &&
                deepstack_visual_indexes_[0] == 5 &&
                deepstack_visual_indexes_[1] == 11 &&
                deepstack_visual_indexes_[2] == 17 &&
                deepstack_merger_active() &&
                ds_merger_ln_q_w_.size() == 3 &&
                ds_merger_ln_q_b_.size() == 3 &&
                ds_merger_m0_w_.size() == 3 &&
                ds_merger_m0_b_.size() == 3 &&
                ds_merger_m2_w_.size() == 3 &&
                ds_merger_m2_b_.size() == 3,
            "Qwen3VLVisionModel multiview dry prepare requires the exact "
            "48-row SPM RoPE tables, fused 1024->2560 merger, and taps "
            "[5,11,17]");
        TORCH_CHECK(
            layer_weights_.size() == 24 && layer_bias_norm_.size() == 24,
            "Qwen3VLVisionModel multiview dry prepare requires complete "
            "Vision weights");
        validate_multiview_composite_profile();

        LayoutContext layout;
        layout.chunk_size = num_patches;
        layout.max_kv_seq_len = num_patches;
        layout.num_layers = num_layers();
        layout.use_attn_mask = false;
        layout.is_causal = false;
        multiview_dry_prepared_ = true;
        auto rollback = c10::make_scope_exit(
            [&] { multiview_dry_prepared_ = false; });
        SpmPipelineComponentLayout prepared =
            prepare_spm_pipeline_component_for_cpu_contract(layout);
        rollback.release();
        return prepared;
    }

    void cancel_multiview_dry_layout() {
        TORCH_CHECK(
            !RpuKernelGraph::has_active(),
            "Qwen3VLVisionModel multiview dry cancel must run outside Graph "
            "capture");
        TORCH_CHECK(
            multiview_dry_prepared_,
            "Qwen3VLVisionModel multiview dry cancel requires one prepared "
            "dry layout");
        cancel_spm_pipeline_component_for_cpu_contract();
        multiview_dry_prepared_ = false;
    }

    SpmPipelineComponentLayout prepare_multiview_composite_layout(
        int64_t num_patches,
        bool force_block_twostage) {
        TORCH_CHECK(
            !RpuKernelGraph::has_active(),
            "Qwen3VLVisionModel multiview composite prepare must run outside "
            "Graph capture");
        TORCH_CHECK(
            num_patches == QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                !multiview_dry_prepared_ &&
                !multiview_composite_prepared_ &&
                !multiview_composite_dispatch_ &&
                !multiview_force_block_twostage_ &&
                z1_prepared_num_patches_ == 0 && !z1_inputs_primed_ &&
                !z1_adopted_ && !z1_bound_ && !z1_dispatch_ &&
                z1_retained_deepstack_count_ == 0 &&
                z1_retained_deepstack_bound_count_ == 0 &&
                retained_deepstack_addresses_empty(),
            "Qwen3VLVisionModel multiview composite prepare requires one "
            "fresh exact N256 component lifecycle");
        validate_multiview_composite_profile();
        TORCH_CHECK(
            !force_block_twostage ||
                classify_multiview_linear_profile() ==
                    MultiviewLinearProfile::W8A16,
            "Qwen3VLVisionModel multiview block two-stage is admitted only "
            "for the exact W8A16 composite profile");

        const int64_t local_q_dim =
            (num_q_heads() / NUM_CORES) * head_dim();
        const std::pair<int64_t, int64_t> q_key{
            num_patches, local_q_dim};
        if (q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
            TORCH_CHECK(q_ddr_slots_.size() < 128,
                        "Qwen3VLVisionModel multiview composite cannot create "
                        "its fixed Q staging slot: per-shape cap reached");
            q_ddr_slots_.emplace(
                q_key,
                at::empty(
                    {NUM_CORES, num_patches, local_q_dim},
                    at::TensorOptions()
                        .dtype(at::kHalf)
                        .device(at::kPrivateUse1)));
        }

        current_num_patches_ = num_patches;
        image_batch_count_ = 1;
        set_chunk_size_override(num_patches);
        const int64_t resolved = resolve_chunk_size_for_shape(
            num_patches, /*position=*/0, /*attention_mask=*/std::nullopt,
            /*is_causal=*/false);
        TORCH_CHECK(
            resolved == num_patches,
            "Qwen3VLVisionModel multiview composite requires one full N256 "
            "Vision chunk; planner resolved ", resolved);

        LayoutContext layout;
        layout.chunk_size = resolved;
        layout.max_kv_seq_len = num_patches;
        layout.num_layers = num_layers();
        layout.use_attn_mask = false;
        layout.is_causal = false;
        multiview_force_block_twostage_ = force_block_twostage;
        multiview_composite_prepared_ = true;
        auto rollback = c10::make_scope_exit(
            [&] {
                multiview_composite_prepared_ = false;
                multiview_force_block_twostage_ = false;
            });
        SpmPipelineComponentLayout prepared =
            prepare_spm_pipeline_component(layout);
        TORCH_CHECK(
            prepared.temporary_bytes ==
                QWEN3VL_MULTIVIEW_VISION_SCRATCH_BYTES,
            "Qwen3VLVisionModel multiview composite scratch drifted: "
            "expected ", QWEN3VL_MULTIVIEW_VISION_SCRATCH_BYTES,
            " bytes/core, got ", prepared.temporary_bytes);
        rollback.release();
        return prepared;
    }

    SpmFmbResolvedExecutionProfile
    resolve_multiview_composite_profile() {
        TORCH_CHECK(
            multiview_composite_prepared_ &&
                !multiview_composite_dispatch_,
            "Qwen3VLVisionModel multiview composite profile requires one "
            "prepared, idle component");
        SpmFmbResolvedProfileRequest request;
        const MultiviewLinearProfile linear_profile =
            classify_multiview_linear_profile();
        request.version =
            linear_profile == MultiviewLinearProfile::DenseFp16
            ? QWEN3VL_MULTIVIEW_FP16_PROFILE_VERSION
            : multiview_force_block_twostage_
            ? QWEN3VL_MULTIVIEW_W8A16_TWOSTAGE_PROFILE_VERSION
            : QWEN3VL_MULTIVIEW_W8A16_PROFILE_VERSION;
        request.chunks = {{0, 0, 256, 256}};
        request.kv_insert_chunks = request.chunks;
        return resolve_spm_pipeline_execution_profile_for_build_trace(request);
    }

    uint64_t multiview_composite_occurrence_policy() const {
        TORCH_CHECK(
            multiview_composite_prepared_ &&
                !multiview_composite_dispatch_,
            "Qwen3VLVisionModel multiview policy requires one prepared, "
            "idle component");
        validate_multiview_composite_profile();
        return QWEN3VL_MULTIVIEW_OCCURRENCE_POLICY;
    }

    SpmFmbResolvedPhaseManifest seal_multiview_composite_manifest(
        const SpmFmbResolvedExecutionProfile& profile,
        SpmScratchId arena,
        uint32_t arena_base) const {
        TORCH_CHECK(
            multiview_composite_prepared_ &&
                !multiview_composite_dispatch_,
            "Qwen3VLVisionModel multiview manifest requires one prepared, "
            "idle component");
        validate_multiview_composite_profile();
        return seal_spm_pipeline_live_resolved_manifest(
            profile, arena, arena_base);
    }

    void prime_multiview_composite_position() {
        TORCH_CHECK(
            !RpuKernelGraph::has_active() &&
                multiview_composite_prepared_ &&
                !multiview_composite_dispatch_,
            "Qwen3VLVisionModel multiview position prime requires one idle "
            "prepared component outside Graph capture");
        TORCH_CHECK(
            position_idx_keepalive_.defined() &&
                position_idx_keepalive_.scalar_type() == at::kShort &&
                position_idx_keepalive_.device().type() ==
                    at::kPrivateUse1 &&
                position_idx_keepalive_.is_contiguous() &&
                position_idx_keepalive_.size(0) >=
                    QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                position_idx_keepalive_.size(1) == 2,
            "Qwen3VLVisionModel multiview position owner must be contiguous "
            "int16 RPU [>=256,2]");
        rpu_ddr_flush_force_sized(
            position_idx_keepalive_.data_ptr<int16_t>(),
            QWEN3VL_POOLER_Z1_NUM_PATCHES * 2 * sizeof(int16_t));
        multiview_primed_position_ptr_ =
            position_idx_keepalive_.data_ptr<int16_t>();
        multiview_primed_position_owner_ =
            position_idx_keepalive_.unsafeGetTensorImpl();
    }

    void clear_multiview_composite_position() {
        TORCH_CHECK(
            !RpuKernelGraph::has_active() &&
                !multiview_composite_dispatch_,
            "Qwen3VLVisionModel multiview position clear must run outside "
            "Graph capture");
        multiview_primed_position_ptr_ = nullptr;
        multiview_primed_position_owner_ = nullptr;
    }

    void stage_multiview_vision_outer_fast_component(
        GraphKernelRegisterCensusGuard& guard,
        at::TensorList k_caches,
        at::TensorList v_caches) const {
        validate_multiview_composite_profile();
        TORCH_CHECK(
            multiview_composite_prepared_ &&
                !multiview_composite_dispatch_ && !z1_dispatch_ &&
                multiview_primed_position_ptr_ != nullptr &&
                multiview_primed_position_ptr_ ==
                    position_idx_keepalive_.data_ptr<int16_t>() &&
                multiview_primed_position_owner_ != nullptr &&
                multiview_primed_position_owner_ ==
                    position_idx_keepalive_.unsafeGetTensorImpl(),
            "Qwen3VLVisionModel multiview outer-fast component requires "
            "one prepared idle component with its exact primed position "
            "owner");
        stage_spm_outer_fast_component(guard, k_caches, v_caches);
    }

    void bind_multiview_vision_outer_fast_input(
        GraphKernelRegisterCensusGuard& guard,
        const at::Tensor& input,
        size_t ordinal) {
        validate_multiview_composite_profile();
        TORCH_CHECK(
            multiview_composite_prepared_ &&
                !multiview_composite_dispatch_ && !z1_dispatch_ &&
                multiview_primed_position_ptr_ != nullptr &&
                multiview_primed_position_ptr_ ==
                    position_idx_keepalive_.data_ptr<int16_t>() &&
                multiview_primed_position_owner_ != nullptr &&
                multiview_primed_position_owner_ ==
                    position_idx_keepalive_.unsafeGetTensorImpl(),
            "Qwen3VLVisionModel multiview outer-fast input requires one "
            "prepared idle component with its exact primed position owner");
        TORCH_CHECK(
            ordinal < 3 && input.defined() && input.dim() == 3 &&
                input.size(0) == 1 &&
                input.size(1) == QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                input.size(2) == QWEN3VL_POOLER_Z1_VISION_HIDDEN &&
                input.scalar_type() == at::kHalf &&
                input.device().type() == at::kPrivateUse1 &&
                input.layout() == c10::Layout::Strided &&
                input.is_contiguous(),
            "Qwen3VLVisionModel multiview outer-fast input must be one of "
            "three contiguous FP16 RPU [1,256,1024] tensors");
        bind_spm_outer_fast_composite_input(guard, input, ordinal);
    }

    void unprepare_multiview_composite_layout() {
        TORCH_CHECK(
            !RpuKernelGraph::has_active() &&
                multiview_composite_prepared_ &&
                !multiview_composite_dispatch_ &&
                multiview_primed_position_ptr_ == nullptr &&
                multiview_primed_position_owner_ == nullptr,
            "Qwen3VLVisionModel multiview composite unprepare requires one "
            "prepared, idle component outside Graph capture");
        multiview_composite_prepared_ = false;
        multiview_force_block_twostage_ = false;
    }

    void prime_pooler_z1_inputs(int64_t num_patches) {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen3VLVisionModel pooler Z1 input priming must run "
                    "outside Graph capture");
        TORCH_CHECK(num_patches == QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                        z1_prepared_num_patches_ == num_patches,
                    "Qwen3VLVisionModel pooler Z1 input priming requires the "
                    "prepared 256-patch profile");
        TORCH_CHECK(!z1_inputs_primed_ && !z1_adopted_ && !z1_bound_ &&
                        z1_retained_deepstack_bound_count_ == 0 &&
                        retained_deepstack_addresses_empty() && !z1_dispatch_,
                    "Qwen3VLVisionModel pooler Z1 input priming requires a "
                    "prepared-only component");
        validate_pooler_z1_profile();
        if (z1_retained_deepstack_count_ != 0) {
            validate_pooler_z1_retained_deepstack_profile(
                z1_retained_deepstack_count_);
        }
        TORCH_CHECK(position_idx_keepalive_.defined() &&
                        position_idx_keepalive_.device().type() ==
                            at::kPrivateUse1 &&
                        position_idx_keepalive_.scalar_type() == at::kShort &&
                        position_idx_keepalive_.is_contiguous() &&
                        position_idx_keepalive_.dim() == 2 &&
                        position_idx_keepalive_.size(0) >= num_patches &&
                        position_idx_keepalive_.size(1) == 2,
                    "Qwen3VLVisionModel pooler Z1 requires a contiguous RPU "
                    "int16 position keepalive [>=256,2]");

        rpu_ddr_flush_force_sized(
            position_idx_keepalive_.data_ptr<int16_t>(),
            static_cast<size_t>(num_patches * 2 * sizeof(int16_t)));
        z1_primed_position_ptr_ =
            position_idx_keepalive_.data_ptr<int16_t>();
        z1_primed_position_owner_ =
            position_idx_keepalive_.unsafeGetTensorImpl();
        z1_inputs_primed_ = true;
    }

    void rollback_pooler_z1_inputs() {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen3VLVisionModel pooler Z1 input rollback must run "
                    "outside Graph capture");
        TORCH_CHECK(!z1_adopted_ && !z1_bound_ &&
                        z1_retained_deepstack_bound_count_ == 0 &&
                        retained_deepstack_addresses_empty() && !z1_dispatch_,
                    "Qwen3VLVisionModel pooler Z1 input rollback requires a "
                    "primed-only component");
        z1_inputs_primed_ = false;
        z1_primed_position_ptr_ = nullptr;
        z1_primed_position_owner_ = nullptr;
    }

    void unprepare_pooler_z1_layout() {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen3VLVisionModel pooler Z1 unprepare must run outside "
                    "Graph capture");
        TORCH_CHECK(!z1_adopted_ && !z1_bound_ &&
                        z1_retained_deepstack_bound_count_ == 0 &&
                        retained_deepstack_addresses_empty() && !z1_dispatch_,
                    "Qwen3VLVisionModel pooler Z1 unprepare requires no "
                    "active component lease");
        rollback_pooler_z1_inputs();
        z1_retained_deepstack_count_ = 0;
        z1_retained_deepstack_bound_count_ = 0;
        z1_retained_deepstack_source_addrs_.fill(0);
        z1_prepared_num_patches_ = 0;
    }

    SpmDense2DSpec pooler_z1_source_spec(int64_t num_patches) const {
        TORCH_CHECK(num_patches == QWEN3VL_POOLER_Z1_NUM_PATCHES,
                    "Qwen3VLVisionModel pooler Z1 source admits exactly ",
                    QWEN3VL_POOLER_Z1_NUM_PATCHES, " patches, got ",
                    num_patches);
        validate_pooler_z1_profile();
        SpmDense2DSpec spec;
        spec.dtype = SpmPortDType::Fp16;
        spec.rows = QWEN3VL_POOLER_Z1_MERGED_ROWS;
        spec.cols = QWEN3VL_POOLER_Z1_TEXT_HIDDEN;
        spec.distribution = SpmPortDistribution::Replicated;
        spec.validate();
        return spec;
    }

    SpmDense2DSpec pooler_z1_deepstack_source_spec(
        int64_t num_patches,
        int64_t ordinal) const {
        TORCH_CHECK(num_patches == QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                        z1_prepared_num_patches_ == num_patches &&
                        ordinal >= 0 &&
                        ordinal < z1_retained_deepstack_count_,
                    "Qwen3VLVisionModel pooler Z1 retained DeepStack source "
                    "requires a requested ordinal in the prepared 256-patch "
                    "mode; ordinal=", ordinal, " retained_count=",
                    z1_retained_deepstack_count_);
        validate_pooler_z1_profile();
        validate_pooler_z1_retained_deepstack_profile(
            z1_retained_deepstack_count_);
        SpmDense2DSpec spec;
        spec.dtype = SpmPortDType::Fp16;
        spec.rows = QWEN3VL_POOLER_Z1_MERGED_ROWS;
        spec.cols = QWEN3VL_POOLER_Z1_TEXT_HIDDEN;
        spec.distribution = SpmPortDistribution::Replicated;
        spec.validate();
        return spec;
    }

    void adopt_pooler_z1_layout(const SpmPipelineLease& lease,
                                const SpmTensorView& scratch) {
        validate_pooler_z1_contract();
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen3VLVisionModel pooler Z1 adopt must run outside Graph "
                    "capture");
        TORCH_CHECK(z1_inputs_primed_ && !z1_adopted_ && !z1_bound_ &&
                        z1_retained_deepstack_bound_count_ == 0 &&
                        retained_deepstack_addresses_empty() && !z1_dispatch_,
                    "Qwen3VLVisionModel pooler Z1 adopt requires one primed, "
                    "unbound component");
        adopt_spm_pipeline_component(lease, scratch);
        z1_epoch_ = lease.epoch();
        z1_plan_hash_ = lease.plan_hash();
        z1_adopted_ = true;
    }

    void bind_pooler_z1_source(const SpmPipelineLease& lease,
                               const SpmPortView& source) {
        validate_pooler_z1_contract();
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen3VLVisionModel pooler Z1 bind must run outside Graph "
                    "capture");
        TORCH_CHECK(z1_adopted_ && !z1_bound_ && !z1_dispatch_ &&
                        z1_epoch_ == lease.epoch() &&
                        z1_plan_hash_ == lease.plan_hash(),
                    "Qwen3VLVisionModel pooler Z1 bind requires the adopted "
                    "live lease");
        const auto expected =
            pooler_z1_source_spec(z1_prepared_num_patches_);
        TORCH_CHECK(source.spec() == expected &&
                        source.size_bytes() == expected.storage_bytes(),
                    "Qwen3VLVisionModel pooler Z1 source port must be "
                    "replicated FP16 [64,2048]");
        z1_source_addr_ = source.resolve_physical_addr(/*core=*/0, lease);
        TORCH_CHECK(z1_source_addr_ != 0,
                    "Qwen3VLVisionModel pooler Z1 source resolved a zero "
                    "physical address");
        for (int64_t ordinal = 0;
             ordinal < z1_retained_deepstack_bound_count_; ++ordinal) {
            TORCH_CHECK(
                z1_source_addr_ != z1_retained_deepstack_source_addrs_[ordinal],
                "Qwen3VLVisionModel pooler and retained DeepStack source ",
                ordinal, " must not alias");
        }
        z1_bound_ = true;
    }

    void bind_pooler_z1_deepstack_source(
        const SpmPipelineLease& lease,
        const SpmPortView& source,
        int64_t ordinal) {
        validate_pooler_z1_contract();
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen3VLVisionModel pooler Z1 retained DeepStack bind must run "
                    "outside Graph capture");
        TORCH_CHECK(z1_adopted_ && !z1_dispatch_ &&
                        ordinal >= 0 &&
                        ordinal < z1_retained_deepstack_count_ &&
                        ordinal == z1_retained_deepstack_bound_count_ &&
                        z1_epoch_ == lease.epoch() &&
                        z1_plan_hash_ == lease.plan_hash(),
                    "Qwen3VLVisionModel pooler Z1 retained DeepStack bind "
                    "requires the next requested ordinal on the adopted live "
                    "lease; ordinal=", ordinal, " bound_count=",
                    z1_retained_deepstack_bound_count_, " retained_count=",
                    z1_retained_deepstack_count_);
        const auto expected = pooler_z1_deepstack_source_spec(
            z1_prepared_num_patches_, ordinal);
        TORCH_CHECK(source.spec() == expected &&
                        source.size_bytes() == expected.storage_bytes(),
                    "Qwen3VLVisionModel pooler Z1 retained DeepStack source port must "
                    "be replicated FP16 [64,2048]");
        const uint32_t resolved_addr =
            source.resolve_physical_addr(/*core=*/0, lease);
        TORCH_CHECK(resolved_addr != 0,
                    "Qwen3VLVisionModel pooler Z1 retained DeepStack source ",
                    ordinal, " resolved a zero physical address");
        TORCH_CHECK(!z1_bound_ || resolved_addr != z1_source_addr_,
                    "Qwen3VLVisionModel retained DeepStack source ", ordinal,
                    " and pooler source must not alias");
        for (int64_t prior = 0; prior < ordinal; ++prior) {
            TORCH_CHECK(
                resolved_addr != z1_retained_deepstack_source_addrs_[prior],
                "Qwen3VLVisionModel retained DeepStack sources ", prior,
                " and ", ordinal, " must not alias");
        }
        z1_retained_deepstack_source_addrs_[ordinal] = resolved_addr;
        ++z1_retained_deepstack_bound_count_;
    }

    void validate_pooler_z1_layout(const SpmPipelineLease& lease) const {
        validate_pooler_z1_contract();
        TORCH_CHECK(z1_adopted_ && z1_bound_ && z1_source_addr_ != 0 &&
                        z1_inputs_primed_ &&
                        retained_deepstack_bindings_complete(),
                    "Qwen3VLVisionModel pooler Z1 has no complete active "
                    "source binding");
        for (int64_t ordinal = 0;
             ordinal < z1_retained_deepstack_count_; ++ordinal) {
            TORCH_CHECK(
                z1_source_addr_ != z1_retained_deepstack_source_addrs_[ordinal],
                "Qwen3VLVisionModel pooler Z1 source aliases retained "
                "DeepStack source ", ordinal);
            for (int64_t prior = 0; prior < ordinal; ++prior) {
                TORCH_CHECK(
                    z1_retained_deepstack_source_addrs_[prior] !=
                        z1_retained_deepstack_source_addrs_[ordinal],
                    "Qwen3VLVisionModel retained DeepStack sources ", prior,
                    " and ", ordinal, " alias");
            }
        }
        TORCH_CHECK(z1_epoch_ == lease.epoch() &&
                        z1_plan_hash_ == lease.plan_hash(),
                    "Qwen3VLVisionModel pooler Z1 has a stale lease binding");
        TORCH_CHECK(z1_primed_position_ptr_ ==
                        position_idx_keepalive_.data_ptr<int16_t>() &&
                        z1_primed_position_owner_ ==
                            position_idx_keepalive_.unsafeGetTensorImpl(),
                    "Qwen3VLVisionModel pooler Z1 position keepalive owner or "
                    "address drifted after priming");
        validate_spm_pipeline_component(lease);
    }

    void clear_pooler_z1_layout(uint64_t epoch, uint64_t plan_hash) {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "Qwen3VLVisionModel pooler Z1 clear must run outside Graph "
                    "capture");
        TORCH_CHECK(z1_adopted_ && !z1_dispatch_ &&
                        z1_epoch_ == epoch && z1_plan_hash_ == plan_hash,
                    "Qwen3VLVisionModel pooler Z1 clear received a stale or "
                    "in-flight lease token");
        release_spm_pipeline_component(epoch, plan_hash);
        z1_adopted_ = false;
        z1_bound_ = false;
        z1_source_addr_ = 0;
        z1_retained_deepstack_bound_count_ = 0;
        z1_retained_deepstack_source_addrs_.fill(0);
        z1_epoch_ = 0;
        z1_plan_hash_ = 0;
        z1_inputs_primed_ = false;
        z1_primed_position_ptr_ = nullptr;
        z1_primed_position_owner_ = nullptr;
    }

    void stage_pooler_z1_outer_fast_component(
        GraphKernelRegisterCensusGuard& guard,
        at::TensorList k_caches,
        at::TensorList v_caches) const {
        validate_pooler_z1_contract();
        TORCH_CHECK(z1_adopted_ && z1_bound_ && !z1_dispatch_ &&
                        z1_inputs_primed_ && z1_source_addr_ != 0 &&
                        retained_deepstack_bindings_complete() &&
                        z1_epoch_ != 0 && z1_plan_hash_ != 0,
                    "Qwen3VLVisionModel pooler Z1 outer-fast component "
                    "requires the exact active source binding");
        TORCH_CHECK(z1_primed_position_ptr_ != nullptr &&
                        z1_primed_position_ptr_ ==
                            position_idx_keepalive_.data_ptr<int16_t>() &&
                        z1_primed_position_owner_ != nullptr &&
                        z1_primed_position_owner_ ==
                            position_idx_keepalive_.unsafeGetTensorImpl(),
                    "Qwen3VLVisionModel pooler Z1 outer-fast position owner "
                    "drifted after Graph-external priming");
        stage_spm_outer_fast_component(guard, k_caches, v_caches);
    }

    void bind_pooler_z1_outer_fast_input(
        GraphKernelRegisterCensusGuard& guard,
        const at::Tensor& input) {
        validate_pooler_z1_contract();
        TORCH_CHECK(z1_adopted_ && z1_bound_ && !z1_dispatch_ &&
                        z1_inputs_primed_ && z1_source_addr_ != 0 &&
                        retained_deepstack_bindings_complete() &&
                        z1_epoch_ != 0 && z1_plan_hash_ != 0,
                    "Qwen3VLVisionModel pooler Z1 outer-fast input requires "
                    "the exact active source binding");
        TORCH_CHECK(input.defined() && input.dim() == 3 &&
                        input.size(0) == 1 &&
                        input.size(1) == QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                        input.size(2) == QWEN3VL_POOLER_Z1_VISION_HIDDEN &&
                        input.scalar_type() == at::kHalf &&
                        input.device().type() == at::kPrivateUse1 &&
                        input.layout() == c10::Layout::Strided &&
                        input.is_contiguous(),
                    "Qwen3VLVisionModel pooler Z1 outer-fast input must be "
                    "contiguous FP16 RPU [1,256,1024]");
        bind_spm_outer_fast_input(guard, input);
    }

    void forward_pooler_z1(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches,
        uint64_t epoch,
        uint64_t plan_hash) {
        RECORD_FUNCTION("qwen3vl_pooler_forward_z1", {});
        TORCH_CHECK(RpuKernelGraph::has_active(),
                    "Qwen3VLVisionModel pooler Z1 forward requires one active "
                    "outer Graph");
        validate_pooler_z1_contract();
        TORCH_CHECK(z1_adopted_ && z1_bound_ && z1_inputs_primed_ &&
                        retained_deepstack_bindings_complete() &&
                        z1_epoch_ == epoch && z1_plan_hash_ == plan_hash,
                    "Qwen3VLVisionModel pooler Z1 forward received a stale "
                    "lease token");
        TORCH_CHECK(!z1_dispatch_,
                    "Qwen3VLVisionModel pooler Z1 forward is not reentrant");

        z1_dispatch_ = true;
        try {
            (void)forward_impl(input, k_caches, v_caches, num_patches,
                               /*image_batch_count=*/1,
                               /*z1_dispatch=*/true);
            TORCH_CHECK(get_last_resolved_chunk_size() ==
                            QWEN3VL_POOLER_Z1_NUM_PATCHES,
                        "Qwen3VLVisionModel pooler Z1 chunk drifted after "
                        "prepare");
            z1_dispatch_ = false;
        } catch (...) {
            z1_dispatch_ = false;
            throw;
        }
    }

    void forward_multiview_composite(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches) {
        RECORD_FUNCTION("qwen3vl_multiview_composite_vision", {});
        TORCH_CHECK(
            RpuKernelGraph::has_active() &&
                multiview_composite_prepared_ &&
                !multiview_composite_dispatch_ && !z1_dispatch_,
            "Qwen3VLVisionModel multiview composite forward requires one "
            "prepared, non-reentrant occurrence in the active outer Graph");
        validate_multiview_composite_profile();
        multiview_composite_dispatch_ = true;
        auto clear_dispatch = c10::make_scope_exit(
            [&] { multiview_composite_dispatch_ = false; });
        (void)forward_impl(
            input, k_caches, v_caches,
            QWEN3VL_POOLER_Z1_NUM_PATCHES,
            /*image_batch_count=*/1,
            /*z1_dispatch=*/false);
        TORCH_CHECK(
            get_last_resolved_chunk_size() ==
                QWEN3VL_POOLER_Z1_NUM_PATCHES,
            "Qwen3VLVisionModel multiview composite chunk drifted after "
            "prepare");
    }

    void check_pooler_z1_destroy_allowed() const {
        TORCH_CHECK(!z1_adopted_ && !z1_bound_ && !z1_dispatch_ &&
                        !z1_inputs_primed_ &&
                        z1_retained_deepstack_count_ == 0 &&
                        z1_retained_deepstack_bound_count_ == 0 &&
                        retained_deepstack_addresses_empty() &&
                        z1_prepared_num_patches_ == 0 &&
                        !multiview_dry_prepared_ &&
                        !multiview_composite_prepared_ &&
                        !multiview_composite_dispatch_ &&
                        !multiview_force_block_twostage_ &&
                        multiview_primed_position_ptr_ == nullptr &&
                        multiview_primed_position_owner_ == nullptr,
                    "cannot destroy the Qwen3VL Vision handle while its pooler "
                    "Z1 or multiview layout is prepared, primed, or active; "
                    "clear the outer GraphCache and unprepare/cancel first");
    }

    // Retrieve deepstack snapshots for the most-recent forward.
    //
    // Looks up the shape-keyed `(layer_idx, num_patches)` slot persisted in
    // `deepstack_ddr_slots_`. The slots are written by build_layer_subgraph's
    // DMA emission on BUILD; on REPLAY the same slot DDR address is reused,
    // so the slot tensor stays valid and contains the most-recent forward's
    // hidden state. Caller MUST consume the tensors before issuing the next
    // forward (REPLAY at a different shape would route to a different slot
    // but same-shape REPLAY would overwrite). Adapter flushes the DDR before
    // reading to ensure RPU-DMA-written contents are visible CPU-side.
    std::vector<at::Tensor> pop_deepstack_snapshots() {
        std::vector<at::Tensor> out;
        out.reserve(deepstack_visual_indexes_.size());
        for (int64_t layer_idx : deepstack_visual_indexes_) {
            auto key = std::make_pair(layer_idx, current_num_patches_);
            at::Tensor* slot = deepstack_ddr_slots_.find(key);
            TORCH_CHECK(slot != nullptr,
                        "qwen3vl_vision_pop_deepstack_snapshots: missing slot for "
                        "layer_idx=", layer_idx, " num_patches=", current_num_patches_,
                        " — forward() must run at this shape before popping");
            rpu_ddr_flush_force(slot->data_ptr<c10::Half>());
            out.push_back(*slot);
        }
        return out;
    }

    // Retrieve the in-graph PATCH-merged pooler [num_patches/4, oh] for the most-recent forward
    // (stable slot, fixed DMA → deterministic). Replaces the mutable post_output_tensor return.
    at::Tensor pop_pooler_merged() {
        at::Tensor* slot = pooler_merged_ddr_slots_.find(current_num_patches_);
        TORCH_CHECK(slot != nullptr,
                    "qwen3vl_vision_pop_pooler_merged: missing pooler slot for num_patches=",
                    current_num_patches_, " — forward() with the fused merger must run first");
        rpu_ddr_flush_force(slot->data_ptr<c10::Half>());
        return *slot;
    }

    // Retrieve the in-Graph merged deepstack features [num_patches/4, oh] for the
    // most-recent forward (the fused deepstack merger wrote them in merger_post_fn). Stable
    // slots (fixed DMA), shape-keyed like the snapshots. Used by the adapter instead of
    // pop_deepstack_snapshots() + the eager merger when the fused merger is active.
    std::vector<at::Tensor> pop_deepstack_merged() {
        std::vector<at::Tensor> out;
        out.reserve(deepstack_visual_indexes_.size());
        for (int64_t layer_idx : deepstack_visual_indexes_) {
            auto key = std::make_pair(layer_idx, current_num_patches_);
            at::Tensor* slot = deepstack_merged_ddr_slots_.find(key);
            TORCH_CHECK(slot != nullptr,
                        "qwen3vl_vision_pop_deepstack_merged: missing merged slot for "
                        "layer_idx=", layer_idx, " num_patches=", current_num_patches_,
                        " — forward() with the fused merger must run at this shape first");
            rpu_ddr_flush_force(slot->data_ptr<c10::Half>());
            out.push_back(*slot);
        }
        return out;
    }

    // Convenience: ALL in-graph merged outputs in one call — patch-merged pooler [seq/4, oh]
    // FIRST, then the N deepstack-merged features (same order as pop_deepstack_merged). One
    // op-dispatch instead of two for the fused-merger adapter hot path; composes the two pops
    // above (each does its own DDR flush + stable-slot lookup). All are stable slots returned
    // BY REFERENCE → the caller MUST clone before the next same-shape forward overwrites them.
    std::vector<at::Tensor> pop_merged() {
        std::vector<at::Tensor> out;
        out.reserve(1 + deepstack_visual_indexes_.size());
        out.push_back(pop_pooler_merged());
        auto ds = pop_deepstack_merged();
        out.insert(out.end(), ds.begin(), ds.end());
        return out;
    }

protected:
    SpmFmbTraversalCapability spm_fmb_traversal_capability() const override {
        return multiview_composite_prepared_
            ? SpmFmbTraversalCapability::CanonicalTraversal
            : SpmFmbTraversalCapability::Unsealed;
    }

    SpmFmbPreloadCapability spm_fmb_preload_capability() const override {
        return SpmFmbPreloadCapability::PersistentOutsideResolvedWindow;
    }

    SpmFmbCompositeOccurrenceCapability
    spm_fmb_composite_occurrence_capability() const override {
        return multiview_composite_prepared_
            ? SpmFmbCompositeOccurrenceCapability::
                  StableThreeInputSlotsSameOwnerPreload
            : SpmFmbCompositeOccurrenceCapability::Unsealed;
    }

    uint64_t
    spm_fmb_composite_occurrence_policy_fingerprint() const override {
        return multiview_composite_prepared_
            ? QWEN3VL_MULTIVIEW_OCCURRENCE_POLICY
            : 0;
    }

    SpmFmbLayerProducerYieldCapability
    spm_fmb_layer_producer_yield_capability() const override {
        return multiview_composite_prepared_
            ? SpmFmbLayerProducerYieldCapability::
                  CanonicalDenseReplicatedFp16
            : SpmFmbLayerProducerYieldCapability::Unsealed;
    }

    uint64_t
    spm_fmb_layer_producer_yield_policy_fingerprint() const override {
        return multiview_composite_prepared_
            ? QWEN3VL_MULTIVIEW_YIELD_POLICY
            : 0;
    }

    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        // Graph admission is handled by GraphCache.capture(sig) in the
        // Python vision adapter; this method only describes op emission.
        cfg.num_layers       = num_layers();

        cfg.kv_first_fn = static_cast<void(FusedModelBase::*)(int, const ChunkInfo&)>(
                              &Qwen3VLVisionModel::emit_kv_first_body);

        // Single group: bidirectional vision encoder, same constraint as SigLIP
        // (multi-group REPLAY non-determinism mitigation, see siglip notes).
        cfg.cross_layer_batch_size = num_layers();
        // Fused patch merger: run the merger inside the same capture
        // after the encoder loop, returning [seq/4, oh] from the C++ op. static_config()
        // runs per-forward (run_all_layers step 1) AFTER forward() set current_num_patches_,
        // so post_output_shape reflects this forward's seq.
        if (merger_active()) {
            cfg.post_fn = static_cast<void(FusedModelBase::*)()>(
                multiview_composite_dispatch_
                    ? &Qwen3VLVisionModel::multiview_composite_post_fn
                    : (z1_dispatch_
                           ? &Qwen3VLVisionModel::pooler_z1_post_fn
                           : &Qwen3VLVisionModel::merger_post_fn));
            // NO post_output_shape: the patch merger writes a STABLE pooler_merged slot (fixed DMA,
            // popped via pop_pooler_merged) — the fresh-per-forward post_output_tensor + mutable DMA
            // was the sole warmup_replay non-determinism (the deepstack fixed slots are bit-identical).
            // The forward then returns the (ignored-when-fused) encoder output_tensor_.
        }
        // RhinoVLA fast-replay opt-in (set via qwen3vl_vision_set_fast_replay).
        cfg.fast_replay_skip_layer_loop =
            !pipeline_dispatch_active() && fast_replay_skip_layer_loop_;
        // preload-skip is only safe when the layer loop is also skipped: the skip
        // relies on the fast-replay cursor being set absolutely past the preload
        // region. Without skip_layer_loop, REPLAY still walks the layer body and the
        // un-emitted preload nodes desync the graph cursor. Gate them together.
        cfg.fast_replay_skip_preload =
            !pipeline_dispatch_active() && preload_replay_skip_ &&
            fast_replay_skip_layer_loop_;
        cfg.fast_replay_bake_post_fn =
            !pipeline_dispatch_active() && bake_merger_;
        return cfg;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        TORCH_CHECK(!(merger_active() || image_batch_count_ > 1) || plan.num_chunks == 1,
                    "Qwen3VLVisionModel: fused merger and packed-image minibatch require "
                    "a single chunk, got num_chunks=", plan.num_chunks,
                    " chunk_size=", plan.chunk_size,
                    " patches=", current_num_patches_);
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::KV_FIRST;
        cfg.inter_layer_io = InterLayerIO::DDR_PINGPONG;
        return cfg;
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        int64_t comp_cs = ctx.chunk_size;
        int64_t kv_cs   = ctx.effective_kv_cs();
        int64_t wide_cs = std::max(comp_cs, kv_cs);
        int64_t cs      = comp_cs;  // merger aliases use the compute chunk size
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();

        int64_t local_q_dim = (nq / NUM_CORES) * hd;
        int64_t local_inter = is_ / NUM_CORES;
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        int64_t res    = A(wide_cs * h * DWIDTH);
        int64_t q_kv   = A(kv_cs * local_q_dim * DWIDTH);
        int64_t q_comp = A(comp_cs * local_q_dim * DWIDTH);
        int64_t oproj  = A(comp_cs * h * DWIDTH);
        int64_t fc1    = A(comp_cs * local_inter * DWIDTH);

        SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                            hd, /*nq*/nq, /*nkv*/nq,
                            /*cores*/NUM_CORES, /*mask*/0};
        SdpaTiling t = sdpa_compute_tiling(sdpa_cfg, comp_cs);
        int64_t nkv_per_core = CeilDiv(nq, (int64_t)NUM_CORES);
        int64_t sdpa_tmp = A(t.tile_n_v16 * t.tile_k * nkv_per_core * CeilDiv(comp_cs, t.tile_m) * 32);

        auto dma_safe = [&](int64_t elems) -> int64_t {
            int64_t dma_elems = ((elems + 255) / 256) * 256;
            return A(dma_elems * DWIDTH);
        };
        int64_t norm_w_sz   = dma_safe(h);
        int64_t q_bias_sz   = dma_safe(local_q_dim);
        int64_t fc1_bias_sz = dma_safe(local_inter);
        int64_t full_bias_sz = dma_safe(h);

        int nl = static_cast<int>(num_layers());
        constexpr BufferScope ALL  = BufferScope::LayerWide;
        constexpr BufferScope KVIN = BufferScope::KvInsert;
        constexpr BufferScope COMP = BufferScope::Compute;
        const auto preload_all_persistent = [](
            FusedModelBase& base, int layer_idx, uint32_t) {
            if (layer_idx == 0) {
                static_cast<Qwen3VLVisionModel&>(base)
                    .emit_preload_weights();
            }
        };

        std::vector<BufferDecl> decls = {
            {"residual1",    res,       1, 6, StorageClass::Temp, 0, nullptr, ALL},
            {"input_norm",   res,       1, 6, StorageClass::Temp, 0, nullptr, ALL},
            {"oproj",        oproj,     4, 6, StorageClass::Temp, 0, nullptr, COMP},

            // q/k get rotated in place by rope_2d; v is unchanged. All three
            // share phases 2..3 lifecycle (Phase 2: linear; Phase 2.5: rope;
            // Phase 3: KV insert / SDPA read).
            {"q",            q_kv,      2, 3, StorageClass::Temp, 0, nullptr, KVIN},
            {"k",            q_kv,      2, 3, StorageClass::Temp, 0, nullptr, KVIN},
            {"v",            q_kv,      2, 3, StorageClass::Temp, 0, nullptr, KVIN},
            {"q_comp",       q_comp,    3, 4, StorageClass::Temp, 0, nullptr, COMP},
            {"sdpa_out",     q_comp,    3, 4, StorageClass::Temp, 0, nullptr, COMP},
            {"sdpa_tmp",     sdpa_tmp,  3, 3, StorageClass::Temp, 0, nullptr, COMP},
            {"fc1",          fc1,       5, 6, StorageClass::Temp, 0, nullptr, COMP},

            {"ln1_gamma", norm_w_sz, 0, 0,
             StorageClass::PersistentPerLayer, nl,
             nullptr, ALL, preload_all_persistent},
            {"ln1_beta",     norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL},
            {"ln2_gamma",    norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL},
            {"ln2_beta",     norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL},
            {"q_bias",       q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL},
            {"k_bias",       q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL},
            {"v_bias",       q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL},
            {"fc1_bias",     fc1_bias_sz,  0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL},
            {"o_bias",       full_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL},
            {"fc2_bias",     full_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL},
        };
        // SPM-resident 2D-RoPE cos/sin tables [max_hw, head_dim/4] fp16 (shared across all layers;
        // broadcast once in emit_preload_weights). Tiny. The choice is frozen
        // per handle when set_rope_tables() registers these tables.
        if (rope_spm_enabled_ && max_hw_ > 0) {
            int64_t rope_tbl_sz = A(max_hw_ * (orig_head_dim_ / 4) * DWIDTH);
            decls.push_back({"rope_cos", rope_tbl_sz, 0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"rope_sin", rope_tbl_sz, 0, 0, StorageClass::Persistent, 0, nullptr});
        }
        // Fused merger (patch + 3 deepstack), single- AND batched-image. The merger is
        // row-independent (LayerNorm/GEMM/GELU/all-reduce per-row), so one pass over the packed
        // residual1[seq,h] (seq = current_num_patches_ = g*n_i) is numerically identical to the
        // per-image loop. ALL activation buffers ALIAS dead encoder Temps → the merger adds 0
        // Fixed / 0 Temp peak, AND (critically for batching) NO activation buffer is Persistent:
        // the persistent layout is g- AND cs-INDEPENDENT, so a shared handle can run cs=256 (g=1)
        // and cs=768 (g=3) forwards without tripping the persistent-layout lock (the SPM-overflow
        // + monotonic-cs blocker that the old all-Persistent layout hit). The
        // deepstack snapshot stage `merger_in` (qwen3vl-only, qwen25vl has no
        // deepstack) reuses input_norm — disjoint lifetime from merger_zero (merger_in dies at the
        // LN read, merger_zero is born at the later memset).
        // Lifetimes (post_fn runs strictly AFTER the encoder loop, in emission order):
        //   merger_normed (->oproj):      LN out (read by m0), then REUSED as the m2 PARTIAL.
        //   merger_mid    (->fc1):        m0 col out / GELU.
        //   merger_zero   (->input_norm): all-reduce ZERO residual.
        //   merger_out    (->residual1):  all-reduce out -> SPM->DDR. (patch merger reads residual1
        //                                 in FULL at the LN before all-reduce overwrites it.)
        //   merger_in     (->input_norm): deepstack snapshot DDR->SPM, read by the LN; disjoint zero.
        if (merger_active()) {
            const int64_t mrows    = cs / 4;
            const int64_t local_mh = merge_hidden_ / NUM_CORES;
            const int64_t oh       = merger_out_hidden_;
            const int64_t merger_norm_w_sz = dma_safe(
                deepstack_merger_active() ? merge_hidden_ : h);
            // Each aliased buffer's content must fit its target's allocated size (the alias
            // resolver copies the offset, it does NOT bound the size). h=1024/oh=2048/mh=4096:
            // normed,in = cs*h <= oproj,input_norm (res); mid = mrows*512 <= fc1 (cs*512);
            // zero,out = mrows*oh <= input_norm,residual1 (res).
            TORCH_CHECK(A(cs * h * DWIDTH) <= res && A(mrows * local_mh * DWIDTH) <= fc1
                        && A(mrows * oh * DWIDTH) <= res,
                        "qwen3vl merger Temp-alias size overflow (normed/in vs res, mid vs fc1, "
                        "zero/out vs res)");
            decls.push_back({"merger_normed", A(cs * h * DWIDTH),           0, 0, StorageClass::Temp, 0, "oproj",      COMP});
            decls.push_back({"merger_mid",    A(mrows * local_mh * DWIDTH), 0, 0, StorageClass::Temp, 0, "fc1",        COMP});
            decls.push_back({"merger_zero",   A(mrows * oh * DWIDTH),       0, 0, StorageClass::Temp, 0, "input_norm", ALL});
            decls.push_back({"merger_out",    A(mrows * oh * DWIDTH),       0, 0, StorageClass::Temp, 0, "residual1",  ALL});
            // Multiview taps and pooler consume residual1 in place; only the
            // ordinary fused-merger path reloads a DDR snapshot into merger_in.
            // Omitting that unreachable logical alias also keeps it from
            // overlapping merger_zero in the resolved phase-0 manifest.
            if (!multiview_dry_prepared_ &&
                !multiview_composite_prepared_) {
                decls.push_back({"merger_in", A(cs * h * DWIDTH), 0, 0,
                                 StorageClass::Temp, 0, "input_norm", ALL});
            }
            decls.push_back({"merger_ln_q_w",  merger_norm_w_sz,   0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"merger_ln_q_b",  merger_norm_w_sz,   0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"merger_m0_bias", dma_safe(local_mh), 0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"merger_m2_bias", dma_safe(oh),       0, 0, StorageClass::Persistent, 0, nullptr});
        }
        return decls;
    }

    void emit_preload_weights() {
        int64_t h = hidden_size();
        int64_t local_q_dim = (num_q_heads() / NUM_CORES) * head_dim();
        int64_t local_inter = intermediate_size() / NUM_CORES;

        // Loop 1: 8-core DMAs for all layers (memset + norm + col-partition bias).
        for (int64_t L = 0; L < num_layers(); ++L) {
            auto& bn = layer_bias_norm_[L];

            rpu_launch_memset_spm_multicore(layer_addr(L, 0, "o_bias"), h);
            rpu_launch_memset_spm_multicore(layer_addr(L, 0, "fc2_bias"), h);

            if (pipeline_dispatch_active()) {
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln1_w, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "ln1_gamma"));
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln1_b, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "ln1_beta"));
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln2_w, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "ln2_gamma"));
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln2_b, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "ln2_beta"));

                rpu_launch_ddr_scatter_spm_dma(
                    bn.q_b, /*src_offset_elements=*/0,
                    local_q_dim, local_q_dim * DWIDTH,
                    layer_addr(L, 0, "q_bias"),
                    /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.k_b, /*src_offset_elements=*/0,
                    local_q_dim, local_q_dim * DWIDTH,
                    layer_addr(L, 0, "k_bias"),
                    /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.v_b, /*src_offset_elements=*/0,
                    local_q_dim, local_q_dim * DWIDTH,
                    layer_addr(L, 0, "v_bias"),
                    /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.fc1_b, /*src_offset_elements=*/0,
                    local_inter, local_inter * DWIDTH,
                    layer_addr(L, 0, "fc1_bias"),
                    /*num_cores=*/NUM_CORES);
            } else {
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln1_w.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "ln1_gamma"));
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln1_b.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "ln1_beta"));
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln2_w.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "ln2_gamma"));
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln2_b.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "ln2_beta"));

                rpu_launch_ddr_scatter_spm_dma(
                    bn.q_b.data_ptr<c10::Half>(), local_q_dim,
                    local_q_dim * DWIDTH, layer_addr(L, 0, "q_bias"),
                    /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.k_b.data_ptr<c10::Half>(), local_q_dim,
                    local_q_dim * DWIDTH, layer_addr(L, 0, "k_bias"),
                    /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.v_b.data_ptr<c10::Half>(), local_q_dim,
                    local_q_dim * DWIDTH, layer_addr(L, 0, "v_bias"),
                    /*num_cores=*/NUM_CORES);
                rpu_launch_ddr_scatter_spm_dma(
                    bn.fc1_b.data_ptr<c10::Half>(), local_inter,
                    local_inter * DWIDTH, layer_addr(L, 0, "fc1_bias"),
                    /*num_cores=*/NUM_CORES);
            }
        }

        // Loop 2: 1-core DMAs for all layers (row-partition biases).
        for (int64_t L = 0; L < num_layers(); ++L) {
            auto& bn = layer_bias_norm_[L];
            if (pipeline_dispatch_active()) {
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.o_b, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "o_bias"), /*num_cores=*/1);
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.fc2_b, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "fc2_bias"), /*num_cores=*/1);
            } else {
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.o_b.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "o_bias"), /*num_cores=*/1);
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.fc2_b.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "fc2_bias"), /*num_cores=*/1);
            }
        }

        // Global: broadcast the 2D-RoPE cos/sin tables into SPM once (SPM-rope variant). Flush
        // first — same coherency care the DDR rope launcher takes per-forward for these tensors.
        if (rope_spm_enabled_ && max_hw_ > 0) {
            int64_t tbl_elems = max_hw_ * (orig_head_dim_ / 4);
            rpu_ddr_flush(freq_cos_.data_ptr<c10::Half>());
            rpu_ddr_flush(freq_sin_.data_ptr<c10::Half>());
            if (pipeline_dispatch_active()) {
                rpu_launch_ddr_broadcast_spm_dma(
                    freq_cos_, /*src_offset_elements=*/0, tbl_elems,
                    addr(0, "rope_cos"));
                rpu_launch_ddr_broadcast_spm_dma(
                    freq_sin_, /*src_offset_elements=*/0, tbl_elems,
                    addr(0, "rope_sin"));
            } else {
                rpu_launch_ddr_broadcast_spm_dma(
                    freq_cos_.data_ptr<c10::Half>(), tbl_elems,
                    addr(0, "rope_cos"));
                rpu_launch_ddr_broadcast_spm_dma(
                    freq_sin_.data_ptr<c10::Half>(), tbl_elems,
                    addr(0, "rope_sin"));
            }
        }

        // Merger weights (patch + deepstack) are loaded per-merger INSIDE merger_post_fn (shared
        // SPM weight slots, re-loaded before each merger's compute) — not here — so the per-merger
        // ln_q/biases don't clobber each other and post_fn re-emit on fast-replay stays correct.
    }

    void emit_kv_first_body(int layer_idx, const ChunkInfo& chunk) {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        // Phase 1: DDR→SPM input DMA + LayerNorm1
        emit_layer_input_dma(layer_idx, chunk);
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "ln1_gamma"), layer_addr(layer_idx, 0, "ln1_beta"),
            seq_len, h, eps_, false, 0, NUM_CORES);

        // Phase 2: Q / K / V Linear with bias (SPM→SPM ACC16, col-partition).
        // Python adapter pre-splits the fused [3*dim, dim] HF qkv into three
        // [dim, dim] tensors — we use the SigLIP three-Linear pattern.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, NUM_CORES,
            layer_addr(layer_idx, 0, "q_bias"), false, lw.q_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, NUM_CORES,
            layer_addr(layer_idx, 0, "k_bias"), false, lw.k_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, NUM_CORES,
            layer_addr(layer_idx, 0, "v_bias"), false, lw.v_ws);

        // Phase 2.5: 2D RoPE on Q and K (in-place SPM read/write).
        //
        // Each core holds nq/NUM_CORES = 2 heads worth of `[seq_len, 2, head_dim]`
        // in its slice. The rope_2d kernel takes head_num = local_heads_per_core
        // (because each core processes its own slice independently — same pattern
        // as 1D rope on Q/K of the text decoder).
        const int64_t local_heads = nq / NUM_CORES;
        // `hd` is the physical per-head stride. RoPE rotates only the logical
        // lanes; the controlled 32B pad lanes remain zero.
        const int64_t head_dim_pad = hd;
        if (rope_spm_enabled_) {
            // SPM-resident cos/sin tables (broadcast once in emit_preload_weights).
            const uint32_t cos_a = addr(0, "rope_cos");
            const uint32_t sin_a = addr(0, "rope_sin");
            if (pipeline_dispatch_active()) {
                rpu_launch_rope_2d_spm_kernel(
                    addr(0, "q"), addr(0, "q"), cos_a, sin_a,
                    position_idx_keepalive_,
                    /*pos_offset=*/chunk.offset, seq_len,
                    local_heads, orig_head_dim_, head_dim_pad, NUM_CORES);
                rpu_launch_rope_2d_spm_kernel(
                    addr(0, "k"), addr(0, "k"), cos_a, sin_a,
                    position_idx_keepalive_,
                    /*pos_offset=*/chunk.offset, seq_len,
                    local_heads, orig_head_dim_, head_dim_pad, NUM_CORES);
            } else {
                rpu_launch_rope_2d_spm_kernel(
                    addr(0, "q"), addr(0, "q"), cos_a, sin_a,
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    /*pos_offset=*/chunk.offset, seq_len,
                    local_heads, orig_head_dim_, head_dim_pad, NUM_CORES);
                rpu_launch_rope_2d_spm_kernel(
                    addr(0, "k"), addr(0, "k"), cos_a, sin_a,
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    /*pos_offset=*/chunk.offset, seq_len,
                    local_heads, orig_head_dim_, head_dim_pad, NUM_CORES);
            }
        } else {
            rpu_launch_rope_2d_ddr_kernel(
                addr(0, "q"), addr(0, "q"),
                freq_cos_.data_ptr<c10::Half>(),
                freq_sin_.data_ptr<c10::Half>(),
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset,
                seq_len,
                local_heads, orig_head_dim_, head_dim_pad, NUM_CORES);
            rpu_launch_rope_2d_ddr_kernel(
                addr(0, "k"), addr(0, "k"),
                freq_cos_.data_ptr<c10::Half>(),
                freq_sin_.data_ptr<c10::Half>(),
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset,
                seq_len,
                local_heads, orig_head_dim_, head_dim_pad, NUM_CORES);
        }

        // Phase 3: KV cache insert (position=0 always for vision) + bidir SDPA.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        rpu_launch_insert_kcache_spm_unified(
            k_cache, chunk.offset, addr_offset("k").value,
            seq_len, nq, hd, NUM_CORES);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, chunk.offset, addr_offset("v").value,
            seq_len, nq, hd, NUM_CORES);

        TORCH_CHECK(q_ddr_slots_.count({current_num_patches_, local_heads * hd}) == 1,
                    "Qwen3VLVisionModel: Q staging slot not allocated in KV_FIRST mode");
        const at::Tensor& q_slot = q_ddr_slot();
        const int64_t q_row_stride = local_heads * hd;
        const int64_t q_local_elems = seq_len * q_row_stride;
        const int64_t q_core_stride_bytes = q_slot.size(1) * q_row_stride * DWIDTH;
        if (pipeline_dispatch_active()) {
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "q"), q_slot,
                /*dst_offset_elements=*/chunk.offset * q_row_stride,
                q_local_elems, q_core_stride_bytes,
                /*num_cores=*/NUM_CORES);
        } else {
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "q"),
                q_slot.data_ptr<c10::Half>() + chunk.offset * q_row_stride,
                q_local_elems, q_core_stride_bytes,
                /*num_cores=*/NUM_CORES);
        }
        return;
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();
        int64_t local_inter = is_ / NUM_CORES;
        const int64_t local_heads = nq / NUM_CORES;

        emit_layer_input_dma(layer_idx, chunk);

        TORCH_CHECK(q_ddr_slots_.count({current_num_patches_, local_heads * hd}) == 1,
                    "Qwen3VLVisionModel: Q staging slot not allocated in KV_FIRST mode");
        const at::Tensor& q_slot = q_ddr_slot();
        const int64_t q_row_stride = local_heads * hd;
        const int64_t q_local_elems = seq_len * q_row_stride;
        const int64_t q_core_stride_bytes = q_slot.size(1) * q_row_stride * DWIDTH;
        if (pipeline_dispatch_active()) {
            rpu_launch_ddr_scatter_spm_dma(
                q_slot, /*src_offset_elements=*/chunk.offset * q_row_stride,
                q_local_elems, q_core_stride_bytes,
                addr(0, "q_comp"),
                /*num_cores=*/NUM_CORES);
        } else {
            rpu_launch_ddr_scatter_spm_dma(
                q_slot.data_ptr<c10::Half>() + chunk.offset * q_row_stride,
                q_local_elems, q_core_stride_bytes,
                addr(0, "q_comp"),
                /*num_cores=*/NUM_CORES);
        }

        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];

        double attn_scale = 1.0 / std::sqrt(static_cast<double>(orig_head_dim_));
        if (image_batch_count_ > 1) {
            TORCH_CHECK(chunk.offset == 0 && seq_len == current_num_patches_,
                        "Qwen3VLVisionModel: packed-image minibatch SDPA requires a "
                        "single chunk, got offset=", chunk.offset, " len=", seq_len,
                        " total=", current_num_patches_);
            // Multi-image BATCH: `image_batch_count_` equal-size images packed into
            // seq_len. Per-image attention isolation via the minibatch kernel — image i
            // attends ONLY its own per_image_ctx patches (cache rows [i*ctx,(i+1)*ctx)).
            // Q/K/V were inserted contiguously at pos 0 above. gr00t vision is MASK_NONE
            // per-image (= SigLIP-like), so no per-image mask. Requires single chunk
            // (seq_len == N*per_image_ctx) — the launcher asserts it. Mirrors the SigLIP
            // SigLIP-batch path (see rpu_siglip_model.cpp).
            int64_t per_image_ctx = current_num_patches_ / image_batch_count_;
            rpu_launch_sdpa_spm_minibatch_kernel(
                k_cache, v_cache,
                0 /*MASK_NONE*/, attn_scale,
                addr_offset("q_comp").value,
                addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value, 0,
                seq_len, nq, nq, hd,
                per_image_ctx, image_batch_count_,
                NUM_CORES, NUM_CORES);
        } else {
            rpu_launch_sdpa_spm_unified_kernel_v2(
                k_cache, v_cache,
                0 /*MASK_NONE*/, attn_scale,
                addr_offset("q_comp").value,
                addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value, 0,
                seq_len, nq, nq, hd,
                current_num_patches_, NUM_CORES, NUM_CORES);
        }

        // Phase 4: O_proj (row-partition, with bias) + AllReduce + Residual.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, NUM_CORES,
            layer_addr(layer_idx, 0, "o_bias"), false, lw.o_ws);
        const bool force_block_twostage =
            multiview_composite_dispatch_ &&
            multiview_force_block_twostage_;
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"),
            addr(0, "input_norm"), seq_len, h,
            NUM_CORES, NUM_CORES,
            /*force_twostage=*/force_block_twostage);

        // Phase 5: LayerNorm2 + fc1+bias + GELU (gelu_pytorch_tanh per Q3VL config).
        rpu_launch_layernorm_spm_kernel(
            addr(0, "input_norm"), addr(0, "oproj"),
            layer_addr(layer_idx, 0, "ln2_gamma"), layer_addr(layer_idx, 0, "ln2_beta"),
            seq_len, h, eps_, false, 0, NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "oproj"), lw.fc1_w, addr(0, "fc1"),
            seq_len, is_, h, 1, NUM_CORES,
            layer_addr(layer_idx, 0, "fc1_bias"), false, lw.fc1_ws);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "fc1"), addr(0, "fc1"),
            seq_len * local_inter, ValuOpType::ADD,
            GeluMode::TANH, NUM_CORES);

        // Phase 6: fc2 + AllReduce + Residual (writes back to residual1 in
        // SPM_RESIDENT mode so the next layer's Phase 1 LN1 reads directly).
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "fc1"), lw.fc2_w, addr(0, "oproj"),
            seq_len, h, is_, 0, NUM_CORES,
            layer_addr(layer_idx, 0, "fc2_bias"), false, lw.fc2_ws);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "input_norm"),
            addr(0, "residual1"), seq_len, h,
            NUM_CORES, NUM_CORES,
            /*force_twostage=*/force_block_twostage);

        // Composite multiview handoff: each N256 occurrence emits three typed
        // terminal yields at HF Vision taps 5/11/17.  The yield scope covers
        // only the existing terminal all-reduce; merger prefix kernels remain
        // in the owning LayerBody callback window.
        if (multiview_composite_dispatch_) {
            const auto tap_it = std::find(
                QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.begin(),
                QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.end(),
                static_cast<int64_t>(layer_idx));
            if (tap_it != QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.end()) {
                const size_t ordinal = static_cast<size_t>(
                    tap_it - QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.begin());
                TORCH_CHECK(
                    chunk.idx == 0 && chunk.offset == 0 &&
                        chunk.len == QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                        seq_len == QWEN3VL_POOLER_Z1_NUM_PATCHES,
                    "Qwen3VLVisionModel multiview composite tap ", layer_idx,
                    " requires one full N256 Vision chunk");
                emit_one_merger_yield(
                    addr(0, "residual1"),
                    ds_merger_ln_q_w_.at(ordinal),
                    ds_merger_ln_q_b_.at(ordinal),
                    ds_merger_m0_w_.at(ordinal),
                    ds_merger_m0_b_.at(ordinal),
                    ds_merger_m2_w_.at(ordinal),
                    ds_merger_m2_b_.at(ordinal),
                    /*postshuffle_norm=*/true);
            }
        }

        // Retained DeepStack physical handoff: HF Vision taps 5/11/17 are three
        // distinct post-block values. In the exact one-chunk Z1 mode, run each
        // requested merger immediately while oproj/fc1/input_norm are dead and
        // write its replicated [64,2048] result directly to its own retained
        // typed port. residual1 remains the next Vision block's input; no raw
        // [256,1024] snapshot is stored or reloaded through DDR. Count 1 admits
        // only ordinal 0/tap 5 and therefore preserves the DS1 graph topology.
        if (z1_dispatch_ && z1_retained_deepstack_count_ != 0) {
            const auto retained_end =
                QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.begin() +
                z1_retained_deepstack_count_;
            const auto tap_it = std::find(
                QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.begin(), retained_end,
                static_cast<int64_t>(layer_idx));
            if (tap_it != retained_end) {
                const size_t ordinal = static_cast<size_t>(
                    tap_it - QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.begin());
                TORCH_CHECK(
                    ordinal < static_cast<size_t>(
                                  z1_retained_deepstack_bound_count_) &&
                        z1_retained_deepstack_source_addrs_[ordinal] != 0 &&
                        chunk.idx == 0 && chunk.offset == 0 &&
                        chunk.len == QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                        seq_len == QWEN3VL_POOLER_Z1_NUM_PATCHES,
                    "Qwen3VLVisionModel pooler Z1 retained DeepStack ordinal ",
                    ordinal, " requires one bound full-sequence Vision tap ",
                    layer_idx, " chunk");
                emit_one_merger(
                    addr(0, "residual1"),
                    z1_retained_deepstack_source_addrs_[ordinal],
                    ds_merger_ln_q_w_.at(ordinal),
                    ds_merger_ln_q_b_.at(ordinal),
                    ds_merger_m0_w_.at(ordinal),
                    ds_merger_m0_b_.at(ordinal),
                    ds_merger_m2_w_.at(ordinal),
                    ds_merger_m2_b_.at(ordinal),
                    /*postshuffle_norm=*/true);
            }
        }

        // DeepStack snapshot — DMA `residual1` (post-block hidden
        // state) to DDR so the Python adapter can run its merger after
        // forward(). Slots are pre-allocated in forward() entry to avoid
        // `at::empty` competing with the active graph's queue-buffer reserve.
        if (!pipeline_dispatch_active() &&
            std::find(deepstack_visual_indexes_.begin(),
                      deepstack_visual_indexes_.end(),
                      static_cast<int64_t>(layer_idx)) !=
                deepstack_visual_indexes_.end())
        {
            auto key = std::make_pair(static_cast<int64_t>(layer_idx), current_num_patches_);
            at::Tensor* slot = deepstack_ddr_slots_.find(key);
            TORCH_CHECK(slot != nullptr && slot->defined(),
                        "qwen3vl_vision_build_layer_subgraph: missing pre-allocated "
                        "DeepStack slot for layer_idx=", layer_idx,
                        " num_patches=", current_num_patches_,
                        " — forward() should have allocated it");
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "residual1"),
                slot->data_ptr<c10::Half>() + chunk.offset * h,
                seq_len * h);
        }

        const bool pipeline_final =
            ((z1_dispatch_ && z1_bound_) ||
             multiview_composite_dispatch_) &&
            layer_idx == num_layers() - 1 && !ctx().output_to_spm;
        if (pipeline_final) {
            TORCH_CHECK(chunk.idx == 0 && chunk.offset == 0 &&
                            chunk.len == QWEN3VL_POOLER_Z1_NUM_PATCHES,
                        "Qwen3VLVisionModel pipeline requires one full-sequence "
                        "final chunk");
        }

        // Ordinary final output still lands in its stable DDR registry slot.
        // Z1 leaves the final [256,1024] hidden in residual1 for the immediately
        // following pooler-only post hook, eliminating that bridge store.
        if (!ctx().output_to_spm && !pipeline_final) {
            emit_layer_output_dma(layer_idx, chunk, "residual1");
        }
    }

    // Fused merger active iff the env flag is on AND merger weights are registered.
    // (qwen3vl processes one image per forward, so no batched gate is needed.)
    bool merger_active() const {
        return fused_merger_enabled_ && merger_out_hidden_ > 0;
    }
    bool pipeline_dispatch_active() const {
        return z1_dispatch_ || multiview_composite_dispatch_;
    }
    // Deepstack mergers are folded too (weights registered via set_deepstack_merger_weights).
    bool deepstack_merger_active() const {
        return merger_active() && !ds_merger_m0_w_.empty();
    }

    // ========================================================================
    // merger_post_fn — runs once AFTER the 24-block encoder, INSIDE the same
    // GraphCache capture through post_fn. Computes the Qwen3-VL patch merger:
    //   merged = m2( gelu( m0( layernorm_lnq(residual1[seq,h]).reshape(seq/4, mh) ) ) )
    // output [seq/4, OUT_HID] → post_output_tensor_ (window order; Python applies
    // the visual-embed placement). m0 = col GEMM, m2 = row GEMM + all-reduce
    // (ViT SwiGLU dataflow), with LayerNorm rather than RMSNorm.
    // ========================================================================
    // One merger: re-load its ln_q/biases into the shared SPM weight slots, then
    // LN(in_addr) → m0 col GEMM → GELU → m2 row GEMM → all-reduce, leaving [mrows, oh]
    // in "merger_out". in_addr = the full [seq,h]-per-core SPM input (residual1 for the
    // patch merger; "merger_in" with the snapshot DMA'd in for a deepstack merger). The
    // patch merger normalizes [seq,h] before reshaping; DeepStack mergers set
    // postshuffle_norm and normalize the reinterpreted [mrows,mh], matching HF.
    // per-merger weight re-load (vs emit_preload) keeps the shared slots correct across the
    // 4 sequential mergers AND deterministic under fast-replay post_fn re-emit.
    void emit_one_merger_prefix(
                         uint32_t in_addr,
                         const at::Tensor& ln_q_w, const at::Tensor& ln_q_b,
                         const at::Tensor& m0_w, const at::Tensor& m0_b,
                         const at::Tensor& m2_w, const at::Tensor& m2_b,
                         bool postshuffle_norm = false) {
        const int64_t seq = current_num_patches_, h = hidden_size();
        const int64_t mh = merge_hidden_, oh = merger_out_hidden_;
        const int64_t mrows = seq / 4, local_mh = mh / NUM_CORES;
        const int64_t norm_rows = postshuffle_norm ? mrows : seq;
        const int64_t norm_hidden = postshuffle_norm ? mh : h;
        // weights → shared SPM slots (ln_q γ/β broadcast; m0 bias col-scatter; m2 bias row).
        if (pipeline_dispatch_active()) {
            rpu_launch_ddr_broadcast_spm_dma(
                ln_q_w, /*src_offset_elements=*/0, norm_hidden,
                addr(0, "merger_ln_q_w"));
            rpu_launch_ddr_broadcast_spm_dma(
                ln_q_b, /*src_offset_elements=*/0, norm_hidden,
                addr(0, "merger_ln_q_b"));
            rpu_launch_ddr_scatter_spm_dma(
                m0_b, /*src_offset_elements=*/0, local_mh,
                local_mh * DWIDTH, addr(0, "merger_m0_bias"), NUM_CORES);
        } else {
            rpu_launch_ddr_broadcast_spm_dma(
                ln_q_w.data_ptr<c10::Half>(), norm_hidden,
                addr(0, "merger_ln_q_w"));
            rpu_launch_ddr_broadcast_spm_dma(
                ln_q_b.data_ptr<c10::Half>(), norm_hidden,
                addr(0, "merger_ln_q_b"));
            rpu_launch_ddr_scatter_spm_dma(
                m0_b.data_ptr<c10::Half>(), local_mh,
                local_mh * DWIDTH, addr(0, "merger_m0_bias"), NUM_CORES);
        }
        rpu_launch_memset_spm_multicore(addr(0, "merger_m2_bias"), oh);
        if (pipeline_dispatch_active()) {
            rpu_launch_ddr_broadcast_spm_dma(
                m2_b, /*src_offset_elements=*/0, oh,
                addr(0, "merger_m2_bias"), /*num_cores=*/1);
        } else {
            rpu_launch_ddr_broadcast_spm_dma(
                m2_b.data_ptr<c10::Half>(), oh,
                addr(0, "merger_m2_bias"), /*num_cores=*/1);
        }
        // LN + reshape order is selected above; both paths present [mrows,mh]
        // to m0, then run GELU → m2 row → all-reduce.
        rpu_launch_layernorm_spm_kernel(in_addr, addr(0, "merger_normed"),
            addr(0, "merger_ln_q_w"), addr(0, "merger_ln_q_b"),
            norm_rows, norm_hidden, merger_ln_eps_, false, 0, NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0, "merger_normed"), m0_w, addr(0, "merger_mid"),
            mrows, mh, mh, /*col*/1, NUM_CORES, addr(0, "merger_m0_bias"), false, at::Tensor{});
        rpu_launch_eltwise_unary_spm_kernel(addr(0, "merger_mid"), addr(0, "merger_mid"),
            mrows * local_mh, ValuOpType::ADD,
            GeluMode::ERF, NUM_CORES);
        // m2 row GEMM writes its PARTIAL back into merger_normed (the LN-out buffer, dead once m0
        // consumed it) → no separate merger_partial slot (so it can Temp-alias oproj; mirrors
        // qwen25vl). all-reduce operands stay distinct: partial=merger_normed(oproj),
        // zero=merger_zero(input_norm), out=merger_out(residual1).
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0, "merger_mid"), m2_w, addr(0, "merger_normed"),
            mrows, oh, mh, /*row*/0, NUM_CORES, addr(0, "merger_m2_bias"), false, at::Tensor{});
        rpu_launch_memset_spm_multicore(addr(0, "merger_zero"), mrows * oh);
    }

    void emit_one_merger(uint32_t in_addr, uint32_t final_out_addr,
                         const at::Tensor& ln_q_w, const at::Tensor& ln_q_b,
                         const at::Tensor& m0_w, const at::Tensor& m0_b,
                         const at::Tensor& m2_w, const at::Tensor& m2_b,
                         bool postshuffle_norm = false) {
        TORCH_CHECK(final_out_addr != 0,
                    "Qwen3VLVisionModel merger final output address is unset");
        emit_one_merger_prefix(
            in_addr, ln_q_w, ln_q_b, m0_w, m0_b, m2_w, m2_b,
            postshuffle_norm);
        const int64_t mrows = current_num_patches_ / 4;
        rpu_launch_all_reduce_sum_residual_kernel(addr(0, "merger_normed"), addr(0, "merger_zero"),
            final_out_addr, mrows, merger_out_hidden_, NUM_CORES, NUM_CORES);
    }

    SpmDense2DSpec multiview_yield_spec() const {
        SpmDense2DSpec spec;
        spec.dtype = SpmPortDType::Fp16;
        spec.rows = QWEN3VL_POOLER_Z1_MERGED_ROWS;
        spec.cols = QWEN3VL_MULTIVIEW_TEXT_HIDDEN;
        spec.distribution = SpmPortDistribution::Replicated;
        spec.validate();
        return spec;
    }

    void emit_one_merger_yield(
                         uint32_t in_addr,
                         const at::Tensor& ln_q_w, const at::Tensor& ln_q_b,
                         const at::Tensor& m0_w, const at::Tensor& m0_b,
                         const at::Tensor& m2_w, const at::Tensor& m2_b,
                         bool postshuffle_norm = false) {
        TORCH_CHECK(
            multiview_composite_dispatch_ &&
                merger_out_hidden_ == QWEN3VL_MULTIVIEW_TEXT_HIDDEN,
            "Qwen3VLVisionModel typed merger yield requires the exact "
            "multiview composite dispatch");
        emit_one_merger_prefix(
            in_addr, ln_q_w, ln_q_b, m0_w, m0_b, m2_w, m2_b,
            postshuffle_norm);
        auto yield = begin_spm_pipeline_dense_producer_yield(
            "merger_out", /*allocation_layer=*/-1,
            multiview_yield_spec());
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "merger_normed"), addr(0, "merger_zero"),
            yield.target(), QWEN3VL_POOLER_Z1_MERGED_ROWS,
            QWEN3VL_MULTIVIEW_TEXT_HIDDEN, NUM_CORES, NUM_CORES);
        yield.finish();
    }

    void merger_post_fn() {
        const int64_t seq = current_num_patches_, h = hidden_size(), oh = merger_out_hidden_;
        const int64_t mrows = seq / 4;
        TORCH_CHECK(!pipeline_dispatch_active(),
                    "Qwen3VLVisionModel ordinary merger cannot run during "
                    "a pipeline dispatch");

        // Patch merger: input = residual1 (SPM-resident encoder output). Output → the STABLE
        // pooler_merged slot via FIXED DMA (deterministic; popped by pop_pooler_merged). This
        // replaces the fresh-per-forward post_output_tensor + mutable DMA, which was the entire
        // warmup_replay non-determinism (the deepstack fixed slots were already bit-identical).
        emit_one_merger(addr(0, "residual1"), addr(0, "merger_out"),
            merger_ln_q_w_, merger_ln_q_b_, merger_m0_w_, merger_m0_b_, merger_m2_w_, merger_m2_b_);
        if (prefix_scatter_active_) {
            scatter_merger_out_to_prefix(oh, prefix_pooler_ptr_);
        } else {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "merger_out"), pooler_merged_ddr_slots_.at(seq).data_ptr<c10::Half>(), mrows * oh);
        }

        // Deepstack mergers each read their layer snapshot (DMA DDR→"merger_in") and
        // writes a STABLE merged slot (fixed DMA → deterministic, no post_output drift). Replaces
        // the eager CPU-LN + RPU-GEMM deepstack mergers (kills 3 CPU LNs + 6 eager GEMMs + bounces).
        if (deepstack_merger_active()) {
            for (size_t k = 0; k < deepstack_visual_indexes_.size(); ++k) {
                auto key = std::make_pair(deepstack_visual_indexes_[k], seq);
                rpu_launch_ddr_broadcast_spm_dma(
                    deepstack_ddr_slots_.at(key).data_ptr<c10::Half>(), seq * h, addr(0, "merger_in"));
                emit_one_merger(addr(0, "merger_in"), addr(0, "merger_out"),
                    ds_merger_ln_q_w_[k], ds_merger_ln_q_b_[k], ds_merger_m0_w_[k],
                    ds_merger_m0_b_[k], ds_merger_m2_w_[k], ds_merger_m2_b_[k],
                    /*postshuffle_norm=*/true);
                if (prefix_scatter_active_) {
                    scatter_merger_out_to_prefix(oh, prefix_dense_ptrs_.at(k));
                } else {
                    rpu_launch_spm_copy_ddr_dma(
                        addr(0, "merger_out"), deepstack_merged_ddr_slots_.at(key).data_ptr<c10::Half>(),
                        mrows * oh);
                }
            }
        }
    }

    void multiview_composite_post_fn() {
        TORCH_CHECK(
            multiview_composite_dispatch_ &&
                multiview_composite_prepared_ &&
                current_num_patches_ ==
                    QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                image_batch_count_ == 1,
            "Qwen3VLVisionModel multiview composite post hook requires the "
            "exact active N256 occurrence");
        emit_one_merger_yield(
            addr(0, "residual1"),
            merger_ln_q_w_, merger_ln_q_b_, merger_m0_w_, merger_m0_b_,
            merger_m2_w_, merger_m2_b_);
    }

    void pooler_z1_post_fn() {
        validate_pooler_z1_contract();
        TORCH_CHECK(z1_dispatch_ && z1_adopted_ && z1_bound_ &&
                        z1_source_addr_ != 0 &&
                        current_num_patches_ ==
                            QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                        image_batch_count_ == 1,
                    "Qwen3VLVisionModel pooler Z1 post hook requires the exact "
                    "active 256-patch source binding");

        // This post hook always emits exactly the patch merger. Pooler-only mode
        // suppresses DeepStack entirely; retained mode already emitted the
        // requested ordinals at Vision blocks 5/11/17. There is no snapshot
        // reload or SPM-to-DDR bridge here.
        emit_one_merger(
            addr(0, "residual1"), z1_source_addr_,
            merger_ln_q_w_, merger_ln_q_b_, merger_m0_w_, merger_m0_b_,
            merger_m2_w_, merger_m2_b_);
    }

private:
    bool retained_deepstack_addresses_empty() const {
        return std::all_of(
            z1_retained_deepstack_source_addrs_.begin(),
            z1_retained_deepstack_source_addrs_.end(),
            [](uint32_t addr) { return addr == 0; });
    }

    bool retained_deepstack_bindings_complete() const {
        if (z1_retained_deepstack_bound_count_ !=
            z1_retained_deepstack_count_) {
            return false;
        }
        for (int64_t ordinal = 0;
             ordinal < static_cast<int64_t>(
                           z1_retained_deepstack_source_addrs_.size());
             ++ordinal) {
            const bool requested = ordinal < z1_retained_deepstack_count_;
            if (requested !=
                (z1_retained_deepstack_source_addrs_[ordinal] != 0)) {
                return false;
            }
        }
        return true;
    }

    void validate_multiview_composite_profile() const {
        TORCH_CHECK(
            num_layers() == 24 && num_q_heads() == 16 &&
                head_dim() == 64 && hidden_size() == 1024 &&
                intermediate_size() == 4096 && max_hw_ == 48,
            "Qwen3VLVisionModel multiview composite admits only the exact "
            "N256/L24/H1024/I4096/Q16/D64/max_hw48 profile");
        TORCH_CHECK(
            deepstack_visual_indexes_ ==
                std::vector<int64_t>(
                    QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.begin(),
                    QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.end()),
            "Qwen3VLVisionModel multiview composite requires Vision taps "
            "[5,11,17]");
        validate_pooler_z1_profile(
            QWEN3VL_MULTIVIEW_TEXT_HIDDEN,
            /*require_component_fast_flags_false=*/false,
            /*allow_w8a16=*/true);
        validate_pooler_z1_retained_deepstack_profile(/*retained_count=*/3);
    }

    void validate_pooler_z1_profile() const {
        (void)validate_pooler_z1_profile(
            QWEN3VL_POOLER_Z1_TEXT_HIDDEN,
            /*require_component_fast_flags_false=*/true,
            /*allow_w8a16=*/false);
    }

    MultiviewLinearProfile classify_multiview_linear_profile() const {
        TORCH_CHECK(
            layer_weights_.size() == 24,
            "Qwen3VLVisionModel multiview linear profile requires the exact "
            "24-layer weight inventory");
        const int64_t h = hidden_size();
        const int64_t attn = num_q_heads() * head_dim();
        const int64_t inter = intermediate_size();

        auto classify_role = [](const at::Tensor& weight,
                                const at::Tensor& scale,
                                int64_t out_rows,
                                int64_t in_cols,
                                const char* name,
                                int64_t layer) {
            TORCH_CHECK(
                weight.defined() &&
                    weight.device().type() == at::kPrivateUse1 &&
                    weight.layout() == c10::Layout::Strided &&
                    weight.is_contiguous() && weight.dim() == 2 &&
                    weight.size(0) == out_rows &&
                    weight.size(1) == in_cols &&
                    weight.numel() == out_rows * in_cols,
                "Qwen3VLVisionModel multiview ", name,
                " must be a contiguous 2D RPU [", out_rows, ",",
                in_cols, "] weight at layer ", layer);
            if (weight.scalar_type() == at::kHalf) {
                TORCH_CHECK(
                    !scale.defined(),
                    "Qwen3VLVisionModel multiview dense FP16 ", name,
                    " rejects a bound quantization scale at layer ", layer);
                return MultiviewLinearProfile::DenseFp16;
            }
            TORCH_CHECK(
                weight.scalar_type() == at::kChar,
                "Qwen3VLVisionModel multiview ", name,
                " must use FP16 or signed-int8 kChar storage; kByte/W4 and "
                "other dtypes are rejected at layer ", layer);
            TORCH_CHECK(
                scale.defined() && scale.scalar_type() == at::kHalf &&
                    scale.device().type() == at::kPrivateUse1 &&
                    scale.layout() == c10::Layout::Strided &&
                    scale.is_contiguous() && scale.dim() == 1 &&
                    scale.numel() == out_rows,
                "Qwen3VLVisionModel multiview ", name,
                " W8 scale must be contiguous per-channel FP16 RPU [",
                out_rows, "] at layer ", layer);
            return MultiviewLinearProfile::W8A16;
        };

        const MultiviewLinearProfile expected = classify_role(
            layer_weights_.front().q_w,
            layer_weights_.front().q_ws,
            attn, h, "q_w", /*layer=*/0);
        auto require_same = [&](const at::Tensor& weight,
                                const at::Tensor& scale,
                                int64_t out_rows,
                                int64_t in_cols,
                                const char* name,
                                int64_t layer) {
            const MultiviewLinearProfile actual = classify_role(
                weight, scale, out_rows, in_cols, name, layer);
            TORCH_CHECK(
                actual == expected,
                "Qwen3VLVisionModel multiview rejects mixed/partial linear "
                "profiles: ", name, " at layer ", layer,
                " does not match layer-0 q_w");
        };
        for (size_t layer = 0; layer < layer_weights_.size(); ++layer) {
            const auto& weights = layer_weights_[layer];
            const int64_t layer_id = static_cast<int64_t>(layer);
            require_same(
                weights.q_w, weights.q_ws, attn, h,
                "q_w", layer_id);
            require_same(
                weights.k_w, weights.k_ws, attn, h,
                "k_w", layer_id);
            require_same(
                weights.v_w, weights.v_ws, attn, h,
                "v_w", layer_id);
            require_same(
                weights.o_w, weights.o_ws, h, attn,
                "o_w", layer_id);
            require_same(
                weights.fc1_w, weights.fc1_ws, inter, h,
                "fc1_w", layer_id);
            require_same(
                weights.fc2_w, weights.fc2_ws, h, inter,
                "fc2_w", layer_id);
        }
        return expected;
    }

    MultiviewLinearProfile validate_pooler_z1_profile(
        int64_t expected_text_hidden,
        bool require_component_fast_flags_false,
        bool allow_w8a16) const {
        TORCH_CHECK(num_layers() == 24 && num_q_heads() == 16 &&
                        head_dim() == 64 &&
                        hidden_size() == QWEN3VL_POOLER_Z1_VISION_HIDDEN &&
                        intermediate_size() == 4096,
                    "Qwen3VLVisionModel pooler Z1 requires the exact 24-layer "
                    "Qwen3-VL-2B Vision profile");
        TORCH_CHECK(has_rope_ && rope_spm_enabled_,
                    "Qwen3VLVisionModel pooler Z1 requires SPM 2D RoPE");
        TORCH_CHECK(merger_active() &&
                        merge_hidden_ == QWEN3VL_POOLER_Z1_MERGE_HIDDEN &&
                        merger_out_hidden_ == expected_text_hidden,
                    "Qwen3VLVisionModel pooler Z1 requires the fused "
                    "[1024 -> ", expected_text_hidden, "] patch merger");
        TORCH_CHECK(
            (!require_component_fast_flags_false ||
             (!fast_replay_skip_layer_loop_ && !preload_replay_skip_ &&
              !bake_merger_)) && !prefix_scatter_active_,
            "Qwen3VLVisionModel pipeline rejects active component fast replay "
            "for the pooler Z1 profile and always rejects prefix scatter");
        TORCH_CHECK(!get_debug_export(),
                    "Qwen3VLVisionModel pooler Z1 rejects debug export");

        auto require_fp16_owner = [](const at::Tensor& tensor,
                                     const char* name,
                                     int64_t layer,
                                     int64_t min_numel) {
            TORCH_CHECK(
                tensor.defined() && tensor.scalar_type() == at::kHalf &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.layout() == c10::Layout::Strided &&
                    tensor.is_contiguous() && tensor.numel() >= min_numel,
                "Qwen3VLVisionModel pooler Z1 requires contiguous FP16 RPU ",
                name, " with at least ", min_numel, " elements at layer ",
                layer);
        };
        auto require_fp16_vector = [&require_fp16_owner](
                                         const at::Tensor& tensor,
                                         const char* name,
                                         int64_t layer,
                                         int64_t numel) {
            require_fp16_owner(tensor, name, layer, numel);
            TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == numel,
                        "Qwen3VLVisionModel pooler Z1 requires ", name,
                        " to be a ", numel, "-element vector at layer ", layer);
        };
        auto require_fp16_matrix_min = [&require_fp16_owner](
                                            const at::Tensor& tensor,
                                            const char* name,
                                            int64_t layer,
                                            int64_t min_numel) {
            require_fp16_owner(tensor, name, layer, min_numel);
            // Swizzling is in-place and the launcher consumes N*K elements, but
            // the physical matrix metadata is not part of this Z1 contract.
            TORCH_CHECK(tensor.dim() == 2,
                        "Qwen3VLVisionModel pooler Z1 requires 2D ", name,
                        " at layer ", layer);
        };
        TORCH_CHECK(layer_weights_.size() == 24 &&
                        layer_bias_norm_.size() == 24,
                    "Qwen3VLVisionModel pooler Z1 weight inventory drifted");
        const MultiviewLinearProfile linear_profile =
            classify_multiview_linear_profile();
        TORCH_CHECK(
            allow_w8a16 ||
                linear_profile == MultiviewLinearProfile::DenseFp16,
            "Qwen3VLVisionModel standalone pooler Z1 rejects W8A16; only "
            "the typed multiview composite may admit it");
        const int64_t h = hidden_size();
        const int64_t attn = num_q_heads() * head_dim();
        const int64_t inter = intermediate_size();
        for (size_t layer = 0; layer < layer_weights_.size(); ++layer) {
            const auto& bias_norm = layer_bias_norm_[layer];
            const int64_t layer_id = static_cast<int64_t>(layer);
            require_fp16_vector(bias_norm.ln1_w, "ln1_w", layer_id, h);
            require_fp16_vector(bias_norm.ln1_b, "ln1_b", layer_id, h);
            require_fp16_vector(bias_norm.ln2_w, "ln2_w", layer_id, h);
            require_fp16_vector(bias_norm.ln2_b, "ln2_b", layer_id, h);
            require_fp16_vector(bias_norm.q_b, "q_b", layer_id, attn);
            require_fp16_vector(bias_norm.k_b, "k_b", layer_id, attn);
            require_fp16_vector(bias_norm.v_b, "v_b", layer_id, attn);
            require_fp16_vector(bias_norm.o_b, "o_b", layer_id, h);
            require_fp16_vector(bias_norm.fc1_b, "fc1_b", layer_id, inter);
            require_fp16_vector(bias_norm.fc2_b, "fc2_b", layer_id, h);
        }

        const int64_t rope_cols = orig_head_dim_ / 4;
        require_fp16_owner(
            freq_cos_, "freq_cos", /*layer=*/-1, max_hw_ * rope_cols);
        require_fp16_owner(
            freq_sin_, "freq_sin", /*layer=*/-1, max_hw_ * rope_cols);
        TORCH_CHECK(max_hw_ > 0 && freq_cos_.dim() == 2 &&
                        freq_sin_.sizes() == freq_cos_.sizes() &&
                        freq_cos_.size(0) == max_hw_ &&
                        freq_cos_.size(1) == rope_cols &&
                        freq_cos_.numel() == max_hw_ * rope_cols,
                    "Qwen3VLVisionModel pooler Z1 RoPE owners must be exact "
                    "[max_hw,head_dim/4] tables");
        TORCH_CHECK(position_idx_keepalive_.defined() &&
                        position_idx_keepalive_.scalar_type() == at::kShort &&
                        position_idx_keepalive_.device().type() ==
                            at::kPrivateUse1 &&
                        position_idx_keepalive_.layout() ==
                            c10::Layout::Strided &&
                        position_idx_keepalive_.is_contiguous() &&
                        position_idx_keepalive_.dim() == 2 &&
                        position_idx_keepalive_.size(0) >=
                            QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                        position_idx_keepalive_.size(1) == 2,
                    "Qwen3VLVisionModel pooler Z1 position owner must be "
                    "contiguous int16 RPU [>=256,2]");

        const int64_t mh = merge_hidden_;
        const int64_t oh = merger_out_hidden_;
        require_fp16_vector(
            merger_ln_q_w_, "merger_ln_q_w", /*layer=*/-1, h);
        require_fp16_vector(
            merger_ln_q_b_, "merger_ln_q_b", /*layer=*/-1, h);
        require_fp16_matrix_min(
            merger_m0_w_, "merger_m0_w", /*layer=*/-1, mh * mh);
        require_fp16_vector(
            merger_m0_b_, "merger_m0_b", /*layer=*/-1, mh);
        require_fp16_matrix_min(
            merger_m2_w_, "merger_m2_w", /*layer=*/-1, oh * mh);
        require_fp16_vector(
            merger_m2_b_, "merger_m2_b", /*layer=*/-1, oh);
        return linear_profile;
    }

    void validate_pooler_z1_retained_deepstack_profile(
        int64_t retained_count) const {
        TORCH_CHECK(retained_count == 1 || retained_count == 3,
                    "Qwen3VLVisionModel pooler Z1 retained DeepStack profile "
                    "requires count 1 or 3, got ", retained_count);
        TORCH_CHECK(
            deepstack_visual_indexes_.size() == 3 &&
                deepstack_visual_indexes_[0] ==
                    QWEN3VL_POOLER_Z1_DEEPSTACK1_TAP &&
                deepstack_visual_indexes_[1] == 11 &&
                deepstack_visual_indexes_[2] == 17,
            "Qwen3VLVisionModel pooler Z1 retained DeepStack requires the exact "
            "Vision tap order [5,11,17]");
        TORCH_CHECK(
            deepstack_merger_active() && ds_merger_ln_q_w_.size() == 3 &&
                ds_merger_ln_q_b_.size() == 3 &&
                ds_merger_m0_w_.size() == 3 &&
                ds_merger_m0_b_.size() == 3 &&
                ds_merger_m2_w_.size() == 3 &&
                ds_merger_m2_b_.size() == 3,
            "Qwen3VLVisionModel pooler Z1 retained DeepStack requires all three "
            "registered DeepStack merger weight sets");

        auto require_fp16_owner = [](const at::Tensor& tensor,
                                     const char* name,
                                     int64_t ordinal,
                                     int64_t min_numel) {
            TORCH_CHECK(
                tensor.defined() && tensor.scalar_type() == at::kHalf &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.layout() == c10::Layout::Strided &&
                    tensor.is_contiguous() && tensor.numel() >= min_numel,
                "Qwen3VLVisionModel pooler Z1 retained DeepStack requires "
                "contiguous FP16 RPU ", name, " at ordinal ", ordinal,
                " with at least ", min_numel, " elements");
        };
        auto require_vector = [&require_fp16_owner](
                                  const at::Tensor& tensor,
                                  const char* name,
                                  int64_t ordinal,
                                  int64_t numel) {
            require_fp16_owner(tensor, name, ordinal, numel);
            TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == numel,
                        "Qwen3VLVisionModel pooler Z1 retained DeepStack "
                        "requires ", name, " at ordinal ", ordinal,
                        " to be a ", numel, "-element vector");
        };
        auto require_matrix = [&require_fp16_owner](
                                  const at::Tensor& tensor,
                                  const char* name,
                                  int64_t ordinal,
                                  int64_t min_numel) {
            require_fp16_owner(tensor, name, ordinal, min_numel);
            TORCH_CHECK(tensor.dim() == 2,
                        "Qwen3VLVisionModel pooler Z1 retained DeepStack "
                        "requires 2D ", name, " at ordinal ", ordinal);
        };

        const int64_t mh = merge_hidden_;
        const int64_t oh = merger_out_hidden_;
        for (int64_t ordinal = 0; ordinal < retained_count; ++ordinal) {
            require_vector(
                ds_merger_ln_q_w_[ordinal], "ds_ln_q_w", ordinal, mh);
            require_vector(
                ds_merger_ln_q_b_[ordinal], "ds_ln_q_b", ordinal, mh);
            require_matrix(
                ds_merger_m0_w_[ordinal], "ds_m0_w", ordinal, mh * mh);
            require_vector(
                ds_merger_m0_b_[ordinal], "ds_m0_b", ordinal, mh);
            require_matrix(
                ds_merger_m2_w_[ordinal], "ds_m2_w", ordinal, oh * mh);
            require_vector(
                ds_merger_m2_b_[ordinal], "ds_m2_b", ordinal, oh);
        }
    }

    void validate_pooler_z1_contract() const {
        TORCH_CHECK(z1_prepared_num_patches_ ==
                        QWEN3VL_POOLER_Z1_NUM_PATCHES,
                    "Qwen3VLVisionModel pooler Z1 exact layout was not prepared");
        validate_pooler_z1_profile();
        TORCH_CHECK(z1_retained_deepstack_count_ == 0 ||
                        z1_retained_deepstack_count_ == 1 ||
                        z1_retained_deepstack_count_ == 3,
                    "Qwen3VLVisionModel pooler Z1 retained DeepStack count "
                    "drifted after prepare");
        if (z1_retained_deepstack_count_ != 0) {
            validate_pooler_z1_retained_deepstack_profile(
                z1_retained_deepstack_count_);
        }
        const int64_t local_q_dim =
            (num_q_heads() / NUM_CORES) * head_dim();
        TORCH_CHECK(q_ddr_slots_.count(
                        {QWEN3VL_POOLER_Z1_NUM_PATCHES, local_q_dim}) == 1,
                    "Qwen3VLVisionModel pooler Z1 fixed Q staging slot is "
                    "missing");
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerBiasNorm> layer_bias_norm_;

    // 2D RoPE state
    at::Tensor freq_cos_;
    at::Tensor freq_sin_;
    at::Tensor position_idx_keepalive_;
    int64_t max_hw_ = 0;
    bool has_rope_ = false;
    bool rope_spm_enabled_ = false;
    // RhinoVLA fast-replay opt-in (per-model; flow into ModelStaticConfig in
    // build_config, honored by FusedModelBase::run_all_layers). Default false →
    // other qwen3_vl consumers (gr00t) unaffected.
    bool fast_replay_skip_layer_loop_ = false;
    bool preload_replay_skip_ = false;
    bool bake_merger_ = false;
    // RhinoVLA prefix-scatter: merger writes directly into the persistent prefix
    // buffers (host ptrs kept alive Python-side) at the visual run positions.
    bool prefix_scatter_active_ = false;
    // Retain the scatter-target tensors so their DMA destinations cannot outlive
    // their backing storage. The pointers below are valid while these tensors are.
    at::Tensor prefix_pooler_keep_;
    std::vector<at::Tensor> prefix_dense_keep_;
    c10::Half* prefix_pooler_ptr_ = nullptr;
    std::vector<c10::Half*> prefix_dense_ptrs_;
    std::vector<std::pair<int64_t,int64_t>> prefix_runs_;
   public:
    void set_fast_replay_skip_layer_loop(bool e) { fast_replay_skip_layer_loop_ = e; }
    void set_preload_replay_skip(bool e) { preload_replay_skip_ = e; }
    void set_bake_merger(bool e) { bake_merger_ = e; }
    // Takes the target tensors, not integer addresses. See the
    // keepalive members for why. Validation is now possible at all — an integer
    // could only be checked against 0, whereas a Tensor can be checked for
    // device, dtype and contiguity, every one of which the scatter DMA assumes.
    void set_prefix_scatter(const at::Tensor& inputs_embeds,
                            const std::vector<at::Tensor>& dense,
                            const std::vector<int64_t>& run_starts,
                            const std::vector<int64_t>& run_lengths) {
        TORCH_CHECK(run_starts.size() == run_lengths.size(),
                    "set_prefix_scatter: run_starts/run_lengths size mismatch");
        auto check_target = [](const at::Tensor& t, const char* what) {
            TORCH_CHECK(t.defined(), "set_prefix_scatter: ", what, " is undefined");
            TORCH_CHECK(t.device().type() == at::kPrivateUse1,
                        "set_prefix_scatter: ", what, " must be on the RPU device, got ",
                        t.device());
            TORCH_CHECK(t.scalar_type() == at::kHalf,
                        "set_prefix_scatter: ", what, " must be fp16, got ", t.scalar_type());
            TORCH_CHECK(t.is_contiguous(),
                        "set_prefix_scatter: ", what, " must be contiguous");
        };
        check_target(inputs_embeds, "inputs_embeds (pooler target)");
        for (size_t i = 0; i < dense.size(); ++i)
            check_target(dense[i], "a deepstack dense target");
        for (size_t i = 0; i < run_starts.size(); ++i)
            // len must be > 0: the per-run DMA asserts num_elements > 0, and a
            // negative start/len would scatter out of the merger_out / prefix range.
            TORCH_CHECK(run_starts[i] >= 0 && run_lengths[i] > 0,
                        "set_prefix_scatter: run start/length must be >=0 / >0 at ", i,
                        " (start=", run_starts[i], ", len=", run_lengths[i], ")");
        // Retain FIRST, then derive. The pointers are valid exactly as long as
        // these members hold the storage.
        prefix_pooler_keep_ = inputs_embeds;
        prefix_dense_keep_  = dense;
        prefix_pooler_ptr_  = prefix_pooler_keep_.data_ptr<c10::Half>();
        prefix_dense_ptrs_.clear();
        for (const at::Tensor& t : prefix_dense_keep_)
            prefix_dense_ptrs_.push_back(t.data_ptr<c10::Half>());
        prefix_runs_.clear();
        for (size_t i = 0; i < run_starts.size(); ++i)
            prefix_runs_.emplace_back(run_starts[i], run_lengths[i]);
        prefix_scatter_active_ = true;
    }
    void scatter_merger_out_to_prefix(int64_t oh, c10::Half* target) {
        int64_t off = 0;
        for (const auto& run : prefix_runs_) {
            const int64_t start = run.first, len = run.second;
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "merger_out") + static_cast<uint32_t>(off * oh * sizeof(c10::Half)),
                target + start * oh, len * oh);
            off += len;
        }
    }
   private:
    int64_t current_num_patches_ = 0;
    std::map<std::pair<int64_t, int64_t>, at::Tensor> q_ddr_slots_;
    at::Tensor& q_ddr_slot() {
        return q_ddr_slots_.at({current_num_patches_, (num_q_heads() / NUM_CORES) * head_dim()});
    }
    // Multi-image BATCH: N equal-size images packed into one forward; drives the
    // per-image minibatch SDPA in build_layer_subgraph (=1 => single-image path).
    int64_t image_batch_count_ = 1;
    // Public exact execution control. Zero preserves the established
    // shape/mode-dependent policy above; a positive value is checked against
    // the actual resolved plan by Python after every dispatch.
    int64_t configured_chunk_size_ = 0;

    // Pooler-only physical Z1 canary state. The future coordinator owns the
    // lease/route; this producer retains only its checked lifecycle token and
    // the physical address resolved from a typed SpmPortView.
    bool z1_adopted_ = false;
    bool z1_bound_ = false;
    bool z1_dispatch_ = false;
    bool z1_inputs_primed_ = false;
    int64_t z1_prepared_num_patches_ = 0;
    uint32_t z1_source_addr_ = 0;
    int64_t z1_retained_deepstack_count_ = 0;
    int64_t z1_retained_deepstack_bound_count_ = 0;
    std::array<uint32_t, 3> z1_retained_deepstack_source_addrs_{};
    bool multiview_dry_prepared_ = false;
    uint64_t z1_epoch_ = 0;
    uint64_t z1_plan_hash_ = 0;
    const int16_t* z1_primed_position_ptr_ = nullptr;
    const c10::TensorImpl* z1_primed_position_owner_ = nullptr;
    bool multiview_composite_prepared_ = false;
    bool multiview_composite_dispatch_ = false;
    bool multiview_force_block_twostage_ = false;
    const int16_t* multiview_primed_position_ptr_ = nullptr;
    const c10::TensorImpl* multiview_primed_position_owner_ = nullptr;

    // DeepStack snapshot buffers (per-layer, per-seq_len keyed for shape-stable
    // multi-shape REPLAY safety — same pattern as SigLIP's temp_ddr_slots_).
    // Slots persist for the model lifetime; `pop_deepstack_snapshots()` looks
    // them up by `(layer_idx, current_num_patches_)` after each forward.
    //
    // Bounded, NOT evictable: build_layer_subgraph and merger_post_fn bake
    // `slot.data_ptr()` into fixed DMA at BUILD, and the GraphCache keeps that
    // graph, so freeing a slot hands its VA to the next allocation while a
    // cached REPLAY still writes there (C-1). One entry per (deepstack layer x
    // image shape): a Qwen3-VL 2B handle has 3 deepstack layers, so 128 admits
    // ~42 distinct image resolutions on one handle.
    // The bound leaves headroom for supported multi-image shape sets while
    // preventing unbounded fixed-address slot growth.
    static constexpr size_t kMaxSlotShapes = 128;
    rpu::BoundedShapeMap<std::pair<int64_t, int64_t>, at::Tensor, rpu::PairHash>
        deepstack_ddr_slots_{kMaxSlotShapes, rpu::ShapeMapPolicy::Reject,
                             "Qwen3VLVisionModel::deepstack_ddr_slots_"};
    std::vector<int64_t> deepstack_visual_indexes_;

    // Model config
    double eps_ = 1e-6;
    int64_t orig_head_dim_ = 0;

    // Fused patch merger: LayerNorm(ln_q) -> reshape[seq/4, mh] -> m0 GEMM
    // (col) -> GELU -> m2 GEMM (row) -> all-reduce -> SPM->DDR. In-graph post_fn, single image.
    at::Tensor merger_ln_q_w_, merger_ln_q_b_;   // [HID] fp16 (LayerNorm gamma/beta)
    at::Tensor merger_m0_w_, merger_m0_b_;       // [merge_hidden, merge_hidden], [merge_hidden]
    at::Tensor merger_m2_w_, merger_m2_b_;       // [OUT_HID, merge_hidden], [OUT_HID]
    int64_t merger_out_hidden_ = 0;              // OUT_HID (2048); 0 => merger not configured
    int64_t merge_hidden_ = 0;                   // HID * sms^2 (4096)
    double merger_ln_eps_ = 1e-6;                // merger.norm.eps (distinct from the ViT eps_)
    bool fused_merger_enabled_ = false;           // cold at set_merger_weights()
    // Stable patch-merged pooler slot [num_patches/4, oh] (fixed DMA, popped by pop_pooler_merged).
    // Bounded and non-evictable because merger_post_fn bakes the slot address
    // into fixed DMA at BUILD. It is keyed by num_patches.
    rpu::BoundedShapeMap<int64_t, at::Tensor>
        pooler_merged_ddr_slots_{kMaxSlotShapes, rpu::ShapeMapPolicy::Reject,
                                 "Qwen3VLVisionModel::pooler_merged_ddr_slots_"};

    // The N deepstack mergers (one per deepstack_visual_index). Same shapes as the patch
    // merger; folded in merger_post_fn (each reads its layer's snapshot → stable merged slot).
    std::vector<at::Tensor> ds_merger_ln_q_w_, ds_merger_ln_q_b_;
    std::vector<at::Tensor> ds_merger_m0_w_, ds_merger_m0_b_;
    std::vector<at::Tensor> ds_merger_m2_w_, ds_merger_m2_b_;
    // Stable merged-output slots [num_patches/4, oh], shape-keyed like deepstack_ddr_slots_
    // — same bound, same Reject policy, same fixed-DMA reason.
    rpu::BoundedShapeMap<std::pair<int64_t, int64_t>, at::Tensor, rpu::PairHash>
        deepstack_merged_ddr_slots_{kMaxSlotShapes, rpu::ShapeMapPolicy::Reject,
                                    "Qwen3VLVisionModel::deepstack_merged_ddr_slots_"};
};

}  // namespace v3

using Qwen3VLVisionRegistry = ModelHandleRegistry<v3::Qwen3VLVisionModel>;

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers
// =============================================================================

int64_t rpu_qwen3vl_vision_create() {
    return Qwen3VLVisionRegistry::create();
}

void rpu_qwen3vl_vision_destroy(int64_t handle) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_destroy")
        ->check_pooler_z1_destroy_allowed();
    Qwen3VLVisionRegistry::destroy(handle, "rpu_qwen3vl_vision_destroy");
}

void rpu_qwen3vl_vision_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::IntArrayRef deepstack_visual_indexes)
{
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        num_heads, head_dim, hidden_size, intermediate_size,
        eps, deepstack_visual_indexes);
}

// W8A16 variant: required per-output-channel scale lists for the 6 quantized GEMMs (separate op
// because aten schemas can't default Tensor[] args; mirrors gr00t_vl_encoder_set_weights_w8a16).
void rpu_qwen3vl_vision_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::IntArrayRef deepstack_visual_indexes,
    at::TensorList q_scale_list, at::TensorList k_scale_list,
    at::TensorList v_scale_list, at::TensorList o_scale_list,
    at::TensorList fc1_scale_list, at::TensorList fc2_scale_list)
{
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        num_heads, head_dim, hidden_size, intermediate_size,
        eps, deepstack_visual_indexes,
        q_scale_list, k_scale_list, v_scale_list, o_scale_list,
        fc1_scale_list, fc2_scale_list);
}

void rpu_qwen3vl_vision_set_rope(
    int64_t handle,
    const at::Tensor& freq_cos,
    const at::Tensor& freq_sin)
{
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_rope_tables(freq_cos, freq_sin);
}

// RhinoVLA vision fast-replay opt-ins (per-model).
void rpu_qwen3vl_vision_set_fast_replay(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_fast_replay_skip_layer_loop(enabled);
}

void rpu_qwen3vl_vision_set_preload_replay_skip(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_preload_replay_skip(enabled);
}

void rpu_qwen3vl_vision_set_chunk_size(int64_t handle, int64_t chunk_size) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_set_chunk_size")
        ->set_configured_chunk_size(chunk_size);
}

int64_t rpu_qwen3vl_vision_get_resolved_chunk_size(int64_t handle) {
    return Qwen3VLVisionRegistry::get(
               handle, "rpu_qwen3vl_vision_get_resolved_chunk_size")
        ->resolved_chunk_size();
}

void rpu_qwen3vl_vision_set_bake_merger(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_bake_merger(enabled);
}

void rpu_qwen3vl_vision_set_prefix_scatter(
    int64_t handle, const at::Tensor& inputs_embeds, at::TensorList dense,
    at::IntArrayRef run_starts, at::IntArrayRef run_lengths)
{
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_prefix_scatter(
            inputs_embeds,
            std::vector<at::Tensor>(dense.begin(), dense.end()),
            std::vector<int64_t>(run_starts.begin(), run_starts.end()),
            std::vector<int64_t>(run_lengths.begin(), run_lengths.end()));
}

void rpu_qwen3vl_vision_set_merger_weights(
    int64_t handle,
    const at::Tensor& ln_q_w, const at::Tensor& ln_q_b,
    const at::Tensor& m0_w, const at::Tensor& m0_b,
    const at::Tensor& m2_w, const at::Tensor& m2_b,
    int64_t out_hidden, double ln_eps)
{
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_merger_weights(ln_q_w, ln_q_b, m0_w, m0_b, m2_w, m2_b, out_hidden, ln_eps);
}

void rpu_qwen3vl_vision_set_deepstack_merger_weights(
    int64_t handle,
    at::TensorList ln_q_w, at::TensorList ln_q_b,
    at::TensorList m0_w, at::TensorList m0_b,
    at::TensorList m2_w, at::TensorList m2_b)
{
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_deepstack_merger_weights(ln_q_w, ln_q_b, m0_w, m0_b, m2_w, m2_b);
}

at::Tensor rpu_qwen3vl_vision_position_idx_keepalive(int64_t handle) {
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->position_idx_keepalive();
}

at::Tensor rpu_qwen3vl_vision_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t num_patches,
    int64_t image_batch_count)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->forward(input, k_caches, v_caches, num_patches, image_batch_count);
}

namespace v3::qwen3vl_pooler_z1_internal {

SpmPipelineComponentLayout prepare_vision(int64_t handle,
                                          int64_t num_patches,
                                          int64_t retained_count) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_pooler_z1_prepare_vision")
        ->prepare_pooler_z1_layout(num_patches, retained_count);
}

SpmPipelineComponentLayout prepare_multiview_vision_dry(
    int64_t handle,
    int64_t num_patches) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_multiview_prepare_vision_dry")
        ->prepare_multiview_dry_layout(num_patches);
}

void cancel_multiview_vision_dry(int64_t handle) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_multiview_cancel_vision_dry")
        ->cancel_multiview_dry_layout();
}

SpmPipelineComponentLayout prepare_multiview_vision_composite(
    int64_t handle,
    int64_t num_patches,
    bool force_block_twostage) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_multiview_prepare_vision_composite")
        ->prepare_multiview_composite_layout(
            num_patches, force_block_twostage);
}

SpmPipelineComponentLayout prepare_multiview_vision_composite(
    int64_t handle,
    int64_t num_patches) {
    return prepare_multiview_vision_composite(
        handle, num_patches, /*force_block_twostage=*/false);
}

SpmFmbResolvedExecutionProfile resolve_multiview_vision_profile(
    int64_t handle) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_multiview_resolve_vision_profile")
        ->resolve_multiview_composite_profile();
}

uint64_t multiview_vision_occurrence_policy(int64_t handle) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_multiview_vision_occurrence_policy")
        ->multiview_composite_occurrence_policy();
}

SpmFmbResolvedPhaseManifest seal_multiview_vision_manifest(
    int64_t handle,
    const SpmFmbResolvedExecutionProfile& profile,
    SpmScratchId arena,
    uint32_t arena_base) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_multiview_seal_vision_manifest")
        ->seal_multiview_composite_manifest(profile, arena, arena_base);
}

FusedModelBase& multiview_vision_owner(int64_t handle) {
    return *Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_multiview_vision_owner");
}

void prime_multiview_vision_position(int64_t handle) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_multiview_prime_vision_position")
        ->prime_multiview_composite_position();
}

void clear_multiview_vision_position(int64_t handle) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_multiview_clear_vision_position")
        ->clear_multiview_composite_position();
}

void unprepare_multiview_vision_composite(int64_t handle) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_multiview_unprepare_vision_composite")
        ->unprepare_multiview_composite_layout();
}

void forward_multiview_vision_composite(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches,
    at::TensorList v_caches) {
    std::vector<at::Tensor> k_cache_vec(
        k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> v_cache_vec(
        v_caches.begin(), v_caches.end());
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_multiview_forward_vision_composite")
        ->forward_multiview_composite(input, k_cache_vec, v_cache_vec);
}

void stage_multiview_vision_outer_fast_component(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_multiview_stage_vision_outer_fast_component")
        ->stage_multiview_vision_outer_fast_component(
            guard, k_caches, v_caches);
}

void bind_multiview_vision_outer_fast_input(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& input,
    size_t ordinal) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_multiview_bind_vision_outer_fast_input")
        ->bind_multiview_vision_outer_fast_input(guard, input, ordinal);
}

void prime_vision_inputs(int64_t handle, int64_t num_patches) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_prime_vision_inputs")
        ->prime_pooler_z1_inputs(num_patches);
}

void rollback_vision_inputs(int64_t handle) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_rollback_vision_inputs")
        ->rollback_pooler_z1_inputs();
}

void unprepare_vision(int64_t handle) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_unprepare_vision")
        ->unprepare_pooler_z1_layout();
}

SpmDense2DSpec vision_source_spec(int64_t handle,
                                  int64_t num_patches) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_pooler_z1_vision_source_spec")
        ->pooler_z1_source_spec(num_patches);
}

SpmDense2DSpec vision_deepstack_source_spec(int64_t handle,
                                            int64_t num_patches,
                                            int64_t ordinal) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_pooler_z1_vision_deepstack_source_spec")
        ->pooler_z1_deepstack_source_spec(num_patches, ordinal);
}

SpmDense2DSpec vision_deepstack1_source_spec(int64_t handle,
                                             int64_t num_patches) {
    return vision_deepstack_source_spec(
        handle, num_patches, /*ordinal=*/0);
}

void adopt_vision(int64_t handle,
                  const SpmPipelineLease& lease,
                  const SpmTensorView& scratch) {
    Qwen3VLVisionRegistry::get(handle, "qwen3vl_pooler_z1_adopt_vision")
        ->adopt_pooler_z1_layout(lease, scratch);
}

void bind_vision_source(int64_t handle,
                        const SpmPipelineLease& lease,
                        const SpmPortView& source) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_bind_vision_source")
        ->bind_pooler_z1_source(lease, source);
}

void bind_vision_deepstack_source(int64_t handle,
                                  const SpmPipelineLease& lease,
                                  const SpmPortView& source,
                                  int64_t ordinal) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_bind_vision_deepstack_source")
        ->bind_pooler_z1_deepstack_source(lease, source, ordinal);
}

void bind_vision_deepstack1_source(int64_t handle,
                                   const SpmPipelineLease& lease,
                                   const SpmPortView& source) {
    bind_vision_deepstack_source(
        handle, lease, source, /*ordinal=*/0);
}

void validate_vision(int64_t handle,
                     const SpmPipelineLease& lease) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_validate_vision")
        ->validate_pooler_z1_layout(lease);
}

void clear_vision(int64_t handle,
                  uint64_t epoch,
                  uint64_t plan_hash) {
    Qwen3VLVisionRegistry::get(handle, "qwen3vl_pooler_z1_clear_vision")
        ->clear_pooler_z1_layout(epoch, plan_hash);
}

void stage_vision_outer_fast_component(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_stage_vision_outer_fast_component")
        ->stage_pooler_z1_outer_fast_component(
            guard, k_caches, v_caches);
}

void bind_vision_outer_fast_input(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& input) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_bind_vision_outer_fast_input")
        ->bind_pooler_z1_outer_fast_input(guard, input);
}

void forward_vision_z1(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches,
    at::TensorList v_caches,
    int64_t num_patches,
    uint64_t epoch,
    uint64_t plan_hash) {
    std::vector<at::Tensor> k_cache_vec(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> v_cache_vec(v_caches.begin(), v_caches.end());
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_forward_vision")
        ->forward_pooler_z1(input, k_cache_vec, v_cache_vec, num_patches,
                            epoch, plan_hash);
}

void check_vision_destroy_allowed(int64_t handle) {
    Qwen3VLVisionRegistry::get(
        handle, "qwen3vl_pooler_z1_check_vision_destroy")
        ->check_pooler_z1_destroy_allowed();
}

}  // namespace v3::qwen3vl_pooler_z1_internal

std::vector<at::Tensor> rpu_qwen3vl_vision_pop_deepstack_snapshots(int64_t handle) {
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->pop_deepstack_snapshots();
}

std::vector<at::Tensor> rpu_qwen3vl_vision_pop_deepstack_merged(int64_t handle) {
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->pop_deepstack_merged();
}

at::Tensor rpu_qwen3vl_vision_pop_pooler_merged(int64_t handle) {
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->pop_pooler_merged();
}

std::vector<at::Tensor> rpu_qwen3vl_vision_pop_merged(int64_t handle) {
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->pop_merged();
}
