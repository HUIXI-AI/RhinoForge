// rpu_qwen25vl_vision_model.cpp — Qwen2.5-VL Vision Encoder (Wall-OSS-0.5 P3)
//
// 32-block ViT encoder for Qwen2.5-VL-3B (depth=32, hidden=1280,
// intermediate=3420→padded 3456, num_heads=16, head_dim=80). Forked from
// rpu_qwen3vl_vision_model.cpp (R-Phase 3). Differences from Qwen3-VL ViT:
//   - Norm: RMSNorm (single weight, no bias) — Qwen3-VL used LayerNorm (γ+β).
//   - MLP : SwiGLU `down(silu(gate(x)) * up(x))` with bias — Qwen3-VL used
//           fc1 → GELU(tanh) → fc2. gate/up/down all have bias.
//   - intermediate 3420 is not swizzle-representable (3420 % 128 = 92); the
//     Python adapter zero-pads gate/up/down (+biases) to 3456 and passes
//     intermediate_size=3456 → local_inter=432. silu(0)=0 keeps the pad inert.
//   - head_dim 80 (Qwen3-VL 64). 2D RoPE table is [max_hw, head_dim/4=20].
//   - Window attention (fullatt_block_indexes) → added in P3b; P3a is full-attn
//     (MASK_NONE) only, to isolate the RMSNorm/SwiGLU/RoPE/head-dim port.
//   - No DeepStack (that was Qwen3-VL specific).
//
// Patch embedding + window reorder + merger live in the Python adapter (CPU).
// The fused subsystem expects pre-embedded (and, in P3b, pre-reordered) input
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
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <utility>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

constexpr int64_t QWEN25VL_VISION_SDPA_QUERY_CHUNK = 144;
constexpr int64_t QWEN25VL_VISION_DEBUG_PHASE_LAYER = 17;
constexpr int64_t QWEN25VL_VISION_DEBUG_PHASE_COUNT = 3;
constexpr int64_t QWEN25VL_VISION_MIXED_BOUNDARY_LAYER = 17;

// Stable descriptor-route identities from planner_owners.v1.json.
// QWEN25_FIXED_KERNEL_BASIS: normalization, activation, unconditional bias/
// debug/staging DMAs, fixed mixed-precision conversions, and BufferDecl
// dataflow are model mathematics. Conditional RoPE-table preloads and the
// other variable sites below consume the native planned descriptor.
constexpr int64_t QWEN25_EXACT_QKV_LINEAR_SITE = 2514780956293204872LL;
constexpr int64_t QWEN25_EXACT_Q_ROPE_SPM_SITE = 4828902400580560942LL;
constexpr int64_t QWEN25_EXACT_K_ROPE_SPM_SITE = 1668643865188357011LL;
constexpr int64_t QWEN25_EXACT_Q_ROPE_DDR_SITE = 3480241264239683514LL;
constexpr int64_t QWEN25_EXACT_K_ROPE_DDR_SITE = 7554001762755986171LL;
constexpr int64_t QWEN25_EXACT_WINDOW_ATTN_SITE = 4805829272781421824LL;
constexpr int64_t QWEN25_EXACT_W8_ATTN_SITE = 8187241731964351607LL;
constexpr int64_t QWEN25_EXACT_FP16_ATTN_SITE = 7303897471582959629LL;

constexpr int64_t QWEN25_EXACT_O_LINEAR_SITE = 52823389030817526LL;
constexpr int64_t QWEN25_EXACT_ATTN_MIXED_LOCAL_REDUCE_SITE =
    5025021987307885657LL;
constexpr int64_t QWEN25_EXACT_ATTN_LOCAL_REDUCE_SITE =
    2979339668351978514LL;
constexpr int64_t QWEN25_EXACT_GATE_LINEAR_SITE = 6385653909583992262LL;
constexpr int64_t QWEN25_EXACT_UP_LINEAR_SITE = 4530664251707464878LL;
constexpr int64_t QWEN25_EXACT_DOWN_LINEAR_SITE = 438184008798115415LL;
constexpr int64_t QWEN25_EXACT_DOWN_BF16_LINEAR_SITE = 4378989168527940581LL;
constexpr int64_t QWEN25_EXACT_FINAL_BF16_REDUCE_SITE = 400917888117649833LL;
constexpr int64_t QWEN25_EXACT_FINAL_MIXED_REDUCE_SITE =
    3644880364681387303LL;
constexpr int64_t QWEN25_EXACT_FINAL_REDUCE_SITE = 428060968633182009LL;

constexpr int64_t QWEN25_Q_LINEAR_SITE = 1160484491727409945LL;
constexpr int64_t QWEN25_K_LINEAR_SITE = 2958011619077466318LL;
constexpr int64_t QWEN25_V_LINEAR_SITE = 636614201150917303LL;
constexpr int64_t QWEN25_Q_ROPE_SPM_SITE = 6235182035327233838LL;
constexpr int64_t QWEN25_K_ROPE_SPM_SITE = 1519024389716768287LL;
constexpr int64_t QWEN25_Q_ROPE_DDR_SITE = 2462104509382655163LL;
constexpr int64_t QWEN25_K_ROPE_DDR_SITE = 6836747574341429713LL;
constexpr int64_t QWEN25_KV_INSERT_SITE = 5299313915388297259LL;
constexpr int64_t QWEN25_MISC_RAW_ATTN_SITE = 5587652333716339060LL;
constexpr int64_t QWEN25_MISC_DDR_ATTN_SITE = 2545818117373449458LL;
constexpr int64_t QWEN25_WINDOW_RAW_ATTN_SITE = 5718369161792163142LL;
constexpr int64_t QWEN25_WINDOW_DDR_ATTN_SITE = 913880989367815128LL;
constexpr int64_t QWEN25_STANDARD_RAW_ATTN_SITE = 8716771559110690244LL;
constexpr int64_t QWEN25_STANDARD_DDR_ATTN_SITE = 4186875649763190810LL;
constexpr int64_t QWEN25_O_LINEAR_SITE = 8440022237838681726LL;
constexpr int64_t QWEN25_ATTN_MIXED_REDUCE_SITE = 5997427131279928968LL;
constexpr int64_t QWEN25_ATTN_REDUCE_SITE = 6239376048939361682LL;
constexpr int64_t QWEN25_GATE_LINEAR_SITE = 7069910951808887182LL;
constexpr int64_t QWEN25_UP_LINEAR_SITE = 5072206025442967753LL;
constexpr int64_t QWEN25_DOWN_BF16_LINEAR_SITE = 5551939502927464479LL;
constexpr int64_t QWEN25_DOWN_BF16_REDUCE_SITE = 1209336826686278430LL;
constexpr int64_t QWEN25_DOWN_LINEAR_SITE = 6152590060744692022LL;
constexpr int64_t QWEN25_DOWN_MIXED_REDUCE_SITE = 5805462007129126563LL;
constexpr int64_t QWEN25_DOWN_REDUCE_SITE = 5974126247251259304LL;

constexpr int64_t QWEN25_MERGER_M0_LINEAR_SITE = 7688514023374193399LL;
constexpr int64_t QWEN25_MERGER_M2_LINEAR_SITE = 8264516795122439569LL;
constexpr int64_t QWEN25_MERGER_REDUCE_SITE = 4487386590070984396LL;
constexpr int64_t QWEN25_MERGER_OUTPUT_DMA_SITE = 2975338017698946358LL;
constexpr int64_t QWEN25_ROPE_COS_PRELOAD_SITE = 6623618708631996252LL;
constexpr int64_t QWEN25_ROPE_SIN_PRELOAD_SITE = 3659110372012461435LL;

// These exact RTC receipts retain their fixed M576 image/compute geometry, not
// independent planner search sites.
constexpr int64_t kQwen25ExactPreRmsNormRowsSite =
    4290146531683843434LL;
constexpr int64_t kQwen25ExactComputeRmsNormRowsSite =
    1667235796600535957LL;
constexpr int64_t QWEN25_EXACT_COMPUTE_MIXED_RMSNORM_SITE =
    7582428809785826332LL;
constexpr int64_t QWEN25_INPUT_MIXED_RMSNORM_SITE =
    1443113074468322082LL;
constexpr int64_t QWEN25_POST_MIXED_RMSNORM_SITE =
    7115174263311880709LL;
constexpr int64_t QWEN25_MERGER_MIXED_RMSNORM_SITE =
    4403245657541910202LL;
constexpr int64_t QWEN25_EXACT_MERGER_MIXED_RMSNORM_SITE =
    5209555832750254855LL;

// Separate receipts for the ordinary cold small-image vector normalization.
// They do not reinterpret the public exact RTC FIXED32 capability or sites.
constexpr int64_t QWEN25_SMALL_INPUT_RMSNORM_SITE = 6071795889225267676LL;
constexpr int64_t QWEN25_SMALL_POST_RMSNORM_SITE = 3726621593416076355LL;
constexpr int64_t QWEN25_SMALL_MERGER_RMSNORM_SITE = 518375447078562164LL;
constexpr int64_t QWEN25_SMALL_OWNER_RING_NORM_SITE = 206720128084120001LL;
constexpr int64_t QWEN25_SMALL_DOWN_OWNER_NORM_SITE = 206720128084120002LL;
constexpr int64_t QWEN25_SMALL_QK_ROPE_SPM_SITE = 206720128084120003LL;
constexpr int64_t QWEN25_SMALL_RAGGED_SDPA_SITE = 206720128084120004LL;

constexpr int64_t QWEN25_WALL_SORTING_SITE = 570820779484829457LL;

constexpr uint32_t QWEN25_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16;
// Stable descriptor flags. Generic raw-SPM attention mirrors K/V into RPUCache
// for paired DDR candidates/debugging. The cold three224 native-window route
// has no DDR candidate or DDR consumer and can omit that mirror outside debug.
constexpr int64_t QWEN25_KV_FLAG_DDR_MIRROR = 1;
constexpr int64_t QWEN25_ATTN_DDR_CAPABILITY_FALLBACK = 1;
constexpr int64_t QWEN25_ATTN_DDR_RAW_KERNEL_INCOMPATIBLE = 2;
constexpr int64_t QWEN25_ATTN_DDR_LAYER_GROUP_LAYOUT = 4;

enum class Qwen25VisionAllReduceRoute : int64_t {
    RING_LOCAL_RESIDUAL = 3,
    MIXED_LOCAL_RESIDUAL = 4,
    FP16_PARTIAL_BF16_RESIDUAL = 5,
    BF16_PARTIAL_FP16_RESIDUAL = 6,
    EXACT_M576_BF16_PARTIAL_FP16_RESIDUAL = 7,
    EXACT_M576_FP16_PARTIAL_BF16_RESIDUAL = 8,
    EXACT_M576_SUM_RESIDUAL = 9,
    OWNER_RING_NORM_M672 = 10,
    OWNER_DOWN_RING_NORM_M672 = 11,
};

enum class Qwen25VisionLinearRoute : int64_t {
    AUTO_TILE = static_cast<int64_t>(v3::FmbLinearRouteSelector::AUTO_TILE),
    ACC32_OUT_BF16 = 4,
    WALL_ROW_FULL_LOCAL_K = 5,
    WALL_SINGLE_BASE_GATE_UP_SWIGLU = 6,
};

enum class Qwen25VisionRopeRoute : int64_t {
    ROPE_2D_SPM = 1,
    ROPE_2D_DDR = 2,
    PAIRED_QK_2D_SPM = 3,
};

enum class Qwen25VisionMutableDmaRoute : int64_t {
    SPM_COPY_TO_DDR = 1,
    ROPE_TABLE_DDR_TO_SPM = 2,
};

constexpr int64_t qwen25_vision_linear_route() {
    return static_cast<int64_t>(Qwen25VisionLinearRoute::AUTO_TILE);
}

int64_t qwen25_vision_ring_route(int64_t rows, int64_t cols) {
    return v3::fmb_ring_all_reduce_route_selector(rows, cols);
}

// First physical window-merger canary only. Keep this envelope intentionally
// narrow until the permutation route has proven BUILD -> stable REPLAY.
constexpr int64_t QWEN25VL_Z1_NUM_PATCHES = 256;
constexpr int64_t QWEN25VL_Z1_VISION_HIDDEN = 1280;
constexpr int64_t QWEN25VL_Z1_MERGED_ROWS = 64;
constexpr int64_t QWEN25VL_Z1_TEXT_HIDDEN = 2048;

namespace v3 {

struct ExactMimcPlan {
    static constexpr int64_t kImages = 3;
    static constexpr int64_t kQkvRows = 1728;
    static constexpr int64_t kAttentionRows = 576;
    static constexpr int64_t kMlpRows = 576;
    static constexpr int64_t kHiddenSize = 1280;
    static constexpr int64_t kCores = 8;
    static constexpr int64_t kResidualShardElements =
        kAttentionRows * kHiddenSize / kCores;
    static constexpr int64_t kResidualShardBytes =
        kResidualShardElements * DWIDTH;
    static constexpr int64_t kCompactResidualBytesPerCore =
        kImages * kResidualShardBytes;
    static constexpr int64_t kValidatedTempPeakBytes = 7'630'848;
    static constexpr int64_t kNativeWindowTempPeakBytes = 6'967'296;
    static constexpr int64_t kNativeWindowLocalInoutTempPeakBytes = 6'635'520;
    static constexpr int64_t kFullVlaSuperPersistentBytes = 591'360;
    static constexpr int64_t kEffectiveSpmBudgetBytes =
        static_cast<int64_t>(SpmAllocator::SPM_PLANNING_BUDGET &
                             ~(SpmAllocator::ALIGN - 1));
};
static_assert(ExactMimcPlan::kQkvRows
                  == ExactMimcPlan::kImages
                      * ExactMimcPlan::kAttentionRows
              && ExactMimcPlan::kAttentionRows == ExactMimcPlan::kMlpRows);
static_assert(ExactMimcPlan::kResidualShardElements == 92'160
              && ExactMimcPlan::kResidualShardBytes == 184'320
              && ExactMimcPlan::kCompactResidualBytesPerCore == 552'960);
static_assert(ExactMimcPlan::kValidatedTempPeakBytes
                  + ExactMimcPlan::kFullVlaSuperPersistentBytes
              <= ExactMimcPlan::kEffectiveSpmBudgetBytes);
static_assert(ExactMimcPlan::kNativeWindowTempPeakBytes
                  + ExactMimcPlan::kFullVlaSuperPersistentBytes
              <= ExactMimcPlan::kEffectiveSpmBudgetBytes);
static_assert(ExactMimcPlan::kNativeWindowLocalInoutTempPeakBytes
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

static VisionDispatch resolve_vision_dispatch(
    int64_t image_batch_count, bool legacy_layer_group, bool exact_mimc) {
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

    void set_linear_acc32(bool enabled) {
        TORCH_CHECK(num_layers() == 0 && !has_rope_ && !z1_adopted_ && !z1_bound_,
                    "Wall Vision accumulation must be set before weights/RoPE");
        TORCH_CHECK(!enabled || (!mixed_precision_enabled_ && !layer_group_enabled_ &&
                                !exact_mimc_enabled_ && !per_window_sdpa_enabled_),
                    "Wall Vision ACC32 supports ordinary base only, not RTC/exact/per-window");
        linear_acc32_ = enabled;
        invalidate_model_state();
    }

    void set_wall_sorting_routes(int64_t mask, bool row, bool silu) {
        TORCH_CHECK(num_layers() == 0 && !has_rope_,
                    "Wall Vision sorting routes must be set before weights/RoPE");
        TORCH_CHECK(mask >= 0 && mask <= 7, "Wall FP16 transfer mask must be 0..7");
        wall_sorting_.fp16_transfer_mask = static_cast<uint32_t>(mask);
        wall_sorting_.vision_row_reuse = row;
        wall_sorting_.silu_mul = silu;
    }

    void set_execution_routes(
        bool rope_spm, bool fused_merger, bool mixed_precision,
        bool layer_group, bool exact_mimc,
        int64_t rmsnorm_capability) {
        TORCH_CHECK(num_layers() == 0 && !has_rope_,
                    "qwen25vl vision execution routes must be set before weights/RoPE");
        TORCH_CHECK(!linear_acc32_ || (!mixed_precision && !layer_group && !exact_mimc),
                    "Wall Vision ACC32 cannot use RTC mixed or exact execution routes");
        TORCH_CHECK(!mixed_precision || fused_merger,
                    "qwen25vl vision mixed precision requires fused merger");
        TORCH_CHECK(!(layer_group && exact_mimc),
                    "qwen25vl vision layer-group and exact MIMC are mutually exclusive");
        TORCH_CHECK(!(layer_group && mixed_precision),
                    "qwen25vl vision layer-group is incompatible with mixed precision");
        TORCH_CHECK(
            rmsnorm_capability >= 0 &&
                rmsnorm_capability <=
                    static_cast<int64_t>(RPU_RMSNORM_CAP_ALL) &&
                rpu_rmsnorm_capability_valid(
                    static_cast<uint32_t>(rmsnorm_capability)),
            "qwen25vl Vision RMSNorm capability must be a valid typed mask; got ",
            rmsnorm_capability);
        TORCH_CHECK(
            rmsnorm_capability == RPU_RMSNORM_CAP_BASE ||
                rmsnorm_capability ==
                    (RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V32),
            "qwen25vl Vision RMSNorm capability must be BASE or exact "
            "FIXED32; got ", rmsnorm_capability);
        TORCH_CHECK(
            rmsnorm_capability == RPU_RMSNORM_CAP_BASE ||
                (exact_mimc && mixed_precision && fused_merger &&
                 !layer_group),
            "qwen25vl Vision FIXED32 RMSNorm requires exact MIMC, mixed "
            "precision, and fused merger");
        const auto rmsnorm_contract = rpu_snapshot_rmsnorm_spm_contract(
            static_cast<uint32_t>(rmsnorm_capability));
        TORCH_CHECK(
            rmsnorm_capability == RPU_RMSNORM_CAP_BASE ||
                rmsnorm_contract.capability ==
                    static_cast<uint32_t>(rmsnorm_capability),
            "qwen25vl Vision FIXED32 RMSNorm payload is unavailable in the "
            "loaded op libraries");

        // Commit only after every fallible route dependency has been
        // snapshotted and checked. A rejected FIXED32 request leaves a pending
        // handle unchanged and safe to destroy or retry.
        rope_spm_enabled_ = rope_spm;
        fused_merger_enabled_ = fused_merger;
        mixed_precision_enabled_ = mixed_precision;
        layer_group_enabled_ = layer_group;
        exact_mimc_enabled_ = exact_mimc;
        rmsnorm_capability_ = rmsnorm_contract.capability;
    }

    void validate_acc32_base_input() const {
        if (!linear_acc32_) return;
        TORCH_CHECK(is_generic() && !mixed_precision_ && !per_window_sdpa_enabled_ &&
                        !z1_bound_ && num_layers() == 32 && hidden_size() == 1280 &&
                        num_q_heads() == 16 && num_kv_heads() == 16 && head_dim() == 80 &&
                        (intermediate_size() == 3456 || intermediate_size() == 3584),
                    "Wall Vision ACC32 requires the ordinary 32-layer base profile");
        TORCH_CHECK((valid_patches_per_image_ == 0 &&
                        ((image_batch_count_ >= 1 && image_batch_count_ <= 3 &&
                          current_num_patches_ == 256 * image_batch_count_) ||
                         (image_batch_count_ == 1 && current_num_patches_ == 324))) ||
                        padded_pair_enabled(),
                    "Wall Vision ACC32 requires 224/256 base image geometry");
    }

    VisionDispatch resolve_vision_dispatch(int64_t image_batch_count) const {
        return v3::resolve_vision_dispatch(
            image_batch_count, layer_group_enabled_, exact_mimc_enabled_);
    }

    void set_configured_chunk_size(int64_t chunk_size) {
        TORCH_CHECK(chunk_size == 0
                        || (chunk_size >= 16 && chunk_size <= UINT16_MAX &&
                            chunk_size % 16 == 0),
                    "qwen25vl vision chunk_size must be 0 (auto) or a positive "
                    "multiple of 16 within the uint16 RoPE token domain, got ", chunk_size);
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "qwen25vl vision chunk_size must be set before first forward");
        configured_chunk_size_ = chunk_size;
        invalidate_model_state();
    }

    void stage_configured_chunk_size(uint64_t token, int64_t chunk_size) {
        TORCH_CHECK(chunk_size == 0 ||
                        (chunk_size >= 16 && chunk_size <= UINT16_MAX &&
                         chunk_size % 16 == 0),
                    "qwen25vl vision chunk_size must be AUTO(0) or a positive "
                    "multiple of 16 within the uint16 RoPE token domain, got ", chunk_size);
        const int64_t old_chunk = configured_chunk_size_;
        stage_execution_controls(
            token, chunk_size, std::nullopt,
            [this, chunk_size] { configured_chunk_size_ = chunk_size; },
            [this, old_chunk] { configured_chunk_size_ = old_chunk; },
            "Qwen25VLVisionModel::stage_configured_chunk_size");
    }

    std::vector<int64_t> resolve_stage_domain(
        int64_t num_patches, int64_t image_batch_count,
        bool window_mask_present,
        const std::optional<at::Tensor>& cu_window_seqlens,
        int64_t requested_chunk_size,
        int64_t valid_patches_per_image = 0,
        const std::optional<at::Tensor>& planning_window_mask = std::nullopt) {
        TORCH_CHECK(num_layers() > 0 && has_rope_,
                    "RPU_PLANNER_REJECT:CAPABILITY: Qwen2.5-VL Vision "
                    "weights/RoPE are not initialized");
        TORCH_CHECK(num_patches > 0 && image_batch_count > 0 &&
                        num_patches % image_batch_count == 0,
                    "RPU_PLANNER_REJECT:CAPABILITY: Qwen2.5-VL Vision "
                    "patch count must be divisible by image_batch_count");
        current_num_patches_ = num_patches;
        image_batch_count_ = image_batch_count;
        const VisionDispatch dispatch =
            resolve_vision_dispatch(image_batch_count_);
        schedule_ = dispatch.schedule;
        implementation_ = dispatch.implementation;
        mixed_precision_ = mixed_precision_enabled_;
        window_mask_present_ = window_mask_present;
        valid_patches_per_image_ = valid_patches_per_image;
        validate_acc32_base_input();
        TORCH_CHECK(valid_patches_per_image_ == 0 || padded_pair_enabled(),
                    "RPU_PLANNER_REJECT:CAPABILITY: padded Vision requires "
                    "exact uniform B2/M672/valid324 with fused merger and cold SiLU");

        const bool has_boundaries =
            cu_window_seqlens.has_value() && cu_window_seqlens->defined();
        TORCH_CHECK(!per_window_sdpa_enabled_ || has_boundaries,
                    "RPU_PLANNER_REJECT:CAPABILITY: qwen25vl per-window "
                    "planning requires cu_window_seqlens");
        TORCH_CHECK(!has_boundaries || per_window_sdpa_enabled_ ||
                        is_exact_mimc() || small_image_native_window_shape(),
                    "RPU_PLANNER_REJECT:CAPABILITY: qwen25vl boundaries "
                    "require per-window SDPA or exact MIMC");
        cu_window_.clear();
        if (has_boundaries) {
            const at::Tensor& cu = *cu_window_seqlens;
            TORCH_CHECK(cu.device().is_cpu() && cu.scalar_type() == at::kLong &&
                            cu.is_contiguous() && cu.dim() == 1 && cu.numel() >= 2,
                        "RPU_PLANNER_REJECT:CAPABILITY: qwen25vl boundaries "
                        "must be contiguous CPU int64");
            const int64_t* values = cu.data_ptr<int64_t>();
            const int64_t boundary_seq = per_window_sdpa_enabled_
                ? num_patches : num_patches / image_batch_count;
            TORCH_CHECK(values[0] == 0 &&
                            values[cu.numel() - 1] == boundary_seq,
                        "RPU_PLANNER_REJECT:CAPABILITY: qwen25vl boundary "
                        "extent does not match the attention sequence");
            for (int64_t i = 1; i < cu.numel(); ++i) {
                TORCH_CHECK(values[i] > values[i - 1] &&
                                values[i] % 16 == 0,
                            "RPU_PLANNER_REJECT:CAPABILITY: qwen25vl "
                            "boundaries must increase and be 16-aligned");
            }
            cu_window_.assign(values, values + cu.numel());
            TORCH_CHECK(per_window_sdpa_enabled_ || is_exact_mimc() ||
                            small_image_native_window_enabled(),
                        "RPU_PLANNER_REJECT:CAPABILITY: small Vision native "
                        "windows require exactly four 64-patch windows");
        }

        refresh_small_ragged_mask(planning_window_mask, /*planning=*/true);
        const int64_t per_image = num_patches / image_batch_count;
        TORCH_CHECK(
            rmsnorm_capability_ == RPU_RMSNORM_CAP_BASE ||
                (is_exact_mimc() && image_batch_count == ExactMimcPlan::kImages
                 && per_image == ExactMimcPlan::kAttentionRows),
            "RPU_PLANNER_REJECT:CAPABILITY: qwen25vl Vision FIXED32 "
            "RMSNorm requires exact MIMC 3x576 geometry");
        const int64_t required_chunk = Align(
            is_generic() ? num_patches : per_image, int64_t{16});
        TORCH_CHECK(requested_chunk_size == 0 ||
                        requested_chunk_size == required_chunk,
                    "RPU_PLANNER_REJECT:EXACT_MISMATCH: qwen25vl Vision "
                    "exact chunk must match its native schedule");
        set_chunk_size_override(
            is_generic() ? configured_chunk_size_ : required_chunk);

        std::vector<ChunkInfo> input_chunks{
            {0, 0, num_patches, num_patches}};
        std::vector<FmbExecutionSpan> spans;
        spans.reserve(image_batch_count);
        for (int64_t image = 0; image < image_batch_count; ++image) {
            spans.push_back({image * per_image, per_image, image});
        }
        FmbStageBoundaryPolicies policies{};
        if (is_legacy_layer_group()) {
            policies.qkv = FmbSpanBoundaryPolicy::KEEP_LOCAL;
            policies.compute = FmbSpanBoundaryPolicy::KEEP_LOCAL;
        } else if (is_exact_mimc()) {
            policies.compute = FmbSpanBoundaryPolicy::KEEP_LOCAL;
        }
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                num_patches, /*position=*/0, /*attention_mask=*/std::nullopt,
                /*is_causal=*/false, input_chunks, spans, policies,
                requested_chunk_size, /*logical_len=*/padded_pair_enabled()
                    ? image_batch_count_ * valid_patches_per_image_ : num_patches));
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

        if (rmsnorm_capability_ != RPU_RMSNORM_CAP_BASE) {
            TORCH_CHECK(
                exact_mimc_enabled_ && mixed_precision_enabled_ &&
                    fused_merger_enabled_ && !layer_group_enabled_ &&
                    N == 32 && hidden_size == 1280 && num_heads == 16 &&
                    head_dim == 80 && intermediate_size == 3584 &&
                    (has_scale || (wall_sorting_.fp16_transfer_mask & 1u)) &&
                    fullatt_block_indexes_ ==
                        std::vector<int64_t>({7, 15, 23, 31}),
                "qwen25vl Vision FIXED32 RMSNorm requires the exact RTC W8/FP16 sorting "
                "32xH1280/I3584/heads16/head80 window profile");
            for (int64_t i = 0; i < N; ++i) {
                const auto& lw = layer_weights_[i];
                const auto& bn = layer_bias_norm_[i];
                const auto projection_dtype = (wall_sorting_.fp16_transfer_mask & 1u)
                    && !has_scale ? at::kHalf : at::kChar;
                TORCH_CHECK(
                    lw.q_w.scalar_type() == projection_dtype &&
                        lw.k_w.scalar_type() == projection_dtype &&
                        lw.v_w.scalar_type() == projection_dtype &&
                        lw.o_w.scalar_type() == projection_dtype &&
                        lw.gate_w.scalar_type() == projection_dtype &&
                        lw.up_w.scalar_type() == projection_dtype &&
                        lw.down_w.scalar_type() ==
                            (i == QWEN25VL_VISION_MIXED_BOUNDARY_LAYER
                                 ? at::kHalf : projection_dtype) &&
                        bn.norm1_w.scalar_type() == at::kHalf &&
                        bn.norm2_w.scalar_type() == at::kHalf,
                    "qwen25vl Vision FIXED32 RMSNorm requires exact sorting "
                    "projection weights, FP16 norms, and only block-17 "
                    "down_proj in FP16; mismatch at layer ", i);
            }
        }

        // Optional payload availability is captured with the installed model;
        // never re-query it when planning or replaying a prepared signature.
        // The public exact-RTC capability above remains an independent ABI.
        // Original V32 uses arithmetic to move BF16 partials and changes
        // Inf/NaN semantics. Only the bit-preserving task REF advertises this
        // alias; older REFs must keep the small Vision BASE route.
        small_rmsnorm_capability_ = wall_sorting_.silu_mul &&
                !mixed_precision_enabled_ && !exact_mimc_enabled_ &&
                !layer_group_enabled_ &&
                KernelCache::instance().get_kernel(
                    "wall_base_rmsnorm_v32_bitcopy") != nullptr
            ? rpu_snapshot_rmsnorm_spm_contract(
                  RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V32).capability
            : RPU_RMSNORM_CAP_BASE;
        small_owner_ring_norm_available_ =
            KernelCache::instance().get_kernel(
                "wall_base_vision_owner_ring_norm_m672_h1280") != nullptr &&
            KernelCache::instance().get_program(
                "wall_base_vision_owner_ring_norm_m672_h1280") != nullptr;
        small_down_owner_norm_available_ =
            KernelCache::instance().get_kernel(
                "wall_base_vision_down_owner_ring_norm_m672_h1280") != nullptr &&
            KernelCache::instance().get_program(
                "wall_base_vision_down_owner_ring_norm_m672_h1280") != nullptr;
        small_qk_rope_available_ = KernelCache::instance().has_loaded(
            KernelId::WALL_BASE_VISION_QK_ROPE_2D_SPM);
        small_ragged_sdpa_available_ = KernelCache::instance().has_loaded(
            KernelId::WALL_BASE_VISION_RAGGED_SDPA);
        invalidate_model_state();
    }

    void set_per_window_sdpa(bool enabled) {
        TORCH_CHECK(!z1_adopted_ && !z1_bound_ &&
                        z1_prepared_num_patches_ == 0,
                    "qwen25vl per-window SDPA cannot be enabled on a Z1 handle");
        if (enabled) {
            TORCH_CHECK(!linear_acc32_, "Wall Vision ACC32 does not admit per-window execution");
            TORCH_CHECK(num_layers() > 0,
                        "qwen25vl per-window SDPA must be set after weights");
            TORCH_CHECK(fullatt_block_indexes_ ==
                            std::vector<int64_t>({7, 15, 23, 31}),
                        "qwen25vl per-window SDPA requires the controlled "
                        "32-layer window profile [7, 15, 23, 31]");
            TORCH_CHECK(
                        !mixed_precision_enabled_ &&
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
    // set_merger_weights — post-encoder merger weight storage (Task 1: plumb
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
                    "qwen25vl_vision_set_rope: freq_cos / freq_sin must be on device");
        TORCH_CHECK(freq_cos.is_contiguous() && freq_sin.is_contiguous(),
                    "qwen25vl_vision_set_rope: freq_cos / freq_sin must be contiguous");

        freq_cos_ = freq_cos;
        freq_sin_ = freq_sin;
        max_hw_ = freq_cos.size(0);

        if (!position_idx_keepalive_.defined()) {
            TORCH_CHECK(max_keepalive_seq_ > 0,
                        "qwen25vl_vision_set_rope: bind KV capacity before RoPE tables");
            position_idx_keepalive_ = at::empty(
                {max_keepalive_seq_, 2},
                at::TensorOptions().dtype(at::kShort).device(at::kPrivateUse1));
            position_idx_keepalive_.zero_();
        }

        has_rope_ = true;
        invalidate_model_state();
    }

    void set_execution_envelope(int64_t capacity, int64_t chunk) {
        TORCH_CHECK(capacity > 0,
                    "qwen25vl_vision KV capacity must be positive");
        TORCH_CHECK(!position_idx_keepalive_.defined() ||
                        capacity == max_keepalive_seq_,
                    "qwen25vl_vision position capacity is cold-bound; rebuild the model");
        set_chunk_envelope(capacity, chunk);
        max_keepalive_seq_ = capacity;
    }

    int64_t max_keepalive_seq() const { return max_keepalive_seq_; }
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
        std::optional<at::Tensor> cu_window_seqlens = std::nullopt,
        at::IntArrayRef planned_stage_descriptor = {},
        int64_t valid_patches_per_image = 0,
        std::optional<at::Tensor> full_mask = std::nullopt,
        bool position_published = false)
    {
        TORCH_CHECK(!z1_adopted_,
                    "Qwen25VLVisionModel::forward: physical Z1 lease is active; "
                    "use the coordinator-owned Z1 dispatch");
        return forward_impl(input, k_caches, v_caches, num_patches_in,
                            std::move(window_mask), image_batch_count,
                            std::move(cu_window_seqlens),
                            /*z1_dispatch=*/false,
                            planned_stage_descriptor,
                            valid_patches_per_image, std::move(full_mask),
                            position_published);
    }

    at::Tensor forward_impl(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches_in,
        std::optional<at::Tensor> window_mask,
        int64_t image_batch_count,
        std::optional<at::Tensor> cu_window_seqlens,
        bool z1_dispatch,
        at::IntArrayRef planned_stage_descriptor,
        int64_t valid_patches_per_image = 0,
        std::optional<at::Tensor> full_mask = std::nullopt,
        bool position_published = false)
    {
        TORCH_CHECK(z1_dispatch == z1_bound_,
                    "Qwen25VLVisionModel::forward_impl: Z1 dispatch/binding mismatch");
        TORCH_CHECK(z1_dispatch || !planned_stage_descriptor.empty(),
                    "Qwen2.5-VL Vision production forward requires its native A6 descriptor");
        TORCH_CHECK(num_layers() > 0,
                    "Qwen25VLVisionModel::forward called before set_weights");
        TORCH_CHECK(has_rope_,
                    "Qwen25VLVisionModel::forward called before set_rope_tables");
        TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                    "Qwen25VLVisionModel::forward: input must be on device");
        TORCH_CHECK(input.is_contiguous(),
                    "Qwen25VLVisionModel::forward: input must be contiguous");
        TORCH_CHECK(input.dim() == 3,
                    "Qwen25VLVisionModel::forward: input must be 3D [1,N,H], got ",
                    input.dim(), "D");
        TORCH_CHECK(num_patches_in > 0 && num_patches_in <= max_keepalive_seq_,
                    "Qwen25VLVisionModel::forward: num_patches=", num_patches_in,
                    " must be in (0, MAX_KEEPALIVE_SEQ=", max_keepalive_seq_, "]");
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
        mixed_precision_ = mixed_precision_enabled_;
        window_mask_present_ = window_mask.has_value() && window_mask->defined();
        valid_patches_per_image_ = valid_patches_per_image;
        validate_acc32_base_input();
        TORCH_CHECK(valid_patches_per_image_ == 0 || padded_pair_enabled(),
                    "padded Vision requires exact uniform B2/M672/valid324 "
                    "with fused merger and cold SiLU");
        TORCH_CHECK(padded_pair_enabled() ==
                        (full_mask.has_value() && full_mask->defined()),
                    "padded Vision requires its explicit full-attention mask; "
                    "ordinary Vision must not supply one");
        refresh_small_ragged_mask(window_mask, /*planning=*/false);

        const bool has_window_boundaries =
            cu_window_seqlens.has_value() && cu_window_seqlens->defined();
        TORCH_CHECK(!per_window_sdpa_enabled_ || has_window_boundaries,
                    "qwen25vl per-window SDPA requires cu_window_seqlens");
        TORCH_CHECK(!has_window_boundaries || per_window_sdpa_enabled_ ||
                        is_exact_mimc() || small_image_native_window_shape(),
                    "cu_window_seqlens requires a controlled per-window SDPA "
                    "handle or exact MIMC metadata");
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
        }
        if (has_window_boundaries) {
            const at::Tensor& cu = *cu_window_seqlens;
            TORCH_CHECK(cu.device().is_cpu() && cu.scalar_type() == at::kLong &&
                            cu.is_contiguous() && cu.dim() == 1 && cu.numel() >= 2,
                        "qwen25vl cu_window_seqlens must be contiguous CPU int64 "
                        "with at least two boundaries");
            const int64_t* boundaries = cu.data_ptr<int64_t>();
            const int64_t boundary_seq = per_window_sdpa_enabled_
                ? num_patches_in : num_patches_in / image_batch_count_;
            TORCH_CHECK(boundaries[0] == 0 &&
                            boundaries[cu.numel() - 1] == boundary_seq,
                        "qwen25vl cu_window_seqlens must start at 0 and end at "
                        "the attention sequence");
            for (int64_t i = 1; i < cu.numel(); ++i) {
                TORCH_CHECK(boundaries[i] > boundaries[i - 1] &&
                                boundaries[i] % 16 == 0,
                            "qwen25vl cu_window_seqlens must be strictly "
                            "increasing and 16-aligned");
            }
            cu_window_.assign(boundaries, boundaries + cu.numel());
            TORCH_CHECK(per_window_sdpa_enabled_ || is_exact_mimc() ||
                            small_image_native_window_enabled(),
                        "small Vision native windows require exactly four "
                        "64-patch windows");
        }
        if (mixed_precision_) {
            TORCH_CHECK(
                !is_legacy_layer_group(),
                "qwen25vl Vision mixed precision is incompatible with layer "
                "grouping");
            TORCH_CHECK(
                fused_merger_enabled_ && merger_out_hidden_ > 0,
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
            const bool pure_fp16 = (intermediate_size() == 3456 ||
                ((wall_sorting_.fp16_transfer_mask & 1u) && intermediate_size() == 3584))
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
                !is_exact_mimc() || hybrid_w8a16 ||
                    (pure_fp16 && (wall_sorting_.fp16_transfer_mask & 1u)),
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

        // P3b ordinary path: prepare the dense block-diagonal window mask ONCE
        // per forward (stable DDR slot, Route B). Z1 already primed that slot
        // outside the outer Graph and only validates its stable identity here.
        // Upload to SPM still happens per window-layer because sdpa_mask aliases
        // temporary storage. Without a mask every layer is full attention.
        window_mask_present_ = window_mask.has_value() && window_mask->defined();
        if (padded_pair_enabled()) {
            prepare_padded_pair_masks(*window_mask, *full_mask);
        } else if (window_mask_present_ && !z1_dispatch) {
            // Batched (image_batch_count_>1): window layers run per-image SDPA at
            // per_image_ctx, so the prepared mask is [per_image_ctx, per_image_ctx].
            int64_t mask_seq = (image_batch_count_ > 1)
                ? num_patches_in / image_batch_count_ : num_patches_in;
            // Skip the re-prepare (CPU→device upload + stable-slot copy) when the
            // caller passes the same (memoized) mask at the same shape — the slot
            // still holds it (no other same-shape user touches it between forwards).
            const void* wm_ptr = window_mask->data_ptr();
            const bool exact_ragged_extent = small_ragged_sdpa_available_ && mask_seq == 324;
            const auto& mask_version = window_mask->unsafeGetTensorImpl()->version_counter();
            const bool same_versioned_owner = mask_version.enabled() &&
                small_ragged_prepared_mask_ref_.defined() &&
                small_ragged_prepared_mask_ref_.unsafeGetTensorImpl() == window_mask->unsafeGetTensorImpl() &&
                small_ragged_prepared_mask_version_ == mask_version.current_version();
            if (wm_ptr != last_window_mask_ptr_ || mask_seq != last_mask_seq_ ||
                (exact_ragged_extent && !same_versioned_owner)) {
                prepared_window_mask_ = sdpa_prepare_mask(
                    window_mask, /*is_causal=*/false, mask_seq, mask_seq,
                    sdpa_stable_mask_cache());
                last_window_mask_ptr_ = wm_ptr;
                last_mask_seq_ = mask_seq;
                if (exact_ragged_extent) {
                    small_ragged_prepared_mask_ref_ = *window_mask;
                    small_ragged_prepared_mask_version_ = mask_version.enabled()
                        ? mask_version.current_version() : 0;
                }
            }
        } else if (z1_dispatch) {
            TORCH_CHECK(prepared_window_mask_.mask_type == 4 &&
                            prepared_window_mask_.ddr_tensor.defined(),
                        "Qwen25VLVisionModel Z1 prepared window mask is unavailable");
        }

        // The adapter's CPU copy_ already publishes its private position table.
        // Public callers default to flushing: the exposed keepalive can have
        // untracked CPU-alias writes. Z1 publishes during external priming.
        if (!z1_dispatch && !position_published) {
            rpu_ddr_flush_force_sized(
                position_idx_keepalive_.data_ptr<int16_t>(),
                static_cast<size_t>(num_patches_in * 2 * sizeof(int16_t)));
        }

        // Generic SISC/MISC stays single-chunk. Specialized implementations
        // expose one framework compute chunk per image; exact MIMC supplies a
        // packed QKV plan through FMB's KV_FIRST stage.
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

        at::Tensor result;
        if (z1_dispatch) {
            const int64_t per_image = num_patches_in / image_batch_count_;
            std::vector<ChunkInfo> patch_chunks{
                {0, 0, num_patches_in, num_patches_in}};
            std::vector<FmbExecutionSpan> image_spans;
            image_spans.reserve(image_batch_count_);
            for (int64_t image = 0; image < image_batch_count_; ++image) {
                const int64_t offset = image * per_image;
                image_spans.push_back({offset, per_image, image});
            }
            FmbStageBoundaryPolicies boundary_policies{};
            if (is_legacy_layer_group()) {
                boundary_policies.qkv = FmbSpanBoundaryPolicy::KEEP_LOCAL;
                boundary_policies.compute = FmbSpanBoundaryPolicy::KEEP_LOCAL;
            } else if (is_exact_mimc()) {
                boundary_policies.compute = FmbSpanBoundaryPolicy::KEEP_LOCAL;
            }
            result = run_all_layers(
                input, k_caches, v_caches,
                /*mask=*/std::nullopt, /*position=*/0,
                /*is_causal=*/false, patch_chunks, image_spans,
                boundary_policies);
        } else {
            result = run_all_layers(
                input, k_caches, v_caches,
                /*mask=*/std::nullopt, /*position=*/0,
                /*is_causal=*/false, /*planned_chunk_size=*/0,
                planned_stage_descriptor);
        }

        if (get_debug_export() && per_layer_debug_buf_.defined()) {
            // Publish the stable live buffer as one tensor. The validation
            // harness snapshots it after each forward. get_debug_tensor()
            // performs the post-capture device-to-CPU synchronization.
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
        auto prepared = prepare_spm_pipeline_component(
            layout, compose_fmb_default_three_stage_chunk_plan(
                        layout, num_patches, /*position=*/0,
                        ChunkMode::SEQUENTIAL));
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
        TORCH_CHECK(!linear_acc32_, "Wall Vision ACC32 does not admit Z1 pipelines");
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
                           /*z1_dispatch=*/true,
                           /*planned_stage_descriptor=*/{});
    }

    void check_z1_destroy_allowed() const {
        TORCH_CHECK(!z1_adopted_ && !z1_bound_ && !z1_inputs_primed_ &&
                        z1_prepared_num_patches_ == 0,
                    "cannot destroy the Qwen25VL Vision handle while its Z1 "
                    "layout is prepared, primed, or active; clear the outer "
                    "GraphCache and unprepare Z1 first");
    }

protected:
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        // Existing window geometry is native-owned; keep the diagnostic
        // snapshot bounded without narrowing ordinary forward admission.
        if (cu_window_.size() > 4096) return {};
        return capture_kvinsert_cost_layout_fields(
            current_num_patches_, image_batch_count_,
            schedule_, implementation_,
            mixed_precision_, window_mask_present_,
            valid_patches_per_image_,
            small_ragged_mask_exact_, small_ragged_mask_rows_, cu_window_);
    }

    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers = num_layers();
        cfg.preload_fn = static_cast<void(FusedModelBase::*)()>(
                             &Qwen25VLVisionModel::emit_preload_weights);
        if (is_exact_mimc()) {
            cfg.kv_first_fn =
                static_cast<void(FusedModelBase::*)(int, const ChunkInfo&)>(
                    &Qwen25VLVisionModel::emit_exact_mimc_qkv_body);
            cfg.kv_first_chunk_plan_fn =
                static_cast<ChunkPlan(FusedModelBase::*)(const ChunkPlan&)>(
                    &Qwen25VLVisionModel::plan_exact_mimc_qkv_chunks);
        }
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
            cfg.chunk_mode = ChunkMode::KV_FIRST;
            cfg.attention_policy =
                AttentionExecutionPolicy::SPM_KV_BY_MHA;
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
        // Production generic forwards carry a COMPLETE per-site descriptor;
        // its ATTENTION routes are the sole dispatch authority. The descriptor-
        // free Z1 path retains its established DDR implementation.
        cfg.attention_policy = ctx().has_complete_physical_manifest()
            ? AttentionExecutionPolicy::AUTO
            : AttentionExecutionPolicy::DDR_KV;
        return cfg;
    }

    bool subclass_chunk_size_valid(
        int64_t cs, int64_t seq_len, int64_t /*position*/) const override {
        TORCH_INTERNAL_ASSERT(image_batch_count_ > 0);
        const int64_t native_rows = is_generic()
            ? seq_len : seq_len / image_batch_count_;
        return native_rows > 0 && native_rows <= UINT16_MAX &&
            cs <= UINT16_MAX && cs == Align(native_rows, int64_t{16});
    }

    void consume_manifest_route(
        FmbRouteFamily family, int64_t site_id, int64_t selector,
        int64_t invocation = 0,
        at::IntArrayRef resolved_arguments = {},
        int64_t resolved_flags = 0) {
        if (!ctx().has_complete_physical_manifest()) return;
        ctx().consume_physical_route(
            family, site_id, selector, resolved_flags,
            resolved_arguments, invocation);
    }

    struct GenericAttentionGeometry {
        int64_t site_id;
        int64_t invocation;
        int64_t batch;
        int64_t seq_q;
        int64_t seq_k;
        int64_t mask_domain;
    };

    bool generic_attention_layer_is_full(int64_t layer) const {
        return !(window_mask_present_ || per_window_sdpa_enabled_) ||
            std::find(fullatt_block_indexes_.begin(),
                      fullatt_block_indexes_.end(), layer) !=
                fullatt_block_indexes_.end();
    }

    int64_t generic_shared_attention_mask_domain() const {
        if (padded_pair_enabled()) return int64_t{1} << 4;
        if (small_image_ragged_sdpa_enabled(current_num_patches_)) return int64_t{1};
        int64_t domain = 0;
        for (int64_t layer = 0; layer < num_layers(); ++layer) {
            const int mask_type =
                generic_attention_layer_is_full(layer) ||
                    per_window_sdpa_enabled_
                ? 0 : 4;
            domain |= int64_t{1} << mask_type;
        }
        return domain == 0 ? int64_t{1} : domain;
    }

    std::vector<GenericAttentionGeometry>
    generic_attention_geometries(
        const FmbThreeStageChunkPlan& plan) const {
        std::vector<GenericAttentionGeometry> sites;
        bool any_full = false;
        bool any_window = false;
        for (int64_t layer = 0; layer < num_layers(); ++layer) {
            if (generic_attention_layer_is_full(layer)) {
                any_full = true;
            } else {
                any_window = true;
            }
        }
        const int64_t shared_mask_domain =
            generic_shared_attention_mask_domain();
        for (const ChunkInfo& chunk : plan.compute.chunks) {
            if (small_image_native_window_enabled()) {
                if (any_window) {
                    sites.push_back({QWEN25_WINDOW_DDR_ATTN_SITE, chunk.idx,
                                     image_batch_count_ * 4, 64, 64, int64_t{1}});
                }
                if (any_full) {
                    sites.push_back({QWEN25_MISC_DDR_ATTN_SITE, chunk.idx,
                                     image_batch_count_, 256, 256, int64_t{1}});
                }
            } else if (schedule_ == VisionSchedule::MISC) {
                const int64_t per_image = chunk.len / image_batch_count_;
                for (int64_t image = 0; image < image_batch_count_; ++image) {
                    sites.push_back({
                        QWEN25_MISC_DDR_ATTN_SITE,
                        chunk.idx * max_keepalive_seq_ + image,
                        /*batch=*/1, per_image, per_image,
                        shared_mask_domain});
                }
            } else if (per_window_sdpa_enabled_ && any_window) {
                int64_t invocation = 0;
                for (size_t window = 0;
                     window + 1 < cu_window_.size(); ++window) {
                    const int64_t rows =
                        cu_window_[window + 1] - cu_window_[window];
                    const int64_t query_chunk = chunk.len > 256
                        ? QWEN25VL_VISION_SDPA_QUERY_CHUNK : chunk.len;
                    for (int64_t off = 0; off < rows; off += query_chunk) {
                        sites.push_back({
                            QWEN25_WINDOW_DDR_ATTN_SITE,
                            chunk.idx * max_keepalive_seq_ +
                                invocation++,
                            /*batch=*/1, std::min(query_chunk, rows - off),
                            rows, /*MASK_NONE=*/1});
                    }
                }
                if (!any_full) continue;
                const int64_t query_chunk = chunk.len > 256
                    ? QWEN25VL_VISION_SDPA_QUERY_CHUNK : chunk.len;
                for (int64_t off = 0, invocation = 0;
                     off < chunk.len;
                     off += query_chunk, ++invocation) {
                    sites.push_back({
                        QWEN25_STANDARD_DDR_ATTN_SITE,
                        chunk.idx * max_keepalive_seq_ +
                            invocation,
                        /*batch=*/1,
                        std::min(query_chunk, chunk.len - off), chunk.len,
                        /*MASK_NONE=*/1});
                }
            } else {
                const int64_t query_chunk = small_image_single_raw_query_enabled()
                    ? chunk.len : (chunk.len > 256
                        ? QWEN25VL_VISION_SDPA_QUERY_CHUNK : chunk.len);
                for (int64_t off = 0, invocation = 0;
                     off < chunk.len;
                     off += query_chunk, ++invocation) {
                    sites.push_back({
                        QWEN25_STANDARD_DDR_ATTN_SITE,
                        chunk.idx * max_keepalive_seq_ +
                            invocation,
                        /*batch=*/1,
                        std::min(query_chunk, chunk.len - off), chunk.len,
                        shared_mask_domain});
                }
            }
        }
        return sites;
    }

    std::vector<int64_t> generic_attention_arguments(
        const GenericAttentionGeometry& site) const {
        return {
            site.batch, site.seq_q, site.seq_k,
            num_q_heads(), num_q_heads(), head_dim(), NUM_CORES,
            site.mask_domain, static_cast<int64_t>(schedule_),
            per_window_sdpa_enabled_ ? 1 : 0};
    }

    bool generic_attention_raw_valid(
        const GenericAttentionGeometry& site) const {
        // MISC stores image-local K/V segments back-to-back in one packed
        // buffer. Raw K uses an align16(seq_k) stride, so an unaligned image
        // extent would overlap the next image's segment.
        if (site.site_id == QWEN25_MISC_DDR_ATTN_SITE &&
            image_batch_count_ > 1 && site.seq_k % 16 != 0) {
            return false;
        }
        for (const int mask_type : {0, 1, 4}) {
            if ((site.mask_domain & (int64_t{1} << mask_type)) == 0) {
                continue;
            }
            if (!sdpa_by_mha_spm_is_valid(
                    site.batch, site.seq_q, site.seq_k,
                    num_q_heads(), num_q_heads(), head_dim(), NUM_CORES,
                    mask_type)) {
                return false;
            }
        }
        return site.mask_domain != 0;
    }

    bool generic_attention_site_raw_valid(
        const FmbThreeStageChunkPlan& plan, int64_t site_id) const {
        bool found = false;
        for (const GenericAttentionGeometry& site :
             generic_attention_geometries(plan)) {
            if (site.site_id != site_id) continue;
            found = true;
            if (!generic_attention_raw_valid(site)) return false;
        }
        return found;
    }

    int64_t generic_attention_physical_site_id(
        int64_t ddr_site_id, AttentionExecutionPolicy policy) const {
        if (policy == AttentionExecutionPolicy::DDR_KV) {
            return ddr_site_id;
        }
        TORCH_CHECK(
            policy == AttentionExecutionPolicy::SPM_KV_BY_MHA,
            "Qwen2.5-VL Vision generic attention has an invalid policy");
        if (ddr_site_id == QWEN25_MISC_DDR_ATTN_SITE) {
            return QWEN25_MISC_RAW_ATTN_SITE;
        }
        if (ddr_site_id == QWEN25_WINDOW_DDR_ATTN_SITE) {
            return QWEN25_WINDOW_RAW_ATTN_SITE;
        }
        TORCH_CHECK(
            ddr_site_id == QWEN25_STANDARD_DDR_ATTN_SITE,
            "Qwen2.5-VL Vision generic attention has an invalid site");
        return QWEN25_STANDARD_RAW_ATTN_SITE;
    }

    AttentionExecutionPolicy generic_attention_policy(
        const GenericAttentionGeometry& site,
        int actual_mask_type) {
        TORCH_CHECK(
            (site.mask_domain & (int64_t{1} << actual_mask_type)) != 0,
            "Qwen2.5-VL Vision attention mask escaped its descriptor domain");
        if (!ctx().has_complete_physical_manifest()) {
            return AttentionExecutionPolicy::DDR_KV;
        }
        const bool raw_requested = ctx().attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const AttentionExecutionPolicy expected_policy =
            raw_requested && generic_attention_site_raw_valid(
                ctx().stage_plan, site.site_id)
            ? AttentionExecutionPolicy::SPM_KV_BY_MHA
            : AttentionExecutionPolicy::DDR_KV;
        const int64_t physical_site_id =
            generic_attention_physical_site_id(
                site.site_id, expected_policy);
        const FmbRouteManifestEntry& route = ctx().find_physical_route(
            FmbRouteFamily::ATTENTION, physical_site_id, site.invocation);
        TORCH_CHECK(
            fmb_attention_execution_policy(route) == expected_policy,
            "Qwen2.5-VL Vision attention descriptor policy drifted");
        return expected_policy;
    }

    void consume_generic_attention_route(
        const GenericAttentionGeometry& site,
        int actual_mask_type,
        AttentionExecutionPolicy expected_policy,
        int64_t declared_site_id) {
        TORCH_CHECK(
            generic_attention_physical_site_id(
                site.site_id, expected_policy) == declared_site_id &&
                (site.mask_domain &
                 (int64_t{1} << actual_mask_type)) != 0,
            "Qwen2.5-VL Vision attention site or mask escaped its "
            "descriptor domain");
        if (!ctx().has_complete_physical_manifest()) {
            TORCH_CHECK(
                expected_policy == AttentionExecutionPolicy::DDR_KV,
                "Qwen2.5-VL Vision legacy execution requires DDR attention");
            return;
        }
        const FmbRouteManifestEntry& route = ctx().find_physical_route(
            FmbRouteFamily::ATTENTION, declared_site_id, site.invocation);
        TORCH_CHECK(
            fmb_attention_execution_policy(route) == expected_policy,
            "Qwen2.5-VL Vision attention descriptor policy drifted");
        ctx().consume_physical_route(
            FmbRouteFamily::ATTENTION, declared_site_id, route.selector,
            route.flags, generic_attention_arguments(site), site.invocation);
    }

    RpuRmsNormSpmRoute consume_exact_rmsnorm_route(
        int64_t site_id, int64_t rows, int64_t cols,
        int64_t invocation) {
        const RpuRmsNormSpmRoute route =
            rpu_resolve_rmsnorm_spm_route(
                rmsnorm_capability_, rows, cols);
        if (rmsnorm_capability_ == RPU_RMSNORM_CAP_BASE) {
            return route;
        }
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "qwen25vl Vision RMSNorm route requires a COMPLETE "
                    "physical descriptor");
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION, site_id,
            static_cast<int64_t>(route), /*resolved_flags=*/0,
            {rows, cols, static_cast<int64_t>(rmsnorm_capability_)},
            invocation);
        return route;
    }

    RpuMixedRmsNormSpmRoute consume_mixed_rmsnorm_route(
        int64_t site_id, int64_t rows, int64_t cols,
        int64_t invocation) {
        const RpuMixedRmsNormSpmRoute route =
            rpu_resolve_mixed_rmsnorm_spm_route(rows, cols);
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "qwen25vl Vision mixed RMSNorm route requires a COMPLETE "
                    "physical descriptor");
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION, site_id,
            static_cast<int64_t>(route), /*resolved_flags=*/0,
            {rows, cols}, invocation);
        return route;
    }

    bool padded_pair_enabled() const {
        if (valid_patches_per_image_ != 324 || current_num_patches_ != 672 ||
            image_batch_count_ != 2 || schedule_ != VisionSchedule::MISC ||
            !is_generic() || !merger_active() || !wall_sorting_.silu_mul ||
            !window_mask_present_ || mixed_precision_ || per_window_sdpa_enabled_ ||
            z1_bound_ || get_debug_export() || num_layers() != 32 ||
            layer_weights_.size() != 32 || hidden_size() != 1280 ||
            num_q_heads() != 16 || num_kv_heads() != 16 || head_dim() != 80 ||
            fullatt_block_indexes_ != std::vector<int64_t>({7, 15, 23, 31})) {
            return false;
        }
        const auto dtype = layer_weights_.front().q_w.scalar_type();
        if (!((dtype == at::kHalf && (intermediate_size() == 3456 ||
                                     intermediate_size() == 3584)) ||
              (dtype == at::kChar && intermediate_size() == 3584))) return false;
        return std::all_of(layer_weights_.begin(), layer_weights_.end(),
            [dtype](const LayerWeights& lw) {
                return lw.q_w.scalar_type() == dtype && lw.k_w.scalar_type() == dtype &&
                    lw.v_w.scalar_type() == dtype && lw.o_w.scalar_type() == dtype &&
                    lw.gate_w.scalar_type() == dtype && lw.up_w.scalar_type() == dtype &&
                    lw.down_w.scalar_type() == dtype;
            });
    }

    static void validate_padded_pair_mask(const at::Tensor& mask, bool full) {
        TORCH_CHECK(mask.defined() && mask.device().is_cpu() &&
                        mask.scalar_type() == at::kHalf && mask.is_contiguous() &&
                        mask.dim() == 2 && mask.size(0) == 336 && mask.size(1) == 336,
                    "padded Vision masks must be contiguous CPU fp16 [336,336]");
        const auto* data = mask.data_ptr<c10::Half>();
        for (int64_t q = 0; q < 336; ++q) {
            for (int64_t k = 0; k < 336; ++k) {
                const float value = static_cast<float>(data[q * 336 + k]);
                const bool excluded = std::isinf(value) && value < 0;
                if ((q < 324) != (k < 324)) {
                    TORCH_CHECK(excluded, "padded Vision mask must isolate valid and dummy rows");
                } else if (q >= 324 || full || q == k) {
                    TORCH_CHECK(value == 0, "padded Vision mask requires finite full/dummy/self attention");
                } else {
                    TORCH_CHECK(value == 0 || excluded,
                                "padded Vision window mask must contain only zero and -inf");
                }
            }
        }
    }

    void prepare_padded_pair_masks(const at::Tensor& window, const at::Tensor& full) {
        const std::array<at::Tensor, 2> masks{window, full};
        for (size_t i = 0; i < masks.size(); ++i) {
            TORCH_CHECK(masks[i].defined() && masks[i].device().is_cpu() &&
                            masks[i].scalar_type() == at::kHalf && masks[i].is_contiguous() &&
                            masks[i].dim() == 2 && masks[i].size(0) == 336 && masks[i].size(1) == 336,
                        "padded Vision mask metadata changed outside its owner");
            const auto& version = masks[i].unsafeGetTensorImpl()->version_counter();
            const bool versioned = version.enabled();
            const uint32_t current_version = versioned ? version.current_version() : 0;
            if (!versioned || !pair_mask_refs_[i].defined() ||
                masks[i].data_ptr() != pair_mask_refs_[i].data_ptr() ||
                current_version != pair_mask_versions_[i]) {
                validate_padded_pair_mask(masks[i], /*full=*/i == 1);
                // Separate ordinals also isolate this owner from ordinary
                // same-shape masks (ordinal0) on this shared Vision handle.
                prepared_pair_masks_[i] = sdpa_prepare_mask(
                    masks[i], false, 336, 336, sdpa_stable_mask_cache(), i + 1);
                pair_mask_refs_[i] = masks[i];
                pair_mask_versions_[i] = current_version;
            }
        }
    }

    std::vector<int64_t> sorting_schedule_arguments() const {
        std::vector<int64_t> args{
            static_cast<int64_t>(wall_sorting_.fp16_transfer_mask),
            static_cast<int64_t>(wall_sorting_.vision_row_reuse),
            static_cast<int64_t>(wall_sorting_.silu_mul)};
        if (padded_pair_enabled()) {
            args.insert(args.end(), {324, 336, 2, /*MASK_2D domain=*/16});
        }
        if (linear_acc32_) args.push_back(32);
        return args;
    }

    bool small_image_row_weight_outer_enabled(int64_t rows) const {
        // Reuse the sorting row kernels' dynamic M loop, keeping the ordinary
        // replicated residuals, image-local attention and native SPM plan.
        // 224px is one packed 2x256 or 3x256 pass; 256px is resized to 252px and remains
        // a single 324-patch pass per image. The latter cannot share raw K/V
        // storage because its image extent is not 16-aligned.
        if (!wall_sorting_.vision_row_reuse || !is_generic() ||
            mixed_precision_ || per_window_sdpa_enabled_ || z1_bound_ ||
            num_layers() != 32 || layer_weights_.size() != 32 ||
            hidden_size() != 1280 ||
            num_q_heads() != 16 || head_dim() != 80 ||
            intermediate_size() != 3584 ||
            !((rows == 256 * image_batch_count_ &&
               (image_batch_count_ == 2 || image_batch_count_ == 3) &&
               current_num_patches_ == rows) ||
              (rows == 324 && image_batch_count_ == 1 &&
               current_num_patches_ == 324) ||
              (rows == 672 && padded_pair_enabled()))) {
            return false;
        }
        const auto weight_dtype = layer_weights_.front().q_w.scalar_type();
        if (weight_dtype != at::kHalf && weight_dtype != at::kChar) return false;
        const bool fp16 = weight_dtype == at::kHalf;
        if (fp16 && !(wall_sorting_.fp16_transfer_mask & 1u)) return false;
        const auto dtype = fp16 ? at::kHalf : at::kChar;
        return std::all_of(
            layer_weights_.begin(), layer_weights_.end(),
            [dtype, fp16](const LayerWeights& lw) {
                return lw.q_w.scalar_type() == dtype &&
                    lw.k_w.scalar_type() == dtype &&
                    lw.v_w.scalar_type() == dtype &&
                    lw.o_w.scalar_type() == dtype &&
                    lw.gate_w.scalar_type() == dtype &&
                    lw.up_w.scalar_type() == dtype &&
                    lw.down_w.scalar_type() == dtype &&
                    lw.o_ws.defined() == !fp16 &&
                    lw.down_ws.defined() == !fp16;
            });
    }

    // The shared layout proof above also admits native windows, RoPE and
    // activation-only owner norms. Only these projection kernels are ACC16-only.
    bool small_image_acc16_row_projection_enabled(int64_t rows) const {
        return !linear_acc32_ && small_image_row_weight_outer_enabled(rows);
    }

    bool small_image_acc16_gate_up_projection_enabled(int64_t rows) const {
        return !linear_acc32_ && small_image_gate_up_enabled(rows);
    }

    bool small_image_fp16_row_profile(int64_t rows) const {
        // Projection outputs and all residual/norm slabs remain FP16. Require
        // the padded row-reuse layout and every layer's complete, unquantized
        // weights before sharing activation-only payloads with W8 Vision.
        return wall_sorting_.silu_mul &&
            small_image_row_weight_outer_enabled(rows) &&
            layer_weights_.front().q_w.scalar_type() == at::kHalf &&
            std::all_of(layer_weights_.begin(), layer_weights_.end(),
                [](const LayerWeights& lw) {
                    for (const auto* weight : {&lw.q_w, &lw.k_w, &lw.v_w,
                            &lw.o_w, &lw.gate_w, &lw.up_w, &lw.down_w}) {
                        if (!weight->defined() ||
                            weight->scalar_type() != at::kHalf) return false;
                    }
                    for (const auto* scale : {&lw.q_ws, &lw.k_ws, &lw.v_ws,
                            &lw.o_ws, &lw.gate_ws, &lw.up_ws, &lw.down_ws}) {
                        if (scale->defined()) return false;
                    }
                    return true;
                });
    }

    bool small_image_gate_up_enabled(int64_t rows) const {
        // M324 benefits from this single-block launch. Keep packed M768 on
        // separate projections plus SiLU/Mul, which measured faster there.
        // The explicit M672 pair uses two runtime-bounded M352 tiles.
        return (rows == 324 || (rows == 672 && padded_pair_enabled())) &&
            small_image_row_weight_outer_enabled(rows) &&
            wall_sorting_.silu_mul &&
            layer_weights_.front().q_w.scalar_type() == at::kChar &&
            std::all_of(layer_weights_.begin(), layer_weights_.end(),
                [](const LayerWeights& lw) {
                    return lw.gate_ws.defined() && lw.up_ws.defined();
                });
    }

    std::vector<int64_t> small_image_gate_up_arguments(int64_t rows) const {
        return {rows, 1280, 3584, NUM_CORES, 352, 96};
    }

    bool small_image_native_window_shape() const {
        return current_num_patches_ == 256 * image_batch_count_ &&
            (image_batch_count_ == 2 || image_batch_count_ == 3) &&
            window_mask_present_ &&
            small_image_row_weight_outer_enabled(current_num_patches_) &&
            fullatt_block_indexes_ == std::vector<int64_t>({7, 15, 23, 31});
    }

    bool small_image_native_window_enabled() const {
        return small_image_native_window_shape() &&
            cu_window_ == std::vector<int64_t>({0, 64, 128, 192, 256});
    }

    bool small_image_raw_kv_only_enabled() const {
        // These exact cold-opt-in schedules admit only all-site raw-SPM plans.
        // M324 keeps one image per invocation: aligned scratch does not add
        // logical rows or make multiple unaligned images safe to pack together.
        // Debug and other generic shapes retain their durable DDR mirror.
        return !get_debug_export() &&
            (small_image_native_window_enabled() ||
             (current_num_patches_ == 324 && image_batch_count_ == 1 &&
              small_image_resident_mask_enabled()) || padded_pair_enabled());
    }

    bool small_image_resident_mask_enabled() const {
        if (padded_pair_enabled()) return true;
        // The mask DMA is independent of Linear tiling and its I padding.
        // Keep this lifetime change cold-opt-in, including FP16 I3456 with
        // row reuse disabled. Ordinary default generic layouts stay unchanged.
        if (!wall_sorting_.silu_mul || !window_mask_present_ || !is_generic() ||
            mixed_precision_ || per_window_sdpa_enabled_ || z1_bound_ ||
            num_layers() != 32 || layer_weights_.size() != 32 ||
            hidden_size() != 1280 ||
            num_q_heads() != 16 || head_dim() != 80 ||
            fullatt_block_indexes_ != std::vector<int64_t>({7, 15, 23, 31}) ||
            !((current_num_patches_ == 256 * image_batch_count_ &&
               (image_batch_count_ == 2 || image_batch_count_ == 3)) ||
              (current_num_patches_ == 324 && image_batch_count_ == 1)) ||
            small_image_native_window_enabled()) {
            return false;
        }
        const auto weight_dtype = layer_weights_.front().q_w.scalar_type();
        if (weight_dtype != at::kHalf && weight_dtype != at::kChar) return false;
        const bool fp16 = weight_dtype == at::kHalf;
        if (!(intermediate_size() == 3584 ||
              (fp16 && intermediate_size() == 3456))) return false;
        const auto dtype = fp16 ? at::kHalf : at::kChar;
        return std::all_of(
            layer_weights_.begin(), layer_weights_.end(),
            [dtype](const LayerWeights& lw) {
                return lw.q_w.scalar_type() == dtype &&
                    lw.k_w.scalar_type() == dtype &&
                    lw.v_w.scalar_type() == dtype &&
                    lw.o_w.scalar_type() == dtype &&
                    lw.gate_w.scalar_type() == dtype &&
                    lw.up_w.scalar_type() == dtype &&
                    lw.down_w.scalar_type() == dtype;
            });
    }

    bool small_image_single_raw_query_enabled() const {
        // The exact cold M324 raw-only plan supports one Sq=Sk324 launch for
        // both dense window MASK_2D and full MASK_NONE layers. Keep the old
        // 144-row query strips for debug, DDR-capable and all other shapes.
        return current_num_patches_ == 324 && image_batch_count_ == 1 &&
            schedule_ == VisionSchedule::SISC &&
            small_image_raw_kv_only_enabled();
    }

    RpuRmsNormSpmRoute small_image_rmsnorm_route(int64_t rows) const {
        if (small_rmsnorm_capability_ !=
                (RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V32) ||
            !wall_sorting_.silu_mul || !is_generic() || mixed_precision_ ||
            per_window_sdpa_enabled_ || z1_bound_ || get_debug_export() ||
            !merger_active() || num_layers() != 32 ||
            layer_bias_norm_.size() != 32 || num_kv_heads() != 16 ||
            rows != current_num_patches_ ||
            !((rows == 256 * image_batch_count_ &&
               (image_batch_count_ == 2 || image_batch_count_ == 3)) ||
              (rows == 672 && padded_pair_enabled())) ||
            !(small_image_native_window_enabled() ||
              small_image_resident_mask_enabled()) ||
            !merger_ln_q_w_.defined() ||
            merger_ln_q_w_.scalar_type() != at::kHalf ||
            !std::all_of(layer_bias_norm_.begin(), layer_bias_norm_.end(),
                [](const LayerBiasNorm& bn) {
                    return bn.norm1_w.defined() && bn.norm2_w.defined() &&
                        bn.norm1_w.scalar_type() == at::kHalf &&
                        bn.norm2_w.scalar_type() == at::kHalf;
                })) {
            return RpuRmsNormSpmRoute::BASE;
        }
        // M512, M768 and physical M672 are divisible by 32. The third M324
        // image stays BASE; do not round it up or split its row tail.
        return rpu_resolve_rmsnorm_spm_route(
            small_rmsnorm_capability_, rows, hidden_size());
    }

    RpuRmsNormSpmRoute consume_small_image_rmsnorm_route(
        int64_t site_id, int64_t rows, int64_t invocation) {
        const auto route = small_image_rmsnorm_route(rows);
        if (route == RpuRmsNormSpmRoute::BASE) return route;
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "small Vision vector RMSNorm requires a COMPLETE descriptor");
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION, site_id,
            static_cast<int64_t>(route), /*resolved_flags=*/0,
            {rows, hidden_size(), static_cast<int64_t>(small_rmsnorm_capability_)},
            invocation);
        return route;
    }

    bool small_image_owner_ring_norm_enabled(int64_t rows) const {
        // Both weight precisions retire the O partial and raw residual to FP16.
        // Down's ring alone reads the owner84 raw rows; Gate/Up reads the full
        // normalized oproj. Preserve the W8 proof and admit only the explicit
        // FP16 padded row profile, with the identical V32 norm/epsilon contract.
        return small_owner_ring_norm_available_ && rows == 672 &&
            padded_pair_enabled() && eps_ == 1e-6 &&
            (layer_weights_.front().q_w.scalar_type() == at::kChar ||
             small_image_fp16_row_profile(rows)) &&
            small_image_rmsnorm_route(rows) == RpuRmsNormSpmRoute::V32_GRID8;
    }

    std::vector<int64_t> small_image_owner_ring_norm_arguments(int64_t rows) const {
        // ABI version, physical shape/topology, owner and block rows, and
        // owner-only raw output with normalized output aliasing old partial.
        return {1, rows, hidden_size(), NUM_CORES, 84, 12, 1, 1};
    }

    void consume_small_image_owner_ring_norm(int64_t rows, int64_t invocation) {
        TORCH_CHECK(small_image_owner_ring_norm_enabled(rows) &&
                        ctx().has_complete_physical_manifest(),
                    "small Vision owner ring/norm requires its COMPLETE route");
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE, QWEN25_SMALL_OWNER_RING_NORM_SITE,
            static_cast<int64_t>(Qwen25VisionAllReduceRoute::OWNER_RING_NORM_M672),
            0, small_image_owner_ring_norm_arguments(rows), invocation);
    }

    bool small_image_down_owner_norm_enabled(int64_t rows) const {
        // The following O ring is the sole consumer of owner-only raw rows.
        return small_down_owner_norm_available_ &&
            small_image_owner_ring_norm_enabled(rows);
    }

    void validate_small_image_down_owner_plan(
        const FmbThreeStageChunkPlan& plan) const {
        // Cold planning must validate its argument, not a previous ctx plan.
        TORCH_CHECK(plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                        plan.spans.size() == 2,
                    "small Vision Down/next norm requires sequential paired execution");
        for (const auto* stage : {&plan.input, &plan.qkv, &plan.compute}) {
            TORCH_CHECK(stage->plan.chunk_size == 672 && stage->plan.num_chunks == 1 &&
                            stage->chunks.size() == 1 &&
                            stage->chunks[0].idx == 0 && stage->chunks[0].offset == 0 &&
                            stage->chunks[0].len == 672 && stage->chunks[0].kv_seq_len == 672,
                        "small Vision Down/next norm requires one full physical chunk");
        }
        for (int64_t i = 0; i < 2; ++i) {
            TORCH_CHECK(plan.spans[i].offset == i * 336 &&
                            plan.spans[i].len == 336 && plan.spans[i].group_id == i,
                        "small Vision Down/next norm requires two physical image spans");
        }
    }

    void validate_small_image_down_owner_context(int64_t layer, const ChunkInfo& chunk) {
        TORCH_CHECK(small_image_down_owner_norm_enabled(chunk.len) &&
                        ctx().has_complete_physical_manifest() &&
                        layer >= 0 && layer < 32 &&
                        chunk.idx == 0 && chunk.offset == 0 &&
                        chunk.len == 672 && chunk.kv_seq_len == 672,
                    "small Vision Down/next norm requires its COMPLETE layer context");
        validate_small_image_down_owner_plan(ctx().stage_plan);
        const auto& manifest = ctx().physical_manifest();
        TORCH_CHECK(manifest.logical_length == 648 && manifest.physical_length == 672 &&
                        manifest.execution_padding_rows == 24 &&
                        ctx().input_in_spm == (layer > 0) &&
                        ctx().output_to_spm == (layer < 31),
                    "small Vision Down/next norm requires one SPM-resident 32-layer group");
    }

    std::vector<int64_t> small_image_down_owner_norm_arguments(
        int64_t rows, int64_t layer) const {
        // Version, shape/topology, owner/block rows, old-residual norm alias,
        // owner-only raw output, producer and next-gamma layer.
        return {1, rows, hidden_size(), NUM_CORES, 84, 12, 2, 1, layer, layer + 1};
    }

    void consume_small_image_down_owner_norm(int64_t rows, int64_t layer) {
        TORCH_CHECK(small_image_down_owner_norm_enabled(rows) &&
                        ctx().has_complete_physical_manifest() && layer >= 0 && layer < 31,
                    "small Vision Down/next norm requires its nonfinal COMPLETE route");
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE, QWEN25_SMALL_DOWN_OWNER_NORM_SITE,
            static_cast<int64_t>(Qwen25VisionAllReduceRoute::OWNER_DOWN_RING_NORM_M672),
            0, small_image_down_owner_norm_arguments(rows, layer), layer);
    }

    static bool is_small_ragged_mask(const at::Tensor& mask, int64_t rows) {
        if ((rows != 324 && rows != 336) || !mask.defined() ||
            !mask.device().is_cpu() || mask.scalar_type() != at::kHalf ||
            mask.layout() != c10::Layout::Strided || !mask.is_contiguous() ||
            mask.dim() != 2 || mask.size(0) != rows || mask.size(1) != rows)
            return false;
        constexpr std::array<int64_t, 11> ends{0,64,128,144,208,272,288,304,320,324,336};
        const auto* data = mask.data_ptr<c10::Half>();
        for (int64_t q = 0, qw = 0; q < rows; ++q) {
            while (q >= ends[qw + 1]) ++qw;
            for (int64_t k = 0, kw = 0; k < rows; ++k) {
                while (k >= ends[kw + 1]) ++kw;
                const float value = static_cast<float>(data[q * rows + k]);
                if (qw == kw ? value != 0 : !(std::isinf(value) && value < 0))
                    return false;
            }
        }
        return true;
    }

    void refresh_small_ragged_mask(
        const std::optional<at::Tensor>& mask, bool planning) {
        small_ragged_mask_exact_ = false;
        small_ragged_mask_rows_ = 0;
        if (!small_ragged_sdpa_available_ || image_batch_count_ <= 0) return;
        const int64_t rows = current_num_patches_ / image_batch_count_;
        if (rows != 324 && rows != 336) return;
        const size_t slot = rows == 336 ? 0 : 1;
        const bool metadata = mask.has_value() && mask->defined() &&
            mask->device().is_cpu() && mask->scalar_type() == at::kHalf &&
            mask->layout() == c10::Layout::Strided && mask->is_contiguous() &&
            mask->dim() == 2 && mask->size(0) == rows && mask->size(1) == rows;
        if (!metadata || !mask->unsafeGetTensorImpl()->version_counter().enabled()) {
            if (planning) {
                small_ragged_mask_refs_[slot] = at::Tensor();
                small_ragged_mask_valid_[slot] = false;
            }
            return;
        }
        auto& owner = small_ragged_mask_refs_[slot];
        const bool same_owner = owner.defined() &&
            owner.unsafeGetTensorImpl() == mask->unsafeGetTensorImpl() &&
            owner.data_ptr() == mask->data_ptr();
        // A legacy dry planner without the optional mask proof must stay dense.
        if (!planning && !same_owner) return;
        const auto version = mask->unsafeGetTensorImpl()->version_counter().current_version();
        if (!same_owner || version != small_ragged_mask_versions_[slot]) {
            small_ragged_mask_valid_[slot] = is_small_ragged_mask(*mask, rows);
            owner = *mask;
            small_ragged_mask_versions_[slot] = version;
        }
        small_ragged_mask_exact_ = small_ragged_mask_valid_[slot];
        small_ragged_mask_rows_ = rows;
    }

    bool small_image_ragged_sdpa_enabled(int64_t rows) const {
        return small_ragged_sdpa_available_ && small_ragged_mask_exact_ &&
            small_image_qk_rope_enabled(rows) && small_image_resident_mask_enabled() &&
            small_ragged_mask_rows_ == rows / image_batch_count_ &&
            fullatt_block_indexes_ == std::vector<int64_t>({7,15,23,31});
    }

    std::vector<int64_t> small_image_ragged_sdpa_arguments(int64_t rows) const {
        return {1, rows, rows, 16, 16, 80, 8, 16,
                static_cast<int64_t>(schedule_), 0,
                324, 336, 9, 64, 80, 256,
                0,64,128,144,208,272,288,304,320,324, rows};
    }

    void consume_small_image_ragged_sdpa(int64_t rows, int64_t invocation) {
        TORCH_CHECK(small_image_ragged_sdpa_enabled(current_num_patches_) &&
                        rows == small_ragged_mask_rows_ && invocation >= 0 &&
                        invocation < image_batch_count_ && ctx().has_complete_physical_manifest(),
                    "Wall ragged SDPA requires its exact mask owner and COMPLETE geometry");
        validate_small_image_qk_rope_plan(ctx().stage_plan, current_num_patches_);
        ctx().consume_physical_route(FmbRouteFamily::ATTENTION, QWEN25_SMALL_RAGGED_SDPA_SITE,
            static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA), 0,
            small_image_ragged_sdpa_arguments(rows), invocation);
    }

    bool small_image_qk_rope_enabled(int64_t rows) const {
        // Paired RoPE and ragged attention consume FP16 activations, independent
        // of projection weight precision. Preserve the original W8 proof and
        // admit only the cold FP16 row-reuse profile with all 32 layers uniform
        // and no quantization scales. This does not enable W8 GateUp for FP16,
        // or admit the separate row0/I3456 layout without its own validation.
        return small_qk_rope_available_ && has_rope_ && rope_spm_enabled_ &&
            max_hw_ == 128 && freq_cos_.size(1) == 20 && freq_sin_.size(1) == 20 &&
            !get_debug_export() && merger_active() && num_kv_heads() == 16 &&
            (small_image_gate_up_enabled(rows) || small_image_fp16_row_profile(rows)) &&
            small_image_raw_kv_only_enabled() &&
            ((rows == 672 && padded_pair_enabled()) ||
             (rows == 324 && image_batch_count_ == 1 && schedule_ == VisionSchedule::SISC));
    }

    std::vector<int64_t> small_image_qk_rope_arguments(int64_t rows, int64_t offset) const {
        return {1, rows, 1280, 16, 2, 80, 80, NUM_CORES, 128, offset,
                (rows + 127) / 128, max_hw_, image_batch_count_, 324,
                rows == 672 ? 336 : 324};
    }

    void validate_small_image_qk_rope_plan(
        const FmbThreeStageChunkPlan& plan, int64_t rows) const {
        TORCH_CHECK(small_image_qk_rope_enabled(rows) &&
                        plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                        plan.spans.size() == static_cast<size_t>(image_batch_count_),
                    "Wall Vision paired RoPE requires its exact sequential image plan");
        // Resolved stage size is the actual row extent. M324's separate A6
        // QKV/compute capacity336 does not create twelve extra logical rows.
        for (const auto* stage : {&plan.input, &plan.qkv, &plan.compute}) {
            TORCH_CHECK(stage->plan.chunk_size == rows && stage->plan.num_chunks == 1 &&
                            stage->chunks.size() == 1 && stage->chunks[0].idx == 0 &&
                            stage->chunks[0].offset == 0 && stage->chunks[0].len == rows &&
                            stage->chunks[0].kv_seq_len == rows,
                        "Wall Vision paired RoPE requires one full resolved chunk");
        }
        const int64_t span_rows = rows == 672 ? 336 : 324;
        for (int64_t i = 0; i < image_batch_count_; ++i)
            TORCH_CHECK(plan.spans[i].offset == i * span_rows &&
                            plan.spans[i].len == span_rows && plan.spans[i].group_id == i,
                        "Wall Vision paired RoPE image spans changed");
    }

    void consume_small_image_qk_rope(int64_t rows, int64_t offset, int64_t invocation) {
        TORCH_CHECK(small_image_qk_rope_enabled(rows) && offset == 0 && invocation == 0 &&
                        ctx().has_complete_physical_manifest(),
                    "Wall Vision paired RoPE requires its COMPLETE chunk context");
        validate_small_image_qk_rope_plan(ctx().stage_plan, rows);
        ctx().consume_physical_route(FmbRouteFamily::ROPE, QWEN25_SMALL_QK_ROPE_SPM_SITE,
            static_cast<int64_t>(Qwen25VisionRopeRoute::PAIRED_QK_2D_SPM),
            FMB_ROUTE_FLAG_ROPE_TABLE_SPM, small_image_qk_rope_arguments(rows, offset), invocation);
    }

    int64_t small_image_schedule_flags() const {
        return (small_image_resident_mask_enabled() ? 1 : 0) |
            (small_image_native_window_enabled() ? 2 : 0) |
            (small_image_raw_kv_only_enabled() ? 4 : 0) |
            (small_image_single_raw_query_enabled() ? 32 : 0) |
            (padded_pair_enabled() ? 64 : 0) |
            (small_image_rmsnorm_route(current_num_patches_) !=
                 RpuRmsNormSpmRoute::BASE ? 128 : 0) |
            (small_image_owner_ring_norm_enabled(current_num_patches_) ? 256 : 0) |
            (small_image_down_owner_norm_enabled(current_num_patches_) ? 512 : 0) |
            (small_image_qk_rope_enabled(current_num_patches_) ? 1024 : 0) |
            (small_image_ragged_sdpa_enabled(current_num_patches_) ? 2048 : 0);
    }

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout, int64_t physical_len,
        int64_t logical_len, int64_t position) const override {
        (void)position;
        TORCH_CHECK(!padded_pair_enabled() || (physical_len == 672 && logical_len == 648),
                    "padded Vision descriptor requires logical648/physical672");
        if (small_image_down_owner_norm_enabled(physical_len)) {
            validate_small_image_down_owner_plan(plan);
        }
        if (small_image_qk_rope_enabled(physical_len)) {
            validate_small_image_qk_rope_plan(plan, physical_len);
        }
        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = position + logical_len;
        manifest.kv_insert_physical_rows = physical_len;
        manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation = mixed_precision_
            ? FmbLinearAccumulationPolicy::MIXED_BY_SITE
            : (linear_acc32_ ? FmbLinearAccumulationPolicy::ACC32
                             : FmbLinearAccumulationPolicy::ACC16);

        auto append = [&](FmbRouteFamily family, int64_t site_id,
                          int64_t selector, int64_t invocation = 0,
                          std::vector<int64_t> arguments = {},
                          int64_t flags = 0) {
            manifest.routes.push_back({
                site_id, family, selector, flags,
                std::move(arguments), invocation});
        };
        append(FmbRouteFamily::GRAPH_SCHEDULE, QWEN25_WALL_SORTING_SITE,
               1, 0, sorting_schedule_arguments(),
               small_image_schedule_flags());
        auto append_linear = [&](int64_t site_id, int64_t invocation = 0,
                                 std::vector<int64_t> arguments = {}) {
            append(FmbRouteFamily::LINEAR, site_id,
                   qwen25_vision_linear_route(), invocation,
                   std::move(arguments));
        };
        auto append_ring = [&](int64_t site_id, int64_t rows, int64_t cols,
                               int64_t invocation = 0) {
            append(FmbRouteFamily::ALL_REDUCE, site_id,
                   qwen25_vision_ring_route(rows, cols), invocation);
        };
        auto append_mixed_rmsnorm = [&](int64_t site_id, int64_t rows,
                                        int64_t invocation = 0) {
            append(
                FmbRouteFamily::NORMALIZATION, site_id,
                static_cast<int64_t>(
                    rpu_resolve_mixed_rmsnorm_spm_route(
                        rows, hidden_size())),
                invocation, {rows, hidden_size()});
        };
        auto append_kv = [&](const ChunkInfo& chunk) {
            const KvInsertSegmentPlan kv_plan =
                resolve_kvinsert_plan_auto(
                    QWEN25_KV_INSERT_SITE, manifest.graph_lifecycle,
                    /*position=*/0, chunk.len, chunk.len, NUM_CORES,
                    num_q_heads(), head_dim(), QWEN25_KV_CAPABILITIES);
            const KvInsertRouteArguments arguments =
                rpu_kvinsert_route_arguments(
                    kv_plan, NUM_CORES, num_q_heads(), head_dim());
            manifest.routes.push_back({
                QWEN25_KV_INSERT_SITE, FmbRouteFamily::KV_INSERT,
                static_cast<int64_t>(kv_plan.route()),
                QWEN25_KV_FLAG_DDR_MIRROR,
                {arguments.begin(), arguments.end()}, chunk.idx});
            manifest.kv_insert_physical_rows = std::max(
                manifest.kv_insert_physical_rows,
                kv_plan.physical_rows());
        };
        if (rope_spm_enabled_ && max_hw_ > 0) {
            const int64_t selector = static_cast<int64_t>(
                Qwen25VisionMutableDmaRoute::ROPE_TABLE_DDR_TO_SPM);
            manifest.routes.push_back({
                QWEN25_ROPE_COS_PRELOAD_SITE,
                FmbRouteFamily::MUTABLE_DMA, selector, /*flags=*/0,
                {rope_spm_enabled_ ? 1 : 0, max_hw_, head_dim()},
                /*invocation=*/0});
            manifest.routes.push_back({
                QWEN25_ROPE_SIN_PRELOAD_SITE,
                FmbRouteFamily::MUTABLE_DMA, selector, /*flags=*/0,
                {rope_spm_enabled_ ? 1 : 0, max_hw_, head_dim()},
                /*invocation=*/0});
        }
        const auto layer_is_full = [&](int64_t layer) {
            return !(window_mask_present_ || per_window_sdpa_enabled_) ||
                std::find(fullatt_block_indexes_.begin(),
                          fullatt_block_indexes_.end(), layer) !=
                    fullatt_block_indexes_.end();
        };
        const auto any_layer = [&](const auto& predicate) {
            for (int64_t layer = 0; layer < num_layers(); ++layer) {
                if (predicate(layer)) return true;
            }
            return false;
        };

        if (is_exact_mimc()) {
            const bool use_w8_qkv = exact_mimc_w8_qkv_enabled();
            if (!use_w8_qkv) {
                // The one source call in packed_linear emits V/K/Q in order.
                for (int64_t invocation = 0; invocation < 3; ++invocation) {
                    append_linear(QWEN25_EXACT_QKV_LINEAR_SITE, invocation);
                }
            }
            for (int64_t image = 0; image < image_batch_count_; ++image) {
                if (rope_spm_enabled_) {
                    append(FmbRouteFamily::ROPE,
                           QWEN25_EXACT_Q_ROPE_SPM_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionRopeRoute::ROPE_2D_SPM), image,
                           {rope_spm_enabled_ ? 1 : 0},
                           FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                    append(FmbRouteFamily::ROPE,
                           QWEN25_EXACT_K_ROPE_SPM_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionRopeRoute::ROPE_2D_SPM), image,
                           {rope_spm_enabled_ ? 1 : 0},
                           FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                } else {
                    append(FmbRouteFamily::ROPE,
                           QWEN25_EXACT_Q_ROPE_DDR_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionRopeRoute::ROPE_2D_DDR), image,
                           {rope_spm_enabled_ ? 1 : 0},
                           FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
                    append(FmbRouteFamily::ROPE,
                           QWEN25_EXACT_K_ROPE_DDR_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionRopeRoute::ROPE_2D_DDR), image,
                           {rope_spm_enabled_ ? 1 : 0},
                           FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
                }
            }

            const bool native_window =
                exact_mimc_native_window_sdpa_enabled();
            const bool any_full = any_layer(layer_is_full);
            const bool any_window = any_layer(
                [&](int64_t layer) { return !layer_is_full(layer); });
            if (use_w8_qkv) {
                if (native_window && any_window) {
                    append(FmbRouteFamily::ATTENTION,
                           QWEN25_EXACT_WINDOW_ATTN_SITE,
                           static_cast<int64_t>(
                               AttentionExecutionPolicy::SPM_KV_BY_MHA),
                           /*invocation=*/0,
                           {static_cast<int64_t>(schedule_),
                            per_window_sdpa_enabled_ ? 1 : 0});
                }
                if (!native_window || any_full) {
                    append(FmbRouteFamily::ATTENTION,
                           QWEN25_EXACT_W8_ATTN_SITE,
                           static_cast<int64_t>(
                               AttentionExecutionPolicy::SPM_KV_BY_MHA),
                           /*invocation=*/0,
                           {static_cast<int64_t>(schedule_),
                            per_window_sdpa_enabled_ ? 1 : 0});
                }
            } else {
                append(FmbRouteFamily::ATTENTION,
                       QWEN25_EXACT_FP16_ATTN_SITE,
                       static_cast<int64_t>(
                           AttentionExecutionPolicy::SPM_KV_BY_MHA),
                       /*invocation=*/0,
                       {static_cast<int64_t>(schedule_),
                        per_window_sdpa_enabled_ ? 1 : 0});
            }

            const bool compact = exact_mimc_compact_local_inout_enabled();
            const bool fused_gate_up =
                exact_mimc_fused_w8_gate_up_enabled();
            if (rmsnorm_capability_ != RPU_RMSNORM_CAP_BASE) {
                for (int64_t image = 0;
                     image < image_batch_count_; ++image) {
                    append(
                        FmbRouteFamily::NORMALIZATION,
                        kQwen25ExactPreRmsNormRowsSite,
                        static_cast<int64_t>(rpu_resolve_rmsnorm_spm_route(
                            rmsnorm_capability_,
                            ExactMimcPlan::kAttentionRows, hidden_size())),
                        image,
                        {ExactMimcPlan::kAttentionRows, hidden_size(),
                         static_cast<int64_t>(rmsnorm_capability_)});
                }
                if (!compact) {
                    for (const ChunkInfo& chunk : plan.compute.chunks) {
                        append(
                            FmbRouteFamily::NORMALIZATION,
                            kQwen25ExactComputeRmsNormRowsSite,
                            static_cast<int64_t>(
                                rpu_resolve_rmsnorm_spm_route(
                                    rmsnorm_capability_, chunk.len,
                                    hidden_size())),
                            chunk.idx,
                            {chunk.len, hidden_size(),
                             static_cast<int64_t>(
                                 rmsnorm_capability_)});
                    }
                }
            }
            if (mixed_precision_ && !compact &&
                num_layers() - 1 > QWEN25VL_VISION_MIXED_BOUNDARY_LAYER) {
                for (const ChunkInfo& chunk : plan.compute.chunks) {
                    append_mixed_rmsnorm(
                        QWEN25_EXACT_COMPUTE_MIXED_RMSNORM_SITE,
                        chunk.len, chunk.idx);
                }
            }
            const size_t active_chunks = compact ? 1 : plan.compute.chunks.size();
            const bool row_weight_outer = wall_sorting_.vision_row_reuse && compact;
            append(FmbRouteFamily::LINEAR, QWEN25_EXACT_O_LINEAR_SITE,
                   row_weight_outer
                       ? static_cast<int64_t>(Qwen25VisionLinearRoute::WALL_ROW_FULL_LOCAL_K)
                       : qwen25_vision_linear_route(),
                   plan.compute.chunks.front().idx);
            for (size_t index = 0; index < active_chunks; ++index) {
                const ChunkInfo& chunk = plan.compute.chunks[index];
                if (!compact) {
                    // Every non-empty model has at least one pre-boundary
                    // layer. The same semantic site is consumed repeatedly by
                    // those layers for this image/chunk invocation.
                    append(FmbRouteFamily::ALL_REDUCE,
                           QWEN25_EXACT_ATTN_LOCAL_REDUCE_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionAllReduceRoute::
                                   RING_LOCAL_RESIDUAL),
                           chunk.idx);
                    if (mixed_precision_ &&
                        num_layers() - 1 >
                            QWEN25VL_VISION_MIXED_BOUNDARY_LAYER) {
                        append(FmbRouteFamily::ALL_REDUCE,
                               QWEN25_EXACT_ATTN_MIXED_LOCAL_REDUCE_SITE,
                               static_cast<int64_t>(
                                   Qwen25VisionAllReduceRoute::
                                       MIXED_LOCAL_RESIDUAL),
                               chunk.idx);
                    }
                }
                if (!fused_gate_up) {
                    append_linear(QWEN25_EXACT_GATE_LINEAR_SITE, chunk.idx);
                    append_linear(QWEN25_EXACT_UP_LINEAR_SITE, chunk.idx);
                }
                append(FmbRouteFamily::LINEAR, QWEN25_EXACT_DOWN_LINEAR_SITE,
                       row_weight_outer
                           ? static_cast<int64_t>(Qwen25VisionLinearRoute::WALL_ROW_FULL_LOCAL_K)
                           : qwen25_vision_linear_route(),
                       chunk.idx, {mixed_precision_ ? 1 : 0});
            }
            if (mixed_precision_ &&
                QWEN25VL_VISION_MIXED_BOUNDARY_LAYER < num_layers()) {
                for (int64_t image = 0;
                     image < image_batch_count_; ++image) {
                    append(
                        FmbRouteFamily::LINEAR,
                        QWEN25_EXACT_DOWN_BF16_LINEAR_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionLinearRoute::ACC32_OUT_BF16), image);
                }
            }
            if (!compact) {
                const int64_t last = num_layers() - 1;
                for (const ChunkInfo& chunk : plan.compute.chunks) {
                    if (mixed_precision_ &&
                        last == QWEN25VL_VISION_MIXED_BOUNDARY_LAYER) {
                        append(FmbRouteFamily::ALL_REDUCE,
                               QWEN25_EXACT_FINAL_BF16_REDUCE_SITE,
                               static_cast<int64_t>(
                                   Qwen25VisionAllReduceRoute::
                                       EXACT_M576_BF16_PARTIAL_FP16_RESIDUAL),
                               chunk.idx);
                    } else if (mixed_precision_ &&
                               last >
                                   QWEN25VL_VISION_MIXED_BOUNDARY_LAYER) {
                        append(FmbRouteFamily::ALL_REDUCE,
                               QWEN25_EXACT_FINAL_MIXED_REDUCE_SITE,
                               static_cast<int64_t>(
                                   Qwen25VisionAllReduceRoute::
                                       EXACT_M576_FP16_PARTIAL_BF16_RESIDUAL),
                               chunk.idx);
                    } else {
                        append(FmbRouteFamily::ALL_REDUCE,
                               QWEN25_EXACT_FINAL_REDUCE_SITE,
                               static_cast<int64_t>(
                                   Qwen25VisionAllReduceRoute::
                                       EXACT_M576_SUM_RESIDUAL),
                               chunk.idx);
                    }
                }
            }
        } else {
            const bool has_pre_boundary_layer =
                !mixed_precision_ || num_layers() > 0;
            const bool has_post_boundary_layer = mixed_precision_ &&
                num_layers() - 1 > QWEN25VL_VISION_MIXED_BOUNDARY_LAYER;
            const bool has_boundary_layer = mixed_precision_ &&
                QWEN25VL_VISION_MIXED_BOUNDARY_LAYER < num_layers();
            for (const ChunkInfo& chunk : plan.compute.chunks) {
                const auto small_norm = small_image_rmsnorm_route(chunk.len);
                const bool owner_ring_norm = small_image_owner_ring_norm_enabled(chunk.len);
                if (small_norm != RpuRmsNormSpmRoute::BASE) {
                    for (const int64_t site : {QWEN25_SMALL_INPUT_RMSNORM_SITE,
                                               QWEN25_SMALL_POST_RMSNORM_SITE}) {
                        if (owner_ring_norm && site == QWEN25_SMALL_POST_RMSNORM_SITE)
                            continue;
                        append(FmbRouteFamily::NORMALIZATION, site,
                               static_cast<int64_t>(small_norm), chunk.idx,
                               {chunk.len, hidden_size(),
                                static_cast<int64_t>(small_rmsnorm_capability_)});
                    }
                }
                const int64_t row_route = small_image_acc16_row_projection_enabled(chunk.len)
                    ? static_cast<int64_t>(Qwen25VisionLinearRoute::WALL_ROW_FULL_LOCAL_K)
                    : qwen25_vision_linear_route();
                for (const int64_t site_id : {
                         QWEN25_Q_LINEAR_SITE, QWEN25_K_LINEAR_SITE,
                         QWEN25_V_LINEAR_SITE}) {
                    append_linear(site_id, chunk.idx);
                }
                if (small_image_acc16_gate_up_projection_enabled(chunk.len)) {
                    append(FmbRouteFamily::LINEAR, QWEN25_GATE_LINEAR_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionLinearRoute::WALL_SINGLE_BASE_GATE_UP_SWIGLU),
                           chunk.idx, small_image_gate_up_arguments(chunk.len));
                } else {
                    append_linear(QWEN25_GATE_LINEAR_SITE, chunk.idx);
                    append_linear(QWEN25_UP_LINEAR_SITE, chunk.idx);
                }
                append(FmbRouteFamily::LINEAR, QWEN25_O_LINEAR_SITE,
                       row_route, chunk.idx);
                append(FmbRouteFamily::LINEAR,
                    QWEN25_DOWN_LINEAR_SITE, row_route, chunk.idx,
                    {mixed_precision_ ? 1 : 0});
                if (has_boundary_layer) {
                    append(
                        FmbRouteFamily::LINEAR,
                        QWEN25_DOWN_BF16_LINEAR_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionLinearRoute::ACC32_OUT_BF16),
                        chunk.idx);
                }
                if (small_image_qk_rope_enabled(chunk.len)) {
                    append(FmbRouteFamily::ROPE, QWEN25_SMALL_QK_ROPE_SPM_SITE,
                           static_cast<int64_t>(Qwen25VisionRopeRoute::PAIRED_QK_2D_SPM),
                           chunk.idx, small_image_qk_rope_arguments(chunk.len, chunk.offset),
                           FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                } else if (rope_spm_enabled_) {
                    append(FmbRouteFamily::ROPE, QWEN25_Q_ROPE_SPM_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionRopeRoute::ROPE_2D_SPM),
                           chunk.idx, {rope_spm_enabled_ ? 1 : 0},
                           FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                    append(FmbRouteFamily::ROPE, QWEN25_K_ROPE_SPM_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionRopeRoute::ROPE_2D_SPM),
                           chunk.idx, {rope_spm_enabled_ ? 1 : 0},
                           FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                } else {
                    append(FmbRouteFamily::ROPE, QWEN25_Q_ROPE_DDR_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionRopeRoute::ROPE_2D_DDR),
                           chunk.idx, {rope_spm_enabled_ ? 1 : 0},
                           FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
                    append(FmbRouteFamily::ROPE, QWEN25_K_ROPE_DDR_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionRopeRoute::ROPE_2D_DDR),
                           chunk.idx, {rope_spm_enabled_ ? 1 : 0},
                           FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
                }
                if (!small_image_raw_kv_only_enabled()) {
                    append_kv(chunk);
                }

                if (has_pre_boundary_layer) {
                    if (owner_ring_norm) {
                        append(FmbRouteFamily::ALL_REDUCE,
                               QWEN25_SMALL_OWNER_RING_NORM_SITE,
                               static_cast<int64_t>(
                                   Qwen25VisionAllReduceRoute::OWNER_RING_NORM_M672),
                               chunk.idx, small_image_owner_ring_norm_arguments(chunk.len));
                    } else {
                        append_ring(QWEN25_ATTN_REDUCE_SITE, chunk.len,
                                    hidden_size(), chunk.idx);
                    }
                    // The final layer still publishes the full raw residual.
                    append_ring(QWEN25_DOWN_REDUCE_SITE, chunk.len,
                                hidden_size(), chunk.idx);
                    if (small_image_down_owner_norm_enabled(chunk.len)) {
                        for (int64_t layer = 0; layer < num_layers() - 1; ++layer) {
                            append(FmbRouteFamily::ALL_REDUCE,
                                   QWEN25_SMALL_DOWN_OWNER_NORM_SITE,
                                   static_cast<int64_t>(
                                       Qwen25VisionAllReduceRoute::OWNER_DOWN_RING_NORM_M672),
                                   layer, small_image_down_owner_norm_arguments(chunk.len, layer));
                        }
                    }
                }
                if (has_post_boundary_layer) {
                    append_mixed_rmsnorm(
                        QWEN25_INPUT_MIXED_RMSNORM_SITE,
                        chunk.len, chunk.idx);
                    append_mixed_rmsnorm(
                        QWEN25_POST_MIXED_RMSNORM_SITE,
                        chunk.len, chunk.idx);
                    append(
                        FmbRouteFamily::ALL_REDUCE,
                        QWEN25_ATTN_MIXED_REDUCE_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionAllReduceRoute::
                                FP16_PARTIAL_BF16_RESIDUAL),
                        chunk.idx);
                    append(
                        FmbRouteFamily::ALL_REDUCE,
                        QWEN25_DOWN_MIXED_REDUCE_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionAllReduceRoute::
                                FP16_PARTIAL_BF16_RESIDUAL),
                        chunk.idx);
                }
                if (has_boundary_layer) {
                    append(
                        FmbRouteFamily::ALL_REDUCE,
                        QWEN25_DOWN_BF16_REDUCE_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionAllReduceRoute::
                                BF16_PARTIAL_FP16_RESIDUAL),
                        chunk.idx);
                }
            }
            if (small_image_ragged_sdpa_enabled(physical_len)) {
                TORCH_CHECK(layout.attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA,
                            "Wall ragged SDPA requires the raw-SPM plan");
                validate_small_image_qk_rope_plan(plan, physical_len);
                for (int64_t image = 0; image < image_batch_count_; ++image)
                    append(FmbRouteFamily::ATTENTION, QWEN25_SMALL_RAGGED_SDPA_SITE,
                           static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA),
                           image, small_image_ragged_sdpa_arguments(physical_len / image_batch_count_));
            }
            const bool raw_requested =
                is_generic() && layout.attention_policy ==
                    AttentionExecutionPolicy::SPM_KV_BY_MHA;
            for (const GenericAttentionGeometry& site :
                 generic_attention_geometries(plan)) {
                const bool raw_valid = generic_attention_site_raw_valid(
                    plan, site.site_id);
                const AttentionExecutionPolicy policy =
                    raw_requested && raw_valid
                    ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                    : AttentionExecutionPolicy::DDR_KV;
                append(
                    FmbRouteFamily::ATTENTION,
                    generic_attention_physical_site_id(
                        site.site_id, policy),
                    static_cast<int64_t>(policy), site.invocation,
                    generic_attention_arguments(site),
                    policy == AttentionExecutionPolicy::SPM_KV_BY_MHA
                        ? 0
                        : (!is_generic()
                               ? QWEN25_ATTN_DDR_LAYER_GROUP_LAYOUT
                               : (raw_valid
                                      ? QWEN25_ATTN_DDR_CAPABILITY_FALLBACK
                                      : QWEN25_ATTN_DDR_RAW_KERNEL_INCOMPATIBLE)));
            }
        }

        if (merger_active()) {
            // Pair merger normalizes all 672 physical rows before Python
            // crops the two dummy merge-unit tails back to 648 logical rows.
            const auto small_norm = small_image_rmsnorm_route(physical_len);
            if (small_norm != RpuRmsNormSpmRoute::BASE) {
                append(FmbRouteFamily::NORMALIZATION,
                       QWEN25_SMALL_MERGER_RMSNORM_SITE,
                       static_cast<int64_t>(small_norm), /*invocation=*/0,
                       {physical_len, hidden_size(),
                        static_cast<int64_t>(small_rmsnorm_capability_)});
            }
            auto append_merger = [&](int64_t rows, int64_t invocation) {
                append_linear(QWEN25_MERGER_M0_LINEAR_SITE, invocation);
                append_linear(QWEN25_MERGER_M2_LINEAR_SITE, invocation);
                append_ring(QWEN25_MERGER_REDUCE_SITE, rows / 4,
                            merger_out_hidden_, invocation);
                if (!z1_bound_) {
                    append(FmbRouteFamily::MUTABLE_DMA,
                           QWEN25_MERGER_OUTPUT_DMA_SITE,
                           static_cast<int64_t>(
                               Qwen25VisionMutableDmaRoute::SPM_COPY_TO_DDR),
                           invocation);
                }
            };
            if (is_generic() || exact_mimc_w8_qkv_enabled()) {
                append_merger(logical_len, /*invocation=*/0);
            } else {
                const int64_t rows = logical_len / image_batch_count_;
                for (int64_t image = 0;
                     image < image_batch_count_; ++image) {
                    append_merger(rows, image);
                }
            }
            if (mixed_precision_) {
                if (is_generic()) {
                    append_mixed_rmsnorm(
                        QWEN25_MERGER_MIXED_RMSNORM_SITE,
                        logical_len, /*invocation=*/0);
                } else if (exact_mimc_w8_qkv_enabled() &&
                           !exact_mimc_compact_local_inout_enabled()) {
                    const int64_t rows = logical_len / image_batch_count_;
                    for (int64_t image = 0;
                         image < image_batch_count_; ++image) {
                        append_mixed_rmsnorm(
                            QWEN25_EXACT_MERGER_MIXED_RMSNORM_SITE,
                            rows, image);
                    }
                }
            }
        }

        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA);
        std::sort(
            manifest.routes.begin(), manifest.routes.end(),
            [](const FmbRouteManifestEntry& lhs,
               const FmbRouteManifestEntry& rhs) {
                return std::make_tuple(
                           static_cast<int64_t>(lhs.family), lhs.site_id,
                           lhs.invocation) <
                    std::make_tuple(
                           static_cast<int64_t>(rhs.family), rhs.site_id,
                           rhs.invocation);
            });
        return manifest;
    }

    std::vector<FmbPhysicalExecutionManifest>
    physical_manifest_domain_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout, int64_t physical_len,
        int64_t logical_len, int64_t position) const override {
        if (!is_generic()) {
            return {physical_manifest_for_candidate(
                plan, layout, physical_len, logical_len, position)};
        }

        if (small_image_native_window_enabled() || small_image_raw_kv_only_enabled()) {
            // These controlled small-image schedules consume raw SPM K/V
            // directly. Omitting the mirror requires every consumer to be raw.
            LayoutContext raw_layout = layout;
            raw_layout.attention_policy = AttentionExecutionPolicy::SPM_KV_BY_MHA;
            TORCH_CHECK(subclass_spm_kv_by_mha_eligible(plan, raw_layout, position),
                        "RPU_PLANNER_REJECT:CAPABILITY: small-image "
                        "Vision requires a feasible raw-SPM attention plan");
            const auto sites = generic_attention_geometries(plan);
            TORCH_CHECK(!sites.empty() && std::all_of(
                            sites.begin(), sites.end(),
                            [this, &plan](const GenericAttentionGeometry& site) {
                                return generic_attention_site_raw_valid(plan, site.site_id);
                            }),
                        "RPU_PLANNER_REJECT:CAPABILITY: small raw-only Vision "
                        "requires every attention site to support raw SPM");
            return {physical_manifest_for_candidate(
                plan, raw_layout, physical_len, logical_len, position)};
        }

        LayoutContext ddr_layout = layout;
        ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;
        std::vector<FmbPhysicalExecutionManifest> domain{
            physical_manifest_for_candidate(
                plan, ddr_layout, physical_len, logical_len, position)};

        LayoutContext raw_layout = layout;
        raw_layout.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        if (subclass_spm_kv_by_mha_eligible(
                plan, raw_layout, position)) {
            domain.push_back(physical_manifest_for_candidate(
                plan, raw_layout, physical_len, logical_len, position));
        }
        return domain;
    }

    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& manifest) const override {
        const auto accumulation = mixed_precision_
            ? FmbLinearAccumulationPolicy::MIXED_BY_SITE
            : (linear_acc32_ ? FmbLinearAccumulationPolicy::ACC32
                             : FmbLinearAccumulationPolicy::ACC16);
        if (manifest.linear_accumulation != accumulation)
            return {false, FmbGraphLifecycle::COMPOSITE_CHILD};
        const bool ragged = small_image_ragged_sdpa_enabled(current_num_patches_);
        bool schedule_ok = !ragged;
        std::array<bool, 2> seen{false, false};
        int64_t count = 0;
        for (const auto& route : manifest.routes) {
            if (route.site_id == QWEN25_WALL_SORTING_SITE &&
                route.family == FmbRouteFamily::GRAPH_SCHEDULE)
                schedule_ok = ((route.flags & 2048) != 0) == ragged;
            if (route.site_id != QWEN25_SMALL_RAGGED_SDPA_SITE) continue;
            if (!ragged || route.family != FmbRouteFamily::ATTENTION ||
                route.selector != static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA) ||
                route.flags != 0 || route.invocation < 0 || route.invocation >= image_batch_count_ ||
                route.invocation >= 2 || seen[route.invocation] ||
                route.arguments != small_image_ragged_sdpa_arguments(small_ragged_mask_rows_))
                return {false, FmbGraphLifecycle::COMPOSITE_CHILD};
            seen[route.invocation] = true;
            ++count;
        }
        return {schedule_ok && count == (ragged ? image_batch_count_ : 0),
                FmbGraphLifecycle::COMPOSITE_CHILD};
    }

    ChunkPlan plan_exact_mimc_qkv_chunks(const ChunkPlan& compute_plan) {
        TORCH_CHECK(
            is_exact_mimc() &&
                compute_plan.chunk_size == ExactMimcPlan::kAttentionRows &&
                compute_plan.num_chunks == ExactMimcPlan::kImages,
            "qwen25vl exact MIMC QKV plan requires 3x576 compute chunks");
        return {ExactMimcPlan::kQkvRows, 1};
    }

    bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const override {
        if (is_generic()) {
            if (z1_bound_ || position != 0 || layout.is_causal ||
                layout.use_attn_mask || layout.batch_size != 1 ||
                layout.attention_policy !=
                    AttentionExecutionPolicy::SPM_KV_BY_MHA ||
                plan.chunk_mode != ChunkMode::SEQUENTIAL ||
                plan.input.chunks.size() != 1 ||
                plan.qkv.chunks.size() != 1 ||
                plan.compute.chunks.size() != 1 ||
                image_batch_count_ <= 0 ||
                plan.spans.size() !=
                    static_cast<size_t>(image_batch_count_)) {
                return false;
            }
            const int64_t rows = current_num_patches_;
            const ChunkInfo& input = plan.input.chunks.front();
            const ChunkInfo& qkv = plan.qkv.chunks.front();
            const ChunkInfo& compute = plan.compute.chunks.front();
            if (rows <= 0 || rows % image_batch_count_ != 0 ||
                input.offset != 0 || input.len != rows ||
                qkv.offset != 0 || qkv.len != rows ||
                compute.offset != 0 || compute.len != rows ||
                compute.kv_seq_len != rows || layout.chunk_size != rows ||
                layout.effective_kv_cs() != rows ||
                layout.max_kv_seq_len != rows) {
                return false;
            }
            const int64_t per_image = rows / image_batch_count_;
            for (int64_t image = 0; image < image_batch_count_; ++image) {
                if (plan.spans[image].offset != image * per_image ||
                    plan.spans[image].len != per_image) {
                    return false;
                }
            }
            const std::vector<GenericAttentionGeometry> sites =
                generic_attention_geometries(plan);
            return std::any_of(
                sites.begin(), sites.end(),
                [this, &plan](const GenericAttentionGeometry& site) {
                    return generic_attention_site_raw_valid(
                        plan, site.site_id);
                });
        }

        if (!is_exact_mimc() || position != 0 || layout.is_causal ||
            layout.use_attn_mask || layout.batch_size != 1 ||
            layout.attention_policy !=
                AttentionExecutionPolicy::SPM_KV_BY_MHA ||
            plan.chunk_mode != ChunkMode::KV_FIRST ||
            plan.input.chunks.size() != 1 ||
            plan.qkv.chunks.size() != 1 ||
            plan.compute.chunks.size() != ExactMimcPlan::kImages ||
            plan.spans.size() != ExactMimcPlan::kImages ||
            layout.chunk_size != ExactMimcPlan::kAttentionRows ||
            layout.effective_kv_cs() != ExactMimcPlan::kQkvRows ||
            layout.max_kv_seq_len != ExactMimcPlan::kQkvRows) {
            return false;
        }
        if (plan.input.chunks.front().len != ExactMimcPlan::kQkvRows ||
            plan.qkv.chunks.front().len != ExactMimcPlan::kQkvRows) {
            return false;
        }
        for (int64_t image = 0; image < ExactMimcPlan::kImages; ++image) {
            const int64_t offset = image * ExactMimcPlan::kAttentionRows;
            if (plan.compute.chunks[image].offset != offset ||
                plan.compute.chunks[image].len !=
                    ExactMimcPlan::kAttentionRows ||
                plan.spans[image].offset != offset ||
                plan.spans[image].len != ExactMimcPlan::kAttentionRows) {
                return false;
            }
        }
        return num_layers() == 32 && hidden_size() == 1280 &&
            num_q_heads() == 16 && num_kv_heads() == 16 &&
            head_dim() == 80;
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        int64_t cs = ctx.chunk_size;
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();   // padded (3456)
        const bool rope_spm_layout =
            ctx.rope_table_residency == FmbRopeTableResidency::SPM ||
            (ctx.rope_table_residency ==
                 FmbRopeTableResidency::UNSPECIFIED &&
             rope_spm_enabled_);
        TORCH_CHECK(
            ctx.rope_table_residency ==
                    FmbRopeTableResidency::UNSPECIFIED ||
                rope_spm_layout == rope_spm_enabled_,
            "RPU_PLANNER_REJECT:EXACT_MISMATCH: Qwen2.5-VL Vision "
            "descriptor conflicts with the cold exact 2D RoPE route");

        int64_t local_q_dim = (nq / NUM_CORES) * hd;
        int64_t local_inter = is_ / NUM_CORES;   // 3456/8 = 432
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        int64_t res  = A(cs * h * DWIDTH);
        const bool raw_generic_layout = is_generic() &&
            ctx.attention_policy ==
                AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const int64_t qkv_rows = raw_generic_layout
            ? Align(cs, int64_t{16}) : cs;
        int64_t qkv  = A(qkv_rows * local_q_dim * DWIDTH);
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
        if (raw_generic_layout) {
            // The raw pair uses this slot for V^T
            // [nkv/core, head_dim, align16(seq_k)]. Exact joint feasibility is
            // checked by FMB against this descriptor-specific declaration.
            sdpa_tmp = std::max(
                sdpa_tmp,
                A(Align(cs, int64_t{16}) * nkv_per_core * hd * DWIDTH));
        }

        auto dma_safe = [&](int64_t elems) -> int64_t {
            int64_t dma_elems = ((elems + 255) / 256) * 256;
            return A(dma_elems * DWIDTH);
        };
        int64_t norm_w_sz     = dma_safe(h);
        int64_t q_bias_sz     = dma_safe(local_q_dim);
        int64_t inter_bias_sz = dma_safe(local_inter);   // gate/up per-core 432
        int64_t full_bias_sz  = dma_safe(h);             // o/down full width 1280

        // P3b: dense block-diagonal mask [seq_q, seq_k_v16*16] fp16, broadcast to
        // all cores. seq_k = seq_q = cs (single-chunk vision). Same formula as
        // Gemma (rpu_gemma_model.cpp). Only declared when windowing.
        // Batched: window SDPA is per-image (per_image_ctx queries), so the mask is
        // [per_image_ctx, per_image_ctx], not [cs, cs] — much smaller in SPM.
        int64_t mask_cs = packed_batch ? (cs / image_batch_count_) : cs;
        int64_t mask_sz = A(mask_cs * CeilDiv(mask_cs, (int64_t)16) * 32);

        int nl = static_cast<int>(num_layers());

        if (is_exact_mimc()) {
            const bool w8_direct_sdpa_output =
                exact_mimc_w8_qkv_enabled();
            const bool fused_w8_gate_up =
                exact_mimc_fused_w8_gate_up_enabled();
            const bool native_window_sdpa =
                exact_mimc_native_window_sdpa_enabled();
            const bool compact_local_inout =
                exact_mimc_compact_local_inout_enabled();
            const int64_t packed_seq = ExactMimcPlan::kQkvRows;
            const int64_t packed_res = A(packed_seq * h * DWIDTH);
            const int64_t packed_qkv =
                A(packed_seq * local_q_dim * DWIDTH);
            const int64_t mlp_rows = ExactMimcPlan::kMlpRows;
            const int64_t mlp_res = A(mlp_rows * h * DWIDTH);
            const int64_t mlp_inter =
                A(mlp_rows * local_inter * DWIDTH);
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

            // O and Down reuse the current image's input_norm slice as their
            // partial/postnorm carrier. The exact native-window merger path
            // carries raw residuals only in compact per-core shards.
            std::vector<BufferDecl> decls = {
                {"input_norm",   packed_res,   1,
                 merger_active() ? 15 : 9,
                 StorageClass::Temp, 0, nullptr},
                {"compact_residual",
                                  packed_qkv,   1, 9,
                                  StorageClass::Temp, 0, nullptr},
                {"oproj",        packed_res,   4, 7, StorageClass::Temp, 0, "input_norm"},
            };
            if (!compact_local_inout) {
                // Fallbacks retain the full image-local post-attention raw.
                decls.push_back(
                    {"o_result", mlp_res, 5, 9,
                     StorageClass::Temp, 0, nullptr});
            }
            decls.insert(decls.end(), {
                {"q",            packed_qkv,   2,
                 w8_direct_sdpa_output ? 3 : 4,
                 StorageClass::Temp, 0, nullptr},
                {"k",            packed_qkv,   2, 3, StorageClass::Temp, 0, nullptr},
                {"v",            packed_qkv,   2,
                 w8_direct_sdpa_output ? 4 : 3,
                 StorageClass::Temp, 0, nullptr},

                // The compact layer-major path completes all three O/Norm2
                // stages before one multibase GateUp, so keep three fused
                // outputs live for the following image-local Down stages.
                {"gate",         compact_local_inout
                                     ? ExactMimcPlan::kImages * mlp_inter
                                     : mlp_inter,
                                  7, 8, StorageClass::Temp, 0, nullptr},
            });
            // The exact W8 GateUp kernel writes the fused SwiGLU result directly
            // to gate, so its fallback-only Up output must not consume SPM.
            if (!fused_w8_gate_up) {
                decls.push_back(
                    {"up", mlp_inter, 7, 7,
                     StorageClass::Temp, 0, nullptr});
            }
            // One current-layer parameter set. The layer-major graph stages
            // these addresses at the start of each layer, avoiding the 32-copy
            // persistent footprint of the normal schedule.
            decls.insert(decls.end(), {
                {"norm1_w",   norm_w_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"norm2_w",   norm_w_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"q_bias",    q_bias_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"k_bias",    q_bias_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"v_bias",    q_bias_sz,     0, 0, StorageClass::Persistent, 0, nullptr},
                {"gate_bias", inter_bias_sz, 0, 0, StorageClass::Persistent, 0, nullptr},
                {"up_bias",   inter_bias_sz, 0, 0, StorageClass::Persistent, 0, nullptr},
                {"o_bias",    full_bias_sz,  0, 0, StorageClass::Persistent, 0, nullptr},
                {"down_bias", full_bias_sz,  0, 0, StorageClass::Persistent, 0, nullptr},
            });
            if (window_mask_present_ && !native_window_sdpa) {
                // Exact fused W8 keeps the one shared [576,576] mask alive for
                // every encoder layer and uploads it once before the loop. It
                // replaces the unused Up footprint plus 147,456 bytes. FP16
                // fallback retains the phase-local slot and per-layer upload.
                decls.push_back(
                    {"sdpa_mask", mask_sz,
                     fused_w8_gate_up ? 1 : 3,
                     fused_w8_gate_up ? 9 : 3,
                     StorageClass::Temp, 0, nullptr});
            }
            if (rope_spm_layout && max_hw_ > 0) {
                int64_t rope_tbl_sz = A(max_hw_ * (hd / 4) * DWIDTH);
                decls.push_back(
                    {"rope_cos", rope_tbl_sz, 0, 0,
                     StorageClass::Persistent, 0, nullptr});
                decls.push_back(
                    {"rope_sin", rope_tbl_sz, 0, 0,
                     StorageClass::Persistent, 0, nullptr});
            }
            if (merger_active()) {
                const int64_t local_mh = merge_hidden_ / NUM_CORES;
                const int64_t oh = merger_out_hidden_;
                if (exact_mimc_w8_qkv_enabled()) {
                    const int64_t mrows = packed_seq / 4;
                    const int64_t mid_sz = A(mrows * local_mh * DWIDTH);
                    const int64_t out_sz = A(mrows * oh * DWIDTH);
                    TORCH_CHECK(2 * out_sz <= packed_res,
                                "exact W8 merger outputs exceed input_norm");
                    decls.push_back(
                        {"merger_mid", mid_sz, 11, 13,
                         StorageClass::Temp, 0, nullptr});
                    decls.push_back(
                        {"merger_zero", out_sz, 14, 14,
                         StorageClass::Temp, 0, nullptr});
                } else {
                    const int64_t mrows = cs / 4;
                    const int64_t normed_sz = A(cs * h * DWIDTH);
                    const int64_t mid_sz = A(mrows * local_mh * DWIDTH);
                    const int64_t out_sz = A(mrows * oh * DWIDTH);
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
                }
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
            TORCH_CHECK(
                compact_local_inout
                    ? temp_peak <= ExactMimcPlan::kNativeWindowLocalInoutTempPeakBytes
                    : !native_window_sdpa ||
                        temp_peak <=
                            ExactMimcPlan::kNativeWindowTempPeakBytes,
                "exact MIMC native-window layout exceeds its validated "
                "temporary SPM envelope: temp_peak=", temp_peak,
                " native_window_cap=",
                compact_local_inout
                    ? ExactMimcPlan::kNativeWindowLocalInoutTempPeakBytes
                    : ExactMimcPlan::kNativeWindowTempPeakBytes);
            return decls;
        }

        std::vector<BufferDecl> decls = {
            {"residual1",  res,    1, 6, StorageClass::Temp, 0, nullptr},
            {"input_norm", res,    1, 6, StorageClass::Temp, 0, nullptr},
            {"oproj",      res,    4, 6, StorageClass::Temp, 0, nullptr},

            // Place the SDPA output before Q/K/V in the allocator's phase-2
            // group so its phase-4 lifetime leaves a contiguous O-proj slot.
            {"sdpa_out",   qkv,    2, 4, StorageClass::Temp, 0, nullptr},
            {"q",          qkv,    2, 3, StorageClass::Temp, 0, nullptr},
            {"k",          qkv,    2, 3, StorageClass::Temp, 0, nullptr},
            {"v",          qkv,    2, 3, StorageClass::Temp, 0, nullptr},
            {"sdpa_tmp",   sdpa_tmp, 3, 3, StorageClass::Temp, 0, nullptr},

            // SwiGLU: gate + up live across phases 5..6 (gate also holds the
            // silu*up product fed to down).
            {"gate",       inter,  5, 6, StorageClass::Temp, 0, nullptr},
            // `up` lives ONLY in phase 5 (written by up_proj, consumed by the SwiGLU
            // mul, dead before phase 6). `residual1`'s CONTENT is dead in phase 5 (read
            // by the phase-4 residual-add, then overwritten by the phase-6 block-output
            // write — never read in phase 5/6). up (inter) ⊆ residual1 (res), so up
            // safely reuses residual1's slot → drops the phase-5 peak by one `inter`
            // (3·res+2·inter → 3·res+1·inter), which lets 3×256=768 fit a single chunk
            // in the full VLA.
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
        if (window_mask_present_ && !small_image_native_window_enabled()) {
            const bool resident_mask = small_image_resident_mask_enabled();
            decls.push_back({"sdpa_mask", mask_sz,
                             resident_mask ? 1 : 3, resident_mask ? 9 : 4,
                             StorageClass::Temp, 0, nullptr});
        }
        if (padded_pair_enabled()) {
            decls.push_back({"sdpa_full_mask", mask_sz, 1, 9,
                             StorageClass::Temp, 0, nullptr});
        }
        // SPM-resident 2D-RoPE cos/sin tables [max_hw, head_dim/4] fp16 (shared
        // across all layers; broadcast once in emit_preload_weights). Tiny
        // (max_hw=128, hd/4=20 → ~5 KB each). See rope_spm_enabled_.
        if (rope_spm_layout && max_hw_ > 0) {
            int64_t rope_tbl_sz = A(max_hw_ * (hd / 4) * DWIDTH);
            decls.push_back({"rope_cos", rope_tbl_sz, 0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"rope_sin", rope_tbl_sz, 0, 0, StorageClass::Persistent, 0, nullptr});
        }
        // Post-encoder merger (single- AND batched-image, UNIFIED — no per-g
        // branch). The merger is row-independent (RMSNorm per-row, GEMMs per-row,
        // GELU per-elem, all-reduce per-row), so one full-batch pass over the
        // packed residual1[seq,h] (seq = current_num_patches_ = g*n_i) is
        // numerically identical to the old per-image loop — and crucially has NO
        // cross-iteration reuse, so it cannot have the per-image-loop WAR race on
        // merger_out (image i's async SPM->DDR vs image i+1's all-reduce overwrite)
        // that made the batched fused merger replay non-deterministically.
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
        if (linear_acc32_) h = detail::layout_mix(h, 32);
        h = detail::layout_mix(h, per_window_sdpa_enabled_ ? 1 : 0);
        h = detail::layout_mix(h, wall_sorting_.fp16_transfer_mask & 1u);
        h = detail::layout_mix(h, wall_sorting_.vision_row_reuse ? 1 : 0);
        h = detail::layout_mix(h, wall_sorting_.silu_mul ? 1 : 0);
        h = detail::layout_mix(h, small_image_schedule_flags());
        if (valid_patches_per_image_ != 0) {
            h = detail::layout_mix(h, valid_patches_per_image_);
        }
        h = detail::layout_mix(
            h, exact_mimc_native_window_sdpa_enabled() ? 1 : 0);
        return h;
    }

    void emit_preload_weights() {
        TORCH_CHECK(ctx().has_complete_physical_manifest() ||
                        (!(wall_sorting_.fp16_transfer_mask & 1u) &&
                         !wall_sorting_.vision_row_reuse && !wall_sorting_.silu_mul),
                    "Wall Vision sorting requires a COMPLETE physical descriptor");
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, QWEN25_WALL_SORTING_SITE,
                1, small_image_schedule_flags(),
                sorting_schedule_arguments(), 0);
            if (padded_pair_enabled()) {
                const auto& manifest = ctx().physical_manifest();
                TORCH_CHECK(manifest.logical_length == 648 &&
                                manifest.physical_length == 672 &&
                                manifest.execution_padding_rows == 24,
                            "padded Vision forward descriptor lost its logical/physical rows");
            }
        }
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

        // The exact W8 path owns a full-layer-lifetime mask slot. The prepared
        // mask is shared by all 28 window layers, so one graph-front upload is
        // sufficient. The sorting FP16 opt-in declares the same full lifetime.
        if (exact_mimc_fused_w8_gate_up_enabled() &&
            window_mask_present_ &&
            !exact_mimc_native_window_sdpa_enabled()) {
            TORCH_CHECK(prepared_window_mask_.mask_type == 4 &&
                            prepared_window_mask_.ddr_tensor.defined(),
                        "exact W8 resident Vision mask is unavailable");
            sdpa_dma_mask_to_spm(
                prepared_window_mask_, addr_offset("sdpa_mask").value,
                ExactMimcPlan::kAttentionRows,
                ExactMimcPlan::kAttentionRows, NUM_CORES);
        }
        if (padded_pair_enabled()) {
            for (size_t i = 0; i < prepared_pair_masks_.size(); ++i) {
                TORCH_CHECK(prepared_pair_masks_[i].mask_type == 4 &&
                                prepared_pair_masks_[i].ddr_tensor.defined(),
                            "padded Vision resident masks are unavailable");
                sdpa_dma_mask_to_spm(
                    prepared_pair_masks_[i],
                    addr_offset(i == 0 ? "sdpa_mask" : "sdpa_full_mask").value,
                    336, 336, NUM_CORES);
            }
        } else if (small_image_resident_mask_enabled()) {
            TORCH_CHECK(prepared_window_mask_.mask_type == 4 &&
                            prepared_window_mask_.ddr_tensor.defined(),
                        "small resident Vision mask is unavailable");
            const int64_t rows = current_num_patches_ / image_batch_count_;
            sdpa_dma_mask_to_spm(
                prepared_window_mask_, addr_offset("sdpa_mask").value,
                rows, rows, NUM_CORES);
        }

        // Global: broadcast the 2D-RoPE cos/sin tables into SPM once (SPM-rope
        // variant). Flush first — same coherency care the DDR rope launcher
        // takes per-forward for these exact tensors.
        if (rope_spm_enabled_ && max_hw_ > 0) {
            int64_t tbl_elems = max_hw_ * (head_dim() / 4);
            rpu_ddr_flush(freq_cos_.data_ptr<c10::Half>());
            rpu_ddr_flush(freq_sin_.data_ptr<c10::Half>());
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN25_ROPE_COS_PRELOAD_SITE,
                static_cast<int64_t>(
                    Qwen25VisionMutableDmaRoute::ROPE_TABLE_DDR_TO_SPM),
                /*resolved_flags=*/0,
                {rope_spm_enabled_ ? 1 : 0, max_hw_, head_dim()});
            rpu_launch_ddr_broadcast_spm_dma(
                freq_cos_.data_ptr<c10::Half>(), tbl_elems, addr(0, "rope_cos"));
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN25_ROPE_SIN_PRELOAD_SITE,
                static_cast<int64_t>(
                    Qwen25VisionMutableDmaRoute::ROPE_TABLE_DDR_TO_SPM),
                /*resolved_flags=*/0,
                {rope_spm_enabled_ ? 1 : 0, max_hw_, head_dim()});
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
        if (layer_idx == 0) {
            rpu_launch_ddr_broadcast_spm_dma(
                bn.norm1_w.data_ptr<c10::Half>(), h, addr(0, "norm1_w"));
        }
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

    void emit_exact_mimc_qkv_body(
        int layer_idx, const ChunkInfo& packed_chunk) {
        TORCH_CHECK(
            packed_chunk.idx == 0 && packed_chunk.offset == 0 &&
                packed_chunk.len == ExactMimcPlan::kQkvRows
                && image_batch_count_ == ExactMimcPlan::kImages,
            "exact MIMC QKV stage requires one packed 1728-row chunk, got "
            "images=", image_batch_count_, " idx=", packed_chunk.idx,
            " offset=", packed_chunk.offset, " len=", packed_chunk.len);
        TORCH_CHECK(
            ctx().attention_policy ==
                AttentionExecutionPolicy::SPM_KV_BY_MHA,
            "exact MIMC requires the common raw-SPM attention policy");

        const auto& lw = layer_weights_[layer_idx];
        const int64_t images = ExactMimcPlan::kImages;
        const int64_t packed_seq = ExactMimcPlan::kQkvRows;
        const int64_t per_image_seq = ExactMimcPlan::kAttentionRows;
        const int64_t h = hidden_size();
        const int64_t nq = num_q_heads();
        const int64_t hd = head_dim();
        const int64_t local_q_dim = (nq / NUM_CORES) * hd;
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
                (image_idx + 1) * per_image_seq,
            };
        };
        auto packed_linear = [&](uint32_t input_addr,
                                 const at::Tensor& weight,
                                 uint32_t output_addr,
                                 int64_t N, int64_t K,
                                 int partition, uint32_t bias_addr,
                                 const at::Tensor& scale,
                                 int64_t invocation) {
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_EXACT_QKV_LINEAR_SITE,
                qwen25_vision_linear_route(), invocation);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                input_addr, weight, output_addr,
                packed_seq, N, K, partition, NUM_CORES, bias_addr, scale);
        };

        // Stage one layer's small parameters into the shared persistent slots.
        emit_exact_mimc_layer_params(layer_idx);

        // Phase 1: layer 0 packs caller DDR, extracts compact residual shards,
        // and normalizes. Later layers consume normalized full output plus the
        // compact raw shards produced by the previous layer's Down ring.
        for (int64_t i = 0; i < images; ++i) {
            const ChunkInfo chunk = image_chunk(i);
            const uint32_t norm_out = addr(0, "input_norm")
                + static_cast<uint32_t>(i) * h_image_bytes;
            if (layer_idx == 0) {
                emit_layer_input_dma(
                    layer_idx, chunk, "input_norm",
                    i * per_image_seq);
                const uint32_t compact_residual =
                    addr(0, "compact_residual")
                    + static_cast<uint32_t>(i)
                        * ExactMimcPlan::kResidualShardBytes;
                rpu_launch_spm_local_shard_copy_kernel(
                    norm_out, ExactMimcPlan::kResidualShardElements,
                    ExactMimcPlan::kResidualShardBytes,
                    compact_residual, NUM_CORES);
                rpu_launch_rmsnorm_spm_kernel(
                    norm_out, norm_out, addr(0, "norm1_w"),
                    per_image_seq, h, eps_,
                    consume_exact_rmsnorm_route(
                        kQwen25ExactPreRmsNormRowsSite,
                        per_image_seq, h, i));
            }
        }

        // Phase 2: exact mixed W8A16 emits Q/K/V together. The normalized
        // input becomes dead after that launch, so it receives transposed V;
        // this preserves the existing three Q/K/V buffers and SPM layout.
        const bool use_w8_multibase_qkv = exact_mimc_w8_qkv_enabled();
        if (use_w8_multibase_qkv) {
            rpu_launch_wall_vision_qkv_w8a16_multibase_spm_kernel(
                addr(0, "input_norm"), lw.q_w, lw.k_w, lw.v_w,
                addr(0, "q"), addr(0, "k"), addr(0, "v"),
                addr(0, "q_bias"), addr(0, "k_bias"), addr(0, "v_bias"),
                lw.q_ws, lw.k_ws, lw.v_ws,
                wall_sorting_);
            rpu_launch_v_transpose_spm(
                addr(0, "v"), addr(0, "input_norm"),
                ExactMimcPlan::kImages, ExactMimcPlan::kAttentionRows,
                nq, hd, NUM_CORES);
        } else {
            // FP16 keeps the original V-first sequence and buffer placement.
            packed_linear(
                addr(0, "input_norm"), lw.v_w, addr(0, "q"),
                nq * hd, h, /*col=*/1, addr(0, "v_bias"), lw.v_ws,
                /*invocation=*/0);
            rpu_launch_v_transpose_spm(
                addr(0, "q"), addr(0, "v"),
                ExactMimcPlan::kImages, ExactMimcPlan::kAttentionRows,
                nq, hd, NUM_CORES);
            packed_linear(
                addr(0, "input_norm"), lw.k_w, addr(0, "k"),
                nq * hd, h, /*col=*/1, addr(0, "k_bias"), lw.k_ws,
                /*invocation=*/1);
            packed_linear(
                addr(0, "input_norm"), lw.q_w, addr(0, "q"),
                nq * hd, h, /*col=*/1, addr(0, "q_bias"), lw.q_ws,
                /*invocation=*/2);
        }

        // Phase 3: RoPE remains image-local because each image owns its 2-D
        // positions. The by-MHA kernel then consumes all three independent
        // batches in one launch; K/V never leave SPM or enter the KV cache.
        const int64_t local_heads = nq / NUM_CORES;
        const int64_t head_dim_pad = hd;
        const double attn_scale =
            1.0 / std::sqrt(static_cast<double>(orig_head_dim_));
        const bool native_window_sdpa =
            exact_mimc_native_window_sdpa_enabled();
        const bool is_full = !window_mask_present_ ||
            std::find(fullatt_block_indexes_.begin(),
                      fullatt_block_indexes_.end(),
                      static_cast<int64_t>(layer_idx))
                != fullatt_block_indexes_.end();
        const int mask_type = is_full || native_window_sdpa
            ? 0 : prepared_window_mask_.mask_type;
        uint32_t mask_addr = 0;
        if (!is_full && !native_window_sdpa) {
            mask_addr = addr(0, "sdpa_mask");
            if (!exact_mimc_fused_w8_gate_up_enabled()) {
                sdpa_dma_mask_to_spm(
                    prepared_window_mask_, addr_offset("sdpa_mask").value,
                    per_image_seq, per_image_seq, NUM_CORES);
            }
        }

        for (int64_t i = 0; i < images; ++i) {
            const uint32_t q_abs =
                addr(0, "q") + static_cast<uint32_t>(i) * q_image_bytes;
            const uint32_t k_abs =
                addr(0, "k") + static_cast<uint32_t>(i) * q_image_bytes;
            const int64_t pos_offset = i * per_image_seq;
            if (rope_spm_enabled_) {
                consume_manifest_route(
                    FmbRouteFamily::ROPE,
                    QWEN25_EXACT_Q_ROPE_SPM_SITE,
                    static_cast<int64_t>(
                        Qwen25VisionRopeRoute::ROPE_2D_SPM), i,
                    {rope_spm_enabled_ ? 1 : 0},
                    FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                rpu_launch_rope_2d_spm_kernel(
                    q_abs, q_abs, addr(0, "rope_cos"), addr(0, "rope_sin"),
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    pos_offset, per_image_seq, local_heads,
                    hd, head_dim_pad, NUM_CORES);
                consume_manifest_route(
                    FmbRouteFamily::ROPE,
                    QWEN25_EXACT_K_ROPE_SPM_SITE,
                    static_cast<int64_t>(
                        Qwen25VisionRopeRoute::ROPE_2D_SPM), i,
                    {rope_spm_enabled_ ? 1 : 0},
                    FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                rpu_launch_rope_2d_spm_kernel(
                    k_abs, k_abs, addr(0, "rope_cos"), addr(0, "rope_sin"),
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    pos_offset, per_image_seq, local_heads,
                    hd, head_dim_pad, NUM_CORES);
            } else {
                consume_manifest_route(
                    FmbRouteFamily::ROPE,
                    QWEN25_EXACT_Q_ROPE_DDR_SITE,
                    static_cast<int64_t>(
                        Qwen25VisionRopeRoute::ROPE_2D_DDR), i,
                    {rope_spm_enabled_ ? 1 : 0},
                    FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
                rpu_launch_rope_2d_ddr_kernel(
                    q_abs, q_abs,
                    freq_cos_.data_ptr<c10::Half>(),
                    freq_sin_.data_ptr<c10::Half>(),
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    pos_offset, per_image_seq, local_heads,
                    hd, head_dim_pad, NUM_CORES);
                consume_manifest_route(
                    FmbRouteFamily::ROPE,
                    QWEN25_EXACT_K_ROPE_DDR_SITE,
                    static_cast<int64_t>(
                        Qwen25VisionRopeRoute::ROPE_2D_DDR), i,
                    {rope_spm_enabled_ ? 1 : 0},
                    FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
                rpu_launch_rope_2d_ddr_kernel(
                    k_abs, k_abs,
                    freq_cos_.data_ptr<c10::Half>(),
                    freq_sin_.data_ptr<c10::Half>(),
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    pos_offset, per_image_seq, local_heads,
                    hd, head_dim_pad, NUM_CORES);
            }

        }
        // Q/output aliasing is not safe for this multi-block kernel. W8 can
        // overwrite the dead pre-transpose V slot; FP16 keeps its transposed V
        // there and therefore retains the established scratch-to-Q copy.
        if (use_w8_multibase_qkv) {
            if (native_window_sdpa && !is_full) {
                consume_manifest_route(
                    FmbRouteFamily::ATTENTION,
                    QWEN25_EXACT_WINDOW_ATTN_SITE,
                    static_cast<int64_t>(
                        AttentionExecutionPolicy::SPM_KV_BY_MHA),
                    /*invocation=*/0,
                    {static_cast<int64_t>(schedule_),
                     per_window_sdpa_enabled_ ? 1 : 0});
                rpu_launch_wall_window_sdpa_by_mha_spm(
                    addr(0, "q"), addr(0, "k"), addr(0, "input_norm"),
                    addr(0, "v"), attn_scale);
            } else {
                consume_manifest_route(
                    FmbRouteFamily::ATTENTION,
                    QWEN25_EXACT_W8_ATTN_SITE,
                    static_cast<int64_t>(
                        AttentionExecutionPolicy::SPM_KV_BY_MHA),
                    /*invocation=*/0,
                    {static_cast<int64_t>(schedule_),
                     per_window_sdpa_enabled_ ? 1 : 0});
                rpu_launch_sdpa_by_mha_spm(
                    addr(0, "q"), addr(0, "k"), addr(0, "input_norm"),
                    addr(0, "v"), mask_addr, mask_type, attn_scale,
                    ExactMimcPlan::kImages,
                    ExactMimcPlan::kAttentionRows,
                    ExactMimcPlan::kAttentionRows,
                    nq, nq, hd, NUM_CORES);
            }
        } else {
            const uint32_t attn_scratch = addr(0, "input_norm")
                + packed_h_bytes - packed_q_bytes;
            consume_manifest_route(
                FmbRouteFamily::ATTENTION,
                QWEN25_EXACT_FP16_ATTN_SITE,
                static_cast<int64_t>(
                    AttentionExecutionPolicy::SPM_KV_BY_MHA),
                /*invocation=*/0,
                {static_cast<int64_t>(schedule_),
                 per_window_sdpa_enabled_ ? 1 : 0});
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q"), addr(0, "k"), addr(0, "v"),
                attn_scratch, mask_addr, mask_type, attn_scale,
                ExactMimcPlan::kImages,
                ExactMimcPlan::kAttentionRows,
                ExactMimcPlan::kAttentionRows,
                nq, nq, hd, NUM_CORES);
            rpu_launch_spm_local_copy_dma(
                attn_scratch, packed_seq * local_q_dim,
                addr(0, "q"), NUM_CORES);
        }

    }

    void build_exact_mimc_compute_body(
        int layer_idx, const ChunkInfo& chunk) {
        TORCH_CHECK(
            chunk.idx >= 0 && chunk.idx < ExactMimcPlan::kImages &&
                chunk.offset ==
                    static_cast<int64_t>(chunk.idx) *
                        ExactMimcPlan::kMlpRows &&
                chunk.len == ExactMimcPlan::kMlpRows,
            "exact MIMC compute stage requires three contiguous M576 chunks");

        const auto& lw = layer_weights_[layer_idx];
        const int64_t packed_seq = ExactMimcPlan::kQkvRows;
        const int64_t mlp_rows = chunk.len;
        const int64_t mlp_begin = chunk.offset;
        const int64_t h = hidden_size();
        const int64_t nq = num_q_heads();
        const int64_t hd = head_dim();
        const int64_t is_ = intermediate_size();
        const int64_t local_inter = is_ / NUM_CORES;
        const bool residual_is_bf16 =
            mixed_precision_
            && layer_idx > QWEN25VL_VISION_MIXED_BOUNDARY_LAYER;
        const bool boundary_down_is_bf16 =
            mixed_precision_
            && layer_idx == QWEN25VL_VISION_MIXED_BOUNDARY_LAYER;
        const bool fuse_w8_silu_mul =
            exact_mimc_fused_w8_gate_up_enabled();
        const bool compact_local_inout =
            exact_mimc_compact_local_inout_enabled();
        const bool row_weight_outer =
            wall_sorting_.vision_row_reuse && compact_local_inout;
        const bool has_next_layer = layer_idx + 1 < num_layers();
        TORCH_INTERNAL_ASSERT(!compact_local_inout || fuse_w8_silu_mul);
        // The compact merger path is layer-major across all three images.
        // Chunk 0 records the whole layer; the remaining scheduled chunks are
        // intentionally empty so GateUp and W8 Down can reuse weight tiles.
        if (compact_local_inout && chunk.idx != 0) {
            return;
        }
        // The first compute chunk performs the one packed O projection. Its
        // output aliases input_norm; later image chunks consume their untouched
        // slices before overwriting them with the final MLP result.
        if (chunk.idx == 0) {
            const uint32_t attention_output =
                exact_mimc_w8_qkv_enabled()
                    ? addr(0, "v") : addr(0, "q");
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_EXACT_O_LINEAR_SITE,
                row_weight_outer
                    ? static_cast<int64_t>(Qwen25VisionLinearRoute::WALL_ROW_FULL_LOCAL_K)
                    : qwen25_vision_linear_route(), chunk.idx);
            if (row_weight_outer) {
                rpu_launch_wall_vision_row_weight_outer_w8a16_spm_kernel(
                    attention_output, lw.o_w, addr(0, "oproj"),
                packed_seq, h, nq * hd, addr(0, "o_bias"), lw.o_ws, wall_sorting_);
            } else {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                attention_output, lw.o_w, addr(0, "oproj"),
                packed_seq, h, nq * hd, /*row=*/0, NUM_CORES,
                addr(0, "o_bias"), lw.o_ws);
            }
        }

        const int first_image = compact_local_inout ? 0 : chunk.idx;
        const int image_count = compact_local_inout
            ? ExactMimcPlan::kImages : 1;
        const uint32_t hidden_image_stride = static_cast<uint32_t>(
            ExactMimcPlan::kMlpRows * h * DWIDTH);
        const uint32_t gate_image_stride = static_cast<uint32_t>(
            ExactMimcPlan::kMlpRows * local_inter * DWIDTH);
        for (int image = first_image;
             image < first_image + image_count; ++image) {
            const uint32_t image_offset =
                static_cast<uint32_t>(image) * hidden_image_stride;
            const uint32_t packed_h_addr =
                addr(0, "input_norm") + image_offset;
            const uint32_t partial_addr = addr(0, "oproj") + image_offset;
            const uint32_t residual_local = addr(0, "compact_residual")
                + static_cast<uint32_t>(image)
                    * ExactMimcPlan::kResidualShardBytes;
            if (compact_local_inout) {
                TORCH_INTERNAL_ASSERT(partial_addr == packed_h_addr);
                if (residual_is_bf16) {
                    rpu_launch_wall_ring_postnorm_m576_fp16_partial_bf16_residual_local_inout_kernel(
                        partial_addr, residual_local, addr(0, "norm2_w"), eps_);
                } else {
                    rpu_launch_wall_ring_postnorm_m576_fp16_partial_fp16_residual_local_inout_kernel(
                        partial_addr, residual_local, addr(0, "norm2_w"), eps_);
                }
            } else {
                if (residual_is_bf16) {
                    consume_manifest_route(
                        FmbRouteFamily::ALL_REDUCE,
                        QWEN25_EXACT_ATTN_MIXED_LOCAL_REDUCE_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionAllReduceRoute::
                                MIXED_LOCAL_RESIDUAL),
                        chunk.idx);
                    rpu_launch_all_reduce_fp16_partial_bf16_residual_local_spm_kernel(
                        partial_addr, residual_local, addr(0, "o_result"),
                        mlp_rows, h);
                    rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                        addr(0, "o_result"), packed_h_addr,
                        addr(0, "norm2_w"), mlp_rows, h, eps_,
                        consume_mixed_rmsnorm_route(
                            QWEN25_EXACT_COMPUTE_MIXED_RMSNORM_SITE,
                            mlp_rows, h, image));
                } else {
                    consume_manifest_route(
                        FmbRouteFamily::ALL_REDUCE,
                        QWEN25_EXACT_ATTN_LOCAL_REDUCE_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionAllReduceRoute::
                                RING_LOCAL_RESIDUAL),
                        chunk.idx);
                    rpu_launch_all_reduce_sum_residual_local_spm_kernel(
                        partial_addr, residual_local, addr(0, "o_result"),
                        mlp_rows, h, NUM_CORES, NUM_CORES);
                    rpu_launch_rmsnorm_spm_kernel(
                        addr(0, "o_result"), packed_h_addr,
                        addr(0, "norm2_w"), mlp_rows, h, eps_,
                        consume_exact_rmsnorm_route(
                            kQwen25ExactComputeRmsNormRowsSite,
                            mlp_rows, h, image));
                }
            }
        }

        const uint32_t first_packed_h_addr = addr(0, "input_norm")
            + static_cast<uint32_t>(first_image) * hidden_image_stride;
        const uint32_t gate_base = addr(0, "gate");
        if (fuse_w8_silu_mul) {
            rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel(
                first_packed_h_addr, lw.gate_w, lw.up_w, gate_base,
                addr(0, "gate_bias"), addr(0, "up_bias"),
                lw.gate_ws, lw.up_ws,
                compact_local_inout
                    ? first_packed_h_addr + hidden_image_stride : 0,
                compact_local_inout
                    ? first_packed_h_addr + 2 * hidden_image_stride : 0,
                compact_local_inout ? gate_base + gate_image_stride : 0,
                compact_local_inout ? gate_base + 2 * gate_image_stride : 0, wall_sorting_);
        } else {
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_EXACT_GATE_LINEAR_SITE,
                qwen25_vision_linear_route(), chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                first_packed_h_addr, lw.gate_w, gate_base,
                mlp_rows, is_, h, /*col=*/1, NUM_CORES,
                addr(0, "gate_bias"), lw.gate_ws);
            rpu_launch_eltwise_unary_spm_kernel(
                gate_base, gate_base,
                mlp_rows * local_inter, ValuOpType::SILU);
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_EXACT_UP_LINEAR_SITE,
                qwen25_vision_linear_route(), chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                first_packed_h_addr, lw.up_w, addr(0, "up"),
                mlp_rows, is_, h, /*col=*/1, NUM_CORES,
                addr(0, "up_bias"), lw.up_ws);
            rpu_launch_eltwise_binary_spm_kernel(
                gate_base, addr(0, "up"), gate_base,
                mlp_rows * local_inter, ValuOpType::MUL,
                c10::Half(1.0));
        }
        if (has_next_layer && chunk.idx == 0) {
            auto& next_bn = layer_bias_norm_[layer_idx + 1];
            rpu_launch_ddr_broadcast_spm_dma(
                next_bn.norm1_w.data_ptr<c10::Half>(), h,
                addr(0, "norm1_w"));
        }
        const int64_t down_rows = compact_local_inout
            ? packed_seq : mlp_rows;
        if (!boundary_down_is_bf16) {
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_EXACT_DOWN_LINEAR_SITE,
                row_weight_outer
                    ? static_cast<int64_t>(Qwen25VisionLinearRoute::WALL_ROW_FULL_LOCAL_K)
                    : qwen25_vision_linear_route(), chunk.idx,
                {mixed_precision_ ? 1 : 0});
            if (row_weight_outer) {
                rpu_launch_wall_vision_row_weight_outer_w8a16_spm_kernel(
                    gate_base, lw.down_w, first_packed_h_addr,
                down_rows, h, is_, addr(0, "down_bias"), lw.down_ws, wall_sorting_);
            } else {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                gate_base, lw.down_w, first_packed_h_addr,
                down_rows, h, is_, /*row=*/0, NUM_CORES,
                addr(0, "down_bias"), lw.down_ws);
            }
        }
        for (int image = first_image;
             image < first_image + image_count; ++image) {
            const uint32_t image_offset =
                static_cast<uint32_t>(image) * hidden_image_stride;
            const uint32_t packed_h_addr =
                addr(0, "input_norm") + image_offset;
            if (boundary_down_is_bf16) {
                const uint32_t gate_addr = gate_base
                    + (compact_local_inout
                           ? static_cast<uint32_t>(image) * gate_image_stride
                           : 0);
                consume_manifest_route(
                    FmbRouteFamily::LINEAR,
                    QWEN25_EXACT_DOWN_BF16_LINEAR_SITE,
                    static_cast<int64_t>(
                        Qwen25VisionLinearRoute::ACC32_OUT_BF16), image);
                rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(
                    gate_addr, lw.down_w, packed_h_addr,
                    mlp_rows, h, is_, /*row=*/0, NUM_CORES,
                    addr(0, "down_bias"));
            }
            const uint32_t residual_local = addr(0, "compact_residual")
                + static_cast<uint32_t>(image)
                    * ExactMimcPlan::kResidualShardBytes;
            if (compact_local_inout) {
                const uint32_t postnorm_gamma = has_next_layer
                    ? addr(0, "norm1_w") : addr(0, "merger_ln_q_w");
                if (boundary_down_is_bf16) {
                    rpu_launch_wall_ring_postnorm_m576_bf16_partial_fp16_residual_local_inout_kernel(
                        packed_h_addr, residual_local, postnorm_gamma, eps_);
                } else if (residual_is_bf16) {
                    rpu_launch_wall_ring_postnorm_m576_fp16_partial_bf16_residual_local_inout_kernel(
                        packed_h_addr, residual_local, postnorm_gamma, eps_);
                } else {
                    rpu_launch_wall_ring_postnorm_m576_fp16_partial_fp16_residual_local_inout_kernel(
                        packed_h_addr, residual_local, postnorm_gamma, eps_);
                }
            } else if (has_next_layer) {
                if (boundary_down_is_bf16) {
                    rpu_launch_wall_down_postnorm_m576_bf16_partial_fp16_residual_kernel(
                        packed_h_addr, addr(0, "o_result"), residual_local,
                        addr(0, "norm1_w"), eps_);
                } else if (residual_is_bf16) {
                    rpu_launch_wall_down_postnorm_m576_fp16_partial_bf16_residual_kernel(
                        packed_h_addr, addr(0, "o_result"), residual_local,
                        addr(0, "norm1_w"), eps_);
                } else {
                    rpu_launch_wall_down_postnorm_m576_fp16_partial_fp16_residual_kernel(
                        packed_h_addr, addr(0, "o_result"), residual_local,
                        addr(0, "norm1_w"), eps_);
                }
            } else {
                if (boundary_down_is_bf16) {
                    consume_manifest_route(
                        FmbRouteFamily::ALL_REDUCE,
                        QWEN25_EXACT_FINAL_BF16_REDUCE_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionAllReduceRoute::
                                EXACT_M576_BF16_PARTIAL_FP16_RESIDUAL),
                        image);
                    rpu_launch_wall_vision_all_reduce_bf16_partial_fp16_residual_m576_inplace_kernel(
                        packed_h_addr, addr(0, "o_result"));
                } else if (residual_is_bf16) {
                    consume_manifest_route(
                        FmbRouteFamily::ALL_REDUCE,
                        QWEN25_EXACT_FINAL_MIXED_REDUCE_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionAllReduceRoute::
                                EXACT_M576_FP16_PARTIAL_BF16_RESIDUAL),
                        image);
                    rpu_launch_wall_vision_all_reduce_fp16_partial_bf16_residual_m576_inplace_kernel(
                        packed_h_addr, addr(0, "o_result"));
                } else {
                    consume_manifest_route(
                        FmbRouteFamily::ALL_REDUCE,
                        QWEN25_EXACT_FINAL_REDUCE_SITE,
                        static_cast<int64_t>(
                            Qwen25VisionAllReduceRoute::
                                EXACT_M576_SUM_RESIDUAL),
                        image);
                    rpu_launch_wall_vision_all_reduce_sum_residual_m576_inplace_kernel(
                        packed_h_addr, addr(0, "o_result"));
                }
            }
        }

        // Inner layers stay entirely in SPM. Only a standalone encoder result
        // needs a final DDR publication; the fused merger consumes input_norm.
        if (layer_idx + 1 == num_layers() && !merger_active()) {
            emit_layer_output_dma(
                layer_idx, chunk, "input_norm",
                static_cast<uint32_t>(mlp_begin * h * DWIDTH));
        }
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        if (is_exact_mimc()) {
            build_exact_mimc_compute_body(layer_idx, chunk);
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

        const bool down_owner_norm = small_image_down_owner_norm_enabled(seq_len);
        if (down_owner_norm) {
            validate_small_image_down_owner_context(layer_idx, chunk);
        }
        // The preceding Down already left the next Norm1 in input_norm.
        // Layer zero still imports and normalizes the original input.
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }
        if (residual_is_bf16) {
            rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                addr(0, "residual1"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "norm1_w"), seq_len, h, eps_,
                consume_mixed_rmsnorm_route(
                    QWEN25_INPUT_MIXED_RMSNORM_SITE,
                    seq_len, h, chunk.idx));
        } else if (!down_owner_norm || layer_idx == 0) {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual1"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "norm1_w"), seq_len, h, eps_,
                consume_small_image_rmsnorm_route(
                    QWEN25_SMALL_INPUT_RMSNORM_SITE, seq_len, chunk.idx));
        }
        emit_debug_phase(0, "input_norm");

        // Phase 2: Q / K / V Linear with bias (col-partition).
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN25_Q_LINEAR_SITE,
            qwen25_vision_linear_route(), chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "q_bias"), lw.q_ws,
                /*nvfp4_scale=*/0, /*nvfp4_layer=*/0, /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN25_K_LINEAR_SITE,
            qwen25_vision_linear_route(), chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "k_bias"), lw.k_ws,
                /*nvfp4_scale=*/0, /*nvfp4_layer=*/0, /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN25_V_LINEAR_SITE,
            qwen25_vision_linear_route(), chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "v_bias"), lw.v_ws,
                /*nvfp4_scale=*/0, /*nvfp4_layer=*/0, /*force_acc32=*/linear_acc32_);

        // Phase 2.5: 2D RoPE on Q and K (in-place). Each core holds nq/8=2 heads.
        const int64_t local_heads = nq / NUM_CORES;
        const int64_t head_dim_pad = hd;  // head_dim=80, no padding.
        if (small_image_qk_rope_enabled(seq_len)) {
            consume_small_image_qk_rope(seq_len, chunk.offset, chunk.idx);
            rpu_launch_wall_base_vision_qk_rope_2d_spm_kernel(
                addr(0, "q"), addr(0, "k"), addr(0, "rope_cos"), addr(0, "rope_sin"),
                position_idx_keepalive_, chunk.offset, seq_len, NUM_CORES);
        } else if (rope_spm_enabled_) {
            // SPM-resident cos/sin tables (broadcast once in emit_preload_weights).
            const uint32_t cos_a = addr(0, "rope_cos");
            const uint32_t sin_a = addr(0, "rope_sin");
            consume_manifest_route(
                FmbRouteFamily::ROPE, QWEN25_Q_ROPE_SPM_SITE,
                static_cast<int64_t>(
                    Qwen25VisionRopeRoute::ROPE_2D_SPM), chunk.idx,
                {rope_spm_enabled_ ? 1 : 0},
                FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
            rpu_launch_rope_2d_spm_kernel(
                addr(0, "q"), addr(0, "q"), cos_a, sin_a,
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset, seq_len,
                local_heads, hd, head_dim_pad, NUM_CORES);
            consume_manifest_route(
                FmbRouteFamily::ROPE, QWEN25_K_ROPE_SPM_SITE,
                static_cast<int64_t>(
                    Qwen25VisionRopeRoute::ROPE_2D_SPM), chunk.idx,
                {rope_spm_enabled_ ? 1 : 0},
                FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
            rpu_launch_rope_2d_spm_kernel(
                addr(0, "k"), addr(0, "k"), cos_a, sin_a,
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset, seq_len,
                local_heads, hd, head_dim_pad, NUM_CORES);
        } else {
            consume_manifest_route(
                FmbRouteFamily::ROPE, QWEN25_Q_ROPE_DDR_SITE,
                static_cast<int64_t>(
                    Qwen25VisionRopeRoute::ROPE_2D_DDR), chunk.idx,
                {rope_spm_enabled_ ? 1 : 0},
                FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
            rpu_launch_rope_2d_ddr_kernel(
                addr(0, "q"), addr(0, "q"),
                freq_cos_.data_ptr<c10::Half>(), freq_sin_.data_ptr<c10::Half>(),
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset, seq_len,
                local_heads, hd, head_dim_pad, NUM_CORES);
            consume_manifest_route(
                FmbRouteFamily::ROPE, QWEN25_K_ROPE_DDR_SITE,
                static_cast<int64_t>(
                    Qwen25VisionRopeRoute::ROPE_2D_DDR), chunk.idx,
                {rope_spm_enabled_ ? 1 : 0},
                FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
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
        if (small_image_raw_kv_only_enabled()) {
            TORCH_CHECK(ctx().has_complete_physical_manifest() &&
                            ctx().attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA,
                        "small raw-only Vision requires its COMPLETE raw-SPM plan");
            // Q/K/V remain in the declared phase-2..3 SPM buffers. The exact
            // full/window attention consumers below never read RPUCache. Do
            // not emit a durable DDR copy solely for the discarded candidate.
        } else {
            const KvInsertSegmentPlan kv_plan = [&] {
                if (!ctx().has_complete_physical_manifest()) {
                    return rpu_resolve_kvinsert_segment_plan_auto(
                        /*position=*/0, seq_len, seq_len, NUM_CORES, nq, hd,
                        QWEN25_KV_CAPABILITIES);
                }
                const FmbRouteManifestEntry& route = ctx().find_physical_route(
                    FmbRouteFamily::KV_INSERT, QWEN25_KV_INSERT_SITE,
                    chunk.idx);
                const KvInsertSegmentPlan planned =
                    restore_kvinsert_plan(
                        QWEN25_KV_INSERT_SITE, route.arguments, NUM_CORES, nq, hd);
                TORCH_CHECK(
                    planned.logical_rows() == seq_len &&
                        planned.physical_rows() == seq_len &&
                        planned.segment(0).position == 0,
                    "Qwen2.5-VL Vision KV descriptor geometry drift");
                ctx().consume_physical_route(
                    FmbRouteFamily::KV_INSERT, QWEN25_KV_INSERT_SITE,
                    static_cast<int64_t>(planned.route()),
                    QWEN25_KV_FLAG_DDR_MIRROR, route.arguments, chunk.idx);
                return planned;
            }();
            rpu_launch_insert_kvcache_spm_unified_with_plan(
                k_cache, v_cache,
                addr_offset("k").value, addr_offset("v").value,
                nq, hd, NUM_CORES,
                /*k_cache_batch_offset_elems=*/0,
                /*v_cache_batch_offset_elems=*/0,
                /*spm_rows=*/0, kv_plan);
        }

        double attn_scale = 1.0 / std::sqrt(static_cast<double>(orig_head_dim_));

        // P3b: window layers get a dense block-diagonal MASK_2D; fullatt layers
        // (and the whole P3a no-window path) stay MASK_NONE. The mask MUST be
        // re-uploaded for the ordinary phase-aliased mask. The cold small-image
        // route retains it across layers, or uses actual native windows.
        bool is_full = !(window_mask_present_ || per_window_sdpa_enabled_) ||
            std::find(fullatt_block_indexes_.begin(), fullatt_block_indexes_.end(),
                      static_cast<int64_t>(layer_idx)) != fullatt_block_indexes_.end();
        int mask_type = padded_pair_enabled() ? 4 :
            ((is_full || per_window_sdpa_enabled_)
                ? 0 : prepared_window_mask_.mask_type);  // 0 / 4

        if (!is_full && small_image_ragged_sdpa_enabled(seq_len)) {
            const int64_t rows = seq_len / image_batch_count_;
            const int64_t qd = (nq / NUM_CORES) * hd;
            for (int64_t image = 0; image < image_batch_count_; ++image) {
                consume_small_image_ragged_sdpa(rows, image);
                const uint32_t offset = static_cast<uint32_t>(image * rows * qd * DWIDTH);
                rpu_launch_v_transpose_spm(addr(0, "v") + offset, addr(0, "sdpa_tmp"),
                    1, rows, nq, hd, NUM_CORES);
                rpu_launch_wall_base_vision_ragged_sdpa_spm(
                    addr(0, "q") + offset, addr(0, "k") + offset, addr(0, "sdpa_tmp"),
                    addr(0, "sdpa_out") + offset, addr(0, "sdpa_mask"),
                    attn_scale, rows, NUM_CORES);
            }
        } else if (small_image_native_window_enabled()) {
            const GenericAttentionGeometry site{
                is_full ? QWEN25_MISC_DDR_ATTN_SITE : QWEN25_WINDOW_DDR_ATTN_SITE,
                chunk.idx, image_batch_count_ * (is_full ? 1 : 4), is_full ? 256 : 64,
                is_full ? 256 : 64, int64_t{1}};
            const AttentionExecutionPolicy policy = generic_attention_policy(site, 0);
            TORCH_CHECK(policy == AttentionExecutionPolicy::SPM_KV_BY_MHA,
                        "small native-window Vision requires its planned raw-SPM route");
            consume_generic_attention_route(
                site, 0, policy,
                is_full ? QWEN25_MISC_RAW_ATTN_SITE : QWEN25_WINDOW_RAW_ATTN_SITE);
            rpu_launch_v_transpose_spm(
                addr(0, "v"), addr(0, "sdpa_tmp"), site.batch, site.seq_k,
                nq, hd, NUM_CORES);
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q"), addr(0, "k"), addr(0, "sdpa_tmp"),
                addr(0, "sdpa_out"), 0, 0, attn_scale,
                site.batch, site.seq_q, site.seq_k, nq, nq, hd, NUM_CORES);
        } else if (is_generic() && schedule_ == VisionSchedule::MISC) {
            // Batched multi-image: g per-image SDPA calls, each at the proven ≤256
            // config (full → MASK_NONE, window → shared [pic,pic] MASK_2D). Keeping
            // every SDPA at per_image_ctx (a) dodges the documented ≥512 unified
            // numeric failure and (b) lets sdpa_tmp be sized for 256 not the packed
            // 768 — the smaller scratch is what lets 3×256 fit a single chunk. image
            // i's K/V live at cache rows [i*pic:(i+1)*pic] (contiguous insert at 0).
            int64_t per_image_ctx = seq_len / image_batch_count_;
            uint32_t mask_off = 0;
            if (mask_type == 4) {
                mask_off = addr_offset(padded_pair_enabled() && is_full
                    ? "sdpa_full_mask" : "sdpa_mask").value;
                if (!small_image_resident_mask_enabled()) {
                    sdpa_dma_mask_to_spm(prepared_window_mask_, mask_off,
                                         per_image_ctx, per_image_ctx, NUM_CORES);
                }
            }
            int64_t chunks = per_image_ctx / 16;          // seq-chunks per image
            int64_t qd = (nq / NUM_CORES) * hd;           // per-core q stride (elems)
            const int64_t mask_domain =
                generic_shared_attention_mask_domain();
            for (int64_t i = 0; i < image_batch_count_; ++i) {
                auto k_slice = k_cache.narrow(1, i * chunks, chunks);
                auto v_slice = v_cache.narrow(1, i * chunks, chunks);
                TORCH_CHECK(k_slice.is_contiguous() && v_slice.is_contiguous(),
                            "per-image KV slice not contiguous; need an offset launcher");
                uint32_t q_off   = addr_offset("q").value
                                   + (uint32_t)(i * per_image_ctx * qd * DWIDTH);
                uint32_t out_off = addr_offset("sdpa_out").value
                                   + (uint32_t)(i * per_image_ctx * qd * DWIDTH);
                const GenericAttentionGeometry site{
                    QWEN25_MISC_DDR_ATTN_SITE,
                    chunk.idx * max_keepalive_seq_ + i,
                    /*batch=*/1, per_image_ctx, per_image_ctx, mask_domain};
                const AttentionExecutionPolicy policy =
                    generic_attention_policy(site, mask_type);
                if (policy ==
                    AttentionExecutionPolicy::SPM_KV_BY_MHA) {
                    consume_generic_attention_route(
                        site, mask_type,
                        AttentionExecutionPolicy::SPM_KV_BY_MHA,
                        QWEN25_MISC_RAW_ATTN_SITE);
                    const uint32_t byte_offset = static_cast<uint32_t>(
                        i * per_image_ctx * qd * DWIDTH);
                    rpu_launch_v_transpose_spm(
                        addr(0, "v") + byte_offset, addr(0, "sdpa_tmp"),
                        /*batch=*/1, per_image_ctx, nq, hd, NUM_CORES);
                    rpu_launch_sdpa_by_mha_spm(
                        addr(0, "q") + byte_offset,
                        addr(0, "k") + byte_offset,
                        addr(0, "sdpa_tmp"),
                        addr(0, "sdpa_out") + byte_offset,
                        mask_type == 4 ? SPM_ALLOC.addr(0, mask_off) : 0,
                        mask_type, attn_scale,
                        /*batch=*/1, per_image_ctx, per_image_ctx,
                        nq, nq, hd, NUM_CORES);
                } else {
                    consume_generic_attention_route(
                        site, mask_type, AttentionExecutionPolicy::DDR_KV,
                        QWEN25_MISC_DDR_ATTN_SITE);
                    rpu_launch_sdpa_spm_unified_kernel_v2(
                        k_slice, v_slice, mask_type, attn_scale,
                        q_off, out_off, addr_offset("sdpa_tmp").value,
                        mask_off, per_image_ctx, nq, nq, hd,
                        per_image_ctx, NUM_CORES, NUM_CORES);
                }
            }
        } else if (!is_full && per_window_sdpa_enabled_) {
            // Controlled single-image path: window-reordered patches make each
            // [begin,end) range contiguous in Q/output and KV-cache storage. Run
            // MASK_NONE inside each window, avoiding the dense [seq,seq] mask.
            const int64_t qd = (nq / NUM_CORES) * hd;
            int64_t route_invocation = 0;
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
                const uint32_t window_byte_offset = static_cast<uint32_t>(
                    window_begin * qd * DWIDTH);
                bool raw_v_ready = false;
                for (int64_t q_rel = 0; q_rel < window_len;
                     q_rel += query_chunk) {
                    const int64_t query_len =
                        std::min(query_chunk, window_len - q_rel);
                    const int64_t query_begin = window_begin + q_rel;
                    const uint32_t q_off = addr_offset("q").value
                        + static_cast<uint32_t>(query_begin * qd * DWIDTH);
                    const uint32_t out_off = addr_offset("sdpa_out").value
                        + static_cast<uint32_t>(query_begin * qd * DWIDTH);
                    const GenericAttentionGeometry site{
                        QWEN25_WINDOW_DDR_ATTN_SITE,
                        chunk.idx * max_keepalive_seq_ +
                            route_invocation++,
                        /*batch=*/1, query_len, window_len,
                        /*MASK_NONE=*/1};
                    const AttentionExecutionPolicy policy =
                        generic_attention_policy(site, /*mask_type=*/0);
                    if (policy ==
                        AttentionExecutionPolicy::SPM_KV_BY_MHA) {
                        consume_generic_attention_route(
                            site, /*mask_type=*/0,
                            AttentionExecutionPolicy::SPM_KV_BY_MHA,
                            QWEN25_WINDOW_RAW_ATTN_SITE);
                        if (!raw_v_ready) {
                            rpu_launch_v_transpose_spm(
                                addr(0, "v") + window_byte_offset,
                                addr(0, "sdpa_tmp"), /*batch=*/1,
                                window_len, nq, hd, NUM_CORES);
                            raw_v_ready = true;
                        }
                        const uint32_t query_byte_offset =
                            static_cast<uint32_t>(
                                query_begin * qd * DWIDTH);
                        rpu_launch_sdpa_by_mha_spm(
                            addr(0, "q") + query_byte_offset,
                            addr(0, "k") + window_byte_offset,
                            addr(0, "sdpa_tmp"),
                            addr(0, "sdpa_out") + query_byte_offset,
                            /*mask_spm=*/0, /*mask_type=*/0, attn_scale,
                            /*batch=*/1, query_len, window_len,
                            nq, nq, hd, NUM_CORES);
                    } else {
                        consume_generic_attention_route(
                            site, /*mask_type=*/0,
                            AttentionExecutionPolicy::DDR_KV,
                            QWEN25_WINDOW_DDR_ATTN_SITE);
                        raw_v_ready = false;
                        rpu_launch_sdpa_spm_unified_kernel_v2(
                            k_slice, v_slice, /*mask_type=*/0, attn_scale,
                            q_off, out_off, addr_offset("sdpa_tmp").value,
                            /*mask_off=*/0, query_len, nq, nq, hd,
                            window_len, NUM_CORES, NUM_CORES);
                    }
                }
            }
        } else {
            uint32_t mask_off = 0;
            if (!is_full) {
                mask_off = addr_offset("sdpa_mask").value;
                if (!small_image_resident_mask_enabled()) {
                    sdpa_dma_mask_to_spm(prepared_window_mask_, mask_off,
                                         seq_len, seq_len, NUM_CORES);
                }
            }

            // The unified fp16 kernel is numerically unsafe for a 576-query
            // launch. Keep the full K/V cache (cross-patch attention remains
            // global) and normally strip the Q/output rows, mirroring Qwen3-VL's
            // proven 144-query KV_FIRST compute phase. Exact cold M324 raw-only
            // uses one by-MHA query launch with its own COMPLETE geometry.
            // MASK_2D is row-major
            // [seq_q, aligned_seq_k], so each strip advances the mask base by
            // q_begin rows; MASK_NONE ignores the address.
            const int64_t qd = (nq / NUM_CORES) * hd;
            const int64_t mask_row_elems = Align(seq_len, 16);
            const bool single_raw_query = small_image_single_raw_query_enabled();
            const int64_t query_chunk = single_raw_query
                ? seq_len : ((seq_len > 256)
                    ? QWEN25VL_VISION_SDPA_QUERY_CHUNK : seq_len);
            int64_t route_invocation = 0;
            bool raw_v_ready = false;
            const int64_t mask_domain = per_window_sdpa_enabled_
                ? int64_t{1} : generic_shared_attention_mask_domain();
            for (int64_t q_begin = 0; q_begin < seq_len; q_begin += query_chunk) {
                const int64_t query_len = std::min(query_chunk, seq_len - q_begin);
                const uint32_t q_off = addr_offset("q").value
                    + static_cast<uint32_t>(q_begin * qd * DWIDTH);
                const uint32_t out_off = addr_offset("sdpa_out").value
                    + static_cast<uint32_t>(q_begin * qd * DWIDTH);
                const uint32_t query_mask_off = is_full ? 0
                    : mask_off + static_cast<uint32_t>(
                        q_begin * mask_row_elems * DWIDTH);
                const GenericAttentionGeometry site{
                    QWEN25_STANDARD_DDR_ATTN_SITE,
                    chunk.idx * max_keepalive_seq_ +
                        route_invocation++,
                    /*batch=*/1, query_len, seq_len, mask_domain};
                const AttentionExecutionPolicy policy =
                    generic_attention_policy(site, mask_type);
                TORCH_CHECK(!single_raw_query ||
                                policy == AttentionExecutionPolicy::SPM_KV_BY_MHA,
                            "single-query M324 Vision requires its planned raw-SPM route");
                if (policy ==
                    AttentionExecutionPolicy::SPM_KV_BY_MHA) {
                    consume_generic_attention_route(
                        site, mask_type,
                        AttentionExecutionPolicy::SPM_KV_BY_MHA,
                        QWEN25_STANDARD_RAW_ATTN_SITE);
                    if (!raw_v_ready) {
                        rpu_launch_v_transpose_spm(
                            addr(0, "v"), addr(0, "sdpa_tmp"),
                            /*batch=*/1, seq_len, nq, hd, NUM_CORES);
                        raw_v_ready = true;
                    }
                    const uint32_t query_byte_offset =
                        static_cast<uint32_t>(q_begin * qd * DWIDTH);
                    const uint32_t raw_mask = mask_type == 4
                        ? addr(0, "sdpa_mask") + static_cast<uint32_t>(
                              q_begin * mask_row_elems * DWIDTH)
                        : 0;
                    rpu_launch_sdpa_by_mha_spm(
                        addr(0, "q") + query_byte_offset, addr(0, "k"),
                        addr(0, "sdpa_tmp"),
                        addr(0, "sdpa_out") + query_byte_offset,
                        raw_mask, mask_type, attn_scale,
                        /*batch=*/1, query_len, seq_len,
                        nq, nq, hd, NUM_CORES);
                } else {
                    consume_generic_attention_route(
                        site, mask_type, AttentionExecutionPolicy::DDR_KV,
                        QWEN25_STANDARD_DDR_ATTN_SITE);
                    raw_v_ready = false;
                    rpu_launch_sdpa_spm_unified_kernel_v2(
                        k_cache, v_cache, mask_type, attn_scale,
                        q_off, out_off, addr_offset("sdpa_tmp").value,
                        query_mask_off, query_len, nq, nq, hd,
                        seq_len, NUM_CORES, NUM_CORES);
                }
            }
        }

        // Phase 4: O_proj (row-partition, with bias) + AllReduce + Residual.
        // input_norm := reduce(oproj) + residual1 = post-attn hidden. The fused
        // route retains only each core's owner84 rows for the later Down ring;
        // oproj receives the full normalized result before Gate/Up reads it.
        const bool small_row_weight_outer =
            small_image_acc16_row_projection_enabled(seq_len);
        const bool owner_ring_norm = small_image_owner_ring_norm_enabled(seq_len);
        const int64_t row_route = small_row_weight_outer
            ? static_cast<int64_t>(Qwen25VisionLinearRoute::WALL_ROW_FULL_LOCAL_K)
            : qwen25_vision_linear_route();
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN25_O_LINEAR_SITE,
            row_route, chunk.idx);
        if (small_row_weight_outer) {
            rpu_launch_wall_vision_row_weight_outer_w8a16_spm_kernel(
                addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
                seq_len, h, nq * hd, layer_addr(layer_idx, 0, "o_bias"),
                lw.o_ws, wall_sorting_);
        } else {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
                seq_len, h, nq * hd, 0, NUM_CORES,
                layer_addr(layer_idx, 0, "o_bias"), lw.o_ws,
                /*nvfp4_scale=*/0, /*nvfp4_layer=*/0, /*force_acc32=*/linear_acc32_);
        }
        if (owner_ring_norm) {
            consume_small_image_owner_ring_norm(seq_len, chunk.idx);
            rpu_launch_wall_base_vision_owner_ring_norm_m672_kernel(
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "norm2_w"), eps_);
        } else if (residual_is_bf16) {
            consume_manifest_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN25_ATTN_MIXED_REDUCE_SITE,
                static_cast<int64_t>(
                    Qwen25VisionAllReduceRoute::
                        FP16_PARTIAL_BF16_RESIDUAL),
                chunk.idx);
            rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(
                addr(0, "oproj"), addr(0, "residual1"),
                addr(0, "input_norm"), seq_len, h);
        } else {
            consume_manifest_route(
                FmbRouteFamily::ALL_REDUCE, QWEN25_ATTN_REDUCE_SITE,
                qwen25_vision_ring_route(seq_len, h), chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual1"),
                addr(0, "input_norm"), seq_len, h, NUM_CORES, NUM_CORES);
        }
        emit_debug_phase(1, "input_norm");

        // Phase 5: RMSNorm2 (input_norm → oproj) + biased SwiGLU.
        if (residual_is_bf16) {
            rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                addr(0, "input_norm"), addr(0, "oproj"),
                layer_addr(layer_idx, 0, "norm2_w"), seq_len, h, eps_,
                consume_mixed_rmsnorm_route(
                    QWEN25_POST_MIXED_RMSNORM_SITE,
                    seq_len, h, chunk.idx));
        } else if (!owner_ring_norm) {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "input_norm"), addr(0, "oproj"),
                layer_addr(layer_idx, 0, "norm2_w"), seq_len, h, eps_,
                consume_small_image_rmsnorm_route(
                    QWEN25_SMALL_POST_RMSNORM_SITE, seq_len, chunk.idx));
        }
        emit_debug_phase(2, "oproj");

        if (small_image_acc16_gate_up_projection_enabled(seq_len)) {
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_GATE_LINEAR_SITE,
                static_cast<int64_t>(
                    Qwen25VisionLinearRoute::WALL_SINGLE_BASE_GATE_UP_SWIGLU),
                chunk.idx, small_image_gate_up_arguments(seq_len));
            rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel(
                addr(0, "oproj"), lw.gate_w, lw.up_w, addr(0, "gate"),
                layer_addr(layer_idx, 0, "gate_bias"),
                layer_addr(layer_idx, 0, "up_bias"), lw.gate_ws, lw.up_ws,
                0, 0, 0, 0, wall_sorting_, seq_len);
        } else {
            const int64_t elems = seq_len * local_inter;
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_GATE_LINEAR_SITE,
                qwen25_vision_linear_route(), chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "oproj"), lw.gate_w, addr(0, "gate"),
                seq_len, is_, h, 1, NUM_CORES,
                layer_addr(layer_idx, 0, "gate_bias"), lw.gate_ws,
                /*nvfp4_scale=*/0, /*nvfp4_layer=*/0, /*force_acc32=*/linear_acc32_);
            const bool fuse_silu_mul = wall_sorting_.silu_mul &&
                (mixed_precision_ || small_image_row_weight_outer_enabled(seq_len));
            if (!fuse_silu_mul) {
                rpu_launch_eltwise_unary_spm_kernel(
                    addr(0, "gate"), addr(0, "gate"), elems, ValuOpType::SILU);
            }
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_UP_LINEAR_SITE,
                qwen25_vision_linear_route(), chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "oproj"), lw.up_w, addr(0, "up"),
                seq_len, is_, h, 1, NUM_CORES,
                layer_addr(layer_idx, 0, "up_bias"), lw.up_ws,
                /*nvfp4_scale=*/0, /*nvfp4_layer=*/0, /*force_acc32=*/linear_acc32_);
            if (fuse_silu_mul) {
                rpu_launch_silu_mul_spm_kernel(
                    addr(0, "gate"), addr(0, "up"), addr(0, "gate"), elems, NUM_CORES);
            } else {
                rpu_launch_eltwise_binary_spm_kernel(
                    addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
                    elems, ValuOpType::MUL, c10::Half(1.0));
            }
        }

        // Phase 6: down_proj (row-partition, with bias) + AllReduce + Residual.
        // residual1 := reduce(oproj) + input_norm = block output (SPM_RESIDENT).
        if (mixed_precision_
            && layer_idx == QWEN25VL_VISION_MIXED_BOUNDARY_LAYER) {
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_DOWN_BF16_LINEAR_SITE,
                static_cast<int64_t>(
                    Qwen25VisionLinearRoute::ACC32_OUT_BF16), chunk.idx);
            rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(
                addr(0, "gate"), lw.down_w, addr(0, "oproj"),
                seq_len, h, is_, /*row=*/0, NUM_CORES,
                layer_addr(layer_idx, 0, "down_bias"));
            consume_manifest_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN25_DOWN_BF16_REDUCE_SITE,
                static_cast<int64_t>(
                    Qwen25VisionAllReduceRoute::
                        BF16_PARTIAL_FP16_RESIDUAL),
                chunk.idx);
            rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(
                addr(0, "oproj"), addr(0, "input_norm"),
                addr(0, "residual1"), seq_len, h);
        } else {
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN25_DOWN_LINEAR_SITE,
                row_route, chunk.idx,
                {mixed_precision_ ? 1 : 0});
            if (small_row_weight_outer) {
                rpu_launch_wall_vision_row_weight_outer_w8a16_spm_kernel(
                    addr(0, "gate"), lw.down_w, addr(0, "oproj"),
                    seq_len, h, is_, layer_addr(layer_idx, 0, "down_bias"),
                    lw.down_ws, wall_sorting_);
            } else {
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    addr(0, "gate"), lw.down_w, addr(0, "oproj"),
                    seq_len, h, is_, 0, NUM_CORES,
                    layer_addr(layer_idx, 0, "down_bias"), lw.down_ws,
                /*nvfp4_scale=*/0, /*nvfp4_layer=*/0, /*force_acc32=*/linear_acc32_);
            }
            if (residual_is_bf16) {
                consume_manifest_route(
                    FmbRouteFamily::ALL_REDUCE,
                    QWEN25_DOWN_MIXED_REDUCE_SITE,
                    static_cast<int64_t>(
                        Qwen25VisionAllReduceRoute::
                            FP16_PARTIAL_BF16_RESIDUAL),
                    chunk.idx);
                rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(
                    addr(0, "oproj"), addr(0, "input_norm"),
                    addr(0, "residual1"), seq_len, h);
            } else if (down_owner_norm && layer_idx + 1 < num_layers()) {
                consume_small_image_down_owner_norm(seq_len, layer_idx);
                rpu_launch_wall_base_vision_down_owner_ring_norm_m672_kernel(
                    addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"),
                    layer_addr(layer_idx + 1, 0, "norm1_w"), eps_);
            } else {
                consume_manifest_route(
                    FmbRouteFamily::ALL_REDUCE, QWEN25_DOWN_REDUCE_SITE,
                    qwen25_vision_ring_route(seq_len, h), chunk.idx);
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
    // GraphCache capture (D-501 post_fn). Computes the Qwen2.5-VL vision merger:
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
        uint32_t output_addr = 0, bool input_is_normalized = false,
        int64_t invocation = 0) {
        const int64_t h        = hidden_size();           // 1280
        const int64_t mh       = merge_hidden_;           // 5120
        const int64_t oh       = merger_out_hidden_;      // 2048
        const int64_t mrows    = seq / 4;                 // merged rows (SMS*SMS=4)
        const int64_t local_mh = mh / NUM_CORES;          // 640

        // 1. RMSNorm(ln_q) over h: input_addr[seq,h] (full per core) ->
        //    merger_normed.
        const uint32_t normalized_addr = input_is_normalized
            ? input_addr : addr(0, "merger_normed");
        if (!input_is_normalized) {
            if (mixed_precision_) {
                rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                    input_addr, normalized_addr,
                    addr(0, "merger_ln_q_w"), seq, h, eps_,
                    consume_mixed_rmsnorm_route(
                        QWEN25_MERGER_MIXED_RMSNORM_SITE,
                        seq, h, invocation));
            } else {
                rpu_launch_rmsnorm_spm_kernel(
                    input_addr, normalized_addr,
                    addr(0, "merger_ln_q_w"), seq, h, eps_,
                    consume_small_image_rmsnorm_route(
                        QWEN25_SMALL_MERGER_RMSNORM_SITE, seq, invocation));
            }
        }

        // 2. m0 GEMM (col): merger_normed reinterpreted [mrows, mh] -> merger_mid
        //    col [mrows, mh] (each core holds [mrows, local_mh]) + col bias.
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN25_MERGER_M0_LINEAR_SITE,
            qwen25_vision_linear_route(), invocation);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            normalized_addr, merger_m0_w_, addr(0, "merger_mid"),
            mrows, mh, mh, /*col*/1, NUM_CORES, addr(0, "merger_m0_bias"), at::Tensor{},
            /*nvfp4_scale=*/0, /*nvfp4_layer=*/0, /*force_acc32=*/linear_acc32_);

        // 3. GELU, per-core element count (col slice).
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "merger_mid"), addr(0, "merger_mid"),
            mrows * local_mh, ValuOpType::ADD,
            GeluMode::ERF, NUM_CORES);

        // 4. m2 GEMM (row): merger_mid [mrows, mh] (col slice = K-slice) ->
        //    merger_normed REUSED as PARTIAL [mrows, oh] per core + bias (core 0).
        //    merger_normed's RMSNorm content is dead after step 2 (m0 read it),
        //    so overwriting it here is safe.
        const uint32_t partial_addr = input_is_normalized
            ? input_addr : addr(0, "merger_normed");
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN25_MERGER_M2_LINEAR_SITE,
            qwen25_vision_linear_route(), invocation);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "merger_mid"), merger_m2_w_, partial_addr,
            mrows, oh, mh, /*row*/0, NUM_CORES, addr(0, "merger_m2_bias"), at::Tensor{},
            /*nvfp4_scale=*/0, /*nvfp4_layer=*/0, /*force_acc32=*/linear_acc32_);

        // 5. all-reduce(partial=merger_normed) + zero residual(merger_zero) ->
        //    merger_out [mrows, oh]. Three DISTINCT live slots (oproj/input_norm/
        //    residual1 offsets): partial=merger_normed, zero=merger_zero,
        //    out=merger_out. (no non-residual reduce variant; zero first.)
        rpu_launch_memset_spm_multicore(addr(0, "merger_zero"), mrows * oh);
        const uint32_t merger_out_addr = z1_bound_
            ? z1_source_addr_
            : (output_addr != 0 ? output_addr : addr(0, "merger_out"));
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE, QWEN25_MERGER_REDUCE_SITE,
            qwen25_vision_ring_route(mrows, oh), invocation);
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_addr, addr(0, "merger_zero"), merger_out_addr,
            mrows, oh, NUM_CORES, NUM_CORES);

        if (!z1_bound_) {
            // 6. SPM(core 0) -> the fresh post_output_tensor_. Its DDR address
            //    drifts across REPLAYs, so only the base is mutable; each image
            //    has a fixed byte offset within that live allocation.
            consume_manifest_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN25_MERGER_OUTPUT_DMA_SITE,
                static_cast<int64_t>(
                    Qwen25VisionMutableDmaRoute::SPM_COPY_TO_DDR),
                invocation);
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
            // The framework allocates this tensor fresh on every forward (P4).
            // No CPU writer touches it before the complete device overwrite.
            merger_out_dst_base_ =
                ::rhino_lkn::RpuGetDevAddr(post_output_tensor().data_ptr());
        }

        if (is_generic()) {
            emit_merger_chunk(
                total_seq, /*dst_offset_elems=*/0, addr(0, "residual1"));
            return;
        }

        if (exact_mimc_w8_qkv_enabled()) {
            const uint32_t packed_addr = addr(0, "input_norm");
            const int64_t seq = total_seq / image_batch_count_;
            if (!exact_mimc_compact_local_inout_enabled()) {
                for (int64_t i = 0; i < image_batch_count_; ++i) {
                    const uint32_t image_addr = packed_addr
                        + static_cast<uint32_t>(i * seq * h * DWIDTH);
                    rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
                        image_addr, image_addr, addr(0, "merger_ln_q_w"),
                        seq, h, eps_,
                        consume_mixed_rmsnorm_route(
                            QWEN25_EXACT_MERGER_MIXED_RMSNORM_SITE,
                            seq, h, i));
                }
            }
            const int64_t out_bytes = (total_seq / 4) * oh * DWIDTH;
            emit_merger_chunk(
                total_seq, /*dst_offset_elems=*/0, packed_addr,
                packed_addr + static_cast<uint32_t>(out_bytes),
                /*input_is_normalized=*/true);
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
                // The legacy layer-group schedule ends in DDR ping-pong.
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
                merger_output_addr, /*input_is_normalized=*/false, i);
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

    bool exact_mimc_fp16_enabled() const {
        return (wall_sorting_.fp16_transfer_mask & 1u) && is_exact_mimc() &&
            mixed_precision_ && intermediate_size() == 3584 &&
            !layer_weights_.empty() &&
            std::all_of(layer_weights_.begin(), layer_weights_.end(),
                [](const LayerWeights& lw) {
                    return lw.q_w.scalar_type() == at::kHalf &&
                        lw.k_w.scalar_type() == at::kHalf &&
                        lw.v_w.scalar_type() == at::kHalf &&
                        lw.o_w.scalar_type() == at::kHalf &&
                        lw.gate_w.scalar_type() == at::kHalf &&
                        lw.up_w.scalar_type() == at::kHalf &&
                        lw.down_w.scalar_type() == at::kHalf &&
                        !lw.q_ws.defined();
                });
    }

    bool exact_mimc_w8_qkv_enabled() const {
        if (exact_mimc_fp16_enabled()) return true;
        return is_exact_mimc() && mixed_precision_
            && !layer_weights_.empty()
            && std::all_of(
                layer_weights_.begin(), layer_weights_.end(),
                [](const LayerWeights& lw) {
                    return lw.q_w.scalar_type() == at::kChar
                        && lw.k_w.scalar_type() == at::kChar
                        && lw.v_w.scalar_type() == at::kChar
                        && lw.q_ws.defined() && lw.k_ws.defined()
                        && lw.v_ws.defined();
                });
    }

    bool exact_mimc_fused_w8_gate_up_enabled() const {
        if (exact_mimc_fp16_enabled()) return true;
        return exact_mimc_w8_qkv_enabled()
            && std::all_of(
                layer_weights_.begin(), layer_weights_.end(),
                [](const LayerWeights& lw) {
                    return lw.gate_w.scalar_type() == at::kChar
                        && lw.up_w.scalar_type() == at::kChar
                        && lw.gate_ws.defined() && lw.up_ws.defined();
                });
    }

    bool exact_mimc_native_window_sdpa_enabled() const {
        return exact_mimc_fused_w8_gate_up_enabled() &&
            window_mask_present_ &&
            fullatt_block_indexes_ ==
                std::vector<int64_t>({7, 15, 23, 31}) &&
            cu_window_ == std::vector<int64_t>({
                0, 64, 128, 192, 256, 320, 384,
                448, 512, 528, 544, 560, 576});
    }

    bool exact_mimc_compact_local_inout_enabled() const {
        return exact_mimc_native_window_sdpa_enabled() && merger_active();
    }

    // Fused merger is active when weights are registered and the cold handle
    // snapshot enables it. Normal packed vision is one pass; layer grouping uses
    // the chunked path above.
    bool merger_active() const {
        return fused_merger_enabled_ && merger_out_hidden_ > 0;
    }

    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        if (linear_acc32_) identity.insert(identity.end(), {QWEN25_WALL_SORTING_SITE, 32});
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.insert(identity.end(), {
            static_cast<int64_t>(has_rope_),
            static_cast<int64_t>(rope_spm_enabled_),
            static_cast<int64_t>(fused_merger_enabled_),
            static_cast<int64_t>(mixed_precision_enabled_),
            static_cast<int64_t>(layer_group_enabled_),
            static_cast<int64_t>(exact_mimc_enabled_),
            static_cast<int64_t>(per_window_sdpa_enabled_),
            static_cast<int64_t>(rmsnorm_capability_),
            static_cast<int64_t>(wall_sorting_.fp16_transfer_mask),
            static_cast<int64_t>(wall_sorting_.vision_row_reuse),
            static_cast<int64_t>(wall_sorting_.silu_mul)});
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.gate_w, &weights.up_w, &weights.down_w, &weights.q_ws,
                    &weights.k_ws, &weights.v_ws, &weights.o_ws, &weights.gate_ws,
                    &weights.up_ws, &weights.down_ws}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        identity.push_back(static_cast<int64_t>(layer_bias_norm_.size()));
        for (const auto& weights : layer_bias_norm_) {
            for (const auto* tensor : {
                    &weights.norm1_w, &weights.norm2_w, &weights.q_b, &weights.k_b,
                    &weights.v_b, &weights.o_b, &weights.gate_b, &weights.up_b,
                    &weights.down_b}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &merger_ln_q_w_, &merger_m0_w_, &merger_m0_b_, &merger_m2_w_,
                &merger_m2_b_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        if (small_owner_ring_norm_available_) {
            identity.insert(identity.end(), {QWEN25_SMALL_OWNER_RING_NORM_SITE,
                1, static_cast<int64_t>(small_rmsnorm_capability_)});
        }
        if (small_down_owner_norm_available_) {
            identity.insert(identity.end(), {QWEN25_SMALL_DOWN_OWNER_NORM_SITE,
                1, static_cast<int64_t>(small_rmsnorm_capability_)});
        }
        if (small_qk_rope_available_)
            identity.insert(identity.end(), {QWEN25_SMALL_QK_ROPE_SPM_SITE, 1});
        if (small_ragged_sdpa_available_)
            identity.insert(identity.end(), {QWEN25_SMALL_RAGGED_SDPA_SITE, 1});
        return identity;
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
    uint64_t merger_out_dst_base_ = 0;      // live device addr of post_output_tensor_ (mutable DMA)

    // 2D RoPE state
    at::Tensor freq_cos_;
    at::Tensor freq_sin_;
    int64_t max_keepalive_seq_ = 0;
    at::Tensor position_idx_keepalive_;
    int64_t max_hw_ = 0;
    bool has_rope_ = false;
    int64_t current_num_patches_ = 0;
    int64_t image_batch_count_ = 1;   // >1 => batched multi-image (per-image SDPA isolation)
    int64_t configured_chunk_size_ = 0; // 0=auto; otherwise exact physical capacity
    // Cold adapter translation is the sole authority. Forward, candidate
    // resolution, and launchers consume only these handle-owned bits.
    bool rope_spm_enabled_ = false;
    bool fused_merger_enabled_ = false;
    bool mixed_precision_enabled_ = false;
    bool linear_acc32_ = false;
    bool layer_group_enabled_ = false;
    bool exact_mimc_enabled_ = false;
    WallSortingConfig wall_sorting_;
    uint32_t rmsnorm_capability_ = RPU_RMSNORM_CAP_BASE;
    uint32_t small_rmsnorm_capability_ = RPU_RMSNORM_CAP_BASE;
    bool small_owner_ring_norm_available_ = false;
    bool small_down_owner_norm_available_ = false;
    bool small_qk_rope_available_ = false;
    bool small_ragged_sdpa_available_ = false;
    bool small_ragged_mask_exact_ = false;
    int64_t small_ragged_mask_rows_ = 0;
    std::array<at::Tensor, 2> small_ragged_mask_refs_;
    std::array<uint32_t, 2> small_ragged_mask_versions_{0, 0};
    std::array<bool, 2> small_ragged_mask_valid_{false, false};
    at::Tensor small_ragged_prepared_mask_ref_;
    uint32_t small_ragged_prepared_mask_version_ = 0;
    VisionSchedule schedule_ = VisionSchedule::SISC;
    VisionImplementation implementation_ = VisionImplementation::Generic;
    bool mixed_precision_ = false;     // block-17 BF16 residual transition (RTC only)
    at::Tensor per_layer_debug_buf_;   // [layers, 1, seq, hidden], debug-only
    at::Tensor phase_debug_buf_;       // [3, 1, seq, hidden], layer-17 debug-only
    // Model config
    double eps_ = 1e-6;
    int64_t orig_head_dim_ = 0;

    // P3b window attention: per-layer dense block-diagonal MASK_2D. Layers in
    // fullatt_block_indexes_ stay MASK_NONE; the rest use prepared_window_mask_.
    std::vector<int64_t> fullatt_block_indexes_;
    bool window_mask_present_ = false;
    PreparedMask prepared_window_mask_;
    int64_t valid_patches_per_image_ = 0;  // explicit pair owner; 0 retains ordinary semantics
    std::array<PreparedMask, 2> prepared_pair_masks_;
    std::array<at::Tensor, 2> pair_mask_refs_;
    std::array<uint32_t, 2> pair_mask_versions_{};
    bool per_window_sdpa_enabled_ = false;
    std::vector<int64_t> cu_window_;
    // Cache the prepared window mask across forwards: the window_mask is
    // frame-invariant for a fixed camera (adapter memoizes it in _group_layout),
    // so re-running sdpa_prepare_mask (a [ctx,ctx] CPU→device upload + stable-slot
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

std::vector<int64_t> rpu_qwen25vl_vision_planner_cache_identity(int64_t handle) {
    return Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_qwen25vl_vision_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen25vl_vision_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_qwen25vl_vision_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision_kvinsert_cost_domain")
        ->kvinsert_cost_domain("qwen25vl_vision", descriptor);
}

std::string rpu_qwen25vl_vision_kvinsert_cost_catalog_sha256(int64_t handle) {
    return Qwen25VLVisionRegistry::get(
        handle, "rpu_qwen25vl_vision_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

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

void rpu_qwen25vl_vision_set_linear_acc32(int64_t handle, bool enabled) {
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision_set_linear_acc32")
        ->set_linear_acc32(enabled);
}

void rpu_qwen25vl_vision_set_wall_sorting_routes(
    int64_t handle, int64_t mask, bool row, bool silu) {
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->set_wall_sorting_routes(mask, row, silu);
}

void rpu_qwen25vl_vision_set_execution_routes(
    int64_t handle, bool rope_spm, bool fused_merger, bool mixed_precision,
    bool layer_group, bool exact_mimc, int64_t rmsnorm_capability) {
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->set_execution_routes(
            rope_spm, fused_merger, mixed_precision, layer_group, exact_mimc,
            rmsnorm_capability);
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

void rpu_qwen25vl_vision_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision_set_chunk_envelope")
        ->set_execution_envelope(max_kv_len, chunk);
}

void rpu_qwen25vl_vision_set_chunk_size(
    int64_t handle, int64_t chunk_size)
{
    Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->set_configured_chunk_size(chunk_size);
}

std::vector<int64_t> rpu_qwen25vl_vision_resolve_stage_domain(
    int64_t handle, int64_t num_patches, int64_t image_batch_count,
    bool window_mask_present,
    const std::optional<at::Tensor>& cu_window_seqlens,
    int64_t requested_chunk_size, int64_t valid_patches_per_image,
    const std::optional<at::Tensor>& planning_window_mask) {
    return Qwen25VLVisionRegistry::get(
               handle, "rpu_qwen25vl_vision_resolve_stage_domain")
        ->resolve_stage_domain(
            num_patches, image_batch_count, window_mask_present,
            cu_window_seqlens, requested_chunk_size, valid_patches_per_image, planning_window_mask);
}

int64_t rpu_qwen25vl_vision_get_resolved_chunk_size(int64_t handle) {
    return Qwen25VLVisionRegistry::get(
               handle, "rpu_qwen25vl_vision_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

void rpu_qwen25vl_vision_enable_execution_reconfigure(int64_t handle) {
    Qwen25VLVisionRegistry::get(
        handle, "rpu_qwen25vl_vision_enable_execution_reconfigure")
        ->enable_execution_reconfigure_guard();
}

void rpu_qwen25vl_vision_stage_chunk_size(
    int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0,
                "Qwen2.5-VL Vision execution transaction token must be positive");
    Qwen25VLVisionRegistry::get(
        handle, "rpu_qwen25vl_vision_stage_chunk_size")
        ->stage_configured_chunk_size(
            static_cast<uint64_t>(token), chunk_size);
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
    std::optional<at::Tensor> cu_window_seqlens,
    at::IntArrayRef planned_stage_descriptor,
    int64_t valid_patches_per_image,
    std::optional<at::Tensor> full_mask,
    bool position_published)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Qwen25VLVisionRegistry::get(handle, "rpu_qwen25vl_vision")
        ->forward(input, k_caches, v_caches, num_patches, window_mask,
                  image_batch_count, cu_window_seqlens,
                  planned_stage_descriptor, valid_patches_per_image, full_mask,
                  position_published);
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
