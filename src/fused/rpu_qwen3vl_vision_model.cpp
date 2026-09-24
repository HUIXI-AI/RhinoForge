// rpu_qwen3vl_vision_model.cpp — Qwen3-VL Vision Encoder (Phase 3 GAP-B*)
//
// 24-block ViT encoder for Qwen3-VL (2B/4B variants, depth=24, hidden=1024,
// intermediate=4096, num_heads=16, head_dim=64). Architecture is identical to
// SigLIP-So400m EXCEPT:
//   - 2D RoPE applied to Q/K between fused-QKV+bias and bidir SDPA (GAP-B3).
//   - No post-LN + projector (merger moved to Python after C++ forward
//     because the merger requires `spatial_merge_size² × hidden` reshape, see
//     findings F7 #6 — single-layer SPM op chain instead of fused into graph).
//   - DeepStack snapshots at layers {5, 11, 17} → DMA to Python-side mergers
//     (GAP-B7). Reuses spm_copy_ddr_dma the same way SigLIP's post-LN-to-DDR
//     transfer works (see siglip's `temp_ddr_slots_` pattern).
//
// Do not inherit/reuse SigLIPModel C++ class: this vision tower
// architectural differences are large enough (Conv3d, 2D RoPE, fused QKV,
// merger reshape) that bolting onto SigLIP becomes a leaky abstraction.
// Karpathy "no premature abstraction": copy-modify is acceptable here; we
// extract _vit_common.h only if real maintenance duplication emerges.
//
// Patch embedding + pos_emb interpolate are performed OUTSIDE the fused graph:
// Python adapter runs Conv3d→Conv2d folded GEMM via existing rpu_conv2d /
// rpu_linear, then adds `fast_pos_embed_interpolate` output via rpu_add. The
// fused subsystem expects pre-embedded input `[num_patches, hidden]`.
//
// 2D RoPE uses `rope_2d_ddr` kernel (registered Phase 3 GAP-B3); FreqCos /
// FreqSin are static tables `[max_hw, head_dim/4]` set once via set_rope.
// position_idx is a `[max_seq, 2] int16` keepalive (Option β absolute index
// contract — see findings F26 — extended from text-domain mrope to vision).

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_qwen3vl_vision_sdpa.h"
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
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

using namespace at;
using namespace ::rhino_lkn;

#define DWIDTH 2

// Position-table keepalive cap. 4096 patches covers up to 64×64 grid (typical
// single-image is 28×28 = 784; pre-merger grid for 4096 is overkill but cheap
// at 4096 × 2 × 2 = 16 KB DDR. Aligned to allow 256B-boundary writes).
constexpr int64_t RHINOVLA_VISION_HIGH_ACTIVATION_SITE = 108086773956923057LL;
constexpr int64_t RHINOVLA_VISION_HIGH_ENCODER_ACTIVATION_SITE = 4642981117179055386LL;
constexpr int64_t QWEN3VL_VISION_MAX_KEEPALIVE_SEQ = 4096;

constexpr int64_t QWEN3VL_VISION_DEFAULT_CHUNK_SIZE = 144;
constexpr int64_t QWEN3VL_VISION_LARGE_IMAGE_THRESHOLD = 1024;
constexpr int64_t QWEN3VL_VISION_LARGE_IMAGE_CHUNK_CAP = 608;

// Stable native route identities from planner_owners.v1.json.
// QWEN3VL_FIXED_KERNEL_BASIS: norms, activations, unconditional bias/debug/
// staging DMAs, and BufferDecl-driven data movement implement fixed model
// math/dataflow. Conditional RoPE-table preloads and the other choices below
// are descriptor routes.
constexpr int64_t QWEN3VL_Q_LINEAR_SITE = 1427990588395672103LL;
constexpr int64_t QWEN3VL_K_LINEAR_SITE = 6011173041503464904LL;
constexpr int64_t QWEN3VL_V_LINEAR_SITE = 3337936004446054324LL;
constexpr int64_t QWEN3VL_Q_ROPE_SPM_PIPELINE_SITE = 6478246981822727230LL;
constexpr int64_t QWEN3VL_K_ROPE_SPM_PIPELINE_SITE = 7490107664646219332LL;
constexpr int64_t QWEN3VL_Q_ROPE_SPM_SITE = 6529655989819384719LL;
constexpr int64_t QWEN3VL_K_ROPE_SPM_SITE = 8802678050123635721LL;
constexpr int64_t QWEN3VL_Q_ROPE_DDR_SITE = 1110626026745786641LL;
constexpr int64_t QWEN3VL_K_ROPE_DDR_SITE = 7925958889783462480LL;
constexpr int64_t QWEN3VL_KV_INSERT_SITE = 2982723969499025021LL;
constexpr int64_t QWEN3VL_MINIBATCH_ATTN_SITE = 7299740951628094462LL;
constexpr int64_t QWEN3VL_UNIFIED_ATTN_SITE = 5297603665400356094LL;
constexpr int64_t QWEN3VL_N1200_TM160_ATTN_SITE = 139704655818769LL;
constexpr int64_t QWEN3VL_RAW_SPM_ATTN_SITE = 6757430431325841851LL;
constexpr int64_t QWEN3VL_O_LINEAR_SITE = 7749931284684938753LL;
constexpr int64_t QWEN3VL_ATTN_ALL_REDUCE_SITE = 6328608535873931861LL;
constexpr int64_t QWEN3VL_FC1_LINEAR_SITE = 6682031402431228349LL;
constexpr int64_t QWEN3VL_FC2_LINEAR_SITE = 886301000936414295LL;
constexpr int64_t QWEN3VL_MLP_ALL_REDUCE_SITE = 3565420092428644692LL;
constexpr int64_t QWEN3VL_MERGER_M0_LINEAR_SITE = 7854280711758341275LL;
constexpr int64_t QWEN3VL_MERGER_M2_LINEAR_SITE = 58355670836217970LL;
constexpr int64_t QWEN3VL_MERGER_ALL_REDUCE_SITE = 1170949137345989383LL;
constexpr int64_t QWEN3VL_MERGER_YIELD_ALL_REDUCE_SITE = 4481167224376802776LL;
constexpr int64_t QWEN3VL_ROPE_COS_PRELOAD_PIPELINE_SITE =
    7865795941291055524LL;
constexpr int64_t QWEN3VL_ROPE_SIN_PRELOAD_PIPELINE_SITE =
    7646375323102352740LL;
constexpr int64_t QWEN3VL_ROPE_COS_PRELOAD_DIRECT_SITE =
    1089927205617401262LL;
constexpr int64_t QWEN3VL_ROPE_SIN_PRELOAD_DIRECT_SITE =
    9144463814971908982LL;
constexpr int64_t QWEN3VL_DEEPSTACK_SNAPSHOT_SITE =
    559423039313734461LL;
constexpr int64_t QWEN3VL_POOLER_MERGED_COPY_SITE =
    2748564172848473782LL;
constexpr int64_t QWEN3VL_DEEPSTACK_MERGED_COPY_SITE =
    8851919130771715685LL;
constexpr int64_t QWEN3VL_CHUNKED_MERGER_INPUT_SITE =
    6804220642413844377LL;
constexpr int64_t QWEN3VL_CHUNKED_MERGER_OUTPUT_SITE =
    3098979908918923079LL;

constexpr uint32_t QWEN3VL_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
    KV_INSERT_CAP_HYBRID2 | KV_INSERT_CAP_HYBRID3;
constexpr int64_t QWEN3VL_KV_REASON_DDR_REQUIRED = 1;

inline constexpr int64_t QWEN3VL_W8_RESIDENT_SCHEDULE_SITE = 293709511256992144LL;
inline constexpr int64_t QWEN3VL_FP16_COMPACT_SCHEDULE_SITE = 6646982877720472090LL;
// Compact launch sites have their own canonical registry identities; ordinary
// attention/MLP rings retain the existing keys above.
inline constexpr int64_t QWEN3VL_FP16_COMPACT_COPY_SITE = 6332452221222858102LL;
inline constexpr int64_t QWEN3VL_COMPACT_ATTN_ALL_REDUCE_SITE = 8757963121743615819LL;
inline constexpr int64_t QWEN3VL_COMPACT_MLP_ALL_REDUCE_SITE = 860551662759870427LL;
inline constexpr int64_t QWEN3VL_W8_COMPACT_SCHEDULE_SITE = 872985762413057128LL;

inline constexpr int64_t QWEN3VL_DELIVERY_VISION_SCHEDULE_SITE = 1309103528283123878LL;

enum class Qwen3VLVisionRopeRoute : int64_t {
    ROPE_2D_SPM = 1,
    ROPE_2D_DDR = 2,
};

enum class Qwen3VLVisionRopeRouteRequest : int64_t {
    AUTO = 0,
    SPM = 1,
    DDR = 2,
};

enum class Qwen3VLVisionMutableDmaRoute : int64_t {
    ROPE_TABLE_DDR_TO_SPM = 1,
    SPM_COPY_TO_DDR = 2,
    MERGER_INPUT_DDR_TO_SPM = 3,
};

// First Qwen3-VL pooler-only physical handoff canary. Keep the envelope exact:
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
constexpr uint64_t QWEN3VL_MULTIVIEW_YIELD_POLICY =
    UINT64_C(0x51564d56594c4401);  // QVMVYLD1
constexpr uint64_t QWEN3VL_MULTIVIEW_OCCURRENCE_POLICY =
    UINT64_C(0x51564d5633494e01);  // QVMV3IN1
// The cold adapter supplies AUTO or translates the legacy env into an exact
// 2D-RoPE table-residency request. GR00T and ordinary LingBot2 pin SPM. The
// selected COMPLETE descriptor runs the 2D-RoPE
// from descriptor-selected SPM-resident cos/sin tables (persistent preload for
// native composites; a Graph-owned pre-layer upload for ordinary execution)
// instead of re-streaming the DDR tables per layer. This preserves the table
// values and mirrors the qwen25vl
// RPU_WALL_OSS_VISION_ROPE_SPM path.
// Registering merger weights is the single cold authority that enables the
// post-encoder patch merger (LayerNorm -> m0 GEMM -> GELU -> m2 GEMM
// -> all-reduce) runs INSIDE the qwen3vl_vision_forward graph via a post_fn (single
// image per forward). gr00t opts in; plain qwen3_vl stays byte-identical. Mirrors the
// qwen25vl RPU_WALL_OSS_VISION_FUSED_MERGER fold.

// W8A16 per-projection scale validation. Same four checks as the already-strong
// sites (rpu_siglip_model.cpp:227, rpu_pi05_denoise_step_model.cpp:68): list
// length, int8 weight dtype, 1D contiguous fp16 RPU scale, and
// scale.numel() == output channels. Checking only the list length is not enough —
// a wrong-length or wrong-dtype scale reaches the generated W8A16 family as garbage
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

static void check_qwen3vl_merger_vector(
    const at::Tensor& tensor, int64_t size, const char* name) {
    TORCH_CHECK(tensor.defined() && tensor.scalar_type() == at::kHalf &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.is_contiguous() && tensor.dim() == 1 && tensor.numel() == size,
                "Qwen3-VL merger ", name, " must be contiguous FP16 RPU [", size, "]");
}

static void check_qwen3vl_merger_projection(
    const at::Tensor& weight, const at::Tensor& scale,
    int64_t rows, int64_t cols, const char* name) {
    TORCH_CHECK(weight.defined() && weight.device().type() == at::kPrivateUse1 &&
                    weight.is_contiguous() && weight.dim() == 2 &&
                    weight.size(0) == rows && weight.size(1) == cols &&
                    (weight.scalar_type() == at::kHalf || weight.scalar_type() == at::kChar),
                "Qwen3-VL merger ", name, " must be contiguous FP16/int8 RPU [", rows, ",", cols, "]");
    if (weight.scalar_type() == at::kChar) {
        check_qwen3vl_merger_vector(scale, rows, name);
        rpu_ddr_flush_force(scale.data_ptr<c10::Half>());
        for (int64_t i = 0; i < rows; ++i) {
            const float value = static_cast<float>(scale.data_ptr<c10::Half>()[i]);
            TORCH_CHECK(std::isfinite(value) && value > 0,
                        "Qwen3-VL merger ", name, " scale must be finite and positive");
        }
    } else {
        TORCH_CHECK(!scale.defined(), "Qwen3-VL FP16 merger ", name, " rejects quantization scale");
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

    void configure_execution_cores(int cores) {
        TORCH_CHECK(cores == 4 || cores == 8,
                    "Qwen3-VL Vision admits four or eight execution cores");
        set_execution_core_count(cores);
    }

    std::vector<int64_t> execution_topology() const {
        TORCH_CHECK(num_layers() > 0,
                    "Vision topology requires installed model weights");
        return {num_cores(), attn_tp(), mlp_tp(), 8};
    }

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

        if (num_cores() != 8) {
            const bool regular = N == 24 && num_heads == 16 &&
                head_dim == 64 && hidden_size == 1024 && intermediate_size == 4096 &&
                deepstack_visual_indexes == std::vector<int64_t>{5, 11, 17};
            const bool padded = padded_32b &&
                deepstack_visual_indexes == std::vector<int64_t>{8, 16, 24};
            TORCH_CHECK(num_cores() == 4 && (regular || padded) && eps == 1e-6,
                        "four-core Vision requires the exact dense Qwen3-VL 2B/4B/8B tower");
            TORCH_CHECK(q_scale_list.empty() && k_scale_list.empty() &&
                            v_scale_list.empty() && o_scale_list.empty() &&
                            fc1_scale_list.empty() && fc2_scale_list.empty(),
                        "four-core Vision requires FP16 weights without quantization scales");
            for (const auto* list : {&q_w_list, &k_w_list, &v_w_list, &o_w_list,
                    &fc1_w_list, &fc2_w_list, &ln1_w_list, &ln1_b_list,
                    &ln2_w_list, &ln2_b_list, &q_b_list, &k_b_list, &v_b_list,
                    &o_b_list, &fc1_b_list, &fc2_b_list}) {
                for (const auto& tensor : *list) {
                    TORCH_CHECK(tensor.scalar_type() == at::kHalf &&
                                    tensor.device().type() == at::kPrivateUse1 &&
                                    tensor.is_contiguous(),
                                "four-core Vision requires contiguous FP16 RPU weights and biases");
                }
            }
        }

        // Vision encoder is MHA (no GQA): num_kv_heads == num_q_heads.
        set_model_params(num_heads, num_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;

        // Keep logical attention math separate from the physical RPU stride.
        // The controlled 32B tower pads each head 72->80 and its MLP
        // 4304->4352 at the Python conversion boundary; 2B/4B remain exact.
        orig_head_dim_ = logical_head_dim;

        // DeepStack snapshot indices (GAP-B7). Empty list → no snapshots emitted.
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
        TORCH_CHECK(!padded_32b || !any_scale || num_cores() == 8,
                    "qwen3vl_vision_set_weights: padded W8A16 Vision requires TP8");
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

        rhinovla_high_precision_ = false;
        rhinovla_precision_bound_ = false;
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
    void set_rope_tables(
        const at::Tensor& freq_cos, const at::Tensor& freq_sin,
        bool rope_spm_enabled)
    {
        set_rope_tables_with_route(
            freq_cos, freq_sin,
            static_cast<int64_t>(
                rope_spm_enabled
                    ? Qwen3VLVisionRopeRouteRequest::SPM
                    : Qwen3VLVisionRopeRouteRequest::DDR));
    }

    void set_rope_tables_with_route(
        const at::Tensor& freq_cos, const at::Tensor& freq_sin,
        int64_t route_request)
    {
        TORCH_CHECK(
            route_request >= static_cast<int64_t>(
                                 Qwen3VLVisionRopeRouteRequest::AUTO) &&
                route_request <= static_cast<int64_t>(
                                     Qwen3VLVisionRopeRouteRequest::DDR),
            "qwen3vl_vision_set_rope_route: route_request must be "
            "0=AUTO, 1=SPM, or 2=DDR, got ", route_request);
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
        rope_route_request_ = static_cast<Qwen3VLVisionRopeRouteRequest>(
            route_request);
        // AUTO has no route authority until a descriptor is selected.  DDR is
        // only the inert pre-plan state; forward overwrites it from the exact
        // selected manifest before any layout or launcher decision.
        rope_spm_enabled_ =
            rope_route_request_ == Qwen3VLVisionRopeRouteRequest::SPM;

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
    // the in-graph fold (ROUND-3 opt #1). LayerNorm gamma/beta [HID] + m0
    // [merge_hidden, merge_hidden] (col-swizzled) + m2 [OUT_HID, merge_hidden]
    // (row-swizzled), all fp16 DDR-resident (swizzled by the Python adapter).
    // Mirrors qwen25vl set_merger_weights + ln_q_b for LayerNorm.
    // ========================================================================
    void set_merger_weights(
        const at::Tensor& ln_q_w, const at::Tensor& ln_q_b,
        const at::Tensor& m0_w, const at::Tensor& m0_b,
        const at::Tensor& m2_w, const at::Tensor& m2_b,
        int64_t out_hidden, double ln_eps,
        const at::Tensor& m0_scale, const at::Tensor& m2_scale)
    {
        TORCH_CHECK(num_cores() == 8,
                    "fused Vision merger retains its eight-core profile");
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
        TORCH_CHECK(m0_w.size(0) == hidden_size() * 4 && out_hidden % 128 == 0 &&
                    std::isfinite(ln_eps) && ln_eps > 0,
                    "Qwen3-VL merger requires spatial merge 2, TP8 output geometry, and positive epsilon");
        check_qwen3vl_merger_vector(ln_q_w, hidden_size(), "norm weight");
        check_qwen3vl_merger_vector(ln_q_b, hidden_size(), "norm bias");
        check_qwen3vl_merger_vector(m0_b, m0_w.size(0), "m0 bias");
        check_qwen3vl_merger_vector(m2_b, out_hidden, "m2 bias");
        check_qwen3vl_merger_projection(m0_w, m0_scale, m0_w.size(0), m0_w.size(0), "m0");
        check_qwen3vl_merger_projection(m2_w, m2_scale, out_hidden, m0_w.size(0), "m2");
        TORCH_CHECK(m0_w.scalar_type() == m2_w.scalar_type(),
                    "Qwen3-VL merger projections must use the same weight precision");
        merger_ln_q_w_ = ln_q_w.contiguous();
        merger_ln_q_b_ = ln_q_b.contiguous();
        merger_m0_w_   = m0_w.contiguous();   merger_m0_b_ = m0_b.contiguous();
        merger_m2_w_   = m2_w.contiguous();   merger_m2_b_ = m2_b.contiguous();
        merger_m0_scale_ = m0_scale; merger_m2_scale_ = m2_scale;
        merge_hidden_      = m0_w.size(0);
        merger_out_hidden_ = out_hidden;
        merger_ln_eps_     = ln_eps;
        fused_merger_enabled_ = true;
        invalidate_model_state();
    }

    // inc-2: register the N deepstack mergers' weights (one per deepstack_visual_index). Same
    // shapes as the patch merger; m0 col-swizzled, m2 row-swizzled, ln_q raw [HID]. Folded in
    // merger_post_fn (each reads its layer's snapshot, writes a stable merged slot).
    void set_deepstack_merger_weights(
        at::TensorList ln_q_w, at::TensorList ln_q_b,
        at::TensorList m0_w, at::TensorList m0_b,
        at::TensorList m2_w, at::TensorList m2_b,
        at::TensorList m0_scale, at::TensorList m2_scale)
    {
        TORCH_CHECK(merger_out_hidden_ > 0,
                    "set_deepstack_merger_weights: call set_merger_weights first");
        const int64_t n = static_cast<int64_t>(m0_w.size());
        TORCH_CHECK(n == static_cast<int64_t>(deepstack_visual_indexes_.size()),
                    "set_deepstack_merger_weights: count ", n, " != n_deepstack ",
                    deepstack_visual_indexes_.size());
        TORCH_CHECK(ln_q_w.size() == n && ln_q_b.size() == n && m0_b.size() == n &&
                    m2_w.size() == n && m2_b.size() == n &&
                    (m0_scale.empty() || m0_scale.size() == n) &&
                    (m2_scale.empty() || m2_scale.size() == n),
                    "Qwen3-VL DeepStack merger weight/scale list lengths disagree");
        // Validate every list before replacing any retained owner.
        for (int64_t k = 0; k < n; ++k) {
            check_qwen3vl_merger_vector(ln_q_w[k], merge_hidden_, "DeepStack norm weight");
            check_qwen3vl_merger_vector(ln_q_b[k], merge_hidden_, "DeepStack norm bias");
            check_qwen3vl_merger_vector(m0_b[k], merge_hidden_, "DeepStack m0 bias");
            check_qwen3vl_merger_vector(m2_b[k], merger_out_hidden_, "DeepStack m2 bias");
            check_qwen3vl_merger_projection(m0_w[k], m0_scale.empty() ? at::Tensor{} : m0_scale[k],
                                           merge_hidden_, merge_hidden_, "DeepStack m0");
            check_qwen3vl_merger_projection(m2_w[k], m2_scale.empty() ? at::Tensor{} : m2_scale[k],
                                           merger_out_hidden_, merge_hidden_, "DeepStack m2");
            TORCH_CHECK(m0_w[k].scalar_type() == merger_m0_w_.scalar_type() &&
                        m2_w[k].scalar_type() == merger_m2_w_.scalar_type(),
                        "Qwen3-VL DeepStack and final mergers must use the same weight precision");
        }
        ds_merger_ln_q_w_.clear(); ds_merger_ln_q_b_.clear();
        ds_merger_m0_w_.clear();   ds_merger_m0_b_.clear();
        ds_merger_m2_w_.clear();   ds_merger_m2_b_.clear();
        ds_merger_m0_scale_.clear(); ds_merger_m2_scale_.clear();
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
            ds_merger_m0_scale_.push_back(m0_scale.empty() ? at::Tensor{} : m0_scale[k]);
            ds_merger_m2_scale_.push_back(m2_scale.empty() ? at::Tensor{} : m2_scale[k]);
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

    void stage_configured_chunk_size(uint64_t token, int64_t chunk_size) {
        TORCH_CHECK(chunk_size == 0
                        || (chunk_size >= 16 && chunk_size % 16 == 0),
                    "Qwen3-VL vision chunk size must be 0 (auto) or a "
                    "positive multiple of 16, got ", chunk_size);
        const int64_t old_chunk_size = configured_chunk_size_;
        stage_execution_controls(
            token, std::nullopt, std::nullopt,
            [this, chunk_size] { configured_chunk_size_ = chunk_size; },
            [this, old_chunk_size] {
                configured_chunk_size_ = old_chunk_size;
            },
            "Qwen3VLVisionModel::stage_configured_chunk_size");
    }

    void set_rhinovla_high_precision(bool enabled) {
        RpuExecutionCoordinator::require_graph_quiescent("rhinovla_vision_high_precision");
        RpuExecutionCoordinator::check_current_thread_execution_allowed("rhinovla_vision_high_precision");
        TORCH_CHECK(!RpuKernelGraph::has_active() &&
                    get_last_resolved_chunk_size() == 0 && !rhinovla_precision_bound_,
                    "RhinoVLA vision precision must be cold-bound once before forward");
        if (enabled) {
            TORCH_CHECK(num_cores() == 8 && num_layers() == 24 &&
                        hidden_size() == 1024 && intermediate_size() == 4096 &&
                        num_q_heads() == 16 && head_dim() == 64 &&
                        fused_merger_enabled_ && merger_out_hidden_ == 2048 &&
                        merge_hidden_ == 4096 && deepstack_visual_indexes_ ==
                            std::vector<int64_t>({5, 11, 17}) &&
                        ds_merger_m0_w_.size() == 3,
                        "RhinoVLA high precision requires the full v3 vision tower and mergers");
            for (const auto& layer : layer_weights_) {
                for (const auto* w : {&layer.q_w, &layer.k_w, &layer.v_w,
                                      &layer.o_w, &layer.fc1_w, &layer.fc2_w}) {
                    TORCH_CHECK(w->defined() && w->scalar_type() == at::kChar,
                                "RhinoVLA high precision requires full W8 vision weights");
                }
            }
            for (const auto* w : {&merger_m0_w_, &merger_m2_w_}) {
                TORCH_CHECK(w->defined() && w->scalar_type() == at::kChar,
                            "RhinoVLA high precision requires W8 mergers");
            }
            for (size_t i = 0; i < ds_merger_m0_w_.size(); ++i) {
                TORCH_CHECK(ds_merger_m0_w_[i].scalar_type() == at::kChar &&
                            ds_merger_m2_w_[i].scalar_type() == at::kChar,
                            "RhinoVLA high precision requires W8 deepstack mergers");
            }
            rpu_require_high_precision_math_kernels();
        }
        rhinovla_high_precision_ = enabled;
        rhinovla_precision_bound_ = true;
        invalidate_model_state();
    }

    void set_linear_acc32(bool enabled) {
        TORCH_CHECK(
            get_last_resolved_chunk_size() == 0,
            "Qwen3-VL vision linear accumulation mode must be set before the first forward");
        linear_acc32_ = enabled;
        invalidate_model_state();
    }

    void set_large_image_auto_chunk(bool enabled) {
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "Qwen3-VL large-image AUTO chunk policy must be set before the first forward");
        TORCH_CHECK(!enabled || large_image_auto_chunk_profile(),
                    "Qwen3-VL large-image AUTO chunk policy requires an ordinary "
                    "TP8 FP16/ACC32 24-layer Vision tower with taps [5,11,17]");
        large_image_auto_chunk_ = enabled;
        invalidate_model_state();
    }

    void set_n1200_tm160(bool enabled) {
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "Qwen3-VL N1200 Tm160 must be set before the first forward");
        TORCH_CHECK(!enabled || ordinary_n1200_tm160_profile(),
                    "Qwen3-VL N1200 Tm160 requires ordinary 4B W8 Vision");
        n1200_tm160_enabled_ = enabled;
        invalidate_model_state();
    }

    void set_fp16_chunked_merger(bool enabled) {
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "Qwen3-VL FP16 chunked merger must be set before the first forward");
        TORCH_CHECK(!enabled || ordinary_fp16_chunked_merger_profile(),
                    "Qwen3-VL FP16 chunked merger requires ordinary TP8 FP16/ACC32 "
                    "L24/H1024/I4096/Q16/D64 with 2048/2560 output and all three mergers");
        fp16_chunked_merger_enabled_ = enabled;
        invalidate_model_state();
    }

    void set_fp16_compact_encoder(bool enabled) {
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "Qwen3-VL FP16 compact encoder must be set before the first forward");
        TORCH_CHECK(!enabled || (fp16_chunked_merger_enabled_ &&
                                ordinary_fp16_chunked_merger_profile()),
                    "Qwen3-VL FP16 compact encoder requires the ordinary TP8 ACC32 "
                    "L24/H1024/I4096/Q16/D64 chunked-merger profile");
        fp16_compact_encoder_enabled_ = enabled;
        invalidate_model_state();
    }

    void set_w8_compact_encoder(bool enabled) {
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "Qwen3-VL W8 compact encoder must be set before the first forward");
        TORCH_CHECK(!enabled || ordinary_w8_compact_encoder_profile(),
                    "Qwen3-VL W8 compact encoder requires ordinary TP8 ACC16 "
                    "L24/H1024/I4096/Q16/D64 and W8 final/three DeepStack mergers");
        w8_compact_encoder_enabled_ = enabled;
        invalidate_model_state();
    }

    void set_delivery_compatibility(bool enabled) {
        TORCH_CHECK(!enabled || num_cores() == 8,
                    "Vision delivery compatibility retains its eight-core profile");
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "Qwen3-VL vision delivery compatibility must be set before the first forward");
        TORCH_CHECK(!enabled ||
                        (num_layers() == 24 && hidden_size() == 1024 &&
                         intermediate_size() == 4096 && num_q_heads() == 16 &&
                         head_dim() == 64 && !linear_acc32_),
                    "Qwen3-VL vision delivery compatibility requires the 2B ACC16 profile");
        TORCH_CHECK(!enabled || (has_rope_ && max_hw_ >= 24),
                    "delivery vision requires registered 24x24 RoPE tables");
        delivery_compatibility_ = enabled;
        if (enabled) {
            rope_route_request_ = Qwen3VLVisionRopeRouteRequest::SPM;
            rope_spm_enabled_ = true;
        }
        invalidate_model_state();
    }

    int64_t resolved_chunk_size() const {
        return get_last_resolved_chunk_size();
    }

    std::vector<int64_t> resolve_stage_domain(
        int64_t num_patches,
        int64_t image_batch_count)
    {
        TORCH_CHECK(!multiview_dry_prepared_ &&
                        !multiview_composite_prepared_ &&
                        !multiview_composite_dispatch_ &&
                        z1_prepared_num_patches_ == 0 && !z1_adopted_ &&
                        !z1_bound_ && !z1_dispatch_,
                    "Qwen3VLVisionModel::resolve_stage_domain: pooler Z1 or "
                    "multiview layout is prepared or active");
        TORCH_CHECK(num_patches > 0 &&
                        num_patches <= QWEN3VL_VISION_MAX_KEEPALIVE_SEQ,
                    "Qwen3VLVisionModel::resolve_stage_domain: num_patches=",
                    num_patches, " must be in (0, MAX_KEEPALIVE_SEQ=",
                    QWEN3VL_VISION_MAX_KEEPALIVE_SEQ, "]");
        TORCH_CHECK(image_batch_count >= 1 &&
                        num_patches % image_batch_count == 0,
                    "Qwen3VLVisionModel::resolve_stage_domain: num_patches=",
                    num_patches, " must be divisible by image_batch_count=",
                    image_batch_count);

        const int64_t saved_num_patches = current_num_patches_;
        const int64_t saved_image_batch_count = image_batch_count_;
        const int64_t saved_chunk_override = get_chunk_size_override();
        auto restore = c10::make_scope_exit([&] {
            current_num_patches_ = saved_num_patches;
            image_batch_count_ = saved_image_batch_count;
            set_chunk_size_override(saved_chunk_override);
        });

        current_num_patches_ = num_patches;
        image_batch_count_ = image_batch_count;
        // The descriptor resolver must expose the real bounded native domain,
        // not turn the legacy default into a hidden exact override.  The
        // subclass cap below retains C144 for legacy/small-image AUTO and
        // exposes the cold ordinary large-image cap separately. Its validity
        // predicate retains the existing packed-image/fused-merger restrictions.
        set_chunk_size_override(0);

        const int64_t patches_per_image =
            num_patches / image_batch_count_;
        const std::vector<ChunkInfo> input_chunks{
            {0, 0, num_patches, num_patches}};
        std::vector<FmbExecutionSpan> spans;
        spans.reserve(image_batch_count_);
        for (int64_t image = 0; image < image_batch_count_; ++image) {
            spans.push_back(
                {image * patches_per_image, patches_per_image, image});
        }
        const FmbStageBoundaryPolicies boundary_policies{};
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                num_patches, /*position=*/0, /*attention_mask=*/std::nullopt,
                /*is_causal=*/false, input_chunks, spans, boundary_policies,
                configured_chunk_size_));
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
        int64_t image_batch_count,
        at::IntArrayRef planned_stage_descriptor = {})
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
                            image_batch_count, /*z1_dispatch=*/false,
                            planned_stage_descriptor);
    }

    at::Tensor forward_impl(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches_in,
        int64_t image_batch_count,
        bool z1_dispatch,
        at::IntArrayRef planned_stage_descriptor)
    {
        const bool pipeline_dispatch = pipeline_dispatch_active();
        TORCH_CHECK(!planned_stage_descriptor.empty(),
                    "Qwen3VLVisionModel forward requires the native planned "
                    "stage descriptor resolved before BUILD");
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
        const bool previous_rope_spm = rope_spm_enabled_;
        rope_spm_enabled_ =
            selected_rope_residency(
                planned_stage_descriptor,
                /*require_spm=*/pipeline_dispatch) ==
            FmbRopeTableResidency::SPM;
        auto restore_rope_route = c10::make_scope_exit(
            [&] { rope_spm_enabled_ = previous_rope_spm; });
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
        if (num_cores() != 8) {
            TORCH_CHECK(input.scalar_type() == at::kHalf && input.size(2) == hidden_size() &&
                            image_batch_count == 1 && !pipeline_dispatch,
                        "four-core Vision requires one ordinary FP16 image per dispatch");
            TORCH_CHECK(static_cast<int64_t>(k_caches.size()) == num_layers() &&
                            static_cast<int64_t>(v_caches.size()) == num_layers(),
                        "four-core Vision requires one paired cache per layer");
            int64_t blocks = 0;
            for (size_t layer = 0; layer < k_caches.size(); ++layer) {
                for (const auto& cache : {k_caches[layer], v_caches[layer]}) {
                    TORCH_CHECK(cache.defined() && cache.device() == input.device() &&
                                    cache.scalar_type() == at::kHalf && cache.is_contiguous() &&
                                    cache.dim() == 7,
                                "four-core Vision requires contiguous FP16 RPU 7D caches");
                    if (blocks == 0) blocks = cache.size(1);
                    TORCH_CHECK(blocks > 0 && blocks <= std::numeric_limits<int64_t>::max() / 16 &&
                                    cache.sizes().vec() == std::vector<int64_t>(
                                    {1, blocks, 4, head_dim() / 16, 8, 16, 16}) &&
                                    num_patches_in <= blocks * 16,
                                "four-core Vision cache must match TP4 and eight physical stripes");
                }
            }
        }
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
                // inc-2: stable merged-output slot [num_patches/4, oh] for the in-graph deepstack
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
        const int64_t local_q_dim = (num_q_heads() / num_cores()) * head_dim();
        const std::pair<int64_t, int64_t> q_key{num_patches_in, local_q_dim};
        if (!pipeline_dispatch &&
            q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
            TORCH_CHECK(q_ddr_slots_.size() < 128,
                        "Qwen3VLVisionModel: q_ddr_slots_ exceeded 128 distinct "
                        "(num_patches,width) shapes on one handle");
            q_ddr_slots_.emplace(q_key, at::empty(
                {num_cores(), num_patches_in, local_q_dim},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1)));
        }
        TORCH_CHECK(q_ddr_slots_.count(q_key) == 1,
                    "Qwen3VLVisionModel pooler Z1 requires its fixed Q staging "
                    "slot to be created during Graph-external prepare");

        // The decoded descriptor is the sole stage-plan authority.  A stale
        // legacy override must neither veto it nor trigger AUTO reselection
        // inside production forward.
        set_chunk_size_override(0);
        at::Tensor result = run_all_layers(
            input, k_caches, v_caches,
            /*mask=*/std::nullopt, /*position=*/0,
            /*is_causal=*/false, /*planned_chunk_size=*/0,
            planned_stage_descriptor);

        restore_rope_route.release();
        return result;
    }

    SpmPipelineComponentLayout prepare_pooler_z1_layout(
        int64_t num_patches,
        int64_t retained_count, at::IntArrayRef selected_descriptor = {}) {
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
                        retained_deepstack_addresses_empty() &&
                        native_composite_stage_descriptor_.empty(),
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
            (num_q_heads() / num_cores()) * head_dim();
        const std::pair<int64_t, int64_t> q_key{num_patches, local_q_dim};
        if (q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
            TORCH_CHECK(q_ddr_slots_.size() < 128,
                        "Qwen3VLVisionModel pooler Z1 cannot create its Q "
                        "staging slot: per-shape slot cap reached");
            q_ddr_slots_.emplace(
                q_key,
                at::empty(
                    {num_cores(), num_patches, local_q_dim},
                    at::TensorOptions()
                        .dtype(at::kHalf)
                        .device(at::kPrivateUse1)));
        }
        const at::Tensor& q_slot = q_ddr_slots_.at(q_key);
        TORCH_CHECK(q_slot.scalar_type() == at::kHalf &&
                        q_slot.device().type() == at::kPrivateUse1 &&
                        q_slot.is_contiguous() &&
                        q_slot.dim() == 3 &&
                        q_slot.size(0) == num_cores() &&
                        q_slot.size(1) == num_patches &&
                        q_slot.size(2) == local_q_dim,
                    "Qwen3VLVisionModel pooler Z1 Q staging slot drifted");

        std::vector<int64_t> stage_descriptor;
        if (selected_descriptor.empty()) {
            stage_descriptor = build_native_composite_stage_descriptor(
                /*multiview=*/false, retained_count);
        } else {
            (void)kvinsert_cost_domain("qwen3vl_vision", selected_descriptor);
            TORCH_CHECK(kvinsert_cost_owner_prefix(selected_descriptor) ==
                            (std::vector<int64_t>{INT64_C(0x51564d5656434f53), 0, retained_count}),
                        "Z1 Vision selection was not produced by its exact native lifecycle");
            stage_descriptor.assign(selected_descriptor.begin(), selected_descriptor.end());
        }

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
        FmbThreeStageChunkPlan stage_plan =
            compose_fmb_default_three_stage_chunk_plan(
                layout, num_patches, /*position=*/0,
                ChunkMode::KV_FIRST);
        // One image is semantic group 0 in the selected composite descriptor.
        stage_plan.spans.front().group_id = 0;
        bind_descriptor_layout(
            layout, stage_plan, stage_descriptor, /*require_spm=*/true);
        native_composite_prepare_active_ = true;
        auto clear_prepare = c10::make_scope_exit(
            [&] { native_composite_prepare_active_ = false; });
        auto prepared = prepare_spm_pipeline_component(layout, stage_plan);
        clear_prepare.release();
        native_composite_prepare_active_ = false;
        TORCH_CHECK(prepared.temporary_bytes ==
                        QWEN3VL_POOLER_Z1_SCRATCH_BYTES,
                    "Qwen3VLVisionModel pooler Z1 scratch layout drifted: "
                    "expected ", QWEN3VL_POOLER_Z1_SCRATCH_BYTES,
                    " bytes/core, got ", prepared.temporary_bytes);
        z1_retained_deepstack_count_ = retained_count;
        z1_prepared_num_patches_ = num_patches;
        native_composite_stage_descriptor_ = std::move(stage_descriptor);
        return prepared;
    }

    SpmPipelineComponentLayout prepare_multiview_dry_layout(
        int64_t num_patches) {
        TORCH_CHECK(num_cores() == 8, "Vision multiview dry profile requires eight cores");
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
                z1_primed_position_owner_ == nullptr &&
                native_composite_stage_descriptor_.empty(),
            "Qwen3VLVisionModel multiview dry prepare requires a fresh "
            "lifecycle without a physical Z1 profile");
        TORCH_CHECK(
            num_patches == 256 && num_layers() == 24 &&
                num_q_heads() == 16 && head_dim() == 64 &&
                hidden_size() == 1024 && intermediate_size() == 4096,
            "Qwen3VLVisionModel multiview dry prepare accepts only the exact "
            "N256/L24/H1024/I4096/Q16/D64 Vision profile");
        TORCH_CHECK(
            spm_rope_route_available() && max_hw_ == 48 &&
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

        std::vector<int64_t> stage_descriptor =
            build_native_composite_stage_descriptor(
                /*multiview=*/true, /*retained_count=*/0);

        LayoutContext layout;
        layout.chunk_size = num_patches;
        layout.max_kv_seq_len = num_patches;
        layout.num_layers = num_layers();
        layout.use_attn_mask = false;
        layout.is_causal = false;
        FmbThreeStageChunkPlan stage_plan =
            compose_fmb_default_three_stage_chunk_plan(
                layout, num_patches, /*position=*/0,
                ChunkMode::KV_FIRST);
        // Match the single-image span used by native descriptor selection.
        stage_plan.spans.front().group_id = 0;
        bind_descriptor_layout(
            layout, stage_plan, stage_descriptor, /*require_spm=*/true);
        multiview_dry_prepared_ = true;
        auto rollback = c10::make_scope_exit(
            [&] { multiview_dry_prepared_ = false; });
        SpmPipelineComponentLayout prepared =
            prepare_spm_pipeline_component_for_cpu_contract(
                layout, stage_plan);
        rollback.release();
        native_composite_stage_descriptor_ = std::move(stage_descriptor);
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
        native_composite_stage_descriptor_.clear();
    }

    SpmPipelineComponentLayout prepare_multiview_composite_layout(
        int64_t num_patches, at::IntArrayRef selected_descriptor = {}) {
        TORCH_CHECK(
            !RpuKernelGraph::has_active(),
            "Qwen3VLVisionModel multiview composite prepare must run outside "
            "Graph capture");
        TORCH_CHECK(
            num_patches == QWEN3VL_POOLER_Z1_NUM_PATCHES &&
                !multiview_dry_prepared_ &&
                !multiview_composite_prepared_ &&
                !multiview_composite_dispatch_ &&
                z1_prepared_num_patches_ == 0 && !z1_inputs_primed_ &&
                        !z1_adopted_ && !z1_bound_ && !z1_dispatch_ &&
                        z1_retained_deepstack_count_ == 0 &&
                        z1_retained_deepstack_bound_count_ == 0 &&
                        retained_deepstack_addresses_empty() &&
                        native_composite_stage_descriptor_.empty(),
            "Qwen3VLVisionModel multiview composite prepare requires one "
            "fresh exact N256 component lifecycle");
        validate_multiview_composite_profile();
        std::vector<int64_t> stage_descriptor;
        if (selected_descriptor.empty()) {
            stage_descriptor = build_native_composite_stage_descriptor(
                /*multiview=*/true, /*retained_count=*/0);
        } else {
            (void)kvinsert_cost_domain("qwen3vl_vision", selected_descriptor);
            TORCH_CHECK(kvinsert_cost_owner_prefix(selected_descriptor) ==
                            (std::vector<int64_t>{INT64_C(0x51564d5656434f53), 1, 0}),
                        "multiview Vision selection was not produced by its exact native lifecycle");
            stage_descriptor.assign(selected_descriptor.begin(), selected_descriptor.end());
        }
        const int64_t local_q_dim =
            (num_q_heads() / num_cores()) * head_dim();
        const std::pair<int64_t, int64_t> q_key{
            num_patches, local_q_dim};
        if (q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
            TORCH_CHECK(q_ddr_slots_.size() < 128,
                        "Qwen3VLVisionModel multiview composite cannot create "
                        "its fixed Q staging slot: per-shape cap reached");
            q_ddr_slots_.emplace(
                q_key,
                at::empty(
                    {num_cores(), num_patches, local_q_dim},
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
        FmbThreeStageChunkPlan stage_plan =
            compose_fmb_default_three_stage_chunk_plan(
                layout, num_patches, /*position=*/0,
                ChunkMode::KV_FIRST);
        // Match the single-image span used by native descriptor selection.
        stage_plan.spans.front().group_id = 0;
        bind_descriptor_layout(
            layout, stage_plan, stage_descriptor, /*require_spm=*/true);
        multiview_composite_prepared_ = true;
        auto rollback = c10::make_scope_exit(
            [&] {
                multiview_composite_prepared_ = false;
            });
        SpmPipelineComponentLayout prepared =
            prepare_spm_pipeline_component(layout, stage_plan);
        TORCH_CHECK(
            prepared.temporary_bytes ==
                QWEN3VL_MULTIVIEW_VISION_SCRATCH_BYTES,
            "Qwen3VLVisionModel multiview composite scratch drifted: "
            "expected ", QWEN3VL_MULTIVIEW_VISION_SCRATCH_BYTES,
            " bytes/core, got ", prepared.temporary_bytes);
        rollback.release();
        native_composite_stage_descriptor_ = std::move(stage_descriptor);
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
        native_composite_stage_descriptor_.clear();
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
        native_composite_stage_descriptor_.clear();
    }

    std::vector<int64_t> native_composite_stage_descriptor() const {
        TORCH_CHECK(
            (z1_prepared_num_patches_ != 0 || multiview_dry_prepared_ ||
             multiview_composite_prepared_) &&
                !native_composite_stage_descriptor_.empty(),
            "Qwen3VLVisionModel native composite stage descriptor requires "
            "one prepared Vision child");
        return native_composite_stage_descriptor_;
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
        TORCH_CHECK(!native_composite_stage_descriptor_.empty(),
                    "Qwen3VLVisionModel pooler Z1 forward has no retained "
                    "physical descriptor");

        z1_dispatch_ = true;
        try {
            (void)forward_impl(input, k_caches, v_caches, num_patches,
                               /*image_batch_count=*/1,
                               /*z1_dispatch=*/true,
                               native_composite_stage_descriptor_);
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
        TORCH_CHECK(!native_composite_stage_descriptor_.empty(),
                    "Qwen3VLVisionModel multiview forward has no retained "
                    "physical descriptor");
        multiview_composite_dispatch_ = true;
        auto clear_dispatch = c10::make_scope_exit(
            [&] { multiview_composite_dispatch_ = false; });
        (void)forward_impl(
            input, k_caches, v_caches,
            QWEN3VL_POOLER_Z1_NUM_PATCHES,
            /*image_batch_count=*/1,
            /*z1_dispatch=*/false,
            native_composite_stage_descriptor_);
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
                        native_composite_stage_descriptor_.empty() &&
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

    // inc-2: retrieve the IN-GRAPH merged deepstack features [num_patches/4, oh] for the
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
    DecoderExecutionTopology resolve_model_execution_topology(
        int64_t q, int64_t kv, int64_t d, int64_t h, int64_t intermediate) const override {
        if (num_cores() == 8) {
            return FusedModelBase::resolve_model_execution_topology(q, kv, d, h, intermediate);
        }
        TORCH_CHECK(num_cores() == 4 && q == 16 && kv == 16 &&
                        ((h == 1024 && intermediate == 4096 && d == 64) ||
                         (h == 1152 && intermediate == 4352 && d == 80)),
                    "no admitted four-core Vision physical geometry");
        return {4, 4, 4};
    }

    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            current_num_patches_, image_batch_count_,
            z1_dispatch_, multiview_composite_dispatch_,
            z1_retained_deepstack_count_);
    }

    int64_t subclass_layout_hash() const override {
        int64_t hash = rhinovla_high_precision_
            ? detail::layout_mix(0, RHINOVLA_VISION_HIGH_ACTIVATION_SITE) : 0;
        if (rhinovla_high_precision_) hash = detail::layout_mix(hash, 2);
        if (num_cores() == 8) return hash;
        hash = detail::layout_mix(hash, 2);
        for (const int value : {num_cores(), attn_tp(), mlp_tp(), 8}) {
            hash = detail::layout_mix(hash, value);
        }
        return hash;
    }

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
        // Fused patch merger (ROUND-3 opt #1): run the merger inside the same capture
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
        // Chunked post hooks refresh the fresh raw-output mutable DMA owner.
        cfg.fast_replay_bake_post_fn =
            !pipeline_dispatch_active() && bake_merger_ &&
            !fp16_chunked_merger_enabled_ && !w8_compact_encoder_active();
        if (rope_spm_enabled_ &&
            !rope_tables_use_persistent_storage()) {
            cfg.pre_layers_fn = static_cast<void(FusedModelBase::*)()>(
                &Qwen3VLVisionModel::emit_rope_table_preload);
        }
        return cfg;
    }

    // Ordinary 2B/4B W8 capability. This is independent of the fixed 2B
    // delivery and typed composite contracts, including their SDPA tiling.
    bool ordinary_w8_resident_profile() const {
        if (pipeline_dispatch_active() || qwen3vl_2b_delivery_profile_active() ||
            !adaptive_w8_merger() || linear_acc32_ ||
            (image_batch_count_ != 1 &&
             !(image_batch_count_ == 2 &&
               (current_num_patches_ == 512 || current_num_patches_ == 1024)) &&
             !(image_batch_count_ == 3 && current_num_patches_ == 768)) ||
            num_cores() != 8 || num_layers() != 24 || hidden_size() != 1024 ||
            intermediate_size() != 4096 || num_q_heads() != 16 ||
            head_dim() != 64 || orig_head_dim_ != 64 ||
            (merger_out_hidden_ != 2048 && merger_out_hidden_ != 2560) ||
            layer_weights_.size() != 24) {
            return false;
        }
        return std::all_of(layer_weights_.begin(), layer_weights_.end(),
            [](const auto& weights) {
                for (const at::Tensor* weight : {
                        &weights.q_w, &weights.k_w, &weights.v_w,
                        &weights.o_w, &weights.fc1_w, &weights.fc2_w}) {
                    if (weight->scalar_type() != at::kChar) return false;
                }
                return true;
            });
    }

    bool ordinary_w8_resident_chunk(int64_t chunk_size,
                                    int64_t sequence_length) const {
        return sequence_length > 0 && chunk_size >= sequence_length &&
            (image_batch_count_ == 1 || sequence_length == current_num_patches_) &&
            ordinary_w8_resident_profile();
    }

    bool ordinary_n1200_tm160_profile() const {
        return ordinary_w8_resident_profile() && image_batch_count_ == 1 &&
            !delivery_compatibility_ && !prefix_scatter_active_ &&
            !native_composite_prepare_active_ && z1_prepared_num_patches_ == 0 &&
            !z1_adopted_ && !z1_bound_ && !multiview_dry_prepared_ &&
            !multiview_composite_prepared_ &&
            deepstack_visual_indexes_ == std::vector<int64_t>{5, 11, 17};
    }

    bool n1200_tm160_eligible(
        const FmbThreeStageChunkPlan& plan, int64_t physical_len,
        int64_t logical_len, int64_t position) const {
        if (!n1200_tm160_enabled_ || !ordinary_n1200_tm160_profile() ||
            physical_len != 1200 || logical_len != 1200 || position != 0 ||
            plan.chunk_mode != ChunkMode::KV_FIRST ||
            plan.compute.plan.chunk_size != 608 || plan.compute.chunks.size() != 2)
            return false;
        const auto& first = plan.compute.chunks[0];
        const auto& tail = plan.compute.chunks[1];
        return first.idx == 0 && first.offset == 0 && first.len == 608 &&
            tail.idx == 1 && tail.offset == 608 && tail.len == 592 &&
            rpu_qwen3vl_n1200_tm160_tiling(first.len, physical_len, QWEN3VL_N1200_SDPA_TMP_BYTES) &&
            rpu_qwen3vl_n1200_tm160_tiling(tail.len, physical_len, QWEN3VL_N1200_SDPA_TMP_BYTES);
    }

    std::vector<int64_t> n1200_tm160_arguments(const ChunkInfo& chunk) const {
        const auto t = rpu_qwen3vl_n1200_tm160_tiling(
            chunk.len, 1200, QWEN3VL_N1200_SDPA_TMP_BYTES);
        TORCH_CHECK(t && ((chunk.idx == 0 && chunk.offset == 0 && chunk.len == 608) ||
                         (chunk.idx == 1 && chunk.offset == 608 && chunk.len == 592)),
                    "Qwen3-VL N1200 Tm160 invocation drift");
        const int64_t grid_x = CeilDiv(chunk.len, t->tile_m);
        const int64_t required_tmp = t->tile_n_v16 * t->tile_k * 2 * grid_x * 32;
        return {/*version=*/1, /*B=*/1, /*Sk=*/1200, chunk.offset, chunk.len,
                /*Q=*/16, /*KV=*/16, /*D=*/64, /*cores=*/8, /*vtp=*/8,
                /*MASK_NONE=*/0, float32_to_uint32(0.125f),
                t->tile_m, t->tile_n, t->tile_k, grid_x, 2, 1,
                required_tmp, QWEN3VL_N1200_SDPA_TMP_BYTES};
    }

    bool ordinary_w8_raw_chunk(int64_t chunk_size,
                              int64_t sequence_length) const {
        // N768's whole-query grid is illegal (ceil(768/80)=10). The owner
        // emits two legal Sq384 calls sharing the complete Sk768 instead.
        if (image_batch_count_ <= 0 || sequence_length % image_batch_count_ != 0)
            return false;
        const int64_t image_rows = sequence_length / image_batch_count_;
        const int64_t query_rows = image_rows == 768 ? 384 : image_rows;
        return ((image_batch_count_ == 1 &&
                 (sequence_length == 256 || sequence_length == 512 || sequence_length == 768)) ||
                (image_batch_count_ == 2 &&
                 (sequence_length == 512 || sequence_length == 1024)) ||
                (image_batch_count_ == 3 && sequence_length == 768)) &&
            chunk_size == sequence_length && sequence_length == current_num_patches_ && ordinary_w8_resident_profile() &&
            sdpa_by_mha_spm_is_valid(
                image_batch_count_, query_rows, image_rows, num_q_heads(), num_q_heads(),
                head_dim(), num_cores(), /*MASK_NONE=*/0);
    }

    bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const override {
        if (!(ordinary_w8_raw_chunk(layout.chunk_size, current_num_patches_) ||
              compact_encoder_chunk(layout.chunk_size, current_num_patches_)) ||
            position != 0 ||
            layout.is_causal || layout.use_attn_mask || layout.batch_size != 1 ||
            layout.attention_policy != AttentionExecutionPolicy::SPM_KV_BY_MHA ||
            plan.chunk_mode != ChunkMode::SEQUENTIAL ||
            plan.input.chunks.size() != 1 || plan.qkv.chunks.size() != 1 ||
            plan.compute.chunks.size() != 1 ||
            plan.spans.size() != static_cast<size_t>(image_batch_count_)) {
            return false;
        }
        // Bounded ordinary 2B/4B W8 route; larger images retain the DDR candidate.
        // This is separate from the fixed 2B delivery schedule and tiling.
        const int64_t rows = current_num_patches_;
        if (layout.effective_kv_cs() != rows || layout.max_kv_seq_len != rows) {
            return false;
        }
        const int64_t image_rows = rows / image_batch_count_;
        for (int64_t image = 0; image < image_batch_count_; ++image) {
            const auto& span = plan.spans[image];
            if (span.offset != image * image_rows || span.len != image_rows ||
                span.group_id != image) return false;
        }
        for (const auto* stage : {&plan.input, &plan.qkv, &plan.compute}) {
            const ChunkInfo& chunk = stage->chunks.front();
            if (chunk.idx != 0 || chunk.offset != 0 || chunk.len != rows ||
                chunk.kv_seq_len != rows) return false;
        }
        return true;
    }

    void consume_ordinary_w8_resident_schedule() {
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN3VL_W8_RESIDENT_SCHEDULE_SITE,
            /*selector=*/1, /*resolved_flags=*/0,
            {current_num_patches_, image_batch_count_, 24, 1024, 4096, 16, 64, 8, merger_out_hidden_});
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        TORCH_CHECK(!((merger_active() && !adaptive_w8_merger()) || image_batch_count_ > 1) || plan.num_chunks == 1 ||
                        (fp16_chunked_merger_active() && plan.chunk_size == 608 && plan.num_chunks == 2),
                    "Qwen3VLVisionModel: fused merger and packed-image minibatch require "
                    "a single chunk, got num_chunks=", plan.num_chunks,
                    " chunk_size=", plan.chunk_size,
                    " patches=", current_num_patches_);
        ModelDynamicConfig cfg;
        const bool delivery = qwen3vl_2b_delivery_profile_active();
        TORCH_CHECK(!delivery || (current_num_patches_ == 576 && image_batch_count_ == 1 &&
                    plan.num_chunks == 1 && plan.chunk_size == 576),
                    "fixed delivery Vision requires one 576-patch image per graph");
        const bool resident = delivery ||
            (plan.num_chunks == 1 &&
             (ordinary_w8_resident_chunk(plan.chunk_size, current_num_patches_) ||
              compact_encoder_chunk(plan.chunk_size, current_num_patches_)));
        cfg.chunk_mode = resident
            ? ChunkMode::SEQUENTIAL
            : ChunkMode::KV_FIRST;
        cfg.inter_layer_io = resident
            ? InterLayerIO::SPM_RESIDENT
            : InterLayerIO::DDR_PINGPONG;
        // AUTO admits both descriptor-owned physical routes only where the
        // same shape/profile predicate used by the raw capability allows it.
        cfg.attention_policy = compact_encoder_chunk(plan.chunk_size, current_num_patches_)
            ? AttentionExecutionPolicy::SPM_KV_BY_MHA
            : plan.num_chunks == 1 && ordinary_w8_raw_chunk(
                plan.chunk_size, current_num_patches_)
            ? AttentionExecutionPolicy::AUTO
            : AttentionExecutionPolicy::DDR_KV;
        return cfg;
    }

    void consume_manifest_route(
        FmbRouteFamily family, int64_t site_id, int64_t selector,
        int64_t invocation = 0,
        at::IntArrayRef resolved_arguments = {},
        int64_t resolved_flags = 0) {
        if (!ctx().has_complete_physical_manifest()) return;
        std::vector<int64_t> arguments(resolved_arguments.begin(), resolved_arguments.end());
        if (num_cores() != 8 &&
            (family == FmbRouteFamily::LINEAR || family == FmbRouteFamily::ALL_REDUCE)) {
            arguments.insert(arguments.end(), {2, num_cores(), attn_tp(), mlp_tp(), 8});
        }
        ctx().consume_physical_route(
            family, site_id, selector, resolved_flags,
            arguments, invocation);
    }

    FmbPhysicalExecutionManifest make_physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout, int64_t physical_len,
        int64_t logical_len, int64_t position,
        bool rope_spm) const {
        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = position + logical_len;
        manifest.kv_insert_physical_rows = physical_len;
        manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation = linear_acc32_
            ? FmbLinearAccumulationPolicy::ACC32
            : FmbLinearAccumulationPolicy::ACC16;
        const bool raw_spm =
            layout.attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const bool n1200_tm160 = !raw_spm && !layout.is_causal && !layout.use_attn_mask &&
            n1200_tm160_eligible(plan, physical_len, logical_len, position);
        TORCH_CHECK(!raw_spm ||
                        (physical_len == logical_len &&
                         physical_len == current_num_patches_ &&
                         subclass_spm_kv_by_mha_eligible(plan, layout, position)),
                    "Qwen3-VL Vision raw-SPM descriptor exceeds its exact owner capability");
        const bool compact = compact_encoder_active();
        TORCH_CHECK(!compact ||
                        (raw_spm && compact_encoder_plan(
                            plan, physical_len, logical_len, position)),
                    "Qwen3-VL compact encoder requires unpadded B1/N1200/C1200 raw-SPM");

        auto append = [&](FmbRouteFamily family, int64_t site_id,
                          int64_t selector,
                          std::vector<int64_t> arguments = {},
                          int64_t invocation = 0,
                          int64_t flags = 0) {
            if (num_cores() != 8 &&
                (family == FmbRouteFamily::LINEAR || family == FmbRouteFamily::ALL_REDUCE)) {
                arguments.insert(arguments.end(), {2, num_cores(), attn_tp(), mlp_tp(), 8});
            }
            manifest.routes.push_back({
                site_id, family, selector, flags,
                std::move(arguments), invocation});
        };
        auto append_linear = [&](int64_t site_id, int64_t invocation) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                   {}, invocation);
        };
        if (rhinovla_high_precision_) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, RHINOVLA_VISION_HIGH_ACTIVATION_SITE,
                   1, {2, 1024, 4096, 2048, 4, 8});
            append(FmbRouteFamily::GRAPH_SCHEDULE, RHINOVLA_VISION_HIGH_ENCODER_ACTIVATION_SITE,
                   1, {2, 1024, 4096, 2048, 4, 8});
        }
        const bool pipeline = pipeline_dispatch_active();
        if (compact) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, compact_encoder_schedule_site(),
                   1, compact_encoder_schedule_arguments());
            for (int64_t invocation = 0; invocation < 2; ++invocation) {
                append(FmbRouteFamily::COLLECTIVE, QWEN3VL_FP16_COMPACT_COPY_SITE,
                       1, fp16_compact_residual_arguments(), invocation);
            }
        }
        if (qwen3vl_2b_delivery_profile_active()) {
            TORCH_CHECK(physical_len == 576 && logical_len == 576 && image_batch_count_ == 1 &&
                        plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                        rope_spm,
                        "fixed delivery Vision physical descriptor geometry/schedule drift");
            append(FmbRouteFamily::GRAPH_SCHEDULE, QWEN3VL_DELIVERY_VISION_SCHEDULE_SITE,
                   1, {current_num_patches_, 24, 1024, 4096, 16, 64, 1});
        }
        if (!compact && ordinary_w8_resident_chunk(plan.compute.plan.chunk_size, physical_len)) {
            TORCH_CHECK(plan.chunk_mode == ChunkMode::SEQUENTIAL &&
                            plan.compute.chunks.size() == 1 &&
                            physical_len == logical_len,
                        "ordinary W8 Vision resident descriptor schedule drift");
            append(FmbRouteFamily::GRAPH_SCHEDULE,
                   QWEN3VL_W8_RESIDENT_SCHEDULE_SITE, 1,
                   {physical_len, image_batch_count_, 24, 1024, 4096, 16, 64, 8, merger_out_hidden_});
        }
        if (rope_spm && max_hw_ > 0) {
            const int64_t selector = static_cast<int64_t>(
                Qwen3VLVisionMutableDmaRoute::ROPE_TABLE_DDR_TO_SPM);
            append(
                FmbRouteFamily::MUTABLE_DMA,
                pipeline ? QWEN3VL_ROPE_COS_PRELOAD_PIPELINE_SITE
                         : QWEN3VL_ROPE_COS_PRELOAD_DIRECT_SITE,
                selector,
                {rope_spm ? 1 : 0, max_hw_, orig_head_dim_});
            append(
                FmbRouteFamily::MUTABLE_DMA,
                pipeline ? QWEN3VL_ROPE_SIN_PRELOAD_PIPELINE_SITE
                         : QWEN3VL_ROPE_SIN_PRELOAD_DIRECT_SITE,
                selector,
                {rope_spm ? 1 : 0, max_hw_, orig_head_dim_});
        }
        for (const ChunkInfo& chunk : plan.qkv.chunks) {
            append_linear(QWEN3VL_Q_LINEAR_SITE, chunk.idx);
            append_linear(QWEN3VL_K_LINEAR_SITE, chunk.idx);
            append_linear(QWEN3VL_V_LINEAR_SITE, chunk.idx);
            append(
                FmbRouteFamily::ROPE,
                rope_spm
                    ? (pipeline ? QWEN3VL_Q_ROPE_SPM_PIPELINE_SITE
                                : QWEN3VL_Q_ROPE_SPM_SITE)
                    : QWEN3VL_Q_ROPE_DDR_SITE,
                static_cast<int64_t>(
                    rope_spm ? Qwen3VLVisionRopeRoute::ROPE_2D_SPM
                             : Qwen3VLVisionRopeRoute::ROPE_2D_DDR),
                {rope_spm ? 1 : 0, pipeline ? 1 : 0,
                 chunk.offset, chunk.len, num_q_heads() / num_cores(),
                 orig_head_dim_, head_dim(), num_cores()},
                chunk.idx,
                rope_spm ? FMB_ROUTE_FLAG_ROPE_TABLE_SPM
                         : FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
            append(
                FmbRouteFamily::ROPE,
                rope_spm
                    ? (pipeline ? QWEN3VL_K_ROPE_SPM_PIPELINE_SITE
                                : QWEN3VL_K_ROPE_SPM_SITE)
                    : QWEN3VL_K_ROPE_DDR_SITE,
                static_cast<int64_t>(
                    rope_spm ? Qwen3VLVisionRopeRoute::ROPE_2D_SPM
                             : Qwen3VLVisionRopeRoute::ROPE_2D_DDR),
                {rope_spm ? 1 : 0, pipeline ? 1 : 0,
                 chunk.offset, chunk.len, num_q_heads() / num_cores(),
                 orig_head_dim_, head_dim(), num_cores()},
                chunk.idx,
                rope_spm ? FMB_ROUTE_FLAG_ROPE_TABLE_SPM
                         : FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
            if (raw_spm) continue;
            const KvInsertSegmentPlan kv_plan =
                resolve_kvinsert_plan_auto(
                    QWEN3VL_KV_INSERT_SITE, manifest.graph_lifecycle,
                    position + chunk.offset, chunk.len, chunk.len,
                    num_cores(), num_q_heads(), head_dim(),
                    QWEN3VL_KV_CAPABILITIES);
            const KvInsertRouteArguments arguments =
                rpu_kvinsert_route_arguments(
                    kv_plan, num_cores(), num_q_heads(), head_dim());
            manifest.routes.push_back({
                QWEN3VL_KV_INSERT_SITE, FmbRouteFamily::KV_INSERT,
                static_cast<int64_t>(kv_plan.route()),
                QWEN3VL_KV_REASON_DDR_REQUIRED,
                {arguments.begin(), arguments.end()}, chunk.idx});
            manifest.kv_insert_physical_rows = std::max(
                manifest.kv_insert_physical_rows,
                kv_plan.physical_rows());
        }
        for (const ChunkInfo& chunk : plan.compute.chunks) {
            if (raw_spm) {
                const int64_t image_rows = chunk.len / image_batch_count_;
                const int64_t query_rows = vision_raw_query_chunk_size(image_rows);
                for (int64_t query_offset = 0; query_offset < image_rows;
                     query_offset += query_rows) {
                    append(FmbRouteFamily::ATTENTION, QWEN3VL_RAW_SPM_ATTN_SITE,
                           static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA),
                           vision_raw_attention_arguments(chunk.len, query_offset),
                           query_offset / query_rows);
                }
            } else if (n1200_tm160) {
                append(FmbRouteFamily::ATTENTION, QWEN3VL_N1200_TM160_ATTN_SITE,
                       static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                       n1200_tm160_arguments(chunk), chunk.idx);
            } else {
                append(FmbRouteFamily::ATTENTION,
                       image_batch_count_ > 1 ? QWEN3VL_MINIBATCH_ATTN_SITE
                                             : QWEN3VL_UNIFIED_ATTN_SITE,
                       static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                       vision_attention_arguments(logical_len / image_batch_count_, chunk.len),
                       chunk.idx);
            }
            append_linear(QWEN3VL_O_LINEAR_SITE, chunk.idx);
            append(
                FmbRouteFamily::ALL_REDUCE,
                compact ? QWEN3VL_COMPACT_ATTN_ALL_REDUCE_SITE
                        : QWEN3VL_ATTN_ALL_REDUCE_SITE,
                vision_ring_route(
                    chunk.len, hidden_size()),
                compact ? fp16_compact_residual_arguments() : std::vector<int64_t>{}, chunk.idx);
            append_linear(QWEN3VL_FC1_LINEAR_SITE, chunk.idx);
            append_linear(QWEN3VL_FC2_LINEAR_SITE, chunk.idx);
            append(
                FmbRouteFamily::ALL_REDUCE,
                compact ? QWEN3VL_COMPACT_MLP_ALL_REDUCE_SITE
                        : QWEN3VL_MLP_ALL_REDUCE_SITE,
                vision_ring_route(
                    chunk.len, hidden_size()),
                compact ? fp16_compact_residual_arguments() : std::vector<int64_t>{}, chunk.idx);
        }
        if (!pipeline_dispatch_active()) {
            const int64_t num_compute_chunks =
                static_cast<int64_t>(plan.compute.chunks.size());
            for (int64_t layer_idx : deepstack_visual_indexes_) {
                for (const ChunkInfo& chunk : plan.compute.chunks) {
                    append(
                        FmbRouteFamily::MUTABLE_DMA,
                        QWEN3VL_DEEPSTACK_SNAPSHOT_SITE,
                        static_cast<int64_t>(
                            Qwen3VLVisionMutableDmaRoute::SPM_COPY_TO_DDR),
                        {static_cast<int64_t>(
                             deepstack_visual_indexes_.size()),
                         layer_idx, chunk.offset, chunk.len, hidden_size()},
                        layer_idx * num_compute_chunks + chunk.idx);
                }
            }
        }
        if (fp16_chunked_merger_active()) {
            TORCH_CHECK(fp16_chunked_merger_plan(plan, physical_len, logical_len, position) &&
                            !layout.is_causal && !layout.use_attn_mask &&
                            layout.batch_size == 1 && (compact || !raw_spm),
                        "Qwen3-VL FP16 chunked merger requires its exact unpadded N1200/B1 encoder plan");
        }
        const auto merger_chunks = merger_compute_chunks(plan);
        const bool chunked_merger = merger_chunks.size() > 1 &&
            image_batch_count_ == 1 && !prefix_scatter_active_ &&
            (ordinary_w8_resident_profile() || fp16_chunked_merger_active());
        if (chunked_merger) {
            TORCH_CHECK(physical_len == logical_len && logical_len == current_num_patches_ &&
                            logical_len % 4 == 0,
                        "Qwen3-VL chunked merger requires complete unpadded patch groups");
            int64_t offset = 0;
            const int64_t chunks = static_cast<int64_t>(merger_chunks.size());
            for (int64_t index = 0; index < chunks; ++index) {
                const auto& chunk = merger_chunks[index];
                TORCH_CHECK(chunk.idx == index && chunk.offset == offset &&
                                chunk.len > 0 && chunk.len % 4 == 0 &&
                                chunk.len <= merger_compute_chunk_size(plan) &&
                                chunk.len <= logical_len - offset,
                            "Qwen3-VL chunked merger requires contiguous whole patch groups");
                offset += chunk.len;
            }
            TORCH_CHECK(offset == logical_len,
                        "Qwen3-VL chunked merger must cover the complete raw output");
            const int64_t merger_count = 1 + (deepstack_merger_active()
                ? static_cast<int64_t>(deepstack_visual_indexes_.size()) : 0);
            for (int64_t ordinal = 0; ordinal < merger_count; ++ordinal) {
                for (const auto& chunk : merger_chunks) {
                    const int64_t invocation = ordinal * chunks + chunk.idx;
                    append_linear(QWEN3VL_MERGER_M0_LINEAR_SITE, invocation);
                    append_linear(QWEN3VL_MERGER_M2_LINEAR_SITE, invocation);
                    append(FmbRouteFamily::ALL_REDUCE, QWEN3VL_MERGER_ALL_REDUCE_SITE,
                           vision_ring_route(chunk.len / 4, merger_out_hidden_), {}, invocation);
                    append(FmbRouteFamily::MUTABLE_DMA, QWEN3VL_CHUNKED_MERGER_INPUT_SITE,
                           static_cast<int64_t>(Qwen3VLVisionMutableDmaRoute::MERGER_INPUT_DDR_TO_SPM),
                           {ordinal, logical_len, chunk.offset, chunk.len, hidden_size()}, invocation);
                    append(FmbRouteFamily::MUTABLE_DMA, QWEN3VL_CHUNKED_MERGER_OUTPUT_SITE,
                           static_cast<int64_t>(Qwen3VLVisionMutableDmaRoute::SPM_COPY_TO_DDR),
                           {ordinal, logical_len / 4, chunk.offset / 4, chunk.len / 4,
                            merger_out_hidden_}, invocation);
                }
            }
        } else if (merger_active() && (!adaptive_w8_merger() || plan.compute.chunks.size() == 1)) {
            const int64_t merger_count = z1_dispatch_
                ? 1 + z1_retained_deepstack_count_
                : 1 + (deepstack_merger_active()
                           ? static_cast<int64_t>(
                                 deepstack_visual_indexes_.size())
                           : 0);
            const int64_t merger_all_reduce_site =
                multiview_composite_dispatch_
                ? QWEN3VL_MERGER_YIELD_ALL_REDUCE_SITE
                : QWEN3VL_MERGER_ALL_REDUCE_SITE;
            for (int64_t invocation = 0; invocation < merger_count;
                 ++invocation) {
                append_linear(QWEN3VL_MERGER_M0_LINEAR_SITE, invocation);
                append_linear(QWEN3VL_MERGER_M2_LINEAR_SITE, invocation);
                append(
                    FmbRouteFamily::ALL_REDUCE,
                    merger_all_reduce_site,
                    vision_ring_route(
                        logical_len / 4, merger_out_hidden_),
                    {}, invocation);
            }
            if (!pipeline_dispatch_active() && !prefix_scatter_active_) {
                const std::vector<int64_t> copy_arguments{
                    logical_len / 4, merger_out_hidden_,
                    prefix_scatter_active_ ? 1 : 0};
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    QWEN3VL_POOLER_MERGED_COPY_SITE,
                    static_cast<int64_t>(
                        Qwen3VLVisionMutableDmaRoute::SPM_COPY_TO_DDR),
                    copy_arguments);
                if (deepstack_merger_active()) {
                    for (size_t k = 0;
                         k < deepstack_visual_indexes_.size(); ++k) {
                        append(
                            FmbRouteFamily::MUTABLE_DMA,
                            QWEN3VL_DEEPSTACK_MERGED_COPY_SITE,
                            static_cast<int64_t>(
                                Qwen3VLVisionMutableDmaRoute::SPM_COPY_TO_DDR),
                            copy_arguments, static_cast<int64_t>(k) + 1);
                    }
                }
            }
        }
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA,
            1, num_cores(), mlp_tp());
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

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout, int64_t physical_len,
        int64_t logical_len, int64_t position) const override {
        return make_physical_manifest_for_candidate(
            plan, layout, physical_len, logical_len, position,
            rope_spm_enabled_);
    }

    std::vector<FmbPhysicalExecutionManifest>
    physical_manifest_domain_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout, int64_t physical_len,
        int64_t logical_len, int64_t position) const override {
        TORCH_CHECK(
            has_rope_ && max_hw_ > 0,
            "Qwen3VLVisionModel physical domain requires registered 2D "
            "RoPE tables");
        if (pipeline_dispatch_active()) {
            TORCH_CHECK(
                rope_route_request_ != Qwen3VLVisionRopeRouteRequest::DDR,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3-VL native composite "
                "requires SPM 2D RoPE but the cold exact request selects DDR");
            return {make_physical_manifest_for_candidate(
                plan, layout, physical_len, logical_len, position,
                /*rope_spm=*/true)};
        }
        std::vector<FmbPhysicalExecutionManifest> domain;
        if (compact_encoder_active() &&
            !compact_encoder_plan(plan, physical_len, logical_len, position)) {
            return domain;
        }
        auto append_rope_domain = [&](const LayoutContext& attention_layout) {
            if (rope_route_request_ != Qwen3VLVisionRopeRouteRequest::SPM) {
                domain.push_back(make_physical_manifest_for_candidate(
                    plan, attention_layout, physical_len, logical_len, position,
                    /*rope_spm=*/false));
            }
            if (rope_route_request_ != Qwen3VLVisionRopeRouteRequest::DDR) {
                domain.push_back(make_physical_manifest_for_candidate(
                    plan, attention_layout, physical_len, logical_len, position,
                    /*rope_spm=*/true));
            }
        };
        LayoutContext ddr_layout = layout;
        ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;
        if (!compact_encoder_active()) append_rope_domain(ddr_layout);
        LayoutContext raw_layout = layout;
        raw_layout.attention_policy = AttentionExecutionPolicy::SPM_KV_BY_MHA;
        if (physical_len == logical_len &&
            subclass_spm_kv_by_mha_eligible(plan, raw_layout, position)) {
            append_rope_domain(raw_layout);
        }
        return domain;
    }

    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& /*manifest*/) const override {
        return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
    }

    bool large_image_auto_chunk_profile() const {
        if (num_cores() != 8 || image_batch_count_ != 1 ||
            num_layers() != 24 || hidden_size() != 1024 ||
            intermediate_size() != 4096 || num_q_heads() != 16 ||
            head_dim() != 64 || orig_head_dim_ != 64 || !linear_acc32_ ||
            delivery_compatibility_ || merger_active() || prefix_scatter_active_ ||
            pipeline_dispatch_active() || native_composite_prepare_active_ ||
            z1_prepared_num_patches_ != 0 || z1_adopted_ || z1_bound_ ||
            multiview_dry_prepared_ || multiview_composite_prepared_ ||
            deepstack_visual_indexes_.size() != 3 ||
            deepstack_visual_indexes_[0] != 5 ||
            deepstack_visual_indexes_[1] != 11 ||
            deepstack_visual_indexes_[2] != 17 ||
            layer_weights_.size() != 24) {
            return false;
        }
        return std::all_of(layer_weights_.begin(), layer_weights_.end(),
            [](const auto& weights) {
                for (const at::Tensor* weight : {
                        &weights.q_w, &weights.k_w, &weights.v_w,
                        &weights.o_w, &weights.fc1_w, &weights.fc2_w}) {
                    if (weight->scalar_type() != at::kHalf) return false;
                }
                return true;
            });
    }

    int64_t subclass_chunk_size_cap(
        int64_t seq_len, int64_t /*position*/) const override {
        if (configured_chunk_size_ > 0) {
            return configured_chunk_size_;
        }
        if (compact_encoder_active() && seq_len == 1200) return 1200;
        if (fp16_chunked_merger_active() && seq_len == 1200) return 608;
        if (merger_active() || image_batch_count_ > 1) {
            return ((seq_len + 15) / 16) * 16;
        }
        // This cold opt-in belongs to the ordinary 2B/4B adapter, not shared
        // tower geometry alone. Preserve formal N784 and exact requests. The
        // native planner still rejects candidates exceeding SPM/SDPA limits.
        if (large_image_auto_chunk_ &&
            seq_len > QWEN3VL_VISION_LARGE_IMAGE_THRESHOLD &&
            large_image_auto_chunk_profile()) {
            return QWEN3VL_VISION_LARGE_IMAGE_CHUNK_CAP;
        }
        return QWEN3VL_VISION_DEFAULT_CHUNK_SIZE;
    }

    bool subclass_chunk_size_valid(
        int64_t chunk_size,
        int64_t seq_len,
        int64_t position) const override {
        if (compact_encoder_active()) {
            return seq_len == 1200 && position == 0 && chunk_size == 1200;
        }
        if (fp16_chunked_merger_active()) {
            return seq_len == 1200 && position == 0 && chunk_size == 608;
        }
        return !((merger_active() && !adaptive_w8_merger()) || image_batch_count_ > 1) ||
            chunk_size >= seq_len;
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        int64_t comp_cs = ctx.chunk_size;
        int64_t kv_cs   = ctx.effective_kv_cs();
        int64_t wide_cs = std::max(comp_cs, kv_cs);
        const bool compact = compact_encoder_chunk(comp_cs, ctx.max_kv_seq_len);
        int64_t cs      = compact ? 608 : comp_cs;  // independent post-merger tile
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();
        const bool rope_spm_layout =
            ctx.rope_table_residency == FmbRopeTableResidency::UNSPECIFIED
            ? rope_route_request_ == Qwen3VLVisionRopeRouteRequest::SPM
            : ctx.rope_table_residency == FmbRopeTableResidency::SPM;

        int64_t local_q_dim = (nq / num_cores()) * hd;
        int64_t local_inter = is_ / num_cores();
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        int64_t res    = A(wide_cs * h * DWIDTH);
        int64_t q_kv   = A(kv_cs * local_q_dim * DWIDTH);
        int64_t q_comp = A(comp_cs * local_q_dim * DWIDTH);
        int64_t oproj  = A(comp_cs * h * DWIDTH);
        int64_t fc1    = A(comp_cs * local_inter * DWIDTH);

        SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                            hd, /*nq*/nq, /*nkv*/nq,
                            /*cores*/num_cores(), /*mask*/0};
        SdpaTiling t = sdpa_compute_tiling(sdpa_cfg, comp_cs);
        int64_t nkv_per_core = CeilDiv(nq, (int64_t)num_cores());
        int64_t sdpa_tmp = A(t.tile_n_v16 * t.tile_k * nkv_per_core * CeilDiv(comp_cs, t.tile_m) * 32);
        if (n1200_tm160_enabled_ && ordinary_n1200_tm160_profile() &&
            ctx.max_kv_seq_len == 1200 && comp_cs == 608) {
            TORCH_CHECK(sdpa_tmp == QWEN3VL_N1200_SDPA_TMP_BYTES,
                        "Qwen3-VL N1200 Tm160 must preserve the declared baseline workspace");
        }
        if (compact || ctx.attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            // Vt is [B, local_heads, head_dim, align16(N/B)] per core. The
            // bounded B2 route has aligned N/B=256, so total rows suffice. It
            // overlaps neither V nor Q/K/output during attention (phase 3).
            sdpa_tmp = A(CeilDiv(ctx.max_kv_seq_len, int64_t{16}) * 16 *
                         local_q_dim * DWIDTH);
        }

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
        const bool ordinary_resident = compact || ordinary_w8_resident_chunk(
            comp_cs, ctx.max_kv_seq_len);
        const BufferScope qkv_scope =
            qwen3vl_2b_delivery_profile_active() || ordinary_resident
            ? COMP
            : KVIN;
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
            {"q",            q_kv,      2, 3, StorageClass::Temp, 0, nullptr, qkv_scope},
            {"k",            q_kv,      2, 3, StorageClass::Temp, 0, nullptr, qkv_scope},
            {"v",            q_kv,      2, 3, StorageClass::Temp, 0, nullptr, qkv_scope},
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
        if (ordinary_resident) {
            // The sequential body feeds attention from q directly. Keep the
            // remaining declarations in their original order for other routes.
            decls.erase(std::remove_if(decls.begin(), decls.end(),
                [](const BufferDecl& decl) { return std::string(decl.name) == "q_comp"; }),
                decls.end());
        }
        if (compact) {
            // Scalar chunk enumeration precedes the physical domain and uses
            // an unbound DDR-default context. Account for the only admitted
            // compact route there; a descriptor-bound DDR layout stays invalid.
            const bool unbound_probe = ctx.stage_plan_fingerprint == 0 &&
                ctx.rope_table_residency == FmbRopeTableResidency::UNSPECIFIED;
            TORCH_CHECK((ctx.attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA ||
                         unbound_probe) &&
                            kv_cs == 1200 && !ctx.is_causal && !ctx.use_attn_mask &&
                            ctx.batch_size == 1,
                        "Qwen3-VL compact buffers require one complete raw-SPM encoder chunk");
            // A=residual1 stays raw. B=input_norm is first the normalized input,
            // then the O/fc2 partial. Both rings read B + compact R and write A.
            // fc1 is independent; phases 5..6 reuse dead attention scratch.
            for (auto& decl : decls) {
                if (std::string(decl.name) == "oproj") {
                    decl.alias_of = "input_norm";
                    decl.scope = ALL;
                }
            }
            decls.push_back({"compact_residual", A(1200 * h / num_cores() * DWIDTH),
                             1, 6, StorageClass::Temp, 0, nullptr, ALL});
        }
        // The selected descriptor owns residency. Ordinary GraphCache routes
        // use LayerWide Temp so one handle may rebuild DDR <-> SPM plans; a
        // native composite keeps the tables Persistent across its outer Graph.
        if (rope_spm_layout && max_hw_ > 0) {
            int64_t rope_tbl_sz = A(max_hw_ * (orig_head_dim_ / 4) * DWIDTH);
            const StorageClass storage =
                rope_tables_use_persistent_storage()
                ? StorageClass::Persistent
                : StorageClass::Temp;
            const int phase_end = storage == StorageClass::Temp ? 6 : 0;
            decls.push_back({"rope_cos", rope_tbl_sz, 0, phase_end,
                             storage, 0, nullptr});
            decls.push_back({"rope_sin", rope_tbl_sz, 0, phase_end,
                             storage, 0, nullptr});
        }
        // Fused merger (patch + 3 deepstack), single- AND batched-image. The merger is
        // row-independent (LayerNorm/GEMM/GELU/all-reduce per-row), so one pass over the packed
        // residual1[seq,h] (seq = current_num_patches_ = g*n_i) is numerically identical to the
        // per-image loop. ALL activation buffers ALIAS dead encoder Temps → the merger adds 0
        // Fixed / 0 Temp peak, AND (critically for batching) NO activation buffer is Persistent:
        // the persistent layout is g- AND cs-INDEPENDENT, so a shared handle can run cs=256 (g=1)
        // and cs=768 (g=3) forwards without tripping the persistent-layout lock (the SPM-overflow
        // + monotonic-cs blocker that the old all-Persistent layout hit). Mirrors qwen25vl
        // merger layout; the deepstack snapshot stage `merger_in` (qwen3vl-only, qwen25vl has no
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
            const int64_t local_mh = merge_hidden_ / num_cores();
            const int64_t oh       = merger_out_hidden_;
            const int64_t merger_norm_w_sz = dma_safe(
                deepstack_merger_active() ? merge_hidden_ : h);
            if (fp16_chunked_merger_active()) {
                // All encoder/tap values cross this boundary through DDR.
                // Separate post-loop scratch can reuse the encoder arena; no
                // new Persistent slots may appear/disappear with image shape.
                // Keep every merger operand live for the entire post_fn, so
                // neither parameters nor ring operands can accidentally alias.
                constexpr auto POST = BufferScope::OutsideLayerLoop;
                decls.insert(decls.end(), {
                    {"merger_normed", A(cs * h * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_mid", A(mrows * local_mh * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_zero", A(mrows * oh * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_out", A(mrows * oh * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_in", A(cs * h * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_ln_q_w", merger_norm_w_sz, 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_ln_q_b", merger_norm_w_sz, 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_m0_bias", dma_safe(local_mh), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_m2_bias", dma_safe(oh), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                });
                return decls;
            }
            if (compact && w8_compact_encoder_active()) {
                // Keep W8's four parameter slots Persistent for every shape.
                // Only these activations cross the post-loop boundary via DDR;
                // encoder aliases would make merger partial and zero overlap B.
                constexpr auto POST = BufferScope::OutsideLayerLoop;
                decls.insert(decls.end(), {
                    {"merger_normed", A(cs * h * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_mid", A(mrows * local_mh * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_zero", A(mrows * oh * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_out", A(mrows * oh * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                    {"merger_in", A(cs * h * DWIDTH), 0, 0, StorageClass::Temp, 0, nullptr, POST},
                });
            } else if (!adaptive_w8_merger() || cs >= current_num_patches_ ||
                (image_batch_count_ == 1 && !prefix_scatter_active_ &&
                 ordinary_w8_resident_profile())) {
                // Each aliased buffer's content must fit its target's allocated size (the alias
                // resolver copies the offset, it does NOT bound the size). h=1024/oh=2048/mh=4096:
                // normed,in = cs*h <= oproj,input_norm (res); mid = mrows*512 <= fc1 (cs*512);
                // zero,out = mrows*oh <= input_norm,residual1 (res).
                TORCH_CHECK(A(cs * h * DWIDTH) <= oproj && A(cs * h * DWIDTH) <= res &&
                            A(mrows * oh * DWIDTH) <= oproj && A(mrows * local_mh * DWIDTH) <= fc1
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
            }
            // Cold merger residency is identical for single- and multi-chunk
            // plans. Chunked ordinary W8 mergers reuse these same weight slots.
            decls.push_back({"merger_ln_q_w",  merger_norm_w_sz,   0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"merger_ln_q_b",  merger_norm_w_sz,   0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"merger_m0_bias", dma_safe(local_mh), 0, 0, StorageClass::Persistent, 0, nullptr});
            decls.push_back({"merger_m2_bias", dma_safe(oh),       0, 0, StorageClass::Persistent, 0, nullptr});
        }
        return decls;
    }

    void emit_preload_weights() {
        int64_t h = hidden_size();
        int64_t local_q_dim = (num_q_heads() / num_cores()) * head_dim();
        int64_t local_inter = intermediate_size() / num_cores();

        // Loop 1: owner-core DMAs for all layers (zero + norm + partitioned bias).
        for (int64_t L = 0; L < num_layers(); ++L) {
            auto& bn = layer_bias_norm_[L];

            zero_owner_spm(layer_addr(L, 0, "o_bias"), h);
            zero_owner_spm(layer_addr(L, 0, "fc2_bias"), h);

            if (pipeline_dispatch_active()) {
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln1_w, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "ln1_gamma"), num_cores());
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln1_b, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "ln1_beta"), num_cores());
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln2_w, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "ln2_gamma"), num_cores());
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln2_b, /*src_offset_elements=*/0, h,
                    layer_addr(L, 0, "ln2_beta"), num_cores());

                rpu_launch_ddr_scatter_spm_dma(
                    bn.q_b, /*src_offset_elements=*/0,
                    local_q_dim, local_q_dim * DWIDTH,
                    layer_addr(L, 0, "q_bias"),
                    /*num_cores=*/num_cores());
                rpu_launch_ddr_scatter_spm_dma(
                    bn.k_b, /*src_offset_elements=*/0,
                    local_q_dim, local_q_dim * DWIDTH,
                    layer_addr(L, 0, "k_bias"),
                    /*num_cores=*/num_cores());
                rpu_launch_ddr_scatter_spm_dma(
                    bn.v_b, /*src_offset_elements=*/0,
                    local_q_dim, local_q_dim * DWIDTH,
                    layer_addr(L, 0, "v_bias"),
                    /*num_cores=*/num_cores());
                rpu_launch_ddr_scatter_spm_dma(
                    bn.fc1_b, /*src_offset_elements=*/0,
                    local_inter, local_inter * DWIDTH,
                    layer_addr(L, 0, "fc1_bias"),
                    /*num_cores=*/num_cores());
            } else {
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln1_w.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "ln1_gamma"), num_cores());
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln1_b.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "ln1_beta"), num_cores());
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln2_w.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "ln2_gamma"), num_cores());
                rpu_launch_ddr_broadcast_spm_dma(
                    bn.ln2_b.data_ptr<c10::Half>(), h,
                    layer_addr(L, 0, "ln2_beta"), num_cores());

                rpu_launch_ddr_scatter_spm_dma(
                    bn.q_b.data_ptr<c10::Half>(), local_q_dim,
                    local_q_dim * DWIDTH, layer_addr(L, 0, "q_bias"),
                    /*num_cores=*/num_cores());
                rpu_launch_ddr_scatter_spm_dma(
                    bn.k_b.data_ptr<c10::Half>(), local_q_dim,
                    local_q_dim * DWIDTH, layer_addr(L, 0, "k_bias"),
                    /*num_cores=*/num_cores());
                rpu_launch_ddr_scatter_spm_dma(
                    bn.v_b.data_ptr<c10::Half>(), local_q_dim,
                    local_q_dim * DWIDTH, layer_addr(L, 0, "v_bias"),
                    /*num_cores=*/num_cores());
                rpu_launch_ddr_scatter_spm_dma(
                    bn.fc1_b.data_ptr<c10::Half>(), local_inter,
                    local_inter * DWIDTH, layer_addr(L, 0, "fc1_bias"),
                    /*num_cores=*/num_cores());
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

        if (rope_spm_enabled_ && rope_tables_use_persistent_storage()) {
            emit_rope_table_preload();
        }

        // Merger weights (patch + deepstack) are loaded per-merger INSIDE merger_post_fn (shared
        // SPM weight slots, re-loaded before each merger's compute) — not here — so the per-merger
        // ln_q/biases don't clobber each other and post_fn re-emit on fast-replay stays correct.
    }

    void emit_rope_table_preload() {
        TORCH_INTERNAL_ASSERT(rope_spm_enabled_ && max_hw_ > 0);
        const int64_t tbl_elems = max_hw_ * (orig_head_dim_ / 4);
        rpu_ddr_flush(freq_cos_.data_ptr<c10::Half>());
        rpu_ddr_flush(freq_sin_.data_ptr<c10::Half>());
        if (pipeline_dispatch_active()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN3VL_ROPE_COS_PRELOAD_PIPELINE_SITE,
                static_cast<int64_t>(
                    Qwen3VLVisionMutableDmaRoute::ROPE_TABLE_DDR_TO_SPM),
                /*resolved_flags=*/0,
                {1, max_hw_, orig_head_dim_});
            rpu_launch_ddr_broadcast_spm_dma(
                freq_cos_, /*src_offset_elements=*/0, tbl_elems,
                addr(0, "rope_cos"), num_cores());
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN3VL_ROPE_SIN_PRELOAD_PIPELINE_SITE,
                static_cast<int64_t>(
                    Qwen3VLVisionMutableDmaRoute::ROPE_TABLE_DDR_TO_SPM),
                /*resolved_flags=*/0,
                {1, max_hw_, orig_head_dim_});
            rpu_launch_ddr_broadcast_spm_dma(
                freq_sin_, /*src_offset_elements=*/0, tbl_elems,
                addr(0, "rope_sin"), num_cores());
            return;
        }
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN3VL_ROPE_COS_PRELOAD_DIRECT_SITE,
            static_cast<int64_t>(
                Qwen3VLVisionMutableDmaRoute::ROPE_TABLE_DDR_TO_SPM),
            /*resolved_flags=*/0,
            {1, max_hw_, orig_head_dim_});
        rpu_launch_ddr_broadcast_spm_dma(
            freq_cos_.data_ptr<c10::Half>(), tbl_elems,
            addr(0, "rope_cos"), num_cores());
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN3VL_ROPE_SIN_PRELOAD_DIRECT_SITE,
            static_cast<int64_t>(
                Qwen3VLVisionMutableDmaRoute::ROPE_TABLE_DDR_TO_SPM),
            /*resolved_flags=*/0,
            {1, max_hw_, orig_head_dim_});
        rpu_launch_ddr_broadcast_spm_dma(
            freq_sin_.data_ptr<c10::Half>(), tbl_elems,
            addr(0, "rope_sin"), num_cores());
    }

    void emit_kv_first_body(int layer_idx, const ChunkInfo& chunk) {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        // Phase 1: DDR→SPM input DMA + LayerNorm1
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }
        if (compact_encoder_active()) emit_fp16_compact_residual_copy(0);
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "ln1_gamma"), layer_addr(layer_idx, 0, "ln1_beta"),
            seq_len, h, eps_, false, 0, num_cores());

        // Phase 2: Q / K / V Linear with bias (SPM→SPM ACC16, col-partition).
        // Python adapter pre-splits the fused [3*dim, dim] HF qkv into three
        // [dim, dim] tensors — we use the SigLIP three-Linear pattern.
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN3VL_Q_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, num_cores(),
            layer_addr(layer_idx, 0, "q_bias"), lw.q_ws,
            /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
            /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN3VL_K_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, num_cores(),
            layer_addr(layer_idx, 0, "k_bias"), lw.k_ws,
            /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
            /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN3VL_V_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, num_cores(),
            layer_addr(layer_idx, 0, "v_bias"), lw.v_ws,
            /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
            /*force_acc32=*/linear_acc32_);

        // Phase 2.5: 2D RoPE on Q and K (in-place SPM read/write).
        //
        // Each core holds nq/num_cores() = 2 heads worth of `[seq_len, 2, head_dim]`
        // in its slice. The rope_2d kernel takes head_num = local_heads_per_core
        // (because each core processes its own slice independently — same pattern
        // as 1D rope on Q/K of the text decoder).
        const int64_t local_heads = nq / num_cores();
        // `hd` is the physical per-head stride. RoPE rotates only the logical
        // lanes; the controlled 32B pad lanes remain zero.
        const int64_t head_dim_pad = hd;
        const std::vector<int64_t> rope_route_arguments{
            rope_spm_enabled_ ? 1 : 0,
            pipeline_dispatch_active() ? 1 : 0,
            chunk.offset, seq_len, local_heads,
            orig_head_dim_, head_dim_pad, num_cores()};
        if (rope_spm_enabled_) {
            // Descriptor-selected SPM-resident tables. Native composites own
            // a persistent preload; ordinary Graph execution uploads its Temp
            // table slots in the pre-layer hook.
            const uint32_t cos_a = addr(0, "rope_cos");
            const uint32_t sin_a = addr(0, "rope_sin");
            if (pipeline_dispatch_active()) {
                consume_manifest_route(
                    FmbRouteFamily::ROPE,
                    QWEN3VL_Q_ROPE_SPM_PIPELINE_SITE,
                    static_cast<int64_t>(
                        rope_spm_enabled_
                            ? Qwen3VLVisionRopeRoute::ROPE_2D_SPM
                            : Qwen3VLVisionRopeRoute::ROPE_2D_DDR),
                    chunk.idx,
                    rope_route_arguments,
                    FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                rpu_launch_rope_2d_spm_kernel(
                    addr(0, "q"), addr(0, "q"), cos_a, sin_a,
                    position_idx_keepalive_,
                    /*pos_offset=*/chunk.offset, seq_len,
                    local_heads, orig_head_dim_, head_dim_pad, num_cores());
                consume_manifest_route(
                    FmbRouteFamily::ROPE,
                    QWEN3VL_K_ROPE_SPM_PIPELINE_SITE,
                    static_cast<int64_t>(
                        Qwen3VLVisionRopeRoute::ROPE_2D_SPM), chunk.idx,
                    rope_route_arguments,
                    FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                rpu_launch_rope_2d_spm_kernel(
                    addr(0, "k"), addr(0, "k"), cos_a, sin_a,
                    position_idx_keepalive_,
                    /*pos_offset=*/chunk.offset, seq_len,
                    local_heads, orig_head_dim_, head_dim_pad, num_cores());
            } else {
                consume_manifest_route(
                    FmbRouteFamily::ROPE, QWEN3VL_Q_ROPE_SPM_SITE,
                    static_cast<int64_t>(
                        Qwen3VLVisionRopeRoute::ROPE_2D_SPM), chunk.idx,
                    rope_route_arguments,
                    FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                rpu_launch_rope_2d_spm_kernel(
                    addr(0, "q"), addr(0, "q"), cos_a, sin_a,
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    /*pos_offset=*/chunk.offset, seq_len,
                    local_heads, orig_head_dim_, head_dim_pad, num_cores());
                consume_manifest_route(
                    FmbRouteFamily::ROPE, QWEN3VL_K_ROPE_SPM_SITE,
                    static_cast<int64_t>(
                        Qwen3VLVisionRopeRoute::ROPE_2D_SPM), chunk.idx,
                    rope_route_arguments,
                    FMB_ROUTE_FLAG_ROPE_TABLE_SPM);
                rpu_launch_rope_2d_spm_kernel(
                    addr(0, "k"), addr(0, "k"), cos_a, sin_a,
                    position_idx_keepalive_.data_ptr<int16_t>(),
                    /*pos_offset=*/chunk.offset, seq_len,
                    local_heads, orig_head_dim_, head_dim_pad, num_cores());
            }
        } else {
            consume_manifest_route(
                FmbRouteFamily::ROPE, QWEN3VL_Q_ROPE_DDR_SITE,
                static_cast<int64_t>(
                    Qwen3VLVisionRopeRoute::ROPE_2D_DDR), chunk.idx,
                rope_route_arguments,
                FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
            rpu_launch_rope_2d_ddr_kernel(
                addr(0, "q"), addr(0, "q"),
                freq_cos_.data_ptr<c10::Half>(),
                freq_sin_.data_ptr<c10::Half>(),
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset,
                seq_len,
                local_heads, orig_head_dim_, head_dim_pad, num_cores());
            consume_manifest_route(
                FmbRouteFamily::ROPE, QWEN3VL_K_ROPE_DDR_SITE,
                static_cast<int64_t>(
                    Qwen3VLVisionRopeRoute::ROPE_2D_DDR), chunk.idx,
                rope_route_arguments,
                FMB_ROUTE_FLAG_ROPE_TABLE_DDR);
            rpu_launch_rope_2d_ddr_kernel(
                addr(0, "k"), addr(0, "k"),
                freq_cos_.data_ptr<c10::Half>(),
                freq_sin_.data_ptr<c10::Half>(),
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/chunk.offset,
                seq_len,
                local_heads, orig_head_dim_, head_dim_pad, num_cores());
        }

        // The COMPLETE raw-SPM route consumes Q/K/V in place. It has no
        // KV_INSERT site and must not read or write the DDR cache tensors.
        if (ctx().attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            return;
        }

        // Phase 3: KV cache insert (position=0 always for vision) + bidir SDPA.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const KvInsertSegmentPlan kv_plan = [&] {
            if (!ctx().has_complete_physical_manifest()) {
                return rpu_resolve_kvinsert_segment_plan_auto(
                    chunk.offset, seq_len, seq_len, num_cores(), nq, hd,
                    QWEN3VL_KV_CAPABILITIES);
            }
            const FmbRouteManifestEntry& route = ctx().find_physical_route(
                FmbRouteFamily::KV_INSERT, QWEN3VL_KV_INSERT_SITE,
                chunk.idx);
            const KvInsertSegmentPlan resolved_plan = restore_kvinsert_plan(
                QWEN3VL_KV_INSERT_SITE, route.arguments, num_cores(), nq, hd);
            TORCH_CHECK(
                resolved_plan.logical_rows() == seq_len &&
                    resolved_plan.physical_rows() == seq_len &&
                    resolved_plan.segment(0).position == chunk.offset,
                "Qwen3-VL Vision KV descriptor geometry drift at invocation ",
                chunk.idx);
            ctx().consume_physical_route(
                FmbRouteFamily::KV_INSERT, QWEN3VL_KV_INSERT_SITE,
                static_cast<int64_t>(resolved_plan.route()),
                QWEN3VL_KV_REASON_DDR_REQUIRED, route.arguments, chunk.idx);
            return resolved_plan;
        }();
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache,
            addr_offset("k").value, addr_offset("v").value,
            nq, hd, num_cores(),
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            /*spm_rows=*/0, kv_plan);

        if (ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL) {
            return;
        }

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
                /*num_cores=*/num_cores());
        } else {
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "q"),
                q_slot.data_ptr<c10::Half>() + chunk.offset * q_row_stride,
                q_local_elems, q_core_stride_bytes,
                /*num_cores=*/num_cores());
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
        int64_t local_inter = is_ / num_cores();
        const int64_t local_heads = nq / num_cores();

        const bool delivery = qwen3vl_2b_delivery_profile_active();
        const bool compact = compact_encoder_active();
        const bool sequential = ctx().stage_plan.chunk_mode == ChunkMode::SEQUENTIAL;
        TORCH_CHECK(!sequential || delivery || compact || ordinary_w8_resident_chunk(
                        chunk.len, current_num_patches_),
                    "Qwen3-VL Vision sequential body requires its admitted resident profile");
        if (sequential && !delivery && layer_idx == 0 && chunk.idx == 0) {
            if (compact) {
                ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                    compact_encoder_schedule_site(), 1, 0,
                    compact_encoder_schedule_arguments());
            } else {
                consume_ordinary_w8_resident_schedule();
            }
        }
        if (delivery && layer_idx == 0 && chunk.idx == 0) {
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                QWEN3VL_DELIVERY_VISION_SCHEDULE_SITE, 1, 0,
                {current_num_patches_, 24, 1024, 4096, 16, 64, 1});
        }
        if (sequential) {
            emit_kv_first_body(layer_idx, chunk);
        } else {
            emit_layer_input_dma(layer_idx, chunk);

            TORCH_CHECK(q_ddr_slots_.count({current_num_patches_, local_heads * hd}) == 1,
                        "Qwen3VLVisionModel: Q staging slot not allocated in KV_FIRST mode");
            const at::Tensor& q_slot = q_ddr_slot();
            const int64_t q_row_stride = local_heads * hd;
            const int64_t q_local_elems = seq_len * q_row_stride;
            const int64_t q_core_stride_bytes =
                q_slot.size(1) * q_row_stride * DWIDTH;
            if (pipeline_dispatch_active()) {
                rpu_launch_ddr_scatter_spm_dma(
                    q_slot, /*src_offset_elements=*/chunk.offset * q_row_stride,
                    q_local_elems, q_core_stride_bytes,
                    addr(0, "q_comp"),
                    /*num_cores=*/num_cores());
            } else {
                rpu_launch_ddr_scatter_spm_dma(
                    q_slot.data_ptr<c10::Half>() + chunk.offset * q_row_stride,
                    q_local_elems, q_core_stride_bytes,
                    addr(0, "q_comp"),
                    /*num_cores=*/num_cores());
            }
        }

        double attn_scale = 1.0 / std::sqrt(static_cast<double>(orig_head_dim_));
        if (ctx().attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            const int64_t image_rows = seq_len / image_batch_count_;
            const int64_t query_rows = vision_raw_query_chunk_size(image_rows);
            for (int64_t query_offset = 0; query_offset < image_rows;
                 query_offset += query_rows) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION, QWEN3VL_RAW_SPM_ATTN_SITE,
                    static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA),
                    /*resolved_flags=*/0, vision_raw_attention_arguments(seq_len, query_offset),
                    query_offset / query_rows);
                if (query_offset == 0) {
                    rpu_launch_v_transpose_spm(
                        addr(0, "v"), addr(0, "sdpa_tmp"),
                        image_batch_count_, image_rows, nq, hd, num_cores());
                }
                const uint32_t query_byte_offset = static_cast<uint32_t>(
                    query_offset * local_heads * hd * DWIDTH);
                // K and Vt always cover the complete image. Only Q/output
                // advance; this is one resident encoder chunk, not two images.
                rpu_launch_sdpa_by_mha_spm(
                    addr(0, "q") + query_byte_offset, addr(0, "k"), addr(0, "sdpa_tmp"),
                    addr(0, "sdpa_out") + query_byte_offset, /*mask_spm=*/0, /*MASK_NONE=*/0,
                    attn_scale, image_batch_count_, std::min(query_rows, image_rows - query_offset),
                    image_rows, nq, nq, hd, num_cores());
            }
        } else if (image_batch_count_ > 1) {
            auto& k_cache = (*ctx().k_caches)[layer_idx];
            auto& v_cache = (*ctx().v_caches)[layer_idx];
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
            // SigLIP-batch path (rpu_siglip_model.cpp:888).
            int64_t per_image_ctx = current_num_patches_ / image_batch_count_;
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION,
                    QWEN3VL_MINIBATCH_ATTN_SITE,
                    static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                    /*resolved_flags=*/0,
                    {image_batch_count_, per_image_ctx, seq_len, nq, hd,
                     num_cores()},
                    chunk.idx);
            }
            rpu_launch_sdpa_spm_minibatch_kernel(
                k_cache, v_cache,
                0 /*MASK_NONE*/, attn_scale,
                addr_offset(sequential ? "q" : "q_comp").value,
                addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value, 0,
                seq_len, nq, nq, hd,
                per_image_ctx, image_batch_count_,
                num_cores(), 8);
        } else {
            auto& k_cache = (*ctx().k_caches)[layer_idx];
            auto& v_cache = (*ctx().v_caches)[layer_idx];
            bool n1200_tm160 = false;
            if (ctx().has_complete_physical_manifest()) {
                const auto& manifest = ctx().physical_manifest();
                n1200_tm160 = std::any_of(manifest.routes.begin(), manifest.routes.end(),
                    [&](const auto& route) {
                        return route.family == FmbRouteFamily::ATTENTION &&
                            route.site_id == QWEN3VL_N1200_TM160_ATTN_SITE &&
                            route.invocation == chunk.idx;
                    });
            }
            if (n1200_tm160) {
                const auto& manifest = ctx().physical_manifest();
                TORCH_CHECK(!ctx().is_causal && !ctx().attention_mask.has_value() &&
                                n1200_tm160_eligible(ctx().stage_plan,
                                    manifest.physical_length, manifest.logical_length,
                                    ctx().position),
                            "Qwen3-VL N1200 Tm160 descriptor exceeds its owner capability");
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION, QWEN3VL_N1200_TM160_ATTN_SITE,
                    static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV), 0,
                    n1200_tm160_arguments(chunk), chunk.idx);
                rpu_launch_sdpa_qwen3vl_n1200_tm160(
                    k_cache, v_cache, addr_offset("q_comp").value,
                    addr_offset("sdpa_out").value, addr_offset("sdpa_tmp").value,
                    seq_len, QWEN3VL_N1200_SDPA_TMP_BYTES);
            } else {
                if (ctx().has_complete_physical_manifest()) {
                    ctx().consume_physical_route(
                        FmbRouteFamily::ATTENTION, QWEN3VL_UNIFIED_ATTN_SITE,
                        static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV), 0,
                        vision_attention_arguments(current_num_patches_, seq_len), chunk.idx);
                }
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    k_cache, v_cache,
                    0 /*MASK_NONE*/, attn_scale,
                    addr_offset(sequential ? "q" : "q_comp").value,
                    addr_offset("sdpa_out").value,
                    addr_offset("sdpa_tmp").value, 0,
                    seq_len, nq, nq, hd,
                    current_num_patches_, num_cores(), 8,
                    /*cache_batch_offset_elems=*/0, /*use_16b=*/false,
                    /*qwen3vl_tm160=*/delivery);
            }
        }

        // Phase 4: O_proj (row-partition, with bias) + AllReduce + Residual.
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN3VL_O_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, num_cores(),
            layer_addr(layer_idx, 0, "o_bias"), lw.o_ws,
            /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
            /*force_acc32=*/linear_acc32_);
        if (compact) {
            emit_fp16_compact_attention_ring();
            emit_fp16_compact_residual_copy(1);
        } else {
            consume_manifest_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN3VL_ATTN_ALL_REDUCE_SITE,
                vision_ring_route(seq_len, h), chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual1"),
                addr(0, "input_norm"), seq_len, h,
                num_cores(), num_cores());
        }

        // Phase 5: LayerNorm2 + fc1+bias + GELU (gelu_pytorch_tanh per Q3VL config).
        rpu_launch_layernorm_spm_kernel(
            addr(0, compact ? "residual1" : "input_norm"), addr(0, "oproj"),
            layer_addr(layer_idx, 0, "ln2_gamma"), layer_addr(layer_idx, 0, "ln2_beta"),
            seq_len, h, eps_, false, 0, num_cores());
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN3VL_FC1_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "oproj"), lw.fc1_w, addr(0, "fc1"),
            seq_len, is_, h, 1, num_cores(),
            layer_addr(layer_idx, 0, "fc1_bias"), lw.fc1_ws,
            /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
            /*force_acc32=*/linear_acc32_);
        if (rhinovla_high_precision_) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, RHINOVLA_VISION_HIGH_ENCODER_ACTIVATION_SITE,
                1, 0, {2, 1024, 4096, 2048, 4, 8});
        }
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "fc1"), addr(0, "fc1"),
            seq_len * local_inter, ValuOpType::ADD,
            GeluMode::TANH, num_cores(), rhinovla_high_precision_
                ? RpuUnaryPrecision::HIGH : RpuUnaryPrecision::BASE);

        // Phase 6: fc2 + AllReduce + Residual (writes back to residual1 in
        // SPM_RESIDENT mode so the next layer's Phase 1 LN1 reads directly).
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN3VL_FC2_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "fc1"), lw.fc2_w, addr(0, "oproj"),
            seq_len, h, is_, 0, num_cores(),
            layer_addr(layer_idx, 0, "fc2_bias"), lw.fc2_ws,
            /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
            /*force_acc32=*/linear_acc32_);
        if (compact) {
            emit_fp16_compact_mlp_ring();
        } else {
            consume_manifest_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN3VL_MLP_ALL_REDUCE_SITE,
                vision_ring_route(seq_len, h), chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "input_norm"),
                addr(0, "residual1"), seq_len, h,
                num_cores(), num_cores());
        }

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
                    /*postshuffle_norm=*/true,
                    /*invocation=*/static_cast<int64_t>(ordinal) + 1);
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
                    /*postshuffle_norm=*/true,
                    /*invocation=*/static_cast<int64_t>(ordinal) + 1);
            }
        }

        // GAP-B7: DeepStack snapshot — DMA `residual1` (post-block hidden
        // state) to DDR so the Python adapter can run its merger after
        // forward(). Slots are pre-allocated in forward() entry to avoid
        // `at::empty` competing with the active graph's queue-buffer reserve.
        const auto deepstack_it = std::find(
            deepstack_visual_indexes_.begin(),
            deepstack_visual_indexes_.end(),
            static_cast<int64_t>(layer_idx));
        if (!pipeline_dispatch_active() &&
            deepstack_it != deepstack_visual_indexes_.end())
        {
            auto key = std::make_pair(static_cast<int64_t>(layer_idx), current_num_patches_);
            at::Tensor* slot = deepstack_ddr_slots_.find(key);
            TORCH_CHECK(slot != nullptr && slot->defined(),
                        "qwen3vl_vision_build_layer_subgraph: missing pre-allocated "
                        "DeepStack slot for layer_idx=", layer_idx,
                        " num_patches=", current_num_patches_,
                        " — forward() should have allocated it");
            const int64_t num_compute_chunks = static_cast<int64_t>(
                ctx().stage_plan.compute.chunks.size());
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN3VL_DEEPSTACK_SNAPSHOT_SITE,
                static_cast<int64_t>(
                    Qwen3VLVisionMutableDmaRoute::SPM_COPY_TO_DDR),
                /*resolved_flags=*/0,
                {static_cast<int64_t>(deepstack_visual_indexes_.size()),
                 layer_idx, chunk.offset, seq_len, h},
                layer_idx * num_compute_chunks + chunk.idx);
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

    bool ordinary_fp16_chunked_merger_profile() const {
        if (num_cores() != 8 || image_batch_count_ != 1 ||
            num_layers() != 24 || hidden_size() != 1024 ||
            intermediate_size() != 4096 || num_q_heads() != 16 ||
            head_dim() != 64 || orig_head_dim_ != 64 || !linear_acc32_ ||
            delivery_compatibility_ || prefix_scatter_active_ ||
            pipeline_dispatch_active() || native_composite_prepare_active_ ||
            z1_prepared_num_patches_ != 0 || z1_adopted_ || z1_bound_ ||
            multiview_dry_prepared_ || multiview_composite_prepared_ ||
            !fused_merger_enabled_ || merge_hidden_ != 4096 ||
            (merger_out_hidden_ != 2048 && merger_out_hidden_ != 2560) ||
            deepstack_visual_indexes_ != std::vector<int64_t>{5, 11, 17} ||
            layer_weights_.size() != 24 || ds_merger_m0_w_.size() != 3 ||
            ds_merger_m2_w_.size() != 3 ||
            merger_m0_w_.scalar_type() != at::kHalf ||
            merger_m2_w_.scalar_type() != at::kHalf) return false;
        for (const auto* weights : {&ds_merger_m0_w_, &ds_merger_m2_w_}) {
            for (const auto& weight : *weights) {
                if (weight.scalar_type() != at::kHalf) return false;
            }
        }
        return std::all_of(layer_weights_.begin(), layer_weights_.end(),
            [](const auto& weights) {
                for (const at::Tensor* weight : {
                        &weights.q_w, &weights.k_w, &weights.v_w,
                        &weights.o_w, &weights.fc1_w, &weights.fc2_w}) {
                    if (weight->scalar_type() != at::kHalf) return false;
                }
                return true;
            });
    }

    bool fp16_chunked_merger_active() const {
        return fp16_chunked_merger_enabled_ && current_num_patches_ == 1200 &&
            (configured_chunk_size_ == 0 || configured_chunk_size_ == 608) &&
            ordinary_fp16_chunked_merger_profile();
    }

    bool fp16_compact_encoder_active() const {
        return fp16_compact_encoder_enabled_ && configured_chunk_size_ == 0 &&
            fp16_chunked_merger_active();
    }

    bool ordinary_w8_compact_encoder_profile() const {
        if (!ordinary_n1200_tm160_profile() || fp16_chunked_merger_enabled_ ||
            merge_hidden_ != 4096 || ds_merger_m0_w_.size() != 3 ||
            ds_merger_m2_w_.size() != 3 || ds_merger_m0_scale_.size() != 3 ||
            ds_merger_m2_scale_.size() != 3) return false;
        // Registration already checks finite positive scale values. Recheck
        // every retained projection's metadata without reading DDR in forward.
        auto w8_projection = [](const at::Tensor& weight, const at::Tensor& scale,
                                int64_t rows) {
            return weight.defined() && weight.scalar_type() == at::kChar &&
                weight.device().type() == at::kPrivateUse1 && weight.is_contiguous() &&
                weight.dim() == 2 && weight.size(0) == rows && weight.size(1) == 4096 &&
                scale.defined() && scale.scalar_type() == at::kHalf &&
                scale.device().type() == at::kPrivateUse1 && scale.is_contiguous() &&
                scale.dim() == 1 && scale.numel() == rows;
        };
        if (!w8_projection(merger_m0_w_, merger_m0_scale_, 4096) ||
            !w8_projection(merger_m2_w_, merger_m2_scale_, merger_out_hidden_)) return false;
        for (size_t i = 0; i < 3; ++i) {
            if (!w8_projection(ds_merger_m0_w_[i], ds_merger_m0_scale_[i], 4096) ||
                !w8_projection(ds_merger_m2_w_[i], ds_merger_m2_scale_[i], merger_out_hidden_))
                return false;
        }
        return true;
    }

    bool w8_compact_encoder_active() const {
        return w8_compact_encoder_enabled_ && configured_chunk_size_ == 0 &&
            current_num_patches_ == 1200 && ordinary_w8_compact_encoder_profile();
    }

    bool compact_encoder_active() const {
        return fp16_compact_encoder_active() || w8_compact_encoder_active();
    }

    bool compact_encoder_chunk(int64_t chunk_size, int64_t sequence_length) const {
        return compact_encoder_active() && chunk_size == 1200 &&
            sequence_length == 1200 &&
            sdpa_by_mha_spm_is_valid(1, 608, 1200, 16, 16, 64, 8, 0) &&
            sdpa_by_mha_spm_is_valid(1, 592, 1200, 16, 16, 64, 8, 0);
    }

    bool compact_encoder_plan(const FmbThreeStageChunkPlan& plan,
                              int64_t physical_len, int64_t logical_len,
                              int64_t position) const {
        if (!compact_encoder_chunk(plan.compute.plan.chunk_size, physical_len) ||
            logical_len != 1200 || position != 0 ||
            plan.chunk_mode != ChunkMode::SEQUENTIAL) return false;
        for (const auto* stage : {&plan.input, &plan.qkv, &plan.compute}) {
            if (stage->plan.chunk_size != 1200 || stage->plan.num_chunks != 1 ||
                stage->chunks.size() != 1) return false;
            const auto& chunk = stage->chunks.front();
            if (chunk.idx != 0 || chunk.offset != 0 || chunk.len != 1200 ||
                chunk.kv_seq_len != 1200) return false;
        }
        return true;
    }

    bool fp16_compact_encoder_chunk(int64_t chunk_size, int64_t sequence_length) const {
        return fp16_compact_encoder_active() && compact_encoder_chunk(chunk_size, sequence_length);
    }

    bool fp16_compact_encoder_plan(const FmbThreeStageChunkPlan& plan,
                                   int64_t physical_len, int64_t logical_len,
                                   int64_t position) const {
        return fp16_compact_encoder_chunk(plan.compute.plan.chunk_size, physical_len) &&
            compact_encoder_plan(plan, physical_len, logical_len, position);
    }

    int64_t merger_compute_chunk_size(const FmbThreeStageChunkPlan& plan) const {
        return compact_encoder_active() ? 608 : plan.compute.plan.chunk_size;
    }

    std::vector<ChunkInfo> merger_compute_chunks(const FmbThreeStageChunkPlan& plan) const {
        // The post loop owns its own tiling; compact encoder C1200 must not
        // resize merger scratch or change the M152/M148 geometry.
        if (compact_encoder_active()) return {{0, 0, 608, 608}, {1, 608, 592, 1200}};
        return plan.compute.chunks;
    }

    std::vector<int64_t> fp16_compact_residual_arguments() const {
        // version, local-SPM residual mode, full matrix, cores, owner shard
        // elements/byte stride. N1200/H1024 divides the aligned ring shard.
        std::vector<int64_t> arguments{1, 2, 1200, 1024, 8, 153600, 307200};
        // The payload consumes FP16 activations for both profiles. Bind W8's
        // distinct weight/accumulator precision without changing FP16 receipts.
        if (w8_compact_encoder_active()) arguments.insert(arguments.end(), {8, 16});
        return arguments;
    }

    int64_t compact_encoder_schedule_site() const {
        return w8_compact_encoder_active() ? QWEN3VL_W8_COMPACT_SCHEDULE_SITE
                                          : QWEN3VL_FP16_COMPACT_SCHEDULE_SITE;
    }

    std::vector<int64_t> compact_encoder_schedule_arguments() const {
        std::vector<int64_t> arguments{
            1200, 1, 24, 1024, 4096, 16, 64, 8, merger_out_hidden_, 608, 592};
        if (w8_compact_encoder_active()) arguments.insert(arguments.end(), {8, 16});
        return arguments;
    }

    void emit_fp16_compact_residual_copy(int64_t invocation) {
        TORCH_CHECK(compact_encoder_active() && (invocation == 0 || invocation == 1),
                    "Qwen3-VL compact residual copy exceeds its exact owner capability");
        ctx().consume_physical_route(FmbRouteFamily::COLLECTIVE,
            QWEN3VL_FP16_COMPACT_COPY_SITE, 1, 0,
            fp16_compact_residual_arguments(), invocation);
        rpu_launch_spm_local_shard_copy_kernel(
            addr(0, "residual1"), 153600, 307200, addr(0, "compact_residual"), 8);
    }

    void emit_fp16_compact_attention_ring() {
        TORCH_CHECK(compact_encoder_active(),
                    "Qwen3-VL compact residual ring exceeds its exact owner capability");
        ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
            QWEN3VL_COMPACT_ATTN_ALL_REDUCE_SITE, vision_ring_route(1200, 1024), 0,
            fp16_compact_residual_arguments(), 0);
        rpu_launch_all_reduce_sum_residual_local_spm_kernel(
            addr(0, "oproj"), addr(0, "compact_residual"), addr(0, "residual1"),
            1200, 1024, 8, 8);
    }

    void emit_fp16_compact_mlp_ring() {
        TORCH_CHECK(compact_encoder_active(),
                    "Qwen3-VL compact residual ring exceeds its exact owner capability");
        ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
            QWEN3VL_COMPACT_MLP_ALL_REDUCE_SITE, vision_ring_route(1200, 1024), 0,
            fp16_compact_residual_arguments(), 0);
        rpu_launch_all_reduce_sum_residual_local_spm_kernel(
            addr(0, "oproj"), addr(0, "compact_residual"), addr(0, "residual1"),
            1200, 1024, 8, 8);
    }

    bool fp16_chunked_merger_plan(const FmbThreeStageChunkPlan& plan,
                                 int64_t physical_len, int64_t logical_len,
                                 int64_t position) const {
        if (fp16_compact_encoder_active()) {
            return fp16_compact_encoder_plan(plan, physical_len, logical_len, position);
        }
        if (!fp16_chunked_merger_active() || physical_len != 1200 ||
            logical_len != 1200 || position != 0 ||
            plan.chunk_mode != ChunkMode::KV_FIRST) return false;
        for (const auto* stage : {&plan.qkv, &plan.compute}) {
            if (stage->plan.chunk_size != 608 || stage->plan.num_chunks != 2 ||
                stage->chunks.size() != 2) return false;
            for (int64_t index = 0; index < 2; ++index) {
                const auto& chunk = stage->chunks[index];
                if (chunk.idx != index || chunk.offset != index * 608 ||
                    chunk.len != (index == 0 ? 608 : 592) ||
                    chunk.kv_seq_len != position + chunk.offset + chunk.len) return false;
            }
        }
        return true;
    }

    // The cold FP16 candidate is shape-local; registering its weights must not
    // replace eager mergers for other images. Legacy single-chunk fusion is
    // unchanged when that separate capability is disabled.
    bool merger_active() const {
        if (fp16_chunked_merger_enabled_) return fp16_chunked_merger_active();
        return fused_merger_enabled_ && merger_out_hidden_ > 0;
    }
    bool adaptive_w8_merger() const {
        return merger_active() && merger_m0_w_.scalar_type() == at::kChar;
    }
    bool pipeline_dispatch_active() const {
        return z1_dispatch_ || multiview_composite_dispatch_;
    }
    bool rope_tables_use_persistent_storage() const {
        return pipeline_dispatch_active() ||
            native_composite_prepare_active_ ||
            z1_prepared_num_patches_ != 0 ||
            multiview_dry_prepared_ ||
            multiview_composite_prepared_;
    }
    bool spm_rope_route_available() const {
        return has_rope_ &&
            rope_route_request_ != Qwen3VLVisionRopeRouteRequest::DDR;
    }
    std::vector<int64_t> vision_attention_arguments(int64_t image_rows, int64_t chunk_rows) const {
        std::vector<int64_t> arguments{image_batch_count_, image_rows, chunk_rows,
            num_q_heads(), head_dim(), num_cores()};
        if (qwen3vl_2b_delivery_profile_active()) arguments.push_back(160);
        return arguments;
    }
    int64_t vision_raw_query_chunk_size(int64_t image_rows) const {
        if (compact_encoder_active() && image_rows == 1200) return 608;
        return image_rows == 768 ? 384 : image_rows;
    }
    std::vector<int64_t> vision_raw_attention_arguments(
        int64_t rows, int64_t query_offset = 0) const {
        const int64_t image_rows = rows / image_batch_count_;
        const int64_t query_rows = std::min(
            vision_raw_query_chunk_size(image_rows), image_rows - query_offset);
        std::vector<int64_t> arguments{image_batch_count_, query_rows, image_rows, num_q_heads(), num_q_heads(),
                                      head_dim(), num_cores(), /*MASK_NONE=*/0, orig_head_dim_};
        // Preserve the existing full-query descriptors; the split route also
        // binds its actual query/output row offset for each invocation.
        if (image_rows == 768 || compact_encoder_active()) arguments.push_back(query_offset);
        return arguments;
    }
    bool qwen3vl_2b_delivery_profile_active() const {
        return !pipeline_dispatch_active() &&
            delivery_compatibility_ && !linear_acc32_ &&
            num_layers() == 24 && hidden_size() == 1024 &&
            intermediate_size() == 4096 && num_q_heads() == 16 &&
            head_dim() == 64;
    }
    // inc-2: deepstack mergers folded too (weights registered via set_deepstack_merger_weights).
    int64_t vision_ring_route(int64_t rows, int64_t cols) const {
        return fmb_ring_all_reduce_route_selector(rows, cols, num_cores());
    }

    void zero_owner_spm(uint32_t address, int64_t elements) const {
        if (num_cores() == 8) {
            rpu_launch_memset_spm_multicore(address, elements);
        } else {
            rpu_launch_fill_spm_kernel(address, elements, c10::Half(0.0f), num_cores());
        }
    }

    bool deepstack_merger_active() const {
        return merger_active() && !ds_merger_m0_w_.empty();
    }

    // ========================================================================
    // merger_post_fn — runs once AFTER the 24-block encoder, INSIDE the same
    // GraphCache capture (D-501 post_fn). Computes the Qwen3-VL patch merger:
    //   merged = m2( gelu( m0( layernorm_lnq(residual1[seq,h]).reshape(seq/4, mh) ) ) )
    // output [seq/4, OUT_HID] → post_output_tensor_ (window order; Python applies
    // the visual-embed placement). m0 = col GEMM, m2 = row GEMM + all-reduce
    // (ViT SwiGLU dataflow). Mirrors qwen25vl with LayerNorm not RMSNorm.
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
                         bool postshuffle_norm = false,
                         int64_t invocation = 0,
                         const at::Tensor& m0_scale = {},
                         const at::Tensor& m2_scale = {},
                         int64_t sequence_rows = 0) {
        const int64_t seq = sequence_rows == 0 ? current_num_patches_ : sequence_rows;
        TORCH_CHECK(seq > 0 && seq % 4 == 0,
                    "Qwen3-VL merger requires complete four-patch groups");
        const int64_t h = hidden_size();
        const int64_t mh = merge_hidden_, oh = merger_out_hidden_;
        const int64_t mrows = seq / 4, local_mh = mh / num_cores();
        const int64_t norm_rows = postshuffle_norm ? mrows : seq;
        const int64_t norm_hidden = postshuffle_norm ? mh : h;
        // weights → shared SPM slots (ln_q γ/β broadcast; m0 bias col-scatter; m2 bias row).
        if (pipeline_dispatch_active()) {
            rpu_launch_ddr_broadcast_spm_dma(
                ln_q_w, /*src_offset_elements=*/0, norm_hidden,
                addr(0, "merger_ln_q_w"), num_cores());
            rpu_launch_ddr_broadcast_spm_dma(
                ln_q_b, /*src_offset_elements=*/0, norm_hidden,
                addr(0, "merger_ln_q_b"), num_cores());
            rpu_launch_ddr_scatter_spm_dma(
                m0_b, /*src_offset_elements=*/0, local_mh,
                local_mh * DWIDTH, addr(0, "merger_m0_bias"), num_cores());
        } else {
            rpu_launch_ddr_broadcast_spm_dma(
                ln_q_w.data_ptr<c10::Half>(), norm_hidden,
                addr(0, "merger_ln_q_w"), num_cores());
            rpu_launch_ddr_broadcast_spm_dma(
                ln_q_b.data_ptr<c10::Half>(), norm_hidden,
                addr(0, "merger_ln_q_b"), num_cores());
            rpu_launch_ddr_scatter_spm_dma(
                m0_b.data_ptr<c10::Half>(), local_mh,
                local_mh * DWIDTH, addr(0, "merger_m0_bias"), num_cores());
        }
        zero_owner_spm(addr(0, "merger_m2_bias"), oh);
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
            norm_rows, norm_hidden, merger_ln_eps_, false, 0, num_cores());
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN3VL_MERGER_M0_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            invocation);
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0, "merger_normed"), m0_w, addr(0, "merger_mid"),
            mrows, mh, mh, /*col*/1, num_cores(), addr(0, "merger_m0_bias"), m0_scale,
            /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
            /*force_acc32=*/linear_acc32_);
        if (rhinovla_high_precision_) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, RHINOVLA_VISION_HIGH_ACTIVATION_SITE,
                1, 0, {2, 1024, 4096, 2048, 4, 8});
        }
        rpu_launch_eltwise_unary_spm_kernel(addr(0, "merger_mid"), addr(0, "merger_mid"),
            mrows * local_mh, ValuOpType::ADD,
            GeluMode::ERF, num_cores(), rhinovla_high_precision_
                ? RpuUnaryPrecision::HIGH : RpuUnaryPrecision::BASE);
        // m2 row GEMM writes its PARTIAL back into merger_normed (the LN-out buffer, dead once m0
        // consumed it) → no separate merger_partial slot (so it can Temp-alias oproj; mirrors
        // qwen25vl). all-reduce operands stay distinct: partial=merger_normed(oproj),
        // zero=merger_zero(input_norm), out=merger_out(residual1).
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN3VL_MERGER_M2_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            invocation);
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0, "merger_mid"), m2_w, addr(0, "merger_normed"),
            mrows, oh, mh, /*row*/0, num_cores(), addr(0, "merger_m2_bias"), m2_scale,
            /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
            /*force_acc32=*/linear_acc32_);
        zero_owner_spm(addr(0, "merger_zero"), mrows * oh);
    }

    void emit_one_merger(uint32_t in_addr, uint32_t final_out_addr,
                         const at::Tensor& ln_q_w, const at::Tensor& ln_q_b,
                         const at::Tensor& m0_w, const at::Tensor& m0_b,
                         const at::Tensor& m2_w, const at::Tensor& m2_b,
                         bool postshuffle_norm = false,
                         int64_t invocation = 0,
                         const at::Tensor& m0_scale = {},
                         const at::Tensor& m2_scale = {},
                         int64_t sequence_rows = 0) {
        TORCH_CHECK(final_out_addr != 0,
                    "Qwen3VLVisionModel merger final output address is unset");
        emit_one_merger_prefix(
            in_addr, ln_q_w, ln_q_b, m0_w, m0_b, m2_w, m2_b,
            postshuffle_norm, invocation, m0_scale, m2_scale, sequence_rows);
        const int64_t mrows = (sequence_rows == 0 ? current_num_patches_ : sequence_rows) / 4;
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN3VL_MERGER_ALL_REDUCE_SITE,
            vision_ring_route(
                mrows, merger_out_hidden_), invocation);
        rpu_launch_all_reduce_sum_residual_kernel(addr(0, "merger_normed"), addr(0, "merger_zero"),
            final_out_addr, mrows, merger_out_hidden_, num_cores(), num_cores());
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
                         bool postshuffle_norm = false,
                         int64_t invocation = 0) {
        TORCH_CHECK(
            multiview_composite_dispatch_ &&
                merger_out_hidden_ == QWEN3VL_MULTIVIEW_TEXT_HIDDEN,
            "Qwen3VLVisionModel typed merger yield requires the exact "
            "multiview composite dispatch");
        emit_one_merger_prefix(
            in_addr, ln_q_w, ln_q_b, m0_w, m0_b, m2_w, m2_b,
            postshuffle_norm, invocation);
        auto yield = begin_spm_pipeline_dense_producer_yield(
            "merger_out", /*allocation_layer=*/-1,
            multiview_yield_spec());
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN3VL_MERGER_YIELD_ALL_REDUCE_SITE,
            vision_ring_route(
                QWEN3VL_POOLER_Z1_MERGED_ROWS,
                QWEN3VL_MULTIVIEW_TEXT_HIDDEN), invocation);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "merger_normed"), addr(0, "merger_zero"),
            yield.target(), QWEN3VL_POOLER_Z1_MERGED_ROWS,
            QWEN3VL_MULTIVIEW_TEXT_HIDDEN, num_cores(), num_cores());
        yield.finish();
    }

    void emit_chunked_merger_post_fn() {
        const int64_t seq = current_num_patches_, h = hidden_size(), oh = merger_out_hidden_;
        const auto merger_chunks = merger_compute_chunks(ctx().stage_plan);
        const int64_t chunks = static_cast<int64_t>(merger_chunks.size());
        TORCH_CHECK(chunks > 1 && image_batch_count_ == 1 &&
                        !prefix_scatter_active_ &&
                        (ordinary_w8_resident_profile() ||
                         fp16_chunked_merger_plan(ctx().stage_plan, seq, seq, 0)) &&
                        seq > 0 && seq % 4 == 0,
                    "Qwen3-VL chunked merger exceeds its ordinary W8 or exact FP16 profile");
        int64_t offset = 0;
        for (int64_t index = 0; index < chunks; ++index) {
            const auto& chunk = merger_chunks[index];
            TORCH_CHECK(chunk.idx == index && chunk.offset == offset &&
                            chunk.len > 0 && chunk.len % 4 == 0 &&
                            chunk.len <= merger_compute_chunk_size(ctx().stage_plan) &&
                            chunk.len <= seq - offset,
                        "Qwen3-VL chunked merger patch coverage drift");
            offset += chunk.len;
        }
        TORCH_CHECK(offset == seq, "Qwen3-VL chunked merger omitted raw rows");

        // All routes cross the post-loop scratch boundary through the complete
        // raw DDR output, even when the compact encoder retained all N rows.
        const at::Tensor& raw = output_tensor();
        chunked_merger_raw_src_base_ = RpuGetDevAddr(raw.data_ptr<c10::Half>());
        rpu_ddr_flush(raw.data_ptr<c10::Half>());
        if (RpuKernelGraph::has_active()) RpuKernelGraph::active().keep_alive(raw);
        const int64_t merger_count = 1 + (deepstack_merger_active()
            ? static_cast<int64_t>(deepstack_visual_indexes_.size()) : 0);
        for (int64_t ordinal = 0; ordinal < merger_count; ++ordinal) {
            const auto key = std::make_pair(
                ordinal == 0 ? int64_t{-1} : deepstack_visual_indexes_.at(ordinal - 1), seq);
            const at::Tensor& destination = ordinal == 0
                ? pooler_merged_ddr_slots_.at(seq) : deepstack_merged_ddr_slots_.at(key);
            for (const auto& chunk : merger_chunks) {
                const int64_t invocation = ordinal * chunks + chunk.idx;
                ctx().consume_physical_route(
                    FmbRouteFamily::MUTABLE_DMA, QWEN3VL_CHUNKED_MERGER_INPUT_SITE,
                    static_cast<int64_t>(Qwen3VLVisionMutableDmaRoute::MERGER_INPUT_DDR_TO_SPM),
                    0, {ordinal, seq, chunk.offset, chunk.len, h}, invocation);
                if (ordinal == 0) {
                    rpu_launch_ddr_broadcast_spm_dma_mutable(
                        chunked_merger_raw_src_base_, raw, chunk.offset * h * DWIDTH,
                        chunk.len * h, addr(0, "merger_in"), num_cores());
                    emit_one_merger(addr(0, "merger_in"), addr(0, "merger_out"),
                        merger_ln_q_w_, merger_ln_q_b_, merger_m0_w_, merger_m0_b_,
                        merger_m2_w_, merger_m2_b_, false, invocation,
                        merger_m0_scale_, merger_m2_scale_, chunk.len);
                } else {
                    const size_t k = static_cast<size_t>(ordinal - 1);
                    rpu_launch_ddr_broadcast_spm_dma(
                        deepstack_ddr_slots_.at(key), chunk.offset * h, chunk.len * h,
                        addr(0, "merger_in"), num_cores());
                    emit_one_merger(addr(0, "merger_in"), addr(0, "merger_out"),
                        ds_merger_ln_q_w_[k], ds_merger_ln_q_b_[k], ds_merger_m0_w_[k],
                        ds_merger_m0_b_[k], ds_merger_m2_w_[k], ds_merger_m2_b_[k],
                        true, invocation, ds_merger_m0_scale_[k], ds_merger_m2_scale_[k], chunk.len);
                }
                ctx().consume_physical_route(
                    FmbRouteFamily::MUTABLE_DMA, QWEN3VL_CHUNKED_MERGER_OUTPUT_SITE,
                    static_cast<int64_t>(Qwen3VLVisionMutableDmaRoute::SPM_COPY_TO_DDR),
                    0, {ordinal, seq / 4, chunk.offset / 4, chunk.len / 4, oh}, invocation);
                rpu_launch_spm_copy_ddr_dma(
                    addr(0, "merger_out"), destination,
                    (chunk.offset / 4) * oh, (chunk.len / 4) * oh);
            }
        }
    }

    void merger_post_fn() {
        if (fp16_chunked_merger_enabled_) {
            if (fp16_chunked_merger_active()) emit_chunked_merger_post_fn();
            return;
        }
        if (w8_compact_encoder_active()) {
            TORCH_CHECK(compact_encoder_plan(ctx().stage_plan,
                            current_num_patches_, current_num_patches_, 0),
                        "Qwen3-VL W8 compact merger requires its exact encoder plan");
            emit_chunked_merger_post_fn();
            return;
        }
        if (adaptive_w8_merger() && ctx().stage_plan.compute.chunks.size() != 1) {
            if (image_batch_count_ == 1 && !prefix_scatter_active_ &&
                ordinary_w8_resident_profile()) {
                emit_chunked_merger_post_fn();
            }
            // Other profiles keep their original eager merger and descriptor.
            return;
        }
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
            merger_ln_q_w_, merger_ln_q_b_, merger_m0_w_, merger_m0_b_, merger_m2_w_, merger_m2_b_,
            false, 0, merger_m0_scale_, merger_m2_scale_);
        if (prefix_scatter_active_) {
            scatter_merger_out_to_prefix(oh, prefix_pooler_ptr_);
        } else {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN3VL_POOLER_MERGED_COPY_SITE,
                static_cast<int64_t>(
                    Qwen3VLVisionMutableDmaRoute::SPM_COPY_TO_DDR),
                /*resolved_flags=*/0,
                {mrows, oh, prefix_scatter_active_ ? 1 : 0});
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "merger_out"), pooler_merged_ddr_slots_.at(seq).data_ptr<c10::Half>(), mrows * oh);
        }

        // inc-2: deepstack mergers. Each reads its layer's snapshot (DMA DDR→"merger_in") and
        // writes a STABLE merged slot (fixed DMA → deterministic, no post_output drift). Replaces
        // the eager CPU-LN + RPU-GEMM deepstack mergers (kills 3 CPU LNs + 6 eager GEMMs + bounces).
        if (deepstack_merger_active()) {
            for (size_t k = 0; k < deepstack_visual_indexes_.size(); ++k) {
                auto key = std::make_pair(deepstack_visual_indexes_[k], seq);
                rpu_launch_ddr_broadcast_spm_dma(
                    deepstack_ddr_slots_.at(key).data_ptr<c10::Half>(), seq * h, addr(0, "merger_in"), num_cores());
                emit_one_merger(addr(0, "merger_in"), addr(0, "merger_out"),
                    ds_merger_ln_q_w_[k], ds_merger_ln_q_b_[k], ds_merger_m0_w_[k],
                    ds_merger_m0_b_[k], ds_merger_m2_w_[k], ds_merger_m2_b_[k],
                    /*postshuffle_norm=*/true,
                    /*invocation=*/static_cast<int64_t>(k) + 1,
                    ds_merger_m0_scale_[k], ds_merger_m2_scale_[k]);
                if (prefix_scatter_active_) {
                    scatter_merger_out_to_prefix(oh, prefix_dense_ptrs_.at(k));
                } else {
                    ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        QWEN3VL_DEEPSTACK_MERGED_COPY_SITE,
                        static_cast<int64_t>(
                            Qwen3VLVisionMutableDmaRoute::SPM_COPY_TO_DDR),
                        /*resolved_flags=*/0,
                        {mrows, oh, prefix_scatter_active_ ? 1 : 0},
                        static_cast<int64_t>(k) + 1);
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
    FmbRopeTableResidency selected_rope_residency(
        at::IntArrayRef descriptor,
        bool require_spm) const {
        const auto prepared_candidate = prepare_stage_candidate(descriptor);
        const auto& candidate = prepared_candidate->candidate();
        const FmbRopeTableResidency residency =
            fmb_rope_table_residency(candidate.physical_manifest);
        TORCH_CHECK(
            residency != FmbRopeTableResidency::UNSPECIFIED,
            "RPU_PLANNER_REJECT:CAPABILITY: Qwen3-VL Vision descriptor has "
            "no 2D RoPE table-residency authority");
        if (rope_route_request_ != Qwen3VLVisionRopeRouteRequest::AUTO) {
            const FmbRopeTableResidency exact =
                rope_route_request_ == Qwen3VLVisionRopeRouteRequest::SPM
                ? FmbRopeTableResidency::SPM
                : FmbRopeTableResidency::DDR;
            TORCH_CHECK(
                residency == exact,
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Qwen3-VL Vision "
                "descriptor conflicts with the cold exact RoPE request");
        }
        TORCH_CHECK(
            !require_spm || residency == FmbRopeTableResidency::SPM,
            "RPU_PLANNER_REJECT:CAPABILITY: Qwen3-VL native composite "
            "requires descriptor-selected SPM 2D RoPE");
        return residency;
    }

    void bind_descriptor_layout(
        LayoutContext& layout,
        const FmbThreeStageChunkPlan& stage_plan,
        at::IntArrayRef descriptor,
        bool require_spm) {
        const auto prepared_candidate = prepare_stage_candidate(descriptor);
        const auto& candidate = prepared_candidate->candidate();
        const uint64_t stage_fingerprint =
            fmb_three_stage_chunk_plan_fingerprint(stage_plan);
        TORCH_CHECK(
            fmb_three_stage_chunk_plan_fingerprint(candidate.stage_plan) ==
                stage_fingerprint,
            "RPU_PLANNER_REJECT:EXACT_MISMATCH: Qwen3-VL Vision prepare "
            "stage plan differs from its selected descriptor");
        layout.stage_plan_fingerprint = stage_fingerprint;
        layout.physical_manifest_fingerprint =
            fmb_physical_manifest_fingerprint(candidate.physical_manifest);
        layout.rope_table_residency =
            selected_rope_residency(descriptor, require_spm);
        rope_spm_enabled_ =
            layout.rope_table_residency == FmbRopeTableResidency::SPM;
    }

    std::vector<int64_t> build_native_composite_stage_descriptor(
        bool multiview,
        int64_t retained_count) {
        TORCH_INTERNAL_ASSERT(
            !pipeline_dispatch_active() &&
                native_composite_stage_descriptor_.empty());
        const int64_t saved_num_patches = current_num_patches_;
        const int64_t saved_image_batch_count = image_batch_count_;
        const int64_t saved_chunk_override = get_chunk_size_override();
        const int64_t saved_retained_count = z1_retained_deepstack_count_;
        auto restore = c10::make_scope_exit([&] {
            z1_dispatch_ = false;
            multiview_composite_dispatch_ = false;
            z1_retained_deepstack_count_ = saved_retained_count;
            current_num_patches_ = saved_num_patches;
            image_batch_count_ = saved_image_batch_count;
            set_chunk_size_override(saved_chunk_override);
        });

        current_num_patches_ = QWEN3VL_POOLER_Z1_NUM_PATCHES;
        image_batch_count_ = 1;
        set_chunk_size_override(0);
        z1_retained_deepstack_count_ = retained_count;
        z1_dispatch_ = !multiview;
        multiview_composite_dispatch_ = multiview;

        const std::vector<ChunkInfo> input_chunks{{
            0, 0, QWEN3VL_POOLER_Z1_NUM_PATCHES,
            QWEN3VL_POOLER_Z1_NUM_PATCHES}};
        const std::vector<FmbExecutionSpan> spans{{
            0, QWEN3VL_POOLER_Z1_NUM_PATCHES, 0}};
        const FmbStageBoundaryPolicies boundary_policies{};
        std::vector<FmbPrefillStageCandidate> candidates =
            resolve_prefill_stage_domain_for_shape(
                QWEN3VL_POOLER_Z1_NUM_PATCHES,
                /*position=*/0, /*attention_mask=*/std::nullopt,
                /*is_causal=*/false, input_chunks, spans,
                boundary_policies,
                /*requested_chunk_size=*/QWEN3VL_POOLER_Z1_NUM_PATCHES,
                /*logical_len=*/QWEN3VL_POOLER_Z1_NUM_PATCHES);
        TORCH_CHECK(
            candidates.size() == 1 &&
                candidates.front().physical_manifest.state ==
                    FmbPhysicalManifestState::COMPLETE &&
                candidates.front().physical_manifest.graph_lifecycle ==
                    FmbGraphLifecycle::COMPOSITE_CHILD,
            "Qwen3VLVisionModel native composite Vision must resolve exactly "
            "one COMPLETE COMPOSITE_CHILD descriptor");
        auto descriptor = encode_fmb_prefill_stage_candidate(candidates.front());
        bind_kvinsert_cost_owner_prefix(descriptor,
            {INT64_C(0x51564d5656434f53), multiview ? 1 : 0, retained_count});
        return descriptor;
    }

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
        TORCH_CHECK(num_cores() == 8, "Vision composite profiles require eight cores");
        TORCH_CHECK(num_layers() == 24 && num_q_heads() == 16 &&
                        head_dim() == 64 &&
                        hidden_size() == QWEN3VL_POOLER_Z1_VISION_HIDDEN &&
                        intermediate_size() == 4096,
                    "Qwen3VLVisionModel pooler Z1 requires the exact 24-layer "
                    "Qwen3-VL-2B Vision profile");
        TORCH_CHECK(spm_rope_route_available(),
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
            (num_q_heads() / num_cores()) * head_dim();
        TORCH_CHECK(q_ddr_slots_.count(
                        {QWEN3VL_POOLER_Z1_NUM_PATCHES, local_q_dim}) == 1,
                    "Qwen3VLVisionModel pooler Z1 fixed Q staging slot is "
                    "missing");
    }

    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        if (rhinovla_high_precision_) {
            identity.insert(identity.end(), {RHINOVLA_VISION_HIGH_ACTIVATION_SITE, 2});
        }
        if (num_cores() != 8) {
            identity.insert(identity.end(), {0x434f5245, num_cores(), attn_tp(), mlp_tp(), 8});
        }
        if (large_image_auto_chunk_) {
            identity.insert(identity.end(), {
                0x4c415247, QWEN3VL_VISION_LARGE_IMAGE_THRESHOLD,
                QWEN3VL_VISION_LARGE_IMAGE_CHUNK_CAP});
        }
        if (n1200_tm160_enabled_) {
            identity.insert(identity.end(), {0x544d3136, 1, 1200, 608, 592, 160,
                                            QWEN3VL_N1200_SDPA_TMP_BYTES});
        }
        if (fp16_chunked_merger_enabled_) {
            identity.insert(identity.end(), {0x4631364d, 1, 1200, 608, 592});
        }
        if (fp16_compact_encoder_enabled_) {
            identity.insert(identity.end(), {0x46313643, 1, 1200, 608, 592, 153600});
        }
        if (w8_compact_encoder_enabled_) {
            identity.insert(identity.end(), {0x573843, 1, 1200, 608, 592, 153600, 8, 16});
        }
        append_kvinsert_cost_scalar_identity(identity, eps_);
        append_kvinsert_cost_scalar_identity(identity, merger_ln_eps_);
        identity.push_back(static_cast<int64_t>(deepstack_visual_indexes_.size()));
        identity.insert(identity.end(), deepstack_visual_indexes_.begin(), deepstack_visual_indexes_.end());
        for (const auto* weights : {
                &ds_merger_ln_q_w_, &ds_merger_ln_q_b_, &ds_merger_m0_w_,
                &ds_merger_m0_b_, &ds_merger_m2_w_, &ds_merger_m2_b_}) {
            identity.push_back(static_cast<int64_t>(weights->size()));
            for (const auto& tensor : *weights) {
                append_kvinsert_cost_tensor_identity(identity, tensor);
            }
        }
        if (adaptive_w8_merger()) {
            append_kvinsert_cost_tensor_identity(identity, merger_m0_scale_);
            append_kvinsert_cost_tensor_identity(identity, merger_m2_scale_);
            for (const auto* scales : {&ds_merger_m0_scale_, &ds_merger_m2_scale_}) {
                for (const auto& scale : *scales) append_kvinsert_cost_tensor_identity(identity, scale);
            }
        }
        identity.insert(identity.end(), {
            static_cast<int64_t>(has_rope_),
            static_cast<int64_t>(rope_spm_enabled_),
            static_cast<int64_t>(fast_replay_skip_layer_loop_),
            static_cast<int64_t>(preload_replay_skip_),
            static_cast<int64_t>(bake_merger_),
            static_cast<int64_t>(linear_acc32_),
            static_cast<int64_t>(fused_merger_enabled_)});
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.fc1_w, &weights.fc2_w, &weights.q_ws, &weights.k_ws,
                    &weights.v_ws, &weights.o_ws, &weights.fc1_ws, &weights.fc2_ws}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        identity.push_back(static_cast<int64_t>(layer_bias_norm_.size()));
        for (const auto& weights : layer_bias_norm_) {
            for (const auto* tensor : {
                    &weights.ln1_w, &weights.ln1_b, &weights.ln2_w, &weights.ln2_b,
                    &weights.q_b, &weights.k_b, &weights.v_b, &weights.o_b,
                    &weights.fc1_b, &weights.fc2_b}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &merger_ln_q_w_, &merger_ln_q_b_, &merger_m0_w_, &merger_m0_b_,
                &merger_m2_w_, &merger_m2_b_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerBiasNorm> layer_bias_norm_;

    // 2D RoPE state
    at::Tensor freq_cos_;
    at::Tensor freq_sin_;
    at::Tensor position_idx_keepalive_;
    int64_t max_hw_ = 0;
    bool has_rope_ = false;
    Qwen3VLVisionRopeRouteRequest rope_route_request_ =
        Qwen3VLVisionRopeRouteRequest::DDR;
    bool rope_spm_enabled_ = false;
    bool native_composite_prepare_active_ = false;
    // RhinoVLA fast-replay opt-in (per-model; flow into ModelStaticConfig in
    // build_config, honored by FusedModelBase::run_all_layers). Default false →
    // other qwen3_vl consumers (gr00t) unaffected.
    bool fast_replay_skip_layer_loop_ = false;
    bool preload_replay_skip_ = false;
    bool bake_merger_ = false;
    // RhinoVLA prefix-scatter: merger writes directly into the persistent prefix
    // buffers (host ptrs kept alive Python-side) at the visual run positions.
    bool prefix_scatter_active_ = false;
    // MR-C / T31 (B3): the scatter targets are RETAINED here, not just pointed at.
    //
    // This API used to take raw integer addresses and reinterpret_cast them,
    // holding no reference to the storage behind them (the C-1 shape). Whether
    // the pointers stayed valid depended entirely on the caller keeping its own
    // tensors alive by convention — rhino_e2e.py does it via
    // cache["inputs_embeds_persist"] / cache["deepstack_dense_persist"], and
    // nothing enforced or even checked that. A caller that dropped them would
    // have this model DMA merger output into reused memory: no crash, no
    // exception, silently corrupt neighbours.
    //
    // Holding the Tensors makes the lifetime a fact instead of an agreement. The
    // pointers below are derived from them and are only valid while they are.
    at::Tensor prefix_pooler_keep_;
    std::vector<at::Tensor> prefix_dense_keep_;
    c10::Half* prefix_pooler_ptr_ = nullptr;
    std::vector<c10::Half*> prefix_dense_ptrs_;
    std::vector<std::pair<int64_t,int64_t>> prefix_runs_;
   public:
    void set_fast_replay_skip_layer_loop(bool e) { if (fast_replay_skip_layer_loop_ != e) { fast_replay_skip_layer_loop_ = e; invalidate_model_state(); } }
    void set_preload_replay_skip(bool e) { if (preload_replay_skip_ != e) { preload_replay_skip_ = e; invalidate_model_state(); } }
    void set_bake_merger(bool e) { if (bake_merger_ != e) { bake_merger_ = e; invalidate_model_state(); } }
    // MR-C / T31 (B3): takes the target TENSORS, not integer addresses. See the
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
    uint64_t chunked_merger_raw_src_base_ = 0;
    std::map<std::pair<int64_t, int64_t>, at::Tensor> q_ddr_slots_;
    at::Tensor& q_ddr_slot() {
        return q_ddr_slots_.at({current_num_patches_, (num_q_heads() / num_cores()) * head_dim()});
    }
    // Multi-image BATCH: N equal-size images packed into one forward; drives the
    // per-image minibatch SDPA in build_layer_subgraph (=1 => single-image path).
    int64_t image_batch_count_ = 1;
    // Public exact execution control. Zero preserves the established
    // shape/mode-dependent policy above; a positive value is checked against
    // the actual resolved plan by Python after every dispatch.
    int64_t configured_chunk_size_ = 0;
    // Enabled only by the cold ordinary 2B/4B root after its profile validation.
    // Shared Vision/VLA installers keep the existing AUTO policy by default.
    bool large_image_auto_chunk_ = false;
    bool n1200_tm160_enabled_ = false;
    bool fp16_chunked_merger_enabled_ = false;
    bool fp16_compact_encoder_enabled_ = false;
    bool w8_compact_encoder_enabled_ = false;
    bool linear_acc32_ = false;
    bool delivery_compatibility_ = false;

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
    std::vector<int64_t> native_composite_stage_descriptor_;
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
    //
    static constexpr size_t kMaxSlotShapes = 128;
    rpu::BoundedShapeMap<std::pair<int64_t, int64_t>, at::Tensor, rpu::PairHash>
        deepstack_ddr_slots_{kMaxSlotShapes, rpu::ShapeMapPolicy::Reject,
                             "Qwen3VLVisionModel::deepstack_ddr_slots_"};
    std::vector<int64_t> deepstack_visual_indexes_;

    // Model config
    double eps_ = 1e-6;
    int64_t orig_head_dim_ = 0;

    // Fused patch merger (ROUND-3 opt #1): LayerNorm(ln_q) -> reshape[seq/4, mh] -> m0 GEMM
    // (col) -> GELU -> m2 GEMM (row) -> all-reduce -> SPM->DDR. In-graph post_fn, single image.
    at::Tensor merger_ln_q_w_, merger_ln_q_b_;   // [HID] fp16 (LayerNorm gamma/beta)
    at::Tensor merger_m0_w_, merger_m0_b_;       // [merge_hidden, merge_hidden], [merge_hidden]
    at::Tensor merger_m2_w_, merger_m2_b_;       // [OUT_HID, merge_hidden], [OUT_HID]
    at::Tensor merger_m0_scale_, merger_m2_scale_;
    int64_t merger_out_hidden_ = 0;              // OUT_HID (2048); 0 => merger not configured
    int64_t merge_hidden_ = 0;                   // HID * sms^2 (4096)
    double merger_ln_eps_ = 1e-6;                // merger.norm.eps (distinct from the ViT eps_)
    bool rhinovla_high_precision_ = false;
    bool rhinovla_precision_bound_ = false;
    bool fused_merger_enabled_ = false;           // cold at set_merger_weights()
    // Stable patch-merged pooler slot [num_patches/4, oh] (fixed DMA, popped by pop_pooler_merged).
    // Bounded, NOT evictable for the same reason as deepstack_ddr_slots_: the
    // slot address is baked into merger_post_fn's fixed DMA at BUILD. Keyed on
    // num_patches alone (one entry per distinct image shape), so it needs no
    // per-layer multiplier — measured high-water 2 on the gr00t 1img+2img run.
    rpu::BoundedShapeMap<int64_t, at::Tensor>
        pooler_merged_ddr_slots_{kMaxSlotShapes, rpu::ShapeMapPolicy::Reject,
                                 "Qwen3VLVisionModel::pooler_merged_ddr_slots_"};

    // inc-2: the N deepstack mergers (one per deepstack_visual_index). Same shapes as the patch
    // merger; folded in merger_post_fn (each reads its layer's snapshot → stable merged slot).
    std::vector<at::Tensor> ds_merger_ln_q_w_, ds_merger_ln_q_b_;
    std::vector<at::Tensor> ds_merger_m0_w_, ds_merger_m0_b_;
    std::vector<at::Tensor> ds_merger_m2_w_, ds_merger_m2_b_;
    std::vector<at::Tensor> ds_merger_m0_scale_, ds_merger_m2_scale_;
    // Stable merged-output slots [num_patches/4, oh], shape-keyed like deepstack_ddr_slots_
    // — same bound, same Reject policy, same fixed-DMA reason.
    rpu::BoundedShapeMap<std::pair<int64_t, int64_t>, at::Tensor, rpu::PairHash>
        deepstack_merged_ddr_slots_{kMaxSlotShapes, rpu::ShapeMapPolicy::Reject,
                                    "Qwen3VLVisionModel::deepstack_merged_ddr_slots_"};
};

}  // namespace v3

using Qwen3VLVisionRegistry = ModelHandleRegistry<v3::Qwen3VLVisionModel>;

std::vector<int64_t> rpu_qwen3vl_vision_planner_cache_identity(int64_t handle) {
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_qwen3vl_vision_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen3vl_vision_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_qwen3vl_vision_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_kvinsert_cost_domain")
        ->kvinsert_cost_domain("qwen3vl_vision", descriptor);
}

std::string rpu_qwen3vl_vision_kvinsert_cost_catalog_sha256(int64_t handle) {
    return Qwen3VLVisionRegistry::get(
        handle, "rpu_qwen3vl_vision_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers
// =============================================================================

int64_t rpu_qwen3vl_vision_create(int64_t num_cores) {
    TORCH_CHECK(num_cores == 4 || num_cores == 8,
                "Qwen3-VL Vision admits four or eight execution cores");
    const int64_t handle = Qwen3VLVisionRegistry::create();
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_create")
        ->configure_execution_cores(static_cast<int>(num_cores));
    return handle;
}

std::vector<int64_t> rpu_qwen3vl_vision_topology(int64_t handle) {
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_topology")
        ->execution_topology();
}

void rpu_qwen3vl_vision_destroy(int64_t handle) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_destroy")
        ->check_pooler_z1_destroy_allowed();
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_destroy")
        ->check_execution_reconfigure_destroy_allowed(
            "rpu_qwen3vl_vision_destroy");
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
    const at::Tensor& freq_sin,
    bool rope_spm_enabled)
{
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_rope_tables(freq_cos, freq_sin, rope_spm_enabled);
}

void rpu_qwen3vl_vision_set_rope_route(
    int64_t handle,
    const at::Tensor& freq_cos,
    const at::Tensor& freq_sin,
    int64_t route_request)
{
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_rope_tables_with_route(
            freq_cos, freq_sin, route_request);
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

void rpu_qwen3vl_vision_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_set_chunk_envelope")
        ->set_chunk_envelope(std::min(max_kv_len, QWEN3VL_VISION_MAX_KEEPALIVE_SEQ), chunk);
}

void rpu_qwen3vl_vision_set_chunk_size(int64_t handle, int64_t chunk_size) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_set_chunk_size")
        ->set_configured_chunk_size(chunk_size);
}

void rpu_qwen3vl_vision_enable_execution_reconfigure(int64_t handle) {
    Qwen3VLVisionRegistry::get(
        handle, "rpu_qwen3vl_vision_enable_execution_reconfigure")
        ->enable_execution_reconfigure_guard();
}

void rpu_qwen3vl_vision_stage_chunk_size(
        int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0,
                "Qwen3-VL vision hot-reconfigure token must be positive");
    Qwen3VLVisionRegistry::get(
        handle, "rpu_qwen3vl_vision_stage_chunk_size")
        ->stage_configured_chunk_size(
            static_cast<uint64_t>(token), chunk_size);
}

void rpu_qwen3vl_vision_set_rhinovla_high_precision(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_set_rhinovla_high_precision")
        ->set_rhinovla_high_precision(enabled);
}

void rpu_qwen3vl_vision_set_linear_acc32(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(
        handle, "rpu_qwen3vl_vision_set_linear_acc32")
        ->set_linear_acc32(enabled);
}

void rpu_qwen3vl_vision_set_large_image_auto_chunk(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(
        handle, "rpu_qwen3vl_vision_set_large_image_auto_chunk")
        ->set_large_image_auto_chunk(enabled);
}

void rpu_qwen3vl_vision_set_n1200_tm160(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_set_n1200_tm160")
        ->set_n1200_tm160(enabled);
}

void rpu_qwen3vl_vision_set_fp16_chunked_merger(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_set_fp16_chunked_merger")
        ->set_fp16_chunked_merger(enabled);
}

void rpu_qwen3vl_vision_set_fp16_compact_encoder(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_set_fp16_compact_encoder")
        ->set_fp16_compact_encoder(enabled);
}

void rpu_qwen3vl_vision_set_w8_compact_encoder(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision_set_w8_compact_encoder")
        ->set_w8_compact_encoder(enabled);
}

void rpu_qwen3vl_vision_set_delivery_compatibility(int64_t handle, bool enabled) {
    Qwen3VLVisionRegistry::get(
        handle, "rpu_qwen3vl_vision_set_delivery_compatibility")
        ->set_delivery_compatibility(enabled);
}

int64_t rpu_qwen3vl_vision_get_resolved_chunk_size(int64_t handle) {
    return Qwen3VLVisionRegistry::get(
               handle, "rpu_qwen3vl_vision_get_resolved_chunk_size")
        ->resolved_chunk_size();
}

std::vector<int64_t> rpu_qwen3vl_vision_resolve_stage_domain(
        int64_t handle,
        int64_t num_patches,
        int64_t image_batch_count) {
    return Qwen3VLVisionRegistry::get(
               handle, "rpu_qwen3vl_vision_resolve_stage_domain")
        ->resolve_stage_domain(num_patches, image_batch_count);
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
    int64_t out_hidden, double ln_eps,
    const c10::optional<at::Tensor>& m0_scale,
    const c10::optional<at::Tensor>& m2_scale)
{
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_merger_weights(ln_q_w, ln_q_b, m0_w, m0_b, m2_w, m2_b, out_hidden, ln_eps,
                             m0_scale.value_or(at::Tensor{}), m2_scale.value_or(at::Tensor{}));
}

void rpu_qwen3vl_vision_set_deepstack_merger_weights(
    int64_t handle,
    at::TensorList ln_q_w, at::TensorList ln_q_b,
    at::TensorList m0_w, at::TensorList m0_b,
    at::TensorList m2_w, at::TensorList m2_b,
    const c10::optional<std::vector<at::Tensor>>& m0_scale,
    const c10::optional<std::vector<at::Tensor>>& m2_scale)
{
    const auto m0_scales = m0_scale.value_or(std::vector<at::Tensor>{});
    const auto m2_scales = m2_scale.value_or(std::vector<at::Tensor>{});
    Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->set_deepstack_merger_weights(ln_q_w, ln_q_b, m0_w, m0_b, m2_w, m2_b, m0_scales, m2_scales);
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
    int64_t image_batch_count,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Qwen3VLVisionRegistry::get(handle, "rpu_qwen3vl_vision")
        ->forward(input, k_caches, v_caches, num_patches, image_batch_count,
                  planned_stage_descriptor);
}

namespace v3::qwen3vl_pooler_z1_internal {

SpmPipelineComponentLayout prepare_vision(int64_t handle,
                                          int64_t num_patches,
                                          int64_t retained_count,
                                          at::IntArrayRef selected_descriptor) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_pooler_z1_prepare_vision")
        ->prepare_pooler_z1_layout(num_patches, retained_count, selected_descriptor);
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
    int64_t num_patches, at::IntArrayRef selected_descriptor) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_multiview_prepare_vision_composite")
        ->prepare_multiview_composite_layout(num_patches, selected_descriptor);
}

std::vector<int64_t> vision_stage_descriptor(int64_t handle) {
    return Qwen3VLVisionRegistry::get(
               handle, "qwen3vl_native_composite_vision_stage_descriptor")
        ->native_composite_stage_descriptor();
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
