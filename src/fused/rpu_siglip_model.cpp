// rpu_siglip_model.cpp — SigLIP ViT all-layers-once model (v3 FusedModelBase port)
//
// Plan 01-03: ports SigLIPModel from the v2 two-class framework pair to the
// v3 flat FusedModelBase single-class (from Plan 01-01). The port exercises
// D-501 .preload_fn for the first time in production — SigLIP's v2
// `build_preload_subgraph` virtual becomes a plain `emit_preload_weights()`
// member registered via the preload-fn slot inside `static_config()`. The body
// is verbatim from v2 (10 LN/bias × 27 layers + 2 post-LN DMAs).
//
// Pitfall 4 structural mitigation (from the 12-round siglip-multi-group-replay
// debug session): `projector_output_` is now allocated via
// `allocate_tracked_output(...)` — fresh per forward. Pi0.5 embed_prefix
// appends each image's projector output to a Python list before torch.cat;
// a stable-DDR allocation would corrupt earlier slots when later forwards
// overwrite the buffer. `test_siglip_multi_image_regression.py` (this plan)
// guards the regression.
//
// Key transformations vs v2:
//   - Inherit from v3::FusedModelBase (lives in `namespace v3`).
//   - Replace the v2 `build_preload_subgraph()` virtual with
//     `emit_preload_weights()` plain member + register the pointer-to-member
//     on the D-501 preload-fn slot of ModelStaticConfig.
//     Framework opens the weights-graph scope externally; callback body
//     emits only `rpu_launch_*` DMAs (EXT-5: framework owns batch context).
//   - Drop unreachable v2 overrides `build_kv_insert_subgraph` and
//     `build_compute_subgraph` — v3 has neither virtual (D-201).
//   - Replace `buf(name)` at SDPA + KV-insert call sites with
//     `addr_offset(name).value` (Pitfall 3 structural fix via typed SpmOffset).
//   - Access model params via pimpl getters (`num_layers()`, `hidden_size()`,
//     etc.) instead of v2 protected members (`num_layers_`, etc.).
//   - Replace the v2 two-step alloc+track pattern with
//     `projector_output_ = allocate_tracked_output({...})` — framework helper
//     centralizes the fresh-per-forward discipline (Pitfall 4).
//   - `set_weights` ENDS with `invalidate_model_state();` as last non-empty
//     statement (D-503 per-function awk contract).
//
// SEQUENTIAL→KV_FIRST refactor (halo-vit-cache-kvfirst-refactor-design-2026-06-18):
//   - D-508's "ALWAYS single chunk" is RETIRED. SigLIP is now a KV_FIRST model
//     (dynamic_config: ChunkMode::KV_FIRST + InterLayerIO::AUTO) so the encoder
//     can token-chunk and stay under the 8 MB SPM ceiling for HALO's 1564-patch
//     single-image ViT (the old single-chunk encoder hung at ~960 tokens — see
//     the seq-ceiling probe note). The KV_FIRST trio mirrors GemmaModel but is
//     SIMPLIFIED: SigLIP has NO rope and NO attention mask (MASK_NONE).
//   - emit_kv_first_body (Phase 1: LN1+QKV+KV-insert+Q→DDR) +
//     build_layer_subgraph (Phase 2: DDR→Q + full-KV SDPA + O/MLP) +
//     plan_kv_first_chunks + subclass_chunk_size_valid are the new methods.
//
// Scope preserved from v2:
//   - cfg.cross_layer_batch_size = num_layers() (force single group; the
//     Section 3 Round 6 debug session proved multi-group REPLAY is
//     non-deterministic for SigLIP regardless of runtime knob).
//   - 10 PersistentPerLayer + 2 Persistent buffers; no `.preload_callback`
//     on any BufferDecl (SigLIP uses one-shot `.preload_fn` per PATTERNS.md,
//     NOT Gemma-style per-buffer callbacks).
//
// Framework contract: docs/architecture.md#fusedmodelbase-v3--framework-contract

#include "fused_model_base.h"
#include "pi05_kernel_policy.h"
#include "rpu_pi05_vision_fc.h"
#include "rpu_pi05_owner_norm.h"
#include "rpu_kernel_cache.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_spm_allocator.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <array>
#include <optional>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <utility>

using namespace at;
using namespace ::rhino_lkn;

#define DWIDTH 2

constexpr int64_t SIGLIP_CORE_PROFILE_SITE = 2831089885636848922LL;

// Stable native route identities. Norms, fixed activations, debug staging, and
// preload launchers implement fixed SigLIP math/BufferDecl dataflow. The
// selector-bearing launch sites below are consumed from the descriptor.
constexpr int64_t SIGLIP_Q_LINEAR_SITE = 50221634946316412LL;
constexpr int64_t SIGLIP_K_LINEAR_SITE = 2185350348864592434LL;
constexpr int64_t SIGLIP_V_LINEAR_SITE = 4454356738631095076LL;
constexpr int64_t SIGLIP_KV_INSERT_SITE = 4340742943631646068LL;
constexpr int64_t SIGLIP_MINIBATCH_ATTN_SITE = 3592870858110241415LL;
constexpr int64_t SIGLIP_UNIFIED_ATTN_SITE = 871279723585541140LL;
constexpr int64_t SIGLIP_RAW_SPM_ATTN_SITE = 1638421364140696436LL;
constexpr int64_t SIGLIP_O_LINEAR_SITE = 5652000568558546719LL;
constexpr int64_t SIGLIP_ATTN_ALL_REDUCE_SITE = 558716769043213323LL;
constexpr int64_t SIGLIP_FC1_LINEAR_SITE = 628733187506223803LL;
constexpr int64_t SIGLIP_FC2_LINEAR_SITE = 2130098820486441137LL;
constexpr int64_t SIGLIP_MLP_ALL_REDUCE_SITE = 2323671932454929043LL;
constexpr int64_t SIGLIP_PROJECTOR_LINEAR_SITE = 2282358810048681094LL;
constexpr int64_t SIGLIP_PATCH_INPUT_DMA_SITE = 1143336462482724752LL;
constexpr int64_t SIGLIP_PATCH_POSITION_DMA_SITE = 1099468962711310446LL;
constexpr int64_t SIGLIP_PATCH_LINEAR_SITE = 6500914803307480014LL;
constexpr int64_t SIGLIP_PATCH_OUTPUT_DMA_SITE = 7261580027961404620LL;
constexpr int64_t SIGLIP_Q_RESIDENT_SITE = 0x5349474c5153504dll;
constexpr int64_t SIGLIP_FC1_WEIGHT_OUTER_SITE = 0x5349474643313736LL;
constexpr int64_t SIGLIP_FC2_WEIGHT_OUTER_SITE = 0x5349474643323736LL;
// Owner-local ALL_REDUCE selectors retain the existing two semantic sites.
constexpr int64_t SIGLIP_OWNER_LN_ROUTE = 101;
constexpr int64_t SIGLIP_OWNER_LN_COMPACT_ROUTE = 102;

enum class SiglipPatchMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM_MUTABLE = 1,
    SPM_COPY_TO_DDR_MUTABLE = 2,
};

constexpr uint32_t SIGLIP_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
    KV_INSERT_CAP_HYBRID2 | KV_INSERT_CAP_HYBRID3;
constexpr int64_t SIGLIP_KV_FLAG_DDR_MIRROR = 1;
constexpr int64_t SIGLIP_KV_FLAG_RAW_SPM_ONLY = 2;
constexpr int64_t SIGLIP_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED = 1LL << 0;

// =============================================================================
// v5-05 D-07: Fused patch embedding (lifted byte-equal from former
// rpu_fused_patch_embedding.cpp; that file is deleted in C2). Internal helper;
// not registered as a torch op (the fused_patch_embedding op surface is
// removed in C2). The only remaining call site is rpu_siglip_model.cpp's
// SigLIPModel::forward 4D auto-detect path (~L278).
// =============================================================================
namespace {

struct PatchEmbeddingLiveBases {
    static constexpr size_t kMaxPackedImages = 8;
    std::array<uint64_t, kMaxPackedImages> input{};
    uint64_t position = 0;
    uint64_t output = 0;
};

// Writes this image's [num_patches, cout] patch embedding into `out`
// ([1, total_seq, cout]) at sequence row `seq_off`. `img_slot` selects a
// stable per-image live-base slot for the mutable input DMA: the SigLIP-batch
// path packs N images through ONE captured graph, so a single shared live-base
// would make every image read the LAST image's pixels on REPLAY (mutable-DMA
// trap — CLAUDE.md "Fixed DMA trap"). The output/pos_emb live-bases stay shared
// (same packed-tensor base / same registered pos_emb across all N images).
//
// `offsets` are the 6 SPM temp-buffer offsets, allocated ONCE by the caller and
// reused for every image (all images share H/W/cout so one allocation fits all).
// Reusing the SAME offsets — instead of reset_temporary + realloc per image —
// (a) bounds SPM to one image's footprint (3x would OOM) and (b) keeps the
// graph's SPM read/write dependency tracking intact, so image i+1's GEMM
// (writing gemm_out) is serialized after image i's output DMA (reading
// gemm_out). reset_temporary between images clears that tracking → async DMA
// vs GEMM race → non-deterministic REPLAY.
static void fused_patch_embedding(
    const at::Tensor& input_nchw,    // [1, cin_orig, H, W] fp16 RPU DDR
    const at::Tensor& weight,        // [cout, K] fp16 RPU DDR, col-swizzled (1 core)
    const at::Tensor& pos_emb_fused, // [1, num_patches, cout] fp16 RPU DDR
    int64_t kh, int64_t kw,
    int64_t cin_orig, int64_t cin_padded, int64_t cout,
    int64_t strideh, int64_t stridew,
    at::Tensor& out, int64_t seq_off, int img_slot,
    const std::vector<uint32_t>& offsets,
    PatchEmbeddingLiveBases& live_bases,
    const v3::InferenceContext* physical_ctx, bool linear_acc32_)
{
    RECORD_FUNCTION("rpu::fused_patch_embedding", {});

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    int64_t H = input_nchw.size(2);
    int64_t W = input_nchw.size(3);
    int64_t HW = H * W;
    int64_t K = kh * kw * cin_padded;

    int64_t outh = (H - kh) / strideh + 1;
    int64_t outw = (W - kw) / stridew + 1;
    int64_t num_patches = outh * outw;

    uint32_t raw_off      = offsets[0];
    uint32_t padded_off   = offsets[1];
    uint32_t nhwc_off     = offsets[2];
    uint32_t im2col_off   = offsets[3];
    uint32_t gemm_out_off = offsets[4];
    uint32_t pos_emb_off  = offsets[5];

    uint32_t raw_addr      = SPM_ALLOC.addr(0, raw_off);
    uint32_t gemm_out_addr = SPM_ALLOC.addr(0, gemm_out_off);
    uint32_t pos_emb_addr  = SPM_ALLOC.addr(0, pos_emb_off);

    // ===== Step ① DMA raw input DDR → SPM =====
    // Mutable DMA: input_nchw is caller-supplied per-forward, address drifts.
    // Fixed variant would bake BUILD-time addr that sync-only fast-path can't
    // rewrite.
    //
    // NOTE: input_nchw MUST already be DDR-coherent here. The driver
    // (run_packed_patch_embed_core's per-image loop) flushes each image before
    // this call — the flush lives at the batch driver, explicit, rather than
    // buried per-helper. See the rpu_ddr_flush_force there for the why (a
    // torch.cat / CPU-fallback input is left un-flushed → stale DDR).
    // Per-image live-base slot (see fn doc): each packed image bakes a distinct
    // &slot so REPLAY rewrites the correct per-image source addr.
    TORCH_CHECK(img_slot >= 0 &&
                    static_cast<size_t>(img_slot) <
                        PatchEmbeddingLiveBases::kMaxPackedImages,
                "fused_patch_embedding: img_slot ", img_slot,
                " out of range [0,",
                PatchEmbeddingLiveBases::kMaxPackedImages, ")");
    live_bases.input[img_slot] = ::rhino_lkn::RpuGetDevAddr(
        const_cast<c10::Half*>(input_nchw.data_ptr<c10::Half>()));
    if (physical_ctx != nullptr) {
        physical_ctx->consume_physical_route(
            v3::FmbRouteFamily::MUTABLE_DMA,
            SIGLIP_PATCH_INPUT_DMA_SITE,
            static_cast<int64_t>(
                SiglipPatchMutableDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
            /*resolved_flags=*/0,
            {num_patches, cin_orig, cin_padded, kh, kw,
             strideh, stridew, 1}, img_slot);
    }
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &live_bases.input[img_slot],
        /*src_offset_bytes=*/0,
        cin_orig * HW, raw_addr, /*num_cores=*/1);

    // ===== Step ② Pad channels =====
    rpu_launch_pad_channel_spm(
        raw_off, padded_off, /*N=*/1, /*C=*/cin_orig * HW,
        /*pad_front=*/0, /*pad_tail=*/(cin_padded - cin_orig) * HW);

    // ===== Step ③ NCHW → NHWC =====
    rpu_launch_transpose_nchw_to_nhwc_spm(
        padded_off, nhwc_off, (int)cin_padded, (int)H, (int)W);

    // ===== Step ④ im2col =====
    rpu_launch_im2col_spm(
        nhwc_off, im2col_off,
        /*batch=*/1, (int)H, (int)W, (int)cin_padded,
        (int)kh, (int)kw,
        /*padh=*/0, /*padH=*/0, /*padw=*/0, /*padW=*/0,
        (int)strideh, (int)stridew, /*holeh=*/1, /*holew=*/1,
        /*num_cores=*/1);

    // ===== Step ⑤ GEMM =====
    uint32_t im2col_addr = SPM_ALLOC.addr(0, im2col_off);
    if (physical_ctx != nullptr) {
        physical_ctx->consume_physical_route(
            v3::FmbRouteFamily::LINEAR,
            SIGLIP_PATCH_LINEAR_SITE,
            static_cast<int64_t>(v3::FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {num_patches, cout, K, 1, 1, 0}, img_slot);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        im2col_addr, weight, gemm_out_addr,
        num_patches, cout, K,
        /*partition=*/1, /*num_cores=*/1, /*bias_spm_addr=*/0, at::Tensor(), 0, 0,
            /*force_acc32=*/linear_acc32_);

    // ===== Step ⑥ pos_emb add =====
    // Mutable form for symmetry with step ① (pos_emb_fused.data_ptr is
    // actually stable — registered model weight — so fixed would also work).
    // No flush is needed here; unlike the per-forward conditioning input in
    // rpu_adarms_model.cpp, pos_emb_fused is immutable after installation and is
    // patch_emb_pos_emb_, a registered weight whose only host write is the
    // set_weights-time CPU->RPU copy, and rpu_copy_impl already force-flushes
    // that boundary (rpu_tensor_ops.inc). Nothing writes it per forward, so
    // there are no CPU-dirty lines left for this DMA to miss.
    live_bases.position = ::rhino_lkn::RpuGetDevAddr(
        const_cast<c10::Half*>(pos_emb_fused.data_ptr<c10::Half>()));
    if (physical_ctx != nullptr) {
        physical_ctx->consume_physical_route(
            v3::FmbRouteFamily::MUTABLE_DMA,
            SIGLIP_PATCH_POSITION_DMA_SITE,
            static_cast<int64_t>(
                SiglipPatchMutableDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
            /*resolved_flags=*/0,
            {num_patches, cout, 1}, img_slot);
    }
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &live_bases.position,
        /*src_offset_bytes=*/0,
        num_patches * cout, pos_emb_addr, /*num_cores=*/1);

    rpu_launch_eltwise_binary_spm_kernel(
        gemm_out_addr, pos_emb_addr, gemm_out_addr,
        num_patches * cout,
        ValuOpType::ADD, c10::Half(1.0f), /*num_cores=*/1);

    // ===== Step ⑦ DMA output → caller's packed slice =====
    // Mutable DMA: `out` is the caller's packed [1, total_seq, cout] tensor
    // (fresh per forward → dst drifts → mutable, not fixed). All N images share
    // the SAME base, so a single shared output live-base is consistent; each
    // image writes its num_patches rows at byte offset seq_off*cout*DWIDTH.
    rpu_ddr_flush(out.data_ptr<c10::Half>());
    live_bases.output =
        ::rhino_lkn::RpuGetDevAddr(out.data_ptr());
    if (physical_ctx != nullptr) {
        physical_ctx->consume_physical_route(
            v3::FmbRouteFamily::MUTABLE_DMA,
            SIGLIP_PATCH_OUTPUT_DMA_SITE,
            static_cast<int64_t>(
                SiglipPatchMutableDmaRoute::SPM_COPY_TO_DDR_MUTABLE),
            /*resolved_flags=*/0,
            {seq_off * cout * DWIDTH, num_patches * cout}, img_slot);
    }
    rpu_launch_spm_copy_ddr_dma_mutable(
        gemm_out_addr,
        &live_bases.output,
        /*dst_offset_bytes=*/seq_off * cout * DWIDTH,
        num_patches * cout);
}

static void check_siglip_w8a16_scale_lists(
    int64_t n_layers,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w,
    at::TensorList o_w, at::TensorList fc1_w, at::TensorList fc2_w,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList fc1_ws, at::TensorList fc2_ws)
{
    auto check_projection = [&](const at::TensorList& weights,
                                const at::TensorList& scales,
                                const char* name) {
        const bool quantized = !scales.empty();
        if (quantized) {
            TORCH_CHECK(static_cast<int64_t>(scales.size()) == n_layers,
                        "siglip_set_weights: ", name, "_scale.size()=",
                        scales.size(), " != num_layers=", n_layers);
        }
        for (int64_t i = 0; i < n_layers; ++i) {
            const at::Tensor& w = weights[i];
            TORCH_CHECK(w.device().type() == at::kPrivateUse1 && w.is_contiguous(),
                        "siglip_set_weights: ", name, "[", i,
                        "] must be contiguous RPU tensor");
            TORCH_CHECK(w.scalar_type() == at::kHalf || w.scalar_type() == at::kChar,
                        "siglip_set_weights: ", name, "[", i,
                        "] must be fp16 or int8, got ", w.scalar_type());
            if (quantized) {
                const at::Tensor& scale = scales[i];
                TORCH_CHECK(w.scalar_type() == at::kChar,
                            "siglip_set_weights: W8A16 mode requires int8 ",
                            name, "[", i, "], got ", w.scalar_type());
                TORCH_CHECK(scale.defined() && scale.dim() == 1
                            && scale.scalar_type() == at::kHalf
                            && scale.device().type() == at::kPrivateUse1
                            && scale.is_contiguous(),
                            "siglip_set_weights: ", name, "_scale[", i,
                            "] must be 1D contiguous fp16 RPU tensor");
                TORCH_CHECK(scale.numel() == w.size(0),
                            "siglip_set_weights: ", name, "_scale[", i,
                            "].numel()=", scale.numel(),
                            " != output dim=", w.size(0));
            } else {
                TORCH_CHECK(w.scalar_type() != at::kChar,
                            "siglip_set_weights: int8 ", name, "[", i,
                            "] requires a non-empty scale list for this projection");
            }
        }
    };

    check_projection(q_w,   q_ws,   "q_w");
    check_projection(k_w,   k_ws,   "k_w");
    check_projection(v_w,   v_ws,   "v_w");
    check_projection(o_w,   o_ws,   "o_w");
    check_projection(fc1_w, fc1_ws, "fc1_w");
    check_projection(fc2_w, fc2_ws, "fc2_w");
}

}  // namespace

namespace v3 {

// SIGLIP_FIXED_KERNEL_BASIS: remaining launches are fixed math/transport kernels.

// =============================================================================
// SigLIPModel — v3::FusedModelBase subclass (SigLIP ViT Encoder)
// =============================================================================

class SigLIPModel : public FusedModelBase {
public:
    // Per-layer DDR weight tensors (col/row-partition swizzled)
    struct LayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;   // attention
        at::Tensor fc1_w, fc2_w;           // MLP
        at::Tensor q_ws, k_ws, v_ws, o_ws; // W8A16 per-output-channel scales
        at::Tensor fc1_ws, fc2_ws;
    };

    // Per-layer bias/norm tensors stored as at::Tensor to keep DDR pointers
    // alive between set_weights and emit_preload_weights DMA time.
    struct LayerBiasNorm {
        at::Tensor ln1_w, ln1_b, ln2_w, ln2_b;       // LayerNorm gamma/beta
        at::Tensor q_b, k_b, v_b, o_b;                // attention biases
        at::Tensor fc1_b, fc2_b;                       // MLP biases
    };

    explicit SigLIPModel(std::optional<bool> linear_acc32 = std::nullopt)
        : configured_linear_acc32_(linear_acc32), linear_acc32_(linear_acc32.value_or(false)) {}

private:
    const std::optional<bool> configured_linear_acc32_;
    const bool linear_acc32_;
    FmbLinearAccumulationPolicy linear_accumulation_policy() const {
        return linear_acc32_ ? FmbLinearAccumulationPolicy::ACC32
                             : FmbLinearAccumulationPolicy::ACC16;
    }
public:

    void set_execution_cores(int64_t cores) {
        TORCH_CHECK(cores == 4 || cores == 8,
                    "SigLIP execution cores must be 4 or 8");
        TORCH_CHECK(layer_weights_.empty(),
                    "SigLIP execution cores must precede weight installation");
        set_execution_core_count(static_cast<int>(cores));
    }

    std::vector<int64_t> execution_topology() const {
        TORCH_CHECK(num_layers() > 0 && !layer_weights_.empty(),
                    "SigLIP topology requires installed model weights");
        return {1, num_cores(), attn_tp(), mlp_tp(), 1, 8,
                logical_intermediate_size_, intermediate_size()};
    }

    std::vector<int64_t> core_profile_arguments() const {
        auto args = execution_topology();
        args.push_back(reduced_w8_projection_mask_);
        return args;
    }

    DecoderExecutionTopology resolve_model_execution_topology(
            int64_t nq, int64_t nkv, int64_t hd, int64_t h,
            int64_t intermediate) const override {
        if (num_cores() == 8)
            return FusedModelBase::resolve_model_execution_topology(
                nq, nkv, hd, h, intermediate);
        TORCH_CHECK(num_cores() == 4 && nq == 16 && nkv == 16 &&
                        hd == 80 && h == 1152 && intermediate == 4352,
                    "no admitted reduced-core Pi0.5 SigLIP profile");
        return {4, 4, 4};
    }




    void set_configured_chunk_size(int64_t chunk_size) {
        TORCH_CHECK(chunk_size == 0
                        || (chunk_size >= 16 && chunk_size % 16 == 0),
                    "SigLIP chunk size must be 0 (auto) or a positive "
                    "multiple of 16, got ", chunk_size);
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "SigLIP chunk size must be set before the first forward");
        configured_chunk_size_ = chunk_size;
        invalidate_model_state();
    }

    int64_t resolve_vision_chunk_size(
        int64_t seq_len, int64_t packed_image_count)
    {
        require_pi05_vision_owner_ln_profile(seq_len, packed_image_count);
        configure_chunk_for_shape(seq_len, packed_image_count);
        return resolve_chunk_size_for_shape(
            seq_len, /*position=*/0, std::nullopt, /*is_causal=*/false);
    }

    std::vector<int64_t> resolve_stage_domain(
        int64_t seq_len, int64_t packed_image_count,
        bool external_patch_prologue)
    {
        TORCH_CHECK(num_layers() > 0,
                    "RPU_PLANNER_REJECT:CAPABILITY: SigLIP weights are not "
                    "initialized");
        require_pi05_vision_owner_ln_profile(seq_len, packed_image_count);
        configure_chunk_for_shape(seq_len, packed_image_count);
        TORCH_CHECK(seq_len % packed_image_count == 0,
                    "RPU_PLANNER_REJECT:CAPABILITY: SigLIP packed sequence "
                    "must divide into image spans");
        const int64_t per_image = seq_len / packed_image_count;
        std::vector<ChunkInfo> patch_chunks;
        std::vector<FmbExecutionSpan> image_spans;
        patch_chunks.reserve(packed_image_count);
        image_spans.reserve(packed_image_count);
        for (int64_t image = 0; image < packed_image_count; ++image) {
            const int64_t offset = image * per_image;
            patch_chunks.push_back({
                static_cast<int>(image), offset, per_image,
                offset + per_image});
            image_spans.push_back({offset, per_image, image});
        }
        const FmbStageBoundaryPolicies boundary_policies{
            FmbSpanBoundaryPolicy::KEEP_LOCAL,
            FmbSpanBoundaryPolicy::ALLOW_CROSS,
            FmbSpanBoundaryPolicy::ALLOW_CROSS};
        const bool saved_external_patch_prologue =
            planning_external_patch_prologue_;
        planning_external_patch_prologue_ = external_patch_prologue;
        auto restore_external_patch_prologue = c10::make_scope_exit([&] {
            planning_external_patch_prologue_ =
                saved_external_patch_prologue;
        });
        // Packed minibatch SDPA requires one full-sequence QKV/compute chunk;
        // publish only that domain; single-image retains its cold request.
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                seq_len, /*position=*/0,
                /*attention_mask=*/std::nullopt, /*is_causal=*/false,
                patch_chunks, image_spans, boundary_policies,
                /*requested_chunk_size=*/packed_image_count > 1
                    ? seq_len : configured_chunk_size_,
                /*logical_len=*/seq_len));
    }

    // ========================================================================
    // set_weights — stores all per-layer and global weights
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
        const at::Tensor& post_ln_w, const at::Tensor& post_ln_b,
        const at::Tensor& proj_w, const at::Tensor& proj_b,
        int64_t num_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        int64_t projection_dim, double eps,
        at::TensorList q_ws_list, at::TensorList k_ws_list,
        at::TensorList v_ws_list, at::TensorList o_ws_list,
        at::TensorList fc1_ws_list, at::TensorList fc2_ws_list)
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "siglip_set_weights: empty weight lists");
        TORCH_CHECK(num_heads > 0 && head_dim > 0 && hidden_size > 0
                    && intermediate_size > 0 && projection_dim > 0,
                    "siglip_set_weights: dim params must be positive");

        // All 16 per-layer lists must have the same length
        auto check_list = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "siglip_set_weights: ", n, ".size()=", l.size(),
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

        // Global tensor checks — post_ln/proj are always fp16, DMA'd as Half
        // (add fp16 + PrivateUse1 + contiguous).
        auto check_global_fp16_rpu = [&](const at::Tensor& t, const char* name) {
            TORCH_CHECK(t.defined(), "siglip_set_weights: ", name, " must be defined");
            TORCH_CHECK(t.scalar_type() == at::kHalf
                        && t.device().type() == at::kPrivateUse1
                        && t.is_contiguous(),
                        "siglip_set_weights: ", name,
                        " must be fp16 contiguous RPU tensor, got dtype=",
                        t.scalar_type(), " device=", t.device().type(),
                        " contiguous=", t.is_contiguous());
        };
        check_global_fp16_rpu(post_ln_w, "post_ln_w");
        check_global_fp16_rpu(post_ln_b, "post_ln_b");
        check_global_fp16_rpu(proj_w,    "proj_w");
        check_global_fp16_rpu(proj_b,    "proj_b");

        // Per-layer defined/rank checks. require_fp16_rpu adds the fp16 +
        // PrivateUse1 + contiguous guard for the ALWAYS-fp16 tensors (norms,
        // biases) that are later DMA'd via data_ptr<c10::Half>()
        // weight-validate fix. The main q/k/v/o/fc projection weights skip it
        // (they may be int8 W8A16 and are validated by check_projection above).
        auto check_defined_rank = [&](const at::TensorList& list,
                                      const char* name, int64_t expected_rank,
                                      bool require_fp16_rpu = false) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "siglip_set_weights: ", name, "[", i, "] is undefined");
                TORCH_CHECK(list[i].dim() == expected_rank,
                            "siglip_set_weights: ", name, "[", i, "] must be ",
                            expected_rank, "D, got ", list[i].dim(), "D");
                if (require_fp16_rpu) {
                    TORCH_CHECK(list[i].scalar_type() == at::kHalf
                                && list[i].device().type() == at::kPrivateUse1
                                && list[i].is_contiguous(),
                                "siglip_set_weights: ", name, "[", i,
                                "] must be fp16 contiguous RPU tensor, got dtype=",
                                list[i].scalar_type(), " device=",
                                list[i].device().type(), " contiguous=",
                                list[i].is_contiguous());
                }
            }
        };
        check_defined_rank(q_w_list,   "q_w_list",   2);
        check_defined_rank(k_w_list,   "k_w_list",   2);
        check_defined_rank(v_w_list,   "v_w_list",   2);
        check_defined_rank(o_w_list,   "o_w_list",   2);
        check_defined_rank(fc1_w_list, "fc1_w_list", 2);
        check_defined_rank(fc2_w_list, "fc2_w_list", 2);
        // Norm weights: 1D [hidden_size] — always fp16, DMA'd as Half.
        check_defined_rank(ln1_w_list, "ln1_w_list", 1, /*require_fp16_rpu=*/true);
        check_defined_rank(ln1_b_list, "ln1_b_list", 1, /*require_fp16_rpu=*/true);
        check_defined_rank(ln2_w_list, "ln2_w_list", 1, /*require_fp16_rpu=*/true);
        check_defined_rank(ln2_b_list, "ln2_b_list", 1, /*require_fp16_rpu=*/true);
        // Biases: 1D — always fp16, DMA'd as Half.
        check_defined_rank(q_b_list,   "q_b_list",   1, /*require_fp16_rpu=*/true);
        check_defined_rank(k_b_list,   "k_b_list",   1, /*require_fp16_rpu=*/true);
        check_defined_rank(v_b_list,   "v_b_list",   1, /*require_fp16_rpu=*/true);
        check_defined_rank(o_b_list,   "o_b_list",   1, /*require_fp16_rpu=*/true);
        check_defined_rank(fc1_b_list, "fc1_b_list", 1, /*require_fp16_rpu=*/true);
        check_defined_rank(fc2_b_list, "fc2_b_list", 1, /*require_fp16_rpu=*/true);
        check_siglip_w8a16_scale_lists(
            N,
            q_w_list, k_w_list, v_w_list, o_w_list, fc1_w_list, fc2_w_list,
            q_ws_list, k_ws_list, v_ws_list, o_ws_list, fc1_ws_list, fc2_ws_list);

        int64_t precision_mask = 0;
        if (num_cores() != 8) {
            TORCH_CHECK(num_cores() == 4 && N == 27 && num_heads == 16 &&
                            head_dim == 80 && hidden_size == 1152 &&
                            intermediate_size == 4352 && projection_dim == 2048,
                        "no admitted reduced-core Pi0.5 SigLIP geometry");
            TORCH_CHECK(proj_w.sizes() == at::IntArrayRef({2048, 1152}) &&
                            proj_b.numel() == 2048 &&
                            post_ln_w.numel() == 1152 && post_ln_b.numel() == 1152,
                        "Pi0.5 SigLIP global projection/norm shape mismatch");
            const std::array<std::array<int64_t, 2>, 6> expected{{
                {1280, 1152}, {1280, 1152}, {1280, 1152},
                {1152, 1280}, {4352, 1152}, {1152, 4352}}};
            for (int64_t i = 0; i < N; ++i) {
                size_t j = 0;
                int64_t layer_mask = 0;
                for (const auto* tensor : {&q_w_list[i], &k_w_list[i],
                        &v_w_list[i], &o_w_list[i], &fc1_w_list[i],
                        &fc2_w_list[i]}) {
                    TORCH_CHECK((tensor->scalar_type() == at::kHalf ||
                                 tensor->scalar_type() == at::kChar) &&
                                tensor->size(0) == expected[j][0] &&
                                tensor->size(1) == expected[j][1],
                                "Pi0.5 SigLIP projection shape/precision mismatch");
                    if (tensor->scalar_type() == at::kChar) layer_mask |= 1LL << j;
                    ++j;
                }
                if (i == 0) precision_mask = layer_mask;
                TORCH_CHECK(layer_mask == precision_mask,
                            "Pi0.5 SigLIP precision scope must match across layers");
                for (const auto* tensor : {&ln1_w_list[i], &ln1_b_list[i],
                        &ln2_w_list[i], &ln2_b_list[i], &o_b_list[i],
                        &fc2_b_list[i]})
                    TORCH_CHECK(tensor->numel() == 1152,
                                "Pi0.5 SigLIP norm/row-bias shape mismatch");
                for (const auto* tensor : {&q_b_list[i], &k_b_list[i], &v_b_list[i]})
                    TORCH_CHECK(tensor->numel() == 1280,
                                "Pi0.5 SigLIP QKV bias shape mismatch");
                TORCH_CHECK(fc1_b_list[i].numel() == 4352,
                            "Pi0.5 SigLIP FC1 bias shape mismatch");
            }
        }
        logical_intermediate_size_ =
            hidden_size == 1152 && intermediate_size == 4352 ? 4304 : intermediate_size;
        reduced_w8_projection_mask_ = precision_mask;
        // These routes change only FP16 activation movement/collectives. Keep
        // the original uniformly FP16 or uniformly W8 weights and scale owners.
        if (pi05_vision_owner_ln_opt_in_) {
            TORCH_CHECK(num_cores() == 8 && pi05_ring_xor3_opt_in_ && eps == 1e-6,
                "Pi Vision owner-LN requires eight cores, XOR3 and epsilon 1e-6");
            // Image count is not known at set_weights. Shape planning below
            // requires the exact selected pair before any graph submission.
            require_pi05_vision_owner_ln_payloads(0);
            TORCH_CHECK(proj_w.sizes() == at::IntArrayRef({2048, 1152}) &&
                            proj_b.numel() == 2048 && post_ln_w.numel() == 1152 &&
                            post_ln_b.numel() == 1152,
                        "Pi Vision owner-LN global projection/norm shape mismatch");
            for (int64_t i = 0; i < N; ++i) {
                TORCH_CHECK(ln1_w_list[i].numel() == 1152 && ln1_b_list[i].numel() == 1152 &&
                                ln2_w_list[i].numel() == 1152 && ln2_b_list[i].numel() == 1152,
                            "Pi Vision owner-LN requires exact FP16 affine norm owners");
            }
        }
        if (pi05_vision_kv_spm_only_opt_in_ || pi05_vision_owner_ln_opt_in_) {
            TORCH_CHECK(num_cores() == 8 && N == 27 && num_heads == 16 && head_dim == 80 &&
                    hidden_size == 1152 && intermediate_size == 4352 &&
                    projection_dim == 2048 && hidden_size / num_heads == 72,
                "Pi Vision residency requires the exact FP16/W8 Pi SigLIP profile");
            auto exact = [](const at::Tensor& tensor, at::ScalarType dtype,
                            at::IntArrayRef shape) {
                return tensor.defined() && tensor.device().type() == at::kPrivateUse1 &&
                    tensor.scalar_type() == dtype && tensor.is_contiguous() && tensor.sizes() == shape;
            };
            const auto dtype = q_w_list.front().scalar_type();
            TORCH_CHECK(dtype == at::kChar || dtype == at::kHalf,
                        "Pi Vision residency requires genuine W8 or FP16 weights");
            const bool w8 = dtype == at::kChar;
            for (const auto* scales : {&q_ws_list, &k_ws_list, &v_ws_list,
                                      &o_ws_list, &fc1_ws_list, &fc2_ws_list}) {
                TORCH_CHECK(w8 ? scales->size() == 27 : scales->empty(),
                            "Pi Vision residency requires W8 scales only with W8 weights");
            }
            for (int64_t i = 0; i < N; ++i) {
                TORCH_CHECK(exact(q_w_list[i], dtype, {1280, 1152}) &&
                        exact(k_w_list[i], dtype, {1280, 1152}) &&
                        exact(v_w_list[i], dtype, {1280, 1152}) &&
                        exact(o_w_list[i], dtype, {1152, 1280}) &&
                        exact(fc1_w_list[i], dtype, {4352, 1152}) &&
                        exact(fc2_w_list[i], dtype, {1152, 4352}),
                    "Pi Vision residency requires uniform exact projection owners at layer ", i);
                TORCH_CHECK(!w8 || (exact(q_ws_list[i], at::kHalf, {1280}) &&
                        exact(k_ws_list[i], at::kHalf, {1280}) &&
                        exact(v_ws_list[i], at::kHalf, {1280}) &&
                        exact(o_ws_list[i], at::kHalf, {1152}) &&
                        exact(fc1_ws_list[i], at::kHalf, {4352}) &&
                        exact(fc2_ws_list[i], at::kHalf, {1152})),
                    "Pi Vision residency scale mismatch at layer ", i);
            }
        }

        // The opt-in is exact and checked before mutating the installed owner.
        if (pi05_vision_fc_weight_outer_opt_in_) {
            TORCH_CHECK(N==27 && num_heads==16 && head_dim==80 && hidden_size==1152 &&
                intermediate_size==4352 && projection_dim==2048,
                "Pi Vision FC weight-outer requires the exact W8 Pi SigLIP profile");
            auto exact=[](const at::Tensor& t,at::ScalarType dtype,at::IntArrayRef shape) {
                return t.defined() && t.device().type()==at::kPrivateUse1 &&
                    t.scalar_type()==dtype && t.is_contiguous() && t.sizes()==shape;
            };
            TORCH_CHECK(fc1_ws_list.size()==27 && fc2_ws_list.size()==27,
                "Pi Vision FC weight-outer requires all original scale owners");
            for(int64_t i=0;i<N;++i) {
                TORCH_CHECK(q_w_list[i].scalar_type()==at::kChar && k_w_list[i].scalar_type()==at::kChar &&
                    v_w_list[i].scalar_type()==at::kChar && o_w_list[i].scalar_type()==at::kChar &&
                    exact(fc1_w_list[i],at::kChar,{4352,1152}) && exact(fc2_w_list[i],at::kChar,{1152,4352}) &&
                    exact(fc1_ws_list[i],at::kHalf,{4352}) && exact(fc2_ws_list[i],at::kHalf,{1152}) &&
                    exact(fc1_b_list[i],at::kHalf,{4352}) && exact(fc2_b_list[i],at::kHalf,{1152}),
                    "Pi Vision FC weight-outer weight/scale/original bias mismatch at layer ",i);
            }
        }

        // SigLIP uses MHA: num_kv_heads == num_q_heads
        set_model_params(num_heads, num_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;
        projection_dim_ = projection_dim;

        // orig_head_dim = hidden_size / num_heads (= 72 for SigLIP, NOT from
        // padded weight shape which would give 80). v2 reference:
        // rpu_siglip_fused_encoder_layer.cpp:584.
        orig_head_dim_ = hidden_size / num_heads;

        // Site9 (KV_FIRST): per-core local Q head count for q_ddr_buf_ staging
        // and the Phase1 Q→DDR / Phase2 DDR→Q scatter/gather (mirror Gemma's
        // local_q_heads_ = num_q_heads / attn_tp()). SigLIP fixes tp = num_cores().
        local_q_heads_ = num_q_heads() / num_cores();

        // Build layer weights
        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                fc1_w_list[i], fc2_w_list[i],
                !q_ws_list.empty()   ? q_ws_list[i]   : at::Tensor(),
                !k_ws_list.empty()   ? k_ws_list[i]   : at::Tensor(),
                !v_ws_list.empty()   ? v_ws_list[i]   : at::Tensor(),
                !o_ws_list.empty()   ? o_ws_list[i]   : at::Tensor(),
                !fc1_ws_list.empty() ? fc1_ws_list[i] : at::Tensor(),
                !fc2_ws_list.empty() ? fc2_ws_list[i] : at::Tensor(),
            });
        }

        // Build bias/norm tensor storage (keeps DDR pointers alive for DMA)
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

        // Global weights
        post_ln_w_ = post_ln_w;
        post_ln_b_ = post_ln_b;
        proj_w_ = proj_w;
        proj_b_ = proj_b;

        invalidate_model_state();  // D-503: last non-empty statement of set_weights
    }

    // ========================================================================
    // set_patch_emb_params — optional patch embedding configuration
    // ========================================================================
    void set_patch_emb_params(const at::Tensor& weight, const at::Tensor& pos_emb,
                              int64_t kernel_size, int64_t stride)
    {
        TORCH_CHECK(weight.defined() && weight.dim() == 2,
                    "siglip_set_patch_emb: weight must be defined 2D");
        TORCH_CHECK(pos_emb.defined(),
                    "siglip_set_patch_emb: pos_emb must be defined");
        if (num_cores() != 8) {
            TORCH_CHECK(kernel_size == 14 && stride == 14 &&
                            weight.sizes() == at::IntArrayRef({1152, 3136}) &&
                            pos_emb.sizes() == at::IntArrayRef({1, 256, 1152}),
                        "Pi0.5 SigLIP4 requires the fixed 224x224 patch profile");
            for (const auto* tensor : {&weight, &pos_emb})
                TORCH_CHECK(tensor->scalar_type() == at::kHalf &&
                                tensor->device().type() == at::kPrivateUse1 &&
                                tensor->is_contiguous(),
                            "Pi0.5 SigLIP patch/position weights must remain FP16 RPU");
        }

        patch_emb_weight_ = weight;
        patch_emb_pos_emb_ = pos_emb;
        patch_kernel_size_ = kernel_size;
        patch_stride_ = stride;

        // Derive params from stored tensors
        pe_kh_ = kernel_size;
        pe_kw_ = kernel_size;
        pe_strideh_ = stride;
        pe_stridew_ = stride;
        pe_cin_orig_ = 3;  // RGB input
        pe_cin_padded_ = weight.size(1) / (pe_kh_ * pe_kw_);
        pe_cout_ = weight.size(0);

        has_patch_emb_ = true;
    }

    // ========================================================================
    // forward — main entry point
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        at::IntArrayRef planned_stage_descriptor)
    {
        TORCH_CHECK(num_layers() > 0,
                    "SigLIPModel::forward called before set_weights");
        TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                    "SigLIPModel::forward: input must be on RPU device");
        TORCH_CHECK(input.is_contiguous(),
                    "SigLIPModel::forward: input must be contiguous");
        TORCH_CHECK(input.dim() == 3 || input.dim() == 4,
                    "SigLIP forward expects 3D [B,S,H] or 4D [B,C,H,W] input, got ",
                    input.dim(), "D");
        if (input.dim() == 4) {
            TORCH_CHECK(has_patch_emb_,
                        "4D SigLIP input requires patch embedding params — "
                        "call siglip_model_set_patch_emb before forward with 4D input");
        }

        at::Tensor hidden_states;
        int64_t packed_image_count = 1;

        // D-507: Auto-detect input shape.
        //  - 4D [N,C,H,W]: SigLIP-batch packs the N camera images into one
        //    [1,N*256,hidden] hidden via run_packed_patch_embed (all inside this
        //    one capture); image_batch_count_ = N drives the per-image minibatch
        //    SDPA. N=1 is the legacy single-image path, byte-identical.
        //  - 3D [1,S,hidden]: pre-embedded single hidden (image_batch_count_=1).
        if (input.dim() == 4 && has_patch_emb_) {
            TORCH_CHECK(
                !planned_stage_descriptor.empty(),
                "SigLIP patch-embedding forward requires its native A6 "
                "stage descriptor");
            const int64_t patch_rows =
                input.size(0) *
                ((input.size(2) - pe_kh_) / pe_strideh_ + 1) *
                ((input.size(3) - pe_kw_) / pe_stridew_ + 1);
            TORCH_CHECK(num_cores() == 8 ||
                            (input.sizes() == at::IntArrayRef({1, 3, 224, 224}) &&
                             input.scalar_type() == at::kHalf && patch_rows == 256),
                        "Pi0.5 SigLIP4 requires serial single-image M256 execution");
            // Validate the descriptor against this request's actual cameras,
            // even if a different admitted shape was the last planned/replayed one.
            image_batch_count_ = input.size(0);
            const auto prepared_planned = prepare_stage_candidate(planned_stage_descriptor);
            const auto& planned = prepared_planned->candidate();
            validate_pi05_vision_kv_spm_only_policy(
                planned, patch_rows, input.size(0));
            validate_pi05_vision_owner_ln_policy(
                planned, patch_rows, input.size(0));
            require_pi05_vision_fc_profile(patch_rows,input.size(0));
            validate_pi05_vision_fc_manifest(planned.physical_manifest);
            if(pi05_vision_fc_weight_outer_opt_in_)
                require_pi05_vision_fc_schedule(planned.stage_plan);
            begin_external_physical_manifest_prologue(
                planned.physical_manifest, patch_rows, /*position=*/0);
            auto cancel_patch_prologue = c10::make_scope_exit(
                [&] { cancel_external_physical_manifest_prologue(); });
            hidden_states = run_packed_patch_embed(
                input, /*consume_external_prologue=*/true);
            packed_image_count = image_batch_count_;
            at::Tensor result = forward_packed(
                hidden_states, packed_image_count, k_caches, v_caches,
                planned_stage_descriptor);
            cancel_patch_prologue.release();
            return result;
        } else {
            hidden_states = input;
        }

        return forward_packed(
            hidden_states, packed_image_count, k_caches, v_caches,
            planned_stage_descriptor);
    }

    // ========================================================================
    // forward_multi — all-RPU SigLIP-batch entry. Takes the N camera images as
    // a LIST of [1,C,H,W] tensors (no torch.cat upstream), packs them into one
    // [1, N*256, hidden] hidden via run_packed_patch_embed_list, then runs the
    // identical encoder + projector tail as forward(). image_batch_count_ = N
    // (set by the packer) drives the per-image minibatch SDPA in
    // build_layer_subgraph — bit-exact vs forward()'s 4D torch.cat'd path
    // (test_siglip_batch_3image.py), just without the cat/slice round-trip.
    // ========================================================================
    at::Tensor forward_multi(
        at::TensorList images,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        at::IntArrayRef planned_stage_descriptor)
    {
        TORCH_CHECK(num_layers() > 0,
                    "SigLIPModel::forward_multi called before set_weights");
        TORCH_CHECK(has_patch_emb_,
                    "SigLIPModel::forward_multi requires patch embedding params — "
                    "call siglip_model_set_patch_emb before forward_multi");
        TORCH_CHECK(images.size() > 0,
                    "SigLIPModel::forward_multi: empty image list");
        for (const auto& im : images) {
            TORCH_CHECK(im.device().type() == at::kPrivateUse1,
                        "SigLIPModel::forward_multi: each image must be on RPU device");
            TORCH_CHECK(im.is_contiguous(),
                        "SigLIPModel::forward_multi: each image must be contiguous");
            TORCH_CHECK(im.dim() == 4,
                        "SigLIPModel::forward_multi: each image must be 4D [1,C,H,W], "
                        "got ", im.dim(), "D");
        }

        // Patch-embed + pack all N images into [1, N*256, hidden] (sets
        // image_batch_count_ = N for the per-image minibatch SDPA).
        TORCH_CHECK(
            !planned_stage_descriptor.empty(),
            "SigLIP patch-embedding forward requires its native A6 stage "
            "descriptor");
        const at::Tensor& first_image = images.front();
        const int64_t patch_rows =
            static_cast<int64_t>(images.size()) *
            ((first_image.size(2) - pe_kh_) / pe_strideh_ + 1) *
            ((first_image.size(3) - pe_kw_) / pe_stridew_ + 1);
        TORCH_CHECK(num_cores() == 8 ||
                        (images.size() == 1 && patch_rows == 256 &&
                         first_image.sizes() == at::IntArrayRef({1, 3, 224, 224}) &&
                         first_image.scalar_type() == at::kHalf),
                    "Pi0.5 SigLIP4 requires serial single-image M256 execution");
        image_batch_count_ = static_cast<int64_t>(images.size());
        const auto prepared_planned = prepare_stage_candidate(planned_stage_descriptor);
        const auto& planned = prepared_planned->candidate();
        validate_pi05_vision_kv_spm_only_policy(
            planned, patch_rows, static_cast<int64_t>(images.size()));
        validate_pi05_vision_owner_ln_policy(
            planned, patch_rows, static_cast<int64_t>(images.size()));
        require_pi05_vision_fc_profile(patch_rows,static_cast<int64_t>(images.size()));
        validate_pi05_vision_fc_manifest(planned.physical_manifest);
        if(pi05_vision_fc_weight_outer_opt_in_)
            require_pi05_vision_fc_schedule(planned.stage_plan);
        begin_external_physical_manifest_prologue(
            planned.physical_manifest, patch_rows, /*position=*/0);
        auto cancel_patch_prologue = c10::make_scope_exit(
            [&] { cancel_external_physical_manifest_prologue(); });
        at::Tensor hidden_states = run_packed_patch_embed_list(
            images, /*consume_external_prologue=*/true);
        at::Tensor result = forward_packed(
            hidden_states, image_batch_count_, k_caches, v_caches,
            planned_stage_descriptor);
        cancel_patch_prologue.release();
        return result;
    }

    at::Tensor patch_embed(const at::Tensor& input) {
        TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                    "SigLIPModel::patch_embed: input must be on RPU device");
        TORCH_CHECK(input.is_contiguous(),
                    "SigLIPModel::patch_embed: input must be contiguous");
        TORCH_CHECK(input.dim() == 4,
                    "SigLIPModel::patch_embed expects 4D [B,C,H,W] input, got ",
                    input.dim(), "D");
        TORCH_CHECK(has_patch_emb_,
                    "SigLIPModel::patch_embed requires patch embedding params");
        return run_packed_patch_embed(
            input, /*consume_external_prologue=*/false);
    }

    at::Tensor patch_embed_multi(at::TensorList images) {
        TORCH_CHECK(has_patch_emb_,
                    "SigLIPModel::patch_embed_multi requires patch embedding params");
        TORCH_CHECK(images.size() > 0,
                    "SigLIPModel::patch_embed_multi: empty image list");
        for (const auto& im : images) {
            TORCH_CHECK(im.device().type() == at::kPrivateUse1,
                        "SigLIPModel::patch_embed_multi: each image must be on RPU");
            TORCH_CHECK(im.is_contiguous(),
                        "SigLIPModel::patch_embed_multi: each image must be contiguous");
            TORCH_CHECK(im.dim() == 4 && im.size(0) == 1,
                        "SigLIPModel::patch_embed_multi: each image must be [1,C,H,W]");
        }
        return run_packed_patch_embed_list(
            images, /*consume_external_prologue=*/false);
    }

    at::Tensor forward_packed(
        const at::Tensor& hidden_states,
        int64_t packed_image_count,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        at::IntArrayRef planned_stage_descriptor)
    {
        TORCH_CHECK(!planned_stage_descriptor.empty(),
                    "SigLIP production forward requires its native A6 stage "
                    "descriptor");
        TORCH_CHECK(hidden_states.dim() == 3,
                    "SigLIPModel::forward_packed: hidden input must be 3D, got ",
                    hidden_states.dim(), "D");
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "SigLIPModel::forward_packed: hidden input must be on RPU");
        TORCH_CHECK(hidden_states.is_contiguous(),
                    "SigLIPModel::forward_packed: hidden input must be contiguous");
        TORCH_CHECK(packed_image_count > 0,
                    "SigLIPModel::forward_packed: packed_image_count must be positive");
        image_batch_count_ = packed_image_count;

        int64_t seq_len = hidden_states.size(1);

        validate_pi05_vision_owner_ln_policy(
            decode_fmb_prefill_stage_candidate(planned_stage_descriptor),
            seq_len, packed_image_count);

        if (pi05_vision_kv_spm_only_opt_in_) {
            validate_pi05_vision_kv_spm_only_policy(
                decode_fmb_prefill_stage_candidate(planned_stage_descriptor),
                seq_len, packed_image_count);
        }

        // Site5 (KV_FIRST): seq_len_ is this forward's full sequence length (the SDPA
        // kv_seq_len). The STABLE per-shape Q staging slot is allocated BELOW, AFTER
        // the shape-validity checks (cold-panel re-review #edge: don't allocate before
        // validation — an invalid packed shape would otherwise leak a slot).
        seq_len_ = seq_len;

        // Minibatch (N>1 packed images) mutual-exclusion with chunking: the
        // per-image minibatch SDPA launcher asserts a SINGLE chunk spanning the
        // whole packed sequence. For N>1, pin the compute chunk size to the full
        // sequence via the override; plan_kv_first_chunks then keeps QKV
        // identical to that plan. This bypasses the auto-scan, exactly like the
        // retired D-508 single-chunk path did. N==1 (HALO single-image) leaves
        // the override at 0 so the auto-scan can pick a fitting comp_cs and
        // token-chunk the 1564 path.
        //
        // the override is floored to (override/16)*16
        // (fused_model_base.cpp:487). A packed seq not a multiple of 16 — or not
        // divisible by image_batch_count_ — would split a TAIL chunk, violating the
        // single-block minibatch assertion (TORCH_CHECK(chunk.len==seq_len_) in the
        // KV_FIRST body below) deep on-board where it is hard to root-cause. Fail
        // fast HERE with a clear shape error. SigLIP packs per-image npp(=256)
        // tokens so seq_len = N*256 is 16-aligned in practice; this guards the
        // invariant the forced-single-block override silently relies on.
        configure_chunk_for_shape(seq_len, image_batch_count_);

        // Site5 (KV_FIRST): shape now validated → ensure a STABLE Q staging slot for
        // THIS forward (mirror GemmaModel). Keyed by {seq_len, q_width =
        // local_q_heads_*head_dim()} → created once, NEVER reallocated, so a smaller
        // shape's already-captured graph keeps a valid fixed-DMA address after a
        // larger-seq forward, and a set_weights head-dim change gets a FRESH slot
        // (see q_ddr_slots_ declaration). Mirrors
        // temp_ddr_slots_; bounded by the handle's shape set (cap catches runaway).
        const int64_t q_width = local_q_heads_ * head_dim();
        const std::pair<int64_t, int64_t> q_key{seq_len, q_width};
        if (q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
            TORCH_CHECK(q_ddr_slots_.size() < 128,
                        "SigLIPModel: q_ddr_slots_ exceeded 128 distinct (seq_len,width) "
                        "shapes on one handle — likely a runaway shape loop (per-shape Q "
                        "staging is never freed for replay-address stability)");
            q_ddr_slots_.emplace(q_key, at::empty(
                {int64_t{8}, seq_len, q_width},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1)));
        }

        // GUARD (design note §2 三件套 ①): host-side per-chunk SPM budget
        // pre-check BEFORE dispatch. The probe note proved a naive single-chunk
        // 1564-token forward exhausts the 8 MB SPM block and enters an infinite
        // allocator retry that POISONS the whole RPU board (run-queue wedged
        // until reboot). The framework's auto chunk-size scan
        // (compute_chunks_impl) already rejects an over-budget seq with a clear
        // TORCH_CHECK on the same planner path used by the per-forward override,
        // so this is a defense-in-depth fail-fast: if even the SMALLEST
        // legal chunk (cs=16) cannot fit, refuse loudly here rather than relying
        // solely on the framework's [Fix #4] min-chunk check. The framework owns
        // the authoritative budget gate; this guard only converts the worst case
        // into an explicit, readable error at the model boundary.
        {
            LayoutContext probe_ctx;
            probe_ctx.chunk_size = 16;  // smallest legal chunk
            probe_ctx.num_layers = num_layers();
            const std::vector<BufferDecl> decls = declare_buffers(probe_ctx);
            const int64_t peak = detail::estimate_temporary_total(decls);
            const int64_t usable = static_cast<int64_t>(
                SpmAllocator::SPM_PLANNING_BUDGET);
            TORCH_CHECK(peak <= usable,
                        "SigLIPModel::forward_packed: even the minimum chunk "
                        "(cs=16) per-chunk SPM peak (", peak, " B) exceeds the "
                        "usable SPM budget (", usable, " B). Refusing to dispatch "
                        "— a naive oversize forward would wedge the RPU board. "
                        "seq_len=", seq_len, ", hidden=", hidden_size(), ".");
        }

        // Fast/deep REPLAY skips build_layer_subgraph entirely. Allocate the
        // externally returned tensor here on EVERY forward and refresh the
        // owner-local mutable DMA base before FMB can take that skip path.
        // Otherwise a retained previous Vision return is overwritten by the
        // next request even though its Tensor/DDR owner is still alive.
        projector_output_ = allocate_tracked_output({1, seq_len_, projection_dim_});
        projector_out_live_base_ =
            ::rhino_lkn::RpuGetDevAddr(projector_output_.data_ptr());

        at::Tensor result = run_all_layers(
            hidden_states, k_caches, v_caches,
            /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false,
            /*planned_chunk_size=*/0, planned_stage_descriptor);

        // Last layer writes the fresh projector_output_ (Pitfall 4); flush for
        // CPU coherency before returning to Python (framework doesn't track it).
        if (projector_output_.defined()) {
            rpu_ddr_flush(projector_output_.data_ptr<c10::Half>());
            return projector_output_;
        }
        return result;
    }

protected:
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            seq_len_, image_batch_count_,
            planning_external_patch_prologue_);
    }

    // ========================================================================
    // static_config — D-501 preload + KV_FIRST callback registration
    //
    // Registers the pointer-to-member `&SigLIPModel::emit_preload_weights`
    // (cast to FusedModelBase pointer-to-member) on the preload-fn slot.
    // Framework dispatches via std::invoke; launches join the adapter-owned
    // capture when one is active.
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers       = num_layers();

        // D-501 one-shot .preload_fn: replaces v2's `build_preload_subgraph`
        // virtual. The body emits only rpu_launch_* operations and never opens
        // its own graph or batch lifecycle.
        // PRESERVED through the SEQUENTIAL→KV_FIRST refactor (Site3): SigLIP is
        // the only KV_FIRST consumer that ALSO uses one-shot preload_fn (Gemma
        // uses per-buffer .preload_callback instead) — both coexist fine here.
        cfg.preload_fn = static_cast<void(FusedModelBase::*)()>(
                             &SigLIPModel::emit_preload_weights);

        // D-501 KV_FIRST: two pointer-to-member slots for the two-phase dispatch
        // (mirror GemmaModel::static_config). The static_cast is required because
        // the base struct types the slots as pointer-to-member-of-FusedModelBase
        // (std::invoke resolves to the concrete subclass at runtime).
        // SigLIP has NO rope and NO attention mask (MASK_NONE), so the bodies are
        // simplified vs Gemma — see emit_kv_first_body / plan_kv_first_chunks.
        cfg.kv_first_fn = static_cast<void(FusedModelBase::*)(int, const ChunkInfo&)>(
                              &SigLIPModel::emit_kv_first_body);
        cfg.kv_first_chunk_plan_fn =
            static_cast<ChunkPlan(FusedModelBase::*)(const ChunkPlan&)>(
                &SigLIPModel::plan_kv_first_chunks);

        // SigLIP REQUIRES single group (matches v2 design at
        // rpu_siglip_fused_encoder_layer.cpp:634-645). With cross_batch < num_layers,
        // splitting this encoder changes its required cross-layer SPM/dataflow
        // contract and produced non-deterministic output in the original A/B.
        //
        // The Qwen3 global runtime knob (g_cross_layer_batch_size, default 12) leaks into
        // SigLIP if not explicitly set. Other models (Qwen3, Gemma) work because their
        // tests explicitly set cross_batch >= num_layers (test_qwen3_decode.py sets 36).
        // SigLIP enforces this invariant in C++ to be defensive.
        cfg.cross_layer_batch_size = num_layers();
        return cfg;
    }

    // ========================================================================
    // dynamic_config — KV_FIRST + AUTO inter-layer I/O.
    //
    // SEQUENTIAL→KV_FIRST refactor (halo-vit-cache-kvfirst-refactor-design
    // -2026-06-18): the former single-chunk SPM_RESIDENT encoder hit the 8 MB
    // SPM ceiling at ~960 tokens (the MLP-intermediate working set), hanging
    // HALO's 1564-patch single-image ViT path (probe note
    // halo-vit-cache-seq-ceiling-probe-2026-06-18). KV_FIRST keeps the full
    // [seq,h] K/V in DDR and feeds SPM one query-chunk at a time, so the
    // per-chunk SPM footprint stays chunk-bounded; AUTO then picks SPM_RESIDENT
    // for a lone chunk (seq≤ceiling, byte-identical to the old path) and
    // DDR_PINGPONG when the sequence splits into multiple chunks.
    //
    // The old "DDR ping-pong 8-core broadcast-write race / corruption" worry
    // (former Round-9 SPM_RESIDENT rationale) does NOT reproduce: the base
    // ping-pong DMA is a single core-0 path (rpu_memcpy.cpp:1146), not the old
    // multicore broadcast — cross-ref design note §0b for the static proof.
    // No build_chunk_masks: SigLIP attention is MASK_NONE (bidirectional, no
    // additive mask), unlike Gemma's per-chunk causal masks.
    // ========================================================================
    ModelDynamicConfig dynamic_config(const ChunkPlan& /*plan*/) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::KV_FIRST;
        cfg.inter_layer_io = pi05_vision_owner_ln_opt_in_
            ? InterLayerIO::SPM_RESIDENT : InterLayerIO::AUTO;
        // RAW_SPM is descriptor-owned: legacy/no-descriptor entry points keep
        // the historical DDR route and cannot silently select a physical ABI.
        cfg.attention_policy = ctx().has_complete_physical_manifest()
            ? AttentionExecutionPolicy::AUTO
            : AttentionExecutionPolicy::DDR_KV;
        return cfg;
    }

    void consume_manifest_route(
        FmbRouteFamily family, int64_t site_id, int64_t selector,
        int64_t invocation = 0) {
        if (!ctx().has_complete_physical_manifest()) return;
        ctx().consume_physical_route(
            family, site_id, selector, /*resolved_flags=*/0,
            /*resolved_arguments=*/{}, invocation);
    }

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout, int64_t physical_len,
        int64_t logical_len, int64_t position) const override {
        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = position + logical_len;
        manifest.kv_insert_physical_rows = physical_len;
        manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation = linear_accumulation_policy();
        if(pi05_vision_fc_weight_outer_opt_in_) {
            require_pi05_vision_fc_profile(physical_len,image_batch_count_);
            TORCH_CHECK(logical_len==768 && position==0 && layout.batch_size==1 &&
                !layout.is_causal && !layout.use_attn_mask && layout.chunk_size==768 &&
                layout.effective_kv_cs()==768,"Pi Vision FC weight-outer rejects noncanonical layout");
            require_pi05_vision_fc_schedule(plan);
        }

        LayoutContext spm_layout = layout;
        spm_layout.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const bool raw_spm_eligible = subclass_spm_kv_by_mha_eligible(
            plan, spm_layout, position);
        const bool raw_spm = layout.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        require_pi05_vision_kv_spm_only_candidate(
            plan, layout, physical_len, logical_len, position);
        if (pi05_vision_owner_ln_opt_in_) {
            require_pi05_vision_owner_ln_profile(physical_len, image_batch_count_);
            require_pi05_vision_kv_spm_only_schedule(plan);
            TORCH_CHECK(logical_len == physical_len && position == 0 &&
                            layout.batch_size == 1 && !layout.is_causal &&
                            !layout.use_attn_mask && layout.chunk_size == physical_len &&
                            layout.effective_kv_cs() == physical_len && raw_spm && raw_spm_eligible,
                        "Pi Vision owner-LN requires exact SPM-resident input2-or-3/QKV1/compute1 candidate");
        }
        TORCH_CHECK(
            !raw_spm || raw_spm_eligible,
            "RPU_PLANNER_REJECT:CAPABILITY: SigLIP raw-SPM attention was "
            "selected without one full packed QKV/compute schedule");

        auto append = [&](FmbRouteFamily family, int64_t site_id,
                          int64_t selector,
                          std::vector<int64_t> arguments = {},
                          int64_t invocation = 0) {
            manifest.routes.push_back({
                site_id, family, selector, /*flags=*/0,
                std::move(arguments), invocation});
        };
        auto append_linear = [&](int64_t site_id, int64_t invocation) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                   {}, invocation);
        };
        if (num_cores() != 8) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, SIGLIP_CORE_PROFILE_SITE,
                   1, core_profile_arguments());
        }
        for (const ChunkInfo& chunk : plan.qkv.chunks) {
            if (pi05_vision_q_resident_opt_in_) {
                const bool carry_q = use_pi05_vision_q_resident(
                    chunk.len, layout.attention_policy) &&
                    plan.qkv.chunks.size() == 1 && plan.compute.chunks.size() == 1 &&
                    plan.compute.chunks.front().len == chunk.len;
                append(FmbRouteFamily::GRAPH_SCHEDULE, SIGLIP_Q_RESIDENT_SITE,
                       carry_q ? 2 : 1, {}, chunk.idx);
            }
            append_linear(SIGLIP_Q_LINEAR_SITE, chunk.idx);
            append_linear(SIGLIP_K_LINEAR_SITE, chunk.idx);
            append_linear(SIGLIP_V_LINEAR_SITE, chunk.idx);
            const KvInsertSegmentPlan kv_plan =
                resolve_kvinsert_plan_auto(
                    SIGLIP_KV_INSERT_SITE, manifest.graph_lifecycle,
                    position + chunk.offset, chunk.len, chunk.len,
                    num_cores(), num_q_heads(), head_dim(),
                    SIGLIP_KV_CAPABILITIES);
            const KvInsertRouteArguments arguments =
                rpu_kvinsert_route_arguments(
                    kv_plan, num_cores(), num_q_heads(), head_dim());
            manifest.routes.push_back({
                SIGLIP_KV_INSERT_SITE, FmbRouteFamily::KV_INSERT,
                static_cast<int64_t>(kv_plan.route()),
                pi05_vision_kv_spm_only_opt_in_
                    ? SIGLIP_KV_FLAG_RAW_SPM_ONLY
                    : SIGLIP_KV_FLAG_DDR_MIRROR,
                {arguments.begin(), arguments.end()}, chunk.idx});
            manifest.kv_insert_physical_rows = std::max(
                manifest.kv_insert_physical_rows,
                kv_plan.physical_rows());
        }
        for (const ChunkInfo& chunk : plan.compute.chunks) {
            const int64_t per_image_ctx =
                physical_len / image_batch_count_;
            manifest.routes.push_back({
                raw_spm
                    ? SIGLIP_RAW_SPM_ATTN_SITE
                    : (image_batch_count_ > 1
                           ? SIGLIP_MINIBATCH_ATTN_SITE
                           : SIGLIP_UNIFIED_ATTN_SITE),
                FmbRouteFamily::ATTENTION,
                static_cast<int64_t>(
                    raw_spm ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                            : AttentionExecutionPolicy::DDR_KV),
                raw_spm || raw_spm_eligible
                    ? 0 : SIGLIP_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                {image_batch_count_, per_image_ctx, chunk.len, physical_len,
                 num_q_heads(), head_dim()}, chunk.idx});
            append_linear(SIGLIP_O_LINEAR_SITE, chunk.idx);
            append(
                FmbRouteFamily::ALL_REDUCE,
                SIGLIP_ATTN_ALL_REDUCE_SITE,
                pi05_vision_owner_ln_opt_in_ ? SIGLIP_OWNER_LN_ROUTE :
                    fmb_ring_all_reduce_route_selector(
                        chunk.len, hidden_size(), use_pi05_ring_xor3(chunk.len), num_cores()),
                pi05_vision_owner_ln_opt_in_ ? pi05_vision_owner_ln_arguments(false, chunk.len) :
                    std::vector<int64_t>{}, chunk.idx);
            if(pi05_vision_fc_weight_outer_opt_in_) {
                append(FmbRouteFamily::GRAPH_SCHEDULE,SIGLIP_FC1_WEIGHT_OUTER_SITE,
                    2,pi05_vision_fc_arguments(false),chunk.idx);
                append(FmbRouteFamily::GRAPH_SCHEDULE,SIGLIP_FC2_WEIGHT_OUTER_SITE,
                    2,pi05_vision_fc_arguments(true),chunk.idx);
            } else {
                append_linear(SIGLIP_FC1_LINEAR_SITE, chunk.idx);
                append_linear(SIGLIP_FC2_LINEAR_SITE, chunk.idx);
            }
            append(
                FmbRouteFamily::ALL_REDUCE,
                SIGLIP_MLP_ALL_REDUCE_SITE,
                pi05_vision_owner_ln_opt_in_ ? SIGLIP_OWNER_LN_COMPACT_ROUTE :
                    fmb_ring_all_reduce_route_selector(
                        chunk.len, hidden_size(), use_pi05_ring_xor3(chunk.len), num_cores()),
                pi05_vision_owner_ln_opt_in_ ? pi05_vision_owner_ln_arguments(true, chunk.len) :
                    std::vector<int64_t>{}, chunk.idx);
            append_linear(SIGLIP_PROJECTOR_LINEAR_SITE, chunk.idx);
        }
        if (planning_external_patch_prologue_) {
            TORCH_CHECK(
                has_patch_emb_ &&
                    static_cast<int64_t>(plan.spans.size()) ==
                        image_batch_count_,
                "RPU_PLANNER_REJECT:CAPABILITY: SigLIP patch prologue "
                "requires initialized patch weights and one span per image");
            const int64_t K = pe_kh_ * pe_kw_ * pe_cin_padded_;
            for (int64_t image = 0; image < image_batch_count_; ++image) {
                const FmbExecutionSpan& span = plan.spans[image];
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    SIGLIP_PATCH_INPUT_DMA_SITE,
                    static_cast<int64_t>(SiglipPatchMutableDmaRoute::
                        DDR_BROADCAST_TO_SPM_MUTABLE),
                    {span.len, pe_cin_orig_, pe_cin_padded_, pe_kh_, pe_kw_,
                     pe_strideh_, pe_stridew_, 1}, image);
                append(
                    FmbRouteFamily::LINEAR,
                    SIGLIP_PATCH_LINEAR_SITE,
                    static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                    {span.len, pe_cout_, K, 1, 1, 0}, image);
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    SIGLIP_PATCH_POSITION_DMA_SITE,
                    static_cast<int64_t>(SiglipPatchMutableDmaRoute::
                        DDR_BROADCAST_TO_SPM_MUTABLE),
                    {span.len, pe_cout_, 1}, image);
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    SIGLIP_PATCH_OUTPUT_DMA_SITE,
                    static_cast<int64_t>(SiglipPatchMutableDmaRoute::
                        SPM_COPY_TO_DDR_MUTABLE),
                    {span.offset * pe_cout_ * DWIDTH,
                     span.len * pe_cout_}, image);
            }
        }
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA, 1, num_cores(), mlp_tp());
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
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
        if (pi05_vision_kv_spm_only_opt_in_ || pi05_vision_owner_ln_opt_in_) {
            LayoutContext spm_only_layout = layout;
            spm_only_layout.attention_policy =
                AttentionExecutionPolicy::SPM_KV_BY_MHA;
            return {physical_manifest_for_candidate(
                plan, spm_only_layout, physical_len, logical_len, position)};
        }
        LayoutContext ddr_layout = layout;
        ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;
        std::vector<FmbPhysicalExecutionManifest> domain{
            physical_manifest_for_candidate(
                plan, ddr_layout, physical_len, logical_len, position)};
        LayoutContext spm_layout = layout;
        spm_layout.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        if (subclass_spm_kv_by_mha_eligible(
                plan, spm_layout, position)) {
            domain.push_back(physical_manifest_for_candidate(
                plan, spm_layout, physical_len, logical_len, position));
        }
        return domain;
    }

    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& manifest) const override {
        validate_pi05_vision_fc_manifest(manifest);
        validate_pi05_vision_kv_spm_only_manifest(manifest);
        validate_pi05_vision_owner_ln_manifest(manifest);
        return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
    }

    bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const override {
        if (position != 0 || layout.is_causal || layout.use_attn_mask ||
            layout.batch_size != 1 ||
            layout.attention_policy !=
                AttentionExecutionPolicy::SPM_KV_BY_MHA ||
            plan.chunk_mode != ChunkMode::KV_FIRST ||
            plan.qkv.chunks.size() != 1 ||
            plan.compute.chunks.size() != 1 || image_batch_count_ <= 0 ||
            static_cast<int64_t>(plan.input.chunks.size()) !=
                image_batch_count_ ||
            static_cast<int64_t>(plan.spans.size()) != image_batch_count_) {
            return false;
        }
        const ChunkInfo& chunk = plan.compute.chunks.front();
        const int64_t seq = chunk.len;
        // The candidate describes its complete packed sequence. Live seq_len_
        // may still be zero or describe a prior shape during dry planning.
        if (seq <= 0 || seq % image_batch_count_ != 0 ||
            chunk.offset != 0 || chunk.kv_seq_len != seq ||
            plan.qkv.chunks.front().offset != 0 ||
            plan.qkv.chunks.front().len != seq) {
            return false;
        }
        const int64_t per_image = seq / image_batch_count_;
        for (int64_t image = 0; image < image_batch_count_; ++image) {
            const int64_t offset = image * per_image;
            if (plan.input.chunks[image].offset != offset ||
                plan.input.chunks[image].len != per_image ||
                plan.spans[image].offset != offset ||
                plan.spans[image].len != per_image) {
                return false;
            }
        }
        return num_layers() == 27 && layer_weights_.size() == 27 &&
            hidden_size() == 1152 && intermediate_size() == 4352 &&
            num_q_heads() == 16 && num_kv_heads() == 16 &&
            head_dim() == 80 && orig_head_dim_ == 72 &&
            sdpa_by_mha_spm_is_valid(
                image_batch_count_, per_image, per_image,
                num_q_heads(), num_kv_heads(), head_dim(), num_cores(),
                /*MASK_NONE=*/0);
    }

    // ========================================================================
    // declare_buffers — SPM buffer layout with lifecycle aliasing
    //
    // 12 Persistent buffers (10 PersistentPerLayer + 2 Persistent global):
    //   norm gamma/beta for LN1/LN2, biases, post-LN gamma/beta.
    // Persistent buffers 跨 Gemma/AdaRMS 子系统 reset_all() 保留; Pi0.5 流水线
    // cycle 2+ 时 SIGLIP_WEIGHTS DMA graph + SIGLIP_COMPUTE compute graph 都直接
    // SKIP/REPLAY (省去 272 个权重 DMA + 382 个 compute kernel).
    //
    // SigLIP uses one-shot preload-fn (D-501) instead of per-buffer
    // `.preload_callback` (which would be the Gemma pattern) — so NO
    // `.preload_callback` field is set on any decl here.
    // ========================================================================
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        // Site4 (KV_FIRST 3-scope): three chunk sizes drive three buffer scopes
        // (mirror Gemma / image_flow). comp_cs feeds Phase 2 (SDPA + MLP);
        // kv_cs feeds Phase 1 (QKV + KV-insert); wide_cs = max sizes the
        // LayerWide band read in BOTH phases. SigLIP runs single-chunk-size
        // today (plan_kv_first_chunks returns compute_plan), so kv_cs == comp_cs
        // == wide_cs, but the scope split keeps the estimator honest and leaves
        // dual-chunk headroom for free.
        int64_t comp_cs = ctx.chunk_size;
        int64_t kv_cs   = ctx.effective_kv_cs();
        int64_t wide_cs = std::max(comp_cs, kv_cs);
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();

        int64_t local_q_dim = (nq / num_cores()) * hd;
        int64_t local_inter = is_ / mlp_tp();
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
        const bool raw_spm = ctx.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const bool carry_q = comp_cs == kv_cs &&
            use_pi05_vision_q_resident(kv_cs, ctx.attention_policy);

        // LayerWide: read in both phases → sized at wide_cs.
        int64_t res  = A(wide_cs * h * DWIDTH);
        // KvInsert: Phase-1 QKV outputs (q renamed → q_kv; Phase 1 writes q_kv,
        // dumps it to q_ddr_buf_, never reads it again — Phase 2 reloads from
        // DDR into q_comp).
        const int64_t raw_kv_rows = raw_spm
            ? image_batch_count_ * Align(
                  CeilDiv(kv_cs, image_batch_count_), int64_t{16})
            : kv_cs;
        int64_t qkv  = A(kv_cs * local_q_dim * DWIDTH);
        const int64_t raw_kv = A(raw_kv_rows * local_q_dim * DWIDTH);
        // Compute: Phase-2 working buffers → sized at comp_cs.
        int64_t q_comp = A(comp_cs * local_q_dim * DWIDTH);
        int64_t sdpa_out = A(comp_cs * local_q_dim * DWIDTH);
        int64_t fc1  = A(comp_cs * local_inter * DWIDTH);

        // SDPA tmp sizing (same formula as before, now sized at comp_cs — the
        // Phase-2 query-chunk length, NOT the full sequence).
        SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                            hd, /*nq*/nq, /*nkv*/nq,
                            /*cores*/num_cores(), /*mask*/0 /*MASK_NONE*/};
        SdpaTiling t = sdpa_compute_tiling(sdpa_cfg, comp_cs);
        int64_t nkv_per_core = CeilDiv(nq, (int64_t)num_cores());
        int64_t sdpa_tmp = A(t.tile_n_v16 * t.tile_k * nkv_per_core * CeilDiv(comp_cs, t.tile_m) * 32);
        if (raw_spm) sdpa_tmp = std::max(sdpa_tmp, raw_kv);

        // DMA-safe sizing for persistent buffers
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

        // Phase map (single 1..8 numbering spanning both KV_FIRST phases, so the
        // estimator's per-scope phase peak is well-defined):
        //   Phase 1 (kv_first body): LN1 → QKV(q_kv/k/v) → KV-insert → Q→DDR.
        //   Phase 2 (build_layer_subgraph): DDR→q_comp → SDPA(sdpa_out/sdpa_tmp)
        //     → o_proj/LN2/fc1(GELU)/fc2 (oproj reused as scratch). LayerWide
        //     residual1/input_norm/oproj stay alive across both.
        std::vector<BufferDecl> decls;

        // Structural — alive across both phases (always-conflict with KVIN/COMP).
        decls.push_back({"residual1",  res, 1, 8, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"input_norm", res, 1, 8, StorageClass::Temp, 0, nullptr, ALL});
        // oproj: Phase-2 o_proj out / LN2 out / fc2 partial scratch. Held in the
        // LayerWide band per design (could be COMP-scoped at comp_cs as a future
        // SPM optimization; kept LayerWide for the conservative first cut).
        decls.push_back({"oproj",      res, 4, 8, StorageClass::Temp, 0, nullptr, ALL});

        // KvInsert-only (Phase 1) — aliasable against Compute-only buffers.
        decls.push_back({"q_kv", qkv, 1, carry_q ? 5 : 3,
                         StorageClass::Temp, 0, nullptr, carry_q ? ALL : KVIN});
        // Raw attention consumes K/V in Phase 2, so its full packed source is
        // LayerWide.  DDR candidates retain the original KV_FIRST aliasing.
        decls.push_back({"k", raw_spm ? raw_kv : qkv, 1,
                         raw_spm ? 5 : 3, StorageClass::Temp, 0, nullptr,
                         raw_spm ? ALL : KVIN});
        decls.push_back({"v", raw_spm ? raw_kv : qkv, 1,
                         raw_spm ? 5 : 3, StorageClass::Temp, 0, nullptr,
                         raw_spm ? ALL : KVIN});

        // Compute-only (Phase 2) — lifecycle aliasing across attention→MLP.
        decls.push_back({"q_comp", carry_q ? 0 : q_comp, 4, 5,
                         StorageClass::Temp, 0, carry_q ? "q_kv" : nullptr,
                         carry_q ? ALL : COMP});
        decls.push_back({"sdpa_out",   sdpa_out, 5, 6, StorageClass::Temp, 0, nullptr, COMP});
        decls.push_back({"sdpa_tmp",   sdpa_tmp, 5, 5, StorageClass::Temp, 0, nullptr, COMP});
        decls.push_back({"fc1",        fc1,      7, 8, StorageClass::Temp, 0, nullptr, COMP});

        // Persistent per-layer: norm weights (gamma/beta for LN1, LN2)
        decls.push_back({"ln1_gamma", norm_w_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        decls.push_back({"ln1_beta",  norm_w_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        decls.push_back({"ln2_gamma", norm_w_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        decls.push_back({"ln2_beta",  norm_w_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        // Persistent per-layer: biases
        decls.push_back({"q_bias",    q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        decls.push_back({"k_bias",    q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        decls.push_back({"v_bias",    q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        decls.push_back({"fc1_bias",  fc1_bias_sz,  0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        decls.push_back({"o_bias",    full_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        decls.push_back({"fc2_bias",  full_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr, ALL});
        // Persistent global: post-LN gamma/beta
        decls.push_back({"post_ln_gamma", norm_w_sz, 0, 0, StorageClass::Persistent, 0, nullptr, ALL});
        decls.push_back({"post_ln_beta",  norm_w_sz, 0, 0, StorageClass::Persistent, 0, nullptr, ALL});

        return decls;
    }

    // ========================================================================
    // emit_preload_weights — 272 persistent weight DMAs (D-501 callback body)
    //
    // Was v2 `build_preload_subgraph()` virtual; v3 registers this via the
    // preload-fn slot in static_config(). Body is verbatim from v2 — framework
    // opens the weights-graph scope around this call.
    //
    // EXT-5 contract: body emits only rpu_launch_* DMAs; no subclass-side
    // batch-context calls (framework owns them). CRITICAL: DMA length is num_elements
    // (NOT bytes). Per layer (27 layers, 10 DMA ops each) + 2 global = 272 total.
    // ========================================================================
    void emit_preload_weights() {
        int64_t h = hidden_size();
        int64_t local_q_dim = (num_q_heads() / num_cores()) * head_dim();
        int64_t local_inter = intermediate_size() / mlp_tp();

        // Loop 1: 8-core DMAs for all layers (memset + norm + col-partition bias)
        // Matches v2 ordering at rpu_siglip_fused_encoder_layer.cpp:316-333.
        for (int64_t L = 0; L < num_layers(); ++L) {
            auto& bn = layer_bias_norm_[L];

            // Zero row-partition biases before summing the selected root cores.
            rpu_launch_memset_spm_multicore(layer_addr(L, 0, "o_bias"), h, num_cores());
            rpu_launch_memset_spm_multicore(layer_addr(L, 0, "fc2_bias"), h, num_cores());

            // Broadcast LayerNorm gamma/beta to all cores (num_elements, NOT bytes)
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

            // Col-partition scatter: each core gets its own slice of the bias.
            // DMA path via rpu_launch_ddr_scatter_spm_dma — weights are static
            // (model-construction-time alloc), so non-mutable is safe.
            rpu_launch_ddr_scatter_spm_dma(
                bn.q_b.data_ptr<c10::Half>(),
                /*elements_per_core=*/local_q_dim,
                /*core_stride_bytes=*/local_q_dim * DWIDTH,
                layer_addr(L, 0, "q_bias"),
                /*num_cores=*/num_cores());
            rpu_launch_ddr_scatter_spm_dma(
                bn.k_b.data_ptr<c10::Half>(),
                local_q_dim, local_q_dim * DWIDTH,
                layer_addr(L, 0, "k_bias"),
                /*num_cores=*/num_cores());
            rpu_launch_ddr_scatter_spm_dma(
                bn.v_b.data_ptr<c10::Half>(),
                local_q_dim, local_q_dim * DWIDTH,
                layer_addr(L, 0, "v_bias"),
                /*num_cores=*/num_cores());
            rpu_launch_ddr_scatter_spm_dma(
                bn.fc1_b.data_ptr<c10::Half>(),
                local_inter, local_inter * DWIDTH,
                layer_addr(L, 0, "fc1_bias"),
                /*num_cores=*/num_cores());
        }

        // Loop 2: 1-core DMAs for all layers (row-partition biases).
        // Matches v2 ordering at rpu_siglip_fused_encoder_layer.cpp:334-338.
        // Mixing 1-core and 8-core DMAs in the same loop (as before) may cause
        // kernel-ordering/synchronization issues in graph replay; v2's grouped
        // pattern avoids it.
        for (int64_t L = 0; L < num_layers(); ++L) {
            auto& bn = layer_bias_norm_[L];
            rpu_launch_ddr_broadcast_spm_dma(
                bn.o_b.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "o_bias"), /*num_cores=*/1);
            rpu_launch_ddr_broadcast_spm_dma(
                bn.fc2_b.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "fc2_bias"), /*num_cores=*/1);
        }

        // Global: post-LayerNorm weights (broadcast to all cores)
        rpu_launch_ddr_broadcast_spm_dma(
            post_ln_w_.data_ptr<c10::Half>(), h,
            addr(0, "post_ln_gamma"), num_cores());
        rpu_launch_ddr_broadcast_spm_dma(
            post_ln_b_.data_ptr<c10::Half>(), h,
            addr(0, "post_ln_beta"), num_cores());
    }

    // ========================================================================
    // emit_kv_first_body — D-501 kv_first_fn (KV_FIRST Phase 1 body).
    //
    // SEQUENTIAL→KV_FIRST refactor Site6: extracts the former build_layer_subgraph
    // Phase-1/2/3-front (LayerNorm1 → QKV+bias → KV-insert) and ADDS a Q→DDR save
    // so Phase 2 (build_layer_subgraph) can reload Q per query-chunk. Mirrors
    // GemmaModel::emit_kv_first_body but SIMPLIFIED: SigLIP has NO rope (position
    // embedding is added in patch_embed, not per-layer) and NO attention mask.
    //
    // CRITICAL (load-bearing) vs the old single-chunk body: KV-insert position
    // is now `chunk.offset` (was a hardcoded 0). With position=0 and ctx().position
    // also 0 (SigLIP forward passes position=0), chunk.offset is the absolute KV
    // row for this chunk — so multi-chunk runs insert each chunk's K/V at the
    // right rows of the full [seq,h] cache.
    //
    // Pitfall 3 structural fix: SDPA / KV-insert sites take SPM OFFSETS via
    // addr_offset(name).value (typed-distinct from the absolute addr() return).
    // ========================================================================
    void emit_kv_first_body(int layer_idx, const ChunkInfo& chunk) {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t kv_pos  = ctx().position + chunk.offset;  // absolute KV row
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();

        // Phase 1: DDR→SPM input DMA + LayerNorm1.
        // input_in_spm guard preserved (SigLIP-specific deviation from Gemma's
        // unconditional re-read): in single-chunk AUTO→SPM_RESIDENT mode the
        // prior layer left its output in "residual1" SPM and the ping-pong DDR
        // buffers are nullptr, so an unconditional emit_layer_input_dma for
        // inner layers would read a null base. In multi-chunk DDR_PINGPONG mode
        // input_in_spm is always false, so this re-reads each chunk's input.
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "ln1_gamma"), layer_addr(layer_idx, 0, "ln1_beta"),
            seq_len, h, eps_, false, 0, num_cores());

        // QKV Linear with bias (SPM-to-SPM ACC16, col-partition). q → q_kv.
        consume_manifest_route(
            FmbRouteFamily::LINEAR, SIGLIP_Q_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q_kv"),
            seq_len, nq * hd, h, 1, num_cores(),
            layer_addr(layer_idx, 0, "q_bias"), lw.q_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::LINEAR, SIGLIP_K_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, num_cores(),
            layer_addr(layer_idx, 0, "k_bias"), lw.k_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::LINEAR, SIGLIP_V_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, num_cores(),
            layer_addr(layer_idx, 0, "v_bias"), lw.v_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);

        // No RoPE (SigLIP has none).

        // KV cache insert at absolute position chunk.offset.
        // Pitfall 3 structural fix: takes SPM offsets via addr_offset(name).value.
        require_pi05_vision_kv_spm_only_runtime(chunk);
        const auto& kv_route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, SIGLIP_KV_INSERT_SITE, chunk.idx);
        const KvInsertSegmentPlan kv_plan =
            restore_kvinsert_plan(
                SIGLIP_KV_INSERT_SITE, kv_route.arguments, num_cores(), nq, hd);
        TORCH_CHECK(
            kv_plan.logical_rows() == seq_len &&
                kv_plan.physical_rows() == seq_len &&
                kv_plan.segment(0).position == kv_pos,
            "SigLIP KV descriptor geometry drift at invocation ", chunk.idx);
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, SIGLIP_KV_INSERT_SITE,
            static_cast<int64_t>(kv_plan.route()),
            pi05_vision_kv_spm_only_opt_in_
                ? SIGLIP_KV_FLAG_RAW_SPM_ONLY
                : SIGLIP_KV_FLAG_DDR_MIRROR,
            kv_route.arguments, chunk.idx);
        if (!pi05_vision_kv_spm_only_opt_in_) {
            auto& k_cache = (*ctx().k_caches)[layer_idx];
            auto& v_cache = (*ctx().v_caches)[layer_idx];
            rpu_launch_insert_kvcache_spm_unified_with_plan(
                k_cache, v_cache,
                addr_offset("k").value, addr_offset("v").value,
                nq, hd, num_cores(),
                /*k_cache_batch_offset_elems=*/0,
                /*v_cache_batch_offset_elems=*/0,
                /*spm_rows=*/0, kv_plan);
        }

        // Save Q to the per-seq Q staging slot via spm_scatter_ddr_dma
        // (position-indexed by chunk.offset). The slot (keyed by seq_len_) is
        // created once in forward_packed and NEVER reallocated → stable data_ptr,
        // safe for non-mutable DMA across REPLAY (matching GemmaModel:
        // stable storage). Phase 2 reloads it into "q_comp".
        TORCH_CHECK(q_ddr_slots_.count({seq_len_, local_q_heads_ * head_dim()}) == 1,
                    "SigLIPModel: Q staging slot not allocated in KV_FIRST mode");
        const at::Tensor& q_slot = q_ddr_slot();
        int64_t q_local_elems = seq_len * local_q_heads_ * hd;
        c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
        int64_t q_row_stride = local_q_heads_ * hd;
        int64_t q_elem_offset = chunk.offset * q_row_stride;
        int64_t q_core_stride_bytes = q_slot.size(1) * q_row_stride * DWIDTH;
        const bool carry_q = use_pi05_vision_q_resident(
            seq_len, ctx().attention_policy);
        if (pi05_vision_q_resident_opt_in_) {
            consume_manifest_route(FmbRouteFamily::GRAPH_SCHEDULE,
                                   SIGLIP_Q_RESIDENT_SITE, carry_q ? 2 : 1,
                                   chunk.idx);
        }
        if (carry_q) {
            TORCH_CHECK(chunk.idx == 0 && chunk.offset == 0 && seq_len == seq_len_ &&
                            addr(0, "q_kv") == addr(0, "q_comp"),
                        "Pi0.5 Vision Q carry requires one shared live SPM slot");
        } else {
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "q_kv"), q_ddr_base + q_elem_offset,
                q_local_elems, q_core_stride_bytes,
                /*num_cores=*/num_cores());
        }

        // No residual output DMA — Phase 2 re-reads the original input (matching
        // Gemma); the SPM_RESIDENT single-chunk case keeps residual1 live.
    }

    // ========================================================================
    // build_layer_subgraph — KV_FIRST Phase 2 body (was the full SEQUENTIAL
    // 6-phase pipeline; Site7 removed LN1/QKV/KV-insert → now in Phase 1).
    //
    // Phase 2: reload Q from q_ddr_buf_ → q_comp → SDPA (full-KV) → O_proj+resid
    //          → LayerNorm2+fc1+GELU → fc2+resid.
    //
    // SDPA uses seq_q = chunk.len (the query-chunk) and kv_seq_len = seq_len_
    // (the FULL bidirectional KV) — the two are decoupled by the launcher (see
    // rpu_launch_sdpa_spm_unified_kernel_v2 signature). mask=0 (MASK_NONE).
    //
    // Last layer: fuses post-LN + projector Linear (D-501).
    //
    // Pitfall 3 structural fix: SDPA call site uses `addr_offset(name).value`
    // (typed SPM offset). Non-SDPA sites use absolute `addr(0, name)` /
    // `layer_addr(L, 0, name)`.
    // ========================================================================
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        require_pi05_vision_owner_ln_runtime(chunk, layer_idx);
        if (num_cores() != 8 && layer_idx == 0 && chunk.idx == 0 &&
            ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                SIGLIP_CORE_PROFILE_SITE, 1, 0, core_profile_arguments());
        }
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;        // this query-chunk length
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();
        int64_t local_inter = is_ / mlp_tp();

        // Re-read original input into residual1 (Phase 2 reads the same source
        // as Phase 1). Guarded by input_in_spm (see emit_kv_first_body note):
        // single-chunk SPM_RESIDENT keeps residual1 live across both phases, so
        // skip; multi-chunk DDR_PINGPONG re-reads per chunk.
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        // Reload Q from the per-seq Q staging slot (position-indexed by
        // chunk.offset) into "q_comp" (mirror GemmaModel build_layer_subgraph
        // DDR→SPM scatter; NO rope). The slot is keyed by seq_len_, so its
        // size(1) == seq_len_ gives the per-core pitch — and being never
        // reallocated, its data_ptr stays valid across REPLAY .
        const at::Tensor& q_slot = q_ddr_slot();
        int64_t q_local_elems = seq_len * local_q_heads_ * hd;
        c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
        int64_t q_row_stride = local_q_heads_ * hd;
        int64_t q_elem_offset = chunk.offset * q_row_stride;
        int64_t q_core_stride = q_slot.size(1) * q_row_stride * DWIDTH;
        if (use_pi05_vision_q_resident(seq_len, ctx().attention_policy)) {
            TORCH_CHECK(chunk.idx == 0 && chunk.offset == 0 && seq_len == seq_len_ &&
                            addr(0, "q_kv") == addr(0, "q_comp"),
                        "Pi0.5 Vision Q carry lost its single-chunk SPM lifetime");
        } else {
            rpu_launch_ddr_scatter_spm_dma(
                q_ddr_base + q_elem_offset,
                /*elements_per_core=*/q_local_elems,
                /*core_stride_bytes=*/q_core_stride,
                addr(0, "q_comp"),
                /*num_cores=*/num_cores());
        }

        // SDPA over the FULL bidirectional KV cache (kv_seq_len = seq_len_),
        // query = this chunk (seq_q = chunk.len). Source Q from "q_comp".
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        double attn_scale = 1.0 / std::sqrt(static_cast<double>(orig_head_dim_));

        // Minibatch SDPA (image_batch_count_>1) asserts a single chunk
        // (seq_q == full packed seq); forward_packed pins that compute plan for
        // N>1, so this guard is belt-and-suspenders.
        TORCH_CHECK(image_batch_count_ == 1 || chunk.len == seq_len_,
                    "SigLIPModel: minibatch (N=", image_batch_count_,
                    ") SDPA requires a single chunk (chunk.len=", chunk.len,
                    " == seq_len_=", seq_len_, "); forward_packed must pin one "
                    "compute chunk for N>1.");
        const int64_t per_image_ctx = seq_len_ / image_batch_count_;

        if (ctx().attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            TORCH_CHECK(
                chunk.offset == 0 && chunk.len == seq_len_ &&
                    seq_len_ % image_batch_count_ == 0,
                "SigLIP raw-SPM attention requires one full packed chunk");
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION, SIGLIP_RAW_SPM_ATTN_SITE,
                static_cast<int64_t>(
                    AttentionExecutionPolicy::SPM_KV_BY_MHA),
                /*resolved_flags=*/0,
                {image_batch_count_, per_image_ctx, seq_len, seq_len_,
                 nq, hd}, chunk.idx);
            rpu_launch_v_transpose_spm(
                addr(0, "v"), addr(0, "sdpa_tmp"),
                image_batch_count_, per_image_ctx, nq, hd, num_cores());
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q_comp"), addr(0, "k"), addr(0, "sdpa_tmp"),
                addr(0, "sdpa_out"), /*mask_spm=*/0,
                /*MASK_NONE=*/0, attn_scale,
                image_batch_count_, per_image_ctx, per_image_ctx,
                nq, nq, hd, num_cores());
        } else if (image_batch_count_ > 1) {
            // SigLIP-batch: N images packed into seq_len_. Per-image attention
            // isolation via the minibatch kernel — image i attends ONLY its own
            // per_image_ctx tokens. K/V were inserted contiguously at position 0
            // in Phase 1, so the kernel's image_idx*per_image_sKeyVx offset lines
            // up. Single-chunk only (guarded above).
            if (ctx().has_complete_physical_manifest()) {
                LayoutContext spm_layout;
                spm_layout.attention_policy =
                    AttentionExecutionPolicy::SPM_KV_BY_MHA;
                spm_layout.batch_size = ctx().batch_size;
                spm_layout.is_causal = ctx().is_causal;
                spm_layout.use_attn_mask = ctx().attention_mask.has_value();
                const bool raw_profile = subclass_spm_kv_by_mha_eligible(
                    ctx().stage_plan, spm_layout, ctx().position);
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION, SIGLIP_MINIBATCH_ATTN_SITE,
                    static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                    raw_profile
                        ? 0
                        : SIGLIP_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                    {image_batch_count_, per_image_ctx, seq_len, seq_len_,
                     nq, hd}, chunk.idx);
            }
            rpu_launch_sdpa_spm_minibatch_kernel(
                k_cache, v_cache,
                0 /*MASK_NONE*/, attn_scale,
                addr_offset("q_comp").value,
                addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value, 0,
                seq_len_, nq, nq, hd,
                per_image_ctx, image_batch_count_,
                num_cores(), /*physical_kv_cores=*/8);
        } else {
            if (ctx().has_complete_physical_manifest()) {
                LayoutContext spm_layout;
                spm_layout.attention_policy =
                    AttentionExecutionPolicy::SPM_KV_BY_MHA;
                spm_layout.batch_size = ctx().batch_size;
                spm_layout.is_causal = ctx().is_causal;
                spm_layout.use_attn_mask = ctx().attention_mask.has_value();
                const bool raw_profile = subclass_spm_kv_by_mha_eligible(
                    ctx().stage_plan, spm_layout, ctx().position);
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION, SIGLIP_UNIFIED_ATTN_SITE,
                    static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                    raw_profile
                        ? 0
                        : SIGLIP_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                    {image_batch_count_, per_image_ctx, seq_len, seq_len_,
                     nq, hd}, chunk.idx);
            }
            rpu_launch_sdpa_spm_unified_kernel_v2(
                k_cache, v_cache,
                0 /*MASK_NONE*/, attn_scale,
                addr_offset("q_comp").value,
                addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value, 0,
                /*seq_q=*/seq_len, nq, nq, hd,
                /*kv_seq_len=*/seq_len_, num_cores(), /*physical_kv_cores=*/8);
        }

        // Phase 4: O_proj (row-partition, with bias) + AllReduce + Residual
        consume_manifest_route(
            FmbRouteFamily::LINEAR, SIGLIP_O_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, num_cores(),
            layer_addr(layer_idx, 0, "o_bias"), lw.o_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        if (pi05_vision_owner_ln_opt_in_) {
            consume_pi05_vision_owner_ln_route(false, chunk);
            // Reduce first, retire compact FP16 raw, then original LN2 and
            // gather. The full normalized output reuses the dead O partial.
            rpu_launch_pi05_vision_owner_layernorm_spm_kernel(
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
                addr(0, "oproj"), layer_addr(layer_idx, 0, "ln2_gamma"),
                layer_addr(layer_idx, 0, "ln2_beta"), eps_, seq_len);
        } else {
            consume_manifest_route(
                FmbRouteFamily::ALL_REDUCE, SIGLIP_ATTN_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(seq_len, h, use_pi05_ring_xor3(seq_len), num_cores()), chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
                seq_len, h, num_cores(), num_cores(), use_pi05_ring_xor3(seq_len));
            rpu_launch_layernorm_spm_kernel(
                addr(0, "input_norm"), addr(0, "oproj"),
                layer_addr(layer_idx, 0, "ln2_gamma"), layer_addr(layer_idx, 0, "ln2_beta"),
                seq_len, h, eps_, false, 0, num_cores());
        }

        // Phase 5: fc1+bias + GELU consumes the same full normalized root.
        if(pi05_vision_fc_weight_outer_opt_in_) {
            consume_pi05_vision_fc_route(false,chunk);
            rpu_launch_pi05_vision_fc_weight_outer_spm_kernel(
                Pi05VisionFcProjection::Fc1Column,
                addr(0,"oproj"),lw.fc1_w,addr(0,"fc1"),
                layer_addr(layer_idx,0,"fc1_bias"),lw.fc1_ws);
        } else {
            consume_manifest_route(
                FmbRouteFamily::LINEAR, SIGLIP_FC1_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "oproj"), lw.fc1_w, addr(0, "fc1"),
                seq_len, is_, h, 1, num_cores(),
                layer_addr(layer_idx, 0, "fc1_bias"), lw.fc1_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        }
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "fc1"), addr(0, "fc1"),
            seq_len * local_inter, ValuOpType::ADD, GeluMode::TANH, num_cores());

        // Phase 6: fc2 (row-partition, with bias) + AllReduce + Residual
        // Round 9: final output goes to "residual1" instead of "oproj" so the
        // next layer (which reads from "residual1") gets it directly via SPM
        // without needing DDR ping-pong.
        // Phase 6 uses "oproj" as temporary storage for fc2 partial results,
        // then all_reduce(oproj + input_norm) lands back in "residual1".
        if(pi05_vision_fc_weight_outer_opt_in_) {
            consume_pi05_vision_fc_route(true,chunk);
            rpu_launch_pi05_vision_fc_weight_outer_spm_kernel(
                Pi05VisionFcProjection::Fc2Row,
                addr(0,"fc1"),lw.fc2_w,addr(0,"oproj"),
                layer_addr(layer_idx,0,"fc2_bias"),lw.fc2_ws);
        } else {
            consume_manifest_route(
                FmbRouteFamily::LINEAR, SIGLIP_FC2_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "fc1"), lw.fc2_w, addr(0, "oproj"),
                seq_len, h, is_, 0, num_cores(),
                layer_addr(layer_idx, 0, "fc2_bias"), lw.fc2_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        }
        if (pi05_vision_owner_ln_opt_in_) {
            consume_pi05_vision_owner_ln_route(true, chunk);
            rpu_launch_pi05_vision_compact_residual_spm_kernel(
                addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"), seq_len);
        } else {
            consume_manifest_route(
                FmbRouteFamily::ALL_REDUCE, SIGLIP_MLP_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(seq_len, h, use_pi05_ring_xor3(seq_len), num_cores()), chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"),
                seq_len, h, num_cores(), num_cores(), use_pi05_ring_xor3(seq_len));
        }

        // ----------------------------------------------------------------
        // SPM→DDR output DMA or post-LN+projector (last layer)
        // ----------------------------------------------------------------
        if (layer_idx < static_cast<int>(num_layers()) - 1) {
            // Standard layer: write this chunk's Phase-6 output ("residual1")
            // to the inter-layer chain. In single-chunk AUTO→SPM_RESIDENT mode
            // output_to_spm=true and this is skipped (residual1 stays live for
            // the next layer's Phase 1). In multi-chunk AUTO→DDR_PINGPONG mode
            // output_to_spm=false, so emit_layer_output_dma stages each chunk at
            // its chunk.offset into the ping-pong buffer the next layer re-reads.
            if (!ctx().output_to_spm) {
                emit_layer_output_dma(layer_idx, chunk, "residual1");
            }
        } else {
            // Last layer (D-501): Post-LayerNorm + Projector.
            // Phase-6 output is in "residual1"; post-LN reads residual1 → writes
            // input_norm (SPM).
            rpu_launch_layernorm_spm_kernel(
                addr(0, "residual1"), addr(0, "input_norm"),
                addr(0, "post_ln_gamma"), addr(0, "post_ln_beta"),
                seq_len, h, eps_, false, 0, num_cores());

            // SPM → DDR temp (num_elements, NOT bytes). Shape-keyed staging slot
            // sized at chunk.len rows — an internal member Python never sees, so
            // Pitfall 4 doesn't apply. Per-shape entry stays alive for multi-
            // shape REPLAY (A→B→A keeps A's baked DMA pointer valid).
            auto key = std::make_pair(seq_len, h);
            auto& slot = temp_ddr_slots_[key];
            if (!slot.defined()) {
                slot = at::empty({seq_len, h},
                    at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
            }
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "input_norm"),
                slot.data_ptr<c10::Half>(),
                seq_len * h);

            // forward_packed already allocated the full sequence output and
            // refreshed its DMA base, including when this body is skipped on
            // fast REPLAY. All query chunks write into that one fresh owner.
            TORCH_CHECK(projector_output_.defined(),
                        "SigLIPModel: projector_output_ undefined on a non-first "
                        "chunk (chunk.offset=", chunk.offset,
                        ") — first chunk must run with offset 0.");

            // Projector Linear (DDR col-partition) writes THIS chunk's rows.
            // For multi-chunk, target a [chunk.len, proj] view at the chunk's
            // row offset. A dim-1 slice of a contiguous [1,S,P] tensor is itself
            // contiguous (dim-0 size 1), so its data_ptr lands at offset*P and
            // the kernel writes the right rows. Single-chunk → the slice is the
            // whole tensor (no-op view).
            // NOTE: do NOT .contiguous() the slice — a dim-1 slice of a
            // contiguous [1,S,P] tensor (dim-0 size 1) is already contiguous and
            // aliases projector_output_'s storage; .contiguous() would be a no-op
            // here but in general could detach into a fresh copy whose write
            // would NOT land in projector_output_. Pass the aliasing view.
            at::Tensor proj_out_chunk =
                (chunk.len == seq_len_)
                    ? projector_output_
                    : projector_output_.slice(1, chunk.offset,
                                              chunk.offset + chunk.len);
            // The DDR launcher consumes matrices. This view removes only the
            // singleton batch axis and preserves storage and the row offset.
            proj_out_chunk = proj_out_chunk.view({seq_len, projection_dim_});
            // MUTABLE write-back: projector_output_ is allocate_tracked_output
            // (fresh every forward, P4) so its data_ptr DRIFTS. A fixed DMA
            // freezes the BUILD-time dst in kd_buf and REPLAY's sync-only fast
            // path (this graph is single-segment ⇒ it always hits) never
            // rewrites it — the projector then writes the PREVIOUS forward's
            // buffer and the returned tensor is uninitialized. Same shape and
            // same fix as the patch_emb step ①/⑦ live-base sites above
            // All chunks share one base; the per-chunk row
            // offset rides in dst_offset_bytes exactly like step ⑦'s seq_off.
            // proj_b_ is a registered weight, but set_weights can REBIND it to a
            // different tensor on a live handle (set_weights only invalidates
            // model state, not the built graph). The bias DMA is the one path a
            // rebind could not reach — the weight rides a kernel REGISTER, which
            // REPLAY re-syncs. Live base so the bias follows the rebind too.
            static thread_local uint64_t projector_bias_live_base = 0;
            projector_bias_live_base =
                ::rhino_lkn::RpuGetDevAddr(proj_b_.data_ptr());
            consume_manifest_route(
                FmbRouteFamily::LINEAR, SIGLIP_PROJECTOR_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                chunk.idx);
            rpu_launch_linear_ddr_kernel(
                slot, proj_w_, proj_out_chunk, proj_b_,
                /*has_bias=*/true, /*partition=*/1,
                &projector_out_live_base_,
                /*dst_offset_bytes=*/chunk.offset * projection_dim_
                                     * (int64_t)sizeof(c10::Half),
                &projector_bias_live_base, num_cores(), {}, configured_linear_acc32_);
        }
    }

    // ========================================================================
    // plan_kv_first_chunks — D-501 kv_first_chunk_plan_fn (Site8).
    //
    // SigLIP keeps Phase 1 (KV-insert) and Phase 2 (compute) on the SAME chunk
    // size — return the compute_plan unchanged (mirror image_flow, which also
    // disables the kv_cs doubling). The auto chunk-size scan finds the largest
    // comp_cs that fits SPM; a single kv_cs simplifies SPM budgeting and avoids
    // the two-phase estimator/allocator mismatch image_flow R5 documents.
    //
    // Multi-image compute is already pinned to one full-sequence chunk in
    // forward_packed. Keep QKV identical to that canonical compute plan; FMB's
    // public FMB plan now owns the SISC/SIMC/MISC topology.
    // ========================================================================
    ChunkPlan plan_kv_first_chunks(const ChunkPlan& compute_plan) {
        return compute_plan;
    }

    // ========================================================================
    // subclass_chunk_size_valid — kernel-validity hook for the auto chunk-size
    // scan (Site10, REQUIRED override). The base default returns true (only the
    // SPM budget is checked), but the SigLIP SDPA imposes real tile/structural
    // constraints the budget predicate does NOT capture. COPY-image_flow:287-290.
    //
    // head_dim() = 80 (padded), NOT orig_head_dim_ = 72: sdpa_is_valid_chunk_size
    // hard-requires head_dim % 16 == 0 (rpu_helpers.h:358); 72%16≠0 would reject
    // EVERY cs, 80%16==0 passes. This matches declare_buffers' sdpa_compute_tiling
    // (which also uses head_dim()). mask=0 (MASK_NONE) skips the LTM-only
    // sQryAcc%16 constraints that don't apply to SigLIP's bidirectional attention.
    // nkv = nq (MHA), num_cores = NUM_CORES.
    // ========================================================================
    bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                   int64_t position) const override {
        // Packed minibatch attention consumes one full physical window.
        // A6 enumerates all capacities under the legacy override, so express
        // this owner capability here as well as in forward's fixed override.
        if (image_batch_count_ > 1 && cs != seq_len) return false;

        SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM,
                       head_dim(), num_q_heads(), num_q_heads(),
                       num_cores(), /*mask=*/0 /*MASK_NONE*/};
        if (!sdpa_is_valid_chunk_size(cfg, cs, seq_len, position)) return false;

        // Board-proven safe ceiling (2026-06-18). cs=256/512 RUN (clean, finite);
        // cs=880 HANGS on the cs-independent 8 MB SPM-block request
        // (`requested=8388608 available=0` infinite retry). Root cause: persistent
        // weights (~569 KB) + the STACKED per-chunk temps (KvInsert + Compute do
        // NOT alias, see below) push the FULL 8 MB SPM
        // block over, even though the bump-pool temps alone fit. The current
        // framework uses SPM_PLANNING_BUDGET, but its generic scope model still
        // does not express this board-observed stacked allocation. Cap cs at the proven-working 512
        // (1564 → ~4 chunks of ≤512, each with the cs=512 footprint that runs)
        // until the kernel-internal SPM headroom beyond declared temps is
        // characterized. The stacked-temp check below is the secondary defense.
        //
        // this 512 ceiling is the SINGLE-IMAGE auto-scan cap.
        // Multi-image packed (image_batch_count_>1) pins ONE chunk == seq_len via
        // set_chunk_size_override (forward_packed:624), which STILL routes through
        // this validator (fused_model_base.cpp:490 TORCH_CHECK(valid_fn(...))) — so a
        // legit 3-image 768-token packed forward (3*256) would be wrongly rejected by
        // `cs>512`, regressing the pre-MR single-block path. Gate the heuristic cap to
        // the single-image auto-scan; the multi-image forced-single-block path is
        // bounded by the stacked-temp SPM budget gate below (the real safety check),
        // which sizes the actual peak for this exact cs.
        if (image_batch_count_ <= 1 && cs > 512) return false;

        // SPM budget gate (residual-risk #3, board-confirmed 2026-06-18). The
        // framework's auto-scan budget check (estimate_temporary_total) models
        // KvInsert and Compute scopes as ALIASING (LayerWide + max(KvInsert,
        // Compute)). The real bump allocator (rpu_spm_allocator) does NOT free the
        // Phase-1 KvInsert temps (q_kv/k/v) before Phase-2 Compute allocates — the
        // two scopes STACK. Board proof: cs=944 estimated 7.78 MB (planner fits=1)
        // but really allocated 8 458 240 B → SpmAllocator OOM. Reject cs whose
        // ACTUAL stacked peak (LayerWide + KvInsert + Compute, the bump-allocator
        // truth for the aliased layout) exceeds the usable
        // profile-specific 0.97 * SPM_USABLE ceiling, so the auto-scan picks a
        // cs that fits at real allocation.
        const int64_t h   = hidden_size();
        const int64_t nq  = num_q_heads();
        const int64_t hd  = head_dim();
        const int64_t is_ = intermediate_size();
        const int64_t local_q_dim = (nq / num_cores()) * hd;
        const int64_t local_inter = is_ / mlp_tp();
        auto Aln = [](int64_t b) -> int64_t { return Align(b, 256); };
        const int64_t res = Aln(cs * h * DWIDTH);            // residual1/input_norm/oproj
        const int64_t qkv = Aln(cs * local_q_dim * DWIDTH);  // q_kv/k/v/q_comp/sdpa_out
        const int64_t fc1 = Aln(cs * local_inter * DWIDTH);
        SdpaTiling t = sdpa_compute_tiling(cfg, cs);
        const int64_t nkv_per_core = CeilDiv(nq, (int64_t)num_cores());
        const int64_t sdpa_tmp =
            Aln(t.tile_n_v16 * t.tile_k * nkv_per_core * CeilDiv(cs, t.tile_m) * 32);
        const int64_t layer_wide = 3 * res;                       // alive both phases
        const int64_t kv_insert  = 3 * qkv;                       // q_kv/k/v (Phase 1)
        const int64_t compute    = std::max(2 * qkv + sdpa_tmp,   // phase-5: q_comp+sdpa_out+tmp
                                            fc1);                 // phase-7/8: fc1
        const int64_t stacked_peak = layer_wide + kv_insert + compute;
        const int64_t budget =
            static_cast<int64_t>(SpmAllocator::SPM_USABLE * 0.97);
        if (stacked_peak > budget) return false;
        return true;
    }

    // ========================================================================
    // run_packed_patch_embed_core — pack N per-image [1,C,H,W] tensors through
    // fused_patch_embedding into one [1, N*num_patches, hidden] tensor, each at
    // its per-image seq slice. Allocates the 6 patch_emb temp buffers ONCE and
    // reuses the offsets for every image (NO reset between images): reuse keeps
    // the graph's SPM read/write dependency tracking intact so image i+1's GEMM
    // serializes after image i's output DMA, and bounds SPM to a single image's
    // footprint (3× would OOM). One reset_temporary AFTER the loop frees the
    // patch_emb temp before the encoder. Sets image_batch_count_ = N.
    //
    // Shared by both the single-tensor 4D forward (run_packed_patch_embed, which
    // slices [N,C,H,W] into the vector) and the all-RPU list path (forward_multi
    // via run_packed_patch_embed_list, which receives N distinct image tensors).
    // ========================================================================
    at::Tensor run_packed_patch_embed_core(
        const std::vector<at::Tensor>& images,
        bool consume_external_prologue) {
        if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
        int64_t N = static_cast<int64_t>(images.size());
        TORCH_CHECK(N > 0, "run_packed_patch_embed: empty image list");
        int64_t H = images[0].size(2);
        int64_t W = images[0].size(3);
        int64_t HW = H * W;
        int64_t K = pe_kh_ * pe_kw_ * pe_cin_padded_;
        int64_t outh = (H - pe_kh_) / pe_strideh_ + 1;
        int64_t outw = (W - pe_kw_) / pe_stridew_ + 1;
        int64_t npp = outh * outw;       // patches per image (256 for So400m)
        int64_t cout = pe_cout_;

        // Allocate the 6 patch_emb temp buffers ONCE and reuse the offsets for
        // every image (all images share H/W/cout). NO reset between images:
        // reusing the offsets keeps the graph's SPM dependency tracking intact
        // so image i+1's GEMM serializes after image i's output DMA, and bounds
        // SPM to a single image's footprint (3x would OOM). See
        // fused_patch_embedding doc.
        using AR = SpmAllocator::AllocRequest;
        std::vector<uint32_t> offsets = SPM_ALLOC.alloc_temporary_aliased({
            AR{pe_cin_orig_ * HW * 2,   1, 2},  // raw
            AR{pe_cin_padded_ * HW * 2, 2, 3},  // padded
            AR{HW * pe_cin_padded_ * 2, 3, 4},  // nhwc
            AR{npp * K * 2,             4, 5},  // im2col
            AR{npp * cout * 2,          5, 7},  // gemm_out
            AR{npp * cout * 2,          6, 7},  // pos_emb
        });

        // allocate_tracked_output, not a bare at::empty: `hidden` is BOTH a
        // per-forward Python-returned tensor (P4) and the live base of a
        // deferred DMA — fused_patch_embedding step (7) bakes
        // RpuGetDevAddr(out.data_ptr()) into patch_emb_output_live_base, which
        // the graph dereferences at end(), long after this frame returns
        // (docs/pitfalls.md#c-1, write side). The helper keeps the graph's own
        // reference until after execution.
        at::Tensor hidden = allocate_tracked_output({1, N * npp, cout});

        auto& graph = RpuKernelGraph::active();
        const auto graph_state = graph.state();
        const bool recording_or_replaying =
            graph_state == RpuKernelGraph::State::RECORDING ||
            graph_state == RpuKernelGraph::State::REPLAYING;
        const bool merge_patch_segment = pi05_vision_merge_segment_opt_in_ &&
            (N == 2 || N == 3) && num_cores() == 8 && npp == 256 && cout == 1152 &&
            num_layers() == 27 && intermediate_size() == 4352 &&
            num_q_heads() == 16 && num_kv_heads() == 16 && head_dim() == 80 &&
            orig_head_dim_ == 72 && pi05_vision_supported_owner();
        if (recording_or_replaying && merge_patch_segment) {
            // The reset below changes only the host allocator cursor. Keep a
            // topology marker before the patch nodes so their prepared batch
            // can be retained together with the encoder on warm REPLAY.
            graph.record_branch(
                0x5349474c49505f4dull,  // "SIGLIP_M"
                GraphSignature{}, "siglip_patch_encoder_merged");
        }

        for (int64_t i = 0; i < N; ++i) {
            // ② Flush each image at the batch driver, before its mutable input
            // DMA reads it (fused_patch_embedding step ①). A batched caller's
            // image may be a torch.cat / CPU-fallback result that PyTorch left
            // un-flushed to device DDR → the DMA would read stale bytes →
            // non-deterministic output. Idempotent for already-coherent inputs
            // (e.g. the 4D single-image .to('rpu') path), so N=1 stays
            // byte-identical. (CLAUDE.md "mutable DMA reading a torch.cat result
            // MUST be fed a flushed tensor"; memory rpu-adapter-pitfalls #5.)
            rpu_ddr_flush_force(
                const_cast<c10::Half*>(images[i].data_ptr<c10::Half>()));
            // C-1: step (1) bakes RpuGetDevAddr(images[i]) into
            // patch_emb_input_live_base[i], read at end(). `images` is a local
            // vector — on the TensorList path (forward_multi) the elements are
            // not otherwise owned past this frame, so anchor them on the graph.
            if (RpuKernelGraph::has_active()) {
                RpuKernelGraph::active().keep_alive(images[i]);
            }
            fused_patch_embedding(
                images[i], patch_emb_weight_, patch_emb_pos_emb_,
                pe_kh_, pe_kw_, pe_cin_orig_, pe_cin_padded_, pe_cout_,
                pe_strideh_, pe_stridew_,
                hidden, /*seq_off=*/i * npp, /*img_slot=*/(int)i, offsets,
                patch_embedding_live_bases_,
                consume_external_prologue ? &ctx() : nullptr, linear_acc32_);
        }
        if (recording_or_replaying && !merge_patch_segment) {
            graph.record_branch(
                0x5349474c49505f52ull,  // "SIGLIP_R"
                GraphSignature{},
                "siglip_patch_embed_spm_reset");
        }
        // Patch output is a channel-0 DMA. Encoder input broadcast joins every
        // other DMA stream to channel 0 before reading that DDR output; the
        // unchanged absolute SPM addresses preserve SDK read/write hazards
        // across this host-only cursor reset when both parts share a segment.
        SPM_ALLOC.reset_temporary();  // free patch_emb temp before the encoder
        image_batch_count_ = N;
        return hidden;
    }

    // 4D [N,C,H,W] single-tensor path (forward auto-detect): slice into N
    // contiguous [1,C,H,W] images, then pack via the shared core. A dim-0 slice
    // of a contiguous [N,C,H,W] is already contiguous, so .contiguous() is a
    // metadata no-op (no kernel/copy). N=1 stays byte-identical to before.
    at::Tensor run_packed_patch_embed(
        const at::Tensor& input, bool consume_external_prologue) {
        int64_t N = input.size(0);
        std::vector<at::Tensor> images;
        images.reserve(N);
        for (int64_t i = 0; i < N; ++i) {
            images.push_back(input.slice(0, i, i + 1).contiguous());
        }
        return run_packed_patch_embed_core(
            images, consume_external_prologue);
    }

    // All-RPU list path (forward_multi): the N camera images arrive as distinct
    // [1,C,H,W] tensors — no torch.cat upstream, no slice here.
    at::Tensor run_packed_patch_embed_list(
        at::TensorList images, bool consume_external_prologue) {
        std::vector<at::Tensor> imgs(images.begin(), images.end());
        return run_packed_patch_embed_core(
            imgs, consume_external_prologue);
    }

private:
    void require_pi05_vision_owner_ln_payloads(int64_t rows) const {
        const auto& cache = KernelCache::instance();
        const bool m768 = cache.has_loaded(KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152) &&
            cache.has_loaded(KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152);
        const bool m512 = cache.has_loaded(KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M512N1152) &&
            cache.has_loaded(KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M512N1152);
        TORCH_CHECK((rows == 0 && (m768 || m512)) || (rows == 768 && m768) ||
                        (rows == 512 && m512),
            "Pi Vision owner-LN ON requires the exact selected owner-LN and compact-residual payloads; no fallback");
    }

    std::vector<int64_t> pi05_vision_owner_ln_arguments(bool compact, int64_t rows) const {
        TORCH_CHECK(rows == 512 || rows == 768, "Pi Vision owner-LN rejects unsupported rows");
        // ABI v1: full M/N, cores, owner rows, reduce rounds/elements,
        // FP16 raw/full bytes, epsilon, affine, partial==normalized alias, epoch.
        return {1, compact ? 1 : 0, rows, 1152, 8, rows / 8, 2, 63488,
                rows / 8 * 1152 * 2, rows * 1152 * 2, 1000000,
                compact ? 0 : 1, compact ? 0 : 1, 0};
    }

    void require_pi05_vision_owner_ln_profile(int64_t rows, int64_t images) const {
        if (!pi05_vision_owner_ln_opt_in_) return;
        TORCH_CHECK(
            (images == 2 || images == 3) && rows == images * 256 && num_cores() == 8 &&
                num_layers() == 27 && layer_bias_norm_.size() == 27 &&
                hidden_size() == 1152 && intermediate_size() == 4352 &&
                num_q_heads() == 16 && num_kv_heads() == 16 && head_dim() == 80 &&
                orig_head_dim_ == 72 && projection_dim_ == 2048 && eps_ == 1e-6 &&
                pi05_ring_xor3_opt_in_ && pi05_vision_supported_owner(),
            "Pi Vision owner-LN ON requires exact 2x256 or 3x256 eight-core FP16/W8 Pi profile and epsilon 1e-6; no fallback");
        require_pi05_vision_owner_ln_payloads(rows);
    }

    void validate_pi05_vision_owner_ln_manifest(
        const FmbPhysicalExecutionManifest& manifest) const {
        int owner_routes = 0, compact_routes = 0, attention_routes = 0;
        for (const auto& route : manifest.routes) {
            const bool first = route.site_id == SIGLIP_ATTN_ALL_REDUCE_SITE;
            const bool second = route.site_id == SIGLIP_MLP_ALL_REDUCE_SITE;
            const bool owner_selector = route.family == FmbRouteFamily::ALL_REDUCE &&
                (route.selector == SIGLIP_OWNER_LN_ROUTE ||
                 route.selector == SIGLIP_OWNER_LN_COMPACT_ROUTE);
            if (!pi05_vision_owner_ln_opt_in_) {
                TORCH_CHECK(!owner_selector && (!(first || second) || route.arguments.empty()),
                    "Pi Vision owner-LN OFF rejects ON descriptor");
                continue;
            }
            TORCH_CHECK(!owner_selector || first || second,
                "Pi Vision owner-LN descriptor has a foreign collective site");
            if (first || second) {
                first ? ++owner_routes : ++compact_routes;
                TORCH_CHECK(route.family == FmbRouteFamily::ALL_REDUCE &&
                    route.selector == (second ? SIGLIP_OWNER_LN_COMPACT_ROUTE : SIGLIP_OWNER_LN_ROUTE) &&
                    route.flags == 0 && route.invocation == 0 &&
                    route.arguments == pi05_vision_owner_ln_arguments(second, manifest.physical_length),
                    "Pi Vision owner-LN route ABI mismatch");
            }
            if (route.family == FmbRouteFamily::ATTENTION) {
                ++attention_routes;
                const std::vector<int64_t> expected{image_batch_count_, 256,
                    manifest.physical_length, manifest.physical_length, 16, 80};
                TORCH_CHECK(route.site_id == SIGLIP_RAW_SPM_ATTN_SITE &&
                    route.selector == static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA) &&
                    route.flags == 0 && route.invocation == 0 && route.arguments == expected,
                    "Pi Vision owner-LN requires the current raw-SPM attention route");
            }
        }
        if (!pi05_vision_owner_ln_opt_in_) return;
        require_pi05_vision_owner_ln_profile(manifest.physical_length, image_batch_count_);
        TORCH_CHECK(manifest.state == FmbPhysicalManifestState::COMPLETE &&
            manifest.logical_length == image_batch_count_ * 256 && manifest.physical_length == manifest.logical_length &&
            manifest.execution_padding_rows == 0 && manifest.kv_logical_length == manifest.logical_length &&
            manifest.kv_insert_physical_rows == manifest.logical_length &&
            manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
            manifest.linear_accumulation == linear_accumulation_policy() &&
            owner_routes == 1 && compact_routes == 1 && attention_routes == 1,
            "Pi Vision owner-LN requires COMPLETE exact physical profile and both collective roles");
    }

    void validate_pi05_vision_owner_ln_policy(
        const FmbPrefillStageCandidate& candidate, int64_t rows, int64_t images) const {
        validate_pi05_vision_owner_ln_manifest(candidate.physical_manifest);
        if (!pi05_vision_owner_ln_opt_in_) return;
        require_pi05_vision_owner_ln_profile(rows, images);
        TORCH_CHECK(candidate.physical_manifest.physical_length == rows &&
                        images == image_batch_count_,
                    "Pi Vision owner-LN request and descriptor shape differ");
        // Same independent image spans and one full QKV/compute chunk.
        require_pi05_vision_kv_spm_only_schedule(candidate.stage_plan);
    }

    void require_pi05_vision_owner_ln_runtime(const ChunkInfo& chunk, int layer) const {
        if (!pi05_vision_owner_ln_opt_in_) return;
        require_pi05_vision_owner_ln_profile(ctx().seq_len, image_batch_count_);
        require_pi05_vision_kv_spm_only_schedule(ctx().stage_plan);
        TORCH_CHECK(ctx().has_complete_physical_manifest() &&
            ctx().attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA &&
            ctx().position == 0 && ctx().batch_size == 1 && !ctx().is_causal &&
            !ctx().attention_mask.has_value() && chunk.idx == 0 && chunk.offset == 0 &&
            chunk.len == image_batch_count_ * 256 && chunk.kv_seq_len == image_batch_count_ * 256 &&
            (layer == 0 || ctx().input_in_spm) &&
            (layer == num_layers() - 1 || ctx().output_to_spm),
            "Pi Vision owner-LN runtime lost its exact SPM-resident lifetime");
        auto& graph = RpuKernelGraph::active();
        TORCH_CHECK((graph.state() == RpuKernelGraph::State::RECORDING ||
                     graph.state() == RpuKernelGraph::State::REPLAYING) && graph.replayable(),
                    "Pi Vision owner-LN requires replayable BUILD/REPLAY");
    }

    void consume_pi05_vision_owner_ln_route(bool compact, const ChunkInfo& chunk) {
        TORCH_CHECK(pi05_vision_owner_ln_opt_in_ && ctx().has_complete_physical_manifest(),
            "Pi Vision owner-LN requires its COMPLETE manifest");
        ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
            compact ? SIGLIP_MLP_ALL_REDUCE_SITE : SIGLIP_ATTN_ALL_REDUCE_SITE,
            compact ? SIGLIP_OWNER_LN_COMPACT_ROUTE : SIGLIP_OWNER_LN_ROUTE,
            0, pi05_vision_owner_ln_arguments(compact, chunk.len), chunk.idx);
    }

    void configure_chunk_for_shape(
        int64_t seq_len, int64_t packed_image_count)
    {
        TORCH_CHECK(num_cores() == 8 ||
                        (seq_len == 256 && packed_image_count == 1),
                    "Pi0.5 SigLIP4 requires serial single-image M256 execution");
        require_pi05_vision_owner_ln_profile(seq_len, packed_image_count);
        require_pi05_vision_fc_profile(seq_len,packed_image_count);
        require_pi05_vision_kv_spm_only_profile(
            seq_len, packed_image_count);
        TORCH_CHECK(seq_len > 0,
                    "SigLIP sequence length must be positive");
        TORCH_CHECK(packed_image_count > 0,
                    "SigLIP packed image count must be positive");
        TORCH_CHECK(
            packed_image_count == 1 ||
                (seq_len % 16 == 0 && seq_len % packed_image_count == 0),
            "SigLIP packed forward: seq_len (", seq_len,
            ") must be divisible by 16 and by image_batch_count (",
            packed_image_count,
            ") so the forced single-block KV_FIRST override does not split a tail chunk");
        if (packed_image_count > 1) {
            TORCH_CHECK(configured_chunk_size_ == 0
                            || configured_chunk_size_ >= seq_len,
                        "SigLIP packed multi-image attention requires one "
                        "chunk spanning the full sequence (", seq_len,
                        "); configured chunk_size must be auto or at least the "
                        "packed sequence length, got ", configured_chunk_size_);
            image_batch_count_ = packed_image_count;
            set_chunk_size_override(seq_len);
        } else {
            image_batch_count_ = 1;
            set_chunk_size_override(configured_chunk_size_);
        }
    }

    bool pi05_vision_supported_owner() const {
        if (layer_weights_.size() != 27 || !layer_weights_.front().q_w.defined()) return false;
        const auto dtype = layer_weights_.front().q_w.scalar_type();
        if (dtype != at::kHalf && dtype != at::kChar) return false;
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {&weights.q_w, &weights.k_w, &weights.v_w,
                                      &weights.o_w, &weights.fc1_w, &weights.fc2_w})
                if (!tensor->defined() || tensor->scalar_type() != dtype) return false;
            for (const auto* scale : {&weights.q_ws, &weights.k_ws, &weights.v_ws,
                                     &weights.o_ws, &weights.fc1_ws, &weights.fc2_ws}) {
                if (dtype == at::kChar ? (!scale->defined() || scale->scalar_type() != at::kHalf)
                                     : scale->defined()) return false;
            }
        }
        return true;
    }

    void require_pi05_vision_kv_spm_only_profile(
        int64_t rows, int64_t images) const {
        if (!pi05_vision_kv_spm_only_opt_in_) return;
        TORCH_CHECK(
            (images == 2 || images == 3) && rows == images * 256 && num_cores() == 8 && num_layers() == 27 &&
                hidden_size() == 1152 && intermediate_size() == 4352 &&
                num_q_heads() == 16 && num_kv_heads() == 16 &&
                head_dim() == 80 && orig_head_dim_ == 72 &&
                projection_dim_ == 2048 && pi05_vision_supported_owner(),
            "Pi Vision KV raw-SPM-only ON requires exact 2x256 or 3x256 FP16/W8 Pi profile; no fallback");
    }

    void require_pi05_vision_kv_spm_only_schedule(
        const FmbThreeStageChunkPlan& plan) const {
        TORCH_CHECK(
            plan.chunk_mode == ChunkMode::KV_FIRST &&
                static_cast<int64_t>(plan.input.chunks.size()) == image_batch_count_ &&
                static_cast<int64_t>(plan.spans.size()) == image_batch_count_ &&
                (image_batch_count_ == 2 || image_batch_count_ == 3) &&
                plan.qkv.chunks.size() == 1 &&
                plan.compute.chunks.size() == 1 &&
                plan.boundary_policies.input ==
                    FmbSpanBoundaryPolicy::KEEP_LOCAL &&
                plan.boundary_policies.qkv ==
                    FmbSpanBoundaryPolicy::ALLOW_CROSS &&
                plan.boundary_policies.compute ==
                    FmbSpanBoundaryPolicy::ALLOW_CROSS,
            "Pi Vision KV raw-SPM-only requires input2-or-3/QKV1/compute1 schedule");
        for (int64_t image = 0; image < image_batch_count_; ++image) {
            const auto& chunk = plan.input.chunks[image];
            const auto& span = plan.spans[image];
            TORCH_CHECK(
                chunk.idx == image && chunk.offset == image * 256 &&
                    chunk.len == 256 && chunk.kv_seq_len == (image + 1) * 256 &&
                    span.offset == image * 256 && span.len == 256 &&
                    span.group_id == image,
                "Pi Vision KV raw-SPM-only requires independent 256-token image spans");
        }
        for (const auto* stage : {&plan.qkv, &plan.compute}) {
            const auto& chunk = stage->chunks.front();
            TORCH_CHECK(
                chunk.idx == 0 && chunk.offset == 0 && chunk.len == image_batch_count_ * 256 &&
                    chunk.kv_seq_len == image_batch_count_ * 256,
                "Pi Vision KV raw-SPM-only rejects split/tail/offset QKV or compute");
        }
    }

    void require_pi05_vision_kv_spm_only_candidate(
        const FmbThreeStageChunkPlan& plan, const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len, int64_t position) const {
        if (!pi05_vision_kv_spm_only_opt_in_) return;
        require_pi05_vision_kv_spm_only_profile(
            physical_len, image_batch_count_);
        require_pi05_vision_kv_spm_only_schedule(plan);
        TORCH_CHECK(
            physical_len == image_batch_count_ * 256 && logical_len == physical_len && position == 0 &&
                layout.batch_size == 1 && !layout.is_causal &&
                !layout.use_attn_mask && layout.chunk_size == physical_len &&
                layout.effective_kv_cs() == physical_len &&
                layout.attention_policy ==
                    AttentionExecutionPolicy::SPM_KV_BY_MHA &&
                subclass_spm_kv_by_mha_eligible(plan, layout, position),
            "Pi Vision KV raw-SPM-only requires the exact SPM_KV_BY_MHA physical candidate");
    }

    void validate_pi05_vision_kv_spm_only_manifest(
        const FmbPhysicalExecutionManifest& manifest) const {
        size_t kv_routes = 0;
        size_t raw_attention_routes = 0;
        for (const auto& route : manifest.routes) {
            if (route.site_id == SIGLIP_KV_INSERT_SITE) {
                ++kv_routes;
                const int64_t expected_flags =
                    pi05_vision_kv_spm_only_opt_in_
                    ? SIGLIP_KV_FLAG_RAW_SPM_ONLY
                    : SIGLIP_KV_FLAG_DDR_MIRROR;
                TORCH_CHECK(
                    route.family == FmbRouteFamily::KV_INSERT &&
                        route.flags == expected_flags,
                    pi05_vision_kv_spm_only_opt_in_
                        ? "Pi Vision KV raw-SPM-only route identity mismatch"
                        : "Pi Vision KV mirror route rejects raw-SPM-only descriptor");
                if (pi05_vision_kv_spm_only_opt_in_) {
                    TORCH_CHECK(
                        route.invocation == 0 && route.arguments.size() ==
                            kKvInsertRouteArgumentWords &&
                            route.selector == route.arguments[1] &&
                            route.arguments[0] == 2 &&
                            route.arguments[2] == manifest.physical_length &&
                            route.arguments[3] == manifest.physical_length &&
                            route.arguments[4] == 1 &&
                            route.arguments[6] == 0 &&
                            route.arguments[7] == 0 &&
                            route.arguments[8] == manifest.physical_length,
                        "Pi Vision KV raw-SPM-only KV topology is not canonical");
                }
            } else if (route.site_id == SIGLIP_RAW_SPM_ATTN_SITE) {
                ++raw_attention_routes;
                if (pi05_vision_kv_spm_only_opt_in_) {
                    const std::vector<int64_t> expected_arguments{
                        image_batch_count_, 256, manifest.physical_length, manifest.physical_length, 16, 80};
                    TORCH_CHECK(
                        route.family == FmbRouteFamily::ATTENTION &&
                            route.selector == static_cast<int64_t>(
                                AttentionExecutionPolicy::SPM_KV_BY_MHA) &&
                            route.flags == 0 && route.invocation == 0 &&
                            route.arguments == expected_arguments,
                        "Pi Vision KV raw-SPM-only attention route is not canonical");
                }
            } else if (pi05_vision_kv_spm_only_opt_in_ &&
                       (route.site_id == SIGLIP_MINIBATCH_ATTN_SITE ||
                        route.site_id == SIGLIP_UNIFIED_ATTN_SITE)) {
                TORCH_CHECK(
                    false,
                    "Pi Vision KV raw-SPM-only rejects a DDR attention route");
            }
        }
        if (!pi05_vision_kv_spm_only_opt_in_) return;
        require_pi05_vision_kv_spm_only_profile(manifest.physical_length, image_batch_count_);
        TORCH_CHECK(
            manifest.state == FmbPhysicalManifestState::COMPLETE &&
                manifest.logical_length == image_batch_count_ * 256 &&
                manifest.physical_length == manifest.logical_length &&
                manifest.execution_padding_rows == 0 &&
                manifest.kv_logical_length == manifest.logical_length &&
                manifest.kv_insert_physical_rows == manifest.logical_length &&
                manifest.graph_lifecycle ==
                    FmbGraphLifecycle::COMPOSITE_CHILD &&
                manifest.linear_accumulation ==
                    linear_accumulation_policy(),
            "Pi Vision KV raw-SPM-only requires exact COMPLETE physical profile");
        TORCH_CHECK(
            kv_routes == 1 && raw_attention_routes == 1,
            "Pi Vision KV raw-SPM-only routes are missing or duplicated");
    }

    void validate_pi05_vision_kv_spm_only_policy(
        const FmbPrefillStageCandidate& candidate, int64_t rows,
        int64_t images) const {
        if (!pi05_vision_kv_spm_only_opt_in_) return;
        require_pi05_vision_kv_spm_only_profile(rows, images);
        TORCH_CHECK(candidate.physical_manifest.physical_length == rows &&
                        images == image_batch_count_,
                    "Pi Vision raw-SPM-only request and descriptor shape differ");
        require_pi05_vision_kv_spm_only_schedule(candidate.stage_plan);
        validate_pi05_vision_kv_spm_only_manifest(
            candidate.physical_manifest);
    }

    void require_pi05_vision_kv_spm_only_runtime(
        const ChunkInfo& chunk) const {
        if (!pi05_vision_kv_spm_only_opt_in_) return;
        require_pi05_vision_kv_spm_only_profile(
            ctx().seq_len, image_batch_count_);
        require_pi05_vision_kv_spm_only_schedule(ctx().stage_plan);
        TORCH_CHECK(
            ctx().has_complete_physical_manifest() &&
                ctx().attention_policy ==
                    AttentionExecutionPolicy::SPM_KV_BY_MHA &&
                ctx().position == 0 && ctx().batch_size == 1 &&
                !ctx().is_causal && !ctx().attention_mask.has_value() &&
                chunk.idx == 0 && chunk.offset == 0 && chunk.len == image_batch_count_ * 256 &&
                chunk.kv_seq_len == image_batch_count_ * 256,
            "Pi Vision KV raw-SPM-only runtime geometry drift");
        auto& graph = RpuKernelGraph::active();
        TORCH_CHECK(
            (graph.state() == RpuKernelGraph::State::RECORDING ||
             graph.state() == RpuKernelGraph::State::REPLAYING) &&
                graph.replayable(),
            "Pi Vision KV raw-SPM-only requires replayable BUILD/REPLAY");
    }

    // One immutable opt-in per owner; no getenv or graph selection in REPLAY.
    std::vector<int64_t> pi05_vision_fc_arguments(bool second) const {
        return {1,768,second?1152:544,second?544:1152,8,second?0:1,
            80,128,second?15:7,1,1,second?1:8,374};
    }
    void require_pi05_vision_fc_profile(int64_t rows,int64_t images) const {
        if(!pi05_vision_fc_weight_outer_opt_in_)return;
        TORCH_CHECK(rows==768 && images==3 && num_layers()==27 && layer_weights_.size()==27 &&
            layer_bias_norm_.size()==27 && hidden_size()==1152 && intermediate_size()==4352 &&
            num_q_heads()==16 && num_kv_heads()==16 && head_dim()==80 && orig_head_dim_==72 &&
            projection_dim_==2048 && layer_weights_.front().q_w.scalar_type()==at::kChar,
            "Pi Vision FC weight-outer ON requires exact 3x256 W8 Pi profile; no fallback");
    }
    void require_pi05_vision_fc_schedule(const FmbThreeStageChunkPlan& plan) const {
        TORCH_CHECK(plan.chunk_mode==ChunkMode::KV_FIRST && plan.input.chunks.size()==3 &&
            plan.spans.size()==3 && plan.qkv.chunks.size()==1 && plan.compute.chunks.size()==1 &&
            plan.boundary_policies.input==FmbSpanBoundaryPolicy::KEEP_LOCAL &&
            plan.boundary_policies.qkv==FmbSpanBoundaryPolicy::ALLOW_CROSS &&
            plan.boundary_policies.compute==FmbSpanBoundaryPolicy::ALLOW_CROSS,
            "Pi Vision FC weight-outer requires input3/QKV1/compute1 schedule");
        for(int i=0;i<3;++i) {
            const auto& c=plan.input.chunks[i];const auto& span=plan.spans[i];
            TORCH_CHECK(c.idx==i && c.offset==i*256 && c.len==256 && c.kv_seq_len==(i+1)*256 &&
                span.offset==i*256 && span.len==256 && span.group_id==i,
                "Pi Vision FC weight-outer requires three independent 256-token image spans");
        }
        for(const auto* stage:{&plan.qkv,&plan.compute}) {
            const auto& c=stage->chunks.front();
            TORCH_CHECK(c.idx==0 && c.offset==0 && c.len==768 && c.kv_seq_len==768,
                "Pi Vision FC weight-outer rejects split/tail/offset compute");
        }
    }
    void validate_pi05_vision_fc_manifest(const FmbPhysicalExecutionManifest& manifest) const {
        if(!pi05_vision_fc_weight_outer_opt_in_) {
            for(const auto& r:manifest.routes)
                TORCH_CHECK(r.site_id!=SIGLIP_FC1_WEIGHT_OUTER_SITE && r.site_id!=SIGLIP_FC2_WEIGHT_OUTER_SITE,
                    "Pi Vision FC weight-outer OFF rejects ON descriptor");
            return;
        }
        require_pi05_vision_fc_profile(768,3);
        TORCH_CHECK(manifest.state==FmbPhysicalManifestState::COMPLETE &&
            manifest.logical_length==768 && manifest.physical_length==768 &&
            manifest.execution_padding_rows==0 && manifest.kv_logical_length==768 &&
            manifest.kv_insert_physical_rows==768 && manifest.graph_lifecycle==FmbGraphLifecycle::COMPOSITE_CHILD &&
            manifest.linear_accumulation==linear_accumulation_policy(),
            "Pi Vision FC weight-outer ON requires exact COMPLETE physical profile");
        for(bool second:{false,true}) {
            const int64_t site=second?SIGLIP_FC2_WEIGHT_OUTER_SITE:SIGLIP_FC1_WEIGHT_OUTER_SITE;
            int count=0;
            for(const auto& r:manifest.routes) {
                TORCH_CHECK(r.site_id!=SIGLIP_FC1_LINEAR_SITE && r.site_id!=SIGLIP_FC2_LINEAR_SITE,
                    "Pi Vision FC weight-outer ON rejects original AUTO_TILE FC routes");
                if(r.site_id==site) {
                    ++count;
                    TORCH_CHECK(r.family==FmbRouteFamily::GRAPH_SCHEDULE && r.selector==2 &&
                        r.flags==0 && r.invocation==0 && r.arguments==pi05_vision_fc_arguments(second),
                        "Pi Vision FC weight-outer route identity mismatch");
                }
            }
            TORCH_CHECK(count==1,"Pi Vision FC weight-outer requires one explicit route per role");
        }
    }
    void consume_pi05_vision_fc_route(bool second,const ChunkInfo& chunk) {
        TORCH_CHECK(pi05_vision_fc_weight_outer_opt_in_ && ctx().has_complete_physical_manifest(),
            "Pi Vision FC weight-outer requires its COMPLETE manifest");
        require_pi05_vision_fc_profile(ctx().seq_len,image_batch_count_);
        require_pi05_vision_fc_schedule(ctx().stage_plan);
        TORCH_CHECK(ctx().position==0 && ctx().batch_size==1 && !ctx().is_causal &&
            !ctx().attention_mask.has_value() && chunk.idx==0 && chunk.offset==0 &&
            chunk.len==768 && chunk.kv_seq_len==768,"Pi Vision FC weight-outer runtime shape drift");
        auto& graph=RpuKernelGraph::active();
        TORCH_CHECK((graph.state()==RpuKernelGraph::State::RECORDING ||
            graph.state()==RpuKernelGraph::State::REPLAYING) && graph.replayable(),
            "Pi Vision FC weight-outer requires a replayable BUILD/REPLAY graph");
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            second?SIGLIP_FC2_WEIGHT_OUTER_SITE:SIGLIP_FC1_WEIGHT_OUTER_SITE,
            2,0,pi05_vision_fc_arguments(second),chunk.idx);
    }
    int64_t subclass_layout_hash() const override {
        if (num_cores() != 8) {
            int64_t hash = 0;
            for (const auto value : core_profile_arguments()) hash = detail::layout_mix(hash, value);
            return configured_linear_acc32_.has_value()
                ? detail::layout_mix(hash, linear_acc32_ ? 32 : 16) : hash;
        }
        if(!pi05_vision_fc_weight_outer_opt_in_ &&
           !pi05_vision_kv_spm_only_opt_in_ && !pi05_vision_owner_ln_opt_in_)return configured_linear_acc32_.has_value() ? (linear_acc32_ ? 32 : 16) : 0;
        int64_t hash=0x5657464337363831LL;
        for(int64_t value:{int64_t(1),num_layers(),hidden_size(),intermediate_size(),
            num_q_heads(),num_kv_heads(),head_dim(),orig_head_dim_,projection_dim_,
            int64_t(pi05_ring_xor3_opt_in_),int64_t(pi05_vision_merge_segment_opt_in_),
            int64_t(pi05_vision_q_resident_opt_in_)})hash=detail::layout_mix(hash,value);
        if(pi05_vision_fc_weight_outer_opt_in_)
            for(bool second:{false,true})for(int64_t value:pi05_vision_fc_arguments(second))
                hash=detail::layout_mix(hash,value);
        if(pi05_vision_kv_spm_only_opt_in_) {
            hash=detail::layout_mix(hash,0x4b5653504d4f4e4cLL);
            hash=detail::layout_mix(hash,SIGLIP_KV_FLAG_RAW_SPM_ONLY);
        }
        if (pi05_vision_owner_ln_opt_in_) {
            hash = detail::layout_mix(hash, 0x5649534f574e4c4eLL);
            for (int64_t rows : {512, 768})
                for (bool compact : {false, true})
                    for (int64_t value : pi05_vision_owner_ln_arguments(compact, rows))
                        hash = detail::layout_mix(hash, value);
        }
        return configured_linear_acc32_.has_value()
                ? detail::layout_mix(hash, linear_acc32_ ? 32 : 16) : hash;
    }

    // ----- Model state -----
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        // Bind cold ring policy independently of per-forward image/chunk rows.
        std::vector<int64_t> identity{
            1, 0x50493035434f5354LL, 2, pi05_ring_xor3_opt_in_ ? 1 : 0};
        if (pi05_vision_merge_segment_opt_in_) {
            identity[2] = 3;
            identity.push_back(1);  // cold patch/encoder segment policy
        }
        if (pi05_vision_q_resident_opt_in_) {
            identity = {1, 0x50493035434f5354LL, 4,
                        pi05_ring_xor3_opt_in_ ? 1 : 0,
                        pi05_vision_merge_segment_opt_in_ ? 1 : 0, 1};
        }
        if(pi05_vision_fc_weight_outer_opt_in_) {
            identity={1,0x50493035434f5354LL,5,pi05_ring_xor3_opt_in_?1:0,
                pi05_vision_merge_segment_opt_in_?1:0,pi05_vision_q_resident_opt_in_?1:0,1};
            const auto first=pi05_vision_fc_arguments(false),second=pi05_vision_fc_arguments(true);
            identity.insert(identity.end(),first.begin(),first.end());
            identity.insert(identity.end(),second.begin(),second.end());
        }
        if(pi05_vision_kv_spm_only_opt_in_) {
            identity={1,0x50493035434f5354LL,6,pi05_ring_xor3_opt_in_?1:0,
                pi05_vision_merge_segment_opt_in_?1:0,
                pi05_vision_q_resident_opt_in_?1:0,
                pi05_vision_fc_weight_outer_opt_in_?1:0,
                SIGLIP_KV_FLAG_RAW_SPM_ONLY};
            if(pi05_vision_fc_weight_outer_opt_in_) {
                const auto first=pi05_vision_fc_arguments(false),second=pi05_vision_fc_arguments(true);
                identity.insert(identity.end(),first.begin(),first.end());
                identity.insert(identity.end(),second.begin(),second.end());
            }
        }
        if (pi05_vision_owner_ln_opt_in_) {
            identity[2] = 7;
            identity.insert(identity.end(), {0x5649534f574e4c4eLL, 1,
                SIGLIP_OWNER_LN_ROUTE, SIGLIP_OWNER_LN_COMPACT_ROUTE});
            for (int64_t rows : {512, 768}) {
                for (bool compact : {false, true}) {
                    const auto arguments = pi05_vision_owner_ln_arguments(compact, rows);
                    identity.insert(identity.end(), arguments.begin(), arguments.end());
                }
            }
        }
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.insert(identity.end(), {
            static_cast<int64_t>(has_patch_emb_)});
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
                &post_ln_w_, &post_ln_b_, &proj_w_, &proj_b_,
                &patch_emb_weight_, &patch_emb_pos_emb_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        if (configured_linear_acc32_.has_value())
            identity.insert(identity.end(), {0x4143433332LL, linear_acc32_ ? 1 : 0});
        return identity;
    }

    int64_t logical_intermediate_size_ = 0;
    int64_t reduced_w8_projection_mask_ = 0;
    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerBiasNorm> layer_bias_norm_;

    // Global weights
    at::Tensor post_ln_w_, post_ln_b_;
    at::Tensor proj_w_, proj_b_;

    // Intermediate DDR staging buffer (post-LN → DDR → projector). Shape-gated
    // reuse is SAFE here because this tensor is a class member that never
    // escapes to Python — no caller accumulates references to it. The
    // per-forward spm_copy_ddr_dma writes to this stable DDR address, and the
    // projector Linear (still inside the same main graph batch) reads from
    // it immediately. Shape-keyed map: each (seq_len, hidden_size) gets its
    // own slot so multi-shape REPLAY keeps every shape's DMA address valid.
    std::map<std::pair<int64_t, int64_t>, at::Tensor> temp_ddr_slots_;

    // Final projector linear output — Pitfall 4: allocated fresh per forward
    // via allocate_tracked_output({1, seq_len, projection_dim_}) in
    // forward_packed before FMB's fast-REPLAY boundary. The v2 stable-DDR reuse
    // pattern is structurally replaced by the framework's Pitfall 4 helper.
    at::Tensor projector_output_;
    uint64_t projector_out_live_base_ = 0;

    const bool pi05_vision_fc_weight_outer_opt_in_ =
        !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_VISION_FC_WEIGHT_OUTER");
    const bool pi05_ring_xor3_opt_in_ = !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_RING_XOR3");
    const bool pi05_vision_merge_segment_opt_in_ =
        !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_VISION_MERGE_SEGMENT");
    const bool pi05_vision_q_resident_opt_in_ =
        !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_VISION_Q_RESIDENT");
    const bool pi05_vision_kv_spm_only_opt_in_ =
        !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_VISION_KV_SPM_ONLY");
    const bool pi05_vision_owner_ln_opt_in_ =
        !linear_acc32_ && pi05_kernel_opt_in("RPU_PI05_VISION_OWNER_LN");
    bool use_pi05_vision_q_resident(
        int64_t rows, AttentionExecutionPolicy policy) const {
        return pi05_vision_q_resident_opt_in_ &&
            policy == AttentionExecutionPolicy::SPM_KV_BY_MHA &&
            (image_batch_count_ == 2 || image_batch_count_ == 3) &&
            rows == image_batch_count_ * 256 && num_cores() == 8 && num_layers() == 27 &&
            hidden_size() == 1152 && intermediate_size() == 4352 &&
            num_q_heads() == 16 && num_kv_heads() == 16 && head_dim() == 80 &&
            orig_head_dim_ == 72 && pi05_vision_supported_owner();
    }
    bool use_pi05_ring_xor3(int64_t rows) const {
        return pi05_ring_xor3_opt_in_ && (image_batch_count_ == 2 || image_batch_count_ == 3) &&
            rows == image_batch_count_ * 256 && num_cores() == 8 &&
            num_layers() == 27 && hidden_size() == 1152 && intermediate_size() == 4352 &&
            num_q_heads() == 16 && num_kv_heads() == 16 && head_dim() == 80 &&
            orig_head_dim_ == 72 && pi05_vision_supported_owner();
    }

    // Model config
    double eps_ = 1e-6;
    int64_t projection_dim_ = 0;
    int64_t orig_head_dim_ = 0;  // SigLIP head_dim=72, padded to 80

    // Site5 (KV_FIRST state) — mirror GemmaModel. Stages Q across the two phases
    // (Phase 1 scatters Q→DDR, Phase 2 gathers DDR→SPM "q_comp"). local_q_heads_ =
    // num_q_heads()/NUM_CORES (set in set_weights). seq_len_ = this forward's
    // full sequence length (the SDPA kv_seq_len, decoupled from per-chunk seq_q).
    //
    // keyed by seq_len so each shape keeps its OWN
    // never-reallocated DDR address. The earlier single grow-only q_ddr_buf_ would,
    // on a later larger-seq realloc, invalidate a smaller shape's already-captured
    // graph: the Q save/load use FIXED scatter DMA (rpu_launch_spm_scatter_ddr_dma /
    // ddr_scatter_spm_dma) whose kd_buf address is baked at capture and NOT refreshed
    // by sync-only REPLAY → small→large→small alternation would read/write a freed
    // address. Per-seq slots (mirror temp_ddr_slots_) are created once and never
    // reallocated, so every shape's baked address stays valid. q_ddr_slot() returns
    // this forward's slot.
    //
    // Key = {seq_len, q_width} with q_width = local_q_heads_ * head_dim() (cold-panel
    // re-review): set_weights can change local_q_heads_/head_dim on a REUSED handle,
    // and invalidate_model_state() is base-class — it does NOT clear these SigLIP
    // slots (same blind spot the old single q_ddr_buf_ had: its realloc guard only
    // checked size(0)=NUM_CORES + size(1)=seq_len, never the width). Without q_width
    // in the key, a later forward at the SAME seq_len after a width change would reuse
    // the old undersized slot while the fixed DMA stride uses the NEW width → OOB
    // write. q_width in the key forces a fresh slot per (seq_len, width).
    std::map<std::pair<int64_t, int64_t>, at::Tensor> q_ddr_slots_;
    at::Tensor& q_ddr_slot() {
        return q_ddr_slots_.at({seq_len_, local_q_heads_ * head_dim()});
    }
    int64_t local_q_heads_ = 0;
    int64_t seq_len_ = 0;

    // SigLIP-batch: number of images packed along seq this forward (1 = legacy
    // single-image path → unified_v2 SDPA; >1 → per-image minibatch SDPA). Set
    // in forward() from input.size(0); read in build_layer_subgraph.
    int64_t image_batch_count_ = 1;
    int64_t configured_chunk_size_ = 0;
    bool planning_external_patch_prologue_ = false;

    // Patch embedding params
    at::Tensor patch_emb_weight_, patch_emb_pos_emb_;
    int64_t patch_kernel_size_ = 0, patch_stride_ = 0;
    int64_t pe_kh_ = 0, pe_kw_ = 0;
    int64_t pe_cin_orig_ = 0, pe_cin_padded_ = 0, pe_cout_ = 0;
    int64_t pe_strideh_ = 0, pe_stridew_ = 0;
    bool has_patch_emb_ = false;
    PatchEmbeddingLiveBases patch_embedding_live_bases_;
};

}  // namespace v3

// =============================================================================
// Instance registry (C-01) — uses ModelHandleRegistry<v3::SigLIPModel> template.
// =============================================================================

using SigLIPRegistry = ModelHandleRegistry<v3::SigLIPModel>;

std::vector<int64_t> rpu_siglip_planner_cache_identity(int64_t handle) {
    return SigLIPRegistry::get(handle, "rpu_siglip_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_siglip_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    SigLIPRegistry::get(handle, "rpu_siglip_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_siglip_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return SigLIPRegistry::get(handle, "rpu_siglip_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_siglip_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return SigLIPRegistry::get(handle, "rpu_siglip_kvinsert_cost_domain")
        ->kvinsert_cost_domain("siglip", descriptor);
}

std::string rpu_siglip_kvinsert_cost_catalog_sha256(int64_t handle) {
    return SigLIPRegistry::get(
        handle, "rpu_siglip_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

void rpu_siglip_set_execution_core_count(int64_t handle, int64_t cores) {
    SigLIPRegistry::get(handle, "rpu_siglip_set_execution_core_count")
        ->set_execution_cores(cores);
}

std::vector<int64_t> rpu_siglip_get_execution_topology(int64_t handle) {
    return SigLIPRegistry::get(handle, "rpu_siglip_get_execution_topology")
        ->execution_topology();
}

int64_t rpu_siglip_create(const std::optional<bool>& linear_acc32) {
    return SigLIPRegistry::create(linear_acc32);
}

void rpu_siglip_destroy(int64_t handle) {
    SigLIPRegistry::destroy(handle, "rpu_siglip_destroy");
}

void rpu_siglip_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    const at::Tensor& post_ln_w, const at::Tensor& post_ln_b,
    const at::Tensor& proj_w, const at::Tensor& proj_b,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    int64_t projection_dim, double eps)
{
    std::vector<at::Tensor> empty_scales;
    SigLIPRegistry::get(handle, "rpu_siglip")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        post_ln_w, post_ln_b, proj_w, proj_b,
        num_heads, head_dim, hidden_size, intermediate_size,
        projection_dim, eps,
        empty_scales, empty_scales, empty_scales, empty_scales,
        empty_scales, empty_scales);
}

void rpu_siglip_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    const at::Tensor& post_ln_w, const at::Tensor& post_ln_b,
    const at::Tensor& proj_w, const at::Tensor& proj_b,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    int64_t projection_dim, double eps,
    at::TensorList q_ws_list, at::TensorList k_ws_list,
    at::TensorList v_ws_list, at::TensorList o_ws_list,
    at::TensorList fc1_ws_list, at::TensorList fc2_ws_list)
{
    SigLIPRegistry::get(handle, "rpu_siglip")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        post_ln_w, post_ln_b, proj_w, proj_b,
        num_heads, head_dim, hidden_size, intermediate_size,
        projection_dim, eps,
        q_ws_list, k_ws_list, v_ws_list, o_ws_list,
        fc1_ws_list, fc2_ws_list);
}

void rpu_siglip_model_set_patch_emb(
    int64_t handle,
    const at::Tensor& weight,
    const at::Tensor& pos_emb,
    int64_t kernel_size,
    int64_t stride)
{
    SigLIPRegistry::get(handle, "rpu_siglip")->set_patch_emb_params(
        weight, pos_emb, kernel_size, stride);
}

void rpu_siglip_set_chunk_size(int64_t handle, int64_t chunk_size) {
    SigLIPRegistry::get(handle, "rpu_siglip_set_chunk_size")
        ->set_configured_chunk_size(chunk_size);
}

void rpu_siglip_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    SigLIPRegistry::get(handle, "rpu_siglip_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

int64_t rpu_siglip_get_resolved_chunk_size(int64_t handle) {
    return SigLIPRegistry::get(handle, "rpu_siglip_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

int64_t rpu_siglip_resolve_chunk_size(
    int64_t handle, int64_t seq_len, int64_t packed_image_count)
{
    return SigLIPRegistry::get(handle, "rpu_siglip_resolve_chunk_size")
        ->resolve_vision_chunk_size(seq_len, packed_image_count);
}

void rpu_siglip_prepare_persistent_spm(
    int64_t handle, int64_t seq_len, int64_t packed_image_count)
{
    TORCH_CHECK(seq_len > 0 && packed_image_count > 0 &&
                    seq_len % packed_image_count == 0 &&
                    (packed_image_count == 1 || seq_len % 16 == 0),
                "siglip_prepare_persistent_spm: invalid packed image geometry");
    SigLIPRegistry::get(handle, "rpu_siglip_prepare_persistent_spm")
        ->prepare_persistent_spm(
            seq_len, /*position=*/0, /*is_causal=*/false);
}

std::vector<int64_t> rpu_siglip_resolve_stage_domain(
    int64_t handle, int64_t seq_len, int64_t packed_image_count,
    bool external_patch_prologue)
{
    return SigLIPRegistry::get(handle, "rpu_siglip_resolve_stage_domain")
        ->resolve_stage_domain(
            seq_len, packed_image_count, external_patch_prologue);
}

at::Tensor rpu_siglip_patch_embed(
    int64_t handle,
    const at::Tensor& input)
{
    return SigLIPRegistry::get(handle, "rpu_siglip")->patch_embed(input);
}

at::Tensor rpu_siglip_patch_embed_multi(
    int64_t handle,
    at::TensorList images)
{
    return SigLIPRegistry::get(handle, "rpu_siglip")->patch_embed_multi(images);
}

at::Tensor rpu_siglip_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return SigLIPRegistry::get(handle, "rpu_siglip")->forward(
        input, k_caches, v_caches, planned_stage_descriptor);
}

at::Tensor rpu_siglip_forward_packed(
    int64_t handle,
    const at::Tensor& hidden,
    int64_t image_batch_count,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return SigLIPRegistry::get(handle, "rpu_siglip")->forward_packed(
        hidden, image_batch_count, k_caches, v_caches,
        planned_stage_descriptor);
}

at::Tensor rpu_siglip_forward_multi(
    int64_t handle,
    at::TensorList images,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return SigLIPRegistry::get(handle, "rpu_siglip")->forward_multi(
        images, k_caches, v_caches, planned_stage_descriptor);
}
