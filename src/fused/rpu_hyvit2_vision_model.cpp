// rpu_hyvit2_vision_model.cpp — HYViT2-400M AnyRes ViT，all-layers-once
// (v3 FusedModelBase)。**自 `rpu_siglip_model.cpp` 派生**。
//
// 为什么派生而不是复用：HYViT2 与 SigLIP-so400m 的几何完全相同
// (hidden 1152 / 27 层 / 16 heads / head_dim 72 / intermediate 4304)，block 结构也同式
// (`x = x + attn(norm1(x))`; `x = x + mlp(norm2(x))`，ls1/ls2 与 drop_path 均为 Identity)，
// 因此 KV_FIRST 三件套、declare_buffers、preload、SDPA、MLP 全部原样保留。
// 唯一的实质差异在**尾部**：
//
//   SigLIP : residual1 --post_ln--> input_norm --DMA--> slot --projector--> out
//   HYViT2 : residual1 --------------------------DMA--> slot --projector--> out
//
// HYViT2 在动作路径上没有 post-norm（`_HYViT2VisionTransformer.forward_head` 因
// `cal_attn_pool=False` 是死代码，encoder 输出直接进 tower 外的 merger）。
// 不能用"传 identity 权重"绕过：LayerNorm(w=1,b=0) 仍执行 (x-mean)/std，
// 这是逐行非线性变换，无法用 identity 权重抵消。
// 故 `set_weights` 相比 SigLIP 少了 `post_ln_w/post_ln_b` 两个参数（24 个而非 26）。
// projector 槽接 merger 的 `proj1 [2048,1152]`（逐 token 纯 Linear，与 SigLIP projector 同语义）；
// merger 余下的 DwPooler→GELU→proj2 由独立 merger 路径承担。
//
// 其他与 SigLIP 的共性约束（padding / scale / 打包）：
//   - Python 侧把 fused qkv [3456,1152] 按 [3, heads, head_dim] 拆成 q/k/v；
//     head_dim 72→80、intermediate 4304→4352 补零（与 adapters/siglip.py 同式）。
//   - ⚠️ attention scale 用 ORIG head_dim=72，不是 padded 的 80（见下方 orig_head_dim_）。
//   - 单图路径的 seq=196、image_batch_count_=1；多图打包使用独立布局。
//
// 下面保留的注释来自 SigLIP 原文件，用于说明框架契约的来龙去脉（D-501 preload_fn、
// KV_FIRST 重构、Pitfall 2/3/4 的结构性缓解等），对本文件同样适用。
//
// ============================ 以下为 SigLIP 原始设计注释 ============================
//
#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_spm_allocator.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

// RPU_HY_VLA_FAST_REPLAY is a per-component opt-in. Skipping the host op walk
// requires stable kernel-register addresses and no body-local host side effects.
// Request changes must reach stable position/mask buffers or mutable DMA bases.
// RPU_HY_VLA_FAST_REPLAY_PRELOAD additionally skips clean-weight preload emission;
// the built graph still executes the preload DMAs into persistent SPM.
//
// RPU_HY_VLA_MASK_ONCE preserves a request's mask across layers. Its storage must
// move from Compute to LayerWide as well as extending its phase lifetime:
// KvInsert and Compute scopes may alias even when their phase ranges overlap.
//
// RPU_HY_VLA_Q_INPLACE applies to the single-chunk KV_FIRST schedule. Phase 1
// produces Q immediately before Phase 2 consumes it, so q_comp can alias q_kv
// instead of round-tripping through DDR. q_kv must be LayerWide and live through
// phase 6; otherwise SDPA scratch may overwrite it. The temporary estimate
// changes from max(2*qkv+tmp, fc1) to max(qkv+tmp, fc1).
//
// RPU_HY_VLA_KVPAD16 rounds K/V storage to a v16 capacity. Attention continues to
// use the logical KV length, so extra padded rows remain inaccessible.
// RPU_HY_VLA_FUSED_MERGER uses member-major ordering from two permute3d calls.
// Python resolves the cold choice and binds it through set_weights.
using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

constexpr int64_t HYVIT2_PATCH_INPUT_DMA_SITE = 6327224917790673981LL;
constexpr int64_t HYVIT2_PATCH_LINEAR_SITE = 5594838142388648123LL;
constexpr int64_t HYVIT2_PATCH_ALL_GATHER_SITE = 2358476529501646058LL;
constexpr int64_t HYVIT2_PATCH_POSITION_DMA_SITE = 2702679098023419389LL;
constexpr int64_t HYVIT2_PATCH_OUTPUT_DMA_SITE = 1374295323147233269LL;

enum class HyViT2PatchMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM_MUTABLE = 1,
    SPM_COPY_TO_DDR_MUTABLE = 2,
};

// =============================================================================
// v5-05 D-07: Fused patch embedding (lifted byte-equal from former
// rpu_fused_patch_embedding.cpp; that file is deleted in C2). Internal helper;
// not registered as a torch op (the fused_patch_embedding op surface is
// removed in C2). The only remaining call site is rpu_hyvit2_model.cpp's
// HYViT2VisionModel::forward 4D auto-detect path (~L278).
// =============================================================================
namespace {

constexpr int64_t kMaxPackedImages = 8;

// PATCH_EMBED_MC selects the multicore patch-embedding path. It broadcasts raw
// input, performs local pad/transpose and permute3d, runs column-partitioned GEMM,
// and all-gathers the full output. Python's weight swizzle and native core count
// must use the same cold setting.
//
// PROJ1_IN_MERGER leaves member-major [S, hidden] in the ViT tracked output and
// moves proj1 into hyvla_merger_fused. Python binds this choice to both handles
// so there is one authority for the boundary layout.
struct HYViT2ColdConfig {
    bool fast_replay = false;
    bool fast_replay_preload = false;
    bool mask_once = false;
    bool kvpad16 = false;
    bool q_inplace = false;
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
    const std::vector<uint32_t>& offsets, int nc,
    const v3::InferenceContext* physical_ctx)
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
    uint32_t gathered_off = offsets[5];
    uint32_t pos_emb_off  = offsets[6];

    uint32_t raw_addr      = SPM_ALLOC.addr(0, raw_off);
    uint32_t gemm_out_addr = SPM_ALLOC.addr(0, gemm_out_off);
    uint32_t pos_emb_addr  = SPM_ALLOC.addr(0, pos_emb_off);

    TORCH_CHECK(cout % nc == 0,
                "fused_patch_embedding: cout(", cout, ") must be divisible by "
                "num_cores(", nc, ") for the col-partition GEMM");

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
    TORCH_CHECK(img_slot >= 0 && img_slot < kMaxPackedImages,
                "fused_patch_embedding: img_slot ", img_slot,
                " out of range [0,", kMaxPackedImages, ")");
    static thread_local uint64_t patch_emb_input_live_base[kMaxPackedImages] = {0};
    patch_emb_input_live_base[img_slot] = ::rhino_lkn::RpuGetDevAddr(
        const_cast<c10::Half*>(input_nchw.data_ptr<c10::Half>()));
    if (physical_ctx != nullptr) {
        physical_ctx->consume_physical_route(
            v3::FmbRouteFamily::MUTABLE_DMA,
            HYVIT2_PATCH_INPUT_DMA_SITE,
            static_cast<int64_t>(HyViT2PatchMutableDmaRoute::
                DDR_BROADCAST_TO_SPM_MUTABLE),
            /*resolved_flags=*/0,
            {num_patches, cin_orig, cin_padded, kh, kw,
             strideh, stridew, nc}, img_slot);
    }
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &patch_emb_input_live_base[img_slot],
        /*src_offset_bytes=*/0,
        cin_orig * HW, raw_addr, /*num_cores=*/nc);

    // ===== Step ② Pad channels =====
    // nc>1: every core pads its own broadcast copy (SPMD). Steps ②③④ are
    // redundant across cores — they cost the same wall-clock as one core and
    // leave the GEMM operand resident on all `nc` cores.
    rpu_launch_pad_channel_spm(
        raw_off, padded_off, /*N=*/1, /*C=*/cin_orig * HW,
        /*pad_front=*/0, /*pad_tail=*/(cin_padded - cin_orig) * HW, nc);

    // ===== Step ③ NCHW → NHWC =====
    rpu_launch_transpose_nchw_to_nhwc_spm(
        padded_off, nhwc_off, (int)cin_padded, (int)H, (int)W, nc);

    // ===== Step ④ im2col =====
    // Multi-core: the im2col kernel is NOT a clean SPMD kernel — it splits its
    // grid by core id, so replicating it leaves every core with a DIFFERENT
    // partial. Since stride == kernel here the patches don't
    // overlap and im2col is a PURE PERMUTE, so we do it with permute3d, which
    // is a plain broadcast SPMD kernel (same family as the transpose above):
    //   nhwc[(ph*kh+i)*W + pw*kw+j][c]  →  col[ph*gw+pw][(i*kw+j)*cin+c]
    // Fixing ph leaves (i, pw, j*cin+c) → (pw, i, j*cin+c) = perm{1,0,2}, and
    // both sides have the same ph-stride (kh*gw*kw*cin), so it is gh calls at
    // the same offset on both ends.
    if (nc > 1) {
        TORCH_CHECK(strideh == kh && stridew == kw,
                    "fused_patch_embedding: the multi-core permute form of "
                    "im2col requires non-overlapping patches (stride == kernel)");
        const int64_t gh = H / kh, gw = W / kw;
        const int64_t row = kw * cin_padded;      // j*cin+c
        const int64_t plane = kh * gw * row;      // elements per ph on both sides
        const uint32_t nhwc_addr = SPM_ALLOC.addr(0, nhwc_off);
        for (int64_t p = 0; p < gh; ++p) {
            const uint32_t byte_off = (uint32_t)(p * plane * DWIDTH);
            rpu_launch_permute3d_spm_kernel(
                nhwc_addr + byte_off, SPM_ALLOC.addr(0, im2col_off) + byte_off,
                kh, gw, row, {1, 0, 2}, nc);
        }
    } else {
        rpu_launch_im2col_spm(
            nhwc_off, im2col_off,
            /*batch=*/1, (int)H, (int)W, (int)cin_padded,
            (int)kh, (int)kw,
            /*padh=*/0, /*padH=*/0, /*padw=*/0, /*padW=*/0,
            (int)strideh, (int)stridew, /*holeh=*/1, /*holew=*/1,
            /*num_cores=*/1);
    }

    // ===== Step ⑤ GEMM (col-partition over nc cores) =====
    uint32_t im2col_addr = SPM_ALLOC.addr(0, im2col_off);
    if (physical_ctx != nullptr) {
        physical_ctx->consume_physical_route(
            v3::FmbRouteFamily::LINEAR,
            HYVIT2_PATCH_LINEAR_SITE,
            static_cast<int64_t>(v3::FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {num_patches, cout, K, 1, nc, 0}, img_slot);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        im2col_addr, weight, gemm_out_addr,
        num_patches, cout, K,
        /*partition=*/1, nc, /*bias_spm_addr=*/0);

    // ===== Step ⑤b all_gather: per-core [num_patches, cout/nc] → full width =====
    // Pairs with partition=1 (column-parallel): shards are different COLUMNS of
    // one answer, so they concat. Steps ⑥⑦ then run single-core off core 0.
    uint32_t result_addr = gemm_out_addr;
    if (nc > 1) {
        result_addr = SPM_ALLOC.addr(0, gathered_off);
        const RpuAllGatherSchedule schedule =
            rpu_resolve_all_gather_schedule(cout / nc, DWIDTH);
        if (physical_ctx != nullptr) {
            physical_ctx->consume_physical_route(
                v3::FmbRouteFamily::COLLECTIVE,
                HYVIT2_PATCH_ALL_GATHER_SITE,
                static_cast<int64_t>(schedule),
                /*resolved_flags=*/0,
                {num_patches, cout / nc, DWIDTH, nc}, img_slot);
        }
        rpu_launch_all_gather_spm_kernel(
            gemm_out_addr, result_addr,
            /*n=*/num_patches, /*chunk_elems=*/cout / nc, DWIDTH, nc,
            schedule);
    }

    // ===== Step ⑥ pos_emb add =====
    // Mutable form for symmetry with step ① (pos_emb_fused.data_ptr is
    // actually stable — registered model weight — so fixed would also work).
    // rpu_ddr_flush_force(pos_emb_fused.data_ptr<c10::Half>());
    static thread_local uint64_t patch_emb_pos_live_base = 0;
    patch_emb_pos_live_base = ::rhino_lkn::RpuGetDevAddr(
        const_cast<c10::Half*>(pos_emb_fused.data_ptr<c10::Half>()));
    if (physical_ctx != nullptr) {
        physical_ctx->consume_physical_route(
            v3::FmbRouteFamily::MUTABLE_DMA,
            HYVIT2_PATCH_POSITION_DMA_SITE,
            static_cast<int64_t>(HyViT2PatchMutableDmaRoute::
                DDR_BROADCAST_TO_SPM_MUTABLE),
            /*resolved_flags=*/0,
            {num_patches, cout, 1}, img_slot);
    }
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &patch_emb_pos_live_base,
        /*src_offset_bytes=*/0,
        num_patches * cout, pos_emb_addr, /*num_cores=*/1);

    rpu_launch_eltwise_binary_spm_kernel(
        result_addr, pos_emb_addr, result_addr,
        num_patches * cout,
        ValuOpType::ADD, c10::Half(1.0f), /*num_cores=*/1);

    // ===== Step ⑦ DMA output → caller's packed slice =====
    // Mutable DMA: `out` is the caller's packed [1, total_seq, cout] tensor
    // (fresh per forward → dst drifts → mutable, not fixed). All N images share
    // the SAME base, so a single shared output live-base is consistent; each
    // image writes its num_patches rows at byte offset seq_off*cout*DWIDTH.
    rpu_ddr_flush(out.data_ptr<c10::Half>());
    static thread_local uint64_t patch_emb_output_live_base = 0;
    patch_emb_output_live_base =
        ::rhino_lkn::RpuGetDevAddr(out.data_ptr());
    if (physical_ctx != nullptr) {
        physical_ctx->consume_physical_route(
            v3::FmbRouteFamily::MUTABLE_DMA,
            HYVIT2_PATCH_OUTPUT_DMA_SITE,
            static_cast<int64_t>(HyViT2PatchMutableDmaRoute::
                SPM_COPY_TO_DDR_MUTABLE),
            /*resolved_flags=*/0,
            {seq_off * cout * DWIDTH, num_patches * cout}, img_slot);
    }
    rpu_launch_spm_copy_ddr_dma_mutable(
        result_addr,
        &patch_emb_output_live_base,
        /*dst_offset_bytes=*/seq_off * cout * DWIDTH,
        num_patches * cout);
}

static void check_hyvit2_w8a16_scale_lists(
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
                        "hyvit2_set_weights: ", name, "_scale.size()=",
                        scales.size(), " != num_layers=", n_layers);
        }
        for (int64_t i = 0; i < n_layers; ++i) {
            const at::Tensor& w = weights[i];
            TORCH_CHECK(w.device().type() == at::kPrivateUse1 && w.is_contiguous(),
                        "hyvit2_set_weights: ", name, "[", i,
                        "] must be contiguous RPU tensor");
            TORCH_CHECK(w.scalar_type() == at::kHalf || w.scalar_type() == at::kChar,
                        "hyvit2_set_weights: ", name, "[", i,
                        "] must be fp16 or int8, got ", w.scalar_type());
            if (quantized) {
                const at::Tensor& scale = scales[i];
                TORCH_CHECK(w.scalar_type() == at::kChar,
                            "hyvit2_set_weights: W8A16 mode requires int8 ",
                            name, "[", i, "], got ", w.scalar_type());
                TORCH_CHECK(scale.defined() && scale.dim() == 1
                            && scale.scalar_type() == at::kHalf
                            && scale.device().type() == at::kPrivateUse1
                            && scale.is_contiguous(),
                            "hyvit2_set_weights: ", name, "_scale[", i,
                            "] must be 1D contiguous fp16 RPU tensor");
                TORCH_CHECK(scale.numel() == w.size(0),
                            "hyvit2_set_weights: ", name, "_scale[", i,
                            "].numel()=", scale.numel(),
                            " != output dim=", w.size(0));
            } else {
                TORCH_CHECK(w.scalar_type() != at::kChar,
                            "hyvit2_set_weights: int8 ", name, "[", i,
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

// HYVIT2_FIXED_KERNEL_BASIS: remaining non-manifest launchers are fixed patch,
// norm/activation, merger, or BufferDecl transport steps. They expose no
// candidate selector; a selectable implementation must enter the manifest.

// Stable planner-owner registry IDs.  Keep these tied one-to-one to the two
// source attention calls below: packed block-diagonal and single-image.
constexpr int64_t HYVIT2_PACKED_ATTN_SITE = 5767182756843666109LL;
constexpr int64_t HYVIT2_SINGLE_ATTN_SITE = 9203248630619952644LL;
constexpr int64_t HYVIT2_PACKED_RAW_SPM_ATTN_SITE = 2830105912667431460LL;
constexpr int64_t HYVIT2_SINGLE_RAW_SPM_ATTN_SITE = 2565594785490903973LL;
constexpr int64_t HYVIT2_KV_SITE = 3316688202874054057LL;
constexpr int64_t HYVIT2_GRAPH_SCHEDULE_SITE = 3923102957416140944LL;
constexpr int64_t HYVIT2_MASK_SCHEDULE_SITE = 2185747737952178549LL;
constexpr int64_t HYVIT2_KV_PADDING_SITE = 8743411889070049254LL;
constexpr int64_t HYVIT2_MERGER_PERMUTE1_SITE = 1256999960682954151LL;
constexpr int64_t HYVIT2_MERGER_PERMUTE2_SITE = 1906212928857069507LL;

constexpr int64_t HYVIT2_KV_Q_LINEAR_SITE = 3821976038473806707LL;
constexpr int64_t HYVIT2_KV_K_LINEAR_SITE = 8147072609028639932LL;
constexpr int64_t HYVIT2_KV_V_LINEAR_SITE = 8613419147536865054LL;
constexpr int64_t HYVIT2_Q_SAVE_DMA_SITE = 4520706683310192342LL;
constexpr int64_t HYVIT2_Q_LOAD_DMA_SITE = 2558943548608657790LL;
constexpr int64_t HYVIT2_O_LINEAR_SITE = 5411123264736285894LL;
constexpr int64_t HYVIT2_ATTN_ALL_REDUCE_SITE = 7296386757428037628LL;
constexpr int64_t HYVIT2_FC1_LINEAR_SITE = 816252863600784417LL;
constexpr int64_t HYVIT2_FC2_LINEAR_SITE = 2679807282578599669LL;
constexpr int64_t HYVIT2_MLP_ALL_REDUCE_SITE = 7190094110879491471LL;
constexpr int64_t HYVIT2_TAIL_SLOT_DMA_SITE = 3131457000840871027LL;
constexpr int64_t HYVIT2_TAIL_OUTPUT_DMA_SITE = 6474014664636894258LL;
constexpr int64_t HYVIT2_PROJECTOR_LINEAR_SITE = 7630442159474739795LL;
constexpr uint32_t HYVIT2_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 | KV_INSERT_CAP_PAD16 |
    KV_INSERT_CAP_HYBRID2;
constexpr int64_t HYVIT2_KV_FLAG_DDR_MIRROR = 1;
constexpr int64_t HYVIT2_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED = 1LL << 0;

enum class HyViT2GraphScheduleRoute : int64_t {
    REEMIT_LAYER_LOOP = 1,
    FAST_REPLAY_SKIP_LAYER_LOOP = 2,
    MASK_EACH_LAYER = 3,
    MASK_FIRST_LAYER_ONLY = 4,
    PATCH_MAJOR_LAYOUT = 5,
    MERGER_MEMBER_MAJOR_LAYOUT = 6,
};

enum class HyViT2KvPaddingRoute : int64_t {
    LOGICAL_ROWS = 1,
    PAD16_ROWS = 2,
};

enum class HyViT2DmaRoute : int64_t {
    DDR_SCATTER_TO_SPM_FIXED = 1,
    SPM_SCATTER_TO_DDR_FIXED = 2,
    SPM_TO_DDR_FIXED = 3,
    KEEP_Q_IN_SPM = 4,
    SPM_TO_DDR_MUTABLE = 5,
};

// =============================================================================
// HYViT2VisionModel — v3::FusedModelBase subclass (SigLIP ViT Encoder)
// =============================================================================

class HYViT2VisionModel : public FusedModelBase {
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

    HYViT2VisionModel() = default;

    void set_runtime_config(
        bool fast_replay, bool fast_replay_preload, bool mask_once,
        bool kvpad16, bool q_inplace) {
        TORCH_CHECK(
            !cold_config_bound_ && !production_config_bound_,
            "hyvit2_set_runtime_config must run exactly once before "
            "set_weights");
        cold_config_ = {
            fast_replay, fast_replay_preload, mask_once, kvpad16, q_inplace};
        cold_config_bound_ = true;
        invalidate_model_state();
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
        const at::Tensor& proj_w, const at::Tensor& proj_b,
        int64_t num_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        int64_t projection_dim, double eps,
        bool fused_merger, bool proj1_in_merger,
        int64_t patch_embed_cores,
        at::TensorList q_ws_list, at::TensorList k_ws_list,
        at::TensorList v_ws_list, at::TensorList o_ws_list,
        at::TensorList fc1_ws_list, at::TensorList fc2_ws_list)
    {
        TORCH_CHECK(
            cold_config_bound_,
            "hyvit2_set_weights requires an explicit immutable runtime "
            "config snapshot");
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "hyvit2_set_weights: empty weight lists");
        TORCH_CHECK(num_heads > 0 && head_dim > 0 && hidden_size > 0
                    && intermediate_size > 0 && projection_dim > 0,
                    "hyvit2_set_weights: dim params must be positive");
        TORCH_CHECK(patch_embed_cores == 1 || patch_embed_cores == NUM_CORES,
                    "hyvit2_set_weights: patch_embed_cores must be 1 or ",
                    NUM_CORES, ", got ", patch_embed_cores);
        if (production_config_bound_) {
            TORCH_CHECK(
                merger_member_major_ == fused_merger &&
                    proj1_in_merger_ == proj1_in_merger &&
                    patch_embed_cores_ == patch_embed_cores,
                "hyvit2_set_weights: production configuration is immutable "
                "for a native handle");
        }

        // All 16 per-layer lists must have the same length
        auto check_list = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "hyvit2_set_weights: ", n, ".size()=", l.size(),
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

        // Global tensor checks — proj is always fp16, DMA'd as Half
        // (add fp16 + PrivateUse1 + contiguous).
        auto check_global_fp16_rpu = [&](const at::Tensor& t, const char* name) {
            TORCH_CHECK(t.defined(), "hyvit2_set_weights: ", name, " must be defined");
            TORCH_CHECK(t.scalar_type() == at::kHalf
                        && t.device().type() == at::kPrivateUse1
                        && t.is_contiguous(),
                        "hyvit2_set_weights: ", name,
                        " must be fp16 contiguous RPU tensor, got dtype=",
                        t.scalar_type(), " device=", t.device().type(),
                        " contiguous=", t.is_contiguous());
        };
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
                            "hyvit2_set_weights: ", name, "[", i, "] is undefined");
                TORCH_CHECK(list[i].dim() == expected_rank,
                            "hyvit2_set_weights: ", name, "[", i, "] must be ",
                            expected_rank, "D, got ", list[i].dim(), "D");
                if (require_fp16_rpu) {
                    TORCH_CHECK(list[i].scalar_type() == at::kHalf
                                && list[i].device().type() == at::kPrivateUse1
                                && list[i].is_contiguous(),
                                "hyvit2_set_weights: ", name, "[", i,
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
        check_hyvit2_w8a16_scale_lists(
            N,
            q_w_list, k_w_list, v_w_list, o_w_list, fc1_w_list, fc2_w_list,
            q_ws_list, k_ws_list, v_ws_list, o_ws_list, fc1_ws_list, fc2_ws_list);

        // SigLIP uses MHA: num_kv_heads == num_q_heads
        set_model_params(num_heads, num_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;
        projection_dim_ = projection_dim;

        // orig_head_dim = hidden_size / num_heads (= 72 for SigLIP, NOT from
        // padded weight shape which would give 80). v2 reference:
        // rpu_hyvit2_fused_encoder_layer.cpp:584.
        orig_head_dim_ = hidden_size / num_heads;

        // Site9 (KV_FIRST): per-core local Q head count for q_ddr_buf_ staging
        // and the Phase1 Q→DDR / Phase2 DDR→Q scatter/gather (mirror Gemma's
        // local_q_heads_ = num_q_heads / attn_tp()). SigLIP fixes tp = NUM_CORES.
        local_q_heads_ = num_q_heads() / NUM_CORES;

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
        proj_w_ = proj_w;
        proj_b_ = proj_b;

        merger_member_major_ = fused_merger;
        proj1_in_merger_ = proj1_in_merger;
        patch_embed_cores_ = static_cast<int>(patch_embed_cores);
        production_config_bound_ = true;

        invalidate_model_state();  // D-503: last non-empty statement of set_weights
    }

    // ========================================================================
    // set_patch_emb_params — optional patch embedding configuration
    // ========================================================================
    void set_patch_emb_params(const at::Tensor& weight, const at::Tensor& pos_emb,
                              int64_t kernel_size, int64_t stride)
    {
        TORCH_CHECK(kernel_size > 0 && stride > 0,
                    "hyvit2_set_patch_emb: kernel_size and stride must be positive");
        TORCH_CHECK(weight.defined() && weight.dim() == 2 &&
                        weight.size(0) > 0 && weight.size(1) > 0 &&
                        weight.scalar_type() == at::kHalf &&
                        weight.device().type() == at::kPrivateUse1 &&
                        weight.is_contiguous(),
                    "hyvit2_set_patch_emb: weight must be nonempty contiguous FP16 RPU [cout,K]");
        TORCH_CHECK(kernel_size <= weight.size(1) / kernel_size,
                    "hyvit2_set_patch_emb: kernel area exceeds weight K");
        const int64_t kernel_area = kernel_size * kernel_size;
        TORCH_CHECK(weight.size(1) % kernel_area == 0 &&
                        weight.size(1) / kernel_area >= 3 &&
                        (weight.size(1) / kernel_area) % 16 == 0,
                    "hyvit2_set_patch_emb: weight K must contain RGB patches "
                    "with padded channels divisible by 16 for NCHW-to-NHWC");
        TORCH_CHECK(pos_emb.defined() && pos_emb.numel() > 0 &&
                        pos_emb.scalar_type() == at::kHalf &&
                        pos_emb.device().type() == at::kPrivateUse1 &&
                        pos_emb.is_contiguous() &&
                        pos_emb.numel() % weight.size(0) == 0,
                    "hyvit2_set_patch_emb: pos_emb must contain contiguous FP16 RPU rows of cout elements");

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
        at::IntArrayRef planned_stage_descriptor = {})
    {
        TORCH_CHECK(num_layers() > 0,
                    "HYViT2VisionModel::forward called before set_weights");
        TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                    "HYViT2VisionModel::forward: input must be on RPU device");
        TORCH_CHECK(input.is_contiguous(),
                    "HYViT2VisionModel::forward: input must be contiguous");
        TORCH_CHECK(input.dim() == 3 || input.dim() == 4,
                    "HYViT2 forward expects 3D [B,S,H] or 4D [B,C,H,W] input, got ",
                    input.dim(), "D");
        if (input.dim() == 4) {
            TORCH_CHECK(has_patch_emb_,
                        "4D HYViT2 input requires patch embedding params — "
                        "call hyvit2_model_set_patch_emb before forward with 4D input");
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
            const int64_t patch_rows = validate_patch_inputs(
                {input}, /*allow_batched_tensor=*/true);
            TORCH_CHECK(
                !planned_stage_descriptor.empty(),
                "HYViT2 patch-embedding forward requires its native A6 "
                "stage descriptor");
            const auto prepared_planned = prepare_stage_candidate(planned_stage_descriptor);
            const auto& planned = prepared_planned->candidate();
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

        return forward_packed(hidden_states, packed_image_count, k_caches,
                              v_caches, planned_stage_descriptor);
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
        at::IntArrayRef planned_stage_descriptor = {})
    {
        TORCH_CHECK(num_layers() > 0,
                    "HYViT2VisionModel::forward_multi called before set_weights");
        const int64_t patch_rows = validate_patch_inputs(
            images, /*allow_batched_tensor=*/false);

        // Patch-embed + encoder are one physical candidate.  The prologue
        // consumes its routes before run_all_layers transfers the same
        // manifest and verifies every body receipt.
        TORCH_CHECK(
            !planned_stage_descriptor.empty(),
            "HYViT2 patch-embedding forward requires its native A6 stage "
            "descriptor");
        const auto prepared_planned = prepare_stage_candidate(planned_stage_descriptor);
        const auto& planned = prepared_planned->candidate();
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
        validate_patch_inputs({input}, /*allow_batched_tensor=*/true);
        return run_packed_patch_embed(
            input, /*consume_external_prologue=*/false);
    }

    at::Tensor patch_embed_multi(at::TensorList images) {
        validate_patch_inputs(images, /*allow_batched_tensor=*/false);
        return run_packed_patch_embed_list(
            images, /*consume_external_prologue=*/false);
    }

    at::Tensor forward_packed(
        const at::Tensor& hidden_states,
        int64_t packed_image_count,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        at::IntArrayRef planned_stage_descriptor = {})
    {
        TORCH_CHECK(hidden_states.dim() == 3,
                    "HYViT2VisionModel::forward_packed: hidden input must be 3D, got ",
                    hidden_states.dim(), "D");
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "HYViT2VisionModel::forward_packed: hidden input must be on RPU");
        TORCH_CHECK(hidden_states.is_contiguous(),
                    "HYViT2VisionModel::forward_packed: hidden input must be contiguous");
        TORCH_CHECK(packed_image_count > 0,
                    "HYViT2VisionModel::forward_packed: packed_image_count must be positive");
        image_batch_count_ = packed_image_count;

        int64_t seq_len = hidden_states.size(1);

        // Site5 (KV_FIRST): seq_len_ is this forward's full sequence length (the SDPA
        // kv_seq_len). The STABLE per-shape Q staging slot is allocated BELOW, AFTER
        // the shape-validity checks (cold-panel re-review #edge: don't allocate before
        // validation — an invalid packed shape would otherwise leak a slot).
        seq_len_ = seq_len;

        // Minibatch (N>1 packed images) mutual-exclusion with chunking: the
        // per-image minibatch SDPA launcher asserts a SINGLE chunk spanning the
        // whole packed sequence. plan_kv_first_chunks (Site8) forces the
        // KV-insert plan to one chunk, but the framework's COMPUTE chunk plan is
        // computed independently by compute_chunks_impl — so for N>1 we also pin
        // the compute chunk size to the full sequence via the override (which
        // bypasses the auto-scan, exactly like the retired D-508 single-chunk
        // path did). N==1 (HALO single-image) leaves the override at 0 so the
        // auto-scan can pick a fitting comp_cs and token-chunk the 1564 path.
        //
        // the override is floored to (override/16)*16
        // (fused_model_base.cpp:487). A packed seq not a multiple of 16 — or not
        // divisible by image_batch_count_ — would split a TAIL chunk, violating the
        // single-block minibatch assertion (TORCH_CHECK(chunk.len==seq_len_) in the
        // KV_FIRST body below) deep on-board where it is hard to root-cause. Fail
        // fast HERE with a clear shape error. SigLIP packs per-image npp(=256)
        // tokens so seq_len = N*256 is 16-aligned in practice; this guards the
        // invariant the forced-single-block override silently relies on.
        // 2026-08-05: 原先还要求 seq_len % 16 == 0, 理由是 override 会被
        // (v/16)*16 下取整 (fused_model_base.cpp:500) ⇒ 传 seq_len 会得到一个
        // **小于** seq_len 的 cs, 从而切出 tail chunk 并违反下面的单块断言。
        // 传 **ceil16(seq_len)** 就没有这个问题: 框架把它 clamp 到
        // hi=ceil16(seq_len) (:501), 再按 len=min(cs, seq_len-off) 建块 ⇒
        // 恰好一块、len==seq_len。于是 16 整除这条约束可以去掉 ——
        // Hy-VLA 的 3 相机 × 196 token = 588 (588%16=12) 因此得以走 packed 路径。
        // 仍然要求能被 image_batch_count_ 整除: minibatch SDPA 的
        // per_image_ctx = seq_len / N 必须是整数。
        TORCH_CHECK(
            image_batch_count_ == 1 || (seq_len % image_batch_count_ == 0),
            "HYViT2 packed forward: seq_len (", seq_len,
            ") must be divisible by image_batch_count (", image_batch_count_,
            ") — minibatch SDPA 的 per_image_ctx 必须整除");
        // A production descriptor carries the selected geometry.  Keep the
        // old direct entry usable before the execution guard is enabled, but
        // do not mutate a guarded handle during forward.
        const int64_t saved_chunk_override = get_chunk_size_override();
        const bool legacy_geometry = planned_stage_descriptor.empty();
        if (!legacy_geometry) {
            const auto prepared_planned = prepare_stage_candidate(planned_stage_descriptor);
            const auto& planned = prepared_planned->candidate();
            TORCH_CHECK(
                planned.physical_manifest.state ==
                        FmbPhysicalManifestState::COMPLETE &&
                    planned.stage_plan.qkv.chunks.size() == 1,
                "RPU_PLANNER_REJECT:CAPABILITY: HYViT2 production descriptor "
                "requires one frozen KV_FIRST segment plan");
        }
        if (legacy_geometry) {
            set_chunk_size_override(
                image_batch_count_ > 1 ? ((seq_len + 15) / 16) * 16 : 0);
        }
        auto restore_chunk_override = c10::make_scope_exit([&] {
            if (legacy_geometry) set_chunk_size_override(saved_chunk_override);
        });

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
                        "HYViT2VisionModel: q_ddr_slots_ exceeded 128 distinct (seq_len,width) "
                        "shapes on one handle — likely a runaway shape loop (per-shape Q "
                        "staging is never freed for replay-address stability)");
            q_ddr_slots_.emplace(q_key, at::empty(
                {(int64_t)NUM_CORES, seq_len, q_width},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1)));
        }

        // GUARD (design note §2 三件套 ①): host-side per-chunk SPM budget
        // pre-check BEFORE dispatch. The probe note proved a naive single-chunk
        // 1564-token forward exhausts the 8 MB SPM block and enters an infinite
        // allocator retry that POISONS the whole RPU board (run-queue wedged
        // until reboot). The framework's auto chunk-size scan
        // (compute_chunks_impl) already rejects an over-budget seq with a clear
        // TORCH_CHECK on the AUTO path we now take (set_chunk_size_override is
        // gone), so this is a defense-in-depth fail-fast: if even the SMALLEST
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
            const int64_t usable =
                static_cast<int64_t>(SpmAllocator::SPM_USABLE * 0.99);
            TORCH_CHECK(peak <= usable,
                        "HYViT2VisionModel::forward_packed: even the minimum chunk "
                        "(cs=16) per-chunk SPM peak (", peak, " B) exceeds the "
                        "usable SPM budget (", usable, " B). Refusing to dispatch "
                        "— a naive oversize forward would wedge the RPU board. "
                        "seq_len=", seq_len, ", hidden=", hidden_size(), ".");
        }

        const int64_t rows_per_image = seq_len / image_batch_count_;
        std::vector<ChunkInfo> input_chunks;
        input_chunks.reserve(image_batch_count_);
        std::vector<FmbExecutionSpan> spans;
        spans.reserve(image_batch_count_);
        for (int64_t image = 0; image < image_batch_count_; ++image) {
            const int64_t offset = image * rows_per_image;
            input_chunks.push_back(
                {static_cast<int>(image), offset, rows_per_image,
                 offset + rows_per_image});
            spans.push_back(
                {image * rows_per_image, rows_per_image, image});
        }
        const FmbStageBoundaryPolicies boundary_policies{
            FmbSpanBoundaryPolicy::KEEP_LOCAL,
            FmbSpanBoundaryPolicy::ALLOW_CROSS,
            FmbSpanBoundaryPolicy::ALLOW_CROSS};
        // Fast replay can skip every layer callback. Prepare caller-visible
        // storage here on every invocation, then rebind the retained DMA nodes
        // through member addresses that remain valid for this handle's lifetime.
        const bool p1m = proj1_in_merger_ && merger_member_major_;
        const int64_t projector_out_dim = p1m ? hidden_size() : projection_dim_;
        at::Tensor result;
        try {
            projector_output_ = allocate_tracked_output(
                {1, seq_len_, projector_out_dim});
            projector_out_live_base_ =
                ::rhino_lkn::RpuGetDevAddr(projector_output_.data_ptr());
            projector_bias_live_base_ = p1m ? 0 :
                ::rhino_lkn::RpuGetDevAddr(proj_b_.data_ptr());
            if (planned_stage_descriptor.empty()) {
                result = run_all_layers(
                    hidden_states, k_caches, v_caches,
                    /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false,
                    input_chunks, spans, boundary_policies);
            } else {
                result = run_all_layers(
                    hidden_states, k_caches, v_caches,
                    /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false,
                    /*planned_chunk_size=*/0, planned_stage_descriptor);
            }
        } catch (...) {
            projector_output_ = at::Tensor{};
            projector_out_live_base_ = 0;
            projector_bias_live_base_ = 0;
            throw;
        }

        // Deferred execution owns the fresh output until Graph completion.
        // Flush for CPU coherency before returning it to Python.
        if (projector_output_.defined()) {
            rpu_ddr_flush(projector_output_.data_ptr<c10::Half>());
            return projector_output_;
        }
        return result;
    }

    std::vector<int64_t> resolve_stage_domain(
        int64_t seq_len, int64_t image_batch_count,
        bool external_patch_prologue) {
        TORCH_CHECK(seq_len > 0 && image_batch_count > 0 &&
                        seq_len % image_batch_count == 0,
                    "hyvit2_resolve_stage_domain: seq_len must be positive "
                    "and divisible by image_batch_count");
        const int64_t saved_seq_len = seq_len_;
        const int64_t saved_image_batch_count = image_batch_count_;
        const int64_t saved_chunk_override = get_chunk_size_override();
        auto restore = c10::make_scope_exit([&] {
            seq_len_ = saved_seq_len;
            image_batch_count_ = saved_image_batch_count;
            set_chunk_size_override(saved_chunk_override);
        });
        seq_len_ = seq_len;
        image_batch_count_ = image_batch_count;
        set_chunk_size_override(0);
        const bool saved_external_patch_prologue =
            planning_external_patch_prologue_;
        planning_external_patch_prologue_ = external_patch_prologue;
        auto restore_external_patch_prologue = c10::make_scope_exit([&] {
            planning_external_patch_prologue_ =
                saved_external_patch_prologue;
        });

        const int64_t rows_per_image = seq_len / image_batch_count;
        std::vector<ChunkInfo> input_chunks;
        std::vector<FmbExecutionSpan> spans;
        input_chunks.reserve(image_batch_count);
        spans.reserve(image_batch_count);
        for (int64_t image = 0; image < image_batch_count; ++image) {
            const int64_t offset = image * rows_per_image;
            input_chunks.push_back(
                {static_cast<int>(image), offset, rows_per_image,
                 offset + rows_per_image});
            spans.push_back({offset, rows_per_image, image});
        }
        const FmbStageBoundaryPolicies boundary_policies{
            FmbSpanBoundaryPolicy::KEEP_LOCAL,
            FmbSpanBoundaryPolicy::ALLOW_CROSS,
            FmbSpanBoundaryPolicy::ALLOW_CROSS};
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                seq_len, /*position=*/0, /*attention_mask=*/std::nullopt,
                /*is_causal=*/false, input_chunks, spans, boundary_policies));
    }

protected:
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            seq_len_, image_batch_count_,
            planning_external_patch_prologue_);
    }

    // ========================================================================
    // static_config — D-501: uses SIGLIP_WEIGHTS for preload, SIGLIP_COMPUTE for layers
    //
    // Registers the pointer-to-member `&HYViT2VisionModel::emit_preload_weights`
    // (cast to FusedModelBase pointer-to-member) on the preload-fn slot.
    // Framework dispatches via std::invoke inside a framework-managed
    // GRAPH_CACHE.begin(SIGLIP_WEIGHTS) / end pair (EXT-5 locked).
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers       = num_layers();

        // D-501 one-shot .preload_fn: replaces v2's `build_preload_subgraph`
        // virtual. Framework wraps the callback body in ONE
        // GRAPH_CACHE.begin(SIGLIP_WEIGHTS)/end pair. The body emits only
        // rpu_launch_* DMAs — no subclass-side batch-context calls (EXT-5).
        // PRESERVED through the SEQUENTIAL→KV_FIRST refactor (Site3): SigLIP is
        // the only KV_FIRST consumer that ALSO uses one-shot preload_fn (Gemma
        // uses per-buffer .preload_callback instead) — both coexist fine here.
        cfg.preload_fn = static_cast<void(FusedModelBase::*)()>(
                             &HYViT2VisionModel::emit_preload_weights);

        // D-501 KV_FIRST: two pointer-to-member slots for the two-phase dispatch
        // (mirror GemmaModel::static_config). The static_cast is required because
        // the base struct types the slots as pointer-to-member-of-FusedModelBase
        // (std::invoke resolves to the concrete subclass at runtime).
        // SigLIP has NO rope and NO attention mask (MASK_NONE), so the bodies are
        // simplified vs Gemma — see emit_kv_first_body / plan_kv_first_chunks.
        cfg.kv_first_fn = static_cast<void(FusedModelBase::*)(int, const ChunkInfo&)>(
                              &HYViT2VisionModel::emit_kv_first_body);
        cfg.kv_first_chunk_plan_fn =
            static_cast<ChunkPlan(FusedModelBase::*)(const ChunkPlan&)>(
                &HYViT2VisionModel::plan_kv_first_chunks);

        // SigLIP REQUIRES single group (matches v2 design at
        // rpu_hyvit2_fused_encoder_layer.cpp:634-645). With cross_batch < num_layers,
        // groups that share (group_size, contains_last_layer) cache_key force REPLAY
        // of cached kernels with different layer-specific pointers (weights, KV caches,
        // per-layer SPM offsets). The cursor mechanism updates registers via setup_regs,
        // but produces non-deterministic output between forwards (verified Section 3
        // failure: rel_diff=0.49 between same-input forwards with cross_batch=12 default).
        //
        // The Qwen3 global runtime knob (g_cross_layer_batch_size, default 12) leaks into
        // SigLIP if not explicitly set. Other models (Qwen3, Gemma) work because their
        // tests explicitly set cross_batch >= num_layers (test_qwen3_decode.py sets 36).
        // SigLIP enforces this invariant in C++ to be defensive.
        cfg.cross_layer_batch_size = num_layers();
        cfg.fast_replay_skip_layer_loop = cold_config_.fast_replay;
        cfg.fast_replay_skip_preload = cold_config_.fast_replay_preload;
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
        cfg.inter_layer_io = InterLayerIO::AUTO;
        // RAW_SPM is descriptor-owned: legacy/no-descriptor entry points keep
        // the historical DDR route and cannot silently select a physical ABI.
        cfg.attention_policy = ctx().has_complete_physical_manifest()
            ? AttentionExecutionPolicy::AUTO
            : AttentionExecutionPolicy::DDR_KV;
        return cfg;
    }

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = position + logical_len;
        manifest.kv_insert_physical_rows = physical_len;
        manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation = FmbLinearAccumulationPolicy::ACC16;

        LayoutContext spm_layout = layout;
        spm_layout.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const bool raw_spm_eligible = subclass_spm_kv_by_mha_eligible(
            plan, spm_layout, position);
        const bool raw_spm = layout.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        TORCH_CHECK(
            !raw_spm || raw_spm_eligible,
            "RPU_PLANNER_REJECT:CAPABILITY: HYViT2 raw-SPM attention was "
            "selected without one full 196-row/image schedule");

        const int64_t h = hidden_size();
        const int64_t nq = num_q_heads();
        const int64_t hd = head_dim();
        const int64_t tp = NUM_CORES;
        const int64_t local_q_dim = (nq / tp) * hd;
        const int64_t inter = intermediate_size();
        const int64_t local_inter = inter / tp;
        const int64_t compute_cs = layout.chunk_size;
        const int64_t kv_cs = layout.effective_kv_cs();
        const bool q_inplace = cold_config_.q_inplace &&
            compute_cs == kv_cs && compute_cs >= physical_len &&
            physical_len > 0;
        const int64_t kv_storage_rows = cold_config_.kvpad16
            ? Align(kv_cs, int64_t{16}) : 0;
        const bool p1m = proj1_in_merger_ && merger_member_major_;

        const auto append = [&](FmbRouteFamily family, int64_t site_id,
                                int64_t selector,
                                std::vector<int64_t> arguments = {},
                                int64_t flags = 0,
                                int64_t invocation = 0) {
            manifest.routes.push_back({site_id, family, selector, flags,
                                       std::move(arguments), invocation});
        };
        const auto append_linear = [&](int64_t site_id,
                                       std::vector<int64_t> arguments,
                                       int64_t invocation) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                   std::move(arguments), /*flags=*/0, invocation);
        };

        append(
            FmbRouteFamily::GRAPH_SCHEDULE, HYVIT2_GRAPH_SCHEDULE_SITE,
            static_cast<int64_t>(
                cold_config_.fast_replay
                    ? HyViT2GraphScheduleRoute::FAST_REPLAY_SKIP_LAYER_LOOP
                    : HyViT2GraphScheduleRoute::REEMIT_LAYER_LOOP),
            {cold_config_.fast_replay ? 1 : 0,
             cold_config_.fast_replay_preload ? 1 : 0,
             num_layers()});

        for (const ChunkInfo& kv_chunk : plan.qkv.chunks) {
            const int64_t invocation = kv_chunk.idx;
            // SPM capacity can exceed this chunk (especially the final tail).
            // The KV route describes only the rows actually written to DDR.
            const int64_t kv_physical_rows = cold_config_.kvpad16 &&
                    (position + kv_chunk.offset) % 16 == 0
                ? Align(kv_chunk.len, int64_t{16}) : kv_chunk.len;
            const KvInsertSegmentPlan kv_plan =
                resolve_kvinsert_plan_auto(
                    HYVIT2_KV_SITE, manifest.graph_lifecycle,
                    position + kv_chunk.offset, kv_chunk.len,
                    kv_physical_rows, tp, nq, hd,
                    HYVIT2_KV_CAPABILITIES);
            const KvInsertRouteArguments kv_arguments =
                rpu_kvinsert_route_arguments(kv_plan, tp, nq, hd);
            manifest.kv_insert_physical_rows = std::max(
                manifest.kv_insert_physical_rows,
                kv_chunk.offset + kv_plan.physical_rows());

            append_linear(
                HYVIT2_KV_Q_LINEAR_SITE,
                {kv_chunk.len, nq * hd, h, 1, tp}, invocation);
            append_linear(
                HYVIT2_KV_K_LINEAR_SITE,
                {kv_chunk.len, nq * hd, h, 1, tp}, invocation);
            append_linear(
                HYVIT2_KV_V_LINEAR_SITE,
                {kv_chunk.len, nq * hd, h, 1, tp}, invocation);
            append(
                FmbRouteFamily::KV_INSERT, HYVIT2_KV_PADDING_SITE,
                static_cast<int64_t>(
                    kv_physical_rows > kv_chunk.len
                        ? HyViT2KvPaddingRoute::PAD16_ROWS
                        : HyViT2KvPaddingRoute::LOGICAL_ROWS),
                {cold_config_.kvpad16 ? 1 : 0, kv_chunk.len,
                 kv_storage_rows, kv_physical_rows, tp, nq, hd},
                /*flags=*/0, invocation);
            append(
                FmbRouteFamily::KV_INSERT, HYVIT2_KV_SITE,
                static_cast<int64_t>(kv_plan.route()),
                std::vector<int64_t>(kv_arguments.begin(), kv_arguments.end()),
                HYVIT2_KV_FLAG_DDR_MIRROR, invocation);
            append(
                FmbRouteFamily::MUTABLE_DMA, HYVIT2_Q_SAVE_DMA_SITE,
                static_cast<int64_t>(
                    q_inplace ? HyViT2DmaRoute::KEEP_Q_IN_SPM
                              : HyViT2DmaRoute::SPM_SCATTER_TO_DDR_FIXED),
                {cold_config_.q_inplace ? 1 : 0, q_inplace ? 1 : 0,
                 kv_chunk.offset, kv_chunk.len,
                 kv_chunk.len * local_q_dim,
                 physical_len * local_q_dim * DWIDTH, tp},
                /*flags=*/0, invocation);
        }

        for (const ChunkInfo& chunk : plan.compute.chunks) {
            const int64_t invocation = chunk.idx;
            append(
                FmbRouteFamily::MUTABLE_DMA, HYVIT2_Q_LOAD_DMA_SITE,
                static_cast<int64_t>(
                    q_inplace ? HyViT2DmaRoute::KEEP_Q_IN_SPM
                              : HyViT2DmaRoute::DDR_SCATTER_TO_SPM_FIXED),
                {cold_config_.q_inplace ? 1 : 0, q_inplace ? 1 : 0,
                 chunk.offset, chunk.len, chunk.len * local_q_dim,
                 physical_len * local_q_dim * DWIDTH, tp},
                /*flags=*/0, invocation);

            const int64_t attention_site = raw_spm
                ? (image_batch_count_ > 1
                       ? HYVIT2_PACKED_RAW_SPM_ATTN_SITE
                       : HYVIT2_SINGLE_RAW_SPM_ATTN_SITE)
                : (image_batch_count_ > 1
                       ? HYVIT2_PACKED_ATTN_SITE
                       : HYVIT2_SINGLE_ATTN_SITE);
            append(
                FmbRouteFamily::ATTENTION, attention_site,
                static_cast<int64_t>(
                    raw_spm ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                            : AttentionExecutionPolicy::DDR_KV),
                {image_batch_count_, chunk.len, physical_len, nq, hd,
                 orig_head_dim_, tp},
                raw_spm || raw_spm_eligible
                    ? 0 : HYVIT2_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                invocation);
            if (image_batch_count_ > 1) {
                append(
                    FmbRouteFamily::GRAPH_SCHEDULE,
                    HYVIT2_MASK_SCHEDULE_SITE,
                    static_cast<int64_t>(
                        cold_config_.mask_once
                            ? HyViT2GraphScheduleRoute::MASK_FIRST_LAYER_ONLY
                            : HyViT2GraphScheduleRoute::MASK_EACH_LAYER),
                    {cold_config_.mask_once ? 1 : 0, image_batch_count_,
                     chunk.len, physical_len, num_layers(), tp},
                    /*flags=*/0, invocation);
            }
            append_linear(
                HYVIT2_O_LINEAR_SITE,
                {chunk.len, h, nq * hd, 0, tp}, invocation);
            append(
                FmbRouteFamily::ALL_REDUCE,
                HYVIT2_ATTN_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(chunk.len, h),
                {chunk.len, h, tp, NUM_CORES},
                /*flags=*/0, invocation);
            append_linear(
                HYVIT2_FC1_LINEAR_SITE,
                {chunk.len, inter, h, 1, tp}, invocation);
            append_linear(
                HYVIT2_FC2_LINEAR_SITE,
                {chunk.len, h, inter, 0, tp}, invocation);
            append(
                FmbRouteFamily::ALL_REDUCE,
                HYVIT2_MLP_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(chunk.len, h),
                {chunk.len, h, tp, NUM_CORES},
                /*flags=*/0, invocation);

            const auto append_merger_permute = [&](int64_t site_id) {
                append(
                    FmbRouteFamily::GRAPH_SCHEDULE, site_id,
                    static_cast<int64_t>(
                        merger_member_major_
                            ? HyViT2GraphScheduleRoute::MERGER_MEMBER_MAJOR_LAYOUT
                            : HyViT2GraphScheduleRoute::PATCH_MAJOR_LAYOUT),
                    {merger_member_major_ ? 1 : 0,
                     proj1_in_merger_ ? 1 : 0, p1m ? 1 : 0,
                     image_batch_count_, chunk.offset, chunk.len,
                     physical_len, h},
                    /*flags=*/0, invocation);
            };
            append_merger_permute(HYVIT2_MERGER_PERMUTE1_SITE);
            append_merger_permute(HYVIT2_MERGER_PERMUTE2_SITE);
            if (p1m) {
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    HYVIT2_TAIL_OUTPUT_DMA_SITE,
                    static_cast<int64_t>(HyViT2DmaRoute::SPM_TO_DDR_MUTABLE),
                    {merger_member_major_ ? 1 : 0,
                     proj1_in_merger_ ? 1 : 0, p1m ? 1 : 0,
                     chunk.offset, chunk.len, h, h},
                    /*flags=*/0, invocation);
            } else {
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    HYVIT2_TAIL_SLOT_DMA_SITE,
                    static_cast<int64_t>(HyViT2DmaRoute::SPM_TO_DDR_FIXED),
                    {merger_member_major_ ? 1 : 0,
                     proj1_in_merger_ ? 1 : 0, p1m ? 1 : 0,
                     chunk.offset, chunk.len, h},
                    /*flags=*/0, invocation);
                append_linear(
                    HYVIT2_PROJECTOR_LINEAR_SITE,
                    {proj1_in_merger_ ? 1 : 0,
                     merger_member_major_ ? 1 : 0, p1m ? 1 : 0,
                     chunk.len, projection_dim_, h, 1, 1},
                    invocation);
            }
        }

        if (planning_external_patch_prologue_) {
            TORCH_CHECK(
                has_patch_emb_ &&
                    static_cast<int64_t>(plan.spans.size()) ==
                        image_batch_count_,
                "RPU_PLANNER_REJECT:CAPABILITY: HYViT2 patch prologue "
                "requires initialized patch weights and one span per image");
            const int64_t K = pe_kh_ * pe_kw_ * pe_cin_padded_;
            for (int64_t image = 0; image < image_batch_count_; ++image) {
                const FmbExecutionSpan& span = plan.spans[image];
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    HYVIT2_PATCH_INPUT_DMA_SITE,
                    static_cast<int64_t>(HyViT2PatchMutableDmaRoute::
                        DDR_BROADCAST_TO_SPM_MUTABLE),
                    {span.len, pe_cin_orig_, pe_cin_padded_, pe_kh_, pe_kw_,
                     pe_strideh_, pe_stridew_, patch_embed_cores_},
                    /*flags=*/0, /*invocation=*/image);
                append(
                    FmbRouteFamily::LINEAR,
                    HYVIT2_PATCH_LINEAR_SITE,
                    static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                    {span.len, pe_cout_, K, 1, patch_embed_cores_, 0},
                    /*flags=*/0, /*invocation=*/image);
                if (patch_embed_cores_ > 1) {
                    const RpuAllGatherSchedule schedule =
                        rpu_resolve_all_gather_schedule(
                            pe_cout_ / patch_embed_cores_, DWIDTH);
                    append(
                        FmbRouteFamily::COLLECTIVE,
                        HYVIT2_PATCH_ALL_GATHER_SITE,
                        static_cast<int64_t>(schedule),
                        {span.len, pe_cout_ / patch_embed_cores_, DWIDTH,
                         patch_embed_cores_},
                        /*flags=*/0, /*invocation=*/image);
                }
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    HYVIT2_PATCH_POSITION_DMA_SITE,
                    static_cast<int64_t>(HyViT2PatchMutableDmaRoute::
                        DDR_BROADCAST_TO_SPM_MUTABLE),
                    {span.len, pe_cout_, 1},
                    /*flags=*/0, /*invocation=*/image);
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    HYVIT2_PATCH_OUTPUT_DMA_SITE,
                    static_cast<int64_t>(HyViT2PatchMutableDmaRoute::
                        SPM_COPY_TO_DDR_MUTABLE),
                    {span.offset * pe_cout_ * DWIDTH,
                     span.len * pe_cout_},
                    /*flags=*/0, /*invocation=*/image);
            }
        }
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA);
        return manifest;
    }

    std::vector<FmbPhysicalExecutionManifest>
    physical_manifest_domain_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
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
        const FmbPhysicalExecutionManifest& /*manifest*/) const override {
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
            image_batch_count_ > 8 ||
            static_cast<int64_t>(plan.input.chunks.size()) !=
                image_batch_count_ ||
            static_cast<int64_t>(plan.spans.size()) != image_batch_count_) {
            return false;
        }
        const ChunkInfo& chunk = plan.compute.chunks.front();
        const int64_t seq = chunk.len;
        if (seq <= 0 || seq != seq_len_ ||
            seq != image_batch_count_ * 196 || chunk.offset != 0 ||
            chunk.kv_seq_len != seq ||
            plan.qkv.chunks.front().offset != 0 ||
            plan.qkv.chunks.front().len != seq) {
            return false;
        }
        for (int64_t image = 0; image < image_batch_count_; ++image) {
            const int64_t offset = image * 196;
            if (plan.input.chunks[image].offset != offset ||
                plan.input.chunks[image].len != 196 ||
                plan.spans[image].offset != offset ||
                plan.spans[image].len != 196) {
                return false;
            }
        }
        const int mask_type = image_batch_count_ > 1 ? 4 : 0;
        return num_layers() == 27 && layer_weights_.size() == 27 &&
            hidden_size() == 1152 && intermediate_size() == 4352 &&
            num_q_heads() == 16 && num_kv_heads() == 16 &&
            head_dim() == 80 && orig_head_dim_ == 72 &&
            sdpa_by_mha_spm_is_valid(
                /*batch=*/1, seq, seq, num_q_heads(), num_kv_heads(),
                head_dim(), NUM_CORES, mask_type);
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

        int64_t local_q_dim = (nq / NUM_CORES) * hd;
        int64_t local_inter = is_ / NUM_CORES;
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
        const bool raw_spm = ctx.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;

        // LayerWide: read in both phases → sized at wide_cs.
        int64_t res  = A(wide_cs * h * DWIDTH);
        // KvInsert: Phase-1 QKV outputs (q renamed → q_kv; Phase 1 writes q_kv,
        // dumps it to q_ddr_buf_, never reads it again — Phase 2 reloads from
        // DDR into q_comp).
        int64_t qkv  = A(kv_cs   * local_q_dim * DWIDTH);
        const int64_t raw_kv_rows = raw_spm
            ? Align(kv_cs, int64_t{16}) : kv_cs;
        const int64_t raw_kv = A(raw_kv_rows * local_q_dim * DWIDTH);
        // Compute: Phase-2 working buffers → sized at comp_cs.
        int64_t q_comp = A(comp_cs * local_q_dim * DWIDTH);
        int64_t sdpa_out = A(comp_cs * local_q_dim * DWIDTH);
        int64_t fc1  = A(comp_cs * local_inter * DWIDTH);

        // SDPA tmp sizing (same formula as before, now sized at comp_cs — the
        // Phase-2 query-chunk length, NOT the full sequence).
        SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                            hd, /*nq*/nq, /*nkv*/nq,
                            /*cores*/NUM_CORES,
                            /*mask*/image_batch_count_ > 1 ? 4 : 0};
        SdpaTiling t = sdpa_compute_tiling(sdpa_cfg, comp_cs);
        int64_t nkv_per_core = CeilDiv(nq, (int64_t)NUM_CORES);
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

        // Q_INPLACE（见文件头）：只在**单 chunk**（两相同一个 chunk 且覆盖整段
        // 序列）下成立 —— 否则 Phase 1/Phase 2 之间隔着别的 chunk，q 必须过 DDR。
        q_inplace_ = cold_config_.q_inplace
                     && comp_cs == kv_cs && comp_cs >= seq_len_ && seq_len_ > 0;

        // KvInsert-only (Phase 1) — aliasable against Compute-only buffers.
        // Q_INPLACE 下 q_kv 提到 LayerWide 且活到 SDPA 消费完（相 1..6）。
        decls.push_back({"q_kv",       qkv, 1, q_inplace_ ? 6 : 3, StorageClass::Temp, 0,
                         nullptr, q_inplace_ ? ALL : KVIN});
        // KVPAD16（见文件头）：k/v 槽补到 16 的倍数行，KV-insert 一次 v16 打完。
        kv_pad_rows_ = cold_config_.kvpad16 ? Align(kv_cs, (int64_t)16) : 0;
        const int64_t kv_sz = raw_spm
            ? raw_kv
            : (kv_pad_rows_ > 0
                   ? std::max(qkv, A(kv_pad_rows_ * local_q_dim * DWIDTH))
                   : qkv);
        decls.push_back({"k", kv_sz, 1, raw_spm ? 5 : 3,
                         StorageClass::Temp, 0, nullptr,
                         raw_spm ? ALL : KVIN});
        decls.push_back({"v", kv_sz, 1, raw_spm ? 5 : 3,
                         StorageClass::Temp, 0, nullptr,
                         raw_spm ? ALL : KVIN});

        // Compute-only (Phase 2) — lifecycle aliasing across attention→MLP.
        // Q_INPLACE 下 q_comp 就是 q_kv 本身（alias_of，不占新字节）。
        if (q_inplace_)
            decls.push_back({"q_comp", 0, 0, 0, StorageClass::Temp, 0, "q_kv", ALL});
        else
            decls.push_back({"q_comp", q_comp, 4, 5, StorageClass::Temp, 0, nullptr, COMP});
        decls.push_back({"sdpa_out",   sdpa_out, 5, 6, StorageClass::Temp, 0, nullptr, COMP});
        decls.push_back({"sdpa_tmp",   sdpa_tmp, 5, 5, StorageClass::Temp, 0, nullptr, COMP});
        // packed 路径的显式 2D mask 槽（相位窗口贴着 SDPA 的消费窗，不与
        // sdpa_tmp 混叠）。尺寸公式同基类 rpu_qwen3_model.h:1177。
        // MASK_ONCE 打开时改成 LayerWide + 撑满相位（见文件头），这样它才既不被
        // Phase-2 的 fc1 压、也不被下一层 Phase-1 的 q_kv/k/v 压。
        if (image_batch_count_ > 1) {
            const bool once = cold_config_.mask_once;
            decls.push_back({"sdpa_mask",
                             A(comp_cs * CeilDiv(seq_len_, (int64_t)16) * 32),
                             once ? 1 : 4, once ? 8 : 5, StorageClass::Temp, 0, nullptr,
                             once ? ALL : COMP});
        }
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
        int64_t local_q_dim = (num_q_heads() / NUM_CORES) * head_dim();
        int64_t local_inter = intermediate_size() / NUM_CORES;

        // Loop 1: 8-core DMAs for all layers (memset + norm + col-partition bias)
        // Matches v2 ordering at rpu_hyvit2_fused_encoder_layer.cpp:316-333.
        for (int64_t L = 0; L < num_layers(); ++L) {
            auto& bn = layer_bias_norm_[L];

            // Zero row-partition biases (these accumulate partial sums from 8 cores)
            rpu_launch_memset_spm_multicore(layer_addr(L, 0, "o_bias"), h);
            rpu_launch_memset_spm_multicore(layer_addr(L, 0, "fc2_bias"), h);

            // Broadcast LayerNorm gamma/beta to all cores (num_elements, NOT bytes)
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

            // Col-partition scatter: each core gets its own slice of the bias.
            // DMA path via rpu_launch_ddr_scatter_spm_dma — weights are static
            // (model-construction-time alloc), so non-mutable is safe.
            rpu_launch_ddr_scatter_spm_dma(
                bn.q_b.data_ptr<c10::Half>(),
                /*elements_per_core=*/local_q_dim,
                /*core_stride_bytes=*/local_q_dim * DWIDTH,
                layer_addr(L, 0, "q_bias"),
                /*num_cores=*/NUM_CORES);
            rpu_launch_ddr_scatter_spm_dma(
                bn.k_b.data_ptr<c10::Half>(),
                local_q_dim, local_q_dim * DWIDTH,
                layer_addr(L, 0, "k_bias"),
                /*num_cores=*/NUM_CORES);
            rpu_launch_ddr_scatter_spm_dma(
                bn.v_b.data_ptr<c10::Half>(),
                local_q_dim, local_q_dim * DWIDTH,
                layer_addr(L, 0, "v_bias"),
                /*num_cores=*/NUM_CORES);
            rpu_launch_ddr_scatter_spm_dma(
                bn.fc1_b.data_ptr<c10::Half>(),
                local_inter, local_inter * DWIDTH,
                layer_addr(L, 0, "fc1_bias"),
                /*num_cores=*/NUM_CORES);
        }

        // Loop 2: 1-core DMAs for all layers (row-partition biases).
        // Matches v2 ordering at rpu_hyvit2_fused_encoder_layer.cpp:334-338.
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

        // HYViT2: no post-LayerNorm — nothing global to preload here.
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
        TORCH_CHECK(
            ctx().has_complete_physical_manifest(),
            "HYViT2VisionModel requires a COMPLETE physical descriptor");
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
            seq_len, h, eps_, false, 0, NUM_CORES);

        // QKV Linear with bias (SPM-to-SPM ACC16, col-partition). q → q_kv.
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVIT2_GRAPH_SCHEDULE_SITE,
                static_cast<int64_t>(
                    cold_config_.fast_replay
                        ? HyViT2GraphScheduleRoute::FAST_REPLAY_SKIP_LAYER_LOOP
                        : HyViT2GraphScheduleRoute::REEMIT_LAYER_LOOP),
                /*resolved_flags=*/0,
                {cold_config_.fast_replay ? 1 : 0,
                 cold_config_.fast_replay_preload ? 1 : 0,
                 num_layers()});
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVIT2_KV_Q_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, nq * hd, h, 1, NUM_CORES}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q_kv"),
            seq_len, nq * hd, h, 1, NUM_CORES,
            layer_addr(layer_idx, 0, "q_bias"), lw.q_ws);
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVIT2_KV_K_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, nq * hd, h, 1, NUM_CORES}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, NUM_CORES,
            layer_addr(layer_idx, 0, "k_bias"), lw.k_ws);
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVIT2_KV_V_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, nq * hd, h, 1, NUM_CORES}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, NUM_CORES,
            layer_addr(layer_idx, 0, "v_bias"), lw.v_ws);

        // No RoPE (SigLIP has none).

        // KV cache insert at absolute position chunk.offset.
        // Pitfall 3 structural fix: takes SPM offsets via addr_offset(name).value.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const int64_t kv_physical_rows = cold_config_.kvpad16 && kv_pos % 16 == 0
            ? Align(seq_len, int64_t{16}) : seq_len;
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, HYVIT2_KV_PADDING_SITE,
            static_cast<int64_t>(
                kv_physical_rows > seq_len
                    ? HyViT2KvPaddingRoute::PAD16_ROWS
                    : HyViT2KvPaddingRoute::LOGICAL_ROWS),
            /*resolved_flags=*/0,
            {cold_config_.kvpad16 ? 1 : 0, seq_len,
             kv_pad_rows_, kv_physical_rows, NUM_CORES, nq, hd},
            chunk.idx);
        const FmbRouteManifestEntry& route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, HYVIT2_KV_SITE, chunk.idx);
        const KvInsertSegmentPlan kv_plan =
            restore_kvinsert_plan(
                HYVIT2_KV_SITE, route.arguments, NUM_CORES, nq, hd);
        TORCH_CHECK(
            kv_plan.logical_rows() == seq_len &&
                kv_plan.physical_rows() == kv_physical_rows &&
                kv_plan.segment(0).position == kv_pos,
            "HYViT2VisionModel KV descriptor geometry drift at invocation ",
            chunk.idx);
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, HYVIT2_KV_SITE,
            static_cast<int64_t>(kv_plan.route()),
            HYVIT2_KV_FLAG_DDR_MIRROR,
            route.arguments, chunk.idx);
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache,
            addr_offset("k").value, addr_offset("v").value,
            nq, hd, NUM_CORES,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            kv_pad_rows_, kv_plan);

        // Save Q to the per-seq Q staging slot via spm_scatter_ddr_dma
        // (position-indexed by chunk.offset). The slot (keyed by seq_len_) is
        // created once in forward_packed and NEVER reallocated → stable data_ptr,
        // safe for non-mutable DMA across REPLAY (matching GemmaModel:
        // stable storage). Phase 2 reloads it into "q_comp".
        // Q_INPLACE: 单 chunk 下 Phase 2 紧跟其后、直接读 q_kv ⇒ 整趟往返不发。
        const int64_t q_local_elems = seq_len * local_q_heads_ * hd;
        const int64_t q_core_stride_bytes =
            seq_len_ * local_q_heads_ * hd * DWIDTH;
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, HYVIT2_Q_SAVE_DMA_SITE,
                static_cast<int64_t>(
                    q_inplace_ ? HyViT2DmaRoute::KEEP_Q_IN_SPM
                               : HyViT2DmaRoute::SPM_SCATTER_TO_DDR_FIXED),
                /*resolved_flags=*/0,
                {cold_config_.q_inplace ? 1 : 0, q_inplace_ ? 1 : 0,
                 chunk.offset, seq_len, q_local_elems,
                 q_core_stride_bytes, NUM_CORES}, chunk.idx);
        }
        if (q_inplace_) {
            TORCH_CHECK(chunk.len == seq_len_ && chunk.offset == 0,
                        "HYViT2VisionModel: Q_INPLACE 只在单 chunk 下成立 "
                        "(chunk.offset=", chunk.offset, " len=", chunk.len,
                        " seq_len_=", seq_len_, ")");
            return;
        }
        TORCH_CHECK(q_ddr_slots_.count({seq_len_, local_q_heads_ * head_dim()}) == 1,
                    "HYViT2VisionModel: Q staging slot not allocated in KV_FIRST mode");
        const at::Tensor& q_slot = q_ddr_slot();
        c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
        int64_t q_row_stride = local_q_heads_ * hd;
        int64_t q_elem_offset = chunk.offset * q_row_stride;
        TORCH_CHECK(
            q_core_stride_bytes == q_slot.size(1) * q_row_stride * DWIDTH,
            "HYViT2VisionModel Q staging descriptor/core stride drift");
        rpu_launch_spm_scatter_ddr_dma(
            addr(0, "q_kv"), q_ddr_base + q_elem_offset,
            q_local_elems, q_core_stride_bytes,
            /*num_cores=*/NUM_CORES);

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
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;        // this query-chunk length
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();
        int64_t local_inter = is_ / NUM_CORES;

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
        // Q_INPLACE: q 还留在 q_kv 里，而 "q_comp" 已 alias 到它 ⇒ 不必读回。
        const int64_t q_local_elems = seq_len * local_q_heads_ * hd;
        const int64_t q_core_stride_bytes =
            seq_len_ * local_q_heads_ * hd * DWIDTH;
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, HYVIT2_Q_LOAD_DMA_SITE,
                static_cast<int64_t>(
                    q_inplace_ ? HyViT2DmaRoute::KEEP_Q_IN_SPM
                               : HyViT2DmaRoute::DDR_SCATTER_TO_SPM_FIXED),
                /*resolved_flags=*/0,
                {cold_config_.q_inplace ? 1 : 0, q_inplace_ ? 1 : 0,
                 chunk.offset, seq_len, q_local_elems,
                 q_core_stride_bytes, NUM_CORES}, chunk.idx);
        }
        if (!q_inplace_) {
            const at::Tensor& q_slot = q_ddr_slot();
            c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
            int64_t q_row_stride = local_q_heads_ * hd;
            int64_t q_elem_offset = chunk.offset * q_row_stride;
            int64_t q_core_stride = q_slot.size(1) * q_row_stride * DWIDTH;
            TORCH_CHECK(
                q_core_stride == q_core_stride_bytes,
                "HYViT2VisionModel Q reload descriptor/core stride drift");
            rpu_launch_ddr_scatter_spm_dma(
                q_ddr_base + q_elem_offset,
                /*elements_per_core=*/q_local_elems,
                /*core_stride_bytes=*/q_core_stride,
                addr(0, "q_comp"),
                /*num_cores=*/NUM_CORES);
        } else {
            TORCH_CHECK(chunk.len == seq_len_ && chunk.offset == 0,
                        "HYViT2VisionModel: Q_INPLACE 只在单 chunk 下成立");
        }

        // SDPA over the FULL bidirectional KV cache (kv_seq_len = seq_len_),
        // query = this chunk (seq_q = chunk.len). Source Q from "q_comp".
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        double attn_scale = 1.0 / std::sqrt(static_cast<double>(orig_head_dim_));

        // Minibatch SDPA (image_batch_count_>1) asserts a single chunk
        // (seq_q == full packed seq); plan_kv_first_chunks (Site8) forces a
        // single chunk for N>1 so this guard is belt-and-suspenders.
        TORCH_CHECK(image_batch_count_ == 1 || chunk.len == seq_len_,
                    "HYViT2VisionModel: minibatch (N=", image_batch_count_,
                    ") SDPA requires a single chunk (chunk.len=", chunk.len,
                    " == seq_len_=", seq_len_, "); plan_kv_first_chunks must "
                    "force ChunkPlan{seq_len_,1} for N>1.");

        if (image_batch_count_ > 1) {
            // SigLIP-batch: N images packed into seq_len_. Per-image attention
            // isolation via the minibatch kernel — image i attends ONLY its own
            // per_image_ctx tokens. K/V were inserted contiguously at position 0
            // in Phase 1, so the kernel's image_idx*per_image_sKeyVx offset lines
            // up. Single-chunk only (guarded above).
            // **不用 minibatch kernel** —— 它要求 `per_image_ctx % tile_m == 0`，
            // 而 tile_m 恒为 16 的倍数，Hy-VLA 的 per_image_ctx=196 无解
            // （SigLIP 的 256 才碰巧满足）。改用普通 unified SDPA + 显式 2D
            // 块对角 mask：等价的逐图隔离，代价是算满 seq_len_² 的 score 矩阵。
            const PreparedMask& pm =
                packed_block_mask(seq_len_, image_batch_count_);
            const uint32_t mask_off = addr_offset("sdpa_mask").value;
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::GRAPH_SCHEDULE,
                    HYVIT2_MASK_SCHEDULE_SITE,
                    static_cast<int64_t>(
                        cold_config_.mask_once
                            ? HyViT2GraphScheduleRoute::MASK_FIRST_LAYER_ONLY
                            : HyViT2GraphScheduleRoute::MASK_EACH_LAYER),
                    /*resolved_flags=*/0,
                    {cold_config_.mask_once ? 1 : 0, image_batch_count_,
                     seq_len, seq_len_, num_layers(), NUM_CORES},
                    chunk.idx);
            }
            // 块对角 mask 逐层恒定（且此路径已断言单 chunk）⇒ 槽撑满相位后
            // 只需在层 0 灌一次（见文件头 MASK_ONCE）。
            if (!cold_config_.mask_once || layer_idx == 0)
                sdpa_dma_mask_to_spm(pm, mask_off, seq_len_, seq_len_, NUM_CORES);
            if (ctx().attention_policy ==
                AttentionExecutionPolicy::SPM_KV_BY_MHA) {
                TORCH_CHECK(
                    chunk.offset == 0 && chunk.len == seq_len_ &&
                        pm.mask_type == 4,
                    "HYViT2 raw-SPM packed attention requires one full "
                    "packed chunk and MASK_2D");
                rpu_launch_v_transpose_spm(
                    addr(0, "v"), addr(0, "sdpa_tmp"),
                    /*batch=*/1, seq_len_, nq, hd, NUM_CORES);
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION,
                    HYVIT2_PACKED_RAW_SPM_ATTN_SITE,
                    static_cast<int64_t>(
                        AttentionExecutionPolicy::SPM_KV_BY_MHA),
                    /*resolved_flags=*/0,
                    {image_batch_count_, seq_len, seq_len_, nq, hd,
                     orig_head_dim_, NUM_CORES}, chunk.idx);
                rpu_launch_sdpa_by_mha_spm(
                    addr(0, "q_comp"), addr(0, "k"),
                    addr(0, "sdpa_tmp"), addr(0, "sdpa_out"),
                    addr(0, "sdpa_mask"), pm.mask_type, attn_scale,
                    /*batch=*/1, seq_len, seq_len_, nq, nq, hd,
                    NUM_CORES);
            } else {
                if (ctx().has_complete_physical_manifest()) {
                    LayoutContext spm_layout;
                    spm_layout.attention_policy =
                        AttentionExecutionPolicy::SPM_KV_BY_MHA;
                    spm_layout.batch_size = ctx().batch_size;
                    spm_layout.is_causal = ctx().is_causal;
                    spm_layout.use_attn_mask =
                        ctx().attention_mask.has_value();
                    const bool raw_profile =
                        subclass_spm_kv_by_mha_eligible(
                            ctx().stage_plan, spm_layout, ctx().position);
                    ctx().consume_physical_route(
                        FmbRouteFamily::ATTENTION, HYVIT2_PACKED_ATTN_SITE,
                        static_cast<int64_t>(
                            AttentionExecutionPolicy::DDR_KV),
                        raw_profile
                            ? 0
                            : HYVIT2_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                        {image_batch_count_, seq_len, seq_len_, nq, hd,
                         orig_head_dim_, NUM_CORES}, chunk.idx);
                }
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    k_cache, v_cache,
                    pm.mask_type, attn_scale,
                    addr_offset("q_comp").value,
                    addr_offset("sdpa_out").value,
                    addr_offset("sdpa_tmp").value, mask_off,
                    /*seq_q=*/seq_len, nq, nq, hd,
                    /*kv_seq_len=*/seq_len_, NUM_CORES, NUM_CORES);
            }
        } else {
            if (ctx().attention_policy ==
                AttentionExecutionPolicy::SPM_KV_BY_MHA) {
                TORCH_CHECK(chunk.offset == 0 && chunk.len == seq_len_,
                            "HYViT2 raw-SPM attention requires one full chunk");
                rpu_launch_v_transpose_spm(
                    addr(0, "v"), addr(0, "sdpa_tmp"),
                    /*batch=*/1, seq_len_, nq, hd, NUM_CORES);
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION,
                    HYVIT2_SINGLE_RAW_SPM_ATTN_SITE,
                    static_cast<int64_t>(
                        AttentionExecutionPolicy::SPM_KV_BY_MHA),
                    /*resolved_flags=*/0,
                    {image_batch_count_, seq_len, seq_len_, nq, hd,
                     orig_head_dim_, NUM_CORES}, chunk.idx);
                rpu_launch_sdpa_by_mha_spm(
                    addr(0, "q_comp"), addr(0, "k"),
                    addr(0, "sdpa_tmp"), addr(0, "sdpa_out"),
                    /*mask_spm=*/0, /*MASK_NONE=*/0, attn_scale,
                    /*batch=*/1, seq_len, seq_len_, nq, nq, hd,
                    NUM_CORES);
            } else {
                if (ctx().has_complete_physical_manifest()) {
                    LayoutContext spm_layout;
                    spm_layout.attention_policy =
                        AttentionExecutionPolicy::SPM_KV_BY_MHA;
                    spm_layout.batch_size = ctx().batch_size;
                    spm_layout.is_causal = ctx().is_causal;
                    spm_layout.use_attn_mask =
                        ctx().attention_mask.has_value();
                    const bool raw_profile =
                        subclass_spm_kv_by_mha_eligible(
                            ctx().stage_plan, spm_layout, ctx().position);
                    ctx().consume_physical_route(
                        FmbRouteFamily::ATTENTION, HYVIT2_SINGLE_ATTN_SITE,
                        static_cast<int64_t>(
                            AttentionExecutionPolicy::DDR_KV),
                        raw_profile
                            ? 0
                            : HYVIT2_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                        {image_batch_count_, seq_len, seq_len_, nq, hd,
                         orig_head_dim_, NUM_CORES}, chunk.idx);
                }
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    k_cache, v_cache,
                    0 /*MASK_NONE*/, attn_scale,
                    addr_offset("q_comp").value,
                    addr_offset("sdpa_out").value,
                    addr_offset("sdpa_tmp").value, 0,
                    /*seq_q=*/seq_len, nq, nq, hd,
                    /*kv_seq_len=*/seq_len_, NUM_CORES, NUM_CORES);
            }
        }

        // Phase 4: O_proj (row-partition, with bias) + AllReduce + Residual
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVIT2_O_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, h, nq * hd, 0, NUM_CORES}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, NUM_CORES,
            layer_addr(layer_idx, 0, "o_bias"), lw.o_ws);
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                HYVIT2_ATTN_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(seq_len, h),
                /*resolved_flags=*/0,
                {seq_len, h, NUM_CORES, NUM_CORES}, chunk.idx);
        }
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
            seq_len, h, NUM_CORES, NUM_CORES);

        // Phase 5: LayerNorm2 + fc1+bias + GELU
        rpu_launch_layernorm_spm_kernel(
            addr(0, "input_norm"), addr(0, "oproj"),
            layer_addr(layer_idx, 0, "ln2_gamma"), layer_addr(layer_idx, 0, "ln2_beta"),
            seq_len, h, eps_, false, 0, NUM_CORES);
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVIT2_FC1_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, is_, h, 1, NUM_CORES}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "oproj"), lw.fc1_w, addr(0, "fc1"),
            seq_len, is_, h, 1, NUM_CORES,
            layer_addr(layer_idx, 0, "fc1_bias"), lw.fc1_ws);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "fc1"), addr(0, "fc1"),
            seq_len * local_inter, ValuOpType::ADD, /*is_gelu=*/true, NUM_CORES);

        // Phase 6: fc2 (row-partition, with bias) + AllReduce + Residual
        // Round 9: final output goes to "residual1" instead of "oproj" so the
        // next layer (which reads from "residual1") gets it directly via SPM
        // without needing DDR ping-pong.
        // Phase 6 uses "oproj" as temporary storage for fc2 partial results,
        // then all_reduce(oproj + input_norm) lands back in "residual1".
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVIT2_FC2_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, h, is_, 0, NUM_CORES}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "fc1"), lw.fc2_w, addr(0, "oproj"),
            seq_len, h, is_, 0, NUM_CORES,
            layer_addr(layer_idx, 0, "fc2_bias"), lw.fc2_ws);
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                HYVIT2_MLP_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(seq_len, h),
                /*resolved_flags=*/0,
                {seq_len, h, NUM_CORES, NUM_CORES}, chunk.idx);
        }
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"),
            seq_len, h, NUM_CORES, NUM_CORES);

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
            // Last layer (D-501): Projector ONLY.
            // DIVERGENCE FROM SigLIP: HYViT2 has NO post-LayerNorm on the action
            // path — `_HYViT2VisionTransformer.forward_head` is dead code because
            // `cal_attn_pool=False`, and the encoder output feeds the tower-external
            // `merger` directly. Passing an identity LayerNorm is NOT an option:
            // LayerNorm(w=1,b=0) still applies (x-mean)/std
            // and is per-row nonlinear, so no weight choice
            // can cancel it. Phase-6 output stays in "residual1" and is staged to
            // DDR as-is.

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
            // merger 的 member-major 重排（项 A，`RPU_HY_VLA_FUSED_MERGER`）。
            // tower 外 merger 把每张图的 14×14 patch 按 2×2 分成 7×7 组，组轴上做
            // pooled / softmax / 加权求和。**只有 member-major `[4, G, D]`（G = B·49）
            // 才能让这三件事退化成扁平算子**（同一成员的全部组在内存里连成一片），
            // 否则要么用跨步 kernel（没有），要么 588 次逐行 DMA。
            //
            // 源行下标 row = (((cam*7+r7)*2 + a)*7 + c7)*2 + b，
            // 轴序 X=(cam,r7)[B·gr] · a[2] · c7[gr] · b[2] · D ⇒ **两次 perm{1,0,2}**：
            //   #1  (X·a·c7, n=b=2, c=D)      -> (b, X, a, c7, D)
            //   #2  (b·X,    n=a=2, c=c7·D)   -> (a, b, X, c7, D) = (m, g, D) ✅
            // 两次都是 transpose_nbc（`rpu_permute.cpp:700` 的 "102" 分支），
            // c 都是 16 的倍数且 < 65536（reg 是 u16）。
            //
            // 放在 proj1 **之前**：这里是 h=1152 而不是 proj 后的 2048，少搬一半；
            // 且 residual1 本来就在 SPM，不需要任何 DMA。乒乓复用同层已死的
            // input_norm / oproj（两者与 residual1 相位重叠 ⇒ 框架保证不混叠）
            // ⇒ **不新增任何 SPM**。
            // 末端不需要反向置换：加权求和后的 147 组顺序就是 (cam,r7,c7)，
            // 正好是 VLM 期望的顺序。
            uint32_t tail_src = addr(0, "residual1");
            const bool p1m = proj1_in_merger_ && merger_member_major_;
            if (merger_member_major_) {
                TORCH_CHECK(chunk.len == seq_len_,
                            "HYViT2VisionModel: member-major merger requires a single "
                            "chunk spanning the whole packed sequence (chunk.len=",
                            chunk.len, ", seq_len=", seq_len_, ")");
                const int64_t B = image_batch_count_;
                const int64_t side = (B > 0 && seq_len % B == 0)
                                   ? (int64_t)std::lround(std::sqrt((double)(seq_len / B)))
                                   : 0;
                TORCH_CHECK(side > 0 && side % 2 == 0 && B * side * side == seq_len,
                            "HYViT2VisionModel: member-major merger expects seq_len = "
                            "B*(2g)^2, got seq_len=", seq_len, " B=", B);
                const int64_t gr = side / 2;
                TORCH_CHECK(gr * h < 65536,
                            "HYViT2VisionModel: permute3d c-reg is u16, gr*h=", gr * h);
                if (ctx().has_complete_physical_manifest()) {
                    ctx().consume_physical_route(
                        FmbRouteFamily::GRAPH_SCHEDULE,
                        HYVIT2_MERGER_PERMUTE1_SITE,
                        static_cast<int64_t>(
                            HyViT2GraphScheduleRoute::MERGER_MEMBER_MAJOR_LAYOUT),
                        /*resolved_flags=*/0,
                        {merger_member_major_ ? 1 : 0,
                         proj1_in_merger_ ? 1 : 0, p1m ? 1 : 0,
                         image_batch_count_, chunk.offset, seq_len,
                         seq_len_, h}, chunk.idx);
                }
                rpu_launch_permute3d_spm_kernel(
                    tail_src, addr(0, "input_norm"),
                    /*b=*/B * gr * 2 * gr, /*n=*/2, /*c=*/h, {1, 0, 2}, /*num_cores=*/1);
                if (ctx().has_complete_physical_manifest()) {
                    ctx().consume_physical_route(
                        FmbRouteFamily::GRAPH_SCHEDULE,
                        HYVIT2_MERGER_PERMUTE2_SITE,
                        static_cast<int64_t>(
                            HyViT2GraphScheduleRoute::MERGER_MEMBER_MAJOR_LAYOUT),
                        /*resolved_flags=*/0,
                        {merger_member_major_ ? 1 : 0,
                         proj1_in_merger_ ? 1 : 0, p1m ? 1 : 0,
                         image_batch_count_, chunk.offset, seq_len,
                         seq_len_, h}, chunk.idx);
                }
                rpu_launch_permute3d_spm_kernel(
                    addr(0, "input_norm"), addr(0, "oproj"),
                    /*b=*/2 * B * gr, /*n=*/2, /*c=*/gr * h, {1, 0, 2}, /*num_cores=*/1);
                tail_src = addr(0, "oproj");
            } else if (ctx().has_complete_physical_manifest()) {
                const std::vector<int64_t> arguments{
                    merger_member_major_ ? 1 : 0,
                    proj1_in_merger_ ? 1 : 0, p1m ? 1 : 0,
                    image_batch_count_, chunk.offset, seq_len,
                    seq_len_, h};
                ctx().consume_physical_route(
                    FmbRouteFamily::GRAPH_SCHEDULE,
                    HYVIT2_MERGER_PERMUTE1_SITE,
                    static_cast<int64_t>(
                        HyViT2GraphScheduleRoute::PATCH_MAJOR_LAYOUT),
                    /*resolved_flags=*/0, arguments, chunk.idx);
                ctx().consume_physical_route(
                    FmbRouteFamily::GRAPH_SCHEDULE,
                    HYVIT2_MERGER_PERMUTE2_SITE,
                    static_cast<int64_t>(
                        HyViT2GraphScheduleRoute::PATCH_MAJOR_LAYOUT),
                    /*resolved_flags=*/0, arguments, chunk.idx);
            }
            if (!p1m) {
                if (ctx().has_complete_physical_manifest()) {
                    ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        HYVIT2_TAIL_SLOT_DMA_SITE,
                        static_cast<int64_t>(
                            HyViT2DmaRoute::SPM_TO_DDR_FIXED),
                        /*resolved_flags=*/0,
                        {merger_member_major_ ? 1 : 0,
                         proj1_in_merger_ ? 1 : 0, p1m ? 1 : 0,
                         chunk.offset, seq_len, h}, chunk.idx);
                }
                rpu_launch_spm_copy_ddr_dma(
                    tail_src,
                    slot.data_ptr<c10::Half>(),
                    seq_len * h);
            }

            // forward_packed prepared the full output before entering this
            // fast-skippable layer body. Every chunk writes its own row range.
            // proj1 进 merger 图之后，本塔的输出维就是 hidden（1152），不是 2048。
            const int64_t out_dim = p1m ? h : projection_dim_;
            TORCH_CHECK(
                projector_output_.defined() && projector_output_.dim() == 3 &&
                    projector_output_.size(0) == 1 &&
                    projector_output_.size(1) == seq_len_ &&
                    projector_output_.size(2) == out_dim,
                "HYViT2VisionModel: projector output was not prepared for "
                "the admitted forward shape");

            // Projector Linear (DDR col-partition) writes THIS chunk's rows.
            // For multi-chunk, target a [1, chunk.len, proj] view at the chunk's
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
            if (p1m) {
                // The merger consumes this fresh output directly. Update its
                // destination on replay instead of retaining the BUILD address.
                if (ctx().has_complete_physical_manifest()) {
                    ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        HYVIT2_TAIL_OUTPUT_DMA_SITE,
                        static_cast<int64_t>(
                            HyViT2DmaRoute::SPM_TO_DDR_MUTABLE),
                        /*resolved_flags=*/0,
                        {merger_member_major_ ? 1 : 0,
                         proj1_in_merger_ ? 1 : 0, p1m ? 1 : 0,
                         chunk.offset, seq_len, h, h}, chunk.idx);
                }
                rpu_launch_spm_copy_ddr_dma_mutable(
                    tail_src, &projector_out_live_base_,
                    /*dst_offset_bytes=*/chunk.offset * out_dim * DWIDTH,
                    chunk.len * h);
            } else {
                if (ctx().has_complete_physical_manifest()) {
                    ctx().consume_physical_route(
                        FmbRouteFamily::LINEAR,
                        HYVIT2_PROJECTOR_LINEAR_SITE,
                        static_cast<int64_t>(
                            FmbLinearRouteSelector::AUTO_TILE),
                        /*resolved_flags=*/0,
                        {proj1_in_merger_ ? 1 : 0,
                         merger_member_major_ ? 1 : 0, p1m ? 1 : 0,
                         seq_len, projection_dim_, h, 1, 1}, chunk.idx);
                }
                // The shared DDR launcher accepts matrix metadata; retain the
                // output storage alias and its mutable per-chunk DMA address.
                auto proj_out_matrix = proj_out_chunk.view({seq_len, out_dim});
                rpu_launch_linear_ddr_kernel(
                    slot, proj_w_, proj_out_matrix, proj_b_,
                    /*has_bias=*/true, /*partition=*/1,
                    &projector_out_live_base_,
                    /*dst_offset_bytes=*/chunk.offset * out_dim * DWIDTH,
                    &projector_bias_live_base_);
            }
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
    // EXCEPTION (minibatch mutual-exclusion, design note §0/Site8): the per-image
    // minibatch SDPA launcher asserts a single chunk (seq_q == full packed seq).
    // For N>1 packed images we therefore FORCE a single chunk spanning the whole
    // sequence. HALO's current single-image action path is N==1, so this branch
    // is the multi-camera safety net, not the hot path.
    // ========================================================================
    ChunkPlan plan_kv_first_chunks(const ChunkPlan& compute_plan) {
        if (image_batch_count_ > 1) {
            return ChunkPlan{seq_len_, 1};
        }
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
        // packed 走显式 2D mask（见 build_layer_subgraph）⇒ tiling 约束不同，
        // planner 的 cfg 必须跟着走 mask=4，否则与实际发射不一致。
        SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM,
                       head_dim(), num_q_heads(), num_q_heads(),
                       NUM_CORES, image_batch_count_ > 1 ? 4 : 0};
        if (!sdpa_is_valid_chunk_size(cfg, cs, seq_len, position)) return false;
        // Packed images use one block-diagonal attention matrix and therefore
        // require a single compute chunk.  Publish that constraint in the
        // native domain instead of hiding it in a per-forward override.
        if (image_batch_count_ > 1 && cs < seq_len) return false;

        // Board-proven safe ceiling (2026-06-18). cs=256/512 RUN (clean, finite);
        // cs=880 HANGS on the cs-independent 8 MB SPM-block request
        // (`requested=8388608 available=0` infinite retry). Root cause: persistent
        // weights (~569 KB) + the STACKED per-chunk temps (KvInsert + Compute do
        // NOT alias, see below) push the FULL 8 MB SPM
        // block over, even though the bump-pool temps alone fit. The framework's
        // budget basis (SPM_USABLE = full 8 MB) does not subtract the persistent
        // reserve, so the auto-scan over-picks. Cap cs at the proven-working 512
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
        // budget, so the auto-scan picks a cs that fits at real allocation.
        const int64_t h   = hidden_size();
        const int64_t nq  = num_q_heads();
        const int64_t hd  = head_dim();
        const int64_t is_ = intermediate_size();
        const int64_t local_q_dim = (nq / NUM_CORES) * hd;
        const int64_t local_inter = is_ / NUM_CORES;
        auto Aln = [](int64_t b) -> int64_t { return Align(b, 256); };
        const int64_t res = Aln(cs * h * DWIDTH);            // residual1/input_norm/oproj
        const int64_t qkv = Aln(cs * local_q_dim * DWIDTH);  // q_kv/k/v/q_comp/sdpa_out
        const int64_t fc1 = Aln(cs * local_inter * DWIDTH);
        SdpaTiling t = sdpa_compute_tiling(cfg, cs);
        const int64_t nkv_per_core = CeilDiv(nq, (int64_t)NUM_CORES);
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

    // Validate the whole request before any external prologue, SPM allocation,
    // flush or DMA. Tensor input packs its B images; list input packs one image
    // per element. All elements share the SPM geometry allocated for image 0.
    int64_t validate_patch_inputs(
        at::TensorList images, bool allow_batched_tensor) const {
        TORCH_CHECK(has_patch_emb_,
                    "HYViT2 patch embedding requires patch embedding params");
        TORCH_CHECK(!images.empty() && images.size() <= kMaxPackedImages &&
                        (!allow_batched_tensor || images.size() == 1),
                    "HYViT2 patch embedding requires 1..", kMaxPackedImages,
                    " images (one tensor for batched input)");
        int64_t image_count = 0;
        for (const auto& im : images) {
            TORCH_CHECK(im.defined() && im.dim() == 4 &&
                            im.scalar_type() == at::kHalf &&
                            im.device().type() == at::kPrivateUse1 &&
                            im.is_contiguous(),
                        "HYViT2 patch embedding requires contiguous FP16 RPU [B,3,H,W]");
            TORCH_CHECK(im.size(0) > 0 && im.size(0) <= kMaxPackedImages &&
                            (allow_batched_tensor || im.size(0) == 1) &&
                            im.size(1) == pe_cin_orig_,
                        "HYViT2 patch embedding requires RGB and list elements with batch=1");
            TORCH_CHECK(im.size(2) >= pe_kh_ && im.size(3) >= pe_kw_ &&
                            im.size(2) == images.front().size(2) &&
                            im.size(3) == images.front().size(3),
                        "HYViT2 patch embedding images must share H/W and fit a full patch");
            image_count += im.size(0);
        }
        TORCH_CHECK(image_count <= kMaxPackedImages,
                    "HYViT2 patch embedding exceeds mutable input slots");
        const int64_t h = images.front().size(2);
        const int64_t w = images.front().size(3);
        if (patch_embed_cores_ > 1) {
            // This permute uses packed source rows of (w / kw) * kw pixels.
            // A width remainder changes their stride. A height remainder is
            // a valid unused bottom strip, just as in the single-core im2col.
            TORCH_CHECK(pe_strideh_ == pe_kh_ && pe_stridew_ == pe_kw_ &&
                            w % pe_kw_ == 0,
                        "HYViT2 multi-core patch embedding requires stride == kernel "
                        "and width divisible by kernel width");
        }
        TORCH_CHECK(pe_cout_ % patch_embed_cores_ == 0,
                    "HYViT2 patch embedding cout must be divisible by patch cores");
        // Mirror the raw-input DMA limit before touching the allocator.
        constexpr int64_t max_raw_pixels = (8LL << 20) / (3 * DWIDTH);
        TORCH_CHECK(h <= max_raw_pixels / w &&
                        (h * w * 3 * DWIDTH) % 16 == 0,
                    "HYViT2 patch input DMA must be 16-byte aligned and at most 8 MiB");
        const int64_t rows = ((h - pe_kh_) / pe_strideh_ + 1) *
                             ((w - pe_kw_) / pe_stridew_ + 1);
        TORCH_CHECK(rows <= patch_emb_pos_emb_.numel() / pe_cout_,
                    "HYViT2 patch position table is shorter than the image patch grid");
        TORCH_CHECK(pe_cout_ <= (8LL << 20) / DWIDTH &&
                        rows <= (8LL << 20) / (pe_cout_ * DWIDTH) &&
                        (rows * pe_cout_ * DWIDTH) % 16 == 0,
                    "HYViT2 patch output DMA must be 16-byte aligned and at most 8 MiB");
        return image_count * rows;
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
        // Step numbering follows fused_patch_embedding's ①..⑦ (+⑤b all_gather
        // = step 6 here, so pos_emb/output shift to 7/8). Lifetimes are CLOSED
        // intervals — nhwc{3,4} and im2col{4,5} both live at step 4, which is
        // exactly the im2col kernel reading one and writing the other.
        // The peak is unchanged by the multi-core path (nhwc+im2col at step 4
        // dominates the gathered/pos_emb pair at step 8).
        const int nc = patch_embed_cores_;
        using AR = SpmAllocator::AllocRequest;
        std::vector<uint32_t> offsets = SPM_ALLOC.alloc_temporary_aliased({
            AR{pe_cin_orig_ * HW * 2,   1, 2},  // raw
            AR{pe_cin_padded_ * HW * 2, 2, 3},  // padded
            AR{HW * pe_cin_padded_ * 2, 3, 4},  // nhwc
            AR{npp * K * 2,             4, 5},  // im2col
            // gemm_out is held to step 8 unconditionally: when nc==1 there is no
            // all_gather and it IS the result that ⑥⑦ read. Holding the (small)
            // per-core shard that long when nc>1 costs nothing — the peak is set
            // by nhwc+im2col at step 4 either way.
            AR{npp * (cout / nc) * 2,   5, 8},  // gemm_out (per-core shard)
            AR{npp * cout * 2,          6, 8},  // gathered (unused when nc==1)
            AR{npp * cout * 2,          7, 8},  // pos_emb
        });

        auto opts = at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1);
        at::Tensor hidden = at::empty({1, N * npp, cout}, opts);

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
            fused_patch_embedding(
                images[i], patch_emb_weight_, patch_emb_pos_emb_,
                pe_kh_, pe_kw_, pe_cin_orig_, pe_cin_padded_, pe_cout_,
                pe_strideh_, pe_stridew_,
                hidden, /*seq_off=*/i * npp, /*img_slot=*/(int)i, offsets, nc,
                consume_external_prologue ? &ctx() : nullptr);
        }
        auto& graph = RpuKernelGraph::active();
        const auto graph_state = graph.state();
        if (graph_state == RpuKernelGraph::State::RECORDING ||
            graph_state == RpuKernelGraph::State::REPLAYING) {
            graph.record_branch(
                0x5349474c49505f52ull,  // "SIGLIP_R"
                GraphSignature{},
                "hyvit2_patch_embed_spm_reset");
        }
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
    // ── packed 路径的 2D 块对角 mask（按 (seq_len, N) 缓存）───────────────────
    // minibatch SDPA 走不通：它要 `per_image_ctx % tile_m == 0`，而
    // `tile_m = tile_m_v16 * 16` 恒为 16 的倍数，Hy-VLA 的 per_image_ctx=196
    // (=2²×7²) 没有 16 的倍数因子。改用普通 unified SDPA + 显式 2D mask：
    // image i 的 query 只对自己那 per_image_ctx 列开口，其余为 MASK_NEG。
    // ⚠️ **必须缓存**：seq_len 非 16 对齐时 `sdpa_prepare_mask` 会 pad 出**新张量**，
    //    每帧重建会让地址漂移，而它的 DMA 源在 BUILD 期就烘进图了（跨帧 REPLAY
    //    会读到已释放的块）。按 (seq_len,N) 缓存 ⇒ 地址跨帧稳定。
    static constexpr float kPackedMaskNeg = -50000.0f;   // 与 Python 侧 MASK_NEG 一致
    std::map<std::pair<int64_t, int64_t>, PreparedMask> packed_masks_;

    const PreparedMask& packed_block_mask(int64_t seq_len, int64_t n_img) {
        auto key = std::make_pair(seq_len, n_img);
        auto it = packed_masks_.find(key);
        if (it != packed_masks_.end()) return it->second;
        TORCH_CHECK(n_img > 0 && seq_len % n_img == 0,
                    "packed_block_mask: seq_len(", seq_len, ") % n_img(", n_img, ") != 0");
        const int64_t per = seq_len / n_img;
        at::Tensor m = at::full({1, 1, seq_len, seq_len}, kPackedMaskNeg,
                                at::TensorOptions().dtype(at::kHalf));
        for (int64_t i = 0; i < n_img; ++i) {
            m.slice(2, i * per, (i + 1) * per)
             .slice(3, i * per, (i + 1) * per).fill_(0.0f);
        }
        // sdpa_prepare_mask is the single CPU-normalization/DDR-publication
        // boundary.  Uploading here and immediately reading the same immutable
        // mask back on the host would force an unrelated, already-recorded
        // patch-embed prefix through Graph sync_point on the first capture.
        m = m.contiguous();
        packed_masks_.emplace(
            key, sdpa_prepare_mask(c10::optional<at::Tensor>(m), /*is_causal=*/false,
                                   seq_len, seq_len, sdpa_stable_mask_cache(),
                                   n_img));
        return packed_masks_.at(key);
    }

    // ----- Model state -----
    HYViT2ColdConfig cold_config_;
    bool cold_config_bound_ = false;
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.insert(identity.end(), {
            static_cast<int64_t>(merger_member_major_),
            static_cast<int64_t>(proj1_in_merger_),
            static_cast<int64_t>(cold_config_.fast_replay),
            static_cast<int64_t>(cold_config_.fast_replay_preload),
            static_cast<int64_t>(cold_config_.mask_once),
            static_cast<int64_t>(cold_config_.kvpad16),
            static_cast<int64_t>(cold_config_.q_inplace),
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
                &proj_w_, &proj_b_, &patch_emb_weight_, &patch_emb_pos_emb_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerBiasNorm> layer_bias_norm_;

    // Global weights
    at::Tensor proj_w_, proj_b_;
    // 项 A：proj1 之前把行序换成 member-major，供 tower 外 merger 的组轴算子用。
    bool merger_member_major_ = false;
    bool proj1_in_merger_ = true;
    int patch_embed_cores_ = NUM_CORES;
    bool production_config_bound_ = false;

    // Intermediate DDR staging buffer (post-LN → DDR → projector). Shape-gated
    // reuse is SAFE here because this tensor is a class member that never
    // escapes to Python — no caller accumulates references to it. The
    // per-forward spm_copy_ddr_dma writes to this stable DDR address, and the
    // projector Linear (still inside the same main graph batch) reads from
    // it immediately. Shape-keyed map: each (seq_len, hidden_size) gets its
    // own slot so multi-shape REPLAY keeps every shape's DMA address valid.
    std::map<std::pair<int64_t, int64_t>, at::Tensor> temp_ddr_slots_;

    // Fresh on every forward, including when fast replay skips the layer body.
    // Retained DMA nodes refer to these member slots and resolve their current
    // values while holding the destination's submission lease.
    at::Tensor projector_output_;
    uint64_t projector_out_live_base_ = 0;
    uint64_t projector_bias_live_base_ = 0;

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

    // Q_INPLACE（RPU_HY_VLA_Q_INPLACE）：单 chunk 下 Q 不再走 DDR 中转。
    // declare_buffers 每次 forward 重算，两个 DMA 站点读它（并各自再断言一次
    // `chunk.len == seq_len_ && chunk.offset == 0`，防止规划与执行不一致）。
    bool q_inplace_ = false;

    // KVPAD16（见文件头）：k/v 槽被保证的 SPM 行容量；0 = 关闭。
    int64_t kv_pad_rows_ = 0;

    // SigLIP-batch: number of images packed along seq this forward (1 = legacy
    // single-image path → unified_v2 SDPA; >1 → per-image minibatch SDPA). Set
    // in forward() from input.size(0); read in build_layer_subgraph.
    int64_t image_batch_count_ = 1;
    bool planning_external_patch_prologue_ = false;

    // Patch embedding params
    at::Tensor patch_emb_weight_, patch_emb_pos_emb_;
    int64_t patch_kernel_size_ = 0, patch_stride_ = 0;
    int64_t pe_kh_ = 0, pe_kw_ = 0;
    int64_t pe_cin_orig_ = 0, pe_cin_padded_ = 0, pe_cout_ = 0;
    int64_t pe_strideh_ = 0, pe_stridew_ = 0;
    bool has_patch_emb_ = false;
};

}  // namespace v3

// =============================================================================
// Instance registry (C-01) — uses ModelHandleRegistry<v3::HYViT2VisionModel> template.
// =============================================================================

using HYViT2Registry = ModelHandleRegistry<v3::HYViT2VisionModel>;

std::vector<int64_t> rpu_hyvit2_planner_cache_identity(int64_t handle) {
    return HYViT2Registry::get(handle, "rpu_hyvit2_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_hyvit2_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    HYViT2Registry::get(handle, "rpu_hyvit2_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_hyvit2_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return HYViT2Registry::get(handle, "rpu_hyvit2_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_hyvit2_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return HYViT2Registry::get(handle, "rpu_hyvit2_kvinsert_cost_domain")
        ->kvinsert_cost_domain("hyvit2", descriptor);
}

std::string rpu_hyvit2_kvinsert_cost_catalog_sha256(int64_t handle) {
    return HYViT2Registry::get(
        handle, "rpu_hyvit2_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

int64_t rpu_hyvit2_create() {
    return HYViT2Registry::create();
}

void rpu_hyvit2_set_runtime_config(
    int64_t handle,
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool kvpad16, bool q_inplace) {
    HYViT2Registry::get(handle, "rpu_hyvit2_set_runtime_config")
        ->set_runtime_config(
            fast_replay, fast_replay_preload, mask_once, kvpad16,
            q_inplace);
}

void rpu_hyvit2_destroy(int64_t handle) {
    HYViT2Registry::get(handle, "rpu_hyvit2_destroy")
        ->check_execution_reconfigure_destroy_allowed("rpu_hyvit2_destroy");
    HYViT2Registry::destroy(handle, "rpu_hyvit2_destroy");
}

void rpu_hyvit2_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    const at::Tensor& proj_w, const at::Tensor& proj_b,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    int64_t projection_dim, double eps,
    bool fused_merger, bool proj1_in_merger,
    int64_t patch_embed_cores)
{
    std::vector<at::Tensor> empty_scales;
    HYViT2Registry::get(handle, "rpu_hyvit2")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        proj_w, proj_b,
        num_heads, head_dim, hidden_size, intermediate_size,
        projection_dim, eps, fused_merger, proj1_in_merger,
        patch_embed_cores,
        empty_scales, empty_scales, empty_scales, empty_scales,
        empty_scales, empty_scales);
}

void rpu_hyvit2_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    const at::Tensor& proj_w, const at::Tensor& proj_b,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    int64_t projection_dim, double eps,
    bool fused_merger, bool proj1_in_merger,
    int64_t patch_embed_cores,
    at::TensorList q_ws_list, at::TensorList k_ws_list,
    at::TensorList v_ws_list, at::TensorList o_ws_list,
    at::TensorList fc1_ws_list, at::TensorList fc2_ws_list)
{
    HYViT2Registry::get(handle, "rpu_hyvit2")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        proj_w, proj_b,
        num_heads, head_dim, hidden_size, intermediate_size,
        projection_dim, eps, fused_merger, proj1_in_merger,
        patch_embed_cores,
        q_ws_list, k_ws_list, v_ws_list, o_ws_list,
        fc1_ws_list, fc2_ws_list);
}

void rpu_hyvit2_model_set_patch_emb(
    int64_t handle,
    const at::Tensor& weight,
    const at::Tensor& pos_emb,
    int64_t kernel_size,
    int64_t stride)
{
    HYViT2Registry::get(handle, "rpu_hyvit2")->set_patch_emb_params(
        weight, pos_emb, kernel_size, stride);
}

at::Tensor rpu_hyvit2_patch_embed(
    int64_t handle,
    const at::Tensor& input)
{
    return HYViT2Registry::get(handle, "rpu_hyvit2")->patch_embed(input);
}

at::Tensor rpu_hyvit2_patch_embed_multi(
    int64_t handle,
    at::TensorList images)
{
    return HYViT2Registry::get(handle, "rpu_hyvit2")->patch_embed_multi(images);
}

at::Tensor rpu_hyvit2_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return HYViT2Registry::get(handle, "rpu_hyvit2")->forward(
        input, k_caches, v_caches, planned_stage_descriptor);
}

at::Tensor rpu_hyvit2_forward_packed(
    int64_t handle,
    const at::Tensor& hidden,
    int64_t image_batch_count,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return HYViT2Registry::get(handle, "rpu_hyvit2")->forward_packed(
        hidden, image_batch_count, k_caches, v_caches,
        planned_stage_descriptor);
}

at::Tensor rpu_hyvit2_forward_multi(
    int64_t handle,
    at::TensorList images,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return HYViT2Registry::get(handle, "rpu_hyvit2")->forward_multi(
        images, k_caches, v_caches, planned_stage_descriptor);
}

std::vector<int64_t> rpu_hyvit2_resolve_stage_domain(
    int64_t handle, int64_t seq_len, int64_t image_batch_count,
    bool external_patch_prologue) {
    return HYViT2Registry::get(handle, "rpu_hyvit2_resolve_stage_domain")
        ->resolve_stage_domain(
            seq_len, image_batch_count, external_patch_prologue);
}

int64_t rpu_hyvit2_get_resolved_chunk_size(int64_t handle) {
    return HYViT2Registry::get(
               handle, "rpu_hyvit2_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

void rpu_hyvit2_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    HYViT2Registry::get(handle, "rpu_hyvit2_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_hyvit2_set_chunk_size_override(
    int64_t handle, int64_t chunk_size) {
    HYViT2Registry::get(handle, "rpu_hyvit2_set_chunk_size_override")
        ->set_control_chunk_size_override(
            chunk_size, "rpu_hyvit2_set_chunk_size_override");
}

void rpu_hyvit2_enable_execution_reconfigure(int64_t handle) {
    HYViT2Registry::get(handle, "rpu_hyvit2_enable_execution_reconfigure")
        ->enable_execution_reconfigure_guard();
}

void rpu_hyvit2_stage_chunk_size_override(
    int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0,
                "HYViT2 hot-reconfigure token must be positive");
    HYViT2Registry::get(handle, "rpu_hyvit2_stage_chunk_size_override")
        ->stage_control_chunk_size_override(
            static_cast<uint64_t>(token), chunk_size,
            "rpu_hyvit2_stage_chunk_size_override");
}
