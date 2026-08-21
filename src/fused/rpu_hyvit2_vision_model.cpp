// rpu_hyvit2_vision_model.cpp — HYViT2-400M AnyRes ViT，all-layers-once
// (v3 FusedModelBase)。与 `rpu_siglip_model.cpp` 共享主体结构。
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
// 且逐行非线性、无法用任何权重取值抵消。
// 故 `set_weights` 相比 SigLIP 少了 `post_ln_w/post_ln_b` 两个参数（24 个而非 26）。
// projector 槽接 merger 的 `proj1 [2048,1152]`（逐 token 纯 Linear，与 SigLIP projector 同语义）；
// merger 余下的 DwPooler→GELU→proj2 由 host 承担。
//
// 其他与 SigLIP 的共性约束（padding / scale / 打包）：
//   - Python 侧把 fused qkv [3456,1152] 按 [3, heads, head_dim] 拆成 q/k/v；
//     head_dim 72→80、intermediate 4304→4352 补零（与 adapters/siglip.py 同式）。
//   - ⚠️ attention scale 用 ORIG head_dim=72，不是 padded 的 80（见下方 orig_head_dim_）。
//   - 推理路径上每次 forward 只处理 1 张图（seq=196，image_batch_count_=1）；
//     多图打包不走这条路。
//
// 下方注释说明与 SigLIP 共用的 preload_fn、KV_FIRST 和缓冲区生命周期契约。
#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_spm_allocator.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_FAST_REPLAY —— **逐座 opt-in** 的 fast-replay，默认 OFF。
//
// 打开后 REPLAY 期不再重走该座的 op stream，避免逐帧重新 emit 节点的 host 开销。
//
// ⚠️ 契约：**body 不得有 kernel 会读的
// per-call host 副作用**。合法出口只有：position_ids 的稳定 keepalive、
// dynamic_config 拷进稳定 DDR 槽的显式 2D mask、`hidden_in_src_base_` 与
// op 序言里 `*_mutable(&member)` 的可变基址。
//
// 该开关按子系统独立控制，避免多个变化同时启用后无法定位问题。
// 取值：逗号分隔的座名（vit / vlm / expert），或 1/on/true 表示三座全开。
// ─────────────────────────────────────────────────────────────────────────────
static bool hyvla_fast_replay_on(const char* who) {
    static const std::string v = [] {
        const char* e = std::getenv("RPU_HY_VLA_FAST_REPLAY");
        return std::string(e ? e : "");
    }();
    if (v.empty() || v == "0" || v == "off" || v == "false") return false;
    if (v == "1" || v == "on" || v == "true") return true;
    return v.find(who) != std::string::npos;
}

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_FAST_REPLAY_PRELOAD —— 同样**逐座 opt-in**，默认 OFF。
// 打开后在 REPLAY（且权重 clean）跳过 preload 回调与 `preload_fn` 的
// host 重走 —— 已建好的 preload DMA 节点仍由 segment launch 重放到同一块
// persistent SPM，逐位不变。
// 依赖 `fast_replay_skip_layer_loop` 一起开（框架内部与它 AND）。
// ─────────────────────────────────────────────────────────────────────────────
static bool hyvla_fast_replay_preload_on(const char* who) {
    static const std::string v = [] {
        const char* e = std::getenv("RPU_HY_VLA_FAST_REPLAY_PRELOAD");
        return std::string(e ? e : "");
    }();
    if (v.empty() || v == "0" || v == "off" || v == "false") return false;
    if (v == "1" || v == "on" || v == "true") return true;
    return v.find(who) != std::string::npos;
}

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_MASK_ONCE —— 同样**逐座 opt-in**，默认 OFF。
// 打包 3 图后本座的块对角 2D mask 是 [588,592] = 696 KB/核，且逐层恒定。
// 其机制与其他子系统的同名开关一致。
//
// ⚠️ 本座与 VLM/expert 有一处**关键差别**：这里的 `sdpa_mask` 原本声明在
// `Compute` 域，而分配器规定 `KvInsert` 与 `Compute` **恒不冲突**
// —— 只把相位撑满、留在 Compute 域，
// 下一层 Phase 1 的 `q_kv/k/v` 照样会压在同一段地址上、静默读到坏 mask。
// 所以打开时必须**同时**把它挪到 `LayerWide`。代价是不再能与 `fc1`（相 7..8）
// 混叠，本座 SPM 峰值 +696 KB。
// ─────────────────────────────────────────────────────────────────────────────
static bool hyvla_mask_once_on(const char* who) {
    static const std::string v = [] {
        const char* e = std::getenv("RPU_HY_VLA_MASK_ONCE");
        return std::string(e ? e : "");
    }();
    if (v.empty() || v == "0" || v == "off" || v == "false") return false;
    if (v == "1" || v == "on" || v == "true") return true;
    return v.find(who) != std::string::npos;
}

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_Q_INPLACE —— 单 chunk 专用，默认 OFF。
// 打开后 Q 保持在 LayerWide SPM 直到 SDPA 消费，q_comp alias 到 q_kv，
// 省略相间的 Q staging DMA。q_kv 必须使用 LayerWide scope，避免与 Compute
// 临时区混叠。
// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_KVPAD16 —— 逐座 opt-in，默认 OFF。把 k/v 槽行容量向上补齐
// 到 v16 边界；逻辑 KV 长度不变。
static bool hyvla_kvpad16_on(const char* who) {
    static const std::string v = [] {
        const char* e = std::getenv("RPU_HY_VLA_KVPAD16");
        return std::string(e ? e : "");
    }();
    if (v.empty() || v == "0" || v == "off" || v == "false") return false;
    if (v == "1" || v == "on" || v == "true") return true;
    return v.find(who) != std::string::npos;
}

static bool hyvla_q_inplace_on() {
    static const bool on = [] {
        const char* e = std::getenv("RPU_HY_VLA_Q_INPLACE");
        const std::string v(e ? e : "");
        return !(v.empty() || v == "0" || v == "off" || v == "false");
    }();
    return on;
}

// RPU_HY_VLA_FUSED_MERGER —— 全局默认 OFF。打开后 ViT 尾部在 proj1 之前把行序换成
// member-major（两次 permute3d），tower 外 merger 的组轴改走
// `rpu.hyvla_merger_pool` / `_combine` 两个 op。**Python 与 C++ 必须同开同关**：
// 它改的是行布局，不是数值。
static bool hyvla_fused_merger_on() {
    static const bool on = [] {
        const char* e = std::getenv("RPU_HY_VLA_FUSED_MERGER");
        return e && *e && std::string(e) != "0" &&
               std::string(e) != "false" && std::string(e) != "False";
    }();
    return on;
}

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

// =============================================================================
// Fused patch embedding is an internal helper used by
// HYViT2VisionModel::forward's 4D auto-detect path; it is not registered as a
// torch op.
// =============================================================================
namespace {

// RPU_HY_VLA_PATCH_EMBED_MC — 默认 ON（设 0 使用单核）。Python 与 C++
// 必须一致，因为权重 swizzle 的 num_cores 跟随此开关。
// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_PROJ1_IN_MERGER —— C++ 侧默认 ON。打开时 ViT 输出
// member-major [S, hidden]，由 merger 图执行 proj1。Python 必须使用同一设置。
static bool hyvla_proj1_in_merger() {
    static const bool on = [] {
        const char* e = std::getenv("RPU_HY_VLA_PROJ1_IN_MERGER");
        if (!e || !*e) return true;
        const std::string v(e);
        return !(v == "0" || v == "off" || v == "false" || v == "False");
    }();
    return on;
}

static int hyvla_patch_embed_cores() {
    static const int nc = [] {
        const char* e = std::getenv("RPU_HY_VLA_PATCH_EMBED_MC");
        const bool off = e && *e && (std::string(e) == "0" ||
                                     std::string(e) == "false" ||
                                     std::string(e) == "False");
        return off ? 1 : NUM_CORES;
    }();
    return nc;
}

// Writes one image's [num_patches, cout] embedding into `out` at `seq_off`.
// `img_slot` selects a stable per-image mutable-DMA base in the shared Graph;
// output and position-embedding bases remain shared. The caller allocates one
// reusable set of SPM offsets, preserving bounded usage and Graph dependencies.
static void fused_patch_embedding(
    const at::Tensor& input_nchw,    // [1, cin_orig, H, W] fp16 RPU DDR
    const at::Tensor& weight,        // [cout, K] fp16 RPU DDR, col-swizzled (1 core)
    const at::Tensor& pos_emb_fused, // [1, num_patches, cout] fp16 RPU DDR
    int64_t kh, int64_t kw,
    int64_t cin_orig, int64_t cin_padded, int64_t cout,
    int64_t strideh, int64_t stridew,
    at::Tensor& out, int64_t seq_off, int img_slot,
    const std::vector<uint32_t>& offsets)
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

    const int nc = hyvla_patch_embed_cores();
    TORCH_CHECK(cout % nc == 0,
                "fused_patch_embedding: cout(", cout, ") must be divisible by "
                "num_cores(", nc, ") for the col-partition GEMM");

    // ===== Step ① DMA raw input DDR → SPM =====
    // Mutable DMA: input_nchw is caller-supplied per-forward, address drifts.
    // Fixed variant would bake a BUILD-time address that the sync-only path cannot rewrite.
    //
    // NOTE: input_nchw MUST already be DDR-coherent here. The driver
    // (run_packed_patch_embed_core's per-image loop) flushes each image before
    // this call — the flush lives at the batch driver, explicit, rather than
    // buried per-helper. See the rpu_ddr_flush_force there for the why (a
    // torch.cat / CPU-fallback input is left un-flushed → stale DDR).
    // Per-image live-base slot (see fn doc): each packed image bakes a distinct
    // &slot so REPLAY rewrites the correct per-image source addr.
    static constexpr int kMaxPackedImages = 8;
    TORCH_CHECK(img_slot >= 0 && img_slot < kMaxPackedImages,
                "fused_patch_embedding: img_slot ", img_slot,
                " out of range [0,", kMaxPackedImages, ")");
    static thread_local uint64_t patch_emb_input_live_base[kMaxPackedImages] = {0};
    patch_emb_input_live_base[img_slot] = ::rhino_lkn::RpuGetDevAddr(
        const_cast<c10::Half*>(input_nchw.data_ptr<c10::Half>()));
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &patch_emb_input_live_base[img_slot],
        /*src_offset_bytes=*/0,
        cin_orig * HW, raw_addr, /*num_cores=*/nc);

    // ===== Step ② Pad channels on each participating core =====
    rpu_launch_pad_channel_spm(
        raw_off, padded_off, /*N=*/1, /*C=*/cin_orig * HW,
        /*pad_front=*/0, /*pad_tail=*/(cin_padded - cin_orig) * HW, nc);

    // ===== Step ③ NCHW → NHWC =====
    rpu_launch_transpose_nchw_to_nhwc_spm(
        padded_off, nhwc_off, (int)cin_padded, (int)H, (int)W, nc);

    // ===== Step ④ im2col =====
    // For multi-core non-overlapping patches, use the equivalent permute3d form.
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
        rpu_launch_all_gather_spm_kernel(
            gemm_out_addr, result_addr,
            /*n=*/num_patches, /*chunk_elems=*/cout / nc, DWIDTH, nc);
    }

    // ===== Step ⑥ pos_emb add =====
    // Mutable form for symmetry with step ① (pos_emb_fused.data_ptr is
    // actually stable — registered model weight — so fixed would also work).
    // rpu_ddr_flush_force(pos_emb_fused.data_ptr<c10::Half>());
    static thread_local uint64_t patch_emb_pos_live_base = 0;
    patch_emb_pos_live_base = ::rhino_lkn::RpuGetDevAddr(
        const_cast<c10::Half*>(pos_emb_fused.data_ptr<c10::Half>()));
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
        at::TensorList q_ws_list, at::TensorList k_ws_list,
        at::TensorList v_ws_list, at::TensorList o_ws_list,
        at::TensorList fc1_ws_list, at::TensorList fc2_ws_list)
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "hyvit2_set_weights: empty weight lists");
        TORCH_CHECK(num_heads > 0 && head_dim > 0 && hidden_size > 0
                    && intermediate_size > 0 && projection_dim > 0,
                    "hyvit2_set_weights: dim params must be positive");

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
        // (fp16 + PrivateUse1 + contiguous).
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

        // Per-layer rank checks. require_fp16_rpu adds the fp16, PrivateUse1 and
        // contiguous preconditions for tensors accessed as c10::Half. Projection
        // weights may be int8 W8A16 and are checked separately above.
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

        // orig_head_dim = hidden_size / num_heads (= 72 for SigLIP), not from
        // the padded weight shape, which would give 80.
        orig_head_dim_ = hidden_size / num_heads;

        // Per-core local Q head count for KV_FIRST q_ddr_buf_ staging
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

        invalidate_model_state();  // Must remain the last statement of set_weights.
    }

    // ========================================================================
    // set_patch_emb_params — optional patch embedding configuration
    // ========================================================================
    void set_patch_emb_params(const at::Tensor& weight, const at::Tensor& pos_emb,
                              int64_t kernel_size, int64_t stride)
    {
        TORCH_CHECK(weight.defined() && weight.dim() == 2,
                    "hyvit2_set_patch_emb: weight must be defined 2D");
        TORCH_CHECK(pos_emb.defined(),
                    "hyvit2_set_patch_emb: pos_emb must be defined");

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
        std::vector<at::Tensor>& v_caches)
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

        // Input formats:
        //  - 4D [N,C,H,W]: pack N images into one hidden tensor in this capture.
        //  - 3D [1,S,hidden]: use a pre-embedded single hidden tensor.
        if (input.dim() == 4 && has_patch_emb_) {
            hidden_states = run_packed_patch_embed(input);  // sets image_batch_count_
            packed_image_count = image_batch_count_;
        } else {
            hidden_states = input;
        }

        return forward_packed(hidden_states, packed_image_count, k_caches, v_caches);
    }

    // ========================================================================
    // forward_multi — all-RPU SigLIP-batch entry. Takes the N camera images as
    // a LIST of [1,C,H,W] tensors (no torch.cat upstream), packs them into one
    // [1, N*256, hidden] hidden via run_packed_patch_embed_list, then runs the
    // identical encoder + projector tail as forward(). image_batch_count_ = N
    // (set by the packer) drives the per-image minibatch SDPA in
    // build_layer_subgraph — bit-exact with forward()'s 4D torch.cat path,
    // without the cat/slice round trip.
    // ========================================================================
    at::Tensor forward_multi(
        at::TensorList images,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches)
    {
        TORCH_CHECK(num_layers() > 0,
                    "HYViT2VisionModel::forward_multi called before set_weights");
        TORCH_CHECK(has_patch_emb_,
                    "HYViT2VisionModel::forward_multi requires patch embedding params — "
                    "call hyvit2_model_set_patch_emb before forward_multi");
        TORCH_CHECK(images.size() > 0,
                    "HYViT2VisionModel::forward_multi: empty image list");
        for (const auto& im : images) {
            TORCH_CHECK(im.device().type() == at::kPrivateUse1,
                        "HYViT2VisionModel::forward_multi: each image must be on RPU device");
            TORCH_CHECK(im.is_contiguous(),
                        "HYViT2VisionModel::forward_multi: each image must be contiguous");
            TORCH_CHECK(im.dim() == 4,
                        "HYViT2VisionModel::forward_multi: each image must be 4D [1,C,H,W], "
                        "got ", im.dim(), "D");
        }

        // Patch-embed + pack all N images into [1, N*256, hidden] (sets
        // image_batch_count_ = N for the per-image minibatch SDPA).
        at::Tensor hidden_states = run_packed_patch_embed_list(images);
        return forward_packed(hidden_states, image_batch_count_, k_caches, v_caches);
    }

    at::Tensor patch_embed(const at::Tensor& input) {
        TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                    "HYViT2VisionModel::patch_embed: input must be on RPU device");
        TORCH_CHECK(input.is_contiguous(),
                    "HYViT2VisionModel::patch_embed: input must be contiguous");
        TORCH_CHECK(input.dim() == 4,
                    "HYViT2VisionModel::patch_embed expects 4D [B,C,H,W] input, got ",
                    input.dim(), "D");
        TORCH_CHECK(has_patch_emb_,
                    "HYViT2VisionModel::patch_embed requires patch embedding params");
        return run_packed_patch_embed(input);
    }

    at::Tensor patch_embed_multi(at::TensorList images) {
        TORCH_CHECK(has_patch_emb_,
                    "HYViT2VisionModel::patch_embed_multi requires patch embedding params");
        TORCH_CHECK(images.size() > 0,
                    "HYViT2VisionModel::patch_embed_multi: empty image list");
        for (const auto& im : images) {
            TORCH_CHECK(im.device().type() == at::kPrivateUse1,
                        "HYViT2VisionModel::patch_embed_multi: each image must be on RPU");
            TORCH_CHECK(im.is_contiguous(),
                        "HYViT2VisionModel::patch_embed_multi: each image must be contiguous");
            TORCH_CHECK(im.dim() == 4 && im.size(0) == 1,
                        "HYViT2VisionModel::patch_embed_multi: each image must be [1,C,H,W]");
        }
        return run_packed_patch_embed_list(images);
    }

    at::Tensor forward_packed(
        const at::Tensor& hidden_states,
        int64_t packed_image_count,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches)
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
        // kv_seq_len). The stable per-shape Q staging slot is allocated below,
        // after validation, so an invalid packed shape cannot leak a slot.
        seq_len_ = seq_len;

        // Minibatch (N>1 packed images) mutual-exclusion with chunking: the
        // per-image minibatch SDPA launcher asserts a SINGLE chunk spanning the
        // whole packed sequence. plan_kv_first_chunks (Site8) forces the
        // KV-insert plan to one chunk, but the framework's COMPUTE chunk plan is
        // computed independently by compute_chunks_impl — so for N>1 we also pin
        // the compute chunk size to the full sequence via the override (which
        // bypasses the auto-scan). N==1 (HALO single-image) leaves the override
        // at 0 so the
        // auto-scan can pick a fitting comp_cs and token-chunk the 1564 path.
        //
        // The override is floored to (override/16)*16. A packed seq not a multiple of 16 — or not
        // divisible by image_batch_count_ — would split a TAIL chunk, violating the
        // single-block minibatch assertion (TORCH_CHECK(chunk.len==seq_len_) in the
        // KV_FIRST body below). Fail
        // fast HERE with a clear shape error. SigLIP packs per-image npp(=256)
        // tokens so seq_len = N*256 is 16-aligned in practice; this guards the
        // invariant the forced-single-block override silently relies on.
        // 传 **ceil16(seq_len)** 时，框架把 override clamp 到
        // hi=ceil16(seq_len) (:501), 再按 len=min(cs, seq_len-off) 建块 ⇒
        // 恰好一块、len==seq_len，因此 seq_len 无需被 16 整除。
        // 仍然要求能被 image_batch_count_ 整除: minibatch SDPA 的
        // per_image_ctx = seq_len / N 必须是整数。
        TORCH_CHECK(
            image_batch_count_ == 1 || (seq_len % image_batch_count_ == 0),
            "HYViT2 packed forward: seq_len (", seq_len,
            ") must be divisible by image_batch_count (", image_batch_count_,
            ") — minibatch SDPA 的 per_image_ctx 必须整除");
        set_chunk_size_override(
            image_batch_count_ > 1 ? ((seq_len + 15) / 16) * 16 : 0);

        // KV_FIRST: shape now validated → ensure a STABLE Q staging slot for
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

        // Fail fast when even the minimum legal chunk exceeds the SPM budget.
        // The framework's auto planner remains the authoritative per-candidate gate.
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

        at::Tensor result = run_all_layers(
            hidden_states, k_caches, v_caches,
            /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false);

        // Last layer freshly allocated projector_output_ (Pitfall 4); flush for
        // CPU coherency before returning to Python (framework doesn't track it).
        if (projector_output_.defined()) {
            rpu_ddr_flush(projector_output_.data_ptr<c10::Half>());
            return projector_output_;
        }
        return result;
    }

protected:
    // ========================================================================
    // static_config — SIGLIP_WEIGHTS preload and SIGLIP_COMPUTE layer graphs.
    //
    // Registers the pointer-to-member `&HYViT2VisionModel::emit_preload_weights`
    // (cast to FusedModelBase pointer-to-member) on the preload-fn slot.
    // Framework dispatches via std::invoke inside a framework-managed
    // GRAPH_CACHE.begin(SIGLIP_WEIGHTS) / end pair.
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers       = num_layers();

        // One-shot .preload_fn. Framework wraps the callback body in ONE
        // GRAPH_CACHE.begin(SIGLIP_WEIGHTS)/end pair. The body emits only
        // rpu_launch_* DMAs with no subclass-side batch-context calls.
        // SigLIP uses one-shot preload_fn while Gemma uses per-buffer
        // .preload_callback instead; both coexist here.
        cfg.preload_fn = static_cast<void(FusedModelBase::*)()>(
                             &HYViT2VisionModel::emit_preload_weights);

        // Two pointer-to-member slots implement the KV_FIRST two-phase dispatch
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

        // SigLIP requires a single group. With cross_batch < num_layers,
        // cached kernels may bind different layer-specific weights, KV caches,
        // and per-layer SPM offsets across groups, producing non-deterministic
        // output. Pinning the full layer count keeps that dataflow invariant.
        cfg.cross_layer_batch_size = num_layers();
        cfg.fast_replay_skip_layer_loop = hyvla_fast_replay_on("vit");
        cfg.fast_replay_skip_preload = hyvla_fast_replay_preload_on("vit");
        return cfg;
    }

    // ========================================================================
    // dynamic_config — KV_FIRST + AUTO inter-layer I/O.
    //
    // KV_FIRST keeps full-sequence K/V in DDR and feeds SPM one query chunk at a
    // time. AUTO selects SPM_RESIDENT for one chunk and DDR_PINGPONG otherwise.
    //
    // The base ping-pong DMA uses a single core-0 path, not multicore broadcast.
    // No build_chunk_masks: SigLIP attention is MASK_NONE (bidirectional, no
    // additive mask), unlike Gemma's per-chunk causal masks.
    // ========================================================================
    ModelDynamicConfig dynamic_config(const ChunkPlan& /*plan*/) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::KV_FIRST;
        cfg.inter_layer_io = InterLayerIO::AUTO;
        return cfg;
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
    // SigLIP uses one-shot preload-fn instead of per-buffer
    // `.preload_callback` (which would be the Gemma pattern) — so NO
    // `.preload_callback` field is set on any decl here.
    // ========================================================================
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        // Three KV_FIRST chunk sizes drive three buffer scopes
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

        // LayerWide: read in both phases → sized at wide_cs.
        int64_t res  = A(wide_cs * h * DWIDTH);
        // KvInsert: Phase-1 QKV outputs (q renamed → q_kv; Phase 1 writes q_kv,
        // dumps it to q_ddr_buf_, never reads it again — Phase 2 reloads from
        // DDR into q_comp).
        int64_t qkv  = A(kv_cs   * local_q_dim * DWIDTH);
        // Compute: Phase-2 working buffers → sized at comp_cs.
        int64_t q_comp = A(comp_cs * local_q_dim * DWIDTH);
        int64_t sdpa_out = A(comp_cs * local_q_dim * DWIDTH);
        int64_t fc1  = A(comp_cs * local_inter * DWIDTH);

        // Size SDPA temporaries for the Phase-2 query-chunk length.
        SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                            hd, /*nq*/nq, /*nkv*/nq,
                            /*cores*/NUM_CORES,
                            /*mask*/image_batch_count_ > 1 ? 4 : 0};
        SdpaTiling t = sdpa_compute_tiling(sdpa_cfg, comp_cs);
        int64_t nkv_per_core = CeilDiv(nq, (int64_t)NUM_CORES);
        int64_t sdpa_tmp = A(t.tile_n_v16 * t.tile_k * nkv_per_core * CeilDiv(comp_cs, t.tile_m) * 32);

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
        // oproj: Phase-2 o_proj output / LN2 output / fc2 partial scratch,
        // retained in the LayerWide band across both phases.
        decls.push_back({"oproj",      res, 4, 8, StorageClass::Temp, 0, nullptr, ALL});

        // Q_INPLACE（见文件头）：只在**单 chunk**（两相同一个 chunk 且覆盖整段
        // 序列）下成立 —— 否则 Phase 1/Phase 2 之间隔着别的 chunk，q 必须过 DDR。
        q_inplace_ = hyvla_q_inplace_on()
                     && comp_cs == kv_cs && comp_cs >= seq_len_ && seq_len_ > 0;

        // KvInsert-only (Phase 1) — aliasable against Compute-only buffers.
        // Q_INPLACE 下 q_kv 提到 LayerWide 且活到 SDPA 消费完（相 1..6）。
        decls.push_back({"q_kv",       qkv, 1, q_inplace_ ? 6 : 3, StorageClass::Temp, 0,
                         nullptr, q_inplace_ ? ALL : KVIN});
        // KVPAD16（见文件头）：k/v 槽补到 16 的倍数行，KV-insert 一次 v16 打完。
        kv_pad_rows_ = hyvla_kvpad16_on("vit") ? Align(kv_cs, (int64_t)16) : 0;
        const int64_t kv_sz = kv_pad_rows_ > 0
            ? std::max(qkv, A(kv_pad_rows_ * local_q_dim * DWIDTH)) : qkv;
        decls.push_back({"k",          kv_sz, 1, 3, StorageClass::Temp, 0, nullptr, KVIN});
        decls.push_back({"v",          kv_sz, 1, 3, StorageClass::Temp, 0, nullptr, KVIN});

        // Compute-only (Phase 2) — lifecycle aliasing across attention→MLP.
        // Q_INPLACE 下 q_comp 就是 q_kv 本身（alias_of，不占新字节）。
        if (q_inplace_)
            decls.push_back({"q_comp", 0, 0, 0, StorageClass::Temp, 0, "q_kv", ALL});
        else
            decls.push_back({"q_comp", q_comp, 4, 5, StorageClass::Temp, 0, nullptr, COMP});
        decls.push_back({"sdpa_out",   sdpa_out, 5, 6, StorageClass::Temp, 0, nullptr, COMP});
        decls.push_back({"sdpa_tmp",   sdpa_tmp, 5, 5, StorageClass::Temp, 0, nullptr, COMP});
        // packed 路径的显式 2D mask 槽（相位窗口贴着 SDPA 的消费窗，不与
        // sdpa_tmp 混叠）。尺寸公式与基类一致。
        // MASK_ONCE 打开时改成 LayerWide + 撑满相位（见文件头），这样它才既不被
        // Phase-2 的 fc1 压、也不被下一层 Phase-1 的 q_kv/k/v 压。
        if (image_batch_count_ > 1) {
            const bool once = hyvla_mask_once_on("vit");
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
    // emit_preload_weights — 272 persistent weight DMAs
    //
    // Registered through the preload-fn slot in static_config(); the framework
    // opens the weights-graph scope around this call.
    //
    // The body emits only rpu_launch_* DMAs; no subclass-side
    // batch-context calls (framework owns them). CRITICAL: DMA length is num_elements
    // (NOT bytes). Per layer (27 layers, 10 DMA ops each) + 2 global = 272 total.
    // ========================================================================
    void emit_preload_weights() {
        int64_t h = hidden_size();
        int64_t local_q_dim = (num_q_heads() / NUM_CORES) * head_dim();
        int64_t local_inter = intermediate_size() / NUM_CORES;

        // Loop 1: 8-core DMAs for all layers (memset + norm + col-partition bias)
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

        // Loop 2: group 1-core DMAs for row-partition biases separately from
        // multicore preload operations.
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
    // emit_kv_first_body — kv_first_fn (KV_FIRST Phase 1 body).
    //
    // Phase 1 runs LayerNorm1 → QKV+bias → KV-insert and saves Q to DDR
    // so Phase 2 (build_layer_subgraph) can reload Q per query-chunk. Mirrors
    // GemmaModel::emit_kv_first_body but SIMPLIFIED: SigLIP has NO rope (position
    // embedding is added in patch_embed, not per-layer) and NO attention mask.
    //
    // KV-insert uses ctx().position + chunk.offset as its absolute KV row.
    // SDPA and KV-insert receive typed SPM offsets via addr_offset(name).value.
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
            seq_len, h, eps_, false, 0, NUM_CORES);

        // QKV Linear with bias (SPM-to-SPM ACC16, col-partition). q → q_kv.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q_kv"),
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

        // No RoPE (SigLIP has none).

        // KV cache insert at absolute position chunk.offset.
        // Pitfall 3 structural fix: takes SPM offsets via addr_offset(name).value.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        rpu_launch_insert_kcache_spm_unified(
            k_cache, kv_pos, addr_offset("k").value,
            seq_len, nq, hd, NUM_CORES, /*cache_batch_offset_elems=*/0,
            /*allow_non8_v16=*/false, /*allow_hybrid_v16=*/false,
            /*spm_rows=*/kv_pad_rows_);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, kv_pos, addr_offset("v").value,
            seq_len, nq, hd, NUM_CORES, /*cache_batch_offset_elems=*/0,
            /*allow_non8_v16=*/false, /*allow_hybrid_v16=*/false,
            /*spm_rows=*/kv_pad_rows_);

        // Save Q to the per-seq Q staging slot via spm_scatter_ddr_dma
        // (position-indexed by chunk.offset). The slot (keyed by seq_len_) is
        // created once in forward_packed and NEVER reallocated → stable data_ptr,
        // safe for non-mutable DMA across REPLAY. Phase 2 reloads it into "q_comp".
        // Q_INPLACE: 单 chunk 下 Phase 2 紧跟其后、直接读 q_kv ⇒ 整趟往返不发。
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
        int64_t q_local_elems = seq_len * local_q_heads_ * hd;
        c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
        int64_t q_row_stride = local_q_heads_ * hd;
        int64_t q_elem_offset = chunk.offset * q_row_stride;
        int64_t q_core_stride_bytes = q_slot.size(1) * q_row_stride * DWIDTH;
        rpu_launch_spm_scatter_ddr_dma(
            addr(0, "q_kv"), q_ddr_base + q_elem_offset,
            q_local_elems, q_core_stride_bytes,
            /*num_cores=*/NUM_CORES);

        // No residual output DMA — Phase 2 re-reads the original input (matching
        // Gemma); the SPM_RESIDENT single-chunk case keeps residual1 live.
    }

    // ========================================================================
    // build_layer_subgraph — KV_FIRST Phase 2 body.
    //
    // Phase 2: reload Q from q_ddr_buf_ → q_comp → SDPA (full-KV) → O_proj+resid
    //          → LayerNorm2+fc1+GELU → fc2+resid.
    //
    // SDPA uses seq_q = chunk.len (the query-chunk) and kv_seq_len = seq_len_
    // (the FULL bidirectional KV) — the two are decoupled by the launcher (see
    // rpu_launch_sdpa_spm_unified_kernel_v2 signature). mask=0 (MASK_NONE).
    //
    // Last layer: fuses post-LN + projector Linear.
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
        // reallocated, its data_ptr stays valid across REPLAY.
        // Q_INPLACE: q 还留在 q_kv 里，而 "q_comp" 已 alias 到它 ⇒ 不必读回。
        if (!q_inplace_) {
            const at::Tensor& q_slot = q_ddr_slot();
            int64_t q_local_elems = seq_len * local_q_heads_ * hd;
            c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
            int64_t q_row_stride = local_q_heads_ * hd;
            int64_t q_elem_offset = chunk.offset * q_row_stride;
            int64_t q_core_stride = q_slot.size(1) * q_row_stride * DWIDTH;
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
            // 块对角 mask 逐层恒定（且此路径已断言单 chunk）⇒ 槽撑满相位后
            // 只需在层 0 灌一次（见文件头 MASK_ONCE）。
            if (!hyvla_mask_once_on("vit") || layer_idx == 0)
                sdpa_dma_mask_to_spm(pm, mask_off, seq_len_, seq_len_, NUM_CORES);
            rpu_launch_sdpa_spm_unified_kernel_v2(
                k_cache, v_cache,
                pm.mask_type, attn_scale,
                addr_offset("q_comp").value,
                addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value, mask_off,
                /*seq_q=*/seq_len, nq, nq, hd,
                /*kv_seq_len=*/seq_len_, NUM_CORES, NUM_CORES);
        } else {
            rpu_launch_sdpa_spm_unified_kernel_v2(
                k_cache, v_cache,
                0 /*MASK_NONE*/, attn_scale,
                addr_offset("q_comp").value,
                addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value, 0,
                /*seq_q=*/seq_len, nq, nq, hd,
                /*kv_seq_len=*/seq_len_, NUM_CORES, NUM_CORES);
        }

        // Phase 4: O_proj (row-partition, with bias) + AllReduce + Residual
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, NUM_CORES,
            layer_addr(layer_idx, 0, "o_bias"), false, lw.o_ws);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
            seq_len, h, NUM_CORES, NUM_CORES);

        // Phase 5: LayerNorm2 + fc1+bias + GELU
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
            seq_len * local_inter, ValuOpType::ADD, /*is_gelu=*/true, NUM_CORES);

        // Phase 6: fc2 (row-partition, with bias) + AllReduce + Residual
        // The final output goes to "residual1" instead of "oproj" so the
        // next layer (which reads from "residual1") gets it directly via SPM
        // without needing DDR ping-pong.
        // Phase 6 uses "oproj" as temporary storage for fc2 partial results,
        // then all_reduce(oproj + input_norm) lands back in "residual1".
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "fc1"), lw.fc2_w, addr(0, "oproj"),
            seq_len, h, is_, 0, NUM_CORES,
            layer_addr(layer_idx, 0, "fc2_bias"), false, lw.fc2_ws);
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
            // The action path has no post-LayerNorm: the encoder output feeds the
            // tower-external merger directly. Stage residual1 to DDR as-is.

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
            // Fused merger mode reorders patch rows to member-major before proj1.
            // Two equivalent permutations reuse existing temporary slots, adding
            // no SPM allocation; the resulting group order matches the merger.
            uint32_t tail_src = addr(0, "residual1");
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
                rpu_launch_permute3d_spm_kernel(
                    tail_src, addr(0, "input_norm"),
                    /*b=*/B * gr * 2 * gr, /*n=*/2, /*c=*/h, {1, 0, 2}, /*num_cores=*/1);
                rpu_launch_permute3d_spm_kernel(
                    addr(0, "input_norm"), addr(0, "oproj"),
                    /*b=*/2 * B * gr, /*n=*/2, /*c=*/gr * h, {1, 0, 2}, /*num_cores=*/1);
                tail_src = addr(0, "oproj");
            }
            if (!(hyvla_proj1_in_merger() && merger_member_major_))
                rpu_launch_spm_copy_ddr_dma(
                    tail_src,
                    slot.data_ptr<c10::Half>(),
                    seq_len * h);

            // The projector output spans the full sequence and is allocated once
            // on the first query chunk. A fresh tracked output prevents references
            // retained by callers from aliasing a later forward.
            // proj1 进 merger 图之后，本塔的输出维就是 hidden（1152），不是 2048。
            const bool p1m = hyvla_proj1_in_merger() && merger_member_major_;
            const int64_t out_dim = p1m ? h : projection_dim_;
            if (chunk.offset == 0) {
                projector_output_ = allocate_tracked_output(
                    {1, seq_len_, out_dim});
            }
            TORCH_CHECK(projector_output_.defined(),
                        "HYViT2VisionModel: projector_output_ undefined on a non-first "
                        "chunk (chunk.offset=", chunk.offset,
                        ") — first chunk must run with offset 0.");

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
                // Write [chunk.len, h] directly to the tracked output for the
                // merger-owned proj1. The fixed SPM-to-DDR destination is captured
                // with the Graph.
                rpu_launch_spm_copy_ddr_dma(
                    tail_src, proj_out_chunk.data_ptr<c10::Half>(),
                    chunk.len * h);
            } else {
                rpu_launch_linear_ddr_kernel(
                    slot, proj_w_, proj_out_chunk, proj_b_,
                    /*has_bias=*/true, /*partition=*/1);
            }
        }
    }

    // ========================================================================
    // plan_kv_first_chunks — kv_first_chunk_plan_fn.
    //
    // SigLIP keeps Phase 1 (KV-insert) and Phase 2 (compute) on the SAME chunk
    // size — return the compute_plan unchanged (mirror image_flow, which also
    // disables the kv_cs doubling). The auto chunk-size scan finds the largest
    // comp_cs that fits SPM; a single kv_cs simplifies SPM budgeting and avoids
    // a two-phase estimator/allocator mismatch.
    //
    // The per-image
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
    // subclass_chunk_size_valid — required kernel-validity hook for the auto
    // chunk-size scan. The base default returns true (only the
    // SPM budget is checked), but the SigLIP SDPA imposes real tile/structural
    // constraints the budget predicate does not capture.
    //
    // head_dim() = 80 (padded), NOT orig_head_dim_ = 72: sdpa_is_valid_chunk_size
    // hard-requires head_dim % 16 == 0; 72%16≠0 would reject
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

        // The 512 ceiling applies to the single-image auto scan. Packed multi-image
        // single-chunk plans are instead constrained by the exact stacked SPM budget
        // check below.
        if (image_batch_count_ <= 1 && cs > 512) return false;

        // SPM budget gate. The
        // framework's auto-scan budget check (estimate_temporary_total) models
        // KvInsert and Compute scopes as ALIASING (LayerWide + max(KvInsert,
        // Compute)). The real bump allocator (rpu_spm_allocator) does NOT free the
        // Phase-1 KvInsert temps (q_kv/k/v) before Phase-2 Compute allocates — the
        // two scopes stack. Reject cs whose actual stacked peak
        // (LayerWide + KvInsert + Compute) exceeds the usable
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
    at::Tensor run_packed_patch_embed_core(const std::vector<at::Tensor>& images) {
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
        const int nc = hyvla_patch_embed_cores();
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
            // byte-identical.
            rpu_ddr_flush_force(
                const_cast<c10::Half*>(images[i].data_ptr<c10::Half>()));
            fused_patch_embedding(
                images[i], patch_emb_weight_, patch_emb_pos_emb_,
                pe_kh_, pe_kw_, pe_cin_orig_, pe_cin_padded_, pe_cout_,
                pe_strideh_, pe_stridew_,
                hidden, /*seq_off=*/i * npp, /*img_slot=*/(int)i, offsets);
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

    // 4D [N,C,H,W] path: slice into contiguous [1,C,H,W] images and pack through
    // the shared core. A dim-0 slice is already contiguous.
    at::Tensor run_packed_patch_embed(const at::Tensor& input) {
        int64_t N = input.size(0);
        std::vector<at::Tensor> images;
        images.reserve(N);
        for (int64_t i = 0; i < N; ++i) {
            images.push_back(input.slice(0, i, i + 1).contiguous());
        }
        return run_packed_patch_embed_core(images);
    }

    // All-RPU list path (forward_multi): the N camera images arrive as distinct
    // [1,C,H,W] tensors — no torch.cat upstream, no slice here.
    at::Tensor run_packed_patch_embed_list(at::TensorList images) {
        std::vector<at::Tensor> imgs(images.begin(), images.end());
        return run_packed_patch_embed_core(imgs);
    }

private:
    // Packed attention uses a 2D block-diagonal mask cached by (seq_len, N).
    // Each image attends only to its own columns. Caching preserves the fixed DMA
    // source address across Graph replay, including when mask preparation pads it.
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
        m = m.to(at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1))
             .contiguous();
        packed_masks_.emplace(
            key, sdpa_prepare_mask(c10::optional<at::Tensor>(m), /*is_causal=*/false,
                                   seq_len, seq_len, sdpa_stable_mask_cache(),
                                   n_img));
        return packed_masks_.at(key);
    }

    // ----- Model state -----
    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerBiasNorm> layer_bias_norm_;

    // Global weights
    at::Tensor proj_w_, proj_b_;
    // 项 A：proj1 之前把行序换成 member-major，供 tower 外 merger 的组轴算子用。
    const bool merger_member_major_ = hyvla_fused_merger_on();

    // Intermediate DDR staging buffer (post-LN → DDR → projector). Shape-gated
    // reuse is SAFE here because this tensor is a class member that never
    // escapes to Python — no caller accumulates references to it. The
    // per-forward spm_copy_ddr_dma writes to this stable DDR address, and the
    // projector Linear (still inside the same main graph batch) reads from
    // it immediately. Shape-keyed map: each (seq_len, hidden_size) gets its
    // own slot so multi-shape REPLAY keeps every shape's DMA address valid.
    std::map<std::pair<int64_t, int64_t>, at::Tensor> temp_ddr_slots_;

    // Final projector output is allocated fresh per forward so caller-held
    // references do not alias later outputs.
    at::Tensor projector_output_;

    // Model config
    double eps_ = 1e-6;
    int64_t projection_dim_ = 0;
    int64_t orig_head_dim_ = 0;  // SigLIP head_dim=72, padded to 80

    // Site5 (KV_FIRST state) — mirror GemmaModel. Stages Q across the two phases
    // (Phase 1 scatters Q→DDR, Phase 2 gathers DDR→SPM "q_comp"). local_q_heads_ =
    // num_q_heads()/NUM_CORES (set in set_weights). seq_len_ = this forward's
    // full sequence length (the SDPA kv_seq_len, decoupled from per-chunk seq_q).
    //
    // Each {seq_len, q_width} shape keeps a never-reallocated DDR slot because the
    // Q scatter/gather DMA address is fixed at Graph capture. Including q_width
    // also preserves validity when a reused handle loads different weight geometry.
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

    // Number of images packed along sequence this forward. One selects the
    // single-image path; more than one selects block-masked packed attention.
    int64_t image_batch_count_ = 1;

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

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

int64_t rpu_hyvit2_create() {
    return HYViT2Registry::create();
}

void rpu_hyvit2_destroy(int64_t handle) {
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
    int64_t projection_dim, double eps)
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
        projection_dim, eps,
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
        projection_dim, eps,
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
    at::TensorList v_caches_list)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return HYViT2Registry::get(handle, "rpu_hyvit2")->forward(
        input, k_caches, v_caches);
}

at::Tensor rpu_hyvit2_forward_packed(
    int64_t handle,
    const at::Tensor& hidden,
    int64_t image_batch_count,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return HYViT2Registry::get(handle, "rpu_hyvit2")->forward_packed(
        hidden, image_batch_count, k_caches, v_caches);
}

at::Tensor rpu_hyvit2_forward_multi(
    int64_t handle,
    at::TensorList images,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return HYViT2Registry::get(handle, "rpu_hyvit2")->forward_multi(
        images, k_caches, v_caches);
}
