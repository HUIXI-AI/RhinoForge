// rpu_hyvla_expert_model.cpp — Hy-Embodied-0.5-VLA 的 action expert denoise 解码器。
// 设计与易错点见 rpu_hyvla_expert_model.h 的文件头。
//
// 层体是 rpu_hyvla_vlm_model.cpp 的**单塔版**: 同样的 rope→QK-norm 换序, 但去掉了
// 全部孪生分支 (双 GEMM / 行掩码 merge / act 侧 Temp 槽 / 掩码 Persistent 槽) ——
// expert 的 modality_mask 恒 True，参考模型只走 `_v` 一条路。
// 因此本文件不 override declare_buffers: 用到的槽基类全都声明过。
#include "rpu_hyvla_expert_model.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"

#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_FAST_REPLAY —— **逐座 opt-in** 的 fast-replay，默认 OFF。
//
// 打开后 REPLAY 期不再重走该座的 op stream，省掉重复 emit 的 host 开销。
//
// ⚠️ 契约：**body 不得有 kernel 会读的
// per-call host 副作用**。合法出口只有：position_ids 的稳定 keepalive、
// dynamic_config 拷进稳定 DDR 槽的显式 2D mask、`hidden_in_src_base_` 与
// op 序言里 `*_mutable(&member)` 的可变基址。
//
// 逐座开关用于隔离各子系统的 fast-replay 正确性边界。
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
// 打开后在 REPLAY（且权重 clean）跳过 persistent preload 回调与 `preload_fn` 的
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
// RPU_HY_VLA_ACTION_MLP_MC selects eight cores by default; 0/off/false
// selects one. Multi-core mode uses a row-partitioned projection, applies bias
// after reduction, and requires the matching Python weight layout. It changes
// reduction order and therefore requires numerical validation.
// ─────────────────────────────────────────────────────────────────────────────
static int hyvla_action_mlp_cores() {
    static const int nc = [] () -> int {
        const char* e = std::getenv("RPU_HY_VLA_ACTION_MLP_MC");
        if (!e || !*e) return v3::NUM_CORES;   // 本 helper 在 v3 外 ⇒ 要限定
        const std::string v(e);
        // ⚠️ 只有 0/off/false 关 —— **`=1` 是"开"**，与本仓库其余开关同约定
        //    （`_patch_embed_cores()` 也是这样）。值 1 仍表示多核路径。
        if (v == "0" || v == "off" || v == "false") return 1;
        return v3::NUM_CORES;
    }();
    return nc;
}

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_MASK_ONCE —— 同样**逐座 opt-in**，默认 OFF。
//
// 显式 2D mask 在整个 forward 里逐层恒定（形状 `(seq_q, kv_seq_len)` 由
// dynamic_config 固定，单 chunk），但 `sdpa_mask` 槽默认是**相位混叠**的 Temp
// （相位窗口 `[3,5]`），别的 Temp 会压在同一段 SPM 上 ⇒ 每层 SDPA 之前都得
// 把它从 DDR 重新广播进 SPM 一次。这是**槽的相位窗口**带来的，
// 不是 mask 本身要变。
//
// 打开后做两件事：① 把 `sdpa_mask` 的相位窗口撑满整个层体 ⇒ 分配器不再让别的
// Temp 混叠它；② 只在 `layer_idx == 0` 发一次 DMA。
// SDPA 读取的 SPM 字节保持不变。
//
// ⚠️ 为什么是"每个 body 一次"而不是"整图一次"：expert 的 `pre_layers_fn` 临时量
// （`enc_h_spm`/`emb_core0_spm`/… 相 `[0,0]`）与 mask 的相位窗口不重叠，分配器
// **允许**它们共用地址，而 pre-hook 在每个 body 开头就会写它们。每 body 重发一次
// 且不依赖任何跨 body 的存活假设。
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

// `sdpa_mask` 的相位窗口撑满层体 `[1,8]` 并落到 LayerWide —— LayerWide 与
// KvInsert/Compute **恒冲突**、与同域 LayerWide 按相位重叠判冲突
// （`rpu_spm_allocator.cpp::has_conflict`），撑满即等于"独占一块，谁也压不到"。
// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_SILU_MUL —— 逐座 opt-in，默认 OFF。
//
// Fuses SwiGLU SILU and multiply. The up projection must be emitted first;
// the fused rounding path requires numerical validation.
// ─────────────────────────────────────────────────────────────────────────────
static bool hyvla_silu_mul_on(const char* who) {
    static const std::string v = [] {
        const char* e = std::getenv("RPU_HY_VLA_SILU_MUL");
        return std::string(e ? e : "");
    }();
    if (v.empty() || v == "0" || v == "off" || v == "false") return false;
    if (v == "1" || v == "on" || v == "true") return true;
    return v.find(who) != std::string::npos;
}

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_KVPAD16 —— 逐座 opt-in，默认 OFF。
//
// Rounds K/V SPM row capacity to a multiple of 16 and passes that physical
// capacity to the launcher. Padded cache rows remain outside logical
// `kv_seq_len` and are never read.
// ─────────────────────────────────────────────────────────────────────────────
static bool hyvla_kvpad16_on(const char* who) {
    static const std::string v = [] {
        const char* e = std::getenv("RPU_HY_VLA_KVPAD16");
        return std::string(e ? e : "");
    }();
    if (v.empty() || v == "0" || v == "off" || v == "false") return false;
    if (v == "1" || v == "on" || v == "true") return true;
    return v.find(who) != std::string::npos;
}

// 把 k/v 槽撑到 `rows` 行（只增不减）。返回实际保证的行容量。
static int64_t hyvla_grow_kv_slots(std::vector<v3::BufferDecl>& decls,
                                   int64_t rows, int64_t local_kv_elems) {
    const int64_t need = Align(rows * local_kv_elems * 2, 256);
    for (auto& b : decls)
        if (b.name && (std::strcmp(b.name, "k") == 0 || std::strcmp(b.name, "v") == 0))
            b.size = std::max(b.size, need);
    return rows;
}

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_RMSNORM_PAD16 —— 逐座 opt-in，默认 OFF。
//
// Pads RMSNorm storage to the required 16-row physical capacity. RMSNorm is
// row-independent and downstream operators consume only logical rows.
// ─────────────────────────────────────────────────────────────────────────────
// `RPU_HY_VLA_PARTIAL_ROPE`：把 `llama_rope` 换成 `partial_mrope`。
//
// 取值："expert" / "vlm_vision" / "1" / 空（默认 OFF）。逐塔 opt-in。
//
// `partial_mrope` consumes per-token [seq, rotary_dim/2] tables. Existing
// model-owned cos/sin tables are used with `cos_sin_start` as the row offset.
//
static bool hyvla_partial_rope_on(const char* who) {
    static const std::string v = [] {
        const char* e = std::getenv("RPU_HY_VLA_PARTIAL_ROPE");
        return std::string(e ? e : "");
    }();
    if (v.empty() || v == "0" || v == "off" || v == "false") return false;
    if (v == "1" || v == "on" || v == "true") return true;
    return v.find(who) != std::string::npos;
}

static bool hyvla_rmsnorm_pad16_on(const char* who) {
    static const std::string v = [] {
        const char* e = std::getenv("RPU_HY_VLA_RMSNORM_PAD16");
        return std::string(e ? e : "");
    }();
    if (v.empty() || v == "0" || v == "off" || v == "false") return false;
    if (v == "1" || v == "on" || v == "true") return true;
    return v.find(who) != std::string::npos;
}

// 把指定的槽撑到至少 `need` 字节（只增不减），返回是否全部命中。
static bool hyvla_grow_slot(std::vector<v3::BufferDecl>& decls,
                            const char* name, int64_t need) {
    for (auto& b : decls)
        if (b.name && std::strcmp(b.name, name) == 0) {
            b.size = std::max(b.size, Align(need, (int64_t)256));
            return true;
        }
    return false;
}

static void hyvla_widen_sdpa_mask_slot(std::vector<v3::BufferDecl>& decls) {
    for (auto& b : decls) {
        if (b.name && std::strcmp(b.name, "sdpa_mask") == 0) {
            b.phase_start = 1;
            b.phase_end   = 8;
            b.scope       = v3::BufferScope::LayerWide;
        }
    }
}

namespace v3 {

HyVlaExpertModel::HyVlaExpertModel()  = default;
HyVlaExpertModel::~HyVlaExpertModel() = default;

// ─────────────────────────────────────────────────────────────────────────────
// set_weights_expert — 基类委托 + 子类影子
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::set_weights_expert(
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps,
    int64_t chunk_size,
    at::TensorList q_ws_list, at::TensorList k_ws_list,
    at::TensorList v_ws_list, at::TensorList o_ws_list,
    at::TensorList gate_ws_list, at::TensorList up_ws_list,
    at::TensorList down_ws_list)
{
    TORCH_CHECK(chunk_size > 0,
                "hyvla_expert set_weights: chunk_size must be > 0, got ", chunk_size);
    // 校验/状态提交全部委托基类 (含 q/k norm 全套检查 + W8A16 scale 的
    // 全给/全空、dtype、rank、numel==N 校验)。Hy-VLA attention_bias=false
    // ⇒ QKV bias 三个列表恒空; MLP 固定 SwiGLU ⇒ use_silu=true。
    CausalDecoderModel::set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list, q_norm_list, k_norm_list,
        input_norm_list, post_norm_list, gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
        eps, /*use_silu=*/true,
        /*mrope_section=*/{}, /*deepstack_lang_layers=*/{},
        /*q_bias=*/{}, /*k_bias=*/{}, /*v_bias=*/{},
        q_ws_list, k_ws_list, v_ws_list, o_ws_list,
        gate_ws_list, up_ws_list, down_ws_list);

    TORCH_CHECK(has_qk_norm_,
                "hyvla_expert set_weights: q/k_norm lists must be non-empty "
                "(Hy-VLA is Qwen3-style). ⚠️ 且必须是 **VLM 层** 的 "
                "query/key_layernorm —— 两塔共享 VLM 那一份，expert 自己的同名 "
                "张量不参与计算。");

    // 影子副本 — build_layer_subgraph 要逐层权重, 而基类 layer_weights_ 是 private。
    // 只复制 Tensor 句柄 (引用计数), 数据仍由基类持有的同一 DDR 张量拥有。
    const int64_t N = static_cast<int64_t>(q_w_list.size());
    const bool q8 = !q_ws_list.empty();
    layers_.clear();
    layers_.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
        layers_.push_back({q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                           gate_list[i], up_list[i], down_list[i],
                           q8 ? q_ws_list[i]    : at::Tensor(),
                           q8 ? k_ws_list[i]    : at::Tensor(),
                           q8 ? v_ws_list[i]    : at::Tensor(),
                           q8 ? o_ws_list[i]    : at::Tensor(),
                           q8 ? gate_ws_list[i] : at::Tensor(),
                           q8 ? up_ws_list[i]   : at::Tensor(),
                           q8 ? down_ws_list[i] : at::Tensor()});
    }
    cos_ref_ = cos;
    sin_ref_ = sin;
    eps_expert_ = eps;
    expert_chunk_size_ = chunk_size;

    invalidate_model_state();   // set_weights 的末条非空语句。
}

// ─────────────────────────────────────────────────────────────────────────────
// set_action_weights — 打开 in-graph unroll 模式
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::set_action_weights(
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias, const at::Tensor& time_all,
    int64_t action_dim, int64_t num_steps)
{
    TORCH_CHECK(hidden_size() > 0 && expert_chunk_size_ > 0,
                "hyvla_expert set_action_weights 必须在 set_weights_expert 之后调用");
    TORCH_CHECK(action_dim > 0 && num_steps > 0,
                "action_dim/num_steps 必须为正, 得到 ", action_dim, "/", num_steps);
    const int64_t h = hidden_size();
    action_dim_     = action_dim;
    action_dim_pad_ = ((action_dim + 15) / 16) * 16;
    // 图内 Euler 是 `x_t_spm[cs,pad] += dt·v_t[cs,pad]` 的逐元素加 —— x 与 v 的
    // 行宽必须相同, 所以 encoder 的 K 与 out_proj 的 N 都取 action_dim_pad。
    TORCH_CHECK(action_dim_pad_ % 16 == 0, "action_dim_pad 必须是 16 的倍数");
    TORCH_CHECK(mo_bias.dim() == 1 && mo_bias.size(0) == h
             && mo_bias.scalar_type() == at::kHalf
             && mo_bias.device().type() == at::kPrivateUse1,
                "mo_bias 必须是 [hidden] fp16 RPU");
    TORCH_CHECK(op_bias.dim() == 1 && op_bias.size(0) == action_dim_pad_
             && op_bias.scalar_type() == at::kHalf
             && op_bias.device().type() == at::kPrivateUse1,
                "op_bias 必须是 [action_dim_pad] fp16 RPU");
    TORCH_CHECK(time_all.dim() == 2 && time_all.size(0) == num_steps
             && time_all.size(1) == h && time_all.scalar_type() == at::kHalf
             && time_all.is_contiguous()
             && time_all.device().type() == at::kPrivateUse1,
                "time_all 必须是 [num_steps, hidden] fp16 contig RPU");
    wc_ = wc; mo_ = mo; mo_bias_ = mo_bias;
    op_ = op; op_bias_ = op_bias; time_all_ = time_all;
    num_steps_ = num_steps;
    // 稳定 staging DDR: pre-hook 把 encoder 的行 1:cs 写这里, 层 0 的
    // emit_layer_input_dma 经 FMB hidden_in_src_base_ 读回 (固定 src —— data_ptr 稳定)。
    emb_stage_ = at::empty({1, expert_chunk_size_, h},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    unroll_mode_ = true;
    invalidate_model_state();
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers / static_config — 只在 unroll 下追加
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BufferDecl> HyVlaExpertModel::declare_buffers(const LayoutContext& ctx) {
    auto d = CausalDecoderModel::declare_buffers(ctx);
    if (hyvla_mask_once_on("expert")) hyvla_widen_sdpa_mask_slot(d);
    // KVPAD16（见文件头）：把 k/v 槽撑到 16 的倍数行，KV-insert 一次 v16 打完。
    kv_pad_rows_ = 0;
    if (hyvla_kvpad16_on("expert")) {
        const int64_t local_kv = num_kv_heads() * head_dim() / attn_tp();
        kv_pad_rows_ = hyvla_grow_kv_slots(
            d, Align(ctx.chunk_size, (int64_t)16), local_kv);
    }
    // RMSNORM_PAD16（见文件头）：把 rmsnorm 读写的四个槽撑到 16 的倍数行。
    rmsnorm_pad_rows_ = 0;
    rmsnorm_pad_qrows_ = 0;
    if (hyvla_rmsnorm_pad16_on("expert")) {
        const int64_t hh = hidden_size(), hd_ = head_dim();
        const int64_t lq = num_q_heads() / attn_tp();
        rmsnorm_pad_rows_  = Align(ctx.chunk_size, (int64_t)16);
        rmsnorm_pad_qrows_ = Align(ctx.chunk_size * lq, (int64_t)16);
        const bool ok =
            hyvla_grow_slot(d, "residual1",  rmsnorm_pad_rows_  * hh  * 2) &&
            hyvla_grow_slot(d, "input_norm", rmsnorm_pad_rows_  * hh  * 2) &&
            hyvla_grow_slot(d, "q",          rmsnorm_pad_qrows_ * hd_ * 2) &&
            hyvla_grow_slot(d, "output",     rmsnorm_pad_qrows_ * hd_ * 2);
        // 四个槽名一个都不能漏 —— 漏了就是**写越界**，不是慢一点。
        TORCH_CHECK(ok, "hyvla RMSNORM_PAD16: 基类槽名对不上（residual1/input_norm/q/output）");
    }
    if (!unroll_mode_) return d;      // 单步路径: SPM 布局逐字节不变
    const int64_t cs = ctx.chunk_size, h = hidden_size(), np = action_dim_pad_;
    constexpr int DW = 2;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
    using SC = StorageClass;
    constexpr BufferScope ALL = BufferScope::LayerWide;
    // x_t_spm 要横跨整个 body (相 0..9) 才能被 post-hook 的图内 Euler 读到 ⇒ 独立槽,
    // 不与 residual1 (1..8) 混叠。pre-hook 的临时量活在相 0..0, 层 0 之前就死了,
    // 分配器可以安全复用它们的地址。
    d.push_back({"x_t_spm",       A(cs * np * DW), 0, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"enc_h_spm",     A(cs * h * DW),  0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"emb_core0_spm", A(cs * h * DW),  0, 0, SC::Temp, 0, nullptr, ALL});
    // A6 走多核时的部分和槽（见文件头 ACTION_MLP_MC）。相位窗口同为 [0,0] ⇒ 与
    // 层体的 Temp（相 1..8）不重叠，分配器可以复用同一段地址，净 SPM 代价 ~0。
    if (hyvla_action_mlp_cores() > 1)
        d.push_back({"mo_part_spm", A(cs * h * DW), 0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"time_spm",      A(h * DW),       0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"mo_b_spm",      A(h * DW),       0, 0, SC::Temp, 0, nullptr, ALL});
    // post-hook 的产物落在相 8..9 —— **不得**与 residual1 (1..8) 混叠, post-hook 要读它。
    d.push_back({"v_t_core0_spm", A(cs * np * DW), 8, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"op_b_spm",      A(np * DW),      8, 9, SC::Temp, 0, nullptr, ALL});
    return d;
}

ModelStaticConfig HyVlaExpertModel::static_config() {
    ModelStaticConfig cfg = CausalDecoderModel::static_config();
    cfg.fast_replay_skip_layer_loop = hyvla_fast_replay_on("expert");
    cfg.fast_replay_skip_preload = hyvla_fast_replay_preload_on("expert");
    if (!unroll_mode_) return cfg;
    cfg.pre_layers_fn  = reinterpret_cast<void (FusedModelBase::*)()>(
        &HyVlaExpertModel::emit_pre_layers_body);
    cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
        &HyVlaExpertModel::emit_post_layers_body);
    cfg.body_iterations = num_steps_;
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// unroll 的 pre/post hook
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::emit_pre_layers_body() {
    const int64_t h = hidden_size(), cs = expert_chunk_size_, np = action_dim_pad_;
    const int64_t bit = ctx().body_iter;

    // [A1] x0 DDR → x_t_spm (MUTABLE), **只在第 0 步**。第 1..N-1 步直接读上一步
    //      post-hook 留在 x_t_spm 里的 Euler 结果 (槽的生命期是 [0,9], 跨相存活)。
    if (bit == 0) {
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &x_t_src_base_, /*src_offset_bytes=*/0, /*num_elements=*/cs * np,
            addr(0, "x_t_spm"), /*num_cores=*/NUM_CORES);
    } else if (hyvla_action_mlp_cores() > 1) {
        // A4 是列并行，每个核都需要完整的 x_t；图内 Euler 只在核 0
        // 更新该值，因此从 x_traj[bit-1] 广播回所有核。
        // 与 A7→emb_stage_→层 0 那条 DDR 往返同型，不引入新的存活假设。
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &x_traj_dst_base_, /*src_offset_bytes=*/(bit - 1) * cs * np * DWIDTH,
            /*num_elements=*/cs * np, addr(0, "x_t_spm"), /*num_cores=*/NUM_CORES);
    }
    const int amc = hyvla_action_mlp_cores();
    // [A2] time_all[bit] → time_spm, 作为 encoder 第一个 GEMM (A4) 的 bias。
    //      FIXED src: time_all_ 是稳定的注册权重, bit 偏移在 BUILD 期烘死
    //      (每个 body 单独 emit 一次, body_iter 是编译期常量)。
    //      ⚠️ 多核档 A4 走**列并行** ⇒ 核 c 只负责输出列 [c*h/amc, (c+1)*h/amc)，
    //         bias 必须**按核切片**（scatter），不能广播 —— 广播会让每个核都加
    //         bias[0:h/amc]，是静默算错。
    if (amc == 1) {
        rpu_launch_ddr_broadcast_spm_dma(
            time_all_.data_ptr<c10::Half>() + bit * h, h, addr(0, "time_spm"),
            /*num_cores=*/1);
    } else {
        rpu_launch_ddr_scatter_spm_dma(
            time_all_.data_ptr<c10::Half>() + bit * h,
            /*elements_per_core=*/h / amc,
            /*core_stride_bytes=*/(h / amc) * DWIDTH,
            addr(0, "time_spm"), /*num_cores=*/amc);
    }
    // [A3] mo_bias → mo_b_spm (内容恒定, FIXED src)。
    //      单核档只有核 0 用得到；多核档 all-reduce 后每核都要按列加一次 bias
    //      （`1xC_NxC` 没有 num_cores 参数，8 核齐发），所以要广播满 8 核。
    rpu_launch_ddr_broadcast_spm_dma(
        mo_bias_.data_ptr<c10::Half>(), h, addr(0, "mo_b_spm"), /*num_cores=*/amc);

    // [A4] enc_h = x_t_spm · wcᵀ + time_bias  (M=cs, N=h, K=np)。
    //      ⚠️ 多核档必须**列并行**：A6 是行(K)并行，它要求核 c 手里正好是输入的
    //      第 c 段 K；列并行 A4 的输出正好提供该分片。
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "x_t_spm"), wc_, addr(0, "enc_h_spm"),
        /*M=*/cs, /*N=*/h, /*K=*/np, /*partition=*/1, /*num_cores=*/amc,
        /*bias_spm_addr=*/addr(0, "time_spm"));
    // [A5] SiLU 原地。逐元素 ⇒ 每核只处理自己那 h/amc 列。
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "enc_h_spm"), addr(0, "enc_h_spm"), cs * (h / amc), ValuOpType::SILU,
        /*is_gelu=*/false, /*num_cores=*/amc);
    // [A6] emb = silu(enc_h) · moᵀ + mo_bias → emb_core0_spm。
    //      本模型 h=1024，mo 是 2 MB。多核档把 K 切 8 份并执行 all-reduce，
    //      见文件头 ACTION_MLP_MC。
    if (amc == 1) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "enc_h_spm"), mo_, addr(0, "emb_core0_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
            /*bias_spm_addr=*/addr(0, "mo_b_spm"));
    } else {
        // A6-1 K 方向切核 ⇒ 每核一份 [cs,h] 部分和。**bias 必须留到 all-reduce 之后**。
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "enc_h_spm"), mo_, addr(0, "mo_part_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/h, /*partition=*/0, /*num_cores=*/amc,
            /*bias_spm_addr=*/0u);
        // A6-2 enc_h_spm 到这里已死 ⇒ 就地清零，当 all-reduce 的零残差
        //      （该 kernel 要求 input/residual/output 三地址互不相同）。
        rpu_launch_fill_spm_kernel(
            addr(0, "enc_h_spm"), cs * h, c10::Half(0.0f), /*num_cores=*/amc);
        // A6-3 两段式 all-reduce（M*N=cs*h 满足 %16==0 且 ≤ v2 cap ⇒ 走 v2 快路径）。
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "mo_part_spm"), addr(0, "enc_h_spm"), addr(0, "emb_core0_spm"),
            /*M=*/cs, /*N=*/h, /*input_num_cores=*/amc, /*output_num_cores=*/amc);
        // A6-4 bias 按列广播加一次（8 核齐发，各核结果一致；A7 只读核 0）。
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            addr(0, "mo_b_spm"), addr(0, "emb_core0_spm"), addr(0, "emb_core0_spm"),
            /*n=*/cs, /*c=*/h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
    }
    // [A7] emb_core0_spm[1:cs] → emb_stage_[1:cs] (src/dst 都偏一行)。行 0 保持
    //      unroll_forward 写进去的 state token 不动 —— encoder 的行 0 是 x_t 行 0
    //      (恒为 0 且不被任何东西消费) 的产物, 直接丢弃。
    rpu_launch_spm_copy_ddr_dma(
        addr(0, "emb_core0_spm") + h * DWIDTH,
        emb_stage_.data_ptr<c10::Half>() + h, (cs - 1) * h);
}

void HyVlaExpertModel::emit_post_layers_body() {
    const int64_t h = hidden_size(), cs = expert_chunk_size_, np = action_dim_pad_;
    const int64_t bit = ctx().body_iter;
    // [Z1] op_bias → op_b_spm (核 0, 内容恒定, FIXED src)。
    rpu_launch_ddr_broadcast_spm_dma(
        op_bias_.data_ptr<c10::Half>(), np, addr(0, "op_b_spm"), /*num_cores=*/1);
    // [Z2] v_t = residual1 · opᵀ + op_bias (单核)。residual1 已被末层的 final
    //      RMSNorm 原地归一化过 —— **不要再归一化一次**。
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), op_, addr(0, "v_t_core0_spm"),
        /*M=*/cs, /*N=*/np, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
        /*bias_spm_addr=*/addr(0, "op_b_spm"));
    // [Z3] 图内 Euler: x_t_spm = x_t_spm + dt·v_t (SPM 原地, 核 0)。dt 在 BUILD 烘死。
    //      行 0 (state 位) 也会被更新, 但它的 encoder 输出从不进 emb_stage_ ⇒ 死路。
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "x_t_spm"), addr(0, "v_t_core0_spm"), addr(0, "x_t_spm"),
        cs * np, ValuOpType::ADD, dt_, /*num_cores=*/1);
    // [Z4] x_t_spm → x_traj[bit] DDR (MUTABLE)。终值 = x_traj[-1]。
    rpu_launch_spm_copy_ddr_dma_mutable(
        addr(0, "x_t_spm"), &x_traj_dst_base_,
        /*dst_offset_bytes=*/bit * cs * np * 2, cs * np);
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamic_config — 基类断言单 chunk + 备一份子类可见的 PreparedMask
// ─────────────────────────────────────────────────────────────────────────────
ModelDynamicConfig HyVlaExpertModel::dynamic_config(const ChunkPlan& plan) {
    ModelDynamicConfig cfg = CausalDecoderModel::dynamic_config(plan);
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "hyvla_expert dynamic_config: explicit attention mask required "
                "(denoise 是非因果 prefix attention, prefix 块还要屏蔽 padding 行)");
    prepared_mask_ = sdpa_prepare_mask(
        ctx().attention_mask, /*is_causal=*/false,
        ctx().seq_len, ctx().position + ctx().seq_len,
        sdpa_stable_mask_cache());
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// subclass_chunk_size_valid — suffix 整块进 SDPA
// ─────────────────────────────────────────────────────────────────────────────
bool HyVlaExpertModel::subclass_chunk_size_valid(
    int64_t cs, int64_t seq_len, int64_t /*position*/) const {
    return cs >= seq_len;
}

// ─────────────────────────────────────────────────────────────────────────────
// build_layer_subgraph — 单塔层体 (基类 8 相, 步 3 换序)
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    TORCH_CHECK(!layers_.empty(),
                "hyvla_expert build_layer_subgraph: set_weights_expert must "
                "have been called");
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "hyvla_expert build_layer_subgraph: explicit 2D mask required");

    const auto& lw = layers_[layer_idx];
    const int64_t seq_len = chunk.len;
    const int64_t cos_sin_start = ctx().position + chunk.offset;
    const bool is_last_layer = (layer_idx == num_layers() - 1);
    const int64_t h   = hidden_size();
    const int64_t nq  = num_q_heads();
    const int64_t nkv = num_kv_heads();
    const int64_t hd  = head_dim();
    const int tp = attn_tp();
    const int64_t local_q  = nq / tp;
    const int64_t elems_mlp = seq_len * (intermediate_size() / NUM_CORES);

    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }

    // ── 步 1: input RMSNorm ──
    // RMSNORM_PAD16：补齐到 16 的倍数行才能走 `llama_rms_norm_v16`（见文件头）。
    // 多算的 [seq_len, pad) 行下游按 M=seq_len 消费 ⇒ 永不被读。
    const int64_t nrows  = rmsnorm_pad_rows_  ? rmsnorm_pad_rows_  : seq_len;
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "norm_w"), nrows, h, eps_expert_);

    // ── 步 2: QKV (无 bias — Hy-VLA attention_bias=false) ──
    auto qkv = [&](const at::Tensor& w, const at::Tensor& ws,
                   const char* out, int64_t n) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), w, addr(0, out),
            seq_len, n, h, /*partition=*/1, /*num_cores=*/tp,
            /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/ws);
    };
    qkv(lw.q_w, lw.q_ws, "q", nq * hd);
    qkv(lw.k_w, lw.k_ws, "k", nkv * hd);
    qkv(lw.v_w, lw.v_ws, "v", nkv * hd);

    // ── 步 3: **先 RoPE, 再 q/k head-norm**(与基类及其余 7 个 emitter 相反) ──
    // 参考模型在 apply_rotary_pos_emb 之后调用
    // query_layernorm/key_layernorm, 且两塔共用 VLM 层的那一份 (绑定方负责喂对)。
    // 与 VLM 塔 (rpu_hyvla_vlm_model.cpp 步 3) 完全同构, 只是这里没有孪生分支。
    // 换序可行的前提: 两个 kernel 都是 src/dst 分离、无隐藏状态的独立 SPM launch,
    // 且 QK-norm 的 framing (M=seq*heads, C=head_dim) 不受 RoPE 影响 —— RoPE 不改
    // [seq, heads, hd] 布局。
    c10::Half* cos_ptr = cos_ref_.data_ptr<c10::Half>();
    c10::Half* sin_ptr = sin_ref_.data_ptr<c10::Half>();

    // PARTIAL_ROPE（见文件头）：与 llama_rope 逐位相等，使用折叠后的 grid，
    // 并原样复用表和偏移。
    const bool prope = hyvla_partial_rope_on("expert");
    const int64_t local_kv_heads = nkv / tp;

    if (prope) {
        rpu_launch_partial_mrope_spm_kernel(
            addr(0, "q"), addr(0, "output"), cos_ptr, sin_ptr,
            cos_sin_start, seq_len, local_q, hd, /*rotary_dim=*/hd, tp);
    } else {
        rpu_launch_rope_spm_kernel(addr(0, "q"), addr(0, "output"),
                                   cos_ptr, sin_ptr, seq_len, local_q, hd, cos_sin_start);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "output"), addr(0, "q"),
        layer_addr(layer_idx, 0, "q_norm_w"),
        rmsnorm_pad_qrows_ ? rmsnorm_pad_qrows_ : seq_len * local_q, hd, eps_expert_);

    if (prope) {
        rpu_launch_partial_mrope_spm_kernel(
            addr(0, "k"), addr(0, "input_norm"), cos_ptr, sin_ptr,
            cos_sin_start, seq_len, local_kv_heads, hd, /*rotary_dim=*/hd, tp);
    } else {
        rpu_launch_rope_spm_kernel(addr(0, "k"), addr(0, "input_norm"),
                                   cos_ptr, sin_ptr, seq_len, local_kv_heads, hd, cos_sin_start);
    }
    // k 槽已被 KVPAD16 撑到 Align(cs,16) 行 × local_kv 元素 ⇒ 这里补齐不需要额外撑槽。
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "input_norm"), addr(0, "k"),
        layer_addr(layer_idx, 0, "k_norm_w"),
        (rmsnorm_pad_rows_ ? rmsnorm_pad_rows_ : seq_len) * local_kv_heads, hd, eps_expert_);

    // ── 步 4: KV insert (追加在 prefix 之后) + SDPA + o_proj + 残差归约 ──
    // 调用方每步前 reset_to_position(prefix_len), 所以这 51 行每步覆写同一段,
    // 等价于参考模型冻结 prefix，并在本步结束后丢弃 suffix KV。
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    rpu_launch_insert_kcache_spm_unified(
        k_cache, ctx().position + chunk.offset, addr_offset("k").value,
        seq_len, nkv, hd, tp, /*cache_batch_offset_elems=*/0,
        /*allow_non8_v16=*/false, /*allow_hybrid_v16=*/false,
        /*spm_rows=*/kv_pad_rows_);
    rpu_launch_insert_vcache_spm_unified(
        v_cache, ctx().position + chunk.offset, addr_offset("v").value,
        seq_len, nkv, hd, tp, /*cache_batch_offset_elems=*/0,
        /*allow_non8_v16=*/false, /*allow_hybrid_v16=*/false,
        /*spm_rows=*/kv_pad_rows_);
    const uint32_t mask_off = addr_offset("sdpa_mask").value;
    // mask 逐层恒定；槽撑满相位后每个 body 只需在层 0 灌一次（见文件头 MASK_ONCE）。
    if (!hyvla_mask_once_on("expert") || layer_idx == 0)
        sdpa_dma_mask_to_spm(prepared_mask_, mask_off, seq_len, chunk.kv_seq_len, tp);
    rpu_launch_sdpa_spm_dispatch(
        SdpaKernelType::FLASH_ATTN_SPM,
        k_cache, v_cache, prepared_mask_.mask_type, c10::nullopt,
        addr_offset("q").value, addr_offset("output").value,
        addr_offset("sdpa_tmp").value, mask_off,
        seq_len, nq, nkv, hd, chunk.kv_seq_len, tp, NUM_CORES);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/lw.o_ws);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
        seq_len, h, tp, NUM_CORES);

    // ── 步 5: post-attention RMSNorm ──
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "residual1"),
        layer_addr(layer_idx, 0, "post_norm_w"), nrows, h, eps_expert_);

    // ── 步 6: SwiGLU MLP + 残差归约 ──
    // 手写而非用基类 emit_mlp_pipeline: 保持与 VLM 塔同构的可读性, 且这里的
    // 槽名/核数组合与基类默认一致, 没有额外分支。
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
        seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
        /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/lw.gate_ws);
    if (!hyvla_silu_mul_on("expert"))
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "gate"), addr(0, "gate"), elems_mlp, ValuOpType::SILU,
            /*is_gelu=*/false, NUM_CORES);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), lw.up_w, addr(0, "up"),
        seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
        /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/lw.up_ws);
    if (hyvla_silu_mul_on("expert"))
        rpu_launch_silu_mul_spm_kernel(
            addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
            elems_mlp, NUM_CORES);
    else
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
            elems_mlp, ValuOpType::MUL, c10::Half(1.0f), NUM_CORES);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "gate"), lw.down_w, addr(0, "down"),
        seq_len, h, intermediate_size(), /*partition=*/0, NUM_CORES,
        /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/lw.down_ws);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
        seq_len, h, NUM_CORES, NUM_CORES);

    // ── 步 7 (末层): final RMSNorm (原地, 单份 γ) ──
    if (is_last_layer) {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "residual1"),
            addr(0, "final_norm_w"), nrows, h, eps_expert_);
    }

    if (!ctx().output_to_spm) {
        emit_layer_output_dma(layer_idx, chunk);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// step_forward — 一次 denoise step
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor HyVlaExpertModel::step_forward(
    const at::Tensor& x_emb, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len)
{
    TORCH_CHECK(!layers_.empty() && expert_chunk_size_ > 0,
                "hyvla_expert step_forward before set_weights_expert");
    const int64_t cs = expert_chunk_size_;
    const int64_t h  = hidden_size();
    TORCH_CHECK(x_emb.dim() == 3 && x_emb.size(0) == 1 && x_emb.size(1) == cs
                && x_emb.size(2) == h && x_emb.scalar_type() == at::kHalf
                && x_emb.is_contiguous()
                && x_emb.device().type() == at::kPrivateUse1,
                "hyvla_expert step_forward: x_emb must be [1,", cs, ",", h,
                "] fp16 contig RPU");
    TORCH_CHECK(attn_mask_4d.defined(),
                "hyvla_expert step_forward: explicit attention mask required");
    TORCH_CHECK(prefix_len >= 0,
                "hyvla_expert step_forward: prefix_len must be >= 0");

    // per-forward cos/sin: 覆盖影子成员, 赋值即保活过 graph capture;
    // build_layer_subgraph 读它们的 raw c10::Half*。
    TORCH_CHECK(cos.defined() && sin.defined()
                && cos.device().type() == at::kPrivateUse1
                && sin.device().type() == at::kPrivateUse1
                && cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf
                && cos.is_contiguous() && sin.is_contiguous(),
                "hyvla_expert step_forward: cos/sin must be fp16 contig RPU");
    TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes()
                && cos.size(1) == head_dim() / 2,
                "hyvla_expert step_forward: cos/sin must be [rows, head_dim/2=",
                head_dim() / 2, "], got ", cos.sizes());
    TORCH_CHECK(cos.size(0) >= prefix_len + cs,
                "hyvla_expert step_forward: cos rows (", cos.size(0),
                ") < prefix_len+cs (", prefix_len + cs, ")");
    cos_ref_ = cos;
    sin_ref_ = sin;

    // KV headroom: 7-D K 布局的 max_seq = size(1)*size(5)。
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + cs <= kc.size(1) * kc.size(5),
                    "hyvla_expert step_forward: prefix_len(", prefix_len, ")+cs(",
                    cs, ") exceeds k_cache capacity ", kc.size(1) * kc.size(5));
    }

    // suffix 必须单块。override 取 ceil16(cs) —— override 路径的 (v/16)*16 会把
    // 非 16 对齐值向下取整。RAII 恢复 (异常路径也恢复)。
    const int64_t single_cs = ((cs + 15) / 16) * 16;
    const int64_t saved_chunk_size = get_chunk_size_override();
    set_chunk_size_override(single_cs);
    auto chunk_guard = c10::make_scope_exit([this, saved_chunk_size] {
        set_chunk_size_override(saved_chunk_size);
    });

    rpu_ddr_flush_force(x_emb.data_ptr<c10::Half>());

    return CausalDecoderModel::forward(
        x_emb, k_caches, v_caches,
        std::optional<at::Tensor>(attn_mask_4d),
        /*position=*/prefix_len, /*is_causal=*/false,
        /*position_ids=*/std::nullopt, /*deepstack=*/std::nullopt);
}

// ─────────────────────────────────────────────────────────────────────────────
// unroll_forward — 一次 forward 跑完 num_steps 步 Euler
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::unroll_forward(
    const at::Tensor& x0_rpu, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches, const at::Tensor& state_emb,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& attn_mask_4d,
    at::Tensor& x_traj, double dt, int64_t prefix_len, int64_t num_steps)
{
    TORCH_CHECK(unroll_mode_, "hyvla_expert unroll_forward 之前必须 set_action_weights");
    TORCH_CHECK(num_steps == num_steps_, "num_steps(", num_steps,
                ") 必须等于 set_action_weights 的 num_steps(", num_steps_, ")");
    const int64_t cs = expert_chunk_size_, np = action_dim_pad_, h = hidden_size();

    // 节点数护栏: graph_runtime 在 batch 溢出 (上限 32768) 时**静默丢弃**。
    // 单步约 700 个节点，预留约 2 倍余量。
    TORCH_CHECK(num_steps * 1500 < 32000,
                "denoise unroll (", num_steps, " 个 body) 会逼近 32768 的 batch 上限");

    TORCH_CHECK(x0_rpu.dim() == 3 && x0_rpu.size(0) == 1 && x0_rpu.size(1) == cs
             && x0_rpu.size(2) == np && x0_rpu.scalar_type() == at::kHalf
             && x0_rpu.is_contiguous() && x0_rpu.device().type() == at::kPrivateUse1,
                "x0_rpu 必须是 [1,", cs, ",", np, "] fp16 contig RPU");
    TORCH_CHECK(x_traj.dim() == 3 && x_traj.size(0) == num_steps && x_traj.size(1) == cs
             && x_traj.size(2) == np && x_traj.scalar_type() == at::kHalf
             && x_traj.is_contiguous() && x_traj.device().type() == at::kPrivateUse1,
                "x_traj 必须是 [", num_steps, ",", cs, ",", np, "] fp16 contig RPU");
    TORCH_CHECK(state_emb.dim() == 1 && state_emb.size(0) == h
             && state_emb.scalar_type() == at::kHalf
             && state_emb.device().type() == at::kPrivateUse1,
                "state_emb 必须是 [hidden] fp16 RPU");
    TORCH_CHECK(attn_mask_4d.defined(), "hyvla_expert unroll_forward: 必须显式给 mask");

    // dt 是烘进图的 Euler 系数 (x += dt·v), N 步相同。Hy-VLA 的 t: 1→0 ⇒ dt<0。
    TORCH_CHECK(dt != 0.0 && std::isfinite(dt), "dt 必须是非零有限值");
    const c10::Half dt_half = c10::Half(static_cast<float>(dt));
    if (!dt_pinned_) { dt_ = dt_half; dt_pinned_ = true; }
    else TORCH_CHECK(dt_ == dt_half, "dt 在 replay 之间变了 (它已烘进图)");

    // cos/sin 影子 (build 期取 raw 指针); 与 step_forward 同一套校验。
    TORCH_CHECK(cos.defined() && sin.defined()
             && cos.device().type() == at::kPrivateUse1
             && sin.device().type() == at::kPrivateUse1
             && cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf
             && cos.is_contiguous() && sin.is_contiguous()
             && cos.dim() == 2 && sin.sizes() == cos.sizes()
             && cos.size(1) == head_dim() / 2 && cos.size(0) >= prefix_len + cs,
                "hyvla_expert unroll_forward: cos/sin 必须是 [>=prefix_len+cs, head_dim/2] "
                "fp16 contig RPU");
    cos_ref_ = cos;
    sin_ref_ = sin;

    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + cs <= kc.size(1) * kc.size(5),
                    "prefix_len(", prefix_len, ")+cs(", cs, ") 超出 k_cache 容量 ",
                    kc.size(1) * kc.size(5));
    }

    // emb_stage_ 行 0 = state token, N 步不变 (pre-hook 只碰行 1:cs)。每次调用写一次
    // 并 flush, 好让图内层 0 的 DMA 读到。
    state_ref_ = state_emb;
    emb_stage_[0][0].copy_(state_emb);
    rpu_ddr_flush_force(emb_stage_.data_ptr<c10::Half>());

    // 钉住本次调用的 mutable DMA 基址 + flush x0 (wrapper 自己不 flush)。
    x0_ref_ = x0_rpu; x_traj_ref_ = x_traj;
    x_t_src_base_    = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
    x_traj_dst_base_ = ::rhino_lkn::RpuGetDevAddr(x_traj.data_ptr());
    rpu_ddr_flush_force(x0_rpu.data_ptr<c10::Half>());

    // 与 step_forward 同一条 chunk 约束: suffix 必须单块。
    const int64_t single_cs = ((cs + 15) / 16) * 16;
    const int64_t saved_chunk_size = get_chunk_size_override();
    set_chunk_size_override(single_cs);
    auto chunk_guard = c10::make_scope_exit([this, saved_chunk_size] {
        set_chunk_size_override(saved_chunk_size);
    });

    // 一次 forward —— run_all_layers 内部循环 body_iterations(=num_steps_) 次,
    // pre/post hook 每次以递增的 ctx().body_iter 重新 emit。
    (void) CausalDecoderModel::forward(
        emb_stage_, k_caches, v_caches,
        std::optional<at::Tensor>(attn_mask_4d),
        /*position=*/prefix_len, /*is_causal=*/false,
        /*position_ids=*/std::nullopt, /*deepstack=*/std::nullopt);
}

}  // namespace v3

// =============================================================================
// Instance registry + C API (decls in rpu_kernel_decls.h)
// =============================================================================
using HyVlaExpertRegistry = ModelHandleRegistry<v3::HyVlaExpertModel>;

int64_t rpu_hyvla_expert_create() {
    return HyVlaExpertRegistry::create();
}

void rpu_hyvla_expert_destroy(int64_t handle) {
    HyVlaExpertRegistry::destroy(handle, "rpu_hyvla_expert_destroy");
}

void rpu_hyvla_expert_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps,
    int64_t chunk_size)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_weights")
        ->set_weights_expert(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, chunk_size);
}

void rpu_hyvla_expert_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps,
    int64_t chunk_size,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_weights_w8a16")
        ->set_weights_expert(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, chunk_size,
            q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws);
}

at::Tensor rpu_hyvla_expert_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len)
{
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_step_forward")
        ->step_forward(x_emb, k_caches, v_caches, cos, sin, attn_mask_4d, prefix_len);
}

void rpu_hyvla_expert_set_action_weights(
    int64_t handle,
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias, const at::Tensor& time_all,
    int64_t action_dim, int64_t num_steps)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_action_weights")
        ->set_action_weights(wc, mo, mo_bias, op, op_bias, time_all,
                             action_dim, num_steps);
}

void rpu_hyvla_expert_unroll_forward(
    int64_t handle, const at::Tensor& x0_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& state_emb, const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, at::Tensor x_traj,
    double dt, int64_t prefix_len, int64_t num_steps)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_unroll_forward")
        ->unroll_forward(x0_rpu, k_caches, v_caches, state_emb, cos, sin,
                         attn_mask_4d, x_traj, dt, prefix_len, num_steps);
}
