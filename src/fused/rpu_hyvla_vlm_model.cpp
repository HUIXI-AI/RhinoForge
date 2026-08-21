// rpu_hyvla_vlm_model.cpp — Hy-Embodied-0.5-VLA 的 VLM prefill fused 解码器。
// 派生自 rpu_halo_action_expert_model.cpp (同为 MoT 双权重 + 常量行掩码)。
//
// 继承 v3::CausalDecoderModel, 每层做"双算 + 行掩码合并": text 行结果与 *_v 孪生
// 权重结果按常量行掩码逐元素混合 (out = x_t*text + x_v*vis)。掩码常量化的前提是
// 一次 prefill 内 prefix 的 modality 布局固定; 掩码在 set_moe_weights 时按宽度展开
// 为 DDR 常量并经 Persistent SPM 槽 preload。
//
// 与 HALO 的实质差异只有**步 3** (rope→QK-norm 换序 + norm 两塔共享), 其余为参数
// 化差异 (无 QKV bias、final norm 单份、eps=1e-5)。凡注释提到 HALO 处均为共享机制对照。
#include "rpu_hyvla_vlm_model.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"

#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <cstring>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_FAST_REPLAY —— **逐座 opt-in** 的 fast-replay，默认 OFF。
// 说明与契约见 `rpu_hyvit2_vision_model.cpp` 里同名函数的注释（三座各保留一份，
// 避免为单个开关增加共享接口）。
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
// persistent SPM，逐位不变。框架契约见 `fused_model_base.cpp` 的
// `fast_replay_skip_preload` 注释。
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
// 显式 2D mask 逐层恒定，但 `sdpa_mask` 槽默认相位混叠 ⇒ 每层都要重灌一次。
// 完整机理与"为什么按 body 而不是按整图"见
// `rpu_hyvla_expert_model.cpp` 里同名函数的注释（三座各一份，避免为一个开关
// 往共享头文件里塞东西）。**逐位等价**。
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

// `RPU_HY_VLA_PARTIAL_ROPE` uses `who="vlm"` here. The selector semantics
// and numerical-equivalence conditions are documented beside the matching
// expert helper. Keep a local static copy to avoid shared-header state.
static bool hyvla_partial_rope_on(const char* who) {
    static const std::string v = [] {
        const char* e = std::getenv("RPU_HY_VLA_PARTIAL_ROPE");
        return std::string(e ? e : "");
    }();
    if (v.empty() || v == "0" || v == "off" || v == "false") return false;
    if (v == "1" || v == "on" || v == "true") return true;
    return v.find(who) != std::string::npos;
}

// RPU_HY_VLA_MOT_NORM_NOMERGE skips two row merges: the input-norm merge
// before QKV and the post-norm merge before the MLP. Text rows consume the
// text norm and action rows consume the action norm; the later 0/1 row mask
// discards the other branch. This is bitwise equivalent while both branch
// values remain finite, which RMSNorm guarantees for zero-padding rows.
//
// Skipping a merge extends the matching action-norm SPM lifetime by one phase,
// so QKV and MLP can be selected independently:
//     off / 0        keep both merges
//     qkv            skip only the input-norm merge
//     mlp            skip only the post-norm merge
//     on / 1 / both  skip both merges (default)
// The SPM allocator rejects a selection that exceeds the active profile.
enum : unsigned { MOT_NOMERGE_QKV = 1u, MOT_NOMERGE_MLP = 2u };

static unsigned hyvla_mot_norm_nomerge_mask() {
    static const unsigned m = [] () -> unsigned {
        const char* e = std::getenv("RPU_HY_VLA_MOT_NORM_NOMERGE");
        if (!e || !*e) return MOT_NOMERGE_QKV | MOT_NOMERGE_MLP;   // 默认两处都省
        const std::string v(e);
        if (v == "0" || v == "off" || v == "false") return 0u;
        if (v == "1" || v == "on" || v == "true" || v == "both")
            return MOT_NOMERGE_QKV | MOT_NOMERGE_MLP;
        unsigned r = 0;
        if (v.find("qkv") != std::string::npos) r |= MOT_NOMERGE_QKV;
        if (v.find("mlp") != std::string::npos) r |= MOT_NOMERGE_MLP;
        return r;
    }();
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_SILU_MUL —— 逐座 opt-in，默认 OFF。机理与等价条件见
// `rpu_hyvla_expert_model.cpp` 里同名函数的注释（三座各一份）。
// 本座是 MoT 双分支 ⇒ 每层两次 SwiGLU，合并后每层省 2 次 launch。
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

ModelStaticConfig HyVlaVlmModel::static_config() {
    ModelStaticConfig cfg = CausalDecoderModel::static_config();
    cfg.fast_replay_skip_layer_loop = hyvla_fast_replay_on("vlm");
    cfg.fast_replay_skip_preload = hyvla_fast_replay_preload_on("vlm");
    return cfg;
}

HyVlaVlmModel::HyVlaVlmModel()  = default;
HyVlaVlmModel::~HyVlaVlmModel() = default;

// ─────────────────────────────────────────────────────────────────────────────
// set_weights_hyvla — text 主权重 (基类委托 + 子类影子)
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaVlmModel::set_weights_hyvla(
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::TensorList q_bias_list, at::TensorList k_bias_list,
    at::TensorList v_bias_list,
    at::TensorList q_ws_list, at::TensorList k_ws_list,
    at::TensorList v_ws_list, at::TensorList o_ws_list,
    at::TensorList gate_ws_list, at::TensorList up_ws_list,
    at::TensorList down_ws_list)
{
    // Hy-VLA 的 MLP 固定 SwiGLU; 手写 MLP 相 (步 6) 写死 SILU, 不接受
    // use_silu=false 的静默错配。
    TORCH_CHECK(use_silu,
                "hyvla set_weights: use_silu must be true (Hy-VLA MLP is SwiGLU)");
    // 校验/状态提交全部委托基类 (含 q/k norm + bias 全套检查)。
    CausalDecoderModel::set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list, q_norm_list, k_norm_list,
        input_norm_list, post_norm_list, gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
        eps, use_silu,
        /*mrope_section=*/{}, /*deepstack_lang_layers=*/{},
        q_bias_list, k_bias_list, v_bias_list,
        q_ws_list, k_ws_list, v_ws_list, o_ws_list,
        gate_ws_list, up_ws_list, down_ws_list);

    // 影子副本 — build_layer_subgraph 的双算需要逐层 text 权重, 而基类
    // layer_weights_/cos_/sin_/eps_ 都是 private。这里只复制 Tensor 句柄
    // (引用计数), 数据所有权与生命周期仍由基类持有的同一 DDR 张量保证。
    const int64_t N = static_cast<int64_t>(q_w_list.size());
    text_layers_.clear();
    text_layers_.reserve(N);
    const bool bias = (q_bias_list.size() > 0);
    const bool q8   = !q_ws_list.empty();
    for (int64_t i = 0; i < N; ++i) {
        text_layers_.push_back({
            q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
            q_norm_list[i], k_norm_list[i],
            input_norm_list[i], post_norm_list[i],
            gate_list[i], up_list[i], down_list[i],
            bias ? q_bias_list[i] : at::Tensor(),
            bias ? k_bias_list[i] : at::Tensor(),
            bias ? v_bias_list[i] : at::Tensor(),
            q8 ? q_ws_list[i]    : at::Tensor(),
            q8 ? k_ws_list[i]    : at::Tensor(),
            q8 ? v_ws_list[i]    : at::Tensor(),
            q8 ? o_ws_list[i]    : at::Tensor(),
            q8 ? gate_ws_list[i] : at::Tensor(),
            q8 ? up_ws_list[i]   : at::Tensor(),
            q8 ? down_ws_list[i] : at::Tensor(),
        });
    }
    cos_hyvla_ = cos;
    sin_hyvla_ = sin;
    eps_hyvla_ = eps;
    final_norm_text_ = final_norm_w;   // set_moe_weights 的同一性断言要用

    invalidate_model_state();   // 影子提交后再失效一次，保持末语句语义。
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamic_config — 基类断言单 chunk + 准备其 private mask; 再备一份子类可见的
// PreparedMask 供步 4 的 sdpa_dma_mask_to_spm/SDPA 使用 (每 forward 一次)。
// ─────────────────────────────────────────────────────────────────────────────
ModelDynamicConfig HyVlaVlmModel::dynamic_config(const ChunkPlan& plan) {
    ModelDynamicConfig cfg = CausalDecoderModel::dynamic_config(plan);
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "hyvla dynamic_config: explicit attention mask required");
    prepared_mask_hyvla_ = sdpa_prepare_mask(
        ctx().attention_mask, /*is_causal=*/false,
        ctx().seq_len, ctx().position + ctx().seq_len,
        sdpa_stable_mask_cache());
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// subclass_chunk_size_valid — prefill 强制单块语义
//
// 整个 prefix 必须一次进 MASK_2D 非因果 SDPA。Hy-VLA seq=192 > 176 ⇒
// tile_m_v16 封顶 8 → tile_m=128 → grid_dim_x=2；16/4 GQA 在 attn_tp=4 下
// gqa_group_size=4 ⇒ product=8，**正好压在** keeper 的 product<=8 上限。
// 这里 override 是因为行掩码与 KV merge 按整块设计：
// 单块 (cs>=seq_len ⇒ num_chunks==1) 合法，多块非法。
// ⚠️ 更长的 prefix 会把 grid 顶过上限 —— 届时要改成分块 + 掩码按 chunk.offset
// 取行切片，不是放宽这里。
// ─────────────────────────────────────────────────────────────────────────────
bool HyVlaVlmModel::subclass_chunk_size_valid(
    int64_t cs, int64_t seq_len, int64_t /*position*/) const {
    return cs >= seq_len;
}

// ─────────────────────────────────────────────────────────────────────────────
// set_moe_weights — vision (*_v) 孪生权重 + 行掩码绑定
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaVlmModel::set_moe_weights(
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList q_bias_list, at::TensorList k_bias_list,
    at::TensorList v_bias_list,
    const at::Tensor& final_norm_moe_w,
    const at::Tensor& text_row_mask,
    int64_t chunk_size,
    at::TensorList q_ws_list, at::TensorList k_ws_list,
    at::TensorList v_ws_list, at::TensorList o_ws_list,
    at::TensorList gate_ws_list, at::TensorList up_ws_list,
    at::TensorList down_ws_list)
{
    const int64_t N = num_layers();
    TORCH_CHECK(N > 0,
                "hyvla set_moe_weights must follow CausalDecoderModel::set_weights "
                "(text weights first; num_layers comes from them)");
    TORCH_CHECK(chunk_size > 0,
                "hyvla set_moe_weights: chunk_size must be > 0, got ", chunk_size);

    // 孪生列表长度必须与基类层数一致 — MoT 双权重逐层对应, 不允许部分覆盖。
    auto check_list = [&](const at::TensorList& list, const char* name) {
        TORCH_CHECK(static_cast<int64_t>(list.size()) == N,
                    "hyvla set_moe_weights: ", name, ".size()=", list.size(),
                    " != num_layers=", N);
    };
    check_list(q_w_list,         "q_w_list");
    check_list(k_w_list,         "k_w_list");
    check_list(v_w_list,         "v_w_list");
    check_list(o_w_list,         "o_w_list");
    check_list(q_norm_list,      "q_norm_list");
    check_list(k_norm_list,      "k_norm_list");
    check_list(input_norm_list,  "input_norm_list");
    check_list(post_norm_list,   "post_norm_list");
    check_list(gate_list,        "gate_list");
    check_list(up_list,          "up_list");
    check_list(down_list,        "down_list");

    // Hy-VLA 是 Qwen3 系 (带 q/k head-norm); 基类必须已按该结构配置, 否则
    // build_layer_subgraph 步 3 的 head-norm 相取不到权重。注意这一份 norm
    // 两塔共享, 孪生侧传同一张量即可 (见 .h 的差异说明 1)。
    TORCH_CHECK(has_qk_norm_,
                "hyvla set_moe_weights: base set_weights must have non-empty "
                "q/k_norm lists (Hy-VLA is Qwen3-style)");

    // QKV bias 与 text 侧同进退: 三个列表要么全空 (基类 has_qkv_bias_=false),
    // 要么全 N (true)。不一致 = 主/孪生 ABI 错配, 立即报错。
    const bool moe_bias = (q_bias_list.size() > 0);
    TORCH_CHECK((q_bias_list.size() > 0) == (k_bias_list.size() > 0)
                && (k_bias_list.size() > 0) == (v_bias_list.size() > 0),
                "hyvla set_moe_weights: q/k/v_bias_list must all be empty or all "
                "size==num_layers; got q=", q_bias_list.size(),
                " k=", k_bias_list.size(), " v=", v_bias_list.size());
    TORCH_CHECK(moe_bias == has_qkv_bias_,
                "hyvla set_moe_weights: act-twin bias presence (", moe_bias,
                ") must match text-side has_qkv_bias_ (", has_qkv_bias_, ")");
    if (moe_bias) {
        check_list(q_bias_list, "q_bias_list");
        check_list(k_bias_list, "k_bias_list");
        check_list(v_bias_list, "v_bias_list");
    }

    // 秩 + dtype/device/contiguous 检查。这些 act-twin 权重
    // 经 moe_layers_ keepalive 后在 preload 回调以 data_ptr<c10::Half>() DMA 读 → 必须
    // fp16 + PrivateUse1 + contiguous,否则错 device/dtype 张量在入口通过却在 DMA 静默腐蚀。
    // Shape details remain launcher-validated against GEMM N/K.
    auto check_fp16_rpu = [&](const at::TensorList& list, const char* name,
                             int64_t rank) {
        for (int64_t i = 0; i < N; ++i) {
            TORCH_CHECK(list[i].defined(),
                        "hyvla set_moe_weights: ", name, "[", i, "] undefined");
            TORCH_CHECK(list[i].dim() == rank,
                        "hyvla set_moe_weights: ", name, "[", i, "] must be ",
                        rank, "D, got ", list[i].dim(), "D");
            TORCH_CHECK(list[i].scalar_type() == at::kHalf
                        && list[i].device().type() == at::kPrivateUse1
                        && list[i].is_contiguous(),
                        "hyvla set_moe_weights: ", name, "[", i,
                        "] must be fp16 contiguous RPU tensor, got dtype=",
                        list[i].scalar_type(), " device=", list[i].device().type(),
                        " contiguous=", list[i].is_contiguous());
        }
    };
    // 七个**投影**权重在 W8A16 下是 int8 ⇒ 单独一条检查链 (dtype 由下面的
    // check_scaled 按 scale 是否给出来定); 其余 (norm / bias / final_norm)
    // 恒为 fp16, 继续走 check_fp16_rpu。
    // ⚠️ 孪生权重不经基类 CausalDecoderModel::set_weights ⇒ 基类那套 W8A16
    //    校验对它们不生效, 必须在这里自己做, 否则错 dtype 会一路走到 DMA。
    const bool moe_q8 = (q_ws_list.size() > 0);
    auto check_scale_list = [&](const at::TensorList& lst, const char* nm) {
        TORCH_CHECK((lst.size() > 0) == moe_q8,
                    "hyvla set_moe_weights: ", nm, " must be empty unless ALL "
                    "seven W8A16 scale lists are provided");
        if (moe_q8) check_list(lst, nm);
    };
    check_scale_list(k_ws_list,    "k_ws_list");
    check_scale_list(v_ws_list,    "v_ws_list");
    check_scale_list(o_ws_list,    "o_ws_list");
    check_scale_list(gate_ws_list, "gate_ws_list");
    check_scale_list(up_ws_list,   "up_ws_list");
    check_scale_list(down_ws_list, "down_ws_list");
    if (moe_q8) check_list(q_ws_list, "q_ws_list");
    auto check_scaled = [&](const at::TensorList& w, const at::TensorList& sc,
                            const char* name) {
        for (int64_t i = 0; i < N; ++i) {
            TORCH_CHECK(w[i].defined() && w[i].dim() == 2
                        && w[i].device().type() == at::kPrivateUse1
                        && w[i].is_contiguous(),
                        "hyvla set_moe_weights: ", name, "[", i,
                        "] must be 2D contiguous RPU tensor");
            // 给了 scale ⇒ 权重必须是量化的：int8 [N,K] (W8A16) 或
            // nibble-packed uint8 [N,K/2] (W4A16)。launcher 按 dtype 派发。
            TORCH_CHECK(moe_q8 ? (w[i].scalar_type() == at::kChar
                                  || w[i].scalar_type() == at::kByte)
                               : (w[i].scalar_type() == at::kHalf),
                        "hyvla set_moe_weights: ", name, "[", i, "] must be ",
                        (moe_q8 ? "int8 or packed-uint8 (quant scales given)" : "fp16"),
                        ", got ", w[i].scalar_type());
            if (moe_q8) {
                TORCH_CHECK(sc[i].defined() && sc[i].dim() == 1
                            && sc[i].scalar_type() == at::kHalf
                            && sc[i].device().type() == at::kPrivateUse1
                            && sc[i].is_contiguous()
                            && sc[i].numel() == w[i].size(0),
                            "hyvla set_moe_weights: ", name, "_scale[", i,
                            "] must be 1D contiguous fp16 RPU [N=",
                            w[i].size(0), "]");
            }
        }
    };
    check_scaled(q_w_list,    q_ws_list,    "q_w_list");
    check_scaled(k_w_list,    k_ws_list,    "k_w_list");
    check_scaled(v_w_list,    v_ws_list,    "v_w_list");
    check_scaled(o_w_list,    o_ws_list,    "o_w_list");
    check_scaled(gate_list,   gate_ws_list, "gate_list");
    check_scaled(up_list,     up_ws_list,   "up_list");
    check_scaled(down_list,   down_ws_list, "down_list");
    check_fp16_rpu(q_norm_list,     "q_norm_list",     1);
    check_fp16_rpu(k_norm_list,     "k_norm_list",     1);
    check_fp16_rpu(input_norm_list, "input_norm_list", 1);
    check_fp16_rpu(post_norm_list,  "post_norm_list",  1);
    if (moe_bias) {
        check_fp16_rpu(q_bias_list, "q_bias_list", 1);
        check_fp16_rpu(k_bias_list, "k_bias_list", 1);
        check_fp16_rpu(v_bias_list, "v_bias_list", 1);
    }

    TORCH_CHECK(final_norm_moe_w.defined() && final_norm_moe_w.dim() == 1
                && final_norm_moe_w.size(0) == hidden_size()
                && final_norm_moe_w.scalar_type() == at::kHalf
                && final_norm_moe_w.device().type() == at::kPrivateUse1
                && final_norm_moe_w.is_contiguous(),
                "hyvla set_moe_weights: final_norm_moe_w must be 1D [hidden=",
                hidden_size(), "] fp16 contig RPU, got ", final_norm_moe_w.sizes(),
                " dtype=", final_norm_moe_w.scalar_type());
    // ⚠️ Hy-VLA 的 ckpt 无 norm_v.weight ⇒ final norm 两塔共享。步 7 据此**单算**,
    // 不再双算 + merge。若调用方在这里传了一个不同的 γ, 那个 γ 会被完全忽略 ——
    // 静默算错。所以强制它必须是与 text 侧同一个张量 (同 data_ptr), 而不是"形状对
    // 就行"。换成真有两份 final norm 的模型时, 这条断言会立刻炸, 提醒改回双算。
    TORCH_CHECK(final_norm_text_.defined()
                && final_norm_moe_w.data_ptr() == final_norm_text_.data_ptr(),
                "hyvla set_moe_weights: final_norm_moe_w must be the SAME tensor as "
                "the text-side final_norm_w (Hy-VLA has no norm_v; step 7 computes it "
                "once). Pass the identical tensor object.");

    // 行掩码: [chunk_size] fp16 RPU contig, 取值 {0,1}。
    TORCH_CHECK(text_row_mask.defined() && text_row_mask.dim() == 1
                && text_row_mask.size(0) == chunk_size
                && text_row_mask.scalar_type() == at::kHalf
                && text_row_mask.is_contiguous()
                && text_row_mask.device().type() == at::kPrivateUse1,
                "hyvla set_moe_weights: text_row_mask must be [", chunk_size,
                "] fp16 contig RPU");
    at::Tensor mask_cpu = text_row_mask.to(at::kCPU).to(at::kFloat);
    TORCH_CHECK(((mask_cpu == 0.0f) | (mask_cpu == 1.0f)).all().item<bool>(),
                "hyvla set_moe_weights: text_row_mask values must be exactly 0/1");

    // ── 掩码常量化: [cs, 1] fp16 行向量 (act = 1 - text) ──
    // Masks are stored as [cs,1] row vectors and broadcast during merge. Exact
    // 0/1 multiplication preserves the selected branch bit-for-bit.
    // 常量在 host (CPU) 生成后整体复制到 RPU DDR，不在 device 上填充。
    const at::Tensor text_cpu = mask_cpu.to(at::kHalf);
    const auto rpu_opts =
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1);
    text_mask_ddr_ = text_cpu.reshape({chunk_size, 1}).contiguous().to(rpu_opts);
    act_mask_ddr_  = (1.0f - mask_cpu).to(at::kHalf)
                         .reshape({chunk_size, 1}).contiguous().to(rpu_opts);

    // ── 提交状态 (DDR keepalive: 跨 forward 持有, 直至下次 set_moe_weights) ──
    moe_layers_.clear();
    moe_layers_.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
        moe_layers_.push_back({
            q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
            q_norm_list[i], k_norm_list[i],
            input_norm_list[i], post_norm_list[i],
            gate_list[i], up_list[i], down_list[i],
            moe_bias ? q_bias_list[i] : at::Tensor(),
            moe_bias ? k_bias_list[i] : at::Tensor(),
            moe_bias ? v_bias_list[i] : at::Tensor(),
            moe_q8 ? q_ws_list[i]    : at::Tensor(),
            moe_q8 ? k_ws_list[i]    : at::Tensor(),
            moe_q8 ? v_ws_list[i]    : at::Tensor(),
            moe_q8 ? o_ws_list[i]    : at::Tensor(),
            moe_q8 ? gate_ws_list[i] : at::Tensor(),
            moe_q8 ? up_ws_list[i]   : at::Tensor(),
            moe_q8 ? down_ws_list[i] : at::Tensor(),
        });
    }
    text_row_mask_    = text_row_mask;
    moe_chunk_size_   = chunk_size;

    invalidate_model_state();   // set_moe_weights 末非空语句。
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers — 基类布局 + vision 孪生 Temp 槽 + 掩码/孪生权重 preload 槽
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BufferDecl> HyVlaVlmModel::declare_buffers(const LayoutContext& lctx) {
    auto d = CausalDecoderModel::declare_buffers(lctx);
    if (hyvla_mask_once_on("vlm")) hyvla_widen_sdpa_mask_slot(d);
    TORCH_CHECK(moe_chunk_size_ > 0,
                "hyvla declare_buffers: set_moe_weights must precede forward");
    // 框架解析出的 chunk_size 是单块 SPM 规划宽度 (ceil16(seq)), 实际行数
    // = chunk.len = seq_len = moe_chunk_size_。基类/孪生双算槽按 lctx.chunk_size
    // 分配 (容纳); 行掩码槽按 moe_chunk_size_(=cs_mask) 分配, 与 set_moe_weights
    // 展开的 [cs_mask,w] 常量张量及 build_layer_subgraph 的 seq_len 行消费一致。
    // 单块要求 resolved cs 覆盖整个 prefix (subclass_chunk_size_valid 已保证)。
    TORCH_CHECK(lctx.chunk_size >= moe_chunk_size_,
                "hyvla declare_buffers: resolved chunk_size=", lctx.chunk_size,
                " < moe chunk_size=", moe_chunk_size_,
                " (prefill 单块必须覆盖整个 prefix)");

    const int64_t cs      = lctx.chunk_size;   // 基类/孪生双算槽容纳宽度
    const int64_t cs_mask = moe_chunk_size_;   // 行掩码实际行数 (=seq_len)
    const int64_t h   = hidden_size();
    const int64_t hd  = head_dim();
    const int     tp  = attn_tp();
    const int64_t nl  = num_layers();
    const int64_t local_q  = num_q_heads() / tp;
    const int64_t local_kv = num_kv_heads() * hd / tp;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
    const int64_t res = A(cs * h * DWIDTH);
    const int64_t q   = A(cs * local_q * hd * DWIDTH);
    const int64_t kv  = A(cs * local_kv * DWIDTH);
    const int64_t mlp = A(cs * (intermediate_size() / NUM_CORES) * DWIDTH);
    const int64_t nw  = A(h * DWIDTH);
    const int64_t hnw = A(hd * DWIDTH);

    // act 分支 Temp 槽 — phase 区间镜像基类对应槽,
    // 使 SPM 别名窗口一致: q_a 2..5, k_a/v_a 2..4, oproj_a 5..5, gate_a 7..8,
    // up_a 7..7, down_a 7..8。norm_a 是 act 侧 norm 暂存 (input/headnorm/post/
    // final 各相复用), 生命周期与 input_norm 同 (1..8)。
    d.push_back({"q_a",     q,   2, 5, StorageClass::Temp, 0, nullptr});
    d.push_back({"k_a",     kv,  2, 4, StorageClass::Temp, 0, nullptr});
    d.push_back({"v_a",     kv,  2, 4, StorageClass::Temp, 0, nullptr});
    d.push_back({"oproj_a", res, 5, 5, StorageClass::Temp, 0, nullptr});
    // The serial branches reuse the base gate/up slots; only down_a needs an
    // independent lifetime for the row merge.
    d.push_back({"down_a",  res, 7, 8, StorageClass::Temp, 0, nullptr});
    // norm_a1/norm_a2 cover the two independent twin-norm lifetimes.
    // `MOT_NORM_NOMERGE` 打开对应半边时，act 侧 norm 暂存要活到下游 GEMM
    // 那一相（QKV 是 phase 2、MLP 是 phase 7）⇒ 相应把窗口拉长一格。
    // 关闭开关时恢复原有窗口布局。
    const unsigned nomerge_mask = hyvla_mot_norm_nomerge_mask();
    const int a1_hi = (nomerge_mask & MOT_NOMERGE_QKV) ? 2 : 1;
    const int a2_hi = (nomerge_mask & MOT_NOMERGE_MLP) ? 7 : 6;
    d.push_back({"norm_a1", res, 1, a1_hi, StorageClass::Temp, 0, nullptr});
    d.push_back({"norm_a2", res, 6, a2_hi, StorageClass::Temp, 0, nullptr});

    // ── 行掩码 Persistent 槽 (text/act 各一个 [cs,1] 行向量) ──
    // 常量掩码整个模型生命周期不变 → Persistent + preload_callback 一次 DMA,
    // REPLAY 零成本。见 set_moe_weights 里"为什么是行向量"的说明。
    for (int is_text = 1; is_text >= 0; --is_text) {
        BufferDecl b;
        b.name    = is_text ? "hyvla_mask_txt_row" : "hyvla_mask_act_row";
        b.size    = A(cs_mask * DWIDTH);
        b.storage = StorageClass::Persistent;
        b.scope   = BufferScope::LayerWide;
        b.preload_callback =
            [this, is_text, cs_mask](FusedModelBase&, int, uint32_t core0_addr) {
                const at::Tensor& t = is_text ? text_mask_ddr_ : act_mask_ddr_;
                TORCH_CHECK(t.defined(), "hyvla 行掩码未构建 (set_moe_weights 未调用?)");
                rpu_launch_ddr_broadcast_spm_dma(
                    t.data_ptr<c10::Half>(), cs_mask, core0_addr, NUM_CORES);
            };
        d.push_back(b);
    }

    // ── act 孪生 norm / bias preload 槽 (镜像基类 norm_w/post_norm_w/q_norm_w/
    //    k_norm_w + q/k/v_bias 的 PersistentPerLayer 模式) ──
    auto add_norm = [&](const char* name, int64_t sz, int64_t elems,
                        at::Tensor MoeLayerWeights::* member) {
        BufferDecl b;
        b.name      = name;
        b.size      = sz;
        b.storage   = StorageClass::PersistentPerLayer;
        b.per_layer = nl;
        b.scope     = BufferScope::LayerWide;
        b.preload_callback =
            [this, member, elems](FusedModelBase&, int L, uint32_t core0_addr) {
                rpu_launch_ddr_broadcast_spm_dma(
                    (moe_layers_[L].*member).data_ptr<c10::Half>(),
                    elems, core0_addr);
            };
        d.push_back(b);
    };
    add_norm("norm_w_act",      nw,  h,  &MoeLayerWeights::input_norm_w);
    add_norm("post_norm_w_act", nw,  h,  &MoeLayerWeights::post_norm_w);
    add_norm("q_norm_w_act",    hnw, hd, &MoeLayerWeights::q_norm_w);
    add_norm("k_norm_w_act",    hnw, hd, &MoeLayerWeights::k_norm_w);

    if (has_qkv_bias_) {
        // bias 按 col-partition scatter: core c 收到其输出通道切片 (numel/tp),
        // 与基类 q/k_bias 槽的布局一致。
        auto dma_safe = [&A](int64_t elems) -> int64_t {
            return A(((elems + 255) / 256) * 256 * DWIDTH);
        };
        auto add_bias = [&](const char* name, int64_t sz,
                            at::Tensor MoeLayerWeights::* member) {
            BufferDecl b;
            b.name      = name;
            b.size      = sz;
            b.storage   = StorageClass::PersistentPerLayer;
            b.per_layer = nl;
            b.scope     = BufferScope::LayerWide;
            b.preload_callback =
                [this, member](FusedModelBase&, int L, uint32_t core0_addr) {
                    const int tp_ = attn_tp();
                    const at::Tensor& bw = moe_layers_[L].*member;
                    const int64_t per_core = bw.numel() / tp_;
                    rpu_launch_ddr_scatter_spm_dma(
                        bw.data_ptr<c10::Half>(),
                        per_core, per_core * DWIDTH, core0_addr, tp_);
                };
            d.push_back(b);
        };
        add_bias("q_bias_act", dma_safe(local_q * hd), &MoeLayerWeights::q_bias);
        add_bias("k_bias_act", dma_safe(local_kv),     &MoeLayerWeights::k_bias);
        add_bias("v_bias_act", dma_safe(local_kv),     &MoeLayerWeights::v_bias);
    }

    // (HALO 在此还声明了一个 "final_norm_moe_w" Persistent 槽 —— Hy-VLA 的 final
    //  norm 只有一份, 步 7 直接用基类的 "final_norm_w", 该槽已随双算一并删除。)
    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_row_merge — out = text*text_mask + act*act_mask (3 次 eltwise)
// 就地语义: MUL 覆写 text/act 两个输入槽; out 与 text 同槽是常态 (text 槽即基类
// 流水的下游输入)。掩码槽 Persistent, 不被覆写。
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaVlmModel::emit_row_merge(
    uint32_t text_addr, uint32_t act_addr, uint32_t out_addr,
    uint32_t text_mask_addr, uint32_t act_mask_addr,
    int64_t rows, int64_t cols, int num_cores)
{
    // 两次行广播 MUL (掩码 [rows,1]) + 一次全尺寸 ADD。掩码恒 0/1 ⇒ 逐位精确。
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        text_mask_addr, text_addr, text_addr, rows, cols,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false, num_cores);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        act_mask_addr, act_addr, act_addr, rows, cols,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false, num_cores);
    rpu_launch_eltwise_binary_spm_kernel(
        text_addr, act_addr, out_addr,
        rows * cols, ValuOpType::ADD, c10::Half(1.0f), num_cores);
}

// ─────────────────────────────────────────────────────────────────────────────
// build_layer_subgraph — 基类 8 相的 MoT 双算版 (规格 7 步)
//
// 行独立性论证: RMSNorm/GEMM/bias/SiLU 全部逐行运算, 故"双算全 cs 行后按行掩码
// 合并"与逐行选分支严格等价; SDPA 是唯一跨行算子, 但 q/k/v 在其之前已按行合并,
// 与 HALO 参考语义一致。RoPE 行位置只依赖 position+offset, 与分支无关 → 合并后
// 单次 in-place。
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaVlmModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    TORCH_CHECK(!text_layers_.empty() && !moe_layers_.empty(),
                "hyvla build_layer_subgraph: set_weights_hyvla + set_moe_weights "
                "must both have been called");
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "hyvla build_layer_subgraph: explicit 2D mask required "
                "(act step is non-causal prefix attention)");
    TORCH_CHECK(chunk.len == moe_chunk_size_,
                "hyvla build_layer_subgraph: chunk.len=", chunk.len,
                " != moe chunk_size=", moe_chunk_size_);

    const auto& lw = text_layers_[layer_idx];     // text 权重影子 (引用计数副本)
    const auto& mw = moe_layers_[layer_idx];      // act 孪生权重
    const int64_t seq_len = chunk.len;
    const int64_t cos_sin_start = ctx().position + chunk.offset;
    const bool is_last_layer = (layer_idx == num_layers() - 1);
    const int64_t h   = hidden_size();
    const int64_t nq  = num_q_heads();
    const int64_t nkv = num_kv_heads();
    const int64_t hd  = head_dim();
    const int tp = attn_tp();
    const int64_t local_q  = nq / tp;
    const int64_t local_kv = nkv * hd / tp;
    const int64_t elems_mlp = seq_len * (intermediate_size() / NUM_CORES);
    // 单一 [seq_len,1] 行掩码对，三个 merge 宽度共用（行广播）。
    const uint32_t m_txt = addr(0, "hyvla_mask_txt_row");
    const uint32_t m_act = addr(0, "hyvla_mask_act_row");
    const int64_t cols_h  = h;                  // input_norm / oproj / down
    const int64_t cols_q  = local_q * hd;       // q 槽（col-partition over tp）
    const int64_t cols_kv = local_kv;           // k/v 槽

    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }

    // ── 步 1: input RMSNorm 双算 (→ merge 入 "input_norm"，除非 NOMERGE) ──
    // text γ 经基类 "norm_w" 槽, act γ 经 "norm_w_act"。
    // merge 路径下 input_norm 持逐行选支的 normed hidden, 供两套 QKV 共读。
    // 两套 QKV 分别发射时可各读各的 norm 输出，省掉这次 merge。
    // 逐位等价的逐行论证见 `hyvla_mot_norm_nomerge_mask` 的注释。
    const unsigned nomerge = hyvla_mot_norm_nomerge_mask();
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "norm_w"), seq_len, h, eps_hyvla_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "norm_a1"),
        layer_addr(layer_idx, 0, "norm_w_act"), seq_len, h, eps_hyvla_);
    if (!(nomerge & MOT_NOMERGE_QKV))
        emit_row_merge(addr(0, "input_norm"), addr(0, "norm_a1"), addr(0, "input_norm"),
                       m_txt, m_act, seq_len, cols_h, NUM_CORES);

    // ── 步 2: QKV 双发射 (text→q/k/v, act→q_a/k_a/v_a) + merge ──
    // bias 各用自家 PersistentPerLayer 槽。NOMERGE 时 text 读 "input_norm"、
    // act 读 "norm_a1"；否则两边都读 merge 后的 "input_norm"。
    auto qkv = [&](uint32_t in_addr, const at::Tensor& w, const at::Tensor& ws,
                   const char* out, const char* bias_slot, int64_t n) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in_addr, w, addr(0, out),
            seq_len, n, h, /*partition=*/1, /*num_cores=*/tp,
            has_qkv_bias_ ? layer_addr(layer_idx, 0, bias_slot) : 0u,
            /*force_gemm=*/false, /*scale=*/ws);
    };
    const uint32_t in_txt = addr(0, "input_norm");
    const uint32_t in_act = (nomerge & MOT_NOMERGE_QKV) ? addr(0, "norm_a1")
                                                        : addr(0, "input_norm");
    qkv(in_txt, lw.q_w, lw.q_ws, "q",   "q_bias",     nq * hd);
    qkv(in_txt, lw.k_w, lw.k_ws, "k",   "k_bias",     nkv * hd);
    qkv(in_txt, lw.v_w, lw.v_ws, "v",   "v_bias",     nkv * hd);
    qkv(in_act, mw.q_w, mw.q_ws, "q_a", "q_bias_act", nq * hd);
    qkv(in_act, mw.k_w, mw.k_ws, "k_a", "k_bias_act", nkv * hd);
    qkv(in_act, mw.v_w, mw.v_ws, "v_a", "v_bias_act", nkv * hd);
    emit_row_merge(addr(0, "q"), addr(0, "q_a"), addr(0, "q"), m_txt, m_act, seq_len, cols_q,  tp);
    emit_row_merge(addr(0, "k"), addr(0, "k_a"), addr(0, "k"), m_txt, m_act, seq_len, cols_kv, tp);
    emit_row_merge(addr(0, "v"), addr(0, "v_a"), addr(0, "v"), m_txt, m_act, seq_len, cols_kv, tp);

    // ── 步 3: **先 RoPE，再 q/k head-norm**（与 HALO 相反），norm 为两塔共享 ──
    // Hy-VLA 的 HyDualTower.forward:456-469 先对拼接后的 q/k 施 RoPE，再按塔切分做
    // head-dim RMSNorm；且该 norm 只有一份（query_layernorm / key_layernorm 非 _v）。
    // 现有 7 个 RPU emitter（gemma4 / halo_* / qwen3_5 / qwenpi05 / rhino_vla / qwen3）
    // 全是 norm→rope，本文件是**唯一** rope→norm 的。
    // 可行性依据：两个 kernel 都是 src/dst 分离、无隐藏状态的独立 SPM launch
    //   rpu_launch_rope_spm_kernel(x_addr, y_addr, cos, sin, seq, heads, hd, start)
    //   rpu_launch_rmsnorm_spm_kernel(in_addr, out_addr, w_addr, M, C, eps)
    // 且 QK-norm 的 framing 是 (M=seq*heads, C=head_dim)，RoPE 不改变 [seq,heads,hd]
    // 布局 ⇒ 同一 framing 在 RoPE 前后都成立，换序即调用顺序调换。
    // 使用模型提供的 bf16 inv_freq。
    // 因 norm 共享，此处**不做双算/merge**（mw.q_norm_w / mw.k_norm_w 不参与计算）。
    c10::Half* cos_ptr = cos_hyvla_.data_ptr<c10::Half>();
    c10::Half* sin_ptr = sin_hyvla_.data_ptr<c10::Half>();

    // PARTIAL_ROPE uses the model-owned tables and the same supported RoPE
    // semantics as the default path.
    const bool prope = hyvla_partial_rope_on("vlm");
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
        layer_addr(layer_idx, 0, "q_norm_w"), seq_len * local_q, hd, eps_hyvla_);

    if (prope) {
        rpu_launch_partial_mrope_spm_kernel(
            addr(0, "k"), addr(0, "input_norm"), cos_ptr, sin_ptr,
            cos_sin_start, seq_len, local_kv_heads, hd, /*rotary_dim=*/hd, tp);
    } else {
        rpu_launch_rope_spm_kernel(addr(0, "k"), addr(0, "input_norm"),
                                   cos_ptr, sin_ptr, seq_len, local_kv_heads, hd, cos_sin_start);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "input_norm"), addr(0, "k"),
        layer_addr(layer_idx, 0, "k_norm_w"), seq_len * local_kv_heads, hd, eps_hyvla_);

    // ── 步 4: KV insert + SDPA 单次; O 双算 + merge → 残差归约 ──
    // K/V 已按行合并, insert/SDPA 与基类完全一致。O_proj 行分到 tp 核, 结果
    // 各持全宽 h 部分和; 两分支各自行分一致 → 部分和也逐行可掩码合并。
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    rpu_launch_insert_kcache_spm_unified(
        k_cache, ctx().position + chunk.offset, addr_offset("k").value,
        seq_len, nkv, hd, tp);
    rpu_launch_insert_vcache_spm_unified(
        v_cache, ctx().position + chunk.offset, addr_offset("v").value,
        seq_len, nkv, hd, tp);
    const uint32_t mask_off = addr_offset("sdpa_mask").value;
    // mask 逐层恒定；槽撑满相位后只需在层 0 灌一次（见 MASK_ONCE）。
    if (!hyvla_mask_once_on("vlm") || layer_idx == 0)
        sdpa_dma_mask_to_spm(prepared_mask_hyvla_, mask_off, seq_len, chunk.kv_seq_len, tp);
    rpu_launch_sdpa_spm_dispatch(
        SdpaKernelType::FLASH_ATTN_SPM,
        k_cache, v_cache, prepared_mask_hyvla_.mask_type, c10::nullopt,
        addr_offset("q").value, addr_offset("output").value,
        addr_offset("sdpa_tmp").value, mask_off,
        seq_len, nq, nkv, hd, chunk.kv_seq_len, tp, NUM_CORES);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0, /*force_gemm=*/false, /*scale=*/lw.o_ws);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), mw.o_w, addr(0, "oproj_a"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0, /*force_gemm=*/false, /*scale=*/mw.o_ws);
    emit_row_merge(addr(0, "oproj"), addr(0, "oproj_a"), addr(0, "oproj"),
                   m_txt, m_act, seq_len, cols_h, tp);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
        seq_len, h, tp, NUM_CORES);

    // ── 步 5: post-attention RMSNorm 双算 (+ merge → residual1，除非 NOMERGE) ──
    // 与步 1 同理：两个 mlp_half 本来就是分别发射的 ⇒ 各读各的 norm 输出。
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "residual1"),
        layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, eps_hyvla_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "norm_a2"),
        layer_addr(layer_idx, 0, "post_norm_w_act"), seq_len, h, eps_hyvla_);
    if (!(nomerge & MOT_NOMERGE_MLP))
        emit_row_merge(addr(0, "residual1"), addr(0, "norm_a2"), addr(0, "residual1"),
                       m_txt, m_act, seq_len, cols_h, NUM_CORES);

    // ── 步 6: 手写双 MLP (SwiGLU) → down merge → 残差归约 ──
    // 不用基类 emit_mlp_pipeline: 它把 all_reduce_sum_residual 写死在 down 后,
    // 而双分支必须在归约前合并 (act 分支用独立 gate_a/up_a/down_a 槽)。
    auto mlp_half = [&](uint32_t in_addr,
                        const at::Tensor& gate_w, const at::Tensor& up_w,
                        const at::Tensor& down_w, const at::Tensor& gate_ws,
                        const at::Tensor& up_ws, const at::Tensor& down_ws,
                        const char* g, const char* u, const char* dn) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in_addr, gate_w, addr(0, g),
            seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
            /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/gate_ws);
        if (!hyvla_silu_mul_on("vlm"))
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, g), addr(0, g), elems_mlp, ValuOpType::SILU,
                /*is_gelu=*/false, NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in_addr, up_w, addr(0, u),
            seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
            /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/up_ws);
        if (hyvla_silu_mul_on("vlm"))
            rpu_launch_silu_mul_spm_kernel(
                addr(0, g), addr(0, u), addr(0, g), elems_mlp, NUM_CORES);
        else
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, g), addr(0, u), addr(0, g),
                elems_mlp, ValuOpType::MUL, c10::Half(1.0f), NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, g), down_w, addr(0, dn),
            seq_len, h, intermediate_size(), /*partition=*/0, NUM_CORES,
            /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/down_ws);
    };
    mlp_half(addr(0, "residual1"),
             lw.gate_w, lw.up_w, lw.down_w,
             lw.gate_ws, lw.up_ws, lw.down_ws, "gate", "up", "down");
    // 孪生分支复用 text 的 gate/up 暂存 (见 declare_buffers 的说明): 此刻 text
    // 的 SwiGLU 中间量已消费完, 其结果在 "down" 里等着 merge。
    // ⚠️ NOMERGE 时它的输入是 "norm_a2"（自己那一路的 post-norm 输出），
    //    不是 "residual1" —— residual1 此时持的是 **text** 那一路的 norm 输出。
    mlp_half((nomerge & MOT_NOMERGE_MLP) ? addr(0, "norm_a2") : addr(0, "residual1"),
             mw.gate_w, mw.up_w, mw.down_w,
             mw.gate_ws, mw.up_ws, mw.down_ws, "gate", "up", "down_a");
    emit_row_merge(addr(0, "down"), addr(0, "down_a"), addr(0, "down"),
                   m_txt, m_act, seq_len, cols_h, NUM_CORES);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
        seq_len, h, NUM_CORES, NUM_CORES);

    // ── 步 7 (末层): final RMSNorm —— **单次, 不双算** ──
    // Hy-VLA 的 ckpt 里 final norm 只有一份 (641 = 32×20 + 1, 无 norm_v.weight),
    // set_moe_weights 已断言 final_norm_moe_w 与 text 侧是同一张量。对同一个 γ
    // 双算再按行掩码 merge 恒等于单算 —— HALO 那份双算在这里是纯冗余, 每次末层
    // 白烧 1 次全宽 rmsnorm + 3 次全宽 eltwise, 还把 norm_a 的生命周期钉到
    // phase 8 (见 declare_buffers 的 norm_a1/norm_a2 说明)。rmsnorm 支持原地
    // 基类使用相同的原地形式, 故直接写回 residual1。
    if (is_last_layer) {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "residual1"),
            addr(0, "final_norm_w"), seq_len, h, eps_hyvla_);
    }

    if (!ctx().output_to_spm) {
        emit_layer_output_dma(layer_idx, chunk);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// step_forward — 一次 act step (prefix-KV 非因果 forward + hidden 输出)
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor HyVlaVlmModel::step_forward(
    const at::Tensor& x_emb, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len)
{
    TORCH_CHECK(!text_layers_.empty() && moe_chunk_size_ > 0,
                "hyvla step_forward before set_weights_hyvla / set_moe_weights");
    const int64_t cs = moe_chunk_size_;
    const int64_t h  = hidden_size();
    TORCH_CHECK(x_emb.dim() == 3 && x_emb.size(0) == 1 && x_emb.size(1) == cs
                && x_emb.size(2) == h && x_emb.scalar_type() == at::kHalf
                && x_emb.is_contiguous()
                && x_emb.device().type() == at::kPrivateUse1,
                "hyvla step_forward: x_emb must be [1,", cs, ",", h,
                "] fp16 contig RPU");
    TORCH_CHECK(attn_mask_4d.defined(),
                "hyvla step_forward: explicit attention mask required");
    TORCH_CHECK(prefix_len >= 0, "hyvla step_forward: prefix_len must be >= 0");

    // per-forward cos/sin (镜像 qwenpi05 cos_ref_/sin_ref_): 覆盖 RoPE 影子成员,
    // 赋值即保活过 graph capture; build_layer_subgraph (:525-548) 读 cos_hyvla_/sin_hyvla_
    // 的 raw c10::Half*。CFG 三分支共用一个句柄, 每分支传自己 rope_position 的表。
    // 校验范式对齐 qwenpi05_forward (kernel 按 raw c10::Half* 消费)。
    TORCH_CHECK(cos.defined() && sin.defined(),
                "hyvla step_forward: cos/sin must be defined");
    TORCH_CHECK(cos.device().type() == at::kPrivateUse1
                && sin.device().type() == at::kPrivateUse1,
                "hyvla step_forward: cos/sin must be on RPU");
    TORCH_CHECK(cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf,
                "hyvla step_forward: cos/sin must be fp16");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
                "hyvla step_forward: cos/sin must be contiguous");
    TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes(),
                "hyvla step_forward: cos/sin must be 2-D, equal shape; cos=",
                cos.sizes(), " sin=", sin.sizes());
    TORCH_CHECK(cos.size(1) == head_dim() / 2,
                "hyvla step_forward: cos.size(1) must be head_dim/2=", head_dim() / 2,
                ", got ", cos.size(1));
    TORCH_CHECK(cos.size(0) >= prefix_len + cs,
                "hyvla step_forward: cos row count (", cos.size(0),
                ") < prefix_len+cs (", prefix_len + cs, ")");
    cos_hyvla_ = cos;
    sin_hyvla_ = sin;

    // KV headroom: 7-D K 布局的 max_seq = size(1)*size(5)。
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + cs <= kc.size(1) * kc.size(5),
                    "hyvla step_forward: prefix_len(", prefix_len, ")+cs(", cs,
                    ") exceeds k_cache capacity ", kc.size(1) * kc.size(5));
    }

    // prefill 必须单块 (MASK_2D 行掩码/KV merge 按整块设计)。把本实例 chunk
    // override 设为 ceil16(seq): base forward 走 override 路径, compute_chunks
    // 切出单块 len=seq (subclass_chunk_size_valid 放行)。取 ceil16 而非 seq 本身 —
    // override 路径的 (override/16)*16 会把非 16 对齐值向下取整。RAII 恢复
    // (异常路径也恢复; Wall-OSS guard 先例, 但值是 single_cs 而非 0)。
    const int64_t single_cs = ((moe_chunk_size_ + 15) / 16) * 16;
    const int64_t saved_chunk_size = get_chunk_size_override();
    set_chunk_size_override(single_cs);
    auto chunk_guard = c10::make_scope_exit([this, saved_chunk_size] {
        set_chunk_size_override(saved_chunk_size);
    });

    // host 侧合成的 x_emb 可能带脏 cache 行; BUILD 前显式 flush。
    rpu_ddr_flush_force(x_emb.data_ptr<c10::Half>());

    // 基类 forward 持有掩码准备 + run_all_layers; 行位置恒定 → 1D RoPE,
    // position_ids/deepstack 均空。返回基类 output_tensor_ (graph 写目标): 在
    // capture 作用域内它尚未执行, 必须由 Python 在作用域**退出后**读。
    // 切勿在此 copy 到外部张量 —— 会读到尚未执行的全 0。
    return CausalDecoderModel::forward(
        x_emb, k_caches, v_caches,
        std::optional<at::Tensor>(attn_mask_4d),
        /*position=*/prefix_len, /*is_causal=*/false,
        /*position_ids=*/std::nullopt, /*deepstack=*/std::nullopt);
}

}  // namespace v3

// =============================================================================
// Instance registry — ModelHandleRegistry<v3::HyVlaVlmModel>
// =============================================================================
using HyVlaVlmRegistry = ModelHandleRegistry<v3::HyVlaVlmModel>;

// =============================================================================
// Public C API for TORCH_LIBRARY wrappers (file-scope; decls in rpu_kernel_decls.h)
// =============================================================================
int64_t rpu_hyvla_vlm_create() {
    return HyVlaVlmRegistry::create();
}

void rpu_hyvla_vlm_destroy(int64_t handle) {
    HyVlaVlmRegistry::destroy(handle, "rpu_hyvla_vlm_destroy");
}

// text 主权重: 单 schema 包装 set_weights_hyvla (内部转基类 set_weights,
// mrope/deepstack 空; q/k norm 与 QKV bias 走 Qwen3+bias 路径)。
// 下面还有一个 `_w8a16` 变体, 只多七条 scale 列表, 落到同一个 set_weights_hyvla。
void rpu_hyvla_vlm_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias)
{
    HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_set_weights")
        ->set_weights_hyvla(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, use_silu, q_bias, k_bias, v_bias);
}

void rpu_hyvla_vlm_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws)
{
    HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_set_weights_w8a16")
        ->set_weights_hyvla(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, use_silu, q_bias, k_bias, v_bias,
            q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws);
}

void rpu_hyvla_vlm_set_moe_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    const at::Tensor& final_norm_moe_w, const at::Tensor& text_row_mask,
    int64_t chunk_size)
{
    HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_set_moe_weights")
        ->set_moe_weights(q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm,
                          post_norm, gate_w, up_w, down_w, q_bias, k_bias,
                          v_bias, final_norm_moe_w, text_row_mask, chunk_size);
}

void rpu_hyvla_vlm_set_moe_weights_w8a16(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    const at::Tensor& final_norm_moe_w, const at::Tensor& text_row_mask,
    int64_t chunk_size,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws)
{
    HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_set_moe_weights_w8a16")
        ->set_moe_weights(q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm,
                          post_norm, gate_w, up_w, down_w, q_bias, k_bias,
                          v_bias, final_norm_moe_w, text_row_mask, chunk_size,
                          q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws);
}

at::Tensor rpu_hyvla_vlm_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len)
{
    return HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_step_forward")
        ->step_forward(x_emb, k_caches, v_caches, cos, sin, attn_mask_4d, prefix_len);
}
