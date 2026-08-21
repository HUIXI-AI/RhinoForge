// rpu_halo_action_expert_model.cpp — HALO action expert (mode="act") fused 解码器。
//
// 继承 v3::CausalDecoderModel, 每层做"双算 + 行掩码合并":text 行结果与 act 孪生
// 权重结果按常量行掩码逐元素混合 (out = x_t*text + x_a*act)。掩码常量化的前提是
// HALO act 模式的行结构固定 (chunk_size=18, text 行 {0,17}); 掩码在 set_moe_weights
// 时按宽度展开为 DDR 常量并经 Persistent SPM 槽 preload, REPLAY 期零开销。
//
#include "rpu_halo_action_expert_model.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"

#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

namespace v3 {

HaloActionExpertModel::HaloActionExpertModel()  = default;
HaloActionExpertModel::~HaloActionExpertModel() = default;

// ─────────────────────────────────────────────────────────────────────────────
// set_weights_halo — text 主权重 (基类委托 + 子类影子)
// ─────────────────────────────────────────────────────────────────────────────
void HaloActionExpertModel::set_weights_halo(
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::TensorList q_bias_list, at::TensorList k_bias_list,
    at::TensorList v_bias_list)
{
    // HALO act expert MLP 固定 SwiGLU; 手写 MLP 相 (步 6) 写死 SILU, 不接受
    // use_silu=false 的静默错配。
    TORCH_CHECK(use_silu,
                "halo set_weights: use_silu must be true (HALO MLP is SwiGLU)");
    // 校验/状态提交全部委托基类 (含 q/k norm + bias 全套检查)。
    CausalDecoderModel::set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list, q_norm_list, k_norm_list,
        input_norm_list, post_norm_list, gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
        eps, use_silu,
        /*mrope_section=*/{}, /*deepstack_lang_layers=*/{},
        q_bias_list, k_bias_list, v_bias_list);

    // These fused subsystems inherit CausalDecoderModel, so they inherit its
    // deny-by-default envelope gate too. Their sequence length is NOT
    // caller-controlled the way a text prefill is -- it is structurally fixed by
    // the subsystem (action horizon / image rows) -- so the length half carries the
    // repo-wide KV ceiling and the CHUNK half stays 0 (= "auto, exactly as today"),
    // which keeps every one of these paths byte-identical. Declared here rather
    // than in the adapter because the bound is intrinsic to the subsystem, not a
    // policy an adapter gets to pick.
    set_chunk_envelope(/*max_kv_len=*/8192, /*chunk=*/0);

    // 影子副本 — build_layer_subgraph 的双算需要逐层 text 权重, 而基类
    // layer_weights_/cos_/sin_/eps_ 都是 private。这里只复制 Tensor 句柄
    // (引用计数), 数据所有权与生命周期仍由基类持有的同一 DDR 张量保证。
    const int64_t N = static_cast<int64_t>(q_w_list.size());
    text_layers_.clear();
    text_layers_.reserve(N);
    const bool bias = (q_bias_list.size() > 0);
    for (int64_t i = 0; i < N; ++i) {
        text_layers_.push_back({
            q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
            q_norm_list[i], k_norm_list[i],
            input_norm_list[i], post_norm_list[i],
            gate_list[i], up_list[i], down_list[i],
            bias ? q_bias_list[i] : at::Tensor(),
            bias ? k_bias_list[i] : at::Tensor(),
            bias ? v_bias_list[i] : at::Tensor(),
        });
    }
    cos_halo_ = cos;
    sin_halo_ = sin;
    eps_halo_ = eps;

    invalidate_model_state();   // 影子提交后再失效一次，保持末语句语义。
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamic_config — 基类断言单 chunk + 准备其 private mask; 再备一份子类可见的
// PreparedMask 供步 4 的 sdpa_dma_mask_to_spm/SDPA 使用 (每 forward 一次)。
// ─────────────────────────────────────────────────────────────────────────────
ModelDynamicConfig HaloActionExpertModel::dynamic_config(const ChunkPlan& plan) {
    ModelDynamicConfig cfg = CausalDecoderModel::dynamic_config(plan);
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "halo dynamic_config: explicit attention mask required");
    prepared_mask_halo_ = sdpa_prepare_mask(
        ctx().attention_mask, /*is_causal=*/false,
        ctx().seq_len, ctx().position + ctx().seq_len,
        sdpa_stable_mask_cache());
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// subclass_chunk_size_valid — HALO act 强制单块语义
//
// HALO act query=18 必须整块进 MASK_2D 非因果 SDPA。基类
// 当前 tiling 是 tile_m=32, grid=1, gqa=6, product=6，满足严格的
// product<=8 keeper。这里仍 override，因为 HALO 行掩码与 KV merge
// 按整块 18 设计：单块
// (cs>=seq_len ⇒ num_chunks==1) 合法，多块非法。当前 envelope 由
// 该约束由 HALO 组件的形状契约决定。
// ─────────────────────────────────────────────────────────────────────────────
bool HaloActionExpertModel::subclass_chunk_size_valid(
    int64_t cs, int64_t seq_len, int64_t /*position*/) const {
    return cs >= seq_len;
}

// ─────────────────────────────────────────────────────────────────────────────
// set_moe_weights — act 孪生权重 + 行掩码绑定
// ─────────────────────────────────────────────────────────────────────────────
void HaloActionExpertModel::set_moe_weights(
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList q_bias_list, at::TensorList k_bias_list,
    at::TensorList v_bias_list,
    const at::Tensor& final_norm_moe_w,
    const at::Tensor& text_row_mask,
    int64_t chunk_size)
{
    const int64_t N = num_layers();
    TORCH_CHECK(N > 0,
                "halo set_moe_weights must follow CausalDecoderModel::set_weights "
                "(text weights first; num_layers comes from them)");
    TORCH_CHECK(chunk_size > 0,
                "halo set_moe_weights: chunk_size must be > 0, got ", chunk_size);

    // 孪生列表长度必须与基类层数一致 — MoT 双权重逐层对应, 不允许部分覆盖。
    auto check_list = [&](const at::TensorList& list, const char* name) {
        TORCH_CHECK(static_cast<int64_t>(list.size()) == N,
                    "halo set_moe_weights: ", name, ".size()=", list.size(),
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

    // HALO action expert 是 Qwen3 系 (带 q/k head-norm); 基类必须已按该结构配置,
    // 否则 build_layer_subgraph 的双 head-norm 相无 text 对应物。
    TORCH_CHECK(has_qk_norm_,
                "halo set_moe_weights: base set_weights must have non-empty "
                "q/k_norm lists (HALO act expert is Qwen3-style)");

    // QKV bias 与 text 侧同进退: 三个列表要么全空 (基类 has_qkv_bias_=false),
    // 要么全 N (true)。不一致 = 主/孪生 ABI 错配, 立即报错。
    const bool moe_bias = (q_bias_list.size() > 0);
    TORCH_CHECK((q_bias_list.size() > 0) == (k_bias_list.size() > 0)
                && (k_bias_list.size() > 0) == (v_bias_list.size() > 0),
                "halo set_moe_weights: q/k/v_bias_list must all be empty or all "
                "size==num_layers; got q=", q_bias_list.size(),
                " k=", k_bias_list.size(), " v=", v_bias_list.size());
    TORCH_CHECK(moe_bias == has_qkv_bias_,
                "halo set_moe_weights: act-twin bias presence (", moe_bias,
                ") must match text-side has_qkv_bias_ (", has_qkv_bias_, ")");
    if (moe_bias) {
        check_list(q_bias_list, "q_bias_list");
        check_list(k_bias_list, "k_bias_list");
        check_list(v_bias_list, "v_bias_list");
    }

    // Act-twin weights are DMA inputs, so require fp16 contiguous RPU tensors.
    // Detailed shapes remain validated by the GEMM launcher.
    auto check_fp16_rpu = [&](const at::TensorList& list, const char* name,
                             int64_t rank) {
        for (int64_t i = 0; i < N; ++i) {
            TORCH_CHECK(list[i].defined(),
                        "halo set_moe_weights: ", name, "[", i, "] undefined");
            TORCH_CHECK(list[i].dim() == rank,
                        "halo set_moe_weights: ", name, "[", i, "] must be ",
                        rank, "D, got ", list[i].dim(), "D");
            TORCH_CHECK(list[i].scalar_type() == at::kHalf
                        && list[i].device().type() == at::kPrivateUse1
                        && list[i].is_contiguous(),
                        "halo set_moe_weights: ", name, "[", i,
                        "] must be fp16 contiguous RPU tensor, got dtype=",
                        list[i].scalar_type(), " device=", list[i].device().type(),
                        " contiguous=", list[i].is_contiguous());
        }
    };
    check_fp16_rpu(q_w_list,        "q_w_list",        2);
    check_fp16_rpu(k_w_list,        "k_w_list",        2);
    check_fp16_rpu(v_w_list,        "v_w_list",        2);
    check_fp16_rpu(o_w_list,        "o_w_list",        2);
    check_fp16_rpu(q_norm_list,     "q_norm_list",     1);
    check_fp16_rpu(k_norm_list,     "k_norm_list",     1);
    check_fp16_rpu(input_norm_list, "input_norm_list", 1);
    check_fp16_rpu(post_norm_list,  "post_norm_list",  1);
    check_fp16_rpu(gate_list,       "gate_list",       2);
    check_fp16_rpu(up_list,         "up_list",         2);
    check_fp16_rpu(down_list,       "down_list",       2);
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
                "halo set_moe_weights: final_norm_moe_w must be 1D [hidden=",
                hidden_size(), "] fp16 contig RPU, got ", final_norm_moe_w.sizes(),
                " dtype=", final_norm_moe_w.scalar_type());

    // 行掩码: [chunk_size] fp16 RPU contig, 取值 {0,1}。
    TORCH_CHECK(text_row_mask.defined() && text_row_mask.dim() == 1
                && text_row_mask.size(0) == chunk_size
                && text_row_mask.scalar_type() == at::kHalf
                && text_row_mask.is_contiguous()
                && text_row_mask.device().type() == at::kPrivateUse1,
                "halo set_moe_weights: text_row_mask must be [", chunk_size,
                "] fp16 contig RPU");
    at::Tensor mask_cpu = text_row_mask.to(at::kCPU).to(at::kFloat);
    TORCH_CHECK(((mask_cpu == 0.0f) | (mask_cpu == 1.0f)).all().item<bool>(),
                "halo set_moe_weights: text_row_mask values must be exactly 0/1");

    // ── 掩码常量化: 按 5 个 merge 宽度展开为 [cs, W] fp16 DDR 常量 ──
    // 宽度即各 SPM 槽的 per-core 行宽 (单位: fp16 元素):
    //   local_q*hd  : q 槽 (col-partition over attn_tp)
    //   local_kv    : k/v 槽 (nkv*hd/tp)
    //   hidden      : input_norm/oproj/down 等全宽广播槽
    //   hidden/8    : h col-partition 槽 (M2 boundary 入模预留)
    //   is/8        : MLP gate/up 槽 (M2 预留; down 使用全宽 merge)
    // act 掩码 = 1 - text 掩码。展开在 host (CPU) 完成后整体复制到 RPU DDR —
    // 不在 RPU device 上做常量填充构造 (Pitfall B-1)。
    const int tp = attn_tp();
    const int64_t widths[5] = {
        (num_q_heads() / tp) * head_dim(),
        num_kv_heads() * head_dim() / tp,
        hidden_size(),
        hidden_size() / NUM_CORES,
        intermediate_size() / NUM_CORES,
    };
    const at::Tensor text_cpu = mask_cpu.to(at::kHalf);
    const at::Tensor act_cpu  = (1.0f - mask_cpu).to(at::kHalf);
    const auto rpu_opts =
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1);
    text_mask_ddr_.clear();
    act_mask_ddr_.clear();
    for (int64_t w : widths) {
        if (text_mask_ddr_.count(w)) continue;   // 宽度去重 (维度巧合相等时)
        text_mask_ddr_[w] = text_cpu.unsqueeze(1).expand({chunk_size, w})
                                .contiguous().to(rpu_opts);
        act_mask_ddr_[w]  = act_cpu.unsqueeze(1).expand({chunk_size, w})
                                .contiguous().to(rpu_opts);
    }

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
        });
    }
    final_norm_moe_w_ = final_norm_moe_w;
    text_row_mask_    = text_row_mask;
    moe_chunk_size_   = chunk_size;

    invalidate_model_state();   // set_moe_weights 末非空语句。
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers — 基类布局 + act 孪生 Temp 槽 + 掩码/act 权重 preload 槽
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BufferDecl> HaloActionExpertModel::declare_buffers(const LayoutContext& lctx) {
    auto d = CausalDecoderModel::declare_buffers(lctx);
    TORCH_CHECK(moe_chunk_size_ > 0,
                "halo declare_buffers: set_moe_weights must precede forward");
    // 框架解析出的 chunk_size 是单块 SPM 规划宽度 (ceil16(seq)=32), 实际行数
    // = chunk.len = seq_len = moe_chunk_size_(18)。基类/act 双算槽按 lctx.chunk_size
    // 分配 (容纳); 行掩码槽按 moe_chunk_size_(=cs_mask) 分配, 与 set_moe_weights
    // 展开的 [18,w] 常量张量及 build_layer_subgraph 的 seq_len 行消费一致。
    // 单块要求 resolved cs 覆盖整个 18 行 (subclass_chunk_size_valid 已保证)。
    TORCH_CHECK(lctx.chunk_size >= moe_chunk_size_,
                "halo declare_buffers: resolved chunk_size=", lctx.chunk_size,
                " < moe chunk_size=", moe_chunk_size_,
                " (HALO act 单块必须覆盖整个 query)");

    const int64_t cs      = lctx.chunk_size;   // 基类/act 双算槽容纳宽度
    const int64_t cs_mask = moe_chunk_size_;   // 行掩码实际行数 (=seq_len=18)
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
    d.push_back({"gate_a",  mlp, 7, 8, StorageClass::Temp, 0, nullptr});
    d.push_back({"up_a",    mlp, 7, 7, StorageClass::Temp, 0, nullptr});
    d.push_back({"down_a",  res, 7, 8, StorageClass::Temp, 0, nullptr});
    d.push_back({"norm_a",  res, 1, 8, StorageClass::Temp, 0, nullptr});

    // ── 行掩码 Persistent 槽 (text/act × 5 宽度) ──
    // 常量掩码整个模型生命周期不变 → Persistent + preload_callback 一次 DMA,
    // REPLAY 零成本。宽度与 set_moe_weights 的展开表一致。
    struct MaskSlot { const char* txt; const char* act; int64_t width; };
    const MaskSlot mask_slots[5] = {
        {"halo_mask_txt_q",    "halo_mask_act_q",    local_q * hd},
        {"halo_mask_txt_kv",   "halo_mask_act_kv",   local_kv},
        {"halo_mask_txt_h",    "halo_mask_act_h",    h},
        {"halo_mask_txt_hcol", "halo_mask_act_hcol", h / NUM_CORES},
        {"halo_mask_txt_mlp",  "halo_mask_act_mlp",  intermediate_size() / NUM_CORES},
    };
    for (const auto& s : mask_slots) {
        const int64_t w = s.width;
        // is_text 决定回调查 text 表还是 act 表; 表按宽度索引 (set_moe_weights
        // 已展开全部 5 个宽度), at() 缺宽即抛 — 防御掩码未重建。
        auto add_mask = [&](const char* name, bool is_text) {
            BufferDecl b;
            b.name    = name;
            b.size    = A(cs_mask * w * DWIDTH);
            b.storage = StorageClass::Persistent;
            b.scope   = BufferScope::LayerWide;
            b.preload_callback =
                [this, is_text, w, cs_mask](FusedModelBase&, int, uint32_t core0_addr) {
                    const auto& tbl = is_text ? text_mask_ddr_ : act_mask_ddr_;
                    rpu_launch_ddr_broadcast_spm_dma(
                        tbl.at(w).data_ptr<c10::Half>(), cs_mask * w, core0_addr,
                        NUM_CORES);
                };
            d.push_back(b);
        };
        add_mask(s.txt, /*is_text=*/true);
        add_mask(s.act, /*is_text=*/false);
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

    {
        BufferDecl fn;
        fn.name    = "final_norm_moe_w";
        fn.size    = nw;
        fn.storage = StorageClass::Persistent;
        fn.scope   = BufferScope::LayerWide;
        fn.preload_callback =
            [this](FusedModelBase&, int, uint32_t core0_addr) {
                rpu_launch_ddr_broadcast_spm_dma(
                    final_norm_moe_w_.data_ptr<c10::Half>(),
                    hidden_size(), core0_addr);
            };
        d.push_back(fn);
    }
    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_row_merge — out = text*text_mask + act*act_mask (3 次 eltwise)
// 就地语义: MUL 覆写 text/act 两个输入槽; out 与 text 同槽是常态 (text 槽即基类
// 流水的下游输入)。掩码槽 Persistent, 不被覆写。
// ─────────────────────────────────────────────────────────────────────────────
void HaloActionExpertModel::emit_row_merge(
    uint32_t text_addr, uint32_t act_addr, uint32_t out_addr,
    uint32_t text_mask_addr, uint32_t act_mask_addr,
    int64_t num_elements, int num_cores)
{
    rpu_launch_eltwise_binary_spm_kernel(
        text_addr, text_mask_addr, text_addr,
        num_elements, ValuOpType::MUL, c10::Half(1.0f), num_cores);
    rpu_launch_eltwise_binary_spm_kernel(
        act_addr, act_mask_addr, act_addr,
        num_elements, ValuOpType::MUL, c10::Half(1.0f), num_cores);
    rpu_launch_eltwise_binary_spm_kernel(
        text_addr, act_addr, out_addr,
        num_elements, ValuOpType::ADD, c10::Half(1.0f), num_cores);
}

// ─────────────────────────────────────────────────────────────────────────────
// build_layer_subgraph — 基类 8 相的 MoT 双算版 (规格 7 步)
//
// 行独立性论证: RMSNorm/GEMM/bias/SiLU 全部逐行运算, 故"双算全 cs 行后按行掩码
// 合并"与逐行选分支严格等价; SDPA 是唯一跨行算子, 但 q/k/v 在其之前已按行合并,
// 与 HALO 参考语义一致。RoPE 行位置只依赖 position+offset, 与分支无关 → 合并后
// 单次 in-place。
// ─────────────────────────────────────────────────────────────────────────────
void HaloActionExpertModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    TORCH_CHECK(!text_layers_.empty() && !moe_layers_.empty(),
                "halo build_layer_subgraph: set_weights_halo + set_moe_weights "
                "must both have been called");
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "halo build_layer_subgraph: explicit 2D mask required "
                "(act step is non-causal prefix attention)");
    TORCH_CHECK(chunk.len == moe_chunk_size_,
                "halo build_layer_subgraph: chunk.len=", chunk.len,
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
    const int64_t elems_h   = seq_len * h;
    const int64_t elems_q   = seq_len * local_q * hd;
    const int64_t elems_kv  = seq_len * local_kv;
    const int64_t elems_mlp = seq_len * (intermediate_size() / NUM_CORES);
    const uint32_t txt_q  = addr(0, "halo_mask_txt_q"),  act_q  = addr(0, "halo_mask_act_q");
    const uint32_t txt_kv = addr(0, "halo_mask_txt_kv"), act_kv = addr(0, "halo_mask_act_kv");
    const uint32_t txt_h  = addr(0, "halo_mask_txt_h"),  act_h  = addr(0, "halo_mask_act_h");

    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }

    // ── 步 1: input RMSNorm 双算 → merge 入 "input_norm" ──
    // text γ 经基类 "norm_w" 槽, act γ 经 "norm_w_act"。merge 后 input_norm 持
    // 逐行选支的 normed hidden, 供两套 QKV 共读 (行独立 ⇒ 先 merge 再 GEMM 等价)。
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "norm_w"), seq_len, h, eps_halo_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "norm_a"),
        layer_addr(layer_idx, 0, "norm_w_act"), seq_len, h, eps_halo_);
    emit_row_merge(addr(0, "input_norm"), addr(0, "norm_a"), addr(0, "input_norm"),
                   txt_h, act_h, elems_h, NUM_CORES);

    // ── 步 2: QKV 双发射 (text→q/k/v, act→q_a/k_a/v_a) + merge ──
    // 双方读同一 merged input_norm; bias 各用自家 PersistentPerLayer 槽。
    auto qkv = [&](const at::Tensor& w, const char* out, const char* bias_slot,
                   int64_t n) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), w, addr(0, out),
            seq_len, n, h, /*partition=*/1, /*num_cores=*/tp,
            has_qkv_bias_ ? layer_addr(layer_idx, 0, bias_slot) : 0u,
            /*force_gemm=*/false, /*scale=*/{});
    };
    qkv(lw.q_w, "q",   "q_bias",     nq * hd);
    qkv(lw.k_w, "k",   "k_bias",     nkv * hd);
    qkv(lw.v_w, "v",   "v_bias",     nkv * hd);
    qkv(mw.q_w, "q_a", "q_bias_act", nq * hd);
    qkv(mw.k_w, "k_a", "k_bias_act", nkv * hd);
    qkv(mw.v_w, "v_a", "v_bias_act", nkv * hd);
    emit_row_merge(addr(0, "q"), addr(0, "q_a"), addr(0, "q"), txt_q,  act_q,  elems_q,  tp);
    emit_row_merge(addr(0, "k"), addr(0, "k_a"), addr(0, "k"), txt_kv, act_kv, elems_kv, tp);
    emit_row_merge(addr(0, "v"), addr(0, "v_a"), addr(0, "v"), txt_kv, act_kv, elems_kv, tp);

    // ── 步 3: q/k head-norm 双算 + merge, RoPE 在 merge 后单次 in-place ──
    // head 维行排布 [seq*local_heads, hd] 与掩码 [seq, local_heads*hd] 同一段
    // 内存 → q/kv 宽度掩码可直接复用。norm 暂存: text→output/input_norm (基类
    // 同槽用法), act→norm_a (1..8 槽, 步 1 之后空闲)。
    c10::Half* cos_ptr = cos_halo_.data_ptr<c10::Half>();
    c10::Half* sin_ptr = sin_halo_.data_ptr<c10::Half>();
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "q"), addr(0, "output"),
        layer_addr(layer_idx, 0, "q_norm_w"), seq_len * local_q, hd, eps_halo_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "q"), addr(0, "norm_a"),
        layer_addr(layer_idx, 0, "q_norm_w_act"), seq_len * local_q, hd, eps_halo_);
    emit_row_merge(addr(0, "output"), addr(0, "norm_a"), addr(0, "output"),
                   txt_q, act_q, elems_q, tp);
    rpu_launch_rope_spm_kernel(addr(0, "output"), addr(0, "q"),
                               cos_ptr, sin_ptr, seq_len, local_q, hd, cos_sin_start);

    const int64_t local_kv_heads = nkv / tp;
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "k"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "k_norm_w"), seq_len * local_kv_heads, hd, eps_halo_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "k"), addr(0, "norm_a"),
        layer_addr(layer_idx, 0, "k_norm_w_act"), seq_len * local_kv_heads, hd, eps_halo_);
    emit_row_merge(addr(0, "input_norm"), addr(0, "norm_a"), addr(0, "input_norm"),
                   txt_kv, act_kv, elems_kv, tp);
    rpu_launch_rope_spm_kernel(addr(0, "input_norm"), addr(0, "k"),
                               cos_ptr, sin_ptr, seq_len, local_kv_heads, hd, cos_sin_start);

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
    sdpa_dma_mask_to_spm(prepared_mask_halo_, mask_off, seq_len, chunk.kv_seq_len, tp);
    rpu_launch_sdpa_spm_dispatch(
        SdpaKernelType::FLASH_ATTN_SPM,
        k_cache, v_cache, prepared_mask_halo_.mask_type, c10::nullopt,
        addr_offset("q").value, addr_offset("output").value,
        addr_offset("sdpa_tmp").value, mask_off,
        seq_len, nq, nkv, hd, chunk.kv_seq_len, tp, NUM_CORES);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0, /*force_gemm=*/false, /*scale=*/{});
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), mw.o_w, addr(0, "oproj_a"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0, /*force_gemm=*/false, /*scale=*/{});
    emit_row_merge(addr(0, "oproj"), addr(0, "oproj_a"), addr(0, "oproj"),
                   txt_h, act_h, elems_h, tp);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
        seq_len, h, tp, NUM_CORES);

    // ── 步 5: post-attention RMSNorm 双算 + merge → residual1 ──
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "residual1"),
        layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, eps_halo_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "norm_a"),
        layer_addr(layer_idx, 0, "post_norm_w_act"), seq_len, h, eps_halo_);
    emit_row_merge(addr(0, "residual1"), addr(0, "norm_a"), addr(0, "residual1"),
                   txt_h, act_h, elems_h, NUM_CORES);

    // ── 步 6: 手写双 MLP (SwiGLU) → down merge → 残差归约 ──
    // 不用基类 emit_mlp_pipeline: 它把 all_reduce_sum_residual 写死在 down 后,
    // 而双分支必须在归约前合并 (act 分支用独立 gate_a/up_a/down_a 槽)。
    auto mlp_half = [&](const at::Tensor& gate_w, const at::Tensor& up_w,
                        const at::Tensor& down_w, const char* g, const char* u,
                        const char* dn) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), gate_w, addr(0, g),
            seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
            /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/{});
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, g), addr(0, g), elems_mlp, ValuOpType::SILU,
            /*is_gelu=*/false, NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), up_w, addr(0, u),
            seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
            /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/{});
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, g), addr(0, u), addr(0, g),
            elems_mlp, ValuOpType::MUL, c10::Half(1.0f), NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, g), down_w, addr(0, dn),
            seq_len, h, intermediate_size(), /*partition=*/0, NUM_CORES,
            /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/{});
    };
    mlp_half(lw.gate_w, lw.up_w, lw.down_w, "gate",   "up",   "down");
    mlp_half(mw.gate_w, mw.up_w, mw.down_w, "gate_a", "up_a", "down_a");
    emit_row_merge(addr(0, "down"), addr(0, "down_a"), addr(0, "down"),
                   txt_h, act_h, elems_h, NUM_CORES);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
        seq_len, h, NUM_CORES, NUM_CORES);

    // ── 步 7 (末层): final RMSNorm 双算 + merge → residual1 ──
    // text γ 经基类 "final_norm_w" 槽, act γ 经 "final_norm_moe_w"。暂存复用
    // down (7..8, 归约后空闲) + norm_a。
    if (is_last_layer) {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "down"),
            addr(0, "final_norm_w"), seq_len, h, eps_halo_);
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "norm_a"),
            addr(0, "final_norm_moe_w"), seq_len, h, eps_halo_);
        emit_row_merge(addr(0, "down"), addr(0, "norm_a"), addr(0, "residual1"),
                       txt_h, act_h, elems_h, NUM_CORES);
    }

    if (!ctx().output_to_spm) {
        emit_layer_output_dma(layer_idx, chunk);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// step_forward — 一次 act step (prefix-KV 非因果 forward + hidden 输出)
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor HaloActionExpertModel::step_forward(
    const at::Tensor& x_emb, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len)
{
    TORCH_CHECK(!text_layers_.empty() && moe_chunk_size_ > 0,
                "halo step_forward before set_weights_halo / set_moe_weights");
    const int64_t cs = moe_chunk_size_;
    const int64_t h  = hidden_size();
    TORCH_CHECK(x_emb.dim() == 3 && x_emb.size(0) == 1 && x_emb.size(1) == cs
                && x_emb.size(2) == h && x_emb.scalar_type() == at::kHalf
                && x_emb.is_contiguous()
                && x_emb.device().type() == at::kPrivateUse1,
                "halo step_forward: x_emb must be [1,", cs, ",", h,
                "] fp16 contig RPU");
    TORCH_CHECK(attn_mask_4d.defined(),
                "halo step_forward: explicit attention mask required");
    TORCH_CHECK(prefix_len >= 0, "halo step_forward: prefix_len must be >= 0");

    // per-forward cos/sin (镜像 qwenpi05 cos_ref_/sin_ref_): 覆盖 RoPE 影子成员,
    // 赋值即保活过 graph capture; build_layer_subgraph (:525-548) 读 cos_halo_/sin_halo_
    // 的 raw c10::Half*。CFG 三分支共用一个句柄, 每分支传自己 rope_position 的表。
    // 校验范式对齐 qwenpi05_forward (kernel 按 raw c10::Half* 消费)。
    TORCH_CHECK(cos.defined() && sin.defined(),
                "halo step_forward: cos/sin must be defined");
    TORCH_CHECK(cos.device().type() == at::kPrivateUse1
                && sin.device().type() == at::kPrivateUse1,
                "halo step_forward: cos/sin must be on RPU");
    TORCH_CHECK(cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf,
                "halo step_forward: cos/sin must be fp16");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
                "halo step_forward: cos/sin must be contiguous");
    TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes(),
                "halo step_forward: cos/sin must be 2-D, equal shape; cos=",
                cos.sizes(), " sin=", sin.sizes());
    TORCH_CHECK(cos.size(1) == head_dim() / 2,
                "halo step_forward: cos.size(1) must be head_dim/2=", head_dim() / 2,
                ", got ", cos.size(1));
    TORCH_CHECK(cos.size(0) >= prefix_len + cs,
                "halo step_forward: cos row count (", cos.size(0),
                ") < prefix_len+cs (", prefix_len + cs, ")");
    cos_halo_ = cos;
    sin_halo_ = sin;

    // KV headroom: 7-D K 布局的 max_seq = size(1)*size(5)。
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + cs <= kc.size(1) * kc.size(5),
                    "halo step_forward: prefix_len(", prefix_len, ")+cs(", cs,
                    ") exceeds k_cache capacity ", kc.size(1) * kc.size(5));
    }

    // HALO act 必须单块 (MASK_2D 行掩码/KV merge 按整块 18 设计)。chunk override
    // 设为 ceil16(seq=18)=32: compute_chunks 走 override 路径, 切出单块
    // len=min(32,18)=18 (subclass_chunk_size_valid 放行)。
    // 不能设 18 — override 路径 (override/16)*16 会向下取整成 16。
    //
    // Store the override directly on this handle.
    const int64_t single_cs = ((moe_chunk_size_ + 15) / 16) * 16;
    set_chunk_size_override(single_cs);

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
// Instance registry — ModelHandleRegistry<v3::HaloActionExpertModel>
// =============================================================================
using HaloActionExpertRegistry = ModelHandleRegistry<v3::HaloActionExpertModel>;

// =============================================================================
// Public C API for TORCH_LIBRARY wrappers (file-scope; decls in rpu_kernel_decls.h)
// =============================================================================
int64_t rpu_halo_action_expert_create() {
    return HaloActionExpertRegistry::create();
}

void rpu_halo_action_expert_destroy(int64_t handle) {
    HaloActionExpertRegistry::destroy(handle, "rpu_halo_action_expert_destroy");
}

// text 主权重: 单 schema 包装 set_weights_halo (内部转基类 set_weights,
// mrope/deepstack 空; q/k norm 与 QKV bias 走 Qwen3+bias 路径, fp16-only)。
void rpu_halo_action_expert_set_weights(
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
    HaloActionExpertRegistry::get(handle, "rpu_halo_action_expert_set_weights")
        ->set_weights_halo(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, use_silu, q_bias, k_bias, v_bias);
}

void rpu_halo_action_expert_set_moe_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    const at::Tensor& final_norm_moe_w, const at::Tensor& text_row_mask,
    int64_t chunk_size)
{
    HaloActionExpertRegistry::get(handle, "rpu_halo_action_expert_set_moe_weights")
        ->set_moe_weights(q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm,
                          post_norm, gate_w, up_w, down_w, q_bias, k_bias,
                          v_bias, final_norm_moe_w, text_row_mask, chunk_size);
}

at::Tensor rpu_halo_action_expert_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len)
{
    return HaloActionExpertRegistry::get(handle, "rpu_halo_action_expert_step_forward")
        ->step_forward(x_emb, k_caches, v_caches, cos, sin, attn_mask_4d, prefix_len);
}
