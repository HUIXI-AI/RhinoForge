// rpu_halo_image_flow_model.cpp — HALO image flow denoise step (mode="gen")。
//
// 组合 = action expert 双流 MoT(每层双算 + 行掩码合并)⊕ GemmaModel KV_FIRST
// 两阶段(多块非因果: Phase 1 插全部 K/V, Phase 2 每块对满 KV SDPA)。query =
// 1378 行 = 2 text(start/end_of_image, packed_text_indexes=[0,1377]) 走 und/base
// 孪生 + 1376 vae 走 gen(*_moe_gen) 孪生; per-twin 投影后合并成单一 Q/K/V →
// 单次 RoPE → 单次全 attend SDPA(MASK_NONE)。行掩码逐 chunk 按 chunk.offset 显式
// body 内 DMA(镜像 Gemma gemma_upload_chunk_pm; preload_callback 无 chunk.offset)。
// emit_row_merge / mlp_half 与 action expert 使用相同机制。
//
// 正确性必须由板上端到端验证，单独编译通过不足以证明功能正确。
#include "rpu_halo_image_flow_model.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"

#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <limits>
#include <vector>

namespace v3 {

HaloImageFlowModel::HaloImageFlowModel()  = default;
HaloImageFlowModel::~HaloImageFlowModel() = default;

// ─────────────────────────────────────────────────────────────────────────────
// set_weights_image_flow — und/base 主权重 (基类委托 + 子类影子)
// ─────────────────────────────────────────────────────────────────────────────
void HaloImageFlowModel::set_weights_image_flow(
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
    TORCH_CHECK(use_silu,
                "halo image_flow set_weights: use_silu must be true (HALO MLP is SwiGLU)");
    // The base CausalDecoderModel::set_weights checks
    // defined/rank/shape but NOT dtype/device/contiguous, yet these und tensors
    // are mirrored into und_layers_ and DMA'd via data_ptr<c10::Half>(). Validate
    // fp16 + PrivateUse1 + contiguous at the subclass entry (mirrors set_gen_weights
    // check_rank) before delegating, leaving the shared base setter untouched.
    auto check_fp16_rpu = [&](const at::TensorList& list, const char* name, int64_t rank) {
        for (size_t i = 0; i < list.size(); ++i) {
            TORCH_CHECK(list[i].defined() && list[i].dim() == rank,
                        "halo image_flow set_weights: ", name, "[", i, "] must be ",
                        rank, "D, got ", list[i].defined() ? list[i].dim() : -1, "D");
            TORCH_CHECK(list[i].scalar_type() == at::kHalf
                        && list[i].device().type() == at::kPrivateUse1
                        && list[i].is_contiguous(),
                        "halo image_flow set_weights: ", name, "[", i,
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
    if (q_bias_list.size() > 0) {
        check_fp16_rpu(q_bias_list, "q_bias_list", 1);
        check_fp16_rpu(k_bias_list, "k_bias_list", 1);
        check_fp16_rpu(v_bias_list, "v_bias_list", 1);
    }
    auto check_global = [&](const at::Tensor& t, const char* name, int64_t rank) {
        TORCH_CHECK(t.defined() && t.dim() == rank
                    && t.scalar_type() == at::kHalf
                    && t.device().type() == at::kPrivateUse1 && t.is_contiguous(),
                    "halo image_flow set_weights: ", name,
                    " must be ", rank, "D fp16 contiguous RPU tensor");
    };
    check_global(cos,          "cos",          2);
    check_global(sin,          "sin",          2);
    check_global(final_norm_w, "final_norm_w", 1);
    // 校验/状态提交全部委托基类 (含 q/k norm + bias 全套检查; mrope/deepstack 空)。
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

    // 影子副本 — KV_FIRST 两阶段 build 的双算需逐层 und 权重 (基类 layer_weights_/
    // cos_/sin_/eps_ 都是 private)。仅复制 Tensor 句柄 (引用计数), 数据所有权
    // 与生命周期由基类持有的同一 DDR 张量保证。喂入无后缀 base/und 孪生。
    const int64_t N = static_cast<int64_t>(q_w_list.size());
    und_layers_.clear();
    und_layers_.reserve(N);
    has_qkv_bias_halo_ = (q_bias_list.size() > 0);
    for (int64_t i = 0; i < N; ++i) {
        und_layers_.push_back({
            q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
            q_norm_list[i], k_norm_list[i],
            input_norm_list[i], post_norm_list[i],
            gate_list[i], up_list[i], down_list[i],
            has_qkv_bias_halo_ ? q_bias_list[i] : at::Tensor(),
            has_qkv_bias_halo_ ? k_bias_list[i] : at::Tensor(),
            has_qkv_bias_halo_ ? v_bias_list[i] : at::Tensor(),
        });
    }
    cos_halo_ = cos;
    sin_halo_ = sin;
    final_norm_w_halo_ = final_norm_w;
    eps_halo_ = eps;

    invalidate_model_state();
}

// ─────────────────────────────────────────────────────────────────────────────
// set_gen_weights — gen(*_moe_gen) 孪生权重 + 行掩码绑定
// (= action expert set_moe_weights 角色; 减去 chunk_size SDPA override —— image_flow
//  的 SDPA 块计划来自 plan_kv_first_chunks, 非本 setter)。
// ─────────────────────────────────────────────────────────────────────────────
void HaloImageFlowModel::set_gen_weights(
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList q_bias_list, at::TensorList k_bias_list,
    at::TensorList v_bias_list,
    const at::Tensor& final_norm_gen_w,
    const at::Tensor& row_mask,
    int64_t query_len)
{
    const int64_t N = num_layers();
    TORCH_CHECK(N > 0,
                "halo image_flow set_gen_weights must follow set_weights_image_flow "
                "(und weights first; num_layers comes from them)");
    TORCH_CHECK(query_len > 0,
                "halo image_flow set_gen_weights: query_len must be > 0, got ", query_len);

    auto check_list = [&](const at::TensorList& list, const char* name) {
        TORCH_CHECK(static_cast<int64_t>(list.size()) == N,
                    "halo image_flow set_gen_weights: ", name, ".size()=", list.size(),
                    " != num_layers=", N);
    };
    check_list(q_w_list,        "q_w_list");
    check_list(k_w_list,        "k_w_list");
    check_list(v_w_list,        "v_w_list");
    check_list(o_w_list,        "o_w_list");
    check_list(q_norm_list,     "q_norm_list");
    check_list(k_norm_list,     "k_norm_list");
    check_list(input_norm_list, "input_norm_list");
    check_list(post_norm_list,  "post_norm_list");
    check_list(gate_list,       "gate_list");
    check_list(up_list,         "up_list");
    check_list(down_list,       "down_list");

    // HALO gen 孪生是 Qwen3 系 (带 q/k head-norm); base 必须已按该结构配置, 否则
    // 双 head-norm 相无 und 对应物。
    TORCH_CHECK(has_qk_norm_,
                "halo image_flow set_gen_weights: base set_weights must have "
                "non-empty q/k_norm lists (HALO gen twin is Qwen3-style)");

    // QKV bias 与 und 侧同进退: 三列表要么全空 (has_qkv_bias_=false) 要么全 N。
    const bool gen_bias = (q_bias_list.size() > 0);
    TORCH_CHECK((q_bias_list.size() > 0) == (k_bias_list.size() > 0)
                && (k_bias_list.size() > 0) == (v_bias_list.size() > 0),
                "halo image_flow set_gen_weights: q/k/v_bias_list must all be empty "
                "or all size==num_layers; got q=", q_bias_list.size(),
                " k=", k_bias_list.size(), " v=", v_bias_list.size());
    TORCH_CHECK(gen_bias == has_qkv_bias_,
                "halo image_flow set_gen_weights: gen-twin bias presence (", gen_bias,
                ") must match und-side has_qkv_bias_ (", has_qkv_bias_, ")");
    if (gen_bias) {
        check_list(q_bias_list, "q_bias_list");
        check_list(k_bias_list, "k_bias_list");
        check_list(v_bias_list, "v_bias_list");
    }

    auto check_rank = [&](const at::TensorList& list, const char* name, int64_t rank) {
        for (int64_t i = 0; i < N; ++i) {
            TORCH_CHECK(list[i].defined(),
                        "halo image_flow set_gen_weights: ", name, "[", i, "] undefined");
            TORCH_CHECK(list[i].dim() == rank,
                        "halo image_flow set_gen_weights: ", name, "[", i, "] must be ",
                        rank, "D, got ", list[i].dim(), "D");
            // Gen tensors feed DDR DMA (data_ptr<c10::Half>) + fp16
            // kernels in declare_buffers/build — validate dtype/device/contiguous like
            // row_mask, else a CPU/bf16/non-contiguous tensor records an invalid device
            // address or is fp16-misread.
            TORCH_CHECK(list[i].scalar_type() == at::kHalf
                        && list[i].device().type() == at::kPrivateUse1
                        && list[i].is_contiguous(),
                        "halo image_flow set_gen_weights: ", name, "[", i,
                        "] must be fp16 + RPU(PrivateUse1) + contiguous, got dtype=",
                        list[i].scalar_type(), " device=", list[i].device().type(),
                        " contiguous=", list[i].is_contiguous());
        }
    };
    check_rank(q_w_list,        "q_w_list",        2);
    check_rank(k_w_list,        "k_w_list",        2);
    check_rank(v_w_list,        "v_w_list",        2);
    check_rank(o_w_list,        "o_w_list",        2);
    check_rank(q_norm_list,     "q_norm_list",     1);
    check_rank(k_norm_list,     "k_norm_list",     1);
    check_rank(input_norm_list, "input_norm_list", 1);
    check_rank(post_norm_list,  "post_norm_list",  1);
    check_rank(gate_list,       "gate_list",       2);
    check_rank(up_list,         "up_list",         2);
    check_rank(down_list,       "down_list",       2);
    if (gen_bias) {
        check_rank(q_bias_list, "q_bias_list", 1);
        check_rank(k_bias_list, "k_bias_list", 1);
        check_rank(v_bias_list, "v_bias_list", 1);
    }

    TORCH_CHECK(final_norm_gen_w.defined() && final_norm_gen_w.dim() == 1
                && final_norm_gen_w.size(0) == hidden_size()
                && final_norm_gen_w.scalar_type() == at::kHalf
                && final_norm_gen_w.device().type() == at::kPrivateUse1
                && final_norm_gen_w.is_contiguous(),
                "halo image_flow set_gen_weights: final_norm_gen_w must be 1D [hidden=",
                hidden_size(), "] fp16 + RPU(PrivateUse1) + contiguous, got ",
                final_norm_gen_w.sizes(), " dtype=", final_norm_gen_w.scalar_type());

    // 行掩码: [query_len] fp16 RPU contig, 取值 {0,1} (1 = und/text 行)。
    TORCH_CHECK(row_mask.defined() && row_mask.dim() == 1
                && row_mask.size(0) == query_len
                && row_mask.scalar_type() == at::kHalf
                && row_mask.is_contiguous()
                && row_mask.device().type() == at::kPrivateUse1,
                "halo image_flow set_gen_weights: row_mask must be [", query_len,
                "] fp16 contig RPU");
    at::Tensor mask_cpu = row_mask.to(at::kCPU).to(at::kFloat);
    TORCH_CHECK(((mask_cpu == 0.0f) | (mask_cpu == 1.0f)).all().item<bool>(),
                "halo image_flow set_gen_weights: row_mask values must be exactly 0/1");

    // ── 行掩码常量化: [query_len,1] fp16 DDR (und=行掩码, gen=1-行掩码) ──
    // merge 用 Nx1_NxC_spm 把 [n,1] 行掩码广播到任意宽度 (q=local_q*hd / kv / h) →
    // 单一 [query_len,1] 行掩码可广播到任意目标宽度。
    // 展开在 host (CPU) 后整体复制到 RPU DDR (Pitfall B-1: 不在 device 上构造常量);
    // body 内按 chunk 行切片 DMA 进 SPM (preload_callback 无 chunk.offset)。
    // DMA 16-byte 对齐: 行掩码上传字节 = len*2 须 %16 → 行数 padto 8 的倍数 (tail chunk
    // 的 len 非 8 倍数时读到 padding 行; merge 只用 seq_len 真行, padding=0 不参与)。
    const auto rpu_opts =
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1);
    const int64_t pad_len = ((query_len + 7) / 8) * 8;
    auto pad_row_mask = [&](const at::Tensor& m1d) {           // m1d: [query_len] fp16
        at::Tensor p = at::zeros({pad_len, 1},
                                 at::TensorOptions().dtype(at::kHalf));
        p.narrow(0, 0, query_len).copy_(m1d.unsqueeze(1));
        return p.contiguous().to(rpu_opts);                    // [pad_len,1] RPU
    };
    und_mask_row_ = pad_row_mask(mask_cpu.to(at::kHalf));
    gen_mask_row_ = pad_row_mask((1.0f - mask_cpu).to(at::kHalf));

    // ── 提交状态 (DDR keepalive: 跨 forward 持有, 直至下次 set_gen_weights) ──
    gen_layers_.clear();
    gen_layers_.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
        gen_layers_.push_back({
            q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
            q_norm_list[i], k_norm_list[i],
            input_norm_list[i], post_norm_list[i],
            gate_list[i], up_list[i], down_list[i],
            gen_bias ? q_bias_list[i] : at::Tensor(),
            gen_bias ? k_bias_list[i] : at::Tensor(),
            gen_bias ? v_bias_list[i] : at::Tensor(),
        });
    }
    final_norm_gen_w_ = final_norm_gen_w;
    row_mask_         = row_mask;
    merge_seq_len_    = query_len;

    invalidate_model_state();
}

// ─────────────────────────────────────────────────────────────────────────────
// static_config — KV_FIRST 两阶段派发注册 (镜像 Gemma)
// ─────────────────────────────────────────────────────────────────────────────
ModelStaticConfig HaloImageFlowModel::static_config() {
    ModelStaticConfig cfg;
    cfg.num_layers = num_layers();
    // 使用单个 cross-layer group。如果单段超出 Launch 批量条目限额，
    // prepare_segment_queue 会检查 build_batch 返回值并立即报错。
    cfg.cross_layer_batch_size = num_layers();
    cfg.preload_fn = nullptr;                     // 用 per-buffer preload_callback
    cfg.kv_first_fn =
        static_cast<void (FusedModelBase::*)(int, const ChunkInfo&)>(
            &HaloImageFlowModel::emit_kv_first_body);
    cfg.kv_first_chunk_plan_fn =
        static_cast<ChunkPlan (FusedModelBase::*)(const ChunkPlan&)>(
            &HaloImageFlowModel::plan_kv_first_chunks);
    cfg.post_fn = nullptr;
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamic_config — 始终 KV_FIRST (image gen 双向)。image gen 全 attend ⇒
// MASK_NONE, 无 SDPA per-chunk mask 准备。行掩码 ≠ SDPA mask, 经 declare_buffers
// 槽 + body 内 per-chunk DMA。
// ─────────────────────────────────────────────────────────────────────────────
ModelDynamicConfig HaloImageFlowModel::dynamic_config(const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode = ChunkMode::KV_FIRST;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// subclass_chunk_size_valid — VLM SDPA tile 约束门，按实际 kernel mask 校验。
//
// Image generation uses non-causal MASK_NONE, so validation must use mask=0
// and exclude LTM-only position constraints. The remaining tile, VLM and
// product constraints admit the multi-chunk KV_FIRST execution selected by the
// SPM budget planner.
// MASK_NONE 无 LTM 的 acc/tile_k 约束: sdpa_call_valid:330-335 注释自证 (单 token
// decode 同理跳过）。正确性边界仍由板上数值门限（cos≥0.999）判定，而不是
// 编译/校验通过 —— L2≠功能对。
// ─────────────────────────────────────────────────────────────────────────────
bool HaloImageFlowModel::subclass_chunk_size_valid(
    int64_t cs, int64_t seq_len, int64_t position) const {
    return sdpa_is_valid_chunk_size(make_sdpa_config(/*mask=*/0), cs, seq_len, position);
}

// ─────────────────────────────────────────────────────────────────────────────
// plan_kv_first_chunks — KV-insert 相 (无 SDPA/MLP) 用更大块 (镜像 Gemma)
// ─────────────────────────────────────────────────────────────────────────────
ChunkPlan HaloImageFlowModel::plan_kv_first_chunks(const ChunkPlan& compute_plan) {
    // R5 (SPM 预算): 双流双算已把 COMP 池 MLP/oproj 槽翻倍 (gate/up/down/oproj 各 ×2)。
    // 若再加倍 kv_cs, declare_buffers 的 wide_cs=max(comp_cs,kv_cs) 翻倍 → LayerWide
    // (residual1/input_norm/norm_g/h-mask) 峰值翻倍; 更糟: auto chunk-size scan 用
    // kv_insert_chunk_size 未设 (wide_cs=comp_cs) 估值, 而 step-7 重配按 kv_cs (=2*comp_cs)
    // 分配 → 两阶段不一致, ensure_allocated 不复检 SPM_USABLE → 越界 OOM
    // (rpu_spm_allocator TORCH_CHECK)。故暂禁用 kv_cs 加倍: Phase 1 与 Phase 2 同
    // comp_cs 分块, scan 自洽找到合身 comp_cs。预算感知的加倍 (仅当 2*comp_cs 的
    // declare_buffers 峰值仍 ≤ SPM_PLANNING_BUDGET 才倍) 列为后续优化。
    return compute_plan;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_row_merge — out = und*und_mask + gen*gen_mask (行掩码广播版)
// und/gen_mask 是单一行掩码 [n,1] (uploadDDR broadcast 到全 8 核); Nx1_NxC_spm 把它
// 广播乘到数据 [n,c] (全 8 核 broadcast 模式, 用到的 tp/NUM_CORES 核结果正确, 其余核
// 无害 superset)。就地 MUL 覆写 und/gen 输入槽; 末 ADD sameshape (n*c) 仍按 num_cores。
// out 与 und 同槽是常态 (und 槽即基类流水的下游输入)。
// ─────────────────────────────────────────────────────────────────────────────
void HaloImageFlowModel::emit_row_merge(
    uint32_t und_addr, uint32_t gen_addr, uint32_t out_addr,
    uint32_t und_mask_addr, uint32_t gen_mask_addr,
    int64_t n, int64_t c, int num_cores)
{
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        und_mask_addr, und_addr, und_addr, n, c,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        gen_mask_addr, gen_addr, gen_addr, n, c,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_eltwise_binary_spm_kernel(
        und_addr, gen_addr, out_addr,
        n * c, ValuOpType::ADD, c10::Half(1.0f), num_cores);
}

// ─────────────────────────────────────────────────────────────────────────────
// upload_merge_masks — 把行掩码 [query_len,1] 的 [chunk.offset:offset+len) 行切片
// 显式 DMA 进 2 个行掩码槽 (und/gen_mask_row, 全 8 核 broadcast)。preload_callback
// 不知 chunk.offset → 必须在 body 内做 (镜像 Gemma gemma_upload_chunk_pm)。行优先
// [query_len,1] → 行切片起点 = data_ptr + offset, 元素数 = len。两相用同一行掩码
// (Nx1_NxC 广播到任意宽度), 故 h_only 不再分槽 (保留签名兼容调用方)。BUILD 期把
// chunk.offset 烤进 DMA 源指针 → REPLAY 同 offset (1378 行结构每步不变) 正确。
// ─────────────────────────────────────────────────────────────────────────────
void HaloImageFlowModel::upload_merge_masks(const ChunkInfo& chunk, bool /*h_only*/) {
    const int64_t off = chunk.offset;
    // DMA 16-byte 对齐: 上传行数上取整到 8 的倍数 (chunk.offset 是 cs 倍数 ⇒ %8; pad_len
    // 保证 off+len_up ≤ pad_len)。SPM 槽 ≥ wide_cs 行容纳; merge 只读 chunk.len 真行。
    const int64_t len_up = ((chunk.len + 7) / 8) * 8;
    TORCH_CHECK(off + len_up <= und_mask_row_.size(0),
                "halo image_flow upload_merge_masks: off+len_up (", off + len_up,
                ") > row mask pad_len (", und_mask_row_.size(0), ")");
    rpu_launch_ddr_broadcast_spm_dma(
        und_mask_row_.data_ptr<c10::Half>() + off, len_up,
        addr(0, "und_mask_row"), NUM_CORES);
    rpu_launch_ddr_broadcast_spm_dma(
        gen_mask_row_.data_ptr<c10::Half>() + off, len_up,
        addr(0, "gen_mask_row"), NUM_CORES);
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers — KV_FIRST SPM 布局 (KVIN/COMP 相位别名) + und/gen 双算槽 +
// 行掩码槽 (body 内填) + und/gen Qwen2.5 norm/bias preload 槽。无 sdpa_mask 槽
// (MASK_NONE 全 attend)。
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BufferDecl> HaloImageFlowModel::declare_buffers(const LayoutContext& ctx) {
    TORCH_CHECK(!und_layers_.empty() && !gen_layers_.empty(),
                "halo image_flow declare_buffers: set_weights_image_flow + "
                "set_gen_weights must both precede forward");
    const int64_t comp_cs = ctx.chunk_size;
    const int64_t kv_cs   = ctx.effective_kv_cs();
    const int64_t wide_cs = std::max(comp_cs, kv_cs);
    const int64_t h   = hidden_size();
    const int64_t hd  = head_dim();
    const int64_t is_ = intermediate_size();
    const int     tp  = attn_tp();
    const int64_t local_q  = num_q_heads() / tp;
    const int64_t local_kv = num_kv_heads() * hd / tp;
    const int     nl  = static_cast<int>(num_layers());

    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

    const int64_t res    = A(wide_cs * h * DWIDTH);           // LayerWide
    const int64_t kv     = A(kv_cs   * local_kv * DWIDTH);    // KVIN
    const int64_t q_kv   = A(kv_cs   * local_q * hd * DWIDTH);// KVIN
    const int64_t q_comp = A(comp_cs * local_q * hd * DWIDTH);// COMP
    const int64_t out    = A(comp_cs * local_q * hd * DWIDTH);// COMP
    const int64_t oproj  = A(comp_cs * h * DWIDTH);           // COMP
    const int64_t mlp    = A(comp_cs * (is_ / NUM_CORES) * DWIDTH);
    const int64_t down_sz= A(comp_cs * h * DWIDTH);
    // sdpa_tmp 必须按**实际派发的** mask 类型算。本步实际跑 MASK_NONE
    // (mask_type=0; 双向全 attend)。LTM 多一条 tk%tm==0 约束 → choose_tile_k
    // 候选更少 → tile_k ≤ MASK_NONE 选的 → mask=1 的 tmp 可能**小于** mask=0 实跑所需,
    // SDPA 写越界污染相邻 SPM。与 validator 对齐用 mask=0。
    const SdpaConfig sdpa_cfg = make_sdpa_config(/*mask=*/0);
    const int64_t tmp    = A(sdpa_compute_tmp_v16_size(sdpa_cfg, comp_cs) * 32);
    const int64_t nw     = A(h * DWIDTH);
    const int64_t hnw    = A(hd * DWIDTH);

    constexpr BufferScope ALL  = BufferScope::LayerWide;
    constexpr BufferScope KVIN = BufferScope::KvInsert;
    constexpr BufferScope COMP = BufferScope::Compute;

    std::vector<BufferDecl> d;

    // ── 跨两阶段存活 (LayerWide; 与 KVIN/COMP 总冲突) ──
    d.push_back({"residual1",  res, 1, 8, StorageClass::Temp, 0, nullptr, ALL});
    d.push_back({"input_norm", res, 1, 8, StorageClass::Temp, 0, nullptr, ALL});
    d.push_back({"residual2",  0,   0, 0, StorageClass::Temp, 0, "input_norm", ALL});
    // gen 双算 norm/head-norm 暂存 (Phase 1 input-norm/head-norm + Phase 2 post/
    // final-norm 复用; LayerWide 全程存活)。action norm_a 同构。
    d.push_back({"norm_g",     res, 1, 8, StorageClass::Temp, 0, nullptr, ALL});

    // ── 行掩码槽 (und/gen 各一 [cs,1]; body 内 DMA 按 chunk 行切片填, 无 preload) ──
    // merge 将 [cs,1] 行掩码广播到 q/kv/h；两阶段均使用，因此按
    // LayerWide @ wide_cs 分配。
    d.push_back({"und_mask_row", A(wide_cs * DWIDTH), 1, 8, StorageClass::Temp, 0, nullptr, ALL});
    d.push_back({"gen_mask_row", A(wide_cs * DWIDTH), 1, 8, StorageClass::Temp, 0, nullptr, ALL});

    // ── Q split: KVIN 写 q_kv→dump; COMP 从 q_ddr_buf_ 重载到 q_comp (别名) ──
    d.push_back({"q_kv",   q_kv,   1, 8, StorageClass::Temp, 0, nullptr, KVIN});
    d.push_back({"q_comp", q_comp, 2, 4, StorageClass::Temp, 0, nullptr, COMP});
    d.push_back({"k",      kv,     1, 8, StorageClass::Temp, 0, nullptr, KVIN});
    d.push_back({"v",      kv,     1, 8, StorageClass::Temp, 0, nullptr, KVIN});
    // gen 孪生 Phase 1 QKV 槽 (KVIN, 镜像 und q_kv/k/v 相位)
    d.push_back({"q_kv_g", q_kv,   1, 8, StorageClass::Temp, 0, nullptr, KVIN});
    d.push_back({"k_g",    kv,     1, 8, StorageClass::Temp, 0, nullptr, KVIN});
    d.push_back({"v_g",    kv,     1, 8, StorageClass::Temp, 0, nullptr, KVIN});

    // ── COMP-only (real-phase 别名) ──
    d.push_back({"output",   out,     4,  5, StorageClass::Temp, 0, nullptr, COMP});
    d.push_back({"oproj",    oproj,   5,  6, StorageClass::Temp, 0, nullptr, COMP});
    d.push_back({"sdpa_tmp", tmp,     4,  4, StorageClass::Temp, 0, nullptr, COMP});
    d.push_back({"gate",     mlp,     8, 12, StorageClass::Temp, 0, nullptr, COMP});
    d.push_back({"up",       mlp,    10, 11, StorageClass::Temp, 0, nullptr, COMP});
    d.push_back({"down",     down_sz,12, 13, StorageClass::Temp, 0, nullptr, COMP});
    // The gen twin reuses the und oproj/gate/up/down slots serially. Its
    // masked twin results are reduced into residual, so it needs no separate
    // Phase-2 COMP slots. See build_layer_subgraph.

    // ── Qwen2.5 norm preload 槽 (标准 RMSNorm: 仅 DMA, 无 +1 eltwise) ──
    // und 经 und_layers_ / final_norm_w_halo_; gen 经 gen_layers_ / final_norm_gen_w_。
    auto add_norm = [&](const char* name, int64_t sz, int64_t elems,
                        const std::vector<GenLayerWeights>* layers,
                        at::Tensor GenLayerWeights::* member,
                        const at::Tensor* model_level) {
        BufferDecl b;
        b.name      = name;
        b.size      = sz;
        b.storage   = (model_level != nullptr) ? StorageClass::Persistent
                                               : StorageClass::PersistentPerLayer;
        b.per_layer = (model_level != nullptr) ? 0 : nl;
        b.scope     = BufferScope::LayerWide;
        b.preload_callback =
            [this, layers, member, elems, model_level](FusedModelBase&, int L,
                                                       uint32_t core0_addr) {
                const at::Tensor& w = (model_level != nullptr)
                                          ? *model_level
                                          : ((*layers)[L].*member);
                rpu_launch_ddr_broadcast_spm_dma(
                    w.data_ptr<c10::Half>(), elems, core0_addr);
            };
        d.push_back(b);
    };
    // und (base) norm 槽
    add_norm("input_norm_w", nw,  h,  &und_layers_, &GenLayerWeights::input_norm_w, nullptr);
    add_norm("post_norm_w",  nw,  h,  &und_layers_, &GenLayerWeights::post_norm_w,  nullptr);
    add_norm("q_norm_w",     hnw, hd, &und_layers_, &GenLayerWeights::q_norm_w,     nullptr);
    add_norm("k_norm_w",     hnw, hd, &und_layers_, &GenLayerWeights::k_norm_w,     nullptr);
    add_norm("final_norm_w", nw,  h,  nullptr,      &GenLayerWeights::input_norm_w, &final_norm_w_halo_);
    // gen norm 槽
    add_norm("input_norm_w_gen", nw,  h,  &gen_layers_, &GenLayerWeights::input_norm_w, nullptr);
    add_norm("post_norm_w_gen",  nw,  h,  &gen_layers_, &GenLayerWeights::post_norm_w,  nullptr);
    add_norm("q_norm_w_gen",     hnw, hd, &gen_layers_, &GenLayerWeights::q_norm_w,     nullptr);
    add_norm("k_norm_w_gen",     hnw, hd, &gen_layers_, &GenLayerWeights::k_norm_w,     nullptr);
    add_norm("final_norm_gen_w", nw,  h,  nullptr,      &GenLayerWeights::input_norm_w, &final_norm_gen_w_);

    // ── QKV bias preload 槽 (col-partition scatter), und/gen 各一套 ──
    if (has_qkv_bias_halo_) {
        auto dma_safe = [&A](int64_t elems) -> int64_t {
            return A(((elems + 255) / 256) * 256 * DWIDTH);
        };
        auto add_bias = [&](const char* name, int64_t sz,
                            const std::vector<GenLayerWeights>* layers,
                            at::Tensor GenLayerWeights::* member) {
            BufferDecl b;
            b.name      = name;
            b.size      = sz;
            b.storage   = StorageClass::PersistentPerLayer;
            b.per_layer = nl;
            b.scope     = BufferScope::LayerWide;
            b.preload_callback =
                [this, layers, member](FusedModelBase&, int L, uint32_t core0_addr) {
                    const int tp_ = attn_tp();
                    const at::Tensor& bw = (*layers)[L].*member;
                    const int64_t per_core = bw.numel() / tp_;
                    rpu_launch_ddr_scatter_spm_dma(
                        bw.data_ptr<c10::Half>(),
                        per_core, per_core * DWIDTH, core0_addr, tp_);
                };
            d.push_back(b);
        };
        add_bias("q_bias", dma_safe(local_q * hd), &und_layers_, &GenLayerWeights::q_bias);
        add_bias("k_bias", dma_safe(local_kv),     &und_layers_, &GenLayerWeights::k_bias);
        add_bias("v_bias", dma_safe(local_kv),     &und_layers_, &GenLayerWeights::v_bias);
        add_bias("q_bias_gen", dma_safe(local_q * hd), &gen_layers_, &GenLayerWeights::q_bias);
        add_bias("k_bias_gen", dma_safe(local_kv),     &gen_layers_, &GenLayerWeights::k_bias);
        add_bias("v_bias_gen", dma_safe(local_kv),     &gen_layers_, &GenLayerWeights::v_bias);
    }

    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_kv_first_body — KV_FIRST Phase 1: input-norm 双算+merge / QKV(+bias) 双算
// +merge / qk head-norm 双算+merge / RoPE(merged) / KV insert(merged) / 存 merged
// rope_q 到 q_ddr_buf_。und/gen 双流, 行掩码合并。模板 = action expert 步 1-3 +
// 现有单流 KV-insert/dump。
// ─────────────────────────────────────────────────────────────────────────────
void HaloImageFlowModel::emit_kv_first_body(int layer_idx, const ChunkInfo& chunk) {
    const auto& uw = und_layers_[layer_idx];
    const auto& gw = gen_layers_[layer_idx];
    const int64_t seq_len = chunk.len;
    const int64_t cos_sin_start = ctx().position + chunk.offset;
    const int64_t h   = hidden_size();
    const int64_t nq  = num_q_heads();
    const int64_t nkv = num_kv_heads();
    const int64_t hd  = head_dim();
    const int tp = attn_tp();
    const int64_t local_q = nq / tp;
    const int64_t local_kv_heads = nkv / tp;
    const int64_t local_kv = nkv * hd / tp;
    // merge 每行宽 (Nx1_NxC 的 c; 行掩码 n=seq_len 广播): h / q(local_q*hd) / kv(local_kv)。
    // head-norm 的 [seq*local_q, hd] 连续 = [seq, local_q*hd], 故 c=local_q*hd 正确。
    const int64_t w_h  = h;
    const int64_t w_q  = local_q * hd;
    const int64_t w_kv = local_kv;
    const uint32_t und_m = addr(0, "und_mask_row"), gen_m = addr(0, "gen_mask_row");

    // SPM_RESIDENT contract, shared with the sibling action-expert model.
    // With a SINGLE compute chunk run_all_layers resolves InterLayerIO::AUTO to
    // SPM_RESIDENT and deliberately leaves
    // ddr_bufA/B_ptr_ NULL (need_chain_bufs false): layer L>0 reads its input
    // straight out of "residual1". Emitting the DMA anyway dereferences those
    // nulls at layer 1. This model already had the mirror guard on the OUTPUT
    // side (`!ctx().output_to_spm` below).
    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }
    upload_merge_masks(chunk, /*h_only=*/false);

    // ── 步 1: input RMSNorm 双算 → merge 入 "input_norm" ──
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "input_norm_w"), seq_len, h, eps_halo_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "norm_g"),
        layer_addr(layer_idx, 0, "input_norm_w_gen"), seq_len, h, eps_halo_);
    emit_row_merge(addr(0, "input_norm"), addr(0, "norm_g"), addr(0, "input_norm"),
                   und_m, gen_m, seq_len, w_h, NUM_CORES);

    // ── 步 2: QKV 双发射 (und→q_kv/k/v, gen→q_kv_g/k_g/v_g) + merge ──
    auto qkv = [&](const at::Tensor& w, const char* out, const char* bias_slot,
                   int64_t n) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), w, addr(0, out),
            seq_len, n, h, /*partition=*/1, /*num_cores=*/tp,
            has_qkv_bias_halo_ ? layer_addr(layer_idx, 0, bias_slot) : 0u,
            /*force_gemm=*/false, /*scale=*/{});
    };
    qkv(uw.q_w, "q_kv",   "q_bias",     nq * hd);
    qkv(uw.k_w, "k",      "k_bias",     nkv * hd);
    qkv(uw.v_w, "v",      "v_bias",     nkv * hd);
    qkv(gw.q_w, "q_kv_g", "q_bias_gen", nq * hd);
    qkv(gw.k_w, "k_g",    "k_bias_gen", nkv * hd);
    qkv(gw.v_w, "v_g",    "v_bias_gen", nkv * hd);
    emit_row_merge(addr(0, "q_kv"), addr(0, "q_kv_g"), addr(0, "q_kv"), und_m, gen_m, seq_len, w_q,  tp);
    emit_row_merge(addr(0, "k"),    addr(0, "k_g"),    addr(0, "k"),    und_m, gen_m, seq_len, w_kv, tp);
    emit_row_merge(addr(0, "v"),    addr(0, "v_g"),    addr(0, "v"),    und_m, gen_m, seq_len, w_kv, tp);

    // ── 步 3: q/k head-norm 双算 + merge, RoPE 在 merge 后单次 in-place ──
    // head 维行排布 [seq*local_heads, hd] 与掩码 [seq, local_heads*hd] 同一段内存
    // → q/kv 宽度掩码可直接复用。und 暂存 input_norm (步 2 后空闲), gen 暂存 norm_g。
    c10::Half* cos_ptr = cos_halo_.data_ptr<c10::Half>();
    c10::Half* sin_ptr = sin_halo_.data_ptr<c10::Half>();
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "q_kv"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "q_norm_w"), seq_len * local_q, hd, eps_halo_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "q_kv"), addr(0, "norm_g"),
        layer_addr(layer_idx, 0, "q_norm_w_gen"), seq_len * local_q, hd, eps_halo_);
    emit_row_merge(addr(0, "input_norm"), addr(0, "norm_g"), addr(0, "input_norm"),
                   und_m, gen_m, seq_len, w_q, tp);
    rpu_launch_rope_spm_kernel(addr(0, "input_norm"), addr(0, "q_kv"),
                               cos_ptr, sin_ptr, seq_len, local_q, hd, cos_sin_start);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "k"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "k_norm_w"), seq_len * local_kv_heads, hd, eps_halo_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "k"), addr(0, "norm_g"),
        layer_addr(layer_idx, 0, "k_norm_w_gen"), seq_len * local_kv_heads, hd, eps_halo_);
    emit_row_merge(addr(0, "input_norm"), addr(0, "norm_g"), addr(0, "input_norm"),
                   und_m, gen_m, seq_len, w_kv, tp);
    rpu_launch_rope_spm_kernel(addr(0, "input_norm"), addr(0, "k"),
                               cos_ptr, sin_ptr, seq_len, local_kv_heads, hd, cos_sin_start);

    // ── KV cache insert (merged K/V, 绝对位置) + 存 merged rope_q ──
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    rpu_launch_insert_kcache_spm_unified(
        k_cache, cos_sin_start, addr_offset("k").value, seq_len, nkv, hd, tp);
    rpu_launch_insert_vcache_spm_unified(
        v_cache, cos_sin_start, addr_offset("v").value, seq_len, nkv, hd, tp);

    TORCH_CHECK(
        q_ddr_slots_.count({seq_len_, (num_q_heads() / attn_tp()) * head_dim()}) == 1,
        "HaloImageFlowModel: Q staging slot not allocated in KV_FIRST mode");
    const at::Tensor& q_slot = q_ddr_slot();
    const int64_t q_local_elems = seq_len * local_q * hd;
    c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
    const int64_t q_row_stride = local_q * hd;
    const int64_t q_elem_offset = chunk.offset * q_row_stride;
    const int64_t q_core_stride_bytes = q_slot.size(1) * q_row_stride * DWIDTH;
    rpu_launch_spm_scatter_ddr_dma(
        addr(0, "q_kv"), q_ddr_base + q_elem_offset,
        q_local_elems, q_core_stride_bytes, /*num_cores=*/tp);
    // Phase 2 从原 input 重读残差; 此处不输出残差。
}

// ─────────────────────────────────────────────────────────────────────────────
// build_layer_subgraph — KV_FIRST Phase 2: load-q / 满 KV SDPA(MASK_NONE) /
// O 双算+merge / 残差 / post-norm 双算+merge / SwiGLU MLP 双算+merge / (末层)
// final-norm 双算+merge / output DMA。模板 = action expert 步 4-7。
// ─────────────────────────────────────────────────────────────────────────────
void HaloImageFlowModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    const auto& uw = und_layers_[layer_idx];
    const auto& gw = gen_layers_[layer_idx];
    const int64_t seq_len = chunk.len;
    const int64_t h   = hidden_size();
    const int64_t nq  = num_q_heads();
    const int64_t nkv = num_kv_heads();
    const int64_t hd  = head_dim();
    const int tp = attn_tp();
    const int64_t local_q = nq / tp;
    const bool is_last_layer = (layer_idx == num_layers() - 1);
    const int64_t elems_mlp = seq_len * (intermediate_size() / NUM_CORES);
    const uint32_t und_m = addr(0, "und_mask_row"), gen_m = addr(0, "gen_mask_row");

    // 重读原 input → residual1 (残差源, 与 Phase 1 同源)。
    // 同 Phase 1 的 SPM_RESIDENT 守卫: 该模式下 DDR ping-pong 缓冲不存在, 且
    // Phase 1 只从 residual1 读、未覆写它, 故 Phase 2 直接复用 SPM 里的同一份 input。
    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }
    upload_merge_masks(chunk, /*h_only=*/true);

    // 从 per-shape Q staging slot 重载 merged rope_q → q_comp
    const int64_t q_local_elems = seq_len * local_q * hd;
    const at::Tensor& q_slot = q_ddr_slot();
    c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
    const int64_t q_row_stride = local_q * hd;
    const int64_t q_elem_offset = chunk.offset * q_row_stride;
    const int64_t q_core_stride = q_slot.size(1) * q_row_stride * DWIDTH;
    rpu_launch_ddr_scatter_spm_dma(
        q_ddr_base + q_elem_offset, q_local_elems, q_core_stride,
        addr(0, "q_comp"), /*num_cores=*/tp);

    // 满 KV SDPA (MASK_NONE 全 attend; image gen 非因果双向 + 全 prefix 可见)。
    // total_kv = prefix + 全部 query (Phase 1 已插入全部 query K/V)。MASK_NONE ⇒
    // 不需 mask DMA, sdpa_tmp offset 作占位 mask 入参 (不解引用)。
    const int64_t total_kv_seq_len = ctx().position + seq_len_;
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    rpu_launch_sdpa_spm_dispatch(
        SdpaKernelType::FLASH_ATTN_SPM,
        k_cache, v_cache, /*mask_type=*/0 /*MASK_NONE*/, c10::nullopt,
        addr_offset("q_comp").value, addr_offset("output").value,
        addr_offset("sdpa_tmp").value, addr_offset("sdpa_tmp").value /*mask 占位*/,
        seq_len, nq, nkv, hd, total_kv_seq_len, tp, NUM_CORES);

    // ── O_proj 串行双算 + 残差折叠 (R1: 无 oproj_g; gen 复用 oproj 槽) ──
    // residual2 = residual1 + AR(und·m_u) + AR(gen·m_g)。**out-of-place**: gen 先折进
    // norm_g(中转), und 再从 norm_g 折进 residual2 → 两次 AR 三地址互异(避免
    // all_reduce in-place; norm_g 此刻空闲, LayerWide 全 phase 存活)。attn_out 落
    // residual2(与下游约定一致)。"output" 两次只读不写; 行掩码 m 在 AR 前广播乘
    // (AR 线性 + mask 逐行均匀 ⇒ 等价 merge 后 AR; 疑点3 已证位级一致)。
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), gw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/{});
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        gen_m, addr(0, "oproj"), addr(0, "oproj"), seq_len, h,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "residual1"), addr(0, "norm_g"),
        seq_len, h, tp, NUM_CORES);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), uw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0u, /*force_gemm=*/false, /*scale=*/{});
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        und_m, addr(0, "oproj"), addr(0, "oproj"), seq_len, h,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "norm_g"), addr(0, "residual2"),
        seq_len, h, tp, NUM_CORES);

    // ── post-attention RMSNorm 双算 + merge → residual1 ──
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "residual1"),
        layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, eps_halo_);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "norm_g"),
        layer_addr(layer_idx, 0, "post_norm_w_gen"), seq_len, h, eps_halo_);
    emit_row_merge(addr(0, "residual1"), addr(0, "norm_g"), addr(0, "residual1"),
                   und_m, gen_m, seq_len, h, NUM_CORES);

    // ── 手写双 MLP (SwiGLU) → down 残差折叠 ──
    // mlp_half 与 action expert 使用相同结构。不用基类 emit_mlp_pipeline:
    // 它把 all_reduce_sum_residual 写死在 down 后, 而双分支须在归约前合并。
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
    // **out-of-place 折叠** (base=residual2=attn_out, input=residual1=post-norm, 中转=norm_g):
    // und MLP → down; *=m_u; AR(down, residual2 → norm_g)(不写 residual1 —— gen mlp 还要
    // 读它作输入; norm_g 此刻空闲)。
    mlp_half(uw.gate_w, uw.up_w, uw.down_w, "gate", "up", "down");
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        und_m, addr(0, "down"), addr(0, "down"), seq_len, h,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "down"), addr(0, "residual2"), addr(0, "norm_g"),
        seq_len, h, NUM_CORES, NUM_CORES);
    // gen MLP 复用 gate/up/down 槽 → down; *=m_g; AR(down, norm_g → residual1) = layer_out。
    // gen mlp_half 的 gate/up 先读 residual1, 之后 AR 才写 residual1 ⇒ 无 RAW 冲突; 三地址
    // (down/norm_g/residual1) 互异 ⇒ 非 in-place。
    mlp_half(gw.gate_w, gw.up_w, gw.down_w, "gate", "up", "down");
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        gen_m, addr(0, "down"), addr(0, "down"), seq_len, h,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "down"), addr(0, "norm_g"), addr(0, "residual1"),
        seq_len, h, NUM_CORES, NUM_CORES);

    // ── 末层: final RMSNorm 双算 + merge → residual1 ──
    // und γ 经 "final_norm_w", gen γ 经 "final_norm_gen_w"。暂存复用 down + norm_g。
    if (is_last_layer) {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "down"),
            addr(0, "final_norm_w"), seq_len, h, eps_halo_);
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "norm_g"),
            addr(0, "final_norm_gen_w"), seq_len, h, eps_halo_);
        emit_row_merge(addr(0, "down"), addr(0, "norm_g"), addr(0, "residual1"),
                       und_m, gen_m, seq_len, h, NUM_CORES);
    }

    if (!ctx().output_to_spm) {
        emit_layer_output_dma(layer_idx, chunk);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// step_forward — 一次 image flow denoise step
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor HaloImageFlowModel::step_forward(
    const at::Tensor& x_emb, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& cos, const at::Tensor& sin, int64_t prefix_len)
{
    TORCH_CHECK(!und_layers_.empty() && !gen_layers_.empty() && row_mask_.defined(),
                "halo image_flow step_forward before set_weights_image_flow + "
                "set_gen_weights");
    const int64_t h = hidden_size();
    TORCH_CHECK(x_emb.dim() == 3 && x_emb.size(0) == 1 && x_emb.size(2) == h
                && x_emb.scalar_type() == at::kHalf && x_emb.is_contiguous()
                && x_emb.device().type() == at::kPrivateUse1,
                "halo image_flow step_forward: x_emb must be [1,seq,", h,
                "] fp16 contig RPU");
    TORCH_CHECK(prefix_len >= 0, "halo image_flow step_forward: prefix_len >= 0");
    const int64_t seq_len = x_emb.size(1);
    TORCH_CHECK(seq_len == merge_seq_len_,
                "halo image_flow step_forward: x_emb seq (", seq_len,
                ") != row_mask query_len (", merge_seq_len_, ")");

    // per-forward cos/sin (镜像 qwenpi05 / action expert): 覆盖 RoPE 影子成员,
    // 赋值即保活过 graph capture。图像 token 共享单一 rope_position → 位置广播表;
    // CFG 三分支各传自己 rope_position 的表 (每分支 sig 独立 → 独立 BUILD →
    // 各捕各自 cos/sin 指针)。
    TORCH_CHECK(cos.defined() && sin.defined()
                && cos.device().type() == at::kPrivateUse1
                && sin.device().type() == at::kPrivateUse1
                && cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf
                && cos.is_contiguous() && sin.is_contiguous()
                && cos.dim() == 2 && sin.sizes() == cos.sizes()
                && cos.size(1) == head_dim() / 2,
                "halo image_flow step_forward: cos/sin must be [*, head_dim/2] fp16 "
                "contig RPU, equal shape");
    TORCH_CHECK(cos.size(0) >= prefix_len + seq_len,
                "halo image_flow step_forward: cos rows (", cos.size(0),
                ") < prefix_len+seq (", prefix_len + seq_len, ")");
    cos_halo_ = cos;
    sin_halo_ = sin;

    // KV headroom (7-D K 布局 max_seq = size(1)*size(5))
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + seq_len <= kc.size(1) * kc.size(5),
                    "halo image_flow step_forward: prefix_len(", prefix_len, ")+seq(",
                    seq_len, ") exceeds k_cache capacity ", kc.size(1) * kc.size(5));
    }

    // KV_FIRST 跨两阶段 rope_q DDR 暂存 (Gemma 同构)。稳定 per-shape slot 按
    // {seq_len, q_width} 双键 —— 创建后永不 realloc,已捕获图的 fixed-DMA 地址跨
    // REPLAY 始终有效；宽度入键支持 set_weights 改变 head 维度。
    const int64_t q_ddr_local_dim = (num_q_heads() / attn_tp()) * head_dim();
    const std::pair<int64_t, int64_t> q_key{seq_len, q_ddr_local_dim};
    if (q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
        TORCH_CHECK(q_ddr_slots_.size() < 128,
                    "HaloImageFlowModel: q_ddr_slots_ exceeded 128 distinct "
                    "(seq_len,width) shapes on one handle — likely a runaway shape "
                    "loop (per-shape Q staging is never freed for replay-address "
                    "stability)");
        q_ddr_slots_.emplace(q_key, at::empty(
            {static_cast<int64_t>(NUM_CORES), seq_len, q_ddr_local_dim},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1)));
    }
    seq_len_ = seq_len;

    // chunk_size 强制钩子；0 表示由 compute_chunks 按预算与有效性
    // 自动选择。该值按 handle 冻结，避免跨模型实例相互影响。

    // host 合成的 x_emb 可能带脏 cache 行; BUILD 前显式 flush。
    rpu_ddr_flush_force(x_emb.data_ptr<c10::Half>());

    // 直接走 run_all_layers (绕 CausalDecoderModel::forward 的 is_causal||mask
    // 守卫): nullopt mask + is_causal=false → use_explicit_mask=false →
    // max_kv_seq_len = position+seq (MASK_NONE 全 attend)。返回基类 output_tensor_
    // (graph 写目标), 调用方须在 capture 作用域**退出后**读。
    return run_all_layers(
        x_emb, k_caches, v_caches,
        /*attention_mask=*/std::nullopt,
        /*position=*/prefix_len, /*is_causal=*/false);
}

}  // namespace v3

// =============================================================================
// Instance registry + Public C API
// =============================================================================
using HaloImageFlowRegistry = ModelHandleRegistry<v3::HaloImageFlowModel>;

int64_t rpu_halo_image_flow_create() {
    return HaloImageFlowRegistry::create();
}

void rpu_halo_image_flow_destroy(int64_t handle) {
    HaloImageFlowRegistry::destroy(handle, "rpu_halo_image_flow_destroy");
}

// Per-handle chunk-size override; 0 selects automatic sizing.
void rpu_halo_image_flow_set_chunk_size_override(int64_t handle, int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0 || chunk_size >= 16,
                "halo_image_flow_set_chunk_size_override: chunk_size must be 0 "
                "(auto) or >= 16, got ", chunk_size);
    HaloImageFlowRegistry::get(handle, "rpu_halo_image_flow_set_chunk_size_override")
        ->set_chunk_size_override(chunk_size);
}

// und/base 权重: 单 schema 包装 set_weights_image_flow (与 action expert set_weights
// 同签名; 喂入无后缀 base/und 孪生, start/end_of_image 那 2 个 text 行用)。
void rpu_halo_image_flow_set_weights(
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
    HaloImageFlowRegistry::get(handle, "rpu_halo_image_flow_set_weights")
        ->set_weights_image_flow(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, use_silu, q_bias, k_bias, v_bias);
}

// gen(*_moe_gen) 孪生 + 行掩码: 包装 set_gen_weights (= action set_moe_weights 角色)。
void rpu_halo_image_flow_set_gen_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    const at::Tensor& final_norm_gen_w, const at::Tensor& row_mask,
    int64_t query_len)
{
    HaloImageFlowRegistry::get(handle, "rpu_halo_image_flow_set_gen_weights")
        ->set_gen_weights(q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm,
                          post_norm, gate_w, up_w, down_w, q_bias, k_bias,
                          v_bias, final_norm_gen_w, row_mask, query_len);
}

at::Tensor rpu_halo_image_flow_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin, int64_t prefix_len)
{
    return HaloImageFlowRegistry::get(handle, "rpu_halo_image_flow_step_forward")
        ->step_forward(x_emb, k_caches, v_caches, cos, sin, prefix_len);
}
