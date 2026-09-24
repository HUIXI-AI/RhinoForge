// rpu_hyvla_expert_model.cpp — Hy-Embodied-0.5-VLA action-expert decoder.
// The expert uses the _v branch because its suffix modality mask is true.
// Unlike the VLM it needs no twin GEMMs or row-mask merge. It applies RoPE
// before shared QK normalization. The base declares layer buffers; this class
// adds unroll hooks and plans mask lifetimes across bodies.
#include "rpu_hyvla_expert_model.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"
#include "rpu_spm_residency.h"

#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// RPU_HY_VLA_FAST_REPLAY skips the per-component host op walk only when all
// request changes are represented by stable buffers or mutable DMA bases.
// RPU_HY_VLA_FAST_REPLAY_PRELOAD also requires the layer-loop skip and clean
// weights; captured preload nodes still execute on replay.
//
// RPU_HY_VLA_ACTION_MLP_MC partitions A6 along K and reduces the partial outputs.
// Do not add bias inside row-partitioned GEMM: add it once after all-reduce.
// The consumed enc_h_spm input can then be cleared and reused as the zero
// residual; all-reduce requires distinct input, residual and output addresses.
// Python supplies the same core count to swizzling and native set_weights.
// Partitioning changes accumulation order and may change rounding.
//
// RPU_HY_VLA_MASK_ONCE extends a constant request mask's lifetime. The allocator
// tries whole-forward residency after all hooks are declared, including phase 0.
// Use it only when the temporary peak does not grow; otherwise preserve one
// load per body. Every invocation still refreshes the current request's mask.
//
// RPU_HY_VLA_SILU_MUL replaces separate SiLU and multiply launches with one fused
// operation. The up GEMM must precede it; up and gate have independent inputs.
// The fused operation may have a different intermediate rounding boundary.
//
// RPU_HY_VLA_KVPAD16 rounds K/V SPM capacity to a multiple of 16 for one v16
// insertion. The additional cache rows are excluded by the logical kv_seq_len.
// Grow K/V capacity to at least rows; return the guaranteed row capacity.
static int64_t hyvla_grow_kv_slots(std::vector<v3::BufferDecl>& decls,
                                   int64_t rows, int64_t local_kv_elems) {
    const int64_t need = Align(rows * local_kv_elems * 2, 256);
    for (auto& b : decls)
        if (b.name && (std::strcmp(b.name, "k") == 0 || std::strcmp(b.name, "v") == 0))
            b.size = std::max(b.size, need);
    return rows;
}

// RPU_HY_VLA_RMSNORM_PAD16 grows residual1/input_norm/q/output to aligned row
// capacities so a vector route can use one launch. Aliases such as residual2
// follow their backing allocation. Downstream consumers use only logical rows;
// RMSNorm is row-local, so padded values cannot affect live rows.
//
// RPU_HY_VLA_PARTIAL_ROPE selects partial_mrope with rotary_dim == head_dim.
// Its grid is (ceil(rotary_dim/2/64), ceil(seq/64)); cos/sin remain the existing
// [max_seq, head_dim/2] tables and pos_offset is a table-row offset.
// Grow the requested buffers without shrinking them; report whether all exist.
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

// HYVLA_EXPERT_FIXED_KERNEL_BASIS: launchers outside the selector-controlled
// sites below are invariant model-dataflow glue.  Every cold/per-handle route
// that changes the graph, kernel, DMA, collective, RoPE, or layout is bound to
// an explicit physical-manifest entry before its launcher is emitted.

constexpr int64_t HYVLA_EXPERT_ATTN_SITE = 8372732983358876378LL;
constexpr int64_t HYVLA_EXPERT_KV_SITE = 2502630319608950519LL;
constexpr int64_t HYVLA_EXPERT_KV_PADDING_SITE = 3353793944280539666LL;
constexpr int64_t HYVLA_EXPERT_GRAPH_SCHEDULE_SITE = 5602663956511155425LL;
constexpr int64_t HYVLA_EXPERT_MASK_SCHEDULE_SITE = 1728596774874433151LL;

constexpr int64_t HYVLA_EXPERT_PRE_X0_DMA_SITE = 8468013054253886391LL;
constexpr int64_t HYVLA_EXPERT_PRE_X_FEEDBACK_DMA_SITE = 8068344894213309215LL;
constexpr int64_t HYVLA_EXPERT_PRE_TIME_BROADCAST_DMA_SITE = 6202700778193353884LL;
constexpr int64_t HYVLA_EXPERT_PRE_TIME_SCATTER_DMA_SITE = 5681167721722131756LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_BIAS_DMA_SITE = 3234409676337683315LL;
constexpr int64_t HYVLA_EXPERT_PRE_WC_LINEAR_SITE = 94006821360779226LL;
constexpr int64_t HYVLA_EXPERT_PRE_ENCODER_ACTIVATION_SITE = 4663734415968545282LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_SINGLE_LINEAR_SITE = 8695816214428940237LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_MULTI_LINEAR_SITE = 2981602481703720578LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_ZERO_SITE = 4262731488614715412LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_ALL_REDUCE_SITE = 414612096590535714LL;
constexpr int64_t HYVLA_EXPERT_PRE_MO_BIAS_ADD_SITE = 2288102790066046320LL;
constexpr int64_t HYVLA_EXPERT_POST_OP_LINEAR_SITE = 2567201997024428744LL;
constexpr int64_t HYVLA_EXPERT_POST_TRAJECTORY_DMA_SITE = 601652272865429711LL;

constexpr int64_t HYVLA_EXPERT_INPUT_NORM_SITE = 2943148670015766034LL;
constexpr int64_t HYVLA_EXPERT_QKV_LINEAR_SITE = 1236943476379916654LL;
constexpr int64_t HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE = 4314852815466214180LL;
constexpr int64_t HYVLA_EXPERT_Q_ROPE_SITE = 2913243967564739575LL;
constexpr int64_t HYVLA_EXPERT_Q_NORM_SITE = 2998501871001866367LL;
constexpr int64_t HYVLA_EXPERT_K_PARTIAL_ROPE_SITE = 6204149761964354754LL;
constexpr int64_t HYVLA_EXPERT_K_ROPE_SITE = 2703116739699322018LL;
constexpr int64_t HYVLA_EXPERT_K_NORM_SITE = 739473353641483894LL;
constexpr int64_t HYVLA_EXPERT_PREPARE_ATTN_ALL_REDUCE_SITE = 7991151858475231951LL;
constexpr int64_t HYVLA_EXPERT_O_LINEAR_SITE = 36160785207660705LL;
constexpr int64_t HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE = 5454728429937755845LL;
constexpr int64_t HYVLA_EXPERT_POST_NORM_SITE = 48200961442401552LL;
constexpr int64_t HYVLA_EXPERT_GATE_LINEAR_SITE = 2105943201978379215LL;
constexpr int64_t HYVLA_EXPERT_MLP_SILU_SITE = 1848219361374021523LL;
constexpr int64_t HYVLA_EXPERT_UP_LINEAR_SITE = 4699339857693172978LL;
constexpr int64_t HYVLA_EXPERT_MLP_SILU_MUL_SITE = 82400951984648727LL;
constexpr int64_t HYVLA_EXPERT_MLP_MUL_SITE = 6313036385062334532LL;
constexpr int64_t HYVLA_EXPERT_DOWN_LINEAR_SITE = 2699481217144264799LL;
constexpr int64_t HYVLA_EXPERT_MLP_ALL_REDUCE_SITE = 2337931566137374851LL;
constexpr int64_t HYVLA_EXPERT_FINAL_NORM_SITE = 6930260388322325380LL;

constexpr uint32_t HYVLA_EXPERT_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 | KV_INSERT_CAP_PAD16 |
    KV_INSERT_CAP_HYBRID2 | KV_INSERT_CAP_NON8_TP_V16;
constexpr int64_t HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED = 2;

enum class HyVlaExpertGraphScheduleRoute : int64_t {
    REEMIT_LAYER_LOOP = 1,
    FAST_REPLAY_SKIP_LAYER_LOOP = 2,
    MASK_EACH_LAYER = 3,
    MASK_FIRST_LAYER_ONLY = 4,
    ZERO_A6_RESIDUAL = 5,
    ADD_A6_BIAS_AFTER_REDUCE = 6,
    MASK_FIRST_BODY_FIRST_LAYER = 7,
};

enum class HyVlaExpertNormalizationRoute : int64_t {
    RMSNORM_SCALAR_ROWS = 1,
    RMSNORM_V16_ROWS = 2,
};

enum class HyVlaExpertActivationRoute : int64_t {
    SILU_UNARY = 1,
    MUL = 2,
    SILU_MUL_FUSED = 3,
};

enum class HyVlaExpertRopeRoute : int64_t {
    ROPE_1D = 1,
    PARTIAL_MROPE = 2,
};

enum class HyVlaExpertKvPaddingRoute : int64_t {
    LOGICAL_ROWS = 1,
    PAD16_ROWS = 2,
};

enum class HyVlaExpertDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM_MUTABLE = 1,
    DDR_BROADCAST_TO_SPM_FIXED = 2,
    DDR_SCATTER_TO_SPM_FIXED = 3,
    SPM_TO_DDR_MUTABLE = 4,
};

enum class HyVlaExpertAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

HyVlaExpertModel::HyVlaExpertModel() = default;
HyVlaExpertModel::~HyVlaExpertModel() = default;

void HyVlaExpertModel::set_runtime_config(
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool silu_mul, bool kvpad16, bool partial_rope,
    bool rmsnorm_pad16) {
    TORCH_CHECK(
        !cold_config_bound_ && !production_config_bound_,
        "hyvla_expert_set_runtime_config must run exactly once before "
        "set_weights");
    cold_fast_replay_ = fast_replay;
    cold_fast_replay_preload_ = fast_replay_preload;
    cold_mask_once_ = mask_once;
    cold_silu_mul_ = silu_mul;
    cold_kvpad16_ = kvpad16;
    cold_partial_rope_ = partial_rope;
    cold_rmsnorm_pad16_ = rmsnorm_pad16;
    cold_config_bound_ = true;
    invalidate_model_state();
}

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
    int64_t chunk_size, int64_t action_mlp_cores,
    at::TensorList q_ws_list, at::TensorList k_ws_list,
    at::TensorList v_ws_list, at::TensorList o_ws_list,
    at::TensorList gate_ws_list, at::TensorList up_ws_list,
    at::TensorList down_ws_list)
{
    TORCH_CHECK(
        cold_config_bound_,
        "hyvla_expert set_weights requires an explicit immutable runtime "
        "config snapshot");
    TORCH_CHECK(chunk_size == 51,
                "hyvla_expert set_weights: the certified Hy-VLA suffix has "
                "exactly 51 rows, got ", chunk_size);
    TORCH_CHECK(action_mlp_cores == 1 || action_mlp_cores == NUM_CORES,
                "hyvla_expert set_weights: action_mlp_cores must be 1 or ",
                NUM_CORES, ", got ", action_mlp_cores);
    if (production_config_bound_) {
        TORCH_CHECK(action_mlp_cores_ == action_mlp_cores,
                    "hyvla_expert set_weights: action MLP core count is "
                    "immutable for a native handle");
    }
    // Fail the cold-only check before the base setter retains any weights.
    set_chunk_envelope(/*max_kv_len=*/291, /*chunk=*/64);
    // 校验/状态提交全部委托基类 (含 q/k norm 全套检查 + quant scale 的
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
                "query/key_layernorm —— 两塔共享 VLM 那一份, expert 自己的同名 "
                "张量是死权重 (P0 hook calls=0)。");

    // 影子副本 — build_layer_subgraph 要逐层权重, 而基类 layer_weights_ 是 private。
    // W4 scale 必须单独 retain 基类相同的 4096B-aligned view。
    const int64_t N = static_cast<int64_t>(q_w_list.size());
    const bool quantized = !q_ws_list.empty();
    layers_.clear();
    layers_.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
        layers_.push_back({q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                           gate_list[i], up_list[i], down_list[i],
                           quantized ? rpu_retain_linear_quant_scale(q_w_list[i], q_ws_list[i]) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(k_w_list[i], k_ws_list[i]) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(v_w_list[i], v_ws_list[i]) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(o_w_list[i], o_ws_list[i]) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(gate_list[i], gate_ws_list[i]) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(up_list[i], up_ws_list[i]) : at::Tensor(),
                           quantized ? rpu_retain_linear_quant_scale(down_list[i], down_ws_list[i]) : at::Tensor()});
    }
    cos_ref_ = cos;
    sin_ref_ = sin;
    eps_expert_ = eps;
    expert_chunk_size_ = chunk_size;
    action_mlp_cores_ = static_cast<int>(action_mlp_cores);
    production_config_bound_ = true;

    invalidate_model_state();   // D-503: set_weights 的末条非空语句
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
std::vector<BufferDecl> HyVlaExpertModel::baseline_buffer_declarations(
    const LayoutContext& ctx) const {
    // The base only constructs preload callbacks here; none executes during
    // this pure layout probe. Keep the non-const bridge at this one boundary.
    auto d = const_cast<HyVlaExpertModel*>(this)
        ->causal_baseline_buffer_declarations(ctx);
    if (cold_mask_once_) hyvla_widen_sdpa_mask_slot(d);
    // KVPAD16（见文件头）：把 k/v 槽撑到 16 的倍数行，KV-insert 一次 v16 打完。
    if (cold_kvpad16_) {
        const int64_t local_kv = num_kv_heads() * head_dim() / attn_tp();
        hyvla_grow_kv_slots(
            d, Align(ctx.chunk_size, (int64_t)16), local_kv);
    }
    // RMSNORM_PAD16（见文件头）：把 rmsnorm 读写的四个槽撑到 16 的倍数行。
    if (cold_rmsnorm_pad16_) {
        const int64_t hh = hidden_size(), hd_ = head_dim();
        const int64_t lq = num_q_heads() / attn_tp();
        const int64_t rmsnorm_pad_rows_ = Align(ctx.chunk_size, (int64_t)16);
        const int64_t rmsnorm_pad_qrows_ = Align(ctx.chunk_size * lq, (int64_t)16);
        const bool ok =
            hyvla_grow_slot(d, "residual1",  rmsnorm_pad_rows_  * hh  * 2) &&
            hyvla_grow_slot(d, "input_norm", rmsnorm_pad_rows_  * hh  * 2) &&
            hyvla_grow_slot(d, "q",          rmsnorm_pad_qrows_ * hd_ * 2) &&
            hyvla_grow_slot(d, "output",     rmsnorm_pad_qrows_ * hd_ * 2);
        // 四个槽名一个都不能漏 —— 漏了就是**写越界**，不是慢一点。
        TORCH_CHECK(ok, "hyvla RMSNORM_PAD16: 基类槽名对不上（residual1/input_norm/q/output）");
    }
    if (!unroll_mode_) return d;
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
    if (action_mlp_cores_ > 1)
        d.push_back({"mo_part_spm", A(cs * h * DW), 0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"time_spm",      A(h * DW),       0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"mo_b_spm",      A(h * DW),       0, 0, SC::Temp, 0, nullptr, ALL});
    // post-hook 的产物落在相 8..9 —— **不得**与 residual1 (1..8) 混叠, post-hook 要读它。
    d.push_back({"v_t_core0_spm", A(cs * np * DW), 8, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"op_b_spm",      A(np * DW),      8, 9, SC::Temp, 0, nullptr, ALL});
    return d;
}

std::vector<BufferDecl> HyVlaExpertModel::planned_buffer_declarations(
    const LayoutContext& layout, bool* retained) const {
    auto declarations = baseline_buffer_declarations(layout);
    const bool keep = cold_mask_once_ && layout.use_attn_mask &&
        detail::plan_forward_spm_residency(declarations, {"sdpa_mask"}).retained;
    if (retained) *retained = keep;
    return declarations;
}

std::vector<BufferDecl> HyVlaExpertModel::declare_buffers(
    const LayoutContext& layout) {
    return planned_buffer_declarations(layout);
}

int64_t HyVlaExpertModel::mask_schedule_for_layout(
    const LayoutContext& layout) const {
    bool retained = false;
    planned_buffer_declarations(layout, &retained);
    return static_cast<int64_t>(retained
        ? HyVlaExpertGraphScheduleRoute::MASK_FIRST_BODY_FIRST_LAYER
        : cold_mask_once_ ? HyVlaExpertGraphScheduleRoute::MASK_FIRST_LAYER_ONLY
                          : HyVlaExpertGraphScheduleRoute::MASK_EACH_LAYER);
}

ModelStaticConfig HyVlaExpertModel::static_config() {
    ModelStaticConfig cfg = CausalDecoderModel::static_config();
    cfg.fast_replay_skip_layer_loop = cold_fast_replay_;
    cfg.fast_replay_skip_preload = cold_fast_replay_preload_;
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
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_X0_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
                /*resolved_flags=*/0,
                {unroll_mode_ ? 1 : 0, cs * np, NUM_CORES});
        }
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &x_t_src_base_, /*src_offset_bytes=*/0, /*num_elements=*/cs * np,
            addr(0, "x_t_spm"), /*num_cores=*/NUM_CORES);
    } else if (action_mlp_cores_ > 1) {
        // A4 is column-parallel, so every core needs the complete x_t. Euler updates
        // only core 0; subsequent steps reload x_traj[bit-1] and broadcast it to all
        // cores, using the existing trajectory buffer lifetime.
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_X_FEEDBACK_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
                /*resolved_flags=*/0,
                {action_mlp_cores_, num_steps_, cs * np, NUM_CORES});
        }
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &x_traj_dst_base_, /*src_offset_bytes=*/(bit - 1) * cs * np * DWIDTH,
            /*num_elements=*/cs * np, addr(0, "x_t_spm"), /*num_cores=*/NUM_CORES);
    }
    const int amc = action_mlp_cores_;
    // [A2] time_all[bit] → time_spm, 作为 encoder 第一个 GEMM (A4) 的 bias。
    //      FIXED src: time_all_ 是稳定的注册权重, bit 偏移在 BUILD 期烘死
    //      (每个 body 单独 emit 一次, body_iter 是编译期常量)。
    //      ⚠️ 多核档 A4 走**列并行** ⇒ 核 c 只负责输出列 [c*h/amc, (c+1)*h/amc)，
    //         bias 必须**按核切片**（scatter），不能广播 —— 广播会让每个核都加
    //         bias[0:h/amc]，是静默算错。
    if (amc == 1) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_TIME_BROADCAST_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_FIXED),
                /*resolved_flags=*/0,
                {action_mlp_cores_, h, h / action_mlp_cores_, DWIDTH});
        }
        rpu_launch_ddr_broadcast_spm_dma(
            time_all_.data_ptr<c10::Half>() + bit * h, h, addr(0, "time_spm"),
            /*num_cores=*/1);
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_TIME_SCATTER_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_SCATTER_TO_SPM_FIXED),
                /*resolved_flags=*/0,
                {action_mlp_cores_, h, h / action_mlp_cores_, DWIDTH});
        }
        rpu_launch_ddr_scatter_spm_dma(
            time_all_.data_ptr<c10::Half>() + bit * h,
            /*elements_per_core=*/h / amc,
            /*core_stride_bytes=*/(h / amc) * DWIDTH,
            addr(0, "time_spm"), /*num_cores=*/amc);
    }
    // [A3] mo_bias → mo_b_spm (内容恒定, FIXED src)。
    //      单核档只有核 0 用得到；多核档 all-reduce 后每核都要按列加一次 bias
    //      （`1xC_NxC` 没有 num_cores 参数，8 核齐发），所以要广播满 8 核。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA,
            HYVLA_EXPERT_PRE_MO_BIAS_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_FIXED),
            /*resolved_flags=*/0, {action_mlp_cores_, h});
    }
    rpu_launch_ddr_broadcast_spm_dma(
        mo_bias_.data_ptr<c10::Half>(), h, addr(0, "mo_b_spm"), /*num_cores=*/amc);

    // [A4] enc_h = x_t_spm · wcᵀ + time_bias  (M=cs, N=h, K=np)。
    //      ⚠️ 多核档必须**列并行**：A6 是行(K)并行，它要求核 c 手里正好是输入的
    //      第 c 段 K —— 而列并行 A4 的输出天然就是那一段。两者相接**零通信**，
    //      就是层体里 gate/up(列) → down(行) 的同一套 TP 模式。
    //      若 A4 只在单核执行，其他核的 enc_h_spm 无有效输入。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_EXPERT_PRE_WC_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {cs, h, np, 1, action_mlp_cores_, 1});
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "x_t_spm"), wc_, addr(0, "enc_h_spm"),
        /*M=*/cs, /*N=*/h, /*K=*/np, /*partition=*/1, /*num_cores=*/amc,
        /*bias_spm_addr=*/addr(0, "time_spm"));
    // [A5] SiLU 原地。逐元素 ⇒ 每核只处理自己那 h/amc 列。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ACTIVATION,
            HYVLA_EXPERT_PRE_ENCODER_ACTIVATION_SITE,
            static_cast<int64_t>(HyVlaExpertActivationRoute::SILU_UNARY),
            /*resolved_flags=*/0,
            {action_mlp_cores_, cs * (h / action_mlp_cores_)});
    }
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "enc_h_spm"), addr(0, "enc_h_spm"), cs * (h / amc), ValuOpType::SILU,
        /*is_gelu=*/false, /*num_cores=*/amc);
    // [A6] emb = silu(enc_h) * mo^T + mo_bias -> emb_core0_spm.
    // The multicore path partitions K and all-reduces before adding bias.
    if (amc == 1) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                HYVLA_EXPERT_PRE_MO_SINGLE_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {cs, h, h, 1, action_mlp_cores_, 1});
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "enc_h_spm"), mo_, addr(0, "emb_core0_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
            /*bias_spm_addr=*/addr(0, "mo_b_spm"));
    } else {
        // A6-1 K 方向切核 ⇒ 每核一份 [cs,h] 部分和。**bias 必须留到 all-reduce 之后**。
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR,
                HYVLA_EXPERT_PRE_MO_MULTI_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {cs, h, h, 0, action_mlp_cores_, 0});
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "enc_h_spm"), mo_, addr(0, "mo_part_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/h, /*partition=*/0, /*num_cores=*/amc,
            /*bias_spm_addr=*/0u);
        // A6-2 enc_h_spm 到这里已死 ⇒ 就地清零，当 all-reduce 的零残差
        //      （该 kernel 要求 input/residual/output 三地址互不相同）。
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_PRE_MO_ZERO_SITE,
                static_cast<int64_t>(
                    HyVlaExpertGraphScheduleRoute::ZERO_A6_RESIDUAL),
                /*resolved_flags=*/0,
                {action_mlp_cores_, cs * h});
        }
        rpu_launch_fill_spm_kernel(
            addr(0, "enc_h_spm"), cs * h, c10::Half(0.0f), /*num_cores=*/amc);
        // A6-3 fused ring all-reduce；调度器按 per-core chunk 自动选 pace/nopace。
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                HYVLA_EXPERT_PRE_MO_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(cs, h),
                /*resolved_flags=*/0,
                {cs, h, action_mlp_cores_, action_mlp_cores_});
        }
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "mo_part_spm"), addr(0, "enc_h_spm"), addr(0, "emb_core0_spm"),
            /*M=*/cs, /*N=*/h, /*input_num_cores=*/amc, /*output_num_cores=*/amc);
        // A6-4 bias 按列广播加一次（8 核齐发，各核结果一致；A7 只读核 0）。
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_PRE_MO_BIAS_ADD_SITE,
                static_cast<int64_t>(
                    HyVlaExpertGraphScheduleRoute::ADD_A6_BIAS_AFTER_REDUCE),
                /*resolved_flags=*/0,
                {action_mlp_cores_, cs, h});
        }
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
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_EXPERT_POST_OP_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {cs, np, h, 1, 1, 1});
    }
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
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA,
            HYVLA_EXPERT_POST_TRAJECTORY_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::SPM_TO_DDR_MUTABLE),
            /*resolved_flags=*/0, {num_steps_, cs * np});
    }
    rpu_launch_spm_copy_ddr_dma_mutable(
        addr(0, "x_t_spm"), &x_traj_dst_base_,
        /*dst_offset_bytes=*/bit * cs * np * 2, cs * np);
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamic_config — 基类断言单 chunk，并统一准备显式 mask；本类直接复用
// CausalDecoderModel 的稳定 PreparedMask。
// ─────────────────────────────────────────────────────────────────────────────
ModelDynamicConfig HyVlaExpertModel::dynamic_config(const ChunkPlan& plan) {
    ModelDynamicConfig cfg = CausalDecoderModel::dynamic_config(plan);
    // Prefix history is shared with the VLM tower through DDR KV.  Keep this
    // owner on that certified path; raw-SPM KV attention is not admissible.
    cfg.attention_policy = AttentionExecutionPolicy::DDR_KV;
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "hyvla_expert dynamic_config: explicit attention mask required "
                "(denoise 是非因果 prefix attention, prefix 块还要屏蔽 padding 行)");
    return cfg;
}

ModelDynamicConfig HyVlaExpertModel::planning_dynamic_config(
    const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    cfg.attention_policy = AttentionExecutionPolicy::DDR_KV;
    return cfg;
}

FmbPhysicalExecutionManifest HyVlaExpertModel::physical_manifest_for_candidate(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len,
    int64_t position) const {
    FmbPhysicalExecutionManifest manifest;
    manifest.state = FmbPhysicalManifestState::COMPLETE;
    manifest.logical_length = logical_len;
    manifest.physical_length = physical_len;
    manifest.execution_padding_rows = physical_len - logical_len;
    manifest.kv_logical_length = position + logical_len;
    TORCH_CHECK(
        plan.qkv.chunks.size() == 1 && plan.compute.chunks.size() == 1,
        "HyVlaExpertModel COMPLETE descriptor requires one action chunk");
    const ChunkInfo& kv_chunk = plan.qkv.chunks.front();
    const ChunkInfo& chunk = plan.compute.chunks.front();
    TORCH_CHECK(
        kv_chunk.idx == chunk.idx && kv_chunk.offset == chunk.offset &&
            kv_chunk.len == chunk.len,
        "HyVlaExpertModel COMPLETE descriptor requires identical QKV and "
        "compute action chunks");
    const int64_t invocation = chunk.idx;
    const int64_t h = hidden_size();
    const int64_t nq = num_q_heads();
    const int64_t nkv = num_kv_heads();
    const int64_t hd = head_dim();
    const int64_t tp = attn_tp();
    const int64_t local_q = nq / tp;
    const int64_t local_kv_heads = nkv / tp;
    const int64_t padded_rows = Align(layout.chunk_size, int64_t{16});
    const int64_t norm_rows = cold_rmsnorm_pad16_
        ? padded_rows : chunk.len;
    const int64_t q_norm_rows = cold_rmsnorm_pad16_
        ? Align(layout.chunk_size * local_q, int64_t{16})
        : chunk.len * local_q;
    const int64_t k_norm_rows = (cold_rmsnorm_pad16_
        ? padded_rows : chunk.len) * local_kv_heads;
    const int64_t kv_physical_rows = cold_kvpad16_
        ? Align(layout.chunk_size, int64_t{16}) : kv_chunk.len;
    manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
    const KvInsertSegmentPlan kv_plan =
        resolve_kvinsert_plan_auto(
            HYVLA_EXPERT_KV_SITE, manifest.graph_lifecycle,
            position + kv_chunk.offset, kv_chunk.len, kv_physical_rows,
            attn_tp(), num_kv_heads(), head_dim(),
            HYVLA_EXPERT_KV_CAPABILITIES);
    const KvInsertRouteArguments kv_arguments =
        rpu_kvinsert_route_arguments(
            kv_plan, attn_tp(), num_kv_heads(), head_dim());
    manifest.kv_insert_physical_rows = kv_plan.physical_rows();
    manifest.linear_accumulation = FmbLinearAccumulationPolicy::ACC16;

    const auto append = [&](FmbRouteFamily family, int64_t site_id,
                            int64_t selector,
                            std::vector<int64_t> arguments = {},
                            int64_t flags = 0) {
        manifest.routes.push_back({site_id, family, selector, flags,
                                   std::move(arguments), invocation});
    };
    const auto append_linear = [&](int64_t site_id,
                                   std::vector<int64_t> arguments = {}) {
        append(FmbRouteFamily::LINEAR, site_id,
               static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
               std::move(arguments));
    };
    const auto append_norm = [&](int64_t site_id, int64_t rows,
                                 int64_t cols) {
        append(
            FmbRouteFamily::NORMALIZATION, site_id,
            static_cast<int64_t>(
                cold_rmsnorm_pad16_
                    ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                    : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
            {cold_rmsnorm_pad16_ ? 1 : 0, rows, cols});
    };

    append(
        FmbRouteFamily::GRAPH_SCHEDULE,
        HYVLA_EXPERT_GRAPH_SCHEDULE_SITE,
        static_cast<int64_t>(
            cold_fast_replay_
                ? HyVlaExpertGraphScheduleRoute::FAST_REPLAY_SKIP_LAYER_LOOP
                : HyVlaExpertGraphScheduleRoute::REEMIT_LAYER_LOOP),
        {cold_fast_replay_ ? 1 : 0,
         cold_fast_replay_preload_ ? 1 : 0,
         unroll_mode_ ? 1 : 0,
         unroll_mode_ ? num_steps_ : 1});
    append(
        FmbRouteFamily::GRAPH_SCHEDULE,
        HYVLA_EXPERT_MASK_SCHEDULE_SITE,
        mask_schedule_for_layout(layout),
        {cold_mask_once_ ? 1 : 0, chunk.len, kv_chunk.kv_seq_len,
         tp, num_layers()});
    append(
        FmbRouteFamily::KV_INSERT, HYVLA_EXPERT_KV_PADDING_SITE,
        static_cast<int64_t>(
            cold_kvpad16_ ? HyVlaExpertKvPaddingRoute::PAD16_ROWS
                          : HyVlaExpertKvPaddingRoute::LOGICAL_ROWS),
        {cold_kvpad16_ ? 1 : 0, kv_chunk.len,
         cold_kvpad16_ ? kv_physical_rows : 0,
         kv_physical_rows, tp, nkv, hd});

    append_norm(HYVLA_EXPERT_INPUT_NORM_SITE, norm_rows, h);
    append_linear(HYVLA_EXPERT_QKV_LINEAR_SITE);
    append(
        FmbRouteFamily::ROPE,
        cold_partial_rope_ ? HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE
                           : HYVLA_EXPERT_Q_ROPE_SITE,
        static_cast<int64_t>(
            cold_partial_rope_ ? HyVlaExpertRopeRoute::PARTIAL_MROPE
                               : HyVlaExpertRopeRoute::ROPE_1D),
        {cold_partial_rope_ ? 1 : 0, position + chunk.offset,
         chunk.len, local_q, hd, tp});
    append_norm(HYVLA_EXPERT_Q_NORM_SITE, q_norm_rows, hd);
    append(
        FmbRouteFamily::ROPE,
        cold_partial_rope_ ? HYVLA_EXPERT_K_PARTIAL_ROPE_SITE
                           : HYVLA_EXPERT_K_ROPE_SITE,
        static_cast<int64_t>(
            cold_partial_rope_ ? HyVlaExpertRopeRoute::PARTIAL_MROPE
                               : HyVlaExpertRopeRoute::ROPE_1D),
        {cold_partial_rope_ ? 1 : 0, position + chunk.offset,
         chunk.len, local_kv_heads, hd, tp});
    append_norm(HYVLA_EXPERT_K_NORM_SITE, k_norm_rows, hd);
    // The shared VLM prefix remains DDR_REQUIRED across every denoise step.
    // Freeze the exact pure-resolver segment plan; no legacy selector is read
    // while emitting the graph.
    append(
        FmbRouteFamily::KV_INSERT, HYVLA_EXPERT_KV_SITE,
        static_cast<int64_t>(kv_plan.route()),
        std::vector<int64_t>(kv_arguments.begin(), kv_arguments.end()),
        HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED);
    append(
        FmbRouteFamily::ATTENTION, HYVLA_EXPERT_ATTN_SITE,
        static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV));
    append(
        FmbRouteFamily::ALL_REDUCE,
        HYVLA_EXPERT_PREPARE_ATTN_ALL_REDUCE_SITE,
        static_cast<int64_t>(
            HyVlaExpertAllReduceRoute::PREPARE_RING_INPUT),
        {chunk.len, h, tp});
    append_linear(
        HYVLA_EXPERT_O_LINEAR_SITE,
        {chunk.len, h, nq * hd, 0, tp});
    append(
        FmbRouteFamily::ALL_REDUCE,
        HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE,
        fmb_ring_all_reduce_route_selector(chunk.len, h),
        {chunk.len, h, tp, NUM_CORES});
    append_norm(HYVLA_EXPERT_POST_NORM_SITE, norm_rows, h);
    append_linear(
        HYVLA_EXPERT_GATE_LINEAR_SITE,
        {chunk.len, intermediate_size(), h, 1, NUM_CORES});
    append_linear(
        HYVLA_EXPERT_UP_LINEAR_SITE,
        {chunk.len, intermediate_size(), h, 1, NUM_CORES});
    const int64_t elems_mlp = chunk.len * (intermediate_size() / NUM_CORES);
    if (cold_silu_mul_) {
        append(
            FmbRouteFamily::ACTIVATION,
            HYVLA_EXPERT_MLP_SILU_MUL_SITE,
            static_cast<int64_t>(
                HyVlaExpertActivationRoute::SILU_MUL_FUSED),
            {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES});
    } else {
        append(
            FmbRouteFamily::ACTIVATION, HYVLA_EXPERT_MLP_SILU_SITE,
            static_cast<int64_t>(
                HyVlaExpertActivationRoute::SILU_UNARY),
            {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES});
        append(
            FmbRouteFamily::ACTIVATION, HYVLA_EXPERT_MLP_MUL_SITE,
            static_cast<int64_t>(HyVlaExpertActivationRoute::MUL),
            {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES});
    }
    append_linear(
        HYVLA_EXPERT_DOWN_LINEAR_SITE,
        {chunk.len, h, intermediate_size(), 0, NUM_CORES});
    append(
        FmbRouteFamily::ALL_REDUCE,
        HYVLA_EXPERT_MLP_ALL_REDUCE_SITE,
        fmb_ring_all_reduce_route_selector(chunk.len, h),
        {chunk.len, h, NUM_CORES, NUM_CORES});
    append_norm(HYVLA_EXPERT_FINAL_NORM_SITE, norm_rows, h);

    if (unroll_mode_) {
        TORCH_CHECK(
            action_dim_pad_ > 0 && num_steps_ > 0 &&
                (action_mlp_cores_ == 1 || action_mlp_cores_ == NUM_CORES),
            "HyVlaExpertModel COMPLETE unroll descriptor requires bound "
            "action weights and a supported core count");
        const int64_t amc = action_mlp_cores_;
        const int64_t cs = expert_chunk_size_;
        const int64_t np = action_dim_pad_;
        append(
            FmbRouteFamily::MUTABLE_DMA, HYVLA_EXPERT_PRE_X0_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
            {unroll_mode_ ? 1 : 0, cs * np, NUM_CORES});
        if (amc > 1 && num_steps_ > 1) {
            append(
                FmbRouteFamily::MUTABLE_DMA,
                HYVLA_EXPERT_PRE_X_FEEDBACK_DMA_SITE,
                static_cast<int64_t>(
                    HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_MUTABLE),
                {amc, num_steps_, cs * np, NUM_CORES});
        }
        append(
            FmbRouteFamily::MUTABLE_DMA,
            amc == 1 ? HYVLA_EXPERT_PRE_TIME_BROADCAST_DMA_SITE
                     : HYVLA_EXPERT_PRE_TIME_SCATTER_DMA_SITE,
            static_cast<int64_t>(
                amc == 1
                    ? HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_FIXED
                    : HyVlaExpertDmaRoute::DDR_SCATTER_TO_SPM_FIXED),
            {amc, h, h / amc, DWIDTH});
        append(
            FmbRouteFamily::MUTABLE_DMA,
            HYVLA_EXPERT_PRE_MO_BIAS_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::DDR_BROADCAST_TO_SPM_FIXED),
            {amc, h});
        append_linear(
            HYVLA_EXPERT_PRE_WC_LINEAR_SITE,
            {cs, h, np, 1, amc, 1});
        append(
            FmbRouteFamily::ACTIVATION,
            HYVLA_EXPERT_PRE_ENCODER_ACTIVATION_SITE,
            static_cast<int64_t>(HyVlaExpertActivationRoute::SILU_UNARY),
            {amc, cs * (h / amc)});
        append_linear(
            amc == 1 ? HYVLA_EXPERT_PRE_MO_SINGLE_LINEAR_SITE
                     : HYVLA_EXPERT_PRE_MO_MULTI_LINEAR_SITE,
            {cs, h, h, amc == 1 ? 1 : 0, amc, amc == 1 ? 1 : 0});
        if (amc > 1) {
            append(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_PRE_MO_ZERO_SITE,
                static_cast<int64_t>(
                    HyVlaExpertGraphScheduleRoute::ZERO_A6_RESIDUAL),
                {amc, cs * h});
            append(
                FmbRouteFamily::ALL_REDUCE,
                HYVLA_EXPERT_PRE_MO_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(cs, h),
                {cs, h, amc, amc});
            append(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_EXPERT_PRE_MO_BIAS_ADD_SITE,
                static_cast<int64_t>(
                    HyVlaExpertGraphScheduleRoute::ADD_A6_BIAS_AFTER_REDUCE),
                {amc, cs, h});
        }
        append_linear(
            HYVLA_EXPERT_POST_OP_LINEAR_SITE,
            {cs, np, h, 1, 1, 1});
        append(
            FmbRouteFamily::MUTABLE_DMA,
            HYVLA_EXPERT_POST_TRAJECTORY_DMA_SITE,
            static_cast<int64_t>(
                HyVlaExpertDmaRoute::SPM_TO_DDR_MUTABLE),
            {num_steps_, cs * np});
    }
    append_causal_decoder_preload_manifest_routes(manifest);
    append_fmb_shared_runtime_routes(
        manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA);
    return manifest;
}

FmbPhysicalManifestForwardCapability
HyVlaExpertModel::physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& /*manifest*/) const {
    return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
}

// ─────────────────────────────────────────────────────────────────────────────
// subclass_chunk_size_valid — suffix 整块进 SDPA
// ─────────────────────────────────────────────────────────────────────────────
bool HyVlaExpertModel::subclass_chunk_size_valid(
    int64_t cs, int64_t seq_len, int64_t /*position*/) const {
    return cs >= seq_len && chunk_within_envelope(cs);
}

std::vector<int64_t> HyVlaExpertModel::resolve_stage_domain(
    int64_t seq_len, int64_t prefix_len, int64_t mask_kv_len) {
    TORCH_CHECK(seq_len == expert_chunk_size_,
                "hyvla_expert_resolve_stage_domain: seq_len must match the "
                "fixed action suffix geometry");
    TORCH_CHECK(prefix_len >= 0 && mask_kv_len == prefix_len + seq_len,
                "hyvla_expert_resolve_stage_domain: mask width must equal "
                "prefix_len + seq_len");
    const at::Tensor mask = at::empty(
        {1, mask_kv_len},
        at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
    return encode_fmb_prefill_stage_domain(
        resolve_prefill_stage_domain_for_shape(
            seq_len, prefix_len, std::optional<at::Tensor>(mask),
            /*is_causal=*/false));
}

// ─────────────────────────────────────────────────────────────────────────────
// build_layer_subgraph — 单塔层体 (基类 8 相, 步 3 换序)
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    TORCH_CHECK(
        ctx().has_complete_physical_manifest(),
        "HyVlaExpertModel requires a COMPLETE physical descriptor");
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
    const int64_t allocation_rows = ctx().stage_plan.compute.chunks.front().len;
    const int64_t kv_pad_rows_ = cold_kvpad16_
        ? Align(allocation_rows, int64_t{16}) : 0;
    const int64_t rmsnorm_pad_rows_ = cold_rmsnorm_pad16_
        ? Align(allocation_rows, int64_t{16}) : 0;
    const int64_t rmsnorm_pad_qrows_ = cold_rmsnorm_pad16_
        ? Align(allocation_rows * local_q, int64_t{16}) : 0;


    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_EXPERT_GRAPH_SCHEDULE_SITE,
            static_cast<int64_t>(
                cold_fast_replay_
                    ? HyVlaExpertGraphScheduleRoute::FAST_REPLAY_SKIP_LAYER_LOOP
                    : HyVlaExpertGraphScheduleRoute::REEMIT_LAYER_LOOP),
            /*resolved_flags=*/0,
            {cold_fast_replay_ ? 1 : 0,
             cold_fast_replay_preload_ ? 1 : 0,
             unroll_mode_ ? 1 : 0,
             unroll_mode_ ? num_steps_ : 1},
            chunk.idx);
    }

    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }

    // ── 步 1: input RMSNorm ──
    // RMSNORM_PAD16：补齐到 16 的倍数行才能走 `llama_rms_norm_v16`（见文件头）。
    // 多算的 [seq_len, pad) 行下游按 M=seq_len 消费 ⇒ 永不被读。
    const int64_t nrows  = rmsnorm_pad_rows_  ? rmsnorm_pad_rows_  : seq_len;
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION,
            HYVLA_EXPERT_INPUT_NORM_SITE,
            static_cast<int64_t>(
                cold_rmsnorm_pad16_
                    ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                    : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
            /*resolved_flags=*/0,
            {cold_rmsnorm_pad16_ ? 1 : 0, nrows, h}, chunk.idx);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "norm_w"), nrows, h, eps_expert_);

    // ── 步 2: QKV (无 bias — Hy-VLA attention_bias=false) ──
    auto qkv = [&](const at::Tensor& w, const at::Tensor& ws,
                   const char* out, int64_t n) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVLA_EXPERT_QKV_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), w, addr(0, out),
            seq_len, n, h, /*partition=*/1, /*num_cores=*/tp,
            /*bias_spm_addr=*/0u, /*scale=*/ws);
    };
    qkv(lw.q_w, lw.q_ws, "q", nq * hd);
    qkv(lw.k_w, lw.k_ws, "k", nkv * hd);
    qkv(lw.v_w, lw.v_ws, "v", nkv * hd);

    // ── 步 3: **先 RoPE, 再 q/k head-norm**(与基类及其余 7 个 emitter 相反) ──
    // 依据 vendor modeling_dual_tower.py: apply_rotary_pos_emb 之后才调
    // query_layernorm/key_layernorm, 且两塔共用 VLM 层的那一份 (绑定方负责喂对)。
    // 与 VLM 塔 (rpu_hyvla_vlm_model.cpp 步 3) 完全同构, 只是这里没有孪生分支。
    // 换序可行的前提: 两个 kernel 都是 src/dst 分离、无隐藏状态的独立 SPM launch,
    // 且 QK-norm 的 framing (M=seq*heads, C=head_dim) 不受 RoPE 影响 —— RoPE 不改
    // [seq, heads, hd] 布局。
    c10::Half* cos_ptr = cos_ref_.data_ptr<c10::Half>();
    c10::Half* sin_ptr = sin_ref_.data_ptr<c10::Half>();

    // PARTIAL_ROPE uses the same tables and offsets with grid
    // (ceil(hd/2/64), ceil(seq/64)).
    const bool prope = cold_partial_rope_;
    const int64_t local_kv_heads = nkv / tp;

    if (prope) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE,
                HYVLA_EXPERT_Q_PARTIAL_ROPE_SITE,
                static_cast<int64_t>(HyVlaExpertRopeRoute::PARTIAL_MROPE),
                /*resolved_flags=*/0,
                {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                 seq_len, local_q, hd, tp}, chunk.idx);
        }
        rpu_launch_partial_mrope_spm_kernel(
            addr(0, "q"), addr(0, "output"), cos_ptr, sin_ptr,
            cos_sin_start, seq_len, local_q, hd, /*rotary_dim=*/hd, tp);
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, HYVLA_EXPERT_Q_ROPE_SITE,
                static_cast<int64_t>(HyVlaExpertRopeRoute::ROPE_1D),
                /*resolved_flags=*/0,
                {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                 seq_len, local_q, hd, tp}, chunk.idx);
        }
        rpu_launch_rope_spm_kernel(addr(0, "q"), addr(0, "output"),
                                   cos_ptr, sin_ptr, seq_len, local_q, hd,
                                   cos_sin_start, tp);
    }
    const int64_t q_norm_rows = rmsnorm_pad_qrows_
        ? rmsnorm_pad_qrows_ : seq_len * local_q;
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION, HYVLA_EXPERT_Q_NORM_SITE,
            static_cast<int64_t>(
                cold_rmsnorm_pad16_
                    ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                    : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
            /*resolved_flags=*/0,
            {cold_rmsnorm_pad16_ ? 1 : 0, q_norm_rows, hd}, chunk.idx);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "output"), addr(0, "q"),
        layer_addr(layer_idx, 0, "q_norm_w"),
        q_norm_rows, hd, eps_expert_);

    if (prope) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE,
                HYVLA_EXPERT_K_PARTIAL_ROPE_SITE,
                static_cast<int64_t>(HyVlaExpertRopeRoute::PARTIAL_MROPE),
                /*resolved_flags=*/0,
                {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                 seq_len, local_kv_heads, hd, tp}, chunk.idx);
        }
        rpu_launch_partial_mrope_spm_kernel(
            addr(0, "k"), addr(0, "input_norm"), cos_ptr, sin_ptr,
            cos_sin_start, seq_len, local_kv_heads, hd, /*rotary_dim=*/hd, tp);
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, HYVLA_EXPERT_K_ROPE_SITE,
                static_cast<int64_t>(HyVlaExpertRopeRoute::ROPE_1D),
                /*resolved_flags=*/0,
                {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                 seq_len, local_kv_heads, hd, tp}, chunk.idx);
        }
        rpu_launch_rope_spm_kernel(addr(0, "k"), addr(0, "input_norm"),
                                   cos_ptr, sin_ptr, seq_len, local_kv_heads,
                                   hd, cos_sin_start, tp);
    }
    // k 槽已被 KVPAD16 撑到 Align(cs,16) 行 × local_kv 元素 ⇒ 这里补齐不需要额外撑槽。
    const int64_t k_norm_rows =
        (rmsnorm_pad_rows_ ? rmsnorm_pad_rows_ : seq_len) * local_kv_heads;
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION, HYVLA_EXPERT_K_NORM_SITE,
            static_cast<int64_t>(
                cold_rmsnorm_pad16_
                    ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                    : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
            /*resolved_flags=*/0,
            {cold_rmsnorm_pad16_ ? 1 : 0, k_norm_rows, hd}, chunk.idx);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "input_norm"), addr(0, "k"),
        layer_addr(layer_idx, 0, "k_norm_w"),
        k_norm_rows, hd, eps_expert_);

    // ── 步 4: KV insert (追加在 prefix 之后) + SDPA + o_proj + 残差归约 ──
    // 调用方每步前 reset_to_position(prefix_len), 所以这 51 行每步覆写同一段,
    // 等价于 vendor 的 copy.deepcopy 冻结 prefix、suffix KV 用完即弃。
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    const int64_t expected_kv_physical_rows = cold_kvpad16_
        ? kv_pad_rows_ : seq_len;
    ctx().consume_physical_route(
        FmbRouteFamily::KV_INSERT,
        HYVLA_EXPERT_KV_PADDING_SITE,
        static_cast<int64_t>(
            cold_kvpad16_ ? HyVlaExpertKvPaddingRoute::PAD16_ROWS
                          : HyVlaExpertKvPaddingRoute::LOGICAL_ROWS),
        /*resolved_flags=*/0,
        {cold_kvpad16_ ? 1 : 0, seq_len, kv_pad_rows_,
         expected_kv_physical_rows, tp, nkv, hd}, chunk.idx);
    const FmbRouteManifestEntry& route = ctx().find_physical_route(
        FmbRouteFamily::KV_INSERT, HYVLA_EXPERT_KV_SITE, chunk.idx);
    const KvInsertSegmentPlan kv_plan =
        restore_kvinsert_plan(
            HYVLA_EXPERT_KV_SITE, route.arguments, tp, nkv, hd);
    TORCH_CHECK(
        kv_plan.logical_rows() == seq_len &&
            kv_plan.physical_rows() == expected_kv_physical_rows &&
            kv_plan.segment(0).position == cos_sin_start,
        "HyVlaExpertModel KV descriptor geometry drift at invocation ",
        chunk.idx);
    ctx().consume_physical_route(
        FmbRouteFamily::KV_INSERT, HYVLA_EXPERT_KV_SITE,
        static_cast<int64_t>(kv_plan.route()),
        HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED,
        route.arguments, chunk.idx);
    rpu_launch_insert_kvcache_spm_unified_with_plan(
        k_cache, v_cache,
        addr_offset("k").value, addr_offset("v").value,
        nkv, hd, tp,
        /*k_cache_batch_offset_elems=*/0,
        /*v_cache_batch_offset_elems=*/0,
        kv_pad_rows_, kv_plan);
    const uint32_t mask_off = addr_offset("sdpa_mask").value;
    const auto& mask_route = ctx().find_physical_route(
        FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_MASK_SCHEDULE_SITE,
        chunk.idx);
    if (ctx().body_iter == 0 && layer_idx == 0) {
        // Revalidate the minted schedule against exactly the complete layout
        // used for allocation, once before any mask reuse (not per layer).
        LayoutContext layout;
        layout.chunk_size = allocation_rows;
        layout.max_kv_seq_len = ctx().attention_mask->size(-1);
        layout.num_layers = num_layers();
        layout.use_attn_mask = true;
        layout.is_causal = ctx().is_causal;
        layout.batch_size = ctx().batch_size;
        layout.attention_policy = ctx().attention_policy;
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, HYVLA_EXPERT_MASK_SCHEDULE_SITE,
            mask_schedule_for_layout(layout), /*resolved_flags=*/0,
            {cold_mask_once_ ? 1 : 0, seq_len, chunk.kv_seq_len,
             tp, num_layers()}, chunk.idx);
    }
    const auto mask_schedule =
        static_cast<HyVlaExpertGraphScheduleRoute>(mask_route.selector);
    const bool upload_mask =
        mask_schedule == HyVlaExpertGraphScheduleRoute::MASK_EACH_LAYER ||
        (layer_idx == 0 &&
         (mask_schedule == HyVlaExpertGraphScheduleRoute::MASK_FIRST_LAYER_ONLY ||
          ctx().body_iter == 0));
    if (upload_mask) {
        sdpa_dma_mask_to_spm(prepared_attn_mask_, mask_off, seq_len, chunk.kv_seq_len, tp);
    }
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_attention_route(
            HYVLA_EXPERT_ATTN_SITE, AttentionExecutionPolicy::DDR_KV,
            chunk.idx);
    }
    rpu_launch_sdpa_spm_dispatch(
        SdpaKernelType::FLASH_ATTN_SPM,
        k_cache, v_cache, prepared_attn_mask_.mask_type, c10::nullopt,
        addr_offset("q").value, addr_offset("output").value,
        addr_offset("sdpa_tmp").value, mask_off,
        seq_len, nq, nkv, hd, chunk.kv_seq_len, tp, NUM_CORES);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            HYVLA_EXPERT_PREPARE_ATTN_ALL_REDUCE_SITE,
            static_cast<int64_t>(
                HyVlaExpertAllReduceRoute::PREPARE_RING_INPUT),
            /*resolved_flags=*/0, {seq_len, h, tp}, chunk.idx);
    }
    rpu_prepare_ring_all_reduce_input(addr(0, "oproj"), seq_len, h, tp);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_EXPERT_O_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {seq_len, h, nq * hd, 0, tp}, chunk.idx);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0u, /*scale=*/lw.o_ws);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            HYVLA_EXPERT_ATTN_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0,
            {seq_len, h, tp, NUM_CORES}, chunk.idx);
    }
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
        seq_len, h, tp, NUM_CORES);

    // ── 步 5: post-attention RMSNorm ──
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION,
            HYVLA_EXPERT_POST_NORM_SITE,
            static_cast<int64_t>(
                cold_rmsnorm_pad16_
                    ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                    : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
            /*resolved_flags=*/0,
            {cold_rmsnorm_pad16_ ? 1 : 0, nrows, h}, chunk.idx);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "residual1"),
        layer_addr(layer_idx, 0, "post_norm_w"), nrows, h, eps_expert_);

    // ── 步 6: SwiGLU MLP + 残差归约 ──
    // 手写而非用基类 emit_mlp_pipeline: 保持与 VLM 塔同构的可读性, 且这里的
    // 槽名/核数组合与基类默认一致, 没有额外分支。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_EXPERT_GATE_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {seq_len, intermediate_size(), h, 1, NUM_CORES}, chunk.idx);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
        seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
        /*bias_spm_addr=*/0u, /*scale=*/lw.gate_ws);
    if (!cold_silu_mul_) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ACTIVATION,
                HYVLA_EXPERT_MLP_SILU_SITE,
                static_cast<int64_t>(
                    HyVlaExpertActivationRoute::SILU_UNARY),
                /*resolved_flags=*/0,
                {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES}, chunk.idx);
        }
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "gate"), addr(0, "gate"), elems_mlp, ValuOpType::SILU,
            /*is_gelu=*/false, NUM_CORES);
    }
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_EXPERT_UP_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {seq_len, intermediate_size(), h, 1, NUM_CORES}, chunk.idx);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), lw.up_w, addr(0, "up"),
        seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
        /*bias_spm_addr=*/0u, /*scale=*/lw.up_ws);
    if (cold_silu_mul_) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ACTIVATION,
                HYVLA_EXPERT_MLP_SILU_MUL_SITE,
                static_cast<int64_t>(
                    HyVlaExpertActivationRoute::SILU_MUL_FUSED),
                /*resolved_flags=*/0,
                {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES}, chunk.idx);
        }
        rpu_launch_silu_mul_spm_kernel(
            addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
            elems_mlp, NUM_CORES);
    } else {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ACTIVATION,
                HYVLA_EXPERT_MLP_MUL_SITE,
                static_cast<int64_t>(HyVlaExpertActivationRoute::MUL),
                /*resolved_flags=*/0,
                {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES}, chunk.idx);
        }
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
            elems_mlp, ValuOpType::MUL, c10::Half(1.0f), NUM_CORES);
    }
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_EXPERT_DOWN_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {seq_len, h, intermediate_size(), 0, NUM_CORES}, chunk.idx);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "gate"), lw.down_w, addr(0, "down"),
        seq_len, h, intermediate_size(), /*partition=*/0, NUM_CORES,
        /*bias_spm_addr=*/0u, /*scale=*/lw.down_ws);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            HYVLA_EXPERT_MLP_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0,
            {seq_len, h, NUM_CORES, NUM_CORES}, chunk.idx);
    }
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
        seq_len, h, NUM_CORES, NUM_CORES);

    // ── 步 7 (末层): final RMSNorm (原地, 单份 γ) ──
    if (is_last_layer) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::NORMALIZATION,
                HYVLA_EXPERT_FINAL_NORM_SITE,
                static_cast<int64_t>(
                    cold_rmsnorm_pad16_
                        ? HyVlaExpertNormalizationRoute::RMSNORM_V16_ROWS
                        : HyVlaExpertNormalizationRoute::RMSNORM_SCALAR_ROWS),
                /*resolved_flags=*/0,
                {cold_rmsnorm_pad16_ ? 1 : 0, nrows, h}, chunk.idx);
        }
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
    const at::Tensor& attn_mask_4d, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor)
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

    rpu_ddr_flush_force(x_emb.data_ptr<c10::Half>());

    return CausalDecoderModel::forward(
        x_emb, k_caches, v_caches,
        std::optional<at::Tensor>(attn_mask_4d),
        /*position=*/prefix_len, /*is_causal=*/false,
        /*position_ids=*/std::nullopt, /*deepstack=*/std::nullopt,
        /*rope_cos_il=*/std::nullopt, /*rope_sin_il=*/std::nullopt,
        /*cos_sin_offset=*/-1, /*batch_slot=*/0,
        /*allow_batch_decode=*/false, /*planned_chunk_size=*/0,
        planned_stage_descriptor);
}

// ─────────────────────────────────────────────────────────────────────────────
// unroll_forward — 一次 forward 跑完 num_steps 步 Euler
// ─────────────────────────────────────────────────────────────────────────────
void HyVlaExpertModel::unroll_forward(
    const at::Tensor& x0_rpu, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches, const at::Tensor& state_emb,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& attn_mask_4d,
    at::Tensor& x_traj, double dt, int64_t prefix_len, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor)
{
    TORCH_CHECK(unroll_mode_, "hyvla_expert unroll_forward 之前必须 set_action_weights");
    TORCH_CHECK(num_steps == num_steps_, "num_steps(", num_steps,
                ") 必须等于 set_action_weights 的 num_steps(", num_steps_, ")");
    const int64_t cs = expert_chunk_size_, np = action_dim_pad_, h = hidden_size();

    // Bound the unrolled graph size before recording the repeated body.
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

    // 一次 forward —— run_all_layers 内部循环 body_iterations(=num_steps_) 次,
    // pre/post hook 每次以递增的 ctx().body_iter 重新 emit。
    (void) CausalDecoderModel::forward(
        emb_stage_, k_caches, v_caches,
        std::optional<at::Tensor>(attn_mask_4d),
        /*position=*/prefix_len, /*is_causal=*/false,
        /*position_ids=*/std::nullopt, /*deepstack=*/std::nullopt,
        /*rope_cos_il=*/std::nullopt, /*rope_sin_il=*/std::nullopt,
        /*cos_sin_offset=*/-1, /*batch_slot=*/0,
        /*allow_batch_decode=*/false, /*planned_chunk_size=*/0,
        planned_stage_descriptor);
}

}  // namespace v3

// =============================================================================
// Instance registry + C API (decls in rpu_kernel_decls.h)
// =============================================================================
using HyVlaExpertRegistry = ModelHandleRegistry<v3::HyVlaExpertModel>;

std::vector<int64_t> rpu_hyvla_expert_planner_cache_identity(int64_t handle) {
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_hyvla_expert_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_hyvla_expert_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_hyvla_expert_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_kvinsert_cost_domain")
        ->kvinsert_cost_domain("hyvla_expert", descriptor);
}

std::string rpu_hyvla_expert_kvinsert_cost_catalog_sha256(int64_t handle) {
    return HyVlaExpertRegistry::get(
        handle, "rpu_hyvla_expert_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

int64_t rpu_hyvla_expert_create() {
    return HyVlaExpertRegistry::create();
}

void rpu_hyvla_expert_set_runtime_config(
    int64_t handle,
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool silu_mul, bool kvpad16, bool partial_rope,
    bool rmsnorm_pad16) {
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_runtime_config")
        ->set_runtime_config(
            fast_replay, fast_replay_preload, mask_once, silu_mul,
            kvpad16, partial_rope, rmsnorm_pad16);
}

void rpu_hyvla_expert_destroy(int64_t handle) {
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_destroy")
        ->check_execution_reconfigure_destroy_allowed(
            "rpu_hyvla_expert_destroy");
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
    int64_t chunk_size, int64_t action_mlp_cores)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_weights")
        ->set_weights_expert(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, chunk_size, action_mlp_cores);
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
    int64_t chunk_size, int64_t action_mlp_cores,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_set_weights_w8a16")
        ->set_weights_expert(
            q_w, k_w, v_w, o_w, q_norm, k_norm, input_norm, post_norm,
            gate_w, up_w, down_w, cos, sin, final_norm_w,
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size,
            eps, chunk_size, action_mlp_cores,
            q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws);
}

at::Tensor rpu_hyvla_expert_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor)
{
    return HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_step_forward")
        ->step_forward(x_emb, k_caches, v_caches, cos, sin, attn_mask_4d,
                       prefix_len, planned_stage_descriptor);
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
    double dt, int64_t prefix_len, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor)
{
    HyVlaExpertRegistry::get(handle, "rpu_hyvla_expert_unroll_forward")
        ->unroll_forward(x0_rpu, k_caches, v_caches, state_emb, cos, sin,
                         attn_mask_4d, x_traj, dt, prefix_len, num_steps,
                         planned_stage_descriptor);
}

std::vector<int64_t> rpu_hyvla_expert_resolve_stage_domain(
    int64_t handle, int64_t seq_len, int64_t prefix_len,
    int64_t mask_kv_len) {
    return HyVlaExpertRegistry::get(
               handle, "rpu_hyvla_expert_resolve_stage_domain")
        ->resolve_stage_domain(seq_len, prefix_len, mask_kv_len);
}

int64_t rpu_hyvla_expert_get_resolved_chunk_size(int64_t handle) {
    return HyVlaExpertRegistry::get(
               handle, "rpu_hyvla_expert_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

void rpu_hyvla_expert_set_chunk_size_override(
    int64_t handle, int64_t chunk_size) {
    HyVlaExpertRegistry::get(
        handle, "rpu_hyvla_expert_set_chunk_size_override")
        ->set_control_chunk_size_override(
            chunk_size, "rpu_hyvla_expert_set_chunk_size_override");
}

void rpu_hyvla_expert_enable_execution_reconfigure(int64_t handle) {
    HyVlaExpertRegistry::get(
        handle, "rpu_hyvla_expert_enable_execution_reconfigure")
        ->enable_execution_reconfigure_guard();
}

void rpu_hyvla_expert_stage_chunk_size_override(
    int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0,
                "Hy-VLA action expert hot-reconfigure token must be positive");
    HyVlaExpertRegistry::get(
        handle, "rpu_hyvla_expert_stage_chunk_size_override")
        ->stage_control_chunk_size_override(
            static_cast<uint64_t>(token), chunk_size,
            "rpu_hyvla_expert_stage_chunk_size_override");
}
