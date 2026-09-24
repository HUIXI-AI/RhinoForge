// rpu_hyvla_vlm_model.cpp — Hy-Embodied-0.5-VLA VLM prefill decoder.
// Each layer computes text and vision branches and merges them using constant
// per-row modality masks. set_moe_weights expands the masks into persistent
// preload storage for the fixed prefill layout.
// RoPE precedes QK normalization, whose weights are shared with the expert.
// The model has no QKV bias and uses one final norm.
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

// FAST_REPLAY and FAST_REPLAY_PRELOAD use the same stable-buffer and mutable-DMA
// contracts as the HYViT2 and expert components. Clean preload nodes remain
// part of the captured segment even when their host emission is skipped.
// MASK_ONCE extends the mask lifetime using the expert component's allocator
// rules. Python resolves these choices once and binds them to the handle.
// PARTIAL_ROPE preserves the existing tables and table-row offset.
//
// MOT_NORM_NOMERGE omits merging the input and post-attention norm outputs before
// the two independent GEMMs. Each branch reads its own norm output; the later
// Q/K/V or down merge still selects the required branch per row. This requires
// binary masks and finite discarded branch values, since 0 * NaN remains NaN.
// norm_a1 must live through phase 2 and norm_a2 through phase 7. The qkv/mlp
// selectors control these lifetime extensions independently; both enables both.
// The allocator must account for the longer lifetimes before admitting a layout.
enum : unsigned { MOT_NOMERGE_QKV = 1u, MOT_NOMERGE_MLP = 2u };

// ─────────────────────────────────────────────────────────────────────────────
// RPU_HY_VLA_SILU_MUL —— 逐座 opt-in，默认 OFF。机理与验收口径见
// `rpu_hyvla_expert_model.cpp` 里同名函数的注释（三座各一份）。
// 本座是 MoT 双分支 ⇒ 每层两次 SwiGLU，合并后每层省 2 次 launch。
// ─────────────────────────────────────────────────────────────────────────────
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

// HYVLA_VLM_FIXED_KERNEL_BASIS: launchers outside the selector-controlled
// sites below are invariant model-dataflow glue.  Every cold route that
// changes graph scheduling, normalization topology, activation, RoPE, or DMA
// cadence is bound to an explicit physical-manifest entry before dispatch.

constexpr int64_t HYVLA_VLM_ATTN_SITE = 4354974985578103569LL;
constexpr int64_t HYVLA_VLM_KV_SITE = 2260457013508596554LL;
constexpr int64_t HYVLA_VLM_GRAPH_SCHEDULE_SITE = 1891892172693751815LL;
constexpr int64_t HYVLA_VLM_MASK_SCHEDULE_SITE = 8543552822435768982LL;
constexpr int64_t HYVLA_VLM_INPUT_NORM_TOPOLOGY_SITE = 7126209149859475005LL;
constexpr int64_t HYVLA_VLM_POST_NORM_TOPOLOGY_SITE = 7589253846984036685LL;
constexpr int64_t kHyVlaVlmRmsNormFullRowsSite = 561884935551131924LL;
constexpr int64_t kHyVlaVlmRmsNormQHeadRowsSite = 8544308333460224633LL;
constexpr int64_t kHyVlaVlmRmsNormKHeadRowsSite = 1883301826947881937LL;
constexpr int64_t HYVLA_VLM_QKV_LINEAR_SITE = 1055061191250944583LL;
constexpr int64_t HYVLA_VLM_Q_PARTIAL_ROPE_SITE = 7310872600321580915LL;
constexpr int64_t HYVLA_VLM_Q_ROPE_SITE = 5787941996819323604LL;
constexpr int64_t HYVLA_VLM_K_PARTIAL_ROPE_SITE = 8480165489785921729LL;
constexpr int64_t HYVLA_VLM_K_ROPE_SITE = 7482633060228256420LL;
constexpr int64_t HYVLA_VLM_PREPARE_ATTN_ALL_REDUCE_SITE = 7244482564358429924LL;
constexpr int64_t HYVLA_VLM_O_TEXT_LINEAR_SITE = 1140698111824322874LL;
constexpr int64_t HYVLA_VLM_O_ACT_LINEAR_SITE = 3216096526013279443LL;
constexpr int64_t HYVLA_VLM_ATTN_ALL_REDUCE_SITE = 2887898270408281163LL;
constexpr int64_t HYVLA_VLM_MLP_GATE_LINEAR_SITE = 565715283461372786LL;
constexpr int64_t HYVLA_VLM_MLP_SILU_SITE = 3082555866678770980LL;
constexpr int64_t HYVLA_VLM_MLP_UP_LINEAR_SITE = 6842600829612506442LL;
constexpr int64_t HYVLA_VLM_MLP_SILU_MUL_SITE = 3430121266977245422LL;
constexpr int64_t HYVLA_VLM_MLP_MUL_SITE = 4141421508680707465LL;
constexpr int64_t HYVLA_VLM_MLP_DOWN_LINEAR_SITE = 1914100820284426885LL;
constexpr int64_t HYVLA_VLM_MLP_ALL_REDUCE_SITE = 740273045227131643LL;
constexpr int64_t HYVLA_VLM_QKV_BIAS_PRELOAD_DMA_SITE =
    7757347464819546080LL;
constexpr uint32_t HYVLA_VLM_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
    KV_INSERT_CAP_NON8_TP_V16;
constexpr int64_t HYVLA_VLM_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED = 2;

enum class HyVlaVlmGraphScheduleRoute : int64_t {
    REEMIT_LAYER_LOOP = 1,
    FAST_REPLAY_SKIP_LAYER_LOOP = 2,
    MASK_EACH_LAYER = 3,
    MASK_FIRST_LAYER_ONLY = 4,
};

enum class HyVlaVlmNormalizationRoute : int64_t {
    MOT_MERGED = 1,
    MOT_BRANCH_LOCAL = 2,
};

enum class HyVlaVlmActivationRoute : int64_t {
    SILU_UNARY = 1,
    MUL = 2,
    SILU_MUL_FUSED = 3,
};

enum class HyVlaVlmRopeRoute : int64_t {
    ROPE_1D = 1,
    PARTIAL_MROPE = 2,
};

enum class HyVlaVlmAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

enum class HyVlaVlmDmaRoute : int64_t {
    DDR_SCATTER_TO_SPM_FIXED = 1,
};

ModelStaticConfig HyVlaVlmModel::static_config() {
    ModelStaticConfig cfg = CausalDecoderModel::static_config();
    cfg.fast_replay_skip_layer_loop = cold_fast_replay_;
    cfg.fast_replay_skip_preload = cold_fast_replay_preload_;
    return cfg;
}

HyVlaVlmModel::HyVlaVlmModel() = default;
HyVlaVlmModel::~HyVlaVlmModel() = default;

void HyVlaVlmModel::set_runtime_config(
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool partial_rope, int64_t mot_norm_nomerge, bool silu_mul,
    int64_t rmsnorm_capability) {
    TORCH_CHECK(
        !cold_config_bound_ && text_layers_.empty(),
        "hyvla_vlm_set_runtime_config must run exactly once before "
        "set_weights");
    TORCH_CHECK(
        mot_norm_nomerge >= 0 &&
            mot_norm_nomerge <=
                static_cast<int64_t>(MOT_NOMERGE_QKV | MOT_NOMERGE_MLP),
        "hyvla_vlm_set_runtime_config: mot_norm_nomerge must be a "
        "two-bit mask in [0,3], got ", mot_norm_nomerge);
    TORCH_CHECK(
        rmsnorm_capability >= 0 &&
            rmsnorm_capability <= static_cast<int64_t>(RPU_RMSNORM_CAP_ALL) &&
            rpu_rmsnorm_capability_valid(
                static_cast<uint32_t>(rmsnorm_capability)),
        "hyvla_vlm_set_runtime_config: rmsnorm_capability must include BASE "
        "and contain only BASE/V16/V32; got ", rmsnorm_capability);

    // Snapshot every fallible runtime dependency before committing this
    // one-shot configuration.  In particular, KernelCache::has_loaded() may
    // reject a call outside the runtime-ready lifecycle; such a rejection must
    // leave the handle eligible for a clean retry with no partially installed
    // cold fields.
    const RpuRmsNormSpmContract rmsnorm_spm_contract =
        rpu_snapshot_rmsnorm_spm_contract(
            static_cast<uint32_t>(rmsnorm_capability));
    invalidate_model_state();

    cold_fast_replay_ = fast_replay;
    cold_fast_replay_preload_ = fast_replay_preload;
    cold_mask_once_ = mask_once;
    cold_partial_rope_ = partial_rope;
    cold_mot_norm_nomerge_ = static_cast<unsigned>(mot_norm_nomerge);
    cold_silu_mul_ = silu_mul;
    cold_rmsnorm_spm_contract_ = rmsnorm_spm_contract;
    cold_config_bound_ = true;
}

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
    TORCH_CHECK(
        cold_config_bound_,
        "hyvla set_weights requires an explicit immutable runtime config "
        "snapshot");
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
    // layer_weights_/cos_/sin_/eps_ 都是 private。W4 scale 必须复用相同的
    // 4096B retain-alignment contract；其他 tensor 只复制句柄。
    const int64_t N = static_cast<int64_t>(q_w_list.size());
    text_layers_.clear();
    text_layers_.reserve(N);
    const bool bias = (q_bias_list.size() > 0);
    const bool quantized = !q_ws_list.empty();
    for (int64_t i = 0; i < N; ++i) {
        text_layers_.push_back({
            q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
            q_norm_list[i], k_norm_list[i],
            input_norm_list[i], post_norm_list[i],
            gate_list[i], up_list[i], down_list[i],
            bias ? q_bias_list[i] : at::Tensor(),
            bias ? k_bias_list[i] : at::Tensor(),
            bias ? v_bias_list[i] : at::Tensor(),
            quantized ? rpu_retain_linear_quant_scale(q_w_list[i], q_ws_list[i]) : at::Tensor(),
            quantized ? rpu_retain_linear_quant_scale(k_w_list[i], k_ws_list[i]) : at::Tensor(),
            quantized ? rpu_retain_linear_quant_scale(v_w_list[i], v_ws_list[i]) : at::Tensor(),
            quantized ? rpu_retain_linear_quant_scale(o_w_list[i], o_ws_list[i]) : at::Tensor(),
            quantized ? rpu_retain_linear_quant_scale(gate_list[i], gate_ws_list[i]) : at::Tensor(),
            quantized ? rpu_retain_linear_quant_scale(up_list[i], up_ws_list[i]) : at::Tensor(),
            quantized ? rpu_retain_linear_quant_scale(down_list[i], down_ws_list[i]) : at::Tensor(),
        });
    }
    cos_hyvla_ = cos;
    sin_hyvla_ = sin;
    eps_hyvla_ = eps;
    final_norm_text_ = final_norm_w;   // set_moe_weights 的同一性断言要用

    invalidate_model_state();   // D-503: 影子提交后再失效一次, 保证末语句语义
}

// ─────────────────────────────────────────────────────────────────────────────
// dynamic_config — 基类断言单 chunk，并统一准备显式 mask。HyVLA 直接复用
// CausalDecoderModel 的稳定 PreparedMask，避免同一输入在一个 forward 内重复
// RPU→CPU→RPU 归一化。
// ─────────────────────────────────────────────────────────────────────────────
ModelDynamicConfig HyVlaVlmModel::dynamic_config(const ChunkPlan& plan) {
    ModelDynamicConfig cfg = CausalDecoderModel::dynamic_config(plan);
    // The action expert consumes this prefix history from the shared DDR KV
    // cache.  This owner does not advertise the unrelated raw-SPM/MHA path.
    cfg.attention_policy = AttentionExecutionPolicy::DDR_KV;
    TORCH_CHECK(ctx().attention_mask.has_value(),
                "hyvla dynamic_config: explicit attention mask required");
    return cfg;
}

ModelDynamicConfig HyVlaVlmModel::planning_dynamic_config(
    const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    cfg.attention_policy = AttentionExecutionPolicy::DDR_KV;
    return cfg;
}

FmbPhysicalExecutionManifest HyVlaVlmModel::physical_manifest_for_candidate(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& /*layout*/,
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
        "HyVlaVlmModel COMPLETE descriptor requires one prefix chunk");
    const ChunkInfo& kv_chunk = plan.qkv.chunks.front();
    const ChunkInfo& chunk = plan.compute.chunks.front();
    TORCH_CHECK(
        kv_chunk.idx == chunk.idx && kv_chunk.offset == chunk.offset &&
            kv_chunk.len == chunk.len,
        "HyVlaVlmModel COMPLETE descriptor requires identical QKV and "
        "compute prefix chunks");
    const int64_t invocation = chunk.idx;
    const int64_t h = hidden_size();
    const int64_t nq = num_q_heads();
    const int64_t nkv = num_kv_heads();
    const int64_t hd = head_dim();
    const int64_t tp = attn_tp();
    const int64_t local_q = nq / tp;
    const int64_t local_kv_heads = nkv / tp;
    manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
    const KvInsertSegmentPlan kv_plan =
        resolve_kvinsert_plan_auto(
            HYVLA_VLM_KV_SITE, manifest.graph_lifecycle,
            position + kv_chunk.offset, kv_chunk.len, kv_chunk.len,
            attn_tp(), num_kv_heads(), head_dim(),
            HYVLA_VLM_KV_CAPABILITIES);
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

    append(
        FmbRouteFamily::GRAPH_SCHEDULE,
        HYVLA_VLM_GRAPH_SCHEDULE_SITE,
        static_cast<int64_t>(
            cold_fast_replay_
                ? HyVlaVlmGraphScheduleRoute::FAST_REPLAY_SKIP_LAYER_LOOP
                : HyVlaVlmGraphScheduleRoute::REEMIT_LAYER_LOOP),
        {cold_fast_replay_ ? 1 : 0,
         cold_fast_replay_preload_ ? 1 : 0});
    append(
        FmbRouteFamily::GRAPH_SCHEDULE,
        HYVLA_VLM_MASK_SCHEDULE_SITE,
        static_cast<int64_t>(
            cold_mask_once_
                ? HyVlaVlmGraphScheduleRoute::MASK_FIRST_LAYER_ONLY
                : HyVlaVlmGraphScheduleRoute::MASK_EACH_LAYER),
        {cold_mask_once_ ? 1 : 0, chunk.len, kv_chunk.kv_seq_len,
         tp, num_layers()});
    append(
        FmbRouteFamily::NORMALIZATION,
        HYVLA_VLM_INPUT_NORM_TOPOLOGY_SITE,
        static_cast<int64_t>(
            (cold_mot_norm_nomerge_ & MOT_NOMERGE_QKV)
                ? HyVlaVlmNormalizationRoute::MOT_BRANCH_LOCAL
                : HyVlaVlmNormalizationRoute::MOT_MERGED),
        {static_cast<int64_t>(cold_mot_norm_nomerge_),
         (cold_mot_norm_nomerge_ & MOT_NOMERGE_QKV) ? 1 : 0,
         chunk.len, h});
    append(
        FmbRouteFamily::NORMALIZATION,
        HYVLA_VLM_POST_NORM_TOPOLOGY_SITE,
        static_cast<int64_t>(
            (cold_mot_norm_nomerge_ & MOT_NOMERGE_MLP)
                ? HyVlaVlmNormalizationRoute::MOT_BRANCH_LOCAL
                : HyVlaVlmNormalizationRoute::MOT_MERGED),
        {static_cast<int64_t>(cold_mot_norm_nomerge_),
         (cold_mot_norm_nomerge_ & MOT_NOMERGE_MLP) ? 1 : 0,
         chunk.len, h});
    const auto append_rmsnorm = [&](int64_t site_id, int64_t rows,
                                    int64_t cols) {
        append(
            FmbRouteFamily::NORMALIZATION, site_id,
            static_cast<int64_t>(cold_rmsnorm_spm_contract_.resolve(rows, cols)),
            cold_rmsnorm_spm_contract_.manifest_arguments(rows, cols));
    };
    append_rmsnorm(kHyVlaVlmRmsNormFullRowsSite, chunk.len, h);
    append_rmsnorm(
        kHyVlaVlmRmsNormQHeadRowsSite, chunk.len * local_q, hd);
    append_rmsnorm(
        kHyVlaVlmRmsNormKHeadRowsSite,
        chunk.len * local_kv_heads, hd);
    if (has_qkv_bias_) {
        const int64_t q_bias_elems = nq * hd;
        const int64_t kv_bias_elems = nkv * hd;
        const auto append_bias_dma = [&](int64_t total_elems,
                                         int64_t bias_invocation) {
            const int64_t per_core = total_elems / tp;
            manifest.routes.push_back({
                HYVLA_VLM_QKV_BIAS_PRELOAD_DMA_SITE,
                FmbRouteFamily::MUTABLE_DMA,
                static_cast<int64_t>(
                    HyVlaVlmDmaRoute::DDR_SCATTER_TO_SPM_FIXED),
                /*flags=*/0,
                {has_qkv_bias_ ? 1 : 0, total_elems, per_core,
                 per_core * DWIDTH, tp},
                bias_invocation});
        };
        append_bias_dma(q_bias_elems, /*bias_invocation=*/0);
        append_bias_dma(kv_bias_elems, /*bias_invocation=*/1);
        append_bias_dma(kv_bias_elems, /*bias_invocation=*/2);
    }
    append_linear(HYVLA_VLM_QKV_LINEAR_SITE);
    append(
        FmbRouteFamily::ROPE,
        cold_partial_rope_ ? HYVLA_VLM_Q_PARTIAL_ROPE_SITE
                           : HYVLA_VLM_Q_ROPE_SITE,
        static_cast<int64_t>(
            cold_partial_rope_ ? HyVlaVlmRopeRoute::PARTIAL_MROPE
                               : HyVlaVlmRopeRoute::ROPE_1D),
        {cold_partial_rope_ ? 1 : 0, position + chunk.offset,
         chunk.len, local_q, hd, tp});
    append(
        FmbRouteFamily::ROPE,
        cold_partial_rope_ ? HYVLA_VLM_K_PARTIAL_ROPE_SITE
                           : HYVLA_VLM_K_ROPE_SITE,
        static_cast<int64_t>(
            cold_partial_rope_ ? HyVlaVlmRopeRoute::PARTIAL_MROPE
                               : HyVlaVlmRopeRoute::ROPE_1D),
        {cold_partial_rope_ ? 1 : 0, position + chunk.offset,
         chunk.len, local_kv_heads, hd, tp});
    // Prefix history has no raw-SPM ownership port in this model.  Freeze the
    // exact pure-resolver segment plan into the descriptor; forward restores
    // this typed value and never consults the legacy environment selectors.
    append(
        FmbRouteFamily::KV_INSERT, HYVLA_VLM_KV_SITE,
        static_cast<int64_t>(kv_plan.route()),
        std::vector<int64_t>(kv_arguments.begin(), kv_arguments.end()),
        HYVLA_VLM_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED);
    append(
        FmbRouteFamily::ATTENTION, HYVLA_VLM_ATTN_SITE,
        static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV));
    append(
        FmbRouteFamily::ALL_REDUCE,
        HYVLA_VLM_PREPARE_ATTN_ALL_REDUCE_SITE,
        static_cast<int64_t>(HyVlaVlmAllReduceRoute::PREPARE_RING_INPUT),
        {chunk.len, h, tp});
    append_linear(
        HYVLA_VLM_O_TEXT_LINEAR_SITE,
        {chunk.len, h, nq * hd, 0, tp});
    append_linear(
        HYVLA_VLM_O_ACT_LINEAR_SITE,
        {chunk.len, h, nq * hd, 0, tp});
    append(
        FmbRouteFamily::ALL_REDUCE,
        HYVLA_VLM_ATTN_ALL_REDUCE_SITE,
        fmb_ring_all_reduce_route_selector(chunk.len, h),
        {chunk.len, h, tp, NUM_CORES});
    append_linear(
        HYVLA_VLM_MLP_GATE_LINEAR_SITE,
        {chunk.len, intermediate_size(), h, 1, NUM_CORES});
    append_linear(
        HYVLA_VLM_MLP_UP_LINEAR_SITE,
        {chunk.len, intermediate_size(), h, 1, NUM_CORES});
    const int64_t elems_mlp = chunk.len * (intermediate_size() / NUM_CORES);
    if (cold_silu_mul_) {
        append(
            FmbRouteFamily::ACTIVATION, HYVLA_VLM_MLP_SILU_MUL_SITE,
            static_cast<int64_t>(HyVlaVlmActivationRoute::SILU_MUL_FUSED),
            {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES});
    } else {
        append(
            FmbRouteFamily::ACTIVATION, HYVLA_VLM_MLP_SILU_SITE,
            static_cast<int64_t>(HyVlaVlmActivationRoute::SILU_UNARY),
            {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES});
        append(
            FmbRouteFamily::ACTIVATION, HYVLA_VLM_MLP_MUL_SITE,
            static_cast<int64_t>(HyVlaVlmActivationRoute::MUL),
            {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES});
    }
    append_linear(
        HYVLA_VLM_MLP_DOWN_LINEAR_SITE,
        {chunk.len, h, intermediate_size(), 0, NUM_CORES});
    append(
        FmbRouteFamily::ALL_REDUCE,
        HYVLA_VLM_MLP_ALL_REDUCE_SITE,
        fmb_ring_all_reduce_route_selector(chunk.len, h),
        {chunk.len, h, NUM_CORES, NUM_CORES});
    append_causal_decoder_preload_manifest_routes(manifest);
    append_fmb_shared_runtime_routes(
        manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA);
    return manifest;
}

FmbPhysicalManifestForwardCapability
HyVlaVlmModel::physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& /*manifest*/) const {
    return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
}

// ─────────────────────────────────────────────────────────────────────────────
// subclass_chunk_size_valid — prefill 强制单块语义
//
// 整个 prefix 必须一次进 MASK_2D 非因果 SDPA。Hy-VLA seq=192 > 176 ⇒
// tile_m_v16 封顶 8 → tile_m=128 → grid_dim_x=2；16/4 GQA 在 attn_tp=4 下
// gqa_group_size=4 ⇒ product=8，**正好压在** keeper 的 product<=8 上限
// (rpu_helpers.h:322)。这里 override 是因为行掩码与 KV merge 按整块设计：
// 单块 (cs>=seq_len ⇒ num_chunks==1) 合法，多块非法。
// ⚠️ 更长的 prefix 会把 grid 顶过上限 —— 届时要改成分块 + 掩码按 chunk.offset
// 取行切片，而不是放宽单块约束。
// ─────────────────────────────────────────────────────────────────────────────
bool HyVlaVlmModel::subclass_chunk_size_valid(
    int64_t cs, int64_t seq_len, int64_t /*position*/) const {
    return cs >= seq_len && chunk_within_envelope(cs);
}

std::vector<int64_t> HyVlaVlmModel::resolve_stage_domain(
    int64_t seq_len, int64_t prefix_len, int64_t mask_kv_len) {
    TORCH_CHECK(seq_len == moe_chunk_size_,
                "hyvla_vlm_resolve_stage_domain: seq_len must match the "
                "fixed MoT row-mask geometry");
    TORCH_CHECK(prefix_len >= 0 && mask_kv_len == prefix_len + seq_len,
                "hyvla_vlm_resolve_stage_domain: mask width must equal "
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
    TORCH_CHECK(chunk_size >= 16 && chunk_size <= 240 && chunk_size % 16 == 0,
                "hyvla set_moe_weights: chunk_size must be a multiple of 16 "
                "in the certified [16,240] prefix envelope, got ", chunk_size);

    // Fail the cold-only check before retaining any MoE weights or masks.
    set_chunk_envelope(/*max_kv_len=*/chunk_size, /*chunk=*/chunk_size);

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
    // 原仅 check_rank(defined+dim)是 CLASS-B 并行路径漂移:孪生 setter 漏抄 image_flow
    // set_gen_weights 的 check_fp16_rpu 校验。形状细节仍由 GEMM 启动器按 N/K 校验。
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
    const bool moe_quantized = (q_ws_list.size() > 0);
    auto check_scale_list = [&](const at::TensorList& lst, const char* nm) {
        TORCH_CHECK((lst.size() > 0) == moe_quantized,
                    "hyvla set_moe_weights: ", nm, " must be empty unless ALL "
                    "seven quantized scale lists are provided");
        if (moe_quantized) check_list(lst, nm);
    };
    check_scale_list(k_ws_list,    "k_ws_list");
    check_scale_list(v_ws_list,    "v_ws_list");
    check_scale_list(o_ws_list,    "o_ws_list");
    check_scale_list(gate_ws_list, "gate_ws_list");
    check_scale_list(up_ws_list,   "up_ws_list");
    check_scale_list(down_ws_list, "down_ws_list");
    if (moe_quantized) check_list(q_ws_list, "q_ws_list");
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
            TORCH_CHECK(moe_quantized ? (w[i].scalar_type() == at::kChar
                                  || w[i].scalar_type() == at::kByte)
                               : (w[i].scalar_type() == at::kHalf),
                        "hyvla set_moe_weights: ", name, "[", i, "] must be ",
                        (moe_quantized ? "int8 or packed-uint8 (quant scales given)" : "fp16"),
                        ", got ", w[i].scalar_type());
            if (moe_quantized) {
                TORCH_CHECK(sc[i].defined() && sc[i].scalar_type() == at::kHalf
                            && sc[i].device().type() == at::kPrivateUse1
                            && sc[i].is_contiguous(),
                            "hyvla set_moe_weights: ", name, "_scale[", i,
                            "] must be a contiguous fp16 RPU tensor");
                if (w[i].scalar_type() == at::kChar) {
                    TORCH_CHECK(sc[i].dim() == 1 &&
                                    sc[i].numel() == w[i].size(0),
                                "hyvla set_moe_weights: W8A16 ", name,
                                "_scale[", i, "] must be [N=", w[i].size(0), "]");
                } else {
                    TORCH_CHECK(sc[i].dim() == 2 &&
                                    (sc[i].size(0) == 32 || sc[i].size(0) == 64 ||
                                     sc[i].size(0) == 128),
                                "hyvla set_moe_weights: W4A16 ", name,
                                "_scale[", i,
                                "] must be a controller-striped pgrp 2D payload");
                }
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
    // 掩码逐行恒定, 只有 cs 个不同的值。历史版本按 3 个 merge 宽度各展开成
    // [cs, W] —— S=240 时 2×(240+60+960) KB = **2520 KB** 的 Persistent, 存的却是
    // 240 bit 的信息, 且 Persistent 走 alloc_super_persistent (单调不回收) ⇒ 它是
    // 三子系统跨帧共存的头号阻塞。改成行向量后两个槽合计 1 KB。
    // merge 侧用 `rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel` 做行广播乘,
    // **位精确**: 掩码恒 0/1, fp16 下 x*1.0 / x*0.0 / x+0.0 都是精确的。
    // ⚠️ 不要为了省掉一个掩码而改写成 `b + m*(a-b)` —— 那个不是位精确的。
    // 展开在 host (CPU) 完成后整体复制到 RPU DDR — 不在 device 上做常量填充 (Pitfall B-1)。
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
            moe_quantized ? rpu_retain_linear_quant_scale(q_w_list[i], q_ws_list[i]) : at::Tensor(),
            moe_quantized ? rpu_retain_linear_quant_scale(k_w_list[i], k_ws_list[i]) : at::Tensor(),
            moe_quantized ? rpu_retain_linear_quant_scale(v_w_list[i], v_ws_list[i]) : at::Tensor(),
            moe_quantized ? rpu_retain_linear_quant_scale(o_w_list[i], o_ws_list[i]) : at::Tensor(),
            moe_quantized ? rpu_retain_linear_quant_scale(gate_list[i], gate_ws_list[i]) : at::Tensor(),
            moe_quantized ? rpu_retain_linear_quant_scale(up_list[i], up_ws_list[i]) : at::Tensor(),
            moe_quantized ? rpu_retain_linear_quant_scale(down_list[i], down_ws_list[i]) : at::Tensor(),
        });
    }
    text_row_mask_    = text_row_mask;
    moe_chunk_size_   = chunk_size;

    invalidate_model_state();   // D-503: set_moe_weights 末非空语句
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers — 基类布局 + vision 孪生 Temp 槽 + 掩码/孪生权重 preload 槽
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BufferDecl> HyVlaVlmModel::declare_buffers(const LayoutContext& lctx) {
    auto d = CausalDecoderModel::declare_buffers(lctx);
    if (cold_mask_once_) hyvla_widen_sdpa_mask_slot(d);
    TORCH_CHECK(moe_chunk_size_ > 0,
                "hyvla declare_buffers: set_moe_weights must precede forward");
    // 框架解析出的 chunk_size 是单块 SPM 规划宽度 (ceil16(seq)), 实际行数
    // = chunk.len = seq_len = moe_chunk_size_。基类/孪生双算槽按 lctx.chunk_size
    // 分配 (容纳); 行掩码槽按 moe_chunk_size_(=cs_mask) 分配, 与 set_moe_weights
    // 展开的 [cs_mask,w] 常量张量及 build_layer_subgraph 的 seq_len 行消费一致。
    // The generic planner first calls declare_buffers with chunk_size=16 to
    // measure fixed overhead, even when an exact larger override is installed.
    // That dry probe is not an admitted execution shape. Single-chunk admission
    // is enforced by subclass_chunk_size_valid, and build_layer_subgraph checks
    // the actual chunk length before emitting any work.

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

    // act 分支 Temp 槽 — phase 区间镜像基类对应槽 (rpu_qwen3_model.h:785-807),
    // 使 SPM 别名窗口一致: q_a 2..5, k_a/v_a 2..4, oproj_a 5..5, gate_a 7..8,
    // up_a 7..7, down_a 7..8。norm_a 是 act 侧 norm 暂存 (input/headnorm/post/
    // final 各相复用), 生命周期与 input_norm 同 (1..8)。
    d.push_back({"q_a",     q,   2, 5, StorageClass::Temp, 0, nullptr});
    d.push_back({"k_a",     kv,  2, 4, StorageClass::Temp, 0, nullptr});
    d.push_back({"v_a",     kv,  2, 4, StorageClass::Temp, 0, nullptr});
    d.push_back({"oproj_a", res, 5, 5, StorageClass::Temp, 0, nullptr});
    // (HALO 在此还有 gate_a / up_a 两个 mlp 槽 —— **不需要**: 两条 MLP 是串行
    //  发射的, text 分支跑完时它的 gate/up 中间量已经死透 (结果落在 down),
    //  而 gate/up 又不参与行掩码 merge —— 只有最终的 down 参与。所以孪生分支
    //  直接复写基类的 "gate"/"up" 槽即可, 唯一必须独立的是 down_a。HALO 的
    //  cs=18 下这两个槽只值 ~55 KB 没人注意, Hy-VLA 的 S=240 下是 0.72 MB,
    //  且峰值恰好落在 phase 7-8 的 MLP 相。)
    d.push_back({"down_a",  res, 7, 8, StorageClass::Temp, 0, nullptr});
    // norm_a1/norm_a2 — 孪生 norm 的暂存。HALO 用**单个** norm_a 且 phase 1..8,
    // 因为它的 final norm 也要双算; Hy-VLA 的 final norm 只有一份 (步 7 已改单算),
    // 于是孪生 norm 只剩两个瞬时点: 步 1 (input norm) 与步 5 (post norm)。拆成两个
    // 一格区间后, 二者都能与 q/k/v/output/oproj/gate/up/down 复用地址 —— 一个横跨
    // 1..8 的 res 槽 (S=240 时 0.94 MB) 是 SPM 峰值里最贵且最没必要的一项。
    // R43：`MOT_NORM_NOMERGE` 打开对应半边时，act 侧 norm 暂存要活到下游 GEMM
    // 那一相（QKV 是 phase 2、MLP 是 phase 7）⇒ 相应把窗口拉长一格。
    // 开关关掉时窗口与改动前**逐字节相同**，保证 A/B 两臂比的是同一个布局问题。
    const unsigned nomerge_mask = cold_mot_norm_nomerge_;
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
        // 与基类 q/k_bias 槽的布局一致 (rpu_qwen3_model.h:917-941)。
        auto dma_safe = [&A](int64_t elems) -> int64_t {
            return A(((elems + 255) / 256) * 256 * DWIDTH);
        };
        auto add_bias = [&](const char* name, int64_t sz,
                            at::Tensor MoeLayerWeights::* member,
                            int64_t bias_invocation) {
            BufferDecl b;
            b.name      = name;
            b.size      = sz;
            b.storage   = StorageClass::PersistentPerLayer;
            b.per_layer = nl;
            b.scope     = BufferScope::LayerWide;
            b.preload_callback =
                [this, member, bias_invocation](
                    FusedModelBase&, int L, uint32_t core0_addr) {
                    const int tp_ = attn_tp();
                    const at::Tensor& bw = moe_layers_[L].*member;
                    const int64_t per_core = bw.numel() / tp_;
                    if (ctx().has_complete_physical_manifest()) {
                        ctx().consume_physical_route(
                            FmbRouteFamily::MUTABLE_DMA,
                            HYVLA_VLM_QKV_BIAS_PRELOAD_DMA_SITE,
                            static_cast<int64_t>(
                                HyVlaVlmDmaRoute::DDR_SCATTER_TO_SPM_FIXED),
                            /*resolved_flags=*/0,
                            {has_qkv_bias_ ? 1 : 0, bw.numel(), per_core,
                             per_core * DWIDTH, tp_},
                            bias_invocation);
                    }
                    rpu_launch_ddr_scatter_spm_dma(
                        bw.data_ptr<c10::Half>(),
                        per_core, per_core * DWIDTH, core0_addr, tp_);
                };
            d.push_back(b);
        };
        add_bias("q_bias_act", dma_safe(local_q * hd),
                 &MoeLayerWeights::q_bias, /*bias_invocation=*/0);
        add_bias("k_bias_act", dma_safe(local_kv),
                 &MoeLayerWeights::k_bias, /*bias_invocation=*/1);
        add_bias("v_bias_act", dma_safe(local_kv),
                 &MoeLayerWeights::v_bias, /*bias_invocation=*/2);
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
    TORCH_CHECK(
        ctx().has_complete_physical_manifest(),
        "HyVlaVlmModel requires a COMPLETE physical descriptor");
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
    const auto consume_rmsnorm = [&](int64_t site_id, int64_t rows,
                                     int64_t cols) {
        const RpuRmsNormSpmRoute route =
            cold_rmsnorm_spm_contract_.resolve(rows, cols);
        TORCH_CHECK(
            !cold_rmsnorm_spm_contract_.has_vector_payload() ||
                ctx().has_complete_physical_manifest(),
            "HyVlaVlmModel non-BASE RMSNorm route requires a COMPLETE "
            "physical descriptor");
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::NORMALIZATION, site_id,
                static_cast<int64_t>(route), /*resolved_flags=*/0,
                cold_rmsnorm_spm_contract_.manifest_arguments(rows, cols),
                chunk.idx);
        }
        return route;
    };
    const RpuRmsNormSpmRoute full_rmsnorm_route = consume_rmsnorm(
        kHyVlaVlmRmsNormFullRowsSite, seq_len, h);
    const RpuRmsNormSpmRoute q_head_rmsnorm_route = consume_rmsnorm(
        kHyVlaVlmRmsNormQHeadRowsSite, seq_len * local_q, hd);
    const RpuRmsNormSpmRoute k_head_rmsnorm_route = consume_rmsnorm(
        kHyVlaVlmRmsNormKHeadRowsSite,
        seq_len * (nkv / tp), hd);

    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE,
            HYVLA_VLM_GRAPH_SCHEDULE_SITE,
            static_cast<int64_t>(
                cold_fast_replay_
                    ? HyVlaVlmGraphScheduleRoute::FAST_REPLAY_SKIP_LAYER_LOOP
                    : HyVlaVlmGraphScheduleRoute::REEMIT_LAYER_LOOP),
            /*resolved_flags=*/0,
            {cold_fast_replay_ ? 1 : 0,
             cold_fast_replay_preload_ ? 1 : 0},
            chunk.idx);
    }

    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }

    // ── 步 1: input RMSNorm 双算 (→ merge 入 "input_norm"，除非 NOMERGE) ──
    // text γ 经基类 "norm_w" 槽, act γ 经 "norm_w_act"。
    // 老路：merge 后 input_norm 持逐行选支的 normed hidden, 供两套 QKV 共读。
    // R43：两套 QKV 本来就是分别发射的 ⇒ **各读各的 norm 输出**，省掉这次 merge。
    // 逐位等价的逐行论证见 `hyvla_mot_norm_nomerge_mask` 的注释。
    const unsigned nomerge = cold_mot_norm_nomerge_;
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION,
            HYVLA_VLM_INPUT_NORM_TOPOLOGY_SITE,
            static_cast<int64_t>(
                (cold_mot_norm_nomerge_ & MOT_NOMERGE_QKV)
                    ? HyVlaVlmNormalizationRoute::MOT_BRANCH_LOCAL
                    : HyVlaVlmNormalizationRoute::MOT_MERGED),
            /*resolved_flags=*/0,
            {static_cast<int64_t>(cold_mot_norm_nomerge_),
             (cold_mot_norm_nomerge_ & MOT_NOMERGE_QKV) ? 1 : 0,
             seq_len, h}, chunk.idx);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "norm_w"), seq_len, h, eps_hyvla_,
        full_rmsnorm_route);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "norm_a1"),
        layer_addr(layer_idx, 0, "norm_w_act"), seq_len, h, eps_hyvla_,
        full_rmsnorm_route);
    if (!(nomerge & MOT_NOMERGE_QKV))
        emit_row_merge(addr(0, "input_norm"), addr(0, "norm_a1"), addr(0, "input_norm"),
                       m_txt, m_act, seq_len, cols_h, NUM_CORES);

    // ── 步 2: QKV 双发射 (text→q/k/v, act→q_a/k_a/v_a) + merge ──
    // bias 各用自家 PersistentPerLayer 槽。NOMERGE 时 text 读 "input_norm"、
    // act 读 "norm_a1"；否则两边都读 merge 后的 "input_norm"（老路）。
    auto qkv = [&](uint32_t in_addr, const at::Tensor& w, const at::Tensor& ws,
                   const char* out, const char* bias_slot, int64_t n) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVLA_VLM_QKV_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in_addr, w, addr(0, out),
            seq_len, n, h, /*partition=*/1, /*num_cores=*/tp,
            has_qkv_bias_ ? layer_addr(layer_idx, 0, bias_slot) : 0u, /*scale=*/ws);
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
    // 因 norm 共享，此处**不做双算/merge**（mw.q_norm_w / mw.k_norm_w 不参与计算）。
    c10::Half* cos_ptr = cos_hyvla_.data_ptr<c10::Half>();
    c10::Half* sin_ptr = sin_hyvla_.data_ptr<c10::Half>();

    // PARTIAL_ROPE（开关与依据见 rpu_hyvla_expert_model.cpp 文件头）：
    // 与 llama_rope 逐位相等，grid 从 (seq,1,1) 塌成
    // (⌈hd/2/64⌉, ⌈seq/64⌉)。本座 seq 更长（prefill 整段），塌缩比例比 expert 更大。
    const bool prope = cold_partial_rope_;
    const int64_t local_kv_heads = nkv / tp;

    if (prope) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE,
                HYVLA_VLM_Q_PARTIAL_ROPE_SITE,
                static_cast<int64_t>(HyVlaVlmRopeRoute::PARTIAL_MROPE),
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
                FmbRouteFamily::ROPE, HYVLA_VLM_Q_ROPE_SITE,
                static_cast<int64_t>(HyVlaVlmRopeRoute::ROPE_1D),
                /*resolved_flags=*/0,
                {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                 seq_len, local_q, hd, tp}, chunk.idx);
        }
        rpu_launch_rope_spm_kernel(addr(0, "q"), addr(0, "output"),
                                   cos_ptr, sin_ptr, seq_len, local_q, hd,
                                   cos_sin_start, tp);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "output"), addr(0, "q"),
        layer_addr(layer_idx, 0, "q_norm_w"), seq_len * local_q, hd,
        eps_hyvla_, q_head_rmsnorm_route);

    if (prope) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE,
                HYVLA_VLM_K_PARTIAL_ROPE_SITE,
                static_cast<int64_t>(HyVlaVlmRopeRoute::PARTIAL_MROPE),
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
                FmbRouteFamily::ROPE, HYVLA_VLM_K_ROPE_SITE,
                static_cast<int64_t>(HyVlaVlmRopeRoute::ROPE_1D),
                /*resolved_flags=*/0,
                {cold_partial_rope_ ? 1 : 0, cos_sin_start,
                 seq_len, local_kv_heads, hd, tp}, chunk.idx);
        }
        rpu_launch_rope_spm_kernel(addr(0, "k"), addr(0, "input_norm"),
                                   cos_ptr, sin_ptr, seq_len, local_kv_heads,
                                   hd, cos_sin_start, tp);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "input_norm"), addr(0, "k"),
        layer_addr(layer_idx, 0, "k_norm_w"), seq_len * local_kv_heads, hd,
        eps_hyvla_, k_head_rmsnorm_route);

    // ── 步 4: KV insert + SDPA 单次; O 双算 + merge → 残差归约 ──
    // K/V 已按行合并, insert/SDPA 与基类完全一致。O_proj 行分到 tp 核, 结果
    // 各持全宽 h 部分和; 两分支各自行分一致 → 部分和也逐行可掩码合并。
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    const FmbRouteManifestEntry& route = ctx().find_physical_route(
        FmbRouteFamily::KV_INSERT, HYVLA_VLM_KV_SITE, chunk.idx);
    const KvInsertSegmentPlan kv_plan =
        restore_kvinsert_plan(
            HYVLA_VLM_KV_SITE, route.arguments, tp, nkv, hd);
    TORCH_CHECK(
        kv_plan.logical_rows() == seq_len &&
            kv_plan.physical_rows() == seq_len &&
            kv_plan.segment(0).position == cos_sin_start,
        "HyVlaVlmModel KV descriptor geometry drift at invocation ",
        chunk.idx);
    ctx().consume_physical_route(
        FmbRouteFamily::KV_INSERT, HYVLA_VLM_KV_SITE,
        static_cast<int64_t>(kv_plan.route()),
        HYVLA_VLM_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED,
        route.arguments, chunk.idx);
    rpu_launch_insert_kvcache_spm_unified_with_plan(
        k_cache, v_cache,
        addr_offset("k").value, addr_offset("v").value,
        nkv, hd, tp,
        /*k_cache_batch_offset_elems=*/0,
        /*v_cache_batch_offset_elems=*/0,
        /*spm_rows=*/0, kv_plan);
    const uint32_t mask_off = addr_offset("sdpa_mask").value;
    // mask 逐层恒定；槽撑满相位后只需在层 0 灌一次（见 MASK_ONCE）。
    if (!cold_mask_once_ || layer_idx == 0) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                HYVLA_VLM_MASK_SCHEDULE_SITE,
                static_cast<int64_t>(
                    cold_mask_once_
                        ? HyVlaVlmGraphScheduleRoute::MASK_FIRST_LAYER_ONLY
                        : HyVlaVlmGraphScheduleRoute::MASK_EACH_LAYER),
                /*resolved_flags=*/0,
                {cold_mask_once_ ? 1 : 0, seq_len, chunk.kv_seq_len,
                 tp, num_layers()}, chunk.idx);
        }
        sdpa_dma_mask_to_spm(prepared_attn_mask_, mask_off, seq_len, chunk.kv_seq_len, tp);
    }
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_attention_route(
            HYVLA_VLM_ATTN_SITE, AttentionExecutionPolicy::DDR_KV,
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
            HYVLA_VLM_PREPARE_ATTN_ALL_REDUCE_SITE,
            static_cast<int64_t>(HyVlaVlmAllReduceRoute::PREPARE_RING_INPUT),
            /*resolved_flags=*/0, {seq_len, h, tp}, chunk.idx);
    }
    rpu_prepare_ring_all_reduce_input(addr(0, "oproj"), seq_len, h, tp);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_VLM_O_TEXT_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {seq_len, h, nq * hd, 0, tp}, chunk.idx);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0, /*scale=*/lw.o_ws);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, HYVLA_VLM_O_ACT_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            {seq_len, h, nq * hd, 0, tp}, chunk.idx);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), mw.o_w, addr(0, "oproj_a"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0, /*scale=*/mw.o_ws);
    emit_row_merge(addr(0, "oproj"), addr(0, "oproj_a"), addr(0, "oproj"),
                   m_txt, m_act, seq_len, cols_h, tp);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            HYVLA_VLM_ATTN_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0,
            {seq_len, h, tp, NUM_CORES}, chunk.idx);
    }
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
        seq_len, h, tp, NUM_CORES);

    // ── 步 5: post-attention RMSNorm 双算 (+ merge → residual1，除非 NOMERGE) ──
    // 与步 1 同理：两个 mlp_half 本来就是分别发射的 ⇒ 各读各的 norm 输出。
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION,
            HYVLA_VLM_POST_NORM_TOPOLOGY_SITE,
            static_cast<int64_t>(
                (cold_mot_norm_nomerge_ & MOT_NOMERGE_MLP)
                    ? HyVlaVlmNormalizationRoute::MOT_BRANCH_LOCAL
                    : HyVlaVlmNormalizationRoute::MOT_MERGED),
            /*resolved_flags=*/0,
            {static_cast<int64_t>(cold_mot_norm_nomerge_),
             (cold_mot_norm_nomerge_ & MOT_NOMERGE_MLP) ? 1 : 0,
             seq_len, h}, chunk.idx);
    }
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "residual1"),
        layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, eps_hyvla_,
        full_rmsnorm_route);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "norm_a2"),
        layer_addr(layer_idx, 0, "post_norm_w_act"), seq_len, h, eps_hyvla_,
        full_rmsnorm_route);
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
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVLA_VLM_MLP_GATE_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, intermediate_size(), h, 1, NUM_CORES}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in_addr, gate_w, addr(0, g),
            seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
            /*bias_spm_addr=*/0u, /*scale=*/gate_ws);
        if (!cold_silu_mul_) {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ACTIVATION,
                    HYVLA_VLM_MLP_SILU_SITE,
                    static_cast<int64_t>(
                        HyVlaVlmActivationRoute::SILU_UNARY),
                    /*resolved_flags=*/0,
                    {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES},
                    chunk.idx);
            }
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, g), addr(0, g), elems_mlp, ValuOpType::SILU,
                /*is_gelu=*/false, NUM_CORES);
        }
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVLA_VLM_MLP_UP_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, intermediate_size(), h, 1, NUM_CORES}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in_addr, up_w, addr(0, u),
            seq_len, intermediate_size(), h, /*partition=*/1, NUM_CORES,
            /*bias_spm_addr=*/0u, /*scale=*/up_ws);
        if (cold_silu_mul_) {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ACTIVATION,
                    HYVLA_VLM_MLP_SILU_MUL_SITE,
                    static_cast<int64_t>(
                        HyVlaVlmActivationRoute::SILU_MUL_FUSED),
                    /*resolved_flags=*/0,
                    {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES},
                    chunk.idx);
            }
            rpu_launch_silu_mul_spm_kernel(
                addr(0, g), addr(0, u), addr(0, g), elems_mlp, NUM_CORES);
        } else {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ACTIVATION,
                    HYVLA_VLM_MLP_MUL_SITE,
                    static_cast<int64_t>(HyVlaVlmActivationRoute::MUL),
                    /*resolved_flags=*/0,
                    {cold_silu_mul_ ? 1 : 0, elems_mlp, NUM_CORES},
                    chunk.idx);
            }
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, g), addr(0, u), addr(0, g),
                elems_mlp, ValuOpType::MUL, c10::Half(1.0f), NUM_CORES);
        }
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, HYVLA_VLM_MLP_DOWN_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0,
                {seq_len, h, intermediate_size(), 0, NUM_CORES}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, g), down_w, addr(0, dn),
            seq_len, h, intermediate_size(), /*partition=*/0, NUM_CORES,
            /*bias_spm_addr=*/0u, /*scale=*/down_ws);
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
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            HYVLA_VLM_MLP_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0,
            {seq_len, h, NUM_CORES, NUM_CORES}, chunk.idx);
    }
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
        seq_len, h, NUM_CORES, NUM_CORES);

    // ── 步 7 (末层): final RMSNorm —— **单次, 不双算** ──
    // Hy-VLA 的 ckpt 里 final norm 只有一份 (641 = 32×20 + 1, 无 norm_v.weight),
    // set_moe_weights 已断言 final_norm_moe_w 与 text 侧是同一张量。对同一个 γ
    // 双算再按行掩码 merge 恒等于单算 —— HALO 那份双算在这里是纯冗余, 每次末层
    // 白烧 1 次全宽 rmsnorm + 3 次全宽 eltwise, 还把 norm_a 的生命周期钉到
    // phase 8 (见 declare_buffers 的 norm_a1/norm_a2 说明)。rmsnorm 支持原地
    // (基类 rpu_qwen3_model.h:2146 同式), 故直接写回 residual1。
    if (is_last_layer) {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "residual1"),
            addr(0, "final_norm_w"), seq_len, h, eps_hyvla_,
            full_rmsnorm_route);
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
    const at::Tensor& attn_mask_4d, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor)
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

    // KV headroom: 7-D K 布局的 max_seq = size(1)*size(5)
    // (pi05 rpu_pi05_denoise_step_model.cpp:256-266 先例)。
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + cs <= kc.size(1) * kc.size(5),
                    "hyvla step_forward: prefix_len(", prefix_len, ")+cs(", cs,
                    ") exceeds k_cache capacity ", kc.size(1) * kc.size(5));
    }

    // host 侧合成的 x_emb 可能带脏 cache 行; BUILD 前显式 flush。
    rpu_ddr_flush_force(x_emb.data_ptr<c10::Half>());

    // 基类 forward 持有掩码准备 + run_all_layers; 行位置恒定 → 1D RoPE,
    // position_ids/deepstack 均空。返回基类 output_tensor_ (graph 写目标): 在
    // capture 作用域内它尚未执行, 必须由 Python 在作用域**退出后**读 (已证模式
    // wall_oss/pi05)。切勿在此 copy 到外部张量 —— 会读到尚未执行的全 0
    // (fused_model_base.cpp:1066-1086 契约)。
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

}  // namespace v3

// =============================================================================
// Instance registry — ModelHandleRegistry<v3::HyVlaVlmModel>
// =============================================================================
using HyVlaVlmRegistry = ModelHandleRegistry<v3::HyVlaVlmModel>;

std::vector<int64_t> rpu_hyvla_vlm_planner_cache_identity(int64_t handle) {
    return HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_hyvla_vlm_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_hyvla_vlm_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_hyvla_vlm_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_kvinsert_cost_domain")
        ->kvinsert_cost_domain("hyvla_vlm", descriptor);
}

std::string rpu_hyvla_vlm_kvinsert_cost_catalog_sha256(int64_t handle) {
    return HyVlaVlmRegistry::get(
        handle, "rpu_hyvla_vlm_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

// =============================================================================
// Public C API for TORCH_LIBRARY wrappers (file-scope; decls in rpu_kernel_decls.h)
// =============================================================================
int64_t rpu_hyvla_vlm_create() {
    return HyVlaVlmRegistry::create();
}

void rpu_hyvla_vlm_set_runtime_config(
    int64_t handle,
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool partial_rope, int64_t mot_norm_nomerge, bool silu_mul,
    int64_t rmsnorm_capability) {
    HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_set_runtime_config")
        ->set_runtime_config(
            fast_replay, fast_replay_preload, mask_once, partial_rope,
            mot_norm_nomerge, silu_mul, rmsnorm_capability);
}

void rpu_hyvla_vlm_destroy(int64_t handle) {
    HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_destroy")
        ->check_execution_reconfigure_destroy_allowed(
            "rpu_hyvla_vlm_destroy");
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
    const at::Tensor& attn_mask_4d, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor)
{
    return HyVlaVlmRegistry::get(handle, "rpu_hyvla_vlm_step_forward")
        ->step_forward(x_emb, k_caches, v_caches, cos, sin, attn_mask_4d,
                       prefix_len, planned_stage_descriptor);
}

std::vector<int64_t> rpu_hyvla_vlm_resolve_stage_domain(
    int64_t handle, int64_t seq_len, int64_t prefix_len,
    int64_t mask_kv_len) {
    return HyVlaVlmRegistry::get(
               handle, "rpu_hyvla_vlm_resolve_stage_domain")
        ->resolve_stage_domain(seq_len, prefix_len, mask_kv_len);
}

int64_t rpu_hyvla_vlm_get_resolved_chunk_size(int64_t handle) {
    return HyVlaVlmRegistry::get(
               handle, "rpu_hyvla_vlm_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

void rpu_hyvla_vlm_set_chunk_size_override(
    int64_t handle, int64_t chunk_size) {
    HyVlaVlmRegistry::get(
        handle, "rpu_hyvla_vlm_set_chunk_size_override")
        ->set_control_chunk_size_override(
            chunk_size, "rpu_hyvla_vlm_set_chunk_size_override");
}

void rpu_hyvla_vlm_enable_execution_reconfigure(int64_t handle) {
    HyVlaVlmRegistry::get(
        handle, "rpu_hyvla_vlm_enable_execution_reconfigure")
        ->enable_execution_reconfigure_guard();
}

void rpu_hyvla_vlm_stage_chunk_size_override(
    int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0, "Hy-VLA VLM hot-reconfigure token must be positive");
    HyVlaVlmRegistry::get(
        handle, "rpu_hyvla_vlm_stage_chunk_size_override")
        ->stage_control_chunk_size_override(
            static_cast<uint64_t>(token), chunk_size,
            "rpu_hyvla_vlm_stage_chunk_size_override");
}
