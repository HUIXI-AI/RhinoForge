// rpu_adarms_model.cpp — Pi0.5 Action Expert AdaRMS all-layers-once model
//                         (v3 FusedModelBase port)
//
// Plan 01-04: ports AdaRMSModel from v2's two-class framework pair to the
// v3 flat FusedModelBase single-class (from Plan 01-01). AdaRMS is the
// "negative control" subclass — flat-phase lifecycle, no D-501 callbacks,
// no persistent weights, no post-graph. Any ceremony required here would
// indicate framework scope creep; the port validates the v3 API for the
// cleanest shape (mirroring Qwen3 01-02 minimalism, without Qwen3's fused
// lm_head optional post-step).
//
// Key transformations vs v2 (mechanical; compute logic untouched):
//   - Inherit from v3::FusedModelBase (lives in `namespace v3`).
//   - Replace `buf(name)` at SDPA + KV-insert call sites with
//     `addr_offset(name).value` (Pitfall 3 structural fix via typed SpmOffset).
//   - Access model params via pimpl getters (num_layers() / hidden_size() /
//     num_q_heads() / num_kv_heads() / head_dim() / intermediate_size() /
//     attn_tp()) instead of v2 protected members.
//   - Drop the legacy subclass-side tensor-tracking call — class-member
//     `cond_ref_` assignment alone keeps the tensor alive for the synchronous
//     forward() (run_all_layers consumes the DDR pointer inside the batched
//     dispatch).
//   - set_weights ENDS with `invalidate_model_state();` as last non-empty
//     statement (D-503 per-function awk contract).
//   - Delete v2-only virtual overrides (chunk-size clip, KV-insert / compute /
//     preload / post subgraph stubs, post-alloc hook, layout-change hook,
//     persistent-invalidated hook, temporary-total estimator) — all absent
//     from the v3 subclass contract per D-203.
//
// Scope preserved from v2:
//   - Per-forward cond DMA (layer 0 / chunk 0 gate): `cond_loaded_this_forward_`
//     subclass flag toggled on the first build_layer_subgraph invocation.
//   - Per-layer GEMV (cond * dense_w + bias -> [scale|shift|gate]) at
//     chunk.idx == 0, result persists in SPM across that layer's remaining
//     chunks.
//   - (1+scale) preprocessing via scalar ADD to the live GEMV output SPM buffer.
//   - Gated residual 3-step: SUB full-size → 1xC->NxC MUL gate slice → ADD.
//   - Always SEQUENTIAL (causal autoregressive); flat phases (all LayerWide).
//   - No final_norm fusion (Pi0.5 flow-matching head runs on Python).
//   - cfg.cross_layer_batch_size = num_layers() (force single group).
//
// Design contract: docs/architecture.md#fusedmodelbase-v3--framework-contract

#include "pi05_nvfp4.h"
#include "fused_model_base.h"
#include "pi05_execution_topology.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"  // v5-07: shared runtime globals (g_chunk_size_override, get_cross_layer_batch_size)
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define DWIDTH 2

namespace v3 {

namespace {

// ADARMS_FIXED_KERNEL_BASIS: every non-manifest launcher below is a model-math
// normalization/activation or fixed BufferDecl transport with no alternative
// implementation. A selectable implementation must become a typed route first.

constexpr int64_t ADARMS_CORE_PROFILE_SITE = 1727285745909984687LL;
constexpr int64_t ADARMS_COND_PREPARE_SITE = 4545122598025177250LL;
constexpr int64_t ADARMS_COND_DMA_SITE = 1365898338277373764LL;
constexpr int64_t ADARMS_Q_LINEAR_SITE = 3720663967042490210LL;
constexpr int64_t ADARMS_K_LINEAR_SITE = 737407077005117954LL;
constexpr int64_t ADARMS_V_LINEAR_SITE = 2757305400171169493LL;
constexpr int64_t ADARMS_Q_ROPE_SITE = 5408876450787035953LL;
constexpr int64_t ADARMS_K_ROPE_SITE = 7158713397924778966LL;
constexpr int64_t ADARMS_KV_INSERT_SITE = 1489108772795312297LL;
constexpr int64_t ADARMS_ATTENTION_SITE = 892830053731625482LL;
constexpr int64_t ADARMS_PREPARE_ALL_REDUCE_SITE = 2713753424009463107LL;
constexpr int64_t ADARMS_O_LINEAR_SITE = 8920591606190302727LL;
constexpr int64_t ADARMS_ATTENTION_ALL_REDUCE_SITE = 1378336082325261091LL;
constexpr int64_t ADARMS_GEMV_LINEAR_SITE = 73920063418775617LL;
constexpr int64_t ADARMS_GEMV_ALL_REDUCE_SITE = 3363421386255464359LL;

enum class AdarmsAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

enum class AdarmsRopeRoute : int64_t {
    ROPE_1D = 1,
};

enum class AdarmsMutableDmaRoute : int64_t {
    DDR_SCATTER_TO_SPM = 1,
};

int64_t adarms_ring_route(int64_t rows, int64_t cols, int cores) {
    return fmb_ring_all_reduce_route_selector(rows, cols, cores);
}

void check_adarms_scale_lists(
    int64_t n_layers,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w,
    at::TensorList o_w, at::TensorList gate_w, at::TensorList up_w,
    at::TensorList down_w,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws, bool nvfp4)
{
    const bool has_scale = !q_ws.empty();
    auto check_scale_list = [&](const at::TensorList& list, const char* name) {
        if (has_scale) {
            TORCH_CHECK(static_cast<int64_t>(list.size()) == n_layers,
                        "adarms_set_weights: ", name, ".size()=", list.size(),
                        " != num_layers=", n_layers);
        } else {
            TORCH_CHECK(list.empty(),
                        "adarms_set_weights: ", name,
                        " must be empty unless all quantized scale lists are provided");
        }
    };
    check_scale_list(k_ws,    "k_w_scale");
    check_scale_list(v_ws,    "v_w_scale");
    check_scale_list(o_ws,    "o_w_scale");
    check_scale_list(gate_ws, "gate_scale");
    check_scale_list(up_ws,   "up_scale");
    check_scale_list(down_ws, "down_scale");
    if (has_scale) {
        TORCH_CHECK(static_cast<int64_t>(q_ws.size()) == n_layers,
                    "adarms_set_weights: q_w_scale.size()=", q_ws.size(),
                    " != num_layers=", n_layers);
    }

    auto check_weight_dtype = [&](const at::Tensor& w,
                                  const at::Tensor& scale,
                                  const char* name,
                                  int64_t i) {
        TORCH_CHECK(w.device().type() == at::kPrivateUse1 && w.is_contiguous(),
                    "adarms_set_weights: ", name, "[", i,
                    "] must be contiguous RPU tensor");
        TORCH_CHECK(w.scalar_type() == at::kHalf || w.scalar_type() == at::kChar
                    || w.scalar_type() == at::kByte,
                    "adarms_set_weights: ", name, "[", i,
                    "] must be fp16 / int8 / packed-uint8, got ", w.scalar_type());
        if (has_scale) {
            TORCH_CHECK(w.scalar_type() == at::kChar || w.scalar_type() == at::kByte,
                        "adarms_set_weights: quantized mode requires int8(W8A16) or "
                        "uint8(packed-INT4) ", name, "[", i, "], got ", w.scalar_type());
            TORCH_CHECK(scale.defined() && (scale.scalar_type() == at::kHalf || (nvfp4 && scale.scalar_type() == at::kByte))
                        && scale.device().type() == at::kPrivateUse1
                        && scale.is_contiguous(),
                        "adarms_set_weights: ", name, "_scale[", i,
                        "] must be a contiguous fp16 RPU tensor");
            if (w.scalar_type() == at::kChar) {
                TORCH_CHECK(scale.scalar_type() == at::kHalf && scale.dim() == 1 && scale.numel() == w.size(0),
                            "adarms_set_weights: W8A16 ", name, "_scale[", i,
                            "] must be per-channel [N=", w.size(0), "]");
            } else if (nvfp4) {
                TORCH_CHECK(scale.scalar_type() == at::kByte && scale.dim() == 2 &&
                                scale.size(0) == w.size(1) / 8 && scale.size(1) == w.size(0),
                            "Pi05 NVFP4 v2 requires FP8 striped [K/16,N] scales");
            } else {
                TORCH_CHECK(scale.dim() == 2 &&
                                (scale.size(0) == 32 || scale.size(0) == 64 ||
                                 scale.size(0) == 128),
                            "adarms_set_weights: W4A16 ", name, "_scale[", i,
                            "] must be a controller-striped pgrp 2D payload");
            }
        } else {
            TORCH_CHECK(w.scalar_type() != at::kChar && w.scalar_type() != at::kByte,
                        "adarms_set_weights: quantized ", name, "[", i,
                        "] requires scale lists");
        }
    };

    for (int64_t i = 0; i < n_layers; ++i) {
        if (nvfp4) {
            TORCH_CHECK(has_scale && q_w[i].scalar_type() == at::kByte &&
                o_w[i].scalar_type() == at::kByte && gate_w[i].scalar_type() == at::kByte &&
                up_w[i].scalar_type() == at::kByte && down_w[i].scalar_type() == at::kByte &&
                k_w[i].scalar_type() == at::kChar && v_w[i].scalar_type() == at::kChar,
                "Pi05 NVFP4 v2 scope must be Q/O/Gate/Up/Down FP4, K/V W8");
        }
        check_weight_dtype(q_w[i],    has_scale ? q_ws[i]    : at::Tensor(), "q_w", i);
        check_weight_dtype(k_w[i],    has_scale ? k_ws[i]    : at::Tensor(), "k_w", i);
        check_weight_dtype(v_w[i],    has_scale ? v_ws[i]    : at::Tensor(), "v_w", i);
        check_weight_dtype(o_w[i],    has_scale ? o_ws[i]    : at::Tensor(), "o_w", i);
        check_weight_dtype(gate_w[i], has_scale ? gate_ws[i] : at::Tensor(), "gate_w", i);
        check_weight_dtype(up_w[i],   has_scale ? up_ws[i]   : at::Tensor(), "up_w", i);
        check_weight_dtype(down_w[i], has_scale ? down_ws[i] : at::Tensor(), "down_w", i);
    }
}

}  // namespace

// =============================================================================
// AdaRMSModel — v3::FusedModelBase subclass (Pi0.5 Action Expert)
// =============================================================================

class AdaRMSModel : public FusedModelBase {
public:
    // Per-layer weights. Dense weights are expanded+transformed on Python side
    // to [8*3*hidden, hidden] (col-partition swizzled). Dense biases are
    // [3*hidden] fp16.
    struct LayerWeights {
        // Attention
        at::Tensor q_w, k_w, v_w, o_w;
        // MLP
        at::Tensor gate_proj_w, up_proj_w, down_proj_w;
        at::Tensor q_ws, k_ws, v_ws, o_ws;
        at::Tensor gate_ws, up_ws, down_ws;
        // AdaRMS GEMV (per-layer, not shared)
        at::Tensor attn_dense_w;    // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor attn_dense_b;    // [3*hidden] fp16
        at::Tensor mlp_dense_w;     // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor mlp_dense_b;     // [3*hidden] fp16
    };

    explicit AdaRMSModel(bool linear_acc32 = false) : linear_acc32_(linear_acc32) {}

private:
    const bool linear_acc32_;
public:

    void set_execution_cores(int64_t cores) {
        TORCH_CHECK(cores == 4 || cores == 6 || cores == 8,
                    "AdaRMS execution cores must be 4, 6 or 8");
        TORCH_CHECK(layer_weights_.empty(),
                    "AdaRMS execution cores must precede weight installation");
        set_execution_core_count(static_cast<int>(cores));
    }

    int condition_tp() const { return num_cores() == 8 ? 8 : 4; }

    std::vector<int64_t> execution_topology() const {
        TORCH_CHECK(num_layers() > 0 && !layer_weights_.empty(),
                    "AdaRMS topology requires installed model weights");
        return {1, num_cores(), attn_tp(), mlp_tp(), condition_tp(), 8,
                 logical_intermediate_size_, intermediate_size()};
    }

    std::vector<int64_t> core_profile_arguments() const {
        auto args = execution_topology();
        args.push_back(reduced_w8a16_ ? 1 : 0);
        return args;
    }

    DecoderExecutionTopology resolve_model_execution_topology(
            int64_t nq, int64_t nkv, int64_t hd, int64_t h,
            int64_t intermediate) const override {
        if (num_cores() == 8)
            return FusedModelBase::resolve_model_execution_topology(
                nq, nkv, hd, h, intermediate);
        return resolve_pi05_reduced_physical_topology(
            num_cores(), nq, nkv, hd, h, intermediate, true);
    }

    int64_t subclass_layout_hash() const override {
        if (num_cores() == 8) return linear_acc32_ ? 32 : 0;
        int64_t hash = 0;
        for (const auto value : core_profile_arguments())
            hash = detail::layout_mix(hash, value);
        return linear_acc32_ ? detail::layout_mix(hash, 32) : hash;
    }


    std::vector<int64_t> resolve_action_stage_domain(
        int64_t execution_len, int64_t logical_len, int64_t position,
        int64_t kv_len, bool use_attention_mask, bool is_causal,
        int64_t requested_chunk_size)
    {
        detail::validate_fmb_planning_shape(
            execution_len, position, "AdaRMS action planner");
        TORCH_CHECK(
            logical_len > 0 && logical_len <= execution_len,
            "RPU_PLANNER_REJECT:CAPABILITY: AdaRMS action logical length "
            "must be in [1, execution_len]");
        TORCH_CHECK(
            kv_len >= position + execution_len,
            "RPU_PLANNER_REJECT:CAPABILITY: AdaRMS action KV length does "
            "not cover prefix plus execution rows");
        std::optional<at::Tensor> mask_shape = std::nullopt;
        if (use_attention_mask) {
            mask_shape = at::empty(
                {1, 1, execution_len, kv_len},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        }
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                execution_len, position, mask_shape, is_causal,
                requested_chunk_size, logical_len));
    }

    // ========================================================================
    // set_weights -- D-405 / D-407 (Python converter calls this)
    //
    // Layer-specific AdaRMS dense weights (attn & mlp) are per-layer.
    // cos/sin are globals computed on RPU device with kernel format [max_pos, head_dim/2].
    // eps is the RMSNorm epsilon (typically 1e-6).
    // ========================================================================
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
        at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
        const at::Tensor& cos, const at::Tensor& sin,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps,
        at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
        at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
        at::TensorList gate_scale_list, at::TensorList up_scale_list,
        at::TensorList down_scale_list, at::TensorList nvfp4_tensor_scales)
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "adarms_set_weights: empty weight lists");
        TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0
                    && hidden_size > 0 && intermediate_size > 0,
                    "adarms_set_weights: dim params must be positive");
        TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                    "adarms_set_weights: num_q_heads (", num_q_heads,
                    ") must be divisible by num_kv_heads (", num_kv_heads, ")");

        // All 11 per-layer lists must have the same length
        auto check_list = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "adarms_set_weights: ", n, ".size()=", l.size(),
                        " != num_layers=", N);
        };
        check_list(k_w_list,          "k_w_list");
        check_list(v_w_list,          "v_w_list");
        check_list(o_w_list,          "o_w_list");
        check_list(gate_list,         "gate_list");
        check_list(up_list,           "up_list");
        check_list(down_list,         "down_list");
        check_list(attn_dense_w_list, "attn_dense_w_list");
        check_list(attn_dense_b_list, "attn_dense_b_list");
        check_list(mlp_dense_w_list,  "mlp_dense_w_list");
        check_list(mlp_dense_b_list,  "mlp_dense_b_list");
        check_adarms_scale_lists(
            N,
            q_w_list, k_w_list, v_w_list, o_w_list,
            gate_list, up_list, down_list,
            q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
            gate_scale_list, up_scale_list, down_scale_list, !nvfp4_tensor_scales.empty());
        Pi05Nvfp4Tables nvfp4_tables;
        nvfp4_tables.install(nvfp4_tensor_scales, N, num_cores());
        const bool has_scale = !q_w_scale_list.empty();

        // Global tensor checks
        TORCH_CHECK(cos.defined() && sin.defined(),
                    "adarms_set_weights: cos/sin must be defined");
        TORCH_CHECK(cos.dim() == 2,
                    "adarms_set_weights: cos must be 2D, got ", cos.dim(), "D");
        TORCH_CHECK(sin.sizes() == cos.sizes(),
                    "adarms_set_weights: sin.sizes() must equal cos.sizes()");
        TORCH_CHECK(cos.size(-1) == head_dim || cos.size(-1) == head_dim / 2,
                    "adarms_set_weights: cos last dim must be head_dim or head_dim/2");

        // Per-layer defined/rank checks (2D for linear weights, 1D for bias)
        auto check_all_defined_rank = [&](const at::TensorList& list,
                                          const char* name, int64_t expected_rank) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "adarms_set_weights: ", name, "[", i, "] is undefined");
                TORCH_CHECK(list[i].dim() == expected_rank,
                            "adarms_set_weights: ", name, "[", i, "] must be ",
                            expected_rank, "D, got ", list[i].dim(), "D");
            }
        };
        check_all_defined_rank(q_w_list,          "q_w_list",          2);
        check_all_defined_rank(k_w_list,          "k_w_list",          2);
        check_all_defined_rank(v_w_list,          "v_w_list",          2);
        check_all_defined_rank(o_w_list,          "o_w_list",          2);
        check_all_defined_rank(gate_list,         "gate_list",         2);
        check_all_defined_rank(up_list,           "up_list",           2);
        check_all_defined_rank(down_list,         "down_list",         2);
        // Dense weight is expanded+transformed [8*3*hidden, hidden] (2D); bias is [3*hidden] (1D)
        check_all_defined_rank(attn_dense_w_list, "attn_dense_w_list", 2);
        check_all_defined_rank(attn_dense_b_list, "attn_dense_b_list", 1);
        check_all_defined_rank(mlp_dense_w_list,  "mlp_dense_w_list",  2);
        check_all_defined_rank(mlp_dense_b_list,  "mlp_dense_b_list",  1);

        // Commit model params (pimpl: hidden/q/kv/head_dim/intermediate + attn_tp_)
        int64_t physical_intermediate = intermediate_size;
        if (num_cores() != 8) {
            const auto dtype = q_w_list[0].scalar_type();
            TORCH_CHECK(dtype == at::kHalf || dtype == at::kChar,
                        "reduced Pi0.5 supports FP16 or W8A16 projections");
            physical_intermediate = validate_pi05_reduced_geometry(
                num_cores(), num_q_heads, num_kv_heads, head_dim,
                hidden_size, intermediate_size, N, true,
                dtype == at::kChar);
            for (int64_t i = 0; i < N; ++i) {
                std::array<std::array<int64_t, 2>, 7> shapes;
                size_t j = 0;
                for (const auto* tensor : {&q_w_list[i], &k_w_list[i],
                        &v_w_list[i], &o_w_list[i], &gate_list[i],
                        &up_list[i], &down_list[i]}) {
                    TORCH_CHECK(tensor->scalar_type() == dtype && tensor->dim() == 2,
                                "Pi0.5 reduced projection precision/rank mismatch");
                    shapes[j++] = {tensor->size(0), tensor->size(1)};
                }
                validate_pi05_reduced_projection_shapes(
                    shapes, hidden_size, physical_intermediate);
                // Baseline orchestration keeps modulation FP16. The fused
                // denoise owner independently accepts its cold W8 dense path.
                for (const auto* tensor : {&attn_dense_w_list[i], &mlp_dense_w_list[i]})
                    TORCH_CHECK(tensor->scalar_type() == at::kHalf &&
                                    tensor->device().type() == at::kPrivateUse1 &&
                                    tensor->is_contiguous() &&
                                    tensor->size(0) == 3 * hidden_size &&
                                    tensor->size(1) == hidden_size,
                                "Pi0.5 AdaRMS modulation must be FP16 [3H,H]");
                for (const auto* tensor : {&attn_dense_b_list[i], &mlp_dense_b_list[i]})
                    TORCH_CHECK(tensor->scalar_type() == at::kHalf &&
                                    tensor->device().type() == at::kPrivateUse1 &&
                                    tensor->is_contiguous() &&
                                    tensor->numel() == 3 * hidden_size,
                                "Pi0.5 AdaRMS modulation bias must be FP16 [3H]");
            }
        }
        logical_intermediate_size_ = intermediate_size;
        reduced_w8a16_ = q_w_list[0].scalar_type() == at::kChar;
        set_model_params(num_q_heads, num_kv_heads, head_dim,
                         hidden_size, physical_intermediate);
        set_num_layers(N);
        eps_ = eps;

        // AdaRMS-specific derived values (the v3 framework's set_model_params
        // already sets pimpl attn_tp_ = std::min(8, num_kv_heads), matching
        // v2's std::min(NUM_CORES, num_kv_heads)). Cache here for readability
        // inside the hot build_layer_subgraph path.
        local_q_heads_ = num_q_heads / attn_tp();
        local_kv_dim_  = num_kv_heads * head_dim / attn_tp();

        // Copy weights into layer state
        nvfp4_ = std::move(nvfp4_tables);
        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                gate_list[i], up_list[i], down_list[i],
                has_scale ? rpu_retain_linear_quant_scale(q_w_list[i], q_w_scale_list[i]) : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(k_w_list[i], k_w_scale_list[i]) : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(v_w_list[i], v_w_scale_list[i]) : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(o_w_list[i], o_w_scale_list[i]) : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(gate_list[i], gate_scale_list[i]) : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(up_list[i], up_scale_list[i]) : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(down_list[i], down_scale_list[i]) : at::Tensor(),
                attn_dense_w_list[i], attn_dense_b_list[i],
                mlp_dense_w_list[i],  mlp_dense_b_list[i],
            });
        }
        cos_ = cos;
        sin_ = sin;

        invalidate_model_state();  // D-503: last non-empty statement of set_weights
    }


    std::vector<int64_t> planner_cache_identity() const {
        auto identity = FusedModelBase::planner_cache_identity();
        identity.push_back(rope_position_);
        for (const auto& table : nvfp4_.values) append_kvinsert_cost_tensor_identity(identity, table);
        return identity;
    }

    void set_rope_position(int64_t position) {
        rope_position_ = position;
    }

    // ========================================================================
    // forward -- D-405: cond is explicit per-call argument on RPU device
    //
    // Each forward = one Pi0.5 denoising step with a new cond.
    // Cache key (D-404) excludes cond values; REPLAY cursor-patches the
    // cond DDR pointer on each subsequent denoise step.
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& hidden_states,
        const at::Tensor& cond,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal,
        at::IntArrayRef planned_stage_descriptor = {})
    {
        TORCH_CHECK(num_layers() > 0,
                    "AdaRMSModel::forward called before set_weights");
        TORCH_CHECK(!layer_weights_.empty(),
                    "AdaRMSModel::forward: internal state inconsistent");
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "AdaRMSModel::forward: hidden_states must be on RPU device");
        TORCH_CHECK(cond.defined() && cond.dim() == 1 &&
                    cond.size(0) == hidden_size(),
                    "AdaRMSModel::forward: cond must be 1D [hidden_size=",
                    hidden_size(), "], got dim=", (cond.defined() ? cond.dim() : -1),
                    " size=", (cond.defined() ? cond.size(0) : -1));
        TORCH_CHECK(cond.device().type() == at::kPrivateUse1,
                    "AdaRMSModel::forward: cond must be on RPU device");
        TORCH_CHECK(cond.scalar_type() == at::kHalf,
                    "AdaRMSModel::forward: cond must be fp16");
        TORCH_CHECK(position >= 0,
                    "AdaRMSModel::forward: position must be non-negative, got ", position);
        TORCH_CHECK(hidden_states.is_contiguous(),
                    "AdaRMSModel::forward: hidden_states must be contiguous "
                    "(downstream kernels use raw data_ptr())");
        TORCH_CHECK(cond.is_contiguous(),
                    "AdaRMSModel::forward: cond must be contiguous "
                    "(DMA source pointer assumes row-major layout)");

        TORCH_CHECK(
            !RpuKernelGraph::has_active() || !planned_stage_descriptor.empty(),
            "AdaRMS production Graph forward requires one complete planner "
            "stage descriptor");

        // Reset per-forward cond state: a new denoise step is a new cond.
        // cond DMA/GEMV will re-emit into the graph on the first chunk of layer 0.
        // cond_ref_ keeps the tensor alive for the synchronous forward (batch
        // end executes before run_all_layers returns); the DDR pointer is
        // cursor-patched per REPLAY via mutable DMA + cond_src_base_.
        cond_loaded_this_forward_ = false;
        cond_ref_ = cond;
        // cond_src_base_ feeds rpu_launch_ddr_scatter_spm_dma_mutable in the
        // cond scatter site below (Phase 0). Set once per forward; REPLAY end()
        // patches each per-core DMA's src via Queue_t::update_dma_kernel from
        // this live storage. Per-forward flush mirrors fused_model_base's
        // hidden_in_src_base_ pattern (caller may have CPU-dirty lines).
        cond_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond.data_ptr());
        // Conditioning is a per-forward caller-owned tensor, so the CPU may
        // have written cached data since the last submission. SDK command
        // publication does not publish those tensor writes. Flush the live
        // input before the mutable DMA reads it, as in CPU fallback write-back.
        //
        //
        //
        //
        //
        rpu_ddr_flush_force(cond.data_ptr<c10::Half>());

        // PERF L1 (2026-05-07): hoist SDPA mask prepare out of the per-layer
        // Phase 6. The mask is invariant across all expert layers within one
        // forward; preparing once + DMA-only per layer drops constant_pad_nd
        // calls 18× (Pi0.5 5-step e2e: 90 → 5). The single-chunk-per-forward
        // assumption (suffix_len <= chunk_size auto-pick) matches the existing
        // sdpa_prepare_mask shape contract: it asserts mask_2d.size(0)==seq_q,
        // which already requires seq_q == ctx().seq_len == chunk.len.
        //
        // PERF L2 (2026-05-07): also cache the prepared mask across forwards
        // when the caller passes the SAME attention_mask tensor. Pi0.5 denoise
        // builds the mask once outside the 5-step loop (runtime.py L2 hoist),
        // so steps 2-5 see the same data_ptr → cache hit → skip the prepare
        // (saves 4 constant_pad_nd + 4 DDR copies per inference; 5 → 1).
        // We hold a strong ref to the input tensor so its storage cannot be
        // reallocated to a different tensor while the cache is valid.
        const int64_t prep_seq_q = hidden_states.size(1);
        const int64_t prep_seq_k = position + prep_seq_q;
        const bool    prep_is_causal = is_causal && (prep_seq_q > 1);
        const bool    has_mask = attention_mask.has_value() && attention_mask->defined();
        const void*   in_ptr = has_mask ? attention_mask->data_ptr() : nullptr;
        const void*   cached_ptr = prepared_mask_input_ref_.defined()
                                   ? prepared_mask_input_ref_.data_ptr() : nullptr;
        const bool cache_hit =
            (cached_ptr               == in_ptr)        &&
            (prepared_mask_seq_q_     == prep_seq_q)    &&
            (prepared_mask_seq_k_     == prep_seq_k)    &&
            (prepared_mask_is_causal_ == prep_is_causal);
        if (!cache_hit) {
            prepared_mask_ = sdpa_prepare_mask(
                attention_mask, prep_is_causal, prep_seq_q, prep_seq_k,
                sdpa_stable_mask_cache());
            prepared_mask_input_ref_  = has_mask ? attention_mask.value() : at::Tensor();
            prepared_mask_seq_q_      = prep_seq_q;
            prepared_mask_seq_k_      = prep_seq_k;
            prepared_mask_is_causal_  = prep_is_causal;
        }

        return run_all_layers(hidden_states, k_caches, v_caches,
                              attention_mask, position, is_causal,
                              /*planned_chunk_size=*/0,
                              planned_stage_descriptor);
    }

protected:
    // ========================================================================
    // static_config -- graph-naive flat-phase subclass; no legacy graph slot.
    //
    // AdaRMS is the "no D-501 callbacks" shape — all four pointer-to-member
    // slots (preload_fn / kv_first_fn / kv_first_chunk_plan_fn / post_fn)
    // stay nullptr by default. Flat-phase baseline; no persistent weights.
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers       = num_layers();
        // Force single group — matches Gemma/SigLIP/QwenPi05/Qwen3 discipline.
        // Round-2 codex-review cleanup of the CR-merge twin: the previous
        // get_cross_layer_batch_size() read + rt>0?rt:num_layers() was always
        // overwritten by the next line with num_layers(), making the runtime
        // knob read dead. See rpu_siglip_model.cpp:358 for the authoritative
        // note on why every v3 subclass pins a single group and ignores the
        // global runtime knob.
        cfg.cross_layer_batch_size = num_layers();
        // D-501 callbacks: all nullptr for AdaRMS (no preload / no KV_FIRST / no post).
        return cfg;
    }

    // ========================================================================
    // dynamic_config -- always SEQUENTIAL
    // ========================================================================
    ModelDynamicConfig dynamic_config(const ChunkPlan& /*plan*/) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;   // AdaRMS never uses KV_FIRST
        cfg.inter_layer_io = InterLayerIO::AUTO;
        return cfg;
    }

    bool subclass_chunk_size_valid(
        int64_t chunk_size, int64_t seq_len,
        int64_t /*position*/) const override {
        // The action suffix attends bidirectionally whenever prefix history is
        // present. Splitting it would hide later suffix rows from an earlier
        // chunk, so the only valid domain is one physical chunk.
        return chunk_size >= seq_len;
    }

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& /*layout*/,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
        TORCH_CHECK(
            plan.compute.chunks.size() == 1,
            "AdaRMS COMPLETE descriptor requires one action suffix chunk");

        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = position + logical_len;
        manifest.kv_insert_physical_rows = physical_len;
        manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation = linear_acc32_ ? FmbLinearAccumulationPolicy::ACC32
                                                   : FmbLinearAccumulationPolicy::ACC16;

        auto append = [&](FmbRouteFamily family, int64_t site_id,
                          int64_t selector,
                          std::vector<int64_t> arguments = {},
                          int64_t invocation = 0) {
            manifest.routes.push_back({
                site_id, family, selector, /*flags=*/0,
                std::move(arguments), invocation});
        };
        append(
            FmbRouteFamily::MUTABLE_DMA, ADARMS_COND_DMA_SITE,
            static_cast<int64_t>(
                AdarmsMutableDmaRoute::DDR_SCATTER_TO_SPM));
        append(
            FmbRouteFamily::LINEAR, ADARMS_GEMV_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE));
        append(
            FmbRouteFamily::ALL_REDUCE, ADARMS_GEMV_ALL_REDUCE_SITE,
            adarms_ring_route(/*rows=*/1, /*cols=*/3 * hidden_size(), num_cores()));

        constexpr uint32_t kv_capabilities =
            KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16;
        const int64_t rope_base = rope_position_ >= 0
            ? rope_position_ : position;
        for (const ChunkInfo& chunk : plan.compute.chunks) {
            const int64_t invocation = chunk.idx;
            for (const int64_t site_id : {
                     ADARMS_Q_LINEAR_SITE,
                     ADARMS_K_LINEAR_SITE,
                     ADARMS_V_LINEAR_SITE}) {
                append(
                    FmbRouteFamily::LINEAR, site_id,
                    static_cast<int64_t>(
                        FmbLinearRouteSelector::AUTO_TILE),
                    {}, invocation);
            }
            append(
                FmbRouteFamily::ROPE, ADARMS_Q_ROPE_SITE,
                static_cast<int64_t>(AdarmsRopeRoute::ROPE_1D),
                {rope_base + chunk.offset}, invocation);
            append(
                FmbRouteFamily::ROPE, ADARMS_K_ROPE_SITE,
                static_cast<int64_t>(AdarmsRopeRoute::ROPE_1D),
                {rope_base + chunk.offset}, invocation);

            const KvInsertSegmentPlan kv_plan =
                resolve_kvinsert_plan_auto(
                    ADARMS_KV_INSERT_SITE, manifest.graph_lifecycle,
                    position + chunk.offset, chunk.len, chunk.len,
                    attn_tp(), num_kv_heads(), head_dim(), kv_capabilities);
            const KvInsertRouteArguments kv_arguments =
                rpu_kvinsert_route_arguments(
                    kv_plan, attn_tp(), num_kv_heads(), head_dim());
            append(
                FmbRouteFamily::KV_INSERT, ADARMS_KV_INSERT_SITE,
                static_cast<int64_t>(kv_plan.route()),
                {kv_arguments.begin(), kv_arguments.end()}, invocation);
            append(
                FmbRouteFamily::ATTENTION, ADARMS_ATTENTION_SITE,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                {}, invocation);
            append(
                FmbRouteFamily::ALL_REDUCE,
                ADARMS_PREPARE_ALL_REDUCE_SITE,
                static_cast<int64_t>(
                    AdarmsAllReduceRoute::PREPARE_RING_INPUT),
                {}, invocation);
            append(
                FmbRouteFamily::LINEAR, ADARMS_O_LINEAR_SITE,
                static_cast<int64_t>(nvfp4_.enabled() ? (linear_acc32_ ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                                                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16) : FmbLinearRouteSelector::AUTO_TILE),
                {}, invocation);
            append(
                FmbRouteFamily::ALL_REDUCE,
                ADARMS_ATTENTION_ALL_REDUCE_SITE,
                adarms_ring_route(chunk.len, hidden_size(), num_cores()),
                {}, invocation);
        }
        if (num_cores() != 8) {
            manifest.routes.push_back({ADARMS_CORE_PROFILE_SITE,
                FmbRouteFamily::GRAPH_SCHEDULE, 1, 0,
                core_profile_arguments(), 0});
        }
        if (condition_tp() != num_cores()) {
            manifest.routes.push_back({ADARMS_COND_PREPARE_SITE,
                FmbRouteFamily::ALL_REDUCE, 3, 0,
                {condition_tp(), num_cores(), 1}, 0});
        }
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(),
            FMB_SHARED_LAYER_INPUT_DMA |
                FMB_SHARED_MLP_AUTO_TILE |
                FMB_SHARED_MLP_RING_REDUCE, 1, num_cores(), mlp_tp());
        std::sort(
            manifest.routes.begin(), manifest.routes.end(),
            [](const FmbRouteManifestEntry& lhs,
               const FmbRouteManifestEntry& rhs) {
                const auto lhs_family = static_cast<int64_t>(lhs.family);
                const auto rhs_family = static_cast<int64_t>(rhs.family);
                if (lhs_family != rhs_family) return lhs_family < rhs_family;
                if (lhs.site_id != rhs.site_id) return lhs.site_id < rhs.site_id;
                return lhs.invocation < rhs.invocation;
            });
    if (nvfp4_.enabled()) {
        for (auto& route : manifest.routes) {
            if (route.family == FmbRouteFamily::LINEAR &&
                (route.site_id == ADARMS_Q_LINEAR_SITE || route.site_id == ADARMS_O_LINEAR_SITE ||
                 route.site_id == FMB_SHARED_MLP_AUTO_TILE_SITE))
                route.selector = static_cast<int64_t>((linear_acc32_ ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                                                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16));
        }
    }
        return manifest;
    }

    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& /*manifest*/) const override {
        return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
    }

    // ========================================================================
    // declare_buffers -- flat phases, ALL LayerWide (matches v2 verbatim).
    //
    // AdaRMS adds 5 NEW SPM buffers on top of the decoder skeleton:
    //   - cond            [hidden_size]      -- DMA'd once per forward
    //   - attn_gemv       [3 * hidden_size]  -- scale|shift|gate for attn norm
    //   - mlp_gemv        [3 * hidden_size]  -- scale|shift|gate for mlp  norm
    //   - bias_temp       [3 * hidden_size]  -- temporary for bias DMA + add
    //   - gemv_partial    [3 * hidden_size]  -- per-core partial before all_reduce
    // All BufferScope::LayerWide (flat phases — aliasing deferred; the
    // v3 framework handles lifecycle aliasing via phase ranges, but AdaRMS
    // uses 0,0 to match v2's forward parity).
    //
    // No D-502 per-buffer preload hook on any decl — all buffers are StorageClass::Temp.
    // ========================================================================
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        int64_t cs = ctx.chunk_size;
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();

        int64_t local_q  = nq / attn_tp();
        int64_t local_kv = nkv * hd / attn_tp();
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        int64_t res   = A(cs * h * DWIDTH);
        int64_t q     = A(cs * local_q * hd * DWIDTH);
        int64_t kv    = A(cs * local_kv * DWIDTH);
        int64_t out   = A(cs * local_q * hd * DWIDTH);
        int64_t oproj = A(cs * h * DWIDTH);
        int64_t mlp   = A(cs * (is_ / mlp_tp()) * DWIDTH);

        // SDPA tmp and mask sizing
        SdpaConfig sdpa_cfg = make_sdpa_config(ctx.use_attn_mask ? 4 : 1);
        int64_t tmp = A(sdpa_compute_tmp_v16_size(sdpa_cfg, cs) * 32);
        int64_t mask_sz = ctx.use_attn_mask
            ? A(cs * CeilDiv(ctx.max_kv_seq_len, (int64_t)16) * 32)
            : 0;

        // AdaRMS extras (sized per-core; broadcast DMA means same addr on all cores)
        int64_t cond_sz     = A(h * DWIDTH);
        int64_t gemv_out_sz = A(3 * h * DWIDTH);

        constexpr BufferScope ALL = BufferScope::LayerWide;

        std::vector<BufferDecl> decls{
            // Structural (mirrors Gemma, but flat phases 0,0)
            {"residual1",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"input_norm",   res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"residual2",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"q",            q,       0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"k",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"v",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"output",       out,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"oproj",        oproj,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_tmp",     tmp,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_mask",    mask_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"gate",         mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"up",           mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"down",         res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},

            // AdaRMS-specific (flat phases, LayerWide)
            {"cond",         cond_sz,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"attn_gemv",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"mlp_gemv",     gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"bias_temp",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"gemv_partial", gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
        };
        nvfp4_.declare(decls, num_cores());
        return decls;
    }

    // ========================================================================
    // build_layer_subgraph (SEQUENTIAL) -- THE core of Phase 4
    //
    // Execution flow per layer (ported from v2 adarms_gemma_process_single_chunk):
    //
    //   Phase 0 (layer 0, chunk 0 ONLY): DMA cond DDR -> "cond" SPM (scatter)
    //   Phase A (every layer, chunk 0):  GEMV attn_dense_w * cond + attn_dense_b -> "attn_gemv"
    //                                    GEMV mlp_dense_w  * cond + mlp_dense_b  -> "mlp_gemv"
    //                                    +1.0 on first hidden_size slice (scale += 1)
    //                                    Result persists in SPM across chunks of this layer.
    //   Phase 1:  input DMA -> "residual1"
    //   Phase 2:  RMSNorm with (1+scale) weight  "residual1" -> "input_norm"
    //             1xC->NxC ADD shift (broadcast shift slice onto each row) in-place
    //   Phase 3:  QKV Linear (col partition, attn_tp cores)
    //   Phase 4:  RoPE on Q, K (in-place, use cos_/sin_)
    //   Phase 5:  insert_kcache/insert_vcache at absolute position
    //   Phase 6:  SDPA (causal or 2D mask)
    //   Phase 7:  O_proj (row partition, attn_tp cores) -> "oproj"
    //   Phase 8:  all_reduce_sum_residual: oproj + residual1 -> residual2
    //   Phase 9:  GATED attn residual:  residual2 = attn_gate * (residual2 - residual1) + residual1
    //             (3-step: SUB full-size, 1xC->NxC MUL with attn_gate slice, ADD full-size)
    //   Phase 10: post-norm with (1+scale_mlp): residual2 -> residual1
    //             1xC->NxC ADD shift_mlp in-place
    //   Phase 11: MLP (gate/up/GELU/mul/down) + reduce -> residual1 = reduce(down) + residual2
    //   Phase 13: GATED MLP residual: residual1 = mlp_gate * (residual1 - residual2) + residual2
    //   Phase 14: output DMA (unless ctx().output_to_spm)
    //
    // Pitfall 3 structural fix: SDPA + KV-insert use addr_offset(name).value
    // (typed SpmOffset, compile-time distinct from absolute addr() uint32_t).
    // ========================================================================
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        if (num_cores() != 8 && layer_idx == 0 && chunk.idx == 0 &&
            ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                ADARMS_CORE_PROFILE_SITE, 1, 0, core_profile_arguments());
        }
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t cos_sin_start = ctx().position + chunk.offset;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int tp = attn_tp();
        int64_t num_elems_full = seq_len * h;
        bool is_first_layer_first_chunk =
            (layer_idx == 0) && (chunk.idx == 0) && !cond_loaded_this_forward_;
        bool is_chunk_zero = (chunk.idx == 0);

        // --------------------------------------------------------------------
        // Phase 0: cond DMA (once per forward)
        // SCATTER cond_ref_ [hidden_size] fp16 -> "cond" so that each core c
        // holds cond[c*H/NUM_CORES : (c+1)*H/NUM_CORES] at its local SPM
        // offset. This matches the row-partition GEMM used in
        // adarms_gemv_to_spm (each core reads its own local_k=H/NUM_CORES
        // slice of cond at the same local offset).
        //
        // On REPLAY the DDR pointer is cursor-patched to the new cond tensor.
        // --------------------------------------------------------------------
        if (is_first_layer_first_chunk) {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(
                    FmbRouteFamily::MUTABLE_DMA, ADARMS_COND_DMA_SITE,
                    static_cast<int64_t>(
                        AdarmsMutableDmaRoute::DDR_SCATTER_TO_SPM),
                    /*resolved_flags=*/0);
            }
            int64_t local_k           = h / condition_tp();
            int64_t core_stride_bytes = local_k * DWIDTH;
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &cond_src_base_,
                /*src_offset_bytes=*/0,
                /*elements_per_core=*/local_k,
                /*core_stride_bytes=*/core_stride_bytes,
                addr(0, "cond"),
                /*num_cores=*/condition_tp());
            cond_loaded_this_forward_ = true;
        }

        // --------------------------------------------------------------------
        // Phase A: per-layer GEMV at chunk.idx == 0
        // Computes [scale | shift | gate] from cond * dense_w + bias, with
        // +1.0 baked into the scale slice. Result persists in SPM for this
        // layer's remaining chunks.
        // --------------------------------------------------------------------
        if (is_chunk_zero) {
            adarms_gemv_to_spm(addr(0, "cond"),
                               lw.attn_dense_w, lw.attn_dense_b,
                               addr(0, "attn_gemv"),
                               addr(0, "bias_temp"),
                               addr(0, "gemv_partial"));
            adarms_gemv_to_spm(addr(0, "cond"),
                               lw.mlp_dense_w,  lw.mlp_dense_b,
                               addr(0, "mlp_gemv"),
                               addr(0, "bias_temp"),
                               addr(0, "gemv_partial"));
        }

        // GEMV slice offsets inside attn_gemv / mlp_gemv (computed from core 0
        // base address; broadcast DMA means every core shares the same offsets).
        uint32_t attn_scale = addr(0, "attn_gemv");
        uint32_t attn_shift = attn_scale + (uint32_t)(h * DWIDTH);
        uint32_t attn_gate  = attn_scale + (uint32_t)(h * DWIDTH * 2);
        uint32_t mlp_scale  = addr(0, "mlp_gemv");
        uint32_t mlp_shift  = mlp_scale + (uint32_t)(h * DWIDTH);
        uint32_t mlp_gate   = mlp_scale + (uint32_t)(h * DWIDTH * 2);

        // --------------------------------------------------------------------
        // Phase 1: layer input DMA -> "residual1"
        // Skip if previous layer left residual1 in SPM (SPM_RESIDENT mode).
        // --------------------------------------------------------------------
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);  // writes to "residual1"
        }

        // --------------------------------------------------------------------
        // Phase 2: AdaRMSNorm (attn side): residual1 -> input_norm
        // (1+scale) was applied during GEMV, so RMSNorm uses attn_scale directly.
        // Then add shift (broadcast 1xC onto NxC rows).
        // --------------------------------------------------------------------
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            attn_scale,
            seq_len, h, eps_, RpuRmsNormSpmRoute::BASE, num_cores());
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            attn_shift, addr(0, "input_norm"), addr(0, "input_norm"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false, num_cores());

        // --------------------------------------------------------------------
        // Phase 3: QKV Linear (attn_tp cores, col partition)
        // --------------------------------------------------------------------
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, ADARMS_Q_LINEAR_SITE,
                static_cast<int64_t>(nvfp4_.enabled() ? (linear_acc32_ ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                                                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16) : FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        if (nvfp4_.enabled()) {
            rpu_launch_pi05_nvfp4_v2_kernel(addr(0, "input_norm"), lw.q_w, addr(0, "q"),
                seq_len, nq * hd, h, 1, tp, lw.q_ws,
                addr(0, Pi05Nvfp4Tables::names[0]), layer_idx, linear_acc32_);
        } else {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h,
            /*partition=*/1, tp, /*bias_spm_addr=*/0, lw.q_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        }
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, ADARMS_K_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp, /*bias_spm_addr=*/0, lw.k_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, ADARMS_V_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp, /*bias_spm_addr=*/0, lw.v_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);

        // --------------------------------------------------------------------
        // Phase 4: RoPE (in-place, no QK RMSNorm for AdaRMS Gemma)
        // --------------------------------------------------------------------
        int64_t local_kv_heads = nkv / tp;
        c10::Half* cos_ptr = cos_.data_ptr<c10::Half>();
        c10::Half* sin_ptr = sin_.data_ptr<c10::Half>();
        // RoPE reads the LOGICAL position, the KV insert below the physical row
        // (see rope_position_). Equal only when the prefix has no pad rows.
        const int64_t rope_start = (rope_position_ >= 0)
            ? rope_position_ + chunk.offset : cos_sin_start;
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, ADARMS_Q_ROPE_SITE,
                static_cast<int64_t>(AdarmsRopeRoute::ROPE_1D),
                /*resolved_flags=*/0, {rope_start}, chunk.idx);
        }
        rpu_launch_rope_spm_kernel(
            addr(0, "q"), addr(0, "q"),
            cos_ptr, sin_ptr,
            seq_len, local_q_heads_, hd, rope_start, tp);
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, ADARMS_K_ROPE_SITE,
                static_cast<int64_t>(AdarmsRopeRoute::ROPE_1D),
                /*resolved_flags=*/0, {rope_start}, chunk.idx);
        }
        rpu_launch_rope_spm_kernel(
            addr(0, "k"), addr(0, "k"),
            cos_ptr, sin_ptr,
            seq_len, local_kv_heads, hd, rope_start, tp);

        // --------------------------------------------------------------------
        // Phase 5: KV cache insert at absolute position.
        //
        // Pitfall 3 structural fix: KV-insert takes SPM OFFSETS, not absolute
        // addresses. addr_offset("name").value is compile-time distinct from
        // addr(core, "name").
        // --------------------------------------------------------------------
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        KvInsertSegmentPlan kv_plan = [&] {
            if (!ctx().has_complete_physical_manifest()) {
                return rpu_resolve_kvinsert_segment_plan_auto(
                    cos_sin_start, seq_len, seq_len, tp, nkv, hd,
                    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16);
            }
            const auto& route = ctx().find_physical_route(
                FmbRouteFamily::KV_INSERT, ADARMS_KV_INSERT_SITE,
                chunk.idx);
            KvInsertSegmentPlan frozen =
                restore_kvinsert_plan(
                    ADARMS_KV_INSERT_SITE, route.arguments, tp, nkv, hd);
            TORCH_CHECK(
                frozen.logical_rows() == seq_len &&
                    frozen.physical_rows() == seq_len &&
                    frozen.segment(0).position == cos_sin_start,
                "AdaRMS KV descriptor geometry drift");
            ctx().consume_physical_route(
                FmbRouteFamily::KV_INSERT, ADARMS_KV_INSERT_SITE,
                static_cast<int64_t>(frozen.route()),
                /*resolved_flags=*/0, route.arguments, chunk.idx);
            return frozen;
        }();
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache,
            addr_offset("k").value, addr_offset("v").value,
            nkv, hd, tp,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            /*spm_rows=*/0, kv_plan);

        // --------------------------------------------------------------------
        // Phase 6: SDPA (causal or 2D mask).
        //
        // Pitfall 3 structural fix: SDPA args (q / output / sdpa_tmp /
        // sdpa_mask) take SPM offsets via addr_offset(...).value.
        // --------------------------------------------------------------------
        int64_t kv_seq_len = ctx().position + ctx().seq_len;
        // PERF L1 (2026-05-07): mask was prepare()'d once at forward() entry.
        // Per-layer Phase 6 only DMAs the prepared DDR tensor → SPM.
        // mask_type was determined at prepare time (MASK_NONE/MASK_LTM/MASK_2D).
        sdpa_dma_mask_to_spm(prepared_mask_,
                             addr_offset("sdpa_mask").value,
                             seq_len, kv_seq_len, tp);
        int mask_type = prepared_mask_.mask_type;
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_attention_route(
                ADARMS_ATTENTION_SITE,
                AttentionExecutionPolicy::DDR_KV, chunk.idx);
        }
        rpu_launch_sdpa_spm_unified_kernel_v2(
            k_cache, v_cache, mask_type, c10::nullopt,
            addr_offset("q").value,
            addr_offset("output").value,
            addr_offset("sdpa_tmp").value,
            addr_offset("sdpa_mask").value,
            seq_len, nq, nkv, hd,
            kv_seq_len, tp, /*physical_kv_cores=*/8);

        // --------------------------------------------------------------------
        // Phase 7: O_proj (row partition, attn_tp cores)
        // --------------------------------------------------------------------
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                ADARMS_PREPARE_ALL_REDUCE_SITE,
                static_cast<int64_t>(
                    AdarmsAllReduceRoute::PREPARE_RING_INPUT),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        rpu_prepare_ring_all_reduce_input(addr(0, "oproj"), seq_len, h, tp, num_cores());
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, ADARMS_O_LINEAR_SITE,
                static_cast<int64_t>(nvfp4_.enabled() ? (linear_acc32_ ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                                                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16) : FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        if (nvfp4_.enabled()) {
            rpu_launch_pi05_nvfp4_v2_kernel(addr(0, "output"), lw.o_w, addr(0, "oproj"),
                seq_len, h, nq * hd, 0, tp, lw.o_ws,
                addr(0, Pi05Nvfp4Tables::names[1]), layer_idx, linear_acc32_);
        } else {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "output"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd,
            /*partition=*/0, tp, /*bias_spm_addr=*/0, lw.o_ws, 0, 0,
            /*force_acc32=*/linear_acc32_);
        }

        // --------------------------------------------------------------------
        // Phase 8: all_reduce_sum_residual (attn)
        // residual2 = reduce_sum(oproj, attn_tp cores) + residual1
        // Output goes to all NUM_CORES.
        // --------------------------------------------------------------------
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                ADARMS_ATTENTION_ALL_REDUCE_SITE,
                adarms_ring_route(seq_len, h, num_cores()),
                /*resolved_flags=*/0, {}, chunk.idx);
        }
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
            seq_len, h, tp, num_cores());

        // --------------------------------------------------------------------
        // Phase 9: GATED attn residual
        // residual2 = attn_gate * (residual2 - residual1) + residual1
        // Implemented as 3-step: SUB full-size, 1xC->NxC MUL gate slice, ADD full-size
        // --------------------------------------------------------------------
        rpu_launch_eltwise_binary_spm_kernel(          // residual2 -= residual1
            addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
            num_elems_full, ValuOpType::SUB, c10::Half(1.0), num_cores());
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(  // residual2 *= attn_gate (1xC broadcast)
            attn_gate, addr(0, "residual2"), addr(0, "residual2"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false, num_cores());
        rpu_launch_eltwise_binary_spm_kernel(          // residual2 += residual1
            addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0), num_cores());

        // --------------------------------------------------------------------
        // Phase 10: Post-attention AdaRMSNorm (MLP side): residual2 -> residual1
        // (1+scale_mlp) baked into mlp_scale; then add shift_mlp.
        // --------------------------------------------------------------------
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual2"), addr(0, "residual1"),
            mlp_scale,
            seq_len, h, eps_, RpuRmsNormSpmRoute::BASE, num_cores());
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            mlp_shift, addr(0, "residual1"), addr(0, "residual1"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false, num_cores());

        // --------------------------------------------------------------------
        // Phase 11: MLP pipeline
        // --------------------------------------------------------------------
        emit_mlp_pipeline(lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w,
                          seq_len, ActivationKind::GELU,
                          lw.gate_ws, lw.up_ws, lw.down_ws,
                          nvfp4_.enabled() ? addr(0, "nvfp4_gate_ts") : 0,
                          nvfp4_.enabled() ? addr(0, "nvfp4_up_ts") : 0,
                          nvfp4_.enabled() ? addr(0, "nvfp4_down_ts") : 0, layer_idx,
                          false, false, /*acc32=*/linear_acc32_, false, false,
                          /*bind_silu_mul_route=*/false,
                          /*residual_spm_addr=*/0, /*pi05_xor3=*/false, nvfp4_.enabled());

        // --------------------------------------------------------------------
        // Phase 13: GATED MLP residual
        // residual1 = mlp_gate * (residual1 - residual2) + residual2
        // After emit_mlp_pipeline: residual1 = reduce(down) + residual2
        // So (residual1 - residual2) = reduce(down), then gate it, then add back residual2.
        // --------------------------------------------------------------------
        rpu_launch_eltwise_binary_spm_kernel(          // residual1 -= residual2
            addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
            num_elems_full, ValuOpType::SUB, c10::Half(1.0), num_cores());
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(  // residual1 *= mlp_gate (1xC broadcast)
            mlp_gate, addr(0, "residual1"), addr(0, "residual1"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false, num_cores());
        rpu_launch_eltwise_binary_spm_kernel(          // residual1 += residual2
            addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0), num_cores());

        // --------------------------------------------------------------------
        // Phase 14: Output DMA (unless output stays in SPM for next layer)
        // --------------------------------------------------------------------
        if (!ctx().output_to_spm) {
            emit_layer_output_dma(layer_idx, chunk);  // from "residual1"
        }
    }

private:
    SdpaConfig make_sdpa_config(int mask = 1) const {
        return {sdpa_kernel_, head_dim(), num_q_heads(), num_kv_heads(),
                attn_tp(), mask};
    }

    // ========================================================================
    // GEMV helper: row-partition 8-core + fused all_reduce+bias.
    // Synced from QwenPI05Model after the row-partition rework on 2026-04-15.
    //
    // Weight is the natural [3H, H] tensor row-partition swizzled so each core
    // holds K/NUM_CORES=H/NUM_CORES columns of all 3H output rows. Layout of
    // the final output (on every core after reduce):
    //   [scale(hidden) | shift(hidden) | gate(hidden)]
    // The scale portion gets +1.0 in place so later RMSNorm sees (1+scale)
    // directly.
    //
    // Three distinct SPM buffers are required (all_reduce_sum_residual expects
    // input, residual, and output to be non-overlapping SPM regions):
    //   gemv_out_spm_addr    : final reduced+biased [3H] (broadcast to all cores)
    //   bias_temp_spm_addr   : DMA'd bias, used as the reduce-residual [3H]
    //   partial_spm_addr     : per-core partial [3H] from the row-partition GEMM
    // ========================================================================
    void adarms_gemv_to_spm(uint32_t cond_spm_addr,
                            const at::Tensor& dense_w,
                            const at::Tensor& dense_b,
                            uint32_t gemv_out_spm_addr,
                            uint32_t bias_temp_spm_addr,
                            uint32_t partial_spm_addr)
    {
        int64_t h = hidden_size();
        int64_t out_features = 3 * h;

        // Step 1: DMA bias -> bias_temp (replicated across all 8 cores, acts
        // as the residual input to the fused all_reduce+bias kernel).
        rpu_launch_ddr_broadcast_spm_dma(
            dense_b.data_ptr<c10::Half>(), out_features, bias_temp_spm_addr, num_cores());
        if (condition_tp() != num_cores()) {
            if (ctx().has_complete_physical_manifest()) {
                ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
                    ADARMS_COND_PREPARE_SITE, 3, 0,
                    {condition_tp(), num_cores(), 1});
            }
            rpu_prepare_ring_all_reduce_input(partial_spm_addr, 1,
                out_features, condition_tp(), num_cores());
        }
        // Step 2: Row-partition GEMV. partial_spm_addr holds the per-core
        // partial [3H] along the local_k = H/NUM_CORES split.
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, ADARMS_GEMV_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, /*num_cores=*/condition_tp(),
            /*bias_spm_addr=*/0, /*scale=*/at::Tensor(), 0, 0,
            /*force_acc32=*/linear_acc32_);
        // Step 3: Fused all_reduce + bias. reduce(partial) + bias -> gemv_out
        // broadcast to all output cores. Input and output MUST be distinct
        // SPM regions.
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                ADARMS_GEMV_ALL_REDUCE_SITE,
                adarms_ring_route(/*rows=*/1, out_features, num_cores()),
                /*resolved_flags=*/0);
        }
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
            /*M=*/1, /*N=*/out_features,
            /*input_num_cores=*/condition_tp(), /*output_num_cores=*/num_cores());
        // Step 4: (1+scale) -- add +1.0 only to the first hidden_size elements
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0),
            gemv_out_spm_addr, h, ValuOpType::ADD, num_cores());
    }

    int64_t logical_intermediate_size_ = 0;
    bool reduced_w8a16_ = false;

    // ----- Model state -----
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.gate_proj_w, &weights.up_proj_w, &weights.down_proj_w, &weights.q_ws,
                    &weights.k_ws, &weights.v_ws, &weights.o_ws, &weights.gate_ws,
                    &weights.up_ws, &weights.down_ws, &weights.attn_dense_w, &weights.attn_dense_b,
                    &weights.mlp_dense_w, &weights.mlp_dense_b}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        if (linear_acc32_) identity.insert(identity.end(), {0x4143433332LL, 1});
        return identity;
    }

    Pi05Nvfp4Tables nvfp4_;
    std::vector<LayerWeights> layer_weights_;
    at::Tensor cos_, sin_;
    // Logical RoPE start for the action suffix, or -1 to use the physical row.
    //
    // `cos_sin_start` is one number doing two jobs: the RoPE table index and the
    // KV cache write row. The suffix must be WRITTEN at the physical row
    // (runtime.py resets the shared cache to prefix_pad_masks.shape[1], so it
    // lands past the prefix), but its RoPE position is the LOGICAL one the
    // reference uses: sum(prefix_pad_masks). Those differ by exactly the number
    // of pad rows in the prefix. The suffix itself never has pad rows
    // (suffix_pad_masks is all ones), so the positions are a contiguous range
    // and a scalar start is enough — no gathered table, unlike the prefix.
    //
    // Baked into the graph at BUILD, so callers MUST put it in the signature.
    int64_t rope_position_ = -1;
    double eps_ = 1e-6;

    // Derived dims (computed in set_weights, cached for hot path)
    int64_t local_q_heads_ = 0;
    int64_t local_kv_dim_  = 0;

    // Per-forward cond state (reset on every forward() entry).
    // cond_ref_ keeps the tensor alive until run_all_layers returns (the
    // batch end is synchronous inside run_all_layers); the DDR pointer is
    // cursor-patched per REPLAY via cond_src_base_ + mutable DMA.
    at::Tensor cond_ref_;
    uint64_t   cond_src_base_ = 0;
    bool cond_loaded_this_forward_ = false;

    // Per-forward prepared SDPA mask (PERF L1: cross-layer hoist, 2026-05-07).
    // sdpa_prepare_mask() does dim-reduce + fp16-cast + constant_pad_nd +
    // DDR copy; results invariant across all 18 expert layers within one
    // forward(). Hoisting drops 18 redundant constant_pad_nd + DDR-copy calls
    // per denoise step (90 → 5 across a 5-step Pi0.5 inference).
    // sdpa_dma_mask_to_spm() in the per-layer Phase 6 reads from this member;
    // member assignment also keeps DDR alive across the synchronous forward().
    PreparedMask prepared_mask_;

    // PERF L2 (2026-05-07): cross-forward cache key for prepared_mask_.
    // Pi0.5 denoise hoists mask construction outside the 5-step loop
    // (runtime.py), so steps 2-5 pass the SAME at::Tensor (same data_ptr).
    // forward() compares (input_data_ptr, seq_q, seq_k, is_causal) against
    // this cache; on hit, skips sdpa_prepare_mask entirely (5 → 1 per
    // inference). prepared_mask_input_ref_ is a strong ref to the caller's
    // last input tensor — keeps its storage pinned so a different tensor
    // cannot reuse the same data_ptr while our cache is valid.
    at::Tensor prepared_mask_input_ref_;
    int64_t    prepared_mask_seq_q_     = -1;
    int64_t    prepared_mask_seq_k_     = -1;
    bool       prepared_mask_is_causal_ = false;

    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;
};

}  // namespace v3

// =============================================================================
// Instance registry -- ModelHandleRegistry<v3::AdaRMSModel>.
// Handles are monotonic and are not reused; the Python adapter owns graph
// admission independently from this registry.
// =============================================================================

using AdaRMSRegistry = ModelHandleRegistry<v3::AdaRMSModel>;

std::vector<int64_t> rpu_adarms_planner_cache_identity(int64_t handle) {
    return AdaRMSRegistry::get(handle, "rpu_adarms_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_adarms_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    AdaRMSRegistry::get(handle, "rpu_adarms_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_adarms_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return AdaRMSRegistry::get(handle, "rpu_adarms_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_adarms_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return AdaRMSRegistry::get(handle, "rpu_adarms_kvinsert_cost_domain")
        ->kvinsert_cost_domain("adarms", descriptor);
}

std::string rpu_adarms_kvinsert_cost_catalog_sha256(int64_t handle) {
    return AdaRMSRegistry::get(
        handle, "rpu_adarms_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

void rpu_adarms_set_execution_core_count(int64_t handle, int64_t cores) {
    AdaRMSRegistry::get(handle, "rpu_adarms_set_execution_core_count")
        ->set_execution_cores(cores);
}

std::vector<int64_t> rpu_adarms_get_execution_topology(int64_t handle) {
    return AdaRMSRegistry::get(handle, "rpu_adarms_get_execution_topology")
        ->execution_topology();
}

int64_t rpu_adarms_create(bool linear_acc32) {
    return AdaRMSRegistry::create(linear_acc32);
}

void rpu_adarms_destroy(int64_t handle) {
    AdaRMSRegistry::destroy(handle, "rpu_adarms_destroy");
}

void rpu_adarms_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    AdaRMSRegistry::get(handle, "rpu_adarms_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_adarms_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list, const std::optional<std::vector<at::Tensor>>& nvfp4_tensor_scales)
{
    AdaRMSRegistry::get(handle, "rpu_adarms")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_list, up_list, down_list,
        attn_dense_w_list, attn_dense_b_list,
        mlp_dense_w_list,  mlp_dense_b_list,
        cos, sin,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps,
        q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list, nvfp4_tensor_scales.value_or(std::vector<at::Tensor>{}));
}

void rpu_adarms_set_rope_position(int64_t handle, int64_t position) {
    AdaRMSRegistry::get(handle, "rpu_adarms_set_rope_position")
        ->set_rope_position(position);
}

at::Tensor rpu_adarms_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    at::IntArrayRef planned_stage_descriptor)
{
    // TensorList -> std::vector<at::Tensor> (shallow copy; tensors are refcounted)
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());

    return AdaRMSRegistry::get(handle, "rpu_adarms")->forward(
        hidden_states, cond, k_caches, v_caches,
        attention_mask, position, is_causal,
        planned_stage_descriptor);
}

std::vector<int64_t> rpu_adarms_resolve_action_stage_domain(
    int64_t handle, int64_t execution_len, int64_t logical_len,
    int64_t position, int64_t kv_len, bool use_attention_mask,
    bool is_causal, int64_t requested_chunk_size) {
    return AdaRMSRegistry::get(
               handle, "rpu_adarms_resolve_action_stage_domain")
        ->resolve_action_stage_domain(
            execution_len, logical_len, position, kv_len,
            use_attention_mask, is_causal, requested_chunk_size);
}

int64_t rpu_adarms_get_resolved_chunk_size(int64_t handle) {
    return AdaRMSRegistry::get(
               handle, "rpu_adarms_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}
