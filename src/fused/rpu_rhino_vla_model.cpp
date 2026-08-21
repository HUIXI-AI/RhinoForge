// rpu_rhino_vla_model.cpp — RhinoVLA Action Expert all-layers-once model
//                           (v3 FusedModelBase port)
//
// RhinoVLA's action expert shares QwenPI05's math/shape family (depth-N AdaRMS +
// q_norm/k_norm + M-RoPE + prefix-KV cross-attn + SwiGLU + tanh-gated residual),
// while remaining a separate
// op (torch.ops.rpu.rhino_vla_*) so RhinoVLA (72-D / depth-18) can be optimized
// independently of QwenPI05 — changes here do NOT touch QwenPI05 and vice versa.
// Renamed SPM buffer labels also avoid allocator name-collision if both load.
// RhinoVLA-specific bits (72-D IO, mask_condition, instance-LoRA merge, 0->1
// denoise) live in the Python adapter (python/rpu_backend/adapters/rhinovla/);
// LoRA is merged into base weights BEFORE set_weights, so this op never sees it.
//
// RhinoVLA is structurally the closest analog of AdaRMS (flat-phase,
// SEQUENTIAL, per-forward cond + per-layer GEMV) PLUS q_norm/k_norm per-layer
// (Qwen3 pattern), reusing the existing AdaRMS flat-phase and q_norm/k_norm DMA
// machinery.
//
// Key differences vs AdaRMS:
//   - QK norm: per-head RMSNorm with [head_dim] weights (from Qwen3)
//   - RoPE: cos/sin passed per-forward (M-RoPE pre-computed on Python side)
//   - MLP activation: SiLU (SwiGLU) instead of GELU (GeGLU)
//   - Prefix KV: non-causal attention with VLM prefix cache; kv_insert_pos
//     = ctx().position + chunk.offset
//   - RoPE uses a logical prefix position; KV insert uses the physical row
//   - Gated residual = SUB -> TANH -> 1xC->NxC MUL -> ADD (4 steps,
//     vs AdaRMS 3-step without TANH)
//
// Framework integration contracts:
//   - Inherit from v3::FusedModelBase in `namespace v3`.
//   - SDPA + KV-insert use typed SpmOffset values from `addr_offset(name).value`.
//   - Access model params via pimpl getters.
//   - Class-member cond_ref_/cos_ref_/sin_ref_ keep per-forward tensors alive.
//   - set_weights ends with `invalidate_model_state();` as its last non-empty
//     statement.
//
// Runtime scope:
//   - Per-forward cond DMA (layer 0 / chunk 0 gate) via cond_loaded_this_forward_.
//   - Per-layer GEMV (cond * dense_w + bias -> [scale|shift|gate]) at
//     chunk.idx == 0, result persists in SPM across chunks of this layer.
//   - (1+scale) preprocessing via scalar ADD to the live GEMV output SPM buffer.
//   - Gated residual 4-step: SUB full-size -> TANH gate slice -> 1xC->NxC MUL
//     with gate slice -> ADD full-size.
//   - Per-layer q_norm_w/k_norm_w DMA + RMSNorm before RoPE.
//   - Always SEQUENTIAL (no KV_FIRST); flat phases (all LayerWide).
//   - cfg.cross_layer_batch_size = num_layers() (force single group).
//   - Debug export flush machinery for per-core SPM dumps.
//
// Design contract: FusedModelBase.

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_helpers.h"
#include "rpu_spm_allocator.h"
#include "rpu_runtime_state.h"  // shared runtime state and debug tensor registry

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace v3 {

static bool rhino_vla_denoise_static_context_cache_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_RHINOVLA_DENOISE_STATIC_CONTEXT_CACHE");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

static bool rhino_vla_fused_adarms_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_RHINOVLA_FUSED_ADARMS_GEMV");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

static bool rhino_vla_skip_adarms_gemv_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_RHINOVLA_SKIP_ADARMS_GEMV");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

static bool rhino_vla_precompute_adarms_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_RHINOVLA_PRECOMPUTE_ADARMS");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

static bool rhino_vla_precompute_time_proj_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_RHINOVLA_PRECOMPUTE_TIME_PROJ");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

static bool rhino_vla_fold_action_time_in_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_RHINOVLA_FOLD_ACTION_TIME_IN");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

// Small-shape fusion: drop the redundant SUB in the tanh-gated
// residual by feeding the all-reduce a zeroed residual (add 0) instead of the
// real residual + subtracting it back. Net -1 binary kernel per gated site.
static bool rhino_vla_gated_no_sub_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_RHINOVLA_GATED_NO_SUB");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

// Small-shape fusion: fuse the MLP SiLU(gate) + (gate*up) into the
// single `llama_silu_mul` operator. Net -1 unary kernel per MLP.
static bool rhino_vla_fused_silu_mul_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_RHINOVLA_FUSED_SILU_MUL");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

// =============================================================================
// RhinoVLAModel — v3::FusedModelBase subclass (RhinoVLA Action Expert)
// =============================================================================

class RhinoVLAModel : public FusedModelBase {
public:
    struct LayerWeights {
        // Attention
        at::Tensor q_w, k_w, v_w, o_w;
        // MLP
        at::Tensor gate_proj_w, up_proj_w, down_proj_w;
        // AdaRMS GEMV (per-layer)
        at::Tensor attn_dense_w;    // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor attn_dense_b;    // [3*hidden] fp16
        at::Tensor mlp_dense_w;     // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor mlp_dense_b;     // [3*hidden] fp16
        at::Tensor pair_dense_w;    // [6*hidden, hidden] fp16, fused attn+mlp cond projection
        at::Tensor pair_dense_b;    // [6*hidden] fp16
        // QK norm (per-layer, new vs AdaRMS)
        at::Tensor q_norm_w;        // [head_dim] fp16
        at::Tensor k_norm_w;        // [head_dim] fp16
    };

    RhinoVLAModel() = default;

    void set_fast_replay_skip_layer_loop(bool enabled) {
        fast_replay_skip_layer_loop_ = enabled;
    }

    void set_rope_position(int64_t position) {
        TORCH_CHECK(position >= 0,
                    "RhinoVLA logical RoPE position must be non-negative");
        rope_position_ = position;
    }

    void set_denoise_loop_weights(
        const at::Tensor& action_in_w, const at::Tensor& action_in_b,
        const at::Tensor& action_time_in_w, const at::Tensor& action_time_in_b,
        const at::Tensor& time_in_action_w, const at::Tensor& time_in_time_w,
        const at::Tensor& time_in_b,
        const at::Tensor& time_out_w, const at::Tensor& time_out_b,
        const at::Tensor& state_w, const at::Tensor& state_b,
        const at::Tensor& state_mask_w, const at::Tensor& state_mask_b,
        const at::Tensor& action_mask_w, const at::Tensor& action_mask_b,
        const at::Tensor& final_norm_w, const at::Tensor& final_norm_b,
        const at::Tensor& action_out_w, const at::Tensor& action_out_b,
        const at::Tensor& cos, const at::Tensor& sin,
        int64_t action_dim, int64_t action_dim_pad,
        int64_t state_dim, int64_t state_dim_pad,
        int64_t action_horizon, int64_t suffix_len) {
        TORCH_CHECK(num_layers() > 0,
                    "RhinoVLAModel::set_denoise_loop_weights called before set_weights");
        TORCH_CHECK(action_dim > 0 && action_dim_pad >= action_dim &&
                    state_dim > 0 && state_dim_pad >= state_dim &&
                    action_horizon > 0 && suffix_len == action_horizon + 1,
                    "RhinoVLAModel::set_denoise_loop_weights: invalid dims");
        auto check_rpu = [](const at::Tensor& t, const char* name) {
            TORCH_CHECK(t.defined() && t.device().type() == at::kPrivateUse1 &&
                        t.scalar_type() == at::kHalf && t.is_contiguous(),
                        "RhinoVLAModel::set_denoise_loop_weights: ", name,
                        " must be contiguous fp16 RPU tensor");
        };
        check_rpu(action_in_w, "action_in_w");
        check_rpu(action_in_b, "action_in_b");
        check_rpu(action_time_in_w, "action_time_in_w");
        check_rpu(action_time_in_b, "action_time_in_b");
        check_rpu(time_in_action_w, "time_in_action_w");
        check_rpu(time_in_time_w, "time_in_time_w");
        check_rpu(time_in_b, "time_in_b");
        check_rpu(time_out_w, "time_out_w");
        check_rpu(time_out_b, "time_out_b");
        check_rpu(state_w, "state_w");
        check_rpu(state_b, "state_b");
        check_rpu(state_mask_w, "state_mask_w");
        check_rpu(state_mask_b, "state_mask_b");
        check_rpu(action_mask_w, "action_mask_w");
        check_rpu(action_mask_b, "action_mask_b");
        check_rpu(final_norm_w, "final_norm_w");
        check_rpu(final_norm_b, "final_norm_b");
        check_rpu(action_out_w, "action_out_w");
        check_rpu(action_out_b, "action_out_b");
        check_rpu(cos, "cos");
        check_rpu(sin, "sin");
        // Shape contract (mirrors forward()): build_layer_subgraph reads cos/sin as
        // a raw [max_pos, head_dim/2] table. A full-dim or short table silently
        // misreads / runs off the end — validate here, not at first replay.
        TORCH_CHECK(cos.sizes() == sin.sizes(),
                    "RhinoVLAModel::set_denoise_loop_weights: cos/sin shape mismatch: cos=",
                    cos.sizes(), " sin=", sin.sizes());
        TORCH_CHECK(cos.dim() == 2 && cos.size(1) == head_dim() / 2,
                    "RhinoVLAModel::set_denoise_loop_weights: cos/sin must be 2D "
                    "[max_pos, head_dim/2=", head_dim() / 2, "]; got ", cos.sizes());
        TORCH_CHECK(cos.size(0) >= suffix_len,
                    "RhinoVLAModel::set_denoise_loop_weights: cos/sin rows ", cos.size(0),
                    " < suffix_len ", suffix_len, " — the denoise RoPE kernel reads "
                    "[0, suffix_len) and would run past the table end");

        action_in_w_ = action_in_w;
        action_in_b_ = action_in_b;
        action_time_in_w_ = action_time_in_w;
        action_time_in_b_ = action_time_in_b;
        time_in_action_w_ = time_in_action_w;
        time_in_time_w_ = time_in_time_w;
        time_in_b_ = time_in_b;
        time_out_w_ = time_out_w;
        time_out_b_ = time_out_b;
        state_w_ = state_w;
        state_b_ = state_b;
        state_mask_w_ = state_mask_w;
        state_mask_b_ = state_mask_b;
        action_mask_w_ = action_mask_w;
        action_mask_b_ = action_mask_b;
        final_norm_w_ = final_norm_w;
        final_norm_b_ = final_norm_b;
        action_out_w_ = action_out_w;
        action_out_b_ = action_out_b;
        cos_loop_ = cos;
        sin_loop_ = sin;
        action_dim_ = action_dim;
        action_dim_pad_ = action_dim_pad;
        state_dim_ = state_dim;
        state_dim_pad_ = state_dim_pad;
        action_horizon_ = action_horizon;
        suffix_len_ = suffix_len;
        action_emb_stage_ = at::empty(
            {1, suffix_len_, hidden_size()},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        denoise_loop_weights_ready_ = true;
        invalidate_model_state();
    }

    void set_denoise_loop_adarms_tables(
        const at::Tensor& pair_table,
        const at::Tensor& final_table) {
        TORCH_CHECK(num_layers() > 0,
                    "RhinoVLAModel::set_denoise_loop_adarms_tables before set_weights");
        auto check_rpu = [](const at::Tensor& t, const char* name) {
            TORCH_CHECK(t.defined() && t.device().type() == at::kPrivateUse1 &&
                        t.scalar_type() == at::kHalf && t.is_contiguous(),
                        "RhinoVLAModel::set_denoise_loop_adarms_tables: ", name,
                        " must be contiguous fp16 RPU tensor");
        };
        check_rpu(pair_table, "pair_table");
        check_rpu(final_table, "final_table");
        TORCH_CHECK(pair_table.dim() == 3,
                    "RhinoVLA AdaRMS pair_table must be [steps,layers,6H], got ",
                    pair_table.sizes());
        TORCH_CHECK(final_table.dim() == 2,
                    "RhinoVLA AdaRMS final_table must be [steps,3H], got ",
                    final_table.sizes());
        TORCH_CHECK(pair_table.size(0) == final_table.size(0),
                    "RhinoVLA AdaRMS table steps mismatch pair=", pair_table.size(0),
                    " final=", final_table.size(0));
        TORCH_CHECK(pair_table.size(1) == num_layers(),
                    "RhinoVLA AdaRMS pair_table layer count ", pair_table.size(1),
                    " != num_layers=", num_layers());
        TORCH_CHECK(pair_table.size(2) == 6 * hidden_size(),
                    "RhinoVLA AdaRMS pair_table last dim ", pair_table.size(2),
                    " != 6H=", 6 * hidden_size());
        TORCH_CHECK(final_table.size(1) == 3 * hidden_size(),
                    "RhinoVLA AdaRMS final_table last dim ", final_table.size(1),
                    " != 3H=", 3 * hidden_size());

        adarms_pair_table_ = pair_table;
        adarms_final_table_ = final_table;
        adarms_pair_table_src_base_ =
            ::rhino_lkn::RpuGetDevAddr(adarms_pair_table_.data_ptr());
        adarms_final_table_src_base_ =
            ::rhino_lkn::RpuGetDevAddr(adarms_final_table_.data_ptr());
        rpu_ddr_flush_force(adarms_pair_table_.data_ptr<c10::Half>());
        rpu_ddr_flush_force(adarms_final_table_.data_ptr<c10::Half>());
        denoise_adarms_tables_ready_ = true;
        invalidate_model_state();
    }

    void set_denoise_loop_time_proj_table(const at::Tensor& table) {
        TORCH_CHECK(num_layers() > 0,
                    "RhinoVLAModel::set_denoise_loop_time_proj_table before set_weights");
        TORCH_CHECK(table.defined() && table.device().type() == at::kPrivateUse1 &&
                    table.scalar_type() == at::kHalf && table.is_contiguous(),
                    "RhinoVLA time-proj table must be contiguous fp16 RPU tensor");
        TORCH_CHECK(table.dim() == 2,
                    "RhinoVLA time-proj table must be [steps,H], got ",
                    table.sizes());
        TORCH_CHECK(table.size(1) == hidden_size(),
                    "RhinoVLA time-proj table last dim ", table.size(1),
                    " != H=", hidden_size());

        time_proj_table_ = table;
        time_proj_table_src_base_ =
            ::rhino_lkn::RpuGetDevAddr(time_proj_table_.data_ptr());
        rpu_ddr_flush_force(time_proj_table_.data_ptr<c10::Half>());
        denoise_time_proj_table_ready_ = true;
        invalidate_model_state();
    }

    // ========================================================================
    // set_weights — accepts per-layer weights including QK norm
    //
    // Unlike AdaRMS, cos/sin are NOT stored here (they are per-forward).
    // QK norm weights (q_norm, k_norm) are per-layer [head_dim] fp16.
    // ========================================================================
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
        at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
        at::TensorList pair_dense_w_list, at::TensorList pair_dense_b_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps)
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "rhino_vla_set_weights: empty weight lists");
        TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0
                    && hidden_size > 0 && intermediate_size > 0,
                    "rhino_vla_set_weights: dim params must be positive");
        TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                    "rhino_vla_set_weights: num_q_heads (", num_q_heads,
                    ") must be divisible by num_kv_heads (", num_kv_heads, ")");

        // All per-layer lists must have same length
        auto check_list = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "rhino_vla_set_weights: ", n, ".size()=", l.size(),
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
        check_list(pair_dense_w_list, "pair_dense_w_list");
        check_list(pair_dense_b_list, "pair_dense_b_list");
        check_list(q_norm_list,       "q_norm_list");
        check_list(k_norm_list,       "k_norm_list");

        // Per-layer defined/rank checks: 2D for linear weights, 1D for bias/norm
        auto check_rank = [&](const at::TensorList& list, const char* name,
                              int64_t expected_rank) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "rhino_vla_set_weights: ", name, "[", i, "] undefined");
                TORCH_CHECK(list[i].dim() == expected_rank,
                            "rhino_vla_set_weights: ", name, "[", i, "] must be ",
                            expected_rank, "D, got ", list[i].dim(), "D");
            }
        };
        check_rank(q_w_list,          "q_w_list",          2);
        check_rank(k_w_list,          "k_w_list",          2);
        check_rank(v_w_list,          "v_w_list",          2);
        check_rank(o_w_list,          "o_w_list",          2);
        check_rank(gate_list,         "gate_list",         2);
        check_rank(up_list,           "up_list",           2);
        check_rank(down_list,         "down_list",         2);
        check_rank(attn_dense_w_list, "attn_dense_w_list", 2);
        check_rank(attn_dense_b_list, "attn_dense_b_list", 1);
        check_rank(mlp_dense_w_list,  "mlp_dense_w_list",  2);
        check_rank(mlp_dense_b_list,  "mlp_dense_b_list",  1);
        check_rank(pair_dense_w_list, "pair_dense_w_list", 2);
        check_rank(pair_dense_b_list, "pair_dense_b_list", 1);
        check_rank(q_norm_list,       "q_norm_list",       1);
        check_rank(k_norm_list,       "k_norm_list",       1);

        // Commit model params (pimpl: hidden/q/kv/head_dim/intermediate + attn_tp_)
        set_model_params(num_q_heads, num_kv_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;

        // Cached derived dims (v3 framework's set_model_params already sets
        // pimpl attn_tp_ = std::min(8, num_kv_heads), matching v2's
        // std::min(NUM_CORES, num_kv_heads) — cache here for readability
        // inside the hot build_layer_subgraph path).
        local_q_heads_ = num_q_heads / attn_tp();
        local_kv_dim_  = num_kv_heads * head_dim / attn_tp();

        // Copy weights into layer state
        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                gate_list[i], up_list[i], down_list[i],
                attn_dense_w_list[i], attn_dense_b_list[i],
                mlp_dense_w_list[i],  mlp_dense_b_list[i],
                pair_dense_w_list[i], pair_dense_b_list[i],
                q_norm_list[i], k_norm_list[i],
            });
        }

        invalidate_model_state();  // Last non-empty statement of set_weights.
    }

    // ========================================================================
    // forward — cond + cos/sin are per-forward arguments
    //
    // Unlike AdaRMS which stores cos/sin in set_weights, RhinoVLA receives
    // cos/sin per-forward (M-RoPE pre-computed on Python side).
    // Three tensors kept alive via class members: cond_ref_, cos_ref_, sin_ref_.
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& hidden_states,
        const at::Tensor& cond,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cos,
        const at::Tensor& sin,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal)
    {
        TORCH_CHECK(num_layers() > 0,
                    "RhinoVLAModel::forward called before set_weights");
        if (loop_mode_) {
            loop_mode_ = false;
            num_steps_ = 1;
            dt_pinned_ = false;
            invalidate_model_state();
        }
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "RhinoVLAModel::forward: hidden_states must be on RPU device");
        TORCH_CHECK(hidden_states.is_contiguous(),
                    "RhinoVLAModel::forward: hidden_states must be contiguous "
                    "(downstream kernels use raw data_ptr())");
        TORCH_CHECK(position >= 0,
                    "RhinoVLAModel::forward: position must be non-negative, got ", position);

        // cond validation (mirrors AdaRMS:224-240 — cross-boundary tensors
        // consumed as raw c10::Half* in the graph MUST be validated at the
        // forward entry; shape checks alone are insufficient.)
        TORCH_CHECK(cond.defined() && cond.dim() == 1 &&
                    cond.size(0) == hidden_size(),
                    "RhinoVLAModel::forward: cond must be 1D [hidden_size=",
                    hidden_size(), "], got dim=", (cond.defined() ? cond.dim() : -1),
                    " size=", (cond.defined() ? cond.size(0) : -1));
        TORCH_CHECK(cond.device().type() == at::kPrivateUse1,
                    "RhinoVLAModel::forward: cond must be on RPU device");
        TORCH_CHECK(cond.scalar_type() == at::kHalf,
                    "RhinoVLAModel::forward: cond must be fp16");
        TORCH_CHECK(cond.is_contiguous(),
                    "RhinoVLAModel::forward: cond must be contiguous "
                    "(DMA source pointer assumes row-major layout)");

        // cos/sin validation — consumed as raw c10::Half* in rotary kernels.
        TORCH_CHECK(cos.defined() && sin.defined(),
                    "RhinoVLAModel::forward: cos/sin must be defined");
        TORCH_CHECK(cos.device().type() == at::kPrivateUse1 &&
                    sin.device().type() == at::kPrivateUse1,
                    "RhinoVLAModel::forward: cos/sin must be on RPU device");
        TORCH_CHECK(cos.scalar_type() == at::kHalf &&
                    sin.scalar_type() == at::kHalf,
                    "RhinoVLAModel::forward: cos/sin must be fp16");
        TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
                    "RhinoVLAModel::forward: cos/sin must be contiguous "
                    "(rotary kernel assumes row-major layout)");
        TORCH_CHECK(cos.sizes() == sin.sizes(),
                    "RhinoVLAModel::forward: cos/sin shapes must match, got cos=",
                    cos.sizes(), " sin=", sin.sizes());
        // The rotary
        // kernel reads cos_ref_/sin_ref_ as raw c10::Half* starting at
        // `cos_sin_start = rope_position_ + chunk.offset` for `seq_len` rows
        // × head_dim/2
        // columns; the DDR table contract is [max_seq, head_dim/2].
        //
        // Only accept head_dim/2. Accepting either half- or full-dim is incorrect — the kernel
        // consumes raw data_ptr with stride = head_dim/2 fp16 elements per
        // row. A full-dim caller would still reach kernel dispatch (validation
        // passes) but misindex positions. The canonical fused producer always
        // returns [suffix_len, head_dim/2]. Any caller that wants to
        // pass full-dim must slice to half-dim before the call.
        TORCH_CHECK(cos.dim() == 2,
                    "RhinoVLAModel::forward: cos must be 2D [max_pos, head_dim/2], got dim=",
                    cos.dim());
        TORCH_CHECK(sin.dim() == 2,
                    "RhinoVLAModel::forward: sin must be 2D [max_pos, head_dim/2], got dim=",
                    sin.dim());
        TORCH_CHECK(cos.size(1) == head_dim() / 2,
                    "RhinoVLAModel::forward: cos last-dim must equal head_dim/2=",
                    head_dim() / 2, " (kernel DDR table contract); got ",
                    cos.size(1),
                    ". If the caller has full-dim [max_pos, head_dim] tables, "
                    "slice to half-dim before .forward(): "
                    "cos_half = cos[..., :head_dim//2].contiguous().");
        TORCH_CHECK(rope_position_ >= 0,
                    "RhinoVLAModel::forward: logical RoPE position was not set");
        TORCH_CHECK(cos.size(0) >= rope_position_ + hidden_states.size(1),
                    "RhinoVLAModel::forward: cos row count (", cos.size(0),
                    ") must cover logical position + seq_len (", rope_position_,
                    "+", hidden_states.size(1), ")");

        // RhinoVLA always auto-computes chunk_size (not affected by global override)
        set_chunk_size_override(0);

        // Reset per-forward cond state: a new denoise step is a new cond.
        // cond_ref_/cos_ref_/sin_ref_ keep the tensors alive for the
        // synchronous forward (run_all_layers batch end executes before
        // return); cond DDR pointer cursor-patched per REPLAY via mutable DMA.
        cond_loaded_this_forward_ = false;
        cond_ref_ = cond;
        cos_ref_ = cos;
        sin_ref_ = sin;
        // cond_src_base_ feeds rpu_launch_ddr_scatter_spm_dma_mutable in the
        // cond scatter site below; REPLAY end() reads this live address. Per-
        // forward flush mirrors fused_model_base's hidden_in_src_base_ pattern
        // (caller may have CPU-dirty cache lines from an upstream op).
        cond_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond.data_ptr());
        rpu_ddr_flush_force(cond.data_ptr<c10::Half>());

        return run_all_layers(hidden_states, k_caches, v_caches,
                              attention_mask, position, is_causal);
    }

    at::Tensor denoise_loop_forward(
        const at::Tensor& x0_rpu,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cond_all,
        const at::Tensor& attention_mask,
        const at::Tensor& state_rpu,
        const at::Tensor& state_mask_rpu,
        const at::Tensor& action_mask_rpu,
        at::Tensor& x_out,
        double dt,
        int64_t prefix_len,
        int64_t num_steps) {
        TORCH_CHECK(denoise_loop_weights_ready_,
                    "RhinoVLAModel::denoise_loop_forward called before set_denoise_loop_weights");
        TORCH_CHECK(num_steps >= 1, "RhinoVLA denoise loop num_steps must be >= 1");
        TORCH_CHECK(num_steps * 1400 < 32000,
                    "RhinoVLA denoise loop node budget exceeded: num_steps=",
                    num_steps, " estimate=", num_steps * 1400);
        TORCH_CHECK(x0_rpu.dim() == 3 && x0_rpu.size(0) == 1 &&
                    x0_rpu.size(1) == action_horizon_ &&
                    x0_rpu.size(2) == action_dim_pad_ &&
                    x0_rpu.scalar_type() == at::kHalf &&
                    x0_rpu.device().type() == at::kPrivateUse1 &&
                    x0_rpu.is_contiguous(),
                    "RhinoVLA denoise loop x0 must be [1,", action_horizon_,
                    ",", action_dim_pad_, "] contiguous fp16 RPU");
        TORCH_CHECK(action_mask_rpu.sizes() == x0_rpu.sizes() &&
                    action_mask_rpu.scalar_type() == at::kHalf &&
                    action_mask_rpu.device().type() == at::kPrivateUse1 &&
                    action_mask_rpu.is_contiguous(),
                    "RhinoVLA denoise loop action_mask must match padded x0");
        TORCH_CHECK(x_out.sizes() == x0_rpu.sizes() &&
                    x_out.scalar_type() == at::kHalf &&
                    x_out.device().type() == at::kPrivateUse1 &&
                    x_out.is_contiguous(),
                    "RhinoVLA denoise loop x_out must match padded x0");
        TORCH_CHECK(state_rpu.dim() == 2 && state_rpu.size(0) == 1 &&
                    state_rpu.size(1) == state_dim_pad_ &&
                    state_rpu.scalar_type() == at::kHalf &&
                    state_rpu.device().type() == at::kPrivateUse1 &&
                    state_rpu.is_contiguous(),
                    "RhinoVLA denoise loop state must be [1,", state_dim_pad_,
                    "] contiguous fp16 RPU");
        TORCH_CHECK(state_mask_rpu.sizes() == state_rpu.sizes() &&
                    state_mask_rpu.scalar_type() == at::kHalf &&
                    state_mask_rpu.device().type() == at::kPrivateUse1 &&
                    state_mask_rpu.is_contiguous(),
                    "RhinoVLA denoise loop state_mask must match padded state");
        TORCH_CHECK(cond_all.dim() == 2 && cond_all.size(0) == num_steps &&
                    cond_all.size(1) == hidden_size() &&
                    cond_all.scalar_type() == at::kHalf &&
                    cond_all.device().type() == at::kPrivateUse1 &&
                    cond_all.is_contiguous(),
                    "RhinoVLA denoise loop cond_all must be [num_steps, hidden]");
        if (rhino_vla_precompute_adarms_enabled()) {
            TORCH_CHECK(denoise_adarms_tables_ready_,
                        "RPU_RHINOVLA_PRECOMPUTE_ADARMS=1 but AdaRMS tables were not bound");
            TORCH_CHECK(adarms_pair_table_.size(0) >= num_steps &&
                        adarms_final_table_.size(0) >= num_steps,
                        "RhinoVLA AdaRMS tables do not cover num_steps=", num_steps,
                        " pair_steps=", adarms_pair_table_.size(0),
                        " final_steps=", adarms_final_table_.size(0));
            adarms_pair_table_src_base_ =
                ::rhino_lkn::RpuGetDevAddr(adarms_pair_table_.data_ptr());
            adarms_final_table_src_base_ =
                ::rhino_lkn::RpuGetDevAddr(adarms_final_table_.data_ptr());
            rpu_ddr_flush_force(adarms_pair_table_.data_ptr<c10::Half>());
            rpu_ddr_flush_force(adarms_final_table_.data_ptr<c10::Half>());
        }
        if (rhino_vla_precompute_time_proj_enabled()) {
            TORCH_CHECK(denoise_time_proj_table_ready_,
                        "RPU_RHINOVLA_PRECOMPUTE_TIME_PROJ=1 but time-proj table was not bound");
            TORCH_CHECK(time_proj_table_.size(0) >= num_steps,
                        "RhinoVLA time-proj table does not cover num_steps=",
                        num_steps, " table_steps=", time_proj_table_.size(0));
            time_proj_table_src_base_ =
                ::rhino_lkn::RpuGetDevAddr(time_proj_table_.data_ptr());
            rpu_ddr_flush_force(time_proj_table_.data_ptr<c10::Half>());
        }
        TORCH_CHECK(prefix_len >= 0, "RhinoVLA denoise loop prefix_len must be >= 0");
        TORCH_CHECK(rope_position_ >= 0 &&
                        cos_loop_.size(0) >= rope_position_ + suffix_len_,
                    "RhinoVLA denoise loop RoPE table does not cover logical "
                    "position ", rope_position_, " + suffix_len ", suffix_len_);
        if (!k_caches.empty()) {
            const at::Tensor& k0 = k_caches[0];
            const int64_t cache_max_seq = k0.size(1) * k0.size(5);
            TORCH_CHECK(prefix_len + suffix_len_ <= cache_max_seq,
                        "RhinoVLA denoise loop prefix_len(", prefix_len,
                        ") + suffix_len(", suffix_len_,
                        ") exceeds k_cache max_seq_len=", cache_max_seq);
        }
        TORCH_CHECK(dt > 0.0, "RhinoVLA denoise loop dt must be > 0");
        const c10::Half dt_half = c10::Half(static_cast<float>(dt));
        if (!loop_mode_ || num_steps_ != num_steps) {
            loop_mode_ = true;
            num_steps_ = num_steps;
            dt_pinned_ = false;
            invalidate_model_state();
        }
        if (!dt_pinned_) {
            dt_ = dt_half;
            dt_pinned_ = true;
        } else {
            TORCH_CHECK(dt_ == dt_half,
                        "RhinoVLA denoise loop dt changed across replays");
        }

        set_chunk_size_override(0);
        cond_ref_ = cond_all;
        cos_ref_ = cos_loop_;
        sin_ref_ = sin_loop_;
        x0_ref_ = x0_rpu;
        state_ref_ = state_rpu;
        state_mask_ref_ = state_mask_rpu;
        action_mask_ref_ = action_mask_rpu;
        x_out_ref_ = x_out;
        rpu_ddr_flush_force(x0_rpu.data_ptr<c10::Half>());
        rpu_ddr_flush_force(cond_all.data_ptr<c10::Half>());
        rpu_ddr_flush_force(state_rpu.data_ptr<c10::Half>());
        rpu_ddr_flush_force(state_mask_rpu.data_ptr<c10::Half>());
        rpu_ddr_flush_force(action_mask_rpu.data_ptr<c10::Half>());
        rpu_ddr_flush_force(x_out.data_ptr<c10::Half>());
        x0_src_base_ = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
        cond_all_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond_all.data_ptr());
        state_src_base_ = ::rhino_lkn::RpuGetDevAddr(state_rpu.data_ptr());
        state_mask_src_base_ = ::rhino_lkn::RpuGetDevAddr(state_mask_rpu.data_ptr());
        action_mask_src_base_ = ::rhino_lkn::RpuGetDevAddr(action_mask_rpu.data_ptr());
        x_out_dst_base_ = ::rhino_lkn::RpuGetDevAddr(x_out.data_ptr());

        (void) run_all_layers(action_emb_stage_, k_caches, v_caches,
                              std::optional<at::Tensor>(attention_mask),
                              prefix_len, /*is_causal=*/false);
        return x_out;
    }

    // ========================================================================
    // debug_flush_dumps — materialize per-core SPM reads after forward.
    // Called by Python via rhino_vla_debug_flush_dumps op after a debug-export
    // forward finishes, to populate g_debug_tensors.
    // ========================================================================
    void debug_flush_dumps() {
        for (const auto& req : debug_dump_requests_) {
            at::Tensor t = at::empty({req.num_cores, req.elems_per_core},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
            c10::Half* dst = t.data_ptr<c10::Half>();
            for (int c = 0; c < req.num_cores; ++c) {
                const void* src = SPM_ALLOC.cpu_ptr(c, req.buf_off);
                std::memcpy(dst + c * req.elems_per_core, src,
                            req.elems_per_core * sizeof(c10::Half));
            }
            g_debug_tensors[req.name] = t;
        }
        debug_dump_requests_.clear();
    }

protected:
    // ========================================================================
    // static_config — graph-naive flat-phase subclass; no legacy graph slot.
    //
    // RhinoVLA is a flat-phase, SEQUENTIAL subclass — like AdaRMS. All four
    // Optional pointer-to-member slots (preload_fn / kv_first_fn /
    // kv_first_chunk_plan_fn / post_fn) stay nullptr by default. No persistent
    // weights, no KV_FIRST, no post-graph.
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers       = num_layers();
        // Pin to a single fused graph regardless of the global
        // `g_cross_layer_batch_size` (which is shared with Qwen3 text
        // decode where callers sometimes set it to 7 / 12 / 36). For
        // RhinoVLA action expert we always want all layers in one graph
        // replay per DDIM step — setting it equal to num_layers() makes
        // `compute_num_groups` in FusedModelBase return 1 unconditionally.
        cfg.cross_layer_batch_size = num_layers();
        cfg.fast_replay_skip_layer_loop = fast_replay_skip_layer_loop_;
        if (loop_mode_) {
            cfg.pre_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
                &RhinoVLAModel::emit_loop_pre_layers_body);
            cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
                &RhinoVLAModel::emit_loop_post_layers_body);
            cfg.body_iterations = num_steps_;
            cfg.fast_replay_skip_full_body = true;
        }
        return cfg;
    }

    // ========================================================================
    // dynamic_config — always SEQUENTIAL
    // ========================================================================
    ModelDynamicConfig dynamic_config(const ChunkPlan& /*plan*/) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::AUTO;
        return cfg;
    }

    // ========================================================================
    // declare_buffers — flat phases, ALL LayerWide
    //
    // Same as AdaRMS but adds q_norm_w and k_norm_w buffers for QK norm plus
    // debug-snapshot buffers allocated unconditionally so the graph layout
    // stays stable across debug/non-debug invocations.
    //
    // SPM budget (hidden=1024, intermediate=3072, heads=16, kv=8, head_dim=128):
    //   attn_tp = min(8, 8) = 8
    //   local_q_heads = 16/8 = 2
    //   local_kv_dim = 8*128/8 = 128
    //   residual ~ 2KB, q ~ 512B, kv ~ 256B, oproj ~ 2KB, mlp ~ 768B
    //   cond ~ 2KB, gemv ~ 6KB, q/k_norm_w ~ 256B each
    //   Total per core ~ 15KB << 8109 KB SPM budget
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
        int64_t mlp   = A(cs * (is_ / NUM_CORES) * DWIDTH);

        // SDPA tmp and mask sizing
        SdpaConfig sdpa_cfg = make_sdpa_config(ctx.use_attn_mask ? 4 : 1);
        int64_t tmp = A(sdpa_compute_tmp_v16_size(sdpa_cfg, cs) * 32);
        int64_t mask_sz = ctx.use_attn_mask
            ? A(cs * CeilDiv(ctx.max_kv_seq_len, (int64_t)16) * 32)
            : 0;

        // AdaRMS extras (sized per-core; broadcast DMA means same addr on all cores)
        int64_t cond_sz     = A(h * DWIDTH);
        int64_t gemv_out_sz = A(3 * h * DWIDTH);
        int64_t pair_gemv_out_sz = A(6 * h * DWIDTH);

        // QK norm weights (new vs AdaRMS)
        int64_t norm_w_sz = A(hd * DWIDTH);

        constexpr BufferScope ALL = BufferScope::LayerWide;
        int64_t nl = num_layers();

        std::vector<BufferDecl> decls;

        // Structural (mirrors AdaRMS, flat phases 0,0)
        decls.push_back({"residual1",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"input_norm",   res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"residual2",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});

        // Small-shape fusion: a persistent zeroed [seq, hidden] buffer
        // fed to all_reduce as the "residual" (add 0), so the gated-residual
        // path drops its redundant SUB. Zeroed once at BUILD via preload; never
        // written afterwards. Only declared when the flag is on.
        if (rhino_vla_gated_no_sub_enabled()) {
            BufferDecl zr;
            zr.name    = "zero_resid";
            zr.size    = res;
            zr.storage = StorageClass::Persistent;
            zr.scope   = ALL;
            zr.preload_callback =
                [res](FusedModelBase&, int, uint32_t core0_addr) {
                    rpu_launch_memset_spm_multicore(core0_addr, res / DWIDTH);
                };
            decls.push_back(zr);
        }
        decls.push_back({"q",            q,       0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"k",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"v",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"output",       out,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"oproj",        oproj,   0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"sdpa_tmp",     tmp,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"sdpa_mask",    mask_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"gate",         mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"up",           mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"down",         res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});

        // AdaRMS-specific
        decls.push_back({"cond",         cond_sz,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"attn_gemv",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"mlp_gemv",     gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"bias_temp",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"gemv_partial", gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"adarms_pair_gemv",       pair_gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"adarms_pair_bias_temp",  pair_gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"adarms_pair_partial",    pair_gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});

        // DEBUG snapshot buffers (only used when get_debug_export() is true;
        // allocated unconditionally so the graph layout stays stable).
        // These capture Q/K at Phase 3 (prenorm) and Phase 3b (post-norm)
        // without conflicting with later phases that reuse q/k/output/input_norm.
        decls.push_back({"dbg_q_prenorm", q,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_k_prenorm", kv,    0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_q_norm",    q,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_k_norm",    kv,    0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_sdpa_out",  out,   0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_oproj_out", oproj, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_layer_out", res,   0, 0, StorageClass::Temp, 0, nullptr, ALL});

        if (denoise_loop_weights_ready_) {
            int64_t ah = action_horizon_;
            int64_t adp = action_dim_pad_;
            int64_t sdp = state_dim_pad_;
            int64_t x_t_sz = A(ah * adp * DWIDTH);
            int64_t state_sz = A(sdp * DWIDTH);
            int64_t action_h_sz = A(ah * h * DWIDTH);
            int64_t token_h_sz = A(h * DWIDTH);
            int64_t v_t_sz = A(ah * adp * DWIDTH);

            auto preload_h = [](const at::Tensor* t, int64_t elems) {
                return [t, elems](FusedModelBase&, int, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        t->data_ptr<c10::Half>(), elems, core0_addr, /*num_cores=*/1);
                };
            };
            decls.push_back({"x_t_spm", x_t_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"action_mask_spm", x_t_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"state_spm", state_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"state_mask_spm", state_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"action_raw_spm", action_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"action_hidden_spm", action_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"time_spm", token_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"time_proj_spm", token_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"action_mask_hidden_spm", action_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"state_token_spm", token_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"state_mask_hidden_spm", token_h_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"final_gemv", gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"final_out_spm", res, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
            decls.push_back({"v_t_core0_spm", v_t_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});

            decls.push_back({"action_in_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&action_in_b_, h)});
            decls.push_back({"action_time_in_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&action_time_in_b_, h)});
            decls.push_back({"time_in_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&time_in_b_, h)});
            decls.push_back({"time_out_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&time_out_b_, h)});
            decls.push_back({"state_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&state_b_, h)});
            decls.push_back({"state_mask_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&state_mask_b_, h)});
            decls.push_back({"action_mask_b_spm", token_h_sz, 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&action_mask_b_, h)});
            decls.push_back({"action_out_b_spm", A(adp * DWIDTH), 0, 0,
                StorageClass::Persistent, 0, nullptr, ALL,
                preload_h(&action_out_b_, adp)});
        }

        // ─────────────────────────────────────────────────────────────────────
        // QK norm weights use PersistentPerLayer preload callbacks.
        // ─────────────────────────────────────────────────────────────────────
        {
            BufferDecl q_norm;
            q_norm.name      = "q_norm_w";
            q_norm.size      = norm_w_sz;
            q_norm.storage   = StorageClass::PersistentPerLayer;
            q_norm.per_layer = nl;
            q_norm.scope     = BufferScope::LayerWide;
            q_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        layer_weights_[L].q_norm_w.data_ptr<c10::Half>(),
                        head_dim(), core0_addr);
                };
            decls.push_back(q_norm);
        }
        {
            BufferDecl k_norm;
            k_norm.name      = "k_norm_w";
            k_norm.size      = norm_w_sz;
            k_norm.storage   = StorageClass::PersistentPerLayer;
            k_norm.per_layer = nl;
            k_norm.scope     = BufferScope::LayerWide;
            k_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        layer_weights_[L].k_norm_w.data_ptr<c10::Half>(),
                        head_dim(), core0_addr);
                };
            decls.push_back(k_norm);
        }

        return decls;
    }

    // The denoise-loop block above adds non-aliased Temp decls sized from
    // action_horizon_ / action_dim_pad_ / state_dim_pad_, and none of the three
    // reaches the framework's params hash. The persistent guard does not cover
    // them either: `ah` and `sdp` appear in no Persistent size at all, and the
    // one Persistent that carries `adp` is Align(adp*2, 256) = 256 for every
    // adp <= 128. These arrive through set_denoise_loop_weights(), which is a
    // SEPARATE op from set_weights with no C++-enforced ordering against
    // forward() — so denoise_loop_weights_ready_ flipping false->true after a
    // first forward is reachable, and that flip alone repacks the temp arena.
    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(0, denoise_loop_weights_ready_ ? 1 : 0);
        h = detail::layout_mix(h, action_horizon_);
        h = detail::layout_mix(h, action_dim_pad_);
        h = detail::layout_mix(h, state_dim_pad_);
        return h;
    }

    // ========================================================================
    // build_layer_subgraph — complete 16-operation per-layer pipeline
    //
    // Pipeline: cond DMA -> GEMV -> input DMA -> AdaRMSNorm(attn) -> QKV ->
    //   QKNorm -> RoPE -> KV insert -> SDPA -> OProj -> all_reduce_sum_residual
    //   -> gated attn residual(4-step) -> AdaRMSNorm(MLP) -> MLP(SiLU) ->
    //   gated MLP residual(4-step) -> output DMA
    //
    // Key differences from AdaRMS:
    //   - RoPE position and physical KV row are intentionally independent
    //   - kv_insert_pos = ctx().position + chunk.offset (absolute cache position)
    //   - QK Norm before RoPE (Qwen3 pattern)
    //   - Gated residual = SUB -> TANH -> broadcast MUL -> ADD (4 steps, not 3)
    //   - ActivationKind::SILU (not GELU)
    //   - cos_ref_/sin_ref_ DDR pointers (not cos_/sin_)
    //
    // Pitfall 3 structural fix: SDPA + KV-insert use addr_offset(name).value
    // (typed SpmOffset, compile-time distinct from absolute addr() uint32_t).
    // ========================================================================

    // RhinoVLA-only MLP pipeline variant for the small-shape fusion flags.
    // Mirrors FusedModelBase::emit_mlp_pipeline (SiLU path) but optionally:
    //   fuse_silu_mul: gate -> up -> silu_mul(gate,up,gate)  (drops SiLU unary)
    //   zero_residual: all_reduce gets zero_resid  -> residual1 = reduce(down),
    //                  so the caller's gated-residual SUB can be dropped.
    // fp16 only (no weight scales). Leaves the shared base path untouched.
    void emit_rhino_mlp_pipeline(const at::Tensor& gate_w,
                                 const at::Tensor& up_w,
                                 const at::Tensor& down_w,
                                 int64_t seq_len,
                                 bool fuse_silu_mul,
                                 bool zero_residual) {
        const int64_t h   = hidden_size();
        const int64_t is_ = intermediate_size();
        const int64_t elems = seq_len * (is_ / NUM_CORES);

        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), gate_w, addr(0, "gate"),
            seq_len, is_, h, /*partition=*/1, NUM_CORES);

        if (fuse_silu_mul) {
            // up first, then fused silu(gate)*up -> gate
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "residual1"), up_w, addr(0, "up"),
                seq_len, is_, h, /*partition=*/1, NUM_CORES);
            rpu_launch_silu_mul_spm_kernel(
                addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
                elems, NUM_CORES);
        } else {
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "gate"), addr(0, "gate"), elems, ValuOpType::SILU);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "residual1"), up_w, addr(0, "up"),
                seq_len, is_, h, /*partition=*/1, NUM_CORES);
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
                elems, ValuOpType::MUL, c10::Half(1.0));
        }

        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "gate"), down_w, addr(0, "down"),
            seq_len, h, is_, /*partition=*/0, NUM_CORES);

        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "down"),
            zero_residual ? addr(0, "zero_resid") : addr(0, "residual2"),
            addr(0, "residual1"),
            seq_len, h);
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t cos_sin_start = rope_position_ + chunk.offset;
        int64_t kv_insert_pos = ctx().position + chunk.offset;  // absolute cache pos
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int tp = attn_tp();
        int64_t num_elems_full = seq_len * h;
        bool is_first_layer_first_chunk =
            (layer_idx == 0) && (chunk.idx == 0) && !cond_loaded_this_forward_;
        bool is_chunk_zero = (chunk.idx == 0);
        bool fused_adarms = rhino_vla_fused_adarms_enabled();
        bool skip_adarms_gemv = rhino_vla_skip_adarms_gemv_enabled();
        bool precomputed_adarms =
            loop_mode_ && rhino_vla_precompute_adarms_enabled();

        // --------------------------------------------------------------------
        // Phase 0: cond DMA (once per forward)
        //
        // Row-partition GEMV consumes the cond as a per-core K-slice:
        // core c needs cond[c*H/8 : (c+1)*H/8] at its SPM base offset.
        // The GEMV kernel does NOT stride into a replicated buffer; it
        // just reads `local_k = H/8` elements from `core_base + cond_addr`
        // on each core. So we SCATTER cond from DDR to per-core SPM
        // (core c reads cond[c*H/8:(c+1)*H/8] from DDR and writes to the
        // SAME local SPM offset on its own core).
        //
        // A broadcast DMA would make every core read cond[0:H/8]. Scatter is
        // required so core c receives cond[c*H/8:(c+1)*H/8].
        // --------------------------------------------------------------------
        if (!loop_mode_ && is_first_layer_first_chunk) {
            int64_t local_k           = h / NUM_CORES;
            int64_t core_stride_bytes = local_k * DWIDTH;
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &cond_src_base_,
                /*src_offset_bytes=*/0,
                /*elements_per_core=*/local_k,
                /*core_stride_bytes=*/core_stride_bytes,
                addr(0, "cond"),
                /*num_cores=*/NUM_CORES);
            cond_loaded_this_forward_ = true;
        }

        // --------------------------------------------------------------------
        // Phase A: per-layer GEMV (chunk 0 only)
        // Computes [scale | shift | gate] from cond * dense_w + bias, with
        // +1.0 baked into the scale slice. Result persists in SPM for this
        // layer's remaining chunks.
        // --------------------------------------------------------------------
        if (is_chunk_zero) {
            if (skip_adarms_gemv) {
                identity_adarms_pair_to_spm(addr(0, "adarms_pair_gemv"));
            } else if (precomputed_adarms) {
                int64_t step = ctx().body_iter;
                int64_t offset =
                    ((step * num_layers() + layer_idx) * 6 * h) * DWIDTH;
                rpu_launch_ddr_broadcast_spm_dma_mutable(
                    &adarms_pair_table_src_base_,
                    offset,
                    6 * h,
                    addr(0, "adarms_pair_gemv"),
                    NUM_CORES);
            } else if (fused_adarms) {
                adarms_pair_gemv_to_spm(addr(0, "cond"),
                                         lw.pair_dense_w, lw.pair_dense_b,
                                         addr(0, "adarms_pair_gemv"),
                                         addr(0, "adarms_pair_bias_temp"),
                                         addr(0, "adarms_pair_partial"));
            } else {
                adarms_gemv_to_spm(addr(0, "cond"),
                                   lw.attn_dense_w, lw.attn_dense_b,
                                   addr(0, "attn_gemv"), addr(0, "bias_temp"),
                                   addr(0, "gemv_partial"));
                adarms_gemv_to_spm(addr(0, "cond"),
                                   lw.mlp_dense_w, lw.mlp_dense_b,
                                   addr(0, "mlp_gemv"), addr(0, "bias_temp"),
                                   addr(0, "gemv_partial"));
            }
        }

        // GEMV slice offsets (computed from core 0 base address; broadcast
        // DMA means every core shares the same offsets).
        const bool use_pair_adarms_buffer =
            skip_adarms_gemv || fused_adarms || precomputed_adarms;
        uint32_t attn_scale = use_pair_adarms_buffer ? addr(0, "adarms_pair_gemv") : addr(0, "attn_gemv");
        uint32_t attn_shift = attn_scale + (uint32_t)(h * DWIDTH);
        uint32_t attn_gate  = attn_scale + (uint32_t)(h * DWIDTH * 2);
        uint32_t mlp_scale  = use_pair_adarms_buffer
            ? addr(0, "adarms_pair_gemv") + (uint32_t)(h * DWIDTH * 3)
            : addr(0, "mlp_gemv");
        uint32_t mlp_shift  = mlp_scale + (uint32_t)(h * DWIDTH);
        uint32_t mlp_gate   = mlp_scale + (uint32_t)(h * DWIDTH * 2);

        // --------------------------------------------------------------------
        // Phase 1: layer input DMA
        // --------------------------------------------------------------------
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        // --------------------------------------------------------------------
        // Phase 2: AdaRMSNorm (attn): residual1 -> input_norm
        // (1+scale) was applied during GEMV, so RMSNorm uses attn_scale directly.
        // Then add shift (broadcast 1xC onto NxC rows).
        // --------------------------------------------------------------------
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            attn_scale,
            seq_len, h, eps_);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            attn_shift, addr(0, "input_norm"), addr(0, "input_norm"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);

        // --------------------------------------------------------------------
        // Phase 3: QKV Linear (col partition)
        // --------------------------------------------------------------------
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h,
            /*partition=*/1, tp);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp);

        // DEBUG EXPORT (layer 0): Phase 3 Q/K prenorm snapshot.
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            debug_dump_spm_per_core("rhino_vla_cond_L0",
                addr(0, "cond"), h, NUM_CORES,
                addr_offset("cond").value);
            debug_dump_spm_per_core("rhino_vla_qnormw_L0",
                layer_addr(layer_idx, 0, "q_norm_w"), hd, NUM_CORES,
                layer_addr_offset(layer_idx, "q_norm_w").value);
            debug_dump_spm_per_core("rhino_vla_residual1_L0",
                addr(0, "residual1"), seq_len * h, NUM_CORES,
                addr_offset("residual1").value);
            debug_dump_spm_per_core("rhino_vla_input_norm_L0",
                addr(0, "input_norm"), seq_len * h, NUM_CORES,
                addr_offset("input_norm").value);

            // In-graph copy: dbg_q_prenorm <- q
            int64_t q_elems_per_core = seq_len * local_q_heads_ * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "q"), c10::Half(0.0), addr(0, "dbg_q_prenorm"),
                q_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_q_prenorm_L0",
                addr(0, "dbg_q_prenorm"),
                q_elems_per_core, tp,
                addr_offset("dbg_q_prenorm").value);

            int64_t local_kv_heads_dbg = nkv / tp;
            int64_t k_elems_per_core = seq_len * local_kv_heads_dbg * hd;
            // In-graph copy: dbg_k_prenorm <- k
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "k"), c10::Half(0.0), addr(0, "dbg_k_prenorm"),
                k_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_k_prenorm_L0",
                addr(0, "dbg_k_prenorm"),
                k_elems_per_core, tp,
                addr_offset("dbg_k_prenorm").value);

            // V buffer — Phase 3 V linear output, never modified after Phase 5
            // V-cache insert.
            debug_dump_spm_per_core("rhino_vla_v_L0",
                addr(0, "v"),
                k_elems_per_core, tp,
                addr_offset("v").value);
        }

        // --------------------------------------------------------------------
        // Phase 3b: QK Norm (Qwen3 pattern: RMSNorm before RoPE)
        // --------------------------------------------------------------------
        int64_t local_kv_heads = nkv / tp;
        // q_norm_w / k_norm_w are PersistentPerLayer (see declare_buffers).
        // The .preload_callback DMA'd them at BUILD — we just read the slot.
        // Q norm: q -> output
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "q"), addr(0, "output"),
            layer_addr(layer_idx, 0, "q_norm_w"),
            seq_len * local_q_heads_, hd, eps_);
        // K norm: k -> input_norm
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "k"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "k_norm_w"),
            seq_len * local_kv_heads, hd, eps_);

        // DEBUG EXPORT (layer 0, chunk 0 only): snapshot Q norm and K norm
        // output into dedicated buffers BEFORE Phase 4 RoPE / Phase 6 SDPA
        // overwrite `q`/`output`/`input_norm`.
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            int64_t q_elems_per_core = seq_len * local_q_heads_ * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "output"), c10::Half(0.0), addr(0, "dbg_q_norm"),
                q_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_q_norm_L0",
                addr(0, "dbg_q_norm"),
                q_elems_per_core, tp,
                addr_offset("dbg_q_norm").value);

            int64_t k_elems_per_core = seq_len * local_kv_heads * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "input_norm"), c10::Half(0.0), addr(0, "dbg_k_norm"),
                k_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_k_norm_L0",
                addr(0, "dbg_k_norm"),
                k_elems_per_core, tp,
                addr_offset("dbg_k_norm").value);
        }

        // --------------------------------------------------------------------
        // Phase 4: RoPE (per-forward table, logical start is independent of
        // the physical KV insertion row).
        // --------------------------------------------------------------------
        rpu_launch_rope_spm_kernel(
            addr(0, "output"), addr(0, "q"),
            cos_ref_.data_ptr<c10::Half>(), sin_ref_.data_ptr<c10::Half>(),
            seq_len, local_q_heads_, hd, cos_sin_start);
        rpu_launch_rope_spm_kernel(
            addr(0, "input_norm"), addr(0, "k"),
            cos_ref_.data_ptr<c10::Half>(), sin_ref_.data_ptr<c10::Half>(),
            seq_len, local_kv_heads, hd, cos_sin_start);

        // DEBUG EXPORT (layer 0, chunk 0 only): dump Q/K after RoPE
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            debug_dump_spm_per_core("rhino_vla_q_rope_L0",
                addr(0, "q"),
                seq_len * local_q_heads_ * hd, tp,
                addr_offset("q").value);
            debug_dump_spm_per_core("rhino_vla_k_rope_L0",
                addr(0, "k"),
                seq_len * local_kv_heads * hd, tp,
                addr_offset("k").value);
        }

        // --------------------------------------------------------------------
        // Phase 5: KV cache insert (absolute position, NOT cos_sin_start).
        //
        // Pitfall 3 structural fix: KV-insert takes SPM OFFSETS, not absolute
        // addresses. addr_offset("name").value is compile-time distinct from
        // addr(core, "name").
        // --------------------------------------------------------------------
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        rpu_launch_insert_kcache_spm_unified(
            k_cache, kv_insert_pos, addr_offset("k").value,
            seq_len, nkv, hd, tp);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, kv_insert_pos, addr_offset("v").value,
            seq_len, nkv, hd, tp);

        // --------------------------------------------------------------------
        // Phase 6: SDPA (2D mask from Python, non-causal for prefix KV).
        //
        // Pitfall 3 structural fix: SDPA args (q / output / sdpa_tmp /
        // sdpa_mask) take SPM offsets via addr_offset(...).value.
        // --------------------------------------------------------------------
        int64_t kv_seq_len = ctx().position + ctx().seq_len;
        const bool has_2d_mask =
            ctx().attention_mask.has_value() && ctx().attention_mask->defined();
        // The SPM flash-attn kernel mishandles invalid lanes in a non-initial
        // partial K v16 tile. RhinoVLA's 2D mask path pads and masks dummy lanes.
        int64_t sdpa_kv_seq_len = has_2d_mask ? Align(kv_seq_len, (int64_t)16) : kv_seq_len;
        bool sdpa_causal = ctx().is_causal && (seq_len > 1);
        int mask_type = sdpa_load_mask_to_spm(
            ctx().attention_mask, sdpa_causal,
            addr_offset("sdpa_mask").value, seq_len, sdpa_kv_seq_len, tp,
            /*out_mask_ddr_ref=*/nullptr, sdpa_stable_mask_cache());
        rpu_launch_sdpa_spm_unified_kernel_v2(
            k_cache, v_cache, mask_type, c10::nullopt,
            addr_offset("q").value,
            addr_offset("output").value,
            addr_offset("sdpa_tmp").value,
            addr_offset("sdpa_mask").value,
            seq_len, nq, nkv, hd,
            sdpa_kv_seq_len, tp, NUM_CORES);

        // DEBUG EXPORT: snapshot SDPA output before O_proj consumes it
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            int64_t q_elems_per_core = seq_len * local_q_heads_ * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "output"), c10::Half(0.0), addr(0, "dbg_sdpa_out"),
                q_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_sdpa_out_L0",
                addr(0, "dbg_sdpa_out"),
                q_elems_per_core, tp,
                addr_offset("dbg_sdpa_out").value);
        }

        // --------------------------------------------------------------------
        // Phase 7: O_proj (row partition)
        // --------------------------------------------------------------------
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "output"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd,
            /*partition=*/0, tp);

        // DEBUG EXPORT: snapshot O_proj per-core output before all_reduce
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            int64_t oproj_elems_per_core = seq_len * h;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "oproj"), c10::Half(0.0), addr(0, "dbg_oproj_out"),
                oproj_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_oproj_out_L0",
                addr(0, "dbg_oproj_out"),
                oproj_elems_per_core, NUM_CORES,
                addr_offset("dbg_oproj_out").value);
        }

        // --------------------------------------------------------------------
        // Phase 8: all_reduce_sum_residual
        // residual2 = reduce_sum(oproj) + residual1
        //   no_sub: feed zero residual -> residual2 = reduce_sum(oproj), so the
        //   redundant Phase-9 SUB (which only undoes the +residual1) is dropped.
        // --------------------------------------------------------------------
        const bool gated_no_sub = rhino_vla_gated_no_sub_enabled();
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"),
            gated_no_sub ? addr(0, "zero_resid") : addr(0, "residual1"),
            addr(0, "residual2"),
            seq_len, h, tp, NUM_CORES);

        // --------------------------------------------------------------------
        // Phase 9: Gated attn residual (SUB -> TANH -> MUL -> ADD)
        // After all_reduce: residual2 = reduce(oproj) (+residual1 unless no_sub)
        // Goal: residual2 = residual1 + tanh(attn_gate) * reduce(oproj)
        //
        // SUB: residual2 = residual2 - residual1 = reduce(oproj) [skip if no_sub]
        // --------------------------------------------------------------------
        if (!gated_no_sub) {
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
                num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
        }
        // TANH: attn_gate = tanh(attn_gate) in-place [1,H]
        rpu_launch_eltwise_unary_spm_kernel(
            attn_gate, attn_gate,
            h, ValuOpType::TANH,
            /*is_gelu=*/false, /*num_cores=*/NUM_CORES);
        // MUL: residual2 = attn_gate [1,H] * residual2 [S,H]
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            attn_gate, addr(0, "residual2"), addr(0, "residual2"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
        // ADD: residual2 = residual2 + residual1 (gated result in residual2)
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);

        // --------------------------------------------------------------------
        // Phase 10: AdaRMSNorm (MLP): residual2 -> residual1
        // --------------------------------------------------------------------
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual2"), addr(0, "residual1"),
            mlp_scale,
            seq_len, h, eps_);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            mlp_shift, addr(0, "residual1"), addr(0, "residual1"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);

        // --------------------------------------------------------------------
        // Phase 11: MLP pipeline (SiLU, NOT GELU)
        // emit_mlp_pipeline: reads residual1, writes residual1 = reduce(down) + residual2
        //   silu_mul: fuse SiLU(gate)+(gate*up) into one kernel.
        //   no_sub:   feed zero residual -> residual1 = reduce(down), dropping
        //             the redundant Phase-13 SUB below.
        // --------------------------------------------------------------------
        const bool fuse_silu_mul = rhino_vla_fused_silu_mul_enabled();
        if (fuse_silu_mul || gated_no_sub) {
            emit_rhino_mlp_pipeline(lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w,
                                    seq_len, fuse_silu_mul, gated_no_sub);
        } else {
            emit_mlp_pipeline(lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w,
                              seq_len, ActivationKind::SILU);
        }

        // --------------------------------------------------------------------
        // Phase 13: Gated MLP residual (SUB -> TANH -> MUL -> ADD)
        // After MLP: residual1 = reduce(down) (+residual2 unless no_sub)
        // Goal: residual1 = residual2 + tanh(mlp_gate) * reduce(down)
        // --------------------------------------------------------------------
        if (!gated_no_sub) {
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
                num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
        }
        // TANH: mlp_gate = tanh(mlp_gate) in-place [1,H]
        rpu_launch_eltwise_unary_spm_kernel(
            mlp_gate, mlp_gate,
            h, ValuOpType::TANH,
            /*is_gelu=*/false, /*num_cores=*/NUM_CORES);
        // MUL: residual1 = mlp_gate [1,H] * residual1 [S,H]
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            mlp_gate, addr(0, "residual1"), addr(0, "residual1"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
        // ADD: residual1 = residual1 + residual2 (final gated MLP result)
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);

        // DEBUG EXPORT: snapshot per-layer final output (residual1)
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "residual1"), c10::Half(0.0), addr(0, "dbg_layer_out"),
                num_elems_full, ValuOpType::ADD);
            debug_dump_spm_per_core("rhino_vla_layer_out_L0",
                addr(0, "dbg_layer_out"),
                num_elems_full, NUM_CORES,
                addr_offset("dbg_layer_out").value);
        }

        // --------------------------------------------------------------------
        // Phase 14: Output DMA
        // --------------------------------------------------------------------
        if (!ctx().output_to_spm) {
            emit_layer_output_dma(layer_idx, chunk);
        }
    }

private:
    SdpaConfig make_sdpa_config(int mask = 1) const {
        return {SdpaKernelType::FLASH_ATTN_SPM, head_dim(), num_q_heads(), num_kv_heads(),
                attn_tp(), mask};
    }

    // ========================================================================
    // GEMV helper: row-partition 8-core + fused all_reduce+bias.
    //
    // Weight is the natural [3H, H] tensor row-partition swizzled so each core
    // holds K/8=H/8 columns of all 3H output rows. Layout of the final output
    // (on every core after reduce): [scale(hidden) | shift(hidden) | gate(hidden)].
    // The scale portion gets +1.0 in place so later RMSNorm sees (1+scale)
    // directly.
    //
    // Three distinct SPM buffers are required (all_reduce_sum_residual expects
    // input, residual, and output to be non-overlapping SPM regions):
    //   gemv_out_spm_addr     : final reduced+biased [3H] (broadcast to all cores)
    //   bias_temp_spm_addr    : DMA'd bias, used as the reduce-residual [3H]
    //   partial_spm_addr      : per-core partial [3H] from the row-partition GEMV
    //
    // Steps:
    //   1. DMA bias broadcast -> bias_temp (same bias on all 8 cores)
    //   2. GEMV (row partition, 8 cores): cond * dense_w -> partial
    //      Each core computes a partial [3H] along its local_k=H/8 slice.
    //   3. all_reduce_sum_residual(partial, bias) -> gemv_out
    //      Broadcast full [3H] to all cores.
    //   4. scale_slice += 1.0 (only first hidden_size elements)
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
            dense_b.data_ptr<c10::Half>(), out_features, bias_temp_spm_addr);
        // Step 2: Row-partition GEMV. partial_spm_addr holds the per-core [3H]
        // partial along the local_k = H/NUM_CORES split.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, /*num_cores=*/NUM_CORES,
            /*bias_spm_addr=*/0,
            /*force_gemm=*/false);
        // Step 3: Fused all_reduce + bias. reduce(partial) + bias -> gemv_out
        // broadcast to all output cores. Input and output MUST be distinct
        // SPM regions.
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
            /*M=*/1, /*N=*/out_features,
            /*input_num_cores=*/NUM_CORES, /*output_num_cores=*/NUM_CORES);
        // Step 4: (1+scale) -- add +1.0 only to the first hidden_size elements
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0),
            gemv_out_spm_addr, h, ValuOpType::ADD);
    }

    void adarms_pair_gemv_to_spm(uint32_t cond_spm_addr,
                                 const at::Tensor& dense_w,
                                 const at::Tensor& dense_b,
                                 uint32_t gemv_out_spm_addr,
                                 uint32_t bias_temp_spm_addr,
                                 uint32_t partial_spm_addr)
    {
        int64_t h = hidden_size();
        int64_t out_features = 6 * h;

        rpu_launch_ddr_broadcast_spm_dma(
            dense_b.data_ptr<c10::Half>(), out_features, bias_temp_spm_addr);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, /*num_cores=*/NUM_CORES,
            /*bias_spm_addr=*/0,
            /*force_gemm=*/false);
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
            /*M=*/1, /*N=*/out_features,
            /*input_num_cores=*/NUM_CORES, /*output_num_cores=*/NUM_CORES);

        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0),
            gemv_out_spm_addr, h, ValuOpType::ADD);
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr + (uint32_t)(3 * h * DWIDTH), c10::Half(1.0),
            gemv_out_spm_addr + (uint32_t)(3 * h * DWIDTH), h, ValuOpType::ADD);
    }

    void identity_adarms_pair_to_spm(uint32_t gemv_out_spm_addr)
    {
        int64_t h = hidden_size();
        rpu_launch_memset_spm_multicore(gemv_out_spm_addr, 6 * h);
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0),
            gemv_out_spm_addr, h, ValuOpType::ADD);
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr + (uint32_t)(3 * h * DWIDTH), c10::Half(1.0),
            gemv_out_spm_addr + (uint32_t)(3 * h * DWIDTH), h, ValuOpType::ADD);
    }

    void emit_loop_pre_layers_body() {
        const int64_t h = hidden_size();
        const int64_t ah = action_horizon_;
        const int64_t adp = action_dim_pad_;
        const int64_t sdp = state_dim_pad_;
        const int64_t bit = ctx().body_iter;
        const bool emit_static_context =
            !rhino_vla_denoise_static_context_cache_enabled() || bit == 0;
        const bool precomputed_adarms = rhino_vla_precompute_adarms_enabled();
        const bool precomputed_time_proj = rhino_vla_precompute_time_proj_enabled();
        const bool fold_action_time_in = rhino_vla_fold_action_time_in_enabled();

        if (bit == 0) {
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &x0_src_base_, 0, ah * adp, addr(0, "x_t_spm"), /*num_cores=*/1);
        }
        if (emit_static_context) {
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &action_mask_src_base_, 0, ah * adp,
                addr(0, "action_mask_spm"), /*num_cores=*/1);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &state_src_base_, 0, sdp, addr(0, "state_spm"), /*num_cores=*/1);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &state_mask_src_base_, 0, sdp, addr(0, "state_mask_spm"), /*num_cores=*/1);
        }
        if (!precomputed_time_proj) {
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &cond_all_src_base_, bit * h * DWIDTH, h,
                addr(0, "time_spm"), /*num_cores=*/1);
        }
        if (!precomputed_adarms) {
            const int64_t local_k = h / NUM_CORES;
            const int64_t core_stride_bytes = local_k * DWIDTH;
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &cond_all_src_base_, bit * h * DWIDTH,
                local_k, core_stride_bytes, addr(0, "cond"), NUM_CORES);
        }

        if (fold_action_time_in) {
            // Fold action_time_mlp_in(action_in_proj(x_t)) by linearity:
            // W_time_action * (W_action * x + b_action).
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "x_t_spm"), action_time_in_w_,
                addr(0, "action_hidden_spm"),
                ah, h, adp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "action_time_in_b_spm"));
        } else {
            // action_embeds = action_in_proj(x_t)
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "x_t_spm"), action_in_w_, addr(0, "action_raw_spm"),
                ah, h, adp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "action_in_b_spm"));

            // action_time_mlp_in([action_embeds, time_emb]) split by linearity:
            // action part [H] + time part [H] + bias, then SiLU + mlp_out.
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "action_raw_spm"), time_in_action_w_,
                addr(0, "action_hidden_spm"),
                ah, h, h, /*partition=*/1, /*num_cores=*/1);
        }
        if (precomputed_time_proj) {
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &time_proj_table_src_base_,
                bit * h * DWIDTH,
                h,
                addr(0, "time_proj_spm"),
                /*num_cores=*/1);
        } else {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "time_spm"), time_in_time_w_,
                addr(0, "time_proj_spm"),
                1, h, h, /*partition=*/1, /*num_cores=*/1,
                addr(0, "time_in_b_spm"), /*force_gemm=*/false);
        }
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            addr(0, "time_proj_spm"),
            addr(0, "action_hidden_spm"),
            addr(0, "action_hidden_spm"),
            ah, h, c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "action_hidden_spm"),
            addr(0, "action_hidden_spm"),
            ah * h, ValuOpType::SILU,
            /*is_gelu=*/false, /*num_cores=*/1);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "action_hidden_spm"), time_out_w_,
            addr(0, "action_raw_spm"),
            ah, h, h, /*partition=*/1, /*num_cores=*/1,
            addr(0, "time_out_b_spm"));

        // mask_condition: add action_mask_proj(action_mask) to action tokens.
        if (emit_static_context) {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "action_mask_spm"), action_mask_w_,
                addr(0, "action_mask_hidden_spm"),
                ah, h, adp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "action_mask_b_spm"));
        }
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "action_raw_spm"),
            addr(0, "action_mask_hidden_spm"),
            addr(0, "action_raw_spm"),
            ah * h, ValuOpType::ADD, c10::Half(1.0), /*num_cores=*/1);

        // state token = state_proj(state) + state_mask_proj(state_mask).
        if (emit_static_context) {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "state_spm"), state_w_, addr(0, "state_token_spm"),
                1, h, sdp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "state_b_spm"), /*force_gemm=*/false);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "state_mask_spm"), state_mask_w_,
                addr(0, "state_mask_hidden_spm"),
                1, h, sdp, /*partition=*/1, /*num_cores=*/1,
                addr(0, "state_mask_b_spm"), /*force_gemm=*/false);
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "state_token_spm"),
                addr(0, "state_mask_hidden_spm"),
                addr(0, "state_token_spm"),
                h, ValuOpType::ADD, c10::Half(1.0), /*num_cores=*/1);
        }

        if (get_debug_export()) {
            const std::string suffix = "_b" + std::to_string(bit);
            debug_dump_spm_per_core("rhino_vla_loop_state_token" + suffix,
                addr(0, "state_token_spm"), h, 1,
                addr_offset("state_token_spm").value);
            debug_dump_spm_per_core("rhino_vla_loop_action_hidden" + suffix,
                addr(0, "action_raw_spm"), ah * h, 1,
                addr_offset("action_raw_spm").value);
        }

        // Stage [state_token, action_tokens] into stable DDR for layer-0 input DMA.
        c10::Half* stage = action_emb_stage_.data_ptr<c10::Half>();
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "state_token_spm"), stage, h);
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "action_raw_spm"), stage + h, ah * h);
    }

    void emit_loop_post_layers_body() {
        const int64_t h = hidden_size();
        const int64_t ah = action_horizon_;
        const int64_t adp = action_dim_pad_;

        if (rhino_vla_precompute_adarms_enabled()) {
            int64_t step = ctx().body_iter;
            int64_t offset = (step * 3 * h) * DWIDTH;
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &adarms_final_table_src_base_,
                offset,
                3 * h,
                addr(0, "final_gemv"),
                NUM_CORES);
        } else {
            adarms_gemv_to_spm(addr(0, "cond"),
                               final_norm_w_, final_norm_b_,
                               addr(0, "final_gemv"),
                               addr(0, "bias_temp"),
                               addr(0, "gemv_partial"));
        }
        const uint32_t final_scale = addr(0, "final_gemv");
        const uint32_t final_shift = final_scale + (uint32_t)(h * DWIDTH);
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "final_out_spm"),
            final_scale, suffix_len_, h, eps_);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            final_shift, addr(0, "final_out_spm"), addr(0, "final_out_spm"),
            suffix_len_, h,
            c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);

        if (get_debug_export()) {
            const std::string suffix = "_b" + std::to_string(ctx().body_iter);
            debug_dump_spm_per_core("rhino_vla_loop_final_out" + suffix,
                addr(0, "final_out_spm"), suffix_len_ * h, 1,
                addr_offset("final_out_spm").value);
        }

        // Decode only action tokens; row 0 is the state token.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "final_out_spm") + (uint32_t)(h * DWIDTH),
            action_out_w_, addr(0, "v_t_core0_spm"),
            ah, adp, h, /*partition=*/1, /*num_cores=*/1,
            addr(0, "action_out_b_spm"));
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "v_t_core0_spm"),
            addr(0, "action_mask_spm"),
            addr(0, "v_t_core0_spm"),
            ah * adp, ValuOpType::MUL, c10::Half(1.0), /*num_cores=*/1);

        if (get_debug_export()) {
            const std::string suffix = "_b" + std::to_string(ctx().body_iter);
            debug_dump_spm_per_core("rhino_vla_loop_v_t" + suffix,
                addr(0, "v_t_core0_spm"), ah * adp, 1,
                addr_offset("v_t_core0_spm").value);
            debug_dump_spm_per_core("rhino_vla_loop_action_mask" + suffix,
                addr(0, "action_mask_spm"), ah * adp, 1,
                addr_offset("action_mask_spm").value);
        }

        // On-device fp16 Euler + mask: x = (x + dt*v) * action_mask.
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "x_t_spm"),
            addr(0, "v_t_core0_spm"),
            addr(0, "x_t_spm"),
            ah * adp, ValuOpType::ADD, dt_, /*num_cores=*/1);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "x_t_spm"),
            addr(0, "action_mask_spm"),
            addr(0, "x_t_spm"),
            ah * adp, ValuOpType::MUL, c10::Half(1.0), /*num_cores=*/1);
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "x_t_spm"), &x_out_dst_base_, 0, ah * adp);
    }

    // ========================================================================
    // debug_dump_spm_per_core — record metadata for a per-core SPM dump.
    //
    // The actual readback happens OUTSIDE the graph replay. We direct-map
    // per-core SPM via SPM_ALLOC.cpu_ptr(core, buf_off) and memcpy into an
    // at::Tensor. This avoids any kernel-level scatter and is a pure
    // post-execution operation.
    //
    // Call `debug_flush_dumps()` from the Python side after the forward
    // completes and before the next forward.
    // ========================================================================
    struct DebugDumpRequest {
        std::string name;
        uint32_t buf_off;
        int64_t elems_per_core;
        int num_cores;
    };
    std::vector<DebugDumpRequest> debug_dump_requests_;

    void debug_dump_spm_per_core(const std::string& name,
                                 uint32_t /*spm_core0_addr*/,
                                 int64_t elems_per_core,
                                 int num_cores,
                                 uint32_t buf_off)
    {
        if (!get_debug_export()) return;
        DebugDumpRequest req;
        req.name = name;
        req.buf_off = buf_off;
        req.elems_per_core = elems_per_core;
        req.num_cores = num_cores;
        debug_dump_requests_.push_back(req);
    }

    // ----- Model state -----
    std::vector<LayerWeights> layer_weights_;
    double eps_ = 1e-6;

    // Derived dims (computed in set_weights, cached for hot path)
    int64_t local_q_heads_ = 0;
    int64_t local_kv_dim_  = 0;

    // Per-forward tensor refs (kept alive for graph execution/replay via
    // class-member assignment; run_all_layers is synchronous so refcount
    // covers the full dispatch window). `cond_src_base_` is the live RPU
    // dev addr of cond_ref_'s data_ptr — feeds the mutable-DMA cond scatter
    // so REPLAY end() picks up the per-forward base from this storage.
    at::Tensor cond_ref_;
    uint64_t   cond_src_base_ = 0;
    at::Tensor cos_ref_;
    at::Tensor sin_ref_;
    int64_t rope_position_ = -1;
    bool cond_loaded_this_forward_ = false;
    bool fast_replay_skip_layer_loop_ = false;

    // Denoise-loop weights and mutable bases.
    bool denoise_loop_weights_ready_ = false;
    bool loop_mode_ = false;
    int64_t num_steps_ = 1;
    c10::Half dt_ = c10::Half(0.0f);
    bool dt_pinned_ = false;
    int64_t action_dim_ = 0;
    int64_t action_dim_pad_ = 0;
    int64_t state_dim_ = 0;
    int64_t state_dim_pad_ = 0;
    int64_t action_horizon_ = 0;
    int64_t suffix_len_ = 0;
    at::Tensor action_in_w_, action_in_b_;
    at::Tensor action_time_in_w_, action_time_in_b_;
    at::Tensor time_in_action_w_, time_in_time_w_, time_in_b_;
    at::Tensor time_out_w_, time_out_b_;
    at::Tensor state_w_, state_b_;
    at::Tensor state_mask_w_, state_mask_b_;
    at::Tensor action_mask_w_, action_mask_b_;
    at::Tensor final_norm_w_, final_norm_b_;
    at::Tensor action_out_w_, action_out_b_;
    at::Tensor cos_loop_, sin_loop_;
    bool denoise_adarms_tables_ready_ = false;
    at::Tensor adarms_pair_table_, adarms_final_table_;
    uint64_t adarms_pair_table_src_base_ = 0;
    uint64_t adarms_final_table_src_base_ = 0;
    bool denoise_time_proj_table_ready_ = false;
    at::Tensor time_proj_table_;
    uint64_t time_proj_table_src_base_ = 0;
    at::Tensor action_emb_stage_;
    at::Tensor x0_ref_, state_ref_, state_mask_ref_, action_mask_ref_, x_out_ref_;
    uint64_t x0_src_base_ = 0;
    uint64_t cond_all_src_base_ = 0;
    uint64_t state_src_base_ = 0;
    uint64_t state_mask_src_base_ = 0;
    uint64_t action_mask_src_base_ = 0;
    uint64_t x_out_dst_base_ = 0;

};

}  // namespace v3

// =============================================================================
// Instance registry — uses ModelHandleRegistry<v3::RhinoVLAModel> template
// =============================================================================

using RhinoVLARegistry = ModelHandleRegistry<v3::RhinoVLAModel>;

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

int64_t rpu_rhino_vla_create() {
    return RhinoVLARegistry::create();
}

void rpu_rhino_vla_destroy(int64_t handle) {
    RhinoVLARegistry::destroy(handle, "rpu_rhino_vla_destroy");
}

void rpu_rhino_vla_set_fast_replay(int64_t handle, bool enabled) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_fast_replay")
        ->set_fast_replay_skip_layer_loop(enabled);
}

void rpu_rhino_vla_set_rope_position(int64_t handle, int64_t position) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_rope_position")
        ->set_rope_position(position);
}

void rpu_rhino_vla_set_denoise_loop_weights(
    int64_t handle,
    const at::Tensor& action_in_w, const at::Tensor& action_in_b,
    const at::Tensor& action_time_in_w, const at::Tensor& action_time_in_b,
    const at::Tensor& time_in_action_w, const at::Tensor& time_in_time_w,
    const at::Tensor& time_in_b,
    const at::Tensor& time_out_w, const at::Tensor& time_out_b,
    const at::Tensor& state_w, const at::Tensor& state_b,
    const at::Tensor& state_mask_w, const at::Tensor& state_mask_b,
    const at::Tensor& action_mask_w, const at::Tensor& action_mask_b,
    const at::Tensor& final_norm_w, const at::Tensor& final_norm_b,
    const at::Tensor& action_out_w, const at::Tensor& action_out_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t action_dim, int64_t action_dim_pad,
    int64_t state_dim, int64_t state_dim_pad,
    int64_t action_horizon, int64_t suffix_len) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_denoise_loop_weights")
        ->set_denoise_loop_weights(
            action_in_w, action_in_b,
            action_time_in_w, action_time_in_b,
            time_in_action_w, time_in_time_w, time_in_b,
            time_out_w, time_out_b,
            state_w, state_b,
            state_mask_w, state_mask_b,
            action_mask_w, action_mask_b,
            final_norm_w, final_norm_b,
            action_out_w, action_out_b,
            cos, sin,
            action_dim, action_dim_pad,
            state_dim, state_dim_pad,
            action_horizon, suffix_len);
}

void rpu_rhino_vla_set_denoise_loop_adarms_tables(
    int64_t handle,
    const at::Tensor& pair_table,
    const at::Tensor& final_table) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_denoise_loop_adarms_tables")
        ->set_denoise_loop_adarms_tables(pair_table, final_table);
}

void rpu_rhino_vla_set_denoise_loop_time_proj_table(
    int64_t handle,
    const at::Tensor& table) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_set_denoise_loop_time_proj_table")
        ->set_denoise_loop_time_proj_table(table);
}

void rpu_rhino_vla_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    at::TensorList pair_dense_w_list, at::TensorList pair_dense_b_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps)
{
    RhinoVLARegistry::get(handle, "rpu_rhino_vla")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_list, up_list, down_list,
        attn_dense_w_list, attn_dense_b_list,
        mlp_dense_w_list,  mlp_dense_b_list,
        pair_dense_w_list, pair_dense_b_list,
        q_norm_list, k_norm_list,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps);
}

at::Tensor rpu_rhino_vla_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal)
{
    // TensorList -> std::vector<at::Tensor> (shallow copy; tensors are refcounted)
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());

    return RhinoVLARegistry::get(handle, "rpu_rhino_vla")->forward(
        hidden_states, cond, k_caches, v_caches,
        cos, sin, attention_mask, position, is_causal);
}

at::Tensor rpu_rhino_vla_denoise_loop_forward(
    int64_t handle,
    const at::Tensor& x0_rpu,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const at::Tensor& cond_all,
    const at::Tensor& attention_mask,
    const at::Tensor& state_rpu,
    const at::Tensor& state_mask_rpu,
    const at::Tensor& action_mask_rpu,
    at::Tensor x_out,
    double dt,
    int64_t prefix_len,
    int64_t num_steps) {
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return RhinoVLARegistry::get(handle, "rpu_rhino_vla_denoise_loop_forward")
        ->denoise_loop_forward(
            x0_rpu, k_caches, v_caches, cond_all, attention_mask,
            state_rpu, state_mask_rpu, action_mask_rpu, x_out,
            dt, prefix_len, num_steps);
}

// Post-forward helper: materialize per-core SPM dumps requested during the
// graph build (only active when debug export is enabled). This MUST be
// called after a forward finishes; it reads SPM via direct CPU-mapped
// pointers and populates g_debug_tensors, which Python reads via
// rpu_backend.get_debug_tensor(name).
void rpu_rhino_vla_debug_flush_dumps(int64_t handle) {
    RhinoVLARegistry::get(handle, "rpu_rhino_vla_debug_flush_dumps")->debug_flush_dumps();
}
