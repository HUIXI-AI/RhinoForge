// src/fused/rpu_pi05_denoise_step_model.cpp
//
// Pi0.5 num_steps fused denoise step model (FusedModelBase v3 subclass).
//
// The pre-layer, per-layer, and post-layer hooks run the complete denoise step.

#include "rpu_pi05_denoise_step_model.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"

#include <algorithm>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace v3 {

Pi05DenoiseStepModel::Pi05DenoiseStepModel()  = default;
Pi05DenoiseStepModel::~Pi05DenoiseStepModel() = default;

namespace {

void check_pi05_scale_lists(
    const char* ctx,
    int64_t n_layers,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w,
    at::TensorList o_w, at::TensorList gate_w, at::TensorList up_w,
    at::TensorList down_w,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws)
{
    const bool has_scale = !q_ws.empty();
    auto check_scale_list = [&](const at::TensorList& list, const char* name) {
        if (has_scale) {
            TORCH_CHECK(static_cast<int64_t>(list.size()) == n_layers,
                        ctx, ": ", name, ".size()=", list.size(),
                        " != num_layers=", n_layers);
        } else {
            TORCH_CHECK(list.empty(),
                        ctx, ": ", name,
                        " must be empty unless all W8A16 scale lists are provided");
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
                    ctx, ": q_w_scale.size()=", q_ws.size(),
                    " != num_layers=", n_layers);
    }

    auto check_weight_dtype = [&](const at::Tensor& w,
                                  const at::Tensor& scale,
                                  const char* name,
                                  int64_t i) {
        TORCH_CHECK(w.device().type() == at::kPrivateUse1 && w.is_contiguous(),
                    ctx, ": ", name, "[", i, "] must be contiguous RPU tensor");
        TORCH_CHECK(w.scalar_type() == at::kHalf || w.scalar_type() == at::kChar
                    || w.scalar_type() == at::kByte,
                    ctx, ": ", name, "[", i, "] must be fp16 / int8 / packed-uint8, got ",
                    w.scalar_type());
        if (has_scale) {
            TORCH_CHECK(w.scalar_type() == at::kChar || w.scalar_type() == at::kByte,
                        ctx, ": quantized mode requires int8(W8A16) or uint8(packed-INT4) ",
                        name, "[", i, "], got ", w.scalar_type());
            TORCH_CHECK(scale.defined() && scale.dim() == 1
                        && scale.scalar_type() == at::kHalf
                        && scale.device().type() == at::kPrivateUse1
                        && scale.is_contiguous(),
                        ctx, ": ", name, "_scale[", i,
                        "] must be 1D contiguous fp16 RPU tensor");
            TORCH_CHECK(scale.numel() == w.size(0),
                        ctx, ": ", name, "_scale[", i, "].numel()=",
                        scale.numel(), " != output dim=", w.size(0));
        } else {
            TORCH_CHECK(w.scalar_type() != at::kChar && w.scalar_type() != at::kByte,
                        ctx, ": quantized ", name, "[", i,
                        "] requires scale lists");
        }
    };

    for (int64_t i = 0; i < n_layers; ++i) {
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

// ============================================================================
// set_weights: invalidate_model_state() must be the last non-empty statement.
// ============================================================================
void Pi05DenoiseStepModel::set_rope_position(int64_t position) {
    rope_position_ = position;
}

void Pi05DenoiseStepModel::set_configured_chunk_size(int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0
                    || (chunk_size >= 16 && chunk_size % 16 == 0),
                "Pi0.5 action chunk size must be 0 (auto) or a positive "
                "multiple of 16, got ", chunk_size);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Pi0.5 action chunk size must be set before the first forward");
    const int64_t single_chunk_min = ((chunk_size_ + 15) / 16) * 16;
    TORCH_CHECK(chunk_size == 0 || chunk_size >= single_chunk_min,
                "Pi0.5 action expert requires one chunk for the full action "
                "horizon (", chunk_size_, " rows); configured chunk size must "
                "be 0 (auto) or at least ", single_chunk_min, ", got ",
                chunk_size);
    configured_chunk_size_ = chunk_size;
    invalidate_model_state();
}

void Pi05DenoiseStepModel::set_weights(
    at::TensorList q_w, at::TensorList k_w,
    at::TensorList v_w, at::TensorList o_w,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    const at::Tensor& final_norm_dense_w, const at::Tensor& final_norm_dense_b,
    const at::Tensor& action_in_proj_w,   const at::Tensor& action_in_proj_b,
    const at::Tensor& action_out_proj_w,  const at::Tensor& action_out_proj_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t hidden_size, int64_t max_action_dim, int64_t chunk_size,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t num_layers, double eps,
    at::TensorList q_w_scale, at::TensorList k_w_scale,
    at::TensorList v_w_scale, at::TensorList o_w_scale,
    at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale,
    at::TensorList attn_dense_w_scale, at::TensorList mlp_dense_w_scale,
    const at::Tensor& final_norm_dense_w_scale)
{
    TORCH_CHECK(num_layers > 0 && hidden_size > 0 && head_dim > 0
                && num_q_heads > 0 && num_kv_heads > 0
                && max_action_dim > 0 && chunk_size > 0,
                "Pi05DenoiseStepModel::set_weights: dim params must be positive");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "Pi05DenoiseStepModel::set_weights: num_q_heads must be divisible by num_kv_heads");
    auto check_list = [&](const at::TensorList& list, const char* name) {
        TORCH_CHECK(static_cast<int64_t>(list.size()) == num_layers,
                    "Pi05DenoiseStepModel::set_weights: ", name,
                    ".size()=", list.size(), " != num_layers=", num_layers);
    };
    check_list(q_w, "q_w");
    check_list(k_w, "k_w");
    check_list(v_w, "v_w");
    check_list(o_w, "o_w");
    check_list(gate_w, "gate_w");
    check_list(up_w, "up_w");
    check_list(down_w, "down_w");
    check_list(attn_dense_w_list, "attn_dense_w_list");
    check_list(attn_dense_b_list, "attn_dense_b_list");
    check_list(mlp_dense_w_list, "mlp_dense_w_list");
    check_list(mlp_dense_b_list, "mlp_dense_b_list");
    check_pi05_scale_lists(
        "Pi05DenoiseStepModel::set_weights", num_layers,
        q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
        q_w_scale, k_w_scale, v_w_scale, o_w_scale,
        gate_scale, up_scale, down_scale);
    const bool has_scale = !q_w_scale.empty();
    // AdaRMS dense W8A16 is independent of the decoder-projection W8A16 above
    // (the dense can be quantized while projections are fp16, or vice versa).
    const bool has_attn_dense_scale = !attn_dense_w_scale.empty();
    const bool has_mlp_dense_scale = !mlp_dense_w_scale.empty();
    const bool has_final_dense_scale =
        final_norm_dense_w_scale.defined()
        && final_norm_dense_w_scale.numel() != 0;
    TORCH_CHECK(has_attn_dense_scale == has_mlp_dense_scale
                && has_attn_dense_scale == has_final_dense_scale,
                "Pi05DenoiseStepModel::set_weights: dense scales must be all "
                "provided or all empty");
    const bool has_dense_scale = has_attn_dense_scale;
    if (has_dense_scale) {
        TORCH_CHECK(static_cast<int64_t>(attn_dense_w_scale.size()) == num_layers
                    && static_cast<int64_t>(mlp_dense_w_scale.size()) == num_layers,
                    "Pi05DenoiseStepModel::set_weights: dense scale lists must "
                    "have num_layers entries when provided");
    }

    auto check_dense = [&](const at::Tensor& weight,
                           const at::Tensor& bias,
                           const at::Tensor& scale,
                           const char* name) {
        TORCH_CHECK(weight.defined() && weight.dim() == 2
                    && weight.size(0) == 3 * hidden_size
                    && weight.size(1) == hidden_size
                    && weight.device().type() == at::kPrivateUse1
                    && weight.is_contiguous(),
                    "Pi05DenoiseStepModel::set_weights: ", name,
                    " weight must be contiguous RPU [3 * hidden_size, "
                    "hidden_size]");
        const auto expected_weight_dtype =
            has_dense_scale ? at::kChar : at::kHalf;
        TORCH_CHECK(weight.scalar_type() == expected_weight_dtype,
                    "Pi05DenoiseStepModel::set_weights: ", name,
                    " weight dtype must be ", expected_weight_dtype,
                    ", got ", weight.scalar_type());
        TORCH_CHECK(bias.defined() && bias.dim() == 1
                    && bias.numel() == 3 * hidden_size
                    && bias.scalar_type() == at::kHalf
                    && bias.device().type() == at::kPrivateUse1
                    && bias.is_contiguous(),
                    "Pi05DenoiseStepModel::set_weights: ", name,
                    " bias must be contiguous fp16 RPU [3 * hidden_size]");
        if (has_dense_scale) {
            TORCH_CHECK(scale.defined() && scale.dim() == 1
                        && scale.numel() == 3 * hidden_size
                        && scale.scalar_type() == at::kHalf
                        && scale.device().type() == at::kPrivateUse1
                        && scale.is_contiguous(),
                        "Pi05DenoiseStepModel::set_weights: ", name,
                        " scale must be contiguous fp16 RPU [3 * hidden_size]");
        }
    };
    for (int64_t i = 0; i < num_layers; ++i) {
        check_dense(
            attn_dense_w_list[i], attn_dense_b_list[i],
            has_dense_scale ? attn_dense_w_scale[i] : at::Tensor(),
            "attn_dense");
        check_dense(
            mlp_dense_w_list[i], mlp_dense_b_list[i],
            has_dense_scale ? mlp_dense_w_scale[i] : at::Tensor(),
            "mlp_dense");
    }
    check_dense(
        final_norm_dense_w, final_norm_dense_b,
        has_dense_scale ? final_norm_dense_w_scale : at::Tensor(),
        "final_norm_dense");

    // intermediate_size from down_w[0].size(1) (col after row-partition swizzle
    // gives intermediate as the second dim of [hidden, intermediate]).
    // packed-INT4 (uint8) down weight is [hidden, intermediate/2] (nibble-packed
    // K), so size(1) is half the logical intermediate — double it back.
    const int64_t intermediate_size =
        (down_w[0].scalar_type() == at::kByte) ? down_w[0].size(1) * 2
                                               : down_w[0].size(1);
    set_model_params(num_q_heads, num_kv_heads, head_dim,
                     hidden_size, intermediate_size);
    set_num_layers(num_layers);
    eps_              = eps;
    max_action_dim_   = max_action_dim;
    chunk_size_       = chunk_size;
    local_q_heads_    = num_q_heads / attn_tp();
    local_kv_dim_     = num_kv_heads * head_dim / attn_tp();

    layer_weights_.clear();
    layer_weights_.reserve(num_layers);
    for (int64_t i = 0; i < num_layers; ++i) {
        layer_weights_.push_back({
            q_w[i], k_w[i], v_w[i], o_w[i],
            gate_w[i], up_w[i], down_w[i],
            has_scale ? q_w_scale[i] : at::Tensor(),
            has_scale ? k_w_scale[i] : at::Tensor(),
            has_scale ? v_w_scale[i] : at::Tensor(),
            has_scale ? o_w_scale[i] : at::Tensor(),
            has_scale ? gate_scale[i] : at::Tensor(),
            has_scale ? up_scale[i] : at::Tensor(),
            has_scale ? down_scale[i] : at::Tensor(),
            attn_dense_w_list[i], attn_dense_b_list[i],
            mlp_dense_w_list[i],  mlp_dense_b_list[i],
            has_dense_scale ? attn_dense_w_scale[i] : at::Tensor(),
            has_dense_scale ? mlp_dense_w_scale[i]  : at::Tensor(),
        });
    }
    final_norm_dense_w_  = final_norm_dense_w;
    final_norm_dense_b_  = final_norm_dense_b;
    final_norm_dense_ws_ = final_norm_dense_w_scale;
    action_in_proj_w_   = action_in_proj_w;
    action_in_proj_b_   = action_in_proj_b;
    action_out_proj_w_  = action_out_proj_w;
    action_out_proj_b_  = action_out_proj_b;
    cos_ = cos;
    sin_ = sin;

    // Stable RPU staging tensor allocated once — data_ptr
    // valid across step + sample_actions. Passed as hidden_states to satisfy
    // FMB shape contract; layer 0 reads it via FMB's hidden_in_src_base_.
    auto opts = at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1);
    action_emb_stage_ = at::empty({1, chunk_size, hidden_size}, opts);

    invalidate_model_state();  // Last non-empty statement.
}

// ============================================================================
// step_forward — validate inputs and emit one denoise step.
// ============================================================================
void Pi05DenoiseStepModel::step_forward(
    const at::Tensor& x_t_rpu,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const at::Tensor& cond_step,
    const at::Tensor& attention_mask_4d,
    at::Tensor& v_t_buf,
    int64_t prefix_len)
{
    TORCH_CHECK(num_layers() > 0,
                "Pi05DenoiseStepModel::step_forward called before set_weights");
    TORCH_CHECK(x_t_rpu.device().type() == at::kPrivateUse1
                && x_t_rpu.scalar_type() == at::kHalf
                && x_t_rpu.dim() == 3
                && x_t_rpu.size(0) == 1
                && x_t_rpu.size(1) == chunk_size_
                && x_t_rpu.size(2) == max_action_dim_
                && x_t_rpu.is_contiguous(),
                "Pi05DenoiseStepModel::step_forward: x_t_rpu shape "
                "[1,", chunk_size_, ",", max_action_dim_,
                "] fp16 contig RPU required");
    TORCH_CHECK(v_t_buf.sizes() == x_t_rpu.sizes()
                && v_t_buf.scalar_type() == at::kHalf
                && v_t_buf.device().type() == at::kPrivateUse1
                && v_t_buf.is_contiguous(),
                "Pi05DenoiseStepModel::step_forward: v_t_buf shape/dtype mismatch");
    TORCH_CHECK(cond_step.device().type() == at::kPrivateUse1
                && cond_step.scalar_type() == at::kHalf
                && cond_step.dim() == 1
                && cond_step.size(0) == hidden_size()
                && cond_step.is_contiguous(),
                "Pi05DenoiseStepModel::step_forward: cond_step must be 1D [",
                hidden_size(), "] fp16 contig RPU");
    TORCH_CHECK(prefix_len >= 0,
                "Pi05DenoiseStepModel::step_forward: prefix_len must be >= 0");

    // kv_cache must have headroom for
    // prefix + suffix. (Layout-aware: max_seq_len stored on the cache
    // is the dim that K's [...sKeyVx, ..., sKeyChunk...] sums to.)
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        // Re-derive max_seq_len from the cache shape: sKeyVx * sKeyChunk
        // (sKeyVx = dim 1, sKeyChunk = dim 5 in the 7-D K-cache layout).
        const int64_t cache_max_seq = kc.size(1) * kc.size(5);
        TORCH_CHECK(prefix_len + chunk_size_ <= cache_max_seq,
                    "Pi05DenoiseStepModel::step_forward: prefix_len(",
                    prefix_len, ") + chunk_size(", chunk_size_,
                    ") exceeds k_cache max_seq_len=", cache_max_seq);
    }

    // Mode-switch guard: a prior denoise_loop_forward call may have left
    // loop_mode_==true (which would set body_iterations=num_steps_ and corrupt
    // this single-step forward). Reset to the single-step graph before building.
    if (loop_mode_) {
        loop_mode_ = false;
        num_steps_ = 1;
        dt_pinned_ = false;  // loop graph baked dt_; single-step must rebuild cleanly
        invalidate_model_state();
    }

    // Per-step bookkeeping.
    const int64_t action_chunk = configured_chunk_size_ > 0
        ? configured_chunk_size_ : ((chunk_size_ + 15) / 16) * 16;
    set_chunk_size_override(action_chunk);
    // Keep all three mutable-base tensors alive across deferred graph execution.
    // The bases are dereferenced when the graph EXECUTES (RpuKernelGraph::end(),
    // i.e. when the caller's `with cache.capture(sig):` exits), not when this
    // function returns; a caller that drops its local frees the DDR block and
    // the driver may re-issue its device VA before then. Same shape as
    // denoise_loop_forward's x0_ref_/cond_all_ref_/x_out_ref_ below. v_t_buf is
    // the worst case: it is a WRITE destination, so a re-issued VA means the
    // graph writes over whatever now owns that block, not merely reads garbage.
    x_t_ref_  = x_t_rpu;
    cond_ref_ = cond_step;
    v_t_ref_  = v_t_buf;

    // Update mutable bases and flush caller's DDR-dirty lines
    // (FMB applies its own Boundary guard inside run_all_layers; calling
    // rpu_ddr_flush_force here is safe and required for graph BUILD).
    rpu_ddr_flush_force(x_t_rpu.data_ptr<c10::Half>());
    rpu_ddr_flush_force(cond_step.data_ptr<c10::Half>());
    rpu_ddr_flush_force(v_t_buf.data_ptr<c10::Half>());

    x_t_src_base_  = ::rhino_lkn::RpuGetDevAddr(x_t_rpu.data_ptr());
    cond_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond_step.data_ptr());
    v_t_dst_base_  = ::rhino_lkn::RpuGetDevAddr(v_t_buf.data_ptr());

    // SDPA mask cache — mirror the AdaRMSModel pattern.
    // Factored into a shared helper (also called by denoise_loop_forward).
    prepare_sdpa_mask_cached(attention_mask_4d, prefix_len);

    // Drive run_all_layers with action_emb_stage_ as the FMB-contract
    // hidden_states tensor (satisfies hidden_size_=1024 shape check). The
    // actual numerical input flowed via emit_pre_layers_body ([A1..A4])
    // → action_emb_stage_ DDR; layer 0's emit_layer_input_dma reads it
    // through FMB's hidden_in_src_base_ mutable path automatically.
    (void) run_all_layers(action_emb_stage_, k_caches, v_caches,
                          std::optional<at::Tensor>(attention_mask_4d),
                          /*position=*/prefix_len, /*is_causal=*/false);

    // Output v_t was written to v_t_buf via emit_post_layers_body ([Z3] DMA).
}

// ============================================================================
// prepare_sdpa_mask_cached — shared SDPA mask prepare + cache.
// Mirrors the AdaRMSModel pattern. Called by both
// step_forward and denoise_loop_forward.
// ============================================================================
void Pi05DenoiseStepModel::prepare_sdpa_mask_cached(
    const at::Tensor& attention_mask_4d, int64_t prefix_len)
{
    const int64_t prep_seq_q = chunk_size_;
    const int64_t prep_seq_k = prefix_len + chunk_size_;
    const bool    prep_is_causal = false;
    const bool    has_mask = attention_mask_4d.defined();
    const void*   in_ptr   = has_mask ? attention_mask_4d.data_ptr() : nullptr;
    const void*   cached   = prepared_mask_input_ref_.defined()
                              ? prepared_mask_input_ref_.data_ptr() : nullptr;
    const bool cache_hit = (cached == in_ptr)
                        && (prepared_mask_seq_q_ == prep_seq_q)
                        && (prepared_mask_seq_k_ == prep_seq_k)
                        && (prepared_mask_is_causal_ == prep_is_causal);
    if (!cache_hit) {
        prepared_mask_ = sdpa_prepare_mask(
            has_mask ? std::optional<at::Tensor>(attention_mask_4d)
                     : std::nullopt,
            prep_is_causal, prep_seq_q, prep_seq_k,
            sdpa_stable_mask_cache());
        prepared_mask_input_ref_  = has_mask ? attention_mask_4d : at::Tensor();
        prepared_mask_seq_q_      = prep_seq_q;
        prepared_mask_seq_k_      = prep_seq_k;
        prepared_mask_is_causal_  = prep_is_causal;
    }
}

// ============================================================================
// denoise_loop_forward — N-step in-graph unroll.
//
// Gated by loop_mode_: static_config() sets body_iterations=num_steps_, so
// run_all_layers loops the body (and the pre/post hooks) num_steps times in a
// single graph. The pre-hook loads per-iteration conditioning and the post-hook
// performs the on-device Euler update. The single-step path stays byte-identical
// because loop_mode_ is false unless this op is invoked.
// ============================================================================
void Pi05DenoiseStepModel::denoise_loop_forward(
    at::Tensor x0_rpu, std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches, at::Tensor cond_all,
    at::Tensor attention_mask_4d, at::Tensor x_out,
    double dt, int64_t prefix_len, int64_t num_steps) {
    TORCH_CHECK(num_layers() > 0,
                "Pi05DenoiseStepModel::denoise_loop_forward called before set_weights");
    const int64_t cs  = chunk_size_;
    const int64_t mad = max_action_dim_;
    TORCH_CHECK(x0_rpu.dim() == 3 && x0_rpu.size(0) == 1 && x0_rpu.size(1) == cs
        && x0_rpu.size(2) == mad && x0_rpu.scalar_type() == at::kHalf
        && x0_rpu.is_contiguous() && x0_rpu.device().type() == at::kPrivateUse1,
        "denoise_loop_forward: x0_rpu must be [1,", cs, ",", mad, "] fp16 contig rpu");
    TORCH_CHECK(cond_all.dim() == 2 && cond_all.size(0) == num_steps
        && cond_all.size(1) == hidden_size() && cond_all.scalar_type() == at::kHalf
        && cond_all.is_contiguous() && cond_all.device().type() == at::kPrivateUse1,
        "denoise_loop_forward: cond_all must be [", num_steps, ",", hidden_size(),
        "] fp16 contig rpu");
    TORCH_CHECK(x_out.sizes() == x0_rpu.sizes() && x_out.scalar_type() == at::kHalf
        && x_out.is_contiguous() && x_out.device().type() == at::kPrivateUse1,
        "denoise_loop_forward: x_out must match x0_rpu shape, fp16 contig rpu");
    TORCH_CHECK(num_steps >= 1, "num_steps must be >= 1");
    // Node-count guard: a batch overflow is SILENT (build_batch return ignored).
    // Cap is pending.size() <= 32768; budget ~1100 nodes/body with margin.
    TORCH_CHECK(num_steps * 1100 < 32000,
        "denoise_loop_forward: num_steps(", num_steps, ") * 1100 exceeds 32000 node cap");
    TORCH_CHECK(prefix_len >= 0, "denoise_loop_forward: prefix_len must be >= 0");
    if (!k_caches.empty()) {
        const at::Tensor& k0 = k_caches[0];
        const int64_t cache_max_seq = k0.size(1) * k0.size(5);
        TORCH_CHECK(prefix_len + cs <= cache_max_seq,
            "denoise_loop_forward: prefix_len(", prefix_len, ") + chunk(", cs,
            ") exceeds k_cache max_seq_len=", cache_max_seq);
    }

    // dt is baked into the in-graph Euler at BUILD and is identical across all
    // unrolled iterations.
    TORCH_CHECK(dt < 0.0, "denoise_loop_forward: dt must be < 0 (Pi05 flow-matching dt = -1/num_steps)");
    const c10::Half dt_half = c10::Half(static_cast<float>(dt));

    // (Re)build when entering loop mode, or when loop cardinality / horizon changes.
    if (!loop_mode_ || num_steps_ != num_steps || chunk_size_ != cs) {
        loop_mode_ = true;
        num_steps_ = num_steps;
        dt_pinned_ = false;
        invalidate_model_state();
    }
    if (!dt_pinned_) { dt_ = dt_half; dt_pinned_ = true; }
    else TORCH_CHECK(dt_ == dt_half, "dt changed across replays (baked into the graph)");

    const int64_t action_chunk = configured_chunk_size_ > 0
        ? configured_chunk_size_ : ((chunk_size_ + 15) / 16) * 16;
    set_chunk_size_override(action_chunk);

    // SDPA mask cache (shared helper; same block step_forward uses).
    prepare_sdpa_mask_cached(attention_mask_4d, prefix_len);

    // Keepalives across the synchronous forward + flush caller's DDR-dirty lines.
    x0_ref_ = x0_rpu; cond_all_ref_ = cond_all; x_out_ref_ = x_out;
    rpu_ddr_flush_force(x0_rpu.data_ptr<c10::Half>());
    rpu_ddr_flush_force(cond_all.data_ptr<c10::Half>());
    rpu_ddr_flush_force(x_out.data_ptr<c10::Half>());
    x0_src_base_       = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
    cond_all_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond_all.data_ptr());
    x_out_dst_base_    = ::rhino_lkn::RpuGetDevAddr(x_out.data_ptr());
    // The loop-mode pre/post hooks read x0_src_base_ / cond_all_src_base_ (with
    // per-body offsets) and x_out_dst_base_ directly — no single-step base bridge.

    // ONE forward — run_all_layers loops body_iterations(=num_steps_) internally;
    // the pre/post hooks fire per iteration with ctx().body_iter advancing.
    (void) run_all_layers(action_emb_stage_, k_caches, v_caches,
                          std::optional<at::Tensor>(attention_mask_4d),
                          /*position=*/prefix_len, /*is_causal=*/false);
}

// ============================================================================
// FusedModelBase virtuals
// ============================================================================
ModelStaticConfig Pi05DenoiseStepModel::static_config() {
    ModelStaticConfig cfg;
    cfg.num_layers             = num_layers();
    cfg.cross_layer_batch_size = num_layers();
    // pre_layers_fn / post_layers_fn hooks.
    cfg.pre_layers_fn  = reinterpret_cast<void (FusedModelBase::*)()>(
        &Pi05DenoiseStepModel::emit_pre_layers_body);
    cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
        &Pi05DenoiseStepModel::emit_post_layers_body);
    cfg.body_iterations = loop_mode_ ? num_steps_ : 1;  // In-graph denoise loop.
    return cfg;
}

ModelDynamicConfig Pi05DenoiseStepModel::dynamic_config(const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    return cfg;
}

// Pad the denoise KV-insert sequence to the next 16 so the v16 insert engages
// when both sequence and position are 16-aligned. The Pi05 prefix is aligned,
// and zeroed padding rows preserve the logical sequence result.
// Requires position%16==0 (true for the standard config); default off (opt-in).
// LATCHED on first call, matching gr00t_kvpad16_enabled.
// kv_rows in declare_buffers() is derived from this flag but the allocation hash
// cannot see it, so an un-memoized re-read let a mid-process env flip size the SPM
// k/v buffers at BUILD and the insert at REPLAY from two different answers.
static bool kvinsert_pad16_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_PI05_KVINSERT_PAD16");
        cached = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T')) ? 1 : 0;
    }
    return cached != 0;
}
static inline int64_t pad16(int64_t n) { return ((n + 15) / 16) * 16; }

std::vector<BufferDecl> Pi05DenoiseStepModel::declare_buffers(const LayoutContext& ctx) {
    int64_t cs = ctx.chunk_size;
    int64_t h  = hidden_size();
    int64_t nq = num_q_heads();
    int64_t nkv = num_kv_heads();
    int64_t hd = head_dim();
    int64_t is_ = intermediate_size();
    int64_t local_q  = nq / attn_tp();
    int64_t local_kv = nkv * hd / attn_tp();
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

    // v16 KV-insert pad: k/v buffers get padded rows so the insert can read a
    // 16-aligned seq. The padded rows are zeroed before the insert (Phase 5).
    int64_t kv_rows = kvinsert_pad16_enabled() ? pad16(cs) : cs;
    int64_t res   = A(cs * h * DWIDTH);
    int64_t q     = A(cs * local_q * hd * DWIDTH);
    int64_t kv    = A(kv_rows * local_kv * DWIDTH);
    int64_t out   = A(cs * local_q * hd * DWIDTH);
    int64_t oproj = A(cs * h * DWIDTH);
    int64_t mlp   = A(cs * (is_ / NUM_CORES) * DWIDTH);

    SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM, hd, nq, nkv, attn_tp(), ctx.use_attn_mask ? 4 : 1};
    int64_t tmp = A(sdpa_compute_tmp_v16_size(sdpa_cfg, cs) * 32);
    int64_t mask_sz = ctx.use_attn_mask
        ? A(cs * CeilDiv(ctx.max_kv_seq_len, (int64_t)16) * 32)
        : 0;

    int64_t cond_sz     = A(h * DWIDTH);
    int64_t gemv_out_sz = A(3 * h * DWIDTH);

    // Pi05-specific sizes.
    int64_t x_t_spm_sz       = A(cs * max_action_dim_ * DWIDTH);     // 3.2 KB
    int64_t action_emb_sz    = A(cs * h * DWIDTH);                   // 100 KB
    int64_t v_t_core0_sz     = A(cs * max_action_dim_ * DWIDTH);     // 3.2 KB

    // Persistent bias buffers.
    int64_t in_bias_sz   = A(h * DWIDTH);                            // 2 KB
    int64_t out_bias_sz  = A(max_action_dim_ * DWIDTH);              // 64 B

    constexpr BufferScope ALL = BufferScope::LayerWide;
    using SC = StorageClass;

    // Preload callbacks for persistent bias buffers.
    auto in_bias_preload = [this](FusedModelBase& /*self*/, int /*L*/,
                                  uint32_t core0_addr) {
        rpu_launch_ddr_broadcast_spm_dma(
            action_in_proj_b_.data_ptr<c10::Half>(),
            hidden_size(), core0_addr, NUM_CORES);
    };
    auto out_bias_preload = [this](FusedModelBase& /*self*/, int /*L*/,
                                   uint32_t core0_addr) {
        rpu_launch_ddr_broadcast_spm_dma(
            action_out_proj_b_.data_ptr<c10::Half>(),
            max_action_dim_, core0_addr, NUM_CORES);
    };

    return {
        // Structural buffers shared with the AdaRMSModel layer body.
        {"residual1",    res,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"input_norm",   res,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"residual2",    res,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"q",            q,       0, 0, SC::Temp, 0, nullptr, ALL},
        {"k",            kv,      0, 0, SC::Temp, 0, nullptr, ALL},
        {"v",            kv,      0, 0, SC::Temp, 0, nullptr, ALL},
        {"output",       out,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"oproj",        oproj,   0, 0, SC::Temp, 0, nullptr, ALL},
        {"sdpa_tmp",     tmp,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"sdpa_mask",    mask_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"gate",         mlp,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"up",           mlp,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"down",         res,     0, 0, SC::Temp, 0, nullptr, ALL},
        // AdaRMS extras
        {"cond",         cond_sz,     0, 0, SC::Temp, 0, nullptr, ALL},
        {"attn_gemv",    gemv_out_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"mlp_gemv",     gemv_out_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"bias_temp",    gemv_out_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"gemv_partial", gemv_out_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        // Pi05-specific temporaries. action_emb_stage_ is the DDR handoff,
        // so no action_emb_spm buffer is needed.
        {"x_t_spm",              x_t_spm_sz,    0, 0, SC::Temp, 0, nullptr, ALL},
        {"action_emb_core0_spm", action_emb_sz, 0, 0, SC::Temp, 0, nullptr, ALL},
        {"final_gemv",           gemv_out_sz,   0, 0, SC::Temp, 0, nullptr, ALL},
        {"final_out_spm",        res,           0, 0, SC::Temp, 0, nullptr, ALL},
        {"v_t_core0_spm",        v_t_core0_sz,  0, 0, SC::Temp, 0, nullptr, ALL},
        // Pi05-specific persistent buffers.
        {"action_in_proj_b_spm",  in_bias_sz,  0, 0, SC::Persistent, 0, nullptr, ALL,
            in_bias_preload},
        {"action_out_proj_b_spm", out_bias_sz, 0, 0, SC::Persistent, 0, nullptr, ALL,
            out_bias_preload},
    };
}

void Pi05DenoiseStepModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    const auto& lw = layer_weights_[layer_idx];
    int64_t seq_len = chunk.len;
    int64_t cos_sin_start = ctx().position + chunk.offset;
    int64_t h = hidden_size();
    int64_t nq = num_q_heads();
    int64_t nkv = num_kv_heads();
    int64_t hd = head_dim();
    int tp = attn_tp();
    int64_t num_elems_full = seq_len * h;
    bool is_chunk_zero = (chunk.idx == 0);

    // emit_pre_layers_body ([A2]) exclusively owns the condition DMA to avoid
    // double emission.

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
                           addr(0, "gemv_partial"),
                           lw.attn_dense_ws);
        adarms_gemv_to_spm(addr(0, "cond"),
                           lw.mlp_dense_w,  lw.mlp_dense_b,
                           addr(0, "mlp_gemv"),
                           addr(0, "bias_temp"),
                           addr(0, "gemv_partial"),
                           lw.mlp_dense_ws);
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
        seq_len, h, eps_);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
        attn_shift, addr(0, "input_norm"), addr(0, "input_norm"),
        seq_len, h,
        c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);

    // --------------------------------------------------------------------
    // Phase 3: QKV Linear (attn_tp cores, col partition)
    // --------------------------------------------------------------------
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "input_norm"), lw.q_w, addr(0, "q"),
        seq_len, nq * hd, h,
        /*partition=*/1, tp, /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.q_ws);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "input_norm"), lw.k_w, addr(0, "k"),
        seq_len, nkv * hd, h,
        /*partition=*/1, tp, /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.k_ws);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "input_norm"), lw.v_w, addr(0, "v"),
        seq_len, nkv * hd, h,
        /*partition=*/1, tp, /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.v_ws);

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
    rpu_launch_rope_spm_kernel(
        addr(0, "q"), addr(0, "q"),
        cos_ptr, sin_ptr,
        seq_len, local_q_heads_, hd, rope_start, tp);
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
    // v16 pad: pad the insert to 16 so kvinsert_use_v16 engages — but ONLY when
    // v16 will ACTUALLY fire (insert position cos_sin_start % 16 == 0) AND the
    // padded insert fits the cache. Padding when v16 can't fire just inflates the
    // SLOW non-v16 insert by (pad16(seq)-seq) tokens and over-runs the prefix+chunk
    // capacity pre-check (runtime TORCH_CHECK on an otherwise-valid non-16-aligned
    // prefix). cache_cap = sKeyVx(size(1)) * sKeyChunk(16); the remaining v16
    // conditions (num_cores==8, nkv%8==0, hd%16==0) are Pi05-denoise invariants.
    int64_t insert_seq = seq_len;
    if (kvinsert_pad16_enabled()) {
        const int64_t padded = pad16(seq_len);
        const int64_t cache_cap = k_cache.size(1) * 16;  // sKeyVx * sKeyChunk
        if (cos_sin_start % 16 == 0 && cos_sin_start + padded <= cache_cap) {
            insert_seq = padded;
        }
    }
    if (insert_seq > seq_len) {
        // Zero v16 padding rows before insertion. Logical KV length excludes
        // these cache slots, so attention never reads them.
        int64_t local_kv = nkv * hd / tp;
        int64_t pad_elems = (insert_seq - seq_len) * local_kv;
        uint32_t off = (uint32_t)(seq_len * local_kv * DWIDTH);
        rpu_launch_memset_spm_multicore(addr(0, "k") + off, pad_elems);
        rpu_launch_memset_spm_multicore(addr(0, "v") + off, pad_elems);
    }
    rpu_launch_insert_kcache_spm_unified(
        k_cache, cos_sin_start, addr_offset("k").value,
        insert_seq, nkv, hd, tp);
    rpu_launch_insert_vcache_spm_unified(
        v_cache, cos_sin_start, addr_offset("v").value,
        insert_seq, nkv, hd, tp);

    // --------------------------------------------------------------------
    // Phase 6: SDPA (causal or 2D mask).
    //
    // Pitfall 3 structural fix: SDPA args (q / output / sdpa_tmp /
    // sdpa_mask) take SPM offsets via addr_offset(...).value.
    // --------------------------------------------------------------------
    int64_t kv_seq_len = ctx().position + ctx().seq_len;
    // The mask is prepared once at forward() entry.
    // Per-layer Phase 6 only DMAs the prepared DDR tensor → SPM.
    // mask_type was determined at prepare time (MASK_NONE/MASK_LTM/MASK_2D).
    sdpa_dma_mask_to_spm(prepared_mask_,
                         addr_offset("sdpa_mask").value,
                         seq_len, kv_seq_len, tp);
    int mask_type = prepared_mask_.mask_type;
    rpu_launch_sdpa_spm_unified_kernel_v2(
        k_cache, v_cache, mask_type, c10::nullopt,
        addr_offset("q").value,
        addr_offset("output").value,
        addr_offset("sdpa_tmp").value,
        addr_offset("sdpa_mask").value,
        seq_len, nq, nkv, hd,
        kv_seq_len, tp, NUM_CORES);

    // --------------------------------------------------------------------
    // Phase 7: O_proj (row partition, attn_tp cores)
    // --------------------------------------------------------------------
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd,
        /*partition=*/0, tp, /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.o_ws);

    // --------------------------------------------------------------------
    // Phase 8: all_reduce_sum_residual (attn)
    // residual2 = reduce_sum(oproj, attn_tp cores) + residual1
    // Output goes to all NUM_CORES.
    // --------------------------------------------------------------------
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
        seq_len, h, tp, NUM_CORES);

    // --------------------------------------------------------------------
    // Phase 9: GATED attn residual
    // residual2 = attn_gate * (residual2 - residual1) + residual1
    // Implemented as 3-step: SUB full-size, 1xC->NxC MUL gate slice, ADD full-size
    // --------------------------------------------------------------------
    rpu_launch_eltwise_binary_spm_kernel(          // residual2 -= residual1
        addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
        num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(  // residual2 *= attn_gate (1xC broadcast)
        attn_gate, addr(0, "residual2"), addr(0, "residual2"),
        seq_len, h,
        c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_eltwise_binary_spm_kernel(          // residual2 += residual1
        addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
        num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);

    // --------------------------------------------------------------------
    // Phase 10: Post-attention AdaRMSNorm (MLP side): residual2 -> residual1
    // (1+scale_mlp) baked into mlp_scale; then add shift_mlp.
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
    // Phase 11: MLP pipeline
    // --------------------------------------------------------------------
    emit_mlp_pipeline(lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w,
                      seq_len, ActivationKind::GELU,
                      lw.gate_ws, lw.up_ws, lw.down_ws);

    // --------------------------------------------------------------------
    // Phase 13: GATED MLP residual
    // residual1 = mlp_gate * (residual1 - residual2) + residual2
    // After emit_mlp_pipeline: residual1 = reduce(down) + residual2
    // So (residual1 - residual2) = reduce(down), then gate it, then add back residual2.
    // --------------------------------------------------------------------
    rpu_launch_eltwise_binary_spm_kernel(          // residual1 -= residual2
        addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
        num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(  // residual1 *= mlp_gate (1xC broadcast)
        mlp_gate, addr(0, "residual1"), addr(0, "residual1"),
        seq_len, h,
        c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_eltwise_binary_spm_kernel(          // residual1 += residual2
        addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
        num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);

    // --------------------------------------------------------------------
    // Phase 14: Output DMA (unless output stays in SPM for next layer)
    // --------------------------------------------------------------------
    if (!ctx().output_to_spm) {
        emit_layer_output_dma(layer_idx, chunk);  // from "residual1"
    }
}

void Pi05DenoiseStepModel::emit_pre_layers_body() {
    // Pre-layers emission uses *_mutable DMA variants from the first runnable
    // graph. Mutable bases are written in step_forward before run_all_layers.

    const int64_t h    = hidden_size();
    const int64_t cs   = chunk_size_;
    const int64_t mad  = max_action_dim_;

    // [A1] x_t DDR → x_t_spm broadcast (MUTABLE — REPLAY rewrites the base
    // via Queue_t::update_dma_kernel). In loop mode, load the initial noise x0
    // ONLY on body_iter==0; iters 1..N-1 read the in-SPM Euler result the
    // post-hook leaves in x_t_spm. Single-step always loads.
    if (!loop_mode_ || ctx().body_iter == 0) {
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            loop_mode_ ? &x0_src_base_ : &x_t_src_base_,
            /*src_offset_bytes=*/0,
            /*num_elements=*/cs * mad,
            /*spm_addr_unified=*/addr(0, "x_t_spm"),
            /*num_cores=*/NUM_CORES);
    }

    // [A2] cond_step DDR → "cond" SPM scatter (MUTABLE — REPLAY rewrites
    // cond_src_base_). This is exclusively owned by pre_layers_fn; the layer
    // body does not emit another condition DMA.
    {
        const int64_t local_k           = h / NUM_CORES;
        const int64_t core_stride_bytes = local_k * DWIDTH;
        // In loop mode, scatter cond_all[body_iter] (row stride h*DWIDTH bytes);
        // single-step uses cond_src_base_ at offset 0.
        const int64_t cond_off_b = loop_mode_ ? ctx().body_iter * h * DWIDTH : 0;
        rpu_launch_ddr_scatter_spm_dma_mutable(
            loop_mode_ ? &cond_all_src_base_ : &cond_src_base_,
            /*src_offset_bytes=*/cond_off_b,
            /*elements_per_core=*/local_k,
            /*core_stride_bytes=*/core_stride_bytes,
            /*spm_addr_unified=*/addr(0, "cond"),
            /*num_cores=*/NUM_CORES);
    }

    // [A3] action_in_proj single-core SPM linear with bias.
    // M=cs=50, N=H_ada=1024, K=max_action_dim=32 (K%16==0 OK; num_cores=1
    // skips the K%(16*8) requirement). Bias preloaded into
    // action_in_proj_b_spm via its preload callback.
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        /*input_spm_addr=*/addr(0, "x_t_spm"),
        /*weight=*/action_in_proj_w_,
        /*output_spm_addr=*/addr(0, "action_emb_core0_spm"),
        /*M=*/cs, /*N=*/h, /*K=*/mad,
        /*partition=*/1, /*num_cores=*/1,
        /*bias_spm_addr=*/addr(0, "action_in_proj_b_spm"));

    // [A4] core-0 SPM → action_emb_stage_ DDR copy (100 KB; FIXED —
    // action_emb_stage_ data_ptr is stable across step + sample_actions). The
    // layer body's Phase 1 (emit_layer_input_dma) then
    // reads from action_emb_stage_ via FMB's hidden_in_src_base_ mutable
    // mechanism, so no separate broadcast is emitted.
    rpu_launch_spm_copy_ddr_dma(
        /*spm_addr_unified=*/addr(0, "action_emb_core0_spm"),
        /*ddr_ptr=*/action_emb_stage_.data_ptr<c10::Half>(),
        /*num_elements=*/cs * h);
}

void Pi05DenoiseStepModel::emit_post_layers_body() {
    // Post-layers precondition: build_layer_subgraph for the
    // last layer left "residual1" with the final layer output in broadcast SPM.

    const int64_t h   = hidden_size();
    const int64_t cs  = chunk_size_;
    const int64_t mad = max_action_dim_;

    // [Z1] Final PiGemmaRMSNorm — mirrors AdaRMS Phase A + Phase 2 (plain;
    // no gated residual on final norm). GEMV(cond * final_dense_w + bias)
    // → final_gemv. RMSNorm(residual1, final_gemv) → final_out_spm. Then
    // 1xC→NxC ADD shift to final_out_spm in-place.
    adarms_gemv_to_spm(addr(0, "cond"),
                       final_norm_dense_w_, final_norm_dense_b_,
                       addr(0, "final_gemv"),
                       addr(0, "bias_temp"),
                       addr(0, "gemv_partial"),
                       final_norm_dense_ws_);
    const uint32_t final_scale = addr(0, "final_gemv");
    const uint32_t final_shift = final_scale + (uint32_t)(h * DWIDTH);
    // Gate slice (offset 2*h*DWIDTH) is unused for final norm.
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "final_out_spm"),
        final_scale,
        cs, h, eps_);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
        final_shift, addr(0, "final_out_spm"), addr(0, "final_out_spm"),
        cs, h,
        c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);

    // [Z2] action_out_proj single-core SPM linear with bias.
    // M=cs, N=max_action_dim=32, K=H_ada=1024 (K=1024, K%16==0).
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        /*input_spm_addr=*/addr(0, "final_out_spm"),
        /*weight=*/action_out_proj_w_,
        /*output_spm_addr=*/addr(0, "v_t_core0_spm"),
        /*M=*/cs, /*N=*/mad, /*K=*/h,
        /*partition=*/1, /*num_cores=*/1,
        /*bias_spm_addr=*/addr(0, "action_out_proj_b_spm"));

    if (loop_mode_) {
        // [Z3-loop] On-device Euler: x_t_spm = x_t_spm + dt_·v_t_core0_spm
        // (in-place, core 0). x and v are both [cs, mad] (equal widths). dt_ is
        // baked at BUILD. x_t_spm is private (Pi05's all-phase-0 buffer layout
        // gives every buffer its own slot), so
        // this result survives the layer loop and persists to the next body's
        // [A1]-skipped / [A3] read.
        rpu_launch_eltwise_binary_spm_kernel(
            /*a=*/addr(0, "x_t_spm"),
            /*b=*/addr(0, "v_t_core0_spm"),
            /*output=*/addr(0, "x_t_spm"),
            /*num_elements=*/cs * mad,
            ValuOpType::ADD, /*alpha=*/dt_, /*num_cores=*/1);
        // [Z4-loop] core-0 x_t_spm → x_out DDR (MUTABLE). Written every body;
        // the last body leaves the final action (Pi05 needs only final x, not
        // the trajectory).
        rpu_launch_spm_copy_ddr_dma_mutable(
            /*spm_addr_unified=*/addr(0, "x_t_spm"),
            /*live_dst_base=*/&x_out_dst_base_,
            /*dst_offset_bytes=*/0,
            /*num_elements=*/cs * mad);
    } else {
        // [Z3] core-0 v_t_core0_spm → v_t_buf DDR (MUTABLE — REPLAY rewrites
        // v_t_dst_base_). Mutable DMA signature:
        //   (uint32_t spm_addr_unified, const uint64_t* live_dst_base,
        //    int64_t dst_offset_bytes, int64_t num_elements)
        rpu_launch_spm_copy_ddr_dma_mutable(
            /*spm_addr_unified=*/addr(0, "v_t_core0_spm"),
            /*live_dst_base=*/&v_t_dst_base_,
            /*dst_offset_bytes=*/0,
            /*num_elements=*/cs * mad);
    }
}

void Pi05DenoiseStepModel::adarms_gemv_to_spm(uint32_t cond_spm_addr,
                                              const at::Tensor& dense_w,
                                              const at::Tensor& dense_b,
                                              uint32_t gemv_out_spm_addr,
                                              uint32_t bias_temp_spm_addr,
                                              uint32_t partial_spm_addr,
                                              const at::Tensor& dense_scale)
{
    int64_t h = hidden_size();
    int64_t out_features = 3 * h;

    // Step 1: DMA bias -> bias_temp (replicated across all 8 cores, acts
    // as the residual input to the fused all_reduce+bias kernel).
    rpu_launch_ddr_broadcast_spm_dma(
        dense_b.data_ptr<c10::Half>(), out_features, bias_temp_spm_addr);
    // Step 2: Row-partition GEMV. partial_spm_addr holds the per-core
    // partial [3H] along the local_k = H/NUM_CORES split.
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        cond_spm_addr, dense_w, partial_spm_addr,
        /*M=*/1, /*N=*/out_features, /*K=*/h,
        /*partition=*/0, /*num_cores=*/NUM_CORES,
        /*bias_spm_addr=*/0,
        /*force_gemm=*/false,
        /*scale=*/dense_scale);  // W8A16 when defined; fp16 when empty
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

}  // namespace v3

// ============================================================================
// Registry + C API
// ============================================================================
using Pi05DenoiseStepRegistry = ModelHandleRegistry<v3::Pi05DenoiseStepModel>;

int64_t rpu_pi05_denoise_step_create()        { return Pi05DenoiseStepRegistry::create(); }
void    rpu_pi05_denoise_step_destroy(int64_t h) {
    Pi05DenoiseStepRegistry::destroy(h, "rpu_pi05_denoise_step_destroy");
}

void rpu_pi05_denoise_step_set_rope_position(int64_t handle,
                                            int64_t position) {
    Pi05DenoiseStepRegistry::get(
        handle, "rpu_pi05_denoise_step_set_rope_position")
        ->set_rope_position(position);
}

void rpu_pi05_denoise_step_set_chunk_size(int64_t handle,
                                         int64_t chunk_size) {
    Pi05DenoiseStepRegistry::get(
        handle, "rpu_pi05_denoise_step_set_chunk_size")
        ->set_configured_chunk_size(chunk_size);
}

int64_t rpu_pi05_denoise_step_get_resolved_chunk_size(int64_t handle) {
    return Pi05DenoiseStepRegistry::get(
        handle, "rpu_pi05_denoise_step_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

void rpu_pi05_denoise_step_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w,
    at::TensorList v_w, at::TensorList o_w,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList attn_dense_w, at::TensorList attn_dense_b,
    at::TensorList mlp_dense_w,  at::TensorList mlp_dense_b,
    const at::Tensor& final_norm_dense_w, const at::Tensor& final_norm_dense_b,
    const at::Tensor& action_in_proj_w,   const at::Tensor& action_in_proj_b,
    const at::Tensor& action_out_proj_w,  const at::Tensor& action_out_proj_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t hidden_size, int64_t max_action_dim, int64_t chunk_size,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t num_layers, double eps,
    at::TensorList q_w_scale, at::TensorList k_w_scale,
    at::TensorList v_w_scale, at::TensorList o_w_scale,
    at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale,
    at::TensorList attn_dense_w_scale, at::TensorList mlp_dense_w_scale,
    const at::Tensor& final_norm_dense_w_scale)
{
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_set_weights")
        ->set_weights(q_w, k_w, v_w, o_w, gate_w, up_w, down_w,
                      attn_dense_w, attn_dense_b,
                      mlp_dense_w,  mlp_dense_b,
                      final_norm_dense_w, final_norm_dense_b,
                      action_in_proj_w,   action_in_proj_b,
                      action_out_proj_w,  action_out_proj_b,
                      cos, sin, hidden_size, max_action_dim, chunk_size,
                      num_q_heads, num_kv_heads, head_dim, num_layers, eps,
                      q_w_scale, k_w_scale, v_w_scale, o_w_scale,
                      gate_scale, up_scale, down_scale,
                      attn_dense_w_scale, mlp_dense_w_scale,
                      final_norm_dense_w_scale);
}

void rpu_pi05_denoise_step_forward(
    int64_t handle,
    const at::Tensor& x_t_rpu,
    std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches,
    const at::Tensor& cond_step,
    const at::Tensor& attention_mask_4d,
    at::Tensor v_t_buf,
    int64_t prefix_len)
{
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_step_forward")
        ->step_forward(x_t_rpu, k_caches, v_caches, cond_step,
                       attention_mask_4d, v_t_buf, prefix_len);
}

void rpu_pi05_denoise_loop_forward(
    int64_t handle,
    at::Tensor x0_rpu,
    std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches,
    at::Tensor cond_all,
    at::Tensor attention_mask_4d,
    at::Tensor x_out,
    double dt,
    int64_t prefix_len,
    int64_t num_steps)
{
    Pi05DenoiseStepRegistry::get(handle, "rpu_pi05_denoise_loop_forward")
        ->denoise_loop_forward(x0_rpu, k_caches, v_caches, cond_all,
                               attention_mask_4d, x_out, dt, prefix_len, num_steps);
}
