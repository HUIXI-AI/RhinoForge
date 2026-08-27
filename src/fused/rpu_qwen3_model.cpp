// rpu_qwen3_model.cpp — Qwen3 all-layers-once FusedModelBase implementation.
// It uses typed SPM offsets at attention/cache boundaries and invalidates model
// state after every weight or LM-head update.
//
// Scope (preserved from v2):
//   - fuse_lm_head = false by default (lm_head computed by outer HF Qwen3ForCausalLM.lm_head).
//   - final_norm fused into last layer's build_layer_subgraph (SigLIP pattern);
//     no post_fn is configured.
//   - is_causal = true only (Qwen3 fused SDPA is causal-only).
//   - batch_size == 1 except explicitly admitted plain-Qwen3 causal decode;
//     batched prefill is emitted one equal-length sequence/cache slot at a time.
//
// Framework contract: docs/architecture.md#fusedmodelbase-v3--framework-contract

#include "rpu_qwen3_model.h"  // v3::CausalDecoderModel and constants
#include "model_handle_registry.h"

using namespace at;
using namespace ::rhino_lkn;

// =============================================================================
// Instance registry — ModelHandleRegistry<v3::CausalDecoderModel>
//
// Handle 单调递增且不复用;GraphCache 由 Python adapter 按模型实例持有。
// Thread-safety / 并发 destroy 限制见 model_handle_registry.h 注释。
// =============================================================================

using CausalDecoderRegistry = ModelHandleRegistry<v3::CausalDecoderModel>;


// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

int64_t rpu_causal_decoder_create() {
    return CausalDecoderRegistry::create();
}

void rpu_causal_decoder_destroy(int64_t handle) {
    CausalDecoderRegistry::destroy(handle, "rpu_causal_decoder_destroy");
}

void rpu_causal_decoder_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::IntArrayRef deepstack_lang_layers,
    const std::optional<std::vector<at::Tensor>>& q_bias_list,
    const std::optional<std::vector<at::Tensor>>& k_bias_list,
    const std::optional<std::vector<at::Tensor>>& v_bias_list)
{
    // Optional Tensor[] schema args arrive as std::optional<vector>; unwrap to
    // TensorList (empty when None) for the model's set_weights.
    at::TensorList q_bias = q_bias_list ? at::TensorList(*q_bias_list) : at::TensorList{};
    at::TensorList k_bias = k_bias_list ? at::TensorList(*k_bias_list) : at::TensorList{};
    at::TensorList v_bias = v_bias_list ? at::TensorList(*v_bias_list) : at::TensorList{};
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        q_norm_list, k_norm_list,
        input_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        eps, use_silu,
        mrope_section,
        deepstack_lang_layers,
        q_bias, k_bias, v_bias,
        {}, {}, {}, {}, {}, {}, {});
}

void rpu_causal_decoder_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::IntArrayRef deepstack_lang_layers,
    at::TensorList q_w_scale_list,
    at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list,
    at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list,
    at::TensorList up_scale_list,
    at::TensorList down_scale_list,
    const std::optional<std::vector<at::Tensor>>& q_bias_list,
    const std::optional<std::vector<at::Tensor>>& k_bias_list,
    const std::optional<std::vector<at::Tensor>>& v_bias_list)
{
    at::TensorList q_bias = q_bias_list ? at::TensorList(*q_bias_list) : at::TensorList{};
    at::TensorList k_bias = k_bias_list ? at::TensorList(*k_bias_list) : at::TensorList{};
    at::TensorList v_bias = v_bias_list ? at::TensorList(*v_bias_list) : at::TensorList{};
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        q_norm_list, k_norm_list,
        input_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        eps, use_silu,
        mrope_section,
        deepstack_lang_layers,
        q_bias, k_bias, v_bias,
        q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list);
}

void rpu_causal_decoder_set_weights_nvfp4(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::IntArrayRef deepstack_lang_layers,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list,
    const at::Tensor& q_tensor_scales, const at::Tensor& k_tensor_scales,
    const at::Tensor& v_tensor_scales, const at::Tensor& o_tensor_scales,
    const at::Tensor& gate_tensor_scales, const at::Tensor& up_tensor_scales,
    const at::Tensor& down_tensor_scales,
    const std::optional<std::vector<at::Tensor>>& q_bias_list,
    const std::optional<std::vector<at::Tensor>>& k_bias_list,
    const std::optional<std::vector<at::Tensor>>& v_bias_list)
{
    at::TensorList q_bias = q_bias_list ? at::TensorList(*q_bias_list) : at::TensorList{};
    at::TensorList k_bias = k_bias_list ? at::TensorList(*k_bias_list) : at::TensorList{};
    at::TensorList v_bias = v_bias_list ? at::TensorList(*v_bias_list) : at::TensorList{};
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        q_norm_list, k_norm_list,
        input_norm_list, post_norm_list,
        gate_list, up_list, down_list,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        eps, use_silu,
        mrope_section, deepstack_lang_layers,
        q_bias, k_bias, v_bias,
        q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list,
        q_tensor_scales, k_tensor_scales, v_tensor_scales, o_tensor_scales,
        gate_tensor_scales, up_tensor_scales, down_tensor_scales);
}

// Phase 2.5: per-instance fused lm_head 入口
void rpu_causal_decoder_set_lm_head(
    int64_t handle,
    const at::Tensor& lm_head_w,
    const std::optional<at::Tensor>& lm_head_scale) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")
        ->set_lm_head(lm_head_w, lm_head_scale);
}

void rpu_causal_decoder_clear_lm_head(int64_t handle) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->clear_lm_head();
}

void rpu_causal_decoder_set_linear_acc32(int64_t handle, bool enabled) {
    CausalDecoderRegistry::get(
        handle, "rpu_causal_decoder_set_linear_acc32")
        ->set_linear_acc32(enabled);
}

// RhinoVLA text-prefill fast-replay opt-ins (per-model).
void rpu_causal_decoder_set_fast_replay(int64_t handle, bool enabled) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")
        ->set_fast_replay_skip_layer_loop(enabled);
}

void rpu_causal_decoder_set_preload_replay_skip(int64_t handle, bool enabled) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder")
        ->set_preload_replay_skip(enabled);
}

void rpu_causal_decoder_set_chunk_size_cap(int64_t handle, int64_t chunk_size_cap) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_chunk_size_cap")
        ->set_configured_chunk_size_cap(chunk_size_cap);
}

// Per-handle replacement for the process-global chunk-size override.
//
// Deliberately NOT cold-only, unlike the cap: the perf harness sweeps chunk legs
// on a live model, and the resolved chunk is part of the prefill graph
// signature, so a change lands as a SECOND cache entry instead of a stale replay.
// For the same reason it must NOT invalidate the graph cache — that would trade a
// wrong answer for a rebuild. Distinct resolved chunks must produce distinct
// signatures without recapturing an existing signature.
void rpu_causal_decoder_set_chunk_size_override(int64_t handle, int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0
                    || (chunk_size >= 16 && chunk_size % 16 == 0),
                "causal_decoder_set_chunk_size_override: chunk_size must be 0 "
                "(auto) or a positive multiple of 16, got ", chunk_size);
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_chunk_size_override")
        ->set_chunk_size_override(chunk_size);
}

int64_t rpu_causal_decoder_get_chunk_size_override(int64_t handle) {
    return CausalDecoderRegistry::get(
               handle, "rpu_causal_decoder_get_chunk_size_override")
        ->get_chunk_size_override();
}

// Declare this handle's certified chunk envelope. This is deny-by-default:
// without an envelope the planner refuses to prefill.
void rpu_causal_decoder_set_chunk_envelope(int64_t handle, int64_t max_kv_len,
                                           int64_t chunk) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_causal_decoder_set_equal_two_prefill(int64_t handle, bool enabled) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_equal_two_prefill")
        ->set_equal_two_prefill(enabled);
}

// AdaRMS FiLM step-update (LingBot-VLA action expert). Per Euler denoise step,
// the host passes the folded per-layer scale (= (1+γ(cond))·rms_weight, rides on
// the input/post norm-weight slots) + shift (= β(cond)). See rpu_qwen3_model.h
// set_adarms_step. Enables the gated shift-add in build_layer_subgraph.
void rpu_causal_decoder_set_adarms_step(
    int64_t handle,
    at::TensorList input_scales, at::TensorList input_shifts,
    at::TensorList post_scales,  at::TensorList post_shifts) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_adarms_step")
        ->set_adarms_step(input_scales, input_shifts, post_scales, post_shifts);
}

// Replay-safe variant: each arg is one [num_layers, hidden] fp16 RPU tensor (not a list).
// First call builds the mutable graph; subsequent calls refresh keepalives in place (no
// rebuild) so the expert graph replays across Euler steps. See rpu_qwen3_model.h.
void rpu_causal_decoder_set_adarms_step_mutable(
    int64_t handle,
    const at::Tensor& input_scale, const at::Tensor& input_shift,
    const at::Tensor& post_scale,  const at::Tensor& post_shift) {
    CausalDecoderRegistry::get(handle, "rpu_causal_decoder_set_adarms_step_mutable")
        ->set_adarms_step_mutable(input_scale, input_shift, post_scale, post_shift);
}

at::Tensor rpu_causal_decoder_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    const std::optional<at::Tensor>& position_ids,
    const std::optional<std::vector<at::Tensor>>& deepstack_dense_visual_embeds,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il,
    int64_t cos_sin_offset,
    int64_t batch_slot,
    bool allow_batch_decode)
{
    // TensorList → std::vector<at::Tensor> (shallow copy of refcounted tensors)
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());

    return CausalDecoderRegistry::get(handle, "rpu_causal_decoder")->forward(
        hidden_states, k_caches, v_caches, attention_mask, position, is_causal,
        position_ids, deepstack_dense_visual_embeds, rope_cos_il, rope_sin_il,
        cos_sin_offset, batch_slot, allow_batch_decode);
}

// Return the per-handle resolved chunk size.
// Returns 0 if no forward has run yet for this handle (sentinel default).
int64_t rpu_causal_decoder_get_resolved_chunk_size(int64_t handle) {
    return CausalDecoderRegistry::get(handle, "rpu_causal_decoder_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

int64_t rpu_causal_decoder_resolve_prefill_chunk_size(
    int64_t handle, int64_t execution_len, int64_t position) {
    return CausalDecoderRegistry::get(
        handle, "rpu_causal_decoder_resolve_prefill_chunk_size")
        ->resolve_prefill_chunk_size(execution_len, position);
}
