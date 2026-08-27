// src/fused/rpu_pi05_denoise_step_model.h
//
// Pi0.5 num_steps fused denoise step model (FusedModelBase v3 subclass).
//
// Architecture:
//   - Inherits v3::FusedModelBase; reuses SPM allocator + GraphCache lifecycle.
//   - hidden_size_ = H_ada = 1024 (AdaRMS gemma_300m hidden).
//   - Injects action_in_proj (pre) + final PiGemmaRMSNorm + action_out_proj
//     (post) via pre_layers_fn / post_layers_fn hooks.
//   - One step_forward() = ONE denoise step. Python adapter loops num_steps
//     times wrapped in cache.capture(sig) — 1 BUILD + (N-1) REPLAY.
//   - All per-step mutables (x_t src, cond src, v_t dst) routed via *_mutable
//     DMA wrappers (no FIXED-first scaffolding).

#pragma once

#include "fused_model_base.h"
#include "rpu_kernel_decls.h"

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <cstdint>
#include <vector>

namespace v3 {

class Pi05DenoiseStepModel : public FusedModelBase {
public:
    // LayerWeights follows the AdaRMSModel fields used by
    // build_layer_subgraph.
    struct LayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor gate_proj_w, up_proj_w, down_proj_w;
        at::Tensor q_ws, k_ws, v_ws, o_ws;
        at::Tensor gate_ws, up_ws, down_ws;
        // Per-layer AdaRMS dense (row-partition swizzled by Python; reuse
        // the _rpu_dense_w_rp / _rpu_dense_b_rp buffers set by
        // adarms.py::_prepare_adarms_dense_weights).
        at::Tensor attn_dense_w, attn_dense_b;
        at::Tensor mlp_dense_w,  mlp_dense_b;
        // W8A16 per-channel scales for the AdaRMS dense GEMV (empty ⇒ fp16).
        at::Tensor attn_dense_ws, mlp_dense_ws;
    };

    Pi05DenoiseStepModel();
    ~Pi05DenoiseStepModel() override;

    void set_rope_position(int64_t position);
    void set_configured_chunk_size(int64_t chunk_size);

    void set_weights(
        at::TensorList q_w, at::TensorList k_w,
        at::TensorList v_w, at::TensorList o_w,
        at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
        at::TensorList attn_dense_w, at::TensorList attn_dense_b,
        at::TensorList mlp_dense_w,  at::TensorList mlp_dense_b,
        const at::Tensor& final_norm_dense_w, const at::Tensor& final_norm_dense_b,
        const at::Tensor& action_in_proj_w,  const at::Tensor& action_in_proj_b,
        const at::Tensor& action_out_proj_w, const at::Tensor& action_out_proj_b,
        const at::Tensor& cos, const at::Tensor& sin,
        int64_t hidden_size, int64_t max_action_dim, int64_t chunk_size,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t num_layers, double eps,
        at::TensorList q_w_scale, at::TensorList k_w_scale,
        at::TensorList v_w_scale, at::TensorList o_w_scale,
        at::TensorList gate_scale, at::TensorList up_scale,
        at::TensorList down_scale,
        at::TensorList attn_dense_w_scale, at::TensorList mlp_dense_w_scale,
        const at::Tensor& final_norm_dense_w_scale);

    // v_t_buf is written in place; returns void.
    void step_forward(
        const at::Tensor& x_t_rpu,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cond_step,
        const at::Tensor& attention_mask_4d,
        at::Tensor& v_t_buf,
        int64_t prefix_len);

    // In-graph N-step unroll, mirroring WallOssActionStepModel.
    // loop_mode_ off => body_iterations==1 and the single-step path is
    // byte-identical; loop mode emits the on-device Euler update per iteration.
    void denoise_loop_forward(
        at::Tensor x0_rpu, std::vector<at::Tensor> k_caches,
        std::vector<at::Tensor> v_caches, at::Tensor cond_all,
        at::Tensor attention_mask_4d, at::Tensor x_out,
        double dt, int64_t prefix_len, int64_t num_steps);

protected:
    // FMB v3 mandatory virtuals
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    ModelStaticConfig       static_config() override;
    ModelDynamicConfig      dynamic_config(const ChunkPlan& plan) override;
    void                    build_layer_subgraph(int layer_idx,
                                                 const ChunkInfo& chunk) override;

    // max_action_dim_ sizes the Temp "x_t_spm" / "v_t_core0_spm" and is invisible
    // to the framework's params hash. The Persistent "action_out_proj_b_spm" it
    // also sizes would NOT catch a change: Align(dim*2, 256) buckets every
    // max_action_dim_ in [1,128] to the same 256 bytes, so the persistent-shape
    // guard stays silent while the two Temps keep the old size.
    int64_t subclass_layout_hash() const override {
        return detail::layout_mix(0, max_action_dim_);
    }

private:
    void emit_pre_layers_body();
    void emit_post_layers_body();

    // Shared SDPA mask prepare + cache (mirrors AdaRMSModel). Called by both
    // step_forward and denoise_loop_forward so the prepared-mask logic lives in
    // exactly one place. Populates prepared_mask_ / prepared_mask_seq_*_.
    void prepare_sdpa_mask_cached(const at::Tensor& attention_mask_4d,
                                  int64_t prefix_len);

    // Local copy of AdaRMSModel::adarms_gemv_to_spm.
    void adarms_gemv_to_spm(uint32_t cond_spm_addr,
                            const at::Tensor& dense_w,
                            const at::Tensor& dense_b,
                            uint32_t gemv_out_spm_addr,
                            uint32_t bias_temp_spm_addr,
                            uint32_t partial_spm_addr,
                            const at::Tensor& dense_scale = {});

    // --------- Weights (DDR, set_weights owns lifetime) ----------
    std::vector<LayerWeights> layer_weights_;
    at::Tensor final_norm_dense_w_, final_norm_dense_b_;
    at::Tensor final_norm_dense_ws_;  // W8A16 scale (empty ⇒ fp16)
    at::Tensor action_in_proj_w_,  action_in_proj_b_;
    at::Tensor action_out_proj_w_, action_out_proj_b_;
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
    int64_t configured_chunk_size_ = 0;
    double eps_ = 1e-6;

    // --------- Per-call inputs (set in step_forward) -------------
    // Keepalives for the three mutable bases below. The bases are dereferenced
    // when the graph executes, after this model's forward has returned.
    at::Tensor x_t_ref_, cond_ref_, v_t_ref_;

    // --------- Mutable bases (rewritten per step_forward) --------
    // Read by *_mutable DMA wrappers at REPLAY time via update_dma_kernel.
    uint64_t x_t_src_base_  = 0;
    uint64_t cond_src_base_ = 0;
    uint64_t v_t_dst_base_  = 0;

    // --------- In-graph N-step unroll -------
    // loop_mode_ off => body_iterations==1 and the single-step path is
    // byte-identical; loop mode enables the on-device Euler and per-iteration DMAs.
    bool      loop_mode_ = false;
    int64_t   num_steps_ = 1;
    c10::Half dt_;                 // baked at BUILD (x_t += dt*v_t); const across REPLAY
    bool      dt_pinned_ = false;
    uint64_t  x0_src_base_ = 0;        // initial noise DDR base (iter 0 load)
    uint64_t  cond_all_src_base_ = 0;  // [num_steps, h_ada] cond stack DDR base
    uint64_t  x_out_dst_base_ = 0;     // final action DDR base
    // Keepalives (prevent GC mid-forward).
    at::Tensor x0_ref_, cond_all_ref_, x_out_ref_;

    // --------- Stable RPU staging tensor (set_weights allocates) ----
    // data_ptr remains stable across step / sample_actions. Passed to
    // run_all_layers as hidden_states to satisfy FMB hidden_size_=1024
    // shape check; layer 0's emit_layer_input_dma reads from it via the
    // FMB-managed hidden_in_src_base_ mutable mechanism, so no separate
    // broadcast is needed.
    at::Tensor action_emb_stage_;

    // --------- Cached dims (set in set_weights) -------------------
    int64_t max_action_dim_ = 0;
    int64_t chunk_size_     = 0;
    int64_t local_q_heads_  = 0;
    int64_t local_kv_dim_   = 0;

    // --------- SDPA mask cache (mirrors AdaRMSModel pattern) ------
    at::Tensor prepared_mask_input_ref_;
    PreparedMask prepared_mask_;  // global struct from rpu_kernel_decls.h
    int64_t prepared_mask_seq_q_ = 0;
    int64_t prepared_mask_seq_k_ = 0;
    bool    prepared_mask_is_causal_ = false;
};

}  // namespace v3

// ============================================================================
// Public C API (file-scope; mirrors rpu_pi05_create / set_weights / forward).
// ============================================================================

int64_t rpu_pi05_denoise_step_create();
void    rpu_pi05_denoise_step_destroy(int64_t handle);

void rpu_pi05_denoise_step_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w,
    at::TensorList v_w, at::TensorList o_w,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList attn_dense_w, at::TensorList attn_dense_b,
    at::TensorList mlp_dense_w,  at::TensorList mlp_dense_b,
    const at::Tensor& final_norm_dense_w, const at::Tensor& final_norm_dense_b,
    const at::Tensor& action_in_proj_w,  const at::Tensor& action_in_proj_b,
    const at::Tensor& action_out_proj_w, const at::Tensor& action_out_proj_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t hidden_size, int64_t max_action_dim, int64_t chunk_size,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t num_layers, double eps,
    at::TensorList q_w_scale, at::TensorList k_w_scale,
    at::TensorList v_w_scale, at::TensorList o_w_scale,
    at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale,
    at::TensorList attn_dense_w_scale, at::TensorList mlp_dense_w_scale,
    const at::Tensor& final_norm_dense_w_scale);

void rpu_pi05_denoise_step_forward(
    int64_t handle,
    const at::Tensor& x_t_rpu,
    std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches,
    const at::Tensor& cond_step,
    const at::Tensor& attention_mask_4d,
    at::Tensor v_t_buf,
    int64_t prefix_len);
