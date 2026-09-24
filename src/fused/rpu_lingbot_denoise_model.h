// rpu_lingbot_denoise_model.h — fused LingBot-VLA action-expert denoise unroll op.
//
// ONE graph (1 BUILD + N-1 REPLAY) folds the 10-step Euler flow-matching denoise of
// the LingBot-VLA action expert. Inherits the plain Qwen2.5 (768-d) causal decoder
// body from v3::CausalDecoderModel with per-layer AdaRMS FiLM (set_adarms_unroll:
// the fold is read at a body_iter offset), and adds pi05/wall-oss-style pre/post
// layer hooks that run the fused suffix encoder + out_proj + in-graph Euler on-device,
// so the noisy action x never leaves the RPU between steps.
//
// Deltas vs WallOssActionStepModel (the structural template):
//   - Encoder = fused [action_dim→768] W_comp GEMM + per-step time bias + SiLU +
//     action_time_mlp_out GEMM (NOT wall-oss's W_comb+B_step+w3+w2 machinery).
//   - Uniform-51 layout: x_t_spm / encoder / out_proj / Euler all run on cs=51 rows;
//     row 0 carries the state token (emb_stage_[0] = state_emb, set once per call) and
//     is never output (adapter returns x_traj[-1][:, 1:51, :action_dim]). The pre-hook
//     copies encoder rows 1:51 into emb_stage_ (a +1-row byte offset), leaving row 0.
//   - AdaRMS per-step at body_iter offset (set_adarms_unroll; delta 4).
//   - Plain 1D-RoPE (mrope_section=[]) — no wall-oss partial_mrope plumbing.
//
// Gated by RPU_LINGBOT_DENOISE_UNROLL (Python side; default OFF). See
#pragma once

#include "rpu_qwen3_model.h"     // v3::CausalDecoderModel
#include "rpu_kernel_decls.h"
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

namespace v3 {

class LingbotDenoiseModel : public CausalDecoderModel {
public:
    LingbotDenoiseModel();
    ~LingbotDenoiseModel() override;

    // Bind the fused suffix encoder + out_proj (single-core col-swizzled, fp16) plus
    // the per-step time bias table. Must be called AFTER CausalDecoderModel::set_weights
    // (needs num_layers()/hidden_size()) and BEFORE set_adarms_unroll.
    //   wc        [hidden, k_comb_pad]   single-core col-swizzled fp16 (W_comp, K padded)
    //   mo        [hidden, hidden]       single-core col-swizzled fp16 (action_time_mlp_out)
    //   mo_bias   [hidden]               fp16 RPU
    //   op        [action_dim_pad, hidden] single-core col-swizzled fp16 (action_out_proj, out padded)
    //   op_bias   [action_dim_pad]       fp16 RPU (out padded, tail zero)
    //   time_all  [num_steps, hidden]    fp16 RPU (per-step W_comp GEMM bias)
    void set_action_weights(
        const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
        const at::Tensor& op, const at::Tensor& op_bias, const at::Tensor& time_all,
        int64_t action_dim, int64_t k_comb_pad, int64_t num_steps, int64_t chunk_size);

    // Cold execution policy for the fixed 51-row suffix. 0 selects the native
    // planner; 64 is the only exact request because the hooks require one
    // ceil16-aligned chunk.
    void set_execution_chunk_size(int64_t chunk_size);

    // Resolve the fixed masked action suffix through the same native A6
    // authority consumed by unroll_forward.  The public wrapper deliberately
    // exposes no second parser or scalar fallback.
    std::vector<int64_t> resolve_action_stage_domain(
        int64_t execution_len, int64_t prefix_len, int64_t mask_kv_len);

    // N-step in-graph unroll. ONE forward emits num_steps × [pre-encoder → 36 layers →
    // out_proj → Euler]; x persists in x_t_spm across iterations; final x = x_traj[-1].
    void unroll_forward(
        const at::Tensor& x0_rpu,          // [1, cs, k_comb_pad] fp16 RPU (row 0 = 0, rows 1:cs = padded noise)
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& state_emb,       // [hidden] fp16 RPU — emb_stage_ row 0 (set once)
        const std::optional<at::Tensor>& attention_mask,  // [cs, prefix_len+cs] fp16 2D additive
        at::Tensor& x_traj,                // [num_steps, cs, k_comb_pad] fp16 RPU (per-step x)
        double dt, int64_t prefix_len, int64_t num_steps,
        at::IntArrayRef planned_stage_descriptor = {});

protected:
    std::vector<BufferDecl> explicit_mask_baseline_buffer_declarations(
        const LayoutContext& ctx) override;
    int64_t explicit_mask_body_iterations() const override { return num_steps_; }
    ModelStaticConfig       static_config() override;
    ModelDynamicConfig      dynamic_config(const ChunkPlan& plan) override;
    bool subclass_chunk_size_valid(
        int64_t chunk_size, int64_t seq_len,
        int64_t position) const override;
    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override;

    // The two action widths size the appended Temp "x_t_spm" / "v_t_core0_spm" /
    // "op_b_spm". Every decl this subclass appends is Temp, so nothing in the
    // framework would notice them moving. Chained onto the base's contribution.
    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(CausalDecoderModel::subclass_layout_hash(),
                                       k_comb_pad_);
        return detail::layout_mix(h, action_dim_pad_);
    }

private:
    void emit_pre_layers_body();
    void emit_post_layers_body();

    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        auto identity = causal_kvinsert_cost_weight_identity();
        if (identity.empty()) return {};
        for (const auto* tensor : {
                &wc_, &mo_, &mo_bias_, &op_,
                &op_bias_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    at::Tensor wc_, mo_, mo_bias_, op_, op_bias_, time_all_;
    int64_t action_dim_ = 0, action_dim_pad_ = 0, k_comb_pad_ = 0;
    int64_t chunk_size_ = 0, num_steps_ = 0;
    c10::Half dt_ = c10::Half(0.0f);
    bool     dt_pinned_ = false;
    at::Tensor emb_stage_;              // stable staging DDR [1, cs, hidden]; row 0 = state
    uint64_t x_t_src_base_ = 0;         // mutable DMA base for x0 (iter 0 load)
    uint64_t x_traj_dst_base_ = 0;      // mutable DMA base for the per-step x trajectory dst
    at::Tensor x0_ref_, x_traj_ref_, state_ref_;  // keepalives across the synchronous forward
};

}  // namespace v3
