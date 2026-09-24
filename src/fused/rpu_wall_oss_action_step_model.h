// rpu_wall_oss_action_step_model.h — fused Wall-OSS-0.5 action-denoise step op.
//
// One step_forward() = ONE flow-matching denoise step. Inherits the plain Qwen2.5
// causal decoder body from v3::CausalDecoderModel (extracted to rpu_qwen3_model.h);
// adds pi05-style pre/post layer hooks that run the action preprocessor and proj_back
// on-device, so the noisy action `x` never leaves the RPU between steps. The Python
// adapter keeps the 10-step Euler loop (1 BUILD + 9 REPLAY) and the cheap x += dt·v.
//
// Pre-hook: x_t -> (W_comb GEMM + B_step + SiLU + w3 GEMM) -> residual1 (SPM), which
//           layer 0 consumes directly. (preprocessor folded by linearity;
//           see the buffer and execution contracts below.)
// Post-hook: proj_back on the final-normed residual1 -> v_t_buf (DDR).
//
// Gated by RPU_WALL_OSS_FUSED_DENOISE (Python side). See the plan for the codex review
// resolutions folded into the design (base forward() reuse, buffer phases, chunk pin).
#pragma once

#include "rpu_qwen3_model.h"     // v3::CausalDecoderModel (extracted in Task 3)
#include "rpu_kernel_decls.h"
#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

namespace v3 {

class WallOssActionStepModel : public CausalDecoderModel {
public:
    WallOssActionStepModel();
    ~WallOssActionStepModel() override;

    std::vector<int64_t> resolve_action_stage_domain(
        int64_t horizon, int64_t prefix_len, int64_t mask_kv_len,
        int64_t requested_chunk_size, bool loop_mode, int64_t num_steps,
        bool b_step_is_bias, bool have_te);
    void stage_execution_chunk_size(uint64_t token, int64_t chunk_size);

    // Bind the action preprocessor/proj_back weights (fp16; w3 is TP8 col-swizzled,
    // the remaining projections are single-core col-swizzled)
    // plus the padded dims. Must be called AFTER CausalDecoderModel::set_weights.
    void set_action_weights(
        const at::Tensor& w_comb,      // [hidden, k_comb_pad]   single-core col-swizzled fp16
        const at::Tensor& w3,          // [hidden, hidden]       TP8 col-swizzled fp16
        const at::Tensor& proj_back,   // [action_dim_pad, hidden] single-core col-swizzled fp16
        const at::Tensor& w2_t,        // [hidden, hidden] single-core col-swizzled fp16 — the
                                       // preprocessor time-projection (w2[:,AH:]); enables the
                                       // on-device ct = te·w2_t when step_forward gets te_step.
        const at::Tensor& w2_a,        // [hidden, hidden] single-core col-swizzled fp16 — the
                                       // preprocessor action-projection (w2[:,:AH]); enables the
                                       // on-device base = ae_dof·w2_a (b_step then carries ae_dof).
        int64_t action_dim, int64_t k_comb_pad, int64_t chunk_size);

    // ONE denoise step. Sets per-step mutable DMA bases then runs the inherited
    // CausalDecoderModel::forward over emb_stage_ (mRoPE + mask + run_all_layers),
    // with the hooks firing via virtual dispatch.
    at::Tensor step_forward(
        const at::Tensor& x_t_rpu,         // [1,H,k_comb_pad] fp16 RPU (zero-padded)
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& b_step,          // [hidden] (row-constant) or [H,hidden] fp16 RPU, per step.
                                           // = base (const part) when te_step is given, else full B_step.
        const std::optional<at::Tensor>& te_step,  // [hidden] fp16 RPU time emb (row-constant/bias mode
                                           // only). When set: ct = te_step·w2_t added on-device into the
                                           // bias before the W_comb GEMM. None -> b_step is the full bias.
        const std::optional<at::Tensor>& attention_mask,  // [H,P+H] fp16 2D additive
        const at::Tensor& position_ids,    // [H,3] int32 mRoPE (REQUIRED)
        at::Tensor& v_t_buf,               // [1,H,action_dim_pad] fp16 RPU
        at::Tensor& x_out_buf,             // [1,H,k_comb_pad] fp16 RPU — in-graph Euler result x+dt·v
        double dt,                         // Euler step (constant across steps; baked at BUILD)
        int64_t prefix_len,
        const std::optional<at::Tensor>& rope_cos_il = std::nullopt,  // partial_mrope interleaved cos
        const std::optional<at::Tensor>& rope_sin_il = std::nullopt,  // partial_mrope interleaved sin
        at::IntArrayRef planned_stage_descriptor = {});

    // N-step in-graph unroll: ONE graph emits num_steps × [pre → layers → post].
    // x persists in x_t_spm across iterations; bias read at baked per-iteration
    // offsets; final x = x_traj[-1]. b_all/te_all mirror step_forward's b_step/te_step:
    //   te_all present → row-constant: b_all = ae_dof [hidden] (const), te_all =
    //                    [num_steps, hidden]   (the on-device ct path).
    //   te_all absent  → full: b_all = [num_steps, H, hidden] (per-step B_step add).
    void denoise_loop_forward(
        const at::Tensor& x0_rpu,          // [1,H,k_comb_pad] fp16 RPU (zero-padded noise)
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& b_all,
        const std::optional<at::Tensor>& te_all,
        const std::optional<at::Tensor>& attention_mask,  // [H,P+H] fp16 2D additive
        const at::Tensor& position_ids,    // [H,3] int32 mRoPE (REQUIRED)
        at::Tensor& x_traj,                // [num_steps,H,k_comb_pad] fp16 RPU (per-step x)
        at::Tensor& v_traj,                // [num_steps,H,action_dim_pad] fp16 RPU (per-step v_t)
        double dt, int64_t prefix_len, int64_t num_steps,
        const std::optional<at::Tensor>& rope_cos_il = std::nullopt,  // partial_mrope interleaved cos
        const std::optional<at::Tensor>& rope_sin_il = std::nullopt,  // partial_mrope interleaved sin
        at::IntArrayRef planned_stage_descriptor = {});

protected:
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            planning_rope_mode_, planning_graph_lifecycle_,
            planning_mode_, planned_action_route_profile_valid_,
            planned_loop_mode_, planned_num_steps_,
            planned_b_step_is_bias_, planned_have_te_);
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    ModelStaticConfig       static_config() override;
    ModelDynamicConfig      dynamic_config(const ChunkPlan& plan) override;
    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan, const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override;
    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& manifest) const override;
    // build_layer_subgraph / forward: INHERITED from CausalDecoderModel.

    // The two action widths size the appended Temp "x_t_spm" / "v_t_core0_spm",
    // and NO Persistent decl here depends on them, so nothing in the framework
    // would notice them moving. Chained onto the base's contribution.
    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(CausalDecoderModel::subclass_layout_hash(),
                                       k_comb_pad_);
        return detail::layout_mix(h, action_dim_pad_);
    }

private:
    void emit_pre_layers_body();
    void emit_post_layers_body();
    void consume_action_route(
        FmbRouteFamily family, int64_t site_id, int64_t selector,
        std::vector<int64_t> arguments = {});
    void validate_planned_action_route_profile(
        at::IntArrayRef descriptor, bool loop_mode, int64_t num_steps,
        bool b_step_is_bias, bool have_te);

    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        auto identity = causal_kvinsert_cost_weight_identity();
        if (identity.empty()) return {};
        for (const auto* tensor : {
                &w_comb_, &w3_, &proj_back_, &w2_t_,
                &w2_a_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    at::Tensor w_comb_, w3_, proj_back_, w2_t_, w2_a_;
    int64_t action_dim_ = 0, action_dim_pad_ = 0, k_comb_pad_ = 0, chunk_size_ = 0;
    int64_t configured_chunk_size_ = 0;  // 0=auto; otherwise exact physical ceil16(H)
    uint64_t x_t_src_base_ = 0, b_step_src_base_ = 0, v_t_dst_base_ = 0, x_out_dst_base_ = 0;
    uint64_t te_src_base_ = 0;        // mutable DMA base for the per-step te_step input
    c10::Half dt_ = c10::Half(0.0f);  // in-graph Euler step, baked at BUILD; asserted stable
    bool    dt_pinned_ = false;
    bool b_step_is_bias_ = false;     // [hidden] row-constant variant -> GEMM bias
    int  b_step_mode_ = -1;           // -1 unset / 0 add / 1 bias; asserted stable across steps
    int  have_te_ = -1;               // -1 unset / 0 no te / 1 on-device ct; baked at BUILD, stable
    at::Tensor emb_stage_;            // stable staging DDR (allocated in set_action_weights)
    // keepalive across the synchronous forward — one per mutable base above, so
    // the deferred graph can never dereference a freed VA (pitfalls.md C-1).
    at::Tensor x_t_ref_, b_step_ref_, te_ref_, v_t_ref_, x_out_ref_;
    // EXT-unroll (denoise loop). loop_mode_ gates the trajectory DMAs + the
    // body_iter==0 x0-load skip; false (single-step) keeps emit_*_body bit-identical.
    bool     loop_mode_   = false;
    int64_t  num_steps_   = 1;        // body_iterations when loop_mode_
    uint64_t x_traj_dst_base_ = 0, v_traj_dst_base_ = 0;          // mutable trajectory dst bases
    at::Tensor b_all_ref_, te_all_ref_, x_traj_ref_, v_traj_ref_; // keepalive across the forward

    // The planner resolves before forward binds request tensors.  Keep that
    // branch profile separate from the live graph state above: mutating
    // loop_mode_/num_steps_ here would hide the forward's rebuild transition.
    bool planned_action_route_profile_valid_ = false;
    bool planned_loop_mode_ = false;
    int64_t planned_num_steps_ = 1;
    bool planned_b_step_is_bias_ = false;
    bool planned_have_te_ = false;
};

}  // namespace v3

// The C API free-function prototypes (rpu_wall_oss_action_step_{create,destroy,
// set_weights,forward}) live in rpu_kernel_decls.h alongside the other fused-model
// C entries (mirrors the pi05_denoise_step pattern), so rpu_backend.cpp can bind
// them via TORCH_FN without pulling this heavy header.
