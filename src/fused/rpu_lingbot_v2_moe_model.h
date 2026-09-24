// rpu_lingbot_v2_moe_model.h — LingBot2 (V2) sparse-MoE action-expert decoder.
//
// v3::LingbotV2MoeExpertModel inherits the ENTIRE proven v3::CausalDecoderModel
// scaffolding (Qwen2.5 768-d decoder body, AdaRMS FiLM, KV lifecycle, explicit
// 2D-mask SDPA) and overrides ONLY declare_buffers + build_layer_subgraph to
// emit a sparse-MoE MLP in place of the dense MLP.
//
// ─── Scope (deliberately narrow — see the header comment block in the .cpp) ───
//   - A newly constructed native handle starts with body_iterations=1 and can
//     drive the 10-step Euler loop from Python. The ordinary LingBot2 facade now
//     binds the FP16-resident action head and selects the 10-step in-graph
//     unroll by default; an exact per-policy unroll=0 restores the host loop.
//     The unroll reuses existing ACC32 Linear / SPM elementwise kernels; no
//     arithmetic kernel is added here.
//   - This model does NOT inherit LingbotDenoiseModel: that class owns the V1
//     dense decoder body.  Only its pre/post-hook structure is mirrored. V1 is
//     untouched.
//
// ─── Default OFF ───
//   Nothing in the existing code paths references this class. It is reachable
//   only through the lingbot_v2_moe_* ops, which only a V2 adapter calls.
//
// ─── MoE math (qwen2_action_expert.py:274-360) ───
//   router_logits     = Linear(h, gate_w)                 # [T,E]
//   routing_scores    = sigmoid(router_logits)            # [T,E]
//   scores_for_choice = routing_scores + corr_bias        # bias affects SELECTION ONLY
//   selected          = topk(scores_for_choice, top_k)
//   routing_weights   = routing_scores.gather(selected)   # from UNBIASED scores
//   routing_weights  /= routing_weights.sum(-1) + 1e-20   # norm_topk_prob
//   routing_weights  *= routed_scaling                    # 4.0
//   y_routed          = sum_k routing_weights[:,k] * expert_{selected[:,k]}(h)
//   y_shared          = down_s(silu(gate_s(h)) * up_s(h)) # always-on, UNGATED
//   out               = y_routed + y_shared
//
// We implement the DENSE-EINSUM formulation that the official source itself uses
// (qwen2_action_expert.py:342-351): every expert is evaluated for every token and
// combined with a STATIC [E,T] weight matrix whose non-top-k entries are 0. This
// keeps every shape static — no device-side topk/gather/dispatch, no data-dependent
// control flow, no CPU callback (which would force sync_point() -> replayable=false,
// src/graph/custom_cpu_fallback_dispatch.cpp:55-60). Expert evaluation uses
// static shapes even when the selected experts differ between tokens.
#pragma once

#include "rpu_qwen3_model.h"   // v3::CausalDecoderModel
#include "rpu_kernel_decls.h"
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace v3 {

namespace lingbot_v2_moe_internal {

enum class ExactSuffixProfileKind : uint8_t {
    DenseFp16 = 1,
    GroupedW8A16 = 2,
    GroupedW4A16 = 3,
};

struct ExactSuffixProfileDescriptor {
    ExactSuffixProfileKind kind = ExactSuffixProfileKind::DenseFp16;
    uint64_t profile_hash = 0;
    uint64_t layout_hash = 0;
    size_t temporary_bytes = 0;
    FmbForwardOperandResidency mask_residency =
        FmbForwardOperandResidency::UNSPECIFIED;
    uint64_t model_state_generation = 0;
};

}  // namespace lingbot_v2_moe_internal

class LingbotV2MoeExpertModel : public CausalDecoderModel {
public:
    LingbotV2MoeExpertModel();
    ~LingbotV2MoeExpertModel() override;

    // Bind the legacy Python-facing controls as one typed, immutable native
    // snapshot.  The create wrapper calls this before any weight setter, so
    // model planning and graph construction never consult process-global env.
    void configure_moe_runtime(
        bool bufonly, bool addr, int64_t schunk,
        int64_t rchunk_requested, bool dense_soft_router,
        bool fp16_top4, bool routed_only, bool down_acc16);

    // Bind the MoE state. MUST be called AFTER CausalDecoderModel::set_weights
    // (needs num_layers()/hidden_size()); the base set_weights binds the SHARED
    // expert through its dense gate_w/up_w/down_w lists with
    // intermediate_size == shared_inter_pad.
    //
    //   router_gate_w  [nl]    each [E, hidden]  single-core col-swizzled fp16 RPU
    //   router_bias    [nl]    each [E]          fp16 RPU (per-layer correction bias)
    //   expert_gate_w  [nl*E]  each [I, hidden]  8-core col-swizzled fp16 RPU
    //   expert_up_w    [nl*E]  each [I, hidden]  8-core col-swizzled fp16 RPU
    //   expert_down_w  [nl*E]  each [hidden, I]  8-core row-swizzled fp16 RPU
    // Expert bank index is layer_idx * num_experts + e (DDR-resident: 2.25 MB per
    // expert x 32 = 72 MB/layer > 60 MB total SPM, so it is streamed per GEMM by
    // rpu_launch_linear_spm_to_spm_acc16_kernel, whose weight operand IS a DDR
    // at::Tensor — no separate DDR-linear path is required).
    void set_moe_weights(
        at::TensorList router_gate_w, at::TensorList router_bias,
        at::TensorList expert_gate_w, at::TensorList expert_up_w,
        at::TensorList expert_down_w,
        int64_t num_experts, int64_t top_k, int64_t routed_inter,
        double routed_scaling, int64_t chunk_size,
        bool grouped_experts_requested);

    // Optional cold, selection-only router bank: [nl][E,hidden], contiguous
    // single-core col-swizzled FP16 RPU. Call before any forward/Z2 prime.
    // The caller centers the FP32 source weights before the FP16 conversion.
    // Requires strict FP16 top-k and exactly zero installed correction biases;
    // original sigmoid scores continue to supply the normalised weights.
    void set_router_rank_weights(at::TensorList rank_weights);

    std::vector<int64_t> resolve_action_stage_domain(
        int64_t execution_len, int64_t prefix_len,
        int64_t mask_kv_len, int64_t cos_sin_offset);

    // Controlled FP16-state denoise unroll. The three action-head matrices are
    // single-core col-swizzled fp16 weights but launch through the ACC32
    // SPM-to-SPM Linear path; every result (including Euler x_t) is stored fp16.
    // Must run after set_moe_weights and before the first forward.
    void set_denoise_weights(
        const at::Tensor& state_w, const at::Tensor& state_b,
        const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
        const at::Tensor& op, const at::Tensor& op_bias,
        const at::Tensor& time_all,
        const at::Tensor& adarms_is, const at::Tensor& adarms_ish,
        const at::Tensor& adarms_ps, const at::Tensor& adarms_psh,
        int64_t state_dim, int64_t action_dim, int64_t state_dim_pad,
        int64_t action_dim_pad, int64_t num_steps);

    // One caller-owned GraphCache capture emits all num_steps bodies. x0/state
    // and x_final may drift between forwards; the graph patches their mutable
    // DMA bases on REPLAY. x_final is [1, chunk_size, action_dim_pad].
    void denoise_unroll_forward(
        const at::Tensor& x0_rpu, const at::Tensor& state_rpu,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        at::Tensor& x_final, double dt, int64_t prefix_len,
        int64_t cos_sin_offset, int64_t num_steps,
        at::IntArrayRef planned_stage_descriptor = {});

    // Reversible CPU-only admission for the three exact LingBot2 suffix profiles.
    // It validates all weights/scales and returns the dry layout identity before
    // the product orchestration performs any allocator-backed component prepare.
    lingbot_v2_moe_internal::ExactSuffixProfileDescriptor
    describe_exact_suffix_spm_profile(int64_t prefix_len);

    // Control-plane-only cold allocation for an already-admitted exact suffix
    // profile. This performs no compute and emits no preload DMA. The wrapper is
    // intentionally not exposed as a Torch op; only the product orchestration in
    // rpu_qwen3vl_multiview_spm_z2.cpp may call it.
    SpmPipelineComponentLayout prime_exact_suffix_spm_layout(
        int64_t prefix_len,
        const lingbot_v2_moe_internal::ExactSuffixProfileDescriptor& expected);

protected:
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            planning_rope_mode_, planning_graph_lifecycle_,
            planning_mode_, planned_cos_sin_offset_);
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    ModelStaticConfig static_config() override;
    bool subclass_chunk_size_valid(
        int64_t chunk_size, int64_t seq_len,
        int64_t position) const override;
    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override;
    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& manifest) const override;
    FmbForwardOperandResidency physical_forward_operand_residency(
        const FmbPhysicalExecutionManifest& manifest) const override;
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override;

private:
    std::vector<BufferDecl> moe_baseline_buffer_declarations(
        const LayoutContext& ctx);
    FmbRouteManifestEntry mask_residency_route_for_layout(
        const LayoutContext& layout, bool honor_exact_authority = true) const;
    void emit_action_mask_spm(int layer_idx, const ChunkInfo& chunk,
                             int64_t seq_len, int64_t kv_seq_len, int tp,
                             uint32_t mask_off);
    // Cold Z2 ownership authority, published only after an exact successful
    // prime. Its bound layout must also be used by the later Action planner.
    std::optional<lingbot_v2_moe_internal::ExactSuffixProfileDescriptor>
        exact_suffix_mask_authority_;
    lingbot_v2_moe_internal::ExactSuffixProfileKind
    validate_exact_suffix_profile(int64_t prefix_len) const;
    uint64_t exact_suffix_profile_hash(
        lingbot_v2_moe_internal::ExactSuffixProfileKind kind) const;

    void emit_denoise_pre_layers();
    void emit_denoise_post_layers();

    // Emits the sparse-MoE MLP for one layer. Reads residual1 (= normed hidden)
    // and residual2 (= residual stream), writes residual1 (= new hidden).
    void emit_moe_mlp(int layer_idx, int64_t seq_len);
    // GROUPED variant: core-slice-interleaved packed experts (RPU_LINGBOT2_GROUPED_EXPERTS).
    void emit_moe_mlp_grouped(int layer_idx, int64_t seq_len);

    // The fp32 router/select seam. Must produce "moe_rwbc" = dense [E,Tp]
    // routing weights. Currently escalates: no fp32 router-select op exists in
    // any shipped rhinoOpLib. See the .cpp for the full op contract; the
    // executable acceptance spec is rpu_lingbot_v2_moe_select_ref.h.
    void emit_router_select(int layer_idx, int64_t seq_len);

public:
    // DEBUG-ONLY read-back of the dense-soft router's device-computed routing
    // weights: [nl, E, Tp], written by emit_router_select's core0->DDR relay.
    // Exists so the bring-up can MEASURE the router (shape / pad / row-sum)
    // instead of arguing it constructively. Not on any hot path.
    at::Tensor debug_rw_stage() const { return rw_stage_; }
    at::Tensor debug_wtm_stage() const { return wtm_stage_; }
    at::Tensor debug_l0_out() const { return l0_out_; }
    // DEBUG [Tp,h] capture of layer-0's input RMSNorm + AdaRMS output (`input_norm`),
    // i.e. the activation both the QKV projections and the MoE branch consume. Opt-in
    // via RPU_L2_CAPTURE_INNORM; nothing is allocated or emitted when it is off.
    at::Tensor debug_innorm() const { return innorm_; }
    at::Tensor debug_gcap() const { return gcap_; }
    at::Tensor debug_acap() const { return acap_; }
    at::Tensor debug_scap() const { return scap_; }
    at::Tensor debug_ccap() const { return ccap_; }
    // DEBUG-ONLY read-back of the DDR-resident packed expert weight bound to the
    // device path (which: 0=gate, 1=up, 2=down). packed_* are DDR tensors, NOT SPM
    // buffers, so this read-back carries none of the SPM capture-timing hazard.
    at::Tensor debug_packed(int64_t which, int64_t layer) const;
    // Bind per-layer packed expert weights + enable the grouped path (opt-in).
    void set_packed_expert_weights(at::TensorList gate_packed, at::TensorList up_packed,
                                   at::TensorList down_packed);
    // Quantized grouped experts: bind W8 [N] or controller-striped W4 pgrp scales.
    // Called AFTER set_packed_expert_weights, only for int8 or packed-int4 weights.
    void set_packed_expert_scales(at::TensorList gate_scale, at::TensorList up_scale,
                                  at::TensorList down_scale);
    // W8A16 opt-in for the layer's NON-routed linears: attention q/k/v/o and the SHARED expert
    // gate/up/down. The base already carries per-layer *_ws fields and already hands them to the
    // acc16 launcher (rpu_qwen3_model.h:1663/1797/1815) -- only the binding was missing. router
    // is deliberately NOT included.
    void set_base_scales(at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
                         at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
                         at::TensorList down_ws);

    // DEBUG-ONLY read-back of the router's REAL INPUT hidden state (residual1),
    // [nl, Tp, h], one slot per layer. Populated ONLY when
    // RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H=1; otherwise this tensor is undefined and no
    // DMA is emitted, so the default graph is unchanged. Exists so a CPU fp32 router
    // reference can be computed on the SAME activations the fp16 router saw --
    // residual1 sits AFTER attention, so it is not host-computable.
    at::Tensor debug_router_h() const { return h_stage_; }
    // DEBUG-ONLY all-layer residual-stream taps (opt-in via RPU_L2_CAP_RESID). layer_in =
    // residual1 at phase-1 start (= layer input), attn_resid = residual2 after phase-5
    // (= layer_input + attention_output). Together with debug_router_h they localise, per
    // layer, whether the RPU-vs-CPU drift enters at attention or at the MoE MLP.
    at::Tensor debug_layer_in() const { return lin_stage_; }
    at::Tensor debug_attn_resid() const { return ar_stage_; }

private:

    at::Tensor ones_pad_;     // [Tp*E] fp16 RPU — fills the router score pad rows with 1.0 so
                              //          each pad column sums to E and divides to a finite 1/E
                              //          (never 0/0=NaN), with no unaligned denominator patch
    at::Tensor expert_scatter_ids_; // [Tp,E] raw uint16 (expert*Tp+token) held in
                                    // FP16-sized DDR; persistent strict-top-k lookup
    at::Tensor rw_stage_;     // [nl, E, Tp] fp16 RPU — per-layer core0->all-cores relay
    at::Tensor h_stage_;      // [nl, Tp, h] fp16 RPU — DEBUG router-input capture, opt-in,
                              //          one slot PER LAYER (never overwritten downstream)
    at::Tensor lin_stage_;    // [nl, Tp, h] fp16 RPU — DEBUG layer-input (residual1 @ phase1), opt-in
    at::Tensor ar_stage_;     // [nl, Tp, h] fp16 RPU — DEBUG post-attention residual (residual2 @ phase5), opt-in
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        auto identity = causal_kvinsert_cost_weight_identity();
        if (identity.empty()) return {};
        identity.insert(identity.end(), {
            static_cast<int64_t>(packed_w8a16_),
            static_cast<int64_t>(grouped_experts_),
            static_cast<int64_t>(cold_bufonly_),
            static_cast<int64_t>(cold_addr_),
            static_cast<int64_t>(cold_grouped_experts_requested_),
            static_cast<int64_t>(cold_schunk_),
            static_cast<int64_t>(cold_rchunk_),
            static_cast<int64_t>(cold_rchunk_requested_),
            static_cast<int64_t>(cold_rchunk_exact_1632_),
            static_cast<int64_t>(num_experts_),
            static_cast<int64_t>(top_k_),
            static_cast<int64_t>(routed_inter_),
            static_cast<int64_t>(fp16_top4_),
            static_cast<int64_t>(denoise_unroll_),
            static_cast<int64_t>(routed_scaling_.x)});
        identity.push_back(static_cast<int64_t>(router_gate_.size()));
        for (const auto& tensor : router_gate_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        if (!router_rank_weights_.empty()) {
            // Preserve the legacy identity byte-for-byte when the opt-in is off.
            identity.push_back(INT64_C(0x4c3252414e4b31));  // L2RANK1
            identity.push_back(static_cast<int64_t>(router_rank_weights_.size()));
            for (const auto& tensor : router_rank_weights_) {
                append_kvinsert_cost_tensor_identity(identity, tensor);
            }
        }
        identity.push_back(static_cast<int64_t>(router_bias_.size()));
        for (const auto& tensor : router_bias_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        identity.push_back(static_cast<int64_t>(expert_gate_.size()));
        for (const auto& tensor : expert_gate_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        identity.push_back(static_cast<int64_t>(expert_up_.size()));
        for (const auto& tensor : expert_up_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        identity.push_back(static_cast<int64_t>(expert_down_.size()));
        for (const auto& tensor : expert_down_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        identity.push_back(static_cast<int64_t>(packed_gate_.size()));
        for (const auto& tensor : packed_gate_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        identity.push_back(static_cast<int64_t>(packed_up_.size()));
        for (const auto& tensor : packed_up_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        identity.push_back(static_cast<int64_t>(packed_down_.size()));
        for (const auto& tensor : packed_down_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        identity.push_back(static_cast<int64_t>(packed_gate_s_.size()));
        for (const auto& tensor : packed_gate_s_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        identity.push_back(static_cast<int64_t>(packed_up_s_.size()));
        for (const auto& tensor : packed_up_s_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        identity.push_back(static_cast<int64_t>(packed_down_s_.size()));
        for (const auto& tensor : packed_down_s_) {
            append_kvinsert_cost_tensor_identity(identity, tensor);
        }
        for (const auto* tensor : {
                &state_w_, &state_b_, &wc_, &mo_,
                &mo_bias_, &op_, &op_bias_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    std::vector<at::Tensor> router_gate_, router_bias_;   // [nl]
    std::vector<at::Tensor> router_rank_weights_;  // empty by default; cold DDR keepalive
    std::vector<at::Tensor> expert_gate_, expert_up_, expert_down_;  // [nl*E]
    // GROUPED-EXPERTS: per-layer core-slice-interleaved packed weights.
    //   packed_gate_/up_[L]: [E*I, hidden] col-swizzled 8-core
    //   packed_down_[L]:     [hidden, E*I] row-swizzled 8-core
    std::vector<at::Tensor> packed_gate_, packed_up_, packed_down_;
    // Quant scales, one per layer: W8 [N], or W4 pgrp physical payload whose
    // dim 0 carries group_size. Empty => fp16 weights.
    std::vector<at::Tensor> packed_gate_s_, packed_up_s_, packed_down_s_;
    bool packed_w8a16_ = false;
    at::Tensor wtm_stage_;   // [nl, Tp, E] fp16 RPU — token-major routing-weight relay
    at::Tensor l0_out_;      // DEBUG [Tp,h] capture of layer-0 MoE output (residual1)
    at::Tensor innorm_;      // DEBUG [Tp,h] capture of layer-0 input_norm (post AdaRMS)
    at::Tensor gcap_;        // DEBUG [Tp, E*Ic] core-0 gate-GEMM output, layer 0
    at::Tensor acap_;        // DEBUG [Tp, h] core-0 routed moe_acc (post routed-down), layer 0
    at::Tensor scap_, ccap_; // DEBUG core-0 silu-out, scale-out (fresh single-version buffers)
    bool debug_stages_ = false;
    bool debug_down_acc16_ = false;
    bool debug_capture_gate_ = false;
    bool debug_capture_l0_ = false;
    bool debug_capture_innorm_ = false;
    bool debug_cap_resid_ = false;
    bool debug_routed_only_ = false;
    bool grouped_experts_ = false;
    // Python resolves legacy inputs before native create; every native consumer
    // below reads only this typed per-handle snapshot.
    bool runtime_config_bound_ = false;
    bool cold_bufonly_ = false;
    bool cold_addr_ = false;
    bool cold_grouped_experts_requested_ = false;
    int64_t cold_schunk_ = 256 * 127;
    int64_t cold_rchunk_ = 256;
    int64_t cold_rchunk_requested_ = 0;
    bool cold_rchunk_exact_1632_ = false;
    int64_t num_experts_ = 0, top_k_ = 0, routed_inter_ = 0;
    int64_t chunk_size_ = 0, tp_rows_ = 0;   // tp_rows_ = Align(chunk_size_, 16)
    c10::Half routed_scaling_ = c10::Half(1.0f);
    // DEBUG dense-soft router (RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER=1). Default
    // false => emit_router_select keeps the strict TORCH_CHECK(false) hard-fail.
    bool debug_dense_soft_router_ = false;
    // Strict fp16 top-k router (RPU_LINGBOT2_FP16_TOP4=1). Selection is exactly k
    // with lower-index tie-breaking; correction bias is selection-only.
    bool fp16_top4_ = false;
    // DEBUG router-input dump (RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H=1). Independent of
    // debug_dense_soft_router_. Default false => nothing allocated, no DMA emitted.
    bool debug_dump_router_h_ = false;

    // Controlled fused-denoise state. Undefined/zero in the default host-loop
    // path, so it cannot perturb the existing graph or SPM manifest.
    bool denoise_unroll_ = false;
    bool denoise_unroll_active_ = false;
    at::Tensor state_w_, state_b_, wc_, mo_, mo_bias_, op_, op_bias_, time_all_;
    at::Tensor denoise_stage_;  // stable DDR [1, chunk_size, hidden]
    int64_t state_dim_ = 0, action_dim_ = 0;
    int64_t state_dim_pad_ = 0, action_dim_pad_ = 0, num_steps_ = 0;
    c10::Half denoise_dt_ = c10::Half(0.0f);
    bool denoise_dt_pinned_ = false;
    int64_t planned_cos_sin_offset_ = -1;
    uint64_t denoise_x0_src_base_ = 0;
    uint64_t denoise_state_src_base_ = 0;
    uint64_t denoise_final_dst_base_ = 0;
    at::Tensor denoise_x0_ref_, denoise_state_ref_, denoise_final_ref_;
};

namespace lingbot_v2_moe_internal {

FusedModelBase& exact_suffix_owner(int64_t handle);
ExactSuffixProfileDescriptor describe_exact_suffix_spm_profile(
    int64_t handle, int64_t prefix_len);
SpmPipelineComponentLayout prime_exact_suffix_spm_layout(
    int64_t handle, int64_t prefix_len,
    const ExactSuffixProfileDescriptor& expected);

}  // namespace lingbot_v2_moe_internal

}  // namespace v3
