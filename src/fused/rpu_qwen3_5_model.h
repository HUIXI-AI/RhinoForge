// rpu_qwen3_5_model.h — Qwen3.5 hybrid-attention fused model (v3).
//
// Qwen3.5 interleaves full-attention layers and Gated-DeltaNet (GDN) linear
// layers per `config.layer_types` (0.8B: 18 GDN + 6 full, every 4th full).
// This class drives a single all-layers-once fused forward and dispatches each
// layer by type:
//   full → build_full_attention (mirrors the Qwen3 causal-decoder path)
//   GDN  → build_gdn(decode): recurrent (decode) / chunked (prefill), by chunk.len
//
// Full-attention path implements: per-head QK-norm, gated-attention output gate,
// and partial M-RoPE (kernel "partial_mrope" — rotate-half within rotary_dim,
// pass-through the rest; in-place). M-RoPE T/H/W interleaving is baked into the
// host cos/sin tables (text-only degenerates to 1D partial RoPE). The complete
// hybrid stack is validated against CPU-fp32 prefill+decode references.
//
// Standalone (does NOT subclass CausalDecoderModel): the per-layer norm
// preloads in that class assume homogeneous layers and would DMA empty GDN
// attention weights. Duplicating the full-attention emission here keeps the
// four shipping models (Qwen3 / Llama / Qwen3-VL / causal decoder) untouched.
#pragma once

#include "fused_model_base.h"
#include "g05_nvfp4.h"
#include "qwen35_text_nvfp4.h"
#include "rpu_helpers.h"          // SdpaConfig, SdpaKernelType, sdpa_is_valid_chunk_size
#include "rpu_math_precision.h"

#include <ATen/ATen.h>
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

enum class RpuRmsNormSpmRoute : int64_t;

namespace v3 {

struct SpmDense2DSpec;
class SpmPortView;

class Qwen3_5Model : public FusedModelBase {
public:
    Qwen3_5Model();
    ~Qwen3_5Model() override;
    void set_execution_cores(int64_t cores);
    std::vector<int64_t> execution_topology() const;

    // Per-layer lists are length == num_layers. GDN-layer slots hold undefined
    // (empty) attention tensors; only full-layer slots are read. attn_gate_list
    // is the gate half split off q_proj by the adapter (stored, gate deferred).
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        at::TensorList attn_gate_list,
        at::TensorList input_norm_list, at::TensorList post_norm_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        const at::Tensor& cos, const at::Tensor& sin,
        const at::Tensor& final_norm_w,
        at::IntArrayRef layer_is_full,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps, bool use_silu,
        at::IntArrayRef mrope_section,   // [T,H,W] half-dims; empty → 1D RoPE
        // GDN mixer weights (linear_attention slots filled; full slots empty). z/out are
        // col/row-swizzled; gdn_norm is plain; A_log/dt_bias [H] (8-slot padded). q/k/v
        // proj, per-path conv, N_bg b/a are the per-path lists below (decode + prefill).
        at::TensorList gdn_in_z_list, at::TensorList gdn_out_list,
        at::TensorList gdn_A_log_list, at::TensorList gdn_dt_bias_list,
        at::TensorList gdn_norm_list,
        int64_t gdn_num_v_heads, int64_t gdn_key_head_dim,
        int64_t gdn_value_head_dim, int64_t gdn_conv_dim, int64_t gdn_conv_kernel,
        // Prefill (chunk) per-path weights. Every list is length num_layers;
        // full-attention slots are empty and GDN slots are populated.
        at::TensorList gdn_q_list, at::TensorList gdn_k_list,
        at::TensorList gdn_v_list, at::TensorList gdn_cq_list,
        at::TensorList gdn_ck_list, at::TensorList gdn_cv_list,
        at::TensorList gdn_b_bg_list, at::TensorList gdn_a_bg_list,
        at::TensorList q_scale_list, at::TensorList k_scale_list,
        at::TensorList v_scale_list, at::TensorList o_scale_list,
        at::TensorList attn_gate_scale_list,
        at::TensorList gate_scale_list, at::TensorList up_scale_list,
        at::TensorList down_scale_list,
        at::TensorList gdn_in_z_scale_list, at::TensorList gdn_out_scale_list,
        at::TensorList gdn_q_scale_list, at::TensorList gdn_k_scale_list,
        at::TensorList gdn_v_scale_list, at::TensorList gdn_b_bg_scale_list,
        at::TensorList gdn_a_bg_scale_list,
        at::TensorList nvfp4_tensor_scales = {},
        at::TensorList gate_negative_zero_list = {},
        at::TensorList up_negative_zero_list = {},
        at::TensorList down_negative_zero_list = {});

    // Per-forward entry. gdn_states / conv_states are parallel-indexed (full
    // slots empty). Stashed for the GDN seam, then run_all_layers.
    at::Tensor forward(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        std::vector<at::Tensor>& gdn_states,
        std::vector<at::Tensor>& conv_states,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position, bool is_causal,
        int64_t planned_chunk_size = 0,
        at::IntArrayRef planned_stage_descriptor = {});

    // Bounded Qwen3.5-2B contiguous-merger Z2 canary hooks.  The physical
    // coordinator owns the lease and the complete [execution, hidden] port;
    // this component only validates/adopts that storage and consumes it as the
    // layer-0 residual stream.
    SpmPipelineComponentLayout prepare_z2_layout(int64_t execution_len,
                                                  int64_t real_len);
    SpmDense2DSpec z2_storage_spec(int64_t execution_len) const;
    void adopt_z2_layout(const SpmPipelineLease& lease,
                         const SpmTensorView& scratch);
    void bind_z2_port(const SpmPipelineLease& lease,
                      const SpmPortView& port);
    void validate_z2_layout(const SpmPipelineLease& lease) const;
    void clear_z2_layout(uint64_t epoch, uint64_t plan_hash);
    at::Tensor forward_z2(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        std::vector<at::Tensor>& gdn_states,
        std::vector<at::Tensor>& conv_states,
        uint64_t epoch,
        uint64_t plan_hash);

    // Seam accessor: build_gdn reads this layer's recurrent-state DDR tensor.
    at::Tensor* gdn_state_for_layer(int layer_idx);

    // Per-forward PREFILL RoPE tables (B channel). Decode uses the static
    // position-lookup cos_/sin_ from set_weights; PREFILL (seq_len>1) needs the
    // interleaved M-RoPE table built from THIS forward's 3D position_ids (image/
    // text layout ⇒ input-dependent, can't be a static weight-time table). The
    // adapter rebuilds cos/sin each forward and calls this before the forward op.
    // Tables are [N, rotary_dim/2] fp16 on device, indexed by absolute position
    // (kernel reads [pos_offset + token]). Text-only degenerates to arange.
    void set_prefill_rope(const at::Tensor& cos, const at::Tensor& sin);

    // Route B: the REAL (pre-pad) prefill length. The adapter pads input_ids up to a multiple of
    // 64 so every chunk.len is 64-aligned (full-attn SDPA + GDN chunk core both need that), then
    // calls this with the real length P. build_gdn uses it to zero the pad tokens out of the
    // carried recurrent/conv state. -1 (unset) ⇒ no padding (chunk is fully real).
    void set_valid_prefill_len(int64_t n) { valid_prefill_len_ = n; }
    // W2 (decode-after-image M-RoPE): HF compute_3d_position_ids shifts decode
    // positions by rope_deltas (= max_mrope_pos + 1 - num_prefill_tokens). The host
    // sets this once after an image prefill; DECODE indexes the static cos_/sin_ at
    // ctx().position + mrope_pos_delta_. 0 (default) = text-only, no shift.
    void set_mrope_position_delta(int64_t d) { mrope_pos_delta_ = d; }
    void set_chunk_size_cap(int64_t cap);
    void set_prefill_chunk_size(int64_t chunk_size);
    void stage_prefill_execution_controls(
        uint64_t token, int64_t cap, int64_t chunk_size,
        int64_t max_kv_len, int64_t envelope_chunk);
    void set_linear_acc32(bool enabled);
    // G05 opt-in, bound once before weights. Linear accumulation is separate.
    void set_high_precision(bool enabled);
    void set_fast_replay(bool enabled);
    // Opt-in: a fast-replay multi-token prefill from position 0 skips the forced
    // CPU flush of the recurrent/conv states. Its first chunk zero-fills them in
    // SPM and never reads DDR, and every backend CPU writer (zero_, fill_, uploads)
    // publishes its own range, so no dirty CPU line can land on the device's state.
    void set_initial_prefill_state_flush_skip(bool enabled);
    // G0.5 action expert: all-full-attention Qwen3.5 with host-precomputed
    // AdaLN modulation and sparse per-layer read-only prefix KV.
    void enable_action_mode();
    void set_action_io_weights(
        const at::Tensor& input_w, const at::Tensor& input_b,
        const at::Tensor& output_w, const at::Tensor& output_b,
        int64_t action_dim, int64_t action_dim_pad, int64_t action_len);
    at::Tensor forward_action(
        const at::Tensor& hidden_states,
        const at::Tensor& adaptive_mod,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        at::IntArrayRef prefix_lens,
        at::IntArrayRef planned_stage_descriptor);
    at::Tensor forward_action_step(
        const at::Tensor& action,
        const at::Tensor& action_keep_mask,
        at::Tensor& action_out,
        double delta_t,
        const at::Tensor& adaptive_mod,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        at::IntArrayRef prefix_lens,
        int64_t num_steps,
        at::IntArrayRef planned_stage_descriptor);
    at::Tensor forward_action_rtc_step(
        const at::Tensor& action,
        const at::Tensor& action_keep_mask,
        at::Tensor& action_out,
        const at::Tensor& action_prefix,
        int64_t action_prefix_len,
        double delta_t,
        const at::Tensor& adaptive_mod,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        at::IntArrayRef prefix_lens,
        int64_t num_steps,
        at::IntArrayRef planned_stage_descriptor);
    at::Tensor forward_action_rtc_trace(
        const at::Tensor& action,
        const at::Tensor& action_keep_mask,
        at::Tensor& action_out,
        const at::Tensor& action_prefix,
        int64_t action_prefix_len,
        double delta_t,
        const at::Tensor& adaptive_mod,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        at::IntArrayRef prefix_lens,
        at::Tensor& trace_pre_action,
        at::Tensor& trace_action_embed,
        at::Tensor& trace_final_hidden,
        at::Tensor& trace_velocity,
        at::Tensor& trace_post_action,
        at::IntArrayRef planned_stage_descriptor);
    int64_t resolve_prefill_chunk_size(int64_t execution_len);
    std::vector<int64_t> resolve_prefill_stage_domain(
        int64_t execution_len, int64_t logical_len,
        int64_t planning_chunk_size_override = -1);
    std::vector<int64_t> resolve_decode_stage_descriptor();
    std::vector<int64_t> resolve_action_stage_domain(
        int64_t action_len, int64_t max_prefix_len,
        int64_t requested_chunk_size, bool include_action_io,
        int64_t action_route_flags, int64_t graph_lifecycle, int64_t num_steps);
    void set_action_chunk_size(int64_t chunk_size);
    // Valid action rows inside the physical (multiple-of-16) horizon. Padded
    // rows are computed but never attended: attention keys end at prefix + valid.
    void set_action_valid_len(int64_t valid_len);
    void stage_action_chunk_size(uint64_t token, int64_t chunk_size);
    void enable_execution_reconfigure();
    void set_retained_prefill_graph(bool enabled);

protected:
    int64_t resolve_model_mlp_intermediate_size(int64_t logical_size) const override;
    bool mlp_packed_w4_ = false;
    bool mlp_awq_g128_ = false;
    DecoderExecutionTopology resolve_model_execution_topology(
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size) const override;
    int64_t subclass_layout_hash() const override;
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            action_descriptor_include_io_, action_descriptor_route_flags_,
            action_descriptor_graph_lifecycle_, action_step_active_,
            action_loop_active_, action_num_steps_, action_rtc_active_, action_trace_active_);
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    FmbForwardOperandResidency physical_forward_operand_residency(
        const FmbPhysicalExecutionManifest& manifest) const override;
    ModelStaticConfig       static_config() override;
    ModelDynamicConfig      dynamic_config(const ChunkPlan& plan) override;
    void                    build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override;
    bool                    subclass_chunk_size_valid(int64_t cs, int64_t seq,
                                                      int64_t pos) const override;
    // MR-A: once-per-resolve certified-envelope gate (deny-by-default).
    int64_t                 subclass_chunk_size_cap(int64_t seq,
                                                    int64_t pos) const override;
    bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const override;
    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override;
    std::vector<FmbPhysicalExecutionManifest>
    physical_manifest_domain_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override;
    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& manifest) const override;

private:
    std::vector<BufferDecl> declare_legacy_buffers(const LayoutContext& ctx) const;
    FmbForwardOperandResidency gdn_mask_residency_for_candidate(
        const LayoutContext& layout) const;
    std::vector<int64_t> gdn_mask_schedule_arguments(
        const FmbThreeStageChunkPlan& plan) const;
    static constexpr int64_t kTextGdnInputWorkspaceSite = 7200073926787961303LL;
    static constexpr int64_t kTextNvfp4TablesSite = 3184997657402357068LL;
    bool use_text_gdn_input_workspace() const;
    static int gdn_input_workspace_slot(const char* name);
    std::vector<int64_t> gdn_input_workspace_arguments(int64_t rows) const;
    int64_t gdn_input_workspace_offset(const char* name, int64_t rows) const;
    bool has_text_gdn_glast_local_capability() const;
    std::vector<int64_t> text_gdn_glast_local_identity() const;
    bool use_text_gdn_glast_local(int64_t rows) const;
    std::vector<int64_t> text_gdn_glast_local_arguments(int64_t rows) const;
    bool use_g05_gdn_local_slice(int64_t rows) const;
    std::vector<int64_t> g05_gdn_local_slice_arguments(int64_t rows) const;
    int gdn_tp() const { return num_cores() == 8 ? 8 : 4; }
    int lm_head_tp() const { return num_cores(); }
    int64_t text_ring_route(int64_t rows, int64_t cols) const;
    std::vector<int64_t> cold_topology_arguments() const;
    static int64_t linear_invocation(
        int64_t chunk_idx, FmbLinearRouteSelector selector) {
        // The physical route is identical for every layer/role that resolves
        // to the same selector. Reuse that exact route within a chunk so long
        // prefill candidates stay within the descriptor word budget.
        return chunk_idx * 8 + static_cast<int64_t>(selector);
    }
    // planner_owners.v1.json call-site authorities for build_full_attention.
    static constexpr int64_t kFullAttentionSite = 9137328812466618373LL;
    static constexpr int64_t kFullAttentionRawSpmSite =
        5148991311835582171LL;
    static constexpr int64_t kFullAttentionKvInsertSite = 8503568375190929130LL;
    FmbLinearRouteSelector linear_route_selector(
        const at::Tensor& weight, int64_t m) const;
    void consume_manifest_route(
        FmbRouteFamily family, int64_t site_id,
        int64_t selector, int64_t invocation = 0);
    at::Tensor prepare_decode_cpu_input(const at::Tensor& hidden_states);
    at::Tensor forward_impl(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        std::vector<at::Tensor>& gdn_states,
        std::vector<at::Tensor>& conv_states,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position, bool is_causal,
        int64_t planned_chunk_size,
        at::IntArrayRef planned_stage_descriptor);
    uint32_t layer_input_residual_addr(int layer_idx,
                                       const ChunkInfo& chunk) const;
    void validate_z2_prefill_contract() const;

    void launch_linear(
        int64_t chunk_idx,
        uint32_t input, const at::Tensor& weight, uint32_t output,
        int64_t m, int64_t n, int64_t k, int partition, int num_cores,
        uint32_t bias_spm_addr = 0, const at::Tensor& scale = {},
        int nvfp4_family = -1, int layer = -1,
        const at::Tensor& negative_zero = {});
    bool use_action_nvfp4(int64_t rows) const;
    std::vector<int64_t> action_nvfp4_arguments(int family) const;
    RpuUnaryPrecision unary_precision() const;
    RpuRmsNormSpmRoute rmsnorm_route(int64_t rows, int64_t cols) const;
    int64_t high_precision_site() const;
    std::vector<int64_t> high_precision_arguments() const;
    void consume_high_precision_math();
    // Full-attention layer emission. Optional exact G05 Text owner norm sets
    // the caller's layer-local flag only after successful COMPLETE emission.
    void build_full_attention(int layer_idx, const ChunkInfo& chunk,
                              bool* text_owner_norm_emitted = nullptr,
                              bool* action_norm_emitted = nullptr);
    // Shared tail: ordinary full raw in input_norm, or owner raw there plus
    // full normalized residual1 when this layer's owner flag is true.
    void emit_mlp_and_output(int layer_idx, const ChunkInfo& chunk,
                             bool text_owner_norm_emitted = false, bool action_norm_emitted = false);
    bool has_text_fp16_projection_owners() const;
    void prepare_text_owner_rms();
    bool use_text_owner_rms(const FmbThreeStageChunkPlan& plan,
        int64_t physical_len, int64_t logical_len, int64_t position, int64_t batch) const;
    std::vector<int64_t> text_owner_rms_arguments(int64_t logical_len = 245) const;
    void launch_text_owner_rms(int layer, const ChunkInfo& chunk);
    void prepare_text_gdn_owner_rms();
    bool use_text_gdn_owner_rms(const FmbThreeStageChunkPlan& plan,
        int64_t physical_len, int64_t logical_len, int64_t position, int64_t batch) const;
    std::vector<int64_t> text_gdn_owner_rms_arguments(int64_t logical_len = 245) const;
    void launch_text_gdn_owner_rms(int layer, const ChunkInfo& chunk);
    void prepare_text_mlp_tail();
    bool use_text_mlp_tail(const FmbThreeStageChunkPlan& plan,
        int64_t physical_len,int64_t logical_len,int64_t position,int64_t batch) const;
    bool text_mlp_tail_handoff(int layer,const ChunkInfo& chunk,bool producer);
    std::vector<int64_t> text_mlp_tail_arguments(int64_t logical_len = 245) const;
    void launch_text_mlp_tail(int layer,const ChunkInfo& chunk);
    void prepare_text_silu_round_mul();
    bool use_text_silu_round_mul(const FmbThreeStageChunkPlan& plan,
        int64_t physical_len,int64_t logical_len,int64_t position,int64_t batch) const;
    bool use_text_silu_round_mul(int64_t rows);
    std::vector<int64_t> text_silu_round_mul_arguments(int64_t logical_len = 245) const;
    void launch_text_silu_round_mul(int layer,const ChunkInfo& chunk);
    void prepare_text_gateup_pair();
    bool use_text_gateup_pair(int64_t rows) const;
    int64_t text_gateup_pair_site() const;
    std::vector<int64_t> text_gateup_pair_arguments() const;
    void launch_text_gateup_pair(int layer, const ChunkInfo& chunk);
    void load_adaptive_mod_row(int64_t row);
    bool use_action_mod_body() const;
    std::vector<int64_t> action_mod_body_arguments(int64_t body) const;
    void prepare_action_gateup_resident();
    void prepare_action_down_partial();
    bool use_action_down_partial(int64_t rows, int64_t layer) const;
    std::vector<int64_t> action_down_partial_arguments(bool preload) const;
    void preload_action_down_partial();
    void launch_action_down_partial(int layer, const ChunkInfo& chunk);
    void prepare_action_nvfp4_o_resident();
    bool use_action_nvfp4_o_resident(int64_t rows) const;
    int64_t action_nvfp4_o_resident_site() const;
    std::vector<int64_t> action_nvfp4_o_resident_arguments(int64_t layer, int64_t preload) const;
    void preload_action_nvfp4_o_resident();
    void launch_action_nvfp4_o_resident(int layer, const ChunkInfo& chunk);
    void prepare_action_nvfp4_down_resident();
    bool use_action_nvfp4_down_resident(int64_t rows, int64_t layer) const;
    int64_t action_nvfp4_down_resident_site() const;
    std::vector<int64_t> action_nvfp4_down_resident_arguments(int64_t layer, int64_t preload) const;
    void preload_action_nvfp4_down_resident();
    void launch_action_nvfp4_down_resident(int layer, const ChunkInfo& chunk);
    void prepare_action_o_resident();
    bool use_action_o_resident(int64_t rows) const;
    std::vector<int64_t> action_o_resident_arguments(int64_t layer, bool preload) const;
    void preload_action_o_resident();
    void launch_action_o_resident(int layer, const ChunkInfo& chunk);
    void prepare_action_attn_sigmoid_mul();
    bool use_action_attn_sigmoid_mul(int64_t rows) const;
    std::vector<int64_t> action_attn_sigmoid_mul_arguments() const;
    void launch_action_attn_sigmoid_mul(int layer, const ChunkInfo& chunk);
    void prepare_action_nvfp4_swiglu();
    bool use_action_nvfp4_swiglu(int64_t rows) const;
    int64_t action_nvfp4_swiglu_site() const;
    std::vector<int64_t> action_nvfp4_swiglu_arguments() const;
    void launch_action_nvfp4_swiglu(int layer, const ChunkInfo& chunk);
    void prepare_action_silu_round_mul();
    bool use_action_silu_round_mul(int64_t rows) const;
    std::vector<int64_t> action_silu_round_mul_arguments() const;
    void launch_action_silu_round_mul(int layer, const ChunkInfo& chunk);
    void prepare_action_silu_high_mul();
    bool use_action_silu_high_mul(int64_t rows) const;
    std::vector<int64_t> action_silu_high_mul_arguments() const;
    void launch_action_silu_high_mul(int layer, const ChunkInfo& chunk);
    void prepare_action_gateup_pair();
    bool use_action_gateup_pair(int64_t rows) const;
    int64_t action_gateup_pair_site() const;
    std::vector<int64_t> action_gateup_pair_arguments() const;
    void launch_action_gateup_pair(int layer, const ChunkInfo& chunk);
    void prepare_action_q_gate();
    bool use_action_q_gate(int64_t rows) const;
    std::vector<int64_t> action_q_gate_arguments() const;
    void launch_action_q_gate(int layer, const ChunkInfo& chunk);
    void prepare_action_qkv_gate();
    bool use_action_qkv_gate(int64_t rows) const;
    std::vector<int64_t> action_qkv_gate_arguments() const;
    void launch_action_qkv_gate(int layer, const ChunkInfo& chunk);
    void prepare_action_nvfp4_qk_norm_rope();
    bool use_action_nvfp4_qk_norm_rope(int64_t rows,int64_t prefix) const;
    std::vector<int64_t> action_nvfp4_qk_norm_rope_arguments() const;
    void launch_action_nvfp4_qk_norm_rope(int layer,const ChunkInfo& chunk,
        const at::Tensor& query_cos,const at::Tensor& query_sin,int64_t query_position);
    void prepare_action_gate_kv_owner();
    bool use_action_gate_kv_owner(int64_t rows, int64_t prefix) const;
    std::vector<int64_t> action_gate_kv_owner_arguments(int64_t prefix) const;
    void launch_action_gate_kv_owner(int layer, const ChunkInfo& chunk);
    void prepare_action_owner_cache();
    bool use_action_owner_cache(int64_t rows, int64_t max_prefix) const;
    std::vector<int64_t> action_qkv_gate_owner_only_arguments() const;
    std::vector<int64_t> action_k_owner_norm_rope_arguments() const;
    std::vector<int64_t> action_owner_cache_arguments(const KvInsertSegmentPlan& plan) const;
    void prepare_action_owner_cache_joint();
    bool use_action_owner_cache_joint(int64_t rows,int64_t prefix) const;
    std::vector<int64_t> action_owner_cache_joint_abi() const;
    std::vector<int64_t> action_owner_cache_joint_arguments(const KvInsertSegmentPlan& plan) const;
    void launch_action_qkv_gate_owner_only(int layer, const ChunkInfo& chunk);
    void launch_action_k_owner_norm_rope(int layer, const ChunkInfo& chunk);
    void launch_action_owner_cache_insert(int layer, const ChunkInfo& chunk, const KvInsertSegmentPlan& plan);
    void prepare_action_kv_owner();
    bool use_action_kv_owner(int64_t rows) const;
    std::vector<int64_t> action_kv_owner_arguments() const;
    void launch_action_kv_owner(int layer, const ChunkInfo& chunk);
    bool use_action_gateup_resident(int64_t rows) const;
    int64_t action_gateup_resident_layers() const;
    std::vector<int64_t> action_gateup_resident_arguments(int64_t layer, int64_t projection) const;
    void preload_action_gateup_resident();
    void launch_action_gateup_resident(int64_t layer, int64_t projection, uint32_t output);
    int64_t action_norm_route(int64_t rows) const;
    std::vector<int64_t> action_norm_arguments(int64_t rows) const;
    void prepare_action_newton_norm_shift();
    bool use_action_newton_norm_shift(int64_t rows) const;
    std::vector<int64_t> action_newton_norm_shift_arguments() const;
    void apply_action_newton_norm_shift(uint32_t input, uint32_t output, const ChunkInfo& chunk);
    int64_t action_residual_route(int64_t rows) const;
    void prepare_action_tail();
    bool use_action_tail(int64_t rows) const;
    std::vector<int64_t> action_tail_arguments() const;
    void launch_action_tail(int layer, const ChunkInfo& chunk);
    void prepare_action_mlp_tail();
    bool use_action_mlp_tail(const FmbThreeStageChunkPlan& plan, int64_t rows) const;
    bool action_mlp_tail_handoff(int layer, const ChunkInfo& chunk, bool producer);
    std::vector<int64_t> action_mlp_tail_arguments() const;
    void launch_action_mlp_tail(int layer, const ChunkInfo& chunk);
    bool use_action_ring_rtc_residual(int64_t rows) const;
    std::vector<int64_t> action_ring_rtc_residual_arguments() const;
    void launch_action_ring_rtc_residual(
        uint32_t input, uint32_t residual, uint32_t output,
        int64_t ring_site, const ChunkInfo& chunk);
    void apply_adaptive_norm(
        uint32_t input, uint32_t output, const ChunkInfo& chunk);
    void apply_adaptive_residual_gate(
        uint32_t output, uint32_t residual, const ChunkInfo& chunk);
    void apply_raw_residual_gate(
        uint32_t output, uint32_t residual, uint32_t gate,
        int64_t seq_len, int64_t route);
    void emit_action_input_projection();
    void emit_action_output_projection();
    struct ActionRtcTraceOutputs {
        at::Tensor& pre_action;
        at::Tensor& action_embed;
        at::Tensor& final_hidden;
        at::Tensor& velocity;
        at::Tensor& post_action;
    };
    at::Tensor forward_action_step_impl(
        const at::Tensor& action,
        const at::Tensor& action_keep_mask,
        at::Tensor& action_out,
        const at::Tensor* action_prefix,
        int64_t action_prefix_len,
        const ActionRtcTraceOutputs* trace,
        double delta_t,
        const at::Tensor& adaptive_mod,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        at::IntArrayRef prefix_lens,
        int64_t num_steps,
        at::IntArrayRef planned_stage_descriptor);

public:
    // Unified Gated-DeltaNet token mixer. Shared setup/proj/tail with an if(decode) split
    // for the divergent conv + delta-rule core: decode (seq=1) = recurrent single step;
    // prefill (chunk.len = C*N, C=64) = chunked delta-rule. Reads input_norm and
    // adds the original residual1. Ordinary output is full raw residual2;
    // owner output is raw owner rows plus full normalized residual1.
    // Recurrent_state and conv_state DDR updates are unchanged.
    // Exact GDN prefill may publish owner-normalized output; decode leaves
    // this optional layer-local result untouched and keeps the full raw ABI.
    void build_gdn(int layer_idx, const ChunkInfo& chunk, bool decode,
                   bool* text_owner_norm_emitted = nullptr);
    at::Tensor* conv_state_for_layer(int layer_idx);
    // PREFILL 时 planner 实际选中的 chunk_size。基类的 get_last_resolved_chunk_size()
    // 每次 compute_chunks 都覆写,而 decode(seq_len=1 → 固定 cs=16)跑在 prefill 之后,
    // 于是事后读基类那份永远是 16。这份只在 prefill forward 里更新,拿得到真实值。
    // 0 = 本 handle 还没跑过 prefill。与 cold per-handle cap 配套用于对账。
    int64_t last_prefill_chunk_size() const { return last_prefill_chunk_size_; }
    // Diagnostic: recipes bound, bodies skipped, ordinary bodies in scope, last rejection.
    std::vector<int64_t> checked_decode_replay_counters() const;
    // Runs [N, 1024] candidate rows inside every captured decode step and writes
    // the row to `output`. Cold: call it before the first decode Graph is captured.
    void set_decode_candidate_head(const at::Tensor& weight, const at::Tensor& output);

private:
    // Exact 0.8B ordinary decode body replay (rpu_qwen3_5_decode_replay.cpp).
    struct CheckedDecodeReplayState;
    bool begin_checked_decode_replay(
        const at::Tensor& hidden, at::TensorList k, at::TensorList v,
        const std::optional<at::Tensor>& mask, int64_t position, bool is_causal,
        at::IntArrayRef descriptor);
    bool try_checked_layer_body_replay();
    void finish_checked_decode_replay();
    void cancel_checked_decode_replay() noexcept;
    bool checked_decode_replay_skipped() const;
    void emit_decode_candidate_head(const at::Tensor& hidden);
    at::Tensor decode_head_w_, decode_head_out_;
    std::shared_ptr<CheckedDecodeReplayState> checked_decode_replay_state_;
    bool checked_decode_replay_scope_active_ = false;

    SdpaConfig make_sdpa_config(int mask = 1) const {
        // The adapter expands effective full-attention KV heads to NUM_CORES;
        // attn_tp() therefore gives one complete effective KV head per core.
        // Use that resolved width for temporary sizing and chunk validation.
        return {sdpa_kernel_, head_dim(), num_q_heads(), num_kv_heads(),
                /*num_cores=*/(int)attn_tp(), mask};
    }

    struct LayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor q_norm_w, k_norm_w;
        at::Tensor attn_gate_w;          // gate half of fused q_proj (deferred)
        at::Tensor input_norm_w, post_norm_w;
        at::Tensor gate_w, up_w, down_w;
        at::Tensor q_ws, k_ws, v_ws, o_ws, attn_gate_ws;
        at::Tensor gate_ws, up_ws, down_ws;
        at::Tensor gate_negative_zero, up_negative_zero, down_negative_zero;
        // GDN mixer weights (linear_attention layers only). in_proj_* / out_proj
        // are col/row-swizzled for device GEMM; conv_w is the DDR-repacked
        // [num_cores, Kc, ld_pad] conv1d weight; A_log/dt_bias/norm_w are plain.
        // The conv_dim channel order is reordered so each core's contiguous slice
        // holds [2 q-heads | 2 k-heads | 2 v-heads] (see adapter weight prep).
        at::Tensor gdn_in_z_w, gdn_out_w, gdn_A_log, gdn_dt_bias, gdn_norm_w;
        // prefill (chunk) per-path: separate q/k/v proj + conv, N_bg-padded b/a.
        at::Tensor gdn_q_w, gdn_k_w, gdn_v_w, gdn_conv_q_w, gdn_conv_k_w, gdn_conv_v_w;
        at::Tensor gdn_b_bg_w, gdn_a_bg_w;
        at::Tensor gdn_in_z_ws, gdn_out_ws;
        at::Tensor gdn_q_ws, gdn_k_ws, gdn_v_ws;
        at::Tensor gdn_b_bg_ws, gdn_a_bg_ws;
    };
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        if (use_text_gdn_input_workspace()) {
            identity.push_back(kTextGdnInputWorkspaceSite);
            const auto schedule = gdn_input_workspace_arguments(640);
            identity.insert(identity.end(), schedule.begin(), schedule.end());
        }
        if (high_precision_) {
            identity.push_back(high_precision_site());
            const auto math = high_precision_arguments();
            identity.insert(identity.end(), math.begin(), math.end());
        }
        if (num_cores() != 8) {
            const auto topology = cold_topology_arguments();
            identity.insert(identity.end(), topology.begin(), topology.end());
        }
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.push_back(static_cast<int64_t>(action_nvfp4_.enabled()));
        for (const auto& table : action_nvfp4_.values)
            append_kvinsert_cost_tensor_identity(identity, table);
        if (text_nvfp4_.enabled()) {
            identity.push_back(kTextNvfp4TablesSite);
            for (const auto& table : text_nvfp4_.values)
                append_kvinsert_cost_tensor_identity(identity, table);
        }
        identity.insert(identity.end(), {
            static_cast<int64_t>(linear_acc32_),
            static_cast<int64_t>(has_gdn_),
            static_cast<int64_t>(action_mode_),
            static_cast<int64_t>(fast_replay_enabled_),
            static_cast<int64_t>(has_qk_norm_),
            static_cast<int64_t>(use_silu_),
            static_cast<int64_t>(has_mrope_),
            static_cast<int64_t>(chunk_size_cap_),
            static_cast<int64_t>(sdpa_kernel_)});
        if (text_gateup_pair_available_) identity.insert(identity.end(),
            {text_gateup_pair_w8_ ? 3 : 2, 1, text_gateup_pair_site()});
        if (!action_mode_ && text_silu_round_mul_available_) {
            const auto args=text_silu_round_mul_arguments();
            identity.insert(identity.end(),args.begin(),args.end());
        }
        if (!action_mode_ && text_mlp_tail_available_) {
            const auto args=text_mlp_tail_arguments();
            identity.insert(identity.end(),args.begin(),args.end());
        }
        if (!action_mode_ && text_owner_rms_available_) {
            const auto args = text_owner_rms_arguments();
            identity.insert(identity.end(), args.begin(), args.end());
        }
        if (!action_mode_ && text_gdn_owner_rms_available_) {
            const auto args = text_gdn_owner_rms_arguments();
            identity.insert(identity.end(), args.begin(), args.end());
        }
        if (has_text_gdn_glast_local_capability()) {
            const auto schedule = text_gdn_glast_local_identity();
            identity.insert(identity.end(), schedule.begin(), schedule.end());
        }
        if (use_g05_gdn_local_slice(256)) {
            // Same weights, different BASE slice launch/address schedule.
            identity.insert(identity.end(), {5, 1});
            const auto schedule = g05_gdn_local_slice_arguments(256);
            identity.insert(identity.end(), schedule.begin(), schedule.end());
        }
        if (action_mode_) {
            identity.push_back(static_cast<int64_t>(action_norm_shift_available_));
            if (high_precision_) {
                identity.push_back(static_cast<int64_t>(action_newton_norm_shift_available_));
                if (use_action_newton_norm_shift(action_len_)) {
                    identity.push_back(5482108007151016838LL);
                    const auto args = action_newton_norm_shift_arguments();
                    identity.insert(identity.end(), args.begin(), args.end());
                }
            }
            identity.push_back(static_cast<int64_t>(action_mod_body_w8_));
            identity.push_back(static_cast<int64_t>(action_mod_body_fp16_));
            identity.push_back(static_cast<int64_t>(action_gated_residual_available_));
            identity.push_back(static_cast<int64_t>(action_rtc_norm_shift_available_));
            identity.push_back(static_cast<int64_t>(action_sub_gated_residual_available_));
            identity.push_back(static_cast<int64_t>(action_rtc_sub_gated_residual_available_));
            identity.push_back(static_cast<int64_t>(action_ring_rtc_residual_available_));
            identity.push_back(1); // Versioned G05 per-body modulation schedule.
            identity.push_back(action_gateup_resident_w8_ ? 4 : 3); // W8 six layers / FP16 three layers.
            identity.push_back(static_cast<int64_t>(action_gateup_resident_.size()));
            identity.push_back(static_cast<int64_t>(action_gateup_resident_available_));
            identity.push_back(static_cast<int64_t>(action_gateup_resident_w8_));
            identity.push_back(static_cast<int64_t>(action_down_partial_available_));
            append_kvinsert_cost_tensor_identity(identity, action_down_partial_);
            identity.push_back(static_cast<int64_t>(action_nvfp4_o_resident_available_));
            append_kvinsert_cost_tensor_identity(identity, action_nvfp4_o_resident_.first);
            append_kvinsert_cost_tensor_identity(identity, action_nvfp4_o_resident_.second);
            if (use_action_nvfp4_o_resident(action_len_)) {
                identity.push_back(action_nvfp4_o_resident_site());
                const auto args = action_nvfp4_o_resident_arguments(0, -1);
                identity.insert(identity.end(), args.begin(), args.end());
            }
            identity.push_back(static_cast<int64_t>(action_nvfp4_down_resident_available_));
            append_kvinsert_cost_tensor_identity(identity, action_nvfp4_down_resident_.first);
            append_kvinsert_cost_tensor_identity(identity, action_nvfp4_down_resident_.second);
            if (use_action_nvfp4_down_resident(action_len_,0)) {
                identity.push_back(action_nvfp4_down_resident_site());
                const auto args = action_nvfp4_down_resident_arguments(0, -1);
                identity.insert(identity.end(), args.begin(), args.end());
            }
            identity.push_back(static_cast<int64_t>(action_o_resident_available_));
            identity.push_back(static_cast<int64_t>(action_o_resident_n128_));
            identity.push_back(static_cast<int64_t>(action_o_resident_.size()));
            for (const auto& owner : action_o_resident_)
                append_kvinsert_cost_tensor_identity(identity, owner);
            identity.push_back(1); // Versioned KV owner/fanout schedule.
            identity.push_back(static_cast<int64_t>(action_kv_owner_available_));
            identity.push_back(static_cast<int64_t>(action_kv_owner_w8_));
            identity.push_back(1); // Versioned Q/gate double-base launch.
            identity.push_back(static_cast<int64_t>(action_q_gate_available_));
            identity.push_back(static_cast<int64_t>(action_q_gate_w8_));
            identity.push_back(1); // Versioned W8 Q/gate + KV owner launch.
            identity.push_back(static_cast<int64_t>(action_qkv_gate_available_));
            identity.push_back(3); // Bounded P0/P181/P245/P%16=1 owner transport.
            identity.insert(identity.end(),{161,449,16,1,0,245,181});
            identity.push_back(static_cast<int64_t>(action_owner_cache_joint_available_));
            if (action_owner_cache_joint_available_) {
                const auto args = action_owner_cache_joint_abi();
                identity.insert(identity.end(),args.begin(),args.end());
            }
            identity.push_back(static_cast<int64_t>(action_owner_cache_available_));
            identity.push_back(static_cast<int64_t>(action_nvfp4_qk_norm_rope_available_));
            if (action_nvfp4_qk_norm_rope_available_) {
                const auto args = action_nvfp4_qk_norm_rope_arguments();
                identity.insert(identity.end(), args.begin(), args.end());
            }
            identity.push_back(static_cast<int64_t>(action_gate_kv_owner_available_));
            if (action_gate_kv_owner_available_) {
                const auto args = action_gate_kv_owner_arguments(0);
                identity.insert(identity.end(), args.begin(), args.end());
            }
            identity.push_back(1); // Versioned DDR Gate/Up double-base launch.
            identity.push_back(static_cast<int64_t>(action_gateup_pair_available_));
            identity.push_back(static_cast<int64_t>(action_gateup_pair_w8_));
            identity.push_back(static_cast<int64_t>(action_gateup_pair_nvfp4_));
            if (action_gateup_pair_available_)
                identity.push_back(action_gateup_pair_site());
            identity.push_back(1); // Versioned rounded BASE SiLU/MUL fusion.
            identity.push_back(static_cast<int64_t>(action_silu_round_mul_available_));
            if (high_precision_) {
                identity.push_back(static_cast<int64_t>(action_silu_high_mul_available_));
                if (use_action_silu_high_mul(action_len_)) {
                    identity.push_back(5790657297908870019LL);
                    const auto args = action_silu_high_mul_arguments();
                    identity.insert(identity.end(), args.begin(), args.end());
                }
            }
            identity.push_back(static_cast<int64_t>(action_tail_available_));
            if (action_tail_available_) {
                const auto args = action_tail_arguments();
                identity.insert(identity.end(),args.begin(),args.end());
            }
            identity.push_back(static_cast<int64_t>(action_mlp_tail_available_));
            if (action_mlp_tail_available_) {
                const auto args = action_mlp_tail_arguments();
                identity.insert(identity.end(),args.begin(),args.end());
            }
            identity.push_back(static_cast<int64_t>(action_attn_sigmoid_mul_available_));
            if (action_attn_sigmoid_mul_available_) {
                const auto args = action_attn_sigmoid_mul_arguments();
                identity.insert(identity.end(),args.begin(),args.end());
            }
            identity.push_back(static_cast<int64_t>(action_nvfp4_swiglu_available_));
            if (use_action_nvfp4_swiglu(action_len_)) {
                identity.push_back(action_nvfp4_swiglu_site());
                const auto args = action_nvfp4_swiglu_arguments();
                identity.insert(identity.end(), args.begin(), args.end());
            }
            for (const auto& owner : action_gateup_resident_)
                append_kvinsert_cost_tensor_identity(identity, owner);
        }
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.q_norm_w, &weights.k_norm_w, &weights.attn_gate_w, &weights.input_norm_w,
                    &weights.post_norm_w, &weights.gate_w, &weights.up_w, &weights.down_w,
                    &weights.q_ws, &weights.k_ws, &weights.v_ws, &weights.o_ws,
                    &weights.attn_gate_ws, &weights.gate_ws, &weights.up_ws, &weights.down_ws,
                    &weights.gate_negative_zero, &weights.up_negative_zero, &weights.down_negative_zero,
                    &weights.gdn_in_z_w, &weights.gdn_out_w, &weights.gdn_A_log, &weights.gdn_dt_bias,
                    &weights.gdn_norm_w, &weights.gdn_q_w, &weights.gdn_k_w, &weights.gdn_v_w,
                    &weights.gdn_conv_q_w, &weights.gdn_conv_k_w, &weights.gdn_conv_v_w, &weights.gdn_b_bg_w,
                    &weights.gdn_a_bg_w, &weights.gdn_in_z_ws, &weights.gdn_out_ws, &weights.gdn_q_ws,
                    &weights.gdn_k_ws, &weights.gdn_v_ws, &weights.gdn_b_bg_ws, &weights.gdn_a_bg_ws}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &final_norm_w_, &action_input_w_, &action_input_b_, &action_output_w_,
                &action_output_b_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    G05Nvfp4Tables             action_nvfp4_;
    Qwen35TextNvfp4Tables      text_nvfp4_;
    bool                       action_nvfp4_o_resident_available_ = false;
    std::pair<at::Tensor, at::Tensor> action_nvfp4_o_resident_;
    bool                       action_nvfp4_down_resident_available_ = false;
    std::pair<at::Tensor, at::Tensor> action_nvfp4_down_resident_;
    std::vector<LayerWeights>  layer_weights_;
    std::vector<uint8_t>       layer_is_full_;
    at::Tensor                 cos_, sin_, final_norm_w_;
    // One private decode input. Published owners are never written again;
    // changed bytes/generation get a fresh owner, while Graph owns old leases.
    at::Tensor                 decode_cpu_input_owner_;
    std::vector<uint8_t>       decode_cpu_input_bytes_;
    uint64_t                   decode_cpu_input_generation_ = 0;
    bool                       decode_cpu_input_published_ = false;
    std::mutex                 decode_cpu_input_mutex_;
    // PREFILL interleaved M-RoPE tables (B channel; set per-forward via
    // set_prefill_rope). Empty until set → build_full_attention falls back to
    // cos_/sin_ (correct for text-only, where interleaved == arange).
    at::Tensor                 prefill_cos_, prefill_sin_;
    // Diagnostic-only [layers,1,physical_prefill_seq,hidden] DDR staging.
    // Its address is baked into a dedicated debug prefill graph and therefore
    // stays stable across that graph's REPLAY. Undefined on the default path.
    at::Tensor                 per_layer_debug_buf_;
    // Separate pre-final-norm output of the last decoder layer. The regular
    // last per-layer slot intentionally contains post-final-norm output; this
    // debug-only buffer splits those two numerical stages for localization.
    at::Tensor                 last_layer_decoder_debug_buf_;
    int64_t                    valid_prefill_len_ = -1;   // route B: real pre-pad prefill len (see set_valid_prefill_len)
    int64_t                    mrope_pos_delta_ = 0;      // W2: decode M-RoPE index offset after an image (HF rope_deltas); 0 = text-only
    int64_t                    last_prefill_chunk_size_ = 0;  // prefill 的 resolved cs(decode 不覆写); 见 last_prefill_chunk_size()
    int64_t                    chunk_size_cap_ = 0;           // cold per-handle prefill cap; 0 = auto
    bool                       linear_acc32_ = false;          // cold per-handle GEMM accumulation mode
    bool                       high_precision_ = false;        // G05 Newton norms and HIGH SiLU
    bool                       high_precision_bound_ = false;
    bool                       has_gdn_ = false;
    bool                       action_mode_ = false;
    bool                       action_norm_shift_available_ = false;
    bool                       action_newton_norm_shift_available_ = false;
    bool                       action_gated_residual_available_ = false;
    bool                       action_rtc_norm_shift_available_ = false;
    bool                       action_sub_gated_residual_available_ = false;
    bool                       action_rtc_sub_gated_residual_available_ = false;
    bool                       action_ring_rtc_residual_available_ = false;
    bool                       action_gateup_resident_available_ = false;
    bool                       action_gateup_resident_w8_ = false;
    bool                       action_down_partial_available_ = false;
    at::Tensor                 action_down_partial_; // Layer0 first672 columns, INT8[8,344064].
    bool                       action_o_resident_available_ = false;
    bool                       action_o_resident_n128_ = false;
    bool                       action_mod_body_w8_ = false;
    bool                       action_mod_body_fp16_ = false;
    bool                       action_kv_owner_available_ = false;
    bool                       action_kv_owner_w8_ = false;
    bool                       action_q_gate_available_ = false;
    bool                       action_q_gate_w8_ = false;
    bool                       action_qkv_gate_available_ = false;
    bool                       action_owner_cache_available_ = false;
    bool                       action_owner_cache_joint_available_ = false;
    bool                       action_gate_kv_owner_available_ = false;
    bool                       action_nvfp4_qk_norm_rope_available_ = false;
    bool                       action_gateup_pair_available_ = false;
    bool                       action_gateup_pair_w8_ = false;
    bool                       action_gateup_pair_nvfp4_ = false;
    bool                       action_silu_round_mul_available_ = false;
    bool                       action_silu_high_mul_available_ = false;
    bool                       action_nvfp4_swiglu_available_ = false;
    bool                       action_attn_sigmoid_mul_available_ = false;
    bool                       action_tail_available_ = false;
    bool                       action_mlp_tail_available_ = false;
    bool                       text_mlp_tail_available_ = false;
    bool                       text_silu_round_mul_available_ = false;
    bool                       text_gateup_pair_available_ = false;
    bool                       text_gateup_pair_w8_ = false;
    bool                       text_owner_rms_available_ = false;
    bool                       text_gdn_owner_rms_available_ = false;
    std::vector<at::Tensor>     action_gateup_resident_; // FP16 layers 1..3 / W8 1..6, [core, Gate | Up]
    std::vector<at::Tensor>     action_o_resident_; // W8 ACC16: six [core, four O layers] bundles.
    bool                       action_descriptor_include_io_ = false;
    int64_t                    action_descriptor_route_flags_ = 0;
    FmbGraphLifecycle          action_descriptor_graph_lifecycle_ =
        FmbGraphLifecycle::COMPOSITE_CHILD;
    bool                       fast_replay_enabled_ = false;
    bool                       initial_prefill_state_flush_skip_ = false;
    bool                       fast_replay_active_ = false;
    bool                       retained_prefill_graph_ = false;
    at::Tensor                 adaptive_mod_ref_;
    uint64_t                   adaptive_mod_live_base_ = 0;
    uint32_t                   adaptive_mod_row_offset_ = 0;
    std::vector<int64_t>       action_prefix_lens_;
    at::Tensor                 action_input_w_, action_input_b_;
    at::Tensor                 action_output_w_, action_output_b_;
    at::Tensor                 action_hidden_stage_, action_velocity_stage_;
    at::Tensor                 action_input_ref_, action_keep_mask_ref_;
    at::Tensor                 action_output_ref_;
    at::Tensor                 action_rtc_prefix_ref_;
    at::Tensor                 action_trace_pre_action_ref_;
    at::Tensor                 action_trace_action_embed_ref_;
    at::Tensor                 action_trace_final_hidden_ref_;
    at::Tensor                 action_trace_velocity_ref_;
    at::Tensor                 action_trace_post_action_ref_;
    uint64_t                   action_input_live_base_ = 0;
    uint64_t                   action_keep_mask_live_base_ = 0;
    uint64_t                   action_output_live_base_ = 0;
    uint64_t                   action_rtc_prefix_live_base_ = 0;
    uint64_t                   action_trace_pre_action_live_base_ = 0;
    uint64_t                   action_trace_action_embed_live_base_ = 0;
    uint64_t                   action_trace_final_hidden_live_base_ = 0;
    uint64_t                   action_trace_velocity_live_base_ = 0;
    uint64_t                   action_trace_post_action_live_base_ = 0;
    int64_t                    action_dim_ = 0;
    int64_t                    action_dim_pad_ = 0;
    int64_t                    action_len_ = 0;
    int64_t                    action_valid_len_ = 0;
    c10::Half                  action_euler_scale_ = c10::Half(0.0f);
    bool                       action_euler_scale_pinned_ = false;
    bool                       action_step_active_ = false;
    bool                       action_loop_active_ = false;
    bool                       action_rtc_active_ = false;
    bool                       action_trace_active_ = false;
    int64_t                    action_rtc_prefix_len_ = 0;
    int64_t                    action_num_steps_ = 1;
    double                     eps_         = 1e-6;
    bool                       has_qk_norm_ = false;
    bool                       use_silu_    = true;
    SdpaKernelType             sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;

    std::vector<at::Tensor>*   gdn_states_ = nullptr; // per-forward stash (recurrent_state)
    std::vector<at::Tensor>*   conv_states_ = nullptr; // per-forward stash (conv_state)
    // Shared [Dk*Dv] zeros buffer for the recurrent broadcast-tile ops. Stable
    // address (allocated in set_weights), baked into the graph DMAs.
    at::Tensor                 gdn_zero_;
    // GDN prefill-chunk per-layer neg_exp_A = -exp(A_log) (host-precomputed, mirrors
    // rhino's gdn_a_log tensor). Decode feeds raw A_log to a fused kernel that does
    // exp internally; the decomposed chunk path can't (a per-head [vg_c<16] device
    // exp violates elt_num%16), so we precompute on host and just multiply g by it.
    std::vector<at::Tensor>    gdn_neg_exp_A_;
    // Per-layer LIVE DDR byte-addr for the GDN recurrent/conv-state DMAs. The state
    // tensors are CALLER-SUPPLIED (a fresh Qwen3_5Cache per generation), so their
    // DDR address changes whenever the same model is reused with a different cache
    // (two e2e scenarios, or two independent generations). The state load/writeback
    // therefore go through the *_mutable DMA variants, rebound at REPLAY from these
    // slots. A FIXED DMA would bake generation-1's state address; a decode graph
    // replayed for generation-2 (same seq=1 signature) would then read/write
    // generation-1's already-freed state → use-after-free (garbage decode + heap
    // corruption). Sized ONCE in set_weights(num_layers); the graph records
    // &gdn_state_live_addr_[L], so these vectors must never resize after that.
    std::vector<uint64_t>      gdn_state_live_addr_;
    std::vector<uint64_t>      conv_state_live_addr_;
    // GDN prefill-chunk constant masks (built once in set_weights; broadcast to SPM
    // per layer or once per retained forward). tril = lower-incl-diagonal [C,C] (decay mask); strict = strict-lower
    // [C,C] (i>j). Match rhino's host-prepared masks (the M *= -1 and q*=qk_scale are
    // separate binary_scalar ops, not folded into the masks).
    at::Tensor                 gdn_tril_, gdn_strict_;

    // GDN mixer dims (set in set_weights; 0 until then).
    int64_t gdn_conv_dim_ = 0, gdn_nvh_ = 0, gdn_dk_ = 0, gdn_dv_ = 0;
    int64_t gdn_value_dim_ = 0, gdn_kc_ = 0;

    // Partial M-RoPE (kernel "partial_mrope"). Qwen3.5 rotates only the first
    // rotary_dim=64 of head_dim=256 (partial_rotary_factor 0.25), rotate-half
    // WITHIN rotary_dim, pass-through the rest. The kernel is in-place (writes
    // only the rotary span); the M-RoPE T/H/W interleaving is baked into the
    // cos/sin tables on the host side, so no strobe masks / position_ids here.
    bool    has_mrope_  = false;
    int64_t rotary_dim_ = 0;   // 2*sum(mrope_section)

    // Fixed-shape physical-SPM canary state.  The port is the complete padded
    // text input [128, 2048]; the producer owns only its checked image row slice.
    int64_t  z2_prepared_execution_len_ = 0;
    int64_t  z2_prepared_real_len_ = 0;
    uint64_t z2_prefill_cos_addr_ = 0;
    uint64_t z2_prefill_sin_addr_ = 0;
    uint32_t z2_input_port_addr_ = 0;
    uint64_t z2_epoch_ = 0;
    uint64_t z2_plan_hash_ = 0;
    bool     z2_bound_ = false;
};

}  // namespace v3
