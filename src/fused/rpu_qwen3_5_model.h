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
#include "rpu_helpers.h"          // SdpaConfig, SdpaKernelType, sdpa_is_valid_chunk_size

#include <ATen/ATen.h>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

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
        at::TensorList gdn_a_bg_scale_list);

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
    // Tables are [N, rotary_dim/2] fp16 on RPU, indexed by absolute position
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
    void set_fast_replay(bool enabled);
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
    void stage_action_chunk_size(uint64_t token, int64_t chunk_size);
    void enable_execution_reconfigure();
    void set_retained_prefill_graph(bool enabled);

protected:
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
    int gdn_tp() const { return num_cores() == 8 ? 8 : 4; }
    int lm_head_tp() const { return num_cores() == 8 ? 8 : 4; }
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
        uint32_t bias_spm_addr = 0, const at::Tensor& scale = {});
    // Full-attention layer emission (Qwen3 causal-decoder path; mirrors
    // CausalDecoderModel::build_layer_subgraph minus M-RoPE / DeepStack).
    void build_full_attention(int layer_idx, const ChunkInfo& chunk);
    // Shared post-mixer tail (post_norm + MLP + final_norm + output). Contract:
    // residual stream in "input_norm" (== "residual2" alias). See the .cpp definition.
    void emit_mlp_and_output(int layer_idx, const ChunkInfo& chunk);
    void load_adaptive_mod_row(int64_t row);
    void apply_adaptive_norm(
        uint32_t input, uint32_t output, const ChunkInfo& chunk);
    void apply_adaptive_residual_gate(
        uint32_t output, uint32_t residual, const ChunkInfo& chunk);
    void apply_raw_residual_gate(
        uint32_t output, uint32_t residual, uint32_t gate,
        int64_t seq_len);
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
    // prefill (chunk.len = C*N, C=64) = chunked delta-rule. Reads residual1 and writes
    // residual2 = residual1 + mixer; updates recurrent_state + conv_state DDR caches.
    void build_gdn(int layer_idx, const ChunkInfo& chunk, bool decode);
    at::Tensor* conv_state_for_layer(int layer_idx);
    // PREFILL 时 planner 实际选中的 chunk_size。基类的 get_last_resolved_chunk_size()
    // 每次 compute_chunks 都覆写,而 decode(seq_len=1 → 固定 cs=16)跑在 prefill 之后,
    // 于是事后读基类那份永远是 16。这份只在 prefill forward 里更新,拿得到真实值。
    // 0 = 本 handle 还没跑过 prefill。与 cold per-handle cap 配套用于对账。
    int64_t last_prefill_chunk_size() const { return last_prefill_chunk_size_; }

private:

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
        // GDN mixer weights (linear_attention layers only). in_proj_* / out_proj
        // are col/row-swizzled for rpu GEMM; conv_w is the DDR-repacked
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
        if (num_cores() != 8) {
            const auto topology = cold_topology_arguments();
            identity.insert(identity.end(), topology.begin(), topology.end());
        }
        append_kvinsert_cost_scalar_identity(identity, eps_);
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
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.q_norm_w, &weights.k_norm_w, &weights.attn_gate_w, &weights.input_norm_w,
                    &weights.post_norm_w, &weights.gate_w, &weights.up_w, &weights.down_w,
                    &weights.q_ws, &weights.k_ws, &weights.v_ws, &weights.o_ws,
                    &weights.attn_gate_ws, &weights.gate_ws, &weights.up_ws, &weights.down_ws,
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

    std::vector<LayerWeights>  layer_weights_;
    std::vector<uint8_t>       layer_is_full_;
    at::Tensor                 cos_, sin_, final_norm_w_;
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
    bool                       has_gdn_ = false;
    bool                       action_mode_ = false;
    bool                       action_descriptor_include_io_ = false;
    int64_t                    action_descriptor_route_flags_ = 0;
    FmbGraphLifecycle          action_descriptor_graph_lifecycle_ =
        FmbGraphLifecycle::COMPOSITE_CHILD;
    bool                       fast_replay_enabled_ = false;
    bool                       fast_replay_active_ = false;
    bool                       retained_prefill_graph_ = false;
    at::Tensor                 adaptive_mod_ref_;
    uint64_t                   adaptive_mod_live_base_ = 0;
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
