// rpu_qwen3_5_moe_model.h — independent Qwen3.5 sparse-MoE fused model.
//
// Qwen3.5 interleaves full-attention layers and Gated-DeltaNet (GDN) linear
// layers per `config.layer_types`.
// This class drives a single all-layers-once fused forward and dispatches each
// layer by type:
//   full → build_full_attention (mirrors the Qwen3 causal-decoder path)
//   GDN  → build_gdn(decode): recurrent (decode) / chunked (prefill), by chunk.len
//
// Full-attention path implements: per-head QK-norm, gated-attention output gate,
// and partial M-RoPE (kernel "partial_mrope" — rotate-half within rotary_dim,
// pass-through the rest; in-place). M-RoPE T/H/W interleaving is baked into the
// host cos/sin tables (text-only degenerates to 1D partial RoPE). The copied
// mixer core retains the dense model's contract; the sparse tail has its own
// operator and fusion acceptance gates.
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

struct Qwen3_5MoeStaticConfig {
    int64_t num_experts = 256;
    int64_t top_k = 8;
    int64_t routed_intermediate = 512;
    int64_t shared_intermediate = 512;
    int64_t tensor_parallel = 8;
    bool normalize_topk = true;

    void validate(int64_t hidden_size) const;
};

class Qwen3_5MoeModel : public FusedModelBase {
public:
    struct MoeLayerWeights {
        at::Tensor router_w;
        at::Tensor routed_gate_w, routed_up_w, routed_down_w;
        at::Tensor routed_gate_scale, routed_up_scale, routed_down_scale;
        int64_t routed_weight_mode = -1;
        at::Tensor shared_gate_w, shared_up_w, shared_down_w;
        // [16*shared_tp,H]: each core owns 16 repeated rows and selects lane 0.
        at::Tensor shared_scalar_gate_w;
    };

    Qwen3_5MoeModel();
    ~Qwen3_5MoeModel() override;
    void set_execution_cores(int64_t cores);
    std::vector<int64_t> execution_topology() const;
    std::vector<int64_t> execution_topology_v2() const;

    // Per-layer lists are length == num_layers. GDN-layer slots hold undefined
    // (empty) attention tensors; only full-layer slots are read.  Every layer
    // owns a sparse-MoE tail; no dense MLP weights or state are shared.
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        at::TensorList attn_gate_list,
        at::TensorList q_scale_list, at::TensorList k_scale_list,
        at::TensorList v_scale_list, at::TensorList o_scale_list,
        at::TensorList attn_gate_scale_list,
        at::TensorList input_norm_list, at::TensorList post_norm_list,
        at::TensorList router_list,
        at::TensorList routed_gate_list, at::TensorList routed_up_list,
        at::TensorList routed_down_list,
        at::TensorList routed_gate_scale_list,
        at::TensorList routed_up_scale_list,
        at::TensorList routed_down_scale_list,
        at::TensorList shared_gate_list, at::TensorList shared_up_list,
        at::TensorList shared_down_list,
        at::TensorList shared_scalar_gate_list,
        const at::Tensor& cos, const at::Tensor& sin,
        const at::Tensor& final_norm_w,
        const at::Tensor& expert_ids, const at::Tensor& token_ids,
        at::IntArrayRef layer_is_full,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size,
        int64_t num_experts, int64_t top_k,
        int64_t routed_intermediate, int64_t shared_intermediate,
        int64_t tensor_parallel, int64_t routed_weight_mode,
        double eps, bool use_silu,
        at::IntArrayRef mrope_section,   // [T,H,W] half-dims; empty → 1D RoPE
        // GDN mixer weights (linear_attention slots filled; full slots empty). z/out are
        // col/row-swizzled; gdn_norm is plain; A_log/dt_bias [H] (8-slot padded). q/k/v
        // proj, per-path conv, N_bg b/a are the per-path lists below (decode + prefill).
        at::TensorList gdn_in_z_list, at::TensorList gdn_out_list,
        at::TensorList gdn_in_z_scale_list,
        at::TensorList gdn_out_scale_list,
        at::TensorList gdn_A_log_list, at::TensorList gdn_dt_bias_list,
        at::TensorList gdn_norm_list,
        int64_t gdn_num_v_heads, int64_t gdn_key_head_dim,
        int64_t gdn_value_head_dim, int64_t gdn_conv_dim, int64_t gdn_conv_kernel,
        // Prefill (chunk) per-path weights. Every list is length num_layers;
        // full-attention slots are empty and GDN slots are populated.
        at::TensorList gdn_q_list, at::TensorList gdn_k_list,
        at::TensorList gdn_v_list,
        at::TensorList gdn_q_scale_list,
        at::TensorList gdn_k_scale_list,
        at::TensorList gdn_v_scale_list,
        at::TensorList gdn_cq_list,
        at::TensorList gdn_ck_list, at::TensorList gdn_cv_list,
        at::TensorList gdn_b_bg_list, at::TensorList gdn_a_bg_list);

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
        int64_t planned_chunk_size,
        at::IntArrayRef planned_stage_descriptor);

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
    void set_linear_acc32(bool enabled);
    void set_fast_replay(bool enabled);
    void set_retained_prefill_graph(bool enabled);
    int64_t resolve_prefill_chunk_size(int64_t execution_len);
    std::vector<int64_t> resolve_prefill_stage_domain(
        int64_t execution_len, int64_t logical_len,
        int64_t planning_chunk_size_override = -1);
    std::vector<int64_t> resolve_decode_stage_descriptor();

protected:
    DecoderExecutionTopology resolve_model_execution_topology(
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size) const override;
    int64_t                 subclass_layout_hash() const override;
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
    std::vector<int64_t> kvinsert_cost_weight_identity() const override;

private:
    std::vector<BufferDecl> declare_legacy_buffers(const LayoutContext& ctx) const;
    std::vector<int64_t> gdn_mask_schedule_arguments(
        const FmbThreeStageChunkPlan& plan) const;
    FmbForwardOperandResidency gdn_mask_residency_for_candidate(
        const LayoutContext& layout) const;

    // The admitted TP6 profile is deliberately mixed: residual/routing/expert
    // ownership is six-wide while attention, GDN, router projection and the
    // untied head remain four-wide. Physical KV stripes remain eight.
    int gdn_tp() const { return num_cores() == 6 ? 4 : num_cores(); }
    int router_tp() const { return num_cores() == 6 ? 4 : num_cores(); }
    int routing_tp() const { return num_cores(); }
    int routed_expert_tp() const { return num_cores(); }
    int shared_expert_tp() const { return num_cores(); }
    int lm_head_tp() const { return num_cores() == 6 ? 4 : num_cores(); }
    int64_t physical_routed_intermediate() const {
        return routed_expert_tp() == 6 ? 576 : moe_.routed_intermediate;
    }
    int64_t physical_shared_intermediate() const {
        return shared_expert_tp() == 6 ? 576 : moe_.shared_intermediate;
    }
    int64_t text_ring_route(int64_t rows, int64_t cols) const;
    std::vector<int64_t> cold_topology_arguments() const;
    std::vector<int64_t> exact_profile_arguments() const;

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
    void launch_linear(
        int64_t chunk_idx,
        uint32_t input, const at::Tensor& weight, uint32_t output,
        int64_t m, int64_t n, int64_t k, int partition, int num_cores,
        uint32_t bias_spm_addr = 0, const at::Tensor& scale = {},
        bool mlp_slice = false);
    void launch_grouped_linear(
        int64_t chunk_idx,
        uint32_t input, const at::Tensor& weight,
        uint32_t m_sizes, const at::Tensor& scale,
        uint32_t output, int64_t n, int64_t k,
        int64_t num_experts, bool is_col_parallel,
        int num_cores, int64_t weight_mode);
    FmbLinearRouteSelector linear_route_selector(
        const at::Tensor& weight, int64_t rows) const;
    static int64_t linear_invocation(
        int64_t chunk_idx, FmbLinearRouteSelector selector,
        bool mlp_slice = false) {
        // Mixer indices and MLP token-start indices use disjoint domains.
        return (chunk_idx * 2 + (mlp_slice ? 1 : 0)) * 8 +
            static_cast<int64_t>(selector);
    }
    void consume_manifest_route(
        FmbRouteFamily family, int64_t site_id,
        int64_t selector, int64_t invocation = 0);
    // Full-attention layer emission (Qwen3 causal-decoder path; mirrors
    // CausalDecoderModel::build_layer_subgraph minus M-RoPE / DeepStack).
    void build_full_attention(int layer_idx, const ChunkInfo& chunk);
    // Shared post-mixer tail (post_norm + MLP + final_norm + output). Contract:
    // residual stream in "input_norm" (== "residual2" alias). See the .cpp definition.
    void emit_mlp_and_output(int layer_idx, const ChunkInfo& chunk);
    c10::optional<at::Tensor> optional_scale(const at::Tensor& scale) const;
    void append_mlp_manifest_routes(
        FmbPhysicalExecutionManifest& manifest, const ChunkInfo& chunk) const;
    void emit_router_and_shuffle(
        int layer_idx, int64_t invocation, int64_t t, uint32_t input_spm_addr);
    void emit_shared_expert(
        int layer_idx, int64_t invocation, int64_t t, uint32_t input_spm_addr);
    void emit_routed_experts(
        int layer_idx, int64_t invocation, int64_t t, uint32_t input_spm_addr);
    void emit_moe_merge_and_residual(
        int64_t invocation, int64_t t,
        uint32_t residual_spm_addr, uint32_t output_spm_addr);
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
    // 0 = 本 handle 还没跑过 prefill。与 QWEN3_5_TEXT_CHUNK cap 配套用于对账。
    int64_t last_prefill_chunk_size() const { return last_prefill_chunk_size_; }

private:

    SdpaConfig make_sdpa_config(int mask = 1) const {
        // The adapter expands effective full-attention KV heads to attn_tp();
        // attn_tp() therefore gives one complete effective KV head per core.
        // Use that resolved width for temporary sizing and chunk validation.
        return {sdpa_kernel_, head_dim(), num_q_heads(), num_kv_heads(),
                /*num_cores=*/(int)attn_tp(), mask};
    }

    struct LayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor q_scale, k_scale, v_scale, o_scale;
        at::Tensor q_norm_w, k_norm_w;
        at::Tensor attn_gate_w;          // gate half of fused q_proj (deferred)
        at::Tensor attn_gate_scale;
        at::Tensor input_norm_w, post_norm_w;
        // GDN mixer weights (linear_attention layers only). in_proj_* / out_proj
        // are col/row-swizzled for rpu GEMM; conv_w is the DDR-repacked
        // [gdn_tp, Kc, ld_pad] conv1d weight; A_log/dt_bias/norm_w are plain.
        // The conv_dim channel order is reordered so each core's contiguous slice
        // holds [2 q-heads | 2 k-heads | 2 v-heads] (see adapter weight prep).
        at::Tensor gdn_in_z_w, gdn_out_w, gdn_A_log, gdn_dt_bias, gdn_norm_w;
        at::Tensor gdn_in_z_scale, gdn_out_scale;
        // prefill (chunk) per-path: separate q/k/v proj + conv, N_bg-padded b/a.
        at::Tensor gdn_q_w, gdn_k_w, gdn_v_w, gdn_conv_q_w, gdn_conv_k_w, gdn_conv_v_w;
        at::Tensor gdn_q_scale, gdn_k_scale, gdn_v_scale;
        at::Tensor gdn_b_bg_w, gdn_a_bg_w;
    };
    std::vector<LayerWeights>  layer_weights_;
    Qwen3_5MoeStaticConfig       moe_;
    std::vector<MoeLayerWeights> moe_layer_weights_;
    at::Tensor                   expert_ids_, token_ids_;
    // TP6 only: fixed Tensor-owned DDR bridge from router4 full logits on
    // core0 to the route6 SPM domain. Capacity matches the route tables and
    // its address is stable for Graph capture/replay.
    at::Tensor                   router_bridge_;
    std::vector<uint8_t>       layer_is_full_;
    at::Tensor                 cos_, sin_, final_norm_w_;
    // PREFILL interleaved M-RoPE tables (B channel; set per-forward via
    // set_prefill_rope). Empty until set → build_full_attention falls back to
    // cos_/sin_ (correct for text-only, where interleaved == arange).
    at::Tensor                 prefill_cos_, prefill_sin_;
    int64_t                    valid_prefill_len_ = -1;
    int64_t                    mrope_pos_delta_ = 0;
    int64_t                    last_prefill_chunk_size_ = 0;
    int64_t                    chunk_size_cap_ = 0;
    bool                       linear_acc32_ = false;
    bool                       has_gdn_ = false;
    bool                       fast_replay_enabled_ = false;
    bool                       fast_replay_active_ = false;
    bool                       retained_prefill_graph_ = false;
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
    // per layer). tril = lower-incl-diagonal [C,C] (decay mask); strict = strict-lower
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

};

}  // namespace v3
