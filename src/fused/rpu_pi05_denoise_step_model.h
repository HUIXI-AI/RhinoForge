// src/fused/rpu_pi05_denoise_step_model.h
//
// Pi0.5 num_steps fused denoise step model (FusedModelBase v3 subclass).
//
// Architecture: spec §2.1, §5.3.
//   - Inherits v3::FusedModelBase; reuses SPM allocator + GraphCache lifecycle.
//   - hidden_size_ = H_ada = 1024 (AdaRMS gemma_300m hidden).
//   - Injects action_in_proj (pre) + final PiGemmaRMSNorm + action_out_proj
//     (post) via pre_layers_fn / post_layers_fn hooks (EXT-Pi05).
//   - One step_forward() = ONE denoise step. Python adapter loops num_steps
//     times wrapped in cache.capture(sig) — 1 BUILD + (N-1) REPLAY.
//   - All per-step mutables (x_t src, cond src, v_t dst) routed via *_mutable
//     DMA wrappers (no FIXED-first scaffolding).

#pragma once

#include "pi05_nvfp4.h"
#include "fused_model_base.h"
#include "pi05_execution_topology.h"
#include "rpu_kernel_cache.h"
#include "rpu_kernel_decls.h"
#include "rpu_pi_kv_linear_contract.h"

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace v3 {

class Pi05DenoiseStepModel : public FusedModelBase {
public:
    // LayerWeights mirrors the AdaRMSModel struct so the inline-ported
    // build_layer_subgraph body compiles without symbol renames.
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
        // Optional exact-profile KV1 companions.  Stripe-gather stores logical
        // D in order; direct-cache stores core c's rotate-half pair
        // {db=c,db=c+8}.  Both remain independent 32-channel/core owners.
        at::Tensor k_kv1_w, v_kv1_w, k_kv1_ws, v_kv1_ws;
        // Optional tile-contiguous copies of q_ws/o_ws/down_ws for the
        // NVFP4 ACC16 double-buffered Linear and of gate_ws/up_ws for the
        // double-buffered GeGLU (rpu_pack_pi05_nvfp4_tile_scale).
        at::Tensor q_wts, o_wts, down_wts, gate_wts, up_wts;
    };

    explicit Pi05DenoiseStepModel(bool linear_acc32 = false);
    ~Pi05DenoiseStepModel() override;

    void set_execution_cores(int64_t cores) {
        TORCH_CHECK(cores == 4 || cores == 6 || cores == 8,
                    "Pi0.5 denoise execution cores must be 4, 6 or 8");
        TORCH_CHECK(layer_weights_.empty(),
                    "Pi0.5 denoise execution cores must precede weight installation");
        set_execution_core_count(static_cast<int>(cores));
    }

    // Distinct checkpoint KV heads behind the cache's num_kv_heads() heads, which
    // the adapter replicates (repeat_interleave) to one per attention core.
    // Split-KV attention runs a group's query heads against one core's copy, so
    // it requires a single source head; 0 (never set) keeps it off.
    void set_kv_source_heads(int64_t heads) {
        TORCH_CHECK(heads >= 1 && heads <= 8,
                    "Pi0.5 denoise KV source heads must be in [1, 8]");
        TORCH_CHECK(layer_weights_.empty(),
                    "Pi0.5 denoise KV source heads must precede weight installation");
        kv_source_heads_ = heads;
    }

    int condition_tp() const { return num_cores() == 8 ? 8 : 4; }

    std::vector<int64_t> execution_topology() const {
        TORCH_CHECK(num_layers() > 0 && !layer_weights_.empty(),
                    "Pi0.5 denoise topology requires installed model weights");
        return {1, num_cores(), attn_tp(), mlp_tp(), condition_tp(), 8,
                 logical_intermediate_size_, intermediate_size()};
    }

    std::vector<int64_t> core_profile_arguments() const {
        auto args = execution_topology();
        args.push_back(reduced_nvfp4_ ? 2 : reduced_w8a16_ ? 1 : 0);
        args.push_back(final_norm_dense_ws_.defined() &&
                       final_norm_dense_ws_.numel() != 0 ? 1 : 0);
        return args;
    }

    DecoderExecutionTopology resolve_model_execution_topology(
            int64_t nq, int64_t nkv, int64_t hd, int64_t h,
            int64_t intermediate) const override {
        if (num_cores() == 8)
            return FusedModelBase::resolve_model_execution_topology(
                nq, nkv, hd, h, intermediate);
        return resolve_pi05_reduced_physical_topology(
            num_cores(), nq, nkv, hd, h, intermediate, true);
    }

    void set_rope_position(int64_t position);

    std::vector<int64_t> planner_cache_identity() const {
        auto identity = FusedModelBase::planner_cache_identity();
        for (const auto& table : nvfp4_.values) append_kvinsert_cost_tensor_identity(identity, table);
        if (linear_acc32_) identity.insert(identity.end(), {0x4143433332LL, 1});
        return identity;
    }

    void set_configured_chunk_size(int64_t chunk_size);
    std::vector<int64_t> resolve_action_stage_domain(
        int64_t execution_len, int64_t logical_len, int64_t position,
        int64_t kv_len, int64_t cache_capacity,
        int64_t requested_chunk_size, bool prefer_pad16,
        bool loop_mode, int64_t num_steps);

    // Cold fixed-schedule constants; caller mutation cannot alter our copy.
    void set_adarms_table(const at::Tensor& table);

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
        const at::Tensor& final_norm_dense_w_scale, at::TensorList nvfp4_tensor_scales);

    // Spec §2.4 step_forward. v_t_buf written in place; returns void.
    void step_forward(
        const at::Tensor& x_t_rpu,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cond_step,
        const at::Tensor& attention_mask_4d,
        at::Tensor& v_t_buf,
        int64_t prefix_len,
        at::IntArrayRef planned_stage_descriptor = {});

    // In-graph N-step unroll (Task 1 scaffolding; mirrors WallOssActionStepModel).
    // loop_mode_ off => body_iterations==1 and the single-step path is
    // byte-identical. Task 3 fills the on-device Euler; until then the post-hook
    // still emits the single-step v_t output.
    void denoise_loop_forward(
        at::Tensor x0_rpu, std::vector<at::Tensor> k_caches,
        std::vector<at::Tensor> v_caches, at::Tensor cond_all,
        at::Tensor attention_mask_4d, at::Tensor x_out,
        double dt, int64_t prefix_len, int64_t num_steps,
        at::IntArrayRef planned_stage_descriptor = {});

protected:
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            planned_route_profile_valid_, planned_loop_mode_,
            planned_num_steps_, planned_kv_physical_rows_, planned_prefix_len_,
            planned_k_rope_cache_capacity_);
    }

    // FMB v3 mandatory virtuals
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
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
    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& manifest) const override;
    void                    build_layer_subgraph(int layer_idx,
                                                 const ChunkInfo& chunk) override;

    // max_action_dim_ sizes the Temp "x_t_spm" / "v_t_core0_spm" and is invisible
    // to the framework's params hash. The Persistent "action_out_proj_b_spm" it
    // also sizes would NOT catch a change: Align(dim*2, 256) buckets every
    // max_action_dim_ in [1,128] to the same 256 bytes, so the persistent-shape
    // guard stays silent while the two Temps keep the old size.
    int64_t subclass_layout_hash() const override {
        const int64_t original=detail::layout_mix(
            detail::layout_mix(
                detail::layout_mix(
                    detail::layout_mix(detail::layout_mix(0, max_action_dim_),
                                       use_spm_adarms_table()),
                    use_pi05_gateup_resident(chunk_size_)),
                use_pi05_adarms_body_stream()),
            use_pi05_gateup_bundle(chunk_size_) ? pi05_gateup_resident_layers_ : 0);
        const int64_t q_rope = pi05_q_rope_epilogue_opt_in_
            ? detail::layout_mix(original, 1) : original;
        const int64_t xor3 = pi05_xor3_gated_attn_opt_in_
            ? detail::layout_mix(q_rope, 2) : q_rope;
        const int64_t stripe = pi05_kv1_stripe_gather_opt_in_
            ? detail::layout_mix(xor3, 3) : xor3;
        const int64_t direct = pi05_kv1_direct_cache_opt_in_
            ? detail::layout_mix(stripe, 4) : stripe;
        const int64_t pair_owner = pi05_kv1_pair_owner_opt_in_
            ? detail::layout_mix(direct, 5) : direct;
        const int64_t geglu = pi05_geglu_m50_opt_in_
            ? detail::layout_mix(pair_owner, 6) : pair_owner;
        const auto mlp = pi05_xor3_gated_mlp_opt_in_
            ? detail::layout_mix(geglu, 7) : geglu;
        const auto w4 = pi05_w4_fastpath_opt_in_ ? detail::layout_mix(mlp, 8) : mlp;
        const auto w4_mlp = pi05_w4_fastpath_opt_in_ && pi05_xor3_gated_mlp_opt_in_
            ? detail::layout_mix(w4, 9) : w4;
        const auto w4_geglu = pi05_w4_geglu_m50_opt_in_ ? detail::layout_mix(w4_mlp, 10) : w4_mlp;
        int64_t hash = pi05_w4_geglu_n64_opt_in_ ? detail::layout_mix(w4_geglu, 11) : w4_geglu;
        if (nvfp4_.enabled()) hash = detail::layout_mix(hash, 12);
        if (pi05_nvfp4_geglu_m50_opt_in_) hash = detail::layout_mix(hash, 14);
        if (split_kv_share_admitted()) hash = detail::layout_mix(hash, 15);
        if (nvfp4_spm_scale_admitted()) hash = detail::layout_mix(hash, 16);
        if (q_spm_weight_admitted()) hash = detail::layout_mix(hash, 17);
        if (num_cores() != 8)
            for (const auto value : core_profile_arguments()) hash = detail::layout_mix(hash, value);
        return linear_acc32_ ? detail::layout_mix(hash, 32) : hash;
    }

private:
    const bool linear_acc32_;
    FmbLinearAccumulationPolicy linear_accumulation_policy() const {
        return linear_acc32_ ? FmbLinearAccumulationPolicy::ACC32
                             : FmbLinearAccumulationPolicy::ACC16;
    }
    void validate_pi05_runtime_policies(at::IntArrayRef descriptor) const;
    std::vector<int64_t> pi05_q_rope_arguments(int64_t logical_position) const;
    bool use_pi05_q_rope_epilogue(int64_t rows,int64_t logical_position) const;
    void validate_pi05_q_rope_manifest(const FmbPhysicalExecutionManifest& manifest) const;
    void validate_pi05_q_rope_policy(const FmbPrefillStageCandidate* candidate) const;
    void emit_pi05_q_rope_epilogue(const LayerWeights& weights,const ChunkInfo& chunk);
    bool use_pi05_rope_register_patch() const;
    void emit_pre_layers_body();
    void emit_post_layers_body();

    // Shared SDPA mask prepare + versioned cache. Called by both
    // step_forward and denoise_loop_forward so the prepared-mask logic lives in
    // exactly one place. Populates prepared_mask_ / prepared_mask_seq_*_.
    void prepare_sdpa_mask_cached(const at::Tensor& attention_mask_4d,
                                  int64_t prefix_len);
    // Split-KV attention for (seq_q, kv_seq_len) when its payloads are loaded
    // and the shape is admitted; launch also requires a layout that declared it.
    std::optional<Pi05SplitKvPlan> split_kv_plan(int64_t seq_q,
                                                 int64_t kv_seq_len) const;
    // Every Pi05 buffer of the layout except the split-KV K/V share and the
    // NVFP4 scale slab. `resident_table` sizes the AdaRMS table for all ten
    // bodies whatever use_pi05_adarms_body_stream() says, for the one question
    // that decision asks itself.
    std::vector<BufferDecl> base_buffer_declarations(const LayoutContext& ctx,
                                                     bool resident_table = false) const;
    std::vector<BufferDecl> capacity_buffer_declarations(int64_t max_kv_seq_len,
                                                         int64_t chunk,
                                                         bool resident_table = false) const;
    // What the planning budget leaves for the Temps next to these declarations.
    static int64_t spm_room_above(const std::vector<BufferDecl>& decls);
    // In a loop-mode profile with the SPM-scale payloads loaded: the layers
    // [0, N) that read their FP8 scales from the slab in each core's SPM, for
    // chunks of at most `chunk` rows at max_kv_seq_len. Declaration and launch
    // both ask this; 0 otherwise.
    int64_t nvfp4_spm_scale_layers(int64_t max_kv_seq_len, int64_t chunk) const;
    bool nvfp4_spm_scale_admitted() const;
    // This layer's slab run at `offset`, or 0 when its scales stay in DDR.
    uint32_t nvfp4_spm_scale_addr(int layer, int64_t offset) const;
    void emit_pi05_nvfp4_spm_scale_fill(int64_t layers);
    // With the SPM-resident-Q pair owner loaded: the N first layers of the
    // scale slab whose Q weights the first body copies into each core's SPM,
    // sized to what the slab and the split-KV share leave. Declaration and
    // launch both ask this; 0 otherwise.
    int64_t q_spm_weight_layers(int64_t max_kv_seq_len, int64_t chunk) const;
    // The same N from the slab's layers and the share already planned.
    int64_t q_spm_weight_layers(int64_t max_kv_seq_len, int64_t chunk, int64_t scale_layers,
                                std::pair<int64_t, int64_t> share) const;
    bool q_spm_weight_admitted() const;
    // This layer's per-core Q weight run, or 0 when the weight stays in DDR.
    uint32_t q_spm_weight_addr(int layer) const;
    void emit_pi05_q_spm_weight_fill(int64_t layers);
    // The layout carries "rope_rows" with its split-KV buffers.
    bool use_pi05_spm_rope_rows_layout() const;
    // For a forward of one split-KV chunk: the first logical position p whose
    // RoPE rows [p, p + rows) the first body stages in SPM, which every query
    // RoPE then reads; -1 when the tables stay in DDR. Fill and launch both
    // ask this.
    int64_t pi05_spm_rope_rows_start() const;
    void emit_pi05_spm_rope_rows_fill(int64_t start);
    // In a loop-mode profile with the SPM-share attention loaded: the layers
    // [0, first) whose split-KV attention keeps its K/V share in SPM across the
    // bodies, and second = one layer's share bytes, for chunks of at most
    // `chunk` rows at max_kv_seq_len. Declaration and launch both ask this;
    // {0, 0} otherwise.
    std::pair<int64_t, int64_t> split_kv_share_layers(int64_t max_kv_seq_len,
                                                      int64_t chunk,
                                                      bool resident_table = false) const;
    // Without the NVFP4 scale slab: whether the resident ten-body AdaRMS table
    // keeps the split-KV share at the planned geometry from holding every layer.
    bool adarms_table_limits_split_kv_share() const;
    void restore_planned_route_profile(
        at::IntArrayRef descriptor, bool loop_mode, int64_t num_steps,
        int64_t prefix_len, int64_t cache_capacity);
    void validate_planned_route_profile(
        bool loop_mode, int64_t num_steps) const;
    bool use_spm_adarms_table() const;
    bool split_kv_share_admitted() const;
    bool use_pi05_gateup_resident(int64_t rows) const;
    bool use_pi05_adarms_body_stream() const;
    bool use_pi05_gateup_bundle(int64_t rows) const;
    void validate_pi05_storage_policy(at::IntArrayRef descriptor) const;
    void emit_pi05_gateup_bundle_preload();
    bool use_pi05_graph_dma_profile(int64_t rows) const;
    int64_t prepare_pi05_graph_dma_policy(at::IntArrayRef descriptor) const;
    void validate_pi05_graph_dma_runtime() const;
    void emit_pi05_action_spm_input(const ChunkInfo& chunk);
    // Per-forward result of cold/profile/COMPLETE/Graph validation. Legacy
    // descriptor-free PASSTHROUGH and the single-step API always install zero.
    int64_t pi05_graph_dma_policy_ = 0;
    void emit_resident_gateup_mlp(const LayerWeights& weights, int64_t rows,
                                 int64_t layer_idx = 0);
    // Immutable, byte-packed controller stripes, [core, local_N/16,K/32,512].
    // These DDR owners feed planner-owned Temp slots once per Graph execution.
    at::Tensor resident_gate_w8_, resident_up_w8_;
    uint64_t resident_gate_src_base_ = 0, resident_up_src_base_ = 0;
    at::Tensor resident_gateup_bundle_;
    uint64_t resident_gateup_bundle_src_base_ = 0;
    // Row c holds core c's FP8 scales of every layer
    // (rpu_pack_pi05_nvfp4_spm_scale); undefined without the SPM-scale payloads.
    at::Tensor nvfp4_spm_scale_;
    // Row c holds core c's 512-byte planes of every layer's Q weight, in unit
    // order; undefined without the slab.
    at::Tensor q_spm_weight_;
    bool use_pi05_ring_xor3(int64_t rows) const;
    const bool pi05_ring_xor3_opt_in_;
    bool use_pi05_xor3_gated_attn(int64_t rows) const;
    int64_t pi05_denoise_residual_route(int64_t rows) const;
    void validate_pi05_xor3_gated_attn_manifest(
        const FmbPhysicalExecutionManifest& manifest) const;
    void validate_pi05_xor3_gated_attn_policy(
        const FmbPrefillStageCandidate* candidate) const;
    bool use_pi05_kv1_stripe_gather(int64_t rows) const;
    void validate_pi05_kv1_stripe_gather_manifest(
        const FmbPhysicalExecutionManifest& manifest) const;
    void validate_pi05_kv1_stripe_gather_policy(
        const FmbPrefillStageCandidate* candidate) const;
    bool use_pi05_kv1_direct_cache(int64_t rows) const;
    bool use_pi05_kv1_pair_owner(int64_t rows) const;
    // Installed weights and the cold REF: the W8 pair owner runs the layer's
    // SPM-scale NVFP4 Q projection in its launch wherever the layer has a slab.
    bool pi05_q_in_kv1_pair_owner() const;
    // The pair owner's route arguments, with the fused payload when that holds.
    std::vector<int64_t> pi05_kv1_pair_owner_route_arguments() const;
    bool use_pi05_geglu_m50(int64_t rows) const;
    std::vector<int64_t> pi05_geglu_m50_arguments() const;
    void validate_pi05_geglu_m50_manifest(
        const FmbPhysicalExecutionManifest& manifest) const;
    void validate_pi05_geglu_m50_policy(const FmbPrefillStageCandidate* candidate) const;
    void emit_pi05_geglu_m50_mlp(
        const LayerWeights& weights, const ChunkInfo& chunk, uint32_t mlp_gate, int layer_idx);
    // `norm`, when pi05_adarms_in_geglu holds, folds the MLP-side norm into the GeGLU.
    void emit_pi05_w4_xor3_mlp(
        const LayerWeights& weights, const ChunkInfo& chunk, uint32_t mlp_gate, int layer_idx,
        const Pi05AdaRmsNormOperands* norm = nullptr);
    void emit_pi05_down_and_mlp_residual(
        const LayerWeights& weights, const ChunkInfo& chunk, uint32_t mlp_gate, int layer_idx);
    bool use_pi05_xor3_gated_mlp(int64_t rows) const;
    // Both continuations over whole-row shards, the MLP-side AdaRMS norm fused
    // into the attention one. The one decision for routes and dispatch.
    bool use_pi05_xor3_gated_adarms_chain(int64_t rows) const;
    // The MLP-side AdaRMS norm in the layer's SPM-scale GeGLU launch (9..16 rows).
    bool pi05_adarms_in_geglu(int64_t rows, int layer) const;
    // Under that chain, a layer whose output stays in SPM also applies the
    // next layer's attention-side AdaRMS in its MLP continuation; that layer,
    // entered from SPM, skips the launch. Both read the planner's IO context.
    bool pi05_adarms_to_next_layer(int64_t rows);
    bool pi05_adarms_from_previous_layer(int64_t rows);
    // The normalized rows the MLP Gate/Up read: input_norm under that chain.
    uint32_t pi05_mlp_input(int64_t rows);
    std::vector<int64_t> pi05_xor3_gated_mlp_arguments() const;
    void validate_pi05_xor3_gated_mlp_manifest(
        const FmbPhysicalExecutionManifest& manifest) const;
    void validate_pi05_xor3_gated_mlp_policy(const FmbPrefillStageCandidate* candidate) const;
    void validate_pi05_kv1_direct_cache_manifest(
        const FmbPhysicalExecutionManifest& manifest) const;
    void validate_pi05_kv1_direct_cache_policy(
        const FmbPrefillStageCandidate* candidate) const;
    bool use_pi05_k_rope_insert(
        int64_t rows, int64_t position, int64_t logical_position,
        int64_t batch_size, const KvInsertSegmentPlan& plan) const;
    void rebind_kvinsert_exact_candidate(
        FmbPrefillStageCandidate& candidate, const LayoutContext& layout,
        int64_t site_id, int64_t invocation,
        const KvInsertSegmentPlan& plan) const override;
    const bool pi05_k_rope_insert_opt_in_;
    const bool pi05_gateup_resident_opt_in_;
    const bool pi05_adarms_body_stream_opt_in_;
    const int64_t pi05_gateup_resident_layers_;
    const bool pi05_elide_unused_dma_opt_in_;
    const bool pi05_spm_action_staging_opt_in_;
    const bool pi05_q_rope_epilogue_opt_in_;
    const bool pi05_xor3_gated_attn_opt_in_;
    const bool pi05_kv1_stripe_gather_opt_in_;
    const bool pi05_kv1_direct_cache_opt_in_;
    const bool pi05_kv1_pair_owner_opt_in_;
    const bool pi05_geglu_m50_opt_in_;
    const bool pi05_xor3_gated_mlp_opt_in_;
    const bool pi05_w4_fastpath_opt_in_;
    const bool pi05_w4_geglu_m50_opt_in_;
    const bool pi05_w4_geglu_n64_opt_in_;
    const bool pi05_nvfp4_geglu_m50_opt_in_;
    bool use_pi05_nvfp4_geglu_m50(int64_t rows) const;
    std::vector<int64_t> pi05_nvfp4_geglu_m50_arguments() const;
    void validate_pi05_nvfp4_geglu_m50_manifest(const FmbPhysicalExecutionManifest& manifest) const;
    void validate_pi05_nvfp4_geglu_m50_policy(const FmbPrefillStageCandidate* candidate) const;
    int64_t pi05_w4_geglu_m50_selector() const;
    bool use_pi05_w4_geglu_m50(int64_t rows) const;
    std::vector<int64_t> pi05_w4_geglu_m50_arguments() const;
    void validate_pi05_w4_geglu_m50_manifest(const FmbPhysicalExecutionManifest& manifest) const;
    void validate_pi05_w4_geglu_m50_policy(const FmbPrefillStageCandidate* candidate) const;
    bool pi05_w4_fastpath_profile_ = false;
    bool pi05_w4_weight_profile_admitted() const;
    bool pi05_nvfp4_weight_profile_admitted() const;
    bool pi05_projection_fastpath_admitted() const;
    std::vector<int64_t> pi05_w4_fastpath_arguments() const;
    void validate_pi05_w4_fastpath_manifest(const FmbPhysicalExecutionManifest& manifest) const;
    void validate_pi05_w4_fastpath_policy(const FmbPrefillStageCandidate* candidate) const;
    bool pi05_q_rope_tables_owned_ = false;
    bool pi05_k_rope_insert_w8a16_ = false;
    bool pi05_kv1_direct_pair_owners_ = false;
    bool pi05_kv1_pair_owner_full_owners_ = false;
    void emit_gated_residual(uint32_t input_spm, uint32_t residual_spm,
                             uint32_t gate_spm, int64_t rows);
    void emit_adarms_norm_shift(uint32_t input_spm, uint32_t output_spm,
                               uint32_t scale_spm, uint32_t shift_spm,
                               int64_t rows, int64_t site_id,
                               int64_t invocation = 0, bool launch = true);

    // Return the parameters' absolute SPM address, either a read-only table
    // row or the requested scratch output after DMA/GEMV.
    uint32_t adarms_gemv_to_spm(uint32_t cond_spm_addr,
                            const at::Tensor& dense_w,
                            const at::Tensor& dense_b,
                            uint32_t gemv_out_spm_addr,
                            uint32_t bias_temp_spm_addr,
                            uint32_t partial_spm_addr,
                            const at::Tensor& dense_scale,
                            int64_t table_row);
    // The SPM AdaRMS table row's address ([scale | shift | gate]) in this body.
    uint32_t pi05_adarms_table_row(int64_t table_row);
    // Issues `body`'s table rows on channels 1..7 into step_boundary_dma_.
    void emit_pi05_adarms_body_dma(int64_t body);
    // A body the next one follows: the last layer's DDR output, which the
    // Graph caller discards, also goes on channels 1..7.
    bool pi05_step_boundary_output(int layer_idx, const ChunkInfo& chunk) const;

    // --------- Weights (DDR, set_weights owns lifetime) ----------
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        // Cold-OFF preserves the accepted schema8 identity byte-for-byte.
        // KV1 uses schema9 and binds its four immutable companion owners.
        // must never share a native calibration profile or domain certificate.
        // Ordered slots: ring, K-RoPE insert, layer-0 Gate/Up W8 residency,
        // unused loop-output DMA elision, action staging through SPM, AdaRMS
        // body streaming, Gate/Up bundle cardinality, Q-RoPE epilogue, and the
        // attention XOR3+gated continuation.
        // A future SDPA/GeGLU production opt-in must extend this schema and
        // bind its real immutable member before any cost artifact can activate.
        std::vector<int64_t> identity{
            1, 0x50493035434f5354LL, 8,
            pi05_ring_xor3_opt_in_ ? 1 : 0,
            pi05_k_rope_insert_opt_in_ ? 1 : 0,
            pi05_gateup_resident_opt_in_ ? 1 : 0,
            pi05_elide_unused_dma_opt_in_ ? 1 : 0,
            pi05_spm_action_staging_opt_in_ ? 1 : 0,
            pi05_adarms_body_stream_opt_in_ ? 1 : 0,
            pi05_gateup_resident_layers_, pi05_q_rope_epilogue_opt_in_ ? 1 : 0,
            pi05_xor3_gated_attn_opt_in_ ? 1 : 0};
        if (pi05_kv1_stripe_gather_opt_in_) {
            identity[2] = 9;
            identity.push_back(1);
        } else if (pi05_kv1_direct_cache_opt_in_) {
            if (pi05_kv1_pair_owner_opt_in_) {
                identity[2] = 11;
                // Pair-owner schema: cold full-owner proof, exact profile,
                // pair map, zero compact bytes, both kernel IDs, register ABI
                // and producer/consumer grids.
                // H10 (M10/P16) binds its own consumer; H50 bytes are unchanged.
                const bool h10 = chunk_size_ == 10;
                const auto kv_contract = rpu_pi_kv_linear::pair(false);
                identity.insert(identity.end(), {
                    5, pi05_kv1_pair_owner_full_owners_ ? 1 : 0,
                    planned_num_steps_, 800, chunk_size_, h10 ? 16 : 64, 800 + chunk_size_,
                    1024, 4096, 18, 8, 8,
                    256, 32, 16, 1024, 2048,
                    0, 0, 8, 2048, 1024, 2048,
                    static_cast<int64_t>(pi_wall_kernel_id(kv_contract.ordinal)),
                    static_cast<int64_t>(h10
                        ? KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M10D256P16
                        : KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64),
                    68, kv_contract.grid_x, kv_contract.grid_y, 1, 0, 68, h10 ? 1 : 4, 1, 1, 0,
                    rope_position_ >= 0 ? rope_position_ : planned_prefix_len_,
                    kv_contract.tile.m_tile, kv_contract.tile.n_tile, kv_contract.tile.k_tile});
            } else {
                identity[2] = 10;
                // Direct-cache schema, route mode, exact profile, pair map,
                // fixed producer/direct consumer IDs, register ABI and grid.
                const auto kv_tile = rpu_pi_kv_linear::stripe_tile();
                identity.insert(identity.end(), {
                    3, pi05_kv1_direct_pair_owners_ ? 1 : 0,
                    planned_num_steps_, 800, 50, 64, 850, 1024, 4096, 18, 8, 8,
                    256, 32, 16, 1024, 2048, kv_tile.m_tile, kv_tile.n_tile, kv_tile.k_tile, 0, 8, 4, 1,
                    0x4b56314c494e4541LL,  // "KV1LINEA" fixed producer identity
                    0x4b56314449524543LL,  // "KV1DIREC" direct-cache identity
                    0});  // logical RoPE is a replay-mutable register
            }
        }
        if (pi05_geglu_m50_opt_in_) {
            TORCH_CHECK(identity[2] == 11,
                        "Pi M50 GeGLU cost identity requires KV1 pair-owner schema11");
            identity[2] = 12;
            // One canonical ABI vector binds the real cold flag, exact
            // profile, kernel ID, register mapping and memory contract.
            const auto geglu = pi05_geglu_m50_arguments();
            identity.insert(identity.end(), geglu.begin(), geglu.end());
        }
        if (pi05_xor3_gated_mlp_opt_in_) {
            TORCH_CHECK(identity[2] == (pi05_w4_fastpath_opt_in_ ? 11 : 12),
                        "Pi MLP XOR3+gated cost identity requires its W4 pair-owner "
                        "or W8 GeGLU schema");
            if (!pi05_w4_fastpath_opt_in_) identity[2] = 13;
            const auto mlp = pi05_xor3_gated_mlp_arguments();
            identity.insert(identity.end(), mlp.begin(), mlp.end());
        }
        if (pi05_w4_fastpath_opt_in_) {
            const auto previous_schema = identity[2];
            identity[2] = pi05_xor3_gated_mlp_opt_in_ ? 15 : 14;
            identity.push_back(previous_schema);
            const auto w4 = pi05_w4_fastpath_arguments();
            identity.insert(identity.end(), w4.begin(), w4.end());
        }
        if (pi05_w4_geglu_m50_opt_in_) {
            TORCH_CHECK(identity[2] == 15 && pi05_w4_fastpath_opt_in_ &&
                            pi05_xor3_gated_mlp_opt_in_ && !pi05_geglu_m50_opt_in_,
                        "Pi W4 GeGLU cost identity requires the W4 MLP schema15");
            identity[2] = 16;
            const auto geglu = pi05_w4_geglu_m50_arguments();
            identity.insert(identity.end(), geglu.begin(), geglu.end());
        }
        if (pi05_w4_geglu_n64_opt_in_) {
            TORCH_CHECK(identity[2] == 16 && pi05_w4_geglu_m50_opt_in_ &&
                            pi05_w4_weight_profile_admitted() &&
                            layer_weights_[0].q_ws.size(0) == 128,
                        "Pi W4 GeGLU N64 cost identity requires the GS128 schema16 stack");
            // Canonical GeGLU arguments above already bind ABI2/cold N64 flag.
            identity[2] = 17;
        }
        if (pi05_nvfp4_geglu_m50_opt_in_) {
            TORCH_CHECK(identity[2] == 15 && nvfp4_.enabled() && pi05_nvfp4_weight_profile_admitted(),
                        "Pi NVFP4 GeGLU cost identity requires its NVFP4 continuation owners");
            identity[2] = 19;  // ACC16 GeGLU ABI2, distinct from historical ACC32 schema18.
            const auto geglu = pi05_nvfp4_geglu_m50_arguments();
            identity.insert(identity.end(), geglu.begin(), geglu.end());
        }
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.gate_proj_w, &weights.up_proj_w, &weights.down_proj_w, &weights.q_ws,
                    &weights.k_ws, &weights.v_ws, &weights.o_ws, &weights.gate_ws,
                    &weights.up_ws, &weights.down_ws, &weights.attn_dense_w, &weights.attn_dense_b,
                    &weights.mlp_dense_w, &weights.mlp_dense_b, &weights.attn_dense_ws, &weights.mlp_dense_ws}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
            if (pi05_kv1_stripe_gather_opt_in_ ||
                (pi05_kv1_direct_cache_opt_in_ &&
                 !pi05_kv1_pair_owner_opt_in_)) {
                for (const auto* tensor : {
                        &weights.k_kv1_w, &weights.v_kv1_w,
                        &weights.k_kv1_ws, &weights.v_kv1_ws}) {
                    append_kvinsert_cost_tensor_identity(identity, *tensor);
                }
            }
        }
        for (const auto* tensor : {
                &final_norm_dense_w_, &final_norm_dense_b_, &final_norm_dense_ws_, &action_in_proj_w_,
                &action_in_proj_b_, &action_out_proj_w_, &action_out_proj_b_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        append_kvinsert_cost_tensor_identity(identity, adarms_table_);
        // Table dimensions/stride are installed profile facts for Q's epilogue.
        append_kvinsert_cost_tensor_identity(identity, cos_);
        append_kvinsert_cost_tensor_identity(identity, sin_);
        if (linear_acc32_) identity.insert(identity.end(), {0x4143433332LL, 1});
        return identity;
    }

    int64_t logical_intermediate_size_ = 0;
    bool reduced_w8a16_ = false;
    bool reduced_nvfp4_ = false;
    Pi05Nvfp4Tables nvfp4_;
    std::vector<LayerWeights> layer_weights_;
    at::Tensor final_norm_dense_w_, final_norm_dense_b_;
    at::Tensor final_norm_dense_ws_;  // W8A16 scale (empty ⇒ fp16)
    at::Tensor adarms_table_;  // [steps, 2 * layers + 1, 3H], loop only
    uint64_t adarms_table_src_base_ = 0;
    // DMAs left running across a step boundary of a body-streamed table: the
    // last layer's output copy and the next body's table rows. The next
    // body's pre-layers joins them after the action-embedding chain.
    DeferredDmaChannels step_boundary_dma_;
    int64_t adarms_prefetch_body_ = -1;  // the last body whose rows were issued
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
    bool planned_route_profile_valid_ = false;
    bool planned_loop_mode_ = false;
    int64_t planned_num_steps_ = 1;
    int64_t planned_kv_physical_rows_ = 0;
    int64_t planned_prefix_len_ = 0;
    int64_t planned_k_rope_cache_capacity_ = 0;
    // adarms_table_limits_split_kv_share() per planned geometry: declare_buffers
    // asks it on every forward, replays included.
    mutable std::array<int64_t, 5> adarms_table_share_key_{-1, -1, -1, -1, -1};
    mutable bool adarms_table_limits_share_ = false;
    double eps_ = 1e-6;

    // --------- Per-call inputs (set in step_forward) -------------
    // Keepalives for the three mutable bases below — see pitfalls.md C-1: the
    // bases are dereferenced when the graph executes, which is after this
    // model's forward has returned.
    at::Tensor x_t_ref_, cond_ref_, v_t_ref_;

    // --------- Mutable bases (rewritten per step_forward) --------
    // Read by *_mutable DMA wrappers at REPLAY time via update_dma_kernel.
    uint64_t x_t_src_base_  = 0;
    uint64_t cond_src_base_ = 0;
    uint64_t v_t_dst_base_  = 0;

    // --------- In-graph N-step unroll (Task 1 scaffolding) -------
    // loop_mode_ off => body_iterations==1 and the single-step path is
    // byte-identical. Task 3 wires the on-device Euler + per-iteration DMAs.
    bool      loop_mode_ = false;
    int64_t   num_steps_ = 1;
    c10::Half dt_;                 // baked at BUILD (x_t += dt*v_t); const across REPLAY
    bool      dt_pinned_ = false;
    uint64_t  x0_src_base_ = 0;        // initial noise DDR base (iter 0 load)
    uint64_t  cond_all_src_base_ = 0;  // [num_steps, h_ada] cond stack DDR base
    uint64_t  x_out_dst_base_ = 0;     // final action DDR base
    // Keepalives (prevent GC mid-forward).
    at::Tensor x0_ref_, cond_all_ref_, x_out_ref_;

    // --------- Stable device staging tensor (set_weights allocates) ----
    // Spec §3.3: data_ptr stable across step / sample_actions. Passed to
    // run_all_layers as hidden_states to satisfy FMB hidden_size_=1024
    // shape check; layer 0's emit_layer_input_dma reads from it via the
    // FMB-managed hidden_in_src_base_ mutable mechanism (no separate [A5]
    // broadcast needed — see Corrections Log §C5).
    at::Tensor action_emb_stage_;

    // --------- Cached dims (set in set_weights) -------------------
    int64_t max_action_dim_ = 0;
    int64_t chunk_size_     = 0;
    int64_t local_q_heads_  = 0;
    int64_t local_kv_dim_   = 0;

    // --------- SDPA mask cache (live identity + mutation version) ------
    at::Tensor prepared_mask_input_ref_;
    int64_t prepared_mask_input_version_ = -1;
    PreparedMask prepared_mask_;  // global struct from rpu_kernel_decls.h
    int64_t prepared_mask_seq_q_ = 0;
    int64_t prepared_mask_seq_k_ = 0;
    bool    prepared_mask_is_causal_ = false;
    // Split-KV mask slices per (seq_q, kv_seq_len), [8, rows_pad, 16 * width]:
    // stable device slots (Graph DMA bakes their address), refreshed with the
    // prepared mask and kept until the model goes away.
    std::map<std::pair<int64_t, int64_t>, at::Tensor> split_kv_mask_slots_;
    // (chunk_size, max_kv_seq_len) of every layout that declared skv_* buffers,
    // sized for every row count up to the chunk.
    std::set<std::pair<int64_t, int64_t>> split_kv_layouts_;
    int64_t kv_source_heads_ = 0;  // set_kv_source_heads(); 0 = unknown
};

}  // namespace v3

// ============================================================================
// Public C API (file-scope; mirrors rpu_pi05_create / set_weights / forward).
// ============================================================================

int64_t rpu_pi05_denoise_step_create(bool linear_acc32);
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
    const at::Tensor& final_norm_dense_w_scale, const std::optional<std::vector<at::Tensor>>& nvfp4_tensor_scales);

void rpu_pi05_denoise_step_forward(
    int64_t handle,
    const at::Tensor& x_t_rpu,
    std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches,
    const at::Tensor& cond_step,
    const at::Tensor& attention_mask_4d,
    at::Tensor v_t_buf,
    int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor);

std::vector<int64_t> rpu_pi05_denoise_step_resolve_action_stage_domain(
    int64_t handle,
    int64_t execution_len,
    int64_t logical_len,
    int64_t position,
    int64_t kv_len,
    int64_t cache_capacity,
    int64_t requested_chunk_size,
    bool prefer_pad16,
    bool loop_mode,
    int64_t num_steps);
