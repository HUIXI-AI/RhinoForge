// rpu_qwen3_5_vision_model.h — Qwen3.5 Vision Encoder (Option 2, deepstack removed).
//
// ViT encoder for the supported Qwen3.5 2B/4B profile
// (24/1024/16/4096). Shape plumbing for 0.8B/9B remains a validation candidate,
// not production admission. HF's Qwen3.5 vision model inherits Qwen3-VL and
// explicitly removes DeepStack; this subsystem therefore implements the shared
// encoder body plus one final patch merger, with no deepstack outputs or
// text-layer injections.
//
// The production graph contains STEP0 (folded patch embedding + interpolated
// position embedding), the profile's encoder blocks, and the patch merger. The
// merger can write either a standalone pooler buffer or directly into the image-token rows
// of the text hidden tensor through mutable-destination DMA.
//
// 2D RoPE uses `rope_2d_spm`; FreqCos / FreqSin are static tables
// `[max_hw, head_dim/4]` set once via set_rope. position_idx is a
// `[max_seq, 2] int16` keepalive containing one absolute (h, w) table index
// pair per token.
//
// Encoding runs as a KV_FIRST two-phase bidirectional pass (mirrors gemma):
//   Phase 1 emit_kv_first_body()   — LN1 → Q/K/V Linear+bias → 2D-rope → insert
//                                    all K/V at absolute pos → stash rope'd Q to DDR.
//   Phase 2 build_layer_subgraph() — load Q back → SDPA over full KV (bidir) →
//                                    o_proj+AllReduce+resid → LN2 → fc1+GELU →
//                                    fc2+AllReduce+resid.
#pragma once

#include "fused_model_base.h"
#include "rpu_helpers.h"          // SdpaConfig, SdpaKernelType, sdpa_is_valid_chunk_size

#include <ATen/ATen.h>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace v3 {

class SpmPortView;
struct SpmDense2DSpec;

class Qwen3_5VisionModel : public FusedModelBase {
public:
    struct LayerWeights {
        // All [out, in] fp16 DDR, swizzled per partition (Python adapter handles).
        at::Tensor q_w, k_w, v_w, o_w;       // attention (col/col/col/row partition)
        at::Tensor fc1_w, fc2_w;              // MLP (col/row partition)
    };

    struct LayerBiasNorm {
        at::Tensor ln1_w, ln1_b, ln2_w, ln2_b;
        at::Tensor q_b, k_b, v_b, o_b;
        at::Tensor fc1_b, fc2_b;
    };

    Qwen3_5VisionModel();

    // Cold finite tower topology. A root budget of six uses this four-core
    // tower; physical DDR controller striping remains eight-wide.
    void configure_execution_cores(int64_t cores);
    std::vector<int64_t> execution_topology() const;

    // Cold precision policy: public explicit ACC32 may pair with ultra ERF;
    // default ACC16 and independently selected ACC32 retain standard ERF.
    // Both choices must be bound before set_weights().
    void set_linear_acc32(bool enabled);
    void set_gelu_erf_ultra(bool enabled);
    // The verified 35B-A3B bridge selects both independent cold choices.
    void set_moe_precision(bool linear_acc32, bool gelu_erf_ultra);

    // set_weights — per-layer weight + bias storage. QKV bias is present
    // (Qwen3_5VisionAttention sets bias=True on the fused qkv Linear; the Python
    // adapter pre-splits the fused [3*dim, dim] weight into q/k/v and the [3*dim]
    // bias into q_b/k_b/v_b). No deepstack_visual_indexes param.
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList fc1_w_list, at::TensorList fc2_w_list,
        at::TensorList ln1_w_list, at::TensorList ln1_b_list,
        at::TensorList ln2_w_list, at::TensorList ln2_b_list,
        at::TensorList q_b_list, at::TensorList k_b_list,
        at::TensorList v_b_list, at::TensorList o_b_list,
        at::TensorList fc1_b_list, at::TensorList fc2_b_list,
        int64_t num_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps);

    // set_rope_tables — register FreqCos / FreqSin static tables [max_hw, head_dim/4]
    // FP16 DDR (indexed by BOTH row_idx and col_idx) + pre-alloc the position_idx
    // keepalive [MAX_KEEPALIVE_SEQ, 2] int16.
    void set_rope_tables(const at::Tensor& freq_cos, const at::Tensor& freq_sin);


    void set_patch_embed(const at::Tensor& pe_w, const at::Tensor& pe_b);

    // set_merger — opt into the in-graph patch merger (runs as a post_layers_fn,
    // ONCE after the layer loop, inside the same graph scope). Moves HF's
    // `Qwen3VLVisionPatchMerger` off the CPU: LayerNorm(hidden) → reshape
    // [S, hidden] → [S/4, hidden*4] (spatial_merge_size²=4 patches per token) →
    // fc1(+bias)+GELU → fc2(+bias) → text hidden. The tower's forward() RETURN
    // IS UNCHANGED — Python reads the merged output via merger_out().
    //
    // Weights are stored SWIZZLED-AS-IS, exactly like set_weights/set_patch_embed:
    // the linear launcher does NOT re-order the weight (rpu_linear.cpp reads
    // weight.data_ptr directly), so the Python adapter must pre-swizzle —
    //   fc1_w  [hidden*4, hidden*4] fp16, COL-swizzled (transform_linear_weight
    //          partition=1) — fc1 is column-parallel.
    //   fc2_w  [out_hidden, hidden*4] fp16, ROW-swizzled (partition=0) — fc2 is
    //          row-parallel (partial sums → all_reduce, see emit_merger).
    //   fc1_b  [hidden*4] fp16 (col-scatter per core at preload).
    //   fc2_b  [out_hidden] fp16 (core-0-only + row all_reduce → counted once).
    //   norm_w/norm_b [hidden] fp16 (ln_q gamma/beta, broadcast to all cores).
    // ⚠️ A wrong col/row choice at swizzle time is NOT caught here (both
    // constraints hold for these dims) — the only symptom is a wrong result.
    void set_merger(const at::Tensor& fc1_w, const at::Tensor& fc1_b,
                    const at::Tensor& fc2_w, const at::Tensor& fc2_b,
                    const at::Tensor& norm_w, const at::Tensor& norm_b,
                    int64_t out_hidden_size);

    // Reserved compatibility entry point; unsupported in the public runtime.
    void set_temporal(const at::Tensor& temporal_pe,
                      int64_t num_frames,
                      int64_t patches_per_frame);

    // Expose the keepalive bases for adapter-side copy_in.
    at::Tensor& position_idx_keepalive() { return position_idx_keepalive_; }
    // merger_out — emit_merger's output for the LAST forward: [1, N/4, out_hidden]
    // == HF's `merger(last_hidden_state)`. Plain DDR (merged_buf_), so it needs
    // no SPM probe and none of dbg_hidden()'s caveats. Narrowed
    // to the live merged length (N/4 tokens); merged_buf_ is padded to MAX/4.
    at::Tensor merger_out() {
        TORCH_CHECK(has_merger_,
                    "qwen3_5_vision: merger_out is only available when set_merger() "
                    "opted into the in-graph patch merger");
        TORCH_CHECK(!merger_wrote_fusion_target_,
                    "qwen3_5_vision: merger_out is unavailable after direct fusion; "
                    "the merger wrote into the supplied text hidden tensor");
        TORCH_CHECK(merged_buf_.defined() && current_num_patches_ > 0,
                    "qwen3_5_vision: merger_out is unset — run a forward first");
        // / 4 == / QWEN3_5_SPATIAL_MERGE_UNIT (spatial_merge_size²); num_patches
        // is a multiple of 4 by construction (grid is even in both axes).
        return merged_buf_.narrow(1, 0, current_output_patches_ / 4);
    }

    // ── Per-layer debug snapshots (only populated under get_debug_export()) ──────────
    //   dbg_hidden = [num_layers, N, hidden]                 — Phase-2 per-layer output
    //                                                          (residual1 after Phase 6, core 0).
    //   dbg_q      = [num_layers, attention_tp, N, local_q_dim] — Phase-1 rope'd Q (per core).
    // Filled by in-graph DMAs (NOT SPM+cpu_ptr) so they also carry real values under
    // capture/replay; read from Python via the get_dbg_* torch ops AFTER the forward.
    // Enable with torch.rpu.set_debug_export(True) before the forward.
    // SELF-CHECK IS AN INSTRUMENT CHECK, NOT A DIAGNOSIS. dbg_hidden[num_layers-1] and the
    //     forward's return are byte-identical DMAs — both core-0 SPM → DDR, channel 0, zero
    //     barriers (rpu_memcpy.cpp:1049 with num_cores=1 vs :1151) — emitted adjacently on the
    //     same stream off the same SPM. They CANNOT legitimately disagree, so a mismatch only
    //     ever means the probe is broken. It cannot show that the return path is wrong.
    at::Tensor dbg_hidden() {
        TORCH_CHECK(dbg_hidden_.defined(),
                    "qwen3_5_vision: dbg_hidden is unset — run a forward with "
                    "torch.rpu.set_debug_export(True) set before it.");
        return dbg_hidden_;
    }
    at::Tensor dbg_q() {
        TORCH_CHECK(dbg_q_.defined(),
                    "qwen3_5_vision: dbg_q is unset. It is OFF BY DEFAULT even under "
                    "set_debug_export(True) — set QWEN3_5_VISION_DBG_Q=1 too. It is opt-in "
                    "because it is an active-core scatter and injects cross-core fences per "
                    "layer (dbg_hidden is num_cores=1 → zero barriers; see dbg_q_enabled()).");
        return dbg_q_;
    }

    // forward — drive the configured encoder depth. Input is fp16 RPU
    // [1, num_patches, patch_dim] with STEP0; without STEP0 it is
    // [1, num_patches, hidden]. STEP0 adds
    // the supplied position tensor and the optional merger runs in the same
    // graph; output is [1, num_patches, hidden].
    at::Tensor forward(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches_in,
        const std::optional<at::Tensor>& step0_pos,
        const std::optional<at::Tensor>& fusion_target = std::nullopt,
        at::IntArrayRef fusion_row_starts = {},
        int64_t temporal_num_frames = 1,
        int64_t camera_batch_count = 1,
        int64_t expected_stage_plan_fingerprint_hi = 0,
        int64_t expected_stage_plan_fingerprint_lo = 0,
        int64_t expected_layout_hash_hi = 0,
        int64_t expected_layout_hash_lo = 0,
        at::IntArrayRef planned_stage_descriptor = {});

    std::tuple<std::vector<int64_t>, int64_t, int64_t>
    mint_vision_kvinsert_exact_candidate(
        at::IntArrayRef descriptor, int64_t site_id, int64_t invocation, int64_t route);

    // Bounded physical-SPM canary for the exact Qwen3.5-2B synthetic profile:
    // one 256-patch Vision chunk feeds the merger in-place, and the merger writes
    // its dense [64, 2048] result into a checked external row slice.  These entry
    // points are coordinator-only; standalone forward() retains its DDR path.
    SpmPipelineComponentLayout prepare_z2_layout(int64_t num_patches);
    SpmDense2DSpec z2_produced_spec(int64_t num_patches) const;
    void adopt_z2_layout(const SpmPipelineLease& lease,
                         const SpmTensorView& scratch);
    void bind_z2_slice(const SpmPipelineLease& lease,
                       const SpmPortView& slice);
    void validate_z2_layout(const SpmPipelineLease& lease) const;
    void clear_z2_layout(uint64_t epoch, uint64_t plan_hash);
    void forward_z2(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches_in,
        const std::optional<at::Tensor>& step0_pos,
        uint64_t epoch,
        uint64_t plan_hash);

protected:
    DecoderExecutionTopology resolve_model_execution_topology(
        int64_t q, int64_t kv, int64_t d, int64_t h,
        int64_t intermediate) const override;
    int64_t subclass_layout_hash() const override;

    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            current_camera_batch_count_, current_output_patches_, chunk_size_cap_);
    }

    // Graph-plan hooks + SPM layout + per-layer build (framework calls these).
    ModelStaticConfig       static_config() override;
    ModelDynamicConfig      dynamic_config(const ChunkPlan& plan) override;
    bool                    subclass_chunk_size_valid(int64_t cs, int64_t seq,
                                                      int64_t pos) const override;
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
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    void                    build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override;  // Phase 2

    // KV_FIRST helpers + weight preload (referenced by member-fn-pointer in static_config).
    void      emit_preload_weights();
    void      emit_kv_first_body(int layer_idx, const ChunkInfo& chunk);   // Phase 1
    ChunkPlan plan_kv_first_chunks(const ChunkPlan& compute_plan);


    void emit_step0();

    // emit_merger — post_layers_fn (EXT-Pi05 hook): patch merger, ONCE per
    // forward, AFTER the layer loop, inside the same graph scope. Reads the
    // tower's FULL [S, hidden] output back from output_tensor() DDR in chunks
    // (the tower is KV_FIRST-chunked, so at post_layers time the complete result
    // lives in DDR, not SPM) and applies LN → reshape [cs,hidden]→[cs/4,hidden*4]
    // → fc1(col)+GELU → fc2(row)+all_reduce → writes merged_buf_ DDR. Only
    // registered when set_merger() was called.
    void emit_merger();

private:
    struct VisionExecutionPlan {
        int64_t input_chunk_size = 0;
        int64_t qkv_chunk_size = 0;
        int64_t compute_chunk_size = 0;
        int64_t merger_chunk_size = 0;
        int64_t merger_chunks_per_span = 0;
        FmbThreeStageChunkPlan stages;
    };

    void configure_execution_geometry(
        int64_t execution_len,
        int64_t temporal_num_frames,
        int64_t camera_batch_count,
        bool compact_input);
    int64_t input_chunk_size_for_geometry(int64_t execution_len) const;
    VisionExecutionPlan resolve_execution_plan(int64_t execution_len);
    at::Tensor forward_impl(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_patches_in,
        const std::optional<at::Tensor>& step0_pos,
        const std::optional<at::Tensor>& fusion_target,
        at::IntArrayRef fusion_row_starts,
        int64_t temporal_num_frames,
        int64_t camera_batch_count,
        int64_t expected_stage_plan_fingerprint_hi,
        int64_t expected_stage_plan_fingerprint_lo,
        int64_t expected_layout_hash_hi,
        int64_t expected_layout_hash_lo,
        at::IntArrayRef planned_stage_descriptor,
        bool z2_dispatch);
    void consume_manifest_route(
        FmbRouteFamily family, int64_t site_id,
        int64_t selector, int64_t invocation = 0,
        std::vector<int64_t> resolved_arguments = {});

    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.insert(identity.end(), {
            static_cast<int64_t>(has_rope_),
            0,  // Reserved compatibility identity field.
            static_cast<int64_t>(has_step0_),
            static_cast<int64_t>(has_merger_),
            static_cast<int64_t>(linear_acc32_),
            static_cast<int64_t>(gelu_erf_ultra_)});
        if (moe_precision_) identity.insert(identity.end(), {3, 1});
        if (num_cores() != 8) {
            identity.insert(identity.end(),
                            {2, num_cores(), attn_tp(), mlp_tp(), 8});
        }
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.fc1_w, &weights.fc2_w}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        identity.push_back(static_cast<int64_t>(layer_bias_norm_.size()));
        for (const auto& weights : layer_bias_norm_) {
            for (const auto* tensor : {
                    &weights.ln1_w, &weights.ln1_b, &weights.ln2_w, &weights.ln2_b,
                    &weights.q_b, &weights.k_b, &weights.v_b, &weights.o_b,
                    &weights.fc1_b, &weights.fc2_b}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &pe_w_, &pe_b_, &merger_fc1_w_, &merger_fc1_b_,
                &merger_fc2_w_, &merger_fc2_b_, &merger_norm_w_, &merger_norm_b_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerBiasNorm> layer_bias_norm_;

    // 2D RoPE state
    at::Tensor freq_cos_;
    at::Tensor freq_sin_;
    at::Tensor position_idx_keepalive_;
    int64_t max_hw_ = 0;
    bool has_rope_ = false;
    int64_t current_num_patches_ = 0;
    int64_t current_output_patches_ = 0;
    int64_t current_camera_batch_count_ = 1;
    at::Tensor q_ddr_buf_;   // KV_FIRST Phase1→Phase2 的 rope 后 Q（DDR 暂存）

    // ── STEP 0 state (set_patch_embed / emit_step0) ─────────────────────────────
    at::Tensor pe_w_;                    // [hidden, patch_dim] fp16, col-swizzled
    at::Tensor pe_b_;                    // [hidden] fp16
    int64_t    patch_dim_     = 0;       // 1536 for Qwen3.5 (= pe_w_.size(1))
    bool       has_step0_     = false;

    // hidden_buf_ — STEP 0's output and run_all_layers' input. Allocated ONCE at
    // MAX_KEEPALIVE_SEQ and NEVER re-allocated, for the same reason q_ddr_buf_ is
    // (see the long note in forward()): emit_step0's output scatter is a FIXED
    // DMA whose address is baked into kd_buf at BUILD, and REPLAY's sync-only
    // fast path only patches registered MUTABLE DMAs. A grow-only buffer would
    // make a replayed small graph write to a freed address.
    at::Tensor hidden_buf_;              // [1, MAX_KEEPALIVE_SEQ, hidden]

    // pixel_src_base_ — the caller's per-forward pixel tensor, as a live RPU dev
    // addr. emit_step0 hands the mutable-DMA wrapper `&pixel_src_base_`, which
    // the graph registry keeps; REPLAY end() dereferences it and patches each
    // chunk DMA's src via Queue_t::update_dma_kernel. So this member's ADDRESS
    // must be stable while its VALUE is refreshed every forward — exactly the
    // framework's own hidden_in_src_base_ contract (fused_model_base_impl.h:56-60,
    // set at fused_model_base.cpp:971). Reading pixel through a fixed DMA instead
    // would bake forward #1's address and silently feed stale pixels thereafter.
    uint64_t pixel_src_base_ = 0;
    uint64_t step0_pos_src_base_ = 0;
    // Owning refs for the two bases above — the graph dereferences them at
    // EXECUTION, i.e. after forward() has returned (pitfalls.md C-1).
    at::Tensor pixel_src_ref_;
    at::Tensor step0_pos_ref_;

    // ── MERGER state (set_merger / emit_merger) ─────────────────────────────────
    // Patch merger run as a post_layers_fn. Weights stored SWIZZLED-AS-IS (Python
    // pre-swizzles, launcher does not re-order — see set_merger's header comment).
    at::Tensor merger_fc1_w_;   // [hidden*4, hidden*4] fp16, COL-swizzled (partition=1)
    at::Tensor merger_fc1_b_;   // [hidden*4] fp16
    at::Tensor merger_fc2_w_;   // [out_hidden, hidden*4] fp16, ROW-swizzled (partition=0)
    at::Tensor merger_fc2_b_;   // [out_hidden] fp16
    at::Tensor merger_norm_w_;  // [hidden] fp16 — ln_q gamma
    at::Tensor merger_norm_b_;  // [hidden] fp16 — ln_q beta
    int64_t    out_hidden_size_ = 0;   // merger output width (text hidden) = fc2's N
    int64_t    merger_hidden_   = 0;   // = hidden * spatial_merge_size² (=4096) = fc1 N/K, fc2 K
    bool       has_merger_      = false;

    // emit_merger always uses a mutable destination. Standalone vision writes the
    // stable merged_buf_; multimodal fusion points merger_dst_bases_ directly at
    // the current text hidden tensor's one or three image-token runs.
    at::Tensor merged_buf_;     // [1, MAX_KEEPALIVE_SEQ/4, out_hidden]
    at::Tensor merger_dst_ref_; // keeps a per-forward fusion target alive until sync completion
    std::array<uint64_t, 1> merger_dst_bases_{};
    bool merger_wrote_fusion_target_ = false;

    // Physical row-slice canary state.  The coordinator owns the lease; the
    // model stores only its checked identity and resolved unified SPM address.
    bool z2_bound_ = false;
    int64_t z2_prepared_num_patches_ = 0;
    uint32_t z2_slice_addr_ = 0;
    uint64_t z2_epoch_ = 0;
    uint64_t z2_plan_hash_ = 0;

    // Per-layer debug snapshots (get_debug_export() only). See dbg_hidden()/dbg_q().
    at::Tensor dbg_hidden_;  // [num_layers, N, hidden]
    at::Tensor dbg_q_;       // [num_layers, attention_tp, N, local_q_dim]

    // Model config
    double eps_ = 1e-6;
    int64_t orig_head_dim_ = 0;
    int attention_tp_ = 0;
    bool linear_acc32_ = false;
    bool gelu_erf_ultra_ = false;
    bool moe_precision_ = false;

    // MR-C / T34 — the vision chunk cap is PER HANDLE and installed cold.
    //
    // It used to be a function-local `static const` in
    // subclass_chunk_size_valid, so the first forward pinned the cap for every
    // later handle. The per-handle setter closes that cross-model leak.
    //
    // The cap must not change under an already-built same-shape GraphCache entry,
    // or that entry's replay op stream goes inconsistent with its plan. The
    // adapter translates the legacy environment variable once and calls
    // set_chunk_size_cap() before the first forward; native planning never reads
    // the environment. 0 = no cap.
    int64_t chunk_size_cap_ = 0;

public:
    // Planning-only descriptor for the exact geometry used by forward().
    std::vector<int64_t> resolve_prefill_plan(
        int64_t execution_len,
        int64_t temporal_num_frames,
        int64_t camera_batch_count,
        bool compact_input);
    std::vector<std::vector<int64_t>> resolve_prefill_domain(
        int64_t execution_len, int64_t temporal_num_frames,
        int64_t camera_batch_count, bool compact_input);
    // Cold per-handle cap. Must precede this handle's first forward, mirroring
    // Qwen3_5Model::set_chunk_size_cap. 0 = no cap.
    void set_chunk_size_cap(int64_t cap);
    void set_prefill_chunk_size(int64_t chunk_size);
    std::vector<int64_t> get_prefill_execution_controls() const {
        return {chunk_size_cap_, get_chunk_size_override()};
    }
    void stage_prefill_execution_controls(
        uint64_t token, int64_t cap, int64_t chunk_size);
};

}  // namespace v3
