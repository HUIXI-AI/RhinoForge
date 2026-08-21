// rpu_qwen3_5_vision_model.h — Qwen3.5 Vision Encoder (Option 2, deepstack removed).
//
// 24-block ViT encoder for Qwen3.5 (2B/4B variants, depth=24, hidden=1024,
// intermediate=4096, num_heads=16, head_dim=64). HF's Qwen3.5 vision model
// inherits Qwen3-VL and explicitly removes DeepStack; this dedicated subsystem
// therefore implements the shared encoder body plus one final patch merger, with
// no deepstack outputs or text-layer injections.
//
// The production graph contains STEP0 (folded patch embedding + interpolated
// position embedding), 24 encoder blocks, and the patch merger. The merger can
// write either a standalone pooler buffer or directly into the image-token rows
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

    Qwen3_5VisionModel() = default;

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

    // set_patch_embed — opt into in-graph STEP 0 (patch_embed + pos_embed).
    //
    // Calling this switches forward()'s input contract from pre-embedded hidden
    // [1, N, hidden] to raw folded patches [1, N, patch_dim] (patch_dim =
    // in_channels * temporal_patch_size * patch_size^2 = 1536 for Qwen3.5), and
    // makes the tower emit STEP 0 itself instead of the Python adapter doing it
    // on the CPU. If never called, the old CPU path is used unchanged.
    //
    //   pe_w      [hidden, patch_dim] fp16, COL-swizzled (transform_linear_weight
    //             partition=1) — patch_embed is column-parallel.
    //   pe_b      [hidden] fp16 — the launcher scatters hidden/tp per core, which
    //             is the COL-parallel bias convention. NOT the row-parallel
    //             fill+broadcast one.
    // Position embeddings are computed by the adapter with HF's implementation
    // and supplied to forward() as a mutable-DMA source.
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

    // G0.5 factorized temporal profile. The runtime remains spatial-only unless
    // forward() is called with this exact num_frames.
    void set_temporal(const at::Tensor& temporal_pe,
                      int64_t num_frames,
                      int64_t patches_per_frame);

    // Cold per-handle precision policy for the exact G0.5 K1 profile. This
    // keeps two-stage all-reduce enabled but bypasses its v2 scatter kernel.
    void set_allreduce_force_non_v2(bool enabled);
    bool get_allreduce_force_non_v2() const noexcept {
        return allreduce_force_non_v2_;
    }

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
    //   dbg_q      = [num_layers, NUM_CORES, N, local_q_dim] — Phase-1 rope'd Q (per core).
    // Filled by in-graph DMAs (NOT SPM+cpu_ptr) so they also carry real values under
    // capture/replay; read from Python via the get_dbg_* torch ops AFTER the forward.
    // Enable with torch.rpu.set_debug_export(True) before the forward.
    // SELF-CHECK IS AN INSTRUMENT CHECK, NOT A DIAGNOSIS. dbg_hidden[num_layers-1] and the
    //     forward's return are byte-identical DMAs — both core-0 SPM → DDR, channel 0, zero
    //     barriers — emitted adjacently on the same stream off the same SPM. They
    //     CANNOT legitimately disagree, so a mismatch only
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
                    "because it is an 8-core scatter and so injects a 7+7 barrier fence per "
                    "layer (dbg_hidden is num_cores=1 → zero barriers; see dbg_q_enabled()).");
        return dbg_q_;
    }

    // forward — drive the 24-block encoder. Input is fp16 RPU
    // [1, num_patches, patch_dim] with STEP0, or canonical G0.5 compact
    // [1,18,3,256,256]; without STEP0 it is [1,num_patches,hidden]. STEP0 adds
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
        int64_t camera_batch_count = 1);

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
    // Graph-plan hooks + SPM layout + per-layer build (framework calls these).
    ModelStaticConfig       static_config() override;
    ModelDynamicConfig      dynamic_config(const ChunkPlan& plan) override;
    bool                    subclass_chunk_size_valid(int64_t cs, int64_t seq,
                                                      int64_t pos) const override;
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override;
    void                    build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override;  // Phase 2

    // KV_FIRST helpers + weight preload (referenced by member-fn-pointer in static_config).
    void      emit_preload_weights();
    void      emit_kv_first_body(int layer_idx, const ChunkInfo& chunk);   // Phase 1
    ChunkPlan plan_kv_first_chunks(const ChunkPlan& compute_plan);
    void      emit_temporal_layer(int layer_idx);
    void      emit_spatial_group(int first_layer_idx, int last_layer_idx,
                                 int64_t offset,
                                 bool staged_qkv,
                                 int64_t staged_row_offset = 0);

    // emit_step0 — pre_layers_fn hook: pixel → hidden, once per
    // forward, before the layer loop, inside the same graph scope:
    //   broadcast pixel → col-parallel patch_embed → all_gather(axis=1)
    //   → add uploaded HF position embeddings → scatter to hidden_buf_.
    // Only registered when set_patch_embed() was called.
    //
    // Chunks over rows ITSELF: the framework's chunk planner is nested inside
    // the layer loop, and pre_layers_fn takes no ChunkInfo. Chunking is not
    // optional — pixel is broadcast full-width to every core (col-parallel does
    // not split K), so a whole 2640-row sequence would need 7.7MB of the 8MB
    // SPM before counting anything else. The row offset selects the matching
    // pixel and position slices for each mutable DMA.
    void emit_step0();

    // emit_merger — post_layers_fn hook: patch merger, once per
    // forward, AFTER the layer loop, inside the same graph scope. Reads the
    // tower's FULL [S, hidden] output back from output_tensor() DDR in chunks
    // (the tower is KV_FIRST-chunked, so at post_layers time the complete result
    // lives in DDR, not SPM) and applies LN → reshape [cs,hidden]→[cs/4,hidden*4]
    // → fc1(col)+GELU → fc2(row)+all_reduce → writes merged_buf_ DDR. Only
    // registered when set_merger() was called.
    void emit_merger();

private:
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
        bool z2_dispatch);

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

    // G0.5 K=6 factorized temporal attention. Only the temporal SDPA caches
    // live in DDR; Q/K/attention-output staging stays in SPM per camera.
    at::Tensor temporal_pe_;
    at::Tensor temporal_k_cache_;
    at::Tensor temporal_v_cache_;
    int64_t temporal_num_frames_ = 0;
    int64_t temporal_patches_per_frame_ = 0;
    int64_t current_temporal_num_frames_ = 1;
    bool current_compact_step0_ = false;
    bool has_temporal_ = false;

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
    // framework's own hidden_in_src_base_ contract. Reading pixel through a fixed DMA instead
    // would bake forward #1's address and silently feed stale pixels thereafter.
    uint64_t pixel_src_base_ = 0;
    uint64_t step0_pos_src_base_ = 0;
    // Owning refs for the two bases above — the graph dereferences them at
    // execution, i.e. after forward() has returned.
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
    std::array<uint64_t, 3> merger_dst_bases_{};
    bool merger_wrote_fusion_target_ = false;

    // Physical row-slice canary state.  The coordinator owns the lease; the
    // model stores only its checked identity and resolved unified SPM address.
    bool z2_bound_ = false;
    int64_t z2_prepared_num_patches_ = 0;
    uint32_t z2_slice_addr_ = 0;
    uint64_t z2_epoch_ = 0;
    uint64_t z2_plan_hash_ = 0;
    bool allreduce_force_non_v2_ = false;

    // Per-layer debug snapshots (get_debug_export() only). See dbg_hidden()/dbg_q().
    at::Tensor dbg_hidden_;  // [num_layers, N, hidden]
    at::Tensor dbg_q_;       // [num_layers, NUM_CORES, N, local_q_dim]

    // Model config
    double eps_ = 1e-6;
    int64_t orig_head_dim_ = 0;

    // The vision chunk cap is per handle and latched on first use.
    //
    // The cap must not change under an already-built same-shape GraphCache entry,
    // or that entry's replay op stream becomes inconsistent with its plan. It is
    // therefore latched once per handle; -1 means not yet latched.
    // set_chunk_size_cap() is the explicit override; absent that, the first call
    // reads the env.
    mutable int64_t chunk_size_cap_ = -1;

public:
    // Cold per-handle cap. Must precede this handle's first forward, mirroring
    // Qwen3_5Model::set_chunk_size_cap. 0 = no cap.
    void set_chunk_size_cap(int64_t cap);
};

}  // namespace v3
