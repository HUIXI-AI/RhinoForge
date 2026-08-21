// rpu_qwen3_5_vision_model.cpp — Qwen3.5 Vision Encoder (Option 2, deepstack
// removed). See rpu_qwen3_5_vision_model.h for architecture/scope. A 24-block
// bidirectional ViT run as a KV_FIRST two-phase pass: Phase 1 inserts all K/V +
// stashes rope'd Q; Phase 2 loads Q back + SDPA over full KV + MLP.
//
// ── FILE MAP (read order: main flow → layer build → buffers/config → setup → C-API) ──
//   MAIN FLOW
//     forward()               — Python entry's workhorse; allocs q_ddr_buf_, drives run_all_layers
//   LAYER BUILD (KV_FIRST two-phase; framework drives Phase 1 then Phase 2 per chunk)
//     emit_kv_first_body()    — Phase 1: LN1 → Q/K/V Linear+bias → 2D-rope → insert all K/V at
//                               absolute pos → stash rope'd Q to q_ddr_buf_ (DDR)
//     build_layer_subgraph()  — Phase 2: load Q back → SDPA (full KV, bidir) → o_proj+AllReduce+
//                               residual → LN2 → fc1+GELU → fc2+AllReduce+residual
//   SPM LAYOUT       declare_buffers() — ALL ∪ KVIN(Phase 1) ∪ COMP(Phase 2), two-phase aliasing
//   GRAPH-PLAN HOOKS static_config / dynamic_config / plan_kv_first_chunks / subclass_chunk_size_valid
//   SETUP            emit_preload_weights (per-layer norm/bias DMA), set_weights, set_rope_tables
//   C-API (bottom)   rpu_qwen3_5_vision_* free funcs — Python handle ↔ object bridges
#include "rpu_qwen3_5_vision_model.h"

#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_spm_allocator.h"
#include "rpu_runtime_state.h"   // get_debug_export — per-layer debug snapshots
#include "rpu_qwen3_5_spm_z2.h"
#include "graph/graph_runtime.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdlib>   // std::getenv / std::atoll — optional QWEN3_5_VISION_CHUNK cap
#include <cmath>
#include <algorithm>  // std::min — emit_step0 row chunking

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

// Generic images keep their original 4096 limit; 4608 exists only for the exact
// G0.5 camera-major B_camera=3 × K=6 × P=256 temporal profile.
constexpr int64_t QWEN3_5_VISION_GENERIC_MAX_SEQ = 4096;
constexpr int64_t QWEN3_5_VISION_MAX_KEEPALIVE_SEQ = 4608;
constexpr int64_t G05_MAX_CAMERA_BATCH = 3;
constexpr int64_t G05_COMPACT_STEP0_FRAMES = 3;
constexpr int64_t G05_COMPACT_CHANNELS = 3;
constexpr int64_t G05_COMPACT_IMAGE_SIDE = 256;
constexpr int64_t G05_COMPACT_PATCH_SIDE = 16;
constexpr int64_t G05_COMPACT_GRID_SIDE =
    G05_COMPACT_IMAGE_SIDE / G05_COMPACT_PATCH_SIDE;
constexpr int64_t G05_COMPACT_PATCHES_PER_FRAME =
    G05_COMPACT_GRID_SIDE * G05_COMPACT_GRID_SIDE;
constexpr int64_t G05_COMPACT_STEP0_ROWS =
    G05_COMPACT_STEP0_FRAMES * G05_COMPACT_PATCHES_PER_FRAME;

// STEP 0 row-chunk. Chunking is NOT optional: patch_embed is col-parallel, so K
// is not split and every core needs the FULL pixel row — a 2640-row sequence
// would want 2640*1536*2 = 7.7MB of the 8MB SPM for s0_pixel alone. Each chunk
// passes conv3d_chunk_id*conv3d_chunk_size as pos_offset.
//
// This is an INDEPENDENT policy knob, not a function of the layer chunk_size.
// The STEP 0 buffers use BufferScope::OutsideLayerLoop: they are dead before
// layer 0 starts and hand off through DDR, so the SPM total is
// max(layer_peak, step0_peak) — STEP 0 does not spend the layer loop's budget.
// The only constraint is that step0_peak fits SPM on its own:
//
//   step0_peak = cs0 * (max(patch_dim, hidden) + hidden/tp + hidden) * 2B
//              = 896 * (1536 + 128 + 1024) * 2 = 4.82MB   (Qwen3.5 dims)
//
// against ~8MB minus the PersistentPerLayer weights (~350KB).
//
// OutsideLayerLoop keeps this allocation independent of the layer chunk.
constexpr int64_t QWEN3_5_VISION_STEP0_CHUNK = 896;

// Spatial merger — merges each 2×2 spatial block of patches into one token
// (spatial_merge_size=2 ⇒ spatial_merge_size²=4). The merger reshapes the tower
// output [S, hidden] → [S/4, hidden*4] then MLP-projects to the text hidden
// size. See emit_merger.
constexpr int64_t QWEN3_5_SPATIAL_MERGE_UNIT = 4;
constexpr int64_t G05_TEMPORAL_PATCH_GROUP = 16;
constexpr int64_t G05_SPATIAL_GROUP_ROWS = 768;  // three 256-patch frames

// Merger row chunk, over PRE-merge (tower-output) rows. MUST be %4==0: each
// 4-token merge group must stay inside one chunk, or the reshape would fold
// rows from two different chunks into one merged token → garbage. Independent
// policy knob (like QWEN3_5_VISION_STEP0_CHUNK); its buffers use
// BufferScope::OutsideLayerLoop and alias the layer/STEP0 sets, so it does not
// spend the layer budget.
constexpr int64_t QWEN3_5_VISION_MERGER_CHUNK = 512;   // %4==0

// First physical row-slice canary only.  Do not silently widen this envelope:
// the tower-final -> merger identity below relies on one full compute chunk.
constexpr int64_t QWEN3_5_Z2_NUM_PATCHES = 256;
constexpr int64_t QWEN3_5_Z2_VISION_HIDDEN = 1024;
constexpr int64_t QWEN3_5_Z2_TEXT_HIDDEN = 2048;

static at::Tensor empty_256b_aligned_fp16_rpu(
    const std::vector<int64_t>& shape,
    const at::TensorOptions& options,
    const char* name)
{
    int64_t numel = 1;
    for (int64_t dim : shape) numel *= dim;
    constexpr int64_t alignment_elems = 256 / sizeof(c10::Half);
    auto storage = at::empty({numel + alignment_elems}, options);
    const uint64_t raw = RpuGetDevAddr(storage.data_ptr<c10::Half>());
    const int64_t off_bytes = (256 - static_cast<int64_t>(raw & 0xFFu)) & 0xFF;
    TORCH_CHECK(off_bytes % static_cast<int64_t>(sizeof(c10::Half)) == 0,
                name, ": cannot construct a 256B-aligned view");
    auto aligned = storage
        .narrow(0, off_bytes / static_cast<int64_t>(sizeof(c10::Half)), numel)
        .view(shape);
    TORCH_CHECK((RpuGetDevAddr(aligned.data_ptr<c10::Half>()) & 0xFFu) == 0,
                name, ": aligned view landed at a non-256B address");
    return aligned;
}

// QWEN3_5_VISION_DBG_Q=1 enables the Phase-1 RoPE-Q probe. OFF by default because
// its 8-core scatter emits a 7+7 barrier fence per layer, i.e. ~336 extra barriers
// across the tower. The dbg_hidden probe next to it passes num_cores=1, so those loops never run
// and it emits ZERO barriers — it is a plain channel-0 (= stream 0 = the compute stream) DMA that
// adds no cross-stream ordering at all.

static bool dbg_q_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("QWEN3_5_VISION_DBG_Q");
        cached = (e && (std::string(e) == "1" || std::string(e) == "true")) ? 1 : 0;
    }
    return cached != 0;
}

namespace v3 {

// ─────────────────────────────────────────────────────────────────────────────
// MAIN FLOW
// ─────────────────────────────────────────────────────────────────────────────

// ── forward: drive the 24-block encoder — flush position_idx, alloc q_ddr_buf_
//    for the KV_FIRST two phases, then all-layers-once. ─────────────────────────
at::Tensor Qwen3_5VisionModel::forward(
    const at::Tensor& input,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    int64_t num_patches_in,
    const std::optional<at::Tensor>& step0_pos,
    const std::optional<at::Tensor>& fusion_target,
    at::IntArrayRef fusion_row_starts,
    int64_t temporal_num_frames,
    int64_t camera_batch_count)
{
    TORCH_CHECK(!z2_bound_,
                "Qwen3_5VisionModel::forward: physical Z2 lease is active; "
                "use the coordinator-owned Z2 dispatch");
    return forward_impl(input, k_caches, v_caches, num_patches_in,
                        step0_pos, fusion_target, fusion_row_starts,
                        temporal_num_frames, camera_batch_count,
                        /*z2_dispatch=*/false);
}

at::Tensor Qwen3_5VisionModel::forward_impl(
    const at::Tensor& input,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    int64_t num_patches_in,
    const std::optional<at::Tensor>& step0_pos,
    const std::optional<at::Tensor>& fusion_target,
    at::IntArrayRef fusion_row_starts,
    int64_t temporal_num_frames,
    int64_t camera_batch_count,
    bool z2_dispatch)
{
    TORCH_CHECK(z2_dispatch == z2_bound_,
                "Qwen3_5VisionModel::forward_impl: Z2 dispatch/binding mismatch");
    if (z2_dispatch) {
        TORCH_CHECK(!get_debug_export(),
                    "Qwen3_5VisionModel Z2: debug export is incompatible with "
                    "the in-place tower-to-merger handoff");
        TORCH_CHECK(num_patches_in == z2_prepared_num_patches_,
                    "Qwen3_5VisionModel Z2: prepared num_patches=",
                    z2_prepared_num_patches_, " but dispatch got ",
                    num_patches_in);
    }
    TORCH_CHECK(num_layers() > 0,
                "Qwen3_5VisionModel::forward called before set_weights");
    TORCH_CHECK(has_rope_,
                "Qwen3_5VisionModel::forward called before set_rope_tables");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "Qwen3_5VisionModel::forward: input must be on RPU device");
    TORCH_CHECK(input.scalar_type() == at::kHalf,
                "Qwen3_5VisionModel::forward: input must be FP16");
    TORCH_CHECK(input.is_contiguous(),
                "Qwen3_5VisionModel::forward: input must be contiguous");
    const bool compact_input = input.dim() == 5;
    current_compact_step0_ = compact_input;
    TORCH_CHECK(input.dim() == 3 || compact_input,
                "Qwen3_5VisionModel::forward: input must be 3D [1,N,H] or the "
                "canonical compact 5D [1,18,3,256,256], got ", input.dim(), "D");
    TORCH_CHECK(num_patches_in > 0 && num_patches_in <= QWEN3_5_VISION_MAX_KEEPALIVE_SEQ,
                "Qwen3_5VisionModel::forward: num_patches=", num_patches_in,
                " must be in (0, MAX_KEEPALIVE_SEQ=", QWEN3_5_VISION_MAX_KEEPALIVE_SEQ, "]");
    TORCH_CHECK(input.size(0) == 1,
                "Qwen3_5VisionModel::forward: batch must be 1, got ", input.size(0));
    if (!compact_input) {
        TORCH_CHECK(input.size(1) == num_patches_in,
                    "Qwen3_5VisionModel::forward: input.size(1)=", input.size(1),
                    " must match num_patches=", num_patches_in);
    }
    TORCH_CHECK(temporal_num_frames >= 1,
                "Qwen3_5VisionModel::forward: temporal_num_frames must be positive");
    TORCH_CHECK(camera_batch_count == 1 || camera_batch_count == G05_MAX_CAMERA_BATCH,
                "Qwen3_5VisionModel::forward: camera_batch_count must be 1 or 3");
    current_temporal_num_frames_ = temporal_num_frames;
    current_camera_batch_count_ = camera_batch_count;
    if (temporal_num_frames > 1) {
        TORCH_CHECK(has_temporal_ && temporal_num_frames == temporal_num_frames_,
                    "Qwen3_5VisionModel::forward: unsupported temporal frame count ",
                    temporal_num_frames);
        if (camera_batch_count > 1) {
            TORCH_CHECK(temporal_num_frames_ == 6 &&
                        temporal_patches_per_frame_ == 256 &&
                        camera_batch_count == G05_MAX_CAMERA_BATCH,
                        "Qwen3_5VisionModel::forward: packed temporal vision supports "
                        "only B_camera=3,K=6,P=256");
        }
        const int64_t expected_temporal_patches =
            camera_batch_count * temporal_num_frames_ * temporal_patches_per_frame_;
        TORCH_CHECK(num_patches_in == expected_temporal_patches,
                    "Qwen3_5VisionModel::forward: temporal input must contain ",
                    expected_temporal_patches,
                    " patches, got ", num_patches_in);
        current_output_patches_ =
            camera_batch_count * temporal_patches_per_frame_;
    } else {
        TORCH_CHECK(num_patches_in <= QWEN3_5_VISION_GENERIC_MAX_SEQ,
                    "Qwen3_5VisionModel::forward: generic/single-frame vision "
                    "supports at most ", QWEN3_5_VISION_GENERIC_MAX_SEQ,
                    " patches, got ", num_patches_in);
        TORCH_CHECK(camera_batch_count == 1,
                    "Qwen3_5VisionModel::forward: camera batching requires K=6");
        current_output_patches_ = num_patches_in;
    }
    if (has_merger_) {
        TORCH_CHECK(num_patches_in % QWEN3_5_SPATIAL_MERGE_UNIT == 0,
                    "Qwen3_5VisionModel::forward: input patch count must be "
                    "divisible by ", QWEN3_5_SPATIAL_MERGE_UNIT, ", got ",
                    num_patches_in);
        TORCH_CHECK(current_output_patches_ % QWEN3_5_SPATIAL_MERGE_UNIT == 0,
                    "Qwen3_5VisionModel::forward: merger requires num_patches "
                    "divisible by ", QWEN3_5_SPATIAL_MERGE_UNIT, ", got ",
                    current_output_patches_);
    }
    if (compact_input) {
        TORCH_CHECK(
            has_step0_ && temporal_num_frames == 6 &&
                camera_batch_count == G05_MAX_CAMERA_BATCH &&
                patch_dim_ == G05_COMPACT_CHANNELS * 2 *
                                  G05_COMPACT_PATCH_SIDE * G05_COMPACT_PATCH_SIDE &&
                input.size(1) == G05_MAX_CAMERA_BATCH * temporal_num_frames &&
                input.size(2) == G05_COMPACT_CHANNELS &&
                input.size(3) == G05_COMPACT_IMAGE_SIDE &&
                input.size(4) == G05_COMPACT_IMAGE_SIDE,
            "Qwen3_5VisionModel::forward: compact STEP0 requires the canonical "
            "B_camera=3,K=6,C=3,H=W=256,tp=2,patch=16 profile, got ",
            input.sizes());
    } else {
        // set_patch_embed() flips the input contract: raw folded patches instead
        // of pre-embedded hidden.
        const int64_t expect_w = has_step0_ ? patch_dim_ : hidden_size();
        TORCH_CHECK(input.size(2) == expect_w,
                    "Qwen3_5VisionModel::forward: input.size(2)=", input.size(2),
                    " must be ", expect_w,
                    has_step0_ ? " (patch_dim — set_patch_embed() is active, so the "
                                 "tower expects RAW folded patches, not hidden)"
                               : " (hidden — set_patch_embed() was not called, so the "
                                 "adapter must patch-embed on the CPU as before)");
    }

    // Cache the num_patches for build_layer_subgraph to consume (Q/K rope
    // launcher needs the actual token count, which is also the chunk len).
    current_num_patches_ = num_patches_in;

    // Flush the position_idx keepalive — adapter just wrote per-forward
    // (row, col) entries into it via Python `.copy_()`. Without an explicit
    // flush, the CPU write may sit in the caching allocator's write buffer
    // and the RPU DMA reads stale (zero or prior-forward) data, silently
    // collapsing 2D RoPE to identity. Same pattern as Qwen3 M-RoPE in
    // CausalDecoderModel::forward (Phase 1 keepalive flush).
    rpu_ddr_flush_force(position_idx_keepalive_.data_ptr<int16_t>());

    // KV_FIRST 双向分块：不单 chunk override，规划器按 SPM 预算挑 chunk_size
    // （显式 QWEN3_5_VISION_CHUNK 可设上界）。q_ddr_buf_ 存每 core 的 rope 后 Q
    // 跨两相（Phase1 emit_kv_first_body 写 / Phase2 build_layer_subgraph 读）。
    //
    // 按 MAX_KEEPALIVE_SEQ 铺满、一次分配后**永不重分配**。
    // 不能使用 grow-only 缓冲：存/读 Q 走
    // 的是 FIXED DMA，其地址在 BUILD 时写进 kd_buf，而 REPLAY 的 sync-only 快路径
    // 只对登记过的 mutable DMA 调 update_dma_kernel，
    // fixed 的包没人碰。于是「小图 → 大图（换 buf，旧块 free 回 allocator）→ 再来小图（sig
    // 命中旧图 → REPLAY）」会照 BUILD 时的旧地址去写，而那块 DDR 已被分给别的张量 → 静默
    // 糊掉它。（Q 自身反而是对的：存和读烤的是同一个旧地址+旧 stride，自洽。）
    // Replay's sync-only fast path does not rebuild fixed DMA nodes, so the
    // backing allocation must remain stable for the graph lifetime.
    // 铺满可行的前提：num_patches 有硬上限（本函数开头的 TORCH_CHECK）。
    if (temporal_num_frames == 1 && !q_ddr_buf_.defined()) {
        const int64_t local_q_dim = (num_q_heads() / NUM_CORES) * head_dim();
        q_ddr_buf_ = at::empty(
            {(int64_t)NUM_CORES, QWEN3_5_VISION_MAX_KEEPALIVE_SEQ, local_q_dim},
            input.options());
    }
    if (temporal_num_frames > 1 && !temporal_k_cache_.defined()) {
        // The temporal SDPA kernel's validated head envelope is one camera's
        // 16-patch group. B3 reuses this cache sequentially per camera.
        const int64_t composite_heads =
            G05_TEMPORAL_PATCH_GROUP * num_q_heads();
        const int64_t composite_heads_per_core = composite_heads / NUM_CORES;
        const std::vector<int64_t> cache_shape = {
            1, 1, composite_heads_per_core, head_dim() / 16,
            (int64_t)NUM_CORES, 16, 16};
        temporal_k_cache_ = empty_256b_aligned_fp16_rpu(
            cache_shape, input.options(), "G0.5 temporal K cache");
        temporal_v_cache_ = empty_256b_aligned_fp16_rpu(
            cache_shape, input.options(), "G0.5 temporal V cache");
    }

    // Per-layer debug snapshots (get_debug_export() only), read after forward.
    // Fixed DMAs bind their destinations at BUILD, so snapshots are allocated once
    // per shape; shape changes force a rebuild.
    if (get_debug_export()) {
        const int64_t L = num_layers(), N = num_patches_in, H = hidden_size();
        if (!dbg_hidden_.defined() || dbg_hidden_.size(0) != L ||
            dbg_hidden_.size(1) != N || dbg_hidden_.size(2) != H) {
            dbg_hidden_ = at::empty({L, N, H}, input.options());
        }
        if (temporal_num_frames == 1 && dbg_q_enabled()) {
            const int64_t lq = (num_q_heads() / NUM_CORES) * head_dim();
            if (!dbg_q_.defined() || dbg_q_.size(0) != L ||
                dbg_q_.size(1) != (int64_t)NUM_CORES || dbg_q_.size(2) != N ||
                dbg_q_.size(3) != lq) {
                dbg_q_ = at::empty({L, (int64_t)NUM_CORES, N, lq}, input.options());
            }
        }
    }

    // STEP 0: the tower consumes hidden_buf_ (emit_step0's output), not the raw
    // pixel. hidden_buf_ is allocated ONCE at MAX and never re-allocated, for the
    // same reason as q_ddr_buf_ above: emit_step0's output scatter is a FIXED DMA
    // whose address is baked into kd_buf at BUILD, and REPLAY's sync-only fast
    // path only patches registered MUTABLE DMAs.
    at::Tensor step0_out;   // keeps the narrowed view alive across run_all_layers
    const at::Tensor* layer_input = &input;
    if (has_step0_) {
        TORCH_CHECK(step0_pos.has_value(),
                    "Qwen3_5VisionModel::forward: RPU STEP0 requires position embeddings");
        const at::Tensor& pos = *step0_pos;
        const bool temporal_pos = temporal_num_frames > 1;
        const bool shape_ok = temporal_pos
            ? (pos.dim() == 3 && pos.size(0) == NUM_CORES &&
               pos.size(1) == num_patches_in &&
               pos.size(2) == hidden_size() / NUM_CORES)
            : (pos.dim() == 2 && pos.size(0) == num_patches_in &&
               pos.size(1) == hidden_size());
        TORCH_CHECK(pos.device().type() == at::kPrivateUse1 &&
                    pos.scalar_type() == at::kHalf && pos.is_contiguous() && shape_ok,
                    "Qwen3_5VisionModel::forward: step0_pos must be contiguous fp16 RPU ",
                    temporal_pos ? "[8,N,H/8]" : "[N,H]", ", got ", pos.sizes());
        // Refresh the live pixel address for emit_step0's mutable DMAs. The
        // member's ADDRESS is what the graph registry holds; only its value moves.
        // Own both tensors alongside their bases: the DMAs are dereferenced when
        // the graph EXECUTES (RpuKernelGraph::end(), i.e. when the caller's
        // capture scope exits), not when forward() returns. The merger
        // destination below follows the same lifetime rule.
        pixel_src_ref_ = input;
        step0_pos_ref_ = pos;
        pixel_src_base_ = ::rhino_lkn::RpuGetDevAddr(input.data_ptr());
        step0_pos_src_base_ = ::rhino_lkn::RpuGetDevAddr(pos.data_ptr());
        rpu_ddr_flush_force(const_cast<c10::Half*>(input.data_ptr<c10::Half>()));
        rpu_ddr_flush_force(const_cast<c10::Half*>(pos.data_ptr<c10::Half>()));
        if (!hidden_buf_.defined()) {
            hidden_buf_ = at::empty(
                {1, QWEN3_5_VISION_MAX_KEEPALIVE_SEQ, hidden_size()}, input.options());
        }
        // Narrow to the live length — the framework reads seq_len off size(1), and
        // hidden_buf_ is padded to MAX. Offset 0, and dim 0 is size 1, so the view
        // stays contiguous and its data_ptr is hidden_buf_'s base (what emit_step0
        // writes through).
        step0_out = hidden_buf_.narrow(1, 0, num_patches_in);
        layer_input = &step0_out;
    }

    if (z2_dispatch) {
        TORCH_CHECK(!fusion_target.has_value() && fusion_row_starts.empty(),
                    "Qwen3_5VisionModel Z2: DDR fusion target is forbidden");
        TORCH_CHECK(has_merger_ && z2_slice_addr_ != 0,
                    "Qwen3_5VisionModel Z2: typed merger slice is not bound");
        merger_dst_ref_ = at::Tensor();
        merger_dst_bases_.fill(0);
        // Keep merger_out() fail-closed after the lease is released: this
        // invocation wrote neither merged_buf_ nor the ordinary DDR target.
        merger_wrote_fusion_target_ = true;
    } else if (fusion_target.has_value()) {
        const at::Tensor& target = *fusion_target;
        const int64_t merged_rows_per_camera =
            (current_output_patches_ / current_camera_batch_count_) /
            QWEN3_5_SPATIAL_MERGE_UNIT;
        TORCH_CHECK(has_merger_,
                    "Qwen3_5VisionModel::forward: fusion_target requires the RPU merger");
        TORCH_CHECK(target.defined() && target.device().type() == at::kPrivateUse1 &&
                    target.scalar_type() == at::kHalf && target.is_contiguous(),
                    "Qwen3_5VisionModel::forward: fusion_target must be contiguous fp16 RPU");
        TORCH_CHECK(target.dim() == 3 && target.size(0) == 1 &&
                    target.size(2) == out_hidden_size_,
                    "Qwen3_5VisionModel::forward: fusion_target must be [1,S,",
                    out_hidden_size_, "], got ", target.sizes());
        TORCH_CHECK(fusion_row_starts.size() == current_camera_batch_count_,
                    "Qwen3_5VisionModel::forward: expected ",
                    current_camera_batch_count_, " fusion row starts, got ",
                    fusion_row_starts.size());
        merger_dst_ref_ = target;
        const uint64_t target_base =
            ::rhino_lkn::RpuGetDevAddr(target.data_ptr<c10::Half>());
        merger_dst_bases_.fill(0);
        for (int64_t camera = 0; camera < current_camera_batch_count_; ++camera) {
            const int64_t row_start = fusion_row_starts[camera];
            TORCH_CHECK(row_start >= 0 &&
                        row_start + merged_rows_per_camera <= target.size(1),
                        "Qwen3_5VisionModel::forward: fusion rows [", row_start, ", ",
                        row_start + merged_rows_per_camera,
                        ") exceed target sequence ", target.size(1));
            merger_dst_bases_[camera] = target_base +
                row_start * out_hidden_size_ * sizeof(c10::Half);
        }
        merger_wrote_fusion_target_ = true;
        // Embedding may be a host fallback, so flush the untouched text rows too before the
        // decoder consumes this tensor directly from DDR.
        rpu_ddr_flush_force_sized(target.data_ptr<c10::Half>(), target.nbytes());
    } else if (has_merger_) {
        TORCH_CHECK(fusion_row_starts.empty(),
                    "Qwen3_5VisionModel::forward: fusion_row_starts require fusion_target");
        if (!merged_buf_.defined()) {
            merged_buf_ = at::empty(
                {1, QWEN3_5_VISION_MAX_KEEPALIVE_SEQ / QWEN3_5_SPATIAL_MERGE_UNIT,
                 out_hidden_size_},
                input.options());
        }
        merger_dst_ref_ = at::Tensor();
        merger_dst_bases_.fill(0);
        merger_dst_bases_[0] =
            ::rhino_lkn::RpuGetDevAddr(merged_buf_.data_ptr<c10::Half>());
        merger_wrote_fusion_target_ = false;
        rpu_ddr_flush_force_sized(
            merged_buf_.data_ptr<c10::Half>(),
            (current_output_patches_ / QWEN3_5_SPATIAL_MERGE_UNIT) * out_hidden_size_ *
                sizeof(c10::Half));
    }

    at::Tensor result = run_all_layers(
        *layer_input, k_caches, v_caches,
        /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false);

    return result;
}

SpmPipelineComponentLayout Qwen3_5VisionModel::prepare_z2_layout(
    int64_t num_patches) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "Qwen3_5VisionModel Z2 prepare must run outside Graph capture");
    TORCH_CHECK(!z2_bound_,
                "Qwen3_5VisionModel Z2: cannot prepare while a lease is active");
    TORCH_CHECK(num_patches == QWEN3_5_Z2_NUM_PATCHES,
                "Qwen3_5VisionModel Z2 canary admits exactly ",
                QWEN3_5_Z2_NUM_PATCHES, " patches, got ", num_patches);
    TORCH_CHECK(num_layers() == 24 && num_q_heads() == 16 &&
                    head_dim() == 64 &&
                    hidden_size() == QWEN3_5_Z2_VISION_HIDDEN &&
                    intermediate_size() == 4096,
                "Qwen3_5VisionModel Z2 requires the exact shared 2B/4B "
                "Vision tower profile");
    TORCH_CHECK(has_step0_ && has_merger_ &&
                    out_hidden_size_ == QWEN3_5_Z2_TEXT_HIDDEN,
                "Qwen3_5VisionModel Z2 requires in-graph STEP0 and the "
                "[1024 -> 2048] RPU merger");
    TORCH_CHECK(!get_debug_export(),
                "Qwen3_5VisionModel Z2 prepare rejects debug export");

    const int64_t resolved = resolve_chunk_size_for_shape(
        num_patches, /*position=*/0, /*attention_mask=*/std::nullopt,
        /*is_causal=*/false);
    TORCH_CHECK(resolved == num_patches,
                "Qwen3_5VisionModel Z2 requires one full Vision chunk; "
                "planner resolved ", resolved, " for ", num_patches,
                " patches");

    LayoutContext layout;
    layout.chunk_size = resolved;
    layout.max_kv_seq_len = num_patches;
    layout.num_layers = num_layers();
    layout.use_attn_mask = false;
    layout.is_causal = false;
    auto prepared = prepare_spm_pipeline_component(layout);
    z2_prepared_num_patches_ = num_patches;
    return prepared;
}

SpmDense2DSpec Qwen3_5VisionModel::z2_produced_spec(
    int64_t num_patches) const {
    TORCH_CHECK(num_patches == QWEN3_5_Z2_NUM_PATCHES,
                "Qwen3_5VisionModel Z2 produced spec admits exactly ",
                QWEN3_5_Z2_NUM_PATCHES, " patches, got ", num_patches);
    TORCH_CHECK(has_merger_ && hidden_size() == QWEN3_5_Z2_VISION_HIDDEN &&
                    out_hidden_size_ == QWEN3_5_Z2_TEXT_HIDDEN,
                "Qwen3_5VisionModel Z2 produced spec requires the exact merger profile");
    SpmDense2DSpec spec;
    spec.rows = num_patches / QWEN3_5_SPATIAL_MERGE_UNIT;
    spec.cols = out_hidden_size_;
    spec.validate();
    return spec;
}

void Qwen3_5VisionModel::adopt_z2_layout(
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    TORCH_CHECK(z2_prepared_num_patches_ == QWEN3_5_Z2_NUM_PATCHES,
                "Qwen3_5VisionModel Z2: prepare must precede begin");
    TORCH_CHECK(!z2_bound_,
                "Qwen3_5VisionModel Z2: binding is already active");
    adopt_spm_pipeline_component(lease, scratch);
}

void Qwen3_5VisionModel::bind_z2_slice(
    const SpmPipelineLease& lease,
    const SpmPortView& slice) {
    const auto expected = z2_produced_spec(z2_prepared_num_patches_);
    TORCH_CHECK(slice.spec() == expected &&
                    slice.size_bytes() == expected.storage_bytes(),
                "Qwen3_5VisionModel Z2: typed row-slice spec mismatch");

    // For the exact cs=256 layout, the last replicated tower residual and
    // merger m_in are the same physical bytes.  This is a checked canary fact,
    // not a general FusedModelBase layout promise.
    TORCH_CHECK(addr(0, "residual1") == addr(0, "m_in"),
                "Qwen3_5VisionModel Z2: tower residual1 and merger m_in "
                "are not a physical identity");

    z2_slice_addr_ = slice.resolve_physical_addr(/*core=*/0, lease);
    z2_epoch_ = lease.epoch();
    z2_plan_hash_ = lease.plan_hash();
    z2_bound_ = true;
}

void Qwen3_5VisionModel::validate_z2_layout(
    const SpmPipelineLease& lease) const {
    TORCH_CHECK(z2_bound_,
                "Qwen3_5VisionModel Z2: no active binding");
    TORCH_CHECK(z2_epoch_ == lease.epoch() &&
                    z2_plan_hash_ == lease.plan_hash(),
                "Qwen3_5VisionModel Z2: stale epoch/plan hash");
    TORCH_CHECK(addr(0, "residual1") == addr(0, "m_in"),
                "Qwen3_5VisionModel Z2: tower-to-merger identity drifted");
    validate_spm_pipeline_component(lease);
}

void Qwen3_5VisionModel::clear_z2_layout(
    uint64_t epoch,
    uint64_t plan_hash) {
    if (z2_bound_) {
        TORCH_CHECK(z2_epoch_ == epoch && z2_plan_hash_ == plan_hash,
                    "Qwen3_5VisionModel Z2 clear: stale epoch/plan hash");
    }
    release_spm_pipeline_component(epoch, plan_hash);
    z2_bound_ = false;
    z2_slice_addr_ = 0;
    z2_epoch_ = 0;
    z2_plan_hash_ = 0;
}

void Qwen3_5VisionModel::forward_z2(
    const at::Tensor& input,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    int64_t num_patches_in,
    const std::optional<at::Tensor>& step0_pos,
    uint64_t epoch,
    uint64_t plan_hash) {
    RECORD_FUNCTION("qwen3_5_vision_forward_z2", {});
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "Qwen3_5VisionModel Z2 forward requires one active outer Graph");
    TORCH_CHECK(z2_bound_ && z2_epoch_ == epoch &&
                    z2_plan_hash_ == plan_hash,
                "Qwen3_5VisionModel Z2 forward: stale epoch/plan hash");
    TORCH_CHECK(num_patches_in == QWEN3_5_Z2_NUM_PATCHES,
                "Qwen3_5VisionModel Z2 forward requires exactly ",
                QWEN3_5_Z2_NUM_PATCHES, " patches");
    (void)forward_impl(
        input, k_caches, v_caches, num_patches_in, step0_pos,
        /*fusion_target=*/std::nullopt, /*fusion_row_starts=*/{},
        /*temporal_num_frames=*/1, /*camera_batch_count=*/1,
        /*z2_dispatch=*/true);
}

// ── emit_step0: pre_layers_fn — pixel → hidden, once per forward, in-graph,
//    before the layer loop:
//
//      [DDR] pixel [seq, patch_dim]
//        │ broadcast (每核一份完整 pixel — col-parallel 不切 K)
//      [SPM] s0_pixel                                    每核 identical
//        │ ★ DIVERGE ★ linear partition=1 (col): 核 i 只算列 [i*h/8, (i+1)*h/8)
//      [SPM] s0_patch [cs0, h/8]                         每核不同
//        │ broadcast uploaded HF position tensor
//      [SPM] s0_pos_emb [cs0, h]                         每核 identical
//        │ ★ CONVERGE ★ all_gather(axis=1) then += s0_pos_emb
//      [SPM] s0_attn_in [cs0, h]                         每核 identical
//        │ scatter → hidden_buf_[chunk]
//
//    The all-gather and residual add are emitted as two separate launches.
void Qwen3_5VisionModel::emit_step0() {
    const int64_t N  = current_num_patches_;
    const int64_t h  = hidden_size();
    const int64_t pd = patch_dim_;
    const int64_t local_h = h / NUM_CORES;
    const bool temporal_pos = current_temporal_num_frames_ > 1;
    const int64_t cs0 = current_compact_step0_
        ? G05_COMPACT_STEP0_ROWS
        : temporal_pos ? temporal_num_frames_ * temporal_patches_per_frame_
                       : QWEN3_5_VISION_STEP0_CHUNK;

    // Bias: COL-parallel splits it across cores with `torch.split(b, local_n)`,
    // matching the tower's fc1_bias convention.
    // ⚠️ NOT the row-parallel convention (fill-zero + broadcast to core 0 only) —
    // that one is for o_proj/fc2, whose partial sums get all-reduced afterwards.
    rpu_launch_ddr_scatter_spm_dma(
        pe_b_.data_ptr<c10::Half>(),
        local_h, local_h * DWIDTH,
        addr(0, "s0_bias"), NUM_CORES);

    for (int64_t off = 0; off < N; off += cs0) {
        const int64_t len = std::min<int64_t>(cs0, N - off);
        uint32_t patch_input_addr;
        if (current_compact_step0_) {
            TORCH_CHECK(len == G05_COMPACT_STEP0_ROWS,
                        "compact STEP0 requires full three-frame chunks");
            const int64_t frame_start = off / G05_COMPACT_PATCHES_PER_FRAME;
            const int64_t compact_frame_elems =
                G05_COMPACT_CHANNELS * G05_COMPACT_IMAGE_SIDE *
                G05_COMPACT_IMAGE_SIDE;
            const uint32_t layout_a = addr(0, "s0_layout_a");
            const uint32_t layout_b = addr(0, "s0_layout_b");

            // Upload three unexpanded CHW frames per camera chunk. Every core
            // performs the same pure-layout sequence so patch_embed can consume
            // the final full-K rows without a cross-core SPM copy.
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &pixel_src_base_,
                /*src_offset_bytes=*/frame_start * compact_frame_elems * DWIDTH,
                /*num_elements=*/G05_COMPACT_STEP0_FRAMES * compact_frame_elems,
                layout_a, NUM_CORES);
            rpu_launch_tile_spm_kernel(
                layout_a, layout_b,
                /*dim0=*/G05_COMPACT_STEP0_FRAMES * G05_COMPACT_CHANNELS,
                /*dim1=*/G05_COMPACT_IMAGE_SIDE,
                /*dim2=*/G05_COMPACT_IMAGE_SIDE,
                /*repeat0=*/1, /*repeat1=*/2, /*repeat2=*/1, NUM_CORES);

            // ① patchify each plane: [gh,ph,gw,pw] -> [gh,gw,ph,pw].
            rpu_launch_permute4d_0213_spm_kernel(
                layout_b, layout_a,
                /*q=*/G05_COMPACT_STEP0_FRAMES * G05_COMPACT_CHANNELS * 2 *
                    G05_COMPACT_GRID_SIDE,
                /*b=*/G05_COMPACT_PATCH_SIDE,
                /*n=*/G05_COMPACT_GRID_SIDE,
                /*c=*/G05_COMPACT_PATCH_SIDE, NUM_CORES);
            // ② plane-major -> patch-major.
            rpu_launch_permute4d_0213_spm_kernel(
                layout_a, layout_b,
                /*q=*/G05_COMPACT_STEP0_FRAMES,
                /*b=*/G05_COMPACT_CHANNELS * 2,
                /*n=*/G05_COMPACT_PATCHES_PER_FRAME,
                /*c=*/G05_COMPACT_PATCH_SIDE * G05_COMPACT_PATCH_SIDE,
                NUM_CORES);
            // ③ Reinterpret T×C before swapping to reproduce the official
            // [c0,c1,c0,c2,c1,c2] block order exactly.
            rpu_launch_permute4d_0213_spm_kernel(
                layout_b, layout_a,
                /*q=*/G05_COMPACT_STEP0_ROWS,
                /*b=*/2, /*n=*/G05_COMPACT_CHANNELS,
                /*c=*/G05_COMPACT_PATCH_SIDE * G05_COMPACT_PATCH_SIDE,
                NUM_CORES);
            // ④ Raster rows -> 2x2 spatial-merge group row order.
            rpu_launch_permute4d_0213_spm_kernel(
                layout_a, layout_b,
                /*q=*/G05_COMPACT_STEP0_FRAMES * (G05_COMPACT_GRID_SIDE / 2),
                /*b=*/2, /*n=*/G05_COMPACT_GRID_SIDE / 2,
                /*c=*/2 * pd, NUM_CORES);
            patch_input_addr = layout_b;
        } else {
            // Folded pixel rows are fresh per forward, so their address must be
            // patched at REPLAY. A fixed DMA would silently reuse forward #1.
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &pixel_src_base_,
                /*src_offset_bytes=*/off * pd * DWIDTH,
                /*num_elements=*/len * pd,
                addr(0, "s0_pixel"), NUM_CORES);
            patch_input_addr = addr(0, "s0_pixel");
        }

        // patch_embed: col-parallel GEMM. m=len, n=h (split → h/8 per core), k=pd.
        // Dims are FULL/global here — the graph-layer op divides by tp internally.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            patch_input_addr, pe_w_, addr(0, "s0_patch"),
            len, h, pd, /*partition=*/1, NUM_CORES, addr(0, "s0_bias"));

        if (temporal_pos) {
            // Python packs position rows as [core,N,H/8]. Add each core's
            // matching col shard before gather; gather is a copy, so this is
            // byte-equivalent to the full-width add while avoiding 8 broadcasts.
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &step0_pos_src_base_,
                /*src_offset_bytes=*/off * local_h * DWIDTH,
                /*elements_per_core=*/len * local_h,
                /*core_stride_bytes=*/N * local_h * DWIDTH,
                addr(0, "s0_pos_emb"), NUM_CORES);
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "s0_patch"), addr(0, "s0_pos_emb"), addr(0, "s0_patch"),
                len * local_h, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
            rpu_launch_all_gather_spm_kernel(
                addr(0, "s0_patch"), addr(0, "s0_attn_in"),
                /*n=*/len, /*chunk_elems=*/local_h, /*dwidth=*/DWIDTH, NUM_CORES);
        } else {
            // Generic single-image path retains the full-width position broadcast.
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &step0_pos_src_base_,
                /*src_offset_bytes=*/off * h * DWIDTH,
                /*num_elements=*/len * h,
                addr(0, "s0_pos_emb"), NUM_CORES);
            rpu_launch_all_gather_spm_kernel(
                addr(0, "s0_patch"), addr(0, "s0_attn_in"),
                /*n=*/len, /*chunk_elems=*/local_h, /*dwidth=*/DWIDTH, NUM_CORES);
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "s0_attn_in"), addr(0, "s0_pos_emb"), addr(0, "s0_attn_in"),
                len * h, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
        }

        // → hidden_buf_[off..off+len). Core 0's copy is complete (every core holds
        // an identical full-width result), so a 1-core copy suffices. hidden_buf_
        // is allocated once and never re-allocated, so a fixed DMA is safe here.
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "s0_attn_in"),
            hidden_buf_.data_ptr<c10::Half>() + off * h,
            len * h);
    }
}

// ── emit_merger: post_layers_fn — patch merger, once per forward, in-graph,
//    AFTER the layer loop. HF's Qwen3VLVisionPatchMerger:
//        x = fc2(gelu(fc1(ln_q(x).view(-1, hidden*4))))
//    The tower is KV_FIRST-chunked, so at post_layers time the complete
//    [S, hidden] output lives in output_tensor() DDR (NOT SPM) — we read it back
//    in chunks. Per merger chunk (over pre-merge rows, MERGER_CHUNK, %4==0):
//
//      [DDR] output_tensor()[off..off+len)  full-width [len, hidden]
//        │ broadcast (每核一份 — LN + col-linear both consume full width)
//      [SPM] m_in [len, hidden]                            每核 identical
//        │ LayerNorm(hidden), eps inside variance, in-place
//      [SPM] m_in (normed)
//        │ reinterpret [len, hidden] → [len/4, hidden*4] (byte-identical: only
//        │ M/K change, m_in is per-core dense full-width & len%4==0)
//        │ ★ fc1 linear partition=1 (COL): core i computes cols [i*mh/8,(i+1)*mh/8)
//      [SPM] m_fc1 [len/4, hidden*4/8]                     每核不同 (col shard)
//        │ GELU (explicit ERF kernel = HF nn.GELU()) — per-core on the shard
//      [SPM] m_fc1 (gelu'd)
//        │ ★ fc2 linear partition=0 (ROW): core i owns K-slice = fc1's col shard
//        │   → per-core PARTIAL sum [len/4, out_hidden]
//      [SPM] m_fc2 [len/4, out_hidden]                     每核 partial
//        │ all_reduce_sum_residual(+m_zero) → sum partials, broadcast全核
//      [SPM] m_out [len/4, out_hidden]                     每核 identical (full)
//        │ core-0 copy → merged_buf_[off/4 ..)
//
//    ⚠️ fc1 MUST be col (partition=1), fc2 MUST be row (partition=0) — see the
//    铁律 note. The col shard from fc1 lines up exactly with the K-slice fc2 row
//    needs, so NO all_gather between them (identical to the encoder MLP at
//    build_layer_subgraph's Phase 5/6).
void Qwen3_5VisionModel::emit_merger() {
    const int64_t cameras = current_camera_batch_count_;
    const int64_t camera_patches = current_output_patches_ / cameras;
    const int64_t h   = hidden_size();               // 1024 — per-patch dim / LN width
    const int64_t mu  = QWEN3_5_SPATIAL_MERGE_UNIT;  // 4
    const int64_t mh  = merger_hidden_;              // hidden*4 = 4096 — fc1 N/K, fc2 K
    const int64_t oh  = out_hidden_size_;            // merger output width (text hidden)
    const int64_t cs  = QWEN3_5_VISION_MERGER_CHUNK; // %4==0
    const int64_t local_mh = mh / NUM_CORES;         // 4096/8 = 512 — fc1 col output per core
    // HF nn.LayerNorm(eps=1e-6) — eps is INSIDE the variance. Hard-coded, NOT
    // eps_ (that is the encoder LN1/LN2 eps; the merger norm is a distinct module
    // and we do not assume they coincide).
    constexpr double merger_eps = 1e-6;

    if (z2_bound_) {
        TORCH_CHECK(current_output_patches_ == QWEN3_5_Z2_NUM_PATCHES &&
                        z2_prepared_num_patches_ == current_output_patches_ &&
                        z2_slice_addr_ != 0,
                    "qwen3_5_vision Z2 merger requires the prepared 256-patch "
                    "typed row slice");
        TORCH_CHECK(addr(0, "residual1") == addr(0, "m_in"),
                    "qwen3_5_vision Z2 merger lost the residual1/m_in identity");
    } else {
        TORCH_CHECK(merger_dst_bases_[0] != 0,
                    "qwen3_5_vision merger destination is unset");
    }

    for (int64_t camera = 0; camera < cameras; ++camera) {
        const int64_t source_base = current_temporal_num_frames_ > 1
            ? (camera * current_temporal_num_frames_ +
               current_temporal_num_frames_ - 1) * camera_patches
            : 0;
        const int64_t merged_camera_base =
            camera * (camera_patches / QWEN3_5_SPATIAL_MERGE_UNIT);
        TORCH_CHECK(
            !merger_wrote_fusion_target_ || merger_dst_bases_[camera] != 0,
            "qwen3_5_vision merger camera destination is unset");
        for (int64_t off = 0; off < camera_patches; off += cs) {
            const int64_t len = std::min<int64_t>(cs, camera_patches - off);
            const int64_t m = len / mu;  // merged rows this chunk

            if (z2_bound_) {
                TORCH_CHECK(cameras == 1 && off == 0 &&
                                len == current_output_patches_,
                        "qwen3_5_vision Z2 merger requires one full input chunk");
                // The final tower all-reduce already left the complete
                // [256,1024] tensor at m_in's exact physical address.
            } else {
            // (1) 读回本 chunk 全宽输出 → 每核一份。FIXED DMA：output_tensor() 的地址
            //     是 per-shape registry-stable，烤地址安全。
            rpu_launch_ddr_broadcast_spm_dma(
                output_tensor().data_ptr<c10::Half>() +
                    (source_base + off) * h,
                len * h,
                addr(0, "m_in"), NUM_CORES);
            }

            // (2) LayerNorm over hidden (真 LayerNorm，eps 在方差内，无 skip)。gamma/beta
            //     由 preload broadcast 到所有核。
            rpu_launch_layernorm_spm_kernel(
                addr(0, "m_in"), addr(0, "m_in"),
                addr(0, "m_norm_g"), addr(0, "m_norm_b"),
                len, h, merger_eps, /*has_skip=*/false, /*skip=*/0, NUM_CORES);

            // (3) reinterpret [len, hidden] → [m, hidden*4]（只改 M/K，字节等价）+ fc1
            //     COL 线性 (partition=1)。⚠️ fc1 必 col、fc2 必 row，两者维度对 col/row
            //     约束双满足、assert 拦不住，只能靠角色手工钉死。
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "m_in"), merger_fc1_w_, addr(0, "m_fc1"),
                /*M=*/m, /*N=*/mh, /*K=*/mh, /*partition=*/1, NUM_CORES,
                addr(0, "m_fc1_bias"));

            // (4) HF nn.GELU() default: exact erf, per-core on the col shard.
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "m_fc1"), addr(0, "m_fc1"),
                m * local_mh, ValuOpType::ADD, GeluMode::ERF, NUM_CORES);

            // (5) fc2 ROW 线性 (partition=0)。输入是 fc1 的 col shard（正好 = 本核的
            //     K-slice），产每核部分和。fc2 bias core-0-only（行约定，reduce 后计一次）。
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "m_fc1"), merger_fc2_w_, addr(0, "m_fc2"),
                /*M=*/m, /*N=*/oh, /*K=*/mh, /*partition=*/0, NUM_CORES,
                addr(0, "m_fc2_bias"));

            // (6) 清零 residual buffer：all_reduce_sum_residual 必须有真 residual 地址
            //     （不能传 0），merger 无 residual → 用 graph-aware fill 灌零。
            rpu_launch_fill_spm_kernel(
                addr(0, "m_zero"), m * oh, c10::Half(0.0f), NUM_CORES);

            // (7) all_reduce 行分区部分和 + 零 residual → 完整结果，broadcast 全核。
            //     output = reduce(input) + residual。in≠residual≠out（三块独立，kernel 要求）。
            //     exact G0.5 K1 Vision keeps two-stage enabled but uses
            //     the non-v2 scatter precision policy; generic Vision retains
            //     the process selector's automatic v2 choice.
            const uint32_t merger_out =
                z2_bound_ ? z2_slice_addr_ : addr(0, "m_out");
            if (allreduce_force_non_v2_) {
                rpu_launch_all_reduce_sum_residual_non_v2_kernel(
                    addr(0, "m_fc2"), addr(0, "m_zero"), merger_out,
                    m, oh, NUM_CORES, NUM_CORES);
            } else {
                rpu_launch_all_reduce_sum_residual_kernel(
                    addr(0, "m_fc2"), addr(0, "m_zero"), merger_out,
                    m, oh, NUM_CORES, NUM_CORES);
            }

            // (8) Camera-major temporal output keeps each last frame in a separate
            //     source slice. Standalone packs those slices contiguously; direct
            //     fusion writes each one to its independently mutable text run.
            if (z2_bound_) {
                continue;
            } else if (!merger_wrote_fusion_target_) {
                rpu_launch_spm_copy_ddr_dma_mutable(
                    addr(0, "m_out"), &merger_dst_bases_[0],
                    (merged_camera_base + off / mu) * oh * sizeof(c10::Half),
                    m * oh);
            } else {
                rpu_launch_spm_copy_ddr_dma_mutable(
                    addr(0, "m_out"), &merger_dst_bases_[camera],
                    (off / mu) * oh * sizeof(c10::Half), m * oh);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LAYER BUILD (KV_FIRST two-phase)
// ─────────────────────────────────────────────────────────────────────────────

// ── emit_kv_first_body: KV_FIRST Phase 1 — LN1 + Q/K/V Linear + 2D rope
//    (pos_offset=cur_pos) + KV insert(@cur_pos 累积) + 把 rope 后的 Q 存进
//    q_ddr_buf_（使用两相拆分）。 ───────────────────────────────────────────────
void Qwen3_5VisionModel::emit_temporal_layer(int layer_idx) {
    // The framework still visits all 24 layers, but each owner emits one
    // group-major block. Spatial-only followers are therefore intentional
    // no-ops when their framework callbacks arrive later.
    if (layer_idx != 0 && layer_idx % 4 != 3) return;
    const int last_layer_idx = static_cast<int>(std::min<int64_t>(
        num_layers() - 1, layer_idx == 0 ? 2 : layer_idx + 3));
    const auto& lw = layer_weights_[layer_idx];
    const int64_t frames = temporal_num_frames_;
    const int64_t cameras = current_camera_batch_count_;
    const int64_t patches = temporal_patches_per_frame_;
    const int64_t group = G05_TEMPORAL_PATCH_GROUP;
    const int64_t h = hidden_size();
    const int64_t nq = num_q_heads();
    const int64_t hd = head_dim();
    const int64_t local_q_dim = (nq / NUM_CORES) * hd;
    const int64_t composite_heads = group * nq;
    const int64_t camera_rows = frames * group;
    const int64_t patch_groups = patches / group;
    const int64_t camera_hidden_rows = frames * patches;
    const int64_t permute_inner = group * local_q_dim;
    const bool spacetime = layer_idx % 4 == 3;

    if (spacetime) {
        for (int64_t camera = 0; camera < cameras; ++camera) {
            ChunkInfo camera_slice{
                static_cast<int>(camera),
                camera * camera_hidden_rows,
                camera_hidden_rows,
                current_num_patches_,
            };
            emit_layer_input_dma(layer_idx, camera_slice, "t_frame");
            rpu_launch_permute4d_0213_spm_kernel(
                addr(0, "t_frame"), addr(0, "t_grouped"),
                1, frames, patch_groups, group * h, NUM_CORES);

            for (int64_t patch_off = 0; patch_off < patches; patch_off += group) {
                const uint32_t temporal_hidden =
                    addr(0, "t_grouped") +
                    (patch_off / group) * camera_rows * h * DWIDTH;

                rpu_launch_eltwise_binary_Bx1xC_BxNxC_spm_kernel(
                    addr(0, "temporal_pe"), temporal_hidden,
                    temporal_hidden, frames, group, h,
                    c10::Half(1.0f), ValuOpType::ADD, false);

                // Accumulate each [frames,group] result patch-group-major in
                // SPM. One permute below restores frame-major spatial rows.
                const uint32_t stage_delta =
                    (patch_off / group) * camera_rows * local_q_dim * DWIDTH;
                rpu_launch_layernorm_spm_kernel(
                    temporal_hidden, temporal_hidden,
                    layer_addr(layer_idx, 0, "ln1_gamma"),
                    layer_addr(layer_idx, 0, "ln1_beta"),
                    camera_rows, h, eps_, false, 0, NUM_CORES);
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    temporal_hidden, lw.q_w,
                    addr(0, "t_q_stage") + stage_delta,
                    camera_rows, nq * hd, h, 1, NUM_CORES,
                    layer_addr(layer_idx, 0, "q_bias"));
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    temporal_hidden, lw.k_w,
                    addr(0, "t_k_stage") + stage_delta,
                    camera_rows, nq * hd, h, 1, NUM_CORES,
                    layer_addr(layer_idx, 0, "k_bias"));
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    temporal_hidden, lw.v_w, addr(0, "t_v"),
                    camera_rows, nq * hd, h, 1, NUM_CORES,
                    layer_addr(layer_idx, 0, "v_bias"));

                rpu_launch_insert_kcache_spm_unified(
                    temporal_k_cache_, 0,
                    addr_offset("t_k_stage").value + stage_delta,
                    frames, composite_heads, hd, NUM_CORES);
                rpu_launch_insert_vcache_spm_unified(
                    temporal_v_cache_, 0,
                    addr_offset("t_v").value,
                    frames, composite_heads, hd, NUM_CORES);
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    temporal_k_cache_, temporal_v_cache_,
                    1 /*MASK_LTM*/, 1.0 / std::sqrt(static_cast<double>(hd)),
                    addr_offset("t_q_stage").value + stage_delta,
                    addr_offset("t_out_stage").value + stage_delta,
                    addr_offset("t_sdpa_tmp").value, 0,
                    frames, composite_heads, composite_heads, hd, frames,
                    NUM_CORES, NUM_CORES);
            }

            rpu_launch_permute3d_spm_kernel(
                addr(0, "t_q_stage"), addr(0, "q"),
                patch_groups, frames, permute_inner, {1, 0, 2}, NUM_CORES);
            rpu_launch_permute3d_spm_kernel(
                addr(0, "t_k_stage"), addr(0, "k"),
                patch_groups, frames, permute_inner, {1, 0, 2}, NUM_CORES);
            rpu_launch_permute3d_spm_kernel(
                addr(0, "t_out_stage"), addr(0, "v"),
                patch_groups, frames, permute_inner, {1, 0, 2}, NUM_CORES);

            for (int64_t half = 0; half < 2; ++half) {
                emit_spatial_group(
                    layer_idx, last_layer_idx,
                    (camera * frames + half * 3) * patches,
                    true,
                    half * G05_SPATIAL_GROUP_ROWS);
            }
        }
    } else {
        for (int64_t offset = 0; offset < current_num_patches_;
             offset += G05_SPATIAL_GROUP_ROWS) {
            emit_spatial_group(layer_idx, last_layer_idx, offset, false);
        }
    }
}

void Qwen3_5VisionModel::emit_spatial_group(
    int first_layer_idx, int last_layer_idx,
    int64_t offset, bool staged_qkv,
    int64_t staged_row_offset)
{
    const int64_t seq_len = G05_SPATIAL_GROUP_ROWS;
    const int64_t h = hidden_size();
    const int64_t nq = num_q_heads();
    const int64_t hd = head_dim();
    const int64_t is_ = intermediate_size();
    const int64_t local_q_dim = (nq / NUM_CORES) * hd;
    const int64_t local_inter = is_ / NUM_CORES;
    const int64_t camera_stage_rows =
        temporal_num_frames_ * temporal_patches_per_frame_;
    ChunkInfo group{
        static_cast<int>(offset / seq_len),
        offset,
        seq_len,
        current_num_patches_,
    };

    emit_layer_input_dma(first_layer_idx, group);
    if (staged_qkv) {
        TORCH_CHECK(staged_row_offset >= 0 &&
                    staged_row_offset + seq_len <= camera_stage_rows,
                    "G0.5 temporal SPM stage offset is out of range");
    }

    const int64_t local_heads = nq / NUM_CORES;
    for (int layer_idx = first_layer_idx;
         layer_idx <= last_layer_idx; ++layer_idx) {
        const auto& lw = layer_weights_[layer_idx];
        const bool use_staged_qkv =
            staged_qkv && layer_idx == first_layer_idx;
        const uint32_t staged_delta = use_staged_qkv
            ? staged_row_offset * local_q_dim * DWIDTH
            : 0;
        const uint32_t q_addr = addr(0, "q") + staged_delta;
        const uint32_t k_addr = addr(0, "k") + staged_delta;
        const uint32_t q_off = addr_offset("q").value + staged_delta;
        const uint32_t k_off = addr_offset("k").value + staged_delta;
        const uint32_t v_off = addr_offset("v").value + staged_delta;

        if (!use_staged_qkv) {
            rpu_launch_layernorm_spm_kernel(
                addr(0, "residual1"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "ln1_gamma"),
                layer_addr(layer_idx, 0, "ln1_beta"),
                seq_len, h, eps_, false, 0, NUM_CORES);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.q_w, addr(0, "q"),
                seq_len, nq * hd, h, 1, NUM_CORES,
                layer_addr(layer_idx, 0, "q_bias"));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.k_w, addr(0, "k"),
                seq_len, nq * hd, h, 1, NUM_CORES,
                layer_addr(layer_idx, 0, "k_bias"));
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.v_w, addr(0, "v"),
                seq_len, nq * hd, h, 1, NUM_CORES,
                layer_addr(layer_idx, 0, "v_bias"));
        }

        rpu_launch_rope_2d_spm_kernel(
            q_addr, q_addr,
            addr(0, "freq_cos_spm"), addr(0, "freq_sin_spm"),
            position_idx_keepalive_.data_ptr<int16_t>(),
            offset, seq_len, local_heads, hd, hd, NUM_CORES);
        rpu_launch_rope_2d_spm_kernel(
            k_addr, k_addr,
            addr(0, "freq_cos_spm"), addr(0, "freq_sin_spm"),
            position_idx_keepalive_.data_ptr<int16_t>(),
            offset, seq_len, local_heads, hd, hd, NUM_CORES);

        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        rpu_launch_insert_kcache_spm_unified(
            k_cache, 0, k_off,
            seq_len, nq, hd, NUM_CORES);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, 0, v_off,
            seq_len, nq, hd, NUM_CORES);
        rpu_launch_sdpa_spm_minibatch_kernel(
            k_cache, v_cache,
            0 /*MASK_NONE*/, 1.0 / std::sqrt(static_cast<double>(hd)),
            q_off,
            addr_offset("sdpa_out").value,
            addr_offset("sdpa_tmp").value, 0,
            seq_len, nq, nq, hd,
            temporal_patches_per_frame_, 3,
            NUM_CORES, NUM_CORES);

        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, NUM_CORES,
            layer_addr(layer_idx, 0, "o_bias"));
        if (allreduce_force_non_v2_) {
            rpu_launch_all_reduce_sum_residual_non_v2_kernel(
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
                seq_len, h, NUM_CORES, NUM_CORES);
        } else {
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
                seq_len, h, NUM_CORES, NUM_CORES);
        }
        rpu_launch_layernorm_spm_kernel(
            addr(0, "input_norm"), addr(0, "oproj"),
            layer_addr(layer_idx, 0, "ln2_gamma"),
            layer_addr(layer_idx, 0, "ln2_beta"),
            seq_len, h, eps_, false, 0, NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "oproj"), lw.fc1_w, addr(0, "fc1"),
            seq_len, is_, h, 1, NUM_CORES,
            layer_addr(layer_idx, 0, "fc1_bias"));
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "fc1"), addr(0, "fc1"),
            seq_len * local_inter, ValuOpType::ADD,
            GeluMode::TANH, NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "fc1"), lw.fc2_w, addr(0, "oproj"),
            seq_len, h, is_, 0, NUM_CORES,
            layer_addr(layer_idx, 0, "fc2_bias"));
        if (allreduce_force_non_v2_) {
            rpu_launch_all_reduce_sum_residual_non_v2_kernel(
                addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"),
                seq_len, h, NUM_CORES, NUM_CORES);
        } else {
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"),
                seq_len, h, NUM_CORES, NUM_CORES);
        }

        if (layer_idx == last_layer_idx) {
            emit_layer_output_dma(last_layer_idx, group, "residual1");
        }
        if (get_debug_export()) {
            c10::Half* dh = dbg_hidden_.data_ptr<c10::Half>() +
                ((int64_t)layer_idx * current_num_patches_ + offset) * h;
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "residual1"), dh, seq_len * h,
                seq_len * h * DWIDTH, 1);
        }
    }
}

void Qwen3_5VisionModel::emit_kv_first_body(int layer_idx, const ChunkInfo& chunk) {
    if (current_temporal_num_frames_ > 1) {
        if (chunk.idx == 0) emit_temporal_layer(layer_idx);
        return;
    }
    const auto& lw = layer_weights_[layer_idx];
    int64_t seq_len = chunk.len;
    int64_t cur_pos = ctx().position + chunk.offset;   // vision position=0 ⇒ =chunk.offset
    int64_t h  = hidden_size();
    int64_t nq = num_q_heads();
    int64_t hd = head_dim();
    const int64_t local_heads   = nq / NUM_CORES;
    const int64_t head_dim_pad  = hd;

    // Phase 1: input DMA + LN1
    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }
    rpu_launch_layernorm_spm_kernel(
        addr(0, "residual1"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "ln1_gamma"), layer_addr(layer_idx, 0, "ln1_beta"),
        seq_len, h, eps_, false, 0, NUM_CORES);

    // Phase 2: Q / K / V Linear (+bias)
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "input_norm"), lw.q_w, addr(0, "q"),
        seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "q_bias"));
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "input_norm"), lw.k_w, addr(0, "k"),
        seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "k_bias"));
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "input_norm"), lw.v_w, addr(0, "v"),
        seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "v_bias"));

    // Phase 2.5: 2D RoPE on Q,K — pos_offset=cur_pos, num_tokens=本 chunk 行数.
    // SPM variant: cos/sin read from the SPM-resident tables populated by
    // emit_preload_weights (the retained graph replays that preload), not DDR.
    // position_idx stays in DDR. Same broadcast/addr(0,...) convention as x/y.
    rpu_launch_rope_2d_spm_kernel(
        addr(0, "q"), addr(0, "q"),
        addr(0, "freq_cos_spm"), addr(0, "freq_sin_spm"),
        position_idx_keepalive_.data_ptr<int16_t>(),
        cur_pos, seq_len, local_heads, hd, head_dim_pad, NUM_CORES);
    rpu_launch_rope_2d_spm_kernel(
        addr(0, "k"), addr(0, "k"),
        addr(0, "freq_cos_spm"), addr(0, "freq_sin_spm"),
        position_idx_keepalive_.data_ptr<int16_t>(),
        cur_pos, seq_len, local_heads, hd, head_dim_pad, NUM_CORES);

    // Phase 3a: KV cache insert at ABSOLUTE position (累积；单 chunk 时 cur_pos=0 等价旧行为)
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    rpu_launch_insert_kcache_spm_unified(
        k_cache, cur_pos, addr_offset("k").value, seq_len, nq, hd, NUM_CORES);
    rpu_launch_insert_vcache_spm_unified(
        v_cache, cur_pos, addr_offset("v").value, seq_len, nq, hd, NUM_CORES);

    // 存 rope 后的 Q 到 q_ddr_buf_（position-indexed，照 gemma:823-836）
    TORCH_CHECK(q_ddr_buf_.defined(),
                "Qwen3_5VisionModel: q_ddr_buf_ not allocated in KV_FIRST mode");
    int64_t local_q_dim         = local_heads * hd;
    int64_t q_local_elems       = seq_len * local_q_dim;
    c10::Half* q_ddr_base       = q_ddr_buf_.data_ptr<c10::Half>();
    int64_t q_elem_offset       = chunk.offset * local_q_dim;
    int64_t q_core_stride_bytes = q_ddr_buf_.size(1) * local_q_dim * DWIDTH;
    rpu_launch_spm_scatter_ddr_dma(
        addr(0, "q"), q_ddr_base + q_elem_offset,
        q_local_elems, q_core_stride_bytes, NUM_CORES);

    // DEBUG: snapshot rope'd Q per-layer per-core → dbg_q_[layer, core, chunk.offset:, :].
    // Same live "q" SPM source as the q_ddr_buf_ store above. Core c writes to
    // base + c*(N*local_q_dim); rows offset by chunk.offset (multi-chunk fills all rows).
    if (get_debug_export() && dbg_q_enabled()) {
        const int64_t N = current_num_patches_;
        c10::Half* dq = dbg_q_.data_ptr<c10::Half>()
                      + ((int64_t)layer_idx * NUM_CORES * N + chunk.offset) * local_q_dim;
        rpu_launch_spm_scatter_ddr_dma(
            addr(0, "q"), dq, q_local_elems,
            /*core_stride_bytes=*/N * local_q_dim * DWIDTH, NUM_CORES);
    }
}

// ── build_layer_subgraph: KV_FIRST Phase 2 — Phase 1 已做 LN/QKV/rope/存 Q 并插满
//    全量 KV。这里：重读 input(残差用) + 载回本 chunk 的 rope 后 Q + SDPA(读全量 KV) + MLP。──
void Qwen3_5VisionModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    if (current_temporal_num_frames_ > 1) return;
    const auto& lw = layer_weights_[layer_idx];
    int64_t seq_len = chunk.len;
    int64_t h = hidden_size();
    int64_t nq = num_q_heads();
    int64_t hd = head_dim();
    int64_t is_ = intermediate_size();
    int64_t local_inter = is_ / NUM_CORES;

    // KV_FIRST Phase 2。Phase 1(emit_kv_first_body) 已做 LN/QKV/rope/存 Q，并把全量 KV 插满。
    // 这里：重读 input(残差用) + 载回本 chunk 的 rope 后 Q + SDPA(读全量 KV) + MLP。
    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }

    // 载回本 chunk 的 rope 后 Q(从 q_ddr_buf_ 的 chunk.offset 处)到 SPM "q_comp"(照 gemma:863-876)
    const int64_t local_heads = nq / NUM_CORES;
    int64_t local_q_dim   = local_heads * hd;
    int64_t q_local_elems = seq_len * local_q_dim;
    c10::Half* q_ddr_base = q_ddr_buf_.data_ptr<c10::Half>();
    int64_t q_elem_offset = chunk.offset * local_q_dim;
    int64_t q_core_stride = q_ddr_buf_.size(1) * local_q_dim * DWIDTH;
    rpu_launch_ddr_scatter_spm_dma(
        q_ddr_base + q_elem_offset, q_local_elems, q_core_stride,
        addr(0, "q_comp"), NUM_CORES);

    // SDPA：query = 本 chunk 的 q_comp(seq_len 行)，K/V = cache 全量(current_num_patches_) → 双向。
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    double attn_scale = 1.0 / std::sqrt(static_cast<double>(orig_head_dim_));
    rpu_launch_sdpa_spm_unified_kernel_v2(
        k_cache, v_cache,
        0 /*MASK_NONE*/, attn_scale,
        addr_offset("q_comp").value,
        addr_offset("sdpa_out").value,
        addr_offset("sdpa_tmp").value, 0,
        seq_len, nq, nq, hd,
        current_num_patches_, NUM_CORES, NUM_CORES);

    // Phase 4: O_proj (row-partition, with bias) + AllReduce + Residual.
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, 0, NUM_CORES,
        layer_addr(layer_idx, 0, "o_bias"));
    if (allreduce_force_non_v2_) {
        rpu_launch_all_reduce_sum_residual_non_v2_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
            seq_len, h, NUM_CORES, NUM_CORES);
    } else {
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
            seq_len, h, NUM_CORES, NUM_CORES);
    }

    // Phase 5: LayerNorm2 + fc1+bias + GELU (gelu_pytorch_tanh per Q3.5 config).
    rpu_launch_layernorm_spm_kernel(
        addr(0, "input_norm"), addr(0, "oproj"),
        layer_addr(layer_idx, 0, "ln2_gamma"), layer_addr(layer_idx, 0, "ln2_beta"),
        seq_len, h, eps_, false, 0, NUM_CORES);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "oproj"), lw.fc1_w, addr(0, "fc1"),
        seq_len, is_, h, 1, NUM_CORES,
        layer_addr(layer_idx, 0, "fc1_bias"));
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "fc1"), addr(0, "fc1"),
        seq_len * local_inter, ValuOpType::ADD,
        GeluMode::TANH, NUM_CORES);

    // Phase 6: fc2 + AllReduce + Residual (writes back to residual1 in
    // SPM_RESIDENT mode so the next layer's Phase 1 LN1 reads directly).
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "fc1"), lw.fc2_w, addr(0, "oproj"),
        seq_len, h, is_, 0, NUM_CORES,
        layer_addr(layer_idx, 0, "fc2_bias"));
    if (allreduce_force_non_v2_) {
        rpu_launch_all_reduce_sum_residual_non_v2_kernel(
            addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"),
            seq_len, h, NUM_CORES, NUM_CORES);
    } else {
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"),
            seq_len, h, NUM_CORES, NUM_CORES);
    }

    const bool z2_final = z2_bound_ &&
                          layer_idx == num_layers() - 1 &&
                          !ctx().output_to_spm;
    if (z2_final) {
        TORCH_CHECK(chunk.idx == 0 && chunk.offset == 0 &&
                        chunk.len == z2_prepared_num_patches_,
                    "Qwen3_5VisionModel Z2 requires one full-sequence final chunk");
    }

    // Output DMA (SPM_RESIDENT skips it for non-last layers; the last layer writes the caller's
    // output_tensor_). Channel 0 shares the compute stream, so queue ordering already places this
    // after the all-reduce above; no synthetic compute/barrier is needed.
    if (!ctx().output_to_spm && !z2_final) {
        emit_layer_output_dma(layer_idx, chunk, "residual1");
    }

    // DEBUG: snapshot this layer's output hidden (residual1 is replicated → read core 0) →
    // dbg_hidden_[layer, chunk.offset:, :].
    //
    // Deliberately emitted AFTER the output DMA so debug mode does not perturb the production
    // operation order. It still reads the same value because both copies are on stream 0 and
    // nothing writes residual1 in between.
    //
    // num_cores=1 is also deliberate: it makes rpu_memcpy.cpp's `for (ch = 1; ch < num_cores)`
    // fences no-ops, so this injects ZERO barriers (unlike dbg_q, which is 8-core — see
    // dbg_q_enabled() at the top of this file).
    if (get_debug_export()) {
        const int64_t N = current_num_patches_;
        c10::Half* dh = dbg_hidden_.data_ptr<c10::Half>()
                      + ((int64_t)layer_idx * N + chunk.offset) * h;
        rpu_launch_spm_scatter_ddr_dma(
            addr(0, "residual1"), dh, seq_len * h,
            /*core_stride_bytes=*/seq_len * h * DWIDTH, /*num_cores=*/1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SPM LAYOUT
// ─────────────────────────────────────────────────────────────────────────────

// ── declare_buffers: two-phase SPM set — ALL (both phases) ∪ KVIN (Phase 1)
//    ∪ COMP (Phase 2); KVIN/COMP alias the same SPM region. ─────────────────────
std::vector<BufferDecl> Qwen3_5VisionModel::declare_buffers(const LayoutContext& ctx) {
    int64_t cs = ctx.chunk_size;
    int64_t h  = hidden_size();
    int64_t nq = num_q_heads();
    int64_t hd = head_dim();
    int64_t is_ = intermediate_size();

    int64_t local_q_dim = (nq / NUM_CORES) * hd;
    int64_t local_inter = is_ / NUM_CORES;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

    int64_t res  = A(cs * h * DWIDTH);
    int64_t qkv  = A(cs * local_q_dim * DWIDTH);
    int64_t fc1  = A(cs * local_inter * DWIDTH);

    SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                        hd, /*nq*/nq, /*nkv*/nq,
                        /*cores*/NUM_CORES, /*mask*/0};
    SdpaTiling t = sdpa_compute_tiling(sdpa_cfg, cs);
    int64_t nkv_per_core = CeilDiv(nq, (int64_t)NUM_CORES);
    int64_t sdpa_tmp = A(t.tile_n_v16 * t.tile_k * nkv_per_core * CeilDiv(cs, t.tile_m) * 32);

    auto dma_safe = [&](int64_t elems) -> int64_t {
        int64_t dma_elems = ((elems + 255) / 256) * 256;
        return A(dma_elems * DWIDTH);
    };
    int64_t norm_w_sz   = dma_safe(h);
    int64_t q_bias_sz   = dma_safe(local_q_dim);
    int64_t fc1_bias_sz = dma_safe(local_inter);
    int64_t full_bias_sz = dma_safe(h);

    // 2D-RoPE freq tables [max_hw, head_dim/4] fp16, resident in SPM. Sized from
    // the registered freq_cos_ (set_rope_tables runs before layout; forward()
    // TORCH_CHECKs has_rope_ so freq_cos_ is defined by the time a real forward
    // reaches here). Fallback to max_hw_·hd/4 keeps layout well-defined if it is
    // ever probed before the tables are set.
    int64_t rope_tbl_elems = freq_cos_.defined() ? freq_cos_.numel() : max_hw_ * (hd / 4);
    int64_t rope_tbl_sz    = dma_safe(rope_tbl_elems);

    int nl = static_cast<int>(num_layers());

    // KV_FIRST 两相 scope：ALL 跨两相；KVIN 仅插 KV 相(Phase1)；COMP 仅计算相(Phase2)。
    // KVIN 与 COMP 别名同一 SPM 区(Phase1 全跑完才进 Phase2) → 峰值 = ALL + max(KVIN, COMP)。
    constexpr BufferScope ALL  = BufferScope::LayerWide;
    constexpr BufferScope KVIN = BufferScope::KvInsert;
    constexpr BufferScope COMP = BufferScope::Compute;
    constexpr BufferScope OUTSIDE = BufferScope::OutsideLayerLoop;

    // STEP 0 (pre_layers_fn) buffers — only when set_patch_embed() opted in.
    //
    // Sized by QWEN3_5_VISION_STEP0_CHUNK, NOT ctx.chunk_size: STEP 0 runs before
    // the layer loop and does its own row chunking (the framework's planner is
    // nested inside the layer loop, and pre_layers_fn gets no ChunkInfo).
    //
    // OutsideLayerLoop scope ⇒ these alias the layer buffers rather than adding
    // to them. s0_pixel dies after patch_embed, before all_gather writes
    // s0_attn_in, so those two names explicitly share the larger of their slots.
    const bool temporal_pos = current_temporal_num_frames_ > 1;
    const int64_t cs0 = current_compact_step0_
        ? G05_COMPACT_STEP0_ROWS
        : temporal_pos ? temporal_num_frames_ * temporal_patches_per_frame_
                       : QWEN3_5_VISION_STEP0_CHUNK;
    std::vector<BufferDecl> step0;
    if (has_step0_) {
        const int64_t s0_pos_width = temporal_pos ? h / NUM_CORES : h;
        if (current_compact_step0_) {
            // Two exact packed-row ping-pong roots dominate compact STEP0:
            // 2*2,359,296 + 2*196,608 + 512 = 5,112,320 B/core. The temporal
            // layer peak remains larger (6,881,280 B), before the shared
            // 386,048-B persistent region.
            const int64_t layout = A(cs0 * patch_dim_ * DWIDTH);
            step0 = {
                {"s0_layout_a", layout, 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_layout_b", layout, 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_patch", A(cs0 * (h / NUM_CORES) * DWIDTH), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_pos_emb", A(cs0 * s0_pos_width * DWIDTH), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_attn_in", 0, 1, 1, StorageClass::Temp, 0, "s0_layout_a", OUTSIDE},
                {"s0_bias", dma_safe(h / NUM_CORES), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
            };
        } else {
            const int64_t s0_io = A(cs0 * std::max(patch_dim_, h) * DWIDTH);
            step0 = {
                {"s0_pixel",   s0_io,                            1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_patch",   A(cs0 * (h / NUM_CORES) * DWIDTH), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_pos_emb", A(cs0 * s0_pos_width * DWIDTH),  1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_attn_in", 0, 1, 1, StorageClass::Temp, 0, "s0_pixel", OUTSIDE},
                {"s0_bias",    dma_safe(h / NUM_CORES), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
            };
        }
    }

    // MERGER (post_layers_fn) buffers — only when set_merger() opted in.
    //
    // OutsideLayerLoop scope (same as the STEP 0 buffers): the merger runs AFTER
    // the layer loop and hands off through DDR, so its SPM does not coexist with
    // the layer set → the group peak is max(layer, step0, merger), not a sum.
    //
    // Phase (2,2) — DISJOINT from the STEP 0 buffers' (1,1). Within
    // OutsideLayerLoop the allocator does a standard phase-overlap check, so the
    // STEP 0 and MERGER groups alias the same SPM region.
    // m_in dies before fc2 writes m_fc2; m_fc1 dies before fill writes m_zero.
    // Explicit aliases reuse those slots while preserving the three distinct
    // m_fc2/m_zero/m_out addresses required by all_reduce.
    // This aliasing relies on the phase windows staying disjoint; a resolved
    // chunk-size drop means STEP0 and MERGER stopped aliasing and are being summed.
    std::vector<BufferDecl> merger;
    if (has_merger_) {
        const int64_t csm      = QWEN3_5_VISION_MERGER_CHUNK;
        const int64_t mh       = merger_hidden_;                       // hidden*4
        const int64_t oh       = out_hidden_size_;
        const int64_t mrows    = csm / QWEN3_5_SPATIAL_MERGE_UNIT;     // merged rows/chunk
        const int64_t local_mh = mh / NUM_CORES;                       // fc1 col shard width
        const int64_t mfc2     = A(mrows * oh * DWIDTH);               // [mrows, out_hidden]
        const int64_t minput   = A(csm * h * DWIDTH);
        const int64_t mfc1     = A(mrows * local_mh * DWIDTH);
        merger = {
            {"m_in",   std::max(minput, mfc2),        2, 2, StorageClass::Temp, 0, nullptr, OUTSIDE},
            {"m_fc1",  std::max(mfc1, mfc2),          2, 2, StorageClass::Temp, 0, nullptr, OUTSIDE},
            {"m_fc2",  0,                             2, 2, StorageClass::Temp, 0, "m_in", OUTSIDE},
            {"m_zero", 0,                             2, 2, StorageClass::Temp, 0, "m_fc1", OUTSIDE},
            {"m_out",  mfc2,                         2, 2, StorageClass::Temp, 0, nullptr, OUTSIDE},

            // Persistent weight slots (populated by emit_preload_weights and
            // addressed via addr(0, name)). count 1, not per-layer.
            {"m_norm_g",   dma_safe(h),        0, 0, StorageClass::Persistent, 0, nullptr},
            {"m_norm_b",   dma_safe(h),        0, 0, StorageClass::Persistent, 0, nullptr},
            {"m_fc1_bias", dma_safe(local_mh), 0, 0, StorageClass::Persistent, 0, nullptr},
            {"m_fc2_bias", dma_safe(oh),       0, 0, StorageClass::Persistent, 0, nullptr},
        };
    }

    std::vector<BufferDecl> decls;
    if (current_temporal_num_frames_ > 1) {
        const int64_t temporal_rows =
            temporal_num_frames_ * G05_TEMPORAL_PATCH_GROUP;
        const int64_t temporal_stage_rows =
            temporal_num_frames_ * temporal_patches_per_frame_;
        const int64_t spatial_rows = G05_SPATIAL_GROUP_ROWS;
        SdpaConfig temporal_cfg{
            SdpaKernelType::FLASH_ATTN_SPM, hd,
            G05_TEMPORAL_PATCH_GROUP * nq,
            G05_TEMPORAL_PATCH_GROUP * nq,
            NUM_CORES, 1};
        SdpaConfig spatial_cfg{
            SdpaKernelType::FLASH_ATTN_SPM, hd, nq, nq, NUM_CORES, 0};
        const int64_t temporal_tmp = A(
            sdpa_compute_tmp_v16_size(temporal_cfg, temporal_num_frames_) * 32);
        const int64_t spatial_tmp = A(
            sdpa_compute_tmp_v16_size(spatial_cfg, spatial_rows) * 32);
        const int64_t temporal_camera =
            A(temporal_stage_rows * h * DWIDTH);
        const int64_t temporal_qkv = A(temporal_rows * local_q_dim * DWIDTH);
        const int64_t temporal_stage =
            A(temporal_stage_rows * local_q_dim * DWIDTH);
        const int64_t spatial_res = A(spatial_rows * h * DWIDTH);
        const int64_t spatial_qkv = A(spatial_rows * local_q_dim * DWIDTH);
        const int64_t spatial_fc1 = A(spatial_rows * local_inter * DWIDTH);
        decls = {
            // Coalesce one camera's frame-major DDR rows into one 3 MiB DMA,
            // then transpose once to patch-group-major SPM. Declaration order
            // is intentional: greedy first-fit places t_grouped below t_frame,
            // keeping the exact temporary peak at 6720 KiB.
            {"t_grouped",  temporal_camera, 1, 2, StorageClass::Temp, 0, nullptr, KVIN},
            {"t_frame",    temporal_camera, 1, 1, StorageClass::Temp, 0, nullptr, KVIN},
            {"t_q_stage",  temporal_stage,  2, 3, StorageClass::Temp, 0, nullptr, KVIN},
            {"t_k_stage",  temporal_stage,  2, 3, StorageClass::Temp, 0, nullptr, KVIN},
            {"t_sdpa_tmp", temporal_tmp,    2, 2, StorageClass::Temp, 0, nullptr, KVIN},
            {"t_out_stage",temporal_stage,  2, 3, StorageClass::Temp, 0, nullptr, KVIN},
            {"t_v",        temporal_qkv,    2, 2, StorageClass::Temp, 0, nullptr, KVIN},
            // Two three-frame spatial groups consume SPM slices directly.
            {"residual1", spatial_res, 4, 9, StorageClass::Temp, 0, nullptr, KVIN},
            {"input_norm",spatial_res, 4, 9, StorageClass::Temp, 0, nullptr, KVIN},
            // The upper 768 rows must survive the lower spatial group's o_proj
            // and MLP, so all three camera-wide buffers remain live through 9.
            {"q",         temporal_stage, 3, 9, StorageClass::Temp, 0, nullptr, KVIN},
            {"k",         temporal_stage, 3, 9, StorageClass::Temp, 0, nullptr, KVIN},
            {"v",         temporal_stage, 3, 9, StorageClass::Temp, 0, nullptr, KVIN},
            {"sdpa_out",  spatial_qkv, 6, 7, StorageClass::Temp, 0, nullptr, KVIN},
            {"sdpa_tmp",  spatial_tmp, 6, 6, StorageClass::Temp, 0, nullptr, KVIN},
            {"oproj",     spatial_res, 7, 9, StorageClass::Temp, 0, nullptr, KVIN},
            {"fc1",       spatial_fc1, 8, 9, StorageClass::Temp, 0, nullptr, KVIN},
        };
    } else {
        decls = {
            // ALL：Phase1 LN 读 residual1/写 input_norm；Phase2 残差读写两者。
            {"residual1",  res,      1, 6, StorageClass::Temp, 0, nullptr, ALL},
            {"input_norm", res,      1, 6, StorageClass::Temp, 0, nullptr, ALL},
            {"q",          qkv,      2, 3, StorageClass::Temp, 0, nullptr, KVIN},
            {"k",          qkv,      2, 3, StorageClass::Temp, 0, nullptr, KVIN},
            {"v",          qkv,      2, 3, StorageClass::Temp, 0, nullptr, KVIN},
            {"q_comp",     qkv,      4, 4, StorageClass::Temp, 0, nullptr, COMP},
            {"oproj",      res,      4, 6, StorageClass::Temp, 0, nullptr, COMP},
            {"sdpa_out",   qkv,      4, 5, StorageClass::Temp, 0, nullptr, COMP},
            {"sdpa_tmp",   sdpa_tmp, 4, 4, StorageClass::Temp, 0, nullptr, COMP},
            {"fc1",        fc1,      5, 6, StorageClass::Temp, 0, nullptr, COMP},
        };
    }

    std::vector<BufferDecl> persistent = {
        {"ln1_gamma",    norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        {"ln1_beta",     norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        {"ln2_gamma",    norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        {"ln2_beta",     norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        {"q_bias",       q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        {"k_bias",       q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        {"v_bias",       q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        {"fc1_bias",     fc1_bias_sz,  0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        {"o_bias",       full_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        {"fc2_bias",     full_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr},

        // 2D-RoPE freq tables in SPM (Persistent, model-static — count 0, loaded
        // once in emit_preload_weights). Broadcast to all cores so each core reads
        // the full [max_hw, head_dim/4] table; addressed via addr(0, name). This
        // replaces the DDR variant's per-block re-reads of the freq tables.
        {"freq_cos_spm", rope_tbl_sz,  0, 0, StorageClass::Persistent, 0, nullptr},
        {"freq_sin_spm", rope_tbl_sz,  0, 0, StorageClass::Persistent, 0, nullptr},
    };
    if (has_temporal_) {
        persistent.push_back(
            {"temporal_pe", dma_safe(temporal_num_frames_ * h),
             0, 0, StorageClass::Persistent, 0, nullptr});
    }

    decls.insert(decls.end(), persistent.begin(), persistent.end());
    decls.insert(decls.end(), step0.begin(), step0.end());
    decls.insert(decls.end(), merger.begin(), merger.end());
    return decls;
}

// ─────────────────────────────────────────────────────────────────────────────
// GRAPH-PLAN HOOKS
// ─────────────────────────────────────────────────────────────────────────────

// ── static_config: single group + preload hook + KV_FIRST two-phase hooks. ──
ModelStaticConfig Qwen3_5VisionModel::static_config() {
    ModelStaticConfig cfg;
    // Graph admission is handled by the Python graph scope in
    // adapters/qwen3_5/vision.py; this method only describes op emission.
    cfg.num_layers       = num_layers();

    cfg.preload_fn = static_cast<void(FusedModelBase::*)()>(
                         &Qwen3_5VisionModel::emit_preload_weights);

    // Single group: bidirectional vision encoder, same constraint as SigLIP
    // (multi-group REPLAY non-determinism mitigation, see siglip notes).
    cfg.cross_layer_batch_size = num_layers();
    cfg.fast_replay_skip_layer_loop = has_temporal_;
    cfg.fast_replay_skip_preload = has_temporal_;

    // KV_FIRST 双向分块两相（照 rpu_gemma_model.cpp）：Phase1 把全量 KV 插满 + 存 rope 后 Q，
    // Phase2 逐 query chunk 对全量 KV 做 SDPA + MLP。
    cfg.kv_first_fn = static_cast<void(FusedModelBase::*)(int, const ChunkInfo&)>(
                         &Qwen3_5VisionModel::emit_kv_first_body);
    cfg.kv_first_chunk_plan_fn =
        static_cast<ChunkPlan(FusedModelBase::*)(const ChunkPlan&)>(
            &Qwen3_5VisionModel::plan_kv_first_chunks);

    // STEP 0 (patch_embed + uploaded HF-exact positions) as pre_layers_fn —
    // once per forward, before the layer loop, in the same graph scope. Only
    // when set_patch_embed() opted in; otherwise Python does the full CPU path.
    if (has_step0_) {
        cfg.pre_layers_fn = static_cast<void(FusedModelBase::*)()>(
                                &Qwen3_5VisionModel::emit_step0);
    }

    // MERGER (patch merger) as post_layers_fn — once per forward, AFTER the layer
    // loop, in the same graph scope. Reads the tower's full DDR output back in
    // chunks and projects each 2×2 patch block to the text hidden size. Only when
    // set_merger() opted in; the tower's forward return is unchanged either way.
    if (has_merger_) {
        cfg.post_layers_fn = static_cast<void(FusedModelBase::*)()>(
                                 &Qwen3_5VisionModel::emit_merger);
    }
    return cfg;
}

// ── dynamic_config: KV_FIRST chunk mode + AUTO inter-layer IO. ──
ModelDynamicConfig Qwen3_5VisionModel::dynamic_config(const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode     = ChunkMode::KV_FIRST;    // 双向分块两相
    cfg.inter_layer_io = current_temporal_num_frames_ > 1
        ? InterLayerIO::DDR_PINGPONG
        : InterLayerIO::AUTO;
    return cfg;
}

// ── plan_kv_first_chunks: KV-insert 相分块计划（v1：与 compute 用同一 cs）。──
ChunkPlan Qwen3_5VisionModel::plan_kv_first_chunks(const ChunkPlan& compute_plan) {
    return compute_plan;
}

// Cold per-handle cap. Same contract as
// Qwen3_5Model::set_chunk_size_cap: it keys the plan the GraphCache entries are
// built from, so a live handle must not be hot-switched.
void Qwen3_5VisionModel::set_chunk_size_cap(int64_t cap) {
    TORCH_CHECK(cap >= 0, "Qwen3.5 vision chunk cap must be >= 0 (0 = no cap), got ", cap);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 vision chunk cap must be set before the first forward");
    chunk_size_cap_ = cap;
    invalidate_model_state();
}

// SPM feasibility is already checked by the exact first-fit planner. Keep only
// the SDPA kernel constraint and an optional explicit cap for controlled tests.
//
// The cap is latched ONCE — changing the plan under an already-built same-shape
// GraphCache entry would make its replay op stream inconsistent with it.
// Keep the latch per handle: a function-local `static` would let the first
// forward's environment pin every later handle in the process.
bool Qwen3_5VisionModel::subclass_chunk_size_valid(int64_t cs, int64_t seq, int64_t pos) const {
    if (current_temporal_num_frames_ > 1) return true;
    // 真·kernel 约束，任何时候都不能摘。
    SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                        head_dim(), num_q_heads(), num_q_heads(),
                        NUM_CORES, /*mask=*/0};
    if (!sdpa_is_valid_chunk_size(sdpa_cfg, cs, seq, pos)) return false;

    if (chunk_size_cap_ < 0) {   // not latched yet → take this handle's default
        const char* e = std::getenv("QWEN3_5_VISION_CHUNK");
        chunk_size_cap_ = e ? static_cast<int64_t>(std::atoll(e)) : 0;
    }
    if (chunk_size_cap_ > 0 && cs > chunk_size_cap_) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────────────────────

// ── emit_preload_weights: per-layer norm + col/row-partition bias DMA into
//    PersistentPerLayer SPM slots. BUILD records this preload prefix; retained
//    graph REPLAY executes it again. ──────────────────────────────────────────
void Qwen3_5VisionModel::emit_preload_weights() {
    int64_t h = hidden_size();
    int64_t local_q_dim = (num_q_heads() / NUM_CORES) * head_dim();
    int64_t local_inter = intermediate_size() / NUM_CORES;

    // Loop 1: 8-core DMAs for all layers (zero + norm + col-partition bias).
    for (int64_t L = 0; L < num_layers(); ++L) {
        auto& bn = layer_bias_norm_[L];

        // Zero the row-partition bias slots with the graph-aware `fill`, exactly
        // as build_gdn does. Zero all 8 cores so the 1-core bias load
        // below leaves cores 1..7 at 0 for the row-partition all_reduce.
        rpu_launch_fill_spm_kernel(layer_addr(L, 0, "o_bias"),   h, c10::Half(0.0f), NUM_CORES);
        rpu_launch_fill_spm_kernel(layer_addr(L, 0, "fc2_bias"), h, c10::Half(0.0f), NUM_CORES);

        rpu_launch_ddr_broadcast_spm_dma(
            bn.ln1_w.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "ln1_gamma"));
        rpu_launch_ddr_broadcast_spm_dma(
            bn.ln1_b.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "ln1_beta"));
        rpu_launch_ddr_broadcast_spm_dma(
            bn.ln2_w.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "ln2_gamma"));
        rpu_launch_ddr_broadcast_spm_dma(
            bn.ln2_b.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "ln2_beta"));

        rpu_launch_ddr_scatter_spm_dma(
            bn.q_b.data_ptr<c10::Half>(),
            local_q_dim, local_q_dim * DWIDTH,
            layer_addr(L, 0, "q_bias"),
            /*num_cores=*/NUM_CORES);
        rpu_launch_ddr_scatter_spm_dma(
            bn.k_b.data_ptr<c10::Half>(),
            local_q_dim, local_q_dim * DWIDTH,
            layer_addr(L, 0, "k_bias"),
            /*num_cores=*/NUM_CORES);
        rpu_launch_ddr_scatter_spm_dma(
            bn.v_b.data_ptr<c10::Half>(),
            local_q_dim, local_q_dim * DWIDTH,
            layer_addr(L, 0, "v_bias"),
            /*num_cores=*/NUM_CORES);
        rpu_launch_ddr_scatter_spm_dma(
            bn.fc1_b.data_ptr<c10::Half>(),
            local_inter, local_inter * DWIDTH,
            layer_addr(L, 0, "fc1_bias"),
            /*num_cores=*/NUM_CORES);
    }

    // Loop 2: 1-core DMAs for all layers (row-partition biases).
    for (int64_t L = 0; L < num_layers(); ++L) {
        auto& bn = layer_bias_norm_[L];
        rpu_launch_ddr_broadcast_spm_dma(
            bn.o_b.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "o_bias"), /*num_cores=*/1);
        rpu_launch_ddr_broadcast_spm_dma(
            bn.fc2_b.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "fc2_bias"), /*num_cores=*/1);
    }

    // 2D-RoPE freq tables → SPM. Broadcast so every core holds the full
    // [max_hw, head_dim/4] table; the SPM rope launcher (emit_kv_first_body) reads
    // them from here instead of re-reading DDR each block. freq_cos_/freq_sin_
    // stay as the DDR source (registered by set_rope_tables, which runs before
    // this preload — forward() TORCH_CHECKs has_rope_).
    TORCH_CHECK(freq_cos_.defined() && freq_sin_.defined(),
                "emit_preload_weights: rope tables not set (call set_rope_tables first)");
    rpu_launch_ddr_broadcast_spm_dma(
        freq_cos_.data_ptr<c10::Half>(), freq_cos_.numel(), addr(0, "freq_cos_spm"));
    rpu_launch_ddr_broadcast_spm_dma(
        freq_sin_.data_ptr<c10::Half>(), freq_sin_.numel(), addr(0, "freq_sin_spm"));

    if (current_temporal_num_frames_ > 1) {
        rpu_launch_ddr_broadcast_spm_dma(
            temporal_pe_.data_ptr<c10::Half>(), temporal_pe_.numel(),
            addr(0, "temporal_pe"));
    }

    // MERGER weights (only when set_merger() opted in). Same three bias/norm
    // templates as the per-layer loops above, into the count-1 Persistent slots.
    if (has_merger_) {
        const int64_t mh       = merger_hidden_;          // hidden*4 = fc1 N
        const int64_t oh       = out_hidden_size_;         // fc2 N
        const int64_t local_mh = mh / NUM_CORES;           // fc1 col shard width

        // norm gamma/beta — broadcast to all 8 cores (LN reads full width).
        rpu_launch_ddr_broadcast_spm_dma(
            merger_norm_w_.data_ptr<c10::Half>(), h, addr(0, "m_norm_g"));
        rpu_launch_ddr_broadcast_spm_dma(
            merger_norm_b_.data_ptr<c10::Half>(), h, addr(0, "m_norm_b"));

        // fc1 bias — COL-scatter: core i gets its own local_mh (=512) slice
        // (col-parallel layout, same as q/k/v/fc1 bias above).
        rpu_launch_ddr_scatter_spm_dma(
            merger_fc1_b_.data_ptr<c10::Half>(),
            local_mh, local_mh * DWIDTH,
            addr(0, "m_fc1_bias"), /*num_cores=*/NUM_CORES);

        // fc2 bias — ROW convention: zero all 8 cores with fill, then
        // load the real bias into core 0 ONLY. The row all_reduce sums the
        // partials, so bias must be present on exactly one core → counted once.
        rpu_launch_fill_spm_kernel(addr(0, "m_fc2_bias"), oh, c10::Half(0.0f), NUM_CORES);
        rpu_launch_ddr_broadcast_spm_dma(
            merger_fc2_b_.data_ptr<c10::Half>(), oh,
            addr(0, "m_fc2_bias"), /*num_cores=*/1);
    }
}

// ── set_weights: per-layer weight + bias storage (see header for the QKV-bias
//    split contract). Vision encoder is MHA (num_kv_heads == num_q_heads). ─────
void Qwen3_5VisionModel::set_weights(
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
    double eps)
{
    int64_t N = static_cast<int64_t>(q_w_list.size());
    TORCH_CHECK(N > 0, "qwen3_5_vision_set_weights: empty weight lists");
    TORCH_CHECK(num_heads > 0 && head_dim > 0 && hidden_size > 0
                && intermediate_size > 0,
                "qwen3_5_vision_set_weights: dim params must be positive");

    auto check_list = [&](const at::TensorList& l, const char* n) {
        TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                    "qwen3_5_vision_set_weights: ", n, ".size()=", l.size(),
                    " != num_layers=", N);
    };
    check_list(k_w_list,   "k_w_list");
    check_list(v_w_list,   "v_w_list");
    check_list(o_w_list,   "o_w_list");
    check_list(fc1_w_list, "fc1_w_list");
    check_list(fc2_w_list, "fc2_w_list");
    check_list(ln1_w_list, "ln1_w_list");
    check_list(ln1_b_list, "ln1_b_list");
    check_list(ln2_w_list, "ln2_w_list");
    check_list(ln2_b_list, "ln2_b_list");
    check_list(q_b_list,   "q_b_list");
    check_list(k_b_list,   "k_b_list");
    check_list(v_b_list,   "v_b_list");
    check_list(o_b_list,   "o_b_list");
    check_list(fc1_b_list, "fc1_b_list");
    check_list(fc2_b_list, "fc2_b_list");

    auto check_rank = [&](const at::TensorList& list, const char* name, int64_t r) {
        for (int64_t i = 0; i < N; i++) {
            TORCH_CHECK(list[i].defined(),
                        "qwen3_5_vision_set_weights: ", name, "[", i, "] undefined");
            TORCH_CHECK(list[i].dim() == r,
                        "qwen3_5_vision_set_weights: ", name, "[", i, "] must be ",
                        r, "D, got ", list[i].dim(), "D");
        }
    };
    check_rank(q_w_list,   "q_w_list",   2);
    check_rank(k_w_list,   "k_w_list",   2);
    check_rank(v_w_list,   "v_w_list",   2);
    check_rank(o_w_list,   "o_w_list",   2);
    check_rank(fc1_w_list, "fc1_w_list", 2);
    check_rank(fc2_w_list, "fc2_w_list", 2);
    check_rank(ln1_w_list, "ln1_w_list", 1);
    check_rank(ln1_b_list, "ln1_b_list", 1);
    check_rank(ln2_w_list, "ln2_w_list", 1);
    check_rank(ln2_b_list, "ln2_b_list", 1);
    check_rank(q_b_list,   "q_b_list",   1);
    check_rank(k_b_list,   "k_b_list",   1);
    check_rank(v_b_list,   "v_b_list",   1);
    check_rank(o_b_list,   "o_b_list",   1);
    check_rank(fc1_b_list, "fc1_b_list", 1);
    check_rank(fc2_b_list, "fc2_b_list", 1);

    // Vision encoder is MHA (no GQA): num_kv_heads == num_q_heads.
    set_model_params(num_heads, num_heads, head_dim,
                     hidden_size, intermediate_size);
    set_num_layers(N);
    eps_ = eps;

    // head_dim is exact for Qwen3.5 (64 — no SigLIP-style 72→80 pad).
    orig_head_dim_ = head_dim;

    layer_weights_.clear();
    layer_weights_.reserve(N);
    for (int64_t i = 0; i < N; i++) {
        layer_weights_.push_back({
            q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
            fc1_w_list[i], fc2_w_list[i],
        });
    }

    layer_bias_norm_.clear();
    layer_bias_norm_.reserve(N);
    for (int64_t i = 0; i < N; i++) {
        layer_bias_norm_.push_back({
            ln1_w_list[i], ln1_b_list[i],
            ln2_w_list[i], ln2_b_list[i],
            q_b_list[i], k_b_list[i], v_b_list[i], o_b_list[i],
            fc1_b_list[i], fc2_b_list[i],
        });
    }

    invalidate_model_state();
}

// ── set_rope_tables: register FreqCos / FreqSin (validated fp16 RPU tables) +
//    lazy-alloc the position_idx keepalive. ─────────────────────────────────────
void Qwen3_5VisionModel::set_rope_tables(const at::Tensor& freq_cos, const at::Tensor& freq_sin)
{
    TORCH_CHECK(freq_cos.defined() && freq_sin.defined(),
                "qwen3_5_vision_set_rope: freq_cos / freq_sin must be defined");
    TORCH_CHECK(freq_cos.dim() == 2 && freq_sin.dim() == 2,
                "qwen3_5_vision_set_rope: freq_cos/sin must be 2D, got ",
                freq_cos.dim(), "/", freq_sin.dim(), "D");
    TORCH_CHECK(freq_cos.sizes() == freq_sin.sizes(),
                "qwen3_5_vision_set_rope: freq_cos / freq_sin shape mismatch");
    TORCH_CHECK(freq_cos.scalar_type() == at::kHalf && freq_sin.scalar_type() == at::kHalf,
                "qwen3_5_vision_set_rope: freq_cos / freq_sin must be fp16");
    TORCH_CHECK(freq_cos.device().type() == at::kPrivateUse1
                && freq_sin.device().type() == at::kPrivateUse1,
                "qwen3_5_vision_set_rope: freq_cos / freq_sin must be on RPU device");
    TORCH_CHECK(freq_cos.is_contiguous() && freq_sin.is_contiguous(),
                "qwen3_5_vision_set_rope: freq_cos / freq_sin must be contiguous");

    freq_cos_ = freq_cos;
    freq_sin_ = freq_sin;
    max_hw_ = freq_cos.size(0);

    // Lazy-alloc on first call; reuse on subsequent calls (sizes never change).
    // rope_2d packs the DDR address as addr/256, while the RPU allocator only
    // guarantees 32B. Mirror the adapter's overallocate+narrow alignment scheme.
    if (!position_idx_keepalive_.defined()) {
        auto storage = at::empty(
            {QWEN3_5_VISION_MAX_KEEPALIVE_SEQ * 2 + 128},
            at::TensorOptions().dtype(at::kShort).device(at::kPrivateUse1));
        const uint64_t raw = RpuGetDevAddr(storage.data_ptr<int16_t>());
        const int64_t off_bytes =
            (256 - static_cast<int64_t>(raw & 0xFFu)) & 0xFF;
        TORCH_CHECK(off_bytes % static_cast<int64_t>(sizeof(int16_t)) == 0,
                    "qwen3_5_vision_set_rope: cannot 256B-align position_idx keepalive");
        position_idx_keepalive_ = storage
            .narrow(0, off_bytes / static_cast<int64_t>(sizeof(int16_t)),
                    QWEN3_5_VISION_MAX_KEEPALIVE_SEQ * 2)
            .view({QWEN3_5_VISION_MAX_KEEPALIVE_SEQ, 2});
        position_idx_keepalive_.zero_();
        TORCH_CHECK(
            (RpuGetDevAddr(position_idx_keepalive_.data_ptr<int16_t>()) & 0xFFu) == 0,
            "qwen3_5_vision_set_rope: position_idx keepalive landed unaligned");
    }

    has_rope_ = true;
    invalidate_model_state();
}

// ── set_patch_embed: opt into in-graph STEP 0. See the header for the contract.
void Qwen3_5VisionModel::set_patch_embed(
    const at::Tensor& pe_w, const at::Tensor& pe_b)
{
    TORCH_CHECK(pe_w.defined() && pe_b.defined(),
                "qwen3_5_vision_set_patch_embed: pe_w/pe_b must be defined");
    TORCH_CHECK(pe_w.dim() == 2 && pe_b.dim() == 1,
                "qwen3_5_vision_set_patch_embed: pe_w must be 2D, pe_b 1D");
    TORCH_CHECK(pe_w.scalar_type() == at::kHalf && pe_b.scalar_type() == at::kHalf,
                "qwen3_5_vision_set_patch_embed: pe_w/pe_b must be fp16");
    TORCH_CHECK(pe_w.device().type() == at::kPrivateUse1
                && pe_b.device().type() == at::kPrivateUse1,
                "qwen3_5_vision_set_patch_embed: pe_w/pe_b must be on RPU device");
    TORCH_CHECK(pe_w.is_contiguous() && pe_b.is_contiguous(),
                "qwen3_5_vision_set_patch_embed: pe_w/pe_b must be contiguous");
    TORCH_CHECK(hidden_size() > 0,
                "qwen3_5_vision_set_patch_embed: call set_weights() first (needs hidden_size)");

    const int64_t h = hidden_size();
    TORCH_CHECK(pe_w.size(0) == h && pe_b.size(0) == h,
                "qwen3_5_vision_set_patch_embed: pe_w.size(0)=", pe_w.size(0),
                " pe_b.size(0)=", pe_b.size(0), " must both equal hidden=", h);
    const int64_t pd = pe_w.size(1);
    // Col-parallel swizzle constraints for partition=1.
    // ⚠️ N=1024 also satisfies the ROW constraint (N%16), so a wrong `partition`
    // at swizzle time is NOT caught here — the only symptom is a wrong result.
    TORCH_CHECK(h % (16 * NUM_CORES) == 0,
                "qwen3_5_vision_set_patch_embed: hidden=", h,
                " must be divisible by 16*tp=", 16 * NUM_CORES, " (col swizzle)");
    TORCH_CHECK(pd % 16 == 0,
                "qwen3_5_vision_set_patch_embed: patch_dim=", pd,
                " must be divisible by 16 (fp16 col swizzle)");

    pe_w_ = pe_w;
    pe_b_ = pe_b;
    patch_dim_ = pd;

    has_step0_ = true;
    invalidate_model_state();
}

// ── set_merger: opt into the in-graph patch merger. See the header for the
//    contract. Weights are stored SWIZZLED-AS-IS (Python pre-swizzles; the linear
//    launcher does NOT re-order the weight — verified in rpu_linear.cpp, it reads
//    weight.data_ptr directly), exactly like set_weights / set_patch_embed. ──────
void Qwen3_5VisionModel::set_merger(
    const at::Tensor& fc1_w, const at::Tensor& fc1_b,
    const at::Tensor& fc2_w, const at::Tensor& fc2_b,
    const at::Tensor& norm_w, const at::Tensor& norm_b,
    int64_t out_hidden_size)
{
    TORCH_CHECK(fc1_w.defined() && fc1_b.defined() && fc2_w.defined()
                && fc2_b.defined() && norm_w.defined() && norm_b.defined(),
                "qwen3_5_vision_set_merger: all tensors must be defined");
    TORCH_CHECK(hidden_size() > 0,
                "qwen3_5_vision_set_merger: call set_weights() first (needs hidden_size)");
    TORCH_CHECK(out_hidden_size > 0,
                "qwen3_5_vision_set_merger: out_hidden_size must be positive");

    auto check = [](const at::Tensor& t, const char* n, int64_t rank) {
        TORCH_CHECK(t.scalar_type() == at::kHalf,
                    "qwen3_5_vision_set_merger: ", n, " must be fp16");
        TORCH_CHECK(t.device().type() == at::kPrivateUse1,
                    "qwen3_5_vision_set_merger: ", n, " must be on RPU device");
        TORCH_CHECK(t.is_contiguous(),
                    "qwen3_5_vision_set_merger: ", n, " must be contiguous");
        TORCH_CHECK(t.dim() == rank,
                    "qwen3_5_vision_set_merger: ", n, " must be ", rank,
                    "D, got ", t.dim(), "D");
    };
    check(fc1_w, "fc1_w", 2);   check(fc2_w, "fc2_w", 2);
    check(fc1_b, "fc1_b", 1);   check(fc2_b, "fc2_b", 1);
    check(norm_w, "norm_w", 1); check(norm_b, "norm_b", 1);

    const int64_t h  = hidden_size();
    const int64_t mh = h * QWEN3_5_SPATIAL_MERGE_UNIT;   // 4096 — merger context dim
    // Stored swizzled, but .size() still reads the LOGICAL [out, in] shape, so
    // validate against that.
    TORCH_CHECK(fc1_w.size(0) == mh && fc1_w.size(1) == mh,
                "qwen3_5_vision_set_merger: fc1_w must be [hidden*4, hidden*4]=[",
                mh, ",", mh, "], got [", fc1_w.size(0), ",", fc1_w.size(1), "]");
    TORCH_CHECK(fc1_b.size(0) == mh,
                "qwen3_5_vision_set_merger: fc1_b must be [hidden*4]=", mh,
                ", got ", fc1_b.size(0));
    TORCH_CHECK(fc2_w.size(0) == out_hidden_size && fc2_w.size(1) == mh,
                "qwen3_5_vision_set_merger: fc2_w must be [out_hidden, hidden*4]=[",
                out_hidden_size, ",", mh, "], got [",
                fc2_w.size(0), ",", fc2_w.size(1), "]");
    TORCH_CHECK(fc2_b.size(0) == out_hidden_size,
                "qwen3_5_vision_set_merger: fc2_b must be [out_hidden]=", out_hidden_size,
                ", got ", fc2_b.size(0));
    TORCH_CHECK(norm_w.size(0) == h && norm_b.size(0) == h,
                "qwen3_5_vision_set_merger: norm_w/b must be [hidden]=", h);

    // ⚠️ COL/ROW swizzle role is NOT checked here: fc1 (N=mh) and fc2 (N=out_hidden,
    // K=mh) satisfy BOTH the col (N%(16*tp)) and row (K%16) swizzle constraints, so
    // a wrong `partition` picked at Python swizzle time is NOT caught — the only
    // symptom is a wrong result. fc1 MUST be col-swizzled (partition=1), fc2 MUST
    // be row-swizzled (partition=0). See emit_merger. (Same hazard as the
    // set_patch_embed col-swizzle note.)
    TORCH_CHECK(mh % (16 * NUM_CORES) == 0,
                "qwen3_5_vision_set_merger: hidden*4=", mh,
                " must be divisible by 16*tp=", 16 * NUM_CORES,
                " (fc1 col swizzle N / fc2 row K-split)");
    TORCH_CHECK(out_hidden_size % 16 == 0,
                "qwen3_5_vision_set_merger: out_hidden=", out_hidden_size,
                " must be divisible by 16 (fc2 row swizzle N)");

    merger_fc1_w_  = fc1_w;   merger_fc1_b_  = fc1_b;
    merger_fc2_w_  = fc2_w;   merger_fc2_b_  = fc2_b;
    merger_norm_w_ = norm_w;  merger_norm_b_ = norm_b;
    out_hidden_size_ = out_hidden_size;
    merger_hidden_   = mh;
    has_merger_      = true;

    invalidate_model_state();
}

void Qwen3_5VisionModel::set_temporal(
    const at::Tensor& temporal_pe,
    int64_t num_frames,
    int64_t patches_per_frame)
{
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "qwen3_5_vision_set_temporal must run before the first forward");
    TORCH_CHECK(num_layers() == 24 && hidden_size() == 1024 &&
                num_q_heads() == 16 && head_dim() == 64,
                "qwen3_5_vision_set_temporal requires the canonical "
                "24-layer 1024/16/64 tower");
    TORCH_CHECK(num_frames == 6 && patches_per_frame == 256,
                "qwen3_5_vision_set_temporal supports only K=6, patches/frame=256");
    TORCH_CHECK(temporal_pe.device().type() == at::kPrivateUse1 &&
                temporal_pe.scalar_type() == at::kHalf && temporal_pe.is_contiguous(),
                "qwen3_5_vision_set_temporal: temporal_pe must be contiguous fp16 RPU");
    TORCH_CHECK(temporal_pe.dim() == 2 && temporal_pe.size(0) == num_frames &&
                temporal_pe.size(1) == hidden_size(),
                "qwen3_5_vision_set_temporal: temporal_pe must be [6,1024], got ",
                temporal_pe.sizes());

    temporal_pe_ = temporal_pe;
    temporal_num_frames_ = num_frames;
    temporal_patches_per_frame_ = patches_per_frame;
    has_temporal_ = true;
    invalidate_model_state();
}

void Qwen3_5VisionModel::set_allreduce_force_non_v2(bool enabled)
{
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "qwen3_5_vision_set_allreduce_force_non_v2 must run before "
                "the first forward");
    allreduce_force_non_v2_ = enabled;
    invalidate_model_state();
}

}  // namespace v3

using Qwen3_5VisionRegistry = ModelHandleRegistry<v3::Qwen3_5VisionModel>;

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers
// =============================================================================

int64_t rpu_qwen3_5_vision_create() {
    return Qwen3_5VisionRegistry::create();
}

void rpu_qwen3_5_vision_destroy(int64_t handle) {
    v3::qwen3_5_z2_internal::check_vision_destroy_allowed(handle);
    Qwen3_5VisionRegistry::destroy(handle, "rpu_qwen3_5_vision_destroy");
}

void rpu_qwen3_5_vision_set_weights(
    int64_t handle,
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
    double eps)
{
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        fc1_w_list, fc2_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, k_b_list, v_b_list, o_b_list,
        fc1_b_list, fc2_b_list,
        num_heads, head_dim, hidden_size, intermediate_size,
        eps);
}

void rpu_qwen3_5_vision_set_rope(
    int64_t handle,
    const at::Tensor& freq_cos,
    const at::Tensor& freq_sin)
{
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision")
        ->set_rope_tables(freq_cos, freq_sin);
}

void rpu_qwen3_5_vision_set_patch_embed(
    int64_t handle,
    const at::Tensor& pe_w,
    const at::Tensor& pe_b)
{
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision")
        ->set_patch_embed(pe_w, pe_b);
}

at::Tensor rpu_qwen3_5_vision_position_idx_keepalive(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision")
        ->position_idx_keepalive();
}

void rpu_qwen3_5_vision_set_merger(
    int64_t handle,
    const at::Tensor& fc1_w, const at::Tensor& fc1_b,
    const at::Tensor& fc2_w, const at::Tensor& fc2_b,
    const at::Tensor& norm_w, const at::Tensor& norm_b,
    int64_t out_hidden_size)
{
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision")
        ->set_merger(fc1_w, fc1_b, fc2_w, fc2_b, norm_w, norm_b, out_hidden_size);
}

void rpu_qwen3_5_vision_set_temporal(
    int64_t handle,
    const at::Tensor& temporal_pe,
    int64_t num_frames,
    int64_t patches_per_frame)
{
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_set_temporal")
        ->set_temporal(temporal_pe, num_frames, patches_per_frame);
}

void rpu_qwen3_5_vision_set_allreduce_force_non_v2(
    int64_t handle, bool enabled)
{
    Qwen3_5VisionRegistry::get(
        handle, "rpu_qwen3_5_vision_set_allreduce_force_non_v2")
        ->set_allreduce_force_non_v2(enabled);
}

bool rpu_qwen3_5_vision_get_allreduce_force_non_v2(int64_t handle)
{
    return Qwen3_5VisionRegistry::get(
        handle, "rpu_qwen3_5_vision_get_allreduce_force_non_v2")
        ->get_allreduce_force_non_v2();
}

at::Tensor rpu_qwen3_5_vision_get_merger_out(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_get_merger_out")
        ->merger_out();
}

at::Tensor rpu_qwen3_5_vision_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t num_patches,
    const std::optional<at::Tensor>& step0_pos,
    const std::optional<at::Tensor>& fusion_target,
    at::IntArrayRef fusion_row_starts,
    int64_t temporal_num_frames,
    int64_t camera_batch_count)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision")
        ->forward(input, k_caches, v_caches, num_patches, step0_pos,
                  fusion_target, fusion_row_starts, temporal_num_frames,
                  camera_batch_count);
}

namespace v3::qwen3_5_z2_internal {

SpmPipelineComponentLayout prepare_vision(int64_t handle,
                                          int64_t num_patches) {
    return Qwen3_5VisionRegistry::get(handle, "qwen3_5_spm_z2_prepare_vision")
        ->prepare_z2_layout(num_patches);
}

SpmDense2DSpec vision_produced_spec(int64_t handle,
                                    int64_t num_patches) {
    return Qwen3_5VisionRegistry::get(handle, "qwen3_5_spm_z2_vision_spec")
        ->z2_produced_spec(num_patches);
}

void adopt_vision(int64_t handle,
                  const SpmPipelineLease& lease,
                  const SpmTensorView& scratch) {
    Qwen3_5VisionRegistry::get(handle, "qwen3_5_spm_z2_adopt_vision")
        ->adopt_z2_layout(lease, scratch);
}

void bind_vision_slice(int64_t handle,
                       const SpmPipelineLease& lease,
                       const SpmPortView& slice) {
    Qwen3_5VisionRegistry::get(handle, "qwen3_5_spm_z2_bind_vision")
        ->bind_z2_slice(lease, slice);
}

void validate_vision(int64_t handle,
                     const SpmPipelineLease& lease) {
    Qwen3_5VisionRegistry::get(handle, "qwen3_5_spm_z2_validate_vision")
        ->validate_z2_layout(lease);
}

void clear_vision(int64_t handle,
                  uint64_t epoch,
                  uint64_t plan_hash) {
    Qwen3_5VisionRegistry::get(handle, "qwen3_5_spm_z2_clear_vision")
        ->clear_z2_layout(epoch, plan_hash);
}

void forward_vision_z2(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches,
    at::TensorList v_caches,
    int64_t num_patches,
    const std::optional<at::Tensor>& step0_pos,
    uint64_t epoch,
    uint64_t plan_hash) {
    validate_vision_dispatch(handle, epoch, plan_hash);
    std::vector<at::Tensor> k_cache_vec(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> v_cache_vec(v_caches.begin(), v_caches.end());
    Qwen3_5VisionRegistry::get(handle, "qwen3_5_spm_z2_forward_vision")
        ->forward_z2(input, k_cache_vec, v_cache_vec, num_patches,
                     step0_pos, epoch, plan_hash);
}

}  // namespace v3::qwen3_5_z2_internal

// Per-handle SPM-budget-resolved chunk_size (0 before any forward), matching
// rpu_qwen3_5_get_resolved_chunk_size.
int64_t rpu_qwen3_5_vision_get_resolved_chunk_size(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

// Cold per-handle vision chunk cap; each handle owns its value.
void rpu_qwen3_5_vision_set_chunk_size_cap(int64_t handle, int64_t cap) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_set_chunk_size_cap")
        ->set_chunk_size_cap(cap);
}

// Per-layer debug snapshots (empty tensor until a get_debug_export() forward runs).
// dbg_hidden = [num_layers, N, hidden]; dbg_q = [num_layers, NUM_CORES, N, local_q_dim].
at::Tensor rpu_qwen3_5_vision_get_dbg_hidden(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_get_dbg_hidden")->dbg_hidden();
}
at::Tensor rpu_qwen3_5_vision_get_dbg_q(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_get_dbg_q")->dbg_q();
}
