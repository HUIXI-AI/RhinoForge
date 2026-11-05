// rpu_qwen3_5_vision_model.cpp — Qwen3.5 Vision Encoder (Option 2, deepstack
// removed). See rpu_qwen3_5_vision_model.h for architecture/scope. A profile-sized
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
#include "rpu_pi05_owner_norm.h"
#include "rpu_kernel_cache.h"
#include "rpu_linear_tiling.h"

#include "graph/graph_runtime.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <cstdlib>   // std::getenv — debug-only Q capture
#include <cmath>
#include <algorithm>  // std::min — emit_step0 row chunking
#include <limits>
#include <numeric>    // std::gcd — attention TP resolver

using namespace at;
using namespace ::rhino_lkn;

#define DWIDTH 2

constexpr int64_t G05_MAX_CAMERA_BATCH = 3;
// Temporal execution has a fixed semantic shape, independent of spatial KV
// capacity. Reserve its position rows before Python retains the fixed buffer.
constexpr int64_t G05_TEMPORAL_MAX_SEQ = G05_MAX_CAMERA_BATCH * 6 * 256;
constexpr int64_t QWEN3_5_VISION_SDPA_MAX_KV_ROWS = int64_t{UINT16_MAX} * 16;
constexpr int64_t G05_VISION_HIGH_ENCODER_ACTIVATION_SITE = 3127033570831610403LL;
constexpr int64_t G05_VISION_HIGH_MERGER_ACTIVATION_SITE = 7199989347316866274LL;
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
// would want 2640*1536*2 = 7.7MB of the 8MB SPM for s0_pixel alone. rhino chunks
// it the same way (qwen3_vl_vision_simc_graph.cc:472 passes
// conv3d_chunk_id*conv3d_chunk_size as pos_offset).
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
// 128 was the old value, chosen only because these buffers used to be LayerWide
// and so were subtracted from the layer budget (measured: it cost 66 rows of
// layer chunk, 960 → 894). With OutsideLayerLoop that coupling is gone.
constexpr int64_t QWEN3_5_VISION_STEP0_CHUNK = 896;

// Spatial merger — merges each 2×2 spatial block of patches into one token
// (spatial_merge_size=2 ⇒ spatial_merge_size²=4). The merger reshapes the tower
// output [S, hidden] → [S/4, hidden*4] then MLP-projects to the text hidden
// size. See emit_merger.
constexpr int64_t QWEN3_5_SPATIAL_MERGE_UNIT = 4;
constexpr int64_t G05_TEMPORAL_PATCH_GROUP = 16;
constexpr int64_t G05_SPATIAL_GROUP_ROWS = 768;  // three 256-patch frames

constexpr int64_t QWEN35_VISION_ATTN_SITE = 1711359184452140201LL;
constexpr int64_t QWEN35_VISION_RAW_SPM_ATTN_SITE = 7600841761525222001LL;
constexpr int64_t QWEN35_VISION_KV_SITE = 7862402590903398933LL;
constexpr int64_t QWEN35_VISION_TEMPORAL_ATTN_SITE = 97797974637846192LL;
constexpr int64_t QWEN35_VISION_TEMPORAL_KV_SITE = 3433090853780504513LL;
constexpr int64_t QWEN35_VISION_SPATIAL_ATTN_SITE = 6909027152890455551LL;
constexpr int64_t QWEN35_VISION_SPATIAL_RAW_SPM_ATTN_SITE =
    809625674292115789LL;
constexpr int64_t QWEN35_VISION_SPATIAL_KV_SITE = 3308017190679002622LL;
constexpr uint32_t QWEN35_VISION_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
    KV_INSERT_CAP_NON8_TP_V16;
constexpr int64_t QWEN35_VISION_KV_REASON_DDR_REQUIRED = 1;
constexpr int64_t G05_VISION_KV_RAW_SPM_ONLY = 2;
constexpr int64_t G05_VISION_QKV_SPM_TRANSPORT_SITE = 8596948981803816035LL;
constexpr int64_t G05_VISION_QKV_LINEAR_SITE = 3357339527802085618LL;
constexpr int64_t G05_VISION_QKV_W8_LINEAR_SITE = 1317639897773331476LL;

enum Qwen35VisionAttentionRouteFlags : int64_t {
    QWEN35_VISION_ATTN_PREFIX_HISTORY_DDR_REQUIRED = 1LL << 0,
    QWEN35_VISION_ATTN_MULTI_CHUNK_DDR_REQUIRED = 1LL << 1,
    QWEN35_VISION_ATTN_RAW_KERNEL_INCOMPATIBLE_DDR_REQUIRED = 1LL << 2,
    QWEN35_VISION_ATTN_TEMPORAL_PACKED_KV_DDR_REQUIRED = 1LL << 3,
    QWEN35_VISION_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED = 1LL << 4,
};

// Stable native call-site IDs from planner_owners.v1.json.  These keep the
// descriptor authoritative without moving the launchers behind a wrapper (the
// source-scope/call-ordinal identity therefore remains stable).
constexpr int64_t QWEN35_VISION_STEP0_COMPACT_PIXEL_DMA_SITE =
    6712651654965044368LL;
constexpr int64_t QWEN35_VISION_STEP0_PIXEL_DMA_SITE = 8646450006490728151LL;
constexpr int64_t QWEN35_VISION_STEP0_LINEAR_SITE = 7044106151793639083LL;
constexpr int64_t QWEN35_VISION_STEP0_POS_SCATTER_DMA_SITE =
    591189870717419008LL;
constexpr int64_t QWEN35_VISION_STEP0_POS_BROADCAST_DMA_SITE =
    2705006413692791092LL;
constexpr int64_t QWEN35_VISION_STEP0_TEMPORAL_ALL_GATHER_SITE =
    9093737527971091235LL;
constexpr int64_t QWEN35_VISION_STEP0_GENERIC_ALL_GATHER_SITE =
    813569572519969877LL;

constexpr int64_t QWEN35_VISION_MERGER_FC1_SITE = 8408537352208396712LL;
constexpr int64_t QWEN35_VISION_MERGER_FC2_SITE = 4197638843121247872LL;
constexpr int64_t QWEN35_VISION_MERGER_ALL_REDUCE_SITE =
    4635180821953091595LL;
constexpr int64_t QWEN35_VISION_MERGER_OUTPUT_DMA_SITE =
    3314507152549266008LL;
constexpr int64_t QWEN35_VISION_TEMPORAL_PE_PRELOAD_SITE =
    8539869012261735791LL;
constexpr int64_t QWEN35_VISION_MERGER_NORM_G_PRELOAD_SITE =
    5543483196120457049LL;
constexpr int64_t QWEN35_VISION_MERGER_NORM_B_PRELOAD_SITE =
    1422421047835217586LL;
constexpr int64_t QWEN35_VISION_MERGER_FC1_BIAS_PRELOAD_SITE =
    8860010505634381437LL;
constexpr int64_t QWEN35_VISION_MERGER_FC2_BIAS_PRELOAD_SITE =
    5828121683551566290LL;

constexpr int64_t QWEN35_VISION_TEMPORAL_Q_SITE = 2246079301597880555LL;
constexpr int64_t QWEN35_VISION_TEMPORAL_K_SITE = 8825008333763691754LL;
constexpr int64_t QWEN35_VISION_TEMPORAL_V_SITE = 3158979552166677030LL;

constexpr int64_t QWEN35_VISION_SPATIAL_Q_SITE = 3360668686577324191LL;
constexpr int64_t QWEN35_VISION_SPATIAL_K_SITE = 1376033408560675313LL;
constexpr int64_t QWEN35_VISION_SPATIAL_V_SITE = 8493902959466033314LL;
constexpr int64_t QWEN35_VISION_SPATIAL_Q_ROPE_SITE = 4805537792544409401LL;
constexpr int64_t QWEN35_VISION_SPATIAL_K_ROPE_SITE = 2744991367458575500LL;
constexpr int64_t QWEN35_VISION_SPATIAL_O_SITE = 6800283320519983155LL;
constexpr int64_t QWEN35_VISION_SPATIAL_ATTN_ALL_REDUCE_SITE =
    3906697876182566086LL;
constexpr int64_t QWEN35_VISION_SPATIAL_FC1_SITE = 8391413315612938912LL;
constexpr int64_t QWEN35_VISION_SPATIAL_FC2_SITE = 606812431302212752LL;
constexpr int64_t QWEN35_VISION_SPATIAL_MLP_ALL_REDUCE_SITE =
    2902592988427317455LL;

constexpr int64_t QWEN35_VISION_Q_SITE = 3028526227305094594LL;
constexpr int64_t QWEN35_VISION_K_SITE = 4828134604955183557LL;
constexpr int64_t QWEN35_VISION_V_SITE = 6139494043019347068LL;
constexpr int64_t QWEN35_VISION_Q_ROPE_SITE = 161047416806883214LL;
constexpr int64_t QWEN35_VISION_K_ROPE_SITE = 2633310719855575545LL;
constexpr int64_t QWEN35_VISION_PREPARE_ALL_REDUCE_SITE =
    1938642469217626275LL;
constexpr int64_t QWEN35_VISION_O_SITE = 1743862006147235979LL;
constexpr int64_t QWEN35_VISION_ATTN_ALL_REDUCE_SITE =
    6062184355686500273LL;
constexpr int64_t QWEN35_VISION_FC1_SITE = 8324800155720525482LL;
constexpr int64_t QWEN35_VISION_FC2_SITE = 4104768967659171793LL;
constexpr int64_t QWEN35_VISION_MLP_ALL_REDUCE_SITE =
    8300267361097465404LL;

constexpr int64_t G05_VISION_K1_B3_SCHEDULE_SITE = 1989334778456497324LL;
constexpr int64_t G05_VISION_K1_B3_FC2_SITE = 8890075896770338674LL;
constexpr int64_t G05_VISION_K1_B3_QKV_SITE = 3001042279856051514LL;
constexpr int64_t G05_VISION_K1_B3_OWNER_LN_SITE = 4931260444092209415LL;
constexpr int64_t G05_VISION_K1_B3_COMPACT_SITE = 5956654682946814833LL;
constexpr int64_t G05_VISION_K1_B3_Q_ROPE_SITE = 4444946969262085648LL;
constexpr int64_t G05_VISION_K1_B3_K_ROPE_SITE = 1024385423218200374LL;
constexpr int64_t G05_VISION_K1_B3_ATTN_SITE = 2606301310892565949LL;
constexpr int64_t G05_VISION_K1_B3_PREPARE_SITE = 303833679029979601LL;
constexpr int64_t G05_VISION_K1_B3_O_SITE = 4313929404421908286LL;
constexpr int64_t G05_VISION_K1_B3_FC1_SITE = 7317754097448584713LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_SCHEDULE_SITE = 7234885135216489262LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_FC2_SITE = 6974677568427610430LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_QKV_SITE = 583455119213555928LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_OWNER_LN_SITE = 6580584106573854246LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_COMPACT_SITE = 7943480501751969662LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_Q_ROPE_SITE = 3569859980690610112LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_K_ROPE_SITE = 8893169073786468458LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_ATTN_SITE = 886292775184363593LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_PREPARE_SITE = 4580452915284906990LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_O_SITE = 2848646348355120877LL;
constexpr int64_t G05_VISION_K1_CAMERA_GROUP_FC1_SITE = 457424401846535204LL;

constexpr int64_t G05_VISION_OWNER_LN_SITE = 7401236839773657641LL;
constexpr int64_t G05_VISION_COMPACT_RESIDUAL_SITE = 8040294111507309046LL;

enum class Qwen35VisionAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

enum class Qwen35VisionRopeRoute : int64_t {
    ROPE_2D_SPM = 1,
};

enum class Qwen35VisionMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM = 1,
    DDR_SCATTER_TO_SPM = 2,
    SPM_COPY_TO_DDR = 3,
};

int64_t qwen35_vision_ring_route(int64_t rows, int64_t cols, int cores) {
    return v3::fmb_ring_all_reduce_route_selector(rows, cols, cores);
}

constexpr int64_t qwen35_vision_linear_route() {
    return static_cast<int64_t>(v3::FmbLinearRouteSelector::AUTO_TILE);
}

constexpr int64_t qwen35_vision_rope_route() {
    return static_cast<int64_t>(Qwen35VisionRopeRoute::ROPE_2D_SPM);
}

constexpr int64_t qwen35_vision_dma_route(Qwen35VisionMutableDmaRoute route) {
    return static_cast<int64_t>(route);
}

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

// QWEN3_5_VISION_DBG_Q=1 enables the Phase-1 rope'd-Q probe. OFF by default because
// its active-core scatter emits cross-core barrier fences per layer
// (rpu_memcpy.cpp:936-944 — the `for (ch = 1; ch < num_cores)` loops), i.e. ~336 extra barriers
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

Qwen3_5VisionModel::Qwen3_5VisionModel() {
    // Compatibility default for direct native callers; adapters bind the
    // requested cache capacity before installing any weights.
    cache_capacity_ = 4096;
    keepalive_capacity_ = std::max(cache_capacity_, G05_TEMPORAL_MAX_SEQ);
    set_chunk_envelope(cache_capacity_, 0);
}

void Qwen3_5VisionModel::configure_cache_capacity(int64_t max_seq_len) {
    TORCH_CHECK(max_seq_len > 0 &&
                    max_seq_len <= (std::numeric_limits<int64_t>::max() - 128) / 2,
                "Qwen3.5 Vision capacity must be positive and fit position storage");
    TORCH_CHECK(num_layers() == 0 && get_last_resolved_chunk_size() == 0 &&
                    !position_idx_keepalive_.defined() && !q_ddr_buf_.defined() &&
                    !hidden_buf_.defined() && !merged_buf_.defined(),
                "Qwen3.5 Vision cache capacity must be bound before weights or fixed-DMA storage");
    set_chunk_envelope(max_seq_len, 0);
    cache_capacity_ = max_seq_len;
    keepalive_capacity_ = std::max(max_seq_len, G05_TEMPORAL_MAX_SEQ);
    invalidate_model_state();
}

void Qwen3_5VisionModel::configure_execution_cores(int64_t cores) {
    TORCH_CHECK(cores == 4 || cores == 8,
                "Qwen3.5 Vision admits four or eight execution cores");
    TORCH_CHECK(cores == 8 || (!has_temporal_ && !z2_bound_),
                "Reduced Qwen3.5 Vision excludes temporal and Z2 profiles");
    set_execution_core_count(static_cast<int>(cores));
}

std::vector<int64_t> Qwen3_5VisionModel::execution_topology() const {
    TORCH_CHECK(num_layers() > 0,
                "Qwen3.5 Vision topology requires installed model weights");
    return {num_cores(), attn_tp(), mlp_tp(), 8};
}

DecoderExecutionTopology Qwen3_5VisionModel::resolve_model_execution_topology(
    int64_t q, int64_t kv, int64_t d, int64_t h, int64_t intermediate) const {
    if (num_cores() == 8) {
        // Vision swizzles whole heads over gcd(heads, owner). In particular,
        // the 0.8B tower has 12 heads and four attention producers, while
        // its MLP and residual consumers still use eight cores.
        TORCH_CHECK(q > 0 && q == kv, "Qwen3.5 Vision requires positive MHA heads");
        return {8, static_cast<int>(std::gcd<int64_t>(q, 8)), 8};
    }
    TORCH_CHECK(num_cores() == 4 && q == kv &&
                    ((q == 12 && h == 768 && intermediate == 3072 && d == 64) ||
                     (q == 16 &&
                      ((h == 1024 && intermediate == 4096 && d == 64) ||
                       (h == 1152 && intermediate == 4352 && d == 80)))),
                "No admitted four-core Qwen3.5 Vision physical geometry");
    return {4, 4, 4};
}

int64_t Qwen3_5VisionModel::subclass_layout_hash() const {
    int64_t hash = 0;
    if (high_precision_) {
        hash = detail::layout_mix(hash, G05_VISION_HIGH_ENCODER_ACTIVATION_SITE);
        hash = detail::layout_mix(hash, 1);
        hash = detail::layout_mix(hash, G05_VISION_HIGH_MERGER_ACTIVATION_SITE);
    }
    if (num_cores() != 8) {
        hash = detail::layout_mix(hash, 2);
        for (const int value : {num_cores(), attn_tp(), mlp_tp(), 8}) {
            hash = detail::layout_mix(hash, value);
        }
    }
    if (g05_qkv_spm_available_) {
        hash = detail::layout_mix(hash, G05_VISION_QKV_SPM_TRANSPORT_SITE);
        hash = detail::layout_mix(hash, 1);
        if (!g05_w8a16_ && !linear_acc32_)
            hash = detail::layout_mix(hash, 16);
    }
    if (g05_qkv_linear_available_) {
        hash = detail::layout_mix(hash, g05_w8a16_ ?
            G05_VISION_QKV_W8_LINEAR_SITE : G05_VISION_QKV_LINEAR_SITE);
        hash = detail::layout_mix(hash, 1);
    }
    if (g05_w8a16_) {
        hash = detail::layout_mix(hash, 0x4730355738ULL);
        hash = detail::layout_mix(hash, linear_acc32_ ? 32 : 16);
    }
    if (is_g05_k1_b3_profile()) {
        hash = detail::layout_mix(hash,G05_VISION_K1_B3_SCHEDULE_SITE);
        for(auto value:g05_k1_b3_arguments())hash=detail::layout_mix(hash,value);
    }
    if (is_g05_k1_camera_group_profile()) {
        hash = detail::layout_mix(hash,G05_VISION_K1_CAMERA_GROUP_SCHEDULE_SITE);
        for(auto value:g05_k1_camera_group_arguments())hash=detail::layout_mix(hash,value);
    }
    if (g05_owner_ln_available_) {
        hash = detail::layout_mix(hash, G05_VISION_OWNER_LN_SITE);
        hash = detail::layout_mix(hash, 1); // compact raw + full normalized ABI
    }
    return hash;
}

// QWEN35_VISION_FIXED_KERNEL_BASIS: non-manifest launchers implement fixed
// patch/merger math, normalization/activation, or BufferDecl transport. Any
// selectable physical implementation must first become a typed manifest route.

static std::vector<ChunkInfo> make_vision_chunks(
    int64_t execution_len, int64_t chunk_size) {
    TORCH_INTERNAL_ASSERT(execution_len > 0 && chunk_size > 0);
    std::vector<ChunkInfo> chunks;
    for (int64_t offset = 0; offset < execution_len; offset += chunk_size) {
        const int64_t len = std::min(chunk_size, execution_len - offset);
        chunks.push_back({static_cast<int>(chunks.size()), offset, len,
                          offset + len});
    }
    return chunks;
}

void Qwen3_5VisionModel::configure_execution_geometry(
    int64_t execution_len,
    int64_t temporal_num_frames,
    int64_t camera_batch_count,
    bool compact_input) {
    TORCH_CHECK(num_cores() == 8 ||
                    (temporal_num_frames == 1 && camera_batch_count == 1 && !compact_input),
                "Reduced Qwen3.5 Vision admits only single-image spatial execution");
    TORCH_CHECK(execution_len > 0 && execution_len <= keepalive_capacity_,
                "Qwen3_5VisionModel: execution_len=", execution_len,
                " exceeds installed physical capacity ", keepalive_capacity_);
    TORCH_CHECK(execution_len <= QWEN3_5_VISION_SDPA_MAX_KV_ROWS,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 Vision key length exceeds the kernel register domain");
    TORCH_CHECK(temporal_num_frames >= 1,
                "Qwen3_5VisionModel: temporal_num_frames must be positive");
    TORCH_CHECK(camera_batch_count >= 1 &&
                    camera_batch_count <= G05_MAX_CAMERA_BATCH,
                "Qwen3_5VisionModel: camera_batch_count must be 1, 2 or 3");

    current_temporal_num_frames_ = temporal_num_frames;
    current_camera_batch_count_ = camera_batch_count;
    current_compact_step0_ = compact_input;
    if (temporal_num_frames > 1) {
        TORCH_CHECK(has_temporal_ && temporal_num_frames == temporal_num_frames_,
                    "Qwen3_5VisionModel: unsupported temporal frame count ",
                    temporal_num_frames);
        if (camera_batch_count > 1) {
            TORCH_CHECK(temporal_num_frames_ == 6 &&
                            temporal_patches_per_frame_ == 256 &&
                            camera_batch_count == G05_MAX_CAMERA_BATCH,
                        "Qwen3_5VisionModel: packed temporal vision supports "
                        "only B_camera=3,K=6,P=256");
        }
        const int64_t expected = camera_batch_count * temporal_num_frames_ *
                                 temporal_patches_per_frame_;
        TORCH_CHECK(execution_len == expected,
                    "Qwen3_5VisionModel: temporal input must contain ", expected,
                    " patches, got ", execution_len);
        current_output_patches_ =
            camera_batch_count * temporal_patches_per_frame_;
    } else {
        TORCH_CHECK(execution_len <= cache_capacity_,
                    "Qwen3_5VisionModel: spatial input exceeds installed KV capacity ",
                    cache_capacity_, "; got ", execution_len);
        TORCH_CHECK(camera_batch_count == 1 ||
                        (execution_len == camera_batch_count * 256 &&
                         g05_k1_camera_group_available(camera_batch_count)),
                    "Qwen3_5VisionModel: K1 camera grouping requires exact installed W8 mode/geometry");
        TORCH_CHECK(!compact_input,
                    "Qwen3_5VisionModel: compact STEP0 requires temporal vision");
        current_output_patches_ = execution_len;
    }
    if (compact_input) {
        TORCH_CHECK(
            has_step0_ && temporal_num_frames == 6 &&
                camera_batch_count == G05_MAX_CAMERA_BATCH &&
                temporal_patches_per_frame_ == G05_COMPACT_PATCHES_PER_FRAME &&
                patch_dim_ == G05_COMPACT_CHANNELS * 2 *
                                  G05_COMPACT_PATCH_SIDE * G05_COMPACT_PATCH_SIDE,
            "Qwen3_5VisionModel: compact STEP0 requires the canonical "
            "B_camera=3,K=6,C=3,H=W=256,tp=2,patch=16 profile");
    }
}

int64_t Qwen3_5VisionModel::input_chunk_size_for_geometry(
    int64_t execution_len) const {
    if (is_g05_k1_b3_profile() || is_g05_k1_camera_group_profile()) return 256;
    const int64_t exact = get_chunk_size_override();
    if (!has_step0_) return execution_len;
    if (current_compact_step0_) return G05_COMPACT_STEP0_ROWS;
    if (current_temporal_num_frames_ > 1) {
        return current_temporal_num_frames_ * temporal_patches_per_frame_;
    }
    return exact > 0
        ? std::min(exact, QWEN3_5_VISION_STEP0_CHUNK)
        : QWEN3_5_VISION_STEP0_CHUNK;
}

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
    int64_t camera_batch_count,
    int64_t expected_stage_plan_fingerprint_hi,
    int64_t expected_stage_plan_fingerprint_lo,
    int64_t expected_layout_hash_hi,
    int64_t expected_layout_hash_lo,
    at::IntArrayRef planned_stage_descriptor)
{
    TORCH_CHECK(!z2_bound_,
                "Qwen3_5VisionModel::forward: physical Z2 lease is active; "
                "use the coordinator-owned Z2 dispatch");
    return forward_impl(input, k_caches, v_caches, num_patches_in,
                        step0_pos, fusion_target, fusion_row_starts,
                        temporal_num_frames, camera_batch_count,
                        expected_stage_plan_fingerprint_hi,
                        expected_stage_plan_fingerprint_lo,
                        expected_layout_hash_hi,
                        expected_layout_hash_lo,
                        planned_stage_descriptor,
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
    int64_t expected_stage_plan_fingerprint_hi,
    int64_t expected_stage_plan_fingerprint_lo,
    int64_t expected_layout_hash_hi,
    int64_t expected_layout_hash_lo,
    at::IntArrayRef planned_stage_descriptor,
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
    if (num_cores() != 8) {
        TORCH_CHECK(static_cast<int64_t>(k_caches.size()) == num_layers() &&
                        static_cast<int64_t>(v_caches.size()) == num_layers(),
                    "Reduced Qwen3.5 Vision requires one K/V cache per layer");
        const int64_t physical_slots = 8 * num_kv_heads() / attention_tp_;
        for (int64_t layer = 0; layer < num_layers(); ++layer) {
            for (const auto* cache : {&k_caches[layer], &v_caches[layer]}) {
                TORCH_CHECK(cache->defined() && cache->dim() == 7 &&
                                cache->scalar_type() == at::kHalf &&
                                cache->device().type() == at::kPrivateUse1 &&
                                cache->is_contiguous() && cache->size(0) == 1 &&
                                cache->size(1) * 16 >= num_patches_in &&
                                cache->size(2) == physical_slots / 8 &&
                                cache->size(3) == head_dim() / 16 &&
                                cache->size(4) == 8 && cache->size(5) == 16 &&
                                cache->size(6) == 16,
                            "Reduced Qwen3.5 Vision cache must match attention TP4 and physical DDR8");
            }
        }
    }
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "Qwen3_5VisionModel::forward: input must be on device");
    TORCH_CHECK(input.scalar_type() == at::kHalf,
                "Qwen3_5VisionModel::forward: input must be FP16");
    TORCH_CHECK(input.is_contiguous(),
                "Qwen3_5VisionModel::forward: input must be contiguous");
    const bool compact_input = input.dim() == 5;
    TORCH_CHECK(input.dim() == 3 || compact_input,
                "Qwen3_5VisionModel::forward: input must be 3D [1,N,H] or the "
                "canonical compact 5D [1,18,3,256,256], got ", input.dim(), "D");
    TORCH_CHECK(num_patches_in > 0 && num_patches_in <= keepalive_capacity_,
                "Qwen3_5VisionModel::forward: num_patches=", num_patches_in,
                " exceeds installed physical capacity ", keepalive_capacity_);
    TORCH_CHECK(input.size(0) == 1,
                "Qwen3_5VisionModel::forward: batch must be 1, got ", input.size(0));
    if (!compact_input) {
        TORCH_CHECK(input.size(1) == num_patches_in,
                    "Qwen3_5VisionModel::forward: input.size(1)=", input.size(1),
                    " must match num_patches=", num_patches_in);
    }
    configure_execution_geometry(
        num_patches_in, temporal_num_frames, camera_batch_count, compact_input);
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

    VisionExecutionPlan execution_plan;
    if (z2_dispatch) {
        TORCH_CHECK(
            planned_stage_descriptor.empty(),
            "Qwen3_5VisionModel Z2 dispatch owns its native stage plan");
        execution_plan = resolve_execution_plan(num_patches_in);
    } else {
        TORCH_CHECK(
            !planned_stage_descriptor.empty(),
            "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 Vision forward requires "
            "one complete planner stage descriptor");
        const auto prepared_candidate = prepare_stage_candidate(planned_stage_descriptor);
        const auto& candidate = prepared_candidate->candidate();
        execution_plan.input_chunk_size = candidate.input_chunk_size;
        execution_plan.qkv_chunk_size = candidate.qkv_chunk_size;
        execution_plan.compute_chunk_size = candidate.compute_chunk_size;
        execution_plan.stages = candidate.stage_plan;
    }
    TORCH_CHECK(
        expected_stage_plan_fingerprint_hi >= 0 &&
            expected_stage_plan_fingerprint_hi <= 0xffffffffLL &&
            expected_stage_plan_fingerprint_lo >= 0 &&
            expected_stage_plan_fingerprint_lo <= 0xffffffffLL,
        "Qwen3_5VisionModel: expected stage-plan fingerprint words are invalid");
    if (expected_stage_plan_fingerprint_hi != 0 ||
        expected_stage_plan_fingerprint_lo != 0) {
        const uint64_t expected_fingerprint =
            (static_cast<uint64_t>(expected_stage_plan_fingerprint_hi) << 32) |
            static_cast<uint64_t>(expected_stage_plan_fingerprint_lo);
        TORCH_CHECK(
            fmb_three_stage_chunk_plan_fingerprint(execution_plan.stages) ==
                expected_fingerprint,
            "Qwen3_5VisionModel: physical plan changed before Graph execution");
        TORCH_CHECK(
            expected_layout_hash_hi >= 0 &&
                expected_layout_hash_hi <= 0xffffffffLL &&
                expected_layout_hash_lo >= 0 &&
                expected_layout_hash_lo <= 0xffffffffLL,
            "Qwen3_5VisionModel: expected layout-hash words are invalid");
        const uint64_t expected_layout_hash =
            (static_cast<uint64_t>(expected_layout_hash_hi) << 32) |
            static_cast<uint64_t>(expected_layout_hash_lo);
        TORCH_CHECK(
            expected_layout_hash != 0,
            "Qwen3_5VisionModel: expected layout hash must be nonzero");
    }

    // Cache the num_patches for build_layer_subgraph to consume (Q/K rope
    // launcher needs the actual token count, which is also the chunk len).
    current_num_patches_ = num_patches_in;

    // Flush the position_idx keepalive — adapter just wrote per-forward
    // (row, col) entries into it via Python `.copy_()`. Without an explicit
    // flush, the CPU write may sit in the caching allocator's write buffer
    // and the device DMA reads stale (zero or prior-forward) data, silently
    // collapsing 2D RoPE to identity. Same pattern as Qwen3 M-RoPE in
    // CausalDecoderModel::forward (Phase 1 keepalive flush).
    // Only the installed G05 K1 owner profile may defer these four acquires.
    // Prove all current sized ranges before skipping the first scalar call;
    // unknown owners/rounded bounds keep every original scalar call in place.
    bool batch_input_acquire = false;
    std::array<void*, 4> input_acquire_ptrs{};
    std::array<size_t, 4> input_acquire_sizes{};
    if (g_rpu_ddr_flush_force_enabled && is_g05_k1_fast_replay_profile() &&
        (g05_w8a16_ || g05_qkv_spm_available_) && !high_precision_ &&
        !z2_dispatch && !get_debug_export() && !compact_input &&
        temporal_num_frames == 1 &&
        (camera_batch_count == 1 || (camera_batch_count == 3 &&
            (is_g05_k1_b3_profile() || is_g05_k1_camera_group_profile()))) &&
        num_patches_in == camera_batch_count * 256 &&
        step0_pos.has_value() && fusion_target.has_value()) {
        const auto& pos = *step0_pos;
        const auto& target = *fusion_target;
        if (pos.defined() && pos.device().type() == at::kPrivateUse1 &&
            pos.scalar_type() == at::kHalf && pos.is_contiguous() &&
            pos.dim() == 2 && pos.size(0) == num_patches_in &&
            pos.size(1) == hidden_size() &&
            target.defined() && target.device().type() == at::kPrivateUse1 &&
            target.scalar_type() == at::kHalf && target.is_contiguous() &&
            target.dim() == 3 && target.size(0) == 1 &&
            target.size(2) == out_hidden_size_ && target.size(1) >= 64 &&
            fusion_row_starts.size() == camera_batch_count &&
            std::all_of(fusion_row_starts.begin(), fusion_row_starts.end(),
                [&](int64_t row) { return row >= 0 && row <= target.size(1) - 64; })) {
            input_acquire_ptrs = {{position_idx_keepalive_.data_ptr<int16_t>(),
                input.data_ptr<c10::Half>(), pos.data_ptr<c10::Half>(),
                target.data_ptr<c10::Half>()}};
            input_acquire_sizes = {{position_idx_keepalive_.nbytes(), input.nbytes(),
                pos.nbytes(), target.nbytes()}};
            batch_input_acquire = true;
            for (size_t i = 0; i < input_acquire_ptrs.size(); ++i) {
                auto* owner = ::rhino_lkn::RpuGetHostddr(input_acquire_ptrs[i]);
                const uintptr_t base = owner
                    ? reinterpret_cast<uintptr_t>(owner->get_cpu_ptr()) : 0;
                const size_t owner_bytes = owner ? owner->get_memory_size() : 0;
                const uintptr_t address = reinterpret_cast<uintptr_t>(input_acquire_ptrs[i]);
                const size_t bytes = input_acquire_sizes[i];
                const uintptr_t limit = std::numeric_limits<uintptr_t>::max();
                bool bounded = owner && base != 0 && bytes > 0 && owner_bytes > 0 &&
                    owner_bytes <= limit - base && address >= base &&
                    address <= limit - 63 && bytes <= limit - address - 63;
                if (bounded) {
                    const size_t offset = address - base;
                    bounded = offset <= owner_bytes && bytes <= owner_bytes - offset &&
                        (address & ~uintptr_t(63)) >= base &&
                        ((address + bytes + 63) & ~uintptr_t(63)) <= base + owner_bytes;
                }
                if (!bounded) {
                    batch_input_acquire = false;
                    break;
                }
            }
        }
    }
    if (!batch_input_acquire) {
        rpu_ddr_flush_force_sized(position_idx_keepalive_.data_ptr<int16_t>(),
                                  position_idx_keepalive_.nbytes());
    }

    // KV_FIRST 双向分块：不单 chunk override，规划器按 SPM 预算挑 chunk_size；
    // adapter 可在建图前把 legacy cap 翻译到本 handle。q_ddr_buf_ 存每 core 的 rope 后 Q
    // 跨两相（Phase1 emit_kv_first_body 写 / Phase2 build_layer_subgraph 读）。
    //
    // 按 cold install 绑定的 keepalive_capacity_ 铺满，Graph 存活期间不重分配。
    // 不能沿用 gemma 的 grow-only（rpu_gemma_model.cpp:440-448，本文件原样抄来）：存/读 Q 走
    // 的是 FIXED DMA，其地址在 BUILD 时写进 kd_buf，而 REPLAY 的 sync-only 快路径
    // （graph_runtime_execute.cpp:842-885）只对登记过的 mutable DMA 调 update_dma_kernel，
    // fixed 的包没人碰。于是「小图 → 大图（换 buf，旧块 free 回 allocator）→ 再来小图（sig
    // 命中旧图 → REPLAY）」会照 BUILD 时的旧地址去写，而那块 DDR 已被分给别的张量 → 静默
    // 糊掉它。（Q 自身反而是对的：存和读烤的是同一个旧地址+旧 stride，自洽。）
    // Replay's sync-only fast path does not rebuild fixed DMA nodes, so the
    // backing allocation must remain stable for the graph lifetime.
    // 请求必须在安装时分配的容量内（本函数开头的 TORCH_CHECK）。
    if (temporal_num_frames == 1 && !q_ddr_buf_.defined()) {
        const int64_t local_q_dim = (num_q_heads() / attention_tp_) * head_dim();
        q_ddr_buf_ = at::empty(
            {(int64_t)attention_tp_, keepalive_capacity_, local_q_dim},
            input.options());
    }
    if (temporal_num_frames > 1 && !temporal_k_cache_.defined()) {
        // The temporal SDPA kernel's validated head envelope is one camera's
        // 16-patch group. B3 reuses this cache sequentially per camera.
        const int64_t composite_heads =
            G05_TEMPORAL_PATCH_GROUP * num_q_heads();
        const int64_t composite_heads_per_core = composite_heads / num_cores();
        const std::vector<int64_t> cache_shape = {
            1, 1, composite_heads_per_core, head_dim() / 16,
            (int64_t)num_cores(), 16, 16};
        temporal_k_cache_ = empty_256b_aligned_fp16_rpu(
            cache_shape, input.options(), "G0.5 temporal K cache");
        temporal_v_cache_ = empty_256b_aligned_fp16_rpu(
            cache_shape, input.options(), "G0.5 temporal V cache");
    }

    // Per-layer debug snapshots (get_debug_export() only). The in-graph DMAs in
    // emit_kv_first_body / build_layer_subgraph fill them during (capture: replay /
    // passthrough: sync) execution — Python reads them AFTER forward via the get_dbg_*
    // torch ops. dbg_hidden = per-layer output [L,N,H]; dbg_q = Phase-1 rope'd Q per-core
    // [L, attention_tp, N, local_q_dim].
    //
    // Allocated ONCE per shape, NOT per forward — same rule as q_ddr_buf_/hidden_buf_
    // below: both snapshots are written by FIXED DMAs whose dst is baked into kd_buf at
    // BUILD, and REPLAY's sync-only fast path (this tower is single-segment ⇒ it always
    // hits) only rebinds MUTABLE DMAs. A fresh at::empty per forward therefore left the
    // graph writing the PREVIOUS forward's buffer while the getter returned an
    // untouched one — a probe that silently reports stale data. A shape change forces a
    // re-BUILD, so keying the allocation on shape is sufficient.
    if (get_debug_export()) {
        const int64_t L = num_layers(), N = num_patches_in, H = hidden_size();
        if (!dbg_hidden_.defined() || dbg_hidden_.size(0) != L ||
            dbg_hidden_.size(1) != N || dbg_hidden_.size(2) != H) {
            dbg_hidden_ = at::empty({L, N, H}, input.options());
        }
        if (temporal_num_frames == 1 && dbg_q_enabled()) {
            const int64_t lq = (num_q_heads() / attention_tp_) * head_dim();
            if (!dbg_q_.defined() || dbg_q_.size(0) != L ||
                dbg_q_.size(1) != (int64_t)attention_tp_ || dbg_q_.size(2) != N ||
                dbg_q_.size(3) != lq) {
                dbg_q_ = at::empty({L, (int64_t)attention_tp_, N, lq}, input.options());
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
                    "Qwen3_5VisionModel::forward: device STEP0 requires position embeddings");
        const at::Tensor& pos = *step0_pos;
        const bool temporal_pos = temporal_num_frames > 1;
        const bool shape_ok = temporal_pos
            ? (pos.dim() == 3 && pos.size(0) == num_cores() &&
               pos.size(1) == num_patches_in &&
               pos.size(2) == hidden_size() / num_cores())
            : (pos.dim() == 2 && pos.size(0) == num_patches_in &&
               pos.size(1) == hidden_size());
        TORCH_CHECK(pos.device().type() == at::kPrivateUse1 &&
                    pos.scalar_type() == at::kHalf && pos.is_contiguous() && shape_ok,
                    "Qwen3_5VisionModel::forward: step0_pos must be contiguous fp16 device ",
                    temporal_pos ? "[8,N,H/8]" : "[N,H]", ", got ", pos.sizes());
        // Refresh the live pixel address for emit_step0's mutable DMAs. The
        // member's ADDRESS is what the graph registry holds; only its value moves.
        // Own both tensors alongside their bases: the DMAs are dereferenced when
        // the graph EXECUTES (RpuKernelGraph::end(), i.e. when the caller's
        // capture scope exits), not when forward() returns — same reason
        // merger_dst_ref_ exists below (pitfalls.md C-1).
        pixel_src_ref_ = input;
        step0_pos_ref_ = pos;
        pixel_src_base_ = ::rhino_lkn::RpuGetDevAddr(input.data_ptr());
        step0_pos_src_base_ = ::rhino_lkn::RpuGetDevAddr(pos.data_ptr());
        if (!batch_input_acquire) {
            rpu_ddr_flush_force_sized(input.data_ptr<c10::Half>(), input.nbytes());
            rpu_ddr_flush_force_sized(pos.data_ptr<c10::Half>(), pos.nbytes());
        }
        if (!hidden_buf_.defined()) {
            hidden_buf_ = at::empty(
                {1, keepalive_capacity_, hidden_size()}, input.options());
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
                    "Qwen3_5VisionModel::forward: fusion_target requires the device merger");
        TORCH_CHECK(target.defined() && target.device().type() == at::kPrivateUse1 &&
                    target.scalar_type() == at::kHalf && target.is_contiguous(),
                    "Qwen3_5VisionModel::forward: fusion_target must be contiguous fp16 device");
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
        if (batch_input_acquire) {
            // No input DMA occurs before run_all_layers below. Preserve every
            // original sized range, including public dirty aliases/text rows.
            ::rhino_lkn::RpuDdrFlushBatch(
                input_acquire_ptrs.data(), input_acquire_sizes.data(),
                input_acquire_ptrs.size());
#if defined(__aarch64__)
            __asm__ volatile("dsb sy" ::: "memory");
#endif
        } else {
            rpu_ddr_flush_force_sized(target.data_ptr<c10::Half>(), target.nbytes());
        }
    } else if (has_merger_) {
        TORCH_CHECK(fusion_row_starts.empty(),
                    "Qwen3_5VisionModel::forward: fusion_row_starts require fusion_target");
        if (!merged_buf_.defined()) {
            merged_buf_ = at::empty(
                {1, keepalive_capacity_ / QWEN3_5_SPATIAL_MERGE_UNIT,
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

    TORCH_INTERNAL_ASSERT(layer_input->size(1) == num_patches_in);
    const uint64_t expected_layout_hash =
        (static_cast<uint64_t>(expected_layout_hash_hi) << 32) |
        static_cast<uint64_t>(expected_layout_hash_lo);
    at::Tensor result = z2_dispatch
        ? run_all_layers(
              *layer_input, k_caches, v_caches,
              /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false,
              execution_plan.stages.input.chunks,
              execution_plan.stages.spans,
              execution_plan.stages.boundary_policies,
              expected_layout_hash)
        : run_all_layers(
              *layer_input, k_caches, v_caches,
              /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false,
              /*planned_chunk_size=*/0, planned_stage_descriptor,
              expected_layout_hash);
    TORCH_CHECK(get_last_resolved_chunk_size() ==
                    execution_plan.compute_chunk_size,
                "Qwen3_5VisionModel: dispatch compute chunk drifted from the "
                "planning descriptor");
    TORCH_CHECK(
        fmb_three_stage_chunk_plan_fingerprint(ctx().stage_plan) ==
            fmb_three_stage_chunk_plan_fingerprint(execution_plan.stages),
        "Qwen3_5VisionModel: dispatch stage plan drifted from the planning "
        "descriptor");

    return result;
}

SpmPipelineComponentLayout Qwen3_5VisionModel::prepare_z2_layout(
    int64_t num_patches) {
    TORCH_CHECK(!high_precision_, "G05 HIGH Vision math requires its retained COMPLETE schedule");
    TORCH_CHECK(num_cores() == 8,
                "Qwen3.5 Vision Z2 canary requires eight execution cores");
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
                "[1024 -> 2048] device merger");
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
    auto prepared = prepare_spm_pipeline_component(
        layout, compose_fmb_default_three_stage_chunk_plan(
                    layout, num_patches, /*position=*/0,
                    ChunkMode::KV_FIRST));
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
        /*expected_stage_plan_fingerprint_hi=*/0,
        /*expected_stage_plan_fingerprint_lo=*/0,
        /*expected_layout_hash_hi=*/0,
        /*expected_layout_hash_lo=*/0,
        /*planned_stage_descriptor=*/{},
        /*z2_dispatch=*/true);
}

// ── emit_step0: pre_layers_fn — pixel → hidden, once per forward, in-graph,
//    before the layer loop. Mirrors rhino's STEP 0
//    (qwen3_vl_vision_sisc_graph.cc:394-407):
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
//    rhino's VitAllGather folds the residual add in, but decomposes internally
//    into gather + a separate llama_eltwise_binary (vit.cc:451-459) — the two
//    launches below ARE that decomposition, not a shortcut.
void Qwen3_5VisionModel::emit_step0() {
    const int64_t N  = current_num_patches_;
    const int64_t h  = hidden_size();
    const int64_t pd = patch_dim_;
    const int64_t local_h = h / num_cores();
    const bool temporal_pos = current_temporal_num_frames_ > 1;
    const int64_t cs0 = input_chunk_size_for_geometry(N);

    // Bias: COL-parallel splits it across cores (ParallelLinearGenerator.py:399
    // `torch.split(b, local_n)`), same as the tower's fc1_bias at :581-585.
    // ⚠️ NOT the row-parallel convention (fill-zero + broadcast to core 0 only) —
    // that one is for o_proj/fc2, whose partial sums get all-reduced afterwards.
    rpu_launch_ddr_scatter_spm_dma(
        pe_b_.data_ptr<c10::Half>(),
        local_h, local_h * DWIDTH,
        addr(0, "s0_bias"), num_cores());

    for (int64_t off = 0; off < N; off += cs0) {
        const int64_t len = std::min<int64_t>(cs0, N - off);
        const int64_t invocation = off / cs0;
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
            consume_manifest_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_VISION_STEP0_COMPACT_PIXEL_DMA_SITE,
                qwen35_vision_dma_route(
                    Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                invocation, {current_compact_step0_ ? 1 : 0});
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &pixel_src_base_,
                /*src_offset_bytes=*/frame_start * compact_frame_elems * DWIDTH,
                /*num_elements=*/G05_COMPACT_STEP0_FRAMES * compact_frame_elems,
                layout_a, num_cores());
            rpu_launch_tile_spm_kernel(
                layout_a, layout_b,
                /*dim0=*/G05_COMPACT_STEP0_FRAMES * G05_COMPACT_CHANNELS,
                /*dim1=*/G05_COMPACT_IMAGE_SIDE,
                /*dim2=*/G05_COMPACT_IMAGE_SIDE,
                /*repeat0=*/1, /*repeat1=*/2, /*repeat2=*/1, num_cores());

            // ① patchify each plane: [gh,ph,gw,pw] -> [gh,gw,ph,pw].
            rpu_launch_permute4d_0213_spm_kernel(
                layout_b, layout_a,
                /*q=*/G05_COMPACT_STEP0_FRAMES * G05_COMPACT_CHANNELS * 2 *
                    G05_COMPACT_GRID_SIDE,
                /*b=*/G05_COMPACT_PATCH_SIDE,
                /*n=*/G05_COMPACT_GRID_SIDE,
                /*c=*/G05_COMPACT_PATCH_SIDE, num_cores());
            // ② plane-major -> patch-major.
            rpu_launch_permute4d_0213_spm_kernel(
                layout_a, layout_b,
                /*q=*/G05_COMPACT_STEP0_FRAMES,
                /*b=*/G05_COMPACT_CHANNELS * 2,
                /*n=*/G05_COMPACT_PATCHES_PER_FRAME,
                /*c=*/G05_COMPACT_PATCH_SIDE * G05_COMPACT_PATCH_SIDE,
                num_cores());
            // ③ Reinterpret T×C before swapping to reproduce the official
            // [c0,c1,c0,c2,c1,c2] block order exactly.
            rpu_launch_permute4d_0213_spm_kernel(
                layout_b, layout_a,
                /*q=*/G05_COMPACT_STEP0_ROWS,
                /*b=*/2, /*n=*/G05_COMPACT_CHANNELS,
                /*c=*/G05_COMPACT_PATCH_SIDE * G05_COMPACT_PATCH_SIDE,
                num_cores());
            // ④ Raster rows -> 2x2 spatial-merge group row order.
            rpu_launch_permute4d_0213_spm_kernel(
                layout_a, layout_b,
                /*q=*/G05_COMPACT_STEP0_FRAMES * (G05_COMPACT_GRID_SIDE / 2),
                /*b=*/2, /*n=*/G05_COMPACT_GRID_SIDE / 2,
                /*c=*/2 * pd, num_cores());
            patch_input_addr = layout_b;
        } else {
            // Folded pixel rows are fresh per forward, so their address must be
            // patched at REPLAY. A fixed DMA would silently reuse forward #1.
            consume_manifest_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_VISION_STEP0_PIXEL_DMA_SITE,
                qwen35_vision_dma_route(
                    Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                invocation, {current_compact_step0_ ? 1 : 0});
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &pixel_src_base_,
                /*src_offset_bytes=*/off * pd * DWIDTH,
                /*num_elements=*/len * pd,
                addr(0, "s0_pixel"), num_cores());
            patch_input_addr = addr(0, "s0_pixel");
        }

        // patch_embed: col-parallel GEMM. m=len, n=h (split → h/8 per core), k=pd.
        // Dims are FULL/global here — the graph-layer op divides by tp internally.
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN35_VISION_STEP0_LINEAR_SITE,
            qwen35_vision_linear_route(), invocation);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            patch_input_addr, pe_w_, addr(0, "s0_patch"),
            len, h, pd, /*partition=*/1, num_cores(), addr(0, "s0_bias"),
            /*scale=*/{}, /*nvfp4_tensor_scale_spm_addr=*/0,
            /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);

        if (temporal_pos) {
            // Python packs position rows as [core,N,H/8]. Add each core's
            // matching col shard before gather; gather is a copy, so this is
            // byte-equivalent to the full-width add while avoiding 8 broadcasts.
            consume_manifest_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_VISION_STEP0_POS_SCATTER_DMA_SITE,
                qwen35_vision_dma_route(
                    Qwen35VisionMutableDmaRoute::DDR_SCATTER_TO_SPM),
                invocation);
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &step0_pos_src_base_,
                /*src_offset_bytes=*/off * local_h * DWIDTH,
                /*elements_per_core=*/len * local_h,
                /*core_stride_bytes=*/N * local_h * DWIDTH,
                addr(0, "s0_pos_emb"), num_cores());
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "s0_patch"), addr(0, "s0_pos_emb"), addr(0, "s0_patch"),
                len * local_h, ValuOpType::ADD, c10::Half(1.0), num_cores());
            const RpuAllGatherSchedule schedule =
                rpu_resolve_all_gather_schedule(local_h, DWIDTH);
            consume_manifest_route(
                FmbRouteFamily::COLLECTIVE,
                QWEN35_VISION_STEP0_TEMPORAL_ALL_GATHER_SITE,
                static_cast<int64_t>(schedule), invocation,
                {len, local_h, DWIDTH, num_cores()});
            rpu_launch_all_gather_spm_kernel(
                addr(0, "s0_patch"), addr(0, "s0_attn_in"),
                /*n=*/len, /*chunk_elems=*/local_h, /*dwidth=*/DWIDTH,
                num_cores(), schedule);
        } else {
            // Generic single-image path retains the full-width position broadcast.
            consume_manifest_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_VISION_STEP0_POS_BROADCAST_DMA_SITE,
                qwen35_vision_dma_route(
                    Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                invocation);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &step0_pos_src_base_,
                /*src_offset_bytes=*/off * h * DWIDTH,
                /*num_elements=*/len * h,
                addr(0, "s0_pos_emb"), num_cores());
            const RpuAllGatherSchedule schedule =
                rpu_resolve_all_gather_schedule(local_h, DWIDTH);
            consume_manifest_route(
                FmbRouteFamily::COLLECTIVE,
                QWEN35_VISION_STEP0_GENERIC_ALL_GATHER_SITE,
                static_cast<int64_t>(schedule), invocation,
                {len, local_h, DWIDTH, num_cores()});
            rpu_launch_all_gather_spm_kernel(
                addr(0, "s0_patch"), addr(0, "s0_attn_in"),
                /*n=*/len, /*chunk_elems=*/local_h, /*dwidth=*/DWIDTH,
                num_cores(), schedule);
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "s0_attn_in"), addr(0, "s0_pos_emb"), addr(0, "s0_attn_in"),
                len * h, ValuOpType::ADD, c10::Half(1.0), num_cores());
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
    const int64_t local_mh = mh / num_cores();         // 4096/8 = 512 — fc1 col output per core
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
            : camera * camera_patches;
        const int64_t merged_camera_base =
            camera * (camera_patches / QWEN3_5_SPATIAL_MERGE_UNIT);
        TORCH_CHECK(
            !merger_wrote_fusion_target_ || merger_dst_bases_[camera] != 0,
            "qwen3_5_vision merger camera destination is unset");
        for (int64_t off = 0; off < camera_patches; off += cs) {
            const int64_t len = std::min<int64_t>(cs, camera_patches - off);
            const int64_t m = len / mu;  // merged rows this chunk
            const int64_t invocation = off / cs;

            if (z2_bound_) {
                TORCH_CHECK(cameras == 1 && off == 0 &&
                                len == current_output_patches_,
                        "qwen3_5_vision Z2 merger requires one full input chunk");
                // The final tower all-reduce already left the complete
                // [256,1024] tensor at m_in's exact physical address.
            } else {
            // (1) 读回本 chunk 全宽输出 → 每核一份。FIXED DMA：output_tensor() 的地址
            //     是 per-shape registry-stable (fused_model_base.cpp:691)，烤地址安全。
            rpu_launch_ddr_broadcast_spm_dma(
                output_tensor().data_ptr<c10::Half>() +
                    (source_base + off) * h,
                len * h,
                addr(0, "m_in"), num_cores());
            }

            // (2) LayerNorm over hidden (真 LayerNorm，eps 在方差内，无 skip)。gamma/beta
            //     由 preload broadcast 到所有核。
            rpu_launch_layernorm_spm_kernel(
                addr(0, "m_in"), addr(0, "m_in"),
                addr(0, "m_norm_g"), addr(0, "m_norm_b"),
                len, h, merger_eps, /*has_skip=*/false, /*skip=*/0, num_cores());

            // (3) reinterpret [len, hidden] → [m, hidden*4]（只改 M/K，字节等价）+ fc1
            //     COL 线性 (partition=1)。⚠️ fc1 必 col、fc2 必 row，两者维度对 col/row
            //     约束双满足、assert 拦不住，只能靠角色手工钉死。
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_MERGER_FC1_SITE,
                qwen35_vision_linear_route(), invocation);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "m_in"), merger_fc1_w_, addr(0, "m_fc1"),
                /*M=*/m, /*N=*/mh, /*K=*/mh, /*partition=*/1, num_cores(),
                addr(0, "m_fc1_bias"),
                /*scale=*/{}, /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);

            // (4) HF nn.GELU() default: exact erf, per-core on the col shard.
            consume_g05_high_precision_merger_activation(m, invocation);
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "m_fc1"), addr(0, "m_fc1"),
                m * local_mh, ValuOpType::ADD,
                gelu_erf_ultra_ ? GeluMode::ERF_ULTRA : GeluMode::ERF,
                num_cores(), high_precision_ ? RpuUnaryPrecision::HIGH : RpuUnaryPrecision::BASE);

            // (5) fc2 ROW 线性 (partition=0)。输入是 fc1 的 col shard（正好 = 本核的
            //     K-slice），产每核部分和。fc2 bias core-0-only（行约定，reduce 后计一次）。
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_MERGER_FC2_SITE,
                qwen35_vision_linear_route(), invocation);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "m_fc1"), merger_fc2_w_, addr(0, "m_fc2"),
                /*M=*/m, /*N=*/oh, /*K=*/mh, /*partition=*/0, num_cores(),
                addr(0, "m_fc2_bias"),
                /*scale=*/{}, /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);

            // (6) 清零 residual buffer：all_reduce_sum_residual 必须有真 residual 地址
            //     （不能传 0），merger 无 residual → 用 graph-aware fill 灌零。
            rpu_launch_fill_spm_kernel(
                addr(0, "m_zero"), m * oh, c10::Half(0.0f), num_cores());

            // (7) Ring-reduce row-partition partials and add the zero residual.
            //     input, residual, and output are three independent buffers.
            const uint32_t merger_out =
                z2_bound_ ? z2_slice_addr_ : addr(0, "m_out");
            consume_manifest_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN35_VISION_MERGER_ALL_REDUCE_SITE,
                qwen35_vision_ring_route(m, oh, num_cores()), invocation);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "m_fc2"), addr(0, "m_zero"), merger_out,
                m, oh, num_cores(), num_cores());

            // (8) Camera-major temporal output keeps each last frame in a separate
            //     source slice. Standalone packs those slices contiguously; direct
            //     fusion writes each one to its independently mutable text run.
            if (z2_bound_) {
                continue;
            }
            uint64_t* destination_base = merger_wrote_fusion_target_
                ? &merger_dst_bases_[camera] : &merger_dst_bases_[0];
            const int64_t destination_row = merger_wrote_fusion_target_
                ? off / mu : merged_camera_base + off / mu;
            consume_manifest_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_VISION_MERGER_OUTPUT_DMA_SITE,
                qwen35_vision_dma_route(
                    Qwen35VisionMutableDmaRoute::SPM_COPY_TO_DDR),
                invocation);
            rpu_launch_spm_copy_ddr_dma_mutable(
                addr(0, "m_out"), destination_base,
                destination_row * oh * sizeof(c10::Half), m * oh);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LAYER BUILD (KV_FIRST two-phase)
// ─────────────────────────────────────────────────────────────────────────────

// ── emit_kv_first_body: KV_FIRST Phase 1 — LN1 + Q/K/V Linear + 2D rope
//    (pos_offset=cur_pos) + KV insert(@cur_pos 累积) + 把 rope 后的 Q 存进
//    q_ddr_buf_（照 rpu_gemma_model.cpp:762-836 的两相拆分）。 ───────────────────
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
    const int64_t local_q_dim = (nq / num_cores()) * hd;
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
                1, frames, patch_groups, group * h, num_cores());

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
                    camera_rows, h, eps_, false, 0, num_cores());
                consume_manifest_route(
                    FmbRouteFamily::LINEAR, QWEN35_VISION_TEMPORAL_Q_SITE,
                    qwen35_vision_linear_route());
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    temporal_hidden, lw.q_w,
                    addr(0, "t_q_stage") + stage_delta,
                    camera_rows, nq * hd, h, 1, num_cores(),
                    layer_addr(layer_idx, 0, "q_bias"),
                    /*scale=*/lw.q_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
                    /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
                consume_manifest_route(
                    FmbRouteFamily::LINEAR, QWEN35_VISION_TEMPORAL_K_SITE,
                    qwen35_vision_linear_route());
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    temporal_hidden, lw.k_w,
                    addr(0, "t_k_stage") + stage_delta,
                    camera_rows, nq * hd, h, 1, num_cores(),
                    layer_addr(layer_idx, 0, "k_bias"),
                    /*scale=*/lw.k_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
                    /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
                consume_manifest_route(
                    FmbRouteFamily::LINEAR, QWEN35_VISION_TEMPORAL_V_SITE,
                    qwen35_vision_linear_route());
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    temporal_hidden, lw.v_w, addr(0, "t_v"),
                    camera_rows, nq * hd, h, 1, num_cores(),
                    layer_addr(layer_idx, 0, "v_bias"),
                    /*scale=*/lw.v_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
                    /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);

                const KvInsertSegmentPlan kv_plan = [&] {
                    if (!ctx().has_complete_physical_manifest()) {
                        TORCH_CHECK(
                            z2_bound_,
                            "Qwen3.5 temporal Vision KV insert requires a "
                            "COMPLETE physical descriptor");
                        return rpu_resolve_kvinsert_segment_plan_auto(
                            /*position=*/0, frames, frames, num_cores(),
                            composite_heads, hd,
                            QWEN35_VISION_KV_CAPABILITIES);
                    }
                    const auto& route = ctx().find_physical_route(
                        FmbRouteFamily::KV_INSERT,
                        QWEN35_VISION_TEMPORAL_KV_SITE);
                    KvInsertSegmentPlan resolved_plan =
                        restore_kvinsert_plan(
                            QWEN35_VISION_TEMPORAL_KV_SITE,
                            route.arguments, num_cores(), composite_heads, hd);
                    TORCH_CHECK(
                        resolved_plan.logical_rows() == frames &&
                            resolved_plan.physical_rows() == frames &&
                            resolved_plan.segment(0).position == 0,
                        "Qwen3.5 temporal Vision KV descriptor geometry drift");
                    ctx().consume_physical_route(
                        FmbRouteFamily::KV_INSERT,
                        QWEN35_VISION_TEMPORAL_KV_SITE,
                        static_cast<int64_t>(resolved_plan.route()),
                        QWEN35_VISION_KV_REASON_DDR_REQUIRED,
                        route.arguments);
                    return resolved_plan;
                }();
                rpu_launch_insert_kvcache_spm_unified_with_plan(
                    temporal_k_cache_, temporal_v_cache_,
                    addr_offset("t_k_stage").value + stage_delta,
                    addr_offset("t_v").value,
                    composite_heads, hd, num_cores(),
                    /*k_cache_batch_offset_elems=*/0,
                    /*v_cache_batch_offset_elems=*/0,
                    /*spm_rows=*/0, kv_plan);
                if (ctx().has_complete_physical_manifest()) {
                    const std::vector<int64_t> arguments{
                        /*batch=*/1, frames, frames, composite_heads,
                        composite_heads, hd, num_cores(), /*MASK_LTM=*/1};
                    ctx().consume_physical_route(
                        FmbRouteFamily::ATTENTION,
                        QWEN35_VISION_TEMPORAL_ATTN_SITE,
                        static_cast<int64_t>(
                            AttentionExecutionPolicy::DDR_KV),
                        QWEN35_VISION_ATTN_TEMPORAL_PACKED_KV_DDR_REQUIRED,
                        arguments);
                }
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    temporal_k_cache_, temporal_v_cache_,
                    1 /*MASK_LTM*/, 1.0 / std::sqrt(static_cast<double>(hd)),
                    addr_offset("t_q_stage").value + stage_delta,
                    addr_offset("t_out_stage").value + stage_delta,
                    addr_offset("t_sdpa_tmp").value, 0,
                    frames, composite_heads, composite_heads, hd, frames,
                    num_cores(), num_cores());
            }

            rpu_launch_permute3d_spm_kernel(
                addr(0, "t_q_stage"), addr(0, "q"),
                patch_groups, frames, permute_inner, {1, 0, 2}, num_cores());
            rpu_launch_permute3d_spm_kernel(
                addr(0, "t_k_stage"), addr(0, "k"),
                patch_groups, frames, permute_inner, {1, 0, 2}, num_cores());
            rpu_launch_permute3d_spm_kernel(
                addr(0, "t_out_stage"), addr(0, "v"),
                patch_groups, frames, permute_inner, {1, 0, 2}, num_cores());

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
    const int64_t local_q_dim = (nq / num_cores()) * hd;
    const int64_t local_inter = is_ / num_cores();
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

    const int64_t local_heads = nq / num_cores();
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
        const uint32_t v_addr = addr(0, "v") + staged_delta;
        const uint32_t q_off = addr_offset("q").value + staged_delta;
        const uint32_t k_off = addr_offset("k").value + staged_delta;
        const uint32_t v_off = addr_offset("v").value + staged_delta;

        if (!use_staged_qkv) {
            rpu_launch_layernorm_spm_kernel(
                addr(0, "residual1"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "ln1_gamma"),
                layer_addr(layer_idx, 0, "ln1_beta"),
                seq_len, h, eps_, false, 0, num_cores());
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_SPATIAL_Q_SITE,
                qwen35_vision_linear_route());
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.q_w, addr(0, "q"),
                seq_len, nq * hd, h, 1, num_cores(),
                layer_addr(layer_idx, 0, "q_bias"),
                /*scale=*/lw.q_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_SPATIAL_K_SITE,
                qwen35_vision_linear_route());
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.k_w, addr(0, "k"),
                seq_len, nq * hd, h, 1, num_cores(),
                layer_addr(layer_idx, 0, "k_bias"),
                /*scale=*/lw.k_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
            consume_manifest_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_SPATIAL_V_SITE,
                qwen35_vision_linear_route());
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.v_w, addr(0, "v"),
                seq_len, nq * hd, h, 1, num_cores(),
                layer_addr(layer_idx, 0, "v_bias"),
                /*scale=*/lw.v_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
        }

        consume_manifest_route(
            FmbRouteFamily::ROPE, QWEN35_VISION_SPATIAL_Q_ROPE_SITE,
            qwen35_vision_rope_route());
        rpu_launch_rope_2d_spm_kernel(
            q_addr, q_addr,
            addr(0, "freq_cos_spm"), addr(0, "freq_sin_spm"),
            position_idx_keepalive_.data_ptr<int16_t>(),
            offset, seq_len, local_heads, hd, hd, num_cores());
        consume_manifest_route(
            FmbRouteFamily::ROPE, QWEN35_VISION_SPATIAL_K_ROPE_SITE,
            qwen35_vision_rope_route());
        rpu_launch_rope_2d_spm_kernel(
            k_addr, k_addr,
            addr(0, "freq_cos_spm"), addr(0, "freq_sin_spm"),
            position_idx_keepalive_.data_ptr<int16_t>(),
            offset, seq_len, local_heads, hd, hd, num_cores());

        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const KvInsertSegmentPlan kv_plan = [&] {
            if (!ctx().has_complete_physical_manifest()) {
                TORCH_CHECK(
                    z2_bound_,
                    "Qwen3.5 spatial Vision KV insert requires a COMPLETE "
                    "physical descriptor");
                return rpu_resolve_kvinsert_segment_plan_auto(
                    /*position=*/0, seq_len, seq_len, num_cores(), nq, hd,
                    QWEN35_VISION_KV_CAPABILITIES);
            }
            const auto& route = ctx().find_physical_route(
                FmbRouteFamily::KV_INSERT,
                QWEN35_VISION_SPATIAL_KV_SITE);
            KvInsertSegmentPlan resolved_plan =
                restore_kvinsert_plan(
                    QWEN35_VISION_SPATIAL_KV_SITE,
                    route.arguments, num_cores(), nq, hd);
            TORCH_CHECK(
                resolved_plan.logical_rows() == seq_len &&
                    resolved_plan.physical_rows() == seq_len &&
                    resolved_plan.segment(0).position == 0,
                "Qwen3.5 spatial Vision KV descriptor geometry drift");
            ctx().consume_physical_route(
                FmbRouteFamily::KV_INSERT,
                QWEN35_VISION_SPATIAL_KV_SITE,
                static_cast<int64_t>(resolved_plan.route()),
                QWEN35_VISION_KV_REASON_DDR_REQUIRED,
                route.arguments);
            return resolved_plan;
        }();
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache, k_off, v_off,
            nq, hd, num_cores(),
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            /*spm_rows=*/0, kv_plan);
        const bool raw_spm = ctx().attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const int64_t attention_site = raw_spm
            ? QWEN35_VISION_SPATIAL_RAW_SPM_ATTN_SITE
            : QWEN35_VISION_SPATIAL_ATTN_SITE;
        const FmbRouteManifestEntry* attention_route = nullptr;
        std::vector<int64_t> attention_arguments;
        if (ctx().has_complete_physical_manifest()) {
            attention_route = &ctx().find_physical_route(
                FmbRouteFamily::ATTENTION, attention_site);
            attention_arguments = {
                /*batch=*/3, temporal_patches_per_frame_,
                temporal_patches_per_frame_, nq, nq, hd, num_cores(),
                /*MASK_NONE=*/0};
            TORCH_CHECK(
                (raw_spm && attention_route->flags == 0) ||
                    (!raw_spm && attention_route->flags != 0),
                "Qwen3.5 spatial attention descriptor lacks its exact "
                "RAW_SPM or DDR_REQUIRED reason");
        }
        if (raw_spm) {
            if (attention_route != nullptr) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION,
                    QWEN35_VISION_SPATIAL_RAW_SPM_ATTN_SITE,
                    static_cast<int64_t>(
                        AttentionExecutionPolicy::SPM_KV_BY_MHA),
                    attention_route->flags, attention_arguments);
            }
            TORCH_CHECK(
                sdpa_by_mha_spm_is_valid(
                    /*batch=*/3, temporal_patches_per_frame_,
                    temporal_patches_per_frame_, nq, nq, hd, num_cores(),
                    /*MASK_NONE=*/0),
                "Qwen3.5 spatial RAW_SPM attention escaped its exact "
                "three-frame capability");
            rpu_launch_v_transpose_spm(
                v_addr, addr(0, "sdpa_tmp"),
                /*batch=*/3, temporal_patches_per_frame_, nq, hd,
                num_cores());
            rpu_launch_sdpa_by_mha_spm(
                q_addr, k_addr, addr(0, "sdpa_tmp"),
                addr(0, "sdpa_out"), /*mask_spm=*/0, /*MASK_NONE=*/0,
                1.0 / std::sqrt(static_cast<double>(hd)),
                /*batch=*/3, temporal_patches_per_frame_,
                temporal_patches_per_frame_, nq, nq, hd, num_cores());
        } else {
            if (attention_route != nullptr) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION,
                    QWEN35_VISION_SPATIAL_ATTN_SITE,
                    static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                    attention_route->flags, attention_arguments);
            }
            rpu_launch_sdpa_spm_minibatch_kernel(
                k_cache, v_cache,
                0 /*MASK_NONE*/, 1.0 / std::sqrt(static_cast<double>(hd)),
                q_off,
                addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value, 0,
                seq_len, nq, nq, hd,
                temporal_patches_per_frame_, 3,
                num_cores(), num_cores());
        }

        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN35_VISION_SPATIAL_O_SITE,
            qwen35_vision_linear_route());
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, num_cores(),
            layer_addr(layer_idx, 0, "o_bias"),
            /*scale=*/lw.o_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
            /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_VISION_SPATIAL_ATTN_ALL_REDUCE_SITE,
            qwen35_vision_ring_route(seq_len, h, num_cores()));
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
            seq_len, h, num_cores(), num_cores());
        rpu_launch_layernorm_spm_kernel(
            addr(0, "input_norm"), addr(0, "oproj"),
            layer_addr(layer_idx, 0, "ln2_gamma"),
            layer_addr(layer_idx, 0, "ln2_beta"),
            seq_len, h, eps_, false, 0, num_cores());
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN35_VISION_SPATIAL_FC1_SITE,
            qwen35_vision_linear_route());
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "oproj"), lw.fc1_w, addr(0, "fc1"),
            seq_len, is_, h, 1, num_cores(),
            layer_addr(layer_idx, 0, "fc1_bias"),
            /*scale=*/lw.fc1_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
            /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "fc1"), addr(0, "fc1"),
            seq_len * local_inter, ValuOpType::ADD,
            GeluMode::TANH, num_cores());
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN35_VISION_SPATIAL_FC2_SITE,
            qwen35_vision_linear_route());
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "fc1"), lw.fc2_w, addr(0, "oproj"),
            seq_len, h, is_, 0, num_cores(),
            layer_addr(layer_idx, 0, "fc2_bias"),
            /*scale=*/lw.fc2_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
            /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_VISION_SPATIAL_MLP_ALL_REDUCE_SITE,
            qwen35_vision_ring_route(seq_len, h, num_cores()));
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"),
            seq_len, h, num_cores(), num_cores());

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
    if (is_g05_k1_b3_profile()) { consume_g05_k1_b3_schedule(chunk,false); return; }
    if (is_g05_k1_camera_group_profile()) { consume_g05_k1_camera_group_schedule(chunk,false); return; }
    if (current_temporal_num_frames_ > 1) {
        if (chunk.idx == 0) emit_temporal_layer(layer_idx);
        return;
    }
    const bool carry_qkv = consume_g05_qkv_spm_transport(chunk, false);
    const bool combined_qkv = consume_g05_qkv_linear(chunk, carry_qkv);
    const auto& lw = layer_weights_[layer_idx];
    int64_t seq_len = chunk.len;
    int64_t cur_pos = ctx().position + chunk.offset;   // vision position=0 ⇒ =chunk.offset
    int64_t h  = hidden_size();
    int64_t nq = num_q_heads();
    int64_t hd = head_dim();
    const int64_t real_hd = orig_head_dim_;
    const int tp = attention_tp_;
    const int64_t local_heads   = nq / tp;
    const int64_t head_dim_pad  = hd;

    // Phase 1: input DMA + LN1
    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }
    rpu_launch_layernorm_spm_kernel(
        addr(0, "residual1"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "ln1_gamma"), layer_addr(layer_idx, 0, "ln1_beta"),
        seq_len, h, eps_, false, 0, num_cores());

    // Phase 2: Q / K / V Linear (+bias). The triple-base route retains each
    // projection's independent FP16 output and bias before the existing RoPE.
    if (combined_qkv && g05_w8a16_) {
        rpu_launch_g05_vision_qkv_linear_w8a16_acc32_kernel(
            addr(0, "input_norm"), lw.q_w, lw.k_w, lw.v_w,
            addr(0, "q"), addr(0, "k"), addr(0, "v"),
            layer_addr(layer_idx, 0, "q_bias"), layer_addr(layer_idx, 0, "k_bias"),
            layer_addr(layer_idx, 0, "v_bias"), lw.q_ws, lw.k_ws, lw.v_ws,
            linear_acc32_);
    } else if (combined_qkv) {
        rpu_launch_g05_vision_qkv_linear_fp16_acc32_kernel(
            addr(0, "input_norm"), lw.q_w, lw.k_w, lw.v_w,
            addr(0, "q"), addr(0, "k"), addr(0, "v"),
            layer_addr(layer_idx, 0, "q_bias"), layer_addr(layer_idx, 0, "k_bias"),
            layer_addr(layer_idx, 0, "v_bias"));
    } else {
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN35_VISION_Q_SITE,
            qwen35_vision_linear_route(), chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, tp, layer_addr(layer_idx, 0, "q_bias"),
            /*scale=*/lw.q_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
            /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN35_VISION_K_SITE,
            qwen35_vision_linear_route(), chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, tp, layer_addr(layer_idx, 0, "k_bias"),
            /*scale=*/lw.k_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
            /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
        consume_manifest_route(
            FmbRouteFamily::LINEAR, QWEN35_VISION_V_SITE,
            qwen35_vision_linear_route(), chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, tp, layer_addr(layer_idx, 0, "v_bias"),
            /*scale=*/lw.v_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
            /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
    }

    // Phase 2.5: 2D RoPE on Q,K — pos_offset=cur_pos, num_tokens=本 chunk 行数.
    // SPM variant: cos/sin read from the SPM-resident tables populated by
    // emit_preload_weights (the retained graph replays that preload), not DDR.
    // position_idx stays in DDR. Same broadcast/addr(0,...) convention as x/y.
    consume_manifest_route(
        FmbRouteFamily::ROPE, QWEN35_VISION_Q_ROPE_SITE,
        qwen35_vision_rope_route(), chunk.idx);
    rpu_launch_rope_2d_spm_kernel(
        addr(0, "q"), addr(0, "q"),
        addr(0, "freq_cos_spm"), addr(0, "freq_sin_spm"),
        position_idx_keepalive_.data_ptr<int16_t>(),
        cur_pos, seq_len, local_heads, real_hd, head_dim_pad, tp);
    consume_manifest_route(
        FmbRouteFamily::ROPE, QWEN35_VISION_K_ROPE_SITE,
        qwen35_vision_rope_route(), chunk.idx);
    rpu_launch_rope_2d_spm_kernel(
        addr(0, "k"), addr(0, "k"),
        addr(0, "freq_cos_spm"), addr(0, "freq_sin_spm"),
        position_idx_keepalive_.data_ptr<int16_t>(),
        cur_pos, seq_len, local_heads, real_hd, head_dim_pad, tp);

    // Phase 3a: KV cache insert at ABSOLUTE position (累积；单 chunk 时 cur_pos=0 等价旧行为)
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    const KvInsertSegmentPlan kv_plan = [&] {
        if (!ctx().has_complete_physical_manifest()) {
            TORCH_CHECK(
                z2_bound_,
                "Qwen3.5 Vision KV insert requires a COMPLETE physical "
                "descriptor");
            return rpu_resolve_kvinsert_segment_plan_auto(
                cur_pos, seq_len, seq_len, tp, nq, hd,
                QWEN35_VISION_KV_CAPABILITIES);
        }
        const auto& route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, QWEN35_VISION_KV_SITE,
            chunk.idx);
        KvInsertSegmentPlan resolved_plan =
            restore_kvinsert_plan(
                QWEN35_VISION_KV_SITE, route.arguments, tp, nq, hd);
        TORCH_CHECK(
            resolved_plan.logical_rows() == seq_len &&
                resolved_plan.physical_rows() == seq_len &&
                resolved_plan.segment(0).position == cur_pos,
            "Qwen3.5 Vision KV descriptor geometry drift at invocation ",
            chunk.idx);
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, QWEN35_VISION_KV_SITE,
            static_cast<int64_t>(resolved_plan.route()),
            carry_qkv ? G05_VISION_KV_RAW_SPM_ONLY : QWEN35_VISION_KV_REASON_DDR_REQUIRED,
            route.arguments, chunk.idx);
        return resolved_plan;
    }();
    if (!carry_qkv) {
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache,
            addr_offset("k").value, addr_offset("v").value,
            nq, hd, tp,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            /*spm_rows=*/0, kv_plan);
    }

    // 存 rope 后的 Q 到 q_ddr_buf_（position-indexed，照 gemma:823-836）
    TORCH_CHECK(q_ddr_buf_.defined(),
                "Qwen3_5VisionModel: q_ddr_buf_ not allocated in KV_FIRST mode");
    int64_t local_q_dim         = local_heads * hd;
    int64_t q_local_elems       = seq_len * local_q_dim;
    c10::Half* q_ddr_base       = q_ddr_buf_.data_ptr<c10::Half>();
    int64_t q_elem_offset       = chunk.offset * local_q_dim;
    int64_t q_core_stride_bytes = q_ddr_buf_.size(1) * local_q_dim * DWIDTH;
    if (!carry_qkv) {
        rpu_launch_spm_scatter_ddr_dma(
            addr(0, "q"), q_ddr_base + q_elem_offset,
            q_local_elems, q_core_stride_bytes, tp);
    }

    // DEBUG: snapshot rope'd Q per-layer per-core → dbg_q_[layer, core, chunk.offset:, :].
    // Same live "q" SPM source as the q_ddr_buf_ store above. Core c writes to
    // base + c*(N*local_q_dim); rows offset by chunk.offset (multi-chunk fills all rows).
    if (get_debug_export() && dbg_q_enabled()) {
        const int64_t N = current_num_patches_;
        c10::Half* dq = dbg_q_.data_ptr<c10::Half>()
                      + ((int64_t)layer_idx * tp * N + chunk.offset) * local_q_dim;
        rpu_launch_spm_scatter_ddr_dma(
            addr(0, "q"), dq, q_local_elems,
            /*core_stride_bytes=*/N * local_q_dim * DWIDTH, tp);
    }
}

// ── build_layer_subgraph: KV_FIRST Phase 2 — Phase 1 已做 LN/QKV/rope/存 Q 并插满
//    全量 KV。这里：重读 input(残差用) + 载回本 chunk 的 rope 后 Q + SDPA(读全量 KV) + MLP。──
void Qwen3_5VisionModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    if (is_g05_k1_b3_profile()) { emit_g05_k1_b3_layer(layer_idx,chunk); return; }
    if (is_g05_k1_camera_group_profile()) { emit_g05_k1_camera_group_layer(layer_idx,chunk); return; }
    if (current_temporal_num_frames_ > 1) return;
    const bool carry_qkv = consume_g05_qkv_spm_transport(chunk, true);
    const auto& lw = layer_weights_[layer_idx];
    int64_t seq_len = chunk.len;
    int64_t h = hidden_size();
    int64_t nq = num_q_heads();
    int64_t hd = head_dim();
    int64_t is_ = intermediate_size();
    const int tp = attention_tp_;
    int64_t local_inter = is_ / num_cores();

    // KV_FIRST Phase 2。Phase 1(emit_kv_first_body) 已做 LN/QKV/rope/存 Q，并把全量 KV 插满。
    // 这里：重读 input(残差用) + 载回本 chunk 的 rope 后 Q + SDPA(读全量 KV) + MLP。
    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }

    // 载回本 chunk 的 rope 后 Q(从 q_ddr_buf_ 的 chunk.offset 处)到 SPM "q_comp"(照 gemma:863-876)
    const int64_t local_heads = nq / tp;
    int64_t local_q_dim   = local_heads * hd;
    int64_t q_local_elems = seq_len * local_q_dim;
    c10::Half* q_ddr_base = q_ddr_buf_.data_ptr<c10::Half>();
    int64_t q_elem_offset = chunk.offset * local_q_dim;
    int64_t q_core_stride = q_ddr_buf_.size(1) * local_q_dim * DWIDTH;
    if (!carry_qkv) {
        rpu_launch_ddr_scatter_spm_dma(
            q_ddr_base + q_elem_offset, q_local_elems, q_core_stride,
            addr(0, "q_comp"), tp);
    }

    // SDPA：query = 本 chunk 的 q_comp(seq_len 行)，K/V = cache 全量(current_num_patches_) → 双向。
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    double attn_scale = 1.0 / std::sqrt(static_cast<double>(orig_head_dim_));
    const bool raw_spm = ctx().attention_policy ==
        AttentionExecutionPolicy::SPM_KV_BY_MHA;
    const int64_t attention_site = raw_spm
        ? QWEN35_VISION_RAW_SPM_ATTN_SITE
        : QWEN35_VISION_ATTN_SITE;
    const FmbRouteManifestEntry* attention_route = nullptr;
    std::vector<int64_t> attention_arguments;
    if (ctx().has_complete_physical_manifest()) {
        attention_route = &ctx().find_physical_route(
            FmbRouteFamily::ATTENTION, attention_site, chunk.idx);
        attention_arguments = {
            /*batch=*/1, seq_len, current_num_patches_, nq, nq, hd, tp,
            /*MASK_NONE=*/0};
        TORCH_CHECK(
            (raw_spm && attention_route->flags == 0) ||
                (!raw_spm && attention_route->flags != 0),
            "Qwen3.5 Vision attention descriptor lacks its exact RAW_SPM "
            "or DDR_REQUIRED reason");
    }
    if (raw_spm) {
        if (attention_route != nullptr) {
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                QWEN35_VISION_RAW_SPM_ATTN_SITE,
                static_cast<int64_t>(
                    AttentionExecutionPolicy::SPM_KV_BY_MHA),
                attention_route->flags, attention_arguments, chunk.idx);
        }
        TORCH_CHECK(
            sdpa_by_mha_spm_is_valid(
                /*batch=*/1, seq_len, current_num_patches_, nq, nq, hd,
                tp, /*MASK_NONE=*/0),
            "Qwen3.5 Vision RAW_SPM attention escaped its exact P0 "
            "single-chunk capability");
        rpu_launch_v_transpose_spm(
            addr(0, "v"), addr(0, "sdpa_tmp"),
            /*batch=*/1, current_num_patches_, nq, hd, tp);
        rpu_launch_sdpa_by_mha_spm(
            addr(0, "q_comp"), addr(0, "k"), addr(0, "sdpa_tmp"),
            addr(0, "sdpa_out"), /*mask_spm=*/0, /*MASK_NONE=*/0,
            attn_scale, /*batch=*/1, seq_len, current_num_patches_,
            nq, nq, hd, tp);
    } else {
        if (attention_route != nullptr) {
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION, QWEN35_VISION_ATTN_SITE,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                attention_route->flags, attention_arguments, chunk.idx);
        }
        rpu_launch_sdpa_spm_unified_kernel_v2(
            k_cache, v_cache,
            0 /*MASK_NONE*/, attn_scale,
            addr_offset("q_comp").value,
            addr_offset("sdpa_out").value,
            addr_offset("sdpa_tmp").value, 0,
            seq_len, nq, nq, hd,
            current_num_patches_, tp, /*physical_kv_cores=*/8);
    }

    // Phase 4: O_proj (row-partition, with bias) + AllReduce + Residual.
    consume_manifest_route(
        FmbRouteFamily::ALL_REDUCE,
        QWEN35_VISION_PREPARE_ALL_REDUCE_SITE,
        static_cast<int64_t>(
            Qwen35VisionAllReduceRoute::PREPARE_RING_INPUT),
        chunk.idx);
    rpu_prepare_ring_all_reduce_input(
        addr(0, "oproj"), seq_len, h, tp, num_cores());
    consume_manifest_route(
        FmbRouteFamily::LINEAR, QWEN35_VISION_O_SITE,
        qwen35_vision_linear_route(), chunk.idx);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, 0, tp,
        layer_addr(layer_idx, 0, "o_bias"),
        /*scale=*/lw.o_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
        /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
    // Exact G05 K1 keeps the raw residual compact until the MLP companion.
    // The three full-size BufferDecl roots retain their existing lifetimes.
    const bool owner_ln = use_g05_owner_ln(seq_len);
    if (owner_ln) {
        launch_g05_owner_ln(layer_idx, chunk);
    } else {
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_VISION_ATTN_ALL_REDUCE_SITE,
            qwen35_vision_ring_route(seq_len, h, num_cores()), chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
            seq_len, h, tp, num_cores());
        rpu_launch_layernorm_spm_kernel(
            addr(0, "input_norm"), addr(0, "oproj"),
            layer_addr(layer_idx, 0, "ln2_gamma"), layer_addr(layer_idx, 0, "ln2_beta"),
            seq_len, h, eps_, false, 0, num_cores());
    }

    // Phase 5: fc1+bias + GELU (gelu_pytorch_tanh per Q3.5 config).
    consume_manifest_route(
        FmbRouteFamily::LINEAR, QWEN35_VISION_FC1_SITE,
        qwen35_vision_linear_route(), chunk.idx);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "oproj"), lw.fc1_w, addr(0, "fc1"),
        seq_len, is_, h, 1, num_cores(),
        layer_addr(layer_idx, 0, "fc1_bias"),
        /*scale=*/lw.fc1_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
        /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
    consume_g05_high_precision_encoder_activation(seq_len, chunk.idx);
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "fc1"), addr(0, "fc1"),
        seq_len * local_inter, ValuOpType::ADD,
        GeluMode::TANH, num_cores(),
        high_precision_ ? RpuUnaryPrecision::HIGH : RpuUnaryPrecision::BASE);

    // Phase 6: fc2 + AllReduce + Residual (writes back to residual1 in
    // SPM_RESIDENT mode so the next layer's Phase 1 LN1 reads directly).
    consume_manifest_route(
        FmbRouteFamily::LINEAR, QWEN35_VISION_FC2_SITE,
        qwen35_vision_linear_route(), chunk.idx);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "fc1"), lw.fc2_w, addr(0, "oproj"),
        seq_len, h, is_, 0, num_cores(),
        layer_addr(layer_idx, 0, "fc2_bias"),
        /*scale=*/lw.fc2_ws, /*nvfp4_tensor_scale_spm_addr=*/0,
        /*nvfp4_layer_id=*/0, /*force_acc32=*/linear_acc32_);
    if (owner_ln) {
        launch_g05_compact_residual(chunk);
    } else {
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_VISION_MLP_ALL_REDUCE_SITE,
            qwen35_vision_ring_route(seq_len, h, num_cores()), chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"),
            seq_len, h, num_cores(), num_cores());
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
    // fences no-ops, so this injects ZERO barriers (unlike dbg_q, which spans the active cores — see
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
    const int tp = attention_tp_;

    int64_t local_q_dim = (nq / tp) * hd;
    int64_t local_inter = is_ / num_cores();
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
    const bool raw_spm = ctx.attention_policy ==
        AttentionExecutionPolicy::SPM_KV_BY_MHA;

    const bool carry_qkv = use_g05_qkv_spm_layout(ctx);
    int64_t res  = A(cs * h * DWIDTH);
    int64_t qkv  = A(cs * local_q_dim * DWIDTH);
    int64_t raw_kv = A(Align(cs, static_cast<int64_t>(16)) *
                       local_q_dim * DWIDTH);
    int64_t fc1  = A(cs * local_inter * DWIDTH);

    SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                        hd, /*nq*/nq, /*nkv*/nq,
                        /*cores*/tp, /*mask*/0};
    SdpaTiling t = sdpa_compute_tiling(sdpa_cfg, cs);
    int64_t nkv_per_core = CeilDiv(nq, (int64_t)tp);
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
    int64_t rope_tbl_elems = freq_cos_.defined()
        ? freq_cos_.numel()
        : max_hw_ * (orig_head_dim_ / 4);
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
    const int64_t cs0 =
        input_chunk_size_for_geometry(ctx.max_kv_seq_len);
    std::vector<BufferDecl> step0;
    if (has_step0_) {
        const int64_t s0_pos_width = temporal_pos ? h / num_cores() : h;
        if (current_compact_step0_) {
            // Two exact packed-row ping-pong roots dominate compact STEP0:
            // 2*2,359,296 + 2*196,608 + 512 = 5,112,320 B/core. The temporal
            // layer peak remains larger (6,881,280 B), before the shared
            // 386,048-B persistent region.
            const int64_t layout = A(cs0 * patch_dim_ * DWIDTH);
            step0 = {
                {"s0_layout_a", layout, 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_layout_b", layout, 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_patch", A(cs0 * (h / num_cores()) * DWIDTH), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_pos_emb", A(cs0 * s0_pos_width * DWIDTH), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_attn_in", 0, 1, 1, StorageClass::Temp, 0, "s0_layout_a", OUTSIDE},
                {"s0_bias", dma_safe(h / num_cores()), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
            };
        } else {
            const int64_t s0_io = A(cs0 * std::max(patch_dim_, h) * DWIDTH);
            step0 = {
                {"s0_pixel",   s0_io,                            1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_patch",   A(cs0 * (h / num_cores()) * DWIDTH), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_pos_emb", A(cs0 * s0_pos_width * DWIDTH),  1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
                {"s0_attn_in", 0, 1, 1, StorageClass::Temp, 0, "s0_pixel", OUTSIDE},
                {"s0_bias",    dma_safe(h / num_cores()), 1, 1, StorageClass::Temp, 0, nullptr, OUTSIDE},
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
        const int64_t local_mh = mh / num_cores();                       // fc1 col shard width
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
    if (is_g05_k1_b3_profile() || is_g05_k1_camera_group_profile()) {
        const int64_t camera_full_bytes=current_camera_batch_count_*524288;
        const int64_t camera_fc1_bytes=current_camera_batch_count_*262144;
        // Independent residual/MLP slices, shared 256-row attention scratch.
        // All are alive throughout the layer callback: no cross-camera alias.
        decls = {
            {"residual1",camera_full_bytes,0,6,StorageClass::Temp,0,nullptr,ALL},
            {"input_norm",camera_full_bytes,0,6,StorageClass::Temp,0,nullptr,ALL},
            {"oproj",camera_full_bytes,0,6,StorageClass::Temp,0,nullptr,ALL},
            {"fc1",camera_fc1_bytes,0,6,StorageClass::Temp,0,nullptr,ALL},
            {"q",65536,0,6,StorageClass::Temp,0,nullptr,ALL},
            {"k",65536,0,6,StorageClass::Temp,0,nullptr,ALL},
            {"v",65536,0,6,StorageClass::Temp,0,nullptr,ALL},
            {"q_comp",0,0,6,StorageClass::Temp,0,"q",ALL},
            {"sdpa_out",65536,0,6,StorageClass::Temp,0,nullptr,ALL},
            {"sdpa_tmp",131072,0,6,StorageClass::Temp,0,nullptr,ALL},
        };
    } else if (current_temporal_num_frames_ > 1) {
        const int64_t temporal_rows =
            temporal_num_frames_ * G05_TEMPORAL_PATCH_GROUP;
        const int64_t temporal_stage_rows =
            temporal_num_frames_ * temporal_patches_per_frame_;
        const int64_t spatial_rows = G05_SPATIAL_GROUP_ROWS;
        SdpaConfig temporal_cfg{
            SdpaKernelType::FLASH_ATTN_SPM, hd,
            G05_TEMPORAL_PATCH_GROUP * nq,
            G05_TEMPORAL_PATCH_GROUP * nq,
            num_cores(), 1};
        SdpaConfig spatial_cfg{
            SdpaKernelType::FLASH_ATTN_SPM, hd, nq, nq, num_cores(), 0};
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
            {"sdpa_tmp",  raw_spm ? std::max(spatial_tmp, spatial_qkv)
                                   : spatial_tmp,
                           6, 6, StorageClass::Temp, 0, nullptr, KVIN},
            {"oproj",     spatial_res, 7, 9, StorageClass::Temp, 0, nullptr, KVIN},
            {"fc1",       spatial_fc1, 8, 9, StorageClass::Temp, 0, nullptr, KVIN},
        };
    } else {
        decls = {
            // ALL：Phase1 LN 读 residual1/写 input_norm；Phase2 残差读写两者。
            {"residual1",  res,      1, 6, StorageClass::Temp, 0, nullptr, ALL},
            {"input_norm", res,      1, 6, StorageClass::Temp, 0, nullptr, ALL},
            // Q stays live across KV_FIRST callbacks for exact K1 raw SPM.
            {"q",          qkv,      2, carry_qkv ? 4 : 3,
                           StorageClass::Temp, 0, nullptr, carry_qkv ? ALL : KVIN},
            {"k",          raw_spm ? raw_kv : qkv, 2, 4,
                           StorageClass::Temp, 0, nullptr,
                           raw_spm ? ALL : KVIN},
            {"v",          raw_spm ? raw_kv : qkv, 2, 4,
                           StorageClass::Temp, 0, nullptr,
                           raw_spm ? ALL : KVIN},
            {"q_comp",     carry_qkv ? 0 : qkv, 4, 4,
                           StorageClass::Temp, 0, carry_qkv ? "q" : nullptr,
                           carry_qkv ? ALL : COMP},
            {"oproj",      res,      4, 6, StorageClass::Temp, 0, nullptr, COMP},
            {"sdpa_out",   qkv,      4, 5, StorageClass::Temp, 0, nullptr, COMP},
            {"sdpa_tmp",   raw_spm ? std::max(sdpa_tmp, raw_kv) : sdpa_tmp,
                           4, 4, StorageClass::Temp, 0, nullptr, COMP},
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

bool Qwen3_5VisionModel::is_g05_k1_fast_replay_profile() const {
    // ACC16 requires the complete private projection-owner proof. Replay,
    // preload and FP16 activation transport do not depend on Linear precision.
    // Public ACC32 ultra ERF and MoE remain excluded.
    return (linear_acc32_ || g05_w8a16_ || g05_qkv_spm_available_) &&
        !gelu_erf_ultra_ && !moe_precision_ &&
        !has_temporal_ && num_cores() == 8 && attn_tp() == 8 && mlp_tp() == 8 &&
        num_layers() == 24 && hidden_size() == 1024 && intermediate_size() == 4096 &&
        num_q_heads() == 16 && num_kv_heads() == 16 && head_dim() == 64 &&
        orig_head_dim_ == 64 && eps_ == 1e-6 &&
        has_rope_ && has_step0_ && patch_dim_ == 1536 && has_merger_ &&
        merger_hidden_ == 4096 && out_hidden_size_ == 2048;
}

std::vector<int64_t> Qwen3_5VisionModel::g05_high_precision_encoder_arguments(int64_t rows) const {
    // ABI, HIGH unary, original TANH GELU, rows, full/local MLP widths,
    // physical cores and independent camera spans.
    return {1, 1, 1, rows, intermediate_size(), intermediate_size() / num_cores(),
            num_cores(), current_camera_batch_count_};
}

std::vector<int64_t> Qwen3_5VisionModel::g05_high_precision_merger_arguments(int64_t rows) const {
    // The patch merger keeps HF's exact ERF formula.
    return {1, 1, 2, rows, merger_hidden_, merger_hidden_ / num_cores(),
            num_cores(), current_camera_batch_count_};
}

void Qwen3_5VisionModel::consume_g05_high_precision_encoder_activation(
    int64_t rows, int64_t invocation) {
    if (!high_precision_) return;
    TORCH_CHECK(ctx().has_complete_physical_manifest(),
                "G05 HIGH encoder GELU requires its COMPLETE descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::GRAPH_SCHEDULE, G05_VISION_HIGH_ENCODER_ACTIVATION_SITE,
        1, 0, g05_high_precision_encoder_arguments(rows), invocation);
}

void Qwen3_5VisionModel::consume_g05_high_precision_merger_activation(
    int64_t rows, int64_t invocation) {
    if (!high_precision_) return;
    TORCH_CHECK(ctx().has_complete_physical_manifest(),
                "G05 HIGH merger GELU requires its COMPLETE descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::GRAPH_SCHEDULE, G05_VISION_HIGH_MERGER_ACTIVATION_SITE,
        1, 0, g05_high_precision_merger_arguments(rows), invocation);
}

bool Qwen3_5VisionModel::is_g05_qkv_spm_profile() const {
    return g05_qkv_spm_available_ && is_g05_k1_fast_replay_profile() &&
        !z2_bound_ && !get_debug_export() && current_temporal_num_frames_ == 1 &&
        current_camera_batch_count_ == 1 && !current_compact_step0_ &&
        current_output_patches_ == 256;
}

bool Qwen3_5VisionModel::use_g05_qkv_spm_layout(const LayoutContext& layout) const {
    return is_g05_qkv_spm_profile() &&
        layout.attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA &&
        layout.batch_size == 1 && !layout.is_causal && !layout.use_attn_mask &&
        layout.chunk_size == 256 && layout.effective_kv_cs() == 256 &&
        layout.max_kv_seq_len == 256;
}

bool Qwen3_5VisionModel::g05_qkv_spm_schedule(const FmbThreeStageChunkPlan& plan) const {
    if (plan.chunk_mode != ChunkMode::KV_FIRST || plan.spans.size() != 1 ||
        plan.spans.front().offset != 0 || plan.spans.front().len != 256 ||
        plan.spans.front().group_id != 0 || plan.qkv.chunks.size() != 1 ||
        plan.compute.chunks.size() != 1) return false;
    for (const auto* stage : {&plan.qkv, &plan.compute}) {
        const auto& chunk = stage->chunks.front();
        if (chunk.idx != 0 || chunk.offset != 0 || chunk.len != 256 ||
            chunk.kv_seq_len != 256) return false;
    }
    return true;
}

std::vector<int64_t> Qwen3_5VisionModel::g05_qkv_spm_arguments() const {
    // Version, physical/logical rows, H/Q/KV/D, TP, Q bytes/core, Q alias,
    // KV transport flag, single QKV/compute chunk and position.
    return {1,256,256,1024,16,16,64,8,65536,1,G05_VISION_KV_RAW_SPM_ONLY,1,1,0};
}

bool Qwen3_5VisionModel::consume_g05_qkv_spm_transport(
    const ChunkInfo& chunk, bool compute) {
    if (!is_g05_qkv_spm_profile() || ctx().attention_policy !=
            AttentionExecutionPolicy::SPM_KV_BY_MHA) return false;
    TORCH_CHECK(ctx().has_complete_physical_manifest() &&
        g05_qkv_spm_schedule(ctx().stage_plan) && ctx().position == 0 &&
        ctx().seq_len == 256 && ctx().batch_size == 1 && !ctx().is_causal &&
        !ctx().attention_mask.has_value() && chunk.idx == 0 && chunk.offset == 0 &&
        chunk.len == 256 && chunk.kv_seq_len == 256 &&
        addr(0, "q") == addr(0, "q_comp"),
        "G05 QKV SPM transport requires its exact COMPLETE schedule and live Q alias");
    // Validate both transport intents before either callback emits DMA/kernels.
    const auto& kv = ctx().find_physical_route(
        FmbRouteFamily::KV_INSERT, QWEN35_VISION_KV_SITE, 0);
    TORCH_CHECK(kv.flags == G05_VISION_KV_RAW_SPM_ONLY,
                "G05 QKV SPM transport lost its raw-SPM-only KV intent");
    ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
        G05_VISION_QKV_SPM_TRANSPORT_SITE, 1, 0,
        g05_qkv_spm_arguments(), compute ? 1 : 0);
    return true;
}

std::vector<int64_t> Qwen3_5VisionModel::g05_qkv_linear_arguments() const {
    // Version, M/local N/K, TP, ACC bits, parent M/N/K tile, blocks per
    // projection/total grid, byte row strides, owners, bias bytes, has bias.
    if (g05_w8a16_) return {1,256,128,1024,8,linear_acc32_ ? 32 : 16,
        linear_acc32_ ? 400 : 512,48,128,3,9,2048,256,3,256,1,8,3};
    return {1,256,128,1024,8,32,352,64,128,2,6,2048,256,3,256,1};
}

bool Qwen3_5VisionModel::consume_g05_qkv_linear(
    const ChunkInfo& chunk, bool carry_qkv) {
    if (!g05_qkv_linear_available_ || !carry_qkv) return false;
    TORCH_CHECK(is_g05_qkv_spm_profile() && ctx().has_complete_physical_manifest() &&
        g05_qkv_spm_schedule(ctx().stage_plan) && chunk.idx == 0 &&
        chunk.offset == 0 && chunk.len == 256 && chunk.kv_seq_len == 256,
        "G05 Vision QKV Linear requires its exact COMPLETE raw-SPM schedule");
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,
        g05_w8a16_ ? G05_VISION_QKV_W8_LINEAR_SITE : G05_VISION_QKV_LINEAR_SITE,
        1, 0, g05_qkv_linear_arguments(), chunk.idx);
    return true;
}

bool Qwen3_5VisionModel::g05_k1_b3_available() const {
    return g05_k1_b3_available_ && !linear_acc32_ && g05_w8a16_ &&
        g05_owner_ln_available_ && g05_qkv_linear_available_ &&
        is_g05_k1_fast_replay_profile() && !z2_bound_ && !get_debug_export();
}

bool Qwen3_5VisionModel::g05_k1_camera_group_available(int64_t cameras) const {
    if (cameras != 2 && cameras != 3) return false;
    // Preserve the historical B3/ACC16 optional weight-outer path verbatim.
    if (cameras == 3 && !linear_acc32_) return g05_k1_b3_available();
    if (!g05_w8a16_ || !g05_owner_ln_available_ || !g05_qkv_linear_available_ ||
        !is_g05_k1_fast_replay_profile() || z2_bound_ || get_debug_export()) return false;
    return g05_k1_generated_group_available_[cameras-2];
}

bool Qwen3_5VisionModel::is_g05_k1_camera_group_profile() const {
    return current_temporal_num_frames_ == 1 && !current_compact_step0_ &&
        (current_camera_batch_count_ == 2 ||
         (current_camera_batch_count_ == 3 && linear_acc32_)) &&
        current_output_patches_ == 256 * current_camera_batch_count_ &&
        g05_k1_camera_group_available(current_camera_batch_count_);
}


bool Qwen3_5VisionModel::is_g05_k1_b3_profile() const {
    return g05_k1_b3_available() && current_temporal_num_frames_ == 1 &&
        current_camera_batch_count_ == 3 && current_output_patches_ == 768 &&
        !current_compact_step0_;
}

bool Qwen3_5VisionModel::g05_k1_b3_schedule(const FmbThreeStageChunkPlan& plan) const {
    if (!is_g05_k1_b3_profile() || plan.chunk_mode != ChunkMode::KV_FIRST ||
        plan.spans.size() != 3 || plan.input.chunks.size() != 3 || plan.qkv.chunks.size() != 1 ||
        plan.compute.chunks.size() != 1 ||
        plan.boundary_policies.input != FmbSpanBoundaryPolicy::KEEP_LOCAL ||
        plan.boundary_policies.qkv != FmbSpanBoundaryPolicy::ALLOW_CROSS ||
        plan.boundary_policies.compute != FmbSpanBoundaryPolicy::ALLOW_CROSS) return false;
    for (int camera=0; camera<3; ++camera) {
        const auto& span=plan.spans[camera];
        if (span.offset!=camera*256 || span.len!=256 || span.group_id!=camera) return false;
        const auto& input=plan.input.chunks[camera];
        if(input.idx!=camera || input.offset!=camera*256 || input.len!=256 ||
            input.kv_seq_len!=(camera+1)*256) return false;
    }
    for (const auto* stage : {&plan.qkv,&plan.compute}) {
        const auto& c=stage->chunks.front();
        if (c.idx!=0 || c.offset!=0 || c.len!=768 || c.kv_seq_len!=768) return false;
    }
    return true;
}

std::vector<int64_t> Qwen3_5VisionModel::g05_k1_b3_arguments() const {
    // Version, K, B, rows/camera, combined rows, original dimensions/ACC;
    // per-camera residual/fc1 byte strides; shared scratch; raw-only intent;
    // FC2 inner tile/grid and unchanged parent retirement ABI.
    return {1,1,3,256,768,1024,16,16,64,4096,8,16,524288,262144,
            65536,131072,1,G05_VISION_KV_RAW_SPM_ONLY,256,80,128,13,1};
}

void Qwen3_5VisionModel::append_g05_k1_b3_routes(FmbPhysicalExecutionManifest& manifest) const {
    auto add=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t camera,
                 std::vector<int64_t> args) {
        manifest.routes.push_back({site,f,selector,0,std::move(args),camera});
    };
    for(int phase=0;phase<2;++phase)
        add(FmbRouteFamily::GRAPH_SCHEDULE,G05_VISION_K1_B3_SCHEDULE_SITE,1,phase,g05_k1_b3_arguments());
    auto fc2=g05_k1_b3_arguments();
    fc2.insert(fc2.end(),{768,1024,512,0,8,1,1024,2048,2,20});
    add(FmbRouteFamily::LINEAR,G05_VISION_K1_B3_FC2_SITE,1,0,std::move(fc2));
    for(int camera=0;camera<3;++camera) {
        auto qkv=g05_qkv_linear_arguments();qkv.insert(qkv.end(),{3,camera,camera*256});
        auto owner=g05_owner_ln_arguments();owner.insert(owner.end(),{3,camera,camera*524288});
        add(FmbRouteFamily::LINEAR,G05_VISION_K1_B3_QKV_SITE,1,camera,std::move(qkv));
        add(FmbRouteFamily::NORMALIZATION,G05_VISION_K1_B3_OWNER_LN_SITE,1,camera,owner);
        add(FmbRouteFamily::ALL_REDUCE,G05_VISION_K1_B3_COMPACT_SITE,1,camera,std::move(owner));
        for(auto site:{G05_VISION_K1_B3_Q_ROPE_SITE,G05_VISION_K1_B3_K_ROPE_SITE})
            add(FmbRouteFamily::ROPE,site,qwen35_vision_rope_route(),camera,{3,camera,camera*256,256,16,64,8});
        add(FmbRouteFamily::ATTENTION,G05_VISION_K1_B3_ATTN_SITE,
            static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA),camera,
            {1,256,256,16,16,64,8,0,3,camera});
        add(FmbRouteFamily::ALL_REDUCE,G05_VISION_K1_B3_PREPARE_SITE,
            static_cast<int64_t>(Qwen35VisionAllReduceRoute::PREPARE_RING_INPUT),camera,
            {256,1024,8,8,3,camera});
        add(FmbRouteFamily::LINEAR,G05_VISION_K1_B3_O_SITE,qwen35_vision_linear_route(),camera,
            {256,1024,1024,0,8,16,1,3,camera});
        add(FmbRouteFamily::LINEAR,G05_VISION_K1_B3_FC1_SITE,qwen35_vision_linear_route(),camera,
            {256,4096,1024,1,8,16,1,3,camera});
        if (high_precision_)
            add(FmbRouteFamily::GRAPH_SCHEDULE,G05_VISION_HIGH_ENCODER_ACTIVATION_SITE,
                1,camera,g05_high_precision_encoder_arguments(256));
    }
}

void Qwen3_5VisionModel::consume_g05_k1_b3_schedule(const ChunkInfo& chunk,bool compute) {
    TORCH_CHECK(is_g05_k1_b3_profile() && ctx().has_complete_physical_manifest() &&
        g05_k1_b3_schedule(ctx().stage_plan) && ctx().attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA &&
        ctx().position==0 && ctx().seq_len==768 && ctx().batch_size==1 && !ctx().is_causal &&
        !ctx().attention_mask.has_value() && chunk.idx==0 && chunk.offset==0 &&
        chunk.len==768 && chunk.kv_seq_len==768,
        "G05 K1 B3 requires its exact COMPLETE three-camera raw-SPM schedule");
    // Validate every internal route before any per-camera side effect. No DDR
    // KV is materialized: all three independent 256x256 attention calls consume
    // the raw-only transport authority in this schedule, never a 768x768 mask.
    FmbPhysicalExecutionManifest expected;
    append_g05_k1_b3_routes(expected);
    for(const auto& route:expected.routes) {
        const auto& actual=ctx().find_physical_route(route.family,route.site_id,route.invocation);
        TORCH_CHECK(actual.selector==route.selector && actual.flags==route.flags &&
            actual.arguments==route.arguments,"G05 K1 B3 internal route drift");
    }
    ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
        G05_VISION_K1_B3_SCHEDULE_SITE,1,0,g05_k1_b3_arguments(),compute?1:0);
}

void Qwen3_5VisionModel::emit_g05_k1_b3_layer(int layer,const ChunkInfo& chunk) {
    consume_g05_k1_b3_schedule(chunk,true);
    TORCH_CHECK(layer>=0 && layer<24,"G05 K1 B3 invalid layer");
    if(!ctx().input_in_spm)emit_layer_input_dma(layer,chunk);
    const auto& lw=layer_weights_[layer];
    for(int camera=0;camera<3;++camera) {
        const uint32_t full=camera*524288,fc=camera*262144;
        const auto residual=addr(0,"residual1")+full;
        const auto norm=addr(0,"input_norm")+full;
        const auto projected=addr(0,"oproj")+full;
        const auto activation=addr(0,"fc1")+fc;
        rpu_launch_layernorm_spm_kernel(residual,norm,
            layer_addr(layer,0,"ln1_gamma"),layer_addr(layer,0,"ln1_beta"),
            256,1024,eps_,false,0,8);
        auto qkv_args=g05_qkv_linear_arguments();
        qkv_args.insert(qkv_args.end(),{3,camera,camera*256});
        consume_manifest_route(FmbRouteFamily::LINEAR,G05_VISION_K1_B3_QKV_SITE,
            1,camera,std::move(qkv_args));
        rpu_launch_g05_vision_qkv_linear_w8a16_acc32_kernel(norm,
            lw.q_w,lw.k_w,lw.v_w,addr(0,"q"),addr(0,"k"),addr(0,"v"),
            layer_addr(layer,0,"q_bias"),layer_addr(layer,0,"k_bias"),layer_addr(layer,0,"v_bias"),
            lw.q_ws,lw.k_ws,lw.v_ws,false);
        consume_manifest_route(FmbRouteFamily::ROPE,G05_VISION_K1_B3_Q_ROPE_SITE,
            qwen35_vision_rope_route(),camera,{3,camera,camera*256,256,16,64,8});
        rpu_launch_rope_2d_spm_kernel(addr(0,"q"),addr(0,"q"),
            addr(0,"freq_cos_spm"),addr(0,"freq_sin_spm"),position_idx_keepalive_.data_ptr<int16_t>(),
            camera*256,256,2,64,64,8);
        consume_manifest_route(FmbRouteFamily::ROPE,G05_VISION_K1_B3_K_ROPE_SITE,
            qwen35_vision_rope_route(),camera,{3,camera,camera*256,256,16,64,8});
        rpu_launch_rope_2d_spm_kernel(addr(0,"k"),addr(0,"k"),
            addr(0,"freq_cos_spm"),addr(0,"freq_sin_spm"),position_idx_keepalive_.data_ptr<int16_t>(),
            camera*256,256,2,64,64,8);
        rpu_launch_v_transpose_spm(addr(0,"v"),addr(0,"sdpa_tmp"),1,256,16,64,8);
        consume_manifest_route(FmbRouteFamily::ATTENTION,G05_VISION_K1_B3_ATTN_SITE,
            static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA),camera,
            {1,256,256,16,16,64,8,0,3,camera});
        rpu_launch_sdpa_by_mha_spm(addr(0,"q"),addr(0,"k"),addr(0,"sdpa_tmp"),
            addr(0,"sdpa_out"),0,0,1.0/std::sqrt(64.0),1,256,256,16,16,64,8);
        consume_manifest_route(FmbRouteFamily::ALL_REDUCE,G05_VISION_K1_B3_PREPARE_SITE,
            static_cast<int64_t>(Qwen35VisionAllReduceRoute::PREPARE_RING_INPUT),camera,
            {256,1024,8,8,3,camera});
        rpu_prepare_ring_all_reduce_input(projected,256,1024,8,8);
        consume_manifest_route(FmbRouteFamily::LINEAR,G05_VISION_K1_B3_O_SITE,
            qwen35_vision_linear_route(),camera,{256,1024,1024,0,8,16,1,3,camera});
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0,"sdpa_out"),lw.o_w,projected,
            256,1024,1024,0,8,layer_addr(layer,0,"o_bias"),lw.o_ws,0,0,false);
        auto owner_args=g05_owner_ln_arguments();
        owner_args.insert(owner_args.end(),{3,camera,camera*524288});
        consume_manifest_route(FmbRouteFamily::NORMALIZATION,G05_VISION_K1_B3_OWNER_LN_SITE,
            1,camera,std::move(owner_args));
        rpu_launch_g05_vision_owner_reduce_layernorm_m256n1024_kernel(
            projected,residual,norm,projected,
            layer_addr(layer,0,"ln2_gamma"),layer_addr(layer,0,"ln2_beta"));
        consume_manifest_route(FmbRouteFamily::LINEAR,G05_VISION_K1_B3_FC1_SITE,
            qwen35_vision_linear_route(),camera,{256,4096,1024,1,8,16,1,3,camera});
        rpu_launch_linear_spm_to_spm_acc16_kernel(projected,lw.fc1_w,activation,
            256,4096,1024,1,8,layer_addr(layer,0,"fc1_bias"),lw.fc1_ws,0,0,false);
        consume_g05_high_precision_encoder_activation(256,camera);
        rpu_launch_eltwise_unary_spm_kernel(activation,activation,
            256*512,ValuOpType::ADD,GeluMode::TANH,8,
            high_precision_ ? RpuUnaryPrecision::HIGH : RpuUnaryPrecision::BASE);
    }
    auto fc2_args=g05_k1_b3_arguments();
    fc2_args.insert(fc2_args.end(),{768,1024,512,0,8,1,1024,2048,2,20});
    consume_manifest_route(FmbRouteFamily::LINEAR,G05_VISION_K1_B3_FC2_SITE,1,0,std::move(fc2_args));
    rpu_launch_g05_vision_fc2_weight_outer_linear_w8a16_acc16_kernel(
        addr(0,"fc1"),lw.fc2_w,addr(0,"oproj"),layer_addr(layer,0,"fc2_bias"),lw.fc2_ws);
    for(int camera=0;camera<3;++camera) {
        const uint32_t full=camera*524288;
        auto owner_args=g05_owner_ln_arguments();
        owner_args.insert(owner_args.end(),{3,camera,camera*524288});
        consume_manifest_route(FmbRouteFamily::ALL_REDUCE,G05_VISION_K1_B3_COMPACT_SITE,
            1,camera,std::move(owner_args));
        rpu_launch_g05_vision_compact_residual_m256n1024_kernel(
            addr(0,"oproj")+full,addr(0,"input_norm")+full,addr(0,"residual1")+full);
    }
    if(!ctx().output_to_spm)emit_layer_output_dma(layer,chunk,"residual1");
}

bool Qwen3_5VisionModel::g05_k1_camera_group_schedule(const FmbThreeStageChunkPlan& plan) const {
    if (!is_g05_k1_camera_group_profile() || plan.chunk_mode != ChunkMode::KV_FIRST ||
        plan.spans.size() != static_cast<size_t>(current_camera_batch_count_) ||
        plan.input.chunks.size() != static_cast<size_t>(current_camera_batch_count_) || plan.qkv.chunks.size() != 1 ||
        plan.compute.chunks.size() != 1 ||
        plan.boundary_policies.input != FmbSpanBoundaryPolicy::KEEP_LOCAL ||
        plan.boundary_policies.qkv != FmbSpanBoundaryPolicy::ALLOW_CROSS ||
        plan.boundary_policies.compute != FmbSpanBoundaryPolicy::ALLOW_CROSS) return false;
    for (int camera=0; camera<current_camera_batch_count_; ++camera) {
        const auto& span=plan.spans[camera];
        if (span.offset!=camera*256 || span.len!=256 || span.group_id!=camera) return false;
        const auto& input=plan.input.chunks[camera];
        if(input.idx!=camera || input.offset!=camera*256 || input.len!=256 ||
            input.kv_seq_len!=(camera+1)*256) return false;
    }
    for (const auto* stage : {&plan.qkv,&plan.compute}) {
        const auto& c=stage->chunks.front();
        if (c.idx!=0 || c.offset!=0 || c.len!=current_output_patches_ || c.kv_seq_len!=current_output_patches_) return false;
    }
    return true;
}

std::vector<int64_t> Qwen3_5VisionModel::g05_k1_camera_group_arguments() const {
    const int rows = static_cast<int>(current_output_patches_);
    const auto tile = linear_acc32_
        ? rpu_pl_tiling::select_tile_acc32(rows,1024,512,false)
        : rpu_pl_tiling::select_tile_acc16(rows,1024,512,false);
    // Version2 explicitly names real accumulation and generated FC2 M/N/K,
    // tile/grid; fixed M256 primitives still act on isolated camera slices.
    return {2,1,current_camera_batch_count_,256,rows,1024,16,16,64,4096,8,
            linear_acc32_ ? 32 : 16,524288,262144,65536,131072,1,
            G05_VISION_KV_RAW_SPM_ONLY,tile.m_tile,tile.n_tile,128,
            (1024+tile.n_tile-1)/tile.n_tile,(rows+tile.m_tile-1)/tile.m_tile};
}

void Qwen3_5VisionModel::append_g05_k1_camera_group_routes(FmbPhysicalExecutionManifest& manifest) const {
    auto add=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t camera,
                 std::vector<int64_t> args) {
        manifest.routes.push_back({site,f,selector,0,std::move(args),camera});
    };
    for(int phase=0;phase<2;++phase)
        add(FmbRouteFamily::GRAPH_SCHEDULE,G05_VISION_K1_CAMERA_GROUP_SCHEDULE_SITE,1,phase,g05_k1_camera_group_arguments());
    auto fc2=g05_k1_camera_group_arguments();
    fc2.insert(fc2.end(),{current_output_patches_,1024,4096,512,0,8,linear_acc32_ ? 32 : 16,1,1024,2048,2,3,20,21});
    add(FmbRouteFamily::LINEAR,G05_VISION_K1_CAMERA_GROUP_FC2_SITE,qwen35_vision_linear_route(),0,std::move(fc2));
    for(int camera=0;camera<current_camera_batch_count_;++camera) {
        auto qkv=g05_qkv_linear_arguments();qkv.insert(qkv.end(),{current_camera_batch_count_,camera,camera*256});
        auto owner=g05_owner_ln_arguments();owner.insert(owner.end(),{current_camera_batch_count_,camera,camera*524288});
        add(FmbRouteFamily::LINEAR,G05_VISION_K1_CAMERA_GROUP_QKV_SITE,1,camera,std::move(qkv));
        add(FmbRouteFamily::NORMALIZATION,G05_VISION_K1_CAMERA_GROUP_OWNER_LN_SITE,1,camera,owner);
        add(FmbRouteFamily::ALL_REDUCE,G05_VISION_K1_CAMERA_GROUP_COMPACT_SITE,1,camera,std::move(owner));
        for(auto site:{G05_VISION_K1_CAMERA_GROUP_Q_ROPE_SITE,G05_VISION_K1_CAMERA_GROUP_K_ROPE_SITE})
            add(FmbRouteFamily::ROPE,site,qwen35_vision_rope_route(),camera,{current_camera_batch_count_,camera,camera*256,256,16,64,8});
        add(FmbRouteFamily::ATTENTION,G05_VISION_K1_CAMERA_GROUP_ATTN_SITE,
            static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA),camera,
            {1,256,256,16,16,64,8,0,current_camera_batch_count_,camera});
        add(FmbRouteFamily::ALL_REDUCE,G05_VISION_K1_CAMERA_GROUP_PREPARE_SITE,
            static_cast<int64_t>(Qwen35VisionAllReduceRoute::PREPARE_RING_INPUT),camera,
            {256,1024,8,8,current_camera_batch_count_,camera});
        add(FmbRouteFamily::LINEAR,G05_VISION_K1_CAMERA_GROUP_O_SITE,qwen35_vision_linear_route(),camera,
            {256,1024,1024,0,8,linear_acc32_ ? 32 : 16,1,current_camera_batch_count_,camera});
        add(FmbRouteFamily::LINEAR,G05_VISION_K1_CAMERA_GROUP_FC1_SITE,qwen35_vision_linear_route(),camera,
            {256,4096,1024,1,8,linear_acc32_ ? 32 : 16,1,current_camera_batch_count_,camera});
        if (high_precision_)
            add(FmbRouteFamily::GRAPH_SCHEDULE,G05_VISION_HIGH_ENCODER_ACTIVATION_SITE,
                1,camera,g05_high_precision_encoder_arguments(256));
    }
}

void Qwen3_5VisionModel::consume_g05_k1_camera_group_schedule(const ChunkInfo& chunk,bool compute) {
    TORCH_CHECK(is_g05_k1_camera_group_profile() && ctx().has_complete_physical_manifest() &&
        g05_k1_camera_group_schedule(ctx().stage_plan) && ctx().attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA &&
        ctx().position==0 && ctx().seq_len==current_output_patches_ && ctx().batch_size==1 && !ctx().is_causal &&
        !ctx().attention_mask.has_value() && chunk.idx==0 && chunk.offset==0 &&
        chunk.len==current_output_patches_ && chunk.kv_seq_len==current_output_patches_,
        "G05 K1 generated camera group requires its exact COMPLETE camera-isolated raw-SPM schedule");
    // Validate every internal route before any per-camera side effect. No DDR
    // KV is materialized: all independent 256x256 attention calls consume
    // the raw-only transport authority in this schedule, never a cross-camera attention matrix.
    FmbPhysicalExecutionManifest expected;
    append_g05_k1_camera_group_routes(expected);
    for(const auto& route:expected.routes) {
        const auto& actual=ctx().find_physical_route(route.family,route.site_id,route.invocation);
        TORCH_CHECK(actual.selector==route.selector && actual.flags==route.flags &&
            actual.arguments==route.arguments,"G05 K1 generated camera group internal route drift");
    }
    ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
        G05_VISION_K1_CAMERA_GROUP_SCHEDULE_SITE,1,0,g05_k1_camera_group_arguments(),compute?1:0);
}

void Qwen3_5VisionModel::emit_g05_k1_camera_group_layer(int layer,const ChunkInfo& chunk) {
    consume_g05_k1_camera_group_schedule(chunk,true);
    TORCH_CHECK(layer>=0 && layer<24,"G05 K1 generated camera group invalid layer");
    if(!ctx().input_in_spm)emit_layer_input_dma(layer,chunk);
    const auto& lw=layer_weights_[layer];
    for(int camera=0;camera<current_camera_batch_count_;++camera) {
        const uint32_t full=camera*524288,fc=camera*262144;
        const auto residual=addr(0,"residual1")+full;
        const auto norm=addr(0,"input_norm")+full;
        const auto projected=addr(0,"oproj")+full;
        const auto activation=addr(0,"fc1")+fc;
        rpu_launch_layernorm_spm_kernel(residual,norm,
            layer_addr(layer,0,"ln1_gamma"),layer_addr(layer,0,"ln1_beta"),
            256,1024,eps_,false,0,8);
        auto qkv_args=g05_qkv_linear_arguments();
        qkv_args.insert(qkv_args.end(),{current_camera_batch_count_,camera,camera*256});
        consume_manifest_route(FmbRouteFamily::LINEAR,G05_VISION_K1_CAMERA_GROUP_QKV_SITE,
            1,camera,std::move(qkv_args));
        rpu_launch_g05_vision_qkv_linear_w8a16_acc32_kernel(norm,
            lw.q_w,lw.k_w,lw.v_w,addr(0,"q"),addr(0,"k"),addr(0,"v"),
            layer_addr(layer,0,"q_bias"),layer_addr(layer,0,"k_bias"),layer_addr(layer,0,"v_bias"),
            lw.q_ws,lw.k_ws,lw.v_ws,linear_acc32_);
        consume_manifest_route(FmbRouteFamily::ROPE,G05_VISION_K1_CAMERA_GROUP_Q_ROPE_SITE,
            qwen35_vision_rope_route(),camera,{current_camera_batch_count_,camera,camera*256,256,16,64,8});
        rpu_launch_rope_2d_spm_kernel(addr(0,"q"),addr(0,"q"),
            addr(0,"freq_cos_spm"),addr(0,"freq_sin_spm"),position_idx_keepalive_.data_ptr<int16_t>(),
            camera*256,256,2,64,64,8);
        consume_manifest_route(FmbRouteFamily::ROPE,G05_VISION_K1_CAMERA_GROUP_K_ROPE_SITE,
            qwen35_vision_rope_route(),camera,{current_camera_batch_count_,camera,camera*256,256,16,64,8});
        rpu_launch_rope_2d_spm_kernel(addr(0,"k"),addr(0,"k"),
            addr(0,"freq_cos_spm"),addr(0,"freq_sin_spm"),position_idx_keepalive_.data_ptr<int16_t>(),
            camera*256,256,2,64,64,8);
        rpu_launch_v_transpose_spm(addr(0,"v"),addr(0,"sdpa_tmp"),1,256,16,64,8);
        consume_manifest_route(FmbRouteFamily::ATTENTION,G05_VISION_K1_CAMERA_GROUP_ATTN_SITE,
            static_cast<int64_t>(AttentionExecutionPolicy::SPM_KV_BY_MHA),camera,
            {1,256,256,16,16,64,8,0,current_camera_batch_count_,camera});
        rpu_launch_sdpa_by_mha_spm(addr(0,"q"),addr(0,"k"),addr(0,"sdpa_tmp"),
            addr(0,"sdpa_out"),0,0,1.0/std::sqrt(64.0),1,256,256,16,16,64,8);
        consume_manifest_route(FmbRouteFamily::ALL_REDUCE,G05_VISION_K1_CAMERA_GROUP_PREPARE_SITE,
            static_cast<int64_t>(Qwen35VisionAllReduceRoute::PREPARE_RING_INPUT),camera,
            {256,1024,8,8,current_camera_batch_count_,camera});
        rpu_prepare_ring_all_reduce_input(projected,256,1024,8,8);
        consume_manifest_route(FmbRouteFamily::LINEAR,G05_VISION_K1_CAMERA_GROUP_O_SITE,
            qwen35_vision_linear_route(),camera,{256,1024,1024,0,8,linear_acc32_ ? 32 : 16,1,current_camera_batch_count_,camera});
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0,"sdpa_out"),lw.o_w,projected,
            256,1024,1024,0,8,layer_addr(layer,0,"o_bias"),lw.o_ws,0,0,linear_acc32_);
        auto owner_args=g05_owner_ln_arguments();
        owner_args.insert(owner_args.end(),{current_camera_batch_count_,camera,camera*524288});
        consume_manifest_route(FmbRouteFamily::NORMALIZATION,G05_VISION_K1_CAMERA_GROUP_OWNER_LN_SITE,
            1,camera,std::move(owner_args));
        rpu_launch_g05_vision_owner_reduce_layernorm_m256n1024_kernel(
            projected,residual,norm,projected,
            layer_addr(layer,0,"ln2_gamma"),layer_addr(layer,0,"ln2_beta"));
        consume_manifest_route(FmbRouteFamily::LINEAR,G05_VISION_K1_CAMERA_GROUP_FC1_SITE,
            qwen35_vision_linear_route(),camera,{256,4096,1024,1,8,linear_acc32_ ? 32 : 16,1,current_camera_batch_count_,camera});
        rpu_launch_linear_spm_to_spm_acc16_kernel(projected,lw.fc1_w,activation,
            256,4096,1024,1,8,layer_addr(layer,0,"fc1_bias"),lw.fc1_ws,0,0,linear_acc32_);
        consume_g05_high_precision_encoder_activation(256,camera);
        rpu_launch_eltwise_unary_spm_kernel(activation,activation,
            256*512,ValuOpType::ADD,GeluMode::TANH,8,
            high_precision_ ? RpuUnaryPrecision::HIGH : RpuUnaryPrecision::BASE);
    }
    auto fc2_args=g05_k1_camera_group_arguments();
    fc2_args.insert(fc2_args.end(),{current_output_patches_,1024,4096,512,0,8,linear_acc32_ ? 32 : 16,1,1024,2048,2,3,20,21});
    consume_manifest_route(FmbRouteFamily::LINEAR,G05_VISION_K1_CAMERA_GROUP_FC2_SITE,qwen35_vision_linear_route(),0,std::move(fc2_args));
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0,"fc1"),lw.fc2_w,addr(0,"oproj"),current_output_patches_,
        1024,4096,0,8,layer_addr(layer,0,"fc2_bias"),lw.fc2_ws,0,0,linear_acc32_);
    for(int camera=0;camera<current_camera_batch_count_;++camera) {
        const uint32_t full=camera*524288;
        auto owner_args=g05_owner_ln_arguments();
        owner_args.insert(owner_args.end(),{current_camera_batch_count_,camera,camera*524288});
        consume_manifest_route(FmbRouteFamily::ALL_REDUCE,G05_VISION_K1_CAMERA_GROUP_COMPACT_SITE,
            1,camera,std::move(owner_args));
        rpu_launch_g05_vision_compact_residual_m256n1024_kernel(
            addr(0,"oproj")+full,addr(0,"input_norm")+full,addr(0,"residual1")+full);
    }
    if(!ctx().output_to_spm)emit_layer_output_dma(layer,chunk,"residual1");
}

void Qwen3_5VisionModel::prepare_g05_owner_ln() {
    g05_k1_b3_available_ = false;
    g05_k1_generated_group_available_ = {false,false};
    g05_owner_ln_available_ = false;
    g05_qkv_spm_available_ = false;
    g05_qkv_linear_available_ = false;
    g05_w8a16_ = false;
    // Installation precedes patch/RoPE/merger setup. Bind immutable tensors and
    // payload availability here; use_g05_owner_ln binds the remaining geometry.
    if (gelu_erf_ultra_ || moe_precision_ || has_temporal_ ||
        num_cores() != 8 || attn_tp() != 8 || mlp_tp() != 8 ||
        num_layers() != 24 || hidden_size() != 1024 || intermediate_size() != 4096 ||
        num_q_heads() != 16 || num_kv_heads() != 16 || head_dim() != 64 ||
        orig_head_dim_ != 64 || eps_ != 1e-6 ||
        layer_weights_.size() != 24 || layer_bias_norm_.size() != 24) return;
    if (!layer_weights_.front().q_w.defined()) return;
    const bool w8 = layer_weights_.front().q_w.scalar_type() == at::kChar;
    const bool fp16_acc16 = !linear_acc32_ && !w8;
    for (size_t layer = 0; layer < layer_weights_.size(); ++layer) {
        const auto& w = layer_weights_[layer];
        const auto& n = layer_bias_norm_[layer];
        const at::Tensor* weights[] = {&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w};
        const at::Tensor* scales[] = {&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.fc1_ws,&w.fc2_ws};
        for (int projection = 0; projection < 6; ++projection) {
            const auto& weight = *weights[projection];
            const auto& scale = *scales[projection];
            if (!weight.defined() || weight.scalar_type() != (w8 ? at::kChar : at::kHalf) ||
                weight.device().type() != at::kPrivateUse1 || !weight.is_contiguous()) return;
            if (fp16_acc16 && (weight.dim() != 2 ||
                weight.size(0) != (projection == 4 ? 4096 : 1024) ||
                weight.size(1) != (projection == 5 ? 4096 : 1024))) return;
            if (w8) {
                if (!scale.defined() || scale.scalar_type() != at::kHalf ||
                    scale.device().type() != at::kPrivateUse1 || !scale.is_contiguous() ||
                    scale.dim() != 1 || scale.size(0) != weight.size(0)) return;
            } else if (scale.defined()) return;
        }
        for (const auto* t : {&n.ln1_w, &n.ln1_b, &n.ln2_w, &n.ln2_b,
                &n.q_b, &n.k_b, &n.v_b, &n.o_b, &n.fc1_b, &n.fc2_b}) {
            if (!t->defined() || t->scalar_type() != at::kHalf ||
                t->device().type() != at::kPrivateUse1 || !t->is_contiguous()) return;
            if (fp16_acc16 && (t->dim() != 1 ||
                t->size(0) != (t == &n.fc1_b ? 4096 : 1024))) return;
        }
    }
    // Activations, normalization and raw-QKV transport stay FP16. A complete
    // six-projection owner/scale proof uses those schedules in both modes.
    // FP16 ACC16 retains separate genuine generated Linears: the installed
    // fixed FP16 triple-projection payload is ACC32-only.
    g05_w8a16_ = w8;
    g05_qkv_spm_available_ = true;
    g05_qkv_linear_available_ = (w8 || linear_acc32_) && KernelCache::instance().has_loaded(w8 ?
        (linear_acc32_ ? KernelId::G05_VISION_QKV_W8A16_ACC32_M256N128K1024 :
                        KernelId::G05_VISION_QKV_W8A16_ACC16_M256N128K1024) :
        KernelId::G05_VISION_QKV_FP16_ACC32_M256N128K1024);
    g05_owner_ln_available_ =
        KernelCache::instance().has_loaded(KernelId::G05_VISION_OWNER_REDUCE_LAYERNORM_M256N1024) &&
        KernelCache::instance().has_loaded(KernelId::G05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M256N1024);
    g05_k1_b3_available_ = w8 && !linear_acc32_ && g05_owner_ln_available_ &&
        g05_qkv_linear_available_ && KernelCache::instance().has_loaded(
            KernelId::G05_VISION_FC2_WEIGHT_OUTER_W8A16_ACC16_M768N1024K512);
    if (w8 && g05_owner_ln_available_ && g05_qkv_linear_available_) {
        for (int cameras : {2,3}) {
            if (cameras == 3 && !linear_acc32_) continue;
            const auto tile = linear_acc32_
                ? rpu_pl_tiling::select_tile_acc32(cameras*256,1024,512,false)
                : rpu_pl_tiling::select_tile_acc16(cameras*256,1024,512,false);
            // The dispatcher's mapping helpers have internal linkage. This
            // exact W8 camera profile admits only these canonical generated IDs;
            // require the real selector to agree before publishing capability.
            if (tile.m_tile != (linear_acc32_ ? 288 : 512) ||
                tile.n_tile != (linear_acc32_ ? 80 : 48)) continue;
            const auto kernel = linear_acc32_
                ? KernelId::PL_AT_W8A16_ACC32_M288N80
                : KernelId::PL_AT_W8A16_M512N48;
            g05_k1_generated_group_available_[cameras-2] =
                KernelCache::instance().is_initialized() &&
                KernelCache::instance().get(kernel) != nullptr;
        }
    }
}

bool Qwen3_5VisionModel::use_g05_owner_ln(int64_t rows) const {
    return g05_owner_ln_available_ && is_g05_k1_fast_replay_profile() &&
        !z2_bound_ && !get_debug_export() && rows == 256 &&
        current_temporal_num_frames_ == 1 && current_camera_batch_count_ == 1 &&
        !current_compact_step0_ && current_output_patches_ == 256;
}

std::vector<int64_t> Qwen3_5VisionModel::g05_owner_ln_arguments() const {
    // ABI version, rows, width, cores, owner rows, raw/full bytes, LN eps*C
    // (FP16), active LN warps, nPerThread, input/normalized alias permission.
    return {1,256,1024,8,32,65536,524288,0x1432,1,2,1};
}

void Qwen3_5VisionModel::launch_g05_owner_ln(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_g05_owner_ln(chunk.len) && layer >= 0 && layer < num_layers() &&
                    ctx().has_complete_physical_manifest(),
                "G05 Vision owner LayerNorm requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::NORMALIZATION, G05_VISION_OWNER_LN_SITE, 1, 0,
        g05_owner_ln_arguments(), chunk.idx);
    rpu_launch_g05_vision_owner_reduce_layernorm_m256n1024_kernel(
        addr(0, "oproj"), addr(0, "residual1"), addr(0, "input_norm"),
        addr(0, "oproj"), layer_addr(layer, 0, "ln2_gamma"), layer_addr(layer, 0, "ln2_beta"));
}

void Qwen3_5VisionModel::launch_g05_compact_residual(const ChunkInfo& chunk) {
    TORCH_CHECK(use_g05_owner_ln(chunk.len) && ctx().has_complete_physical_manifest(),
                "G05 Vision compact residual requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::ALL_REDUCE, G05_VISION_COMPACT_RESIDUAL_SITE, 1, 0,
        g05_owner_ln_arguments(), chunk.idx);
    rpu_launch_g05_vision_compact_residual_m256n1024_kernel(
        addr(0, "oproj"), addr(0, "input_norm"), addr(0, "residual1"));
}

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
    const bool g05_k1_fast_replay = is_g05_k1_fast_replay_profile() &&
        !z2_bound_ && !get_debug_export() && current_temporal_num_frames_ == 1 &&
        current_camera_batch_count_ == 1 && !current_compact_step0_ &&
        current_output_patches_ == 256;
    // forward_impl refreshes/retains pixel, position and merger destination
    // bases before run_all_layers. The retained graph executes its recorded
    // preloads, STEP0, tower and merger even when their host emission is skipped.
    // FMB permits the skip only on REPLAY with the matching physical manifest;
    // standalone output cloning remains after capture in the Python caller.
    cfg.fast_replay_skip_layer_loop = has_temporal_ || g05_k1_fast_replay || is_g05_k1_b3_profile() || is_g05_k1_camera_group_profile();
    cfg.fast_replay_skip_preload = has_temporal_ || g05_k1_fast_replay || is_g05_k1_b3_profile() || is_g05_k1_camera_group_profile();

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
    // COMPLETE per-site routes select bounded raw-SPM candidates and retain
    // explicit DDR_REQUIRED authority for every incompatible site.
    cfg.attention_policy = AttentionExecutionPolicy::AUTO;
    return cfg;
}

void Qwen3_5VisionModel::consume_manifest_route(
    FmbRouteFamily family, int64_t site_id,
    int64_t selector, int64_t invocation,
    std::vector<int64_t> resolved_arguments) {
    if (!ctx().has_complete_physical_manifest()) return;
    if (num_cores() != 8 &&
        (family == FmbRouteFamily::LINEAR || family == FmbRouteFamily::ALL_REDUCE ||
         family == FmbRouteFamily::MUTABLE_DMA || family == FmbRouteFamily::ROPE)) {
        resolved_arguments.insert(resolved_arguments.end(),
                                  {2, num_cores(), attn_tp(), mlp_tp(), 8});
    }
    ctx().consume_physical_route(
        family, site_id, selector, /*resolved_flags=*/0,
        std::move(resolved_arguments), invocation);
}

bool Qwen3_5VisionModel::subclass_spm_kv_by_mha_eligible(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t position) const {
    if (z2_bound_ ||
        layout.attention_policy !=
            AttentionExecutionPolicy::SPM_KV_BY_MHA ||
        position != 0 || layout.batch_size != 1 || layout.is_causal ||
        layout.use_attn_mask || plan.chunk_mode != ChunkMode::KV_FIRST) {
        return false;
    }

    if (current_temporal_num_frames_ > 1) {
        return current_temporal_num_frames_ == temporal_num_frames_ &&
            temporal_num_frames_ == 6 &&
            temporal_patches_per_frame_ == 256 &&
            current_camera_batch_count_ > 0 &&
            sdpa_by_mha_spm_is_valid(
                /*batch=*/3, /*seq_q=*/256, /*seq_k=*/256,
                num_q_heads(), num_q_heads(), head_dim(), num_cores(),
                /*MASK_NONE=*/0);
    }

    if (is_g05_k1_camera_group_profile()) {
        return g05_k1_camera_group_schedule(plan) &&
            layout.chunk_size==current_output_patches_ &&
            layout.effective_kv_cs()==current_output_patches_ &&
            layout.max_kv_seq_len==current_output_patches_ &&
            sdpa_by_mha_spm_is_valid(1,256,256,16,16,64,8,0);
    }
    if (is_g05_k1_b3_profile()) {
        return g05_k1_b3_schedule(plan) && layout.chunk_size==768 &&
            layout.effective_kv_cs()==768 && layout.max_kv_seq_len==768 &&
            sdpa_by_mha_spm_is_valid(1,256,256,16,16,64,8,0);
    }
    if (current_camera_batch_count_ != 1 || plan.spans.size() != 1 ||
        plan.qkv.chunks.size() != 1 || plan.compute.chunks.size() != 1) {
        return false;
    }
    const ChunkInfo& qkv = plan.qkv.chunks.front();
    const ChunkInfo& compute = plan.compute.chunks.front();
    const FmbExecutionSpan& span = plan.spans.front();
    const int64_t seq = compute.len;
    if (qkv.offset != 0 || compute.offset != 0 || span.offset != 0 ||
        qkv.len != seq || span.len != seq || compute.kv_seq_len != seq ||
        layout.chunk_size != seq || layout.effective_kv_cs() != seq ||
        layout.max_kv_seq_len != seq) {
        return false;
    }
    return sdpa_by_mha_spm_is_valid(
        /*batch=*/1, seq, seq, num_q_heads(), num_q_heads(), head_dim(),
        attention_tp_, /*MASK_NONE=*/0);
}

std::vector<FmbPhysicalExecutionManifest>
Qwen3_5VisionModel::physical_manifest_domain_for_candidate(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len,
    int64_t position) const {
    if (is_g05_k1_camera_group_profile()) {
        LayoutContext exact=layout;
        exact.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
        if (physical_len!=current_output_patches_ || logical_len!=current_output_patches_ ||
            !subclass_spm_kv_by_mha_eligible(plan,exact,position)) return {};
        return {physical_manifest_for_candidate(plan,exact,physical_len,logical_len,position)};
    }
    if (is_g05_k1_b3_profile()) {
        LayoutContext exact=layout;
        exact.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
        if (physical_len!=768 || logical_len!=768 ||
            !subclass_spm_kv_by_mha_eligible(plan,exact,position)) return {};
        return {physical_manifest_for_candidate(plan,exact,physical_len,logical_len,position)};
    }
    LayoutContext ddr_layout = layout;
    ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;
    std::vector<FmbPhysicalExecutionManifest> domain{
        physical_manifest_for_candidate(
            plan, ddr_layout, physical_len, logical_len, position)};
    LayoutContext raw_layout = layout;
    raw_layout.attention_policy =
        AttentionExecutionPolicy::SPM_KV_BY_MHA;
    if (subclass_spm_kv_by_mha_eligible(plan, raw_layout, position)) {
        domain.push_back(physical_manifest_for_candidate(
            plan, raw_layout, physical_len, logical_len, position));
    }
    return domain;
}

FmbPhysicalExecutionManifest
Qwen3_5VisionModel::physical_manifest_for_candidate(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len,
    int64_t position) const {
    if (z2_bound_) return {};

    FmbPhysicalExecutionManifest manifest;
    manifest.state = FmbPhysicalManifestState::COMPLETE;
    manifest.logical_length = logical_len;
    manifest.physical_length = physical_len;
    manifest.execution_padding_rows = physical_len - logical_len;
    manifest.kv_logical_length = position + logical_len;
    manifest.kv_insert_physical_rows = physical_len;
    manifest.graph_lifecycle = FmbGraphLifecycle::RETAINED_CACHE;
    manifest.linear_accumulation = linear_acc32_
        ? FmbLinearAccumulationPolicy::ACC32
        : FmbLinearAccumulationPolicy::ACC16;

    LayoutContext raw_probe = layout;
    raw_probe.attention_policy =
        AttentionExecutionPolicy::SPM_KV_BY_MHA;
    const bool raw_eligible = subclass_spm_kv_by_mha_eligible(
        plan, raw_probe, position);
    const bool raw_spm = layout.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA &&
        raw_eligible;

    const bool carry_qkv = raw_spm && use_g05_qkv_spm_layout(layout) &&
        physical_len == 256 && logical_len == 256 && position == 0 &&
        g05_qkv_spm_schedule(plan);
    auto append_kv_route = [&](int64_t site_id, int64_t invocation,
                               int64_t insert_position, int64_t rows,
                               int num_cores, int64_t num_heads,
                               int64_t flags = QWEN35_VISION_KV_REASON_DDR_REQUIRED) {
        const KvInsertSegmentPlan kv_plan =
            resolve_kvinsert_plan_auto(
                site_id, manifest.graph_lifecycle,
                insert_position, rows, rows, num_cores, num_heads,
                head_dim(), QWEN35_VISION_KV_CAPABILITIES);
        const KvInsertRouteArguments arguments =
            rpu_kvinsert_route_arguments(
                kv_plan, num_cores, num_heads, head_dim());
        manifest.routes.push_back({
            site_id, FmbRouteFamily::KV_INSERT,
            static_cast<int64_t>(kv_plan.route()), flags,
            {arguments.begin(), arguments.end()}, invocation});
    };
    auto append_route = [&](FmbRouteFamily family, int64_t site_id,
                            int64_t selector, int64_t invocation = 0,
                            std::vector<int64_t> arguments = {}) {
        if (num_cores() != 8 &&
            (family == FmbRouteFamily::LINEAR || family == FmbRouteFamily::ALL_REDUCE ||
             family == FmbRouteFamily::MUTABLE_DMA || family == FmbRouteFamily::ROPE)) {
            arguments.insert(arguments.end(), {2, num_cores(), attn_tp(), mlp_tp(), 8});
        }
        manifest.routes.push_back({
            site_id, family, selector, /*flags=*/0,
            std::move(arguments), invocation});
    };

    if (has_step0_) {
        for (const ChunkInfo& chunk : plan.input.chunks) {
            append_route(
                FmbRouteFamily::MUTABLE_DMA,
                current_compact_step0_
                    ? QWEN35_VISION_STEP0_COMPACT_PIXEL_DMA_SITE
                    : QWEN35_VISION_STEP0_PIXEL_DMA_SITE,
                qwen35_vision_dma_route(
                    Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                chunk.idx, {current_compact_step0_ ? 1 : 0});
            append_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_STEP0_LINEAR_SITE,
                qwen35_vision_linear_route(), chunk.idx);
            append_route(
                FmbRouteFamily::MUTABLE_DMA,
                current_temporal_num_frames_ > 1
                    ? QWEN35_VISION_STEP0_POS_SCATTER_DMA_SITE
                    : QWEN35_VISION_STEP0_POS_BROADCAST_DMA_SITE,
                qwen35_vision_dma_route(
                    current_temporal_num_frames_ > 1
                        ? Qwen35VisionMutableDmaRoute::DDR_SCATTER_TO_SPM
                        : Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                chunk.idx);
            const RpuAllGatherSchedule schedule =
                rpu_resolve_all_gather_schedule(
                    hidden_size() / num_cores(), DWIDTH);
            append_route(
                FmbRouteFamily::COLLECTIVE,
                current_temporal_num_frames_ > 1
                    ? QWEN35_VISION_STEP0_TEMPORAL_ALL_GATHER_SITE
                    : QWEN35_VISION_STEP0_GENERIC_ALL_GATHER_SITE,
                static_cast<int64_t>(schedule), chunk.idx,
                {chunk.len, hidden_size() / num_cores(), DWIDTH, num_cores()});
        }
    }

    if (has_merger_) {
        append_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_MERGER_NORM_G_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM),
            /*invocation=*/0, {has_merger_ ? 1 : 0});
        append_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_MERGER_NORM_B_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        append_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_MERGER_FC1_BIAS_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_SCATTER_TO_SPM));
        append_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_MERGER_FC2_BIAS_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        const int64_t camera_patches =
            current_output_patches_ / current_camera_batch_count_;
        for (int64_t off = 0; off < camera_patches;
             off += QWEN3_5_VISION_MERGER_CHUNK) {
            const int64_t invocation =
                off / QWEN3_5_VISION_MERGER_CHUNK;
            const int64_t rows = std::min<int64_t>(
                QWEN3_5_VISION_MERGER_CHUNK, camera_patches - off);
            const int64_t merged_rows =
                rows / QWEN3_5_SPATIAL_MERGE_UNIT;
            if (high_precision_)
                append_route(FmbRouteFamily::GRAPH_SCHEDULE,
                    G05_VISION_HIGH_MERGER_ACTIVATION_SITE, 1, invocation,
                    g05_high_precision_merger_arguments(merged_rows));
            append_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_MERGER_FC1_SITE,
                qwen35_vision_linear_route(), invocation);
            append_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_MERGER_FC2_SITE,
                qwen35_vision_linear_route(), invocation);
            append_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN35_VISION_MERGER_ALL_REDUCE_SITE,
                qwen35_vision_ring_route(merged_rows, out_hidden_size_, num_cores()),
                invocation);
            append_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_VISION_MERGER_OUTPUT_DMA_SITE,
                qwen35_vision_dma_route(
                    Qwen35VisionMutableDmaRoute::SPM_COPY_TO_DDR),
                invocation);
        }
    }

    if (is_g05_k1_b3_profile()) {
        append_g05_k1_b3_routes(manifest);
    } else if (is_g05_k1_camera_group_profile()) {
        append_g05_k1_camera_group_routes(manifest);
    } else if (current_temporal_num_frames_ > 1) {
        append_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_TEMPORAL_PE_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM),
            /*invocation=*/0, {current_temporal_num_frames_});
        manifest.routes.push_back({
            QWEN35_VISION_TEMPORAL_ATTN_SITE, FmbRouteFamily::ATTENTION,
            static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
            QWEN35_VISION_ATTN_TEMPORAL_PACKED_KV_DDR_REQUIRED,
            {/*batch=*/1, temporal_num_frames_, temporal_num_frames_,
             G05_TEMPORAL_PATCH_GROUP * num_q_heads(),
             G05_TEMPORAL_PATCH_GROUP * num_q_heads(), head_dim(),
             num_cores(), /*MASK_LTM=*/1}});
        manifest.routes.push_back({
            raw_spm ? QWEN35_VISION_SPATIAL_RAW_SPM_ATTN_SITE
                    : QWEN35_VISION_SPATIAL_ATTN_SITE,
            FmbRouteFamily::ATTENTION,
            static_cast<int64_t>(
                raw_spm ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                        : AttentionExecutionPolicy::DDR_KV),
            raw_spm ? 0
                    : raw_eligible
                        ? QWEN35_VISION_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED
                        : QWEN35_VISION_ATTN_RAW_KERNEL_INCOMPATIBLE_DDR_REQUIRED,
            {/*batch=*/3, temporal_patches_per_frame_,
             temporal_patches_per_frame_, num_q_heads(), num_q_heads(),
             head_dim(), num_cores(), /*MASK_NONE=*/0}});
        for (const int64_t site_id : {
                 QWEN35_VISION_TEMPORAL_Q_SITE,
                 QWEN35_VISION_TEMPORAL_K_SITE,
                 QWEN35_VISION_TEMPORAL_V_SITE,
                 QWEN35_VISION_SPATIAL_Q_SITE,
                 QWEN35_VISION_SPATIAL_K_SITE,
                 QWEN35_VISION_SPATIAL_V_SITE,
                 QWEN35_VISION_SPATIAL_O_SITE,
                 QWEN35_VISION_SPATIAL_FC1_SITE,
                 QWEN35_VISION_SPATIAL_FC2_SITE}) {
            append_route(
                FmbRouteFamily::LINEAR, site_id,
                qwen35_vision_linear_route());
        }
        append_route(
            FmbRouteFamily::ROPE, QWEN35_VISION_SPATIAL_Q_ROPE_SITE,
            qwen35_vision_rope_route());
        append_route(
            FmbRouteFamily::ROPE, QWEN35_VISION_SPATIAL_K_ROPE_SITE,
            qwen35_vision_rope_route());
        append_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_VISION_SPATIAL_ATTN_ALL_REDUCE_SITE,
            qwen35_vision_ring_route(
                G05_SPATIAL_GROUP_ROWS, hidden_size(), num_cores()));
        append_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_VISION_SPATIAL_MLP_ALL_REDUCE_SITE,
            qwen35_vision_ring_route(
                G05_SPATIAL_GROUP_ROWS, hidden_size(), num_cores()));
        append_kv_route(
            QWEN35_VISION_TEMPORAL_KV_SITE, /*invocation=*/0,
            /*insert_position=*/0, temporal_num_frames_, num_cores(),
            G05_TEMPORAL_PATCH_GROUP * num_q_heads());
        append_kv_route(
            QWEN35_VISION_SPATIAL_KV_SITE, /*invocation=*/0,
            /*insert_position=*/0, G05_SPATIAL_GROUP_ROWS, num_cores(),
            num_q_heads());
    } else {
        const int64_t ddr_reason = position != 0
            ? QWEN35_VISION_ATTN_PREFIX_HISTORY_DDR_REQUIRED
            : plan.qkv.chunks.size() != 1 ||
                    plan.compute.chunks.size() != 1
                ? QWEN35_VISION_ATTN_MULTI_CHUNK_DDR_REQUIRED
                : raw_eligible
                    ? QWEN35_VISION_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED
                    : QWEN35_VISION_ATTN_RAW_KERNEL_INCOMPATIBLE_DDR_REQUIRED;
        for (const ChunkInfo& chunk : plan.compute.chunks) {
            manifest.routes.push_back({
                raw_spm ? QWEN35_VISION_RAW_SPM_ATTN_SITE
                        : QWEN35_VISION_ATTN_SITE,
                FmbRouteFamily::ATTENTION,
                static_cast<int64_t>(
                    raw_spm ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                            : AttentionExecutionPolicy::DDR_KV),
                raw_spm ? 0 : ddr_reason,
                {/*batch=*/1, chunk.len, physical_len,
                 num_q_heads(), num_q_heads(), head_dim(), attention_tp_,
                 /*MASK_NONE=*/0},
                /*invocation=*/chunk.idx});
            append_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN35_VISION_PREPARE_ALL_REDUCE_SITE,
                static_cast<int64_t>(
                    Qwen35VisionAllReduceRoute::PREPARE_RING_INPUT),
                chunk.idx);
            append_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_O_SITE,
                qwen35_vision_linear_route(), chunk.idx);
            if (use_g05_owner_ln(chunk.len)) {
                append_route(FmbRouteFamily::NORMALIZATION, G05_VISION_OWNER_LN_SITE,
                             1, chunk.idx, g05_owner_ln_arguments());
            } else {
                append_route(
                    FmbRouteFamily::ALL_REDUCE,
                    QWEN35_VISION_ATTN_ALL_REDUCE_SITE,
                    qwen35_vision_ring_route(chunk.len, hidden_size(), num_cores()),
                    chunk.idx);
            }
            append_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_FC1_SITE,
                qwen35_vision_linear_route(), chunk.idx);
            if (high_precision_)
                append_route(FmbRouteFamily::GRAPH_SCHEDULE,
                    G05_VISION_HIGH_ENCODER_ACTIVATION_SITE, 1, chunk.idx,
                    g05_high_precision_encoder_arguments(chunk.len));
            append_route(
                FmbRouteFamily::LINEAR, QWEN35_VISION_FC2_SITE,
                qwen35_vision_linear_route(), chunk.idx);
            if (use_g05_owner_ln(chunk.len)) {
                append_route(FmbRouteFamily::ALL_REDUCE, G05_VISION_COMPACT_RESIDUAL_SITE,
                             1, chunk.idx, g05_owner_ln_arguments());
            } else {
                append_route(
                    FmbRouteFamily::ALL_REDUCE,
                    QWEN35_VISION_MLP_ALL_REDUCE_SITE,
                    qwen35_vision_ring_route(chunk.len, hidden_size(), num_cores()),
                    chunk.idx);
            }
        }
        for (const ChunkInfo& chunk : plan.qkv.chunks) {
            if (carry_qkv && g05_qkv_linear_available_) {
                append_route(FmbRouteFamily::LINEAR,
                             g05_w8a16_ ? G05_VISION_QKV_W8_LINEAR_SITE : G05_VISION_QKV_LINEAR_SITE,
                             1, chunk.idx, g05_qkv_linear_arguments());
            } else {
                for (const int64_t site_id : {
                         QWEN35_VISION_Q_SITE,
                         QWEN35_VISION_K_SITE,
                         QWEN35_VISION_V_SITE}) {
                    append_route(
                        FmbRouteFamily::LINEAR, site_id,
                        qwen35_vision_linear_route(), chunk.idx);
                }
            }
            append_route(
                FmbRouteFamily::ROPE, QWEN35_VISION_Q_ROPE_SITE,
                qwen35_vision_rope_route(), chunk.idx);
            append_route(
                FmbRouteFamily::ROPE, QWEN35_VISION_K_ROPE_SITE,
                qwen35_vision_rope_route(), chunk.idx);
            append_kv_route(
                QWEN35_VISION_KV_SITE, chunk.idx,
                position + chunk.offset, chunk.len, attention_tp_, num_q_heads(),
                carry_qkv ? G05_VISION_KV_RAW_SPM_ONLY : QWEN35_VISION_KV_REASON_DDR_REQUIRED);
            if (carry_qkv) {
                for (int phase : {0,1}) {
                    append_route(FmbRouteFamily::GRAPH_SCHEDULE,
                        G05_VISION_QKV_SPM_TRANSPORT_SITE, 1, phase,
                        g05_qkv_spm_arguments());
                }
            }
        }
    }
    append_fmb_shared_runtime_routes(
        manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA,
        /*compute_row_multiplier=*/1, num_cores(), mlp_tp());
    std::sort(
        manifest.routes.begin(), manifest.routes.end(),
        [](const FmbRouteManifestEntry& lhs,
           const FmbRouteManifestEntry& rhs) {
            const int64_t lhs_family = static_cast<int64_t>(lhs.family);
            const int64_t rhs_family = static_cast<int64_t>(rhs.family);
            if (lhs_family != rhs_family) return lhs_family < rhs_family;
            if (lhs.site_id != rhs.site_id) return lhs.site_id < rhs.site_id;
            return lhs.invocation < rhs.invocation;
        });
    return manifest;
}

FmbPhysicalManifestForwardCapability
Qwen3_5VisionModel::physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& /*manifest*/) const {
    if (z2_bound_) return {};
    return {true, FmbGraphLifecycle::RETAINED_CACHE};
}

Qwen3_5VisionModel::VisionExecutionPlan
Qwen3_5VisionModel::resolve_execution_plan(int64_t execution_len) {
    TORCH_CHECK(
        current_temporal_num_frames_ == 1 || get_chunk_size_override() == 0,
        "planner capability reject: Qwen3.5 temporal/compact Vision uses its "
        "certified fixed schedule and does not admit a scalar exact chunk");
    const int64_t compute_chunk_size = resolve_chunk_size_for_shape(
        execution_len, /*position=*/0, std::nullopt, /*is_causal=*/false);
    TORCH_CHECK(
        current_temporal_num_frames_ == 1 ||
            compute_chunk_size == execution_len,
        "planner capability reject: Qwen3.5 temporal Vision must resolve one "
        "full physical compute schedule");
    auto compute_chunks = make_vision_chunks(execution_len, compute_chunk_size);
    const ChunkPlan compute_plan{
        compute_chunks.front().len,
        static_cast<int64_t>(compute_chunks.size())};
    const ChunkPlan qkv_plan = plan_kv_first_chunks(compute_plan);
    TORCH_CHECK(qkv_plan.chunk_size > 0,
                "Qwen3_5VisionModel: planner returned an invalid QKV chunk");
    const int64_t qkv_chunk_size =
        qkv_plan.chunk_size == compute_plan.chunk_size
        ? compute_chunk_size : qkv_plan.chunk_size;
    auto qkv_chunks = make_vision_chunks(execution_len, qkv_chunk_size);

    const int64_t input_chunk_size =
        input_chunk_size_for_geometry(execution_len);
    auto input_chunks = make_vision_chunks(execution_len, input_chunk_size);
    TORCH_INTERNAL_ASSERT(execution_len % current_camera_batch_count_ == 0);
    const int64_t rows_per_camera =
        execution_len / current_camera_batch_count_;
    std::vector<FmbExecutionSpan> spans;
    spans.reserve(current_camera_batch_count_);
    for (int64_t camera = 0; camera < current_camera_batch_count_; ++camera) {
        spans.push_back(
            {camera * rows_per_camera, rows_per_camera, camera});
    }
    FmbStageBoundaryPolicies boundary_policies{};
    if (has_step0_) {
        boundary_policies.input = FmbSpanBoundaryPolicy::KEEP_LOCAL;
    }

    VisionExecutionPlan result;
    result.input_chunk_size = input_chunk_size;
    result.qkv_chunk_size = qkv_chunk_size;
    result.compute_chunk_size = compute_chunk_size;
    if (has_merger_) {
        TORCH_CHECK(execution_len % QWEN3_5_SPATIAL_MERGE_UNIT == 0 &&
                        current_output_patches_ % QWEN3_5_SPATIAL_MERGE_UNIT == 0,
                    "Qwen3_5VisionModel: merger requires patch counts divisible by ",
                    QWEN3_5_SPATIAL_MERGE_UNIT);
        const int64_t rows =
            current_output_patches_ / current_camera_batch_count_;
        result.merger_chunk_size = QWEN3_5_VISION_MERGER_CHUNK;
        result.merger_chunks_per_span =
            (rows + QWEN3_5_VISION_MERGER_CHUNK - 1) /
            QWEN3_5_VISION_MERGER_CHUNK;
    }
    result.stages = compose_fmb_three_stage_chunk_plan(
        std::move(input_chunks), std::move(qkv_chunks),
        std::move(compute_chunks), std::move(spans), ChunkMode::KV_FIRST,
        boundary_policies);
    return result;
}

std::vector<std::vector<int64_t>> Qwen3_5VisionModel::resolve_prefill_domain(
    int64_t execution_len,
    int64_t temporal_num_frames,
    int64_t camera_batch_count,
    bool compact_input) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "Qwen3_5VisionModel::resolve_prefill_domain must run outside "
                "Graph capture/replay");
    RpuExecutionCleanupGuard exclusive_planning(
        "Qwen3_5VisionModel::resolve_prefill_domain");
    begin_kvinsert_cost_domain_oracle();
    bool completed = false;
    auto discard_failed_oracle = c10::make_scope_exit([&] {
        if (!completed) begin_kvinsert_cost_domain_oracle();
    });
    const int64_t saved_temporal = current_temporal_num_frames_;
    const int64_t saved_camera = current_camera_batch_count_;
    const int64_t saved_output = current_output_patches_;
    const int64_t saved_cap = chunk_size_cap_;
    const bool saved_compact_step0 = current_compact_step0_;
    auto restore = c10::make_scope_exit([&] {
        current_temporal_num_frames_ = saved_temporal;
        current_camera_batch_count_ = saved_camera;
        current_output_patches_ = saved_output;
        chunk_size_cap_ = saved_cap;
        current_compact_step0_ = saved_compact_step0;
    });
    configure_execution_geometry(
        execution_len, temporal_num_frames, camera_batch_count, compact_input);
    const VisionExecutionPlan plan = resolve_execution_plan(execution_len);
    // Keep STEP0/spans/merger geometry independent; enumerate every admitted
    // QKV/compute capacity and manifest through the shared native oracle.
    auto physical_candidates = resolve_prefill_stage_domain_for_shape(
        execution_len, 0, std::nullopt, false,
        plan.stages.input.chunks, plan.stages.spans,
        plan.stages.boundary_policies, (is_g05_k1_b3_profile() || is_g05_k1_camera_group_profile()) ?
            current_output_patches_ : get_chunk_size_override());
    std::vector<std::vector<int64_t>> descriptors;
    std::vector<FmbPrefillStageCandidate> returned_candidates;
    for (const auto& candidate : physical_candidates) {
        const uint64_t stage_fingerprint =
            fmb_three_stage_chunk_plan_fingerprint(candidate.stage_plan);
        TORCH_CHECK(stage_fingerprint != 0,
                    "Qwen3_5VisionModel: stage-plan fingerprint must be nonzero");
        LayoutContext layout;
        layout.chunk_size = candidate.stage_plan.compute.plan.chunk_size;
        layout.max_kv_seq_len = execution_len;
        layout.num_layers = num_layers();
        layout.is_causal = false;
        layout.stage_plan_fingerprint = stage_fingerprint;
        if (candidate.stage_plan.qkv.plan.chunk_size != layout.chunk_size) {
            layout.kv_insert_chunk_size = candidate.stage_plan.qkv.plan.chunk_size;
        }
        const auto& physical_manifest = candidate.physical_manifest;
        layout.physical_manifest_fingerprint =
            fmb_physical_manifest_fingerprint(physical_manifest);
        layout.rope_table_residency =
            fmb_rope_table_residency(physical_manifest);
        const auto selected_attention_policy =
            FmbPhysicalManifestConsumer(physical_manifest)
                .attention_layout_policy();
        if (selected_attention_policy.has_value()) {
            layout.attention_policy = *selected_attention_policy;
        }
        if (!spm_pipeline_component_layout_fits_for_cpu_contract(
                layout, candidate.stage_plan)) {
            continue;
        }
        const SpmPipelineComponentLayout planned_layout =
            resolve_spm_pipeline_component_layout_for_cpu_contract(
                layout, candidate.stage_plan);
        TORCH_CHECK(planned_layout.layout_hash != 0,
                    "Qwen3_5VisionModel: planned layout hash must be nonzero");
        std::vector<int64_t> descriptor{
            3,
            execution_len,
            candidate.input_chunk_size,
            candidate.qkv_chunk_size,
            candidate.compute_chunk_size,
            plan.merger_chunk_size,
            plan.merger_chunks_per_span,
            static_cast<int64_t>(stage_fingerprint >> 32),
            static_cast<int64_t>(stage_fingerprint & 0xffffffffULL),
            static_cast<int64_t>(planned_layout.layout_hash >> 32),
            static_cast<int64_t>(planned_layout.layout_hash & 0xffffffffULL),
            static_cast<int64_t>(candidate.stage_plan.spans.size()),
            has_step0_ ? 1 : 0,
            has_merger_ ? 1 : 0,
            temporal_num_frames,
            camera_batch_count,
            compact_input ? 1 : 0,
        };
        for (const auto& span : candidate.stage_plan.spans) {
            descriptor.push_back(span.offset);
            descriptor.push_back(span.len);
        }
        const std::vector<int64_t> stage_descriptor =
            encode_fmb_prefill_stage_candidate(candidate);
        bind_kvinsert_cost_owner_prefix(stage_descriptor, descriptor);
        descriptor.insert(
            descriptor.end(), stage_descriptor.begin(), stage_descriptor.end());
        descriptors.push_back(std::move(descriptor));
        returned_candidates.push_back(candidate);
    }
    TORCH_CHECK(!descriptors.empty(),
                "RPU_PLANNER_REJECT:NO_FEASIBLE: Qwen3.5 Vision has no exact "
                "physical route whose joint SPM layout fits");
    finish_kvinsert_cost_domain_oracle(returned_candidates);
    completed = true;
    return descriptors;
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
Qwen3_5VisionModel::mint_vision_kvinsert_exact_candidate(
        at::IntArrayRef descriptor, int64_t site_id, int64_t invocation, int64_t route) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "Qwen3.5 Vision exact mint must run outside Graph capture");
    RpuExecutionCleanupGuard exclusive_planning("qwen35_vision_exact_candidate");
    // The production StageTuple/forward ABI contains the existing inner v3
    // descriptor. Its custom prefix belongs to this native oracle, not input.
    auto prefix = kvinsert_cost_owner_prefix(descriptor);
    // Shared mint restores the actual spatial/temporal/camera member snapshot,
    // recomputes the COMPLETE manifest and the real SPM layout, then restores.
    auto minted = mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
    const auto& stage_descriptor = std::get<0>(minted);
    prefix[9] = std::get<1>(minted);
    prefix[10] = std::get<2>(minted);
    bind_kvinsert_cost_owner_prefix(stage_descriptor, prefix);
    return minted;
}

std::vector<int64_t> Qwen3_5VisionModel::resolve_prefill_plan(
        int64_t execution_len, int64_t temporal_num_frames,
        int64_t camera_batch_count, bool compact_input) {
    // Compatibility-only structural selection. Production A6 consumes the
    // complete domain above, including same-chunk physical alternatives.
    RpuExecutionCleanupGuard exclusive_planning(
        "Qwen3_5VisionModel::resolve_prefill_plan");
    auto domain = resolve_prefill_domain(
        execution_len, temporal_num_frames, camera_batch_count, compact_input);
    size_t selected = 0;
    for (size_t i = 1; i < domain.size(); ++i) {
        if (detail::prefer_balanced_chunk(
                execution_len, domain[i][4], domain[selected][4])) selected = i;
    }
    auto descriptor = std::move(domain[selected]);
    const size_t stage_offset = 17 + 2 * static_cast<size_t>(descriptor[11]);
    finish_kvinsert_cost_domain_oracle({decode_fmb_prefill_stage_candidate(
        at::IntArrayRef(descriptor).slice(stage_offset))});
    return descriptor;
}

// ── plan_kv_first_chunks: KV-insert 相分块计划（v1：与 compute 用同一 cs）。──
ChunkPlan Qwen3_5VisionModel::plan_kv_first_chunks(const ChunkPlan& compute_plan) {
    return compute_plan;
}

// MR-C / T34 — cold per-handle cap. Same contract as
// Qwen3_5Model::set_chunk_size_cap: it keys the plan the GraphCache entries are
// built from, so a live handle must not be hot-switched.
void Qwen3_5VisionModel::set_chunk_size_cap(int64_t cap) {
    TORCH_CHECK(cap >= 0, "Qwen3.5 vision chunk cap must be >= 0 (0 = no cap), got ", cap);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 vision chunk cap must be set before the first forward");
    chunk_size_cap_ = cap;
    set_chunk_size_override(0);
    invalidate_model_state();
}

void Qwen3_5VisionModel::set_prefill_chunk_size(int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0 ||
                    (chunk_size > 0 && chunk_size % 16 == 0),
                "Qwen3.5 vision chunk size must be 0 (auto) or a positive "
                "multiple of 16, got ", chunk_size);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 vision chunk size must be set before the first forward");
    set_chunk_size_override(chunk_size);
    invalidate_model_state();
}

void Qwen3_5VisionModel::stage_prefill_execution_controls(
        uint64_t token, int64_t cap, int64_t chunk_size) {
    TORCH_CHECK(cap >= 0,
                "Qwen3.5 vision chunk cap must be >= 0, got ", cap);
    TORCH_CHECK(chunk_size == 0 ||
                    (chunk_size > 0 && chunk_size % 16 == 0),
                "Qwen3.5 vision chunk size must be 0 (auto) or a positive "
                "multiple of 16, got ", chunk_size);
    const int64_t old_cap = chunk_size_cap_;
    stage_execution_controls(
        token,
        chunk_size,
        std::nullopt,
        [this, cap] { chunk_size_cap_ = cap; },
        [this, old_cap] { chunk_size_cap_ = old_cap; },
        "Qwen3_5VisionModel::stage_prefill_execution_controls");
}

// SPM feasibility is already checked by the exact first-fit planner. Keep only
// the SDPA kernel constraint and an optional explicit cap for controlled tests.
//
// The cap is installed once per handle by the adapter's cold legacy/canonical
// translator. Native planning never reads process environment state.
bool Qwen3_5VisionModel::subclass_chunk_size_valid(int64_t cs, int64_t seq, int64_t pos) const {
    if (is_g05_k1_b3_profile()) return cs==768 && seq==768 && pos==0;
    if (is_g05_k1_camera_group_profile())
        return cs==current_output_patches_ && seq==current_output_patches_ && pos==0;
    if (current_temporal_num_frames_ > 1) return cs == seq;
    // 真·kernel 约束，任何时候都不能摘。
    SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                        head_dim(), num_q_heads(), num_q_heads(),
                        attention_tp_, /*mask=*/0};
    if (!sdpa_is_valid_chunk_size(sdpa_cfg, cs, seq, pos)) return false;

    return true;
}

int64_t Qwen3_5VisionModel::subclass_chunk_size_cap(
        int64_t /*seq*/, int64_t /*pos*/) const {
    return chunk_size_cap_;
}

// ─────────────────────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────────────────────

// ── emit_preload_weights: per-layer norm + col/row-partition bias DMA into
//    PersistentPerLayer SPM slots. BUILD records this preload prefix; retained
//    graph REPLAY executes it again. ──────────────────────────────────────────
void Qwen3_5VisionModel::emit_preload_weights() {
    int64_t h = hidden_size();
    int64_t local_q_dim = (num_q_heads() / attention_tp_) * head_dim();
    int64_t local_inter = intermediate_size() / num_cores();

    // Loop 1: active-core DMAs for all layers (zero + norm + col-partition bias).
    for (int64_t L = 0; L < num_layers(); ++L) {
        auto& bn = layer_bias_norm_[L];

        // Zero the row-partition bias slots with the graph-aware `fill`, exactly
        // as build_gdn does. Zero all active cores so the 1-core bias load
        // below leaves the other active cores at 0 for the row-partition all_reduce.
        rpu_launch_fill_spm_kernel(layer_addr(L, 0, "o_bias"),   h, c10::Half(0.0f), num_cores());
        rpu_launch_fill_spm_kernel(layer_addr(L, 0, "fc2_bias"), h, c10::Half(0.0f), num_cores());

        rpu_launch_ddr_broadcast_spm_dma(
            bn.ln1_w.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "ln1_gamma"), num_cores());
        rpu_launch_ddr_broadcast_spm_dma(
            bn.ln1_b.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "ln1_beta"), num_cores());
        rpu_launch_ddr_broadcast_spm_dma(
            bn.ln2_w.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "ln2_gamma"), num_cores());
        rpu_launch_ddr_broadcast_spm_dma(
            bn.ln2_b.data_ptr<c10::Half>(), h,
            layer_addr(L, 0, "ln2_beta"), num_cores());

        rpu_launch_ddr_scatter_spm_dma(
            bn.q_b.data_ptr<c10::Half>(),
            local_q_dim, local_q_dim * DWIDTH,
            layer_addr(L, 0, "q_bias"),
            /*num_cores=*/attention_tp_);
        rpu_launch_ddr_scatter_spm_dma(
            bn.k_b.data_ptr<c10::Half>(),
            local_q_dim, local_q_dim * DWIDTH,
            layer_addr(L, 0, "k_bias"),
            /*num_cores=*/attention_tp_);
        rpu_launch_ddr_scatter_spm_dma(
            bn.v_b.data_ptr<c10::Half>(),
            local_q_dim, local_q_dim * DWIDTH,
            layer_addr(L, 0, "v_bias"),
            /*num_cores=*/attention_tp_);
        rpu_launch_ddr_scatter_spm_dma(
            bn.fc1_b.data_ptr<c10::Half>(),
            local_inter, local_inter * DWIDTH,
            layer_addr(L, 0, "fc1_bias"),
            /*num_cores=*/num_cores());
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
        freq_cos_.data_ptr<c10::Half>(), freq_cos_.numel(), addr(0, "freq_cos_spm"), num_cores());
    rpu_launch_ddr_broadcast_spm_dma(
        freq_sin_.data_ptr<c10::Half>(), freq_sin_.numel(), addr(0, "freq_sin_spm"), num_cores());

    if (current_temporal_num_frames_ > 1) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_TEMPORAL_PE_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM),
            /*invocation=*/0, {current_temporal_num_frames_});
        rpu_launch_ddr_broadcast_spm_dma(
            temporal_pe_.data_ptr<c10::Half>(), temporal_pe_.numel(),
            addr(0, "temporal_pe"), num_cores());
    }

    // MERGER weights (only when set_merger() opted in). Same three bias/norm
    // templates as the per-layer loops above, into the count-1 Persistent slots.
    if (has_merger_) {
        const int64_t mh       = merger_hidden_;          // hidden*4 = fc1 N
        const int64_t oh       = out_hidden_size_;         // fc2 N
        const int64_t local_mh = mh / num_cores();           // fc1 col shard width

        // norm gamma/beta — broadcast to all active cores (LN reads full width).
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_MERGER_NORM_G_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM),
            /*invocation=*/0, {has_merger_ ? 1 : 0});
        rpu_launch_ddr_broadcast_spm_dma(
            merger_norm_w_.data_ptr<c10::Half>(), h, addr(0, "m_norm_g"), num_cores());
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_MERGER_NORM_B_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        rpu_launch_ddr_broadcast_spm_dma(
            merger_norm_b_.data_ptr<c10::Half>(), h, addr(0, "m_norm_b"), num_cores());

        // fc1 bias — COL-scatter: core i gets its own local_mh (=512) slice
        // (ParallelLinearGenerator col convention, same as q/k/v/fc1 bias above).
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_MERGER_FC1_BIAS_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_SCATTER_TO_SPM));
        rpu_launch_ddr_scatter_spm_dma(
            merger_fc1_b_.data_ptr<c10::Half>(),
            local_mh, local_mh * DWIDTH,
            addr(0, "m_fc1_bias"), /*num_cores=*/num_cores());

        // fc2 bias — ROW convention: zero all active cores with fill, then
        // load the real bias into core 0 ONLY. The row all_reduce sums the
        // partials, so bias must be present on exactly one core → counted once.
        rpu_launch_fill_spm_kernel(addr(0, "m_fc2_bias"), oh, c10::Half(0.0f), num_cores());
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_VISION_MERGER_FC2_BIAS_PRELOAD_SITE,
            qwen35_vision_dma_route(
                Qwen35VisionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        rpu_launch_ddr_broadcast_spm_dma(
            merger_fc2_b_.data_ptr<c10::Half>(), oh,
            addr(0, "m_fc2_bias"), /*num_cores=*/1);
    }
}

void Qwen3_5VisionModel::set_moe_precision(bool linear_acc32, bool gelu_erf_ultra) {
    TORCH_CHECK(!high_precision_, "G05 HIGH math cannot select the MoE Vision profile");
    TORCH_CHECK(layer_weights_.empty() && num_layers() == 0,
                "qwen3_5_vision_set_moe_precision must be called before set_weights");
    TORCH_CHECK(num_cores() == 4 || num_cores() == 8,
                "Qwen3.5-MoE Vision precision requires four or eight execution cores");
    // Scope the independent mode to the exact tower/merger at weight binding.
    moe_precision_ = true;
    linear_acc32_ = linear_acc32;
    gelu_erf_ultra_ = gelu_erf_ultra;
    invalidate_model_state();
}

void Qwen3_5VisionModel::set_linear_acc32(bool enabled) {
    TORCH_CHECK(!enabled || num_cores() == 4 || num_cores() == 8,
                "Qwen3.5 Vision ACC32 requires four or eight execution cores");
    TORCH_CHECK(enabled || !gelu_erf_ultra_,
                "Qwen3.5 ultra ERF GELU requires ACC32 Linear");
    TORCH_CHECK(
        layer_weights_.empty() && num_layers() == 0,
        "qwen3_5_vision_set_linear_acc32 must be called before set_weights");
    linear_acc32_ = enabled;
    invalidate_model_state();
}

void Qwen3_5VisionModel::set_gelu_erf_ultra(bool enabled) {
    TORCH_CHECK(!enabled || !high_precision_,
                "G05 HIGH math preserves the standard ERF merger formula");
    TORCH_CHECK(layer_weights_.empty() && num_layers() == 0,
                "qwen3_5_vision_set_gelu_erf_ultra must be called before set_weights");
    TORCH_CHECK(!enabled || (linear_acc32_ &&
                    (num_cores() == 4 || num_cores() == 8)),
                "Qwen3.5 ultra ERF GELU requires the cold four/eight-core ACC32 profile");
    gelu_erf_ultra_ = enabled;
    invalidate_model_state();
}

void Qwen3_5VisionModel::set_high_precision(bool enabled) {
    constexpr const char* operation = "qwen3_5_vision_set_high_precision";
    RpuExecutionCoordinator::require_graph_quiescent(operation);
    RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
    TORCH_CHECK(!RpuKernelGraph::has_active() && !high_precision_bound_ &&
                    layer_weights_.empty() && num_layers() == 0 &&
                    get_last_resolved_chunk_size() == 0,
                "G05 Vision math precision must be cold-bound once before weights");
    TORCH_CHECK(!enabled || (num_cores() == 8 && !gelu_erf_ultra_ &&
                    !moe_precision_ && !has_temporal_ && !z2_bound_),
                "G05 HIGH Vision math requires the ordinary eight-core K1 profile");
    if (enabled) rpu_require_high_precision_math_kernels();
    high_precision_ = enabled;
    high_precision_bound_ = true;
    invalidate_model_state();
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
    TORCH_CHECK(hidden_size == num_heads * head_dim,
                "qwen3_5_vision_set_weights: hidden_size=", hidden_size,
                " must equal num_heads*head_dim=", num_heads * head_dim);
    TORCH_CHECK(!high_precision_ ||
                    (num_cores() == 8 && N == 24 && num_heads == 16 &&
                     head_dim == 64 && hidden_size == 1024 &&
                     intermediate_size == 4096 && eps == 1e-6 &&
                     !has_temporal_ && !gelu_erf_ultra_ && !moe_precision_),
                "G05 HIGH Vision math requires the exact 24-layer K1 tower");
    TORCH_CHECK(!moe_precision_ ||
                    (N == 27 && num_heads == 16 && head_dim == 72 &&
                     hidden_size == 1152 && intermediate_size == 4304 &&
                     eps == 1e-6 && !has_temporal_),
                "MoE Vision precision requires the exact 35B-A3B tower");
    if (num_cores() != 8) {
        const bool small = N == 12 && num_heads == 12 && head_dim == 64 &&
                           hidden_size == 768 && intermediate_size == 3072;
        const bool regular = N == 24 && num_heads == 16 && head_dim == 64 &&
                             hidden_size == 1024 && intermediate_size == 4096;
        const bool padded = N == 27 && num_heads == 16 && head_dim == 72 &&
                            hidden_size == 1152 && intermediate_size == 4304;
        TORCH_CHECK(num_cores() == 4 && (small || regular || padded) && eps == 1e-6 &&
                        !has_temporal_,
                    "Reduced Qwen3.5 Vision requires an exact dense 0.8B/2B/4B/9B FP16 tower");
    }
    const int64_t padded_head_dim = Align(head_dim, 16);
    const int64_t padded_attention_size = num_heads * padded_head_dim;
    // Preserve the existing physical FP16 width (9B: 4304 -> 4352) across
    // core choices; cold swizzle alone repartitions it over the active cores.
    const int64_t padded_intermediate_size =
        Align(intermediate_size, 8 * 32);
    const int selected_attention_tp = resolve_model_execution_topology(
        num_heads, num_heads, padded_head_dim, hidden_size,
        padded_intermediate_size).attention_tp;
    TORCH_CHECK(selected_attention_tp > 0 && num_heads % selected_attention_tp == 0,
                "qwen3_5_vision_set_weights: invalid attention TP ", selected_attention_tp,
                " for num_heads=", num_heads);
    TORCH_CHECK(padded_attention_size % (16 * selected_attention_tp) == 0,
                "qwen3_5_vision_set_weights: padded attention size=",
                padded_attention_size,
                " must be divisible by fp16 attention row/col alignment ",
                16 * selected_attention_tp);
    TORCH_CHECK(padded_intermediate_size % (32 * num_cores()) == 0,
                "qwen3_5_vision_set_weights: padded intermediate_size=",
                padded_intermediate_size,
                " must be divisible by fp16 MLP core alignment ", 32 * num_cores());

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
            if (num_cores() != 8) {
                TORCH_CHECK(list[i].scalar_type() == at::kHalf &&
                                list[i].device().type() == at::kPrivateUse1 &&
                                list[i].is_contiguous(),
                            "Reduced Qwen3.5 Vision weights must be contiguous FP16 device tensors");
            }
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

    auto check_shape = [&](const at::TensorList& list, const char* name,
                           at::IntArrayRef expected) {
        for (int64_t i = 0; i < N; i++) {
            TORCH_CHECK(list[i].sizes() == expected,
                        "qwen3_5_vision_set_weights: ", name, "[", i,
                        "] shape=", list[i].sizes(), " expected=", expected);
        }
    };
    check_shape(q_w_list,   "q_w_list",   {padded_attention_size, hidden_size});
    check_shape(k_w_list,   "k_w_list",   {padded_attention_size, hidden_size});
    check_shape(v_w_list,   "v_w_list",   {padded_attention_size, hidden_size});
    check_shape(o_w_list,   "o_w_list",   {hidden_size, padded_attention_size});
    check_shape(fc1_w_list, "fc1_w_list", {padded_intermediate_size, hidden_size});
    check_shape(fc2_w_list, "fc2_w_list", {hidden_size, padded_intermediate_size});
    check_shape(ln1_w_list, "ln1_w_list", {hidden_size});
    check_shape(ln1_b_list, "ln1_b_list", {hidden_size});
    check_shape(ln2_w_list, "ln2_w_list", {hidden_size});
    check_shape(ln2_b_list, "ln2_b_list", {hidden_size});
    check_shape(q_b_list,   "q_b_list",   {padded_attention_size});
    check_shape(k_b_list,   "k_b_list",   {padded_attention_size});
    check_shape(v_b_list,   "v_b_list",   {padded_attention_size});
    check_shape(o_b_list,   "o_b_list",   {hidden_size});
    check_shape(fc1_b_list, "fc1_b_list", {padded_intermediate_size});
    check_shape(fc2_b_list, "fc2_b_list", {hidden_size});

    // Vision encoder is MHA (no GQA): num_kv_heads == num_q_heads.
    // The base model stores physical layout dims. Keep the real head dim
    // separately for RoPE and qk_scale.
    set_model_params(num_heads, num_heads, padded_head_dim,
                     hidden_size, padded_intermediate_size);
    attention_tp_ = attn_tp();
    set_num_layers(N);
    eps_ = eps;

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

    prepare_g05_owner_ln();
    invalidate_model_state();
}

void Qwen3_5VisionModel::set_weight_scales(
    at::TensorList q, at::TensorList k, at::TensorList v,
    at::TensorList o, at::TensorList fc1, at::TensorList fc2) {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0 &&
        !gelu_erf_ultra_ && !moe_precision_ && !has_temporal_ && num_cores() == 8 &&
        attn_tp() == 8 && mlp_tp() == 8 && num_layers() == 24 &&
        hidden_size() == 1024 && intermediate_size() == 4096 &&
        num_q_heads() == 16 && num_kv_heads() == 16 && head_dim() == 64 &&
        orig_head_dim_ == 64 && eps_ == 1e-6 && layer_weights_.size() == 24,
        "G05 Vision W8 scales require the cold exact K1 tower");
    for (const auto scales : {q,k,v,o,fc1,fc2})
        TORCH_CHECK(scales.size() == 24, "G05 Vision W8 requires 24 scales per projection");
    // Validate all owners before publishing any scale or invalidating identity.
    for (size_t layer = 0; layer < 24; ++layer) {
        const auto& w = layer_weights_[layer];
        const at::Tensor* weights[] = {&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.fc1_w,&w.fc2_w};
        const at::Tensor* scales[] = {&q[layer],&k[layer],&v[layer],&o[layer],&fc1[layer],&fc2[layer]};
        for (int projection = 0; projection < 6; ++projection) {
            const auto& weight = *weights[projection];
            const auto& scale = *scales[projection];
            TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kChar &&
                weight.device().type() == at::kPrivateUse1 && weight.is_contiguous() &&
                weight.dim() == 2 && scale.defined() && scale.scalar_type() == at::kHalf &&
                scale.device().type() == at::kPrivateUse1 && scale.is_contiguous() &&
                scale.dim() == 1 && scale.size(0) == weight.size(0),
                "G05 Vision W8 requires independent INT8 weights and FP16 scale [N]");
        }
    }
    for (size_t layer = 0; layer < 24; ++layer) {
        auto& w = layer_weights_[layer];
        w.q_ws=q[layer];w.k_ws=k[layer];w.v_ws=v[layer];w.o_ws=o[layer];
        w.fc1_ws=fc1[layer];w.fc2_ws=fc2[layer];
    }
    prepare_g05_owner_ln();
    invalidate_model_state();
}

// ── set_rope_tables: register FreqCos / FreqSin (validated fp16 device tables) +
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
    TORCH_CHECK(freq_cos.size(0) > 0 && freq_cos.size(0) <= int64_t{INT16_MAX} + 1,
                "qwen3_5_vision_set_rope: table height exceeds int16 position indices");
    TORCH_CHECK(orig_head_dim_ > 0,
                "qwen3_5_vision_set_rope: call set_weights() first");
    TORCH_CHECK(freq_cos.size(1) == orig_head_dim_ / 4,
                "qwen3_5_vision_set_rope: table width=", freq_cos.size(1),
                " expected real_head_dim/4=", orig_head_dim_ / 4);
    TORCH_CHECK(freq_cos.scalar_type() == at::kHalf && freq_sin.scalar_type() == at::kHalf,
                "qwen3_5_vision_set_rope: freq_cos / freq_sin must be fp16");
    TORCH_CHECK(freq_cos.device().type() == at::kPrivateUse1
                && freq_sin.device().type() == at::kPrivateUse1,
                "qwen3_5_vision_set_rope: freq_cos / freq_sin must be on device");
    TORCH_CHECK(freq_cos.is_contiguous() && freq_sin.is_contiguous(),
                "qwen3_5_vision_set_rope: freq_cos / freq_sin must be contiguous");

    freq_cos_ = freq_cos;
    freq_sin_ = freq_sin;
    max_hw_ = freq_cos.size(0);

    // Lazy-alloc on first call; reuse on subsequent calls (sizes never change).
    // rope_2d packs the DDR address as addr/256, while the device allocator only
    // guarantees 32B. Mirror the adapter's overallocate+narrow alignment scheme.
    if (!position_idx_keepalive_.defined()) {
        auto storage = at::empty(
            {keepalive_capacity_ * 2 + 128},
            at::TensorOptions().dtype(at::kShort).device(at::kPrivateUse1));
        const uint64_t raw = RpuGetDevAddr(storage.data_ptr<int16_t>());
        const int64_t off_bytes =
            (256 - static_cast<int64_t>(raw & 0xFFu)) & 0xFF;
        TORCH_CHECK(off_bytes % static_cast<int64_t>(sizeof(int16_t)) == 0,
                    "qwen3_5_vision_set_rope: cannot 256B-align position_idx keepalive");
        position_idx_keepalive_ = storage
            .narrow(0, off_bytes / static_cast<int64_t>(sizeof(int16_t)),
                    keepalive_capacity_ * 2)
            .view({keepalive_capacity_, 2});
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
                "qwen3_5_vision_set_patch_embed: pe_w/pe_b must be on device");
    TORCH_CHECK(pe_w.is_contiguous() && pe_b.is_contiguous(),
                "qwen3_5_vision_set_patch_embed: pe_w/pe_b must be contiguous");
    TORCH_CHECK(hidden_size() > 0,
                "qwen3_5_vision_set_patch_embed: call set_weights() first (needs hidden_size)");

    const int64_t h = hidden_size();
    TORCH_CHECK(pe_w.size(0) == h && pe_b.size(0) == h,
                "qwen3_5_vision_set_patch_embed: pe_w.size(0)=", pe_w.size(0),
                " pe_b.size(0)=", pe_b.size(0), " must both equal hidden=", h);
    const int64_t pd = pe_w.size(1);
    TORCH_CHECK(num_cores() == 8 || pd == 1536,
                "Reduced Qwen3.5 Vision patch embedding requires patch_dim=1536");
    // Col-parallel swizzle constraints (runtime/weights.py transform_linear_weight
    // partition=1, identical to rhino-ops ParallelLinearGenerator.py:346-353).
    // ⚠️ N=1024 also satisfies the ROW constraint (N%16), so a wrong `partition`
    // at swizzle time is NOT caught here — the only symptom is a wrong result.
    TORCH_CHECK(h % (16 * num_cores()) == 0,
                "qwen3_5_vision_set_patch_embed: hidden=", h,
                " must be divisible by 16*tp=", 16 * num_cores(), " (col swizzle)");
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
    TORCH_CHECK(!moe_precision_ || out_hidden_size == 2048,
                "MoE Vision precision requires the 35B-A3B merger output width 2048");

    auto check = [](const at::Tensor& t, const char* n, int64_t rank) {
        TORCH_CHECK(t.scalar_type() == at::kHalf,
                    "qwen3_5_vision_set_merger: ", n, " must be fp16");
        TORCH_CHECK(t.device().type() == at::kPrivateUse1,
                    "qwen3_5_vision_set_merger: ", n, " must be on device");
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
    const int64_t mh = h * QWEN3_5_SPATIAL_MERGE_UNIT;
    TORCH_CHECK(num_cores() == 8 ||
                    (h == 768 && out_hidden_size == 1024) ||
                    (h == 1024 && (out_hidden_size == 2048 || out_hidden_size == 2560)) ||
                    (h == 1152 &&
                     (out_hidden_size == 2048 || out_hidden_size == 4096 ||
                      out_hidden_size == 5120)),
                "Reduced Qwen3.5 Vision merger requires the exact "
                "0.8B/2B/4B/9B/27B/35B-A3B output width");
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
    TORCH_CHECK(mh % (16 * num_cores()) == 0,
                "qwen3_5_vision_set_merger: hidden*4=", mh,
                " must be divisible by 16*tp=", 16 * num_cores(),
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
    TORCH_CHECK(!high_precision_, "G05 HIGH Vision math requires single-frame cameras");
    TORCH_CHECK(num_cores() == 8,
                "Qwen3.5 Vision temporal profile requires eight execution cores");
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
                "qwen3_5_vision_set_temporal: temporal_pe must be contiguous fp16 device");
    TORCH_CHECK(temporal_pe.dim() == 2 && temporal_pe.size(0) == num_frames &&
                temporal_pe.size(1) == hidden_size(),
                "qwen3_5_vision_set_temporal: temporal_pe must be [6,1024], got ",
                temporal_pe.sizes());

    // Keep spatial capacity while admitting the exact temporal input geometry.
    set_chunk_envelope(keepalive_capacity_, 0);
    temporal_pe_ = temporal_pe;
    temporal_num_frames_ = num_frames;
    temporal_patches_per_frame_ = patches_per_frame;
    has_temporal_ = true;
    invalidate_model_state();
}

}  // namespace v3

using Qwen3_5VisionRegistry = ModelHandleRegistry<v3::Qwen3_5VisionModel>;

void rpu_qwen3_5_vision_set_cache_capacity(int64_t handle, int64_t max_seq_len) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_set_cache_capacity")
        ->configure_cache_capacity(max_seq_len);
}

void rpu_qwen3_5_vision_prepare_persistent_spm(int64_t handle, int64_t execution_len) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_prepare_persistent_spm")
        ->prepare_persistent_spm(execution_len, 0, false);
}

std::vector<int64_t> rpu_qwen3_5_vision_planner_cache_identity(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_qwen3_5_vision_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen3_5_vision_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_kvinsert_exact_candidate")
        ->mint_vision_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_qwen3_5_vision_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_kvinsert_cost_domain")
        ->kvinsert_cost_domain("qwen3_5_vision", descriptor);
}

std::string rpu_qwen3_5_vision_kvinsert_cost_catalog_sha256(int64_t handle) {
    return Qwen3_5VisionRegistry::get(
        handle, "rpu_qwen3_5_vision_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers
// =============================================================================

int64_t rpu_qwen3_5_vision_create() {
    return Qwen3_5VisionRegistry::create();
}

void rpu_qwen3_5_vision_set_execution_cores(int64_t handle, int64_t num_cores) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_set_execution_cores")
        ->configure_execution_cores(num_cores);
}

bool rpu_qwen3_5_vision_k1_b3_available(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle,"g05_k1_b3_available")->g05_k1_b3_available();
}

bool rpu_qwen3_5_vision_k1_camera_group_available(int64_t handle,int64_t cameras) {
    return Qwen3_5VisionRegistry::get(handle,"g05_k1_camera_group_available")->g05_k1_camera_group_available(cameras);
}

std::vector<int64_t> rpu_qwen3_5_vision_get_execution_topology(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_get_execution_topology")
        ->execution_topology();
}

void rpu_qwen3_5_vision_set_moe_precision(
    int64_t handle, bool linear_acc32, bool gelu_erf_ultra) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_set_moe_precision")
        ->set_moe_precision(linear_acc32, gelu_erf_ultra);
}

void rpu_qwen3_5_vision_set_linear_acc32(int64_t handle, bool enabled) {
    Qwen3_5VisionRegistry::get(
        handle, "rpu_qwen3_5_vision_set_linear_acc32")
        ->set_linear_acc32(enabled);
}

void rpu_qwen3_5_vision_set_high_precision(int64_t handle, bool enabled) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_set_high_precision")
        ->set_high_precision(enabled);
}

void rpu_qwen3_5_vision_set_gelu_erf_ultra(int64_t handle, bool enabled) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_set_gelu_erf_ultra")
        ->set_gelu_erf_ultra(enabled);
}

void rpu_qwen3_5_vision_destroy(int64_t handle) {
    v3::qwen3_5_z2_internal::check_vision_destroy_allowed(handle);
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_destroy")
        ->check_execution_reconfigure_destroy_allowed(
            "rpu_qwen3_5_vision_destroy");
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

void rpu_qwen3_5_vision_set_weight_scales(
    int64_t handle, at::TensorList q, at::TensorList k, at::TensorList v,
    at::TensorList o, at::TensorList fc1, at::TensorList fc2) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_set_weight_scales")
        ->set_weight_scales(q,k,v,o,fc1,fc2);
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
    int64_t camera_batch_count,
    int64_t expected_stage_plan_fingerprint_hi,
    int64_t expected_stage_plan_fingerprint_lo,
    int64_t expected_layout_hash_hi,
    int64_t expected_layout_hash_lo,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision")
        ->forward(input, k_caches, v_caches, num_patches, step0_pos,
                  fusion_target, fusion_row_starts, temporal_num_frames,
                  camera_batch_count, expected_stage_plan_fingerprint_hi,
                  expected_stage_plan_fingerprint_lo,
                  expected_layout_hash_hi,
                  expected_layout_hash_lo,
                  planned_stage_descriptor);
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

// Per-handle SPM-budget-resolved chunk_size (0 before any forward). Mirrors
// rpu_qwen3_5_get_resolved_chunk_size (rpu_qwen3_5_model.cpp:1297).
int64_t rpu_qwen3_5_vision_get_resolved_chunk_size(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

std::vector<int64_t> rpu_qwen3_5_vision_get_prefill_execution_controls(int64_t handle) {
    return Qwen3_5VisionRegistry::get(
        handle, "rpu_qwen3_5_vision_get_prefill_execution_controls")
        ->get_prefill_execution_controls();
}

// Planning-only query. It serializes the same staged descriptor consumed by
// forward, including semantic spans and the independent merger schedule.
std::vector<int64_t> rpu_qwen3_5_vision_resolve_prefill_plan(
    int64_t handle, int64_t execution_len, int64_t temporal_num_frames,
    int64_t camera_batch_count, bool compact_input) {
    return Qwen3_5VisionRegistry::get(
        handle, "rpu_qwen3_5_vision_resolve_prefill_plan")
        ->resolve_prefill_plan(
            execution_len, temporal_num_frames, camera_batch_count,
            compact_input);
}

std::vector<std::vector<int64_t>> rpu_qwen3_5_vision_resolve_prefill_domain(
    int64_t handle, int64_t execution_len, int64_t temporal_num_frames,
    int64_t camera_batch_count, bool compact_input) {
    return Qwen3_5VisionRegistry::get(
        handle, "rpu_qwen3_5_vision_resolve_prefill_domain")
        ->resolve_prefill_domain(
            execution_len, temporal_num_frames, camera_batch_count, compact_input);
}

// MR-C/T34: cold per-handle vision chunk cap. The adapter translates the
// canonical request (or legacy alias) before the first forward; native planning
// never reads that environment variable.
void rpu_qwen3_5_vision_set_chunk_size_cap(int64_t handle, int64_t cap) {
    Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_set_chunk_size_cap")
        ->set_chunk_size_cap(cap);
}

void rpu_qwen3_5_vision_set_prefill_chunk_size(
    int64_t handle, int64_t chunk_size) {
    Qwen3_5VisionRegistry::get(
        handle, "rpu_qwen3_5_vision_set_prefill_chunk_size")
        ->set_prefill_chunk_size(chunk_size);
}

void rpu_qwen3_5_vision_stage_prefill_execution_controls(
        int64_t handle, int64_t token, int64_t cap, int64_t chunk_size) {
    TORCH_CHECK(token > 0,
                "Qwen3.5 vision hot-reconfigure token must be positive");
    Qwen3_5VisionRegistry::get(
        handle, "rpu_qwen3_5_vision_stage_prefill_execution_controls")
        ->stage_prefill_execution_controls(
            static_cast<uint64_t>(token), cap, chunk_size);
}

// Per-layer debug snapshots (empty tensor until a get_debug_export() forward runs).
// dbg_hidden = [num_layers, N, hidden]; dbg_q = [num_layers, attention_tp, N, local_q_dim].
at::Tensor rpu_qwen3_5_vision_get_dbg_hidden(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_get_dbg_hidden")->dbg_hidden();
}
at::Tensor rpu_qwen3_5_vision_get_dbg_q(int64_t handle) {
    return Qwen3_5VisionRegistry::get(handle, "rpu_qwen3_5_vision_get_dbg_q")->dbg_q();
}
