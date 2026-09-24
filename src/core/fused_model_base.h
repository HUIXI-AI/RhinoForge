// fused_model_base.h — Flat single-class framework contract (Plan 01-01)
//
// Collapses v2's FusedLayerBase + FusedModelBase into ONE public header.
// Exposes: 4 mandatory virtuals (D-201/202), narrow facade (D-203), run_all_layers.
// Internal state hides behind PImpl (Impl defined in fused_model_base_impl.h).
// v2 headers untouched during shim-coexist (D-301) until Plan 01-07b.
//
// DEVIATION NOTE (Rule 3, blocking issue found during Task 2 build):
//   v2's src/core/fused_model_base.{h,cpp} already defines a top-level
//   `class FusedModelBase`. Defining a second top-level class with the same
//   name in this TU causes a linker collision (multiple definition). To
//   satisfy D-301 (shim-coexist, both compile alongside), v3 lives inside
//   `namespace v3 { ... }`. Plan 01-07b's atomic cleanup removes the v2
//   files AND lifts v3's class out of the namespace into the canonical
//   top-level, then renames fused_model_base.{h,cpp} →
//   fused_model_base.{h,cpp}. No subclass code changes during the rename.

#pragma once

#include <ATen/ATen.h>
#include <c10/util/ScopeExit.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "ops/rpu_kvinsert_segment_plan.h"
#include "execution_topology.h"
#include "rpu_math_precision.h"

class RpuKernelGraph;
class GraphDmaSemanticEndpoint;
class GraphKernelRegisterCensusGuard;
class SdpaStableMaskCache;
class KvInsertSegmentPlan;

namespace v3 {

class FusedModelBase;  // forward decl for pointer-to-member in ModelStaticConfig
class SpmPipelineLease;
class SpmTensorView;
class SpmScratchId;
class SpmChunkKey;
struct SpmDense2DSpec;
class SpmFmbSealedPhaseManifest;
class SpmFmbResolvedExecutionProfile;
class SpmFmbResolvedPhaseManifest;
class SpmFmbConsumerRowSliceEndpoint;
class SpmFmbPostFnYieldScope;
class SpmFmbSealedPostFnYields;
class SpmFmbSealedCallbackYields;
class SpmFmbCompletedBuildTrace;
class SpmCompositeTraceCoordinator;
class SpmFmbSealedBuildTrace;
class SpmFmbSealedActiveGroups;
class SpmFmbSealedDenseDdrMembers;
class SpmFmbSealedDmaEndpointBindings;
class SpmFmbSealedSpmPeerBindings;
struct SpmFmbResolvedExecutionStep;
struct SpmFmbOccurrenceSchedule;
struct SpmFmbConsumerOccurrenceSelector;

// Public enums keep their v2 numeric values; new values are append-only.
enum class ChunkMode       : int { SEQUENTIAL = 0, KV_FIRST = 1 };
enum class InterLayerIO    : int { AUTO = 0, SPM_RESIDENT = 1, DDR_PINGPONG = 2 };
// AUTO is the public default. SPM_KV_BY_MHA means raw transient K/V remain in
// SPM and a model-certified by-MHA kernel consumes them; ordinary
// FLASH_ATTN_SPM still resolves to DDR_KV because its K/V operands are DDR
// RPUCache tensors. Large/unsupported contexts permanently retain DDR_KV.
enum class AttentionExecutionPolicy : int {
    AUTO = 0,
    DDR_KV = 1,
    SPM_KV_BY_MHA = 2,
};
// Resolved table residency carried by a COMPLETE physical manifest.  It is a
// layout input, not a public tuning field: UNSPECIFIED preserves every owner
// that has no route-dependent RoPE allocation.
enum class FmbRopeTableResidency : int {
    UNSPECIFIED = 0,
    DDR = 1,
    SPM = 2,
};
enum class ActivationKind  : int {
    NONE = 0,
    SILU = 1,
    GELU = 2,
};
enum class StorageClass { Temp, TempPerLayer, Persistent, PersistentPerLayer };
// Buffer lifetime scopes. LayerWide spans every layer-loop subgraph;
// KvInsert/Compute are the KV_FIRST two phases and never coexist.
//
// OutsideLayerLoop is for temporary SPM used by pre_layers_fn or
// post_layers_fn. Its data must cross the layer-loop boundary through DDR, so
// the SPM allocation itself never coexists with a layer-loop allocation.
// Phase ranges still describe overlap between OutsideLayerLoop buffers (for
// example, disjoint pre-layer and post-layer phases).
enum class BufferScope : uint8_t {
    LayerWide = 0,
    KvInsert = 1,
    Compute = 2,
    OutsideLayerLoop = 3,
};

// Buffer / layout / chunk structs (verbatim from v2, plus D-502 field)
struct BufferDecl {
    const char*  name         = nullptr;
    int64_t      size         = 0;
    int          phase_start  = 0;
    int          phase_end    = 0;
    StorageClass storage      = StorageClass::Temp;
    int          per_layer    = 0;
    const char*  alias_of     = nullptr;
    BufferScope  scope        = BufferScope::LayerWide;

    // D-502: for Persistent / PersistentPerLayer buffers, framework auto-wires:
    //   (1) captures SPM_ALLOC.persistent_generation() at allocation
    //   (2) re-fires on generation advance OR on preload_callbacks_dirty_ (EXT-4)
    //   (3) passes layer_idx (-1 for non-per-layer) + core-0 absolute SPM addr
    //
    // EXT-5 graph-capture contract (LOCKED):
    //   The caller's GraphCache/raw-Graph scope owns capture. Every rpu_launch_*
    //   emitted by the complete preload-callback loop enters that active graph
    //   through RpuQueue/graph_dma. Callbacks MUST NOT open a nested graph or raw
    //   batch scope themselves.
    //   Audit (must return 0 in every ported model):
    //     grep -c 'BATCH_CTX\.begin\|BATCH_CTX\.end' src/fused/<model>.cpp
    std::function<void(FusedModelBase& /*self*/,
                       int             /*layer_idx*/,
                       uint32_t        /*core0_addr*/)> preload_callback;

    // B1 (AdaRMS fused broadcast): allocate this PersistentPerLayer decl's layers
    // in DESCENDING layer order (L = per_layer-1 .. 0). alloc_super_persistent is a
    // DOWNWARD bump allocator (rpu_spm_allocator.cpp:172), so the default ascending
    // loop puts layer 0 at the HIGHEST address — the reverse of every [num_layers, h]
    // DDR-side buffer. Allocating descending makes layer L sit at
    // `layer_addr(0) + L*align_up(size)`, i.e. SAME direction as DDR, so all
    // `per_layer` layers can be filled by ONE contiguous DMA instead of `per_layer`
    // tiny ones. Purely an address permutation — every consumer resolves addresses
    // through layer_addr()'s per-layer map, so nothing else observes the change.
    // Default false ⇒ byte-identical allocation for every existing decl.
    bool reverse_layer_alloc = false;
};

namespace detail {
// Fold one extra layout input into a FusedModelBase::subclass_layout_hash()
// accumulator (splitmix64 finalizer). Chain it — `h = layout_mix(h, x)` — rather
// than XOR-ing raw values together: two equal raw values would cancel, and a
// bare small integer collides badly against the params-hash terms it is XORed
// into. Order-sensitive, which is what you want when two members can hold the
// same number.
inline int64_t layout_mix(int64_t acc, int64_t value) {
    uint64_t x = static_cast<uint64_t>(acc) * 0x9e3779b97f4a7c15ull
               + static_cast<uint64_t>(value) + 0x165667b19e3779f9ull;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27; x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return static_cast<int64_t>(x);
}

// Pure SPM-budget estimator used by chunk planning and CPU-only contract tests.
int64_t estimate_temporary_total(const std::vector<BufferDecl>& decls);

struct FmbPreparedSpmCost {
    int64_t persistent = 0;
    int64_t temp_per_layer = 0;
    int64_t temporary = 0;
    int64_t available(int64_t after_reset_free, bool persistent_allocated,
                      int64_t planning_budget) const;
    bool fits(int64_t after_reset_free, bool persistent_allocated,
              int64_t planning_budget) const;
};

// Cache pure allocator planning, not an admission result or live resource
// state. Match the full current declaration input, including alias presence
// and lifetime, rather than assuming every model's layout hash is complete.
class FmbPreparedSpmCostCache {
public:
    static constexpr size_t kCapacity = 32;
    FmbPreparedSpmCost prepare(const std::vector<BufferDecl>& declarations);
    void clear() { entries_.clear(); }
    size_t size() const { return entries_.size(); }
private:
    struct Input {
        int64_t size;
        int phase_start, phase_end;
        StorageClass storage;
        int per_layer;
        BufferScope scope;
        bool alias;
        bool matches(const BufferDecl& declaration) const;
    };
    struct Entry {
        std::vector<Input> inputs;
        FmbPreparedSpmCost cost;
    };
    std::list<Entry> entries_;
};
// Prefer fewer chunks, then the most even fixed-size chunk + tail split.
bool prefer_balanced_chunk(int64_t seq_len, int64_t candidate,
                           int64_t current);
}  // namespace detail

enum class FmbForwardOperandResidency : uint8_t {
    UNSPECIFIED = 0,
    PER_LAYER = 1,
    FORWARD = 2,
};

struct LayoutContext {
    int64_t chunk_size = 0;
    int64_t kv_insert_chunk_size = 0;  // 0 = same as chunk_size
    int64_t max_kv_seq_len = 0;
    int64_t num_layers = 0;
    bool    use_attn_mask = false;
    // the attention layout mode (causal vs bidirectional-
    // no-mask) drives subclass declare_buffers layout (KV_FIRST q_kv split, k/v
    // sizing, buffer scopes, sdpa_tmp mask — see CausalDecoderModel). It MUST be
    // part of the allocation identity (compute_params_hash_impl) so a handle that
    // switches modes at the same chunk_size re-allocates instead of reusing the
    // wrong layout. declare_buffers derives kv_first_layout from THIS field, not
    // from the runtime InferenceContext, so the layout is a pure function of the
    // hashed LayoutContext. Default true = causal (zero hash perturbation for all
    // existing causal/explicit-mask allocations).
    bool    is_causal = true;

    // Batch decode (Qwen3): number of independent sequences whose rows are
    // packed into the GEMM M dimension. Row-parallel buffers (residual/q/k/v/
    // mlp/...) are sized batch_size * chunk_size; per-sequence buffers
    // (sdpa_tmp) stay at chunk_size because SDPA is launched once per sequence.
    // Part of the allocation identity — B=8 and B=16 need distinct layouts.
    // Default 1 = zero hash perturbation for every existing allocation.
    int64_t batch_size = 1;

    // Non-zero only for a resolved run_all_layers plan. It prevents two
    // same-shape calls with different input partitions/semantic spans from
    // reusing a layout selected by a plan-sensitive specialization. Absolute
    // kv_seq_len/position is intentionally excluded: decode must not allocate
    // a fresh layout at every token.
    uint64_t stage_plan_fingerprint = 0;
    // Non-zero only for a COMPLETE physical descriptor. Route selectors can
    // alter BufferDecl shape without changing the logical stage plan, so this
    // identity independently prevents incompatible allocation reuse.
    uint64_t physical_manifest_fingerprint = 0;
    // Resolved value only (never AUTO) once final allocation begins.
    AttentionExecutionPolicy attention_policy =
        AttentionExecutionPolicy::DDR_KV;
    FmbRopeTableResidency rope_table_residency =
        FmbRopeTableResidency::UNSPECIFIED;
    // Selected by an adopting owner's COMPLETE physical schedule. Never infer
    // this from live free space in declare_buffers() or during Graph replay.
    FmbForwardOperandResidency forward_operand_residency =
        FmbForwardOperandResidency::UNSPECIFIED;
    // Cold oracle hint only: the configurable capacity can exceed the actual
    // first chunk (e.g. M18/C32). A growing candidate must also fit that exact
    // capacity probe. This is not an allocation input or part of its hash.
    int64_t planning_chunk_capacity = 0;

    int64_t effective_kv_cs() const {
        return kv_insert_chunk_size > 0 ? kv_insert_chunk_size : chunk_size;
    }
    // Rows packed into one GEMM: batch-major, [batch][chunk] contiguous.
    int64_t rows() const { return batch_size * chunk_size; }
};

struct ChunkInfo { int idx = 0; int64_t offset = 0; int64_t len = 0; int64_t kv_seq_len = 0; };
struct ChunkPlan { int64_t chunk_size = 0; int64_t num_chunks = 0; };

// Resolved three-stage execution contract shared by language, action, and
// Vision models. The input stage is owned by the caller or a model pre-layer
// hook, outside the canonical layer traversal. For
// SEQUENTIAL, qkv and compute describe the same ordinary chunks; for KV_FIRST,
// qkv describes kv_insert_chunks and compute describes the layer body. This
// host-side description reuses the canonical allocator/traversal; it does not
// add a second executor or select a model kernel.
struct ResolvedFmbStageChunks {
    ChunkPlan plan;
    std::vector<ChunkInfo> chunks;
};

enum class FmbSpanBoundaryPolicy : uint8_t {
    ALLOW_CROSS,
    KEEP_LOCAL,
    GROUP_LOCAL,
};

struct FmbStageBoundaryPolicies {
    FmbSpanBoundaryPolicy input = FmbSpanBoundaryPolicy::ALLOW_CROSS;
    FmbSpanBoundaryPolicy qkv = FmbSpanBoundaryPolicy::ALLOW_CROSS;
    FmbSpanBoundaryPolicy compute = FmbSpanBoundaryPolicy::ALLOW_CROSS;
};

// One independent semantic span on the physical sequence axis. Vision uses
// image boundaries; language/action default to one full-sequence span and may
// specialize when several independent streams are packed along sequence.
struct FmbExecutionSpan {
    int64_t offset = 0;
    int64_t len = 0;
    int64_t group_id = -1;
};

struct FmbThreeStageChunkPlan {
    ResolvedFmbStageChunks input;
    ResolvedFmbStageChunks qkv;
    ResolvedFmbStageChunks compute;
    std::vector<FmbExecutionSpan> spans;
    FmbStageBoundaryPolicies boundary_policies;
    ChunkMode chunk_mode = ChunkMode::SEQUENTIAL;
};

// Historical SISC/SIMC/MISC/MIMC labels are retained, with "input" meaning
// one semantic span rather than specifically one image.
enum class FmbChunkTopology : uint8_t {
    SISC,
    SIMC,
    MISC,
    MIMC,
};

FmbThreeStageChunkPlan compose_fmb_three_stage_chunk_plan(
    std::vector<ChunkInfo> input_chunks,
    std::vector<ChunkInfo> qkv_chunks,
    std::vector<ChunkInfo> compute_chunks,
    std::vector<FmbExecutionSpan> spans,
    ChunkMode chunk_mode);
FmbThreeStageChunkPlan compose_fmb_three_stage_chunk_plan(
    std::vector<ChunkInfo> input_chunks,
    std::vector<ChunkInfo> qkv_chunks,
    std::vector<ChunkInfo> compute_chunks,
    std::vector<FmbExecutionSpan> spans,
    ChunkMode chunk_mode,
    FmbStageBoundaryPolicies boundary_policies);
// Common six-argument run_all_layers plan: one full-sequence input stage and
// one semantic span, with QKV/compute schedules derived from the exact
// allocation chunk sizes. Physical-pipeline prepare uses this to seal the same
// plan identity before BUILD that run_all_layers will resolve inside BUILD.
FmbThreeStageChunkPlan compose_fmb_default_three_stage_chunk_plan(
    const LayoutContext& layout,
    int64_t execution_len,
    int64_t position,
    ChunkMode chunk_mode);
void validate_fmb_three_stage_chunk_plan(
    const FmbThreeStageChunkPlan& plan);
FmbChunkTopology fmb_chunk_topology(
    const FmbThreeStageChunkPlan& plan);
uint64_t fmb_three_stage_chunk_plan_fingerprint(
    const FmbThreeStageChunkPlan& plan);

namespace detail {
int64_t validate_resolved_chunk_coverage(
    const std::vector<ChunkInfo>& chunks,
    const char* contract,
    const char* role);
void validate_fmb_execution_spans(
    const std::vector<FmbExecutionSpan>& spans,
    int64_t execution_len,
    const char* contract);
bool fmb_chunks_respect_boundary_policy(
    const std::vector<ChunkInfo>& chunks,
    const std::vector<FmbExecutionSpan>& spans,
    FmbSpanBoundaryPolicy policy);
void validate_fmb_planning_shape(
    int64_t execution_len,
    int64_t position,
    const char* contract);
void validate_fmb_exact_chunk_size(
    int64_t requested_chunk_size,
    int64_t execution_len,
    const char* contract);
AttentionExecutionPolicy resolve_attention_execution_policy(
    AttentionExecutionPolicy requested,
    bool model_kernel_eligible,
    bool exact_spm_layout_fits);

// Logical coverage remains canonical. An owner may separately opt into the
// physical KVIN1, KVIN0, COMP0, COMP1 schedule for one equal, zero-position
// pair. Both the profile builder and runtime traversal use this same gate.
inline bool fmb_kv_first_pair_carry_eligible(
    const std::vector<ChunkInfo>& compute,
    const std::vector<ChunkInfo>& kv_insert, int64_t rows) {
    if (rows <= 0 || rows > std::numeric_limits<int64_t>::max() / 2 ||
        compute.size() != 2 || kv_insert.size() != 2) return false;
    for (int i = 0; i != 2; ++i) {
        for (const auto* chunks : {&compute, &kv_insert}) {
            const auto& c = (*chunks)[i];
            if (c.idx != i || c.offset != i * rows || c.len != rows ||
                c.kv_seq_len != (i + 1) * rows) return false;
        }
    }
    return true;
}
}  // namespace detail

// Board-free causal-prefill planning result.  `resolved_chunk_size` is the
// v16 candidate selected by the planner; `allocation_layout.chunk_size` is the
// first real ChunkInfo length used by run_all_layers for SPM allocation.  They
// intentionally differ for a short one-chunk tail such as S225 (240 vs 225).
struct SpmPipelineCausalPrefillShape {
    int64_t resolved_chunk_size = 0;
    std::vector<ChunkInfo> chunks;
    std::vector<ChunkInfo> kv_insert_chunks;
    LayoutContext allocation_layout;
    FmbThreeStageChunkPlan stage_plan;
    ChunkMode chunk_mode = ChunkMode::SEQUENTIAL;
    InterLayerIO inter_layer_io = InterLayerIO::AUTO;
    bool chunk_outer_within_group = false;
};

// Version-2 physical descriptor wire.  The codec is deliberately capable of
// carrying every native selector without assigning model-specific meaning to
// selector/argument values: each owner must publish and validate those values
// before a COMPLETE manifest is admitted by forward.  UNSPECIFIED is the
// migration-safe state emitted by owners that have not supplied that contract;
// it must never be interpreted as a default route.
enum class FmbPhysicalManifestState : int {
    UNSPECIFIED = 0,
    COMPLETE = 1,
};

enum class FmbGraphLifecycle : int {
    UNSPECIFIED = 0,
    RETAINED_CACHE = 1,
    BOUNDED_ONESHOT = 2,
    NATIVE_COMPOSITE1 = 3,
    COMPOSITE_CHILD = 4,
};

enum class FmbLinearAccumulationPolicy : int {
    UNSPECIFIED = 0,
    ACC16 = 1,
    ACC32 = 2,
    MIXED_BY_SITE = 3,
};

enum class FmbRouteFamily : int {
    ATTENTION = 1,
    LINEAR = 2,
    ALL_REDUCE = 3,
    ROPE = 4,
    KV_INSERT = 5,
    MUTABLE_DMA = 6,
    NORMALIZATION = 7,
    ACTIVATION = 8,
    GRAPH_SCHEDULE = 9,
    COLLECTIVE = 10,
};

// Shared high-bit route flags.  Owner-local low bits remain available for
// kernel-specific reasons; these two bits make table residency recoverable
// from the immutable descriptor before declare_buffers() runs.
inline constexpr int64_t FMB_ROUTE_FLAG_ROPE_TABLE_DDR = 1LL << 56;
inline constexpr int64_t FMB_ROUTE_FLAG_ROPE_TABLE_SPM = 1LL << 57;

// Shared LINEAR selector IDs.  A COMPLETE owner manifest must name the
// physical launcher route; AUTO_TILE is an already-resolved route here, not
// an instruction to choose again during forward.
enum class FmbLinearRouteSelector : int {
    AUTO_TILE = 1,
    GEMV = 2,
    ACC32_OUT_BF16 = 4,
    ROW_WEIGHT_REUSE = 5,
    PI05_NVFP4_V2_ACC32 = 6,
    PI05_NVFP4_V2_ACC16 = 7,
};

// Shared FusedModelBase emitters are native physical sites in their own
// right.  Owners add the subset they actually use to their COMPLETE manifest
// with append_fmb_shared_runtime_routes(); the emitter then consumes the same
// entry immediately before dispatch.  Keeping this list here avoids copying
// shared source-site IDs into every model.
inline constexpr int64_t FMB_SHARED_LAYER_INPUT_DMA_SITE =
    8256462186627229745LL;
inline constexpr int64_t FMB_SHARED_LAYER_INPUT_ROW_RUN_DMA_SITE =
    7164742071837285576LL;
inline constexpr int64_t FMB_SHARED_PIPELINE_INGRESS_DMA_SITE =
    1763009975491617456LL;
inline constexpr int64_t FMB_SHARED_MLP_AUTO_TILE_SITE =
    6066351929848019041LL;
inline constexpr int64_t FMB_SHARED_MLP_ACC32_OUT_BF16_SITE =
    5407567195452051675LL;
inline constexpr int64_t FMB_SHARED_MLP_BF16_FP16_REDUCE_SITE =
    8878474719496009170LL;
inline constexpr int64_t FMB_SHARED_MLP_FP16_BF16_REDUCE_SITE =
    1641660405853831496LL;
inline constexpr int64_t FMB_SHARED_MLP_RING_REDUCE_SITE =
    1201194854597513173LL;
inline constexpr int64_t FMB_SHARED_MLP_PREPARE_INPUT_SITE =
    47322565134527036LL;
inline constexpr int64_t FMB_SHARED_MLP_SILU_MUL_SITE =
    379290273870460553LL;

enum FmbSharedRuntimeRouteMask : uint32_t {
    FMB_SHARED_LAYER_INPUT_DMA = 1U << 0,
    FMB_SHARED_LAYER_INPUT_ROW_RUN_DMA = 1U << 1,
    FMB_SHARED_PIPELINE_INGRESS_DMA = 1U << 2,
    FMB_SHARED_MLP_AUTO_TILE = 1U << 3,
    FMB_SHARED_MLP_ACC32_OUT_BF16 = 1U << 4,
    FMB_SHARED_MLP_BF16_FP16_REDUCE = 1U << 5,
    FMB_SHARED_MLP_FP16_BF16_REDUCE = 1U << 6,
    FMB_SHARED_MLP_RING_REDUCE = 1U << 7,
    FMB_SHARED_MLP_PREPARE_INPUT = 1U << 8,
    FMB_SHARED_MLP_GEMV = 1U << 9,
};

enum class FmbSharedMutableDmaRouteSelector : int64_t {
    DDR_BROADCAST_TO_SPM = 1,
    CANONICAL_DDR_BROADCAST_TO_SPM_MUTABLE_SRC = 2,
};

enum class FmbSharedAllReduceRouteSelector : int64_t {
    RING_NOPACE = 1,
    RING_PACED = 2,
    PREPARE_RING_INPUT = 3,
    FP16_PARTIAL_BF16_RESIDUAL = 4,
    BF16_PARTIAL_FP16_RESIDUAL = 5,
    RING_PI05_XOR3 = 6,
};

struct FmbRouteManifestEntry {
    // Positive, owner-stable native call-site ID.  Repeated physical
    // invocations of that site use the zero-based invocation discriminator;
    // (family, site_id, invocation) is unique and encoded in that order.
    int64_t site_id = 0;
    FmbRouteFamily family = FmbRouteFamily::ATTENTION;
    // Append-only selector ID and non-negative physical arguments (for
    // example, a hybrid KV-insert segment partition). ATTENTION is the shared
    // exception to owner-defined selector semantics: selector is exactly a
    // resolved AttentionExecutionPolicy (DDR_KV or SPM_KV_BY_MHA).
    // Site-specific attention-kernel variants may use flags/arguments, but
    // forward must consume those exact values through consume_physical_route.
    int64_t selector = 0;
    int64_t flags = 0;
    std::vector<int64_t> arguments;
    int64_t invocation = 0;
};

struct FmbPhysicalExecutionManifest {
    FmbPhysicalManifestState state =
        FmbPhysicalManifestState::UNSPECIFIED;
    // Execution geometry is independent from KV-cache horizon and from the
    // physical row count emitted by the selected KV-insert route.
    int64_t logical_length = 0;
    int64_t physical_length = 0;
    int64_t execution_padding_rows = 0;
    int64_t kv_logical_length = 0;
    int64_t kv_insert_physical_rows = 0;
    FmbGraphLifecycle graph_lifecycle = FmbGraphLifecycle::UNSPECIFIED;
    FmbLinearAccumulationPolicy linear_accumulation =
        FmbLinearAccumulationPolicy::UNSPECIFIED;
    std::vector<FmbRouteManifestEntry> routes;
};

int64_t fmb_ring_all_reduce_route_selector(int64_t rows, int64_t cols,
                                          int num_cores = 8);
int64_t fmb_ring_all_reduce_route_selector(int64_t rows, int64_t cols, bool pi05_xor3, int num_cores = 8);
void append_fmb_shared_runtime_routes(
    FmbPhysicalExecutionManifest& manifest,
    const FmbThreeStageChunkPlan& plan,
    int64_t hidden_size,
    uint32_t route_mask,
    int64_t compute_row_multiplier = 1,
    int num_cores = 8,
    int mlp_num_cores = 8);
void append_fmb_shared_runtime_routes(
    FmbPhysicalExecutionManifest& manifest, const FmbThreeStageChunkPlan& plan,
    int64_t hidden_size, uint32_t route_mask, int64_t compute_row_multiplier,
    bool pi05_xor3, int num_cores = 8, int mlp_num_cores = 8);

// Explicit native opt-in for COMPLETE descriptors. The zero/default value is
// deliberately incapable of accepting authority; an adopting owner must also
// declare the Graph lifecycle of the current call.
struct FmbPhysicalManifestForwardCapability {
    bool complete_descriptor = false;
    FmbGraphLifecycle graph_lifecycle = FmbGraphLifecycle::UNSPECIFIED;
};

uint64_t fmb_physical_manifest_fingerprint(
    const FmbPhysicalExecutionManifest& manifest);
void validate_fmb_physical_manifest(
    const FmbPhysicalExecutionManifest& manifest,
    int64_t stage_physical_length,
    int64_t position_base);
void validate_fmb_physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& manifest,
    const FmbPhysicalManifestForwardCapability& capability);
AttentionExecutionPolicy fmb_attention_execution_policy(
    const FmbRouteManifestEntry& route);
FmbRopeTableResidency fmb_rope_table_residency(
    const FmbPhysicalExecutionManifest& manifest);

// One per-forward route receipt. Lookup is read-only; consume_route records a
// site only after the caller states the selector/flags/arguments it actually
// dispatched. Repeated loop invocations of one semantic site are valid, while
// missing/wrong-family/mismatched policy and omitted sites fail closed.
class FmbPreparedStageCandidate;

class FmbPhysicalManifestConsumer {
public:
    FmbPhysicalManifestConsumer() = default;
    explicit FmbPhysicalManifestConsumer(
        const FmbPhysicalExecutionManifest& manifest) {
        reset(manifest);
    }

    void reset(const FmbPhysicalExecutionManifest& manifest);
    void reset_prepared(
        std::shared_ptr<const FmbPreparedStageCandidate> prepared);
    void retain_consumed_if_matching(
        const FmbPhysicalExecutionManifest& manifest) const;
    bool complete() const;
    const FmbPhysicalExecutionManifest& manifest() const;
    uint64_t fingerprint() const;
    const FmbRouteManifestEntry& find_route(
        FmbRouteFamily family, int64_t site_id,
        int64_t invocation = 0) const;
    const FmbRouteManifestEntry& consume_route(
        FmbRouteFamily family,
        int64_t site_id,
        int64_t resolved_selector,
        int64_t resolved_flags,
        at::IntArrayRef resolved_arguments = {},
        int64_t invocation = 0) const;
    AttentionExecutionPolicy attention_policy_for_site(
        int64_t site_id, int64_t invocation = 0) const;
    std::optional<AttentionExecutionPolicy> attention_layout_policy() const;
    const FmbRouteManifestEntry& consume_attention_route(
        int64_t site_id,
        AttentionExecutionPolicy dispatched_policy,
        int64_t invocation = 0) const;
    // Valid only after the active REPLAY consumed the Graph branch node whose
    // key is this exact manifest fingerprint. A replayable Graph could only
    // have been built after the original forward consumed every route.
    void inherit_matching_graph_replay_receipt(
        uint64_t replayed_manifest_fingerprint) const;
    // Only the composite preload authority may carry this subset between
    // occurrences. Reading/merging receipts never changes the route schema.
    std::vector<uint8_t> consumed_route_receipt() const;
    void inherit_consumed_route_subset(
        uint64_t manifest_fingerprint,
        const std::vector<uint8_t>& receipt) const;
    void require_all_consumed() const;

private:
    FmbPhysicalExecutionManifest manifest_;
    std::shared_ptr<const FmbPreparedStageCandidate> prepared_;
    uint64_t fingerprint_ = 0;
    mutable std::vector<uint8_t> consumed_;
};

// One native-feasible point in the bounded A6 stage domain. Chunk values are
// capacities (so a one-chunk tail may be shorter); the complete physical
// schedules remain owned by FmbThreeStageChunkPlan.
struct FmbPrefillStageCandidate {
    int64_t input_chunk_size = 0;
    int64_t qkv_chunk_size = 0;
    int64_t compute_chunk_size = 0;
    FmbThreeStageChunkPlan stage_plan;
    FmbPhysicalExecutionManifest physical_manifest;
};

namespace detail {
bool fmb_prefill_stage_candidate_less(
    const FmbPrefillStageCandidate& lhs,
    const FmbPrefillStageCandidate& rhs);
bool fmb_prefill_stage_candidate_same_identity(
    const FmbPrefillStageCandidate& lhs,
    const FmbPrefillStageCandidate& rhs);
}  // namespace detail

// Versioned int-only descriptor used at the Python/C++ planning boundary.
// Capacities stay separate from the resolved schedules because a legal
// one-tail plan can have execution_len=225, compute capacity=240, and a single
// physical compute chunk of length 225.  The descriptor carries every chunk,
// span, boundary policy, mode, and the native stage-plan fingerprint. V2/V3
// append a separately fingerprinted physical manifest (V3 adds invocation);
// V1 remains decode-only compatibility and yields an UNSPECIFIED manifest.
std::vector<int64_t> encode_fmb_prefill_stage_candidate(
    const FmbPrefillStageCandidate& candidate);
FmbPrefillStageCandidate decode_fmb_prefill_stage_candidate(
    at::IntArrayRef descriptor);

// Only the descriptor decoder can publish these immutable objects. A cache
// hit compares every wire word; fingerprints are Graph receipts, not cache
// identity. The cache belongs to one native owner and retains at most 32
// descriptors (each already bounded by the wire decoder).
class FmbPreparedStageCandidate {
public:
    const FmbPrefillStageCandidate& candidate() const { return candidate_; }
    uint64_t manifest_fingerprint() const { return manifest_fingerprint_; }
    uint64_t stage_fingerprint() const { return stage_fingerprint_; }
    FmbChunkTopology topology() const { return topology_; }
    FmbRopeTableResidency rope_residency() const { return rope_residency_; }
    std::optional<AttentionExecutionPolicy> attention_policy() const {
        return attention_policy_;
    }
    void validate_request(int64_t physical_length, int64_t position) const;

private:
    friend class FmbPreparedStageCandidateCache;
    explicit FmbPreparedStageCandidate(FmbPrefillStageCandidate candidate);
    FmbPrefillStageCandidate candidate_;
    uint64_t manifest_fingerprint_ = 0;
    uint64_t stage_fingerprint_ = 0;
    FmbChunkTopology topology_ = FmbChunkTopology::SISC;
    FmbRopeTableResidency rope_residency_ = FmbRopeTableResidency::UNSPECIFIED;
    std::optional<AttentionExecutionPolicy> attention_policy_;
    int64_t physical_length_ = 0;
    int64_t position_ = 0;
};

class FmbPreparedStageCandidateCache {
public:
    static constexpr size_t kCapacity = 32;
    std::shared_ptr<const FmbPreparedStageCandidate> prepare(
        at::IntArrayRef descriptor);
    void clear() { entries_.clear(); }
    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        std::vector<int64_t> descriptor;
        std::shared_ptr<const FmbPreparedStageCandidate> prepared;
    };
    std::list<Entry> entries_;
};
// A retained single-token Graph intentionally excludes the absolute decode
// position from its identity.  Rebase only the decoded working copy's
// absolute KV schedule; the immutable wire descriptor and both fingerprints
// remain unchanged.
void rebase_fmb_retained_decode_candidate(
    FmbPrefillStageCandidate& candidate, int64_t position);
std::vector<int64_t> encode_fmb_prefill_stage_domain(
    const std::vector<FmbPrefillStageCandidate>& candidates);

// Schema-v7 is fail-closed: existing production models do not acquire a
// resolved-profile claim merely by inheriting FusedModelBase.  A model must be
// audited and explicitly opt into the canonical framework traversal.  Hidden
// layer/post_fn loops remain unsealed and must STOP until represented by a
// future typed extension.
enum class SpmFmbTraversalCapability : uint8_t {
    Unsealed = 0,
    CanonicalTraversal = 1,
};

// Persistent weight preloads run outside the resolved temporary traversal.
// This capability admits only Persistent/PersistentPerLayer BufferDecl
// callbacks; temporary preloads and ModelStaticConfig::preload_fn remain
// opaque and fail closed.
enum class SpmFmbPreloadCapability : uint8_t {
    Unsealed = 0,
    PersistentOutsideResolvedWindow = 1,
};

// Composite-only opt-in for one owner invoked up to three times in the same
// outer Graph.  Each occurrence receives a distinct stable mutable-input
// address cell; occurrence zero emits the persistent preload callback prefix,
// while same-allocation followers consume no preload nodes.  This capability
// is not part of a legacy resolved-profile hash.
enum class SpmFmbCompositeOccurrenceCapability : uint8_t {
    Unsealed = 0,
    StableThreeInputSlotsSameOwnerPreload = 1,
};

// A post_fn remains opaque unless the model opts into one canonical dense
// producer-yield contract.  The capability only permits profile/BUILD
// observation; a terminal writer marker and a later consumer binding are still
// required before any physical plan can use the yield.
enum class SpmFmbPostFnYieldCapability : uint8_t {
    Unsealed = 0,
    CanonicalDenseReplicatedFp16 = 1,
};

// Independent opt-in for sparse terminal-writer windows inside canonical
// LayerBody callbacks.  Keeping this separate preserves the original PostFn
// capability's exact-coverage contract and frozen profile/seal identities.
// It is observation authority only; physical targets remain unavailable.
enum class SpmFmbLayerProducerYieldCapability : uint8_t {
    Unsealed = 0,
    CanonicalDenseReplicatedFp16 = 1,
};

enum class SpmFmbTerminalWriterKind : uint8_t {
    AllReduceSumResidual = 1,
};

// Schema-v9 remains fail-closed independently of the schema-v7 traversal
// capability.  This declaration is a model-owned semantic claim: the exact
// consecutive callbacks for one layer are one logical chunk run, the first
// callback is the only active carrier, and all remaining callbacks are true
// no-ops.  The run length comes from the exact resolved callback sequence.
// BUILD observation verifies the claim; it never infers it from early returns.
enum class SpmFmbActiveGroupCapability : uint8_t {
    Unsealed = 0,
    CanonicalActiveFirstChunkRun = 1,
};

// Schema-v10 is a separate, CPU-only semantic proof.  It lets FMB mint member
// identities from the canonical active-first run and derive dense DDR ingress /
// egress slices without exposing the global offsets as caller-authored
// ChunkInfo.  It is not register, Graph-DMA, lifetime, or physical proof.
enum class SpmFmbDenseDdrMemberCapability : uint8_t {
    Unsealed = 0,
    IndependentLocalSequentialDdrPingPong = 1,
};

// Schema-v11 separately opts into FMB-minted semantic IDs on canonical Graph
// DMA nodes.  It proves member/role occurrence and replay endpoint stability,
// not the semantic ownership of the fixed SPM address or physical admission.
enum class SpmFmbDmaEndpointCapability : uint8_t {
    Unsealed = 0,
    DenseDdrMemberRoleBindings = 1,
};

// Schema-v12 is an independent, stronger runtime claim.  The subclass may no
// longer mint or retain raw semantic endpoints: it can only ask FMB for an
// opaque member reference during the current canonical callback, then emit the
// fixed ingress/egress roles through the framework wrappers below.  This is
// still a CPU-contract proof; it grants no live or physical admission.
enum class SpmFmbRuntimeMemberCapability : uint8_t {
    Unsealed = 0,
    CurrentCallbackDenseDdrRefs = 1,
};

// Schema-v13 independently seals the fixed SPM side of each schema-v12
// canonical DMA.  FMB derives shared-stage versus packed-global ownership from
// the prepared allocation snapshot; subclasses cannot author a peer kind,
// slice offset, core mask, or raw SPM address.  This remains CPU-contract only.
enum class SpmFmbSpmPeerCapability : uint8_t {
    Unsealed = 0,
    TypedDensePackedOrShared = 1,
};

enum class SpmFmbDmaEndpointRole : uint8_t {
    Ingress = 1,
    Egress = 2,
};

class SpmFmbRuntimeDenseDdrMemberRef {
public:
    SpmFmbRuntimeDenseDdrMemberRef(
        const SpmFmbRuntimeDenseDdrMemberRef&) = default;
    SpmFmbRuntimeDenseDdrMemberRef(
        SpmFmbRuntimeDenseDdrMemberRef&&) noexcept = default;
    SpmFmbRuntimeDenseDdrMemberRef& operator=(
        const SpmFmbRuntimeDenseDdrMemberRef&) = default;
    SpmFmbRuntimeDenseDdrMemberRef& operator=(
        SpmFmbRuntimeDenseDdrMemberRef&&) noexcept = default;
    ~SpmFmbRuntimeDenseDdrMemberRef() = default;

private:
    struct State;
    explicit SpmFmbRuntimeDenseDdrMemberRef(
        std::shared_ptr<const State> state) : state_(std::move(state)) {}

    std::shared_ptr<const State> state_;

    friend class FusedModelBase;
};

class SpmFmbIndependentLocalContext {
public:
    size_t member_ordinal() const { return member_ordinal_; }
    int64_t local_row_count() const { return local_row_count_; }
    int64_t local_position_begin() const {
        return local_position_begin_;
    }
    int64_t local_kv_seq_len() const { return local_kv_seq_len_; }

private:
    SpmFmbIndependentLocalContext(
        size_t member_ordinal,
        int64_t local_row_count,
        int64_t local_position_begin,
        int64_t local_kv_seq_len)
        : member_ordinal_(member_ordinal),
          local_row_count_(local_row_count),
          local_position_begin_(local_position_begin),
          local_kv_seq_len_(local_kv_seq_len) {}

    size_t member_ordinal_ = 0;
    int64_t local_row_count_ = 0;
    int64_t local_position_begin_ = 0;
    int64_t local_kv_seq_len_ = 0;

    friend class FusedModelBase;
};

// CPU-only input describing the already-resolved chunk policy.  It replaces
// hand-authored occurrence lists, not the framework's chunk planner.  Future
// production wiring must feed this from the exact run_all_layers resolution;
// this stage deliberately exposes only the protected CPU contract seam.
struct SpmFmbResolvedProfileRequest {
    uint64_t version = 0;
    std::vector<ChunkInfo> chunks;
    std::vector<ChunkInfo> kv_insert_chunks;
};

// Pitfall 3 structural fix (D-203): typed SPM offset for SDPA / KV-insert only.
// explicit operator uint32_t() forces conversion at each call site; mixing
// offset vs absolute address becomes a compile-time error, not a runtime bug.
struct SpmOffset {
    uint32_t value = 0;
    explicit operator uint32_t() const { return value; }
};

// Canary-stage description of one component's already-planned, zero-based
// temporary layout.  It deliberately carries no offsets: the physical pipeline
// may only adopt the exact cached FusedModelBase layout after validating every
// buffer against its checked scratch view.
struct SpmPipelineComponentLayout {
    size_t temporary_bytes = 0;
    uint64_t layout_hash = 0;
};

struct SpmPipelineCausalPrefillDryLayout {
    SpmPipelineCausalPrefillShape shape;
    SpmPipelineComponentLayout component;
    std::vector<int64_t> stage_descriptor;
};

// D-501: ModelStaticConfig carries optional pointer-to-member callbacks.
// Framework discovers features by inspecting which fields are non-null —
// no extra virtuals on the subclass header.
//
// Graph admission is owned by the Python adapter. C++ run_all_layers emits
// operations and does not select cache entries or construct signatures.
struct ModelStaticConfig {
    int64_t                 num_layers           = 0;
    int64_t                 cross_layer_batch_size = 0;

    // D-501 opt-ins — null = disabled.
    // Dispatched via std::invoke(cfg.xxx_fn, *this) inside run_all_layers.
    void      (FusedModelBase::*preload_fn)()                              = nullptr;  // SigLIP
    void      (FusedModelBase::*kv_first_fn)(int, const ChunkInfo&)        = nullptr;  // Gemma Phase 1
    ChunkPlan (FusedModelBase::*kv_first_chunk_plan_fn)(const ChunkPlan&)  = nullptr;  // Gemma dual-chunk
    void      (FusedModelBase::*post_fn)()                                 = nullptr;  // Qwen3 fused lm_head
    // EXT-Pi05: pre_layers_fn / post_layers_fn — called by run_all_layers
    // BEFORE the first layer / AFTER the last layer, inside the same
    // caller-owned capture when active. Used by Pi05DenoiseStepModel to emit action_in_proj
    // (pre) and final PiGemmaRMSNorm + action_out_proj (post). Subclass
    // bodies MUST only call rpu_launch_* / addr() / etc.; they MUST NOT
    // open a raw batch or nested GraphCache scope.
    void (FusedModelBase::*pre_layers_fn)()                                = nullptr;
    void (FusedModelBase::*post_layers_fn)()                               = nullptr;
    // After allocation, before persistent preloads: false keeps their ordinary
    // emission and the body; true proves and advances this same Graph to its end.
    // Only admitted without D-501 preload/pre/post hooks; captured DMA still runs.
    bool (FusedModelBase::*checked_layer_body_replay_fn)()                 = nullptr;
    std::vector<int64_t> post_output_shape;
    // EXT-unroll: repeat the pre_layers_fn → layer-loop → post_layers_fn body
    // this many times in ONE graph (default 1 = unchanged for all other models).
    // Only WallOssActionStepModel sets > 1 (in-graph denoise unroll).
    int64_t body_iterations = 1;

    // Fast replay opt-in (RhinoVLA): on GraphCache REPLAYING, run the framework
    // prelude (shape checks, chunk/allocation/preload, input live_base refresh),
    // then skip the per-layer op-emission loop and let graph.end() replay the
    // built stream. Default false keeps existing model paths unchanged.
    bool fast_replay_skip_layer_loop = false;

    // Narrow opt-in for in-graph denoise loops: the captured stream includes
    // pre_layers_fn/body_iterations/post_layers_fn, and replay-time drift is
    // refreshed before run_all_layers via live_base variables. Default false
    // preserves the older fast-replay guardrails for all other models.
    bool fast_replay_skip_full_body = false;

    // Skip the preload_fn host re-walk on GraphCache REPLAYING (RhinoVLA vision
    // opt-in via RPU_RHINOVLA_VISION_PRELOAD_REPLAY_SKIP). Only honored when
    // weights are clean; default false leaves all other models unchanged.
    bool fast_replay_skip_preload = false;

    // Persistent read-only buffers may omit their preload callbacks when a NEW
    // graph is recorded after the same persistent generation was already
    // populated. The first graph after allocation/invalidation still records
    // and executes every callback. Default false preserves existing models.
    bool clean_recording_skip_preload_callbacks = false;

    // Bake the post_fn into the BUILT graph: skip its host re-emission on REPLAY
    // (RhinoVLA vision merger opt-in via RPU_RHINOVLA_VISION_BAKE_MERGER).
    // Requires fast_replay_skip_layer_loop. Default false.
    bool fast_replay_bake_post_fn = false;

    // Per-forward capability stamp for the only currently supported batched
    // decoder path (plain Qwen3).  False is the fail-closed default for every
    // other FusedModelBase subclass.  Batched cache-slot mutation must also
    // keep the host op stream live on REPLAY, so the driver uses this bit to
    // disable both fast-replay skip variants.
    bool batch_decode_active = false;

    // Owner-certified, default-off two-chunk KV_FIRST carry. Logical chunk
    // vectors stay in canonical order; physical order is KVIN1, KVIN0,
    // COMP0, COMP1. The owner must keep Q and residual alive across these
    // callbacks and bind this policy to its physical manifest/cost identity.
    int64_t kv_first_pair_carry_rows = 0;
    bool kv_first_pair_carry_across_layers = false;
};

struct ModelDynamicConfig {
    ChunkMode     chunk_mode     = ChunkMode::SEQUENTIAL;
    InterLayerIO  inter_layer_io = InterLayerIO::AUTO;
    // Optional traversal for independent chunks (for example, equal-size images):
    // group -> chunk -> layer. This keeps each chunk SPM-resident for one
    // cross-layer group, then spills only at the group boundary. The default
    // group -> layer -> chunk order is unchanged for every existing model.
    bool chunk_outer_within_group = false;
    // Models normally leave AUTO. A specialization must separately implement
    // subclass_spm_kv_by_mha_eligible(), SPM buffer declarations, and the
    // matching kernel dispatch. Unsupported/oversized AUTO resolves DDR_KV.
    AttentionExecutionPolicy attention_policy =
        AttentionExecutionPolicy::AUTO;
};

// Inference context (per-forward request state)
struct InferenceContext {
    const at::Tensor*         hidden_states = nullptr;
    std::vector<at::Tensor>*  k_caches      = nullptr;
    std::vector<at::Tensor>*  v_caches      = nullptr;
    std::optional<at::Tensor> attention_mask;
    int64_t position       = 0;
    int64_t seq_len        = 0;
    // Batch decode: independent sequences packed into the GEMM M dim. Only
    // > 1 for seq_len == 1 (see run_all_layers). Each sequence owns KV-cache
    // slot b, so KV-insert / SDPA are emitted once per b while every
    // row-parallel op runs a single launch over batch_size * chunk.len rows.
    int64_t batch_size     = 1;
    bool    is_causal      = true;
    bool    input_in_spm   = false;  // set by run_all_layers layer-group loop
    bool    output_to_spm  = false;
    int64_t body_iter      = 0;  // current body iteration (EXT-unroll); 0 unless body_iterations>1
    // Zero-based physical chunk invocation of the callback currently being
    // emitted. Shared sites whose resolved route can differ on the tail use
    // this V3 discriminator; repeated layers/body iterations reuse it.
    int64_t physical_route_invocation = 0;
    // Resolved for every run_all_layers call, including legacy six-argument
    // callers. Subclasses may inspect it to select a certified specialization;
    // the framework remains the sole owner of traversal and allocation.
    FmbThreeStageChunkPlan stage_plan;
    FmbChunkTopology chunk_topology = FmbChunkTopology::SISC;
    AttentionExecutionPolicy attention_policy =
        AttentionExecutionPolicy::DDR_KV;

    bool has_complete_physical_manifest() const {
        return physical_manifest_consumer_.complete();
    }
    const FmbPhysicalExecutionManifest& physical_manifest() const {
        return physical_manifest_consumer_.manifest();
    }
    uint64_t physical_manifest_fingerprint() const {
        return physical_manifest_consumer_.fingerprint();
    }
    const FmbRouteManifestEntry& find_physical_route(
        FmbRouteFamily family, int64_t site_id,
        int64_t invocation = 0) const {
        return physical_manifest_consumer_.find_route(
            family, site_id, invocation);
    }
    const FmbRouteManifestEntry& consume_physical_route(
        FmbRouteFamily family,
        int64_t site_id,
        int64_t resolved_selector,
        int64_t resolved_flags,
        at::IntArrayRef resolved_arguments = {},
        int64_t invocation = 0) const {
        return physical_manifest_consumer_.consume_route(
            family, site_id, resolved_selector, resolved_flags,
            resolved_arguments, invocation);
    }
    AttentionExecutionPolicy physical_attention_policy_for_site(
        int64_t site_id, int64_t invocation = 0) const {
        return physical_manifest_consumer_.attention_policy_for_site(
            site_id, invocation);
    }
    const FmbRouteManifestEntry& consume_physical_attention_route(
        int64_t site_id,
        AttentionExecutionPolicy dispatched_policy,
        int64_t invocation = 0) const {
        return physical_manifest_consumer_.consume_attention_route(
            site_id, dispatched_policy, invocation);
    }

private:
    friend class FusedModelBase;
    std::vector<uint8_t> consumed_physical_route_receipt() const {
        return physical_manifest_consumer_.consumed_route_receipt();
    }
    void inherit_physical_route_subset(
        uint64_t manifest_fingerprint,
        const std::vector<uint8_t>& receipt) const {
        physical_manifest_consumer_.inherit_consumed_route_subset(
            manifest_fingerprint, receipt);
    }
    void begin_external_physical_manifest_prologue(
        const FmbPhysicalExecutionManifest& manifest,
        bool matching_replay_branch) {
        TORCH_CHECK(
            !external_physical_manifest_prologue_pending_,
            "RPU_PLANNER_REJECT:CAPABILITY: an external physical-manifest "
            "prologue is already pending transfer");
        physical_manifest_consumer_.reset(manifest);
        TORCH_CHECK(
            physical_manifest_consumer_.complete(),
            "RPU_PLANNER_REJECT:CAPABILITY: an external prologue requires a "
            "COMPLETE physical manifest");
        external_physical_manifest_prologue_pending_ = true;
        external_physical_manifest_prologue_replay_ = matching_replay_branch;
    }
    bool bind_physical_manifest(
        const FmbPhysicalExecutionManifest& manifest,
        std::shared_ptr<const FmbPreparedStageCandidate> prepared = {}) {
        if (external_physical_manifest_prologue_pending_) {
            physical_manifest_consumer_.retain_consumed_if_matching(manifest);
            external_physical_manifest_prologue_pending_ = false;
            return true;
        }
        if (prepared) physical_manifest_consumer_.reset_prepared(std::move(prepared));
        else physical_manifest_consumer_.reset(manifest);
        external_physical_manifest_prologue_replay_ = false;
        return false;
    }
    bool external_physical_manifest_prologue_replay() const {
        return external_physical_manifest_prologue_replay_;
    }
    void cancel_external_physical_manifest_prologue() {
        external_physical_manifest_prologue_pending_ = false;
        external_physical_manifest_prologue_replay_ = false;
        physical_manifest_consumer_.reset({});
    }
    void require_all_physical_routes_consumed() const {
        physical_manifest_consumer_.require_all_consumed();
    }
    std::optional<AttentionExecutionPolicy>
    physical_attention_layout_policy() const {
        return physical_manifest_consumer_.attention_layout_policy();
    }
    void inherit_physical_routes_from_matching_graph_replay() const {
        physical_manifest_consumer_.inherit_matching_graph_replay_receipt(
            physical_manifest_consumer_.fingerprint());
    }
    FmbPhysicalManifestConsumer physical_manifest_consumer_;
    bool external_physical_manifest_prologue_pending_ = false;
    bool external_physical_manifest_prologue_replay_ = false;
};

// FusedModelBase — flat framework (v3). Subclass contract:
//   4 mandatory virtuals (declare_buffers, static_config, dynamic_config, build_layer_subgraph)
//   Optional D-501 features via ModelStaticConfig callback fields
//   Call invalidate_model_state() after set_weights(...) (D-503)
class FusedModelBase {
public:
    // PImpl forward decl — the Impl BODY is defined only in the private header
    // fused_model_base_impl.h which subclass .cpp files never include.
    // Forward-decl is public so file-static helpers in fused_model_base.cpp
    // can take `FusedModelBase::Impl&` parameters. The actual fields stay
    // fully hidden (the body isn't in this header).
    class Impl;

    FusedModelBase();
    virtual ~FusedModelBase();

    FusedModelBase(const FusedModelBase&)            = delete;
    FusedModelBase& operator=(const FusedModelBase&) = delete;
    FusedModelBase(FusedModelBase&&)                 = delete;
    FusedModelBase& operator=(FusedModelBase&&)      = delete;

    at::Tensor run_all_layers(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal,
        int64_t planned_chunk_size = 0,
        at::IntArrayRef planned_stage_descriptor = {},
        uint64_t expected_layout_hash = 0);

    // Specialized entry point: FMB resolves and consumes qkv/compute chunks
    // while the model supplies its independently executed input-stage
    // partition and semantic boundaries. Kernel selection remains
    // model-specialized. The six-argument overload creates one default input
    // chunk and one full-sequence span, so every model uses the same contract.
    at::Tensor run_all_layers(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal,
        const std::vector<ChunkInfo>& input_chunks,
        const std::vector<FmbExecutionSpan>& spans,
        uint64_t expected_layout_hash = 0);
    at::Tensor run_all_layers(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal,
        const std::vector<ChunkInfo>& input_chunks,
        const std::vector<FmbExecutionSpan>& spans,
        FmbStageBoundaryPolicies boundary_policies,
        uint64_t expected_layout_hash = 0);

    void    set_chunk_size_override(int64_t cs);
    int64_t get_chunk_size_override() const;
    // TASK-1.5 (codex round-3 path-(b) BC2-01 closure): per-handle resolved chunk_size
    // from the most-recent forward; 0 if no forward has run yet on this instance.
    int64_t get_last_resolved_chunk_size() const;

    // Cold composite barrier: reserve only the installed owner's fixed
    // Persistent declarations before any peer's domain is searched. No Temp,
    // preload/DMA, Graph, plan selection, or forward receipt is produced.
    void prepare_persistent_spm(int64_t execution_len, int64_t position,
                                bool is_causal, int64_t mask_kv_len = 0);

    // Admitted chunk envelope: native SPM arithmetic and profile admission are
    // separate guards. A candidate must satisfy both before hardware submission.
    //
    // chunk bounds the physical chunk on every path. max_kv_len additionally
    // bounds the explicit-mask allocation, whose size scales as
    // comp_cs * ceil16(kv) * 32. Without an explicit mask, it bounds the admitted
    // AUTO request range rather than a mask allocation.
    struct ChunkEnvelope {
        int64_t max_kv_len = 0;   // 0 = UNDECLARED => deny
        int64_t chunk      = 0;   // 0 = "auto is certified within max_kv_len"
        bool declared() const { return max_kv_len > 0; }
    };

    // Cold setter — must precede the first forward, mirroring the existing
    // chunk_size_cap contract so a live GraphCache entry cannot be hot-switched.
    void set_chunk_envelope(int64_t max_kv_len, int64_t chunk);
    const ChunkEnvelope& chunk_envelope() const;

    // The trusted Python binder supplies a verified external catalog once,
    // before dispatch. Hot chunk changes retain this immutable owner value.
    void bind_kvinsert_costs(at::IntArrayRef identity,
                            const std::string& catalog_sha256,
                            at::IntArrayRef certificate_rows);
    const std::string& kvinsert_cost_catalog_sha256() const;
    // Exact lookup in this owner's latest successful native stage-domain query.
    // The supplied descriptor is only a key, never a source of geometry/caps.
    KvInsertCostDomainQuery kvinsert_cost_domain(
        const char* owner_kind, at::IntArrayRef actual_stage_descriptor) const;
    // Diagnostic only: mint an existing EXACT route after native feasibility
    // and SPM revalidation. No measured-cost authority is created.
    std::tuple<std::vector<int64_t>, int64_t, int64_t>
    mint_kvinsert_exact_candidate(
        at::IntArrayRef actual_stage_descriptor, int64_t site_id,
        int64_t invocation, int64_t route);
    // Read-only installed geometry/precision/cold policy, shared by leaf cost
    // queries and the existing native-composite execution identity.
    std::vector<int64_t> installed_profile_identity() const;
    uint64_t installed_model_state_generation() const;
    // Cache identity for native candidate preparation, combined with the caller's
    // native owner prefix and non-reused handle. It excludes runtime addresses,
    // temporary oracle state, and other owners' resource generations.
    std::vector<int64_t> planner_cache_identity() const;

    // Public control-plane writes may remain legacy-hot unless an adapter
    // opts its handle into the common transaction. Internal per-forward
    // routing continues to use set_chunk_size_override() directly.
    void enable_execution_reconfigure_guard();
    void set_control_chunk_size_override(int64_t cs, const char* operation);
    void stage_control_chunk_size_override(uint64_t token, int64_t cs,
                                           const char* operation);
    void check_execution_reconfigure_destroy_allowed(
        const char* operation) const;

    // Shared by the generic planner, runtime and custom component planners.
    // Bind the immutable physical selections before declaring/hashing buffers.
    LayoutContext bind_physical_layout_context(
        LayoutContext layout,
        const FmbPhysicalExecutionManifest& manifest) const;

    // Shared with native prologues which inspect a descriptor before entering
    // run_all_layers. Returned storage survives cache eviction and re-keying.
    std::shared_ptr<const FmbPreparedStageCandidate> prepare_stage_candidate(
        at::IntArrayRef descriptor) const;
protected:
    // Valid only during this dispatch, after actual temporary layout matched
    // the COMPLETE plan. Persistent addresses require separate owner checks.
    uint64_t checked_layer_body_replay_layout_hash() const;

    // Cold admission uses the same complete fixed/temporary/held-SPM budget
    // as actual allocation. Runtime must validate its selected descriptor,
    // never use this query to select a replacement schedule.
    bool declared_spm_layout_fits(const std::vector<BufferDecl>& declarations) const;
    virtual FmbForwardOperandResidency physical_forward_operand_residency(
        const FmbPhysicalExecutionManifest& /*manifest*/) const {
        return FmbForwardOperandResidency::UNSPECIFIED;
    }

    // One typed, owner-local snapshot of the latest oracle's temporary layout
    // fields. Arguments MUST be members of this owner, never stack locals.
    // The closure cannot outlive the owner and restores even a failed mint.
    using KvCostLayoutScope = std::function<void(const std::function<void()>&)>;
    template <typename... Fields>
    static KvCostLayoutScope capture_kvinsert_cost_layout_fields(Fields&... fields) {
        return [members = std::tie(fields...), values = std::make_tuple(fields...)](
                   const std::function<void()>& operation) mutable {
            auto saved = std::apply(
                [](auto&... value) { return std::make_tuple(value...); }, members);
            auto restore = c10::make_scope_exit([&] { members = std::move(saved); });
            members = values;
            operation();
        };
    }
    virtual KvCostLayoutScope capture_kvinsert_cost_layout_scope() {
        return [](const std::function<void()>& operation) { operation(); };
    }
    // Empty means unverified precision: geometry may be inspected, but no cost
    // artifact may activate. Overrides must inspect installed native weights.
    virtual std::vector<int64_t> kvinsert_cost_weight_identity() const {
        return {};
    }
    static void append_kvinsert_cost_tensor_identity(
        std::vector<int64_t>& identity, const at::Tensor& tensor);
    static void append_kvinsert_cost_scalar_identity(
        std::vector<int64_t>& identity, double value);
    // Reused by the generic oracle and native geometry-specific producers.
    // The callback must return only its actual post-SPM-admission candidates.
    void begin_kvinsert_cost_domain_oracle();
    void remember_kvinsert_cost_planning_context(const InferenceContext& context);
    std::vector<FmbPrefillStageCandidate> observe_kvinsert_cost_candidates(
        const std::function<std::vector<FmbPrefillStageCandidate>()>& producer,
        const LayoutContext* actual_layout = nullptr);
    void finish_kvinsert_cost_domain_oracle(
        const std::vector<FmbPrefillStageCandidate>& returned_candidates);
    void bind_kvinsert_cost_owner_prefix(
        at::IntArrayRef descriptor, at::IntArrayRef native_prefix);
    std::vector<int64_t> kvinsert_cost_owner_prefix(at::IntArrayRef descriptor) const;
    KvInsertSegmentPlan resolve_kvinsert_plan_auto(
        int64_t site_id, FmbGraphLifecycle graph_lifecycle,
        int64_t position, int64_t logical_rows,
        int64_t physical_rows, int num_cores, int64_t num_kv_heads,
        int64_t head_dim, uint32_t capabilities) const;
    KvInsertSegmentPlan restore_kvinsert_plan(
        int64_t site_id, at::IntArrayRef arguments, int num_cores,
        int64_t num_kv_heads, int64_t head_dim) const;
    // Planning/cold-policy validators inspect detached candidates before the
    // selected manifest is installed in ctx(). Bind decode authority to the
    // COMPLETE candidate's exact KV_INSERT site/invocation and lifecycle;
    // runtime emitters continue to use restore_kvinsert_plan() above.
    KvInsertSegmentPlan restore_kvinsert_plan_from_manifest(
        const FmbPhysicalExecutionManifest& manifest,
        int64_t site_id, int64_t invocation, int num_cores,
        int64_t num_kv_heads, int64_t head_dim) const;

    void stage_execution_controls(
        uint64_t token,
        std::optional<int64_t> chunk_size_override,
        std::optional<ChunkEnvelope> chunk_envelope,
        std::function<void()> apply_extra,
        std::function<void()> rollback_extra,
        const char* operation);

    // Hard half of the envelope, for the ONCE-per-resolve hook
    // (subclass_chunk_size_cap). Throws with an actionable message; returns the
    // chunk ceiling to feed the auto search. Participating subclasses call this
    // as the first statement of their cap hook.
    int64_t enforce_chunk_envelope(int64_t seq_len, int64_t position,
                                   const char* model_name) const;

    // Cheap per-candidate half, for subclass_chunk_size_valid. This is the half
    // that closes the explicit-override route (compute_chunks_impl:499 calls
    // valid_fn there but NEVER reads the cap), so a participating subclass MUST
    // call it in its validity hook, not only in its cap hook.
    bool chunk_within_envelope(int64_t cs) const {
        const ChunkEnvelope& e = chunk_envelope();
        // An `auto` row certifies only the planner result for the measured
        // length.  It is not evidence for an arbitrary explicit override.
        // Profiles that expose public exact chunk control therefore declare a
        // positive, measured-safe ceiling; keep future zero-ceiling rows
        // closed on the override path instead of silently widening them.
        if (e.chunk <= 0) return get_chunk_size_override() <= 0;
        return cs <= e.chunk;
    }

    // Mandatory subclass virtuals (D-201 + D-202)
    virtual std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) = 0;
    virtual ModelStaticConfig       static_config() = 0;
    virtual ModelDynamicConfig      dynamic_config(const ChunkPlan& plan) = 0;
    // Read-only sibling used by the finite-domain oracle. Subclasses whose
    // forward dynamic_config materializes masks or other runtime state must
    // override this with the same configuration minus those side effects.
    virtual ModelDynamicConfig planning_dynamic_config(
        const ChunkPlan& plan) {
        return dynamic_config(plan);
    }
    virtual void                    build_layer_subgraph(int layer_idx,
                                                         const ChunkInfo& chunk) = 0;

    // Deny-by-default capability hook for raw-SPM K/V + by-MHA. Returning true
    // promises that declare_buffers(ctx) understands ctx.attention_policy and
    // that every SPM-selected attention site has a shape/mask-certified kernel.
    // The common resolver still checks the exact joint SPM layout and falls
    // AUTO back to DDR when it does not fit.
    virtual bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& /*plan*/,
        const LayoutContext& /*ctx*/,
        int64_t /*position*/) const {
        return false;
    }

    // COMPLETE physical descriptors are inert unless a concrete native owner
    // opts in and names the current Graph lifecycle. This does not adopt any
    // route by itself: each exact launcher site must consume its own entry.
    virtual FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& /*manifest*/) const {
        return {};
    }

    // Optional owner authority added to each native A6 candidate.  The
    // default preserves UNSPECIFIED for every owner that has not adopted the
    // descriptor/receipt contract.
    virtual FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& /*plan*/,
        const LayoutContext& /*layout*/,
        int64_t /*physical_len*/, int64_t /*logical_len*/,
        int64_t /*position*/) const {
        return {};
    }

    // A physical choice can change the exact SPM footprint without changing
    // any of the three chunk capacities.  The common resolver therefore asks
    // for the complete same-chunk domain and runs declare_buffers() for each
    // manifest.  Existing owners keep their single-candidate behaviour.
    virtual std::vector<FmbPhysicalExecutionManifest>
    physical_manifest_domain_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const {
        return {physical_manifest_for_candidate(
            plan, layout, physical_len, logical_len, position)};
    }

    // Optional kernel-validity hook for auto chunk_size search (compute_chunks_impl).
    // Default: every 16-multiple is valid. Override when subclass kernels (e.g.
    // SDPA tile-shape / VLM register / mask-tiling constraints in
    // sdpa_is_valid_chunk_size, rpu_helpers.h) impose extra constraints that
    // the SPM-budget predicate alone does NOT capture. The validity predicate
    // is non-monotone in cs, which is why compute_chunks_impl uses a linear
    // scan rather than a binary search.
    //
    // Called on BOTH planner paths. The auto scan calls it per candidate
    // (compute_chunks_impl:536); the explicit-override path calls it exactly
    // once, on the sequence-clamped override (:499). It was documented here as
    // "auto-pick only ... the override path bypasses this hook by design" until
    // MR-A — that was stale, and the correction matters: this hook is the only
    // consumer choke point BOTH routes pass through, which is why the certified
    // envelope's per-candidate check has to live here and not just in the cap.
    virtual bool subclass_chunk_size_valid(int64_t /*cs*/,
                                           int64_t /*seq_len*/,
                                           int64_t /*position*/) const {
        return true;
    }

    // Optional upper bound for the auto-pick path. Unlike
    // set_chunk_size_override(), this keeps both the SPM-budget probe and the
    // subclass validity scan active; 0 means no cap. Use this when an adapter
    // needs to stay inside a proven kernel envelope without requiring every
    // sequence length to be divisible by one exact chunk size.
    virtual int64_t subclass_chunk_size_cap(int64_t /*seq_len*/,
                                            int64_t /*position*/) const {
        return 0;
    }

    // Subclass contribution to the SPM-layout allocation identity.
    //
    // compute_params_hash_impl keys the allocation on a FIXED set of fields —
    // the five set_model_params values plus LayoutContext's chunk_size,
    // kv_insert_chunk_size, num_layers, use_attn_mask, is_causal, batch_size,
    // stage_plan_fingerprint, physical_manifest_fingerprint,
    // attention_policy, descriptor-derived RoPE table residency, and
    // (only when use_attn_mask) max_kv_seq_len. ensure_allocated reuses the existing layout
    // whenever that hash is unchanged. So a subclass whose declare_buffers sizes
    // any BufferDecl from state OUTSIDE that set — image grid, patch count,
    // window length, action horizon, deepstack depth, expert count, ... — can
    // change its layout with the hash unchanged, and the framework will happily
    // run the new shape against SPM sized for the old one. There is no error and
    // no other layout lever: invalidate_model_state() advances model state but
    // deliberately does not perturb the allocation hash, so this virtual is the
    // ONLY way to force a re-layout for subclass-owned sizing state.
    //
    // Contract: return a well-mixed value over exactly the extra state that
    // declare_buffers reads. It is XOR-folded into the params hash, so mix
    // multiplicatively yourself (a bare small integer collides badly). Default 0
    // means "declare_buffers depends only on already-hashed state" and leaves the
    // hash byte-identical for every subclass that does not override.
    virtual int64_t subclass_layout_hash() const { return 0; }

    virtual SpmFmbTraversalCapability spm_fmb_traversal_capability() const {
        return SpmFmbTraversalCapability::Unsealed;
    }
    virtual SpmFmbPreloadCapability spm_fmb_preload_capability() const {
        return SpmFmbPreloadCapability::Unsealed;
    }
    virtual SpmFmbCompositeOccurrenceCapability
    spm_fmb_composite_occurrence_capability() const {
        return SpmFmbCompositeOccurrenceCapability::Unsealed;
    }
    virtual uint64_t
    spm_fmb_composite_occurrence_policy_fingerprint() const {
        return 0;
    }
    virtual SpmFmbPostFnYieldCapability
    spm_fmb_post_fn_yield_capability() const {
        return SpmFmbPostFnYieldCapability::Unsealed;
    }
    virtual uint64_t spm_fmb_post_fn_yield_policy_fingerprint() const {
        return 0;
    }
    virtual SpmFmbLayerProducerYieldCapability
    spm_fmb_layer_producer_yield_capability() const {
        return SpmFmbLayerProducerYieldCapability::Unsealed;
    }
    virtual uint64_t
    spm_fmb_layer_producer_yield_policy_fingerprint() const {
        return 0;
    }
    virtual SpmFmbActiveGroupCapability
    spm_fmb_active_group_capability() const {
        return SpmFmbActiveGroupCapability::Unsealed;
    }
    virtual SpmFmbDenseDdrMemberCapability
    spm_fmb_dense_ddr_member_capability() const {
        return SpmFmbDenseDdrMemberCapability::Unsealed;
    }
    virtual SpmFmbDmaEndpointCapability
    spm_fmb_dma_endpoint_capability() const {
        return SpmFmbDmaEndpointCapability::Unsealed;
    }
    virtual SpmFmbRuntimeMemberCapability
    spm_fmb_runtime_member_capability() const {
        return SpmFmbRuntimeMemberCapability::Unsealed;
    }
    virtual SpmFmbSpmPeerCapability spm_fmb_spm_peer_capability() const {
        return SpmFmbSpmPeerCapability::Unsealed;
    }
    // Nonzero, handle-stable identity of every semantic choice that can alter
    // which sealed buffer name a schema-v13 callback selects.  It enters the
    // Graph signature; a policy change must resolve a new profile/Graph.
    virtual uint64_t spm_fmb_spm_peer_policy_fingerprint() const {
        return 0;
    }

    // Append after the existing virtual sequence: earlier slot order stays
    // unchanged. All base/derived native objects must still be rebuilt together;
    // an old compiled derived vtable cannot service this new slot.
    // The shared mint has already installed the actual KV selector/22-word plan.
    // Owners may now rebind dependent routes before validation/layout/encoding.
    // Existing owners intentionally keep the default no-op behavior.
    virtual void rebind_kvinsert_exact_candidate(
        FmbPrefillStageCandidate& /*candidate*/, const LayoutContext& /*layout*/,
        int64_t /*site_id*/, int64_t /*invocation*/,
        const KvInsertSegmentPlan& /*plan*/) const {}

    // Planning-only query for adapters that must shape input before dispatch.
    // Uses the same exact SPM and model validator path as run_all_layers().
    int64_t resolve_chunk_size_for_shape(
        int64_t seq_len, int64_t position,
        const std::optional<at::Tensor>& attention_mask, bool is_causal);
    std::vector<FmbPrefillStageCandidate>
    resolve_prefill_stage_domain_for_shape(
        int64_t seq_len, int64_t position,
        const std::optional<at::Tensor>& attention_mask, bool is_causal,
        int64_t requested_chunk_size = 0, int64_t logical_len = 0,
        int64_t planning_chunk_size_override = -1);
    std::vector<FmbPrefillStageCandidate>
    resolve_prefill_stage_domain_for_shape(
        int64_t seq_len, int64_t position,
        const std::optional<at::Tensor>& attention_mask, bool is_causal,
        const std::vector<ChunkInfo>& input_chunks,
        const std::vector<FmbExecutionSpan>& spans,
        FmbStageBoundaryPolicies boundary_policies,
        int64_t requested_chunk_size = 0, int64_t logical_len = 0,
        int64_t planning_chunk_size_override = -1);
    SpmPipelineCausalPrefillShape
    resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract(
        int64_t execution_len);

    // Transitional physical-pipeline hooks. prepare establishes the normal
    // zero-based cached component layout. adopt validates a checked physical
    // scratch view and applies its base only through the typed addr accessors;
    // the cached layout and layout hash remain zero-based and immutable.
    SpmPipelineComponentLayout prepare_spm_pipeline_component(
        const LayoutContext& layout_ctx,
        const FmbThreeStageChunkPlan& stage_plan);
    // Board-free contract seam.  Builds the same owned zero-based allocation
    // snapshot with the pure first-fit planner but never admits physical use.
    SpmPipelineComponentLayout
    prepare_spm_pipeline_component_for_cpu_contract(
        const LayoutContext& layout_ctx,
        const FmbThreeStageChunkPlan& stage_plan);
    // Pure layout identity for Graph admission. It reuses the CPU first-fit
    // planner but publishes no allocator, manifest, or handle state.
    SpmPipelineComponentLayout
    resolve_spm_pipeline_component_layout_for_cpu_contract(
        const LayoutContext& layout_ctx,
        const FmbThreeStageChunkPlan& stage_plan);
    // Non-mutating exact feasibility predicate for custom native stage
    // resolvers.  The caller supplies descriptor-derived layout fields; this
    // uses the same BufferDecl budget and current persistent occupancy as A6.
    bool spm_pipeline_component_layout_fits_for_cpu_contract(
        const LayoutContext& layout_ctx,
        const FmbThreeStageChunkPlan& stage_plan);
    // Symmetric retirement for a live-handle board-free dry probe.  It drops
    // only CPU-contract manifest/declaration authority; allocator state,
    // cached ordinary offsets, weights, and model generation are untouched.
    void cancel_spm_pipeline_component_for_cpu_contract();
    void set_spm_pipeline_occurrence_schedule(
        const SpmFmbOccurrenceSchedule& schedule);
    SpmFmbSealedPhaseManifest seal_spm_pipeline_manifest(
        SpmScratchId arena,
        uint32_t arena_base) const;
    SpmFmbResolvedExecutionProfile
    resolve_spm_pipeline_execution_profile_for_cpu_contract(
        const SpmFmbResolvedProfileRequest& request);
    // Live counterpart used only by the explicit BUILD access-trace path.  It
    // binds the profile to a real allocation snapshot; the schema-v7 manifest
    // seal below continues to accept CPU-dry profiles only.
    SpmFmbResolvedExecutionProfile
    resolve_spm_pipeline_execution_profile_for_build_trace(
        const SpmFmbResolvedProfileRequest& request);
    SpmFmbResolvedPhaseManifest seal_spm_pipeline_resolved_manifest(
        const SpmFmbResolvedExecutionProfile& profile,
        SpmScratchId arena,
        uint32_t arena_base) const;
    // Live sibling for physical-pipeline planning.  It carries no physical
    // admission by itself; only a later opaque endpoint/yield compiler may
    // consume it.
    SpmFmbResolvedPhaseManifest seal_spm_pipeline_live_resolved_manifest(
        const SpmFmbResolvedExecutionProfile& profile,
        SpmScratchId arena,
        uint32_t arena_base) const;
    SpmFmbConsumerRowSliceEndpoint
    seal_spm_pipeline_consumer_row_slice_endpoint(
        const SpmFmbResolvedExecutionProfile& profile,
        const SpmFmbResolvedPhaseManifest& live_manifest,
        const char* owned_buffer_name,
        int allocation_layer,
        SpmChunkKey consumer_chunk,
        const SpmDense2DSpec& storage_spec,
        int64_t row_begin,
        int64_t row_count,
        const SpmFmbConsumerOccurrenceSelector& occurrence) const;
    SpmFmbPostFnYieldScope begin_spm_pipeline_post_fn_dense_yield(
        const char* owned_preflight_output_name,
        int allocation_layer,
        const SpmDense2DSpec& produced_spec);
    // Generic spelling for callback-window producers.  The legacy PostFn
    // spelling remains as a compatibility facade; admission is determined by
    // the sealed capability and the current canonical callback kind.
    SpmFmbPostFnYieldScope begin_spm_pipeline_dense_producer_yield(
        const char* owned_preflight_output_name,
        int allocation_layer,
        const SpmDense2DSpec& produced_spec);
    SpmFmbSealedPostFnYields seal_spm_pipeline_post_fn_yields(
        const SpmFmbSealedBuildTrace& trace,
        const ::RpuKernelGraph& graph) const;
    void validate_spm_pipeline_post_fn_yields(
        const SpmFmbSealedPostFnYields& yields,
        const ::RpuKernelGraph& graph) const;
    void arm_spm_pipeline_post_fn_yield_replay(
        const SpmFmbSealedPostFnYields& yields,
        ::RpuKernelGraph& graph) const;
    void cancel_spm_pipeline_post_fn_yield_replay() noexcept;
    SpmFmbSealedCallbackYields seal_spm_pipeline_callback_yields(
        const SpmFmbSealedBuildTrace& trace,
        const ::RpuKernelGraph& graph) const;
    void validate_spm_pipeline_callback_yields(
        const SpmFmbSealedCallbackYields& yields,
        const ::RpuKernelGraph& graph) const;
    void arm_spm_pipeline_callback_yield_replay(
        const SpmFmbSealedCallbackYields& yields,
        ::RpuKernelGraph& graph) const;
    void drive_spm_pipeline_callback_yield_replay_for_cpu_contract(
        const SpmFmbSealedCallbackYields& yields);
    // Board-free composite contract driver.  Production occurrences re-enter
    // run_all_layers normally; this helper replays the already-installed
    // opaque callback authority without exposing its token to tests/callers.
    void drive_spm_pipeline_composite_callback_yield_replay_for_cpu_contract();
    void cancel_spm_pipeline_callback_yield_replay() noexcept;
    // Schema-v8 CPU-dry foundation.  begin() must run inside the exact Graph
    // RECORDING scope whose signature segment key equals profile_hash().  A
    // production run_all_layers consumes only a live profile; the protected
    // CPU driver consumes only a CPU-dry profile and invokes the same resolved
    // kv/layer callbacks without requiring RPU tensors.  seal() is legal only
    // after Graph::end() has committed the same build as BUILT.  The initial
    // fail-closed contract permits one armed FMB trace per owner thread/outer
    // Graph; composite multi-owner tracing needs a later coordinator token.
    // The trace is thread-affine through seal/cancel.  If Graph::end() throws,
    // the owner must catch, cancel the FMB trace, and abort the Graph; moving or
    // destroying an armed owner on another thread is unsupported.
    void begin_spm_pipeline_build_trace(
        const SpmFmbResolvedExecutionProfile& profile);
    void drive_spm_pipeline_build_trace_for_cpu_contract(
        const SpmFmbResolvedExecutionProfile& profile);
    void drive_spm_pipeline_post_fn_yield_replay_for_cpu_contract(
        const SpmFmbSealedPostFnYields& yields);
    SpmFmbSealedBuildTrace seal_spm_pipeline_build_trace(
        const SpmFmbResolvedExecutionProfile& profile,
        const ::RpuKernelGraph& graph);
    void validate_spm_pipeline_build_trace(
        const SpmFmbSealedBuildTrace& trace,
        const ::RpuKernelGraph& graph) const;
    // Schema-v9 CPU-only logical grouping.  The opaque result is derived from
    // the private schema-v8 callback payload plus the model's typed capability;
    // callers cannot author group boundaries or convert it into admission.
    SpmFmbSealedActiveGroups seal_spm_pipeline_active_groups(
        const SpmFmbSealedBuildTrace& trace,
        const ::RpuKernelGraph& graph) const;
    void validate_spm_pipeline_active_groups(
        const SpmFmbSealedActiveGroups& groups,
        const ::RpuKernelGraph& graph) const;
    SpmFmbSealedDenseDdrMembers seal_spm_pipeline_dense_ddr_members(
        const SpmFmbSealedActiveGroups& groups,
        const ::RpuKernelGraph& graph) const;
    void validate_spm_pipeline_dense_ddr_members(
        const SpmFmbSealedDenseDdrMembers& members,
        const ::RpuKernelGraph& graph) const;
    GraphDmaSemanticEndpoint mint_spm_pipeline_dma_endpoint(
        const SpmFmbResolvedExecutionProfile& profile,
        size_t layer_group_ordinal,
        size_t member_ordinal,
        SpmFmbDmaEndpointRole role,
        uint8_t expected_occurrence_count) const;
    SpmFmbSealedDmaEndpointBindings
    seal_spm_pipeline_dma_endpoint_bindings(
        const SpmFmbSealedDenseDdrMembers& members,
        const ::RpuKernelGraph& graph) const;
    void validate_spm_pipeline_dma_endpoint_bindings(
        const SpmFmbSealedDmaEndpointBindings& bindings,
        const ::RpuKernelGraph& graph) const;
    void arm_spm_pipeline_dma_endpoint_replay(
        const SpmFmbSealedDmaEndpointBindings& bindings,
        ::RpuKernelGraph& graph) const;
    void drive_spm_pipeline_dma_endpoint_replay_for_cpu_contract(
        const SpmFmbSealedDmaEndpointBindings& bindings);
    SpmFmbSealedSpmPeerBindings seal_spm_pipeline_spm_peer_bindings(
        const SpmFmbSealedDmaEndpointBindings& bindings,
        const ::RpuKernelGraph& graph) const;
    void validate_spm_pipeline_spm_peer_bindings(
        const SpmFmbSealedSpmPeerBindings& bindings,
        const ::RpuKernelGraph& graph) const;
    void arm_spm_pipeline_spm_peer_replay(
        const SpmFmbSealedSpmPeerBindings& bindings,
        ::RpuKernelGraph& graph) const;
    void drive_spm_pipeline_spm_peer_replay_for_cpu_contract(
        const SpmFmbSealedSpmPeerBindings& bindings);
    void cancel_spm_pipeline_build_trace();
    void adopt_spm_pipeline_component(const SpmPipelineLease& lease,
                                      const SpmTensorView& scratch);
    void validate_spm_pipeline_component(
        const SpmPipelineLease& lease) const;
    void release_spm_pipeline_component(uint64_t epoch,
                                        uint64_t plan_hash);

    // Schema-5 outer-fast component authority.  Concrete composite models
    // call this once per invocation before emitting or atomically skipping
    // their Graph window.  The Graph receives strong current K/V owners and
    // weak generation stamps only; no SPM/DDR address escapes this facade.
    void stage_spm_outer_fast_component(
        ::GraphKernelRegisterCensusGuard& guard,
        at::TensorList k_caches,
        at::TensorList v_caches,
        std::vector<
            std::pair<std::shared_ptr<const uint64_t>, uint64_t>>
            extra_owner_stamps = {}) const;
    void bind_spm_outer_fast_input(
        ::GraphKernelRegisterCensusGuard& guard,
        const at::Tensor& hidden_states);
    void bind_spm_outer_fast_composite_input(
        ::GraphKernelRegisterCensusGuard& guard,
        const at::Tensor& hidden_states,
        size_t producer_ordinal);

    // Narrow facade (D-203)
    InferenceContext&       ctx();
    const InferenceContext& ctx() const;
    void begin_external_physical_manifest_prologue(
        const FmbPhysicalExecutionManifest& manifest,
        int64_t physical_length, int64_t position);
    void cancel_external_physical_manifest_prologue();
    SdpaStableMaskCache& sdpa_stable_mask_cache();

    uint32_t  addr(int core, const char* name) const;
    uint32_t  layer_addr(int layer, int core, const char* name) const;
    SpmOffset addr_offset(const char* name) const;                     // SDPA / KV-insert only
    SpmOffset layer_addr_offset(int layer, const char* name) const;    // SDPA / KV-insert only

    // Batch decode row accounting: a chunk carries ctx().batch_size * chunk.len
    // rows, laid out batch-major. batch_size == 1 → chunk.len / chunk.offset
    // unchanged. Subclasses use these for every row-parallel kernel's M.
    int64_t chunk_rows_(const ChunkInfo& chunk) const;
    int64_t chunk_row_offset_(const ChunkInfo& chunk) const;

    void emit_layer_input_dma (int layer_idx, const ChunkInfo& chunk,
                               const char* dst_buf = "residual1",
                               int64_t dst_row_offset = 0);
    // Copy one contiguous row run from the current caller-owned layer-0 input
    // into the matching rows of the current chunk's SPM allocation.  This is a
    // graph-aware mutable DMA helper for composite consumers whose producer
    // fills the other rows directly in SPM.  It does not grant physical target
    // authority; callers must already own/adopt the destination allocation.
    void emit_layer_input_row_run_dma(
        int layer_idx,
        const ChunkInfo& chunk,
        int64_t global_row_begin,
        int64_t row_count,
        const char* dst_buf = "residual1");
    // Explicit-slot counterpart for retained composite Graphs with several
    // independently bound row runs from one logical input tensor.  Each run
    // must retain its own live-base address so outer-fast REPLAY can validate
    // and patch the exact captured mutable-DMA slot without re-walking ops.
    void emit_layer_input_row_run_dma(
        uint64_t& live_src_base,
        int layer_idx,
        const ChunkInfo& chunk,
        int64_t global_row_begin,
        int64_t row_count,
        const char* dst_buf = "residual1");
    // Consume one retained composite payload with the existing alpha=1 SPM
    // ADD launcher.  Placement stays opaque: the caller supplies only its
    // owned residual geometry, while the lexical physical authority resolves
    // and consumes the next sealed tap-major target.
    void emit_spm_pipeline_composite_run_add(
        int layer_idx,
        const ChunkInfo& chunk,
        int64_t global_row_begin,
        int64_t row_count,
        const char* dst_buf = "residual1");
    void require_spm_pipeline_composite_consumer_physical_authority() const;
    void emit_layer_output_dma(int layer_idx, const ChunkInfo& chunk,
                               const char* src_buf = "residual1",
                               uint32_t src_offset_bytes = 0);
    // DDR scratch at the current layer's ping-pong output side. Unlike
    // emit_layer_output_dma(), this never redirects the last layer to the final
    // output tensor. Multi-stage layer implementations use the pair to spill an
    // intermediate, consume it later in the same layer, and then emit the real
    // layer output with emit_layer_output_dma().
    void emit_layer_chain_scratch_store(int layer_idx, const ChunkInfo& chunk,
                                        const char* src_buf);
    // Registry-stable DDR addresses backing DDR_PINGPONG. These are for
    // kernels that consume DDR directly; layer 0's caller-owned input is not
    // stable and is intentionally rejected by stable_layer_input_ddr_ptr().
    const c10::Half* stable_layer_input_ddr_ptr(
        int layer_idx, const ChunkInfo& chunk) const;
    const c10::Half* layer_chain_scratch_ddr_ptr(
        int layer_idx, const ChunkInfo& chunk) const;

    size_t current_spm_pipeline_dense_ddr_member_count() const;
    SpmFmbRuntimeDenseDdrMemberRef
    current_spm_pipeline_dense_ddr_member(size_t ordinal) const;
    SpmFmbIndependentLocalContext spm_pipeline_member_local_context(
        const SpmFmbRuntimeDenseDdrMemberRef& member) const;
    void emit_spm_pipeline_member_ingress_dma(
        const SpmFmbRuntimeDenseDdrMemberRef& member,
        const char* dst_buf = "residual1");
    void emit_spm_pipeline_member_egress_dma(
        const SpmFmbRuntimeDenseDdrMemberRef& member,
        const char* src_buf = "residual1");
    void emit_mlp_pipeline(const at::Tensor& gate_w, const at::Tensor& up_w,
                           const at::Tensor& down_w, int64_t seq_len,
                           ActivationKind act,
                           const at::Tensor& gate_ws = {},
                           const at::Tensor& up_ws = {},
                           const at::Tensor& down_ws = {},
                           uint32_t gate_nvfp4_ts_addr = 0,
                           uint32_t up_nvfp4_ts_addr = 0,
                           uint32_t down_nvfp4_ts_addr = 0,
                           uint16_t nvfp4_layer_id = 0,
                           bool down_out_bf16 = false,
                           bool residual_is_bf16 = false,
                           bool acc32 = false,
                           bool fuse_silu_mul = false,
                           bool skip_down = false,
                           bool bind_silu_mul_route = false,
                           uint32_t residual_spm_addr = 0,
                           bool pi05_xor3 = false,
                           bool pi05_nvfp4_v2 = false,
                           bool prefer_gemv = false,
                           RpuUnaryPrecision unary_precision = RpuUnaryPrecision::BASE,
                           bool high_precision_silu_mul = false);

    // D-503 — stays PROTECTED. External callers must go through a concrete
    // subclass's public wrapper (e.g., SmokeModelV3::public_invalidate_for_test).
    // Promoting to public would leak a test-only API onto every production subclass.
    //
    // EXT-4: sets BOTH weights_dirty_ AND preload_callbacks_dirty_; each cleared
    // at its OWN dispatch point (see run_preload_callbacks_ and run_all_layers).
    // Keep Graph/preload invalidation even for a verified numerical-only refresh.
    // Only unchanged tensor layouts and execution routes may preserve plan reuse.
    void invalidate_model_state(bool planning_domain_changed = true);
    void invalidate_planner_cache();

    // Post-graph output (Phase 2.5 — e.g. Qwen3 fused lm_head)
    at::Tensor&       post_output_tensor();
    const at::Tensor& post_output_tensor() const;

    // Final DDR output
    c10::Half*        output_ptr();
    const at::Tensor& output_tensor() const;

    // Model-param setter (subclass calls from set_weights / ctor)
    // Reduced owners explicitly resolve their own admitted physical geometry.
    // The default preserves the dense causal-decoder admission boundary.
    virtual DecoderExecutionTopology resolve_model_execution_topology(
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size) const;
    void set_model_params(int64_t num_q_heads, int64_t num_kv_heads,
                          int64_t head_dim, int64_t hidden_size,
                          int64_t intermediate_size);
    void set_num_layers(int64_t n);
    // Cold model configuration: must precede weight/layout installation.
    void set_execution_core_count(int num_cores);

    // Pitfall 4 mitigation — fresh at::empty per forward for caller-accumulated lists
    at::Tensor allocate_tracked_output(at::IntArrayRef shape);

    // Test-only helper used by SmokeModelV3 (Plan 01-01 smoke test). Runs the
    // preload path (ensure_allocated + run_preload_callbacks_) without
    // requiring the full run_all_layers input contract (hidden_states /
    // kv caches / position / mask). Not called by production subclasses.
    void drive_preload_for_test();
    // Run the callbacks already owned by a prepared layout without changing
    // allocation identity. Used by access-scope tests inside an active trace.
    void drive_preload_callbacks_for_test();

    // CPU-dry contract probes for the private composite occurrence runtime.
    // Production adapters consume the same state only through run_all_layers.
    std::pair<uintptr_t, uint64_t>
    exchange_spm_pipeline_composite_input_slot_for_cpu_contract(
        uint64_t value);
    bool consume_spm_pipeline_composite_preload_role_for_cpu_contract();

    // Model-param accessors (subclass reads from build_layer_subgraph)
    int64_t num_q_heads()       const;
    int64_t num_kv_heads()      const;
    int64_t head_dim()          const;
    int64_t hidden_size()       const;
    int64_t intermediate_size() const;
    int     attn_tp()           const;
    int     num_cores()         const;
    int     mlp_tp()            const;
    int64_t num_layers()        const;

    // PImpl — internal framework state hides here (Impl forward-declared in public).
    std::unique_ptr<Impl> pimpl_;

private:
    // Framework-private dispatch helpers
    std::vector<FmbPrefillStageCandidate>
    resolve_prefill_stage_domain_for_shape_impl(
        int64_t seq_len, int64_t position,
        const std::optional<at::Tensor>& attention_mask, bool is_causal,
        const std::vector<ChunkInfo>* input_chunks,
        const std::vector<FmbExecutionSpan>* spans,
        const FmbStageBoundaryPolicies* boundary_policies,
        int64_t requested_chunk_size, int64_t logical_len,
        int64_t planning_chunk_size_override);
    SpmPipelineComponentLayout prepare_spm_pipeline_component_impl(
        const LayoutContext& layout_ctx);
    SpmPipelineComponentLayout
    prepare_spm_pipeline_component_for_cpu_contract_impl(
        const LayoutContext& layout_ctx);
    at::Tensor run_all_layers_impl(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal,
        const std::vector<ChunkInfo>* input_chunks,
        const std::vector<FmbExecutionSpan>* spans,
        const FmbStageBoundaryPolicies* boundary_policies,
        uint64_t expected_layout_hash,
        int64_t planned_chunk_size,
        const FmbPrefillStageCandidate* planned_stage_candidate,
        std::shared_ptr<const FmbPreparedStageCandidate> prepared_candidate = {});
    void run_preload_callbacks_(const std::vector<BufferDecl>& decls,
                                FusedModelBase& self);
    SpmFmbResolvedExecutionProfile
    resolve_spm_pipeline_execution_profile_impl(
        const SpmFmbResolvedProfileRequest& request,
        bool require_cpu_dry);
    SpmFmbResolvedPhaseManifest
    seal_spm_pipeline_resolved_manifest_impl(
        const SpmFmbResolvedExecutionProfile& profile,
        SpmScratchId arena,
        uint32_t arena_base,
        bool require_cpu_dry) const;
    void begin_spm_pipeline_runtime_member_callback_for_build(
        const SpmFmbResolvedExecutionProfile& profile,
        const SpmFmbResolvedExecutionStep& step);
    void begin_spm_pipeline_runtime_member_callback_for_replay(
        const SpmFmbSealedDmaEndpointBindings& bindings,
        size_t callback_index);
    void begin_spm_pipeline_runtime_member_callback_for_replay(
        const SpmFmbSealedSpmPeerBindings& bindings,
        size_t callback_index);
    void end_spm_pipeline_runtime_member_callback();
    void cancel_spm_pipeline_runtime_member_callback() noexcept;
    void finish_spm_pipeline_post_fn_dense_yield(
        SpmFmbPostFnYieldScope& scope);
    SpmFmbPostFnYieldScope begin_spm_pipeline_dense_producer_yield_impl(
        const char* owned_preflight_output_name,
        int allocation_layer,
        const SpmDense2DSpec& produced_spec,
        bool callback_window_contract);
    void cancel_spm_pipeline_post_fn_dense_yield(
        SpmFmbPostFnYieldScope& scope) noexcept;
    void begin_spm_pipeline_build_trace_impl(
        const SpmFmbResolvedExecutionProfile& profile,
        bool composite,
        uint64_t composite_identity,
        size_t composite_occurrence_ordinal,
        bool composite_producer,
        size_t composite_producer_local_ordinal,
        SpmFmbCompositeOccurrenceCapability composite_capability,
        uint64_t composite_policy_fingerprint,
        bool composite_preload_follower);
    void arm_spm_pipeline_composite_occurrence_runtime(
        uint64_t composite_identity,
        size_t producer_local_ordinal,
        SpmFmbCompositeOccurrenceCapability capability,
        uint64_t policy_fingerprint,
        uint64_t expected_layout_hash,
        uint64_t expected_allocation_hash,
        uint64_t expected_persistent_generation,
        uint64_t composite_occurrence_profile_hash,
        bool preload_follower);
    void finish_spm_pipeline_composite_occurrence_runtime(
        uint64_t composite_identity,
        size_t producer_local_ordinal);
    void cancel_spm_pipeline_composite_occurrence_runtime() noexcept;
    void adopt_spm_pipeline_composite_component(
        const SpmPipelineLease& lease,
        bool producer);
    void validate_spm_pipeline_composite_component(
        const SpmPipelineLease& lease,
        bool producer) const;
    void arm_spm_pipeline_composite_physical_execution(
        const SpmPipelineLease& lease,
        uint64_t composite_identity,
        bool producer,
        size_t producer_local_ordinal,
        bool require_committed);
    void finish_spm_pipeline_composite_physical_execution(
        const SpmPipelineLease& lease,
        uint64_t composite_identity,
        bool producer,
        size_t producer_local_ordinal);
    void cancel_spm_pipeline_composite_physical_execution() noexcept;
    uint64_t& current_hidden_in_src_base();
    bool consume_spm_pipeline_composite_preload_follower();
    SpmFmbCompletedBuildTrace
    complete_spm_pipeline_composite_build_trace();
    SpmFmbSealedBuildTrace seal_spm_pipeline_completed_build_trace(
        SpmFmbCompletedBuildTrace&& completed,
        const ::RpuKernelGraph& graph);
    const SpmFmbRuntimeDenseDdrMemberRef::State&
    validate_spm_pipeline_runtime_member_ref(
        const SpmFmbRuntimeDenseDdrMemberRef& member) const;
    void end_spm_pipeline_build_trace_callback();

    friend class SpmFmbPostFnYieldScope;
    friend class SpmCompositeTraceCoordinator;
};

}  // namespace v3
