// fused_model_base.h — flat single-class framework contract
//
// Exposes four mandatory virtuals, a narrow facade, and run_all_layers.
// Internal state hides behind PImpl (Impl defined in fused_model_base_impl.h).

#pragma once

#include <ATen/ATen.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

class RpuKernelGraph;
class GraphDmaSemanticEndpoint;
class GraphKernelRegisterCensusGuard;
class SdpaStableMaskCache;

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

// Public enum numeric values are stable; new values are append-only.
enum class ChunkMode       : int { SEQUENTIAL = 0, KV_FIRST = 1 };
enum class InterLayerIO    : int { AUTO = 0, SPM_RESIDENT = 1, DDR_PINGPONG = 2 };
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

// Buffer, layout, and chunk structs.
struct BufferDecl {
    const char*  name         = nullptr;
    int64_t      size         = 0;
    int          phase_start  = 0;
    int          phase_end    = 0;
    StorageClass storage      = StorageClass::Temp;
    int          per_layer    = 0;
    const char*  alias_of     = nullptr;
    BufferScope  scope        = BufferScope::LayerWide;

    // For Persistent / PersistentPerLayer buffers, the framework auto-wires:
    //   (1) captures SPM_ALLOC.persistent_generation() at allocation
    //   (2) re-fires on generation advance or preload_callbacks_dirty_
    //   (3) passes layer_idx (-1 for non-per-layer) + core-0 absolute SPM addr
    //
    // Graph-capture contract:
    //   The caller's GraphCache/raw-Graph scope owns capture. Every rpu_launch_*
    //   emitted by the complete preload-callback loop enters that active graph
    //   through RpuQueue/graph_dma. Callbacks MUST NOT open a nested graph or raw
    //   batch scope themselves.
    std::function<void(FusedModelBase& /*self*/,
                       int             /*layer_idx*/,
                       uint32_t        /*core0_addr*/)> preload_callback;

    // For fused AdaRMS broadcast, allocate this PersistentPerLayer decl's layers
    // in DESCENDING layer order (L = per_layer-1 .. 0). alloc_super_persistent is a
    // DOWNWARD bump allocator (see rpu_spm_allocator.cpp), so the default ascending
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
// Prefer fewer chunks, then the most even fixed-size chunk + tail split.
bool prefer_balanced_chunk(int64_t seq_len, int64_t candidate,
                           int64_t current);
}  // namespace detail

struct LayoutContext {
    int64_t chunk_size = 0;
    int64_t kv_insert_chunk_size = 0;  // 0 = same as chunk_size
    int64_t max_kv_seq_len = 0;
    int64_t num_layers = 0;
    bool    use_attn_mask = false;
    // The attention layout mode (causal vs bidirectional-
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

    int64_t effective_kv_cs() const {
        return kv_insert_chunk_size > 0 ? kv_insert_chunk_size : chunk_size;
    }
    // Rows packed into one GEMM: batch-major, [batch][chunk] contiguous.
    int64_t rows() const { return batch_size * chunk_size; }
};

struct ChunkInfo { int idx = 0; int64_t offset = 0; int64_t len = 0; int64_t kv_seq_len = 0; };
struct ChunkPlan { int64_t chunk_size = 0; int64_t num_chunks = 0; };

// Board-free causal-prefill planning result.  `resolved_chunk_size` is the
// v16 candidate selected by the planner; `allocation_layout.chunk_size` is the
// first real ChunkInfo length used by run_all_layers for SPM allocation.  They
// intentionally differ for a short one-chunk tail such as S225 (240 vs 225).
struct SpmPipelineCausalPrefillShape {
    int64_t resolved_chunk_size = 0;
    std::vector<ChunkInfo> chunks;
    std::vector<ChunkInfo> kv_insert_chunks;
    LayoutContext allocation_layout;
    ChunkMode chunk_mode = ChunkMode::SEQUENTIAL;
    InterLayerIO inter_layer_io = InterLayerIO::AUTO;
    bool chunk_outer_within_group = false;
};

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

// Typed SPM offset for SDPA / KV-insert only.
// explicit operator uint32_t() forces conversion at each call site; mixing
// offset vs absolute address becomes a compile-time error, not a runtime bug.
struct SpmOffset {
    uint32_t value = 0;
    explicit operator uint32_t() const { return value; }
};

// Description of one component's already-planned, zero-based
// temporary layout.  It deliberately carries no offsets: the physical pipeline
// adopts the exact cached FusedModelBase layout only after validating every
// buffer against its checked scratch view.
struct SpmPipelineComponentLayout {
    size_t temporary_bytes = 0;
    uint64_t layout_hash = 0;
};

struct SpmPipelineCausalPrefillDryLayout {
    SpmPipelineCausalPrefillShape shape;
    SpmPipelineComponentLayout component;
};

// ModelStaticConfig carries optional pointer-to-member callbacks.
// Framework discovers features by inspecting which fields are non-null —
// no extra virtuals on the subclass header.
//
// Graph admission is owned by the Python adapter. C++ run_all_layers emits
// operations and does not select cache entries or construct signatures.
struct ModelStaticConfig {
    int64_t                 num_layers           = 0;
    int64_t                 cross_layer_batch_size = 0;

    // Optional callbacks — null means disabled.
    // Dispatched via std::invoke(cfg.xxx_fn, *this) inside run_all_layers.
    void      (FusedModelBase::*preload_fn)()                              = nullptr;  // SigLIP
    void      (FusedModelBase::*kv_first_fn)(int, const ChunkInfo&)        = nullptr;  // Gemma Phase 1
    ChunkPlan (FusedModelBase::*kv_first_chunk_plan_fn)(const ChunkPlan&)  = nullptr;  // Gemma dual-chunk
    void      (FusedModelBase::*post_fn)()                                 = nullptr;  // Qwen3 fused lm_head
    // pre_layers_fn / post_layers_fn are called by run_all_layers
    // BEFORE the first layer / AFTER the last layer, inside the same
    // caller-owned capture when active. Used by Pi05DenoiseStepModel to emit action_in_proj
    // (pre) and final PiGemmaRMSNorm + action_out_proj (post). Subclass
    // bodies MUST only call rpu_launch_* / addr() / etc.; they MUST NOT
    // open a raw batch or nested GraphCache scope.
    void (FusedModelBase::*pre_layers_fn)()                                = nullptr;
    void (FusedModelBase::*post_layers_fn)()                               = nullptr;
    std::vector<int64_t> post_output_shape;
    // Repeat the pre_layers_fn → layer-loop → post_layers_fn body
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
};

struct ModelDynamicConfig {
    ChunkMode     chunk_mode     = ChunkMode::SEQUENTIAL;
    InterLayerIO  inter_layer_io = InterLayerIO::AUTO;
    // Optional traversal for independent chunks (for example, equal-size images):
    // group -> chunk -> layer. This keeps each chunk SPM-resident for one
    // cross-layer group, then spills only at the group boundary. The default
    // group -> layer -> chunk order is unchanged for every existing model.
    bool chunk_outer_within_group = false;
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
    int64_t body_iter      = 0;  // Current body iteration; 0 unless body_iterations > 1.
};

// FusedModelBase — flat framework. Subclass contract:
//   4 mandatory virtuals (declare_buffers, static_config, dynamic_config, build_layer_subgraph)
//   Optional features via ModelStaticConfig callback fields
//   Call invalidate_model_state() after set_weights(...)
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
        bool is_causal);

    void    set_chunk_size_override(int64_t cs);
    int64_t get_chunk_size_override() const;
    // Per-handle resolved chunk_size
    // from the most-recent forward; 0 if no forward has run yet on this instance.
    int64_t get_last_resolved_chunk_size() const;

    // Certified chunk envelope. Profiles can bound automatic planning more
    // tightly than the generic SPM budget; undeclared combinations are denied.
    //
    // Two fields are used because two different quantities scale:
    //   chunk       every path's ceiling; the thing that busts SPM.
    //   max_kv_len  load-bearing ONLY on the explicit-mask path, where
    //               declare_buffers adds sdpa_mask = comp_cs * ceil16(kv) * 32
    //               (see rpu_qwen3_model.h). With no mask the layout is FLAT in
    //               length, so there it only bounds the certified range over
    //               which `auto` has actually been certified.
    struct ChunkEnvelope {
        int64_t max_kv_len = 0;   // 0 = UNDECLARED => deny
        int64_t chunk      = 0;   // 0 = "auto is certified within max_kv_len"
        bool declared() const { return max_kv_len > 0; }
    };

    // Cold setter — must precede the first forward, mirroring the existing
    // chunk_size_cap contract so a live GraphCache entry cannot be hot-switched.
    void set_chunk_envelope(int64_t max_kv_len, int64_t chunk);
    const ChunkEnvelope& chunk_envelope() const;

protected:
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
        // An `auto` row does not authorize an arbitrary explicit override.
        // Zero-ceiling rows therefore remain closed on the override path.
        if (e.chunk <= 0) return get_chunk_size_override() <= 0;
        return cs <= e.chunk;
    }

    // Mandatory subclass virtuals.
    virtual std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) = 0;
    virtual ModelStaticConfig       static_config() = 0;
    virtual ModelDynamicConfig      dynamic_config(const ChunkPlan& plan) = 0;
    virtual void                    build_layer_subgraph(int layer_idx,
                                                         const ChunkInfo& chunk) = 0;

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
    // This hook is the only
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
    // kv_insert_chunk_size, num_layers, use_attn_mask, is_causal, batch_size and
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

    // Planning-only query for adapters that must shape input before dispatch.
    // Uses the same exact SPM and model validator path as run_all_layers().
    int64_t resolve_chunk_size_for_shape(
        int64_t seq_len, int64_t position,
        const std::optional<at::Tensor>& attention_mask, bool is_causal);
    SpmPipelineCausalPrefillShape
    resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract(
        int64_t execution_len);

    // Transitional physical-pipeline hooks. prepare establishes the normal
    // zero-based cached component layout. adopt validates a checked physical
    // scratch view and applies its base only through the typed addr accessors;
    // the cached layout and layout hash remain zero-based and immutable.
    SpmPipelineComponentLayout prepare_spm_pipeline_component(
        const LayoutContext& layout_ctx);
    // Board-free contract seam.  Builds the same owned zero-based allocation
    // snapshot with the pure first-fit planner but never admits physical use.
    SpmPipelineComponentLayout
    prepare_spm_pipeline_component_for_cpu_contract(
        const LayoutContext& layout_ctx);
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
    // opaque callback authority without exposing its token to external callers.
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

    // Narrow facade.
    InferenceContext&       ctx();
    const InferenceContext& ctx() const;
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
                           bool force_twostage_chunk_v2 = false,
                           bool down_out_bf16 = false,
                           bool residual_is_bf16 = false,
                           bool acc32 = false);

    // Stays protected. External callers must go through a concrete
    // subclass's public wrapper (e.g., SmokeModelV3::public_invalidate_for_test).
    // Promoting to public would leak a test-only API onto every production subclass.
    //
    // Sets both weights_dirty_ and preload_callbacks_dirty_; each is cleared
    // at its OWN dispatch point (see run_preload_callbacks_ and run_all_layers).
    void invalidate_model_state();

    // Post-graph output (Phase 2.5 — e.g. Qwen3 fused lm_head)
    at::Tensor&       post_output_tensor();
    const at::Tensor& post_output_tensor() const;

    // Final DDR output
    c10::Half*        output_ptr();
    const at::Tensor& output_tensor() const;

    // Model-param setter (subclass calls from set_weights / ctor)
    void set_model_params(int64_t num_q_heads, int64_t num_kv_heads,
                          int64_t head_dim, int64_t hidden_size,
                          int64_t intermediate_size);
    void set_num_layers(int64_t n);

    // Pitfall 4 mitigation — fresh at::empty per forward for caller-accumulated lists
    at::Tensor allocate_tracked_output(at::IntArrayRef shape);

    // Diagnostic helper used by the conformance model. Runs the
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
    int64_t num_layers()        const;

    // PImpl — internal framework state hides here (Impl forward-declared in public).
    std::unique_ptr<Impl> pimpl_;

private:
    // Framework-private dispatch helpers
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
