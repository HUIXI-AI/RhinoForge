#pragma once

#include "rpu_spm_allocator.h"
#include "graph/execution_coordinator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

class RpuKernelGraph;
class RpuPhysicalArenaAuthority;

namespace v3 {

struct BufferDecl;
enum class BufferScope : uint8_t;
class FusedModelBase;
struct SpmFmbYieldTargetLauncherAccess;
class SpmPipelineLease;
class SpmFmbSealedSpmPeerBindings;
class SpmFmbSealedCallbackYields;
class SpmFmbCompletedBuildTrace;
class SpmFmbSealedCompositeTrace;
class SpmCompositeOccurrenceScope;
class SpmCompositeOuterFastReplayTicket;
class SpmCompositeTraceCoordinator;
class SpmFmbDirectProducedPostFnBinding;
class SpmFmbDirectProducedPostFnRunAddBinding;
class SpmFmbCompositeYieldBinding;
class SpmFmbCompositeDirectProducedBinding;
class SpmFmbCompositeRunAddBinding;
class SpmFmbCompositePhysicalReservation;
class SpmFmbCompositePhysicalPlan;
class SpmPermutationRowRunsView;
class SpmChunkedPermutationView;
class SpmDirectProducedRowRunsView;
class SpmDirectProducedPayloadRunAddView;
class SpmPhaseChunkView;
enum class SpmFmbActiveGroupCapability : uint8_t;
enum class SpmFmbPreloadCapability : uint8_t;
enum class SpmFmbCompositeOccurrenceCapability : uint8_t;
enum class SpmFmbPostFnYieldCapability : uint8_t;
enum class SpmFmbLayerProducerYieldCapability : uint8_t;
enum class SpmFmbTerminalWriterKind : uint8_t;
enum class SpmFmbDenseDdrMemberCapability : uint8_t;
enum class SpmFmbDmaEndpointCapability : uint8_t;
enum class SpmFmbRuntimeMemberCapability : uint8_t;
enum class SpmFmbSpmPeerCapability : uint8_t;
enum class SpmFmbDmaEndpointRole : uint8_t;

// Strong keys for the typed V2 path.  Scratch and port IDs intentionally use
// separate types/namespaces so equal numeric values cannot be confused.
class SpmScratchId {
public:
    explicit constexpr SpmScratchId(uint32_t value) : value_(value) {}
    constexpr uint32_t value() const { return value_; }

    friend constexpr bool operator==(SpmScratchId lhs, SpmScratchId rhs) {
        return lhs.value_ == rhs.value_;
    }
    friend constexpr bool operator!=(SpmScratchId lhs, SpmScratchId rhs) {
        return !(lhs == rhs);
    }

private:
    uint32_t value_;
};

class SpmPortId {
public:
    explicit constexpr SpmPortId(uint32_t value) : value_(value) {}
    constexpr uint32_t value() const { return value_; }

    friend constexpr bool operator==(SpmPortId lhs, SpmPortId rhs) {
        return lhs.value_ == rhs.value_;
    }
    friend constexpr bool operator!=(SpmPortId lhs, SpmPortId rhs) {
        return !(lhs == rhs);
    }

private:
    uint32_t value_;
};

// Schema-v5 uses one global, monotonically ordered event namespace.  Lifetimes
// are closed intervals: two slices at the same byte range may be overlaid only
// when one last event is strictly less than the other's first event.
class SpmPipelineEvent {
public:
    explicit constexpr SpmPipelineEvent(uint32_t value) : value_(value) {}
    constexpr uint32_t value() const { return value_; }

    friend constexpr bool operator==(SpmPipelineEvent lhs,
                                     SpmPipelineEvent rhs) {
        return lhs.value_ == rhs.value_;
    }
    friend constexpr bool operator!=(SpmPipelineEvent lhs,
                                     SpmPipelineEvent rhs) {
        return !(lhs == rhs);
    }
    friend constexpr bool operator<(SpmPipelineEvent lhs,
                                    SpmPipelineEvent rhs) {
        return lhs.value_ < rhs.value_;
    }

private:
    uint32_t value_;
};

// Chunk identity is local to one logical port.  Ordinals seal declaration
// order and therefore the global row boundaries used by schema-v5 lowering.
class SpmChunkKey {
public:
    constexpr SpmChunkKey(SpmPortId port, uint32_t ordinal)
        : port_(port), ordinal_(ordinal) {}
    constexpr SpmPortId port() const { return port_; }
    constexpr uint32_t ordinal() const { return ordinal_; }

    friend constexpr bool operator==(const SpmChunkKey& lhs,
                                     const SpmChunkKey& rhs) {
        return lhs.port_ == rhs.port_ && lhs.ordinal_ == rhs.ordinal_;
    }
    friend constexpr bool operator!=(const SpmChunkKey& lhs,
                                     const SpmChunkKey& rhs) {
        return !(lhs == rhs);
    }

private:
    SpmPortId port_;
    uint32_t ordinal_;
};

// V2 deliberately admits only the first proven identity seam.  New memory
// dtypes/distributions must be added explicitly rather than silently lowered.
enum class SpmPortDType : uint8_t {
    Fp16 = 1,
};

enum class SpmPortDistribution : uint8_t {
    Replicated = 1,
};

// A dense row-major [rows, cols] physical contract.  Batch-1 model views may
// flatten to this representation, but padded, strided, permuted, or sharded
// layouts are not identity-compatible with V2.
struct SpmDense2DSpec {
    SpmPortDType dtype = SpmPortDType::Fp16;
    int64_t rows = 0;
    int64_t cols = 0;
    SpmPortDistribution distribution = SpmPortDistribution::Replicated;

    void validate() const;
    size_t storage_bytes() const;

    friend bool operator==(const SpmDense2DSpec& lhs,
                           const SpmDense2DSpec& rhs) {
        return lhs.dtype == rhs.dtype && lhs.rows == rhs.rows &&
               lhs.cols == rhs.cols &&
               lhs.distribution == rhs.distribution;
    }
    friend bool operator!=(const SpmDense2DSpec& lhs,
                           const SpmDense2DSpec& rhs) {
        return !(lhs == rhs);
    }
};

struct SpmScratchDecl {
    SpmScratchId id;
    int64_t bytes;
    int first_event;
    int last_event;
};

// One exact producer -> consumer edge.  bind() validates both endpoints and
// seals their common spec before a plan or physical arena can be created.
class SpmIdentityRoute {
public:
    static SpmIdentityRoute bind(SpmPortId id,
                                 const SpmDense2DSpec& produced,
                                 const SpmDense2DSpec& required,
                                 int first_write,
                                 int last_read);

    SpmPortId id() const { return id_; }
    const SpmDense2DSpec& spec() const { return spec_; }
    int first_write() const { return first_write_; }
    int last_read() const { return last_read_; }

private:
    SpmIdentityRoute(SpmPortId id,
                     const SpmDense2DSpec& spec,
                     int first_write,
                     int last_read)
        : id_(id),
          spec_(spec),
          first_write_(first_write),
          last_read_(last_read) {}

    SpmPortId id_;
    SpmDense2DSpec spec_;
    int first_write_ = 0;
    int last_read_ = 0;
};

// One dense producer writes a contiguous row range of a larger dense consumer
// storage port.  The destination range is sealed into the plan; no scatter,
// reshape, cast, or redistribution is implied.
class SpmRowSliceRoute {
public:
    static SpmRowSliceRoute bind(SpmPortId id,
                                 const SpmDense2DSpec& produced,
                                 const SpmDense2DSpec& consumer_storage,
                                 int64_t dst_row_begin,
                                 int64_t dst_row_count,
                                 int first_write,
                                 int last_read);

    SpmPortId id() const { return id_; }
    const SpmDense2DSpec& produced_spec() const { return produced_spec_; }
    const SpmDense2DSpec& storage_spec() const { return storage_spec_; }
    int64_t dst_row_begin() const { return dst_row_begin_; }
    int64_t dst_row_count() const { return dst_row_count_; }
    int first_write() const { return first_write_; }
    int last_read() const { return last_read_; }

private:
    SpmRowSliceRoute(SpmPortId id,
                     const SpmDense2DSpec& produced_spec,
                     const SpmDense2DSpec& storage_spec,
                     int64_t dst_row_begin,
                     int64_t dst_row_count,
                     int first_write,
                     int last_read)
        : id_(id),
          produced_spec_(produced_spec),
          storage_spec_(storage_spec),
          dst_row_begin_(dst_row_begin),
          dst_row_count_(dst_row_count),
          first_write_(first_write),
          last_read_(last_read) {}

    SpmPortId id_;
    SpmDense2DSpec produced_spec_;
    SpmDense2DSpec storage_spec_;
    int64_t dst_row_begin_ = 0;
    int64_t dst_row_count_ = 0;
    int first_write_ = 0;
    int last_read_ = 0;
};

// One destination row run.  Runs are interpreted in declaration order: the
// first run consumes the first `row_count` entries of the sealed permutation,
// the next run consumes the next entries, and so on.
struct SpmDstRowRun {
    int64_t dst_row_begin = 0;
    int64_t row_count = 0;
};

// One maximal ascending source-row span lowered into one contiguous
// destination span.  Spans never cross a declared destination-run boundary.
struct SpmContiguousRowRun {
    int64_t source_row = 0;
    int64_t destination_row = 0;
    int64_t row_count = 0;
};

struct SpmPermutationRowRunsEvents {
    int source_first_write = 0;
    int destination_first_write = 0;
    int transform = 0;
    int destination_last_read = 0;
};

// One replicated dense source is gathered by an exact row permutation directly
// into ordered row runs of a larger replicated dense consumer storage.  The
// sealed mapping is lowered explicitly by the coordinator into maximal
// contiguous SPM source runs.  This route describes Z1 semantics only; it never
// implies an index buffer or a DDR fallback.
class SpmPermutationRowRunsRoute {
public:
    static SpmPermutationRowRunsRoute bind(
        SpmPortId source_id,
        const SpmDense2DSpec& source,
        SpmPortId destination_id,
        const SpmDense2DSpec& destination_storage,
        const std::vector<int32_t>& source_row_for_output,
        const std::vector<SpmDstRowRun>& destination_runs,
        const SpmPermutationRowRunsEvents& events);

    SpmPortId source_id() const { return source_id_; }
    SpmPortId destination_id() const { return destination_id_; }
    const SpmDense2DSpec& source_spec() const { return source_spec_; }
    const SpmDense2DSpec& destination_spec() const {
        return destination_spec_;
    }
    const std::vector<int32_t>& permutation() const { return permutation_; }
    const std::vector<SpmDstRowRun>& runs() const { return runs_; }
    const SpmPermutationRowRunsEvents& events() const { return events_; }
    uint64_t hash() const { return hash_; }

private:
    SpmPermutationRowRunsRoute(
        SpmPortId source_id,
        const SpmDense2DSpec& source_spec,
        SpmPortId destination_id,
        const SpmDense2DSpec& destination_spec,
        const std::vector<int32_t>& permutation,
        const std::vector<SpmDstRowRun>& runs,
        const SpmPermutationRowRunsEvents& events,
        uint64_t hash)
        : source_id_(source_id),
          destination_id_(destination_id),
          source_spec_(source_spec),
          destination_spec_(destination_spec),
          permutation_(permutation),
          runs_(runs),
          events_(events),
          hash_(hash) {}

    SpmPortId source_id_;
    SpmPortId destination_id_;
    SpmDense2DSpec source_spec_;
    SpmDense2DSpec destination_spec_;
    std::vector<int32_t> permutation_;
    std::vector<SpmDstRowRun> runs_;
    SpmPermutationRowRunsEvents events_;
    uint64_t hash_ = 0;
};

// One fixed byte range within a component arena.  offset is relative to the
// owning manifest's arena base.  first_event/last_event are inclusive.
struct SpmPhaseSlice {
    SpmPhaseSlice(SpmScratchId arena,
                  uint32_t offset,
                  size_t bytes,
                  SpmPipelineEvent first_event,
                  SpmPipelineEvent last_event)
        : arena(arena),
          offset(offset),
          bytes(bytes),
          first_event(first_event),
          last_event(last_event) {}

    SpmScratchId arena;
    uint32_t offset = 0;
    size_t bytes = 0;
    SpmPipelineEvent first_event;
    SpmPipelineEvent last_event;

    friend bool operator==(const SpmPhaseSlice& lhs,
                           const SpmPhaseSlice& rhs) {
        return lhs.arena == rhs.arena && lhs.offset == rhs.offset &&
               lhs.bytes == rhs.bytes &&
               lhs.first_event == rhs.first_event &&
               lhs.last_event == rhs.last_event;
    }
};

// A caller-supplied component layout at one fixed placement.  This public type
// is synthetic CPU-only: schema-v5 validates the declared live_ranges but
// cannot prove them complete.  Production admission requires a future opaque
// FusedModelBase-sealed exporter, not this type.  The overlay never remaps a
// component buffer.
struct SpmComponentPhaseManifest {
    SpmScratchId arena;
    uint32_t arena_base = 0;
    size_t arena_extent = 0;
    uint64_t layout_hash = 0;
    std::vector<SpmPhaseSlice> live_ranges;
};

enum class SpmFmbOccurrenceKind : uint8_t {
    LayerBody = 1,
    KvInsertBody = 2,
    PreLayers = 3,
    PostLayers = 4,
    PostFn = 5,
    ExternalFill = 6,
};

// Public selector only; it carries no allocation or address authority.  FMB
// must resolve it to exactly one private execution step before minting a
// consumer endpoint.
struct SpmFmbConsumerOccurrenceSelector {
    SpmFmbOccurrenceKind kind = SpmFmbOccurrenceKind::LayerBody;
    uint32_t body_id = 0;
    uint32_t group_id = 0;
    int layer = -1;
    int chunk_index = -1;
};

// One model-owned execution occurrence maps a local BufferDecl phase window
// into the pipeline-global event namespace.  The exporter applies it to every
// matching declaration; models do not hand-author ordinary byte ranges.
struct SpmFmbOccurrence {
    uint32_t key = 0;
    SpmFmbOccurrenceKind kind = SpmFmbOccurrenceKind::LayerBody;
    uint32_t body_id = 0;
    uint32_t group_id = 0;
    uint32_t chunk_ordinal = 0;
    uint32_t repetition_ordinal = 0;
    BufferScope scope;
    int layer = -1;
    int local_phase_begin = 0;
    int local_phase_end = 0;
    SpmPipelineEvent global_origin;
};

// An additive use outside a declaration's ordinary occurrence mapping.  It
// may append another checked local window but can never remove or shrink the
// ranges derived automatically from BufferDecl.
struct SpmFmbNamedUseExtension {
    std::string name;
    int layer = -1;
    uint32_t occurrence_key = 0;
    int local_phase_begin = 0;
    int local_phase_end = 0;
};

struct SpmFmbOccurrenceSchedule {
    uint64_t version = 0;
    std::vector<SpmFmbOccurrence> occurrences;
    std::vector<SpmFmbNamedUseExtension> use_extensions;
};

// Opaque FMB-owned seal.  There is no public constructor or range mutator.
// This CPU-only foundation records the full allocation snapshot and a weak
// owner-generation guard; physical admission remains disabled in this phase.
class SpmFmbSealedPhaseManifest {
public:
    size_t required_extent() const { return required_extent_; }
    uint64_t layout_hash() const { return layout_hash_; }
    uint64_t persistent_hash() const { return persistent_hash_; }
    uint64_t allocation_hash() const { return allocation_hash_; }
    uint64_t schedule_hash() const { return schedule_hash_; }
    uint64_t range_hash() const { return range_hash_; }
    size_t allocation_count() const { return allocation_count_; }
    size_t range_count() const { return manifest_.live_ranges.size(); }
    uint64_t identity_hash() const { return identity_hash_; }
    bool cpu_dry() const { return cpu_dry_; }

private:
    SpmFmbSealedPhaseManifest(
        const SpmComponentPhaseManifest& manifest,
        size_t required_extent,
        uint64_t layout_hash,
        uint64_t declaration_hash,
        uint64_t persistent_hash,
        uint64_t allocation_hash,
        uint64_t schedule_hash,
        uint64_t range_hash,
        size_t allocation_count,
        uint64_t identity_hash,
        bool cpu_dry,
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation)
        : manifest_(manifest),
          required_extent_(required_extent),
          layout_hash_(layout_hash),
          declaration_hash_(declaration_hash),
          persistent_hash_(persistent_hash),
          allocation_hash_(allocation_hash),
          schedule_hash_(schedule_hash),
          range_hash_(range_hash),
          allocation_count_(allocation_count),
          identity_hash_(identity_hash),
          cpu_dry_(cpu_dry),
          owner_generation_(owner_generation),
          expected_owner_generation_(expected_owner_generation) {}

    SpmComponentPhaseManifest manifest_;
    size_t required_extent_ = 0;
    uint64_t layout_hash_ = 0;
    uint64_t declaration_hash_ = 0;
    uint64_t persistent_hash_ = 0;
    uint64_t allocation_hash_ = 0;
    uint64_t schedule_hash_ = 0;
    uint64_t range_hash_ = 0;
    size_t allocation_count_ = 0;
    uint64_t identity_hash_ = 0;
    bool cpu_dry_ = true;
    std::weak_ptr<const uint64_t> owner_generation_;
    uint64_t expected_owner_generation_ = 0;

    friend class FusedModelBase;
    friend class SpmPipelinePlan;
};

using SpmFmbScopeMask = uint8_t;

constexpr SpmFmbScopeMask spm_fmb_scope_bit(BufferScope scope) {
    return static_cast<SpmFmbScopeMask>(
        SpmFmbScopeMask{1} << static_cast<uint8_t>(scope));
}

// Schema-v7 execution steps are resolved by FusedModelBase from the same
// traversal helper used by run_all_layers.  The public record has no authority
// by itself: only the opaque profile below can carry steps into sealing.
struct SpmFmbResolvedExecutionStep {
    SpmFmbResolvedExecutionStep(
        uint32_t key,
        SpmFmbOccurrenceKind kind,
        uint32_t body_id,
        uint32_t group_id,
        int layer,
        uint32_t traversal_ordinal,
        bool has_chunk,
        int chunk_index,
        int64_t chunk_offset,
        int64_t chunk_len,
        int64_t chunk_kv_seq_len,
        SpmFmbScopeMask scope_mask,
        bool updates_io_context,
        bool input_in_spm,
        bool output_to_spm,
        int local_phase_begin,
        int local_phase_end,
        SpmPipelineEvent global_origin)
        : key(key),
          kind(kind),
          body_id(body_id),
          group_id(group_id),
          layer(layer),
          traversal_ordinal(traversal_ordinal),
          has_chunk(has_chunk),
          chunk_index(chunk_index),
          chunk_offset(chunk_offset),
          chunk_len(chunk_len),
          chunk_kv_seq_len(chunk_kv_seq_len),
          scope_mask(scope_mask),
          updates_io_context(updates_io_context),
          input_in_spm(input_in_spm),
          output_to_spm(output_to_spm),
          local_phase_begin(local_phase_begin),
          local_phase_end(local_phase_end),
          global_origin(global_origin) {}

    uint32_t key = 0;
    SpmFmbOccurrenceKind kind = SpmFmbOccurrenceKind::LayerBody;
    uint32_t body_id = 0;
    uint32_t group_id = 0;
    int layer = -1;
    uint32_t traversal_ordinal = 0;
    bool has_chunk = false;
    int chunk_index = -1;
    int64_t chunk_offset = 0;
    int64_t chunk_len = 0;
    int64_t chunk_kv_seq_len = 0;
    SpmFmbScopeMask scope_mask = 0;
    bool updates_io_context = false;
    bool input_in_spm = false;
    bool output_to_spm = false;
    int local_phase_begin = 0;
    int local_phase_end = 0;
    SpmPipelineEvent global_origin;
};

// Opaque schema-v7 BUILD-topology profile.  There is no public constructor or
// step getter: callers can compare its sealed count/hash but cannot forge,
// delete, reorder, or rewrite the traversal required by FusedModelBase.
class SpmFmbResolvedExecutionProfile {
public:
    uint64_t profile_hash() const { return profile_hash_; }
    size_t step_count() const { return steps_.size(); }
    uint32_t build_body_iterations() const {
        return build_body_iterations_;
    }
    bool cpu_dry() const { return cpu_dry_; }

private:
    SpmFmbResolvedExecutionProfile(
        uint64_t request_version,
        uint32_t build_body_iterations,
        const std::vector<SpmFmbResolvedExecutionStep>& steps,
        uint64_t profile_hash,
        uint64_t expected_layout_hash,
        uint64_t expected_allocation_hash,
        SpmFmbPreloadCapability preload_capability,
        SpmFmbPostFnYieldCapability post_fn_yield_capability,
        uint64_t post_fn_yield_policy_fingerprint,
        SpmFmbLayerProducerYieldCapability layer_producer_yield_capability,
        uint64_t layer_producer_yield_policy_fingerprint,
        SpmFmbActiveGroupCapability active_group_capability,
        SpmFmbDenseDdrMemberCapability dense_ddr_member_capability,
        SpmFmbDmaEndpointCapability dma_endpoint_capability,
        SpmFmbRuntimeMemberCapability runtime_member_capability,
        SpmFmbSpmPeerCapability spm_peer_capability,
        uint64_t spm_peer_policy_fingerprint,
        uint8_t resolved_chunk_mode,
        uint8_t resolved_inter_layer_io,
        bool resolved_chunk_outer_within_group,
        int64_t resolved_cross_batch,
        int64_t resolved_hidden_size,
        int64_t resolved_num_layers,
        bool cpu_dry,
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation)
        : request_version_(request_version),
          build_body_iterations_(build_body_iterations),
          steps_(steps),
          profile_hash_(profile_hash),
          expected_layout_hash_(expected_layout_hash),
          expected_allocation_hash_(expected_allocation_hash),
          preload_capability_(preload_capability),
          post_fn_yield_capability_(post_fn_yield_capability),
          post_fn_yield_policy_fingerprint_(
              post_fn_yield_policy_fingerprint),
          layer_producer_yield_capability_(
              layer_producer_yield_capability),
          layer_producer_yield_policy_fingerprint_(
              layer_producer_yield_policy_fingerprint),
          active_group_capability_(active_group_capability),
          dense_ddr_member_capability_(dense_ddr_member_capability),
          dma_endpoint_capability_(dma_endpoint_capability),
          runtime_member_capability_(runtime_member_capability),
          spm_peer_capability_(spm_peer_capability),
          spm_peer_policy_fingerprint_(spm_peer_policy_fingerprint),
          resolved_chunk_mode_(resolved_chunk_mode),
          resolved_inter_layer_io_(resolved_inter_layer_io),
          resolved_chunk_outer_within_group_(
              resolved_chunk_outer_within_group),
          resolved_cross_batch_(resolved_cross_batch),
          resolved_hidden_size_(resolved_hidden_size),
          resolved_num_layers_(resolved_num_layers),
          cpu_dry_(cpu_dry),
          owner_generation_(owner_generation),
          expected_owner_generation_(expected_owner_generation) {}

    uint64_t request_version_ = 0;
    uint32_t build_body_iterations_ = 0;
    std::vector<SpmFmbResolvedExecutionStep> steps_;
    uint64_t profile_hash_ = 0;
    uint64_t expected_layout_hash_ = 0;
    uint64_t expected_allocation_hash_ = 0;
    SpmFmbPreloadCapability preload_capability_ =
        static_cast<SpmFmbPreloadCapability>(0);
    SpmFmbPostFnYieldCapability post_fn_yield_capability_ =
        static_cast<SpmFmbPostFnYieldCapability>(0);
    uint64_t post_fn_yield_policy_fingerprint_ = 0;
    SpmFmbLayerProducerYieldCapability layer_producer_yield_capability_ =
        static_cast<SpmFmbLayerProducerYieldCapability>(0);
    uint64_t layer_producer_yield_policy_fingerprint_ = 0;
    SpmFmbActiveGroupCapability active_group_capability_ =
        static_cast<SpmFmbActiveGroupCapability>(0);
    SpmFmbDenseDdrMemberCapability dense_ddr_member_capability_ =
        static_cast<SpmFmbDenseDdrMemberCapability>(0);
    SpmFmbDmaEndpointCapability dma_endpoint_capability_ =
        static_cast<SpmFmbDmaEndpointCapability>(0);
    SpmFmbRuntimeMemberCapability runtime_member_capability_ =
        static_cast<SpmFmbRuntimeMemberCapability>(0);
    SpmFmbSpmPeerCapability spm_peer_capability_ =
        static_cast<SpmFmbSpmPeerCapability>(0);
    uint64_t spm_peer_policy_fingerprint_ = 0;
    uint8_t resolved_chunk_mode_ = 0;
    uint8_t resolved_inter_layer_io_ = 0;
    bool resolved_chunk_outer_within_group_ = false;
    int64_t resolved_cross_batch_ = 0;
    int64_t resolved_hidden_size_ = 0;
    int64_t resolved_num_layers_ = 0;
    bool cpu_dry_ = true;
    std::weak_ptr<const uint64_t> owner_generation_;
    uint64_t expected_owner_generation_ = 0;

    friend class FusedModelBase;
};

// Opaque resolved manifest.  The original schema-v7 API mints CPU-dry tokens;
// a separate live API may mint the same allocation/range identity without
// granting plan or physical-admission authority by itself.
class SpmFmbResolvedPhaseManifest {
public:
    size_t required_extent() const { return required_extent_; }
    uint64_t layout_hash() const { return layout_hash_; }
    uint64_t persistent_hash() const { return persistent_hash_; }
    uint64_t allocation_hash() const { return allocation_hash_; }
    uint64_t profile_hash() const { return profile_hash_; }
    uint64_t range_hash() const { return range_hash_; }
    size_t allocation_count() const { return allocation_count_; }
    size_t range_count() const { return manifest_.live_ranges.size(); }
    uint64_t identity_hash() const { return identity_hash_; }
    bool cpu_dry() const { return cpu_dry_; }

private:
    SpmFmbResolvedPhaseManifest(
        const SpmComponentPhaseManifest& manifest,
        size_t required_extent,
        uint64_t layout_hash,
        uint64_t declaration_hash,
        uint64_t persistent_hash,
        uint64_t allocation_hash,
        uint64_t profile_hash,
        uint64_t range_hash,
        size_t allocation_count,
        uint64_t identity_hash,
        bool cpu_dry,
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation)
        : manifest_(manifest),
          required_extent_(required_extent),
          layout_hash_(layout_hash),
          declaration_hash_(declaration_hash),
          persistent_hash_(persistent_hash),
          allocation_hash_(allocation_hash),
          profile_hash_(profile_hash),
          range_hash_(range_hash),
          allocation_count_(allocation_count),
          identity_hash_(identity_hash),
          cpu_dry_(cpu_dry),
          owner_generation_(owner_generation),
          expected_owner_generation_(expected_owner_generation) {}

    SpmComponentPhaseManifest manifest_;
    size_t required_extent_ = 0;
    uint64_t layout_hash_ = 0;
    uint64_t declaration_hash_ = 0;
    uint64_t persistent_hash_ = 0;
    uint64_t allocation_hash_ = 0;
    uint64_t profile_hash_ = 0;
    uint64_t range_hash_ = 0;
    size_t allocation_count_ = 0;
    uint64_t identity_hash_ = 0;
    bool cpu_dry_ = true;
    std::weak_ptr<const uint64_t> owner_generation_;
    uint64_t expected_owner_generation_ = 0;

    friend class FusedModelBase;
    friend class SpmPipelinePlan;
    friend class SpmFmbDirectProducedPostFnBinding;
    friend class SpmFmbDirectProducedPostFnRunAddBinding;
    friend class SpmFmbCompositeYieldBinding;
    friend class SpmFmbCompositePhysicalReservation;
    friend class SpmFmbCompositePhysicalPlan;
    friend class SpmPipelineLease;
};

enum class SpmFmbBuildTraceAccessorKind : uint8_t {
    Addr = 1,
    LayerAddr = 2,
    AddrOffset = 3,
    LayerAddrOffset = 4,
};

enum class SpmFmbDdrArenaRole : uint8_t {
    LiveInput = 1,
    PingPongA = 2,
    PingPongB = 3,
    FinalOutput = 4,
};

enum class SpmFmbDdrAccess : uint8_t {
    Read = 1,
    Write = 2,
};

enum class SpmFmbDdrAddressMode : uint8_t {
    MutableBase = 1,
    Stable = 2,
};

// Schema-v8 is an opaque BUILD observation token.  It does not expose raw
// allocation records or Graph node windows and has no plan compiler or
// physical-admission authority.  FusedModelBase constructs it only after an
// explicitly armed canonical traversal has completed and the owning Graph has
// transitioned from RECORDING to BUILT.
class SpmFmbSealedBuildTrace {
public:
    uint64_t trace_hash() const { return trace_hash_; }
    uint64_t profile_hash() const { return profile_hash_; }
    uint64_t allocation_hash() const { return allocation_hash_; }
    uint64_t graph_build_generation() const {
        return graph_build_generation_;
    }
    uint64_t graph_signature_identity() const {
        return graph_signature_identity_;
    }
    uint64_t graph_topology_hash() const { return graph_topology_hash_; }
    uint16_t graph_node_kind_mask() const {
        return graph_node_kind_mask_;
    }
    size_t callback_count() const { return callback_count_; }
    size_t active_callback_count() const { return active_callback_count_; }
    size_t noop_callback_count() const { return noop_callback_count_; }
    size_t access_count() const { return access_count_; }
    bool cpu_dry() const { return cpu_dry_; }

private:
    struct AccessSummary {
        uint32_t allocation_ordinal = 0;
        uint32_t declaration_ordinal = 0;
        uint32_t alias_root_ordinal = 0;
        uint8_t storage = 0;
        uint8_t scope = 0;
        int layer = -1;
        SpmFmbBuildTraceAccessorKind accessor =
            SpmFmbBuildTraceAccessorKind::Addr;
        uint8_t core_mask = 0;
        uint32_t root_offset = 0;
        size_t protected_bytes = 0;
        uint32_t resolve_count = 0;
    };

    struct SemanticDmaSummary {
        size_t relative_node_index = 0;
        uint64_t endpoint_id = 0;
        uint8_t variant = 0;
        uint64_t src_addr = 0;
        uint64_t dst_addr = 0;
        size_t bytes = 0;
        uint8_t channel = 0;
        int64_t live_offset = 0;
        const uint64_t* live_base = nullptr;
        std::weak_ptr<const uint64_t> owner_generation;
        uint64_t expected_owner_generation = 0;
        bool canonical_member_emission = false;
        uint64_t spm_peer_id = 0;
    };

    struct SpmPeerSummary {
        size_t layer_group_ordinal = 0;
        size_t member_ordinal = 0;
        SpmFmbDmaEndpointRole role =
            static_cast<SpmFmbDmaEndpointRole>(0);
        uint8_t kind = 0;
        uint64_t endpoint_id = 0;
        uint64_t peer_id = 0;
        uint32_t allocation_ordinal = 0;
        uint32_t declaration_ordinal = 0;
        uint32_t alias_root_ordinal = 0;
        uint64_t declaration_name_hash = 0;
        uint64_t alias_root_name_hash = 0;
        uint8_t storage = 0;
        uint8_t scope = 0;
        int layer = -1;
        int phase_start = 0;
        int phase_end = 0;
        uint32_t root_offset = 0;
        uint32_t allocation_offset_from_root = 0;
        size_t logical_capacity = 0;
        size_t aligned_capacity = 0;
        size_t peer_slice_begin = 0;
        size_t peer_slice_count = 0;
        uint8_t core_mask = 0;
        uint8_t expected_occurrence_count = 0;
        uint64_t semantic_hash = 0;
    };

    struct ProducerYieldSummary {
        size_t ordinal = 0;
        uint64_t semantic_id = 0;
        SpmFmbTerminalWriterKind writer_kind =
            static_cast<SpmFmbTerminalWriterKind>(0);
        SpmDense2DSpec spec;
        uint32_t allocation_ordinal = 0;
        uint32_t declaration_ordinal = 0;
        uint32_t alias_root_ordinal = 0;
        uint64_t declaration_name_hash = 0;
        uint64_t alias_root_name_hash = 0;
        uint32_t root_offset = 0;
        uint32_t allocation_offset_from_root = 0;
        size_t logical_capacity = 0;
        size_t aligned_capacity = 0;
        size_t graph_node_begin = 0;
        size_t graph_node_end = 0;
        uint16_t graph_node_kind_mask = 0;
        uint64_t graph_topology_hash = 0;
    };

    struct CallbackSummary {
        SpmFmbOccurrenceKind kind = SpmFmbOccurrenceKind::LayerBody;
        uint32_t key = 0;
        uint32_t body_id = 0;
        uint32_t group_id = 0;
        int layer = -1;
        uint32_t traversal_ordinal = 0;
        int chunk_index = -1;
        int64_t chunk_offset = 0;
        int64_t chunk_len = 0;
        int64_t chunk_kv_seq_len = 0;
        SpmFmbScopeMask scope_mask = 0;
        bool updates_io_context = false;
        bool input_in_spm = false;
        bool output_to_spm = false;
        int local_phase_begin = 0;
        int local_phase_end = 0;
        uint32_t global_origin = 0;
        size_t graph_node_begin = 0;
        size_t graph_node_end = 0;
        uint16_t graph_node_kind_mask = 0;
        uint64_t graph_topology_hash = 0;
        size_t access_count = 0;
        uint64_t semantic_hash = 0;
        size_t legacy_dma_count = 0;
        std::vector<AccessSummary> accesses;
        std::vector<SemanticDmaSummary> semantic_dmas;
        std::vector<SpmPeerSummary> spm_peers;
        std::vector<ProducerYieldSummary> producer_yields;
    };

    SpmFmbSealedBuildTrace(
        uint64_t trace_hash,
        uint64_t profile_hash,
        uint64_t allocation_hash,
        uint64_t graph_build_generation,
        uint64_t graph_signature_identity,
        uint64_t graph_signature_segment_key,
        uint64_t graph_topology_hash,
        uint16_t graph_node_kind_mask,
        size_t callback_count,
        size_t active_callback_count,
        size_t noop_callback_count,
        size_t access_count,
        size_t graph_node_begin,
        size_t graph_node_end,
        SpmFmbPostFnYieldCapability post_fn_yield_capability,
        uint64_t post_fn_yield_policy_fingerprint,
        SpmFmbLayerProducerYieldCapability layer_producer_yield_capability,
        uint64_t layer_producer_yield_policy_fingerprint,
        SpmFmbActiveGroupCapability active_group_capability,
        SpmFmbDenseDdrMemberCapability dense_ddr_member_capability,
        SpmFmbDmaEndpointCapability dma_endpoint_capability,
        SpmFmbRuntimeMemberCapability runtime_member_capability,
        SpmFmbSpmPeerCapability spm_peer_capability,
        uint64_t spm_peer_policy_fingerprint,
        uint32_t resolved_build_body_iterations,
        uint8_t resolved_chunk_mode,
        uint8_t resolved_inter_layer_io,
        bool resolved_chunk_outer_within_group,
        int64_t resolved_cross_batch,
        int64_t resolved_hidden_size,
        int64_t resolved_num_layers,
        const std::vector<CallbackSummary>& callback_summaries,
        bool cpu_dry,
        const void* graph_identity,
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation,
        bool composite,
        uint64_t composite_identity,
        size_t composite_occurrence_ordinal,
        bool composite_producer,
        size_t composite_producer_local_ordinal,
        SpmFmbCompositeOccurrenceCapability composite_capability,
        uint64_t composite_policy_fingerprint,
        uint64_t composite_occurrence_profile_hash,
        bool composite_preload_follower,
        size_t composite_outer_begin)
        : trace_hash_(trace_hash),
          profile_hash_(profile_hash),
          allocation_hash_(allocation_hash),
          graph_build_generation_(graph_build_generation),
          graph_signature_identity_(graph_signature_identity),
          graph_signature_segment_key_(graph_signature_segment_key),
          graph_topology_hash_(graph_topology_hash),
          graph_node_kind_mask_(graph_node_kind_mask),
          callback_count_(callback_count),
          active_callback_count_(active_callback_count),
          noop_callback_count_(noop_callback_count),
          access_count_(access_count),
          graph_node_begin_(graph_node_begin),
          graph_node_end_(graph_node_end),
          post_fn_yield_capability_(post_fn_yield_capability),
          post_fn_yield_policy_fingerprint_(
              post_fn_yield_policy_fingerprint),
          layer_producer_yield_capability_(
              layer_producer_yield_capability),
          layer_producer_yield_policy_fingerprint_(
              layer_producer_yield_policy_fingerprint),
          active_group_capability_(active_group_capability),
          dense_ddr_member_capability_(dense_ddr_member_capability),
          dma_endpoint_capability_(dma_endpoint_capability),
          runtime_member_capability_(runtime_member_capability),
          spm_peer_capability_(spm_peer_capability),
          spm_peer_policy_fingerprint_(spm_peer_policy_fingerprint),
          resolved_build_body_iterations_(resolved_build_body_iterations),
          resolved_chunk_mode_(resolved_chunk_mode),
          resolved_inter_layer_io_(resolved_inter_layer_io),
          resolved_chunk_outer_within_group_(
              resolved_chunk_outer_within_group),
          resolved_cross_batch_(resolved_cross_batch),
          resolved_hidden_size_(resolved_hidden_size),
          resolved_num_layers_(resolved_num_layers),
          callback_summaries_(callback_summaries),
          cpu_dry_(cpu_dry),
          graph_identity_(graph_identity),
          owner_generation_(owner_generation),
          expected_owner_generation_(expected_owner_generation),
          composite_(composite),
          composite_identity_(composite_identity),
          composite_occurrence_ordinal_(composite_occurrence_ordinal),
          composite_producer_(composite_producer),
          composite_producer_local_ordinal_(
              composite_producer_local_ordinal),
          composite_capability_(composite_capability),
          composite_policy_fingerprint_(
              composite_policy_fingerprint),
          composite_occurrence_profile_hash_(
              composite_occurrence_profile_hash),
          composite_preload_follower_(composite_preload_follower),
          composite_outer_begin_(composite_outer_begin) {}

    uint64_t trace_hash_ = 0;
    uint64_t profile_hash_ = 0;
    uint64_t allocation_hash_ = 0;
    uint64_t graph_build_generation_ = 0;
    uint64_t graph_signature_identity_ = 0;
    uint64_t graph_signature_segment_key_ = 0;
    uint64_t graph_topology_hash_ = 0;
    uint16_t graph_node_kind_mask_ = 0;
    size_t callback_count_ = 0;
    size_t active_callback_count_ = 0;
    size_t noop_callback_count_ = 0;
    size_t access_count_ = 0;
    size_t graph_node_begin_ = 0;
    size_t graph_node_end_ = 0;
    SpmFmbPostFnYieldCapability post_fn_yield_capability_ =
        static_cast<SpmFmbPostFnYieldCapability>(0);
    uint64_t post_fn_yield_policy_fingerprint_ = 0;
    SpmFmbLayerProducerYieldCapability layer_producer_yield_capability_ =
        static_cast<SpmFmbLayerProducerYieldCapability>(0);
    uint64_t layer_producer_yield_policy_fingerprint_ = 0;
    SpmFmbActiveGroupCapability active_group_capability_ =
        static_cast<SpmFmbActiveGroupCapability>(0);
    SpmFmbDenseDdrMemberCapability dense_ddr_member_capability_ =
        static_cast<SpmFmbDenseDdrMemberCapability>(0);
    SpmFmbDmaEndpointCapability dma_endpoint_capability_ =
        static_cast<SpmFmbDmaEndpointCapability>(0);
    SpmFmbRuntimeMemberCapability runtime_member_capability_ =
        static_cast<SpmFmbRuntimeMemberCapability>(0);
    SpmFmbSpmPeerCapability spm_peer_capability_ =
        static_cast<SpmFmbSpmPeerCapability>(0);
    uint64_t spm_peer_policy_fingerprint_ = 0;
    uint32_t resolved_build_body_iterations_ = 0;
    uint8_t resolved_chunk_mode_ = 0;
    uint8_t resolved_inter_layer_io_ = 0;
    bool resolved_chunk_outer_within_group_ = false;
    int64_t resolved_cross_batch_ = 0;
    int64_t resolved_hidden_size_ = 0;
    int64_t resolved_num_layers_ = 0;
    std::vector<CallbackSummary> callback_summaries_;
    bool cpu_dry_ = true;
    const void* graph_identity_ = nullptr;
    std::weak_ptr<uint64_t> owner_generation_;
    uint64_t expected_owner_generation_ = 0;
    bool composite_ = false;
    uint64_t composite_identity_ = 0;
    size_t composite_occurrence_ordinal_ = 0;
    bool composite_producer_ = false;
    size_t composite_producer_local_ordinal_ = 0;
    SpmFmbCompositeOccurrenceCapability composite_capability_ =
        static_cast<SpmFmbCompositeOccurrenceCapability>(0);
    uint64_t composite_policy_fingerprint_ = 0;
    uint64_t composite_occurrence_profile_hash_ = 0;
    bool composite_preload_follower_ = false;
    size_t composite_outer_begin_ = 0;

    friend class FusedModelBase;
    friend class SpmCompositeTraceCoordinator;
    friend class SpmFmbCompositeYieldBinding;
    friend class SpmFmbCompositePhysicalPlan;
    friend class SpmFmbSealedSpmPeerBindings;
    friend class SpmFmbSealedPostFnYields;
    friend class SpmFmbSealedCallbackYields;
    friend class SpmFmbDirectProducedPostFnBinding;
    friend class SpmFmbDirectProducedPostFnRunAddBinding;
};

// Post-BUILD authority for the exact semantic terminal writers emitted by one
// resolved PostFn callback.  It retains Graph/owner stale guards and exposes no
// node index, allocation, address, or target conversion.
class SpmFmbSealedPostFnYields {
public:
    uint64_t yield_hash() const { return yield_hash_; }
    uint64_t source_trace_hash() const {
        return source_trace_.trace_hash();
    }
    size_t yield_count() const { return yields_.size(); }
    bool cpu_dry() const { return source_trace_.cpu_dry(); }

private:
    struct YieldSummary {
        size_t ordinal = 0;
        uint64_t semantic_id = 0;
        SpmFmbTerminalWriterKind writer_kind =
            static_cast<SpmFmbTerminalWriterKind>(0);
        SpmDense2DSpec spec;
        uint32_t allocation_ordinal = 0;
        uint32_t declaration_ordinal = 0;
        uint32_t alias_root_ordinal = 0;
        uint64_t declaration_name_hash = 0;
        uint64_t alias_root_name_hash = 0;
        uint32_t root_offset = 0;
        uint32_t allocation_offset_from_root = 0;
        size_t logical_capacity = 0;
        size_t aligned_capacity = 0;
        size_t relative_node_begin = 0;
        size_t relative_node_end = 0;
        uint64_t semantic_hash = 0;
    };

    SpmFmbSealedPostFnYields(
        const SpmFmbSealedBuildTrace& source_trace,
        uint64_t yield_hash,
        uint64_t semantic_id_digest,
        SpmFmbPostFnYieldCapability capability,
        uint64_t policy_fingerprint,
        const std::vector<YieldSummary>& yields)
        : source_trace_(source_trace),
          yield_hash_(yield_hash),
          semantic_id_digest_(semantic_id_digest),
          capability_(capability),
          policy_fingerprint_(policy_fingerprint),
          yields_(yields) {}

    SpmFmbSealedBuildTrace source_trace_;
    uint64_t yield_hash_ = 0;
    uint64_t semantic_id_digest_ = 0;
    SpmFmbPostFnYieldCapability capability_ =
        static_cast<SpmFmbPostFnYieldCapability>(0);
    uint64_t policy_fingerprint_ = 0;
    std::vector<YieldSummary> yields_;

    friend class FusedModelBase;
    friend class SpmFmbDirectProducedPostFnBinding;
    friend class SpmFmbDirectProducedPostFnRunAddBinding;
};

// Post-BUILD authority for sparse dense-FP16 terminal-writer windows inside
// canonical LayerBody callbacks and, when present, the terminal PostFn.  It is
// intentionally distinct from SpmFmbSealedPostFnYields so the legacy PostFn
// token keeps its exact whole-callback coverage contract and frozen identity.
class SpmFmbSealedCallbackYields {
public:
    uint64_t yield_hash() const { return yield_hash_; }
    uint64_t source_trace_hash() const {
        return source_trace_.trace_hash();
    }
    size_t yield_count() const { return yields_.size(); }
    bool cpu_dry() const { return source_trace_.cpu_dry(); }

private:
    struct YieldSummary {
        size_t ordinal = 0;
        size_t callback_index = 0;
        SpmFmbOccurrenceKind callback_kind =
            SpmFmbOccurrenceKind::LayerBody;
        uint32_t callback_key = 0;
        uint64_t semantic_id = 0;
        SpmFmbTerminalWriterKind writer_kind =
            static_cast<SpmFmbTerminalWriterKind>(0);
        SpmDense2DSpec spec;
        uint32_t allocation_ordinal = 0;
        uint32_t declaration_ordinal = 0;
        uint32_t alias_root_ordinal = 0;
        uint64_t declaration_name_hash = 0;
        uint64_t alias_root_name_hash = 0;
        uint32_t root_offset = 0;
        uint32_t allocation_offset_from_root = 0;
        size_t logical_capacity = 0;
        size_t aligned_capacity = 0;
        size_t relative_node_begin = 0;
        size_t relative_node_end = 0;
        uint64_t semantic_hash = 0;
    };

    SpmFmbSealedCallbackYields(
        const SpmFmbSealedBuildTrace& source_trace,
        uint64_t yield_hash,
        uint64_t semantic_id_digest,
        SpmFmbLayerProducerYieldCapability capability,
        uint64_t policy_fingerprint,
        const std::vector<YieldSummary>& yields)
        : source_trace_(source_trace),
          yield_hash_(yield_hash),
          semantic_id_digest_(semantic_id_digest),
          capability_(capability),
          policy_fingerprint_(policy_fingerprint),
          yields_(yields) {}

    SpmFmbSealedBuildTrace source_trace_;
    uint64_t yield_hash_ = 0;
    uint64_t semantic_id_digest_ = 0;
    SpmFmbLayerProducerYieldCapability capability_ =
        static_cast<SpmFmbLayerProducerYieldCapability>(0);
    uint64_t policy_fingerprint_ = 0;
    std::vector<YieldSummary> yields_;

    friend class FusedModelBase;
    friend class SpmCompositeTraceCoordinator;
    friend class SpmFmbCompositeYieldBinding;
};

// Move-only RECORDING result for one canonical FMB traversal inside a larger
// signed outer Graph.  Completion releases the thread-local FMB trace slot so
// the next occurrence may record before Graph::end(); no sealed Graph or
// physical authority is granted until the coordinator batch-seals every
// occurrence after the outer Graph reaches BUILT.
class SpmFmbCompletedBuildTrace {
public:
    ~SpmFmbCompletedBuildTrace();
    SpmFmbCompletedBuildTrace(SpmFmbCompletedBuildTrace&&) noexcept;
    SpmFmbCompletedBuildTrace& operator=(
        SpmFmbCompletedBuildTrace&&) noexcept;

    SpmFmbCompletedBuildTrace(const SpmFmbCompletedBuildTrace&) = delete;
    SpmFmbCompletedBuildTrace& operator=(
        const SpmFmbCompletedBuildTrace&) = delete;

    size_t occurrence_ordinal() const;
    uint64_t profile_hash() const;

private:
    struct Payload;
    explicit SpmFmbCompletedBuildTrace(std::unique_ptr<Payload> payload);

    std::unique_ptr<Payload> payload_;

    friend class FusedModelBase;
    friend class SpmCompositeTraceCoordinator;
};

// Immutable post-BUILD proof for an exact ordered component sequence.  It
// retains component owner/generation and Graph identities privately; callers
// can neither select an occurrence nor arm a component directly.
class SpmFmbSealedCompositeTrace {
public:
    ~SpmFmbSealedCompositeTrace();
    SpmFmbSealedCompositeTrace(const SpmFmbSealedCompositeTrace&) noexcept;
    SpmFmbSealedCompositeTrace& operator=(
        const SpmFmbSealedCompositeTrace&) noexcept;
    SpmFmbSealedCompositeTrace(SpmFmbSealedCompositeTrace&&) noexcept;
    SpmFmbSealedCompositeTrace& operator=(
        SpmFmbSealedCompositeTrace&&) noexcept;

    uint64_t composite_hash() const;
    size_t occurrence_count() const;
    size_t producer_yield_count() const;

private:
    struct Payload;
    explicit SpmFmbSealedCompositeTrace(
        std::shared_ptr<const Payload> payload);

    std::shared_ptr<const Payload> payload_;

    friend class SpmCompositeTraceCoordinator;
    friend class SpmFmbCompositeYieldBinding;
    friend class SpmFmbCompositePhysicalPlan;
};

// Lexical occurrence proof.  BUILD finish detaches one complete FMB trace;
// REPLAY finish verifies that the exact retained component window was consumed.
// Destruction without finish poisons the active outer Graph and cancels the
// coordinator transaction.
class SpmCompositeOccurrenceScope {
public:
    ~SpmCompositeOccurrenceScope();
    SpmCompositeOccurrenceScope(SpmCompositeOccurrenceScope&&) noexcept;
    SpmCompositeOccurrenceScope& operator=(
        SpmCompositeOccurrenceScope&&) noexcept;

    SpmCompositeOccurrenceScope(const SpmCompositeOccurrenceScope&) = delete;
    SpmCompositeOccurrenceScope& operator=(
        const SpmCompositeOccurrenceScope&) = delete;

    void finish();

private:
    SpmCompositeOccurrenceScope(
        SpmCompositeTraceCoordinator* coordinator,
        std::weak_ptr<const uint8_t> coordinator_lifetime,
        uint64_t scope_epoch);

    SpmCompositeTraceCoordinator* coordinator_ = nullptr;
    std::weak_ptr<const uint8_t> coordinator_lifetime_;
    uint64_t scope_epoch_ = 0;
    bool active_ = true;

    friend class SpmCompositeTraceCoordinator;
};

// One-shot proof that a composite REPLAY is still at its exact sealed outer
// boundary and can be completed only after the Graph register census commits
// its own outer-fast ticket.  Dropping an uncommitted ticket cancels the
// coordinator transaction.
class SpmCompositeOuterFastReplayTicket final {
public:
    ~SpmCompositeOuterFastReplayTicket();
    SpmCompositeOuterFastReplayTicket(
        SpmCompositeOuterFastReplayTicket&&) noexcept;
    SpmCompositeOuterFastReplayTicket& operator=(
        SpmCompositeOuterFastReplayTicket&&) = delete;

    SpmCompositeOuterFastReplayTicket(
        const SpmCompositeOuterFastReplayTicket&) = delete;
    SpmCompositeOuterFastReplayTicket& operator=(
        const SpmCompositeOuterFastReplayTicket&) = delete;

private:
    SpmCompositeOuterFastReplayTicket(
        SpmCompositeTraceCoordinator* coordinator,
        std::weak_ptr<const uint8_t> coordinator_lifetime,
        uint64_t nonce,
        size_t graph_extent,
        size_t occurrence_count,
        size_t producer_count,
        size_t producer_yield_count);
    void release() noexcept;

    SpmCompositeTraceCoordinator* coordinator_ = nullptr;
    std::weak_ptr<const uint8_t> coordinator_lifetime_;
    uint64_t nonce_ = 0;
    size_t graph_extent_ = 0;
    size_t occurrence_count_ = 0;
    size_t producer_count_ = 0;
    size_t producer_yield_count_ = 0;

    friend class SpmCompositeTraceCoordinator;
};

// Model-neutral outer trace coordinator for repeated producer traversals and
// their consumer in one Graph.  The first implementation deliberately accepts
// one producer owner across all marked occurrences; a second producer owner or
// a component using the legacy canonical semantic-DMA trace fails closed.
// This is sufficient for Vision N256 x3 -> Text while preserving every legacy
// single-component API/hash byte stream.  An open BUILD/REPLAY transaction is
// thread-affine: cancel and destruction must run on the constructing thread.
// Violating that contract is fail-stop because another thread cannot safely
// clear the owner's thread-local FMB authorities.
class SpmCompositeTraceCoordinator {
public:
    SpmCompositeTraceCoordinator(
        size_t expected_occurrence_count,
        size_t expected_producer_yield_count,
        uint64_t policy_fingerprint);
    ~SpmCompositeTraceCoordinator();

    SpmCompositeTraceCoordinator(const SpmCompositeTraceCoordinator&) = delete;
    SpmCompositeTraceCoordinator& operator=(
        const SpmCompositeTraceCoordinator&) = delete;
    SpmCompositeTraceCoordinator(SpmCompositeTraceCoordinator&&) = delete;
    SpmCompositeTraceCoordinator& operator=(
        SpmCompositeTraceCoordinator&&) = delete;

    void begin_build(RpuKernelGraph& graph);
    void begin_build(RpuKernelGraph& graph,
                     const SpmPipelineLease& lease,
                     bool require_committed);
    void adopt_physical_component(FusedModelBase& owner,
                                  const SpmPipelineLease& lease,
                                  bool producer);
    void release_physical_components(const SpmPipelineLease& lease);
    SpmCompositeOccurrenceScope begin_build_occurrence(
        FusedModelBase& owner,
        const SpmFmbResolvedExecutionProfile& profile);
    SpmFmbSealedCompositeTrace seal_build(RpuKernelGraph& graph);

    void begin_replay(
        const SpmFmbSealedCompositeTrace& sealed,
        RpuKernelGraph& graph);
    void begin_replay(
        const SpmFmbSealedCompositeTrace& sealed,
        RpuKernelGraph& graph,
        const SpmPipelineLease& lease);
    SpmCompositeOccurrenceScope begin_replay_occurrence(
        FusedModelBase& owner);
    SpmCompositeOuterFastReplayTicket validate_outer_fast_replay();
    void commit_outer_fast_replay(
        SpmCompositeOuterFastReplayTicket&& ticket) noexcept;
    void finish_replay();
    void cancel() noexcept;

    size_t expected_occurrence_count() const;
    size_t completed_occurrence_count() const;

private:
    struct Impl;
    void finish_occurrence(uint64_t scope_epoch);
    void cancel_occurrence(uint64_t scope_epoch) noexcept;
    void cancel_outer_fast_replay_ticket(uint64_t nonce) noexcept;

    std::unique_ptr<Impl> pimpl_;

    friend class SpmCompositeOccurrenceScope;
    friend class SpmCompositeOuterFastReplayTicket;
};

// Schema-v9 logical active-group proof.  It privately retains the exact
// schema-v8 source trace, so ordinary REPLAY validation and stale/ABA rejection
// cannot be separated from the grouping claim.  It has no raw-member getter,
// plan compiler, route conversion, or physical-admission authority.
class SpmFmbSealedActiveGroups {
public:
    uint64_t group_hash() const { return group_hash_; }
    uint64_t source_trace_hash() const { return source_trace_.trace_hash(); }
    uint64_t profile_hash() const { return source_trace_.profile_hash(); }
    uint64_t graph_build_generation() const {
        return source_trace_.graph_build_generation();
    }
    size_t group_count() const { return group_count_; }
    size_t member_callback_count() const { return member_callback_count_; }
    size_t active_callback_count() const { return active_callback_count_; }
    size_t noop_callback_count() const { return noop_callback_count_; }
    size_t component_node_count() const { return component_node_count_; }
    bool cpu_dry() const { return source_trace_.cpu_dry(); }

private:
    struct GroupSummary {
        size_t first_callback = 0;
        size_t member_count = 0;
        uint64_t semantic_hash = 0;
    };

    SpmFmbSealedActiveGroups(
        const SpmFmbSealedBuildTrace& source_trace,
        uint64_t group_hash,
        size_t group_count,
        size_t member_callback_count,
        size_t active_callback_count,
        size_t noop_callback_count,
        size_t component_node_count,
        SpmFmbActiveGroupCapability capability,
        const std::vector<GroupSummary>& group_summaries)
        : source_trace_(source_trace),
          group_hash_(group_hash),
          group_count_(group_count),
          member_callback_count_(member_callback_count),
          active_callback_count_(active_callback_count),
          noop_callback_count_(noop_callback_count),
          component_node_count_(component_node_count),
          capability_(capability),
          group_summaries_(group_summaries) {}

    SpmFmbSealedBuildTrace source_trace_;
    uint64_t group_hash_ = 0;
    size_t group_count_ = 0;
    size_t member_callback_count_ = 0;
    size_t active_callback_count_ = 0;
    size_t noop_callback_count_ = 0;
    size_t component_node_count_ = 0;
    SpmFmbActiveGroupCapability capability_ =
        static_cast<SpmFmbActiveGroupCapability>(0);
    std::vector<GroupSummary> group_summaries_;

    friend class FusedModelBase;
};

// Schema-v10 pointer-free dense-DDR member universe.  This token proves the
// FMB-derived geometry and arena roles that a future typed DMA path must use;
// it does not claim that existing Graph DMA nodes used those endpoints.
class SpmFmbSealedDenseDdrMembers {
public:
    uint64_t semantic_hash() const { return semantic_hash_; }
    uint64_t source_group_hash() const { return source_groups_.group_hash(); }
    uint64_t profile_hash() const { return source_groups_.profile_hash(); }
    uint64_t graph_build_generation() const {
        return source_groups_.graph_build_generation();
    }
    size_t layer_group_count() const { return layer_group_count_; }
    size_t members_per_group() const { return members_per_group_; }
    size_t total_member_count() const { return member_summaries_.size(); }
    size_t row_bytes() const { return row_bytes_; }
    size_t rows_per_group() const { return rows_per_group_; }
    size_t dense_bytes_per_group() const { return dense_bytes_per_group_; }
    bool cpu_dry() const { return source_groups_.cpu_dry(); }

private:
    struct DdrSlice {
        SpmFmbDdrArenaRole arena = SpmFmbDdrArenaRole::LiveInput;
        SpmFmbDdrAccess access = SpmFmbDdrAccess::Read;
        SpmFmbDdrAddressMode address_mode =
            SpmFmbDdrAddressMode::Stable;
        size_t byte_begin = 0;
        size_t byte_count = 0;
    };

    struct MemberSummary {
        size_t layer_group_ordinal = 0;
        size_t member_ordinal = 0;
        size_t member_count = 0;
        size_t callback_index = 0;
        uint32_t callback_key = 0;
        uint32_t body_id = 0;
        uint32_t framework_group_id = 0;
        int layer = -1;
        int64_t packed_row_begin = 0;
        int64_t row_count = 0;
        int64_t local_position_begin = 0;
        int64_t local_kv_seq_len = 0;
        DdrSlice ingress;
        DdrSlice egress;
        uint64_t semantic_hash = 0;
    };

    SpmFmbSealedDenseDdrMembers(
        const SpmFmbSealedActiveGroups& source_groups,
        uint64_t semantic_hash,
        size_t layer_group_count,
        size_t members_per_group,
        size_t row_bytes,
        size_t rows_per_group,
        size_t dense_bytes_per_group,
        SpmFmbDenseDdrMemberCapability capability,
        const std::vector<MemberSummary>& member_summaries)
        : source_groups_(source_groups),
          semantic_hash_(semantic_hash),
          layer_group_count_(layer_group_count),
          members_per_group_(members_per_group),
          row_bytes_(row_bytes),
          rows_per_group_(rows_per_group),
          dense_bytes_per_group_(dense_bytes_per_group),
          capability_(capability),
          member_summaries_(member_summaries) {}

    SpmFmbSealedActiveGroups source_groups_;
    uint64_t semantic_hash_ = 0;
    size_t layer_group_count_ = 0;
    size_t members_per_group_ = 0;
    size_t row_bytes_ = 0;
    size_t rows_per_group_ = 0;
    size_t dense_bytes_per_group_ = 0;
    SpmFmbDenseDdrMemberCapability capability_ =
        static_cast<SpmFmbDenseDdrMemberCapability>(0);
    std::vector<MemberSummary> member_summaries_;

    friend class FusedModelBase;
};

// Schema-v11 opaque binding between v10 logical member roles and the ordered
// nonzero semantic DMA occurrences observed in the same Graph BUILD.  It
// retains v10 stale/ABA provenance and deliberately exposes neither endpoint
// IDs, callback windows, member ranges, channels, nor physical addresses.
class SpmFmbSealedDmaEndpointBindings {
public:
    uint64_t binding_hash() const { return binding_hash_; }
    uint64_t source_member_hash() const {
        return source_members_.semantic_hash();
    }
    uint64_t profile_hash() const { return source_members_.profile_hash(); }
    uint64_t graph_build_generation() const {
        return source_members_.graph_build_generation();
    }
    size_t logical_endpoint_count() const {
        return binding_summaries_.size();
    }
    size_t dma_node_count() const { return dma_node_count_; }
    size_t fixed_dma_node_count() const { return fixed_dma_node_count_; }
    size_t mutable_dma_node_count() const {
        return mutable_dma_node_count_;
    }
    size_t legacy_dma_node_count() const { return legacy_dma_node_count_; }
    bool cpu_dry() const { return source_members_.cpu_dry(); }

private:
    struct BindingSummary {
        size_t layer_group_ordinal = 0;
        size_t member_ordinal = 0;
        SpmFmbDmaEndpointRole role =
            static_cast<SpmFmbDmaEndpointRole>(0);
        uint64_t endpoint_id = 0;
        uint8_t variant = 0;
        size_t byte_begin = 0;
        size_t byte_count = 0;
        size_t occurrence_begin = 0;
        size_t occurrence_count = 0;
        uint16_t channel_mask = 0;
        uint64_t semantic_hash = 0;
    };

    SpmFmbSealedDmaEndpointBindings(
        const SpmFmbSealedDenseDdrMembers& source_members,
        uint64_t binding_hash,
        size_t dma_node_count,
        size_t fixed_dma_node_count,
        size_t mutable_dma_node_count,
        size_t legacy_dma_node_count,
        SpmFmbDmaEndpointCapability capability,
        const std::vector<BindingSummary>& binding_summaries)
        : source_members_(source_members),
          binding_hash_(binding_hash),
          dma_node_count_(dma_node_count),
          fixed_dma_node_count_(fixed_dma_node_count),
          mutable_dma_node_count_(mutable_dma_node_count),
          legacy_dma_node_count_(legacy_dma_node_count),
          capability_(capability),
          binding_summaries_(binding_summaries) {}

    SpmFmbSealedDenseDdrMembers source_members_;
    uint64_t binding_hash_ = 0;
    size_t dma_node_count_ = 0;
    size_t fixed_dma_node_count_ = 0;
    size_t mutable_dma_node_count_ = 0;
    size_t legacy_dma_node_count_ = 0;
    SpmFmbDmaEndpointCapability capability_ =
        static_cast<SpmFmbDmaEndpointCapability>(0);
    std::vector<BindingSummary> binding_summaries_;

    friend class FusedModelBase;
};

// Schema-v13 opaque binding between each v11 logical endpoint and the exact
// FMB-owned dense SPM allocation/slice selected by the schema-v12 wrapper.
// No raw address, name, range, peer ID, or plan/lease conversion is public.
class SpmFmbSealedSpmPeerBindings {
public:
    uint64_t binding_hash() const { return binding_hash_; }
    uint64_t source_dma_binding_hash() const {
        return source_dma_bindings_.binding_hash();
    }
    size_t logical_peer_count() const { return peer_summaries_.size(); }
    size_t shared_stage_peer_count() const {
        return shared_stage_peer_count_;
    }
    size_t packed_global_peer_count() const {
        return packed_global_peer_count_;
    }
    bool cpu_dry() const { return source_dma_bindings_.cpu_dry(); }

private:
    using PeerSummary = SpmFmbSealedBuildTrace::SpmPeerSummary;

    SpmFmbSealedSpmPeerBindings(
        const SpmFmbSealedDmaEndpointBindings& source_dma_bindings,
        uint64_t binding_hash,
        size_t shared_stage_peer_count,
        size_t packed_global_peer_count,
        SpmFmbSpmPeerCapability capability,
        const std::vector<PeerSummary>& peer_summaries)
        : source_dma_bindings_(source_dma_bindings),
          binding_hash_(binding_hash),
          shared_stage_peer_count_(shared_stage_peer_count),
          packed_global_peer_count_(packed_global_peer_count),
          capability_(capability),
          peer_summaries_(peer_summaries) {}

    SpmFmbSealedDmaEndpointBindings source_dma_bindings_;
    uint64_t binding_hash_ = 0;
    size_t shared_stage_peer_count_ = 0;
    size_t packed_global_peer_count_ = 0;
    SpmFmbSpmPeerCapability capability_ =
        static_cast<SpmFmbSpmPeerCapability>(0);
    std::vector<PeerSummary> peer_summaries_;

    friend class FusedModelBase;
};

struct SpmSourceChunkRef {
    SpmChunkKey key;
    SpmDense2DSpec spec;
    SpmPhaseSlice backing;
};

struct SpmBorrowedConsumerChunkRef {
    SpmChunkKey key;
    SpmDense2DSpec spec;
    SpmPhaseSlice backing;
};

// Retained payload contains only routed image rows for its consumer chunk; it
// is never a second full consumer storage allocation.
struct SpmRetainedPayloadChunkRef {
    SpmChunkKey key;
    SpmDense2DSpec spec;
    SpmPhaseSlice backing;
};

// Opaque consumer-owned authority for one dense row slice.  FusedModelBase
// mints it from a live allocation snapshot and a resolved occurrence.  No
// buffer name, byte range, SPM address, or plan conversion is public; a later
// producer-yield compiler must consume it before physical admission.
class SpmFmbConsumerRowSliceEndpoint {
public:
    uint64_t identity_hash() const { return identity_hash_; }
    const SpmDense2DSpec& storage_spec() const {
        return consumer_.spec;
    }
    int64_t row_begin() const { return row_begin_; }
    int64_t row_count() const { return row_count_; }

private:
    SpmFmbConsumerRowSliceEndpoint(
        const SpmFmbResolvedPhaseManifest& manifest,
        const SpmBorrowedConsumerChunkRef& consumer,
        int64_t row_begin,
        int64_t row_count,
        uint32_t allocation_ordinal,
        uint32_t declaration_ordinal,
        uint32_t alias_root_ordinal,
        uint64_t declaration_name_hash,
        uint64_t alias_root_name_hash,
        uint32_t allocation_offset_from_root,
        size_t root_aligned_capacity,
        size_t logical_capacity,
        size_t aligned_capacity,
        SpmFmbOccurrenceKind occurrence_kind,
        uint32_t occurrence_key,
        uint32_t body_id,
        uint32_t group_id,
        int occurrence_layer,
        uint32_t traversal_ordinal,
        int chunk_index,
        SpmPipelineEvent global_first_event,
        SpmPipelineEvent global_last_event,
        uint64_t identity_hash)
        : manifest_(manifest),
          consumer_(consumer),
          row_begin_(row_begin),
          row_count_(row_count),
          allocation_ordinal_(allocation_ordinal),
          declaration_ordinal_(declaration_ordinal),
          alias_root_ordinal_(alias_root_ordinal),
          declaration_name_hash_(declaration_name_hash),
          alias_root_name_hash_(alias_root_name_hash),
          allocation_offset_from_root_(allocation_offset_from_root),
          root_aligned_capacity_(root_aligned_capacity),
          logical_capacity_(logical_capacity),
          aligned_capacity_(aligned_capacity),
          occurrence_kind_(occurrence_kind),
          occurrence_key_(occurrence_key),
          body_id_(body_id),
          group_id_(group_id),
          occurrence_layer_(occurrence_layer),
          traversal_ordinal_(traversal_ordinal),
          chunk_index_(chunk_index),
          global_first_event_(global_first_event),
          global_last_event_(global_last_event),
          identity_hash_(identity_hash) {}

    // Allocation identity and lexical lifetime are intentionally distinct.
    // One SPM-resident root may have a different canonical event slice in each
    // layer callback while retaining the same owned storage.  Composite
    // RunAdd admission uses this predicate across taps, then separately checks
    // exact callback/event identity within each tap.
    bool shares_owned_root_storage_with(
        const SpmFmbConsumerRowSliceEndpoint& other) const {
        const SpmPhaseSlice& backing = consumer_.backing;
        const SpmPhaseSlice& other_backing = other.consumer_.backing;
        return consumer_.spec == other.consumer_.spec &&
               backing.arena == other_backing.arena &&
               backing.offset == other_backing.offset &&
               backing.bytes == other_backing.bytes &&
               allocation_ordinal_ == other.allocation_ordinal_ &&
               declaration_ordinal_ == other.declaration_ordinal_ &&
               alias_root_ordinal_ == other.alias_root_ordinal_ &&
               declaration_name_hash_ == other.declaration_name_hash_ &&
               alias_root_name_hash_ == other.alias_root_name_hash_ &&
               allocation_offset_from_root_ ==
                   other.allocation_offset_from_root_ &&
               root_aligned_capacity_ == other.root_aligned_capacity_ &&
               logical_capacity_ == other.logical_capacity_ &&
               aligned_capacity_ == other.aligned_capacity_;
    }

    SpmFmbResolvedPhaseManifest manifest_;
    SpmBorrowedConsumerChunkRef consumer_;
    int64_t row_begin_ = 0;
    int64_t row_count_ = 0;
    uint32_t allocation_ordinal_ = 0;
    uint32_t declaration_ordinal_ = 0;
    uint32_t alias_root_ordinal_ = 0;
    uint64_t declaration_name_hash_ = 0;
    uint64_t alias_root_name_hash_ = 0;
    uint32_t allocation_offset_from_root_ = 0;
    size_t root_aligned_capacity_ = 0;
    size_t logical_capacity_ = 0;
    size_t aligned_capacity_ = 0;
    SpmFmbOccurrenceKind occurrence_kind_ =
        SpmFmbOccurrenceKind::LayerBody;
    uint32_t occurrence_key_ = 0;
    uint32_t body_id_ = 0;
    uint32_t group_id_ = 0;
    int occurrence_layer_ = -1;
    uint32_t traversal_ordinal_ = 0;
    int chunk_index_ = -1;
    SpmPipelineEvent global_first_event_;
    SpmPipelineEvent global_last_event_;
    uint64_t identity_hash_ = 0;

    friend class FusedModelBase;
    friend class SpmPipelinePlan;
    friend class SpmFmbDirectProducedPostFnBinding;
    friend class SpmFmbDirectProducedPostFnRunAddBinding;
    friend class SpmFmbCompositeYieldBinding;
    friend class SpmFmbCompositePhysicalReservation;
    friend class SpmFmbCompositePhysicalPlan;
};

// Opaque output operand consumed only by a trusted existing kernel launcher.
// The historical name is retained for source compatibility; callback-window
// capability also uses this type from canonical LayerBody callbacks.  The
// model cannot inspect or convert the target into a numeric SPM address.
class SpmFmbPostFnYieldTarget {
public:
    SpmFmbPostFnYieldTarget(const SpmFmbPostFnYieldTarget&) = delete;
    SpmFmbPostFnYieldTarget& operator=(
        const SpmFmbPostFnYieldTarget&) = delete;
    SpmFmbPostFnYieldTarget(SpmFmbPostFnYieldTarget&&) = delete;
    SpmFmbPostFnYieldTarget& operator=(
        SpmFmbPostFnYieldTarget&&) = delete;

private:
    SpmFmbPostFnYieldTarget(
        uint32_t address,
        const SpmDense2DSpec& spec,
        uint64_t semantic_id,
        SpmFmbTerminalWriterKind writer_kind,
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation)
        : address_(address),
          spec_(spec),
          semantic_id_(semantic_id),
          writer_kind_(writer_kind),
          owner_generation_(owner_generation),
          expected_owner_generation_(expected_owner_generation) {}

    uint32_t address_ = 0;
    SpmDense2DSpec spec_;
    uint64_t semantic_id_ = 0;
    SpmFmbTerminalWriterKind writer_kind_ =
        static_cast<SpmFmbTerminalWriterKind>(0);
    std::weak_ptr<uint64_t> owner_generation_;
    uint64_t expected_owner_generation_ = 0;

    friend class FusedModelBase;
    friend class SpmFmbPostFnYieldScope;
    friend struct SpmFmbYieldTargetLauncherAccess;
};

// Move-only lexical proof that one producer yield exactly covers a contiguous
// Graph window.  Destruction without finish() poisons the candidate but never
// throws from the destructor.
class SpmFmbPostFnYieldScope {
public:
    SpmFmbPostFnYieldScope(const SpmFmbPostFnYieldScope&) = delete;
    SpmFmbPostFnYieldScope& operator=(
        const SpmFmbPostFnYieldScope&) = delete;
    SpmFmbPostFnYieldScope(SpmFmbPostFnYieldScope&& other) noexcept;
    SpmFmbPostFnYieldScope& operator=(
        SpmFmbPostFnYieldScope&& other) noexcept;
    ~SpmFmbPostFnYieldScope();

    const SpmFmbPostFnYieldTarget& target() const;
    void finish();

private:
    enum class Contract : uint8_t {
        PostFnExact = 1,
        CallbackWindow = 2,
    };

    SpmFmbPostFnYieldScope(
        FusedModelBase* owner,
        std::unique_ptr<SpmFmbPostFnYieldTarget> target,
        Contract contract)
        : owner_(owner), target_(std::move(target)), active_(true),
          contract_(contract) {}

    FusedModelBase* owner_ = nullptr;
    std::unique_ptr<SpmFmbPostFnYieldTarget> target_;
    bool active_ = false;
    Contract contract_ = Contract::PostFnExact;

    friend class FusedModelBase;
};

// Preferred generic vocabulary for new composite subsystems.  These aliases
// intentionally add no constructors or address accessors.
using SpmFmbDenseProducerYieldTarget = SpmFmbPostFnYieldTarget;
using SpmFmbDenseProducerYieldScope = SpmFmbPostFnYieldScope;

// The consumer operation is part of route identity. Overwrite copies the
// payload; RunAdd consumes both payload and destination and
// lowers later to the existing SPM eltwise-ADD launcher.
enum class SpmConsumerTransform : uint8_t {
    Overwrite = 1,
    RunAdd = 2,
};

// ConsumerPacked preserves the original one-payload-per-consumer layout.
// SourceConsumerSplit creates one payload for each non-empty
// (source chunk, consumer chunk) pair, allowing sequential image producers to
// remain independently addressable even when Text has one full-sequence chunk.
enum class SpmPayloadPartition : uint8_t {
    ConsumerPacked = 1,
    SourceConsumerSplit = 2,
};

// Maximal dense copies after lowering the strict global permutation into
// chunk-local source and packed-payload coordinates.
struct SpmSourcePayloadRowRun {
    SpmChunkKey source_chunk;
    int64_t source_row_begin = 0;
    SpmChunkKey payload_chunk;
    int64_t payload_row_begin = 0;
    int64_t row_count = 0;
    SpmPipelineEvent transfer_event;
};

// Maximal dense copies from one packed retained payload into its borrowed
// consumer chunk.  Global destination runs crossing a chunk boundary are split
// automatically into separate entries.
struct SpmPayloadConsumerRowRun {
    SpmChunkKey payload_chunk;
    int64_t payload_row_begin = 0;
    SpmChunkKey consumer_chunk;
    int64_t consumer_row_begin = 0;
    int64_t row_count = 0;
    SpmPipelineEvent transfer_event;
};

// Schema-v5 route.  Endpoint storage is borrowed from exact manifest slices;
// only explicitly placed retained payload slices are added by the route.
class SpmChunkedPermutationRoute {
public:
    static SpmChunkedPermutationRoute bind(
        SpmPortId source_id,
        SpmPortId destination_id,
        const std::vector<SpmSourceChunkRef>& source_chunks,
        const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks,
        const std::vector<SpmRetainedPayloadChunkRef>& payload_chunks,
        const std::vector<int32_t>& source_row_for_output,
        const std::vector<SpmDstRowRun>& destination_runs,
        uint64_t policy_version,
        SpmConsumerTransform consumer_transform =
            SpmConsumerTransform::Overwrite,
        SpmPayloadPartition payload_partition =
            SpmPayloadPartition::ConsumerPacked);

    SpmPortId source_id() const { return source_id_; }
    SpmPortId destination_id() const { return destination_id_; }
    const std::vector<SpmSourceChunkRef>& source_chunks() const {
        return source_chunks_;
    }
    const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks() const {
        return consumer_chunks_;
    }
    const std::vector<SpmRetainedPayloadChunkRef>& payload_chunks() const {
        return payload_chunks_;
    }
    const std::vector<int32_t>& permutation() const { return permutation_; }
    const std::vector<SpmDstRowRun>& destination_runs() const {
        return destination_runs_;
    }
    const std::vector<SpmSourcePayloadRowRun>& source_payload_runs() const {
        return source_payload_runs_;
    }
    const std::vector<SpmPayloadConsumerRowRun>&
    payload_consumer_runs() const {
        return payload_consumer_runs_;
    }
    uint64_t policy_version() const { return policy_version_; }
    SpmConsumerTransform consumer_transform() const {
        return consumer_transform_;
    }
    SpmPayloadPartition payload_partition() const {
        return payload_partition_;
    }
    uint64_t route_schema_version() const { return route_schema_version_; }
    uint64_t hash() const { return hash_; }

private:
    SpmChunkedPermutationRoute(
        SpmPortId source_id,
        SpmPortId destination_id,
        const std::vector<SpmSourceChunkRef>& source_chunks,
        const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks,
        const std::vector<SpmRetainedPayloadChunkRef>& payload_chunks,
        const std::vector<int32_t>& permutation,
        const std::vector<SpmDstRowRun>& destination_runs,
        const std::vector<SpmSourcePayloadRowRun>& source_payload_runs,
        const std::vector<SpmPayloadConsumerRowRun>& payload_consumer_runs,
        uint64_t policy_version,
        SpmConsumerTransform consumer_transform,
        SpmPayloadPartition payload_partition,
        uint64_t route_schema_version,
        uint64_t hash)
        : source_id_(source_id),
          destination_id_(destination_id),
          source_chunks_(source_chunks),
          consumer_chunks_(consumer_chunks),
          payload_chunks_(payload_chunks),
          permutation_(permutation),
          destination_runs_(destination_runs),
          source_payload_runs_(source_payload_runs),
          payload_consumer_runs_(payload_consumer_runs),
          policy_version_(policy_version),
          consumer_transform_(consumer_transform),
          payload_partition_(payload_partition),
          route_schema_version_(route_schema_version),
          hash_(hash) {}

    SpmPortId source_id_;
    SpmPortId destination_id_;
    std::vector<SpmSourceChunkRef> source_chunks_;
    std::vector<SpmBorrowedConsumerChunkRef> consumer_chunks_;
    std::vector<SpmRetainedPayloadChunkRef> payload_chunks_;
    std::vector<int32_t> permutation_;
    std::vector<SpmDstRowRun> destination_runs_;
    std::vector<SpmSourcePayloadRowRun> source_payload_runs_;
    std::vector<SpmPayloadConsumerRowRun> payload_consumer_runs_;
    uint64_t policy_version_ = 0;
    SpmConsumerTransform consumer_transform_ =
        SpmConsumerTransform::Overwrite;
    SpmPayloadPartition payload_partition_ =
        SpmPayloadPartition::ConsumerPacked;
    uint64_t route_schema_version_ = 0;
    uint64_t hash_ = 0;
};

// One logical producer invocation whose output address is supplied by a
// borrowed consumer row-run.  There is deliberately no source backing: the
// producer writes the final consumer storage directly at production_event.
struct SpmDirectProducedChunkRef {
    SpmChunkKey key;
    SpmDense2DSpec spec;
    SpmPipelineEvent production_event;
};

// One lowered direct target.  The producer writes a complete dense chunk into
// a contiguous row range of exactly one borrowed consumer chunk.  No payload,
// copy, scatter, or transform is implied.
struct SpmDirectProducedRowRun {
    SpmChunkKey producer_chunk;
    SpmChunkKey consumer_chunk;
    int64_t consumer_row_begin = 0;
    int64_t row_count = 0;
    SpmPipelineEvent production_event;
};

class SpmDirectProducedRowRunsRoute {
public:
    static SpmDirectProducedRowRunsRoute bind(
        SpmPortId producer_id,
        SpmPortId destination_id,
        const std::vector<SpmDirectProducedChunkRef>& produced_chunks,
        const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks,
        const std::vector<SpmDstRowRun>& destination_runs,
        uint64_t policy_version);

    SpmPortId producer_id() const { return producer_id_; }
    SpmPortId destination_id() const { return destination_id_; }
    const std::vector<SpmDirectProducedChunkRef>& produced_chunks() const {
        return produced_chunks_;
    }
    const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks() const {
        return consumer_chunks_;
    }
    const std::vector<SpmDstRowRun>& destination_runs() const {
        return destination_runs_;
    }
    const std::vector<SpmDirectProducedRowRun>& lowered_runs() const {
        return lowered_runs_;
    }
    uint64_t policy_version() const { return policy_version_; }
    uint64_t route_schema_version() const { return route_schema_version_; }
    uint64_t hash() const { return hash_; }

private:
    SpmDirectProducedRowRunsRoute(
        SpmPortId producer_id,
        SpmPortId destination_id,
        const std::vector<SpmDirectProducedChunkRef>& produced_chunks,
        const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks,
        const std::vector<SpmDstRowRun>& destination_runs,
        const std::vector<SpmDirectProducedRowRun>& lowered_runs,
        uint64_t policy_version,
        uint64_t route_schema_version,
        uint64_t hash)
        : producer_id_(producer_id),
          destination_id_(destination_id),
          produced_chunks_(produced_chunks),
          consumer_chunks_(consumer_chunks),
          destination_runs_(destination_runs),
          lowered_runs_(lowered_runs),
          policy_version_(policy_version),
          route_schema_version_(route_schema_version),
          hash_(hash) {}

    SpmPortId producer_id_;
    SpmPortId destination_id_;
    std::vector<SpmDirectProducedChunkRef> produced_chunks_;
    std::vector<SpmBorrowedConsumerChunkRef> consumer_chunks_;
    std::vector<SpmDstRowRun> destination_runs_;
    std::vector<SpmDirectProducedRowRun> lowered_runs_;
    uint64_t policy_version_ = 0;
    uint64_t route_schema_version_ = 0;
    uint64_t hash_ = 0;
};

// Logical-only binding between one sealed PostFn terminal writer and one live
// consumer row slice.  It cannot be converted into a route, plan, target view,
// or lease; the future physical compiler must consume it together with all
// component/composite authorities.
class SpmFmbDirectProducedPostFnBinding {
public:
    static SpmFmbDirectProducedPostFnBinding bind(
        const SpmFmbSealedPostFnYields& producer,
        size_t yield_ordinal,
        SpmChunkKey producer_chunk,
        const SpmFmbConsumerRowSliceEndpoint& consumer,
        uint64_t policy_version);

    uint64_t binding_hash() const { return binding_hash_; }
    size_t yield_ordinal() const { return yield_ordinal_; }
    uint64_t policy_version() const { return policy_version_; }

private:
    SpmFmbDirectProducedPostFnBinding(
        const SpmFmbSealedPostFnYields& producer,
        size_t yield_ordinal,
        SpmChunkKey producer_chunk,
        const SpmDense2DSpec& produced_spec,
        const SpmFmbConsumerRowSliceEndpoint& consumer,
        uint64_t policy_version,
        uint64_t binding_hash)
        : producer_(producer),
          yield_ordinal_(yield_ordinal),
          producer_chunk_(producer_chunk),
          produced_spec_(produced_spec),
          consumer_(consumer),
          policy_version_(policy_version),
          binding_hash_(binding_hash) {}

    SpmFmbSealedPostFnYields producer_;
    size_t yield_ordinal_ = 0;
    SpmChunkKey producer_chunk_{SpmPortId(0), 0};
    SpmDense2DSpec produced_spec_;
    SpmFmbConsumerRowSliceEndpoint consumer_;
    uint64_t policy_version_ = 0;
    uint64_t binding_hash_ = 0;

    friend class SpmPipelinePlan;
};

// Logical-only alpha=1 sibling for a terminal output retained as one compact
// payload until a consumer row-slice RunAdd.  The payload backing and all
// composite timeline events are intentionally absent: only a later opaque
// physical compiler may derive them from a sealed outer schedule.
class SpmFmbDirectProducedPostFnRunAddBinding {
public:
    static SpmFmbDirectProducedPostFnRunAddBinding bind(
        const SpmFmbSealedPostFnYields& producer,
        size_t yield_ordinal,
        SpmChunkKey payload_chunk,
        const SpmFmbConsumerRowSliceEndpoint& consumer,
        uint64_t policy_version);

    uint64_t binding_hash() const { return binding_hash_; }
    size_t yield_ordinal() const { return yield_ordinal_; }
    uint64_t policy_version() const { return policy_version_; }
    int alpha() const { return 1; }

private:
    SpmFmbDirectProducedPostFnRunAddBinding(
        const SpmFmbSealedPostFnYields& producer,
        size_t yield_ordinal,
        SpmChunkKey payload_chunk,
        const SpmDense2DSpec& produced_spec,
        const SpmFmbConsumerRowSliceEndpoint& consumer,
        uint64_t policy_version,
        uint64_t binding_hash)
        : producer_(producer),
          yield_ordinal_(yield_ordinal),
          payload_chunk_(payload_chunk),
          produced_spec_(produced_spec),
          consumer_(consumer),
          policy_version_(policy_version),
          binding_hash_(binding_hash) {}

    SpmFmbSealedPostFnYields producer_;
    size_t yield_ordinal_ = 0;
    SpmChunkKey payload_chunk_{SpmPortId(0), 0};
    SpmDense2DSpec produced_spec_;
    SpmFmbConsumerRowSliceEndpoint consumer_;
    uint64_t policy_version_ = 0;
    uint64_t binding_hash_ = 0;

    friend class SpmPipelinePlan;
};

// Private stable summary for one producer-yield -> consumer-row endpoint in a
// sealed composite BUILD.  The summary retains the Graph-specific authority
// needed to reject foreign/stale use, but its stable hash deliberately excludes
// Graph identity, build generation, node windows, semantic marker IDs, and
// topology.  It has no public surface of its own.
class SpmFmbCompositeYieldBinding {
private:
    enum class Kind : uint8_t {
        DirectProduced = 1,
        RunAdd = 2,
    };

    static SpmFmbCompositeYieldBinding bind(
        const SpmFmbSealedCompositeTrace& composite,
        size_t producer_occurrence_ordinal,
        size_t producer_yield_ordinal,
        size_t consumer_occurrence_ordinal,
        const SpmFmbConsumerRowSliceEndpoint& consumer,
        uint64_t policy_version,
        Kind kind);

    SpmFmbCompositeYieldBinding(
        const SpmFmbSealedCompositeTrace& composite,
        const void* composite_authority,
        size_t composite_occurrence_count,
        size_t composite_producer_yield_count,
        uint64_t composite_policy_fingerprint,
        size_t producer_occurrence_ordinal,
        size_t producer_local_ordinal,
        size_t producer_yield_ordinal,
        size_t producer_occurrence_yield_count,
        size_t consumer_occurrence_ordinal,
        uint64_t producer_profile_hash,
        uint64_t producer_allocation_hash,
        uint64_t producer_occurrence_profile_hash,
        const std::shared_ptr<uint64_t>& producer_owner_generation,
        uint64_t expected_producer_owner_generation,
        uint64_t producer_yield_policy_fingerprint,
        size_t callback_index,
        SpmFmbOccurrenceKind callback_kind,
        uint32_t callback_key,
        uint32_t callback_body_id,
        uint32_t callback_group_id,
        int callback_layer,
        uint32_t callback_traversal_ordinal,
        int callback_chunk_index,
        int64_t callback_chunk_offset,
        int64_t callback_chunk_len,
        int64_t callback_chunk_kv_seq_len,
        int callback_local_phase_begin,
        int callback_local_phase_end,
        uint32_t callback_global_origin,
        uint64_t semantic_yield_id,
        SpmFmbTerminalWriterKind writer_kind,
        const SpmDense2DSpec& produced_spec,
        uint32_t allocation_ordinal,
        uint32_t declaration_ordinal,
        uint32_t alias_root_ordinal,
        uint64_t declaration_name_hash,
        uint64_t alias_root_name_hash,
        uint32_t root_offset,
        uint32_t allocation_offset_from_root,
        size_t logical_capacity,
        size_t aligned_capacity,
        const SpmFmbConsumerRowSliceEndpoint& consumer,
        uint64_t policy_version,
        Kind kind,
        uint64_t stable_hash)
        : composite_(composite),
          composite_authority_(composite_authority),
          composite_occurrence_count_(composite_occurrence_count),
          composite_producer_yield_count_(composite_producer_yield_count),
          composite_policy_fingerprint_(composite_policy_fingerprint),
          producer_occurrence_ordinal_(producer_occurrence_ordinal),
          producer_local_ordinal_(producer_local_ordinal),
          producer_yield_ordinal_(producer_yield_ordinal),
          producer_occurrence_yield_count_(producer_occurrence_yield_count),
          consumer_occurrence_ordinal_(consumer_occurrence_ordinal),
          producer_profile_hash_(producer_profile_hash),
          producer_allocation_hash_(producer_allocation_hash),
          producer_occurrence_profile_hash_(
              producer_occurrence_profile_hash),
          producer_owner_generation_(producer_owner_generation),
          expected_producer_owner_generation_(
              expected_producer_owner_generation),
          producer_yield_policy_fingerprint_(
              producer_yield_policy_fingerprint),
          callback_index_(callback_index),
          callback_kind_(callback_kind),
          callback_key_(callback_key),
          callback_body_id_(callback_body_id),
          callback_group_id_(callback_group_id),
          callback_layer_(callback_layer),
          callback_traversal_ordinal_(callback_traversal_ordinal),
          callback_chunk_index_(callback_chunk_index),
          callback_chunk_offset_(callback_chunk_offset),
          callback_chunk_len_(callback_chunk_len),
          callback_chunk_kv_seq_len_(callback_chunk_kv_seq_len),
          callback_local_phase_begin_(callback_local_phase_begin),
          callback_local_phase_end_(callback_local_phase_end),
          callback_global_origin_(callback_global_origin),
          semantic_yield_id_(semantic_yield_id),
          writer_kind_(writer_kind),
          produced_spec_(produced_spec),
          allocation_ordinal_(allocation_ordinal),
          declaration_ordinal_(declaration_ordinal),
          alias_root_ordinal_(alias_root_ordinal),
          declaration_name_hash_(declaration_name_hash),
          alias_root_name_hash_(alias_root_name_hash),
          root_offset_(root_offset),
          allocation_offset_from_root_(allocation_offset_from_root),
          logical_capacity_(logical_capacity),
          aligned_capacity_(aligned_capacity),
          consumer_(consumer),
          policy_version_(policy_version),
          kind_(kind),
          stable_hash_(stable_hash) {}

    void validate_live_authority(const void* expected_composite) const;

    SpmFmbSealedCompositeTrace composite_;
    const void* composite_authority_ = nullptr;
    size_t composite_occurrence_count_ = 0;
    size_t composite_producer_yield_count_ = 0;
    uint64_t composite_policy_fingerprint_ = 0;
    size_t producer_occurrence_ordinal_ = 0;
    size_t producer_local_ordinal_ = 0;
    size_t producer_yield_ordinal_ = 0;
    size_t producer_occurrence_yield_count_ = 0;
    size_t consumer_occurrence_ordinal_ = 0;
    uint64_t producer_profile_hash_ = 0;
    uint64_t producer_allocation_hash_ = 0;
    uint64_t producer_occurrence_profile_hash_ = 0;
    std::weak_ptr<uint64_t> producer_owner_generation_;
    uint64_t expected_producer_owner_generation_ = 0;
    uint64_t producer_yield_policy_fingerprint_ = 0;
    size_t callback_index_ = 0;
    SpmFmbOccurrenceKind callback_kind_ = SpmFmbOccurrenceKind::LayerBody;
    uint32_t callback_key_ = 0;
    uint32_t callback_body_id_ = 0;
    uint32_t callback_group_id_ = 0;
    int callback_layer_ = -1;
    uint32_t callback_traversal_ordinal_ = 0;
    int callback_chunk_index_ = -1;
    int64_t callback_chunk_offset_ = 0;
    int64_t callback_chunk_len_ = 0;
    int64_t callback_chunk_kv_seq_len_ = 0;
    int callback_local_phase_begin_ = 0;
    int callback_local_phase_end_ = 0;
    uint32_t callback_global_origin_ = 0;
    uint64_t semantic_yield_id_ = 0;
    SpmFmbTerminalWriterKind writer_kind_ =
        static_cast<SpmFmbTerminalWriterKind>(0);
    SpmDense2DSpec produced_spec_;
    uint32_t allocation_ordinal_ = 0;
    uint32_t declaration_ordinal_ = 0;
    uint32_t alias_root_ordinal_ = 0;
    uint64_t declaration_name_hash_ = 0;
    uint64_t alias_root_name_hash_ = 0;
    uint32_t root_offset_ = 0;
    uint32_t allocation_offset_from_root_ = 0;
    size_t logical_capacity_ = 0;
    size_t aligned_capacity_ = 0;
    SpmFmbConsumerRowSliceEndpoint consumer_;
    uint64_t policy_version_ = 0;
    Kind kind_ = Kind::DirectProduced;
    uint64_t stable_hash_ = 0;

    friend class SpmFmbCompositeDirectProducedBinding;
    friend class SpmFmbCompositeRunAddBinding;
    friend class SpmFmbCompositePhysicalPlan;
};

// Opaque direct-final-output binding.  Callers select only sealed semantic
// ordinals and a consumer-owned row endpoint; no event, backing, offset, or
// address can be supplied or recovered.
class SpmFmbCompositeDirectProducedBinding {
public:
    static SpmFmbCompositeDirectProducedBinding bind(
        const SpmFmbSealedCompositeTrace& composite,
        size_t producer_occurrence_ordinal,
        size_t producer_yield_ordinal,
        size_t consumer_occurrence_ordinal,
        const SpmFmbConsumerRowSliceEndpoint& consumer,
        uint64_t policy_version);

    uint64_t binding_hash() const { return binding_.stable_hash_; }
    size_t producer_occurrence_ordinal() const {
        return binding_.producer_occurrence_ordinal_;
    }
    size_t producer_yield_ordinal() const {
        return binding_.producer_yield_ordinal_;
    }
    size_t consumer_occurrence_ordinal() const {
        return binding_.consumer_occurrence_ordinal_;
    }
    uint64_t policy_version() const { return binding_.policy_version_; }

private:
    explicit SpmFmbCompositeDirectProducedBinding(
        SpmFmbCompositeYieldBinding binding)
        : binding_(std::move(binding)) {}

    SpmFmbCompositeYieldBinding binding_;

    friend class SpmFmbCompositePhysicalPlan;
};

// Opaque alpha=1 retained-payload sibling.  Payload placement and its RunAdd
// destination remain private to the composite physical compiler.
class SpmFmbCompositeRunAddBinding {
public:
    static SpmFmbCompositeRunAddBinding bind(
        const SpmFmbSealedCompositeTrace& composite,
        size_t producer_occurrence_ordinal,
        size_t producer_yield_ordinal,
        size_t consumer_occurrence_ordinal,
        const SpmFmbConsumerRowSliceEndpoint& consumer,
        uint64_t policy_version);

    uint64_t binding_hash() const { return binding_.stable_hash_; }
    size_t producer_occurrence_ordinal() const {
        return binding_.producer_occurrence_ordinal_;
    }
    size_t producer_yield_ordinal() const {
        return binding_.producer_yield_ordinal_;
    }
    size_t consumer_occurrence_ordinal() const {
        return binding_.consumer_occurrence_ordinal_;
    }
    uint64_t policy_version() const { return binding_.policy_version_; }
    static constexpr int alpha() { return 1; }

private:
    explicit SpmFmbCompositeRunAddBinding(
        SpmFmbCompositeYieldBinding binding)
        : binding_(std::move(binding)) {}

    SpmFmbCompositeYieldBinding binding_;

    friend class SpmFmbCompositePhysicalPlan;
};

// One retained payload whose storage is also the producer's final output.
// There is no source backing or source-to-payload copy: production_event is
// the first inclusive event of backing.
struct SpmDirectProducedPayloadRef {
    SpmChunkKey key;
    SpmDense2DSpec spec;
    SpmPhaseSlice backing;
    SpmPipelineEvent production_event;
};

// One complete payload consumed by dst += payload at add_event.  The payload
// always begins at row zero and maps wholly into one consumer chunk.
struct SpmDirectProducedPayloadRunAdd {
    SpmChunkKey payload_chunk;
    SpmChunkKey consumer_chunk;
    int64_t consumer_row_begin = 0;
    int64_t row_count = 0;
    SpmPipelineEvent production_event;
    SpmPipelineEvent add_event;
};

// Independent schema-v1 direct-produced-payload route.  It deliberately does
// not reuse nullable schema-v5 sources: every payload is produced in place,
// retained until its consumer starts, then consumed by alpha=1 RunAdd.
class SpmDirectProducedPayloadRunAddRoute {
public:
    static SpmDirectProducedPayloadRunAddRoute bind(
        SpmPortId payload_id,
        SpmPortId destination_id,
        const std::vector<SpmDirectProducedPayloadRef>& payloads,
        const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks,
        const std::vector<SpmDstRowRun>& destination_runs,
        uint64_t policy_version);

    SpmPortId payload_id() const { return payload_id_; }
    SpmPortId destination_id() const { return destination_id_; }
    const std::vector<SpmDirectProducedPayloadRef>& payloads() const {
        return payloads_;
    }
    const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks() const {
        return consumer_chunks_;
    }
    const std::vector<SpmDstRowRun>& destination_runs() const {
        return destination_runs_;
    }
    const std::vector<SpmDirectProducedPayloadRunAdd>& run_adds() const {
        return run_adds_;
    }
    static constexpr int alpha() { return 1; }
    uint64_t policy_version() const { return policy_version_; }
    uint64_t route_schema_version() const { return route_schema_version_; }
    uint64_t hash() const { return hash_; }

private:
    SpmDirectProducedPayloadRunAddRoute(
        SpmPortId payload_id,
        SpmPortId destination_id,
        const std::vector<SpmDirectProducedPayloadRef>& payloads,
        const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks,
        const std::vector<SpmDstRowRun>& destination_runs,
        const std::vector<SpmDirectProducedPayloadRunAdd>& run_adds,
        uint64_t policy_version,
        uint64_t route_schema_version,
        uint64_t hash)
        : payload_id_(payload_id),
          destination_id_(destination_id),
          payloads_(payloads),
          consumer_chunks_(consumer_chunks),
          destination_runs_(destination_runs),
          run_adds_(run_adds),
          policy_version_(policy_version),
          route_schema_version_(route_schema_version),
          hash_(hash) {}

    SpmPortId payload_id_;
    SpmPortId destination_id_;
    std::vector<SpmDirectProducedPayloadRef> payloads_;
    std::vector<SpmBorrowedConsumerChunkRef> consumer_chunks_;
    std::vector<SpmDstRowRun> destination_runs_;
    std::vector<SpmDirectProducedPayloadRunAdd> run_adds_;
    uint64_t policy_version_ = 0;
    uint64_t route_schema_version_ = 0;
    uint64_t hash_ = 0;
};

// CPU-only layout for a coarse, ordered multi-component SPM pipeline.  V1
// deliberately accepts only Temp + LayerWide BufferDecl regions; model-specific
// aliases, per-layer expansion, callbacks, and typed tensor layouts stay outside
// this foundation until a concrete handoff needs them.
class SpmPipelinePlan {
public:
    static SpmPipelinePlan compile(
        const std::vector<BufferDecl>& regions,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    static SpmPipelinePlan compile(
        const std::vector<SpmScratchDecl>& scratch_regions,
        const std::vector<SpmIdentityRoute>& identity_routes,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    static SpmPipelinePlan compile(
        const std::vector<SpmScratchDecl>& scratch_regions,
        const std::vector<SpmRowSliceRoute>& row_slice_routes,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    static SpmPipelinePlan compile(
        const std::vector<SpmScratchDecl>& scratch_regions,
        const std::vector<SpmPermutationRowRunsRoute>& permutation_routes,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    // Mixed typed composition for independent retained identity ports plus
    // existing permutation edges.  Port IDs remain plan-wide unique: this
    // overload deliberately does not admit a shared source, 1->N routing, or
    // an implicit RunAdd consumer.
    static SpmPipelinePlan compile(
        const std::vector<SpmScratchDecl>& scratch_regions,
        const std::vector<SpmIdentityRoute>& identity_routes,
        const std::vector<SpmPermutationRowRunsRoute>& permutation_routes,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    // Public phase manifests are logical, synthetic CPU-only admissions.
    // Their plans deliberately fail physical lease acquisition.
    static SpmPipelinePlan compile_phase_overlay(
        const std::vector<SpmComponentPhaseManifest>& manifests,
        const std::vector<SpmChunkedPermutationRoute>& chunked_routes,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    static SpmPipelinePlan compile_phase_overlay(
        const std::vector<SpmComponentPhaseManifest>& manifests,
        const std::vector<SpmChunkedPermutationRoute>& chunked_routes,
        const std::vector<SpmDirectProducedRowRunsRoute>& direct_routes,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    static SpmPipelinePlan compile_phase_overlay(
        const std::vector<SpmComponentPhaseManifest>& manifests,
        const std::vector<SpmChunkedPermutationRoute>& chunked_routes,
        const std::vector<SpmDirectProducedRowRunsRoute>& direct_routes,
        const std::vector<SpmDirectProducedPayloadRunAddRoute>&
            direct_payload_run_add_routes,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    static SpmPipelinePlan compile_fmb_sealed_phase_overlay(
        const std::vector<SpmFmbSealedPhaseManifest>& manifests,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    static SpmPipelinePlan compile_fmb_resolved_phase_overlay(
        const std::vector<SpmFmbResolvedPhaseManifest>& manifests,
        size_t reserved_top_bytes = 0,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);

    size_t peak_bytes() const { return peak_bytes_; }
    size_t reserved_top_bytes() const { return reserved_top_bytes_; }
    size_t total_bytes() const { return reserved_top_bytes_ + peak_bytes_; }
    size_t budget_bytes() const { return budget_bytes_; }
    size_t effective_budget_bytes() const {
        return budget_bytes_ & ~(SpmAllocator::ALIGN - 1);
    }
    uint64_t hash() const { return hash_; }
    size_t region_count() const { return regions_.size(); }
    bool contains(const std::string& name) const;
    bool contains(SpmScratchId id) const;
    bool contains(SpmPortId id) const;
    const SpmDense2DSpec& port_spec(SpmPortId id) const;
    uint64_t permutation_row_runs_hash(SpmPortId source_id,
                                       SpmPortId destination_id) const;
    size_t phase_manifest_count() const { return phase_manifests_.size(); }
    bool contains_phase_arena(SpmScratchId arena) const;
    uint64_t chunked_permutation_hash(SpmPortId source_id,
                                      SpmPortId destination_id) const;
    uint64_t direct_produced_row_runs_hash(SpmPortId producer_id,
                                           SpmPortId destination_id) const;
    uint64_t direct_produced_payload_run_add_hash(
        SpmPortId payload_id,
        SpmPortId destination_id) const;

private:
    struct Region {
        std::string name;
        uint32_t scratch_id = 0;
        uint32_t port_id = 0;
        size_t bytes = 0;
        int first_event = 0;
        int last_event = 0;
        uint32_t offset = 0;
        SpmDense2DSpec port_spec;
        bool row_slice_route = false;
        int64_t dst_row_begin = 0;
        int64_t dst_row_count = 0;
    };

    struct PhaseSealGuard {
        std::weak_ptr<const uint64_t> owner_generation;
        uint64_t expected_owner_generation = 0;
    };

    const Region& region(const std::string& name) const;
    const Region& region(SpmScratchId id) const;
    const Region& region(SpmPortId id) const;
    const SpmPermutationRowRunsRoute& permutation_route(
        SpmPortId source_id,
        SpmPortId destination_id) const;
    const SpmComponentPhaseManifest& phase_manifest(SpmScratchId arena) const;
    const SpmChunkedPermutationRoute& chunked_permutation_route(
        SpmPortId source_id,
        SpmPortId destination_id) const;
    const SpmDirectProducedRowRunsRoute& direct_produced_row_runs_route(
        SpmPortId producer_id,
        SpmPortId destination_id) const;
    const SpmDirectProducedPayloadRunAddRoute&
    direct_produced_payload_run_add_route(
        SpmPortId payload_id,
        SpmPortId destination_id) const;
    uint32_t phase_slice_absolute_offset(const SpmPhaseSlice& slice) const;
    void validate_phase_seals() const;

    std::vector<Region> regions_;
    std::vector<SpmPermutationRowRunsRoute> permutation_routes_;
    std::vector<SpmComponentPhaseManifest> phase_manifests_;
    std::vector<SpmChunkedPermutationRoute> chunked_permutation_routes_;
    std::vector<SpmDirectProducedRowRunsRoute>
        direct_produced_row_runs_routes_;
    std::vector<SpmDirectProducedPayloadRunAddRoute>
        direct_produced_payload_run_add_routes_;
    std::vector<PhaseSealGuard> phase_seal_guards_;
    size_t peak_bytes_ = 0;
    size_t reserved_top_bytes_ = 0;
    size_t budget_bytes_ = 0;
    uint64_t hash_ = 0;
    uint64_t phase_overlay_admission_ = 0;

    friend class SpmPipelineLease;
    friend class SpmFmbCompositePhysicalReservation;
    friend class SpmFmbCompositePhysicalPlan;
};

// Graph-independent physical placement for one repeated dense producer and
// one row-slice consumer.  It is compiled before the bootstrap BUILD from live
// component/consumer authority only.  It exposes aggregate capacity and count
// metadata, never an SPM offset, address, event, or backing allocation.
class SpmFmbCompositePhysicalReservation {
public:
    static SpmFmbCompositePhysicalReservation
    compile_repeated_dense_producer(
        const SpmFmbResolvedPhaseManifest& producer_manifest,
        const SpmFmbResolvedPhaseManifest& consumer_manifest,
        const std::vector<SpmFmbConsumerRowSliceEndpoint>&
            ordered_direct_endpoints,
        const std::vector<SpmFmbConsumerRowSliceEndpoint>&
            tap_major_run_add_endpoints,
        const SpmDense2DSpec& produced_spec,
        size_t yields_per_producer,
        uint64_t policy_version,
        size_t reserved_top_bytes,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);

    uint64_t hash() const { return plan_.hash(); }
    size_t dynamic_bytes() const { return plan_.peak_bytes(); }
    size_t reserved_top_bytes() const { return plan_.reserved_top_bytes(); }
    size_t total_bytes() const { return plan_.total_bytes(); }
    size_t effective_budget_bytes() const {
        return plan_.effective_budget_bytes();
    }
    size_t headroom_bytes() const {
        return effective_budget_bytes() - total_bytes();
    }
    size_t producer_occurrence_count() const {
        return producer_occurrence_count_;
    }
    size_t yields_per_producer() const { return yields_per_producer_; }
    size_t producer_yield_count() const {
        return producer_occurrence_count_ * yields_per_producer_;
    }
    size_t direct_binding_count() const { return direct_targets_.size(); }
    size_t run_add_binding_count() const { return run_add_targets_.size(); }
    size_t retained_payload_count() const { return run_add_targets_.size(); }
    uint64_t policy_version() const { return policy_version_; }

private:
    struct DirectTarget {
        uint64_t endpoint_hash = 0;
        size_t producer_local_ordinal = 0;
        size_t producer_yield_ordinal = 0;
        uint32_t target_relative_offset = 0;
        size_t bytes = 0;
    };
    struct RunAddTarget {
        uint64_t endpoint_hash = 0;
        size_t producer_local_ordinal = 0;
        size_t producer_yield_ordinal = 0;
        size_t payload_index = 0;
        uint32_t payload_relative_offset = 0;
        uint32_t consumer_relative_offset = 0;
        size_t bytes = 0;
        int consumer_layer = -1;
        int64_t consumer_row_begin = 0;
        int64_t consumer_row_count = 0;
        SpmDense2DSpec spec;
    };

    SpmFmbCompositePhysicalReservation(
        const SpmFmbResolvedPhaseManifest& producer_manifest,
        const SpmFmbResolvedPhaseManifest& consumer_manifest,
        SpmPipelinePlan plan,
        size_t producer_occurrence_count,
        size_t yields_per_producer,
        uint64_t policy_version,
        const SpmDense2DSpec& produced_spec,
        std::vector<DirectTarget> direct_targets,
        std::vector<RunAddTarget> run_add_targets)
        : producer_manifest_(producer_manifest),
          consumer_manifest_(consumer_manifest),
          plan_(std::move(plan)),
          producer_occurrence_count_(producer_occurrence_count),
          yields_per_producer_(yields_per_producer),
          policy_version_(policy_version),
          produced_spec_(produced_spec),
          direct_targets_(std::move(direct_targets)),
          run_add_targets_(std::move(run_add_targets)) {}

    void validate_live_authority() const;

    SpmFmbResolvedPhaseManifest producer_manifest_;
    SpmFmbResolvedPhaseManifest consumer_manifest_;
    SpmPipelinePlan plan_;
    size_t producer_occurrence_count_ = 0;
    size_t yields_per_producer_ = 0;
    uint64_t policy_version_ = 0;
    SpmDense2DSpec produced_spec_;
    std::vector<DirectTarget> direct_targets_;
    std::vector<RunAddTarget> run_add_targets_;

    friend class FusedModelBase;
    friend class SpmFmbCompositePhysicalPlan;
    friend class SpmPipelineLease;
};

// Opaque committed physical placement.  commit() consumes a sealed bootstrap
// composite only long enough to validate the exact callback-yield cover.  The
// returned plan retains the stable reservation hash and one scalar authority
// digest, not the bootstrap Graph/composite token.
class SpmFmbCompositePhysicalPlan {
public:
    static SpmFmbCompositePhysicalPlan compile(
        const SpmFmbSealedCompositeTrace& composite,
        const SpmFmbResolvedPhaseManifest& producer_manifest,
        const SpmFmbResolvedPhaseManifest& consumer_manifest,
        const std::vector<SpmFmbCompositeDirectProducedBinding>&
            direct_bindings,
        const std::vector<SpmFmbCompositeRunAddBinding>& run_add_bindings,
        size_t reserved_top_bytes,
        size_t budget_bytes = SpmAllocator::SPM_PLANNING_BUDGET);
    static SpmFmbCompositePhysicalPlan commit(
        const SpmFmbCompositePhysicalReservation& reservation,
        const SpmFmbSealedCompositeTrace& composite,
        const std::vector<SpmFmbCompositeDirectProducedBinding>&
            direct_bindings,
        const std::vector<SpmFmbCompositeRunAddBinding>& run_add_bindings);

    uint64_t hash() const { return plan_.hash(); }
    uint64_t authority_digest() const { return authority_digest_; }
    size_t dynamic_bytes() const { return plan_.peak_bytes(); }
    size_t reserved_top_bytes() const { return plan_.reserved_top_bytes(); }
    size_t total_bytes() const { return plan_.total_bytes(); }
    size_t effective_budget_bytes() const {
        return plan_.effective_budget_bytes();
    }
    size_t headroom_bytes() const {
        return effective_budget_bytes() - total_bytes();
    }
    size_t producer_occurrence_count() const {
        return producer_occurrence_count_;
    }
    size_t producer_yield_count() const { return producer_yield_count_; }
    size_t direct_binding_count() const { return direct_targets_.size(); }
    size_t run_add_binding_count() const { return run_add_targets_.size(); }
    size_t retained_payload_count() const { return run_add_targets_.size(); }
    bool has_direct_target(size_t producer_local_ordinal,
                           size_t producer_yield_ordinal) const;
    size_t retained_payload_index(size_t producer_local_ordinal,
                                  size_t producer_yield_ordinal) const;

private:
    struct DirectTarget {
        uint64_t binding_hash = 0;
        uint64_t endpoint_hash = 0;
        size_t producer_local_ordinal = 0;
        size_t producer_yield_ordinal = 0;
        uint32_t target_relative_offset = 0;
        size_t bytes = 0;
    };
    struct RunAddTarget {
        uint64_t binding_hash = 0;
        uint64_t endpoint_hash = 0;
        size_t producer_local_ordinal = 0;
        size_t producer_yield_ordinal = 0;
        size_t payload_index = 0;
        uint32_t payload_relative_offset = 0;
        uint32_t consumer_relative_offset = 0;
        size_t bytes = 0;
        int consumer_layer = -1;
        int64_t consumer_row_begin = 0;
        int64_t consumer_row_count = 0;
        SpmDense2DSpec spec;
    };

    SpmFmbCompositePhysicalPlan(
        const SpmFmbResolvedPhaseManifest& producer_manifest,
        const SpmFmbResolvedPhaseManifest& consumer_manifest,
        SpmPipelinePlan plan,
        size_t producer_occurrence_count,
        size_t producer_yield_count,
        const std::vector<DirectTarget>& direct_targets,
        const std::vector<RunAddTarget>& run_add_targets,
        uint64_t authority_digest)
        : producer_manifest_(producer_manifest),
          consumer_manifest_(consumer_manifest),
          plan_(std::move(plan)),
          producer_occurrence_count_(producer_occurrence_count),
          producer_yield_count_(producer_yield_count),
          direct_targets_(direct_targets),
          run_add_targets_(run_add_targets),
          authority_digest_(authority_digest) {}

    void validate_live_authority() const;

    SpmFmbResolvedPhaseManifest producer_manifest_;
    SpmFmbResolvedPhaseManifest consumer_manifest_;
    SpmPipelinePlan plan_;
    size_t producer_occurrence_count_ = 0;
    size_t producer_yield_count_ = 0;
    std::vector<DirectTarget> direct_targets_;
    std::vector<RunAddTarget> run_add_targets_;
    uint64_t authority_digest_ = 0;

    friend class FusedModelBase;
    friend class SpmPipelineLease;
};

// A checked reference to a byte range within one planned region.  The returned
// offset is relative to the future pipeline arena: this V1 object never reads or
// mutates the hardware allocator.
class SpmTensorView {
public:
    size_t size_bytes() const { return size_bytes_; }
    const std::string& region_name() const;

    uint32_t resolve_relative(const SpmPipelineLease& lease,
                              uint64_t current_allocator_generation) const;
    uint32_t resolve_physical_offset(const SpmPipelineLease& lease) const;
    uint32_t resolve_physical_addr(int core,
                                   const SpmPipelineLease& lease) const;

private:
    std::string region_name_;
    uint32_t scratch_id_ = 0;
    uint32_t port_id_ = 0;
    uint32_t relative_offset_ = 0;
    size_t size_bytes_ = 0;
    uint64_t plan_hash_ = 0;
    uint64_t lease_epoch_ = 0;

    friend class SpmPipelineLease;
    friend class SpmPortView;
};

// A typed checked view.  Its underlying byte view is private, so callers cannot
// accidentally bind a scratch range as a producer/consumer identity port.
class SpmPortView {
public:
    size_t size_bytes() const { return bytes_.size_bytes(); }
    const SpmDense2DSpec& spec() const { return spec_; }
    SpmPortView row_slice(int64_t row_begin, int64_t row_count) const;

    uint32_t resolve_relative(const SpmPipelineLease& lease,
                              uint64_t current_allocator_generation) const;
    uint32_t resolve_physical_offset(const SpmPipelineLease& lease) const;
    uint32_t resolve_physical_addr(int core,
                                   const SpmPipelineLease& lease) const;

private:
    SpmPortView() = default;

    SpmTensorView bytes_;
    SpmDense2DSpec spec_;
    bool row_slice_route_ = false;
    int64_t dst_row_begin_ = 0;
    int64_t dst_row_count_ = 0;

    friend class SpmPipelineLease;
    friend class SpmPermutationRowRunsView;
};

// Checked role-preserving views for one sealed permutation/row-runs route.  A
// caller cannot obtain this object by merely pairing arbitrary typed ports; the
// lease first resolves the exact source/destination pair recorded by the plan.
class SpmPermutationRowRunsView {
public:
    const SpmPortView& source() const { return source_; }
    const SpmPortView& destination() const { return destination_; }
    const std::vector<int32_t>& permutation() const { return permutation_; }
    const std::vector<SpmDstRowRun>& runs() const { return runs_; }
    uint64_t route_hash() const { return route_hash_; }
    std::vector<SpmContiguousRowRun> lower_contiguous_runs() const;

private:
    SpmPermutationRowRunsView() = default;

    SpmPortView source_;
    SpmPortView destination_;
    std::vector<int32_t> permutation_;
    std::vector<SpmDstRowRun> runs_;
    uint64_t route_hash_ = 0;

    friend class SpmPipelineLease;
};

// Checked view of one fixed schema-v5 chunk slice.  Construction is private so
// an arbitrary arena range cannot be reinterpreted as a sealed route endpoint.
class SpmPhaseChunkView {
public:
    const SpmChunkKey& key() const { return key_; }
    const SpmDense2DSpec& spec() const { return spec_; }
    const SpmPhaseSlice& backing() const { return backing_; }
    size_t size_bytes() const { return backing_.bytes; }

    uint32_t resolve_relative(const SpmPipelineLease& lease,
                              uint64_t current_allocator_generation) const;
    uint32_t resolve_physical_offset(const SpmPipelineLease& lease) const;
    uint32_t resolve_physical_addr(int core,
                                   const SpmPipelineLease& lease) const;

private:
    SpmPhaseChunkView(const SpmChunkKey& key,
                      const SpmDense2DSpec& spec,
                      const SpmPhaseSlice& backing,
                      uint32_t relative_offset,
                      uint64_t plan_hash,
                      uint64_t lease_epoch)
        : key_(key),
          spec_(spec),
          backing_(backing),
          relative_offset_(relative_offset),
          plan_hash_(plan_hash),
          lease_epoch_(lease_epoch) {}

    SpmChunkKey key_;
    SpmDense2DSpec spec_;
    SpmPhaseSlice backing_;
    uint32_t relative_offset_ = 0;
    uint64_t plan_hash_ = 0;
    uint64_t lease_epoch_ = 0;

    friend class SpmPipelineLease;
};

class SpmChunkedPermutationView {
public:
    const std::vector<SpmPhaseChunkView>& source_chunks() const {
        return source_chunks_;
    }
    const std::vector<SpmPhaseChunkView>& payload_chunks() const {
        return payload_chunks_;
    }
    const std::vector<SpmPhaseChunkView>& consumer_chunks() const {
        return consumer_chunks_;
    }
    const std::vector<int32_t>& permutation() const { return permutation_; }
    const std::vector<SpmDstRowRun>& destination_runs() const {
        return destination_runs_;
    }
    const std::vector<SpmSourcePayloadRowRun>& source_payload_runs() const {
        return source_payload_runs_;
    }
    const std::vector<SpmPayloadConsumerRowRun>&
    payload_consumer_runs() const {
        return payload_consumer_runs_;
    }
    uint64_t policy_version() const { return policy_version_; }
    SpmConsumerTransform consumer_transform() const {
        return consumer_transform_;
    }
    SpmPayloadPartition payload_partition() const {
        return payload_partition_;
    }
    uint64_t route_schema_version() const { return route_schema_version_; }
    uint64_t route_hash() const { return route_hash_; }

private:
    SpmChunkedPermutationView() = default;

    std::vector<SpmPhaseChunkView> source_chunks_;
    std::vector<SpmPhaseChunkView> payload_chunks_;
    std::vector<SpmPhaseChunkView> consumer_chunks_;
    std::vector<int32_t> permutation_;
    std::vector<SpmDstRowRun> destination_runs_;
    std::vector<SpmSourcePayloadRowRun> source_payload_runs_;
    std::vector<SpmPayloadConsumerRowRun> payload_consumer_runs_;
    uint64_t policy_version_ = 0;
    SpmConsumerTransform consumer_transform_ =
        SpmConsumerTransform::Overwrite;
    SpmPayloadPartition payload_partition_ =
        SpmPayloadPartition::ConsumerPacked;
    uint64_t route_schema_version_ = 0;
    uint64_t route_hash_ = 0;

    friend class SpmPipelineLease;
};

// Checked final-output address for one direct-produced dense chunk.  The
// address resolves inside the sealed consumer backing; callers never receive a
// forgeable source or payload allocation.
class SpmDirectProducedTargetView {
public:
    const SpmChunkKey& producer_chunk() const { return producer_chunk_; }
    const SpmChunkKey& consumer_chunk() const { return consumer_.key(); }
    const SpmDense2DSpec& spec() const { return spec_; }
    int64_t consumer_row_begin() const { return consumer_row_begin_; }
    SpmPipelineEvent production_event() const { return production_event_; }
    size_t size_bytes() const { return spec_.storage_bytes(); }

    uint32_t resolve_relative(const SpmPipelineLease& lease,
                              uint64_t current_allocator_generation) const;
    uint32_t resolve_physical_offset(const SpmPipelineLease& lease) const;
    uint32_t resolve_physical_addr(int core,
                                   const SpmPipelineLease& lease) const;

private:
    SpmDirectProducedTargetView(
        const SpmDirectProducedChunkRef& produced,
        const SpmPhaseChunkView& consumer,
        int64_t consumer_row_begin)
        : producer_chunk_(produced.key),
          spec_(produced.spec),
          production_event_(produced.production_event),
          consumer_(consumer),
          consumer_row_begin_(consumer_row_begin) {}

    uint32_t add_row_offset(uint32_t base) const;

    SpmChunkKey producer_chunk_;
    SpmDense2DSpec spec_;
    SpmPipelineEvent production_event_;
    SpmPhaseChunkView consumer_;
    int64_t consumer_row_begin_ = 0;

    friend class SpmPipelineLease;
};

class SpmDirectProducedRowRunsView {
public:
    const std::vector<SpmDirectProducedTargetView>& targets() const {
        return targets_;
    }
    const std::vector<SpmDstRowRun>& destination_runs() const {
        return destination_runs_;
    }
    uint64_t policy_version() const { return policy_version_; }
    uint64_t route_schema_version() const { return route_schema_version_; }
    uint64_t route_hash() const { return route_hash_; }

private:
    SpmDirectProducedRowRunsView() = default;

    std::vector<SpmDirectProducedTargetView> targets_;
    std::vector<SpmDstRowRun> destination_runs_;
    uint64_t policy_version_ = 0;
    uint64_t route_schema_version_ = 0;
    uint64_t route_hash_ = 0;

    friend class SpmPipelineLease;
};

// Checked schema-v15 view.  Both endpoint sets retain the ordinary plan,
// lease-epoch, and allocator-generation stale gates through SpmPhaseChunkView.
class SpmDirectProducedPayloadRunAddView {
public:
    const std::vector<SpmPhaseChunkView>& payloads() const {
        return payloads_;
    }
    const std::vector<SpmPhaseChunkView>& consumer_chunks() const {
        return consumer_chunks_;
    }
    const std::vector<SpmDirectProducedPayloadRunAdd>& run_adds() const {
        return run_adds_;
    }
    static constexpr int alpha() { return 1; }
    uint64_t policy_version() const { return policy_version_; }
    uint64_t route_schema_version() const { return route_schema_version_; }
    uint64_t route_hash() const { return route_hash_; }

private:
    SpmDirectProducedPayloadRunAddView() = default;

    std::vector<SpmPhaseChunkView> payloads_;
    std::vector<SpmPhaseChunkView> consumer_chunks_;
    std::vector<SpmDirectProducedPayloadRunAdd> run_adds_;
    uint64_t policy_version_ = 0;
    uint64_t route_schema_version_ = 0;
    uint64_t route_hash_ = 0;

    friend class SpmPipelineLease;
};

// Process-global logical ownership for one pipeline invocation.  It copies the
// sealed plan, so checked views do not depend on the caller's plan lifetime.
// Acquisition and resolution are CPU-only; allocator_generation is supplied by
// the caller and must still match at resolution time.
class SpmPipelineLease {
public:
    static SpmPipelineLease acquire(const SpmPipelinePlan& plan,
                                    uint64_t allocator_generation);
    static std::unique_ptr<SpmPipelineLease> acquire_physical(
        const SpmPipelinePlan& plan);
    static std::unique_ptr<SpmPipelineLease> acquire_physical(
        const SpmFmbCompositePhysicalReservation& reservation);
    static std::unique_ptr<SpmPipelineLease> acquire_physical(
        const SpmFmbCompositePhysicalPlan& plan);

    SpmPipelineLease(const SpmPipelineLease&) = delete;
    SpmPipelineLease& operator=(const SpmPipelineLease&) = delete;
    SpmPipelineLease(SpmPipelineLease&&) = delete;
    SpmPipelineLease& operator=(SpmPipelineLease&&) = delete;
    ~SpmPipelineLease();

    SpmTensorView view(const std::string& region_name) const;
    SpmTensorView view(const std::string& region_name,
                       size_t byte_offset,
                       size_t byte_count) const;
    SpmTensorView scratch(SpmScratchId id) const;
    SpmPortView port(SpmPortId id) const;
    SpmPermutationRowRunsView permutation_row_runs(
        SpmPortId source_id,
        SpmPortId destination_id) const;
    SpmChunkedPermutationView chunked_permutation(
        SpmPortId source_id,
        SpmPortId destination_id) const;
    SpmDirectProducedRowRunsView direct_produced_row_runs(
        SpmPortId producer_id,
        SpmPortId destination_id) const;
    SpmDirectProducedPayloadRunAddView direct_produced_payload_run_add(
        SpmPortId payload_id,
        SpmPortId destination_id) const;
    void release();
    void commit_composite(const SpmFmbCompositePhysicalPlan& plan);
    void seal_physical();
    RpuPhysicalArenaAuthority physical_graph_authority() const;
    void validate_physical_identity(uint64_t expected_epoch,
                                    uint64_t expected_plan_hash) const;
    void validate_physical(uint64_t expected_epoch,
                           uint64_t expected_plan_hash,
                           bool require_sealed = true) const;
    void validate_physical_range(uint32_t offset,
                                 size_t byte_count) const;
    bool active() const { return active_; }
    bool physical() const { return physical_; }
    bool physical_sealed() const { return physical_sealed_; }
    uint64_t epoch() const { return epoch_; }
    uint64_t plan_hash() const { return plan_.hash(); }
    uint64_t allocator_generation() const { return allocator_generation_; }
    size_t physical_arena_base() const { return physical_arena_base_; }
    size_t physical_arena_end() const { return physical_arena_end_; }
    size_t physical_top_reservation() const {
        return physical_top_reservation_;
    }

private:
    struct CompositeRunAddResolvedAddrs {
        uint32_t payload_addr = 0;
        uint32_t consumer_addr = 0;
        size_t bytes = 0;
        int consumer_layer = -1;
        int64_t consumer_row_begin = 0;
        int64_t consumer_row_count = 0;
        SpmDense2DSpec spec;
    };
    struct CompositeComponentPlacement {
        uint32_t relative_offset = 0;
        size_t bytes = 0;
        uint64_t layout_hash = 0;
        uint64_t allocation_hash = 0;
    };

    SpmPipelineLease(const SpmPipelinePlan& plan,
                     uint64_t allocator_generation,
                     uint64_t epoch,
                     std::thread::id owner_thread);
    SpmPipelineLease(
        const SpmPipelinePlan& plan,
        uint64_t allocator_generation,
        uint64_t epoch,
        std::thread::id owner_thread,
        RpuExecutionCoordinator::Claim&& physical_execution_claim,
        uint64_t physical_persistent_generation,
        size_t physical_arena_base,
        size_t physical_arena_end,
        size_t physical_top_reservation);

    void validate_live_owner(bool validate_phase_seals = true) const;
    void validate_physical_identity_impl(
        uint64_t expected_epoch,
        uint64_t expected_plan_hash,
        bool validate_phase_seals) const;
    void validate_physical_impl(
        uint64_t expected_epoch,
        uint64_t expected_plan_hash,
        bool require_sealed,
        bool validate_phase_seals) const;
    uint32_t resolve(const SpmTensorView& view,
                     uint64_t current_allocator_generation) const;
    uint32_t resolve_physical_offset(const SpmTensorView& view) const;
    uint32_t resolve_physical_addr(int core,
                                   const SpmTensorView& view) const;
    uint32_t resolve(const SpmPhaseChunkView& view,
                     uint64_t current_allocator_generation) const;
    uint32_t resolve_physical_offset(const SpmPhaseChunkView& view) const;
    uint32_t resolve_physical_addr(int core,
                                   const SpmPhaseChunkView& view) const;
    uint32_t resolve_composite_producer_target_addr(
        int core,
        size_t producer_local_ordinal,
        size_t producer_yield_ordinal) const;
    CompositeRunAddResolvedAddrs resolve_composite_run_add_addrs(
        int core,
        size_t producer_local_ordinal,
        size_t producer_yield_ordinal) const;
    CompositeComponentPlacement resolve_composite_component_placement(
        bool producer) const;
    void validate_composite_execution_authority(
        uint64_t expected_epoch,
        uint64_t expected_plan_hash,
        bool require_committed) const;
    size_t composite_producer_occurrence_count() const;
    size_t composite_yields_per_producer() const;
    size_t composite_run_add_count() const;
    bool composite_committed() const;
    // Kept private while existing FMB call sites migrate to the unified
    // producer/RunAdd resolvers above.
    uint32_t resolve_composite_direct_target_addr(
        int core,
        size_t producer_local_ordinal,
        size_t producer_yield_ordinal) const;
    uint32_t resolve_composite_run_add_payload_addr(
        int core,
        size_t producer_local_ordinal,
        size_t producer_yield_ordinal) const;
    uint32_t resolve_composite_run_add_consumer_addr(
        int core,
        size_t producer_local_ordinal,
        size_t producer_yield_ordinal) const;
    void release_noexcept() noexcept;
    static std::unique_ptr<SpmPipelineLease> acquire_physical_impl(
        const SpmPipelinePlan& plan);

    SpmPipelinePlan plan_;
    uint64_t allocator_generation_ = 0;
    uint64_t epoch_ = 0;
    std::thread::id owner_thread_;
    bool active_ = false;
    bool physical_ = false;
    bool physical_sealed_ = false;
    uint64_t physical_persistent_generation_ = 0;
    size_t physical_arena_base_ = 0;
    size_t physical_arena_end_ = 0;
    size_t physical_top_reservation_ = 0;
    std::shared_ptr<uint64_t> physical_lease_generation_;
    std::optional<RpuExecutionCoordinator::Claim>
        physical_execution_claim_;
    std::shared_ptr<const SpmFmbCompositePhysicalReservation>
        composite_reservation_;
    std::shared_ptr<const SpmFmbCompositePhysicalPlan> composite_plan_;

    friend class SpmTensorView;
    friend class SpmPhaseChunkView;
    friend class FusedModelBase;
    friend class SpmCompositeTraceCoordinator;
};

}  // namespace v3
