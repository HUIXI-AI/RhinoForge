#include "rpu_spm_pipeline.h"

#include "fused_model_base.h"
#include "graph/graph_runtime.h"

#include <c10/util/Exception.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <utility>

namespace v3 {
namespace {

constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;
constexpr uint64_t kPlanSchemaVersion = 1;
constexpr uint64_t kTypedPlanSchemaVersion = 2;
constexpr uint64_t kTypedRowSlicePlanSchemaVersion = 3;
constexpr uint64_t kTypedPermutationRowRunsPlanSchemaVersion = 4;
constexpr uint64_t kPhaseOverlayPlanSchemaVersion = 5;
constexpr uint64_t kPhaseOverlaySyntheticCpuAdmission = 1;
constexpr uint64_t kPhaseOverlayPhysicalFmbSealedAdmission = 2;
constexpr uint64_t kPhaseOverlayFmbSealedCpuDryAdmission = 3;
constexpr uint64_t kFmbSealedOverlayPlanSchemaVersion = 6;
constexpr uint64_t kPhaseOverlayFmbResolvedCpuDryAdmission = 4;
constexpr uint64_t kPhaseOverlayPhysicalCompositeAdmission = 5;
constexpr uint64_t kFmbResolvedOverlayPlanSchemaVersion = 7;
constexpr uint64_t kTypedMixedIdentityPermutationPlanSchemaVersion = 8;
constexpr uint64_t kPermutationRowRunsRouteSchemaVersion = 1;
constexpr uint64_t kChunkedPermutationRouteSchemaVersion = 1;
constexpr uint64_t kChunkedPermutationRouteSchemaVersionV2 = 2;
constexpr uint64_t kDirectProducedRowRunsRouteSchemaVersion = 1;
constexpr uint64_t kPhaseOverlayDirectProducedPlanSchemaVersion = 14;
constexpr uint64_t kDirectProducedPayloadRunAddRouteSchemaVersion = 1;
constexpr uint64_t kPhaseOverlayDirectProducedPayloadPlanSchemaVersion = 15;
constexpr uint64_t kFmbCompositePhysicalPlanSchemaVersion = 16;
constexpr uint64_t kFmbCompositePhysicalReservationSchemaVersion = 17;
constexpr uint64_t kFmbCompositePhysicalAuthorityDigestSchemaVersion = 18;
constexpr uint64_t kNonzeroHashFallback = 0x9e3779b97f4a7c15ULL;
constexpr size_t kFp16Bytes = 2;
constexpr uint64_t kTypedScratchRegionTag = 1;
constexpr uint64_t kTypedPortRegionTag = 2;
constexpr uint64_t kTypedRowSlicePortRegionTag = 3;
constexpr uint64_t kTypedPermutationSourceRegionTag = 4;
constexpr uint64_t kTypedPermutationDestinationRegionTag = 5;

uint64_t fnv_byte(uint64_t hash, uint8_t value) {
    return (hash ^ value) * kFnvPrime;
}

uint64_t fnv_u64(uint64_t hash, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        hash = fnv_byte(hash, static_cast<uint8_t>(value >> shift));
    }
    return hash;
}

uint64_t fnv_string(uint64_t hash, const std::string& value) {
    hash = fnv_u64(hash, value.size());
    for (unsigned char byte : value) {
        hash = fnv_byte(hash, byte);
    }
    return hash;
}

size_t align_up(size_t bytes) {
    TORCH_CHECK(bytes <= std::numeric_limits<size_t>::max() -
                             (SpmAllocator::ALIGN - 1),
                "SpmPipelinePlan: byte count overflows alignment");
    return (bytes + SpmAllocator::ALIGN - 1) &
           ~(SpmAllocator::ALIGN - 1);
}

uint64_t fnv_dense_spec(uint64_t hash, const SpmDense2DSpec& spec) {
    hash = fnv_u64(hash, static_cast<uint8_t>(spec.dtype));
    hash = fnv_u64(hash, static_cast<uint64_t>(spec.rows));
    hash = fnv_u64(hash, static_cast<uint64_t>(spec.cols));
    return fnv_u64(hash, static_cast<uint8_t>(spec.distribution));
}

uint64_t fnv_chunk_key(uint64_t hash, const SpmChunkKey& key) {
    hash = fnv_u64(hash, key.port().value());
    return fnv_u64(hash, key.ordinal());
}

uint64_t fnv_phase_slice(uint64_t hash, const SpmPhaseSlice& slice) {
    hash = fnv_u64(hash, slice.arena.value());
    hash = fnv_u64(hash, slice.offset);
    hash = fnv_u64(hash, slice.bytes);
    hash = fnv_u64(hash, slice.first_event.value());
    return fnv_u64(hash, slice.last_event.value());
}

void validate_phase_slice(const SpmPhaseSlice& slice,
                          SpmScratchId expected_arena,
                          size_t arena_extent,
                          const char* role) {
    TORCH_CHECK(slice.arena.value() != 0,
                role, ": arena ID zero is reserved");
    TORCH_CHECK(slice.arena == expected_arena,
                role, ": slice arena ", slice.arena.value(),
                " does not match manifest arena ", expected_arena.value());
    TORCH_CHECK(slice.bytes > 0,
                role, ": slice byte count must be positive");
    TORCH_CHECK(slice.offset % SpmAllocator::ALIGN == 0 &&
                    slice.bytes % SpmAllocator::ALIGN == 0,
                role, ": slice offset and byte count must be 256-byte "
                "aligned; got offset=", slice.offset,
                " bytes=", slice.bytes);
    TORCH_CHECK(slice.first_event.value() <= slice.last_event.value(),
                role, ": invalid inclusive event lifetime [",
                slice.first_event.value(), ", ", slice.last_event.value(),
                "]");
    TORCH_CHECK(static_cast<size_t>(slice.offset) <= arena_extent &&
                    slice.bytes <= arena_extent - slice.offset,
                role, ": slice [", slice.offset, ", ",
                static_cast<uint64_t>(slice.offset) + slice.bytes,
                ") exceeds arena extent ", arena_extent);
}

bool byte_ranges_overlap(uint64_t lhs_begin,
                         uint64_t lhs_end,
                         uint64_t rhs_begin,
                         uint64_t rhs_end) {
    return lhs_begin < rhs_end && rhs_begin < lhs_end;
}

bool event_ranges_overlap(const SpmPhaseSlice& lhs,
                          const SpmPhaseSlice& rhs) {
    return !(lhs.last_event < rhs.first_event ||
             rhs.last_event < lhs.first_event);
}

struct GlobalLeaseState {
    std::mutex mutex;
    bool active = false;
    uint64_t next_epoch = 0;
    uint64_t epoch = 0;
    uint64_t plan_hash = 0;
    uint64_t allocator_generation = 0;
    std::thread::id owner_thread;
};

GlobalLeaseState& global_lease_state() {
    static GlobalLeaseState state;
    return state;
}

}  // namespace

void SpmDense2DSpec::validate() const {
    TORCH_CHECK(dtype == SpmPortDType::Fp16,
                "SpmDense2DSpec: V2 accepts FP16 storage only");
    TORCH_CHECK(distribution == SpmPortDistribution::Replicated,
                "SpmDense2DSpec: V2 accepts replicated storage only");
    TORCH_CHECK(rows > 0 && cols > 0,
                "SpmDense2DSpec: rows and cols must be positive; got [",
                rows, ", ", cols, "]");
}

size_t SpmDense2DSpec::storage_bytes() const {
    validate();
    const size_t dense_rows = static_cast<size_t>(rows);
    const size_t dense_cols = static_cast<size_t>(cols);
    TORCH_CHECK(dense_rows <=
                    std::numeric_limits<size_t>::max() / dense_cols,
                "SpmDense2DSpec: element count overflows size_t");
    const size_t elements = dense_rows * dense_cols;
    TORCH_CHECK(elements <=
                    std::numeric_limits<size_t>::max() / kFp16Bytes,
                "SpmDense2DSpec: storage byte count overflows size_t");
    return elements * kFp16Bytes;
}

SpmIdentityRoute SpmIdentityRoute::bind(
    SpmPortId id,
    const SpmDense2DSpec& produced,
    const SpmDense2DSpec& required,
    int first_write,
    int last_read) {
    TORCH_CHECK(id.value() != 0,
                "SpmIdentityRoute: port ID zero is reserved");
    produced.validate();
    required.validate();
    TORCH_CHECK(produced == required,
                "SpmIdentityRoute: producer/consumer dense identity specs "
                "do not match; producer=[", produced.rows, ", ",
                produced.cols, "] consumer=[", required.rows, ", ",
                required.cols, "]");
    TORCH_CHECK(first_write >= 0 && first_write <= last_read,
                "SpmIdentityRoute: invalid lifetime [", first_write, ", ",
                last_read, "]");
    (void)produced.storage_bytes();
    return SpmIdentityRoute(id, produced, first_write, last_read);
}

SpmRowSliceRoute SpmRowSliceRoute::bind(
    SpmPortId id,
    const SpmDense2DSpec& produced,
    const SpmDense2DSpec& consumer_storage,
    int64_t dst_row_begin,
    int64_t dst_row_count,
    int first_write,
    int last_read) {
    TORCH_CHECK(id.value() != 0,
                "SpmRowSliceRoute: port ID zero is reserved");
    produced.validate();
    consumer_storage.validate();
    TORCH_CHECK(dst_row_begin >= 0,
                "SpmRowSliceRoute: destination row begin must be non-negative; "
                "got ", dst_row_begin);
    TORCH_CHECK(dst_row_count > 0,
                "SpmRowSliceRoute: destination row count must be positive; got ",
                dst_row_count);
    TORCH_CHECK(dst_row_begin <= consumer_storage.rows &&
                    dst_row_count <= consumer_storage.rows - dst_row_begin,
                "SpmRowSliceRoute: destination rows [", dst_row_begin, ", ",
                dst_row_begin, "+", dst_row_count,
                ") exceed consumer storage rows ", consumer_storage.rows);

    SpmDense2DSpec required = consumer_storage;
    required.rows = dst_row_count;
    TORCH_CHECK(produced == required,
                "SpmRowSliceRoute: producer spec does not match the consumer "
                "row slice; producer=[", produced.rows, ", ", produced.cols,
                "] required=[", required.rows, ", ", required.cols, "]");
    TORCH_CHECK(first_write >= 0 && first_write <= last_read,
                "SpmRowSliceRoute: invalid lifetime [", first_write, ", ",
                last_read, "]");
    (void)produced.storage_bytes();
    (void)consumer_storage.storage_bytes();
    return SpmRowSliceRoute(id, produced, consumer_storage, dst_row_begin,
                            dst_row_count, first_write, last_read);
}

SpmPermutationRowRunsRoute SpmPermutationRowRunsRoute::bind(
    SpmPortId source_id,
    const SpmDense2DSpec& source,
    SpmPortId destination_id,
    const SpmDense2DSpec& destination_storage,
    const std::vector<int32_t>& source_row_for_output,
    const std::vector<SpmDstRowRun>& destination_runs,
    const SpmPermutationRowRunsEvents& events) {
    TORCH_CHECK(source_id.value() != 0 && destination_id.value() != 0,
                "SpmPermutationRowRunsRoute: port ID zero is reserved");
    TORCH_CHECK(source_id != destination_id,
                "SpmPermutationRowRunsRoute: source and destination ports "
                "must be distinct");

    source.validate();
    destination_storage.validate();
    TORCH_CHECK(source.dtype == destination_storage.dtype &&
                    source.distribution == destination_storage.distribution &&
                    source.cols == destination_storage.cols,
                "SpmPermutationRowRunsRoute: source/destination dtype, "
                "distribution, and columns must match");
    TORCH_CHECK(source.cols % 16 == 0,
                "SpmPermutationRowRunsRoute: columns must be a multiple of 16; "
                "got ", source.cols);
    (void)source.storage_bytes();
    (void)destination_storage.storage_bytes();

    TORCH_CHECK(source.rows <= std::numeric_limits<int32_t>::max(),
                "SpmPermutationRowRunsRoute: source rows exceed int32 index range");
    TORCH_CHECK(source_row_for_output.size() ==
                    static_cast<size_t>(source.rows),
                "SpmPermutationRowRunsRoute: permutation length ",
                source_row_for_output.size(), " does not equal source rows ",
                source.rows);
    TORCH_CHECK(!source_row_for_output.empty(),
                "SpmPermutationRowRunsRoute: permutation must not be empty");

    std::vector<uint8_t> seen(source_row_for_output.size(), 0);
    for (int32_t row : source_row_for_output) {
        TORCH_CHECK(row >= 0 && row < source.rows,
                    "SpmPermutationRowRunsRoute: source row index ", row,
                    " is outside [0, ", source.rows, ")");
        TORCH_CHECK(seen[static_cast<size_t>(row)] == 0,
                    "SpmPermutationRowRunsRoute: source row index ", row,
                    " appears more than once; a strict permutation is required");
        seen[static_cast<size_t>(row)] = 1;
    }

    TORCH_CHECK(!destination_runs.empty(),
                "SpmPermutationRowRunsRoute: destination runs must not be empty");
    int64_t routed_rows = 0;
    int64_t previous_end = 0;
    bool first_run = true;
    for (const SpmDstRowRun& run : destination_runs) {
        TORCH_CHECK(run.dst_row_begin >= 0,
                    "SpmPermutationRowRunsRoute: destination run begin must be "
                    "non-negative; got ", run.dst_row_begin);
        TORCH_CHECK(run.row_count > 0,
                    "SpmPermutationRowRunsRoute: destination run count must be "
                    "positive; got ", run.row_count);
        TORCH_CHECK(run.dst_row_begin <= destination_storage.rows &&
                        run.row_count <=
                            destination_storage.rows - run.dst_row_begin,
                    "SpmPermutationRowRunsRoute: destination run [",
                    run.dst_row_begin, ", ", run.dst_row_begin, "+",
                    run.row_count, ") exceeds storage rows ",
                    destination_storage.rows);
        TORCH_CHECK(first_run || run.dst_row_begin >= previous_end,
                    "SpmPermutationRowRunsRoute: destination runs must be "
                    "ordered and non-overlapping");
        TORCH_CHECK(routed_rows <= source.rows &&
                        run.row_count <= source.rows - routed_rows,
                    "SpmPermutationRowRunsRoute: destination run row total "
                    "exceeds source rows ", source.rows);
        routed_rows += run.row_count;
        previous_end = run.dst_row_begin + run.row_count;
        first_run = false;
    }
    TORCH_CHECK(routed_rows == source.rows,
                "SpmPermutationRowRunsRoute: destination runs cover ",
                routed_rows, " rows but source has ", source.rows);

    TORCH_CHECK(events.source_first_write >= 0 &&
                    events.source_first_write <= events.transform,
                "SpmPermutationRowRunsRoute: invalid source lifetime [",
                events.source_first_write, ", ", events.transform, "]");
    TORCH_CHECK(events.destination_first_write >= 0 &&
                    events.destination_first_write <= events.transform &&
                    events.transform <= events.destination_last_read,
                "SpmPermutationRowRunsRoute: invalid destination/transform "
                "events [", events.destination_first_write, ", ",
                events.transform, ", ", events.destination_last_read, "]");

    uint64_t hash = fnv_u64(kFnvOffset,
                            kPermutationRowRunsRouteSchemaVersion);
    hash = fnv_u64(hash, source_id.value());
    hash = fnv_u64(hash, destination_id.value());
    hash = fnv_dense_spec(hash, source);
    hash = fnv_dense_spec(hash, destination_storage);
    hash = fnv_u64(hash, static_cast<uint64_t>(events.source_first_write));
    hash = fnv_u64(hash,
                   static_cast<uint64_t>(events.destination_first_write));
    hash = fnv_u64(hash, static_cast<uint64_t>(events.transform));
    hash = fnv_u64(hash,
                   static_cast<uint64_t>(events.destination_last_read));
    hash = fnv_u64(hash, source_row_for_output.size());
    for (int32_t row : source_row_for_output) {
        hash = fnv_u64(hash, static_cast<uint32_t>(row));
    }
    hash = fnv_u64(hash, destination_runs.size());
    for (const SpmDstRowRun& run : destination_runs) {
        hash = fnv_u64(hash, static_cast<uint64_t>(run.dst_row_begin));
        hash = fnv_u64(hash, static_cast<uint64_t>(run.row_count));
    }
    if (hash == 0) hash = kNonzeroHashFallback;

    return SpmPermutationRowRunsRoute(
        source_id, source, destination_id, destination_storage,
        source_row_for_output, destination_runs, events, hash);
}

SpmChunkedPermutationRoute SpmChunkedPermutationRoute::bind(
    SpmPortId source_id,
    SpmPortId destination_id,
    const std::vector<SpmSourceChunkRef>& source_chunks,
    const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks,
    const std::vector<SpmRetainedPayloadChunkRef>& payload_chunks,
    const std::vector<int32_t>& source_row_for_output,
    const std::vector<SpmDstRowRun>& destination_runs,
    uint64_t policy_version,
    SpmConsumerTransform consumer_transform,
    SpmPayloadPartition payload_partition) {
    TORCH_CHECK(source_id.value() != 0 && destination_id.value() != 0,
                "SpmChunkedPermutationRoute: port ID zero is reserved");
    TORCH_CHECK(source_id != destination_id,
                "SpmChunkedPermutationRoute: source and destination ports "
                "must be distinct");
    TORCH_CHECK(policy_version != 0,
                "SpmChunkedPermutationRoute: policy version zero is reserved");
    TORCH_CHECK(!source_chunks.empty(),
                "SpmChunkedPermutationRoute: source chunks must not be empty");
    TORCH_CHECK(!consumer_chunks.empty(),
                "SpmChunkedPermutationRoute: consumer chunks must not be empty");
    TORCH_CHECK(
        consumer_transform == SpmConsumerTransform::Overwrite ||
            consumer_transform == SpmConsumerTransform::RunAdd,
        "SpmChunkedPermutationRoute: unsupported consumer transform");
    TORCH_CHECK(
        payload_partition == SpmPayloadPartition::ConsumerPacked ||
            payload_partition ==
                SpmPayloadPartition::SourceConsumerSplit,
        "SpmChunkedPermutationRoute: unsupported payload partition");
    if (payload_partition == SpmPayloadPartition::ConsumerPacked) {
        TORCH_CHECK(payload_chunks.size() == consumer_chunks.size(),
                    "SpmChunkedPermutationRoute: one retained payload chunk is "
                    "required for each consumer chunk");
    } else {
        TORCH_CHECK(!payload_chunks.empty(),
                    "SpmChunkedPermutationRoute: split payload chunks must "
                    "not be empty");
    }

    auto validate_backing = [](const SpmDense2DSpec& spec,
                               const SpmPhaseSlice& backing,
                               const char* role) {
        spec.validate();
        TORCH_CHECK(backing.arena.value() != 0,
                    role, ": arena ID zero is reserved");
        TORCH_CHECK(backing.bytes == spec.storage_bytes(),
                    role, ": backing bytes ", backing.bytes,
                    " do not match dense storage bytes ",
                    spec.storage_bytes());
        TORCH_CHECK(backing.first_event.value() <=
                        backing.last_event.value(),
                    role, ": invalid inclusive event lifetime [",
                    backing.first_event.value(), ", ",
                    backing.last_event.value(), "]");
    };
    auto add_rows = [](int64_t total, int64_t rows, const char* role) {
        TORCH_CHECK(rows > 0 &&
                        total <= std::numeric_limits<int64_t>::max() - rows,
                    "SpmChunkedPermutationRoute: ", role,
                    " row count overflows int64");
        return total + rows;
    };

    const SpmDense2DSpec& first_source_spec = source_chunks.front().spec;
    first_source_spec.validate();
    TORCH_CHECK(first_source_spec.cols % 16 == 0,
                "SpmChunkedPermutationRoute: columns must be a multiple of "
                "16; got ", first_source_spec.cols);
    int64_t source_rows = 0;
    std::vector<int64_t> source_row_ends;
    source_row_ends.reserve(source_chunks.size());
    for (size_t i = 0; i < source_chunks.size(); ++i) {
        const SpmSourceChunkRef& chunk = source_chunks[i];
        TORCH_CHECK(chunk.key.port() == source_id &&
                        chunk.key.ordinal() == i,
                    "SpmChunkedPermutationRoute: source chunks must use "
                    "contiguous ordinals on the source port");
        validate_backing(chunk.spec, chunk.backing, "source chunk");
        TORCH_CHECK(chunk.spec.dtype == first_source_spec.dtype &&
                        chunk.spec.distribution ==
                            first_source_spec.distribution &&
                        chunk.spec.cols == first_source_spec.cols,
                    "SpmChunkedPermutationRoute: source chunk layouts must "
                    "match except for rows");
        source_rows = add_rows(source_rows, chunk.spec.rows, "source");
        source_row_ends.push_back(source_rows);
    }
    TORCH_CHECK(source_rows <= std::numeric_limits<int32_t>::max(),
                "SpmChunkedPermutationRoute: total source rows exceed int32 "
                "permutation index range");

    int64_t consumer_rows = 0;
    std::vector<int64_t> consumer_row_ends;
    consumer_row_ends.reserve(consumer_chunks.size());
    for (size_t i = 0; i < consumer_chunks.size(); ++i) {
        const SpmBorrowedConsumerChunkRef& chunk = consumer_chunks[i];
        TORCH_CHECK(chunk.key.port() == destination_id &&
                        chunk.key.ordinal() == i,
                    "SpmChunkedPermutationRoute: consumer chunks must use "
                    "contiguous ordinals on the destination port");
        validate_backing(chunk.spec, chunk.backing, "consumer chunk");
        TORCH_CHECK(chunk.spec.dtype == first_source_spec.dtype &&
                        chunk.spec.distribution ==
                            first_source_spec.distribution &&
                        chunk.spec.cols == first_source_spec.cols,
                    "SpmChunkedPermutationRoute: source/consumer layouts "
                    "must match except for rows");
        consumer_rows = add_rows(consumer_rows, chunk.spec.rows, "consumer");
        consumer_row_ends.push_back(consumer_rows);
    }

    TORCH_CHECK(source_row_for_output.size() ==
                    static_cast<size_t>(source_rows),
                "SpmChunkedPermutationRoute: permutation length ",
                source_row_for_output.size(), " does not equal total source "
                "rows ", source_rows);
    std::vector<uint8_t> seen(source_row_for_output.size(), 0);
    for (int32_t row : source_row_for_output) {
        TORCH_CHECK(row >= 0 && row < source_rows,
                    "SpmChunkedPermutationRoute: source row index ", row,
                    " is outside [0, ", source_rows, ")");
        TORCH_CHECK(seen[static_cast<size_t>(row)] == 0,
                    "SpmChunkedPermutationRoute: source row index ", row,
                    " appears more than once; a strict global permutation is "
                    "required");
        seen[static_cast<size_t>(row)] = 1;
    }

    TORCH_CHECK(!destination_runs.empty(),
                "SpmChunkedPermutationRoute: destination runs must not be empty");
    std::vector<int64_t> output_destination_rows;
    output_destination_rows.reserve(source_row_for_output.size());
    int64_t routed_rows = 0;
    int64_t previous_end = 0;
    bool first_run = true;
    for (const SpmDstRowRun& run : destination_runs) {
        TORCH_CHECK(run.dst_row_begin >= 0 && run.row_count > 0,
                    "SpmChunkedPermutationRoute: destination runs require "
                    "non-negative begin and positive count");
        TORCH_CHECK(run.dst_row_begin <= consumer_rows &&
                        run.row_count <= consumer_rows - run.dst_row_begin,
                    "SpmChunkedPermutationRoute: destination run [",
                    run.dst_row_begin, ", ", run.dst_row_begin, "+",
                    run.row_count, ") exceeds total consumer rows ",
                    consumer_rows);
        TORCH_CHECK(first_run || run.dst_row_begin >= previous_end,
                    "SpmChunkedPermutationRoute: destination runs must be "
                    "ordered and non-overlapping");
        TORCH_CHECK(routed_rows <= source_rows &&
                        run.row_count <= source_rows - routed_rows,
                    "SpmChunkedPermutationRoute: destination run row total "
                    "exceeds source rows ", source_rows);
        for (int64_t row = 0; row < run.row_count; ++row) {
            output_destination_rows.push_back(run.dst_row_begin + row);
        }
        routed_rows += run.row_count;
        previous_end = run.dst_row_begin + run.row_count;
        first_run = false;
    }
    TORCH_CHECK(routed_rows == source_rows,
                "SpmChunkedPermutationRoute: destination runs cover ",
                routed_rows, " rows but sources contain ", source_rows);

    auto chunk_for_row = [](int64_t row,
                            const std::vector<int64_t>& row_ends) -> size_t {
        for (size_t i = 0; i < row_ends.size(); ++i) {
            if (row < row_ends[i]) return i;
        }
        TORCH_INTERNAL_ASSERT(false, "row was outside sealed chunk bounds");
        return 0;
    };

    std::vector<int64_t> payload_rows(payload_chunks.size(), 0);
    std::vector<size_t> payload_consumer_indices(
        payload_chunks.size(), std::numeric_limits<size_t>::max());
    std::vector<std::pair<size_t, size_t>> split_payload_pairs;
    if (payload_partition == SpmPayloadPartition::ConsumerPacked) {
        for (size_t i = 0; i < payload_chunks.size(); ++i) {
            payload_consumer_indices[i] = i;
        }
    }
    std::vector<SpmSourcePayloadRowRun> source_payload_rows;
    source_payload_rows.reserve(source_row_for_output.size());
    std::vector<SpmPayloadConsumerRowRun> payload_consumer_rows;
    payload_consumer_rows.reserve(source_row_for_output.size());
    for (size_t output_row = 0;
         output_row < source_row_for_output.size(); ++output_row) {
        const int64_t source_global = source_row_for_output[output_row];
        const size_t source_chunk_index =
            chunk_for_row(source_global, source_row_ends);
        const int64_t source_chunk_begin = source_chunk_index == 0
            ? 0 : source_row_ends[source_chunk_index - 1];
        const int64_t source_local = source_global - source_chunk_begin;

        const int64_t destination_global =
            output_destination_rows[output_row];
        const size_t consumer_chunk_index =
            chunk_for_row(destination_global, consumer_row_ends);
        const int64_t consumer_chunk_begin = consumer_chunk_index == 0
            ? 0 : consumer_row_ends[consumer_chunk_index - 1];
        const int64_t consumer_local =
            destination_global - consumer_chunk_begin;

        size_t payload_index = consumer_chunk_index;
        if (payload_partition ==
            SpmPayloadPartition::SourceConsumerSplit) {
            const std::pair<size_t, size_t> pair{
                source_chunk_index, consumer_chunk_index};
            auto it = std::find(
                split_payload_pairs.begin(), split_payload_pairs.end(), pair);
            if (it == split_payload_pairs.end()) {
                payload_index = split_payload_pairs.size();
                TORCH_CHECK(
                    payload_index < payload_chunks.size(),
                    "SpmChunkedPermutationRoute: split payload list is "
                    "shorter than the non-empty source/consumer pair set");
                split_payload_pairs.push_back(pair);
                payload_consumer_indices[payload_index] =
                    consumer_chunk_index;
            } else {
                payload_index = static_cast<size_t>(
                    it - split_payload_pairs.begin());
            }
        }
        const int64_t payload_local = payload_rows[payload_index]++;

        const SpmChunkKey& source_key =
            source_chunks[source_chunk_index].key;
        const SpmChunkKey& payload_key =
            payload_chunks[payload_index].key;
        const SpmChunkKey& consumer_key =
            consumer_chunks[consumer_chunk_index].key;

        const SpmPhaseSlice& source_backing =
            source_chunks[source_chunk_index].backing;
        TORCH_CHECK(source_backing.last_event.value() !=
                        std::numeric_limits<uint32_t>::max(),
                    "SpmChunkedPermutationRoute: source-to-payload transfer "
                    "event overflows uint32");
        source_payload_rows.push_back(
            {source_key, source_local, payload_key, payload_local, 1,
             SpmPipelineEvent(source_backing.last_event.value() + 1)});
        const SpmPhaseSlice& payload_backing =
            payload_chunks[payload_index].backing;
        TORCH_CHECK(payload_backing.last_event.value() !=
                        std::numeric_limits<uint32_t>::max(),
                    "SpmChunkedPermutationRoute: payload-to-consumer transfer "
                    "event overflows uint32");
        payload_consumer_rows.push_back(
            {payload_key, payload_local, consumer_key, consumer_local, 1,
             SpmPipelineEvent(payload_backing.last_event.value() + 1)});
    }

    if (payload_partition ==
        SpmPayloadPartition::SourceConsumerSplit) {
        TORCH_CHECK(
            split_payload_pairs.size() == payload_chunks.size(),
            "SpmChunkedPermutationRoute: split payload list has ",
            payload_chunks.size(), " entries but lowering found ",
            split_payload_pairs.size(),
            " non-empty source/consumer pairs");
    }

    std::sort(
        source_payload_rows.begin(), source_payload_rows.end(),
        [](const SpmSourcePayloadRowRun& lhs,
           const SpmSourcePayloadRowRun& rhs) {
            if (lhs.transfer_event != rhs.transfer_event) {
                return lhs.transfer_event < rhs.transfer_event;
            }
            if (lhs.source_chunk.ordinal() !=
                rhs.source_chunk.ordinal()) {
                return lhs.source_chunk.ordinal() <
                       rhs.source_chunk.ordinal();
            }
            if (lhs.payload_chunk.ordinal() !=
                rhs.payload_chunk.ordinal()) {
                return lhs.payload_chunk.ordinal() <
                       rhs.payload_chunk.ordinal();
            }
            if (lhs.payload_row_begin != rhs.payload_row_begin) {
                return lhs.payload_row_begin < rhs.payload_row_begin;
            }
            return lhs.source_row_begin < rhs.source_row_begin;
        });
    std::vector<SpmSourcePayloadRowRun> source_payload_runs;
    source_payload_runs.reserve(source_payload_rows.size());
    for (const SpmSourcePayloadRowRun& row : source_payload_rows) {
        if (!source_payload_runs.empty()) {
            SpmSourcePayloadRowRun& previous = source_payload_runs.back();
            if (previous.transfer_event == row.transfer_event &&
                previous.source_chunk == row.source_chunk &&
                previous.payload_chunk == row.payload_chunk &&
                previous.source_row_begin + previous.row_count ==
                    row.source_row_begin &&
                previous.payload_row_begin + previous.row_count ==
                    row.payload_row_begin) {
                ++previous.row_count;
                continue;
            }
        }
        source_payload_runs.push_back(row);
    }
    std::sort(
        payload_consumer_rows.begin(), payload_consumer_rows.end(),
        [](const SpmPayloadConsumerRowRun& lhs,
           const SpmPayloadConsumerRowRun& rhs) {
            if (lhs.transfer_event != rhs.transfer_event) {
                return lhs.transfer_event < rhs.transfer_event;
            }
            if (lhs.consumer_chunk.ordinal() !=
                rhs.consumer_chunk.ordinal()) {
                return lhs.consumer_chunk.ordinal() <
                       rhs.consumer_chunk.ordinal();
            }
            if (lhs.payload_chunk.ordinal() !=
                rhs.payload_chunk.ordinal()) {
                return lhs.payload_chunk.ordinal() <
                       rhs.payload_chunk.ordinal();
            }
            if (lhs.payload_row_begin != rhs.payload_row_begin) {
                return lhs.payload_row_begin < rhs.payload_row_begin;
            }
            return lhs.consumer_row_begin < rhs.consumer_row_begin;
        });
    std::vector<SpmPayloadConsumerRowRun> payload_consumer_runs;
    payload_consumer_runs.reserve(payload_consumer_rows.size());
    for (const SpmPayloadConsumerRowRun& row : payload_consumer_rows) {
        if (!payload_consumer_runs.empty()) {
            SpmPayloadConsumerRowRun& previous =
                payload_consumer_runs.back();
            if (previous.transfer_event == row.transfer_event &&
                previous.payload_chunk == row.payload_chunk &&
                previous.consumer_chunk == row.consumer_chunk &&
                previous.payload_row_begin + previous.row_count ==
                    row.payload_row_begin &&
                previous.consumer_row_begin + previous.row_count ==
                    row.consumer_row_begin) {
                ++previous.row_count;
                continue;
            }
        }
        payload_consumer_runs.push_back(row);
    }

    for (size_t i = 0; i < payload_chunks.size(); ++i) {
        const SpmRetainedPayloadChunkRef& payload = payload_chunks[i];
        TORCH_CHECK(
            payload.key.port() == destination_id &&
                payload.key.ordinal() == i,
            "SpmChunkedPermutationRoute: payload chunks must use contiguous "
            "ordinals on the destination port");
        TORCH_CHECK(
            payload_consumer_indices[i] < consumer_chunks.size(),
            "SpmChunkedPermutationRoute: payload chunk has no consumer");
        if (payload_partition == SpmPayloadPartition::ConsumerPacked) {
            TORCH_CHECK(
                payload.key == consumer_chunks[i].key,
                "SpmChunkedPermutationRoute: packed payload chunk key must "
                "match its consumer chunk key");
        }
        validate_backing(payload.spec, payload.backing, "payload chunk");
        TORCH_CHECK(payload.spec.dtype == first_source_spec.dtype &&
                        payload.spec.distribution ==
                            first_source_spec.distribution &&
                        payload.spec.cols == first_source_spec.cols,
                    "SpmChunkedPermutationRoute: payload layout must match "
                    "source columns, dtype, and distribution");
        TORCH_CHECK(payload.spec.rows == payload_rows[i],
                    "SpmChunkedPermutationRoute: payload chunk ", i,
                    " stores ", payload.spec.rows,
                    " rows but lowering routes ", payload_rows[i],
                    " rows to its consumer");
    }

    std::vector<uint64_t> earliest_payload_transfer(
        payload_chunks.size(), std::numeric_limits<uint64_t>::max());
    for (const SpmSourcePayloadRowRun& run : source_payload_runs) {
        const size_t source_index = run.source_chunk.ordinal();
        const size_t payload_index = run.payload_chunk.ordinal();
        TORCH_INTERNAL_ASSERT(source_index < source_chunks.size() &&
                              payload_index < payload_chunks.size());
        const SpmPhaseSlice& source = source_chunks[source_index].backing;
        const SpmPhaseSlice& payload = payload_chunks[payload_index].backing;
        TORCH_INTERNAL_ASSERT(source.last_event.value() !=
                              std::numeric_limits<uint32_t>::max());
        const uint32_t transfer = run.transfer_event.value();
        TORCH_INTERNAL_ASSERT(transfer == source.last_event.value() + 1);
        TORCH_CHECK(transfer >= payload.first_event.value() &&
                        transfer <= payload.last_event.value(),
                    "SpmChunkedPermutationRoute: source-to-payload transfer "
                    "event ", transfer, " is outside payload lifetime [",
                    payload.first_event.value(), ", ",
                    payload.last_event.value(), "]");
        earliest_payload_transfer[payload_index] = std::min(
            earliest_payload_transfer[payload_index],
            static_cast<uint64_t>(transfer));
    }
    for (const SpmPayloadConsumerRowRun& run : payload_consumer_runs) {
        const size_t payload_index = run.payload_chunk.ordinal();
        const size_t consumer_index = run.consumer_chunk.ordinal();
        TORCH_INTERNAL_ASSERT(payload_index < payload_chunks.size() &&
                              consumer_index < consumer_chunks.size());
        const SpmPhaseSlice& payload = payload_chunks[payload_index].backing;
        const SpmPhaseSlice& consumer =
            consumer_chunks[consumer_index].backing;
        TORCH_INTERNAL_ASSERT(payload.last_event.value() !=
                              std::numeric_limits<uint32_t>::max());
        TORCH_INTERNAL_ASSERT(run.transfer_event.value() ==
                              payload.last_event.value() + 1);
        TORCH_CHECK(run.transfer_event == consumer.first_event,
                    "SpmChunkedPermutationRoute: payload-to-consumer transfer "
                    "event does not match the consumer first event");
    }
    for (size_t i = 0; i < payload_chunks.size(); ++i) {
        const SpmPhaseSlice& payload = payload_chunks[i].backing;
        const SpmPhaseSlice& consumer =
            consumer_chunks[payload_consumer_indices[i]].backing;
        TORCH_INTERNAL_ASSERT(earliest_payload_transfer[i] !=
                              std::numeric_limits<uint64_t>::max());
        TORCH_CHECK(payload.first_event.value() ==
                        earliest_payload_transfer[i],
                    "SpmChunkedPermutationRoute: payload first event must "
                    "equal its earliest incoming transfer boundary");
        TORCH_CHECK(payload.last_event.value() !=
                        std::numeric_limits<uint32_t>::max(),
                    "SpmChunkedPermutationRoute: payload-to-consumer transfer "
                    "event overflows uint32");
        TORCH_CHECK(payload.last_event.value() + 1 ==
                        consumer.first_event.value(),
                    "SpmChunkedPermutationRoute: payload last event must "
                    "strictly precede its consumer first event");
    }

    const bool legacy_schema =
        consumer_transform == SpmConsumerTransform::Overwrite &&
        payload_partition == SpmPayloadPartition::ConsumerPacked;
    const uint64_t route_schema_version = legacy_schema
        ? kChunkedPermutationRouteSchemaVersion
        : kChunkedPermutationRouteSchemaVersionV2;
    uint64_t hash = fnv_u64(kFnvOffset, route_schema_version);
    hash = fnv_u64(hash, source_id.value());
    hash = fnv_u64(hash, destination_id.value());
    hash = fnv_u64(hash, policy_version);
    if (!legacy_schema) {
        hash = fnv_u64(hash, static_cast<uint8_t>(consumer_transform));
        hash = fnv_u64(hash, static_cast<uint8_t>(payload_partition));
    }
    hash = fnv_u64(hash, source_chunks.size());
    for (const SpmSourceChunkRef& chunk : source_chunks) {
        hash = fnv_chunk_key(hash, chunk.key);
        hash = fnv_dense_spec(hash, chunk.spec);
        hash = fnv_phase_slice(hash, chunk.backing);
    }
    hash = fnv_u64(hash, consumer_chunks.size());
    for (const SpmBorrowedConsumerChunkRef& chunk : consumer_chunks) {
        hash = fnv_chunk_key(hash, chunk.key);
        hash = fnv_dense_spec(hash, chunk.spec);
        hash = fnv_phase_slice(hash, chunk.backing);
    }
    hash = fnv_u64(hash, payload_chunks.size());
    for (const SpmRetainedPayloadChunkRef& chunk : payload_chunks) {
        hash = fnv_chunk_key(hash, chunk.key);
        hash = fnv_dense_spec(hash, chunk.spec);
        hash = fnv_phase_slice(hash, chunk.backing);
    }
    hash = fnv_u64(hash, source_row_for_output.size());
    for (int32_t row : source_row_for_output) {
        hash = fnv_u64(hash, static_cast<uint32_t>(row));
    }
    hash = fnv_u64(hash, destination_runs.size());
    for (const SpmDstRowRun& run : destination_runs) {
        hash = fnv_u64(hash, static_cast<uint64_t>(run.dst_row_begin));
        hash = fnv_u64(hash, static_cast<uint64_t>(run.row_count));
    }
    hash = fnv_u64(hash, source_payload_runs.size());
    for (const SpmSourcePayloadRowRun& run : source_payload_runs) {
        hash = fnv_chunk_key(hash, run.source_chunk);
        hash = fnv_u64(hash, static_cast<uint64_t>(run.source_row_begin));
        hash = fnv_chunk_key(hash, run.payload_chunk);
        hash = fnv_u64(hash, static_cast<uint64_t>(run.payload_row_begin));
        hash = fnv_u64(hash, static_cast<uint64_t>(run.row_count));
        hash = fnv_u64(hash, run.transfer_event.value());
    }
    hash = fnv_u64(hash, payload_consumer_runs.size());
    for (const SpmPayloadConsumerRowRun& run : payload_consumer_runs) {
        hash = fnv_chunk_key(hash, run.payload_chunk);
        hash = fnv_u64(hash, static_cast<uint64_t>(run.payload_row_begin));
        hash = fnv_chunk_key(hash, run.consumer_chunk);
        hash = fnv_u64(hash, static_cast<uint64_t>(run.consumer_row_begin));
        hash = fnv_u64(hash, static_cast<uint64_t>(run.row_count));
        hash = fnv_u64(hash, run.transfer_event.value());
    }
    if (hash == 0) hash = kNonzeroHashFallback;

    return SpmChunkedPermutationRoute(
        source_id, destination_id, source_chunks, consumer_chunks,
        payload_chunks, source_row_for_output, destination_runs,
        source_payload_runs, payload_consumer_runs, policy_version,
        consumer_transform, payload_partition, route_schema_version, hash);
}

SpmDirectProducedRowRunsRoute SpmDirectProducedRowRunsRoute::bind(
    SpmPortId producer_id,
    SpmPortId destination_id,
    const std::vector<SpmDirectProducedChunkRef>& produced_chunks,
    const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks,
    const std::vector<SpmDstRowRun>& destination_runs,
    uint64_t policy_version) {
    TORCH_CHECK(producer_id.value() != 0 && destination_id.value() != 0,
                "SpmDirectProducedRowRunsRoute: port ID zero is reserved");
    TORCH_CHECK(producer_id != destination_id,
                "SpmDirectProducedRowRunsRoute: producer and destination "
                "ports must be distinct");
    TORCH_CHECK(policy_version != 0,
                "SpmDirectProducedRowRunsRoute: policy version zero is "
                "reserved");
    TORCH_CHECK(!produced_chunks.empty(),
                "SpmDirectProducedRowRunsRoute: produced chunks must not be "
                "empty");
    TORCH_CHECK(!consumer_chunks.empty(),
                "SpmDirectProducedRowRunsRoute: consumer chunks must not be "
                "empty");
    TORCH_CHECK(destination_runs.size() == produced_chunks.size(),
                "SpmDirectProducedRowRunsRoute: one destination run is "
                "required for each produced chunk");

    auto validate_consumer = [](const SpmBorrowedConsumerChunkRef& chunk) {
        chunk.spec.validate();
        TORCH_CHECK(chunk.backing.arena.value() != 0,
                    "SpmDirectProducedRowRunsRoute: consumer arena ID zero "
                    "is reserved");
        TORCH_CHECK(chunk.backing.bytes == chunk.spec.storage_bytes(),
                    "SpmDirectProducedRowRunsRoute: consumer backing bytes ",
                    chunk.backing.bytes,
                    " do not match dense storage bytes ",
                    chunk.spec.storage_bytes());
        TORCH_CHECK(chunk.backing.first_event.value() <=
                        chunk.backing.last_event.value(),
                    "SpmDirectProducedRowRunsRoute: invalid consumer "
                    "inclusive event lifetime");
    };

    const SpmDense2DSpec& first_produced_spec = produced_chunks.front().spec;
    first_produced_spec.validate();
    TORCH_CHECK(first_produced_spec.cols % 16 == 0,
                "SpmDirectProducedRowRunsRoute: columns must be a multiple "
                "of 16; got ", first_produced_spec.cols);
    for (size_t i = 0; i < produced_chunks.size(); ++i) {
        const SpmDirectProducedChunkRef& chunk = produced_chunks[i];
        TORCH_CHECK(chunk.key.port() == producer_id &&
                        chunk.key.ordinal() == i,
                    "SpmDirectProducedRowRunsRoute: produced chunks must use "
                    "contiguous ordinals on the producer port");
        chunk.spec.validate();
        TORCH_CHECK(chunk.spec.dtype == first_produced_spec.dtype &&
                        chunk.spec.distribution ==
                            first_produced_spec.distribution &&
                        chunk.spec.cols == first_produced_spec.cols,
                    "SpmDirectProducedRowRunsRoute: produced chunk layouts "
                    "must match except for rows");
    }

    int64_t consumer_rows = 0;
    std::vector<int64_t> consumer_row_ends;
    consumer_row_ends.reserve(consumer_chunks.size());
    for (size_t i = 0; i < consumer_chunks.size(); ++i) {
        const SpmBorrowedConsumerChunkRef& chunk = consumer_chunks[i];
        TORCH_CHECK(chunk.key.port() == destination_id &&
                        chunk.key.ordinal() == i,
                    "SpmDirectProducedRowRunsRoute: consumer chunks must use "
                    "contiguous ordinals on the destination port");
        validate_consumer(chunk);
        TORCH_CHECK(chunk.spec.dtype == first_produced_spec.dtype &&
                        chunk.spec.distribution ==
                            first_produced_spec.distribution &&
                        chunk.spec.cols == first_produced_spec.cols,
                    "SpmDirectProducedRowRunsRoute: producer/consumer "
                    "layouts must match except for rows");
        TORCH_CHECK(
            chunk.spec.rows > 0 &&
                consumer_rows <= std::numeric_limits<int64_t>::max() -
                                     chunk.spec.rows,
            "SpmDirectProducedRowRunsRoute: consumer row count overflows "
            "int64");
        consumer_rows += chunk.spec.rows;
        consumer_row_ends.push_back(consumer_rows);
    }

    std::vector<SpmDirectProducedRowRun> lowered_runs;
    lowered_runs.reserve(destination_runs.size());
    int64_t previous_end = 0;
    for (size_t i = 0; i < destination_runs.size(); ++i) {
        const SpmDstRowRun& run = destination_runs[i];
        const SpmDirectProducedChunkRef& produced = produced_chunks[i];
        TORCH_CHECK(run.dst_row_begin >= 0 && run.row_count > 0,
                    "SpmDirectProducedRowRunsRoute: destination runs require "
                    "non-negative begin and positive count");
        TORCH_CHECK(run.row_count == produced.spec.rows,
                    "SpmDirectProducedRowRunsRoute: destination run ", i,
                    " has ", run.row_count, " rows but produced chunk has ",
                    produced.spec.rows);
        TORCH_CHECK(i == 0 || run.dst_row_begin >= previous_end,
                    "SpmDirectProducedRowRunsRoute: destination runs must be "
                    "ordered and non-overlapping");
        TORCH_CHECK(run.dst_row_begin <= consumer_rows &&
                        run.row_count <=
                            consumer_rows - run.dst_row_begin,
                    "SpmDirectProducedRowRunsRoute: destination run exceeds "
                    "total consumer rows ", consumer_rows);
        previous_end = run.dst_row_begin + run.row_count;

        size_t consumer_index = 0;
        while (consumer_index < consumer_row_ends.size() &&
               run.dst_row_begin >= consumer_row_ends[consumer_index]) {
            ++consumer_index;
        }
        TORCH_INTERNAL_ASSERT(consumer_index < consumer_chunks.size());
        const int64_t consumer_begin = consumer_index == 0
            ? 0 : consumer_row_ends[consumer_index - 1];
        TORCH_CHECK(
            previous_end <= consumer_row_ends[consumer_index],
            "SpmDirectProducedRowRunsRoute: destination run ", i,
            " crosses consumer chunk boundary; direct producer requires one "
            "contiguous final output address");
        const SpmBorrowedConsumerChunkRef& consumer =
            consumer_chunks[consumer_index];
        TORCH_CHECK(
            consumer.backing.first_event.value() <=
                    produced.production_event.value() &&
                produced.production_event.value() <=
                    consumer.backing.last_event.value(),
            "SpmDirectProducedRowRunsRoute: production event ",
            produced.production_event.value(),
            " is outside consumer backing lifetime [",
            consumer.backing.first_event.value(), ", ",
            consumer.backing.last_event.value(), "]");
        lowered_runs.push_back(
            {produced.key, consumer.key,
             run.dst_row_begin - consumer_begin, run.row_count,
             produced.production_event});
    }

    uint64_t hash = fnv_u64(
        kFnvOffset, kDirectProducedRowRunsRouteSchemaVersion);
    hash = fnv_u64(hash, producer_id.value());
    hash = fnv_u64(hash, destination_id.value());
    hash = fnv_u64(hash, policy_version);
    hash = fnv_u64(hash, produced_chunks.size());
    for (const SpmDirectProducedChunkRef& chunk : produced_chunks) {
        hash = fnv_chunk_key(hash, chunk.key);
        hash = fnv_dense_spec(hash, chunk.spec);
        hash = fnv_u64(hash, chunk.production_event.value());
    }
    hash = fnv_u64(hash, consumer_chunks.size());
    for (const SpmBorrowedConsumerChunkRef& chunk : consumer_chunks) {
        hash = fnv_chunk_key(hash, chunk.key);
        hash = fnv_dense_spec(hash, chunk.spec);
        hash = fnv_phase_slice(hash, chunk.backing);
    }
    hash = fnv_u64(hash, destination_runs.size());
    for (const SpmDstRowRun& run : destination_runs) {
        hash = fnv_u64(hash, static_cast<uint64_t>(run.dst_row_begin));
        hash = fnv_u64(hash, static_cast<uint64_t>(run.row_count));
    }
    hash = fnv_u64(hash, lowered_runs.size());
    for (const SpmDirectProducedRowRun& run : lowered_runs) {
        hash = fnv_chunk_key(hash, run.producer_chunk);
        hash = fnv_chunk_key(hash, run.consumer_chunk);
        hash = fnv_u64(hash,
                       static_cast<uint64_t>(run.consumer_row_begin));
        hash = fnv_u64(hash, static_cast<uint64_t>(run.row_count));
        hash = fnv_u64(hash, run.production_event.value());
    }
    if (hash == 0) hash = kNonzeroHashFallback;

    return SpmDirectProducedRowRunsRoute(
        producer_id, destination_id, produced_chunks, consumer_chunks,
        destination_runs, lowered_runs, policy_version,
        kDirectProducedRowRunsRouteSchemaVersion, hash);
}

SpmFmbDirectProducedPostFnBinding
SpmFmbDirectProducedPostFnBinding::bind(
    const SpmFmbSealedPostFnYields& producer,
    size_t yield_ordinal,
    SpmChunkKey producer_chunk,
    const SpmFmbConsumerRowSliceEndpoint& consumer,
    uint64_t policy_version) {
    TORCH_CHECK(policy_version != 0,
                "post_fn direct-produced binding policy version zero is "
                "reserved");
    TORCH_CHECK(!producer.cpu_dry() && !consumer.manifest_.cpu_dry_,
                "post_fn direct-produced binding requires live producer "
                "and consumer authority");
    TORCH_CHECK(producer.yield_hash_ != 0 &&
                    yield_ordinal < producer.yields_.size(),
                "post_fn direct-produced binding yield ordinal is out of "
                "range");
    const std::shared_ptr<const uint64_t> producer_owner =
        producer.source_trace_.owner_generation_.lock();
    const std::shared_ptr<const uint64_t> consumer_owner =
        consumer.manifest_.owner_generation_.lock();
    TORCH_CHECK(
        producer_owner != nullptr && consumer_owner != nullptr &&
            *producer_owner ==
                producer.source_trace_.expected_owner_generation_ &&
            *consumer_owner ==
                consumer.manifest_.expected_owner_generation_,
        "post_fn direct-produced binding owner is stale");
    TORCH_CHECK(producer_chunk.port().value() != 0,
                "post_fn direct-produced binding producer port ID zero is "
                "reserved");
    const auto& yield = producer.yields_[yield_ordinal];
    yield.spec.validate();
    TORCH_CHECK(
        yield.ordinal == yield_ordinal &&
            yield.writer_kind ==
                SpmFmbTerminalWriterKind::AllReduceSumResidual,
        "post_fn direct-produced binding terminal writer differs from the "
        "sealed yield");
    TORCH_CHECK(
        yield.spec.rows == consumer.row_count_ &&
            yield.spec.cols == consumer.consumer_.spec.cols &&
            yield.spec.dtype == consumer.consumer_.spec.dtype &&
            yield.spec.distribution ==
                consumer.consumer_.spec.distribution,
        "post_fn direct-produced binding producer and consumer row-slice "
        "geometry differ");

    uint64_t hash = fnv_u64(kFnvOffset, 30);
    hash = fnv_u64(hash, producer.yield_hash_);
    hash = fnv_u64(hash, yield_ordinal);
    hash = fnv_chunk_key(hash, producer_chunk);
    hash = fnv_dense_spec(hash, yield.spec);
    hash = fnv_u64(hash, consumer.identity_hash_);
    hash = fnv_u64(hash, policy_version);
    if (hash == 0) hash = kNonzeroHashFallback;
    return SpmFmbDirectProducedPostFnBinding(
        producer, yield_ordinal, producer_chunk, yield.spec, consumer,
        policy_version, hash);
}

SpmFmbDirectProducedPostFnRunAddBinding
SpmFmbDirectProducedPostFnRunAddBinding::bind(
    const SpmFmbSealedPostFnYields& producer,
    size_t yield_ordinal,
    SpmChunkKey payload_chunk,
    const SpmFmbConsumerRowSliceEndpoint& consumer,
    uint64_t policy_version) {
    TORCH_CHECK(policy_version != 0,
                "post_fn RunAdd binding policy version zero is reserved");
    TORCH_CHECK(!producer.cpu_dry() && !consumer.manifest_.cpu_dry_,
                "post_fn RunAdd binding requires live producer and "
                "consumer authority");
    TORCH_CHECK(producer.yield_hash_ != 0 &&
                    yield_ordinal < producer.yields_.size(),
                "post_fn RunAdd binding yield ordinal is out of range");
    const std::shared_ptr<const uint64_t> producer_owner =
        producer.source_trace_.owner_generation_.lock();
    const std::shared_ptr<const uint64_t> consumer_owner =
        consumer.manifest_.owner_generation_.lock();
    TORCH_CHECK(
        producer_owner != nullptr && consumer_owner != nullptr &&
            *producer_owner ==
                producer.source_trace_.expected_owner_generation_ &&
            *consumer_owner ==
                consumer.manifest_.expected_owner_generation_,
        "post_fn RunAdd binding owner is stale");
    TORCH_CHECK(payload_chunk.port().value() != 0,
                "post_fn RunAdd binding payload port ID zero is reserved");
    const auto& yield = producer.yields_[yield_ordinal];
    yield.spec.validate();
    TORCH_CHECK(
        yield.ordinal == yield_ordinal &&
            yield.writer_kind ==
                SpmFmbTerminalWriterKind::AllReduceSumResidual,
        "post_fn RunAdd binding terminal writer differs from the sealed "
        "yield");
    TORCH_CHECK(
        yield.spec.rows == consumer.row_count_ &&
            yield.spec.cols == consumer.consumer_.spec.cols &&
            yield.spec.dtype == consumer.consumer_.spec.dtype &&
            yield.spec.distribution ==
                consumer.consumer_.spec.distribution,
        "post_fn RunAdd binding payload and consumer row-slice geometry "
        "differ");

    uint64_t hash = fnv_u64(kFnvOffset, 31);
    hash = fnv_u64(hash, producer.yield_hash_);
    hash = fnv_u64(hash, yield_ordinal);
    hash = fnv_chunk_key(hash, payload_chunk);
    hash = fnv_dense_spec(hash, yield.spec);
    hash = fnv_u64(hash, consumer.identity_hash_);
    hash = fnv_u64(hash, policy_version);
    if (hash == 0) hash = kNonzeroHashFallback;
    return SpmFmbDirectProducedPostFnRunAddBinding(
        producer, yield_ordinal, payload_chunk, yield.spec, consumer,
        policy_version, hash);
}

SpmDirectProducedPayloadRunAddRoute
SpmDirectProducedPayloadRunAddRoute::bind(
    SpmPortId payload_id,
    SpmPortId destination_id,
    const std::vector<SpmDirectProducedPayloadRef>& payloads,
    const std::vector<SpmBorrowedConsumerChunkRef>& consumer_chunks,
    const std::vector<SpmDstRowRun>& destination_runs,
    uint64_t policy_version) {
    TORCH_CHECK(payload_id.value() != 0 && destination_id.value() != 0,
                "SpmDirectProducedPayloadRunAddRoute: port ID zero is "
                "reserved");
    TORCH_CHECK(payload_id != destination_id,
                "SpmDirectProducedPayloadRunAddRoute: payload and "
                "destination ports must be distinct");
    TORCH_CHECK(policy_version != 0,
                "SpmDirectProducedPayloadRunAddRoute: policy version zero "
                "is reserved");
    TORCH_CHECK(!payloads.empty(),
                "SpmDirectProducedPayloadRunAddRoute: payloads must not be "
                "empty");
    TORCH_CHECK(!consumer_chunks.empty(),
                "SpmDirectProducedPayloadRunAddRoute: consumer chunks must "
                "not be empty");
    TORCH_CHECK(destination_runs.size() == payloads.size(),
                "SpmDirectProducedPayloadRunAddRoute: one destination run "
                "is required for each payload");

    const SpmDense2DSpec& first_payload_spec = payloads.front().spec;
    first_payload_spec.validate();
    TORCH_CHECK(first_payload_spec.cols % 16 == 0,
                "SpmDirectProducedPayloadRunAddRoute: columns must be a "
                "multiple of 16; got ", first_payload_spec.cols);
    for (size_t i = 0; i < payloads.size(); ++i) {
        const SpmDirectProducedPayloadRef& payload = payloads[i];
        TORCH_CHECK(payload.key.port() == payload_id &&
                        payload.key.ordinal() == i,
                    "SpmDirectProducedPayloadRunAddRoute: payloads must use "
                    "contiguous ordinals on the payload port");
        payload.spec.validate();
        TORCH_CHECK(payload.spec.dtype == first_payload_spec.dtype &&
                        payload.spec.distribution ==
                            first_payload_spec.distribution &&
                        payload.spec.cols == first_payload_spec.cols,
                    "SpmDirectProducedPayloadRunAddRoute: payload layouts "
                    "must match except for rows");
        TORCH_CHECK(payload.backing.arena.value() != 0,
                    "SpmDirectProducedPayloadRunAddRoute: payload arena ID "
                    "zero is reserved");
        TORCH_CHECK(payload.backing.bytes == payload.spec.storage_bytes(),
                    "SpmDirectProducedPayloadRunAddRoute: payload backing "
                    "bytes do not match dense storage bytes");
        TORCH_CHECK(payload.backing.first_event.value() <=
                        payload.backing.last_event.value(),
                    "SpmDirectProducedPayloadRunAddRoute: invalid payload "
                    "inclusive event lifetime");
        TORCH_CHECK(payload.backing.first_event ==
                        payload.production_event,
                    "SpmDirectProducedPayloadRunAddRoute: payload first "
                    "event must equal its production event");
        TORCH_CHECK(payload.backing.last_event.value() !=
                        std::numeric_limits<uint32_t>::max(),
                    "SpmDirectProducedPayloadRunAddRoute: payload-to-"
                    "consumer add event overflows uint32");
    }

    int64_t consumer_rows = 0;
    std::vector<int64_t> consumer_row_ends;
    consumer_row_ends.reserve(consumer_chunks.size());
    for (size_t i = 0; i < consumer_chunks.size(); ++i) {
        const SpmBorrowedConsumerChunkRef& consumer = consumer_chunks[i];
        TORCH_CHECK(consumer.key.port() == destination_id &&
                        consumer.key.ordinal() == i,
                    "SpmDirectProducedPayloadRunAddRoute: consumer chunks "
                    "must use contiguous ordinals on the destination port");
        consumer.spec.validate();
        TORCH_CHECK(consumer.spec.dtype == first_payload_spec.dtype &&
                        consumer.spec.distribution ==
                            first_payload_spec.distribution &&
                        consumer.spec.cols == first_payload_spec.cols,
                    "SpmDirectProducedPayloadRunAddRoute: payload/consumer "
                    "layouts must match except for rows");
        TORCH_CHECK(consumer.backing.arena.value() != 0,
                    "SpmDirectProducedPayloadRunAddRoute: consumer arena ID "
                    "zero is reserved");
        TORCH_CHECK(consumer.backing.bytes == consumer.spec.storage_bytes(),
                    "SpmDirectProducedPayloadRunAddRoute: consumer backing "
                    "bytes do not match dense storage bytes");
        TORCH_CHECK(consumer.backing.first_event.value() <=
                        consumer.backing.last_event.value(),
                    "SpmDirectProducedPayloadRunAddRoute: invalid consumer "
                    "inclusive event lifetime");
        TORCH_CHECK(
            consumer.spec.rows > 0 &&
                consumer_rows <= std::numeric_limits<int64_t>::max() -
                                     consumer.spec.rows,
            "SpmDirectProducedPayloadRunAddRoute: consumer row count "
            "overflows int64");
        consumer_rows += consumer.spec.rows;
        consumer_row_ends.push_back(consumer_rows);
    }

    std::vector<SpmDirectProducedPayloadRunAdd> run_adds;
    run_adds.reserve(payloads.size());
    int64_t previous_end = 0;
    for (size_t i = 0; i < payloads.size(); ++i) {
        const SpmDirectProducedPayloadRef& payload = payloads[i];
        const SpmDstRowRun& run = destination_runs[i];
        TORCH_CHECK(run.dst_row_begin >= 0 && run.row_count > 0,
                    "SpmDirectProducedPayloadRunAddRoute: destination runs "
                    "require non-negative begin and positive count");
        TORCH_CHECK(run.row_count == payload.spec.rows,
                    "SpmDirectProducedPayloadRunAddRoute: destination run ",
                    i, " row count does not match its complete payload");
        TORCH_CHECK(i == 0 || run.dst_row_begin >= previous_end,
                    "SpmDirectProducedPayloadRunAddRoute: destination runs "
                    "must be ordered and non-overlapping");
        TORCH_CHECK(run.dst_row_begin <= consumer_rows &&
                        run.row_count <=
                            consumer_rows - run.dst_row_begin,
                    "SpmDirectProducedPayloadRunAddRoute: destination run "
                    "exceeds total consumer rows ", consumer_rows);
        previous_end = run.dst_row_begin + run.row_count;

        size_t consumer_index = 0;
        while (consumer_index < consumer_row_ends.size() &&
               run.dst_row_begin >= consumer_row_ends[consumer_index]) {
            ++consumer_index;
        }
        TORCH_INTERNAL_ASSERT(consumer_index < consumer_chunks.size());
        const int64_t consumer_begin = consumer_index == 0
            ? 0 : consumer_row_ends[consumer_index - 1];
        TORCH_CHECK(
            previous_end <= consumer_row_ends[consumer_index],
            "SpmDirectProducedPayloadRunAddRoute: destination run ", i,
            " crosses a consumer chunk boundary");
        const SpmBorrowedConsumerChunkRef& consumer =
            consumer_chunks[consumer_index];
        const uint32_t add_event = payload.backing.last_event.value() + 1;
        TORCH_CHECK(
            add_event == consumer.backing.first_event.value(),
            "SpmDirectProducedPayloadRunAddRoute: payload last event plus "
            "one must equal the consumer first/add event");
        run_adds.push_back({
            payload.key, consumer.key,
            run.dst_row_begin - consumer_begin, run.row_count,
            payload.production_event, SpmPipelineEvent(add_event)});
    }

    uint64_t hash = fnv_u64(
        kFnvOffset, kDirectProducedPayloadRunAddRouteSchemaVersion);
    hash = fnv_u64(hash, payload_id.value());
    hash = fnv_u64(hash, destination_id.value());
    hash = fnv_u64(hash, policy_version);
    hash = fnv_u64(
        hash, static_cast<uint8_t>(SpmConsumerTransform::RunAdd));
    hash = fnv_u64(hash, SpmDirectProducedPayloadRunAddRoute::alpha());
    hash = fnv_u64(hash, payloads.size());
    for (const SpmDirectProducedPayloadRef& payload : payloads) {
        hash = fnv_chunk_key(hash, payload.key);
        hash = fnv_dense_spec(hash, payload.spec);
        hash = fnv_phase_slice(hash, payload.backing);
        hash = fnv_u64(hash, payload.production_event.value());
    }
    hash = fnv_u64(hash, consumer_chunks.size());
    for (const SpmBorrowedConsumerChunkRef& consumer : consumer_chunks) {
        hash = fnv_chunk_key(hash, consumer.key);
        hash = fnv_dense_spec(hash, consumer.spec);
        hash = fnv_phase_slice(hash, consumer.backing);
    }
    hash = fnv_u64(hash, destination_runs.size());
    for (const SpmDstRowRun& run : destination_runs) {
        hash = fnv_u64(hash, static_cast<uint64_t>(run.dst_row_begin));
        hash = fnv_u64(hash, static_cast<uint64_t>(run.row_count));
    }
    hash = fnv_u64(hash, run_adds.size());
    for (const SpmDirectProducedPayloadRunAdd& run_add : run_adds) {
        hash = fnv_chunk_key(hash, run_add.payload_chunk);
        hash = fnv_chunk_key(hash, run_add.consumer_chunk);
        hash = fnv_u64(
            hash, static_cast<uint64_t>(run_add.consumer_row_begin));
        hash = fnv_u64(hash, static_cast<uint64_t>(run_add.row_count));
        hash = fnv_u64(hash, run_add.production_event.value());
        hash = fnv_u64(hash, run_add.add_event.value());
    }
    if (hash == 0) hash = kNonzeroHashFallback;

    return SpmDirectProducedPayloadRunAddRoute(
        payload_id, destination_id, payloads, consumer_chunks,
        destination_runs, run_adds, policy_version,
        kDirectProducedPayloadRunAddRouteSchemaVersion, hash);
}

SpmPipelinePlan SpmPipelinePlan::compile(
    const std::vector<BufferDecl>& regions,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    TORCH_CHECK(budget_bytes > 0,
                "SpmPipelinePlan: budget must be positive");
    TORCH_CHECK(budget_bytes <= SpmAllocator::SPM_PLANNING_BUDGET,
                "SpmPipelinePlan: budget ", budget_bytes,
                " exceeds canonical ceiling ",
                SpmAllocator::SPM_PLANNING_BUDGET);
    TORCH_CHECK(!regions.empty(),
                "SpmPipelinePlan: at least one region is required");
    TORCH_CHECK(reserved_top_bytes <= budget_bytes,
                "SpmPipelinePlan: reserved_top_bytes ", reserved_top_bytes,
                " exceeds budget ", budget_bytes);

    SpmPipelinePlan result;
    result.budget_bytes_ = budget_bytes;
    result.reserved_top_bytes_ = align_up(reserved_top_bytes);
    TORCH_CHECK(result.reserved_top_bytes_ <= budget_bytes,
                "SpmPipelinePlan: aligned reserved_top_bytes ",
                result.reserved_top_bytes_, " exceeds budget ", budget_bytes);

    std::unordered_set<std::string> names;
    std::vector<SpmAllocator::AllocRequest> requests;
    requests.reserve(regions.size());
    result.regions_.reserve(regions.size());

    for (const BufferDecl& decl : regions) {
        TORCH_CHECK(decl.name != nullptr && decl.name[0] != '\0',
                    "SpmPipelinePlan: every region needs a non-empty name");
        const std::string name(decl.name);
        TORCH_CHECK(names.insert(name).second,
                    "SpmPipelinePlan: duplicate region name '", name, "'");
        TORCH_CHECK(decl.size > 0,
                    "SpmPipelinePlan: region '", name,
                    "' must have positive size");
        const size_t region_bytes = static_cast<size_t>(decl.size);
        const size_t aligned_region_bytes = align_up(region_bytes);
        TORCH_CHECK(aligned_region_bytes <= budget_bytes,
                    "SpmPipelinePlan: aligned region '", name, "' needs ",
                    aligned_region_bytes, " bytes, exceeding budget ",
                    budget_bytes);
        TORCH_CHECK(aligned_region_bytes <=
                        std::numeric_limits<uint32_t>::max(),
                    "SpmPipelinePlan: aligned region '", name,
                    "' exceeds the uint32 offset range");
        TORCH_CHECK(decl.phase_start >= 0 &&
                        decl.phase_start <= decl.phase_end,
                    "SpmPipelinePlan: invalid lifetime for region '", name,
                    "': [", decl.phase_start, ", ", decl.phase_end, "]");
        TORCH_CHECK(decl.storage == StorageClass::Temp,
                    "SpmPipelinePlan: region '", name,
                    "' uses unsupported storage class; V1 accepts Temp only");
        TORCH_CHECK(decl.scope == BufferScope::LayerWide,
                    "SpmPipelinePlan: region '", name,
                    "' uses unsupported scope; V1 accepts LayerWide only");
        TORCH_CHECK(decl.per_layer == 0,
                    "SpmPipelinePlan: region '", name,
                    "' uses unsupported per_layer expansion");
        TORCH_CHECK(decl.alias_of == nullptr,
                    "SpmPipelinePlan: region '", name,
                    "' uses unsupported alias_of");
        TORCH_CHECK(!decl.preload_callback,
                    "SpmPipelinePlan: region '", name,
                    "' uses unsupported preload_callback");
        TORCH_CHECK(!decl.reverse_layer_alloc,
                    "SpmPipelinePlan: region '", name,
                    "' uses unsupported reverse_layer_alloc");

        requests.push_back({decl.size,
                            decl.phase_start,
                            decl.phase_end,
                            static_cast<int>(decl.scope)});
        result.regions_.push_back({name,
                                   0,
                                   0,
                                   region_bytes,
                                   decl.phase_start,
                                   decl.phase_end,
                                   0,
                                   {}});
    }

    const SpmAllocator::AliasedPlan allocation =
        SpmAllocator::plan_temporary_aliased(requests);
    TORCH_CHECK(allocation.offsets.size() == result.regions_.size(),
                "SpmPipelinePlan: planner returned the wrong offset count");
    for (size_t i = 0; i < result.regions_.size(); ++i) {
        result.regions_[i].offset = allocation.offsets[i];
    }
    result.peak_bytes_ = allocation.peak_bytes;

    TORCH_CHECK(result.peak_bytes_ <= budget_bytes &&
                    result.reserved_top_bytes_ <=
                        budget_bytes - result.peak_bytes_,
                "SpmPipelinePlan: plan requires ", result.total_bytes(),
                " bytes/core (temporary peak ", result.peak_bytes_,
                " + reserved top ", result.reserved_top_bytes_,
                "), exceeding budget ", budget_bytes);

    uint64_t hash = fnv_u64(kFnvOffset, kPlanSchemaVersion);
    hash = fnv_u64(hash, result.budget_bytes_);
    hash = fnv_u64(hash, result.reserved_top_bytes_);
    hash = fnv_u64(hash, result.peak_bytes_);
    hash = fnv_u64(hash, result.regions_.size());
    for (const Region& region : result.regions_) {
        hash = fnv_string(hash, region.name);
        hash = fnv_u64(hash, region.bytes);
        hash = fnv_u64(hash, static_cast<uint64_t>(region.first_event));
        hash = fnv_u64(hash, static_cast<uint64_t>(region.last_event));
        hash = fnv_u64(hash, region.offset);
    }
    result.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    return result;
}

SpmPipelinePlan SpmPipelinePlan::compile(
    const std::vector<SpmScratchDecl>& scratch_regions,
    const std::vector<SpmRowSliceRoute>& row_slice_routes,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    std::vector<SpmIdentityRoute> storage_routes;
    storage_routes.reserve(row_slice_routes.size());
    for (const SpmRowSliceRoute& route : row_slice_routes) {
        storage_routes.push_back(SpmIdentityRoute::bind(
            route.id(), route.storage_spec(), route.storage_spec(),
            route.first_write(), route.last_read()));
    }

    SpmPipelinePlan result = compile(scratch_regions, storage_routes,
                                     reserved_top_bytes, budget_bytes);
    TORCH_INTERNAL_ASSERT(result.regions_.size() ==
                          scratch_regions.size() + row_slice_routes.size());
    for (size_t i = 0; i < row_slice_routes.size(); ++i) {
        Region& region = result.regions_[scratch_regions.size() + i];
        const SpmRowSliceRoute& route = row_slice_routes[i];
        TORCH_INTERNAL_ASSERT(region.port_id == route.id().value());
        region.row_slice_route = true;
        region.dst_row_begin = route.dst_row_begin();
        region.dst_row_count = route.dst_row_count();
    }

    uint64_t hash = fnv_u64(kFnvOffset, kTypedRowSlicePlanSchemaVersion);
    hash = fnv_u64(hash, result.budget_bytes_);
    hash = fnv_u64(hash, result.reserved_top_bytes_);
    hash = fnv_u64(hash, result.peak_bytes_);
    hash = fnv_u64(hash, result.regions_.size());
    for (const Region& region : result.regions_) {
        if (region.scratch_id != 0) {
            hash = fnv_u64(hash, kTypedScratchRegionTag);
            hash = fnv_u64(hash, region.scratch_id);
        } else {
            TORCH_INTERNAL_ASSERT(region.port_id != 0 &&
                                  region.row_slice_route);
            hash = fnv_u64(hash, kTypedRowSlicePortRegionTag);
            hash = fnv_u64(hash, region.port_id);
        }
        hash = fnv_u64(hash, region.bytes);
        hash = fnv_u64(hash, static_cast<uint64_t>(region.first_event));
        hash = fnv_u64(hash, static_cast<uint64_t>(region.last_event));
        hash = fnv_u64(hash, region.offset);
        if (region.port_id != 0) {
            hash = fnv_u64(hash,
                           static_cast<uint8_t>(region.port_spec.dtype));
            hash = fnv_u64(hash,
                           static_cast<uint64_t>(region.port_spec.rows));
            hash = fnv_u64(hash,
                           static_cast<uint64_t>(region.port_spec.cols));
            hash = fnv_u64(
                hash, static_cast<uint8_t>(region.port_spec.distribution));
            hash = fnv_u64(hash,
                           static_cast<uint64_t>(region.dst_row_begin));
            hash = fnv_u64(hash,
                           static_cast<uint64_t>(region.dst_row_count));
        }
    }
    result.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    return result;
}

SpmPipelinePlan SpmPipelinePlan::compile(
    const std::vector<SpmScratchDecl>& scratch_regions,
    const std::vector<SpmPermutationRowRunsRoute>& permutation_routes,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    TORCH_CHECK(!permutation_routes.empty(),
                "SpmPipelinePlan: permutation route list must not be empty");

    std::vector<SpmIdentityRoute> all_ports;
    all_ports.reserve(permutation_routes.size() * 2);
    for (const SpmPermutationRowRunsRoute& route : permutation_routes) {
        const SpmPermutationRowRunsEvents& events = route.events();
        all_ports.push_back(SpmIdentityRoute::bind(
            route.source_id(), route.source_spec(), route.source_spec(),
            events.source_first_write, events.transform));
        all_ports.push_back(SpmIdentityRoute::bind(
            route.destination_id(), route.destination_spec(),
            route.destination_spec(), events.destination_first_write,
            events.destination_last_read));
    }

    SpmPipelinePlan result = compile(scratch_regions, all_ports,
                                     reserved_top_bytes, budget_bytes);
    result.permutation_routes_ = permutation_routes;

    uint64_t hash = fnv_u64(
        kFnvOffset, kTypedPermutationRowRunsPlanSchemaVersion);
    hash = fnv_u64(hash, result.budget_bytes_);
    hash = fnv_u64(hash, result.reserved_top_bytes_);
    hash = fnv_u64(hash, result.peak_bytes_);
    hash = fnv_u64(hash, result.regions_.size());
    for (const Region& region : result.regions_) {
        uint64_t region_tag = 0;
        if (region.scratch_id != 0) {
            region_tag = kTypedScratchRegionTag;
            hash = fnv_u64(hash, region_tag);
            hash = fnv_u64(hash, region.scratch_id);
        } else {
            TORCH_INTERNAL_ASSERT(region.port_id != 0);
            for (const SpmPermutationRowRunsRoute& route :
                 result.permutation_routes_) {
                if (route.source_id().value() == region.port_id) {
                    TORCH_INTERNAL_ASSERT(region_tag == 0);
                    region_tag = kTypedPermutationSourceRegionTag;
                }
                if (route.destination_id().value() == region.port_id) {
                    TORCH_INTERNAL_ASSERT(region_tag == 0);
                    region_tag = kTypedPermutationDestinationRegionTag;
                }
            }
            TORCH_INTERNAL_ASSERT(region_tag != 0);
            hash = fnv_u64(hash, region_tag);
            hash = fnv_u64(hash, region.port_id);
        }
        hash = fnv_u64(hash, region.bytes);
        hash = fnv_u64(hash, static_cast<uint64_t>(region.first_event));
        hash = fnv_u64(hash, static_cast<uint64_t>(region.last_event));
        hash = fnv_u64(hash, region.offset);
        if (region.port_id != 0) {
            hash = fnv_dense_spec(hash, region.port_spec);
        }
    }
    hash = fnv_u64(hash, result.permutation_routes_.size());
    for (const SpmPermutationRowRunsRoute& route :
         result.permutation_routes_) {
        hash = fnv_u64(hash, route.hash());
    }
    result.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    return result;
}

SpmPipelinePlan SpmPipelinePlan::compile(
    const std::vector<SpmScratchDecl>& scratch_regions,
    const std::vector<SpmIdentityRoute>& identity_routes,
    const std::vector<SpmPermutationRowRunsRoute>& permutation_routes,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    TORCH_CHECK(!identity_routes.empty(),
                "SpmPipelinePlan: mixed typed plan requires at least one "
                "identity route");
    TORCH_CHECK(!permutation_routes.empty(),
                "SpmPipelinePlan: mixed typed plan requires at least one "
                "permutation route");

    // Reuse the exact typed allocator/validation path.  Appending every
    // permutation endpoint to the identity regions makes its existing port-ID
    // set the single plan-wide uniqueness gate, including cross-kind clashes.
    std::vector<SpmIdentityRoute> all_ports;
    all_ports.reserve(identity_routes.size() + permutation_routes.size() * 2);
    all_ports.insert(all_ports.end(), identity_routes.begin(),
                     identity_routes.end());
    for (const SpmPermutationRowRunsRoute& route : permutation_routes) {
        const SpmPermutationRowRunsEvents& events = route.events();
        all_ports.push_back(SpmIdentityRoute::bind(
            route.source_id(), route.source_spec(), route.source_spec(),
            events.source_first_write, events.transform));
        all_ports.push_back(SpmIdentityRoute::bind(
            route.destination_id(), route.destination_spec(),
            route.destination_spec(), events.destination_first_write,
            events.destination_last_read));
    }

    SpmPipelinePlan result = compile(scratch_regions, all_ports,
                                     reserved_top_bytes, budget_bytes);
    result.permutation_routes_ = permutation_routes;

    std::unordered_set<uint32_t> identity_port_ids;
    identity_port_ids.reserve(identity_routes.size());
    for (const SpmIdentityRoute& route : identity_routes) {
        const bool inserted = identity_port_ids.insert(route.id().value()).second;
        TORCH_INTERNAL_ASSERT(inserted);
    }

    uint64_t hash = fnv_u64(
        kFnvOffset, kTypedMixedIdentityPermutationPlanSchemaVersion);
    hash = fnv_u64(hash, result.budget_bytes_);
    hash = fnv_u64(hash, result.reserved_top_bytes_);
    hash = fnv_u64(hash, result.peak_bytes_);
    hash = fnv_u64(hash, result.regions_.size());
    for (const Region& region : result.regions_) {
        uint64_t region_tag = 0;
        if (region.scratch_id != 0) {
            region_tag = kTypedScratchRegionTag;
            hash = fnv_u64(hash, region_tag);
            hash = fnv_u64(hash, region.scratch_id);
        } else {
            TORCH_INTERNAL_ASSERT(region.port_id != 0);
            if (identity_port_ids.count(region.port_id) != 0) {
                region_tag = kTypedPortRegionTag;
            } else {
                for (const SpmPermutationRowRunsRoute& route :
                     result.permutation_routes_) {
                    if (route.source_id().value() == region.port_id) {
                        TORCH_INTERNAL_ASSERT(region_tag == 0);
                        region_tag = kTypedPermutationSourceRegionTag;
                    }
                    if (route.destination_id().value() == region.port_id) {
                        TORCH_INTERNAL_ASSERT(region_tag == 0);
                        region_tag = kTypedPermutationDestinationRegionTag;
                    }
                }
            }
            TORCH_INTERNAL_ASSERT(region_tag != 0);
            hash = fnv_u64(hash, region_tag);
            hash = fnv_u64(hash, region.port_id);
        }
        hash = fnv_u64(hash, region.bytes);
        hash = fnv_u64(hash, static_cast<uint64_t>(region.first_event));
        hash = fnv_u64(hash, static_cast<uint64_t>(region.last_event));
        hash = fnv_u64(hash, region.offset);
        if (region.port_id != 0) {
            hash = fnv_dense_spec(hash, region.port_spec);
        }
    }
    hash = fnv_u64(hash, identity_routes.size());
    hash = fnv_u64(hash, result.permutation_routes_.size());
    for (const SpmPermutationRowRunsRoute& route :
         result.permutation_routes_) {
        hash = fnv_u64(hash, route.hash());
    }
    result.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    return result;
}

SpmPipelinePlan SpmPipelinePlan::compile_phase_overlay(
    const std::vector<SpmComponentPhaseManifest>& manifests,
    const std::vector<SpmChunkedPermutationRoute>& chunked_routes,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    return compile_phase_overlay(
        manifests, chunked_routes,
        std::vector<SpmDirectProducedRowRunsRoute>{},
        std::vector<SpmDirectProducedPayloadRunAddRoute>{},
        reserved_top_bytes, budget_bytes);
}

SpmPipelinePlan SpmPipelinePlan::compile_phase_overlay(
    const std::vector<SpmComponentPhaseManifest>& manifests,
    const std::vector<SpmChunkedPermutationRoute>& chunked_routes,
    const std::vector<SpmDirectProducedRowRunsRoute>& direct_routes,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    return compile_phase_overlay(
        manifests, chunked_routes, direct_routes,
        std::vector<SpmDirectProducedPayloadRunAddRoute>{},
        reserved_top_bytes, budget_bytes);
}

SpmPipelinePlan SpmPipelinePlan::compile_phase_overlay(
    const std::vector<SpmComponentPhaseManifest>& manifests,
    const std::vector<SpmChunkedPermutationRoute>& chunked_routes,
    const std::vector<SpmDirectProducedRowRunsRoute>& direct_routes,
    const std::vector<SpmDirectProducedPayloadRunAddRoute>&
        direct_payload_run_add_routes,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    TORCH_CHECK(budget_bytes > 0,
                "SpmPipelinePlan: budget must be positive");
    TORCH_CHECK(budget_bytes <= SpmAllocator::SPM_PLANNING_BUDGET,
                "SpmPipelinePlan: budget ", budget_bytes,
                " exceeds canonical ceiling ",
                SpmAllocator::SPM_PLANNING_BUDGET);
    TORCH_CHECK(!manifests.empty(),
                "SpmPipelinePlan: phase-overlay manifests must not be empty");
    TORCH_CHECK(!chunked_routes.empty() || !direct_routes.empty() ||
                    !direct_payload_run_add_routes.empty(),
                "SpmPipelinePlan: phase-overlay routes must not be empty");
    TORCH_CHECK(reserved_top_bytes <= budget_bytes,
                "SpmPipelinePlan: reserved_top_bytes ", reserved_top_bytes,
                " exceeds budget ", budget_bytes);

    SpmPipelinePlan result;
    result.budget_bytes_ = budget_bytes;
    result.phase_overlay_admission_ =
        kPhaseOverlaySyntheticCpuAdmission;
    result.reserved_top_bytes_ = align_up(reserved_top_bytes);
    TORCH_CHECK(result.reserved_top_bytes_ <= budget_bytes,
                "SpmPipelinePlan: aligned reserved_top_bytes ",
                result.reserved_top_bytes_, " exceeds budget ", budget_bytes);

    struct PlacedSlice {
        SpmPhaseSlice slice;
        uint64_t absolute_begin = 0;
        uint64_t absolute_end = 0;
    };
    std::vector<PlacedSlice> placed_slices;
    std::unordered_set<uint32_t> arena_ids;
    result.phase_manifests_.reserve(manifests.size());

    for (const SpmComponentPhaseManifest& manifest : manifests) {
        TORCH_CHECK(manifest.arena.value() != 0,
                    "SpmPipelinePlan: phase arena ID zero is reserved");
        TORCH_CHECK(arena_ids.insert(manifest.arena.value()).second,
                    "SpmPipelinePlan: duplicate phase arena ID ",
                    manifest.arena.value());
        TORCH_CHECK(manifest.arena_extent > 0,
                    "SpmPipelinePlan: phase arena ", manifest.arena.value(),
                    " must have positive extent");
        TORCH_CHECK(manifest.layout_hash != 0,
                    "SpmPipelinePlan: phase arena ", manifest.arena.value(),
                    " has an unsealed zero layout hash");
        TORCH_CHECK(!manifest.live_ranges.empty(),
                    "SpmPipelinePlan: phase arena ", manifest.arena.value(),
                    " must declare live ranges; production declarations must "
                    "come from the opaque FusedModelBase exporter");
        TORCH_CHECK(manifest.arena_base % SpmAllocator::ALIGN == 0 &&
                        manifest.arena_extent % SpmAllocator::ALIGN == 0,
                    "SpmPipelinePlan: phase arena base and extent must be "
                    "256-byte aligned");
        TORCH_CHECK(manifest.arena_extent <=
                        std::numeric_limits<uint32_t>::max() -
                            manifest.arena_base,
                    "SpmPipelinePlan: phase arena extent exceeds uint32 range");
        const size_t arena_end =
            static_cast<size_t>(manifest.arena_base) + manifest.arena_extent;
        result.peak_bytes_ = std::max(result.peak_bytes_, arena_end);

        for (const SpmPhaseSlice& slice : manifest.live_ranges) {
            validate_phase_slice(slice, manifest.arena,
                                 manifest.arena_extent,
                                 "manifest live range");
            const uint64_t absolute_begin =
                static_cast<uint64_t>(manifest.arena_base) + slice.offset;
            placed_slices.push_back(
                {slice, absolute_begin, absolute_begin + slice.bytes});
        }
        result.phase_manifests_.push_back(manifest);
    }

    TORCH_CHECK(result.peak_bytes_ <= budget_bytes &&
                    result.reserved_top_bytes_ <=
                        budget_bytes - result.peak_bytes_,
                "SpmPipelinePlan: phase overlay requires ",
                result.total_bytes(), " bytes/core (fixed peak ",
                result.peak_bytes_, " + reserved top ",
                result.reserved_top_bytes_, "), exceeding budget ",
                budget_bytes);

    auto manifest_for = [&](SpmScratchId arena)
        -> const SpmComponentPhaseManifest& {
        for (const SpmComponentPhaseManifest& manifest :
             result.phase_manifests_) {
            if (manifest.arena == arena) return manifest;
        }
        TORCH_CHECK(false, "SpmPipelinePlan: unknown phase arena ID ",
                    arena.value());
    };
    auto require_exact_live_range = [&](const SpmPhaseSlice& slice,
                                        const char* role) {
        const SpmComponentPhaseManifest& manifest =
            manifest_for(slice.arena);
        bool found = false;
        for (const SpmPhaseSlice& live : manifest.live_ranges) {
            if (live == slice) {
                found = true;
                break;
            }
        }
        TORCH_CHECK(found, "SpmPipelinePlan: ", role,
                    " does not exactly match a complete manifest live range");
    };
    auto absolute_bounds = [&](const SpmPhaseSlice& slice) {
        const SpmComponentPhaseManifest& manifest =
            manifest_for(slice.arena);
        const uint64_t begin =
            static_cast<uint64_t>(manifest.arena_base) + slice.offset;
        return std::pair<uint64_t, uint64_t>(begin, begin + slice.bytes);
    };

    std::vector<SpmPhaseSlice> endpoint_slices;
    std::unordered_set<uint32_t> endpoint_port_ids;
    result.chunked_permutation_routes_.reserve(chunked_routes.size());
    for (const SpmChunkedPermutationRoute& route : chunked_routes) {
        TORCH_CHECK(
            endpoint_port_ids.count(route.source_id().value()) == 0 &&
                endpoint_port_ids.count(route.destination_id().value()) == 0,
            "SpmPipelinePlan: phase-overlay endpoint port IDs are plan-wide "
            "unique; multi-consumer routing is not supported in schema-v5");
        endpoint_port_ids.insert(route.source_id().value());
        endpoint_port_ids.insert(route.destination_id().value());
        for (const SpmSourceChunkRef& source : route.source_chunks()) {
            require_exact_live_range(source.backing, "source chunk");
            for (const SpmPhaseSlice& existing : endpoint_slices) {
                TORCH_CHECK(!(existing == source.backing),
                            "SpmPipelinePlan: one manifest live range cannot "
                            "back multiple logical endpoint chunks");
            }
            endpoint_slices.push_back(source.backing);
        }
        for (const SpmBorrowedConsumerChunkRef& consumer :
             route.consumer_chunks()) {
            require_exact_live_range(consumer.backing, "consumer chunk");
            for (const SpmPhaseSlice& existing : endpoint_slices) {
                TORCH_CHECK(!(existing == consumer.backing),
                            "SpmPipelinePlan: one manifest live range cannot "
                            "back multiple logical endpoint chunks");
            }
            endpoint_slices.push_back(consumer.backing);
        }
        for (const SpmRetainedPayloadChunkRef& payload :
             route.payload_chunks()) {
            const SpmComponentPhaseManifest& manifest =
                manifest_for(payload.backing.arena);
            validate_phase_slice(payload.backing, manifest.arena,
                                 manifest.arena_extent,
                                 "retained payload");
            const uint64_t absolute_begin =
                static_cast<uint64_t>(manifest.arena_base) +
                payload.backing.offset;
            placed_slices.push_back(
                {payload.backing, absolute_begin,
                 absolute_begin + payload.backing.bytes});
        }
        for (const SpmSourcePayloadRowRun& run :
             route.source_payload_runs()) {
            const SpmPhaseSlice& source =
                route.source_chunks()[run.source_chunk.ordinal()].backing;
            const SpmPhaseSlice& payload =
                route.payload_chunks()[run.payload_chunk.ordinal()].backing;
            const auto source_bounds = absolute_bounds(source);
            const auto payload_bounds = absolute_bounds(payload);
            TORCH_CHECK(
                !byte_ranges_overlap(source_bounds.first,
                                     source_bounds.second,
                                     payload_bounds.first,
                                     payload_bounds.second),
                "SpmPipelinePlan: source-to-payload transfer endpoints must "
                "be physically disjoint");
        }
        for (const SpmPayloadConsumerRowRun& run :
             route.payload_consumer_runs()) {
            const SpmPhaseSlice& payload =
                route.payload_chunks()[run.payload_chunk.ordinal()].backing;
            const SpmPhaseSlice& consumer =
                route.consumer_chunks()[run.consumer_chunk.ordinal()].backing;
            const auto payload_bounds = absolute_bounds(payload);
            const auto consumer_bounds = absolute_bounds(consumer);
            TORCH_CHECK(
                !byte_ranges_overlap(payload_bounds.first,
                                     payload_bounds.second,
                                     consumer_bounds.first,
                                     consumer_bounds.second),
                "SpmPipelinePlan: payload-to-consumer transfer endpoints "
                "must be physically disjoint");
        }
        result.chunked_permutation_routes_.push_back(route);
    }
    result.direct_produced_row_runs_routes_.reserve(direct_routes.size());
    for (const SpmDirectProducedRowRunsRoute& route : direct_routes) {
        TORCH_CHECK(
            endpoint_port_ids.count(route.producer_id().value()) == 0 &&
                endpoint_port_ids.count(route.destination_id().value()) == 0,
            "SpmPipelinePlan: phase-overlay endpoint port IDs are plan-wide "
            "unique; direct-produced multi-consumer routing is not supported");
        endpoint_port_ids.insert(route.producer_id().value());
        endpoint_port_ids.insert(route.destination_id().value());
        for (const SpmBorrowedConsumerChunkRef& consumer :
             route.consumer_chunks()) {
            require_exact_live_range(
                consumer.backing, "direct-produced consumer chunk");
            for (const SpmPhaseSlice& existing : endpoint_slices) {
                TORCH_CHECK(
                    !(existing == consumer.backing),
                    "SpmPipelinePlan: one manifest live range cannot back "
                    "multiple logical endpoint chunks");
            }
            endpoint_slices.push_back(consumer.backing);
        }
        result.direct_produced_row_runs_routes_.push_back(route);
    }
    result.direct_produced_payload_run_add_routes_.reserve(
        direct_payload_run_add_routes.size());
    for (const SpmDirectProducedPayloadRunAddRoute& route :
         direct_payload_run_add_routes) {
        TORCH_CHECK(
            endpoint_port_ids.count(route.payload_id().value()) == 0 &&
                endpoint_port_ids.count(route.destination_id().value()) == 0,
            "SpmPipelinePlan: phase-overlay endpoint port IDs are plan-wide "
            "unique; direct-produced payload multi-consumer routing is not "
            "supported");
        endpoint_port_ids.insert(route.payload_id().value());
        endpoint_port_ids.insert(route.destination_id().value());
        for (const SpmDirectProducedPayloadRef& payload :
             route.payloads()) {
            const SpmComponentPhaseManifest& manifest =
                manifest_for(payload.backing.arena);
            validate_phase_slice(
                payload.backing, manifest.arena, manifest.arena_extent,
                "direct-produced payload");
            const uint64_t absolute_begin =
                static_cast<uint64_t>(manifest.arena_base) +
                payload.backing.offset;
            placed_slices.push_back(
                {payload.backing, absolute_begin,
                 absolute_begin + payload.backing.bytes});
        }
        for (const SpmBorrowedConsumerChunkRef& consumer :
             route.consumer_chunks()) {
            require_exact_live_range(
                consumer.backing,
                "direct-produced payload RunAdd consumer chunk");
            for (const SpmPhaseSlice& existing : endpoint_slices) {
                TORCH_CHECK(
                    !(existing == consumer.backing),
                    "SpmPipelinePlan: one manifest live range cannot back "
                    "multiple logical endpoint chunks");
            }
            endpoint_slices.push_back(consumer.backing);
        }
        for (const SpmDirectProducedPayloadRunAdd& run_add :
             route.run_adds()) {
            const SpmPhaseSlice& payload =
                route.payloads()[run_add.payload_chunk.ordinal()].backing;
            const SpmPhaseSlice& consumer = route.consumer_chunks()[
                run_add.consumer_chunk.ordinal()].backing;
            const auto payload_bounds = absolute_bounds(payload);
            const auto consumer_bounds = absolute_bounds(consumer);
            TORCH_CHECK(
                !byte_ranges_overlap(
                    payload_bounds.first, payload_bounds.second,
                    consumer_bounds.first, consumer_bounds.second),
                "SpmPipelinePlan: direct-produced payload and RunAdd "
                "consumer endpoints must be physically disjoint");
        }
        result.direct_produced_payload_run_add_routes_.push_back(route);
    }

    for (size_t i = 0; i < placed_slices.size(); ++i) {
        for (size_t j = i + 1; j < placed_slices.size(); ++j) {
            const PlacedSlice& lhs = placed_slices[i];
            const PlacedSlice& rhs = placed_slices[j];
            TORCH_CHECK(
                !byte_ranges_overlap(lhs.absolute_begin, lhs.absolute_end,
                                     rhs.absolute_begin, rhs.absolute_end) ||
                    !event_ranges_overlap(lhs.slice, rhs.slice),
                "SpmPipelinePlan: fixed phase slices overlap in bytes [",
                std::max(lhs.absolute_begin, rhs.absolute_begin), ", ",
                std::min(lhs.absolute_end, rhs.absolute_end),
                ") and inclusive events [",
                std::max(lhs.slice.first_event.value(),
                         rhs.slice.first_event.value()),
                ", ",
                std::min(lhs.slice.last_event.value(),
                         rhs.slice.last_event.value()), "]");
        }
    }

    const uint64_t plan_schema_version =
        !direct_payload_run_add_routes.empty()
        ? kPhaseOverlayDirectProducedPayloadPlanSchemaVersion
        : (direct_routes.empty()
               ? kPhaseOverlayPlanSchemaVersion
               : kPhaseOverlayDirectProducedPlanSchemaVersion);
    uint64_t hash = fnv_u64(kFnvOffset, plan_schema_version);
    hash = fnv_u64(hash, result.phase_overlay_admission_);
    hash = fnv_u64(hash, result.budget_bytes_);
    hash = fnv_u64(hash, result.reserved_top_bytes_);
    hash = fnv_u64(hash, result.peak_bytes_);
    hash = fnv_u64(hash, result.phase_manifests_.size());
    for (const SpmComponentPhaseManifest& manifest :
         result.phase_manifests_) {
        hash = fnv_u64(hash, manifest.arena.value());
        hash = fnv_u64(hash, manifest.arena_base);
        hash = fnv_u64(hash, manifest.arena_extent);
        hash = fnv_u64(hash, manifest.layout_hash);
        hash = fnv_u64(hash, manifest.live_ranges.size());
        for (const SpmPhaseSlice& slice : manifest.live_ranges) {
            hash = fnv_phase_slice(hash, slice);
        }
    }
    hash = fnv_u64(hash, result.chunked_permutation_routes_.size());
    for (const SpmChunkedPermutationRoute& route :
         result.chunked_permutation_routes_) {
        hash = fnv_u64(hash, route.hash());
    }
    if (!direct_routes.empty() || !direct_payload_run_add_routes.empty()) {
        hash = fnv_u64(
            hash, result.direct_produced_row_runs_routes_.size());
        for (const SpmDirectProducedRowRunsRoute& route :
             result.direct_produced_row_runs_routes_) {
            hash = fnv_u64(hash, route.hash());
        }
    }
    if (!direct_payload_run_add_routes.empty()) {
        hash = fnv_u64(
            hash, result.direct_produced_payload_run_add_routes_.size());
        for (const SpmDirectProducedPayloadRunAddRoute& route :
             result.direct_produced_payload_run_add_routes_) {
            hash = fnv_u64(hash, route.hash());
        }
    }
    result.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    return result;
}

SpmPipelinePlan SpmPipelinePlan::compile_fmb_sealed_phase_overlay(
    const std::vector<SpmFmbSealedPhaseManifest>& sealed_manifests,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    TORCH_CHECK(budget_bytes > 0,
                "SpmPipelinePlan: budget must be positive");
    TORCH_CHECK(budget_bytes <= SpmAllocator::SPM_PLANNING_BUDGET,
                "SpmPipelinePlan: budget ", budget_bytes,
                " exceeds canonical ceiling ",
                SpmAllocator::SPM_PLANNING_BUDGET);
    TORCH_CHECK(!sealed_manifests.empty(),
                "SpmPipelinePlan: FMB-sealed manifests must not be empty");
    TORCH_CHECK(reserved_top_bytes <= budget_bytes,
                "SpmPipelinePlan: reserved_top_bytes ", reserved_top_bytes,
                " exceeds budget ", budget_bytes);

    SpmPipelinePlan result;
    result.budget_bytes_ = budget_bytes;
    result.phase_overlay_admission_ =
        kPhaseOverlayFmbSealedCpuDryAdmission;
    result.reserved_top_bytes_ = align_up(reserved_top_bytes);
    TORCH_CHECK(result.reserved_top_bytes_ <= budget_bytes,
                "SpmPipelinePlan: aligned reserved_top_bytes ",
                result.reserved_top_bytes_, " exceeds budget ", budget_bytes);

    struct PlacedSlice {
        SpmPhaseSlice slice;
        uint64_t absolute_begin = 0;
        uint64_t absolute_end = 0;
    };
    std::vector<PlacedSlice> placed_slices;
    std::vector<uint64_t> sealed_identity_hashes;
    std::unordered_set<uint32_t> arena_ids;
    result.phase_manifests_.reserve(sealed_manifests.size());
    result.phase_seal_guards_.reserve(sealed_manifests.size());
    sealed_identity_hashes.reserve(sealed_manifests.size());

    for (const SpmFmbSealedPhaseManifest& sealed : sealed_manifests) {
        const SpmComponentPhaseManifest& manifest = sealed.manifest_;
        TORCH_CHECK(manifest.arena.value() != 0,
                    "SpmPipelinePlan: sealed arena ID zero is reserved");
        TORCH_CHECK(arena_ids.insert(manifest.arena.value()).second,
                    "SpmPipelinePlan: duplicate sealed arena ID ",
                    manifest.arena.value());
        TORCH_CHECK(sealed.required_extent_ > 0 &&
                        sealed.required_extent_ == manifest.arena_extent,
                    "SpmPipelinePlan: sealed arena extent mismatch");
        TORCH_CHECK(manifest.arena_base % SpmAllocator::ALIGN == 0 &&
                        manifest.arena_extent % SpmAllocator::ALIGN == 0,
                    "SpmPipelinePlan: sealed arena base and extent must be "
                    "256-byte aligned");
        TORCH_CHECK(manifest.arena_extent <=
                        std::numeric_limits<uint32_t>::max() -
                            manifest.arena_base,
                    "SpmPipelinePlan: sealed arena exceeds uint32 range");
        TORCH_CHECK(sealed.layout_hash_ != 0 &&
                        sealed.declaration_hash_ != 0 &&
                        sealed.persistent_hash_ != 0 &&
                        sealed.allocation_hash_ != 0 &&
                        sealed.schedule_hash_ != 0 &&
                        sealed.range_hash_ != 0 &&
                        sealed.identity_hash_ != 0,
                    "SpmPipelinePlan: sealed manifest contains a zero hash");
        TORCH_CHECK(sealed.allocation_count_ > 0 &&
                        !manifest.live_ranges.empty(),
                    "SpmPipelinePlan: sealed manifest has an empty allocation "
                    "or range table");

        uint64_t range_hash = fnv_u64(kFnvOffset, 5);
        range_hash = fnv_u64(range_hash, manifest.live_ranges.size());
        for (const SpmPhaseSlice& slice : manifest.live_ranges) {
            validate_phase_slice(slice, manifest.arena,
                                 manifest.arena_extent,
                                 "FMB-sealed live range");
            range_hash = fnv_u64(range_hash, slice.offset);
            range_hash = fnv_u64(range_hash, slice.bytes);
            range_hash = fnv_u64(range_hash,
                                 slice.first_event.value());
            range_hash = fnv_u64(range_hash,
                                 slice.last_event.value());
            const uint64_t absolute_begin =
                static_cast<uint64_t>(manifest.arena_base) + slice.offset;
            placed_slices.push_back(
                {slice, absolute_begin, absolute_begin + slice.bytes});
        }
        if (range_hash == 0) range_hash = kNonzeroHashFallback;
        TORCH_CHECK(range_hash == sealed.range_hash_,
                    "SpmPipelinePlan: sealed range hash mismatch");

        uint64_t identity_hash = fnv_u64(kFnvOffset, 6);
        identity_hash = fnv_u64(identity_hash, manifest.arena.value());
        identity_hash = fnv_u64(identity_hash, manifest.arena_base);
        identity_hash = fnv_u64(identity_hash, sealed.required_extent_);
        identity_hash = fnv_u64(identity_hash, sealed.layout_hash_);
        identity_hash = fnv_u64(identity_hash,
                                sealed.declaration_hash_);
        identity_hash = fnv_u64(identity_hash, sealed.persistent_hash_);
        identity_hash = fnv_u64(identity_hash, sealed.allocation_hash_);
        identity_hash = fnv_u64(identity_hash, sealed.schedule_hash_);
        identity_hash = fnv_u64(identity_hash, sealed.range_hash_);
        identity_hash = fnv_u64(identity_hash, sealed.allocation_count_);
        identity_hash = fnv_u64(identity_hash,
                                manifest.live_ranges.size());
        identity_hash = fnv_u64(identity_hash, sealed.cpu_dry_);
        if (identity_hash == 0) identity_hash = kNonzeroHashFallback;
        TORCH_CHECK(identity_hash == sealed.identity_hash_ &&
                        manifest.layout_hash == sealed.identity_hash_,
                    "SpmPipelinePlan: sealed manifest identity mismatch");

        result.peak_bytes_ = std::max(
            result.peak_bytes_,
            static_cast<size_t>(manifest.arena_base) +
                manifest.arena_extent);
        result.phase_manifests_.push_back(manifest);
        result.phase_seal_guards_.push_back(
            {sealed.owner_generation_,
             sealed.expected_owner_generation_});
        sealed_identity_hashes.push_back(sealed.identity_hash_);
    }

    for (size_t lhs = 0; lhs < placed_slices.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < placed_slices.size(); ++rhs) {
            TORCH_CHECK(
                !byte_ranges_overlap(placed_slices[lhs].absolute_begin,
                                     placed_slices[lhs].absolute_end,
                                     placed_slices[rhs].absolute_begin,
                                     placed_slices[rhs].absolute_end) ||
                    !event_ranges_overlap(placed_slices[lhs].slice,
                                          placed_slices[rhs].slice),
                "SpmPipelinePlan: FMB-sealed ranges overlap in byte x "
                "global-event space");
        }
    }
    TORCH_CHECK(result.peak_bytes_ <= budget_bytes &&
                    result.reserved_top_bytes_ <=
                        budget_bytes - result.peak_bytes_,
                "SpmPipelinePlan: FMB-sealed overlay requires ",
                result.total_bytes(), " bytes/core, exceeding budget ",
                budget_bytes);

    uint64_t hash = fnv_u64(kFnvOffset,
                            kFmbSealedOverlayPlanSchemaVersion);
    hash = fnv_u64(hash, result.phase_overlay_admission_);
    hash = fnv_u64(hash, result.budget_bytes_);
    hash = fnv_u64(hash, result.reserved_top_bytes_);
    hash = fnv_u64(hash, result.peak_bytes_);
    hash = fnv_u64(hash, sealed_identity_hashes.size());
    for (uint64_t identity_hash : sealed_identity_hashes) {
        hash = fnv_u64(hash, identity_hash);
    }
    result.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    result.validate_phase_seals();
    return result;
}

SpmPipelinePlan SpmPipelinePlan::compile_fmb_resolved_phase_overlay(
    const std::vector<SpmFmbResolvedPhaseManifest>& resolved_manifests,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    TORCH_CHECK(budget_bytes > 0,
                "SpmPipelinePlan: budget must be positive");
    TORCH_CHECK(budget_bytes <= SpmAllocator::SPM_PLANNING_BUDGET,
                "SpmPipelinePlan: budget ", budget_bytes,
                " exceeds canonical ceiling ",
                SpmAllocator::SPM_PLANNING_BUDGET);
    TORCH_CHECK(!resolved_manifests.empty(),
                "SpmPipelinePlan: FMB-resolved manifests must not be empty");
    TORCH_CHECK(reserved_top_bytes <= budget_bytes,
                "SpmPipelinePlan: reserved_top_bytes ", reserved_top_bytes,
                " exceeds budget ", budget_bytes);

    SpmPipelinePlan result;
    result.budget_bytes_ = budget_bytes;
    result.phase_overlay_admission_ =
        kPhaseOverlayFmbResolvedCpuDryAdmission;
    result.reserved_top_bytes_ = align_up(reserved_top_bytes);
    TORCH_CHECK(result.reserved_top_bytes_ <= budget_bytes,
                "SpmPipelinePlan: aligned reserved_top_bytes ",
                result.reserved_top_bytes_, " exceeds budget ", budget_bytes);

    struct PlacedSlice {
        SpmPhaseSlice slice;
        uint64_t absolute_begin = 0;
        uint64_t absolute_end = 0;
    };
    std::vector<PlacedSlice> placed_slices;
    std::vector<uint64_t> resolved_identity_hashes;
    std::unordered_set<uint32_t> arena_ids;
    result.phase_manifests_.reserve(resolved_manifests.size());
    result.phase_seal_guards_.reserve(resolved_manifests.size());
    resolved_identity_hashes.reserve(resolved_manifests.size());

    for (const SpmFmbResolvedPhaseManifest& resolved :
         resolved_manifests) {
        TORCH_CHECK(
            resolved.cpu_dry_,
            "SpmPipelinePlan: CPU resolved overlay rejects a live manifest; "
            "physical admission requires opaque consumer and producer-yield "
            "authority");
        const SpmComponentPhaseManifest& manifest = resolved.manifest_;
        TORCH_CHECK(manifest.arena.value() != 0,
                    "SpmPipelinePlan: resolved arena ID zero is reserved");
        TORCH_CHECK(arena_ids.insert(manifest.arena.value()).second,
                    "SpmPipelinePlan: duplicate resolved arena ID ",
                    manifest.arena.value());
        TORCH_CHECK(resolved.required_extent_ > 0 &&
                        resolved.required_extent_ == manifest.arena_extent,
                    "SpmPipelinePlan: resolved arena extent mismatch");
        TORCH_CHECK(manifest.arena_base % SpmAllocator::ALIGN == 0 &&
                        manifest.arena_extent % SpmAllocator::ALIGN == 0,
                    "SpmPipelinePlan: resolved arena base and extent must be "
                    "256-byte aligned");
        TORCH_CHECK(manifest.arena_extent <=
                        std::numeric_limits<uint32_t>::max() -
                            manifest.arena_base,
                    "SpmPipelinePlan: resolved arena exceeds uint32 range");
        TORCH_CHECK(resolved.layout_hash_ != 0 &&
                        resolved.declaration_hash_ != 0 &&
                        resolved.persistent_hash_ != 0 &&
                        resolved.allocation_hash_ != 0 &&
                        resolved.profile_hash_ != 0 &&
                        resolved.range_hash_ != 0 &&
                        resolved.identity_hash_ != 0,
                    "SpmPipelinePlan: resolved manifest contains a zero "
                    "hash");
        TORCH_CHECK(resolved.allocation_count_ > 0 &&
                        !manifest.live_ranges.empty(),
                    "SpmPipelinePlan: resolved manifest has an empty "
                    "allocation or range table");

        uint64_t range_hash = fnv_u64(kFnvOffset, 7);
        range_hash = fnv_u64(range_hash, manifest.live_ranges.size());
        for (const SpmPhaseSlice& slice : manifest.live_ranges) {
            validate_phase_slice(slice, manifest.arena,
                                 manifest.arena_extent,
                                 "FMB-resolved live range");
            range_hash = fnv_u64(range_hash, slice.offset);
            range_hash = fnv_u64(range_hash, slice.bytes);
            range_hash = fnv_u64(range_hash,
                                 slice.first_event.value());
            range_hash = fnv_u64(range_hash,
                                 slice.last_event.value());
            const uint64_t absolute_begin =
                static_cast<uint64_t>(manifest.arena_base) + slice.offset;
            placed_slices.push_back(
                {slice, absolute_begin, absolute_begin + slice.bytes});
        }
        if (range_hash == 0) range_hash = kNonzeroHashFallback;
        TORCH_CHECK(range_hash == resolved.range_hash_,
                    "SpmPipelinePlan: resolved range hash mismatch");

        uint64_t identity_hash = fnv_u64(kFnvOffset, 8);
        identity_hash = fnv_u64(identity_hash, manifest.arena.value());
        identity_hash = fnv_u64(identity_hash, manifest.arena_base);
        identity_hash = fnv_u64(identity_hash, resolved.required_extent_);
        identity_hash = fnv_u64(identity_hash, resolved.layout_hash_);
        identity_hash = fnv_u64(identity_hash,
                                resolved.declaration_hash_);
        identity_hash = fnv_u64(identity_hash,
                                resolved.persistent_hash_);
        identity_hash = fnv_u64(identity_hash,
                                resolved.allocation_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.profile_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.range_hash_);
        identity_hash = fnv_u64(identity_hash,
                                resolved.allocation_count_);
        identity_hash = fnv_u64(identity_hash,
                                manifest.live_ranges.size());
        if (identity_hash == 0) identity_hash = kNonzeroHashFallback;
        TORCH_CHECK(identity_hash == resolved.identity_hash_ &&
                        manifest.layout_hash == resolved.identity_hash_,
                    "SpmPipelinePlan: resolved manifest identity mismatch");

        result.peak_bytes_ = std::max(
            result.peak_bytes_,
            static_cast<size_t>(manifest.arena_base) +
                manifest.arena_extent);
        result.phase_manifests_.push_back(manifest);
        result.phase_seal_guards_.push_back(
            {resolved.owner_generation_,
             resolved.expected_owner_generation_});
        resolved_identity_hashes.push_back(resolved.identity_hash_);
    }

    for (size_t lhs = 0; lhs < placed_slices.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < placed_slices.size(); ++rhs) {
            TORCH_CHECK(
                !byte_ranges_overlap(placed_slices[lhs].absolute_begin,
                                     placed_slices[lhs].absolute_end,
                                     placed_slices[rhs].absolute_begin,
                                     placed_slices[rhs].absolute_end) ||
                    !event_ranges_overlap(placed_slices[lhs].slice,
                                          placed_slices[rhs].slice),
                "SpmPipelinePlan: FMB-resolved ranges overlap in byte x "
                "global-event space");
        }
    }
    TORCH_CHECK(result.peak_bytes_ <= budget_bytes &&
                    result.reserved_top_bytes_ <=
                        budget_bytes - result.peak_bytes_,
                "SpmPipelinePlan: FMB-resolved overlay requires ",
                result.total_bytes(), " bytes/core, exceeding budget ",
                budget_bytes);

    uint64_t hash = fnv_u64(kFnvOffset,
                            kFmbResolvedOverlayPlanSchemaVersion);
    hash = fnv_u64(hash, result.phase_overlay_admission_);
    hash = fnv_u64(hash, result.budget_bytes_);
    hash = fnv_u64(hash, result.reserved_top_bytes_);
    hash = fnv_u64(hash, result.peak_bytes_);
    hash = fnv_u64(hash, resolved_identity_hashes.size());
    for (uint64_t identity_hash : resolved_identity_hashes) {
        hash = fnv_u64(hash, identity_hash);
    }
    result.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    result.validate_phase_seals();
    return result;
}

SpmFmbCompositePhysicalReservation
SpmFmbCompositePhysicalReservation::compile_repeated_dense_producer(
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
    size_t budget_bytes) {
    TORCH_CHECK(budget_bytes > 0 &&
                    budget_bytes <= SpmAllocator::SPM_PLANNING_BUDGET,
                "composite physical reservation budget must be in (0, ",
                SpmAllocator::SPM_PLANNING_BUDGET, "]");
    const size_t effective_budget =
        budget_bytes & ~(SpmAllocator::ALIGN - 1);
    const size_t aligned_reserved_top = align_up(reserved_top_bytes);
    TORCH_CHECK(aligned_reserved_top <= effective_budget,
                "composite physical reservation persistent top exceeds the "
                "aligned budget");
    produced_spec.validate();
    TORCH_CHECK(
        produced_spec.dtype == SpmPortDType::Fp16 &&
            produced_spec.distribution == SpmPortDistribution::Replicated,
        "composite physical reservation v1 requires dense replicated FP16");
    TORCH_CHECK(!ordered_direct_endpoints.empty() &&
                    yields_per_producer > 1 && policy_version != 0,
                "composite physical reservation requires producers, at "
                "least two yields per producer, and a nonzero policy");
    const size_t producer_count = ordered_direct_endpoints.size();
    const size_t retained_tap_count = yields_per_producer - 1;
    TORCH_CHECK(
        retained_tap_count <= std::numeric_limits<size_t>::max() /
                                  producer_count,
        "composite physical reservation RunAdd count overflows size_t");
    const size_t expected_run_add_count =
        retained_tap_count * producer_count;
    TORCH_CHECK(
        tap_major_run_add_endpoints.size() == expected_run_add_count,
        "composite physical reservation requires an exact tap-major "
        "RunAdd endpoint cover: expected ", expected_run_add_count,
        ", got ", tap_major_run_add_endpoints.size());

    auto validate_manifest = [](const SpmFmbResolvedPhaseManifest& resolved,
                                const char* label) {
        const auto& manifest = resolved.manifest_;
        TORCH_CHECK(
            !resolved.cpu_dry_ && manifest.arena.value() != 0 &&
                resolved.required_extent_ > 0 &&
                resolved.required_extent_ == manifest.arena_extent &&
                manifest.arena_base % SpmAllocator::ALIGN == 0 &&
                manifest.arena_extent % SpmAllocator::ALIGN == 0 &&
                manifest.arena_extent <=
                    std::numeric_limits<uint32_t>::max() -
                        manifest.arena_base &&
                resolved.layout_hash_ != 0 &&
                resolved.declaration_hash_ != 0 &&
                resolved.persistent_hash_ != 0 &&
                resolved.allocation_hash_ != 0 &&
                resolved.profile_hash_ != 0 && resolved.range_hash_ != 0 &&
                resolved.identity_hash_ != 0 &&
                resolved.allocation_count_ > 0 &&
                !manifest.live_ranges.empty(),
            "composite physical reservation ", label,
            " manifest is CPU-dry, empty, unaligned, or malformed");
        const std::shared_ptr<const uint64_t> owner =
            resolved.owner_generation_.lock();
        TORCH_CHECK(owner != nullptr &&
                        *owner == resolved.expected_owner_generation_,
                    "composite physical reservation ", label,
                    " manifest owner is stale");

        uint64_t range_hash = fnv_u64(kFnvOffset, 7);
        range_hash = fnv_u64(range_hash, manifest.live_ranges.size());
        for (const SpmPhaseSlice& slice : manifest.live_ranges) {
            validate_phase_slice(slice, manifest.arena,
                                 manifest.arena_extent, label);
            range_hash = fnv_u64(range_hash, slice.offset);
            range_hash = fnv_u64(range_hash, slice.bytes);
            range_hash = fnv_u64(range_hash, slice.first_event.value());
            range_hash = fnv_u64(range_hash, slice.last_event.value());
        }
        if (range_hash == 0) range_hash = kNonzeroHashFallback;
        TORCH_CHECK(range_hash == resolved.range_hash_,
                    "composite physical reservation ", label,
                    " manifest range hash drifted");

        uint64_t identity_hash = fnv_u64(kFnvOffset, 8);
        identity_hash = fnv_u64(identity_hash, manifest.arena.value());
        identity_hash = fnv_u64(identity_hash, manifest.arena_base);
        identity_hash = fnv_u64(identity_hash, resolved.required_extent_);
        identity_hash = fnv_u64(identity_hash, resolved.layout_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.declaration_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.persistent_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.allocation_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.profile_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.range_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.allocation_count_);
        identity_hash = fnv_u64(identity_hash,
                                manifest.live_ranges.size());
        if (identity_hash == 0) identity_hash = kNonzeroHashFallback;
        TORCH_CHECK(identity_hash == resolved.identity_hash_ &&
                        manifest.layout_hash == resolved.identity_hash_,
                    "composite physical reservation ", label,
                    " manifest identity drifted");
        return owner;
    };

    const std::shared_ptr<const uint64_t> producer_manifest_owner =
        validate_manifest(producer_manifest, "producer");
    const std::shared_ptr<const uint64_t> consumer_manifest_owner =
        validate_manifest(consumer_manifest, "consumer");
    TORCH_CHECK(
        producer_manifest.manifest_.arena !=
            consumer_manifest.manifest_.arena,
        "composite physical reservation producer and consumer arena IDs "
        "must differ");

    auto validate_endpoint = [&](
                                 const SpmFmbConsumerRowSliceEndpoint& endpoint,
                                 const char* label) {
        const std::shared_ptr<const uint64_t> owner =
            endpoint.manifest_.owner_generation_.lock();
        TORCH_CHECK(
            owner != nullptr &&
                owner.get() == consumer_manifest_owner.get() &&
                *owner == endpoint.manifest_.expected_owner_generation_ &&
                endpoint.manifest_.identity_hash_ ==
                    consumer_manifest.identity_hash_ &&
                endpoint.manifest_.profile_hash_ ==
                    consumer_manifest.profile_hash_ &&
                endpoint.manifest_.allocation_hash_ ==
                    consumer_manifest.allocation_hash_ &&
                endpoint.identity_hash_ != 0,
            "composite physical reservation ", label,
            " endpoint does not belong to the consumer manifest");
    };
    (void)producer_manifest_owner;

    const auto& first_direct = ordered_direct_endpoints.front();
    validate_endpoint(first_direct, "direct");
    const auto& consumer_root = first_direct.consumer_.backing;
    const auto& consumer_storage = first_direct.consumer_.spec;
    consumer_storage.validate();
    TORCH_CHECK(
        consumer_manifest.manifest_.arena_base == 0 &&
            consumer_root.arena == consumer_manifest.manifest_.arena &&
            consumer_root.bytes == consumer_storage.storage_bytes() &&
            consumer_root.offset <= consumer_manifest.required_extent_ &&
            consumer_root.bytes <=
                consumer_manifest.required_extent_ - consumer_root.offset,
        "composite physical reservation requires one zero-based, "
        "in-manifest consumer root");
    TORCH_CHECK(
        consumer_storage.dtype == produced_spec.dtype &&
            consumer_storage.cols == produced_spec.cols &&
            consumer_storage.distribution == produced_spec.distribution,
        "composite physical reservation producer and consumer row formats "
        "must match exactly");
    const size_t row_bytes =
        consumer_storage.storage_bytes() /
        static_cast<size_t>(consumer_storage.rows);
    TORCH_CHECK(row_bytes > 0 && row_bytes % SpmAllocator::ALIGN == 0,
                "composite physical reservation consumer rows must be "
                "256-byte aligned");
    const size_t consumer_root_begin =
        static_cast<size_t>(consumer_manifest.manifest_.arena_base) +
        consumer_root.offset;
    TORCH_CHECK(
        consumer_root_begin <= std::numeric_limits<size_t>::max() -
                                   consumer_root.bytes,
        "composite physical reservation consumer root end overflows size_t");
    const size_t consumer_root_end = consumer_root_begin + consumer_root.bytes;
    TORCH_CHECK(
        producer_manifest.manifest_.arena_base == align_up(consumer_root_end),
        "composite physical reservation producer arena must start "
        "immediately after the protected consumer root");

    std::vector<DirectTarget> direct_targets;
    direct_targets.reserve(producer_count);
    int64_t previous_direct_row_end = 0;
    for (size_t image = 0; image < producer_count; ++image) {
        const auto& endpoint = ordered_direct_endpoints[image];
        validate_endpoint(endpoint, "direct");
        TORCH_CHECK(
            endpoint.consumer_.backing == consumer_root &&
                endpoint.consumer_.spec == consumer_storage &&
                endpoint.root_aligned_capacity_ == consumer_root.bytes &&
                endpoint.row_begin_ >= previous_direct_row_end &&
                endpoint.row_count_ == produced_spec.rows,
            "composite physical reservation direct endpoints must be "
            "ordered non-overlapping equal-shape slices of one root");
        const size_t row_begin = static_cast<size_t>(endpoint.row_begin_);
        const size_t row_count = static_cast<size_t>(endpoint.row_count_);
        TORCH_CHECK(
            row_begin <= static_cast<size_t>(consumer_storage.rows) &&
                row_count <=
                    static_cast<size_t>(consumer_storage.rows) - row_begin &&
                row_begin <= std::numeric_limits<size_t>::max() / row_bytes &&
                row_count <= std::numeric_limits<size_t>::max() / row_bytes,
            "composite physical reservation direct endpoint row geometry "
            "overflows");
        const size_t target_offset = consumer_root_begin + row_begin * row_bytes;
        const size_t target_bytes = row_count * row_bytes;
        TORCH_CHECK(
            target_offset <= std::numeric_limits<uint32_t>::max() &&
                target_bytes == produced_spec.storage_bytes() &&
                target_offset <= consumer_root_end &&
                target_bytes <= consumer_root_end - target_offset,
            "composite physical reservation direct target exceeds its "
            "consumer root");
        direct_targets.push_back(
            {endpoint.identity_hash_, image, yields_per_producer - 1,
             static_cast<uint32_t>(target_offset), target_bytes});
        previous_direct_row_end = endpoint.row_begin_ + endpoint.row_count_;
    }

    size_t component_peak = std::max(
        static_cast<size_t>(consumer_manifest.manifest_.arena_base) +
            consumer_manifest.required_extent_,
        static_cast<size_t>(producer_manifest.manifest_.arena_base) +
            producer_manifest.required_extent_);
    component_peak = align_up(component_peak);
    size_t payload_cursor = component_peak;
    std::vector<RunAddTarget> run_add_targets;
    run_add_targets.reserve(expected_run_add_count);
    int previous_consumer_layer = std::numeric_limits<int>::min();
    uint32_t previous_consumer_event = 0;
    bool have_previous_tap = false;
    for (size_t tap = 0; tap < retained_tap_count; ++tap) {
        const auto& first_endpoint =
            tap_major_run_add_endpoints[tap * producer_count];
        validate_endpoint(first_endpoint, "RunAdd");
        if (have_previous_tap) {
            TORCH_CHECK(
                std::make_pair(first_endpoint.global_first_event_.value(),
                               first_endpoint.occurrence_layer_) >
                    std::make_pair(previous_consumer_event,
                                   previous_consumer_layer),
                "composite physical reservation RunAdd taps must follow "
                "consumer callback event/layer order");
        }
        previous_consumer_event = first_endpoint.global_first_event_.value();
        previous_consumer_layer = first_endpoint.occurrence_layer_;
        have_previous_tap = true;

        for (size_t image = 0; image < producer_count; ++image) {
            const size_t payload_index = tap * producer_count + image;
            const auto& endpoint =
                tap_major_run_add_endpoints[payload_index];
            validate_endpoint(endpoint, "RunAdd");
            const auto& direct_endpoint = ordered_direct_endpoints[image];
            TORCH_CHECK(
                endpoint.shares_owned_root_storage_with(first_direct) &&
                    endpoint.consumer_.backing ==
                        first_endpoint.consumer_.backing &&
                    endpoint.consumer_.spec == consumer_storage &&
                    endpoint.root_aligned_capacity_ == consumer_root.bytes &&
                    endpoint.row_begin_ == direct_endpoint.row_begin_ &&
                    endpoint.row_count_ == direct_endpoint.row_count_ &&
                    endpoint.occurrence_kind_ ==
                        first_endpoint.occurrence_kind_ &&
                    endpoint.occurrence_key_ ==
                        first_endpoint.occurrence_key_ &&
                    endpoint.body_id_ == first_endpoint.body_id_ &&
                    endpoint.group_id_ == first_endpoint.group_id_ &&
                    endpoint.occurrence_layer_ ==
                        first_endpoint.occurrence_layer_ &&
                    endpoint.traversal_ordinal_ ==
                        first_endpoint.traversal_ordinal_ &&
                    endpoint.chunk_index_ == first_endpoint.chunk_index_ &&
                    endpoint.global_first_event_ ==
                        first_endpoint.global_first_event_ &&
                    endpoint.global_last_event_ ==
                        first_endpoint.global_last_event_,
                "composite physical reservation RunAdd tap endpoints must "
                "share one callback and preserve image rows");

            const size_t payload_bytes = produced_spec.storage_bytes();
            payload_cursor = align_up(payload_cursor);
            TORCH_CHECK(
                payload_cursor <= std::numeric_limits<uint32_t>::max() &&
                    payload_bytes <=
                        std::numeric_limits<size_t>::max() - payload_cursor,
                "composite physical reservation retained payload overflows "
                "placement");
            const size_t consumer_offset =
                consumer_root_begin +
                static_cast<size_t>(endpoint.row_begin_) * row_bytes;
            TORCH_CHECK(
                consumer_offset <= std::numeric_limits<uint32_t>::max(),
                "composite physical reservation RunAdd consumer offset "
                "exceeds uint32");
            run_add_targets.push_back(
                {endpoint.identity_hash_, image, tap, payload_index,
                 static_cast<uint32_t>(payload_cursor),
                 static_cast<uint32_t>(consumer_offset), payload_bytes,
                 endpoint.occurrence_layer_, endpoint.row_begin_,
                 endpoint.row_count_, endpoint.consumer_.spec});
            payload_cursor += payload_bytes;
        }
    }
    const size_t dynamic_bytes = align_up(payload_cursor);
    TORCH_CHECK(
        dynamic_bytes <= effective_budget &&
            aligned_reserved_top <= effective_budget - dynamic_bytes,
        "composite physical reservation requires ",
        dynamic_bytes + aligned_reserved_top,
        " bytes/core, exceeding aligned budget ", effective_budget);

    SpmPipelinePlan plan;
    plan.phase_overlay_admission_ =
        kPhaseOverlayPhysicalCompositeAdmission;
    plan.budget_bytes_ = budget_bytes;
    plan.reserved_top_bytes_ = aligned_reserved_top;
    plan.peak_bytes_ = dynamic_bytes;
    plan.phase_manifests_ = {consumer_manifest.manifest_,
                             producer_manifest.manifest_};
    plan.phase_seal_guards_ = {
        {consumer_manifest.owner_generation_,
         consumer_manifest.expected_owner_generation_},
        {producer_manifest.owner_generation_,
         producer_manifest.expected_owner_generation_},
    };

    uint64_t hash = fnv_u64(
        kFnvOffset, kFmbCompositePhysicalReservationSchemaVersion);
    hash = fnv_u64(hash, plan.phase_overlay_admission_);
    hash = fnv_u64(hash, plan.budget_bytes_);
    hash = fnv_u64(hash, plan.reserved_top_bytes_);
    hash = fnv_u64(hash, plan.peak_bytes_);
    hash = fnv_u64(hash, producer_manifest.identity_hash_);
    hash = fnv_u64(hash, consumer_manifest.identity_hash_);
    hash = fnv_dense_spec(hash, produced_spec);
    hash = fnv_u64(hash, producer_count);
    hash = fnv_u64(hash, yields_per_producer);
    hash = fnv_u64(hash, policy_version);
    hash = fnv_u64(hash, direct_targets.size());
    for (const auto& target : direct_targets) {
        hash = fnv_u64(hash, target.endpoint_hash);
        hash = fnv_u64(hash, target.producer_local_ordinal);
        hash = fnv_u64(hash, target.producer_yield_ordinal);
        hash = fnv_u64(hash, target.target_relative_offset);
        hash = fnv_u64(hash, target.bytes);
    }
    hash = fnv_u64(hash, run_add_targets.size());
    for (const auto& target : run_add_targets) {
        hash = fnv_u64(hash, target.endpoint_hash);
        hash = fnv_u64(hash, target.producer_local_ordinal);
        hash = fnv_u64(hash, target.producer_yield_ordinal);
        hash = fnv_u64(hash, target.payload_index);
        hash = fnv_u64(hash, target.payload_relative_offset);
        hash = fnv_u64(hash, target.consumer_relative_offset);
        hash = fnv_u64(hash, target.bytes);
        hash = fnv_u64(hash, static_cast<uint64_t>(target.consumer_layer));
        hash = fnv_u64(hash,
                       static_cast<uint64_t>(target.consumer_row_begin));
        hash = fnv_u64(hash,
                       static_cast<uint64_t>(target.consumer_row_count));
        hash = fnv_dense_spec(hash, target.spec);
    }
    plan.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    plan.validate_phase_seals();

    return SpmFmbCompositePhysicalReservation(
        producer_manifest, consumer_manifest, std::move(plan),
        producer_count, yields_per_producer, policy_version, produced_spec,
        std::move(direct_targets), std::move(run_add_targets));
}

void SpmFmbCompositePhysicalReservation::validate_live_authority() const {
    plan_.validate_phase_seals();
    produced_spec_.validate();
    TORCH_CHECK(
        plan_.phase_overlay_admission_ ==
                kPhaseOverlayPhysicalCompositeAdmission &&
            plan_.hash_ != 0 && producer_occurrence_count_ > 0 &&
            yields_per_producer_ > 1 && policy_version_ != 0 &&
            produced_spec_.dtype == SpmPortDType::Fp16 &&
            produced_spec_.distribution ==
                SpmPortDistribution::Replicated &&
            producer_occurrence_count_ <=
                std::numeric_limits<size_t>::max() /
                    yields_per_producer_ &&
            direct_targets_.size() == producer_occurrence_count_ &&
            run_add_targets_.size() ==
                producer_occurrence_count_ * (yields_per_producer_ - 1) &&
            plan_.total_bytes() <= plan_.effective_budget_bytes(),
        "composite physical reservation retained placement is malformed");
    const std::shared_ptr<const uint64_t> producer_owner =
        producer_manifest_.owner_generation_.lock();
    const std::shared_ptr<const uint64_t> consumer_owner =
        consumer_manifest_.owner_generation_.lock();
    TORCH_CHECK(
        producer_owner != nullptr && consumer_owner != nullptr &&
            *producer_owner ==
                producer_manifest_.expected_owner_generation_ &&
            *consumer_owner ==
                consumer_manifest_.expected_owner_generation_,
        "composite physical reservation component owner is stale");
}

SpmFmbCompositePhysicalPlan SpmFmbCompositePhysicalPlan::compile(
    const SpmFmbSealedCompositeTrace& composite,
    const SpmFmbResolvedPhaseManifest& producer_manifest,
    const SpmFmbResolvedPhaseManifest& consumer_manifest,
    const std::vector<SpmFmbCompositeDirectProducedBinding>& direct_bindings,
    const std::vector<SpmFmbCompositeRunAddBinding>& run_add_bindings,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    TORCH_CHECK(budget_bytes > 0 &&
                    budget_bytes <= SpmAllocator::SPM_PLANNING_BUDGET,
                "composite physical plan budget must be in (0, ",
                SpmAllocator::SPM_PLANNING_BUDGET, "]");
    const size_t effective_budget =
        budget_bytes & ~(SpmAllocator::ALIGN - 1);
    const size_t aligned_reserved_top = align_up(reserved_top_bytes);
    TORCH_CHECK(aligned_reserved_top <= effective_budget,
                "composite physical plan persistent top exceeds the aligned "
                "budget");
    TORCH_CHECK(!producer_manifest.cpu_dry_ && !consumer_manifest.cpu_dry_,
                "composite physical plan requires live producer and consumer "
                "manifests");
    TORCH_CHECK(!direct_bindings.empty(),
                "composite physical plan requires direct-produced bindings");

    auto validate_manifest = [](const SpmFmbResolvedPhaseManifest& resolved,
                                const char* label) {
        const auto& manifest = resolved.manifest_;
        TORCH_CHECK(
            manifest.arena.value() != 0 && resolved.required_extent_ > 0 &&
                resolved.required_extent_ == manifest.arena_extent &&
                manifest.arena_base % SpmAllocator::ALIGN == 0 &&
                manifest.arena_extent % SpmAllocator::ALIGN == 0 &&
                manifest.arena_extent <=
                    std::numeric_limits<uint32_t>::max() -
                        manifest.arena_base &&
                resolved.layout_hash_ != 0 &&
                resolved.declaration_hash_ != 0 &&
                resolved.persistent_hash_ != 0 &&
                resolved.allocation_hash_ != 0 &&
                resolved.profile_hash_ != 0 && resolved.range_hash_ != 0 &&
                resolved.identity_hash_ != 0 &&
                resolved.allocation_count_ > 0 &&
                !manifest.live_ranges.empty(),
            "composite physical plan ", label,
            " manifest is empty, unaligned, or malformed");
        const std::shared_ptr<const uint64_t> owner =
            resolved.owner_generation_.lock();
        TORCH_CHECK(owner != nullptr &&
                        *owner == resolved.expected_owner_generation_,
                    "composite physical plan ", label,
                    " manifest owner is stale");

        uint64_t range_hash = fnv_u64(kFnvOffset, 7);
        range_hash = fnv_u64(range_hash, manifest.live_ranges.size());
        for (const SpmPhaseSlice& slice : manifest.live_ranges) {
            validate_phase_slice(slice, manifest.arena,
                                 manifest.arena_extent, label);
            range_hash = fnv_u64(range_hash, slice.offset);
            range_hash = fnv_u64(range_hash, slice.bytes);
            range_hash = fnv_u64(range_hash, slice.first_event.value());
            range_hash = fnv_u64(range_hash, slice.last_event.value());
        }
        if (range_hash == 0) range_hash = kNonzeroHashFallback;
        TORCH_CHECK(range_hash == resolved.range_hash_,
                    "composite physical plan ", label,
                    " manifest range hash drifted");

        uint64_t identity_hash = fnv_u64(kFnvOffset, 8);
        identity_hash = fnv_u64(identity_hash, manifest.arena.value());
        identity_hash = fnv_u64(identity_hash, manifest.arena_base);
        identity_hash = fnv_u64(identity_hash, resolved.required_extent_);
        identity_hash = fnv_u64(identity_hash, resolved.layout_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.declaration_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.persistent_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.allocation_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.profile_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.range_hash_);
        identity_hash = fnv_u64(identity_hash, resolved.allocation_count_);
        identity_hash = fnv_u64(identity_hash, manifest.live_ranges.size());
        if (identity_hash == 0) identity_hash = kNonzeroHashFallback;
        TORCH_CHECK(identity_hash == resolved.identity_hash_ &&
                        manifest.layout_hash == resolved.identity_hash_,
                    "composite physical plan ", label,
                    " manifest identity drifted");
        return owner;
    };

    const std::shared_ptr<const uint64_t> producer_manifest_owner =
        validate_manifest(producer_manifest, "producer");
    const std::shared_ptr<const uint64_t> consumer_manifest_owner =
        validate_manifest(consumer_manifest, "consumer");
    TORCH_CHECK(
        producer_manifest.manifest_.arena !=
            consumer_manifest.manifest_.arena,
        "composite physical plan producer and consumer arena IDs must differ");

    const size_t producer_count = direct_bindings.size();
    TORCH_CHECK(
        composite.occurrence_count() == producer_count + 1,
        "composite physical plan requires repeated producer occurrences "
        "followed by exactly one consumer occurrence");
    const size_t consumer_occurrence_ordinal = producer_count;
    const void* composite_authority = composite.payload_.get();
    TORCH_CHECK(composite_authority != nullptr,
                "composite physical plan received an empty composite seal");

    std::vector<const SpmFmbCompositeYieldBinding*> direct_by_image(
        producer_count, nullptr);
    std::unordered_set<uint64_t> unique_binding_hashes;
    size_t yields_per_producer = 0;
    uint64_t policy_version = 0;

    auto validate_binding_owners = [&](const SpmFmbCompositeYieldBinding& b) {
        const std::shared_ptr<uint64_t> producer_owner =
            b.producer_owner_generation_.lock();
        const std::shared_ptr<const uint64_t> consumer_owner =
            b.consumer_.manifest_.owner_generation_.lock();
        TORCH_CHECK(
            producer_owner != nullptr && consumer_owner != nullptr &&
                producer_owner.get() == producer_manifest_owner.get() &&
                consumer_owner.get() == consumer_manifest_owner.get() &&
                *producer_owner == b.expected_producer_owner_generation_ &&
                *consumer_owner ==
                    b.consumer_.manifest_.expected_owner_generation_ &&
                b.producer_profile_hash_ == producer_manifest.profile_hash_ &&
                b.producer_allocation_hash_ ==
                    producer_manifest.allocation_hash_ &&
                b.consumer_.manifest_.identity_hash_ ==
                    consumer_manifest.identity_hash_ &&
                b.consumer_.manifest_.profile_hash_ ==
                    consumer_manifest.profile_hash_ &&
                b.consumer_.manifest_.allocation_hash_ ==
                    consumer_manifest.allocation_hash_,
            "composite physical binding does not belong to the supplied "
            "producer/consumer manifests");
    };

    for (const auto& direct : direct_bindings) {
        const auto& b = direct.binding_;
        b.validate_live_authority(composite_authority);
        validate_binding_owners(b);
        TORCH_CHECK(
            b.kind_ == SpmFmbCompositeYieldBinding::Kind::DirectProduced &&
                b.producer_local_ordinal_ < producer_count &&
                b.producer_occurrence_ordinal_ ==
                    b.producer_local_ordinal_ &&
                b.consumer_occurrence_ordinal_ ==
                    consumer_occurrence_ordinal &&
                b.producer_occurrence_yield_count_ > 1 &&
                b.producer_yield_ordinal_ + 1 ==
                    b.producer_occurrence_yield_count_,
            "composite physical direct binding is not the final yield of one "
            "canonical producer occurrence");
        TORCH_CHECK(
            direct_by_image[b.producer_local_ordinal_] == nullptr,
            "composite physical plan has duplicate direct bindings for one "
            "producer occurrence");
        if (yields_per_producer == 0) {
            yields_per_producer = b.producer_occurrence_yield_count_;
            policy_version = b.policy_version_;
        }
        TORCH_CHECK(
            b.producer_occurrence_yield_count_ == yields_per_producer &&
                b.policy_version_ == policy_version &&
                unique_binding_hashes.insert(b.stable_hash_).second,
            "composite physical direct bindings differ in yield count/policy "
            "or stable identity");
        direct_by_image[b.producer_local_ordinal_] = &b;
    }
    TORCH_CHECK(
        std::all_of(direct_by_image.begin(), direct_by_image.end(),
                    [](const auto* binding) { return binding != nullptr; }),
        "composite physical direct bindings do not exactly cover producers");
    TORCH_CHECK(
        producer_count <= std::numeric_limits<size_t>::max() /
                              yields_per_producer &&
            composite.producer_yield_count() ==
                producer_count * yields_per_producer,
        "composite physical binding matrix does not cover every sealed "
        "producer yield");

    const size_t retained_tap_count = yields_per_producer - 1;
    TORCH_CHECK(
        retained_tap_count <= std::numeric_limits<size_t>::max() /
                                  producer_count,
        "composite physical RunAdd binding count overflows size_t");
    const size_t expected_run_add_count =
        retained_tap_count * producer_count;
    TORCH_CHECK(
        run_add_bindings.size() == expected_run_add_count,
        "composite physical plan requires an exact non-final yield cover: "
        "expected ", expected_run_add_count, ", got ",
        run_add_bindings.size());
    std::vector<const SpmFmbCompositeYieldBinding*> run_add_by_payload(
        expected_run_add_count, nullptr);
    for (const auto& run_add : run_add_bindings) {
        const auto& b = run_add.binding_;
        b.validate_live_authority(composite_authority);
        validate_binding_owners(b);
        TORCH_CHECK(
            b.kind_ == SpmFmbCompositeYieldBinding::Kind::RunAdd &&
                b.producer_local_ordinal_ < producer_count &&
                b.producer_occurrence_ordinal_ ==
                    b.producer_local_ordinal_ &&
                b.producer_yield_ordinal_ < retained_tap_count &&
                b.producer_occurrence_yield_count_ == yields_per_producer &&
                b.consumer_occurrence_ordinal_ ==
                    consumer_occurrence_ordinal &&
                b.policy_version_ == policy_version,
            "composite physical RunAdd binding is outside the canonical "
            "producer/yield matrix");
        const size_t payload_index =
            b.producer_yield_ordinal_ * producer_count +
            b.producer_local_ordinal_;
        TORCH_INTERNAL_ASSERT(payload_index < run_add_by_payload.size());
        TORCH_CHECK(
            run_add_by_payload[payload_index] == nullptr &&
                unique_binding_hashes.insert(b.stable_hash_).second,
            "composite physical RunAdd bindings contain a duplicate cell or "
            "stable identity");
        run_add_by_payload[payload_index] = &b;
    }
    TORCH_CHECK(
        std::all_of(run_add_by_payload.begin(), run_add_by_payload.end(),
                    [](const auto* binding) { return binding != nullptr; }),
        "composite physical RunAdd bindings do not exactly cover tap-major / "
        "producer-minor cells");

    const auto& first_direct = *direct_by_image.front();
    const auto& consumer_root = first_direct.consumer_.consumer_.backing;
    const auto& consumer_storage = first_direct.consumer_.consumer_.spec;
    consumer_storage.validate();
    TORCH_CHECK(
        consumer_manifest.manifest_.arena_base == 0 &&
            consumer_root.arena == consumer_manifest.manifest_.arena &&
            consumer_root.bytes == consumer_storage.storage_bytes() &&
            consumer_root.offset <= consumer_manifest.required_extent_ &&
            consumer_root.bytes <=
                consumer_manifest.required_extent_ - consumer_root.offset,
        "composite physical plan requires one zero-based, in-manifest "
        "consumer root");
    const size_t row_bytes =
        consumer_storage.storage_bytes() /
        static_cast<size_t>(consumer_storage.rows);
    TORCH_CHECK(row_bytes > 0 && row_bytes % SpmAllocator::ALIGN == 0,
                "composite physical consumer rows must be 256-byte aligned");
    const size_t consumer_root_begin =
        static_cast<size_t>(consumer_manifest.manifest_.arena_base) +
        consumer_root.offset;
    TORCH_CHECK(
        consumer_root_begin <= std::numeric_limits<size_t>::max() -
                                   consumer_root.bytes,
        "composite physical consumer root end overflows size_t");
    const size_t consumer_root_end =
        consumer_root_begin + consumer_root.bytes;
    TORCH_CHECK(
        producer_manifest.manifest_.arena_base == align_up(consumer_root_end),
        "composite physical producer arena must start immediately after the "
        "protected consumer root");

    std::vector<DirectTarget> direct_targets;
    direct_targets.reserve(producer_count);
    int64_t previous_direct_row_end = 0;
    for (size_t image = 0; image < producer_count; ++image) {
        const auto& b = *direct_by_image[image];
        const auto& endpoint = b.consumer_;
        TORCH_CHECK(
            endpoint.consumer_.backing == consumer_root &&
                endpoint.consumer_.spec == consumer_storage &&
                endpoint.root_aligned_capacity_ == consumer_root.bytes &&
                endpoint.row_begin_ >= previous_direct_row_end &&
                endpoint.row_count_ > 0,
            "composite physical direct endpoints must be ordered, "
            "non-overlapping slices of one consumer root");
        const size_t row_begin = static_cast<size_t>(endpoint.row_begin_);
        const size_t row_count = static_cast<size_t>(endpoint.row_count_);
        TORCH_CHECK(
            row_begin <= static_cast<size_t>(consumer_storage.rows) &&
                row_count <=
                    static_cast<size_t>(consumer_storage.rows) - row_begin &&
                row_begin <= std::numeric_limits<size_t>::max() / row_bytes &&
                row_count <= std::numeric_limits<size_t>::max() / row_bytes,
            "composite physical direct endpoint row geometry overflows");
        const size_t target_offset =
            consumer_root_begin + row_begin * row_bytes;
        const size_t target_bytes = row_count * row_bytes;
        TORCH_CHECK(
            target_offset <= std::numeric_limits<uint32_t>::max() &&
                target_bytes == b.produced_spec_.storage_bytes() &&
                target_offset <= consumer_root_end &&
                target_bytes <= consumer_root_end - target_offset,
            "composite physical direct target exceeds its consumer root");
        direct_targets.push_back(
            {b.stable_hash_, endpoint.identity_hash_, image,
             b.producer_yield_ordinal_,
             static_cast<uint32_t>(target_offset), target_bytes});
        previous_direct_row_end = endpoint.row_begin_ + endpoint.row_count_;
    }

    size_t component_peak = std::max(
        static_cast<size_t>(consumer_manifest.manifest_.arena_base) +
            consumer_manifest.required_extent_,
        static_cast<size_t>(producer_manifest.manifest_.arena_base) +
            producer_manifest.required_extent_);
    component_peak = align_up(component_peak);
    size_t payload_cursor = component_peak;
    std::vector<RunAddTarget> run_add_targets;
    run_add_targets.reserve(expected_run_add_count);
    int previous_consumer_layer = std::numeric_limits<int>::min();
    uint32_t previous_consumer_event = 0;
    bool have_previous_tap = false;
    for (size_t tap = 0; tap < retained_tap_count; ++tap) {
        const auto& first_tap = *run_add_by_payload[tap * producer_count];
        const auto& first_endpoint = first_tap.consumer_;
        if (have_previous_tap) {
            TORCH_CHECK(
                std::make_pair(first_endpoint.global_first_event_.value(),
                               first_endpoint.occurrence_layer_) >
                    std::make_pair(previous_consumer_event,
                                   previous_consumer_layer),
                "composite physical RunAdd taps must follow consumer callback "
                "event/layer order");
        }
        previous_consumer_event = first_endpoint.global_first_event_.value();
        previous_consumer_layer = first_endpoint.occurrence_layer_;
        have_previous_tap = true;

        for (size_t image = 0; image < producer_count; ++image) {
            const size_t payload_index = tap * producer_count + image;
            const auto& b = *run_add_by_payload[payload_index];
            const auto& endpoint = b.consumer_;
            const auto& direct_endpoint = direct_by_image[image]->consumer_;
            TORCH_CHECK(
                endpoint.shares_owned_root_storage_with(
                    first_direct.consumer_) &&
                    endpoint.consumer_.backing ==
                        first_endpoint.consumer_.backing &&
                    endpoint.consumer_.spec == consumer_storage &&
                    endpoint.root_aligned_capacity_ == consumer_root.bytes &&
                    endpoint.row_begin_ == direct_endpoint.row_begin_ &&
                    endpoint.row_count_ == direct_endpoint.row_count_ &&
                    endpoint.occurrence_kind_ ==
                        first_endpoint.occurrence_kind_ &&
                    endpoint.occurrence_key_ ==
                        first_endpoint.occurrence_key_ &&
                    endpoint.body_id_ == first_endpoint.body_id_ &&
                    endpoint.group_id_ == first_endpoint.group_id_ &&
                    endpoint.occurrence_layer_ ==
                        first_endpoint.occurrence_layer_ &&
                    endpoint.traversal_ordinal_ ==
                        first_endpoint.traversal_ordinal_ &&
                    endpoint.chunk_index_ == first_endpoint.chunk_index_ &&
                    endpoint.global_first_event_ ==
                        first_endpoint.global_first_event_ &&
                    endpoint.global_last_event_ ==
                        first_endpoint.global_last_event_,
                "composite physical RunAdd tap endpoints must share one "
                "consumer callback and preserve image row geometry");

            const size_t payload_bytes = b.produced_spec_.storage_bytes();
            payload_cursor = align_up(payload_cursor);
            TORCH_CHECK(
                payload_cursor <= std::numeric_limits<uint32_t>::max() &&
                    payload_bytes == b.logical_capacity_ &&
                    payload_bytes == b.aligned_capacity_ &&
                    payload_bytes <=
                        std::numeric_limits<size_t>::max() - payload_cursor,
                "composite physical retained payload overflows placement");
            const size_t consumer_offset =
                consumer_root_begin +
                static_cast<size_t>(endpoint.row_begin_) * row_bytes;
            TORCH_CHECK(
                consumer_offset <= std::numeric_limits<uint32_t>::max(),
                "composite physical RunAdd consumer offset exceeds uint32");
            run_add_targets.push_back(
                {b.stable_hash_, endpoint.identity_hash_, image, tap,
                 payload_index,
                 static_cast<uint32_t>(payload_cursor),
                 static_cast<uint32_t>(consumer_offset), payload_bytes,
                 endpoint.occurrence_layer_, endpoint.row_begin_,
                 endpoint.row_count_, endpoint.consumer_.spec});
            payload_cursor += payload_bytes;
        }
    }
    const size_t dynamic_bytes = align_up(payload_cursor);
    TORCH_CHECK(
        dynamic_bytes <= effective_budget &&
            aligned_reserved_top <= effective_budget - dynamic_bytes,
        "composite physical plan requires ",
        dynamic_bytes + aligned_reserved_top,
        " bytes/core, exceeding aligned budget ", effective_budget);

    SpmPipelinePlan plan;
    plan.phase_overlay_admission_ =
        kPhaseOverlayPhysicalCompositeAdmission;
    plan.budget_bytes_ = budget_bytes;
    plan.reserved_top_bytes_ = aligned_reserved_top;
    plan.peak_bytes_ = dynamic_bytes;
    plan.phase_manifests_ = {consumer_manifest.manifest_,
                             producer_manifest.manifest_};
    plan.phase_seal_guards_ = {
        {consumer_manifest.owner_generation_,
         consumer_manifest.expected_owner_generation_},
        {producer_manifest.owner_generation_,
         producer_manifest.expected_owner_generation_},
    };

    uint64_t hash = fnv_u64(
        kFnvOffset, kFmbCompositePhysicalPlanSchemaVersion);
    hash = fnv_u64(hash, plan.phase_overlay_admission_);
    hash = fnv_u64(hash, plan.budget_bytes_);
    hash = fnv_u64(hash, plan.reserved_top_bytes_);
    hash = fnv_u64(hash, plan.peak_bytes_);
    hash = fnv_u64(hash, producer_manifest.identity_hash_);
    hash = fnv_u64(hash, consumer_manifest.identity_hash_);
    hash = fnv_u64(hash, producer_count);
    hash = fnv_u64(hash, yields_per_producer);
    hash = fnv_u64(hash, composite.producer_yield_count());
    hash = fnv_u64(hash, policy_version);
    hash = fnv_u64(hash, direct_targets.size());
    for (const auto& target : direct_targets) {
        hash = fnv_u64(hash, target.binding_hash);
        hash = fnv_u64(hash, target.endpoint_hash);
        hash = fnv_u64(hash, target.producer_local_ordinal);
        hash = fnv_u64(hash, target.producer_yield_ordinal);
        hash = fnv_u64(hash, target.target_relative_offset);
        hash = fnv_u64(hash, target.bytes);
    }
    hash = fnv_u64(hash, run_add_targets.size());
    for (const auto& target : run_add_targets) {
        hash = fnv_u64(hash, target.binding_hash);
        hash = fnv_u64(hash, target.endpoint_hash);
        hash = fnv_u64(hash, target.producer_local_ordinal);
        hash = fnv_u64(hash, target.producer_yield_ordinal);
        hash = fnv_u64(hash, target.payload_index);
        hash = fnv_u64(hash, target.payload_relative_offset);
        hash = fnv_u64(hash, target.consumer_relative_offset);
        hash = fnv_u64(hash, target.bytes);
        hash = fnv_u64(hash, static_cast<uint64_t>(target.consumer_layer));
        hash = fnv_u64(hash,
                       static_cast<uint64_t>(target.consumer_row_begin));
        hash = fnv_u64(hash,
                       static_cast<uint64_t>(target.consumer_row_count));
        hash = fnv_dense_spec(hash, target.spec);
    }
    plan.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    plan.validate_phase_seals();

    uint64_t authority_digest = fnv_u64(
        kFnvOffset, kFmbCompositePhysicalAuthorityDigestSchemaVersion);
    authority_digest = fnv_u64(authority_digest,
                               composite.composite_hash());
    authority_digest = fnv_u64(authority_digest, plan.hash_);
    authority_digest = fnv_u64(authority_digest,
                               direct_targets.size());
    for (const auto& target : direct_targets) {
        authority_digest = fnv_u64(authority_digest,
                                   target.binding_hash);
    }
    authority_digest = fnv_u64(authority_digest,
                               run_add_targets.size());
    for (const auto& target : run_add_targets) {
        authority_digest = fnv_u64(authority_digest,
                                   target.binding_hash);
    }
    if (authority_digest == 0) {
        authority_digest = kNonzeroHashFallback;
    }

    return SpmFmbCompositePhysicalPlan(
        producer_manifest, consumer_manifest, std::move(plan),
        producer_count, composite.producer_yield_count(), direct_targets,
        run_add_targets, authority_digest);
}

SpmFmbCompositePhysicalPlan SpmFmbCompositePhysicalPlan::commit(
    const SpmFmbCompositePhysicalReservation& reservation,
    const SpmFmbSealedCompositeTrace& composite,
    const std::vector<SpmFmbCompositeDirectProducedBinding>& direct_bindings,
    const std::vector<SpmFmbCompositeRunAddBinding>& run_add_bindings) {
    reservation.validate_live_authority();
    TORCH_CHECK(!direct_bindings.empty() &&
                    direct_bindings.front().binding_.policy_version_ ==
                        reservation.policy_version_,
                "composite physical commit policy differs from reservation");
    for (const auto& binding : direct_bindings) {
        TORCH_CHECK(binding.binding_.produced_spec_ ==
                        reservation.produced_spec_,
                    "composite physical commit direct producer spec differs "
                    "from reservation");
    }
    for (const auto& binding : run_add_bindings) {
        TORCH_CHECK(binding.binding_.produced_spec_ ==
                        reservation.produced_spec_,
                    "composite physical commit RunAdd producer spec differs "
                    "from reservation");
    }
    SpmFmbCompositePhysicalPlan candidate = compile(
        composite, reservation.producer_manifest_,
        reservation.consumer_manifest_, direct_bindings, run_add_bindings,
        reservation.plan_.reserved_top_bytes_,
        reservation.plan_.budget_bytes_);

    TORCH_CHECK(
        candidate.authority_digest_ != 0 &&
            candidate.producer_occurrence_count_ ==
                reservation.producer_occurrence_count_ &&
            candidate.producer_yield_count_ ==
                reservation.producer_yield_count() &&
            candidate.plan_.peak_bytes_ == reservation.plan_.peak_bytes_ &&
            candidate.plan_.reserved_top_bytes_ ==
                reservation.plan_.reserved_top_bytes_ &&
            candidate.plan_.budget_bytes_ == reservation.plan_.budget_bytes_ &&
            candidate.direct_targets_.size() ==
                reservation.direct_targets_.size() &&
            candidate.run_add_targets_.size() ==
                reservation.run_add_targets_.size(),
        "composite physical commit does not match the reserved placement");

    for (size_t i = 0; i < candidate.direct_targets_.size(); ++i) {
        const auto& committed = candidate.direct_targets_[i];
        const auto& reserved = reservation.direct_targets_[i];
        TORCH_CHECK(
            committed.endpoint_hash == reserved.endpoint_hash &&
                committed.producer_local_ordinal ==
                    reserved.producer_local_ordinal &&
                committed.producer_yield_ordinal ==
                    reserved.producer_yield_ordinal &&
                committed.target_relative_offset ==
                    reserved.target_relative_offset &&
                committed.bytes == reserved.bytes,
            "composite physical commit direct binding ", i,
            " does not match its reserved endpoint");
    }
    for (size_t i = 0; i < candidate.run_add_targets_.size(); ++i) {
        const auto& committed = candidate.run_add_targets_[i];
        const auto& reserved = reservation.run_add_targets_[i];
        TORCH_CHECK(
            committed.endpoint_hash == reserved.endpoint_hash &&
                committed.producer_local_ordinal ==
                    reserved.producer_local_ordinal &&
                committed.producer_yield_ordinal ==
                    reserved.producer_yield_ordinal &&
                committed.payload_index == reserved.payload_index &&
                committed.payload_relative_offset ==
                    reserved.payload_relative_offset &&
                committed.consumer_relative_offset ==
                    reserved.consumer_relative_offset &&
                committed.bytes == reserved.bytes &&
                committed.consumer_layer == reserved.consumer_layer &&
                committed.consumer_row_begin ==
                    reserved.consumer_row_begin &&
                committed.consumer_row_count ==
                    reserved.consumer_row_count &&
                committed.spec == reserved.spec,
            "composite physical commit RunAdd binding ", i,
            " does not match its reserved endpoint");
    }

    SpmFmbCompositePhysicalPlan result(
        reservation.producer_manifest_, reservation.consumer_manifest_,
        reservation.plan_, reservation.producer_occurrence_count_,
        reservation.producer_yield_count(), candidate.direct_targets_,
        candidate.run_add_targets_, candidate.authority_digest_);
    TORCH_INTERNAL_ASSERT(result.hash() == reservation.hash());
    result.validate_live_authority();
    return result;
}

bool SpmFmbCompositePhysicalPlan::has_direct_target(
    size_t producer_local_ordinal,
    size_t producer_yield_ordinal) const {
    return std::any_of(
        direct_targets_.begin(), direct_targets_.end(),
        [&](const DirectTarget& target) {
            return target.producer_local_ordinal == producer_local_ordinal &&
                target.producer_yield_ordinal == producer_yield_ordinal;
        });
}

size_t SpmFmbCompositePhysicalPlan::retained_payload_index(
    size_t producer_local_ordinal,
    size_t producer_yield_ordinal) const {
    const auto found = std::find_if(
        run_add_targets_.begin(), run_add_targets_.end(),
        [&](const RunAddTarget& target) {
            return target.producer_local_ordinal == producer_local_ordinal &&
                target.producer_yield_ordinal == producer_yield_ordinal;
        });
    TORCH_CHECK(found != run_add_targets_.end(),
                "composite physical plan has no retained payload for "
                "producer-local ", producer_local_ordinal, " yield ",
                producer_yield_ordinal);
    return found->payload_index;
}

void SpmFmbCompositePhysicalPlan::validate_live_authority() const {
    plan_.validate_phase_seals();
    TORCH_CHECK(
        plan_.phase_overlay_admission_ ==
                kPhaseOverlayPhysicalCompositeAdmission &&
            plan_.hash_ != 0 && producer_occurrence_count_ > 0 &&
            authority_digest_ != 0 &&
            producer_yield_count_ > producer_occurrence_count_ &&
            producer_yield_count_ % producer_occurrence_count_ == 0 &&
            direct_targets_.size() == producer_occurrence_count_ &&
            direct_targets_.size() + run_add_targets_.size() ==
                producer_yield_count_ &&
            plan_.total_bytes() <= plan_.effective_budget_bytes(),
        "composite physical plan retained placement is malformed");
    const std::shared_ptr<const uint64_t> producer_owner =
        producer_manifest_.owner_generation_.lock();
    const std::shared_ptr<const uint64_t> consumer_owner =
        consumer_manifest_.owner_generation_.lock();
    TORCH_CHECK(
        producer_owner != nullptr && consumer_owner != nullptr &&
            *producer_owner ==
                producer_manifest_.expected_owner_generation_ &&
            *consumer_owner ==
                consumer_manifest_.expected_owner_generation_,
        "composite physical plan component owner is stale");
}

SpmPipelinePlan SpmPipelinePlan::compile(
    const std::vector<SpmScratchDecl>& scratch_regions,
    const std::vector<SpmIdentityRoute>& identity_routes,
    size_t reserved_top_bytes,
    size_t budget_bytes) {
    TORCH_CHECK(budget_bytes > 0,
                "SpmPipelinePlan: budget must be positive");
    TORCH_CHECK(budget_bytes <= SpmAllocator::SPM_PLANNING_BUDGET,
                "SpmPipelinePlan: budget ", budget_bytes,
                " exceeds canonical ceiling ",
                SpmAllocator::SPM_PLANNING_BUDGET);
    TORCH_CHECK(!scratch_regions.empty() || !identity_routes.empty(),
                "SpmPipelinePlan: at least one typed region is required");
    TORCH_CHECK(reserved_top_bytes <= budget_bytes,
                "SpmPipelinePlan: reserved_top_bytes ", reserved_top_bytes,
                " exceeds budget ", budget_bytes);

    SpmPipelinePlan result;
    result.budget_bytes_ = budget_bytes;
    result.reserved_top_bytes_ = align_up(reserved_top_bytes);
    TORCH_CHECK(result.reserved_top_bytes_ <= budget_bytes,
                "SpmPipelinePlan: aligned reserved_top_bytes ",
                result.reserved_top_bytes_, " exceeds budget ", budget_bytes);

    std::unordered_set<uint32_t> scratch_ids;
    std::unordered_set<uint32_t> port_ids;
    std::vector<SpmAllocator::AllocRequest> requests;
    const size_t region_count =
        scratch_regions.size() + identity_routes.size();
    requests.reserve(region_count);
    result.regions_.reserve(region_count);

    for (const SpmScratchDecl& decl : scratch_regions) {
        TORCH_CHECK(decl.id.value() != 0,
                    "SpmPipelinePlan: scratch ID zero is reserved");
        TORCH_CHECK(scratch_ids.insert(decl.id.value()).second,
                    "SpmPipelinePlan: duplicate scratch ID ",
                    decl.id.value());
        TORCH_CHECK(decl.bytes > 0,
                    "SpmPipelinePlan: scratch ", decl.id.value(),
                    " must have positive size");
        TORCH_CHECK(decl.first_event >= 0 &&
                        decl.first_event <= decl.last_event,
                    "SpmPipelinePlan: invalid scratch lifetime [",
                    decl.first_event, ", ", decl.last_event, "] for ID ",
                    decl.id.value());
        const size_t bytes = static_cast<size_t>(decl.bytes);
        const size_t aligned_bytes = align_up(bytes);
        TORCH_CHECK(aligned_bytes <= budget_bytes,
                    "SpmPipelinePlan: aligned scratch ", decl.id.value(),
                    " needs ", aligned_bytes,
                    " bytes, exceeding budget ", budget_bytes);
        TORCH_CHECK(aligned_bytes <= std::numeric_limits<uint32_t>::max(),
                    "SpmPipelinePlan: aligned scratch ", decl.id.value(),
                    " exceeds the uint32 offset range");
        requests.push_back({decl.bytes, decl.first_event,
                            decl.last_event, 0});
        result.regions_.push_back({"", decl.id.value(), 0, bytes,
                                   decl.first_event, decl.last_event, 0, {}});
    }

    for (const SpmIdentityRoute& route : identity_routes) {
        TORCH_CHECK(route.id().value() != 0,
                    "SpmPipelinePlan: port ID zero is reserved");
        TORCH_CHECK(port_ids.insert(route.id().value()).second,
                    "SpmPipelinePlan: duplicate port ID ",
                    route.id().value());
        route.spec().validate();
        TORCH_CHECK(route.first_write() >= 0 &&
                        route.first_write() <= route.last_read(),
                    "SpmPipelinePlan: invalid port lifetime for ID ",
                    route.id().value());
        const size_t bytes = route.spec().storage_bytes();
        const size_t aligned_bytes = align_up(bytes);
        TORCH_CHECK(aligned_bytes <= budget_bytes,
                    "SpmPipelinePlan: aligned port ", route.id().value(),
                    " needs ", aligned_bytes,
                    " bytes, exceeding budget ", budget_bytes);
        TORCH_CHECK(aligned_bytes <= std::numeric_limits<uint32_t>::max(),
                    "SpmPipelinePlan: aligned port ", route.id().value(),
                    " exceeds the uint32 offset range");
        requests.push_back({static_cast<int64_t>(bytes),
                            route.first_write(), route.last_read(), 0});
        result.regions_.push_back({"", 0, route.id().value(), bytes,
                                   route.first_write(), route.last_read(), 0,
                                   route.spec()});
    }

    const SpmAllocator::AliasedPlan allocation =
        SpmAllocator::plan_temporary_aliased(requests);
    TORCH_CHECK(allocation.offsets.size() == result.regions_.size(),
                "SpmPipelinePlan: planner returned the wrong offset count");
    for (size_t i = 0; i < result.regions_.size(); ++i) {
        result.regions_[i].offset = allocation.offsets[i];
    }
    result.peak_bytes_ = allocation.peak_bytes;

    TORCH_CHECK(result.peak_bytes_ <= budget_bytes &&
                    result.reserved_top_bytes_ <=
                        budget_bytes - result.peak_bytes_,
                "SpmPipelinePlan: plan requires ", result.total_bytes(),
                " bytes/core (temporary peak ", result.peak_bytes_,
                " + reserved top ", result.reserved_top_bytes_,
                "), exceeding budget ", budget_bytes);

    uint64_t hash = fnv_u64(kFnvOffset, kTypedPlanSchemaVersion);
    hash = fnv_u64(hash, result.budget_bytes_);
    hash = fnv_u64(hash, result.reserved_top_bytes_);
    hash = fnv_u64(hash, result.peak_bytes_);
    hash = fnv_u64(hash, result.regions_.size());
    for (const Region& region : result.regions_) {
        if (region.scratch_id != 0) {
            hash = fnv_u64(hash, kTypedScratchRegionTag);
            hash = fnv_u64(hash, region.scratch_id);
        } else {
            TORCH_INTERNAL_ASSERT(region.port_id != 0);
            hash = fnv_u64(hash, kTypedPortRegionTag);
            hash = fnv_u64(hash, region.port_id);
        }
        hash = fnv_u64(hash, region.bytes);
        hash = fnv_u64(hash, static_cast<uint64_t>(region.first_event));
        hash = fnv_u64(hash, static_cast<uint64_t>(region.last_event));
        hash = fnv_u64(hash, region.offset);
        if (region.port_id != 0) {
            hash = fnv_u64(hash,
                           static_cast<uint8_t>(region.port_spec.dtype));
            hash = fnv_u64(hash,
                           static_cast<uint64_t>(region.port_spec.rows));
            hash = fnv_u64(hash,
                           static_cast<uint64_t>(region.port_spec.cols));
            hash = fnv_u64(
                hash, static_cast<uint8_t>(region.port_spec.distribution));
        }
    }
    result.hash_ = hash == 0 ? kNonzeroHashFallback : hash;
    return result;
}

bool SpmPipelinePlan::contains(const std::string& name) const {
    for (const Region& region : regions_) {
        if (region.scratch_id == 0 && region.port_id == 0 &&
            region.name == name) return true;
    }
    return false;
}

bool SpmPipelinePlan::contains(SpmScratchId id) const {
    TORCH_CHECK(id.value() != 0,
                "SpmPipelinePlan: scratch ID zero is reserved");
    for (const Region& region : regions_) {
        if (region.scratch_id == id.value()) return true;
    }
    return false;
}

bool SpmPipelinePlan::contains(SpmPortId id) const {
    TORCH_CHECK(id.value() != 0,
                "SpmPipelinePlan: port ID zero is reserved");
    for (const Region& region : regions_) {
        if (region.port_id == id.value()) return true;
    }
    return false;
}

const SpmDense2DSpec& SpmPipelinePlan::port_spec(SpmPortId id) const {
    return region(id).port_spec;
}

uint64_t SpmPipelinePlan::permutation_row_runs_hash(
    SpmPortId source_id,
    SpmPortId destination_id) const {
    return permutation_route(source_id, destination_id).hash();
}

bool SpmPipelinePlan::contains_phase_arena(SpmScratchId arena) const {
    TORCH_CHECK(arena.value() != 0,
                "SpmPipelinePlan: phase arena ID zero is reserved");
    for (const SpmComponentPhaseManifest& manifest : phase_manifests_) {
        if (manifest.arena == arena) return true;
    }
    return false;
}

uint64_t SpmPipelinePlan::chunked_permutation_hash(
    SpmPortId source_id,
    SpmPortId destination_id) const {
    return chunked_permutation_route(source_id, destination_id).hash();
}

uint64_t SpmPipelinePlan::direct_produced_row_runs_hash(
    SpmPortId producer_id,
    SpmPortId destination_id) const {
    return direct_produced_row_runs_route(
        producer_id, destination_id).hash();
}

uint64_t SpmPipelinePlan::direct_produced_payload_run_add_hash(
    SpmPortId payload_id,
    SpmPortId destination_id) const {
    return direct_produced_payload_run_add_route(
        payload_id, destination_id).hash();
}

const SpmPipelinePlan::Region& SpmPipelinePlan::region(
    const std::string& name) const {
    for (const Region& region : regions_) {
        if (region.scratch_id == 0 && region.port_id == 0 &&
            region.name == name) return region;
    }
    TORCH_CHECK(false, "SpmPipelinePlan: unknown region '", name, "'");
}

const SpmPipelinePlan::Region& SpmPipelinePlan::region(
    SpmScratchId id) const {
    TORCH_CHECK(id.value() != 0,
                "SpmPipelinePlan: scratch ID zero is reserved");
    for (const Region& region : regions_) {
        if (region.scratch_id == id.value()) return region;
    }
    TORCH_CHECK(false, "SpmPipelinePlan: unknown scratch ID ", id.value());
}

const SpmPipelinePlan::Region& SpmPipelinePlan::region(SpmPortId id) const {
    TORCH_CHECK(id.value() != 0,
                "SpmPipelinePlan: port ID zero is reserved");
    for (const Region& region : regions_) {
        if (region.port_id == id.value()) return region;
    }
    TORCH_CHECK(false, "SpmPipelinePlan: unknown port ID ", id.value());
}

const SpmPermutationRowRunsRoute& SpmPipelinePlan::permutation_route(
    SpmPortId source_id,
    SpmPortId destination_id) const {
    TORCH_CHECK(source_id.value() != 0 && destination_id.value() != 0,
                "SpmPipelinePlan: permutation route port ID zero is reserved");
    for (const SpmPermutationRowRunsRoute& route : permutation_routes_) {
        if (route.source_id() == source_id &&
            route.destination_id() == destination_id) {
            return route;
        }
    }
    TORCH_CHECK(false, "SpmPipelinePlan: unknown permutation route ",
                source_id.value(), " -> ", destination_id.value());
}

const SpmComponentPhaseManifest& SpmPipelinePlan::phase_manifest(
    SpmScratchId arena) const {
    TORCH_CHECK(arena.value() != 0,
                "SpmPipelinePlan: phase arena ID zero is reserved");
    for (const SpmComponentPhaseManifest& manifest : phase_manifests_) {
        if (manifest.arena == arena) return manifest;
    }
    TORCH_CHECK(false, "SpmPipelinePlan: unknown phase arena ID ",
                arena.value());
}

const SpmChunkedPermutationRoute&
SpmPipelinePlan::chunked_permutation_route(
    SpmPortId source_id,
    SpmPortId destination_id) const {
    TORCH_CHECK(source_id.value() != 0 && destination_id.value() != 0,
                "SpmPipelinePlan: chunked route port ID zero is reserved");
    for (const SpmChunkedPermutationRoute& route :
         chunked_permutation_routes_) {
        if (route.source_id() == source_id &&
            route.destination_id() == destination_id) {
            return route;
        }
    }
    TORCH_CHECK(false, "SpmPipelinePlan: unknown chunked permutation route ",
                source_id.value(), " -> ", destination_id.value());
}

const SpmDirectProducedRowRunsRoute&
SpmPipelinePlan::direct_produced_row_runs_route(
    SpmPortId producer_id,
    SpmPortId destination_id) const {
    TORCH_CHECK(producer_id.value() != 0 && destination_id.value() != 0,
                "SpmPipelinePlan: direct-produced route port ID zero is "
                "reserved");
    for (const SpmDirectProducedRowRunsRoute& route :
         direct_produced_row_runs_routes_) {
        if (route.producer_id() == producer_id &&
            route.destination_id() == destination_id) {
            return route;
        }
    }
    TORCH_CHECK(false, "SpmPipelinePlan: unknown direct-produced route ",
                producer_id.value(), " -> ", destination_id.value());
}

const SpmDirectProducedPayloadRunAddRoute&
SpmPipelinePlan::direct_produced_payload_run_add_route(
    SpmPortId payload_id,
    SpmPortId destination_id) const {
    TORCH_CHECK(payload_id.value() != 0 && destination_id.value() != 0,
                "SpmPipelinePlan: direct-produced payload route port ID "
                "zero is reserved");
    for (const SpmDirectProducedPayloadRunAddRoute& route :
         direct_produced_payload_run_add_routes_) {
        if (route.payload_id() == payload_id &&
            route.destination_id() == destination_id) {
            return route;
        }
    }
    TORCH_CHECK(
        false, "SpmPipelinePlan: unknown direct-produced payload RunAdd route ",
        payload_id.value(), " -> ", destination_id.value());
}

uint32_t SpmPipelinePlan::phase_slice_absolute_offset(
    const SpmPhaseSlice& slice) const {
    const SpmComponentPhaseManifest& manifest = phase_manifest(slice.arena);
    validate_phase_slice(slice, manifest.arena, manifest.arena_extent,
                         "phase chunk view");
    TORCH_CHECK(slice.offset <=
                    std::numeric_limits<uint32_t>::max() -
                        manifest.arena_base,
                "SpmPipelinePlan: phase chunk absolute offset exceeds uint32");
    return manifest.arena_base + slice.offset;
}

void SpmPipelinePlan::validate_phase_seals() const {
    for (const PhaseSealGuard& guard : phase_seal_guards_) {
        const std::shared_ptr<const uint64_t> owner =
            guard.owner_generation.lock();
        TORCH_CHECK(owner != nullptr,
                    "SpmPipelinePlan: FMB-sealed manifest owner expired");
        TORCH_CHECK(*owner == guard.expected_owner_generation,
                    "SpmPipelinePlan: FMB-sealed manifest is stale after "
                    "relayout, prepare, or model-state change");
    }
}

SpmPipelineLease SpmPipelineLease::acquire(
    const SpmPipelinePlan& plan,
    uint64_t allocator_generation) {
    plan.validate_phase_seals();
    GlobalLeaseState& state = global_lease_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(!state.active,
                "SpmPipelineLease: another process-global lease is active");
    TORCH_CHECK(state.next_epoch != std::numeric_limits<uint64_t>::max(),
                "SpmPipelineLease: epoch exhausted");

    const uint64_t epoch = ++state.next_epoch;
    state.active = true;
    state.epoch = epoch;
    state.plan_hash = plan.hash();
    state.allocator_generation = allocator_generation;
    state.owner_thread = std::this_thread::get_id();
    return SpmPipelineLease(plan, allocator_generation, epoch,
                            state.owner_thread);
}

std::unique_ptr<SpmPipelineLease> SpmPipelineLease::acquire_physical(
    const SpmPipelinePlan& plan) {
    plan.validate_phase_seals();
    TORCH_CHECK(
        plan.phase_overlay_admission_ !=
            kPhaseOverlayPhysicalCompositeAdmission,
        "SpmPipelineLease: composite physical admission requires the opaque "
        "SpmFmbCompositePhysicalPlan overload");
    const bool physical_admission =
        plan.phase_overlay_admission_ == 0 ||
        plan.phase_overlay_admission_ ==
            kPhaseOverlayPhysicalFmbSealedAdmission;
    if (plan.phase_overlay_admission_ ==
        kPhaseOverlaySyntheticCpuAdmission) {
        TORCH_CHECK(
            false,
            "SpmPipelineLease: public phase-overlay plans are synthetic "
            "CPU-only; physical admission requires an opaque "
            "FusedModelBase-sealed exporter");
    }
    TORCH_CHECK(
        physical_admission,
        "SpmPipelineLease: phase-overlay admission ",
        plan.phase_overlay_admission_,
        " is not on the physical allow-list; CPU-dry FMB seals remain "
        "logical-only");
    return acquire_physical_impl(plan);
}

std::unique_ptr<SpmPipelineLease> SpmPipelineLease::acquire_physical(
    const SpmFmbCompositePhysicalReservation& reservation) {
    reservation.validate_live_authority();
    // Retain/copy before process or allocator ownership changes.  The same
    // reservation supplies bootstrap-BUILD targets and later survives only as
    // Graph-independent placement authority.
    auto retained =
        std::make_shared<const SpmFmbCompositePhysicalReservation>(
            reservation);
    auto result = acquire_physical_impl(retained->plan_);
    result->composite_reservation_ = std::move(retained);
    return result;
}

std::unique_ptr<SpmPipelineLease> SpmPipelineLease::acquire_physical(
    const SpmFmbCompositePhysicalPlan& plan) {
    plan.validate_live_authority();
    // Allocate/copy all retained metadata before taking allocator or global
    // execution ownership.  The composite plan contains no bootstrap Graph
    // token, so this retained copy remains valid after that Graph is evicted.
    auto retained = std::make_shared<const SpmFmbCompositePhysicalPlan>(plan);
    auto result = acquire_physical_impl(retained->plan_);
    result->composite_plan_ = std::move(retained);
    return result;
}

std::unique_ptr<SpmPipelineLease> SpmPipelineLease::acquire_physical_impl(
    const SpmPipelinePlan& plan) {
    plan.validate_phase_seals();

    // Claim process-wide execution before the first allocator or SDK-facing
    // side effect.  A failed admission releases only this still-local token;
    // once construction succeeds the move-only token lives in the lease.
    auto execution_claim = RpuExecutionCoordinator::enter_physical(
        "SpmPipelineLease::acquire_physical");
    bool temporary_arena_started = false;
    try {

        GlobalLeaseState& state = global_lease_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        TORCH_CHECK(!state.active,
                    "SpmPipelineLease: another process-global lease is active");
        TORCH_CHECK(state.next_epoch != std::numeric_limits<uint64_t>::max(),
                    "SpmPipelineLease: epoch exhausted");

        SpmAllocator& allocator = SPM_ALLOC;
        if (!allocator.is_initialized()) allocator.init();
        TORCH_CHECK(!allocator.pipeline_arena_locked_,
                    "SpmPipelineLease: allocator already has a physical arena");

        const size_t actual_top = allocator.persistent_used() +
                                  allocator.super_persistent_used();
        TORCH_CHECK(actual_top == plan.reserved_top_bytes(),
                    "SpmPipelineLease: physical top reservation drifted: plan=",
                    plan.reserved_top_bytes(), " actual=", actual_top);
        TORCH_CHECK(plan.total_bytes() <= plan.effective_budget_bytes(),
                    "SpmPipelineLease: aligned physical plan requires ",
                    plan.total_bytes(),
                    " bytes/core, exceeding effective budget ",
                    plan.effective_budget_bytes());

        allocator.reset_temporary();
        temporary_arena_started = true;
        const uint64_t allocator_generation = allocator.generation();
        const uint32_t arena_base =
            allocator.alloc_temporary(plan.peak_bytes());
        TORCH_CHECK(arena_base == 0,
                    "SpmPipelineLease: physical arena must be zero-based after "
                    "reset; got base ", arena_base);
        const size_t arena_end = static_cast<size_t>(arena_base) +
                                 plan.peak_bytes();
        TORCH_CHECK(arena_end == allocator.temporary_used(),
                    "SpmPipelineLease: physical arena reservation mismatch");

        const uint64_t epoch = ++state.next_epoch;
        auto result = std::unique_ptr<SpmPipelineLease>(new SpmPipelineLease(
            plan,
            allocator_generation,
            epoch,
            std::this_thread::get_id(),
            std::move(execution_claim),
            allocator.persistent_generation(),
            arena_base,
            arena_end,
            actual_top));

        // From here through return every operation is non-throwing scalar
        // bookkeeping.  This ordering ensures that a constructor failure
        // leaves the process claim local and recoverable by the catch below.
        allocator.pipeline_arena_locked_ = true;
        allocator.pipeline_arena_epoch_ = epoch;
        state.active = true;
        state.epoch = epoch;
        state.plan_hash = plan.hash();
        state.allocator_generation = allocator_generation;
        state.owner_thread = std::this_thread::get_id();
        return result;
    } catch (...) {
        if (temporary_arena_started && execution_claim.valid()) {
            // No physical lock/global lease has been published while the
            // token remains local, so this partial reservation is recoverable.
            SPM_ALLOC.reset_temporary();
            SPM_ALLOC.release_deferred_floor_if_idle();
        }
        RpuExecutionCoordinator::exit_physical_noexcept(execution_claim);
        throw;
    }
}

SpmPipelineLease::SpmPipelineLease(const SpmPipelinePlan& plan,
                                   uint64_t allocator_generation,
                                   uint64_t epoch,
                                   std::thread::id owner_thread)
    : plan_(plan),
      allocator_generation_(allocator_generation),
      epoch_(epoch),
      owner_thread_(owner_thread),
      active_(true) {}

SpmPipelineLease::SpmPipelineLease(
    const SpmPipelinePlan& plan,
    uint64_t allocator_generation,
    uint64_t epoch,
    std::thread::id owner_thread,
    RpuExecutionCoordinator::Claim&& physical_execution_claim,
    uint64_t physical_persistent_generation,
    size_t physical_arena_base,
    size_t physical_arena_end,
    size_t physical_top_reservation)
    : plan_(plan),
      allocator_generation_(allocator_generation),
      epoch_(epoch),
      owner_thread_(owner_thread),
      active_(true),
      physical_(true),
      physical_persistent_generation_(physical_persistent_generation),
      physical_arena_base_(physical_arena_base),
      physical_arena_end_(physical_arena_end),
      physical_top_reservation_(physical_top_reservation),
      physical_lease_generation_(std::make_shared<uint64_t>(epoch)),
      physical_execution_claim_(
          std::in_place, std::move(physical_execution_claim)) {}

SpmPipelineLease::~SpmPipelineLease() {
    release_noexcept();
}

void SpmPipelineLease::validate_live_owner(bool validate_phase_seals) const {
    RpuExecutionCoordinator::check_current_thread_execution_allowed(
        "SpmPipelineLease owner validation");
    TORCH_CHECK(active_, "SpmPipelineLease: lease is not active");
    TORCH_CHECK(std::this_thread::get_id() == owner_thread_,
                "SpmPipelineLease: operation attempted from a non-owner thread");
    if (physical_) {
        TORCH_CHECK(physical_execution_claim_.has_value(),
                    "SpmPipelineLease: physical lease has no process claim");
        RpuExecutionCoordinator::validate_physical(
            *physical_execution_claim_, "SpmPipelineLease owner validation");
    }

    {
        GlobalLeaseState& state = global_lease_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        TORCH_CHECK(state.active && state.epoch == epoch_ &&
                        state.plan_hash == plan_.hash() &&
                        state.allocator_generation == allocator_generation_ &&
                        state.owner_thread == owner_thread_,
                    "SpmPipelineLease: process-global lease state is stale");
    }
    if (validate_phase_seals) plan_.validate_phase_seals();
}

SpmTensorView SpmPipelineLease::view(
    const std::string& region_name) const {
    validate_live_owner();
    const SpmPipelinePlan::Region& region = plan_.region(region_name);
    return view(region_name, 0, region.bytes);
}

SpmTensorView SpmPipelineLease::view(
    const std::string& region_name,
    size_t byte_offset,
    size_t byte_count) const {
    validate_live_owner();
    const SpmPipelinePlan::Region& region = plan_.region(region_name);
    TORCH_CHECK(byte_offset <= region.bytes &&
                    byte_count <= region.bytes - byte_offset,
                "SpmPipelineLease: view [", byte_offset, ", ",
                byte_offset + byte_count, ") exceeds region '", region_name,
                "' size ", region.bytes);
    TORCH_CHECK(static_cast<uint64_t>(region.offset) + byte_offset <=
                    std::numeric_limits<uint32_t>::max(),
                "SpmPipelineLease: relative offset exceeds uint32 range");

    SpmTensorView result;
    result.region_name_ = region_name;
    result.relative_offset_ =
        region.offset + static_cast<uint32_t>(byte_offset);
    result.size_bytes_ = byte_count;
    result.plan_hash_ = plan_.hash();
    result.lease_epoch_ = epoch_;
    return result;
}

SpmTensorView SpmPipelineLease::scratch(SpmScratchId id) const {
    validate_live_owner();
    const SpmPipelinePlan::Region& region = plan_.region(id);

    SpmTensorView result;
    result.scratch_id_ = id.value();
    result.relative_offset_ = region.offset;
    result.size_bytes_ = region.bytes;
    result.plan_hash_ = plan_.hash();
    result.lease_epoch_ = epoch_;
    return result;
}

SpmPortView SpmPipelineLease::port(SpmPortId id) const {
    validate_live_owner();
    const SpmPipelinePlan::Region& region = plan_.region(id);
    TORCH_INTERNAL_ASSERT(region.bytes == region.port_spec.storage_bytes());

    SpmPortView result;
    result.bytes_.port_id_ = id.value();
    result.bytes_.relative_offset_ = region.offset;
    result.bytes_.size_bytes_ = region.bytes;
    result.bytes_.plan_hash_ = plan_.hash();
    result.bytes_.lease_epoch_ = epoch_;
    result.spec_ = region.port_spec;
    result.row_slice_route_ = region.row_slice_route;
    result.dst_row_begin_ = region.dst_row_begin;
    result.dst_row_count_ = region.dst_row_count;
    return result;
}

SpmPermutationRowRunsView SpmPipelineLease::permutation_row_runs(
    SpmPortId source_id,
    SpmPortId destination_id) const {
    validate_live_owner();
    const SpmPermutationRowRunsRoute& route =
        plan_.permutation_route(source_id, destination_id);

    SpmPermutationRowRunsView result;
    result.source_ = port(route.source_id());
    result.destination_ = port(route.destination_id());
    result.permutation_ = route.permutation();
    result.runs_ = route.runs();
    result.route_hash_ = route.hash();

    TORCH_INTERNAL_ASSERT(result.source_.spec() == route.source_spec());
    TORCH_INTERNAL_ASSERT(result.destination_.spec() ==
                          route.destination_spec());
    return result;
}

SpmChunkedPermutationView SpmPipelineLease::chunked_permutation(
    SpmPortId source_id,
    SpmPortId destination_id) const {
    validate_live_owner();
    const SpmChunkedPermutationRoute& route =
        plan_.chunked_permutation_route(source_id, destination_id);

    SpmChunkedPermutationView result;
    result.source_chunks_.reserve(route.source_chunks().size());
    for (const SpmSourceChunkRef& chunk : route.source_chunks()) {
        SpmPhaseChunkView view(
            chunk.key, chunk.spec, chunk.backing,
            plan_.phase_slice_absolute_offset(chunk.backing),
            plan_.hash(), epoch_);
        result.source_chunks_.push_back(view);
    }
    result.payload_chunks_.reserve(route.payload_chunks().size());
    for (const SpmRetainedPayloadChunkRef& chunk : route.payload_chunks()) {
        SpmPhaseChunkView view(
            chunk.key, chunk.spec, chunk.backing,
            plan_.phase_slice_absolute_offset(chunk.backing),
            plan_.hash(), epoch_);
        result.payload_chunks_.push_back(view);
    }
    result.consumer_chunks_.reserve(route.consumer_chunks().size());
    for (const SpmBorrowedConsumerChunkRef& chunk :
         route.consumer_chunks()) {
        SpmPhaseChunkView view(
            chunk.key, chunk.spec, chunk.backing,
            plan_.phase_slice_absolute_offset(chunk.backing),
            plan_.hash(), epoch_);
        result.consumer_chunks_.push_back(view);
    }
    result.permutation_ = route.permutation();
    result.destination_runs_ = route.destination_runs();
    result.source_payload_runs_ = route.source_payload_runs();
    result.payload_consumer_runs_ = route.payload_consumer_runs();
    result.policy_version_ = route.policy_version();
    result.consumer_transform_ = route.consumer_transform();
    result.payload_partition_ = route.payload_partition();
    result.route_schema_version_ = route.route_schema_version();
    result.route_hash_ = route.hash();
    return result;
}

SpmDirectProducedRowRunsView SpmPipelineLease::direct_produced_row_runs(
    SpmPortId producer_id,
    SpmPortId destination_id) const {
    validate_live_owner();
    const SpmDirectProducedRowRunsRoute& route =
        plan_.direct_produced_row_runs_route(
            producer_id, destination_id);

    SpmDirectProducedRowRunsView result;
    result.targets_.reserve(route.lowered_runs().size());
    for (const SpmDirectProducedRowRun& run : route.lowered_runs()) {
        TORCH_INTERNAL_ASSERT(
            run.producer_chunk.ordinal() < route.produced_chunks().size() &&
            run.consumer_chunk.ordinal() < route.consumer_chunks().size());
        const SpmDirectProducedChunkRef& produced =
            route.produced_chunks()[run.producer_chunk.ordinal()];
        const SpmBorrowedConsumerChunkRef& consumer =
            route.consumer_chunks()[run.consumer_chunk.ordinal()];
        TORCH_INTERNAL_ASSERT(produced.key == run.producer_chunk &&
                              consumer.key == run.consumer_chunk &&
                              produced.spec.rows == run.row_count &&
                              produced.production_event ==
                                  run.production_event);
        const SpmPhaseChunkView consumer_view(
            consumer.key, consumer.spec, consumer.backing,
            plan_.phase_slice_absolute_offset(consumer.backing),
            plan_.hash(), epoch_);
        result.targets_.push_back(SpmDirectProducedTargetView(
            produced, consumer_view, run.consumer_row_begin));
    }
    result.destination_runs_ = route.destination_runs();
    result.policy_version_ = route.policy_version();
    result.route_schema_version_ = route.route_schema_version();
    result.route_hash_ = route.hash();
    return result;
}

SpmDirectProducedPayloadRunAddView
SpmPipelineLease::direct_produced_payload_run_add(
    SpmPortId payload_id,
    SpmPortId destination_id) const {
    validate_live_owner();
    const SpmDirectProducedPayloadRunAddRoute& route =
        plan_.direct_produced_payload_run_add_route(
            payload_id, destination_id);

    SpmDirectProducedPayloadRunAddView result;
    result.payloads_.reserve(route.payloads().size());
    for (const SpmDirectProducedPayloadRef& payload : route.payloads()) {
        result.payloads_.push_back(SpmPhaseChunkView(
            payload.key, payload.spec, payload.backing,
            plan_.phase_slice_absolute_offset(payload.backing),
            plan_.hash(), epoch_));
    }
    result.consumer_chunks_.reserve(route.consumer_chunks().size());
    for (const SpmBorrowedConsumerChunkRef& consumer :
         route.consumer_chunks()) {
        result.consumer_chunks_.push_back(SpmPhaseChunkView(
            consumer.key, consumer.spec, consumer.backing,
            plan_.phase_slice_absolute_offset(consumer.backing),
            plan_.hash(), epoch_));
    }
    result.run_adds_ = route.run_adds();
    result.policy_version_ = route.policy_version();
    result.route_schema_version_ = route.route_schema_version();
    result.route_hash_ = route.hash();
    return result;
}

uint32_t SpmPipelineLease::resolve(
    const SpmTensorView& view,
    uint64_t current_allocator_generation) const {
    validate_live_owner();
    TORCH_CHECK(current_allocator_generation == allocator_generation_,
                "SpmPipelineLease: allocator generation changed from ",
                allocator_generation_, " to ", current_allocator_generation);
    TORCH_CHECK(view.plan_hash_ == plan_.hash(),
                "SpmPipelineLease: view belongs to a different plan");
    TORCH_CHECK(view.lease_epoch_ == epoch_,
                "SpmPipelineLease: view belongs to a stale lease epoch");

    const bool named = !view.region_name_.empty();
    const bool scratch = view.scratch_id_ != 0;
    const bool port = view.port_id_ != 0;
    TORCH_CHECK(static_cast<int>(named) + static_cast<int>(scratch) +
                        static_cast<int>(port) ==
                    1,
                "SpmPipelineLease: view has an invalid region key");
    const SpmPipelinePlan::Region* region_ptr = nullptr;
    if (named) {
        region_ptr = &plan_.region(view.region_name_);
    } else if (scratch) {
        region_ptr = &plan_.region(SpmScratchId(view.scratch_id_));
    } else {
        region_ptr = &plan_.region(SpmPortId(view.port_id_));
    }
    const SpmPipelinePlan::Region& region = *region_ptr;
    const uint64_t region_begin = region.offset;
    const uint64_t region_end = region_begin + region.bytes;
    const uint64_t view_begin = view.relative_offset_;
    const uint64_t view_end = view_begin + view.size_bytes_;
    TORCH_CHECK(view_begin >= region_begin && view_end <= region_end,
                "SpmPipelineLease: view range no longer matches its region");
    return view.relative_offset_;
}

uint32_t SpmPipelineLease::resolve(
    const SpmPhaseChunkView& view,
    uint64_t current_allocator_generation) const {
    validate_live_owner();
    TORCH_CHECK(current_allocator_generation == allocator_generation_,
                "SpmPipelineLease: allocator generation changed from ",
                allocator_generation_, " to ", current_allocator_generation);
    TORCH_CHECK(view.plan_hash_ == plan_.hash(),
                "SpmPipelineLease: phase chunk belongs to a different plan");
    TORCH_CHECK(view.lease_epoch_ == epoch_,
                "SpmPipelineLease: phase chunk belongs to a stale lease epoch");
    TORCH_CHECK(view.backing_.bytes == view.spec_.storage_bytes(),
                "SpmPipelineLease: phase chunk dense storage drifted");
    TORCH_CHECK(view.relative_offset_ ==
                    plan_.phase_slice_absolute_offset(view.backing_),
                "SpmPipelineLease: phase chunk fixed placement drifted");
    TORCH_CHECK(view.relative_offset_ <= plan_.peak_bytes() &&
                    view.backing_.bytes <=
                        plan_.peak_bytes() - view.relative_offset_,
                "SpmPipelineLease: phase chunk exceeds fixed arena peak");
    return view.relative_offset_;
}

void SpmPipelineLease::validate_physical(
    uint64_t expected_epoch,
    uint64_t expected_plan_hash,
    bool require_sealed) const {
    validate_physical_impl(expected_epoch, expected_plan_hash,
                           require_sealed, /*validate_phase_seals=*/true);
}

void SpmPipelineLease::validate_physical_impl(
    uint64_t expected_epoch,
    uint64_t expected_plan_hash,
    bool require_sealed,
    bool validate_phase_seals) const {
    validate_physical_identity_impl(expected_epoch, expected_plan_hash,
                                    validate_phase_seals);
    TORCH_CHECK(!require_sealed || physical_sealed_,
                "SpmPipelineLease: physical arena has not been sealed");

    const SpmAllocator& allocator = SPM_ALLOC;
    TORCH_CHECK(allocator.pipeline_arena_locked_ &&
                    allocator.pipeline_arena_epoch_ == epoch_,
                "SpmPipelineLease: allocator physical-arena ownership drifted");
    TORCH_CHECK(allocator.generation() == allocator_generation_,
                "SpmPipelineLease: allocator generation changed from ",
                allocator_generation_, " to ", allocator.generation());
    TORCH_CHECK(allocator.persistent_generation() ==
                    physical_persistent_generation_,
                "SpmPipelineLease: persistent generation changed while the "
                "physical arena was live");
    TORCH_CHECK(allocator.temporary_used() == physical_arena_end_,
                "SpmPipelineLease: physical arena extent drifted from ",
                physical_arena_end_, " to ", allocator.temporary_used());
    TORCH_CHECK(allocator.persistent_used() +
                    allocator.super_persistent_used() ==
                    physical_top_reservation_,
                "SpmPipelineLease: physical top reservation changed while live");
}

void SpmPipelineLease::validate_physical_identity(
    uint64_t expected_epoch,
    uint64_t expected_plan_hash) const {
    validate_physical_identity_impl(expected_epoch, expected_plan_hash,
                                    /*validate_phase_seals=*/true);
}

void SpmPipelineLease::validate_physical_identity_impl(
    uint64_t expected_epoch,
    uint64_t expected_plan_hash,
    bool validate_phase_seals) const {
    validate_live_owner(validate_phase_seals);
    TORCH_CHECK(physical_,
                "SpmPipelineLease: operation requires a physical arena lease");
    TORCH_CHECK(epoch_ == expected_epoch,
                "SpmPipelineLease: physical epoch mismatch: expected ",
                expected_epoch, " actual ", epoch_);
    TORCH_CHECK(plan_.hash() == expected_plan_hash,
                "SpmPipelineLease: physical plan hash mismatch");
}

void SpmPipelineLease::validate_physical_range(
    uint32_t offset,
    size_t byte_count) const {
    validate_physical(epoch_, plan_.hash(), /*require_sealed=*/false);
    const uint64_t begin = offset;
    const uint64_t end = begin + byte_count;
    TORCH_CHECK(end >= begin && begin >= physical_arena_base_ &&
                    end <= physical_arena_end_,
                "SpmPipelineLease: physical range [", begin, ", ", end,
                ") exceeds arena [", physical_arena_base_, ", ",
                physical_arena_end_, ")");
}

uint32_t SpmPipelineLease::resolve_physical_offset(
    const SpmTensorView& view) const {
    const uint32_t relative = resolve(view, allocator_generation_);
    TORCH_CHECK(static_cast<uint64_t>(physical_arena_base_) + relative <=
                    std::numeric_limits<uint32_t>::max(),
                "SpmPipelineLease: physical offset exceeds uint32 range");
    const uint32_t absolute_offset =
        static_cast<uint32_t>(physical_arena_base_ + relative);
    validate_physical_range(absolute_offset, view.size_bytes_);
    return absolute_offset;
}

uint32_t SpmPipelineLease::resolve_physical_addr(
    int core,
    const SpmTensorView& view) const {
    TORCH_CHECK(core >= 0 && core < SpmAllocator::NUM_CORES,
                "SpmPipelineLease: core ", core, " outside [0, ",
                SpmAllocator::NUM_CORES, ")");
    return SPM_ALLOC.addr(core, resolve_physical_offset(view));
}

uint32_t SpmPipelineLease::resolve_physical_offset(
    const SpmPhaseChunkView& view) const {
    const uint32_t relative = resolve(view, allocator_generation_);
    TORCH_CHECK(static_cast<uint64_t>(physical_arena_base_) + relative <=
                    std::numeric_limits<uint32_t>::max(),
                "SpmPipelineLease: phase chunk physical offset exceeds uint32");
    const uint32_t absolute_offset =
        static_cast<uint32_t>(physical_arena_base_ + relative);
    validate_physical_range(absolute_offset, view.backing_.bytes);
    return absolute_offset;
}

uint32_t SpmPipelineLease::resolve_physical_addr(
    int core,
    const SpmPhaseChunkView& view) const {
    TORCH_CHECK(core >= 0 && core < SpmAllocator::NUM_CORES,
                "SpmPipelineLease: core ", core, " outside [0, ",
                SpmAllocator::NUM_CORES, ")");
    return SPM_ALLOC.addr(core, resolve_physical_offset(view));
}

void SpmPipelineLease::validate_composite_execution_authority(
    uint64_t expected_epoch,
    uint64_t expected_plan_hash,
    bool require_committed) const {
    validate_physical(expected_epoch, expected_plan_hash,
                      /*require_sealed=*/false);
    TORCH_CHECK(
        (composite_reservation_ != nullptr || composite_plan_ != nullptr) &&
            (!require_committed || composite_plan_ != nullptr),
        require_committed
            ? "SpmPipelineLease: composite execution requires committed "
              "BUILD authority"
            : "SpmPipelineLease: no composite reservation or plan authority");
    if (composite_reservation_ != nullptr) {
        TORCH_CHECK(composite_reservation_->hash() == plan_.hash(),
                    "SpmPipelineLease: composite reservation hash drifted");
        composite_reservation_->validate_live_authority();
    }
    if (composite_plan_ != nullptr) {
        TORCH_CHECK(composite_plan_->hash() == plan_.hash(),
                    "SpmPipelineLease: composite plan hash drifted");
        composite_plan_->validate_live_authority();
    }
}

size_t SpmPipelineLease::composite_producer_occurrence_count() const {
    validate_composite_execution_authority(epoch_, plan_.hash(), false);
    return composite_plan_ != nullptr
        ? composite_plan_->producer_occurrence_count_
        : composite_reservation_->producer_occurrence_count_;
}

size_t SpmPipelineLease::composite_yields_per_producer() const {
    validate_composite_execution_authority(epoch_, plan_.hash(), false);
    if (composite_plan_ != nullptr) {
        TORCH_INTERNAL_ASSERT(composite_plan_->producer_occurrence_count_ > 0);
        TORCH_INTERNAL_ASSERT(
            composite_plan_->producer_yield_count_ %
                composite_plan_->producer_occurrence_count_ ==
            0);
        return composite_plan_->producer_yield_count_ /
            composite_plan_->producer_occurrence_count_;
    }
    return composite_reservation_->yields_per_producer_;
}

size_t SpmPipelineLease::composite_run_add_count() const {
    validate_composite_execution_authority(epoch_, plan_.hash(), false);
    return composite_plan_ != nullptr
        ? composite_plan_->run_add_targets_.size()
        : composite_reservation_->run_add_targets_.size();
}

bool SpmPipelineLease::composite_committed() const {
    validate_composite_execution_authority(epoch_, plan_.hash(), false);
    return composite_plan_ != nullptr;
}

SpmPipelineLease::CompositeComponentPlacement
SpmPipelineLease::resolve_composite_component_placement(
    bool producer) const {
    validate_composite_execution_authority(epoch_, plan_.hash(), false);
    const SpmFmbResolvedPhaseManifest& manifest = composite_plan_ != nullptr
        ? (producer ? composite_plan_->producer_manifest_
                    : composite_plan_->consumer_manifest_)
        : (producer ? composite_reservation_->producer_manifest_
                    : composite_reservation_->consumer_manifest_);
    TORCH_CHECK(
        manifest.manifest_.arena_base <= plan_.peak_bytes_ &&
            manifest.required_extent_ <=
                plan_.peak_bytes_ - manifest.manifest_.arena_base,
        "SpmPipelineLease: composite component placement exceeds arena");
    return {
        manifest.manifest_.arena_base,
        manifest.required_extent_,
        manifest.layout_hash_,
        manifest.allocation_hash_,
    };
}

uint32_t SpmPipelineLease::resolve_composite_producer_target_addr(
    int core,
    size_t producer_local_ordinal,
    size_t producer_yield_ordinal) const {
    validate_composite_execution_authority(epoch_, plan_.hash(), false);
    TORCH_CHECK(core >= 0 && core < SpmAllocator::NUM_CORES,
                "SpmPipelineLease: composite producer target core ", core,
                " outside [0, ", SpmAllocator::NUM_CORES, ")");
    uint32_t relative_offset = 0;
    size_t bytes = 0;
    size_t matches = 0;
    if (composite_plan_ != nullptr) {
        for (const auto& target : composite_plan_->direct_targets_) {
            if (target.producer_local_ordinal == producer_local_ordinal &&
                target.producer_yield_ordinal == producer_yield_ordinal) {
                relative_offset = target.target_relative_offset;
                bytes = target.bytes;
                ++matches;
            }
        }
        for (const auto& target : composite_plan_->run_add_targets_) {
            if (target.producer_local_ordinal == producer_local_ordinal &&
                target.producer_yield_ordinal == producer_yield_ordinal) {
                relative_offset = target.payload_relative_offset;
                bytes = target.bytes;
                ++matches;
            }
        }
    } else {
        for (const auto& target : composite_reservation_->direct_targets_) {
            if (target.producer_local_ordinal == producer_local_ordinal &&
                target.producer_yield_ordinal == producer_yield_ordinal) {
                relative_offset = target.target_relative_offset;
                bytes = target.bytes;
                ++matches;
            }
        }
        for (const auto& target : composite_reservation_->run_add_targets_) {
            if (target.producer_local_ordinal == producer_local_ordinal &&
                target.producer_yield_ordinal == producer_yield_ordinal) {
                relative_offset = target.payload_relative_offset;
                bytes = target.bytes;
                ++matches;
            }
        }
    }
    TORCH_CHECK(matches == 1,
                "SpmPipelineLease: composite producer target must resolve "
                "exactly once");
    TORCH_CHECK(
        static_cast<size_t>(relative_offset) <=
            std::numeric_limits<uint32_t>::max() - physical_arena_base_,
        "SpmPipelineLease: composite producer target offset exceeds uint32");
    const uint32_t offset = static_cast<uint32_t>(
        physical_arena_base_ + relative_offset);
    validate_physical_range(offset, bytes);
    return SPM_ALLOC.addr(core, offset);
}

SpmPipelineLease::CompositeRunAddResolvedAddrs
SpmPipelineLease::resolve_composite_run_add_addrs(
    int core,
    size_t producer_local_ordinal,
    size_t producer_yield_ordinal) const {
    validate_composite_execution_authority(epoch_, plan_.hash(), false);
    TORCH_CHECK(core >= 0 && core < SpmAllocator::NUM_CORES,
                "SpmPipelineLease: composite RunAdd core ", core,
                " outside [0, ", SpmAllocator::NUM_CORES, ")");
    uint32_t payload_relative_offset = 0;
    uint32_t consumer_relative_offset = 0;
    size_t bytes = 0;
    int consumer_layer = -1;
    int64_t consumer_row_begin = 0;
    int64_t consumer_row_count = 0;
    SpmDense2DSpec spec;
    size_t matches = 0;
    if (composite_plan_ != nullptr) {
        for (const auto& target : composite_plan_->run_add_targets_) {
            if (target.producer_local_ordinal == producer_local_ordinal &&
                target.producer_yield_ordinal == producer_yield_ordinal) {
                payload_relative_offset = target.payload_relative_offset;
                consumer_relative_offset = target.consumer_relative_offset;
                bytes = target.bytes;
                consumer_layer = target.consumer_layer;
                consumer_row_begin = target.consumer_row_begin;
                consumer_row_count = target.consumer_row_count;
                spec = target.spec;
                ++matches;
            }
        }
    } else {
        for (const auto& target : composite_reservation_->run_add_targets_) {
            if (target.producer_local_ordinal == producer_local_ordinal &&
                target.producer_yield_ordinal == producer_yield_ordinal) {
                payload_relative_offset = target.payload_relative_offset;
                consumer_relative_offset = target.consumer_relative_offset;
                bytes = target.bytes;
                consumer_layer = target.consumer_layer;
                consumer_row_begin = target.consumer_row_begin;
                consumer_row_count = target.consumer_row_count;
                spec = target.spec;
                ++matches;
            }
        }
    }
    TORCH_CHECK(matches == 1,
                "SpmPipelineLease: composite RunAdd target must resolve "
                "exactly once");
    TORCH_CHECK(
        static_cast<size_t>(payload_relative_offset) <=
                std::numeric_limits<uint32_t>::max() - physical_arena_base_ &&
            static_cast<size_t>(consumer_relative_offset) <=
            std::numeric_limits<uint32_t>::max() - physical_arena_base_,
        "SpmPipelineLease: composite RunAdd offset exceeds uint32");
    const uint32_t payload_offset = static_cast<uint32_t>(
        physical_arena_base_ + payload_relative_offset);
    const uint32_t consumer_offset = static_cast<uint32_t>(
        physical_arena_base_ + consumer_relative_offset);
    validate_physical_range(payload_offset, bytes);
    validate_physical_range(consumer_offset, bytes);
    return {
        SPM_ALLOC.addr(core, payload_offset),
        SPM_ALLOC.addr(core, consumer_offset),
        bytes,
        consumer_layer,
        consumer_row_begin,
        consumer_row_count,
        spec,
    };
}

uint32_t SpmPipelineLease::resolve_composite_direct_target_addr(
    int core,
    size_t producer_local_ordinal,
    size_t producer_yield_ordinal) const {
    const size_t yields_per_producer = composite_yields_per_producer();
    TORCH_INTERNAL_ASSERT(yields_per_producer > 1);
    TORCH_CHECK(
        producer_yield_ordinal == yields_per_producer - 1,
        "SpmPipelineLease: requested composite yield is not direct-final");
    return resolve_composite_producer_target_addr(
        core, producer_local_ordinal, producer_yield_ordinal);
}

uint32_t SpmPipelineLease::resolve_composite_run_add_payload_addr(
    int core,
    size_t producer_local_ordinal,
    size_t producer_yield_ordinal) const {
    return resolve_composite_run_add_addrs(
        core, producer_local_ordinal, producer_yield_ordinal).payload_addr;
}

uint32_t SpmPipelineLease::resolve_composite_run_add_consumer_addr(
    int core,
    size_t producer_local_ordinal,
    size_t producer_yield_ordinal) const {
    return resolve_composite_run_add_addrs(
        core, producer_local_ordinal, producer_yield_ordinal).consumer_addr;
}

uint32_t SpmTensorView::resolve_relative(
    const SpmPipelineLease& lease,
    uint64_t current_allocator_generation) const {
    return lease.resolve(*this, current_allocator_generation);
}

uint32_t SpmTensorView::resolve_physical_offset(
    const SpmPipelineLease& lease) const {
    return lease.resolve_physical_offset(*this);
}

uint32_t SpmTensorView::resolve_physical_addr(
    int core,
    const SpmPipelineLease& lease) const {
    return lease.resolve_physical_addr(core, *this);
}

const std::string& SpmTensorView::region_name() const {
    TORCH_CHECK(!region_name_.empty() && scratch_id_ == 0 && port_id_ == 0,
                "SpmTensorView: typed regions do not have string names");
    return region_name_;
}

uint32_t SpmPortView::resolve_relative(
    const SpmPipelineLease& lease,
    uint64_t current_allocator_generation) const {
    return bytes_.resolve_relative(lease, current_allocator_generation);
}

SpmPortView SpmPortView::row_slice(int64_t row_begin,
                                   int64_t row_count) const {
    spec_.validate();
    TORCH_CHECK(row_begin >= 0,
                "SpmPortView: row-slice begin must be non-negative; got ",
                row_begin);
    TORCH_CHECK(row_count > 0,
                "SpmPortView: row-slice count must be positive; got ",
                row_count);
    TORCH_CHECK(row_begin <= spec_.rows &&
                    row_count <= spec_.rows - row_begin,
                "SpmPortView: row slice [", row_begin, ", ", row_begin, "+",
                row_count, ") exceeds storage rows ", spec_.rows);
    TORCH_CHECK(row_slice_route_,
                "SpmPortView: port has no planned row-slice route");
    TORCH_CHECK(row_begin == dst_row_begin_ && row_count == dst_row_count_,
                "SpmPortView: requested row slice [", row_begin, ", ",
                row_begin, "+", row_count, ") does not match planned slice [",
                dst_row_begin_, ", ", dst_row_begin_, "+", dst_row_count_,
                ")");

    SpmDense2DSpec one_row = spec_;
    one_row.rows = 1;
    const size_t row_bytes = one_row.storage_bytes();
    const size_t begin = static_cast<size_t>(row_begin);
    const size_t count = static_cast<size_t>(row_count);
    TORCH_CHECK(begin <= std::numeric_limits<size_t>::max() / row_bytes,
                "SpmPortView: row-slice byte offset overflows size_t");
    TORCH_CHECK(count <= std::numeric_limits<size_t>::max() / row_bytes,
                "SpmPortView: row-slice byte count overflows size_t");
    const size_t byte_offset = begin * row_bytes;
    const size_t byte_count = count * row_bytes;
    TORCH_CHECK(byte_offset <= bytes_.size_bytes_ &&
                    byte_count <= bytes_.size_bytes_ - byte_offset,
                "SpmPortView: row-slice byte range exceeds port storage");
    TORCH_CHECK(byte_offset <= std::numeric_limits<uint32_t>::max() -
                                   bytes_.relative_offset_,
                "SpmPortView: row-slice relative offset exceeds uint32 range");

    SpmPortView result = *this;
    result.bytes_.relative_offset_ += static_cast<uint32_t>(byte_offset);
    result.bytes_.size_bytes_ = byte_count;
    result.spec_.rows = row_count;
    result.row_slice_route_ = false;
    result.dst_row_begin_ = 0;
    result.dst_row_count_ = 0;
    return result;
}

std::vector<SpmContiguousRowRun>
SpmPermutationRowRunsView::lower_contiguous_runs() const {
    TORCH_INTERNAL_ASSERT(!permutation_.empty());
    TORCH_INTERNAL_ASSERT(!runs_.empty());

    std::vector<SpmContiguousRowRun> lowered;
    lowered.reserve(permutation_.size());
    size_t cursor = 0;
    for (const SpmDstRowRun& destination_run : runs_) {
        int64_t local_row = 0;
        while (local_row < destination_run.row_count) {
            TORCH_INTERNAL_ASSERT(cursor < permutation_.size());
            const int64_t source_begin = permutation_[cursor];
            int64_t count = 1;
            while (local_row + count < destination_run.row_count &&
                   permutation_[cursor + static_cast<size_t>(count)] ==
                       source_begin + count) {
                ++count;
            }
            lowered.push_back({source_begin,
                               destination_run.dst_row_begin + local_row,
                               count});
            cursor += static_cast<size_t>(count);
            local_row += count;
        }
    }
    TORCH_INTERNAL_ASSERT(cursor == permutation_.size());
    return lowered;
}

uint32_t SpmPortView::resolve_physical_offset(
    const SpmPipelineLease& lease) const {
    return bytes_.resolve_physical_offset(lease);
}

uint32_t SpmPortView::resolve_physical_addr(
    int core,
    const SpmPipelineLease& lease) const {
    return bytes_.resolve_physical_addr(core, lease);
}

uint32_t SpmPhaseChunkView::resolve_relative(
    const SpmPipelineLease& lease,
    uint64_t current_allocator_generation) const {
    return lease.resolve(*this, current_allocator_generation);
}

uint32_t SpmPhaseChunkView::resolve_physical_offset(
    const SpmPipelineLease& lease) const {
    return lease.resolve_physical_offset(*this);
}

uint32_t SpmPhaseChunkView::resolve_physical_addr(
    int core,
    const SpmPipelineLease& lease) const {
    return lease.resolve_physical_addr(core, *this);
}

uint32_t SpmDirectProducedTargetView::add_row_offset(uint32_t base) const {
    TORCH_INTERNAL_ASSERT(spec_.rows > 0 && consumer_row_begin_ >= 0);
    const size_t row_bytes = spec_.storage_bytes() /
        static_cast<size_t>(spec_.rows);
    const size_t byte_offset =
        static_cast<size_t>(consumer_row_begin_) * row_bytes;
    TORCH_CHECK(byte_offset <= std::numeric_limits<uint32_t>::max() - base,
                "SpmDirectProducedTargetView: resolved row offset exceeds "
                "uint32 range");
    return base + static_cast<uint32_t>(byte_offset);
}

uint32_t SpmDirectProducedTargetView::resolve_relative(
    const SpmPipelineLease& lease,
    uint64_t current_allocator_generation) const {
    return add_row_offset(
        consumer_.resolve_relative(lease, current_allocator_generation));
}

uint32_t SpmDirectProducedTargetView::resolve_physical_offset(
    const SpmPipelineLease& lease) const {
    return add_row_offset(consumer_.resolve_physical_offset(lease));
}

uint32_t SpmDirectProducedTargetView::resolve_physical_addr(
    int core,
    const SpmPipelineLease& lease) const {
    return add_row_offset(consumer_.resolve_physical_addr(core, lease));
}

void SpmPipelineLease::commit_composite(
    const SpmFmbCompositePhysicalPlan& plan) {
    RpuExecutionCoordinator::require_graph_quiescent(
        "SpmPipelineLease::commit_composite");
    validate_physical(epoch_, plan_.hash(), /*require_sealed=*/false);
    TORCH_CHECK(!physical_sealed_ && composite_reservation_ != nullptr &&
                    composite_plan_ == nullptr,
                "SpmPipelineLease: composite commit requires one unsealed "
                "reservation lease and may occur only once");
    composite_reservation_->validate_live_authority();
    plan.validate_live_authority();
    TORCH_CHECK(
        plan.hash() == plan_.hash() &&
            plan.hash() == composite_reservation_->hash() &&
            plan.producer_occurrence_count_ ==
                composite_reservation_->producer_occurrence_count_ &&
            plan.producer_yield_count_ ==
                composite_reservation_->producer_yield_count() &&
            plan.plan_.peak_bytes_ ==
                composite_reservation_->plan_.peak_bytes_ &&
            plan.plan_.reserved_top_bytes_ ==
                composite_reservation_->plan_.reserved_top_bytes_ &&
            plan.plan_.budget_bytes_ ==
                composite_reservation_->plan_.budget_bytes_ &&
            plan.direct_targets_.size() ==
                composite_reservation_->direct_targets_.size() &&
            plan.run_add_targets_.size() ==
                composite_reservation_->run_add_targets_.size(),
        "SpmPipelineLease: committed composite plan does not match the "
        "acquired reservation");
    for (size_t i = 0; i < plan.direct_targets_.size(); ++i) {
        const auto& committed = plan.direct_targets_[i];
        const auto& reserved =
            composite_reservation_->direct_targets_[i];
        TORCH_CHECK(
            committed.endpoint_hash == reserved.endpoint_hash &&
                committed.producer_local_ordinal ==
                    reserved.producer_local_ordinal &&
                committed.producer_yield_ordinal ==
                    reserved.producer_yield_ordinal &&
                committed.target_relative_offset ==
                    reserved.target_relative_offset &&
                committed.bytes == reserved.bytes,
            "SpmPipelineLease: committed direct target ", i,
            " drifted from its reservation");
    }
    for (size_t i = 0; i < plan.run_add_targets_.size(); ++i) {
        const auto& committed = plan.run_add_targets_[i];
        const auto& reserved =
            composite_reservation_->run_add_targets_[i];
        TORCH_CHECK(
            committed.endpoint_hash == reserved.endpoint_hash &&
                committed.producer_local_ordinal ==
                    reserved.producer_local_ordinal &&
                committed.producer_yield_ordinal ==
                    reserved.producer_yield_ordinal &&
                committed.payload_index == reserved.payload_index &&
                committed.payload_relative_offset ==
                    reserved.payload_relative_offset &&
                committed.consumer_relative_offset ==
                    reserved.consumer_relative_offset &&
                committed.bytes == reserved.bytes &&
                committed.consumer_layer == reserved.consumer_layer &&
                committed.consumer_row_begin ==
                    reserved.consumer_row_begin &&
                committed.consumer_row_count ==
                    reserved.consumer_row_count &&
                committed.spec == reserved.spec,
            "SpmPipelineLease: committed RunAdd target ", i,
            " drifted from its reservation");
    }

    // Allocate the retained copy before publishing committed state.  No lease,
    // allocator, placement, epoch, or generation field changes at commit.
    auto retained =
        std::make_shared<const SpmFmbCompositePhysicalPlan>(plan);
    composite_plan_ = std::move(retained);
}

void SpmPipelineLease::seal_physical() {
    RpuExecutionCoordinator::require_graph_quiescent(
        "SpmPipelineLease::seal_physical");
    validate_physical(epoch_, plan_.hash(), /*require_sealed=*/false);
    TORCH_CHECK(composite_reservation_ == nullptr ||
                    composite_plan_ != nullptr,
                "SpmPipelineLease: composite reservation must be committed "
                "before physical seal");
    physical_sealed_ = true;
}

RpuPhysicalArenaAuthority SpmPipelineLease::physical_graph_authority() const {
    validate_physical(epoch_, plan_.hash(), /*require_sealed=*/true);
    TORCH_CHECK(physical_lease_generation_ != nullptr &&
                    *physical_lease_generation_ == epoch_,
                "SpmPipelineLease: physical Graph authority is stale");
    const uint64_t physical_execution_token =
        RpuExecutionCoordinator::current_physical_token_if_owned(
            "SpmPipelineLease::physical_graph_authority");
    return RpuPhysicalArenaAuthority(
        physical_execution_token, epoch_, plan_.hash(),
        allocator_generation_, physical_persistent_generation_,
        physical_arena_base_, physical_arena_end_,
        physical_top_reservation_, physical_lease_generation_, epoch_,
        owner_thread_);
}

void SpmPipelineLease::release() {
    RpuExecutionCoordinator::check_current_thread_execution_allowed(
        "SpmPipelineLease::release");
    if (!active_) return;
    if (physical_) {
        TORCH_CHECK(physical_execution_claim_.has_value(),
                    "SpmPipelineLease: physical lease has no process claim");
        RpuExecutionCoordinator::validate_physical(
            *physical_execution_claim_, "SpmPipelineLease::release");
        RpuExecutionCoordinator::require_graph_quiescent(
            "SpmPipelineLease::release");
        const bool cancelling_uncommitted_reservation =
            composite_reservation_ != nullptr &&
            composite_plan_ == nullptr && !physical_sealed_;
        validate_physical_impl(epoch_, plan_.hash(),
                               /*require_sealed=*/
                                   !cancelling_uncommitted_reservation,
                               /*validate_phase_seals=*/false);

        GlobalLeaseState& state = global_lease_state();
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            TORCH_CHECK(state.active && state.epoch == epoch_ &&
                            state.plan_hash == plan_.hash(),
                        "SpmPipelineLease: cannot release stale physical state");
            SPM_ALLOC.pipeline_arena_locked_ = false;
            SPM_ALLOC.pipeline_arena_epoch_ = 0;
            state.active = false;
            state.epoch = 0;
            state.plan_hash = 0;
            state.allocator_generation = 0;
            state.owner_thread = std::thread::id{};
            SPM_ALLOC.reset_temporary();
            SPM_ALLOC.release_deferred_floor_if_idle();
        }
        active_ = false;
        physical_sealed_ = false;
        composite_plan_.reset();
        composite_reservation_.reset();
        RpuExecutionCoordinator::exit_physical(
            *physical_execution_claim_, "SpmPipelineLease::release");
        physical_lease_generation_.reset();
        return;
    }
    validate_live_owner(/*validate_phase_seals=*/false);

    GlobalLeaseState& state = global_lease_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.active && state.epoch == epoch_ &&
                    state.plan_hash == plan_.hash(),
                "SpmPipelineLease: cannot release stale global state");
    state.active = false;
    state.epoch = 0;
    state.plan_hash = 0;
    state.allocator_generation = 0;
    state.owner_thread = std::thread::id{};
    active_ = false;
}

void SpmPipelineLease::release_noexcept() noexcept {
    if (!RpuExecutionCoordinator::current_thread_execution_allowed_noexcept()) {
        return;
    }
    if (!active_) return;
    // A destructor or explicit teardown on a foreign thread must be entirely
    // inert: in particular it may not clear the allocator/global owner before
    // discovering that the process claim belongs elsewhere.
    if (std::this_thread::get_id() != owner_thread_) return;
    GlobalLeaseState& state = global_lease_state();
    if (physical_) {
        if (!physical_execution_claim_.has_value() ||
            !RpuExecutionCoordinator::physical_exit_allowed_noexcept(
                *physical_execution_claim_)) {
            return;
        }
        SpmAllocator& allocator = SPM_ALLOC;
        // Never clear the process-global lease while the allocator remains
        // locked: that would orphan the physical arena with no recoverable
        // owner.  Destruction in an active Graph or after ownership drift is a
        // fail-closed contract violation; the explicit Graph-external end path
        // is the only supported release path.
        {
            std::lock_guard<std::mutex> lock(state.mutex);
            if (!allocator.pipeline_arena_locked_ ||
                allocator.pipeline_arena_epoch_ != epoch_ ||
                !state.active || state.epoch != epoch_ ||
                state.plan_hash != plan_.hash() ||
                state.owner_thread != owner_thread_) {
                return;
            }
            allocator.pipeline_arena_locked_ = false;
            allocator.pipeline_arena_epoch_ = 0;
            allocator.reset_temporary();
            allocator.release_deferred_floor_if_idle();
            state.active = false;
            state.epoch = 0;
            state.plan_hash = 0;
            state.allocator_generation = 0;
            state.owner_thread = std::thread::id{};
        }
        active_ = false;
        physical_sealed_ = false;
        composite_plan_.reset();
        composite_reservation_.reset();
        RpuExecutionCoordinator::exit_physical_noexcept(
            *physical_execution_claim_);
        physical_lease_generation_.reset();
        return;
    }

    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.active && state.epoch == epoch_ &&
        state.plan_hash == plan_.hash() &&
        state.owner_thread == owner_thread_) {
        state.active = false;
        state.epoch = 0;
        state.plan_hash = 0;
        state.allocator_generation = 0;
        state.owner_thread = std::thread::id{};
        active_ = false;
    }
}

}  // namespace v3
