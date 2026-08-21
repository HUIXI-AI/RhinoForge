#include "rpu_wall_oss_spm_z1.h"

#include "core/rpu_kernel_decls.h"
#include "core/rpu_ops.h"
#include "core/rpu_spm_allocator.h"

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace {

constexpr int kNumCores = 8;
constexpr int64_t kCanaryPatches = 256;
constexpr int64_t kCanaryMergedRows = 64;
constexpr int64_t kCanaryExecutionLen = 192;
constexpr int64_t kCanaryImageRowBegin = 62;
constexpr int64_t kCanaryRealLen = 180;
constexpr int64_t kCanaryTextHidden = 2048;
constexpr v3::SpmScratchId kVisionScratch{1};
constexpr v3::SpmScratchId kTextScratch{2};
constexpr v3::SpmPortId kMergerSourcePort{1};
constexpr v3::SpmPortId kTextDestinationPort{2};

// Exact reverse_index for the first Wall-OSS grid [1,16,16]. The generic
// route owns only permutation/row-run semantics; this fixed-profile coordinator
// seals the first hardware canary's mapping into its plan hash.
const std::vector<int32_t>& canary_permutation() {
    static const std::vector<int32_t> value = {
         0,  1,  2,  3, 16, 17, 18, 19,
         4,  5,  6,  7, 20, 21, 22, 23,
         8,  9, 10, 11, 24, 25, 26, 27,
        12, 13, 14, 15, 28, 29, 30, 31,
        32, 33, 34, 35, 48, 49, 50, 51,
        36, 37, 38, 39, 52, 53, 54, 55,
        40, 41, 42, 43, 56, 57, 58, 59,
        44, 45, 46, 47, 60, 61, 62, 63,
    };
    return value;
}

struct WallOssZ1State {
    std::mutex mutex;
    std::optional<v3::SpmPipelinePlan> plan;
    std::unique_ptr<v3::SpmPipelineLease> lease;
    int64_t vision_handle = 0;
    int64_t text_handle = 0;
    int64_t num_patches = 0;
    int64_t execution_len = 0;
    int64_t image_row_begin = 0;
    int64_t real_len = 0;
    size_t vision_scratch_bytes = 0;
    size_t text_scratch_bytes = 0;
    size_t persistent_top_bytes = 0;
    uint64_t plan_hash = 0;
    uint64_t route_hash = 0;
    uint32_t source_addr = 0;
    uint32_t destination_addr = 0;
    uint64_t hidden_src_base = 0;
    at::Tensor hidden_ref;
    std::vector<v3::SpmContiguousRowRun> lowered_runs;
    bool pipeline_has_run = false;
};

WallOssZ1State& state() {
    static WallOssZ1State value;
    return value;
}

int64_t encode_u64(uint64_t value) {
    int64_t encoded = 0;
    static_assert(sizeof(encoded) == sizeof(value));
    std::memcpy(&encoded, &value, sizeof(value));
    return encoded;
}

uint64_t decode_u64(int64_t value) {
    uint64_t decoded = 0;
    std::memcpy(&decoded, &value, sizeof(value));
    return decoded;
}

size_t live_persistent_top_bytes() {
    return SPM_ALLOC.super_persistent_used() + SPM_ALLOC.persistent_used();
}

bool ranges_overlap(size_t lhs_begin,
                    size_t lhs_size,
                    size_t rhs_begin,
                    size_t rhs_size) {
    return lhs_begin < rhs_begin + rhs_size &&
           rhs_begin < lhs_begin + lhs_size;
}

void check_canary_shape(int64_t num_patches,
                        int64_t execution_len,
                        int64_t image_row_begin,
                        int64_t real_len,
                        const char* op) {
    TORCH_CHECK(num_patches == kCanaryPatches &&
                    execution_len == kCanaryExecutionLen &&
                    image_row_begin == kCanaryImageRowBegin &&
                    real_len == kCanaryRealLen,
                op, ": first canary accepts only patches/execution/image_begin/real=(",
                kCanaryPatches, ",", kCanaryExecutionLen, ",",
                kCanaryImageRowBegin, ",", kCanaryRealLen, "), got (",
                num_patches, ",", execution_len, ",", image_row_begin,
                ",", real_len, ")");
}

void check_identity(const WallOssZ1State& s,
                    int64_t vision_handle,
                    int64_t text_handle,
                    int64_t num_patches,
                    int64_t execution_len,
                    int64_t image_row_begin,
                    int64_t real_len,
                    uint64_t plan_hash,
                    const char* op) {
    TORCH_CHECK(s.plan.has_value(), op, ": prepare must run first");
    TORCH_CHECK(s.vision_handle == vision_handle &&
                    s.text_handle == text_handle &&
                    s.num_patches == num_patches &&
                    s.execution_len == execution_len &&
                    s.image_row_begin == image_row_begin &&
                    s.real_len == real_len,
                op, ": prepared identity mismatch");
    TORCH_CHECK(s.plan_hash == plan_hash,
                op, ": stale plan hash; expected ", s.plan_hash,
                ", got ", plan_hash);
}

v3::SpmPermutationRowRunsRoute endpoint_route(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin) {
    const auto source =
        v3::wall_oss_z1_internal::vision_source_spec(
            vision_handle, num_patches);
    const auto destination =
        v3::wall_oss_z1_internal::text_destination_spec(
            text_handle, execution_len);
    return v3::SpmPermutationRowRunsRoute::bind(
        kMergerSourcePort, source,
        kTextDestinationPort, destination,
        canary_permutation(),
        {v3::SpmDstRowRun{image_row_begin, kCanaryMergedRows}},
        {/*source_first_write=*/0,
         /*destination_first_write=*/1,
         /*transform=*/1,
         /*destination_last_read=*/2});
}

v3::SpmPermutationRowRunsRoute endpoint_route(const WallOssZ1State& s) {
    return endpoint_route(
        s.vision_handle, s.text_handle, s.num_patches,
        s.execution_len, s.image_row_begin);
}

void validate_endpoint_specs_locked(const WallOssZ1State& s,
                                    const char* op) {
    TORCH_CHECK(s.plan.has_value(), op, ": plan is missing");
    const auto route = endpoint_route(s);
    TORCH_CHECK(route.source_spec() ==
                    s.plan->port_spec(kMergerSourcePort) &&
                    route.destination_spec() ==
                    s.plan->port_spec(kTextDestinationPort),
                op, ": endpoint specs drifted after plan compilation");
    TORCH_CHECK(route.hash() == s.route_hash &&
                    route.hash() == s.plan->permutation_row_runs_hash(
                                        kMergerSourcePort,
                                        kTextDestinationPort),
                op, ": permutation route hash drifted");
}

template <typename Fn>
void cleanup_step(std::exception_ptr& failure, Fn&& fn) noexcept {
    try {
        fn();
    } catch (...) {
        if (!failure) failure = std::current_exception();
    }
}

void reset_begin_state_locked(WallOssZ1State& s) {
    TORCH_INTERNAL_ASSERT(!s.lease);
    s.source_addr = 0;
    s.destination_addr = 0;
    s.hidden_src_base = 0;
    s.hidden_ref = at::Tensor{};
    s.lowered_runs.clear();
    s.pipeline_has_run = false;
}

void reset_prepared_state_locked(WallOssZ1State& s) {
    TORCH_INTERNAL_ASSERT(!s.lease);
    reset_begin_state_locked(s);
    s.plan.reset();
    s.vision_handle = 0;
    s.text_handle = 0;
    s.num_patches = 0;
    s.execution_len = 0;
    s.image_row_begin = 0;
    s.real_len = 0;
    s.vision_scratch_bytes = 0;
    s.text_scratch_bytes = 0;
    s.persistent_top_bytes = 0;
    s.plan_hash = 0;
    s.route_hash = 0;
}

void validate_active_pipeline_locked(WallOssZ1State& s,
                                     uint64_t epoch,
                                     uint64_t plan_hash,
                                     const char* op) {
    TORCH_CHECK(s.lease, op, ": no physical lease is active");
    TORCH_CHECK(s.plan_hash == plan_hash,
                op, ": stale plan hash; expected ", s.plan_hash,
                ", got ", plan_hash);
    s.lease->validate_physical(epoch, plan_hash, /*require_sealed=*/true);
    v3::wall_oss_z1_internal::validate_vision(s.vision_handle, *s.lease);
    v3::wall_oss_z1_internal::validate_text(s.text_handle, *s.lease);
}

void stage_non_image_rows_locked(WallOssZ1State& s,
                                 const at::Tensor& hidden,
                                 uint64_t epoch,
                                 uint64_t plan_hash) {
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "wall_oss_spm_z1 stage text input");
    TORCH_CHECK(hidden.defined() &&
                    hidden.device().type() == at::kPrivateUse1 &&
                    hidden.scalar_type() == at::kHalf &&
                    hidden.is_contiguous() && hidden.dim() == 3 &&
                    hidden.size(0) == 1 &&
                    hidden.size(1) == s.execution_len &&
                    hidden.size(2) == kCanaryTextHidden,
                "wall_oss_spm_z1: text_hidden must be contiguous fp16 RPU "
                "[1,192,2048]");
    TORCH_CHECK(s.destination_addr != 0,
                "wall_oss_spm_z1: destination port is not bound");

    const int64_t image_end = s.image_row_begin + kCanaryMergedRows;
    const int64_t suffix_rows = s.real_len - image_end;
    const int64_t pad_rows = s.execution_len - s.real_len;
    TORCH_CHECK(s.image_row_begin > 0 && suffix_rows > 0 && pad_rows > 0,
                "wall_oss_spm_z1: prefix/suffix/pad must all be non-empty");
    const int64_t row_bytes =
        kCanaryTextHidden * static_cast<int64_t>(sizeof(c10::Half));

    s.hidden_ref = hidden;
    s.hidden_src_base =
        ::rhino_lkn::RpuGetDevAddr(hidden.data_ptr<c10::Half>());
    rpu_ddr_flush_force_sized(hidden.data_ptr<c10::Half>(), hidden.nbytes());

    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &s.hidden_src_base,
        /*src_offset_bytes=*/0,
        s.image_row_begin * kCanaryTextHidden,
        s.destination_addr,
        kNumCores);
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &s.hidden_src_base,
        image_end * row_bytes,
        suffix_rows * kCanaryTextHidden,
        s.destination_addr + image_end * row_bytes,
        kNumCores);
    rpu_launch_fill_spm_kernel(
        s.destination_addr + s.real_len * row_bytes,
        pad_rows * kCanaryTextHidden,
        c10::Half(0.0f),
        kNumCores);
}

void emit_permutation_locked(WallOssZ1State& s,
                             uint64_t epoch,
                             uint64_t plan_hash) {
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "wall_oss_spm_z1 permutation");
    TORCH_CHECK(s.source_addr != 0 && s.destination_addr != 0 &&
                    !s.lowered_runs.empty(),
                "wall_oss_spm_z1: permutation ports are not bound");
    const std::vector<int64_t> input_shape = {
        kCanaryMergedRows, kCanaryTextHidden};
    for (const v3::SpmContiguousRowRun& run : s.lowered_runs) {
        rpu_launch_slice_spm_kernel(
            s.source_addr,
            s.destination_addr + static_cast<uint32_t>(
                run.destination_row * kCanaryTextHidden *
                static_cast<int64_t>(sizeof(c10::Half))),
            input_shape,
            {run.row_count, kCanaryTextHidden},
            {run.source_row, 0},
            kNumCores);
    }
}

}  // namespace

int64_t rpu_wall_oss_spm_z1_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "wall_oss_spm_z1_prepare must run outside Graph capture");
    TORCH_CHECK(vision_handle > 0 && text_handle > 0,
                "wall_oss_spm_z1_prepare: handles must be positive");
    check_canary_shape(num_patches, execution_len, image_row_begin,
                       real_len, "wall_oss_spm_z1_prepare");

    WallOssZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(!s.lease,
                "wall_oss_spm_z1_prepare: physical lease is active");
    if (s.plan.has_value()) {
        check_identity(s, vision_handle, text_handle, num_patches,
                       execution_len, image_row_begin, real_len,
                       s.plan_hash, "wall_oss_spm_z1_prepare");
        validate_endpoint_specs_locked(s, "wall_oss_spm_z1_prepare");
        TORCH_CHECK(live_persistent_top_bytes() == s.persistent_top_bytes,
                    "wall_oss_spm_z1_prepare: persistent top changed after prepare");
        return encode_u64(s.plan_hash);
    }

    bool vision_prepare_attempted = false;
    bool text_prepare_attempted = false;
    try {
        vision_prepare_attempted = true;
        const auto vision_layout =
            v3::wall_oss_z1_internal::prepare_vision(
                vision_handle, num_patches);
        text_prepare_attempted = true;
        const auto text_layout =
            v3::wall_oss_z1_internal::prepare_text(
                text_handle, execution_len, real_len);
        TORCH_CHECK(vision_layout.temporary_bytes > 0 &&
                        text_layout.temporary_bytes > 0,
                    "wall_oss_spm_z1_prepare: component scratch must be non-zero");
        TORCH_CHECK(vision_layout.temporary_bytes <=
                        static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
                        text_layout.temporary_bytes <=
                        static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                    "wall_oss_spm_z1_prepare: component scratch exceeds int64");

        const size_t persistent_top_bytes = live_persistent_top_bytes();
        const std::vector<v3::SpmScratchDecl> scratch = {
            {kVisionScratch,
             static_cast<int64_t>(vision_layout.temporary_bytes),
             /*first_event=*/0, /*last_event=*/0},
            {kTextScratch,
             static_cast<int64_t>(text_layout.temporary_bytes),
             /*first_event=*/2, /*last_event=*/2},
        };
        const auto route = endpoint_route(
            vision_handle, text_handle, num_patches,
            execution_len, image_row_begin);
        auto plan = v3::SpmPipelinePlan::compile(
            scratch, std::vector<v3::SpmPermutationRowRunsRoute>{route},
            persistent_top_bytes, SpmAllocator::SPM_PLANNING_BUDGET);
        const uint64_t route_hash = route.hash();
        const uint64_t plan_hash = plan.hash();
        TORCH_CHECK(route.source_spec() ==
                        plan.port_spec(kMergerSourcePort) &&
                        route.destination_spec() ==
                        plan.port_spec(kTextDestinationPort) &&
                        route_hash == plan.permutation_row_runs_hash(
                                          kMergerSourcePort,
                                          kTextDestinationPort),
                    "wall_oss_spm_z1_prepare: compiled endpoint route drifted");

        // Commit the process-global identity only after both component prepares,
        // route binding, and plan compilation have succeeded.
        s.vision_handle = vision_handle;
        s.text_handle = text_handle;
        s.num_patches = num_patches;
        s.execution_len = execution_len;
        s.image_row_begin = image_row_begin;
        s.real_len = real_len;
        s.vision_scratch_bytes = vision_layout.temporary_bytes;
        s.text_scratch_bytes = text_layout.temporary_bytes;
        s.persistent_top_bytes = persistent_top_bytes;
        s.route_hash = route_hash;
        s.plan_hash = plan_hash;
        s.plan = std::move(plan);
        validate_endpoint_specs_locked(s, "wall_oss_spm_z1_prepare");
        return encode_u64(s.plan_hash);
    } catch (...) {
        std::exception_ptr failure = std::current_exception();
        if (text_prepare_attempted) cleanup_step(failure, [&] {
            v3::wall_oss_z1_internal::unprepare_text(text_handle);
        });
        if (vision_prepare_attempted) cleanup_step(failure, [&] {
            v3::wall_oss_z1_internal::unprepare_vision(vision_handle);
        });
        reset_prepared_state_locked(s);
        std::rethrow_exception(failure);
    }
}

void rpu_wall_oss_spm_z1_unprepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "wall_oss_spm_z1_unprepare must run outside Graph capture");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    WallOssZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.plan.has_value(),
                "wall_oss_spm_z1_unprepare: no prepared plan");
    TORCH_CHECK(!s.lease,
                "wall_oss_spm_z1_unprepare: physical lease is active");
    TORCH_CHECK(s.vision_handle == vision_handle &&
                    s.text_handle == text_handle &&
                    s.plan_hash == plan_hash,
                "wall_oss_spm_z1_unprepare: stale handle or plan token");
    validate_endpoint_specs_locked(s, "wall_oss_spm_z1_unprepare");

    std::exception_ptr failure;
    cleanup_step(failure, [&] {
        v3::wall_oss_z1_internal::unprepare_text(text_handle);
    });
    cleanup_step(failure, [&] {
        v3::wall_oss_z1_internal::unprepare_vision(vision_handle);
    });
    if (failure) std::rethrow_exception(failure);
    reset_prepared_state_locked(s);
}

int64_t rpu_wall_oss_spm_z1_begin(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len,
    const at::Tensor& window_mask,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "wall_oss_spm_z1_begin must run outside Graph capture");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    WallOssZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    check_identity(s, vision_handle, text_handle, num_patches,
                   execution_len, image_row_begin, real_len,
                   plan_hash, "wall_oss_spm_z1_begin");
    TORCH_CHECK(!s.lease,
                "wall_oss_spm_z1_begin: a physical lease is already active");
    validate_endpoint_specs_locked(s, "wall_oss_spm_z1_begin");
    TORCH_CHECK(live_persistent_top_bytes() == s.persistent_top_bytes,
                "wall_oss_spm_z1_begin: persistent top changed after prepare");

    reset_begin_state_locked(s);
    bool vision_prime_attempted = false;
    bool text_prime_attempted = false;
    bool vision_adopted = false;
    bool text_adopted = false;
    std::unique_ptr<v3::SpmPipelineLease> lease;
    uint32_t source_addr = 0;
    uint32_t destination_addr = 0;
    std::vector<v3::SpmContiguousRowRun> lowered_runs;
    try {
        // Host mask materialization and position keepalive flush belong outside
        // the composite Graph. Mark attempted before each call because stable
        // DDR state may be partially overwritten before validation throws.
        vision_prime_attempted = true;
        v3::wall_oss_z1_internal::prime_vision_inputs(
            vision_handle, window_mask, num_patches);
        text_prime_attempted = true;
        v3::wall_oss_z1_internal::prime_text_inputs(
            text_handle, position_ids, rope_cos_il, rope_sin_il,
            execution_len);

        lease = v3::SpmPipelineLease::acquire_physical(*s.plan);
        const auto vision_scratch = lease->scratch(kVisionScratch);
        const auto text_scratch = lease->scratch(kTextScratch);
        const auto route = lease->permutation_row_runs(
            kMergerSourcePort, kTextDestinationPort);
        TORCH_CHECK(route.route_hash() == s.route_hash,
                    "wall_oss_spm_z1_begin: checked route hash drifted");

        const size_t vision_scratch_offset =
            vision_scratch.resolve_physical_offset(*lease);
        const size_t text_scratch_offset =
            text_scratch.resolve_physical_offset(*lease);
        const size_t source_offset =
            route.source().resolve_physical_offset(*lease);
        const size_t destination_offset =
            route.destination().resolve_physical_offset(*lease);
        TORCH_CHECK(!ranges_overlap(source_offset, route.source().size_bytes(),
                                    destination_offset,
                                    route.destination().size_bytes()),
                    "wall_oss_spm_z1_begin: transform source/destination overlap");
        TORCH_CHECK(!ranges_overlap(
                        vision_scratch_offset, vision_scratch.size_bytes(),
                        source_offset, route.source().size_bytes()),
                    "wall_oss_spm_z1_begin: Vision scratch/source overlap at producer write");
        TORCH_CHECK(!ranges_overlap(
                        text_scratch_offset, text_scratch.size_bytes(),
                        destination_offset, route.destination().size_bytes()),
                    "wall_oss_spm_z1_begin: text scratch/destination overlap at consume");

        v3::wall_oss_z1_internal::adopt_vision(
            vision_handle, *lease, vision_scratch);
        vision_adopted = true;
        v3::wall_oss_z1_internal::adopt_text(
            text_handle, *lease, text_scratch);
        text_adopted = true;
        v3::wall_oss_z1_internal::bind_vision_source(
            vision_handle, *lease, route.source());
        v3::wall_oss_z1_internal::bind_text_destination(
            text_handle, *lease, route.destination());
        lease->seal_physical();
        v3::wall_oss_z1_internal::validate_vision(
            vision_handle, *lease);
        v3::wall_oss_z1_internal::validate_text(
            text_handle, *lease);
        source_addr =
            route.source().resolve_physical_addr(/*core=*/0, *lease);
        destination_addr =
            route.destination().resolve_physical_addr(/*core=*/0, *lease);
        lowered_runs = route.lower_contiguous_runs();
        TORCH_CHECK(lowered_runs.size() == 15,
                    "wall_oss_spm_z1: fixed reverse permutation must lower "
                    "to 15 contiguous source runs, got ",
                    lowered_runs.size());
    } catch (...) {
        std::exception_ptr failure = std::current_exception();
        if (text_adopted) cleanup_step(failure, [&] {
            v3::wall_oss_z1_internal::clear_text(
                text_handle, lease->epoch(), lease->plan_hash());
        }); else if (text_prime_attempted) cleanup_step(failure, [&] {
            v3::wall_oss_z1_internal::rollback_text_inputs(text_handle);
        });
        if (vision_adopted) cleanup_step(failure, [&] {
            v3::wall_oss_z1_internal::clear_vision(
                vision_handle, lease->epoch(), lease->plan_hash());
        }); else if (vision_prime_attempted) cleanup_step(failure, [&] {
            v3::wall_oss_z1_internal::rollback_vision_inputs(vision_handle);
        });
        if (lease) {
            if (!lease->physical_sealed()) cleanup_step(failure, [&] {
                lease->seal_physical();
            });
            cleanup_step(failure, [&] { lease->release(); });
        }
        reset_begin_state_locked(s);
        std::rethrow_exception(failure);
    }

    s.source_addr = source_addr;
    s.destination_addr = destination_addr;
    s.lowered_runs.swap(lowered_runs);
    s.pipeline_has_run = false;
    const uint64_t epoch = lease->epoch();
    s.lease = std::move(lease);
    return encode_u64(epoch);
}

void rpu_wall_oss_spm_z1_end(int64_t encoded_epoch,
                             int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "wall_oss_spm_z1_end must run outside Graph capture");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    WallOssZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.lease,
                "wall_oss_spm_z1_end: no physical lease is active");
    s.lease->validate_physical_identity(epoch, plan_hash);

    std::exception_ptr failure;
    cleanup_step(failure, [&] {
        v3::wall_oss_z1_internal::validate_vision(
            s.vision_handle, *s.lease);
        v3::wall_oss_z1_internal::validate_text(
            s.text_handle, *s.lease);
    });
    cleanup_step(failure, [&] {
        v3::wall_oss_z1_internal::clear_text(
            s.text_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] {
        v3::wall_oss_z1_internal::clear_vision(
            s.vision_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] { s.lease->release(); });
    s.lease.reset();
    s.source_addr = 0;
    s.destination_addr = 0;
    s.hidden_src_base = 0;
    s.hidden_ref = at::Tensor{};
    s.lowered_runs.clear();
    s.pipeline_has_run = false;
    if (failure) std::rethrow_exception(failure);
}

at::Tensor rpu_wall_oss_spm_z1_pipeline_forward(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& vision_input,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& window_mask,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t encoded_epoch,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "wall_oss_spm_z1_pipeline_forward requires one active outer Graph");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    {
        WallOssZ1State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        TORCH_CHECK(s.vision_handle == vision_handle &&
                        s.text_handle == text_handle,
                    "wall_oss_spm_z1_pipeline_forward: active handle mismatch");
        validate_active_pipeline_locked(
            s, epoch, plan_hash, "wall_oss_spm_z1_pipeline_forward");
    }

    v3::wall_oss_z1_internal::forward_vision_z1(
        vision_handle, vision_input, vision_k_caches, vision_v_caches,
        kCanaryPatches, window_mask, epoch, plan_hash);

    {
        WallOssZ1State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        stage_non_image_rows_locked(s, text_hidden, epoch, plan_hash);
        emit_permutation_locked(s, epoch, plan_hash);
    }

    at::Tensor result = v3::wall_oss_z1_internal::forward_text_z1(
        text_handle, text_hidden, text_k_caches, text_v_caches,
        position_ids, rope_cos_il, rope_sin_il, epoch, plan_hash);
    {
        WallOssZ1State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        validate_active_pipeline_locked(
            s, epoch, plan_hash, "wall_oss_spm_z1_pipeline_forward complete");
        s.pipeline_has_run = true;
    }
    return result;
}

at::Tensor rpu_wall_oss_spm_z1_export_destination(
    int64_t encoded_epoch,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "wall_oss_spm_z1_export_destination must run outside Graph capture");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    WallOssZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "wall_oss_spm_z1_export_destination");
    TORCH_CHECK(s.pipeline_has_run && s.hidden_ref.defined() &&
                    s.destination_addr != 0,
                "wall_oss_spm_z1_export_destination: pipeline has not run");
    at::Tensor out = at::empty(
        {1, kCanaryExecutionLen, kCanaryTextHidden},
        s.hidden_ref.options());
    rpu_ddr_flush_force_sized(out.data_ptr<c10::Half>(), out.nbytes());
    rpu_launch_spm_copy_ddr_dma(
        s.destination_addr, out.data_ptr<c10::Half>(),
        kCanaryExecutionLen * kCanaryTextHidden);
    rpu_ddr_flush_force_sized(out.data_ptr<c10::Half>(), out.nbytes());
    return out;
}
