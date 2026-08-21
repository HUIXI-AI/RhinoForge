#include "rpu_qwen3_5_spm_z2.h"

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

namespace {

constexpr int kNumCores = 8;
constexpr int64_t kCanaryPatches = 256;
constexpr int64_t kCanaryExecutionLen = 128;
constexpr int64_t kCanaryImageRowBegin = 4;
constexpr int64_t kCanaryImageRows = 64;
constexpr int64_t kCanaryRealLen = 72;
constexpr int64_t kCanaryTextHidden = 2048;
constexpr v3::SpmScratchId kScratchRegion{1};
constexpr v3::SpmPortId kTextInputPort{1};

struct Qwen3_5Z2State {
    std::mutex mutex;
    std::optional<v3::SpmPipelinePlan> plan;
    std::unique_ptr<v3::SpmPipelineLease> lease;
    int64_t vision_handle = 0;
    int64_t text_handle = 0;
    int64_t num_patches = 0;
    int64_t execution_len = 0;
    int64_t image_row_begin = 0;
    int64_t real_len = 0;
    size_t component_scratch_bytes = 0;
    size_t persistent_top_bytes = 0;
    uint64_t plan_hash = 0;
    uint32_t text_port_addr = 0;
    uint32_t image_slice_addr = 0;
    uint64_t hidden_src_base = 0;
    at::Tensor hidden_ref;
};

Qwen3_5Z2State& state() {
    static Qwen3_5Z2State value;
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

size_t align_256(size_t value) {
    TORCH_CHECK(value <= std::numeric_limits<size_t>::max() - 255,
                "qwen3_5_spm_z2: byte alignment overflow");
    return (value + 255) & ~static_cast<size_t>(255);
}

size_t live_persistent_top_bytes() {
    return SPM_ALLOC.super_persistent_used() + SPM_ALLOC.persistent_used();
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

void check_identity(const Qwen3_5Z2State& s,
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

v3::SpmRowSliceRoute endpoint_route(const Qwen3_5Z2State& s) {
    const auto produced =
        v3::qwen3_5_z2_internal::vision_produced_spec(
            s.vision_handle, s.num_patches);
    const auto storage =
        v3::qwen3_5_z2_internal::text_storage_spec(
            s.text_handle, s.execution_len);
    return v3::SpmRowSliceRoute::bind(
        kTextInputPort, produced, storage,
        s.image_row_begin, produced.rows,
        /*first_write=*/0, /*last_read=*/2);
}

void validate_endpoint_specs_locked(const Qwen3_5Z2State& s,
                                    const char* op) {
    TORCH_CHECK(s.plan.has_value(), op, ": plan is missing");
    const auto route = endpoint_route(s);
    TORCH_CHECK(route.dst_row_count() == kCanaryImageRows &&
                    route.dst_row_begin() + route.dst_row_count() <= s.real_len,
                op, ": row-slice semantics drifted");
    TORCH_CHECK(route.storage_spec() == s.plan->port_spec(kTextInputPort),
                op, ": endpoint storage spec drifted after plan compilation");
}

template <typename Fn>
void cleanup_step(std::exception_ptr& failure, Fn&& fn) noexcept {
    try {
        fn();
    } catch (...) {
        if (!failure) failure = std::current_exception();
    }
}

void validate_active_pipeline_locked(Qwen3_5Z2State& s,
                                     uint64_t epoch,
                                     uint64_t plan_hash,
                                     const char* op) {
    TORCH_CHECK(s.lease, op, ": no physical lease is active");
    TORCH_CHECK(s.plan_hash == plan_hash,
                op, ": stale plan hash; expected ", s.plan_hash,
                ", got ", plan_hash);
    s.lease->validate_physical(epoch, plan_hash, /*require_sealed=*/true);
    v3::qwen3_5_z2_internal::validate_vision(
        s.vision_handle, *s.lease);
    v3::qwen3_5_z2_internal::validate_text(
        s.text_handle, *s.lease);
}

void stage_non_image_rows_locked(Qwen3_5Z2State& s,
                                 const at::Tensor& hidden,
                                 uint64_t epoch,
                                 uint64_t plan_hash) {
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "qwen3_5_spm_z2 stage text input");
    TORCH_CHECK(hidden.defined() && hidden.device().type() == at::kPrivateUse1 &&
                    hidden.scalar_type() == at::kHalf && hidden.is_contiguous() &&
                    hidden.dim() == 3 && hidden.size(0) == 1 &&
                    hidden.size(1) == s.execution_len &&
                    hidden.size(2) == kCanaryTextHidden,
                "qwen3_5_spm_z2: hidden must be contiguous fp16 RPU [1,128,2048]");
    TORCH_CHECK(s.text_port_addr != 0 && s.image_slice_addr != 0,
                "qwen3_5_spm_z2: physical text port is not bound");

    const int64_t image_end = s.image_row_begin + kCanaryImageRows;
    const int64_t suffix_rows = s.real_len - image_end;
    const int64_t pad_rows = s.execution_len - s.real_len;
    TORCH_CHECK(s.image_row_begin > 0 && suffix_rows > 0 && pad_rows > 0,
                "qwen3_5_spm_z2: canary prefix/suffix/pad must be non-empty");
    const int64_t row_bytes =
        kCanaryTextHidden * static_cast<int64_t>(sizeof(c10::Half));

    s.hidden_ref = hidden;
    s.hidden_src_base =
        ::rhino_lkn::RpuGetDevAddr(hidden.data_ptr<c10::Half>());
    rpu_ddr_flush_force_sized(
        hidden.data_ptr<c10::Half>(), hidden.nbytes());

    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &s.hidden_src_base,
        /*src_offset_bytes=*/0,
        s.image_row_begin * kCanaryTextHidden,
        s.text_port_addr,
        kNumCores);
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &s.hidden_src_base,
        image_end * row_bytes,
        suffix_rows * kCanaryTextHidden,
        s.text_port_addr + image_end * row_bytes,
        kNumCores);
    rpu_launch_fill_spm_kernel(
        s.text_port_addr + s.real_len * row_bytes,
        pad_rows * kCanaryTextHidden,
        c10::Half(0.0f),
        kNumCores);
}

}  // namespace

int64_t rpu_qwen3_5_spm_z2_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3_5_spm_z2_prepare must run outside Graph capture");
    TORCH_CHECK(vision_handle > 0 && text_handle > 0,
                "qwen3_5_spm_z2_prepare: handles must be positive");
    check_canary_shape(num_patches, execution_len, image_row_begin,
                       real_len, "qwen3_5_spm_z2_prepare");

    Qwen3_5Z2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(!s.lease,
                "qwen3_5_spm_z2_prepare: physical lease is active");
    if (s.plan.has_value()) {
        check_identity(s, vision_handle, text_handle, num_patches,
                       execution_len, image_row_begin, real_len,
                       s.plan_hash, "qwen3_5_spm_z2_prepare");
        validate_endpoint_specs_locked(s, "qwen3_5_spm_z2_prepare");
        TORCH_CHECK(live_persistent_top_bytes() == s.persistent_top_bytes,
                    "qwen3_5_spm_z2_prepare: persistent top changed after prepare");
        return encode_u64(s.plan_hash);
    }

    const auto vision_layout =
        v3::qwen3_5_z2_internal::prepare_vision(
            vision_handle, num_patches);
    const auto text_layout =
        v3::qwen3_5_z2_internal::prepare_text(
            text_handle, execution_len, real_len);
    TORCH_CHECK(vision_layout.temporary_bytes > 0 &&
                    text_layout.temporary_bytes > 0,
                "qwen3_5_spm_z2_prepare: component scratch must be non-zero");

    s.vision_handle = vision_handle;
    s.text_handle = text_handle;
    s.num_patches = num_patches;
    s.execution_len = execution_len;
    s.image_row_begin = image_row_begin;
    s.real_len = real_len;
    s.component_scratch_bytes = std::max(
        vision_layout.temporary_bytes, text_layout.temporary_bytes);
    s.persistent_top_bytes = live_persistent_top_bytes();

    const v3::SpmScratchDecl scratch{
        kScratchRegion,
        static_cast<int64_t>(s.component_scratch_bytes),
        /*first_event=*/0,
        /*last_event=*/2};
    const auto route = endpoint_route(s);
    auto plan = v3::SpmPipelinePlan::compile(
        {scratch}, {route}, s.persistent_top_bytes,
        SpmAllocator::SPM_PLANNING_BUDGET);
    TORCH_CHECK(plan.peak_bytes() ==
                    align_256(s.component_scratch_bytes) +
                        align_256(route.storage_spec().storage_bytes()),
                "qwen3_5_spm_z2_prepare: scratch and row port unexpectedly aliased");
    s.plan_hash = plan.hash();
    s.plan = std::move(plan);
    return encode_u64(s.plan_hash);
}

int64_t rpu_qwen3_5_spm_z2_begin(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3_5_spm_z2_begin must run outside Graph capture");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3_5Z2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    check_identity(s, vision_handle, text_handle, num_patches,
                   execution_len, image_row_begin, real_len,
                   plan_hash, "qwen3_5_spm_z2_begin");
    TORCH_CHECK(!s.lease,
                "qwen3_5_spm_z2_begin: a physical lease is already active");
    validate_endpoint_specs_locked(s, "qwen3_5_spm_z2_begin");
    TORCH_CHECK(live_persistent_top_bytes() == s.persistent_top_bytes,
                "qwen3_5_spm_z2_begin: persistent top changed after prepare");

    auto lease = v3::SpmPipelineLease::acquire_physical(*s.plan);
    const auto scratch = lease->scratch(kScratchRegion);
    const auto storage = lease->port(kTextInputPort);
    const auto image_slice = storage.row_slice(
        s.image_row_begin, kCanaryImageRows);
    TORCH_CHECK(storage.resolve_physical_offset(*lease) ==
                    scratch.resolve_physical_offset(*lease) +
                        align_256(s.component_scratch_bytes),
                "qwen3_5_spm_z2_begin: row port must be physically disjoint "
                "from the live component scratch");

    bool vision_adopted = false;
    bool text_adopted = false;
    try {
        v3::qwen3_5_z2_internal::adopt_vision(
            vision_handle, *lease, scratch);
        vision_adopted = true;
        v3::qwen3_5_z2_internal::adopt_text(
            text_handle, *lease, scratch);
        text_adopted = true;
        v3::qwen3_5_z2_internal::bind_vision_slice(
            vision_handle, *lease, image_slice);
        v3::qwen3_5_z2_internal::bind_text(
            text_handle, *lease, storage);
        lease->seal_physical();
        v3::qwen3_5_z2_internal::validate_vision(
            vision_handle, *lease);
        v3::qwen3_5_z2_internal::validate_text(
            text_handle, *lease);
    } catch (...) {
        std::exception_ptr failure = std::current_exception();
        if (text_adopted) cleanup_step(failure, [&] {
            v3::qwen3_5_z2_internal::clear_text(
                text_handle, lease->epoch(), lease->plan_hash());
        });
        if (vision_adopted) cleanup_step(failure, [&] {
            v3::qwen3_5_z2_internal::clear_vision(
                vision_handle, lease->epoch(), lease->plan_hash());
        });
        if (!lease->physical_sealed()) cleanup_step(failure, [&] {
            lease->seal_physical();
        });
        cleanup_step(failure, [&] { lease->release(); });
        std::rethrow_exception(failure);
    }

    s.text_port_addr = storage.resolve_physical_addr(/*core=*/0, *lease);
    s.image_slice_addr =
        image_slice.resolve_physical_addr(/*core=*/0, *lease);
    const uint64_t epoch = lease->epoch();
    s.lease = std::move(lease);
    return encode_u64(epoch);
}

void rpu_qwen3_5_spm_z2_end(int64_t encoded_epoch,
                            int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3_5_spm_z2_end must run outside Graph capture");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3_5Z2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.lease,
                "qwen3_5_spm_z2_end: no physical lease is active");
    s.lease->validate_physical_identity(epoch, plan_hash);

    std::exception_ptr failure;
    cleanup_step(failure, [&] {
        v3::qwen3_5_z2_internal::validate_vision(
            s.vision_handle, *s.lease);
        v3::qwen3_5_z2_internal::validate_text(
            s.text_handle, *s.lease);
    });
    cleanup_step(failure, [&] {
        v3::qwen3_5_z2_internal::clear_text(
            s.text_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] {
        v3::qwen3_5_z2_internal::clear_vision(
            s.vision_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] { s.lease->release(); });
    s.lease.reset();
    s.text_port_addr = 0;
    s.image_slice_addr = 0;
    s.hidden_src_base = 0;
    s.hidden_ref = at::Tensor{};
    if (failure) std::rethrow_exception(failure);
}

at::Tensor rpu_qwen3_5_spm_z2_pipeline_forward(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& vision_input,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& step0_pos,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    at::TensorList gdn_states,
    at::TensorList conv_states,
    int64_t encoded_epoch,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "qwen3_5_spm_z2_pipeline_forward requires one active outer Graph");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    {
        Qwen3_5Z2State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        TORCH_CHECK(s.vision_handle == vision_handle &&
                        s.text_handle == text_handle,
                    "qwen3_5_spm_z2_pipeline_forward: active handle mismatch");
        stage_non_image_rows_locked(
            s, text_hidden, epoch, plan_hash);
    }
    v3::qwen3_5_z2_internal::forward_vision_z2(
        vision_handle, vision_input, vision_k_caches, vision_v_caches,
        kCanaryPatches, step0_pos, epoch, plan_hash);
    return v3::qwen3_5_z2_internal::forward_text_z2(
        text_handle, text_hidden, text_k_caches, text_v_caches,
        gdn_states, conv_states, epoch, plan_hash);
}

at::Tensor rpu_qwen3_5_spm_z2_export_image_slice(
    int64_t encoded_epoch,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3_5_spm_z2_export_image_slice must run outside Graph capture");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3_5Z2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "qwen3_5_spm_z2_export_image_slice");
    TORCH_CHECK(s.hidden_ref.defined() && s.image_slice_addr != 0,
                "qwen3_5_spm_z2_export_image_slice: pipeline has not run");
    at::Tensor out = at::empty(
        {1, kCanaryImageRows, kCanaryTextHidden}, s.hidden_ref.options());
    rpu_ddr_flush_force_sized(out.data_ptr<c10::Half>(), out.nbytes());
    rpu_launch_spm_copy_ddr_dma(
        s.image_slice_addr, out.data_ptr<c10::Half>(),
        kCanaryImageRows * kCanaryTextHidden);
    rpu_ddr_flush_force_sized(out.data_ptr<c10::Half>(), out.nbytes());
    return out;
}

namespace v3::qwen3_5_z2_internal {

void validate_vision_dispatch(int64_t handle,
                              uint64_t epoch,
                              uint64_t plan_hash) {
    Qwen3_5Z2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.vision_handle == handle,
                "Qwen3.5 Vision Z2 dispatch: active handle mismatch");
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "Qwen3.5 Vision Z2 dispatch");
}

void validate_text_dispatch(int64_t handle,
                            uint64_t epoch,
                            uint64_t plan_hash) {
    Qwen3_5Z2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.text_handle == handle,
                "Qwen3.5 text Z2 dispatch: active handle mismatch");
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "Qwen3.5 text Z2 dispatch");
}

void check_vision_destroy_allowed(int64_t handle) {
    Qwen3_5Z2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(!s.lease || s.vision_handle != handle,
                "cannot destroy the Qwen3.5 Vision handle while its Z2 lease is active");
}

void check_text_destroy_allowed(int64_t handle) {
    Qwen3_5Z2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(!s.lease || s.text_handle != handle,
                "cannot destroy the Qwen3.5 text handle while its Z2 lease is active");
}

}  // namespace v3::qwen3_5_z2_internal
