// rpu_runtime_extras.cpp — Miscellaneous runtime helpers.
//
// This translation unit owns three helper sets used by surviving callers and
// Python bindings:
//
//   - set_spm_debug / get_spm_debug                          (Python-bound)
//   - set_cross_layer_batch_prefill / get_cross_layer_batch_prefill  (Python-bound)
//   - rpu_launch_all_reduce_sum_residual_kernel              (fused-model launch paths)
//
// Forward declarations stay in `src/core/rpu_kernel_decls.h`.

#include "rhino_launch_buffer.h"
#include "rhino_launch_kernel.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "fused_model_base.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rpu_spm_pipeline.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using namespace ::rhino_lkn;  // Kernel_t / Queue_t live in this namespace

namespace v3 {

struct SpmFmbYieldTargetLauncherAccess {
    static uint32_t resolve_and_stage_all_reduce(
        const SpmFmbPostFnYieldTarget& target,
        int64_t rows,
        int64_t cols,
        int input_num_cores,
        int output_num_cores) {
        target.spec_.validate();
        TORCH_CHECK(
            target.semantic_id_ != 0 &&
                target.writer_kind_ ==
                    SpmFmbTerminalWriterKind::AllReduceSumResidual &&
                target.spec_.dtype == SpmPortDType::Fp16 &&
                target.spec_.distribution ==
                    SpmPortDistribution::Replicated &&
                target.spec_.rows == rows && target.spec_.cols == cols &&
                cols % 16 == 0 && input_num_cores == 8 &&
                output_num_cores == 8,
            "typed post_fn yield all-reduce target/spec/core mismatch");
        const std::shared_ptr<uint64_t> owner =
            target.owner_generation_.lock();
        TORCH_CHECK(
            owner != nullptr && target.expected_owner_generation_ != 0 &&
                *owner == target.expected_owner_generation_,
            "typed post_fn yield all-reduce owner is stale");
        RpuKernelGraph& graph = RpuKernelGraph::active();
        graph.stage_semantic_spm_producer_yield_for_fmb(
            owner, target.expected_owner_generation_, target.semantic_id_,
            static_cast<uint8_t>(target.writer_kind_));
        return target.address_;
    }

    static void cancel_pending() noexcept {
        if (RpuKernelGraph::has_active()) {
            RpuKernelGraph::active()
                .cancel_semantic_spm_producer_yield_for_fmb();
        }
    }

    static bool is_cpu_dry(const SpmFmbPostFnYieldTarget& target) {
        return target.cpu_dry_;
    }
};

}  // namespace v3

// =============================================================================
// File-local statics
// =============================================================================

static bool g_spm_debug_enabled = false;            // SPM/chunk debug output
static bool g_cross_layer_batch_prefill = false;    // cross-layer batch for multi-chunk prefill

// =============================================================================
// SPM debug toggles (Python-bound at rpu_backend.cpp)
// =============================================================================

void set_spm_debug(bool enabled) {
    g_spm_debug_enabled = enabled;
}

bool get_spm_debug() {
    return g_spm_debug_enabled;
}

// =============================================================================
// Cross-layer batch prefill toggles (Python-bound at rpu_backend.cpp)
// =============================================================================

void set_cross_layer_batch_prefill(bool enabled) {
    g_cross_layer_batch_prefill = enabled;
}

bool get_cross_layer_batch_prefill() {
    return g_cross_layer_batch_prefill;
}

// =============================================================================
// Fused ring all-reduce + residual
// =============================================================================
//
// Operator-asset ABI contract: every standard all-reduce uses one fused ring
// kernel. The strict per-core crossover selects nopace below 11 KiB and paced
// otherwise.

namespace {

constexpr int kRingCoreNum = 8;
constexpr uint32_t kRingWarpNum = 8;
constexpr uint32_t kRingRoundElemMax = 31U * 256U * kRingWarpNum;
constexpr uint64_t kRingNopaceChunkBytes = 11U * 1024U;

bool ring_all_reduce_sum_residual_admitted(
    uint32_t input_spm,
    uint32_t residual_spm,
    const c10::Half* residual_ddr,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int core_num,
    bool cpu_dry = false) {
    if (core_num != kRingCoreNum ||
        M <= 0 || N <= 0 || !SPM_ALLOC.is_initialized() ||
        (cpu_dry && residual_ddr != nullptr)) {
        return false;
    }
    constexpr uint64_t kFp16Bytes = sizeof(c10::Half);
    const uint64_t spm_capacity = SpmAllocator::SPM_USABLE;
    if (static_cast<uint64_t>(N) > spm_capacity / kFp16Bytes) return false;
    const uint64_t row_bytes = static_cast<uint64_t>(N) * kFp16Bytes;
    if (static_cast<uint64_t>(M) > spm_capacity / row_bytes) return false;
    const uint64_t tensor_bytes = static_cast<uint64_t>(M) * row_bytes;
    const uint64_t total_elements = tensor_bytes / kFp16Bytes;
    if (total_elements > UINT32_MAX) return false;
    const uint64_t avg_elements =
        ((total_elements + static_cast<uint64_t>(core_num) - 1) /
             static_cast<uint64_t>(core_num) +
         15U) /
        16U * 16U;
    const uint64_t residual_bytes = tensor_bytes;

    const uint64_t spm_base0 = cpu_dry ? 0 : SPM_ALLOC.addr(0, 0);
    const auto range_is_current_core0_spm =
        [spm_base0, spm_capacity](uint32_t address, uint64_t bytes) {
            const uint64_t begin = address;
            return bytes <= spm_capacity && begin >= spm_base0 &&
                begin - spm_base0 <= spm_capacity - bytes;
        };
    if (!range_is_current_core0_spm(input_spm, tensor_bytes) ||
        !range_is_current_core0_spm(output_spm, tensor_bytes) ||
        (residual_ddr == nullptr &&
         !range_is_current_core0_spm(residual_spm, residual_bytes))) {
        return false;
    }

    const auto ranges_are_disjoint =
        [](uint32_t lhs, uint64_t lhs_bytes,
           uint32_t rhs, uint64_t rhs_bytes) {
            const uint64_t lhs_begin = lhs;
            const uint64_t rhs_begin = rhs;
            return lhs_begin + lhs_bytes <= rhs_begin ||
                rhs_begin + rhs_bytes <= lhs_begin;
        };
    if (!ranges_are_disjoint(
            input_spm, tensor_bytes, output_spm, tensor_bytes)) {
        return false;
    }
    return residual_ddr != nullptr ||
        (ranges_are_disjoint(
             input_spm, tensor_bytes, residual_spm, residual_bytes) &&
         ranges_are_disjoint(
             residual_spm, residual_bytes, output_spm, tensor_bytes));
}

void launch_ring_all_reduce_sum_residual(
    uint32_t input_spm,
    uint32_t residual_spm,
    const c10::Half* residual_ddr,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int core_num) {
    TORCH_CHECK(
        ring_all_reduce_sum_residual_admitted(
            input_spm, residual_spm, residual_ddr, output_spm,
            M, N, core_num),
        "ring all-reduce requires 8 cores, positive FP16 geometry, "
        "complete out-of-place SPM operands, and an initialized current "
        "SPM arena");

    constexpr uint64_t kFp16Bytes = sizeof(c10::Half);
    const uint64_t total_elements =
        static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
    const uint64_t spm_base0 = SPM_ALLOC.addr(0, 0);

    // Operator-asset contract: ceil-align(ceil-div(total, cores), 16 elements).
    const uint64_t avg_elements_unaligned =
        (total_elements + static_cast<uint64_t>(core_num) - 1) /
        static_cast<uint64_t>(core_num);
    const uint64_t avg_elements =
        ((avg_elements_unaligned + 15U) / 16U) * 16U;
    TORCH_INTERNAL_ASSERT(avg_elements <= UINT32_MAX);
    const uint64_t chunk_bytes = avg_elements * kFp16Bytes;
    const bool use_nopace = chunk_bytes < kRingNopaceChunkBytes;

    std::array<uint32_t, kRingCoreNum> chunk_offsets{};
    std::array<uint32_t, kRingCoreNum> chunk_sizes{};
    uint32_t num_rounds = 1;
    for (int core = 0; core < core_num; ++core) {
        const uint64_t offset = std::min<uint64_t>(
            static_cast<uint64_t>(core) * avg_elements, total_elements);
        const uint64_t size = std::min<uint64_t>(
            avg_elements, total_elements - offset);
        chunk_offsets[core] = static_cast<uint32_t>(offset);
        chunk_sizes[core] = static_cast<uint32_t>(size);
        const uint32_t rounds = static_cast<uint32_t>(
            (size + kRingRoundElemMax - 1) / kRingRoundElemMax);
        num_rounds = std::max<uint32_t>(num_rounds, rounds);
    }
    TORCH_INTERNAL_ASSERT(num_rounds <= UINT16_MAX);

    const KernelId kernel_id = use_nopace
        ? KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE
        : KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED;
    RpuKernelGraph& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active()) {
        TORCH_CHECK(residual_ddr == nullptr,
                    "typed ring all-reduce census rejects a DDR residual");
        graph.stage_kernel_no_ddr(
            kernel_id,
            GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    TORCH_CHECK(
        kernel != nullptr,
        "Failed to get ",
        use_nopace
            ? "llm_all_reduce_residual_nopace"
            : "llm_all_reduce_residual",
        " kernel");
    kernel->reset_regs();

    uint32_t residual_base = 0;
    if (residual_ddr != nullptr) {
        const uint64_t residual_dev_addr =
            RpuGetDevAddr(const_cast<c10::Half*>(residual_ddr));
        TORCH_CHECK((residual_dev_addr & 0xFFU) == 0,
                    "ring all-reduce DDR residual must be 256-byte aligned");
        TORCH_CHECK((residual_dev_addr >> 8) <= UINT32_MAX,
                    "ring all-reduce DDR residual exceeds register encoding");
        residual_base = static_cast<uint32_t>(residual_dev_addr >> 8);
    } else {
        residual_base = static_cast<uint32_t>(
            static_cast<uint64_t>(residual_spm) - spm_base0);
    }
    kernel->set_regs(0, static_cast<uint16_t>(core_num));
    kernel->set_regs(1, static_cast<uint16_t>(num_rounds));
    kernel->set_regs(
        2, static_cast<uint16_t>(kRingRoundElemMax & 0xFFFFU));
    kernel->set_regs(
        3, static_cast<uint16_t>(kRingRoundElemMax >> 16));
    kernel->set_regs(4, static_cast<uint16_t>(residual_base & 0xFFFFU));
    kernel->set_regs(5, static_cast<uint16_t>(residual_base >> 16));
    kernel->set_regs(
        6, static_cast<uint16_t>(residual_ddr != nullptr));
    kernel->set_regs(64, static_cast<uint16_t>(kRingWarpNum));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    constexpr uint32_t kScmBase = 4096;
    const uint32_t input_local = static_cast<uint32_t>(
        static_cast<uint64_t>(input_spm) - spm_base0);
    const uint32_t output_local = static_cast<uint32_t>(
        static_cast<uint64_t>(output_spm) - spm_base0);
    for (int core = 0; core < core_num; ++core) {
        const uint32_t input_base =
            SPM_ALLOC.addr(core, input_local) -
            static_cast<uint32_t>(spm_base0);
        const uint32_t output_base =
            SPM_ALLOC.addr(core, output_local) -
            static_cast<uint32_t>(spm_base0);
        const uint32_t chunk_offset_bytes =
            chunk_offsets[core] * static_cast<uint32_t>(kFp16Bytes);
        const uint32_t chunk_size_elements = chunk_sizes[core];
        rpu_set_legacy_scm_u16_checked(
            kernel,
            kScmBase + core,
            static_cast<uint16_t>(input_base & 0xFFFFU), "ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel,
            kScmBase + 8 + core,
            static_cast<uint16_t>(input_base >> 16), "ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel,
            kScmBase + 16 + core,
            static_cast<uint16_t>(output_base & 0xFFFFU), "ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel,
            kScmBase + 24 + core,
            static_cast<uint16_t>(output_base >> 16), "ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel,
            kScmBase + 32 + core,
            static_cast<uint16_t>(chunk_offset_bytes & 0xFFFFU),
            "ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel,
            kScmBase + 40 + core,
            static_cast<uint16_t>(chunk_offset_bytes >> 16),
            "ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel,
            kScmBase + 48 + core,
            static_cast<uint16_t>(chunk_size_elements & 0xFFFFU),
            "ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel,
            kScmBase + 56 + core,
            static_cast<uint16_t>(chunk_size_elements >> 16),
            "ring all-reduce");
    }

    auto* queue = GET_QUEUE(core_num);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    cores.reserve(core_num);
    for (int core = 0; core < core_num; ++core) {
        cores.push_back(static_cast<uint8_t>(core));
    }
    queue->enqueu_kernel(
        *kernel,
        {static_cast<uint16_t>(kRingWarpNum), 1, 1},
        cores);
}

void rpu_launch_all_reduce_sum_residual_impl(
    uint32_t input_spm,
    uint32_t residual_spm,
    const c10::Half* residual_ddr,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores) {
    TORCH_CHECK(
        input_num_cores >= 1 &&
            input_num_cores <= output_num_cores &&
            output_num_cores == kRingCoreNum,
        "ring all-reduce requires 1 <= input cores <= 8 and exactly 8 "
        "output cores; "
        "got input_cores=", input_num_cores,
        " output_cores=", output_num_cores);
    TORCH_CHECK(
        ring_all_reduce_sum_residual_admitted(
            input_spm, residual_spm, residual_ddr, output_spm,
            M, N, output_num_cores),
        "ring all-reduce requires positive FP16 geometry, complete "
        "out-of-place SPM operands, and an initialized current SPM arena");

    // For partial producers, rpu_prepare_ring_all_reduce_input() must have
    // cleared all 8 input shards before the producer overwrote the active
    // leading cores. Subset-core fill queues are not hardware-safe.
    TORCH_CHECK(
        input_num_cores == output_num_cores ||
            (static_cast<uint64_t>(M) * static_cast<uint64_t>(N)) %
                    (SpmAllocator::ALIGN / sizeof(c10::Half)) ==
                0,
        "partial-producer ring input must use a 256-byte-aligned tensor");
    launch_ring_all_reduce_sum_residual(
        input_spm, residual_spm, residual_ddr, output_spm,
        M, N, output_num_cores);
}

}  // namespace

void rpu_prepare_ring_all_reduce_input(
    uint32_t input_spm,
    int64_t M,
    int64_t N,
    int input_num_cores) {
    TORCH_CHECK(
        input_num_cores >= 1 && input_num_cores <= kRingCoreNum,
        "ring input preparation requires 1..8 producer cores; got ",
        input_num_cores);
    TORCH_CHECK(
        M > 0 && N > 0 &&
            static_cast<uint64_t>(M) <=
                UINT32_MAX / static_cast<uint64_t>(N),
        "ring input preparation requires positive M*N within uint32");
    if (input_num_cores == kRingCoreNum) return;

    constexpr uint64_t kFillAlignmentElements =
        SpmAllocator::ALIGN / sizeof(c10::Half);
    static_assert(
        SpmAllocator::ALIGN % sizeof(c10::Half) == 0 &&
        kFillAlignmentElements == 128);
    const uint64_t elements =
        static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
    TORCH_CHECK(
        elements % kFillAlignmentElements == 0,
        "partial-producer ring input must be 256-byte aligned; got M*N=",
        elements);

    // This runs before the partial producer: clear every shard with the proven
    // leading 8-core launch, then let the producer overwrite cores [0, tp).
    // Never launch fill only on the inactive non-zero core suffix.
    rpu_launch_fill_spm_kernel(
        input_spm, static_cast<int64_t>(elements), c10::Half(0.0f),
        kRingCoreNum);
}

void rpu_launch_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores) {
    rpu_launch_all_reduce_sum_residual_impl(
        input_spm, residual_spm, nullptr, output_spm,
        M, N, input_num_cores, output_num_cores);
}

void rpu_launch_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    const v3::SpmFmbPostFnYieldTarget& output,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores) {
    const bool cpu_dry =
        v3::SpmFmbYieldTargetLauncherAccess::is_cpu_dry(output);
    const uint32_t output_spm =
        v3::SpmFmbYieldTargetLauncherAccess::
            resolve_and_stage_all_reduce(
                output, M, N, input_num_cores, output_num_cores);
    try {
        if (cpu_dry) {
            TORCH_CHECK(
                ring_all_reduce_sum_residual_admitted(
                    input_spm, residual_spm, nullptr, output_spm,
                    M, N, output_num_cores,
                    /*cpu_dry=*/true),
                "CPU-dry typed ring requires complete, out-of-place logical "
                "SPM offsets in the initialized current arena");
            rpu_launch_all_reduce_sum_residual_impl(
                SPM_ALLOC.addr(0, input_spm),
                SPM_ALLOC.addr(0, residual_spm), nullptr,
                SPM_ALLOC.addr(0, output_spm),
                M, N, input_num_cores, output_num_cores);
        } else {
            rpu_launch_all_reduce_sum_residual_impl(
                input_spm, residual_spm, nullptr, output_spm,
                M, N, input_num_cores, output_num_cores);
        }
    } catch (...) {
        v3::SpmFmbYieldTargetLauncherAccess::cancel_pending();
        throw;
    }
}

void rpu_launch_all_reduce_sum_residual_ddr_kernel(
    uint32_t input_spm,
    const c10::Half* residual_ddr,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores) {
    TORCH_CHECK(residual_ddr != nullptr,
                "all-reduce DDR residual pointer must not be null");
    rpu_launch_all_reduce_sum_residual_impl(
        input_spm, 0, residual_ddr, output_spm,
        M, N, input_num_cores, output_num_cores);
}
