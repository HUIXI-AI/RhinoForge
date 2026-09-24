// rpu_runtime_extras.cpp — Misc runtime helpers carried over from v5-08
//
// Runtime helpers with public bindings or fused-model callers are collected
// here. Their state and launch contracts remain shared across all handles.
//
//
//
//
//
//   - set_spm_debug / get_spm_debug                          (Python-bound)
//   - set_cross_layer_batch_prefill / get_cross_layer_batch_prefill  (Python-bound)
//   - rpu_launch_all_reduce_sum_residual_kernel              (called by 4 v3 files)
//
// Forward declarations stay in `src/core/rpu_kernel_decls.h`. v5.1+ may rehome
// the kernel-launch wrapper to `src/ops/` when the ops/ layout settles.

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
// File-local statics (v5-08 migrated byte-equal from the deleted v2 file)
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
    bool residual_local_shard = false,
    bool cpu_dry = false,
    bool allow_exact_m576_inplace = false) {
    if (core_num < 1 || core_num > kRingCoreNum ||
        (allow_exact_m576_inplace && core_num != kRingCoreNum) ||
        M <= 0 || N <= 0 || !SPM_ALLOC.is_initialized() ||
        (cpu_dry && residual_ddr != nullptr)) {
        return false;
    }
    const bool admitted_local_shape =
        (M == 576 && N == 1280) || (M == 320 && N == 2048) ||
        (M == 1200 && N == 1024);
    if (residual_local_shard &&
        (residual_ddr != nullptr || core_num != kRingCoreNum ||
         !admitted_local_shape)) {
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
    const uint64_t residual_bytes = residual_local_shard
        ? avg_elements * kFp16Bytes
        : tensor_bytes;

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
    const bool exact_m576_inplace =
        allow_exact_m576_inplace && M == 576 && N == 1280 &&
        residual_ddr == nullptr && input_spm == output_spm;
    if (allow_exact_m576_inplace != exact_m576_inplace ||
        (!exact_m576_inplace &&
         !ranges_are_disjoint(
             input_spm, tensor_bytes, output_spm, tensor_bytes))) {
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
    int core_num,
    bool residual_local_shard = false,
    bool allow_exact_m576_inplace = false,
    bool pi05_xor3 = false) {
    TORCH_CHECK(
        ring_all_reduce_sum_residual_admitted(
            input_spm, residual_spm, residual_ddr, output_spm,
            M, N, core_num, residual_local_shard,
            /*cpu_dry=*/false, allow_exact_m576_inplace),
        "ring all-reduce requires 1..8 cores, positive FP16 geometry, "
        "complete SPM operands, and either disjoint input/output or the "
        "validated exact M576N1280 in-place layout");

    TORCH_CHECK(!pi05_xor3 ||
        (residual_ddr == nullptr && !residual_local_shard && !allow_exact_m576_inplace &&
         v3::fmb_ring_all_reduce_route_selector(M, N, true, core_num) ==
             static_cast<int64_t>(v3::FmbSharedAllReduceRouteSelector::RING_PI05_XOR3) &&
         (input_spm & 255U) == 0 && (residual_spm & 255U) == 0 && (output_spm & 255U) == 0),
        "Pi XOR3 ring requires an exact shape and aligned disjoint SPM operands");

    constexpr uint64_t kFp16Bytes = sizeof(c10::Half);
    const uint64_t total_elements =
        static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
    const uint64_t spm_base0 = SPM_ALLOC.addr(0, 0);

    // ceil-align(ceil-div(total, cores), 16 elements).
    const uint64_t avg_elements_unaligned =
        (total_elements + static_cast<uint64_t>(core_num) - 1) /
        static_cast<uint64_t>(core_num);
    const uint64_t avg_elements =
        ((avg_elements_unaligned + 15U) / 16U) * 16U;
    TORCH_INTERNAL_ASSERT(avg_elements <= UINT32_MAX);
    const uint64_t chunk_bytes = avg_elements * kFp16Bytes;
    const bool use_nopace =
        !residual_local_shard && chunk_bytes < kRingNopaceChunkBytes;

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

    const KernelId kernel_id = pi05_xor3
        ? (M == 512 ? KernelId::PI05_RING_XOR3_M512N1152
           : M == 272 ? KernelId::PI05_RING_XOR3_M272N2048
           : M == 432 ? KernelId::PI05_RING_XOR3_M432N2048
           : M == 448 ? KernelId::PI05_RING_XOR3_M448N2048
           : M == 416 ? KernelId::PI05_RING_XOR3_M416N2048
           : M == 288 ? KernelId::PI05_RING_XOR3_M288N2048
           : M == 304 ? KernelId::PI05_RING_XOR3_M304N2048
           : M == 320 ? KernelId::PI05_RING_XOR3_M320N2048
           : M == 768 ? KernelId::PI05_RING_XOR3_M768N1152
           : M == 400 ? KernelId::PI05_RING_XOR3_M400N2048 : KernelId::PI05_RING_XOR3_M50N1024)
        : residual_local_shard
        ? KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM
        : (use_nopace
            ? KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE
            : KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED);
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
        residual_local_shard
            ? "llm_all_reduce_residual_local_spm"
            : (use_nopace
                ? "llm_all_reduce_residual_nopace"
                : "llm_all_reduce_residual"),
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
        6, static_cast<uint16_t>(
               residual_local_shard ? 2 : (residual_ddr != nullptr)));
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
            output_num_cores <= kRingCoreNum,
        "ring all-reduce requires 1 <= input cores <= output cores <= 8; "
        "got input_cores=", input_num_cores,
        " output_cores=", output_num_cores);
    TORCH_CHECK(
        ring_all_reduce_sum_residual_admitted(
            input_spm, residual_spm, residual_ddr, output_spm,
            M, N, output_num_cores),
        "ring all-reduce requires positive FP16 geometry, complete "
        "out-of-place SPM operands, and an initialized current SPM arena");

    // For partial producers, rpu_prepare_ring_all_reduce_input() must have
    // cleared every consumer input shard before the producer overwrote its
    // leading cores. Never fill just a non-zero inactive core suffix.
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
    int input_num_cores,
    int output_num_cores) {
    TORCH_CHECK(
        input_num_cores >= 1 && input_num_cores <= output_num_cores &&
            output_num_cores <= kRingCoreNum,
        "ring input preparation requires 1 <= input cores <= output cores <= 8; got ",
        input_num_cores, " -> ", output_num_cores);
    TORCH_CHECK(
        M > 0 && N > 0 &&
            static_cast<uint64_t>(M) <=
                UINT32_MAX / static_cast<uint64_t>(N),
        "ring input preparation requires positive M*N within uint32");
    if (input_num_cores == output_num_cores) return;

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

    // This runs before the partial producer: clear every consumer shard with
    // a leading-core launch, then let the producer overwrite cores [0, tp).
    // Never launch fill only on the inactive non-zero core suffix.
    rpu_launch_fill_spm_kernel(
        input_spm, static_cast<int64_t>(elements), c10::Half(0.0f),
        output_num_cores);
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
    uint32_t input_spm, uint32_t residual_spm, uint32_t output_spm,
    int64_t M, int64_t N, int input_num_cores, int output_num_cores,
    bool pi05_xor3) {
    if (!pi05_xor3) {
        rpu_launch_all_reduce_sum_residual_kernel(
            input_spm, residual_spm, output_spm, M, N, input_num_cores, output_num_cores);
        return;
    }
    TORCH_CHECK(input_num_cores == 8 && output_num_cores == 8,
                "Pi XOR3 ring requires eight input and output cores");
    launch_ring_all_reduce_sum_residual(input_spm, residual_spm, nullptr, output_spm,
        M, N, 8, /*local=*/false, /*inplace=*/false, /*pi05_xor3=*/true);
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
                    /*residual_local_shard=*/false, /*cpu_dry=*/true),
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

void rpu_launch_all_reduce_sum_residual_local_spm_kernel(
    uint32_t input_spm,
    uint32_t residual_local_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores)
{
    TORCH_CHECK(
        (M == 1200 && N == 1024) &&
            input_num_cores == kRingCoreNum &&
            output_num_cores == kRingCoreNum,
        "compact-residual ring requires Qwen3-VL Vision M1200N1024 on 8 cores; got M=", M, " N=", N,
        " input_cores=", input_num_cores,
        " output_cores=", output_num_cores);
    launch_ring_all_reduce_sum_residual(
        input_spm, residual_local_spm, nullptr, output_spm,
        M, N, input_num_cores,
        /*residual_local_shard=*/true);
}
