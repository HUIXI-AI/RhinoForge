// Runtime helpers shared by public Python bindings and fused models:
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
#include <cmath>
#include <cstdint>
#include <utility>
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
//
// ABI source of truth: rhino-ops
// generator/LlmAllReduceSumResidualGenerator.py. Every standard all-reduce
// uses one fused ring kernel. The generator's strict per-core crossover selects
// nopace below 11 KiB and paced otherwise.

namespace {

constexpr int kRingCoreNum = 8;
constexpr uint32_t kRingWarpNum = 8;
constexpr uint32_t kRingRoundElemMax = 31U * 256U * kRingWarpNum;
constexpr uint64_t kRingNopaceChunkBytes = 11U * 1024U;

struct RingAdaRmsOperands {
    uint32_t norm_output_spm;
    uint32_t scale_spm;
    uint32_t shift_spm;
    double eps;
};

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
    bool allow_exact_m576_inplace = false,
    bool allow_exact_qwen_residual_alias = false,
    bool allow_exact_qwen_local_shard = false) {
    if (core_num < 1 || core_num > kRingCoreNum ||
        (allow_exact_m576_inplace && core_num != kRingCoreNum) ||
        M <= 0 || N <= 0 || !SPM_ALLOC.is_initialized() ||
        (cpu_dry && residual_ddr != nullptr)) {
        return false;
    }
    const bool exact_qwen_local_shard = allow_exact_qwen_local_shard &&
        residual_local_shard && !cpu_dry && residual_ddr == nullptr &&
        !allow_exact_m576_inplace && !allow_exact_qwen_residual_alias &&
        M == 128 && N == 5120 && core_num == kRingCoreNum &&
        (input_spm & 255U) == 0 && (residual_spm & 255U) == 0 &&
        (output_spm & 255U) == 0;
    if (allow_exact_qwen_local_shard != exact_qwen_local_shard) return false;
    const bool admitted_local_shape =
        (M == 576 && N == 1280) || (M == 320 && N == 2048) ||
        (M == 1200 && N == 1024) || exact_qwen_local_shard;
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
    // The C192/C256 owners also use this path for a final C128 chunk. Their
    // original paced ring consumes each residual shard before publishing the
    // same shard, as verified for these three complete FP16 geometries only.
    const bool exact_qwen_residual_alias =
        allow_exact_qwen_residual_alias && !allow_exact_m576_inplace &&
        !residual_local_shard && !cpu_dry && residual_ddr == nullptr &&
        (M == 256 || M == 192 || M == 128) && N == 5120 && core_num == kRingCoreNum &&
        residual_spm == output_spm &&
        (input_spm & 255U) == 0 && (residual_spm & 255U) == 0 &&
        uint64_t(input_spm) + tensor_bytes <= (UINT64_C(1) << 32) &&
        uint64_t(residual_spm) + tensor_bytes <= (UINT64_C(1) << 32);
    if (allow_exact_m576_inplace != exact_m576_inplace ||
        allow_exact_qwen_residual_alias != exact_qwen_residual_alias ||
        (!exact_m576_inplace &&
         !ranges_are_disjoint(
             input_spm, tensor_bytes, output_spm, tensor_bytes))) {
        return false;
    }
    return residual_ddr != nullptr ||
        (ranges_are_disjoint(
             input_spm, tensor_bytes, residual_spm, residual_bytes) &&
         (exact_qwen_residual_alias || ranges_are_disjoint(
             residual_spm, residual_bytes, output_spm, tensor_bytes)));
}

struct G05RingRtcEpilogue {
    uint32_t clean_gate;
    uint32_t noisy_gate;
    int64_t clean_rows;
};

struct G05ActionTailNorm {
    uint32_t clean_scale, clean_shift, noisy_scale, noisy_shift;
    double eps;
};

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
    bool pi05_xor3 = false,
    const G05RingRtcEpilogue* g05_rtc = nullptr,
    const G05ActionTailNorm* g05_norm = nullptr,
    const RingAdaRmsOperands* attn_adarms = nullptr,
    bool allow_exact_qwen_residual_alias = false,
    bool allow_exact_qwen_local_shard = false) {
    TORCH_CHECK(!(allow_exact_qwen_residual_alias || allow_exact_qwen_local_shard) ||
        (!pi05_xor3 && g05_rtc == nullptr && g05_norm == nullptr && attn_adarms == nullptr),
        "Qwen bounded residual alias requires the original ordinary paced ring");
    TORCH_CHECK(
        ring_all_reduce_sum_residual_admitted(
            input_spm, residual_spm, residual_ddr, output_spm,
            M, N, core_num, residual_local_shard,
            /*cpu_dry=*/false, allow_exact_m576_inplace, allow_exact_qwen_residual_alias,
            allow_exact_qwen_local_shard),
        "ring all-reduce requires 1..8 cores, positive FP16 geometry, "
        "complete SPM operands, and either disjoint input/output or the "
        "validated exact M576N1280 or Qwen C256/C192/C128 residual in-place layout");

    TORCH_CHECK(!pi05_xor3 ||
        (residual_ddr == nullptr && !residual_local_shard && !allow_exact_m576_inplace &&
         v3::fmb_ring_all_reduce_route_selector(M, N, true, core_num) ==
             static_cast<int64_t>(v3::FmbSharedAllReduceRouteSelector::RING_PI05_XOR3) &&
         (input_spm & 255U) == 0 && (residual_spm & 255U) == 0 && (output_spm & 255U) == 0),
        "Pi XOR3 ring requires an exact shape and aligned disjoint SPM operands");

    TORCH_CHECK(attn_adarms == nullptr ||
        (M == 51 && N == 768 && core_num == 8 && residual_ddr == nullptr &&
         !residual_local_shard && !allow_exact_m576_inplace && !pi05_xor3 &&
         attn_adarms->norm_output_spm == residual_spm && attn_adarms->eps == 1.0e-6),
        "LingBot2 ring/AdaRMS requires the exact ordinary T51/H768/TP8 residual alias");

    constexpr uint64_t kFp16Bytes = sizeof(c10::Half);
    const uint64_t total_elements =
        static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
    const uint64_t spm_base0 = SPM_ALLOC.addr(0, 0);

    // Generator contract: ceil-align(ceil-div(total, cores), 16 elements).
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

    TORCH_CHECK(g05_rtc == nullptr ||
        (M == 32 && N == 1024 && core_num == 8 && use_nopace && num_rounds == 1 &&
         !pi05_xor3 && attn_adarms == nullptr && !residual_local_shard && !allow_exact_m576_inplace && residual_ddr == nullptr),
        "G05 RTC epilogue requires the exact original M32/H1024 TP8 nopace ring");
    TORCH_CHECK(g05_norm == nullptr || g05_rtc != nullptr,
                "G05 Action tail norm requires its original RTC ring");
    const KernelId kernel_id = g05_norm != nullptr
        ? KernelId::G05_RING_RTC_THEN_NORM_SHIFT_M32H1024
        : g05_rtc != nullptr
        ? KernelId::G05_RING_RTC_SUB_GATED_RESIDUAL_M32H1024
        : attn_adarms != nullptr
        ? KernelId::LINGBOT2_ATTN_ALL_REDUCE_ADARMS_T51_H768
        : pi05_xor3
        ? (M == 512 ? KernelId::PI05_RING_XOR3_M512N1152
           : M == 272 ? KernelId::PI05_RING_XOR3_M272N2048
           : M == 432 ? KernelId::PI05_RING_XOR3_M432N2048
           : M == 464 ? KernelId::PI05_RING_XOR3_M464N2048
           : M == 480 ? KernelId::PI05_RING_XOR3_M480N2048
           : M == 448 ? KernelId::PI05_RING_XOR3_M448N2048
           : M == 416 ? KernelId::PI05_RING_XOR3_M416N2048
           : M == 288 ? KernelId::PI05_RING_XOR3_M288N2048
           : M == 304 ? KernelId::PI05_RING_XOR3_M304N2048
           : M == 336 ? KernelId::PI05_RING_XOR3_M336N2048
           : M == 320 ? KernelId::PI05_RING_XOR3_M320N2048
           : M == 768 ? KernelId::PI05_RING_XOR3_M768N1152
           : M == 400 ? KernelId::PI05_RING_XOR3_M400N2048
           : M == 10 ? KernelId::PI05_RING_XOR3_M10N1024 : KernelId::PI05_RING_XOR3_M50N1024)
        : residual_local_shard
        ? KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM
        : (use_nopace
            ? KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE
            : KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED);
    // Resolve the optional cached Program before staging a census occurrence.
    // Missing payloads must not leave a partially staged new kernel.
    TORCH_CHECK(
        KernelCache::instance().get(kernel_id) != nullptr,
        "Failed to get ",
        attn_adarms != nullptr
            ? "lingbot2_ring_adarms_norm_shift_t51_h768"
            : residual_local_shard
            ? "llm_all_reduce_residual_local_spm"
            : (use_nopace
                ? "llm_all_reduce_residual_nopace"
                : "llm_all_reduce_residual"),
        " kernel");
    auto* queue = attn_adarms == nullptr ? nullptr : GET_QUEUE(core_num);
    TORCH_CHECK(attn_adarms == nullptr || queue != nullptr,
                "LingBot2 ring/AdaRMS requires an available eight-core queue");
    RpuKernelGraph& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active()) {
        TORCH_CHECK(residual_ddr == nullptr,
                    "typed ring all-reduce census rejects a DDR residual");
        graph.stage_kernel_no_ddr(
            kernel_id,
            GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    // GET_KERNEL is graph-aware: it consumes the staged typed occurrence.
    // The availability check above must not perform this register lookup.
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    if (attn_adarms != nullptr) {
        TORCH_CHECK(kernel->reset_regs() == kKernelRegOk,
                    "LingBot2 ring/AdaRMS rejected register reset");
    } else {
        kernel->reset_regs();
    }

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
    if (g05_rtc != nullptr) {
        for (const auto& entry : {
                 std::pair<uint32_t,uint32_t>{18, g05_rtc->clean_gate},
                 {20, residual_spm}, {22, g05_rtc->noisy_gate}}) {
            kernel->set_regs(entry.first, static_cast<uint16_t>(entry.second));
            kernel->set_regs(entry.first + 1, static_cast<uint16_t>(entry.second >> 16));
        }
        kernel->set_regs(24, static_cast<uint16_t>(g05_rtc->clean_rows));
    }
    if (g05_norm != nullptr) {
        const auto set = [&](uint32_t reg, uint16_t value) {
            TORCH_CHECK(kernel->set_regs(reg, value) == kKernelRegOk,
                        "G05 Action tail register rejected: ", reg);
        };
        const auto pair = [&](uint32_t reg, uint32_t value) {
            set(reg, uint16_t(value)); set(reg + 1, uint16_t(value >> 16));
        };
        set(32, 1024); set(33, c10::Half(1.0f / 1024).x); pair(34, 2048);
        pair(36, output_spm); pair(38, residual_spm);
        pair(40, g05_norm->clean_scale); set(42, c10::Half(g05_norm->eps).x);
        pair(44, g05_norm->clean_shift); pair(46, g05_norm->noisy_scale);
        pair(48, g05_norm->noisy_shift); set(50, uint16_t(g05_rtc->clean_rows));
        // Original ring atomics and the new local completion counter0 are
        // explicit zero-valued packet inputs, including every replay.
        for (uint32_t reg = 128; reg < 256; ++reg) set(reg, 0);
    }
    kernel->set_regs(64, static_cast<uint16_t>(kRingWarpNum));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    if (attn_adarms != nullptr) {
        const auto set = [kernel](uint32_t index, uint16_t value) {
            TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                        "LingBot2 ring/AdaRMS rejected register ", index);
        };
        const auto pair = [&set](uint32_t index, uint32_t value) {
            set(index, static_cast<uint16_t>(value));
            set(index + 1, static_cast<uint16_t>(value >> 16));
        };
        // Original norm GP0..13 shifted intact to GP34..47. Its input and
        // output remain absolute SPM addresses; the ring GP4 remains local.
        set(34, 768);
        set(35, c10::Half(1.0f / 768).x);
        set(36, 768 * sizeof(c10::Half));
        set(37, 0);
        pair(38, output_spm);
        pair(40, attn_adarms->norm_output_spm);
        pair(42, attn_adarms->scale_spm);
        set(44, c10::Half(attn_adarms->eps).x);
        set(45, 0);
        pair(46, attn_adarms->shift_spm);
        // Submit all64 original/new local/global atomic values, rather than
        // relying on allocator freshness. This includes new core counter0 and
        // original ring local24..31/global24 on every BUILD and REPLAY packet.
        for (uint32_t index = 128; index <= 254; index += 2) {
            TORCH_CHECK(kernel->set_regs(index, uint32_t{0}) == kKernelRegOk,
                        "LingBot2 ring/AdaRMS rejected atomic register ", index);
        }
    }

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

    if (queue == nullptr) queue = GET_QUEUE(core_num);
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

void rpu_launch_g05_action_tail_spm_kernel(
    uint32_t input, uint32_t residual, uint32_t raw,
    uint32_t clean_gate, uint32_t noisy_gate,
    uint32_t clean_scale, uint32_t clean_shift,
    uint32_t noisy_scale, uint32_t noisy_shift, int64_t clean_rows, double eps) {
    auto& graph = RpuKernelGraph::active();
    TORCH_CHECK((graph.state() == RpuKernelGraph::State::RECORDING ||
                 graph.state() == RpuKernelGraph::State::REPLAYING) &&
                    SPM_ALLOC.is_initialized() && clean_rows >= 0 && clean_rows <= 32 &&
                    std::isfinite(eps) && eps == 1e-6,
                "G05 Action tail requires retained M32/H1024 TP8 RTC and original epsilon");
    constexpr uint64_t matrix_bytes = 32 * 1024 * sizeof(c10::Half);
    constexpr uint64_t row_bytes = 1024 * sizeof(c10::Half);
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const auto valid = [base](uint32_t address, uint64_t bytes) {
        return address % 256 == 0 && address >= base &&
            uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE;
    };
    const auto disjoint = [](uint32_t a, uint64_t an, uint32_t b, uint64_t bn) {
        return uint64_t(a) + an <= b || uint64_t(b) + bn <= a;
    };
    const std::array<uint32_t,3> matrices{input,residual,raw};
    for (size_t i = 0; i < matrices.size(); ++i) {
        TORCH_CHECK(valid(matrices[i],matrix_bytes), "G05 Action tail incomplete matrix");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(disjoint(matrices[i],matrix_bytes,matrices[j],matrix_bytes),
                        "G05 Action tail requires independent partial, skip and raw matrices");
    }
    const std::array<uint32_t,6> parameters{
        clean_gate,noisy_gate,clean_scale,clean_shift,noisy_scale,noisy_shift};
    for (size_t i = 0; i < parameters.size(); ++i) {
        TORCH_CHECK(valid(parameters[i],row_bytes), "G05 Action tail incomplete parameter row");
        for (auto matrix : matrices)
            TORCH_CHECK(disjoint(parameters[i],row_bytes,matrix,matrix_bytes),
                        "G05 Action tail parameter overlaps a matrix");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(parameters[i] == parameters[j] ||
                        disjoint(parameters[i],row_bytes,parameters[j],row_bytes),
                        "G05 Action tail partial parameter alias");
    }
    TORCH_CHECK(KernelCache::instance().has_loaded(
                    KernelId::G05_RING_RTC_THEN_NORM_SHIFT_M32H1024),
                "G05 Action tail optional payload unavailable");
    // The original ring's global retirement barrier makes the old skip dead.
    // New norm writes that exact full skip root; raw remains fully retained.
    const G05RingRtcEpilogue ring{clean_gate,noisy_gate,clean_rows};
    const G05ActionTailNorm norm{clean_scale,clean_shift,noisy_scale,noisy_shift,eps};
    launch_ring_all_reduce_sum_residual(input,residual,nullptr,raw,32,1024,8,
                                       false,false,false,&ring,&norm);
}

void rpu_launch_g05_ring_rtc_residual_spm_kernel(
    uint32_t input_spm, uint32_t residual_spm, uint32_t clean_gate_spm,
    uint32_t noisy_gate_spm, uint32_t output_spm, int64_t clean_rows, int cores) {
    TORCH_CHECK(clean_rows >= 0 && clean_rows <= 32 && cores == 8 &&
                    SPM_ALLOC.is_initialized(),
                "G05 ring RTC residual requires M32/H1024, cores8, clean_rows0..32 and live SPM");
    constexpr uint64_t matrix_bytes = 32 * 1024 * sizeof(c10::Half);
    constexpr uint64_t gate_bytes = 1024 * sizeof(c10::Half);
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    auto checked = [base](uint32_t address, uint64_t bytes) {
        return address % 256 == 0 && address >= base &&
            uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE;
    };
    auto disjoint = [](uint32_t a, uint64_t an, uint32_t b, uint64_t bn) {
        return uint64_t(a) + an <= b || uint64_t(b) + bn <= a;
    };
    TORCH_CHECK(checked(input_spm, matrix_bytes) && checked(residual_spm, matrix_bytes) &&
                    checked(output_spm, matrix_bytes) && checked(clean_gate_spm, gate_bytes) &&
                    checked(noisy_gate_spm, gate_bytes) &&
                    ring_all_reduce_sum_residual_admitted(
                        input_spm, residual_spm, nullptr, output_spm, 32, 1024, 8),
                "G05 ring RTC residual requires full aligned, disjoint input/residual/output SPM");
    for (auto gate : {clean_gate_spm, noisy_gate_spm}) {
        TORCH_CHECK(disjoint(gate, gate_bytes, input_spm, matrix_bytes) &&
                        disjoint(gate, gate_bytes, residual_spm, matrix_bytes) &&
                        disjoint(gate, gate_bytes, output_spm, matrix_bytes),
                    "G05 ring RTC residual gate overlaps a matrix owner");
    }
    TORCH_CHECK(clean_gate_spm == noisy_gate_spm ||
                    disjoint(clean_gate_spm, gate_bytes, noisy_gate_spm, gate_bytes),
                "G05 ring RTC residual rejects partial gate alias");
    TORCH_CHECK(KernelCache::instance().has_loaded(
                    KernelId::G05_RING_RTC_SUB_GATED_RESIDUAL_M32H1024),
                "G05 ring RTC residual optional payload is unavailable");
    // Gate/residual replication is a model-owned COMPLETE invariant, not a
    // new generic ring option. All shared generator parameters, SCM and census
    // staging remain centralized in the original ring launcher.
    const G05RingRtcEpilogue epilogue{clean_gate_spm, noisy_gate_spm, clean_rows};
    launch_ring_all_reduce_sum_residual(
        input_spm, residual_spm, nullptr, output_spm, 32, 1024, 8,
        false, false, false, &epilogue);
}

void rpu_launch_qwen3_c512_all_reduce_residual_local_spm_kernel(
    uint32_t partial_spm, uint32_t compact_residual_spm, uint32_t output_spm)
{
    // One M128 segment, with only the owning M16 row stripe on each core.
    // All three operands remain disjoint; no generic alias rule is relaxed.
    launch_ring_all_reduce_sum_residual(
        partial_spm, compact_residual_spm, nullptr, output_spm,
        /*M=*/128, /*N=*/5120, /*core_num=*/8,
        /*residual_local_shard=*/true, /*allow_exact_m576_inplace=*/false,
        /*pi05_xor3=*/false, /*g05_rtc=*/nullptr, /*g05_norm=*/nullptr,
        /*attn_adarms=*/nullptr,
        /*allow_exact_qwen_residual_alias=*/false,
        /*allow_exact_qwen_local_shard=*/true);
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

void rpu_launch_lingbot2_attn_all_reduce_adarms_norm_shift_spm_kernel(
    uint32_t partial_spm, uint32_t residual_spm, uint32_t raw_output_spm,
    uint32_t norm_output_spm, uint32_t scale_spm, uint32_t shift_spm,
    int64_t rows, int64_t hidden, double eps, int cores) {
    TORCH_CHECK(rows == 51 && hidden == 768 && cores == 8 && eps == 1.0e-6,
                "LingBot2 ring/AdaRMS requires T51/H768/TP8 and eps=1e-6");
    TORCH_CHECK(norm_output_spm == residual_spm && SPM_ALLOC.is_initialized(),
                "LingBot2 ring/AdaRMS requires live SPM and exact residual/norm output alias");
    constexpr uint64_t row_bytes = 768 * sizeof(c10::Half);
    constexpr uint64_t matrix_bytes = 51 * row_bytes;
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 5> ranges{{
        {partial_spm, matrix_bytes}, {residual_spm, matrix_bytes},
        {raw_output_spm, matrix_bytes}, {scale_spm, row_bytes},
        {shift_spm, row_bytes},
    }};
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [address, bytes] = ranges[i];
        TORCH_CHECK(
            (address & 255U) == 0 && address >= base &&
                bytes <= SpmAllocator::SPM_USABLE &&
                uint64_t(address) - base <= uint64_t(SpmAllocator::SPM_USABLE) - bytes &&
                uint64_t(address) + bytes <= (UINT64_C(1) << 32),
            "LingBot2 ring/AdaRMS requires complete aligned absolute core-0 SPM operands");
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(
                uint64_t(address) + bytes <= ranges[j].first ||
                    uint64_t(ranges[j].first) + ranges[j].second <= address,
                "LingBot2 ring/AdaRMS requires five disjoint SPM owners");
        }
    }
    const RingAdaRmsOperands norm{norm_output_spm, scale_spm, shift_spm, eps};
    launch_ring_all_reduce_sum_residual(
        partial_spm, residual_spm, nullptr, raw_output_spm,
        rows, hidden, cores, /*local=*/false, /*inplace=*/false,
        /*pi05_xor3=*/false, /*g05_rtc=*/nullptr, /*g05_norm=*/nullptr, &norm);
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

void rpu_launch_gr00t_all_reduce_residual_xor3_kernel(
    uint32_t input_spm, uint32_t residual_spm, uint32_t output_spm) {
    TORCH_CHECK((input_spm & 255U) == 0 && (residual_spm & 255U) == 0 &&
                    (output_spm & 255U) == 0 &&
                    ring_all_reduce_sum_residual_admitted(
                        input_spm, residual_spm, nullptr, output_spm, 41, 1536, 8),
                "GR00T XOR3 requires initialized SPM and three aligned, "
                "complete, disjoint FP16 [41,1536] SPM operands");
    constexpr KernelId id = KernelId::GR00T_ALL_REDUCE_RESIDUAL_XOR3_M41N1536;
    auto& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active()) {
        graph.stage_kernel_no_ddr(id, GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    Kernel_t* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel != nullptr, "GR00T XOR3 expansion kernel is absent");
    kernel->reset_regs();
    const auto set = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                    "GR00T XOR3 rejected register ", index);
    };
    const auto pair = [&](uint32_t index, uint32_t value) {
        set(index, static_cast<uint16_t>(value));
        set(index + 1, static_cast<uint16_t>(value >> 16));
    };
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    // Keep the original paced reduce-scatter/residual parameters and full
    // replicated output. Only the kernel's all-gather schedule uses XOR3.
    set(0, 8);
    set(1, 1);
    pair(2, 31 * 256 * 8);
    pair(4, residual_spm - base);
    set(6, 0);
    set(64, 8);
    set(65, 1);
    set(66, 1);
    // All counter halfwords must be present in every BUILD/REPLAY packet.
    // The dedicated payload uses nine prefix epochs and three XOR3 epochs.
    for (uint32_t index = 128; index <= 255; ++index) set(index, 0);
    constexpr uint32_t kChunkElements = 7872;
    constexpr uint32_t kChunkBytes = kChunkElements * sizeof(c10::Half);
    for (uint32_t core = 0; core < 8; ++core) {
        // The shard stride is 15744 bytes (128-byte aligned); keep it exact.
        const std::array<std::pair<uint32_t, uint32_t>, 4> fields{{
            {0, SPM_ALLOC.addr(core, input_spm - base) - base},
            {16, SPM_ALLOC.addr(core, output_spm - base) - base},
            {32, core * kChunkBytes}, {48, kChunkElements}}};
        for (const auto& field : fields) {
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + field.first + core,
                static_cast<uint16_t>(field.second), "GR00T XOR3");
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + field.first + 8 + core,
                static_cast<uint16_t>(field.second >> 16), "GR00T XOR3");
        }
    }
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_wall_vision_all_reduce_sum_residual_m576_inplace_kernel(
    uint32_t partial_output_spm,
    uint32_t residual_spm)
{
    launch_ring_all_reduce_sum_residual(
        partial_output_spm, residual_spm, nullptr, partial_output_spm,
        /*M=*/576, /*N=*/1280, /*core_num=*/8,
        /*residual_local_shard=*/false,
        /*allow_exact_m576_inplace=*/true);
}

void rpu_launch_qwen3_c192_all_reduce_residual_inplace_kernel(
    uint32_t partial_spm, uint32_t residual_output_spm, int64_t rows)
{
    launch_ring_all_reduce_sum_residual(
        partial_spm, residual_output_spm, nullptr, residual_output_spm,
        rows, /*N=*/5120, /*core_num=*/8,
        /*residual_local_shard=*/false, /*allow_exact_m576_inplace=*/false,
        /*pi05_xor3=*/false, /*g05_rtc=*/nullptr, /*g05_norm=*/nullptr,
        /*attn_adarms=*/nullptr,
        /*allow_exact_qwen_residual_alias=*/true);
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
        ((M == 576 && N == 1280) || (M == 320 && N == 2048) ||
         (M == 1200 && N == 1024)) &&
            input_num_cores == kRingCoreNum &&
            output_num_cores == kRingCoreNum,
        "compact-residual ring is validated only for Wall Vision M576N1280 "
        "Wall Text M320N2048, or Qwen3-VL Vision M1200N1024 on 8 cores; got M=", M, " N=", N,
        " input_cores=", input_num_cores,
        " output_cores=", output_num_cores);
    launch_ring_all_reduce_sum_residual(
        input_spm, residual_local_spm, nullptr, output_spm,
        M, N, input_num_cores,
        /*residual_local_shard=*/true);
}

namespace {

struct WallRingPostNormParams {
    uint32_t raw_residual_local_spm;
    uint32_t gamma_spm;
    double eps;
};

void rpu_launch_wall_oss_mixed_ring_allreduce(
    uint32_t input_spm,
    uint32_t residual_spm,
    const c10::Half* residual_ddr,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    bool partial_is_bf16,
    bool residual_local_shard,
    const WallRingPostNormParams* postnorm = nullptr,
    bool allow_exact_m576_inplace = false,
    KernelId exact_m576_postnorm_id = KernelId::_COUNT)
{
    const bool fused_postnorm = postnorm != nullptr;
    const bool exact_m576_full_postnorm_kernel =
        exact_m576_postnorm_id ==
            KernelId::WALL_DOWN_POSTNORM_M576_FP16_PARTIAL_FP16_RESIDUAL ||
        exact_m576_postnorm_id ==
            KernelId::WALL_DOWN_POSTNORM_M576_BF16_PARTIAL_FP16_RESIDUAL ||
        exact_m576_postnorm_id ==
            KernelId::WALL_DOWN_POSTNORM_M576_FP16_PARTIAL_BF16_RESIDUAL;
    const bool exact_m576_local_postnorm_kernel =
        exact_m576_postnorm_id ==
            KernelId::WALL_RING_POSTNORM_M576_FP16_PARTIAL_FP16_RESIDUAL_LOCAL_INOUT ||
        exact_m576_postnorm_id ==
            KernelId::WALL_RING_POSTNORM_M576_BF16_PARTIAL_FP16_RESIDUAL_LOCAL_INOUT ||
        exact_m576_postnorm_id ==
            KernelId::WALL_RING_POSTNORM_M576_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT;
    const bool exact_m576_postnorm =
        fused_postnorm &&
        (exact_m576_full_postnorm_kernel ||
         exact_m576_local_postnorm_kernel) &&
        M == 576 && N == 1280 && residual_ddr == nullptr &&
        residual_local_shard == exact_m576_local_postnorm_kernel &&
        input_spm == output_spm;
    const bool exact_m320_local_postnorm_inplace =
        exact_m576_postnorm_id == KernelId::_COUNT && fused_postnorm &&
        M == 320 && N == 2048 && !partial_is_bf16 &&
        residual_ddr == nullptr && residual_local_shard &&
        input_spm == output_spm &&
        postnorm->raw_residual_local_spm == residual_spm;
    TORCH_CHECK(
        exact_m576_postnorm_id == KernelId::_COUNT || exact_m576_postnorm,
        "Wall Vision down postnorm requires an exact M576N1280 kernel, "
        "the matching full/compact SPM residual mode, and normalized output "
        "aliased to partial");
    const bool exact_m576_inplace =
        allow_exact_m576_inplace && !fused_postnorm && M == 576 &&
        N == 1280 && residual_ddr == nullptr && !residual_local_shard &&
        input_spm == output_spm;
    TORCH_CHECK(
        allow_exact_m576_inplace == exact_m576_inplace,
        "Wall Vision in-place mixed ring requires exact M576N1280, one "
        "aliased partial/output SPM range, and a separate full residual");
    TORCH_CHECK(
        ((M == 288 || M == 576) && N == 1280) || (M == 320 && N == 2048)
            || (M == 336 && N == 2048),
        "Wall-OSS mixed ring all-reduce supports only Vision (288/576,1280) or "
        "RTC prefill (320/336,2048); got M=",
        M, " N=", N);
    TORCH_CHECK(
        !residual_local_shard ||
            (residual_ddr == nullptr &&
             ((!partial_is_bf16 &&
               ((M == 576 && N == 1280) ||
                (M == 320 && N == 2048))) ||
              (partial_is_bf16 && exact_m576_local_postnorm_kernel))),
        "Wall-OSS mixed compact-residual ring is validated only for the "
        "exact M576N1280 Vision or FP16-partial/BF16-residual M320N2048 "
        "Text projection");
    TORCH_CHECK(
        !fused_postnorm ||
            exact_m576_postnorm ||
            (M == 320 && N == 2048 && !partial_is_bf16 &&
             residual_ddr == nullptr),
        "Wall-OSS fused ring postnorm is validated only for the exact "
        "M320N2048 Text or M576N1280 Vision SPM profiles");
    if (fused_postnorm) {
        if (exact_m576_local_postnorm_kernel) {
            TORCH_CHECK(
                input_spm == output_spm &&
                    residual_spm == postnorm->raw_residual_local_spm &&
                    input_spm != residual_spm &&
                    postnorm->gamma_spm != input_spm &&
                    postnorm->gamma_spm != residual_spm,
                "Wall Vision local-inout postnorm requires aliased "
                "partial/normalized output, an in-place compact residual, "
                "and a distinct gamma");
        } else if (exact_m576_postnorm) {
            TORCH_CHECK(
                input_spm == output_spm && input_spm != residual_spm &&
                    input_spm != postnorm->raw_residual_local_spm &&
                    residual_spm != postnorm->raw_residual_local_spm &&
                    postnorm->gamma_spm != input_spm &&
                    postnorm->gamma_spm != residual_spm &&
                    postnorm->gamma_spm !=
                        postnorm->raw_residual_local_spm,
                "Wall Vision down postnorm requires aliased partial/normalized "
                "output and distinct full residual, compact raw, and gamma");
        } else if (residual_local_shard) {
            TORCH_CHECK(
                input_spm != residual_spm &&
                    (input_spm != output_spm ||
                     exact_m320_local_postnorm_inplace) &&
                    residual_spm != output_spm &&
                    postnorm->raw_residual_local_spm == residual_spm,
                "Wall-OSS local fused ring postnorm requires a disjoint compact "
                "residual and either distinct partial/output slabs or the exact "
                "Wall Text carrier alias");
        } else {
            TORCH_CHECK(
                input_spm != residual_spm && residual_spm == output_spm,
                "Wall-OSS fused ring postnorm requires the BF16 residual and "
                "FP16 normalized output to use the same full SPM slab");
        }
    } else if (exact_m576_inplace) {
        TORCH_CHECK(
            ring_all_reduce_sum_residual_admitted(
                input_spm, residual_spm, nullptr, output_spm,
                M, N, /*core_num=*/8,
                /*residual_local_shard=*/false,
                /*cpu_dry=*/false,
                /*allow_exact_m576_inplace=*/true),
            "Wall Vision in-place mixed ring requires complete aliased "
            "partial/output and disjoint residual SPM ranges");
    } else {
        TORCH_CHECK(
            input_spm != output_spm
                && (residual_ddr != nullptr
                    || (input_spm != residual_spm
                        && residual_spm != output_spm)),
            "Wall-OSS mixed ring all-reduce requires out-of-place operands");
    }
    if (residual_local_shard) {
        TORCH_CHECK(
            exact_m320_local_postnorm_inplace ||
                ring_all_reduce_sum_residual_admitted(
                    input_spm, residual_spm, nullptr, output_spm,
                    M, N, /*core_num=*/8,
                    /*residual_local_shard=*/true,
                    /*cpu_dry=*/false,
                    /*allow_exact_m576_inplace=*/
                        exact_m576_local_postnorm_kernel),
            "Wall-OSS mixed compact-residual ring requires complete "
            "input/output and one in-range local residual shard");
    }

    constexpr uint32_t kCoreNum = 8;
    constexpr uint32_t kWarpNum = 8;
    constexpr uint32_t kDwidth = 2;
    constexpr uint32_t kRoundElemMax = 31 * 256 * kWarpNum;
    const uint32_t total = static_cast<uint32_t>(M * N);
    const uint32_t avg_elem =
        (((total + kCoreNum - 1) / kCoreNum) + 15u) & ~15u;

    uint32_t chunk_off[kCoreNum] = {};
    uint32_t chunk_size[kCoreNum] = {};
    uint32_t num_rounds = 1;
    for (uint32_t j = 0; j < kCoreNum; ++j) {
        chunk_off[j] = std::min(j * avg_elem, total);
        chunk_size[j] = std::min(avg_elem, total - chunk_off[j]);
        num_rounds = std::max(
            num_rounds,
            (chunk_size[j] + kRoundElemMax - 1) / kRoundElemMax);
    }
    const bool use_nopace =
        !residual_local_shard &&
        static_cast<uint64_t>(avg_elem) * kDwidth <
            kRingNopaceChunkBytes;

    const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(
        input_spm >= spm_base0 && output_spm >= spm_base0,
        "Wall-OSS mixed ring received an invalid SPM address");
    const uint32_t input_off = input_spm - spm_base0;
    const uint32_t output_off = output_spm - spm_base0;
    uint32_t residual_base = 0;
    if (residual_ddr != nullptr) {
        const uint64_t residual_dev_addr =
            RpuGetDevAddr(const_cast<c10::Half*>(residual_ddr));
        TORCH_CHECK((residual_dev_addr & 0xFFU) == 0
                        && (residual_dev_addr >> 8) <= UINT32_MAX,
                    "Wall-OSS mixed ring DDR residual has an invalid address");
        residual_base = static_cast<uint32_t>(residual_dev_addr >> 8);
    } else {
        TORCH_CHECK(residual_spm >= spm_base0,
                    "Wall-OSS mixed ring received an invalid residual SPM address");
        residual_base = residual_spm - spm_base0;
    }

    if (fused_postnorm) {
        const uint64_t tensor_bytes =
            static_cast<uint64_t>(total) * kDwidth;
        const uint64_t raw_bytes = tensor_bytes / kCoreNum;
        const uint64_t gamma_bytes = static_cast<uint64_t>(N) * kDwidth;
        const auto in_current_spm =
            [spm_base0](uint32_t address, uint64_t bytes) {
                const uint64_t offset =
                    static_cast<uint64_t>(address) - spm_base0;
                return address >= spm_base0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    offset <= SpmAllocator::SPM_USABLE - bytes;
            };
        const auto disjoint = [](uint32_t lhs, uint64_t lhs_bytes,
                                 uint32_t rhs, uint64_t rhs_bytes) {
            const uint64_t lhs_begin = lhs;
            const uint64_t rhs_begin = rhs;
            return lhs_begin + lhs_bytes <= rhs_begin ||
                rhs_begin + rhs_bytes <= lhs_begin;
        };
        if (exact_m576_local_postnorm_kernel) {
            TORCH_CHECK(
                in_current_spm(input_spm, tensor_bytes) &&
                    in_current_spm(residual_spm, raw_bytes) &&
                    in_current_spm(postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             residual_spm, raw_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(residual_spm, raw_bytes,
                             postnorm->gamma_spm, gamma_bytes),
                "Wall Vision local-inout postnorm requires complete, "
                "disjoint carrier, compact, and gamma SPM ranges");
        } else if (exact_m320_local_postnorm_inplace) {
            TORCH_CHECK(
                in_current_spm(input_spm, tensor_bytes) &&
                    in_current_spm(residual_spm, raw_bytes) &&
                    in_current_spm(postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             residual_spm, raw_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(residual_spm, raw_bytes,
                             postnorm->gamma_spm, gamma_bytes),
                "Wall Text local-inout postnorm requires complete, disjoint "
                "carrier, compact, and gamma SPM ranges");
        } else if (residual_local_shard) {
            TORCH_CHECK(
                in_current_spm(input_spm, tensor_bytes) &&
                    in_current_spm(residual_spm, raw_bytes) &&
                    in_current_spm(output_spm, tensor_bytes) &&
                    in_current_spm(postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             residual_spm, raw_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             output_spm, tensor_bytes) &&
                    disjoint(residual_spm, raw_bytes,
                             output_spm, tensor_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(residual_spm, raw_bytes,
                             postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(output_spm, tensor_bytes,
                             postnorm->gamma_spm, gamma_bytes),
                "Wall-OSS local fused ring postnorm requires complete, "
                "disjoint partial, normalized, compact, and gamma SPM ranges");
        } else {
            TORCH_CHECK(
                in_current_spm(input_spm, tensor_bytes) &&
                    in_current_spm(residual_spm, tensor_bytes) &&
                    in_current_spm(postnorm->raw_residual_local_spm, raw_bytes) &&
                    in_current_spm(postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             residual_spm, tensor_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             postnorm->raw_residual_local_spm, raw_bytes) &&
                    disjoint(residual_spm, tensor_bytes,
                             postnorm->raw_residual_local_spm, raw_bytes) &&
                    disjoint(input_spm, tensor_bytes,
                             postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(residual_spm, tensor_bytes,
                             postnorm->gamma_spm, gamma_bytes) &&
                    disjoint(postnorm->raw_residual_local_spm, raw_bytes,
                             postnorm->gamma_spm, gamma_bytes),
                "Wall-OSS fused ring postnorm requires complete, disjoint "
                "partial/normalized carrier, full residual, compact raw, "
                "and gamma SPM ranges");
        }
    }

    KernelId id = exact_m576_postnorm
        ? exact_m576_postnorm_id
        : fused_postnorm
        ? (residual_local_shard
            ? KernelId::WALL_RING_POSTNORM_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320N2048
            : KernelId::WALL_RING_POSTNORM_FP16_PARTIAL_BF16_RESIDUAL_M320N2048)
        : residual_local_shard
        ? KernelId::LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_SPM
        : (partial_is_bf16
            ? (use_nopace
                ? KernelId::LLM_ALL_REDUCE_BF16_PARTIAL_FP16_RESIDUAL_NOPACE
                : KernelId::LLM_ALL_REDUCE_BF16_PARTIAL_FP16_RESIDUAL)
            : (use_nopace
                ? KernelId::LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_NOPACE
                : KernelId::LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL));
    TORCH_INTERNAL_ASSERT(
        !exact_m320_local_postnorm_inplace ||
        id == KernelId::WALL_RING_POSTNORM_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320N2048);
    Kernel_t* kernel = GET_KERNEL(id);
    TORCH_CHECK(
        kernel != nullptr,
        "Wall-OSS mixed ring all-reduce kernel is unavailable; install the "
        "combined rhinoExpansionOpLib_current.ref or set "
        "RPU_KERNEL_LIB_PATH_EXPANSION");
    kernel->reset_regs();
    kernel->set_regs(0, static_cast<uint16_t>(kCoreNum));
    kernel->set_regs(1, static_cast<uint16_t>(num_rounds));
    kernel->set_regs(2, static_cast<uint16_t>(kRoundElemMax & 0xFFFF));
    kernel->set_regs(3, static_cast<uint16_t>(kRoundElemMax >> 16));
    kernel->set_regs(4, static_cast<uint16_t>(residual_base & 0xFFFF));
    kernel->set_regs(5, static_cast<uint16_t>(residual_base >> 16));
    kernel->set_regs(
        6, static_cast<uint16_t>(
               residual_local_shard ? 2 : (residual_ddr != nullptr)));
    if (fused_postnorm) {
        const uint32_t raw_base =
            postnorm->raw_residual_local_spm - spm_base0;
        const uint32_t gamma_base = postnorm->gamma_spm - spm_base0;
        kernel->set_regs(7, static_cast<uint16_t>(raw_base & 0xFFFF));
        kernel->set_regs(8, static_cast<uint16_t>(raw_base >> 16));
        kernel->set_regs(9, static_cast<uint16_t>(0));
        kernel->set_regs(10, static_cast<uint16_t>(gamma_base & 0xFFFF));
        kernel->set_regs(11, static_cast<uint16_t>(gamma_base >> 16));
        kernel->set_regs(
            12, static_cast<c10::Half>(1.0f / static_cast<float>(N)).x);
        kernel->set_regs(13, static_cast<c10::Half>(postnorm->eps).x);
    }
    kernel->set_regs(64, static_cast<uint16_t>(kWarpNum));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    constexpr uint32_t S = 4096;
    for (uint32_t j = 0; j < kCoreNum; ++j) {
        const uint32_t in_base = SPM_ALLOC.addr(j, input_off) - spm_base0;
        const uint32_t out_base = SPM_ALLOC.addr(j, output_off) - spm_base0;
        const uint32_t chunk_byte_off = chunk_off[j] * kDwidth;
        rpu_set_legacy_scm_u16_checked(
            kernel, S + j, static_cast<uint16_t>(in_base & 0xFFFF),
            "Wall-OSS ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 8 + j, static_cast<uint16_t>(in_base >> 16),
            "Wall-OSS ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 16 + j, static_cast<uint16_t>(out_base & 0xFFFF),
            "Wall-OSS ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 24 + j, static_cast<uint16_t>(out_base >> 16),
            "Wall-OSS ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 32 + j,
            static_cast<uint16_t>(chunk_byte_off & 0xFFFF),
            "Wall-OSS ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 40 + j, static_cast<uint16_t>(chunk_byte_off >> 16),
            "Wall-OSS ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 48 + j,
            static_cast<uint16_t>(chunk_size[j] & 0xFFFF),
            "Wall-OSS ring all-reduce");
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 56 + j, static_cast<uint16_t>(chunk_size[j] >> 16),
            "Wall-OSS ring all-reduce");
    }

    auto* queue = GET_QUEUE(kCoreNum);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel,
        {static_cast<uint16_t>(kWarpNum), static_cast<uint16_t>(1),
         static_cast<uint16_t>(1)},
        {0, 1, 2, 3, 4, 5, 6, 7});
}

void launch_wall_down_postnorm_m576(
    uint32_t partial_normalized_spm,
    uint32_t residual_spm,
    uint32_t raw_residual_local_spm,
    uint32_t gamma_spm,
    double eps,
    bool partial_is_bf16,
    KernelId kernel_id)
{
    const WallRingPostNormParams postnorm{
        raw_residual_local_spm, gamma_spm, eps};
    rpu_launch_wall_oss_mixed_ring_allreduce(
        partial_normalized_spm, residual_spm, nullptr,
        partial_normalized_spm,
        /*M=*/576, /*N=*/1280, partial_is_bf16,
        /*residual_local_shard=*/false, &postnorm,
        /*allow_exact_m576_inplace=*/false, kernel_id);
}

}  // namespace

void rpu_launch_wall_down_postnorm_m576_fp16_partial_fp16_residual_kernel(
    uint32_t partial_normalized_spm,
    uint32_t residual_spm,
    uint32_t raw_residual_local_spm,
    uint32_t gamma_spm,
    double eps)
{
    launch_wall_down_postnorm_m576(
        partial_normalized_spm, residual_spm, raw_residual_local_spm,
        gamma_spm, eps, /*partial_is_bf16=*/false,
        KernelId::WALL_DOWN_POSTNORM_M576_FP16_PARTIAL_FP16_RESIDUAL);
}

void rpu_launch_wall_down_postnorm_m576_bf16_partial_fp16_residual_kernel(
    uint32_t partial_normalized_spm,
    uint32_t residual_spm,
    uint32_t raw_residual_local_spm,
    uint32_t gamma_spm,
    double eps)
{
    launch_wall_down_postnorm_m576(
        partial_normalized_spm, residual_spm, raw_residual_local_spm,
        gamma_spm, eps, /*partial_is_bf16=*/true,
        KernelId::WALL_DOWN_POSTNORM_M576_BF16_PARTIAL_FP16_RESIDUAL);
}

void rpu_launch_wall_down_postnorm_m576_fp16_partial_bf16_residual_kernel(
    uint32_t partial_normalized_spm,
    uint32_t residual_spm,
    uint32_t raw_residual_local_spm,
    uint32_t gamma_spm,
    double eps)
{
    launch_wall_down_postnorm_m576(
        partial_normalized_spm, residual_spm, raw_residual_local_spm,
        gamma_spm, eps, /*partial_is_bf16=*/false,
        KernelId::WALL_DOWN_POSTNORM_M576_FP16_PARTIAL_BF16_RESIDUAL);
}

namespace {

void launch_wall_ring_postnorm_m576_local_inout(
    uint32_t partial_normalized_spm,
    uint32_t residual_local_spm,
    uint32_t gamma_spm,
    double eps,
    bool partial_is_bf16,
    KernelId kernel_id)
{
    const WallRingPostNormParams postnorm{
        residual_local_spm, gamma_spm, eps};
    rpu_launch_wall_oss_mixed_ring_allreduce(
        partial_normalized_spm, residual_local_spm, nullptr,
        partial_normalized_spm,
        /*M=*/576, /*N=*/1280, partial_is_bf16,
        /*residual_local_shard=*/true, &postnorm,
        /*allow_exact_m576_inplace=*/false, kernel_id);
}

}  // namespace

void rpu_launch_wall_ring_postnorm_m576_fp16_partial_fp16_residual_local_inout_kernel(
    uint32_t partial_normalized_spm,
    uint32_t residual_local_spm,
    uint32_t gamma_spm,
    double eps)
{
    launch_wall_ring_postnorm_m576_local_inout(
        partial_normalized_spm, residual_local_spm, gamma_spm, eps,
        /*partial_is_bf16=*/false,
        KernelId::WALL_RING_POSTNORM_M576_FP16_PARTIAL_FP16_RESIDUAL_LOCAL_INOUT);
}

void rpu_launch_wall_ring_postnorm_m576_bf16_partial_fp16_residual_local_inout_kernel(
    uint32_t partial_normalized_spm,
    uint32_t residual_local_spm,
    uint32_t gamma_spm,
    double eps)
{
    launch_wall_ring_postnorm_m576_local_inout(
        partial_normalized_spm, residual_local_spm, gamma_spm, eps,
        /*partial_is_bf16=*/true,
        KernelId::WALL_RING_POSTNORM_M576_BF16_PARTIAL_FP16_RESIDUAL_LOCAL_INOUT);
}

void rpu_launch_wall_ring_postnorm_m576_fp16_partial_bf16_residual_local_inout_kernel(
    uint32_t partial_normalized_spm,
    uint32_t residual_local_spm,
    uint32_t gamma_spm,
    double eps)
{
    launch_wall_ring_postnorm_m576_local_inout(
        partial_normalized_spm, residual_local_spm, gamma_spm, eps,
        /*partial_is_bf16=*/false,
        KernelId::WALL_RING_POSTNORM_M576_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT);
}

void rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_spm_kernel(
    uint32_t partial_spm,
    uint32_t residual_norm_spm,
    uint32_t raw_residual_local_spm,
    uint32_t gamma_spm,
    double eps)
{
    const WallRingPostNormParams postnorm{
        raw_residual_local_spm, gamma_spm, eps};
    rpu_launch_wall_oss_mixed_ring_allreduce(
        partial_spm, residual_norm_spm, nullptr, residual_norm_spm,
        /*M=*/320, /*N=*/2048,
        /*partial_is_bf16=*/false,
        /*residual_local_shard=*/false,
        &postnorm);
}

void rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_local_spm_kernel(
    uint32_t partial_spm,
    uint32_t residual_local_spm,
    uint32_t normalized_spm,
    uint32_t gamma_spm,
    double eps)
{
    const WallRingPostNormParams postnorm{
        residual_local_spm, gamma_spm, eps};
    rpu_launch_wall_oss_mixed_ring_allreduce(
        partial_spm, residual_local_spm, nullptr, normalized_spm,
        /*M=*/320, /*N=*/2048,
        /*partial_is_bf16=*/false,
        /*residual_local_shard=*/true,
        &postnorm);
}

void rpu_launch_wall_ring_postnorm_pair_fp16_partial_bf16_residual_local_spm_kernel(
    uint32_t partial_normalized_pair_spm,
    uint32_t residual0_local_spm,
    uint32_t residual1_local_spm,
    uint32_t gamma_spm,
    double eps, bool tile40)
{
    constexpr uint32_t kCoreNum = 8;
    constexpr uint32_t kWarpNum = 8;
    constexpr uint32_t kRows = 320;
    constexpr uint32_t kHidden = 2048;
    constexpr uint32_t kDwidth = 2;
    constexpr uint32_t kTensorBytes = kRows * kHidden * kDwidth;
    constexpr uint32_t kShardElems = kRows * kHidden / kCoreNum;
    constexpr uint32_t kShardBytes = kShardElems * kDwidth;
    // Exact paired carrier ABI: tile40 uses400 VLM slots and preserves peer order.
    const uint32_t kRoundElems = (tile40 ? 40u : 31u) * 256 * (kWarpNum / 2);
    const uint32_t kNumRounds =
        (kShardElems + kRoundElems - 1) / kRoundElems;

    const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
    const auto in_current_spm = [spm_base0](uint32_t address, uint64_t bytes) {
        const uint64_t offset = static_cast<uint64_t>(address) - spm_base0;
        return address >= spm_base0 &&
            bytes <= SpmAllocator::SPM_USABLE &&
            offset <= SpmAllocator::SPM_USABLE - bytes;
    };
    const auto disjoint = [](uint32_t lhs, uint64_t lhs_bytes,
                             uint32_t rhs, uint64_t rhs_bytes) {
        const uint64_t lhs_begin = lhs;
        const uint64_t rhs_begin = rhs;
        return lhs_begin + lhs_bytes <= rhs_begin ||
            rhs_begin + rhs_bytes <= lhs_begin;
    };
    TORCH_CHECK(
        in_current_spm(partial_normalized_pair_spm, 2 * kTensorBytes) &&
            in_current_spm(residual0_local_spm, kShardBytes) &&
            in_current_spm(residual1_local_spm, kShardBytes) &&
            in_current_spm(gamma_spm, kHidden * kDwidth) &&
            disjoint(partial_normalized_pair_spm, 2 * kTensorBytes,
                     residual0_local_spm, kShardBytes) &&
            disjoint(partial_normalized_pair_spm, 2 * kTensorBytes,
                     residual1_local_spm, kShardBytes) &&
            disjoint(partial_normalized_pair_spm, 2 * kTensorBytes,
                     gamma_spm, kHidden * kDwidth) &&
            disjoint(residual0_local_spm, kShardBytes,
                     residual1_local_spm, kShardBytes) &&
            disjoint(residual0_local_spm, kShardBytes,
                     gamma_spm, kHidden * kDwidth) &&
            disjoint(residual1_local_spm, kShardBytes,
                     gamma_spm, kHidden * kDwidth),
        "Wall P640 paired ring/postnorm requires contiguous carriers and "
        "disjoint compact residual/gamma ranges");

    Kernel_t* kernel = GET_KERNEL(tile40
        ? KernelId::WALL_RING_POSTNORM_PAIR_TILE40_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320X2N2048
        : KernelId::WALL_RING_POSTNORM_PAIR_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320X2N2048);
    TORCH_CHECK(
        kernel != nullptr,
        "Wall P640 paired ring/postnorm kernel is unavailable; install the "
        "matching expansion ref");
    const uint32_t pair_off = partial_normalized_pair_spm - spm_base0;
    const uint32_t residual0_off = residual0_local_spm - spm_base0;
    const uint32_t residual1_off = residual1_local_spm - spm_base0;
    const uint32_t gamma_off = gamma_spm - spm_base0;

    kernel->reset_regs();
    kernel->set_regs(0, static_cast<uint16_t>(kCoreNum));
    kernel->set_regs(1, static_cast<uint16_t>(kNumRounds));
    kernel->set_regs(2, static_cast<uint16_t>(kRoundElems));
    kernel->set_regs(3, static_cast<uint16_t>(0));
    kernel->set_regs(4, static_cast<uint16_t>(residual0_off & 0xFFFF));
    kernel->set_regs(5, static_cast<uint16_t>(residual0_off >> 16));
    kernel->set_regs(6, static_cast<uint16_t>(2));
    kernel->set_regs(7, static_cast<uint16_t>(residual0_off & 0xFFFF));
    kernel->set_regs(8, static_cast<uint16_t>(residual0_off >> 16));
    kernel->set_regs(10, static_cast<uint16_t>(gamma_off & 0xFFFF));
    kernel->set_regs(11, static_cast<uint16_t>(gamma_off >> 16));
    kernel->set_regs(
        12, static_cast<c10::Half>(1.0f / static_cast<float>(kHidden)).x);
    kernel->set_regs(13, static_cast<c10::Half>(eps).x);
    kernel->set_regs(14, static_cast<uint16_t>(residual1_off & 0xFFFF));
    kernel->set_regs(15, static_cast<uint16_t>(residual1_off >> 16));
    kernel->set_regs(64, static_cast<uint16_t>(kWarpNum));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    constexpr uint32_t kScmBase = 4096;
    for (uint32_t peer = 0; peer < kCoreNum; ++peer) {
        const uint32_t carrier = SPM_ALLOC.addr(peer, pair_off) - spm_base0;
        const uint32_t chunk_bytes = peer * kShardBytes;
        rpu_set_legacy_scm_u16_checked(
            kernel, kScmBase + peer,
            static_cast<uint16_t>(carrier & 0xFFFF), "Wall P640 pair");
        rpu_set_legacy_scm_u16_checked(
            kernel, kScmBase + 8 + peer,
            static_cast<uint16_t>(carrier >> 16), "Wall P640 pair");
        rpu_set_legacy_scm_u16_checked(
            kernel, kScmBase + 16 + peer,
            static_cast<uint16_t>(carrier & 0xFFFF), "Wall P640 pair");
        rpu_set_legacy_scm_u16_checked(
            kernel, kScmBase + 24 + peer,
            static_cast<uint16_t>(carrier >> 16), "Wall P640 pair");
        rpu_set_legacy_scm_u16_checked(
            kernel, kScmBase + 32 + peer,
            static_cast<uint16_t>(chunk_bytes & 0xFFFF), "Wall P640 pair");
        rpu_set_legacy_scm_u16_checked(
            kernel, kScmBase + 40 + peer,
            static_cast<uint16_t>(chunk_bytes >> 16), "Wall P640 pair");
        rpu_set_legacy_scm_u16_checked(
            kernel, kScmBase + 48 + peer,
            static_cast<uint16_t>(kShardElems & 0xFFFF), "Wall P640 pair");
        rpu_set_legacy_scm_u16_checked(
            kernel, kScmBase + 56 + peer,
            static_cast<uint16_t>(kShardElems >> 16), "Wall P640 pair");
    }

    auto* queue = GET_QUEUE(kCoreNum);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel,
        {static_cast<uint16_t>(kWarpNum), static_cast<uint16_t>(1),
         static_cast<uint16_t>(1)},
        {0, 1, 2, 3, 4, 5, 6, 7});
}

static void launch_wall_base_owner_ring_norm(
    uint32_t partial_spm,
    uint32_t residual_spm,
    uint32_t raw_spm,
    uint32_t gamma_spm,
    double eps,
    KernelId id,
    bool norm_over_residual, uint32_t rows, uint32_t hidden)
{
    TORCH_CHECK((rows == 672 && hidden == 1280) || (rows == 224 && hidden == 2048),
                "Wall base owner ring/norm requires an admitted exact geometry");
    const uint32_t kRows = rows, kHidden = hidden;
    constexpr uint32_t kCores = 8;
    const uint32_t kFullBytes = kRows * kHidden * sizeof(c10::Half);
    const uint32_t kOwnerElems = kRows * kHidden / kCores;
    constexpr uint32_t kRoundElems = 31 * 256 * kCores;
    TORCH_CHECK(std::isfinite(eps) && eps == 1e-6,
                "Wall base owner ring/norm requires epsilon 1e-6");
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "Wall base owner ring/norm requires initialized SPM");
    const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint64_t, uint64_t>, 4> slabs{{
        {partial_spm, kFullBytes}, {residual_spm, kFullBytes},
        {raw_spm, kFullBytes}, {gamma_spm, kHidden * sizeof(c10::Half)}}};
    for (size_t i = 0; i < slabs.size(); ++i) {
        const auto [address, bytes] = slabs[i];
        TORCH_CHECK(address != 0 && address >= spm_base && address % 256 == 0 &&
                        bytes <= SpmAllocator::SPM_USABLE &&
                        address - spm_base <= SpmAllocator::SPM_USABLE - bytes,
                    "Wall base owner ring/norm requires complete aligned absolute SPM slabs");
        for (uint32_t core = 0; core < kCores; ++core) {
            const uint64_t peer_base = SPM_ALLOC.addr(core, 0);
            TORCH_CHECK(peer_base >= spm_base &&
                            peer_base + address - spm_base + bytes - 1 <= UINT32_MAX,
                        "Wall base owner ring/norm SPM range exceeds the register ABI");
        }
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(address + bytes <= slabs[j].first ||
                            slabs[j].first + slabs[j].second <= address,
                        "Wall base owner ring/norm requires disjoint partial, residual, raw and gamma");
        }
    }
    TORCH_CHECK(KernelCache::instance().has_loaded(id),
                "Wall base owner ring/norm expansion payload is unavailable");
    auto& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active()) {
        graph.stage_kernel_no_ddr(id, GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "Wall base owner ring/norm kernel is missing");
    kernel->reset_regs();
    const auto u16 = [&](uint32_t reg, uint16_t value) {
        TORCH_CHECK(kernel->set_regs(reg, value) == kKernelRegOk,
                    "Wall base owner ring/norm register write failed: ", reg);
    };
    const auto u32 = [&](uint32_t reg, uint32_t value) {
        u16(reg, static_cast<uint16_t>(value));
        u16(reg + 1, static_cast<uint16_t>(value >> 16));
    };
    u16(0, kCores); u16(1, (kOwnerElems + kRoundElems - 1) / kRoundElems);
    u32(2, kRoundElems); u32(4, residual_spm - spm_base); u16(6, 0);
    // Unchanged R19 bitcopy V32 math. Seven blocks cover either 12 Vision
    // or 4 Prefill owner rows each; block seven only joins the barriers.
    u16(8, kHidden); u16(9, c10::Half(1.0f / kHidden).x);
    u16(10, kHidden * sizeof(c10::Half)); u16(11, kRows / kCores / 7);
    const uint32_t normalized_spm = norm_over_residual ? residual_spm : partial_spm;
    u32(12, raw_spm); u32(14, normalized_spm); u32(16, gamma_spm);
    u16(18, c10::Half(eps).x);
    const uint32_t partial_offset = partial_spm - spm_base;
    const uint32_t raw_offset = raw_spm - spm_base;
    const uint32_t normalized_offset = normalized_spm - spm_base;
    for (uint32_t core = 0; core < kCores; ++core) {
        const uint32_t peer_partial = SPM_ALLOC.addr(core, partial_offset) - spm_base;
        const uint32_t peer_raw = SPM_ALLOC.addr(core, raw_offset) - spm_base;
        const uint32_t peer_normalized = SPM_ALLOC.addr(core, normalized_offset) - spm_base;
        const std::array<std::pair<uint32_t, uint32_t>, 5> fields{{
            {0, peer_partial}, {16, peer_raw}, {32, core * kOwnerElems * 2},
            {48, kOwnerElems}, {64, peer_normalized}}};
        for (const auto [field, value] : fields) {
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + field + core,
                static_cast<uint16_t>(value), "Wall base owner ring/norm");
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + field + 8 + core,
                static_cast<uint16_t>(value >> 16), "Wall base owner ring/norm");
        }
    }
    u16(64, 8); u16(65, 1); u16(66, 1);
    // Reduction/gather use core24..31/global24; the norm bridge uses
    // core0/1 and global0..7. Reset the complete bank for every packet.
    for (uint32_t reg = 128; reg < 256; reg += 2) {
        TORCH_CHECK(kernel->set_regs(reg, uint32_t{0}) == kKernelRegOk,
                    "Wall base owner ring/norm counter reset failed: ", reg);
    }
    auto* queue = GET_QUEUE(kCores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_wall_base_vision_owner_ring_norm_m672_kernel(
    uint32_t partial_norm_spm,
    uint32_t residual_spm,
    uint32_t raw_spm,
    uint32_t gamma_spm,
    double eps)
{
    launch_wall_base_owner_ring_norm(
        partial_norm_spm, residual_spm, raw_spm, gamma_spm, eps,
        KernelId::WALL_BASE_VISION_OWNER_RING_NORM_M672,
        /*norm_over_residual=*/false, 672, 1280);
}

void rpu_launch_wall_base_vision_down_owner_ring_norm_m672_kernel(
    uint32_t partial_spm,
    uint32_t residual_norm_spm,
    uint32_t raw_spm,
    uint32_t gamma_spm,
    double eps)
{
    launch_wall_base_owner_ring_norm(
        partial_spm, residual_norm_spm, raw_spm, gamma_spm, eps,
        KernelId::WALL_BASE_VISION_DOWN_OWNER_RING_NORM_M672,
        /*norm_over_residual=*/true, 672, 1280);
}

void rpu_launch_wall_base_prefill_o_owner_ring_norm_m224_kernel(
    uint32_t partial_spm, uint32_t residual_norm_spm,
    uint32_t raw_spm, uint32_t gamma_spm, double eps)
{
    launch_wall_base_owner_ring_norm(
        partial_spm, residual_norm_spm, raw_spm, gamma_spm, eps,
        KernelId::WALL_BASE_PREFILL_O_OWNER_RING_NORM_M224,
        /*norm_over_residual=*/true, 224, 2048);
}

void rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N)
{
    rpu_launch_wall_oss_mixed_ring_allreduce(
        input_spm, residual_spm, nullptr, output_spm, M, N,
        /*partial_is_bf16=*/true,
        /*residual_local_shard=*/false);
}

void rpu_launch_wall_vision_all_reduce_bf16_partial_fp16_residual_m576_inplace_kernel(
    uint32_t partial_output_spm,
    uint32_t residual_spm)
{
    rpu_launch_wall_oss_mixed_ring_allreduce(
        partial_output_spm, residual_spm, nullptr, partial_output_spm,
        /*M=*/576, /*N=*/1280,
        /*partial_is_bf16=*/true,
        /*residual_local_shard=*/false,
        /*postnorm=*/nullptr,
        /*allow_exact_m576_inplace=*/true);
}

void rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N)
{
    rpu_launch_wall_oss_mixed_ring_allreduce(
        input_spm, residual_spm, nullptr, output_spm, M, N,
        /*partial_is_bf16=*/false,
        /*residual_local_shard=*/false);
}

void rpu_launch_wall_vision_all_reduce_fp16_partial_bf16_residual_m576_inplace_kernel(
    uint32_t partial_output_spm,
    uint32_t residual_spm)
{
    rpu_launch_wall_oss_mixed_ring_allreduce(
        partial_output_spm, residual_spm, nullptr, partial_output_spm,
        /*M=*/576, /*N=*/1280,
        /*partial_is_bf16=*/false,
        /*residual_local_shard=*/false,
        /*postnorm=*/nullptr,
        /*allow_exact_m576_inplace=*/true);
}

void rpu_launch_all_reduce_fp16_partial_bf16_residual_local_spm_kernel(
    uint32_t input_spm,
    uint32_t residual_local_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N)
{
    rpu_launch_wall_oss_mixed_ring_allreduce(
        input_spm, residual_local_spm, nullptr, output_spm, M, N,
        /*partial_is_bf16=*/false,
        /*residual_local_shard=*/true);
}
