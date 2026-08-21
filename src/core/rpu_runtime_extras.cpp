// Miscellaneous runtime helpers shared by Python bindings and fused models:
//
//   - set_spm_debug / get_spm_debug                          (Python-bound)
//   - set_cross_layer_batch_prefill / get_cross_layer_batch_prefill  (Python-bound)
//   - rpu_launch_all_reduce_sum_residual_kernel              (called by 4 v3 files)
//
// Forward declarations stay in `src/core/rpu_kernel_decls.h`.

#include "rhino_launch_buffer.h"
#include "rhino_launch_kernel.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "fused_model_base.h"
#include "rpu_lingbot2_ring_collective.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"  // SPM_ALLOC (absolute→local SPM offset for two-stage)
#include "rpu_spm_pipeline.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
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
};

}  // namespace v3

// =============================================================================
// File-local runtime state
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
// Two-stage all-reduce fast-path toggles (env, read while emitting a graph)
// =============================================================================
//
// RPU_ALLREDUCE_TWOSTAGE   — default OFF; "1"/"true" → on. This helper is shared
//                           by non-Pi05 fused models (rpu_qwen3_model, etc.), and
//                           equivalence is model/profile-specific, so the global
//                           default does not opt into two-stage. The ring has a
//                           separate default-off selector; with both off, the
//                           legacy single-stage route remains the fallback.
//                           Pi05, Wall-OSS, GR00T, RhinoVLA, and the validated
//                           G0.5 RTC/FM profile opt in from
//                           their adapters; other G0.5/models need their own gates.
// RPU_ALLREDUCE_TWOSTAGE_TRACE — default off; "1"/"true" → one stderr line/launch.

static bool allreduce_twostage_enabled() {
    const char* e = std::getenv("RPU_ALLREDUCE_TWOSTAGE");
    return e && (std::string(e) == "1" || std::string(e) == "true");
}

bool rpu_get_allreduce_twostage_enabled() {
    return allreduce_twostage_enabled();
}

static bool allreduce_twostage_trace_enabled() {
    const char* e = std::getenv("RPU_ALLREDUCE_TWOSTAGE_TRACE");
    return e && (std::string(e) == "1" || std::string(e) == "true");
}

// A traced 4096-token prefill emits hundreds of reductions.  Keep tracing
// decision-grade without turning the log itself into the dominant cost.
static bool allreduce_trace_first_time(
    const char* path,
    const char* residual_kind,
    int cores,
    int64_t M,
    int64_t N) {
    static std::mutex mutex;
    static std::set<std::tuple<
        std::string, std::string, int, int64_t, int64_t>> seen;
    std::lock_guard<std::mutex> lock(mutex);
    return seen.emplace(path, residual_kind, cores, M, N).second;
}

// Explicit, globally-off ring selector. `profile` admits the profile-scoped
// shapes; `auto` applies the automatic crossover policy.
enum class AllreduceRingMode {
    kOff,
    kAuto,
    kNopace,
    kPaced,
    kProfile,
};

static AllreduceRingMode allreduce_ring_mode() {
    const char* value = std::getenv("RPU_ALLREDUCE_RING");
    const std::string mode = value ? std::string(value) : std::string();
    if (mode == "nopace") return AllreduceRingMode::kNopace;
    if (mode == "paced") return AllreduceRingMode::kPaced;
    if (mode == "1" || mode == "true" || mode == "auto") {
        return AllreduceRingMode::kAuto;
    }
    if (mode == "profile") return AllreduceRingMode::kProfile;
    return AllreduceRingMode::kOff;
}

// Opt-in: when M*N exceeds the two-stage cap, split M into cap-sized row
// pieces. The reduction is row-independent. Default off.
static bool allreduce_chunk_v2_enabled() {
    const char* e = std::getenv("RPU_ALLREDUCE_CHUNK_V2");
    return e && (std::string(e) == "1" || std::string(e) == "true");
}

// =============================================================================
// Fused ring all-reduce + residual
// =============================================================================
//
// An explicit process- or handle-scoped two-stage policy has priority. Otherwise the
// public residual launcher admits this single-kernel route before the legacy
// fallbacks.

namespace {

constexpr int kRingCoreNumMax = 8;
constexpr uint32_t kRingWarpNum = 8;
constexpr uint32_t kRingRoundElemMax =
    31U * 256U * kRingWarpNum;
constexpr uint64_t kRingNopaceChunkBytes = 11U * 1024U;
constexpr const char* kRingNopaceKernel =
    "llm_all_reduce_residual_nopace";
constexpr const char* kRingPacedKernel =
    "llm_all_reduce_residual";
constexpr const char* kRingLocalSpmKernel =
    "llm_all_reduce_residual_local_spm";

bool ring_all_reduce_sum_residual_admitted(
    uint32_t input_spm,
    uint32_t residual_spm,
    const c10::Half* residual_ddr,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores,
    bool residual_local_shard = false) {
    if (input_num_cores != output_num_cores || input_num_cores < 2 ||
        input_num_cores > kRingCoreNumMax || M <= 0 || N <= 0 ||
        !SPM_ALLOC.is_initialized()) {
        return false;
    }
    const bool admitted_local_shape =
        (M == 576 && N == 1280) || (M == 320 && N == 2048);
    if (residual_local_shard &&
        (residual_ddr != nullptr || input_num_cores != kRingCoreNumMax ||
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
        ((total_elements + static_cast<uint64_t>(input_num_cores) - 1) /
             static_cast<uint64_t>(input_num_cores) +
         15U) /
        16U * 16U;
    const uint64_t residual_bytes = residual_local_shard
        ? avg_elements * kFp16Bytes
        : tensor_bytes;

    const uint64_t spm_base0 = SPM_ALLOC.addr(0, 0);
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

bool launch_ring_all_reduce_sum_residual(
    uint32_t input_spm,
    uint32_t residual_spm,
    const c10::Half* residual_ddr,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores,
    AllreduceRingMode mode = AllreduceRingMode::kProfile,
    bool require_kernel = true,
    bool residual_local_shard = false) {
    TORCH_CHECK(
        ring_all_reduce_sum_residual_admitted(
            input_spm, residual_spm, residual_ddr, output_spm,
            M, N, input_num_cores, output_num_cores,
            residual_local_shard),
        "ring all-reduce requires 2..8 equal input/output cores, positive "
        "FP16 geometry, complete out-of-place SPM operands, and an "
        "initialized current SPM arena");

    const int core_num = input_num_cores;
    constexpr uint64_t kFp16Bytes = sizeof(c10::Half);
    const uint64_t total_elements =
        static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
    const uint64_t spm_base0 = SPM_ALLOC.addr(0, 0);

    // Launch ABI contract: ceil-align(ceil-div(total, cores), 16 elements).
    const uint64_t avg_elements_unaligned =
        (total_elements + static_cast<uint64_t>(core_num) - 1) /
        static_cast<uint64_t>(core_num);
    const uint64_t avg_elements =
        ((avg_elements_unaligned + 15U) / 16U) * 16U;
    TORCH_INTERNAL_ASSERT(avg_elements <= UINT32_MAX);
    const uint64_t chunk_bytes = avg_elements * kFp16Bytes;
    const bool use_nopace =
        mode == AllreduceRingMode::kNopace ||
        (mode != AllreduceRingMode::kPaced &&
         chunk_bytes < kRingNopaceChunkBytes);
    TORCH_CHECK(!residual_local_shard || !use_nopace,
                "compact-residual ring supports paced mode only");
    const char* kernel_name = residual_local_shard
        ? kRingLocalSpmKernel
        : (use_nopace ? kRingNopaceKernel : kRingPacedKernel);

    std::array<uint32_t, kRingCoreNumMax> chunk_offsets{};
    std::array<uint32_t, kRingCoreNumMax> chunk_sizes{};
    uint32_t num_rounds = 1;
    for (int core = 0; core < core_num; ++core) {
        const uint64_t offset = std::min<uint64_t>(
            static_cast<uint64_t>(core) * avg_elements, total_elements);
        const uint64_t size = std::min<uint64_t>(
            avg_elements, total_elements - offset);
        chunk_offsets[core] = static_cast<uint32_t>(offset);
        chunk_sizes[core] = static_cast<uint32_t>(size);
        const uint32_t rounds = static_cast<uint32_t>(
            (size + kRingRoundElemMax - 1) /
            kRingRoundElemMax);
        num_rounds = std::max<uint32_t>(num_rounds, rounds);
    }
    TORCH_INTERNAL_ASSERT(num_rounds <= UINT16_MAX);

    const KernelId kernel_id = residual_local_shard
        ? KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM
        : (use_nopace
            ? KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE
            : KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED);
    RpuKernelGraph& graph = RpuKernelGraph::active();
    const bool register_census_active =
        graph.kernel_register_census_active();
    if (register_census_active) {
        TORCH_CHECK(residual_ddr == nullptr,
                    "typed ring all-reduce census rejects a DDR residual");
        graph.stage_kernel_no_ddr(
            kernel_id,
            GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    Kernel_t* kernel = GET_KERNEL(kernel_id);
    if (kernel == nullptr) {
        TORCH_CHECK(
            !require_kernel && !register_census_active,
            "Failed to get ring all-reduce kernel ", kernel_name);
        return false;
    }
    if (allreduce_twostage_trace_enabled() &&
        allreduce_trace_first_time(
            use_nopace ? "ring_nopace" : "ring_paced",
            residual_local_shard
                ? "local_spm"
                : (residual_ddr ? "ddr" : "spm"),
            core_num, M, N)) {
        std::fprintf(
            stderr,
            "ALLREDUCE_RING path=%s residual=%s M=%lld N=%lld "
            "chunkB=%llu rounds=%u\n",
            use_nopace ? "ring_nopace" : "ring_paced",
            residual_local_shard
                ? "local_spm"
                : (residual_ddr ? "ddr" : "spm"),
            static_cast<long long>(M), static_cast<long long>(N),
            static_cast<unsigned long long>(chunk_bytes), num_rounds);
    }
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
    kernel->set_regs(65, 1);
    kernel->set_regs(66, 1);

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
        kernel->set_regs(
            kScmBase + core,
            static_cast<uint16_t>(input_base & 0xFFFFU));
        kernel->set_regs(
            kScmBase + 8 + core,
            static_cast<uint16_t>(input_base >> 16));
        kernel->set_regs(
            kScmBase + 16 + core,
            static_cast<uint16_t>(output_base & 0xFFFFU));
        kernel->set_regs(
            kScmBase + 24 + core,
            static_cast<uint16_t>(output_base >> 16));
        kernel->set_regs(
            kScmBase + 32 + core,
            static_cast<uint16_t>(chunk_offset_bytes & 0xFFFFU));
        kernel->set_regs(
            kScmBase + 40 + core,
            static_cast<uint16_t>(chunk_offset_bytes >> 16));
        kernel->set_regs(
            kScmBase + 48 + core,
            static_cast<uint16_t>(chunk_size_elements & 0xFFFFU));
        kernel->set_regs(
            kScmBase + 56 + core,
            static_cast<uint16_t>(chunk_size_elements >> 16));
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
    return true;
}

}  // namespace

void rpu_launch_lingbot2_ring_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores) {
    launch_ring_all_reduce_sum_residual(
        input_spm, residual_spm, nullptr, output_spm, M, N,
        input_num_cores, output_num_cores);
}

void rpu_launch_lingbot2_ring_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    const v3::SpmFmbPostFnYieldTarget& output,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores) {
    const uint32_t output_spm =
        v3::SpmFmbYieldTargetLauncherAccess::
            resolve_and_stage_all_reduce(
                output, M, N, input_num_cores, output_num_cores);
    try {
        launch_ring_all_reduce_sum_residual(
            input_spm, residual_spm, nullptr, output_spm, M, N,
            input_num_cores, output_num_cores);
    } catch (...) {
        v3::SpmFmbYieldTargetLauncherAccess::cancel_pending();
        throw;
    }
}

// Configurable M*N ceiling for the two-stage v2 path. The default is 400*1024;
// deployments must validate any larger value before enabling it.
inline int64_t allreduce_v2_cap() {
    static int64_t cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_ALLREDUCE_V2_CAP");
        cached = (e && *e) ? std::atoll(e) : (400 * 1024);
        if (cached <= 0) cached = 400 * 1024;
    }
    return cached;
}

// Opt-in companion to CHUNK_V2: split M evenly across the same number of chunks
// instead of greedily filling cap_rows (which leaves a runt tail). See the call
// site for the shapes. Default off => other models keep their current split.
//
// This changes reduction boundaries, so it is disabled by default and requires
// numerical validation before use.
static bool allreduce_chunk_balanced_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_ALLREDUCE_CHUNK_BALANCED");
        cached = (e && (std::string(e) == "1" || std::string(e) == "true")) ? 1 : 0;
    }
    return cached != 0;
}

// `RPU_ALLREDUCE_FUSED` selects the one-launch fused all-reduce variant instead
// of scatter-reduce-residual followed by all-gather. The fused path accumulates
// in fp32 and is not bitwise equivalent to the two-stage reduction, so the
// process-wide default is OFF and each enabled profile requires numerical
// validation.
static bool allreduce_fused_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_ALLREDUCE_FUSED");
        cached = (e && (std::string(e) == "1" || std::string(e) == "true")) ? 1 : 0;
    }
    return cached != 0;
}

// 单枪融合版的 host launch 契约。
// 三个地址参数都是**核 0 的绝对 SPM 地址**（生产调用方一律传 `SPM_ALLOC.addr(0,·)`），
// 这里反推出本地偏移再按核展开。算子需要：
//   resBase   = **本地偏移**（kernel 自己加 spm_base + 核偏移）
//   inBase[j] / outBase[j] = **每核绝对地址**
static void launch_allreduce_fused(uint32_t input_spm, uint32_t residual_spm,
                                   uint32_t output_spm, int64_t M, int64_t N,
                                   int core_num, Kernel_t* kernel) {
    constexpr uint32_t kWarpNum = 8;
    const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
    const uint32_t x_off = input_spm    - spm_base0;
    const uint32_t y_off = output_spm   - spm_base0;
    const uint32_t r_off = residual_spm - spm_base0;

    const uint32_t total = (uint32_t)(M * N);
    const uint32_t round_elem_max = 31u * 256u * kWarpNum;
    auto ceil_div_u = [](uint32_t a, uint32_t b) { return (a + b - 1) / b; };
    auto ceil_align_u = [](uint32_t n, uint32_t a) { return (n + a - 1) & ~(a - 1); };

    // 元素域切分，16 元素对齐，使用 CEIL-div；floor 会丢尾元素。
    const uint32_t avg_elem = ceil_align_u(ceil_div_u(total, (uint32_t)core_num), 16);
    uint32_t off_elem[8] = {}, size_elem[8] = {};
    uint32_t num_rounds = 1;
    for (int j = 0; j < core_num; ++j) {
        off_elem[j]  = std::min((uint32_t)j * avg_elem, total);
        size_elem[j] = std::min(avg_elem, total - off_elem[j]);
        num_rounds = std::max(num_rounds, ceil_div_u(size_elem[j], round_elem_max));
    }

    kernel->reset_regs();
    kernel->set_regs(0, (uint16_t)core_num);
    kernel->set_regs(1, (uint16_t)num_rounds);
    kernel->set_regs(2, (uint16_t)(round_elem_max & 0xFFFF));
    kernel->set_regs(3, (uint16_t)((round_elem_max >> 16) & 0xFFFF));
    kernel->set_regs(4, (uint16_t)(r_off & 0xFFFF));
    kernel->set_regs(5, (uint16_t)((r_off >> 16) & 0xFFFF));
    kernel->set_regs(6, (uint16_t)0);                 // residual 在 SPM
    kernel->set_regs(64, (uint16_t)kWarpNum);
    kernel->set_regs(65, (uint16_t)1);
    kernel->set_regs(66, (uint16_t)1);
    constexpr uint32_t S = 4096;
    for (int j = 0; j < core_num; ++j) {
        const uint32_t ib = SPM_ALLOC.addr(j, x_off);
        const uint32_t ob = SPM_ALLOC.addr(j, y_off);
        const uint32_t coff = off_elem[j] * (uint32_t)sizeof(c10::Half);   // BYTE
        kernel->set_regs(S +  0 + j, (uint16_t)(ib & 0xFFFF));
        kernel->set_regs(S +  8 + j, (uint16_t)((ib >> 16) & 0xFFFF));
        kernel->set_regs(S + 16 + j, (uint16_t)(ob & 0xFFFF));
        kernel->set_regs(S + 24 + j, (uint16_t)((ob >> 16) & 0xFFFF));
        kernel->set_regs(S + 32 + j, (uint16_t)(coff & 0xFFFF));
        kernel->set_regs(S + 40 + j, (uint16_t)((coff >> 16) & 0xFFFF));
        kernel->set_regs(S + 48 + j, (uint16_t)(size_elem[j] & 0xFFFF));
        kernel->set_regs(S + 56 + j, (uint16_t)((size_elem[j] >> 16) & 0xFFFF));
    }

    auto* wq = GET_QUEUE(core_num);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < core_num; ++i) core_list.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {(uint16_t)kWarpNum, 1, 1}, core_list);
}

// Opt-in: let the two-stage path also serve a PARTIAL producer, i.e. the case
// where only `input_num_cores` (< 8) cores hold partial sums but the result must
// land on all 8. That is exactly a GQA tower's attention output: o_proj runs
// row-partition at `attn_tp = min(8, num_kv_heads)`, so with 4 KV heads only
// cores 0..3 wrote partials and cores 4..7 still hold the previous layer's
// bytes. Zero those cores first and the 8-core two-stage sum is IDENTICAL
// (sum over 0..7 == sum over 0..tp-1 + 0).
//
// The `output_num_cores == 8` guard below otherwise routes this case to the
// single-stage `llama_reduce_sum`; zero-filling inactive producers makes the
// two-stage reduction valid for the partial-producer shape.
//
// Default OFF — the zero-fill assumes the caller's input buffer above
// `input_num_cores` is dead, which is a per-model fact. Hy-VLA opts in from
// `build_hy_vla`; every other model keeps its existing path byte-for-byte.
static bool allreduce_partial_twostage_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_ALLREDUCE_PARTIAL_TWOSTAGE");
        cached = (e && (std::string(e) == "1" || std::string(e) == "true")) ? 1 : 0;
    }
    return cached != 0;
}

// =============================================================================
// All-reduce sum + residual kernel launch (used by 4 v3 files: rpu_qwen3_model,
// rpu_adarms_model, fused_model_base, rpu_test_adarms_gemv)
// =============================================================================

static void rpu_launch_all_reduce_sum_residual_impl(
    uint32_t input_spm,                // Partial results from linear (each core has [M, N])
    uint32_t residual_spm,             // Residual (each core has same [M, N])
    const c10::Half* residual_ddr,      // Or one stable shared DDR copy (nullptr => SPM)
    uint32_t output_spm,               // Output (broadcast to all cores)
    int64_t M,                         // rows (seq_len)
    int64_t N,                         // columns (hidden_size)
    int input_num_cores,               // input cores (attn_tp for attn, tp for MLP)
    int output_num_cores,              // output cores (tp = 8)
    bool force_twostage,               // cold, product-scoped fast-path policy
    bool force_chunk_v2,               // cold, product-scoped chunk-v2 policy
    bool force_non_v2)                 // selected two-stage must use non-v2 scatter
{
    TORCH_CHECK(
        !force_chunk_v2 || force_twostage,
        "force_chunk_v2 requires force_twostage");
    TORCH_CHECK(
        !force_chunk_v2 || !force_non_v2,
        "force_chunk_v2 and force_non_v2 are mutually exclusive");
    const bool global_twostage_enabled = allreduce_twostage_enabled();
    const bool twostage_enabled =
        force_twostage || global_twostage_enabled;
    const bool global_chunk_v2_enabled = allreduce_chunk_v2_enabled();
    const bool chunk_enabled =
        !force_non_v2 && (force_chunk_v2 || global_chunk_v2_enabled);
    const bool register_census_active =
        RpuKernelGraph::active().kernel_register_census_active();
    if (register_census_active) {
        TORCH_CHECK(
            residual_ddr == nullptr,
            "typed DDR-register census rejects an all-reduce DDR residual");
    }

    // One range proof is shared by the explicit chunk-v2 recursion and the
    // canonical single-stage census route.  Besides proving that the complete
    // fp16 operands live in the current allocator's core-0 SPM, its division
    // guards make the row/full byte products and every uint32_t offset/address
    // addition below representable before any recursive launch is attempted.
    const uint64_t spm_capacity = SpmAllocator::SPM_USABLE;
    const bool valid_row_byte_count =
        M > 0 && N > 0 &&
        static_cast<uint64_t>(N) <=
            spm_capacity / sizeof(c10::Half) &&
        static_cast<uint64_t>(N) <=
            static_cast<uint64_t>(UINT32_MAX) / sizeof(c10::Half);
    const uint64_t row_bytes = valid_row_byte_count
        ? static_cast<uint64_t>(N) * sizeof(c10::Half)
        : 0;
    const bool valid_spm_byte_count =
        valid_row_byte_count &&
        static_cast<uint64_t>(M) <= spm_capacity / row_bytes &&
        static_cast<uint64_t>(M) <=
            static_cast<uint64_t>(UINT32_MAX) / row_bytes;
    const uint64_t spm_bytes = valid_spm_byte_count
        ? static_cast<uint64_t>(M) * row_bytes
        : 0;
    const auto resolves_to_current_core0_spm =
        [valid_spm_byte_count, spm_bytes](uint32_t address) {
            if (!valid_spm_byte_count || !SPM_ALLOC.is_initialized()) {
                return false;
            }
            const uint64_t base = SPM_ALLOC.addr(0, 0);
            const uint64_t candidate = address;
            return candidate >= base &&
                candidate - base <=
                    SpmAllocator::SPM_USABLE - spm_bytes &&
                candidate <=
                    static_cast<uint64_t>(UINT32_MAX) - spm_bytes;
        };

    // Hy-VLA's profile-scoped fused all-reduce is independent of the generic
    // ring selector. It only admits the validated 8-core, all-SPM envelope.
    if (allreduce_fused_enabled()
        && residual_ddr == nullptr
        && input_num_cores == output_num_cores && output_num_cores == 8
        && ring_all_reduce_sum_residual_admitted(
            input_spm, residual_spm, nullptr, output_spm,
            M, N, input_num_cores, output_num_cores)
        && launch_ring_all_reduce_sum_residual(
            input_spm, residual_spm, nullptr, output_spm,
            M, N, input_num_cores, output_num_cores,
            AllreduceRingMode::kProfile,
            /*require_kernel=*/false)) {
        return;
    }

    // A profile may promote a partial producer by zeroing the inactive cores;
    // the current two-stage/ring routing below then remains authoritative.
    if (allreduce_partial_twostage_enabled()
        && twostage_enabled
        && residual_ddr == nullptr
        && output_num_cores == 8
        && input_num_cores > 0 && input_num_cores < 8
        && valid_spm_byte_count
        && resolves_to_current_core0_spm(input_spm)
        && input_spm != residual_spm && input_spm != output_spm
        && residual_spm != output_spm
        && ((M * N) % 16) == 0) {
        rpu_launch_fill_spm_kernel(
            input_spm, M * N, c10::Half(0.0f),
            /*num_cores=*/8 - input_num_cores,
            /*core_begin=*/input_num_cores);
        input_num_cores = 8;
    }

    // Generic fused-ring routing is globally off. Qwen3 may opt in through
    // RPU_ALLREDUCE_RING, while exact owners which already froze the ring in
    // their graph census use the private `profile` value. That reserved value
    // is authoritative even when the owner also carries a component-local
    // two-stage bit for its outer-fast lifecycle (LingBot2 grouped Z2). All
    // generic ring modes remain below an explicit two-stage selection, which
    // preserves G0.5 K1 and other two-stage-bound trajectories.
    const AllreduceRingMode ring_mode = allreduce_ring_mode();
    const bool profile_ring_owned =
        ring_mode == AllreduceRingMode::kProfile;
    const bool ring_core_profile_admitted =
        profile_ring_owned ||
        (input_num_cores == 8 && output_num_cores == 8);
    const bool ring_precedes_twostage =
        (profile_ring_owned && !global_twostage_enabled) ||
        !twostage_enabled;
    if (ring_precedes_twostage && ring_mode != AllreduceRingMode::kOff &&
        ring_core_profile_admitted &&
        ring_all_reduce_sum_residual_admitted(
            input_spm, residual_spm, residual_ddr, output_spm,
            M, N, input_num_cores, output_num_cores)) {
        const uint64_t total_elements =
            static_cast<uint64_t>(M) * static_cast<uint64_t>(N);
        const uint64_t avg_elements =
            ((total_elements + static_cast<uint64_t>(input_num_cores) - 1) /
                 static_cast<uint64_t>(input_num_cores) +
             15U) /
            16U * 16U;
        const uint64_t chunk_bytes = avg_elements * sizeof(c10::Half);
        const bool ring_wanted =
            ring_mode != AllreduceRingMode::kAuto ||
            chunk_bytes >= kRingNopaceChunkBytes;
        if (ring_wanted && launch_ring_all_reduce_sum_residual(
                input_spm, residual_spm, residual_ddr, output_spm,
                M, N, input_num_cores, output_num_cores, ring_mode,
                /*require_kernel=*/ring_mode == AllreduceRingMode::kProfile)) {
            return;
        }
    }

    // Chunk-to-v2 (opt-in): split M so each piece stays under the v2 cap and
    // recurses onto the fast two-stage path (vs non-v2). Per-row independent =
    // exact. Recursion is 1-level (sub-calls have M*N <= cap → no re-chunk).
    // Default off (RPU_ALLREDUCE_CHUNK_V2) → other models fall straight through.
    {
        const uint64_t kV2Cap = static_cast<uint64_t>(allreduce_v2_cap());
        const bool chunk_shape =
            M > 1 && N > 0 && (N % 16) == 0 &&
            static_cast<uint64_t>(N) <= kV2Cap &&
            static_cast<uint64_t>(M) >
                kV2Cap / static_cast<uint64_t>(N);
        const bool chunk_route =
            residual_ddr == nullptr && chunk_enabled &&
            twostage_enabled &&
            input_num_cores == output_num_cores && output_num_cores == 8 &&
            chunk_shape;
        TORCH_CHECK(
            !register_census_active || !chunk_route || force_chunk_v2,
            "typed DDR-register census requires an explicit force_chunk_v2 "
            "for chunked two-stage all-reduce");
        if (chunk_route) {
            TORCH_CHECK(
                valid_spm_byte_count &&
                    resolves_to_current_core0_spm(input_spm) &&
                    resolves_to_current_core0_spm(residual_spm) &&
                    resolves_to_current_core0_spm(output_spm),
                "chunked two-stage all-reduce requires complete input, "
                "residual, and output ranges in the current allocator's "
                "core-0 SPM with uint32-safe byte geometry");
            int64_t cap_rows = static_cast<int64_t>(
                kV2Cap / static_cast<uint64_t>(N));
            if (allreduce_chunk_balanced_enabled()) {
                const int64_t chunk_count = (M + cap_rows - 1) / cap_rows;
                cap_rows = (M + chunk_count - 1) / chunk_count;
            }
            TORCH_INTERNAL_ASSERT(cap_rows >= 1);
            const uint64_t max_row_offset = spm_bytes - row_bytes;
            TORCH_INTERNAL_ASSERT(max_row_offset <= UINT32_MAX);
            for (int64_t r = 0; r < M; r += cap_rows) {
                const int64_t m_sub = (cap_rows < (M - r)) ? cap_rows : (M - r);
                const uint64_t off_bytes =
                    static_cast<uint64_t>(r) * row_bytes;
                TORCH_INTERNAL_ASSERT(off_bytes <= max_row_offset);
                const uint32_t off = static_cast<uint32_t>(off_bytes);
                rpu_launch_all_reduce_sum_residual_impl(
                    input_spm + off,
                    residual_spm + off,
                    nullptr,
                    output_spm + off,
                    m_sub, N, input_num_cores, output_num_cores,
                    /*force_twostage=*/true,
                    /*force_chunk_v2=*/false,
                    /*force_non_v2=*/false);
            }
            return;
        }
    }

    // -------------------------------------------------------------------------
    // Two-stage path (guarded): scatter-reduce-residual + all-gather.
    // The residual may be an absolute
    // core-0 SPM address or a stable DDR pointer; scatter param[5] selects the
    // memory kind and params[0/1] carry either a local SPM offset or DDR>>8.
    //
    // input_spm / residual_spm / output_spm are ABSOLUTE core-0 SPM addresses
    // when used. Convert SPM operands to addr(0,0)-relative local offsets.
    // -------------------------------------------------------------------------
    {
        const int64_t MN = M * N;
        // v2 covers SPM residuals up to the cap (RPU_ALLREDUCE_V2_CAP, default
        // 409600 elements) with M*N %16==0; over the cap → non-v2. DDR residuals
        // stay on the non-v2 path required for DDR residuals.
        const bool use_v2 =
            residual_ddr == nullptr
            && !force_non_v2
            && MN <= allreduce_v2_cap();
        const bool residual_operands_distinct =
            residual_ddr != nullptr
                || (input_spm != residual_spm
                    && residual_spm != output_spm);
        const bool twostage_ok =
            twostage_enabled &&
            input_num_cores == output_num_cores && output_num_cores == 8 &&
            M > 0 && N > 0 && (MN % 16) == 0 &&
            input_spm != output_spm && residual_operands_distinct;
        if (twostage_ok) {
            if (register_census_active) {
                TORCH_CHECK(
                    use_v2,
                    "typed DDR-register census admits only the SPM-residual "
                    "two-stage v2 scatter path");
            }
            const KernelId scatter_kernel_id = use_v2
                ? KernelId::LLM_REDUCE_SUM_SCATTER_RESIDUAL_V2
                : KernelId::LLM_REDUCE_SUM_SCATTER_RESIDUAL;
            if (register_census_active) {
                RpuKernelGraph::active().stage_kernel_no_ddr(
                    scatter_kernel_id,
                    GraphKernelNoDdrProof::AllReduceSpmResidualV2);
            }

            // ---- absolute → local SPM offsets (launch convention) ----
            const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
            const uint32_t x_base = input_spm - spm_base0;     // partials base
            const uint32_t y_base = output_spm - spm_base0;    // scatter-out / gather dst
            uint32_t res_base = residual_ddr != nullptr
                ? 0 : residual_spm - spm_base0;
            if (residual_ddr != nullptr) {
                const uint64_t residual_dev_addr =
                    RpuGetDevAddr(const_cast<c10::Half*>(residual_ddr));
                TORCH_CHECK(
                    (residual_dev_addr & 0xFFu) == 0,
                    "DDR residual must be 256-byte aligned, got 0x",
                    std::hex, residual_dev_addr, std::dec);
                TORCH_CHECK(
                    (residual_dev_addr >> 8) <= UINT32_MAX,
                    "DDR residual address exceeds kernel encoding: 0x",
                    std::hex, residual_dev_addr, std::dec);
                // `llm_reduce_sum_scatter_residual{,_v2}` uses 256-byte
                // units for DDR operands and raw local offsets for SPM.
                res_base = static_cast<uint32_t>(residual_dev_addr >> 8);
            }

            constexpr uint32_t kDwidth = 2;            // fp16
            constexpr uint32_t kSpmBank = 32;          // spm_bank_size
            constexpr uint32_t kWarpSize = 16;         // hardware warp_size
            constexpr uint32_t kWarpNum = 8;           // hardware warp_num
            constexpr uint32_t kLinePerBlock = 7;      // line_per_block
            const int core_num = 8;

            const uint32_t mn_bytes = (uint32_t)MN * kDwidth;  // x_bytes = y_bytes = res_bytes
            const uint32_t line_byte_size = kWarpNum * kWarpSize * 16 * kDwidth;
            const uint32_t block_byte_size = line_byte_size * kLinePerBlock;

            // chunk split over the FLAT M*N*dwidth byte span, core_num chunks.
            const uint32_t avg_x_bytes =
                ((mn_bytes / (uint32_t)core_num) + (kSpmBank - 1)) & ~(kSpmBank - 1);
            std::vector<uint32_t> chunk_base(core_num), chunk_bytes(core_num);
            {
                uint32_t cb = 0;
                for (int j = 0; j < core_num; ++j) {
                    chunk_base[j] = cb;
                    chunk_bytes[j] = std::min(avg_x_bytes, mn_bytes - cb);
                    cb += chunk_bytes[j];
                }
            }
            auto blocks_of = [&](uint32_t bytes) -> uint16_t {
                return (uint16_t)((bytes + block_byte_size - 1) / block_byte_size);
            };

            if (allreduce_twostage_trace_enabled() &&
                allreduce_trace_first_time(
                    use_v2 ? "v2" : "non_v2",
                    residual_ddr ? "ddr" : "spm", output_num_cores,
                    M, N)) {
                std::fprintf(stderr,
                    "ALLREDUCE_TWOSTAGE path=%s residual=%s M=%lld N=%lld cores=%d\n",
                    use_v2 ? "v2" : "non_v2",
                    residual_ddr ? "ddr" : "spm",
                    (long long)M, (long long)N, output_num_cores);
            }

            auto* wq = GET_QUEUE(8);
            wq->set_broadcast_mode(true);
            std::vector<uint8_t> cores;
            for (int i = 0; i < 8; ++i) cores.push_back((uint8_t)i);

            // ----- kernel 0: reduce_sum_scatter(_v2) (residual folded) -----
            Kernel_t* k0 = GET_KERNEL(scatter_kernel_id);
            TORCH_CHECK(k0 != nullptr,
                        "Failed to get two-stage scatter kernel (v2=", use_v2, ")");
            k0->reset_regs();
            k0->set_regs(0, (uint16_t)(res_base & 0xFFFF));            // [0/1] SPM local offset or DDR>>8
            k0->set_regs(1, (uint16_t)((res_base >> 16) & 0xFFFF));
            k0->set_regs(2, (uint16_t)core_num);                      // [2] core_num
            k0->set_regs(3, (uint16_t)line_byte_size);                // [3] line_byte_size
            k0->set_regs(4, (uint16_t)block_byte_size);               // [4] block_byte_size
            k0->set_regs(5, (uint16_t)(residual_ddr != nullptr));     // [5] residual memory kind
            k0->set_regs(64, (uint16_t)kWarpNum);                     // grid.x = warp_num
            k0->set_regs(65, (uint16_t)1);
            k0->set_regs(66, (uint16_t)1);
            for (int j = 0; j < core_num; ++j) {
                const uint32_t S = 4096;
                k0->set_regs(S + j,      (uint16_t)(chunk_bytes[j] & 0xFFFF));          // [j] chunk_byte lo
                k0->set_regs(S + 8 + j,  (uint16_t)((chunk_bytes[j] >> 16) & 0xFFFF));  // [8+j] chunk_byte hi
                k0->set_regs(S + 16 + j, blocks_of(chunk_bytes[j]));                    // [16+j] blocks
                const uint32_t pj = SPM_ALLOC.addr(j, x_base);                          // [24+j]/[32+j] partial_j addr
                k0->set_regs(S + 24 + j, (uint16_t)(pj & 0xFFFF));
                k0->set_regs(S + 32 + j, (uint16_t)((pj >> 16) & 0xFFFF));
                const uint32_t sj = SPM_ALLOC.addr(j, y_base);                          // [40+j]/[48+j] scatter-out_j addr
                k0->set_regs(S + 40 + j, (uint16_t)(sj & 0xFFFF));
                k0->set_regs(S + 48 + j, (uint16_t)((sj >> 16) & 0xFFFF));
                k0->set_regs(S + 56 + j, (uint16_t)(chunk_base[j] & 0xFFFF));           // [56+j]/[64+j] chunk_base byte-offset
                k0->set_regs(S + 64 + j, (uint16_t)((chunk_base[j] >> 16) & 0xFFFF));
            }
            wq->enqueu_kernel(*k0, {(uint16_t)kWarpNum, 1, 1}, cores);

            // ----- kernel 1: all_gather_multi_core -----
            if (register_census_active) {
                RpuKernelGraph::active().stage_kernel_no_ddr(
                    KernelId::ALL_GATHER_MULTI_CORE,
                    GraphKernelNoDdrProof::AllGatherSpm);
            }
            Kernel_t* k1 = GET_KERNEL(KernelId::ALL_GATHER_MULTI_CORE);
            TORCH_CHECK(k1 != nullptr, "Failed to get ALL_GATHER_MULTI_CORE kernel");
            k1->reset_regs();
            k1->set_regs(0, (uint16_t)core_num);                     // [0] core_num
            k1->set_regs(1, (uint16_t)1);                            // [1] = 1
            k1->set_regs(2, (uint16_t)(mn_bytes & 0xFFFF));          // [2/3] y_bytes
            k1->set_regs(3, (uint16_t)((mn_bytes >> 16) & 0xFFFF));
            k1->set_regs(4, (uint16_t)line_byte_size);               // [4] line_byte_size
            k1->set_regs(5, (uint16_t)block_byte_size);              // [5] block_byte_size
            k1->set_regs(64, (uint16_t)8);                           // grid.x = 8
            k1->set_regs(65, (uint16_t)1);
            k1->set_regs(66, (uint16_t)1);
            for (int j = 0; j < core_num; ++j) {
                const uint32_t S = 4096;
                k1->set_regs(S + j,      (uint16_t)(chunk_bytes[j] & 0xFFFF));          // [j]
                k1->set_regs(S + 8 + j,  (uint16_t)((chunk_bytes[j] >> 16) & 0xFFFF));  // [8+j]
                k1->set_regs(S + 16 + j, blocks_of(chunk_bytes[j]));                    // [16+j] blocks
                k1->set_regs(S + 24 + j, (uint16_t)(chunk_bytes[j] & 0xFFFF));          // [24+j]
                k1->set_regs(S + 32 + j, (uint16_t)((chunk_bytes[j] >> 16) & 0xFFFF));  // [32+j]
                // [40+j]/[48+j] src addr = y_base + chunk_base[j] + j*off (spm_base stripped)
                const uint32_t srcj = SPM_ALLOC.addr(j, y_base + chunk_base[j]) - spm_base0;
                k1->set_regs(S + 40 + j, (uint16_t)(srcj & 0xFFFF));
                k1->set_regs(S + 48 + j, (uint16_t)((srcj >> 16) & 0xFFFF));
                // [56+j*16+k]/[64+j*16+k] dst table = y_base + chunk_base[j] + k*off (spm_base stripped)
                for (int kk = 0; kk < 8; ++kk) {
                    const uint32_t dstk = SPM_ALLOC.addr(kk, y_base + chunk_base[j]) - spm_base0;
                    k1->set_regs(S + 56 + j * 16 + kk, (uint16_t)(dstk & 0xFFFF));
                    k1->set_regs(S + 64 + j * 16 + kk, (uint16_t)((dstk >> 16) & 0xFFFF));
                }
            }
            wq->enqueu_kernel(*k1, {(uint16_t)8, 1, 1}, cores);
            return;
        }

        if (allreduce_twostage_trace_enabled() &&
            allreduce_trace_first_time(
                "single", residual_ddr ? "ddr" : "spm",
                output_num_cores, M, N)) {
            const char* reason =
                !twostage_enabled                    ? "disabled" :
                (input_num_cores != output_num_cores ||
                 output_num_cores != 8)              ? "not_8core" :
                (M <= 0 || N <= 0)                   ? "empty" :
                ((MN % 16) != 0)                     ? "mn_mod16" :
                                                       "spm_overlap";
            std::fprintf(stderr,
                "ALLREDUCE_TWOSTAGE path=single reason=%s M=%lld N=%lld cores=%d\n",
                reason, (long long)M, (long long)N, output_num_cores);
        }
    }

    TORCH_CHECK(
        residual_ddr == nullptr,
        "DDR residual requires the 8-core two-stage all-reduce path "
        "(RPU_ALLREDUCE_TWOSTAGE=1, distinct input/output, M*N % 16 == 0)");

    // The public launch ABI fuses reduce_sum with the residual add.
    // Parameters:
    //   param0: elements_cnt (elements per row = N)
    //   param2: is_test = 0
    //   param3: store_allcore_val = 1 (enable multi-core output)
    //   param4/5: residual address (core 0's SPM base)
    //   param6/7: input address (core 0's SPM base)
    //   param8/9: output address (core 0's SPM base)
    //   param21: input_core_num (read from input_num_cores)
    //   param22: output_core_num (broadcast to output_num_cores)

    int64_t blk_cnt = M;        // grid.dim.x = number of rows
    int64_t elements_cnt = N;   // elements per row

    int input_core_num = input_num_cores;
    int output_core_num = output_num_cores;
    int is_test = 0;
    int store_allcore_val = 1;

    auto setup_regs = [=](Kernel_t* kernel) {
        kernel->set_regs(0, (uint16_t)(elements_cnt));                    // param0: elements per row
        kernel->set_regs(2, (uint16_t)(is_test));                         // param2: is_test = 0
        kernel->set_regs(3, (uint16_t)(store_allcore_val));               // param3: store_allcore_val = 1
        kernel->set_regs(4, (uint16_t)(residual_spm & 0xFFFF));           // param4: residual addr low
        kernel->set_regs(5, (uint16_t)((residual_spm >> 16) & 0xFFFF));   // param5: residual addr high
        kernel->set_regs(6, (uint16_t)(input_spm & 0xFFFF));              // param6: input addr low
        kernel->set_regs(7, (uint16_t)((input_spm >> 16) & 0xFFFF));      // param7: input addr high
        kernel->set_regs(8, (uint16_t)(output_spm & 0xFFFF));             // param8: output addr low
        kernel->set_regs(9, (uint16_t)((output_spm >> 16) & 0xFFFF));     // param9: output addr high
        kernel->set_regs(21, (uint16_t)(input_core_num));                 // param21: input_core_num
        kernel->set_regs(22, (uint16_t)(output_core_num));                // param22: output_core_num
    };

    if (register_census_active) {
        TORCH_CHECK(
            residual_ddr == nullptr &&
                resolves_to_current_core0_spm(input_spm) &&
                resolves_to_current_core0_spm(residual_spm) &&
                resolves_to_current_core0_spm(output_spm),
            "typed DDR-register census admits the legacy all-reduce only "
            "when input, residual, and output resolve to the current "
            "allocator's core-0 SPM and no DDR residual is present");
        RpuKernelGraph::active().stage_kernel_no_ddr(
            KernelId::LLM_ALL_REDUCE_SUM_RESIDUAL,
            GraphKernelNoDdrProof::LegacyAllReduceSpmResidual);
    }
    Kernel_t* kernel = GET_KERNEL(KernelId::LLM_ALL_REDUCE_SUM_RESIDUAL);
    TORCH_CHECK(kernel != nullptr, "Failed to get LLM_ALL_REDUCE_SUM_RESIDUAL kernel");
    kernel->reset_regs();
    setup_regs(kernel);

    // Use max of input/output cores since kernel reads from input cores and writes to output cores.
    int max_cores = std::max(input_num_cores, output_num_cores);
    auto* wq = GET_QUEUE(max_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < max_cores; ++i) core_list.push_back(i);
    wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, 1, 1}, core_list);
}

void rpu_launch_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores)
{
    rpu_launch_all_reduce_sum_residual_impl(
        input_spm, residual_spm, nullptr, output_spm,
        M, N, input_num_cores, output_num_cores,
        /*force_twostage=*/false,
        /*force_chunk_v2=*/false,
        /*force_non_v2=*/false);
}

void rpu_launch_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores,
    bool force_twostage)
{
    rpu_launch_all_reduce_sum_residual_impl(
        input_spm, residual_spm, nullptr, output_spm,
        M, N, input_num_cores, output_num_cores,
        force_twostage,
        /*force_chunk_v2=*/false,
        /*force_non_v2=*/false);
}

void rpu_launch_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores,
    bool force_twostage,
    bool force_chunk_v2)
{
    rpu_launch_all_reduce_sum_residual_impl(
        input_spm, residual_spm, nullptr, output_spm,
        M, N, input_num_cores, output_num_cores,
        force_twostage, force_chunk_v2,
        /*force_non_v2=*/false);
}

void rpu_launch_all_reduce_sum_residual_non_v2_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores)
{
    rpu_launch_all_reduce_sum_residual_impl(
        input_spm, residual_spm, nullptr, output_spm,
        M, N, input_num_cores, output_num_cores,
        /*force_twostage=*/false,
        /*force_chunk_v2=*/false,
        /*force_non_v2=*/true);
}

void rpu_launch_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    const v3::SpmFmbPostFnYieldTarget& output,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores) {
    TORCH_CHECK(
        !allreduce_twostage_enabled() &&
            !allreduce_chunk_v2_enabled(),
        "typed post_fn yield all-reduce initially requires the canonical "
        "single-stage terminal writer");
    const uint32_t output_spm =
        v3::SpmFmbYieldTargetLauncherAccess::
            resolve_and_stage_all_reduce(
                output, M, N, input_num_cores, output_num_cores);
    try {
        rpu_launch_all_reduce_sum_residual_impl(
            input_spm, residual_spm, nullptr, output_spm,
            M, N, input_num_cores, output_num_cores,
            /*force_twostage=*/false,
            /*force_chunk_v2=*/false,
            /*force_non_v2=*/false);
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
    int output_num_cores)
{
    TORCH_CHECK(residual_ddr != nullptr,
                "all-reduce DDR residual pointer must not be null");
    rpu_launch_all_reduce_sum_residual_impl(
        input_spm, 0, residual_ddr, output_spm,
        M, N, input_num_cores, output_num_cores,
        /*force_twostage=*/false,
        /*force_chunk_v2=*/false,
        /*force_non_v2=*/false);
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
        ((M == 576 && N == 1280) || (M == 320 && N == 2048)) &&
            input_num_cores == kRingCoreNumMax &&
            output_num_cores == kRingCoreNumMax,
        "compact-residual ring is validated only for Wall Vision M576N1280 "
        "or Wall Text M320N2048 on 8 cores; got M=", M, " N=", N,
        " input_cores=", input_num_cores,
        " output_cores=", output_num_cores);
    launch_ring_all_reduce_sum_residual(
        input_spm, residual_local_spm, nullptr, output_spm,
        M, N, input_num_cores, output_num_cores,
        AllreduceRingMode::kPaced, /*require_kernel=*/true,
        /*residual_local_shard=*/true);
}

namespace {

void rpu_launch_wall_oss_mixed_ring_allreduce(
    uint32_t input_spm,
    uint32_t residual_spm,
    const c10::Half* residual_ddr,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    bool partial_is_bf16,
    bool residual_local_shard)
{
    TORCH_CHECK(
        ((M == 288 || M == 576) && N == 1280) || (M == 320 && N == 2048)
            || (M == 336 && N == 2048),
        "Wall-OSS mixed ring all-reduce supports only Vision (288/576,1280) or "
        "RTC prefill (320/336,2048); got M=",
        M, " N=", N);
    TORCH_CHECK(
        !residual_local_shard ||
            (!partial_is_bf16 && residual_ddr == nullptr &&
             ((M == 576 && N == 1280) ||
              (M == 320 && N == 2048))),
        "Wall-OSS mixed compact-residual ring is validated only for the "
        "FP16-partial/BF16-residual M576N1280 Vision or M320N2048 Text "
        "projection");
    TORCH_CHECK(
        input_spm != output_spm
            && (residual_ddr != nullptr
                || (input_spm != residual_spm
                    && residual_spm != output_spm)),
        "Wall-OSS mixed ring all-reduce requires out-of-place operands");
    if (residual_local_shard) {
        TORCH_CHECK(
            ring_all_reduce_sum_residual_admitted(
                input_spm, residual_spm, nullptr, output_spm,
                M, N, /*input_num_cores=*/8, /*output_num_cores=*/8,
                /*residual_local_shard=*/true),
            "Wall-OSS mixed compact-residual ring requires complete "
            "out-of-place input/output and one in-range local residual shard");
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
    TORCH_CHECK(avg_elem * kDwidth >= 11 * 1024,
                "Wall-OSS mixed ring shape unexpectedly selected nopace");

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

    const KernelId id = residual_local_shard
        ? KernelId::LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_SPM
        : (partial_is_bf16
            ? KernelId::LLM_ALL_REDUCE_BF16_PARTIAL_FP16_RESIDUAL
            : KernelId::LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL);
    TORCH_CHECK(
        KernelCache::instance().get(id) != nullptr,
        "Wall-OSS mixed ring all-reduce kernel is unavailable in the "
        "configured combined operator library");
    Kernel_t* kernel = GET_KERNEL(id);
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
    kernel->set_regs(64, static_cast<uint16_t>(kWarpNum));
    kernel->set_regs(65, 1);
    kernel->set_regs(66, 1);

    constexpr uint32_t S = 4096;
    for (uint32_t j = 0; j < kCoreNum; ++j) {
        // The ring operator uses SPM-relative addresses: preserve the per-core
        // offset while stripping core 0's absolute SPM base.
        const uint32_t in_base = SPM_ALLOC.addr(j, input_off) - spm_base0;
        const uint32_t out_base = SPM_ALLOC.addr(j, output_off) - spm_base0;
        const uint32_t chunk_byte_off = chunk_off[j] * kDwidth;
        kernel->set_regs(S + j,      static_cast<uint16_t>(in_base & 0xFFFF));
        kernel->set_regs(S + 8 + j,  static_cast<uint16_t>(in_base >> 16));
        kernel->set_regs(S + 16 + j, static_cast<uint16_t>(out_base & 0xFFFF));
        kernel->set_regs(S + 24 + j, static_cast<uint16_t>(out_base >> 16));
        kernel->set_regs(S + 32 + j, static_cast<uint16_t>(chunk_byte_off & 0xFFFF));
        kernel->set_regs(S + 40 + j, static_cast<uint16_t>(chunk_byte_off >> 16));
        kernel->set_regs(S + 48 + j, static_cast<uint16_t>(chunk_size[j] & 0xFFFF));
        kernel->set_regs(S + 56 + j, static_cast<uint16_t>(chunk_size[j] >> 16));
    }

    auto* queue = GET_QUEUE(kCoreNum);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel,
        {static_cast<uint16_t>(kWarpNum), static_cast<uint16_t>(1),
         static_cast<uint16_t>(1)},
        {0, 1, 2, 3, 4, 5, 6, 7});
}

}  // namespace

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
