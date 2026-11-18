#include "rpu_pi05_producer_chain.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rhino_launch_buffer.h"
#include "rhino_launch_queue.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

using namespace ::rhino_lkn;

namespace {

// The whole-row-shard kernels sum each element's eight partials in the ring
// order the flat-shard kernel uses, starting at the owner of the element's
// flat shard (full_bytes / cores), so every FP16 rounding is the published
// one. That needs a whole-row shard to start inside flat shard `core` and end
// by the end of flat shard core + 1; SCM64/72 is its element count up to the
// end of flat shard `core`.
uint32_t whole_row_first_segment(
        uint32_t core, uint32_t cores, uint32_t full_bytes,
        uint32_t offset_bytes, uint32_t shard_bytes, const char* context) {
    const uint32_t flat_bytes = full_bytes / cores;
    TORCH_CHECK(
        full_bytes % (cores * 32) == 0 && offset_bytes >= core * flat_bytes &&
            offset_bytes < (core + 1) * flat_bytes &&
            offset_bytes + shard_bytes <= (core + 2) * flat_bytes,
        context, ": whole-row shard ", core,
        " must start in its flat shard and end by the next one");
    return (std::min(offset_bytes + shard_bytes, (core + 1) * flat_bytes) -
            offset_bytes) / sizeof(c10::Half);
}

}  // namespace

void rpu_launch_pi05_xor3_gated_residual_spm_kernel(
    uint32_t partial, uint32_t zero, uint32_t skip, uint32_t gate,
    uint32_t output, int64_t rows) {
    TORCH_CHECK(
        rows == 50 || rows == 10,
        "Pi XOR3+gated continuation is published for M50 and M10 only");
    constexpr uint32_t Cores = 8;
    const uint32_t FullBytes = uint32_t(rows) * 1024 * sizeof(c10::Half);
    constexpr uint32_t GateBytes = 1024 * sizeof(c10::Half);
    const uint32_t ShardElements = uint32_t(rows) * 1024 / Cores;
    const uint32_t ShardBytes = ShardElements * sizeof(c10::Half);
    constexpr uint32_t RoundElements = 63488;

    TORCH_CHECK(
        SPM_ALLOC.is_initialized(),
        "Pi XOR3+gated continuation requires initialized SPM");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint32_t>, 5> ranges{{
        {partial, FullBytes},
        {zero, FullBytes},
        {skip, FullBytes},
        {gate, GateBytes},
        {output, FullBytes},
    }};
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [address, bytes] = ranges[i];
        TORCH_CHECK(
            address >= base && (address & 255U) == 0 &&
                uint64_t(address) - base <=
                    uint64_t(SpmAllocator::SPM_USABLE) - bytes,
            "Pi XOR3+gated continuation requires complete 256-byte-aligned "
            "absolute SPM operands");
        for (size_t j = 0; j < i; ++j) {
            const uint64_t lhs = address;
            const uint64_t rhs = ranges[j].first;
            TORCH_CHECK(
                lhs + bytes <= rhs || rhs + ranges[j].second <= lhs,
                "Pi XOR3+gated continuation requires five mutually "
                "disjoint SPM operands");
        }
    }

    // At M10 the low-barrier sibling keeps this ABI, the SCM tables, the
    // atomics reset and the output bytes; an expansion REF without it keeps
    // XOR3. M50 has no such sibling: there the per-peer ring barriers pay.
    const KernelId id = rows == 50 ? KernelId::PI05_XOR3_GATED_RESIDUAL_M50N1024
        : KernelCache::instance().has_loaded(KernelId::PI05_RING_GATED_RESIDUAL_LB_M10N1024)
            ? KernelId::PI05_RING_GATED_RESIDUAL_LB_M10N1024
            : KernelId::PI05_XOR3_GATED_RESIDUAL_M10N1024;
    auto& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active()) {
        graph.stage_kernel_no_ddr(
            id, GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(
        kernel,
        "Pi XOR3+gated continuation expansion symbol is missing");
    kernel->reset_regs();
    const auto set = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(
            kernel->set_regs(index, value) == kKernelRegOk,
            "Pi XOR3+gated continuation rejected register ", index);
    };
    const auto pair = [&set](uint32_t index, uint32_t value) {
        set(index, uint16_t(value));
        set(index + 1, uint16_t(value >> 16));
    };

    // Frozen continuation ABI: preserve the original one-round equal-shard
    // XOR3 reduction, then apply the existing separately-rounded gate/add tail.
    set(0, Cores);
    set(1, 1);
    pair(2, RoundElements);
    pair(4, zero - base);
    set(6, 0);
    pair(8, skip);
    pair(10, gate);
    for (uint32_t index = 56; index <= 58; ++index) set(index, 0x3c00);
    set(59, 0);
    set(60, 0);
    set(61, 1);
    set(62, 0x0492);
    set(63, 0);
    set(64, Cores);
    set(65, 1);
    set(66, 1);
    for (uint32_t index = 128; index <= 255; ++index) set(index, 0);

    for (uint32_t core = 0; core < Cores; ++core) {
        const std::array<std::pair<uint32_t, uint32_t>, 4> fields{{
            {0, SPM_ALLOC.addr(core, partial - base) - base},
            {16, SPM_ALLOC.addr(core, output - base) - base},
            {32, core * ShardBytes},
            {48, ShardElements},
        }};
        for (const auto& [table, value] : fields) {
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + table + core, uint16_t(value),
                "Pi XOR3+gated continuation");
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + table + 8 + core,
                uint16_t(value >> 16),
                "Pi XOR3+gated continuation");
        }
    }

    auto* queue = GET_QUEUE(Cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {Cores, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_xor3_gated_adarms_spm_kernel(
    uint32_t partial, uint32_t zero, uint32_t skip, uint32_t gate,
    uint32_t raw, uint32_t norm, uint32_t scale, uint32_t shift,
    int64_t rows, double eps) {
    TORCH_CHECK(
        rows == 50,
        "Pi XOR3+gated+AdaRMS continuation is published for M50 only");
    constexpr uint32_t Cores = 8;
    constexpr uint32_t Hidden = 1024;
    constexpr uint32_t RowBytes = Hidden * sizeof(c10::Half);
    const uint32_t FullBytes = uint32_t(rows) * RowBytes;
    constexpr uint32_t RoundElements = 63488;

    TORCH_CHECK(
        SPM_ALLOC.is_initialized(),
        "Pi XOR3+gated+AdaRMS continuation requires initialized SPM");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint32_t>, 8> ranges{{
        {partial, FullBytes},
        {zero, FullBytes},
        {skip, FullBytes},
        {gate, RowBytes},
        {raw, FullBytes},
        {norm, FullBytes},
        {scale, RowBytes},
        {shift, RowBytes},
    }};
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [address, bytes] = ranges[i];
        TORCH_CHECK(
            address >= base && (address & 255U) == 0 &&
                uint64_t(address) - base <=
                    uint64_t(SpmAllocator::SPM_USABLE) - bytes,
            "Pi XOR3+gated+AdaRMS continuation requires complete "
            "256-byte-aligned absolute SPM operands");
        for (size_t j = 0; j < i; ++j) {
            const uint64_t lhs = address;
            const uint64_t rhs = ranges[j].first;
            TORCH_CHECK(
                lhs + bytes <= rhs || rhs + ranges[j].second <= lhs,
                "Pi XOR3+gated+AdaRMS continuation requires eight mutually "
                "disjoint SPM operands");
        }
    }

    // The two-peer sibling reads two peers per paced reduce step into the
    // same slots in the same order: same register image (p0 = Cores = 8) and
    // bytes, -107 cycles per launch on the board. The one-warp sibling of the
    // two-peer kernel reduces on warp 0 alone, unpaced in ring order, in two
    // VLD stages: the same slots, order, register image and bytes, a further
    // -568 cycles per launch.
    const auto& cache = KernelCache::instance();
    const KernelId id =
        cache.has_loaded(KernelId::PI05_XOR3_GATED_ADARMS_ONEWARP_M50N1024)
        ? KernelId::PI05_XOR3_GATED_ADARMS_ONEWARP_M50N1024
        : cache.has_loaded(KernelId::PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024)
        ? KernelId::PI05_XOR3_GATED_ADARMS_TWOPEER_M50N1024
        : KernelId::PI05_XOR3_GATED_ADARMS_M50N1024;
    auto& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active()) {
        graph.stage_kernel_no_ddr(
            id, GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(
        kernel,
        "Pi XOR3+gated+AdaRMS continuation expansion symbol is missing");
    kernel->reset_regs();
    const auto set = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(
            kernel->set_regs(index, value) == kKernelRegOk,
            "Pi XOR3+gated+AdaRMS continuation rejected register ", index);
    };
    const auto pair = [&set](uint32_t index, uint32_t value) {
        set(index, uint16_t(value));
        set(index + 1, uint16_t(value >> 16));
    };

    // The continuation's own frozen ABI, with the norm's operands appended:
    // the compact raw residual, the scale and shift rows, epsilon, 1/N, N and
    // the row stride the owner body walks its row with.
    set(0, Cores);
    set(1, 1);
    pair(2, RoundElements);
    pair(4, zero - base);
    set(6, 0);
    pair(8, skip);
    pair(10, gate);
    pair(12, raw);
    pair(14, scale);
    pair(16, shift);
    set(18, c10::Half(eps).x);
    set(19, c10::Half(1.0f / float(Hidden)).x);
    set(20, uint16_t(Hidden));
    set(21, uint16_t(RowBytes));
    for (uint32_t index = 56; index <= 58; ++index) set(index, 0x3c00);
    set(59, 0);
    set(60, 0);
    set(61, 1);
    set(62, 0x0492);
    set(63, 0);
    set(64, Cores);
    set(65, 1);
    set(66, 1);
    for (uint32_t index = 128; index <= 255; ++index) set(index, 0);

    // Whole-row shards: 7, 7, 6, 6, 6, 6, 6, 6 rows at M50. A row never
    // straddles two cores, so its norm has one owner, and no shard exceeds the
    // eight rows the eight warps of the owner body can take.
    uint32_t shard_offset_bytes = 0;
    for (uint32_t core = 0; core < Cores; ++core) {
        const uint32_t shard_rows = uint32_t(rows) / Cores +
            (core < uint32_t(rows) % Cores ? 1U : 0U);
        TORCH_CHECK(
            shard_rows >= 1 && shard_rows <= 8,
            "Pi XOR3+gated+AdaRMS continuation requires one to eight rows "
            "per core");
        const uint32_t first = whole_row_first_segment(
            core, Cores, FullBytes, shard_offset_bytes, shard_rows * RowBytes,
            "Pi XOR3+gated+AdaRMS continuation");
        // One warp loads 256 elements per VLM entry: 25 entries of each
        // 31-entry slot for the first segment, 6 for the rest of the shard.
        TORCH_CHECK(
            id != KernelId::PI05_XOR3_GATED_ADARMS_ONEWARP_M50N1024 ||
                (first <= 25 * 256 && shard_rows * Hidden - first <= 6 * 256),
            "Pi XOR3+gated+AdaRMS one-warp continuation: shard ", core,
            " does not fit one warp's VLM slots");
        const std::array<std::pair<uint32_t, uint32_t>, 5> fields{{
            {0, SPM_ALLOC.addr(core, partial - base) - base},
            {16, SPM_ALLOC.addr(core, norm - base) - base},
            {32, shard_offset_bytes},
            {48, shard_rows * Hidden},
            {64, first},
        }};
        for (const auto& [table, value] : fields) {
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + table + core, uint16_t(value),
                "Pi XOR3+gated+AdaRMS continuation");
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + table + 8 + core,
                uint16_t(value >> 16),
                "Pi XOR3+gated+AdaRMS continuation");
        }
        shard_offset_bytes += shard_rows * RowBytes;
    }
    TORCH_INTERNAL_ASSERT(shard_offset_bytes == FullBytes);

    auto* queue = GET_QUEUE(Cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {Cores, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_xor3_gated_rows_spm_kernel(
    uint32_t partial, uint32_t zero, uint32_t skip, uint32_t gate,
    uint32_t raw, int64_t rows) {
    TORCH_CHECK(
        rows == 50,
        "Pi XOR3+gated row-shard continuation is published for M50 only");
    constexpr uint32_t Cores = 8;
    constexpr uint32_t Hidden = 1024;
    constexpr uint32_t RowBytes = Hidden * sizeof(c10::Half);
    const uint32_t FullBytes = uint32_t(rows) * RowBytes;
    constexpr uint32_t RoundElements = 63488;

    TORCH_CHECK(
        SPM_ALLOC.is_initialized(),
        "Pi XOR3+gated row-shard continuation requires initialized SPM");
    const uint32_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint32_t>, 5> ranges{{
        {partial, FullBytes},
        {zero, FullBytes},
        {skip, FullBytes},
        {gate, RowBytes},
        {raw, FullBytes},
    }};
    for (size_t i = 0; i < ranges.size(); ++i) {
        const auto [address, bytes] = ranges[i];
        TORCH_CHECK(
            address >= base && (address & 255U) == 0 &&
                uint64_t(address) - base <=
                    uint64_t(SpmAllocator::SPM_USABLE) - bytes,
            "Pi XOR3+gated row-shard continuation requires complete "
            "256-byte-aligned absolute SPM operands");
        for (size_t j = 0; j < i; ++j) {
            const uint64_t lhs = address;
            const uint64_t rhs = ranges[j].first;
            TORCH_CHECK(
                lhs + bytes <= rhs || rhs + ranges[j].second <= lhs,
                "Pi XOR3+gated row-shard continuation requires five mutually "
                "disjoint SPM operands");
        }
    }

    constexpr KernelId id = KernelId::PI05_XOR3_GATED_ROWS_M50N1024;
    auto& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active()) {
        graph.stage_kernel_no_ddr(
            id, GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(
        kernel,
        "Pi XOR3+gated row-shard continuation expansion symbol is missing");
    kernel->reset_regs();
    const auto set = [kernel](uint32_t index, uint16_t value) {
        TORCH_CHECK(
            kernel->set_regs(index, value) == kKernelRegOk,
            "Pi XOR3+gated row-shard continuation rejected register ", index);
    };
    const auto pair = [&set](uint32_t index, uint32_t value) {
        set(index, uint16_t(value));
        set(index + 1, uint16_t(value >> 16));
    };

    // The continuation's own frozen ABI, plus the row geometry its owner body
    // walks a whole row with. It has no norm operands: registers 12 to 19 stay
    // zero and the raw residual is published through the all-gather, the way
    // the flat-shard kernel publishes it.
    set(0, Cores);
    set(1, 1);
    pair(2, RoundElements);
    pair(4, zero - base);
    set(6, 0);
    pair(8, skip);
    pair(10, gate);
    set(20, uint16_t(Hidden));
    set(21, uint16_t(RowBytes));
    for (uint32_t index = 56; index <= 58; ++index) set(index, 0x3c00);
    set(59, 0);
    set(60, 0);
    set(61, 1);
    set(62, 0x0492);
    set(63, 0);
    set(64, Cores);
    set(65, 1);
    set(66, 1);
    for (uint32_t index = 128; index <= 255; ++index) set(index, 0);

    // Whole-row shards: 7, 7, 6, 6, 6, 6, 6, 6 rows at M50, the same layout
    // the fused kernel needs, so a chain can mix the two. The output doubles as
    // the reduce staging area, so its peer bases go in the gather table.
    uint32_t shard_offset_bytes = 0;
    for (uint32_t core = 0; core < Cores; ++core) {
        const uint32_t shard_rows = uint32_t(rows) / Cores +
            (core < uint32_t(rows) % Cores ? 1U : 0U);
        TORCH_CHECK(
            shard_rows >= 1 && shard_rows <= 8,
            "Pi XOR3+gated row-shard continuation requires one to eight rows "
            "per core");
        const std::array<std::pair<uint32_t, uint32_t>, 5> fields{{
            {0, SPM_ALLOC.addr(core, partial - base) - base},
            {16, SPM_ALLOC.addr(core, raw - base) - base},
            {32, shard_offset_bytes},
            {48, shard_rows * Hidden},
            {64, whole_row_first_segment(
                      core, Cores, FullBytes, shard_offset_bytes,
                      shard_rows * RowBytes, "Pi XOR3+gated row-shard continuation")},
        }};
        for (const auto& [table, value] : fields) {
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + table + core, uint16_t(value),
                "Pi XOR3+gated row-shard continuation");
            rpu_set_legacy_scm_u16_checked(
                kernel, 4096 + table + 8 + core,
                uint16_t(value >> 16),
                "Pi XOR3+gated row-shard continuation");
        }
        shard_offset_bytes += shard_rows * RowBytes;
    }
    TORCH_INTERNAL_ASSERT(shard_offset_bytes == FullBytes);

    auto* queue = GET_QUEUE(Cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(
        *kernel, {Cores, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}
