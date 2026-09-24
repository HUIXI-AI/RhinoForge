// rpu_all_gather.cpp — multi-core all-gather (axis=1 column concat).
//
// Semantics: core j holds a packed [n, chunk_elems] shard (its own columns);
// afterwards EVERY core holds the full [n, chunk_elems*num_cores], with core j's
// shard occupying columns [j*chunk_elems, (j+1)*chunk_elems) of every row.
//
// Use after a COLUMN-parallel matmul (partition=1): per-core outputs are
// different columns of one answer, so they are concatenated. Contrast
// all_reduce_sum_residual, which pairs with ROW-parallel (partition=0): there
// per-core outputs are full-width PARTIAL SUMS that must be added. Picking the
// wrong one is silently wrong, not a crash.
//
// For `out = concat(shards) + residual`, follow this with
// rpu_launch_eltwise_binary_spm_kernel(out, residual, out, n*chunk*tp,
//                                      ValuOpType::ADD, 1.0, num_cores).
//
// little_chunk supports chunk <= 1024B. Launch geometry is expressed in bytes
// and does not scale with tensor dtype.
// =============================================================================
// Addressing is deliberately GLOBAL here
//
// The usual multi-core rule ("emitters must use per-core-local SPM addresses")
// does NOT apply: the scm tables are indexed BY CORE and the kernel reaches
// across cores, so every core gets the same table of per-core addresses. Do not
// "fix" this to local addressing.
//
// The operator provides the required cross-core synchronization.
// Broadcast mode is configured through the queue API.
// Only uniform, packed source shards are exposed by this wrapper.

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

namespace {
constexpr uint32_t kWarpSize = 16;  // hardware warp_size
constexpr uint32_t kWarpNum  = 8;   // hardware warp_num; grid.x — always 8, see below

// Launch geometry is expressed in bytes and does not scale with dtype.
constexpr uint32_t kLineByteSize  = 4096;
constexpr uint32_t kBlockByteSize = kLineByteSize * 7;  // line_per_block = 7

constexpr uint32_t kLittleChunkMaxBytes = 1024;

static_assert(kLineByteSize == kWarpNum * kWarpSize * 32,
              "line_byte_size must equal grid.x*warp_size*32");

}  // namespace

void rpu_launch_all_gather_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    int64_t n,
    int64_t chunk_elems,
    int64_t dwidth,
    int num_cores,
    RpuAllGatherSchedule schedule)
{
    TORCH_CHECK(n > 0 && chunk_elems > 0,
                "rpu_launch_all_gather_spm_kernel: n/chunk_elems must be positive (got ",
                n, "/", chunk_elems, ")");
    TORCH_CHECK(num_cores > 0 && num_cores <= 8,
                "rpu_launch_all_gather_spm_kernel: num_cores must be in [1, 8], got ",
                num_cores);
    TORCH_CHECK(n <= 65535,
                "rpu_launch_all_gather_spm_kernel: n must fit in u16 (reg[1]), got ", n);
    TORCH_CHECK(dwidth == 2 || dwidth == 4,
                "rpu_launch_all_gather_spm_kernel: dwidth must be 2 (fp16) or 4 "
                "(int32/fp32), got ", dwidth);

    const uint32_t chunk_bytes   = (uint32_t)(chunk_elems * dwidth);   // == src row stride
    const uint32_t dst_row_bytes = chunk_bytes * (uint32_t)num_cores;  // dst row stride
    // The byte count must be even.
    TORCH_CHECK((chunk_bytes % 2) == 0,
                "rpu_launch_all_gather_spm_kernel: chunk_bytes must be even, got ",
                chunk_bytes);

    TORCH_CHECK(
        schedule == RpuAllGatherSchedule::LITTLE_CHUNK ||
            schedule == RpuAllGatherSchedule::MULTI_CORE,
        "rpu_launch_all_gather_spm_kernel: invalid schedule ",
        static_cast<int64_t>(schedule));
    const bool little = schedule == RpuAllGatherSchedule::LITTLE_CHUNK;
    TORCH_CHECK(
        !little || chunk_bytes <= kLittleChunkMaxBytes,
        "rpu_launch_all_gather_spm_kernel: little-chunk schedule exceeds the ",
        kLittleChunkMaxBytes, "B safe threshold (got ", chunk_bytes, "B)");

    // Absolute addresses to local offsets.
    const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
    const uint32_t in_off  = input_spm_addr  - spm_base0;
    const uint32_t out_off = output_spm_addr - spm_base0;

    Kernel_t* k = GET_KERNEL(little ? KernelId::ALL_GATHER_MULTI_CORE_LITTLE_CHUNK
                                    : KernelId::ALL_GATHER_MULTI_CORE);
    TORCH_CHECK(k != nullptr, "Failed to get all_gather kernel (little_chunk=", little, ")");
    k->reset_regs();

    // param6 is the widest chunk in 32-byte units; param7 is the row count per pass.
    // Fill both fields for either variant; little_chunk requires param7 >= 1.
    const uint32_t max_chunk_size_v16 = (chunk_bytes + 31) / 32;
    const uint32_t n_per_thd = 64u / max_chunk_size_v16;  // may be 0 for multi_core shapes
    TORCH_CHECK(!little || n_per_thd >= 1,
                "rpu_launch_all_gather_spm_kernel: little_chunk selected but "
                "n_per_thd == 0 (chunk_bytes=", chunk_bytes, " → param6=",
                max_chunk_size_v16, " > 64). Its row loop would get step 0 and HANG. "
                "The <=1024B selection threshold should make this unreachable.");

    k->set_regs(0, (uint16_t)num_cores);                        // [0] core_num
    k->set_regs(1, (uint16_t)n);                                // [1] row count
    k->set_regs(2, (uint16_t)(dst_row_bytes & 0xFFFF));         // [2]/[3] dst row stride (u32)
    k->set_regs(3, (uint16_t)((dst_row_bytes >> 16) & 0xFFFF));
    k->set_regs(4, (uint16_t)kLineByteSize);                    // [4] VLM line granularity
    k->set_regs(5, (uint16_t)kBlockByteSize);                   // [5] VLM block granularity
    k->set_regs(6, (uint16_t)max_chunk_size_v16);               // [6] little_chunk only
    k->set_regs(7, (uint16_t)n_per_thd);                        // [7] little_chunk only
    k->set_regs(64, (uint16_t)kWarpNum);                        // [64..66] grid dims (parity;
    k->set_regs(65, (uint16_t)1);                               //   the functional channel is
    k->set_regs(66, (uint16_t)1);                               //   enqueu_kernel's grid arg)

    // ---- per-core tables ----
    const uint16_t blocks =
        (uint16_t)((chunk_bytes + kBlockByteSize - 1) / kBlockByteSize);
    for (int j = 0; j < num_cores; ++j) {
        const uint32_t S = 4096;
        rpu_set_legacy_scm_u16_checked(
            k, S + j, (uint16_t)(chunk_bytes & 0xFFFF), "all-gather");      // [j]/[8+j] chunk bytes
        rpu_set_legacy_scm_u16_checked(
            k, S + 8 + j, (uint16_t)((chunk_bytes >> 16) & 0xFFFF),
            "all-gather");                                                  //   (little reads LO only)
        rpu_set_legacy_scm_u16_checked(
            k, S + 16 + j, blocks, "all-gather");                           // [16+j] block count
        rpu_set_legacy_scm_u16_checked(
            k, S + 24 + j, (uint16_t)(chunk_bytes & 0xFFFF),
            "all-gather");                                                  // [24+j]/[32+j] src row
        rpu_set_legacy_scm_u16_checked(
            k, S + 32 + j, (uint16_t)((chunk_bytes >> 16) & 0xFFFF),
            "all-gather");                                                  //   stride (packed)
        // [40+j]/[48+j] src addr = core j's shard base.
        const uint32_t srcj = SPM_ALLOC.addr(j, in_off) - spm_base0;
        rpu_set_legacy_scm_u16_checked(
            k, S + 40 + j, (uint16_t)(srcj & 0xFFFF), "all-gather");
        rpu_set_legacy_scm_u16_checked(
            k, S + 48 + j, (uint16_t)((srcj >> 16) & 0xFFFF),
            "all-gather");
        // [56+j*16+k]/[64+j*16+k] dst table: core j's column base, on every core k.
        for (int kk = 0; kk < num_cores; ++kk) {
            const uint32_t dstk =
                SPM_ALLOC.addr(kk, out_off + (uint32_t)j * chunk_bytes) - spm_base0;
            rpu_set_legacy_scm_u16_checked(
                k, S + 56 + j * 16 + kk, (uint16_t)(dstk & 0xFFFF),
                "all-gather");
            rpu_set_legacy_scm_u16_checked(
                k, S + 64 + j * 16 + kk,
                (uint16_t)((dstk >> 16) & 0xFFFF), "all-gather");
        }
    }

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*k, {(uint16_t)kWarpNum, 1, 1}, cores);
}

void rpu_launch_spm_local_shard_copy_kernel(
    uint32_t spm_src_unified,
    int64_t elements_per_core,
    int64_t core_stride_bytes,
    uint32_t spm_dst_unified,
    int num_cores)
{
    constexpr const char* caller = "spm_local_shard_copy_kernel";
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                caller, ": num_cores must be in [1,8], got ", num_cores);
    TORCH_CHECK(elements_per_core > 0 && core_stride_bytes >= 0,
                caller, ": element count must be positive and stride non-negative");
    const size_t bytes =
        static_cast<size_t>(elements_per_core) * sizeof(c10::Half);
    TORCH_CHECK((bytes % 16) == 0 && (core_stride_bytes % 16) == 0,
                caller, ": byte count and stride must be 16-byte aligned");
    TORCH_CHECK(core_stride_bytes == 0
                    || static_cast<uint64_t>(core_stride_bytes) >= bytes,
                caller, ": non-zero stride must cover one shard");

    const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
    TORCH_CHECK(spm_src_unified >= spm_base && spm_dst_unified >= spm_base,
                caller, ": source and destination must be core-0 unified SPM addresses");
    const uint64_t src_off = spm_src_unified - spm_base;
    const uint64_t dst_off = spm_dst_unified - spm_base;
    const uint64_t last_src_end = src_off
        + static_cast<uint64_t>(num_cores - 1) * core_stride_bytes + bytes;
    TORCH_CHECK(last_src_end <= SpmAllocator::SPM_USABLE
                    && dst_off + bytes <= SpmAllocator::SPM_USABLE,
                caller, ": source or destination exceeds per-core SPM");
    for (int c = 0; c < num_cores; ++c) {
        const uint64_t src_begin =
            src_off + static_cast<uint64_t>(c) * core_stride_bytes;
        TORCH_CHECK(src_begin + bytes <= dst_off
                        || dst_off + bytes <= src_begin,
                    caller, ": overlapping same-core copy is unsupported");
    }

    const uint32_t chunk_bytes = static_cast<uint32_t>(bytes);
    const uint32_t max_chunk_size_v16 = (chunk_bytes + 31) / 32;
    const uint16_t blocks = static_cast<uint16_t>(
        (chunk_bytes + kBlockByteSize - 1) / kBlockByteSize);

    auto& graph = RpuKernelGraph::active();
    graph.stage_kernel_no_ddr(
        KernelId::ALL_GATHER_MULTI_CORE,
        GraphKernelNoDdrProof::AllGatherSpm);
    Kernel_t* kernel = GET_KERNEL(KernelId::ALL_GATHER_MULTI_CORE);
    TORCH_CHECK(kernel != nullptr,
                caller, ": failed to get all_gather_multi_core kernel");
    kernel->reset_regs();

    // core_num=1 skips the ring. Broadcast still dispatches every physical
    // core, so each core performs only its diagonal local copy.
    kernel->set_regs(0, static_cast<uint16_t>(1));
    kernel->set_regs(1, static_cast<uint16_t>(1));
    kernel->set_regs(2, static_cast<uint16_t>(chunk_bytes & 0xFFFF));
    kernel->set_regs(3, static_cast<uint16_t>(chunk_bytes >> 16));
    kernel->set_regs(4, static_cast<uint16_t>(kLineByteSize));
    kernel->set_regs(5, static_cast<uint16_t>(kBlockByteSize));
    kernel->set_regs(6, static_cast<uint16_t>(max_chunk_size_v16));
    kernel->set_regs(7, static_cast<uint16_t>(0));
    kernel->set_regs(64, static_cast<uint16_t>(kWarpNum));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));

    const uint32_t spm_base0 = static_cast<uint32_t>(spm_base);
    constexpr uint32_t S = 4096;
    for (int j = 0; j < num_cores; ++j) {
        rpu_set_legacy_scm_u16_checked(
            kernel, S + j, static_cast<uint16_t>(chunk_bytes & 0xFFFF), caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 8 + j, static_cast<uint16_t>(chunk_bytes >> 16), caller);
        rpu_set_legacy_scm_u16_checked(kernel, S + 16 + j, blocks, caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 24 + j,
            static_cast<uint16_t>(chunk_bytes & 0xFFFF), caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 32 + j,
            static_cast<uint16_t>(chunk_bytes >> 16), caller);

        const uint32_t src = SPM_ALLOC.addr(
            j, static_cast<uint32_t>(
                src_off + static_cast<uint64_t>(j) * core_stride_bytes))
            - spm_base0;
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 40 + j, static_cast<uint16_t>(src & 0xFFFF), caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 48 + j, static_cast<uint16_t>(src >> 16), caller);

        const uint32_t dst =
            SPM_ALLOC.addr(j, static_cast<uint32_t>(dst_off)) - spm_base0;
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 56 + j * 16 + j,
            static_cast<uint16_t>(dst & 0xFFFF), caller);
        rpu_set_legacy_scm_u16_checked(
            kernel, S + 64 + j * 16 + j,
            static_cast<uint16_t>(dst >> 16), caller);
    }

    auto* queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) {
        cores.push_back(static_cast<uint8_t>(i));
    }
    queue->enqueu_kernel(
        *kernel, {static_cast<uint16_t>(kWarpNum), 1, 1}, cores);
}

// =============================================================================
// rpu_all_gather_test — torch op wrapper for isolated all_gather testing.
//
// shards: [num_cores, n, chunk_elems] fp16/int32 RPU DDR — shard j scattered to
// core j. Returns [n, chunk_elems*num_cores], read back from core 0 (every core
// should hold an identical copy).
// =============================================================================

at::Tensor rpu_all_gather_test(
    const at::Tensor& shards, int64_t num_cores, bool force_multi_core)
{
    RECORD_FUNCTION("rpu::all_gather_test", {});

    TORCH_CHECK(shards.dim() == 3,
                "all_gather_test: shards must be [num_cores, n, chunk], got dim=",
                shards.dim());
    TORCH_CHECK(shards.scalar_type() == at::kHalf || shards.scalar_type() == at::kInt,
                "all_gather_test: shards must be fp16 or int32, got ", shards.scalar_type());
    TORCH_CHECK(shards.is_contiguous(), "all_gather_test: shards must be contiguous");
    TORCH_CHECK(shards.device().type() == at::kPrivateUse1,
                "all_gather_test: shards must be on RPU device");
    TORCH_CHECK(shards.size(0) == num_cores,
                "all_gather_test: shards.size(0)=", shards.size(0),
                " must equal num_cores=", num_cores);

    const int64_t n           = shards.size(1);
    const int64_t chunk_elems = shards.size(2);
    const int64_t dwidth      = shards.element_size();
    const int nc = (int)num_cores;

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    const int64_t in_bytes  = n * chunk_elems * dwidth;              // per core
    const int64_t out_bytes = n * chunk_elems * num_cores * dwidth;  // per core (replicated)

    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{in_bytes, 1, 2},   // shard    (DMA-in → kernel-read)
        AR{out_bytes, 2, 3},  // gathered (kernel-write → DMA-out)
    });
    const uint32_t in_addr  = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t out_addr = SPM_ALLOC.addr(0, offsets[1]);

    // SCATTER, not broadcast: core j must get a DIFFERENT shard. Broadcasting
    // would hand every core the same data, making the gather trivially "pass"
    // while testing nothing.
    //
    // The DMA helpers are typed fp16, but they are byte movers; for int32 we
    // reinterpret the pointer and halve the element count. (reinterpret_cast
    // rather than Tensor::view(kHalf) — the latter would dispatch a view op on
    // the RPU device for no benefit.)
    const int64_t in_elems_h  = in_bytes / 2;
    const int64_t out_elems_h = out_bytes / 2;

    rpu_launch_ddr_scatter_spm_dma_immediate(
        reinterpret_cast<c10::Half*>(shards.data_ptr()),
        /*elements_per_core=*/in_elems_h,
        /*core_stride_bytes=*/in_bytes,
        in_addr, nc);

    rpu_launch_all_gather_spm_kernel(
        in_addr, out_addr, n, chunk_elems, dwidth, nc,
        rpu_resolve_all_gather_schedule(
            chunk_elems, dwidth, force_multi_core));

    auto output = at::empty({n, chunk_elems * num_cores}, shards.options());
    rpu_launch_spm_copy_ddr_dma_immediate(
        out_addr, reinterpret_cast<c10::Half*>(output.data_ptr()), out_elems_h);

    SPM_ALLOC.reset_temporary();
    return output;
}
