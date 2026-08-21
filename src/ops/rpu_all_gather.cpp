// Multi-core all-gather over packed [n, chunk_elems] SPM shards.
// After launch every core holds the row-wise concatenation of all shards.
// Use after column-parallel computation; row-parallel partial sums use all-reduce.
// Two schedules cover small and large chunks. The 1024-byte selection threshold
// is part of the validated launch contract.
#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

namespace {
constexpr uint32_t kWarpSize = 16;
constexpr uint32_t kWarpNum  = 8;

// Launch geometry constants are byte counts and never scale by tensor dtype.
constexpr uint32_t kLineByteSize  = 4096;
constexpr uint32_t kBlockByteSize = kLineByteSize * 7;  // line_per_block = 7

constexpr uint32_t kLittleChunkMaxBytes = 1024;

static_assert(kLineByteSize == kWarpNum * kWarpSize * 32,
              "line_byte_size must equal grid.x*warp_size*32");

// RPU_ALL_GATHER_FORCE_MULTI_CORE forces the large-chunk schedule.
bool all_gather_force_multi_core() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_ALL_GATHER_FORCE_MULTI_CORE");
        cached = (e && (std::string(e) == "1" || std::string(e) == "true")) ? 1 : 0;
    }
    return cached != 0;
}
}  // namespace

void rpu_launch_all_gather_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    int64_t n,
    int64_t chunk_elems,
    int64_t dwidth,
    int num_cores)
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
    // The launch ABI counts chunk elements in u16 units.
    TORCH_CHECK((chunk_bytes % 2) == 0,
                "rpu_launch_all_gather_spm_kernel: chunk_bytes must be even, got ",
                chunk_bytes);

    const bool little = (chunk_bytes <= kLittleChunkMaxBytes) && !all_gather_force_multi_core();

    // Absolute addresses are converted to per-core offsets.
    const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
    const uint32_t in_off  = input_spm_addr  - spm_base0;
    const uint32_t out_off = output_spm_addr - spm_base0;

    Kernel_t* k = GET_KERNEL(little ? KernelId::ALL_GATHER_MULTI_CORE_LITTLE_CHUNK
                                    : KernelId::ALL_GATHER_MULTI_CORE);
    TORCH_CHECK(k != nullptr, "Failed to get all_gather kernel (little_chunk=", little, ")");
    k->reset_regs();

    // Parameters 6 and 7 describe 32-byte chunk granules and rows per pass.
    // Fill the complete launch ABI for both schedules.
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
        k->set_regs(S + j,      (uint16_t)(chunk_bytes & 0xFFFF));          // [j]/[8+j] chunk bytes
        k->set_regs(S + 8 + j,  (uint16_t)((chunk_bytes >> 16) & 0xFFFF));  //   (little reads LO only)
        k->set_regs(S + 16 + j, blocks);                                    // [16+j] block count
        k->set_regs(S + 24 + j, (uint16_t)(chunk_bytes & 0xFFFF));          // [24+j]/[32+j] src row
        k->set_regs(S + 32 + j, (uint16_t)((chunk_bytes >> 16) & 0xFFFF));  //   stride (packed)
        // [40+j]/[48+j] src addr = core j's shard base.
        const uint32_t srcj = SPM_ALLOC.addr(j, in_off) - spm_base0;
        k->set_regs(S + 40 + j, (uint16_t)(srcj & 0xFFFF));
        k->set_regs(S + 48 + j, (uint16_t)((srcj >> 16) & 0xFFFF));
        // [56+j*16+k]/[64+j*16+k] dst table: core j's column base, on every core k.
        // The destination table is bounded by num_cores.
        for (int kk = 0; kk < num_cores; ++kk) {
            const uint32_t dstk =
                SPM_ALLOC.addr(kk, out_off + (uint32_t)j * chunk_bytes) - spm_base0;
            k->set_regs(S + 56 + j * 16 + kk, (uint16_t)(dstk & 0xFFFF));
            k->set_regs(S + 64 + j * 16 + kk, (uint16_t)((dstk >> 16) & 0xFFFF));
        }
    }

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*k, {(uint16_t)kWarpNum, 1, 1}, cores);
}
