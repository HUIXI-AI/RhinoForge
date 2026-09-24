// rpu_cumsum.cpp — row-wise prefix sum along the last dim (Qwen3.5 GDN prefill
// chunk: cumsum over the gate/decay g, axis=-1).
//
// Hardware kernel: `cumsum_reduceC_tileN` (KernelId::CUMSUM).
// Input and output are [N,C] fp16; y[n,j] = sum_{i<=j} x[n,i].
// Register layout:
//   reg[0]        normal_blk_n (= 256, rows per tile)
//   reg[1]        last_blk_n   (= N - 256*(blk_cnt-1), rows in the tail tile)
//   reg[2]        N            (total rows)
//   reg[3]        C            (reduction length = last dim)
//   reg[4]/reg[5] input     SPM base (32-bit lo/hi)
//   reg[6]/reg[7] output    SPM base (32-bit lo/hi)
//   reg[8]/reg[9] workspace SPM base (32-bit lo/hi)
//   reg[10]/[11]  x_stride bytes (= 256*C*2, 32-bit lo/hi)
//   reg[12]/[13]  y_stride bytes (= 256*C*2, 32-bit lo/hi)
//   reg[14]       LSU write mode  (1   = local, 2B)
//   reg[15]       VALU write mode (146 = FP16x3)
// Grid: x = ceil(N / 256), y = z = 1.

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

void rpu_launch_cumsum_spm_kernel(uint32_t in_spm_addr, uint32_t out_spm_addr,
                                  uint32_t ws_spm_addr, int64_t N, int64_t C,
                                  int num_cores) {
    TORCH_CHECK(N > 0 && C > 0, "rpu_launch_cumsum_spm_kernel: N/C must be positive");
    TORCH_CHECK(num_cores >= 1, "rpu_launch_cumsum_spm_kernel: num_cores must be >= 1");
    Kernel_t* kernel = GET_KERNEL(KernelId::CUMSUM);
    TORCH_CHECK(kernel != nullptr, "Failed to get cumsum kernel (default ref)");

    const int normal_n = 256;
    const int blk_cnt  = (int)((N + normal_n - 1) / normal_n);
    const int last_n   = (int)(N - (int64_t)normal_n * (blk_cnt - 1));
    const uint32_t stride = (uint32_t)((int64_t)normal_n * C * 2);

    kernel->reset_regs();
    kernel->set_regs(0, (uint16_t)normal_n);
    kernel->set_regs(1, (uint16_t)last_n);
    kernel->set_regs(2, (uint16_t)N);
    kernel->set_regs(3, (uint16_t)C);
    kernel->set_regs(4, (uint16_t)(in_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(in_spm_addr >> 16));
    kernel->set_regs(6, (uint16_t)(out_spm_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(out_spm_addr >> 16));
    kernel->set_regs(8, (uint16_t)(ws_spm_addr & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(ws_spm_addr >> 16));
    kernel->set_regs(10, (uint16_t)(stride & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(stride >> 16));
    kernel->set_regs(12, (uint16_t)(stride & 0xFFFF));
    kernel->set_regs(13, (uint16_t)(stride >> 16));
    kernel->set_regs(14, static_cast<uint16_t>(1));     // GET_LSU_WMODE(local, 2B)
    kernel->set_regs(15, static_cast<uint16_t>(146));   // GET_VALU_WMODE(0, FP16x3)

    const uint16_t grid_x = (uint16_t)blk_cnt;
    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(num_cores > 1);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1}, cores);
}

// Isolated test harness (mirrors rpu_l2norm_test): DMA [N, C] in, run cumsum
// (single core), DMA back. input fp16 RPU; returns cumsum(input, dim=-1).
at::Tensor rpu_cumsum_test(const at::Tensor& input) {
    TORCH_CHECK(input.dim() == 2 && input.scalar_type() == at::kHalf,
                "cumsum_test: input must be 2D fp16 [N, C]");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "cumsum_test: input must be on RPU");
    const int64_t N = input.size(0), C = input.size(1);
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    const int64_t bytes = N * C * 2;
    // Workspace contains min(64, C) * 17 * 16 halfs and does not scale with N.
    const int64_t ws_elems = (int64_t)(C < 64 ? C : 64) * 17 * 16;
    const int64_t ws_bytes = ws_elems * 2;
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{bytes,    1, 2},  // input
        AR{ws_bytes, 1, 2},  // workspace
        AR{bytes,    2, 3},  // output
    });
    const uint32_t in_addr  = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t ws_addr  = SPM_ALLOC.addr(0, offsets[1]);
    const uint32_t out_addr = SPM_ALLOC.addr(0, offsets[2]);

    auto in_c = input.contiguous();
    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(in_c.data_ptr<c10::Half>()), N * C, in_addr, 1);
    rpu_launch_cumsum_spm_kernel(in_addr, out_addr, ws_addr, N, C);
    auto output = at::empty_like(in_c);
    rpu_launch_spm_copy_ddr_dma_immediate(out_addr, output.data_ptr<c10::Half>(),
                                          N * C);
    SPM_ALLOC.reset_temporary();
    return output;
}
