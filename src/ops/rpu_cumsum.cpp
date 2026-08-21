// Row-wise FP16 prefix sum over [N, C] tensors in SPM.
// Kernel launch ABI:
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
    TORCH_CHECK(kernel != nullptr,
                "Failed to get cumsum kernel from the combined operator asset");

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
    kernel->set_regs(14, 1);     // GET_LSU_WMODE(local, 2B)
    kernel->set_regs(15, 146);   // GET_VALU_WMODE(0, FP16x3)

    const uint16_t grid_x = (uint16_t)blk_cnt;
    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(num_cores > 1);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1}, cores);
}
