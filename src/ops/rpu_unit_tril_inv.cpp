// In-place FP16 unit lower-triangular inverse over 64x64 SPM blocks.
// Kernel launch ABI:
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

void rpu_launch_unit_tril_inv_spm_kernel(uint32_t in_spm_addr, uint32_t out_spm_addr,
                                         int64_t num_blocks, int num_cores) {
    TORCH_CHECK(num_blocks > 0, "rpu_launch_unit_tril_inv_spm_kernel: num_blocks must be positive");
    TORCH_CHECK(num_cores >= 1, "rpu_launch_unit_tril_inv_spm_kernel: num_cores must be >= 1");
    Kernel_t* kernel = GET_KERNEL(KernelId::UNIT_TRIL_INV);
    TORCH_CHECK(kernel != nullptr,
                "Failed to get unit_tril_inv kernel from the combined operator asset");

    kernel->reset_regs();
    kernel->set_regs(0, (uint16_t)(in_spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(in_spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(out_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(out_spm_addr >> 16));
    // Populate the required launch grid for num_blocks.
    const uint16_t grid_x = (uint16_t)num_blocks;
    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, (uint16_t)1);
    kernel->set_regs(66, (uint16_t)1);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(num_cores > 1);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1}, cores);
}
