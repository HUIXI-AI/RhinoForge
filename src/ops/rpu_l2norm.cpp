// Row-wise FP16 L2 normalization over [M, D] tensors in SPM.
// Kernel launch ABI:
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

void rpu_launch_l2norm_spm_kernel(uint32_t in_spm_addr, uint32_t out_spm_addr,
                                  int64_t M, int64_t D, double eps, int num_cores) {
    TORCH_CHECK(M > 0 && D > 0, "rpu_launch_l2norm_spm_kernel: M/D must be positive");
    TORCH_CHECK(num_cores >= 1, "rpu_launch_l2norm_spm_kernel: num_cores must be >= 1");
    Kernel_t* kernel = GET_KERNEL(KernelId::L2NORM);
    TORCH_CHECK(kernel != nullptr,
                "Failed to get l2norm kernel from the combined operator asset");

    const c10::Half eps_h = static_cast<c10::Half>(static_cast<float>(eps));
    kernel->reset_regs();
    kernel->set_regs(0, (uint16_t)(in_spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(in_spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(out_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(out_spm_addr >> 16));
    kernel->set_regs(4, (uint16_t)M);
    kernel->set_regs(5, (uint16_t)D);
    kernel->set_regs(6, eps_h.x);

    // num_cores>1: SPMD — every core l2norms its own M rows at the SAME SPM
    // offset (per-core data, e.g. 2 GDN heads/core). broadcast_mode replicates
    // the register params to all cores.
    const uint16_t grid_x = (uint16_t)((M + 15) / 16);
    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(num_cores > 1);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1}, cores);
}
