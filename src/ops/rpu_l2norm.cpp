// rpu_l2norm.cpp — row-wise L2 normalization (Qwen3.5 GDN q/k norm).
//
// Hardware kernel: `l2norm` (KernelId::L2NORM). B=M rows each of length
// K=D reduced over the last axis: y[m] = x[m] * rsqrt(sum(x[m]^2) + eps).
//
// Register layout:
//   reg[0]/reg[1]   input  SPM base (32-bit, lo/hi)
//   reg[2]/reg[3]   output SPM base (32-bit, lo/hi)
//   reg[4]          B  (= M, number of rows)
//   reg[5]          K  (= D, reduction length, e.g. head_k_dim=128)
//   reg[6]          eps (fp16 bit pattern)
// Grid: x = ceil(M / 16), y = z = 1. Single core {0}.

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
    TORCH_CHECK(kernel != nullptr, "Failed to get l2norm kernel (default ref)");

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

// Isolated test harness (mirrors partial_mrope_test): DMA [M,D] in, run l2norm
// (single core), DMA back. input fp16 RPU; returns x*rsqrt(sum(x^2,-1)+eps).
at::Tensor rpu_l2norm_test(const at::Tensor& input, double eps) {
    TORCH_CHECK(input.dim() == 2 && input.scalar_type() == at::kHalf,
                "l2norm_test: input must be 2D fp16 [M, D]");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                "l2norm_test: input must be on RPU");
    const int64_t M = input.size(0), D = input.size(1);
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    const int64_t bytes = M * D * 2;
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{bytes, 1, 2},  // input
        AR{bytes, 2, 3},  // output
    });
    const uint32_t in_addr  = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t out_addr = SPM_ALLOC.addr(0, offsets[1]);

    auto in_c = input.contiguous();
    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(in_c.data_ptr<c10::Half>()), M * D, in_addr, 1);
    rpu_launch_l2norm_spm_kernel(in_addr, out_addr, M, D, eps);
    auto output = at::empty_like(in_c);
    rpu_launch_spm_copy_ddr_dma_immediate(out_addr, output.data_ptr<c10::Half>(),
                                          M * D);
    SPM_ALLOC.reset_temporary();
    return output;
}
