// rpu_unit_tril_inv.cpp — R = (I - M)^-1 for a strict lower-triangular M
// (Qwen3.5 GDN prefill chunk: the UT transform / inverse of the attn matrix).
//
// Hardware kernel: `unit_tril_inv` (KernelId::UNIT_TRIL_INV).
// M is a batch of strict lower-triangular C x C blocks with C = 64 fixed;
// the kernel inverts (I - block)
// in place per block. Single-core per grid block; no workspace.
//
// Register layout:
//   reg[0]/reg[1]   input  SPM base (32-bit lo/hi)
//   reg[2]/reg[3]   output SPM base (32-bit lo/hi)
//   grid.x = H*N (number of 64x64 blocks = numel / (64*64)); grid.y = grid.z = 1
//   (the SDK writes grid into reg[64..66] from the enqueu_kernel grid argument).
// C = 64 is baked into the kernel, so no dim registers are passed.

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
    TORCH_CHECK(kernel != nullptr, "Failed to get unit_tril_inv kernel (default ref)");

    kernel->reset_regs();
    kernel->set_regs(0, (uint16_t)(in_spm_addr & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(in_spm_addr >> 16));
    kernel->set_regs(2, (uint16_t)(out_spm_addr & 0xFFFF));
    kernel->set_regs(3, (uint16_t)(out_spm_addr >> 16));
    // gridDim.x = num_blocks (one C×C block per blockIdx.x). Set reg[64..66] +
    // enqueu block_dims like rpu_rope_2d / rpu_argmax / rpu_im2col.
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

// Isolated test harness (mirrors rpu_l2norm_test): DMA [.., 64, 64] in, run
// unit_tril_inv (single core), DMA back. input fp16 RPU (strict lower-tri M);
// returns (I - M)^-1 per 64x64 block.
at::Tensor rpu_unit_tril_inv_test(const at::Tensor& m) {
    TORCH_CHECK(m.scalar_type() == at::kHalf && m.dim() >= 2 &&
                    m.size(-1) == 64 && m.size(-2) == 64,
                "unit_tril_inv_test: input must be fp16 [.., 64, 64]");
    TORCH_CHECK(m.device().type() == at::kPrivateUse1,
                "unit_tril_inv_test: input must be on RPU");
    auto x = m.contiguous();
    const int64_t numel  = x.numel();
    const int64_t blocks = numel / (64 * 64);
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    // The kernel writes ONLY the lower-triangle + diagonal of each block and runs
    // IN-PLACE; it never touches the upper triangle. So use a SINGLE SPM buffer for
    // in/out: the unwritten upper triangle then stays = the input's upper triangle =
    // 0 (M is strict lower-tri), giving a clean unit-lower-tri (I-M)^-1. A SEPARATE
    // output buffer leaves the upper triangle as uninitialized SPM (stale data from
    // prior ops) -> nondeterministic garbage/NaN above the diagonal (the multi-block
    // flakiness). Real GDN-chunk callers must likewise pass in==out (or pre-zero out).
    const int64_t bytes = numel * 2;
    using AR = SpmAllocator::AllocRequest;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{bytes, 1, 3},  // single in-place buffer (M inverted in place)
    });
    const uint32_t addr = SPM_ALLOC.addr(0, offsets[0]);

    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(x.data_ptr<c10::Half>()), numel, addr, 1);
    rpu_launch_unit_tril_inv_spm_kernel(addr, addr, blocks);   // in == out (in-place)
    auto output = at::empty_like(x);
    rpu_launch_spm_copy_ddr_dma_immediate(addr, output.data_ptr<c10::Half>(),
                                          numel);
    SPM_ALLOC.reset_temporary();
    return output;
}
