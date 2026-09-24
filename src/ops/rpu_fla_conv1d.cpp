// rpu_fla_conv1d.cpp — causal depthwise conv1d decode-update (Qwen3.5 GDN).
//
// Hardware kernel: `llm_fla_conv1d_w16a16_acc16` (KernelId::FLA_CONV1D).
// This repo uses the decode (L=1) and prefill (L>1) paths provided by the
// same kernel (gated by reg[10] is_prefill). NB: the loaded oplib's fla_conv1d
// MUST contain the prefill entry for rpu_launch_fla_conv1d_prefill_* to work.
// Decode step (one new token):
// given conv_state cs[B,Kc,D] (SPM) and x[B,1,D] (SPM) and weight[Kc,D] (DDR,
// repacked per-core), it updates cs IN PLACE:
//   cs[:,3] = x;  t = Σ_k cs[:,k]*w[k];  cs <- shift-left; cs[:,3] = t
// The conv output is cs[:,3] after the call. NO bias, NO activation (the SiLU in
// Qwen3.5 is applied separately afterward).
//
// Register layout:
//   reg[0]/reg[1]   conv_state SPM addr / 32  (v16 units, lo/hi)
//   reg[2]/reg[3]   x SPM addr / 32           (v16 units, lo/hi)
//   reg[4]/reg[5]   weight DDR addr >> 8      (repacked [num_cores,Kc,ld_pad])
//   reg[6]          cs_batch_v16 = Kc * (local_d/16)
//   reg[7]          d_v16 = local_d / 16
//   reg[8]          w_row_bytes = ceil_align(local_d,128) * 2
//   reg[9]          L = 1 (decode)
//   reg[10]         is_prefill = 0
//   reg[13]         x_batch_v16 = d_v16
//   grid: x = ceil(local_d/256), y = B, z = 1. cores 0..num_cores-1.

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <tuple>
#include <vector>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using namespace ::rhino_lkn;

static inline int64_t ceil_align_i(int64_t v, int64_t a) { return ((v + a - 1) / a) * a; }

void rpu_launch_fla_conv1d_spm_kernel(uint32_t cs_spm_addr, uint32_t x_spm_addr,
                                      c10::Half* weight_ddr, int64_t B,
                                      int64_t local_d, int64_t Kc, int num_cores) {
    TORCH_CHECK(local_d % 16 == 0, "fla_conv1d: local_d must be %16, got ", local_d);
    TORCH_CHECK(Kc == 4, "fla_conv1d: kernel size fixed to 4");
    TORCH_CHECK(weight_ddr != nullptr, "fla_conv1d: weight_ddr null");
    rpu_ddr_flush(weight_ddr);

    const uint64_t w_addr = RpuGetDevAddr(weight_ddr) >> 8;
    const int64_t ld_pad = ceil_align_i(local_d, 128);
    const uint16_t d_v16 = (uint16_t)(local_d / 16);

    Kernel_t* kernel = GET_KERNEL(KernelId::FLA_CONV1D);
    TORCH_CHECK(kernel != nullptr, "Failed to get llm_fla_conv1d kernel (default ref)");
    kernel->reset_regs();
    kernel->set_regs(0, (uint16_t)((cs_spm_addr / 32) & 0xFFFF));
    kernel->set_regs(1, (uint16_t)((cs_spm_addr / 32) >> 16));
    kernel->set_regs(2, (uint16_t)((x_spm_addr / 32) & 0xFFFF));
    kernel->set_regs(3, (uint16_t)((x_spm_addr / 32) >> 16));
    kernel->set_regs(4, (uint16_t)(w_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(w_addr >> 16));
    kernel->set_regs(6, (uint16_t)(Kc * d_v16));
    kernel->set_regs(7, d_v16);
    kernel->set_regs(8, (uint16_t)(ld_pad * 2));
    // Decode parameters.
    kernel->set_regs(9, (uint16_t)1);    // L = 1 (decode)
    kernel->set_regs(10, (uint16_t)0);   // is_prefill = 0
    kernel->set_regs(13, d_v16);         // x_batch_v16

    const uint16_t grid_x = (uint16_t)((local_d + 255) / 256);
    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)B, (uint16_t)1}, cores);
}

// ---- Prefill (L>1): causal depthwise conv1d over a whole chunk ---------------
// Input pad = [cs_init(3); x(L)] contiguous in SPM [B, 3+L, D] and remains
// read-only. The kernel writes y[B,L,D] and cs_out[B,3,D] to separate buffers.
// This removes the segmented in-place kernel's pre-saved halo requirement.
void rpu_launch_fla_conv1d_prefill_noinplace_spm_kernel(
    uint32_t pad_spm_addr, uint32_t out_spm_addr, uint32_t cs_out_spm_addr,
    c10::Half* weight_ddr, int64_t B, int64_t local_d, int64_t Kc, int64_t L,
    int num_cores) {
    TORCH_CHECK(local_d % 16 == 0, "fla_conv1d prefill: local_d must be %16, got ", local_d);
    TORCH_CHECK(Kc == 4, "fla_conv1d prefill: kernel size fixed to 4");
    TORCH_CHECK(L > 1, "fla_conv1d prefill: L must be > 1 (use decode for L=1)");
    TORCH_CHECK(weight_ddr != nullptr, "fla_conv1d prefill: weight_ddr null");
    rpu_ddr_flush(weight_ddr);

    const uint64_t w_addr   = RpuGetDevAddr(weight_ddr) >> 8;
    const int64_t  ld_pad   = ceil_align_i(local_d, 128);
    const uint16_t d_v16    = (uint16_t)(local_d / 16);
    const int64_t  tiles    = (local_d + 255) / 256;
    const int64_t  pad_rows = (Kc - 1) + L;                 // 3 + L
    const int64_t  pad_batch_v16 = pad_rows * d_v16;
    const int64_t  N        = (L >= 32) ? 8 : 1;
    const int64_t  seg_len  = L / N;
    const int64_t  seg_rem  = L % N;
    const uint32_t pad_v16    = pad_spm_addr / 32;          // v16 (32B) units
    const uint32_t out_v16    = out_spm_addr / 32;
    const uint32_t cs_out_v16 = cs_out_spm_addr / 32;
    const uint32_t ld_v16     = (uint32_t)(L * d_v16);      // v16 offset of pad row L
    TORCH_CHECK(B == 1 || (B - 1) * pad_batch_v16 < (1 << 16),
                "fla_conv1d prefill: batch stride exceeds kernel's 16-bit multiply");
    TORCH_CHECK(N == 1 || (N - 1) * seg_len * d_v16 < (1 << 16),
                "fla_conv1d prefill: segment offset exceeds kernel's 16-bit multiply");

    Kernel_t* kernel = GET_KERNEL(KernelId::FLA_CONV1D_NOINPLACE);
    TORCH_CHECK(kernel != nullptr, "Failed to get llm_fla_conv1d_noinplace kernel");
    kernel->reset_regs();
    kernel->set_regs(0,  (uint16_t)(pad_v16 & 0xFFFF));     // pad base (= cs_init head)
    kernel->set_regs(1,  (uint16_t)(pad_v16 >> 16));
    kernel->set_regs(4,  (uint16_t)(w_addr & 0xFFFF));
    kernel->set_regs(5,  (uint16_t)(w_addr >> 16));
    kernel->set_regs(6,  (uint16_t)(pad_batch_v16 & 0xFFFF));
    kernel->set_regs(7,  d_v16);
    kernel->set_regs(8,  (uint16_t)(ld_pad * 2));
    kernel->set_regs(9,  (uint16_t)L);
    kernel->set_regs(10, (uint16_t)1);                      // is_prefill
    kernel->set_regs(11, (uint16_t)(cs_out_v16 & 0xFFFF));
    kernel->set_regs(12, (uint16_t)(cs_out_v16 >> 16));
    kernel->set_regs(15, (uint16_t)(N > 1 ? 1 : 0));        // isSegmented
    kernel->set_regs(16, (uint16_t)(ld_v16 & 0xFFFF));
    kernel->set_regs(17, (uint16_t)(ld_v16 >> 16));
    kernel->set_regs(18, (uint16_t)seg_len);                // base = L//N steps per segment
    kernel->set_regs(21, (uint16_t)(N - 1));                // top segment idx (stores cs_out)
    kernel->set_regs(22, (uint16_t)seg_rem);                // rem = L%N (all to seg0)
    kernel->set_regs(23, (uint16_t)(out_v16 & 0xFFFF));
    kernel->set_regs(24, (uint16_t)(out_v16 >> 16));
    kernel->set_regs(64, (uint16_t)tiles);                  // gridDim.x
    kernel->set_regs(65, (uint16_t)B);                      // gridDim.y
    kernel->set_regs(66, (uint16_t)N);                      // gridDim.z = segment count

    const uint16_t grid_x = (uint16_t)tiles;
    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
    wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)B, (uint16_t)N}, cores);
}

// Isolated single-core test: cs [Kc, D], x [D], weight [Kc, D] (B=1, num_cores=1).
// Returns the updated conv_state [Kc, D] (conv output is row 3). weight must be
// already repacked/padded to [1, Kc, ld_pad] by the caller (D%16==0; for D a
// multiple of 128 no padding needed).
at::Tensor rpu_fla_conv1d_test(const at::Tensor& cs, const at::Tensor& x,
                               const at::Tensor& weight_repacked) {
    TORCH_CHECK(cs.dim() == 2 && cs.size(0) == 4 && cs.scalar_type() == at::kHalf,
                "fla_conv1d_test: cs must be [4, D] fp16");
    TORCH_CHECK(x.dim() == 1 && x.size(0) == cs.size(1),
                "fla_conv1d_test: x must be [D] matching cs");
    const int64_t Kc = 4, D = cs.size(1);
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    using AR = SpmAllocator::AllocRequest;
    const int64_t cs_bytes = Kc * D * 2;
    const int64_t x_bytes = D * 2;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{cs_bytes, 1, 3},  // conv_state (DMA-in → kernel update in place → DMA-out)
        AR{x_bytes, 1, 2},   // x input
    });
    const uint32_t cs_addr = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t x_addr = SPM_ALLOC.addr(0, offsets[1]);

    auto cs_c = cs.contiguous(), x_c = x.contiguous();
    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(cs_c.data_ptr<c10::Half>()), Kc * D, cs_addr, 1);
    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(x_c.data_ptr<c10::Half>()), D, x_addr, 1);

    auto w_c = weight_repacked.contiguous();
    rpu_launch_fla_conv1d_spm_kernel(
        cs_addr, x_addr, const_cast<c10::Half*>(w_c.data_ptr<c10::Half>()),
        /*B=*/1, /*local_d=*/D, Kc, /*num_cores=*/1);

    auto out = at::empty({Kc, D}, cs_c.options());
    rpu_launch_spm_copy_ddr_dma_immediate(cs_addr, out.data_ptr<c10::Half>(), Kc * D);
    SPM_ALLOC.reset_temporary();
    return out;
}

// Isolated single-core prefill test: cs_init [3, D], x [L, D], weight [Kc, ld_pad]
// (B=1, num_cores=1). Returns {y [L, D] (conv output, no SiLU), cs_out [3, D] (new
// conv_state = last 3 tokens)}. weight must be repacked/padded to [Kc, ld_pad];
// for D % 128 == 0 pass [4, D] directly.
std::tuple<at::Tensor, at::Tensor> rpu_fla_conv1d_prefill_test(
    const at::Tensor& cs_init, const at::Tensor& x, const at::Tensor& weight_repacked) {
    TORCH_CHECK(cs_init.dim() == 2 && cs_init.size(0) == 3 &&
                    cs_init.scalar_type() == at::kHalf,
                "fla_conv1d_prefill_test: cs_init must be [3, D] fp16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == cs_init.size(1) &&
                    x.scalar_type() == at::kHalf,
                "fla_conv1d_prefill_test: x must be [L, D] matching cs_init");
    const int64_t Kc = 4, D = cs_init.size(1), L = x.size(0);
    TORCH_CHECK(L > 1, "fla_conv1d_prefill_test: L must be > 1");
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    const int64_t pad_rows = (Kc - 1) + L;
    using AR = SpmAllocator::AllocRequest;
    const int64_t pad_bytes    = pad_rows * D * 2;
    const int64_t out_bytes    = L * D * 2;
    const int64_t cs_out_bytes = (Kc - 1) * D * 2;
    auto offsets = SPM_ALLOC.alloc_temporary_aliased({
        AR{pad_bytes,    1, 2},  // pad [3+L, D], read-only kernel input
        AR{out_bytes,    2, 3},  // y [L, D]
        AR{cs_out_bytes, 2, 3},  // cs_out [3, D]
    });
    const uint32_t pad_addr    = SPM_ALLOC.addr(0, offsets[0]);
    const uint32_t out_addr    = SPM_ALLOC.addr(0, offsets[1]);
    const uint32_t cs_out_addr = SPM_ALLOC.addr(0, offsets[2]);

    auto cs_c = cs_init.contiguous(), x_c = x.contiguous();
    // pad = [cs_init(3) ; x(L)] laid out contiguously in the pad region
    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(cs_c.data_ptr<c10::Half>()), (Kc - 1) * D, pad_addr, 1);
    rpu_launch_ddr_broadcast_spm_dma_immediate(
        const_cast<c10::Half*>(x_c.data_ptr<c10::Half>()), L * D,
        pad_addr + (uint32_t)((Kc - 1) * D * 2), 1);

    auto w_c = weight_repacked.contiguous();
    rpu_launch_fla_conv1d_prefill_noinplace_spm_kernel(
        pad_addr, out_addr, cs_out_addr,
        const_cast<c10::Half*>(w_c.data_ptr<c10::Half>()),
        /*B=*/1, /*local_d=*/D, Kc, L, /*num_cores=*/1);

    auto y = at::empty({L, D}, cs_c.options());
    rpu_launch_spm_copy_ddr_dma_immediate(out_addr, y.data_ptr<c10::Half>(), L * D);
    auto cs_out = at::empty({Kc - 1, D}, cs_c.options());
    rpu_launch_spm_copy_ddr_dma_immediate(cs_out_addr, cs_out.data_ptr<c10::Half>(),
                                          (Kc - 1) * D);
    SPM_ALLOC.reset_temporary();
    return std::make_tuple(y, cs_out);
}
