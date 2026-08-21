// Causal depthwise FP16 conv1d for decode and prefill SPM paths.
// Decode updates a [B, Kc, D] state in place; prefill consumes [B, Kc-1+L, D].
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
    TORCH_CHECK(kernel != nullptr,
                "Failed to get llm_fla_conv1d kernel from the combined operator asset");
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
    // Decode mode leaves prefill-only parameters unset.
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
// Input pad = [cs_init(3); x(L)] contiguous in SPM [B, 3+L, D]. The kernel writes
// the conv output y[L,D] IN PLACE over the x rows (pad[3:3+L]) and the new
// conv_state cs_out[3,D] (= the last 3 rows of pad) to a SEPARATE buffer. NO bias,
// NO activation (SiLU applied afterward). The L axis is split into N segments:
// N=8 when L>=32 (halo[B,N*3,D] carries each segment's left-neighbor 3 taps),
// otherwise N=1. Decode-only parameters are unused in prefill mode.
void rpu_launch_fla_conv1d_prefill_spm_kernel(
    uint32_t pad_spm_addr, uint32_t cs_out_spm_addr, uint32_t halo_spm_addr,
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
    const uint32_t cs_out_v16 = cs_out_spm_addr / 32;
    const uint32_t halo_v16   = halo_spm_addr / 32;
    const uint32_t ld_v16     = (uint32_t)(L * d_v16);      // v16 offset of pad row L

    Kernel_t* kernel = GET_KERNEL(KernelId::FLA_CONV1D);
    TORCH_CHECK(kernel != nullptr,
                "Failed to get llm_fla_conv1d kernel from the combined operator asset");
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
    kernel->set_regs(19, (uint16_t)(halo_v16 & 0xFFFF));
    kernel->set_regs(20, (uint16_t)(halo_v16 >> 16));
    kernel->set_regs(21, (uint16_t)(N - 1));                // top segment idx (stores cs_out)
    kernel->set_regs(22, (uint16_t)seg_rem);                // rem = L%N (all to seg0)
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
