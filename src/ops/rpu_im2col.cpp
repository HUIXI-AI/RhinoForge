// rpu_im2col.cpp — Im2col kernel launch wrapper for RPU.
//
// Extracts patches from an input feature map (NHWC, in SPM) into a column matrix
// (in SPM) suitable for GEMM. Used to convert Conv2d(stride==kernel_size) into
// im2col + GEMM for patch embedding fusion.
//
// Kernel: im2col_coren_m{tile_m}xn{tile_n}_buf1_sIsO (SPM input → SPM output)
// Data layout: NHWC input [batch, inh, inw, cin] → output [batch, outh*outw, flth*fltw*cin]
//
// Register mapping:
//   param[0] = flth, param[1] = fltw, param[2] = flthw
//   param[3] = core_num, param[4] = cin*2 (bytes)
//   param[5] = strideh, param[6] = stridew
//   param[7] = holeh (dilation), param[8] = holew (dilation)
//   param[64-66] = grid_dim_x/y/z
//
// SCM layout (per core j, base = 4096 + j*16):
//   [0-1] spm_in byte addr (32-bit split)
//   [2-3] spm_col byte addr (32-bit split)
//   [4] inh, [5] inw, [6] outh, [7] outw
//   [8] inhw, [9] outhw
//   [10] padh, [11] padH, [12] padw, [13] padW
//   [14] batch_per_core, [15] reserved (0)

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using Kernel_t = ::rhino_lkn::Kernel_t;
using Queue_t  = ::rhino_lkn::Queue_t;

static constexpr int SCM_BASE        = 4096;
static constexpr int SCM_SLOTS_PER_CORE = 16;

// Select im2col KernelId based on cin and outhw
static KernelId select_im2col_kernel(int cin, int outhw) {
    // tile_n: choose smallest that covers cin
    // tile_m: choose based on outhw for good grid utilization
    // Available: m64xn32, m128xn32, m64xn64, m128xn64

    int tile_n = (cin <= 32) ? 32 : 64;
    int tile_m = (outhw <= 128) ? 64 : 128;

    if (tile_m == 64 && tile_n == 32) return KernelId::IM2COL_M64XN32;
    if (tile_m == 128 && tile_n == 32) return KernelId::IM2COL_M128XN32;
    if (tile_m == 64 && tile_n == 64) return KernelId::IM2COL_M64XN64;
    return KernelId::IM2COL_M128XN64;
}

static void setup_im2col_regs(
    Kernel_t* kernel,
    uint32_t spm_in_byte_addr,   // per-core SPM input byte address
    uint32_t spm_col_byte_addr,  // per-core SPM output byte address
    int flth, int fltw, int cin,
    int strideh, int stridew,
    int holeh, int holew,
    int padh, int padH, int padw, int padW,
    int inh, int inw, int outh, int outw,
    int batch_per_core, int num_cores,
    uint16_t grid_dim_x, uint16_t grid_dim_y, uint16_t grid_dim_z)
{
    // param registers (broadcast to all cores)
    kernel->set_regs(0, (uint16_t)flth);
    kernel->set_regs(1, (uint16_t)fltw);
    kernel->set_regs(2, (uint16_t)(flth * fltw));
    kernel->set_regs(3, (uint16_t)num_cores);
    kernel->set_regs(4, (uint16_t)(cin * 2));  // bytes (fp16)
    kernel->set_regs(5, (uint16_t)strideh);
    kernel->set_regs(6, (uint16_t)stridew);
    kernel->set_regs(7, (uint16_t)holeh);
    kernel->set_regs(8, (uint16_t)holew);

    kernel->set_regs(64, grid_dim_x);
    kernel->set_regs(65, grid_dim_y);
    kernel->set_regs(66, grid_dim_z);

    int inhw  = inh * inw;
    int outhw = outh * outw;

    // SCM registers (per core j)
    for (int j = 0; j < num_cores; j++) {
        int base = SCM_BASE + j * SCM_SLOTS_PER_CORE;
        // SPM addresses are byte addresses for im2col kernel
        // For single-core (batch split): all cores share same layout since batch=1
        kernel->set_regs(base + 0,  (uint16_t)(spm_in_byte_addr & 0xFFFF));
        kernel->set_regs(base + 1,  (uint16_t)(spm_in_byte_addr >> 16));
        kernel->set_regs(base + 2,  (uint16_t)(spm_col_byte_addr & 0xFFFF));
        kernel->set_regs(base + 3,  (uint16_t)(spm_col_byte_addr >> 16));
        kernel->set_regs(base + 4,  (uint16_t)inh);
        kernel->set_regs(base + 5,  (uint16_t)inw);
        kernel->set_regs(base + 6,  (uint16_t)outh);
        kernel->set_regs(base + 7,  (uint16_t)outw);
        kernel->set_regs(base + 8,  (uint16_t)inhw);
        kernel->set_regs(base + 9,  (uint16_t)outhw);
        kernel->set_regs(base + 10, (uint16_t)padh);
        kernel->set_regs(base + 11, (uint16_t)padH);
        kernel->set_regs(base + 12, (uint16_t)padw);
        kernel->set_regs(base + 13, (uint16_t)padW);
        kernel->set_regs(base + 14, (uint16_t)batch_per_core);
        kernel->set_regs(base + 15, (uint16_t)0);
    }
}

void rpu_launch_im2col_spm(
    uint32_t spm_in_off,    // SPM offset (bytes) for input [batch, inh, inw, cin] NHWC
    uint32_t spm_col_off,   // SPM offset (bytes) for output [batch, outh*outw, flth*fltw*cin]
    int batch, int inh, int inw, int cin,
    int flth, int fltw,
    int padh, int padH, int padw, int padW,
    int strideh, int stridew,
    int holeh, int holew,
    int num_cores)
{
    // Compute output spatial dimensions
    int fltheff = flth + (flth - 1) * (holeh - 1);
    int fltweff = fltw + (fltw - 1) * (holew - 1);
    int outh = (inh + padh + padH - fltheff) / strideh + 1;
    int outw = (inw + padw + padW - fltweff) / stridew + 1;
    int outhw = outh * outw;

    // For batch split: require batch % num_cores == 0
    // Typical use: batch=1, num_cores=1 (single-core im2col)
    TORCH_CHECK(batch % num_cores == 0,
        "rpu_im2col: batch(", batch, ") must be divisible by num_cores(", num_cores, ")");
    int batch_per_core = batch / num_cores;

    KernelId kid = select_im2col_kernel(cin, outhw);

    // Determine tile sizes from kernel variant
    int tile_m, tile_n;
    switch (kid) {
        case KernelId::IM2COL_M64XN32:  tile_m = 64;  tile_n = 32;  break;
        case KernelId::IM2COL_M128XN32: tile_m = 128; tile_n = 32;  break;
        case KernelId::IM2COL_M64XN64:  tile_m = 64;  tile_n = 64;  break;
        case KernelId::IM2COL_M128XN64: tile_m = 128; tile_n = 64;  break;
        default: TORCH_CHECK(false, "rpu_im2col: invalid kernel id"); break;
    }

    uint16_t grid_dim_x = (uint16_t)CeilDiv(cin, tile_n);
    uint16_t grid_dim_y = (uint16_t)CeilDiv(outhw, tile_m);
    uint16_t grid_dim_z = (uint16_t)batch_per_core;

    // SPM byte addresses (core 0)
    uint32_t spm_in_addr  = SPM_ALLOC.addr(0, spm_in_off);
    uint32_t spm_col_addr = SPM_ALLOC.addr(0, spm_col_off);

      Kernel_t* kernel = GET_KERNEL(kid);
      TORCH_CHECK(kernel, "rpu_im2col: kernel not found for id=",
                  static_cast<int>(kid));

      setup_im2col_regs(kernel,
          spm_in_addr, spm_col_addr,
          flth, fltw, cin,
          strideh, stridew, holeh, holew,
          padh, padH, padw, padW,
          inh, inw, outh, outw,
          batch_per_core, num_cores,
          grid_dim_x, grid_dim_y, grid_dim_z);

      auto* wq = GET_QUEUE(num_cores);
      if (num_cores > 1) {
          wq->set_broadcast_mode(true);
          std::vector<uint8_t> core_ids(num_cores);
          for (int i = 0; i < num_cores; i++) core_ids[i] = (uint8_t)i;
          wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, core_ids);
      } else {
          wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, {0});
      }
}
