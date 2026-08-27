// rpu_pad.cpp — Pad kernel launch wrapper for RPU.
//
// Pads a tensor's channel dimension with a constant value in SPM.
// Uses pad_const_NxC_NxCP kernel (normal variant).
//
// For NCHW channel padding [1, cin, H, W] → [1, cin_padded, H, W]:
//   Reshape as [1, cin*H*W] → pad tail by (cin_padded - cin)*H*W → [1, cin_padded*H*W]
//   (appending zero-filled channels is equivalent to tail-padding the flattened view)
//
// Register mapping for the normal variant:
//   params[0]    = normal_blk_n          (N per grid block)
//   params[1]    = last_blk_n            (last block's N count)
//   params[2:3]  = C                     (input last dim, 32-bit split)
//   params[4:5]  = CP                    (output last dim, 32-bit split)
//   params[6:7]  = a_base                (input SPM byte addr, 32-bit split)
//   params[8:9]  = r_base                (output SPM byte addr, 32-bit split)
//   params[10:11]= block_idata_stride    (input stride per block, bytes, 32-bit)
//   params[12:13]= block_odata_stride    (output stride per block, bytes, 32-bit)
//   params[14:15]= PadFrontElementCnt    (front pad elements, 32-bit)
//   params[16:17]= PadTailElementCnt     (tail pad elements, 32-bit)
//   params[18]   = pad_const_val         (fp16 constant as uint16)
//   params[19]   = dwidth                (2 for fp16)
//   params[64:66]= gridDim.x/y/z

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using Kernel_t = ::rhino_lkn::Kernel_t;
using Queue_t  = ::rhino_lkn::Queue_t;

static void set_reg_u32(Kernel_t* k, int lo, uint32_t val) {
    k->set_regs(lo,     (uint16_t)(val & 0xFFFF));
    k->set_regs(lo + 1, (uint16_t)(val >> 16));
}

static void setup_pad_regs(
    Kernel_t* kernel,
    uint32_t spm_in_addr,   // absolute SPM byte address
    uint32_t spm_out_addr,  // absolute SPM byte address
    int64_t N, int64_t C, int64_t CP,
    int64_t pad_front, int64_t pad_tail,
    uint16_t pad_val,       // fp16 constant as uint16
    uint16_t grid_x)
{
    int dwidth = 2;  // fp16

    // Partition N rows across the configured launch grid.
    int normal_blk_n, last_blk_n;
    if (N <= 16) {
        normal_blk_n = 1;
    } else {
        normal_blk_n = (int)(N / 16);
    }
    last_blk_n = (int)(N - (int64_t)normal_blk_n * (grid_x - 1));

    uint32_t block_idata_stride = (uint32_t)(normal_blk_n * C * dwidth);
    uint32_t block_odata_stride = (uint32_t)(normal_blk_n * CP * dwidth);

    kernel->set_regs(0, (uint16_t)normal_blk_n);
    kernel->set_regs(1, (uint16_t)last_blk_n);
    set_reg_u32(kernel, 2,  (uint32_t)C);
    set_reg_u32(kernel, 4,  (uint32_t)CP);
    set_reg_u32(kernel, 6,  spm_in_addr);
    set_reg_u32(kernel, 8,  spm_out_addr);
    set_reg_u32(kernel, 10, block_idata_stride);
    set_reg_u32(kernel, 12, block_odata_stride);
    set_reg_u32(kernel, 14, (uint32_t)pad_front);
    set_reg_u32(kernel, 16, (uint32_t)pad_tail);
    kernel->set_regs(18, pad_val);
    kernel->set_regs(19, (uint16_t)dwidth);

    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, (uint16_t)1);
    kernel->set_regs(66, (uint16_t)1);
}

void rpu_launch_pad_channel_spm(
    uint32_t spm_in_off,   // SPM offset (bytes) for input  [N, C] or [1, cin*H*W]
    uint32_t spm_out_off,  // SPM offset (bytes) for output [N, CP] or [1, cin_padded*H*W]
    int64_t N,             // outer dimension (typically 1)
    int64_t C,             // input inner dimension (cin * H * W)
    int64_t pad_front,     // elements to pad before data
    int64_t pad_tail,      // elements to pad after data
    int num_cores)         // >1: SPMD replicate — every core pads its own copy
{
    int64_t CP = C + pad_front + pad_tail;

    uint16_t grid_x = (N <= 16) ? (uint16_t)N : (uint16_t)16;

    uint32_t spm_in_addr  = SPM_ALLOC.addr(0, spm_in_off);
    uint32_t spm_out_addr = SPM_ALLOC.addr(0, spm_out_off);

    // pad value = 0.0 fp16 → bit pattern 0x0000
    uint16_t pad_val = 0;

      Kernel_t* kernel = GET_KERNEL(KernelId::PAD_CONST_NXC);
      TORCH_CHECK(kernel, "rpu_pad: PAD_CONST_NXC kernel not found");

      setup_pad_regs(kernel, spm_in_addr, spm_out_addr,
                     N, C, CP, pad_front, pad_tail,
                     pad_val, grid_x);

      // num_cores>1: SPMD — every core pads its own data at the SAME SPM offset
      // (per-core input, e.g. a broadcast image). broadcast_mode replicates the
      // register params to all cores; without it only core 0 gets them.
      // The num_cores==1 branch is left byte-identical to the pre-multicore
      // code (no set_broadcast_mode call) — on a 1-core launch that flag still
      // carries meaning for some kernels (see rpu_rope.cpp), so single-core
      // callers must keep inheriting whatever the queue already had.
      auto* wq = GET_QUEUE(num_cores);
      if (num_cores > 1) {
          wq->set_broadcast_mode(true);
          std::vector<uint8_t> cores(num_cores);
          for (int i = 0; i < num_cores; ++i) cores[i] = (uint8_t)i;
          wq->enqueu_kernel(*kernel, {grid_x, 1, 1}, cores);
      } else {
          wq->enqueu_kernel(*kernel, {grid_x, 1, 1}, {0});
      }
}
