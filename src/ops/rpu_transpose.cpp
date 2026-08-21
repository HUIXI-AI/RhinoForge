// rpu_transpose.cpp — Transpose kernel launch wrapper for RPU.
//
// Transposes a 3D tensor [b, n, c] → [n, c, b] (order "120") in SPM.
// Used for NCHW→NHWC by collapsing batch: [C, H, W] → [H, W, C].
//
// Kernel: transpose_ncb_c16 (order "120", get_tailb_config)
//
// Register mapping for get_tailb_config:
//   params[0:1]  = spm_in_byte_base   (32-bit)
//   params[2:3]  = spm_out_byte_base  (32-bit)
//   params[4]    = data_type           (1=fp16)
//   params[6]    = c                   (inner dim, W)
//   params[7]    = c_tail              (ceil_align(c, 16) or c if aligned)
//   params[8]    = b                   (outer dim, C)
//   params[9]    = b_v16              = ceil_div(b, 16)
//   params[10]   = b_tail             = b or ceil_align(b, 16)
//   params[11]   = b_v16_tail         = b_v16 + 1
//   params[12]   = c_per_thd          = c_v16_per_wrp * 16
//   params[13]   = c_per_wrp          = c_v16_per_wrp * 16
//   params[14]   = c_v16_per_wrp      = min(MAX_VLM_V16, c_v16)
//   params[15]   = c_v256_per_wrp     = c_v16_per_wrp / 16
//   params[16]   = b_per_thd          = b_v16_per_wrp * 16
//   params[17]   = b_per_wrp          = b_v16_per_wrp * 16
//   params[18]   = b_v16_per_wrp
//   params[19]   = n                   (middle dim, H)
//   params[64:66]= gridDim.x/y/z

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

using Kernel_t = ::rhino_lkn::Kernel_t;
using Queue_t  = ::rhino_lkn::Queue_t;

// V16_NUM and MAX_FP16_VLM_V16_PER_THD are defined in rpu_helpers.h (via rpu_ops.h)

static void set_reg_u32(Kernel_t* k, int lo, uint32_t val) {
    k->set_regs(lo,     (uint16_t)(val & 0xFFFF));
    k->set_regs(lo + 1, (uint16_t)(val >> 16));
}

static void setup_transpose_ncb_regs(
    Kernel_t* kernel,
    uint32_t spm_in_addr,
    uint32_t spm_out_addr,
    int b, int n, int c,
    uint16_t grid_x, uint16_t grid_y, uint16_t grid_z)
{
    int c_v16 = CeilDiv(c, V16_NUM);
    int b_v16 = CeilDiv(b, V16_NUM);

    // No tail padding needed when dimensions are v16-aligned
    int c_tail = c;  // assumes c % 16 == 0
    int b_tail = b;  // assumes b % 16 == 0
    int b_v16_tail = b_v16 + 1;

    int c_v16_per_wrp = std::min(MAX_FP16_VLM_V16_PER_THD, c_v16);
    int b_v16_per_wrp = std::min(b_v16,
                                  MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
    int c_v256_per_wrp = c_v16_per_wrp / V16_NUM;

    int c_per_thd = c_v16_per_wrp * V16_NUM;
    int c_per_wrp = c_v16_per_wrp * V16_NUM;
    int b_per_thd = b_v16_per_wrp * V16_NUM;
    int b_per_wrp = b_v16_per_wrp * V16_NUM;

    set_reg_u32(kernel, 0, spm_in_addr);
    set_reg_u32(kernel, 2, spm_out_addr);
    kernel->set_regs(4, (uint16_t)1);  // data_type=1 (fp16)

    kernel->set_regs(6,  (uint16_t)c);
    kernel->set_regs(7,  (uint16_t)c_tail);
    kernel->set_regs(8,  (uint16_t)b);
    kernel->set_regs(9,  (uint16_t)b_v16);
    kernel->set_regs(10, (uint16_t)b_tail);
    kernel->set_regs(11, (uint16_t)b_v16_tail);
    kernel->set_regs(12, (uint16_t)c_per_thd);
    kernel->set_regs(13, (uint16_t)c_per_wrp);
    kernel->set_regs(14, (uint16_t)c_v16_per_wrp);
    kernel->set_regs(15, (uint16_t)c_v256_per_wrp);
    kernel->set_regs(16, (uint16_t)b_per_thd);
    kernel->set_regs(17, (uint16_t)b_per_wrp);
    kernel->set_regs(18, (uint16_t)b_v16_per_wrp);
    kernel->set_regs(19, (uint16_t)n);

    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, grid_y);
    kernel->set_regs(66, grid_z);
}

void rpu_launch_transpose_nchw_to_nhwc_spm(
    uint32_t spm_in_off,   // SPM offset (bytes) for input  [C, H, W] (NCHW with batch collapsed)
    uint32_t spm_out_off,  // SPM offset (bytes) for output [H, W, C] (NHWC with batch collapsed)
    int C, int H, int W,   // C must be v16-aligned (divisible by 16)
    int num_cores)         // >1: SPMD replicate — every core transposes its own copy
{
    TORCH_CHECK(C % V16_NUM == 0,
        "rpu_transpose_nchw_to_nhwc: C (", C, ") must be divisible by ", V16_NUM);

    // Map to 3D transpose [b, n, c] → [n, c, b] (order "120")
    // b=C, n=H, c=W
    int b = C, n = H, c = W;

    int c_v16 = CeilDiv(c, V16_NUM);
    int b_v16 = CeilDiv(b, V16_NUM);

    int c_v16_per_wrp = std::min(MAX_FP16_VLM_V16_PER_THD, c_v16);
    int b_v16_per_wrp = std::min(b_v16,
                                  MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);

    uint16_t grid_x = (uint16_t)CeilDiv(b_v16, b_v16_per_wrp);
    uint16_t grid_y = (uint16_t)CeilDiv(c_v16, c_v16_per_wrp);
    uint16_t grid_z = (uint16_t)n;

    uint32_t spm_in_addr  = SPM_ALLOC.addr(0, spm_in_off);
    uint32_t spm_out_addr = SPM_ALLOC.addr(0, spm_out_off);

      Kernel_t* kernel = GET_KERNEL(KernelId::TRANSPOSE_NCB_C16);
      TORCH_CHECK(kernel, "rpu_transpose: TRANSPOSE_NCB_C16 kernel not found");

      setup_transpose_ncb_regs(kernel, spm_in_addr, spm_out_addr,
                                b, n, c, grid_x, grid_y, grid_z);

      // num_cores>1: SPMD — see rpu_launch_pad_channel_spm's note, including
      // why the 1-core branch stays byte-identical to the pre-multicore code.
      auto* wq = GET_QUEUE(num_cores);
      if (num_cores > 1) {
          wq->set_broadcast_mode(true);
          std::vector<uint8_t> cores(num_cores);
          for (int i = 0; i < num_cores; ++i) cores[i] = (uint8_t)i;
          wq->enqueu_kernel(*kernel, {grid_x, grid_y, grid_z}, cores);
      } else {
          wq->enqueu_kernel(*kernel, {grid_x, grid_y, grid_z}, {0});
      }
}

// Direct [B,N,C] -> [N,B,C] (order "102") in SPM. This preserves the
// contiguous C dimension, so [seq,heads,head_dim] becomes
// [heads,seq,head_dim] in one data-movement pass.
void rpu_launch_transpose_bnc_to_nbc_spm(
    uint32_t spm_in_off,
    uint32_t spm_out_off,
    int B, int N, int C)
{
    TORCH_CHECK(B > 0 && N > 0 && C > 0,
        "rpu_transpose_bnc_to_nbc: dimensions must be positive");
    TORCH_CHECK(C % V16_NUM == 0,
        "rpu_transpose_bnc_to_nbc: C (", C, ") must be divisible by ", V16_NUM);

    const int c_v16 = CeilDiv(C, V16_NUM);
    const int n_v16 = CeilDiv(N, V16_NUM);
    const int c_v16_per_wrp = std::min(MAX_FP16_VLM_V16_PER_THD, c_v16);
    const int n_v16_per_wrp = std::min(
        n_v16, MAX_FP16_VLM_V16_PER_THD / c_v16_per_wrp);
    const int c_v256_per_wrp = c_v16_per_wrp / V16_NUM;
    const int c_per_wrp = c_v16_per_wrp * V16_NUM;
    const int n_per_wrp = n_v16_per_wrp * V16_NUM;

    const uint16_t grid_x = static_cast<uint16_t>(CeilDiv(n_v16, n_v16_per_wrp));
    const uint16_t grid_y = static_cast<uint16_t>(CeilDiv(c_v16, c_v16_per_wrp));
    const uint16_t grid_z = static_cast<uint16_t>(B);
    const uint32_t spm_in_addr = SPM_ALLOC.addr(0, spm_in_off);
    const uint32_t spm_out_addr = SPM_ALLOC.addr(0, spm_out_off);

    Kernel_t* kernel = GET_KERNEL(KernelId::TRANSPOSE_NBC_C16);
    TORCH_CHECK(kernel, "rpu_transpose: TRANSPOSE_NBC_C16 kernel not found");
    kernel->set_regs(0, static_cast<uint16_t>(spm_in_addr & 0xFFFF));
    kernel->set_regs(1, static_cast<uint16_t>(spm_in_addr >> 16));
    kernel->set_regs(2, static_cast<uint16_t>(spm_out_addr & 0xFFFF));
    kernel->set_regs(3, static_cast<uint16_t>(spm_out_addr >> 16));
    kernel->set_regs(4, static_cast<uint16_t>(1));
    kernel->set_regs(6, static_cast<uint16_t>(C));
    kernel->set_regs(7, static_cast<uint16_t>(C));
    kernel->set_regs(8, static_cast<uint16_t>(C));
    kernel->set_regs(9, static_cast<uint16_t>(N));
    kernel->set_regs(10, static_cast<uint16_t>(n_v16));
    kernel->set_regs(12, static_cast<uint16_t>(c_per_wrp));
    kernel->set_regs(13, static_cast<uint16_t>(c_per_wrp));
    kernel->set_regs(14, static_cast<uint16_t>(c_v16_per_wrp));
    kernel->set_regs(15, static_cast<uint16_t>(c_v256_per_wrp));
    kernel->set_regs(16, static_cast<uint16_t>(n_per_wrp));
    kernel->set_regs(17, static_cast<uint16_t>(n_per_wrp));
    kernel->set_regs(18, static_cast<uint16_t>(n_v16_per_wrp));
    kernel->set_regs(19, static_cast<uint16_t>(B));
    kernel->set_regs(64, grid_x);
    kernel->set_regs(65, grid_y);
    kernel->set_regs(66, grid_z);

    auto* wq = GET_QUEUE(1);
    wq->enqueu_kernel(*kernel, {grid_x, grid_y, grid_z}, {0});
}
