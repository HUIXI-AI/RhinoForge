#include "rhino_launch_buffer.h"

#include "rhino_launch_program.h" // for Program_t
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <cstdint>
#include <stdlib.h> // for uint16_t, setenv, unsetenv
#include <string.h> // for memcpy
#include <string>   // for string
#include <vector>   // for vector

using namespace ::rhino_lkn;

#define MAX_VLM_CAN_BE_USED 253

void rpu_launch_softmax_c16_kernel(const at::Tensor &input,
                                   at::Tensor &output) {

  TORCH_CHECK(input.dim() == 3 && input.size(0) == 1,
              "softmax_c16 expects input shape [1, N, C]");
  TORCH_CHECK(input.scalar_type() == at::kHalf
                  && output.scalar_type() == at::kHalf,
              "softmax_c16 supports FP16 only");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
              "softmax_c16 expects contiguous input and output");
  TORCH_CHECK(input.sizes() == output.sizes(),
              "softmax_c16 input/output shapes must match");

  auto n = input.size(1);
  auto c = input.size(2);
  TORCH_CHECK(n > 0 && n <= UINT16_MAX,
              "softmax_c16 N must be in [1, 65535], got ", n);
  TORCH_CHECK(c > 0 && c <= 2048 && c % 16 == 0,
              "softmax_c16 C must be 16-aligned and in [16, 2048], got ", c);
  TORCH_CHECK(static_cast<uint64_t>(n) * static_cast<uint64_t>(c)
                  * sizeof(c10::Half) * 2 <=
              SpmAllocator::SPM_PLANNING_BUDGET,
              "softmax_c16 input and output exceed the 7.919 MiB core-0 SPM budget");
  uint64_t dwidth = sizeof(c10::Half);
  uint64_t dwidth_v16 = dwidth * 16;
  uint64_t tail = 0;
  uint64_t tail_v16 = tail == 1 ? 1 : 0;

  uint64_t c_v16 = CeilDiv(c, 16);
  uint64_t c_v256 = CeilDiv(c, 256);
  uint64_t c_v16_tail = c_v16 + tail_v16;
  uint64_t c_tail = c_v16_tail * 16;

  uint64_t in_v16_size = n * c_v16_tail;
  uint64_t out_v16_size = n * c_v16_tail;

  uint64_t num_vlm_per_c = c_v16 + 2;

  uint64_t min_n_per_thd = 1;
  uint64_t min_n_per_warp = NUM_THD_PER_WARP;
  uint64_t min_n_per_launch = NUM_THD_PER_LAUNCH;

  uint64_t max_n_per_thd = MAX_VLM_CAN_BE_USED / num_vlm_per_c;
  uint64_t max_n_per_warp = max_n_per_thd * NUM_THD_PER_WARP;
  uint64_t max_n_per_launch = max_n_per_warp * NUM_WARP_PER_LAUNCH;

  uint64_t n_per_thd = 0;

  uint64_t grid_dim_x = 1;
  uint64_t grid_dim_y = 1;
  uint64_t grid_dim_z = 1;

  if (n <= min_n_per_launch) {
    grid_dim_x = CeilDiv(n, min_n_per_warp);
    n_per_thd = 1;
  } else if (n > min_n_per_launch && n <= max_n_per_launch) {
    grid_dim_x = NUM_WARP_PER_LAUNCH;
    n_per_thd = CeilDiv(n, min_n_per_launch);
  } else if (n > max_n_per_launch) {
    grid_dim_x = CeilDiv(n, max_n_per_warp);
    n_per_thd = max_n_per_thd;
  }

  // 从缓存获取 Kernel (O(1) 数组索引)
  Kernel_t* kernel = GET_KERNEL(KernelId::SOFTMAX_C16_GAUTO);
  TORCH_CHECK(kernel != nullptr, "Failed to get softmax_c16_gauto kernel from cache");

  auto* wq = GET_QUEUE(1);

  LocalSPM_t spm_core0_input(in_v16_size * dwidth_v16, read_write, kStride32B,
                             2, 0);
  RPU_CHECK_SPM(spm_core0_input, in_v16_size * dwidth_v16);
  LocalSPM_t spm_core0_output(out_v16_size * dwidth_v16, read_write, kStride32B,
                              2, 0);
  RPU_CHECK_SPM(spm_core0_output, out_v16_size * dwidth_v16);

  const c10::Half *in = input.data_ptr<c10::Half>();
  c10::Half *out = output.data_ptr<c10::Half>();

  ddr_to_spm(spm_core0_input.get_cpu_ptr(), in, in_v16_size * dwidth_v16);

  kernel->set_regs(0,
                  (uint16_t)((spm_core0_input.get_rpu_addr() / 32) & 0xFFFF));
  kernel->set_regs(1, (uint16_t)((spm_core0_input.get_rpu_addr() / 32) >> 16));
  kernel->set_regs(2,
                  (uint16_t)((spm_core0_output.get_rpu_addr() / 32) & 0xFFFF));
  kernel->set_regs(3, (uint16_t)((spm_core0_output.get_rpu_addr() / 32) >> 16));

  kernel->set_regs(4, (uint16_t)n);
  kernel->set_regs(5, (uint16_t)c);
  kernel->set_regs(6, (uint16_t)c_v16);
  kernel->set_regs(7, (uint16_t)c_v256);
  kernel->set_regs(8, (uint16_t)c_v16_tail);
  kernel->set_regs(9, (uint16_t)c_tail);
  kernel->set_regs(10, (uint16_t)n_per_thd);

  /* grid dim x */
  kernel->set_regs(64, (uint16_t)grid_dim_x);

  /* grid dim y */
  kernel->set_regs(65, (uint16_t)grid_dim_y);

  /* grid dim z */
  kernel->set_regs(66, (uint16_t)grid_dim_z);

  wq->enqueu_kernel(
      *kernel,
      {(uint16_t)grid_dim_x, (uint16_t)grid_dim_y, (uint16_t)grid_dim_z}, {0});

  spm_to_ddr(out, spm_core0_output.get_cpu_ptr(), out_v16_size * dwidth_v16);
}
