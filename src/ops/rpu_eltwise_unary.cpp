#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h" // for Program_t
#include "rhino_launch_queue.h"

#include "rpu_ops.h"
#include <c10/util/Half.h>
#include <stdlib.h> // for uint16_t, setenv, unsetenv
#include <string.h> // for memcpy
#include <string>   // for string
#include <vector>   // for vector

using namespace ::rhino_lkn;

void rpu_launch_eltwise_unary_kernel(const at::Tensor &input,
                                     at::Tensor &output, ValuOpType op_type) {

  size_t dwidth = sizeof(c10::Half);
  size_t n = 1;
  for (auto s : input.sizes())
    n *= s;
  size_t normal_blk_n = 100 * 256;
  size_t blk_cnt = CeilDiv(n, normal_blk_n);
  size_t last_blk_n = n - (blk_cnt - 1) * normal_blk_n;

  // 从缓存获取 DDR 版本 Kernel (O(1) 数组索引)
  Kernel_t* kernel = GET_KERNEL(KernelId::UNARY_DDR);
  TORCH_CHECK(kernel != nullptr, "Failed to get unary kernel from cache");
  kernel->reset_regs();

  auto* wq = GET_QUEUE(1);

  // 直接使用 tensor 指针 (已在 DDR)
  c10::Half *ap = input.data_ptr<c10::Half>();
  c10::Half *out = output.data_ptr<c10::Half>();

  // Flush 输入数据到 DDR
  rpu_ddr_flush(ap);
  rpu_ddr_flush(out);

  // 获取 DDR 设备地址 (v128 = 256B 单位)
  uint64_t a_addr_v128 = RpuGetDevAddr(ap) >> 8;
  uint64_t out_addr_v128 = RpuGetDevAddr(out) >> 8;

  kernel->set_regs(0, (uint16_t)normal_blk_n);
  kernel->set_regs(1, (uint16_t)last_blk_n);
  kernel->set_regs(4, (uint16_t)(a_addr_v128 & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(a_addr_v128 >> 16));
  kernel->set_regs(6, (uint16_t)(out_addr_v128 & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(out_addr_v128 >> 16));
  // block_data_stride 也要转换为 v128 单位，供算子与 DDR 基址相加。
  uint32_t block_data_stride_v128 = (normal_blk_n * dwidth) >> 8;
  kernel->set_regs(8, (uint16_t)(block_data_stride_v128 & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(block_data_stride_v128 >> 16));

  c10::Half scale_a = 1.0;
  c10::Half scale_b = 1.0;
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;

  if (op_type == ValuOpType::HARDSIGMOID) {
    scale_a = 0.16666666666666666;
    scale_b = 0.5;
  }

  kernel->set_regs(56, (scale_a.x));
  kernel->set_regs(57, (scale_b.x));
  kernel->set_regs(58, (scale_r.x));
  kernel->set_regs(59, (clip_max.x));
  kernel->set_regs(60, (clip_min.x));
  kernel->set_regs(61, (uint16_t)0x3);  // DDR 模式标志

  const auto& op_info = get_valu_op_info(op_type);
  kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
  kernel->set_regs(63, (uint16_t)(op_info.custom_op));

  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, {0});

  // Flush 输出缓存以读取结果
  rpu_ddr_flush(out);
}

// =============================================================================
// SPM Unary Kernel (for fused decoder layer MLP)
// =============================================================================
// Input/Output data in SPM, runs on all 8 cores.
// Each core processes its own portion of the data.
//
// Differences from DDR version:
//   - reg[61] = 0x1 (SPM mode) instead of 0x3 (DDR mode)
//   - Addresses are 32-bit SPM addresses (not v128 units)
//   - block_data_stride is in bytes (not v128 units)
// =============================================================================

#define NUM_CORES_UNARY 8

void rpu_launch_eltwise_unary_spm_kernel(
    uint32_t input_spm_addr,           // SPM input address (core 0 base)
    uint32_t output_spm_addr,          // SPM output address (core 0 base), can be same as input for in-place
    int64_t num_elements,              // Total elements per core
    ValuOpType op_type,
    GeluMode gelu_mode,
    int num_cores)                     // number of cores
{
  size_t dwidth = sizeof(c10::Half);
  size_t n = num_elements;
  size_t normal_blk_n = 100 * 256;
  size_t blk_cnt = CeilDiv(n, normal_blk_n);
  size_t last_blk_n = n - (blk_cnt - 1) * normal_blk_n;

  // SPM addresses (32-bit, NOT shifted)
  uint32_t in_addr = input_spm_addr;
  uint32_t out_addr = output_spm_addr;

  // block_data_stride in bytes for SPM (NOT shifted)
  uint32_t block_data_stride = normal_blk_n * dwidth;

  // Scale parameters
  c10::Half scale_a = 1.0;
  c10::Half scale_b = 1.0;
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;

  if (op_type == ValuOpType::HARDSIGMOID) {
    scale_a = 0.16666666666666666;
    scale_b = 0.5;
  }

  const auto& op_info = get_valu_op_info(op_type);

  // 定义寄存器配置 lambda
  auto setup_regs = [=](Kernel_t* kernel) {
    kernel->set_regs(0, (uint16_t)normal_blk_n);
    kernel->set_regs(1, (uint16_t)last_blk_n);
    kernel->set_regs(4, (uint16_t)(in_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(in_addr >> 16));
    kernel->set_regs(6, (uint16_t)(out_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(out_addr >> 16));

    kernel->set_regs(8, (uint16_t)(block_data_stride & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(block_data_stride >> 16));

    kernel->set_regs(56, (scale_a.x));
    kernel->set_regs(57, (scale_b.x));
    kernel->set_regs(58, (scale_r.x));
    kernel->set_regs(59, (clip_max.x));
    kernel->set_regs(60, (clip_min.x));
    kernel->set_regs(61, (uint16_t)0x1);  // SPM mode flag

    kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
    kernel->set_regs(63, (uint16_t)(op_info.custom_op));
  };

  KernelId kid = KernelId::UNARY_SPM;
  if (gelu_mode == GeluMode::TANH) {
    kid = KernelId::UNARY_GELU_TANH_SPM;
  } else if (gelu_mode == GeluMode::ERF) {
    kid = KernelId::UNARY_GELU_ERF_SPM;
  } else if (op_type == ValuOpType::SOFTPLUS) {
    kid = KernelId::UNARY_SOFTPLUS_SPM;
  }

  // 普通模式：立即执行
  Kernel_t* kernel = GET_KERNEL(kid);
  TORCH_CHECK(kernel != nullptr, "Failed to get unary SPM kernel from cache");
  kernel->reset_regs();
  setup_regs(kernel);

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1},
                    core_list);
}

// SwiGLU activation fusion: out = silu(x) * y, flat per-core elementwise.
// Host register layout:
//   [0]=normal_blk_n  [1]=last_blk_n  [4/5]=x  [6/7]=y  [8/9]=out
//   gridDim.x = blk_cnt (via enqueu_kernel). All SPM addrs absolute core-0 base.
// param0 is consumed as signed 16-bit by the blob. The current 256-aligned
// 32,512-element block keeps both normal and tail counts in range.
void rpu_launch_silu_mul_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    uint32_t out_spm_addr,
    int64_t num_elements,
    int num_cores)
{
  TORCH_CHECK(num_elements > 0,
              "silu_mul: num_elements must be positive, got ", num_elements);
  size_t n = (size_t)num_elements;
  // The kernel ABI consumes param0 as signed 16-bit, so keep every block
  // below INT16_MAX while preserving the 256-element alignment.
  size_t normal_blk_n = 256 * 127;
  size_t blk_cnt = CeilDiv(n, normal_blk_n);
  size_t last_blk_n = n - (blk_cnt - 1) * normal_blk_n;

  // The blob advances each block by param0 * sizeof(fp16) * blkid.x, so one
  // grid launch covers every signed-16-safe block without host-side relaunches.
  Kernel_t* kernel = GET_KERNEL(KernelId::LLAMA_SILU_MUL);
  TORCH_CHECK(kernel != nullptr,
              "Failed to get llama_silu_mul SPM kernel from cache");
  kernel->reset_regs();
  kernel->set_regs(0, (uint16_t)normal_blk_n);
  kernel->set_regs(1, (uint16_t)last_blk_n);
  kernel->set_regs(4, (uint16_t)(x_spm_addr & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(x_spm_addr >> 16));
  kernel->set_regs(6, (uint16_t)(y_spm_addr & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(y_spm_addr >> 16));
  kernel->set_regs(8, (uint16_t)(out_spm_addr & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(out_spm_addr >> 16));

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
  wq->enqueu_kernel(
      *kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, core_list);
}
