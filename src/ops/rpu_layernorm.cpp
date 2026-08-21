#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h" // for Program_t
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cmath> // std::sqrt for the layer_norm high-range ABI (reg20 = sqrt(LC))
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdlib.h> // for uint16_t, setenv, unsetenv
#include <string.h> // for memcpy
#include <string>   // for string
#include <vector>   // for vector

using namespace ::rhino_lkn;

static uint32_t float_to_u32_le(float x) {
  uint32_t u;
  static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4, "Size mismatch");
  std::memcpy(&u, &x, 4); // bitwise copy
  return u;               // On little-endian systems this matches Python "<I"
}

void rpu_launch_layernorm_kernel(const at::Tensor &input, at::Tensor &output,
                                 c10::IntArrayRef normalized_shape,
                                 const at::Tensor &weight,
                                 const at::Tensor &bias, double eps) {

  size_t lc = 1;
  for (auto s : normalized_shape)
    lc *= s;
  size_t n = 1;
  for (auto s : input.sizes())
    n *= s;
  n /= lc;
  size_t lc_v16 = CeilDiv(lc, 16);

  size_t dwidth = sizeof(input.options().dtype());
  size_t lc_tail = lc;
  size_t bank_width = SPM_BANK_SIZE / dwidth;
  size_t input_n_byte_step = lc_tail * dwidth;

  size_t input_bytes = n * input_n_byte_step;
  size_t gamma_bytes = lc * sizeof(c10::Half);
  size_t beta_bytes = lc * sizeof(c10::Half);
  size_t output_bytes = n * input_n_byte_step;
  size_t n_per_thd;
  if (lc <= 512) {
    n_per_thd = 4;
  } else if (lc <= 1024) {
    n_per_thd = 2;
  } else {
    n_per_thd = 1;
  }

  size_t n_per_wrp = NUM_THD_PER_WARP * n_per_thd;

  float one_lcth = 1.0 / lc;
  // High-range fp16 layer_norm ABI: reg12 = fp16(eps*LC) and
  // reg20 = fp16(sqrt(LC)).
  c10::Half eps_lc_half  = static_cast<c10::Half>((float)eps * (float)lc);
  c10::Half sqrt_lc_half = static_cast<c10::Half>(std::sqrt((float)lc));

  // 从缓存获取 Kernel (O(1) 数组索引)
  // Both LAYER_NORM (C>512) and LAYER_NORM_SIMPLE (C<=512) take the same high-range
  // reg ABI (reg12=fp16(eps*LC), reg20=fp16(sqrt(LC)), reg21=has_affine) set below.
  Kernel_t* kernel = (lc > 512)
      ? GET_KERNEL(KernelId::LAYER_NORM)
      : GET_KERNEL(KernelId::LAYER_NORM_SIMPLE);
  TORCH_CHECK(kernel != nullptr, "Failed to get layer_norm kernel from cache");

  auto* wq = GET_QUEUE(1);

  LocalSPM_t spm_core0_input(Align(input_bytes, SPM_BANK_SIZE), read_write,
                             kStride32B, 2, 0);
  RPU_CHECK_SPM(spm_core0_input, Align(input_bytes, SPM_BANK_SIZE));
  LocalSPM_t spm_core0_gamma(Align(gamma_bytes, SPM_BANK_SIZE), read_write,
                             kStride32B, 2, 0);
  RPU_CHECK_SPM(spm_core0_gamma, Align(gamma_bytes, SPM_BANK_SIZE));
  LocalSPM_t spm_core0_beta(Align(beta_bytes, SPM_BANK_SIZE), read_write,
                            kStride32B, 2, 0);
  RPU_CHECK_SPM(spm_core0_beta, Align(beta_bytes, SPM_BANK_SIZE));
  LocalSPM_t spm_core0_output(Align(output_bytes, SPM_BANK_SIZE), read_write,
                              kStride32B, 2, 0);
  RPU_CHECK_SPM(spm_core0_output, Align(output_bytes, SPM_BANK_SIZE));

  c10::Half *inputp = input.data_ptr<c10::Half>();
  c10::Half *gammap = weight.data_ptr<c10::Half>();
  c10::Half *betap = bias.data_ptr<c10::Half>();
  c10::Half *outputp = output.data_ptr<c10::Half>();

  ddr_to_spm(spm_core0_input.get_cpu_ptr(), inputp, input_bytes);
  ddr_to_spm(spm_core0_gamma.get_cpu_ptr(), gammap, gamma_bytes);
  ddr_to_spm(spm_core0_beta.get_cpu_ptr(), betap, beta_bytes);

  uint32_t one_lcth_u32 = float_to_u32_le(one_lcth);

  kernel->set_regs(0, (uint16_t)(spm_core0_input.get_rpu_addr() & 0xFFFF));
  kernel->set_regs(1, (uint16_t)(spm_core0_input.get_rpu_addr() >> 16));
  kernel->set_regs(2, (uint16_t)(spm_core0_output.get_rpu_addr() & 0xFFFF));
  kernel->set_regs(3, (uint16_t)(spm_core0_output.get_rpu_addr() >> 16));
  kernel->set_regs(4, (uint16_t)(spm_core0_gamma.get_rpu_addr() & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(spm_core0_gamma.get_rpu_addr() >> 16));
  kernel->set_regs(6, (uint16_t)(spm_core0_beta.get_rpu_addr() & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(spm_core0_beta.get_rpu_addr() >> 16));
  kernel->set_regs(8, (uint16_t)(n));
  kernel->set_regs(9, (uint16_t)(lc));
  kernel->set_regs(10, (uint16_t)(one_lcth_u32 & 0xFFFF));
  kernel->set_regs(11, (uint16_t)(one_lcth_u32 >> 16));
  kernel->set_regs(12, eps_lc_half.x); // high-range ABI: fp16(eps*LC)
  kernel->set_regs(13, (uint16_t)(0)); // has_skip
  kernel->set_regs(14, (uint16_t)(0)); // skip_base
  kernel->set_regs(15, (uint16_t)(0));
  kernel->set_regs(16, (uint16_t)(input_n_byte_step & 0xFFFF));
  kernel->set_regs(17, (uint16_t)(input_n_byte_step >> 16));
  kernel->set_regs(18, (uint16_t)(n_per_thd));
  kernel->set_regs(19, (uint16_t)(n_per_wrp));
  kernel->set_regs(20, sqrt_lc_half.x); // high-range ABI: fp16(sqrt(LC)); required or rstd=0
  kernel->set_regs(21, (uint16_t)1);    // has_affine (launcher always loads gamma+beta)

  wq->enqueu_kernel(
      *kernel, {(uint16_t)CeilDiv(n, n_per_wrp), (uint16_t)1, (uint16_t)1}, {0});

  spm_to_ddr(outputp, spm_core0_output.get_cpu_ptr(), output_bytes);
}

// ============================================================================
// Direct Address SPM LayerNorm Kernel (for fused execution)
// ============================================================================
// All data already in SPM. No DDR↔SPM copies.
// Uses same layer_norm / layer_norm_simple kernel as DDR version.
// Supports batch mode and immediate mode (8-core broadcast).
//
// Register layout:
//   reg[0-1]:   input_base   (uint32 SPM addr)
//   reg[2-3]:   output_base  (uint32 SPM addr)
//   reg[4-5]:   gamma_base   (uint32 SPM addr)
//   reg[6-7]:   beta_base    (uint32 SPM addr)
//   reg[8]:     n            (num rows)
//   reg[9]:     lc           (num cols, normalized dim)
//   reg[10-11]: one_lcth     (float32, 1.0/lc)
//   reg[12]:    epsilon      (fp16)
//   reg[13]:    has_skip     (bool)
//   reg[14-15]: skip_base    (uint32 SPM addr, only if has_skip)
//   reg[16-17]: input_n_byte_step (uint32, lc * dwidth bytes)
//   reg[18]:    n_per_thd
//   reg[19]:    n_per_wrp
//   grid: [CeilDiv(n, n_per_wrp), 1, 1]
// ============================================================================

void rpu_launch_layernorm_spm_kernel(
    uint32_t input_spm_addr,           // Input SPM address (core 0 base)
    uint32_t output_spm_addr,          // Output SPM address (core 0 base)
    uint32_t gamma_spm_addr,           // Gamma (weight) SPM address (core 0 base)
    uint32_t beta_spm_addr,            // Beta (bias) SPM address (core 0 base)
    int64_t M,                         // Number of rows (per core)
    int64_t C,                         // Number of columns (normalized dim)
    double eps,
    bool has_skip,                     // Skip connection: output = LN(input + skip)
    uint32_t skip_spm_addr,            // Skip input SPM address (only if has_skip)
    int num_cores)                     // Number of cores
{
  size_t dwidth = sizeof(c10::Half);
  size_t input_n_byte_step = C * dwidth;

  // Populate the operator launch geometry.
  size_t n_per_thd;
  if (C <= 512) {
    n_per_thd = 4;
  } else if (C <= 1024) {
    n_per_thd = 2;
  } else {
    n_per_thd = 1;
  }
  size_t n_per_wrp = NUM_THD_PER_WARP * n_per_thd;

  // Precompute constants
  float one_lcth = 1.0f / C;
  uint32_t one_lcth_u32 = float_to_u32_le(one_lcth);
  // High-range fp16 layer_norm ABI: reg12 = fp16(eps*C) and
  // reg20 = fp16(sqrt(C)).
  c10::Half eps_lc_half  = static_cast<c10::Half>((float)eps * (float)C);
  c10::Half sqrt_lc_half = static_cast<c10::Half>(std::sqrt((float)C));

  // Select kernel based on C
  KernelId kid = (C > 512) ? KernelId::LAYER_NORM : KernelId::LAYER_NORM_SIMPLE;

  uint16_t grid_x = (uint16_t)CeilDiv((size_t)M, n_per_wrp);

  auto setup_regs = [=](Kernel_t* kernel) {
    kernel->set_regs(0,  (uint16_t)(input_spm_addr & 0xFFFF));
    kernel->set_regs(1,  (uint16_t)(input_spm_addr >> 16));
    kernel->set_regs(2,  (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(3,  (uint16_t)(output_spm_addr >> 16));
    kernel->set_regs(4,  (uint16_t)(gamma_spm_addr & 0xFFFF));
    kernel->set_regs(5,  (uint16_t)(gamma_spm_addr >> 16));
    kernel->set_regs(6,  (uint16_t)(beta_spm_addr & 0xFFFF));
    kernel->set_regs(7,  (uint16_t)(beta_spm_addr >> 16));
    kernel->set_regs(8,  (uint16_t)(M));
    kernel->set_regs(9,  (uint16_t)(C));
    kernel->set_regs(10, (uint16_t)(one_lcth_u32 & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(one_lcth_u32 >> 16));
    kernel->set_regs(12, eps_lc_half.x);   // ABI: fp16(eps*LC)
    kernel->set_regs(13, (uint16_t)(has_skip ? 1 : 0));
    kernel->set_regs(14, (uint16_t)(skip_spm_addr & 0xFFFF));
    kernel->set_regs(15, (uint16_t)(skip_spm_addr >> 16));
    kernel->set_regs(16, (uint16_t)(input_n_byte_step & 0xFFFF));
    kernel->set_regs(17, (uint16_t)(input_n_byte_step >> 16));
    kernel->set_regs(18, (uint16_t)(n_per_thd));
    kernel->set_regs(19, (uint16_t)(n_per_wrp));
    kernel->set_regs(20, sqrt_lc_half.x);  // ABI: fp16(sqrt(LC)); required or rstd=0
    kernel->set_regs(21, (uint16_t)1);     // ABI: has_affine; 0 -> gamma/beta skipped
  };

  Kernel_t* kernel = GET_KERNEL(kid);
  TORCH_CHECK(kernel != nullptr, "Failed to get LayerNorm SPM kernel");
  kernel->reset_regs();
  setup_regs(kernel);

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list(num_cores);
  for (int i = 0; i < num_cores; ++i) core_list[i] = (uint8_t)i;
  wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1},
                    core_list);
}

// ============================================================================
// Direct Address SPM LayerNorm bf16 Kernel — mixed precision
// ============================================================================
// Same public register layout as the fp16 LayerNorm launcher. All SPM inputs
// and outputs remain fp16 at the wrapper boundary; the combined operator asset
// owns the mixed-precision implementation. `epsilon` is encoded as fp16 and
// `1/lc` as fp32.
// ============================================================================

void rpu_launch_layernorm_bf16_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    uint32_t gamma_spm_addr,
    uint32_t beta_spm_addr,
    int64_t M,
    int64_t C,
    double eps,
    bool has_skip,
    uint32_t skip_spm_addr,
    int num_cores)
{
  size_t dwidth = sizeof(c10::BFloat16);
  size_t input_n_byte_step = C * dwidth;

  // Populate the operator launch geometry using the fp16-compatible policy.
  size_t n_per_thd;
  if (C <= 512) {
    n_per_thd = 4;
  } else if (C <= 1024) {
    n_per_thd = 2;
  } else {
    n_per_thd = 1;
  }
  size_t n_per_wrp = NUM_THD_PER_WARP * n_per_thd;

  // Precompute constants.
  float one_lcth = 1.0f / C;
  uint32_t one_lcth_u32 = float_to_u32_le(one_lcth);
  // Epsilon stays fp16-encoded. The bf16 operator is mixed-precision: only the
  // `var = Σ(x-mean)²/n` write-back goes through a fp16→bf16 cast, and
  // cfg6 (`rstd`) reads `a`=var as bf16 while `b` (eps slot) remains fp16.
  // Encoding eps as bf16 would mis-feed cfg6's fp16 b input.
  c10::Half eps_half = static_cast<c10::Half>(eps);

  uint16_t grid_x = (uint16_t)CeilDiv((size_t)M, n_per_wrp);

  Kernel_t* kernel = GET_KERNEL(KernelId::LAYER_NORM_BF16);
  TORCH_CHECK(kernel != nullptr, "Failed to get LayerNorm bf16 SPM kernel");
  kernel->reset_regs();
  kernel->set_regs(0,  (uint16_t)(input_spm_addr & 0xFFFF));
  kernel->set_regs(1,  (uint16_t)(input_spm_addr >> 16));
  kernel->set_regs(2,  (uint16_t)(output_spm_addr & 0xFFFF));
  kernel->set_regs(3,  (uint16_t)(output_spm_addr >> 16));
  kernel->set_regs(4,  (uint16_t)(gamma_spm_addr & 0xFFFF));
  kernel->set_regs(5,  (uint16_t)(gamma_spm_addr >> 16));
  kernel->set_regs(6,  (uint16_t)(beta_spm_addr & 0xFFFF));
  kernel->set_regs(7,  (uint16_t)(beta_spm_addr >> 16));
  kernel->set_regs(8,  (uint16_t)(M));
  kernel->set_regs(9,  (uint16_t)(C));
  kernel->set_regs(10, (uint16_t)(one_lcth_u32 & 0xFFFF));
  kernel->set_regs(11, (uint16_t)(one_lcth_u32 >> 16));
  kernel->set_regs(12, eps_half.x);
  kernel->set_regs(13, (uint16_t)(has_skip ? 1 : 0));
  kernel->set_regs(14, (uint16_t)(skip_spm_addr & 0xFFFF));
  kernel->set_regs(15, (uint16_t)(skip_spm_addr >> 16));
  kernel->set_regs(16, (uint16_t)(input_n_byte_step & 0xFFFF));
  kernel->set_regs(17, (uint16_t)(input_n_byte_step >> 16));
  kernel->set_regs(18, (uint16_t)(n_per_thd));
  kernel->set_regs(19, (uint16_t)(n_per_wrp));

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list(num_cores);
  for (int i = 0; i < num_cores; ++i) core_list[i] = (uint8_t)i;
  wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1},
                    core_list);
}
