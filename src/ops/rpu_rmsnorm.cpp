#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <ATen/ATen.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <utility>

using namespace ::rhino_lkn;

// Helper: convert half to uint16_t raw bits
static uint16_t half_to_u16(c10::Half h) {
  return h.x;
}

void rpu_launch_rmsnorm_kernel(const at::Tensor &input, at::Tensor &output,
                               const at::Tensor &weight, double eps) {
  // Legacy DDR launcher requires 32-bit byte addresses.
  // The public eager op bypasses it.
  // RMSNorm: y = w * x * rsqrt(mean(x^2) + eps)
  // Input shape: [N, C] or [*, C] where C is the normalized dimension
  // Weight shape: [C]

#if RPU_PROFILE_ENABLED
  KernelProfileData kp;
  PROFILE_START();
#endif

  TORCH_CHECK(input.is_contiguous(), "RMSNorm input must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "RMSNorm weight must be contiguous");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
              "RMSNorm only supports FP16 input");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half,
              "RMSNorm only supports FP16 weight");

  // Get dimensions
  int64_t c = input.size(-1);  // normalized dimension (last dim)
  int64_t n = input.numel() / c;  // batch dimension (all dims except last)

  TORCH_CHECK(weight.numel() == c,
              "RMSNorm weight size must match last dim of input");

  size_t dwidth = sizeof(c10::Half);

  // Get kernel from cache
  Kernel_t* kernel = GET_KERNEL(KernelId::RMS_NORM_BF16);
  TORCH_CHECK(kernel != nullptr, "Failed to get llama_rms_norm_bf16 kernel");

  // Reset kernel registers
  kernel->reset_regs();

  auto* wq = GET_QUEUE(1);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.preprocess_ms);
#endif

  // Get tensor data pointers (tensors are already in DDR)
  c10::Half *inputp = input.data_ptr<c10::Half>();
  c10::Half *weightp = weight.data_ptr<c10::Half>();
  c10::Half *outputp = output.data_ptr<c10::Half>();

  // Flush DDR before kernel execution
  rpu_ddr_flush(inputp);
  rpu_ddr_flush(weightp);
  rpu_ddr_flush(outputp);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.flush_ms);
#endif

  // Calculate parameters
  c10::Half c_reciprocal = static_cast<c10::Half>(1.0f / c);
  uint32_t blk_stride = (c * dwidth);
  c10::Half eps_half = static_cast<c10::Half>(eps);

  // DDR addresses from tensor pointers
  uint64_t x_base = RpuGetDevAddr(inputp);
  uint64_t y_base = RpuGetDevAddr(outputp);
  uint64_t w_base = RpuGetDevAddr(weightp);
  constexpr uint64_t kAG1BAddressLimit = 1ULL << 32;
  TORCH_CHECK(x_base < kAG1BAddressLimit
                  && y_base < kAG1BAddressLimit
                  && w_base < kAG1BAddressLimit,
              "legacy DDR RMSNorm kernel cannot address buffers at or above 4 GiB; "
              "use rpu_rmsnorm's eager fallback or the direct SPM launcher");

  kernel->set_regs(0, (uint16_t)(c));
  kernel->set_regs(1, half_to_u16(c_reciprocal));
  kernel->set_regs(2, (uint16_t)(blk_stride));
  kernel->set_regs(3, (uint16_t)(0));  // reserved

  // DDR addresses: shift right by 8 for 256B alignment
  kernel->set_regs(4, (uint16_t)((x_base >> 8) & 0xFFFF));
  kernel->set_regs(5, (uint16_t)((x_base >> 24) & 0xFFFF));

  kernel->set_regs(6, (uint16_t)((y_base >> 8) & 0xFFFF));
  kernel->set_regs(7, (uint16_t)((y_base >> 24) & 0xFFFF));

  kernel->set_regs(8, (uint16_t)((w_base >> 8) & 0xFFFF));
  kernel->set_regs(9, (uint16_t)((w_base >> 24) & 0xFFFF));

  kernel->set_regs(10, half_to_u16(eps_half));

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.set_regs_ms);
#endif

  // Launch kernel with grid = [n, 1, 1]
  wq->enqueu_kernel(*kernel, {(uint16_t)n, (uint16_t)1, (uint16_t)1}, {0});

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.enqueue_ms);
#endif

  // Flush DDR after kernel writes output
  rpu_ddr_flush(outputp);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.postflush_ms);
  kp.total_ms = kp.preprocess_ms + kp.flush_ms + kp.set_regs_ms + kp.enqueue_ms + kp.postflush_ms;
  g_last_kernel_profile = kp;
#endif
}

// at::Tensor rpu_rmsnorm(const at::Tensor &input, c10::IntArrayRef normalized_shape,
//                        const c10::optional<at::Tensor> &weight_opt, double eps) {
at::Tensor rpu_rmsnorm(const at::Tensor &input, c10::IntArrayRef normalized_shape,
                       const at::Tensor &weight, double eps) {
#if RPU_PROFILE_ENABLED
  WrapperProfileData wp;
  PROFILE_START();
#endif

  // Validate input
  TORCH_CHECK(input.device().type() == at::DeviceType::PrivateUse1,
              "rpu_rmsnorm: input must be on device");
  TORCH_CHECK(weight.device().type() == at::DeviceType::PrivateUse1,
              "rpu_rmsnorm: weight must be on device");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half,
              "rpu_rmsnorm: only FP16 input is supported");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Half,
              "rpu_rmsnorm: only FP16 weight is supported");

  // // Get weight tensor
  // TORCH_CHECK(weight_opt.has_value(), "rpu_rmsnorm requires weight tensor");
  // auto weight = weight_opt.value();

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(wp.preprocess_ms);
#endif

  // Ensure input is contiguous
  auto input_contig = input.contiguous();
  auto weight_contig = weight.contiguous();

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(wp.contiguous_ms);
#endif

  auto result = rpu_empty_strided(input_contig.sizes(), input_contig.strides(),
                                  input_contig.scalar_type(), input_contig.layout(),
                                  input_contig.device(), false);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(wp.alloc_ms);
#endif

  // The standalone DDR kernel requires output addresses below 4 GiB.
  // Eager calls use a CPU fallback; fused models use the SPM launcher.
  auto cpu_input = rpu_to_cpu_zerocopy(input_contig);
  auto cpu_weight = rpu_to_cpu_zerocopy(weight_contig);
  auto cpu_result = at::rms_norm(
      cpu_input.to(at::ScalarType::Float), normalized_shape,
      cpu_weight.to(at::ScalarType::Float), eps);
  auto cpu_output = rpu_to_cpu_zerocopy(result, /*flush=*/false);
  cpu_output.copy_(cpu_result.to(input_contig.scalar_type()));
  rpu_ddr_flush_force(result.data_ptr());

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(wp.kernel_ms);
  wp.total_ms = wp.preprocess_ms + wp.contiguous_ms + wp.alloc_ms + wp.kernel_ms;

  // Get kernel profile data
  const KernelProfileData& kp = g_last_kernel_profile;

  // Use standard profile macro
  PROFILE_RECORD_AND_PRINT("rmsnorm", g_profile_rmsnorm, wp, kp);
#endif

  return result;
}

RpuRmsNormSpmContract rpu_snapshot_rmsnorm_spm_contract(
    uint32_t certified_capability) {
  TORCH_CHECK(
      rpu_rmsnorm_capability_valid(certified_capability),
      "RMSNorm certified capability must include BASE and contain only "
      "BASE/V16/V32; got ", certified_capability);
  const auto& cache = KernelCache::instance();
  uint32_t loaded_capability = RPU_RMSNORM_CAP_BASE;
  if (cache.has_loaded(KernelId::RMS_NORM_SPM_V16)) {
    loaded_capability |= RPU_RMSNORM_CAP_V16;
  }
  if (cache.has_loaded(KernelId::RMS_NORM_SPM_V32)) {
    loaded_capability |= RPU_RMSNORM_CAP_V32;
  }
  return {
      RPU_RMSNORM_CAP_BASE |
      (certified_capability & loaded_capability &
       ~RPU_RMSNORM_CAP_BASE)};
}

// ============================================================================
// Direct Address SPM RMSNorm Kernel
// ============================================================================
// Same as rpu_launch_rmsnorm_spm_unified_kernel but accepts SPM addresses directly
// instead of using SpmAddressCache with char selectors.
// ============================================================================

// RMSNorm multirow variants V16/V32 use a cold owner capability and an exact
// (M,C)-dependent route in the COMPLETE physical manifest. Shared callers
// without an explicit route retain BASE.
// The canonical V16/V32 register contract uses eight workgroups:
//   grid.x = 8, reg[3] = M/8, C <= 8192.
// V32 requires M%32 == 0; V16 requires M%16 == 0.
//
// Selector IDs are append-only physical-manifest ABI. Legacy selectors 2/3
// retain grid=M/V and reg[3]=V; selectors 5/6 use canonical fixed-grid dispatch.
// Do not change an existing selector's meaning without changing the descriptor
// identity that binds its execution semantics.
void rpu_launch_rmsnorm_spm_kernel(
    uint32_t input_spm_addr,           // Input SPM address (core 0 base)
    uint32_t output_spm_addr,          // Output SPM address (core 0 base, can be same for in-place)
    uint32_t weight_spm_addr,          // Weight SPM address (core 0 base)
    int64_t M,                         // Number of rows (seq_len or seq_len * local_heads)
    int64_t C,                         // Number of columns (hidden_size or head_dim)
    double eps,
    RpuRmsNormSpmRoute route,
    int num_cores)
{
  TORCH_CHECK(num_cores >= 1 && num_cores <= UNIFIED_NUM_CORES,
              "RMSNorm SPM num_cores must be in [1,8], got ", num_cores);
  TORCH_CHECK(M > 0, "RMSNorm rows must be positive, got ", M);
  TORCH_CHECK(
      C > 0 && C % 16 == 0 && C <= RPU_RMSNORM_MAX_COLS,
      "RMSNorm width must be 16-aligned and its FP16 row stride must fit uint16 SCM registers; got ",
      C, " columns");
  size_t dwidth = sizeof(c10::Half);

  c10::Half c_reciprocal = static_cast<c10::Half>(1.0f / C);
  uint32_t blk_stride = (C * dwidth);
  c10::Half eps_half = static_cast<c10::Half>(eps);
  uint16_t c_reciprocal_u16 = half_to_u16(c_reciprocal);
  uint16_t eps_u16 = half_to_u16(eps_half);

  // Use a vector route only for M divisible by its row width; do not split a
  // remainder into a separate BASE launch here. AUTO chooses the widest available
  // capability: V32 for M%32 == 0, otherwise V16 for M%16 == 0, otherwise BASE.
  // FIXED32 does not include V16, so incompatible rows must return to BASE.
  Kernel_t* kernel = nullptr;
  uint16_t grid_x = 0;
  uint16_t reg3   = 0;
  switch (route) {
    case RpuRmsNormSpmRoute::QWEN3VL_V64:
      TORCH_CHECK(num_cores == UNIFIED_NUM_CORES && M > 0 && M % 64 == 0 &&
                      M / 8 <= RPU_RMSNORM_SCM_U16_MAX &&
                      C > 0 && C <= 8192 && C % 16 == 0,
                  "Qwen3-VL V64 RMSNorm route requires positive 64-aligned rows "
                  "and 16-aligned width <=8192");
      kernel = GET_KERNEL(KernelId::QWEN3VL_RMS_NORM_MULTIWARP_V64);
      grid_x = 8;
      reg3 = static_cast<uint16_t>(M / 8);
      break;
    case RpuRmsNormSpmRoute::BASE:
      TORCH_CHECK(
          M <= RPU_RMSNORM_SCM_U16_MAX,
          "RMSNorm BASE row grid must fit uint16; got ", M, " rows");
      kernel = GET_KERNEL(KernelId::RMS_NORM_BF16_SPM);
      grid_x = static_cast<uint16_t>(M);
      break;
    case RpuRmsNormSpmRoute::NEWTON:
      TORCH_CHECK(M <= RPU_RMSNORM_SCM_U16_MAX,
                  "Newton RMSNorm BASE row grid must fit uint16; got ", M);
      kernel = GET_KERNEL(KernelId::RMS_NORM_NEWTON_SPM);
      grid_x = static_cast<uint16_t>(M);
      reg3 = 1;  // BASE ABI.
      break;
    case RpuRmsNormSpmRoute::NEWTON_V16_GRID8:
    case RpuRmsNormSpmRoute::NEWTON_V32_GRID8: {
      const int width = route == RpuRmsNormSpmRoute::NEWTON_V32_GRID8 ? 32 : 16;
      TORCH_CHECK(M % width == 0 && M / 8 <= RPU_RMSNORM_SCM_U16_MAX &&
                      C <= 8192,
                  "Newton RMSNorm vector route requires rows divisible by ",
                  width, ", uint16 rows/8, and width <=8192; got rows=", M,
                  " width=", C);
      kernel = GET_KERNEL(width == 32 ? KernelId::RMS_NORM_SPM_NEWTON_V32
                                     : KernelId::RMS_NORM_SPM_NEWTON_V16);
      grid_x = 8;
      reg3 = static_cast<uint16_t>(M / 8);
      break;
    }
    case RpuRmsNormSpmRoute::V16:
      // Append-only legacy descriptor semantics. New plans resolve to
      // V16_GRID8 instead.
      TORCH_CHECK(
          M % 16 == 0 && M / 16 <= RPU_RMSNORM_SCM_U16_MAX && C <= 8192,
          "RMSNorm V16 route requires rows divisible by 16 and a uint16 "
          "row grid, with width <=8192; got rows=", M, " width=", C);
      kernel = GET_KERNEL(KernelId::RMS_NORM_SPM_V16);
      TORCH_CHECK(kernel != nullptr,
                  "RMSNorm V16 route is selected but the kernel is unavailable");
      grid_x = static_cast<uint16_t>(M / 16);
      reg3 = 16;
      break;
    case RpuRmsNormSpmRoute::V16_GRID8:
    case RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8:
      TORCH_CHECK(route != RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8 ||
                      num_cores == UNIFIED_NUM_CORES,
                  "HyVLA bitcopy RMSNorm requires eight cores");
      TORCH_CHECK(M > 0 && M % 16 == 0 &&
                      M / 8 <= RPU_RMSNORM_SCM_U16_MAX &&
                      C > 0 && C <= 8192,
                  "RMSNorm V16_GRID8 route requires positive rows divisible "
                  "by 16, rows/8 representable as uint16, and positive "
                  "width <=8192; got rows=", M, " width=", C);
      kernel = GET_KERNEL(route == RpuRmsNormSpmRoute::HYVLA_V16_BITCOPY_GRID8
          ? KernelId::HYVLA_EXPERT_RMSNORM_V16_BITCOPY : KernelId::RMS_NORM_SPM_V16);
      TORCH_CHECK(
          kernel != nullptr,
          "RMSNorm V16_GRID8 route is selected but the kernel is unavailable");
      grid_x = 8;
      reg3 = static_cast<uint16_t>(M / 8);
      break;
    case RpuRmsNormSpmRoute::V32:
      // Append-only legacy descriptor semantics. New plans resolve to
      // V32_GRID8 instead.
      TORCH_CHECK(
          M % 32 == 0 && M / 32 <= RPU_RMSNORM_SCM_U16_MAX && C <= 8192,
          "RMSNorm V32 route requires rows divisible by 32 and a uint16 "
          "row grid, with width <=8192; got rows=", M, " width=", C);
      kernel = GET_KERNEL(KernelId::RMS_NORM_SPM_V32);
      TORCH_CHECK(kernel != nullptr,
                  "RMSNorm V32 route is selected but the kernel is unavailable");
      grid_x = static_cast<uint16_t>(M / 32);
      reg3 = 32;
      break;
    case RpuRmsNormSpmRoute::V32_GRID8:
      TORCH_CHECK(M > 0 && M % 32 == 0 &&
                      M / 8 <= RPU_RMSNORM_SCM_U16_MAX &&
                      C > 0 && C <= 8192,
                  "RMSNorm V32_GRID8 route requires positive rows divisible "
                  "by 32, rows/8 representable as uint16, and positive "
                  "width <=8192; got rows=", M, " width=", C);
      kernel = GET_KERNEL(KernelId::RMS_NORM_SPM_V32);
      TORCH_CHECK(
          kernel != nullptr,
          "RMSNorm V32_GRID8 route is selected but the kernel is unavailable");
      grid_x = 8;
      reg3 = static_cast<uint16_t>(M / 8);
      break;
    default:
      TORCH_CHECK(false, "unknown RMSNorm SPM route ", static_cast<int64_t>(route));
  }
  TORCH_CHECK(kernel != nullptr, "Failed to get RMSNorm SPM kernel");
  kernel->reset_regs();
  kernel->set_regs(0,  (uint16_t)C);
  kernel->set_regs(1,  c_reciprocal_u16);
  kernel->set_regs(2,  (uint16_t)blk_stride);
  kernel->set_regs(3,  reg3);
  kernel->set_regs(4,  (uint16_t)(input_spm_addr & 0xFFFF));
  kernel->set_regs(5,  (uint16_t)(input_spm_addr >> 16));
  kernel->set_regs(6,  (uint16_t)(output_spm_addr & 0xFFFF));
  kernel->set_regs(7,  (uint16_t)(output_spm_addr >> 16));
  kernel->set_regs(8,  (uint16_t)(weight_spm_addr & 0xFFFF));
  kernel->set_regs(9,  (uint16_t)(weight_spm_addr >> 16));
  kernel->set_regs(10, eps_u16);

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> cores;
  for (int core = 0; core < num_cores; ++core) cores.push_back(core);
  wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1},
                    cores);
}

// Runtime llama_rms_norm_nogamma ABI. The gain was folded into the
// downstream projection in FP32 before quantization or TP swizzling.
RpuRmsNormSpmContract rpu_snapshot_rmsnorm_nogamma_spm_contract() {
  const auto& cache = KernelCache::instance();
  TORCH_CHECK(cache.has_loaded(KernelId::RMSNORM_NOGAMMA),
              "folded RMSNorm requires the MAIN no-gamma BASE payload");
  uint32_t capability = RPU_RMSNORM_CAP_BASE;
  if (cache.has_loaded(KernelId::RMSNORM_NOGAMMA_V16)) capability |= RPU_RMSNORM_CAP_V16;
  if (cache.has_loaded(KernelId::RMSNORM_NOGAMMA_V32)) capability |= RPU_RMSNORM_CAP_V32;
  return {capability};
}

void rpu_launch_rmsnorm_nogamma_spm_kernel(
    uint32_t input_spm_addr, uint32_t output_spm_addr,
    int64_t M, int64_t C, double eps, RpuRmsNormSpmRoute route, int num_cores) {
  TORCH_CHECK(num_cores == 8 && M > 0 && C > 0 && C % 16 == 0 &&
                  C <= RPU_RMSNORM_MAX_COLS,
              "no-gamma RMSNorm requires TP8, positive rows and valid FP16 width");
  KernelId id = KernelId::RMSNORM_NOGAMMA;
  uint16_t grid_x = 0, rows_per_warp = 1;
  if (route == RpuRmsNormSpmRoute::BASE) {
    TORCH_CHECK(M <= RPU_RMSNORM_SCM_U16_MAX, "no-gamma RMSNorm BASE grid overflows");
    grid_x = static_cast<uint16_t>(M);
  } else {
    TORCH_CHECK(route == RpuRmsNormSpmRoute::V16_GRID8 ||
                    route == RpuRmsNormSpmRoute::V32_GRID8,
                "no-gamma RMSNorm requires BASE or canonical V16/V32 grid8 route");
    const int width = route == RpuRmsNormSpmRoute::V32_GRID8 ? 32 : 16;
    TORCH_CHECK(M % width == 0 && M / 8 <= RPU_RMSNORM_SCM_U16_MAX && C <= 8192,
                "no-gamma RMSNorm vector shape is not admitted");
    id = width == 32 ? KernelId::RMSNORM_NOGAMMA_V32 : KernelId::RMSNORM_NOGAMMA_V16;
    grid_x = 8;
    rows_per_warp = static_cast<uint16_t>(M / 8);
  }
  Kernel_t* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "selected no-gamma RMSNorm payload is unavailable");
  kernel->reset_regs();
  kernel->set_regs(0, static_cast<uint16_t>(C));
  kernel->set_regs(1, half_to_u16(static_cast<c10::Half>(1.0f / C)));
  kernel->set_regs(2, static_cast<uint16_t>(C * sizeof(c10::Half)));
  kernel->set_regs(3, rows_per_warp);
  kernel->set_regs(4, static_cast<uint16_t>(input_spm_addr));
  kernel->set_regs(5, static_cast<uint16_t>(input_spm_addr >> 16));
  kernel->set_regs(6, static_cast<uint16_t>(output_spm_addr));
  kernel->set_regs(7, static_cast<uint16_t>(output_spm_addr >> 16));
  // Registers 8/9 intentionally unbound: there is no gamma operand.
  kernel->set_regs(10, half_to_u16(static_cast<c10::Half>(eps)));
  kernel->set_regs(21, static_cast<uint16_t>(num_cores));
  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1},
                    {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_adarms_norm_shift_spm_kernel(
    uint32_t input_spm, uint32_t output_spm, uint32_t scale_spm,
    uint32_t shift_spm, int64_t rows, double eps, int cores, bool denoise_siblings) {
  TORCH_CHECK(rows >= 1 && rows <= 64 &&
                  (cores == 1 || cores == 4 || cores == 6 || cores == 8),
              "Pi0.5 norm-shift requires H1024, M1..64 and 1, 4, 6 or 8 cores");
  TORCH_CHECK(std::isfinite(eps) && eps > 0 && eps <= 1.0,
              "Pi0.5 norm-shift requires finite eps in (0,1]");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi0.5 norm-shift requires live SPM");
  constexpr uint64_t row_bytes = 1024 * sizeof(c10::Half);
  const uint64_t matrix_bytes = rows * row_bytes;
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  auto checked_range = [base](uint32_t address, uint64_t bytes) {
    return address % 256 == 0 && address >= base &&
        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE;
  };
  TORCH_CHECK(checked_range(input_spm, matrix_bytes) &&
                  checked_range(output_spm, matrix_bytes) &&
                  checked_range(scale_spm, row_bytes) &&
                  checked_range(shift_spm, row_bytes),
              "Pi0.5 norm-shift requires complete aligned absolute SPM operands");
  auto disjoint = [](uint32_t a, uint64_t as, uint32_t b, uint64_t bs) {
    return uint64_t(a) + as <= b || uint64_t(b) + bs <= a;
  };
  TORCH_CHECK((input_spm == output_spm ||
                   disjoint(input_spm, matrix_bytes, output_spm, matrix_bytes)) &&
                  disjoint(output_spm, matrix_bytes, scale_spm, row_bytes) &&
                  disjoint(output_spm, matrix_bytes, shift_spm, row_bytes),
              "Pi0.5 norm-shift rejects partial output alias or parameter overwrite");
  // An owner that admits them takes a sibling when the expansion REF provides
  // it, with the same register image and the same output bytes. At exact M50
  // the warp8 payload's eight blocks cover row = block + 8k with scale/shift
  // loaded once. From 9 to 16 rows the row-pair payload's block b normalizes
  // rows 2b and 2b + 1 (register 14 = rows), so one wave of at most eight
  // blocks replaces the second wave of single-row blocks.
  const auto& cache = KernelCache::instance();
  const bool warp8 = denoise_siblings && rows == 50 &&
      cache.has_loaded(KernelId::PI05_ADARMS_NORM_SHIFT_H1024_M50_WARP8);
  const bool row_pair = denoise_siblings && rows > 8 && rows <= 16 &&
      cache.has_loaded(KernelId::PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR);
  auto* kernel = GET_KERNEL(warp8 ? KernelId::PI05_ADARMS_NORM_SHIFT_H1024_M50_WARP8
      : row_pair ? KernelId::PI05_ADARMS_NORM_SHIFT_H1024_ROW_PAIR
                 : KernelId::PI05_ADARMS_NORM_SHIFT_H1024);
  TORCH_CHECK(kernel, "Pi0.5 norm-shift expansion kernel is unavailable");
  kernel->reset_regs();
  kernel->set_regs(0, uint16_t(1024));
  kernel->set_regs(1, half_to_u16(c10::Half(1.0f / 1024)));
  kernel->set_regs(2, uint16_t(row_bytes));
  kernel->set_regs(3, uint16_t(0));
  for (auto entry : {std::pair<uint32_t, uint32_t>{4, input_spm},
                     {6, output_spm}, {8, scale_spm}, {12, shift_spm}}) {
    kernel->set_regs(entry.first, uint16_t(entry.second & 0xffff));
    kernel->set_regs(entry.first + 1, uint16_t(entry.second >> 16));
  }
  kernel->set_regs(10, half_to_u16(c10::Half(eps)));
  if (row_pair) kernel->set_regs(14, uint16_t(rows));
  const uint16_t grid = warp8 ? 8 : row_pair ? uint16_t((rows + 1) / 2) : uint16_t(rows);
  auto* queue = GET_QUEUE(cores);
  queue->set_broadcast_mode(true);
  std::vector<uint8_t> core_ids;
  for (int core = 0; core < cores; ++core) core_ids.push_back(core);
  queue->enqueu_kernel(*kernel, {grid, 1, 1}, core_ids);
}

// One M32 parent launch, with per-row clean/noisy parameter selection only.
void rpu_launch_g05_rtc_adarms_norm_shift_spm_kernel(
    uint32_t input_spm, uint32_t output_spm, uint32_t clean_scale_spm,
    uint32_t clean_shift_spm, uint32_t noisy_scale_spm, uint32_t noisy_shift_spm,
    int64_t clean_rows, double eps, int cores) {
  constexpr int64_t rows = 32;
  TORCH_CHECK(clean_rows >= 0 && clean_rows <= rows && cores == 8,
              "G05 RTC norm-shift requires M32/H1024, clean rows in [0,32] and 8 cores");
  TORCH_CHECK(std::isfinite(eps) && eps > 0 && eps <= 1.0,
              "G05 RTC norm-shift requires finite eps in (0,1]");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 RTC norm-shift requires live SPM");
  constexpr uint64_t row_bytes = 1024 * sizeof(c10::Half);
  const uint64_t matrix_bytes = rows * row_bytes;
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  auto checked_range = [base](uint32_t address, uint64_t bytes) {
    return address % 256 == 0 && address >= base &&
        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE;
  };
  TORCH_CHECK(checked_range(input_spm, matrix_bytes) &&
                  checked_range(output_spm, matrix_bytes) &&
                  checked_range(clean_scale_spm, row_bytes) &&
                  checked_range(clean_shift_spm, row_bytes) &&
                  checked_range(noisy_scale_spm, row_bytes) &&
                  checked_range(noisy_shift_spm, row_bytes),
              "G05 RTC norm-shift requires complete aligned absolute SPM operands");
  auto disjoint = [](uint32_t a, uint64_t as, uint32_t b, uint64_t bs) {
    return uint64_t(a) + as <= b || uint64_t(b) + bs <= a;
  };
  TORCH_CHECK((input_spm == output_spm ||
                   disjoint(input_spm, matrix_bytes, output_spm, matrix_bytes)) &&
                  disjoint(output_spm, matrix_bytes, clean_scale_spm, row_bytes) &&
                  disjoint(output_spm, matrix_bytes, clean_shift_spm, row_bytes) &&
                  disjoint(output_spm, matrix_bytes, noisy_scale_spm, row_bytes) &&
                  disjoint(output_spm, matrix_bytes, noisy_shift_spm, row_bytes),
              "G05 RTC norm-shift rejects partial output alias or parameter overwrite");
  TORCH_CHECK(KernelCache::instance().has_loaded(KernelId::G05_RTC_ADARMS_NORM_SHIFT_M32H1024),
              "G05 RTC norm-shift optional payload is unavailable");
  auto* kernel = GET_KERNEL(KernelId::G05_RTC_ADARMS_NORM_SHIFT_M32H1024);
  TORCH_CHECK(kernel, "G05 RTC norm-shift expansion kernel is unavailable");
  kernel->reset_regs();
  kernel->set_regs(0, uint16_t(1024));
  kernel->set_regs(1, half_to_u16(c10::Half(1.0f / 1024)));
  kernel->set_regs(2, uint16_t(row_bytes));
  kernel->set_regs(3, uint16_t(0));
  for (auto entry : {std::pair<uint32_t, uint32_t>{4, input_spm},
                     {6, output_spm}, {8, clean_scale_spm}, {12, clean_shift_spm},
                     {14, noisy_scale_spm}, {16, noisy_shift_spm}}) {
    kernel->set_regs(entry.first, uint16_t(entry.second & 0xffff));
    kernel->set_regs(entry.first + 1, uint16_t(entry.second >> 16));
  }
  kernel->set_regs(10, half_to_u16(c10::Half(eps)));
  kernel->set_regs(18, uint16_t(clean_rows));
  auto* queue = GET_QUEUE(cores);
  queue->set_broadcast_mode(true);
  std::vector<uint8_t> core_ids;
  for (int core = 0; core < cores; ++core) core_ids.push_back(core);
  queue->enqueu_kernel(*kernel, {uint16_t(rows), 1, 1}, core_ids);
}

// The existing Pi0.5 payload is parameterized by width. The independently
// measured LingBot2 profile preserves BASE RMSNorm then FP16-rounded shift.
void rpu_launch_lingbot2_adarms_norm_shift_h768_spm_kernel(
    uint32_t input_spm, uint32_t output_spm, uint32_t scale_spm,
    uint32_t shift_spm, int64_t rows, double eps, int cores, bool diagonal) {
  TORCH_CHECK(rows == 51 && cores == 8 && eps == 1.0e-6,
              "LingBot2 norm-shift requires H768, M51, TP8 and eps=1e-6");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "LingBot2 norm-shift requires live SPM");
  constexpr uint64_t row_bytes = 768 * sizeof(c10::Half);
  constexpr uint64_t matrix_bytes = 51 * row_bytes;
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  auto checked_range = [base](uint32_t address, uint64_t bytes) {
    return address % 256 == 0 && address >= base &&
        bytes <= SpmAllocator::SPM_USABLE &&
        uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes;
  };
  TORCH_CHECK(checked_range(input_spm, matrix_bytes) &&
                  checked_range(output_spm, matrix_bytes) &&
                  checked_range(scale_spm, row_bytes) &&
                  checked_range(shift_spm, row_bytes),
              "LingBot2 norm-shift requires complete aligned absolute SPM operands");
  auto disjoint = [](uint32_t a, uint64_t as, uint32_t b, uint64_t bs) {
    return uint64_t(a) + as <= b || uint64_t(b) + bs <= a;
  };
  TORCH_CHECK((input_spm == output_spm ||
                   disjoint(input_spm, matrix_bytes, output_spm, matrix_bytes)) &&
                  disjoint(input_spm, matrix_bytes, scale_spm, row_bytes) &&
                  disjoint(input_spm, matrix_bytes, shift_spm, row_bytes) &&
                  disjoint(output_spm, matrix_bytes, scale_spm, row_bytes) &&
                  disjoint(output_spm, matrix_bytes, shift_spm, row_bytes) &&
                  disjoint(scale_spm, row_bytes, shift_spm, row_bytes),
              "LingBot2 norm-shift permits only exact input/output in-place alias");
  auto* kernel = GET_KERNEL(diagonal
      ? KernelId::LINGBOT2_ADARMS_NORM_SHIFT_H768_DIAGONAL
      : KernelId::PI05_ADARMS_NORM_SHIFT_H1024);
  TORCH_CHECK(kernel, "LingBot2 selected norm-shift expansion kernel is unavailable");
  kernel->reset_regs();
  kernel->set_regs(0, uint16_t(768));
  kernel->set_regs(1, half_to_u16(c10::Half(1.0f / 768)));
  kernel->set_regs(2, uint16_t(row_bytes));
  kernel->set_regs(3, uint16_t(0));
  for (auto entry : {std::pair<uint32_t, uint32_t>{4, input_spm},
                     {6, output_spm}, {8, scale_spm}, {12, shift_spm}}) {
    kernel->set_regs(entry.first, uint16_t(entry.second & 0xffff));
    kernel->set_regs(entry.first + 1, uint16_t(entry.second >> 16));
  }
  kernel->set_regs(10, half_to_u16(c10::Half(eps)));
  auto* queue = GET_QUEUE(cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {uint16_t(rows), 1, 1},
                      {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_rhinovla_newton_norm_shift_spm_kernel(
    uint32_t input_spm, uint32_t output_spm, uint32_t scale_spm,
    uint32_t shift_spm, int64_t rows, double eps, int cores) {
  // Broadcast: every listed core normalizes all rows of its own replica, so
  // the 4/6-core expert residual uses the same payload on its budget.
  TORCH_CHECK(rows >= 1 && rows <= 64 && cores >= 1 && cores <= 8,
              "RhinoVLA Newton norm-shift requires H1024, M1..64 and 1..8 cores");
  TORCH_CHECK(std::isfinite(eps) && eps > 0 && eps <= 1.0,
              "RhinoVLA Newton norm-shift requires finite eps in (0,1]");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "RhinoVLA Newton norm-shift requires live SPM");
  constexpr uint64_t row_bytes = 1024 * sizeof(c10::Half);
  const uint64_t matrix_bytes = rows * row_bytes;
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  auto checked_range = [base](uint32_t address, uint64_t bytes) {
    return address % 256 == 0 && address >= base &&
        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE;
  };
  TORCH_CHECK(checked_range(input_spm, matrix_bytes) &&
                  checked_range(output_spm, matrix_bytes) &&
                  checked_range(scale_spm, row_bytes) &&
                  checked_range(shift_spm, row_bytes),
              "RhinoVLA Newton norm-shift requires complete aligned absolute SPM operands");
  auto disjoint = [](uint32_t a, uint64_t as, uint32_t b, uint64_t bs) {
    return uint64_t(a) + as <= b || uint64_t(b) + bs <= a;
  };
  TORCH_CHECK((input_spm == output_spm ||
                   disjoint(input_spm, matrix_bytes, output_spm, matrix_bytes)) &&
                  disjoint(output_spm, matrix_bytes, scale_spm, row_bytes) &&
                  disjoint(output_spm, matrix_bytes, shift_spm, row_bytes),
              "RhinoVLA Newton norm-shift rejects partial output alias or parameter overwrite");
  auto* kernel = GET_KERNEL(KernelId::RHINOVLA_NEWTON_NORM_SHIFT_H1024);
  TORCH_CHECK(kernel, "RhinoVLA Newton norm-shift additive HIGH kernel is unavailable");
  kernel->reset_regs();
  kernel->set_regs(0, uint16_t(1024));
  kernel->set_regs(1, half_to_u16(c10::Half(1.0f / 1024)));
  kernel->set_regs(2, uint16_t(row_bytes));
  kernel->set_regs(3, uint16_t(1));
  for (auto entry : {std::pair<uint32_t, uint32_t>{4, input_spm},
                     {6, output_spm}, {8, scale_spm}, {12, shift_spm}}) {
    kernel->set_regs(entry.first, uint16_t(entry.second & 0xffff));
    kernel->set_regs(entry.first + 1, uint16_t(entry.second >> 16));
  }
  kernel->set_regs(10, half_to_u16(c10::Half(eps)));
  auto* queue = GET_QUEUE(cores);
  queue->set_broadcast_mode(true);
  std::vector<uint8_t> core_ids;
  for (int core = 0; core < cores; ++core) core_ids.push_back(core);
  queue->enqueu_kernel(*kernel, {uint16_t(rows), 1, 1}, core_ids);
}

// rpu_rmsnorm_spm_test is a conformance-only register-contract probe.
// Production models do not call it. Parameters are forwarded directly:
//   kernel_sel: 0 = base, 16 = V16, 32 = V32.
//   grid_x <= 0 uses M.
//   reg2 < 0 uses C*dwidth.
//   reg3 is reserved for BASE and explicit for the vector variants.
at::Tensor rpu_rmsnorm_spm_test(const at::Tensor& input,
                                const at::Tensor& weight,
                                double eps,
                                int64_t kernel_sel,
                                int64_t grid_x,
                                int64_t reg2,
                                int64_t reg3,
                                int64_t reps)
{
  RECORD_FUNCTION("rpu::rmsnorm_spm_test", {});
  TORCH_CHECK(input.dim() == 2 && input.scalar_type() == at::kHalf &&
                  input.is_contiguous(),
              "rmsnorm_spm_test: input 必须是 [M,C] fp16 contiguous");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(1) &&
                  weight.scalar_type() == at::kHalf && weight.is_contiguous(),
              "rmsnorm_spm_test: weight 必须是 [C] fp16 contiguous");
  TORCH_CHECK(input.device().type() == at::kPrivateUse1 &&
                  weight.device().type() == at::kPrivateUse1,
              "rmsnorm_spm_test: 两个张量都必须在 device 上");

  const int64_t M = input.size(0), C = input.size(1);
  const size_t dwidth = sizeof(c10::Half);

  if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
  const int64_t x_bytes = M * C * (int64_t)dwidth;
  const int64_t w_bytes = C * (int64_t)dwidth;
  using AR = SpmAllocator::AllocRequest;
  auto off = SPM_ALLOC.alloc_temporary_aliased({
      AR{x_bytes, 1, 2},   // x  (DMA-in → kernel-read)
      AR{w_bytes, 1, 2},   // gamma
      AR{x_bytes, 2, 3},   // y  (kernel-write → DMA-out)
  });
  const uint32_t x_addr = SPM_ALLOC.addr(0, off[0]);
  const uint32_t w_addr = SPM_ALLOC.addr(0, off[1]);
  const uint32_t y_addr = SPM_ALLOC.addr(0, off[2]);

  // broadcast（不是 scatter）：rms_norm 是 8 核各算全量的同一份。
  rpu_launch_ddr_broadcast_spm_dma_immediate(
      input.data_ptr<c10::Half>(), M * C, x_addr, 8);
  rpu_launch_ddr_broadcast_spm_dma_immediate(
      weight.data_ptr<c10::Half>(), C, w_addr, 8);

  const uint32_t blk_stride =
      (reg2 < 0) ? (uint32_t)(C * (int64_t)dwidth) : (uint32_t)reg2;
  const uint16_t gx = (grid_x <= 0) ? (uint16_t)M : (uint16_t)grid_x;

  // kernel_sel: 0=base | 16=v16 | 32=v32 | 100=newton | 116=newton_v16 | 132=newton_v32
  KernelId kid = KernelId::RMS_NORM_BF16_SPM;
  if      (kernel_sel ==  16) kid = KernelId::RMS_NORM_SPM_V16;
  else if (kernel_sel ==  32) kid = KernelId::RMS_NORM_SPM_V32;
  else if (kernel_sel == 100) kid = KernelId::RMS_NORM_NEWTON_SPM;
  else if (kernel_sel == 116) kid = KernelId::RMS_NORM_SPM_NEWTON_V16;
  else if (kernel_sel == 132) kid = KernelId::RMS_NORM_SPM_NEWTON_V32;
  Kernel_t* kernel = GET_KERNEL(kid);
  TORCH_CHECK(kernel != nullptr,
              "rmsnorm_spm_test: kernel_sel=", kernel_sel, " 对应的核不在 ref 里");

  kernel->reset_regs();
  kernel->set_regs(0, (uint16_t)C);
  kernel->set_regs(1, half_to_u16(static_cast<c10::Half>(1.0f / (float)C)));
  kernel->set_regs(2, (uint16_t)blk_stride);
  kernel->set_regs(3, (uint16_t)reg3);
  kernel->set_regs(4, (uint16_t)(x_addr & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(x_addr >> 16));
  kernel->set_regs(6, (uint16_t)(y_addr & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(y_addr >> 16));
  kernel->set_regs(8, (uint16_t)(w_addr & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(w_addr >> 16));
  kernel->set_regs(10, half_to_u16(static_cast<c10::Half>(eps)));

  // Upload once, launch the kernel reps times, then download once. Repetition
  // separates kernel work from allocation and transfer overhead. Each launch
  // writes the same output, so the returned tensor contains the last result.
  for (int64_t r = 0, n = (reps > 0 ? reps : 1); r < n; ++r) {
    auto* wq = GET_QUEUE(8);
    wq->set_broadcast_mode(true);
    wq->enqueu_kernel(*kernel, {gx, (uint16_t)1, (uint16_t)1},
                      {0, 1, 2, 3, 4, 5, 6, 7});
  }

  auto out = at::empty({M, C}, input.options());
  rpu_launch_spm_copy_ddr_dma_immediate(y_addr, out.data_ptr<c10::Half>(), M * C);
  SPM_ALLOC.reset_temporary();
  return out;
}


void rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    uint32_t weight_spm_addr,
    int64_t M,
    int64_t C,
    double eps,
    RpuMixedRmsNormSpmRoute route)
{
  const RpuMixedRmsNormSpmRoute expected =
      rpu_resolve_mixed_rmsnorm_spm_route(M, C);
  TORCH_CHECK(
      route == expected,
      "Wall-OSS mixed RMSNorm route does not match its certified shape; got ",
      static_cast<int64_t>(route), " expected ",
      static_cast<int64_t>(expected));

  const c10::Half c_reciprocal = static_cast<c10::Half>(1.0f / C);
  const c10::Half eps_half = static_cast<c10::Half>(eps);
  const uint32_t blk_stride =
      static_cast<uint32_t>(C * static_cast<int64_t>(sizeof(c10::Half)));

  Kernel_t* kernel = route == RpuMixedRmsNormSpmRoute::V16
      ? GET_KERNEL(std::string("llama_rms_norm_bf16in_fp16out_v16"))
      : GET_KERNEL(KernelId::RMS_NORM_BF16IN_FP16OUT_V32);
  TORCH_CHECK(
      kernel != nullptr,
      "Wall-OSS mixed RMSNorm kernel is unavailable; install the combined "
      "rhinoExpansionOpLib_current.ref or set RPU_KERNEL_LIB_PATH_EXPANSION");
  kernel->reset_regs();
  kernel->set_regs(0, static_cast<uint16_t>(C));
  kernel->set_regs(1, half_to_u16(c_reciprocal));
  kernel->set_regs(2, static_cast<uint16_t>(blk_stride));
  kernel->set_regs(3, static_cast<uint16_t>(M / 8));  // rows per core/warp
  kernel->set_regs(4, static_cast<uint16_t>(input_spm_addr & 0xFFFF));
  kernel->set_regs(5, static_cast<uint16_t>(input_spm_addr >> 16));
  kernel->set_regs(6, static_cast<uint16_t>(output_spm_addr & 0xFFFF));
  kernel->set_regs(7, static_cast<uint16_t>(output_spm_addr >> 16));
  kernel->set_regs(8, static_cast<uint16_t>(weight_spm_addr & 0xFFFF));
  kernel->set_regs(9, static_cast<uint16_t>(weight_spm_addr >> 16));
  kernel->set_regs(10, half_to_u16(eps_half));
  kernel->set_regs(64, static_cast<uint16_t>(8));
  kernel->set_regs(65, static_cast<uint16_t>(1));
  kernel->set_regs(66, static_cast<uint16_t>(1));

  auto* queue = GET_QUEUE(UNIFIED_NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(
      *kernel, {static_cast<uint16_t>(8), static_cast<uint16_t>(1),
                static_cast<uint16_t>(1)},
      {0, 1, 2, 3, 4, 5, 6, 7});
}
