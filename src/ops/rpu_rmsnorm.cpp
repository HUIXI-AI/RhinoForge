#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <ATen/ATen.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace ::rhino_lkn;

// Helper: convert half to uint16_t raw bits
static uint16_t half_to_u16(c10::Half h) {
  return h.x;
}

// Process-startup opt-in: select the Newton-refined rsqrt kernel over the base
// srsf approximation. This shared SPM launcher is used by multiple fused model
// families, so no adapter enables it implicitly.
static KernelId rmsnorm_spm_kernel_id() {
  static const bool enabled = [] {
    const char* e = std::getenv("RPU_RMSNORM_NEWTON");
    if (!e || !*e || std::strcmp(e, "0") == 0
        || std::strcmp(e, "false") == 0
        || std::strcmp(e, "False") == 0
        || std::strcmp(e, "off") == 0
        || std::strcmp(e, "OFF") == 0) {
      return false;
    }
    TORCH_CHECK(std::strcmp(e, "1") == 0
                    || std::strcmp(e, "true") == 0
                    || std::strcmp(e, "True") == 0
                    || std::strcmp(e, "on") == 0
                    || std::strcmp(e, "ON") == 0,
                "RPU_RMSNORM_NEWTON accepts only 0/1, false/true, or off/on; got ",
                e);
    return true;
  }();
  return enabled ? KernelId::RMS_NORM_NEWTON_SPM
                 : KernelId::RMS_NORM_BF16_SPM;
}

at::Tensor rpu_rmsnorm(const at::Tensor &input, c10::IntArrayRef normalized_shape,
                       const at::Tensor &weight, double eps) {
#if RPU_PROFILE_ENABLED
  WrapperProfileData wp;
  PROFILE_START();
#endif

  // Validate input
  TORCH_CHECK(input.device().type() == at::DeviceType::PrivateUse1,
              "rpu_rmsnorm: input must be on RPU device");
  TORCH_CHECK(weight.device().type() == at::DeviceType::PrivateUse1,
              "rpu_rmsnorm: weight must be on RPU device");
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

  // The standalone DDR kernel is not address-safe above 4 GiB because its output
  // path uses a_g1B. Use a correctness-first CPU fallback for eager calls; fused
  // models invoke the direct-address SPM launcher and are unaffected.
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

// ============================================================================
// Direct Address SPM RMSNorm Kernel
// ============================================================================
// Same as rpu_launch_rmsnorm_spm_unified_kernel but accepts SPM addresses directly
// instead of using SpmAddressCache with char selectors.
// ============================================================================

// RPU_RMSNORM_VWARP selects a validated vectorized launch: 0, 16, 32, or
// `auto`. A vectorized variant is used only when M is divisible by the selected
// width and the kernel is available. Its ABI uses grid.x=M/V and parameter 3=V.
static int rmsnorm_vwarp() {
    static const int v = [] () -> int {
        const char* e = std::getenv("RPU_RMSNORM_VWARP");
        if (!e || !*e) return 0;
        const std::string s(e);
        if (s == "16")  return 16;
        if (s == "32")  return 32;
        // `auto` chooses the widest valid vector width.
        if (s == "auto") return -1;
        return 0;
    }();
    return v;
}

void rpu_launch_rmsnorm_spm_kernel(
    uint32_t input_spm_addr,           // Input SPM address (core 0 base)
    uint32_t output_spm_addr,          // Output SPM address (core 0 base, can be same for in-place)
    uint32_t weight_spm_addr,          // Weight SPM address (core 0 base)
    int64_t M,                         // Number of rows (seq_len or seq_len * local_heads)
    int64_t C,                         // Number of columns (hidden_size or head_dim)
    double eps)
{
  size_t dwidth = sizeof(c10::Half);

  c10::Half c_reciprocal = static_cast<c10::Half>(1.0f / C);
  uint32_t blk_stride = (C * dwidth);
  c10::Half eps_half = static_cast<c10::Half>(eps);
  uint16_t c_reciprocal_u16 = half_to_u16(c_reciprocal);
  uint16_t eps_u16 = half_to_u16(eps_half);

  // Newton mode uses the base variant. Vectorized launches never split a
  // remainder; they fall back to the base kernel unless M is exactly divisible.
  // `auto` selects v32, then v16, based on divisibility.
  const KernelId base_id = rmsnorm_spm_kernel_id();
  const bool newton_requested = (base_id != KernelId::RMS_NORM_BF16_SPM);
  const int vw_req = newton_requested ? 0 : rmsnorm_vwarp();
  int vw = vw_req;
  if (vw_req == -1) {                      // -1 = auto
    vw = (M % 32 == 0) ? 32 : ((M % 16 == 0) ? 16 : 0);
  }
  Kernel_t* kernel = nullptr;
  uint16_t grid_x = (uint16_t)M;
  uint16_t reg3   = 0;
  if (vw != 0 && (M % vw) == 0) {
    kernel = GET_KERNEL(vw == 16 ? KernelId::RMS_NORM_SPM_V16
                                 : KernelId::RMS_NORM_SPM_V32);
    if (kernel != nullptr) {
      grid_x = (uint16_t)(M / vw);
      reg3   = (uint16_t)vw;      // 每 workgroup 的行数；填 0 会静默算错
    }
  }
  if (kernel == nullptr) {
    kernel = GET_KERNEL(base_id);
    grid_x = (uint16_t)M;
    reg3   = 0;
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

  auto* wq = GET_QUEUE(UNIFIED_NUM_CORES);
  wq->set_broadcast_mode(true);
  wq->enqueu_kernel(*kernel, {grid_x, (uint16_t)1, (uint16_t)1},
                    {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_rmsnorm_bf16in_fp16out_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    uint32_t weight_spm_addr,
    int64_t M,
    int64_t C,
    double eps)
{
  TORCH_CHECK(
      (M == 576 && C == 1280) || (M == 320 && C == 2048)
          || (M == 336 && C == 2048),
      "Wall-OSS mixed RMSNorm supports only Vision (576,1280) or "
      "RTC prefill (320/336,2048); got M=",
      M, " C=", C);

  const c10::Half c_reciprocal = static_cast<c10::Half>(1.0f / C);
  const c10::Half eps_half = static_cast<c10::Half>(eps);
  const uint32_t blk_stride =
      static_cast<uint32_t>(C * static_cast<int64_t>(sizeof(c10::Half)));

  // M336 uses the v16 variant; P640/C320 and Vision M576 use v32.
  const char* kernel_name = M == 336
      ? "llama_rms_norm_bf16in_fp16out_v16"
      : "llama_rms_norm_bf16in_fp16out_v32";
  Kernel_t* kernel = M == 336
      ? GET_KERNEL(std::string(kernel_name))
      : GET_KERNEL(KernelId::RMS_NORM_BF16IN_FP16OUT_V32);
  TORCH_CHECK(
      kernel != nullptr,
      "Wall-OSS mixed RMSNorm kernel is unavailable in the configured "
      "operator library");
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
  kernel->set_regs(64, 8);
  kernel->set_regs(65, 1);
  kernel->set_regs(66, 1);

  auto* queue = GET_QUEUE(UNIFIED_NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(
      *kernel, {static_cast<uint16_t>(8), static_cast<uint16_t>(1),
                static_cast<uint16_t>(1)},
      {0, 1, 2, 3, 4, 5, 6, 7});
}
