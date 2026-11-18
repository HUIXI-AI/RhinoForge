#pragma once
#include "rpu_pi_wall_tiling.h"

namespace rpu_wall_fp16_retained {
using LinearArgs = rpu_pl_tiling::ParallelLinearRegisterArgs;

struct PairedDownArgs {
  LinearArgs linear;
  uint64_t second_input, second_output;
  uint32_t rows_per_input;
  bool acc32 = false, zero_point = false;
};
struct SharedSiluArgs {
  LinearArgs linear;
  bool acc32 = false, zero_point = false, fused_silu = true;
};

inline void require(bool valid) {
  if (!valid) throw std::invalid_argument("Unsupported retained Wall FP16 current-ABI contract");
}
inline void precision(const LinearArgs& a, bool acc32, bool zero_point) {
  rpu_pl_tiling::validate_linear_registers(a);
  require(a.dtype_mode == 2 && a.x_dtype == 2 && !acc32 && !zero_point &&
          a.num_cores == 8 && a.has_bias == 0 && a.bias_spm_addr == 0 &&
          a.scale_dev_addr == 0);
}
struct Span { uint64_t begin, size; };
template <std::size_t N>
inline void owners(const std::array<Span, N>& spans) {
  for (std::size_t i = 0; i < N; ++i) {
    const auto& s = spans[i];
    require(s.begin != 0 && s.begin % 16 == 0 && s.begin <= UINT32_MAX &&
            s.size <= UINT64_C(0x100000000) - s.begin);
    for (std::size_t j = 0; j < i; ++j)
      require(s.begin + s.size <= spans[j].begin ||
              spans[j].begin + spans[j].size <= s.begin);
  }
}
inline void validate(const PairedDownArgs& p) {
  const auto& a = p.linear;
  precision(a, p.acc32, p.zero_point);
  require(a.local_m == 640 && a.local_n == 2048 && a.local_k == 1376 &&
          p.rows_per_input == 320 && a.partition == 0 &&
          a.input_m_byte_step == 2752 && a.output_m_byte_step == 4096);
  owners(std::array<Span, 4>{{{a.input_spm_addr, 320 * UINT64_C(2752)},
                            {p.second_input, 320 * UINT64_C(2752)},
                            {a.output_spm_addr, 320 * UINT64_C(4096)},
                            {p.second_output, 320 * UINT64_C(4096)}}});
}
inline void validate(const SharedSiluArgs& p) {
  const auto& a = p.linear;
  precision(a, p.acc32, p.zero_point);
  require(p.fused_silu && a.local_m == 640 && a.local_n == 1376 &&
          a.local_k == 2048 && a.partition == 1 &&
          a.input_m_byte_step == 4096 && a.output_m_byte_step == 2752);
  owners(std::array<Span, 2>{{{a.input_spm_addr, 640 * UINT64_C(4096)},
                            {a.output_spm_addr, 640 * UINT64_C(2752)}}});
}
inline rpu_pi_wall::Selection select(const PairedDownArgs& a) {
  validate(a);
  return rpu_pi_wall::select(rpu_pi_wall::Operation::PairedLinearRetainedFp16,
      rpu_pl_tiling::WeightType::Fp16, false, 320, 2048, 1376);
}
inline rpu_pi_wall::Selection select(const SharedSiluArgs& a) {
  validate(a);
  return rpu_pi_wall::select(rpu_pi_wall::Operation::SharedSiluRetainedFp16,
      rpu_pl_tiling::WeightType::Fp16, false, 640, 1376, 2048);
}
template <typename Kernel>
inline void write(Kernel& kernel, const PairedDownArgs& a) {
  validate(a);
  rpu_pl_tiling::set_parallel_linear_regs(kernel, a.linear);
  rpu_pl_tiling::set_u32_reg_pair(kernel, 48, a.second_input);
  rpu_pl_tiling::set_u32_reg_pair(kernel, 50, a.second_output);
  kernel.set_regs(52, static_cast<uint16_t>(a.rows_per_input));
}
template <typename Kernel>
inline void write(Kernel& kernel, const SharedSiluArgs& a) {
  validate(a);
  rpu_pl_tiling::set_parallel_linear_regs(kernel, a.linear);
  // The typed Graph DDR writer supplies the live Up owner at GP32/33 after
  // this common setup, exactly as for the current generic SharedSilu caller.
}
} // namespace rpu_wall_fp16_retained
