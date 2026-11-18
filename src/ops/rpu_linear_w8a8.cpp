#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rpu_linear_w8a8_contract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

using ::rhino_lkn::kKernelRegOk;
using ::rhino_lkn::RpuGetDevAddr;

namespace {
using rpu_pl_w8a8::Partition;
using rpu_pl_w8a8::Shape;

constexpr KernelId kLinearIds[] = {
#define RPU_W8A8_CANDIDATE(id, name, ordinal, m, n, k, lk, flags) KernelId::id,
#include "rpu_linear_w8a8_candidates.inc"
#undef RPU_W8A8_CANDIDATE
};

Shape shape(int64_t m, int64_t n, int64_t k, int partition, int cores, bool bias) {
  TORCH_CHECK(partition == 0 || partition == 1, "W8A8 partition must be row(0) or col(1)");
  return rpu_pl_w8a8::physical_shape(m, n, k, static_cast<Partition>(partition), cores, bias);
}

void validate_owner(const at::Tensor& t, at::ScalarType dtype, uint64_t bytes) {
  TORCH_CHECK(t.defined() && t.device().type() == c10::DeviceType::PrivateUse1 &&
                  t.layout() == c10::Layout::Strided && t.scalar_type() == dtype &&
                  t.is_contiguous() && t.nbytes() == bytes,
              "W8A8 DDR operand has invalid device, dtype, contiguity or extent");
  const auto offset = t.storage_offset();
  TORCH_CHECK(offset >= 0 && static_cast<bool>(t.storage()) &&
                  uint64_t(offset) <= t.storage().nbytes() / t.element_size() &&
                  bytes <= t.storage().nbytes() - uint64_t(offset) * t.element_size(),
              "W8A8 DDR operand backing is too short");
}

void validate_weights(const at::Tensor& w, const at::Tensor& sw, int64_t n, int64_t k) {
  validate_owner(w, at::kChar, uint64_t(n) * uint64_t(k));
  validate_owner(sw, at::kHalf, uint64_t(n) * 2);
  TORCH_CHECK(w.dim() == 2 && w.size(0) == n && w.size(1) == k &&
                  sw.device() == w.device() && sw.dim() == 1 && sw.numel() == n,
              "W8A8 requires TP-swizzled INT8 [N,K] weight and FP16 [N] scales");
  constexpr uint64_t limit = UINT64_C(1) << 40;
  for (const auto* t : {&w, &sw}) {
    const auto address = RpuGetDevAddr(t->data_ptr());
    TORCH_CHECK(address && address % 256 == 0 && address < limit &&
                    t->nbytes() <= limit - address,
                "W8A8 DDR operand must fit aligned 40-bit address range");
  }
  const auto wa = RpuGetDevAddr(w.data_ptr()), sa = RpuGetDevAddr(sw.data_ptr());
  TORCH_CHECK(wa + w.nbytes() <= sa || sa + sw.nbytes() <= wa,
              "W8A8 weight and scale owners overlap");
}

std::vector<uint8_t> core_list(int cores) {
  std::vector<uint8_t> result(cores);
  std::iota(result.begin(), result.end(), uint8_t{0});
  return result;
}

void require_linear(const Shape& s) {
  const auto candidate = rpu_pl_w8a8::installed_candidate(
      rpu_pl_w8a8::select_tile(s), s.local_k);
  TORCH_CHECK(KernelCache::instance().has_loaded(kLinearIds[candidate.ordinal]),
              "W8A8 main payload missing: ", candidate.name,
              "; install the complete source-matched W8A8 main library");
}

void qwen_up_mul(uint32_t input, uint32_t sx, const at::Tensor& weight,
                 const at::Tensor& scale, uint32_t gate, int64_t rows, bool a8) {
  TORCH_CHECK(rows == 256 || (a8 && rows == 512), "Qwen3 UpMul requires M256 or exact W8A8 M512");
  validate_weights(weight, scale, 25600, 5120);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Qwen3 UpMul requires live SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  std::vector<std::pair<uint32_t, uint64_t>> ranges{
      {input, uint64_t(rows) * 5120 * (a8 ? 1 : 2)},
      {gate, uint64_t(rows) * 3200 * 2}};
  if (a8) ranges.push_back({sx, uint64_t(rows) * 2});
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && address % 32 == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes,
                "Qwen3 UpMul requires complete aligned SPM operands");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(address) + bytes <= ranges[j].first ||
                      uint64_t(ranges[j].first) + ranges[j].second <= address,
                  "Qwen3 UpMul input/Gate/SX must be disjoint");
  }
  const auto id = a8 ? (rows == 512 ? KernelId::QWEN3_UP_MUL_W8A8_M512N64K128
                                  : KernelId::QWEN3_UP_MUL_W8A8_M272N64K384)
                     : KernelId::QWEN3_UP_MUL_W8A16_M320N112K128;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Qwen3 UpMul requires its exact row-specific expansion payload");
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Qwen3 UpMul payload unavailable");
  graph.keep_alive(weight); graph.keep_alive(scale);
  rpu_ddr_flush(weight.data_ptr()); rpu_ddr_flush(scale.data_ptr());
  kernel->reset_regs();
  const auto set = [&](uint32_t index, uint16_t value) {
    TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                "Qwen3 UpMul register rejected: ", index);
  };
  const auto pair = [&](uint32_t index, uint32_t value) {
    set(index, uint16_t(value)); set(index + 1, uint16_t(value >> 16));
  };
  pair(0, input); pair(4, gate); set(8, 1); set(9, static_cast<uint16_t>(rows));
  pair(10, 3200); set(12, 5120); set(13, 320); set(15, 8);
  pair(16, a8 ? 5120 : 10240); pair(18, 6400);
  set(26, 2); set(27, 1);
  if (a8) pair(30, sx);
  writer.write(*kernel);
  const uint16_t grid_x = a8 ? 50 : 29;
  set(64, grid_x); set(65, 1); set(66, 1);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {grid_x, 1, 1}, core_list(8));
}
}  // namespace

void rpu_launch_qwen3_up_mul_w8a16_spm_kernel(
    uint32_t input, const at::Tensor& weight, const at::Tensor& scale,
    uint32_t gate, int64_t rows) {
  qwen_up_mul(input, 0, weight, scale, gate, rows, false);
}

void rpu_launch_qwen3_up_mul_w8a8_spm_kernel(
    uint32_t input, uint32_t sx, const at::Tensor& weight,
    const at::Tensor& scale, uint32_t gate, int64_t rows) {
  qwen_up_mul(input, sx, weight, scale, gate, rows, true);
}

void rpu_launch_linear_w8a8_spm_kernel(
    uint32_t input_int8_spm, uint32_t row_scale_spm,
    const at::Tensor& weight, const at::Tensor& scale, uint32_t output_spm,
    int64_t m, int64_t n, int64_t k, int partition, int num_cores,
    uint32_t bias_spm) {
  const auto s = shape(m, n, k, partition, num_cores, bias_spm != 0);
  validate_weights(weight, scale, n, k);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "W8A8 requires live SPM");
  // DDR addresses are supplied solely by the typed Graph writer below.
  const auto descriptor = rpu_pl_w8a8::make_descriptor(s,
      {input_int8_spm, output_spm, bias_spm, row_scale_spm},
      SPM_ALLOC.addr(0, 0), SpmAllocator::SPM_USABLE);
  const auto id = kLinearIds[descriptor.ordinal];
  require_linear(s);
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "W8A8 kernel unavailable: ", descriptor.name);
  graph.keep_alive(weight);
  graph.keep_alive(scale);
  rpu_ddr_flush(weight.data_ptr());
  rpu_ddr_flush(scale.data_ptr());
  kernel->reset_regs();
  for (uint32_t i = 0; i < descriptor.regs.size(); ++i)
    TORCH_CHECK(kernel->set_regs(i, descriptor.regs[i]) == kKernelRegOk,
                "W8A8 register rejected: ", i);
  writer.write(*kernel);
  for (uint32_t i = 0; i < 3; ++i)
    TORCH_CHECK(kernel->set_regs(64 + i, descriptor.grid[i]) == kKernelRegOk,
                "W8A8 grid register rejected");
  auto* queue = GET_QUEUE(num_cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,
      {descriptor.grid[0], descriptor.grid[1], descriptor.grid[2]}, core_list(num_cores));
}

void rpu_launch_dynamic_quantize_w8a8_spm_kernel(
    uint32_t input_spm, uint32_t output_int8_spm, uint32_t row_scale_spm,
    int64_t m, int64_t local_k, int num_cores) {
  const Shape s{m, 16, local_k, num_cores, Partition::Col, false};
  rpu_pl_w8a8::validate_shape(s);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "W8A8 quantization requires live SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 3> ranges{{
      {input_spm, uint64_t(m) * local_k * 2},
      {output_int8_spm, uint64_t(m) * local_k},
      {row_scale_spm, uint64_t((m + 15) / 16) * 32}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && address % 32 == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes,
                "W8A8 quantization requires complete aligned SPM operands");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(address) + bytes <= ranges[j].first ||
                      uint64_t(ranges[j].first) + ranges[j].second <= address,
                  "W8A8 quantization input/output/scale must be disjoint");
  }
  constexpr auto id = KernelId::DYNAMIC_QUANTIZE_W8A8;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "W8A8 requires dynamic_quantize_w8a8 in the complete expansion library");
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "W8A8 activation quantization payload unavailable");
  kernel->reset_regs();
  const auto set = [&](uint32_t index, uint16_t value) {
    TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                "W8A8 quantization register rejected: ", index);
  };
  const auto pair = [&](uint32_t index, uint32_t value) {
    set(index, uint16_t(value)); set(index + 1, uint16_t(value >> 16));
  };
  pair(0, input_spm); pair(2, row_scale_spm); pair(4, output_int8_spm);
  set(6, uint16_t(m)); set(7, 1); set(8, 1); set(9, uint16_t(local_k));
  set(10, c10::Half(127.0f).x); set(11, c10::Half(-127.0f).x);
  pair(14, uint32_t(local_k * 2)); pair(16, 2); pair(18, uint32_t(local_k));
  set(20, 1);
  const uint16_t grid_y = uint16_t((m + 15) / 16);
  set(64, 1); set(65, grid_y); set(66, 1);
  auto* queue = GET_QUEUE(num_cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {1, grid_y, 1}, core_list(num_cores));
}

void rpu_launch_mul_quantize_w8a8_spm_kernel(
    uint32_t gate, uint32_t up, uint32_t q, uint32_t sx,
    int64_t rows, int64_t local_k, int num_cores) {
  TORCH_CHECK((rows == 128 || rows == 192) && local_k == 3200 && num_cores == 8,
              "Qwen3 MulQuant requires exact M128/M192 K3200/TP8");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Qwen3 MulQuant requires live SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 4> ranges{{
      {gate, uint64_t(rows) * 6400}, {up, uint64_t(rows) * 6400},
      {q, uint64_t(rows) * 3200}, {sx, uint64_t(rows) * 2}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && address % 32 == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes,
                "Qwen3 MulQuant requires complete aligned SPM operands");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(address) + bytes <= ranges[j].first ||
                      uint64_t(ranges[j].first) + ranges[j].second <= address,
                  "Qwen3 MulQuant Gate/Up/Q/SX must be disjoint");
  }
  const auto id = rows == 128 ? KernelId::MUL_QUANTIZE_W8A8_H3200_M128
                              : KernelId::MUL_QUANTIZE_W8A8_H3200_M192;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Qwen3 requires the exact MulQuant expansion payload");
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Qwen3 MulQuant payload unavailable");
  kernel->reset_regs();
  const auto set = [&](uint32_t index, uint16_t value) {
    TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                "Qwen3 MulQuant register rejected: ", index);
  };
  const auto pair = [&](uint32_t index, uint32_t value) {
    set(index, uint16_t(value)); set(index + 1, uint16_t(value >> 16));
  };
  pair(0, gate); pair(2, sx); pair(4, q); pair(22, up);
  set(6, uint16_t(rows)); set(7, 1); set(8, 1); set(9, 3200);
  set(10, c10::Half(127.0f).x); set(11, c10::Half(-127.0f).x);
  pair(14, 6400); pair(16, 2); pair(18, 3200); set(20, 1);
  set(64, 8); set(65, 1); set(66, 1);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, core_list(8));
}

void rpu_launch_rms_norm_quantize_w8a8_spm_kernel(
    uint32_t input_spm, uint32_t output_int8_spm, uint32_t row_scale_spm,
    uint32_t gamma_spm, int64_t m, double eps, int num_cores) {
  constexpr uint32_t hidden = 5120;
  TORCH_CHECK((m == 128 || m == 192 || m == 256) && num_cores == 8 && eps == 1e-6,
              "W8A8 RMSNorm requires exact M128/M192/M256 H5120/TP8 and epsilon 1e-6");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "W8A8 RMSNorm requires live SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 4> ranges{{
      {input_spm, uint64_t(m) * hidden * 2},
      {output_int8_spm, uint64_t(m) * hidden},
      {row_scale_spm, uint64_t(m) * 2}, {gamma_spm, hidden * 2}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && address % 32 == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes,
                "W8A8 RMSNorm requires complete aligned SPM operands");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(address) + bytes <= ranges[j].first ||
                      uint64_t(ranges[j].first) + ranges[j].second <= address,
                  "W8A8 RMSNorm X/Q/SX/gamma must be disjoint");
  }
  const auto id = m == 256 ? KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M256
                 : m == 192 ? KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M192
                           : KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M128;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "W8A8 RMSNorm requires the exact RMSNormQuant payload in the complete expansion library");
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "W8A8 RMSNorm quantization payload unavailable");
  kernel->reset_regs();
  const auto set = [&](uint32_t index, uint16_t value) {
    TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                "W8A8 RMSNorm register rejected: ", index);
  };
  const auto pair = [&](uint32_t index, uint32_t value) {
    set(index, uint16_t(value)); set(index + 1, uint16_t(value >> 16));
  };
  pair(0, input_spm); pair(2, row_scale_spm); pair(4, output_int8_spm);
  set(6, uint16_t(m)); set(7, 1); set(8, 1); set(9, hidden);
  set(10, c10::Half(127.0f).x); set(11, c10::Half(-127.0f).x);
  pair(14, hidden * 2); pair(16, 2); pair(18, hidden); set(20, 1);
  pair(22, gamma_spm);
  set(24, c10::Half(eps).x); set(25, c10::Half(1.0f / float(hidden)).x);
  set(64, 8); set(65, 1); set(66, 1);
  auto* queue = GET_QUEUE(num_cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, core_list(num_cores));
}

void rpu_launch_layer_norm_quantize_w8a8_spm_kernel(
    uint32_t input_spm, uint32_t output_int8_spm, uint32_t row_scale_spm,
    uint32_t gamma_spm, uint32_t beta_spm,
    int64_t m, double eps, int num_cores) {
  constexpr uint32_t hidden = 1152;
  TORCH_CHECK(m >= 1 && m <= 768 && num_cores >= 1 && num_cores <= 8,
              "W8A8 LayerNorm requires H1152, M1..768 and 1..8 cores");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0,
              "W8A8 LayerNorm requires finite positive epsilon");
  // Match the ordinary LayerNorm ABI, including the FP32 product before the
  // FP16 retirement. Reject underflow/overflow before recording any command.
  const c10::Half eps_sum(static_cast<float>(eps) * float(hidden));
  TORCH_CHECK(std::isfinite(float(eps_sum)) && float(eps_sum) > 0.0f,
              "W8A8 LayerNorm epsilon * H must be finite nonzero FP16");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "W8A8 LayerNorm requires live SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 5> ranges{{
      {input_spm, uint64_t(m) * hidden * 2},
      {output_int8_spm, uint64_t(m) * hidden},
      {row_scale_spm, uint64_t((m + 15) / 16) * 32},
      {gamma_spm, hidden * 2}, {beta_spm, hidden * 2}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && address % 32 == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes,
                "W8A8 LayerNorm requires complete aligned SPM operands");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(address) + bytes <= ranges[j].first ||
                      uint64_t(ranges[j].first) + ranges[j].second <= address,
                  "W8A8 LayerNorm input/output/scale/gamma/beta must be disjoint");
  }
  constexpr auto id = KernelId::LAYER_NORM_QUANTIZE_W8A8_H1152;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "W8A8 LayerNorm requires layer_norm_quantize_w8a8_h1152 in the complete expansion library");
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "W8A8 LayerNorm quantization payload unavailable");
  kernel->reset_regs();
  const auto set = [&](uint32_t index, uint16_t value) {
    TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                "W8A8 LayerNorm register rejected: ", index);
  };
  const auto pair = [&](uint32_t index, uint32_t value) {
    set(index, uint16_t(value)); set(index + 1, uint16_t(value >> 16));
  };
  const float reciprocal = 1.0f / float(hidden);
  uint32_t reciprocal_bits;
  std::memcpy(&reciprocal_bits, &reciprocal, sizeof(reciprocal_bits));
  pair(0, input_spm); pair(2, output_int8_spm);
  pair(4, gamma_spm); pair(6, beta_spm);
  set(8, uint16_t(m)); set(9, hidden); pair(10, reciprocal_bits);
  set(12, eps_sum.x); set(13, 0); pair(14, 0); pair(16, hidden * 2);
  set(18, 1); set(19, 16); set(20, c10::Half(std::sqrt(float(hidden))).x);
  set(21, 1); pair(22, row_scale_spm);
  const uint16_t grid_x = uint16_t((m + 15) / 16);
  set(64, grid_x); set(65, 1); set(66, 1);
  auto* queue = GET_QUEUE(num_cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {grid_x, 1, 1}, core_list(num_cores));
}

void rpu_launch_gelu_tanh_quantize_w8a8_spm_kernel(
    uint32_t input_spm, uint32_t output_int8_spm, uint32_t row_scale_spm,
    int64_t m, int num_cores) {
  constexpr uint32_t local_k = 544;
  TORCH_CHECK(m >= 1 && m <= 768 && num_cores >= 1 && num_cores <= 8,
              "W8A8 GELU requires local H544, M1..768 and 1..8 cores");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "W8A8 GELU requires live SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 3> ranges{{
      {input_spm, uint64_t(m) * local_k * 2},
      {output_int8_spm, uint64_t(m) * local_k},
      {row_scale_spm, uint64_t((m + 15) / 16) * 32}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && address % 32 == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes,
                "W8A8 GELU requires complete aligned SPM operands");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(address) + bytes <= ranges[j].first ||
                      uint64_t(ranges[j].first) + ranges[j].second <= address,
                  "W8A8 GELU input/output/scale must be disjoint");
  }
  constexpr auto id = KernelId::GELU_TANH_QUANTIZE_W8A8_H544;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "W8A8 GELU requires gelu_tanh_quantize_w8a8_h544 in the complete expansion library");
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "W8A8 GELU quantization payload unavailable");
  kernel->reset_regs();
  const auto set = [&](uint32_t index, uint16_t value) {
    TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                "W8A8 GELU register rejected: ", index);
  };
  const auto pair = [&](uint32_t index, uint32_t value) {
    set(index, uint16_t(value)); set(index + 1, uint16_t(value >> 16));
  };
  pair(0, input_spm); pair(2, output_int8_spm); pair(4, row_scale_spm);
  set(6, uint16_t(m)); set(7, local_k);
  const uint16_t grid_x = uint16_t((m + 15) / 16);
  set(64, grid_x); set(65, 1); set(66, 1);
  auto* queue = GET_QUEUE(num_cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {grid_x, 1, 1}, core_list(num_cores));
}

at::Tensor rpu_linear_w8a8(const at::Tensor& input, const at::Tensor& weight,
    const at::Tensor& scale, const c10::optional<at::Tensor>& bias,
    int64_t partition, int64_t num_cores) {
  RECORD_FUNCTION("rpu::linear_w8a8", {});
  TORCH_CHECK(!graph_dma::active(),
              "linear_w8a8 allocates fresh DDR output; retained models must use the SPM launcher");
  TORCH_CHECK(input.defined() && input.dim() >= 1 && weight.defined() && weight.dim() == 2 &&
                  input.device().type() == c10::DeviceType::PrivateUse1 &&
                  weight.device() == input.device() && input.scalar_type() == at::kHalf &&
                  input.is_contiguous() && input.size(-1) == weight.size(1) &&
                  input.numel() > 0 && (partition == 0 || partition == 1) &&
                  num_cores >= 1 && num_cores <= 8,
              "linear_w8a8 requires contiguous RPU FP16 [...,K], weight [N,K], row/col and 1..8 cores");
  const int64_t k = weight.size(1), n = weight.size(0), m = input.numel() / k;
  const auto full = shape(m, n, k, int(partition), int(num_cores), bias.has_value());
  validate_owner(input, at::kHalf, uint64_t(m) * k * 2);
  validate_weights(weight, scale, n, k);
  if (bias) {
    validate_owner(*bias, at::kHalf, uint64_t(n) * 2);
    TORCH_CHECK(bias->device() == input.device() && bias->dim() == 1 && bias->numel() == n,
                "linear_w8a8 bias must be RPU FP16 [N]");
  }
  TORCH_CHECK(KernelCache::instance().has_loaded(KernelId::DYNAMIC_QUANTIZE_W8A8),
              "linear_w8a8 requires the source-matched expansion quantization payload");
  if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
  SpmAllocator::ScopedTemporary scope;
  const bool row = partition == 0;
  const size_t lk = full.local_k, ln = full.local_n;
  const size_t fixed = (bias ? Align(ln * 2, SpmAllocator::ALIGN) : 0) +
                       7 * SpmAllocator::ALIGN;
  const size_t per_row = 3 * lk + 2 + 2 * ln + (row ? 4 * n : 0);
  constexpr size_t budget = SpmAllocator::SPM_PLANNING_BUDGET & ~(SpmAllocator::ALIGN - 1);
  const size_t used = SPM_ALLOC.total_used();
  TORCH_CHECK(used < budget, "linear_w8a8 current SPM exceeds planning budget");
  const size_t available = std::min(SPM_ALLOC.free_space(), budget - used);
  TORCH_CHECK(available >= fixed + per_row, "linear_w8a8 insufficient SPM for one row");
  size_t chunk = std::min<size_t>(m, (available - fixed) / per_row);
  if (chunk < size_t(m) && chunk >= 16) chunk = chunk / 16 * 16;
  // Check every selected shape before any upload/quantization. No partial work
  // followed by a missing-tail-kernel failure.
  for (size_t rows : {chunk, size_t(m) % chunk})
    if (rows) require_linear(shape(rows, n, k, int(partition), int(num_cores), bool(bias)));
  const auto alloc = [](size_t bytes) {
    return SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(bytes));
  };
  const uint32_t x = alloc(chunk * lk * 2), q = alloc(chunk * lk);
  const uint32_t sx = alloc(Align(chunk, size_t{16}) * 2), y = alloc(chunk * ln * 2);
  const uint32_t zero = row ? alloc(chunk * n * 2) : 0;
  const uint32_t reduced = row ? alloc(chunk * n * 2) : 0;
  const uint32_t b = bias ? alloc(ln * 2) : 0;
  auto output_shape = input.sizes().vec(); output_shape.back() = n;
  auto output = at::empty(output_shape, input.options());
  if (bias) {
    if (row) {
      // Row kernels add the local bias. Only core0 receives the logical bias;
      // every other core must see zero before the shared FP16 ring reduction.
      rpu_launch_fill_spm_kernel(b, ln, c10::Half(0.0f), int(num_cores));
      rpu_launch_ddr_broadcast_spm_dma(*bias, 0, n, b, 1);
    } else {
      rpu_launch_ddr_scatter_spm_dma(*bias, 0, ln, ln * 2, b, int(num_cores));
    }
  }
  for (size_t first = 0; first < size_t(m); first += chunk) {
    const size_t rows = std::min(chunk, size_t(m) - first);
    if (row) {
      // No host repacking: scatter each physical input row's K shards.
      for (size_t r = 0; r < rows; ++r)
        rpu_launch_ddr_scatter_spm_dma(input, (first + r) * k, lk, lk * 2,
                                      x + r * lk * 2, int(num_cores));
    } else {
      rpu_launch_ddr_broadcast_spm_dma(input, first * k, rows * k, x, int(num_cores));
    }
    rpu_launch_dynamic_quantize_w8a8_spm_kernel(x, q, sx, rows, lk, int(num_cores));
    rpu_launch_linear_w8a8_spm_kernel(q, sx, weight, scale, y, rows, n, k,
                                     int(partition), int(num_cores), b);
    if (row) {
      rpu_launch_fill_spm_kernel(zero, rows * n, c10::Half(0.0f), int(num_cores));
      rpu_launch_all_reduce_sum_residual_kernel(y, zero, reduced, rows, n,
                                               int(num_cores), int(num_cores));
      rpu_launch_spm_copy_ddr_dma(reduced, output.data_ptr<c10::Half>() + first * n, rows * n);
    } else {
      rpu_launch_spm_scatter_rows_ddr_dma_immediate(y,
          output.data_ptr<c10::Half>() + first * n, rows, ln, ln * 2, n * 2, ln * 2,
          int(num_cores));
    }
  }
  return output;
}
