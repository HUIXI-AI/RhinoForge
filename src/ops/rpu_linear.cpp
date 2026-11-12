#include "rhino_launch_buffer.h"

#include "rhino_launch_program.h" // for Program_t
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"  // eager linear stages through the SPM arena
#include "rpu_linear_tiling.h"  // host-side generated-tile selection
#include "rpu_hyvla_gateup_resident.h"
#include "execution_coordinator.h"
#include <ATen/MemoryOverlap.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <utility>
#include <vector>   // for vector

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8

static GraphKernelRegisterWriter wall_precision_registers(
    KernelId kernel_id, bool fp16,
    std::vector<GraphDdrRegisterOperandSpec> operands) {
  if (fp16) {
    operands.erase(std::remove_if(operands.begin(), operands.end(),
        [](const auto& operand) {
          return operand.abi.role == GraphDdrRegisterRole::PerChannelScale;
        }), operands.end());
    for (auto& operand : operands) {
      if (operand.abi.role == GraphDdrRegisterRole::Int8ModelWeight)
        operand.abi.role = GraphDdrRegisterRole::ModelWeight;
    }
  }
  // Resident FP16 kernels can be entirely SPM-backed. Let GET_KERNEL apply
  // their intrinsic-no-DDR policy instead of staging an empty typed-DDR set.
  if (operands.empty()) return {};
  return RpuKernelGraph::active().stage_kernel_ddr_registers(
      kernel_id, std::move(operands));
}

// ============================================================================
// Thread-local Output Tensor Cache for Linear
// ============================================================================
// 对于推理场景，output shape 通常是固定的，通过缓存避免重复分配
// ============================================================================

// Output tensor cache removed: key collision (batch<<32|out_features) caused
// DDR coherency bugs when multiple linears share the same output shape (e.g.
// k_proj and v_proj both with out_features=256).

at::Tensor rpu_retain_linear_quant_scale(
    const at::Tensor& weight,
    const at::Tensor& scale) {
  if (!scale.defined() || weight.scalar_type() != at::kByte ||
      scale.scalar_type() != at::kHalf) {
    return scale;
  }
  TORCH_CHECK(
      scale.dim() == 2 &&
          scale.device().type() == c10::DeviceType::PrivateUse1 &&
          scale.is_contiguous(),
      "W4 pgrp scale must be a contiguous 2D device tensor");

  constexpr int64_t kAlignmentBytes = NUM_CORES * 512;
  const uint64_t source_addr =
      RpuGetDevAddr(scale.data_ptr<c10::Half>());
  if (source_addr % kAlignmentBytes == 0) return scale;

  constexpr int64_t kAlignmentElements =
      kAlignmentBytes / static_cast<int64_t>(sizeof(c10::Half));
  auto storage = at::empty(
      {scale.numel() + kAlignmentElements}, scale.options());
  const uint64_t storage_addr =
      RpuGetDevAddr(storage.data_ptr<c10::Half>());
  const int64_t byte_offset = static_cast<int64_t>(
      (kAlignmentBytes - (storage_addr % kAlignmentBytes)) %
      kAlignmentBytes);
  TORCH_INTERNAL_ASSERT(
      byte_offset % static_cast<int64_t>(sizeof(c10::Half)) == 0);
  auto aligned = storage
      .narrow(0, byte_offset / sizeof(c10::Half), scale.numel())
      .view(scale.sizes());
  aligned.copy_(scale);
  TORCH_INTERNAL_ASSERT(
      RpuGetDevAddr(aligned.data_ptr<c10::Half>()) % kAlignmentBytes == 0);
  return aligned;
}

namespace {
constexpr size_t kPi05ResidentW8PerCore = 524288;

void check_pi05_resident_spm(uint32_t address, size_t bytes) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Pi0.5 resident W8 requires the initialized SPM allocator");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  TORCH_CHECK(address >= base && address % 256 == 0 &&
                  bytes <= SpmAllocator::SPM_USABLE &&
                  address - base <= SpmAllocator::SPM_USABLE - bytes,
              "Pi0.5 resident W8 absolute SPM interval/alignment is invalid");
}
}  // namespace

at::Tensor rpu_pack_pi05_gateup_resident_w8(const at::Tensor& weight) {
  TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kChar &&
                  weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  weight.dim() == 2 && weight.size(0) == 4096 &&
                  weight.size(1) == 1024 && weight.is_contiguous() &&
                  weight.nbytes() == 8 * kPi05ResidentW8PerCore,
              "Pi0.5 resident W8 requires installed contiguous device int8 [4096,1024]");
  // The installed col-partition format is [local_N/16,K/32,core,16,32].
  // Deinterleave 512-byte stripes on CPU only at installation. No arithmetic,
  // transpose within a stripe, quantization, or FP16 conversion is performed.
  auto packed_cpu = weight.cpu().view({32, 32, 8, 512})
      .permute({2, 0, 1, 3}).contiguous().view({8, 524288});
  auto packed = packed_cpu.to(weight.device());
  rpu_ddr_flush_force(packed.data_ptr<int8_t>());
  return packed;
}

void rpu_launch_pi05_gateup_resident_w8_preload(
    uint64_t& live_src_base, const at::Tensor& packed_weight,
    uint32_t weight_spm_addr) {
  TORCH_CHECK(packed_weight.defined() &&
                  packed_weight.scalar_type() == at::kChar &&
                  packed_weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  packed_weight.is_contiguous() && packed_weight.dim() == 2 &&
                  packed_weight.size(0) == 8 &&
                  packed_weight.size(1) == kPi05ResidentW8PerCore &&
                  packed_weight.nbytes() == 8 * kPi05ResidentW8PerCore &&
                  live_src_base == RpuGetDevAddr(packed_weight.data_ptr()),
              "Pi0.5 resident W8 preload owner/base/byte extent mismatch");
  check_pi05_resident_spm(weight_spm_addr, kPi05ResidentW8PerCore);
  rpu_ddr_flush(packed_weight.data_ptr<int8_t>());
  if (graph_dma::active()) {
    RpuKernelGraph::active().keep_alive(packed_weight);
    graph_dma::stage_census_mutable_burst(
        live_src_base, packed_weight, GraphOuterFastDmaSide::Source,
        0, packed_weight.nbytes(), 8);
  }
  // This established raw DMA implementation expresses byte count as halfs;
  // the owner remains signed int8 in the census. It performs no conversion.
  rpu_launch_ddr_scatter_spm_dma_mutable(
      &live_src_base, 0, kPi05ResidentW8PerCore / sizeof(c10::Half),
      kPi05ResidentW8PerCore, weight_spm_addr, 8);
}

void rpu_launch_pi05_gateup_resident_w8_kernel(
    uint32_t input_spm_addr, uint32_t weight_spm_addr,
    uint32_t output_spm_addr, const at::Tensor& scale) {
  constexpr size_t input_bytes = 50 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 50 * 512 * sizeof(c10::Half);
  check_pi05_resident_spm(input_spm_addr, input_bytes);
  check_pi05_resident_spm(weight_spm_addr, kPi05ResidentW8PerCore);
  check_pi05_resident_spm(output_spm_addr, output_bytes);
  auto disjoint = [](uint64_t a, size_t an, uint64_t b, size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input_spm_addr, input_bytes, weight_spm_addr, kPi05ResidentW8PerCore) &&
                  disjoint(output_spm_addr, output_bytes, weight_spm_addr, kPi05ResidentW8PerCore) &&
                  disjoint(input_spm_addr, input_bytes, output_spm_addr, output_bytes),
              "Pi0.5 resident W8 input/output/weight intervals overlap");
  TORCH_CHECK(scale.defined() && scale.scalar_type() == at::kHalf &&
                  scale.device().type() == c10::DeviceType::PrivateUse1 &&
                  scale.dim() == 1 && scale.size(0) == 4096 &&
                  scale.is_contiguous() && scale.nbytes() == 8192,
              "Pi0.5 resident W8 requires original per-channel fp16 scale [4096]");
  const auto tile = rpu_pl_tiling::select_tile_acc16(50, 512, 1024, false);
  TORCH_CHECK(tile.m_tile == 512 && tile.n_tile == 48,
              "Pi0.5 resident W8 original auto-tile identity drift");
  constexpr KernelId id = KernelId::PI05_DENOISE_GATEUP_RESIDENT_W8A16_M512N48K128;
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{
                  GraphDdrRegisterRole::PerChannelScale,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
  rpu_ddr_flush(scale.data_ptr<c10::Half>());
  Kernel_t* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "Pi0.5 resident W8 expansion kernel is absent");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0,
          50, 512, 1024, 0, 8, 2048, 1024, 0, 1, 1});
  // The standard helper encodes DDR weight >>8. This fixed symbol instead
  // accepts absolute local SPM bytes in 2/3; override only that register pair.
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, weight_spm_addr);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {11, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

namespace {
constexpr size_t kGr00tFf1ResidentW8PerCore = 1179648;

void check_gr00t_ff1_resident_spm(uint32_t address, size_t bytes) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "GR00T resident FF1 requires the initialized SPM allocator");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  TORCH_CHECK(address >= base && address % 256 == 0 &&
                  bytes <= SpmAllocator::SPM_USABLE &&
                  address - base <= SpmAllocator::SPM_USABLE - bytes,
              "GR00T resident FF1 absolute SPM interval/alignment is invalid");
}
}  // namespace

at::Tensor rpu_pack_gr00t_ff1_resident_w8(const at::Tensor& weight) {
  TORCH_CHECK(!RpuKernelGraph::has_active() &&
                  RpuExecutionCoordinator::current_physical_token_if_owned(
                      "gr00t_pack_ff1_resident_w8") != 0,
              "GR00T resident FF1 packing requires an owned cold execution claim");
  TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kChar &&
                  weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  weight.dim() == 2 && weight.size(0) == 6144 &&
                  weight.size(1) == 1536 && weight.is_contiguous() &&
                  weight.nbytes() == 8 * kGr00tFf1ResidentW8PerCore,
              "GR00T resident FF1 requires installed contiguous device int8 [6144,1536]");
  // Installed COL TP8 bytes are [local_N/16,K/32,core,16,32].
  // Only deinterleave 512-byte stripes; retain every quantized weight byte.
  auto cpu = weight.cpu().view({48, 48, 8, 512})
      .permute({2, 0, 1, 3}).contiguous().view({8, 1179648});
  auto packed = cpu.to(weight.device());
  rpu_ddr_flush_force(packed.data_ptr<int8_t>());
  return packed;
}

void rpu_launch_gr00t_ff1_resident_w8_preload(
    uint64_t& live_src_base, const at::Tensor& packed_weight,
    uint32_t weight_spm_addr) {
  TORCH_CHECK(packed_weight.defined() &&
                  packed_weight.scalar_type() == at::kChar &&
                  packed_weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  packed_weight.is_contiguous() && packed_weight.dim() == 2 &&
                  packed_weight.size(0) == 8 &&
                  packed_weight.size(1) == kGr00tFf1ResidentW8PerCore &&
                  packed_weight.nbytes() == 8 * kGr00tFf1ResidentW8PerCore &&
                  live_src_base == RpuGetDevAddr(packed_weight.data_ptr()),
              "GR00T resident FF1 preload owner/base/byte extent mismatch");
  check_gr00t_ff1_resident_spm(weight_spm_addr, kGr00tFf1ResidentW8PerCore);
  TORCH_CHECK(graph_dma::active(), "GR00T resident FF1 preload requires Graph BUILD or REPLAY");
  rpu_ddr_flush(packed_weight.data_ptr<int8_t>());
  RpuKernelGraph::active().keep_alive(packed_weight);
  graph_dma::stage_census_mutable_burst(
      live_src_base, packed_weight, GraphOuterFastDmaSide::Source,
      0, packed_weight.nbytes(), 8);
  // The raw DMA byte count is expressed as half elements, without conversion.
  rpu_launch_ddr_scatter_spm_dma_mutable(
      &live_src_base, 0, kGr00tFf1ResidentW8PerCore / sizeof(c10::Half),
      kGr00tFf1ResidentW8PerCore, weight_spm_addr, 8);
}

void rpu_launch_gr00t_ff1_resident_w8_kernel(
    uint32_t input_spm_addr, uint32_t weight_spm_addr,
    uint32_t output_spm_addr, uint32_t bias_spm_addr, const at::Tensor& scale) {
  const std::array<std::pair<uint32_t, size_t>, 4> ranges{{
      {input_spm_addr, 41 * 1536 * sizeof(c10::Half)},
      {weight_spm_addr, kGr00tFf1ResidentW8PerCore},
      {output_spm_addr, 41 * 768 * sizeof(c10::Half)},
      {bias_spm_addr, 768 * sizeof(c10::Half)}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    check_gr00t_ff1_resident_spm(ranges[i].first, ranges[i].second);
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(uint64_t{ranges[i].first} + ranges[i].second <= ranges[j].first ||
                      uint64_t{ranges[j].first} + ranges[j].second <= ranges[i].first,
                  "GR00T resident FF1 input/output/weight/bias intervals overlap");
    }
  }
  TORCH_CHECK(scale.defined() && scale.scalar_type() == at::kHalf &&
                  scale.device().type() == c10::DeviceType::PrivateUse1 &&
                  scale.dim() == 1 && scale.size(0) == 6144 &&
                  scale.is_contiguous() && scale.nbytes() == 12288,
              "GR00T resident FF1 requires original per-channel fp16 scale [6144]");
  const auto tile = rpu_pl_tiling::select_tile_acc16(41, 768, 1536, false);
  TORCH_CHECK(tile.m_tile == 512 && tile.n_tile == 48,
              "GR00T resident FF1 original auto-tile identity drift");
  constexpr KernelId id = KernelId::GR00T_FF1_RESIDENT_W8A16_M512N48K128;
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
  rpu_ddr_flush(scale.data_ptr<c10::Half>());
  Kernel_t* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "GR00T resident FF1 expansion kernel is absent");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, bias_spm_addr,
          41, 768, 1536, 1, 8, 3072, 1536, 0, 1, 1});
  // Only the weight address changes from DDR >> 8 to absolute local SPM.
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, weight_spm_addr);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {16, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

namespace {
void check_gr00t_qkv_multibase_spm(uint32_t address, size_t bytes) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "GR00T QKV multibase requires the initialized SPM allocator");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  TORCH_CHECK(address >= base && address % 256 == 0 &&
                  bytes <= SpmAllocator::SPM_USABLE &&
                  address - base <= SpmAllocator::SPM_USABLE - bytes,
              "GR00T QKV multibase absolute SPM interval/alignment is invalid");
}
}  // namespace

void rpu_launch_gr00t_qkv_multibase_w8_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    uint32_t q_output_spm_addr,
    uint32_t k_output_spm_addr,
    uint32_t v_output_spm_addr,
    uint32_t q_bias_spm_addr,
    uint32_t k_bias_spm_addr,
    uint32_t v_bias_spm_addr,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale) {
  // Exact self-attention COL TP8 M41/N1536/K1536, ACC16 with bias.
  // Retain all six installed DDR owners and the ordinary per-channel scales.
  for (const at::Tensor* weight : {&q_weight, &k_weight, &v_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kChar &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 1536 && weight->size(1) == 1536 &&
                    weight->nbytes() == 1536 * 1536,
                "GR00T QKV multibase requires installed contiguous device int8 [1536,1536]");
  }
  for (const at::Tensor* scale : {&q_scale, &k_scale, &v_scale}) {
    TORCH_CHECK(scale->defined() && scale->scalar_type() == at::kHalf &&
                    scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->is_contiguous() && scale->dim() == 1 &&
                    scale->size(0) == 1536 && scale->nbytes() == 3072,
                "GR00T QKV multibase requires original per-channel device fp16 scale [1536]");
  }
  const std::array<std::pair<uint32_t, size_t>, 7> ranges{{
      {input_spm_addr, 41 * 1536 * sizeof(c10::Half)},
      {q_output_spm_addr, 41 * 192 * sizeof(c10::Half)},
      {k_output_spm_addr, 41 * 192 * sizeof(c10::Half)},
      {v_output_spm_addr, 41 * 192 * sizeof(c10::Half)},
      {q_bias_spm_addr, 192 * sizeof(c10::Half)},
      {k_bias_spm_addr, 192 * sizeof(c10::Half)},
      {v_bias_spm_addr, 192 * sizeof(c10::Half)}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    check_gr00t_qkv_multibase_spm(ranges[i].first, ranges[i].second);
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(uint64_t{ranges[i].first} + ranges[i].second <= ranges[j].first ||
                      uint64_t{ranges[j].first} + ranges[j].second <= ranges[i].first,
                  "GR00T QKV multibase input/output/bias intervals overlap");
    }
  }
  const auto tile = rpu_pl_tiling::select_tile_acc16(41, 192, 1536, false);
  TORCH_CHECK(tile.m_tile == 592 && tile.n_tile == 32,
              "GR00T QKV multibase original auto-tile identity drift");
  for (const at::Tensor* tensor :
       {&q_weight, &k_weight, &v_weight, &q_scale, &k_scale, &v_scale}) {
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "GR00T QKV multibase DDR operand is not 256-byte aligned");
  }
  constexpr KernelId id = KernelId::GR00T_QKV_MULTIBASE_W8A16_M592N32K128;
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, q_weight},
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, q_scale},
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, k_weight},
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27}, k_scale},
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, v_weight},
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, v_scale}});
  for (const at::Tensor* tensor :
       {&q_weight, &k_weight, &v_weight, &q_scale, &k_scale, &v_scale}) {
    rpu_ddr_flush(tensor->data_ptr());
  }
  Kernel_t* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "GR00T QKV multibase expansion kernel is absent");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, q_output_spm_addr, q_bias_spm_addr,
          41, 192, 1536, 1, 8, 3072, 384, 0, 1, 1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 28, k_bias_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 30, k_output_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, v_bias_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 38, v_output_spm_addr);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {18, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

bool rpu_g05_kv_weight_is_replicated_fp16(const at::Tensor& weight) {
  if (!weight.defined() || weight.scalar_type() != at::kHalf ||
      weight.device().type() != c10::DeviceType::PrivateUse1 ||
      weight.dim() != 2 || weight.size(0) != 2048 || weight.size(1) != 1024 ||
      !weight.is_contiguous()) return false;
  // Installed TP8 col swizzle: [local_N/16,K/16,core,16,16].
  const auto stripes = weight.cpu().view({16, 64, 8, 256});
  for (int group : {0, 4}) {
    const auto owner = stripes.select(2, group).contiguous().view(at::kByte);
    for (int replica = 1; replica < 4; ++replica) {
      if (!at::equal(owner, stripes.select(2, group + replica).contiguous().view(at::kByte)))
        return false;
    }
  }
  return true;
}

void rpu_launch_g05_kv_owner_linear_fp16_acc32_kernel(
    uint32_t input, const at::Tensor& key_weight, const at::Tensor& value_weight,
    uint32_t key_output, uint32_t value_output) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 256 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 KV owner requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {key_output,output_bytes},{value_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 KV owner SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,key_output,output_bytes) &&
                  disjoint(input,input_bytes,value_output,output_bytes) &&
                  disjoint(key_output,output_bytes,value_output,output_bytes),
              "G05 KV owner SPM intervals overlap");
  for (const auto* weight : {&key_weight,&value_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kHalf &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 2048 && weight->size(1) == 1024,
                "G05 KV owner requires installed device FP16 [2048,1024]");
  }
  const auto tile = rpu_pl_tiling::select_tile_acc32(32,256,1024,true);
  TORCH_CHECK(tile.m_tile == 240 && tile.n_tile == 112,
              "G05 KV owner original ACC32 auto-tile identity drift");
  constexpr auto id = KernelId::G05_KV_OWNER_FP16_ACC32_M32N256K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id) &&
                  KernelCache::instance().has_loaded(KernelId::G05_KV_OWNER_FANOUT_M32D256),
              "G05 KV owner requires both expansion payloads");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},key_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},value_weight}});
  rpu_ddr_flush(key_weight.data_ptr<c10::Half>());
  rpu_ddr_flush(value_weight.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 KV owner Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,key_output,0,32,256,1024,0,8,2048,512,0,1,2});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,value_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{3,1,1},{0,1,2,3,4,5,6,7});
  // The preceding launch completes all owner writes. No per-core counters or
  // cross-replay synchronization state; fanout preserves the original KV ABI.
  auto* fanout = GET_KERNEL(KernelId::G05_KV_OWNER_FANOUT_M32D256);
  TORCH_CHECK(fanout, "G05 KV owner fanout payload disappeared");
  fanout->reset_regs();
  rpu_pl_tiling::set_u32_reg_pair(*fanout,0,key_output - base);
  rpu_pl_tiling::set_u32_reg_pair(*fanout,2,value_output - base);
  queue->enqueu_kernel(*fanout,{1,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_q_gate_linear_fp16_acc32_kernel(
    uint32_t input, const at::Tensor& query_weight, const at::Tensor& gate_weight,
    uint32_t query_output, uint32_t gate_output) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 256 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Q/gate requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {query_output,output_bytes},{gate_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Q/gate SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,query_output,output_bytes) &&
                  disjoint(input,input_bytes,gate_output,output_bytes) &&
                  disjoint(query_output,output_bytes,gate_output,output_bytes),
              "G05 Q/gate SPM intervals overlap");
  for (const auto* weight : {&query_weight,&gate_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kHalf &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 2048 && weight->size(1) == 1024,
                "G05 Q/gate requires installed device FP16 [2048,1024]");
  }
  const auto tile = rpu_pl_tiling::select_tile_acc32(32,256,1024,true);
  TORCH_CHECK(tile.m_tile == 240 && tile.n_tile == 112,
              "G05 Q/gate original ACC32 auto-tile identity drift");
  constexpr auto id = KernelId::G05_Q_GATE_FP16_ACC32_M32N256K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Q/gate requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},query_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},gate_weight}});
  rpu_ddr_flush(query_weight.data_ptr<c10::Half>());
  rpu_ddr_flush(gate_weight.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Q/gate Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,query_output,0,32,256,1024,0,8,2048,512,0,1,2});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,gate_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{6,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_vision_qkv_linear_fp16_acc32_kernel(
    uint32_t input, const at::Tensor& query_weight, const at::Tensor& key_weight,
    const at::Tensor& value_weight, uint32_t query_output, uint32_t key_output,
    uint32_t value_output, uint32_t query_bias, uint32_t key_bias,
    uint32_t value_bias) {
  constexpr size_t input_bytes = 256 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 256 * 128 * sizeof(c10::Half);
  constexpr size_t bias_bytes = 128 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Vision QKV requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t,size_t>,7> regions{{
      {input,input_bytes}, {query_output,output_bytes}, {key_output,output_bytes},
      {value_output,output_bytes}, {query_bias,bias_bytes}, {key_bias,bias_bytes},
      {value_bias,bias_bytes}}};
  for (const auto& region : regions) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    uint64_t(region.first) - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Vision QKV SPM interval/alignment is invalid");
  }
  // All seven roots are independent in the exact K1 layout. In particular no
  // projection may overwrite another projection's still-live input or bias.
  for (size_t i = 0; i < regions.size(); ++i) {
    for (size_t j = i + 1; j < regions.size(); ++j) {
      TORCH_CHECK(uint64_t(regions[i].first) + regions[i].second <= regions[j].first ||
                      uint64_t(regions[j].first) + regions[j].second <= regions[i].first,
                  "G05 Vision QKV SPM intervals overlap");
    }
  }
  for (const auto* weight : {&query_weight,&key_weight,&value_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kHalf &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 1024 && weight->size(1) == 1024,
                "G05 Vision QKV requires installed device FP16 [1024,1024]");
  }
  const auto tile = rpu_pl_tiling::select_tile_acc32(256,128,1024,true);
  TORCH_CHECK(tile.m_tile == 352 && tile.n_tile == 64,
              "G05 Vision QKV original ACC32 auto-tile identity drift");
  constexpr auto id = KernelId::G05_VISION_QKV_FP16_ACC32_M256N128K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Vision QKV requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},query_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},key_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,30,31},value_weight}});
  for (const auto* weight : {&query_weight,&key_weight,&value_weight})
    rpu_ddr_flush(weight->data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Vision QKV Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,query_output,query_bias,256,128,1024,1,8,2048,256,0,1,2});
  for (const auto& field : {std::pair<uint32_t,uint32_t>{26,key_output},
                            {28,key_bias},{32,value_output},{34,value_bias}})
    rpu_pl_tiling::set_u32_reg_pair(*kernel,field.first,field.second);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{6,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_vision_qkv_linear_w8a16_acc32_kernel(
    uint32_t input, const at::Tensor& query_weight, const at::Tensor& key_weight,
    const at::Tensor& value_weight, uint32_t query_output, uint32_t key_output,
    uint32_t value_output, uint32_t query_bias, uint32_t key_bias,
    uint32_t value_bias, const at::Tensor& query_scale,
    const at::Tensor& key_scale, const at::Tensor& value_scale, bool linear_acc32) {
  constexpr size_t input_bytes = 256 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 256 * 128 * sizeof(c10::Half);
  constexpr size_t bias_bytes = 128 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Vision QKV requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t,size_t>,7> regions{{
      {input,input_bytes}, {query_output,output_bytes}, {key_output,output_bytes},
      {value_output,output_bytes}, {query_bias,bias_bytes}, {key_bias,bias_bytes},
      {value_bias,bias_bytes}}};
  for (const auto& region : regions) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    uint64_t(region.first) - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Vision QKV SPM interval/alignment is invalid");
  }
  // All seven roots are independent in the exact K1 layout. In particular no
  // projection may overwrite another projection's still-live input or bias.
  for (size_t i = 0; i < regions.size(); ++i) {
    for (size_t j = i + 1; j < regions.size(); ++j) {
      TORCH_CHECK(uint64_t(regions[i].first) + regions[i].second <= regions[j].first ||
                      uint64_t(regions[j].first) + regions[j].second <= regions[i].first,
                  "G05 Vision QKV SPM intervals overlap");
    }
  }
  for (const auto* weight : {&query_weight,&key_weight,&value_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kChar &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 1024 && weight->size(1) == 1024,
                "G05 Vision QKV requires installed device INT8 [1024,1024]");
  }
  for (const auto* scale : {&query_scale,&key_scale,&value_scale}) {
    TORCH_CHECK(scale->defined() && scale->scalar_type() == at::kHalf &&
                    scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->is_contiguous() && scale->dim() == 1 && scale->size(0) == 1024,
                "G05 Vision QKV W8 requires FP16 per-channel device scales [1024]");
  }
  // ACC16 retains this specialization's N scheduling with its generated parent.
  if (linear_acc32) {
    const auto tile = rpu_pl_tiling::select_tile_acc32(256,128,1024,false);
    TORCH_CHECK(tile.m_tile == 400 && tile.n_tile == 48,
                "G05 Vision QKV original ACC32 auto-tile identity drift");
  }
  const auto id = linear_acc32 ? KernelId::G05_VISION_QKV_W8A16_ACC32_M256N128K1024
      : KernelId::G05_VISION_QKV_W8A16_ACC16_M256N128K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Vision QKV requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},query_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},key_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,30,31},value_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},query_scale},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,36,37},key_scale},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,38,39},value_scale}});
  for (const auto* weight : {&query_weight,&key_weight,&value_weight})
    rpu_ddr_flush(weight->data_ptr<int8_t>());
  for (const auto* scale : {&query_scale,&key_scale,&value_scale})
    rpu_ddr_flush(scale->data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Vision QKV Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,query_output,query_bias,256,128,1024,1,8,2048,256,0,1,1});
  for (const auto& field : {std::pair<uint32_t,uint32_t>{26,key_output},
                            {28,key_bias},{32,value_output},{34,value_bias}})
    rpu_pl_tiling::set_u32_reg_pair(*kernel,field.first,field.second);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{9,1,1},{0,1,2,3,4,5,6,7});
}


void rpu_launch_g05_gate_up_linear_fp16_acc32_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 512 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {gate_output,output_bytes},{up_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Gate/Up SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,gate_output,output_bytes) &&
                  disjoint(input,input_bytes,up_output,output_bytes) &&
                  disjoint(gate_output,output_bytes,up_output,output_bytes),
              "G05 Gate/Up SPM intervals overlap");
  for (const auto* weight : {&gate_weight,&up_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kHalf &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 4096 && weight->size(1) == 1024,
                "G05 Gate/Up requires installed device FP16 [4096,1024]");
  }
  const auto tile = rpu_pl_tiling::select_tile_acc32(32,512,1024,true);
  TORCH_CHECK(tile.m_tile == 304 && tile.n_tile == 80,
              "G05 Gate/Up original ACC32 auto-tile identity drift");
  constexpr auto id = KernelId::G05_GATE_UP_FP16_ACC32_M32N512K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Gate/Up requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},gate_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},up_weight}});
  rpu_ddr_flush(gate_weight.data_ptr<c10::Half>());
  rpu_ddr_flush(up_weight.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Gate/Up Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,gate_output,0,32,512,1024,0,8,2048,1024,0,1,2});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,up_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{14,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_text_gate_up_linear_fp16_acc32_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output) {
  constexpr size_t input_bytes = 256 * 2048 * sizeof(c10::Half);
  constexpr size_t output_bytes = 256 * 768 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Text Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {gate_output,output_bytes},{up_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    uint64_t(region.first) - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Text Gate/Up SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,gate_output,output_bytes) &&
                  disjoint(input,input_bytes,up_output,output_bytes) &&
                  disjoint(gate_output,output_bytes,up_output,output_bytes),
              "G05 Text Gate/Up SPM intervals overlap");
  for (const auto* weight : {&gate_weight,&up_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kHalf &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 6144 && weight->size(1) == 2048,
                "G05 Text Gate/Up requires installed device FP16 [6144,2048]");
  }
  const auto tile = rpu_pl_tiling::select_tile_acc32(256,768,2048,true);
  TORCH_CHECK(tile.m_tile == 352 && tile.n_tile == 64,
              "G05 Text Gate/Up original ACC32 auto-tile identity drift");
  constexpr auto id = KernelId::G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Text Gate/Up requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},gate_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},up_weight}});
  rpu_ddr_flush(gate_weight.data_ptr<c10::Half>());
  rpu_ddr_flush(up_weight.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Text Gate/Up Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,gate_output,0,256,768,2048,0,8,4096,1536,0,1,2});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,up_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{24,1,1},{0,1,2,3,4,5,6,7});
}

// Optional genuine generated ACC16 pairs; the ACC32 ABI remains independent.
void rpu_launch_g05_gate_up_linear_fp16_acc16_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 512 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {gate_output,output_bytes},{up_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Gate/Up SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,gate_output,output_bytes) &&
                  disjoint(input,input_bytes,up_output,output_bytes) &&
                  disjoint(gate_output,output_bytes,up_output,output_bytes),
              "G05 Gate/Up SPM intervals overlap");
  for (const auto* weight : {&gate_weight,&up_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kHalf &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 4096 && weight->size(1) == 1024,
                "G05 Gate/Up requires installed device FP16 [4096,1024]");
    const uint64_t address = RpuGetDevAddr(weight->data_ptr<c10::Half>());
    constexpr uint64_t ddr_limit = UINT64_C(1) << 40;
    TORCH_CHECK(address != 0 && address % 256 == 0 && address < ddr_limit &&
                    weight->nbytes() <= ddr_limit - address,
                "G05 FP16 ACC16 Gate/Up DDR range is not encodable");
  }
  const auto tile = rpu_pl_tiling::select_tile_acc16(32,512,1024,true);
  TORCH_CHECK(tile.m_tile == 432 && tile.n_tile == 80,
              "G05 Gate/Up original ACC16 auto-tile identity drift");
  constexpr auto id = KernelId::G05_GATE_UP_FP16_ACC16_M32N512K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Gate/Up requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},gate_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},up_weight}});
  rpu_ddr_flush(gate_weight.data_ptr<c10::Half>());
  rpu_ddr_flush(up_weight.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Gate/Up Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,gate_output,0,32,512,1024,0,8,2048,1024,0,1,2});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,up_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{14,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_text_gate_up_linear_fp16_acc16_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output) {
  constexpr size_t input_bytes = 256 * 2048 * sizeof(c10::Half);
  constexpr size_t output_bytes = 256 * 768 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Text Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {gate_output,output_bytes},{up_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    uint64_t(region.first) - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Text Gate/Up SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,gate_output,output_bytes) &&
                  disjoint(input,input_bytes,up_output,output_bytes) &&
                  disjoint(gate_output,output_bytes,up_output,output_bytes),
              "G05 Text Gate/Up SPM intervals overlap");
  for (const auto* weight : {&gate_weight,&up_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kHalf &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 6144 && weight->size(1) == 2048,
                "G05 Text Gate/Up requires installed device FP16 [6144,2048]");
    const uint64_t address = RpuGetDevAddr(weight->data_ptr<c10::Half>());
    constexpr uint64_t ddr_limit = UINT64_C(1) << 40;
    TORCH_CHECK(address != 0 && address % 256 == 0 && address < ddr_limit &&
                    weight->nbytes() <= ddr_limit - address,
                "G05 FP16 ACC16 Gate/Up DDR range is not encodable");
  }
  const auto tile = rpu_pl_tiling::select_tile_acc16(256,768,2048,true);
  TORCH_CHECK(tile.m_tile == 432 && tile.n_tile == 80,
              "G05 Text Gate/Up original ACC16 auto-tile identity drift");
  constexpr auto id = KernelId::G05_TEXT_GATE_UP_FP16_ACC16_M256N768K2048;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Text Gate/Up requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},gate_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},up_weight}});
  rpu_ddr_flush(gate_weight.data_ptr<c10::Half>());
  rpu_ddr_flush(up_weight.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Text Gate/Up Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,gate_output,0,256,768,2048,0,8,4096,1536,0,1,2});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,up_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{20,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_text_gate_up_linear_w8a16_acc32_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output,
    const at::Tensor& gate_scale, const at::Tensor& up_scale, bool linear_acc32) {
  constexpr size_t input_bytes = 256 * 2048 * sizeof(c10::Half);
  constexpr size_t output_bytes = 256 * 768 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Text Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {gate_output,output_bytes},{up_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    uint64_t(region.first) - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Text Gate/Up SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,gate_output,output_bytes) &&
                  disjoint(input,input_bytes,up_output,output_bytes) &&
                  disjoint(gate_output,output_bytes,up_output,output_bytes),
              "G05 Text Gate/Up SPM intervals overlap");
  for (const auto* weight : {&gate_weight,&up_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kChar &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 6144 && weight->size(1) == 2048,
                "G05 Text Gate/Up requires installed device int8 [6144,2048]");
  }
  for (const auto* scale : {&gate_scale,&up_scale}) {
    TORCH_CHECK(scale->defined() && scale->scalar_type() == at::kHalf &&
                    scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->is_contiguous() && scale->dim() == 1 &&
                    scale->numel() == 6144,
                "G05 Text Gate/Up W8 requires independent FP16 device scales [6144]");
  }
  // ACC16 retains this specialization's N scheduling with its generated parent.
  if (linear_acc32) {
    const auto tile = rpu_pl_tiling::select_tile_acc32(256,768,2048,false);
    TORCH_CHECK(tile.m_tile == 288 && tile.n_tile == 80,
                "G05 Text Gate/Up original ACC32 auto-tile identity drift");
  }
  const auto id = linear_acc32 ? KernelId::G05_TEXT_GATE_UP_W8A16_ACC32_M256N768K2048
      : KernelId::G05_TEXT_GATE_UP_W8A16_ACC16_M256N768K2048;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Text Gate/Up requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},gate_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},up_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},gate_scale},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,28,29},up_scale}});
  rpu_ddr_flush(gate_weight.data_ptr<int8_t>());
  rpu_ddr_flush(up_weight.data_ptr<int8_t>());
  rpu_ddr_flush(gate_scale.data_ptr<c10::Half>());
  rpu_ddr_flush(up_scale.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Text Gate/Up Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,gate_output,0,256,768,2048,0,8,4096,1536,0,1,1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,up_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{20,1,1},{0,1,2,3,4,5,6,7});
}

at::Tensor rpu_pack_g05_gateup_resident_fp16(
    const at::Tensor& gate, const at::Tensor& up) {
  auto unpack_core = [](const at::Tensor& weight) {
    TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kHalf &&
                    weight.device().type() == c10::DeviceType::PrivateUse1 &&
                    weight.dim() == 2 && weight.size(0) == 4096 &&
                    weight.size(1) == 1024 && weight.is_contiguous(),
                "G05 resident Gate/Up requires installed device FP16 [4096,1024]");
    // The installed layout is [local_N/16,K/16,core,16,16]. Preserve all
    // 512-byte stripes, changing only which core owns each contiguous range.
    return weight.cpu().view({32, 64, 8, 256})
        .permute({2, 0, 1, 3}).contiguous().view({8, 524288});
  };
  auto packed_cpu = at::cat({unpack_core(gate), unpack_core(up)}, 1);
  auto packed = packed_cpu.to(gate.device());
  rpu_ddr_flush_force(packed.data_ptr<c10::Half>());
  return packed;
}

void rpu_launch_g05_gateup_resident_fp16_kernel(
    uint32_t input_spm_addr, uint32_t weight_spm_addr,
    uint32_t output_spm_addr) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 512 * sizeof(c10::Half);
  constexpr size_t weight_bytes = 512 * 1024 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "G05 resident Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {
           std::pair<uint32_t, size_t>{input_spm_addr, input_bytes},
           {weight_spm_addr, weight_bytes}, {output_spm_addr, output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 resident Gate/Up SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a, size_t an, uint64_t b, size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input_spm_addr, input_bytes, weight_spm_addr, weight_bytes) &&
                  disjoint(output_spm_addr, output_bytes, weight_spm_addr, weight_bytes) &&
                  disjoint(input_spm_addr, input_bytes, output_spm_addr, output_bytes),
              "G05 resident Gate/Up SPM intervals overlap");
  const auto tile = rpu_pl_tiling::select_tile_acc32(32, 512, 1024, true);
  TORCH_CHECK(tile.m_tile == 304 && tile.n_tile == 80,
              "G05 resident Gate/Up original ACC32 auto-tile identity drift");
  auto* kernel = GET_KERNEL(KernelId::G05_GATEUP_RESIDENT_FP16_ACC32_M304N80K128);
  TORCH_CHECK(kernel, "G05 resident Gate/Up expansion payload is absent");
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0,
          32, 512, 1024, 0, 8, 2048, 1024, 0, 1, 2});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, weight_spm_addr);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {7, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

// G05 W8A16 Action variants retain the generic W8 ACC32 arithmetic.
bool rpu_g05_kv_weight_is_replicated_w8a16(const at::Tensor& weight, const at::Tensor& scale) {
  if (!weight.defined() || weight.scalar_type() != at::kChar ||
      weight.device().type() != c10::DeviceType::PrivateUse1 ||
      weight.dim() != 2 || weight.size(0) != 2048 || weight.size(1) != 1024 ||
      !weight.is_contiguous()) return false;
  if (!scale.defined() || scale.scalar_type() != at::kHalf ||
      scale.device().type() != c10::DeviceType::PrivateUse1 ||
      scale.dim() != 1 || scale.size(0) != 2048 || !scale.is_contiguous()) return false;
  const auto scales = scale.cpu().view({8,256});
  // Installed TP8 col swizzle: [local_N/16,K/32,core,16,32].
  const auto stripes = weight.cpu().view({16, 32, 8, 512});
  for (int group : {0, 4}) {
    const auto owner = stripes.select(2, group).contiguous().view(at::kByte);
    for (int replica = 1; replica < 4; ++replica) {
      if (!at::equal(owner, stripes.select(2, group + replica).contiguous().view(at::kByte)))
        return false;
      if (!at::equal(scales.select(0, group).contiguous().view(at::kByte),
                     scales.select(0, group + replica).contiguous().view(at::kByte))) return false;
    }
  }
  return true;
}

// Optional exact G05 mixed profile: W8 attention gate + K/V owner, ACC16, no fanout.
void rpu_launch_g05_gate_kv_owner_only_w8a16_acc16_kernel(
    uint32_t input, const at::Tensor& gate_weight,
    const at::Tensor& key_weight, const at::Tensor& value_weight,
    const at::Tensor& gate_scale, const at::Tensor& key_scale,
    const at::Tensor& value_scale, uint32_t gate_output,
    uint32_t key_output, uint32_t value_output) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 256 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 gate/KV owner requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, size_t>, 4> regions{{
      {input,input_bytes},{gate_output,output_bytes},
      {key_output,output_bytes},{value_output,output_bytes}}};
  for (size_t i=0;i<regions.size();++i) {
    const auto& region=regions[i];
    TORCH_CHECK(region.first>=base && region.first%256==0 &&
                    region.first-base<=SpmAllocator::SPM_USABLE-region.second,
                "G05 gate/KV owner SPM interval/alignment is invalid");
    for (size_t j=0;j<i;++j)
      TORCH_CHECK(uint64_t(region.first)+region.second<=regions[j].first ||
                      uint64_t(regions[j].first)+regions[j].second<=region.first,
                  "G05 gate/KV owner SPM intervals overlap");
  }
  for (const auto* weight:{&gate_weight,&key_weight,&value_weight})
    TORCH_CHECK(weight->defined() && weight->scalar_type()==at::kChar &&
                    weight->device().type()==c10::DeviceType::PrivateUse1 &&
                    weight->device()==gate_weight.device() && weight->is_contiguous() &&
                    weight->storage_offset()==0 && weight->dim()==2 &&
                    weight->size(0)==2048 && weight->size(1)==1024,
                "G05 gate/KV owner requires complete device INT8 [2048,1024]");
  for (const auto* scale:{&gate_scale,&key_scale,&value_scale})
    TORCH_CHECK(scale->defined() && scale->scalar_type()==at::kHalf &&
                    scale->device()==gate_weight.device() && scale->is_contiguous() &&
                    scale->storage_offset()==0 && scale->dim()==1 && scale->size(0)==2048,
                "G05 gate/KV owner requires complete device FP16 [2048] scales");
  const std::array<const at::Tensor*,6> owners{{
      &gate_weight,&key_weight,&value_weight,&gate_scale,&key_scale,&value_scale}};
  std::array<uint64_t,6> addresses{};
  for (size_t i=0;i<owners.size();++i) {
    const uint64_t address=RpuGetDevAddr(owners[i]->data_ptr());
    constexpr uint64_t limit=UINT64_C(1)<<40;
    TORCH_CHECK(address!=0 && address%256==0 && address<limit &&
                    owners[i]->nbytes()<=limit-address,
                "G05 gate/KV owner DDR span/alignment is invalid");
    for (size_t j=0;j<i;++j)
      TORCH_CHECK(address+owners[i]->nbytes()<=addresses[j] ||
                      addresses[j]+owners[j]->nbytes()<=address,
                  "G05 gate/KV owner DDR spans overlap");
    addresses[i]=address;
  }
  constexpr auto id=KernelId::G05_GATE_KV_OWNER_W8A16_ACC16_M32N256K1024_N128;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "G05 gate/KV owner optional payload absent");
  auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(
      id,std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::Int8ModelWeight,GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},gate_weight},
          {{GraphDdrRegisterRole::Int8ModelWeight,GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},key_weight},
          {{GraphDdrRegisterRole::Int8ModelWeight,GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,30,31},value_weight},
          {{GraphDdrRegisterRole::PerChannelScale,GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},gate_scale},
          {{GraphDdrRegisterRole::PerChannelScale,GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,28,29},key_scale},
          {{GraphDdrRegisterRole::PerChannelScale,GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,34,35},value_scale}});
  for (const auto* owner:owners) rpu_ddr_flush(owner->data_ptr());
  auto* kernel=GET_KERNEL(id);TORCH_CHECK(kernel,"G05 gate/KV owner payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,rpu_pl_tiling::ParallelLinearRegisterArgs{
      input,0,gate_output,0,32,256,1024,0,8,2048,512,0,1,1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,key_output);
  rpu_pl_tiling::set_u32_reg_pair(*kernel,32,value_output);
  writer.write(*kernel);
  auto* queue=GET_QUEUE(8);queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{4,1,1},{0,1,2,3,4,5,6,7});
  // K exists only on cores 0/4 and V only on 2/6. The caller must consume the
  // checked owner-only norm/RoPE/cache schedule; no nonowner SPM replica exists.
}

void rpu_launch_g05_kv_owner_linear_w8a16_acc32_kernel(
    uint32_t input, const at::Tensor& key_weight, const at::Tensor& value_weight,
    const at::Tensor& key_scale, const at::Tensor& value_scale,
    uint32_t key_output, uint32_t value_output, bool linear_acc32) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 256 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 KV owner requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {key_output,output_bytes},{value_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 KV owner SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,key_output,output_bytes) &&
                  disjoint(input,input_bytes,value_output,output_bytes) &&
                  disjoint(key_output,output_bytes,value_output,output_bytes),
              "G05 KV owner SPM intervals overlap");
  for (const auto* weight : {&key_weight,&value_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kChar &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 2048 && weight->size(1) == 1024,
                "G05 KV owner requires installed device signed-int8 [2048,1024]");
  }
  for (const auto* scale : {&key_scale,&value_scale}) {
    TORCH_CHECK(scale->defined() && scale->scalar_type() == at::kHalf &&
                    scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->is_contiguous() && scale->dim() == 1 && scale->size(0) == 2048,
                "G05 Action W8 requires an original device FP16 per-channel scale [2048]");
  }
  // ACC16 retains this specialization's N scheduling with its generated parent.
  if (linear_acc32) {
    const auto tile = rpu_pl_tiling::select_tile_acc32(32,256,1024,false);
    TORCH_CHECK(tile.m_tile == 192 && tile.n_tile == 128,
                "G05 KV owner original ACC32 auto-tile identity drift");
  }
  const auto id = linear_acc32 ? KernelId::G05_KV_OWNER_W8A16_ACC32_M32N256K1024
      : KernelId::G05_KV_OWNER_W8A16_ACC16_M32N256K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id) &&
                  KernelCache::instance().has_loaded(KernelId::G05_KV_OWNER_FANOUT_M32D256),
              "G05 KV owner requires both expansion payloads");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},key_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},value_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},key_scale},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,28,29},value_scale}});
  rpu_ddr_flush(key_weight.data_ptr<int8_t>());
  rpu_ddr_flush(key_scale.data_ptr<c10::Half>());
  rpu_ddr_flush(value_weight.data_ptr<int8_t>());
  rpu_ddr_flush(value_scale.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 KV owner Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,key_output,0,32,256,1024,0,8,2048,512,0,1,1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,value_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{2,1,1},{0,1,2,3,4,5,6,7});
  // The preceding launch completes all owner writes. No per-core counters or
  // cross-replay synchronization state; fanout preserves the original KV ABI.
  auto* fanout = GET_KERNEL(KernelId::G05_KV_OWNER_FANOUT_M32D256);
  TORCH_CHECK(fanout, "G05 KV owner fanout payload disappeared");
  fanout->reset_regs();
  rpu_pl_tiling::set_u32_reg_pair(*fanout,0,key_output - base);
  rpu_pl_tiling::set_u32_reg_pair(*fanout,2,value_output - base);
  queue->enqueu_kernel(*fanout,{1,1,1},{0,1,2,3,4,5,6,7});
}

static void launch_g05_qkv_gate_owner_linear_w8a16_acc32(
    uint32_t input, const at::Tensor& query_weight, const at::Tensor& gate_weight,
    const at::Tensor& key_weight, const at::Tensor& value_weight,
    const at::Tensor& query_scale, const at::Tensor& gate_scale,
    const at::Tensor& key_scale, const at::Tensor& value_scale,
    uint32_t query_output, uint32_t gate_output,
    uint32_t key_output, uint32_t value_output, bool restore_fanout, bool linear_acc32) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 256 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 QKV/gate requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, size_t>, 5> regions{{
      {input, input_bytes}, {query_output, output_bytes},
      {gate_output, output_bytes}, {key_output, output_bytes},
      {value_output, output_bytes}}};
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 QKV/gate SPM interval/alignment is invalid");
    for (size_t j = 0; j < i; ++j) {
      const auto& other = regions[j];
      TORCH_CHECK(uint64_t(region.first) + region.second <= other.first ||
                      uint64_t(other.first) + other.second <= region.first,
                  "G05 QKV/gate SPM intervals overlap");
    }
  }
  for (const auto* weight : {&query_weight, &gate_weight, &key_weight, &value_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kChar &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 2048 && weight->size(1) == 1024,
                "G05 QKV/gate requires installed device signed-int8 [2048,1024]");
  }
  for (const auto* scale : {&query_scale, &gate_scale, &key_scale, &value_scale}) {
    TORCH_CHECK(scale->defined() && scale->scalar_type() == at::kHalf &&
                    scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->is_contiguous() && scale->dim() == 1 && scale->size(0) == 2048,
                "G05 QKV/gate requires original device FP16 per-channel scales [2048]");
  }
  // Validate every owner before staging any Graph census. The encoding holds
  // bits8..39; no owner may cross that addressable interval.
  for (const auto* owner : {&query_weight, &gate_weight, &key_weight, &value_weight,
                            &query_scale, &gate_scale, &key_scale, &value_scale}) {
    const uint64_t address = RpuGetDevAddr(owner->data_ptr());
    constexpr uint64_t limit = UINT64_C(1) << 40;
    TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                    owner->nbytes() <= limit - address,
                "G05 QKV/gate DDR owner interval/alignment is invalid");
  }
  // ACC16 retains this specialization's N scheduling with its generated parent.
  if (linear_acc32) {
    const auto tile = rpu_pl_tiling::select_tile_acc32(32, 256, 1024, false);
    TORCH_CHECK(tile.m_tile == 192 && tile.n_tile == 128,
                "G05 QKV/gate original ACC32 auto-tile identity drift");
  }
  const auto id = linear_acc32 ? KernelId::G05_QKV_GATE_OWNER_W8A16_ACC32_M32N256K1024
      : KernelId::G05_QKV_GATE_OWNER_W8A16_ACC16_M32N256K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id) &&
                  (!restore_fanout || KernelCache::instance().has_loaded(KernelId::G05_KV_OWNER_FANOUT_M32D256)),
              "G05 QKV/gate requires both expansion payloads");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},query_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},gate_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,30,31},key_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,36,37},value_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},query_scale},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,28,29},gate_scale},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,34,35},key_scale},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,40,41},value_scale}});
  for (const auto* weight : {&query_weight, &gate_weight, &key_weight, &value_weight})
    rpu_ddr_flush(weight->data_ptr<int8_t>());
  for (const auto* scale : {&query_scale, &gate_scale, &key_scale, &value_scale})
    rpu_ddr_flush(scale->data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 QKV/gate Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,query_output,0,32,256,1024,0,8,2048,512,0,1,1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,gate_output);
  rpu_pl_tiling::set_u32_reg_pair(*kernel,32,key_output);
  rpu_pl_tiling::set_u32_reg_pair(*kernel,38,value_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{6,1,1},{0,1,2,3,4,5,6,7});
  if (!restore_fanout) return;
  // The launch boundary completes K/V owners before restoring the original
  // replicated KV layout. Fetch fanout only after consuming the pending kernel.
  auto* fanout = GET_KERNEL(KernelId::G05_KV_OWNER_FANOUT_M32D256);
  TORCH_CHECK(fanout, "G05 QKV/gate fanout payload disappeared");
  fanout->reset_regs();
  rpu_pl_tiling::set_u32_reg_pair(*fanout,0,key_output - base);
  rpu_pl_tiling::set_u32_reg_pair(*fanout,2,value_output - base);
  queue->enqueu_kernel(*fanout,{1,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_qkv_gate_owner_linear_w8a16_acc32_kernel(
    uint32_t input, const at::Tensor& query_weight, const at::Tensor& gate_weight,
    const at::Tensor& key_weight, const at::Tensor& value_weight,
    const at::Tensor& query_scale, const at::Tensor& gate_scale,
    const at::Tensor& key_scale, const at::Tensor& value_scale,
    uint32_t query_output, uint32_t gate_output,
    uint32_t key_output, uint32_t value_output, bool linear_acc32) {
  launch_g05_qkv_gate_owner_linear_w8a16_acc32(
      input, query_weight, gate_weight, key_weight, value_weight,
      query_scale, gate_scale, key_scale, value_scale,
      query_output, gate_output, key_output, value_output, true, linear_acc32);
}

void rpu_launch_g05_qkv_gate_owner_only_linear_w8a16_acc32_kernel(
    uint32_t input, const at::Tensor& query_weight, const at::Tensor& gate_weight,
    const at::Tensor& key_weight, const at::Tensor& value_weight,
    const at::Tensor& query_scale, const at::Tensor& gate_scale,
    const at::Tensor& key_scale, const at::Tensor& value_scale,
    uint32_t query_output, uint32_t gate_output,
    uint32_t key_output, uint32_t value_output, bool linear_acc32) {
  launch_g05_qkv_gate_owner_linear_w8a16_acc32(
      input, query_weight, gate_weight, key_weight, value_weight,
      query_scale, gate_scale, key_scale, value_scale,
      query_output, gate_output, key_output, value_output, false, linear_acc32);
}

void rpu_launch_g05_q_gate_linear_w8a16_acc32_kernel(
    uint32_t input, const at::Tensor& query_weight, const at::Tensor& gate_weight,
    const at::Tensor& query_scale, const at::Tensor& gate_scale,
    uint32_t query_output, uint32_t gate_output, bool linear_acc32) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 256 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Q/gate requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {query_output,output_bytes},{gate_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Q/gate SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,query_output,output_bytes) &&
                  disjoint(input,input_bytes,gate_output,output_bytes) &&
                  disjoint(query_output,output_bytes,gate_output,output_bytes),
              "G05 Q/gate SPM intervals overlap");
  for (const auto* weight : {&query_weight,&gate_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kChar &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 2048 && weight->size(1) == 1024,
                "G05 Q/gate requires installed device signed-int8 [2048,1024]");
  }
  for (const auto* scale : {&query_scale,&gate_scale}) {
    TORCH_CHECK(scale->defined() && scale->scalar_type() == at::kHalf &&
                    scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->is_contiguous() && scale->dim() == 1 && scale->size(0) == 2048,
                "G05 Action W8 requires an original device FP16 per-channel scale [2048]");
  }
  // ACC16 retains this specialization's N scheduling with its generated parent.
  if (linear_acc32) {
    const auto tile = rpu_pl_tiling::select_tile_acc32(32,256,1024,false);
    TORCH_CHECK(tile.m_tile == 192 && tile.n_tile == 128,
                "G05 Q/gate original ACC32 auto-tile identity drift");
  }
  const auto id = linear_acc32 ? KernelId::G05_Q_GATE_W8A16_ACC32_M32N256K1024
      : KernelId::G05_Q_GATE_W8A16_ACC16_M32N256K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Q/gate requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},query_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},gate_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},query_scale},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,28,29},gate_scale}});
  rpu_ddr_flush(query_weight.data_ptr<int8_t>());
  rpu_ddr_flush(query_scale.data_ptr<c10::Half>());
  rpu_ddr_flush(gate_weight.data_ptr<int8_t>());
  rpu_ddr_flush(gate_scale.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Q/gate Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,query_output,0,32,256,1024,0,8,2048,512,0,1,1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,gate_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{4,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_gate_up_linear_w8a16_acc32_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_output, uint32_t up_output, bool linear_acc32) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 512 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {std::pair<uint32_t,size_t>{input,input_bytes},
                            {gate_output,output_bytes},{up_output,output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 Gate/Up SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a,size_t an,uint64_t b,size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input,input_bytes,gate_output,output_bytes) &&
                  disjoint(input,input_bytes,up_output,output_bytes) &&
                  disjoint(gate_output,output_bytes,up_output,output_bytes),
              "G05 Gate/Up SPM intervals overlap");
  for (const auto* weight : {&gate_weight,&up_weight}) {
    TORCH_CHECK(weight->defined() && weight->scalar_type() == at::kChar &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 4096 && weight->size(1) == 1024,
                "G05 Gate/Up requires installed device signed-int8 [4096,1024]");
  }
  for (const auto* scale : {&gate_scale,&up_scale}) {
    TORCH_CHECK(scale->defined() && scale->scalar_type() == at::kHalf &&
                    scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->is_contiguous() && scale->dim() == 1 && scale->size(0) == 4096,
                "G05 Action W8 requires an original device FP16 per-channel scale [4096]");
  }
  // ACC16 retains this specialization's N scheduling with its generated parent.
  if (linear_acc32) {
    const auto tile = rpu_pl_tiling::select_tile_acc32(32,512,1024,false);
    TORCH_CHECK(tile.m_tile == 240 && tile.n_tile == 96,
                "G05 Gate/Up original ACC32 auto-tile identity drift");
  }
  const auto id = linear_acc32 ? KernelId::G05_GATE_UP_W8A16_ACC32_M32N512K1024
      : KernelId::G05_GATE_UP_W8A16_ACC16_M32N512K1024;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 Gate/Up requires its expansion payload");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},gate_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},up_weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},gate_scale},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,28,29},up_scale}});
  rpu_ddr_flush(gate_weight.data_ptr<int8_t>());
  rpu_ddr_flush(gate_scale.data_ptr<c10::Half>());
  rpu_ddr_flush(up_weight.data_ptr<int8_t>());
  rpu_ddr_flush(up_scale.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 Gate/Up Linear payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input,0,gate_output,0,32,512,1024,0,8,2048,1024,0,1,1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel,26,up_output);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{12,1,1},{0,1,2,3,4,5,6,7});
}

std::pair<at::Tensor, at::Tensor> rpu_pack_g05_o_resident_nvfp4(
    at::TensorList weights, at::TensorList scales) {
  TORCH_CHECK(!RpuKernelGraph::has_active(), "G05 NVFP4 O packing is cold-only");
  TORCH_CHECK(weights.size() == 24 && scales.size() == 24,
              "G05 NVFP4 O requires all 24 installed weight/scale pairs");
  constexpr uint64_t limit = UINT64_C(1) << 40;
  std::vector<std::pair<uint64_t, size_t>> intervals;
  // Prove every original owner before any CPU read or packed allocation.
  for (int layer = 0; layer < 24; ++layer) {
    for (int family = 0; family < 2; ++family) {
      const auto& t = family == 0 ? weights[layer] : scales[layer];
      TORCH_CHECK(t.defined() && t.scalar_type() == at::kByte &&
                      t.device().type() == at::kPrivateUse1 &&
                      t.device() == weights[0].device() && t.is_contiguous() &&
                      t.storage_offset() == 0 && t.sizes() ==
                          (family == 0 ? at::IntArrayRef({1024,1024}) : at::IntArrayRef({128,1024})),
                  "G05 NVFP4 O needs complete striped-v2 uint8 device owners");
      const auto address = RpuGetDevAddr(t.data_ptr<uint8_t>());
      TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                      t.nbytes() <= limit - address, "G05 NVFP4 O invalid DDR interval");
      for (const auto& other : intervals)
        TORCH_CHECK(address + t.nbytes() <= other.first ||
                        other.first + other.second <= address,
                    "G05 NVFP4 O original DDR owners overlap");
      intervals.emplace_back(address, t.nbytes());
    }
  }
  std::vector<at::Tensor> w, scale;
  w.reserve(24); scale.reserve(24);
  for (int layer = 0; layer < 24; ++layer) {
    w.push_back(weights[layer].cpu().view({64,4,8,512})
        .permute({2,0,1,3}).contiguous().view({8,131072}));
    scale.push_back(scales[layer].cpu().view({8,8,64,2,16})
        .permute({1,0,2,3,4}).contiguous().view({8,16384}));
  }
  auto packed_w = at::cat(w,1).to(weights[0].device());
  auto packed_scale = at::cat(scale,1).to(weights[0].device());
  rpu_ddr_flush_force(packed_w.data_ptr<uint8_t>());
  rpu_ddr_flush_force(packed_scale.data_ptr<uint8_t>());
  return {packed_w, packed_scale};
}

void rpu_launch_g05_o_resident_nvfp4_preload(
    const at::Tensor& packed, uint32_t destination, int64_t family) {
  TORCH_CHECK(family == 0 || family == 1, "G05 NVFP4 O preload family must be FP4 or FP8");
  const int64_t bytes = 24 * (family == 0 ? 131072 : 16384);
  TORCH_CHECK(packed.defined() && packed.scalar_type() == at::kByte &&
                  packed.device().type() == at::kPrivateUse1 && packed.is_contiguous() &&
                  packed.storage_offset() == 0 && packed.sizes() == at::IntArrayRef({8,bytes}),
              "G05 NVFP4 O preload requires its complete packed uint8 owner");
  constexpr uint64_t limit = UINT64_C(1) << 40;
  const auto address = RpuGetDevAddr(packed.data_ptr<uint8_t>());
  TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                  packed.nbytes() <= limit - address, "G05 NVFP4 O preload DDR interval invalid");
  check_pi05_resident_spm(destination, bytes);
  rpu_ddr_flush(packed.data_ptr<uint8_t>());
  if (graph_dma::active())
    graph_dma::stage_census_fixed_burst(packed, GraphOuterFastDmaSide::Source, 0, packed.nbytes(), 8);
  // Existing raw scatter ABI counts half units; the typed census owner stays uint8.
  rpu_launch_ddr_scatter_spm_dma(reinterpret_cast<c10::Half*>(packed.data_ptr<uint8_t>()),
      bytes / sizeof(c10::Half), bytes, destination, 8);
}

void rpu_launch_g05_o_resident_nvfp4_acc16_kernel(
    uint32_t input, uint32_t weight, uint32_t output,
    uint32_t scale, uint32_t tensor_scale, uint16_t layer) {
  constexpr auto id = KernelId::G05_O_RESIDENT_NVFP4_ACC16_M32N1024K256_N80;
  TORCH_CHECK(layer < 24 && KernelCache::instance().has_loaded(id),
              "G05 NVFP4 O resident requires layer0..23 and its optional N80 payload");
  const std::array<std::pair<uint32_t,size_t>,5> spans{{
      {input,32*256*2}, {weight,131072}, {output,32*1024*2},
      {scale,16384}, {tensor_scale,96}}};
  for (size_t i = 0; i < spans.size(); ++i) {
    check_pi05_resident_spm(spans[i].first, spans[i].second);
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(spans[i].first) + spans[i].second <= spans[j].first ||
                      uint64_t(spans[j].first) + spans[j].second <= spans[i].first,
                  "G05 NVFP4 O resident absolute SPM operands overlap");
  }
  // Intrinsic no-DDR kernel: GP2/3 and GP20/21 are absolute SPM, never DDR>>8.
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 NVFP4 O resident payload disappeared");
  kernel->reset_regs();
  const auto u16 = [&](int r,uint16_t v) { kernel->set_regs(r,v); };
  const auto u32 = [&](int r,uint32_t v) { u16(r,v & 0xffff); u16(r+1,v >> 16); };
  u32(0,input); u32(2,weight); u32(4,output); u32(6,0);
  u16(8,32); u32(10,1024); u16(12,256); u16(13,4);
  u16(14,0); u16(15,8); u32(16,512); u32(18,2048);
  u32(20,scale); u32(22,2048); u32(24,tensor_scale); u16(26,0); u16(27,layer);
  u16(64,13); u16(65,1); u16(66,1);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{13,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_o_resident_nvfp4_acc32_kernel(
    uint32_t input, uint32_t weight, uint32_t output,
    uint32_t scale, uint32_t tensor_scale, uint16_t layer) {
  constexpr auto id = KernelId::G05_O_RESIDENT_NVFP4_ACC32_M32N1024K256_N80;
  TORCH_CHECK(layer < 24 && KernelCache::instance().has_loaded(id),
              "G05 NVFP4 O resident requires layer0..23 and its optional N80 payload");
  const std::array<std::pair<uint32_t,size_t>,5> spans{{
      {input,32*256*2}, {weight,131072}, {output,32*1024*2},
      {scale,16384}, {tensor_scale,96}}};
  for (size_t i = 0; i < spans.size(); ++i) {
    check_pi05_resident_spm(spans[i].first, spans[i].second);
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(spans[i].first) + spans[i].second <= spans[j].first ||
                      uint64_t(spans[j].first) + spans[j].second <= spans[i].first,
                  "G05 NVFP4 O resident absolute SPM operands overlap");
  }
  // Intrinsic no-DDR kernel: GP2/3 and GP20/21 are absolute SPM, never DDR>>8.
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 NVFP4 O resident payload disappeared");
  kernel->reset_regs();
  const auto u16 = [&](int r,uint16_t v) { kernel->set_regs(r,v); };
  const auto u32 = [&](int r,uint32_t v) { u16(r,v & 0xffff); u16(r+1,v >> 16); };
  u32(0,input); u32(2,weight); u32(4,output); u32(6,0);
  u16(8,32); u32(10,1024); u16(12,256); u16(13,4);
  u16(14,0); u16(15,8); u32(16,512); u32(18,2048);
  u32(20,scale); u32(22,2048); u32(24,tensor_scale); u16(26,0); u16(27,layer);
  u16(64,13); u16(65,1); u16(66,1);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{13,1,1},{0,1,2,3,4,5,6,7});
}

std::pair<at::Tensor, at::Tensor> rpu_pack_g05_down_resident_nvfp4(
    at::TensorList weights, at::TensorList scales) {
  TORCH_CHECK(!RpuKernelGraph::has_active(), "G05 NVFP4 Down packing is cold-only");
  TORCH_CHECK(weights.size() == 10 && scales.size() == 10,
              "G05 NVFP4 Down requires first 10 installed weight/scale pairs");
  constexpr uint64_t limit = UINT64_C(1) << 40;
  std::vector<std::pair<uint64_t, size_t>> intervals;
  // Prove every original owner before any CPU read or packed allocation.
  for (int layer = 0; layer < 10; ++layer) {
    for (int family = 0; family < 2; ++family) {
      const auto& t = family == 0 ? weights[layer] : scales[layer];
      TORCH_CHECK(t.defined() && t.scalar_type() == at::kByte &&
                      t.device().type() == at::kPrivateUse1 &&
                      t.device() == weights[0].device() && t.is_contiguous() &&
                      t.storage_offset() == 0 && t.sizes() ==
                          (family == 0 ? at::IntArrayRef({1024,2048}) : at::IntArrayRef({256,1024})),
                  "G05 NVFP4 Down needs complete striped-v2 uint8 device owners");
      const auto address = RpuGetDevAddr(t.data_ptr<uint8_t>());
      TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                      t.nbytes() <= limit - address, "G05 NVFP4 Down invalid DDR interval");
      for (const auto& other : intervals)
        TORCH_CHECK(address + t.nbytes() <= other.first ||
                        other.first + other.second <= address,
                    "G05 NVFP4 Down original DDR owners overlap");
      intervals.emplace_back(address, t.nbytes());
    }
  }
  std::vector<at::Tensor> w, scale;
  w.reserve(10); scale.reserve(10);
  for (int layer = 0; layer < 10; ++layer) {
    w.push_back(weights[layer].cpu().view({64,8,8,512})
        .permute({2,0,1,3}).contiguous().view({8,262144}));
    scale.push_back(scales[layer].cpu().view({16,8,64,2,16})
        .permute({1,0,2,3,4}).contiguous().view({8,32768}));
  }
  auto packed_w = at::cat(w,1).to(weights[0].device());
  auto packed_scale = at::cat(scale,1).to(weights[0].device());
  rpu_ddr_flush_force(packed_w.data_ptr<uint8_t>());
  rpu_ddr_flush_force(packed_scale.data_ptr<uint8_t>());
  return {packed_w, packed_scale};
}

void rpu_launch_g05_down_resident_nvfp4_preload(
    const at::Tensor& packed, uint32_t destination, int64_t family) {
  TORCH_CHECK(family == 0 || family == 1, "G05 NVFP4 Down preload family must be FP4 or FP8");
  const int64_t bytes = 10 * (family == 0 ? 262144 : 32768);
  TORCH_CHECK(packed.defined() && packed.scalar_type() == at::kByte &&
                  packed.device().type() == at::kPrivateUse1 && packed.is_contiguous() &&
                  packed.storage_offset() == 0 && packed.sizes() == at::IntArrayRef({8,bytes}),
              "G05 NVFP4 Down preload requires its complete packed uint8 owner");
  constexpr uint64_t limit = UINT64_C(1) << 40;
  const auto address = RpuGetDevAddr(packed.data_ptr<uint8_t>());
  TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                  packed.nbytes() <= limit - address, "G05 NVFP4 Down preload DDR interval invalid");
  check_pi05_resident_spm(destination, bytes);
  rpu_ddr_flush(packed.data_ptr<uint8_t>());
  if (graph_dma::active())
    graph_dma::stage_census_fixed_burst(packed, GraphOuterFastDmaSide::Source, 0, packed.nbytes(), 8);
  // Existing raw scatter ABI counts half units; the typed census owner stays uint8.
  rpu_launch_ddr_scatter_spm_dma(reinterpret_cast<c10::Half*>(packed.data_ptr<uint8_t>()),
      bytes / sizeof(c10::Half), bytes, destination, 8);
}

void rpu_launch_g05_down_resident_nvfp4_acc16_kernel(
    uint32_t input, uint32_t weight, uint32_t output,
    uint32_t scale, uint32_t tensor_scale, uint16_t layer) {
  constexpr auto id = KernelId::G05_DOWN_RESIDENT_NVFP4_ACC16_M32N1024K512_N64;
  TORCH_CHECK(layer < 10 && KernelCache::instance().has_loaded(id),
              "G05 NVFP4 Down resident requires layer0..9 and its optional N64 payload");
  const std::array<std::pair<uint32_t,size_t>,5> spans{{
      {input,32*512*2}, {weight,262144}, {output,32*1024*2},
      {scale,32768}, {tensor_scale,96}}};
  for (size_t i = 0; i < spans.size(); ++i) {
    check_pi05_resident_spm(spans[i].first, spans[i].second);
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(spans[i].first) + spans[i].second <= spans[j].first ||
                      uint64_t(spans[j].first) + spans[j].second <= spans[i].first,
                  "G05 NVFP4 Down resident absolute SPM operands overlap");
  }
  // Intrinsic no-DDR kernel: GP2/3 and GP20/21 are absolute SPM, never DDR>>8.
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 NVFP4 Down resident payload disappeared");
  kernel->reset_regs();
  const auto u16 = [&](int r,uint16_t v) { kernel->set_regs(r,v); };
  const auto u32 = [&](int r,uint32_t v) { u16(r,v & 0xffff); u16(r+1,v >> 16); };
  u32(0,input); u32(2,weight); u32(4,output); u32(6,0);
  u16(8,32); u32(10,1024); u16(12,512); u16(13,8);
  u16(14,0); u16(15,8); u32(16,1024); u32(18,2048);
  u32(20,scale); u32(22,2048); u32(24,tensor_scale); u16(26,0); u16(27,layer);
  u16(64,16); u16(65,1); u16(66,1);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{16,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_down_resident_nvfp4_acc32_kernel(
    uint32_t input, uint32_t weight, uint32_t output,
    uint32_t scale, uint32_t tensor_scale, uint16_t layer) {
  constexpr auto id = KernelId::G05_DOWN_RESIDENT_NVFP4_ACC32_M32N1024K512_N64;
  TORCH_CHECK(layer < 10 && KernelCache::instance().has_loaded(id),
              "G05 NVFP4 Down resident requires layer0..9 and its optional N64 payload");
  const std::array<std::pair<uint32_t,size_t>,5> spans{{
      {input,32*512*2}, {weight,262144}, {output,32*1024*2},
      {scale,32768}, {tensor_scale,96}}};
  for (size_t i = 0; i < spans.size(); ++i) {
    check_pi05_resident_spm(spans[i].first, spans[i].second);
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(spans[i].first) + spans[i].second <= spans[j].first ||
                      uint64_t(spans[j].first) + spans[j].second <= spans[i].first,
                  "G05 NVFP4 Down resident absolute SPM operands overlap");
  }
  // Intrinsic no-DDR kernel: GP2/3 and GP20/21 are absolute SPM, never DDR>>8.
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 NVFP4 Down resident payload disappeared");
  kernel->reset_regs();
  const auto u16 = [&](int r,uint16_t v) { kernel->set_regs(r,v); };
  const auto u32 = [&](int r,uint32_t v) { u16(r,v & 0xffff); u16(r+1,v >> 16); };
  u32(0,input); u32(2,weight); u32(4,output); u32(6,0);
  u16(8,32); u32(10,1024); u16(12,512); u16(13,8);
  u16(14,0); u16(15,8); u32(16,1024); u32(18,2048);
  u32(20,scale); u32(22,2048); u32(24,tensor_scale); u16(26,0); u16(27,layer);
  u16(64,16); u16(65,1); u16(66,1);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{16,1,1},{0,1,2,3,4,5,6,7});
}

at::Tensor rpu_pack_g05_o_resident_w8a16(at::TensorList weights) {
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "G05 O residency packing is a cold-install operation");
  TORCH_CHECK(weights.size() == 4,
              "G05 O residency requires exactly four installed O weights");
  constexpr uint64_t weight_bytes = 1024 * 2048;
  constexpr uint64_t address_limit = UINT64_C(1) << 40;
  // Validate all owners before any CPU copy or new allocation.
  for (const auto& weight : weights) {
    TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kChar &&
                    weight.device().type() == c10::DeviceType::PrivateUse1 &&
                    weight.device() == weights[0].device() && weight.is_contiguous() &&
                    weight.dim() == 2 && weight.size(0) == 1024 && weight.size(1) == 2048 &&
                    weight.storage_offset() == 0 && weight.nbytes() == weight_bytes,
                "G05 O residency requires four complete device int8 row-TP8 [1024,2048] owners");
    const uint64_t address = RpuGetDevAddr(weight.data_ptr<int8_t>());
    TORCH_CHECK(address != 0 && address % 256 == 0 && address < address_limit &&
                    weight_bytes <= address_limit - address,
                "G05 O residency weight DDR interval/alignment is invalid");
  }
  std::vector<at::Tensor> core_weights;
  core_weights.reserve(4);
  for (const auto& weight : weights) {
    // Installed row layout: [N/16,local_K/32,core,16,32]. Keep each
    // 512-byte stripe unchanged and place its core's stripes consecutively.
    core_weights.push_back(weight.cpu().view({64, 8, 8, 512})
        .permute({2, 0, 1, 3}).contiguous().view({8, 262144}));
  }
  auto packed = at::cat(core_weights, 1).to(weights[0].device());
  rpu_ddr_flush_force(packed.data_ptr<int8_t>());
  return packed;
}

void rpu_launch_g05_o_resident_w8a16_acc16_kernel(
    uint32_t input_spm_addr, uint32_t weight_spm_addr,
    uint32_t output_spm_addr, const at::Tensor& scale, bool n128) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 resident O requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, size_t>, 3> regions{{
      {input_spm_addr, 32 * 256 * sizeof(c10::Half)},
      {weight_spm_addr, 1024 * 256},
      {output_spm_addr, 32 * 1024 * sizeof(c10::Half)}}};
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    uint64_t(region.first) - base + region.second <= SpmAllocator::SPM_USABLE,
                "G05 resident O requires complete aligned absolute SPM operands");
    for (size_t j = 0; j < i; ++j) {
      const auto& other = regions[j];
      TORCH_CHECK(uint64_t(region.first) + region.second <= other.first ||
                      uint64_t(other.first) + other.second <= region.first,
                  "G05 resident O SPM operands overlap");
    }
  }
  TORCH_CHECK(scale.defined() && scale.scalar_type() == at::kHalf &&
                  scale.device().type() == c10::DeviceType::PrivateUse1 &&
                  scale.is_contiguous() && scale.dim() == 1 && scale.size(0) == 1024 &&
                  scale.storage_offset() == 0 && scale.nbytes() == 2048,
              "G05 resident O requires the complete FP16 DDR scale owner [1024]");
  const uint64_t scale_address = RpuGetDevAddr(scale.data_ptr<c10::Half>());
  constexpr uint64_t address_limit = UINT64_C(1) << 40;
  TORCH_CHECK(scale_address != 0 && scale_address % 256 == 0 &&
                  scale_address < address_limit && scale.nbytes() <= address_limit - scale_address,
              "G05 resident O scale DDR interval/alignment is invalid");
  if (!n128) {
    const auto tile = rpu_pl_tiling::select_tile_acc16(32, 1024, 256, false);
    TORCH_CHECK(tile.m_tile == 448 && tile.n_tile == 64,
                "G05 resident O original ACC16 auto-tile identity drift");
  }
  const auto id = n128 ? KernelId::G05_O_RESIDENT_W8A16_ACC16_M32N1024K256_N128
                       : KernelId::G05_O_RESIDENT_W8A16_ACC16_M32N1024K256;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "G05 resident O optional payload is absent");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},scale}});
  rpu_ddr_flush(scale.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 resident O payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0, 32, 1024, 256, 0, 8,
          512, 2048, 0, 0, 1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, weight_spm_addr);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {uint16_t(n128 ? 8 : 16), 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

at::Tensor rpu_pack_g05_down_partial_w8a16(const at::Tensor& weight) {
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "G05 partial Down packing is a cold-install operation");
  constexpr uint64_t bytes = 1024 * 4096;
  constexpr uint64_t limit = UINT64_C(1) << 40;
  TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kChar &&
                  weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  weight.is_contiguous() && weight.dim() == 2 &&
                  weight.size(0) == 1024 && weight.size(1) == 4096 &&
                  weight.storage_offset() == 0 && weight.nbytes() == bytes,
              "G05 partial Down requires the complete row-TP8 int8 [1024,4096] owner");
  const uint64_t address = RpuGetDevAddr(weight.data_ptr<int8_t>());
  TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                  bytes <= limit - address,
              "G05 partial Down weight DDR interval/alignment is invalid");
  // Keep the first 42 N16 stripes (14 original N48 tiles), including all
  // local K512 bytes. Every installed 512-byte stripe is copied unchanged.
  auto packed = weight.cpu().view({64, 16, 8, 512}).narrow(0, 0, 42)
      .permute({2, 0, 1, 3}).contiguous().view({8, 344064})
      .to(weight.device());
  rpu_ddr_flush_force(packed.data_ptr<int8_t>());
  return packed;
}

void rpu_launch_g05_down_partial_w8a16_preload(
    const at::Tensor& packed, uint32_t weight_spm_addr) {
  constexpr size_t bytes_per_core = 344064;
  constexpr uint64_t limit = UINT64_C(1) << 40;
  TORCH_CHECK(packed.defined() && packed.scalar_type() == at::kChar &&
                  packed.device().type() == c10::DeviceType::PrivateUse1 &&
                  packed.is_contiguous() && packed.dim() == 2 &&
                  packed.size(0) == 8 && packed.size(1) == bytes_per_core &&
                  packed.storage_offset() == 0 && packed.nbytes() == 8 * bytes_per_core,
              "G05 partial Down preload requires the complete int8 [8,344064] owner");
  const uint64_t address = RpuGetDevAddr(packed.data_ptr<int8_t>());
  TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                  packed.nbytes() <= limit - address,
              "G05 partial Down preload DDR interval/alignment is invalid");
  check_pi05_resident_spm(weight_spm_addr, bytes_per_core);
  rpu_ddr_flush(packed.data_ptr<int8_t>());
  if (graph_dma::active()) {
    graph_dma::stage_census_fixed_burst(
        packed, GraphOuterFastDmaSide::Source, 0, packed.nbytes(), 8);
  }
  // Preserve typed INT8 ownership; the established DMA ABI counts half units.
  rpu_launch_ddr_scatter_spm_dma(
      reinterpret_cast<c10::Half*>(packed.data_ptr<int8_t>()),
      bytes_per_core / sizeof(c10::Half), bytes_per_core, weight_spm_addr, 8);
}

void rpu_launch_g05_down_partial_w8a16_acc16_kernel(
    uint32_t input_spm_addr, const at::Tensor& weight,
    uint32_t weight_spm_addr, uint32_t output_spm_addr,
    const at::Tensor& scale) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 partial Down requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, size_t>, 3> regions{{
      {input_spm_addr, 32 * 512 * sizeof(c10::Half)},
      {weight_spm_addr, 344064},
      {output_spm_addr, 32 * 1024 * sizeof(c10::Half)}}};
  for (size_t i = 0; i < regions.size(); ++i) {
    const auto& region = regions[i];
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    uint64_t(region.first) - base + region.second <= SpmAllocator::SPM_USABLE,
                "G05 partial Down requires complete aligned absolute SPM operands");
    for (size_t j = 0; j < i; ++j) {
      const auto& other = regions[j];
      TORCH_CHECK(uint64_t(region.first) + region.second <= other.first ||
                      uint64_t(other.first) + other.second <= region.first,
                  "G05 partial Down SPM operands overlap");
    }
  }
  TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kChar &&
                  weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  weight.is_contiguous() && weight.dim() == 2 &&
                  weight.size(0) == 1024 && weight.size(1) == 4096 &&
                  weight.storage_offset() == 0 && weight.nbytes() == 1024 * 4096,
              "G05 partial Down requires the full original int8 DDR weight [1024,4096]");
  TORCH_CHECK(scale.defined() && scale.scalar_type() == at::kHalf &&
                  scale.device() == weight.device() && scale.is_contiguous() &&
                  scale.dim() == 1 && scale.size(0) == 1024 &&
                  scale.storage_offset() == 0 && scale.nbytes() == 2048,
              "G05 partial Down requires the complete FP16 DDR scale owner [1024]");
  constexpr uint64_t limit = UINT64_C(1) << 40;
  const uint64_t weight_address = RpuGetDevAddr(weight.data_ptr<int8_t>());
  const uint64_t scale_address = RpuGetDevAddr(scale.data_ptr<c10::Half>());
  for (const auto& owner : {std::make_pair(weight_address, weight.nbytes()),
                            std::make_pair(scale_address, scale.nbytes())}) {
    TORCH_CHECK(owner.first != 0 && owner.first % 256 == 0 && owner.first < limit &&
                    owner.second <= limit - owner.first,
                "G05 partial Down DDR operand interval/alignment is invalid");
  }
  TORCH_CHECK(weight_address + weight.nbytes() <= scale_address ||
                  scale_address + scale.nbytes() <= weight_address,
              "G05 partial Down weight and scale owners overlap");
  const auto tile = rpu_pl_tiling::select_tile_acc16(32, 1024, 512, false);
  TORCH_CHECK(tile.m_tile == 512 && tile.n_tile == 48,
              "G05 partial Down original ACC16 auto-tile identity drift");
  constexpr auto id = KernelId::G05_DOWN_PARTIAL672_W8A16_ACC16_M32N1024K512;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "G05 partial Down optional payload is absent");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},scale}});
  rpu_ddr_flush(weight.data_ptr<int8_t>());
  rpu_ddr_flush(scale.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 partial Down payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0,
          32, 1024, 512, 0, 8, 1024, 2048, 0, 0, 1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 24, weight_spm_addr);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {22, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

at::Tensor rpu_pack_g05_gateup_resident_w8a16(
    const at::Tensor& gate, const at::Tensor& up) {
  auto unpack_core = [](const at::Tensor& weight) {
    TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kChar &&
                    weight.device().type() == c10::DeviceType::PrivateUse1 &&
                    weight.dim() == 2 && weight.size(0) == 4096 &&
                    weight.size(1) == 1024 && weight.is_contiguous(),
                "G05 resident Gate/Up requires installed device signed-int8 [4096,1024]");
    // The installed layout is [local_N/16,K/32,core,16,32]. Preserve all
    // 512-byte stripes, changing only which core owns each contiguous range.
    return weight.cpu().view({32, 32, 8, 512})
        .permute({2, 0, 1, 3}).contiguous().view({8, 524288});
  };
  auto packed_cpu = at::cat({unpack_core(gate), unpack_core(up)}, 1);
  auto packed = packed_cpu.to(gate.device());
  rpu_ddr_flush_force(packed.data_ptr<int8_t>());
  return packed;
}

void rpu_launch_g05_gateup_resident_w8a16_kernel(
    uint32_t input_spm_addr, uint32_t weight_spm_addr,
    uint32_t output_spm_addr, const at::Tensor& scale, bool linear_acc32) {
  constexpr size_t input_bytes = 32 * 1024 * sizeof(c10::Half);
  constexpr size_t output_bytes = 32 * 512 * sizeof(c10::Half);
  constexpr size_t weight_bytes = 512 * 1024;
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "G05 resident Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto& region : {
           std::pair<uint32_t, size_t>{input_spm_addr, input_bytes},
           {weight_spm_addr, weight_bytes}, {output_spm_addr, output_bytes}}) {
    TORCH_CHECK(region.first >= base && region.first % 256 == 0 &&
                    region.first - base <= SpmAllocator::SPM_USABLE - region.second,
                "G05 resident Gate/Up SPM interval/alignment is invalid");
  }
  auto disjoint = [](uint64_t a, size_t an, uint64_t b, size_t bn) {
    return a + an <= b || b + bn <= a;
  };
  TORCH_CHECK(disjoint(input_spm_addr, input_bytes, weight_spm_addr, weight_bytes) &&
                  disjoint(output_spm_addr, output_bytes, weight_spm_addr, weight_bytes) &&
                  disjoint(input_spm_addr, input_bytes, output_spm_addr, output_bytes),
              "G05 resident Gate/Up SPM intervals overlap");
  // ACC16 retains this specialization's N scheduling with its generated parent.
  if (linear_acc32) {
    const auto tile = rpu_pl_tiling::select_tile_acc32(32, 512, 1024, false);
    TORCH_CHECK(tile.m_tile == 240 && tile.n_tile == 96,
                "G05 resident Gate/Up original ACC32 auto-tile identity drift");
  }
  TORCH_CHECK(scale.defined() && scale.scalar_type() == at::kHalf &&
                  scale.device().type() == c10::DeviceType::PrivateUse1 &&
                  scale.dim() == 1 && scale.size(0) == 4096 && scale.is_contiguous(),
              "G05 resident W8 requires original per-channel FP16 scale [4096]");
  const auto id = linear_acc32 ? KernelId::G05_GATEUP_RESIDENT_W8A16_ACC32_M240N96K128
      : KernelId::G05_GATEUP_RESIDENT_W8A16_ACC16_M352N96K128;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "G05 resident W8 payload is absent");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
             GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},scale}});
  rpu_ddr_flush(scale.data_ptr<c10::Half>());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 resident Gate/Up expansion payload is absent");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0,
          32, 512, 1024, 0, 8, 2048, 1024, 0, 1, 1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, weight_spm_addr);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {6, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_g05_gateup_resident_w8a16_preload(
    const at::Tensor& packed, uint32_t weight_spm_addr) {
  constexpr size_t bytes_per_core = 1048576;
  TORCH_CHECK(packed.defined() && packed.scalar_type() == at::kChar &&
                  packed.device().type() == c10::DeviceType::PrivateUse1 &&
                  packed.is_contiguous() && packed.dim() == 2 &&
                  packed.size(0) == 8 && packed.size(1) == bytes_per_core &&
                  packed.nbytes() == 8 * bytes_per_core,
              "G05 resident W8 preload requires its installed signed-int8 owner");
  check_pi05_resident_spm(weight_spm_addr, bytes_per_core);
  rpu_ddr_flush(packed.data_ptr<int8_t>());
  if (graph_dma::active()) {
    graph_dma::stage_census_fixed_burst(
        packed, GraphOuterFastDmaSide::Source, 0, packed.nbytes(), 8);
  }
  // The existing raw DMA ABI counts half-width units, with no data conversion.
  // Census ownership and byte extent remain the original signed-int8 tensor.
  rpu_launch_ddr_scatter_spm_dma(
      reinterpret_cast<c10::Half*>(packed.data_ptr<int8_t>()),
      bytes_per_core / sizeof(c10::Half), bytes_per_core, weight_spm_addr, 8);
}

// Helper: check shapes and produce useful error messages
static void check_linear_shapes(const Tensor &input, const Tensor &weight,
                                const c10::optional<Tensor> &bias_opt) {
  TORCH_CHECK(input.options().dtype() == at::kHalf,
              "rpu_linear: only half supported");
  if (weight.dim() != 2) {
    TORCH_CHECK(false, "weight must be 2-D (out_features, in_features), got ",
                weight.dim(), "D");
  }
  if (input.dim() < 1) {
    TORCH_CHECK(false, "input must have at least 1 dim, got ", input.dim(),
                "D");
  }
  int64_t in_features = weight.size(1);
  if (input.size(-1) != in_features) {
    TORCH_CHECK(false, "The last dimension of input (", input.size(-1),
                ") does not match weight.size(1) (in_features=", in_features,
                ")");
  }
  if (bias_opt.has_value()) {
    const Tensor &bias = bias_opt.value();
    TORCH_CHECK(bias.dim() <= 1, "bias must be 1-D or scalar (got ", bias.dim(),
                "D)");
    if (bias.dim() == 1) {
      TORCH_CHECK(bias.size(0) == weight.size(0), "bias length (", bias.size(0),
                  ") must equal out_features (", weight.size(0), ")");
    }
  }
}

// Row partition: 按 K 维度切分 weight
at::Tensor tp_row_swizzle_mc_weight(
        const at::Tensor& w,
        int64_t num_cores,
        int64_t dwidth)
{
    TORCH_CHECK(w.dim() == 2, "w must be 2D");
    auto sizes = w.sizes();
    int64_t N = sizes[0];
    int64_t K = sizes[1];

    int64_t num_ele_32B = 32 / dwidth;

    TORCH_CHECK(N % 16 == 0, "N must be divisible by 16");
    TORCH_CHECK(K % (num_ele_32B * num_cores) == 0,
                "K must be divisible by num_ele_32B * num_cores");

    // new shape = (N/16, 16, num_cores, K/num_ele_32B/num_cores, num_ele_32B)
    std::vector<int64_t> new_shape = {
        N / 16,
        16,
        num_cores,
        K / num_ele_32B / num_cores,
        num_ele_32B
    };

    at::Tensor v = w.view(new_shape);

    // permute(0, 3, 2, 1, 4)
    at::Tensor p = v.permute({0, 3, 2, 1, 4}).reshape({N, K});

    return p.contiguous();
}

// Col partition: 按 N 维度切分 weight
at::Tensor tp_col_swizzle_mc_weight(
        const at::Tensor& w,
        int64_t num_cores,
        int64_t dwidth)
{
    TORCH_CHECK(w.dim() == 2, "w must be 2D");
    auto sizes = w.sizes();
    int64_t N = sizes[0];
    int64_t K = sizes[1];

    int64_t num_ele_32B = 32 / dwidth;

    TORCH_CHECK(N % (16 * num_cores) == 0, "N must be divisible by 16 * num_cores");
    TORCH_CHECK(K % num_ele_32B == 0, "K must be divisible by num_ele_32B");

    // new shape = (num_cores, N/16/num_cores, 16, K/num_ele_32B, num_ele_32B)
    std::vector<int64_t> new_shape = {
        num_cores,
        N / 16 / num_cores,
        16,
        K / num_ele_32B,
        num_ele_32B
    };

    at::Tensor v = w.view(new_shape);

    // permute(1, 3, 0, 2, 4)
    at::Tensor p = v.permute({1, 3, 0, 2, 4}).reshape({N, K});

    return p.contiguous();
}

std::pair<at::Tensor, at::Tensor>
rpu_pack_pi05_kv1_stripe_w8a16(
    const at::Tensor& replicated_weight,
    const at::Tensor& replicated_scale) {
  constexpr int64_t kCores = 8;
  constexpr int64_t kLogicalN = 256;
  constexpr int64_t kPhysicalN = kCores * kLogicalN;
  constexpr int64_t kK = 1024;
  constexpr size_t kLogicalWeightBytes = kLogicalN * kK;
  constexpr size_t kLogicalScaleBytes =
      kLogicalN * sizeof(c10::Half);

  TORCH_CHECK(
      !RpuKernelGraph::has_active(),
      "Pi0.5 KV1 compact W8A16 repack is a cold-install operation and "
      "cannot run during graph capture/replay");
  TORCH_CHECK(
      replicated_weight.defined() &&
          replicated_weight.scalar_type() == at::kChar &&
          replicated_weight.device().type() ==
              c10::DeviceType::PrivateUse1 &&
          replicated_weight.dim() == 2 &&
          replicated_weight.size(0) == kPhysicalN &&
          replicated_weight.size(1) == kK &&
          replicated_weight.is_contiguous() &&
          replicated_weight.storage_offset() == 0 &&
          replicated_weight.nbytes() == kCores * kLogicalWeightBytes,
      "Pi0.5 KV1 compact repack requires the installed contiguous device "
      "int8 TP8-col-swizzled replicated weight [2048,1024]");
  TORCH_CHECK(
      replicated_scale.defined() &&
          replicated_scale.scalar_type() == at::kHalf &&
          replicated_scale.device() == replicated_weight.device() &&
          replicated_scale.dim() == 1 &&
          replicated_scale.size(0) == kPhysicalN &&
          replicated_scale.is_contiguous() &&
          replicated_scale.storage_offset() == 0 &&
          replicated_scale.nbytes() == kCores * kLogicalScaleBytes,
      "Pi0.5 KV1 compact repack requires the matching contiguous device "
      "fp16 replicated scale [2048]");

  // Invert the installed TP8 col partition
  // [local_N/16,K/32,core,16,32] without numeric conversion.  The Python
  // installer replicated the single logical KV head before this swizzle, so
  // all eight recovered [256,1024] byte planes (and scale planes) must match.
  auto installed_weight_cpu = replicated_weight.cpu();
  auto weight_replicas = installed_weight_cpu
      .view({16, 32, kCores, 16, 32})
      .permute({2, 0, 3, 1, 4})
      .contiguous()
      .view({kCores, kLogicalN, kK});
  const auto* weight_bytes = weight_replicas.data_ptr<int8_t>();
  for (int64_t core = 1; core < kCores; ++core) {
    TORCH_CHECK(
        std::memcmp(weight_bytes,
                    weight_bytes + core * kLogicalWeightBytes,
                    kLogicalWeightBytes) == 0,
        "Pi0.5 KV1 compact repack found non-identical replicated weight "
        "plane at core ", core);
  }

  auto installed_scale_cpu = replicated_scale.cpu();
  auto scale_replicas = installed_scale_cpu.view({kCores, kLogicalN});
  const auto* scale_bytes = static_cast<const unsigned char*>(
      static_cast<const void*>(scale_replicas.data_ptr<c10::Half>()));
  for (int64_t core = 1; core < kCores; ++core) {
    TORCH_CHECK(
        std::memcmp(scale_bytes,
                    scale_bytes + core * kLogicalScaleBytes,
                    kLogicalScaleBytes) == 0,
        "Pi0.5 KV1 compact repack found non-identical replicated scale "
        "plane at core ", core);
  }

  auto logical_weight_cpu = weight_replicas.select(0, 0).clone();
  auto logical_scale_cpu = scale_replicas.select(0, 0).clone();
  auto compact_weight_cpu =
      tp_col_swizzle_mc_weight(logical_weight_cpu, kCores, 1);

  // Fail loudly if the documented inverse/forward byte permutations ever
  // drift.  This also proves that the compact owner was derived solely by
  // selecting one exact replicated plane and re-striping it for local N=32.
  auto roundtrip_weight_cpu = tp_col_swizzle_mc_weight(
      logical_weight_cpu.repeat({kCores, 1}), kCores, 1);
  TORCH_CHECK(
      roundtrip_weight_cpu.nbytes() == installed_weight_cpu.nbytes() &&
          std::memcmp(roundtrip_weight_cpu.data_ptr<int8_t>(),
                      installed_weight_cpu.data_ptr<int8_t>(),
                      installed_weight_cpu.nbytes()) == 0,
      "Pi0.5 KV1 compact weight byte-permutation roundtrip failed");
  auto roundtrip_scale_cpu = logical_scale_cpu.repeat({kCores});
  TORCH_CHECK(
      roundtrip_scale_cpu.nbytes() == installed_scale_cpu.nbytes() &&
          std::memcmp(roundtrip_scale_cpu.data_ptr<c10::Half>(),
                      installed_scale_cpu.data_ptr<c10::Half>(),
                      installed_scale_cpu.nbytes()) == 0,
      "Pi0.5 KV1 compact scale byte-selection roundtrip failed");

  auto compact_weight = compact_weight_cpu.to(replicated_weight.device());
  auto compact_scale = logical_scale_cpu.to(replicated_scale.device());
  TORCH_CHECK(
      compact_weight.scalar_type() == at::kChar &&
          compact_weight.dim() == 2 &&
          compact_weight.size(0) == kLogicalN &&
          compact_weight.size(1) == kK && compact_weight.is_contiguous() &&
          compact_weight.storage_offset() == 0 &&
          compact_weight.nbytes() == kLogicalWeightBytes &&
          compact_scale.scalar_type() == at::kHalf &&
          compact_scale.dim() == 1 &&
          compact_scale.size(0) == kLogicalN &&
          compact_scale.is_contiguous() &&
          compact_scale.storage_offset() == 0 &&
          compact_scale.nbytes() == kLogicalScaleBytes,
      "Pi0.5 KV1 compact repack did not create exact contiguous device owners");

  const uint64_t compact_weight_addr =
      RpuGetDevAddr(compact_weight.data_ptr<int8_t>());
  const uint64_t compact_scale_addr =
      RpuGetDevAddr(compact_scale.data_ptr<c10::Half>());
  const uint64_t source_weight_addr =
      RpuGetDevAddr(replicated_weight.data_ptr<int8_t>());
  const uint64_t source_scale_addr =
      RpuGetDevAddr(replicated_scale.data_ptr<c10::Half>());
  auto disjoint = [](uint64_t lhs, size_t lhs_bytes,
                     uint64_t rhs, size_t rhs_bytes) {
    return lhs + lhs_bytes <= rhs || rhs + rhs_bytes <= lhs;
  };
  TORCH_CHECK(
      source_weight_addr != 0 && source_scale_addr != 0 &&
          compact_weight_addr != 0 && compact_scale_addr != 0 &&
          (compact_weight_addr & 255u) == 0 &&
          (compact_scale_addr & 255u) == 0 &&
          (compact_weight_addr >> 8) <= UINT32_MAX &&
          (compact_scale_addr >> 8) <= UINT32_MAX &&
          disjoint(compact_weight_addr, kLogicalWeightBytes,
                   compact_scale_addr, kLogicalScaleBytes) &&
          disjoint(compact_weight_addr, kLogicalWeightBytes,
                   source_weight_addr, replicated_weight.nbytes()) &&
          disjoint(compact_weight_addr, kLogicalWeightBytes,
                   source_scale_addr, replicated_scale.nbytes()) &&
          disjoint(compact_scale_addr, kLogicalScaleBytes,
                   source_weight_addr, replicated_weight.nbytes()) &&
          disjoint(compact_scale_addr, kLogicalScaleBytes,
                   source_scale_addr, replicated_scale.nbytes()),
      "Pi0.5 KV1 compact repack requires independent 256-byte-aligned device "
      "owners addressable by the shift8 ABI");

  rpu_ddr_flush_force(compact_weight.data_ptr<int8_t>());
  rpu_ddr_flush_force(compact_scale.data_ptr<c10::Half>());
  return {std::move(compact_weight), std::move(compact_scale)};
}

std::pair<at::Tensor, at::Tensor>
rpu_pack_pi05_kv1_direct_pair_w8a16(
    const at::Tensor& replicated_weight,
    const at::Tensor& replicated_scale) {
  constexpr int64_t kCores = 8;
  constexpr int64_t kLogicalN = 256;
  constexpr int64_t kPhysicalN = kCores * kLogicalN;
  constexpr int64_t kK = 1024;
  constexpr int64_t kPairWidth = 16;
  constexpr size_t kLogicalWeightBytes = kLogicalN * kK;
  constexpr size_t kLogicalScaleBytes =
      kLogicalN * sizeof(c10::Half);

  TORCH_CHECK(
      !RpuKernelGraph::has_active(),
      "Pi0.5 KV1 direct pair repack is a cold-install operation and "
      "cannot run during graph capture/replay");
  TORCH_CHECK(
      replicated_weight.defined() &&
          replicated_weight.scalar_type() == at::kChar &&
          replicated_weight.device().type() ==
              c10::DeviceType::PrivateUse1 &&
          replicated_weight.dim() == 2 &&
          replicated_weight.size(0) == kPhysicalN &&
          replicated_weight.size(1) == kK &&
          replicated_weight.is_contiguous() &&
          replicated_weight.storage_offset() == 0 &&
          replicated_weight.nbytes() == kCores * kLogicalWeightBytes,
      "Pi0.5 KV1 direct pair repack requires the installed contiguous device "
      "int8 TP8-col-swizzled replicated weight [2048,1024]");
  TORCH_CHECK(
      replicated_scale.defined() &&
          replicated_scale.scalar_type() == at::kHalf &&
          replicated_scale.device() == replicated_weight.device() &&
          replicated_scale.dim() == 1 &&
          replicated_scale.size(0) == kPhysicalN &&
          replicated_scale.is_contiguous() &&
          replicated_scale.storage_offset() == 0 &&
          replicated_scale.nbytes() == kCores * kLogicalScaleBytes,
      "Pi0.5 KV1 direct pair repack requires the matching contiguous device "
      "fp16 replicated scale [2048]");

  // Undo the installed replicated TP8 layout byte-for-byte.  This is the
  // same source proof as the stripe-gather packer, but the new owner below
  // deliberately uses a different logical channel order.
  auto installed_weight_cpu = replicated_weight.cpu();
  auto weight_replicas = installed_weight_cpu
      .view({16, 32, kCores, 16, 32})
      .permute({2, 0, 3, 1, 4})
      .contiguous()
      .view({kCores, kLogicalN, kK});
  const auto* weight_bytes = weight_replicas.data_ptr<int8_t>();
  for (int64_t core = 1; core < kCores; ++core) {
    TORCH_CHECK(
        std::memcmp(weight_bytes,
                    weight_bytes + core * kLogicalWeightBytes,
                    kLogicalWeightBytes) == 0,
        "Pi0.5 KV1 direct pair repack found non-identical replicated "
        "weight plane at core ", core);
  }

  auto installed_scale_cpu = replicated_scale.cpu();
  auto scale_replicas = installed_scale_cpu.view({kCores, kLogicalN});
  const auto* scale_bytes = static_cast<const unsigned char*>(
      static_cast<const void*>(scale_replicas.data_ptr<c10::Half>()));
  for (int64_t core = 1; core < kCores; ++core) {
    TORCH_CHECK(
        std::memcmp(scale_bytes,
                    scale_bytes + core * kLogicalScaleBytes,
                    kLogicalScaleBytes) == 0,
        "Pi0.5 KV1 direct pair repack found non-identical replicated "
        "scale plane at core ", core);
  }

  auto logical_weight_cpu = weight_replicas.select(0, 0).clone();
  auto logical_scale_cpu = scale_replicas.select(0, 0).clone();

  // Before the ordinary TP8 col swizzle, group logical D blocks as
  //   core c -> {db=c, db=c+8}, each db containing 16 adjacent channels.
  // The fixed M592/N32 kernel consequently writes [M, pair=2, lane=16]
  // on every core, which keeps both rotate-half partners local.
  auto pair_weight_cpu = logical_weight_cpu
      .view({2, kCores, kPairWidth, kK})
      .permute({1, 0, 2, 3})
      .contiguous()
      .view({kLogicalN, kK});
  auto pair_scale_cpu = logical_scale_cpu
      .view({2, kCores, kPairWidth})
      .permute({1, 0, 2})
      .contiguous()
      .view({kLogicalN});
  auto compact_weight_cpu =
      tp_col_swizzle_mc_weight(pair_weight_cpu, kCores, 1);

  // Invert the pair map independently of the replicated-source proof.  A
  // future change to either permutation must fail before any device owner exists.
  auto recovered_weight_cpu = pair_weight_cpu
      .view({kCores, 2, kPairWidth, kK})
      .permute({1, 0, 2, 3})
      .contiguous()
      .view({kLogicalN, kK});
  auto recovered_scale_cpu = pair_scale_cpu
      .view({kCores, 2, kPairWidth})
      .permute({1, 0, 2})
      .contiguous()
      .view({kLogicalN});
  TORCH_CHECK(
      std::memcmp(recovered_weight_cpu.data_ptr<int8_t>(),
                  logical_weight_cpu.data_ptr<int8_t>(),
                  kLogicalWeightBytes) == 0 &&
          std::memcmp(recovered_scale_cpu.data_ptr<c10::Half>(),
                      logical_scale_cpu.data_ptr<c10::Half>(),
                      kLogicalScaleBytes) == 0,
      "Pi0.5 KV1 direct pair logical channel permutation did not roundtrip");

  auto replicated_roundtrip = tp_col_swizzle_mc_weight(
      logical_weight_cpu.repeat({kCores, 1}), kCores, 1);
  TORCH_CHECK(
      replicated_roundtrip.nbytes() == installed_weight_cpu.nbytes() &&
          std::memcmp(replicated_roundtrip.data_ptr<int8_t>(),
                      installed_weight_cpu.data_ptr<int8_t>(),
                      installed_weight_cpu.nbytes()) == 0 &&
          std::memcmp(logical_scale_cpu.repeat({kCores}).data_ptr<c10::Half>(),
                      installed_scale_cpu.data_ptr<c10::Half>(),
                      installed_scale_cpu.nbytes()) == 0,
      "Pi0.5 KV1 direct pair replicated-source byte roundtrip failed");

  auto compact_weight = compact_weight_cpu.to(replicated_weight.device());
  auto compact_scale = pair_scale_cpu.to(replicated_scale.device());
  TORCH_CHECK(
      compact_weight.scalar_type() == at::kChar &&
          compact_weight.dim() == 2 &&
          compact_weight.sizes() == at::IntArrayRef({kLogicalN, kK}) &&
          compact_weight.is_contiguous() &&
          compact_weight.storage_offset() == 0 &&
          compact_weight.nbytes() == kLogicalWeightBytes &&
          compact_scale.scalar_type() == at::kHalf &&
          compact_scale.dim() == 1 && compact_scale.size(0) == kLogicalN &&
          compact_scale.is_contiguous() &&
          compact_scale.storage_offset() == 0 &&
          compact_scale.nbytes() == kLogicalScaleBytes,
      "Pi0.5 KV1 direct pair repack did not create exact contiguous device owners");

  const uint64_t compact_weight_addr =
      RpuGetDevAddr(compact_weight.data_ptr<int8_t>());
  const uint64_t compact_scale_addr =
      RpuGetDevAddr(compact_scale.data_ptr<c10::Half>());
  const uint64_t source_weight_addr =
      RpuGetDevAddr(replicated_weight.data_ptr<int8_t>());
  const uint64_t source_scale_addr =
      RpuGetDevAddr(replicated_scale.data_ptr<c10::Half>());
  auto disjoint = [](uint64_t lhs, size_t lhs_bytes,
                     uint64_t rhs, size_t rhs_bytes) {
    return lhs + lhs_bytes <= rhs || rhs + rhs_bytes <= lhs;
  };
  TORCH_CHECK(
      source_weight_addr != 0 && source_scale_addr != 0 &&
          compact_weight_addr != 0 && compact_scale_addr != 0 &&
          (compact_weight_addr & 255u) == 0 &&
          (compact_scale_addr & 255u) == 0 &&
          (compact_weight_addr >> 8) <= UINT32_MAX &&
          (compact_scale_addr >> 8) <= UINT32_MAX &&
          disjoint(compact_weight_addr, kLogicalWeightBytes,
                   compact_scale_addr, kLogicalScaleBytes) &&
          disjoint(compact_weight_addr, kLogicalWeightBytes,
                   source_weight_addr, replicated_weight.nbytes()) &&
          disjoint(compact_weight_addr, kLogicalWeightBytes,
                   source_scale_addr, replicated_scale.nbytes()) &&
          disjoint(compact_scale_addr, kLogicalScaleBytes,
                   source_weight_addr, replicated_weight.nbytes()) &&
          disjoint(compact_scale_addr, kLogicalScaleBytes,
                   source_scale_addr, replicated_scale.nbytes()),
      "Pi0.5 KV1 direct pair repack requires independent 256-byte-aligned "
      "device owners addressable by the shift8 ABI");

  rpu_ddr_flush_force(compact_weight.data_ptr<int8_t>());
  rpu_ddr_flush_force(compact_scale.data_ptr<c10::Half>());
  return {std::move(compact_weight), std::move(compact_scale)};
}

void rpu_validate_pi05_kv1_pair_owner_w8a16(
    const at::Tensor& k_replicated_weight,
    const at::Tensor& k_replicated_scale,
    const at::Tensor& v_replicated_weight,
    const at::Tensor& v_replicated_scale) {
  constexpr int64_t kCores = 8;
  constexpr int64_t kLogicalN = 256;
  constexpr int64_t kPhysicalN = kCores * kLogicalN;
  constexpr int64_t kK = 1024;
  const bool fp16 = k_replicated_weight.defined() &&
      k_replicated_weight.scalar_type() == at::kHalf;
  const int64_t weight_element_bytes = fp16 ? sizeof(c10::Half) : 1;
  const size_t kLogicalWeightBytes = kLogicalN * kK * weight_element_bytes;
  constexpr size_t kLogicalScaleBytes =
      kLogicalN * sizeof(c10::Half);
  const size_t kWeightBytes = kCores * kLogicalWeightBytes;
  constexpr size_t kScaleBytes = kCores * kLogicalScaleBytes;

  TORCH_CHECK(
      !RpuKernelGraph::has_active(),
      "Pi0.5 KV1 pair-owner validation is a cold-install operation and "
      "cannot run during graph capture/replay");

  auto validate_replicas = [&](const at::Tensor& weight,
                               const at::Tensor& scale,
                               const char* role) {
    TORCH_CHECK(
        weight.defined() && weight.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
            weight.device().type() == c10::DeviceType::PrivateUse1 &&
            weight.dim() == 2 && weight.size(0) == kPhysicalN &&
            weight.size(1) == kK && weight.is_contiguous() &&
            weight.storage_offset() == 0 && weight.nbytes() == kWeightBytes,
        "Pi0.5 KV1 pair-owner ", role,
        " weight must be the installed contiguous device matching-dtype "
        "TP8-col-swizzled [2048,1024] owner");
    TORCH_CHECK(fp16 ? !scale.defined() : (
        scale.defined() && scale.scalar_type() == at::kHalf &&
            scale.device() == weight.device() && scale.dim() == 1 &&
            scale.size(0) == kPhysicalN && scale.is_contiguous() &&
            scale.storage_offset() == 0 && scale.nbytes() == kScaleBytes),
        "Pi0.5 KV1 pair-owner ", role,
        " requires no scale for FP16, or the matching device fp16 [2048] W8 scale");

    // Invert the installer TP8 col swizzle without arithmetic. Every recovered
    // [256,1024] plane and every [256] scale plane must match replica zero.
    auto installed_weight_cpu = weight.cpu();
    auto weight_replicas = installed_weight_cpu
        .view({16, kK / (32 / weight_element_bytes), kCores,
               16, 32 / weight_element_bytes})
        .permute({2, 0, 3, 1, 4})
        .contiguous()
        .view({kCores, kLogicalN, kK});
    const auto* weight_bytes = static_cast<const unsigned char*>(weight_replicas.data_ptr());
    for (int64_t core = 1; core < kCores; ++core) {
      TORCH_CHECK(
          std::memcmp(weight_bytes,
                      weight_bytes + core * kLogicalWeightBytes,
                      kLogicalWeightBytes) == 0,
          "Pi0.5 KV1 pair-owner ", role,
          " weight replica differs byte-for-byte at core ", core);
    }

    at::Tensor installed_scale_cpu, scale_replicas;
    if (!fp16) {
    installed_scale_cpu = scale.cpu();
    scale_replicas = installed_scale_cpu.view({kCores, kLogicalN});
    const auto* scale_bytes = static_cast<const unsigned char*>(
        static_cast<const void*>(scale_replicas.data_ptr<c10::Half>()));
    for (int64_t core = 1; core < kCores; ++core) {
      TORCH_CHECK(
          std::memcmp(scale_bytes,
                      scale_bytes + core * kLogicalScaleBytes,
                      kLogicalScaleBytes) == 0,
          "Pi0.5 KV1 pair-owner ", role,
          " scale replica differs byte-for-byte at core ", core);
    }
    }

    // Prove the inverse permutation itself by reconstructing the exact
    // installed bytes. These temporaries remain CPU-only and are discarded.
    auto weight_roundtrip = tp_col_swizzle_mc_weight(
        weight_replicas.select(0, 0).repeat({kCores, 1}), kCores, weight_element_bytes);
    TORCH_CHECK(
        weight_roundtrip.nbytes() == installed_weight_cpu.nbytes() &&
            std::memcmp(weight_roundtrip.data_ptr(),
                        installed_weight_cpu.data_ptr(),
                        installed_weight_cpu.nbytes()) == 0,
        "Pi0.5 KV1 pair-owner ", role,
        " replica inverse-swizzle byte roundtrip failed");
    if (!fp16) {
    auto scale_roundtrip = scale_replicas.select(0, 0).repeat({kCores});
    TORCH_CHECK(
            scale_roundtrip.nbytes() == installed_scale_cpu.nbytes() &&
            std::memcmp(scale_roundtrip.data_ptr<c10::Half>(),
                        installed_scale_cpu.data_ptr<c10::Half>(),
                        installed_scale_cpu.nbytes()) == 0,
        "Pi0.5 KV1 pair-owner ", role,
        " replica inverse-swizzle byte roundtrip failed");
    }
  };

  validate_replicas(k_replicated_weight, k_replicated_scale, "K");
  validate_replicas(v_replicated_weight, v_replicated_scale, "V");
  TORCH_CHECK(
      k_replicated_weight.device() == v_replicated_weight.device() &&
          (fp16 || k_replicated_weight.device() == v_replicated_scale.device()),
      "Pi0.5 KV1 pair-owner K/V owners must share one device");

  std::vector<std::pair<uint64_t, size_t>> owner_intervals{
      {RpuGetDevAddr(k_replicated_weight.data_ptr()), kWeightBytes},
      {RpuGetDevAddr(v_replicated_weight.data_ptr()), kWeightBytes}};
  if (!fp16) {
    owner_intervals.emplace_back(RpuGetDevAddr(k_replicated_scale.data_ptr()), kScaleBytes);
    owner_intervals.emplace_back(RpuGetDevAddr(v_replicated_scale.data_ptr()), kScaleBytes);
  }
  auto disjoint = [](uint64_t lhs, size_t lhs_bytes,
                     uint64_t rhs, size_t rhs_bytes) {
    return lhs + lhs_bytes <= rhs || rhs + rhs_bytes <= lhs;
  };
  for (size_t index = 0; index < owner_intervals.size(); ++index) {
    const uint64_t address = owner_intervals[index].first;
    TORCH_CHECK(
        address != 0 && (address & 255u) == 0 &&
            (address >> 8) <= UINT32_MAX,
        "Pi0.5 KV1 pair-owner cold DDR owner ", index,
        " violates the aligned shift8 ABI");
    for (size_t other = index + 1; other < owner_intervals.size(); ++other) {
      TORCH_CHECK(
          disjoint(address, owner_intervals[index].second,
                   owner_intervals[other].first,
                   owner_intervals[other].second),
          "Pi0.5 KV1 pair-owner cold DDR owner intervals overlap at indices ",
          index, " and ", other);
    }
  }
}

void rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(
    const at::Tensor& k_replicated_weight,
    const at::Tensor& k_replicated_scale,
    const at::Tensor& v_replicated_weight,
    const at::Tensor& v_replicated_scale) {
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "Pi Prefill KV1 replica validation is cold-install only");
  const bool fp16 = k_replicated_weight.defined() &&
                    k_replicated_weight.scalar_type() == at::kHalf;
  const auto dtype = fp16 ? at::kHalf : at::kChar;
  const int64_t ktile = fp16 ? 16 : 32;
  constexpr int64_t cores = 8, logical_n = 256, physical_n = 2048, k = 2048;
  const size_t replica_bytes = logical_n * k * (fp16 ? 2 : 1);
  std::vector<std::pair<uint64_t,size_t>> intervals;
  const auto validate = [&](const at::Tensor& weight, const at::Tensor& scale) {
    TORCH_CHECK(weight.defined() && weight.device().type() == at::kPrivateUse1 &&
        weight.scalar_type() == dtype && weight.is_contiguous() &&
        weight.storage_offset() == 0 && weight.sizes() == at::IntArrayRef({physical_n,k}) &&
        weight.nbytes() == cores * replica_bytes,
        "Pi Prefill KV1 requires exact installed FP16 or W8 TP8 [2048,2048]");
    TORCH_CHECK(fp16 ? !scale.defined() :
        (scale.defined() && scale.device() == weight.device() && scale.scalar_type() == at::kHalf &&
         scale.is_contiguous() && scale.storage_offset() == 0 && scale.sizes() == at::IntArrayRef({physical_n})),
        "Pi Prefill KV1 FP16 must have no scales; W8 needs exact FP16 channel scales");
    auto installed = weight.cpu();
    auto replicas = installed.view({16,k/ktile,cores,16,ktile})
        .permute({2,0,3,1,4}).contiguous().view({cores,logical_n,k});
    const auto* bytes = static_cast<const unsigned char*>(replicas.data_ptr());
    for (int64_t core=1;core<cores;++core)
      TORCH_CHECK(std::memcmp(bytes,bytes+core*replica_bytes,replica_bytes)==0,
                  "Pi Prefill KV1 weight replica differs byte-for-byte at core ",core);
    auto roundtrip = tp_col_swizzle_mc_weight(
        replicas.select(0,0).repeat({cores,1}),cores,fp16 ? 2 : 1);
    TORCH_CHECK(roundtrip.nbytes()==installed.nbytes() &&
                std::memcmp(roundtrip.data_ptr(),installed.data_ptr(),installed.nbytes())==0,
                "Pi Prefill KV1 inverse-swizzle byte roundtrip failed");
    intervals.emplace_back(RpuGetDevAddr(weight.data_ptr()),weight.nbytes());
    if (!fp16) {
      auto scale_cpu=scale.cpu().view({cores,logical_n});
      const auto* scales=static_cast<const unsigned char*>(scale_cpu.data_ptr());
      constexpr size_t scale_bytes=logical_n*sizeof(c10::Half);
      for(int64_t core=1;core<cores;++core)
        TORCH_CHECK(std::memcmp(scales,scales+core*scale_bytes,scale_bytes)==0,
                    "Pi Prefill KV1 scale replica differs byte-for-byte at core ",core);
      intervals.emplace_back(RpuGetDevAddr(scale.data_ptr()),scale.nbytes());
    }
  };
  validate(k_replicated_weight,k_replicated_scale);
  validate(v_replicated_weight,v_replicated_scale);
  TORCH_CHECK(k_replicated_weight.device()==v_replicated_weight.device(),
              "Pi Prefill KV1 K/V owners must share one device");
  for(size_t i=0;i<intervals.size();++i) {
    const auto [address,bytes]=intervals[i];
    TORCH_CHECK(address && address%256==0 && (address>>8)<=UINT32_MAX,
                "Pi Prefill KV1 DDR owner exceeds aligned shift8 ABI");
    for(size_t j=0;j<i;++j)
      TORCH_CHECK(address+bytes<=intervals[j].first ||
                  intervals[j].first+intervals[j].second<=address,
                  "Pi Prefill KV1 DDR owners overlap");
  }
}

void rpu_launch_pi05_kv1_stripe_w8a16_m592n32_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& compact_weight,
    uint32_t output_spm_addr,
    const at::Tensor& compact_scale) {
  constexpr uint32_t kRows = 50;
  constexpr uint32_t kLocalN = 32;
  constexpr uint32_t kK = 1024;
  constexpr uint16_t kCores = 8;
  constexpr size_t kInputBytes =
      kRows * kK * sizeof(c10::Half);
  constexpr size_t kOutputBytes =
      kRows * kLocalN * sizeof(c10::Half);
  constexpr KernelId kKernelId =
      KernelId::PI05_KV1_STRIPE_W8A16_M592N32;

  TORCH_CHECK(
      RpuKernelGraph::has_active(),
      "Pi0.5 KV1 fixed M592N32 launcher requires an active retained graph");
  TORCH_CHECK(
      compact_weight.defined() && compact_weight.scalar_type() == at::kChar &&
          compact_weight.device().type() ==
              c10::DeviceType::PrivateUse1 &&
          compact_weight.dim() == 2 && compact_weight.size(0) == 256 &&
          compact_weight.size(1) == kK && compact_weight.is_contiguous() &&
          compact_weight.storage_offset() == 0 &&
          compact_weight.nbytes() == 256 * kK,
      "Pi0.5 KV1 fixed M592N32 requires an exact contiguous device int8 "
      "compact weight owner [256,1024]");
  TORCH_CHECK(
      compact_scale.defined() && compact_scale.scalar_type() == at::kHalf &&
          compact_scale.device() == compact_weight.device() &&
          compact_scale.dim() == 1 && compact_scale.size(0) == 256 &&
          compact_scale.is_contiguous() &&
          compact_scale.storage_offset() == 0 &&
          compact_scale.nbytes() == 256 * sizeof(c10::Half),
      "Pi0.5 KV1 fixed M592N32 requires an exact contiguous device fp16 "
      "compact scale owner [256]");
  TORCH_CHECK(
      SPM_ALLOC.is_initialized(),
      "Pi0.5 KV1 fixed M592N32 requires the initialized SPM allocator");

  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  auto check_spm_interval = [spm_base](uint32_t address, size_t bytes,
                                      const char* role) {
    const uint64_t absolute = address;
    TORCH_CHECK(
        absolute >= spm_base && (absolute & 255u) == 0 &&
            bytes <= SpmAllocator::SPM_USABLE &&
            absolute - spm_base <= SpmAllocator::SPM_USABLE - bytes,
        "Pi0.5 KV1 fixed M592N32 ", role,
        " must be a 256-byte-aligned absolute core-0 SPM interval");
  };
  check_spm_interval(input_spm_addr, kInputBytes, "input");
  check_spm_interval(output_spm_addr, kOutputBytes, "output");
  auto spm_disjoint = [](uint64_t lhs, size_t lhs_bytes,
                         uint64_t rhs, size_t rhs_bytes) {
    return lhs + lhs_bytes <= rhs || rhs + rhs_bytes <= lhs;
  };
  TORCH_CHECK(
      spm_disjoint(input_spm_addr, kInputBytes,
                   output_spm_addr, kOutputBytes),
      "Pi0.5 KV1 fixed M592N32 input/output SPM intervals overlap");

  const uint64_t weight_addr =
      RpuGetDevAddr(compact_weight.data_ptr<int8_t>());
  const uint64_t scale_addr =
      RpuGetDevAddr(compact_scale.data_ptr<c10::Half>());
  TORCH_CHECK(
      weight_addr != 0 && scale_addr != 0 &&
          (weight_addr & 255u) == 0 && (scale_addr & 255u) == 0 &&
          (weight_addr >> 8) <= UINT32_MAX &&
          (scale_addr >> 8) <= UINT32_MAX,
      "Pi0.5 KV1 fixed M592N32 DDR owners violate the shift8 ABI");

  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      kKernelId,
      std::vector<GraphDdrRegisterOperandSpec>{
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{
                  GraphDdrRegisterRole::Int8ModelWeight,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
              compact_weight},
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{
                  GraphDdrRegisterRole::PerChannelScale,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
              compact_scale}});
  rpu_ddr_flush(compact_weight.data_ptr<int8_t>());
  rpu_ddr_flush(compact_scale.data_ptr<c10::Half>());

  Kernel_t* kernel = GET_KERNEL(kKernelId);
  TORCH_CHECK(kernel != nullptr,
              "Pi0.5 KV1 fixed M592N32 W8A16 kernel is absent");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0,
          kRows, kLocalN, kK, 0, kCores,
          kK * sizeof(c10::Half), kLocalN * sizeof(c10::Half),
          0, 1, 1});
  // Registers 2/3 and 20/21 deliberately remain zero until the typed writer
  // binds the exact retained owners after every other register is final.
  writer.write(*kernel);

  auto* queue = GET_QUEUE(kCores);
  TORCH_CHECK(queue != nullptr,
              "Pi0.5 KV1 fixed M592N32 8-core queue is absent");
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {1, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& k_full_weight,
    const at::Tensor& k_full_scale,
    uint32_t k_output_spm_addr,
    const at::Tensor& v_full_weight,
    const at::Tensor& v_full_scale,
    uint32_t v_output_spm_addr) {
  constexpr uint32_t kRows = 50;
  constexpr uint32_t kLocalN = 32;
  constexpr uint32_t kK = 1024;
  constexpr uint32_t kPhysicalN = 2048;
  constexpr uint16_t kCores = 8;
  constexpr size_t kInputBytes = kRows * kK * sizeof(c10::Half);
  constexpr size_t kOutputBytes = kRows * kLocalN * sizeof(c10::Half);
  const bool fp16 = k_full_weight.defined() && k_full_weight.scalar_type() == at::kHalf;
  const size_t kWeightBytes = kPhysicalN * kK * (fp16 ? sizeof(c10::Half) : 1);
  constexpr size_t kScaleBytes = kPhysicalN * sizeof(c10::Half);
  const KernelId kKernelId = fp16
      ? KernelId::PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024
      : KernelId::PI05_DENOISE_KV1_PAIR_OWNER_W8A16_M50N32X2K1024;

  TORCH_CHECK(
      RpuKernelGraph::has_active(),
      "Pi0.5 KV1 pair-owner launcher requires an active retained graph");
  auto check_weight = [&](const at::Tensor& tensor, const char* role) {
    TORCH_CHECK(
        tensor.defined() && tensor.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
            tensor.device().type() == c10::DeviceType::PrivateUse1 &&
            tensor.dim() == 2 && tensor.size(0) == kPhysicalN &&
            tensor.size(1) == kK && tensor.is_contiguous() &&
            tensor.storage_offset() == 0 && tensor.nbytes() == kWeightBytes,
        "Pi0.5 KV1 pair-owner ", role,
        " weight must be an exact contiguous matching-dtype device [2048,1024] owner");
  };
  auto check_scale = [&](const at::Tensor& tensor, const char* role) {
    if (fp16) {
      TORCH_CHECK(!tensor.defined(), "Pi0.5 FP16 KV1 pair-owner rejects quantization scales");
      return;
    }
    TORCH_CHECK(
        tensor.defined() && tensor.scalar_type() == at::kHalf &&
            tensor.device().type() == c10::DeviceType::PrivateUse1 &&
            tensor.dim() == 1 && tensor.size(0) == kPhysicalN &&
            tensor.is_contiguous() && tensor.storage_offset() == 0 &&
            tensor.nbytes() == kScaleBytes,
        "Pi0.5 KV1 pair-owner ", role,
        " scale must be an exact contiguous device fp16 [2048] owner");
  };
  check_weight(k_full_weight, "K");
  check_scale(k_full_scale, "K");
  check_weight(v_full_weight, "V");
  check_scale(v_full_scale, "V");
  TORCH_CHECK(
      k_full_weight.device() == v_full_weight.device() &&
          (fp16 || (k_full_weight.device() == k_full_scale.device() &&
                    k_full_weight.device() == v_full_scale.device())),
      "Pi0.5 KV1 pair-owner DDR owners must share one device");

  TORCH_CHECK(
      SPM_ALLOC.is_initialized(),
      "Pi0.5 KV1 pair-owner requires the initialized SPM allocator");
  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  auto check_spm_interval = [spm_base](uint32_t address, size_t bytes,
                                      const char* role) {
    const uint64_t absolute = address;
    TORCH_CHECK(
        absolute >= spm_base && (absolute & 255u) == 0 &&
            bytes <= SpmAllocator::SPM_USABLE &&
            absolute - spm_base <= SpmAllocator::SPM_USABLE - bytes,
        "Pi0.5 KV1 pair-owner ", role,
        " must be a 256-byte-aligned absolute core-0 SPM interval");
  };
  check_spm_interval(input_spm_addr, kInputBytes, "input");
  check_spm_interval(k_output_spm_addr, kOutputBytes, "K output");
  check_spm_interval(v_output_spm_addr, kOutputBytes, "V output");
  auto disjoint = [](uint64_t lhs, size_t lhs_bytes,
                     uint64_t rhs, size_t rhs_bytes) {
    return lhs + lhs_bytes <= rhs || rhs + rhs_bytes <= lhs;
  };
  TORCH_CHECK(
      disjoint(input_spm_addr, kInputBytes,
               k_output_spm_addr, kOutputBytes) &&
          disjoint(input_spm_addr, kInputBytes,
                   v_output_spm_addr, kOutputBytes) &&
          disjoint(k_output_spm_addr, kOutputBytes,
                   v_output_spm_addr, kOutputBytes),
      "Pi0.5 KV1 pair-owner input/K-output/V-output SPM intervals overlap");

  std::vector<std::pair<uint64_t, size_t>> owner_intervals{
      {RpuGetDevAddr(k_full_weight.data_ptr()), kWeightBytes},
      {RpuGetDevAddr(v_full_weight.data_ptr()), kWeightBytes}};
  if (!fp16) {
    owner_intervals.emplace_back(RpuGetDevAddr(k_full_scale.data_ptr()), kScaleBytes);
    owner_intervals.emplace_back(RpuGetDevAddr(v_full_scale.data_ptr()), kScaleBytes);
  }
  for (size_t index = 0; index < owner_intervals.size(); ++index) {
    const uint64_t address = owner_intervals[index].first;
    TORCH_CHECK(
        address != 0 && (address & 255u) == 0 &&
            (address >> 8) <= UINT32_MAX,
        "Pi0.5 KV1 pair-owner DDR owner ", index,
        " violates the aligned shift8 ABI");
    for (size_t other = index + 1; other < owner_intervals.size(); ++other) {
      TORCH_CHECK(
          disjoint(address, owner_intervals[index].second,
                   owner_intervals[other].first,
                   owner_intervals[other].second),
          "Pi0.5 KV1 pair-owner DDR owner intervals overlap at indices ",
          index, " and ", other);
    }
  }

  std::vector<GraphDdrRegisterOperandSpec> operands{
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{
                  fp16 ? GraphDdrRegisterRole::ModelWeight : GraphDdrRegisterRole::Int8ModelWeight,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
              k_full_weight},
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{
                  fp16 ? GraphDdrRegisterRole::ModelWeight : GraphDdrRegisterRole::Int8ModelWeight,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25},
              v_full_weight}};
  if (!fp16) {
    operands.push_back(
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{
                  GraphDdrRegisterRole::PerChannelScale,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
              k_full_scale});
    operands.push_back(
          GraphDdrRegisterOperandSpec{
              GraphDdrRegisterAbi{
                  GraphDdrRegisterRole::PerChannelScale,
                  GraphDdrRegisterAccess::Read,
                  GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27},
              v_full_scale});
  }
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(kKernelId, operands);
  rpu_ddr_flush(k_full_weight.data_ptr());
  rpu_ddr_flush(v_full_weight.data_ptr());
  if (!fp16) {
    rpu_ddr_flush(k_full_scale.data_ptr());
    rpu_ddr_flush(v_full_scale.data_ptr());
  }

  Kernel_t* kernel = GET_KERNEL(kKernelId);
  TORCH_CHECK(
      kernel != nullptr,
      "Pi0.5 KV1 pair-owner kernel for the installed weight dtype is absent from the expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, k_output_spm_addr, 0,
          kRows, kLocalN, kK, 0, kCores,
          kK * sizeof(c10::Half), kLocalN * sizeof(c10::Half),
          0, 1, static_cast<uint16_t>(fp16 ? 0 : 1)});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 30, v_output_spm_addr);
  writer.write(*kernel);

  auto* queue = GET_QUEUE(kCores);
  TORCH_CHECK(queue != nullptr,
              "Pi0.5 KV1 pair-owner 8-core queue is absent");
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {2, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_prefill_kv1_pair_owner_w8a16_m400n32x2k2048_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& k_full_weight,
    const at::Tensor& k_full_scale,
    uint32_t k_output_spm_addr,
    const at::Tensor& v_full_weight,
    const at::Tensor& v_full_scale,
    uint32_t v_output_spm_addr, int64_t rows) {
  TORCH_CHECK(rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 336 || rows == 400 || rows == 416 || rows == 432 || rows == 448 || rows == 464,
              "Pi Prefill KV1 requires C272/C288/C304/C320/C336/C400/C416/C432/C448/C464");
  const bool fp16 = k_full_weight.defined() && k_full_weight.scalar_type() == at::kHalf;
  const uint32_t kRows = rows;
  constexpr uint32_t kLocalN = 32;
  constexpr uint32_t kK = 2048;
  constexpr uint32_t kPhysicalN = 2048;
  constexpr uint16_t kCores = 8;
  const size_t kInputBytes = kRows * kK * sizeof(c10::Half);
  const size_t kOutputBytes = kRows * kLocalN * sizeof(c10::Half);
  const size_t kWeightBytes = size_t{kPhysicalN} * kK * (fp16 ? 2 : 1);
  constexpr size_t kScaleBytes = kPhysicalN * sizeof(c10::Half);
  const KernelId kKernelId = rows == 432
      ? (fp16 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M432N32X2K2048
              : KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M432N32X2K2048) :
        rows == 464
      ? (fp16 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M464N32X2K2048
              : KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M464N32X2K2048) :
        rows == 336
      ? (fp16 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M336N32X2K2048
              : KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M336N32X2K2048) :
        rows == 448
      ? (fp16 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M448N32X2K2048
              : KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M448N32X2K2048) :
        rows == 416
      ? (fp16 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M416N32X2K2048
              : KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M416N32X2K2048)
      : rows == 288
      ? (fp16 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M288N32X2K2048
              : KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M288N32X2K2048)
      : rows == 304
      ? (fp16 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M304N32X2K2048
              : KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M304N32X2K2048) :
        rows == 320
      ? (fp16 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M320N32X2K2048
              : KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M320N32X2K2048)
      : fp16 ?
      (rows == 400 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M400N32X2K2048 :
                     KernelId::PI05_PREFILL_KV1_PAIR_OWNER_FP16_M272N32X2K2048) :
      (rows == 400 ? KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M400N32X2K2048 :
                     KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M272N32X2K2048);

  TORCH_CHECK(
      RpuKernelGraph::has_active(),
      "Pi0.5 prefill KV1 pair-owner launcher requires an active retained "
      "graph");
  auto check_weight = [&](const at::Tensor& tensor, const char* role) {
    TORCH_CHECK(
        tensor.defined() && tensor.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
            tensor.device().type() == c10::DeviceType::PrivateUse1 &&
            tensor.dim() == 2 && tensor.size(0) == kPhysicalN &&
            tensor.size(1) == kK && tensor.is_contiguous() &&
            tensor.storage_offset() == 0 && tensor.nbytes() == kWeightBytes,
        "Pi0.5 prefill KV1 pair-owner ", role,
        " weight must be an exact contiguous device int8 [2048,2048] owner");
  };
  auto check_scale = [&](const at::Tensor& tensor, const char* role) {
    TORCH_CHECK(
        tensor.defined() && tensor.scalar_type() == at::kHalf &&
            tensor.device().type() == c10::DeviceType::PrivateUse1 &&
            tensor.dim() == 1 && tensor.size(0) == kPhysicalN &&
            tensor.is_contiguous() && tensor.storage_offset() == 0 &&
            tensor.nbytes() == kScaleBytes,
        "Pi0.5 prefill KV1 pair-owner ", role,
        " scale must be an exact contiguous device fp16 [2048] owner");
  };
  check_weight(k_full_weight, "K");
  if (!fp16) check_scale(k_full_scale, "K");
  else TORCH_CHECK(!k_full_scale.defined(), "FP16 K cannot carry W8 scales");
  check_weight(v_full_weight, "V");
  if (!fp16) check_scale(v_full_scale, "V");
  else TORCH_CHECK(!v_full_scale.defined(), "FP16 V cannot carry W8 scales");
  TORCH_CHECK(
      k_full_weight.device() == v_full_weight.device() &&
          (fp16 || (k_full_weight.device() == k_full_scale.device() &&
                    k_full_weight.device() == v_full_scale.device())),
      "Pi0.5 prefill KV1 pair-owner DDR owners must share one device");

  TORCH_CHECK(
      SPM_ALLOC.is_initialized(),
      "Pi0.5 prefill KV1 pair-owner requires the initialized SPM allocator");
  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  auto check_spm_interval = [spm_base](uint32_t address, size_t bytes,
                                      const char* role) {
    const uint64_t absolute = address;
    TORCH_CHECK(
        absolute >= spm_base && (absolute & 255u) == 0 &&
            bytes <= SpmAllocator::SPM_USABLE &&
            absolute - spm_base <= SpmAllocator::SPM_USABLE - bytes,
        "Pi0.5 prefill KV1 pair-owner ", role,
        " must be a 256-byte-aligned absolute core-0 SPM interval");
  };
  check_spm_interval(input_spm_addr, kInputBytes, "input");
  check_spm_interval(k_output_spm_addr, kOutputBytes, "K output");
  check_spm_interval(v_output_spm_addr, kOutputBytes, "V output");
  auto disjoint = [](uint64_t lhs, size_t lhs_bytes,
                     uint64_t rhs, size_t rhs_bytes) {
    return lhs + lhs_bytes <= rhs || rhs + rhs_bytes <= lhs;
  };
  TORCH_CHECK(
      disjoint(input_spm_addr, kInputBytes,
               k_output_spm_addr, kOutputBytes) &&
          disjoint(input_spm_addr, kInputBytes,
                   v_output_spm_addr, kOutputBytes) &&
          disjoint(k_output_spm_addr, kOutputBytes,
                   v_output_spm_addr, kOutputBytes),
      "Pi0.5 prefill KV1 pair-owner input/K-output/V-output SPM intervals "
      "overlap");

  std::vector<std::pair<uint64_t, size_t>> owner_intervals;
  for (const auto* tensor : {&k_full_weight,&k_full_scale,&v_full_weight,&v_full_scale})
    if (tensor->defined()) owner_intervals.emplace_back(RpuGetDevAddr(tensor->data_ptr()),tensor->nbytes());
  for (size_t index = 0; index < owner_intervals.size(); ++index) {
    const uint64_t address = owner_intervals[index].first;
    TORCH_CHECK(
        address != 0 && (address & 255u) == 0 &&
            (address >> 8) <= UINT32_MAX,
        "Pi0.5 prefill KV1 pair-owner DDR owner ", index,
        " violates the aligned shift8 ABI");
    for (size_t other = index + 1; other < owner_intervals.size(); ++other) {
      TORCH_CHECK(
          disjoint(address, owner_intervals[index].second,
                   owner_intervals[other].first,
                   owner_intervals[other].second),
          "Pi0.5 prefill KV1 pair-owner DDR owner intervals overlap at "
          "indices ", index, " and ", other);
    }
  }

  const auto weight_role=fp16 ? GraphDdrRegisterRole::ModelWeight : GraphDdrRegisterRole::Int8ModelWeight;
  std::vector<GraphDdrRegisterOperandSpec> operands{
      {{weight_role,GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},k_full_weight},
      {{weight_role,GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},v_full_weight}};
  if (!fp16) {
    operands.push_back({{GraphDdrRegisterRole::PerChannelScale,GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},k_full_scale});
    operands.push_back({{GraphDdrRegisterRole::PerChannelScale,GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,26,27},v_full_scale});
  }
  auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(kKernelId,operands);
  for(const auto* tensor:{&k_full_weight,&k_full_scale,&v_full_weight,&v_full_scale}) {
    if (!tensor->defined()) continue;
    RpuKernelGraph::active().keep_alive(*tensor);
    rpu_ddr_flush(tensor->data_ptr());
  }

  Kernel_t* kernel = GET_KERNEL(kKernelId);
  TORCH_CHECK(
      kernel != nullptr,
      "Pi0.5 prefill KV1 pair-owner optional expansion symbol is absent");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, k_output_spm_addr, 0,
          kRows, kLocalN, kK, 0, kCores,
          kK * sizeof(c10::Half), kLocalN * sizeof(c10::Half),
          0, 1, uint16_t(fp16 ? 0 : 1)});
  const auto set_reg = [kernel](uint32_t index, uint16_t value) {
    TORCH_CHECK(
        kernel->set_regs(index, value) == kKernelRegOk,
        "Pi0.5 prefill KV1 pair-owner rejected parameter ", index);
  };
  const auto set_pair = [&set_reg](uint32_t index, uint32_t value) {
    set_reg(index, static_cast<uint16_t>(value & 0xffffu));
    set_reg(index + 1, static_cast<uint16_t>(value >> 16));
  };
  // p24..27 are typed DDR slots written below; all remaining extension
  // parameters are explicitly initialized so all cores receive one exact
  // 68-halfword broadcast image.
  for (uint32_t index = 24; index < 68; ++index) set_reg(index, 0);
  set_pair(30, v_output_spm_addr);
  set_reg(64, 2);
  set_reg(65, 1);
  set_reg(66, 1);
  set_reg(67, 0);
  writer.write(*kernel);

  auto* queue = GET_QUEUE(kCores);
  TORCH_CHECK(queue != nullptr,
              "Pi0.5 prefill KV1 pair-owner 8-core queue is absent");
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {2, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

Tensor cpu_fallback_linear(const Tensor &input, const Tensor &weight,
                           const c10::optional<Tensor> &bias_opt) {
  // 零拷贝 fallback 到 CPU
  auto cpu_input = rpu_to_cpu_zerocopy(input);
  auto cpu_weight = rpu_to_cpu_zerocopy(weight);
  Tensor cpu_bias;
  if (bias_opt.has_value()) {
    cpu_bias = rpu_to_cpu_zerocopy(*bias_opt);
  }

  // 预分配 device 输出
  std::vector<int64_t> out_sizes = input.sizes().vec();
  out_sizes.back() = weight.size(0);
  auto result = at::empty(out_sizes, input.options());
  auto cpu_result = rpu_to_cpu_zerocopy(result);

  // 使用 addmm_out 或 mm_out 直接写入共享内存
  // linear: output = input @ weight.T + bias
  auto input_2d = cpu_input.dim() == 2 ? cpu_input : cpu_input.view({-1, input.size(-1)});
  auto result_2d = cpu_result.dim() == 2 ? cpu_result : cpu_result.view({-1, weight.size(0)});

  if (bias_opt.has_value()) {
    // result = input @ weight.T + bias
    at::mm_out(result_2d, input_2d, cpu_weight.t());
    result_2d.add_(cpu_bias);
  } else {
    at::mm_out(result_2d, input_2d, cpu_weight.t());
  }

  return result;
}


// ---------------------------------------------------------------------------
// Linear via the generated FP16 ACC32 SPM kernels, operands moved by DMA.  The
// The DDR-facing launcher name is kept for its call sites, but computation is
// always staged through the generated SPM auto-tile family.
//
// Both directions are DMA, and both modes (graph-recorded and eager) run the
// same code: the graph DMA wrappers fall back to their `_immediate` twins when
// no graph scope is active.
//
//   col partition (N%128==0):  broadcast X -> GEMM -> all_gather -> ONE core-0
//     SPM->DDR copy. all_gather turns the per-core column blocks into the full
//     [m, N] row layout in SPM, so the write-back is a single contiguous DMA
//     with no host work — which is what makes this replay-safe.
//
//   row partition (N%128!=0):  EAGER ONLY. Each core needs a strided column
//     slice of X and a DMA moves contiguous blocks only, so the repack is a host
//     memcpy; the cross-core partial sums are likewise reduced on the host. Host
//     work does not replay. No graph-mode caller exists (the only one is the
//     SigLIP projector, which is col) — making it replay-safe would mean one DMA
//     per row per core on the way in, plus all_reduce with a zeroed residual on
//     the way out.
//
// Staging is DMA, never memcpy-into-SPM: there is no coherency guarantee between
// the host and GlobalSPM (SPM_ALLOC.cpu_ptr() has no other user in the tree and
// Buffer_t / GlobalSPM_t export no flush/invalidate).
//
// M is chunked so one chunk fits whatever SPM the live fused subsystems left
// free — this can run right after a fused BUILD has taken most of it. There is
// no CPU fallback when even one row does not fit: the weight reaching this
// function is already swizzled, so a CPU GEMM would be silently wrong (P1).
// ---------------------------------------------------------------------------
// DDR-only version: Input, Output, Bias, Weight all in DDR
// partition: 0 = row (按K切分), 1 = col (按N切分)
// NOTE: weight must be pre-transformed using rpu_transform_linear_weight() or
//       Python-side convert_model_for_rpu() before calling this function
namespace {
KernelId autotile_linear_acc32_kernel_id(bool is_fp16, int n_tile);

void validate_parallel_linear_geometry(
    int64_t M, int64_t N, int64_t K, int64_t partition, int64_t num_cores) {
  TORCH_CHECK(num_cores >= 1 && num_cores <= NUM_CORES,
              "parallel Linear num_cores must be in [1,8], got ", num_cores);
  TORCH_CHECK(partition == 0 || partition == 1,
              "parallel Linear partition must be row(0) or col(1), got ", partition);
  TORCH_CHECK(M > 0 && N > 0 && K > 0 && N % 16 == 0 && K % 16 == 0,
              "parallel Linear requires positive M and 16-aligned N/K; got ",
              M, ",", N, ",", K);
  TORCH_CHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX,
              "parallel Linear dimensions exceed the signed tile-selector ABI");
  TORCH_CHECK((partition == 0 ? K : N) % (16 * num_cores) == 0,
              "parallel Linear partition dimension must be divisible by 16*num_cores; got ",
              "N=", N, " K=", K, " partition=", partition, " cores=", num_cores);
}

void validate_parallel_linear_grid(size_t rows, size_t cols,
                                   size_t tile_m, size_t tile_n) {
  TORCH_CHECK(tile_m > 0 && tile_n > 0 &&
                  (rows - 1) / tile_m < UINT16_MAX &&
                  (cols - 1) / tile_n < UINT16_MAX,
              "parallel Linear grid exceeds the uint16 launch ABI");
}
}

// Public W4 DDR projection uses the same pgrp ABI as fused decoder projections.
// Validate before allocating/staging so a malformed packed shape cannot become
// a half-width activation DMA or launch with an undersized scale payload.
static void check_linear_w4a16_operands(
    const Tensor& input, const Tensor& weight, const Tensor& scale,
    const c10::optional<Tensor>& bias_opt) {
  TORCH_CHECK(!graph_dma::active(),
              "linear_w4a16: fresh DDR operands require eager execution");
  TORCH_CHECK(input.dim() >= 1 && weight.dim() == 2,
              "linear_w4a16: input must have at least one dimension and weight be 2-D");
  TORCH_CHECK(input.scalar_type() == at::kHalf && weight.scalar_type() == at::kByte,
              "linear_w4a16: input must be FP16 and packed weight UINT8");
  TORCH_CHECK(input.device().type() == at::kPrivateUse1 &&
                  weight.device() == input.device(),
              "linear_w4a16: input and weight must share a device");
  const int64_t k = input.size(-1), n = weight.size(0);
  TORCH_CHECK(k > 0 && k % 64 == 0 && weight.size(1) == k / 2,
              "linear_w4a16: packed weight must be [N,K/2] with logical K divisible by 64");
  TORCH_CHECK(weight.is_contiguous(),
              "linear_w4a16: pgrp packed weight must be contiguous");
  validate_parallel_linear_geometry(input.numel() / k, n, k, 1, 8);
  TORCH_CHECK(scale.defined() && scale.device() == input.device() &&
                  scale.scalar_type() == at::kHalf && scale.dim() == 2 &&
                  scale.size(0) == 32 && scale.is_contiguous(),
              "linear_w4a16: scale must be contiguous device FP16 pgrp G32 payload");
  const int64_t scale_elements = ((k / 32 + 3) / 4) *
      ((n / 8 + 63) / 64) * 8 * 4 * 64;
  TORCH_CHECK(scale.numel() == scale_elements,
              "linear_w4a16: controller-striped scale payload has ", scale.numel(),
              " elements, expected ", scale_elements);
  if (bias_opt.has_value()) {
    const auto& bias = *bias_opt;
    TORCH_CHECK(bias.device() == input.device() &&
                    bias.scalar_type() == at::kHalf && bias.dim() == 1 &&
                    bias.numel() == n && bias.is_contiguous(),
                "linear_w4a16: bias must be contiguous device FP16 [N]");
  }
}

void rpu_launch_linear_ddr_kernel(const at::Tensor &input, const at::Tensor &weight,
                                  at::Tensor &output, const at::Tensor &bias,
                                  bool has_bias, int partition,
                                  const uint64_t *live_dst_base,
                                  int64_t dst_offset_bytes,
                                  const uint64_t *live_bias_base,
                                  int num_cores,
                                  const at::Tensor& weight_scale,
                                  c10::optional<bool> linear_acc32) {
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && output.dim() == 2,
              "rpu_linear DDR launcher requires rank-2 input/weight/output");
  const bool w4a16 = weight.scalar_type() == at::kByte;
  if (w4a16) {
    TORCH_CHECK(partition == 1 && num_cores == 8,
                "rpu_linear W4A16 DDR supports eager TP8 col partition only");
    check_linear_w4a16_operands(input, weight, weight_scale,
        has_bias ? c10::optional<Tensor>(bias) : c10::optional<Tensor>{});
  }
  const int64_t logical_k = w4a16 ? input.size(1) : weight.size(1);
  validate_parallel_linear_geometry(
      input.size(0), weight.size(0), logical_k, partition, num_cores);
  TORCH_CHECK(input.size(1) == logical_k &&
                  output.size(0) == input.size(0) &&
                  output.size(1) == weight.size(0),
              "rpu_linear DDR input/weight/output geometry mismatch");
  TORCH_CHECK(input.scalar_type() == at::kHalf,
              "rpu_linear: input must be Half, got ", input.scalar_type());
  const bool w8a16 = weight.scalar_type() == at::kChar;
  TORCH_CHECK(weight.scalar_type() == at::kHalf || w8a16 || w4a16,
              "rpu_linear: weight must be Half, scaled Int8 or packed W4, got ", weight.scalar_type());
  if (w8a16) {
    TORCH_CHECK(partition == 1 && !graph_dma::active(),
                "rpu_linear W8A16 DDR supports eager col partition only");
    TORCH_CHECK(weight.size(1) % 32 == 0,
                "rpu_linear W8A16 col-swizzle requires K divisible by 32");
    TORCH_CHECK(weight_scale.defined() &&
                    weight_scale.device() == input.device() &&
                    weight_scale.scalar_type() == at::kHalf &&
                    weight_scale.dim() == 1 &&
                    weight_scale.numel() == weight.size(0) &&
                    weight_scale.is_contiguous(),
                "rpu_linear W8A16 requires contiguous device FP16 scale [N]");
  } else if (!w4a16) {
    TORCH_CHECK(!weight_scale.defined(), "FP16 linear does not take a W8 scale");
  }
  TORCH_CHECK(output.scalar_type() == at::kHalf,
              "rpu_linear: output must be Half, got ", output.scalar_type());
  TORCH_CHECK(output.is_contiguous(), "rpu_linear: output must be contiguous");

  const bool use_acc32 = linear_acc32.value_or(!w8a16 && !w4a16);
  const bool in_graph = graph_dma::active();
  TORCH_CHECK(partition != 0 || !in_graph,
              "rpu_linear: row partition is eager-only — its per-core input "
              "slice is strided (DMA moves contiguous blocks only) and its "
              "cross-core reduce runs on the host, neither of which replays. "
              "Use a valid col-partition shape for the selected cores or the fused "
              "rpu_launch_linear_spm_to_spm_acc16_kernel.");
  // The row path's write-back is a host reduce, not a DMA — there is no DMA
  // destination to rebind, so a live base there would silently do nothing.
  TORCH_CHECK(live_dst_base == nullptr || partition != 0,
              "rpu_linear: live_dst_base is only meaningful for the "
              "col-partition write-back DMAs (row partition reduces on host)");
  // Graph-mode operand-stability gate.
  //
  // BOTH DDR operands of the col path are baked into kd_buf at BUILD: the input
  // upload below is the FIXED `rpu_launch_ddr_broadcast_spm_dma(xp + row0*k, …)`
  // and has no mutable twin, and the write-back only gets one when the caller
  // hands over live_dst_base. REPLAY refreshes a fixed node's addresses only on
  // the full-rebuild path; the sync-only fast path (which a single-segment graph
  // always takes) leaves the BUILD-time address frozen — see the ⚠️ Fixed DMA
  // trap comment on launch_segment_sync_only, graph_runtime_execute.cpp. A
  // caller whose tensors are fresh per forward therefore reads AND writes the
  // PREVIOUS forward's DDR, silently, in both directions.
  //
  // There is deliberately no `live_src_base` twin: no caller needs one. The only
  // in-tree graph-mode caller is the SigLIP projector, whose input is the
  // per-shape `temp_ddr_slots_[{seq_len,h}]` the model owns for exactly this
  // reason and whose output already rides live_dst_base. The two eager callers
  // (`rpu_linear` for aten::linear, and `linear_with_partition`) allocate their
  // output with a fresh `at::empty` per call, so a src parameter alone could not
  // make either correct — the dst would still be baked, and C-1
  // (docs/pitfalls.md) additionally requires keep_alive on any operand a graph
  // dereferences after the caller returned. Refuse the case loudly instead of
  // shipping a parameter nothing routes.
  //
  // Passing live_dst_base is therefore also the caller's assertion that `input`
  // is replay-stable. If you add a graph-mode caller whose input drifts, add the
  // live_src_base twin then — with a caller to route it to.
  TORCH_CHECK(!in_graph || live_dst_base != nullptr,
              "rpu_linear: recorded inside a graph capture by a caller that did "
              "not hand over a live write-back base (live_dst_base). Both DDR "
              "operands are baked at BUILD on this path, so REPLAY would "
              "silently read the input from — and write the result to — the "
              "PREVIOUS forward's buffers. Either give this launcher "
              "replay-stable operands and pass live_dst_base (see the SigLIP "
              "projector, src/fused/rpu_siglip_model.cpp), or call it outside "
              "the capture scope.");

  const size_t a_dwidth = input.element_size();  // fp16 = 2 bytes
  const size_t m = input.size(0);
  const size_t k = input.size(1);
  const size_t n = weight.size(0);
  const size_t local_n = (partition == 0) ? n : n / num_cores;
  const size_t local_k = (partition == 0) ? k / num_cores : k;

  c10::Half *xp = input.data_ptr<c10::Half>();
  c10::Half *yp = output.data_ptr<c10::Half>();
  void *wp = weight.data_ptr();
  c10::Half *bp = has_bias ? bias.data_ptr<c10::Half>() : nullptr;

  // Row partition: bias must NOT go through the kernel. Every core would add the
  // full bias and the cross-core reduce below then sums it num_cores times
  // (measured: output + 7*bias). This DDR path adds it once after reduction.
  // Direct SPM callers instead own the per-core bias distribution.
  const bool kernel_applies_bias = has_bias && partition != 0;

  if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
  // mark/release, NOT reset_temporary(): a fused subsystem's temporaries may be
  // live and reset_temporary() would wipe them too.
  SpmAllocator::ScopedTemporary spm_scope;

  // Per core, per row: x + the GEMM's own y, plus (col) the all-gathered
  // full-width row every core ends up holding.
  size_t row_bytes =
      (local_k + local_n + (partition == 0 ? 0 : n)) * a_dwidth;
  const size_t bias_bytes =
      kernel_applies_bias ? Align(local_n * a_dwidth, SpmAllocator::ALIGN) : 0;
  const size_t slack = bias_bytes + 3 * SpmAllocator::ALIGN;  // per-alloc round-up
  size_t free_bytes = SPM_ALLOC.free_space();
  if (w8a16 || w4a16) {
    constexpr size_t budget = SpmAllocator::SPM_PLANNING_BUDGET &
        ~(SpmAllocator::ALIGN - 1);
    const size_t used = SPM_ALLOC.total_used();
    TORCH_CHECK(used <= budget,
                "rpu_linear quantized: existing SPM exceeds the shared planning budget");
    free_bytes = std::min(free_bytes, budget - used);
  }

  // The col path's all-gather landing zone (the `+ n` term above) dominates a
  // vocab-width GEMV: at N=151936 it is 297 KB on its own, more than the SPM
  // left free once a fused decoder graph holds its persistent buffers. That is
  // a REGRESSION vs the DDR kernel this replaced (it needed no SPM at all) and
  // it bites the eager prefill lm_head — qwen3-1.7b M=1 K=2048 died here while
  // 0.6b/4b happened to still fit.
  //
  // A single eager row never needs the landing zone: let each core DMA its OWN
  // column block straight to DDR. Also use this path when the landing zone
  // does not fit. For a SINGLE row the per-core
  // blocks are exactly that row's contiguous column slices, so the core-major
  // scatter lands them precisely where the gathered copy would have — hence
  // m_chunk is pinned to 1 on this path. Keep the existing captured-Graph and
  // multi-row selection unless the landing zone cannot fit; their general
  // core-major blocks need the gather to restore row-major output.
  bool col_scatter = false;
  if (partition != 0 &&
      ((!in_graph && m == 1) || free_bytes <= slack + row_bytes)) {
    col_scatter = true;
    row_bytes = (local_k + local_n) * a_dwidth;
  }
  TORCH_CHECK(free_bytes > slack + row_bytes,
              "rpu_linear: free SPM (", free_bytes, " B) cannot hold one row of "
              "M=", m, " K=", k, " N=", n, " (needs ", slack + row_bytes, " B",
              col_scatter ? ", already without the all-gather landing zone)" : ")");
  size_t m_chunk = col_scatter ? 1 : (free_bytes - slack) / row_bytes;
  if (m_chunk > m) m_chunk = m;
  if (m_chunk > 65535) m_chunk = 65535;  // reg8 (local_m) / all_gather reg1 are uint16

  const uint32_t x_addr =
      SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(m_chunk * local_k * a_dwidth));
  const uint32_t y_addr =
      SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(m_chunk * local_n * a_dwidth));
  const uint32_t g_addr =
      (partition == 0 || col_scatter)
          ? 0
          : SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(m_chunk * n * a_dwidth));
  uint32_t b_addr = 0;
  if (kernel_applies_bias) {
    b_addr = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(local_n * a_dwidth));
    // col partition: core c owns columns [c*local_n, (c+1)*local_n) of the bias
    //
    // The weight reaches the kernel as a REGISTER (w_addr), which REPLAY re-syncs
    // via sync_mutable_params — so a set_weights that swaps the weight tensor takes
    // effect. The bias reaches it as a DMA, which sync-only does NOT rebind, so the
    // same swap silently kept feeding the OLD bias. live_bias_base closes that
    // asymmetry for callers that can hand over a live base.
    if (live_bias_base) {
      // Unlike the fixed wrapper, mutable DMA does not publish its source.
      // Publish before it can execute synchronously in the eager path.
      rpu_ddr_flush(bp);
      rpu_launch_ddr_scatter_spm_dma_mutable(live_bias_base,
                                             /*src_offset_bytes=*/0, local_n,
                                             local_n * a_dwidth, b_addr,
                                             num_cores);
    } else {
      rpu_launch_ddr_scatter_spm_dma(bp, local_n, local_n * a_dwidth,
                                     b_addr, num_cores);
    }
  }

  // Row partition only: DDR staging for the strided input repack and for the
  // per-core partial sums on the way back. Both are host-touched, so both need
  // the BOUNDARY force flush — `rpu_ddr_flush` is gated on
  // g_rpu_ddr_flush_enabled, which is OFF by default, i.e. a no-op. Without the
  // force flush the DMA reads stale/zero staging (measured: every row-partition
  // case wrong, M=1 all zeros) and the host reads stale outputs coming back.
  at::Tensor x_stage, y_stage;
  c10::Half *xs = nullptr, *ys = nullptr;
  if (partition == 0) {
    x_stage = at::empty({(int64_t)(num_cores * m_chunk * local_k)}, input.options());
    y_stage = at::empty({(int64_t)(num_cores * m_chunk * local_n)}, input.options());
    xs = x_stage.data_ptr<c10::Half>();
    ys = y_stage.data_ptr<c10::Half>();
  }

  // Col input and fixed bias are published by their DDR->SPM wrappers in
  // eager and Graph paths. Row repacking and its host bias addition still
  // need publication here; mutable col bias was published before its DMA.
  if (partition == 0) rpu_ddr_flush(xp);
  rpu_ddr_flush(wp);
  if (has_bias && partition == 0) rpu_ddr_flush(bp);

  auto* wq = GET_QUEUE(num_cores);
  std::vector<uint8_t> core_list;
  for (int core = 0; core < num_cores; ++core) core_list.push_back(core);

  for (size_t row0 = 0; row0 < m; row0 += m_chunk) {
    const size_t cur_m = (m - row0 < m_chunk) ? (m - row0) : m_chunk;

    if (partition == 0) {
      // core c gets the column slice [c*local_k, (c+1)*local_k) of every row
      for (int c = 0; c < num_cores; ++c) {
        for (size_t r = 0; r < cur_m; ++r) {
          memcpy(xs + (c * cur_m + r) * local_k,
                 xp + (row0 + r) * k + c * local_k,
                 local_k * a_dwidth);
        }
      }
      rpu_ddr_flush_force(xs);
      rpu_launch_ddr_scatter_spm_dma(xs, cur_m * local_k,
                                     cur_m * local_k * a_dwidth,
                                     x_addr, num_cores);
    } else {
      // every core needs the whole rows, already contiguous in the input
      rpu_launch_ddr_broadcast_spm_dma(xp + row0 * k, cur_m * k,
                                       x_addr, num_cores);
    }

    if (w8a16 || w4a16 || !use_acc32) {
      rpu_launch_linear_spm_to_spm_acc16_kernel(
          x_addr, weight, y_addr, cur_m, n, k, partition, num_cores,
          b_addr, weight_scale, 0, 0, use_acc32);
    } else {
      const auto tile = rpu_pl_tiling::select_tile_acc32(
          static_cast<int>(cur_m), static_cast<int>(local_n),
          static_cast<int>(local_k), /*is_fp16=*/true);
      const size_t n_per_wrp = static_cast<size_t>(tile.n_tile);
      const size_t m_per_wrp = static_cast<size_t>(tile.m_tile);
      validate_parallel_linear_grid(cur_m, local_n, m_per_wrp, n_per_wrp);
      const KernelId kernel_id =
          autotile_linear_acc32_kernel_id(/*is_fp16=*/true, tile.n_tile);

      auto register_writer =
          RpuKernelGraph::active().stage_kernel_ddr_registers(
              kernel_id,
              std::vector<GraphDdrRegisterOperandSpec>{
                  GraphDdrRegisterOperandSpec{
                      GraphDdrRegisterAbi{
                          GraphDdrRegisterRole::ModelWeight,
                          GraphDdrRegisterAccess::Read,
                          GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                          2, 3},
                      weight}});

      // GET_KERNEL stays INSIDE the loop: it sets pending_kernel_id_ for every
      // graph node, and the recorder needs a distinct Kernel_t instance per chunk
      // so REPLAY does not reuse the tail chunk's registers for every node.
      Kernel_t* kernel = GET_KERNEL(kernel_id);
      TORCH_CHECK(kernel != nullptr,
                  "Failed to get FP16 ACC32 parallel_linear tile m", m_per_wrp,
                  "n", n_per_wrp);
      kernel->reset_regs();
      rpu_pl_tiling::set_parallel_linear_regs(
          *kernel,
          rpu_pl_tiling::ParallelLinearRegisterArgs{
              x_addr, 0, y_addr, b_addr,
              static_cast<uint32_t>(cur_m), static_cast<uint32_t>(local_n),
              static_cast<uint32_t>(local_k),
              static_cast<uint16_t>(kernel_applies_bias),
              static_cast<uint16_t>(num_cores),
              static_cast<uint32_t>(local_k * a_dwidth),
              static_cast<uint32_t>(local_n * a_dwidth),
              0, static_cast<uint16_t>(partition), 2});
      register_writer.write(*kernel);

      wq->set_broadcast_mode(true);
      wq->enqueu_kernel(*kernel,
                        {(uint16_t)CeilDiv(local_n, n_per_wrp),
                         (uint16_t)CeilDiv(cur_m, m_per_wrp), (uint16_t)1},
                        core_list);
    }

    if (partition != 0) {
      // Write-back destination. With live_dst_base the DDR address is bound per
      // replay (mutable DMA) instead of baked at BUILD; the wrapper then does no
      // flush of its own, so mirror the fixed variant's rpu_ddr_flush here to
      // keep coherency behavior identical between the two bindings.
      const int64_t row_dst_off = dst_offset_bytes + (int64_t)(row0 * n * a_dwidth);
      if (live_dst_base) rpu_ddr_flush(yp + row0 * n);
      if (col_scatter) {
        // cur_m == 1: core c holds columns [c*local_n, (c+1)*local_n) of this
        // row, so the core-major scatter IS the row's layout. No gather needed.
        if (live_dst_base) {
          rpu_launch_spm_scatter_ddr_dma_mutable(y_addr, live_dst_base,
                                                 row_dst_off, local_n,
                                                 local_n * a_dwidth, num_cores);
        } else {
          rpu_launch_spm_scatter_ddr_dma(y_addr, yp + row0 * n, local_n,
                                         local_n * a_dwidth, num_cores);
        }
        continue;
      }
      // per-core outputs are different COLUMN blocks of one answer -> concat.
      // (all_reduce_sum is its row-partition counterpart; picking the wrong one
      // is silently wrong, not a crash.)
      rpu_launch_all_gather_spm_kernel(
          y_addr, g_addr, cur_m, local_n, a_dwidth, num_cores,
          rpu_resolve_all_gather_schedule(local_n, a_dwidth));
      if (live_dst_base) {
        rpu_launch_spm_copy_ddr_dma_mutable(g_addr, live_dst_base, row_dst_off,
                                            cur_m * n);
      } else {
        rpu_launch_spm_copy_ddr_dma(g_addr, yp + row0 * n, cur_m * n);
      }
      continue;
    }

    // Row partition (eager only): per-core outputs are full-width PARTIAL SUMS.
    rpu_launch_spm_scatter_ddr_dma(y_addr, ys, cur_m * local_n,
                                   cur_m * local_n * a_dwidth, num_cores);
    rpu_ddr_flush_force(ys);

    c10::Half* out = yp + row0 * n;
    const size_t cnt = cur_m * local_n;
    for (size_t j = 0; j < cnt; ++j) {
      float acc = 0.0f;
      for (int c = 0; c < num_cores; ++c)
        acc += static_cast<float>(ys[c * cnt + j]);
      out[j] = static_cast<c10::Half>(acc);
    }
    // Bias was withheld from the GEMM (see kernel_applies_bias) so it lands
    // exactly ONCE, here, after the cross-core reduce.
    if (has_bias) {
      for (size_t r = 0; r < cur_m; ++r) {
        c10::Half* orow = out + r * local_n;
        for (size_t col = 0; col < local_n; ++col)
          orow[col] = static_cast<c10::Half>(static_cast<float>(orow[col]) +
                                             static_cast<float>(bp[col]));
      }
    }
    rpu_ddr_flush_force(out);  // host wrote it; publish before the next chunk
  }
}


// Get partition strategy for given weight shape
// Returns: 0=row, 1=col, -1=fallback to CPU
int rpu_get_linear_partition(int64_t in_features, int64_t out_features, int64_t dwidth) {
  int64_t num_ele_32B = 32 / dwidth;

  // Row partition 要求: K % (num_ele_32B * NUM_CORES) == 0, N % 16 == 0
  // Col partition 要求: N % (16 * NUM_CORES) == 0, K % num_ele_32B == 0
  bool can_row = (in_features % (num_ele_32B * NUM_CORES) == 0) && (out_features % 16 == 0);
  bool can_col = (out_features % (16 * NUM_CORES) == 0) && (in_features % num_ele_32B == 0);

  if (can_row && can_col) {
    // Tie-break: col. This is what eager `aten::linear` ASSUMES about any
    // weight handed to it — from the shape alone, it cannot see how the weight
    // was actually swizzled.
    //
    // ⚠️ It is NOT what the Python converter always chooses. `o_proj` /
    // `down_proj` are pinned to ROW by name, and attention projections are
    // swizzled over `attn_num_cores` rather than NUM_CORES, because that is
    // what the fused SPM decoders launch them with. For typical LLM dims both
    // partitions are legal, so those weights disagree with this tie-break
    // when both partitions fit. The converter therefore records its choice on the
    // module and installs a forward that honours it —
    // `runtime/weights.py::_record_linear_layout`. Nothing that reaches this
    // function is allowed to carry a layout this tie-break would misread.
    return 1;
  } else if (can_row) {
    return 0;
  } else if (can_col) {
    return 1;
  } else {
    return -1;  // fallback
  }
}

// Transform weight for device linear kernel
// This should be called once before inference (e.g., in model loading)
// partition: 0=row, 1=col (use rpu_get_linear_partition to determine)
Tensor rpu_transform_linear_weight(const Tensor &weight, int partition) {
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, in_features]");
  TORCH_CHECK(partition == 0 || partition == 1, "partition must be 0 (row) or 1 (col)");

  int64_t dwidth = weight.element_size();

  if (partition == 0) {
    return tp_row_swizzle_mc_weight(weight, NUM_CORES, dwidth);
  } else {
    return tp_col_swizzle_mc_weight(weight, NUM_CORES, dwidth);
  }
}

// Main kernel: linear implemented for backend "rpu"
// NOTE: weight is expected to be pre-transformed using rpu_transform_linear_weight()
// The partition parameter must match the transformation applied to weight
Tensor rpu_linear(const Tensor &input, const Tensor &weight,
                  const c10::optional<Tensor> &bias_opt) {
  // return cpu_fallback_linear(input, weight, bias_opt);
#if RPU_PROFILE_ENABLED
  WrapperProfileData wp;
  PROFILE_START();
#endif

  // Basic checks
  check_linear_shapes(input, weight, bias_opt);

  // Decide compute dtype/device
  Device device = input.device();
  ScalarType dtype = input.scalar_type();

  // Ensure contiguous for efficient matmul
  Tensor input_c = input;
  if (!input.is_contiguous()) {
    input_c = input.contiguous();
  }

  // For simplicity: handle no half
  if (dtype != ScalarType::Half) {
    return cpu_fallback_linear(input, weight, bias_opt);
  }

  int64_t in_features = weight.size(1);
  int64_t out_features = weight.size(0);
  int64_t dwidth = input.element_size();  // fp16 = 2 bytes

  // Get partition strategy
  int partition = rpu_get_linear_partition(in_features, out_features, dwidth);
  if (partition < 0) {
    TORCH_WARN_ONCE("Warning: Cannot use multi-core linear for K=", in_features,
                    ", N=", out_features, ". Falling back to CPU.");
    return cpu_fallback_linear(input, weight, bias_opt);
  }

  int64_t batch = 1;
  std::vector<int64_t> leading_shape;
  if (input_c.dim() > 1) {
    leading_shape.assign(input_c.sizes().data(),
                         input_c.sizes().data() + input_c.dim() - 1);
    batch = 1;
    for (auto s : leading_shape)
      batch *= s;
  } else {
    batch = 1;
  }

  Tensor input_2d;
  if (input_c.dim() == 1) {
    input_2d = input_c.unsqueeze(0);
  } else {
    input_2d = input_c.reshape({batch, in_features});
  }

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(wp.preprocess_ms);
#endif

  // NOTE: weight is already pre-transformed, use directly
  // No weight transformation here - should be done in Python before inference

  // Tensor weight_trans = rpu_transform_linear_weight(weight, partition);

  Tensor bias;
  bool has_bias = false;
  if (bias_opt.has_value()) {
    has_bias = true;
    bias = bias_opt.value();
  }

  Tensor out_2d = at::empty({batch, out_features}, input.options());

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(wp.alloc_ms);
  wp.contiguous_ms = 0;  // contiguous 已在 preprocess 阶段完成
#endif

  rpu_launch_linear_ddr_kernel(input_2d, weight, out_2d, bias, has_bias, partition);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(wp.kernel_ms);
#endif

  // Restore output shape - 优化：避免不必要的 reshape
  // 对于最常见的 2D input [batch, in_features]，output 已经是正确的 shape [batch, out_features]
  if (input_c.dim() != 2) {
    if (input_c.dim() == 1) {
      out_2d = out_2d.view({out_features});
    } else {
      leading_shape.push_back(out_features);
      out_2d = out_2d.view(leading_shape);
    }
  }

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(wp.postprocess_ms);
  wp.total_ms = wp.preprocess_ms + wp.alloc_ms + wp.contiguous_ms + wp.kernel_ms + wp.postprocess_ms;
  PROFILE_RECORD_AND_PRINT("linear", g_profile_linear, wp, g_last_kernel_profile);
#endif

  return out_2d;
}

Tensor rpu_linear_with_accumulation(
    const Tensor& input, const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    int64_t partition, int64_t num_cores, bool linear_acc32,
    const c10::optional<Tensor>& scale_opt) {
  RECORD_FUNCTION("rpu::linear_with_accumulation", {});
  const bool packed = weight.scalar_type() == at::kByte;
  if (packed) {
    TORCH_CHECK(scale_opt.has_value(), "packed W4 Linear requires scales");
    check_linear_w4a16_operands(input, weight, *scale_opt, bias_opt);
  } else {
    check_linear_shapes(input, weight, bias_opt);
  }
  TORCH_CHECK(input.device().type() == at::kPrivateUse1 &&
                  weight.device() == input.device(),
              "linear_with_core_count: input and weight must be on the same device");
  TORCH_CHECK(input.scalar_type() == at::kHalf &&
                  (weight.scalar_type() == at::kHalf ||
                   weight.scalar_type() == at::kChar || packed),
              "linear_with_accumulation: FP16 input and FP16/W8/W4 weight required");
  TORCH_CHECK(weight.is_contiguous(),
              "linear_with_core_count: swizzled weight must be contiguous");
  TORCH_CHECK(!graph_dma::active(),
              "linear_with_core_count: fresh DDR operands require execution outside graph capture");

  const int64_t k = packed ? input.size(-1) : weight.size(1);
  const int64_t n = weight.size(0);
  TORCH_CHECK(k > 0, "linear_with_core_count: in_features must be positive");
  const int64_t rows = input.numel() / k;
  validate_parallel_linear_geometry(rows, n, k, partition, num_cores);

  Tensor bias;
  const bool has_bias = bias_opt.has_value();
  if (has_bias) {
    bias = *bias_opt;
    TORCH_CHECK(bias.device() == input.device() &&
                    bias.scalar_type() == at::kHalf &&
                    bias.dim() == 1 && bias.numel() == n && bias.is_contiguous(),
                "linear_with_core_count: bias must be a contiguous FP16 [N] tensor on the input device");
  }
  auto output_shape = input.sizes().vec();
  output_shape.back() = n;
  Tensor input_2d = input.contiguous().view({rows, k});
  Tensor output = at::empty(output_shape, input.options());
  Tensor output_2d = output.view({rows, n});
  const Tensor scale = scale_opt.has_value()
      ? (packed ? rpu_retain_linear_quant_scale(weight, *scale_opt) : *scale_opt)
      : Tensor{};
  // shortcut: the measured TTS Predictor head geometry; benchmark before expanding.
  // A one-shot Graph joins upload/compute/write-back without retaining fresh operands.
  if (!RpuKernelGraph::has_active() && rows == 1 && k == 1024 && n == 2048 && partition == 1 &&
      num_cores == 8 && !linear_acc32 && !has_bias &&
      weight.scalar_type() == at::kHalf && !scale.defined()) {
    const uint64_t dst = RpuGetDevAddr(output_2d.data_ptr<c10::Half>());
    RpuKernelGraph graph;
    graph.begin();
    try {
      graph.keep_alive(input_2d);
      graph.keep_alive(weight);
      graph.keep_alive(output_2d);
      rpu_launch_linear_ddr_kernel(
          input_2d, weight, output_2d, bias, false, 1,
          &dst, 0, nullptr, 8, {}, false);
      graph.end();
    } catch (...) {
      graph.abort();
      throw;
    }
  } else {
    rpu_launch_linear_ddr_kernel(
        input_2d, weight, output_2d, bias, has_bias, static_cast<int>(partition),
        nullptr, 0, nullptr, static_cast<int>(num_cores), scale, linear_acc32);
  }
  return output;
}

namespace {

constexpr int64_t kTtsCodecLayerNormRows = 16;
constexpr size_t kTtsCodecIm2colBudget = 4ULL << 20;

Tensor tts_codec_cpu_nct_to_device_rows(const Tensor& input,
                                         const Tensor& device_owner) {
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                  input.layout() == at::kStrided && input.dim() == 3 &&
                  input.size(0) == 1,
              "TTS codec RPU path expects CPU FP32 [1, channels, samples]");
  auto rows = at::empty(
      {input.size(2), input.size(1)}, device_owner.options().dtype(at::kHalf));
  auto writable = rpu_to_cpu_zerocopy_no_graph_sync(rows, false);
  writable.copy_(input.transpose(1, 2).reshape(rows.sizes()));
  rpu_ddr_flush_force_sized(rows.data_ptr(), rows.nbytes());
  return rows;
}

Tensor tts_codec_device_rows_to_cpu_nct(const Tensor& rows) {
  auto readable = rpu_to_cpu_zerocopy(rows);
  auto output = at::empty(
      {1, rows.size(1), rows.size(0)},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
  output.copy_(readable.transpose(0, 1).unsqueeze(0));
  return output;
}

Tensor tts_codec_im2col_rows(const Tensor& rows, int64_t kernel,
                             int64_t dilation, int64_t left_padding) {
  TORCH_CHECK(rows.device().type() == at::kPrivateUse1 &&
                  rows.scalar_type() == at::kHalf && rows.dim() == 2 &&
                  rows.is_contiguous() && kernel > 0 && dilation > 0 &&
                  left_padding >= 0,
              "TTS codec RPU im2col received invalid rows or geometry");
  const int64_t samples = rows.size(0);
  const int64_t channels = rows.size(1);
  const int64_t width = (kernel - 1) * dilation + 1;
  const int64_t count = samples + left_padding - width + 1;
  TORCH_CHECK(count > 0,
              "TTS codec RPU im2col input is shorter than its kernel");
  auto patches = at::empty({count, kernel * channels}, rows.options());
  if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
  SpmAllocator::ScopedTemporary temporary;
  const size_t fixed_bytes =
      (width - 1) * channels * sizeof(c10::Half);
  const size_t bytes_per_row =
      (kernel + 1) * channels * sizeof(c10::Half);
  const size_t budget = std::min(SPM_ALLOC.free_space(), kTtsCodecIm2colBudget);
  constexpr size_t alignment_headroom = 512;
  TORCH_CHECK(budget >= fixed_bytes + bytes_per_row + alignment_headroom,
              "TTS codec RPU im2col has insufficient free SPM");
  const int64_t max_rows = std::min<int64_t>(
      count, (budget - fixed_bytes - alignment_headroom) / bytes_per_row);
  const int64_t max_segment = max_rows + width - 1;
  const uint32_t input_off = SPM_ALLOC.alloc_temporary(
      max_segment * channels * sizeof(c10::Half));
  const uint32_t column_off = SPM_ALLOC.alloc_temporary(
      max_rows * kernel * channels * sizeof(c10::Half));
  const uint32_t input_spm = SPM_ALLOC.addr(0, input_off);
  const uint32_t column_spm = SPM_ALLOC.addr(0, column_off);
  auto* source = const_cast<c10::Half*>(rows.data_ptr<c10::Half>());
  auto* destination = patches.data_ptr<c10::Half>();
  for (int64_t row = 0; row < count; row += max_rows) {
    const int64_t chunk = std::min(count - row, max_rows);
    const int64_t segment = chunk + width - 1;
    const int64_t source_begin = row - left_padding;
    const int64_t valid_begin = std::max<int64_t>(0, source_begin);
    const int64_t valid_end = std::min(samples, source_begin + segment);
    rpu_launch_fill_spm_kernel(
        input_spm, segment * channels, c10::Half(0.0f), 1);
    if (valid_end > valid_begin) {
      rpu_launch_ddr_broadcast_spm_dma_immediate(
          source + valid_begin * channels,
          (valid_end - valid_begin) * channels,
          input_spm + (valid_begin - source_begin) * channels * sizeof(c10::Half),
          1);
    }
    rpu_launch_im2col_spm(
        input_off, column_off, 1, 1, static_cast<int>(segment),
        static_cast<int>(channels), 1, static_cast<int>(kernel),
        0, 0, 0, 0, 1, 1, 1, static_cast<int>(dilation), 1);
    rpu_launch_spm_copy_ddr_dma_immediate(
        column_spm, destination + row * kernel * channels,
        chunk * kernel * channels);
  }
  return patches;
}

Tensor tts_codec_snake_rows(const Tensor& rows, const Tensor& alpha,
                            const Tensor& scale) {
  auto sine = at::sin(at::mul(rows, alpha));
  return at::add(rows, at::mul(at::mul(sine, sine), scale));
}

Tensor tts_codec_layernorm_rows(const Tensor& rows, const Tensor& weight,
                                const Tensor& bias, double eps) {
  const int64_t channels = rows.size(1);
  auto output = at::empty_like(rows);
  if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
  SpmAllocator::ScopedTemporary temporary;
  const int64_t max_rows = std::min(
      rows.size(0), kTtsCodecLayerNormRows);
  const uint32_t input_spm = SPM_ALLOC.addr(
      0, SPM_ALLOC.alloc_temporary(
             max_rows * channels * sizeof(c10::Half)));
  const uint32_t output_spm = SPM_ALLOC.addr(
      0, SPM_ALLOC.alloc_temporary(
             max_rows * channels * sizeof(c10::Half)));
  const uint32_t weight_spm = SPM_ALLOC.addr(
      0, SPM_ALLOC.alloc_temporary(channels * sizeof(c10::Half)));
  const uint32_t bias_spm = SPM_ALLOC.addr(
      0, SPM_ALLOC.alloc_temporary(channels * sizeof(c10::Half)));
  rpu_launch_ddr_broadcast_spm_dma_immediate(
      const_cast<c10::Half*>(weight.data_ptr<c10::Half>()),
      channels, weight_spm, 1);
  rpu_launch_ddr_broadcast_spm_dma_immediate(
      const_cast<c10::Half*>(bias.data_ptr<c10::Half>()),
      channels, bias_spm, 1);
  auto* source = const_cast<c10::Half*>(rows.data_ptr<c10::Half>());
  auto* destination = output.data_ptr<c10::Half>();
  for (int64_t row = 0; row < rows.size(0); row += max_rows) {
    const int64_t count = std::min(rows.size(0) - row, max_rows);
    rpu_launch_ddr_broadcast_spm_dma_immediate(
        source + row * channels, count * channels, input_spm, 1);
    rpu_launch_layernorm_spm_kernel(
        input_spm, output_spm, weight_spm, bias_spm,
        count, channels, eps, false, 0, 1);
    rpu_launch_spm_copy_ddr_dma_immediate(
        output_spm, destination + row * channels, count * channels);
  }
  return output;
}

void check_tts_codec_device_half(const Tensor& tensor,
                                 c10::IntArrayRef shape,
                                 const char* name) {
  TORCH_CHECK(tensor.device().type() == at::kPrivateUse1 &&
                  tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
                  tensor.sizes().equals(shape),
              name, " has invalid device FP16 geometry");
}

}  // namespace

Tensor rpu_tts_codec_conv1d_spm(
    const Tensor& input, const Tensor& weight,
    const c10::optional<Tensor>& bias,
    int64_t kernel, int64_t dilation, int64_t out_channels) {
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "tts_codec_conv1d_spm: eager tiled execution required");
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                  input.dim() == 3 && input.size(0) == 1,
              "tts_codec_conv1d_spm: expected CPU FP32 input");
  const int64_t channels = input.size(1);
  const bool residual = channels == out_channels &&
      (channels == 96 || channels == 192 || channels == 384 || channels == 768) &&
      ((kernel == 1 && dilation == 1) ||
       (kernel == 7 && (dilation == 1 || dilation == 3 || dilation == 9)));
  const bool other = dilation == 1 &&
      ((channels == 256 && out_channels == 512 && kernel == 1) ||
       (channels == 512 && out_channels == 1024 && kernel == 3) ||
       (channels == 1024 && out_channels == 1024 && kernel == 7) ||
       (channels == 1024 && out_channels == 1536 && kernel == 7) ||
       (channels == 96 && out_channels == 1 && kernel == 7));
  TORCH_CHECK(residual || other,
              "tts_codec_conv1d_spm: outside the frozen decode profile");
  const int64_t count = input.size(2) - (kernel - 1) * dilation;
  TORCH_CHECK(count > 0 && count <= 255 * 1920,
              "tts_codec_conv1d_spm: invalid or unbounded sample count");
  const int64_t padded = CeilDiv(out_channels, int64_t{128}) * 128;
  check_tts_codec_device_half(
      weight, {padded, kernel * channels},
      "tts_codec_conv1d_spm weight");
  TORCH_CHECK(!bias.has_value() ||
                  (bias->device().is_cpu() && bias->scalar_type() == at::kFloat &&
                   bias->is_contiguous() && bias->sizes().equals({out_channels})),
              "tts_codec_conv1d_spm: expected CPU FP32 bias");
  auto rows = tts_codec_cpu_nct_to_device_rows(input, weight);
  auto patches = tts_codec_im2col_rows(rows, kernel, dilation, 0);
  auto projected = rpu_linear_with_accumulation(
      patches, weight, c10::nullopt, 1, 8, true, c10::nullopt)
                       .narrow(1, 0, out_channels);
  auto output = tts_codec_device_rows_to_cpu_nct(projected);
  if (bias.has_value()) output.add_(bias->view({1, out_channels, 1}));
  return output;
}

Tensor rpu_tts_codec_residual_unit(
    const Tensor& input, const Tensor& first_weight, const Tensor& affine,
    const Tensor& second_weight, int64_t dilation, int64_t channels) {
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "tts_codec_residual_unit: eager tiled execution required");
  TORCH_CHECK((channels == 96 || channels == 192 || channels == 384 ||
               channels == 768) &&
                  (dilation == 1 || dilation == 3 || dilation == 9) &&
                  input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                  input.dim() == 3 && input.size(0) == 1 &&
                  input.size(1) == channels && input.size(2) > 0 &&
                  input.size(2) <= 255 * 1920,
              "tts_codec_residual_unit: outside the frozen residual profile");
  const int64_t padded = CeilDiv(channels, int64_t{128}) * 128;
  check_tts_codec_device_half(
      first_weight, {padded, channels * 7},
      "tts_codec_residual_unit first weight");
  check_tts_codec_device_half(
      second_weight, {padded, padded},
      "tts_codec_residual_unit second weight");
  check_tts_codec_device_half(
      affine, {4, padded}, "tts_codec_residual_unit affine");
  const auto channel = [&](int64_t row) {
    return affine.select(0, row);
  };
  auto hidden = tts_codec_cpu_nct_to_device_rows(input, first_weight);
  auto patches = tts_codec_im2col_rows(hidden, 7, dilation, 6 * dilation);
  hidden = rpu_linear_with_accumulation(
      patches, first_weight, c10::nullopt, 1, 8, true, c10::nullopt);
  hidden = at::add(hidden, channel(0));
  hidden = tts_codec_snake_rows(
      hidden, channel(1), channel(2));
  hidden = rpu_linear_with_accumulation(
      hidden, second_weight, c10::nullopt, 1, 8, true, c10::nullopt);
  hidden = at::add(hidden, channel(3));
  return tts_codec_device_rows_to_cpu_nct(hidden.narrow(1, 0, channels));
}

Tensor rpu_tts_codec_snake(
    const Tensor& input, const Tensor& affine, int64_t channels) {
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "tts_codec_snake: eager execution required");
  TORCH_CHECK((channels == 96 || channels == 192 || channels == 384 ||
               channels == 768 || channels == 1536) &&
                  input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                  input.dim() == 3 && input.size(0) == 1 &&
                  input.size(1) == channels && input.size(2) > 0 &&
                  input.size(2) <= 255 * 640,
              "tts_codec_snake: outside the frozen standalone profile");
  const int64_t padded = CeilDiv(channels, int64_t{128}) * 128;
  check_tts_codec_device_half(
      affine, {2, padded}, "tts_codec_snake affine");
  auto alpha = affine.select(0, 0).narrow(0, 0, channels);
  auto scale = affine.select(0, 1).narrow(0, 0, channels);
  auto rows = tts_codec_cpu_nct_to_device_rows(input, affine);
  rows = tts_codec_snake_rows(rows, alpha, scale);
  return tts_codec_device_rows_to_cpu_nct(rows);
}

Tensor rpu_tts_codec_convnext_block(
    const Tensor& input, const Tensor& depthwise_weight,
    const Tensor& depthwise_bias, const Tensor& norm_weight,
    const Tensor& norm_bias, const Tensor& first_weight,
    const Tensor& first_bias, const Tensor& second_weight,
    const Tensor& second_bias, const Tensor& gamma, double eps) {
  constexpr int64_t channels = 1024;
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "tts_codec_convnext_block: eager tiled execution required");
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                  input.dim() == 3 && input.size(0) == 1 &&
                  input.size(1) == channels && input.size(2) > 0 &&
                  input.size(2) <= 255 * 640 &&
                  eps == 1e-6,
              "tts_codec_convnext_block: outside the frozen ConvNeXt profile");
  check_tts_codec_device_half(
      depthwise_weight, {channels, channels * 7},
      "tts_codec_convnext_block depthwise weight");
  check_tts_codec_device_half(depthwise_bias, {channels},
                              "tts_codec_convnext_block depthwise bias");
  check_tts_codec_device_half(norm_weight, {channels},
                              "tts_codec_convnext_block norm weight");
  check_tts_codec_device_half(norm_bias, {channels},
                              "tts_codec_convnext_block norm bias");
  check_tts_codec_device_half(first_weight, {4096, channels},
                              "tts_codec_convnext_block first weight");
  check_tts_codec_device_half(first_bias, {4096},
                              "tts_codec_convnext_block first bias");
  check_tts_codec_device_half(second_weight, {channels, 4096},
                              "tts_codec_convnext_block second weight");
  check_tts_codec_device_half(second_bias, {channels},
                              "tts_codec_convnext_block second bias");
  check_tts_codec_device_half(gamma, {channels},
                              "tts_codec_convnext_block gamma");

  auto residual = tts_codec_cpu_nct_to_device_rows(input, depthwise_weight);
  auto patches = tts_codec_im2col_rows(residual, 7, 1, 6);
  auto hidden = rpu_linear_with_accumulation(
      patches, depthwise_weight, depthwise_bias, 1, 8, true, c10::nullopt);
  auto normalized = tts_codec_layernorm_rows(
      hidden, norm_weight, norm_bias, eps);
  hidden = rpu_linear_with_accumulation(
      normalized, first_weight, first_bias, 1, 8, true, c10::nullopt);
  hidden = rpu_gelu_erf(hidden);
  hidden = rpu_linear_with_accumulation(
      hidden, second_weight, second_bias, 1, 8, true, c10::nullopt);
  hidden = at::mul(hidden, gamma);
  return tts_codec_device_rows_to_cpu_nct(at::add(hidden, residual));
}

Tensor rpu_tts_codec_conv1d(
    const Tensor& input, const Tensor& weight,
    const c10::optional<Tensor>& bias,
    int64_t kernel, int64_t dilation, int64_t out_channels) {
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "tts_codec_conv1d: shared-DDR staging requires eager execution");
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                  input.layout() == at::kStrided && input.dim() == 3 && input.size(0) == 1,
              "tts_codec_conv1d: expected CPU FP32 [1, channels, samples]");
  const int64_t channels = input.size(1);
  const bool residual = channels == out_channels &&
      (channels == 96 || channels == 192 || channels == 384 || channels == 768) &&
      ((kernel == 1 && dilation == 1) ||
       (kernel == 7 && (dilation == 1 || dilation == 3 || dilation == 9)));
  const bool other = dilation == 1 &&
      ((channels == 256 && out_channels == 512 && kernel == 1) ||
       (channels == 512 && out_channels == 1024 && kernel == 3) ||
       (channels == 1024 && out_channels == 1024 && kernel == 7) ||
       (channels == 1024 && out_channels == 1536 && kernel == 7) ||
       (channels == 96 && out_channels == 1 && kernel == 7));
  TORCH_CHECK(residual || other,
              "tts_codec_conv1d: outside the frozen 31-layer decode profile");
  const int64_t width = (kernel - 1) * dilation + 1;
  const int64_t count = input.size(2) - width + 1;
  TORCH_CHECK(count > 0 && count <= 255 * 1920,
              "tts_codec_conv1d: invalid or unbounded sample count");
  const int64_t padded_channels = ((out_channels + 127) / 128) * 128;
  TORCH_CHECK(weight.device().type() == at::kPrivateUse1 &&
                  weight.scalar_type() == at::kHalf && weight.is_contiguous() &&
                  weight.sizes().equals({padded_channels, channels * kernel}),
              "tts_codec_conv1d: expected device FP16 col-swizzled padded weight");
  TORCH_CHECK(!bias.has_value() ||
                  (bias->device().is_cpu() && bias->scalar_type() == at::kFloat &&
                   bias->is_contiguous() && bias->sizes().equals({out_channels})),
              "tts_codec_conv1d: expected CPU FP32 channel bias");

  // Fresh device allocation owns the CPU view through synchronous Linear completion.
  auto patches = at::empty({count, channels * kernel}, weight.options());
  auto writable = rpu_to_cpu_zerocopy_no_graph_sync(patches, false);
  writable.view({1, count, channels, kernel}).copy_(
      input.unfold(2, width, 1).slice(3, 0, width, dilation).permute({0, 2, 1, 3}));
  rpu_ddr_flush_force_sized(patches.data_ptr(), patches.nbytes());
  auto projected = rpu_linear_with_accumulation(
      patches, weight, c10::nullopt, 1, 8, true, c10::nullopt);
  auto readable = rpu_to_cpu_zerocopy(projected);
  auto output = at::empty({1, out_channels, count}, input.options());
  // Crop before conversion; copy directly into the final NCT FP32 layout.
  output.copy_(readable.narrow(1, 0, out_channels).transpose(0, 1).unsqueeze(0));
  if (bias.has_value()) output.add_(bias->view({1, out_channels, 1}));
  return output;
}

Tensor rpu_tts_codec_conv_pair(
    const Tensor& input, const Tensor& first_weight, const Tensor& affine,
    const Tensor& second_weight, const Tensor& second_bias,
    int64_t dilation, int64_t channels) {
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "tts_codec_conv_pair: shared-DDR staging requires eager execution");
  TORCH_CHECK((channels == 768 || channels == 384 || channels == 192 || channels == 96) &&
                  (dilation == 1 || dilation == 3 || dilation == 9),
              "tts_codec_conv_pair: outside the frozen residual profile");
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                  input.layout() == at::kStrided && input.dim() == 3 &&
                  input.size(0) == 1 && input.size(1) == channels,
              "tts_codec_conv_pair: expected CPU FP32 [1, channels, samples]");
  const int64_t width = 6 * dilation + 1;
  const int64_t count = input.size(2) - width + 1;
  const int64_t padded = ((channels + 127) / 128) * 128;
  TORCH_CHECK(count > 0 && count <= 255 * 1920,
              "tts_codec_conv_pair: invalid or unbounded sample count");
  for (const auto& tensor : {first_weight, affine, second_weight}) {
    TORCH_CHECK(tensor.device().type() == at::kPrivateUse1 &&
                    tensor.device() == first_weight.device() &&
                    tensor.scalar_type() == at::kHalf && tensor.is_contiguous(),
                "tts_codec_conv_pair: expected contiguous device FP16 operands");
  }
  TORCH_CHECK(first_weight.sizes().equals({padded, channels * 7}) &&
                  second_weight.sizes().equals({padded, padded}) &&
                  affine.sizes().equals({3, padded}),
              "tts_codec_conv_pair: invalid packed weights or bias/alpha/scale");
  TORCH_CHECK(second_bias.device().is_cpu() && second_bias.scalar_type() == at::kFloat &&
                  second_bias.is_contiguous() && second_bias.sizes().equals({channels}),
              "tts_codec_conv_pair: expected CPU FP32 output bias");

  auto patches = at::empty({count, channels * 7}, first_weight.options());
  auto writable = rpu_to_cpu_zerocopy_no_graph_sync(patches, false);
  writable.view({1, count, channels, 7}).copy_(
      input.unfold(2, width, 1).slice(3, 0, width, dilation).permute({0, 2, 1, 3}));
  rpu_ddr_flush_force_sized(patches.data_ptr(), patches.nbytes());
  auto projected = rpu_linear_with_accumulation(
      patches, first_weight, c10::nullopt, 1, 8, true, c10::nullopt);
  auto hidden = projected + affine.select(0, 0);
  auto sine = at::sin(hidden * affine.select(0, 1));
  auto activated = hidden + (sine * sine) * affine.select(0, 2);
  auto second = rpu_linear_with_accumulation(
      activated, second_weight, c10::nullopt, 1, 8, true, c10::nullopt);
  auto readable = rpu_to_cpu_zerocopy(second);
  auto output = at::empty({1, channels, count}, input.options());
  output.copy_(readable.narrow(1, 0, channels).transpose(0, 1).unsqueeze(0));
  output.add_(second_bias.view({1, channels, 1}));
  return output;
}

Tensor rpu_tts_codec_conv_transpose1d(
    const Tensor& input, const Tensor& weight,
    const c10::optional<Tensor>& bias,
    int64_t out_channels, int64_t kernel, int64_t stride) {
  TORCH_CHECK(!RpuKernelGraph::has_active(),
              "tts_codec_conv_transpose1d: shared-DDR staging requires eager execution");
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kFloat &&
                  input.layout() == at::kStrided && input.dim() == 3 && input.size(0) == 1,
              "tts_codec_conv_transpose1d: expected CPU FP32 [1, channels, samples]");
  const int64_t channels = input.size(1);
  const bool upsample = channels == 1024 && out_channels == 1024 && kernel == 2 && stride == 2;
  const bool decoder =
      (channels == 1536 && out_channels == 768 && kernel == 16 && stride == 8) ||
      (channels == 768 && out_channels == 384 && kernel == 10 && stride == 5) ||
      (channels == 384 && out_channels == 192 && kernel == 8 && stride == 4) ||
      (channels == 192 && out_channels == 96 && kernel == 6 && stride == 3);
  TORCH_CHECK(upsample || decoder,
              "tts_codec_conv_transpose1d: outside the frozen six-layer decode profile");
  const int64_t count = input.size(2);
  const int64_t padded = ((out_channels * kernel + 127) / 128) * 128;
  TORCH_CHECK(count > 0 && count <= 255 * 640,
              "tts_codec_conv_transpose1d: invalid or unbounded sample count");
  TORCH_CHECK(weight.device().type() == at::kPrivateUse1 &&
                  weight.scalar_type() == at::kHalf && weight.is_contiguous() &&
                  weight.sizes().equals({padded, channels}),
              "tts_codec_conv_transpose1d: expected device FP16 col-swizzled padded weight");
  TORCH_CHECK(!bias.has_value() ||
                  (bias->device().is_cpu() && bias->scalar_type() == at::kFloat &&
                   bias->is_contiguous() && bias->sizes().equals({out_channels})),
              "tts_codec_conv_transpose1d: expected CPU FP32 channel bias");

  auto rows = at::empty({count, channels}, weight.options());
  auto writable = rpu_to_cpu_zerocopy_no_graph_sync(rows, false);
  writable.copy_(input.transpose(1, 2).reshape({count, channels}));
  rpu_ddr_flush_force_sized(rows.data_ptr(), rows.nbytes());
  auto projected = rpu_linear_with_accumulation(
      rows, weight, c10::nullopt, 1, 8, true, c10::nullopt);
  auto readable = rpu_to_cpu_zerocopy(projected).narrow(1, 0, out_channels * kernel);
  Tensor output;
  if (kernel == stride) {
    output = at::empty({1, out_channels, count * stride}, input.options());
    output.copy_(readable.view({1, count, out_channels, stride})
                     .permute({0, 2, 1, 3}).reshape(output.sizes()));
  } else {
    auto chunks = readable.view({1, count, out_channels, 2, stride});
    output = at::empty({1, out_channels, count + 1, stride}, input.options());
    output.select(2, 0).copy_(chunks.select(1, 0).select(2, 0));
    output.select(2, count).copy_(chunks.select(1, count - 1).select(2, 1));
    auto middle = output.slice(2, 1, count);
    middle.copy_(chunks.slice(1, 1, count).select(3, 0).permute({0, 2, 1, 3}));
    middle.add_(chunks.slice(1, 0, count - 1).select(3, 1).permute({0, 2, 1, 3}));
    output = output.flatten(2);
  }
  if (bias.has_value()) output.add_(bias->view({1, out_channels, 1}));
  return output;
}

Tensor rpu_linear_with_core_count(
    const Tensor& input, const Tensor& weight,
    const c10::optional<Tensor>& bias_opt,
    int64_t partition, int64_t num_cores) {
  TORCH_CHECK(weight.scalar_type() == at::kHalf,
              "linear_with_core_count: weight must be FP16");
  return rpu_linear_with_accumulation(
      input, weight, bias_opt, partition, num_cores, true, c10::nullopt);
}

void rpu_linear_into(const Tensor &input, const Tensor &weight,
                     const c10::optional<Tensor> &bias_opt,
                     Tensor &output, bool linear_acc32) {
  RECORD_FUNCTION("rpu::linear_into", {});
  check_linear_shapes(input, weight, bias_opt);
  TORCH_CHECK(input.scalar_type() == at::kHalf,
              "rpu_linear_into: input must be Half");
  TORCH_CHECK(weight.scalar_type() == at::kHalf,
              "rpu_linear_into: weight must be Half");
  TORCH_CHECK(output.scalar_type() == at::kHalf,
              "rpu_linear_into: output must be Half");
  TORCH_CHECK(input.device().type() == at::kPrivateUse1 &&
                  weight.device().type() == at::kPrivateUse1 &&
                  output.device().type() == at::kPrivateUse1,
              "rpu_linear_into: input, weight, and output must be on device");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous(),
              "rpu_linear_into: input, weight, and output must be contiguous");

  const int64_t in_features = weight.size(1);
  const int64_t out_features = weight.size(0);
  int64_t batch = 1;
  std::vector<int64_t> output_shape = input.sizes().vec();
  output_shape.back() = out_features;
  TORCH_CHECK(output.sizes() == output_shape,
              "rpu_linear_into: output shape must be ", output_shape,
              ", got ", output.sizes());

  std::vector<int64_t> leading_shape;
  if (input.dim() > 1) {
    leading_shape.assign(input.sizes().data(),
                         input.sizes().data() + input.dim() - 1);
    for (auto s : leading_shape)
      batch *= s;
  }
  Tensor input_2d;
  if (input.dim() == 1) {
    input_2d = input.unsqueeze(0);
  } else {
    input_2d = input.view({batch, in_features});
  }
  Tensor output_2d = output.view({batch, out_features});

  const int partition =
      rpu_get_linear_partition(in_features, out_features, input.element_size());
  TORCH_CHECK(partition >= 0,
              "rpu_linear_into: unsupported K=", in_features,
              ", N=", out_features);

  Tensor bias;
  bool has_bias = false;
  if (bias_opt.has_value()) {
    bias = bias_opt.value();
    has_bias = true;
    TORCH_CHECK(bias.device().type() == at::kPrivateUse1,
                "rpu_linear_into: bias must be on device");
    TORCH_CHECK(bias.scalar_type() == at::kHalf,
                "rpu_linear_into: bias must be Half");
    TORCH_CHECK(bias.dim() == 1 && bias.numel() == out_features,
                "rpu_linear_into: bias must be a vector with ", out_features,
                " elements, got shape ", bias.sizes());
    TORCH_CHECK(bias.is_contiguous(),
                "rpu_linear_into: bias must be contiguous");
  }

  rpu_launch_linear_ddr_kernel(
      input_2d, weight, output_2d, bias, has_bias, partition,
      nullptr, 0, nullptr, NUM_CORES, {}, linear_acc32);
}

void rpu_linear_w8a16_into(const Tensor& input, const Tensor& weight,
                          const Tensor& scale,
                          const c10::optional<Tensor>& bias_opt,
                          Tensor& output, bool linear_acc32) {
  RECORD_FUNCTION("rpu::linear_w8a16_into", {});
  check_linear_shapes(input, weight, bias_opt);
  TORCH_CHECK(!graph_dma::active(),
              "linear_w8a16: eager DDR operands cannot be recorded in a graph");
  TORCH_CHECK(input.device().type() == at::kPrivateUse1 &&
                  weight.device() == input.device() &&
                  output.device() == input.device(),
              "linear_w8a16: input, weight and output must share a device");
  TORCH_CHECK(weight.scalar_type() == at::kChar &&
                  output.scalar_type() == at::kHalf,
              "linear_w8a16: weight must be INT8 and output FP16");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() &&
                  output.is_contiguous(),
              "linear_w8a16: input, swizzled weight and output must be contiguous");
  const int64_t k = weight.size(1), n = weight.size(0);
  TORCH_CHECK(k > 0 && n > 0, "linear_w8a16: empty weight dimensions");
  const int64_t rows = input.numel() / k;
  validate_parallel_linear_geometry(rows, n, k, /*partition=*/1, NUM_CORES);
  auto shape = input.sizes().vec();
  shape.back() = n;
  TORCH_CHECK(output.sizes() == shape,
              "linear_w8a16: output shape must be ", shape);
  at::assert_no_overlap(output, input);
  at::assert_no_overlap(output, weight);
  if (scale.defined()) at::assert_no_overlap(output, scale);
  Tensor bias;
  if (bias_opt.has_value()) {
    bias = *bias_opt;
    TORCH_CHECK(bias.device() == input.device() &&
                    bias.scalar_type() == at::kHalf &&
                    bias.dim() == 1 && bias.numel() == n && bias.is_contiguous(),
                "linear_w8a16: bias must be contiguous device FP16 [N]");
    at::assert_no_overlap(output, bias);
  }
  Tensor input_2d = input.view({rows, k});
  Tensor output_2d = output.view({rows, n});
  rpu_launch_linear_ddr_kernel(input_2d, weight, output_2d, bias,
                              bias_opt.has_value(), /*partition=*/1,
                              nullptr, 0, nullptr, NUM_CORES, scale, linear_acc32);
}

Tensor rpu_linear_w8a16(const Tensor& input, const Tensor& weight,
                       const Tensor& scale,
                       const c10::optional<Tensor>& bias) {
  check_linear_shapes(input, weight, bias);
  TORCH_CHECK(!graph_dma::active(),
              "linear_w8a16: fresh DDR output requires eager execution");
  auto shape = input.sizes().vec();
  shape.back() = weight.size(0);
  auto output = at::empty(shape, input.options());
  rpu_linear_w8a16_into(input.contiguous(), weight, scale, bias, output);
  return output;
}

Tensor rpu_linear_w4a16(const Tensor& input, const Tensor& weight,
                       const Tensor& scale,
                       const c10::optional<Tensor>& bias) {
  RECORD_FUNCTION("rpu::linear_w4a16", {});
  check_linear_w4a16_operands(input, weight, scale, bias);
  auto shape = input.sizes().vec();
  const int64_t k = input.size(-1), n = weight.size(0);
  const int64_t rows = input.numel() / k;
  shape.back() = n;
  auto aligned_scale = rpu_retain_linear_quant_scale(weight, scale);
  auto input_2d = input.contiguous().view({rows, k});
  auto output = at::empty(shape, input.options());
  auto output_2d = output.view({rows, n});
  rpu_launch_linear_ddr_kernel(input_2d, weight, output_2d,
                              bias.has_value() ? *bias : Tensor{},
                              bias.has_value(), /*partition=*/1,
                              nullptr, 0, nullptr, NUM_CORES, aligned_scale);
  return output;
}

// ============================================================================
// O_proj Linear: SPM input -> DDR output
// ============================================================================
// This function implements the o_proj linear layer with input from SDPA output

// Old SpmAddressCache-based overloads removed.
// Use direct address versions below (rpu_launch_linear_spm_to_spm_kernel etc.)

namespace {

int autotile_n_index(int n_tile) {
  switch (n_tile) {
    case 128: return 0;
    case 112: return 1;
    case 96: return 2;
    case 80: return 3;
    case 64: return 4;
    case 48: return 5;
    case 32: return 6;
    default:
      TORCH_CHECK(false, "unsupported parallel-linear n_tile: ", n_tile);
  }
}

KernelId autotile_linear_kernel_id(bool is_fp16, int n_tile) {
  return static_cast<KernelId>(
      static_cast<int>(KernelId::PL_AT_W8A16_M288N128) +
      (is_fp16 ? 7 : 0) + autotile_n_index(n_tile));
}

KernelId autotile_linear_acc32_kernel_id(bool is_fp16, int n_tile) {
  return static_cast<KernelId>(
      static_cast<int>(KernelId::PL_AT_W8A16_ACC32_M192N128) +
      (is_fp16 ? 7 : 0) + autotile_n_index(n_tile));
}

KernelId autotile_pgrp_kernel_id(int n_tile) {
  return static_cast<KernelId>(
      static_cast<int>(KernelId::PL_AT_INT4_PGRP_M304N128) +
      autotile_n_index(n_tile));
}

KernelId autotile_pgrp_acc32_kernel_id(int n_tile) {
  return static_cast<KernelId>(
      static_cast<int>(KernelId::PL_AT_INT4_PGRP_ACC32_M208N128) +
      autotile_n_index(n_tile));
}

KernelId autotile_uint4_pgrp_kernel_id(int n_tile) {
  return static_cast<KernelId>(
      static_cast<int>(KernelId::PL_AT_UINT4_PGRP_M304N128) +
      autotile_n_index(n_tile));
}

KernelId autotile_uint4_pgrp_acc32_kernel_id(int n_tile) {
  return static_cast<KernelId>(
      static_cast<int>(KernelId::PL_AT_UINT4_PGRP_ACC32_M208N128) +
      autotile_n_index(n_tile));
}

KernelId autotile_nvfp4_acc16_kernel_id(int n_tile) {
  return static_cast<KernelId>(
      static_cast<int>(KernelId::PL_AT_NVFP4_ACC16_M176N128) +
      autotile_n_index(n_tile));
}

KernelId autotile_nvfp4_acc32_kernel_id(int n_tile) {
  return static_cast<KernelId>(
      static_cast<int>(KernelId::PL_AT_NVFP4_ACC32_M176N128) +
      autotile_n_index(n_tile));
}

// SPM bias is local to each core. For row partition, callers put the real
// bias on core 0 and zero on the other cores, so the later all-reduce counts
// it once (ParallelLinearGenerator's contract). This applies to M=1 too:
// GEMM adds the local bias internally; GEMV uses the elementwise add below.
void add_m1_gemv_bias(
    uint32_t bias_spm_addr,
    uint32_t output_spm_addr,
    int64_t n,
    int partition,
    int num_cores) {
  if (bias_spm_addr == 0) return;

  const int64_t local_n = (partition == 1) ? n / num_cores : n;
  rpu_launch_eltwise_binary_spm_kernel(
      bias_spm_addr, output_spm_addr, output_spm_addr, local_n,
      ValuOpType::ADD, c10::Half(1.0f), num_cores);
}

}  // namespace

void rpu_launch_linear_spm_to_spm_kernel(
    uint32_t input_spm_addr,           // Input SPM address (core 0 base)
    const at::Tensor &weight,          // DDR weight [out_features, in_features]
    uint32_t output_spm_addr,          // Output SPM address (core 0 base)
    int64_t M,                         // batch * seq_len
    int64_t N,                         // out_features
    int64_t K,                         // in_features
    int partition,                     // 0=row, 1=col
    int num_cores,                     // number of cores for partition (attn_tp or tp)
    uint32_t bias_spm_addr)
{
  validate_parallel_linear_geometry(M, N, K, partition, num_cores);

  size_t dwidth = sizeof(c10::Half);

  // Calculate local dimensions based on partition type
  size_t local_k, local_n, local_m;
  local_m = M;

  if (partition == 0) {
    // Row partition: split K across cores
    local_k = K / num_cores;
    local_n = N;
  } else {
    // Col partition: split N across cores
    local_k = K;
    local_n = N / num_cores;
  }

  auto tile = rpu_pl_tiling::select_tile_acc32(
      static_cast<int>(local_m), static_cast<int>(local_n),
      static_cast<int>(local_k), /*is_fp16=*/true);
  const size_t n_per_wrp = static_cast<size_t>(tile.n_tile);
  const size_t m_per_wrp = static_cast<size_t>(tile.m_tile);
  validate_parallel_linear_grid(local_m, local_n, m_per_wrp, n_per_wrp);
  const KernelId kernel_id =
      autotile_linear_acc32_kernel_id(/*is_fp16=*/true, tile.n_tile);

  // Weight in DDR
  c10::Half *wp = weight.data_ptr<c10::Half>();
  auto register_writer =
      RpuKernelGraph::active().stage_kernel_ddr_registers(
          kernel_id,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                      2, 3},
                  weight}});
  rpu_ddr_flush(wp);

  std::vector<uint16_t> grid_dims = {
    (uint16_t)CeilDiv(local_n, n_per_wrp),
    (uint16_t)CeilDiv(local_m, m_per_wrp),
    (uint16_t)1
  };

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(kernel != nullptr,
              "Failed to get FP16 ACC32 parallel_linear tile m", m_per_wrp,
              "n", n_per_wrp);
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, bias_spm_addr,
          static_cast<uint32_t>(local_m), static_cast<uint32_t>(local_n),
          static_cast<uint32_t>(local_k),
          static_cast<uint16_t>(bias_spm_addr != 0),
          static_cast<uint16_t>(num_cores),
          static_cast<uint32_t>(local_k * dwidth),
          static_cast<uint32_t>(local_n * dwidth),
          0, static_cast<uint16_t>(partition), 2});
  register_writer.write(*kernel);

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
  wq->enqueu_kernel(*kernel, grid_dims, core_list);
}

// Exact core-0 Router fusion. Keep original ACC32 GEMM and FP16 logits;
// BASE sigmoid consumes that retirement and writes the original Tp64 padding.
void rpu_launch_lingbot2_router_sigmoid_pad_spm_kernel(
    uint32_t input_spm_addr, const at::Tensor& weight,
    uint32_t logits_spm_addr, uint32_t scores_spm_addr) {
  TORCH_CHECK(weight.defined() &&
                  weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  weight.scalar_type() == at::kHalf &&
                  weight.layout() == c10::Layout::Strided && weight.is_contiguous() &&
                  weight.dim() == 2 && weight.size(0) == 32 && weight.size(1) == 768 &&
                  weight.storage_offset() >= 0,
              "Lingbot2 Router requires a contiguous device fp16 [32,768] packed weight");
  constexpr uint64_t weight_bytes = 32 * 768 * sizeof(c10::Half);
  const uint64_t storage_bytes = weight.storage().nbytes();
  const uint64_t offset = static_cast<uint64_t>(weight.storage_offset());
  TORCH_CHECK(weight.nbytes() == weight_bytes && offset <= storage_bytes / sizeof(c10::Half) &&
                  weight_bytes <= storage_bytes - offset * sizeof(c10::Half),
              "Lingbot2 Router requires complete packed weight backing");
  const uint64_t weight_address = RpuGetDevAddr(weight.data_ptr());
  constexpr uint64_t ddr_limit = uint64_t{1} << 40;
  TORCH_CHECK(weight_address != 0 && (weight_address & 255u) == 0 &&
                  weight_address < ddr_limit && weight_bytes <= ddr_limit - weight_address,
              "Lingbot2 Router requires a complete aligned shift8 DDR interval");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Lingbot2 Router requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 3> ranges{{
      {input_spm_addr, 51 * 768 * sizeof(c10::Half)},
      {logits_spm_addr, 64 * 32 * sizeof(c10::Half)},
      {scores_spm_addr, 64 * 32 * sizeof(c10::Half)}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && (address & 31u) == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                    uint64_t{address} + bytes <= (uint64_t{1} << 32),
                "Lingbot2 Router requires complete aligned absolute SPM intervals");
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(uint64_t{address} + bytes <= ranges[j].first ||
                      uint64_t{ranges[j].first} + ranges[j].second <= address,
                  "Lingbot2 Router input, full logits and full scores must be disjoint");
    }
  }
  constexpr auto id = KernelId::LINGBOT2_ROUTER_SIGMOID_PAD_T51;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Lingbot2 Router sigmoid/padding optional payload is absent");
  auto* queue = GET_QUEUE(1);
  TORCH_CHECK(queue != nullptr, "Lingbot2 Router core-0 queue is absent");
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{{
          {GraphDdrRegisterRole::ModelWeight, GraphDdrRegisterAccess::Read,
           GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight}});
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "Lingbot2 Router sigmoid/padding kernel is absent");
  graph.keep_alive(weight);
  rpu_ddr_flush(weight.data_ptr());
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, logits_spm_addr, 0, 51, 32, 768, 0, 1, 1536, 64, 0, 1, 2});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 26, scores_spm_addr);
  writer.write(*kernel);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {1, 1, 1}, {0});
}

void rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(
    uint32_t input_spm_addr,
    const at::Tensor &weight,
    uint32_t output_spm_addr,
    int64_t M,
    int64_t N,
    int64_t K,
    int partition,
    int num_cores,
    uint32_t bias_spm_addr)
{
  const bool vision_profile =
      (M == 288 || M == 576) && N == 1280 && (K == 3456 || K == 3584);
  const bool rtc_profile =
      (M == 320 || M == 336) && N == 2048 && K == 11008;
  TORCH_CHECK(
      partition == 0 && num_cores == 8 &&
          (vision_profile || rtc_profile),
      "Wall-OSS mixed Linear supports only Vision (288/576,1280,3456/3584) or "
      "RTC prefill (320/336,2048,11008), row,tp=8; got M=",
      M, " N=", N, " K=", K, " partition=", partition,
      " num_cores=", num_cores);
  TORCH_CHECK(
      (vision_profile && bias_spm_addr != 0) ||
          (rtc_profile && bias_spm_addr == 0),
      "Wall-OSS mixed Linear bias contract drifted for ",
      vision_profile ? "Vision" : "RTC");
  TORCH_CHECK(
      weight.scalar_type() == at::kHalf && weight.dim() == 2 &&
          weight.size(0) == N && weight.size(1) == K &&
          weight.is_contiguous(),
      "Wall-OSS mixed Linear requires contiguous FP16 weight [N,K]=[",
      N, ",", K, "], got ", weight.sizes(), " ", weight.scalar_type());

  const int64_t local_m = M;
  const int64_t local_n = N;
  const int64_t local_k = K / num_cores;
  TORCH_CHECK(
      K % num_cores == 0 && local_k % 16 == 0,
      "Wall-OSS mixed Linear local K must be 16-aligned; got K=", K,
      " num_cores=", num_cores);

  const KernelId kernel_id = vision_profile
      ? KernelId::PL_ACC32_OUT_BF16_M416N48
      : KernelId::PL_ACC32_OUT_BF16_M496N32;
  const uint16_t tile_m = vision_profile ? 416 : 496;
  const uint16_t tile_n = vision_profile ? 48 : 32;
  const uint16_t grid_x =
      static_cast<uint16_t>(CeilDiv(local_n, static_cast<int64_t>(tile_n)));
  const uint16_t grid_y =
      static_cast<uint16_t>(CeilDiv(local_m, static_cast<int64_t>(tile_m)));

  auto register_writer =
      RpuKernelGraph::active().stage_kernel_ddr_registers(
          kernel_id,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                      2, 3},
                  weight}});
  rpu_ddr_flush(weight.data_ptr<c10::Half>());

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      "Wall-OSS mixed Linear kernel is unavailable for ",
      vision_profile ? "Vision" : "RTC",
      "; install the combined expansion op library");
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, bias_spm_addr,
          static_cast<uint32_t>(local_m), static_cast<uint32_t>(local_n),
          static_cast<uint32_t>(local_k),
          static_cast<uint16_t>(bias_spm_addr != 0),
          static_cast<uint16_t>(num_cores),
          static_cast<uint32_t>(local_k * sizeof(c10::Half)),
          static_cast<uint32_t>(local_n * sizeof(c10::Half)),
          0, static_cast<uint16_t>(partition), 2});
  kernel->set_regs(64, grid_x);
  kernel->set_regs(65, grid_y);
  kernel->set_regs(66, static_cast<uint16_t>(1));
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(num_cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(
      *kernel, {grid_x, grid_y, static_cast<uint16_t>(1)},
      {0, 1, 2, 3, 4, 5, 6, 7});
}

// =============================================================================

void rpu_launch_g05_vision_fc2_weight_outer_linear_w8a16_acc16_kernel(
    uint32_t input_spm_addr,const at::Tensor& weight,uint32_t output_spm_addr,
    uint32_t bias_spm_addr,const at::Tensor& scale) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(),"G05 K1 B3 FC2 requires initialized SPM");
  const uint64_t base=SPM_ALLOC.addr(0,0);
  const std::array<std::pair<uint32_t,size_t>,3> regions{{
      {input_spm_addr,768*512*2},{output_spm_addr,768*1024*2},{bias_spm_addr,1024*2}}};
  for(size_t i=0;i<regions.size();++i){
    const auto&r=regions[i];
    TORCH_CHECK(r.first>=base && r.first%256==0 &&
        uint64_t(r.first)-base+r.second<=SpmAllocator::SPM_USABLE,
        "G05 K1 B3 FC2 requires complete aligned absolute SPM operands");
    for(size_t j=0;j<i;++j)TORCH_CHECK(uint64_t(r.first)+r.second<=regions[j].first ||
        uint64_t(regions[j].first)+regions[j].second<=r.first,"G05 K1 B3 FC2 SPM alias");
  }
  TORCH_CHECK(weight.defined() && weight.scalar_type()==at::kChar &&
      weight.device().type()==c10::DeviceType::PrivateUse1 && weight.is_contiguous() &&
      weight.dim()==2 && weight.size(0)==1024 && weight.size(1)==4096 &&
      weight.storage_offset()==0 && weight.nbytes()==4194304 &&
      scale.defined() && scale.scalar_type()==at::kHalf && scale.device()==weight.device() &&
      scale.is_contiguous() && scale.dim()==1 && scale.size(0)==1024 &&
      scale.storage_offset()==0 && scale.nbytes()==2048,"G05 K1 B3 FC2 full W8/scale owner required");
  const uint64_t wa=RpuGetDevAddr(weight.data_ptr()),sa=RpuGetDevAddr(scale.data_ptr());
  constexpr uint64_t limit=UINT64_C(1)<<40;
  for(const auto& p:{std::make_pair(wa,weight.nbytes()),std::make_pair(sa,scale.nbytes())})
    TORCH_CHECK(p.first && p.first%256==0 && p.first<limit && p.second<=limit-p.first,
        "G05 K1 B3 FC2 DDR range is invalid");
  TORCH_CHECK(wa+weight.nbytes()<=sa || sa+scale.nbytes()<=wa,"G05 K1 B3 FC2 DDR alias");
  constexpr auto id=KernelId::G05_VISION_FC2_WEIGHT_OUTER_W8A16_ACC16_M768N1024K512;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),"G05 K1 B3 FC2 optional payload absent");
  auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},scale}});
  rpu_ddr_flush(weight.data_ptr());rpu_ddr_flush(scale.data_ptr());
  auto* kernel=GET_KERNEL(id);TORCH_CHECK(kernel,"G05 K1 B3 FC2 payload disappeared");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,rpu_pl_tiling::ParallelLinearRegisterArgs{
      input_spm_addr,0,output_spm_addr,bias_spm_addr,768,1024,512,1,8,1024,2048,0,0,1});
  writer.write(*kernel);auto*queue=GET_QUEUE(8);queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{13,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_wall_vision_row_weight_outer_w8a16_spm_kernel(
    uint32_t input_spm_addr, const at::Tensor& weight,
    uint32_t output_spm_addr, int64_t M, int64_t N, int64_t K,
    uint32_t bias_spm_addr, const at::Tensor& scale,
    const WallSortingConfig& sorting) {
  const bool fp16 = weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 1u),
              "Wall FP16 Vision transfer requires its cold opt-in");
  const bool small_vision_rows = sorting.vision_row_reuse &&
      (M == 512 || M == 768 || M == 324 || (M == 672 && sorting.silu_mul));
  TORCH_CHECK((M == 1728 || small_vision_rows) && N == 1280 &&
                  (K == 1280 || K == 3584),
              "Wall Vision row weight-outer requires M1728 or cold small-grid "
              "M512/M768/M324 or padded-pair M672, N1280, "
              "K1280 or K3584, TP8 ACC16");
  TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  weight.scalar_type() == (fp16 ? at::kHalf : at::kChar) && weight.dim() == 2 &&
                  weight.size(0) == N && weight.size(1) == K &&
                  weight.is_contiguous(),
              "Wall Vision row weight-outer requires device int8 weight [N,K]");
  TORCH_CHECK((fp16 && !scale.defined()) || (!fp16 && scale.defined() &&
                  scale.device().type() == c10::DeviceType::PrivateUse1 &&
                  scale.scalar_type() == at::kHalf && scale.dim() == 1 &&
                  scale.size(0) == N && scale.is_contiguous()),
              "Wall Vision row weight-outer requires device fp16 scale [N]");
  TORCH_CHECK(SPM_ALLOC.is_initialized() && bias_spm_addr != 0,
              "Wall Vision row weight-outer requires initialized SPM and bias");
  const uint32_t local_k = static_cast<uint32_t>(K / NUM_CORES);
  const uint64_t input_bytes = M * local_k * sizeof(c10::Half);
  const uint64_t output_bytes = M * N * sizeof(c10::Half);
  const uint32_t spm_base = SPM_ALLOC.addr(0, 0);
  auto check_spm = [&](uint32_t address, uint64_t bytes) {
    TORCH_CHECK(address >= spm_base && address % 256 == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    static_cast<uint64_t>(address - spm_base) <=
                        SpmAllocator::SPM_USABLE - bytes,
                "Wall Vision row weight-outer requires aligned local SPM ranges");
  };
  check_spm(input_spm_addr, input_bytes);
  check_spm(output_spm_addr, output_bytes);
  check_spm(bias_spm_addr, N * sizeof(c10::Half));
  TORCH_CHECK(input_spm_addr + input_bytes <= output_spm_addr ||
                  output_spm_addr + output_bytes <= input_spm_addr,
              "Wall Vision row weight-outer input/output SPM must not overlap");

  const KernelId kernel_id = fp16
      ? (K == 1280 ? KernelId::WALL_VISION_ROW_WEIGHT_OUTER_FP16_M352N80K160
                   : KernelId::WALL_VISION_ROW_WEIGHT_OUTER_FP16_M256N80K448)
      : K == 1280
      ? KernelId::WALL_VISION_ROW_WEIGHT_OUTER_W8A16_M352N80K160
      : KernelId::WALL_VISION_ROW_WEIGHT_OUTER_W8A16_M256N80K448;
  auto register_writer = wall_precision_registers(
      kernel_id, fp16, std::vector<GraphDdrRegisterOperandSpec>{
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::Int8ModelWeight,
               GraphDdrRegisterAccess::Read,
               GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
          {GraphDdrRegisterAbi{GraphDdrRegisterRole::PerChannelScale,
               GraphDdrRegisterAccess::Read,
               GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
  for (const at::Tensor* tensor : {&weight, &scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Wall Vision row weight-outer DDR operand must be aligned");
  }
  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(kernel != nullptr,
              "Wall Vision row weight-outer kernel missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel, rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, bias_spm_addr,
          static_cast<uint32_t>(M), static_cast<uint32_t>(N), local_k,
          1, NUM_CORES, local_k * 2, static_cast<uint32_t>(N * 2), 0, 0, 1});
  register_writer.write(*kernel);
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {16, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

// Direct Address SPM-to-SPM Linear Kernel (ACC16 - FP16 accumulation)
// =============================================================================
// Same dispatch contract as ACC32, using the generated FP16-accumulation
// auto-tile family selected from the complete shape table.
// =============================================================================

void rpu_launch_linear_spm_to_spm_acc16_kernel(
    uint32_t input_spm_addr,           // Input SPM address (core 0 base)
    const at::Tensor &weight,          // DDR weight [out_features, in_features]
    uint32_t output_spm_addr,          // Output SPM address (core 0 base)
    int64_t M,                         // batch * seq_len
    int64_t N,                         // out_features
    int64_t K,                         // in_features
    int partition,                     // 0=row, 1=col
    int num_cores,                     // number of cores for partition (attn_tp or tp)
    uint32_t bias_spm_addr,
    const at::Tensor &scale,           // quant scale (fp16 W8/INT4 or fp8 NVFP4)
    uint32_t nvfp4_tensor_scale_spm_addr,
    uint16_t nvfp4_layer_id,
    bool force_acc32,
    bool prefer_gemv,
    bool qkv_planar_direct,
    bool prefer_row_weight_reuse,
    const at::Tensor& negative_zero)
{
  validate_parallel_linear_geometry(M, N, K, partition, num_cores);
  const bool is_uint4 = negative_zero.defined() && negative_zero.numel() != 0;
  const bool packed_w4 = weight.scalar_type() == at::kByte &&
      weight.dim() == 2 && weight.size(0) == N && weight.size(1) == K / 2 &&
      scale.defined() && scale.scalar_type() == at::kHalf;
  TORCH_CHECK(!is_uint4 || packed_w4,
              "negative_zero is only valid for packed W4 with FP16 scale");
  if (prefer_row_weight_reuse) {
    TORCH_CHECK(
        rpu_pl_tiling::supports_w8a16_row_weight_reuse(
            M, N, K, partition, num_cores) &&
            !force_acc32 && !prefer_gemv && !qkv_planar_direct &&
            bias_spm_addr == 0 && nvfp4_tensor_scale_spm_addr == 0,
        "W8A16 row weight reuse requires measured M512/N1024/K2048 row TP8 "
        "ACC16 without bias, GEMV, planar QKV or NVFP4 tensor scale");
    TORCH_CHECK(
        weight.device().type() == c10::DeviceType::PrivateUse1 &&
            weight.scalar_type() == at::kChar && scale.defined() &&
            scale.device() == weight.device() &&
            scale.scalar_type() == at::kHalf,
        "W8A16 row weight reuse requires device signed-int8 weight and FP16 scale");
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "W8A16 row weight reuse requires initialized SPM");
    const uint64_t input_bytes = static_cast<uint64_t>(M) * (K / num_cores) * 2;
    const uint64_t output_bytes = static_cast<uint64_t>(M) * N * 2;
    const uint32_t spm_base = SPM_ALLOC.addr(0, 0);
    const auto check_spm_range = [&](uint32_t address, uint64_t bytes) {
      TORCH_CHECK(
          address >= spm_base && address % 256 == 0 &&
              bytes <= SpmAllocator::SPM_USABLE &&
              static_cast<uint64_t>(address - spm_base) <=
                  SpmAllocator::SPM_USABLE - bytes,
          "W8A16 row weight reuse requires aligned bounded local SPM ranges");
    };
    check_spm_range(input_spm_addr, input_bytes);
    check_spm_range(output_spm_addr, output_bytes);
    TORCH_CHECK(
        static_cast<uint64_t>(input_spm_addr) + input_bytes <= output_spm_addr ||
            static_cast<uint64_t>(output_spm_addr) + output_bytes <= input_spm_addr,
        "W8A16 row weight reuse input/output SPM ranges must not overlap");
    TORCH_CHECK(
        KernelCache::instance().has_loaded(
            KernelId::W8A16_ROW_WEIGHT_REUSE_M352N80K256),
        "Selected W8A16 row weight reuse payload is not loaded");
  }
  const bool register_census_active =
      RpuKernelGraph::active().kernel_register_census_active();
  // Full-byte raw uint8 [N,K] plus FP16 [N] scale is the Qwen3.5 E4M3
  // contract. Packed W4/NVFP4 remains [N,K/2] and is handled below.
  const bool fp8_e4m3_weight =
      weight.scalar_type() == at::kByte && weight.dim() == 2 &&
      weight.size(0) == N && weight.size(1) == K;
  // Frozen per-handle state is the only production authority. The retained
  // compatibility default is deterministic ACC16, never a process fallback.
  const bool use_acc32 = force_acc32;
  if (register_census_active) {
    TORCH_CHECK(weight.scalar_type() == at::kHalf ||
                    weight.scalar_type() == at::kChar ||
                    fp8_e4m3_weight || is_uint4,
                "typed DDR-register census admits FP16 Linear, signed-int8 "
                "W8A16, full-byte E4M3 or packed UINT4 weights");
  }

  if (force_acc32) {
    const bool fp16 = weight.scalar_type() == at::kHalf;
    const bool quantized = weight.scalar_type() == at::kChar ||
        weight.scalar_type() == at::kByte;
    TORCH_CHECK(fp16 || quantized,
                "per-handle ACC32 requires FP16, W8, W4, E4M3 or NVFP4 weights");
    TORCH_CHECK(!fp16 || ((!scale.defined() || scale.numel() == 0) &&
                            nvfp4_tensor_scale_spm_addr == 0),
                "FP16 ACC32 does not take quantization scales");
    // The family branches below validate exact packed layout, scale dtype,
    // extent and alignment before selecting their accumulator-specific table.
    TORCH_CHECK(!quantized || (scale.defined() && scale.numel() > 0),
                "quantized ACC32 requires its family scale payload");
  }

  TORCH_CHECK(!qkv_planar_direct ||
                  (!use_acc32 && !prefer_gemv && weight.scalar_type() == at::kChar &&
                   M == 192 && N == 4096 && K == 2048 && partition == 1 &&
                   num_cores == 8 && bias_spm_addr == 0),
              "Qwen3-VL planar QKV requires exact M192/N4096/K2048 ACC16 W8 "
              "TP8 column partition without bias or GEMV");

  // FP16/W8A16 only. Packed W4/NVFP4 and every caller that leaves the opt-in
  // false retain the generated auto-tile route.
  if (M == 1 && prefer_gemv && !use_acc32 &&
      (weight.scalar_type() == at::kHalf ||
       weight.scalar_type() == at::kChar)) {
    rpu_launch_gemv_spm_to_spm_kernel(
        input_spm_addr, weight, output_spm_addr,
        M, N, K, partition, num_cores, scale);
    add_m1_gemv_bias(
        bias_spm_addr, output_spm_addr, N, partition, num_cores);
    return;
  }

  if (use_acc32 && weight.scalar_type() == at::kHalf) {
    // FP16 ACC32 selects the accumulator-specific auto-tile table.
    rpu_launch_linear_spm_to_spm_kernel(
        input_spm_addr, weight, output_spm_addr,
        M, N, K, partition, num_cores, bias_spm_addr);
    return;
  }

  size_t dwidth = sizeof(c10::Half);

  // Calculate local dimensions based on partition type
  size_t local_k, local_n, local_m;
  local_m = M;

  if (partition == 0) {
    local_k = K / num_cores;
    local_n = N;
  } else {
    local_k = K;
    local_n = N / num_cores;
  }

  if (weight.scalar_type() == at::kChar || fp8_e4m3_weight) {
    const bool is_fp8_e4m3 = fp8_e4m3_weight;
    TORCH_CHECK(scale.defined() && scale.scalar_type() == at::kHalf,
                is_fp8_e4m3
                    ? "E4M3 prefill GEMM requires a defined fp16 scale tensor"
                    : "W8A16 prefill GEMM requires a defined fp16 scale tensor");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == N &&
                    weight.size(1) == K && weight.is_contiguous(),
                is_fp8_e4m3
                    ? "E4M3 prefill GEMM requires contiguous raw-uint8 weight "
                      "[N,K]"
                    : "W8A16 prefill GEMM requires contiguous signed-int8 weight "
                    "[N,K]=[", N, ",", K, "], got ", weight.sizes());
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == N &&
                    scale.is_contiguous(),
                is_fp8_e4m3
                    ? "E4M3 prefill GEMM requires contiguous per-channel fp16 "
                      "scale"
                    : "W8A16 prefill GEMM requires contiguous per-channel fp16 "
                    "scale [N]=[", N, "], got ", scale.sizes());
    TORCH_CHECK(weight.nbytes() ==
                    static_cast<size_t>(N) * static_cast<size_t>(K) &&
                    scale.nbytes() ==
                    static_cast<size_t>(N) * sizeof(c10::Half),
                is_fp8_e4m3
                    ? "E4M3 prefill GEMM weight/scale byte extent drifted"
                    : "W8A16 prefill GEMM weight/scale byte extent drifted");

    if (prefer_row_weight_reuse) {
      // The shared typed writer supplies DDR registers below. Check the full
      // tensor range before staging a census occurrence or mutating a kernel.
      constexpr uint64_t kDdrAddressLimit = UINT64_C(1) << 40;
      for (const at::Tensor* operand : {&weight, &scale}) {
        const uint64_t address = RpuGetDevAddr(operand->data_ptr());
        TORCH_CHECK(address % 256 == 0 && address < kDdrAddressLimit &&
                        operand->nbytes() <= kDdrAddressLimit - address,
                    "W8A16 row weight reuse DDR operand range is not encodable");
      }
    }

    uint32_t x_addr8 = input_spm_addr;
    uint32_t y_addr8 = output_spm_addr;
    uint32_t b_addr8 = bias_spm_addr;
    uint16_t has_bias8 = (bias_spm_addr != 0) ? 1 : 0;

    // Kernel + grid m/n-tiling come from the accumulator-specific rhino-ops
    // table. Fixed base kernels were retired; the current ref requires auto-tile.
    const auto tp8 = use_acc32
        ? rpu_pl_tiling::select_tile_acc32(
              static_cast<int>(local_m), static_cast<int>(local_n),
              static_cast<int>(local_k), /*is_fp16=*/false)
        : rpu_pl_tiling::select_tile_acc16(
              static_cast<int>(local_m), static_cast<int>(local_n),
              static_cast<int>(local_k), /*is_fp16=*/false);
    const KernelId kernel_id8 = prefer_row_weight_reuse
        ? KernelId::W8A16_ROW_WEIGHT_REUSE_M352N80K256
        : (qkv_planar_direct
            ? KernelId::QWEN3VL_PREFILL_QKV_PLANAR_DIRECT
            : (use_acc32
                ? autotile_linear_acc32_kernel_id(/*is_fp16=*/false, tp8.n_tile)
                : autotile_linear_kernel_id(
                      /*is_fp16=*/false, tp8.n_tile)));
    const uint16_t gx8 = static_cast<uint16_t>(CeilDiv(
        local_n, prefer_row_weight_reuse ? size_t{80}
            : (qkv_planar_direct ? size_t{64}
                                : static_cast<size_t>(tp8.n_tile))));
    // The reuse payload walks every M352 block internally while retaining W.
    // An ordinary grid.y would repeat that work and overwrite the same output.
    const uint16_t gy8 = prefer_row_weight_reuse ? uint16_t{1}
        : static_cast<uint16_t>(CeilDiv(
            local_m, qkv_planar_direct ? size_t{448}
                                       : static_cast<size_t>(tp8.m_tile)));

    GraphDdrRegisterAbi scale_abi8{
        GraphDdrRegisterRole::PerChannelScale,
        GraphDdrRegisterAccess::Read,
        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
        20, 21};
    if (qkv_planar_direct) {
      scale_abi8.reg_lo = 16;
      scale_abi8.reg_hi = 17;
    }
    auto register_writer8 =
        RpuKernelGraph::active().stage_kernel_ddr_registers(
            kernel_id8,
            std::vector<GraphDdrRegisterOperandSpec>{
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        is_fp8_e4m3
                            ? GraphDdrRegisterRole::Fp8ModelWeight
                            : GraphDdrRegisterRole::Int8ModelWeight,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        2, 3},
                    weight},
                GraphDdrRegisterOperandSpec{scale_abi8, scale}});

    void *wp8 = weight.data_ptr();
    c10::Half *sp = scale.data_ptr<c10::Half>();
    rpu_ddr_flush(wp8);
    rpu_ddr_flush(sp);

    Kernel_t* kernel8 = GET_KERNEL(kernel_id8);
    TORCH_CHECK(kernel8 != nullptr,
                "autotile: failed to get ",
                is_fp8_e4m3 ? "E4M3 " : "W8A16 ",
                use_acc32 ? "ACC32" : "ACC16", " tile m", tp8.m_tile,
                "n", tp8.n_tile, "k128");
    kernel8->reset_regs();
    if (qkv_planar_direct) {
      // The specialized planar-output expansion kernel retains its
      // generated 0..19 register ABI; ordinary Linear continues with the
      // current widened set_parallel_linear_regs() ABI.
      rpu_pl_tiling::set_u32_reg_pair(*kernel8, 0, x_addr8);
      rpu_pl_tiling::set_u32_reg_pair(*kernel8, 4, y_addr8);
      rpu_pl_tiling::set_u32_reg_pair(*kernel8, 6, b_addr8);
      kernel8->set_regs(8, static_cast<uint16_t>(local_m));
      kernel8->set_regs(9, static_cast<uint16_t>(local_n));
      kernel8->set_regs(11, static_cast<uint16_t>(local_k / 16));
      kernel8->set_regs(12, has_bias8);
      kernel8->set_regs(13, static_cast<uint16_t>(num_cores));
      kernel8->set_regs(14, static_cast<uint16_t>(local_k * dwidth));
      kernel8->set_regs(18, static_cast<uint16_t>(partition));
      kernel8->set_regs(19, static_cast<uint16_t>(1));
    } else {
      rpu_pl_tiling::set_parallel_linear_regs(
          *kernel8,
          rpu_pl_tiling::ParallelLinearRegisterArgs{
              x_addr8, 0, y_addr8, b_addr8,
              static_cast<uint32_t>(local_m), static_cast<uint32_t>(local_n),
              static_cast<uint32_t>(local_k), has_bias8,
              static_cast<uint16_t>(num_cores),
              static_cast<uint32_t>(local_k * dwidth),
              static_cast<uint32_t>(local_n * dwidth),
              0, static_cast<uint16_t>(partition),
              static_cast<uint16_t>(is_fp8_e4m3 ? 5 : 1)});
    }
    register_writer8.write(*kernel8);

    auto* wq8 = GET_QUEUE(num_cores);
    wq8->set_broadcast_mode(true);
    std::vector<uint8_t> core_list8;
    for (int i = 0; i < num_cores; ++i) core_list8.push_back(i);
    wq8->enqueu_kernel(*kernel8, {gx8, gy8, (uint16_t)1}, core_list8);
    return;
  }

  if (weight.scalar_type() == at::kByte) {
    // Packed INT4 uses the generated pgrp family. The logical quantization may
    // still be per-channel, represented by repeating one channel scale across
    // all K groups before applying the controller-striped pgrp layout.
    TORCH_CHECK(scale.defined(),
                "4-bit prefill GEMM requires a defined scale tensor");
    TORCH_CHECK(
        partition != 0 || bias_spm_addr == 0,
        "packed int4: row-partition Linear cannot add bias before the "
        "cross-core all-reduce; add bias after reduction instead");
    if (scale.scalar_type() == at::kByte) {
      TORCH_CHECK(nvfp4_tensor_scale_spm_addr != 0,
                  "NVFP4A16 requires the FP32 tensor-scale SPM address");
      TORCH_CHECK(weight.dim() == 2 && weight.size(0) == N && weight.size(1) == K / 2,
                  "NVFP4A16 packed weight must be [N,K/2]=[", N, ",", K / 2,
                  "], got ", weight.sizes());
      TORCH_CHECK(K % 16 == 0 && scale.dim() == 2 &&
                      scale.size(0) == K / 16 && scale.size(1) == N,
                  "NVFP4A16 striped fp8 block scale metadata must be [K/16,N]=[", K / 16, ",", N,
                  "], got ", scale.sizes());
      TORCH_CHECK(weight.is_contiguous() && scale.is_contiguous(),
                  "NVFP4A16 packed weight and fp8 block scale must be contiguous");
      TORCH_CHECK(local_n % 16 == 0 && local_k % 64 == 0 &&
                      local_m <= UINT16_MAX && local_k <= UINT16_MAX,
                  "NVFP4A16 requires local_n%16==0, local_k%64==0 and "
                  "uint16 local_m/local_k, got ", local_m, ",", local_n, ",", local_k);
      TORCH_CHECK(nvfp4_tensor_scale_spm_addr % 32 == 0,
                  "NVFP4A16 tensor-scale SPM base must be 32-byte aligned");

      auto* wp = weight.data_ptr<uint8_t>();
      auto* sp = scale.data_ptr<uint8_t>();
      rpu_ddr_flush(wp);
      rpu_ddr_flush(sp);
      const uint64_t w_addr = RpuGetDevAddr(wp);
      const uint64_t s_addr = RpuGetDevAddr(sp);
      TORCH_CHECK(w_addr % 256 == 0, "NVFP4A16 weight DDR not 256-byte aligned");
      TORCH_CHECK(s_addr % 256 == 0, "NVFP4A16 scale DDR not 256-byte aligned");

      const auto nvfp4_tile = use_acc32
          ? rpu_pl_tiling::select_tile_nvfp4_acc32(local_m, local_n, local_k)
          : rpu_pl_tiling::select_tile_nvfp4_acc16(local_m, local_n, local_k);
      const KernelId nvfp4_kernel_id = use_acc32
          ? autotile_nvfp4_acc32_kernel_id(nvfp4_tile.n_tile)
          : autotile_nvfp4_acc16_kernel_id(nvfp4_tile.n_tile);
      validate_parallel_linear_grid(
          local_m, local_n, nvfp4_tile.m_tile, nvfp4_tile.n_tile);
      auto* kernel = GET_KERNEL(nvfp4_kernel_id);
      TORCH_CHECK(kernel != nullptr,
                  (use_acc32 ? rpu_pl_tiling::autotile_kernel_name_nvfp4_acc32(
                                   nvfp4_tile.m_tile, nvfp4_tile.n_tile)
                             : rpu_pl_tiling::autotile_kernel_name_nvfp4_acc16(
                                   nvfp4_tile.m_tile, nvfp4_tile.n_tile)),
                  " missing from the selected operator ref");

      kernel->reset_regs();
      // Physical FP8 scales: [localK/32, cores, localN/16, 2, 16].
      const uint64_t scale_kpair_step =
          static_cast<uint64_t>(num_cores) * local_n * 2;
      TORCH_CHECK(scale_kpair_step <= UINT32_MAX,
                  "NVFP4 scale K-pair stride exceeds the 32-bit register ABI");
      rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
          {input_spm_addr, w_addr, output_spm_addr, bias_spm_addr,
           static_cast<uint32_t>(local_m), static_cast<uint32_t>(local_n),
           static_cast<uint32_t>(local_k),
           static_cast<uint16_t>(bias_spm_addr != 0),
           static_cast<uint16_t>(num_cores),
           static_cast<uint32_t>(local_k * dwidth),
           static_cast<uint32_t>(local_n * dwidth), s_addr,
           static_cast<uint16_t>(partition == 1), 0},
          static_cast<uint32_t>(scale_kpair_step),
          nvfp4_tensor_scale_spm_addr, nvfp4_layer_id});
      const uint16_t gx = (uint16_t)CeilDiv(
          local_n, static_cast<size_t>(nvfp4_tile.n_tile));
      const uint16_t gy = (uint16_t)CeilDiv(
          local_m, static_cast<size_t>(nvfp4_tile.m_tile));
      kernel->set_regs(64, gx);
      kernel->set_regs(65, gy);
      kernel->set_regs(66, (uint16_t)1);

      auto* queue = GET_QUEUE(num_cores);
      queue->set_broadcast_mode(true);
      std::vector<uint8_t> core_list;
      for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
      queue->enqueu_kernel(*kernel, {gx, gy, (uint16_t)1}, core_list);
      return;
    }
    TORCH_CHECK(scale.scalar_type() == at::kHalf,
                "wINT4 prefill GEMM requires fp16 scale, got ",
                scale.scalar_type());
    // The physical scale payload is controller-striped
    // [ceil(localG/4),ceil(localN/64),cores,4,64]; its 2D view carries group_size
    // in dim 0 for this dispatcher. The 63b68ace ABI extends common regs 0..23
    // with {24:groupSize,25:localG,26-27:SGBlockByteStep}.
    //
    // Registers, weight packing, scale layout and fallback tiles match
    // ParallelLinearInt4PgrpGenerator at 63b68ace. The conformance matrix
    // includes 8-core col/row and Wall TP2 col/row projections.
    {
      TORCH_CHECK(scale.dim() == 2,
                  "pgrp int4 requires a controller-striped 2D scale payload; got ",
                  scale.sizes());
      const int64_t k_bs = scale.size(0);
      TORCH_CHECK(k_bs == 32 || k_bs == 64 || k_bs == 128,
                  "pgrp int4: scale dim 0 must carry group_size 32/64/128, got ",
                  k_bs, " for shape ", scale.sizes());
      TORCH_CHECK(local_k % static_cast<size_t>(k_bs) == 0,
                  "pgrp int4: localK(", local_k,
                  ") not divisible by group_size(", k_bs, ")");
      const size_t local_g = local_k / static_cast<size_t>(k_bs);
      const size_t scale_group_blocks = CeilDiv(local_g, size_t{4});
      const size_t scale_n_blocks = CeilDiv(local_n, size_t{64});
      constexpr size_t kScaleStripeElements = 4 * 64;
      constexpr size_t kScaleStripeBytes =
          kScaleStripeElements * sizeof(c10::Half);
      const size_t expected_scale_elements =
          scale_group_blocks * scale_n_blocks *
          static_cast<size_t>(num_cores) * kScaleStripeElements;
      TORCH_CHECK(scale.is_contiguous() &&
                      static_cast<size_t>(scale.numel()) == expected_scale_elements,
                  "pgrp int4: controller-striped scale payload has ",
                  scale.numel(), " elements, expected ", expected_scale_elements,
                  " for localG=", local_g, " localN=", local_n,
                  " cores=", num_cores);
      TORCH_CHECK(weight.numel() == N * K / 2, "pgrp int4: packed weight numel ",
                  weight.numel(), " != N*K/2=", N * K / 2);

      // UINT4 alone uses the shared main payload and its typed metadata ABI.
      if (is_uint4) {
        TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1 &&
                        weight.is_contiguous() && scale.device() == weight.device(),
                    "pgrp W4 requires contiguous device weight and scale");
        TORCH_CHECK(local_m <= UINT16_MAX && local_k <= UINT16_MAX &&
                        local_k % 64 == 0 && local_n % 16 == 0,
                    "pgrp W4 requires uint16 localM/localK, K%64 and N%16");
        {
          TORCH_CHECK(negative_zero.device() == weight.device() &&
                          negative_zero.scalar_type() == at::kHalf &&
                          negative_zero.is_contiguous() &&
                          negative_zero.sizes() == scale.sizes(),
                      "pgrp UINT4 negative_zero must match the contiguous device FP16 scale payload");
        }
        const size_t scale_alignment =
            static_cast<size_t>(num_cores) * kScaleStripeBytes;

        const auto tp = use_acc32
            ? rpu_pl_tiling::select_tile_int4_acc32(
                  static_cast<int>(local_m), static_cast<int>(local_n),
                  static_cast<int>(local_k))
            : rpu_pl_tiling::select_tile_int4(
                  static_cast<int>(local_m), static_cast<int>(local_n),
                  static_cast<int>(local_k));
        const KernelId pgrp_kernel_id = use_acc32
            ? autotile_uint4_pgrp_acc32_kernel_id(tp.n_tile)
            : autotile_uint4_pgrp_kernel_id(tp.n_tile);
        std::vector<GraphDdrRegisterOperandSpec> operands{
            {{GraphDdrRegisterRole::PackedUint4ModelWeight,
              GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
            {{GraphDdrRegisterRole::PerGroupScale, GraphDdrRegisterAccess::Read,
              GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}};
        {
          operands.push_back({
              {GraphDdrRegisterRole::NegativeZeroPoint, GraphDdrRegisterAccess::Read,
               GraphDdrRegisterEncoding::DevAddrShift8LoHi, 22, 23}, negative_zero});
        }
        auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
            pgrp_kernel_id, std::move(operands));
        for (const auto* metadata : {&scale, &negative_zero}) {
          if (!metadata->defined() || metadata->numel() == 0) continue;
          TORCH_CHECK(RpuGetDevAddr(metadata->data_ptr()) % scale_alignment == 0,
                      "pgrp W4 scale/negative_zero DDR must be aligned to cores*512=",
                      scale_alignment, " bytes");
          rpu_ddr_flush(metadata->data_ptr());
        }
        TORCH_CHECK(RpuGetDevAddr(weight.data_ptr()) % 256 == 0,
                    "pgrp UINT4: weight DDR not 256-byte aligned");
        rpu_ddr_flush(weight.data_ptr());
        const size_t sgblock_byte_step_size =
            scale_n_blocks * static_cast<size_t>(num_cores) * kScaleStripeBytes;
        TORCH_CHECK(sgblock_byte_step_size <= UINT32_MAX,
                    "pgrp int4: SGBlockByteStep exceeds uint32: ",
                    sgblock_byte_step_size);
        const uint32_t sgblock_byte_step =
            static_cast<uint32_t>(sgblock_byte_step_size);
        Kernel_t *kpg = GET_KERNEL(pgrp_kernel_id);
        TORCH_CHECK(kpg != nullptr, "pgrp int4: failed to get ",
                    use_acc32 ? "ACC32" : "ACC16", " tile m", tp.m_tile,
                    "n", tp.n_tile, "k128");
        kpg->reset_regs();
        rpu_pl_tiling::set_parallel_linear_w4_group_regs(
            *kpg,
            rpu_pl_tiling::ParallelLinearW4GroupRegisterArgs{
                rpu_pl_tiling::ParallelLinearRegisterArgs{
                    input_spm_addr, 0, output_spm_addr, bias_spm_addr,
                    static_cast<uint32_t>(local_m),
                    static_cast<uint32_t>(local_n),
                    static_cast<uint32_t>(local_k),
                    static_cast<uint16_t>(bias_spm_addr != 0),
                    static_cast<uint16_t>(num_cores),
                    static_cast<uint32_t>(local_k * dwidth),
                    static_cast<uint32_t>(local_n * dwidth), 0,
                    static_cast<uint16_t>(partition), 1},
                static_cast<uint16_t>(k_bs),
                static_cast<uint16_t>(local_g),
                sgblock_byte_step, 0, true});
        uint16_t gxg = (uint16_t)CeilDiv(local_n, (size_t)tp.n_tile);
        uint16_t gyg = (uint16_t)CeilDiv(local_m, (size_t)tp.m_tile);
        kpg->set_regs(64, gxg);
        kpg->set_regs(65, gyg);
        kpg->set_regs(66, (uint16_t)1);
        register_writer.write(*kpg);

        auto *wqg = GET_QUEUE(num_cores);
        wqg->set_broadcast_mode(true);
        std::vector<uint8_t> core_list_g;
        for (int i = 0; i < num_cores; ++i) core_list_g.push_back(i);
        wqg->enqueu_kernel(*kpg, {gxg, gyg, (uint16_t)1}, core_list_g);
        return;
      }

      uint8_t *wpg = weight.data_ptr<uint8_t>();
      c10::Half *spg = scale.data_ptr<c10::Half>();
      rpu_ddr_flush(wpg);
      rpu_ddr_flush(spg);
      uint64_t w_addr_g = RpuGetDevAddr(wpg);
      uint64_t s_addr_g = RpuGetDevAddr(spg);
      TORCH_CHECK(w_addr_g % 256 == 0, "pgrp int4: weight DDR not 256-byte aligned");
      const size_t scale_alignment =
          static_cast<size_t>(num_cores) * kScaleStripeBytes;
      TORCH_CHECK(s_addr_g % scale_alignment == 0,
                  "pgrp int4: scale DDR address must be aligned to cores*512=",
                  scale_alignment, " bytes, got 0x", std::hex, s_addr_g,
                  std::dec);

      const auto tp = use_acc32
          ? rpu_pl_tiling::select_tile_int4_acc32(
                static_cast<int>(local_m), static_cast<int>(local_n),
                static_cast<int>(local_k))
          : rpu_pl_tiling::select_tile_int4(
                static_cast<int>(local_m), static_cast<int>(local_n),
                static_cast<int>(local_k));
      // GET_KERNEL(id) routes through the graph-aware get_kernel_reset(id) so the pgrp
      // variant is GRAPH-RECORDABLE (sets pending_kernel_id_); the by-name get_kernel()
      // keep the graph-aware lookup for replay recording.
      // Same fix as the acc16 variants.
      const KernelId pgrp_kernel_id = use_acc32
          ? autotile_pgrp_acc32_kernel_id(tp.n_tile)
          : autotile_pgrp_kernel_id(tp.n_tile);
      Kernel_t *kpg = GET_KERNEL(pgrp_kernel_id);
      TORCH_CHECK(kpg != nullptr, "pgrp int4: failed to get ",
                  use_acc32 ? "ACC32" : "ACC16", " tile m", tp.m_tile,
                  "n", tp.n_tile, "k128");

      const size_t sgblock_byte_step_size =
          scale_n_blocks * static_cast<size_t>(num_cores) * kScaleStripeBytes;
      TORCH_CHECK(sgblock_byte_step_size <= UINT32_MAX,
                  "pgrp int4: SGBlockByteStep exceeds uint32: ",
                  sgblock_byte_step_size);
      const uint32_t sgblock_byte_step =
          static_cast<uint32_t>(sgblock_byte_step_size);
      // Bias uses per-core SPM byte addresses in regs 6/7,
       // in the same address space as input/output regs 0/1 and 4/5.
       // Scale uses regs 20/21 and bias uses reg14. Only col-partition +bias is
      // used by the int4 qkv/gate/up path; o/down are bias-free.
      // Row+bias stays out-of-envelope (needs core-0-only bias staging, unvalidated).
      kpg->reset_regs();
      rpu_pl_tiling::set_parallel_linear_int4_group_regs(
          *kpg,
          rpu_pl_tiling::ParallelLinearInt4GroupRegisterArgs{
              rpu_pl_tiling::ParallelLinearRegisterArgs{
                  input_spm_addr, w_addr_g, output_spm_addr, bias_spm_addr,
                  static_cast<uint32_t>(local_m),
                  static_cast<uint32_t>(local_n),
                  static_cast<uint32_t>(local_k),
                  static_cast<uint16_t>(bias_spm_addr != 0),
                  static_cast<uint16_t>(num_cores),
                  static_cast<uint32_t>(local_k * dwidth),
                  static_cast<uint32_t>(local_n * dwidth), s_addr_g,
                  static_cast<uint16_t>(partition), 1},
              static_cast<uint16_t>(k_bs),
              static_cast<uint16_t>(local_g),
              sgblock_byte_step});
      uint16_t gxg = (uint16_t)CeilDiv(local_n, (size_t)tp.n_tile);
      uint16_t gyg = (uint16_t)CeilDiv(local_m, (size_t)tp.m_tile);
      kpg->set_regs(64, gxg);
      kpg->set_regs(65, gyg);
      kpg->set_regs(66, (uint16_t)1);

      auto *wqg = GET_QUEUE(num_cores);
      wqg->set_broadcast_mode(true);
      std::vector<uint8_t> core_list_g;
      for (int i = 0; i < num_cores; ++i) core_list_g.push_back(i);
      wqg->enqueu_kernel(*kpg, {gxg, gyg, (uint16_t)1}, core_list_g);
      return;
    }
  }

  // Weight in DDR
  c10::Half *wp = weight.data_ptr<c10::Half>();
  const auto tp = rpu_pl_tiling::select_tile_acc16(
      static_cast<int>(local_m), static_cast<int>(local_n),
      static_cast<int>(local_k), /*is_fp16=*/true);
  const size_t at_n = static_cast<size_t>(tp.n_tile);
  const size_t at_m = static_cast<size_t>(tp.m_tile);
  validate_parallel_linear_grid(local_m, local_n, at_m, at_n);
  std::vector<uint16_t> grid_dims = {
    (uint16_t)CeilDiv(local_n, at_n),
    (uint16_t)CeilDiv(local_m, at_m),
    (uint16_t)1
  };

  const KernelId kernel_id =
      autotile_linear_kernel_id(/*is_fp16=*/true, tp.n_tile);

  auto register_writer =
      RpuKernelGraph::active().stage_kernel_ddr_registers(
          kernel_id,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                      2, 3},
                  weight}});
  rpu_ddr_flush(wp);

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(kernel != nullptr, "autotile: failed to get FP16 ACC16 tile m",
              at_m, "n", at_n, "k128");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, bias_spm_addr,
          static_cast<uint32_t>(local_m), static_cast<uint32_t>(local_n),
          static_cast<uint32_t>(local_k),
          static_cast<uint16_t>(bias_spm_addr != 0),
          static_cast<uint16_t>(num_cores),
          static_cast<uint32_t>(local_k * dwidth),
          static_cast<uint32_t>(local_n * dwidth),
          0, static_cast<uint16_t>(partition), 2});
  register_writer.write(*kernel);

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
  wq->enqueu_kernel(*kernel, grid_dims, core_list);
}

// Exact Vision FC1 + BASE tanh-GELU. Preserve each current ACC16 auto tile,
// FP16 bias retirement, unchanged packed owners and typed DDR register ABI.
void rpu_launch_lingbot2_vision_fc1_gelu_spm_kernel(
    uint32_t input_spm_addr, const at::Tensor& weight,
    uint32_t output_spm_addr, uint32_t bias_spm_addr,
    const at::Tensor& scale) {
  TORCH_CHECK(weight.defined() &&
                  (weight.scalar_type() == at::kHalf ||
                   weight.scalar_type() == at::kChar),
              "Lingbot2 Vision FC1 GELU requires FP16 or signed INT8 weights");
  const bool fp16 = weight.scalar_type() == at::kHalf;
  const auto owned = [](const at::Tensor& tensor) {
    if (!(tensor.defined() &&
        tensor.device().type() == c10::DeviceType::PrivateUse1 &&
        tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
        tensor.storage_offset() >= 0)) return false;
    const uint64_t storage_bytes = tensor.storage().nbytes();
    const uint64_t item_bytes = tensor.element_size();
    const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset());
    return item_bytes != 0 && offset <= storage_bytes / item_bytes &&
        tensor.nbytes() <= storage_bytes - offset * item_bytes;
  };
  const auto ddr_ok = [](const at::Tensor& tensor) {
    const uint64_t address = RpuGetDevAddr(tensor.data_ptr());
    constexpr uint64_t limit = uint64_t{1} << 40;
    return address != 0 && (address & 255u) == 0 && address < limit &&
        tensor.nbytes() <= limit - address;
  };
  TORCH_CHECK(owned(weight) && weight.dim() == 2 &&
                  weight.size(0) == 4096 && weight.size(1) == 1024,
              "Lingbot2 Vision FC1 GELU requires a complete contiguous [4096,1024] weight owner");
  TORCH_CHECK(ddr_ok(weight),
              "Lingbot2 Vision FC1 weight violates complete aligned shift8 DDR addressing");
  if (fp16) {
    TORCH_CHECK(!scale.defined(),
                "Lingbot2 FP16 Vision FC1 GELU cannot carry quantization scales");
  } else {
    TORCH_CHECK(owned(scale) && scale.scalar_type() == at::kHalf &&
                    scale.device() == weight.device() && scale.dim() == 1 &&
                    scale.size(0) == 4096,
                "Lingbot2 W8 Vision FC1 GELU requires a complete same-device FP16 [4096] scale owner");
    TORCH_CHECK(ddr_ok(scale),
                "Lingbot2 Vision FC1 scale violates complete aligned shift8 DDR addressing");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Lingbot2 Vision FC1 GELU requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 3> ranges{{
      {input_spm_addr, 768 * 1024 * 2},
      {output_spm_addr, 768 * 512 * 2}, {bias_spm_addr, 512 * 2}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && (address & 31u) == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                    uint64_t{address} + bytes <= (uint64_t{1} << 32),
                "Lingbot2 Vision FC1 GELU requires complete aligned absolute SPM intervals");
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(uint64_t{address} + bytes <= ranges[j].first ||
                      uint64_t{ranges[j].first} + ranges[j].second <= address,
                  "Lingbot2 Vision FC1 GELU input, output and bias must be disjoint");
    }
  }
  auto* queue = GET_QUEUE(8);
  TORCH_CHECK(queue != nullptr, "Lingbot2 Vision FC1 GELU eight-core queue is absent");
  const KernelId id = fp16 ? KernelId::LINGBOT2_VISION_FC1_GELU_FP16
                           : KernelId::LINGBOT2_VISION_FC1_GELU_W8;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Lingbot2 Vision FC1 GELU optional precision payload is absent");
  std::vector<GraphDdrRegisterOperandSpec> operands{{
      {fp16 ? GraphDdrRegisterRole::ModelWeight : GraphDdrRegisterRole::Int8ModelWeight,
       GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight}};
  if (!fp16) {
    operands.push_back({{GraphDdrRegisterRole::PerChannelScale,
        GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
        20, 21}, scale});
  }
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id, std::move(operands));
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "Lingbot2 Vision FC1 GELU kernel is absent");
  graph.keep_alive(weight);
  rpu_ddr_flush(weight.data_ptr());
  if (!fp16) {
    graph.keep_alive(scale);
    rpu_ddr_flush(scale.data_ptr());
  }
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, bias_spm_addr,
          768, 512, 1024, 1, 8, 2048, 1024, 0, 1,
          static_cast<uint16_t>(fp16 ? 2 : 1)});
  writer.write(*kernel);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,
      {static_cast<uint16_t>(fp16 ? 6 : 7), 2, 1},
      {0, 1, 2, 3, 4, 5, 6, 7});
}

// Exact Action T51/H768/Q4096/KV1024/TP8 QKV dispatch. Each branch retains
// the current auto ACC16 tile, independent swizzled owners and bias order.
void rpu_launch_lingbot2_qkv_multibase_t51_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    uint32_t q_output_spm_addr, uint32_t k_output_spm_addr,
    uint32_t v_output_spm_addr,
    uint32_t q_bias_spm_addr, uint32_t k_bias_spm_addr,
    uint32_t v_bias_spm_addr,
    const at::Tensor& q_scale, const at::Tensor& k_scale,
    const at::Tensor& v_scale) {
  TORCH_CHECK(q_weight.defined() &&
                  (q_weight.scalar_type() == at::kHalf ||
                   q_weight.scalar_type() == at::kChar),
              "Lingbot2 QKV requires uniform FP16 or signed INT8 weights");
  const bool fp16 = q_weight.scalar_type() == at::kHalf;
  const std::array<const at::Tensor*, 3> weights{{&q_weight, &k_weight, &v_weight}};
  const std::array<const at::Tensor*, 3> scales{{&q_scale, &k_scale, &v_scale}};
  constexpr std::array<int64_t, 3> channels{{4096, 1024, 1024}};
  const auto owned = [](const at::Tensor& tensor) {
    if (!(tensor.defined() &&
        tensor.device().type() == c10::DeviceType::PrivateUse1 &&
        tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
        tensor.storage_offset() >= 0)) return false;
    const uint64_t storage_bytes = tensor.storage().nbytes();
    const uint64_t item_bytes = tensor.element_size();
    const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset());
    return item_bytes != 0 && offset <= storage_bytes / item_bytes &&
        tensor.nbytes() <= storage_bytes - offset * item_bytes;
  };
  const auto ddr_ok = [](const at::Tensor& tensor) {
    const uint64_t address = RpuGetDevAddr(tensor.data_ptr());
    constexpr uint64_t limit = uint64_t{1} << 40;
    return address != 0 && (address & 255u) == 0 && address < limit &&
        tensor.nbytes() <= limit - address;
  };
  for (size_t i = 0; i < weights.size(); ++i) {
    const auto& weight = *weights[i];
    TORCH_CHECK(owned(weight) && weight.scalar_type() == q_weight.scalar_type() &&
                    weight.device() == q_weight.device() && weight.dim() == 2 &&
                    weight.size(0) == channels[i] && weight.size(1) == 768,
                "Lingbot2 QKV requires complete contiguous same-device weight owners [N,768]");
    TORCH_CHECK(ddr_ok(weight),
                "Lingbot2 QKV weight violates aligned complete shift8 DDR addressing");
    const auto& scale = *scales[i];
    if (fp16) {
      TORCH_CHECK(!scale.defined(), "Lingbot2 FP16 QKV cannot carry quantization scales");
    } else {
      TORCH_CHECK(owned(scale) && scale.scalar_type() == at::kHalf &&
                      scale.device() == q_weight.device() && scale.dim() == 1 &&
                      scale.size(0) == channels[i],
                  "Lingbot2 W8 QKV requires complete same-device FP16 per-channel scale owners");
      TORCH_CHECK(ddr_ok(scale),
                  "Lingbot2 QKV scale violates aligned complete shift8 DDR addressing");
    }
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Lingbot2 QKV requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 7> ranges{{
      {input_spm_addr, 51 * 768 * 2},
      {q_output_spm_addr, 51 * 512 * 2},
      {k_output_spm_addr, 51 * 128 * 2},
      {v_output_spm_addr, 51 * 128 * 2},
      {q_bias_spm_addr, 512 * 2}, {k_bias_spm_addr, 128 * 2},
      {v_bias_spm_addr, 128 * 2}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && (address & 31u) == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                    uint64_t{address} + bytes <= (uint64_t{1} << 32),
                "Lingbot2 QKV requires complete aligned absolute SPM intervals");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t{address} + bytes <= ranges[j].first ||
                      uint64_t{ranges[j].first} + ranges[j].second <= address,
                  "Lingbot2 QKV input, outputs and biases must be disjoint");
  }
  auto* queue = GET_QUEUE(8);
  TORCH_CHECK(queue != nullptr, "Lingbot2 QKV eight-core queue is absent");
  const KernelId id = fp16 ? KernelId::LINGBOT2_QKV_MULTIBASE_FP16_T51
                           : KernelId::LINGBOT2_QKV_MULTIBASE_W8_T51;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Lingbot2 QKV optional precision payload is absent");
  constexpr std::array<uint16_t, 3> weight_regs{{2, 24, 32}};
  constexpr std::array<uint16_t, 3> scale_regs{{20, 26, 34}};
  std::vector<GraphDdrRegisterOperandSpec> operands;
  for (size_t i = 0; i < weights.size(); ++i) {
    operands.push_back({{fp16 ? GraphDdrRegisterRole::ModelWeight
                             : GraphDdrRegisterRole::Int8ModelWeight,
        GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
        weight_regs[i], static_cast<uint16_t>(weight_regs[i] + 1)}, *weights[i]});
    if (!fp16)
      operands.push_back({{GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
          scale_regs[i], static_cast<uint16_t>(scale_regs[i] + 1)}, *scales[i]});
  }
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id, std::move(operands));
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "Lingbot2 QKV optional kernel is absent");
  for (size_t i = 0; i < weights.size(); ++i) {
    graph.keep_alive(*weights[i]);
    rpu_ddr_flush(weights[i]->data_ptr());
    if (!fp16) {
      graph.keep_alive(*scales[i]);
      rpu_ddr_flush(scales[i]->data_ptr());
    }
  }
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, q_output_spm_addr, q_bias_spm_addr,
          51, 512, 768, 1, 8, 1536, 1024, 0, 1,
          static_cast<uint16_t>(fp16 ? 2 : 1)});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 28, k_bias_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 30, k_output_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, v_bias_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 38, v_output_spm_addr);
  writer.write(*kernel);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,
      {static_cast<uint16_t>(fp16 ? 10 : 17), 1, 1}, {0,1,2,3,4,5,6,7});
}

// Exact no-bias M224/H2560/Q4096/KV1024/TP8 language prefill.
// Only dispatch is combined; both original K128/ACC16 bodies are retained.
void rpu_launch_lingbot2_prefill_qkv_m224_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    uint32_t q_output_spm_addr, uint32_t k_output_spm_addr,
    uint32_t v_output_spm_addr,
    const at::Tensor& q_scale, const at::Tensor& k_scale,
    const at::Tensor& v_scale) {
  TORCH_CHECK(q_weight.defined() &&
                  (q_weight.scalar_type() == at::kHalf ||
                   q_weight.scalar_type() == at::kChar),
              "Lingbot2 prefill QKV requires uniform FP16 or signed INT8 weights");
  const bool fp16 = q_weight.scalar_type() == at::kHalf;
  const std::array<const at::Tensor*, 3> weights{{&q_weight, &k_weight, &v_weight}};
  const std::array<const at::Tensor*, 3> scales{{&q_scale, &k_scale, &v_scale}};
  constexpr std::array<int64_t, 3> channels{{4096, 1024, 1024}};
  const auto owned = [](const at::Tensor& tensor) {
    if (!(tensor.defined() &&
        tensor.device().type() == c10::DeviceType::PrivateUse1 &&
        tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
        tensor.storage_offset() >= 0)) return false;
    const uint64_t storage_bytes = tensor.storage().nbytes();
    const uint64_t item_bytes = tensor.element_size();
    const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset());
    return item_bytes != 0 && offset <= storage_bytes / item_bytes &&
        tensor.nbytes() <= storage_bytes - offset * item_bytes;
  };
  const auto ddr_ok = [](const at::Tensor& tensor) {
    const uint64_t address = RpuGetDevAddr(tensor.data_ptr());
    constexpr uint64_t limit = uint64_t{1} << 40;
    return address != 0 && (address & 255u) == 0 && address < limit &&
        tensor.nbytes() <= limit - address;
  };
  for (size_t i = 0; i < weights.size(); ++i) {
    const auto& weight = *weights[i];
    TORCH_CHECK(owned(weight) && weight.scalar_type() == q_weight.scalar_type() &&
                    weight.device() == q_weight.device() && weight.dim() == 2 &&
                    weight.size(0) == channels[i] && weight.size(1) == 2560,
                "Lingbot2 prefill QKV requires complete contiguous same-device weight owners [N,2560]");
    TORCH_CHECK(ddr_ok(weight),
                "Lingbot2 prefill QKV weight violates aligned complete shift8 DDR addressing");
    const auto& scale = *scales[i];
    if (fp16) {
      TORCH_CHECK(!scale.defined(), "Lingbot2 FP16 prefill QKV cannot carry quantization scales");
    } else {
      TORCH_CHECK(owned(scale) && scale.scalar_type() == at::kHalf &&
                      scale.device() == q_weight.device() && scale.dim() == 1 &&
                      scale.size(0) == channels[i],
                  "Lingbot2 W8 prefill QKV requires complete same-device FP16 per-channel scale owners");
      TORCH_CHECK(ddr_ok(scale),
                  "Lingbot2 prefill QKV scale violates aligned complete shift8 DDR addressing");
    }
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Lingbot2 prefill QKV requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 4> ranges{{
      {input_spm_addr, 224 * 2560 * 2},
      {q_output_spm_addr, 224 * 512 * 2},
      {k_output_spm_addr, 224 * 128 * 2},
      {v_output_spm_addr, 224 * 128 * 2}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && (address & 31u) == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                    uint64_t{address} + bytes <= (uint64_t{1} << 32),
                "Lingbot2 prefill QKV requires complete aligned absolute SPM intervals");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t{address} + bytes <= ranges[j].first ||
                      uint64_t{ranges[j].first} + ranges[j].second <= address,
                  "Lingbot2 prefill QKV input and outputs must be disjoint");
  }
  auto* queue = GET_QUEUE(8);
  TORCH_CHECK(queue != nullptr, "Lingbot2 prefill QKV eight-core queue is absent");
  const KernelId id = fp16 ? KernelId::LINGBOT2_PREFILL_QKV_MULTIBASE_FP16_M224
                           : KernelId::LINGBOT2_PREFILL_QKV_MULTIBASE_W8_M224;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Lingbot2 prefill QKV optional precision payload is absent");
  constexpr std::array<uint16_t, 3> weight_regs{{2, 24, 32}};
  constexpr std::array<uint16_t, 3> scale_regs{{20, 26, 34}};
  std::vector<GraphDdrRegisterOperandSpec> operands;
  for (size_t i = 0; i < weights.size(); ++i) {
    operands.push_back({{fp16 ? GraphDdrRegisterRole::ModelWeight
                             : GraphDdrRegisterRole::Int8ModelWeight,
        GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
        weight_regs[i], static_cast<uint16_t>(weight_regs[i] + 1)}, *weights[i]});
    if (!fp16)
      operands.push_back({{GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
          scale_regs[i], static_cast<uint16_t>(scale_regs[i] + 1)}, *scales[i]});
  }
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id, std::move(operands));
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "Lingbot2 prefill QKV optional kernel is absent");
  for (size_t i = 0; i < weights.size(); ++i) {
    graph.keep_alive(*weights[i]);
    rpu_ddr_flush(weights[i]->data_ptr());
    if (!fp16) {
      graph.keep_alive(*scales[i]);
      rpu_ddr_flush(scales[i]->data_ptr());
    }
  }
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, q_output_spm_addr, 0,
          224, 512, 2560, 0, 8, 5120, 1024, 0, 1,
          static_cast<uint16_t>(fp16 ? 2 : 1)});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 30, k_output_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 38, v_output_spm_addr);
  writer.write(*kernel);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,
      {static_cast<uint16_t>(fp16 ? 10 : 11), 1, 1}, {0,1,2,3,4,5,6,7});
}

// Exact biased Vision M768/H1024/QKV1024/TP8 ACC16 dispatch.
// Preserve independent packed owners, per-channel scales and bias retirement.
void rpu_launch_lingbot2_vision_qkv_joint_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    uint32_t q_output_spm_addr, uint32_t k_output_spm_addr,
    uint32_t v_output_spm_addr,
    uint32_t q_bias_spm_addr, uint32_t k_bias_spm_addr,
    uint32_t v_bias_spm_addr,
    const at::Tensor& q_scale, const at::Tensor& k_scale,
    const at::Tensor& v_scale) {
  TORCH_CHECK(q_weight.defined() &&
                  (q_weight.scalar_type() == at::kHalf ||
                   q_weight.scalar_type() == at::kChar),
              "Lingbot2 Vision QKV requires uniform FP16 or signed INT8 weights");
  const bool fp16 = q_weight.scalar_type() == at::kHalf;
  const std::array<const at::Tensor*, 3> weights{{&q_weight, &k_weight, &v_weight}};
  const std::array<const at::Tensor*, 3> scales{{&q_scale, &k_scale, &v_scale}};
  constexpr std::array<int64_t, 3> channels{{1024, 1024, 1024}};
  const auto owned = [](const at::Tensor& tensor) {
    if (!(tensor.defined() &&
        tensor.device().type() == c10::DeviceType::PrivateUse1 &&
        tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
        tensor.storage_offset() >= 0)) return false;
    const uint64_t storage_bytes = tensor.storage().nbytes();
    const uint64_t item_bytes = tensor.element_size();
    const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset());
    return item_bytes != 0 && offset <= storage_bytes / item_bytes &&
        tensor.nbytes() <= storage_bytes - offset * item_bytes;
  };
  const auto ddr_ok = [](const at::Tensor& tensor) {
    const uint64_t address = RpuGetDevAddr(tensor.data_ptr());
    constexpr uint64_t limit = uint64_t{1} << 40;
    return address != 0 && (address & 255u) == 0 && address < limit &&
        tensor.nbytes() <= limit - address;
  };
  for (size_t i = 0; i < weights.size(); ++i) {
    const auto& weight = *weights[i];
    TORCH_CHECK(owned(weight) && weight.scalar_type() == q_weight.scalar_type() &&
                    weight.device() == q_weight.device() && weight.dim() == 2 &&
                    weight.size(0) == channels[i] && weight.size(1) == 1024,
                "Lingbot2 Vision QKV requires complete contiguous same-device weight owners [1024,1024]");
    TORCH_CHECK(ddr_ok(weight),
                "Lingbot2 Vision QKV weight violates aligned complete shift8 DDR addressing");
    const auto& scale = *scales[i];
    if (fp16) {
      TORCH_CHECK(!scale.defined(), "Lingbot2 FP16 Vision QKV cannot carry quantization scales");
    } else {
      TORCH_CHECK(owned(scale) && scale.scalar_type() == at::kHalf &&
                      scale.device() == q_weight.device() && scale.dim() == 1 &&
                      scale.size(0) == channels[i],
                  "Lingbot2 W8 Vision QKV requires complete same-device FP16 per-channel scale owners");
      TORCH_CHECK(ddr_ok(scale),
                  "Lingbot2 Vision QKV scale violates aligned complete shift8 DDR addressing");
    }
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Lingbot2 Vision QKV requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 7> ranges{{
      {input_spm_addr, 768 * 1024 * 2},
      {q_output_spm_addr, 768 * 128 * 2},
      {k_output_spm_addr, 768 * 128 * 2},
      {v_output_spm_addr, 768 * 128 * 2},
      {q_bias_spm_addr, 128 * 2},
      {k_bias_spm_addr, 128 * 2},
      {v_bias_spm_addr, 128 * 2}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && (address & 31u) == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                    uint64_t{address} + bytes <= (uint64_t{1} << 32),
                "Lingbot2 Vision QKV requires complete aligned absolute SPM intervals");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t{address} + bytes <= ranges[j].first ||
                      uint64_t{ranges[j].first} + ranges[j].second <= address,
                  "Lingbot2 Vision QKV input, outputs and biases must be disjoint");
  }
  auto* queue = GET_QUEUE(8);
  TORCH_CHECK(queue != nullptr, "Lingbot2 Vision QKV eight-core queue is absent");
  const KernelId id = fp16 ? KernelId::LINGBOT2_VISION_QKV_JOINT_FP16_M768
                           : KernelId::LINGBOT2_VISION_QKV_JOINT_W8_M768;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Lingbot2 Vision QKV optional precision payload is absent");
  constexpr std::array<uint16_t, 3> weight_regs{{2, 24, 32}};
  constexpr std::array<uint16_t, 3> scale_regs{{20, 26, 34}};
  std::vector<GraphDdrRegisterOperandSpec> operands;
  for (size_t i = 0; i < weights.size(); ++i) {
    operands.push_back({{fp16 ? GraphDdrRegisterRole::ModelWeight
                             : GraphDdrRegisterRole::Int8ModelWeight,
        GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
        weight_regs[i], static_cast<uint16_t>(weight_regs[i] + 1)}, *weights[i]});
    if (!fp16)
      operands.push_back({{GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
          scale_regs[i], static_cast<uint16_t>(scale_regs[i] + 1)}, *scales[i]});
  }
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id, std::move(operands));
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "Lingbot2 Vision QKV optional kernel is absent");
  for (size_t i = 0; i < weights.size(); ++i) {
    graph.keep_alive(*weights[i]);
    rpu_ddr_flush(weights[i]->data_ptr());
    if (!fp16) {
      graph.keep_alive(*scales[i]);
      rpu_ddr_flush(scales[i]->data_ptr());
    }
  }
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, q_output_spm_addr, q_bias_spm_addr,
          768, 128, 1024, 1, 8, 2048, 256, 0, 1,
          static_cast<uint16_t>(fp16 ? 2 : 1)});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 28, k_bias_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 30, k_output_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, v_bias_spm_addr);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 38, v_output_spm_addr);
  writer.write(*kernel);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,
      {static_cast<uint16_t>(fp16 ? 6 : 3),
       static_cast<uint16_t>(fp16 ? 2 : 3), 1}, {0,1,2,3,4,5,6,7});
}

static void check_wall_action_spm_weight_range(
    uint32_t weight_spm_addr,
    uint64_t weight_bytes,
    const char* name) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Wall Action SPM-weight launcher requires initialized SPM");
  const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
  const bool in_range =
      weight_spm_addr >= spm_base0 &&
      weight_bytes <= SpmAllocator::SPM_USABLE &&
      static_cast<uint64_t>(weight_spm_addr - spm_base0) <=
          SpmAllocator::SPM_USABLE - weight_bytes;
  TORCH_CHECK(in_range && weight_spm_addr % 256 == 0,
              "Wall Action ", name,
              " compact weight must be a 256-byte-aligned local SPM range");
}

// HyVLA's M51 Gate/Up pair uses the reviewed runtime-M Wall SPM payload.
// Keep the separate Wall launcher and its M20 owner contract unchanged.
void rpu_launch_hyvla_gateup_resident_spm_kernel(
    uint32_t input_spm, uint32_t gate_output_spm, uint32_t up_output_spm,
    uint32_t gate_weight_spm, uint32_t up_weight_spm, int64_t rows) {
  launch_hyvla_gateup_resident_spm(input_spm, gate_output_spm, up_output_spm,
      gate_weight_spm, up_weight_spm, rows, false);
}

void rpu_launch_wall_action_gate_up_w8a16_multibase_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t gate_output_spm_addr,
    uint32_t up_output_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale,
    uint32_t gate_weight_spm_addr,
    uint32_t up_weight_spm_addr,
    const WallSortingConfig& sorting, int64_t rows) {
  const bool fp16 = gate_weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 4u),
              "Wall FP16 Action transfer requires its cold opt-in");
  TORCH_CHECK(wall_action_fused_rows_supported(rows, sorting),
              "Wall Action fusion requires M20 or cold opted-in M32");
  const int64_t kM = rows;
  constexpr int64_t kK = 1024;
  constexpr int64_t kN = 2048;
  constexpr int64_t kNPerCore = 256;
  const uint64_t kWeightBytes = kNPerCore * kK * (fp16 ? 2 : 1);
  const bool use_spm_weight =
      gate_weight_spm_addr != 0 || up_weight_spm_addr != 0;
  // The M592 SPM-weight payload has the same runtime M loop as DDR. The
  // separate A16-resident payload retains its exact M20 activation layout.
  TORCH_CHECK(rows == 20 || !use_spm_weight || !sorting.action_activation_reuse,
              "Wall Action M32 Gate/Up does not admit resident activation reuse");
  TORCH_CHECK(!use_spm_weight ||
                  (gate_weight_spm_addr != 0 && up_weight_spm_addr != 0),
              "Wall Action gate+up requires both local SPM weight addresses");

  auto check_weight = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                    tensor.is_contiguous() && tensor.dim() == 2 &&
                    tensor.size(0) == kN && tensor.size(1) == kK,
                "Wall Action gate+up multibase ", name,
                " weight must be contiguous device int8 [", kN, ",", kK,
                "], got ", tensor.sizes());
  };
  auto check_scale = [&](const at::Tensor& tensor, const char* name) {
    if (fp16) { TORCH_CHECK(!tensor.defined(), "FP16 has no quant scale"); return; }
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf &&
                    tensor.is_contiguous() && tensor.dim() == 1 &&
                    tensor.size(0) == kN,
                "Wall Action gate+up multibase ", name,
                " scale must be contiguous device fp16 [", kN, "], got ",
                tensor.sizes());
  };
  if (use_spm_weight) {
    check_wall_action_spm_weight_range(
        gate_weight_spm_addr, kWeightBytes, "gate");
    check_wall_action_spm_weight_range(
        up_weight_spm_addr, kWeightBytes, "up");
  } else {
    check_weight(gate_weight, "gate");
    check_weight(up_weight, "up");
  }
  check_scale(gate_scale, "gate");
  check_scale(up_scale, "up");
  TORCH_CHECK(input_spm_addr != 0 && gate_output_spm_addr != 0 &&
                  up_output_spm_addr != 0 &&
                  gate_output_spm_addr != up_output_spm_addr,
              "Wall Action gate+up multibase requires non-zero distinct "
              "gate/up SPM outputs");

  const bool w16_reuse = use_spm_weight && sorting.action_activation_reuse;
  const KernelId legacy_kernel_id = use_spm_weight
      ? KernelId::WALL_ACTION_GATE_UP_W8A16_MULTIBASE_M592N32_SPM_WEIGHT
      : KernelId::WALL_ACTION_GATE_UP_W8A16_MULTIBASE_M592N32;
  TORCH_CHECK(!fp16 || !w16_reuse || use_spm_weight, "FP16 activation reuse requires resident weights");
  const KernelId kernel_id = fp16
      ? (use_spm_weight
          ? (w16_reuse ? KernelId::WALL_ACTION_GATE_UP_FP16_A16RESIDENT_SPM_WEIGHT_M20N32K1024
                       : KernelId::WALL_ACTION_GATE_UP_FP16_SPM_WEIGHT_M592N32K128)
          : KernelId::WALL_ACTION_GATE_UP_FP16_M592N32K128)
      : w16_reuse
          ? KernelId::WALL_ACTION_GATE_UP_W8A16_A16RESIDENT_SPM_WEIGHT_M20N32K1024
          : legacy_kernel_id;
  std::vector<GraphDdrRegisterOperandSpec> operands;
  if (!use_spm_weight) {
    operands.push_back(GraphDdrRegisterOperandSpec{
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::Int8ModelWeight,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
        gate_weight});
  }
  operands.push_back(GraphDdrRegisterOperandSpec{
      GraphDdrRegisterAbi{
          GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
      gate_scale});
  if (!use_spm_weight) {
    operands.push_back(GraphDdrRegisterOperandSpec{
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::Int8ModelWeight,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25},
        up_weight});
  }
  operands.push_back(GraphDdrRegisterOperandSpec{
      GraphDdrRegisterAbi{
          GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27},
      up_scale});
  auto register_writer =
      wall_precision_registers(
          kernel_id, fp16, std::move(operands));

  if (!use_spm_weight) {
    for (const at::Tensor* tensor : {&gate_weight, &up_weight}) {
      if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
      TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                  "Wall Action gate+up multibase DDR weight is not 256-byte "
                  "aligned");
    }
  }
  for (const at::Tensor* tensor : {&gate_scale, &up_scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Wall Action gate+up multibase DDR scale is not 256-byte "
                "aligned");
  }

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      w16_reuse ? KERNEL_ID_NAMES[static_cast<size_t>(kernel_id)] : use_spm_weight
          ? "gate_up_parallel_w8a16_multibase_spm_weight_m592n32k128 missing "
            "from expansion ref"
          : "gate_up_parallel_w8a16_multibase_m592n32k128 missing from "
            "expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, gate_output_spm_addr, 0,
          static_cast<uint32_t>(kM), static_cast<uint32_t>(kNPerCore),
          static_cast<uint32_t>(kK), 0, NUM_CORES,
          static_cast<uint32_t>(kK * sizeof(c10::Half)),
          static_cast<uint32_t>(kNPerCore * sizeof(c10::Half)), 0, 1, 1});
  if (use_spm_weight) {
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, gate_weight_spm_addr);
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 24, up_weight_spm_addr);
  }
  kernel->set_regs(30, static_cast<uint16_t>(up_output_spm_addr));
  kernel->set_regs(31, static_cast<uint16_t>(up_output_spm_addr >> 16));
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  if (w16_reuse) {
    queue->enqueu_kernel(*kernel, {8, 1, 1}, cores);
  } else {
    queue->enqueu_kernel(*kernel, {16, 1, 1}, cores);
  }
}

void rpu_launch_wall_action_linear_ring_postnorm_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& weight,
    uint32_t partial_spm_addr,
    uint32_t residual_spm_addr,
    uint32_t raw_residual_local_spm_addr,
    uint32_t normalized_spm_addr,
    uint32_t gamma_spm_addr,
    const at::Tensor& scale,
    bool residual_local_shard,
    double eps,
    uint32_t weight_spm_addr,
    const WallSortingConfig& sorting, int64_t rows) {
  const bool fp16 = weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 4u),
              "Wall FP16 Action transfer requires its cold opt-in");
  TORCH_CHECK(wall_action_fused_rows_supported(rows, sorting),
              "Wall Action fusion requires M20 or cold opted-in M32");
  const int64_t kM = rows;
  constexpr int64_t kN = 1024;
  constexpr int64_t kGlobalK = 2048;
  constexpr int64_t kLocalK = kGlobalK / NUM_CORES;
  constexpr uint32_t kRoundElemMax = 31 * 256 * NUM_CORES;
  const uint64_t kWeightBytes = kN * kLocalK * (fp16 ? 2 : 1);
  const uint64_t kInputBytes = kM * kLocalK * sizeof(c10::Half);
  const uint64_t kFullBytes = kM * kN * sizeof(c10::Half);
  const uint64_t kCompactBytes =
      wall_action_residual_rows(rows, 0) * kN * sizeof(c10::Half);
  constexpr uint64_t kGammaBytes = kN * sizeof(c10::Half);

  const bool use_spm_weight = weight_spm_addr != 0;
  // SPM weights preserve the DDR payload's dynamic M and SCM row slices.
  if (use_spm_weight) {
    check_wall_action_spm_weight_range(
        weight_spm_addr, kWeightBytes, "row-linear");
  } else {
    TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1 &&
                    weight.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                    weight.is_contiguous() && weight.dim() == 2 &&
                    weight.size(0) == kN && weight.size(1) == kGlobalK,
                "Wall Action linear-ring-postnorm weight must be contiguous "
                "device int8 [1024,2048], got ", weight.sizes());
  }
  if (fp16) { TORCH_CHECK(!scale.defined(), "FP16 has no quant scale"); } else {
  TORCH_CHECK(scale.device().type() == c10::DeviceType::PrivateUse1 &&
                  scale.scalar_type() == at::kHalf && scale.is_contiguous() &&
                  scale.dim() == 1 && scale.size(0) == kN,
              "Wall Action linear-ring-postnorm scale must be contiguous device "
              "fp16 [1024], got ",
              scale.sizes());
  }
  TORCH_CHECK(eps > 0.0,
              "Wall Action linear-ring-postnorm epsilon must be positive");
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Wall Action linear-ring-postnorm requires initialized SPM");

  const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
  const auto in_spm = [spm_base0](uint32_t addr, uint64_t bytes) {
    if (addr < spm_base0 || bytes > SpmAllocator::SPM_USABLE) return false;
    const uint64_t off = static_cast<uint64_t>(addr) - spm_base0;
    return off <= SpmAllocator::SPM_USABLE - bytes;
  };
  const auto disjoint = [](uint32_t a, uint64_t a_bytes,
                           uint32_t b, uint64_t b_bytes) {
    return static_cast<uint64_t>(a) + a_bytes <= b ||
        static_cast<uint64_t>(b) + b_bytes <= a;
  };
  const uint64_t residual_bytes =
      residual_local_shard ? kCompactBytes : kFullBytes;
  TORCH_CHECK(
      in_spm(input_spm_addr, kInputBytes) &&
          in_spm(partial_spm_addr, kFullBytes) &&
          in_spm(residual_spm_addr, residual_bytes) &&
          in_spm(raw_residual_local_spm_addr, kCompactBytes) &&
          in_spm(normalized_spm_addr, kFullBytes) &&
          in_spm(gamma_spm_addr, kGammaBytes),
      "Wall Action linear-ring-postnorm received an out-of-range SPM operand");
  TORCH_CHECK(
      partial_spm_addr != normalized_spm_addr &&
          disjoint(input_spm_addr, kInputBytes,
                   partial_spm_addr, kFullBytes) &&
          disjoint(input_spm_addr, kInputBytes,
                   residual_spm_addr, residual_bytes) &&
          disjoint(input_spm_addr, kInputBytes,
                   raw_residual_local_spm_addr, kCompactBytes) &&
          disjoint(input_spm_addr, kInputBytes,
                   normalized_spm_addr, kFullBytes) &&
          disjoint(partial_spm_addr, kFullBytes,
                   residual_spm_addr, residual_bytes) &&
          disjoint(partial_spm_addr, kFullBytes,
                   raw_residual_local_spm_addr, kCompactBytes) &&
          disjoint(partial_spm_addr, kFullBytes,
                   normalized_spm_addr, kFullBytes) &&
          disjoint(residual_spm_addr, residual_bytes,
                   normalized_spm_addr, kFullBytes) &&
          disjoint(raw_residual_local_spm_addr, kCompactBytes,
                   normalized_spm_addr, kFullBytes) &&
          disjoint(gamma_spm_addr, kGammaBytes,
                   input_spm_addr, kInputBytes) &&
          disjoint(gamma_spm_addr, kGammaBytes,
                   partial_spm_addr, kFullBytes) &&
          disjoint(gamma_spm_addr, kGammaBytes,
                   residual_spm_addr, residual_bytes) &&
          disjoint(gamma_spm_addr, kGammaBytes,
                   raw_residual_local_spm_addr, kCompactBytes) &&
          disjoint(gamma_spm_addr, kGammaBytes,
                   normalized_spm_addr, kFullBytes) &&
          (residual_local_shard
               ? residual_spm_addr == raw_residual_local_spm_addr
               : disjoint(residual_spm_addr, kFullBytes,
                          raw_residual_local_spm_addr, kCompactBytes)),
      "Wall Action linear-ring-postnorm requires distinct input/partial/"
      "normalized/gamma ranges and an in-place compact residual carry");

  const KernelId legacy_kernel_id = use_spm_weight
      ? KernelId::WALL_ACTION_LINEAR_RING_POSTNORM_W8A16_M20N1024K256_SPM_WEIGHT
      : KernelId::WALL_ACTION_LINEAR_RING_POSTNORM_W8A16_M20N1024K256;
  const KernelId kernel_id = fp16
      ? (use_spm_weight ? KernelId::WALL_ACTION_FP16_LINEAR_RING_POSTNORM_SPM_WEIGHT_M20N1024K256
                        : KernelId::WALL_ACTION_FP16_LINEAR_RING_POSTNORM_M20N1024K256)
      : legacy_kernel_id;
  std::vector<GraphDdrRegisterOperandSpec> operands;
  if (!use_spm_weight) {
    operands.push_back(GraphDdrRegisterOperandSpec{
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::Int8ModelWeight,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
        weight});
  }
  operands.push_back(GraphDdrRegisterOperandSpec{
      GraphDdrRegisterAbi{
          GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
      scale});
  auto register_writer =
      wall_precision_registers(
          kernel_id, fp16, std::move(operands));
  if (!use_spm_weight) {
    rpu_ddr_flush(weight.data_ptr());
    TORCH_CHECK(RpuGetDevAddr(weight.data_ptr()) % 256 == 0,
                "Wall Action linear-ring-postnorm DDR weight is not 256-byte "
                "aligned");
  }
  if (scale.defined()) rpu_ddr_flush(scale.data_ptr());
  TORCH_CHECK(!scale.defined() || RpuGetDevAddr(scale.data_ptr()) % 256 == 0,
              "Wall Action linear-ring-postnorm DDR scale is not 256-byte "
              "aligned");

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      KERNEL_ID_NAMES[static_cast<size_t>(kernel_id)],
      " missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, partial_spm_addr, 0,
          static_cast<uint32_t>(kM), static_cast<uint32_t>(kN),
          static_cast<uint32_t>(kLocalK), 0, NUM_CORES,
          static_cast<uint32_t>(kLocalK * sizeof(c10::Half)),
          static_cast<uint32_t>(kN * sizeof(c10::Half)), 0, 0, 1});
  if (use_spm_weight) {
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, weight_spm_addr);
  }

  const uint32_t partial_off = partial_spm_addr - spm_base0;
  const uint32_t normalized_off = normalized_spm_addr - spm_base0;
  kernel->set_regs(24, static_cast<uint16_t>(1));
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 25, kRoundElemMax);
  kernel->set_regs(29, static_cast<uint16_t>(residual_local_shard ? 2 : 0));
  rpu_pl_tiling::set_u32_reg_pair(
      *kernel, 30, raw_residual_local_spm_addr - spm_base0);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 32, gamma_spm_addr - spm_base0);
  kernel->set_regs(
      34, static_cast<c10::Half>(1.0f / static_cast<float>(kN)).x);
  kernel->set_regs(35, static_cast<c10::Half>(eps).x);
  rpu_pl_tiling::set_u32_reg_pair(
      *kernel, 36, residual_spm_addr - spm_base0);
  kernel->set_regs(64, static_cast<uint16_t>(8));
  kernel->set_regs(65, static_cast<uint16_t>(1));
  kernel->set_regs(66, static_cast<uint16_t>(1));

  constexpr uint32_t kScm = 4096;
  for (uint32_t core = 0; core < NUM_CORES; ++core) {
    const uint32_t peer_partial =
        SPM_ALLOC.addr(core, partial_off) - spm_base0;
    const uint32_t peer_normalized =
        SPM_ALLOC.addr(core, normalized_off) - spm_base0;
    const uint32_t row_byte_off =
        wall_action_residual_row_offset(rows, core) * kN * sizeof(c10::Half);
    const uint32_t row_elems = wall_action_residual_rows(rows, core) * kN;
    rpu_set_legacy_scm_u16_checked(
        kernel, kScm + core, static_cast<uint16_t>(peer_partial),
        "Wall Action linear-ring-postnorm");
    rpu_set_legacy_scm_u16_checked(
        kernel, kScm + 8 + core, static_cast<uint16_t>(peer_partial >> 16),
        "Wall Action linear-ring-postnorm");
    rpu_set_legacy_scm_u16_checked(
        kernel, kScm + 16 + core, static_cast<uint16_t>(peer_normalized),
        "Wall Action linear-ring-postnorm");
    rpu_set_legacy_scm_u16_checked(
        kernel, kScm + 24 + core,
        static_cast<uint16_t>(peer_normalized >> 16),
        "Wall Action linear-ring-postnorm");
    rpu_set_legacy_scm_u16_checked(
        kernel, kScm + 32 + core, static_cast<uint16_t>(row_byte_off),
        "Wall Action linear-ring-postnorm");
    rpu_set_legacy_scm_u16_checked(
        kernel, kScm + 40 + core, static_cast<uint16_t>(row_byte_off >> 16),
        "Wall Action linear-ring-postnorm");
    rpu_set_legacy_scm_u16_checked(
        kernel, kScm + 48 + core, static_cast<uint16_t>(row_elems),
        "Wall Action linear-ring-postnorm");
    rpu_set_legacy_scm_u16_checked(
        kernel, kScm + 56 + core, static_cast<uint16_t>(row_elems >> 16),
        "Wall Action linear-ring-postnorm");
  }
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  queue->enqueu_kernel(*kernel, {8, 1, 1}, cores);
}

void rpu_launch_qwen3vl_4b_prefill_gate_up_swiglu_w8a16_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t gate_output_spm_addr,
    int64_t rows,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale) {
  constexpr int64_t kK = 2560;
  constexpr int64_t kN = 9728;
  constexpr int64_t kNPerCore = kN / NUM_CORES;
  TORCH_CHECK(rows >= 16 && rows <= 256 && rows % 16 == 0,
              "Qwen3-VL 4B Prefill fused gate+up+SwiGLU rows must be "
              "16-aligned in [16,256], got ", rows);
  auto check_weight = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined() &&
                    tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kChar && tensor.is_contiguous() &&
                    tensor.dim() == 2 && tensor.size(0) == kN && tensor.size(1) == kK,
                "Qwen3-VL 4B Prefill fused ", name,
                " weight must be contiguous device int8 [9728,2560]");
  };
  auto check_scale = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined() &&
                    tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
                    tensor.dim() == 1 && tensor.size(0) == kN,
                "Qwen3-VL 4B Prefill fused ", name,
                " scale must be contiguous device fp16 [9728]");
  };
  check_weight(gate_weight, "gate");
  check_weight(up_weight, "up");
  check_scale(gate_scale, "gate");
  check_scale(up_scale, "up");
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Qwen3-VL 4B Prefill fused gate+up requires initialized SPM");
  const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
  const uint64_t input_bytes = rows * kK * sizeof(c10::Half);
  const uint64_t output_bytes = rows * kNPerCore * sizeof(c10::Half);
  const auto in_spm = [spm_base0](uint32_t address, uint64_t bytes) {
    return address >= spm_base0 && address % 256 == 0 &&
        bytes <= SpmAllocator::SPM_USABLE &&
        static_cast<uint64_t>(address) - spm_base0 <=
            SpmAllocator::SPM_USABLE - bytes;
  };
  TORCH_CHECK(in_spm(input_spm_addr, input_bytes) &&
                  in_spm(gate_output_spm_addr, output_bytes) &&
                  (static_cast<uint64_t>(input_spm_addr) + input_bytes <=
                       gate_output_spm_addr ||
                   static_cast<uint64_t>(gate_output_spm_addr) + output_bytes <=
                       input_spm_addr),
              "Qwen3-VL 4B Prefill fused gate+up requires disjoint, "
              "256-byte-aligned in-range absolute SPM operands");

  const KernelId kernel_id =
      KernelId::WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M320N112;
  auto register_writer =
      RpuKernelGraph::active().stage_kernel_ddr_registers(
          kernel_id,
          std::vector<GraphDdrRegisterOperandSpec>{
              {GraphDdrRegisterAbi{
                   GraphDdrRegisterRole::Int8ModelWeight,
                   GraphDdrRegisterAccess::Read,
                   GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
               gate_weight},
              {GraphDdrRegisterAbi{
                   GraphDdrRegisterRole::PerChannelScale,
                   GraphDdrRegisterAccess::Read,
                   GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
               gate_scale},
              {GraphDdrRegisterAbi{
                   GraphDdrRegisterRole::Int8ModelWeight,
                   GraphDdrRegisterAccess::Read,
                   GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25},
               up_weight},
              {GraphDdrRegisterAbi{
                   GraphDdrRegisterRole::PerChannelScale,
                   GraphDdrRegisterAccess::Read,
                   GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27},
               up_scale}});
  for (const at::Tensor* tensor :
       {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Qwen3-VL 4B Prefill fused gate+up DDR operand is not "
                "256-byte aligned");
    RpuKernelGraph::active().keep_alive(*tensor);
  }
  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(kernel != nullptr,
              "fused_gate_up_swiglu_w8a16_m320n112k128 missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel, rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, gate_output_spm_addr, 0,
          static_cast<uint32_t>(rows), static_cast<uint32_t>(kNPerCore),
          static_cast<uint32_t>(kK), 0, NUM_CORES,
          static_cast<uint32_t>(kK * sizeof(c10::Half)),
          static_cast<uint32_t>(kNPerCore * sizeof(c10::Half)), 0, 1, 1});
  // This existing direct-store payload ignores the legacy scratch p28..31.
  // The COMPLETE owner keeps input/gate lifetimes; no out-of-band SPM is used.
  register_writer.write(*kernel);
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  queue->enqueu_kernel(*kernel, {11, 1, 1}, cores);
}

// The FP16 payload uses runtime M/N/K and retires each N tile independently.
// Unlike its W8 counterpart it does not read the external scratch registers.
template <int64_t Rows, int64_t Hidden, int64_t Intermediate>
static void launch_hyvla_gateup_swiglu_fp16(
    uint32_t input_spm, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output_spm, int64_t rows) {
  static_assert((Rows == 240 && Hidden == 2048 && Intermediate == 6144) ||
                (Rows == 51 && Hidden == 1024 && Intermediate == 2048));
  constexpr int64_t LocalN = Intermediate / NUM_CORES;
  TORCH_CHECK(rows == Rows && SPM_ALLOC.is_initialized(),
              "HyVLA fused SwiGLU requires its exact row profile and initialized SPM");
  for (const auto* weight : {&gate_weight, &up_weight}) {
    TORCH_CHECK(weight->defined() &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->scalar_type() == at::kHalf && weight->is_contiguous() &&
                    weight->dim() == 2 && weight->size(0) == Intermediate && weight->size(1) == Hidden,
                "HyVLA fused SwiGLU requires exact contiguous FP16 device weights");
    TORCH_CHECK(RpuGetDevAddr(weight->data_ptr()) % 256 == 0,
                "HyVLA fused SwiGLU requires aligned DDR weights");
  }
  TORCH_CHECK(gate_weight.device() == up_weight.device(),
              "HyVLA fused SwiGLU weights must share a device");
  const std::array<std::pair<uint32_t, uint64_t>, 2> regions{{
      {input_spm, Rows * Hidden * sizeof(c10::Half)},
      {output_spm, Rows * LocalN * sizeof(c10::Half)}}};
  const uint64_t base0 = SPM_ALLOC.addr(0, 0);
  for (const auto& region : regions) {
    const uint64_t start = region.first, bytes = region.second;
    TORCH_CHECK(start >= base0 && start % 256 == 0 &&
                    start - base0 <= SpmAllocator::SPM_USABLE - bytes,
                "HyVLA fused SwiGLU requires complete aligned SPM regions");
    for (int core = 0; core < NUM_CORES; ++core)
      TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + start - base0 + bytes - 1 <= UINT32_MAX,
                  "HyVLA fused SwiGLU absolute SPM range exceeds uint32");
  }
  TORCH_CHECK(uint64_t(input_spm) + regions[0].second <= output_spm ||
                  uint64_t(output_spm) + regions[1].second <= input_spm,
              "HyVLA fused SwiGLU requires disjoint input/output SPM");
  constexpr KernelId id = KernelId::WALL_PREFILL_GATE_UP_SWIGLU_FP16_M320N112K128;
  auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, {
      {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
          GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
      {GraphDdrRegisterAbi{GraphDdrRegisterRole::ModelWeight,
          GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, up_weight}});
  rpu_ddr_flush(gate_weight.data_ptr());
  rpu_ddr_flush(up_weight.data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "HyVLA fused SwiGLU FP16 payload unavailable");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm, 0, output_spm, 0, Rows, LocalN, Hidden, 0, NUM_CORES,
          Hidden * sizeof(c10::Half), LocalN * sizeof(c10::Half), 0, 1, 1});
  register_writer.write(*kernel);
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {uint16_t((LocalN + 111) / 112), 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_hyvla_vlm_gateup_swiglu_fp16_spm_kernel(
    uint32_t input_spm, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output_spm, int64_t rows) {
  launch_hyvla_gateup_swiglu_fp16<240, 2048, 6144>(
      input_spm, gate_weight, up_weight, output_spm, rows);
}

void rpu_launch_hyvla_expert_gateup_swiglu_fp16_spm_kernel(
    uint32_t input_spm, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output_spm, int64_t rows) {
  launch_hyvla_gateup_swiglu_fp16<51, 1024, 2048>(
      input_spm, gate_weight, up_weight, output_spm, rows);
}

static void launch_wall_base_prefill_gate_up_shared(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate, uint32_t up, int64_t rows,
    const at::Tensor& gate_scale, const at::Tensor& up_scale, bool fuse_silu, bool omit_up) {
  TORCH_CHECK(!omit_up || fuse_silu, "Up store omission requires in-VLM activation");
  constexpr int64_t kN = 11008, kK = 2048, kLocalN = 1376, kCores = 8;
  TORCH_CHECK(rows == 336 || rows == 384 ||
                  (omit_up && (rows == 224 || rows == 256 || rows == 320)),
              "Wall shared-input Prefill requires M336/M384 or admitted no-Up chunks");
  const at::Tensor* tensors[] = {&gate_weight, &up_weight, &gate_scale, &up_scale};
  for (int i = 0; i < 4; ++i) {
    const auto& t = *tensors[i];
    TORCH_CHECK(t.defined() && t.device().type() == at::kPrivateUse1 &&
                    t.device() == gate_weight.device() &&
                    t.scalar_type() == (i < 2 ? at::kChar : at::kHalf) &&
                    t.is_contiguous() && t.storage_offset() == 0 &&
                    t.sizes() == (i < 2 ? at::IntArrayRef({kN, kK}) : at::IntArrayRef({kN})),
                "Wall shared-input Prefill requires exact W8 weights and FP16 channel scales");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Wall shared-input Prefill requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::pair<uint64_t, uint64_t> slabs[] = {
      {input, rows * kK * 2}, {gate, rows * kLocalN * 2}, {up, rows * kLocalN * 2}};
  for (int i = 0; i < 3; ++i) {
    const auto [address, bytes] = slabs[i];
    TORCH_CHECK(address != 0 && address >= base && address % 256 == 0 &&
                    address - base <= SpmAllocator::SPM_USABLE - bytes,
                "Wall shared-input Prefill requires full aligned absolute SPM operands");
    for (int core = 0; core < kCores; ++core)
      TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) +
                      address - base + bytes - 1 <= UINT32_MAX,
                  "Wall shared-input Prefill SPM range exceeds register ABI");
    for (int j = 0; j < i; ++j)
      TORCH_CHECK(address + bytes <= slabs[j].first ||
                      slabs[j].first + slabs[j].second <= address,
                  "Wall shared-input Prefill input/Gate/Up must be disjoint");
  }
  std::array<uint64_t, 4> addresses{};
  for (int i = 0; i < 4; ++i) {
    addresses[i] = RpuGetDevAddr(tensors[i]->data_ptr());
    TORCH_CHECK(addresses[i] != 0 && addresses[i] % 256 == 0 &&
                    (addresses[i] >> 8) <= UINT32_MAX &&
                    tensors[i]->nbytes() - 1 <= (uint64_t{1} << 40) - 1 - addresses[i],
                "Wall shared-input Prefill requires complete aligned shift8 DDR operands");
    for (int j = 0; j < i; ++j)
      TORCH_CHECK(addresses[i] + tensors[i]->nbytes() <= addresses[j] ||
                      addresses[j] + tensors[j]->nbytes() <= addresses[i],
                  "Wall shared-input Prefill DDR operands must not overlap");
  }
  const auto id = omit_up ? KernelId::WALL_BASE_PREFILL_GATE_UP_SHARED_SILU_NOUP_M384N48K128
      : fuse_silu ? KernelId::WALL_BASE_PREFILL_GATE_UP_SHARED_SILU_M384N48K128
                  : KernelId::WALL_BASE_PREFILL_GATE_UP_SHARED_A_M384N48K128;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "Wall shared-input Prefill payload unavailable");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, gate_scale},
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, up_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27}, up_scale}});
  for (const auto* t : tensors) rpu_ddr_flush(t->data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Wall shared-input Prefill kernel missing");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel, {
      input, 0, gate, 0, static_cast<uint32_t>(rows), kLocalN, kK, 0, kCores,
      kK * 2, kLocalN * 2, 0, 1, 1});
  TORCH_CHECK(kernel->set_regs(28, static_cast<uint16_t>(up)) == 0 &&
                  kernel->set_regs(29, static_cast<uint16_t>(up >> 16)) == 0,
              "Wall shared-input Prefill Up address write failed");
  writer.write(*kernel);
  auto* queue = GET_QUEUE(kCores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {29, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_wall_base_prefill_gate_up_shared_input_spm_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate, uint32_t up, int64_t rows,
    const at::Tensor& gate_scale, const at::Tensor& up_scale) {
  launch_wall_base_prefill_gate_up_shared(input, gate_weight, up_weight, gate, up,
                                          rows, gate_scale, up_scale, false, false);
}

void rpu_launch_wall_base_prefill_gate_up_shared_silu_spm_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate, uint32_t up, int64_t rows,
    const at::Tensor& gate_scale, const at::Tensor& up_scale) {
  launch_wall_base_prefill_gate_up_shared(input, gate_weight, up_weight, gate, up,
                                          rows, gate_scale, up_scale, true, false);
}

void rpu_launch_wall_base_prefill_gate_up_shared_silu_no_up_spm_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate, uint32_t up, int64_t rows,
    const at::Tensor& gate_scale, const at::Tensor& up_scale) {
  launch_wall_base_prefill_gate_up_shared(input, gate_weight, up_weight, gate, up,
                                          rows, gate_scale, up_scale, true, true);
}


void rpu_launch_wall_base_prefill_gate_up_shared_fp16_spm_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate, int64_t rows) {
  constexpr int64_t kN = 11008, kK = 2048, kLocalN = 1376, kCores = 8;
  TORCH_CHECK(rows == 224 || rows == 256 || rows == 320,
              "Wall FP16 shared-input Prefill requires C224/C256/C320");
  const at::Tensor* tensors[] = {&gate_weight, &up_weight};
  for (const auto* t : tensors) {
    TORCH_CHECK(t->defined() && t->device().type() == at::kPrivateUse1 &&
                    t->device() == gate_weight.device() &&
                    t->scalar_type() == at::kHalf && t->is_contiguous() &&
                    t->storage_offset() == 0 && t->sizes() == at::IntArrayRef({kN, kK}),
                "Wall FP16 shared-input Prefill requires exact FP16 weights");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Wall FP16 Prefill requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::pair<uint64_t, uint64_t> slabs[] = {
      {input, rows * kK * 2}, {gate, rows * kLocalN * 2}};
  for (int i = 0; i < 2; ++i) {
    const auto [address, bytes] = slabs[i];
    TORCH_CHECK(address != 0 && address >= base && address % 256 == 0 &&
                    address - base <= SpmAllocator::SPM_USABLE - bytes,
                "Wall FP16 Prefill requires full aligned absolute SPM operands");
    for (int core = 0; core < kCores; ++core)
      TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) +
                      address - base + bytes - 1 <= UINT32_MAX,
                  "Wall FP16 Prefill SPM range exceeds register ABI");
    for (int j = 0; j < i; ++j)
      TORCH_CHECK(address + bytes <= slabs[j].first ||
                      slabs[j].first + slabs[j].second <= address,
                  "Wall FP16 Prefill input and Gate must be disjoint");
  }
  std::array<uint64_t, 2> addresses{};
  for (int i = 0; i < 2; ++i) {
    addresses[i] = RpuGetDevAddr(tensors[i]->data_ptr());
    TORCH_CHECK(addresses[i] != 0 && addresses[i] % 256 == 0 &&
                    (addresses[i] >> 8) <= UINT32_MAX &&
                    tensors[i]->nbytes() - 1 <= (uint64_t{1} << 40) - 1 - addresses[i],
                "Wall FP16 Prefill requires complete aligned shift8 DDR operands");
    for (int j = 0; j < i; ++j)
      TORCH_CHECK(addresses[i] + tensors[i]->nbytes() <= addresses[j] ||
                      addresses[j] + tensors[j]->nbytes() <= addresses[i],
                  "Wall FP16 Prefill DDR weights must not overlap");
  }
  const auto id = KernelId::WALL_BASE_PREFILL_GATE_UP_SHARED_SILU_FP16_M320N80K128;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "Wall FP16 shared-input payload unavailable");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, up_weight}});
  for (const auto* t : tensors) rpu_ddr_flush(t->data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Wall FP16 shared-input kernel missing");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel, {
      input, 0, gate, 0, static_cast<uint32_t>(rows), kLocalN, kK, 0, kCores,
      kK * 2, kLocalN * 2, 0, 1, 1});
  writer.write(*kernel);
  auto* queue = GET_QUEUE(kCores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {18, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_lingbot2_prefill_gate_up_shared_a_w8_m224_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t gate_output_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale) {
  constexpr int64_t kM = 224, kK = 2560, kN = 9728, kNPerCore = 1216;
  static_assert(NUM_CORES == 8, "Lingbot2 shared-A Prefill requires TP8");
  const auto owned = [](const at::Tensor& tensor) {
    if (!(tensor.defined() &&
        tensor.device().type() == c10::DeviceType::PrivateUse1 &&
        tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
        tensor.storage_offset() >= 0)) return false;
    const uint64_t storage_bytes = tensor.storage().nbytes();
    const uint64_t item_bytes = tensor.element_size();
    const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset());
    return item_bytes != 0 && offset <= storage_bytes / item_bytes &&
        tensor.nbytes() <= storage_bytes - offset * item_bytes;
  };
  const std::array<const at::Tensor*, 4> owners{{
      &gate_weight, &gate_scale, &up_weight, &up_scale}};
  for (size_t i = 0; i < owners.size(); ++i) {
    const auto& tensor = *owners[i];
    const bool scale = (i & 1u) != 0;
    TORCH_CHECK(owned(tensor) && tensor.device() == gate_weight.device() &&
                    tensor.scalar_type() == (scale ? at::kHalf : at::kChar) &&
                    tensor.dim() == (scale ? 1 : 2) &&
                    tensor.size(0) == kN && (scale || tensor.size(1) == kK),
                "Lingbot2 shared-A Prefill requires complete same-device "
                "device int8 [9728,2560] weights and fp16 [9728] scales");
    const uint64_t address = RpuGetDevAddr(tensor.data_ptr());
    constexpr uint64_t limit = uint64_t{1} << 40;
    TORCH_CHECK(address != 0 && (address & 255u) == 0 && address < limit &&
                    tensor.nbytes() <= limit - address,
                "Lingbot2 shared-A Prefill requires complete aligned shift8 DDR intervals");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Lingbot2 shared-A Prefill requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 2> ranges{{
      {input_spm_addr, kM * kK * sizeof(c10::Half)},
      {gate_output_spm_addr, kM * kNPerCore * sizeof(c10::Half)}}};
  for (const auto& range : ranges) {
    const auto [address, bytes] = range;
    TORCH_CHECK(address >= base && (address & 255u) == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                    uint64_t{address} + bytes <= (uint64_t{1} << 32),
                "Lingbot2 shared-A Prefill requires complete aligned absolute SPM intervals");
  }
  TORCH_CHECK(uint64_t{input_spm_addr} + ranges[0].second <= gate_output_spm_addr ||
                  uint64_t{gate_output_spm_addr} + ranges[1].second <= input_spm_addr,
              "Lingbot2 shared-A Prefill input/output SPM intervals must be disjoint");
  constexpr auto id = KernelId::LINGBOT2_PREFILL_GATEUP_SHARED_A_W8_M224;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Lingbot2 shared-A Prefill optional W8 payload is absent");
  auto* queue = GET_QUEUE(8);
  TORCH_CHECK(queue != nullptr, "Lingbot2 shared-A Prefill eight-core queue is absent");
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, gate_scale},
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, up_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27}, up_scale}});
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "Lingbot2 shared-A Prefill optional kernel is absent");
  for (const auto* tensor : owners) {
    graph.keep_alive(*tensor);
    rpu_ddr_flush(tensor->data_ptr());
  }
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, gate_output_spm_addr, 0, kM, kNPerCore, kK,
          0, 8, kK * sizeof(c10::Half), kNPerCore * sizeof(c10::Half), 0, 1, 1});
  writer.write(*kernel);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {13, 1, 1}, {0,1,2,3,4,5,6,7});
}

void rpu_launch_lingbot2_shared_gateup_resident_silu_w8_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& packed_weight,
    const at::Tensor& scale,
    uint32_t output_spm_addr) {
  constexpr int64_t kM = 51, kK = 768, kN = 1536, kNPerCore = 192, kI = 96;
  static_assert(NUM_CORES == 8, "Lingbot2 resident shared Gate/Up requires TP8");
  const auto owned = [](const at::Tensor& tensor) {
    if (!(tensor.defined() &&
        tensor.device().type() == c10::DeviceType::PrivateUse1 &&
        tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
        tensor.storage_offset() >= 0)) return false;
    const uint64_t storage_bytes = tensor.storage().nbytes();
    const uint64_t item_bytes = tensor.element_size();
    const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset());
    return item_bytes != 0 && offset <= storage_bytes / item_bytes &&
        tensor.nbytes() <= storage_bytes - offset * item_bytes;
  };
  const std::array<const at::Tensor*, 2> owners{{&packed_weight, &scale}};
  for (size_t i = 0; i < owners.size(); ++i) {
    const auto& tensor = *owners[i];
    const bool is_scale = i == 1;
    TORCH_CHECK(owned(tensor) && tensor.device() == packed_weight.device() &&
                    tensor.scalar_type() == (is_scale ? at::kHalf : at::kChar) &&
                    tensor.dim() == (is_scale ? 1 : 2) &&
                    tensor.size(0) == kN && (is_scale || tensor.size(1) == kK),
                "Lingbot2 resident shared Gate/Up requires complete same-device "
                "device int8 [1536,768] packed weight and fp16 [1536] scale");
    const uint64_t address = RpuGetDevAddr(tensor.data_ptr());
    constexpr uint64_t limit = uint64_t{1} << 40;
    TORCH_CHECK(address != 0 && (address & 255u) == 0 && address < limit &&
                    tensor.nbytes() <= limit - address,
                "Lingbot2 resident shared Gate/Up requires complete aligned shift8 DDR intervals");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Lingbot2 resident shared Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 2> ranges{{
      {input_spm_addr, kM * kK * sizeof(c10::Half)},
      {output_spm_addr, kM * kI * sizeof(c10::Half)}}};
  for (const auto& range : ranges) {
    const auto [address, bytes] = range;
    TORCH_CHECK(address >= base && (address & 255u) == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t{address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                    uint64_t{address} + bytes <= (uint64_t{1} << 32),
                "Lingbot2 resident shared Gate/Up requires complete aligned absolute SPM intervals");
  }
  TORCH_CHECK(uint64_t{input_spm_addr} + ranges[0].second <= output_spm_addr ||
                  uint64_t{output_spm_addr} + ranges[1].second <= input_spm_addr,
              "Lingbot2 resident shared Gate/Up input/output SPM intervals must be disjoint");
  constexpr auto id = KernelId::LINGBOT2_SHARED_GATEUP_RESIDENT_SILU_W8_T51;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Lingbot2 resident shared Gate/Up optional W8 payload is absent");
  auto* queue = GET_QUEUE(8);
  TORCH_CHECK(queue != nullptr, "Lingbot2 resident shared Gate/Up eight-core queue is absent");
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, packed_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "Lingbot2 resident shared Gate/Up optional kernel is absent");
  for (const auto* tensor : owners) {
    graph.keep_alive(*tensor);
    rpu_ddr_flush(tensor->data_ptr());
  }
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0, kM, kNPerCore, kK,
          0, 8, kK * sizeof(c10::Half), kI * sizeof(c10::Half), 0, 1, 1});
  writer.write(*kernel);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {3, 1, 1}, {0,1,2,3,4,5,6,7});
}

void rpu_launch_lingbot2_shared_gateup_resident_silu_fp16_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& packed_weight,
    uint32_t output_spm_addr) {
  constexpr int64_t kM = 51, kK = 768, kN = 1536, kNPerCore = 192, kI = 96;
  static_assert(NUM_CORES == 8, "Lingbot2 resident shared FP16 Gate/Up requires TP8");
  const auto owned = [](const at::Tensor& tensor) {
    if (!(tensor.defined() &&
        tensor.device().type() == c10::DeviceType::PrivateUse1 &&
        tensor.layout() == c10::Layout::Strided && tensor.is_contiguous() &&
        tensor.storage_offset() >= 0)) return false;
    const uint64_t storage_bytes = tensor.storage().nbytes();
    const uint64_t item_bytes = tensor.element_size();
    const uint64_t offset = static_cast<uint64_t>(tensor.storage_offset());
    return item_bytes != 0 && offset <= storage_bytes / item_bytes &&
        tensor.nbytes() <= storage_bytes - offset * item_bytes;
  };
  TORCH_CHECK(owned(packed_weight) && packed_weight.scalar_type() == at::kHalf &&
                  packed_weight.dim() == 2 && packed_weight.size(0) == kN &&
                  packed_weight.size(1) == kK,
              "Lingbot2 resident shared FP16 Gate/Up requires a complete "
              "contiguous device fp16 [1536,768] packed weight owner");
  const uint64_t address = RpuGetDevAddr(packed_weight.data_ptr());
  constexpr uint64_t limit = uint64_t{1} << 40;
  TORCH_CHECK(address != 0 && (address & 255u) == 0 && address < limit &&
                  packed_weight.nbytes() <= limit - address,
              "Lingbot2 resident shared FP16 Gate/Up requires a complete aligned shift8 DDR interval");
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Lingbot2 resident shared FP16 Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint64_t>, 2> ranges{{
      {input_spm_addr, kM * kK * sizeof(c10::Half)},
      {output_spm_addr, kM * kI * sizeof(c10::Half)}}};
  for (const auto& range : ranges) {
    const auto [spm_address, bytes] = range;
    TORCH_CHECK(spm_address >= base && (spm_address & 255u) == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    uint64_t{spm_address} - base <= SpmAllocator::SPM_USABLE - bytes &&
                    uint64_t{spm_address} + bytes <= (uint64_t{1} << 32),
                "Lingbot2 resident shared FP16 Gate/Up requires complete aligned absolute SPM intervals");
  }
  TORCH_CHECK(uint64_t{input_spm_addr} + ranges[0].second <= output_spm_addr ||
                  uint64_t{output_spm_addr} + ranges[1].second <= input_spm_addr,
              "Lingbot2 resident shared FP16 Gate/Up input/output SPM intervals must be disjoint");
  constexpr auto id = KernelId::LINGBOT2_SHARED_GATEUP_RESIDENT_SILU_FP16_T51;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Lingbot2 resident shared Gate/Up optional FP16 payload is absent");
  auto* queue = GET_QUEUE(8);
  TORCH_CHECK(queue != nullptr, "Lingbot2 resident shared FP16 Gate/Up eight-core queue is absent");
  auto& graph = RpuKernelGraph::active();
  auto writer = graph.stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, packed_weight}});
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr, "Lingbot2 resident shared FP16 Gate/Up optional kernel is absent");
  graph.keep_alive(packed_weight);
  rpu_ddr_flush(packed_weight.data_ptr());
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0, kM, kNPerCore, kK,
          0, 8, kK * sizeof(c10::Half), kI * sizeof(c10::Half), 0, 1, 2});
  writer.write(*kernel);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {2, 1, 1}, {0,1,2,3,4,5,6,7});
}

void rpu_launch_wall_prefill_gate_up_swiglu_w8a16_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t gate_output_spm_addr,
    uint32_t scratch_spm_addr,
    int64_t rows,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale,
    const WallSortingConfig& sorting) {
  const bool fp16 = gate_weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 2u),
              "Wall FP16 Prefill transfer requires its cold opt-in");
  constexpr int64_t kK = 2048;
  constexpr int64_t kN = 11008;
  constexpr int64_t kNPerCore = kN / NUM_CORES;
  constexpr uint32_t kScratchBytes = 573440;
  static_assert(kScratchBytes == 8 * 320 * 112 * sizeof(c10::Half));

  TORCH_CHECK(rows == 320 || rows == 640,
              "Wall Prefill fused gate+up+SwiGLU rows must be 320 or 640, got ",
              rows);
  auto check_weight = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                    tensor.is_contiguous() && tensor.dim() == 2 &&
                    tensor.size(0) == kN && tensor.size(1) == kK,
                "Wall Prefill fused gate+up+SwiGLU ", name,
                " weight must be contiguous device int8 [", kN, ",", kK,
                "], got ", tensor.sizes());
  };
  auto check_scale = [&](const at::Tensor& tensor, const char* name) {
    if (fp16) { TORCH_CHECK(!tensor.defined(), "FP16 has no quant scale"); return; }
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf &&
                    tensor.is_contiguous() && tensor.dim() == 1 &&
                    tensor.size(0) == kN,
                "Wall Prefill fused gate+up+SwiGLU ", name,
                " scale must be contiguous device fp16 [", kN, "], got ",
                tensor.sizes());
  };
  check_weight(gate_weight, "gate");
  check_weight(up_weight, "up");
  check_scale(gate_scale, "gate");
  check_scale(up_scale, "up");
  TORCH_CHECK(input_spm_addr != 0 && gate_output_spm_addr != 0 &&
                  scratch_spm_addr != 0 &&
                  input_spm_addr != gate_output_spm_addr &&
                  input_spm_addr != scratch_spm_addr &&
                  gate_output_spm_addr != scratch_spm_addr,
              "Wall Prefill fused gate+up+SwiGLU requires distinct non-zero "
              "input/gate/scratch SPM buffers");

  const bool p640_weight_outer = rows == 640;
  // Exact no-bias M640/N11008/K2048 path: same grid15, scratch and K128 retirement.
  const bool mlp_retile = p640_weight_outer &&
      (fp16 || sorting.prefill_mlp_retile);
  const KernelId legacy_kernel_id = mlp_retile
      ? KernelId::WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M160X4N96K128
      : p640_weight_outer
          ? KernelId::WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M128X5N96K128
          : KernelId::WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M320N112;
  const KernelId kernel_id = fp16
      ? (p640_weight_outer
          ? (sorting.fp16_mlp_m224
                 ? KernelId::WALL_PREFILL_GATE_UP_SWIGLU_FP16_M224X3N96K128
                 : KernelId::WALL_PREFILL_GATE_UP_SWIGLU_FP16_M160X4N96K128)
          : KernelId::WALL_PREFILL_GATE_UP_SWIGLU_FP16_M320N112K128)
      : legacy_kernel_id;
  auto register_writer =
      wall_precision_registers(
          kernel_id, fp16,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
                  gate_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
                  gate_scale},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25},
                  up_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27},
                  up_scale}});

  for (const at::Tensor* tensor :
       {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Wall Prefill fused gate+up+SwiGLU DDR operand is not "
                "256-byte aligned");
  }

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      mlp_retile ? KERNEL_ID_NAMES[static_cast<size_t>(kernel_id)] : p640_weight_outer
          ? "fused_gate_up_swiglu_w8a16_m128x5n96k128 missing from expansion ref"
          : "fused_gate_up_swiglu_w8a16_m320n112k128 missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, gate_output_spm_addr, 0,
          static_cast<uint32_t>(rows), static_cast<uint32_t>(kNPerCore),
          static_cast<uint32_t>(kK), 0, NUM_CORES,
          static_cast<uint32_t>(kK * sizeof(c10::Half)),
          static_cast<uint32_t>(kNPerCore * sizeof(c10::Half)), 0, 1, 1});
  kernel->set_regs(28, static_cast<uint16_t>(scratch_spm_addr));
  kernel->set_regs(29, static_cast<uint16_t>(scratch_spm_addr >> 16));
  kernel->set_regs(30, static_cast<uint16_t>(224));
  kernel->set_regs(31, static_cast<uint16_t>(0));
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  if (p640_weight_outer) {
    queue->enqueu_kernel(*kernel, {15, 1, 1}, cores);
  } else {
    queue->enqueu_kernel(*kernel, {13, 1, 1}, cores);
  }
}

void rpu_launch_pi05_gate_up_geglu_w8a16_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t output_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale) {
  constexpr int64_t kM = 400, kN = 16384, kK = 2048;
  for (const auto* weight : {&gate_weight, &up_weight}) {
    TORCH_CHECK(weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->scalar_type() == at::kChar &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == kN && weight->size(1) == kK,
                "Pi GeGLU requires contiguous device int8 [16384,2048] weights");
  }
  for (const auto* scale : {&gate_scale, &up_scale}) {
    TORCH_CHECK(scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->scalar_type() == at::kHalf &&
                    scale->is_contiguous() && scale->dim() == 1 &&
                    scale->size(0) == kN,
                "Pi GeGLU requires contiguous device FP16 [16384] scales");
  }
  constexpr uint64_t kBufferBytes = kM * kK * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi GeGLU requires initialized SPM");
  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  TORCH_CHECK(input_spm_addr >= spm_base && output_spm_addr >= spm_base &&
                  input_spm_addr - spm_base + kBufferBytes <= SpmAllocator::SPM_USABLE &&
                  output_spm_addr - spm_base + kBufferBytes <= SpmAllocator::SPM_USABLE &&
                  input_spm_addr % 256 == 0 && output_spm_addr % 256 == 0 &&
                  (uint64_t(input_spm_addr) + kBufferBytes <= output_spm_addr ||
                   uint64_t(output_spm_addr) + kBufferBytes <= input_spm_addr),
              "Pi GeGLU requires disjoint aligned input/output SPM slabs");
  constexpr auto id = KernelId::PI05_GATE_UP_GEGLU_W8A16_M400N80;
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, gate_scale},
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, up_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27}, up_scale}});
  for (const auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Pi GeGLU requires 256-byte aligned DDR operands");
  }
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Pi GeGLU kernel missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel, rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0,
          kM, kN / NUM_CORES, kK, 0, NUM_CORES,
          kK * sizeof(c10::Half), kN / NUM_CORES * sizeof(c10::Half),
          0, /*partition=*/1, /*dtype_mode=*/1});
  writer.write(*kernel);
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {26, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_denoise_gate_up_geglu_wint4a16_pgrp_m50_kernel(
    uint32_t input_spm_addr, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output_spm_addr,
    const at::Tensor& gate_scale, const at::Tensor& up_scale) {
  constexpr int64_t kM = 50, kN = 4096, kK = 1024;
  static_assert(NUM_CORES == 8, "Pi W4 GeGLU is the exact TP8 profile");
  for (const auto* weight : {&gate_weight, &up_weight}) {
    TORCH_CHECK(weight->defined() &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->scalar_type() == at::kByte &&
                    weight->is_contiguous() && weight->storage_offset() == 0 &&
                    weight->dim() == 2 && weight->size(0) == kN &&
                    weight->size(1) == kK / 2 && weight->nbytes() == kN * kK / 2,
                "Pi W4 GeGLU requires packed device uint8 [4096,512] weights");
  }
  for (const auto* scale : {&gate_scale, &up_scale}) {
    TORCH_CHECK(scale->defined() &&
                    scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->scalar_type() == at::kHalf &&
                    scale->is_contiguous() && scale->storage_offset() == 0 &&
                    scale->dim() == 2 &&
                    (scale->size(0) == 32 || scale->size(0) == 64 || scale->size(0) == 128),
                "Pi W4 GeGLU requires contiguous FP16 pgrp scales GS32/64/128");
  }
  const int64_t gs = gate_scale.size(0);
  constexpr int64_t kLocalN = kN / NUM_CORES;
  const int64_t local_g = kK / gs;
  constexpr int64_t kScaleNBlocks = (kLocalN + 63) / 64;
  const int64_t expected_scale_elements = ((local_g + 3) / 4) * kScaleNBlocks * NUM_CORES * 4 * 64;
  TORCH_CHECK(up_scale.sizes() == gate_scale.sizes() &&
                  gate_scale.size(1) == expected_scale_elements / gs &&
                  gate_scale.numel() == expected_scale_elements &&
                  gate_scale.nbytes() == expected_scale_elements * sizeof(c10::Half),
              "Pi W4 GeGLU pgrp scale layout/GS differs from the fixed TP8 ABI");
  constexpr uint64_t kInputBytes = kM * kK * sizeof(c10::Half);
  constexpr uint64_t kOutputBytes = kM * kLocalN * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi W4 GeGLU requires initialized SPM");
  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  TORCH_CHECK(input_spm_addr >= spm_base && output_spm_addr >= spm_base &&
                  input_spm_addr - spm_base + kInputBytes <= SpmAllocator::SPM_USABLE &&
                  output_spm_addr - spm_base + kOutputBytes <= SpmAllocator::SPM_USABLE &&
                  input_spm_addr % 256 == 0 && output_spm_addr % 256 == 0 &&
                  (uint64_t(input_spm_addr) + kInputBytes <= output_spm_addr ||
                   uint64_t(output_spm_addr) + kOutputBytes <= input_spm_addr),
              "Pi W4 GeGLU requires disjoint aligned input/output SPM slabs");
  for (const auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    rpu_ddr_flush(tensor->data_ptr());
    const uint64_t address = RpuGetDevAddr(tensor->data_ptr());
    const uint64_t alignment = tensor->scalar_type() == at::kByte ? 256 : NUM_CORES * 512;
    TORCH_CHECK(address != 0 && address % alignment == 0 &&
                    (address >> 8) <= UINT32_MAX,
                "Pi W4 GeGLU requires shift8 DDR, weight alignment256 and scale alignment4096");
  }
  constexpr auto id = KernelId::PI05_DENOISE_GATE_UP_GEGLU_WINT4A16_PGRP_M512N48K128;
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedInt4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, gate_scale},
          {{GraphDdrRegisterRole::PackedInt4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, up_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, up_scale}});
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Pi W4 GeGLU kernel missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_int4_group_regs(
      *kernel, rpu_pl_tiling::ParallelLinearInt4GroupRegisterArgs{
          rpu_pl_tiling::ParallelLinearRegisterArgs{
              input_spm_addr, 0, output_spm_addr, 0,
              kM, kLocalN, kK, 0, NUM_CORES,
              kK * sizeof(c10::Half), kLocalN * sizeof(c10::Half),
              0, /*partition=*/1, /*dtype_mode=*/1},
          static_cast<uint16_t>(gs), static_cast<uint16_t>(local_g),
          static_cast<uint32_t>(kScaleNBlocks * NUM_CORES * 512)});
  writer.write(*kernel);
  kernel->set_regs(64, static_cast<uint16_t>(11));
  kernel->set_regs(65, static_cast<uint16_t>(1));
  kernel->set_regs(66, static_cast<uint16_t>(1));
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {11, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}


void rpu_launch_pi05_denoise_gate_up_geglu_wint4a16_pgrp_m50n64_kernel(
    uint32_t input_spm_addr, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output_spm_addr,
    const at::Tensor& gate_scale, const at::Tensor& up_scale) {
  constexpr int64_t kM = 50, kN = 4096, kK = 1024;
  static_assert(NUM_CORES == 8, "Pi W4 GeGLU is the exact TP8 profile");
  for (const auto* weight : {&gate_weight, &up_weight}) {
    TORCH_CHECK(weight->defined() &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->scalar_type() == at::kByte &&
                    weight->is_contiguous() && weight->storage_offset() == 0 &&
                    weight->dim() == 2 && weight->size(0) == kN &&
                    weight->size(1) == kK / 2 && weight->nbytes() == kN * kK / 2,
                "Pi W4 GeGLU requires packed device uint8 [4096,512] weights");
  }
  for (const auto* scale : {&gate_scale, &up_scale}) {
    TORCH_CHECK(scale->defined() &&
                    scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->scalar_type() == at::kHalf &&
                    scale->is_contiguous() && scale->storage_offset() == 0 &&
                    scale->dim() == 2 &&
                    scale->size(0) == 128,
                "Pi W4 GeGLU requires contiguous FP16 pgrp scales GS128 for N64");
  }
  const int64_t gs = gate_scale.size(0);
  constexpr int64_t kLocalN = kN / NUM_CORES;
  const int64_t local_g = kK / gs;
  constexpr int64_t kScaleNBlocks = (kLocalN + 63) / 64;
  const int64_t expected_scale_elements = ((local_g + 3) / 4) * kScaleNBlocks * NUM_CORES * 4 * 64;
  TORCH_CHECK(up_scale.sizes() == gate_scale.sizes() &&
                  gate_scale.size(1) == expected_scale_elements / gs &&
                  gate_scale.numel() == expected_scale_elements &&
                  gate_scale.nbytes() == expected_scale_elements * sizeof(c10::Half),
              "Pi W4 GeGLU pgrp scale layout/GS differs from the fixed TP8 ABI");
  constexpr uint64_t kInputBytes = kM * kK * sizeof(c10::Half);
  constexpr uint64_t kOutputBytes = kM * kLocalN * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi W4 GeGLU requires initialized SPM");
  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  TORCH_CHECK(input_spm_addr >= spm_base && output_spm_addr >= spm_base &&
                  input_spm_addr - spm_base + kInputBytes <= SpmAllocator::SPM_USABLE &&
                  output_spm_addr - spm_base + kOutputBytes <= SpmAllocator::SPM_USABLE &&
                  input_spm_addr % 256 == 0 && output_spm_addr % 256 == 0 &&
                  (uint64_t(input_spm_addr) + kInputBytes <= output_spm_addr ||
                   uint64_t(output_spm_addr) + kOutputBytes <= input_spm_addr),
              "Pi W4 GeGLU requires disjoint aligned input/output SPM slabs");
  for (const auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    rpu_ddr_flush(tensor->data_ptr());
    const uint64_t address = RpuGetDevAddr(tensor->data_ptr());
    const uint64_t alignment = tensor->scalar_type() == at::kByte ? 256 : NUM_CORES * 512;
    TORCH_CHECK(address != 0 && address % alignment == 0 &&
                    (address >> 8) <= UINT32_MAX,
                "Pi W4 GeGLU requires shift8 DDR, weight alignment256 and scale alignment4096");
  }
  constexpr auto id = KernelId::PI05_DENOISE_GATE_UP_GEGLU_WINT4A16_PGRP_M464N64K128;
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedInt4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, gate_scale},
          {{GraphDdrRegisterRole::PackedInt4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, up_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, up_scale}});
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Pi W4 GeGLU kernel missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_int4_group_regs(
      *kernel, rpu_pl_tiling::ParallelLinearInt4GroupRegisterArgs{
          rpu_pl_tiling::ParallelLinearRegisterArgs{
              input_spm_addr, 0, output_spm_addr, 0,
              kM, kLocalN, kK, 0, NUM_CORES,
              kK * sizeof(c10::Half), kLocalN * sizeof(c10::Half),
              0, /*partition=*/1, /*dtype_mode=*/1},
          static_cast<uint16_t>(gs), static_cast<uint16_t>(local_g),
          static_cast<uint32_t>(kScaleNBlocks * NUM_CORES * 512)});
  writer.write(*kernel);
  kernel->set_regs(64, static_cast<uint16_t>(8));
  kernel->set_regs(65, static_cast<uint16_t>(1));
  kernel->set_regs(66, static_cast<uint16_t>(1));
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}


void rpu_launch_pi05_denoise_gate_up_geglu_w8a16_m50_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t output_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale) {
  constexpr int64_t kM = 50, kN = 4096, kK = 1024;
  TORCH_CHECK(RpuKernelGraph::has_active(),
              "Pi M50 GeGLU requires an active retained graph");
  const bool fp16 = gate_weight.defined() && gate_weight.scalar_type() == at::kHalf;
  const size_t weight_bytes = kN * kK * (fp16 ? sizeof(c10::Half) : 1);
  for (const auto* weight : {&gate_weight, &up_weight}) {
    TORCH_CHECK(weight->defined() &&
                    weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                    weight->is_contiguous() && weight->storage_offset() == 0 &&
                    weight->dim() == 2 &&
                    weight->size(0) == kN && weight->size(1) == kK &&
                    weight->nbytes() == weight_bytes &&
                    weight->device() == gate_weight.device(),
                "Pi M50 GeGLU requires matching contiguous device [4096,1024] weights");
  }
  for (const auto* scale : {&gate_scale, &up_scale}) {
    if (fp16) {
      TORCH_CHECK(!scale->defined(), "Pi FP16 M50 GeGLU rejects quantization scales");
      continue;
    }
    TORCH_CHECK(scale->defined() && scale->device() == gate_weight.device() &&
                    scale->scalar_type() == at::kHalf &&
                    scale->is_contiguous() && scale->storage_offset() == 0 &&
                    scale->dim() == 1 &&
                    scale->size(0) == kN && scale->nbytes() == kN * sizeof(c10::Half),
                "Pi M50 GeGLU requires contiguous device FP16 [4096] scales");
  }
  constexpr uint64_t kInputBytes = kM * kK * sizeof(c10::Half);
  constexpr uint64_t kOutputBytes = kM * (kN / NUM_CORES) * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi M50 GeGLU requires initialized SPM");
  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  TORCH_CHECK(input_spm_addr >= spm_base && output_spm_addr >= spm_base &&
                  input_spm_addr - spm_base + kInputBytes <= SpmAllocator::SPM_USABLE &&
                  output_spm_addr - spm_base + kOutputBytes <= SpmAllocator::SPM_USABLE &&
                  input_spm_addr % 256 == 0 && output_spm_addr % 256 == 0 &&
                  (uint64_t(input_spm_addr) + kInputBytes <= output_spm_addr ||
                   uint64_t(output_spm_addr) + kOutputBytes <= input_spm_addr),
              "Pi M50 GeGLU requires disjoint aligned input/output SPM slabs");
  const auto id = fp16 ? KernelId::PI05_DENOISE_GATE_UP_GEGLU_FP16_M608N32K128
                      : KernelId::PI05_DENOISE_GATE_UP_GEGLU_W8A16_M512N48K128;
  const auto weight_role = fp16 ? GraphDdrRegisterRole::ModelWeight
                                : GraphDdrRegisterRole::Int8ModelWeight;
  std::vector<GraphDdrRegisterOperandSpec> operands{
          {{weight_role, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{weight_role, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, up_weight}};
  if (!fp16) {
    operands.push_back({{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, gate_scale});
    operands.push_back({{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27}, up_scale});
  }
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, operands);
  std::vector<std::pair<uint64_t, size_t>> owner_intervals;
  for (const auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    const uint64_t address = RpuGetDevAddr(tensor->data_ptr());
    TORCH_CHECK(address != 0 && address % 256 == 0 &&
                    (address >> 8) <= UINT32_MAX,
                "Pi M50 GeGLU requires aligned shift8 DDR operands");
    for (const auto& previous : owner_intervals) {
      TORCH_CHECK(address + tensor->nbytes() <= previous.first ||
                      previous.first + previous.second <= address,
                  "Pi M50 GeGLU DDR owners overlap");
    }
    owner_intervals.emplace_back(address, tensor->nbytes());
  }
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Pi M50 GeGLU kernel missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel, rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output_spm_addr, 0,
          kM, kN / NUM_CORES, kK, 0, NUM_CORES,
          kK * sizeof(c10::Half), kN / NUM_CORES * sizeof(c10::Half),
          0, /*partition=*/1, /*dtype_mode=*/static_cast<uint16_t>(fp16 ? 0 : 1)});
  writer.write(*kernel);
  auto* queue = GET_QUEUE(NUM_CORES);
  TORCH_CHECK(queue, "Pi M50 GeGLU 8-core queue is absent");
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {fp16 ? 16u : 11u, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_prefill_gate_up_geglu_weight_outer_w8a16_c400x2_kernel(
    uint32_t input0_spm_addr,
    uint32_t input1_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t output0_spm_addr,
    uint32_t output1_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale, int64_t rows) {
  TORCH_CHECK(rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 336 || rows == 400 || rows == 416 || rows == 432 || rows == 448 || rows == 464,
              "Pi Prefill GateUp pair requires C272/C288/C304/C320/C336/C400/C416/C432/C448/C464");
  const bool fp16 = gate_weight.defined() && gate_weight.scalar_type() == at::kHalf;
  const uint32_t kRows = rows;
  constexpr uint32_t kLocalN = 2048;
  constexpr uint32_t kK = 2048;
  constexpr uint32_t kCores = 8;
  const uint64_t kSlabBytes =
      uint64_t{kRows} * kLocalN * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Pi Prefill GateUp pair requires initialized SPM");
  const uint32_t spm_base = SPM_ALLOC.addr(0, 0);
  const std::array<uint32_t, 4> slabs{
      input0_spm_addr, input1_spm_addr,
      output0_spm_addr, output1_spm_addr};
  for (size_t index = 0; index != slabs.size(); ++index) {
    TORCH_CHECK(
        slabs[index] >= spm_base && slabs[index] % 256 == 0 &&
            uint64_t{slabs[index]} - spm_base <=
                SpmAllocator::SPM_USABLE - kSlabBytes,
        "Pi Prefill GateUp pair has an invalid absolute SPM interval");
    for (size_t prior = 0; prior != index; ++prior) {
      TORCH_CHECK(
          uint64_t{slabs[index]} + kSlabBytes <= slabs[prior] ||
              uint64_t{slabs[prior]} + kSlabBytes <= slabs[index],
          "Pi Prefill GateUp pair requires four disjoint SPM intervals");
    }
  }
  for (const auto* weight : {&gate_weight, &up_weight}) {
    TORCH_CHECK(
        weight->defined() &&
            weight->device().type() == c10::DeviceType::PrivateUse1 &&
            weight->scalar_type() == (fp16 ? at::kHalf : at::kChar) && weight->is_contiguous() &&
            weight->storage_offset() == 0 &&
            weight->sizes() == at::IntArrayRef({16384, 2048}),
        "Pi Prefill GateUp pair requires contiguous device int8 "
        "Gate/Up weights [16384,2048]");
  }
  for (const auto* scale : {&gate_scale, &up_scale}) {
    if (fp16) {
      TORCH_CHECK(!scale->defined(), "FP16 GateUp must not carry W8 scales");
      continue;
    }
    TORCH_CHECK(
        scale->defined() &&
            scale->device().type() == c10::DeviceType::PrivateUse1 &&
            scale->scalar_type() == at::kHalf && scale->is_contiguous() &&
            scale->storage_offset() == 0 &&
            scale->sizes() == at::IntArrayRef({16384}),
        "Pi Prefill GateUp pair requires contiguous device FP16 "
        "Gate/Up scales [16384]");
  }

  const KernelId kernel_id = rows == 432
      ? (fp16 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C432X2_M112N80K128
              : KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C432X2_M112N80K128) :
        rows == 464
      ? (fp16 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C464X2_M80N80K128
              : KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C464X2_M80N80K128) :
        rows == 336
      ? (fp16 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C336X2_M160N80K128
              : KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C336X2_M160N80K128) :
        rows == 448
      ? (fp16 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C448X2_M96N80K128
              : KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C448X2_M96N80K128) :
        rows == 416
      ? (fp16 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C416X2_M128N80K128
              : KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C416X2_M128N80K128)
      : rows == 288
      ? (fp16 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C288X2_M160N80K128
              : KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C288X2_M160N80K128)
      : rows == 304
      ? (fp16 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C304X2_M160N80K128
              : KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C304X2_M160N80K128) :
        rows == 320
      ? (fp16 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C320X2_M160N80K128
              : KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C320X2_M160N80K128)
      : fp16 ?
      (rows == 400 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C400X2_M160N80K128 :
                     KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C272X2_M160N80K128) :
      (rows == 400 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C400X2_M160N80K128 :
                     KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C272X2_M160N80K128);
  const auto role = fp16 ? GraphDdrRegisterRole::ModelWeight : GraphDdrRegisterRole::Int8ModelWeight;
  std::vector<GraphDdrRegisterOperandSpec> operands{
      {{role,GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},gate_weight},
      {{role,GraphDdrRegisterAccess::Read,GraphDdrRegisterEncoding::DevAddrShift8LoHi,24,25},up_weight}};
  if (!fp16) {
    operands.push_back({{GraphDdrRegisterRole::PerChannelScale,GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},gate_scale});
    operands.push_back({{GraphDdrRegisterRole::PerChannelScale,GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,26,27},up_scale});
  }
  auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(kernel_id,operands);
  for (const auto* tensor : {
           &gate_weight, &gate_scale, &up_weight, &up_scale}) {
    if (!tensor->defined()) continue;
    const uint64_t address = RpuGetDevAddr(tensor->data_ptr());
    TORCH_CHECK(address && address % 256 == 0 &&
                    (address >> 8) <= UINT32_MAX,
                "Pi Prefill GateUp pair DDR operand exceeds its aligned "
                "shift8 ABI");
    rpu_ddr_flush(tensor->data_ptr());
  }

  auto* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(kernel,
              "Pi Prefill GateUp pair expansion symbol is missing");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel, rpu_pl_tiling::ParallelLinearRegisterArgs{
                   input0_spm_addr, 0, output0_spm_addr, 0,
                   kRows, kLocalN, kK, 0, kCores,
                   kK * sizeof(c10::Half),
                   kLocalN * sizeof(c10::Half),
                   0, /*partition=*/1, /*dtype_mode=*/uint16_t(fp16 ? 0 : 1)});
  const auto set_spm_pair = [kernel](uint32_t reg, uint32_t address) {
    TORCH_CHECK(
        kernel->set_regs(reg, static_cast<uint16_t>(address)) == kKernelRegOk &&
            kernel->set_regs(reg + 1,
                             static_cast<uint16_t>(address >> 16)) ==
                kKernelRegOk,
        "Pi Prefill GateUp pair rejected an extra SPM register pair");
  };
  set_spm_pair(28, input1_spm_addr);
  set_spm_pair(30, output1_spm_addr);
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(kCores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(
      *kernel, {26, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_prefill_gate_up_geglu_online_w8a8_c272_kernel(
    uint32_t x_stage, uint32_t a8, uint32_t output,
    const at::Tensor& gate_weight, const at::Tensor& up_weight,
    const at::Tensor& gate_scale, const at::Tensor& up_scale) {
  constexpr uint32_t kInputBytes = 272 * 2048 * 2;
  constexpr uint32_t kA8Bytes = 272 * 2080;
  constexpr uint32_t kScaleBytes = 768;
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi C272 A8 requires initialized SPM");
  const uint32_t base = SPM_ALLOC.addr(0, 0);
  // X/stage is the only intentional alias. A8, its padded scale vector and Y
  // are disjoint. Every range is backed by an existing declared native root.
  const std::array<std::pair<uint32_t, uint32_t>, 3> ranges{{
      {x_stage, kInputBytes}, {a8, kA8Bytes + kScaleBytes}, {output, kInputBytes}}};
  for (size_t i = 0; i != ranges.size(); ++i) {
    const auto& range = ranges[i];
    TORCH_CHECK(range.first >= base && range.first % 256 == 0 &&
        uint64_t(range.first) - base <= SpmAllocator::SPM_USABLE - range.second,
        "Pi C272 A8 invalid absolute SPM range");
    for (size_t j = 0; j != i; ++j)
      TORCH_CHECK(uint64_t(range.first) + range.second <= ranges[j].first ||
                  uint64_t(ranges[j].first) + ranges[j].second <= range.first,
                  "Pi C272 A8 roots must be disjoint");
  }
  for (const auto* weight : {&gate_weight, &up_weight})
    TORCH_CHECK(weight->defined() && weight->device().type() == at::kPrivateUse1 &&
        weight->scalar_type() == at::kChar && weight->is_contiguous() &&
        weight->storage_offset() == 0 && weight->sizes() == at::IntArrayRef({16384, 2048}),
        "Pi C272 A8 requires installed device int8 Gate/Up [16384,2048]");
  for (const auto* scale : {&gate_scale, &up_scale})
    TORCH_CHECK(scale->defined() && scale->device().type() == at::kPrivateUse1 &&
        scale->scalar_type() == at::kFloat && scale->is_contiguous() &&
        scale->storage_offset() == 0 && scale->sizes() == at::IntArrayRef({16384}),
        "Pi C272 A8 requires cold promoted device FP32 scales [16384]");
  constexpr auto id = KernelId::PI05_PREFILL_GATE_UP_GEGLU_ONLINE_W8A8_P544_C272;
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
        {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
        {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 4, 5}, up_weight},
        {{GraphDdrRegisterRole::Fp32PerChannelScale, GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 6, 7}, gate_scale},
        {{GraphDdrRegisterRole::Fp32PerChannelScale, GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9}, up_scale}});
  for (const auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    const uint64_t address = RpuGetDevAddr(tensor->data_ptr());
    TORCH_CHECK(address && address % 256 == 0 && (address >> 8) <= UINT32_MAX,
                "Pi C272 A8 DDR operand exceeds aligned shift8 ABI");
    rpu_ddr_flush(tensor->data_ptr());
  }
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Pi C272 A8 expansion symbol missing");
  kernel->reset_regs();
  const auto set = [kernel](uint32_t index, uint16_t value) {
    TORCH_CHECK(kernel->set_regs(index, value) == kKernelRegOk,
                "Pi C272 A8 rejected register ", index);
  };
  for (const auto& item : std::array<std::pair<uint32_t, uint32_t>, 4>{{
      {0, a8}, {14, output}, {16, a8 + kA8Bytes}, {18, x_stage}}}) {
    set(item.first, uint16_t(item.second));
    set(item.first + 1, uint16_t(item.second >> 16));
  }
  constexpr std::array<uint16_t, 18> quant_params{{
      272, 1, 1, 2048, 0x57f0, 0xd7f0, 0, 0, 4096, 0, 2, 0, 2080, 0, 1, 0, 0, 0}};
  for (size_t i = 0; i != quant_params.size(); ++i) set(24 + i, quant_params[i]);
  set(64, 8); set(65, 1); set(66, 1);
  for (uint32_t i = 128; i <= 255; ++i) set(i, 0);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_pi05_prefill_gate_up_geglu_w8a8_split_kernel(
    uint32_t a8,uint32_t row_scale,uint32_t stage,
    const at::Tensor& gate_weight,const at::Tensor& up_weight,
    uint32_t output0,uint32_t output1,
    const at::Tensor& gate_scale,const at::Tensor& up_scale, int64_t rows) {
  TORCH_CHECK(rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 336 || rows == 400 || rows == 416 || rows == 432 || rows == 448 || rows == 464,
              "Pi Prefill A8 requires exact C272/C288/C304/C320/C336/C400/C416/C432/C448/C464");
  const uint32_t m=rows, panel=rows==464 ? 448 : rows==336 ? 320 : (rows>=400 ? 384 : 256);
  TORCH_CHECK(SPM_ALLOC.is_initialized(),"Pi Prefill A8 requires initialized SPM");
  const uint32_t base=SPM_ALLOC.addr(0,0);
  const std::array<std::pair<uint32_t,uint32_t>,5> ranges{{
      {a8,2*m*2080},{row_scale,2*m*2+(rows==272 || rows==288 || rows==304 || rows==320 || rows==336 || rows==416 || rows==432 || rows==448 || rows==464 ? 32u : 0u)},{stage,2*panel*2048},
      {output0,m*2048*2},{output1,m*2048*2}}};
  for(size_t i=0;i<ranges.size();++i) {
    const auto& range=ranges[i];
    TORCH_CHECK(range.first>=base && range.first%256==0 &&
        uint64_t(range.first)-base<=SpmAllocator::SPM_USABLE-range.second,
        "Pi Prefill A8 invalid absolute SPM range");
    for(size_t j=0;j<i;++j)
      TORCH_CHECK(uint64_t(range.first)+range.second<=ranges[j].first ||
                  uint64_t(ranges[j].first)+ranges[j].second<=range.first,
                  "Pi Prefill A8 operands must be disjoint");
  }
  for(const auto* weight:{&gate_weight,&up_weight})
    TORCH_CHECK(weight->defined() && weight->device().type()==at::kPrivateUse1 &&
        weight->scalar_type()==at::kChar && weight->is_contiguous() &&
        weight->storage_offset()==0 && weight->sizes()==at::IntArrayRef({16384,2048}),
        "Pi Prefill A8 requires installed device int8 Gate/Up [16384,2048]");
  for(const auto* scale:{&gate_scale,&up_scale})
    TORCH_CHECK(scale->defined() && scale->device().type()==at::kPrivateUse1 &&
        scale->scalar_type()==at::kFloat && scale->is_contiguous() &&
        scale->storage_offset()==0 && scale->sizes()==at::IntArrayRef({16384}),
        "Pi Prefill A8 requires cold promoted device FP32 scales [16384]");
  const auto id=rows==432 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C432_M48N32K2048 :
               rows==464 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C464_M48N32K2048 :
               rows==336 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C336_M48N32K2048 :
               rows==448 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C448_M48N32K2048 :
               rows==416 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C416_M48N32K2048 :
               rows==288 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C288_M48N32K2048 :
               rows==304 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C304_M48N32K2048
         : rows==320 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C320_M48N32K2048 :
               rows==400 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_M48N32K2048 :
                           KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C272_M48N32K2048;
  auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
        {{GraphDdrRegisterRole::Int8ModelWeight,GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi,2,3},gate_weight},
        {{GraphDdrRegisterRole::Int8ModelWeight,GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi,4,5},up_weight},
        {{GraphDdrRegisterRole::Fp32PerChannelScale,GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi,6,7},gate_scale},
        {{GraphDdrRegisterRole::Fp32PerChannelScale,GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi,8,9},up_scale}});
  for(const auto* tensor:{&gate_weight,&up_weight,&gate_scale,&up_scale}) {
    const uint64_t address=RpuGetDevAddr(tensor->data_ptr());
    TORCH_CHECK(address && address%256==0 && (address>>8)<=UINT32_MAX,
                "Pi Prefill A8 DDR operand exceeds aligned shift8 ABI");
    rpu_ddr_flush(tensor->data_ptr());
  }
  auto* kernel=GET_KERNEL(id);
  TORCH_CHECK(kernel,"Pi Prefill A8 expansion symbol missing");
  kernel->reset_regs();
  const auto set=[kernel](uint32_t index,uint16_t value) {
    TORCH_CHECK(kernel->set_regs(index,value)==kKernelRegOk,
                "Pi Prefill A8 rejected register ",index);
  };
  for(const auto& item:std::array<std::pair<uint32_t,uint32_t>,5>{{
      {0,a8},{14,output0},{16,row_scale},{18,stage},{22,output1}}}) {
    set(item.first,uint16_t(item.second));set(item.first+1,uint16_t(item.second>>16));
  }
  set(64,8);set(65,1);set(66,1);
  for(uint32_t i=128;i<=255;++i)set(i,0);
  writer.write(*kernel);
  auto* queue=GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{8,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t gate_output_spm_addr,
    uint32_t gate_bias_spm_addr,
    uint32_t up_bias_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale,
    uint32_t input1_spm_addr,
    uint32_t input2_spm_addr,
    uint32_t output1_spm_addr,
    uint32_t output2_spm_addr,
    const WallSortingConfig& sorting, int64_t rows) {
  const bool fp16 = gate_weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 1u),
              "Wall FP16 Vision transfer requires its cold opt-in");
  const int64_t kM = rows;
  constexpr int64_t kK = 1280;
  constexpr int64_t kN = 3584;
  constexpr int64_t kNPerCore = kN / NUM_CORES;

  auto check_weight = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                    tensor.is_contiguous() && tensor.dim() == 2 &&
                    tensor.size(0) == kN && tensor.size(1) == kK,
                "Wall Vision fused gate+up+SwiGLU ", name,
                " weight must be contiguous device int8 [", kN, ",", kK,
                "], got ", tensor.sizes());
  };
  auto check_scale = [&](const at::Tensor& tensor, const char* name) {
    if (fp16) { TORCH_CHECK(!tensor.defined(), "FP16 has no quant scale"); return; }
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf &&
                    tensor.is_contiguous() && tensor.dim() == 1 &&
                    tensor.size(0) == kN,
                "Wall Vision fused gate+up+SwiGLU ", name,
                " scale must be contiguous device fp16 [", kN,
                "], got ", tensor.sizes());
  };
  check_weight(gate_weight, "gate");
  check_weight(up_weight, "up");
  check_scale(gate_scale, "gate");
  check_scale(up_scale, "up");
  const bool multibase = input1_spm_addr != 0 && input2_spm_addr != 0 &&
      output1_spm_addr != 0 && output2_spm_addr != 0;
  TORCH_CHECK(
      multibase || (input1_spm_addr == 0 && input2_spm_addr == 0 &&
                    output1_spm_addr == 0 && output2_spm_addr == 0),
      "Wall Vision fused gate+up+SwiGLU multibase addresses must be all "
      "present or all absent");
  // The three-base payload resets its store extent to 192 rows; only the
  // single-base M352 payload preserves the runtime row bound through SwiGLU.
  const bool small_image = !fp16 && !multibase &&
      sorting.vision_row_reuse && sorting.silu_mul &&
      (rows == 324 || rows == 672);
  TORCH_CHECK(rows == 576 || small_image,
              "Wall Vision fused gate+up+SwiGLU requires M576 or cold "
              "single-base W8 M324/M672");
  TORCH_CHECK(input_spm_addr != 0 && gate_output_spm_addr != 0 &&
                  gate_bias_spm_addr != 0 && up_bias_spm_addr != 0 &&
                  input_spm_addr != gate_output_spm_addr &&
                  gate_bias_spm_addr != up_bias_spm_addr,
              "Wall Vision fused gate+up+SwiGLU requires non-zero input, "
              "output, and distinct bias SPM buffers");
  if (small_image) {
    TORCH_CHECK(SPM_ALLOC.is_initialized(),
                "Wall small-image Gate/Up requires initialized SPM");
    const uint64_t input_bytes = rows * kK * sizeof(c10::Half);
    const uint64_t output_bytes = rows * kNPerCore * sizeof(c10::Half);
    const uint32_t spm_base = SPM_ALLOC.addr(0, 0);
    auto check_range = [&](uint32_t address, uint64_t bytes) {
      TORCH_CHECK(address >= spm_base && address % 256 == 0 &&
                      bytes <= SpmAllocator::SPM_USABLE &&
                      static_cast<uint64_t>(address - spm_base) <=
                          SpmAllocator::SPM_USABLE - bytes,
                  "Wall small-image Gate/Up exceeds its local SPM range");
    };
    check_range(input_spm_addr, input_bytes);
    check_range(gate_output_spm_addr, output_bytes);
    check_range(gate_bias_spm_addr, kNPerCore * sizeof(c10::Half));
    check_range(up_bias_spm_addr, kNPerCore * sizeof(c10::Half));
    TORCH_CHECK(input_spm_addr + input_bytes <= gate_output_spm_addr ||
                    gate_output_spm_addr + output_bytes <= input_spm_addr,
                "Wall small-image Gate/Up input/output SPM must not overlap");
  }

  const KernelId legacy_kernel_id = multibase
      ? KernelId::WALL_VISION_GATE_UP_SWIGLU_W8A16_MULTIBASE_M192X3N96
      : KernelId::WALL_VISION_GATE_UP_SWIGLU_W8A16_M352N96;
  TORCH_CHECK(!fp16 || multibase, "FP16 requires three-base tile reuse");
  const KernelId kernel_id = fp16
      ? KernelId::WALL_VISION_GATE_UP_SWIGLU_FP16_MULTIBASE_M192X3N96K128
      : legacy_kernel_id;
  auto register_writer =
      wall_precision_registers(
          kernel_id, fp16,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
                  gate_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
                  gate_scale},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25},
                  up_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27},
                  up_scale}});

  for (const at::Tensor* tensor :
       {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Wall Vision fused gate+up+SwiGLU DDR operand is not "
                "256-byte aligned");
  }

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      multibase
          ? "fused_gate_up_swiglu_w8a16_multibase_m192x3n96k128 missing "
            "from expansion ref"
          : "fused_gate_up_swiglu_w8a16_m352n96k128 missing from expansion "
            "ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, gate_output_spm_addr, gate_bias_spm_addr,
          static_cast<uint32_t>(kM), static_cast<uint32_t>(kNPerCore),
          static_cast<uint32_t>(kK), 1, NUM_CORES,
          static_cast<uint32_t>(kK * sizeof(c10::Half)),
          static_cast<uint32_t>(kNPerCore * sizeof(c10::Half)), 0, 1, 1});
  kernel->set_regs(28, static_cast<uint16_t>(up_bias_spm_addr));
  kernel->set_regs(29, static_cast<uint16_t>(up_bias_spm_addr >> 16));
  if (multibase) {
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 30, input1_spm_addr);
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 32, input2_spm_addr);
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 34, output1_spm_addr);
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, output2_spm_addr);
  }
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  if (multibase) {
    queue->enqueu_kernel(*kernel, {5, 3, 1}, cores);
  } else {
    queue->enqueu_kernel(
        *kernel, {5, static_cast<uint16_t>(CeilDiv(rows, int64_t{352})), 1}, cores);
  }
}

void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    uint32_t q_output_spm_addr,
    uint32_t k_output_spm_addr,
    uint32_t v_output_spm_addr,
    uint32_t q_bias_spm_addr,
    uint32_t k_bias_spm_addr,
    uint32_t v_bias_spm_addr,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale,
    int64_t rows,
    bool kv_tile_owner,
    const WallSortingConfig& sorting) {
  const bool fp16 = q_weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 2u),
              "Wall FP16 Prefill transfer requires its cold opt-in");
  constexpr int64_t kK = 2048;
  constexpr int64_t kQ = 2048;
  constexpr int64_t kKv = 1024;
  constexpr int64_t kQPerCore = 256;

  TORCH_CHECK(rows == 320 || rows == 640 ||
                  (sorting.silu_mul && (rows == 336 || rows == 384 ||
                      rows == 224 || rows == 256)),
              "Wall QKV multibase rows must be 320/640 or cold small-prefix "
              "336/384 or cold long-prefix chunks 224/256, got ", rows);
  TORCH_CHECK(!kv_tile_owner || rows == 640,
              "Wall QKV K/V tile-owner requires exact P640, got rows=",
              rows);
  auto check_weight = [&](const at::Tensor& tensor, int64_t n,
                          const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                    tensor.is_contiguous() && tensor.dim() == 2 &&
                    tensor.size(0) == n && tensor.size(1) == kK,
                "Wall QKV multibase ", name,
                " weight must be contiguous device int8 [", n, ",", kK,
                "], got ", tensor.sizes());
  };
  auto check_scale = [&](const at::Tensor& tensor, int64_t n,
                         const char* name) {
    if (fp16) { TORCH_CHECK(!tensor.defined(), "FP16 has no quant scale"); return; }
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf &&
                    tensor.is_contiguous() && tensor.dim() == 1 &&
                    tensor.size(0) == n,
                "Wall QKV multibase ", name,
                " scale must be contiguous device fp16 [", n, "], got ",
                tensor.sizes());
  };
  check_weight(q_weight, kQ, "Q");
  check_weight(k_weight, kKv, "K");
  check_weight(v_weight, kKv, "V");
  check_scale(q_scale, kQ, "Q");
  check_scale(k_scale, kKv, "K");
  check_scale(v_scale, kKv, "V");
  TORCH_CHECK(q_bias_spm_addr != 0 && k_bias_spm_addr != 0 &&
                  v_bias_spm_addr != 0,
              "Wall QKV multibase requires all three persistent SPM biases");

  const KernelId kernel_id = fp16
      ? (kv_tile_owner ? KernelId::WALL_PREFILL_QKV_FP16_TILE_OWNER_M320X2N64K128
          : rows == 640 ? KernelId::WALL_PREFILL_QKV_FP16_M320X2N64K128
                        : KernelId::WALL_PREFILL_QKV_FP16_M448N64K128)
      : kv_tile_owner
      ? KernelId::WALL_QKV_W8A16_MULTIBASE_TILE_OWNER_M320X2N64
      : rows == 640 ? KernelId::WALL_QKV_W8A16_MULTIBASE_M320X2N64
                    : KernelId::WALL_QKV_W8A16_MULTIBASE_M448N64;
  auto register_writer =
      wall_precision_registers(
          kernel_id, fp16,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
                  q_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
                  q_scale},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25},
                  k_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27},
                  k_scale},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33},
                  v_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35},
                  v_scale}});

  for (const at::Tensor* tensor :
       {&q_weight, &k_weight, &v_weight, &q_scale, &k_scale, &v_scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Wall QKV multibase DDR operand is not 256-byte aligned");
  }

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      kv_tile_owner
          ? "qkv_parallel_w8a16_multibase_tile_owner_m320x2n64k128 "
            "missing from expansion ref"
          : rows == 640
          ? "qkv_parallel_w8a16_multibase_m320x2n64k128 missing from expansion ref"
          : "qkv_parallel_w8a16_multibase_m448n64k128 missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, q_output_spm_addr, q_bias_spm_addr,
          static_cast<uint32_t>(rows), static_cast<uint32_t>(kQPerCore),
          static_cast<uint32_t>(kK), 1, NUM_CORES,
          static_cast<uint32_t>(kK * sizeof(c10::Half)),
          static_cast<uint32_t>(kQPerCore * sizeof(c10::Half)), 0, 1, 1});

  auto set_u32 = [&](int reg, uint32_t value) {
    kernel->set_regs(reg, static_cast<uint16_t>(value));
    kernel->set_regs(reg + 1, static_cast<uint16_t>(value >> 16));
  };
  set_u32(28, k_bias_spm_addr);
  set_u32(30, k_output_spm_addr);
  set_u32(36, v_bias_spm_addr);
  set_u32(38, v_output_spm_addr);
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  queue->enqueu_kernel(*kernel, {8, 1, 1}, cores);
}

void rpu_launch_wall_action_qkv_w8a16_multibase_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    uint32_t q_output_spm_addr,
    uint32_t k_output_spm_addr,
    uint32_t v_output_spm_addr,
    uint32_t q_bias_spm_addr,
    uint32_t k_bias_spm_addr,
    uint32_t v_bias_spm_addr,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale,
    const at::Tensor* cos,
    const at::Tensor* sin,
    int64_t position,
    uint32_t q_weight_spm_addr,
    uint32_t k_weight_spm_addr,
    uint32_t v_weight_spm_addr,
    bool kv_owner_fanout,
    const WallSortingConfig& sorting, int64_t rows) {
  const bool fp16 = q_weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 4u),
              "Wall FP16 Action transfer requires its cold opt-in");
  TORCH_CHECK(wall_action_fused_rows_supported(rows, sorting),
              "Wall Action fusion requires M20 or cold opted-in M32");
  const int64_t kM = rows;
  constexpr int64_t kK = 1024;
  constexpr int64_t kQ = 2048;
  constexpr int64_t kKv = 1024;
  constexpr int64_t kQPerCore = 256;
  constexpr int64_t kKvPerCore = 128;
  const bool use_spm_weight =
      q_weight_spm_addr != 0 || k_weight_spm_addr != 0 ||
      v_weight_spm_addr != 0;
  // Both weight sources share the runtime M/partial-MRoPE tail; the separate
  // owner-fanout payload still has an exact M20 communication layout.
  TORCH_CHECK(rows == 20 || !kv_owner_fanout,
              "Wall Action M32 QKV requires replicated K/V");
  TORCH_CHECK(!use_spm_weight ||
                  (q_weight_spm_addr != 0 && k_weight_spm_addr != 0 &&
                   v_weight_spm_addr != 0),
              "Wall Action QKV requires all three local SPM weight addresses");

  auto check_weight = [&](const at::Tensor& tensor, int64_t n,
                          const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                    tensor.is_contiguous() && tensor.dim() == 2 &&
                    tensor.size(0) == n && tensor.size(1) == kK,
                "Wall Action QKV multibase ", name,
                " weight must be contiguous device int8 [", n, ",", kK,
                "], got ", tensor.sizes());
  };
  auto check_scale = [&](const at::Tensor& tensor, int64_t n,
                         const char* name) {
    if (fp16) { TORCH_CHECK(!tensor.defined(), "FP16 has no quant scale"); return; }
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf &&
                    tensor.is_contiguous() && tensor.dim() == 1 &&
                    tensor.size(0) == n,
                "Wall Action QKV multibase ", name,
                " scale must be contiguous device fp16 [", n, "], got ",
                tensor.sizes());
  };
  if (use_spm_weight) {
    check_wall_action_spm_weight_range(
        q_weight_spm_addr, kQPerCore * kK * (fp16 ? 2 : 1), "Q");
    check_wall_action_spm_weight_range(
        k_weight_spm_addr, kKvPerCore * kK * (fp16 ? 2 : 1), "K");
    check_wall_action_spm_weight_range(
        v_weight_spm_addr, kKvPerCore * kK * (fp16 ? 2 : 1), "V");
  } else {
    check_weight(q_weight, kQ, "Q");
    check_weight(k_weight, kKv, "K");
    check_weight(v_weight, kKv, "V");
  }
  check_scale(q_scale, kQ, "Q");
  check_scale(k_scale, kKv, "K");
  check_scale(v_scale, kKv, "V");
  TORCH_CHECK(q_bias_spm_addr != 0 && k_bias_spm_addr != 0 &&
                  v_bias_spm_addr != 0,
              "Wall Action QKV multibase requires all three persistent SPM "
              "biases");

  const bool fuse_partial_mrope = cos != nullptr || sin != nullptr;
  TORCH_CHECK(rows == 20 || fuse_partial_mrope,
              "Wall Action M32 QKV requires fused partial M-RoPE");
  TORCH_CHECK((cos == nullptr) == (sin == nullptr),
              "Wall Action QKV partial M-RoPE requires both cos and sin");
  TORCH_CHECK(!use_spm_weight || fuse_partial_mrope,
              "Wall Action QKV SPM-weight supports only fused partial M-RoPE");
  TORCH_CHECK(!kv_owner_fanout || (!use_spm_weight && fuse_partial_mrope),
              "Wall Action K/V owner fanout requires DDR weights and fused "
              "partial M-RoPE");
  if (fuse_partial_mrope) {
    const auto is_fp16_table = [](const at::Tensor& tensor) {
      return tensor.defined() &&
          tensor.device().type() == c10::DeviceType::PrivateUse1 &&
          tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
          tensor.dim() == 2;
    };
    TORCH_CHECK(
        is_fp16_table(*cos) && is_fp16_table(*sin) &&
            cos->sizes() == sin->sizes() && cos->size(1) == 64 &&
            position >= 0 && position <= cos->size(0) &&
            kM <= cos->size(0) - position,
        "Wall Action QKV partial M-RoPE cos/sin must be matching contiguous "
        "device fp16 [rows,64] tables covering [", position, ",",
        position + kM, ")");
  }

  const KernelId legacy_kernel_id = use_spm_weight
      ? KernelId::WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64_SPM_WEIGHT
      : kv_owner_fanout
          ? KernelId::WALL_ACTION_QKV_OWNER_PARTIAL_MROPE_W8A16_M20N64
          : fuse_partial_mrope
          ? KernelId::WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64
          : KernelId::WALL_ACTION_QKV_W8A16_MULTIBASE_M592N32;
  TORCH_CHECK(!fp16 || fuse_partial_mrope, "FP16 QKV requires fused partial MRoPE");
  const KernelId kernel_id = fp16
      ? (use_spm_weight ? KernelId::WALL_ACTION_QKV_FP16_PARTIAL_MROPE_SPM_WEIGHT_M20N64K128
           : kv_owner_fanout ? KernelId::WALL_ACTION_QKV_FP16_OWNER_PARTIAL_MROPE_M20N64K128
                             : KernelId::WALL_ACTION_QKV_FP16_PARTIAL_MROPE_M20N64K128)
      : legacy_kernel_id;
  std::vector<GraphDdrRegisterOperandSpec> operands;
  if (!use_spm_weight) {
    operands.push_back(GraphDdrRegisterOperandSpec{
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::Int8ModelWeight,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
        q_weight});
  }
  operands.push_back(GraphDdrRegisterOperandSpec{
      GraphDdrRegisterAbi{
          GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
      q_scale});
  if (!use_spm_weight) {
    operands.push_back(GraphDdrRegisterOperandSpec{
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::Int8ModelWeight,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25},
        k_weight});
  }
  operands.push_back(GraphDdrRegisterOperandSpec{
      GraphDdrRegisterAbi{
          GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27},
      k_scale});
  if (!use_spm_weight) {
    operands.push_back(GraphDdrRegisterOperandSpec{
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::Int8ModelWeight,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33},
        v_weight});
  }
  operands.push_back(GraphDdrRegisterOperandSpec{
      GraphDdrRegisterAbi{
          GraphDdrRegisterRole::PerChannelScale,
          GraphDdrRegisterAccess::Read,
          GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35},
      v_scale});
  if (fuse_partial_mrope) {
    operands.push_back(GraphDdrRegisterOperandSpec{
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 10, 11},
        *cos});
    operands.push_back(GraphDdrRegisterOperandSpec{
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 18, 19},
        *sin});
  }
  auto register_writer =
      wall_precision_registers(
          kernel_id, fp16, std::move(operands));

  if (!use_spm_weight) {
    for (const at::Tensor* tensor : {&q_weight, &k_weight, &v_weight}) {
      if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
      TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                  "Wall Action QKV multibase DDR weight is not 256-byte "
                  "aligned");
    }
  }
  for (const at::Tensor* tensor : {&q_scale, &k_scale, &v_scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Wall Action QKV multibase DDR scale is not 256-byte "
                "aligned");
  }
  if (fuse_partial_mrope) {
    rpu_ddr_flush(cos->data_ptr());
    rpu_ddr_flush(sin->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(cos->data_ptr()) % 256 == 0 &&
                    RpuGetDevAddr(sin->data_ptr()) % 256 == 0,
                "Wall Action QKV partial M-RoPE cos/sin DDR operands are "
                "not 256-byte aligned");
  }

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(kernel != nullptr,
              use_spm_weight
                  ? "qkv_parallel_w8a16_multibase_partial_mrope_"
                    "m20n64k128_spm_weight missing from expansion ref"
                  : kv_owner_fanout
                  ? "qkv_parallel_w8a16_multibase_owner_partial_mrope_"
                    "m20n64k128 missing from expansion ref"
                  : fuse_partial_mrope
                  ? "qkv_parallel_w8a16_multibase_partial_mrope_m20n64k128 "
                    "missing from expansion ref"
                  : "qkv_parallel_w8a16_multibase_m592n32k128 missing from "
                    "expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, q_output_spm_addr, q_bias_spm_addr,
          static_cast<uint32_t>(kM), static_cast<uint32_t>(kQPerCore),
          static_cast<uint32_t>(kK), 1, NUM_CORES,
          static_cast<uint32_t>(kK * sizeof(c10::Half)),
          static_cast<uint32_t>(kQPerCore * sizeof(c10::Half)), 0, 1, 1});

  auto set_u32 = [&](int reg, uint32_t value) {
    kernel->set_regs(reg, static_cast<uint16_t>(value));
    kernel->set_regs(reg + 1, static_cast<uint16_t>(value >> 16));
  };
  if (use_spm_weight) {
    set_u32(2, q_weight_spm_addr);
    set_u32(24, k_weight_spm_addr);
    set_u32(32, v_weight_spm_addr);
  }
  set_u32(28, k_bias_spm_addr);
  set_u32(30, k_output_spm_addr);
  set_u32(36, v_bias_spm_addr);
  set_u32(38, v_output_spm_addr);
  if (fuse_partial_mrope) {
    const uint32_t position_u32 = static_cast<uint32_t>(position);
    kernel->set_regs(9, static_cast<uint16_t>(position_u32));
    kernel->set_regs(12, static_cast<uint16_t>(position_u32 >> 16));
    kernel->set_regs(128, static_cast<uint16_t>(0));
  }
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  queue->enqueu_kernel(
      *kernel,
      {static_cast<uint16_t>(fuse_partial_mrope ? 8 : 16), 1, 1}, cores);

  if (kv_owner_fanout) {
    const uint64_t kKvOutputBytes = kM * kKvPerCore * sizeof(c10::Half);
    const uint32_t spm_base0 = SPM_ALLOC.addr(0, 0);
    auto checked_offset = [&](uint32_t address, const char* name) {
      TORCH_CHECK(
          address >= spm_base0 && address % 256 == 0 &&
              static_cast<uint64_t>(address - spm_base0) <=
                  SpmAllocator::SPM_USABLE - kKvOutputBytes,
          "Wall Action K/V owner fanout ", name,
          " must be a 256-byte-aligned local SPM range");
      return address - spm_base0;
    };
    const uint32_t k_offset = checked_offset(k_output_spm_addr, "K output");
    const uint32_t v_offset = checked_offset(v_output_spm_addr, "V output");

    Kernel_t* fanout = GET_KERNEL(KernelId::WALL_ACTION_KV_OWNER_FANOUT_M20);
    TORCH_CHECK(fanout != nullptr,
                "kv_owner_fanout_m20 missing from expansion ref");
    fanout->reset_regs();
    auto set_fanout_u32 = [&](int reg, uint32_t value) {
      fanout->set_regs(reg, static_cast<uint16_t>(value));
      fanout->set_regs(reg + 1, static_cast<uint16_t>(value >> 16));
    };
    set_fanout_u32(0, k_offset);
    set_fanout_u32(2, v_offset);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*fanout, {1, 1, 1}, cores);
  }
}

void rpu_launch_wall_vision_qkv_w8a16_multibase_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& v_weight,
    uint32_t q_output_spm_addr,
    uint32_t k_output_spm_addr,
    uint32_t v_output_spm_addr,
    uint32_t q_bias_spm_addr,
    uint32_t k_bias_spm_addr,
    uint32_t v_bias_spm_addr,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale,
    const WallSortingConfig& sorting) {
  const bool fp16 = q_weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 1u),
              "Wall FP16 Vision transfer requires its cold opt-in");
  constexpr int64_t kImageM = 576;
  constexpr int64_t kK = 1280;
  constexpr int64_t kN = 1280;
  constexpr int64_t kNPerCore = 160;

  auto check_weight = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                    tensor.is_contiguous() && tensor.dim() == 2 &&
                    tensor.size(0) == kN && tensor.size(1) == kK,
                "Wall Vision QKV multibase ", name,
                " weight must be contiguous device int8 [", kN, ",", kK,
                "], got ", tensor.sizes());
  };
  auto check_scale = [&](const at::Tensor& tensor, const char* name) {
    if (fp16) { TORCH_CHECK(!tensor.defined(), "FP16 has no quant scale"); return; }
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf &&
                    tensor.is_contiguous() && tensor.dim() == 1 &&
                    tensor.size(0) == kN,
                "Wall Vision QKV multibase ", name,
                " scale must be contiguous device fp16 [", kN, "], got ",
                tensor.sizes());
  };
  check_weight(q_weight, "Q");
  check_weight(k_weight, "K");
  check_weight(v_weight, "V");
  check_scale(q_scale, "Q");
  check_scale(k_scale, "K");
  check_scale(v_scale, "V");
  TORCH_CHECK(input_spm_addr != 0 && q_output_spm_addr != 0 &&
                  k_output_spm_addr != 0 && v_output_spm_addr != 0 &&
                  q_output_spm_addr != k_output_spm_addr &&
                  q_output_spm_addr != v_output_spm_addr &&
                  k_output_spm_addr != v_output_spm_addr &&
                  input_spm_addr != q_output_spm_addr &&
                  input_spm_addr != k_output_spm_addr &&
                  input_spm_addr != v_output_spm_addr,
              "Wall Vision QKV multibase requires distinct non-zero input and "
              "Q/K/V SPM buffers");
  TORCH_CHECK(q_bias_spm_addr != 0 && k_bias_spm_addr != 0 &&
                  v_bias_spm_addr != 0,
              "Wall Vision QKV multibase requires all three persistent SPM "
              "biases");

  const KernelId kernel_id = fp16
      ? KernelId::WALL_VISION_QKV_FP16_MULTIBASE_M192X3N96K128
      : KernelId::WALL_VISION_QKV_W8A16_MULTIBASE_M192X3N96;
  auto register_writer =
      wall_precision_registers(
          kernel_id, fp16,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
                  q_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
                  q_scale},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25},
                  k_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27},
                  k_scale},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33},
                  v_weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35},
                  v_scale}});

  for (const at::Tensor* tensor :
       {&q_weight, &k_weight, &v_weight, &q_scale, &k_scale, &v_scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Wall Vision QKV multibase DDR operand is not 256-byte "
                "aligned");
  }

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      "qkv_parallel_w8a16_multibase_m192x3n96k128 missing from expansion "
      "ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, q_output_spm_addr, q_bias_spm_addr,
          static_cast<uint32_t>(kImageM), static_cast<uint32_t>(kNPerCore),
          static_cast<uint32_t>(kK), 1, NUM_CORES,
          static_cast<uint32_t>(kK * sizeof(c10::Half)),
          static_cast<uint32_t>(kNPerCore * sizeof(c10::Half)), 0, 1, 1});

  auto set_u32 = [&](int reg, uint32_t value) {
    kernel->set_regs(reg, static_cast<uint16_t>(value));
    kernel->set_regs(reg + 1, static_cast<uint16_t>(value >> 16));
  };
  set_u32(28, k_bias_spm_addr);
  set_u32(30, k_output_spm_addr);
  set_u32(36, v_bias_spm_addr);
  set_u32(38, v_output_spm_addr);
  constexpr uint32_t kInputImageBytes =
      kImageM * kK * sizeof(c10::Half);
  constexpr uint32_t kOutputImageBytes =
      kImageM * kNPerCore * sizeof(c10::Half);
  set_u32(40, input_spm_addr + kInputImageBytes);
  set_u32(42, input_spm_addr + 2 * kInputImageBytes);
  set_u32(44, q_output_spm_addr + kOutputImageBytes);
  set_u32(46, q_output_spm_addr + 2 * kOutputImageBytes);
  set_u32(48, k_output_spm_addr + kOutputImageBytes);
  set_u32(50, k_output_spm_addr + 2 * kOutputImageBytes);
  set_u32(52, v_output_spm_addr + kOutputImageBytes);
  set_u32(54, v_output_spm_addr + 2 * kOutputImageBytes);
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  queue->enqueu_kernel(*kernel, {6, 3, 1}, cores);
}

void rpu_launch_wall_prefill_down_w8a16_c320x2_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& weight,
    uint32_t output0_spm_addr,
    uint32_t output1_spm_addr,
    const at::Tensor& scale,
    const WallSortingConfig& sorting) {
  const bool fp16 = weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 2u),
              "Wall FP16 Prefill transfer requires its cold opt-in");
  constexpr int64_t kM = 640;
  constexpr int64_t kN = 2048;
  constexpr int64_t kK = 11008;
  constexpr int64_t kLocalK = kK / NUM_CORES;

  TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  weight.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                  weight.is_contiguous() && weight.dim() == 2 &&
                  weight.size(0) == kN && weight.size(1) == kK,
              "Wall Prefill Down C320x2 weight must be contiguous device int8 [",
              kN, ",", kK, "], got ", weight.sizes());
  if (fp16) { TORCH_CHECK(!scale.defined(), "FP16 has no quant scale"); } else {
  TORCH_CHECK(scale.device().type() == c10::DeviceType::PrivateUse1 &&
                  scale.scalar_type() == at::kHalf &&
                  scale.is_contiguous() && scale.dim() == 1 &&
                  scale.size(0) == kN,
              "Wall Prefill Down C320x2 scale must be contiguous device fp16 [",
              kN, "], got ", scale.sizes());
  }
  TORCH_CHECK(input_spm_addr != 0 && output0_spm_addr != 0 &&
                  output1_spm_addr != 0 &&
                  input_spm_addr != output0_spm_addr &&
                  input_spm_addr != output1_spm_addr &&
                  output0_spm_addr != output1_spm_addr,
              "Wall Prefill Down C320x2 requires distinct non-zero input and "
              "output SPM buffers");

  const KernelId kernel_id = fp16
      ? KernelId::WALL_PREFILL_DOWN_FP16_C320X2_N64K128
      : KernelId::WALL_DOWN_WEIGHT_OUTER_W8A16_C320X2_N64;
  auto register_writer =
      wall_precision_registers(
          kernel_id, fp16,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
                  weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
                  scale}});

  for (const at::Tensor* tensor : {&weight, &scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Wall Prefill Down C320x2 DDR operand is not 256-byte aligned");
  }

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      KERNEL_ID_NAMES[static_cast<size_t>(kernel_id)],
      " missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm_addr, 0, output0_spm_addr, 0,
          static_cast<uint32_t>(kM), static_cast<uint32_t>(kN),
          static_cast<uint32_t>(kLocalK), 0, NUM_CORES,
          static_cast<uint32_t>(kLocalK * sizeof(c10::Half)),
          static_cast<uint32_t>(kN * sizeof(c10::Half)), 0,
          /*partition=*/0, /*dtype_mode=*/1});
  kernel->set_regs(24, static_cast<uint16_t>(output1_spm_addr));
  kernel->set_regs(25, static_cast<uint16_t>(output1_spm_addr >> 16));
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  queue->enqueu_kernel(*kernel, {32, 1, 1}, cores);
}

void rpu_launch_wall_prefill_oproj_w8a16_c320x2_spm_kernel(
    uint32_t input0_spm_addr,
    const at::Tensor& weight,
    uint32_t output0_spm_addr,
    uint32_t output1_spm_addr,
    const at::Tensor& scale,
    const WallSortingConfig& sorting) {
  const bool fp16 = weight.scalar_type() == at::kHalf;
  TORCH_CHECK(!fp16 || (sorting.fp16_transfer_mask & 2u),
              "Wall FP16 Prefill transfer requires its cold opt-in");
  constexpr int64_t kM = 640;
  constexpr int64_t kN = 2048;
  constexpr int64_t kK = 2048;
  constexpr int64_t kLocalK = kK / NUM_CORES;

  TORCH_CHECK(weight.device().type() == c10::DeviceType::PrivateUse1 &&
                  weight.scalar_type() == (fp16 ? at::kHalf : at::kChar) &&
                  weight.is_contiguous() && weight.dim() == 2 &&
                  weight.size(0) == kN && weight.size(1) == kK,
              "Wall Prefill O-proj C320x2 weight must be contiguous device int8 [",
              kN, ",", kK, "], got ", weight.sizes());
  if (fp16) { TORCH_CHECK(!scale.defined(), "FP16 has no quant scale"); } else {
  TORCH_CHECK(scale.device().type() == c10::DeviceType::PrivateUse1 &&
                  scale.scalar_type() == at::kHalf &&
                  scale.is_contiguous() && scale.dim() == 1 &&
                  scale.size(0) == kN,
              "Wall Prefill O-proj C320x2 scale must be contiguous device fp16 [",
              kN, "], got ", scale.sizes());
  }
  TORCH_CHECK(input0_spm_addr != 0 && output0_spm_addr != 0 &&
                  output1_spm_addr != 0 &&
                  input0_spm_addr != output0_spm_addr &&
                  input0_spm_addr != output1_spm_addr &&
                  output0_spm_addr != output1_spm_addr,
              "Wall Prefill O-proj C320x2 requires distinct non-zero input "
              "and output SPM buffers");

  const bool activation_reuse = sorting.prefill_row_reuse;
  const KernelId kernel_id = fp16
      ? (activation_reuse ? KernelId::WALL_PREFILL_ROW_FP16_M160X4N96K128
                          : KernelId::WALL_PREFILL_OPROJ_FP16_C320X2_N64K128)
      : activation_reuse
      ? KernelId::WALL_PREFILL_ROW_W8A16_M160X4N96K128
      : KernelId::WALL_OPROJ_WEIGHT_OUTER_W8A16_C320X2_N64;
  auto register_writer =
      wall_precision_registers(
          kernel_id, fp16,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::Int8ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
                  weight},
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::PerChannelScale,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
                  scale}});

  for (const at::Tensor* tensor : {&weight, &scale}) {
    if (!tensor->defined()) continue;
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(
        RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
        "Wall Prefill O-proj C320x2 DDR operand is not 256-byte aligned");
  }

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      KERNEL_ID_NAMES[static_cast<size_t>(kernel_id)],
      " missing from expansion ref");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(
      *kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input0_spm_addr, 0, output0_spm_addr, 0,
          static_cast<uint32_t>(kM), static_cast<uint32_t>(kN),
          static_cast<uint32_t>(kLocalK), 0, NUM_CORES,
          static_cast<uint32_t>(kLocalK * sizeof(c10::Half)),
          static_cast<uint32_t>(kN * sizeof(c10::Half)), 0,
          /*partition=*/0, /*dtype_mode=*/1});
  kernel->set_regs(24, static_cast<uint16_t>(output1_spm_addr));
  kernel->set_regs(25, static_cast<uint16_t>(output1_spm_addr >> 16));
  register_writer.write(*kernel);

  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  const std::vector<uint8_t> cores{0, 1, 2, 3, 4, 5, 6, 7};
  if (activation_reuse) {
    queue->enqueu_kernel(*kernel, {22, 1, 1}, cores);
  } else {
    queue->enqueu_kernel(*kernel, {32, 1, 1}, cores);
  }
}


static void launch_pi05_denoise_gate_up_geglu_nvfp4_m50(
    uint32_t input, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer,
    KernelId id) {
  constexpr int64_t kM = 50, kN = 4096, kK = 1024, kLocalN = 512;
  static_assert(NUM_CORES == 8, "Pi NVFP4 GeGLU requires TP8");
  TORCH_CHECK(layer < 18, "Pi NVFP4 GeGLU requires layer<18");
  for (const auto* weight : {&gate_weight, &up_weight}) {
    TORCH_CHECK(weight->defined() && weight->device().type() == at::kPrivateUse1 &&
                    weight->scalar_type() == at::kByte && weight->is_contiguous() &&
                    weight->storage_offset() == 0 &&
                    weight->sizes() == at::IntArrayRef({kN, kK / 2}) &&
                    weight->nbytes() == kN * kK / 2,
                "Pi NVFP4 GeGLU requires packed device uint8 [4096,512] weights");
  }
  for (const auto* scale : {&gate_scale, &up_scale}) {
    TORCH_CHECK(scale->defined() && scale->device().type() == at::kPrivateUse1 &&
                    scale->scalar_type() == at::kByte && scale->is_contiguous() &&
                    scale->storage_offset() == 0 &&
                    scale->sizes() == at::IntArrayRef({kK / 16, kN}) &&
                    scale->nbytes() == kK / 16 * kN,
                "Pi NVFP4 GeGLU requires striped-v2 FP8 device uint8 [64,4096] scales");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi NVFP4 GeGLU requires initialized SPM");
  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  const std::pair<uint64_t, uint64_t> slabs[] = {
      {input, kM * kK * sizeof(c10::Half)},
      {output, kM * kLocalN * sizeof(c10::Half)},
      {gate_tensor_scale, 24 * sizeof(float)},
      {up_tensor_scale, 24 * sizeof(float)}};
  for (size_t i = 0; i < 4; ++i) {
    const auto [address, bytes] = slabs[i];
    TORCH_CHECK(address != 0 && address >= spm_base && address % 256 == 0 &&
                    address - spm_base + bytes <= SpmAllocator::SPM_USABLE,
                "Pi NVFP4 GeGLU requires aligned in-range absolute SPM slabs");
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(address + bytes <= slabs[j].first ||
                      slabs[j].first + slabs[j].second <= address,
                  "Pi NVFP4 GeGLU requires disjoint input/output/tensor-scale slabs");
    }
  }
  for (const auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    rpu_ddr_flush(tensor->data_ptr());
    const uint64_t address = RpuGetDevAddr(tensor->data_ptr());
    TORCH_CHECK(address != 0 && address % 256 == 0 && (address >> 8) <= UINT32_MAX,
                "Pi NVFP4 GeGLU requires aligned shift8 DDR operands");
  }
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, gate_scale},
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, up_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, up_scale}});
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Pi NVFP4 GeGLU kernel missing from expansion REF");
  kernel->reset_regs();
  const auto u16 = [&](int reg, uint16_t value) { kernel->set_regs(reg, value); };
  const auto u32 = [&](int reg, uint32_t value) { u16(reg, value & 0xffff); u16(reg + 1, value >> 16); };
  u32(0, input); u32(4, output); u32(6, 0);
  u16(8, kM); u32(10, kLocalN); u16(12, kK); u16(13, kK / 64);
  u16(14, 0); u16(15, NUM_CORES); u32(16, kK * 2); u32(18, kLocalN * 2);
  u32(22, NUM_CORES * kLocalN * 2); u32(24, gate_tensor_scale);
  u16(26, 1); u16(27, layer); u32(36, up_tensor_scale);
  u16(64, 8); u16(65, 1); u16(66, 1);
  if (id == KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128) {
    // The Up-to-GeGLU warp barrier owns core counters 0/1 for this launch.
    // Atomic counters are 32-bit pairs of halfword registers, not SCM slots.
    kernel->set_regs(128, uint32_t{0});
    kernel->set_regs(130, uint32_t{0});
  }
  writer.write(*kernel);
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0,1,2,3,4,5,6,7});
}

void rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_m50_kernel(
    uint32_t input, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer) {
  launch_pi05_denoise_gate_up_geglu_nvfp4_m50(
      input, gate_weight, up_weight, output, gate_scale, up_scale,
      gate_tensor_scale, up_tensor_scale, layer,
      KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128);
}

void rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(
    uint32_t input, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer) {
  launch_pi05_denoise_gate_up_geglu_nvfp4_m50(
      input, gate_weight, up_weight, output, gate_scale, up_scale,
      gate_tensor_scale, up_tensor_scale, layer,
      KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128);
}

static void launch_wall_base_nvfp4_gate_up_parallel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate, uint32_t up, const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_ts, uint32_t up_ts, int64_t scale_count, uint16_t layer, bool fuse_silu, bool shared_input) {
  constexpr int64_t kM = 32, kN = 2048, kK = 1024, kLocalN = 256, kCores = 8;
  TORCH_CHECK(scale_count == 40 && layer < 35,
              "Wall parallel Gate/Up requires FP32[40] scales and nonfinal layer");
  const at::Tensor* tensors[] = {&gate_weight, &up_weight, &gate_scale, &up_scale};
  for (int i = 0; i < 4; ++i) {
    const auto& t = *tensors[i];
    TORCH_CHECK(t.defined() && t.device().type() == at::kPrivateUse1 &&
                    t.device() == gate_weight.device() && t.scalar_type() == at::kByte &&
                    t.is_contiguous() && t.storage_offset() == 0 &&
                    t.sizes() == (i < 2 ? at::IntArrayRef({kN, kK / 2})
                                       : at::IntArrayRef({kK / 16, kN})),
                "Wall parallel Gate/Up requires exact packed weights and striped FP8 scales");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Wall parallel Gate/Up requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::pair<uint64_t, uint64_t> slabs[] = {
      {input, kM * kK * 2}, {gate, kM * kLocalN * 2}, {up, kM * kLocalN * 2},
      {gate_ts, 40 * sizeof(float)}, {up_ts, 40 * sizeof(float)}};
  for (int i = 0; i < 5; ++i) {
    const auto [address, bytes] = slabs[i];
    TORCH_CHECK(address != 0 && address >= base && address % 32 == 0 &&
                    address - base <= SpmAllocator::SPM_USABLE - bytes,
                "Wall parallel Gate/Up requires full aligned absolute SPM slabs");
    for (int core = 0; core < kCores; ++core) {
      TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) +
                      address - base + bytes - 1 <= UINT32_MAX,
                  "Wall parallel Gate/Up SPM range exceeds register ABI");
    }
    for (int j = 0; j < i; ++j) {
      TORCH_CHECK(address + bytes <= slabs[j].first ||
                      slabs[j].first + slabs[j].second <= address,
                  "Wall parallel Gate/Up requires disjoint input/output/scale slabs");
    }
  }
  std::array<uint64_t, 4> addresses{};
  for (int i = 0; i < 4; ++i) {
    addresses[i] = RpuGetDevAddr(tensors[i]->data_ptr());
    TORCH_CHECK(addresses[i] != 0 && addresses[i] % 256 == 0 &&
                    (addresses[i] >> 8) <= UINT32_MAX &&
                    tensors[i]->nbytes() - 1 <= (uint64_t{1} << 40) - 1 - addresses[i],
                "Wall parallel Gate/Up requires aligned shift8 DDR operands");
    for (int j = 0; j < i; ++j) {
      TORCH_CHECK(addresses[i] + tensors[i]->nbytes() <= addresses[j] ||
                      addresses[j] + tensors[j]->nbytes() <= addresses[i],
                  "Wall parallel Gate/Up DDR operands must not overlap");
    }
  }
  TORCH_CHECK(!shared_input || fuse_silu, "Wall shared Gate/Up requires fused SiLU");
  const auto id = shared_input ? KernelId::WALL_BASE_NVFP4_GATE_UP_SHARED_SILU_H32
                 : fuse_silu ? KernelId::WALL_BASE_NVFP4_GATE_UP_PARALLEL_SILU_H32
                            : KernelId::WALL_BASE_NVFP4_GATE_UP_PARALLEL_H32;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "Wall parallel Gate/Up payload unavailable");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, gate_scale},
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, up_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, up_scale}});
  for (const auto* t : tensors) rpu_ddr_flush(t->data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Wall parallel Gate/Up kernel missing");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
      {input, 0, gate, 0, kM, kLocalN, kK, 0, kCores,
       kK * 2, kLocalN * 2, 0, 1, 0}, kCores * kLocalN * 2, gate_ts, layer});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, up_ts);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 38, up);
  if (fuse_silu) {
    // Retire all resident projections before switching to the SiLU phase.
    // Shared-input has four blocks; the original parallel route has eight.
    TORCH_CHECK(kernel->set_regs(128, uint32_t{0}) == 0 &&
                    kernel->set_regs(130, uint32_t{0}) == 0,
                "Wall parallel Gate/Up/SiLU counter reset failed");
  }
  writer.write(*kernel);
  auto* queue = GET_QUEUE(kCores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {4, 1, static_cast<uint16_t>(shared_input ? 1 : 2)}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_wall_base_nvfp4_gate_up_parallel_spm_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate, uint32_t up, const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_ts, uint32_t up_ts, int64_t scale_count, uint16_t layer) {
  launch_wall_base_nvfp4_gate_up_parallel(input, gate_weight, up_weight, gate, up,
      gate_scale, up_scale, gate_ts, up_ts, scale_count, layer, false, false);
}

void rpu_launch_wall_base_nvfp4_gate_up_parallel_silu_spm_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate, uint32_t up, const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_ts, uint32_t up_ts, int64_t scale_count, uint16_t layer) {
  launch_wall_base_nvfp4_gate_up_parallel(input, gate_weight, up_weight, gate, up,
      gate_scale, up_scale, gate_ts, up_ts, scale_count, layer, true, false);
}

void rpu_launch_wall_base_nvfp4_gate_up_shared_silu_spm_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate, uint32_t up, const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_ts, uint32_t up_ts, int64_t scale_count, uint16_t layer) {
  launch_wall_base_nvfp4_gate_up_parallel(input, gate_weight, up_weight, gate, up,
      gate_scale, up_scale, gate_ts, up_ts, scale_count, layer, true, true);
}

void rpu_launch_wall_base_nvfp4_qkv_bias_rope_spm_kernel(
    uint32_t input, const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& v_weight, uint32_t q, uint32_t k, uint32_t v,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale,
    uint32_t q_bias, uint32_t k_bias, uint32_t v_bias,
    uint32_t q_ts, uint32_t k_ts, uint32_t v_ts,
    int64_t scale_count, uint16_t layer,
    const at::Tensor& cos, const at::Tensor& sin, int64_t position) {
  constexpr int64_t kM = 32, kK = 1024, kCores = 8;
  TORCH_CHECK(scale_count == 40 && layer < 36,
              "Wall Q/K/V bias/RoPE requires FP32[40] scales and layer<36");
  TORCH_CHECK(wall_base_nvfp4_fused_prefix_supported(position),
              "Wall Q/K/V bias/RoPE requires an admitted H32 prefix");
  for (const auto* table : {&cos, &sin}) {
    TORCH_CHECK(table->defined() && table->device() == q_weight.device() &&
                    table->device().type() == at::kPrivateUse1 &&
                    table->scalar_type() == at::kHalf && table->is_contiguous() &&
                    table->storage_offset() == 0 && table->dim() == 2 &&
                    table->size(1) == 64 && position <= table->size(0) &&
                    kM <= table->size(0) - position,
                "Wall Q/K/V bias/RoPE requires bounded owned FP16 tables [rows,64]");
  }
  TORCH_CHECK(cos.sizes() == sin.sizes(), "Wall Q/K/V bias/RoPE tables must match");
  const at::Tensor* tensors[] = {&q_weight, &k_weight, &v_weight, &q_scale, &k_scale, &v_scale, &cos, &sin};
  for (int i = 0; i < 6; ++i) {
    const auto& t = *tensors[i];
    const int64_t n = i % 3 == 0 ? 2048 : 1024;
    TORCH_CHECK(t.defined() && t.device().type() == at::kPrivateUse1 &&
                    t.device() == q_weight.device() && t.scalar_type() == at::kByte &&
                    t.is_contiguous() && t.storage_offset() == 0 &&
                    t.sizes() == (i < 3 ? at::IntArrayRef({n, kK / 2})
                                       : at::IntArrayRef({kK / 16, n})),
                "Wall Q/K/V bias/RoPE requires exact packed weights and striped FP8 scales");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Wall Q/K/V bias/RoPE requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::pair<uint64_t, uint64_t> slabs[] = {
      {input, kM * kK * 2}, {q, kM * 256 * 2},
      {k, kM * 128 * 2}, {v, kM * 128 * 2},
      {q_bias, 256 * 2}, {k_bias, 128 * 2}, {v_bias, 128 * 2},
      {q_ts, 40 * sizeof(float)}, {k_ts, 40 * sizeof(float)}, {v_ts, 40 * sizeof(float)}};
  for (int i = 0; i < 10; ++i) {
    const auto [address, bytes] = slabs[i];
    TORCH_CHECK(address != 0 && address >= base && address % 32 == 0 &&
                    address - base <= SpmAllocator::SPM_USABLE - bytes,
                "Wall Q/K/V bias/RoPE requires full aligned absolute SPM slabs");
    for (int core = 0; core < kCores; ++core) {
      TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) +
                      address - base + bytes - 1 <= UINT32_MAX,
                  "Wall Q/K/V bias/RoPE SPM range exceeds register ABI");
    }
    for (int j = 0; j < i; ++j) {
      TORCH_CHECK(address + bytes <= slabs[j].first ||
                      slabs[j].first + slabs[j].second <= address,
                  "Wall Q/K/V bias/RoPE requires disjoint input/output/bias/scale slabs");
    }
  }
  std::array<uint64_t, 8> addresses{};
  for (int i = 0; i < 8; ++i) {
    addresses[i] = RpuGetDevAddr(tensors[i]->data_ptr());
    TORCH_CHECK(addresses[i] != 0 && addresses[i] % 256 == 0 &&
                    (addresses[i] >> 8) <= UINT32_MAX &&
                    ((addresses[i] + tensors[i]->nbytes() - 1) >> 8) <= UINT32_MAX,
                "Wall Q/K/V bias/RoPE requires aligned shift8 DDR operands");
    for (int j = 0; j < i; ++j) {
      TORCH_CHECK(addresses[i] + tensors[i]->nbytes() <= addresses[j] ||
                      addresses[j] + tensors[j]->nbytes() <= addresses[i],
                  "Wall Q/K/V bias/RoPE DDR operands must not overlap");
    }
  }
  constexpr auto id = KernelId::WALL_BASE_NVFP4_QKV_BIAS_ROPE_H32;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "Wall Q/K/V bias/RoPE payload unavailable");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, q_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, q_scale},
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, k_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, k_scale},
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 42, 43}, v_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 44, 45}, v_scale},
          {{GraphDdrRegisterRole::SemanticInput, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 52, 53}, cos},
          {{GraphDdrRegisterRole::SemanticInput, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 54, 55}, sin}});
  for (const auto* t : tensors) rpu_ddr_flush(t->data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Wall Q/K/V bias/RoPE kernel missing");
  kernel->reset_regs();
  // Q is the canonical N256 shard. The payload selects N128, output stride256
  // and FP8 stride2048 for K/V, all with zero high halves and common N64 tiles.
  rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
      {input, 0, q, q_bias, kM, 256, kK, 1, kCores,
       kK * 2, 512, 0, 1, 0}, 4096, q_ts, layer});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, k_ts);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 38, k);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 40, k_bias);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 46, v_ts);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 48, v);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 50, v_bias);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 56, static_cast<uint32_t>(position));
  TORCH_CHECK(kernel->set_regs(64, uint16_t{8}) == 0 &&
                  kernel->set_regs(65, uint16_t{1}) == 0 &&
                  kernel->set_regs(66, uint16_t{1}) == 0,
              "Wall Q/K/V bias/RoPE grid register write failed");
  // Q4 + K2 + V2 are exactly eight resident blocks. Retire GEMM before bias;
  // counter1 then retires all bias stores before in-place Q/K RoPE. Each
  // packet, including captured REPLAY, resets both full atomic pairs.
  TORCH_CHECK(kernel->set_regs(128, uint32_t{0}) == 0 &&
                  kernel->set_regs(130, uint32_t{0}) == 0,
              "Wall Q/K/V bias/RoPE counter reset failed");
  writer.write(*kernel);
  auto* queue = GET_QUEUE(kCores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_wall_base_nvfp4_qkv_parallel_bias_spm_kernel(
    uint32_t input, const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& v_weight, uint32_t q, uint32_t k, uint32_t v,
    const at::Tensor& q_scale, const at::Tensor& k_scale, const at::Tensor& v_scale,
    uint32_t q_bias, uint32_t k_bias, uint32_t v_bias,
    uint32_t q_ts, uint32_t k_ts, uint32_t v_ts,
    int64_t scale_count, uint16_t layer) {
  constexpr int64_t kM = 32, kK = 1024, kCores = 8;
  TORCH_CHECK(scale_count == 40 && layer < 36,
              "Wall parallel Q/K/V bias requires FP32[40] scales and layer<36");
  const at::Tensor* tensors[] = {&q_weight, &k_weight, &v_weight, &q_scale, &k_scale, &v_scale};
  for (int i = 0; i < 6; ++i) {
    const auto& t = *tensors[i];
    const int64_t n = i % 3 == 0 ? 2048 : 1024;
    TORCH_CHECK(t.defined() && t.device().type() == at::kPrivateUse1 &&
                    t.device() == q_weight.device() && t.scalar_type() == at::kByte &&
                    t.is_contiguous() && t.storage_offset() == 0 &&
                    t.sizes() == (i < 3 ? at::IntArrayRef({n, kK / 2})
                                       : at::IntArrayRef({kK / 16, n})),
                "Wall parallel Q/K/V bias requires exact packed weights and striped FP8 scales");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Wall parallel Q/K/V bias requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::pair<uint64_t, uint64_t> slabs[] = {
      {input, kM * kK * 2}, {q, kM * 256 * 2},
      {k, kM * 128 * 2}, {v, kM * 128 * 2},
      {q_bias, 256 * 2}, {k_bias, 128 * 2}, {v_bias, 128 * 2},
      {q_ts, 40 * sizeof(float)}, {k_ts, 40 * sizeof(float)}, {v_ts, 40 * sizeof(float)}};
  for (int i = 0; i < 10; ++i) {
    const auto [address, bytes] = slabs[i];
    TORCH_CHECK(address != 0 && address >= base && address % 32 == 0 &&
                    address - base <= SpmAllocator::SPM_USABLE - bytes,
                "Wall parallel Q/K/V bias requires full aligned absolute SPM slabs");
    for (int core = 0; core < kCores; ++core) {
      TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) +
                      address - base + bytes - 1 <= UINT32_MAX,
                  "Wall parallel Q/K/V bias SPM range exceeds register ABI");
    }
    for (int j = 0; j < i; ++j) {
      TORCH_CHECK(address + bytes <= slabs[j].first ||
                      slabs[j].first + slabs[j].second <= address,
                  "Wall parallel Q/K/V bias requires disjoint input/output/bias/scale slabs");
    }
  }
  std::array<uint64_t, 6> addresses{};
  for (int i = 0; i < 6; ++i) {
    addresses[i] = RpuGetDevAddr(tensors[i]->data_ptr());
    TORCH_CHECK(addresses[i] != 0 && addresses[i] % 256 == 0 &&
                    (addresses[i] >> 8) <= UINT32_MAX,
                "Wall parallel Q/K/V bias requires aligned shift8 DDR operands");
    for (int j = 0; j < i; ++j) {
      TORCH_CHECK(addresses[i] + tensors[i]->nbytes() <= addresses[j] ||
                      addresses[j] + tensors[j]->nbytes() <= addresses[i],
                  "Wall parallel Q/K/V bias DDR operands must not overlap");
    }
  }
  constexpr auto id = KernelId::WALL_BASE_NVFP4_QKV_PARALLEL_BIAS_H32;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "Wall parallel Q/K/V bias payload unavailable");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, q_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, q_scale},
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, k_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, k_scale},
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 42, 43}, v_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 44, 45}, v_scale}});
  for (const auto* t : tensors) rpu_ddr_flush(t->data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Wall parallel Q/K/V bias kernel missing");
  kernel->reset_regs();
  // Q is the canonical N256 shard. The payload selects N128, output stride256
  // and FP8 stride2048 for K/V, all with zero high halves and common N64 tiles.
  rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
      {input, 0, q, q_bias, kM, 256, kK, 1, kCores,
       kK * 2, 512, 0, 1, 0}, 4096, q_ts, layer});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, k_ts);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 38, k);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 40, k_bias);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 46, v_ts);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 48, v);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 50, v_bias);
  TORCH_CHECK(kernel->set_regs(64, uint16_t{8}) == 0 &&
                  kernel->set_regs(65, uint16_t{1}) == 0 &&
                  kernel->set_regs(66, uint16_t{1}) == 0,
              "Wall parallel Q/K/V bias grid register write failed");
  // Q4 + K2 + V2 are exactly eight resident blocks. Retire GEMM before bias;
  // each packet, including captured REPLAY, resets both full atomic pairs.
  TORCH_CHECK(kernel->set_regs(128, uint32_t{0}) == 0 &&
                  kernel->set_regs(130, uint32_t{0}) == 0,
              "Wall parallel Q/K/V bias counter reset failed");
  writer.write(*kernel);
  auto* queue = GET_QUEUE(kCores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_wall_base_nvfp4_kv_parallel_bias_spm_kernel(
    uint32_t input, const at::Tensor& k_weight, const at::Tensor& v_weight,
    uint32_t k, uint32_t v, const at::Tensor& k_scale, const at::Tensor& v_scale,
    uint32_t k_bias, uint32_t v_bias, uint32_t k_ts, uint32_t v_ts,
    int64_t scale_count, uint16_t layer) {
  constexpr int64_t kM = 32, kN = 1024, kK = 1024, kLocalN = 128, kCores = 8;
  TORCH_CHECK(scale_count == 40 && layer < 36,
              "Wall parallel K/V bias requires FP32[40] scales and layer<36");
  const at::Tensor* tensors[] = {&k_weight, &v_weight, &k_scale, &v_scale};
  for (int i = 0; i < 4; ++i) {
    const auto& t = *tensors[i];
    TORCH_CHECK(t.defined() && t.device().type() == at::kPrivateUse1 &&
                    t.device() == k_weight.device() && t.scalar_type() == at::kByte &&
                    t.is_contiguous() && t.storage_offset() == 0 &&
                    t.sizes() == (i < 2 ? at::IntArrayRef({kN, kK / 2})
                                       : at::IntArrayRef({kK / 16, kN})),
                "Wall parallel K/V bias requires exact packed weights and striped FP8 scales");
  }
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Wall parallel K/V bias requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const std::pair<uint64_t, uint64_t> slabs[] = {
      {input, kM * kK * 2}, {k, kM * kLocalN * 2}, {v, kM * kLocalN * 2},
      {k_ts, 40 * sizeof(float)}, {v_ts, 40 * sizeof(float)},
      {k_bias, kLocalN * 2}, {v_bias, kLocalN * 2}};
  for (int i = 0; i < 7; ++i) {
    const auto [address, bytes] = slabs[i];
    TORCH_CHECK(address != 0 && address >= base && address % 32 == 0 &&
                    address - base <= SpmAllocator::SPM_USABLE - bytes,
                "Wall parallel K/V bias requires full aligned absolute SPM slabs");
    for (int core = 0; core < kCores; ++core) {
      TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) +
                      address - base + bytes - 1 <= UINT32_MAX,
                  "Wall parallel K/V bias SPM range exceeds register ABI");
    }
    for (int j = 0; j < i; ++j) {
      TORCH_CHECK(address + bytes <= slabs[j].first ||
                      slabs[j].first + slabs[j].second <= address,
                  "Wall parallel K/V bias requires disjoint input/output/scale slabs");
    }
  }
  std::array<uint64_t, 4> addresses{};
  for (int i = 0; i < 4; ++i) {
    addresses[i] = RpuGetDevAddr(tensors[i]->data_ptr());
    TORCH_CHECK(addresses[i] != 0 && addresses[i] % 256 == 0 &&
                    (addresses[i] >> 8) <= UINT32_MAX,
                "Wall parallel K/V bias requires aligned shift8 DDR operands");
    for (int j = 0; j < i; ++j) {
      TORCH_CHECK(addresses[i] + tensors[i]->nbytes() <= addresses[j] ||
                      addresses[j] + tensors[j]->nbytes() <= addresses[i],
                  "Wall parallel K/V bias DDR operands must not overlap");
    }
  }
  constexpr auto id = KernelId::WALL_BASE_NVFP4_KV_PARALLEL_BIAS_H32;
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "Wall parallel K/V bias payload unavailable");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, k_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, k_scale},
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, v_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, v_scale}});
  for (const auto* t : tensors) rpu_ddr_flush(t->data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Wall parallel K/V bias kernel missing");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
      {input, 0, k, k_bias, kM, kLocalN, kK, 1, kCores,
       kK * 2, kLocalN * 2, 0, 1, 0}, kCores * kLocalN * 2, k_ts, layer});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, v_ts);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 38, v);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 40, v_bias);
  TORCH_CHECK(kernel->set_regs(64, uint16_t{4}) == 0 &&
                  kernel->set_regs(65, uint16_t{1}) == 0 &&
                  kernel->set_regs(66, uint16_t{2}) == 0,
              "Wall parallel K/V bias grid register write failed");
  // All eight resident blocks retire GEMM before bias; reset both atomic pairs
  // in every launch packet, including captured REPLAY.
  TORCH_CHECK(kernel->set_regs(128, uint32_t{0}) == 0 &&
                  kernel->set_regs(130, uint32_t{0}) == 0,
              "Wall parallel K/V bias counter reset failed");
  writer.write(*kernel);
  auto* queue = GET_QUEUE(kCores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {4, 1, 2}, {0, 1, 2, 3, 4, 5, 6, 7});
}

static void launch_wall_base_nvfp4_linear_ring_norm(
    uint32_t input, const at::Tensor& weight, uint32_t partial,
    uint32_t residual, uint32_t raw_output, uint32_t normalized_output,
    uint32_t gamma, const at::Tensor& scale, uint32_t tensor_scale,
    int64_t tensor_scale_elems, uint16_t layer, double eps, KernelId id) {
  constexpr int64_t kM = 32, kN = 1024, kK = 2048, kLocalK = 256;
  constexpr int64_t kLayers = 36, kScaleElems = 40;
  constexpr uint32_t kFullBytes = kM * kN * sizeof(c10::Half);
  static_assert(NUM_CORES == 8, "Wall NVFP4 Linear/ring/norm requires TP8");
  TORCH_CHECK(layer < kLayers && tensor_scale_elems == kScaleElems &&
                  std::isfinite(eps) && eps > 0.0,
              "Wall NVFP4 Linear/ring/norm requires layer<36, FP32[40] and positive finite epsilon");
  TORCH_CHECK(weight.defined() && scale.defined() &&
                  weight.device().type() == at::kPrivateUse1 &&
                  scale.device() == weight.device() &&
                  weight.scalar_type() == at::kByte && scale.scalar_type() == at::kByte &&
                  weight.is_contiguous() && scale.is_contiguous() &&
                  weight.storage_offset() == 0 && scale.storage_offset() == 0 &&
                  weight.sizes() == at::IntArrayRef({kN, kK / 2}) &&
                  scale.sizes() == at::IntArrayRef({kK / 16, kN}) &&
                  weight.nbytes() == kN * kK / 2 && scale.nbytes() == kK / 16 * kN,
              "Wall NVFP4 Linear/ring/norm requires exact packed device weight and striped FP8 scale");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Wall NVFP4 Linear/ring/norm requires initialized SPM");
  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  const std::pair<uint64_t, uint64_t> slabs[] = {
      {input, kM * kLocalK * sizeof(c10::Half)}, {partial, kFullBytes},
      {residual, kFullBytes}, {raw_output, kFullBytes},
      {normalized_output, kFullBytes}, {gamma, kN * sizeof(c10::Half)},
      {tensor_scale, kScaleElems * sizeof(float)}};
  for (size_t i = 0; i < 7; ++i) {
    const auto [address, bytes] = slabs[i];
    TORCH_CHECK(address != 0 && address >= spm_base && address % 32 == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    address - spm_base <= SpmAllocator::SPM_USABLE - bytes,
                "Wall NVFP4 Linear/ring/norm requires aligned bounded absolute SPM slabs");
    for (int core = 0; core < NUM_CORES; ++core) {
      TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) +
                      address - spm_base + bytes - 1 <= UINT32_MAX,
                  "Wall NVFP4 Linear/ring/norm SPM range exceeds the register ABI");
    }
    for (size_t j = 0; j < i; ++j) {
      // O and nonfinal Down replace the old full residual with normalized data.
      // The canonical ring's global reduce barrier retires every residual
      // read before norm starts. No partial overlap or other alias is safe.
      const bool residual_alias = i == 4 && j == 2 && address == slabs[j].first;
      TORCH_CHECK(residual_alias || address + bytes <= slabs[j].first ||
                      slabs[j].first + slabs[j].second <= address,
                  "Wall NVFP4 Linear/ring/norm only permits exact normalized/residual alias");
    }
  }
  const uint64_t w_addr = RpuGetDevAddr(weight.data_ptr());
  const uint64_t s_addr = RpuGetDevAddr(scale.data_ptr());
  for (const auto address : {w_addr, s_addr}) {
    TORCH_CHECK(address != 0 && address % 256 == 0 &&
                    (address >> 8) <= UINT32_MAX,
                "Wall NVFP4 Linear/ring/norm requires aligned shift8 DDR operands");
  }
  TORCH_CHECK(w_addr + weight.nbytes() <= s_addr || s_addr + scale.nbytes() <= w_addr,
              "Wall NVFP4 Linear weight and scale must not overlap");
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Wall NVFP4 Linear/ring/norm payload is unavailable");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
  rpu_ddr_flush(weight.data_ptr());
  rpu_ddr_flush(scale.data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Wall NVFP4 Linear/ring/norm kernel missing");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
      {input, w_addr, partial, 0, kM, kN, kLocalK, 0, NUM_CORES,
       kLocalK * 2, kN * 2, s_addr, 0, 1},
      NUM_CORES * kN * 2, tensor_scale, layer});
  const auto u16 = [&](uint32_t reg, uint16_t value) {
    TORCH_CHECK(kernel->set_regs(reg, value) == 0,
                "Wall NVFP4 Linear/ring/norm register write failed");
  };
  const auto u32 = [&](uint32_t reg, uint32_t value) {
    rpu_pl_tiling::set_u32_reg_pair(*kernel, reg, value);
  };
  u16(32, NUM_CORES); u16(33, 1); u32(34, 31 * 256 * NUM_CORES);
  u32(36, residual - spm_base); u16(38, 0);
  u16(40, kN); u16(41, c10::Half(1.0f / kN).x); u16(42, kN * 2);
  u32(44, raw_output); u32(46, normalized_output); u32(48, gamma);
  u16(50, c10::Half(eps).x);
  const uint32_t partial_offset = partial - spm_base;
  const uint32_t raw_offset = raw_output - spm_base;
  for (uint32_t core = 0; core < NUM_CORES; ++core) {
    const uint32_t peer_partial = SPM_ALLOC.addr(core, partial_offset) - spm_base;
    const uint32_t peer_raw = SPM_ALLOC.addr(core, raw_offset) - spm_base;
    const std::pair<uint32_t, uint32_t> entries[] = {
        {0, peer_partial}, {16, peer_raw}, {32, core * kFullBytes / NUM_CORES},
        {48, kM * kN / NUM_CORES}};
    for (const auto [offset, value] : entries) {
      // sldc's legacy low/high tables require the SDK SCM halfword swap.
      rpu_set_legacy_scm_u16_checked(kernel, 4096 + offset + core,
          static_cast<uint16_t>(value), "Wall NVFP4 Linear/ring/norm");
      rpu_set_legacy_scm_u16_checked(kernel, 4096 + offset + 8 + core,
          static_cast<uint16_t>(value >> 16), "Wall NVFP4 Linear/ring/norm");
    }
  }
  u16(64, 8); u16(65, 1); u16(66, 1);
  // Every packet initializes the entire atomic bank: Linear->ring core0/1,
  // ring core24..31/global24, ring->norm core2, and global core-id signals.
  for (uint32_t reg = 128; reg < 256; reg += 2) {
    TORCH_CHECK(kernel->set_regs(reg, uint32_t{0}) == 0,
                "Wall NVFP4 Linear/ring/norm counter reset failed");
  }
  writer.write(*kernel);
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_wall_base_nvfp4_o_ring_norm_spm_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t partial,
    uint32_t residual, uint32_t raw_output, uint32_t normalized_output,
    uint32_t gamma, const at::Tensor& scale, uint32_t tensor_scale,
    int64_t tensor_scale_elems, uint16_t layer, double eps) {
  launch_wall_base_nvfp4_linear_ring_norm(
      input, weight, partial, residual, raw_output, normalized_output,
      gamma, scale, tensor_scale, tensor_scale_elems, layer, eps,
      KernelId::WALL_BASE_NVFP4_O_RING_NORM_H32);
}

void rpu_launch_wall_base_nvfp4_down_ring_norm_spm_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t partial,
    uint32_t residual, uint32_t raw_output, uint32_t normalized_output,
    uint32_t gamma, const at::Tensor& scale, uint32_t tensor_scale,
    int64_t tensor_scale_elems, uint16_t layer, double eps) {
  TORCH_CHECK(layer < 35, "Wall NVFP4 Down/next norm requires a nonfinal layer");
  launch_wall_base_nvfp4_linear_ring_norm(
      input, weight, partial, residual, raw_output, normalized_output,
      gamma, scale, tensor_scale, tensor_scale_elems, layer, eps,
      KernelId::WALL_BASE_NVFP4_DOWN_RING_NORM_H32);
}


void rpu_launch_wall_base_nvfp4_qkv_bias_spm_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t output,
    uint32_t bias, const at::Tensor& scale, uint32_t tensor_scale,
    int64_t tensor_scale_elems, uint16_t layer, int64_t n) {
  constexpr int64_t kM = 32, kK = 1024, kLayers = 36, kScaleElems = 40;
  static_assert(NUM_CORES == 8, "Wall NVFP4 QKV bias requires TP8");
  TORCH_CHECK((n == 2048 || n == 1024) && layer < kLayers &&
                  tensor_scale_elems == kScaleElems,
              "Wall NVFP4 QKV bias requires N2048/1024, 36 layers and padded FP32[40]");
  const int64_t local_n = n / NUM_CORES;
  TORCH_CHECK(weight.defined() && scale.defined() &&
                  weight.device().type() == at::kPrivateUse1 &&
                  scale.device() == weight.device() &&
                  weight.scalar_type() == at::kByte && scale.scalar_type() == at::kByte &&
                  weight.is_contiguous() && scale.is_contiguous() &&
                  weight.storage_offset() == 0 && scale.storage_offset() == 0 &&
                  weight.sizes() == at::IntArrayRef({n, kK / 2}) &&
                  scale.sizes() == at::IntArrayRef({kK / 16, n}) &&
                  weight.nbytes() == n * kK / 2 && scale.nbytes() == kK / 16 * n,
              "Wall NVFP4 QKV bias requires exact packed device weight and striped FP8 scale");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Wall NVFP4 QKV bias requires initialized SPM");
  const uint64_t spm_base = SPM_ALLOC.addr(0, 0);
  const std::pair<uint64_t, uint64_t> slabs[] = {
      {input, kM * kK * sizeof(c10::Half)},
      {output, kM * local_n * sizeof(c10::Half)},
      {bias, local_n * sizeof(c10::Half)},
      {tensor_scale, kScaleElems * sizeof(float)}};
  for (size_t i = 0; i < 4; ++i) {
    const auto [address, bytes] = slabs[i];
    TORCH_CHECK(address != 0 && address >= spm_base && address % 32 == 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    address - spm_base <= SpmAllocator::SPM_USABLE - bytes,
                "Wall NVFP4 QKV bias requires aligned bounded absolute SPM slabs");
    for (int core = 0; core < NUM_CORES; ++core) {
      TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) +
                      (address - spm_base) + bytes - 1 <= UINT32_MAX,
                  "Wall NVFP4 QKV bias SPM range exceeds the register ABI");
    }
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(address + bytes <= slabs[j].first ||
                      slabs[j].first + slabs[j].second <= address,
                  "Wall NVFP4 QKV bias requires independent SPM slabs");
    }
  }
  const uint64_t w_addr = RpuGetDevAddr(weight.data_ptr());
  const uint64_t s_addr = RpuGetDevAddr(scale.data_ptr());
  for (const auto address : {w_addr, s_addr}) {
    TORCH_CHECK(address != 0 && address % 256 == 0 &&
                    (address >> 8) <= UINT32_MAX,
                "Wall NVFP4 QKV bias requires aligned shift8 DDR operands");
  }
  TORCH_CHECK(w_addr + weight.nbytes() <= s_addr ||
                  s_addr + scale.nbytes() <= w_addr,
              "Wall NVFP4 QKV weight and scale must not overlap");
  const auto id = n == 2048 ? KernelId::WALL_BASE_NVFP4_QKV_BIAS_BARRIER_N64
                            : KernelId::WALL_BASE_NVFP4_QKV_BIAS_BARRIER_N32;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "Wall NVFP4 QKV bias barrier payload is unavailable");
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
  rpu_ddr_flush(weight.data_ptr());
  rpu_ddr_flush(scale.data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Wall NVFP4 QKV bias barrier kernel missing");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
      {input, w_addr, output, bias, kM, static_cast<uint32_t>(local_n), kK,
       1, NUM_CORES, kK * 2, static_cast<uint32_t>(local_n * 2), s_addr, 1, 0},
      static_cast<uint32_t>(NUM_CORES * local_n * 2), tensor_scale, layer});
  TORCH_CHECK(kernel->set_regs(64, uint16_t{4}) == 0 &&
                  kernel->set_regs(65, uint16_t{1}) == 0 &&
                  kernel->set_regs(66, uint16_t{1}) == 0,
              "Wall NVFP4 QKV bias grid register write failed");
  // Four resident blocks wait for core counter0 > 3. Both atomic counters
  // are 32-bit register pairs, reset in every BUILD/REPLAY launch packet.
  TORCH_CHECK(kernel->set_regs(128, uint32_t{0}) == 0 &&
                  kernel->set_regs(130, uint32_t{0}) == 0,
              "Wall NVFP4 QKV bias counter reset failed");
  writer.write(*kernel);
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {4, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

static void launch_action_nvfp4_v2_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t output,
    int64_t rows, int64_t n, int64_t k, int partition, int cores,
    const at::Tensor& scale, uint32_t tensor_scale, uint16_t layer,
    bool linear_acc32) {
  const bool col = partition == 1;
  TORCH_CHECK((col && k == 1024 && (n == 2048 || n == 4096)) ||
                  (partition == 0 && n == 1024 && (k == 2048 || k == 4096)),
              "Action NVFP4 v2 requires admitted projection geometry");
  TORCH_CHECK(weight.scalar_type() == at::kByte && scale.scalar_type() == at::kByte &&
                  weight.device().type() == at::kPrivateUse1 && scale.device() == weight.device() &&
                  weight.is_contiguous() && scale.is_contiguous() &&
                  weight.sizes() == at::IntArrayRef({n, k / 2}) &&
                  scale.sizes() == at::IntArrayRef({k / 16, n}),
              "Action NVFP4 v2 packed weight/FP8 scale shape, dtype or device mismatch");
  const int64_t ln = col ? n / cores : n, lk = col ? k : k / cores;
  const bool gate_up = col && n == 4096;
  const auto tile = linear_acc32
      ? rpu_pl_tiling::TilePick{gate_up ? 48 : 64, gate_up ? 384 : 320}
      : rpu_pl_tiling::select_tile_nvfp4_acc16(rows, ln, lk);
  const uint16_t nt = tile.n_tile, mt = tile.m_tile;
  const auto id = linear_acc32
      ? (gate_up ? KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48
                 : KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64)
      : autotile_nvfp4_acc16_kernel_id(tile.n_tile);
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
  rpu_ddr_flush(weight.data_ptr());
  rpu_ddr_flush(scale.data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Action NVFP4 v2 kernel missing: install the striped-v2 expansion REF");
  kernel->reset_regs();
  const auto u16 = [&](int reg, uint16_t value) { kernel->set_regs(reg, value); };
  const auto u32 = [&](int reg, uint32_t value) { u16(reg, value & 0xffff); u16(reg + 1, value >> 16); };
  u32(0, input); u32(4, output); u32(6, 0);
  u16(8, rows); u32(10, ln); u16(12, lk); u16(13, lk / 64);
  u16(14, 0); u16(15, cores); u32(16, lk * 2); u32(18, ln * 2);
  u32(22, cores * ln * 2); u32(24, tensor_scale); u16(26, col); u16(27, layer);
  const uint16_t gx = (ln + nt - 1) / nt, gy = (rows + mt - 1) / mt;
  u16(64, gx); u16(65, gy); u16(66, 1);
  writer.write(*kernel);
  auto* queue = GET_QUEUE(cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {gx, gy, uint16_t{1}}, {0,1,2,3,4,5,6,7});
}


void rpu_launch_pi05_nvfp4_v2_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t output,
    int64_t rows, int64_t n, int64_t k, int partition, int cores,
    const at::Tensor& scale, uint32_t tensor_scale, uint16_t layer,
    bool linear_acc32) {
  TORCH_CHECK(cores == 8 && rows > 0 && rows <= 64 && layer < 18 && tensor_scale != 0,
              "Pi05 NVFP4 v2 requires TP8, <=64 action rows, layer<18 and tensor-scale SPM");
  launch_action_nvfp4_v2_kernel(input, weight, output, rows, n, k, partition,
                               cores, scale, tensor_scale, layer, linear_acc32);
}

void rpu_launch_g05_nvfp4_v2_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t output,
    int64_t rows, int64_t n, int64_t k, int partition, int cores,
    const at::Tensor& scale, uint32_t tensor_scale, uint16_t layer,
    bool linear_acc32) {
  TORCH_CHECK(cores == 8 && rows == 32 && layer < 24 &&
                  tensor_scale != 0 && tensor_scale % 32 == 0,
              "G05 NVFP4 v2 requires exact TP8/M32, layer<24 and aligned tensor-scale SPM");
  const bool col = partition == 1;
  TORCH_CHECK((col && k == 1024 && (n == 2048 || n == 4096)) ||
                  (partition == 0 && n == 1024 && (k == 2048 || k == 4096)),
              "G05 NVFP4 v2 requires exact Action projection geometry");
  const size_t input_bytes = rows * (col ? k : k / cores) * sizeof(c10::Half);
  const size_t output_bytes = rows * (col ? n / cores : n) * sizeof(c10::Half);
  check_pi05_resident_spm(input, input_bytes);
  check_pi05_resident_spm(output, output_bytes);
  check_pi05_resident_spm(tensor_scale, 96);
  const auto disjoint = [](uint32_t a, size_t a_bytes, uint32_t b, size_t b_bytes) {
    return uint64_t(a) + a_bytes <= b || uint64_t(b) + b_bytes <= a;
  };
  TORCH_CHECK(disjoint(input, input_bytes, output, output_bytes) &&
                  disjoint(input, input_bytes, tensor_scale, 96) &&
                  disjoint(output, output_bytes, tensor_scale, 96),
              "G05 NVFP4 v2 input/output/tensor-scale SPM intervals overlap");
  launch_action_nvfp4_v2_kernel(input, weight, output, rows, n, k, partition,
                               cores, scale, tensor_scale, layer, linear_acc32);
}

static void launch_g05_gate_up_nvfp4_typed(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer, KernelId id) {
  constexpr int64_t kM = 32, kN = 4096, kK = 1024, kLocalN = 512;
  static_assert(NUM_CORES == 8, "G05 NVFP4 pair requires TP8");
  TORCH_CHECK(layer < 24, "G05 NVFP4 pair requires layer<24");
  for (const auto* weight : {&gate_weight, &up_weight}) {
    TORCH_CHECK(weight->defined() && weight->device().type() == at::kPrivateUse1 &&
                    weight->device() == gate_weight.device() &&
                    weight->scalar_type() == at::kByte && weight->is_contiguous() &&
                    weight->storage_offset() == 0 &&
                    weight->sizes() == at::IntArrayRef({kN, kK / 2}) &&
                    weight->nbytes() == kN * kK / 2,
                "G05 NVFP4 pair requires packed device uint8 [4096,512] weights");
  }
  for (const auto* scale : {&gate_scale, &up_scale}) {
    TORCH_CHECK(scale->defined() && scale->device() == gate_weight.device() &&
                    scale->scalar_type() == at::kByte && scale->is_contiguous() &&
                    scale->storage_offset() == 0 &&
                    scale->sizes() == at::IntArrayRef({kK / 16, kN}) &&
                    scale->nbytes() == kK / 16 * kN,
                "G05 NVFP4 pair requires striped FP8 device uint8 [64,4096] scales");
  }
  const std::pair<uint32_t, size_t> slabs[] = {
      {input, kM * kK * sizeof(c10::Half)},
      {gate_output, kM * kLocalN * sizeof(c10::Half)},
      {up_output, kM * kLocalN * sizeof(c10::Half)},
      {gate_tensor_scale, 24 * sizeof(float)},
      {up_tensor_scale, 24 * sizeof(float)}};
  for (size_t i = 0; i < 5; ++i) {
    const auto [address, bytes] = slabs[i];
    check_pi05_resident_spm(address, bytes);
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(uint64_t(address) + bytes <= slabs[j].first ||
                      uint64_t(slabs[j].first) + slabs[j].second <= address,
                  "G05 NVFP4 pair requires disjoint input/outputs/tensor-scale SPM");
    }
  }
  for (const auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    const uint64_t address = RpuGetDevAddr(tensor->data_ptr());
    TORCH_CHECK(address != 0 && address % 256 == 0 && address < (uint64_t{1} << 40) &&
                    tensor->nbytes() <= (uint64_t{1} << 40) - address,
                "G05 NVFP4 pair requires aligned in-range shift8 DDR spans");
  }
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, gate_scale},
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, up_weight},
          {{GraphDdrRegisterRole::Fp8BlockScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, up_scale}});
  for (const auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale})
    rpu_ddr_flush(tensor->data_ptr());
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 NVFP4 pair missing from expansion REF");
  kernel->reset_regs();
  const auto u16 = [&](int reg, uint16_t value) { kernel->set_regs(reg, value); };
  const auto u32 = [&](int reg, uint32_t value) { u16(reg, value & 0xffff); u16(reg + 1, value >> 16); };
  u32(0, input); u32(4, gate_output); u32(6, 0);
  u16(8, kM); u32(10, kLocalN); u16(12, kK); u16(13, kK / 64);
  u16(14, 0); u16(15, NUM_CORES); u32(16, kK * 2); u32(18, kLocalN * 2);
  u32(22, NUM_CORES * kLocalN * 2); u32(24, gate_tensor_scale);
  u16(26, 1); u16(27, layer); u32(36, up_tensor_scale); u32(38, up_output);
  u16(64, 8); u16(65, 1); u16(66, 1);
  if (id == KernelId::G05_GATE_UP_SWIGLU_NVFP4_ACC16_M32N512K1024_N128 ||
      id == KernelId::G05_GATE_UP_SWIGLU_NVFP4_ACC32_M32N512K1024_N128) {
    // All eight warps arrive before BASE SiLU; replay must restart both counters.
    kernel->set_regs(128, uint32_t{0});
    kernel->set_regs(130, uint32_t{0});
  }
  writer.write(*kernel);
  auto* queue = GET_QUEUE(NUM_CORES);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer) {
  launch_g05_gate_up_nvfp4_typed(input, gate_weight, up_weight, gate_output, up_output,
      gate_scale, up_scale, gate_tensor_scale, up_tensor_scale, layer,
      KernelId::G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128);
}

void rpu_launch_g05_gate_up_paired_nvfp4_acc32_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer) {
  launch_g05_gate_up_nvfp4_typed(input, gate_weight, up_weight, gate_output, up_output,
      gate_scale, up_scale, gate_tensor_scale, up_tensor_scale, layer,
      KernelId::G05_GATE_UP_PAIRED_NVFP4_ACC32_M32N512K1024_N128);
}

void rpu_launch_g05_gate_up_swiglu_nvfp4_acc16_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer) {
  launch_g05_gate_up_nvfp4_typed(input, gate_weight, up_weight, gate_output, up_output,
      gate_scale, up_scale, gate_tensor_scale, up_tensor_scale, layer,
      KernelId::G05_GATE_UP_SWIGLU_NVFP4_ACC16_M32N512K1024_N128);
}

void rpu_launch_g05_gate_up_swiglu_nvfp4_acc32_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t gate_output, uint32_t up_output,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer) {
  launch_g05_gate_up_nvfp4_typed(input, gate_weight, up_weight, gate_output, up_output,
      gate_scale, up_scale, gate_tensor_scale, up_tensor_scale, layer,
      KernelId::G05_GATE_UP_SWIGLU_NVFP4_ACC32_M32N512K1024_N128);
}


void rpu_parallel_linear_w4_test_out(
    const at::Tensor& input_per_core, const at::Tensor& weight,
    const at::Tensor& scale, const std::optional<at::Tensor>& negative_zero,
    const at::Tensor& output_per_core, bool is_col_parallel, bool linear_acc32) {
  TORCH_CHECK(weight.defined() && weight.scalar_type() == at::kByte &&
                  weight.dim() == 2 && weight.is_contiguous(),
              "W4 probe requires packed uint8 [N,K/2] weight");
  for (const auto* tensor : {&input_per_core, &output_per_core}) {
    TORCH_CHECK(tensor->defined() &&
                    tensor->device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor->scalar_type() == at::kHalf && tensor->dim() == 3 &&
                    tensor->is_contiguous(),
                "W4 probe input/output must be contiguous device FP16 [cores,M,local_dim]");
  }
  const int64_t cores = input_per_core.size(0), m = input_per_core.size(1);
  const int64_t n = weight.size(0), k = weight.size(1) * 2;
  const int partition = is_col_parallel ? 1 : 0;
  validate_parallel_linear_geometry(m, n, k, partition, cores);
  const int64_t local_k = is_col_parallel ? k : k / cores;
  const int64_t local_n = is_col_parallel ? n / cores : n;
  TORCH_CHECK(input_per_core.size(2) == local_k &&
                  output_per_core.sizes() == at::IntArrayRef({cores, m, local_n}),
              "W4 probe per-core input/output shape mismatch");
  TORCH_CHECK(scale.defined() && scale.scalar_type() == at::kHalf &&
                  scale.dim() == 2 && scale.is_contiguous() &&
                  scale.device() == input_per_core.device() &&
                  weight.device() == input_per_core.device(),
              "W4 probe requires device weight and contiguous FP16 2D scale metadata");
  if (negative_zero.has_value() && negative_zero->defined() && negative_zero->numel()) {
    TORCH_CHECK(negative_zero->device() == scale.device() &&
                    negative_zero->scalar_type() == at::kHalf &&
                    negative_zero->is_contiguous() &&
                    negative_zero->sizes() == scale.sizes(),
                "W4 probe negative_zero must match FP16 scale metadata");
  }
  at::assert_no_overlap(output_per_core, input_per_core);
  at::assert_no_overlap(output_per_core, weight);
  at::assert_no_overlap(output_per_core, scale);
  if (negative_zero.has_value() && negative_zero->defined() && negative_zero->numel())
    at::assert_no_overlap(output_per_core, *negative_zero);
  if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
  SpmAllocator::ScopedTemporary scope;
  const uint32_t input_addr = SPM_ALLOC.addr(0,
      SPM_ALLOC.alloc_temporary(m * local_k * sizeof(c10::Half)));
  const uint32_t output_addr = SPM_ALLOC.addr(0,
      SPM_ALLOC.alloc_temporary(m * local_n * sizeof(c10::Half)));
  rpu_launch_ddr_scatter_spm_dma(input_per_core, 0, m * local_k,
      m * local_k * sizeof(c10::Half), input_addr, cores);
  rpu_launch_linear_spm_to_spm_acc16_kernel(
      input_addr, weight, output_addr, m, n, k, partition, cores, 0, scale,
      0, 0, linear_acc32, false, false, false,
      negative_zero.value_or(at::Tensor{}));
  rpu_launch_spm_scatter_ddr_dma(output_addr, output_per_core, 0,
      m * local_n, m * local_n * sizeof(c10::Half), cores);
}
