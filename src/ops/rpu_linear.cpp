#include "rhino_launch_buffer.h"

#include "rhino_launch_program.h" // for Program_t
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"  // eager linear stages through the SPM arena
#include "rpu_linear_tiling.h"  // host-side generated-tile selection
#include <ATen/MemoryOverlap.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <utility>
#include <vector>   // for vector

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8

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
      "W4 pgrp scale must be a contiguous 2D RPU tensor");

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
              "Pi0.5 resident W8 requires installed contiguous RPU int8 [4096,1024]");
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
      "Pi0.5 KV1 compact repack requires the installed contiguous RPU "
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
      "Pi0.5 KV1 compact repack requires the matching contiguous RPU "
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
      "Pi0.5 KV1 compact repack did not create exact contiguous RPU owners");

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
      "Pi0.5 KV1 compact repack requires independent 256-byte-aligned RPU "
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
      "Pi0.5 KV1 direct pair repack requires the installed contiguous RPU "
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
      "Pi0.5 KV1 direct pair repack requires the matching contiguous RPU "
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
  // future change to either permutation must fail before any RPU owner exists.
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
      "Pi0.5 KV1 direct pair repack did not create exact contiguous RPU owners");

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
      "RPU owners addressable by the shift8 ABI");

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
        " weight must be the installed contiguous RPU matching-dtype "
        "TP8-col-swizzled [2048,1024] owner");
    TORCH_CHECK(fp16 ? !scale.defined() : (
        scale.defined() && scale.scalar_type() == at::kHalf &&
            scale.device() == weight.device() && scale.dim() == 1 &&
            scale.size(0) == kPhysicalN && scale.is_contiguous() &&
            scale.storage_offset() == 0 && scale.nbytes() == kScaleBytes),
        "Pi0.5 KV1 pair-owner ", role,
        " requires no scale for FP16, or the matching RPU fp16 [2048] W8 scale");

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
      "Pi0.5 KV1 pair-owner K/V owners must share one RPU device");

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
              "Pi Prefill KV1 K/V owners must share one RPU device");
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
      "Pi0.5 KV1 fixed M592N32 requires an exact contiguous RPU int8 "
      "compact weight owner [256,1024]");
  TORCH_CHECK(
      compact_scale.defined() && compact_scale.scalar_type() == at::kHalf &&
          compact_scale.device() == compact_weight.device() &&
          compact_scale.dim() == 1 && compact_scale.size(0) == 256 &&
          compact_scale.is_contiguous() &&
          compact_scale.storage_offset() == 0 &&
          compact_scale.nbytes() == 256 * sizeof(c10::Half),
      "Pi0.5 KV1 fixed M592N32 requires an exact contiguous RPU fp16 "
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
        " weight must be an exact contiguous matching-dtype RPU [2048,1024] owner");
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
        " scale must be an exact contiguous RPU fp16 [2048] owner");
  };
  check_weight(k_full_weight, "K");
  check_scale(k_full_scale, "K");
  check_weight(v_full_weight, "V");
  check_scale(v_full_scale, "V");
  TORCH_CHECK(
      k_full_weight.device() == v_full_weight.device() &&
          (fp16 || (k_full_weight.device() == k_full_scale.device() &&
                    k_full_weight.device() == v_full_scale.device())),
      "Pi0.5 KV1 pair-owner DDR owners must share one RPU device");

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
  TORCH_CHECK(rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 400 || rows == 416 || rows == 432 || rows == 448,
              "Pi Prefill KV1 requires C272/C288/C304/C320/C400/C416/C432/C448");
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
        " weight must be an exact contiguous RPU int8 [2048,2048] owner");
  };
  auto check_scale = [&](const at::Tensor& tensor, const char* role) {
    TORCH_CHECK(
        tensor.defined() && tensor.scalar_type() == at::kHalf &&
            tensor.device().type() == c10::DeviceType::PrivateUse1 &&
            tensor.dim() == 1 && tensor.size(0) == kPhysicalN &&
            tensor.is_contiguous() && tensor.storage_offset() == 0 &&
            tensor.nbytes() == kScaleBytes,
        "Pi0.5 prefill KV1 pair-owner ", role,
        " scale must be an exact contiguous RPU fp16 [2048] owner");
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
      "Pi0.5 prefill KV1 pair-owner DDR owners must share one RPU device");

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

  // 预分配 RPU 输出
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
              "linear_w4a16: input and weight must share an RPU device");
  const int64_t k = input.size(-1), n = weight.size(0);
  TORCH_CHECK(k > 0 && k % 64 == 0 && weight.size(1) == k / 2,
              "linear_w4a16: packed weight must be [N,K/2] with logical K divisible by 64");
  TORCH_CHECK(weight.is_contiguous(),
              "linear_w4a16: pgrp packed weight must be contiguous");
  validate_parallel_linear_geometry(input.numel() / k, n, k, 1, 8);
  TORCH_CHECK(scale.defined() && scale.device() == input.device() &&
                  scale.scalar_type() == at::kHalf && scale.dim() == 2 &&
                  scale.size(0) == 32 && scale.is_contiguous(),
              "linear_w4a16: scale must be contiguous RPU FP16 pgrp G32 payload");
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
                "linear_w4a16: bias must be contiguous RPU FP16 [N]");
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
                "rpu_linear W8A16 requires contiguous RPU FP16 scale [N]");
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
  // (measured: output + 7*bias). Same rule validate_m1_partitioned_bias()
  // enforces for M=1 — run the GEMM bias-free, add bias once after the reduce.
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

  rpu_ddr_flush(xp);
  rpu_ddr_flush(wp);
  if (has_bias) rpu_ddr_flush(bp);

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
    // when row-partitioned. The converter therefore records its choice on the
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

// Transform weight for RPU linear kernel
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
              "linear_with_core_count: input and weight must be on the same RPU");
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
                "linear_with_core_count: bias must be a contiguous FP16 [N] tensor on the input RPU");
  }
  auto output_shape = input.sizes().vec();
  output_shape.back() = n;
  Tensor input_2d = input.contiguous().view({rows, k});
  Tensor output = at::empty(output_shape, input.options());
  Tensor output_2d = output.view({rows, n});
  const Tensor scale = scale_opt.has_value()
      ? (packed ? rpu_retain_linear_quant_scale(weight, *scale_opt) : *scale_opt)
      : Tensor{};
  rpu_launch_linear_ddr_kernel(
      input_2d, weight, output_2d, bias, has_bias, static_cast<int>(partition),
      nullptr, 0, nullptr, static_cast<int>(num_cores), scale, linear_acc32);
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
              "rpu_linear_into: input, weight, and output must be on RPU");
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
                "rpu_linear_into: bias must be on RPU");
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
              "linear_w8a16: input, weight and output must share an RPU device");
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
                "linear_w8a16: bias must be contiguous RPU FP16 [N]");
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

void validate_m1_partitioned_bias(
    uint32_t bias_spm_addr, int partition, int num_cores) {
  if (bias_spm_addr == 0) return;

  TORCH_CHECK(
      partition == 1 || num_cores == 1,
      "M=1 row-partition Linear cannot add bias before the cross-core "
      "all-reduce; add bias after reduction instead");
}

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
  if (M == 1) {
    validate_m1_partitioned_bias(bias_spm_addr, partition, num_cores);
  }

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

// =============================================================================
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
    bool prefer_row_weight_reuse)
{
  validate_parallel_linear_geometry(M, N, K, partition, num_cores);
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
        "W8A16 row weight reuse requires RPU signed-int8 weight and FP16 scale");
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
  if (M == 1) {
    validate_m1_partitioned_bias(bias_spm_addr, partition, num_cores);
  }
  if (register_census_active) {
    TORCH_CHECK(weight.scalar_type() == at::kHalf ||
                    weight.scalar_type() == at::kChar ||
                    fp8_e4m3_weight,
                "typed DDR-register census admits FP16 Linear, signed-int8 "
                "W8A16, or full-byte raw-uint8 E4M3 weights");
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

    // Select kernel and grid tiling for the requested accumulator precision.
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
    // The row-weight-reuse variant requires grid.y=1.
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
    // in dim 0 for this dispatcher. The provider ABI extends common regs 0..23
    // with {24:groupSize,25:localG,26-27:SGBlockByteStep}.
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
      // path leaves it unset, so the whole segment falls back to PASSTHROUGH.
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
      // IN-KERNEL bias: reg6/7 = bias SPM addr lo/hi as a RAW per-core
      // SPM byte address (same space as in/out regs 0/1,4/5 — NOT >>8-shifted).
      // Scale uses regs 20/21 and bias uses reg14. Only col-partition +bias is
      // exercised by Wall-OSS int4 qkv/gate/up; o/down are bias-free.
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
                " weight must be contiguous RPU int8 [9728,2560]");
  };
  auto check_scale = [&](const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined() &&
                    tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf && tensor.is_contiguous() &&
                    tensor.dim() == 1 && tensor.size(0) == kN,
                "Qwen3-VL 4B Prefill fused ", name,
                " scale must be contiguous RPU fp16 [9728]");
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
      KernelId::FUSED_GATE_UP_SWIGLU_W8A16_M320N112;
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
                "Pi GeGLU requires contiguous RPU int8 [16384,2048] weights");
  }
  for (const auto* scale : {&gate_scale, &up_scale}) {
    TORCH_CHECK(scale->device().type() == c10::DeviceType::PrivateUse1 &&
                    scale->scalar_type() == at::kHalf &&
                    scale->is_contiguous() && scale->dim() == 1 &&
                    scale->size(0) == kN,
                "Pi GeGLU requires contiguous RPU FP16 [16384] scales");
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
                "Pi W4 GeGLU requires packed RPU uint8 [4096,512] weights");
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
                "Pi W4 GeGLU requires packed RPU uint8 [4096,512] weights");
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
                "Pi M50 GeGLU requires matching contiguous RPU [4096,1024] weights");
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
                "Pi M50 GeGLU requires contiguous RPU FP16 [4096] scales");
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
  TORCH_CHECK(rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 400 || rows == 416 || rows == 432 || rows == 448,
              "Pi Prefill GateUp pair requires C272/C288/C304/C320/C400/C416/C432/C448");
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
        "Pi Prefill GateUp pair requires contiguous RPU int8 "
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
        "Pi Prefill GateUp pair requires contiguous RPU FP16 "
        "Gate/Up scales [16384]");
  }

  const KernelId kernel_id = rows == 432
      ? (fp16 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_FP16_C432X2_M112N80K128
              : KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C432X2_M112N80K128) :
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
        "Pi C272 A8 requires installed RPU int8 Gate/Up [16384,2048]");
  for (const auto* scale : {&gate_scale, &up_scale})
    TORCH_CHECK(scale->defined() && scale->device().type() == at::kPrivateUse1 &&
        scale->scalar_type() == at::kFloat && scale->is_contiguous() &&
        scale->storage_offset() == 0 && scale->sizes() == at::IntArrayRef({16384}),
        "Pi C272 A8 requires cold promoted RPU FP32 scales [16384]");
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
  TORCH_CHECK(rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 400 || rows == 416 || rows == 432 || rows == 448,
              "Pi Prefill A8 requires exact C272/C288/C304/C320/C400/C416/C432/C448");
  const uint32_t m=rows, panel=rows>=400 ? 384 : 256;
  TORCH_CHECK(SPM_ALLOC.is_initialized(),"Pi Prefill A8 requires initialized SPM");
  const uint32_t base=SPM_ALLOC.addr(0,0);
  const std::array<std::pair<uint32_t,uint32_t>,5> ranges{{
      {a8,2*m*2080},{row_scale,2*m*2+(rows==272 || rows==288 || rows==304 || rows==320 || rows==416 || rows==432 || rows==448 ? 32u : 0u)},{stage,2*panel*2048},
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
        "Pi Prefill A8 requires installed RPU int8 Gate/Up [16384,2048]");
  for(const auto* scale:{&gate_scale,&up_scale})
    TORCH_CHECK(scale->defined() && scale->device().type()==at::kPrivateUse1 &&
        scale->scalar_type()==at::kFloat && scale->is_contiguous() &&
        scale->storage_offset()==0 && scale->sizes()==at::IntArrayRef({16384}),
        "Pi Prefill A8 requires cold promoted RPU FP32 scales [16384]");
  const auto id=rows==432 ? KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_C432_M48N32K2048 :
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
                "Pi NVFP4 GeGLU requires packed RPU uint8 [4096,512] weights");
  }
  for (const auto* scale : {&gate_scale, &up_scale}) {
    TORCH_CHECK(scale->defined() && scale->device().type() == at::kPrivateUse1 &&
                    scale->scalar_type() == at::kByte && scale->is_contiguous() &&
                    scale->storage_offset() == 0 &&
                    scale->sizes() == at::IntArrayRef({kK / 16, kN}) &&
                    scale->nbytes() == kK / 16 * kN,
                "Pi NVFP4 GeGLU requires striped-v2 FP8 RPU uint8 [64,4096] scales");
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
    // Initialize the required 32-bit counter parameters.
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

void rpu_launch_pi05_nvfp4_v2_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t output,
    int64_t rows, int64_t n, int64_t k, int partition, int cores,
    const at::Tensor& scale, uint32_t tensor_scale, uint16_t layer,
    bool linear_acc32) {
  TORCH_CHECK(cores == 8 && rows > 0 && rows <= 64 && layer < 18 && tensor_scale != 0,
              "Pi05 NVFP4 v2 requires TP8, <=64 action rows, layer<18 and tensor-scale SPM");
  const bool col = partition == 1;
  TORCH_CHECK((col && k == 1024 && (n == 2048 || n == 4096)) ||
                  (partition == 0 && n == 1024 && (k == 2048 || k == 4096)),
              "Pi05 NVFP4 v2 requires admitted Action projection geometry");
  TORCH_CHECK(weight.scalar_type() == at::kByte && scale.scalar_type() == at::kByte &&
                  weight.device().type() == at::kPrivateUse1 && scale.device() == weight.device() &&
                  weight.is_contiguous() && scale.is_contiguous() &&
                  weight.sizes() == at::IntArrayRef({n, k / 2}) &&
                  scale.sizes() == at::IntArrayRef({k / 16, n}),
              "Pi05 NVFP4 v2 packed weight/FP8 scale shape, dtype or device mismatch");
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
  TORCH_CHECK(kernel, "Pi05 NVFP4 v2 kernel missing: install the striped-v2 expansion REF");
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
