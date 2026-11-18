// Explicit Pi NVFP4 candidates. Model routing is unchanged.
// Scale layout/ABI: rhino-ops rasm/pi05_nvfp4_{linear,geglu}_db_sspm.
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rpu_pi05_q_kv1_abi.h"
#include <array>
#include <initializer_list>
#include <utility>
#include <vector>

using namespace ::rhino_lkn;

namespace {
using Span = std::pair<uint64_t, uint64_t>;

void pi_sspm_check_spans(std::initializer_list<Span> spans) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "Pi SSPM requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (auto span = spans.begin(); span != spans.end(); ++span) {
    const auto [address, bytes] = *span;
    TORCH_CHECK(address >= base && address % 256 == 0 && bytes > 0 &&
                    bytes <= SpmAllocator::SPM_USABLE &&
                    address - base <= SpmAllocator::SPM_USABLE - bytes,
                "Pi SSPM requires complete aligned absolute SPM spans");
    for (int core = 0; core < 8; ++core)
      TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes - 1 <= UINT32_MAX,
                  "Pi SSPM per-core span exceeds address ABI");
    for (auto other = spans.begin(); other != span; ++other)
      TORCH_CHECK(address + bytes <= other->first || other->first + other->second <= address,
                  "Pi SSPM input/output/scales must have disjoint complete spans");
  }
}

void pi_sspm_check_owner(const at::Tensor &owner, const at::Tensor &weight,
                         at::IntArrayRef shape) {
  TORCH_CHECK(owner.defined() && owner.device().type() == at::kPrivateUse1 &&
                  owner.device() == weight.device() && owner.scalar_type() == at::kByte &&
                  owner.is_contiguous() && owner.storage_offset() == 0 && owner.sizes() == shape,
              "Pi SSPM requires complete uint8 RPU weight/per-core-scale owners");
  const uint64_t address = RpuGetDevAddr(owner.data_ptr());
  constexpr uint64_t limit = UINT64_C(1) << 40;
  TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                  owner.nbytes() <= limit - address,
              "Pi SSPM DDR owner is outside the aligned shift8 address range");
}

void pi_sspm_check_ddr_disjoint(std::initializer_list<const at::Tensor *> owners) {
  for (auto a = owners.begin(); a != owners.end(); ++a)
    for (auto b = owners.begin(); b != a; ++b) {
      const uint64_t x = RpuGetDevAddr((*a)->data_ptr()), y = RpuGetDevAddr((*b)->data_ptr());
      TORCH_CHECK(x + (*a)->nbytes() <= y || y + (*b)->nbytes() <= x,
                  "Pi SSPM weight and scale owners overlap");
    }
}

void pi_sspm_admit(KernelId id, int64_t rows, int64_t cores, uint16_t layer) {
  TORCH_CHECK(cores == 8 && rows > 0 && rows <= 64 && layer < 18,
              "Pi SSPM candidate requires TP8/M1..64/layer0..17");
  TORCH_CHECK(RpuKernelGraph::has_active() && graph_dma::active(),
              "Pi SSPM candidate requires a retained BUILD/REPLAY context");
  TORCH_CHECK(KernelCache::instance().has_loaded(id), "Pi SSPM candidate payload unavailable");
}

void pi_sspm_preload(const at::Tensor &scale, uint32_t destination) {
  auto &graph = RpuKernelGraph::active();
  graph.keep_alive(scale);
  rpu_ddr_flush(scale.data_ptr());
  graph_dma::stage_census_fixed_burst(scale, GraphOuterFastDmaSide::Source, 0, scale.nbytes(), 8);
  const int64_t bytes = scale.nbytes() / 8;
  // This DMA ABI counts halfwords, while the retained typed source stays uint8.
  // Its existing per-core barriers fence every preload before the compute node.
  rpu_launch_ddr_scatter_spm_dma(reinterpret_cast<c10::Half *>(scale.data_ptr()),
                                bytes / sizeof(c10::Half), bytes, destination, 8);
}

void pi_sspm_registers(Kernel_t &kernel, uint32_t input, uint32_t output,
                       uint32_t block_scale, uint32_t tensor_scale,
                       int64_t rows, int64_t local_n, int64_t local_k,
                       bool col, uint16_t layer) {
  kernel.reset_regs();
  const auto u16 = [&](int reg, uint16_t value) { kernel.set_regs(reg, value); };
  const auto u32 = [&](int reg, uint32_t value) { u16(reg, value & 0xffff); u16(reg + 1, value >> 16); };
  u32(0, input); u32(4, output); u32(6, 0);
  u16(8, rows); u32(10, local_n); u16(12, local_k); u16(13, local_k / 64);
  u16(14, 0); u16(15, 8); u32(16, local_k * 2); u32(18, local_n * 2);
  u32(20, block_scale); // Absolute local-SPM base, never a DDR operand.
  u32(22, 8 * local_n * 2); u32(24, tensor_scale); u16(26, col); u16(27, layer);
  u16(64, local_n / 64); u16(65, 1); u16(66, 1);
}
} // namespace

void rpu_launch_pi05_nvfp4_sspm_candidate_kernel(
    uint32_t input, const at::Tensor &weight, uint32_t output,
    int64_t rows, int64_t n, int64_t k, int partition, int cores,
    const at::Tensor &per_core_scale, uint32_t block_scale,
    uint32_t tensor_scale, uint16_t layer) {
  constexpr auto id = KernelId::PI05_NVFP4_LINEAR_DB_SSPM;
  pi_sspm_admit(id, rows, cores, layer);
  const bool col = partition == 1;
  TORCH_CHECK((col && n == 2048 && k == 1024) ||
                  (partition == 0 && n == 1024 && (k == 2048 || k == 4096)),
              "Pi SSPM Linear candidate admits only Q/O/Down geometry");
  const int64_t ln = col ? n / 8 : n, lk = col ? k : k / 8;
  const uint64_t scale_bytes = uint64_t(ln) * lk / 16;
  pi_sspm_check_owner(weight, weight, {n, k / 2});
  pi_sspm_check_owner(per_core_scale, weight, {8, ln / 64, lk / 128, 512});
  pi_sspm_check_ddr_disjoint({&weight, &per_core_scale});
  pi_sspm_check_spans({{input, uint64_t(rows) * lk * 2}, {output, uint64_t(rows) * ln * 2},
                       {block_scale, scale_bytes}, {tensor_scale, 24 * sizeof(float)}});
  pi_sspm_preload(per_core_scale, block_scale);
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight}});
  rpu_ddr_flush(weight.data_ptr());
  auto *kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Pi SSPM Linear candidate payload unavailable");
  pi_sspm_registers(*kernel, input, output, block_scale, tensor_scale, rows, ln, lk, col, layer);
  writer.write(*kernel);
  auto *queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {static_cast<uint16_t>(ln / 64), 1, 1}, {0,1,2,3,4,5,6,7});
}

void rpu_launch_pi05_nvfp4_geglu_sspm_candidate_kernel(
    uint32_t input, const at::Tensor &gate_weight, const at::Tensor &up_weight,
    uint32_t output, const at::Tensor &gate_per_core_scale, const at::Tensor &up_per_core_scale,
    uint32_t gate_block_scale, uint32_t up_block_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer) {
  constexpr auto id = KernelId::PI05_NVFP4_GEGLU_DB_SSPM;
  constexpr int64_t rows = 50, n = 4096, k = 1024, ln = 512;
  constexpr uint64_t scale_bytes = ln * k / 16;
  pi_sspm_admit(id, rows, 8, layer);
  for (const auto *weight : {&gate_weight, &up_weight})
    pi_sspm_check_owner(*weight, gate_weight, {n, k / 2});
  for (const auto *scale : {&gate_per_core_scale, &up_per_core_scale})
    pi_sspm_check_owner(*scale, gate_weight, {8, ln / 64, k / 128, 512});
  pi_sspm_check_ddr_disjoint({&gate_weight, &up_weight, &gate_per_core_scale, &up_per_core_scale});
  pi_sspm_check_spans({{input, rows * k * 2}, {output, rows * ln * 2},
                       {gate_block_scale, scale_bytes}, {up_block_scale, scale_bytes},
                       {gate_tensor_scale, 24 * sizeof(float)}, {up_tensor_scale, 24 * sizeof(float)}});
  pi_sspm_preload(gate_per_core_scale, gate_block_scale);
  pi_sspm_preload(up_per_core_scale, up_block_scale);
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, gate_weight},
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 32, 33}, up_weight}});
  rpu_ddr_flush(gate_weight.data_ptr()); rpu_ddr_flush(up_weight.data_ptr());
  auto *kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Pi SSPM GeGLU candidate payload unavailable");
  pi_sspm_registers(*kernel, input, output, gate_block_scale, gate_tensor_scale, rows, ln, k, true, layer);
  kernel->set_regs(34, static_cast<uint16_t>(up_block_scale));
  kernel->set_regs(35, static_cast<uint16_t>(up_block_scale >> 16));
  kernel->set_regs(36, static_cast<uint16_t>(up_tensor_scale));
  kernel->set_regs(37, static_cast<uint16_t>(up_tensor_scale >> 16));
  // Reset both 32-bit core barrier counters in every BUILD and REPLAY packet.
  kernel->set_regs(128, uint32_t{0}); kernel->set_regs(130, uint32_t{0});
  writer.write(*kernel);
  auto *queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0,1,2,3,4,5,6,7});
}

void rpu_launch_pi05_q_kv1_sspm_candidate_kernel(
    uint32_t input, const at::Tensor &query_weight,
    const at::Tensor &query_per_core_scale, uint32_t query_block_scale,
    uint32_t query_tensor_scale, uint32_t query_output,
    const at::Tensor &key_weight, const at::Tensor &key_scale, uint32_t key_output,
    const at::Tensor &value_weight, const at::Tensor &value_scale, uint32_t value_output,
    int64_t rows, uint16_t layer) {
  constexpr auto id = KernelId::PI05_Q_KV1_PAIR_DB_SSPM;
  pi_sspm_admit(id, rows, 8, layer);
  pi_sspm_check_owner(query_weight, query_weight, {2048, 512});
  pi_sspm_check_owner(query_per_core_scale, query_weight, {8, 4, 8, 512});
  const auto pair_owner = [&](const at::Tensor &owner, bool scale) {
    TORCH_CHECK(owner.defined() && owner.device() == query_weight.device() &&
                    owner.scalar_type() == (scale ? at::kHalf : at::kChar) &&
                    owner.is_contiguous() && owner.storage_offset() == 0 &&
                    (scale ? owner.dim() == 1 && owner.size(0) == 2048
                           : owner.dim() == 2 && owner.size(0) == 2048 && owner.size(1) == 1024),
                "Pi Q+KV1 requires complete int8 [2048,1024] weights and fp16 [2048] scales");
    const uint64_t address = RpuGetDevAddr(owner.data_ptr());
    constexpr uint64_t limit = UINT64_C(1) << 40;
    TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                    owner.nbytes() <= limit - address,
                "Pi Q+KV1 owner exceeds aligned shift8 range");
  };
  pair_owner(key_weight, false); pair_owner(value_weight, false);
  pair_owner(key_scale, true); pair_owner(value_scale, true);
  pi_sspm_check_ddr_disjoint({&query_weight, &query_per_core_scale,
                             &key_weight, &key_scale, &value_weight, &value_scale});
  pi_sspm_check_spans({{input, uint64_t(rows) * 2048},
                       {query_output, uint64_t(rows) * 512},
                       {key_output, uint64_t(rows) * 64},
                       {value_output, uint64_t(rows) * 64},
                       {query_block_scale, 16384}, {query_tensor_scale, 96}});
  // Stage the complete per-core slab on every BUILD and REPLAY. Its eight
  // DMA packets and fences belong to the candidate's device-call cost.
  pi_sspm_preload(query_per_core_scale, query_block_scale);
  auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id,
      std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::PackedFp4ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, query_weight},
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 34, 35}, key_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 52, 53}, key_scale},
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 54, 55}, value_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 56, 57}, value_scale}});
  for (const auto *owner : {&query_weight, &key_weight, &key_scale, &value_weight, &value_scale})
    rpu_ddr_flush(owner->data_ptr());
  auto *kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "Pi Q+KV1 candidate payload unavailable");
  kernel->reset_regs();
  const auto regs = rpu_pi05_q_kv1::registers({input, query_output, key_output, value_output,
      query_block_scale, query_tensor_scale, static_cast<uint16_t>(rows), layer});
  for (size_t index = 0; index < regs.size(); ++index) kernel->set_regs(index, regs[index]);
  writer.write(*kernel);
  auto *queue = GET_QUEUE(8);
  TORCH_CHECK(queue, "Pi Q+KV1 TP8 queue unavailable");
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {6, 1, 1}, {0,1,2,3,4,5,6,7});
}
