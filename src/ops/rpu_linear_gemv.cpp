#include "rhino_launch_buffer.h"

#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

namespace {

constexpr int kNumCores = 8;
constexpr uint32_t kFp16GemvMaxBufDepth = 16;
constexpr uint32_t kFp16GemvVectorWidth = 16;
constexpr uint32_t kFp16GemvWorkPerDepth = 256;

uint32_t select_fp16_gemv_buf_depth(uint64_t local_k, uint64_t local_n) {
  if (local_k == 0 || local_n == 0 ||
      local_k % kFp16GemvVectorWidth != 0 ||
      local_n % kFp16GemvVectorWidth != 0) {
    return 0;
  }

  const uint64_t work = local_k * local_n;
  // Depth must be a power of two and cannot exceed the per-core K vectors.
  for (uint32_t depth = kFp16GemvMaxBufDepth; depth != 0; depth >>= 1) {
    const uint64_t work_per_loop =
        static_cast<uint64_t>(depth) * kFp16GemvWorkPerDepth;
    if (local_k >= static_cast<uint64_t>(depth) * kFp16GemvVectorWidth &&
        work % work_per_loop == 0) {
      return depth;
    }
  }
  return 0;
}

void validate_weight(
    const at::Tensor& weight, int64_t n, int64_t k) {
  TORCH_CHECK(weight.defined(), "GEMV requires a defined weight tensor");
  TORCH_CHECK(
      weight.device().type() == c10::DeviceType::PrivateUse1 &&
          weight.is_contiguous() && weight.dim() == 2 &&
          weight.size(0) == n && weight.size(1) == k,
      "GEMV weight must be a contiguous RPU tensor [N,K]=[", n, ",", k,
      "], got ", weight.sizes(), " on ", weight.device());
}

}  // namespace

void rpu_launch_qwen3vl_4b_decode_o_ring_norm_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& weight,
    const at::Tensor& scale,
    uint32_t partial_spm_addr,
    uint32_t residual_spm_addr,
    uint32_t raw_spm_addr,
    uint32_t gamma_spm_addr,
    double eps) {
  constexpr int64_t kH = 2560;
  constexpr int64_t kK = 4096;
  constexpr uint32_t kLocalK = kK / 8;
  constexpr uint64_t kWeightBytes = kH * kK;
  constexpr uint64_t kScaleBytes = (kH + 256) * sizeof(c10::Half);
  constexpr uint64_t kRowBytes = kH * sizeof(c10::Half);
  constexpr uint32_t kCoreStride = 0x1000000;
  TORCH_CHECK(eps == 1e-6,
              "Qwen3-VL 4B decode O/ring/Norm requires epsilon 1e-6");
  auto check_tensor = [](const at::Tensor& tensor, bool is_scale) {
    TORCH_CHECK(tensor.defined() &&
                    tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.layout() == c10::Layout::Strided &&
                    tensor.scalar_type() == (is_scale ? at::kHalf : at::kChar) &&
                    tensor.is_contiguous() && tensor.dim() == (is_scale ? 1 : 2) &&
                    tensor.size(0) == kH && (is_scale || tensor.size(1) == kK),
                "Qwen3-VL 4B decode O/ring/Norm requires contiguous RPU "
                "int8 [2560,4096] weight and fp16 [2560] scale");
    const uint64_t element_bytes = is_scale ? sizeof(c10::Half) : 1;
    const uint64_t required_bytes = is_scale ? kScaleBytes : kWeightBytes;
    const int64_t offset = tensor.storage_offset();
    const auto& storage = tensor.storage();
    TORCH_CHECK(static_cast<bool>(storage) && offset >= 0 &&
                    static_cast<uint64_t>(offset) <= storage.nbytes() / element_bytes &&
                    required_bytes <= storage.nbytes() -
                        static_cast<uint64_t>(offset) * element_bytes,
                "Qwen3-VL 4B decode O/ring/Norm DDR backing is too short; "
                "scale requires 256 readable padding halves");
  };
  check_tensor(weight, false);
  check_tensor(scale, true);
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Qwen3-VL 4B decode O/ring/Norm requires initialized SPM");
  const uint32_t base = SPM_ALLOC.addr(0, 0);
  const std::array<uint32_t, 5> operands{
      input_spm_addr, partial_spm_addr, residual_spm_addr,
      raw_spm_addr, gamma_spm_addr};
  const std::array<uint64_t, 5> extents{
      kLocalK * sizeof(c10::Half), kRowBytes, kRowBytes, kRowBytes, kRowBytes};
  for (size_t i = 0; i < operands.size(); ++i) {
    TORCH_CHECK(operands[i] >= base && operands[i] % 256 == 0 &&
                    static_cast<uint64_t>(operands[i]) - base <=
                        SpmAllocator::SPM_USABLE - extents[i],
                "Qwen3-VL 4B decode O/ring/Norm requires 256-byte-aligned "
                "in-range absolute SPM operands");
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(static_cast<uint64_t>(operands[i]) + extents[i] <= operands[j] ||
                      static_cast<uint64_t>(operands[j]) + extents[j] <= operands[i],
                  "Qwen3-VL 4B decode O/ring/Norm SPM operands must be disjoint; "
                  "normalized output reuses only the consumed residual");
    }
  }
  for (int core = 0; core < 8; ++core) {
    TORCH_CHECK(static_cast<uint64_t>(SPM_ALLOC.addr(core, 0)) ==
                    static_cast<uint64_t>(base) + core * kCoreStride,
                "Qwen3-VL 4B decode O/ring/Norm requires the original GEMV "
                "eight-core SPM stride");
  }
  constexpr KernelId id = KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8;
  const uint64_t weight_address = RpuGetDevAddr(weight.data_ptr());
  const uint64_t scale_address = RpuGetDevAddr(scale.data_ptr());
  constexpr uint64_t kDdrLimit = uint64_t{1} << 40;
  TORCH_CHECK(weight_address && scale_address &&
                  weight_address % 256 == 0 && scale_address % 256 == 0 &&
                  weight_address <= kDdrLimit - kWeightBytes &&
                  scale_address <= kDdrLimit - kScaleBytes &&
                  (weight_address + kWeightBytes <= scale_address ||
                   scale_address + kScaleBytes <= weight_address),
              "Qwen3-VL 4B decode O/ring/Norm requires disjoint aligned "
              "40-bit weight and padded scale DDR ranges");
  auto& graph = RpuKernelGraph::active();
  auto register_writer = graph.stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 6, 7}, weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9}, scale}});
  Kernel_t* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr,
              "qwen3vl_o_ring_norm_m1_h2560_w8 missing from main ref; "
              "the cold COMPLETE owner must select an available route");
  for (const at::Tensor* tensor : {&weight, &scale}) {
    rpu_ddr_flush(tensor->data_ptr());
    graph.keep_alive(*tensor);
  }

  // reset_regs also zeros the atomic prefix submitted with the ring SCM.
  kernel->reset_regs();
  auto reg = [kernel](uint32_t index, uint16_t value) {
    kernel->set_regs(index, value);
  };
  auto pair = [&](uint32_t index, uint32_t value) {
    reg(index, static_cast<uint16_t>(value));
    reg(index + 1, static_cast<uint16_t>(value >> 16));
  };
  reg(0, kLocalK); reg(1, kH); reg(2, 0);
  pair(4, input_spm_addr >> 8);
  pair(10, partial_spm_addr >> 8);
  pair(12, kCoreStride >> 8); pair(14, kCoreStride >> 8);
  pair(16, 8 * 320 * kLocalK / 256);
  reg(18, 320); reg(19, 320); reg(21, 8);
  reg(22, 1); pair(24, 31 * 256 * 8);
  pair(26, residual_spm_addr - base); reg(28, 0);
  reg(29, c10::Half(1.0f / kH).x); reg(30, kRowBytes);
  pair(32, raw_spm_addr); pair(34, residual_spm_addr);
  pair(36, gamma_spm_addr); reg(38, c10::Half(static_cast<float>(eps)).x);
  reg(64, 8); reg(65, 1); reg(66, 1);
  for (int core = 0; core < 8; ++core) {
    const std::array<uint32_t, 4> values{
        SPM_ALLOC.addr(core, partial_spm_addr - base) - base,
        SPM_ALLOC.addr(core, raw_spm_addr - base) - base,
        static_cast<uint32_t>(core * 320 * sizeof(c10::Half)), 320};
    for (uint32_t group = 0; group < values.size(); ++group) {
      rpu_set_legacy_scm_u16_checked(
          kernel, 4096 + group * 16 + core,
          static_cast<uint16_t>(values[group]), "Qwen3-VL O/ring/Norm");
      rpu_set_legacy_scm_u16_checked(
          kernel, 4096 + group * 16 + 8 + core,
          static_cast<uint16_t>(values[group] >> 16), "Qwen3-VL O/ring/Norm");
    }
  }
  register_writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_qwen3vl_4b_decode_gate_up_swiglu_w8a16_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t output_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale) {
  constexpr int64_t kK = 2560;
  constexpr int64_t kN = 9728;
  constexpr int64_t kLocalN = kN / 8;
  auto check_tensor = [](const at::Tensor& tensor, bool scale) {
    TORCH_CHECK(tensor.defined() &&
                    tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.layout() == c10::Layout::Strided &&
                    tensor.scalar_type() == (scale ? at::kHalf : at::kChar) &&
                    tensor.is_contiguous() && tensor.dim() == (scale ? 1 : 2) &&
                    tensor.size(0) == kN && (scale || tensor.size(1) == kK),
                "Qwen3-VL 4B decode fused Gate/Up requires contiguous RPU "
                "int8 [9728,2560] weights and fp16 [9728] scales");
    // The unchanged payload reads 256 scales for the final 96-channel warp.
    // Check the underlying storage, not merely the logical view size.
    const uint64_t required_bytes = scale ? (kN + 160) * 2 : kN * kK;
    const uint64_t element_bytes = scale ? 2 : 1;
    const int64_t offset = tensor.storage_offset();
    const auto& storage = tensor.storage();
    TORCH_CHECK(static_cast<bool>(storage) && offset >= 0 &&
                    static_cast<uint64_t>(offset) <= storage.nbytes() / element_bytes &&
                    required_bytes <= storage.nbytes() -
                        static_cast<uint64_t>(offset) * element_bytes,
                "Qwen3-VL 4B decode fused Gate/Up DDR backing is too short; "
                "each scale needs 160 readable padding halves");
  };
  check_tensor(gate_weight, false);
  check_tensor(up_weight, false);
  check_tensor(gate_scale, true);
  check_tensor(up_scale, true);
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Qwen3-VL 4B decode fused Gate/Up requires initialized SPM");
  const uint32_t base = SPM_ALLOC.addr(0, 0);
  constexpr uint64_t input_bytes = kK * sizeof(c10::Half);
  constexpr uint64_t output_bytes = kLocalN * sizeof(c10::Half);
  const auto in_spm = [base](uint32_t address, uint64_t bytes) {
    return address >= base && address % 256 == 0 &&
        bytes <= SpmAllocator::SPM_USABLE &&
        static_cast<uint64_t>(address) - base <= SpmAllocator::SPM_USABLE - bytes;
  };
  TORCH_CHECK(in_spm(input_spm_addr, input_bytes) &&
                  in_spm(output_spm_addr, output_bytes) &&
                  (static_cast<uint64_t>(input_spm_addr) + input_bytes <= output_spm_addr ||
                   static_cast<uint64_t>(output_spm_addr) + output_bytes <= input_spm_addr),
              "Qwen3-VL 4B decode fused Gate/Up requires disjoint, "
              "256-byte-aligned in-range absolute SPM operands");
  constexpr KernelId id = KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV;
  Kernel_t* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel != nullptr,
              "qwen3vl_gateup_swiglu_gemv missing from main ref; "
              "the cold COMPLETE owner must select an available route");
  auto register_writer = RpuKernelGraph::active().stage_kernel_ddr_registers(
      id, std::vector<GraphDdrRegisterOperandSpec>{
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 6, 7}, gate_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 8, 9}, gate_scale},
          {{GraphDdrRegisterRole::Int8ModelWeight, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, up_weight},
          {{GraphDdrRegisterRole::PerChannelScale, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, 26, 27}, up_scale}});
  for (const at::Tensor* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
    rpu_ddr_flush(tensor->data_ptr());
    RpuKernelGraph::active().keep_alive(*tensor);
  }
  kernel->reset_regs();
  kernel->set_regs(0, static_cast<uint16_t>(kK));
  kernel->set_regs(1, static_cast<uint16_t>(kLocalN));
  kernel->set_regs(2, static_cast<uint16_t>(1));
  kernel->set_regs(4, static_cast<uint16_t>(input_spm_addr >> 8));
  kernel->set_regs(5, static_cast<uint16_t>(input_spm_addr >> 24));
  kernel->set_regs(10, static_cast<uint16_t>(output_spm_addr >> 8));
  kernel->set_regs(11, static_cast<uint16_t>(output_spm_addr >> 24));
  kernel->set_regs(12, static_cast<uint16_t>(0));
  kernel->set_regs(13, static_cast<uint16_t>(1));
  kernel->set_regs(14, static_cast<uint16_t>(0));
  kernel->set_regs(15, static_cast<uint16_t>(1));
  kernel->set_regs(16, static_cast<uint16_t>(12800));
  kernel->set_regs(17, static_cast<uint16_t>(0));
  kernel->set_regs(18, static_cast<uint16_t>(160));
  kernel->set_regs(19, static_cast<uint16_t>(96));
  kernel->set_regs(21, static_cast<uint16_t>(8));
  kernel->set_regs(64, static_cast<uint16_t>(8));
  kernel->set_regs(65, static_cast<uint16_t>(1));
  kernel->set_regs(66, static_cast<uint16_t>(1));
  register_writer.write(*kernel);
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

// Direct-address SPM GEMV launcher for the shared Linear M=1 opt-in. Dispatch
// policy and post-kernel bias handling stay in rpu_linear.cpp.
void rpu_launch_gemv_spm_to_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor &weight,
    uint32_t output_spm_addr,
    int64_t m,
    int64_t n,
    int64_t k,
    int partition,
    int num_cores,
    const at::Tensor &scale) {
  TORCH_CHECK(m == 1, "GEMV kernel only supports M=1, got M=", m);
  TORCH_CHECK(n > 0 && k > 0,
              "GEMV dimensions N and K must be positive, got N=", n,
              ", K=", k);
  TORCH_CHECK(partition == 0 || partition == 1,
              "GEMV partition must be 0 (row) or 1 (col), got ", partition);
  TORCH_CHECK(num_cores >= 1 && num_cores <= kNumCores,
              "GEMV num_cores must be in [1, ", kNumCores, "], got ",
              num_cores);
  validate_weight(weight, n, k);

  if (partition == 1) {
    TORCH_CHECK(n % num_cores == 0,
                "Col partition: N must be divisible by num_cores (",
                num_cores, ")");
  } else {
    TORCH_CHECK(k % num_cores == 0,
                "Row partition: K must be divisible by num_cores (",
                num_cores, ")");
  }

  const size_t local_k =
      partition == 0 ? static_cast<size_t>(k / num_cores)
                     : static_cast<size_t>(k);
  const size_t local_n =
      partition == 1 ? static_cast<size_t>(n / num_cores)
                     : static_cast<size_t>(n);
  TORCH_CHECK(local_k <= UINT16_MAX && local_n <= UINT16_MAX,
              "GEMV local dimensions exceed 16-bit kernel registers: local_k=",
              local_k, ", local_n=", local_n);
  TORCH_CHECK(local_n % 16 == 0,
              "GEMV SPM kernel requires local_n % 16 == 0");
  TORCH_CHECK(input_spm_addr % 256 == 0,
              "GEMV SPM kernel requires input SPM address 256-byte aligned");
  TORCH_CHECK(output_spm_addr % 256 == 0,
              "GEMV SPM kernel requires output SPM address 256-byte aligned");

  if (weight.scalar_type() == at::kChar) {
    TORCH_CHECK(scale.defined(),
                "GEMV W8A16 requires a defined scale tensor");
    TORCH_CHECK(
        scale.device().type() == c10::DeviceType::PrivateUse1 &&
            scale.scalar_type() == at::kHalf && scale.is_contiguous() &&
            scale.dim() == 1 && scale.size(0) == n,
        "GEMV W8A16 scale must be contiguous RPU fp16 [N]=[", n,
        "], got ", scale.sizes(), " on ", scale.device());
    TORCH_CHECK(local_k % 32 == 0,
                "GEMV W8A16 local_k=", local_k,
                " must be divisible by 32");

    auto register_writer =
        RpuKernelGraph::active().stage_kernel_ddr_registers(
            KernelId::LLAMA_GEMV_WINT8,
            std::vector<GraphDdrRegisterOperandSpec>{
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::Int8ModelWeight,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        6, 7},
                    weight},
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::PerChannelScale,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        8, 9},
                    scale}});

    auto* weight_ptr = weight.data_ptr<int8_t>();
    auto* scale_ptr = scale.data_ptr<c10::Half>();
    rpu_ddr_flush(weight_ptr);
    rpu_ddr_flush(scale_ptr);
    TORCH_CHECK(RpuGetDevAddr(weight_ptr) % 256 == 0,
                "GEMV W8A16 weight DDR address not 256-byte aligned");
    TORCH_CHECK(RpuGetDevAddr(scale_ptr) % 256 == 0,
                "GEMV W8A16 scale DDR address not 256-byte aligned");

    int block_count = std::min(8, static_cast<int>(local_n / 16));
    auto block_rows = [&] {
      return ((static_cast<int>(local_n) + block_count * 16 - 1) /
              (block_count * 16)) * 16;
    };
    int normal_block_n = block_rows();
    // The kernel has an independent final-block extent (reg19). Match the
    // runtime's ceil16 partition, retaining all available warps when a short
    // tail is legal. Small N can yield an empty tail; reduce the grid then.
    while (block_count > 1 &&
           static_cast<int>(local_n) <= (block_count - 1) * normal_block_n) {
      --block_count;
      normal_block_n = block_rows();
    }
    const int last_block_n =
        static_cast<int>(local_n) - (block_count - 1) * normal_block_n;
    TORCH_CHECK(last_block_n > 0 && last_block_n <= normal_block_n &&
                    last_block_n % 16 == 0,
                "GEMV W8A16 invalid block partition local_n=", local_n,
                " block_count=", block_count,
                " normal_block_n=", normal_block_n,
                " last_block_n=", last_block_n);
    TORCH_CHECK(
        (block_count - 1) * normal_block_n + last_block_n ==
            static_cast<int>(local_n),
        "GEMV W8A16 block partition does not tile local_n=", local_n);
    TORCH_CHECK(normal_block_n <= 4096,
                "GEMV W8A16 normal_block_n=", normal_block_n,
                " exceeds 4096");
    TORCH_CHECK(
        static_cast<uint64_t>(num_cores) * normal_block_n * local_k % 256 == 0,
        "GEMV W8A16 weight block stride is not a 256-byte unit");
    const uint32_t weight_block_stride = static_cast<uint32_t>(
        static_cast<uint64_t>(num_cores) * normal_block_n * local_k / 256);

    constexpr uint32_t input_core_stride = 0x10000;
    constexpr uint32_t output_core_stride = 0x10000;
    auto setup_wint8 = [=](Kernel_t *kernel) {
      kernel->set_regs(0, static_cast<uint16_t>(local_k));
      kernel->set_regs(1, static_cast<uint16_t>(local_n));
      kernel->set_regs(2, static_cast<uint16_t>(partition == 1));
      kernel->set_regs(4, static_cast<uint16_t>((input_spm_addr >> 8) & 0xFFFF));
      kernel->set_regs(5, static_cast<uint16_t>((input_spm_addr >> 24) & 0xFFFF));
      kernel->set_regs(10, static_cast<uint16_t>((output_spm_addr >> 8) & 0xFFFF));
      kernel->set_regs(11, static_cast<uint16_t>((output_spm_addr >> 24) & 0xFFFF));
      kernel->set_regs(12, static_cast<uint16_t>(input_core_stride & 0xFFFF));
      kernel->set_regs(13, static_cast<uint16_t>(input_core_stride >> 16));
      kernel->set_regs(14, static_cast<uint16_t>(output_core_stride & 0xFFFF));
      kernel->set_regs(15, static_cast<uint16_t>(output_core_stride >> 16));
      kernel->set_regs(16, static_cast<uint16_t>(weight_block_stride & 0xFFFF));
      kernel->set_regs(17, static_cast<uint16_t>(weight_block_stride >> 16));
      kernel->set_regs(18, static_cast<uint16_t>(normal_block_n));
      kernel->set_regs(19, static_cast<uint16_t>(last_block_n));
      kernel->set_regs(21, static_cast<uint16_t>(num_cores));
      kernel->set_regs(64, static_cast<uint16_t>(block_count));
      kernel->set_regs(65, static_cast<uint16_t>(1));
      kernel->set_regs(66, static_cast<uint16_t>(1));
    };

    Kernel_t *kernel = GET_KERNEL(KernelId::LLAMA_GEMV_WINT8);
    TORCH_CHECK(kernel != nullptr, "Failed to get llama_gemv_wint8 kernel");
    kernel->reset_regs();
    setup_wint8(kernel);
    register_writer.write(*kernel);

    auto *queue = GET_QUEUE(num_cores);
    queue->set_broadcast_mode(true);
    std::vector<uint8_t> cores;
    for (int core = 0; core < num_cores; ++core) cores.push_back(core);
    queue->enqueu_kernel(
        *kernel, {static_cast<uint16_t>(block_count), 1, 1}, cores);
    return;
  }

  TORCH_CHECK(weight.scalar_type() == at::kHalf,
              "GEMV supports fp16 or int8 weights, got ",
              weight.scalar_type());
  TORCH_CHECK(!scale.defined() || scale.numel() == 0,
              "FP16 GEMV does not accept a quantization scale");
  const uint32_t buffer_depth =
      select_fp16_gemv_buf_depth(local_k, local_n);
  TORCH_CHECK(
      buffer_depth != 0,
      "FP16 GEMV requires positive 16-aligned local_k/local_n and a legal "
      "power-of-two buffer depth, got local_k=",
      local_k, ", local_n=", local_n);
  const uint64_t work_per_loop =
      static_cast<uint64_t>(buffer_depth) * kFp16GemvWorkPerDepth;
  const uint64_t loop_count =
      static_cast<uint64_t>(local_k) * local_n / work_per_loop + 1;
  TORCH_CHECK(loop_count <= UINT16_MAX,
              "FP16 GEMV loop count exceeds the 16-bit kernel register: ",
              loop_count);

  auto register_writer =
      RpuKernelGraph::active().stage_kernel_ddr_registers(
          KernelId::LLAMA_GEMV,
          std::vector<GraphDdrRegisterOperandSpec>{
              GraphDdrRegisterOperandSpec{
                  GraphDdrRegisterAbi{
                      GraphDdrRegisterRole::ModelWeight,
                      GraphDdrRegisterAccess::Read,
                      GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                      6, 7},
                  weight}});

  auto* weight_ptr = weight.data_ptr<c10::Half>();
  rpu_ddr_flush(weight_ptr);
  TORCH_CHECK(RpuGetDevAddr(weight_ptr) % 256 == 0,
              "FP16 GEMV weight DDR address not 256-byte aligned");

  constexpr uint32_t input_core_stride = 0x10000;
  constexpr uint32_t output_core_stride = 0x10000;
  auto setup_fp16 = [=](Kernel_t *kernel) {
    kernel->set_regs(0, static_cast<uint16_t>(local_k));
    kernel->set_regs(1, static_cast<uint16_t>(local_n));
    kernel->set_regs(2, static_cast<uint16_t>(loop_count));
    kernel->set_regs(3, static_cast<uint16_t>(buffer_depth));
    kernel->set_regs(4, static_cast<uint16_t>((input_spm_addr >> 8) & 0xFFFF));
    kernel->set_regs(5, static_cast<uint16_t>((input_spm_addr >> 24) & 0xFFFF));
    kernel->set_regs(8, static_cast<uint16_t>((output_spm_addr >> 8) & 0xFFFF));
    kernel->set_regs(9, static_cast<uint16_t>((output_spm_addr >> 24) & 0xFFFF));
    kernel->set_regs(10, static_cast<uint16_t>(input_core_stride & 0xFFFF));
    kernel->set_regs(11, static_cast<uint16_t>(input_core_stride >> 16));
    kernel->set_regs(12, static_cast<uint16_t>(output_core_stride & 0xFFFF));
    kernel->set_regs(13, static_cast<uint16_t>(output_core_stride >> 16));
    kernel->set_regs(21, static_cast<uint16_t>(num_cores));
    kernel->set_regs(64, static_cast<uint16_t>(1));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));
  };

  Kernel_t *kernel = GET_KERNEL(KernelId::LLAMA_GEMV);
  TORCH_CHECK(kernel != nullptr, "Failed to get llama_gemv kernel");
  kernel->reset_regs();
  setup_fp16(kernel);
  register_writer.write(*kernel);

  auto *queue = GET_QUEUE(num_cores);
  queue->set_broadcast_mode(true);
  std::vector<uint8_t> cores;
  for (int core = 0; core < num_cores; ++core) cores.push_back(core);
  queue->enqueu_kernel(*kernel, {1, 1, 1}, cores);
}
