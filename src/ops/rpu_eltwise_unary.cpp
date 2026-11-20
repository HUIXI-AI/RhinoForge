#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h" // for Program_t
#include "rhino_launch_queue.h"

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <c10/util/Half.h>
#include <stdlib.h> // for uint16_t, setenv, unsetenv
#include <string.h> // for memcpy
#include <string>   // for string
#include <vector>   // for vector

using namespace ::rhino_lkn;

at::Tensor rpu_gelu_erf(const at::Tensor& input) {
  TORCH_CHECK(input.device().type() == at::kPrivateUse1 &&
                  input.scalar_type() == at::kHalf && input.is_contiguous(),
              "gelu_erf requires contiguous device FP16 input");
  TORCH_CHECK(input.numel() > 0 && input.numel() % 8 == 0,
              "gelu_erf requires a positive 16-byte-aligned byte count");
  TORCH_CHECK(!graph_dma::active(),
              "gelu_erf is PASSTHROUGH only; fused graphs use the SPM launcher");
  auto output = at::empty(input.sizes(), input.options());
  if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
  SpmAllocator::ScopedTemporary temporary;
  constexpr int64_t block = 25600;
  const uint32_t spm = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(
      std::min(input.numel(), block) * sizeof(c10::Half)));
  ScopedEagerAtomicBatch batch(/*enabled=*/true, /*num_cores=*/1);
  const size_t flush_threshold = ScopedEagerAtomicBatch::default_flush_node_threshold();
  for (int64_t offset = 0; offset < input.numel(); offset += block) {
    if (batch.recorded_node_count() >= flush_threshold) {
      batch.flush_and_reopen();
    }
    const int64_t count = std::min(input.numel() - offset, block);
    rpu_launch_ddr_broadcast_spm_dma(
        input.data_ptr<c10::Half>() + offset, count, spm, 1);
    rpu_launch_eltwise_unary_spm_kernel(
        spm, spm, count, ValuOpType::ADD, GeluMode::ERF, 1);
    rpu_launch_spm_copy_ddr_dma(
        spm, output.data_ptr<c10::Half>() + offset, count);
  }
  batch.submit();
  rpu_ddr_flush(output.data_ptr<c10::Half>());
  return output;
}

void rpu_require_high_precision_math_kernels() {
  const auto& cache = KernelCache::instance();
  for (const auto id : {KernelId::UNARY_SILU_HIGH_PRECISION,
                        KernelId::UNARY_TANH_HIGH_PRECISION,
                        KernelId::UNARY_GELU_ERF_HIGH_PRECISION,
                        KernelId::UNARY_GELU_TANH_HIGH_PRECISION,
                        KernelId::RMS_NORM_NEWTON_SPM,
                        KernelId::RMS_NORM_SPM_NEWTON_V16,
                        KernelId::RMS_NORM_SPM_NEWTON_V32}) {
    TORCH_CHECK(cache.has_loaded(id),
                "High-precision math requires loaded optional payload ",
                KERNEL_ID_NAMES[static_cast<size_t>(id)]);
  }
}

void rpu_launch_eltwise_unary_kernel(const at::Tensor &input,
                                     at::Tensor &output, ValuOpType op_type,
                                     RpuUnaryPrecision precision) {
  TORCH_CHECK(precision == RpuUnaryPrecision::BASE ||
                  precision == RpuUnaryPrecision::HIGH,
              "Unknown unary precision ", static_cast<int>(precision));
  if (precision == RpuUnaryPrecision::HIGH) {
    // HIGH payloads use absolute SPM addressing. Stage a bounded block
    // through SPM so DDR addresses
    // above 4GiB retain all bits in the existing typed DMA implementation.
    TORCH_CHECK(op_type == ValuOpType::SILU || op_type == ValuOpType::TANH,
                "DDR HIGH unary supports SiLU and TANH only");
    const auto id = op_type == ValuOpType::SILU
        ? KernelId::UNARY_SILU_HIGH_PRECISION : KernelId::UNARY_TANH_HIGH_PRECISION;
    TORCH_CHECK(!graph_dma::active(),
                "DDR HIGH unary staging is cold/PASSTHROUGH only; use the "
                "explicit SPM launcher with graph-owned buffers in capture");
    TORCH_CHECK(GET_KERNEL(id) != nullptr,
                "Selected HIGH unary payload is unavailable");
    TORCH_CHECK(input.device().type() == at::kPrivateUse1 &&
                    output.device() == input.device() &&
                    input.scalar_type() == at::kHalf &&
                    output.scalar_type() == at::kHalf &&
                    input.is_contiguous() && output.is_contiguous() &&
                    input.sizes() == output.sizes(),
                "DDR HIGH unary requires matching contiguous device FP16 tensors");
    const int64_t count = input.numel();
    TORCH_CHECK(count > 0 && count % 8 == 0,
                "DDR HIGH unary needs a positive 16-byte-aligned byte count");
    auto* in = input.data_ptr<c10::Half>();
    auto* out = output.data_ptr<c10::Half>();
    const uintptr_t in_start = reinterpret_cast<uintptr_t>(in);
    const uintptr_t out_start = reinterpret_cast<uintptr_t>(out);
    const uint64_t distance = in_start < out_start
        ? out_start - in_start : in_start - out_start;
    TORCH_CHECK(in == out || distance >= uint64_t(count) * sizeof(c10::Half),
                "DDR HIGH unary does not support partially overlapping tensors");
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    SpmAllocator::ScopedTemporary temporary;
    const int64_t block = op_type == ValuOpType::SILU ? 15872 : 25600;
    const uint32_t spm = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(
        static_cast<size_t>(std::min(count, block)) * sizeof(c10::Half)));
    ScopedEagerAtomicBatch batch(/*enabled=*/true, /*num_cores=*/1);
    const size_t flush_threshold = ScopedEagerAtomicBatch::default_flush_node_threshold();
    for (int64_t offset = 0; offset < count; offset += block) {
      if (batch.recorded_node_count() >= flush_threshold) {
        batch.flush_and_reopen();
      }
      const int64_t elements = std::min(count - offset, block);
      rpu_launch_ddr_broadcast_spm_dma(in + offset, elements, spm, 1);
      rpu_launch_eltwise_unary_spm_kernel(
          spm, spm, elements, op_type, GeluMode::NONE, 1, precision);
      rpu_launch_spm_copy_ddr_dma(spm, out + offset, elements);
    }
    batch.submit();
    rpu_ddr_flush(out);
    return;
  }

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
  // block_data_stride also uses v128 units.
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
    int num_cores,                     // number of cores
    RpuUnaryPrecision precision, double clip_lower, double clip_upper)
{
  TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
              "Unary SPM num_cores must be in [1,8]");
  TORCH_CHECK(num_elements > 0, "Unary SPM element count must be positive");
  TORCH_CHECK(std::isfinite(clip_lower) && std::isfinite(clip_upper) &&
                  clip_lower <= clip_upper && clip_lower >= -65504 && clip_upper <= 65504,
              "Unary SPM clip bounds must be ordered finite FP16 values");
  TORCH_CHECK((clip_lower == 0.0 && clip_upper == 0.0) ||
                  (op_type == ValuOpType::CLIP && gelu_mode == GeluMode::NONE &&
                   precision == RpuUnaryPrecision::BASE),
              "Unary SPM clip bounds require the CLIP route");
  TORCH_CHECK(precision == RpuUnaryPrecision::BASE ||
                  precision == RpuUnaryPrecision::HIGH,
              "Unknown unary precision ", static_cast<int>(precision));
  size_t dwidth = sizeof(c10::Half);
  size_t n = num_elements;
  // ERF_ULTRA requires a smaller block; SPM input/output may alias.
  size_t normal_blk_n = gelu_mode == GeluMode::ERF_ULTRA ? 5888 : 100 * 256;
  KernelId kid = KernelId::UNARY_SPM;
  if (precision == RpuUnaryPrecision::HIGH) {
    // Precision never changes the model's GELU formula.
    if (gelu_mode == GeluMode::TANH) {
      TORCH_CHECK(op_type == ValuOpType::ADD,
                  "TANH GELU must borrow ADD for datatype configuration");
      kid = KernelId::UNARY_GELU_TANH_HIGH_PRECISION;
      normal_blk_n = 20480;
    } else if (gelu_mode == GeluMode::ERF) {
      TORCH_CHECK(op_type == ValuOpType::ADD,
                  "ERF GELU must borrow ADD for datatype configuration");
      kid = KernelId::UNARY_GELU_ERF_HIGH_PRECISION;
      normal_blk_n = 15360;
    } else {
      TORCH_CHECK(gelu_mode == GeluMode::NONE,
                  "HIGH unary does not provide ERF_ULTRA; "
                  "keep the explicitly selected formula");
      if (op_type == ValuOpType::SILU) {
        kid = KernelId::UNARY_SILU_HIGH_PRECISION;
        normal_blk_n = 15872;
      } else {
        TORCH_CHECK(op_type == ValuOpType::TANH,
                    "HIGH unary supports SiLU, TANH and explicit TANH/ERF GELU only");
        kid = KernelId::UNARY_TANH_HIGH_PRECISION;
        normal_blk_n = 25600;
      }
    }
  } else if (gelu_mode == GeluMode::TANH) {
    kid = KernelId::UNARY_GELU_TANH_SPM;
  } else if (gelu_mode == GeluMode::ERF) {
    kid = KernelId::UNARY_GELU_ERF_SPM;
  } else if (gelu_mode == GeluMode::ERF_ULTRA) {
    kid = KernelId::UNARY_GELU_ERF_ULTRA_SPM;
  } else {
    TORCH_CHECK(gelu_mode == GeluMode::NONE, "Unknown GELU formula");
    if (op_type == ValuOpType::SOFTPLUS) kid = KernelId::UNARY_SOFTPLUS_SPM;
  }
  size_t blk_cnt = CeilDiv(n, normal_blk_n);
  TORCH_CHECK(blk_cnt <= UINT16_MAX, "Unary SPM block grid exceeds uint16");
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
  c10::Half clip_min = static_cast<c10::Half>(clip_lower);
  c10::Half clip_max = static_cast<c10::Half>(clip_upper);

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

void rpu_launch_g05_attn_sigmoid_round_mul_spm_kernel(uint32_t gate, uint32_t output) {
  // Exact G05 Action M32/localD256. Preserve the complete original Sigmoid
  // output in gate, then the independently rounded product in output.
  auto& graph = RpuKernelGraph::active();
  TORCH_CHECK(graph.state() == RpuKernelGraph::State::RECORDING ||
                  graph.state() == RpuKernelGraph::State::REPLAYING,
              "G05 attention Sigmoid/Mul requires a retained Graph");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 attention Sigmoid/Mul requires live SPM");
  constexpr uint64_t bytes = 32 * 256 * sizeof(c10::Half);
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  for (const auto address : {gate, output}) {
    TORCH_CHECK(address % 256 == 0 && address >= base &&
                    uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE,
                "G05 attention Sigmoid/Mul requires complete aligned SPM roots");
  }
  TORCH_CHECK(uint64_t(gate) + bytes <= output || uint64_t(output) + bytes <= gate,
              "G05 attention Sigmoid/Mul requires disjoint gate/output roots");
  constexpr auto id = KernelId::G05_ATTN_SIGMOID_ROUND_MUL_M32D256;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
              "G05 attention Sigmoid/Mul optional payload is unavailable");
  // Like the original unary/binary launchers, this ABI has only absolute SPM
  // operands. It introduces no DDR owner/register or new census policy.
  auto* kernel = GET_KERNEL(id);
  TORCH_CHECK(kernel, "G05 attention Sigmoid/Mul payload missing");
  kernel->reset_regs();
  const auto set = [&](uint32_t reg, uint16_t value) {
    TORCH_CHECK(kernel->set_regs(reg, value) == kKernelRegOk,
                "G05 attention Sigmoid/Mul rejected register ", reg);
  };
  const auto pair = [&](uint32_t reg, uint32_t value) {
    set(reg, uint16_t(value)); set(reg + 1, uint16_t(value >> 16));
  };
  set(0, 1024); set(1, 1024);
  pair(4, gate); pair(6, gate); pair(8, 2048); pair(12, output);
  for (uint32_t reg : {56, 57, 58}) set(reg, c10::Half(1.0).x);
  set(59, 0); set(60, 0); set(61, 1);
  set(62, 0x2692); set(63, 0); // original BASE Sigmoid op19/custom0
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

void rpu_launch_g05_silu_round_mul_spm_kernel(uint32_t gate, uint32_t up) {
  constexpr uint64_t bytes = 32 * 512 * sizeof(c10::Half);
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "G05 rounded SiLU/Mul requires live SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  const auto valid = [=](uint32_t address) {
    return address % 256 == 0 && address >= base &&
        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE;
  };
  TORCH_CHECK(valid(gate) && valid(up) &&
                  (uint64_t(gate) + bytes <= up || uint64_t(up) + bytes <= gate),
              "G05 rounded SiLU/Mul requires two disjoint aligned SPM operands");
  TORCH_CHECK(KernelCache::instance().has_loaded(KernelId::G05_SILU_ROUND_MUL_M32N512),
              "G05 rounded SiLU/Mul payload is unavailable");
  auto* kernel = GET_KERNEL(KernelId::G05_SILU_ROUND_MUL_M32N512);
  TORCH_CHECK(kernel, "G05 rounded SiLU/Mul kernel is unavailable");
  kernel->reset_regs();
  kernel->set_regs(0, uint16_t(2048));
  kernel->set_regs(1, uint16_t(2048));
  for (const auto& field : {std::pair<uint32_t,uint32_t>{4,gate}, {6,gate},
                            {8,4096}, {12,up}}) {
    kernel->set_regs(field.first, uint16_t(field.second));
    kernel->set_regs(field.first + 1, uint16_t(field.second >> 16));
  }
  for (int reg : {56,57,58}) kernel->set_regs(reg, c10::Half(1.0).x);
  for (int reg : {59,60}) kernel->set_regs(reg, uint16_t(0));
  kernel->set_regs(61, uint16_t(1));
  const auto& op = get_valu_op_info(ValuOpType::SILU);
  kernel->set_regs(62, uint16_t((op.op << 9) | FP16_RAB_DATA_TYPE_CFG));
  kernel->set_regs(63, uint16_t(op.custom_op));
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {8,1,1}, {0,1,2,3,4,5,6,7});
}

void rpu_launch_g05_text_silu_round_mul_spm_kernel(
    uint32_t gate,uint32_t up,uint32_t product) {
  auto& graph=RpuKernelGraph::active();
  TORCH_CHECK(graph.state()==RpuKernelGraph::State::RECORDING ||
      graph.state()==RpuKernelGraph::State::REPLAYING,
      "G05 Text rounded SiLU/MUL requires a retained Graph");
  TORCH_CHECK(SPM_ALLOC.is_initialized(),"G05 Text rounded SiLU/MUL requires live SPM");
  constexpr uint64_t part=256*768*sizeof(c10::Half),full=256*2048*sizeof(c10::Half);
  const uint64_t base=SPM_ALLOC.addr(0,0);
  const std::array<std::pair<uint32_t,uint64_t>,3> spans{{{gate,part},{up,part},{product,full}}};
  for (const auto& s:spans)
    TORCH_CHECK(s.first%256==0 && s.first>=base &&
        uint64_t(s.first)-base+s.second<=SpmAllocator::SPM_USABLE,
        "G05 Text rounded SiLU/MUL requires complete aligned SPM roots");
  for (size_t i=0;i<spans.size();++i) for (size_t j=i+1;j<spans.size();++j)
    TORCH_CHECK(uint64_t(spans[i].first)+spans[i].second<=spans[j].first ||
        uint64_t(spans[j].first)+spans[j].second<=spans[i].first,
        "G05 Text rounded SiLU/MUL SPM roots overlap");
  constexpr auto id=KernelId::G05_SILU_ROUND_MUL_M32N512;
  TORCH_CHECK(KernelCache::instance().has_loaded(id),
      "G05 Text rounded SiLU/MUL optional payload is unavailable");
  // Same intrinsic no-DDR boundary as ordinary unary/binary and Action rounded
  // SiLU. Only absolute SPM operands; no new census enum or DDR register.
  auto* kernel=GET_KERNEL(id);TORCH_CHECK(kernel,"G05 Text rounded SiLU/MUL payload missing");
  kernel->reset_regs();
  const auto set=[&](uint32_t reg,uint16_t value) {
    TORCH_CHECK(kernel->set_regs(reg,value)==kKernelRegOk,
        "G05 Text rounded SiLU/MUL rejected register ",reg);
  };
  set(0,25600);set(1,17408);
  for (const auto& p:{std::pair<uint32_t,uint32_t>{4,gate},{6,product},{8,51200},{12,up}}) {
    set(p.first,uint16_t(p.second));set(p.first+1,uint16_t(p.second>>16));
  }
  for (int reg:{56,57,58})set(reg,c10::Half(1.0).x);
  for (int reg:{59,60})set(reg,0);
  set(61,1);set(62,0x3e92);set(63,134);
  auto* queue=GET_QUEUE(8);queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel,{8,1,1},{0,1,2,3,4,5,6,7});
}

void rpu_launch_silu_mul_exact_strided_spm_kernel(
    uint32_t packed_spm_addr,
    uint32_t out_spm_addr,
    int64_t rows,
    int64_t row_elements,
    int num_cores)
{
  TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
              "exact strided silu_mul requires 1..8 cores");
  TORCH_CHECK(rows > 0 && rows <= UINT16_MAX,
              "exact strided silu_mul rows out of range: ", rows);
  // The signed row loop masks vector lanes in its final 256-element block.
  // Lingbot grouped Gate/Up uses 64 channels per core; the existing Qwen
  // route uses 768. Require 64-channel row alignment for these paths.
  TORCH_CHECK(row_elements > 0 && row_elements <= INT16_MAX &&
                  row_elements % 64 == 0,
              "exact strided silu_mul row_elements out of range: ", row_elements);
  const uint32_t up_spm_addr = packed_spm_addr +
      static_cast<uint32_t>(row_elements * sizeof(c10::Half));

  Kernel_t* kernel = GET_KERNEL(KernelId::QWEN3VL_SILU_MUL_EXACT_STRIDED);
  TORCH_CHECK(kernel != nullptr,
              "Failed to get llama_silu_mul_exact_strided SPM kernel from cache");
  kernel->reset_regs();
  kernel->set_regs(0, static_cast<uint16_t>(row_elements));
  kernel->set_regs(1, static_cast<uint16_t>(row_elements));
  kernel->set_regs(4, static_cast<uint16_t>(packed_spm_addr & 0xFFFF));
  kernel->set_regs(5, static_cast<uint16_t>(packed_spm_addr >> 16));
  kernel->set_regs(6, static_cast<uint16_t>(up_spm_addr & 0xFFFF));
  kernel->set_regs(7, static_cast<uint16_t>(up_spm_addr >> 16));
  kernel->set_regs(8, static_cast<uint16_t>(out_spm_addr & 0xFFFF));
  kernel->set_regs(9, static_cast<uint16_t>(out_spm_addr >> 16));

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
  wq->enqueu_kernel(*kernel,
                    {static_cast<uint16_t>(rows), 1, 1}, core_list);
}

// Exact shared Gate/Up profile. Keep the existing custom SiLU math and consume
// per-row [gate96,up96], with a separate compact [51,96] output on every core.
void rpu_launch_lingbot2_shared_silu_d96_spm_kernel(
    uint32_t packed_spm_addr, uint32_t output_spm_addr,
    int64_t rows, int num_cores) {
  TORCH_CHECK(rows == 51 && num_cores == 8,
              "Lingbot2 shared SiLU requires M51/D96/TP8");
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Lingbot2 shared SiLU requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  constexpr uint64_t packed_bytes = 51 * 192 * 2;
  constexpr uint64_t output_bytes = 51 * 96 * 2;
  const auto valid = [base](uint32_t address, uint64_t bytes) {
    return address >= base && address % 32 == 0 &&
        bytes <= SpmAllocator::SPM_USABLE &&
        uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes;
  };
  TORCH_CHECK(valid(packed_spm_addr, packed_bytes) &&
                  valid(output_spm_addr, output_bytes),
              "Lingbot2 shared SiLU requires complete 32-byte-aligned absolute SPM intervals");
  TORCH_CHECK(uint64_t(packed_spm_addr) + packed_bytes <= output_spm_addr ||
                  uint64_t(output_spm_addr) + output_bytes <= packed_spm_addr,
              "Lingbot2 shared SiLU input and output must be disjoint");
  auto* kernel = GET_KERNEL(KernelId::QWEN3VL_SILU_MUL_EXACT_STRIDED);
  TORCH_CHECK(kernel != nullptr,
              "Lingbot2 shared SiLU required exact-strided payload is absent");
  kernel->reset_regs();
  kernel->set_regs(0, static_cast<uint16_t>(96));
  kernel->set_regs(1, static_cast<uint16_t>(96));
  const auto address = [kernel](int reg, uint32_t value) {
    kernel->set_regs(reg, static_cast<uint16_t>(value));
    kernel->set_regs(reg + 1, static_cast<uint16_t>(value >> 16));
  };
  address(4, packed_spm_addr);
  address(6, packed_spm_addr + 192);
  address(8, output_spm_addr);
  auto* queue = GET_QUEUE(num_cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {51, 1, 1}, {0,1,2,3,4,5,6,7});
}

// Same custom SiLU arithmetic, flattened 16-half vectors over twenty warps.
void rpu_launch_lingbot2_shared_silu_t51_d96_vector_spm_kernel(
    uint32_t packed_spm_addr, uint32_t output_spm_addr,
    int64_t rows, int num_cores) {
  TORCH_CHECK(rows == 51 && num_cores == 8,
              "Lingbot2 shared SiLU requires M51/D96/TP8");
  TORCH_CHECK(SPM_ALLOC.is_initialized(),
              "Lingbot2 shared SiLU requires initialized SPM");
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  constexpr uint64_t packed_bytes = 51 * 192 * 2;
  constexpr uint64_t output_bytes = 51 * 96 * 2;
  const auto valid = [base](uint32_t address, uint64_t bytes) {
    return address >= base && address % 32 == 0 &&
        bytes <= SpmAllocator::SPM_USABLE &&
        uint64_t(address) - base <= SpmAllocator::SPM_USABLE - bytes &&
        uint64_t(address) + bytes <= (uint64_t{1} << 32);
  };
  TORCH_CHECK(valid(packed_spm_addr, packed_bytes) &&
                  valid(output_spm_addr, output_bytes),
              "Lingbot2 shared SiLU requires complete 32-byte-aligned absolute SPM intervals");
  TORCH_CHECK(uint64_t(packed_spm_addr) + packed_bytes <= output_spm_addr ||
                  uint64_t(output_spm_addr) + output_bytes <= packed_spm_addr,
              "Lingbot2 shared SiLU input and output must be disjoint");
  auto* queue = GET_QUEUE(num_cores);
  TORCH_CHECK(queue != nullptr, "Lingbot2 shared SiLU eight-core queue is absent");
  auto* kernel = GET_KERNEL(KernelId::LINGBOT2_SHARED_SILU_T51_D96_VECTOR);
  TORCH_CHECK(kernel != nullptr,
              "Lingbot2 shared SiLU optional compact-vector payload is absent");
  kernel->reset_regs();
  kernel->set_regs(0, static_cast<uint16_t>(96));
  kernel->set_regs(1, static_cast<uint16_t>(96));
  const auto address = [kernel](int reg, uint32_t value) {
    kernel->set_regs(reg, static_cast<uint16_t>(value));
    kernel->set_regs(reg + 1, static_cast<uint16_t>(value >> 16));
  };
  address(4, packed_spm_addr);
  address(6, packed_spm_addr + 192);
  address(8, output_spm_addr);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {20, 1, 1}, {0,1,2,3,4,5,6,7});
}

// Fixed sparse TP8 profile. Preserve FP16 SiLU*Up retirement before applying
// the routed row weight. The caller owns valid raw U16 token/expert IDs.
void rpu_launch_lingbot2_swiglu_route_r204_d64_spm_kernel(
    uint32_t packed_spm_addr, uint32_t routes_spm_addr,
    uint32_t token_ids_spm_addr, uint32_t expert_ids_spm_addr,
    uint32_t output_spm_addr) {
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "lingbot2_swiglu_route: uninitialized SPM");
  const uint32_t base = SPM_ALLOC.addr(0, 0);
  const std::array<std::pair<uint32_t, uint32_t>, 5> ranges{{
      {packed_spm_addr, 204 * 128 * 2}, {routes_spm_addr, 64 * 32 * 2},
      {token_ids_spm_addr, 256 * 2}, {expert_ids_spm_addr, 256 * 2},
      {output_spm_addr, 204 * 64 * 2}}};
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto [address, bytes] = ranges[i];
    TORCH_CHECK(address >= base && address % 32 == 0 &&
                    uint64_t(address) - base <= uint64_t(SpmAllocator::SPM_USABLE) - bytes,
                "lingbot2_swiglu_route: invalid aligned absolute SPM interval");
    for (size_t j = 0; j < i; ++j)
      TORCH_CHECK(uint64_t(address) + bytes <= ranges[j].first ||
                      uint64_t(ranges[j].first) + ranges[j].second <= address,
                  "lingbot2_swiglu_route: SPM operands must be disjoint");
  }
  auto* kernel = GET_KERNEL(KernelId::LINGBOT2_SWIGLU_ROUTE_R204_D64);
  TORCH_CHECK(kernel != nullptr, "lingbot2_swiglu_route: optional payload missing");
  kernel->reset_regs();
  auto address = [kernel](int index, uint32_t value) {
    kernel->set_regs(index, static_cast<uint16_t>(value));
    kernel->set_regs(index + 1, static_cast<uint16_t>(value >> 16));
  };
  kernel->set_regs(0, static_cast<uint16_t>(64));
  kernel->set_regs(1, static_cast<uint16_t>(64));
  address(4, packed_spm_addr);
  address(6, packed_spm_addr + 128);
  address(8, output_spm_addr);
  address(10, token_ids_spm_addr);
  address(12, expert_ids_spm_addr);
  address(14, routes_spm_addr);
  kernel->set_regs(64, static_cast<uint16_t>(51));
  kernel->set_regs(65, static_cast<uint16_t>(1));
  kernel->set_regs(66, static_cast<uint16_t>(1));
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {51, 1, 1}, {0,1,2,3,4,5,6,7});
}

// SwiGLU activation fusion: out = silu(x) * y, flat per-core elementwise.
// Register layout:
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
  // Block counts must fit signed 16-bit and remain 256-element aligned.
  size_t normal_blk_n = 256 * 127;
  size_t blk_cnt = CeilDiv(n, normal_blk_n);
  size_t last_blk_n = n - (blk_cnt - 1) * normal_blk_n;

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


void rpu_require_high_precision_fusion_kernels() {
  const auto& cache = KernelCache::instance();
  for (const auto id : {KernelId::RHINOVLA_NEWTON_NORM_SHIFT_H1024,
                        KernelId::RHINOVLA_SILU_HIGH_MUL}) {
    TORCH_CHECK(cache.has_loaded(id), "High-precision fusion requires loaded payload ",
                KERNEL_ID_NAMES[static_cast<size_t>(id)]);
  }
}

void rpu_launch_rhinovla_silu_high_mul_spm_kernel(
    uint32_t gate, uint32_t up, uint32_t output, int64_t elements, int cores) {
  constexpr int64_t block = 15872;
  TORCH_CHECK(cores >= 1 && cores <= 8 && elements > 0 &&
                  elements <= block * int64_t(UINT16_MAX),
              "RhinoVLA HIGH SiLU/Mul requires 1..8 cores and representable grid");
  TORCH_CHECK(SPM_ALLOC.is_initialized(), "RhinoVLA HIGH SiLU/Mul requires live SPM");
  const uint64_t bytes = uint64_t(elements) * sizeof(c10::Half);
  const uint64_t base = SPM_ALLOC.addr(0, 0);
  auto valid = [=](uint32_t address) {
    return address % 256 == 0 && address >= base &&
        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE;
  };
  auto disjoint = [=](uint32_t a, uint32_t b) {
    return uint64_t(a) + bytes <= b || uint64_t(b) + bytes <= a;
  };
  TORCH_CHECK(valid(gate) && valid(up) && valid(output),
              "RhinoVLA HIGH SiLU/Mul requires complete aligned SPM operands");
  TORCH_CHECK((output == gate || disjoint(output, gate)) &&
                  (output == up || disjoint(output, up)),
              "RhinoVLA HIGH SiLU/Mul rejects partial output alias");
  auto* kernel = GET_KERNEL(KernelId::RHINOVLA_SILU_HIGH_MUL);
  TORCH_CHECK(kernel, "RhinoVLA HIGH SiLU/Mul kernel is unavailable");
  kernel->reset_regs();
  const int64_t blocks = (elements + block - 1) / block;
  kernel->set_regs(0, uint16_t(block));
  kernel->set_regs(1, uint16_t(elements - (blocks - 1) * block));
  for (auto entry : {std::pair<uint32_t,uint32_t>{4,gate}, {6,output},
                    {8,uint32_t(block * sizeof(c10::Half))}, {12,up}}) {
    kernel->set_regs(entry.first, uint16_t(entry.second));
    kernel->set_regs(entry.first + 1, uint16_t(entry.second >> 16));
  }
  for (int reg : {56,57,58}) kernel->set_regs(reg, c10::Half(1.0).x);
  kernel->set_regs(61, uint16_t(1));
  const auto& op_info = get_valu_op_info(ValuOpType::SILU);
  kernel->set_regs(62, uint16_t((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
  kernel->set_regs(63, uint16_t(op_info.custom_op));
  auto* queue = GET_QUEUE(cores);
  queue->set_broadcast_mode(true);
  std::vector<uint8_t> ids;
  for (int core = 0; core < cores; ++core) ids.push_back(core);
  queue->enqueu_kernel(*kernel, {uint16_t(blocks),1,1}, ids);
}
