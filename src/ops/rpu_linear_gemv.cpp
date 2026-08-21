#include "rhino_launch_buffer.h"

#include "rhino_launch_queue.h"
#include "rpu_ops.h"

#include <algorithm>
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
  // llama_gemv uses `(2 * buf_depth) - 1` as a ring-buffer mask, so depth
  // must be a power of two. It also cannot exceed the number of K vectors
  // available to one core. Prefer the deepest legal pipeline.
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

}  // namespace

// Direct-address SPM GEMV launcher used by non-census M=1 production Linear
// dispatch and the explicit conformance op.
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

  if (partition == 1) {
    TORCH_CHECK(n % num_cores == 0,
                "Col partition: N must be divisible by num_cores (",
                num_cores, ")");
  } else {
    TORCH_CHECK(k % num_cores == 0,
                "Row partition: K must be divisible by num_cores (",
                num_cores, ")");
  }

  size_t local_k;
  size_t local_n;
  if (partition == 0) {
    local_k = k / num_cores;
    local_n = n;
  } else {
    local_k = k;
    local_n = n / num_cores;
  }
  TORCH_CHECK(local_k <= UINT16_MAX && local_n <= UINT16_MAX,
              "GEMV local dimensions exceed 16-bit kernel registers: local_k=",
              local_k, ", local_n=", local_n);

  const uint32_t x_addr = input_spm_addr;
  const uint32_t y_addr = output_spm_addr;
  TORCH_CHECK(local_n % 16 == 0,
              "GEMV SPM kernel requires local_n % 16 == 0");
  TORCH_CHECK(x_addr % 256 == 0,
              "GEMV SPM kernel requires input SPM address 256-byte aligned");
  TORCH_CHECK(y_addr % 256 == 0,
              "GEMV SPM kernel requires output SPM address 256-byte aligned");

  const bool register_census_active =
      RpuKernelGraph::active().kernel_register_census_active();
  if (register_census_active) {
    TORCH_CHECK(weight.scalar_type() == at::kHalf,
                "typed DDR-register census admits FP16 GEMV weights only");
  }

  if (weight.scalar_type() == at::kChar) {
    TORCH_CHECK(scale.defined() && scale.scalar_type() == at::kHalf,
                "GEMV wint8 requires a defined fp16 scale tensor");
    TORCH_CHECK(scale.numel() == n,
                "GEMV wint8: scale.numel()=", scale.numel(), " != N=", n);
    TORCH_CHECK(local_k % 32 == 0,
                "GEMV wint8: local_k=", local_k,
                " must be divisible by 32");

    int8_t *wp = weight.data_ptr<int8_t>();
    c10::Half *sp = scale.data_ptr<c10::Half>();
    rpu_ddr_flush(wp);
    rpu_ddr_flush(sp);
    const uint64_t w_addr = RpuGetDevAddr(wp);
    const uint64_t s_addr = RpuGetDevAddr(sp);
    TORCH_CHECK(w_addr % 256 == 0,
                "GEMV wint8: weight DDR address not 256-byte aligned");
    TORCH_CHECK(s_addr % 256 == 0,
                "GEMV wint8: scale DDR address not 256-byte aligned");

    const uint64_t x_addr_g = static_cast<uint64_t>(x_addr);
    const uint64_t y_addr_g = static_cast<uint64_t>(y_addr);

    int blk_cnt = std::min(8, static_cast<int>(local_n / 16));
    while (blk_cnt > 1 &&
           (static_cast<int>(local_n) % (blk_cnt * 16)) != 0) {
      --blk_cnt;
    }
    TORCH_CHECK(static_cast<int>(local_n) % (blk_cnt * 16) == 0,
                "GEMV wint8: cannot tile local_n=", local_n,
                " with <=8 blocks of 16-aligned output rows");
    const int normal_blk_n = static_cast<int>(local_n) / blk_cnt;
    const int last_blk_n =
        static_cast<int>(local_n) - (blk_cnt - 1) * normal_blk_n;
    TORCH_CHECK(last_blk_n > 0 && last_blk_n <= normal_blk_n &&
                    last_blk_n % 16 == 0,
                "GEMV wint8: invalid block partition local_n=", local_n,
                " blk_cnt=", blk_cnt, " normal_blk_n=", normal_blk_n,
                " last_blk_n=", last_blk_n);
    TORCH_CHECK((blk_cnt - 1) * normal_blk_n + last_blk_n ==
                    static_cast<int>(local_n),
                "GEMV wint8: block partition does not tile local_n=", local_n);
    TORCH_CHECK(normal_blk_n <= 4096,
                "GEMV wint8: normal_blk_n=", normal_blk_n, " exceeds 4096");
    TORCH_CHECK(static_cast<uint64_t>(num_cores) * normal_blk_n * local_k %
                        256 ==
                    0,
                "GEMV wint8: weight block stride is not an integer 256-byte unit");
    const uint32_t weight_blk_stride = static_cast<uint32_t>(
        static_cast<uint64_t>(num_cores) * normal_blk_n * local_k / 256);

    constexpr uint32_t input_core_stride = 0x10000;
    constexpr uint32_t output_core_stride = 0x10000;
    auto setup_wint8 = [=](Kernel_t *kernel) {
      kernel->set_regs(0, static_cast<uint16_t>(local_k));
      kernel->set_regs(1, static_cast<uint16_t>(local_n));
      kernel->set_regs(2, static_cast<uint16_t>(partition == 1 ? 1 : 0));

      kernel->set_regs(4, static_cast<uint16_t>((x_addr_g >> 8) & 0xFFFF));
      kernel->set_regs(5, static_cast<uint16_t>((x_addr_g >> 24) & 0xFFFF));
      kernel->set_regs(6, static_cast<uint16_t>((w_addr >> 8) & 0xFFFF));
      kernel->set_regs(7, static_cast<uint16_t>((w_addr >> 24) & 0xFFFF));
      kernel->set_regs(8, static_cast<uint16_t>((s_addr >> 8) & 0xFFFF));
      kernel->set_regs(9, static_cast<uint16_t>((s_addr >> 24) & 0xFFFF));
      kernel->set_regs(10, static_cast<uint16_t>((y_addr_g >> 8) & 0xFFFF));
      kernel->set_regs(11, static_cast<uint16_t>((y_addr_g >> 24) & 0xFFFF));

      kernel->set_regs(12, static_cast<uint16_t>(input_core_stride & 0xFFFF));
      kernel->set_regs(13,
                       static_cast<uint16_t>((input_core_stride >> 16) & 0xFFFF));
      kernel->set_regs(14, static_cast<uint16_t>(output_core_stride & 0xFFFF));
      kernel->set_regs(
          15, static_cast<uint16_t>((output_core_stride >> 16) & 0xFFFF));
      kernel->set_regs(16, static_cast<uint16_t>(weight_blk_stride & 0xFFFF));
      kernel->set_regs(
          17, static_cast<uint16_t>((weight_blk_stride >> 16) & 0xFFFF));

      kernel->set_regs(18, static_cast<uint16_t>(normal_blk_n));
      kernel->set_regs(19, static_cast<uint16_t>(last_blk_n));
      kernel->set_regs(21, static_cast<uint16_t>(num_cores));
      kernel->set_regs(64, static_cast<uint16_t>(blk_cnt));
      kernel->set_regs(65, static_cast<uint16_t>(1));
      kernel->set_regs(66, static_cast<uint16_t>(1));
    };

    Kernel_t *kernel = GET_KERNEL(KernelId::LLAMA_GEMV_WINT8);
    TORCH_CHECK(kernel != nullptr, "Failed to get llama_gemv_wint8 kernel");
    kernel->reset_regs();
    setup_wint8(kernel);

    auto *wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    std::vector<uint8_t> core_list;
    for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
    wq->enqueu_kernel(*kernel, {static_cast<uint16_t>(blk_cnt), 1, 1},
                       core_list);
    return;
  }

  TORCH_CHECK(weight.scalar_type() == at::kHalf,
              "GEMV supports fp16 or int8 weights, got ",
              weight.scalar_type());
  const uint32_t buf_depth = select_fp16_gemv_buf_depth(local_k, local_n);
  TORCH_CHECK(
      buf_depth != 0,
      "FP16 GEMV requires positive 16-aligned local_k/local_n and a legal "
      "power-of-two buffer depth, got local_k=",
      local_k, ", local_n=", local_n);
  const uint64_t work_per_loop =
      static_cast<uint64_t>(buf_depth) * kFp16GemvWorkPerDepth;
  const uint64_t lp_cnt_u64 =
      static_cast<uint64_t>(local_k) * local_n / work_per_loop + 1;
  TORCH_CHECK(lp_cnt_u64 <= UINT16_MAX,
              "FP16 GEMV lp_cnt exceeds the 16-bit kernel register: ",
              lp_cnt_u64);
  const uint32_t lp_cnt = static_cast<uint32_t>(lp_cnt_u64);

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
  c10::Half *wp = weight.data_ptr<c10::Half>();
  rpu_ddr_flush(wp);

  const uint64_t x_addr_g = static_cast<uint64_t>(x_addr);
  const uint64_t y_addr_g = static_cast<uint64_t>(y_addr);
  constexpr uint32_t input_core_stride = 0x10000;
  constexpr uint32_t output_core_stride = 0x10000;

  auto setup_regs = [=](Kernel_t *kernel) {
    kernel->set_regs(0, static_cast<uint16_t>(local_k));
    kernel->set_regs(1, static_cast<uint16_t>(local_n));
    kernel->set_regs(2, static_cast<uint16_t>(lp_cnt));
    kernel->set_regs(3, static_cast<uint16_t>(buf_depth));

    kernel->set_regs(4, static_cast<uint16_t>((x_addr_g >> 8) & 0xFFFF));
    kernel->set_regs(5, static_cast<uint16_t>((x_addr_g >> 24) & 0xFFFF));
    kernel->set_regs(8, static_cast<uint16_t>((y_addr_g >> 8) & 0xFFFF));
    kernel->set_regs(9, static_cast<uint16_t>((y_addr_g >> 24) & 0xFFFF));

    kernel->set_regs(10, static_cast<uint16_t>(input_core_stride & 0xFFFF));
    kernel->set_regs(11,
                     static_cast<uint16_t>((input_core_stride >> 16) & 0xFFFF));
    kernel->set_regs(12, static_cast<uint16_t>(output_core_stride & 0xFFFF));
    kernel->set_regs(
        13, static_cast<uint16_t>((output_core_stride >> 16) & 0xFFFF));
    kernel->set_regs(21, static_cast<uint16_t>(num_cores));
    kernel->set_regs(64, static_cast<uint16_t>(1));
    kernel->set_regs(65, static_cast<uint16_t>(1));
    kernel->set_regs(66, static_cast<uint16_t>(1));
  };

  Kernel_t *kernel = GET_KERNEL(KernelId::LLAMA_GEMV);
  TORCH_CHECK(kernel != nullptr, "Failed to get llama_gemv kernel");
  kernel->reset_regs();
  setup_regs(kernel);
  register_writer.write(*kernel);

  auto *wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
  wq->enqueu_kernel(*kernel, {1, 1, 1}, core_list);
}
