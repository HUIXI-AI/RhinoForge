// rpu_eltwise_binary.cpp - Elementwise binary operations for RPU backend
#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h" // for Program_t
#include "rhino_launch_queue.h"


#include "rpu_ops.h"
#include "rpu_spm_allocator.h" // for SPM_ALLOC (bx1xc isolated test)
#include <c10/util/Half.h>
#include <stdlib.h> // for uint16_t, setenv, unsetenv
#include <string.h> // for memcpy
#include <string>   // for string
#include <vector>   // for vector

using namespace ::rhino_lkn;
using namespace at;

// ======================== Common Binary Op Wrapper ========================

// 统一的二元操作实现
// out_opt: nullptr = 分配新 tensor, 非 nullptr = inplace 模式使用该 tensor 作为输出
// is_commutative: 仅非 inplace 模式需要，用于处理 Nx1 op NxC 等情况
// Dispatch overhead profiling
static thread_local int dispatch_profile_count = 0;
static thread_local double dispatch_total_entry_to_kernel = 0;

// Can the DDR broadcast kernel actually address this (N, C)? Defined next to
// the launchers, from the same geometry helper they use. False means the fast
// path would compute the WRONG rows, so binary_op_impl must demote the pattern
// and let the sameshape kernel materialise the broadcast instead.
static bool eltwise_1xC_NxC_ddr_launchable(int64_t n, int64_t c);
static bool eltwise_Nx1_NxC_ddr_launchable(int64_t n, int64_t c);

static at::Tensor binary_op_impl(
    const at::Tensor &a,
    const at::Tensor &b,
    const c10::Scalar &alpha,
    ValuOpType op_type,
    bool is_commutative,
    at::Tensor* out_opt) {

#if RPU_PROFILE_ENABLED
  auto entry_time = std::chrono::high_resolution_clock::now();
#endif

  const bool inplace = (out_opt != nullptr);

  TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1,
              "binary_op: a not on RPU");

  // Out-of-place execution follows the promoted dtype; in-place execution
  // follows a's dtype. The device launcher additionally requires fp16 a, so
  // integer and boolean inputs use the CPU fallback. Broadcast fallback must
  // allocate the promoted output shape rather than empty_like(a).
  const auto gate_dtype = inplace ? a.scalar_type() : at::result_type(a, b);
  if (gate_dtype != at::kHalf || a.scalar_type() != at::kHalf) {
    auto a_cpu = rpu_to_cpu_zerocopy(a);
    auto b_cpu = (b.device().type() == c10::DeviceType::PrivateUse1)
                 ? rpu_to_cpu_zerocopy(b) : b;

    if (inplace) {
      switch (op_type) {
        case ValuOpType::ADD: a_cpu.add_(b_cpu, alpha); break;
        case ValuOpType::MUL: a_cpu.mul_(b_cpu); break;
        case ValuOpType::SUB: a_cpu.sub_(b_cpu, alpha); break;
        case ValuOpType::DIV: a_cpu.div_(b_cpu); break;
        default: TORCH_CHECK(false, "Unsupported op type");
      }
      return *out_opt;
    } else {
      // Use non-out variants to let CPU handle broadcast shape inference
      at::Tensor cpu_result;
      switch (op_type) {
        case ValuOpType::ADD: cpu_result = at::add(a_cpu, b_cpu, alpha); break;
        case ValuOpType::MUL: cpu_result = at::mul(a_cpu, b_cpu); break;
        case ValuOpType::SUB: cpu_result = at::sub(a_cpu, b_cpu, alpha); break;
        case ValuOpType::DIV: cpu_result = at::div(a_cpu, b_cpu); break;
        default: TORCH_CHECK(false, "Unsupported op type");
      }
      return cpu_result.to(a.device());
    }
  }

  // 2. Scalar tensor 优化: 直接提取值避免 _to_copy 开销
  if (b.dim() == 0 || b.numel() == 1) {
    auto a_contig = a.contiguous();
    at::Tensor result = inplace ? *out_opt
                                : rpu_empty_strided(a_contig.sizes(),
                                                    compute_contiguous_strides(a_contig.sizes()),
                                                    a_contig.scalar_type(),
                                                    a_contig.layout(), a_contig.device(), false);
    const c10::Half scalar_val = c10::Half(static_cast<float>(b.item<double>()));
    const c10::Half alpha_f = (op_type == ValuOpType::ADD || op_type == ValuOpType::SUB)
                              ? alpha.to<c10::Half>() : c10::Half(1.0f);
    rpu_launch_eltwise_binary_scalar_kernel(a_contig, scalar_val, alpha_f, result, op_type);
    return result;
  }

  // 3. 非 scalar: 处理设备和 dtype 转换
  Tensor b_rpu = b;
  if (b.device() != a.device()) {
    b_rpu = b.to(a.device());
  }
  if (b_rpu.scalar_type() != a.scalar_type()) {
    b_rpu = b_rpu.to(a.scalar_type());
  }

  // 4. 确保 contiguous (避免 reshape 时的隐式拷贝)
#if RPU_PROFILE_ENABLED
  auto contig_start = std::chrono::high_resolution_clock::now();
#endif

  Tensor a_contig = a.is_contiguous() ? a : a.contiguous();
  Tensor b_contig = b_rpu.is_contiguous() ? b_rpu : b_rpu.contiguous();

#if RPU_PROFILE_ENABLED
  double early_contig_ms = PROFILE_ELAPSED_MS(contig_start);
  static thread_local int contig_copy_count = 0;
  static thread_local double total_contig_ms = 0;
  if (!a.is_contiguous() || !b_rpu.is_contiguous()) {
    contig_copy_count++;
    total_contig_ms += early_contig_ms;
    if (contig_copy_count <= 3 || contig_copy_count % 100 == 0) {
      std::cout << "[EARLY_CONTIG #" << contig_copy_count << "] "
                << "a_contig=" << a.is_contiguous() << " b_contig=" << b_rpu.is_contiguous()
                << " time=" << early_contig_ms << "ms avg=" << total_contig_ms/contig_copy_count << "ms"
                << std::endl;
    }
  }
#endif

  const c10::Half alpha_f = (op_type == ValuOpType::ADD || op_type == ValuOpType::SUB)
                            ? alpha.to<c10::Half>() : c10::Half(1.0f);

  // 5. Broadcast pattern matching (分两阶段：先检测 pattern，再执行 kernel)
  // 使用 a_contig 和 b_contig 替代 a 和 b_rpu
  // 支持的 pattern:
  //   - PATTERN_1xC_NxC: a 的非最后维全为 1，最后维相同
  //   - PATTERN_NxC_1xC: b 的非最后维全为 1，最后维相同
  //   - PATTERN_Nx1_NxC: 非最后维相同，a 的最后维为 1
  //   - PATTERN_NxC_Nx1: 非最后维相同，b 的最后维为 1
  enum class BroadcastPattern {
    NONE,           // 无法匹配，fallback 到 sameshape
    SAMESHAPE,      // 形状完全相同
    PATTERN_1xC_NxC,
    PATTERN_NxC_1xC,
    PATTERN_Nx1_NxC,
    PATTERN_NxC_Nx1
  };

  BroadcastPattern pattern = BroadcastPattern::NONE;
  int64_t N_a = 0, N_b = 0, C = 0;
  std::vector<int64_t> out_sizes;

#if RPU_PROFILE_ENABLED
  auto phase1_start = std::chrono::high_resolution_clock::now();
  double entry_to_phase1_ms = std::chrono::duration<double, std::milli>(phase1_start - entry_time).count();
#endif

  // ========== Phase 1: Pattern Matching ==========
  // 使用 a_contig 和 b_contig 进行 pattern matching
  {
    // 先检查是否是 sameshape (不需要 broadcast)
    if (a_contig.sizes() == b_contig.sizes()) {
      pattern = BroadcastPattern::SAMESHAPE;
      // sameshape 直接 fallthrough 到后面的 sameshape kernel
    }

    int a_dim = a_contig.dim();
    int b_dim = b_contig.dim();
    int max_dim = std::max(a_dim, b_dim);

    if (pattern == BroadcastPattern::NONE && max_dim >= 1) {
      // 构建对齐后的 sizes（短的在前面补 1）
      std::vector<int64_t> a_aligned(max_dim), b_aligned(max_dim);
      for (int i = 0; i < max_dim; i++) {
        a_aligned[i] = (i < max_dim - a_dim) ? 1 : a_contig.size(i - (max_dim - a_dim));
        b_aligned[i] = (i < max_dim - b_dim) ? 1 : b_contig.size(i - (max_dim - b_dim));
      }

      int64_t a_last = a_aligned[max_dim - 1];
      int64_t b_last = b_aligned[max_dim - 1];

      // 检查 a 的非最后维是否全为 1
      bool a_prefix_all_one = true;
      for (int i = 0; i < max_dim - 1; i++) {
        if (a_aligned[i] != 1) { a_prefix_all_one = false; break; }
      }

      // 检查 b 的非最后维是否全为 1
      bool b_prefix_all_one = true;
      for (int i = 0; i < max_dim - 1; i++) {
        if (b_aligned[i] != 1) { b_prefix_all_one = false; break; }
      }

      // 检查非最后维是否都相同
      bool prefix_same = true;
      for (int i = 0; i < max_dim - 1; i++) {
        if (a_aligned[i] != b_aligned[i]) { prefix_same = false; break; }
      }

      // 计算 N (非最后维的乘积)
      auto compute_N = [&](const std::vector<int64_t>& aligned) {
        int64_t N = 1;
        for (int i = 0; i < max_dim - 1; i++) N *= aligned[i];
        return N;
      };

      // 计算输出 shape
      auto compute_out_sizes = [&]() {
        std::vector<int64_t> sizes(max_dim);
        for (int i = 0; i < max_dim; i++) {
          sizes[i] = std::max(a_aligned[i], b_aligned[i]);
        }
        return sizes;
      };

      // Case 1: 1xC op NxC
      if (a_prefix_all_one && a_last == b_last && !b_prefix_all_one) {
        pattern = BroadcastPattern::PATTERN_1xC_NxC;
        N_a = 1;
        N_b = compute_N(b_aligned);
        C = a_last;
        out_sizes = compute_out_sizes();
      }
      // Case 2: NxC op 1xC
      else if (b_prefix_all_one && a_last == b_last && !a_prefix_all_one) {
        pattern = BroadcastPattern::PATTERN_NxC_1xC;
        N_a = compute_N(a_aligned);
        N_b = 1;
        C = a_last;
        out_sizes = compute_out_sizes();
      }
      // Case 3: Nx1 op NxC
      else if (prefix_same && a_last == 1 && b_last > 1) {
        pattern = BroadcastPattern::PATTERN_Nx1_NxC;
        N_a = compute_N(a_aligned);
        N_b = N_a;
        C = b_last;
        out_sizes = compute_out_sizes();
      }
      // Case 4: NxC op Nx1
      else if (prefix_same && b_last == 1 && a_last > 1) {
        pattern = BroadcastPattern::PATTERN_NxC_Nx1;
        N_a = compute_N(a_aligned);
        N_b = N_a;
        C = a_last;
        out_sizes = compute_out_sizes();
      }
      // Case 5: 单个 broadcast 维度在非最后维
      else {
        int broadcast_dim = -1;
        int broadcast_count = 0;
        for (int i = 0; i < max_dim - 1; i++) {
          if (a_aligned[i] != b_aligned[i]) {
            if (a_aligned[i] != 1 && b_aligned[i] != 1) {
              broadcast_dim = -1;
              break;
            }
            broadcast_dim = i;
            broadcast_count++;
          }
        }

        if (broadcast_count == 1 && broadcast_dim != -1 && a_last == b_last) {
          N_a = 1; N_b = 1; C = 1;
          for (int i = 0; i <= broadcast_dim; i++) {
            N_a *= a_aligned[i];
            N_b *= b_aligned[i];
          }
          for (int i = broadcast_dim + 1; i < max_dim; i++) {
            C *= a_aligned[i];
          }
          out_sizes = compute_out_sizes();

          if (N_a == 1 && N_b > 1) {
            pattern = BroadcastPattern::PATTERN_1xC_NxC;
          } else if (N_a > 1 && N_b == 1) {
            pattern = BroadcastPattern::PATTERN_NxC_1xC;
          }
        }
      }
    }

    // A matched pattern is not automatically a LAUNCHABLE one: the DDR batch
    // kernels cannot address every (N, C) — see the geometry helpers next to
    // the launchers. Demote what they cannot express back to NONE so it falls
    // through to the sameshape kernel below, which materialises the broadcast
    // and is correct for any shape. Silently computing the wrong rows is worse
    // than one extra copy on a path no fused model takes.
    if (pattern != BroadcastPattern::NONE && pattern != BroadcastPattern::SAMESHAPE) {
      const int64_t kern_n = std::max(N_a, N_b);
      const bool launchable =
          (pattern == BroadcastPattern::PATTERN_1xC_NxC ||
           pattern == BroadcastPattern::PATTERN_NxC_1xC)
              ? eltwise_1xC_NxC_ddr_launchable(kern_n, C)
              : eltwise_Nx1_NxC_ddr_launchable(kern_n, C);
      if (!launchable) pattern = BroadcastPattern::NONE;
    }
  }

  // ========== Phase 2: Kernel Dispatch ==========
  // 只有 broadcast pattern 才进入这里，SAMESHAPE 和 NONE 都 fallthrough 到 sameshape kernel
  if (pattern != BroadcastPattern::NONE && pattern != BroadcastPattern::SAMESHAPE) {
#if RPU_PROFILE_ENABLED
    double phase1_ms = PROFILE_ELAPSED_MS(phase1_start);
    auto alloc_start = std::chrono::high_resolution_clock::now();
#endif

    int64_t out_N = std::max(N_a, N_b);

    // 根据 pattern 确定 kernel 参数
    // small_2d: 小的那个 tensor (1xC 或 Nx1)
    // large_2d: 大的那个 tensor (NxC)
    // is_bopa: 是否需要交换操作数顺序 (对于非交换操作如 sub, div)
    at::Tensor small_2d, large_2d;
    bool use_1xC_kernel = false;  // true: 1xC_NxC kernel, false: Nx1_NxC kernel
    bool is_bopa = false;
    const bool alpha_is_one = (alpha.to<double>() == 1.0);

    // 使用 a_contig 和 b_contig - 已经是 contiguous，view 是零拷贝
    switch (pattern) {
      case BroadcastPattern::PATTERN_1xC_NxC:
        small_2d = a_contig.view({1, C});
        large_2d = b_contig.view({N_b, C});
        use_1xC_kernel = true;
        is_bopa = false;  // a op b = small op large
        TORCH_CHECK(!inplace, "inplace 1xC op NxC not supported");
        break;

      case BroadcastPattern::PATTERN_NxC_1xC:
        small_2d = b_contig.view({1, C});
        large_2d = a_contig.view({N_a, C});
        use_1xC_kernel = true;
        // `scale_b` binds alpha to the kernel's second operand, so swapping is
        // valid only when alpha is 1: a + 2.5*b is not b + 2.5*a.
        is_bopa = !(is_commutative && alpha_is_one);  // a op b = large op small
        break;

      case BroadcastPattern::PATTERN_Nx1_NxC:
        small_2d = a_contig.view({N_a, 1});
        large_2d = b_contig.view({N_b, C});
        use_1xC_kernel = false;
        is_bopa = false;  // a op b = small op large
        TORCH_CHECK(!inplace, "inplace Nx1 op NxC not supported");
        break;

      case BroadcastPattern::PATTERN_NxC_Nx1:
        small_2d = b_contig.view({N_b, 1});
        large_2d = a_contig.view({N_a, C});
        use_1xC_kernel = false;
        // `scale_b` binds alpha to the kernel's second operand, so swapping is
        // valid only when alpha is 1: a + 2.5*b is not b + 2.5*a.
        is_bopa = !(is_commutative && alpha_is_one);  // a op b = large op small
        break;

      default:
        break;
    }

#if RPU_PROFILE_ENABLED
    double reshape_ms = PROFILE_ELAPSED_MS(alloc_start);
    auto real_alloc_start = std::chrono::high_resolution_clock::now();
#endif

    // 分配输出 tensor
    at::Tensor result = inplace ? *out_opt
                                : rpu_empty_strided(out_sizes, compute_contiguous_strides(out_sizes),
                                                    a.scalar_type(), a.layout(), a.device(), false);
    // at::Tensor result_2d = result.reshape({out_N, C});
    at::Tensor result_2d = result;

#if RPU_PROFILE_ENABLED
    double alloc_ms = PROFILE_ELAPSED_MS(real_alloc_start);
    auto kernel_start = std::chrono::high_resolution_clock::now();
#endif

    // 调用对应的 kernel
    if (use_1xC_kernel) {
      rpu_launch_eltwise_binary_1xC_NxC_kernel(small_2d, large_2d, alpha_f, result_2d, op_type, is_bopa);
    } else {
      rpu_launch_eltwise_binary_Nx1_NxC_kernel(small_2d, large_2d, alpha_f, result_2d, op_type, is_bopa);
    }

#if RPU_PROFILE_ENABLED
    double kernel_ms = PROFILE_ELAPSED_MS(kernel_start);
    double total_ms = phase1_ms + alloc_ms + kernel_ms;

    // Build WrapperProfileData
    WrapperProfileData wp;
    wp.preprocess_ms = entry_to_phase1_ms + early_contig_ms + phase1_ms + reshape_ms;
    wp.alloc_ms = alloc_ms;
    wp.contiguous_ms = 0;  // contiguous 已移到 preprocess 阶段
    wp.kernel_ms = kernel_ms;
    wp.total_ms = early_contig_ms + phase1_ms + reshape_ms + alloc_ms + kernel_ms;

    // Get kernel profile data
    const KernelProfileData& kp = g_last_kernel_profile;

    // Use standard profile macro (use 1xC or Nx1 accumulator based on pattern)
    if (use_1xC_kernel) {
      PROFILE_RECORD_AND_PRINT("binary_1xC_NxC", g_profile_binary_1xC_NxC, wp, kp);
    } else {
      PROFILE_RECORD_AND_PRINT("binary_Nx1_NxC", g_profile_binary_Nx1_NxC, wp, kp);
    }
#endif

    return inplace ? *out_opt : result;
  }

  // 6. Sameshape kernel (用于 SAMESHAPE pattern 或无法匹配的 broadcast)
  // 使用已经 contiguous 的 a_contig 和 b_contig
#if RPU_PROFILE_ENABLED
  double phase1_ms = PROFILE_ELAPSED_MS(phase1_start);
  auto preprocess_start = std::chrono::high_resolution_clock::now();
#endif

  Tensor a_final = a_contig;
  Tensor b_final = b_contig;

  if (a_contig.sizes() != b_contig.sizes()) {
    if (inplace) {
      b_final = b_contig.expand(a_contig.sizes());
    } else {
      try {
        auto broadcasted = at::broadcast_tensors({a_contig, b_contig});
        a_final = broadcasted[0];
        b_final = broadcasted[1];
      } catch (...) {
        auto a_cpu = rpu_to_cpu_zerocopy(a_contig);
        auto b_cpu = rpu_to_cpu_zerocopy(b_contig);
        at::Tensor cpu_result;
        switch (op_type) {
          case ValuOpType::ADD: cpu_result = at::add(a_cpu, b_cpu, alpha); break;
          case ValuOpType::MUL: cpu_result = at::mul(a_cpu, b_cpu); break;
          case ValuOpType::SUB: cpu_result = at::sub(a_cpu, b_cpu, alpha); break;
          case ValuOpType::DIV: cpu_result = at::div(a_cpu, b_cpu); break;
          default: cpu_result = at::add(a_cpu, b_cpu);
        }
        return cpu_result.to(a_contig.device());
      }
    }
  }

#if RPU_PROFILE_ENABLED
  double preprocess_ms = PROFILE_ELAPSED_MS(preprocess_start);
  auto alloc_start = std::chrono::high_resolution_clock::now();
#endif

  at::Tensor result = inplace ? *out_opt
                              : rpu_empty_strided(a_final.sizes(),
                                                  c10::contiguous_strides(a_final.sizes()),
                                                  a_contig.scalar_type(), a_contig.layout(), a_contig.device(), false);

#if RPU_PROFILE_ENABLED
  double alloc_ms = PROFILE_ELAPSED_MS(alloc_start);
  auto kernel_start = std::chrono::high_resolution_clock::now();
#endif

  // a_final 和 b_final 可能来自 broadcast_tensors，需要再次确保 contiguous
  auto a_kernel = a_final.is_contiguous() ? a_final : a_final.contiguous();
  auto b_kernel = b_final.is_contiguous() ? b_final : b_final.contiguous();

  rpu_launch_eltwise_binary_sameshape_kernel(a_kernel, b_kernel, alpha_f, result, op_type);

#if RPU_PROFILE_ENABLED
  double kernel_ms = PROFILE_ELAPSED_MS(kernel_start);
  // early_contig_ms 在函数开头已计算，这里 contiguous 已经在预处理阶段完成
  double total_ms = phase1_ms + preprocess_ms + alloc_ms + kernel_ms;

  // Build WrapperProfileData
  WrapperProfileData wp;
  wp.preprocess_ms = entry_to_phase1_ms + early_contig_ms + phase1_ms + preprocess_ms;
  wp.alloc_ms = alloc_ms;
  wp.contiguous_ms = 0;  // contiguous 已移到 preprocess 阶段
  wp.kernel_ms = kernel_ms;
  wp.total_ms = total_ms;

  // Get kernel profile data
  const KernelProfileData& kp = g_last_kernel_profile;

  // Use standard profile macro
  PROFILE_RECORD_AND_PRINT("binary_sameshape", g_profile_binary_sameshape, wp, kp);
#endif

  return result;
}

// 公开接口: 普通模式
at::Tensor binary_op_common_wrapper(
    const at::Tensor &a,
    const at::Tensor &b,
    const c10::Scalar &alpha,
    ValuOpType op_type,
    bool is_commutative) {
  return binary_op_impl(a, b, alpha, op_type, is_commutative, nullptr);
}

// 公开接口: inplace 模式
at::Tensor& binary_op_inplace_wrapper(
    at::Tensor &self,
    const at::Tensor &other,
    const c10::Scalar &alpha,
    ValuOpType op_type) {
  // Operand reversal is valid only for commutative operations. SUB and DIV must
  // preserve the original operand order on the broadcast fast path.
  const bool is_commutative =
      (op_type == ValuOpType::ADD || op_type == ValuOpType::MUL);
  binary_op_impl(self, other, alpha, op_type, is_commutative, &self);
  return self;
}

void rpu_launch_eltwise_binary_sameshape_kernel(const at::Tensor &input_a,
                                         const at::Tensor &input_b,
                                         const c10::Half alpha_f,
                                         at::Tensor &output,
                                         ValuOpType op_type) {
#if RPU_PROFILE_ENABLED
  KernelProfileData kp;
  PROFILE_START();
#endif

  size_t dwidth = sizeof(c10::Half);
  size_t n = 1;
  for (auto s : input_a.sizes())
    n *= s;
  size_t normal_blk_n = 100 * 256;
  size_t blk_cnt = CeilDiv(n, normal_blk_n);
  size_t last_blk_n = n - (blk_cnt - 1) * normal_blk_n;

  // 从缓存获取 DDR 版本 Kernel (O(1) 数组索引)
  Kernel_t* kernel = GET_KERNEL(KernelId::BINARY_SAMESHAPE_DDR);
  TORCH_CHECK(kernel != nullptr, "Failed to get binary_sameshape_ddr kernel from cache");
  kernel->reset_regs();
  auto* wq = GET_QUEUE(1);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.preprocess_ms);
#endif

  // 直接使用 tensor 指针 (已在 DDR)
  c10::Half *ap = input_a.data_ptr<c10::Half>();
  c10::Half *bp = input_b.data_ptr<c10::Half>();
  c10::Half *out = output.data_ptr<c10::Half>();

  // Flush 输入数据到 DDR
  rpu_ddr_flush(ap);
  rpu_ddr_flush(bp);
  rpu_ddr_flush(out);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.flush_ms);
#endif

  // 获取 DDR 设备地址 (v128 = 256B 单位)
  uint64_t a_addr_v128 = RpuGetDevAddr(ap) >> 8;
  uint64_t b_addr_v128 = RpuGetDevAddr(bp) >> 8;
  uint64_t out_addr_v128 = RpuGetDevAddr(out) >> 8;

  kernel->set_regs(0, (uint16_t)normal_blk_n);
  kernel->set_regs(1, (uint16_t)last_blk_n);
  kernel->set_regs(4, (uint16_t)(a_addr_v128 & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(a_addr_v128 >> 16));
  kernel->set_regs(6, (uint16_t)(b_addr_v128 & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(b_addr_v128 >> 16));
  kernel->set_regs(8, (uint16_t)(out_addr_v128 & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(out_addr_v128 >> 16));
  uint32_t block_data_stride_v128 = (normal_blk_n * dwidth) >> 8;
  kernel->set_regs(10, (uint16_t)(block_data_stride_v128 & 0xFFFF));
  kernel->set_regs(11, (uint16_t)(block_data_stride_v128 >> 16));

  c10::Half scale_a = 1.0;
  c10::Half scale_b = alpha_f;
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;
  kernel->set_regs(56, (scale_a.x));
  kernel->set_regs(57, (scale_b.x));
  kernel->set_regs(58, (scale_r.x));
  kernel->set_regs(59, (clip_max.x));
  kernel->set_regs(60, (clip_min.x));
  kernel->set_regs(61, (uint16_t)0x3);

  const auto& op_info = get_valu_op_info(op_type);
  kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
  kernel->set_regs(63, (uint16_t)(op_info.custom_op));

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.set_regs_ms);
#endif

  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, {0});

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.enqueue_ms);
#endif

  // Flush 输出缓存以读取结果
  rpu_ddr_flush(out);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.postflush_ms);
  kp.total_ms = kp.preprocess_ms + kp.flush_ms + kp.set_regs_ms + kp.enqueue_ms + kp.postflush_ms;
  g_last_kernel_profile = kp;
#endif
}

void rpu_launch_eltwise_binary_scalar_kernel(const at::Tensor &input_a,
                                             const c10::Half scalar_val,
                                             const c10::Half alpha_f,
                                             at::Tensor &output,
                                             ValuOpType op_type) {

  size_t dwidth = sizeof(c10::Half);
  size_t n = 1;
  for (auto s : input_a.sizes())
    n *= s;
  size_t normal_blk_n = 100 * 256;
  size_t blk_cnt = CeilDiv(n, normal_blk_n);
  size_t last_blk_n = n - (blk_cnt - 1) * normal_blk_n;

  // 从缓存获取 DDR 版本 Kernel (O(1) 数组索引)
  Kernel_t* kernel = (op_type == ValuOpType::POW)
      ? GET_KERNEL(KernelId::BINARY_SCALAR_POWER_DDR)
      : GET_KERNEL(KernelId::BINARY_SCALAR_DDR);
  TORCH_CHECK(kernel != nullptr, "Failed to get binary_scalar kernel from cache");
  kernel->reset_regs();

  auto* wq = GET_QUEUE(1);

  // 直接使用 tensor 指针 (已在 DDR)
  c10::Half *ap = input_a.data_ptr<c10::Half>();
  c10::Half *out = output.data_ptr<c10::Half>();

  // Flush 输入数据到 DDR
  rpu_ddr_flush(ap);
  rpu_ddr_flush(out);

  // 获取 DDR 设备地址 (v128 = 256B 单位)
  uint64_t a_addr_v128 = RpuGetDevAddr(ap) >> 8;
  uint64_t out_addr_v128 = RpuGetDevAddr(out) >> 8;

  kernel->set_regs(0, (uint16_t)normal_blk_n);
  kernel->set_regs(1, (uint16_t)last_blk_n);

  // For scalar kernel: scalar value goes to reg[2], scaled by alpha_f
  c10::Half scaled_scalar = c10::Half(float(scalar_val) * float(alpha_f));
  kernel->set_regs(2, scaled_scalar.x);

  kernel->set_regs(4, (uint16_t)(a_addr_v128 & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(a_addr_v128 >> 16));
  // reg[6,7] are not used for scalar kernel (no input_b address)
  kernel->set_regs(8, (uint16_t)(out_addr_v128 & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(out_addr_v128 >> 16));
  // block_data_stride 也要转换为 v128 单位
  uint32_t block_data_stride_v128 = (normal_blk_n * dwidth) >> 8;
  kernel->set_regs(10, (uint16_t)(block_data_stride_v128 & 0xFFFF));
  kernel->set_regs(11, (uint16_t)(block_data_stride_v128 >> 16));

  c10::Half scale_a = 1.0;
  c10::Half scale_b = 1.0;  // scalar already scaled above
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;
  kernel->set_regs(56, (scale_a.x));
  kernel->set_regs(57, (scale_b.x));
  kernel->set_regs(58, (scale_r.x));
  kernel->set_regs(59, (clip_max.x));
  kernel->set_regs(60, (clip_min.x));
  kernel->set_regs(61, (uint16_t)0x3);

  const auto& op_info = get_valu_op_info(op_type);
  kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
  kernel->set_regs(63, (uint16_t)(op_info.custom_op));

  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, {0});

  // Flush 输出缓存以读取结果
  rpu_ddr_flush(out);
}
// Launch geometry constants.
#define VLM_ENTRIES 384
#define WARP_VECTOR_SIZE 256
#define THD_VECTOR_SIZE 16
#define WARP_SIZE 16

// DDR batch kernels split N across grid.x. Parameters 10/11 encode the block
// stride in 256-byte units, so multi-block launches require an exact stride.
// C must be 16-aligned. SPM variants below use byte strides.
struct EltwiseDdrBatchGeom {
  bool C256;
  size_t c_loopcnt;
  size_t vlm_b;
  size_t normal_blk_loopcnt;
  size_t blk_cnt;
  size_t last_blk_loopcnt;
  int64_t rows_per_block;               // rows of the NxC operand one block owns
  uint32_t block_data_stride_v128;      // reg10/11
  bool stride_exact;                    // false => reg10/11 cannot address block > 0
  bool degenerate;                      // true => no usable geometry AT ALL (see below)
  uint16_t thread_params[WARP_SIZE];    // v16 kernels only (SCM regs 4096+)
};

// Both geometries divide by a VLM budget that SHRINKS as C grows, because VLM
// is a fixed 384 entries and the kernel reserves a share of them per v16 column
// vector. The budget reaches zero at a finite C — 3072 for 1xC_NxC and 6144
// for Nx1_NxC. Dividing a non-positive budget produces a zero or invalid block
// count and therefore a geometry that cannot address any rows.
// So mark the geometry degenerate instead of dividing. `degenerate` is what the
// _launchable predicates below reject; a rejected pattern is demoted to the
// sameshape kernel, which is correct for any shape. The arithmetic itself is
// pre-existing; the predicates are the contract that says "false means demote",
// so this is where it is enforced.
static void finish_geom(EltwiseDdrBatchGeom &g) {
  if (g.rows_per_block <= 0) {  // nothing to address per block
    g.degenerate = true;
    return;                     // leaves stride_exact false
  }
  const int64_t stride_bytes = g.rows_per_block * (int64_t)sizeof(c10::Half);
  g.block_data_stride_v128 = (uint32_t)(stride_bytes >> 8);
  g.stride_exact = (g.blk_cnt == 1) || (stride_bytes % 256 == 0);
}

static void fill_thread_params(EltwiseDdrBatchGeom &g, int64_t remain_data) {
  g.last_blk_loopcnt = remain_data / WARP_SIZE;
  const int64_t rmd_threads = remain_data % WARP_SIZE;
  for (int i = 0; i < WARP_SIZE; i++) {
    g.thread_params[i] = (uint16_t)g.last_blk_loopcnt + (i < rmd_threads ? 1 : 0);
  }
}

// input_a is (1, C), input_b is (N, C).
static EltwiseDdrBatchGeom eltwise_1xC_NxC_ddr_geom(int64_t n, int64_t c) {
  EltwiseDdrBatchGeom g{};
  g.C256 = (c % 256 == 0);
  // 0 at c == 3072, negative above it — see finish_geom.
  const int64_t vlm_budget =
      (VLM_ENTRIES - 2 * CeilDiv(c, THD_VECTOR_SIZE)) * WARP_VECTOR_SIZE;
  if (vlm_budget <= 0) { g.degenerate = true; return g; }
  const int64_t blk_cnt = CeilDiv(n * c, vlm_budget);
  if (blk_cnt < 1) { g.degenerate = true; return g; }
  g.blk_cnt = (size_t)blk_cnt;

  if (g.C256) {
    g.c_loopcnt = c / WARP_VECTOR_SIZE;
    g.vlm_b = g.c_loopcnt;
    g.normal_blk_loopcnt = CeilDiv(n, (int64_t)g.blk_cnt);
    g.last_blk_loopcnt = n - g.normal_blk_loopcnt * (g.blk_cnt - 1);
    g.rows_per_block = (int64_t)g.normal_blk_loopcnt;
  } else {
    g.c_loopcnt = c / THD_VECTOR_SIZE;
    g.vlm_b = g.c_loopcnt;
    g.normal_blk_loopcnt = CeilDiv(n, (int64_t)(WARP_SIZE * g.blk_cnt));
    g.rows_per_block = (int64_t)(g.normal_blk_loopcnt * WARP_SIZE);
    fill_thread_params(g, n - g.rows_per_block * (int64_t)(g.blk_cnt - 1));
  }
  finish_geom(g);
  return g;
}

// input_a is (N, 1), input_b is (N, C).
static EltwiseDdrBatchGeom eltwise_Nx1_NxC_ddr_geom(int64_t n, int64_t c) {
  EltwiseDdrBatchGeom g{};
  g.C256 = (c == 256);

  if (g.C256) {
    g.c_loopcnt = 1;
    g.vlm_b = 1;
    g.normal_blk_loopcnt = 256;
    g.blk_cnt = CeilDiv(n, (int64_t)g.normal_blk_loopcnt);
    g.last_blk_loopcnt = n - g.normal_blk_loopcnt * (g.blk_cnt - 1);
    g.rows_per_block = (int64_t)g.normal_blk_loopcnt;
  } else {
    // 0 from c == 6144 up, which made this CeilDiv(n, 0) — see finish_geom.
    const int64_t max_possible_loopcnt = VLM_ENTRIES / (c / THD_VECTOR_SIZE + 1);
    const int64_t rows_budget = max_possible_loopcnt * WARP_SIZE;
    if (rows_budget <= 0) { g.degenerate = true; return g; }
    const int64_t blk_cnt = CeilDiv(n, rows_budget);
    if (blk_cnt < 1) { g.degenerate = true; return g; }
    g.blk_cnt = (size_t)blk_cnt;
    g.c_loopcnt = c / THD_VECTOR_SIZE;
    g.vlm_b = (size_t)max_possible_loopcnt;
    g.normal_blk_loopcnt = CeilDiv(n, (int64_t)(WARP_SIZE * g.blk_cnt));
    g.rows_per_block = (int64_t)(g.normal_blk_loopcnt * WARP_SIZE);
    fill_thread_params(g, n - g.rows_per_block * (int64_t)(g.blk_cnt - 1));
  }
  finish_geom(g);
  return g;
}

static bool eltwise_1xC_NxC_ddr_launchable(int64_t n, int64_t c) {
  if (c % THD_VECTOR_SIZE != 0) return false;
  const EltwiseDdrBatchGeom g = eltwise_1xC_NxC_ddr_geom(n, c);
  return !g.degenerate && g.stride_exact;
}

static bool eltwise_Nx1_NxC_ddr_launchable(int64_t n, int64_t c) {
  if (c % THD_VECTOR_SIZE != 0) return false;
  const EltwiseDdrBatchGeom g = eltwise_Nx1_NxC_ddr_geom(n, c);
  return !g.degenerate && g.stride_exact;
}

// Nx1_NxC kernel: input_a is (N, 1), input_b is (N, C), output is (N, C)
// When is_bopa=false: computes A op B (small op large)
// When is_bopa=true:  computes B op A (large op small), used for non-commutative ops
void rpu_launch_eltwise_binary_Nx1_NxC_kernel(const at::Tensor &input_a,
                                              const at::Tensor &input_b,
                                              const c10::Half alpha_f,
                                              at::Tensor &output,
                                              ValuOpType op_type,
                                              bool is_bopa) {
#if RPU_PROFILE_ENABLED
  KernelProfileData kp;
  PROFILE_START();
#endif

  // Get dimensions - input_a is (N, 1), input_b is (N, C)
  int64_t n = input_b.size(0);
  int64_t c = input_b.size(1);

  const EltwiseDdrBatchGeom geom = eltwise_Nx1_NxC_ddr_geom(n, c);
  TORCH_CHECK(eltwise_Nx1_NxC_ddr_launchable(n, c),
              "binary_Nx1_NxC DDR kernel cannot address [", n, ",", c, "]; "
              "binary_op_impl must demote this pattern (see EltwiseDdrBatchGeom)");
  const bool C256 = geom.C256;
  const size_t normal_blk_loopcnt = geom.normal_blk_loopcnt;
  const size_t blk_cnt = geom.blk_cnt;
  const size_t last_blk_loopcnt = geom.last_blk_loopcnt;
  const uint32_t block_data_stride_v128 = geom.block_data_stride_v128;

  // 从缓存获取 DDR 版本 Kernel (O(1) 数组索引)
  Kernel_t* kernel;
  if (is_bopa) {
    kernel = C256
        ? GET_KERNEL(KernelId::BINARY_NX1_NXC256_BATCH_BOPA_DDR)
        : GET_KERNEL(KernelId::BINARY_NX1_NXC_V16_BATCH_BOPA_DDR);
  } else {
    kernel = C256
        ? GET_KERNEL(KernelId::BINARY_NX1_NXC256_BATCH_DDR)
        : GET_KERNEL(KernelId::BINARY_NX1_NXC_V16_BATCH_DDR);
  }
  TORCH_CHECK(kernel != nullptr, "Failed to get binary_Nx1_NxC kernel from cache");
  kernel->reset_regs();

  auto* wq = GET_QUEUE(1);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.preprocess_ms);
#endif

  // 直接使用 tensor 指针 (已在 DDR)
  c10::Half *ap = input_a.data_ptr<c10::Half>();
  c10::Half *bp = input_b.data_ptr<c10::Half>();
  c10::Half *out = output.data_ptr<c10::Half>();

  // Flush 输入数据到 DDR
  rpu_ddr_flush(ap);
  rpu_ddr_flush(bp);
  rpu_ddr_flush(out);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.flush_ms);
#endif

  // 获取 DDR 设备地址 (v128 = 256B 单位)
  uint64_t a_addr_v128 = RpuGetDevAddr(ap) >> 8;
  uint64_t b_addr_v128 = RpuGetDevAddr(bp) >> 8;
  uint64_t out_addr_v128 = RpuGetDevAddr(out) >> 8;

  kernel->set_regs(0, (uint16_t)normal_blk_loopcnt);
  if (C256) {
    kernel->set_regs(2, (uint16_t)last_blk_loopcnt);
  }
  kernel->set_regs(4, (uint16_t)(a_addr_v128 & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(a_addr_v128 >> 16));
  kernel->set_regs(6, (uint16_t)(b_addr_v128 & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(b_addr_v128 >> 16));
  kernel->set_regs(8, (uint16_t)(out_addr_v128 & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(out_addr_v128 >> 16));
  kernel->set_regs(10, (uint16_t)(block_data_stride_v128 & 0xFFFF));
  kernel->set_regs(11, (uint16_t)(block_data_stride_v128 >> 16));
  kernel->set_regs(12, (uint16_t)geom.c_loopcnt);
  kernel->set_regs(13, (uint16_t)geom.vlm_b);

  // Populate the required SCM parameters.
  if (!C256) {
    for (int i = 0; i < WARP_SIZE; i++) {
      kernel->set_regs((uint32_t)i + 4096, geom.thread_params[i]);
    }
  }

  c10::Half scale_a = 1.0;
  c10::Half scale_b = alpha_f;
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;
  kernel->set_regs(56, scale_a.x);
  kernel->set_regs(57, scale_b.x);
  kernel->set_regs(58, scale_r.x);
  kernel->set_regs(59, clip_max.x);
  kernel->set_regs(60, clip_min.x);
  kernel->set_regs(61, (uint16_t)0x3);

  const auto& op_info = get_valu_op_info(op_type);
  kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
  kernel->set_regs(63, (uint16_t)(op_info.custom_op));

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.set_regs_ms);
#endif

  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, {0});

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.enqueue_ms);
#endif

  // Flush 输出缓存以读取结果
  rpu_ddr_flush(out);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.postflush_ms);
  kp.total_ms = kp.preprocess_ms + kp.flush_ms + kp.set_regs_ms + kp.enqueue_ms + kp.postflush_ms;
  g_last_kernel_profile = kp;
#endif
}

// 1xC_NxC kernel: input_a is (1, C), input_b is (N, C), output is (N, C)
// When is_bopa=false: computes A op B (small op large)
// When is_bopa=true:  computes B op A (large op small), used for non-commutative ops
void rpu_launch_eltwise_binary_1xC_NxC_kernel(const at::Tensor &input_a,
                                              const at::Tensor &input_b,
                                              const c10::Half alpha_f,
                                              at::Tensor &output,
                                              ValuOpType op_type,
                                              bool is_bopa) {
#if RPU_PROFILE_ENABLED
  KernelProfileData kp;
  PROFILE_START();
#endif

  // Get dimensions - input_a is (1, C), input_b is (N, C)
  int64_t n = input_b.size(0);
  int64_t c = input_b.size(1);

  const EltwiseDdrBatchGeom geom = eltwise_1xC_NxC_ddr_geom(n, c);
  TORCH_CHECK(eltwise_1xC_NxC_ddr_launchable(n, c),
              "binary_1xC_NxC DDR kernel cannot address [", n, ",", c, "]; "
              "binary_op_impl must demote this pattern (see EltwiseDdrBatchGeom)");
  const bool C256 = geom.C256;
  const size_t normal_blk_loopcnt = geom.normal_blk_loopcnt;
  const size_t blk_cnt = geom.blk_cnt;
  const size_t last_blk_loopcnt = geom.last_blk_loopcnt;
  const uint32_t block_data_stride_v128 = geom.block_data_stride_v128;

  // 从缓存获取 DDR 版本 Kernel (O(1) 数组索引)
  Kernel_t* kernel;
  if (is_bopa) {
    kernel = C256
        ? GET_KERNEL(KernelId::BINARY_1XC_NXC_V256_BATCH_BOPA_DDR)
        : GET_KERNEL(KernelId::BINARY_1XC_NXC_V16_BATCH_BOPA_DDR);
  } else {
    kernel = C256
        ? GET_KERNEL(KernelId::BINARY_1XC_NXC_V256_BATCH_DDR)
        : GET_KERNEL(KernelId::BINARY_1XC_NXC_V16_BATCH_DDR);
  }
  TORCH_CHECK(kernel != nullptr, "Failed to get binary_1xC_NxC kernel from cache");
  kernel->reset_regs();

  auto* wq = GET_QUEUE(1);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.preprocess_ms);
#endif

  // 直接使用 tensor 指针 (已在 DDR)
  c10::Half *ap = input_a.data_ptr<c10::Half>();
  c10::Half *bp = input_b.data_ptr<c10::Half>();
  c10::Half *out = output.data_ptr<c10::Half>();

  // Flush 输入数据到 DDR
  rpu_ddr_flush(ap);
  rpu_ddr_flush(bp);
  rpu_ddr_flush(out);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.flush_ms);
#endif

  // 获取 DDR 设备地址 (v128 = 256B 单位)
  uint64_t a_addr_v128 = RpuGetDevAddr(ap) >> 8;
  uint64_t b_addr_v128 = RpuGetDevAddr(bp) >> 8;
  uint64_t out_addr_v128 = RpuGetDevAddr(out) >> 8;

  kernel->set_regs(0, (uint16_t)normal_blk_loopcnt);
  if (C256) {
    kernel->set_regs(2, (uint16_t)last_blk_loopcnt);
  }
  kernel->set_regs(4, (uint16_t)(a_addr_v128 & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(a_addr_v128 >> 16));
  kernel->set_regs(6, (uint16_t)(b_addr_v128 & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(b_addr_v128 >> 16));
  kernel->set_regs(8, (uint16_t)(out_addr_v128 & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(out_addr_v128 >> 16));
  kernel->set_regs(10, (uint16_t)(block_data_stride_v128 & 0xFFFF));
  kernel->set_regs(11, (uint16_t)(block_data_stride_v128 >> 16));
  kernel->set_regs(12, (uint16_t)geom.c_loopcnt);
  kernel->set_regs(13, (uint16_t)geom.vlm_b);

  // Populate the required SCM parameters.
  if (!C256) {
    for (int i = 0; i < WARP_SIZE; i++) {
      kernel->set_regs((uint32_t)i + 4096, geom.thread_params[i]);
    }
  }

  c10::Half scale_a = 1.0;
  c10::Half scale_b = alpha_f;
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;
  kernel->set_regs(56, scale_a.x);
  kernel->set_regs(57, scale_b.x);
  kernel->set_regs(58, scale_r.x);
  kernel->set_regs(59, clip_max.x);
  kernel->set_regs(60, clip_min.x);
  kernel->set_regs(61, (uint16_t)0x3);

  const auto& op_info = get_valu_op_info(op_type);
  kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
  kernel->set_regs(63, (uint16_t)(op_info.custom_op));

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.set_regs_ms);
#endif

  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1}, {0});

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.enqueue_ms);
#endif

  // Flush 输出缓存以读取结果
  rpu_ddr_flush(out);

#if RPU_PROFILE_ENABLED
  PROFILE_CHECKPOINT(kp.postflush_ms);
  kp.total_ms = kp.preprocess_ms + kp.flush_ms + kp.set_regs_ms + kp.enqueue_ms + kp.postflush_ms;
  g_last_kernel_profile = kp;
#endif
}

// ------------------------ add.Tensor ------------------------
at::Tensor rpu_add(const at::Tensor &a, const at::Tensor &b,
                   const c10::Scalar &alpha) {
  return binary_op_common_wrapper(a, b, alpha, ValuOpType::ADD, /*is_commutative=*/true);
}


at::Tensor &rpu_add_(at::Tensor &self, const at::Tensor &other,
                     const c10::Scalar &alpha) {
  return binary_op_inplace_wrapper(self, other, alpha, ValuOpType::ADD);
}

// ------------------------ sub.Tensor ------------------------
at::Tensor rpu_sub(const at::Tensor &a, const at::Tensor &b,
                   const c10::Scalar &alpha) {
  return binary_op_common_wrapper(a, b, alpha, ValuOpType::SUB, /*is_commutative=*/false);
}

at::Tensor &rpu_sub_(at::Tensor &self, const at::Tensor &other,
                     const c10::Scalar &alpha) {
  return binary_op_inplace_wrapper(self, other, alpha, ValuOpType::SUB);
}

at::Tensor &rpu_sub_scalar_(at::Tensor &self, const c10::Scalar &other,
                            const c10::Scalar &alpha) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_sub_scalar_: self not on RPU");

  if (self.scalar_type() != at::kHalf) {
    auto self_cpu = rpu_to_cpu_zerocopy(self);
    self_cpu.sub_(other, alpha);
    return self;
  }

  // scalar_val = other * alpha
  const c10::Half scalar_val = c10::Half(other.toFloat() * alpha.toFloat());
  const c10::Half alpha_f = 1.0;
  rpu_launch_eltwise_binary_scalar_kernel(self.contiguous(), scalar_val,
                                          alpha_f, self, ValuOpType::SUB);
  return self;
}

// ------------------------ div.Tensor ------------------------
at::Tensor rpu_div(const at::Tensor &a, const at::Tensor &b) {
  return binary_op_common_wrapper(a, b, /*alpha=*/1, ValuOpType::DIV, /*is_commutative=*/false);
}

at::Tensor &rpu_div_(at::Tensor &self, const at::Tensor &other) {
  return binary_op_inplace_wrapper(self, other, /*alpha=*/1, ValuOpType::DIV);
}

at::Tensor rpu_div_scalar(const at::Tensor &a, const c10::Scalar &b) {
  TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_div_scalar: a not on RPU");

  // Only Half is supported by RPU kernel - 使用零拷贝优化
  if (a.scalar_type() != at::kHalf) {
    auto a_cpu = rpu_to_cpu_zerocopy(a);
    auto result = at::empty_like(a);
    auto result_cpu = rpu_to_cpu_zerocopy(result);
    at::div_out(result_cpu, a_cpu, b);
    return result;
  }

  // 获取连续版本的输入
  auto a_contig = a.contiguous();

  // 输出必须是连续的，与 kernel 写入布局匹配
  auto result = rpu_empty_strided(a_contig.sizes(),
                                  compute_contiguous_strides(a_contig.sizes()),
                                  a_contig.scalar_type(),
                                  a_contig.layout(), a_contig.device(), false);

  const c10::Half scalar_val = c10::Half(b.toFloat());
  const c10::Half alpha_f = 1.0;
  rpu_launch_eltwise_binary_scalar_kernel(a_contig, scalar_val,
                                          alpha_f, result, ValuOpType::DIV);

  return result;
}

// ------------------------ mul.Tensor ------------------------
at::Tensor rpu_mul(const at::Tensor &a, const at::Tensor &b) {
  return binary_op_common_wrapper(a, b, /*alpha=*/1, ValuOpType::MUL, /*is_commutative=*/true);
}

at::Tensor rpu_mul_scalar(const at::Tensor &a, const c10::Scalar &b) {
  TORCH_CHECK(a.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_mul_scalar: a not on RPU");

  // Only Half is supported by RPU kernel - 使用零拷贝优化
  if (a.scalar_type() != at::kHalf) {
    auto a_cpu = rpu_to_cpu_zerocopy(a);
    auto result = at::empty_like(a);
    auto result_cpu = rpu_to_cpu_zerocopy(result);
    at::mul_out(result_cpu, a_cpu, b);
    return result;
  }

  // 获取连续版本的输入
  auto a_contig = a.contiguous();

  // 输出必须是连续的，与 kernel 写入布局匹配
  auto result = rpu_empty_strided(a_contig.sizes(),
                                  compute_contiguous_strides(a_contig.sizes()),
                                  a_contig.scalar_type(),
                                  a_contig.layout(), a_contig.device(), false);

  const c10::Half scalar_val = c10::Half(b.toFloat());
  const c10::Half alpha_f = 1.0;
  rpu_launch_eltwise_binary_scalar_kernel(a_contig, scalar_val,
                                          alpha_f, result, ValuOpType::MUL);

  return result;
}

at::Tensor &rpu_mul_(at::Tensor &self, const at::Tensor &other) {
  return binary_op_inplace_wrapper(self, other, /*alpha=*/1, ValuOpType::MUL);
}

at::Tensor &rpu_mul_scalar_(at::Tensor &self, const c10::Scalar &other) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_mul_scalar_: self not on RPU");

  // Only Half is supported by RPU kernel - 使用零拷贝优化
  if (self.scalar_type() != at::kHalf) {
    auto self_cpu = rpu_to_cpu_zerocopy(self);
    self_cpu.mul_(other);
    // 无需 copy_ 回来
    return self;
  }

  const c10::Half scalar_val = c10::Half(other.toFloat());
  const c10::Half alpha_f = 1.0;
  rpu_launch_eltwise_binary_scalar_kernel(self.contiguous(), scalar_val,
                                          alpha_f, self, ValuOpType::MUL);

  return self;
}

// ------------------------ neg ------------------------
at::Tensor rpu_neg(const at::Tensor &self) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_neg: self must be on RPU");

  // neg(x) = -1 * x
  return rpu_mul_scalar(self, c10::Scalar(-1.0f));
}

// ------------------------ pow ------------------------
at::Tensor rpu_pow_tensor_scalar(const at::Tensor &self, const c10::Scalar &exponent) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_pow: self must be on RPU");

  // Only Half is supported by RPU kernel - 使用零拷贝优化
  if (self.scalar_type() != at::kHalf) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::pow_out(cpu_result, cpu_self, exponent);
    return result;
  }

  // 获取连续版本的输入
  auto self_contig = self.contiguous();

  // 输出必须是连续的，与 kernel 写入布局匹配
  auto result = rpu_empty_strided(self_contig.sizes(),
                                  compute_contiguous_strides(self_contig.sizes()),
                                  self_contig.scalar_type(),
                                  self_contig.layout(), self_contig.device(), false);

  const c10::Half scalar_val = c10::Half(exponent.toFloat());
  const c10::Half alpha_f = 1.0;
  rpu_launch_eltwise_binary_scalar_kernel(self_contig, scalar_val,
                                          alpha_f, result, ValuOpType::POW);

  return result;
}

at::Tensor rpu_pow_tensor_tensor(const at::Tensor &self, const at::Tensor &exponent) {
  TORCH_CHECK(self.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_pow: self must be on RPU");

  // Only Half is supported by RPU kernel - 使用零拷贝优化
  if (self.scalar_type() != at::kHalf) {
    auto cpu_self = rpu_to_cpu_zerocopy(self);
    auto cpu_exp = (exponent.device().type() == c10::DeviceType::PrivateUse1)
                   ? rpu_to_cpu_zerocopy(exponent) : exponent;
    auto result = at::empty_like(self);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::pow_out(cpu_result, cpu_self, cpu_exp);
    return result;
  }

  // Handle tensor from different device
  Tensor exp_rpu = exponent;
  if (exponent.device() != self.device()) {
    exp_rpu = exponent.to(self.device());
  }

  if (exp_rpu.scalar_type() != self.scalar_type()) {
    exp_rpu = exp_rpu.to(self.scalar_type());
  }

  // Handle broadcasting
  Tensor self_broadcasted = self;
  Tensor exp_broadcasted = exp_rpu;
  if (self.sizes() != exp_rpu.sizes()) {
    try {
      auto broadcasted = at::broadcast_tensors({self, exp_rpu});
      self_broadcasted = broadcasted[0];
      exp_broadcasted = broadcasted[1];
    } catch (...) {
      // Fallback - 使用零拷贝优化
      auto cpu_self = rpu_to_cpu_zerocopy(self);
      auto cpu_exp = (exponent.device().type() == c10::DeviceType::PrivateUse1)
                     ? rpu_to_cpu_zerocopy(exponent) : exponent;
      auto cpu_out = at::pow(cpu_self, cpu_exp);
      return cpu_out.to(self.device());
    }
  }

  auto result = rpu_empty_strided(self_broadcasted.sizes(),
                                  compute_contiguous_strides(self_broadcasted.sizes()),
                                  self.scalar_type(), self.layout(), self.device(), false);

  const c10::Half alpha_f = 1.0;
  rpu_launch_eltwise_binary_sameshape_kernel(self_broadcasted.contiguous(), exp_broadcasted.contiguous(),
                                             alpha_f, result, ValuOpType::POW);

  return result;
}

at::Tensor rpu_pow_scalar_tensor(const c10::Scalar &self, const at::Tensor &exponent) {
  TORCH_CHECK(exponent.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_pow: exponent must be on RPU");

  // Only Half is supported by RPU kernel - 使用零拷贝优化
  if (exponent.scalar_type() != at::kHalf) {
    auto cpu_exp = rpu_to_cpu_zerocopy(exponent);
    auto result = at::empty_like(exponent);
    auto cpu_result = rpu_to_cpu_zerocopy(result);
    at::pow_out(cpu_result, self, cpu_exp);
    return result;
  }

  // 获取连续版本的输入
  auto exp_contig = exponent.contiguous();

  // Create base tensor and broadcast to exponent's shape
  auto base_tensor = at::full({}, self, exp_contig.options()).expand(exp_contig.sizes()).contiguous();

  // 输出必须是连续的，与 kernel 写入布局匹配
  auto result = rpu_empty_strided(exp_contig.sizes(),
                                  compute_contiguous_strides(exp_contig.sizes()),
                                  exp_contig.scalar_type(),
                                  exp_contig.layout(), exp_contig.device(), false);

  const c10::Half alpha_f = 1.0;
  rpu_launch_eltwise_binary_sameshape_kernel(base_tensor, exp_contig,
                                             alpha_f, result, ValuOpType::POW);

  return result;
}

// =============================================================================
// SPM Binary Kernels (for fused decoder layer MLP)
// =============================================================================
// Input/Output data in SPM, runs on all 8 cores.
// Each core processes its own portion of the data.
//
// Differences from DDR version:
//   - reg[61] = 0x1 (SPM mode) instead of 0x3 (DDR mode)
//   - Addresses are 32-bit SPM addresses (not v128 units)
//   - block_data_stride is in bytes (not v128 units)
//
// Supported patterns:
//   1. sameshape: A[N] op B[N] -> Out[N] (generic binary)
//   2. scalar: A[N] op scalar -> Out[N]
//   3. Nx1_NxC: A[N,1] op B[N,C] -> Out[N,C] (broadcast along C)
//   4. 1xC_NxC: A[1,C] op B[N,C] -> Out[N,C] (broadcast along N)
//
// All operations (ADD, SUB, MUL, DIV, POW) are supported.
// In-place operations are supported (output can be same as input A).
// =============================================================================

#define NUM_CORES_BINARY 8

// ==================== Sameshape SPM Kernel ====================
// Generic sameshape SPM binary kernel
// Supports: ADD, SUB, MUL, DIV, POW
// Supports in-place: output_spm_addr can be same as a_spm_addr
void rpu_launch_eltwise_binary_spm_kernel(
    uint32_t a_spm_addr,               // SPM input A address (core 0 base)
    uint32_t b_spm_addr,               // SPM input B address (core 0 base)
    uint32_t output_spm_addr,          // SPM output address (core 0 base), can be same as A for in-place
    int64_t num_elements,              // Total elements per core
    ValuOpType op_type,
    c10::Half alpha,                   // Scale factor for B (used in add/sub: out = A + alpha*B)
    int num_cores)                     // number of cores
{
  size_t dwidth = sizeof(c10::Half);
  size_t n = num_elements;
  size_t normal_blk_n = 100 * 256;
  size_t blk_cnt = CeilDiv(n, normal_blk_n);
  size_t last_blk_n = n - (blk_cnt - 1) * normal_blk_n;

  // block_data_stride in bytes for SPM (NOT shifted like DDR)
  uint32_t block_data_stride = normal_blk_n * dwidth;

  // Scale parameters
  c10::Half scale_a = 1.0;
  c10::Half scale_b = alpha;  // For add/sub: out = A + alpha * B
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;

  const auto& op_info = get_valu_op_info(op_type);

  // 定义寄存器配置 lambda
  auto setup_regs = [=](Kernel_t* kernel) {
    // SPM addresses (32-bit, NOT shifted like DDR v128 addresses)
    kernel->set_regs(0, (uint16_t)normal_blk_n);
    kernel->set_regs(1, (uint16_t)last_blk_n);
    kernel->set_regs(4, (uint16_t)(a_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(a_spm_addr >> 16));
    kernel->set_regs(6, (uint16_t)(b_spm_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(b_spm_addr >> 16));
    kernel->set_regs(8, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(output_spm_addr >> 16));

    kernel->set_regs(10, (uint16_t)(block_data_stride & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(block_data_stride >> 16));

    kernel->set_regs(56, (scale_a.x));
    kernel->set_regs(57, (scale_b.x));
    kernel->set_regs(58, (scale_r.x));
    kernel->set_regs(59, (clip_max.x));
    kernel->set_regs(60, (clip_min.x));
    kernel->set_regs(61, (uint16_t)0x1);  // SPM mode flag (vs 0x3 for DDR)

    kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
    kernel->set_regs(63, (uint16_t)(op_info.custom_op));
  };

  // 普通模式：立即执行
  Kernel_t* kernel = GET_KERNEL(KernelId::BINARY_SAMESHAPE);
  TORCH_CHECK(kernel != nullptr, "Failed to get binary SPM kernel from cache");
  kernel->reset_regs();
  setup_regs(kernel);

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1},
                    core_list);
}

// ==================== Scalar SPM Kernel ====================
// Tensor op scalar, where tensor is in SPM
// Supports in-place: output_spm_addr can be same as a_spm_addr
void rpu_launch_eltwise_binary_scalar_spm_kernel(
    uint32_t a_spm_addr,               // SPM input address (core 0 base)
    c10::Half scalar_val,              // Scalar value
    uint32_t output_spm_addr,          // SPM output address (core 0 base)
    int64_t num_elements,              // Total elements per core
    ValuOpType op_type)
{
  size_t dwidth = sizeof(c10::Half);
  size_t n = num_elements;
  size_t normal_blk_n = 100 * 256;
  size_t blk_cnt = CeilDiv(n, normal_blk_n);
  size_t last_blk_n = n - (blk_cnt - 1) * normal_blk_n;

  // SPM byte-addressed block_data_stride (NOT shifted, unlike DDR)
  uint32_t block_data_stride = normal_blk_n * dwidth;

  c10::Half scale_a = 1.0;
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;

  const auto& op_info = get_valu_op_info(op_type);

  auto setup_regs = [=](Kernel_t* kernel) {
    kernel->set_regs(0, (uint16_t)normal_blk_n);
    kernel->set_regs(1, (uint16_t)last_blk_n);
    // reg[2] = scalar value (same as DDR version)
    kernel->set_regs(2, scalar_val.x);
    kernel->set_regs(4, (uint16_t)(a_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(a_spm_addr >> 16));
    kernel->set_regs(8, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(output_spm_addr >> 16));
    kernel->set_regs(10, (uint16_t)(block_data_stride & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(block_data_stride >> 16));
    kernel->set_regs(56, (scale_a.x));
    kernel->set_regs(57, (c10::Half(1.0).x));  // scale_b
    kernel->set_regs(58, (scale_r.x));
    kernel->set_regs(59, (clip_max.x));
    kernel->set_regs(60, (clip_min.x));
    kernel->set_regs(61, (uint16_t)0x1);  // SPM mode flag
    kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
    kernel->set_regs(63, (uint16_t)(op_info.custom_op));
  };

  Kernel_t* kernel = GET_KERNEL(KernelId::BINARY_SCALAR);
  TORCH_CHECK(kernel != nullptr, "Failed to get binary_scalar SPM kernel");
  kernel->reset_regs();
  setup_regs(kernel);

  auto* wq = GET_QUEUE(NUM_CORES_BINARY);
  wq->set_broadcast_mode(true);
  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1},
                    {0, 1, 2, 3, 4, 5, 6, 7});
}

// ==================== Nx1_NxC SPM Kernel ====================
// input_a is (N, 1), input_b is (N, C), output is (N, C)
// When is_bopa=false: computes A op B (small op large)
// When is_bopa=true:  computes B op A (large op small), used for non-commutative ops
// Supports in-place: output_spm_addr can be same as b_spm_addr
void rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
    uint32_t a_spm_addr,               // SPM input A address [N, 1]
    uint32_t b_spm_addr,               // SPM input B address [N, C]
    uint32_t output_spm_addr,          // SPM output address [N, C]
    int64_t n,                         // N dimension
    int64_t c,                         // C dimension
    c10::Half alpha_f,
    ValuOpType op_type,
    bool is_bopa,
    int num_cores)                     // participating cores (default 8)
{
  TORCH_CHECK(num_cores >= 1 && num_cores <= NUM_CORES_BINARY,
              "Nx1_NxC spm: num_cores must be in [1,", NUM_CORES_BINARY,
              "], got ", num_cores);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back((uint8_t)i);
  // ---- Tile-based path (C not v16-aligned, or C>1024, or N>8192) ----
  // The v16/v256 batch path has a bounded encoded loop count. Shapes outside
  // that envelope use binary_Nx1_NxC_tileN/tileC, which tile both N and C.
  // reg0-1=n, reg2-3=c (u32);
  // reg10=last_blk_n, reg11=last_blk_c (u16); grid {blk_cnt_x=⌈c/256⌉, blk_cnt_y=⌈n/256⌉, 1}.
  if (c % 16 != 0 || c > 1024 || n > 8192) {
    const bool tile_c = (c >= 256);
    std::string kname = std::string("binary_Nx1_NxC_") +
                        (tile_c ? "tileC" : "tileN") + (is_bopa ? "_bopa" : "");
    Kernel_t* kernel = RpuKernelGraph::active().get_kernel_reset(kname);
    TORCH_CHECK(kernel != nullptr, "Failed to get ", kname, " kernel");

    int64_t blk_cnt_x, blk_cnt_y;
    if (tile_c) {
      blk_cnt_x = CeilDiv(c, (int64_t)256);
      blk_cnt_y = CeilDiv(n, (int64_t)256);
      const int64_t last_blk_n = n - (blk_cnt_y - 1) * 256;
      const int64_t last_blk_c = c - (blk_cnt_x - 1) * 256;
      kernel->set_regs(10, (uint16_t)last_blk_n);
      kernel->set_regs(11, (uint16_t)last_blk_c);
    } else {  // tileN
      blk_cnt_x = 1;
      blk_cnt_y = CeilDiv(n, (int64_t)256);
      const int64_t last_blk_n = n - (blk_cnt_y - 1) * 256;
      kernel->set_regs(10, (uint16_t)last_blk_n);
      kernel->set_regs(11, (uint16_t)0);
    }
    kernel->set_regs(0, (uint16_t)(n & 0xFFFF));
    kernel->set_regs(1, (uint16_t)((n >> 16) & 0xFFFF));
    kernel->set_regs(2, (uint16_t)(c & 0xFFFF));
    kernel->set_regs(3, (uint16_t)((c >> 16) & 0xFFFF));
    kernel->set_regs(4, (uint16_t)(a_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(a_spm_addr >> 16));
    kernel->set_regs(6, (uint16_t)(b_spm_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(b_spm_addr >> 16));
    kernel->set_regs(8, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(output_spm_addr >> 16));
    // eltwise common (regs 56-63) — imm3/imm4 = 1.0 per runtime COMMON_ELTWISE_BINARY_INIT.
    c10::Half scale_a = 1.0, scale_b = alpha_f, scale_r = 1.0, imm3 = 1.0, imm4 = 1.0;
    kernel->set_regs(56, scale_a.x);
    kernel->set_regs(57, scale_b.x);
    kernel->set_regs(58, scale_r.x);
    kernel->set_regs(59, imm3.x);
    kernel->set_regs(60, imm4.x);
    kernel->set_regs(61, (uint16_t)0x1);
    const auto& op_info = get_valu_op_info(op_type);
    kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
    kernel->set_regs(63, (uint16_t)(op_info.custom_op));
    kernel->set_regs(64, (uint16_t)blk_cnt_x);
    kernel->set_regs(65, (uint16_t)blk_cnt_y);
    kernel->set_regs(66, (uint16_t)1);

    auto* wq = GET_QUEUE(num_cores);
    wq->set_broadcast_mode(true);
    wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt_x, (uint16_t)blk_cnt_y, (uint16_t)1},
                      core_list);
    return;
  }

  size_t dwidth = sizeof(c10::Half);

  bool C256 = (c == 256);

  size_t c_loopcnt, vlm_b, normal_blk_loopcnt, blk_cnt, last_blk_loopcnt;
  uint32_t block_data_stride;

  // Prepare the required SCM parameters.
  uint16_t thread_params[WARP_SIZE] = {0};
  int64_t rmd_threads = 0;

  if (C256) {
    c_loopcnt = 1;
    vlm_b = 1;
    normal_blk_loopcnt = 256;
    blk_cnt = CeilDiv(n, normal_blk_loopcnt);
    last_blk_loopcnt = n - normal_blk_loopcnt * (blk_cnt - 1);
    block_data_stride = normal_blk_loopcnt * dwidth;  // SPM: not shifted
  } else {
    size_t max_possible_loopcnt = VLM_ENTRIES / (c / THD_VECTOR_SIZE + 1);
    blk_cnt = CeilDiv(n, max_possible_loopcnt * WARP_SIZE);
    c_loopcnt = c / THD_VECTOR_SIZE;
    vlm_b = max_possible_loopcnt;
    normal_blk_loopcnt = CeilDiv(n, WARP_SIZE * blk_cnt);
    block_data_stride = normal_blk_loopcnt * WARP_SIZE * dwidth;  // SPM: not shifted

    int64_t remain_data = n - normal_blk_loopcnt * WARP_SIZE * (blk_cnt - 1);
    last_blk_loopcnt = remain_data / WARP_SIZE;
    rmd_threads = remain_data % WARP_SIZE;

    for (int i = 0; i < WARP_SIZE; i++) {
      thread_params[i] = (uint16_t)last_blk_loopcnt;
      if (i < rmd_threads) {
        thread_params[i] += 1;
      }
    }
  }

  // Get SPM kernel (no _DDR suffix)
  Kernel_t* kernel;
  if (is_bopa) {
    kernel = C256
        ? GET_KERNEL(KernelId::BINARY_NX1_NXC256_BATCH_BOPA)
        : GET_KERNEL(KernelId::BINARY_NX1_NXC_V16_BATCH_BOPA);
  } else {
    kernel = C256
        ? GET_KERNEL(KernelId::BINARY_NX1_NXC256_BATCH)
        : GET_KERNEL(KernelId::BINARY_NX1_NXC_V16_BATCH);
  }
  TORCH_CHECK(kernel != nullptr, "Failed to get binary_Nx1_NxC SPM kernel from cache");
  kernel->reset_regs();

  auto* wq = GET_QUEUE(num_cores);

  // SPM addresses (32-bit, NOT shifted)
  kernel->set_regs(0, (uint16_t)normal_blk_loopcnt);
  if (C256) {
    kernel->set_regs(2, (uint16_t)last_blk_loopcnt);
  }
  kernel->set_regs(4, (uint16_t)(a_spm_addr & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(a_spm_addr >> 16));
  kernel->set_regs(6, (uint16_t)(b_spm_addr & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(b_spm_addr >> 16));
  kernel->set_regs(8, (uint16_t)(output_spm_addr & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(output_spm_addr >> 16));
  kernel->set_regs(10, (uint16_t)(block_data_stride & 0xFFFF));
  kernel->set_regs(11, (uint16_t)(block_data_stride >> 16));
  kernel->set_regs(12, (uint16_t)c_loopcnt);
  kernel->set_regs(13, (uint16_t)vlm_b);

  // Populate the required SCM parameters.
  if (!C256) {
    for (int i = 0; i < WARP_SIZE; i++) {
      kernel->set_regs((uint32_t)i + 4096, thread_params[i]);
    }
  }

  c10::Half scale_a = 1.0;
  c10::Half scale_b = alpha_f;
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;
  kernel->set_regs(56, scale_a.x);
  kernel->set_regs(57, scale_b.x);
  kernel->set_regs(58, scale_r.x);
  kernel->set_regs(59, clip_max.x);
  kernel->set_regs(60, clip_min.x);
  kernel->set_regs(61, (uint16_t)0x1);  // SPM mode flag

  const auto& op_info = get_valu_op_info(op_type);
  kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
  kernel->set_regs(63, (uint16_t)(op_info.custom_op));

  // Broadcast the same launch over each core's SPM slice.
  wq->set_broadcast_mode(true);
  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1},
                    core_list);
}

// ==================== 1xC_NxC SPM Kernel ====================
// input_a is (1, C), input_b is (N, C), output is (N, C)
// When is_bopa=false: computes A op B (small op large)
// When is_bopa=true:  computes B op A (large op small), used for non-commutative ops
// Supports in-place: output_spm_addr can be same as b_spm_addr
//
// Dispatch contract: the v16/v256 "batch" kernels below iterate C in whole
// v16 vectors (c_loopcnt = c/16 or c/256), so
// they REQUIRE c%16==0 and SILENTLY NO-OP when c<16 (c_loopcnt truncates to 0,
// leaving in-place output == input). For c not v16-aligned, or c>1024, or
// n>8192, route to the tile-based V2 kernel, which passes C
// as a RAW element count and tiles only over N. GDN prep's dt_bias +/ neg_exp_A
// EWa is [1,vg_c] op [L,vg_c] with vg_c=4 -> hits this V2 path.
void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
    uint32_t a_spm_addr,               // SPM input A address [1, C]
    uint32_t b_spm_addr,               // SPM input B address [N, C]
    uint32_t output_spm_addr,          // SPM output address [N, C]
    int64_t n,                         // N dimension
    int64_t c,                         // C dimension
    c10::Half alpha_f,
    ValuOpType op_type,
    bool is_bopa)
{
  // ---- V2 tile-based path (C not v16-aligned, or C>1024, or N>8192) ----
  // V2 tile-based path register contract:
  // reg0-1 = n (u32), reg2-3 = c (u32), reg10-11 = blk_stride (u32),
  // reg12 = normal_blk_n, reg13 = last_blk_n (tileN only), reg4-9 = a/b/out addrs,
  // reg56-63 = eltwise_common_init. tile_c = c>=256 -> tileC (tile over C), else
  // tileN (tile over N). a is always the [1,C] operand, b the [N,C] (is_bopa only
  // selects the op-order kernel variant, matching the v16/v256 branch below).
  if (c % 16 != 0 || c > 1024 || n > 8192) {
    const bool tile_c = (c >= 256);
    std::string kname = std::string("binary_1xC_NxC_") +
                        (tile_c ? "tileC" : "tileN") + (is_bopa ? "_bopa" : "");
    // Graph-aware fetch (NOT raw KernelCache::get_kernel) so the launch records
    // into the fused graph. See [[spm-kernel-graph-recordable-fetch]].
    Kernel_t* kernel = RpuKernelGraph::active().get_kernel_reset(kname);
    TORCH_CHECK(kernel != nullptr, "Failed to get ", kname, " kernel");

    int64_t blk_cnt;
    if (tile_c) {
      blk_cnt = CeilDiv(c, (int64_t)256);
      const uint32_t blk_stride = (uint32_t)(256 * sizeof(c10::Half));
      kernel->set_regs(10, (uint16_t)(blk_stride & 0xFFFF));
      kernel->set_regs(11, (uint16_t)(blk_stride >> 16));
    } else {
      blk_cnt = std::min<int64_t>(CeilDiv(n, (int64_t)256), 32);
      const int64_t normal_blk_n = n / blk_cnt;
      const int64_t last_blk_n   = n - (blk_cnt - 1) * normal_blk_n;
      const uint32_t blk_stride  = (uint32_t)(normal_blk_n * c * sizeof(c10::Half));
      kernel->set_regs(10, (uint16_t)(blk_stride & 0xFFFF));
      kernel->set_regs(11, (uint16_t)(blk_stride >> 16));
      kernel->set_regs(12, (uint16_t)normal_blk_n);
      kernel->set_regs(13, (uint16_t)last_blk_n);
    }
    kernel->set_regs(0, (uint16_t)(n & 0xFFFF));
    kernel->set_regs(1, (uint16_t)((n >> 16) & 0xFFFF));
    kernel->set_regs(2, (uint16_t)(c & 0xFFFF));
    kernel->set_regs(3, (uint16_t)((c >> 16) & 0xFFFF));
    kernel->set_regs(4, (uint16_t)(a_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(a_spm_addr >> 16));
    kernel->set_regs(6, (uint16_t)(b_spm_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(b_spm_addr >> 16));
    kernel->set_regs(8, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(output_spm_addr >> 16));
    // Eltwise common registers 56-63. The key difference from the Bx1xC/Nx1/v16 launchers is
    // that reg59/60 are imm3/imm4 = 1.0, whereas those launchers
    // set them to clip_max/clip_min = 0.0. The tile kernels read imm3/imm4 in their VALU config;
    // feeding 0.0 makes the MUL produce garbage (decay tril-mul: ±691 instead of the masked diff).
    c10::Half scale_a = 1.0, scale_b = alpha_f, scale_r = 1.0, imm3 = 1.0, imm4 = 1.0;
    kernel->set_regs(56, scale_a.x);
    kernel->set_regs(57, scale_b.x);
    kernel->set_regs(58, scale_r.x);
    kernel->set_regs(59, imm3.x);
    kernel->set_regs(60, imm4.x);
    kernel->set_regs(61, (uint16_t)0x1);
    const auto& op_info = get_valu_op_info(op_type);
    kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
    kernel->set_regs(63, (uint16_t)(op_info.custom_op));
    // reg64-66 = grid dims (param[64]=blk_cnt in eltwise_common_init / COMMON_ELTWISE_BINARY_INIT).
    // The V2 tile kernels READ blk_cnt from reg64 (NOT just the enqueu grid); leaving it 0 runs
    // ZERO blocks → the op silently no-ops. This is what broke the decay tril-mul (tileC, blk_cnt=16
    // → upper triangle unmasked → exp overflow). Mirror the Bx1xC launcher which sets these.
    kernel->set_regs(64, (uint16_t)blk_cnt);
    kernel->set_regs(65, (uint16_t)1);
    kernel->set_regs(66, (uint16_t)1);

    auto* wq = GET_QUEUE(NUM_CORES_BINARY);
    wq->set_broadcast_mode(true);
    wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1},
                      {0, 1, 2, 3, 4, 5, 6, 7});
    return;
  }

  size_t dwidth = sizeof(c10::Half);

  bool C256 = (c % 256 == 0);

  size_t c_loopcnt, vlm_b, normal_blk_loopcnt, blk_cnt, last_blk_loopcnt;
  uint32_t block_data_stride;

  // Prepare the required SCM parameters.
  uint16_t thread_params[WARP_SIZE] = {0};
  int64_t rmd_threads = 0;

  blk_cnt = CeilDiv(n * c, (VLM_ENTRIES - 2 * CeilDiv(c, THD_VECTOR_SIZE)) * WARP_VECTOR_SIZE);

  if (C256) {
    c_loopcnt = c / WARP_VECTOR_SIZE;
    vlm_b = c_loopcnt;
    normal_blk_loopcnt = CeilDiv(n, blk_cnt);
    last_blk_loopcnt = n - normal_blk_loopcnt * (blk_cnt - 1);
    block_data_stride = normal_blk_loopcnt * dwidth;  // SPM: not shifted
  } else {
    c_loopcnt = c / THD_VECTOR_SIZE;
    vlm_b = c_loopcnt;
    normal_blk_loopcnt = CeilDiv(n, WARP_SIZE * blk_cnt);
    block_data_stride = normal_blk_loopcnt * WARP_SIZE * dwidth;  // SPM: not shifted

    int64_t remain_data = n - normal_blk_loopcnt * WARP_SIZE * (blk_cnt - 1);
    last_blk_loopcnt = remain_data / WARP_SIZE;
    rmd_threads = remain_data % WARP_SIZE;

    for (int i = 0; i < WARP_SIZE; i++) {
      thread_params[i] = (uint16_t)last_blk_loopcnt;
      if (i < rmd_threads) {
        thread_params[i] += 1;
      }
    }
  }

  // Determine kernel ID
  KernelId kid;
  if (is_bopa) {
    kid = C256 ? KernelId::BINARY_1XC_NXC_V256_BATCH_BOPA
               : KernelId::BINARY_1XC_NXC_V16_BATCH_BOPA;
  } else {
    kid = C256 ? KernelId::BINARY_1XC_NXC_V256_BATCH
               : KernelId::BINARY_1XC_NXC_V16_BATCH;
  }

  c10::Half scale_a = 1.0;
  c10::Half scale_b = alpha_f;
  c10::Half scale_r = 1.0;
  c10::Half clip_min = 0.0;
  c10::Half clip_max = 0.0;
  const auto& op_info = get_valu_op_info(op_type);

  // Copy the SCM values for lambda capture.
  std::array<uint16_t, 32> tp_arr{};
  if (!C256) {
    for (int i = 0; i < WARP_SIZE; i++) tp_arr[i] = thread_params[i];
  }

  auto setup_regs = [=](Kernel_t* kernel) {
    kernel->set_regs(0, (uint16_t)normal_blk_loopcnt);
    if (C256) {
      kernel->set_regs(2, (uint16_t)last_blk_loopcnt);
    }
    kernel->set_regs(4, (uint16_t)(a_spm_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(a_spm_addr >> 16));
    kernel->set_regs(6, (uint16_t)(b_spm_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(b_spm_addr >> 16));
    kernel->set_regs(8, (uint16_t)(output_spm_addr & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(output_spm_addr >> 16));
    kernel->set_regs(10, (uint16_t)(block_data_stride & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(block_data_stride >> 16));
    kernel->set_regs(12, (uint16_t)c_loopcnt);
    kernel->set_regs(13, (uint16_t)vlm_b);
    if (!C256) {
      for (int i = 0; i < WARP_SIZE; i++) {
        kernel->set_regs((uint32_t)i + 4096, tp_arr[i]);
      }
    }
    kernel->set_regs(56, scale_a.x);
    kernel->set_regs(57, scale_b.x);
    kernel->set_regs(58, scale_r.x);
    kernel->set_regs(59, clip_max.x);
    kernel->set_regs(60, clip_min.x);
    kernel->set_regs(61, (uint16_t)0x1);
    kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
    kernel->set_regs(63, (uint16_t)(op_info.custom_op));
  };

  Kernel_t* kernel = GET_KERNEL(kid);
  TORCH_CHECK(kernel != nullptr, "Failed to get binary_1xC_NxC SPM kernel");
  kernel->reset_regs();
  setup_regs(kernel);

  auto* wq = GET_QUEUE(NUM_CORES_BINARY);
  wq->set_broadcast_mode(true);
  wq->enqueu_kernel(*kernel, {(uint16_t)blk_cnt, (uint16_t)1, (uint16_t)1},
                    {0, 1, 2, 3, 4, 5, 6, 7});
}

// ==================== Bx1xC_BxNxC middle-broadcast SPM Kernel ====================
// A = [B, 1, C] broadcasts across the MIDDLE axis against B_in = [B, N, C]:
//   out[g, i, j] = A[g, 0, j] op B_in[g, i, j]   (the i-axis is broadcast)
// is_bopa=false → A op B_in ; is_bopa=true → B_in op A (for non-commutative ops).
// This is the one broadcast shape the Nx1_NxC / 1xC_NxC launchers cannot express
// (a "1" sandwiched in a non-leading, non-last axis). Leading dimensions are
// folded into the batch before launch.
void rpu_launch_eltwise_binary_Bx1xC_BxNxC_spm_kernel(
    uint32_t a_spm_addr,       // [B, 1, C]
    uint32_t b_spm_addr,       // [B, N, C]
    uint32_t output_spm_addr,  // [B, N, C]
    int64_t bsz, int64_t n, int64_t c,
    c10::Half alpha_f, ValuOpType op_type, bool is_bopa)
{
  // tileN is valid for c<=1024; tileC is valid for n<=1024.
  const bool use_tileN = (c <= 1024);
  TORCH_CHECK(use_tileN || n <= 1024,
              "Bx1xC: need c<=1024 (tileN) or n<=1024 (tileC); got n=", n, " c=", c);
  std::string kname = std::string("binary_Bx1xC_BxNxC_") +
                      (use_tileN ? "tileN" : "tileC") + (is_bopa ? "_bopa" : "");
  // Graph-aware fetch (NOT raw KernelCache::get_kernel) so the launch records into the
  // fused graph; the raw fetch breaks recording -> "kernel launched without KernelId
  // or kernel_name path" -> PASSTHROUGH fallback. See [[spm-kernel-graph-recordable-fetch]].
  Kernel_t* kernel = RpuKernelGraph::active().get_kernel_reset(kname);
  TORCH_CHECK(kernel != nullptr, "Failed to get ", kname, " kernel");

  int64_t blk_cnt_n, blk_cnt_c;
  if (use_tileN) {
    const int64_t normal_blk_n = 1024;
    blk_cnt_n = CeilDiv(n, normal_blk_n);
    blk_cnt_c = 1;
    const int64_t last_blk_n = n - normal_blk_n * (blk_cnt_n - 1);
    kernel->set_regs(10, (uint16_t)normal_blk_n);
    kernel->set_regs(11, (uint16_t)last_blk_n);
  } else {  // tileC
    blk_cnt_n = 1;
    blk_cnt_c = CeilDiv(c, 256);
  }

  kernel->set_regs(0, (uint16_t)(n & 0xFFFF));
  kernel->set_regs(1, (uint16_t)((n >> 16) & 0xFFFF));
  kernel->set_regs(2, (uint16_t)(c & 0xFFFF));
  kernel->set_regs(3, (uint16_t)((c >> 16) & 0xFFFF));
  kernel->set_regs(4, (uint16_t)(a_spm_addr & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(a_spm_addr >> 16));
  kernel->set_regs(6, (uint16_t)(b_spm_addr & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(b_spm_addr >> 16));
  kernel->set_regs(8, (uint16_t)(output_spm_addr & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(output_spm_addr >> 16));

  // eltwise common (mirror the runtime COMMON_ELTWISE_BINARY_INIT macro).
  // The tile kernels consume regs 59/60 as imm3/imm4, whose neutral runtime
  // values are 1.0; treating them as zero-valued clip bounds corrupts MUL.
  c10::Half scale_a = 1.0, scale_b = alpha_f, scale_r = 1.0;
  c10::Half imm3 = 1.0, imm4 = 1.0;
  kernel->set_regs(56, scale_a.x);
  kernel->set_regs(57, scale_b.x);
  kernel->set_regs(58, scale_r.x);
  kernel->set_regs(59, imm3.x);
  kernel->set_regs(60, imm4.x);
  kernel->set_regs(61, (uint16_t)0x1);  // SPM / 2-byte element flag
  const auto& op_info = get_valu_op_info(op_type);
  kernel->set_regs(62, (uint16_t)((op_info.op << 9) | FP16_RAB_DATA_TYPE_CFG));
  kernel->set_regs(63, (uint16_t)(op_info.custom_op));

  kernel->set_regs(64, (uint16_t)bsz);        // gridDim.x = B (batch)
  kernel->set_regs(65, (uint16_t)blk_cnt_n);  // gridDim.y
  kernel->set_regs(66, (uint16_t)blk_cnt_c);  // gridDim.z

  auto* wq = GET_QUEUE(NUM_CORES_BINARY);
  wq->set_broadcast_mode(true);
  wq->enqueu_kernel(*kernel, {(uint16_t)bsz, (uint16_t)blk_cnt_n, (uint16_t)blk_cnt_c},
                    {0, 1, 2, 3, 4, 5, 6, 7});
}
// ==================== Convenience Wrappers ====================

// Convenience wrapper for element-wise multiply (used in MLP: gate * up)
void rpu_launch_eltwise_mul_spm_kernel(
    uint32_t a_spm_addr,               // First operand SPM address
    uint32_t b_spm_addr,               // Second operand SPM address
    uint32_t output_spm_addr,          // Output SPM address
    int64_t num_elements)              // Total elements per core
{
  rpu_launch_eltwise_binary_spm_kernel(
      a_spm_addr, b_spm_addr, output_spm_addr,
      num_elements, ValuOpType::MUL, c10::Half(1.0));
}

// Convenience wrapper for element-wise add (used in residual connections)
void rpu_launch_eltwise_add_spm_kernel(
    uint32_t a_spm_addr,               // First operand SPM address
    uint32_t b_spm_addr,               // Second operand SPM address
    uint32_t output_spm_addr,          // Output SPM address
    int64_t num_elements,              // Total elements per core
    c10::Half alpha)                   // Scale for B: out = A + alpha * B
{
  rpu_launch_eltwise_binary_spm_kernel(
      a_spm_addr, b_spm_addr, output_spm_addr,
      num_elements, ValuOpType::ADD, alpha);
}

// Convenience wrapper for element-wise sub
void rpu_launch_eltwise_sub_spm_kernel(
    uint32_t a_spm_addr,
    uint32_t b_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_elements,
    c10::Half alpha)
{
  rpu_launch_eltwise_binary_spm_kernel(
      a_spm_addr, b_spm_addr, output_spm_addr,
      num_elements, ValuOpType::SUB, alpha);
}

// Convenience wrapper for element-wise div
void rpu_launch_eltwise_div_spm_kernel(
    uint32_t a_spm_addr,
    uint32_t b_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_elements)
{
  rpu_launch_eltwise_binary_spm_kernel(
      a_spm_addr, b_spm_addr, output_spm_addr,
      num_elements, ValuOpType::DIV, c10::Half(1.0));
}

// Convenience wrapper for element-wise pow
void rpu_launch_eltwise_pow_spm_kernel(
    uint32_t a_spm_addr,
    uint32_t b_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_elements)
{
  rpu_launch_eltwise_binary_spm_kernel(
      a_spm_addr, b_spm_addr, output_spm_addr,
      num_elements, ValuOpType::POW, c10::Half(1.0));
}
