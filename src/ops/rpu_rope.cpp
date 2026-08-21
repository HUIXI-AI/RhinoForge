#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h"
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ::rhino_lkn;

// Prefill RoPE kernel launch function (DDR version)
// Uses llama_rope_ddr kernel with grid.x = seq_len for parallel processing
// All data in DDR: x, cos, sin, y
// Kernel computes x_addr = x_base + blkid.x * heads_num * head_dim * 2
// and position = param0 + blkid.x
//
// This version takes raw pointers and dimensions directly to avoid tensor ops
static void rpu_launch_rope_kernel_prefill_raw(
    c10::Half *x_ptr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    c10::Half *out_ptr,
    int64_t seq_len,
    int64_t heads_num,
    int64_t head_dim,
    int64_t start_pos
) {

  // Get kernel via KernelId (O(1) access)
  Kernel_t* kernel = GET_KERNEL(KernelId::ROPE_DDR);
  TORCH_CHECK(kernel != nullptr, "Failed to get llama_rope_ddr kernel");
  kernel->reset_regs();

  auto* wq = GET_QUEUE(1);
  wq->set_broadcast_mode(true);  // required for the multi-block launch

  // Flush DDR inputs
  rpu_ddr_flush(x_ptr);
  rpu_ddr_flush(cos_ptr);
  rpu_ddr_flush(sin_ptr);
  rpu_ddr_flush(out_ptr);

  // DDR addresses (glarge format = address / 256)
  uint64_t x_addr = RpuGetDevAddr(x_ptr) >> 8;
  uint64_t y_addr = RpuGetDevAddr(out_ptr) >> 8;
  uint64_t cos_addr = RpuGetDevAddr(cos_ptr) >> 8;
  uint64_t sin_addr = RpuGetDevAddr(sin_ptr) >> 8;

  // Set kernel parameters
  // param0/1: starting position (kernel adds blkid.x internally)
  kernel->set_regs(0, (uint16_t)(start_pos & 0xFFFF));
  kernel->set_regs(1, (uint16_t)(start_pos >> 16));

  // param2: head_dim
  kernel->set_regs(2, (uint16_t)head_dim);

  // param3: heads_num
  kernel->set_regs(3, (uint16_t)heads_num);

  // param4/5: x DDR base address (glarge)
  kernel->set_regs(4, (uint16_t)(x_addr & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(x_addr >> 16));

  // param6/7: cos DDR address (glarge)
  kernel->set_regs(6, (uint16_t)(cos_addr & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(cos_addr >> 16));

  // param8/9: sin DDR address (glarge)
  kernel->set_regs(8, (uint16_t)(sin_addr & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(sin_addr >> 16));

  // param10/11: y DDR base address (glarge)
  kernel->set_regs(10, (uint16_t)(y_addr & 0xFFFF));
  kernel->set_regs(11, (uint16_t)(y_addr >> 16));

  // Launch kernel: grid = [seq_len, 1, 1]
  // Each block processes one position's worth of heads
  wq->enqueu_kernel(*kernel, {(uint16_t)seq_len, (uint16_t)1, (uint16_t)1}, {0});

  // Flush output
  rpu_ddr_flush(out_ptr);
}

// Fallback: use PyTorch element-wise ops
// Formula: y = (x * cos) + (rotate_half(x) * sin)
static std::tuple<at::Tensor, at::Tensor> rpu_apply_rotary_pos_emb_fallback(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &cos,
    const at::Tensor &sin
) {
  int64_t head_dim = query.size(-1);

  // rotate_half: cat(-x[..., head_dim/2:], x[..., :head_dim/2])
  auto rotate_half = [](const at::Tensor& x) {
    auto x1 = x.slice(-1, 0, x.size(-1) / 2);
    auto x2 = x.slice(-1, x.size(-1) / 2);
    return at::cat({-x2, x1}, -1);
  };

  // Handle cos/sin format conversion
  at::Tensor cos_full = cos;
  at::Tensor sin_full = sin;

  // If kernel format [seq_len, head_dim//2], expand to [seq_len, head_dim]
  if (cos.size(-1) == head_dim / 2) {
    cos_full = at::cat({cos, cos}, -1);
    sin_full = at::cat({sin, sin}, -1);
  }

  // Broadcast cos/sin to match q/k shape: [batch, heads, seq_len, head_dim]
  at::Tensor cos_b = cos_full;
  at::Tensor sin_b = sin_full;

  if (cos_full.dim() == 2) {
    // [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
    cos_b = cos_full.unsqueeze(0).unsqueeze(0);
    sin_b = sin_full.unsqueeze(0).unsqueeze(0);
  } else if (cos_full.dim() == 3) {
    // [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    cos_b = cos_full.unsqueeze(1);
    sin_b = sin_full.unsqueeze(1);
  }

  // y = (x * cos) + (rotate_half(x) * sin)
  auto q_out = (query * cos_b) + (rotate_half(query) * sin_b);
  auto k_out = (key * cos_b) + (rotate_half(key) * sin_b);

  return std::make_tuple(q_out, k_out);
}

// High-level RoPE interface for RPU unified layout
// Uses llama_rope kernel with grid.x = seq_len for prefill
// cos/sin must be in [max_seq_len, head_dim//2] format for kernel path
// Formula: y = (x * cos) + (rotate_half(x) * sin)
//
// Input layout: [batch, seq_len, heads, head_dim] (RPU unified layout)
// Note: This is different from PyTorch standard [batch, heads, seq_len, head_dim]
std::tuple<at::Tensor, at::Tensor> rpu_apply_rotary_pos_emb(
    const at::Tensor &query,     // [batch, seq_len, heads, head_dim] (RPU unified)
    const at::Tensor &key,       // [batch, seq_len, heads, head_dim] (RPU unified)
    const at::Tensor &cos,       // [seq_len, head_dim] or [max_seq, head_dim//2]
    const at::Tensor &sin,       // [seq_len, head_dim] or [max_seq, head_dim//2]
    const at::Tensor &position_ids  // [batch, seq_len] (unused, for API compatibility)
) {
  RECORD_FUNCTION("rpu::apply_rotary_pos_emb", {query, key, cos, sin});

  TORCH_CHECK(query.device().type() == c10::DeviceType::PrivateUse1,
              "rpu_apply_rotary_pos_emb: query must be on RPU");
  TORCH_CHECK(query.dim() == 4, "rpu_apply_rotary_pos_emb expects 4D input [batch, seq_len, heads, head_dim]");

  // Input layout: [batch, heads, seq_len, head_dim] (PyTorch standard)
  int64_t batch = query.size(0);
  int64_t q_heads = query.size(2);  // num_heads (e.g. 16)
  int64_t k_heads = key.size(2);    // num_kv_heads (e.g. 8 for GQA)
  int64_t seq_len = query.size(1);
  int64_t head_dim = query.size(3);

  // Currently only batch=1 is supported for kernel path
  if (batch != 1) {
    if (log_at(2)) std::cerr << "[RoPE] Fallback: batch=" << batch << " != 1" << std::endl;
    auto q_std = query.transpose(1, 2).contiguous();
    auto k_std = key.transpose(1, 2).contiguous();
    auto [q_out_std, k_out_std] = rpu_apply_rotary_pos_emb_fallback(q_std, k_std, cos, sin);
    return std::make_tuple(q_out_std.transpose(1, 2).contiguous(),
                           k_out_std.transpose(1, 2).contiguous());
  }

  // Get cos/sin data pointers - need to handle format conversion
  // Qwen3: [batch?, seq_len, head_dim] (duplicated: cos[...,:d/2] == cos[...,d/2:])
  // Kernel expects: [seq_len, head_dim//2] contiguous
  c10::Half *cos_ptr, *sin_ptr;
  at::Tensor cos_buf, sin_buf;  // Hold reference if we create new tensors

  if (cos.size(-1) == head_dim) {
    // Expanded format: slice creates non-contiguous view, must call contiguous()
    cos_buf = cos.slice(-1, 0, head_dim / 2).contiguous();
    sin_buf = sin.slice(-1, 0, head_dim / 2).contiguous();
    cos_ptr = cos_buf.data_ptr<c10::Half>();
    sin_ptr = sin_buf.data_ptr<c10::Half>();
  } else if (cos.size(-1) == head_dim / 2) {
    // Already in kernel format
    if (cos.is_contiguous()) {
      cos_ptr = const_cast<c10::Half*>(cos.data_ptr<c10::Half>());
      sin_ptr = const_cast<c10::Half*>(sin.data_ptr<c10::Half>());
    } else {
      cos_buf = cos.contiguous();
      sin_buf = sin.contiguous();
      cos_ptr = cos_buf.data_ptr<c10::Half>();
      sin_ptr = sin_buf.data_ptr<c10::Half>();
    }
  } else {
    if (log_at(2)) std::cerr << "[RoPE] Fallback: unexpected cos size " << cos.size(-1) << std::endl;
    auto q_std = query.transpose(1, 2).contiguous();
    auto k_std = key.transpose(1, 2).contiguous();
    auto [q_out_std, k_out_std] = rpu_apply_rotary_pos_emb_fallback(q_std, k_std, cos, sin);
    return std::make_tuple(q_out_std.transpose(1, 2).contiguous(),
                           k_out_std.transpose(1, 2).contiguous());
  }

  auto q_contig = query.contiguous();
  auto k_contig = key.contiguous(); 

  // query/key: [1, seq_len, heads, head_dim] contiguous has same memory layout as [seq_len, heads, head_dim]
  // No view/squeeze needed - just use data pointers directly
  TORCH_CHECK(q_contig.is_contiguous() && k_contig.is_contiguous(),
              "rpu_apply_rotary_pos_emb: query and key must be contiguous");
         
  c10::Half *q_ptr = const_cast<c10::Half*>(q_contig.data_ptr<c10::Half>());
  c10::Half *k_ptr = const_cast<c10::Half*>(k_contig.data_ptr<c10::Half>());

  // Allocate output with same shape as input [1, seq_len, heads, head_dim]
  auto q_out = at::empty_like(q_contig);
  auto k_out = at::empty_like(k_contig);

  c10::Half *q_out_ptr = q_out.data_ptr<c10::Half>();
  c10::Half *k_out_ptr = k_out.data_ptr<c10::Half>();

  // Derive start_pos from position_ids for cos/sin row selection.
  int64_t start_pos = 0;
  if (position_ids.defined() && position_ids.numel() > 0) {
    auto pids_cpu = position_ids.to(at::kCPU).contiguous().to(at::kLong);
    start_pos = pids_cpu.data_ptr<int64_t>()[0];
  }

  // DDR RoPE operator uses this corrected stride contract
  // (wr10 = head_dim×heads_num/128 in 256B-units, matches param4 DDR-addr-in-256B-units).
  rpu_launch_rope_kernel_prefill_raw(q_ptr, cos_ptr, sin_ptr, q_out_ptr, seq_len, q_heads, head_dim, start_pos);
  rpu_launch_rope_kernel_prefill_raw(k_ptr, cos_ptr, sin_ptr, k_out_ptr, seq_len, k_heads, head_dim, start_pos);

  return std::make_tuple(q_out, k_out);
}

// ============================================================================
// SPM版本 RoPE Kernel (8核，用于融合执行)
// ============================================================================
// Input 布局和 Linear SPM output 一致：每个 core i 存储 [seq_len, local_heads, head_dim]
// 按 heads 维度切分到 8 个 core
// cos/sin 在 DDR
// ============================================================================

#define ROPE_NUM_CORES 8

void rpu_launch_rope_spm_kernel(
    c10::Half *x_ptr,       // DDR input [seq_len, heads_num, head_dim]
    c10::Half *cos_ptr,     // DDR cos table
    c10::Half *sin_ptr,     // DDR sin table
    c10::Half *out_ptr,     // DDR output
    int64_t seq_len,
    int64_t heads_num,
    int64_t head_dim,
    int64_t start_pos
) {
  // 8核布局：每个 core i 存储 [seq_len, local_heads, head_dim]
  // 其中 local_heads = heads_num / 8

  TORCH_CHECK(heads_num % ROPE_NUM_CORES == 0,
              "RoPE SPM: heads_num must be divisible by 8");

  size_t dwidth = sizeof(c10::Half);
  int64_t local_heads = heads_num / ROPE_NUM_CORES;
  size_t local_x_bytes = seq_len * local_heads * head_dim * dwidth;

  // Get SPM kernel (llama_rope)
  Kernel_t* kernel = GET_KERNEL(KernelId::ROPE);
  TORCH_CHECK(kernel != nullptr, "Failed to get llama_rope kernel (SPM version)");

  auto* wq = GET_QUEUE(ROPE_NUM_CORES);

  // ===== 分配 SPM (8核) =====
  std::vector<LocalSPM_t*> spm_x(ROPE_NUM_CORES);
  std::vector<LocalSPM_t*> spm_y(ROPE_NUM_CORES);

  for (int i = 0; i < ROPE_NUM_CORES; ++i) {
    spm_x[i] = graph_acquire_local_spm(Align(local_x_bytes, SPM_BANK_SIZE), read_write, kStride32B, 2, i);
    spm_y[i] = graph_acquire_local_spm(Align(local_x_bytes, SPM_BANK_SIZE), read_write, kStride32B, 2, i);
  }

  // ===== DDR -> SPM 拷贝 =====
  rpu_ddr_flush(x_ptr);
  rpu_ddr_flush(cos_ptr);
  rpu_ddr_flush(sin_ptr);

  // Input: 每个 core 拿 [seq_len, local_heads, head_dim]
  // DDR layout: [seq_len, heads_num, head_dim]
  // 每个 core i 拿 heads [i*local_heads, (i+1)*local_heads)
  size_t head_stride = head_dim * dwidth;
  size_t row_stride = heads_num * head_dim * dwidth;
  size_t local_row_stride = local_heads * head_dim * dwidth;

  for (int i = 0; i < ROPE_NUM_CORES; ++i) {
    c10::Half* spm_x_ptr = static_cast<c10::Half*>(spm_x[i]->get_cpu_ptr());
    for (int64_t s = 0; s < seq_len; ++s) {
      // 每个 seq position 拷贝 local_heads 个 head
      memcpy(spm_x_ptr + s * local_heads * head_dim,
             x_ptr + s * heads_num * head_dim + i * local_heads * head_dim,
             local_row_stride);
    }
  }

  // ===== Kernel 配置与执行 =====
  kernel->reset_regs();

  // 使用 core 0 的地址作为基准
  uint32_t x_addr = spm_x[0]->get_rpu_addr();
  uint32_t y_addr = spm_y[0]->get_rpu_addr();
  // DDR addresses (64-bit >> 8) for cos/sin
  uint64_t cos_addr = RpuGetDevAddr(cos_ptr) >> 8;
  uint64_t sin_addr = RpuGetDevAddr(sin_ptr) >> 8;

  // param0/1: starting position
  kernel->set_regs(0, (uint16_t)(start_pos & 0xFFFF));
  kernel->set_regs(1, (uint16_t)(start_pos >> 16));

  // param2: head_dim
  kernel->set_regs(2, (uint16_t)head_dim);

  // param3: local_heads (每个 core 处理的 heads 数)
  kernel->set_regs(3, (uint16_t)local_heads);

  // param4/5: x SPM address (32-bit)
  kernel->set_regs(4, (uint16_t)(x_addr & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(x_addr >> 16));

  // param6/7: cos DDR address (>> 8)
  kernel->set_regs(6, (uint16_t)(cos_addr & 0xFFFF));
  kernel->set_regs(7, (uint16_t)(cos_addr >> 16));

  // param8/9: sin DDR address (>> 8)
  kernel->set_regs(8, (uint16_t)(sin_addr & 0xFFFF));
  kernel->set_regs(9, (uint16_t)(sin_addr >> 16));

  // param10/11: y SPM address (32-bit)
  kernel->set_regs(10, (uint16_t)(y_addr & 0xFFFF));
  kernel->set_regs(11, (uint16_t)(y_addr >> 16));

  // Launch kernel: grid = [seq_len, 1, 1], 8 cores
  wq->set_broadcast_mode(true);
  wq->enqueu_kernel(*kernel, {(uint16_t)seq_len, (uint16_t)1, (uint16_t)1}, {0, 1, 2, 3, 4, 5, 6, 7});

  // ===== SPM → DDR 拷贝 (用于测试验证) =====
  for (int i = 0; i < ROPE_NUM_CORES; ++i) {
    c10::Half* spm_y_ptr = static_cast<c10::Half*>(spm_y[i]->get_cpu_ptr());
    for (int64_t s = 0; s < seq_len; ++s) {
      memcpy(out_ptr + s * heads_num * head_dim + i * local_heads * head_dim,
             spm_y_ptr + s * local_heads * head_dim,
             local_row_stride);
    }
  }
  rpu_ddr_flush(out_ptr);

  // ===== 清理 =====
  for (int i = 0; i < ROPE_NUM_CORES; ++i) {
    graph_release_local_spm(spm_x[i]);
    graph_release_local_spm(spm_y[i]);
  }
}

// ============================================================================
// Direct Address SPM RoPE Kernel
// ============================================================================
// Same as rpu_launch_rope_spm_unified_kernel but accepts SPM addresses directly
// instead of using SpmAddressCache with char selectors.
// ============================================================================

void rpu_launch_rope_spm_kernel(
    uint32_t x_spm_addr,             // Input SPM address (core 0 base)
    uint32_t y_spm_addr,             // Output SPM address (core 0 base, can differ from x)
    c10::Half *cos_ptr,              // DDR cos table [max_seq, head_dim/2]
    c10::Half *sin_ptr,              // DDR sin table [max_seq, head_dim/2]
    int64_t seq_len,
    int64_t local_heads,             // Number of heads per core
    int64_t head_dim,
    int64_t start_pos,
    int num_cores)                   // number of cores (attn_tp)
{
  // Flush DDR cos/sin
  rpu_ddr_flush(cos_ptr);
  rpu_ddr_flush(sin_ptr);

  // DDR addresses (64-bit >> 8) for cos/sin
  uint64_t cos_addr = RpuGetDevAddr(cos_ptr) >> 8;
  uint64_t sin_addr = RpuGetDevAddr(sin_ptr) >> 8;

  uint32_t x_addr = x_spm_addr;
  uint32_t y_addr = y_spm_addr;

  auto setup_regs = [=](Kernel_t* kernel) {
    // param0/1: starting position
    kernel->set_regs(0, (uint16_t)(start_pos & 0xFFFF));
    kernel->set_regs(1, (uint16_t)(start_pos >> 16));

    // param2: head_dim
    kernel->set_regs(2, (uint16_t)head_dim);

    // param3: local_heads (每个 core 处理的 heads 数)
    kernel->set_regs(3, (uint16_t)local_heads);

    // param4/5: x SPM address (32-bit)
    kernel->set_regs(4, (uint16_t)(x_addr & 0xFFFF));
    kernel->set_regs(5, (uint16_t)(x_addr >> 16));

    // param6/7: cos DDR address (>> 8)
    kernel->set_regs(6, (uint16_t)(cos_addr & 0xFFFF));
    kernel->set_regs(7, (uint16_t)(cos_addr >> 16));

    // param8/9: sin DDR address (>> 8)
    kernel->set_regs(8, (uint16_t)(sin_addr & 0xFFFF));
    kernel->set_regs(9, (uint16_t)(sin_addr >> 16));

    // param10/11: y SPM address (32-bit)
    kernel->set_regs(10, (uint16_t)(y_addr & 0xFFFF));
    kernel->set_regs(11, (uint16_t)(y_addr >> 16));
    kernel->set_regs(21, (uint16_t)(num_cores));
  };

  Kernel_t* kernel = GET_KERNEL(KernelId::ROPE);
  TORCH_CHECK(kernel != nullptr, "Failed to get llama_rope kernel");
  kernel->reset_regs();
  setup_regs(kernel);

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
  wq->enqueu_kernel(*kernel, {(uint16_t)seq_len, (uint16_t)1, (uint16_t)1},
                    core_list);
}

// In-place convenience wrapper
void rpu_launch_rope_spm_kernel(
    uint32_t x_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t start_pos,
    int num_cores)                   // number of cores (attn_tp)
{
  rpu_launch_rope_spm_kernel(x_spm_addr, x_spm_addr,
                              cos_ptr, sin_ptr,
                              seq_len, local_heads, head_dim, start_pos,
                              num_cores);
}
