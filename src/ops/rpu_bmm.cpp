#include "rhino_launch_buffer.h"
#include "rhino_launch_program.h" // for Program_t
#include "rhino_launch_queue.h"

#include "rpu_ops.h"
#include "rpu_spm_allocator.h" // for SPM_ALLOC (bmm_spm_test only)
#include <stdlib.h> // for uint16_t, setenv, unsetenv
#include <string.h> // for memcpy
#include <string>   // for string
#include <vector>   // for vector
#include <unordered_map> // for local kernel cache

using namespace ::rhino_lkn;

// Thread-local lookup cache avoids repeated KernelCache queries.
static thread_local std::unordered_map<std::string, Kernel_t*> bmm_kernel_cache;

static constexpr uint64_t kPreloadedEagerBmmTiles[][3] = {
    {64, 64, 64},   {64, 64, 128},
    {64, 128, 64},  {64, 128, 128},
    {128, 128, 64}, {128, 128, 128},
};

static bool is_preloaded_eager_bmm_tile(uint64_t m, uint64_t n, uint64_t k) {
  for (const auto &tile : kPreloadedEagerBmmTiles) {
    if (tile[0] == m && tile[1] == n && tile[2] == k) return true;
  }
  return false;
}

void rpu_launch_bmm_kernel(const at::Tensor &input_a, const at::Tensor &input_b,
                           at::Tensor &output) {

  uint64_t M = 0, N = 0, K = 0;
  uint64_t Mv16 = 0, Nv16 = 0, Kv16 = 0;
  uint64_t Mv16Tail = 0, Nv16Tail = 0, Kv16Tail = 0, Kv16Pad = 0;
  auto batch = input_a.size(0);
  M = input_a.size(1);
  K = input_a.size(2);
  N = input_b.size(2);
  std::vector<uint64_t> step_a(3);
  std::vector<uint64_t> step_b(3);
  step_a[2] = 1;
  step_b[2] = 1;
  for (size_t i = 2; i != 0; i--) {
    step_a[i - 1] = step_a[i] * input_a.size(i);
    step_b[i - 1] = step_b[i] * input_b.size(i);
  }
  /* set kernel tiles */
  uint64_t tile_k_per_warp = 0, tile_m_per_warp = 0, tile_n_per_warp = 0;

  if (K <= 64) {
    tile_k_per_warp = 64;
  } else if (K > 64 && K <= 96) {
    tile_k_per_warp = 96;
  } else if (K > 96 && K <= 128) {
    tile_k_per_warp = 128;
  } else if (K > 128 && K % 96 == 0) {
    tile_k_per_warp = 96;
  } else {
    tile_k_per_warp = 128;
  }

  if (M <= 64) {
    tile_m_per_warp = 64;
  } else if (M > 64 && M <= 96) {
    tile_m_per_warp = 96;
  } else if (M > 96 && M <= 128) {
    tile_m_per_warp = 128;
  } else if (M > 128 && M <= 192) {
    tile_m_per_warp = 192;
  } else if (M > 192 && M <= 256) {
    tile_m_per_warp = 256;
  } else if (M > 256 && M % 192 == 0) {
    tile_m_per_warp = 192;
  } else {
    tile_m_per_warp = 128;
  }

  if (N <= 64) {
    tile_n_per_warp = 64;
  } else if (N > 64 && N <= 96) {
    tile_n_per_warp = 96;
  } else if (N > 96 && N <= 128) {
    tile_n_per_warp = 128;
  } else if (N > 128 && N <= 192) {
    tile_n_per_warp = 192;
  } else if (N > 192 && N <= 256) {
    tile_n_per_warp = 256;
  } else if (N > 256 && N % 192 == 0) {
    tile_n_per_warp = 192;
  } else {
    tile_n_per_warp = 128;
  }

  TORCH_CHECK(
      is_preloaded_eager_bmm_tile(tile_m_per_warp, tile_n_per_warp,
                                  tile_k_per_warp),
      "rpu_bmm: unsupported kernel tile (", tile_m_per_warp, ", ",
      tile_n_per_warp, ", ", tile_k_per_warp, ")");

  uint64_t tile_m = tile_m_per_warp, tile_n = tile_n_per_warp, tile_k = tile_k_per_warp;

  Mv16 = CeilDiv(M, 16);
  Kv16 = CeilDiv(K, 16);
  Nv16 = CeilDiv(N, 16);

  int tail_v16 = 0;
  int tile_m_v16 = tile_m / 16;
  int tile_n_v16 = tile_n / 16;
  int tile_k_v16 = tile_k / 16;

  Mv16Tail = Mv16 + tail_v16;
  Nv16Tail = Nv16 + tail_v16;
  Kv16Tail = Kv16 + tail_v16;
  Kv16Pad = Align(Kv16, tile_k_v16);

  int32_t dwidth = sizeof(c10::Half);
  int32_t dwidth_v16 = dwidth * 16;

  int32_t spm_a_v16_size = batch * M * Kv16Tail;
  int32_t spm_b_v16_size = batch * K * Nv16Tail;
  int32_t spm_y_v16_size = batch * M * Nv16Tail;

  // 构建 kernel 名称
  std::string kernel_mode = (K == tile_k_per_warp) ? "peak" : "univ";
  std::string kernel_name = "gemm_fp16_spm_16b_w" + std::to_string(tile_m_per_warp) + "x" +
                std::to_string(tile_n_per_warp) + "_k" +
                std::to_string(tile_k_per_warp) + "_1core_buf1_nt_lpaddr_" + kernel_mode;

  // 从 thread_local 缓存获取 Kernel，避免每次查询主缓存
  Kernel_t* kernel = nullptr;
  auto it = bmm_kernel_cache.find(kernel_name);
  if (it != bmm_kernel_cache.end()) {
    kernel = it->second;
  } else {
    kernel = KernelCache::instance().get_kernel(kernel_name);
    if (kernel != nullptr) {
      bmm_kernel_cache[kernel_name] = kernel;
    }
  }
  TORCH_CHECK(kernel != nullptr, "Failed to get ", kernel_name, " kernel from cache");

  auto* wq = GET_QUEUE(1);

  // kernel->print_kernel_info();

  LocalSPM_t spm_core0_input_a(spm_a_v16_size * dwidth_v16, read_write,
                                kStride32B, 2, 0);
  RPU_CHECK_SPM(spm_core0_input_a, spm_a_v16_size * dwidth_v16);
  LocalSPM_t spm_core0_input_b(spm_b_v16_size * dwidth_v16, read_write,
                                kStride32B, 2, 0);
  RPU_CHECK_SPM(spm_core0_input_b, spm_b_v16_size * dwidth_v16);
  LocalSPM_t spm_core0_output(spm_y_v16_size * dwidth_v16, read_write,
                               kStride32B, 2, 0);
  RPU_CHECK_SPM(spm_core0_output, spm_y_v16_size * dwidth_v16);

  const c10::Half *ap = input_a.data_ptr<c10::Half>();
  const c10::Half *bp = input_b.data_ptr<c10::Half>();
  c10::Half *out = output.data_ptr<c10::Half>();

  ddr_to_spm(spm_core0_input_a.get_cpu_ptr(), ap,
             spm_a_v16_size * dwidth_v16);
  ddr_to_spm(spm_core0_input_b.get_cpu_ptr(), bp,
             spm_b_v16_size * dwidth_v16);

  // Context_t ctx;
  // ctx.recode_queue(&wq);
  // ctx.record_buffer(&spm_core0_input_a);
  // ctx.record_buffer(&spm_core0_input_b);
  // ctx.record_buffer(&spm_core0_output);
  // ctx.dump_init_buf();

  kernel->set_regs(0,
                  (uint16_t)((spm_core0_input_a.get_rpu_addr() / 32) & 0xFFFF));
  kernel->set_regs(1, (uint16_t)((spm_core0_input_a.get_rpu_addr() / 32) >> 16));
  kernel->set_regs(2,
                  (uint16_t)((spm_core0_input_b.get_rpu_addr() / 32) & 0xFFFF));
  kernel->set_regs(3, (uint16_t)((spm_core0_input_b.get_rpu_addr() / 32) >> 16));
  kernel->set_regs(4,
                  (uint16_t)((spm_core0_output.get_rpu_addr() / 32) & 0xFFFF));
  kernel->set_regs(5, (uint16_t)((spm_core0_output.get_rpu_addr() / 32) >> 16));
  kernel->set_regs(6,
                  (uint16_t)((spm_core0_output.get_rpu_addr() / 32) & 0xFFFF));
  kernel->set_regs(7, (uint16_t)((spm_core0_output.get_rpu_addr() / 32) >> 16));
  /* hasABatch */
  uint16_t hasABatch = (input_a.size(0) == 1) ? 0 : 1;
  kernel->set_regs(8, hasABatch);

  /* hasBBatch */
  uint16_t hasBBatch = (input_b.size(0) == 1) ? 0 : 1;
  kernel->set_regs(9, hasBBatch);

  uint16_t hasCType = 4;
  kernel->set_regs(10, hasCType);

  /* The current BMM API does not expose a C scalar term. */
  uint16_t CScalar = 0;
  kernel->set_regs(11, CScalar);

  /* M */
  kernel->set_regs(12, (uint16_t)M);

  /* Mv16 */
  kernel->set_regs(13, (uint16_t)Mv16);

  /* Mv16Tail */
  kernel->set_regs(14, (uint16_t)Mv16Tail);

  /* N */
  kernel->set_regs(15, (uint16_t)N);

  /* Nv16 */
  kernel->set_regs(16, (uint16_t)Nv16);

  /* Nv16Tail */
  kernel->set_regs(17, (uint16_t)Nv16Tail);

  /* K */
  kernel->set_regs(18, (uint16_t)K);

  /* Kv16 */
  kernel->set_regs(19, (uint16_t)Kv16);

  /* Kv16Tail */
  kernel->set_regs(20, (uint16_t)Kv16Tail);

  /* kernel tiling size checkpoint */
  uint16_t vlm_usage = (tile_m_per_warp * tile_k_per_warp / 16) +
                       (tile_n_per_warp * tile_k_per_warp / 16) +
                       (tile_m_per_warp * tile_n_per_warp / 16);

  if (vlm_usage >= 400) {
    /* squeeze the largest tiling to 128 */
    if (tile_m_per_warp >= tile_n_per_warp && tile_m_per_warp > 128) {
      tile_m_per_warp = 128;
    } else if (tile_m_per_warp < tile_n_per_warp && tile_n_per_warp > 128) {
      tile_n_per_warp = 128;
    }
  }

  /* Kv16Pad */
  // uint16_t Kv16Pad = Align(Kv16, (tile_k_per_warp / 16));
  kernel->set_regs(21, (uint16_t)Kv16Pad);

  /* alpha */
  c10::Half alpha = 1.0;
  kernel->set_regs(22, alpha.x);

  /* beta */
  c10::Half beta = 1.0;
  kernel->set_regs(23, beta.x);

  /* relu */
  uint16_t hasRelu = 0;
  kernel->set_regs(24, hasRelu);

  /* grid dim x */
  uint16_t grid_dim_x = CeilDiv(N, tile_n_per_warp);
  kernel->set_regs(64, grid_dim_x);

  /* grid dim y */
  uint16_t grid_dim_y = CeilDiv(M, tile_m_per_warp);
  kernel->set_regs(65, grid_dim_y);

  /* grid dim z */
  // uint16_t batch = std::max(input_a.size[0], input_b.size[0]);
  uint16_t grid_dim_z = batch;
  kernel->set_regs(66, grid_dim_z);

  wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, {0});
  spm_to_ddr(out, spm_core0_output.get_cpu_ptr(), spm_y_v16_size * dwidth_v16);
}

// ============ BMM SPM Kernel Launch (multi-core, fused-graph) ============
// SPM-resident batched matmul: out[batch,M,N] = A[batch,M,K] @ B[batch,K,N], fp16.
// Uses the same gemm_fp16_spm_16b kernel, tiling and register contract as the
// eager rpu_launch_bmm_kernel above. It differs in:
//   - operands are ALREADY in SPM: caller passes byte addresses, no DDR<->SPM DMA;
//   - graph-aware kernel fetch (RpuKernelGraph::active().get_kernel_reset) so the
//     launch records into the fused graph instead of the eager kernel
//     PASSTHROUGH fallback (the eager path's KernelCache::get_kernel is fine in
//     eager but would make the recorded graph non-replayable);
//   - multi-core broadcast: each of num_cores cores runs the batched gemm on its
//     OWN per-core SPM at the same offsets (for the GDN chunk, batch = heads/core).
// No-C path (hasCType=4) stores raw A@B (alpha/beta/C ignored, as in the
// eager path). Assumes M,N,K % 16 == 0 so the v16 SPM
// layout equals the plain contiguous fp16 layout (true for GDN chunk: C=64, Dk=Dv=128).
// batch applies to both operands (symmetric per-head).
void rpu_launch_bmm_spm_kernel(uint32_t a_spm_addr, uint32_t b_spm_addr,
                               uint32_t out_spm_addr, int64_t M, int64_t N,
                               int64_t K, int64_t batch, int num_cores,
                               BmmMode mode) {
  uint64_t Mv16 = 0, Nv16 = 0, Kv16 = 0;
  uint64_t Mv16Tail = 0, Nv16Tail = 0, Kv16Tail = 0, Kv16Pad = 0;

  /* Match the eager-path tile selection. */
  uint64_t tile_k_per_warp = 0, tile_m_per_warp = 0, tile_n_per_warp = 0;
  if (K <= 64) {
    tile_k_per_warp = 64;
  } else if (K > 64 && K <= 96) {
    tile_k_per_warp = 96;
  } else if (K > 96 && K <= 128) {
    tile_k_per_warp = 128;
  } else if (K > 128 && K % 96 == 0) {
    tile_k_per_warp = 96;
  } else {
    tile_k_per_warp = 128;
  }

  if (M <= 64) {
    tile_m_per_warp = 64;
  } else if (M > 64 && M <= 96) {
    tile_m_per_warp = 96;
  } else if (M > 96 && M <= 128) {
    tile_m_per_warp = 128;
  } else if (M > 128 && M <= 192) {
    tile_m_per_warp = 192;
  } else if (M > 192 && M <= 256) {
    tile_m_per_warp = 256;
  } else if (M > 256 && M % 192 == 0) {
    tile_m_per_warp = 192;
  } else {
    tile_m_per_warp = 128;
  }

  if (N <= 64) {
    tile_n_per_warp = 64;
  } else if (N > 64 && N <= 96) {
    tile_n_per_warp = 96;
  } else if (N > 96 && N <= 128) {
    tile_n_per_warp = 128;
  } else if (N > 128 && N <= 192) {
    tile_n_per_warp = 192;
  } else if (N > 192 && N <= 256) {
    tile_n_per_warp = 256;
  } else if (N > 256 && N % 192 == 0) {
    tile_n_per_warp = 192;
  } else {
    tile_n_per_warp = 128;
  }

  uint64_t tile_k = tile_k_per_warp;

  Mv16 = CeilDiv(M, 16);
  Kv16 = CeilDiv(K, 16);
  Nv16 = CeilDiv(N, 16);

  int tail_v16 = 0;
  int tile_k_v16 = tile_k / 16;

  Mv16Tail = Mv16 + tail_v16;
  Nv16Tail = Nv16 + tail_v16;
  Kv16Tail = Kv16 + tail_v16;
  Kv16Pad = Align(Kv16, tile_k_v16);

  // kernel name uses the ORIGINAL tiles (before the vlm squeeze below) — matches eager.
  // Orientation suffix from `mode` (nt/nn/tn/tt). M/N/K + reg setup are identical
  // across modes for contiguous packed 16-aligned operands; only the suffix changes.
  const char* gemm_suffix = "nt";
  switch (mode) {
    case BmmMode::RowByCol: gemm_suffix = "nt"; break;
    case BmmMode::RowByRow: gemm_suffix = "nn"; break;
    case BmmMode::ColByRow: gemm_suffix = "tn"; break;
    case BmmMode::ColByCol: gemm_suffix = "tt"; break;
  }
  std::string kernel_mode = (K == (int64_t)tile_k_per_warp) ? "peak" : "univ";
  std::string kernel_name = "gemm_fp16_spm_16b_w" + std::to_string(tile_m_per_warp) + "x" +
                std::to_string(tile_n_per_warp) + "_k" +
                std::to_string(tile_k_per_warp) + "_1core_buf1_" + gemm_suffix + "_lpaddr_" + kernel_mode;

  Kernel_t* kernel = RpuKernelGraph::active().get_kernel_reset(kernel_name);
  TORCH_CHECK(kernel != nullptr,
              "rpu_launch_bmm_spm_kernel: failed to get ", kernel_name, " kernel");

  // SPM byte addr / 32 (lpaddr granularity) — same scaling as the eager.
  const uint32_t a32 = a_spm_addr / 32, b32 = b_spm_addr / 32, y32 = out_spm_addr / 32;
  kernel->set_regs(0, (uint16_t)(a32 & 0xFFFF));
  kernel->set_regs(1, (uint16_t)(a32 >> 16));
  kernel->set_regs(2, (uint16_t)(b32 & 0xFFFF));
  kernel->set_regs(3, (uint16_t)(b32 >> 16));
  kernel->set_regs(4, (uint16_t)(y32 & 0xFFFF));
  kernel->set_regs(5, (uint16_t)(y32 >> 16));
  kernel->set_regs(6, (uint16_t)(y32 & 0xFFFF));  // C addr = out (ignored when hasCType=4)
  kernel->set_regs(7, (uint16_t)(y32 >> 16));

  uint16_t hasABatch = (batch == 1) ? 0 : 1;
  kernel->set_regs(8, hasABatch);
  uint16_t hasBBatch = (batch == 1) ? 0 : 1;
  kernel->set_regs(9, hasBBatch);
  kernel->set_regs(10, (uint16_t)4);  // hasCType=4 (no C -> store raw A@B)
  kernel->set_regs(11, (uint16_t)0);  // CScalar
  kernel->set_regs(12, (uint16_t)M);
  kernel->set_regs(13, (uint16_t)Mv16);
  kernel->set_regs(14, (uint16_t)Mv16Tail);
  kernel->set_regs(15, (uint16_t)N);
  kernel->set_regs(16, (uint16_t)Nv16);
  kernel->set_regs(17, (uint16_t)Nv16Tail);
  kernel->set_regs(18, (uint16_t)K);
  kernel->set_regs(19, (uint16_t)Kv16);
  kernel->set_regs(20, (uint16_t)Kv16Tail);

  /* kernel tiling size checkpoint (verbatim from eager) */
  uint16_t vlm_usage = (tile_m_per_warp * tile_k_per_warp / 16) +
                       (tile_n_per_warp * tile_k_per_warp / 16) +
                       (tile_m_per_warp * tile_n_per_warp / 16);
  if (vlm_usage >= 400) {
    if (tile_m_per_warp >= tile_n_per_warp && tile_m_per_warp > 128) {
      tile_m_per_warp = 128;
    } else if (tile_m_per_warp < tile_n_per_warp && tile_n_per_warp > 128) {
      tile_n_per_warp = 128;
    }
  }

  kernel->set_regs(21, (uint16_t)Kv16Pad);
  c10::Half alpha = 1.0;
  kernel->set_regs(22, alpha.x);
  c10::Half beta = 1.0;
  kernel->set_regs(23, beta.x);
  kernel->set_regs(24, (uint16_t)0);  // hasRelu

  uint16_t grid_dim_x = CeilDiv(N, tile_n_per_warp);
  uint16_t grid_dim_y = CeilDiv(M, tile_m_per_warp);
  uint16_t grid_dim_z = (uint16_t)batch;
  kernel->set_regs(64, grid_dim_x);
  kernel->set_regs(65, grid_dim_y);
  kernel->set_regs(66, grid_dim_z);

  auto* wq = GET_QUEUE(num_cores);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> cores;
  for (int i = 0; i < num_cores; ++i) cores.push_back((uint8_t)i);
  wq->enqueu_kernel(*kernel, {grid_dim_x, grid_dim_y, grid_dim_z}, cores);
}
