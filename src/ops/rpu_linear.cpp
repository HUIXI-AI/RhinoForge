#include "rhino_launch_buffer.h"

#include "rhino_launch_program.h" // for Program_t
#include "rhino_launch_queue.h"
#include "rpu_ops.h"
#include "rpu_spm_allocator.h"  // eager linear stages through the SPM arena
#include "rpu_linear_tiling.h"  // host-side generated-tile selection
#include <c10/util/Half.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>  // for std::setprecision
#include <stdlib.h> // for uint16_t, setenv, unsetenv
#include <string.h> // for memcpy
#include <string>   // for string
#include <vector>   // for vector
#include <unordered_map>  // for output tensor cache
#include <unistd.h> // for usleep

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
// DDR name is kept for its call sites; nothing here touches parallel_linear_ddr
// any more — that kernel writes its result to DDR with a plain `vst`, which
// needs a watom + fence sequence it does not implement.
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
// the host and GlobalSPM; Buffer_t / GlobalSPM_t export no host
// flush/invalidate contract.
//
// M is chunked so one chunk fits whatever SPM the live fused subsystems left
// free — this can run right after a fused BUILD has taken most of it. There is
// no CPU fallback when even one row does not fit: the weight reaching this
// function is already swizzled, so a CPU GEMM would be silently wrong.
// ---------------------------------------------------------------------------
// DDR-only version: Input, Output, Bias, Weight all in DDR
// partition: 0 = row (按K切分), 1 = col (按N切分)
// NOTE: weight must be pre-transformed using rpu_transform_linear_weight() or
//       Python-side convert_model_for_rpu() before calling this function
namespace {
KernelId autotile_linear_acc32_kernel_id(bool is_fp16, int n_tile);
}

void rpu_launch_linear_ddr_kernel(const at::Tensor &input, const at::Tensor &weight,
                                  at::Tensor &output, const at::Tensor &bias,
                                  bool has_bias, int partition,
                                  const uint64_t *live_dst_base,
                                  int64_t dst_offset_bytes,
                                  const uint64_t *live_bias_base) {
  TORCH_CHECK(input.scalar_type() == at::kHalf,
              "rpu_linear: input must be Half, got ", input.scalar_type());
  TORCH_CHECK(weight.scalar_type() == at::kHalf,
              "rpu_linear: weight must be Half, got ", weight.scalar_type());
  TORCH_CHECK(output.scalar_type() == at::kHalf,
              "rpu_linear: output must be Half, got ", output.scalar_type());
  TORCH_CHECK(output.is_contiguous(), "rpu_linear: output must be contiguous");

  const bool in_graph = graph_dma::active();
  TORCH_CHECK(partition != 0 || !in_graph,
              "rpu_linear: row partition is eager-only — its per-core input "
              "slice is strided (DMA moves contiguous blocks only) and its "
              "cross-core reduce runs on the host, neither of which replays. "
              "Use a col-partition shape (N % 128 == 0) or the fused "
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
  // make either correct — the dst would still be baked, and every operand a
  // graph dereferences after the caller returns also needs a keepalive. Refuse
  // the case loudly instead of shipping a parameter nothing routes.
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
  const size_t local_n = (partition == 0) ? n : n / NUM_CORES;
  const size_t local_k = (partition == 0) ? k / NUM_CORES : k;

  c10::Half *xp = input.data_ptr<c10::Half>();
  c10::Half *yp = output.data_ptr<c10::Half>();
  c10::Half *wp = weight.data_ptr<c10::Half>();
  c10::Half *bp = has_bias ? bias.data_ptr<c10::Half>() : nullptr;

  // Row partition: bias must NOT go through the kernel. Every core would add the
  // full bias and the cross-core reduce below then sums it NUM_CORES times.
  // The same rule validate_m1_partitioned_bias()
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
  const size_t free_bytes = SPM_ALLOC.free_space();

  // The col path's all-gather landing zone (the `+ n` term above) dominates a
  // vocab-width GEMV: at N=151936 it is 297 KB on its own, more than the SPM
  // left free once a fused decoder graph holds its persistent buffers.
  //
  // When the landing zone does not fit, drop the gather and let each core
  // DMA its OWN column block straight to DDR. For a SINGLE row the per-core
  // blocks are exactly that row's contiguous column slices, so the core-major
  // scatter lands them precisely where the gathered copy would have — hence
  // m_chunk is pinned to 1 on this path. One DMA per row instead of one per
  // chunk; only ever taken where the alternative is a hard failure.
  bool col_scatter = false;
  if (partition != 0 && free_bytes <= slack + row_bytes) {
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
                                             NUM_CORES);
    } else {
      rpu_launch_ddr_scatter_spm_dma(bp, local_n, local_n * a_dwidth,
                                     b_addr, NUM_CORES);
    }
  }

  // Row partition only: DDR staging for the strided input repack and for the
  // per-core partial sums on the way back. Both are host-touched, so both need
  // the BOUNDARY force flush — `rpu_ddr_flush` is gated on
  // g_rpu_ddr_flush_enabled, which is OFF by default, i.e. a no-op. Without the
  // force flush the DMA can read stale or zero staging, and the host can read
  // stale outputs coming back.
  at::Tensor x_stage, y_stage;
  c10::Half *xs = nullptr, *ys = nullptr;
  if (partition == 0) {
    x_stage = at::empty({(int64_t)(NUM_CORES * m_chunk * local_k)}, input.options());
    y_stage = at::empty({(int64_t)(NUM_CORES * m_chunk * local_n)}, input.options());
    xs = x_stage.data_ptr<c10::Half>();
    ys = y_stage.data_ptr<c10::Half>();
  }

  rpu_ddr_flush(xp);
  rpu_ddr_flush(wp);
  if (has_bias) rpu_ddr_flush(bp);

  auto* wq = GET_QUEUE(NUM_CORES);
  TORCH_CHECK(rpu_pl_tiling::linear_autotile_enabled(),
              "RPU_LINEAR_AUTOTILE=0 is unsupported by the current operator asset");

  for (size_t row0 = 0; row0 < m; row0 += m_chunk) {
    const size_t cur_m = (m - row0 < m_chunk) ? (m - row0) : m_chunk;

    if (partition == 0) {
      // core c gets the column slice [c*local_k, (c+1)*local_k) of every row
      for (int c = 0; c < NUM_CORES; ++c) {
        for (size_t r = 0; r < cur_m; ++r) {
          memcpy(xs + (c * cur_m + r) * local_k,
                 xp + (row0 + r) * k + c * local_k,
                 local_k * a_dwidth);
        }
      }
      rpu_ddr_flush_force(xs);
      rpu_launch_ddr_scatter_spm_dma(xs, cur_m * local_k,
                                     cur_m * local_k * a_dwidth,
                                     x_addr, NUM_CORES);
    } else {
      // every core needs the whole rows, already contiguous in the input
      rpu_launch_ddr_broadcast_spm_dma(xp + row0 * k, cur_m * k,
                                       x_addr, NUM_CORES);
    }

    const auto tile = rpu_pl_tiling::select_tile_acc32(
        static_cast<int>(cur_m), static_cast<int>(local_n),
        static_cast<int>(local_k), /*is_fp16=*/true);
    const size_t n_per_wrp = static_cast<size_t>(tile.n_tile);
    const size_t m_per_wrp = static_cast<size_t>(tile.m_tile);
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
            static_cast<uint16_t>(NUM_CORES),
            static_cast<uint32_t>(local_k * a_dwidth),
            static_cast<uint32_t>(local_n * a_dwidth),
            0, static_cast<uint16_t>(partition), 2});
    register_writer.write(*kernel);

    wq->set_broadcast_mode(true);
    wq->enqueu_kernel(*kernel,
                      {(uint16_t)CeilDiv(local_n, n_per_wrp),
                       (uint16_t)CeilDiv(cur_m, m_per_wrp), (uint16_t)1},
                      {0, 1, 2, 3, 4, 5, 6, 7});

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
                                                 local_n * a_dwidth, NUM_CORES);
        } else {
          rpu_launch_spm_scatter_ddr_dma(y_addr, yp + row0 * n, local_n,
                                         local_n * a_dwidth, NUM_CORES);
        }
        continue;
      }
      // per-core outputs are different COLUMN blocks of one answer -> concat.
      // (all_reduce_sum is its row-partition counterpart; picking the wrong one
      // is silently wrong, not a crash.)
      rpu_launch_all_gather_spm_kernel(y_addr, g_addr, cur_m, local_n,
                                       a_dwidth, NUM_CORES);
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
                                   cur_m * local_n * a_dwidth, NUM_CORES);
    rpu_ddr_flush_force(ys);

    c10::Half* out = yp + row0 * n;
    const size_t cnt = cur_m * local_n;
    for (size_t j = 0; j < cnt; ++j) {
      float acc = 0.0f;
      for (int c = 0; c < NUM_CORES; ++c)
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
    // partitions are legal, so those weights can disagree with this tie-break.
    // The converter therefore records its choice on the
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

void rpu_linear_into(const Tensor &input, const Tensor &weight,
                     const c10::optional<Tensor> &bias_opt,
                     Tensor &output) {
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
      input_2d, weight, output_2d, bias, has_bias, partition);
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

bool lingbot2_down_acc32_tile_enabled() {
  const char* e = std::getenv("RPU_LINGBOT2_DOWN_ACC32_TILE");
  return e != nullptr && std::strcmp(e, "1") == 0;
}

void validate_m1_partitioned_bias(
    uint32_t bias_spm_addr, int partition, int num_cores) {
  if (bias_spm_addr == 0) return;

  TORCH_CHECK(
      partition == 1 || num_cores == 1,
      "M=1 row-partition Linear cannot add bias before the cross-core "
      "all-reduce; add bias after reduction instead");
}

void add_gemv_bias(
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
    uint32_t bias_spm_addr,            // Bias SPM address (0 = no bias)
    bool force_gemm)                   // skip M=1 GEMV optimization
{
  const bool register_census_active =
      RpuKernelGraph::active().kernel_register_census_active();
  if (M == 1) {
    validate_m1_partitioned_bias(bias_spm_addr, partition, num_cores);
  }
  if (M == 1 && !force_gemm && !register_census_active) {
    rpu_launch_gemv_spm_to_spm_kernel(
        input_spm_addr, weight, output_spm_addr,
        M, N, K, partition, num_cores);
    add_gemv_bias(
        bias_spm_addr, output_spm_addr, N, partition, num_cores);
    return;
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

  // LingBot2 grouped-down has one explicit profile row. Every other shape
  // follows the release FP16-ACC32 table.
  const bool lingbot2_down_tiled =
      M == 51 && N == 768 && K == 16384
      && partition == 0 && num_cores == 8
      && lingbot2_down_acc32_tile_enabled();
  const bool autotile = rpu_pl_tiling::linear_autotile_enabled();
  TORCH_CHECK(autotile || lingbot2_down_tiled,
              "RPU_LINEAR_AUTOTILE=0 is unsupported by the current operator asset");
  auto tile = rpu_pl_tiling::select_tile_acc32(
      static_cast<int>(local_m), static_cast<int>(local_n),
      static_cast<int>(local_k), /*is_fp16=*/true);
  if (lingbot2_down_tiled) tile = {80, 304};
  const size_t n_per_wrp = static_cast<size_t>(tile.n_tile);
  const size_t m_per_wrp = static_cast<size_t>(tile.m_tile);
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
      "; install the combined operator asset");
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
  register_writer.write(*kernel);
  kernel->set_regs(64, grid_x);
  kernel->set_regs(65, grid_y);
  kernel->set_regs(66, 1);

  auto* queue = GET_QUEUE(num_cores);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(
      *kernel, {grid_x, grid_y, static_cast<uint16_t>(1)},
      {0, 1, 2, 3, 4, 5, 6, 7});
}

// Process-startup accumulation override for the tiled FP16/W8A16/W4A16-pgrp
// families. The selected family keeps its weight/scale format; only the
// accumulator-specific tile table and kernel variant change.
static bool linear_acc32_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("RPU_LINEAR_ACC32");
        if (!e || !*e || std::strcmp(e, "0") == 0
                || std::strcmp(e, "false") == 0
                || std::strcmp(e, "False") == 0
                || std::strcmp(e, "off") == 0
                || std::strcmp(e, "OFF") == 0) {
            return false;
        }
        TORCH_CHECK(std::strcmp(e, "1") == 0
                        || std::strcmp(e, "true") == 0
                        || std::strcmp(e, "True") == 0
                        || std::strcmp(e, "on") == 0
                        || std::strcmp(e, "ON") == 0,
                    "RPU_LINEAR_ACC32 accepts only 0/1, false/true, or off/on; got ",
                    e);
        return true;
    }();
    return enabled;
}

bool rpu_get_linear_acc32_enabled() {
  return linear_acc32_enabled();
}
// =============================================================================
// Direct Address SPM-to-SPM Linear Kernel (ACC16 - FP16 accumulation)
// =============================================================================
// Same as acc32 version but uses PARALLEL_LINEAR kernel with FP16 accumulation.
// Its launch geometry differs from the ACC32 variant.
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
    uint32_t bias_spm_addr,            // Bias SPM address (0 = no bias)
    bool force_gemm,                   // skip M=1 GEMV optimization
    const at::Tensor &scale,           // quant scale (fp16 W8/INT4 or fp8 NVFP4)
    uint32_t nvfp4_tensor_scale_spm_addr,
    uint16_t nvfp4_layer_id,
    bool force_acc32)
{
  const bool register_census_active =
      RpuKernelGraph::active().kernel_register_census_active();
  const bool use_acc32 = force_acc32 || linear_acc32_enabled();
  if (M == 1) {
    validate_m1_partitioned_bias(bias_spm_addr, partition, num_cores);
  }
  if (register_census_active) {
    TORCH_CHECK(weight.scalar_type() == at::kHalf ||
                    weight.scalar_type() == at::kChar,
                "typed DDR-register census admits only FP16 Linear or "
                "signed-int8 W8A16 weights");
  }

  if (force_acc32) {
    TORCH_CHECK(
        weight.scalar_type() == at::kHalf,
        "per-handle ACC32 supports fp16 weights only");
    TORCH_CHECK(
        !scale.defined() && nvfp4_tensor_scale_spm_addr == 0,
        "per-handle ACC32 does not support quantized weights");
  }

  if (use_acc32 && weight.scalar_type() == at::kHalf) {
    // FP16 ACC32 selects the accumulator-specific auto-tile table.
    rpu_launch_linear_spm_to_spm_kernel(
        input_spm_addr, weight, output_spm_addr,
        M, N, K, partition, num_cores, bias_spm_addr,
        /*force_gemm=*/true);
    return;
  }

  if (M == 1 && !force_gemm && !use_acc32 && !register_census_active &&
      (weight.scalar_type() == at::kHalf ||
       weight.scalar_type() == at::kChar)) {
    rpu_launch_gemv_spm_to_spm_kernel(
        input_spm_addr, weight, output_spm_addr,
        M, N, K, partition, num_cores, scale);
    add_gemv_bias(
        bias_spm_addr, output_spm_addr, N, partition, num_cores);
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

  if (weight.scalar_type() == at::kChar) {
    TORCH_CHECK(scale.defined() && scale.scalar_type() == at::kHalf,
                "W8A16 prefill GEMM requires a defined fp16 scale tensor");
    TORCH_CHECK(weight.dim() == 2 && weight.size(0) == N &&
                    weight.size(1) == K && weight.is_contiguous(),
                "W8A16 prefill GEMM requires contiguous signed-int8 weight "
                "[N,K]=[", N, ",", K, "], got ", weight.sizes());
    TORCH_CHECK(scale.dim() == 1 && scale.size(0) == N &&
                    scale.is_contiguous(),
                "W8A16 prefill GEMM requires contiguous per-channel fp16 "
                "scale [N]=[", N, "], got ", scale.sizes());
    TORCH_CHECK(weight.nbytes() ==
                    static_cast<size_t>(N) * static_cast<size_t>(K) &&
                    scale.nbytes() ==
                    static_cast<size_t>(N) * sizeof(c10::Half),
                "W8A16 prefill GEMM weight/scale byte extent drifted");

    uint32_t x_addr8 = input_spm_addr;
    uint32_t y_addr8 = output_spm_addr;
    uint32_t b_addr8 = bias_spm_addr;
    uint16_t has_bias8 = (bias_spm_addr != 0) ? 1 : 0;

    // Kernel + grid m/n-tiling come from the accumulator-specific release
    // table. The current operator asset requires auto-tile.
    const bool autotile8 = rpu_pl_tiling::linear_autotile_enabled();
    TORCH_CHECK(autotile8,
                "RPU_LINEAR_AUTOTILE=0 is unsupported by the current operator asset");
    const auto tp8 = use_acc32
        ? rpu_pl_tiling::select_tile_acc32(
              static_cast<int>(local_m), static_cast<int>(local_n),
              static_cast<int>(local_k), /*is_fp16=*/false)
        : rpu_pl_tiling::select_tile_acc16(
              static_cast<int>(local_m), static_cast<int>(local_n),
              static_cast<int>(local_k), /*is_fp16=*/false);
    const KernelId kernel_id8 = use_acc32
        ? autotile_linear_acc32_kernel_id(/*is_fp16=*/false, tp8.n_tile)
        : autotile_linear_kernel_id(/*is_fp16=*/false, tp8.n_tile);
    const uint16_t gx8 =
        static_cast<uint16_t>(CeilDiv(local_n, (size_t)tp8.n_tile));
    const uint16_t gy8 =
        static_cast<uint16_t>(CeilDiv(local_m, (size_t)tp8.m_tile));

    auto register_writer8 =
        RpuKernelGraph::active().stage_kernel_ddr_registers(
            kernel_id8,
            std::vector<GraphDdrRegisterOperandSpec>{
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::Int8ModelWeight,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        2, 3},
                    weight},
                GraphDdrRegisterOperandSpec{
                    GraphDdrRegisterAbi{
                        GraphDdrRegisterRole::PerChannelScale,
                        GraphDdrRegisterAccess::Read,
                        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                        20, 21},
                    scale}});

    int8_t *wp8 = weight.data_ptr<int8_t>();
    c10::Half *sp = scale.data_ptr<c10::Half>();
    rpu_ddr_flush(wp8);
    rpu_ddr_flush(sp);

    Kernel_t* kernel8 = GET_KERNEL(kernel_id8);
    TORCH_CHECK(kernel8 != nullptr,
                "autotile: failed to get W8A16 ",
                use_acc32 ? "ACC32" : "ACC16", " tile m", tp8.m_tile,
                "n", tp8.n_tile, "k128");
    kernel8->reset_regs();
    rpu_pl_tiling::set_parallel_linear_regs(
        *kernel8,
        rpu_pl_tiling::ParallelLinearRegisterArgs{
            x_addr8, 0, y_addr8, b_addr8,
            static_cast<uint32_t>(local_m), static_cast<uint32_t>(local_n),
            static_cast<uint32_t>(local_k), has_bias8,
            static_cast<uint16_t>(num_cores),
            static_cast<uint32_t>(local_k * dwidth),
            static_cast<uint32_t>(local_n * dwidth),
            0, static_cast<uint16_t>(partition), 1});
    register_writer8.write(*kernel8);

    auto* wq8 = GET_QUEUE(num_cores);
    wq8->set_broadcast_mode(true);
    std::vector<uint8_t> core_list8;
    for (int i = 0; i < num_cores; ++i) core_list8.push_back(i);
    wq8->enqueu_kernel(*kernel8, {gx8, gy8, (uint16_t)1}, core_list8);
    return;
  }

  if (weight.scalar_type() == at::kByte) {
    // packed-INT4: weight is uint8 [N, K/2] (swizzled num_ele_32B=64 + nibble-packed
    // by quant/int4_pack.swizzle_pack_int4). Reg layout = wINT4a16_per_channel
    // launch contract; x/y/bias use the production SPM addresses
    // passed in (not the conformance probe's local 256). Pi05 projections are bias-free.
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
                  "NVFP4A16 fp8 block scale must be [K/16,N]=[", K / 16, ",", N,
                  "], got ", scale.sizes());
      TORCH_CHECK(weight.is_contiguous() && scale.is_contiguous(),
                  "NVFP4A16 packed weight and fp8 block scale must be contiguous");

      auto* wp = weight.data_ptr<uint8_t>();
      auto* sp = scale.data_ptr<uint8_t>();
      rpu_ddr_flush(wp);
      rpu_ddr_flush(sp);
      const uint64_t w_addr = RpuGetDevAddr(wp);
      const uint64_t s_addr = RpuGetDevAddr(sp);
      TORCH_CHECK(w_addr % 256 == 0, "NVFP4A16 weight DDR not 256-byte aligned");
      TORCH_CHECK(s_addr % 256 == 0, "NVFP4A16 scale DDR not 256-byte aligned");

      TORCH_CHECK(local_n % 16 == 0 && local_k % 64 == 0,
                  "NVFP4A16 requires local_n%16==0 and local_k%64==0, got ",
                  local_n, " and ", local_k);
      const auto nvfp4_tile = use_acc32
          ? rpu_pl_tiling::kNvfp4Acc32FixedTile
          : rpu_pl_tiling::kNvfp4Acc16FixedTile;
      const KernelId nvfp4_kernel_id = use_acc32
          ? KernelId::PARALLEL_LINEAR_WNVFP4A16_ACC32
          : KernelId::PARALLEL_LINEAR_WNVFP4A16_ACC16;
      auto* kernel = GET_KERNEL(nvfp4_kernel_id);
      TORCH_CHECK(kernel != nullptr,
                  use_acc32 ? "parallel_linear_wNVFP4a16"
                            : "parallel_linear_wNVFP4a16_acc16",
                  " missing from the combined operator asset");

      kernel->reset_regs();
      kernel->set_regs(0,  (uint16_t)(input_spm_addr & 0xFFFF));
      kernel->set_regs(1,  (uint16_t)(input_spm_addr >> 16));
      kernel->set_regs(2,  (uint16_t)((w_addr >> 8) & 0xFFFF));
      kernel->set_regs(3,  (uint16_t)((w_addr >> 24) & 0xFFFF));
      kernel->set_regs(4,  (uint16_t)(output_spm_addr & 0xFFFF));
      kernel->set_regs(5,  (uint16_t)(output_spm_addr >> 16));
      kernel->set_regs(6,  (uint16_t)(bias_spm_addr & 0xFFFF));
      kernel->set_regs(7,  (uint16_t)(bias_spm_addr >> 16));
      kernel->set_regs(8,  (uint16_t)local_m);
      kernel->set_regs(9,  (uint16_t)local_n);
      kernel->set_regs(10, (uint16_t)local_k);
      kernel->set_regs(11, (uint16_t)(local_k / 64));
      kernel->set_regs(12, (uint16_t)(bias_spm_addr != 0));
      kernel->set_regs(13, (uint16_t)num_cores);
      kernel->set_regs(14, (uint16_t)(local_k * dwidth));
      kernel->set_regs(15, (uint16_t)(local_n * dwidth));
      kernel->set_regs(16, (uint16_t)((s_addr >> 8) & 0xFFFF));
      kernel->set_regs(17, (uint16_t)((s_addr >> 24) & 0xFFFF));
      // fp8 block-scale DDR is [K/16,N], one byte per entry.
      kernel->set_regs(18, (uint16_t)(N & 0xFFFF));
      kernel->set_regs(19, (uint16_t)((uint64_t)N >> 16));
      kernel->set_regs(20, (uint16_t)(nvfp4_tensor_scale_spm_addr & 0xFFFF));
      kernel->set_regs(21, (uint16_t)(nvfp4_tensor_scale_spm_addr >> 16));
      kernel->set_regs(22, (uint16_t)(partition == 1));
      kernel->set_regs(23, nvfp4_layer_id);
      // The launch ABI uses one fixed tile per accumulator:
      // ACC16 m208/n128 and ACC32 m128/n128.
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
    // Group-wise int4 (pgrp / AWQ-style): a 2D scale distinguishes it from the
    // per-channel path ([N]). The physical scale payload is controller-striped
    // [ceil(localG/4),ceil(localN/64),cores,4,64]; its 2D view carries group_size
    // in dim 0 for this dispatcher. The kernel ABI extends common regs 0..23
    // with {24:groupSize,25:localG,26-27:SGBlockByteStep}.
    //
    // The host wrapper and linked artifact share the same register, weight
    // packing, scale-layout and fallback-tile contract. Group-wise math differs
    // from per-channel, so a re-quantized checkpoint is required. Supported
    // paths include 8-core col/row and Wall TP2 col/row projections.
    if (scale.dim() == 2) {
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
      // path leaves it unset -> the whole segment falls back to PASSTHROUGH (3x slow).
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
      // IN-KERNEL bias uses the current pgrp asset's per-output broadcast.
      // reg6/7 = bias SPM addr lo/hi as a RAW per-core SPM byte address (same
      // space as in/out regs 0/1,4/5 — NOT >>8-shifted).
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

    TORCH_CHECK(!use_acc32,
                "RPU_LINEAR_ACC32=1 supports W4A16 group-wise pgrp only; "
                "per-channel W4 has no ACC32 kernel family");

    TORCH_CHECK(scale.numel() == N,
                "wINT4 prefill GEMM: scale.numel()=", scale.numel(), " != N=", N);
    TORCH_CHECK(weight.dim() == 2,
                "wINT4 prefill GEMM: packed weight must be uint8 [N, K/2], got dim=",
                weight.dim());
    TORCH_CHECK(K % 2 == 0,
                "wINT4 prefill GEMM: K must be even for packed [N,K/2], got K=", K);
    TORCH_CHECK(weight.size(0) == N && weight.size(1) == K / 2,
                "wINT4 prefill GEMM: packed weight shape must be [N,K/2]=[",
                N, ",", K / 2, "], got [", weight.size(0), ",", weight.size(1), "]");
    TORCH_CHECK(weight.is_contiguous() && scale.is_contiguous(),
                "wINT4 prefill GEMM: packed weight and scale must be contiguous");

    uint8_t *wp4 = weight.data_ptr<uint8_t>();
    c10::Half *sp4 = scale.data_ptr<c10::Half>();
    rpu_ddr_flush(wp4);
    rpu_ddr_flush(sp4);
    uint64_t w_addr4 = RpuGetDevAddr(wp4);
    uint64_t s_addr4 = RpuGetDevAddr(sp4);
    TORCH_CHECK(w_addr4 % 256 == 0, "wINT4 prefill: packed weight DDR not 256-byte aligned");
    TORCH_CHECK(s_addr4 % 256 == 0, "wINT4 prefill: scale DDR not 256-byte aligned");

    uint32_t x_addr4 = input_spm_addr;
    uint32_t y_addr4 = output_spm_addr;
    uint32_t b_addr4 = bias_spm_addr;
    uint16_t has_bias4 = (bias_spm_addr != 0) ? 1 : 0;

    Kernel_t* kernel4 = GET_KERNEL(KernelId::PARALLEL_LINEAR_WINT4_PER_CHANNEL);
    TORCH_CHECK(kernel4 != nullptr, "Failed to get parallel_linear_wINT4a16_per_channel_quant kernel");
    kernel4->reset_regs();
    kernel4->set_regs(0,  (uint16_t)((w_addr4 >> 8) & 0xFFFF));
    kernel4->set_regs(1,  (uint16_t)((w_addr4 >> 24) & 0xFFFF));
    kernel4->set_regs(2,  (uint16_t)(local_n));
    kernel4->set_regs(3,  (uint16_t)(local_k));
    kernel4->set_regs(4,  (uint16_t)(local_k / 16));
    kernel4->set_regs(5,  (uint16_t)(num_cores));
    kernel4->set_regs(6,  (uint16_t)((s_addr4 >> 8) & 0xFFFF));
    kernel4->set_regs(7,  (uint16_t)((s_addr4 >> 24) & 0xFFFF));
    kernel4->set_regs(8,  (uint16_t)(local_m));
    kernel4->set_regs(9,  (uint16_t)(local_k * dwidth));
    kernel4->set_regs(10, (uint16_t)(x_addr4 & 0xFFFF));
    kernel4->set_regs(11, (uint16_t)(x_addr4 >> 16));
    kernel4->set_regs(12, (uint16_t)(y_addr4 & 0xFFFF));
    kernel4->set_regs(13, (uint16_t)(y_addr4 >> 16));
    kernel4->set_regs(14, (uint16_t)(local_n * dwidth));
    kernel4->set_regs(15, (uint16_t)(1));   // dtype_mode: int4 signed
    kernel4->set_regs(16, (uint16_t)(local_n));
    kernel4->set_regs(18, (uint16_t)(partition));
    kernel4->set_regs(19, has_bias4);
    kernel4->set_regs(20, (uint16_t)(b_addr4 & 0xFFFF));
    kernel4->set_regs(21, (uint16_t)(b_addr4 >> 16));
    uint16_t gx4 = (uint16_t)CeilDiv(
        local_n,
        static_cast<size_t>(rpu_pl_tiling::kInt4PerChannelFixedTile.n_tile));
    uint16_t gy4 = (uint16_t)CeilDiv(
        local_m,
        static_cast<size_t>(rpu_pl_tiling::kInt4PerChannelFixedTile.m_tile));
    kernel4->set_regs(64, gx4);
    kernel4->set_regs(65, gy4);
    kernel4->set_regs(66, (uint16_t)1);

    auto* wq4 = GET_QUEUE(num_cores);
    wq4->set_broadcast_mode(true);
    std::vector<uint8_t> core_list4;
    for (int i = 0; i < num_cores; ++i) core_list4.push_back(i);
    wq4->enqueu_kernel(*kernel4, {gx4, gy4, (uint16_t)1}, core_list4);
    return;
  }

  // Weight in DDR
  c10::Half *wp = weight.data_ptr<c10::Half>();
  TORCH_CHECK(rpu_pl_tiling::linear_autotile_enabled(),
              "RPU_LINEAR_AUTOTILE=0 is unsupported by the current operator asset");
  const auto tp = rpu_pl_tiling::select_tile_acc16(
      static_cast<int>(local_m), static_cast<int>(local_n),
      static_cast<int>(local_k), /*is_fp16=*/true);
  const size_t at_n = static_cast<size_t>(tp.n_tile);
  const size_t at_m = static_cast<size_t>(tp.m_tile);
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

  auto* wq = GET_QUEUE(NUM_CORES);
  wq->set_broadcast_mode(true);
  std::vector<uint8_t> core_list;
  for (int i = 0; i < num_cores; ++i) core_list.push_back(i);
  wq->enqueu_kernel(*kernel, grid_dims, core_list);
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
    const at::Tensor& v_scale) {
  constexpr int64_t kM = 320;
  constexpr int64_t kK = 2048;
  constexpr int64_t kQ = 2048;
  constexpr int64_t kKv = 1024;
  constexpr int64_t kQPerCore = 256;

  auto check_weight = [&](const at::Tensor& tensor, int64_t n,
                          const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kChar &&
                    tensor.is_contiguous() && tensor.dim() == 2 &&
                    tensor.size(0) == n && tensor.size(1) == kK,
                "Wall QKV multibase ", name,
                " weight must be contiguous RPU int8 [", n, ",", kK,
                "], got ", tensor.sizes());
  };
  auto check_scale = [&](const at::Tensor& tensor, int64_t n,
                         const char* name) {
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == at::kHalf &&
                    tensor.is_contiguous() && tensor.dim() == 1 &&
                    tensor.size(0) == n,
                "Wall QKV multibase ", name,
                " scale must be contiguous RPU fp16 [", n, "], got ",
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

  constexpr KernelId kernel_id =
      KernelId::WALL_QKV_W8A16_MULTIBASE_M448N64;
  auto register_writer =
      RpuKernelGraph::active().stage_kernel_ddr_registers(
          kernel_id,
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
    rpu_ddr_flush(tensor->data_ptr());
    TORCH_CHECK(RpuGetDevAddr(tensor->data_ptr()) % 256 == 0,
                "Wall QKV multibase DDR operand is not 256-byte aligned");
  }

  Kernel_t* kernel = GET_KERNEL(kernel_id);
  TORCH_CHECK(
      kernel != nullptr,
      "qkv_parallel_w8a16_multibase_m448n64k128 missing from the combined operator asset");
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
