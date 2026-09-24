// rpu_kernel_decls.h — All kernel launch and wrapper function declarations
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <string>
#include <optional>
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include "rpu_helpers.h"  // SdpaKernelType
#include "rpu_math_precision.h"
#include "rpu_eltwise.h"  // ValuOpType
#include "rpu_kvinsert_segment_plan.h"

class GraphDmaSemanticEndpoint;
struct GraphStats;
namespace v3 {
class SpmFmbPostFnYieldTarget;
struct SpmOffset;
}

int64_t rpu_execution_reconfigure_begin(int64_t attempt_token);
void rpu_execution_reconfigure_commit(int64_t token);
void rpu_execution_reconfigure_abort(int64_t token);
bool rpu_execution_reconfigure_abort_attempt(int64_t attempt_token);

// Note: bare 'Tensor' usages rely on 'using namespace at;' in rpu_ops.h umbrella

// =============================================================================
// Eltwise Binary Kernels
// =============================================================================

void rpu_launch_eltwise_binary_sameshape_kernel(const at::Tensor &input_a,
                                                const at::Tensor &input_b,
                                                const c10::Half alpha_f,
                                                at::Tensor &output,
                                                ValuOpType op_type);
void rpu_launch_eltwise_binary_scalar_kernel(const at::Tensor &input_a,
                                             const c10::Half scalar_val,
                                             const c10::Half alpha_f,
                                             at::Tensor &output,
                                             ValuOpType op_type);
void rpu_launch_eltwise_binary_Nx1_NxC_kernel(const at::Tensor &input_a,
                                              const at::Tensor &input_b,
                                              const c10::Half alpha_f,
                                              at::Tensor &output,
                                              ValuOpType op_type,
                                              bool is_bopa = false);
void rpu_launch_eltwise_binary_1xC_NxC_kernel(const at::Tensor &input_a,
                                              const at::Tensor &input_b,
                                              const c10::Half alpha_f,
                                              at::Tensor &output,
                                              ValuOpType op_type,
                                              bool is_bopa = false);

// =============================================================================
// Eltwise Unary Kernels
// =============================================================================

enum class GeluMode : uint8_t {
    NONE = 0,
    TANH = 1,
    ERF = 2,
    ERF_ULTRA = 3,
};

// Cold installation check. Missing optional payloads must reject an opted-in
// model before it records a graph or mutates installed state.
void rpu_require_high_precision_math_kernels();
void rpu_require_high_precision_fusion_kernels();
void rpu_launch_rhinovla_newton_norm_shift_spm_kernel(
    uint32_t input, uint32_t output, uint32_t scale, uint32_t shift,
    int64_t rows, double eps, int cores = 8);
void rpu_launch_rhinovla_silu_high_mul_spm_kernel(
    uint32_t gate, uint32_t up, uint32_t output, int64_t elements, int cores = 8);

// HIGH stages through SPM with synchronous DMA (cold/PASSTHROUGH only), since
// the high-precision payloads use byte addresses and cannot address DDR >4GiB.
// It requires contiguous RPU FP16 operands and a 16-byte-aligned byte count.
void rpu_launch_eltwise_unary_kernel(const at::Tensor &input,
                                     at::Tensor &output, ValuOpType op_type,
                                     RpuUnaryPrecision precision = RpuUnaryPrecision::BASE);
// SPM unary kernel - input/output in SPM
void rpu_launch_eltwise_unary_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_elements,
    ValuOpType op_type,
    GeluMode gelu_mode = GeluMode::NONE,
    int num_cores = 8,
    RpuUnaryPrecision precision = RpuUnaryPrecision::BASE);
// Preserve source compatibility for non-GELU call sites that still pass a
// boolean explicitly.  A true value meant tanh-approximate GELU historically;
// route it to the formula-specific replacement rather than the legacy symbol.
inline void rpu_launch_eltwise_unary_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_elements,
    ValuOpType op_type,
    bool is_gelu,
    int num_cores = 8,
    RpuUnaryPrecision precision = RpuUnaryPrecision::BASE) {
    rpu_launch_eltwise_unary_spm_kernel(
        input_spm_addr, output_spm_addr, num_elements, op_type,
        is_gelu ? GeluMode::TANH : GeluMode::NONE, num_cores, precision);
}
// Sum over rows (axis 0) of an [rows, cols<256] fp16 SPM tensor → [cols] in ONE
// launch (replaces the 7-step tree-reduce). Kernel reduce_sum_non_last_dim_out_seg;
void rpu_launch_reduce_sum_rows_spm_kernel(
    uint32_t in_spm_addr, uint32_t out_spm_addr,
    int64_t rows, int64_t cols, int num_cores = 1);

// Qwen3.5 decode-only GDN fused kernels. All addresses are absolute SPM
// addresses and are valid in immediate and graph modes.
void rpu_launch_qwen3_5_rms_norm_gated(
    uint32_t x_addr, uint32_t z_addr, uint32_t normw_addr, uint32_t y_addr,
    int64_t M, int64_t Dv, double eps, int num_cores = 8);
void rpu_launch_qwen3_5_rank1_fma(
    uint32_t state_addr, uint32_t delta_addr, uint32_t kbc_addr,
    int64_t M, int64_t Dk, int64_t Dv, int num_cores = 1);
void rpu_launch_qwen3_5_mul_reduce_rows(
    uint32_t state_addr, uint32_t out_addr, uint32_t vec_splat_addr,
    uint32_t scratch_addr, int64_t M, int64_t Dk, int64_t Dv,
    int num_cores = 1);

// =============================================================================
// BMM / Softmax / LayerNorm
// =============================================================================

void rpu_launch_bmm_kernel(const at::Tensor &self, const at::Tensor &mat2,
                           at::Tensor &result);
void rpu_launch_softmax_c16_kernel(const at::Tensor &input, at::Tensor &output);
void rpu_launch_layernorm_kernel(const at::Tensor &input, at::Tensor &output,
                                 c10::IntArrayRef normalized_shape,
                                 const at::Tensor &weight,
                                 const at::Tensor &bias, double eps);

// =============================================================================
// Linear Kernels
// =============================================================================

at::Tensor rpu_linear(const at::Tensor &input, const at::Tensor &weight,
                      const c10::optional<at::Tensor> &bias_opt);

// Explicit layout for planner-owned, already-swizzled FP16 projections.
// Fresh output; eager DDR path only. The ordinary aten::linear ABI stays TP8.
at::Tensor rpu_linear_with_core_count(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t partition, int64_t num_cores);

// Explicit cold accumulator for model-owned eager output projections.
at::Tensor rpu_linear_with_accumulation(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    int64_t partition, int64_t num_cores, bool linear_acc32,
    const c10::optional<at::Tensor>& scale_opt);

void rpu_linear_into(const at::Tensor &input, const at::Tensor &weight,
                     const c10::optional<at::Tensor> &bias_opt,
                     at::Tensor &output, bool linear_acc32 = true);

// Eager W8A16 projections: TP8 col-swizzled INT8 weight, natural FP16 [N] scale.
// The returning form owns fresh output storage; neither form may be captured.
at::Tensor rpu_linear_w8a16(const at::Tensor& input, const at::Tensor& weight,
                          const at::Tensor& scale,
                          const c10::optional<at::Tensor>& bias);
void rpu_linear_w8a16_into(const at::Tensor& input, const at::Tensor& weight,
                         const at::Tensor& scale,
                         const c10::optional<at::Tensor>& bias,
                         at::Tensor& output, bool linear_acc32 = false);
// Eager TP8 col-pgrp G32: packed UINT8 [N,K/2], striped FP16 scale [32,...].
// Returns fresh output and rejects graph capture, matching the W8 DDR boundary.
at::Tensor rpu_linear_w4a16(const at::Tensor& input, const at::Tensor& weight,
                          const at::Tensor& scale,
                          const c10::optional<at::Tensor>& bias);

// `live_dst_base` (optional): when non-null, the col-partition write-back DMAs
// bind their DDR destination through the caller-owned live base instead of
// baking `output.data_ptr()` at BUILD. REQUIRED for a graph-mode caller whose
// `output` is freshly allocated per forward (P4) — a fixed DMA's kd_buf dst is
// frozen at BUILD and REPLAY's sync-only fast path never rewrites it, so the
// write lands in the PREVIOUS forward's buffer. Effective dst =
// `*live_dst_base + dst_offset_bytes + row0*N*sizeof(Half)`; the caller updates
// `*live_dst_base` before each forward. Null = the historical fixed behavior
// (only correct for a stable `output`).
//
// `live_bias_base` (optional): same treatment for the bias scatter. The weight
// travels to the kernel in a REGISTER, which REPLAY re-syncs via
// sync_mutable_params, so swapping the weight tensor takes effect; the bias
// travels by DMA, which sync-only does not rebind, so a `set_weights` that
// swaps the bias tensor was silently ignored. Null = the historical fixed
// behavior (only correct for a stable `bias`).
void rpu_launch_linear_ddr_kernel(const at::Tensor &input, const at::Tensor &weight,
                                  at::Tensor &output, const at::Tensor &bias,
                                  bool has_bias, int partition,
                                  const uint64_t *live_dst_base = nullptr,
                                  int64_t dst_offset_bytes = 0,
                                  const uint64_t *live_bias_base = nullptr,
                                  int num_cores = 8,
                                  const at::Tensor& weight_scale = {},
                                  c10::optional<bool> linear_acc32 = c10::nullopt);

// =============================================================================
// SDPA (Scaled Dot Product Attention) Kernels
// =============================================================================

// Eager unified-layout SDPA computes on CPU; fused decoders use the SPM launcher.
at::Tensor rpu_scaled_dot_product_attention_unified(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &attn_mask, double dropout_p, bool is_causal,
    std::optional<double> scale, bool enable_gqa);

// aten::scaled_dot_product_attention — PyTorch layout [batch, heads, seq, dim].
// Separate from the unified op above so that neither entry point has to infer
// its layout from head_dim; see the comment on the definition.
at::Tensor rpu_scaled_dot_product_attention(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &attn_mask, double dropout_p, bool is_causal,
    std::optional<double> scale, bool enable_gqa);

// =============================================================================
// Tile / Permute / Slice Kernels (for rpu_contiguous view fast-paths)
// =============================================================================

// Tile (expand/repeat) kernel for contiguous() of expand-view tensors.
// 2D: input (N, 1) -> output (N, C) where repeats = [1, C]
// 3D: input (B, N, C) -> output (B*r0, N*r1, C*r2) where repeats = [r0, r1, r2]
void rpu_launch_tile_kernel(const at::Tensor &input,
                            at::Tensor &output,
                            const std::vector<int64_t> &repeats);

// Permute (transpose) kernel for contiguous() of permute-view tensors.
// 3D: input (B, N, C) -> output with permuted dimensions according to dims
//     Supported orders: "102", "201", "021", "210", "120"
// 4D: input (Q, B, N, C) -> output with permuted dimensions according to dims
//     Supported orders: "0213", "2103", "0321"
void rpu_launch_permute_kernel(const at::Tensor &input,
                               at::Tensor &output,
                               const std::vector<int64_t> &dims);

// Slice kernel for contiguous() of slice-view tensors.
struct SliceParams {
    std::vector<int64_t> axes;    // which axes are sliced
    std::vector<int64_t> begins;  // start indices
    std::vector<int64_t> ends;    // end indices
    std::vector<int64_t> steps;   // step values
};

// Launch slice kernel. Supports up to 2 sliced axes, FP16 only.
// Uses slice_last_dim_axis for last-dim slices with step=1; otherwise
// slice_single_axis for general single-axis slices.
void rpu_launch_slice_kernel(const at::Tensor &input,
                             at::Tensor &output,
                             const SliceParams &params);

// =============================================================================
// RoPE (Rotary Position Embedding) Kernels
// =============================================================================

void rpu_launch_rope_kernel(const at::Tensor &x,
                            const at::Tensor &cos,
                            const at::Tensor &sin,
                            int64_t pos,
                            at::Tensor &output);

std::tuple<at::Tensor, at::Tensor> rpu_apply_rotary_pos_emb(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &cos,
    const at::Tensor &sin,
    const at::Tensor &position_ids);

// RoPE SPM版本 (tensor-based, legacy)
void rpu_launch_rope_spm_kernel(
    c10::Half *x_ptr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    c10::Half *out_ptr,
    int64_t seq_len,
    int64_t heads_num,
    int64_t head_dim,
    int64_t start_pos);

// RoPE SPM版本 (direct address, non-in-place)
void rpu_launch_rope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t start_pos,
    int num_cores = 8);

// RoPE SPM版本 (in-place convenience wrapper)
void rpu_launch_rope_spm_kernel(
    uint32_t x_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t start_pos,
    int num_cores = 8);

// =============================================================================
// RhinoVLA full-head logical RoPE: owned partial_mrope or original llama_rope.
void rpu_launch_rhinovla_rope_spm_kernel(
    uint32_t input, uint32_t output, const at::Tensor& cos, const at::Tensor& sin,
    int64_t rows, int64_t local_heads, int64_t head_dim, int64_t position,
    bool partial, int num_cores);

// M-RoPE (Qwen3-VL R-Phase 1)
// =============================================================================
//
// cos/sin tables live in DDR with shape [max_seq, head_dim/2]; position_ids
// is a per-forward DDR tensor in absolute-indexing contract (caller writes
// 3D positions [seq_len, 3] int32 starting at row `pos_offset`).
//
// Strobe masks are 3 × WARP_SIZE uint16 values encoding which dims of the
// rotary half-dim belong to the T / H / W axis. Caller computes them once
// (rpu_compute_mrope_strobe_masks) and passes them in; the launcher writes
// them to SCM at register offsets [SCM_REG_OFFSET, SCM_REG_OFFSET + 3 × WARP_SIZE).

void rpu_launch_mrope_spm_kernel(
    uint32_t x_spm_addr,                  // input SPM address (core 0 base)
    uint32_t y_spm_addr,                  // output SPM address (core 0 base; may alias x)
    c10::Half *cos_ptr,                   // DDR cos table [max_seq, head_dim/2]
    c10::Half *sin_ptr,                   // DDR sin table [max_seq, head_dim/2]
    int32_t *position_ids_ddr_ptr,        // DDR keepalive [MAX_KEEPALIVE_SEQ, 3] int32
    int64_t pos_offset,                   // absolute index into position_ids_ddr
    const std::array<std::array<uint16_t, 16>, 3> &strobe_masks,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int num_cores = 8);

// Tensor-owned counterpart for typed DDR-register census.  The three DDR
// operands retain their Tensor/Storage owners; kernel topology and register
// encoding are identical to the raw-pointer overload above.
void rpu_launch_mrope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const at::Tensor& position_ids,
    int64_t pos_offset,
    const std::array<std::array<uint16_t, 16>, 3> &strobe_masks,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int num_cores = 8);

// Compute the 3 × WARP_SIZE uint16 strobe masks (T / H / W axis selector)
// from an mrope_section [T_dim_half, H_dim_half, W_dim_half] vector. Called
// once per set_weights — masks depend only on mrope_section + head_dim.
std::array<std::array<uint16_t, 16>, 3> rpu_compute_mrope_strobe_masks(
    const std::vector<int32_t> &mrope_section,
    int32_t head_dim);

// Qwen3.5 partial M-RoPE (kernel "partial_mrope"). Rotates only the first
// rotary_dim dims per head (rotate-half within rotary_dim), passes the rest
// through. cos/sin are [seq, rotary_dim/2] DDR tables (M-RoPE interleaving baked
// in on the host side → no strobe masks / position_ids).
void rpu_launch_partial_mrope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int64_t pos_offset,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores);

// Tensor-owned variant for retained Graph DDR-register census. cos/sin are
// registered as SemanticInput at regs 0/1 and 2/3 respectively.
void rpu_launch_partial_mrope_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t pos_offset,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores);

// Qwen3.5 partial RoPE 1D (kernel "partial_rope_1d", DECODE). Same partial
// rotate-half + cos/sin DDR-table contract as partial_mrope; differs only in the
// kernel's warp tiling (grid_x = ceil(rotary_dim/2 / 512), grid_y = num_tokens).
void rpu_launch_partial_rope_1d_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    c10::Half *cos_ptr,
    c10::Half *sin_ptr,
    int64_t pos_offset,
    int64_t seq_len,
    int64_t local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores);

// =============================================================================
// 2D RoPE (Qwen3-VL R-Phase 3 vision encoder)
// =============================================================================
//
// Tables (FreqCos / FreqSin) shape:  [max_hw, head_dim/4] FP16
//   Built from `Qwen3VLVisionRotaryEmbedding(head_dim/2).forward(max_hw)` →
//   freqs (max_hw, head_dim/4) → cos/sin element-wise. SAME freq table is
//   indexed by both row_idx and col_idx (see HF rot_pos_emb).
//
// position_idx shape:  [num_tokens, 2] int16 DDR aligned to 256B
//   Per-patch (row_idx, col_idx). Built on CPU by adapter.py per forward.
//
// pos_offset is always 0 for vision encoder (the position table covers every
// token in this forward — Qwen3-VL vision is single-image bidirectional).
//
// Register layout:
//   reg[0]/[1]   position_idx DDR addr (>> 8)
//   reg[2]/[3]   FreqCos addr (DDR >>8 or SPM raw, by variant)
//   reg[4]/[5]   FreqSin addr (DDR >>8 or SPM raw, by variant)
//   reg[6]/[7]   input_hidden SPM addr (32-bit raw)
//   reg[8]/[9]   output_hidden SPM addr (32-bit raw)
//   reg[10]/[11] pos_offset (uint32)
//   reg[12]      head_dim                  (u16)
//   reg[13]      head_num                  (u16)
//   reg[14]      num_tokens                (u16)
//   reg[15]      head_dim_pad              (u16)
//   reg[64]      grid_dim_x = CeilDiv(head_dim/4, 32)   (u16)
//   reg[65]      grid_dim_y = CeilDiv(num_tokens, 128)  (u16)
//   reg[66]      grid_dim_z = 1                          (u16)

// DDR cos/sin variant — vision encoder default.
void rpu_launch_rope_2d_ddr_kernel(
    uint32_t x_spm_addr,                  // input_hidden SPM addr (core 0 base)
    uint32_t y_spm_addr,                  // output_hidden SPM addr (may alias x)
    c10::Half *cos_ptr,                   // DDR FreqCos [max_hw, head_dim/4]
    c10::Half *sin_ptr,                   // DDR FreqSin [max_hw, head_dim/4]
    int16_t *position_idx_ddr_ptr,        // DDR [num_tokens, 2] int16, aligned 256B
    int64_t pos_offset,                   // typically 0 for vision encoder
    int64_t num_tokens,
    int64_t head_num,
    int64_t head_dim,
    int64_t head_dim_pad,                 // padded inner dim (== head_dim if 64-aligned)
    int num_cores = 8);

// SPM cos/sin variant — FreqCos / FreqSin preloaded into per-core SPM. This is
// the vision-encoder perf path: the tables are loaded ONCE (emit_preload_weights)
// into Persistent SPM, so the 2×num_layers rope calls read SPM instead of
// re-reading the DDR tables every block. Register layout is identical to the DDR
// variant EXCEPT reg[2..5] hold the RAW per-core SPM byte offset of cos/sin (no
// >>8) — the same encoding already used for the x/y SPM addr regs. position_idx
// stays in DDR (reg[0]/[1] >>8).
void rpu_launch_rope_2d_spm_kernel(
    uint32_t x_spm_addr,                  // input_hidden SPM addr (core 0 base)
    uint32_t y_spm_addr,                  // output_hidden SPM addr (may alias x)
    uint32_t cos_spm_addr,                // SPM FreqCos [max_hw, head_dim/4] (per-core)
    uint32_t sin_spm_addr,                // SPM FreqSin [max_hw, head_dim/4] (per-core)
    int16_t *position_idx_ddr_ptr,        // DDR [num_tokens, 2] int16, aligned 256B
    int64_t pos_offset,                   // typically 0 for vision encoder
    int64_t num_tokens,
    int64_t head_num,
    int64_t head_dim,
    int64_t head_dim_pad,                 // padded inner dim (== head_dim if 64-aligned)
    int num_cores = 8);

// Tensor-owned counterpart for typed DDR-register census.  Only position_idx
// remains in DDR for the SPM-table variant, so it is the sole retained DDR
// register operand (reg[0]/reg[1]).
void rpu_launch_rope_2d_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    uint32_t cos_spm_addr,
    uint32_t sin_spm_addr,
    const at::Tensor& position_idx,
    int64_t pos_offset,
    int64_t num_tokens,
    int64_t head_num,
    int64_t head_dim,
    int64_t head_dim_pad,
    int num_cores = 8);

// Test-only wrapper: alloc SPM, DMA input in, run rope_2d, DMA out. Single-core.
// Used by tests/ops/test_rope_2d_isolated.py to compare kernel output against
// HF apply_rotary_pos_emb_vision reference per-lane.
at::Tensor rpu_rope_2d_test_ddr(
    const at::Tensor& input,          // [N, num_heads, head_dim] fp16 RPU
    const at::Tensor& cos,            // [max_hw, head_dim/4] fp16 RPU
    const at::Tensor& sin,            // [max_hw, head_dim/4] fp16 RPU
    const at::Tensor& position_idx);  // [N, 2] int16 RPU

// Test-only wrapper (SPM variant): alloc SPM for input/cos/sin/output, DMA all
// three inputs in, run rope_2d_spm, DMA out. Single-core. Used by
// tests/ops/test_rope_2d_isolated.py --variant spm.
at::Tensor rpu_rope_2d_test_spm(
    const at::Tensor& input,          // [N, num_heads, head_dim] fp16 RPU
    const at::Tensor& cos,            // [max_hw, head_dim/4] fp16 RPU
    const at::Tensor& sin,            // [max_hw, head_dim/4] fp16 RPU
    const at::Tensor& position_idx);  // [N, 2] int16 RPU

// =============================================================================
// fast_pos_embed_interpolate — bilinear positional-embedding interpolation.
//
// Computes, per token t, a bilinear sample of a [num_grid_H_side *
// num_grid_W_side, hidden_size] embedding table at the FRACTIONAL grid
// coordinate (h, w) = grid_mapping[pos_offset + t]:
//
//   h0 = floor(h), w0 = floor(w)
//   h1 = min(h0+1, num_grid_H_side-1), w1 = min(w0+1, num_grid_W_side-1)
//   dh = h - h0, dw = w - w0
//   idx(i, j) = i * num_grid_W_side + j          <-- row stride is W_side
//   out[t] = (1-dh)(1-dw)*E[idx(h0,w0)] + (1-dh)*dw*E[idx(h0,w1)]
//          +    dh *(1-dw)*E[idx(h1,w0)] +    dh *dw*E[idx(h1,w1)]
//
// The kernel does the whole bilinear itself: grid_mapping holds raw fractional
// coordinates, NOT precomputed indices or weights.
//
// Two semantics that are easy to get wrong:
//   1. pos_offset indexes grid_mapping ONLY. The output is written densely from
//      the output base — it is NOT shifted by pos_offset.
//   2. grid_mapping must ALREADY be permuted into spatial-merge block order by
//      the caller. The kernel does no reordering (cf. HF's `reorder` gather in
//      transformers/vision_utils.py get_vision_bilinear_indices_and_weights).
//
// Register layout:
//   reg[0]/[1]   grid_mapping base       (u32; DDR: addr/256, SPM: raw bytes)
//   reg[2]/[3]   pos_embed_weight base   (u32; DDR: addr/256, SPM: raw bytes)
//   reg[4]/[5]   patch_pos_embeds output SPM base (u32, raw bytes)
//   reg[6]/[7]   pos_offset              (u32, token offset into grid_mapping)
//   reg[8]       hidden_size             (i16)
//   reg[9]       num_grid_H_side         (i16; used ONLY for the ceil clamp)
//   reg[10]      num_grid_W_side         (i16; row stride AND ceil clamp)
//   reg[11]      num_tokens              (i16)
//   reg[64]      grid_dim_x = CeilDiv(hidden_size, 2048)  (u16)
//   reg[65]      grid_dim_y = CeilDiv(num_tokens, 8)      (u16)
//   reg[66]      grid_dim_z = 1                           (u16)
//
// The kernel needs no workspace/scratch.
//
// Both inputs must live in the SAME space — all DDR or all SPM. The output is
// always SPM, and every core computes the same full-width result redundantly.

// DDR-input variant.
void rpu_launch_fast_pos_embed_interp_ddr_kernel(
    void *grid_mapping_ptr,        // [total_tokens, 2] fp16 DDR, 256B-aligned
    void *pos_embed_weight_ptr,    // [H_side*W_side, hidden] fp16 DDR, 256B-aligned
    uint32_t out_spm_addr,         // [num_tokens, hidden_size] fp16 SPM output
    int64_t pos_offset,            // token offset into grid_mapping
    int64_t hidden_size,
    int64_t num_grid_h_side,
    int64_t num_grid_w_side,
    int64_t num_tokens,
    int num_cores = 8);

// =============================================================================
// all_gather_multi_core — axis=1 (column) concat across cores.
//
// Each core holds a packed [n, chunk_elems] shard; afterwards EVERY core holds
// the full [n, chunk_elems*num_cores], core j's shard occupying columns
// [j*chunk_elems, (j+1)*chunk_elems) of every row.
//
// Pairs with COLUMN-parallel matmul (partition=1): per-core outputs are
// different columns of one answer, so they are concatenated. Contrast
// rpu_launch_all_reduce_sum_residual_kernel, which pairs with ROW-parallel
// (partition=0): per-core outputs are full-width PARTIAL SUMS, so they are
// added. Picking the wrong one is silently wrong, not a crash.
//
// For `out = concat(shards) + residual`, follow this with
// rpu_launch_eltwise_binary_spm_kernel(out, residual, out, n*chunk*tp,
//                                      ValuOpType::ADD, 1.0, num_cores).
//
// Register layout:
//   reg[0]      core_num                                   (u16)   both
//   reg[1]      n = row count                              (u16)   both
//   reg[2]/[3]  dst row stride bytes                       (u32)   both
//   reg[4]      line_byte_size  = 4096                     multi_core only
//   reg[5]      block_byte_size = 28672                    multi_core only
//   reg[6]      max_chunk_size_v16 = ceil(chunk/32)        little_chunk only
//   reg[7]      n_per_thd = 64/reg[6]  (must be >= 1)      little_chunk only
//   reg[64..66] grid = (warp_num=8, 1, 1)                  both
//   scm[0+j]/[8+j]   chunk bytes                           (u32)   both
//   scm[16+j]        block count = ceil(chunk / 28672)     multi_core only
//   scm[24+j]/[32+j] src row stride bytes                  (u32)   both
//   scm[40+j]/[48+j] core j's shard base                   (u32)   both
//   scm[56+j*16+k]/[64+j*16+k] dst table: core j's column base on core k
//
// reg[4]/[5] are VLM staging granularity, NOT layout, and are NOT dtype-scaled:
// they are always 4096/28672
// whatever the tensor dtype. The striding lives in reg[1] + reg[2]/[3] +
// scm[24+j].
//
// Not exposed: non-uniform per-core chunks and strided sources.
enum class RpuAllGatherSchedule : int64_t {
    LITTLE_CHUNK = 1,
    MULTI_CORE = 2,
};

constexpr RpuAllGatherSchedule rpu_resolve_all_gather_schedule(
    int64_t chunk_elems, int64_t dwidth, bool force_multi_core = false) {
    return !force_multi_core && chunk_elems * dwidth <= 1024
        ? RpuAllGatherSchedule::LITTLE_CHUNK
        : RpuAllGatherSchedule::MULTI_CORE;
}

void rpu_launch_all_gather_spm_kernel(
    uint32_t input_spm_addr,   // [n, chunk_elems] SPM, packed, per-core shard
    uint32_t output_spm_addr,  // [n, chunk_elems*num_cores] SPM, replicated
    int64_t n,                 // row count
    int64_t chunk_elems,       // elements per core per row (e.g. hidden/tp)
    int64_t dwidth,             // element size in bytes: 2 = fp16, 4 = int32/fp32
    int num_cores,
    RpuAllGatherSchedule schedule);

// Each core extracts its own contiguous shard from a replicated full tensor.
// This is a local SPM copy; the launcher only reuses all_gather_multi_core with
// its ring disabled.
void rpu_launch_spm_local_shard_copy_kernel(
    uint32_t spm_src_unified,
    int64_t elements_per_core,
    int64_t core_stride_bytes,
    uint32_t spm_dst_unified,
    int num_cores);

// Qwen3.5 GDN: row-wise L2 norm (kernel "l2norm", in the matching runtime asset).
void rpu_launch_l2norm_spm_kernel(uint32_t in_spm_addr, uint32_t out_spm_addr,
                                  int64_t M, int64_t D, double eps,
                                  int num_cores = 1);

// Qwen3.5 GDN prefill chunk: cumulative sum along last dim (kernel
// "cumsum_reduceC_tileN", in the matching runtime asset). Caller supplies a
// workspace SPM buffer of min(64,C)*17*16 halfs (kernel spec, N-independent).
void rpu_launch_cumsum_spm_kernel(uint32_t in_spm_addr, uint32_t out_spm_addr,
                                  uint32_t ws_spm_addr, int64_t N, int64_t C,
                                  int num_cores = 1);

// Qwen3.5 GDN prefill chunk: (I - M)^-1 for strict lower-tri 64x64 blocks of M
// (kernel "unit_tril_inv", in the matching runtime asset). C = 64 fixed, no
// workspace; num_blocks = numel/(64*64).
void rpu_launch_unit_tril_inv_spm_kernel(uint32_t in_spm_addr, uint32_t out_spm_addr,
                                         int64_t num_blocks, int num_cores = 1);

// SPM-resident 3D tile (reuses tile_general_smallC/largeC). GDN GQA-expands q/k via
// the [N,1,Dk] repeat-middle trick (= repeat_interleave on heads). See rpu_tile.cpp.
void rpu_launch_tile_spm_kernel(uint32_t input_spm_addr, uint32_t output_spm_addr,
                                int64_t dim0, int64_t dim1, int64_t dim2,
                                int64_t repeat0, int64_t repeat1, int64_t repeat2,
                                int num_cores = 1);

// BMM operand orientation (mirrors rhino gemmMode -> gemm kernel suffix). out is
// always [batch,M,N]; M/N/K are the LOGICAL dims (output M×N, contraction K), same
// for every mode — only the operand storage layout + kernel suffix differ:
//   RowByCol (nt_): A[M,K] @ B[K,N]      (standard A@B)
//   RowByRow (nn_): A[M,K] @ B[N,K]ᵀ    (= A@Bᵀ)
//   ColByRow (tn_): A[K,M]ᵀ @ B[N,K]ᵀ   (= Aᵀ@Bᵀ)
//   ColByCol (tt_): A[K,M]ᵀ @ B[K,N]    (= Aᵀ@B)
// The nn/nt/tn/tt gemm_fp16_spm kernels all ship in the oplib; the launcher just
// swaps the suffix. Tail/stride regs are mode-independent for contiguous packed,
// 16-aligned operands (the GDN chunk case).
enum class BmmMode { RowByCol = 0, RowByRow = 1, ColByRow = 2, ColByCol = 3 };

// BMM SPM (fused-graph): out[batch,M,N] = A @ B (orientation per `mode`), operands in
// SPM. Same gemm_fp16_spm kernel as the eager rpu_launch_bmm_kernel; no-C (raw A·B).
// `mode` defaults to RowByCol (nt_) so existing call sites are unaffected.
void rpu_launch_bmm_spm_kernel(uint32_t a_spm_addr, uint32_t b_spm_addr,
                               uint32_t out_spm_addr, int64_t M, int64_t N,
                               int64_t K, int64_t batch, int num_cores = 1,
                               BmmMode mode = BmmMode::RowByCol);

// Slice SPM (fused-graph): out[output_shape] = in[input_shape] sliced at begins
// (step 1). Reuses slice_single_axis / slice_last_dim_axis (same as insert_slice).
void rpu_launch_slice_spm_kernel(uint32_t input_spm_addr, uint32_t output_spm_addr,
                                 const std::vector<int64_t>& input_shape,
                                 const std::vector<int64_t>& output_shape,
                                 const std::vector<int64_t>& begins,
                                 int num_cores = 1);

// Permute3d SPM (fused-graph): out = in.permute(perm) for a 3D tensor [b,n,c].
// Reuses the transpose_{nbc,cbn,bcn,cnb,ncb}_c16 family (perm{1,0,2} = transpose_nbc).
void rpu_launch_permute3d_spm_kernel(uint32_t input_spm_addr, uint32_t output_spm_addr,
                                     int64_t b, int64_t n, int64_t c,
                                     const std::vector<int64_t>& perm, int num_cores = 1);

// Permute4d SPM (fused-graph), intentionally limited to 0213:
// [q,b,n,c] -> [q,n,b,c]. Reuses the shipped transpose_qnbc_c16 kernel.
void rpu_launch_permute4d_0213_spm_kernel(
    uint32_t input_spm_addr, uint32_t output_spm_addr,
    int64_t q, int64_t b, int64_t n, int64_t c, int num_cores = 1);

// Qwen3.5 GDN: causal depthwise conv1d decode-update (kernel in the matching runtime asset).
void rpu_launch_fla_conv1d_spm_kernel(uint32_t cs_spm_addr, uint32_t x_spm_addr,
                                      c10::Half* weight_ddr, int64_t B,
                                      int64_t local_d, int64_t Kc, int num_cores);

// Qwen3.5 GDN prefill: causal dwconv1d over a chunk. The input
// pad=[cs_init(3);x(L)] [B,3+L,D] remains intact; y[B,L,D] and cs_out[B,3,D]
// are written to separate buffers. N = 8 if L>=32 else 1 segments.
void rpu_launch_fla_conv1d_prefill_noinplace_spm_kernel(
    uint32_t pad_spm_addr, uint32_t out_spm_addr, uint32_t cs_out_spm_addr,
    c10::Half* weight_ddr, int64_t B, int64_t local_d, int64_t Kc, int64_t L,
    int num_cores);

// Qwen3.5 GDN: batched multi-head recurrent step + on-device gating (decode).
// q/k [H,Dk], v [H,Dv], A_log/a/b/dt_bias [H], state_in [H,Dk,Dv]. Computes
// beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias), g_exp=exp(g), l2norm(q,k),
// q*=1/sqrt(Dk) on device, then the delta rule per head. Returns (out[H,Dv],
// state_out[H,Dk,Dv]).

// Qwen3.5 GDN: validated multi-head recurrent + gating sequence on ABSOLUTE SPM
// addresses (reused by the isolated test and the fused-graph build_gdn_mixer).
void rpu_emit_gdn_recurrent_multihead_seq(
    uint32_t a_q, uint32_t a_k, uint32_t a_v, uint32_t a_state,
    uint32_t a_p, uint32_t a_scratch, uint32_t a_delta, uint32_t a_zero,
    uint32_t a_Al, uint32_t a_a, uint32_t a_b, uint32_t a_dt,
    uint32_t a_gexp, uint32_t a_beta, uint32_t a_gs,
    uint32_t a_out, int64_t H, int64_t Dk, int64_t Dv,
    int num_cores = 1);


// Isolated partial_mrope (prefill) test harness (single-core).
at::Tensor rpu_partial_mrope_test(
    const at::Tensor& input,
    const at::Tensor& cos_il,
    const at::Tensor& sin_il,
    int64_t pos_offset);

// Wall-OSS on-device embed/cast helpers.
at::Tensor rpu_gather_embedding(const at::Tensor& vocab, const at::Tensor& ids);
at::Tensor rpu_assemble_inputs_embeds(const at::Tensor& vocab, const at::Tensor& ids,
                                      const at::Tensor& merged, int64_t img_start);
at::Tensor rpu_assemble_inputs_embeds_rev(const at::Tensor& vocab, const at::Tensor& ids,
                                          const at::Tensor& merged_window,
                                          const at::Tensor& rev_idx, const at::Tensor& runs);
at::Tensor rpu_cast_uint8_fp16(const at::Tensor& input);
void rpu_cast_uint8_fp16_into(const at::Tensor& input, at::Tensor& output);

// =============================================================================
// Eltwise Binary Op Wrappers
// =============================================================================

at::Tensor rpu_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                             std::optional<c10::ScalarType> dtype_opt,
                             std::optional<c10::Layout> layout_opt,
                             std::optional<c10::Device> device_opt,
                             std::optional<bool> pin_memory_opt);
std::vector<int64_t> compute_contiguous_strides(c10::IntArrayRef sizes);

at::Tensor binary_op_common_wrapper(
    const at::Tensor &a,
    const at::Tensor &b,
    const c10::Scalar &alpha,
    ValuOpType op_type,
    bool is_commutative);

at::Tensor& binary_op_inplace_wrapper(
    at::Tensor &self,
    const at::Tensor &other,
    const c10::Scalar &alpha,
    ValuOpType op_type);

// add
at::Tensor rpu_add(const at::Tensor &a, const at::Tensor &b, const c10::Scalar &alpha);
at::Tensor &rpu_add_(at::Tensor &self, const at::Tensor &other, const c10::Scalar &alpha);

// sub
at::Tensor rpu_sub(const at::Tensor &a, const at::Tensor &b, const c10::Scalar &alpha);
at::Tensor &rpu_sub_(at::Tensor &self, const at::Tensor &other, const c10::Scalar &alpha);
at::Tensor &rpu_sub_scalar_(at::Tensor &self, const c10::Scalar &other, const c10::Scalar &alpha);

// div
at::Tensor rpu_div(const at::Tensor &a, const at::Tensor &b);
at::Tensor rpu_div_scalar(const at::Tensor &a, const c10::Scalar &b);
at::Tensor &rpu_div_(at::Tensor &self, const at::Tensor &other);

// mul
at::Tensor rpu_mul(const at::Tensor &a, const at::Tensor &b);
at::Tensor rpu_mul_scalar(const at::Tensor &a, const c10::Scalar &b);
at::Tensor &rpu_mul_(at::Tensor &self, const at::Tensor &other);
at::Tensor &rpu_mul_scalar_(at::Tensor &self, const c10::Scalar &other);

// neg
at::Tensor rpu_neg(const at::Tensor &self);

// pow
at::Tensor rpu_pow_tensor_scalar(const at::Tensor &self, const c10::Scalar &exponent);
at::Tensor rpu_pow_tensor_tensor(const at::Tensor &self, const at::Tensor &exponent);
at::Tensor rpu_pow_scalar_tensor(const c10::Scalar &self, const at::Tensor &exponent);

at::Tensor rpu_tile(const at::Tensor &a, const at::Tensor &b);

// concat
at::Tensor rpu_cat(const c10::IListRef<at::Tensor>& tensors, int64_t dim);
at::Tensor& rpu_cat_out(const c10::IListRef<at::Tensor>& tensors,
                        int64_t dim, at::Tensor& out);

// reduce
void rpu_launch_reduce_mean_last_dim_kernel(const at::Tensor &input,
                                             at::Tensor &output,
                                             int64_t axis);
at::Tensor rpu_mean_dim(const at::Tensor &self, at::OptionalIntArrayRef dim,
                        bool keepdim, c10::optional<at::ScalarType> dtype);

// =============================================================================
// RMSNorm
// =============================================================================

// Certified payload families available to one immutable model handle.
// Capability is deliberately separate from the exact, append-only route
// selector recorded in the physical manifest.
enum RpuRmsNormCapability : uint32_t {
    RPU_RMSNORM_CAP_BASE = 1u << 0,
    RPU_RMSNORM_CAP_V16 = 1u << 1,
    RPU_RMSNORM_CAP_V32 = 1u << 2,
};

inline constexpr uint32_t RPU_RMSNORM_CAP_ALL =
    RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V16 | RPU_RMSNORM_CAP_V32;

// Operator-level conformance ceiling for ordinary FP16 direct-SPM RMSNorm.
// The fixed-grid contract covers V16 and V32; actual payload
// availability is intersected while installing each handle. Shape eligibility
// stays in the resolver below.
inline constexpr uint32_t RPU_RMSNORM_CAP_GENERIC_DIRECT_SPM =
    RPU_RMSNORM_CAP_ALL;

// The current payload ABI stores C and the FP16 row stride in uint16 SCM
// registers. BASE also stores its row grid directly in uint16. Vector routes
// can represent more rows because their fixed grid is eight and reg[3] holds
// rows/8.
inline constexpr int64_t RPU_RMSNORM_SCM_U16_MAX = 65535;
inline constexpr int64_t RPU_RMSNORM_MAX_COLS =
    RPU_RMSNORM_SCM_U16_MAX / 2;

enum class RpuRmsNormSpmRoute : int64_t {
    BASE = 1,
    // Legacy selectors 2/3 are retained because route IDs are append-only.
    // New manifests resolve the same payload families to canonical 5/6.
    V16 = 2,
    V32 = 3,
    QWEN3VL_V64 = 4,
    V16_GRID8 = 5,
    V32_GRID8 = 6,
    // Explicit high-precision opt-in. Existing selector semantics are stable.
    NEWTON = 7,
    NEWTON_V16_GRID8 = 8,
    NEWTON_V32_GRID8 = 9,
};

enum class RpuMixedRmsNormSpmRoute : int64_t {
    V16 = 1,
    V32 = 2,
};

inline bool rpu_rmsnorm_capability_valid(uint32_t capability) {
    return (capability & RPU_RMSNORM_CAP_BASE) != 0 &&
        (capability & ~RPU_RMSNORM_CAP_ALL) == 0;
}

inline RpuRmsNormSpmRoute rpu_resolve_rmsnorm_spm_route(
    uint32_t capability, int64_t rows, int64_t cols) {
    TORCH_CHECK(
        rpu_rmsnorm_capability_valid(capability),
        "RMSNorm capability must include BASE and contain only BASE/V16/V32; "
        "got ", capability);
    TORCH_CHECK(rows > 0, "RMSNorm rows must be positive, got ", rows);
    TORCH_CHECK(cols > 0, "RMSNorm cols must be positive, got ", cols);
    TORCH_CHECK(
        cols <= RPU_RMSNORM_MAX_COLS,
        "RMSNorm width and FP16 row stride must fit uint16 SCM registers; got ",
        cols, " columns");
    const bool canonical_grid8_eligible =
        cols <= 8192 && rows / 8 <= RPU_RMSNORM_SCM_U16_MAX;
    if (canonical_grid8_eligible &&
        (capability & RPU_RMSNORM_CAP_V32) != 0 && rows % 32 == 0) {
        return RpuRmsNormSpmRoute::V32_GRID8;
    }
    if (canonical_grid8_eligible &&
        (capability & RPU_RMSNORM_CAP_V16) != 0 && rows % 16 == 0) {
        return RpuRmsNormSpmRoute::V16_GRID8;
    }
    TORCH_CHECK(
        rows <= RPU_RMSNORM_SCM_U16_MAX,
        "RMSNorm BASE fallback row grid is not representable as uint16; got ",
        rows, " rows");
    return RpuRmsNormSpmRoute::BASE;
}

// Cold installation separately requires all payloads.
inline RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route(
    int64_t rows, int64_t cols) {
    TORCH_CHECK(rows > 0, "Newton RMSNorm rows must be positive, got ", rows);
    TORCH_CHECK(cols > 0 && cols % 16 == 0 && cols <= RPU_RMSNORM_MAX_COLS,
                "Newton RMSNorm width must be 16-aligned and its FP16 row "
                "stride must fit uint16 SCM registers; got ", cols);
    if (cols <= 8192 && rows / 8 <= RPU_RMSNORM_SCM_U16_MAX) {
        if (rows % 32 == 0) return RpuRmsNormSpmRoute::NEWTON_V32_GRID8;
        if (rows % 16 == 0) return RpuRmsNormSpmRoute::NEWTON_V16_GRID8;
    }
    TORCH_CHECK(rows <= RPU_RMSNORM_SCM_U16_MAX,
                "Newton RMSNorm BASE row grid must fit uint16; got ", rows);
    return RpuRmsNormSpmRoute::NEWTON;
}

// Immutable per-handle contract.  `capability` is the intersection of the
// operator-level certified ceiling and payloads actually loaded at runtime;
// it is snapshotted while installing the handle and then reused by both the
// physical-manifest producer and consumer.
struct RpuRmsNormSpmContract {
    uint32_t capability = RPU_RMSNORM_CAP_BASE;

    RpuRmsNormSpmRoute resolve(int64_t rows, int64_t cols) const {
        return rpu_resolve_rmsnorm_spm_route(capability, rows, cols);
    }

    std::vector<int64_t> manifest_arguments(
        int64_t rows, int64_t cols) const {
        return {rows, cols, static_cast<int64_t>(capability)};
    }

    bool has_vector_payload() const {
        return capability != RPU_RMSNORM_CAP_BASE;
    }
};

RpuRmsNormSpmContract rpu_snapshot_rmsnorm_spm_contract(
    uint32_t certified_capability =
        RPU_RMSNORM_CAP_GENERIC_DIRECT_SPM);

void rpu_launch_rmsnorm_kernel(const at::Tensor &input, at::Tensor &output,
                               const at::Tensor &weight, double eps);
at::Tensor rpu_rmsnorm(const at::Tensor &input, c10::IntArrayRef normalized_shape,
                       const at::Tensor &weight, double eps);

// Exact Qwen3-VL-4B W8 decode: Q4/K1/D128, eight cores, in-place Q/K.
// BASE RMSNorm BF16 intermediates and FP16 retirement precede original M-RoPE.
// Position IDs are a stable, caller-refreshed [rows,3] int32 DDR keepalive.
void rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_spm_kernel(
    uint32_t q_spm_addr, uint32_t k_spm_addr,
    uint32_t q_gamma_spm_addr, uint32_t k_gamma_spm_addr,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& position_ids, int64_t position, double eps);

// Same exact QK leaf plus V2 K/V insertion, one physical row, eight cores.
// All five SPM operands are absolute addresses. Packed FP16 cache owners use
// [1, capacity/16, 1, 8, 8, 16, 16]; only the selected physical row is written.
void rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_kv_insert_spm_kernel(
    uint32_t q_spm_addr, uint32_t k_spm_addr, uint32_t v_spm_addr,
    uint32_t q_gamma_spm_addr, uint32_t k_gamma_spm_addr,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& position_ids, const at::Tensor& k_cache,
    const at::Tensor& v_cache, int64_t position, double eps);

// RMSNorm SPM (direct address)
void rpu_launch_rmsnorm_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    uint32_t weight_spm_addr,
    int64_t M,
    int64_t C,
    double eps,
    RpuRmsNormSpmRoute route = RpuRmsNormSpmRoute::BASE,
    int num_cores = 8);

// Qwen3.5 sparse-MoE routing and expert kernels. SPM addresses are unified
// byte addresses; each participating core uses the same local offset.
void rpu_launch_softmax_c16_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    int64_t n,
    int64_t c,
    int num_cores = 1);

enum class GroupedParallelLinearWeightMode : int64_t {
    UINT8 = 0,
    INT8 = 1,
    FP16 = 2,
    FP8_E2M5 = 4,
    FP8_E4M3 = 5,
};

void rpu_launch_grouped_parallel_linear_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& packed_weight,
    uint32_t m_sizes_spm_addr,
    const c10::optional<at::Tensor>& scale,
    uint32_t output_spm_addr,
    uint32_t bias_spm_addr,
    int64_t local_n,
    int64_t local_k,
    int64_t group,
    bool is_col_parallel,
    int num_cores,
    int64_t weight_dtype_mode = -1,
    bool linear_acc32 = false);

// Conformance-only immediate wrapper; not a retained Graph entry point.
at::Tensor rpu_grouped_parallel_linear_test(
    const at::Tensor& input_per_core, const at::Tensor& packed_weight,
    const at::Tensor& m_sizes, const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& bias_per_core, int64_t local_n,
    int64_t local_k, bool is_col_parallel, int64_t num_cores,
    int64_t weight_dtype_mode, bool linear_acc32);

void rpu_launch_moe_topk_scalar_fp16_spm_kernel(
    uint32_t input_key_spm_addr,
    uint32_t input_value_spm_addr,
    uint32_t output_key_spm_addr,
    uint32_t output_value_spm_addr,
    int64_t n,
    int64_t c,
    int64_t k,
    bool descending,
    bool cvt_key,
    int num_cores = 1);

void rpu_launch_moe_topk_by_kvsort_fp16_spm_kernel(
    uint32_t input_key_spm_addr,
    uint32_t input_value_spm_addr,
    uint32_t output_key_spm_addr,
    uint32_t output_value_spm_addr,
    uint32_t workspace_key_spm_addr,
    uint32_t workspace_value_spm_addr,
    int64_t n,
    int64_t c,
    int64_t k,
    bool descending,
    int num_cores = 1);

void rpu_launch_moe_bincount_scalar_spm_kernel(
    uint32_t input_key_spm_addr,
    uint32_t output_key_spm_addr,
    uint32_t output_bin_spm_addr,
    int64_t c,
    int64_t maxk,
    bool cvt_key,
    bool output_cvt_key,
    int num_cores = 1);

void rpu_launch_moe_topk_prob_norm_fp16_spm_kernel(
    uint32_t token_prob_spm_addr,
    uint32_t topk_prob_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t k_threshold,
    int num_cores = 1);

void rpu_launch_cast_uint16_int32_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_elements,
    int num_cores = 1);

void rpu_launch_gather_fp16_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t indices_spm_addr,
    uint32_t output_spm_addr,
    int64_t input_rows,
    int64_t input_cols,
    int64_t axis,
    int64_t indices_num,
    int num_cores = 1);

void rpu_launch_moe_gather_mul_v2_fp16_spm_kernel(
    uint32_t token_spm_addr,
    uint32_t scale_spm_addr,
    uint32_t token_indices_spm_addr,
    uint32_t expert_indices_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_tokens,
    int64_t num_experts,
    int64_t num_indices,
    int64_t hidden_dims,
    int num_cores = 1);

void rpu_launch_scatternd_add_u16_fp16_spm_kernel(
    uint32_t data_spm_addr,
    uint32_t indices_spm_addr,
    uint32_t updates_spm_addr,
    int64_t data_rows,
    int64_t num_indices,
    int64_t row_size,
    int num_cores = 1);
// Pi0.5 H1024 FP16 RMSNorm followed by separately-rounded FP16 shift.
// All addresses are absolute per-core SPM; supports M1..64 and 1 or 8 cores.
void rpu_launch_pi05_adarms_norm_shift_spm_kernel(
    uint32_t input_spm, uint32_t output_spm, uint32_t scale_spm,
    uint32_t shift_spm, int64_t rows, double eps, int cores = 8);

// FP16 broadcast gate*x, rounded before residual ADD; absolute local SPM.
void rpu_launch_pi05_gated_residual_spm_kernel(
    uint32_t input_spm, uint32_t residual_spm, uint32_t gate_spm,
    uint32_t output_spm, int64_t rows, int cores = 8);


// SwiGLU activation fusion: out = silu(x) * y, flat per-core elementwise.
// SPM addresses are absolute core-0 bases.
void rpu_launch_silu_mul_spm_kernel(
    uint32_t x_spm_addr,
    uint32_t y_spm_addr,
    uint32_t out_spm_addr,
    int64_t num_elements,
    int num_cores = 8);

// Exact Pi Prefill M400/H2048/I16384, TP8 W8A16. The generated N80
// Gate/Up GEMMs retain their tiling; GeGLU replaces the Gate output.
void rpu_launch_pi05_gate_up_geglu_w8a16_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t output_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale);

// Exact Pi Denoise M50/H1024/I4096, TP8 NVFP4 striped-v2. The historical
// launcher retains the ACC32 producer; ACC16 has a separate payload identity.
void rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_m50_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t output, const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer);

void rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(
    uint32_t input, const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t output, const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_tensor_scale, uint32_t up_tensor_scale, uint16_t layer);

void rpu_launch_pi05_denoise_gate_up_geglu_wint4a16_pgrp_m50n64_kernel(
    uint32_t input_spm_addr, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output_spm_addr,
    const at::Tensor& gate_scale, const at::Tensor& up_scale);

void rpu_launch_pi05_denoise_gate_up_geglu_wint4a16_pgrp_m50_kernel(
    uint32_t input_spm_addr, const at::Tensor& gate_weight,
    const at::Tensor& up_weight, uint32_t output_spm_addr,
    const at::Tensor& gate_scale, const at::Tensor& up_scale);

void rpu_launch_pi05_denoise_gate_up_geglu_w8a16_m50_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t output_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale);

// Exact Pi Prefill P800/C400x2 Gate+Up+GeGLU. One Gate and one Up weight
// sweep serve two disjoint full SPM inputs; the outputs remain Down-ready.
void rpu_launch_pi05_prefill_gate_up_geglu_weight_outer_w8a16_c400x2_kernel(
    uint32_t input0_spm_addr,
    uint32_t input1_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t output0_spm_addr,
    uint32_t output1_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale, int64_t rows = 400);

// Experimental P800 full-K INT8 GateUp; FP32 scales are exact promotions of
// the installed FP16 W8 scales. Two independent C400 outputs feed original Down.
// P544/C272 online quantization. The residual lives in a separate owner.
void rpu_launch_pi05_prefill_gate_up_geglu_online_w8a8_c272_kernel(
    uint32_t x_stage, uint32_t a8, uint32_t output,
    const at::Tensor& gate_weight, const at::Tensor& up_weight,
    const at::Tensor& gate_scale, const at::Tensor& up_scale);

void rpu_launch_pi05_prefill_gate_up_geglu_w8a8_split_kernel(
    uint32_t a8, uint32_t row_scale, uint32_t stage,
    const at::Tensor& gate_weight, const at::Tensor& up_weight,
    uint32_t output0, uint32_t output1,
    const at::Tensor& gate_scale, const at::Tensor& up_scale, int64_t rows = 400);

// =============================================================================
// LayerNorm SPM (direct address)
// =============================================================================

void rpu_launch_layernorm_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    uint32_t gamma_spm_addr,
    uint32_t beta_spm_addr,
    int64_t M,
    int64_t C,
    double eps,
    bool has_skip = false,
    uint32_t skip_spm_addr = 0,
    int num_cores = 8);

// LayerNorm bf16 SPM kernel — same launcher signature as the fp16 SPM
// LayerNorm, with input/output/gamma/beta SPM data still treated as fp16
// (so it's a drop-in replacement at the call site).
//
// Data-type contract:
//   - All SPM addresses point at fp16 data (NOT bf16).
//   - `eps` is encoded as fp16 inside this launcher.
//   - `1/lc` stays fp32.
void rpu_launch_layernorm_bf16_spm_kernel(
    uint32_t input_spm_addr,
    uint32_t output_spm_addr,
    uint32_t gamma_spm_addr,
    uint32_t beta_spm_addr,
    int64_t M,
    int64_t C,
    double eps,
    bool has_skip = false,
    uint32_t skip_spm_addr = 0,
    int num_cores = 8);

// =============================================================================
// LLaMA KV-Cache Operations
// =============================================================================

void rpu_launch_llama_insert_vcache(
    const at::Tensor &v_input,
    at::Tensor &v_cache,
    int64_t position);

void rpu_launch_llama_insert_kcache(
    const at::Tensor &k_input,
    at::Tensor &k_cache,
    int64_t position);

// Arena-staged variants (RhinoVLA): stage external DDR input through the repo
// SPM_ALLOC arena (not SDK LocalSPM) + unified swizzle kernel. Caller owns
// reset_temporary after the per-layer prefix loop.
void rpu_launch_llama_insert_vcache_arena(
    const at::Tensor &v_input,
    at::Tensor &v_cache,
    int64_t position,
    at::IntArrayRef planned_route_arguments);

void rpu_launch_llama_insert_kcache_arena(
    const at::Tensor &k_input,
    at::Tensor &k_cache,
    int64_t position,
    at::IntArrayRef planned_route_arguments);

void rpu_launch_llama_insert_kvcache_arena(
    const at::Tensor &k_input,
    const at::Tensor &v_input,
    at::Tensor &k_cache,
    at::Tensor &v_cache,
    int64_t position,
    at::IntArrayRef planned_route_arguments);

std::vector<int64_t> rpu_resolve_llama_kvcache_arena_plan(
    int64_t position,
    int64_t rows,
    int64_t num_kv_heads,
    int64_t head_dim);

// KV cache insert from SPM (direct address)
//
// `cache_batch_offset_elems` selects the batch slot of a multi-sequence cache.
// The 7-D cache is [batch, sKeyVx, ...] contiguous, so slot b begins at
// b * (numel / batch) elements. It is passed explicitly rather than handing the
// launcher a `select(0, b)` view so the DDR flush stays on the allocation base
// and nothing assumes how the SDK resolves an interior pointer.
// Default 0 = single-sequence cache; every existing call site is unchanged.
//
// `spm_rows`（默认 0 = 不启用）：调用方声明它那块 SPM K/V buffer **实际有多少行**。
// 当 `spm_rows >= ceil16(seq_len)` 且 `position % 16 == 0` 时，launcher 把插入长度
// 补到 16 的倍数，从而用**一次** v16 取代 "v16 主体 + v2 尾巴" 两次发射。
// 多插进去的 `[seq_len, ceil16(seq_len))` 行是 SPM 里的既有内容（未定义），落到
// cache 的 `[position+seq_len, position+ceil16(seq_len))` —— 调用方必须保证
// 这些位置**永不被读**（SDPA 传的 `kv_seq_len` 恒 ≤ position+seq_len）。
// ⚠️ 这不是"新写了脏数据"：那一段本来就是上一帧留下的陈旧字节，kernel 一直要
// 靠 `kv_seq_len` 把它挡在外面。⇒ 输出**逐位不变**。
// Fixed Pi0.5 K-only RoPE + PAD16 V16 insertion. k_off is a KV-boundary
// SPM offset; K[50:64] must already be zero in the full64-row allocation.
void rpu_launch_pi05_k_rope_insert_spm_kernel(
    uint32_t k_off, const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& k_cache, int64_t logical_position,
    int64_t spm_rows, const KvInsertSegmentPlan& plan);

void rpu_launch_insert_kcache_spm_unified_with_plan(
    at::Tensor &k_cache,
    uint32_t k_off,
    int64_t num_kv_heads, int64_t head_dim,
    int num_cores,
    int64_t cache_batch_offset_elems,
    int64_t spm_rows,
    const KvInsertSegmentPlan& plan,
    bool trace = false);

void rpu_launch_insert_vcache_spm_unified_with_plan(
    at::Tensor &v_cache,
    uint32_t v_off,
    int64_t num_kv_heads, int64_t head_dim,
    int num_cores,
    int64_t cache_batch_offset_elems,
    int64_t spm_rows,
    const KvInsertSegmentPlan& plan,
    bool trace = false);

void rpu_launch_insert_kvcache_spm_unified_with_plan(
    at::Tensor &k_cache, at::Tensor &v_cache,
    uint32_t k_off, uint32_t v_off,
    int64_t num_kv_heads, int64_t head_dim,
    int num_cores,
    int64_t k_cache_batch_offset_elems,
    int64_t v_cache_batch_offset_elems,
    int64_t spm_rows,
    const KvInsertSegmentPlan& plan,
    bool trace = false);

void rpu_launch_insert_kcache_spm_unified(
    at::Tensor &k_cache, int64_t position,
    uint32_t k_off,
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores = 8,
    int64_t cache_batch_offset_elems = 0,
    bool allow_non8_v16 = false,
    bool allow_hybrid_v16 = false,
    int64_t spm_rows = 0);

void rpu_launch_insert_vcache_spm_unified(
    at::Tensor &v_cache, int64_t position,
    uint32_t v_off,
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores = 8,
    int64_t cache_batch_offset_elems = 0,
    bool allow_non8_v16 = false,
    bool allow_hybrid_v16 = false,
    int64_t spm_rows = 0);

void rpu_launch_insert_kvcache_spm_unified(
    at::Tensor &k_cache, at::Tensor &v_cache,
    int64_t position,
    uint32_t k_off, uint32_t v_off,
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores = 8,
    int64_t k_cache_batch_offset_elems = 0,
    int64_t v_cache_batch_offset_elems = 0,
    bool allow_non8_v16 = false,
    bool allow_hybrid_v16 = false,
    int64_t spm_rows = 0);

// =============================================================================
// SDPA Mask Preparation (split into prepare + DMA for cross-layer caching)
// =============================================================================

struct PreparedMask {
    at::Tensor ddr_tensor;  // padded (if needed), on kPrivateUse1
    int mask_type = 0;      // MASK_NONE=0, MASK_LTM=1, MASK_2D=4
};

// Model-owned stable DDR slots used by Graph-captured mask DMA. The pimpl
// keeps mutex/map details out of the kernel declaration surface; destroying
// the owning model releases every slot, so there is no process-lifetime cache
// or raw-address owner identity.
class SdpaStableMaskCache final {
public:
    SdpaStableMaskCache();
    ~SdpaStableMaskCache();

    SdpaStableMaskCache(const SdpaStableMaskCache&) = delete;
    SdpaStableMaskCache& operator=(const SdpaStableMaskCache&) = delete;
    SdpaStableMaskCache(SdpaStableMaskCache&&) = delete;
    SdpaStableMaskCache& operator=(SdpaStableMaskCache&&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;

    at::Tensor stable_slot(int64_t seq_q,
                           int64_t seq_k_aligned,
                           at::TensorOptions options,
                           int64_t ordinal);

    friend PreparedMask sdpa_prepare_mask(
        const c10::optional<at::Tensor>& attn_mask,
        bool is_causal,
        int64_t seq_q,
        int64_t seq_k,
        SdpaStableMaskCache& stable_cache,
        int64_t ordinal);
};

// CPU side: dim reduce → type convert → conditional pad → DDR copy.
// If seq_k is already v16-aligned, pad_nd is skipped entirely.
// `ordinal` distinguishes several SAME-SHAPE masks that must be alive at the
// same time (e.g. one per chunk of a chunked prefill). Shape alone is not
// identity — see SdpaStableMaskCache::Impl::Key in rpu_sdpa.cpp.
PreparedMask sdpa_prepare_mask(
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal,
    int64_t seq_q, int64_t seq_k,
    SdpaStableMaskCache& stable_cache,
    int64_t ordinal = 0);

// RPU side: DMA prepared mask to SPM (batch compatible).
void sdpa_dma_mask_to_spm(
    const PreparedMask &mask,
    uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t seq_k,
    int num_cores);

// Legacy combined API (calls prepare + dma internally).
int sdpa_load_mask_to_spm(
    const c10::optional<at::Tensor> &attn_mask,
    bool is_causal,
    uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t seq_k,
    int num_cores,
    at::Tensor* out_mask_ddr_ref,
    SdpaStableMaskCache& stable_cache);

// =============================================================================
// SDPA SPM Dispatch — select kernel by SdpaKernelType
// =============================================================================

// `cache_batch_offset_elems`: see rpu_launch_insert_kcache_spm_unified. Batch
// decode launches this once per sequence b, each reading only slot b's K/V.
void rpu_launch_sdpa_spm_dispatch(
    SdpaKernelType kernel,
    const at::Tensor &key, const at::Tensor &value,
    int attn_mask_type, std::optional<double> scale,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off, uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t kv_seq_len, int num_cores = 8, int virtual_num_cores = -1,
    int64_t cache_batch_offset_elems = 0);

// =============================================================================
// SDPA SPM Unified Kernel V2
// =============================================================================

void rpu_launch_sdpa_spm_unified_kernel_v2(
    const at::Tensor &key,
    const at::Tensor &value,
    int attn_mask_type,
    std::optional<double> scale,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off, uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t kv_seq_len,
    int num_cores = 8,
    int virtual_num_cores = -1,
    int64_t cache_batch_offset_elems = 0,
    bool use_16b = false,
    bool qwen3vl_tm160 = false);

// Independent ordinary Vision route; its COMPLETE descriptor owns selection.
void rpu_launch_sdpa_qwen3vl_n1200_tm160(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off,
    int64_t seq_q, int64_t declared_tmp_bytes);

// =============================================================================
// SDPA SPM vctxlen Kernel (dynamic context length)
// =============================================================================

void rpu_launch_sdpa_spm_vctxlen_kernel(
    const at::Tensor &key,
    const at::Tensor &value,
    int attn_mask_type,
    std::optional<double> scale,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off, uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t kv_seq_len,
    int num_cores = 8,
    int virtual_num_cores = -1);

// Generic raw-SPM path. Per-core layouts:
//   V input [batch,seq_k,nkv/core,head_dim]
//       -> V^T [batch,nkv/core,head_dim,align16(seq_k)]
//   Q/out [batch,seq_q,nq/core,head_dim]
//   K     [batch,align16(seq_k),nkv/core,head_dim]
// Supports masks NONE(0), LTM(1), and 2D(4).  Unlike the DDR KV-cache
// launcher, each core owns whole KV heads, so nq and nkv must be divisible by
// num_cores.
void rpu_launch_v_transpose_spm(
    uint32_t input_spm,
    uint32_t output_spm,
    int64_t batch,
    int64_t seq_k,
    int64_t num_kv_heads,
    int64_t head_dim,
    int num_cores);

void rpu_launch_sdpa_by_mha_spm(
    uint32_t q_spm,
    uint32_t k_spm,
    uint32_t v_transposed_spm,
    uint32_t output_spm,
    uint32_t mask_spm,
    int attn_mask_type,
    double scale,
    int64_t batch,
    int64_t seq_q,
    int64_t seq_k,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int num_cores);

// =============================================================================
// SDPA SPM vctxlen MINIBATCH Kernel (per-image K/V slicing)
// =============================================================================
// kernel: llm_fp16_32b_prefill_flash_attn_univ_vctxlen_minibatch
// Splits gridDim.x into `image_batch_count` image groups; image i's queries
// attend ONLY its own per_image_ctx_len K/V (cache rows [i*ctx, (i+1)*ctx)).
// No mask. HARD CONSTRAINT: num_kv_heads == num_heads (MHA).
//   seq_q       = total packed query length (= per_image_ctx_len * image_batch_count)
//   kv_seq_len  = per_image_ctx_len (each image's KV length; reg[0,1])
//   image_batch_count -> reg[34]
// grid_dim_x = ceil(seq_q, tile_m) MUST be divisible by image_batch_count.

void rpu_launch_sdpa_spm_minibatch_kernel(
    const at::Tensor &key,
    const at::Tensor &value,
    int attn_mask_type,
    std::optional<double> scale,
    uint32_t q_off, uint32_t output_off, uint32_t sdpa_tmp_off, uint32_t sdpa_mask_off,
    int64_t seq_q, int64_t num_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t per_image_ctx_len, int64_t image_batch_count,
    int num_cores = 8,
    int virtual_num_cores = -1,
    int tile_m_override = 0,
    bool use_16b = false);

// =============================================================================
// All-Reduce Sum + Residual
// =============================================================================

// Partial row producers must call this immediately before their first pure
// write into `input_spm`. It clears the leading `output_num_cores` shards,
// after which the producer overwrites its leading `input_num_cores` shards.
// Equal producer/consumer counts need no fill; no non-zero subset queue is used.
void rpu_prepare_ring_all_reduce_input(
    uint32_t input_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores = 8);

// Cold-selected exact Pi shape variant. False preserves the existing route;
// true requires eight cores, disjoint full aligned FP16 SPM operands and one
// of M768N1152, M400N2048, M50N1024 before staging the new kernel.
void rpu_launch_all_reduce_sum_residual_kernel(
    uint32_t input_spm, uint32_t residual_spm, uint32_t output_spm,
    int64_t M, int64_t N, int input_num_cores, int output_num_cores,
    bool pi05_xor3);

void rpu_launch_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores = 8,
    int output_num_cores = 8);


// Existing all-reduce kernel with an opaque FMB-authorized terminal target.
// The overload adds Graph provenance only; it does not introduce a new
// operator or kernel implementation.
void rpu_launch_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    const v3::SpmFmbPostFnYieldTarget& output,
    int64_t M,
    int64_t N,
    int input_num_cores = 8,
    int output_num_cores = 8);

// Fused ring with one shared, stable DDR residual. The residual address must
// stay fixed across GraphCache replay.
void rpu_launch_all_reduce_sum_residual_ddr_kernel(
    uint32_t input_spm,
    const c10::Half* residual_ddr,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores = 8,
    int output_num_cores = 8);

void rpu_launch_all_reduce_sum_residual_local_spm_kernel(
    uint32_t input_spm,
    uint32_t residual_local_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores = 8,
    int output_num_cores = 8);

// Fused ring reduce-scatter + all-gather. Both variants produce BF16 raw
// 16-bit output on every core; the names describe the two input encodings.














// =============================================================================
// SPM-to-SPM Linear Kernels
// =============================================================================

// Retain a controller-striped W4 pgrp scale at the alignment required by all
// 8 DDR controllers. W8 and NVFP4 scales are returned unchanged.
at::Tensor rpu_retain_linear_quant_scale(
    const at::Tensor& weight,
    const at::Tensor& scale);

void rpu_launch_linear_spm_to_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor &weight,
    uint32_t output_spm_addr,
    int64_t M, int64_t N, int64_t K,
    int partition,
    int num_cores = 8,
    uint32_t bias_spm_addr = 0);

// Pi0.5 Action NVFP4 striped-v2 ABI, per-handle ACC16/ACC32 accumulation.
void rpu_launch_pi05_nvfp4_v2_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t output,
    int64_t rows, int64_t n, int64_t k, int partition, int cores,
    const at::Tensor& scale, uint32_t tensor_scale, uint16_t layer,
    bool linear_acc32);

void rpu_launch_linear_spm_to_spm_acc16_kernel(
    uint32_t input_spm_addr,
    const at::Tensor &weight,
    uint32_t output_spm_addr,
    int64_t M, int64_t N, int64_t K,
    int partition,
    int num_cores = 8,
    uint32_t bias_spm_addr = 0,
    const at::Tensor &scale = {},
    // NVFP4-only: core-0 absolute SPM address of the projection-family FP32
    // tensor-scale array plus the layer index consumed by the canonical kernel.
    // Both are zero for every non-NVFP4 caller.
    uint32_t nvfp4_tensor_scale_spm_addr = 0,
    uint16_t nvfp4_layer_id = 0,
    // Frozen per-handle precision choice. The compatibility default is
    // deterministic ACC16; there is no native process-wide fallback.
    bool force_acc32 = false,
    // Explicit decode-only route. The default preserves generated auto-tiling.
    bool prefer_gemv = false,
    bool qkv_planar_direct = false,
    // Explicit bounded row W8A16 route. A selected route fails closed on any
    // contract/payload mismatch; false retains existing dispatch unchanged.
    bool prefer_row_weight_reuse = false);

// Exact Pi ten-step layer-0 Gate/Up route. Packing is a cold byte permutation
// of installed controller-striped W8 [4096,1024], never another quantization.
at::Tensor rpu_pack_pi05_gateup_resident_w8(const at::Tensor& weight);
void rpu_launch_pi05_gateup_resident_w8_preload(
    uint64_t& live_src_base, const at::Tensor& packed_weight,
    uint32_t weight_spm_addr);
// All three SPM addresses are core-0 absolute bytes; scale keeps its ordinary
// per-channel RPU FP16 [4096] DDR ABI (registers 20/21, device address >> 8).
void rpu_launch_pi05_gateup_resident_w8_kernel(
    uint32_t input_spm_addr, uint32_t weight_spm_addr,
    uint32_t output_spm_addr, const at::Tensor& scale);

// Cold-install only: invert the installed TP8 col swizzle, prove all eight
// logical KV replicas byte-identical, then return independent compact RPU
// owners. The compact weight remains TP8-col-swizzled for local N=32.
std::pair<at::Tensor, at::Tensor>
rpu_pack_pi05_kv1_stripe_w8a16(
    const at::Tensor& replicated_weight,
    const at::Tensor& replicated_scale);

// Cold-install pair map for direct-cache: each destination core c receives
// logical D16 blocks {c,c+8}. The accepted stripe packer above is unchanged.
std::pair<at::Tensor, at::Tensor>
rpu_pack_pi05_kv1_direct_pair_w8a16(
    const at::Tensor& replicated_weight,
    const at::Tensor& replicated_scale);

// Cold-install validation for the pair-owner route. This proves the installed
// K/V full owners contain eight byte-identical logical replicas and that all
// four DDR intervals satisfy the independent aligned shift8 ABI. It creates no
// persistent tensor and performs no RPU allocation.
void rpu_validate_pi05_kv1_pair_owner_w8a16(
    const at::Tensor& k_replicated_weight,
    const at::Tensor& k_replicated_scale,
    const at::Tensor& v_replicated_weight,
    const at::Tensor& v_replicated_scale);

// Cold-install validation for the exact Pi0.5 prefill K2048 pair-owner route.
// It inverts the installed TP8 column swizzle and proves all eight logical
// D256 replicas byte-identical for both K and V without allocating RPU owners.
void rpu_validate_pi05_prefill_kv1_pair_owner_w8a16(
    const at::Tensor& k_replicated_weight,
    const at::Tensor& k_replicated_scale,
    const at::Tensor& v_replicated_weight,
    const at::Tensor& v_replicated_scale);

// Exact Pi0.5 denoise M50/K1024/TP8 KV1 projection. Both SPM arguments are
// core-0 absolute byte addresses. The fixed generated M592N32 symbol runs one
// grid tile and emits each core's local N=32 fp16 stripe.
void rpu_launch_pi05_kv1_stripe_w8a16_m592n32_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& compact_weight,
    uint32_t output_spm_addr,
    const at::Tensor& compact_scale);

// Exact pair-owner projection boundary. The four tensors are the installed
// full replicated TP8 W8A16 owners; both outputs are core-0 absolute SPM byte
// addresses and receive each core's pair-mapped [50,2,16] compact result.
void rpu_launch_pi05_kv1_pair_owner_w8a16_m50n32x2k1024_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& k_full_weight,
    const at::Tensor& k_full_scale,
    uint32_t k_output_spm_addr,
    const at::Tensor& v_full_weight,
    const at::Tensor& v_full_scale,
    uint32_t v_output_spm_addr);

// Exact Pi0.5 prefill pair-owner projection. The input and compact outputs are
// core-0 absolute SPM byte addresses. The installed full K/V W8A16 owners are
// staged through typed DDR-register wrappers; the isolated hardware probe
// supplies node-level Graph census evidence.
void rpu_launch_pi05_prefill_kv1_pair_owner_w8a16_m400n32x2k2048_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& k_full_weight,
    const at::Tensor& k_full_scale,
    uint32_t k_output_spm_addr,
    const at::Tensor& v_full_weight,
    const at::Tensor& v_full_scale,
    uint32_t v_output_spm_addr, int64_t rows = 400);

// Exact direct-cache boundary. k/v offsets are typed FMB addr_offset values;
// the launcher converts each exactly once to a core-0 absolute SPM address.
void rpu_launch_pi05_kv1_direct_cache_m50d256p64_kernel(
    uint32_t k_compact_spm_offset,
    uint32_t v_compact_spm_offset,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t logical_rope_position,
    const KvInsertSegmentPlan& plan);

// Exact Pi0.5 prefill direct-cache boundary. The compact inputs stay typed as
// FMB offsets across the call; this launcher alone converts each once to its
// core-0 absolute SPM address before programming the generated kernel.
void rpu_launch_pi05_prefill_kv1_direct_cache_m400d256p400_kernel(
    v3::SpmOffset k_compact_spm_offset,
    v3::SpmOffset v_compact_spm_offset,
    const at::Tensor& k_cache,
    const at::Tensor& v_cache,
    const at::Tensor& cos,
    const at::Tensor& sin,
    int64_t logical_rope_position,
    const KvInsertSegmentPlan& plan);

void rpu_launch_gemv_spm_to_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor &weight,
    uint32_t output_spm_addr,
    int64_t M, int64_t N, int64_t K,
    int partition,
    int num_cores = 8,
    const at::Tensor &scale = {});



// Ordinary Qwen3-VL-4B prefill TP8 W8A16 Gate/Up/SwiGLU: fixed K2560/N9728,
// no bias, 16-aligned rows in [16,256], and existing COL-swizzled weights.
// Input/output are disjoint absolute SPM ranges. The N112 direct-store payload
// writes [rows,1216] per core and uses no additional scratch. Route admission
// and numerical support remain the COMPLETE model owner's responsibility.
void rpu_launch_qwen3vl_4b_prefill_gate_up_swiglu_w8a16_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t gate_output_spm_addr,
    int64_t rows,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale);

// Exact M1/K2560/N9728 COL-TP8 decode. Both [9728] scale views require
// backing for the original GEMV loader's 160-half final-block overfetch.
// Writes only SwiGLU into disjoint absolute SPM; optional main-ref payload.
void rpu_launch_qwen3vl_4b_decode_gate_up_swiglu_w8a16_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& gate_weight,
    const at::Tensor& up_weight,
    uint32_t output_spm_addr,
    const at::Tensor& gate_scale,
    const at::Tensor& up_scale);

// Exact M1/H2560/global-K4096 W8 ROW-TP8 O projection, original residual
// ring and BASE post RMSNorm. All SPM addresses are absolute and disjoint;
// full raw output is retained while normalized output reuses residual.
// The [2560] scale view requires 256 readable padding halves; eps is 1e-6.
void rpu_launch_qwen3vl_4b_decode_o_ring_norm_spm_kernel(
    uint32_t input_spm_addr,
    const at::Tensor& weight,
    const at::Tensor& scale,
    uint32_t partial_spm_addr,
    uint32_t residual_spm_addr,
    uint32_t raw_spm_addr,
    uint32_t gamma_spm_addr,
    double eps);








// =============================================================================
// SPM Binary Kernels
// =============================================================================

void rpu_launch_eltwise_binary_spm_kernel(
    uint32_t a_spm_addr,
    uint32_t b_spm_addr,
    uint32_t output_spm_addr,
    int64_t num_elements,
    ValuOpType op_type,
    c10::Half alpha = c10::Half(1.0),
    int num_cores = 8);

void rpu_launch_eltwise_binary_scalar_spm_kernel(
    uint32_t a_spm_addr,
    c10::Half scalar_val,
    uint32_t output_spm_addr,
    int64_t num_elements,
    ValuOpType op_type,
    int num_cores = 8);

void rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
    uint32_t a_spm_addr,
    uint32_t b_spm_addr,
    uint32_t output_spm_addr,
    int64_t n, int64_t c,
    c10::Half alpha_f,
    ValuOpType op_type,
    bool is_bopa,
    int num_cores = 8);   // 与 rpu_launch_eltwise_binary_spm_kernel 同义:
                          // 参与的 core 数。默认 8 保持既有调用点行为不变。

void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
    uint32_t a_spm_addr,
    uint32_t b_spm_addr,
    uint32_t output_spm_addr,
    int64_t n, int64_t c,
    c10::Half alpha_f,
    ValuOpType op_type,
    bool is_bopa,
    int num_cores = 8);

// Bx1xC_BxNxC middle-axis broadcast (a=[B,1,C] op b=[B,N,C]). The broadcast
// shape Nx1_NxC/1xC_NxC can't express (a "1" in a non-leading, non-last axis).
// GDN prefill chunk decay-mask: [HN,C,C] op [HN,1,C]. (rpu_eltwise_binary.cpp)
void rpu_launch_eltwise_binary_Bx1xC_BxNxC_spm_kernel(
    uint32_t a_spm_addr,
    uint32_t b_spm_addr,
    uint32_t output_spm_addr,
    int64_t bsz, int64_t n, int64_t c,
    c10::Half alpha_f,
    ValuOpType op_type,
    bool is_bopa,
    int num_cores = 8);
// Convenience wrappers
void rpu_launch_eltwise_mul_spm_kernel(
    uint32_t a_spm_addr, uint32_t b_spm_addr,
    uint32_t output_spm_addr, int64_t num_elements);

void rpu_launch_eltwise_add_spm_kernel(
    uint32_t a_spm_addr, uint32_t b_spm_addr,
    uint32_t output_spm_addr, int64_t num_elements,
    c10::Half alpha = c10::Half(1.0));

void rpu_launch_eltwise_sub_spm_kernel(
    uint32_t a_spm_addr, uint32_t b_spm_addr,
    uint32_t output_spm_addr, int64_t num_elements,
    c10::Half alpha = c10::Half(1.0));

void rpu_launch_eltwise_div_spm_kernel(
    uint32_t a_spm_addr, uint32_t b_spm_addr,
    uint32_t output_spm_addr, int64_t num_elements);

void rpu_launch_eltwise_pow_spm_kernel(
    uint32_t a_spm_addr, uint32_t b_spm_addr,
    uint32_t output_spm_addr, int64_t num_elements);

// =============================================================================
// Gemma Fused Decoder Layer — DELETED v5-09 (whole-file delete + decls retired)
// =============================================================================

// =============================================================================
// AdaRMS Gemma Fused Decoder Layer — DELETED v5-10 (whole-file delete + decl retired)
// =============================================================================

// =============================================================================
// CausalDecoder All-Layers-Once (v2.14 FusedModelBase migration + handle-based API;
// Qwen3 + Llama + future Phi/Mistral share this op family)
// =============================================================================

// Create a new CausalDecoderModel instance. Returns an opaque, monotonic int64
// handle used by set_weights / forward / destroy. Each Python model instance
// owns one handle and its own adapter-side GraphCache.
int64_t rpu_causal_decoder_create(
    bool qwen3_spm_kv_by_mha = true,
    bool adarms_fused_bcast = false,
    int64_t num_cores = 8,
    int64_t vocab_size = 0);

// Destroy a CausalDecoderModel instance. Called when Python model is torn down.
void rpu_causal_decoder_destroy(int64_t handle);
std::vector<int64_t> rpu_causal_decoder_topology(int64_t handle);

// Set model weights on the given instance (marks C++ preload/allocation state dirty;
// the owning adapter is responsible for invalidating/re-warming captured graphs).
// Phase 1: final_norm_w is required (fused into last layer's build_layer_subgraph)

void rpu_causal_decoder_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, bool use_silu,
    // R-Phase 0.5 (Qwen3-VL re-port): reserved for Phase 1 (M-RoPE) +
    // Phase 2 (DeepStack).  Empty defaults preserve existing 1D-RoPE / no-
    // injection behavior.  Non-empty currently rejected with TORCH_CHECK
    // until the corresponding phase lands.
    at::IntArrayRef mrope_section = {},
    at::IntArrayRef deepstack_lang_layers = {},
    // Wall-OSS-0.5 (P1): optional QKV bias (Qwen2.5). None defaults preserve
    // the no-bias path; all three size==num_layers enable per-layer bias.
    const std::optional<std::vector<at::Tensor>>& q_bias_list = std::nullopt,
    const std::optional<std::vector<at::Tensor>>& k_bias_list = std::nullopt,
    const std::optional<std::vector<at::Tensor>>& v_bias_list = std::nullopt);
void rpu_causal_decoder_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::IntArrayRef deepstack_lang_layers,
    at::TensorList q_w_scale_list = {},
    at::TensorList k_w_scale_list = {},
    at::TensorList v_w_scale_list = {},
    at::TensorList o_w_scale_list = {},
    at::TensorList gate_scale_list = {},
    at::TensorList up_scale_list = {},
    at::TensorList down_scale_list = {},
    // Wall-OSS-0.5 W8A16: optional QKV bias (Qwen2.5). None preserves
    // the no-bias qwen3 path; all three size==num_layers enable bias.
    const std::optional<std::vector<at::Tensor>>& q_bias_list = std::nullopt,
    const std::optional<std::vector<at::Tensor>>& k_bias_list = std::nullopt,
    const std::optional<std::vector<at::Tensor>>& v_bias_list = std::nullopt);
void rpu_causal_decoder_set_weights_nvfp4(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::IntArrayRef deepstack_lang_layers,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list,
    const at::Tensor& q_tensor_scales, const at::Tensor& k_tensor_scales,
    const at::Tensor& v_tensor_scales, const at::Tensor& o_tensor_scales,
    const at::Tensor& gate_tensor_scales, const at::Tensor& up_tensor_scales,
    const at::Tensor& down_tensor_scales,
    const std::optional<std::vector<at::Tensor>>& q_bias_list = std::nullopt,
    const std::optional<std::vector<at::Tensor>>& k_bias_list = std::nullopt,
    const std::optional<std::vector<at::Tensor>>& v_bias_list = std::nullopt);

// All-layers-once forward on the given instance.
// Default constraints: is_causal=true, batch_size==1, RPUCache required.
// Plain Qwen3 may explicitly admit causal batch decode via allow_batch_decode.
// Phase 2.5 decode-only fused lm_head:
//   - fuse_lm_head enabled (via rpu_causal_decoder_set_lm_head) AND seq_len==1 (decode):
//     returns [batch, 1, vocab_size] logits (SPM GEMM + CPU memcpy in forward())
//   - all other cases (prefill seq_len>1, or fuse_lm_head disabled):
//     returns [batch, seq_len, hidden_size] hidden states (Phase 2 default)
at::Tensor rpu_causal_decoder_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    // R-Phase 0.5 (Qwen3-VL re-port): reserved for Phase 1 (M-RoPE) +
    // Phase 2 (DeepStack).  None defaults preserve existing 1D-RoPE / no-
    // injection behavior.  Non-None currently rejected with TORCH_CHECK
    // until the corresponding phase lands.
    const std::optional<at::Tensor>& position_ids = std::nullopt,
    const std::optional<std::vector<at::Tensor>>& deepstack_dense_visual_embeds = std::nullopt,
    // partial_mrope path (wall-oss perf): per-forward interleaved cos/sin
    // [seq_len, head_dim/2] fp16 RPU. None preserves llama_mrope_interleave.
    const std::optional<at::Tensor>& rope_cos_il = std::nullopt,
    const std::optional<at::Tensor>& rope_sin_il = std::nullopt,
    // HALO WS-2: 1D-RoPE position-base override (-1 = legacy ctx().position;
    // >=0 decouples the RoPE table index from the cache-insert offset).
    int64_t cos_sin_offset = -1,
    // Batch decode: KV-cache batch slot for a batch==1 forward (batched
    // prefill drives one sequence per call). Batched decode passes 0 and
    // covers slots 0..batch-1 itself. See CausalDecoderModel::forward.
    int64_t batch_slot = 0,
    // Explicitly set only by the plain Qwen3 Python adapter.
    bool allow_batch_decode = false,
    // Positive only for a multi-token A6-selected production plan.
    int64_t planned_chunk_size = 0,
    // Versioned full input/QKV/compute schedule selected by A6. Empty keeps
    // legacy callers; non-empty is mutually exclusive with the scalar pin.
    at::IntArrayRef planned_stage_descriptor = {});

// ── Qwen3.5 hybrid-attention fused model ─────────────────────────────────────
// Heterogeneous per-layer dispatch (full attention vs Gated-DeltaNet seam) keyed
// by layer_is_full. Adds a gdn_states channel (recurrent state, parallel-indexed
// by layer) and an attn_gate_list (gate half split off the fused q_proj).
int64_t rpu_qwen3_5_create();
void rpu_qwen3_5_set_execution_cores(int64_t handle, int64_t num_cores);
std::vector<int64_t> rpu_qwen3_5_get_execution_topology(int64_t handle);
void    rpu_qwen3_5_destroy(int64_t handle);
int64_t rpu_qwen3_5_spm_z2_prepare(
    int64_t vision_handle, int64_t text_handle,
    int64_t num_patches, int64_t execution_len,
    int64_t image_row_begin, int64_t real_len);
int64_t rpu_qwen3_5_spm_z2_begin(
    int64_t vision_handle, int64_t text_handle,
    int64_t num_patches, int64_t execution_len,
    int64_t image_row_begin, int64_t real_len,
    int64_t plan_hash);
void rpu_qwen3_5_spm_z2_end(int64_t epoch, int64_t plan_hash);
at::Tensor rpu_qwen3_5_spm_z2_pipeline_forward(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& vision_input,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& step0_pos,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    at::TensorList gdn_states,
    at::TensorList conv_states,
    int64_t epoch,
    int64_t plan_hash);
at::Tensor rpu_qwen3_5_spm_z2_export_image_slice(
    int64_t epoch, int64_t plan_hash);
int64_t rpu_wall_oss_spm_z1_prepare(
    int64_t vision_handle, int64_t text_handle,
    int64_t num_patches, int64_t execution_len,
    int64_t image_row_begin, int64_t real_len);
void rpu_wall_oss_spm_z1_unprepare(
    int64_t vision_handle, int64_t text_handle, int64_t plan_hash);
int64_t rpu_wall_oss_spm_z1_begin(
    int64_t vision_handle, int64_t text_handle,
    int64_t num_patches, int64_t execution_len,
    int64_t image_row_begin, int64_t real_len,
    const at::Tensor& window_mask,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t plan_hash);
void rpu_wall_oss_spm_z1_end(int64_t epoch, int64_t plan_hash);
at::Tensor rpu_wall_oss_spm_z1_pipeline_forward(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& vision_input,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& window_mask,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t epoch,
    int64_t plan_hash);
at::Tensor rpu_wall_oss_spm_z1_export_destination(
    int64_t epoch, int64_t plan_hash);
std::vector<int64_t> rpu_qwen3vl_multiview_spm_dry_probe(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len);
int64_t rpu_qwen3vl_multiview_spm_z2_canary_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& position_ids);
at::Tensor rpu_qwen3vl_multiview_spm_z2_canary_forward(
    int64_t vision_handle,
    int64_t text_handle,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    int64_t plan_hash);
std::vector<int64_t> rpu_qwen3vl_multiview_spm_z2_canary_stats(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t plan_hash);
void rpu_qwen3vl_multiview_spm_z2_canary_close(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t plan_hash);
std::vector<int64_t> rpu_lingbot2_multiview_spm_z2_dry_probe(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle);
int64_t rpu_lingbot2_multiview_spm_z2_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    at::IntArrayRef planner_descriptor,
    at::IntArrayRef planner_digest,
    int64_t prepare_attempt_token);
void rpu_lingbot2_multiview_spm_z2_cancel_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    int64_t prepare_attempt_token,
    at::IntArrayRef planner_descriptor,
    at::IntArrayRef planner_digest);
int64_t rpu_lingbot2_multiview_spm_z2_reconfigure_begin(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    int64_t encoded_plan_hash,
    at::IntArrayRef planner_descriptor,
    at::IntArrayRef planner_digest,
    int64_t reconfigure_attempt_token);
void rpu_lingbot2_multiview_spm_z2_refresh_text_inputs(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t plan_hash,
    at::IntArrayRef planner_descriptor,
    at::IntArrayRef planner_digest);
void rpu_lingbot2_multiview_spm_z2_forward(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t plan_hash,
    at::IntArrayRef planner_descriptor,
    at::IntArrayRef planner_digest);
std::vector<int64_t> rpu_lingbot2_multiview_spm_z2_stats(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    int64_t plan_hash,
    at::IntArrayRef planner_descriptor,
    at::IntArrayRef planner_digest);
std::vector<std::vector<int64_t>> rpu_lingbot2_multiview_spm_z2_executed_owner(
    int64_t vision_handle, int64_t text_handle, int64_t expert_handle,
    int64_t plan_hash, at::IntArrayRef planner_descriptor,
    at::IntArrayRef planner_digest);
std::vector<GraphStats> rpu_lingbot2_multiview_spm_z2_graph_stats(
    int64_t vision_handle, int64_t text_handle, int64_t expert_handle,
    int64_t plan_hash, at::IntArrayRef planner_descriptor,
    at::IntArrayRef planner_digest);
void rpu_lingbot2_multiview_spm_z2_close(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    int64_t plan_hash,
    at::IntArrayRef planner_descriptor,
    at::IntArrayRef planner_digest);
int64_t rpu_qwen3vl_pooler_spm_z1_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len,
    int64_t retained_deepstack_count,
    at::IntArrayRef vision_stage_descriptor,
    at::IntArrayRef text_stage_descriptor);
void rpu_qwen3vl_pooler_spm_z1_unprepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t plan_hash);
std::vector<int64_t> rpu_qwen3vl_pooler_spm_z1_plan_stats(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t plan_hash);
std::vector<std::vector<int64_t>> rpu_qwen3vl_pooler_spm_z1_selected_owner(
    int64_t vision_handle, int64_t text_handle, int64_t plan_hash);
std::vector<std::vector<int64_t>> rpu_qwen3vl_pooler_spm_z1_recorded_ranges(
    int64_t vision_handle, int64_t text_handle, int64_t plan_hash);
std::vector<int64_t> rpu_qwen3vl_pooler_spm_z1_vision_stage_descriptor(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t plan_hash);
std::vector<int64_t> rpu_qwen3vl_pooler_spm_z1_text_stage_descriptor(
    int64_t vision_handle, int64_t text_handle, int64_t plan_hash,
    bool partial_mrope);
int64_t rpu_qwen3vl_pooler_spm_z1_begin(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len,
    const at::Tensor& position_ids,
    int64_t plan_hash,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il);
void rpu_qwen3vl_pooler_spm_z1_end(
    int64_t epoch,
    int64_t plan_hash);
void rpu_qwen3vl_pooler_spm_z1_abort(
    int64_t epoch,
    int64_t plan_hash);
void rpu_qwen3vl_pooler_spm_z1_cancel(
    int64_t epoch,
    int64_t plan_hash);
at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& vision_input,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    int64_t epoch,
    int64_t plan_hash);
void rpu_qwen3_5_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList attn_gate_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    at::IntArrayRef layer_is_full,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::TensorList gdn_in_z_list, at::TensorList gdn_out_list,
    at::TensorList gdn_A_log_list, at::TensorList gdn_dt_bias_list,
    at::TensorList gdn_norm_list,
    int64_t gdn_num_v_heads, int64_t gdn_key_head_dim,
    int64_t gdn_value_head_dim, int64_t gdn_conv_dim, int64_t gdn_conv_kernel,
    at::TensorList gdn_q_list, at::TensorList gdn_k_list, at::TensorList gdn_v_list,
    at::TensorList gdn_cq_list, at::TensorList gdn_ck_list, at::TensorList gdn_cv_list,
    at::TensorList gdn_b_bg_list, at::TensorList gdn_a_bg_list,
    at::TensorList q_scale_list, at::TensorList k_scale_list,
    at::TensorList v_scale_list, at::TensorList o_scale_list,
    at::TensorList attn_gate_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list,
    at::TensorList gdn_in_z_scale_list, at::TensorList gdn_out_scale_list,
    at::TensorList gdn_q_scale_list, at::TensorList gdn_k_scale_list,
    at::TensorList gdn_v_scale_list, at::TensorList gdn_b_bg_scale_list,
    at::TensorList gdn_a_bg_scale_list);
void rpu_qwen3_5_set_prefill_rope(int64_t handle, const at::Tensor& cos, const at::Tensor& sin);
void rpu_qwen3_5_set_valid_prefill_len(int64_t handle, int64_t n);
void rpu_qwen3_5_set_mrope_position_delta(int64_t handle, int64_t d);
int64_t rpu_qwen3_5_get_resolved_chunk_size(int64_t handle);
void rpu_qwen3_5_prepare_persistent_spm(int64_t handle, int64_t execution_len);
void rpu_qwen3_5_set_chunk_size_cap(int64_t handle, int64_t cap);
void rpu_qwen3_5_set_prefill_chunk_size(int64_t handle, int64_t chunk_size);
void rpu_qwen3_5_stage_prefill_execution_controls(
    int64_t handle, int64_t token, int64_t cap, int64_t chunk_size,
    int64_t max_kv_len, int64_t envelope_chunk);
void rpu_qwen3_5_set_chunk_envelope(int64_t handle, int64_t max_kv_len,
                                    int64_t chunk);
void rpu_qwen3_5_set_linear_acc32(int64_t handle, bool enabled);
void rpu_qwen3_5_set_fast_replay(int64_t handle, bool enabled);
void rpu_qwen3_5_set_retained_prefill_graph(
    int64_t handle, bool enabled);
void rpu_qwen3_5_enable_action_mode(int64_t handle);
void rpu_qwen3_5_set_action_chunk_size(int64_t handle, int64_t chunk_size);
void rpu_qwen3_5_stage_action_chunk_size(
    int64_t handle, int64_t token, int64_t chunk_size);
void rpu_qwen3_5_enable_execution_reconfigure(int64_t handle);
std::vector<int64_t> rpu_qwen3_5_resolve_action_stage_domain(
    int64_t handle, int64_t action_len, int64_t max_prefix_len,
    int64_t requested_chunk_size, bool include_action_io,
    int64_t action_route_flags, int64_t graph_lifecycle, int64_t num_steps);
void rpu_qwen3_5_set_action_io_weights(
    int64_t handle,
    const at::Tensor& input_w,
    const at::Tensor& input_b,
    const at::Tensor& output_w,
    const at::Tensor& output_b,
    int64_t action_dim,
    int64_t action_dim_pad,
    int64_t action_len);
int64_t rpu_qwen3_5_resolve_prefill_chunk_size(
    int64_t handle, int64_t execution_len);
std::vector<int64_t> rpu_qwen3_5_resolve_prefill_stage_domain(
    int64_t handle, int64_t execution_len, int64_t logical_len,
    int64_t planning_chunk_size_override = -1);
std::vector<int64_t> rpu_qwen3_5_resolve_decode_stage_descriptor(
    int64_t handle);
at::Tensor rpu_qwen3_5_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::TensorList gdn_states_list,
    at::TensorList conv_states_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    int64_t planned_chunk_size = 0,
    at::IntArrayRef planned_stage_descriptor = {});

// Independent Qwen3.5 sparse-MoE text model. Keep this dispatcher ABI limited
// to install, planner, retained-graph and forward operations.
int64_t rpu_qwen3_5_moe_create();
void rpu_qwen3_5_moe_set_execution_cores(
    int64_t handle, int64_t num_cores);
std::vector<int64_t> rpu_qwen3_5_moe_get_execution_topology(
    int64_t handle);
std::vector<int64_t> rpu_qwen3_5_moe_get_execution_topology_v2(
    int64_t handle);
void rpu_qwen3_5_moe_destroy(int64_t handle);
void rpu_qwen3_5_moe_prepare_persistent_spm(int64_t handle, int64_t execution_len);
void rpu_qwen3_5_moe_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList attn_gate_list,
    at::TensorList packed_scale_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList router_list,
    at::TensorList routed_gate_list, at::TensorList routed_up_list,
    at::TensorList routed_down_list,
    at::TensorList shared_gate_list, at::TensorList shared_up_list,
    at::TensorList shared_down_list,
    at::TensorList shared_scalar_gate_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    const at::Tensor& expert_ids, const at::Tensor& token_ids,
    at::IntArrayRef layer_is_full,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size,
    int64_t num_experts, int64_t top_k,
    int64_t routed_intermediate, int64_t shared_intermediate,
    int64_t tensor_parallel, int64_t routed_weight_mode,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::TensorList gdn_in_z_list, at::TensorList gdn_out_list,
    at::TensorList gdn_A_log_list, at::TensorList gdn_dt_bias_list,
    at::TensorList gdn_norm_list,
    int64_t gdn_num_v_heads, int64_t gdn_key_head_dim,
    int64_t gdn_value_head_dim, int64_t gdn_conv_dim,
    int64_t gdn_conv_kernel,
    at::TensorList gdn_q_list, at::TensorList gdn_k_list,
    at::TensorList gdn_v_list,
    at::TensorList gdn_cq_list, at::TensorList gdn_ck_list,
    at::TensorList gdn_cv_list,
    at::TensorList gdn_b_bg_list, at::TensorList gdn_a_bg_list);
void rpu_qwen3_5_moe_set_prefill_rope(
    int64_t handle, const at::Tensor& cos, const at::Tensor& sin);
void rpu_qwen3_5_moe_set_valid_prefill_len(int64_t handle, int64_t n);
void rpu_qwen3_5_moe_set_mrope_position_delta(int64_t handle, int64_t d);
void rpu_qwen3_5_moe_set_chunk_size_cap(int64_t handle, int64_t cap);
void rpu_qwen3_5_moe_set_prefill_chunk_size(
    int64_t handle, int64_t chunk_size);
void rpu_qwen3_5_moe_set_linear_acc32(int64_t handle, bool enabled);
void rpu_qwen3_5_moe_set_retained_prefill_graph(
    int64_t handle, bool enabled);
int64_t rpu_qwen3_5_moe_resolve_prefill_chunk_size(
    int64_t handle, int64_t execution_len);
std::vector<int64_t> rpu_qwen3_5_moe_resolve_prefill_stage_domain(
    int64_t handle, int64_t execution_len, int64_t logical_len,
    int64_t planning_chunk_size_override = -1);
std::vector<int64_t> rpu_qwen3_5_moe_resolve_decode_stage_descriptor(
    int64_t handle);
at::Tensor rpu_qwen3_5_moe_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::TensorList gdn_states_list,
    at::TensorList conv_states_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    int64_t planned_chunk_size = 0,
    at::IntArrayRef planned_stage_descriptor = {});

at::Tensor rpu_qwen3_5_action_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    const at::Tensor& adaptive_mod,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef prefix_lens,
    at::IntArrayRef planned_stage_descriptor);
at::Tensor rpu_qwen3_5_action_step_forward(
    int64_t handle,
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor action_out,
    double delta_t,
    const at::Tensor& adaptive_mod,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef prefix_lens,
    int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor);
at::Tensor rpu_qwen3_5_action_rtc_step_forward(
    int64_t handle,
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor action_out,
    const at::Tensor& action_prefix,
    int64_t action_prefix_len,
    double delta_t,
    const at::Tensor& adaptive_mod,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef prefix_lens,
    int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor);
at::Tensor rpu_qwen3_5_action_rtc_trace_forward(
    int64_t handle,
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor action_out,
    const at::Tensor& action_prefix,
    int64_t action_prefix_len,
    double delta_t,
    const at::Tensor& adaptive_mod,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef prefix_lens,
    at::Tensor trace_pre_action,
    at::Tensor trace_action_embed,
    at::Tensor trace_final_hidden,
    at::Tensor trace_velocity,
    at::Tensor trace_post_action,
    at::IntArrayRef planned_stage_descriptor);

// Phase 2.5: per-instance fused lm_head 入口
//   set_lm_head:   注入 [vocab, hidden] fp16 or int8 RPU-resident weight.
//                  int8 requires [vocab] fp16 lm_head_scale. 启用 fuse_lm_head_,
//                  调 invalidate_model_state() → main + post graph cache 自动失效
//   clear_lm_head: 清空 lm_head_w_, 关闭 fuse_lm_head_, 同样 invalidate_model_state()
// 切换走这两个 op 而不是重新 patch Python forward。
void rpu_causal_decoder_set_lm_head(
    int64_t handle,
    const at::Tensor& lm_head_w,
    const std::optional<at::Tensor>& lm_head_scale = std::nullopt,
    bool fuse_prefill_last = false,
    int64_t exact_candidates = 0);
void rpu_causal_decoder_clear_lm_head(int64_t handle);
void rpu_causal_decoder_set_rhinovla_high_precision_fusions(int64_t handle, bool enabled);
void rpu_causal_decoder_set_rhinovla_high_precision(int64_t handle, bool enabled);
void rpu_causal_decoder_set_linear_acc32(int64_t handle, bool enabled);
// RhinoVLA text-prefill fast-replay opt-ins (per-model).
void rpu_causal_decoder_set_fast_replay(int64_t handle, bool enabled);
void rpu_causal_decoder_set_preload_replay_skip(int64_t handle, bool enabled);
void rpu_causal_decoder_set_chunk_size_cap(int64_t handle, int64_t chunk_size_cap);
// MR-D: force this handle's chunk to exactly `chunk_size` (0 = auto). Replaces the
// process-global runtime.control.set_chunk_size, which decided a PER-HANDLE
// quantity — see the note in rpu_runtime_state.h. Sibling of the CAP, not a
// duplicate: the cap bounds the auto search, the override skips it. Both are still
// checked against the MR-A certified envelope.
void rpu_causal_decoder_set_chunk_size_override(int64_t handle, int64_t chunk_size);
void rpu_causal_decoder_set_explicit_mask_spm_residency(int64_t handle, bool enabled);
void rpu_causal_decoder_enable_execution_reconfigure(int64_t handle);
void rpu_causal_decoder_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_causal_decoder_kvinsert_cost_catalog_sha256(int64_t handle);

// Owner-local cold catalogs: use the leaf handle that emits the KV routes.
// Lookup-only metadata from the latest successful native stage oracle.
// Fixed-V2 owners expose actual profile/domain identity but no catalog binder.
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_causal_decoder_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_halo_action_expert_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_lingbot_v2_moe_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_hyvla_vlm_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen3_5_vision_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_gr00t_dit_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_hyvla_expert_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_gr00t_vl_encoder_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_halo_image_flow_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_wall_oss_action_step_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_hyvit2_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen25vl_vision_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwenpi05_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_dinov3_vision_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_siglip_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen3_5_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_lingbot_denoise_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen3vl_vision_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_internvla_nextdit_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_adarms_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_navdp_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_pi05_denoise_step_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_rhino_vla_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor,
    int64_t site_id, int64_t invocation, int64_t route);
KvInsertCostDomainQuery rpu_gemma_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_gemma2_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_gemma4_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
std::string rpu_gemma_kvinsert_cost_catalog_sha256(int64_t handle);
std::string rpu_gemma2_kvinsert_cost_catalog_sha256(int64_t handle);
std::string rpu_gemma4_kvinsert_cost_catalog_sha256(int64_t handle);
KvInsertCostDomainQuery rpu_halo_action_expert_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_lingbot_v2_moe_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_hyvla_vlm_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_qwen3_5_vision_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_gr00t_dit_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_hyvla_expert_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_gr00t_vl_encoder_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_halo_image_flow_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_wall_oss_action_step_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_hyvit2_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_causal_decoder_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_qwen25vl_vision_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_qwenpi05_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_dinov3_vision_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_siglip_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_qwen3_5_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_lingbot_denoise_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_qwen3vl_vision_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_internvla_nextdit_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_adarms_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_navdp_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_pi05_denoise_step_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
KvInsertCostDomainQuery rpu_rhino_vla_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
void rpu_adarms_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_adarms_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_dinov3_vision_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_dinov3_vision_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_gr00t_dit_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_gr00t_dit_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_gr00t_vl_encoder_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_gr00t_vl_encoder_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_halo_action_expert_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_halo_action_expert_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_halo_image_flow_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_halo_image_flow_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_hyvit2_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_hyvit2_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_hyvla_expert_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_hyvla_expert_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_hyvla_vlm_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_hyvla_vlm_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_internvla_nextdit_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_internvla_nextdit_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_lingbot_v2_moe_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_lingbot_v2_moe_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_navdp_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_navdp_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_pi05_denoise_step_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_pi05_denoise_step_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_qwen25vl_vision_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_qwen25vl_vision_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_qwen3_5_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_qwen3_5_kvinsert_cost_catalog_sha256(int64_t handle);
std::vector<int64_t> rpu_qwen3_5_moe_planner_cache_identity(int64_t handle);
void rpu_qwen3_5_moe_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_qwen3_5_moe_kvinsert_cost_catalog_sha256(int64_t handle);
std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen3_5_moe_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route);
KvInsertCostDomainQuery rpu_qwen3_5_moe_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor);
void rpu_qwen3_5_vision_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_qwen3_5_vision_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_qwen3vl_vision_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_qwen3vl_vision_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_qwenpi05_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_qwenpi05_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_rhino_vla_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_rhino_vla_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_siglip_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_siglip_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_wall_oss_action_step_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_wall_oss_action_step_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_lingbot_denoise_bind_kvinsert_costs(
    int64_t handle, at::IntArrayRef identity,
    const std::string& catalog_sha256, at::IntArrayRef certificate_rows);
std::string rpu_lingbot_denoise_kvinsert_cost_catalog_sha256(int64_t handle);
void rpu_causal_decoder_stage_chunk_size_override(
    int64_t handle, int64_t token, int64_t chunk_size);
// Read it back. This is what the adapters fold into the GRAPH SIGNATURE, so that
// two chunk policies at one shape are two cache entries instead of one entry
// replayed under the wrong SPM layout. O(1) on purpose: it must be affordable on
// the decode path, where the override also reaches alloc_ctx.chunk_size.
int64_t rpu_causal_decoder_get_chunk_size_override(int64_t handle);
// MR-A certified chunk envelope (docs/roadmap/chunk_certified_envelope.md).
void rpu_causal_decoder_set_chunk_envelope(int64_t handle, int64_t max_kv_len,
                                           int64_t chunk);
void rpu_causal_decoder_set_equal_two_prefill(int64_t handle, bool enabled);

// AdaRMS FiLM step-update (LingBot-VLA action expert). Per-layer scale rides on
// the norm-weight slots; shift is broadcast-added after each RMSNorm. See
// rpu_qwen3_model.h::set_adarms_step.
void rpu_causal_decoder_set_adarms_step(
    int64_t handle,
    at::TensorList input_scales, at::TensorList input_shifts,
    at::TensorList post_scales,  at::TensorList post_shifts);

// Replay-safe AdaRMS step-update: each arg is one [num_layers, hidden] fp16 RPU tensor.
// Enables build-once/replay-per-step for the denoise loop. See rpu_qwen3_model.h.
void rpu_causal_decoder_set_adarms_step_mutable(
    int64_t handle,
    const at::Tensor& input_scale, const at::Tensor& input_shift,
    const at::Tensor& post_scale,  const at::Tensor& post_shift);

// TASK-1.5 (codex round-3 path-(b) BC2-01 closure): per-handle resolved chunk_size
// from the most-recent forward; sentinel 0 if no forward has run yet for this handle.
// Used by tests/library/test_llama_rpu_for_causal_lm.py BC2-01 3-step protocol;
// closes the codex round-2 BC2-01 sub-blocker.
int64_t rpu_causal_decoder_get_resolved_chunk_size(int64_t handle);
// Planning-only query: run the exact generic causal-decoder SPM/kernel
// validator without dispatching a forward.
// `position` = the cache position the planned forward starts at. Chunk
// validity depends on the KV context length (position + execution_len), so a
// continuation (a second conversational turn) must not be planned at 0.
int64_t rpu_causal_decoder_resolve_prefill_chunk_size(
    int64_t handle, int64_t execution_len, int64_t position);
void rpu_causal_decoder_prepare_persistent_spm(
    int64_t handle, int64_t execution_len, int64_t position,
    bool is_causal, int64_t mask_kv_len);
// Lightweight owner-local cache identity; invalid/destroyed handles fail closed.
std::vector<int64_t> rpu_causal_decoder_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_qwen3_5_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_qwen3_5_vision_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_gemma_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_gemma2_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_gemma4_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_adarms_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_gr00t_dit_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_gr00t_denoise_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_internvla_nextdit_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_navdp_planner_cache_identity(int64_t handle);
void rpu_navdp_validate_execution_admission(
    int64_t handle, int64_t packed_rows, int64_t denoise_steps);
std::vector<int64_t> rpu_gr00t_vl_encoder_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_pi05_denoise_step_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_wall_oss_action_step_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_lingbot_denoise_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_lingbot_v2_moe_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_halo_action_expert_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_halo_image_flow_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_hyvla_vlm_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_hyvla_expert_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_qwenpi05_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_rhino_vla_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_siglip_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_hyvit2_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_qwen3vl_vision_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_qwen25vl_vision_planner_cache_identity(int64_t handle);
std::vector<int64_t> rpu_dinov3_vision_planner_cache_identity(int64_t handle);

std::vector<int64_t> rpu_causal_decoder_resolve_prefill_stage_domain(
    int64_t handle, int64_t execution_len, int64_t position,
    bool is_causal, int64_t mask_kv_len, int64_t rope_mode,
    int64_t graph_lifecycle, int64_t logical_len,
    int64_t planning_chunk_size_override = -1);
std::vector<int64_t> rpu_causal_decoder_resolve_decode_stage_descriptor(
    int64_t handle, int64_t position = 0);

// =============================================================================
// Gemma All-Layers-Once (Phase 3: FusedModelBase migration)
// =============================================================================

// Create a new GemmaModel instance. Returns an opaque int64 handle.
int64_t rpu_gemma_create(bool linear_acc32);
void rpu_gemma_set_execution_core_count(int64_t handle, int64_t cores);
std::vector<int64_t> rpu_gemma_get_execution_topology(int64_t handle);

// Destroy a GemmaModel instance.
void rpu_gemma_destroy(int64_t handle);
void rpu_gemma_set_prefill_pair_rows(int64_t handle, int64_t rows);
void rpu_gemma_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);

// Set model weights on the given instance (no QK norms, Gemma-specific).
void rpu_gemma_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list);

// Per-forward RoPE table gathered by position_ids: row r is the entry for this
// forward's row r, so RoPE indexes the LOGICAL position while the KV insert
// keeps the physical row. Optional — without it the static position lookup from
// set_weights is used, i.e. RoPE == row index.
void rpu_gemma_set_prefill_rope(
    int64_t handle, const at::Tensor& cos, const at::Tensor& sin);

int64_t rpu_gemma_resolve_prefill_chunk_size(
    int64_t handle, int64_t execution_len, int64_t chunk_size = 0);
std::vector<int64_t> rpu_gemma_resolve_prefill_stage_domain(
    int64_t handle, int64_t execution_len,
    int64_t requested_chunk_size = 0, int64_t logical_len = 0);
int64_t rpu_gemma_get_resolved_chunk_size(int64_t handle);

// All-layers-once forward. Supports both causal (SEQUENTIAL) and bidirectional
// (KV_FIRST two-phase mode with pre-prepared chunk masks).
at::Tensor rpu_gemma_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    int64_t chunk_size = 0,
    at::IntArrayRef planned_stage_descriptor = {});

// Pi0.5's explicit prefix-cache-only entry has no hidden-state result.
void rpu_gemma_prefill_kv_only(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    int64_t chunk_size = 0,
    at::IntArrayRef planned_stage_descriptor = {});

// =============================================================================
// Gemma2 All-Layers-Once (PaliGemma2 text decoder)
// =============================================================================

int64_t rpu_gemma2_create();
void rpu_gemma2_destroy(int64_t handle);
void rpu_gemma2_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);

void rpu_gemma2_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList input_norm_list,
    at::TensorList post_attention_norm_list,
    at::TensorList pre_feedforward_norm_list,
    at::TensorList post_feedforward_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    at::IntArrayRef layer_type_ids,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    double query_pre_attn_scalar);

std::vector<int64_t> rpu_gemma2_resolve_prefill_stage_domain(
    int64_t handle, int64_t execution_len, int64_t position,
    int64_t requested_chunk_size = 0);

at::Tensor rpu_gemma2_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    int64_t chunk_size = 0,
    at::IntArrayRef planned_stage_descriptor = {});

// Gemma4Model -- Gemma4 E4B text decoder, mixed geometry (sliding 256 / global
// 512), 4-norm + QK-RMSNorm, direct-weight norms, qk_scale=1.0.
// Implementation: src/fused/rpu_gemma4_model.cpp.

int64_t rpu_gemma4_create();
void rpu_gemma4_destroy(int64_t handle);
void rpu_gemma4_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);

void rpu_gemma4_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_attn_norm_list,
    at::TensorList pre_ff_norm_list, at::TensorList post_ff_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos_sliding, const at::Tensor& sin_sliding,
    const at::Tensor& cos_global,  const at::Tensor& sin_global,
    const at::Tensor& final_norm_w,
    at::TensorList ple_gate_list, at::TensorList ple_proj_list,
    at::TensorList ple_post_norm_list, const at::Tensor& layer_scalar,
    at::IntArrayRef layer_type_ids,
    at::IntArrayRef head_dim_per_layer,
    at::IntArrayRef num_kv_heads_per_layer,
    at::IntArrayRef kv_source_layer,
    at::IntArrayRef is_kv_shared,
    int64_t num_q_heads, int64_t num_kv_heads_max, int64_t head_dim_max,
    int64_t hidden_size, int64_t intermediate_size, int64_t ple_dim,
    double eps, double qk_scale, int64_t sliding_window);

std::vector<int64_t> rpu_gemma4_resolve_stage_domain(
    int64_t handle, int64_t execution_len, int64_t planning_position,
    int64_t max_kv_len, int64_t requested_chunk_size = 0);

at::Tensor rpu_gemma4_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    int64_t chunk_size,
    const std::optional<at::Tensor>& side_input,
    at::IntArrayRef planned_stage_descriptor = {});

// =============================================================================
// AdaRMSModel -- Pi0.5 Action Expert all-layers-once (Phase 4)
// All-layers-once C++ model driven by FusedModelBase. Uses cond GEMV per
// forward (c.idx==0) and flat-phase SPM (D-403). Always SEQUENTIAL mode.
// =============================================================================

// Create a new AdaRMSModel instance; returns its handle.
int64_t rpu_adarms_create(bool linear_acc32);
void rpu_adarms_set_execution_core_count(int64_t handle, int64_t cores);
std::vector<int64_t> rpu_adarms_get_execution_topology(int64_t handle);

// Destroy an AdaRMSModel instance.
void rpu_adarms_destroy(int64_t handle);
void rpu_adarms_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);

// Set model weights. Dense-w list is expanded+transformed [8*3*hidden, hidden]
// fp16 (col-partition swizzled for 8-core GEMV). Dense-b list is [3*hidden] fp16.
void rpu_adarms_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::TensorList q_w_scale_list,
    at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list,
    at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list,
    at::TensorList up_scale_list,
    at::TensorList down_scale_list, const std::optional<std::vector<at::Tensor>>& nvfp4_tensor_scales);

// All-layers-once forward. SEQUENTIAL mode only. cond is [hidden_size] fp16 on
// RPU device, DMA'd once per forward at layer 0 / chunk 0 (D-401).
// Logical RoPE start for the action suffix (-1 = use the physical cache row).
void rpu_adarms_set_rope_position(int64_t handle, int64_t position);

at::Tensor rpu_adarms_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    at::IntArrayRef planned_stage_descriptor);
std::vector<int64_t> rpu_adarms_resolve_action_stage_domain(
    int64_t handle, int64_t execution_len, int64_t logical_len,
    int64_t position, int64_t kv_len, bool use_attention_mask,
    bool is_causal, int64_t requested_chunk_size);
int64_t rpu_adarms_get_resolved_chunk_size(int64_t handle);

// GR00T-N1.7 flow-matching DiT action head (src/fused/rpu_gr00t_dit_model.cpp).
// 32-layer AlternateVLDiT: self-attention and cross-attention use external
// vl_embeds K/V with MASK_2D image/text subsets; norm1 is AdaLayerNorm.
int64_t rpu_gr00t_dit_create();
void rpu_gr00t_dit_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);
void    rpu_gr00t_dit_destroy(int64_t handle);
void    rpu_gr00t_dit_set_weights(
    int64_t handle,
    at::TensorList to_q_list, at::TensorList to_k_list,
    at::TensorList to_v_list, at::TensorList to_out_list,
    at::TensorList ff1_list, at::TensorList ff2_list,
    at::TensorList adaln_w_list, at::TensorList adaln_b_list,
    at::TensorList to_q_b_list, at::TensorList to_k_b_list,
    at::TensorList to_v_b_list, at::TensorList to_out_b_list,
    at::TensorList ff1_b_list, at::TensorList ff2_b_list,
    at::IntArrayRef block_is_cross,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t ff_inter, int64_t cross_dim,
    double eps);
void    rpu_gr00t_dit_set_weights_w8a16(
    int64_t handle,
    at::TensorList to_q_list, at::TensorList to_k_list,
    at::TensorList to_v_list, at::TensorList to_out_list,
    at::TensorList ff1_list, at::TensorList ff2_list,
    at::TensorList adaln_w_list, at::TensorList adaln_b_list,
    at::TensorList to_q_b_list, at::TensorList to_k_b_list,
    at::TensorList to_v_b_list, at::TensorList to_out_b_list,
    at::TensorList ff1_b_list, at::TensorList ff2_b_list,
    at::IntArrayRef block_is_cross,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t ff_inter, int64_t cross_dim,
    double eps,
    at::TensorList to_q_scale_list, at::TensorList to_k_scale_list,
    at::TensorList to_v_scale_list, at::TensorList to_out_scale_list,
    at::TensorList ff1_scale_list, at::TensorList ff2_scale_list,
    at::TensorList adaln_scale_list = {});
at::Tensor rpu_gr00t_dit_forward(
    int64_t handle,
    const at::Tensor& sa_embs,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    const std::optional<at::Tensor>& vl_embeds,
    const std::optional<at::Tensor>& cross_mask,
    const std::optional<at::Tensor>& cross_mask_img,
    at::IntArrayRef planned_stage_descriptor);
std::vector<int64_t> rpu_gr00t_dit_resolve_stage_domain(
    int64_t handle, int64_t rows, int64_t cross_seq_len,
    const std::optional<at::Tensor>& attention_mask);

// =============================================================================
// InternVLA-N1 standalone flow-matching NextDiT core
// (src/fused/rpu_internvla_nextdit_model.cpp).
// =============================================================================
int64_t rpu_internvla_nextdit_create();
void rpu_internvla_nextdit_set_chunk_envelope(int64_t handle, int64_t packed_self_capacity, int64_t cross_capacity, int64_t chunk);
void    rpu_internvla_nextdit_destroy(int64_t handle);
void    rpu_internvla_nextdit_set_debug_export(int64_t handle, bool enabled);
void    rpu_internvla_nextdit_set_16b_sdpa(int64_t handle, bool enabled);
void    rpu_internvla_nextdit_set_fused_qkq_weights(
    int64_t handle, at::TensorList weights);
void    rpu_internvla_nextdit_set_step_prepare_weights(
    int64_t handle, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& fold_norm_weight);
void    rpu_internvla_nextdit_set_weights(
    int64_t handle,
    at::TensorList self_q_w, at::TensorList self_k_w, at::TensorList self_v_w,
    at::TensorList cross_q_w, at::TensorList cross_k_w,
    at::TensorList cross_v_w, at::TensorList out_w,
    at::TensorList ff1_w, at::TensorList ff2_w, at::TensorList ff3_w,
    at::TensorList norm1_w, at::TensorList norm2_w,
    at::TensorList ffn_norm1_w, at::TensorList ffn_norm2_w,
    at::TensorList self_qn_w, at::TensorList self_qn_b,
    at::TensorList self_kn_w, at::TensorList self_kn_b,
    at::TensorList cross_qn_w, at::TensorList cross_qn_b,
    at::TensorList cross_kn_w, at::TensorList cross_kn_b,
    at::TensorList context_norm_w,
    at::TensorList cross_gate_heads,
    const at::Tensor& final_out_w, const at::Tensor& final_out_b,
    const at::Tensor& final_ln_w, const at::Tensor& final_ln_b,
    const at::Tensor& action_enc_w, const at::Tensor& action_enc_b,
    const at::Tensor& action_position,
    const at::Tensor& action_dec_w, const at::Tensor& action_dec_b,
    int64_t num_heads, int64_t head_dim, int64_t hidden_size,
    int64_t ff_intermediate, double eps, double final_norm_eps,
    bool fold_norm_modulation);
at::Tensor rpu_internvla_nextdit_forward(
    int64_t handle, const at::Tensor& x, const at::Tensor& modulation,
    const at::Tensor& final_scale,
    const at::Tensor& raw_actions, bool use_action_encoder, bool populate_cross_cache,
    bool denoise_unroll, bool prepare_modulation_on_rpu,
    at::ArrayRef<double> euler_steps,
    const at::Tensor& cross_context,
    at::TensorList self_k_caches, at::TensorList self_v_caches,
    at::TensorList cross_k_caches, at::TensorList cross_v_caches,
    const at::Tensor& layer_outputs, bool collect_layer_outputs,
    at::IntArrayRef planned_stage_descriptor = {});
std::vector<int64_t> rpu_internvla_nextdit_resolve_stage_domain(
    int64_t handle, int64_t action_rows, int64_t cross_seq_len,
    bool use_action_encoder, bool populate_cross_cache,
    int64_t action_batch, int64_t denoise_steps,
    bool prepare_modulation_on_rpu);
int64_t rpu_internvla_nextdit_get_resolved_chunk_size(int64_t handle);

// InternVLA-N1 NavDP action head (src/fused/rpu_navdp_model.cpp).
// 16-layer nn.TransformerDecoder diffusion policy: pre-LN causal self-attention,
// cross-attention over memory and GELU-FFN; DDPM 20-step.
int64_t rpu_navdp_create(bool allow_numeric_blocked, bool spm_kv_by_mha);
void rpu_navdp_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);
void    rpu_navdp_destroy(int64_t handle);
void    rpu_navdp_set_former_mode(int64_t handle, bool former);  // reuse as RGBD former_net (POST-LN/ReLU/non-causal)
void    rpu_navdp_set_weights(
    int64_t handle,
    at::TensorList sq_w, at::TensorList sk_w, at::TensorList sv_w, at::TensorList so_w,
    at::TensorList sq_b, at::TensorList sk_b, at::TensorList sv_b, at::TensorList so_b,
    at::TensorList cq_w, at::TensorList ck_w, at::TensorList cv_w, at::TensorList co_w,
    at::TensorList cq_b, at::TensorList ck_b, at::TensorList cv_b, at::TensorList co_b,
    at::TensorList ff1_w, at::TensorList ff1_b, at::TensorList ff2_w, at::TensorList ff2_b,
    at::TensorList n1_w, at::TensorList n1_b, at::TensorList n2_w, at::TensorList n2_b,
    at::TensorList n3_w, at::TensorList n3_b,
    int64_t num_heads, int64_t head_dim, int64_t hidden_size,
    int64_t ff_inter, int64_t memory_len, int64_t predict_size,
    int64_t action_dim, double eps);
at::Tensor rpu_navdp_forward(
    int64_t handle,
    const at::Tensor& tgt,
    const at::Tensor& memory,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& tgt_mask,
    at::IntArrayRef planned_stage_descriptor = {});
std::vector<int64_t> rpu_navdp_resolve_stage_domain(
    int64_t handle, int64_t packed_rows, int64_t denoise_steps = 0);
int64_t rpu_navdp_get_resolved_chunk_size(int64_t handle);

void rpu_navdp_set_denoise_glue(
    int64_t handle, at::Tensor ie_w, at::Tensor ie_b, at::Tensor ah_w, at::Tensor ah_b,
    at::Tensor fln_w, at::Tensor fln_b, at::Tensor out_pos);

at::Tensor rpu_navdp_denoise_loop_forward(
    int64_t handle, const at::Tensor& x0, const at::Tensor& mem_all, const at::Tensor& sigz,
    const at::Tensor& coeff, at::TensorList k_caches_list, at::TensorList v_caches_list,
    const at::Tensor& out,
    at::IntArrayRef planned_stage_descriptor = {});

// =============================================================================
// GR00T-N1.7 "全程 RPU" Component A — vlln + 4-layer vl_self_attention encoder
// (src/fused/rpu_gr00t_vl_encoder_model.cpp). SigLIP-twin bidirectional self-attn:
// u = raw⊙inv_w_rms -> LN(vlln) -> 4×[LN1 -> qkv+bias -> KV@0 -> SDPA(MASK_NONE) ->
// o+bias -> resid -> LN3 -> GELU-FFN -> resid]. See FUSED_GLUE_{SPEC,PLAN}.md (Phase A).
// =============================================================================
// GR00T-N1.7 Component B — gr00t_denoise (extends the gr00t_dit handle with the per-step glue:
// state/action encoders, norm_out, action decoder, in-graph Euler). step_idx selects baked
// cond/tau tables. See FUSED_GLUE_{SPEC,PLAN}.md (Phase B) + build_denoise_step.py.
void rpu_gr00t_denoise_set_weights(
    int64_t handle,
    const at::Tensor& se1_w, const at::Tensor& se1_b,
    const at::Tensor& se2_w, const at::Tensor& se2_b,
    const at::Tensor& ae1_w, const at::Tensor& ae1_b,
    const at::Tensor& ae2a_w, const at::Tensor& ae2t_w, const at::Tensor& ae2_b,
    const at::Tensor& ae3_w, const at::Tensor& ae3_b,
    const at::Tensor& dec1_w, const at::Tensor& dec1_b,
    const at::Tensor& dec2_w, const at::Tensor& dec2_b,
    const at::Tensor& po1_w, const at::Tensor& po1_b,
    const at::Tensor& po2_w, const at::Tensor& po2_b,
    const at::Tensor& cond_table, const at::Tensor& tau_table, const at::Tensor& pos_table,
    int64_t action_dim, int64_t action_dim_pad, int64_t num_steps, double dt,
    const std::optional<at::Tensor>& se2_scale = {}, const std::optional<at::Tensor>& ae2a_scale = {},
    const std::optional<at::Tensor>& ae2t_scale = {}, const std::optional<at::Tensor>& ae3_scale = {},
    const std::optional<at::Tensor>& dec1_scale = {}, const std::optional<at::Tensor>& dec2_scale = {},
    const std::optional<at::Tensor>& po1_scale = {}, const std::optional<at::Tensor>& po2_scale = {});
at::Tensor rpu_gr00t_denoise_step_forward(
    int64_t handle,
    const at::Tensor& actions, const at::Tensor& state, int64_t step_idx,
    at::TensorList k_caches_list, at::TensorList v_caches_list,
    const at::Tensor& vl_embeds, const at::Tensor& tmask, const at::Tensor& imask,
    at::IntArrayRef planned_stage_descriptor);
// Phase C — in-graph num_steps Euler unroll (one forward; body_iterations = num_steps).
at::Tensor rpu_gr00t_denoise_unroll_forward(
    int64_t handle,
    const at::Tensor& noise, const at::Tensor& state,
    at::TensorList k_caches_list, at::TensorList v_caches_list,
    const at::Tensor& vl_embeds, const at::Tensor& tmask,
    const at::Tensor& imask,
    at::IntArrayRef planned_stage_descriptor);
std::vector<int64_t> rpu_gr00t_denoise_resolve_stage_domain(
    int64_t handle, int64_t action_rows, int64_t cross_seq_len, int64_t step_idx);
int64_t rpu_gr00t_denoise_get_resolved_chunk_size(int64_t handle);
// Typed physical Z2 canary: GR00T VL-encoder residual1 writes a lease-owned
// dense SPM port consumed directly by denoise cross-attention.  The components
// may run under the legacy two-Graph proof or one composite outer Graph.
// prepare/begin/end expose opaque tokens only; raw SPM addresses never cross
// this C++ boundary.
int64_t rpu_gr00t_spm_z2_prepare(
    int64_t vl_handle, int64_t denoise_handle, int64_t seq_len);
void rpu_gr00t_spm_z2_unprepare(
    int64_t vl_handle, int64_t denoise_handle, int64_t plan_hash);
int64_t rpu_gr00t_spm_z2_begin(
    int64_t vl_handle, int64_t denoise_handle, int64_t seq_len,
    int64_t plan_hash);
void rpu_gr00t_spm_z2_end(int64_t epoch, int64_t plan_hash);
void rpu_gr00t_spm_z2_prepare_masks(
    int64_t denoise_handle,
    const at::Tensor& tmask,
    const at::Tensor& imask,
    int64_t plan_hash);
void rpu_gr00t_vl_encoder_forward_z2(
    int64_t handle,
    const at::Tensor& raw,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t epoch,
    int64_t plan_hash);
at::Tensor rpu_gr00t_denoise_unroll_forward_z2(
    int64_t handle,
    const at::Tensor& noise,
    const at::Tensor& state,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const at::Tensor& tmask,
    const at::Tensor& imask,
    int64_t epoch,
    int64_t plan_hash);
at::Tensor rpu_gr00t_spm_z2_pipeline_forward(
    int64_t vl_handle,
    int64_t denoise_handle,
    const at::Tensor& raw,
    at::TensorList vl_k_caches,
    at::TensorList vl_v_caches,
    const at::Tensor& noise,
    const at::Tensor& state,
    at::TensorList denoise_k_caches,
    at::TensorList denoise_v_caches,
    int64_t epoch,
    int64_t plan_hash);

int64_t rpu_gr00t_vl_encoder_create();
void rpu_gr00t_vl_encoder_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);
void    rpu_gr00t_vl_encoder_destroy(int64_t handle);
void    rpu_gr00t_vl_encoder_set_weights(
    int64_t handle,
    at::TensorList norm1_w_list, at::TensorList norm1_b_list,
    at::TensorList to_q_list, at::TensorList to_q_b_list,
    at::TensorList to_k_list, at::TensorList to_k_b_list,
    at::TensorList to_v_list, at::TensorList to_v_b_list,
    at::TensorList to_out_list, at::TensorList to_out_b_list,
    at::TensorList norm3_w_list, at::TensorList norm3_b_list,
    at::TensorList ff0_list, at::TensorList ff0_b_list,
    at::TensorList ff2_list, at::TensorList ff2_b_list,
    const at::Tensor& vlln_w, const at::Tensor& vlln_b, const at::Tensor& inv_w_rms,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t ff_inter, double eps);
void    rpu_gr00t_vl_encoder_set_weights_w8a16(
    int64_t handle,
    at::TensorList norm1_w_list, at::TensorList norm1_b_list,
    at::TensorList to_q_list, at::TensorList to_q_b_list,
    at::TensorList to_k_list, at::TensorList to_k_b_list,
    at::TensorList to_v_list, at::TensorList to_v_b_list,
    at::TensorList to_out_list, at::TensorList to_out_b_list,
    at::TensorList norm3_w_list, at::TensorList norm3_b_list,
    at::TensorList ff0_list, at::TensorList ff0_b_list,
    at::TensorList ff2_list, at::TensorList ff2_b_list,
    const at::Tensor& vlln_w, const at::Tensor& vlln_b, const at::Tensor& inv_w_rms,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t ff_inter, double eps,
    at::TensorList to_q_scale_list, at::TensorList to_k_scale_list,
    at::TensorList to_v_scale_list, at::TensorList to_out_scale_list,
    at::TensorList ff0_scale_list, at::TensorList ff2_scale_list);
at::Tensor rpu_gr00t_vl_encoder_forward(
    int64_t handle,
    const at::Tensor& raw,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor);
std::vector<int64_t> rpu_gr00t_vl_encoder_resolve_stage_domain(
    int64_t handle, int64_t seq_len);
int64_t rpu_gr00t_vl_encoder_get_resolved_chunk_size(int64_t handle);

// =============================================================================
// Pi0.5 denoise-loop driver (Phase 3 — plain driver op, NOT a FusedModelBase subclass).
// Implementation: src/fused/rpu_pi05_model.cpp
//
// CR-R5 iteration 5 (supersedes iteration 3/4):
//   - Finding 1: set_weights takes action_in_proj_{w,b} + action_out_proj_{w,b}.
//     Timestep-MLP weights stay Python-side (consumed once to build adarms_cond_list).
//     forward re-computes action_in_proj(x_t) + action_out_proj(suffix_tail) PER STEP.
//   - Finding 8: num_steps is a RUNTIME `forward` arg (not baked at set_weights).
//   - CR-R5 BLOCKER 2 (F10): NO per-token positional-offsets Tensor arg —
//     scalar `prefix_len` is passed to the existing `rpu_adarms_forward` C API's
//     `int64_t position` (constant across denoise steps). Incrementing by the
//     step index was a BUG.
//   - CR-R5 BLOCKER 3 (F9): per-entry `.clone()` of K/V at step start (inside
//     forward body) — matches lerobot modeling_pi05.py:889 copy.deepcopy contract.
// =============================================================================
int64_t rpu_pi05_create();
void rpu_pi05_destroy(int64_t handle);
void rpu_pi05_set_weights(
    int64_t handle,
    int64_t adarms_handle,
    const at::Tensor& action_in_proj_w,
    const at::Tensor& action_in_proj_b,
    const at::Tensor& action_out_proj_w,
    const at::Tensor& action_out_proj_b,
    int64_t chunk_size,
    int64_t max_action_dim,
    int64_t width);
at::Tensor rpu_pi05_forward(
    int64_t handle,
    const at::Tensor& initial_noise,
    const at::Tensor& adarms_cond_list,
    const std::optional<at::Tensor>& attention_mask,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t prefix_len,
    int64_t num_steps,
    double dt,
    at::IntArrayRef planned_stage_descriptor);

// =============================================================================
// Pi0.5 num_steps fused denoise step model.
// Implementation: src/fused/rpu_pi05_denoise_step_model.cpp
// The denoise loop retains its state across the requested action steps.
// =============================================================================

int64_t rpu_pi05_denoise_step_create(bool linear_acc32);
void rpu_pi05_denoise_step_set_execution_core_count(int64_t handle, int64_t cores);
std::vector<int64_t> rpu_pi05_denoise_step_get_execution_topology(int64_t handle);
void    rpu_pi05_denoise_step_destroy(int64_t handle);
void rpu_pi05_denoise_step_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);

void rpu_pi05_denoise_step_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w,
    at::TensorList v_w, at::TensorList o_w,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList attn_dense_w, at::TensorList attn_dense_b,
    at::TensorList mlp_dense_w,  at::TensorList mlp_dense_b,
    const at::Tensor& final_norm_dense_w, const at::Tensor& final_norm_dense_b,
    const at::Tensor& action_in_proj_w,   const at::Tensor& action_in_proj_b,
    const at::Tensor& action_out_proj_w,  const at::Tensor& action_out_proj_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t hidden_size, int64_t max_action_dim, int64_t chunk_size,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t num_layers, double eps,
    at::TensorList q_w_scale, at::TensorList k_w_scale,
    at::TensorList v_w_scale, at::TensorList o_w_scale,
    at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale,
    at::TensorList attn_dense_w_scale, at::TensorList mlp_dense_w_scale,
    const at::Tensor& final_norm_dense_w_scale, const std::optional<std::vector<at::Tensor>>& nvfp4_tensor_scales);

// Logical RoPE start for the action suffix (-1 = use the physical cache row).
void rpu_pi05_denoise_step_set_rope_position(int64_t handle, int64_t position);
void rpu_pi05_denoise_step_set_chunk_size(int64_t handle, int64_t chunk_size);
void rpu_pi05_denoise_step_set_adarms_table(int64_t handle, const at::Tensor& table);
int64_t rpu_pi05_denoise_step_get_resolved_chunk_size(int64_t handle);
std::vector<int64_t> rpu_pi05_denoise_step_resolve_action_stage_domain(
    int64_t handle,
    int64_t execution_len,
    int64_t logical_len,
    int64_t position,
    int64_t kv_len,
    int64_t cache_capacity,
    int64_t requested_chunk_size,
    bool prefer_pad16,
    bool loop_mode,
    int64_t num_steps);

void rpu_pi05_denoise_step_forward(
    int64_t handle,
    const at::Tensor& x_t_rpu,
    std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches,
    const at::Tensor& cond_step,
    const at::Tensor& attention_mask_4d,
    at::Tensor v_t_buf,
    int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor = {});

// N-step in-graph unroll (Task 1 scaffolding): one graph runs num_steps denoise
// iterations. Gated on loop_mode_ so the single-step op above stays untouched.
void rpu_pi05_denoise_loop_forward(
    int64_t handle,
    at::Tensor x0_rpu,
    std::vector<at::Tensor> k_caches,
    std::vector<at::Tensor> v_caches,
    at::Tensor cond_all,
    at::Tensor attention_mask_4d,
    at::Tensor x_out,
    double dt,
    int64_t prefix_len,
    int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor = {});

// =============================================================================
// Wall-OSS-0.5 fused action-denoise step model.
// Implementation: src/fused/rpu_wall_oss_action_step_model.cpp
// Fused Wall-OSS denoise-step interface.
// =============================================================================

int64_t rpu_wall_oss_action_step_create();
void    rpu_wall_oss_action_step_destroy(int64_t handle);

void rpu_wall_oss_action_step_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    at::TensorList q_w_scale, at::TensorList k_w_scale, at::TensorList v_w_scale,
    at::TensorList o_w_scale, at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale,
    const at::Tensor& w_comb, const at::Tensor& w3, const at::Tensor& proj_back,
    const at::Tensor& w2_t, const at::Tensor& w2_a,
    int64_t action_dim, int64_t k_comb_pad, int64_t chunk_size,
    const std::optional<at::Tensor>& q_tensor_scales = std::nullopt,
    const std::optional<at::Tensor>& k_tensor_scales = std::nullopt,
    const std::optional<at::Tensor>& v_tensor_scales = std::nullopt,
    const std::optional<at::Tensor>& o_tensor_scales = std::nullopt,
    const std::optional<at::Tensor>& gate_tensor_scales = std::nullopt,
    const std::optional<at::Tensor>& up_tensor_scales = std::nullopt,
    const std::optional<at::Tensor>& down_tensor_scales = std::nullopt);

at::Tensor rpu_wall_oss_action_step_forward(
    int64_t handle, const at::Tensor& x_t_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& b_step, const std::optional<at::Tensor>& te_step,
    const std::optional<at::Tensor>& attention_mask,
    const at::Tensor& position_ids,
    at::Tensor v_t_buf, at::Tensor x_out_buf, double dt, int64_t prefix_len,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il,
    at::IntArrayRef planned_stage_descriptor = {});

std::vector<int64_t> rpu_wall_oss_action_resolve_stage_domain(
    int64_t handle, int64_t horizon, int64_t prefix_len,
    int64_t mask_kv_len, int64_t requested_chunk_size,
    bool loop_mode, int64_t num_steps, bool b_step_is_bias,
    bool have_te);
int64_t rpu_wall_oss_action_get_resolved_chunk_size(int64_t handle);
void rpu_wall_oss_action_enable_execution_reconfigure(int64_t handle);
void rpu_wall_oss_action_stage_chunk_size(
    int64_t handle, int64_t token, int64_t chunk_size);

// N-step in-graph unroll (Phase-2): one graph runs num_steps denoise iterations.
void rpu_wall_oss_action_denoise_loop_forward(
    int64_t handle, const at::Tensor& x0_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& b_all, const std::optional<at::Tensor>& te_all,
    const std::optional<at::Tensor>& attention_mask, const at::Tensor& position_ids,
    at::Tensor x_traj, at::Tensor v_traj, double dt, int64_t prefix_len, int64_t num_steps,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il,
    at::IntArrayRef planned_stage_descriptor = {});

// LingBot-VLA fused action-expert denoise unroll model.
// Implementation: src/fused/rpu_lingbot_denoise_model.cpp.

int64_t rpu_lingbot_denoise_create();
void    rpu_lingbot_denoise_destroy(int64_t handle);
void    rpu_lingbot_denoise_set_chunk_size(int64_t handle, int64_t chunk_size);
int64_t rpu_lingbot_denoise_get_resolved_chunk_size(int64_t handle);
std::vector<int64_t> rpu_lingbot_denoise_resolve_action_stage_domain(
    int64_t handle, int64_t execution_len, int64_t prefix_len,
    int64_t mask_kv_len);

void rpu_lingbot_denoise_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias, const at::Tensor& time_all,
    const at::Tensor& adarms_is, const at::Tensor& adarms_ish,
    const at::Tensor& adarms_ps, const at::Tensor& adarms_psh,
    int64_t action_dim, int64_t k_comb_pad, int64_t num_steps, int64_t chunk_size);

void rpu_lingbot_denoise_unroll_forward(
    int64_t handle, const at::Tensor& x0_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& state_emb, const std::optional<at::Tensor>& attention_mask,
    at::Tensor x_traj, double dt, int64_t prefix_len, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor = {});

// =============================================================================
// LingBot2 (V2) sparse-MoE action-expert decoder.
// Implementation: src/fused/rpu_lingbot_v2_moe_model.cpp
// Default OFF: reachable only via these ops (a V2 adapter opt-in).
// =============================================================================

int64_t rpu_lingbot_v2_moe_create(
    bool adarms_fused_bcast = false,
    bool bufonly = false, bool addr = false,
    int64_t schunk = 256 * 127, int64_t rchunk_requested = 0,
    bool dense_soft_router = false, bool fp16_top4 = true,
    bool routed_only = false, bool down_acc16 = false);
void    rpu_lingbot_v2_moe_destroy(int64_t handle);

void rpu_lingbot_v2_moe_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList shared_gate_w, at::TensorList shared_up_w, at::TensorList shared_down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t shared_inter_pad, double eps,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    at::TensorList router_gate_w, at::TensorList router_bias,
    at::TensorList expert_gate_w, at::TensorList expert_up_w, at::TensorList expert_down_w,
    int64_t num_experts, int64_t top_k, int64_t routed_inter,
    double routed_scaling, int64_t chunk_size,
    bool grouped_experts_requested);

// Cold, optional selection-only centered FP16 router weights; original score
// weights and normalisation remain unchanged. Requires installed zero bias.
void rpu_lingbot_v2_moe_set_router_rank_weights(
    int64_t handle, at::TensorList rank_weights);

void rpu_lingbot_v2_moe_set_adarms_step_mutable(
    int64_t handle,
    const at::Tensor& input_scale, const at::Tensor& input_shift,
    const at::Tensor& post_scale,  const at::Tensor& post_shift);
void rpu_lingbot_v2_moe_bind_adarms_schedule(
    int64_t handle,
    const at::Tensor& input_scale, const at::Tensor& input_shift,
    const at::Tensor& post_scale,  const at::Tensor& post_shift);
void rpu_lingbot_v2_moe_set_denoise_unroll(
    int64_t handle,
    const at::Tensor& state_w, const at::Tensor& state_b,
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias,
    const at::Tensor& time_all,
    const at::Tensor& adarms_is, const at::Tensor& adarms_ish,
    const at::Tensor& adarms_ps, const at::Tensor& adarms_psh,
    int64_t state_dim, int64_t action_dim, int64_t state_dim_pad,
    int64_t action_dim_pad, int64_t num_steps);
void rpu_lingbot_v2_moe_denoise_unroll_forward(
    int64_t handle, const at::Tensor& x0_rpu, const at::Tensor& state_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const std::optional<at::Tensor>& attention_mask, at::Tensor x_final,
    double dt, int64_t prefix_len, int64_t cos_sin_offset,
    int64_t num_steps, at::IntArrayRef planned_stage_descriptor = {});
std::vector<int64_t> rpu_lingbot_v2_moe_resolve_action_stage_domain(
    int64_t handle, int64_t execution_len, int64_t prefix_len,
    int64_t mask_kv_len, int64_t cos_sin_offset);

// DEBUG-ONLY router read-back: [nl, E, Tp] device-computed routing weights.
at::Tensor rpu_lingbot_v2_moe_debug_rw_stage(int64_t handle);
at::Tensor rpu_lingbot_v2_moe_debug_wtm_stage(int64_t handle);
at::Tensor rpu_lingbot_v2_moe_debug_l0_out(int64_t handle);
at::Tensor rpu_lingbot_v2_moe_debug_innorm(int64_t handle);
at::Tensor rpu_lingbot_v2_moe_debug_gcap(int64_t handle);
at::Tensor rpu_lingbot_v2_moe_debug_acap(int64_t handle);
at::Tensor rpu_lingbot_v2_moe_debug_scap(int64_t handle);
at::Tensor rpu_lingbot_v2_moe_debug_ccap(int64_t handle);
// DEBUG-ONLY: hand back the DDR-resident packed expert weight actually bound to the
// device path (which: 0=gate, 1=up, 2=down). Gated on RPU_L2_DBG_PACKED=1 -- throws
// when the flag is absent, so no default-path caller can reach it.
at::Tensor rpu_lingbot_v2_moe_debug_packed(int64_t handle, int64_t which, int64_t layer);
// GROUPED-EXPERTS opt-in: bind per-layer core-slice-interleaved packed expert weights.
void rpu_lingbot_v2_moe_set_packed_weights(
    int64_t handle, at::TensorList gate_packed, at::TensorList up_packed,
    at::TensorList down_packed);
// W8A16 opt-in: per-output-channel fp16 scales for int8 packed expert weights.
void rpu_lingbot_v2_moe_set_packed_scales(
    int64_t handle, at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale);
// W8A16 opt-in for attention q/k/v/o + shared expert gate/up/down (router excluded).
void rpu_lingbot_v2_moe_set_base_scales(
    int64_t handle, at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws, at::TensorList down_ws);
at::Tensor rpu_lingbot_v2_moe_debug_router_h(int64_t handle);
at::Tensor rpu_lingbot_v2_moe_debug_layer_in(int64_t handle);
at::Tensor rpu_lingbot_v2_moe_debug_attn_resid(int64_t handle);

at::Tensor rpu_lingbot_v2_moe_forward(
    int64_t handle, const at::Tensor& hidden_states,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const std::optional<at::Tensor>& attention_mask, int64_t position,
    // RoPE position base; -1 = same as `position` (back-compat). >=0 decouples
    // the RoPE index from the KV-insert offset — REQUIRED for the V2 action
    // expert (M-RoPE-compressed suffix positions != uncompressed KV offset).
    int64_t cos_sin_offset = -1,
    // Controlled schedule path: >=0 selects the immutable AdaRMS schedule slice
    // before entering the cached graph; -1 preserves the legacy setter contract.
    int64_t adarms_schedule_step = -1,
    at::IntArrayRef planned_stage_descriptor = {});

// =============================================================================
// HALO action expert (mode="act") fused MoT step model.
// Implementation: src/fused/rpu_halo_action_expert_model.cpp
// 双权重 (text/act) 解码器, 行掩码合并; M1 boundary 在 host。
// =============================================================================

int64_t rpu_halo_action_expert_create();
void    rpu_halo_action_expert_destroy(int64_t handle);
void rpu_halo_action_expert_set_chunk_envelope(
    int64_t handle, int64_t max_kv_len, int64_t chunk_size);
std::vector<int64_t> rpu_halo_action_expert_resolve_stage_domain(
    int64_t handle, int64_t prefix_len, int64_t requested_chunk_size);

// text 主权重 (内部转 CausalDecoderModel::set_weights; q/k norm + QKV bias 必带,
// mrope/deepstack 空, fp16-only)。
void rpu_halo_action_expert_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias);

// act 孪生权重 + final_norm_moe + 行掩码 (text_row_mask [chunk_size] fp16 1/0)。
void rpu_halo_action_expert_set_moe_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    const at::Tensor& final_norm_moe_w, const at::Tensor& text_row_mask,
    int64_t chunk_size);

// 一次 act step: x_emb [1,cs,hidden] -> 返回末隐藏 [1,cs,hidden] (prefix-KV 非
// 因果 forward, 显式 4D additive mask)。返回的是基类 output_tensor_ (graph 写
// 目标), 调用方须在 capture 作用域退出后读 (已证模式 wall_oss/pi05)。
at::Tensor rpu_halo_action_expert_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor = {});

// =============================================================================
// Hy-VLA VLM prefill (MoT 双权重) — 自 HALO action expert 派生的 fused 子系统。
// 与 HALO 的差异：步 3 为 **RoPE→QK-norm**（HALO 为 norm→rope），且 q/k norm
// 两塔共享（不双算）；无 QKV bias；eps=1e-5。
// 实现见 src/fused/rpu_hyvla_vlm_model.cpp。
// =============================================================================
int64_t rpu_hyvla_vlm_create();
void    rpu_hyvla_vlm_destroy(int64_t handle);
void rpu_hyvla_vlm_set_runtime_config(
    int64_t handle,
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool partial_rope, int64_t mot_norm_nomerge, bool silu_mul,
    int64_t rmsnorm_capability);

// text 主权重 (内部转 CausalDecoderModel::set_weights; q/k norm + QKV bias 必带,
// mrope/deepstack 空, fp16-only)。
void rpu_hyvla_vlm_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias);

// W8A16 变体 (text 塔)：权重 int8 + 7 条 per-output-channel fp16 scale [N]。
// 与上面那个落到同一个 set_weights_hyvla —— scale 为空即 fp16 路径。
void rpu_hyvla_vlm_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws);

// act 孪生权重 + final_norm_moe + 行掩码 (text_row_mask [chunk_size] fp16 1/0)。
void rpu_hyvla_vlm_set_moe_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    const at::Tensor& final_norm_moe_w, const at::Tensor& text_row_mask,
    int64_t chunk_size);

// W8A16 变体 (`_v` 孪生)：与 text 侧**各自独立**的 7 条 scale。孪生权重不经
// 基类 CausalDecoderModel::set_weights ⇒ 校验在 set_moe_weights 里自己做。
void rpu_hyvla_vlm_set_moe_weights_w8a16(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    const at::Tensor& final_norm_moe_w, const at::Tensor& text_row_mask,
    int64_t chunk_size,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws);

// 一次 act step: x_emb [1,cs,hidden] -> 返回末隐藏 [1,cs,hidden] (prefix-KV 非
// 因果 forward, 显式 4D additive mask)。返回的是基类 output_tensor_ (graph 写
// 目标), 调用方须在 capture 作用域退出后读 (已证模式 wall_oss/pi05)。
at::Tensor rpu_hyvla_vlm_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor = {});
std::vector<int64_t> rpu_hyvla_vlm_resolve_stage_domain(
    int64_t handle, int64_t seq_len, int64_t prefix_len,
    int64_t mask_kv_len);
int64_t rpu_hyvla_vlm_get_resolved_chunk_size(int64_t handle);
void rpu_hyvla_vlm_set_chunk_size_override(
    int64_t handle, int64_t chunk_size);
void rpu_hyvla_vlm_enable_execution_reconfigure(int64_t handle);
void rpu_hyvla_vlm_stage_chunk_size_override(
    int64_t handle, int64_t token, int64_t chunk_size);

// =============================================================================
// Hy-VLA action expert denoise (单塔) — 第二座塔, 每个去噪步 51 行 suffix attend
// [VLM prefix KV | 自身 suffix KV]。与 VLM 塔同为 rope→QK-norm, 但**不做 MoT
// 双算** (suffix modality 恒 True ⇒ 只走 `_v`)。q/k head-norm 必须喂 **VLM 层**
// 的那一份 (两塔共享)。实现见 src/fused/rpu_hyvla_expert_model.cpp。
// =============================================================================
int64_t rpu_hyvla_expert_create();
void    rpu_hyvla_expert_destroy(int64_t handle);
void rpu_hyvla_expert_set_runtime_config(
    int64_t handle,
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool silu_mul, bool kvpad16, bool partial_rope,
    bool rmsnorm_pad16);

void rpu_hyvla_expert_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps,
    int64_t chunk_size, int64_t action_mlp_cores);

// W8A16 变体：权重是 int8 `[N,K]`（已按 element_size=1 ⇒ num_ele_32B=32 swizzle），
// 外加 7 条 per-output-channel fp16 scale `[N]`。除此之外与上面那个逐字节等价 ——
// 两者都落到同一个 set_weights_expert，scale 为空即 fp16 路径。
// 校验（全给/全空、int8、rank、numel==N）由基类 CausalDecoderModel::set_weights 做。
void rpu_hyvla_expert_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps,
    int64_t chunk_size, int64_t action_mlp_cores,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws);

at::Tensor rpu_hyvla_expert_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor = {});

// in-graph N 步 Euler unroll (opt-in)。set_action_weights 打开 unroll 模式,
// unroll_forward 一次 forward 跑完 num_steps 步 —— encoder / out_proj / Euler
// 全在图内, x 不离开 SPM。不调用这两个入口时, 上面的单步路径逐位不变。
void rpu_hyvla_expert_set_action_weights(
    int64_t handle,
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias, const at::Tensor& time_all,
    int64_t action_dim, int64_t num_steps);

void rpu_hyvla_expert_unroll_forward(
    int64_t handle, const at::Tensor& x0_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& state_emb, const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& attn_mask_4d, at::Tensor x_traj,
    double dt, int64_t prefix_len, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor = {});
std::vector<int64_t> rpu_hyvla_expert_resolve_stage_domain(
    int64_t handle, int64_t seq_len, int64_t prefix_len,
    int64_t mask_kv_len);
int64_t rpu_hyvla_expert_get_resolved_chunk_size(int64_t handle);
void rpu_hyvla_expert_set_chunk_size_override(
    int64_t handle, int64_t chunk_size);
void rpu_hyvla_expert_enable_execution_reconfigure(int64_t handle);
void rpu_hyvla_expert_stage_chunk_size_override(
    int64_t handle, int64_t token, int64_t chunk_size);

// =============================================================================
// HALO image flow denoise step (mode="gen") — KV_FIRST 多块非因果 双流 MoT。
// Implementation: src/fused/rpu_halo_image_flow_model.cpp
// query 1378 行 = 2 text(start/end_of_image, [0,1377])走 und/base 孪生 + 1376 vae
// 走 gen(*_moe_gen)孪生 ⇒ 双流行掩码合并 (= action expert ⊕ KV_FIRST)。image gen
// 全 attend ⇒ SDPA MASK_NONE (行掩码 ≠ SDPA mask)。组合 Gemma KV_FIRST + HALO
// Qwen2.5 层体 + per-forward cos/sin。
// =============================================================================

int64_t rpu_halo_image_flow_create();
void    rpu_halo_image_flow_destroy(int64_t handle);
void rpu_halo_image_flow_set_chunk_envelope(
    int64_t handle, int64_t max_kv_len, int64_t chunk_size);
std::vector<int64_t> rpu_halo_image_flow_resolve_stage_domain(
    int64_t handle, int64_t prefix_len, int64_t requested_chunk_size);

// MR-D / T35: force this handle's chunk (0 = auto, which is what production uses).
// step_forward used to read the process global on EVERY forward — the only model in
// the tree that did so directly — which let any other handle's set_chunk_size(N)
// silently re-lay-out this one. Tuning/measurement knob; cold-set.
void rpu_halo_image_flow_set_chunk_size_override(int64_t handle, int64_t chunk_size);

// und/base 权重 (内部转 CausalDecoderModel::set_weights; q/k norm + QKV bias 必带,
// mrope/deepstack 空, fp16-only)。与 action expert set_weights 同签名。喂入无后缀
// base 孪生 (start/end_of_image 那 2 个 text 行用)。
void rpu_halo_image_flow_set_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm_w,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size, double eps, bool use_silu,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias);

// gen(*_moe_gen) 孪生 + 行掩码 (= action expert set_moe_weights 角色, 减去
// chunk_size SDPA override)。row_mask [query_len] fp16 RPU {0,1} (1=und/text 行
// {0,1377}); query_len=1378。须在 set_weights 之后调用。
void rpu_halo_image_flow_set_gen_weights(
    int64_t handle,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w, at::TensorList o_w,
    at::TensorList q_norm, at::TensorList k_norm,
    at::TensorList input_norm, at::TensorList post_norm,
    at::TensorList gate_w, at::TensorList up_w, at::TensorList down_w,
    at::TensorList q_bias, at::TensorList k_bias, at::TensorList v_bias,
    const at::Tensor& final_norm_gen_w, const at::Tensor& row_mask,
    int64_t query_len);

// 一次 image flow denoise step: x_emb [1,seq,hidden] -> 末隐藏 [1,seq,hidden]
// (prefix-KV 非因果 forward, MASK_NONE 全 attend; KV_FIRST 内部多块)。cos/sin
// 为 per-forward 表 (CFG 三分支各异)。无 mask / chunk_size 入参。返回基类
// output_tensor_, 调用方须在 capture 作用域退出后读。
at::Tensor rpu_halo_image_flow_step_forward(
    int64_t handle, const at::Tensor& x_emb,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& cos, const at::Tensor& sin, int64_t prefix_len,
    at::IntArrayRef planned_stage_descriptor = {});

// =============================================================================
// QwenPI05 Fused Action Expert (all-layers-once, per-forward cos/sin)
// =============================================================================

// Create a new QwenPI05Model instance; returns its handle.
int64_t rpu_qwenpi05_create();

// Destroy a QwenPI05Model instance.
void rpu_qwenpi05_destroy(int64_t handle);
void rpu_qwenpi05_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);

// Set model weights. Dense-w lists are expanded+swizzled [8*3*hidden, hidden]
// fp16 (col-partition for 8-core GEMV). Dense-b lists are [3*hidden] fp16.
// q_norm/k_norm are per-layer [head_dim] fp16 for QK normalization.
// NOTE: No cos/sin here -- they are per-forward tensors passed to forward().
void rpu_qwenpi05_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps);

// W8A16 weights: expert, conditional projections, and action IO.
void rpu_rhino_vla_set_weights_full_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list, at::TensorList mlp_dense_b_list,
    at::TensorList pair_dense_w_list, at::TensorList pair_dense_b_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list,
    at::TensorList attn_dense_scale_list, at::TensorList mlp_dense_scale_list,
    at::TensorList pair_dense_scale_list);

void rpu_rhino_vla_set_denoise_loop_weights_w8a16(
    int64_t handle,
    const at::Tensor& action_in_w, const at::Tensor& action_in_b,
    const at::Tensor& action_time_in_w, const at::Tensor& action_time_in_b,
    const at::Tensor& time_in_action_w, const at::Tensor& time_in_time_w,
    const at::Tensor& time_in_b,
    const at::Tensor& time_out_w, const at::Tensor& time_out_b,
    const at::Tensor& state_w, const at::Tensor& state_b,
    const at::Tensor& state_mask_w, const at::Tensor& state_mask_b,
    const at::Tensor& action_mask_w, const at::Tensor& action_mask_b,
    const at::Tensor& final_norm_w, const at::Tensor& final_norm_b,
    const at::Tensor& action_out_w, const at::Tensor& action_out_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t action_dim, int64_t action_dim_pad,
    int64_t state_dim, int64_t state_dim_pad,
    int64_t action_horizon, int64_t suffix_len, bool direct_action_input,
    at::TensorList io_scales);

void rpu_rhino_vla_set_high_precision_fusions(int64_t handle, bool enabled);
void rpu_rhino_vla_set_vector_k_norm(int64_t handle, bool enabled);
void rpu_rhino_vla_set_vector_q_norm(int64_t handle, bool enabled);
void rpu_rhino_vla_set_partial_rope(int64_t handle, bool enabled);
void rpu_rhino_vla_set_high_precision(int64_t handle, bool enabled);
at::Tensor rpu_rhino_vla_tanh_high_precision(const at::Tensor& input);

void rpu_rhino_vla_set_denoise_loop_adarms_tables_w8a16(
    int64_t handle,
    const at::Tensor& pair_table,
    const at::Tensor& final_table,
    at::TensorList cold_weights, at::TensorList cold_scales, at::TensorList cold_biases,
    bool gates_tanh_precomputed, bool high_precision);

// All-layers-once forward. cos/sin are per-forward M-RoPE tensors
// pre-computed on CPU and DMA'd to RPU. cond is [hidden_size] fp16 on RPU.
at::Tensor rpu_qwenpi05_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    at::IntArrayRef planned_stage_descriptor);
std::vector<int64_t> rpu_qwenpi05_resolve_action_stage_domain(
    int64_t handle, int64_t execution_len, int64_t logical_len,
    int64_t position, int64_t kv_len, bool use_attention_mask,
    bool is_causal, int64_t requested_chunk_size);
int64_t rpu_qwenpi05_get_resolved_chunk_size(int64_t handle);

// Materialize per-core SPM debug dumps requested during graph build.
// Must be called AFTER a debug-export forward finishes. Copies SPM via
// direct CPU-mapped pointers into `g_debug_tensors` which Python reads via
// `rpu_backend.get_debug_tensor(name)`.
void rpu_qwenpi05_debug_flush_dumps(int64_t handle);

// =============================================================================
// RhinoVLAModel — independent fork of QwenPI05 (RhinoVLA Action Expert)
// =============================================================================

// Create a new RhinoVLAModel instance; returns its handle.
int64_t rpu_rhino_vla_create();

// Destroy a RhinoVLAModel instance.
void rpu_rhino_vla_destroy(int64_t handle);
void rpu_rhino_vla_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);

// Bind all graph-affecting RhinoVLA action selectors before model weights.
void rpu_rhino_vla_set_runtime_config(
    int64_t handle,
    bool denoise_static_context_cache,
    bool fused_adarms,
    bool skip_adarms_gemv,
    bool precompute_adarms,
    bool precompute_time_proj,
    bool fold_action_time_in,
    bool gated_no_sub,
    bool fused_silu_mul,
    bool expert_fusions = false,
    bool adarms_resident = false,
    bool packed_qkv = false,
    bool aligned_kv = false);

// Enable/disable replay-time skipping of RhinoVLA expert layer op emission.
void rpu_rhino_vla_set_fast_replay(int64_t handle, bool enabled);
void rpu_rhino_vla_set_rope_position(int64_t handle, int64_t position);
void rpu_rhino_vla_set_chunk_size(int64_t handle, int64_t chunk_size);
void rpu_rhino_vla_enable_execution_reconfigure(int64_t handle);
void rpu_rhino_vla_stage_chunk_size(
    int64_t handle, int64_t token, int64_t chunk_size);
int64_t rpu_rhino_vla_get_resolved_chunk_size(int64_t handle);
std::vector<int64_t> rpu_rhino_vla_resolve_action_stage_domain(
    int64_t handle, int64_t execution_len, int64_t logical_len,
    int64_t position, int64_t kv_len, bool use_attention_mask,
    bool is_causal, int64_t requested_chunk_size);

// Bind the actual loop schedule after weights and before any dry planning.
void rpu_rhino_vla_prepare_denoise_loop_schedule(
    int64_t handle, int64_t num_steps, double dt);
// Bind RhinoVLA-specific action IO / final norm weights for the Phase-3
// denoise loop path.
void rpu_rhino_vla_set_denoise_loop_weights(
    int64_t handle,
    const at::Tensor& action_in_w, const at::Tensor& action_in_b,
    const at::Tensor& action_time_in_w, const at::Tensor& action_time_in_b,
    const at::Tensor& time_in_action_w, const at::Tensor& time_in_time_w,
    const at::Tensor& time_in_b,
    const at::Tensor& time_out_w, const at::Tensor& time_out_b,
    const at::Tensor& state_w, const at::Tensor& state_b,
    const at::Tensor& state_mask_w, const at::Tensor& state_mask_b,
    const at::Tensor& action_mask_w, const at::Tensor& action_mask_b,
    const at::Tensor& final_norm_w, const at::Tensor& final_norm_b,
    const at::Tensor& action_out_w, const at::Tensor& action_out_b,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t action_dim, int64_t action_dim_pad,
    int64_t state_dim, int64_t state_dim_pad,
    int64_t action_horizon, int64_t suffix_len, bool direct_action_input = false);

void rpu_rhino_vla_set_denoise_loop_adarms_tables(
    int64_t handle,
    const at::Tensor& pair_table,
    const at::Tensor& final_table);

void rpu_rhino_vla_set_denoise_loop_time_proj_table(
    int64_t handle,
    const at::Tensor& table);

// Set model weights (same layout as QwenPI05; LoRA merged into base in Python).
void rpu_rhino_vla_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    at::TensorList pair_dense_w_list, at::TensorList pair_dense_b_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps);

// Expert W8A16 weights. Seven INT8 projections use natural FP16 [N]
// output-channel scales; all AdaRMS, QK norm and action IO weights stay FP16.
void rpu_rhino_vla_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list, at::TensorList mlp_dense_b_list,
    at::TensorList pair_dense_w_list, at::TensorList pair_dense_b_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list);

// All-layers-once forward. cos/sin are per-forward M-RoPE tensors
// pre-computed on CPU and DMA'd to RPU. cond is [hidden_size] fp16 on RPU.
at::Tensor rpu_rhino_vla_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    at::IntArrayRef planned_stage_descriptor);

at::Tensor rpu_rhino_vla_denoise_loop_forward(
    int64_t handle,
    const at::Tensor& x0_rpu,
    at::TensorList k_caches,
    at::TensorList v_caches,
    const at::Tensor& cond_all,
    const at::Tensor& attention_mask,
    const at::Tensor& state_rpu,
    const at::Tensor& state_mask_rpu,
    const at::Tensor& action_mask_rpu,
    at::Tensor x_out,
    double dt,
    int64_t prefix_len,
    int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor);

// Materialize per-core SPM debug dumps requested during graph build (RhinoVLA).
void rpu_rhino_vla_debug_flush_dumps(int64_t handle);

// =============================================================================
// SigLIP Patch Embedding Config (call once during model setup)
// =============================================================================

void rpu_siglip_set_patch_embedding_params(
    const at::Tensor& weight,
    const at::Tensor& pos_emb,
    int64_t kh, int64_t kw,
    int64_t cin_orig, int64_t cin_padded, int64_t cout,
    int64_t stride);

// =============================================================================
// SigLIP ViT Fused Encoder Layer
// =============================================================================

at::Tensor rpu_siglip_fused_encoder_forward(
    const at::Tensor& hidden_states,
    at::TensorList layer_weights,
    const at::Tensor& post_ln_weight, const at::Tensor& post_ln_bias,
    const at::Tensor& projector_weight, const at::Tensor& projector_bias,
    int64_t num_layers, double eps, int64_t num_heads,
    int64_t head_dim, int64_t intermediate_size, int64_t projection_dim);

// Legacy v2 entry point (renamed for A/B testing, Phase 5 D-503)
at::Tensor rpu_siglip_fused_encoder_forward_legacy(
    const at::Tensor& hidden_states,
    at::TensorList layer_weights,
    const at::Tensor& post_ln_weight, const at::Tensor& post_ln_bias,
    const at::Tensor& projector_weight, const at::Tensor& projector_bias,
    int64_t num_layers, double eps, int64_t num_heads,
    int64_t head_dim, int64_t intermediate_size, int64_t projection_dim);

// =============================================================================
// SigLIPModel All-Layers-Once (Phase 5: FusedModelBase migration)
// =============================================================================

int64_t rpu_siglip_create(const std::optional<bool>& linear_acc32);
void rpu_siglip_set_execution_core_count(int64_t handle, int64_t cores);
std::vector<int64_t> rpu_siglip_get_execution_topology(int64_t handle);
void rpu_siglip_destroy(int64_t handle);

void rpu_siglip_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    const at::Tensor& post_ln_w, const at::Tensor& post_ln_b,
    const at::Tensor& proj_w, const at::Tensor& proj_b,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    int64_t projection_dim, double eps);

void rpu_siglip_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    const at::Tensor& post_ln_w, const at::Tensor& post_ln_b,
    const at::Tensor& proj_w, const at::Tensor& proj_b,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    int64_t projection_dim, double eps,
    at::TensorList q_ws_list, at::TensorList k_ws_list,
    at::TensorList v_ws_list, at::TensorList o_ws_list,
    at::TensorList fc1_ws_list, at::TensorList fc2_ws_list);

void rpu_siglip_model_set_patch_emb(
    int64_t handle,
    const at::Tensor& weight,
    const at::Tensor& pos_emb,
    int64_t kernel_size,
    int64_t stride);
void rpu_siglip_set_chunk_size(int64_t handle, int64_t chunk_size);
void rpu_siglip_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);
int64_t rpu_siglip_get_resolved_chunk_size(int64_t handle);
int64_t rpu_siglip_resolve_chunk_size(
    int64_t handle, int64_t seq_len, int64_t packed_image_count);
void rpu_siglip_prepare_persistent_spm(
    int64_t handle, int64_t seq_len, int64_t packed_image_count);
std::vector<int64_t> rpu_siglip_resolve_stage_domain(
    int64_t handle, int64_t seq_len, int64_t packed_image_count,
    bool external_patch_prologue);

at::Tensor rpu_siglip_patch_embed(
    int64_t handle,
    const at::Tensor& input);

at::Tensor rpu_siglip_patch_embed_multi(
    int64_t handle,
    at::TensorList images);

at::Tensor rpu_siglip_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor);

at::Tensor rpu_siglip_forward_packed(
    int64_t handle,
    const at::Tensor& hidden,
    int64_t image_batch_count,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor);

// SigLIP-batch all-RPU path: takes the N camera images as a LIST (each
// [1,3,H,W]) instead of a torch.cat'd [N,3,H,W] tensor — the C++ packs them
// directly, so the Python caller no longer needs torch.cat (a CPU fallback
// that does NOT flush its result to DDR) nor the C++ a per-image slice.
at::Tensor rpu_siglip_forward_multi(
    int64_t handle,
    at::TensorList images,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor);

// =============================================================================
// HYViT2-400M AnyRes ViT (Hy-Embodied-0.5-VLA) — 自 SigLIP 派生的 fused 子系统。
// 与 SigLIP 的唯一接口差异：无 post-LayerNorm，故 set_weights 少 post_ln_w/b 两参。
// 实现见 src/fused/rpu_hyvit2_vision_model.cpp。
// =============================================================================
int64_t rpu_hyvit2_create();
void rpu_hyvit2_destroy(int64_t handle);
void rpu_hyvit2_set_runtime_config(
    int64_t handle,
    bool fast_replay, bool fast_replay_preload, bool mask_once,
    bool kvpad16, bool q_inplace);

void rpu_hyvit2_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    const at::Tensor& proj_w, const at::Tensor& proj_b,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    int64_t projection_dim, double eps,
    bool fused_merger, bool proj1_in_merger,
    int64_t patch_embed_cores);

void rpu_hyvit2_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    const at::Tensor& proj_w, const at::Tensor& proj_b,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    int64_t projection_dim, double eps,
    bool fused_merger, bool proj1_in_merger,
    int64_t patch_embed_cores,
    at::TensorList q_ws_list, at::TensorList k_ws_list,
    at::TensorList v_ws_list, at::TensorList o_ws_list,
    at::TensorList fc1_ws_list, at::TensorList fc2_ws_list);

void rpu_hyvit2_model_set_patch_emb(
    int64_t handle,
    const at::Tensor& weight,
    const at::Tensor& pos_emb,
    int64_t kernel_size,
    int64_t stride);

at::Tensor rpu_hyvit2_patch_embed(
    int64_t handle,
    const at::Tensor& input);

at::Tensor rpu_hyvit2_patch_embed_multi(
    int64_t handle,
    at::TensorList images);

// Hy-VLA merger 的组轴（DwPooler）两个 op —— 输入必须是 member-major [4, G, D]
// （由 ViT 尾部的两次 permute3d 产出）。实现与理由见 src/ops/rpu_hyvla_merger.cpp。
//   pool    : 返回 Σ_m nxc[m] 平铺成 [4,G,D]（**不除 4**，1/4 折进 host 侧的 Wb）
//   combine : 组内 softmax（成员轴）+ 加权求和 → [G, D]
at::Tensor rpu_hyvla_merger_pool(const at::Tensor& nxc);
at::Tensor rpu_hyvla_merger_combine(const at::Tensor& nxc, const at::Tensor& sc);

// merger 余部整条链的图内版本 —— 取代 host 的 4×F.linear + 2×F.gelu + 上面那两个
// immediate op。**只能在 `graph_cache.capture` 之内调用**，且调用前必须
// `spm_alloc_reset_temporary()`（峰值 ~7.07 MB/核）。理由见实现文件的长注释。
// wp1/bp1（可选）：`RPU_HY_VLA_PROJ1_IN_MERGER` 打开时由调用方传入 merger 的
// `proj1`，此时 `nxc` 是 proj1 **之前**的 [4,G,K]，本 op 先做一枪 col-partition
// GEMM + all_gather 再走原链路。理由与不新增 SPM 的做法见实现文件的开关注释。
at::Tensor rpu_hyvla_merger_fused(
    const at::Tensor& nxc,
    const at::Tensor& wa,  const at::Tensor& b0,
    const at::Tensor& wb,
    const at::Tensor& w2,  const at::Tensor& b2,
    const at::Tensor& wp2, const at::Tensor& bp2,
    const c10::optional<at::Tensor>& wp1 = c10::nullopt,
    const c10::optional<at::Tensor>& bp1 = c10::nullopt);

at::Tensor rpu_hyvit2_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor = {});

at::Tensor rpu_hyvit2_forward_packed(
    int64_t handle,
    const at::Tensor& hidden,
    int64_t image_batch_count,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor = {});

// SigLIP-batch all-RPU path: takes the N camera images as a LIST (each
// [1,3,H,W]) instead of a torch.cat'd [N,3,H,W] tensor — the C++ packs them
// directly, so the Python caller no longer needs torch.cat (a CPU fallback
// that does NOT flush its result to DDR) nor the C++ a per-image slice.
at::Tensor rpu_hyvit2_forward_multi(
    int64_t handle,
    at::TensorList images,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    at::IntArrayRef planned_stage_descriptor = {});
std::vector<int64_t> rpu_hyvit2_resolve_stage_domain(
    int64_t handle, int64_t seq_len, int64_t image_batch_count,
    bool external_patch_prologue);
int64_t rpu_hyvit2_get_resolved_chunk_size(int64_t handle);
void rpu_hyvit2_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);
void rpu_hyvit2_set_chunk_size_override(
    int64_t handle, int64_t chunk_size);
void rpu_hyvit2_enable_execution_reconfigure(int64_t handle);
void rpu_hyvit2_stage_chunk_size_override(
    int64_t handle, int64_t token, int64_t chunk_size);

at::Tensor rpu_siglip_fused_encoder_layer(
    const at::Tensor& hidden_states,
    const at::Tensor& ln1_weight, const at::Tensor& ln1_bias,
    const at::Tensor& q_weight, const at::Tensor& k_weight,
    const at::Tensor& v_weight, const at::Tensor& o_weight,
    const at::Tensor& q_bias, const at::Tensor& k_bias,
    const at::Tensor& v_bias, const at::Tensor& o_bias,
    const at::Tensor& ln2_weight, const at::Tensor& ln2_bias,
    const at::Tensor& fc1_weight, const at::Tensor& fc2_weight,
    const at::Tensor& fc1_bias, const at::Tensor& fc2_bias,
    const at::Tensor& post_ln_weight, const at::Tensor& post_ln_bias,
    const at::Tensor& projector_weight, const at::Tensor& projector_bias,
    int64_t layer_idx, int64_t num_layers,
    double eps, int64_t num_heads, int64_t head_dim,
    int64_t intermediate_size, int64_t projection_dim);


// =============================================================================
// Qwen3-VL Vision Encoder (R-Phase 3)
// =============================================================================
//
// 24-block ViT encoder. Architecture mirrors SigLIP-So400m with two additions:
//   - 2D RoPE applied to Q/K between fused QKV+bias and bidir SDPA.
//   - DeepStack snapshots emitted at user-specified layer indices.
// See src/fused/rpu_qwen3vl_vision_model.cpp for full design notes.

int64_t rpu_qwen3vl_vision_create(int64_t num_cores = 8);
std::vector<int64_t> rpu_qwen3vl_vision_topology(int64_t handle);
void rpu_qwen3vl_vision_destroy(int64_t handle);

void rpu_qwen3vl_vision_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::IntArrayRef deepstack_visual_indexes);
void rpu_qwen3vl_vision_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::IntArrayRef deepstack_visual_indexes,
    at::TensorList q_scale_list, at::TensorList k_scale_list,
    at::TensorList v_scale_list, at::TensorList o_scale_list,
    at::TensorList fc1_scale_list, at::TensorList fc2_scale_list);

void rpu_qwen3vl_vision_set_rope(
    int64_t handle,
    const at::Tensor& freq_cos,
    const at::Tensor& freq_sin,
    bool rope_spm_enabled);
void rpu_qwen3vl_vision_set_rope_route(
    int64_t handle,
    const at::Tensor& freq_cos,
    const at::Tensor& freq_sin,
    int64_t route_request);

// ROUND-3 opt #1: register the post-encoder patch-merger weights for the in-graph fold.
// RhinoVLA vision fast-replay opt-ins (per-model).
void rpu_qwen3vl_vision_set_fast_replay(int64_t handle, bool enabled);
void rpu_qwen3vl_vision_set_preload_replay_skip(int64_t handle, bool enabled);
void rpu_qwen3vl_vision_set_bake_merger(int64_t handle, bool enabled);
void rpu_qwen3vl_vision_set_chunk_size(int64_t handle, int64_t chunk_size);
void rpu_qwen3vl_vision_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);
void rpu_qwen3vl_vision_enable_execution_reconfigure(int64_t handle);
void rpu_qwen3vl_vision_stage_chunk_size(
    int64_t handle, int64_t token, int64_t chunk_size);
void rpu_qwen3vl_vision_set_rhinovla_high_precision(int64_t handle, bool enabled);
void rpu_qwen3vl_vision_set_linear_acc32(int64_t handle, bool enabled);
void rpu_qwen3vl_vision_set_large_image_auto_chunk(int64_t handle, bool enabled);
void rpu_qwen3vl_vision_set_n1200_tm160(int64_t handle, bool enabled);
void rpu_qwen3vl_vision_set_fp16_chunked_merger(int64_t handle, bool enabled);
void rpu_qwen3vl_vision_set_fp16_compact_encoder(int64_t handle, bool enabled);
void rpu_qwen3vl_vision_set_w8_compact_encoder(int64_t handle, bool enabled);
int64_t rpu_qwen3vl_vision_get_resolved_chunk_size(int64_t handle);
std::vector<int64_t> rpu_qwen3vl_vision_resolve_stage_domain(
    int64_t handle, int64_t num_patches, int64_t image_batch_count);
void rpu_qwen3vl_vision_set_prefix_scatter(
    int64_t handle, const at::Tensor& inputs_embeds, at::TensorList dense,
    at::IntArrayRef run_starts, at::IntArrayRef run_lengths);

void rpu_qwen3vl_vision_set_merger_weights(
    int64_t handle,
    const at::Tensor& ln_q_w, const at::Tensor& ln_q_b,
    const at::Tensor& m0_w, const at::Tensor& m0_b,
    const at::Tensor& m2_w, const at::Tensor& m2_b,
    int64_t out_hidden, double ln_eps,
    const c10::optional<at::Tensor>& m0_scale,
    const c10::optional<at::Tensor>& m2_scale);

// inc-2: register the N deepstack mergers' weights for the in-graph fold.
void rpu_qwen3vl_vision_set_deepstack_merger_weights(
    int64_t handle,
    at::TensorList ln_q_w, at::TensorList ln_q_b,
    at::TensorList m0_w, at::TensorList m0_b,
    at::TensorList m2_w, at::TensorList m2_b,
    const c10::optional<std::vector<at::Tensor>>& m0_scale,
    const c10::optional<std::vector<at::Tensor>>& m2_scale);

at::Tensor rpu_qwen3vl_vision_position_idx_keepalive(int64_t handle);

at::Tensor rpu_qwen3vl_vision_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t num_patches,
    int64_t image_batch_count,
    at::IntArrayRef planned_stage_descriptor);

std::vector<at::Tensor> rpu_qwen3vl_vision_pop_deepstack_snapshots(int64_t handle);
std::vector<at::Tensor> rpu_qwen3vl_vision_pop_deepstack_merged(int64_t handle);
at::Tensor rpu_qwen3vl_vision_pop_pooler_merged(int64_t handle);
std::vector<at::Tensor> rpu_qwen3vl_vision_pop_merged(int64_t handle);


// =============================================================================
// Qwen3.5 Vision Encoder (Option 2 — qwen3vl vision − DeepStack). Same 24-block
// ViT (2D RoPE + fused QKV) but no DeepStack snapshots. See
// src/fused/rpu_qwen3_5_vision_model.cpp for full design notes.

int64_t rpu_qwen3_5_vision_create();
void rpu_qwen3_5_vision_set_execution_cores(int64_t handle, int64_t num_cores);
std::vector<int64_t> rpu_qwen3_5_vision_get_execution_topology(int64_t handle);
void rpu_qwen3_5_vision_destroy(int64_t handle);
void rpu_qwen3_5_vision_set_linear_acc32(int64_t handle, bool enabled);
void rpu_qwen3_5_vision_set_moe_precision(
    int64_t handle, bool linear_acc32, bool gelu_erf_ultra);
void rpu_qwen3_5_vision_set_gelu_erf_ultra(int64_t handle, bool enabled);

void rpu_qwen3_5_vision_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList fc1_w_list, at::TensorList fc2_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList fc1_b_list, at::TensorList fc2_b_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps);

void rpu_qwen3_5_vision_set_rope(
    int64_t handle,
    const at::Tensor& freq_cos,
    const at::Tensor& freq_sin);

at::Tensor rpu_qwen3_5_vision_position_idx_keepalive(int64_t handle);

// Opt into in-graph STEP 0 (patch_embed + uploaded HF-exact positions). Flips
// forward()'s input contract to raw folded patches [1, N, patch_dim]; see
// set_patch_embed() in rpu_qwen3_5_vision_model.h. Not calling it keeps the CPU
// path unchanged.
void rpu_qwen3_5_vision_set_patch_embed(
    int64_t handle,
    const at::Tensor& pe_w,        // [hidden, patch_dim] fp16, COL-swizzled
    const at::Tensor& pe_b);       // [hidden] fp16
// Opt into the in-graph patch merger (post_layers_fn). Reads the tower's full DDR
// output back and projects each 2×2 patch block (spatial_merge_size²=4) to the
// text hidden size. Weights are stored SWIZZLED-AS-IS (Python pre-swizzles) —
// fc1_w COL-swizzled (partition=1), fc2_w ROW-swizzled (partition=0). The tower's
// forward return is UNCHANGED; read the merged result via get_merger_out().
void rpu_qwen3_5_vision_set_merger(
    int64_t handle,
    const at::Tensor& fc1_w, const at::Tensor& fc1_b,   // [hidden*4, hidden*4] / [hidden*4]
    const at::Tensor& fc2_w, const at::Tensor& fc2_b,   // [out_hidden, hidden*4] / [out_hidden]
    const at::Tensor& norm_w, const at::Tensor& norm_b, // [hidden] ln_q gamma/beta
    int64_t out_hidden_size);
void rpu_qwen3_5_vision_set_temporal(
    int64_t handle,
    const at::Tensor& temporal_pe,
    int64_t num_frames,
    int64_t patches_per_frame);
// emit_merger's output for the last forward — [1, N/4, out_hidden], == HF's
// merger(last_hidden_state). Plain DDR.
at::Tensor rpu_qwen3_5_vision_get_merger_out(int64_t handle);

at::Tensor rpu_qwen3_5_vision_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t num_patches,
    const std::optional<at::Tensor>& step0_pos = std::nullopt,
    const std::optional<at::Tensor>& fusion_target = std::nullopt,
    at::IntArrayRef fusion_row_starts = {},
    int64_t temporal_num_frames = 1,
    int64_t camera_batch_count = 1,
    int64_t expected_stage_plan_fingerprint_hi = 0,
    int64_t expected_stage_plan_fingerprint_lo = 0,
    int64_t expected_layout_hash_hi = 0,
    int64_t expected_layout_hash_lo = 0,
    at::IntArrayRef planned_stage_descriptor = {});

// Per-handle SPM-budget-resolved chunk_size (0 before any forward). Mirrors
// rpu_qwen3_5_get_resolved_chunk_size — lets the test read the actual cs.
int64_t rpu_qwen3_5_vision_get_resolved_chunk_size(int64_t handle);
void rpu_qwen3_5_vision_prepare_persistent_spm(int64_t handle, int64_t execution_len);
// Actual per-handle cap and exact override, not the last selected chunk.
std::vector<int64_t> rpu_qwen3_5_vision_get_prefill_execution_controls(int64_t handle);
// Planning-only query: run the same SPM/kernel feasibility resolver used by
// vision forward without recording or mutating a graph.
std::vector<int64_t> rpu_qwen3_5_vision_resolve_prefill_plan(
    int64_t handle, int64_t execution_len, int64_t temporal_num_frames,
    int64_t camera_batch_count, bool compact_input);
std::vector<std::vector<int64_t>> rpu_qwen3_5_vision_resolve_prefill_domain(
    int64_t handle, int64_t execution_len, int64_t temporal_num_frames,
    int64_t camera_batch_count, bool compact_input);
// MR-C/T34: cold PER-HANDLE vision chunk cap (was a process-wide static).
void rpu_qwen3_5_vision_set_chunk_size_cap(int64_t handle, int64_t cap);
void rpu_qwen3_5_vision_set_prefill_chunk_size(
    int64_t handle, int64_t chunk_size);
void rpu_qwen3_5_vision_stage_prefill_execution_controls(
    int64_t handle, int64_t token, int64_t cap, int64_t chunk_size);

// Per-layer debug snapshots (empty until a get_debug_export() forward). dbg_hidden =
// [num_layers, N, hidden] Phase-2 output; dbg_q = [num_layers, NUM_CORES, N, local_q_dim].
at::Tensor rpu_qwen3_5_vision_get_dbg_hidden(int64_t handle);
at::Tensor rpu_qwen3_5_vision_get_dbg_q(int64_t handle);


// =============================================================================
// Qwen2.5-VL Vision Encoder (Wall-OSS-0.5 P3). RMSNorm + biased SwiGLU + window
// attention. See src/fused/rpu_qwen25vl_vision_model.cpp.
// =============================================================================

int64_t rpu_qwen25vl_vision_create();
void rpu_qwen25vl_vision_destroy(int64_t handle);
void rpu_qwen25vl_vision_set_per_window_sdpa(int64_t handle, bool enabled);
void rpu_qwen25vl_vision_set_execution_routes(
    int64_t handle, bool rope_spm, bool fused_merger);

void rpu_qwen25vl_vision_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_w_list, at::TensorList up_w_list, at::TensorList down_w_list,
    at::TensorList norm1_w_list, at::TensorList norm2_w_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList gate_b_list, at::TensorList up_b_list, at::TensorList down_b_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, at::IntArrayRef fullatt_block_indexes);

void rpu_qwen25vl_vision_set_weights_w8a16(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_w_list, at::TensorList up_w_list, at::TensorList down_w_list,
    at::TensorList norm1_w_list, at::TensorList norm2_w_list,
    at::TensorList q_b_list, at::TensorList k_b_list,
    at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList gate_b_list, at::TensorList up_b_list, at::TensorList down_b_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, at::IntArrayRef fullatt_block_indexes,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list);

void rpu_qwen25vl_vision_set_merger_weights(
    int64_t handle,
    const at::Tensor& ln_q_w, const at::Tensor& m0_w, const at::Tensor& m0_b,
    const at::Tensor& m2_w, const at::Tensor& m2_b, int64_t out_hidden);

void rpu_qwen25vl_vision_set_rope(
    int64_t handle,
    const at::Tensor& freq_cos,
    const at::Tensor& freq_sin);

void rpu_qwen25vl_vision_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);
void rpu_qwen25vl_vision_set_chunk_size(
    int64_t handle,
    int64_t chunk_size);
std::vector<int64_t> rpu_qwen25vl_vision_resolve_stage_domain(
    int64_t handle, int64_t num_patches, int64_t image_batch_count,
    bool window_mask_present,
    const std::optional<at::Tensor>& cu_window_seqlens,
    int64_t requested_chunk_size);
int64_t rpu_qwen25vl_vision_get_resolved_chunk_size(int64_t handle);
void rpu_qwen25vl_vision_enable_execution_reconfigure(int64_t handle);
void rpu_qwen25vl_vision_stage_chunk_size(
    int64_t handle, int64_t token, int64_t chunk_size);

at::Tensor rpu_qwen25vl_vision_position_idx_keepalive(int64_t handle);

at::Tensor rpu_qwen25vl_vision_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t num_patches,
    std::optional<at::Tensor> window_mask,
    int64_t image_batch_count = 1,
    std::optional<at::Tensor> cu_window_seqlens = std::nullopt,
    at::IntArrayRef planned_stage_descriptor = {});


// =============================================================================
// DINOv3 ViT Vision Encoder (P4.1 skeleton: create / destroy only)
// =============================================================================
//
// 12-block (B/16) / 24-block (L/16) ViT encoder. Architecture vs Qwen3-VL:
//   - Separate q_proj/k_proj/v_proj Linears (no fused QKV); k bias=None.
//   - Explicit CLS + 4 register tokens; 2D RoPE applies to patch tokens only.
//   - LayerScale γ_attn / γ_mlp element-wise muls (P1.3: do NOT fold).
//   - Final LayerNorm + CLS slice as pooler; no merger, no DeepStack.
// Full design notes in src/fused/rpu_dinov3_vision_model.cpp.

int64_t rpu_dinov3_vision_create();
void rpu_dinov3_vision_destroy(int64_t handle);

void rpu_dinov3_vision_set_weights(
    int64_t handle,
    at::TensorList q_w_list,  at::TensorList k_w_list,
    at::TensorList v_w_list,  at::TensorList o_w_list,
    at::TensorList up_w_list, at::TensorList down_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list,  at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList up_b_list, at::TensorList down_b_list,
    at::TensorList gamma_attn_list, at::TensorList gamma_mlp_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps);

void rpu_dinov3_vision_set_rope(
    int64_t handle,
    const at::Tensor& freq_cos,
    const at::Tensor& freq_sin);

at::Tensor rpu_dinov3_vision_position_idx_keepalive(int64_t handle);
void rpu_dinov3_vision_set_chunk_size(int64_t handle, int64_t chunk_size);
void rpu_dinov3_vision_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);
void rpu_dinov3_vision_stage_chunk_size(
    int64_t handle, int64_t token, int64_t chunk_size);
int64_t rpu_dinov3_vision_get_chunk_size_override(int64_t handle);
int64_t rpu_dinov3_vision_resolve_chunk_size(
    int64_t handle, int64_t num_tokens);
std::vector<int64_t> rpu_dinov3_vision_resolve_stage_domain(
    int64_t handle, int64_t num_tokens);
int64_t rpu_dinov3_vision_get_resolved_chunk_size(int64_t handle);

void rpu_dinov3_vision_set_hybrid_cpu_sdpa(
    int64_t handle,
    const at::Tensor& sdpa_out,
    int64_t hybrid_layer_count);

void rpu_dinov3_vision_set_acc32(int64_t handle, bool enabled);
void rpu_dinov3_vision_set_16b_sdpa(int64_t handle, bool enabled);
void rpu_dinov3_vision_set_identity_rope(int64_t handle, bool enabled);
void rpu_dinov3_vision_set_final_norm(
    int64_t handle,
    const at::Tensor& weight,
    const at::Tensor& bias);
void rpu_dinov3_vision_set_output_mlp(
    int64_t handle,
    const at::Tensor& first_weight, const at::Tensor& first_bias,
    const at::Tensor& second_weight, const at::Tensor& second_bias);

at::Tensor rpu_dinov3_vision_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t num_tokens,
    at::IntArrayRef planned_stage_descriptor);


// =============================================================================
// Debug / Control Functions
// =============================================================================
//
// Note: set_debug_export / get_debug_export / get_debug_tensor / clear_debug_tensors /
//       list_debug_tensors / set_chunk_size / get_chunk_size / set_cross_layer_batch_size /
//       get_cross_layer_batch_size migrated to rpu_runtime_state.h by v5-07
//       (Phase 07-globals-extract; ADR §3 G1/G2/G3). Single source of truth.

void set_spm_debug(bool enabled);
bool get_spm_debug();

void set_cross_layer_batch_prefill(bool enabled);
bool get_cross_layer_batch_prefill();

// =============================================================================
// DMA Functions (DDR <-> SPM Memory Copy)
// =============================================================================





// =============================================================================
// DMA-based memcpy primitives (graph-aware with PASSTHROUGH fallback)
// =============================================================================
// Under RECORDING/REPLAYING these append DMA nodes to the active graph; under
// PASSTHROUGH the fixed variants synchronously execute a one-shot DMA batch.
// ddr_ptr used by a fixed captured DMA must point into stable storage.
// The DMA src/dst addresses are baked into kd_buf at build_batch time and
// cannot be updated by sync_mutable_params — callers must guarantee pointer
// stability for the lifetime of the active GraphCache slot.
//
// `spm_addr_unified`: the absolute physical SPM address returned by
// FusedModelBase::addr(0, name). It is the same value broadcast to every
// core (the underlying SpmAllocator stores per-core physical bases and
// addr(c, off) yields the per-core absolute). The launcher internally
// recovers the local offset via `spm_addr_unified - SPM_ALLOC.addr(0, 0)`
// and then composes the per-core DMA destinations as `SPM_ALLOC.addr(c, off)`.
//
// Captured DMA nodes retain the ownership and address contracts above.

// DDR → per-core SPM. One DMA per core on channels 0..num_cores-1, with
// 2·(num_cores-1) barriers framing the burst.
void rpu_launch_ddr_broadcast_spm_dma(
    c10::Half* ddr_ptr,
    int64_t   num_elements,
    uint32_t  spm_addr_unified,
    int       num_cores = 8);

// Tensor-owned fixed counterpart used by schema-5 outer fast replay.  The
// emitted DMA topology is identical to the raw-pointer overload; the Tensor
// and byte slice are retained only as private Graph provenance.
void rpu_launch_ddr_broadcast_spm_dma(
    const at::Tensor& ddr_tensor,
    int64_t           src_offset_elements,
    int64_t           num_elements,
    uint32_t          spm_addr_unified,
    int               num_cores = 8);

// DDR → per-core SPM, mutable variant. Same kd_buf layout as the immutable
// version, but src is rewritten per replay via Queue_t::update_dma_kernel
// when the caller-side DDR buffer changes
// per forward (e.g. fresh `hidden_states` tensors) — eliminates the prior
// at::copy_-to-staging workaround needed for graph-mode src baking.
//
// `live_src_base` points to caller-owned uint64_t storage holding the live
// RPU dev addr (RpuGetDevAddr of the per-forward tensor's data_ptr). The
// caller updates `*live_src_base` BEFORE each REPLAY of any GraphSlot that
// references it. `src_offset_bytes` is added at update time so a single
// live-base can feed multiple per-chunk DMAs at fixed offsets.
//
// Caller is also responsible for `rpu_ddr_flush` of the live buffer per
// forward — the wrapper does not flush (it would only flush the initial BUILD
// addr, not later REPLAY addrs).
//
// Captured mode re-dereferences the live source on replay; PASSTHROUGH
// dereferences it once and executes an immediate one-shot batch.
void rpu_launch_ddr_broadcast_spm_dma_mutable(
    const uint64_t* live_src_base,
    int64_t         src_offset_bytes,
    int64_t         num_elements,
    uint32_t        spm_addr_unified,
    int             num_cores = 8);

// Tensor-owned mutable counterpart.  `owner_tensor` is the current invocation
// owner of `live_src_base`; it is validated and retained independently of the
// live address before schema-5 can skip the host op walk.
void rpu_launch_ddr_broadcast_spm_dma_mutable(
    uint64_t&         live_src_base,
    const at::Tensor& owner_tensor,
    int64_t           src_offset_bytes,
    int64_t           num_elements,
    uint32_t          spm_addr_unified,
    int               num_cores = 8);

// core 0 SPM → DDR. Single DMA on channel 0 (stream 0) — same stream as
// surrounding compute, no barriers needed.
void rpu_launch_spm_copy_ddr_dma(
    uint32_t  spm_addr_unified,
    c10::Half* ddr_ptr,
    int64_t   num_elements);

void rpu_launch_spm_copy_ddr_dma(
    uint32_t          spm_addr_unified,
    const at::Tensor& ddr_tensor,
    int64_t           dst_offset_elements,
    int64_t           num_elements);

// FMB-only schema-v12 canonical Graph wrappers.  Physical peers and byte
// ranges are derived by FusedModelBase from a current-callback opaque member
// ref; raw callers cannot emit because Graph requires FMB's private burst
// permit.  These functions have no PASSTHROUGH fallback in the CPU-contract
// stage.
void rpu_launch_canonical_ddr_broadcast_spm_dma_fixed(
    const GraphDmaSemanticEndpoint& endpoint,
    uint64_t ddr_src_addr,
    size_t bytes,
    const std::array<uint64_t, 8>& spm_dst_addrs,
    int num_cores);
void rpu_launch_canonical_ddr_broadcast_spm_dma_mutable_src(
    const GraphDmaSemanticEndpoint& endpoint,
    const uint64_t* live_src_base,
    int64_t src_offset_bytes,
    size_t bytes,
    const std::array<uint64_t, 8>& spm_dst_addrs,
    int num_cores);
void rpu_launch_canonical_spm_copy_ddr_dma_fixed(
    const GraphDmaSemanticEndpoint& endpoint,
    uint64_t spm_src_addr,
    uint64_t ddr_dst_addr,
    size_t bytes);

// Mutable counterpart of rpu_launch_spm_copy_ddr_dma. Use when the DDR
// destination is reassigned each forward (output tensor whose data_ptr may
// shift across cached forwards, position-indexed per-forward write buffers).
// SPM source (core 0) is derived from SPM_ALLOC and stays stable per model
// instance — baked at BUILD; effective dst = `*live_dst_base + dst_offset_bytes`
// rewritten per replay via Queue_t::update_dma_kernel.
//
// Caller is responsible for `rpu_ddr_flush` of the live dst buffer per
// forward (wrapper does not flush). Graph-mode only.
void rpu_launch_spm_copy_ddr_dma_mutable(
    uint32_t        spm_addr_unified,
    const uint64_t* live_dst_base,
    int64_t         dst_offset_bytes,
    int64_t         num_elements);

void rpu_launch_spm_copy_ddr_dma_mutable(
    uint32_t          spm_addr_unified,
    uint64_t&         live_dst_base,
    const at::Tensor& owner_tensor,
    int64_t           dst_offset_bytes,
    int64_t           num_elements);

// Immediate (graph-outside) DMA variants. One-shot batches built on the
// pooled Queue_t and executed synchronously. Replace the immediate-mode
// branches of rpu_launch_ddr2spm_multicore / rpu_launch_spm2ddr_multicore at
// call sites that intentionally run outside graph capture (standalone
// preprocessing/debug paths).
void rpu_launch_ddr_broadcast_spm_dma_immediate(
    c10::Half* ddr_ptr,
    int64_t   num_elements,
    uint32_t  spm_addr_unified,
    int       num_cores = 8);

void rpu_launch_spm_copy_ddr_dma_immediate(
    uint32_t  spm_addr_unified,
    c10::Half* ddr_ptr,
    int64_t   num_elements);

// Byte-granular DDR → core-0 SPM copy (immediate). Raw byte count (not scaled by
// sizeof(Half)) for non-fp16 staging, e.g. the uint8 input of unary_cast_uint8_fp16.
void rpu_launch_ddr_spm_bytes_immediate(
    void*     ddr_ptr,
    int64_t   num_bytes,
    uint32_t  spm_addr_unified);

// Immediate scatter variants — one-shot batches built on the pooled Queue_t
// and executed synchronously. Use outside graph capture (preprocessing /
// debug dumps / standalone init). Each barrier + DMA layout
// matches the graph-mode scatter wrappers.
void rpu_launch_ddr_scatter_spm_dma_immediate(
    c10::Half* ddr_base,
    int64_t   elements_per_core,
    int64_t   core_stride_bytes,
    uint32_t  spm_addr_unified,
    int       num_cores);

void rpu_launch_spm_scatter_ddr_dma_immediate(
    uint32_t  spm_addr_unified,
    c10::Half* ddr_base,
    int64_t   elements_per_core,
    int64_t   core_stride_bytes,
    int       num_cores);

// =============================================================================
// DMA-based scatter primitives (graph-only). Replace the legacy kernel-based
// `rpu_launch_ddr2spm_multicore_scatter_read_broadcast_write` and
// `rpu_launch_spm2ddr_scatter` for production callers — frees compute streams
// and honours the caller's actual `num_cores` (the legacy spm2ddr_scatter
// wrapper hardcoded 8 cores regardless of caller intent).
//
// Layout:
//   ddr_scatter_spm_dma : core c reads ddr_base + c*core_stride_bytes,
//                         writes to its OWN core's SPM at spm_addr_unified
//                         (i.e. same local SPM offset on every core).
//   spm_scatter_ddr_dma : core c reads its OWN core's SPM at spm_addr_unified,
//                         writes to ddr_base + c*core_stride_bytes.
//
// Both: 1 DMA per core on channel c, with the canonical pre/post barrier
// fence (ch=1..N-1 vs stream 0). caller-supplied `num_cores` controls how
// many DMAs are emitted — no hardcoded NUM_CORES.
//
// `core_stride_bytes` MUST be 16-byte aligned (matches add_dma's bytes%16
// check). All current callers' strides are integer-multiples of fp16-size
// times per-core element counts that already satisfy this.
void rpu_launch_ddr_scatter_spm_dma(
    c10::Half* ddr_base,
    int64_t   elements_per_core,
    int64_t   core_stride_bytes,
    uint32_t  spm_addr_unified,
    int       num_cores);

void rpu_launch_ddr_scatter_spm_dma(
    const at::Tensor& ddr_tensor,
    int64_t           src_offset_elements,
    int64_t           elements_per_core,
    int64_t           core_stride_bytes,
    uint32_t          spm_addr_unified,
    int               num_cores);

// core-0 SPM contiguous slices → the same local offset on multiple cores.
void rpu_launch_spm_scatter_spm_dma(
    uint32_t  spm_src_unified,
    int64_t   elements_per_core,
    int64_t   core_stride_bytes,
    uint32_t  spm_dst_unified,
    int       num_cores);

// Each core extracts its own contiguous shard from a replicated full tensor:
// core c reads local src + c*stride and writes local dst. Unlike the scatter
// helper above, this never reads another core's SPM window.
void rpu_launch_spm_local_shard_copy_dma(
    uint32_t spm_src_unified,
    int64_t elements_per_core,
    int64_t core_stride_bytes,
    uint32_t spm_dst_unified,
    int num_cores);

// Same local SPM slice on each core -> another local slice on that core.
void rpu_launch_spm_local_copy_dma(
    uint32_t spm_src_unified,
    int64_t elements_per_core,
    uint32_t spm_dst_unified,
    int num_cores);

// Same local slice on cores [0,num_cores) → contiguous core-0 SPM slices.
void rpu_launch_spm_gather_spm_dma(
    uint32_t spm_src_unified,
    int64_t elements_per_core,
    uint32_t spm_dst_unified,
    int num_cores);

// Mutable variant — for callers whose DDR source tensor is reassigned per
// forward (e.g. AdaRMS `cond_ref_`). `live_src_base` points to caller-owned
// uint64_t storage holding the live RPU dev addr; caller updates it BEFORE
// each REPLAY. Per-core src = `*live_src_base + src_offset_bytes + c * core_stride_bytes`
// (the BUILD-time base offset is summed with the per-core scatter stride).
// Caller is responsible for `rpu_ddr_flush_force` of the live buffer per
// forward — wrapper does not flush.
void rpu_launch_ddr_scatter_spm_dma_mutable(
    const uint64_t* live_src_base,
    int64_t         src_offset_bytes,
    int64_t         elements_per_core,
    int64_t         core_stride_bytes,
    uint32_t        spm_addr_unified,
    int             num_cores);

void rpu_launch_ddr_scatter_spm_dma_mutable(
    uint64_t&         live_src_base,
    const at::Tensor& owner_tensor,
    int64_t           src_offset_bytes,
    int64_t           elements_per_core,
    int64_t           core_stride_bytes,
    uint32_t          spm_addr_unified,
    int               num_cores);

void rpu_launch_spm_scatter_ddr_dma(
    uint32_t  spm_addr_unified,
    c10::Half* ddr_base,
    int64_t   elements_per_core,
    int64_t   core_stride_bytes,
    int       num_cores);

// Tensor-owned fixed counterpart.  `dst_offset_elements` selects the first
// core's destination within the stable owner; the retained range extends
// through the final core's fixed-stride burst.
void rpu_launch_spm_scatter_ddr_dma(
    uint32_t          spm_addr_unified,
    const at::Tensor& ddr_tensor,
    int64_t           dst_offset_elements,
    int64_t           elements_per_core,
    int64_t           core_stride_bytes,
    int               num_cores);

// Mutable counterpart of rpu_launch_spm_scatter_ddr_dma. Use when the DDR
// destination tensor is reassigned each forward (caller-owned output tensor
// whose data_ptr may shift across cached forwards; the symmetric reason that
// `ddr_*_spm_dma_mutable` exists on the src side). Per-core dst =
// `*live_dst_base + dst_offset_bytes + c * core_stride_bytes`. SPM source is
// derived from SPM_ALLOC and stays stable per model instance — baked at BUILD.
//
// Like the src-mutable variant, the caller is responsible for `rpu_ddr_flush`
// of the live dst buffer per forward (wrapper does not flush — it would only
// flush the initial BUILD addr, not later REPLAY addrs).
//
// Captured mode re-dereferences the live destination on replay; PASSTHROUGH
// executes the equivalent immediate one-shot batch.
void rpu_launch_spm_scatter_ddr_dma_mutable(
    uint32_t        spm_addr_unified,
    const uint64_t* live_dst_base,
    int64_t         dst_offset_bytes,
    int64_t         elements_per_core,
    int64_t         core_stride_bytes,
    int             num_cores);



void rpu_launch_memset_spm_multicore(
    uint32_t spm_addr,
    int64_t num_elements, int num_cores = 8);

// Fill `num_elements` halfs with `value` at the per-core SPM addr (multi-core
// broadcast), via the oplib `fill` kernel. Replaces memset_spm_multicore where that
// kernel (memset_spm_multi_core_v2) is absent from the loaded operator library.
// `core_begin` fills cores [core_begin, core_begin+num_cores) instead of [0,n).
void rpu_launch_fill_spm_kernel(uint32_t spm_addr, int64_t num_elements,
                                c10::Half value, int num_cores,
                                int core_begin = 0);

// Strict exactly-k FP16 MoE selector. `correction_bias` affects selection only;
// `dense_weights` are gathered from the unbiased scores and normalised over k.
// The token-major operand supplies ranking values, independently of the
// expert-major normalisation scores. A zero-bias caller may use centered logits
// for ranking while retaining its original sigmoid scores for normalisation.
// All addresses are absolute core-0 SPM byte addresses. `scatter_ids_u16` is a
// raw uint16 [token_rows,num_experts] table whose value is expert*token_rows+token.
void rpu_launch_backend_moe_select_fp16_spm(
    uint32_t scores_token_major_addr,
    uint32_t scores_expert_major_addr,
    uint32_t correction_bias_addr,
    uint32_t scatter_ids_u16_addr,
    uint32_t choice_token_major_addr,
    uint32_t topk_keys_addr,
    uint32_t topk_ids_u16_addr,
    uint32_t topk_ids_i32_addr,
    uint32_t dense_mask_addr,
    uint32_t reduce_scratch_addr,
    uint32_t dense_weights_addr,
    int64_t token_rows,
    int64_t num_experts,
    int64_t top_k);

// =============================================================================
// =============================================================================
// Argmax/Argmin Operations
// =============================================================================

void rpu_launch_argmax_kernel(const at::Tensor &input, at::Tensor &output,
                              int64_t reduce_dim, bool select_last_index = false);
void rpu_launch_argmin_kernel(const at::Tensor &input, at::Tensor &output,
                              int64_t reduce_dim, bool select_last_index = false);
at::Tensor rpu_argmax(const at::Tensor &self, c10::optional<int64_t> dim,
                      bool keepdim = false);
at::Tensor rpu_argmin(const at::Tensor &self, c10::optional<int64_t> dim,
                      bool keepdim = false);
// Host-side argmax over the last dim of a contiguous 2-D tensor: CPU FP16/FP32,
// or RPU FP16 read in place via rpu_to_cpu_zerocopy (no D2H copy). Exists
// because `logits.cpu().float().argmax(-1)` costs ~1.25 ms/step on a
// [1, 151936] greedy-decode row. Ties → lowest index (matches torch.argmax).
at::Tensor rpu_argmax_lastdim_host(const at::Tensor &self);

// =============================================================================
// Conv2d Kernel Launchers
// =============================================================================

bool rpu_conv2d_fits_in_spm(int batch, int inh, int inw, int cin,
                             int cout, int kh, int kw, int groups,
                             int outh, int outw);

void rpu_launch_conv2d_spm_kernel(
    uint32_t spm_in_v16, uint32_t spm_flt_v16,
    uint32_t spm_out_v16, uint32_t spm_bias_v16,
    int batch, int cin, int inh, int inw,
    int cout, int kh, int kw,
    int padh, int padH, int padw, int padW,
    int strideh, int stridew, int dilationh, int dilationw,
    int groups, bool has_bias);

void rpu_launch_conv2d_ddr_kernel(
    const c10::Half* input_nhwc, const c10::Half* weight_nhwc,
    const c10::Half* bias_ptr, c10::Half* output_nhwc,
    int batch, int cin, int inh, int inw,
    int cout, int kh, int kw,
    int padh, int padH, int padw, int padW,
    int strideh, int stridew, int dilationh, int dilationw,
    int groups, bool has_bias);

at::Tensor rpu_conv2d(
    const at::Tensor& input, const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    at::IntArrayRef stride, at::IntArrayRef padding,
    at::IntArrayRef dilation, bool transposed,
    at::IntArrayRef output_padding, int64_t groups);

void rpu_release_conv2d_spm_cache();

bool rpu_conv2d_mc_fits_in_spm(int batch, int inh, int inw, int cin,
                                int cout, int kh, int kw,
                                int outh, int outw, int core_num);

void rpu_launch_conv2d_multicore_ddr_kernel(
    const c10::Half* input_nhwc, const c10::Half* weight_nhwc,
    const c10::Half* bias_ptr, c10::Half* output_nhwc,
    int batch, int cin, int inh, int inw,
    int cout, int kh, int kw,
    int padh, int padH, int padw, int padW,
    int strideh, int stridew, int dilationh, int dilationw,
    bool has_bias, int core_num);

// =============================================================================
// Im2col Kernel Launcher
// =============================================================================

// Extract patches from NHWC input into column matrix for GEMM.
// Input [batch, inh, inw, cin] → output [batch, outh*outw, flth*fltw*cin], all in SPM.
void rpu_launch_im2col_spm(
    uint32_t spm_in_off,    // SPM offset (bytes) for input
    uint32_t spm_col_off,   // SPM offset (bytes) for output
    int batch, int inh, int inw, int cin,
    int flth, int fltw,
    int padh, int padH, int padw, int padW,
    int strideh, int stridew,
    int holeh, int holew,
    int num_cores);

// =============================================================================
// Pad Kernel (constant fill, SPM in → SPM out)
// =============================================================================

// Pad the inner dimension of a 2D tensor: [N, C] → [N, C + pad_front + pad_tail].
// For NCHW channel padding: reshape [1, cin, H, W] as [1, cin*H*W], pad tail.
void rpu_launch_pad_channel_spm(
    uint32_t spm_in_off,   // SPM offset for input [N, C]
    uint32_t spm_out_off,  // SPM offset for output [N, CP]
    int64_t N,             // outer dimension
    int64_t C,             // input inner dimension
    int64_t pad_front,     // elements to pad before
    int64_t pad_tail,      // elements to pad after
    int num_cores = 1);    // >1: SPMD replicate (per-core input at same offset)

// =============================================================================
// Transpose Kernel (3D, SPM in → SPM out)
// =============================================================================

// Transpose [C, H, W] → [H, W, C] in SPM (NCHW→NHWC with batch collapsed).
// C must be divisible by 16.
void rpu_launch_transpose_nchw_to_nhwc_spm(
    uint32_t spm_in_off,   // SPM offset for input [C, H, W]
    uint32_t spm_out_off,  // SPM offset for output [H, W, C]
    int C, int H, int W,
    int num_cores = 1);    // >1: SPMD replicate (per-core input at same offset)

// Transpose [B, N, C] → [N, B, C] in SPM. C must be divisible by 16.
void rpu_launch_transpose_bnc_to_nbc_spm(
    uint32_t spm_in_off, uint32_t spm_out_off,
    int B, int N, int C);

// =============================================================================
// Fused Patch Embedding (SigLIP ViT) — v5-05 D-07: extern decl deleted
// =============================================================================
// The body was lifted into rpu_siglip_model.cpp as an unnamed-namespace
// TU-static helper named `fused_patch_embedding`; the standalone
// `rpu_fused_patch_embedding.cpp` source file is deleted (Ledger §1 S6).
// Only call site is SigLIPModel::forward 4D auto-detect.

// =============================================================================
// Shutdown / Cleanup
// =============================================================================

void rpu_shutdown();

// (v5-12 D-09: SmokeModelV3 op dispatcher forward decls removed; private to
//  librpu_conformance.so via tests/conformance/_cpp/forward_decls.h. Ledger §6 T1.)

// Controlled fused routes, consumed through COMPLETE physical descriptors.
void rpu_launch_partial_mrope_qk_spm_kernel(
    uint32_t q_x_spm_addr,
    uint32_t q_y_spm_addr,
    uint32_t k_x_spm_addr,
    uint32_t k_y_spm_addr,
    const at::Tensor& cos_il,
    const at::Tensor& sin_il,
    int64_t pos_offset,
    int64_t num_tokens,
    int64_t q_local_heads,
    int64_t k_local_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    int num_cores);

void rpu_qwen3vl_scatter_row_parts_unflushed_(
    const at::Tensor& destination,
    const at::Tensor& row_indices,
    at::TensorList sources);

void rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(
    const at::Tensor& destination,
    const at::Tensor& vocabulary,
    const at::Tensor& token_ids,
    int64_t skip_token_id);

void rpu_launch_silu_mul_exact_strided_spm_kernel(
    uint32_t packed_spm_addr,
    uint32_t out_spm_addr,
    int64_t rows,
    int64_t row_elements,
    int num_cores = 8);

void rpu_launch_insert_kv_cache_spm_unified_v16(
    at::Tensor &k_cache, at::Tensor &v_cache, int64_t position,
    uint32_t k_off, uint32_t v_off,
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores = 8);

void rpu_launch_insert_kv_cache_spm_unified_decode(
    at::Tensor &k_cache, at::Tensor &v_cache, int64_t position,
    uint32_t k_off, uint32_t v_off,
    int64_t seq_len, int64_t num_kv_heads, int64_t head_dim,
    int num_cores = 8);

void rpu_qwen3vl_vision_set_delivery_compatibility(int64_t handle, bool enabled);

void rpu_dinov3_vision_set_standalone_compatibility(int64_t handle, bool enabled);
