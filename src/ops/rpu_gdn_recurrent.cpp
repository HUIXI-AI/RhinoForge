// rpu_gdn_recurrent.cpp — single-head GDN (Gated-DeltaNet) recurrent step on SPM.
//
// Validates the recurrent delta-rule kernel composition against the official
// torch_recurrent_gated_delta_rule (per head). State is the OFFICIAL [Dk, Dv]
// layout, so the two reductions (kv, out) fall on the OUTER axis Dk and reduce
// to contiguous half-blocks — a 7-step pairwise-add tree, pure SPM, no DDR
// weight and no 256-byte-alignment trap:
//
//   state *= g_exp                       # scalar decay  (g_exp = exp(g), per head)
//   p     = k[:,None] * state            # Nx1 bcast (k over Dv cols)
//   kv    = sum_Dk(p)                    # [Dv]   tree-reduce contiguous halves
//   delta = (v - kv) * beta              # [Dv]
//   state += k[:,None] * delta[None,:]   # rank-1 outer (Nx1 ⊗ 1xC)
//   p     = q[:,None] * state
//   out   = sum_Dk(p)                    # [Dv]
//
// Caller pre-scales q by 1/sqrt(Dk) and L2-norms q/k. g_exp / beta arrive
// precomputed. Dk must be a power of two (Qwen3.5 GDN head_dim = 128).

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "rpu_ops.h"
#include "rpu_spm_allocator.h"
#include "rpu_eltwise.h"   // ValuOpType

using namespace ::rhino_lkn;

namespace {
// Reduce [rows, cols<256] SPM block to [1, cols] (column sums over rows) in a
// single launch via reduce_sum_non_last_dim_out_seg. Result lands in
// p_addr[0:cols] in place (the reduce accumulates,
// then writes the [cols] result once — in-place safe).
void tree_reduce_rows(uint32_t p_addr, int64_t rows, int64_t cols, int num_cores = 1) {
    rpu_launch_reduce_sum_rows_spm_kernel(p_addr, p_addr, rows, cols, num_cores);
}
}  // namespace

// ---------------------------------------------------------------------------
// Validated batched multi-head recurrent + on-device gating sequence, taking
// ABSOLUTE SPM addresses (so it runs identically in immediate mode and inside a
// fused graph). All inputs (q/k/v/state/A_log/a/b/dt) must already be in SPM;
// q/k are l2-normed and q is scaled in place; state is updated in place; the
// per-head output [H,Dv] is written CONTIGUOUSLY to a_out. Scratch: a_p (size
// H*Dk*Dv), a_delta (H*Dv), a_zero (Dk*Dv, must be pre-zeroed), a_gexp/a_beta/
// a_gs (H each). Matches torch_recurrent_gated_delta_rule (use_qk_l2norm=True).
// num_cores: 1 = single-core (all heads on core 0); 8 = SPMD, each core runs the
// SAME sequence on its OWN H heads at the same SPM offsets (GDN 2-head/core). The
// always-8-core kernels (scalar, Nx1_NxC, 1xC_NxC) work in both modes (single-
// core just wastes cores 1-7); the per-core-count kernels (unary/binary/l2norm/
// tree_reduce) take num_cores.
void rpu_emit_gdn_recurrent_multihead_seq(
    uint32_t a_q, uint32_t a_k, uint32_t a_v, uint32_t a_state,
    uint32_t a_p, uint32_t a_delta, uint32_t a_zero,
    uint32_t a_Al, uint32_t a_a, uint32_t a_b, uint32_t a_dt,
    uint32_t a_gexp, uint32_t a_beta, uint32_t a_gs,
    uint32_t a_out, int64_t H, int64_t Dk, int64_t Dv, int num_cores) {
    // -- Gating (batched over [H]) --
    rpu_launch_eltwise_unary_spm_kernel(a_b, a_beta, H, ValuOpType::SIGMOID, false, num_cores);
    rpu_launch_eltwise_binary_spm_kernel(a_a, a_dt, a_a, H, ValuOpType::ADD, c10::Half(1.0f), num_cores);
    rpu_launch_eltwise_unary_spm_kernel(
        a_a, a_a, H, ValuOpType::SOFTPLUS, false, num_cores);
    rpu_launch_eltwise_unary_spm_kernel(a_Al, a_gs, H, ValuOpType::EXP, false, num_cores);
    rpu_launch_eltwise_binary_scalar_spm_kernel(a_gs, c10::Half(-1.0f), a_gs, H, ValuOpType::MUL);
    rpu_launch_eltwise_binary_spm_kernel(a_gs, a_a, a_gs, H, ValuOpType::MUL, c10::Half(1.0f), num_cores);
    rpu_launch_eltwise_unary_spm_kernel(a_gs, a_gexp, H, ValuOpType::EXP, false, num_cores);

    // -- q/k prep (batched) --
    rpu_launch_l2norm_spm_kernel(a_k, a_k, H, Dk, 1e-6, num_cores);
    rpu_launch_l2norm_spm_kernel(a_q, a_q, H, Dk, 1e-6, num_cores);
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dk));
    rpu_launch_eltwise_binary_scalar_spm_kernel(a_q, c10::Half(scale), a_q, H * Dk, ValuOpType::MUL);

    // -- decay + product-k (batched; broadcast C ≤ Dv) --
    const uint32_t a_gexpk = a_p;  // reuse p scratch (overwritten by product next)
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        a_gexp, a_zero, a_gexpk, H, Dk, c10::Half(1.0f), ValuOpType::ADD, false);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        a_gexpk, a_state, a_state, H * Dk, Dv, c10::Half(1.0f), ValuOpType::MUL, false);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        a_k, a_state, a_p, H * Dk, Dv, c10::Half(1.0f), ValuOpType::MUL, false);

    const uint32_t hstride = (uint32_t)(Dk * Dv * 2);
    const uint32_t vstride = (uint32_t)(Dv * 2);
    const uint32_t kstride = (uint32_t)(Dk * 2);
    // -- per head: reduce kv, raw delta = v - kv --
    for (int64_t h = 0; h < H; ++h) {
        tree_reduce_rows(a_p + h * hstride, Dk, Dv, num_cores);
        rpu_launch_eltwise_binary_spm_kernel(
            a_v + h * vstride, a_p + h * hstride, a_delta + h * vstride,
            Dv, ValuOpType::SUB, c10::Half(1.0f), num_cores);
    }
    // -- beta scale (batched) --
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        a_beta, a_delta, a_delta, H, Dv, c10::Half(1.0f), ValuOpType::MUL, false);
    // -- per head: rank-1 state += k ⊗ delta --
    for (int64_t h = 0; h < H; ++h) {
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
            a_k + h * kstride, a_zero, a_p + h * hstride, Dk, Dv,
            c10::Half(1.0f), ValuOpType::ADD, false);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            a_delta + h * vstride, a_p + h * hstride, a_p + h * hstride, Dk, Dv,
            c10::Half(1.0f), ValuOpType::MUL, false);
        rpu_launch_eltwise_binary_spm_kernel(
            a_state + h * hstride, a_p + h * hstride, a_state + h * hstride,
            Dk * Dv, ValuOpType::ADD, c10::Half(1.0f), num_cores);
    }
    // -- product-q (batched) + per head reduce → contiguous out[H,Dv] --
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        a_q, a_state, a_p, H * Dk, Dv, c10::Half(1.0f), ValuOpType::MUL, false);
    for (int64_t h = 0; h < H; ++h) {
        tree_reduce_rows(a_p + h * hstride, Dk, Dv, num_cores);
        rpu_launch_eltwise_binary_spm_kernel(
            a_p + h * hstride, a_p + h * hstride, a_out + h * vstride,
            Dv, ValuOpType::MAX, c10::Half(1.0f), num_cores);   // SPM→SPM copy
    }
}
