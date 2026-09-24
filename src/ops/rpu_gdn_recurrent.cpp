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

// ---------------------------------------------------------------------------
// Validated batched multi-head recurrent + on-device gating sequence, taking
// ABSOLUTE SPM addresses (so it runs identically in immediate mode and inside a
// fused graph). All inputs (q/k/v/state/A_log/a/b/dt) must already be in SPM;
// q/k are l2-normed and q is scaled in place; state is updated in place; the
// per-head output [H,Dv] is written CONTIGUOUSLY to a_out. Scratch: a_p and
// a_scratch (both H*Dk*Dv), a_delta (H*Dv), a_zero (Dk*Dv, pre-zeroed), and
// a_gexp/a_beta/a_gs (H each). Matches torch_recurrent_gated_delta_rule.
// Each selected core runs the same sequence on its own H heads at the same
// SPM offsets. All kernels use that core prefix; inactive cores are untouched.
void rpu_emit_gdn_recurrent_multihead_seq(
    uint32_t a_q, uint32_t a_k, uint32_t a_v, uint32_t a_state,
    uint32_t a_p, uint32_t a_scratch, uint32_t a_delta, uint32_t a_zero,
    uint32_t a_Al, uint32_t a_a, uint32_t a_b, uint32_t a_dt,
    uint32_t a_gexp, uint32_t a_beta, uint32_t a_gs,
    uint32_t a_out, int64_t H, int64_t Dk, int64_t Dv, int num_cores) {
    // -- Gating --
    // Keep the proven stable softplus path here. The decode-only fused decay
    // kernel is intentionally not wired: for x=-8 its FP16 conformance returns
    // gexp~=1 instead of the expected ~=0.368.
    rpu_launch_eltwise_unary_spm_kernel(
        a_b, a_beta, H, ValuOpType::SIGMOID, false, num_cores);
    rpu_launch_eltwise_binary_spm_kernel(
        a_a, a_dt, a_a, H, ValuOpType::ADD, c10::Half(1.0f), num_cores);
    rpu_launch_eltwise_unary_spm_kernel(
        a_a, a_a, H, ValuOpType::SOFTPLUS, false, num_cores);
    rpu_launch_eltwise_unary_spm_kernel(a_Al, a_gs, H, ValuOpType::EXP, false, num_cores);
    rpu_launch_eltwise_binary_scalar_spm_kernel(a_gs, c10::Half(-1.0f), a_gs, H, ValuOpType::MUL, num_cores);
    rpu_launch_eltwise_binary_spm_kernel(
        a_gs, a_a, a_gs, H, ValuOpType::MUL, c10::Half(1.0f), num_cores);
    rpu_launch_eltwise_unary_spm_kernel(
        a_gs, a_gexp, H, ValuOpType::EXP, false, num_cores);

    // -- q/k prep (batched) --
    rpu_launch_l2norm_spm_kernel(a_k, a_k, H, Dk, 1e-6, num_cores);
    rpu_launch_l2norm_spm_kernel(a_q, a_q, H, Dk, 1e-6, num_cores);
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dk));
    rpu_launch_eltwise_binary_scalar_spm_kernel(a_q, c10::Half(scale), a_q, H * Dk, ValuOpType::MUL, num_cores);

    // -- decay: state *= g_exp --
    const uint32_t a_gexpk = a_p;
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        a_gexp, a_zero, a_gexpk, H, Dk, c10::Half(1.0f), ValuOpType::ADD, false, num_cores);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        a_gexpk, a_state, a_state, H * Dk, Dv, c10::Half(1.0f), ValuOpType::MUL, false, num_cores);
    const uint32_t hstride = (uint32_t)(Dk * Dv * 2);
    const uint32_t vstride = (uint32_t)(Dv * 2);
    const uint32_t kstride = (uint32_t)(Dk * 2);
    const uint32_t sstride = (uint32_t)(Dk * 16 * 2);
    const uint32_t a_splat = a_p;
    const uint32_t a_kv = a_out;

    // -- k·state readout: splat k to 16 lanes, then fused multiply-reduce. --
    for (int64_t h = 0; h < H; ++h) {
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
            a_k + h * kstride, a_zero, a_splat + h * sstride, Dk, 16,
            c10::Half(1.0f), ValuOpType::ADD, false, num_cores);
    }
    rpu_launch_qwen3_5_mul_reduce_rows(
        a_state, a_kv, a_splat, a_scratch, H, Dk, Dv, num_cores);
    for (int64_t h = 0; h < H; ++h) {
        rpu_launch_eltwise_binary_spm_kernel(
            a_v + h * vstride, a_kv + h * vstride, a_delta + h * vstride,
            Dv, ValuOpType::SUB, c10::Half(1.0f), num_cores);
    }
    // -- beta scale (batched) --
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        a_beta, a_delta, a_delta, H, Dv, c10::Half(1.0f), ValuOpType::MUL, false, num_cores);
    // -- rank-1 state += k ⊗ delta. --
    for (int64_t h = 0; h < H; ++h) {
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
            a_k + h * kstride, a_zero, a_p + h * hstride, Dk, Dv,
            c10::Half(1.0f), ValuOpType::ADD, false, num_cores);
    }
    rpu_launch_qwen3_5_rank1_fma(
        a_state, a_delta, a_p, H, Dk, Dv, num_cores);

    // -- q·state readout straight into contiguous out[H,Dv]. --
    for (int64_t h = 0; h < H; ++h) {
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
            a_q + h * kstride, a_zero, a_p + h * sstride, Dk, 16,
            c10::Half(1.0f), ValuOpType::ADD, false, num_cores);
    }
    rpu_launch_qwen3_5_mul_reduce_rows(
        a_state, a_out, a_p, a_scratch, H, Dk, Dv, num_cores);
}

// ---------------------------------------------------------------------------
// Batched multi-head GDN recurrent step + on-device gating.
//
// Layout: state [H, Dk, Dv] (head-major). Gating, l2norm, q-scale, decay and
// the k/q products are BATCHED (Nx1 broadcast over the head axis — so per-head
// scalars never need a single-scalar broadcast); the two reductions and the
// rank-1 update run per head (Dk-contiguous tree-reduce). Matches the official
// Qwen3_5GatedDeltaNet decode path (use_qk_l2norm_in_kernel=True).
// ---------------------------------------------------------------------------
std::tuple<at::Tensor, at::Tensor> rpu_gdn_recurrent_multihead_test(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& A_log, const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& dt_bias, const at::Tensor& state_in) {
    TORCH_CHECK(state_in.dim() == 3, "multihead: state must be [H,Dk,Dv]");
    TORCH_CHECK(q.scalar_type() == at::kHalf && state_in.scalar_type() == at::kHalf,
                "multihead: inputs must be fp16");
    TORCH_CHECK(q.device().type() == at::kPrivateUse1, "multihead: inputs must be on RPU");
    const int64_t H = state_in.size(0), Dk = state_in.size(1), Dv = state_in.size(2);
    TORCH_CHECK((Dk & (Dk - 1)) == 0, "multihead: Dk must be a power of two");
    TORCH_CHECK(q.size(0) == H && q.size(1) == Dk && k.size(1) == Dk && v.size(1) == Dv,
                "multihead: q/k [H,Dk], v [H,Dv]");
    TORCH_CHECK(A_log.size(0) == H && a.size(0) == H && b.size(0) == H && dt_bias.size(0) == H,
                "multihead: A_log/a/b/dt_bias must be [H]");

    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

    const int64_t sH = H * 2, sHk = H * Dk * 2, sHv = H * Dv * 2,
                  sstate = H * Dk * Dv * 2, szero = Dk * Dv * 2;
    const uint32_t o_state = SPM_ALLOC.alloc_temporary(sstate);
    const uint32_t o_p     = SPM_ALLOC.alloc_temporary(sstate);
    const uint32_t o_scratch = SPM_ALLOC.alloc_temporary(sstate);
    const uint32_t o_q     = SPM_ALLOC.alloc_temporary(sHk);
    const uint32_t o_k     = SPM_ALLOC.alloc_temporary(sHk);
    const uint32_t o_v     = SPM_ALLOC.alloc_temporary(sHv);
    const uint32_t o_delta = SPM_ALLOC.alloc_temporary(sHv);
    const uint32_t o_zero  = SPM_ALLOC.alloc_temporary(szero);
    const uint32_t o_Al    = SPM_ALLOC.alloc_temporary(sH);
    const uint32_t o_a     = SPM_ALLOC.alloc_temporary(sH);
    const uint32_t o_b     = SPM_ALLOC.alloc_temporary(sH);
    const uint32_t o_dt    = SPM_ALLOC.alloc_temporary(sH);
    const uint32_t o_gexp  = SPM_ALLOC.alloc_temporary(sH);
    const uint32_t o_beta  = SPM_ALLOC.alloc_temporary(sH);
    const uint32_t o_gs    = SPM_ALLOC.alloc_temporary(sH);
    const uint32_t o_out   = SPM_ALLOC.alloc_temporary(sHv);

    auto A = [&](uint32_t off) { return SPM_ALLOC.addr(0, off); };

    auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous(),
         sc = state_in.contiguous(), Alc = A_log.contiguous(), ac = a.contiguous(),
         bc = b.contiguous(), dtc = dt_bias.contiguous();
    auto zero = at::zeros({Dk, Dv}, at::kHalf).to(q.device());

    auto DMA = [&](const at::Tensor& t, uint32_t off, int64_t n) {
        rpu_launch_ddr_broadcast_spm_dma_immediate(
            const_cast<c10::Half*>(t.data_ptr<c10::Half>()), n, A(off), 1);
    };
    DMA(qc, o_q, H * Dk);   DMA(kc, o_k, H * Dk);   DMA(vc, o_v, H * Dv);
    DMA(sc, o_state, H * Dk * Dv);
    DMA(Alc, o_Al, H);  DMA(ac, o_a, H);  DMA(bc, o_b, H);  DMA(dtc, o_dt, H);
    DMA(zero, o_zero, Dk * Dv);

    rpu_emit_gdn_recurrent_multihead_seq(
        A(o_q), A(o_k), A(o_v), A(o_state), A(o_p), A(o_scratch),
        A(o_delta), A(o_zero),
        A(o_Al), A(o_a), A(o_b), A(o_dt), A(o_gexp), A(o_beta), A(o_gs),
        A(o_out), H, Dk, Dv);

    auto out       = at::empty({H, Dv}, q.options());
    auto state_out = at::empty({H, Dk, Dv}, q.options());
    rpu_launch_spm_copy_ddr_dma_immediate(A(o_out), out.data_ptr<c10::Half>(), H * Dv);
    rpu_launch_spm_copy_ddr_dma_immediate(A(o_state), state_out.data_ptr<c10::Half>(), H * Dk * Dv);

    SPM_ALLOC.reset_temporary();
    return {out, state_out};
}
