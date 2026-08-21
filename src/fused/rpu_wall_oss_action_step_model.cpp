// rpu_wall_oss_action_step_model.cpp — fused Wall-OSS-0.5 action-denoise step op.
// See rpu_wall_oss_action_step_model.h for the public contract.
//
// Inherits v3::CausalDecoderModel (plain Qwen2.5 decoder body) and adds:
//   - pre_layers_fn  : on-device action preprocessor (W_comb GEMM + B_step + SiLU + w3 GEMM)
//   - post_layers_fn : proj_back on the final-normed residual1
//   - step_forward   : per-step mutable DMA bases + base CausalDecoderModel::forward
#include "rpu_wall_oss_action_step_model.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"

#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>

namespace v3 {

WallOssActionStepModel::WallOssActionStepModel()  = default;
WallOssActionStepModel::~WallOssActionStepModel() = default;

// ─────────────────────────────────────────────────────────────────────────────
// set_action_weights
// ─────────────────────────────────────────────────────────────────────────────
void WallOssActionStepModel::set_action_weights(
    const at::Tensor& w_comb, const at::Tensor& w3, const at::Tensor& proj_back,
    const at::Tensor& w2_t, const at::Tensor& w2_a,
    int64_t action_dim, int64_t k_comb_pad, int64_t chunk_size) {
    TORCH_CHECK(num_layers() > 0,
                "set_action_weights must follow CausalDecoderModel::set_weights");
    TORCH_CHECK(action_dim > 0 && k_comb_pad > 0, "action_dim/k_comb_pad must be positive");
    TORCH_CHECK(chunk_size == 0 || (chunk_size >= 16 && chunk_size % 16 == 0),
                "action chunk_size must be 0 (auto) or a positive multiple of 16, got ",
                chunk_size);
    TORCH_CHECK(k_comb_pad % 16 == 0, "k_comb_pad must be a multiple of 16 (single-core col GEMM)");
    w_comb_ = w_comb;
    w3_ = w3;
    proj_back_ = proj_back;
    w2_t_ = w2_t;
    w2_a_ = w2_a;
    action_dim_ = action_dim;
    // proj_back is single-core col-swizzled (shape != [N,K]); derive the padded out-dim
    // from action_dim (round up to 16) rather than proj_back.size(0).
    action_dim_pad_ = ((action_dim + 15) / 16) * 16;     // 26 -> 32
    k_comb_pad_ = k_comb_pad;
    // In-graph Euler (x += dt·v_t) is an elementwise add of x_t_spm [cs,k_comb_pad] and
    // v_t_core0_spm [cs,action_dim_pad]; it requires equal padded widths (mirrors the
    // Python assert). Both are roundup16(action_dim)=32 for Wall-OSS.
    TORCH_CHECK(k_comb_pad_ == action_dim_pad_,
        "in-graph Euler requires k_comb_pad(", k_comb_pad_, ")==action_dim_pad(",
        action_dim_pad_, ")");
    configured_chunk_size_ = chunk_size;
    set_chunk_size_override(configured_chunk_size_);
    // The logical action horizon is known only at forward. It is deliberately
    // separate from the physical, v16-aligned planner chunk above.
    chunk_size_ = 0;
    emb_stage_ = at::Tensor{};
    invalidate_model_state();   // Must remain the last state-changing statement.
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers / static_config / dynamic_config
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BufferDecl> WallOssActionStepModel::declare_buffers(const LayoutContext& ctx) {
    auto d = CausalDecoderModel::declare_buffers(ctx);
    const int64_t cs = ctx.chunk_size, h = hidden_size();
    constexpr int DW = 2;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
    using SC = StorageClass;
    constexpr BufferScope ALL = BufferScope::LayerWide;
    // BufferDecl: {name, size, phase_start, phase_end, storage, per_layer, alias_of, scope}.
    // Pre-hook temps live at phase 0..0 — dead before layer 0 (residual1 is 1..8), so the
    // overlap-only allocator may alias them onto residual1's slot (safe).
    // EXCEPT x_t_spm: it holds x across the WHOLE step (phase 0..9) so the post-hook's
    // in-graph Euler (x += dt·v_t) can read it. Its overlapping lifetime forces a private
    // slot (no alias onto residual1) — ~2 KB/core, trivial vs the 8109 KB budget.
    d.push_back({"x_t_spm",       A(cs * k_comb_pad_ * DW), 0, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"ce_spm",        A(cs * h * DW),           0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"b_step_spm",    A(cs * h * DW),           0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"emb_core0_spm", A(cs * h * DW),           0, 0, SC::Temp, 0, nullptr, ALL});
    // On-device base = ae_dof·w2_a and ct = te·w2_t (phase 0..0, core 0; [1,h] each).
    // Only emitted when have_te_. ae_spm holds the ae_dof input before the w2_a GEMM.
    d.push_back({"ae_spm",        A(h * DW),                0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"te_spm",        A(h * DW),                0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"ct_spm",        A(h * DW),                0, 0, SC::Temp, 0, nullptr, ALL});
    // Post-hook output at phase 8..9 — must NOT alias residual1 (1..8), which the post-hook
    // reads (final-normed hidden). Coexists with lm_head_out (also 8..9) in a separate slot.
    d.push_back({"v_t_core0_spm", A(cs * action_dim_pad_ * DW), 8, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"velocity_blend_spm", A(cs * action_dim_pad_ * DW), 8, 9,
                 SC::Temp, 0, nullptr, ALL});
    return d;
}

ModelStaticConfig WallOssActionStepModel::static_config() {
    ModelStaticConfig cfg = CausalDecoderModel::static_config();
    cfg.pre_layers_fn  = reinterpret_cast<void (FusedModelBase::*)()>(
        &WallOssActionStepModel::emit_pre_layers_body);
    cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
        &WallOssActionStepModel::emit_post_layers_body);
    cfg.body_iterations = loop_mode_ ? num_steps_ : 1;   // In-graph denoise loop.
    return cfg;
}

ModelDynamicConfig WallOssActionStepModel::dynamic_config(const ChunkPlan& plan) {
    ModelDynamicConfig cfg = CausalDecoderModel::dynamic_config(plan);
    cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook bodies
// ─────────────────────────────────────────────────────────────────────────────
void WallOssActionStepModel::emit_pre_layers_body() {
    const int64_t h = hidden_size(), cs = chunk_size_;
    // Current iteration of the in-graph denoise loop.
    // Always 0 for the single-step op (loop_mode_ false) -> every branch below
    // degenerates to the original code (offset 0, x0 always loaded).
    const int64_t bit = ctx().body_iter;
    // Per-iteration bias src offsets (bytes; c10::Half = 2B). The mutable base is
    // pinned by the caller: step_forward pins single [hidden]/[H,hidden] tensors
    // (bit==0 -> offset 0); denoise_loop_forward pins the te_all/b_all arrays.
    const int64_t te_off_b   = bit * h * 2;        // te_all[bit]   (bias+te)
    const int64_t bias_off_b = bit * h * 2;        // b_all[bit]    (bias, no-te)
    const int64_t full_off_b = bit * cs * h * 2;   // b_all[bit]    (full [H,hidden])

    // [A1] x_t DDR -> x_t_spm (MUTABLE; x_t already zero-padded to k_comb_pad_).
    // Loaded ONLY on iteration 0; iters 1..N-1 read the in-SPM Euler result the
    // prior post-hook left in x_t_spm (persistent [0,9] slot).
    if (bit == 0) {
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &x_t_src_base_, /*src_offset_bytes=*/0, /*num_elements=*/cs * k_comb_pad_,
            addr(0, "x_t_spm"), /*num_cores=*/NUM_CORES);
    }
    // RTC hard pin at every Euler-step entry. rtc_delay_ is part of the Python
    // GraphSignature, while the prefix address/value is refreshed through the
    // mutable base on every request. The stock rtc_delay_==0 path emits no node.
    if (rtc_delay_ > 0) {
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &rtc_prefix_src_base_, /*src_offset_bytes=*/0,
            /*num_elements=*/rtc_delay_ * k_comb_pad_, addr(0, "x_t_spm"),
            /*num_cores=*/1);
    }

    if (b_step_is_bias_) {
        // Row-constant [hidden] bias -> b_step_spm (core 0), broadcast per-output-feature over
        // the cs rows by the W_comb GEMM. Two modes:
        //   have_te_: b_step carries ae_dof (= dof·w1_dof); base = ae_dof·w2_a and
        //             ct = te·w2_t are both computed in-SPM here, b_step_spm = base + ct.
        //   else:     b_step is the full host-precomputed B_step, used directly.
        if (have_te_ == 1) {
            // [A1a] ae_dof [hidden] DDR -> ae_spm; base = ae_dof @ w2_a.T -> b_step_spm.
            // This is an exact single-core M=1 projection, so use the production GEMV path.
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &b_step_src_base_, /*src_offset_bytes=*/0, /*num_elements=*/h,
                addr(0, "ae_spm"), /*num_cores=*/1);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "ae_spm"), w2_a_, addr(0, "b_step_spm"),
                /*M=*/1, /*N=*/h, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
                /*bias_spm_addr=*/0, /*force_gemm=*/false);
            // [A1b] te_step [hidden] DDR -> te_spm; ct = te @ w2_t.T -> ct_spm (M=1 GEMV).
            // te_off_b selects te_all[bit] in the loop (0 for single-step).
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &te_src_base_, /*src_offset_bytes=*/te_off_b, /*num_elements=*/h,
                addr(0, "te_spm"), /*num_cores=*/1);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "te_spm"), w2_t_, addr(0, "ct_spm"),
                /*M=*/1, /*N=*/h, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
                /*bias_spm_addr=*/0, /*force_gemm=*/false);
            // [A1c] b_step_spm = base + ct  (in-SPM, core 0).
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "b_step_spm"), addr(0, "ct_spm"), addr(0, "b_step_spm"),
                h, ValuOpType::ADD, c10::Half(1.0), /*num_cores=*/1);
        } else {
            // bias, no-te: b_all[bit] is this iteration's [hidden] bias (bias_off_b=0 single-step).
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &b_step_src_base_, /*src_offset_bytes=*/bias_off_b, /*num_elements=*/h,
                addr(0, "b_step_spm"), /*num_cores=*/1);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "x_t_spm"), w_comb_, addr(0, "ce_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/k_comb_pad_,
            /*partition=*/1, /*num_cores=*/1, /*bias_spm_addr=*/addr(0, "b_step_spm"));
    } else {
        // [H,hidden] B_step: GEMM (no bias) then a same-shape add at num_cores=1.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "x_t_spm"), w_comb_, addr(0, "ce_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/k_comb_pad_,
            /*partition=*/1, /*num_cores=*/1, /*bias_spm_addr=*/0);
        // full [H,hidden] B_step: b_all[bit] is this iteration's [cs,hidden] (full_off_b=0 single-step).
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &b_step_src_base_, /*src_offset_bytes=*/full_off_b, /*num_elements=*/cs * h,
            addr(0, "b_step_spm"), /*num_cores=*/1);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "ce_spm"), addr(0, "b_step_spm"), addr(0, "ce_spm"),
            cs * h, ValuOpType::ADD, c10::Half(1.0), /*num_cores=*/1);
    }

    // [A2] SiLU(ce) in place, core 0. Pass is_gelu explicitly so num_cores=1
    // cannot bind to the is_gelu argument.
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "ce_spm"), addr(0, "ce_spm"), cs * h, ValuOpType::SILU,
        /*is_gelu=*/false, /*num_cores=*/1);

    // [A3] emb = silu(ce) @ w3.T, single-core.
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "ce_spm"), w3_, addr(0, "emb_core0_spm"),
        /*M=*/cs, /*N=*/h, /*K=*/h, /*partition=*/1, /*num_cores=*/1, /*bias_spm_addr=*/0);

    // [A4] emb_core0_spm -> emb_stage_ DDR (FIXED; stable data_ptr). Layer 0's
    // emit_layer_input_dma reads emb_stage_ via FMB hidden_in_src_base_.
    rpu_launch_spm_copy_ddr_dma(
        addr(0, "emb_core0_spm"), emb_stage_.data_ptr<c10::Half>(), cs * h);
}

void WallOssActionStepModel::emit_post_layers_body() {
    const int64_t h = hidden_size(), cs = chunk_size_, np = action_dim_pad_;
    const int64_t bit = ctx().body_iter;   // 0 for single-step execution.
    // [Z1] v_t = residual1(final-normed hidden) @ proj_back.T, single-core. proj_back is
    // zero-padded to np rows; Python slices [:, :, :action_dim]. Do NOT re-normalize:
    // the last layer already applied the final RMSNorm in-place to residual1.
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), proj_back_, addr(0, "v_t_core0_spm"),
        /*M=*/cs, /*N=*/np, /*K=*/h, /*partition=*/1, /*num_cores=*/1, /*bias_spm_addr=*/0);
    // Reference semantics for inactive action dimensions:
    //   v = model_v * dof_mask + (padding_action - x_initial) * (1 - dof_mask).
    // The two request-dependent tensors use mutable bases so same-signature replays
    // consume the current noise/mask values. Reuse one small SPM scratch sequentially.
    if (velocity_blend_) {
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &velocity_mask_src_base_, /*src_offset_bytes=*/0,
            /*num_elements=*/cs * np, addr(0, "velocity_blend_spm"),
            /*num_cores=*/1);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "v_t_core0_spm"), addr(0, "velocity_blend_spm"),
            addr(0, "v_t_core0_spm"), cs * np, ValuOpType::MUL,
            c10::Half(1.0f), /*num_cores=*/1);
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &velocity_override_src_base_, /*src_offset_bytes=*/0,
            /*num_elements=*/cs * np, addr(0, "velocity_blend_spm"),
            /*num_cores=*/1);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "v_t_core0_spm"), addr(0, "velocity_blend_spm"),
            addr(0, "v_t_core0_spm"), cs * np, ValuOpType::ADD,
            c10::Half(1.0f), /*num_cores=*/1);
    }
    // The RTC profile runs its first Euler update before re-pinning, then returns zero
    // velocity for committed rows on subsequent ODE steps. Match that observable
    // trajectory exactly; the step-exit overwrite below hard-pins x after every step.
    if (rtc_delay_ > 0 && bit > 0) {
        rpu_launch_fill_spm_kernel(
            addr(0, "v_t_core0_spm"), rtc_delay_ * np,
            c10::Half(0.0f), /*num_cores=*/1);
    }
    // [Z2] v_t_core0_spm -> DDR (MUTABLE). loop mode writes the per-iteration trajectory
    // slot v_traj[bit]; single-step writes v_t_buf at offset 0.
    if (loop_mode_) {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "v_t_core0_spm"), &v_traj_dst_base_,
            /*dst_offset_bytes=*/bit * cs * np * 2, cs * np);
    } else {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "v_t_core0_spm"), &v_t_dst_base_, /*dst_offset_bytes=*/0, cs * np);
    }

    // [Z3] In-graph Euler: x_t_spm = x_t_spm + dt·v_t (in-place SPM, core 0). x_t_spm holds
    // x (phase 0..9); v_t_core0_spm holds v_t. k_comb_pad_==action_dim_pad_ (checked in
    // set_action_weights) so the two [cs,np] buffers add elementwise; padding cols stay 0
    // (x pad 0 + dt·0). dt_ baked at BUILD — constant across the 10 REPLAY steps.
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "x_t_spm"), addr(0, "v_t_core0_spm"), addr(0, "x_t_spm"),
        cs * np, ValuOpType::ADD, dt_, /*num_cores=*/1);
    // Explicit step-exit/final pin. This restores the exact prefix bytes after the
    // first update and protects against drift on the later zero-velocity steps.
    if (rtc_delay_ > 0) {
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &rtc_prefix_src_base_, /*src_offset_bytes=*/0,
            /*num_elements=*/rtc_delay_ * k_comb_pad_, addr(0, "x_t_spm"),
            /*num_cores=*/1);
    }
    // [Z4] x_t_spm (= x_new) -> DDR (MUTABLE). loop mode writes x_traj[bit] (final x =
    // x_traj[-1]); single-step writes x_out_buf at offset 0.
    if (loop_mode_) {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "x_t_spm"), &x_traj_dst_base_,
            /*dst_offset_bytes=*/bit * cs * k_comb_pad_ * 2, cs * k_comb_pad_);
    } else {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "x_t_spm"), &x_out_dst_base_, /*dst_offset_bytes=*/0, cs * k_comb_pad_);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// step_forward
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor WallOssActionStepModel::step_forward(
    const at::Tensor& x_t_rpu, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches, const at::Tensor& b_step,
    const std::optional<at::Tensor>& te_step,
    const std::optional<at::Tensor>& attention_mask,
    const at::Tensor& position_ids,                 // REQUIRED (mRoPE always on for wall_oss)
    at::Tensor& v_t_buf, at::Tensor& x_out_buf, double dt, int64_t prefix_len,
    const std::optional<at::Tensor>& rope_cos_il,    // partial_mrope interleaved cos/sin (or nullopt)
    const std::optional<at::Tensor>& rope_sin_il) {
    TORCH_CHECK(action_dim_pad_ > 0, "step_forward before set_action_weights");
    rtc_delay_ = 0;                  // stepwise op is the stock non-RTC contract
    rtc_prefix_ref_ = at::Tensor{};  // never leak a prior loop request's keepalive
    velocity_blend_ = false;         // stepwise masked-DOF requests are rejected in Python
    velocity_mask_ref_ = velocity_override_ref_ = at::Tensor{};
    // chunk_size (= horizon H) is runtime-determined by x_t. (Re)allocate the stable
    // emb_stage_ staging + force a rebuild when it changes (first step, or a new horizon).
    const int64_t H = x_t_rpu.size(1);
    TORCH_CHECK(H > 0, "x_t_rpu seq_len (horizon) must be > 0");
    const int64_t required_chunk = ((H + 15) / 16) * 16;
    TORCH_CHECK(configured_chunk_size_ == 0
                    || configured_chunk_size_ == required_chunk,
                "Wall-OSS action exact chunk_size must equal ceil16(horizon): configured=",
                configured_chunk_size_, " horizon=", H, " required=", required_chunk);
    if (loop_mode_ || chunk_size_ != H) {
        loop_mode_ = false;   // leaving the unroll path -> single-step hooks (no trajectory DMAs)
        chunk_size_ = H;
        emb_stage_ = at::empty({1, H, hidden_size()},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        dt_pinned_ = false;   // graph rebuilds -> dt re-baked at the next BUILD
        have_te_ = -1;        // graph rebuilds -> ct path re-decided at the next BUILD
        b_step_mode_ = -1;    // re-decide bias mode (symmetry with denoise_loop_forward)
        invalidate_model_state();
    }
    TORCH_CHECK(x_t_rpu.dim() == 3 && x_t_rpu.size(0) == 1 && x_t_rpu.size(1) == chunk_size_
        && x_t_rpu.size(2) == k_comb_pad_ && x_t_rpu.scalar_type() == at::kHalf
        && x_t_rpu.is_contiguous() && x_t_rpu.device().type() == at::kPrivateUse1,
        "x_t_rpu must be [1,", chunk_size_, ",", k_comb_pad_, "] fp16 contig RPU");
    TORCH_CHECK(v_t_buf.dim() == 3 && v_t_buf.size(0) == 1 && v_t_buf.size(1) == chunk_size_
        && v_t_buf.size(2) == action_dim_pad_ && v_t_buf.scalar_type() == at::kHalf
        && v_t_buf.is_contiguous() && v_t_buf.device().type() == at::kPrivateUse1,
        "v_t_buf must be [1,", chunk_size_, ",", action_dim_pad_, "] fp16 contig RPU");
    TORCH_CHECK(x_out_buf.dim() == 3 && x_out_buf.size(0) == 1 && x_out_buf.size(1) == chunk_size_
        && x_out_buf.size(2) == k_comb_pad_ && x_out_buf.scalar_type() == at::kHalf
        && x_out_buf.is_contiguous() && x_out_buf.device().type() == at::kPrivateUse1,
        "x_out_buf must be [1,", chunk_size_, ",", k_comb_pad_, "] fp16 contig RPU");
    // dt is baked into the in-graph Euler eltwise at BUILD (step 0) and reused on every
    // REPLAY, so it MUST be identical across steps (it is: dt = 1/num_steps, constant).
    TORCH_CHECK(dt > 0.0, "dt must be > 0");
    const c10::Half dt_half = c10::Half(static_cast<float>(dt));
    if (!dt_pinned_) { dt_ = dt_half; dt_pinned_ = true; }
    else TORCH_CHECK(dt_ == dt_half,
        "dt changed across steps (baked into the graph on the first step)");
    TORCH_CHECK(b_step.scalar_type() == at::kHalf && b_step.is_contiguous()
        && b_step.device().type() == at::kPrivateUse1, "b_step must be fp16 contig RPU");

    // b_step mode: [hidden] row-constant bias vs [H,hidden] add. Inferred per call and
    // asserted stable across steps (the per-step graph is built on the first step).
    const bool is_bias = (b_step.dim() == 1);
    TORCH_CHECK(is_bias ? (b_step.size(0) == hidden_size())
                        : (b_step.dim() == 2 && b_step.size(0) == chunk_size_
                           && b_step.size(1) == hidden_size()),
        "b_step must be [hidden] (row-constant) or [H,hidden]");
    if (b_step_mode_ < 0) b_step_mode_ = is_bias ? 1 : 0;
    else TORCH_CHECK((b_step_mode_ == 1) == is_bias,
        "b_step mode changed across steps (graph built for the first step's shape)");
    b_step_is_bias_ = is_bias;

    // te_step: optional per-step time emb -> on-device ct = te·w2_t (bias mode only). Whether
    // the ct path exists is baked into the graph at BUILD, so its presence must be stable.
    const bool want_te = te_step.has_value();
    if (want_te) {
        TORCH_CHECK(is_bias, "te_step requires the row-constant (bias) b_step mode");
        TORCH_CHECK(w2_t_.defined(), "te_step given but w2_t not set (set_action_weights)");
        const at::Tensor& te = te_step.value();
        TORCH_CHECK(te.dim() == 1 && te.size(0) == hidden_size()
            && te.scalar_type() == at::kHalf && te.is_contiguous()
            && te.device().type() == at::kPrivateUse1,
            "te_step must be [hidden] fp16 contig RPU");
    }
    if (have_te_ < 0) have_te_ = want_te ? 1 : 0;
    else TORCH_CHECK((have_te_ == 1) == want_te,
        "te_step presence changed across steps (graph built for the first step)");

    // KV headroom: K layout max_seq = size(1)*size(5).
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + chunk_size_ <= kc.size(1) * kc.size(5),
            "prefix_len(", prefix_len, ")+H(", chunk_size_, ") exceeds k_cache capacity ",
            kc.size(1) * kc.size(5));
    }

    // Pin the per-step mutable DMA bases (caller owns the flush — wrappers do not flush).
    // Every base pinned below is dereferenced when the graph EXECUTES, i.e. in
    // RpuKernelGraph::end() when the caller's `with cache.capture(sig):` exits —
    // not when this function returns. Keep an owning ref on each; the two write
    // destinations matter most, since a re-issued VA means
    // the graph writes over whatever now owns that block.
    x_t_ref_ = x_t_rpu;     // keepalive across the synchronous forward
    b_step_ref_ = b_step;   // keepalive across the synchronous forward
    v_t_ref_ = v_t_buf;     // keepalive — deferred WRITE destination
    x_out_ref_ = x_out_buf; // keepalive — deferred WRITE destination
    // rpu_ddr_flush_force(x_t_rpu.data_ptr<c10::Half>());
    // rpu_ddr_flush_force(b_step.data_ptr<c10::Half>());
    // rpu_ddr_flush_force(v_t_buf.data_ptr<c10::Half>());
    // rpu_ddr_flush_force(x_out_buf.data_ptr<c10::Half>());
    x_t_src_base_    = ::rhino_lkn::RpuGetDevAddr(x_t_rpu.data_ptr());
    b_step_src_base_ = ::rhino_lkn::RpuGetDevAddr(b_step.data_ptr());
    v_t_dst_base_    = ::rhino_lkn::RpuGetDevAddr(v_t_buf.data_ptr());
    x_out_dst_base_  = ::rhino_lkn::RpuGetDevAddr(x_out_buf.data_ptr());
    if (have_te_ == 1) {
        te_ref_ = te_step.value();   // keepalive across the synchronous forward
        // rpu_ddr_flush_force(te_ref_.data_ptr<c10::Half>());
        te_src_base_ = ::rhino_lkn::RpuGetDevAddr(te_ref_.data_ptr());
    }

    // The explicit 2D-mask path requires one physical chunk. Auto resolves to
    // ceil16(H); an exact public request binds that same capacity per handle.
    set_chunk_size_override(configured_chunk_size_);

    // Inherited mRoPE/position_ids keepalive + mask-prep + run_all_layers; the pre/post
    // hooks fire via virtual config dispatch. Output v_t was written to v_t_buf by the
    // post-hook. Return the same final-normed hidden DDR tensor as the base decoder so
    // the opt-in FP32-tail diagnostic can recompute proj_back + Euler on the host while
    // leaving the fused preprocessor/decoder path unchanged. Production callers ignore it.
    return CausalDecoderModel::forward(
        emb_stage_, k_caches, v_caches, attention_mask,
        /*position=*/prefix_len, /*is_causal=*/false,
        std::optional<at::Tensor>(position_ids), /*deepstack=*/std::nullopt,
        rope_cos_il, rope_sin_il);
}

// ─────────────────────────────────────────────────────────────────────────────
// denoise_loop_forward — in-graph unroll
// ─────────────────────────────────────────────────────────────────────────────
void WallOssActionStepModel::denoise_loop_forward(
    const at::Tensor& x0_rpu, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches, const at::Tensor& b_all,
    const std::optional<at::Tensor>& te_all,
    const std::optional<at::Tensor>& attention_mask,
    const at::Tensor& position_ids, at::Tensor& x_traj, at::Tensor& v_traj,
    double dt, int64_t prefix_len, int64_t num_steps,
    const std::optional<at::Tensor>& rope_cos_il,    // partial_mrope interleaved cos/sin (or nullopt)
    const std::optional<at::Tensor>& rope_sin_il,
    const std::optional<at::Tensor>& rtc_prefix,
    int64_t rtc_delay,
    const std::optional<at::Tensor>& velocity_mask,
    const std::optional<at::Tensor>& velocity_override) {
    TORCH_CHECK(action_dim_pad_ > 0, "denoise_loop_forward before set_action_weights");
    TORCH_CHECK(num_steps > 0, "num_steps must be > 0");
    const int64_t H = x0_rpu.size(1);
    TORCH_CHECK(H > 0, "x0_rpu seq_len (horizon) must be > 0");
    const int64_t required_chunk = ((H + 15) / 16) * 16;
    TORCH_CHECK(configured_chunk_size_ == 0
                    || configured_chunk_size_ == required_chunk,
                "Wall-OSS action exact chunk_size must equal ceil16(horizon): configured=",
                configured_chunk_size_, " horizon=", H, " required=", required_chunk);

    // (Re)build when entering loop mode, or when loop cardinality / horizon changes.
    // Mirrors step_forward's chunk_size_!=H guard; also resets the baked-mode flags.
    if (!loop_mode_ || num_steps_ != num_steps || chunk_size_ != H) {
        loop_mode_  = true;
        num_steps_  = num_steps;
        chunk_size_ = H;
        emb_stage_  = at::empty({1, H, hidden_size()},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        dt_pinned_ = false; have_te_ = -1; b_step_mode_ = -1;
        invalidate_model_state();
    }

    // Node-count guard: prepare_segment_queue() does not surface batch overflow.
    // The queue cap is 32768 pending items. Budget about 1100 nodes per body and
    // fail below the cap with margin; per-forward setup is shared once.
    TORCH_CHECK(num_steps * 1100 < 32000,
        "denoise unroll (", num_steps, " bodies × ~1035 nodes) would approach the 32768 "
        "batch-item cap; reduce num_steps");

    TORCH_CHECK(x0_rpu.dim() == 3 && x0_rpu.size(0) == 1 && x0_rpu.size(1) == chunk_size_
        && x0_rpu.size(2) == k_comb_pad_ && x0_rpu.scalar_type() == at::kHalf
        && x0_rpu.is_contiguous() && x0_rpu.device().type() == at::kPrivateUse1,
        "x0_rpu must be [1,", chunk_size_, ",", k_comb_pad_, "] fp16 contig RPU");
    TORCH_CHECK(x_traj.dim() == 3 && x_traj.size(0) == num_steps && x_traj.size(1) == chunk_size_
        && x_traj.size(2) == k_comb_pad_ && x_traj.scalar_type() == at::kHalf
        && x_traj.is_contiguous() && x_traj.device().type() == at::kPrivateUse1,
        "x_traj must be [", num_steps, ",", chunk_size_, ",", k_comb_pad_, "] fp16 contig RPU");
    TORCH_CHECK(v_traj.dim() == 3 && v_traj.size(0) == num_steps && v_traj.size(1) == chunk_size_
        && v_traj.size(2) == action_dim_pad_ && v_traj.scalar_type() == at::kHalf
        && v_traj.is_contiguous() && v_traj.device().type() == at::kPrivateUse1,
        "v_traj must be [", num_steps, ",", chunk_size_, ",", action_dim_pad_, "] fp16 contig RPU");

    TORCH_CHECK(rtc_delay >= 0 && rtc_delay <= chunk_size_,
        "rtc_delay must be in [0,", chunk_size_, "], got ", rtc_delay);
    TORCH_CHECK((rtc_delay > 0) == rtc_prefix.has_value(),
        "rtc_prefix must be present iff rtc_delay > 0");
    if (rtc_delay > 0) {
        TORCH_CHECK(rtc_delay <= 10,
            "rtc_delay exceeds the v728 training envelope (max 10): ", rtc_delay);
        const at::Tensor& prefix = rtc_prefix.value();
        TORCH_CHECK(prefix.dim() == 3 && prefix.size(0) == 1
            && prefix.size(1) == chunk_size_ && prefix.size(2) == k_comb_pad_
            && prefix.scalar_type() == at::kHalf && prefix.is_contiguous()
            && prefix.device().type() == at::kPrivateUse1,
            "rtc_prefix must be [1,", chunk_size_, ",", k_comb_pad_,
            "] fp16 contig RPU");
        rtc_prefix_ref_ = prefix;
        rtc_prefix_src_base_ = ::rhino_lkn::RpuGetDevAddr(prefix.data_ptr());
        rpu_ddr_flush_force_sized(prefix.data_ptr(), prefix.nbytes());
    } else {
        rtc_prefix_ref_ = at::Tensor{};
        rtc_prefix_src_base_ = 0;
    }
    rtc_delay_ = rtc_delay;

    TORCH_CHECK(velocity_mask.has_value() == velocity_override.has_value(),
        "velocity_mask and velocity_override must be present together");
    velocity_blend_ = velocity_mask.has_value();
    if (velocity_blend_) {
        const at::Tensor& mask = velocity_mask.value();
        const at::Tensor& override_tensor = velocity_override.value();
        auto valid_blend_tensor = [&](const at::Tensor& t) {
            return t.dim() == 3 && t.size(0) == 1
                && t.size(1) == chunk_size_ && t.size(2) == action_dim_pad_
                && t.scalar_type() == at::kHalf && t.is_contiguous()
                && t.device().type() == at::kPrivateUse1;
        };
        TORCH_CHECK(valid_blend_tensor(mask),
            "velocity_mask must be [1,", chunk_size_, ",", action_dim_pad_,
            "] fp16 contig RPU");
        TORCH_CHECK(valid_blend_tensor(override_tensor),
            "velocity_override must be [1,", chunk_size_, ",", action_dim_pad_,
            "] fp16 contig RPU");
        velocity_mask_ref_ = mask;
        velocity_override_ref_ = override_tensor;
        velocity_mask_src_base_ = ::rhino_lkn::RpuGetDevAddr(mask.data_ptr());
        velocity_override_src_base_ = ::rhino_lkn::RpuGetDevAddr(override_tensor.data_ptr());
        rpu_ddr_flush_force_sized(mask.data_ptr(), mask.nbytes());
        rpu_ddr_flush_force_sized(override_tensor.data_ptr(), override_tensor.nbytes());
    } else {
        velocity_mask_ref_ = at::Tensor{};
        velocity_override_ref_ = at::Tensor{};
        velocity_mask_src_base_ = 0;
        velocity_override_src_base_ = 0;
    }

    // dt baked into the in-graph Euler at BUILD; identical across the N unrolled iterations.
    TORCH_CHECK(dt > 0.0, "dt must be > 0");
    const c10::Half dt_half = c10::Half(static_cast<float>(dt));
    // dt is baked per Python GraphSignature. Assign it for the active entry so
    // the same handle may own both stock (1/N) and v728 RTC (0.999/N) graphs.
    dt_ = dt_half;
    dt_pinned_ = true;

    // b_all / te_all mode (mirrors the Python adapter):
    //   te_all present → row-constant bias path: b_all = ae_dof [hidden], te_all =
    //                    [num_steps, hidden]; on-device base + ct.
    //   te_all absent  → full path: b_all = [num_steps, H, hidden] (per-step add).
    const bool want_te = te_all.has_value();
    const bool is_bias = want_te;   // non-te loop is always the full [steps,H,hidden] path
    TORCH_CHECK(b_all.scalar_type() == at::kHalf && b_all.is_contiguous()
        && b_all.device().type() == at::kPrivateUse1, "b_all must be fp16 contig RPU");
    if (want_te) {
        TORCH_CHECK(b_all.dim() == 1 && b_all.size(0) == hidden_size(),
            "row-constant b_all (ae_dof) must be [hidden]");
        const at::Tensor& te = te_all.value();
        TORCH_CHECK(te.dim() == 2 && te.size(0) == num_steps && te.size(1) == hidden_size()
            && te.scalar_type() == at::kHalf && te.is_contiguous()
            && te.device().type() == at::kPrivateUse1,
            "te_all must be [", num_steps, ", hidden] fp16 contig RPU");
        TORCH_CHECK(w2_t_.defined(), "te_all given but w2_t not set (set_action_weights)");
    } else {
        TORCH_CHECK(b_all.dim() == 3 && b_all.size(0) == num_steps && b_all.size(1) == chunk_size_
            && b_all.size(2) == hidden_size(),
            "full b_all must be [", num_steps, ",", chunk_size_, ", hidden]");
    }
    // Bias mode is also baked per GraphSignature: stock row-constant graphs use
    // ae_dof+te, RTC graphs use full per-token B. Set the active entry's mode;
    // GraphCache owns the distinct recorded op streams.
    b_step_mode_ = is_bias ? 1 : 0;
    have_te_ = want_te ? 1 : 0;
    b_step_is_bias_ = is_bias;

    // KV headroom (same check as step_forward).
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + chunk_size_ <= kc.size(1) * kc.size(5),
            "prefix_len(", prefix_len, ")+H(", chunk_size_, ") exceeds k_cache capacity ",
            kc.size(1) * kc.size(5));
    }

    // Pin the per-predict mutable DMA bases (caller owns any flush; wrappers do not flush).
    // x0 -> x_t_spm (iter 0 only); b_all/te_all read at baked per-iteration offsets by the
    // pre-hook; x_traj/v_traj written at baked per-iteration offsets by the post-hook.
    b_all_ref_ = b_all; x_traj_ref_ = x_traj; v_traj_ref_ = v_traj;  // keepalive
    x_t_ref_ = x0_rpu;  // keepalive — x_t_src_base_ points at x0 here
    x_t_src_base_    = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
    b_step_src_base_ = ::rhino_lkn::RpuGetDevAddr(b_all.data_ptr());
    x_traj_dst_base_ = ::rhino_lkn::RpuGetDevAddr(x_traj.data_ptr());
    v_traj_dst_base_ = ::rhino_lkn::RpuGetDevAddr(v_traj.data_ptr());
    if (have_te_ == 1) {
        te_all_ref_  = te_all.value();   // keepalive
        te_src_base_ = ::rhino_lkn::RpuGetDevAddr(te_all_ref_.data_ptr());
    }

    // AUTO chunk for the per-handle 2D-mask single-chunk forward.
    set_chunk_size_override(configured_chunk_size_);

    // ONE forward — run_all_layers loops body_iterations(=num_steps_) internally via the
    // base change; the pre/post hooks fire per iteration with ctx().body_iter advancing.
    (void) CausalDecoderModel::forward(
        emb_stage_, k_caches, v_caches, attention_mask,
        /*position=*/prefix_len, /*is_causal=*/false,
        std::optional<at::Tensor>(position_ids), /*deepstack=*/std::nullopt,
        rope_cos_il, rope_sin_il);
}

}  // namespace v3

// =============================================================================
// Instance registry — ModelHandleRegistry<v3::WallOssActionStepModel>
// =============================================================================
using WallOssActionStepRegistry = ModelHandleRegistry<v3::WallOssActionStepModel>;

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================
int64_t rpu_wall_oss_action_step_create() {
    return WallOssActionStepRegistry::create();
}

void rpu_wall_oss_action_step_destroy(int64_t handle) {
    WallOssActionStepRegistry::destroy(handle, "rpu_wall_oss_action_step_destroy");
}

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
    const std::optional<at::Tensor>& q_tensor_scales,
    const std::optional<at::Tensor>& k_tensor_scales,
    const std::optional<at::Tensor>& v_tensor_scales,
    const std::optional<at::Tensor>& o_tensor_scales,
    const std::optional<at::Tensor>& gate_tensor_scales,
    const std::optional<at::Tensor>& up_tensor_scales,
    const std::optional<at::Tensor>& down_tensor_scales)
{
    auto* m = WallOssActionStepRegistry::get(handle, "rpu_wall_oss_action_step_set_weights");
    // Decoder weights via the inherited CausalDecoderModel::set_weights. Qwen2.5 has no
    // q/k norm (empty lists); deepstack_lang_layers empty; scales empty => fp16 path,
    // non-empty => W8A16. QKV bias present (Wall-OSS path).
    m->CausalDecoderModel::set_weights(
        q_w, k_w, v_w, o_w,
        /*q_norm=*/{}, /*k_norm=*/{},
        input_norm, post_norm,
        gate_w, up_w, down_w,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        eps, use_silu,
        mrope_section, /*deepstack_lang_layers=*/{},
        q_bias, k_bias, v_bias,
        q_w_scale, k_w_scale, v_w_scale, o_w_scale,
        gate_scale, up_scale, down_scale,
        q_tensor_scales.value_or(at::Tensor{}),
        k_tensor_scales.value_or(at::Tensor{}),
        v_tensor_scales.value_or(at::Tensor{}),
        o_tensor_scales.value_or(at::Tensor{}),
        gate_tensor_scales.value_or(at::Tensor{}),
        up_tensor_scales.value_or(at::Tensor{}),
        down_tensor_scales.value_or(at::Tensor{}));
    // This fused subsystem inherits CausalDecoderModel, including the
    // deny-by-default certified-envelope gate too. Its sequence length is NOT
    // caller-controlled the way a text prefill is -- it is structurally fixed by
    // the subsystem (action horizon).  Auto keeps the existing zero envelope;
    // an exact public request certifies only that cold-bound ceil16(horizon).
    // Without carrying the exact value here, chunk_within_envelope() rejects the
    // override before the action model can apply its stricter one-chunk check.
    // Declared here because the bound is intrinsic to the subsystem, not an
    // adapter policy.
    m->set_chunk_envelope(/*max_kv_len=*/8192, /*chunk=*/chunk_size);
    m->set_action_weights(w_comb, w3, proj_back, w2_t, w2_a, action_dim, k_comb_pad, chunk_size);
}

at::Tensor rpu_wall_oss_action_step_forward(
    int64_t handle, const at::Tensor& x_t_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& b_step, const std::optional<at::Tensor>& te_step,
    const std::optional<at::Tensor>& attention_mask,
    const at::Tensor& position_ids,
    at::Tensor v_t_buf, at::Tensor x_out_buf, double dt, int64_t prefix_len,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il)
{
    return WallOssActionStepRegistry::get(handle, "rpu_wall_oss_action_step_forward")
        ->step_forward(x_t_rpu, k_caches, v_caches, b_step, te_step, attention_mask,
                       position_ids, v_t_buf, x_out_buf, dt, prefix_len,
                       rope_cos_il, rope_sin_il);
}

void rpu_wall_oss_action_denoise_loop_forward(
    int64_t handle, const at::Tensor& x0_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& b_all, const std::optional<at::Tensor>& te_all,
    const std::optional<at::Tensor>& attention_mask, const at::Tensor& position_ids,
    at::Tensor x_traj, at::Tensor v_traj, double dt, int64_t prefix_len, int64_t num_steps,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il,
    const std::optional<at::Tensor>& rtc_prefix,
    int64_t rtc_delay,
    const std::optional<at::Tensor>& velocity_mask,
    const std::optional<at::Tensor>& velocity_override)
{
    WallOssActionStepRegistry::get(handle, "rpu_wall_oss_action_denoise_loop_forward")
        ->denoise_loop_forward(x0_rpu, k_caches, v_caches, b_all, te_all, attention_mask,
                               position_ids, x_traj, v_traj, dt, prefix_len, num_steps,
                               rope_cos_il, rope_sin_il, rtc_prefix, rtc_delay,
                               velocity_mask, velocity_override);
}
