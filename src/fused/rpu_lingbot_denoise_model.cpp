// rpu_lingbot_denoise_model.cpp — fused LingBot-VLA action-expert denoise unroll op.
// See rpu_lingbot_denoise_model.h for the model contract.
//
// Inherits v3::CausalDecoderModel (plain Qwen2.5 768-d decoder + AdaRMS FiLM) and adds:
//   - pre_layers_fn  : fused suffix encoder (W_comp GEMM + per-step time bias + SiLU +
//                      action_time_mlp_out GEMM) → emb_stage_ rows 1:51 (row 0 = state).
//   - post_layers_fn : out_proj on the final-normed residual1 + in-graph Euler (x += dt·v).
//   - unroll_forward : pins the per-call mutable bases + sets emb_stage_ row 0, then runs
//                      ONE CausalDecoderModel::forward with body_iterations = num_steps.
#include "rpu_lingbot_denoise_model.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"

#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

namespace v3 {

// FIXED_KERNEL_BASIS: the remaining launchers in this owner are fixed by
// model math or BufferDecl dataflow, not selectable execution policy: immutable
// time/mo/op bias broadcasts, SiLU, the stable emb_stage relay, and Euler ADD.
// Mutable endpoints and every Linear/KV/RoPE/attention/reduce choice are carried
// by the COMPLETE descriptor below.

LingbotDenoiseModel::LingbotDenoiseModel() {
    explicit_mask_residency_requested_ = true;
}
LingbotDenoiseModel::~LingbotDenoiseModel() = default;

namespace {

constexpr int64_t LINGBOT_DENOISE_PRE_X_DMA_SITE = 2135667578995030733LL;
constexpr int64_t LINGBOT_DENOISE_PRE_WC_LINEAR_SITE = 5445576830473452253LL;
constexpr int64_t LINGBOT_DENOISE_PRE_MO_LINEAR_SITE = 7499225056090777607LL;
constexpr int64_t LINGBOT_DENOISE_POST_OP_LINEAR_SITE = 7989127291028317735LL;
constexpr int64_t LINGBOT_DENOISE_POST_TRAJECTORY_DMA_SITE = 232719282840744102LL;

enum class LingbotDenoiseMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM = 1,
    SPM_COPY_TO_DDR = 2,
};

}  // namespace

void LingbotDenoiseModel::set_execution_chunk_size(int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0 || chunk_size == 64,
                "LingBot denoise chunk_size must be 0 (auto) or 64 for the "
                "fixed 51-row suffix, got ", chunk_size);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "LingBot denoise chunk_size must be set before the first forward");
    set_control_chunk_size_override(chunk_size, "LingbotDenoiseModel::set_execution_chunk_size");
}

std::vector<int64_t> LingbotDenoiseModel::resolve_action_stage_domain(
    int64_t execution_len, int64_t prefix_len, int64_t mask_kv_len) {
    TORCH_CHECK(
        execution_len == chunk_size_ && mask_kv_len == prefix_len + execution_len,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: LingBot-V1 denoise requires "
        "its complete fixed action suffix; execution=", execution_len,
        ", configured=", chunk_size_, ", mask_kv_len=", mask_kv_len,
        ", required=", prefix_len + execution_len);
    return CausalDecoderModel::resolve_prefill_stage_domain(
        execution_len, prefix_len, /*is_causal=*/false, mask_kv_len,
        /*rope_mode=*/0,
        static_cast<int64_t>(FmbGraphLifecycle::COMPOSITE_CHILD));
}

// ─────────────────────────────────────────────────────────────────────────────
// set_action_weights
// ─────────────────────────────────────────────────────────────────────────────
void LingbotDenoiseModel::set_action_weights(
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias, const at::Tensor& time_all,
    int64_t action_dim, int64_t k_comb_pad, int64_t num_steps, int64_t chunk_size) {
    TORCH_CHECK(num_layers() > 0,
                "set_action_weights must follow CausalDecoderModel::set_weights");
    TORCH_CHECK(action_dim > 0 && k_comb_pad > 0 && num_steps > 0 && chunk_size > 0,
                "action_dim/k_comb_pad/num_steps/chunk_size must be positive");
    TORCH_CHECK(k_comb_pad % 16 == 0, "k_comb_pad must be a multiple of 16 (single-core col GEMM)");
    const int64_t h = hidden_size();
    // out-dim padding for the out_proj (roundup16(action_dim)); the in-graph Euler adds
    // x_t_spm [cs,k_comb_pad] + v_t [cs,action_dim_pad], so the two widths must match.
    action_dim_ = action_dim;
    action_dim_pad_ = ((action_dim + 15) / 16) * 16;
    k_comb_pad_ = k_comb_pad;
    TORCH_CHECK(k_comb_pad_ == action_dim_pad_,
        "in-graph Euler requires k_comb_pad(", k_comb_pad_, ")==action_dim_pad(",
        action_dim_pad_, ")");
    TORCH_CHECK(mo_bias.dim() == 1 && mo_bias.size(0) == h
             && mo_bias.scalar_type() == at::kHalf
             && mo_bias.device().type() == at::kPrivateUse1,
        "mo_bias must be [hidden] fp16 RPU");
    TORCH_CHECK(op_bias.dim() == 1 && op_bias.size(0) == action_dim_pad_
             && op_bias.scalar_type() == at::kHalf
             && op_bias.device().type() == at::kPrivateUse1,
        "op_bias must be [action_dim_pad] fp16 RPU");
    TORCH_CHECK(time_all.dim() == 2 && time_all.size(0) == num_steps && time_all.size(1) == h
             && time_all.scalar_type() == at::kHalf && time_all.is_contiguous()
             && time_all.device().type() == at::kPrivateUse1,
        "time_all must be [num_steps, hidden] fp16 contig RPU");
    wc_ = wc; mo_ = mo; mo_bias_ = mo_bias;
    op_ = op; op_bias_ = op_bias; time_all_ = time_all;
    num_steps_ = num_steps;
    chunk_size_ = chunk_size;   // uniform-cs: state token (row 0) + N_ACTION action rows
    // Stable staging DDR: the pre-hook copies encoder rows 1:cs here; layer 0 reads it via
    // FMB hidden_in_src_base_ (fixed src — emb_stage_ data_ptr stable). Row 0 = state.
    emb_stage_ = at::empty({1, chunk_size_, h},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    invalidate_model_state();
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers / static_config / dynamic_config
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BufferDecl> LingbotDenoiseModel::explicit_mask_baseline_buffer_declarations(const LayoutContext& ctx) {
    auto d = causal_baseline_buffer_declarations(ctx);
    const int64_t cs = ctx.chunk_size, h = hidden_size();
    constexpr int DW = 2;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
    using SC = StorageClass;
    constexpr BufferScope ALL = BufferScope::LayerWide;
    // x_t_spm holds x across the whole step (phase 0..9) so the post-hook's in-graph Euler
    // can read it — private slot (no alias onto residual1). Pre-hook temps live at phase 0..0
    // (dead before layer 0, residual1 is 1..8), so the allocator may alias them safely.
    d.push_back({"x_t_spm",       A(cs * k_comb_pad_ * DW), 0, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"enc_h_spm",     A(cs * h * DW),           0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"emb_core0_spm", A(cs * h * DW),           0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"time_spm",      A(h * DW),                0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"mo_b_spm",      A(h * DW),                0, 0, SC::Temp, 0, nullptr, ALL});
    // Post-hook output at phase 8..9 — must NOT alias residual1 (1..8), which the post-hook reads.
    d.push_back({"v_t_core0_spm", A(cs * action_dim_pad_ * DW), 8, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"op_b_spm",      A(action_dim_pad_ * DW),      8, 9, SC::Temp, 0, nullptr, ALL});
    return d;
}

ModelStaticConfig LingbotDenoiseModel::static_config() {
    ModelStaticConfig cfg = CausalDecoderModel::static_config();
    cfg.pre_layers_fn  = reinterpret_cast<void (FusedModelBase::*)()>(
        &LingbotDenoiseModel::emit_pre_layers_body);
    cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
        &LingbotDenoiseModel::emit_post_layers_body);
    cfg.body_iterations = num_steps_;   // in-graph denoise unroll
    return cfg;
}

ModelDynamicConfig LingbotDenoiseModel::dynamic_config(const ChunkPlan& plan) {
    ModelDynamicConfig cfg = CausalDecoderModel::dynamic_config(plan);
    cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    return cfg;
}

bool LingbotDenoiseModel::subclass_chunk_size_valid(
    int64_t chunk_size, int64_t seq_len, int64_t position) const {
    return chunk_size >= seq_len &&
        CausalDecoderModel::subclass_chunk_size_valid(
            chunk_size, seq_len, position);
}

FmbPhysicalExecutionManifest
LingbotDenoiseModel::physical_manifest_for_candidate(
    const FmbThreeStageChunkPlan& plan, const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len, int64_t position) const {
    TORCH_CHECK(
        layout.use_attn_mask && !layout.is_causal &&
            plan.compute.chunks.size() == 1 && physical_len == chunk_size_,
        "LingBot-V1 denoise COMPLETE descriptor requires one masked action chunk");
    FmbPhysicalExecutionManifest manifest =
        CausalDecoderModel::physical_manifest_for_candidate(
            plan, layout, physical_len, logical_len, position);
    auto append = [&](FmbRouteFamily family, int64_t site_id,
                      int64_t selector, int64_t invocation,
                      std::vector<int64_t> arguments = {}) {
        manifest.routes.push_back({
            site_id, family, selector, /*flags=*/0,
            std::move(arguments), invocation});
    };
    for (int64_t body = 0; body < num_steps_; ++body) {
        if (body == 0) {
            append(
                FmbRouteFamily::MUTABLE_DMA,
                LINGBOT_DENOISE_PRE_X_DMA_SITE,
                static_cast<int64_t>(
                    LingbotDenoiseMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                body, {0});
        }
        append(FmbRouteFamily::LINEAR,
               LINGBOT_DENOISE_PRE_WC_LINEAR_SITE,
               static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE), body);
        append(FmbRouteFamily::LINEAR,
               LINGBOT_DENOISE_PRE_MO_LINEAR_SITE,
               static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE), body);
        append(FmbRouteFamily::LINEAR,
               LINGBOT_DENOISE_POST_OP_LINEAR_SITE,
               static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE), body);
        append(
            FmbRouteFamily::MUTABLE_DMA,
            LINGBOT_DENOISE_POST_TRAJECTORY_DMA_SITE,
            static_cast<int64_t>(
                LingbotDenoiseMutableDmaRoute::SPM_COPY_TO_DDR),
            body, {body * chunk_size_ * k_comb_pad_ * 2});
    }
    std::sort(
        manifest.routes.begin(), manifest.routes.end(),
        [](const FmbRouteManifestEntry& lhs,
           const FmbRouteManifestEntry& rhs) {
            return std::make_tuple(
                       static_cast<int64_t>(lhs.family), lhs.site_id,
                       lhs.invocation) <
                std::make_tuple(
                       static_cast<int64_t>(rhs.family), rhs.site_id,
                       rhs.invocation);
        });
    return manifest;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook bodies
// ─────────────────────────────────────────────────────────────────────────────
void LingbotDenoiseModel::emit_pre_layers_body() {
    const int64_t h = hidden_size(), cs = chunk_size_, kp = k_comb_pad_;
    const int64_t bit = ctx().body_iter;   // in-graph unroll iteration

    // [A1] x0 DDR -> x_t_spm (MUTABLE), iteration 0 only. Iters 1..N-1 read the in-SPM
    // Euler result the prior post-hook left in x_t_spm (persistent [0,9] slot).
    if (bit == 0) {
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA,
            LINGBOT_DENOISE_PRE_X_DMA_SITE,
            static_cast<int64_t>(
                LingbotDenoiseMutableDmaRoute::DDR_BROADCAST_TO_SPM),
            /*resolved_flags=*/0, {0}, bit);
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &x_t_src_base_, /*src_offset_bytes=*/0, /*num_elements=*/cs * kp,
            addr(0, "x_t_spm"), /*num_cores=*/NUM_CORES);
    }
    // [A2] time_all[bit] [hidden] -> time_spm (core 0), used as the W_comp GEMM bias.
    // FIXED src (time_all_ is a stable registered weight); the per-body bit offset is
    // baked at BUILD (body_iter constant per emit).
    rpu_launch_ddr_broadcast_spm_dma(
        time_all_.data_ptr<c10::Half>() + bit * h, h, addr(0, "time_spm"), /*num_cores=*/1);
    // [A3] mo_bias [hidden] -> mo_b_spm (core 0, constant contents, FIXED src).
    rpu_launch_ddr_broadcast_spm_dma(
        mo_bias_.data_ptr<c10::Half>(), h, addr(0, "mo_b_spm"), /*num_cores=*/1);

    // [A4] enc_h = x_t_spm @ wc_.T + time_bias  (single-core; M=cs,N=h,K=kp).
    ctx().consume_physical_route(
        FmbRouteFamily::LINEAR, LINGBOT_DENOISE_PRE_WC_LINEAR_SITE,
        static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
        /*resolved_flags=*/0, {}, bit);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "x_t_spm"), wc_, addr(0, "enc_h_spm"),
        /*M=*/cs, /*N=*/h, /*K=*/kp, /*partition=*/1, /*num_cores=*/1,
        /*bias_spm_addr=*/addr(0, "time_spm"));
    // [A5] SiLU(enc_h) in place, core 0.
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "enc_h_spm"), addr(0, "enc_h_spm"), cs * h, ValuOpType::SILU,
        /*is_gelu=*/false, /*num_cores=*/1);
    // [A6] ate = silu(enc_h) @ mo_.T + mo_bias  (single-core) -> emb_core0_spm.
    ctx().consume_physical_route(
        FmbRouteFamily::LINEAR, LINGBOT_DENOISE_PRE_MO_LINEAR_SITE,
        static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
        /*resolved_flags=*/0, {}, bit);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "enc_h_spm"), mo_, addr(0, "emb_core0_spm"),
        /*M=*/cs, /*N=*/h, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
        /*bias_spm_addr=*/addr(0, "mo_b_spm"));
    // [A7] emb_core0_spm[1:cs] -> emb_stage_[1:cs] (row-1 byte offset on BOTH src and dst;
    // row 0 = state stays what unroll_forward wrote). Layer 0's emit_layer_input_dma reads
    // emb_stage_ (all cs rows) via FMB hidden_in_src_base_.
    rpu_launch_spm_copy_ddr_dma(
        addr(0, "emb_core0_spm") + h * DWIDTH,
        emb_stage_.data_ptr<c10::Half>() + h, (cs - 1) * h);
}

void LingbotDenoiseModel::emit_post_layers_body() {
    const int64_t h = hidden_size(), cs = chunk_size_, np = action_dim_pad_, kp = k_comb_pad_;
    const int64_t bit = ctx().body_iter;
    // [Z1] op_bias [np] -> op_b_spm (core 0, constant contents, FIXED src).
    rpu_launch_ddr_broadcast_spm_dma(
        op_bias_.data_ptr<c10::Half>(), np, addr(0, "op_b_spm"), /*num_cores=*/1);
    // [Z2] v_t = residual1(final-normed hidden) @ op_.T + op_bias, single-core. residual1
    // is [cs,h]; op_ zero-padded to np rows so v_t[:, action_dim:np]=0. Do NOT re-normalize:
    // the last layer already applied the plain final RMSNorm in-place to residual1.
    ctx().consume_physical_route(
        FmbRouteFamily::LINEAR, LINGBOT_DENOISE_POST_OP_LINEAR_SITE,
        static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
        /*resolved_flags=*/0, {}, bit);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), op_, addr(0, "v_t_core0_spm"),
        /*M=*/cs, /*N=*/np, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
        /*bias_spm_addr=*/addr(0, "op_b_spm"));
    // [Z3] In-graph Euler: x_t_spm = x_t_spm + dt·v_t (in-place SPM, core 0). np==kp so the
    // two [cs,np] buffers add elementwise; pad cols stay 0 (x pad 0 + dt·0). dt_ baked at BUILD.
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "x_t_spm"), addr(0, "v_t_core0_spm"), addr(0, "x_t_spm"),
        cs * kp, ValuOpType::ADD, dt_, /*num_cores=*/1);
    // [Z4] x_t_spm (= x_new) -> x_traj[bit] DDR (MUTABLE). Final x = x_traj[-1].
    const int64_t trajectory_offset = bit * cs * kp * 2;
    ctx().consume_physical_route(
        FmbRouteFamily::MUTABLE_DMA,
        LINGBOT_DENOISE_POST_TRAJECTORY_DMA_SITE,
        static_cast<int64_t>(LingbotDenoiseMutableDmaRoute::SPM_COPY_TO_DDR),
        /*resolved_flags=*/0, {trajectory_offset}, bit);
    rpu_launch_spm_copy_ddr_dma_mutable(
        addr(0, "x_t_spm"), &x_traj_dst_base_,
        /*dst_offset_bytes=*/trajectory_offset, cs * kp);
}

// ─────────────────────────────────────────────────────────────────────────────
// unroll_forward
// ─────────────────────────────────────────────────────────────────────────────
void LingbotDenoiseModel::unroll_forward(
    const at::Tensor& x0_rpu, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches, const at::Tensor& state_emb,
    const std::optional<at::Tensor>& attention_mask,
    at::Tensor& x_traj, double dt, int64_t prefix_len, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor) {
    TORCH_CHECK(action_dim_pad_ > 0, "unroll_forward before set_action_weights");
    TORCH_CHECK(num_steps == num_steps_,
        "num_steps(", num_steps, ") must match set_action_weights num_steps(", num_steps_, ")");
    const int64_t cs = chunk_size_, kp = k_comb_pad_, h = hidden_size();

    // Node-count guard (graph_runtime silently drops on batch overflow; cap 32768).
    TORCH_CHECK(num_steps * 1300 < 32000,
        "denoise unroll (", num_steps, " bodies) would approach the 32768 batch-item cap");

    TORCH_CHECK(x0_rpu.dim() == 3 && x0_rpu.size(0) == 1 && x0_rpu.size(1) == cs
        && x0_rpu.size(2) == kp && x0_rpu.scalar_type() == at::kHalf
        && x0_rpu.is_contiguous() && x0_rpu.device().type() == at::kPrivateUse1,
        "x0_rpu must be [1,", cs, ",", kp, "] fp16 contig RPU");
    TORCH_CHECK(x_traj.dim() == 3 && x_traj.size(0) == num_steps && x_traj.size(1) == cs
        && x_traj.size(2) == kp && x_traj.scalar_type() == at::kHalf
        && x_traj.is_contiguous() && x_traj.device().type() == at::kPrivateUse1,
        "x_traj must be [", num_steps, ",", cs, ",", kp, "] fp16 contig RPU");
    TORCH_CHECK(state_emb.dim() == 1 && state_emb.size(0) == h
        && state_emb.scalar_type() == at::kHalf && state_emb.device().type() == at::kPrivateUse1,
        "state_emb must be [hidden] fp16 RPU");

    // dt is the SIGNED Euler coefficient baked into the in-graph Euler at BUILD (x += dt·v);
    // identical across the N unrolled iterations. LingBot's flow runs t: 1→0 so dt = -1/N < 0.
    TORCH_CHECK(dt != 0.0 && std::isfinite(dt), "dt must be a nonzero finite Euler step");
    const c10::Half dt_half = c10::Half(static_cast<float>(dt));
    if (!dt_pinned_) { dt_ = dt_half; dt_pinned_ = true; }
    else TORCH_CHECK(dt_ == dt_half, "dt changed across replays (baked into the graph)");

    // KV headroom (K layout max_seq = size(1)*size(5)).
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + cs <= kc.size(1) * kc.size(5),
            "prefix_len(", prefix_len, ")+cs(", cs, ") exceeds k_cache capacity ",
            kc.size(1) * kc.size(5));
    }

    // emb_stage_ row 0 = state token (constant across the N iters; the pre-hook only
    // touches rows 1:cs). Set once per call + flush so the on-device layer-0 DMA sees it.
    state_ref_ = state_emb;
    emb_stage_[0][0].copy_(state_emb);
    rpu_ddr_flush_force(emb_stage_.data_ptr<c10::Half>());

    // Pin the per-call mutable DMA bases + flush the live x0 (wrappers do not flush).
    x0_ref_ = x0_rpu; x_traj_ref_ = x_traj;   // keepalive across the synchronous forward
    x_t_src_base_    = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
    x_traj_dst_base_ = ::rhino_lkn::RpuGetDevAddr(x_traj.data_ptr());
    rpu_ddr_flush_force(x0_rpu.data_ptr<c10::Half>());

    // ONE forward — run_all_layers loops body_iterations(=num_steps_) internally; the
    // pre/post hooks + AdaRMS refresh fire per iteration with ctx().body_iter advancing.
    // Plain 1D-RoPE (mrope_section=[]) ⇒ no position_ids.
    (void) CausalDecoderModel::forward(
        emb_stage_, k_caches, v_caches, attention_mask,
        /*position=*/prefix_len, /*is_causal=*/false,
        /*position_ids=*/std::nullopt,
        /*deepstack_dense_visual_embeds=*/std::nullopt,
        /*rope_cos_il=*/std::nullopt, /*rope_sin_il=*/std::nullopt,
        /*cos_sin_offset=*/-1, /*batch_slot=*/0,
        /*allow_batch_decode=*/false, /*planned_chunk_size=*/0,
        planned_stage_descriptor);
}

}  // namespace v3

// =============================================================================
// Instance registry — ModelHandleRegistry<v3::LingbotDenoiseModel>
// =============================================================================
using LingbotDenoiseRegistry = ModelHandleRegistry<v3::LingbotDenoiseModel>;

std::vector<int64_t> rpu_lingbot_denoise_planner_cache_identity(int64_t handle) {
    return LingbotDenoiseRegistry::get(handle, "rpu_lingbot_denoise_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_lingbot_denoise_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    LingbotDenoiseRegistry::get(handle, "rpu_lingbot_denoise_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_lingbot_denoise_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return LingbotDenoiseRegistry::get(handle, "rpu_lingbot_denoise_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_lingbot_denoise_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return LingbotDenoiseRegistry::get(handle, "rpu_lingbot_denoise_kvinsert_cost_domain")
        ->kvinsert_cost_domain("lingbot_denoise", descriptor);
}

std::string rpu_lingbot_denoise_kvinsert_cost_catalog_sha256(int64_t handle) {
    return LingbotDenoiseRegistry::get(
        handle, "rpu_lingbot_denoise_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================
int64_t rpu_lingbot_denoise_create() {
    return LingbotDenoiseRegistry::create();
}

void rpu_lingbot_denoise_destroy(int64_t handle) {
    LingbotDenoiseRegistry::destroy(handle, "rpu_lingbot_denoise_destroy");
}

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
    int64_t action_dim, int64_t k_comb_pad, int64_t num_steps, int64_t chunk_size)
{
    auto* m = LingbotDenoiseRegistry::get(handle, "rpu_lingbot_denoise_set_weights");
    // Plain Qwen2 768-d decoder (no q/k norm; QKV bias present; plain 1D-RoPE mrope=[]).
    m->CausalDecoderModel::set_weights(
        q_w, k_w, v_w, o_w,
        /*q_norm=*/{}, /*k_norm=*/{},
        input_norm, post_norm,
        gate_w, up_w, down_w,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size,
        eps, use_silu,
        /*mrope_section=*/{}, /*deepstack_lang_layers=*/{},
        q_bias, k_bias, v_bias);
    // MR-A: this fused subsystem inherits CausalDecoderModel, so it inherits the
    // deny-by-default certified-envelope gate too. The fixed 51-row action
    // suffix requires one ceil16-aligned chunk, so both auto and exact requests
    // share the admitted positive ceiling 64.
    // See docs/roadmap/chunk_certified_envelope.md.
    m->set_chunk_envelope(/*max_kv_len=*/8192, /*chunk=*/64);
    m->set_action_weights(wc, mo, mo_bias, op, op_bias, time_all,
                          action_dim, k_comb_pad, num_steps, chunk_size);
    m->set_adarms_unroll(adarms_is, adarms_ish, adarms_ps, adarms_psh);
}

void rpu_lingbot_denoise_set_chunk_size(int64_t handle, int64_t chunk_size) {
    LingbotDenoiseRegistry::get(
        handle, "rpu_lingbot_denoise_set_chunk_size")
        ->set_execution_chunk_size(chunk_size);
}

int64_t rpu_lingbot_denoise_get_resolved_chunk_size(int64_t handle) {
    return LingbotDenoiseRegistry::get(
        handle, "rpu_lingbot_denoise_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

std::vector<int64_t> rpu_lingbot_denoise_resolve_action_stage_domain(
    int64_t handle, int64_t execution_len, int64_t prefix_len,
    int64_t mask_kv_len) {
    return LingbotDenoiseRegistry::get(
        handle, "rpu_lingbot_denoise_resolve_action_stage_domain")
        ->resolve_action_stage_domain(execution_len, prefix_len, mask_kv_len);
}

void rpu_lingbot_denoise_unroll_forward(
    int64_t handle, const at::Tensor& x0_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& state_emb, const std::optional<at::Tensor>& attention_mask,
    at::Tensor x_traj, double dt, int64_t prefix_len, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor)
{
    LingbotDenoiseRegistry::get(handle, "rpu_lingbot_denoise_unroll_forward")
        ->unroll_forward(x0_rpu, k_caches, v_caches, state_emb, attention_mask,
                         x_traj, dt, prefix_len, num_steps,
                         planned_stage_descriptor);
}
