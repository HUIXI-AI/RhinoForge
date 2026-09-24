// rpu_wall_oss_action_step_model.cpp — fused Wall-OSS-0.5 action-denoise step op.
// See rpu_wall_oss_action_step_model.h and
// rpu_wall_oss_action_step_model.h for the buffer and execution contracts.
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
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

namespace v3 {

// WALL_ACTION_FIXED_KERNEL_BASIS: non-manifest launchers implement fixed action
// preprocessing/Euler/postprocessing or BufferDecl transport with no route
// candidate. Any selectable implementation must enter the typed manifest.

// Stable semantic site IDs from planner_owners.v1.json.  The call sites stay
// in the hook bodies; the descriptor names the concrete route consumed there.
constexpr int64_t WALL_ACTION_PRE_X_DMA_SITE = 5787673053685733969LL;
constexpr int64_t WALL_ACTION_PRE_AE_DMA_SITE = 4362865405816001779LL;
constexpr int64_t WALL_ACTION_PRE_AE_LINEAR_SITE = 1688526320245811297LL;
constexpr int64_t WALL_ACTION_PRE_TE_DMA_SITE = 4402755508028461462LL;
constexpr int64_t WALL_ACTION_PRE_TE_LINEAR_SITE = 4391159404409762012LL;
constexpr int64_t WALL_ACTION_PRE_BIAS_DMA_SITE = 6039350177777656796LL;
constexpr int64_t WALL_ACTION_PRE_BIAS_LINEAR_SITE = 7480532815849100965LL;
constexpr int64_t WALL_ACTION_PRE_FULL_LINEAR_SITE = 4595605214237105006LL;
constexpr int64_t WALL_ACTION_PRE_FULL_DMA_SITE = 1743102137601489253LL;
constexpr int64_t WALL_ACTION_PRE_W3_LINEAR_SITE = 217794874262964361LL;
constexpr int64_t WALL_ACTION_PRE_W3_ALL_GATHER_SITE = 5597784804649223890LL;
constexpr int64_t WALL_ACTION_POST_PROJ_LINEAR_SITE = 5819548213260683213LL;
constexpr int64_t WALL_ACTION_POST_V_TRAJ_DMA_SITE = 3467321171562977027LL;
constexpr int64_t WALL_ACTION_POST_V_STEP_DMA_SITE = 6728649211428525293LL;
constexpr int64_t WALL_ACTION_POST_X_TRAJ_DMA_SITE = 2395865154258890852LL;
constexpr int64_t WALL_ACTION_POST_X_STEP_DMA_SITE = 5413488959205014333LL;

enum class WallActionMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM = 1,
    SPM_COPY_TO_DDR = 2,
};

constexpr int64_t wall_action_linear_route() {
    return static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE);
}

constexpr int64_t wall_action_dma_route(WallActionMutableDmaRoute route) {
    return static_cast<int64_t>(route);
}

WallOssActionStepModel::WallOssActionStepModel() {
    keep_explicit_mask_in_spm_ = true;
    pre_layers_residual1_ready_ = true;
}
WallOssActionStepModel::~WallOssActionStepModel() = default;

// ─────────────────────────────────────────────────────────────────────────────
// set_action_weights — Task 5
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
    invalidate_model_state();   // D-503
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers / static_config / dynamic_config — Task 5 (stubs call base)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BufferDecl> WallOssActionStepModel::declare_buffers(const LayoutContext& ctx) {
    auto d = CausalDecoderModel::declare_buffers(ctx);
    const int64_t cs = ctx.chunk_size, h = hidden_size();
    constexpr int DW = 2;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
    using SC = StorageClass;
    constexpr BufferScope ALL = BufferScope::LayerWide;

    // H20 has a short v2 KV-insert tail. Reserve the existing K/V slots through
    // ceil16(H) so the launcher can emit one v16 insert; SDPA still receives the
    // logical P+H length and cannot read the padded cache tail.
    kv_insert_spm_rows_ = Align(cs, (int64_t)16);
    const int64_t kv_bytes = A(
        kv_insert_spm_rows_ * num_kv_heads() * head_dim() / attn_tp() * DW);
    bool found_k = false, found_v = false, found_mask = !ctx.use_attn_mask;
    for (auto& b : d) {
        if (!b.name) continue;
        if (std::strcmp(b.name, "k") == 0) {
            b.size = std::max(b.size, kv_bytes);
            found_k = true;
        } else if (std::strcmp(b.name, "v") == 0) {
            b.size = std::max(b.size, kv_bytes);
            found_v = true;
        } else if (std::strcmp(b.name, "sdpa_mask") == 0) {
            // The mask is constant within one replay. Keep it live across the
            // pre-hook, every layer, and every unrolled body.
            b.phase_start = 0;
            b.phase_end = 9;
            b.scope = ALL;
            found_mask = true;
        }
    }
    TORCH_CHECK(found_k && found_v && found_mask,
                "Wall Action SPM optimization could not find base K/V/mask slots");

    // BufferDecl: {name, size, phase_start, phase_end, storage, per_layer, alias_of, scope}.
    // Pre-hook temps live at phase 0..0 — dead before layer 0 (residual1 is 1..8), so the
    // overlap-only allocator may alias them onto residual1's slot (safe).
    // EXCEPT x_t_spm: it holds x across the WHOLE step (phase 0..9) so the post-hook's
    // in-graph Euler (x += dt·v_t) can read it. Its overlapping lifetime forces a private
    // slot (no alias onto residual1) — ~2 KB/core, trivial vs the 8109 KB budget.
    d.push_back({"x_t_spm",       A(cs * k_comb_pad_ * DW), 0, 9, SC::Temp, 0, nullptr, ALL});
    d.push_back({"ce_spm",        A(cs * h * DW),           0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"b_step_spm",    A(cs * h * DW),           0, 0, SC::Temp, 0, nullptr, ALL});
    // Keep each TP8 w3 shard distinct from residual1 until all-gather consumes it.
    d.push_back({"emb_core0_spm", A(cs * (h / NUM_CORES) * DW), 0, 1,
                 SC::Temp, 0, nullptr, ALL});
    // On-device base = ae_dof·w2_a and ct = te·w2_t (core 0; [1,h] each).
    // ct_spm keeps the step-invariant base across every unrolled body; ae_spm/te_spm are
    // per-body inputs/temps. All three are used only when have_te_.
    d.push_back({"ae_spm",        A(h * DW),                0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"te_spm",        A(h * DW),                0, 0, SC::Temp, 0, nullptr, ALL});
    d.push_back({"ct_spm",        A(h * DW),                0, 9, SC::Temp, 0, nullptr, ALL});
    // Post-hook output at phase 8..9 — must NOT alias residual1 (1..8), which the post-hook
    // reads (final-normed hidden). Coexists with lm_head_out (also 8..9) in a separate slot.
    d.push_back({"v_t_core0_spm", A(cs * action_dim_pad_ * DW), 8, 9, SC::Temp, 0, nullptr, ALL});
    return d;
}

ModelStaticConfig WallOssActionStepModel::static_config() {
    ModelStaticConfig cfg = CausalDecoderModel::static_config();
    cfg.pre_layers_fn  = reinterpret_cast<void (FusedModelBase::*)()>(
        &WallOssActionStepModel::emit_pre_layers_body);
    cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
        &WallOssActionStepModel::emit_post_layers_body);
    cfg.body_iterations = loop_mode_ ? num_steps_ : 1;   // EXT-unroll (denoise loop)
    return cfg;
}

ModelDynamicConfig WallOssActionStepModel::dynamic_config(const ChunkPlan& plan) {
    ModelDynamicConfig cfg = CausalDecoderModel::dynamic_config(plan);
    cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    // Action queries consume the expert-0 prefix from the shared DDR cache.
    // There is no raw-SPM ownership ABI between those two children.
    cfg.attention_policy = AttentionExecutionPolicy::DDR_KV;
    return cfg;
}

std::vector<int64_t> WallOssActionStepModel::resolve_action_stage_domain(
    int64_t horizon, int64_t prefix_len, int64_t mask_kv_len,
    int64_t requested_chunk_size, bool loop_mode, int64_t num_steps,
    bool b_step_is_bias, bool have_te) {
    TORCH_CHECK(action_dim_pad_ > 0,
                "RPU_PLANNER_REJECT:CAPABILITY: Wall action weights are not initialized");
    TORCH_CHECK(horizon > 0 && prefix_len >= 0 &&
                    mask_kv_len == prefix_len + horizon,
                "RPU_PLANNER_REJECT:CAPABILITY: Wall action requires a positive "
                "horizon and mask width prefix_len+horizon");
    const int64_t required_chunk = Align(horizon, int64_t{16});
    TORCH_CHECK(
        requested_chunk_size == 0 || requested_chunk_size == required_chunk,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: Wall action exact chunk must "
        "equal ceil16(horizon): requested=", requested_chunk_size,
        " horizon=", horizon, " required=", required_chunk);
    TORCH_CHECK(num_steps > 0 && (!loop_mode || num_steps * 1100 < 65536),
                "RPU_PLANNER_REJECT:CAPABILITY: invalid Wall action body count ",
                num_steps);
    TORCH_CHECK(loop_mode || num_steps == 1,
                "RPU_PLANNER_REJECT:CAPABILITY: stepwise Wall action descriptor "
                "must describe one body");
    TORCH_CHECK(!have_te || b_step_is_bias,
                "RPU_PLANNER_REJECT:CAPABILITY: Wall action te route requires "
                "the row-constant bias route");

    std::vector<int64_t> result;
    auto query_scope = capture_kvinsert_cost_layout_scope();
    query_scope([&] {
        planned_action_route_profile_valid_ = true;
        planned_loop_mode_ = loop_mode;
        planned_num_steps_ = num_steps;
        planned_b_step_is_bias_ = b_step_is_bias;
        planned_have_te_ = have_te;
        auto mask_shape = at::empty(
            {horizon, mask_kv_len},
            at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
        result = encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_with_physical_context(
                horizon, prefix_len, std::optional<at::Tensor>(mask_shape),
                /*is_causal=*/false, requested_chunk_size,
                /*logical_len=*/horizon, /*rope_mode=*/1,
                FmbGraphLifecycle::COMPOSITE_CHILD));
    });
    return result;
}

void WallOssActionStepModel::validate_planned_action_route_profile(
    at::IntArrayRef descriptor, bool loop_mode, int64_t num_steps,
    bool b_step_is_bias, bool have_te) {
    // A prepared descriptor owns this request. A later dry query may have
    // described another graph, so its mutable oracle profile is not authority.
    const auto prepared_candidate = prepare_stage_candidate(descriptor);
    const auto& candidate = prepared_candidate->candidate();
    const auto& manifest = candidate.physical_manifest;
    const std::vector<int64_t> expected{
        b_step_is_bias ? 1 : 0,
        loop_mode ? 1 : 0, have_te ? 1 : 0, num_steps};
    const auto route = std::find_if(manifest.routes.begin(), manifest.routes.end(),
        [](const FmbRouteManifestEntry& item) {
            return item.family == FmbRouteFamily::LINEAR &&
                item.site_id == WALL_ACTION_PRE_W3_LINEAR_SITE;
        });
    TORCH_CHECK(
        manifest.state == FmbPhysicalManifestState::COMPLETE &&
            manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD &&
            route != manifest.routes.end() && route->invocation == 0 &&
            route->flags == 0 && route->selector == wall_action_linear_route() &&
            route->arguments == expected,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: Wall action forward branch "
        "does not match the native descriptor request");
    planned_action_route_profile_valid_ = true;
    planned_loop_mode_ = loop_mode;
    planned_num_steps_ = num_steps;
    planned_b_step_is_bias_ = b_step_is_bias;
    planned_have_te_ = have_te;
}

void WallOssActionStepModel::consume_action_route(
    FmbRouteFamily family, int64_t site_id, int64_t selector,
    std::vector<int64_t> arguments) {
    if (!ctx().has_complete_physical_manifest()) return;
    ctx().consume_physical_route(
        family, site_id, selector, /*resolved_flags=*/0,
        std::move(arguments));
}

void WallOssActionStepModel::stage_execution_chunk_size(
    uint64_t token, int64_t chunk_size) {
    TORCH_CHECK(
        chunk_size == 0 || (chunk_size >= 16 && chunk_size % 16 == 0),
        "Wall action chunk must be AUTO(0) or a positive multiple of 16, got ",
        chunk_size);
    const int64_t old_chunk = configured_chunk_size_;
    stage_execution_controls(
        token, chunk_size, ChunkEnvelope{/*max_kv_len=*/8192, chunk_size},
        [this, chunk_size] { configured_chunk_size_ = chunk_size; },
        [this, old_chunk] { configured_chunk_size_ = old_chunk; },
        "WallOssActionStepModel::stage_execution_chunk_size");
}

FmbPhysicalExecutionManifest
WallOssActionStepModel::physical_manifest_for_candidate(
    const FmbThreeStageChunkPlan& plan, const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len, int64_t position) const {
    TORCH_CHECK(
        layout.use_attn_mask && !layout.is_causal &&
            plan.qkv.chunks.size() == 1 && plan.compute.chunks.size() == 1,
        "Wall action COMPLETE descriptor requires one masked action chunk");
    TORCH_CHECK(
        planned_action_route_profile_valid_,
        "RPU_PLANNER_REJECT:CAPABILITY: Wall action route profile was not "
        "supplied to the native planner");
    FmbPhysicalExecutionManifest manifest =
        CausalDecoderModel::physical_manifest_for_candidate(
            plan, layout, physical_len, logical_len, position);
    TORCH_INTERNAL_ASSERT(
        manifest.state == FmbPhysicalManifestState::COMPLETE &&
            manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD);

    auto append = [&](FmbRouteFamily family, int64_t site_id,
                      int64_t selector,
                      std::vector<int64_t> arguments = {}) {
        manifest.routes.push_back(
            {site_id, family, selector, /*flags=*/0,
             std::move(arguments)});
    };
    auto append_linear = [&](int64_t site_id) {
        append(FmbRouteFamily::LINEAR, site_id,
               wall_action_linear_route());
    };
    auto append_broadcast = [&](int64_t site_id) {
        append(FmbRouteFamily::MUTABLE_DMA, site_id,
               wall_action_dma_route(
                   WallActionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
    };
    auto append_store = [&](int64_t site_id) {
        append(FmbRouteFamily::MUTABLE_DMA, site_id,
               wall_action_dma_route(
                   WallActionMutableDmaRoute::SPM_COPY_TO_DDR));
    };

    // pre_layers: x is loaded once even for an unrolled graph; the remaining
    // sites are selected by the request-owned action route profile.
    append_broadcast(WALL_ACTION_PRE_X_DMA_SITE);

    if (planned_b_step_is_bias_) {
        if (planned_have_te_) {
            append_broadcast(WALL_ACTION_PRE_AE_DMA_SITE);
            append_linear(WALL_ACTION_PRE_AE_LINEAR_SITE);
            append_broadcast(WALL_ACTION_PRE_TE_DMA_SITE);
            append_linear(WALL_ACTION_PRE_TE_LINEAR_SITE);
        } else {
            append_broadcast(WALL_ACTION_PRE_BIAS_DMA_SITE);
        }
        append_linear(WALL_ACTION_PRE_BIAS_LINEAR_SITE);
    } else {
        append_linear(WALL_ACTION_PRE_FULL_LINEAR_SITE);
        append_broadcast(WALL_ACTION_PRE_FULL_DMA_SITE);
    }
    append(
        FmbRouteFamily::LINEAR, WALL_ACTION_PRE_W3_LINEAR_SITE,
        wall_action_linear_route(),
        {planned_b_step_is_bias_ ? 1 : 0, planned_loop_mode_ ? 1 : 0,
         planned_have_te_ ? 1 : 0, planned_num_steps_});
    append(
        FmbRouteFamily::COLLECTIVE,
        WALL_ACTION_PRE_W3_ALL_GATHER_SITE,
        static_cast<int64_t>(rpu_resolve_all_gather_schedule(
            hidden_size() / NUM_CORES, sizeof(c10::Half))));

    // post_layers: proj_back is unconditional; mutable input/output sites are
    // present only in the branch that forward will actually dispatch.
    append_linear(WALL_ACTION_POST_PROJ_LINEAR_SITE);

    append_store(planned_loop_mode_ ? WALL_ACTION_POST_V_TRAJ_DMA_SITE
                                    : WALL_ACTION_POST_V_STEP_DMA_SITE);

    append_store(planned_loop_mode_ ? WALL_ACTION_POST_X_TRAJ_DMA_SITE
                                    : WALL_ACTION_POST_X_STEP_DMA_SITE);

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

FmbPhysicalManifestForwardCapability
WallOssActionStepModel::physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& /*manifest*/) const {
    return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
}

// ─────────────────────────────────────────────────────────────────────────────
// Hook bodies — Task 6 / Task 7
// ─────────────────────────────────────────────────────────────────────────────
void WallOssActionStepModel::emit_pre_layers_body() {
    const int64_t h = hidden_size(), cs = chunk_size_;
    // EXT-unroll: which iteration of the in-graph denoise loop we are emitting.
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
        consume_action_route(
            FmbRouteFamily::MUTABLE_DMA, WALL_ACTION_PRE_X_DMA_SITE,
            wall_action_dma_route(
                WallActionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &x_t_src_base_, /*src_offset_bytes=*/0, /*num_elements=*/cs * k_comb_pad_,
            addr(0, "x_t_spm"), /*num_cores=*/NUM_CORES);
    }
    // GraphSignature, while the prefix address/value is refreshed through the

    if (b_step_is_bias_) {
        // Row-constant [hidden] bias -> b_step_spm (core 0), broadcast per-output-feature over
        // the cs rows by the W_comb GEMM. Two modes:
        //   have_te_: b_step carries ae_dof (= dof·w1_dof); base = ae_dof·w2_a and
        //             ct = te·w2_t are both computed in-SPM here, b_step_spm = base + ct.
        //   else:     b_step is the full host-precomputed B_step, used directly.
        if (have_te_ == 1) {
            // [A1a] The action-mask term is constant across the unrolled steps. Compute it
            // once in body 0 and retain it in ct_spm until the replay finishes.
            if (bit == 0) {
                consume_action_route(
                    FmbRouteFamily::MUTABLE_DMA,
                    WALL_ACTION_PRE_AE_DMA_SITE,
                    wall_action_dma_route(
                        WallActionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
                rpu_launch_ddr_broadcast_spm_dma_mutable(
                    &b_step_src_base_, /*src_offset_bytes=*/0, /*num_elements=*/h,
                    addr(0, "ae_spm"), /*num_cores=*/1);
                consume_action_route(
                    FmbRouteFamily::LINEAR,
                    WALL_ACTION_PRE_AE_LINEAR_SITE,
                    wall_action_linear_route());
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    addr(0, "ae_spm"), w2_a_, addr(0, "ct_spm"),
                    /*M=*/1, /*N=*/h, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
                    /*bias_spm_addr=*/0);
            }
            // [A1b] te_step [hidden] DDR -> te_spm; time = te @ w2_t.T -> ae_spm (M=1 GEMV).
            // te_off_b selects te_all[bit] in the loop (0 for single-step).
            consume_action_route(
                FmbRouteFamily::MUTABLE_DMA,
                WALL_ACTION_PRE_TE_DMA_SITE,
                wall_action_dma_route(
                    WallActionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &te_src_base_, /*src_offset_bytes=*/te_off_b, /*num_elements=*/h,
                addr(0, "te_spm"), /*num_cores=*/1);
            consume_action_route(
                FmbRouteFamily::LINEAR,
                WALL_ACTION_PRE_TE_LINEAR_SITE,
                wall_action_linear_route());
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "te_spm"), w2_t_, addr(0, "ae_spm"),
                /*M=*/1, /*N=*/h, /*K=*/h, /*partition=*/1, /*num_cores=*/1,
                /*bias_spm_addr=*/0);
            // [A1c] b_step_spm = base + time (in-SPM, core 0; preserve operand order).
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "ct_spm"), addr(0, "ae_spm"), addr(0, "b_step_spm"),
                h, ValuOpType::ADD, c10::Half(1.0), /*num_cores=*/1);
        } else {
            // bias, no-te: b_all[bit] is this iteration's [hidden] bias (bias_off_b=0 single-step).
            consume_action_route(
                FmbRouteFamily::MUTABLE_DMA,
                WALL_ACTION_PRE_BIAS_DMA_SITE,
                wall_action_dma_route(
                    WallActionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &b_step_src_base_, /*src_offset_bytes=*/bias_off_b, /*num_elements=*/h,
                addr(0, "b_step_spm"), /*num_cores=*/1);
        }
        consume_action_route(
            FmbRouteFamily::LINEAR,
            WALL_ACTION_PRE_BIAS_LINEAR_SITE,
            wall_action_linear_route());
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "x_t_spm"), w_comb_, addr(0, "ce_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/k_comb_pad_,
            /*partition=*/1, /*num_cores=*/1, /*bias_spm_addr=*/addr(0, "b_step_spm"));
    } else {
        // [H,hidden] B_step: GEMM (no bias) then a same-shape add at num_cores=1.
        consume_action_route(
            FmbRouteFamily::LINEAR,
            WALL_ACTION_PRE_FULL_LINEAR_SITE,
            wall_action_linear_route());
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "x_t_spm"), w_comb_, addr(0, "ce_spm"),
            /*M=*/cs, /*N=*/h, /*K=*/k_comb_pad_,
            /*partition=*/1, /*num_cores=*/1, /*bias_spm_addr=*/0);
        // full [H,hidden] B_step: b_all[bit] is this iteration's [cs,hidden] (full_off_b=0 single-step).
        consume_action_route(
            FmbRouteFamily::MUTABLE_DMA,
            WALL_ACTION_PRE_FULL_DMA_SITE,
            wall_action_dma_route(
                WallActionMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &b_step_src_base_, /*src_offset_bytes=*/full_off_b, /*num_elements=*/cs * h,
            addr(0, "b_step_spm"), /*num_cores=*/1);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "ce_spm"), addr(0, "b_step_spm"), addr(0, "ce_spm"),
            cs * h, ValuOpType::ADD, c10::Half(1.0), /*num_cores=*/1);
    }

    // [A2] SiLU(ce) in place, core 0. is_gelu passed EXPLICITLY (else num_cores=1 would
    // bind to the is_gelu bool — codex rnd3).
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "ce_spm"), addr(0, "ce_spm"), cs * h, ValuOpType::SILU,
        /*is_gelu=*/false, /*num_cores=*/1);

    // [A3] Replicate the input, run the col-partitioned w3 on all cores, then
    // gather its shards directly into every core's layer-0 residual input.
    rpu_launch_spm_scatter_spm_dma(
        addr(0, "ce_spm"), cs * h, /*core_stride_bytes=*/0,
        addr(0, "b_step_spm"), NUM_CORES);
    consume_action_route(
        FmbRouteFamily::LINEAR, WALL_ACTION_PRE_W3_LINEAR_SITE,
        wall_action_linear_route(),
        {b_step_is_bias_ ? 1 : 0, loop_mode_ ? 1 : 0,
         have_te_ == 1 ? 1 : 0, loop_mode_ ? num_steps_ : 1});
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "b_step_spm"), w3_, addr(0, "emb_core0_spm"),
        /*M=*/cs, /*N=*/h, /*K=*/h, /*partition=*/1,
        /*num_cores=*/NUM_CORES, /*bias_spm_addr=*/0);
    const RpuAllGatherSchedule schedule =
        rpu_resolve_all_gather_schedule(h / NUM_CORES, sizeof(c10::Half));
    consume_action_route(
        FmbRouteFamily::COLLECTIVE,
        WALL_ACTION_PRE_W3_ALL_GATHER_SITE,
        static_cast<int64_t>(schedule));
    rpu_launch_all_gather_spm_kernel(
        addr(0, "emb_core0_spm"), addr(0, "residual1"),
        /*n=*/cs, /*chunk_elems=*/h / NUM_CORES,
        /*dwidth=*/sizeof(c10::Half), NUM_CORES, schedule);
}

void WallOssActionStepModel::emit_post_layers_body() {
    const int64_t h = hidden_size(), cs = chunk_size_, np = action_dim_pad_;
    const int64_t bit = ctx().body_iter;   // EXT-unroll iteration (0 for single-step)
    // [Z1] v_t = residual1(final-normed hidden) @ proj_back.T, single-core. proj_back is
    // zero-padded to np rows; Python slices [:, :, :action_dim]. Do NOT re-normalize:
    // the last layer already applied the final RMSNorm in-place to residual1.
    consume_action_route(
        FmbRouteFamily::LINEAR, WALL_ACTION_POST_PROJ_LINEAR_SITE,
        wall_action_linear_route());
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), proj_back_, addr(0, "v_t_core0_spm"),
        /*M=*/cs, /*N=*/np, /*K=*/h, /*partition=*/1, /*num_cores=*/1, /*bias_spm_addr=*/0);
    //   v = model_v * dof_mask + (padding_action - x_initial) * (1 - dof_mask).
    // The two request-dependent tensors use mutable bases so same-signature replays
    // consume the current noise/mask values. Reuse one small SPM scratch sequentially.
    // velocity for committed rows on subsequent ODE steps. Match that observable
    // trajectory exactly; the step-exit overwrite below hard-pins x after every step.

    // [Z2] v_t_core0_spm -> DDR (MUTABLE). loop mode writes the per-iteration trajectory
    // slot v_traj[bit]; single-step writes v_t_buf at offset 0.
    if (loop_mode_) {
        consume_action_route(
            FmbRouteFamily::MUTABLE_DMA,
            WALL_ACTION_POST_V_TRAJ_DMA_SITE,
            wall_action_dma_route(
                WallActionMutableDmaRoute::SPM_COPY_TO_DDR));
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "v_t_core0_spm"), &v_traj_dst_base_,
            /*dst_offset_bytes=*/bit * cs * np * 2, cs * np);
    } else {
        consume_action_route(
            FmbRouteFamily::MUTABLE_DMA,
            WALL_ACTION_POST_V_STEP_DMA_SITE,
            wall_action_dma_route(
                WallActionMutableDmaRoute::SPM_COPY_TO_DDR));
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

    // [Z4] x_t_spm (= x_new) -> DDR (MUTABLE). loop mode writes x_traj[bit] (final x =
    // x_traj[-1]); single-step writes x_out_buf at offset 0.
    if (loop_mode_) {
        consume_action_route(
            FmbRouteFamily::MUTABLE_DMA,
            WALL_ACTION_POST_X_TRAJ_DMA_SITE,
            wall_action_dma_route(
                WallActionMutableDmaRoute::SPM_COPY_TO_DDR));
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "x_t_spm"), &x_traj_dst_base_,
            /*dst_offset_bytes=*/bit * cs * k_comb_pad_ * 2, cs * k_comb_pad_);
    } else {
        consume_action_route(
            FmbRouteFamily::MUTABLE_DMA,
            WALL_ACTION_POST_X_STEP_DMA_SITE,
            wall_action_dma_route(
                WallActionMutableDmaRoute::SPM_COPY_TO_DDR));
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "x_t_spm"), &x_out_dst_base_, /*dst_offset_bytes=*/0, cs * k_comb_pad_);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// step_forward — Task 8
// ─────────────────────────────────────────────────────────────────────────────
at::Tensor WallOssActionStepModel::step_forward(
    const at::Tensor& x_t_rpu, std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches, const at::Tensor& b_step,
    const std::optional<at::Tensor>& te_step,
    const std::optional<at::Tensor>& attention_mask,
    const at::Tensor& position_ids,                 // REQUIRED (mRoPE always on for wall_oss)
    at::Tensor& v_t_buf, at::Tensor& x_out_buf, double dt, int64_t prefix_len,
    const std::optional<at::Tensor>& rope_cos_il,    // partial_mrope interleaved cos/sin (or nullopt)
    const std::optional<at::Tensor>& rope_sin_il,
    at::IntArrayRef planned_stage_descriptor) {
    TORCH_CHECK(action_dim_pad_ > 0, "step_forward before set_action_weights");
    TORCH_CHECK(!planned_stage_descriptor.empty(),
                "Wall action production forward requires its native A6 descriptor");
    validate_planned_action_route_profile(
        planned_stage_descriptor, /*loop_mode=*/false, /*num_steps=*/1,
        b_step.dim() == 1, te_step.has_value());
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
        invalidate_model_state(/*planning_domain_changed=*/false);
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

    // KV headroom (pi05 rpu_pi05_denoise_step_model.cpp:247-260): K layout max_seq =
    // size(1)*size(5).
    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(prefix_len + chunk_size_ <= kc.size(1) * kc.size(5),
            "prefix_len(", prefix_len, ")+H(", chunk_size_, ") exceeds k_cache capacity ",
            kc.size(1) * kc.size(5));
    }

    // Pin the per-step mutable DMA bases (caller owns the flush — wrappers do not flush).
    // Every base pinned below is dereferenced when the graph EXECUTES, i.e. in
    // RpuKernelGraph::end() when the caller's `with cache.capture(sig):` exits —
    // not when this function returns. Keep an owning ref on each (pitfalls.md
    // C-1); the two write destinations matter most, since a re-issued VA means
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
        rope_cos_il, rope_sin_il,
        /*cos_sin_offset=*/-1, /*batch_slot=*/0,
        /*allow_batch_decode=*/false, /*planned_chunk_size=*/0,
        planned_stage_descriptor);
}

// ─────────────────────────────────────────────────────────────────────────────
// denoise_loop_forward — Phase-2 in-graph unroll (Task 3)
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
    at::IntArrayRef planned_stage_descriptor) {
    TORCH_CHECK(action_dim_pad_ > 0, "denoise_loop_forward before set_action_weights");
    TORCH_CHECK(!planned_stage_descriptor.empty(),
                "Wall action production forward requires its native A6 descriptor");
    validate_planned_action_route_profile(
        planned_stage_descriptor, /*loop_mode=*/true, num_steps,
        te_all.has_value(), te_all.has_value());
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
        invalidate_model_state(/*planning_domain_changed=*/false);
    }

    // Bound the unrolled graph below the SDK batch-entry limit. Per-forward setup
    // is shared across bodies; reject oversized schedules before recording.
    TORCH_CHECK(num_steps * 1100 < 65536,
        "denoise unroll (", num_steps, " bodies × ~1035 nodes) would approach the 65536 "
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

    // dt baked into the in-graph Euler at BUILD; identical across the N unrolled iterations.
    TORCH_CHECK(dt > 0.0, "dt must be > 0");
    const c10::Half dt_half = c10::Half(static_cast<float>(dt));
    // dt is baked per Python GraphSignature. Assign it for the active entry so
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
    x_t_ref_ = x0_rpu;  // keepalive — x_t_src_base_ points at x0 here (pitfalls.md C-1)
    x_t_src_base_    = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
    b_step_src_base_ = ::rhino_lkn::RpuGetDevAddr(b_all.data_ptr());
    x_traj_dst_base_ = ::rhino_lkn::RpuGetDevAddr(x_traj.data_ptr());
    v_traj_dst_base_ = ::rhino_lkn::RpuGetDevAddr(v_traj.data_ptr());
    if (have_te_ == 1) {
        te_all_ref_  = te_all.value();   // keepalive
        te_src_base_ = ::rhino_lkn::RpuGetDevAddr(te_all_ref_.data_ptr());
    }

    // AUTO chunk for the 2D-mask single-chunk forward (MR-D: per-handle, see the
    // unrolled sibling above).
    set_chunk_size_override(configured_chunk_size_);

    // ONE forward — run_all_layers loops body_iterations(=num_steps_) internally via the
    // base change; the pre/post hooks fire per iteration with ctx().body_iter advancing.
    (void) CausalDecoderModel::forward(
        emb_stage_, k_caches, v_caches, attention_mask,
        /*position=*/prefix_len, /*is_causal=*/false,
        std::optional<at::Tensor>(position_ids), /*deepstack=*/std::nullopt,
        rope_cos_il, rope_sin_il,
        /*cos_sin_offset=*/-1, /*batch_slot=*/0,
        /*allow_batch_decode=*/false, /*planned_chunk_size=*/0,
        planned_stage_descriptor);
}

}  // namespace v3

// =============================================================================
// Instance registry — ModelHandleRegistry<v3::WallOssActionStepModel>
// =============================================================================
using WallOssActionStepRegistry = ModelHandleRegistry<v3::WallOssActionStepModel>;

std::vector<int64_t> rpu_wall_oss_action_step_planner_cache_identity(int64_t handle) {
    return WallOssActionStepRegistry::get(handle, "rpu_wall_oss_action_step_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_wall_oss_action_step_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    WallOssActionStepRegistry::get(handle, "rpu_wall_oss_action_step_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_wall_oss_action_step_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return WallOssActionStepRegistry::get(handle, "rpu_wall_oss_action_step_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_wall_oss_action_step_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return WallOssActionStepRegistry::get(handle, "rpu_wall_oss_action_step_kvinsert_cost_domain")
        ->kvinsert_cost_domain("wall_oss_action_step", descriptor);
}

std::string rpu_wall_oss_action_step_kvinsert_cost_catalog_sha256(int64_t handle) {
    return WallOssActionStepRegistry::get(
        handle, "rpu_wall_oss_action_step_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

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
    // MR-A: this fused subsystem inherits CausalDecoderModel, so it inherits the
    // deny-by-default certified-envelope gate too. Its sequence length is NOT
    // caller-controlled the way a text prefill is -- it is structurally fixed by
    // the subsystem (action horizon).  Auto keeps the existing zero envelope;
    // an exact public request certifies only that cold-bound ceil16(horizon).
    // Without carrying the exact value here, chunk_within_envelope() rejects the
    // override before the action model can apply its stricter one-chunk check.
    // Declared here because the bound is intrinsic to the subsystem, not an
    // adapter policy.
    // See docs/roadmap/chunk_certified_envelope.md.
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
    const std::optional<at::Tensor>& rope_sin_il,
    at::IntArrayRef planned_stage_descriptor)
{
    return WallOssActionStepRegistry::get(handle, "rpu_wall_oss_action_step_forward")
        ->step_forward(x_t_rpu, k_caches, v_caches, b_step, te_step, attention_mask,
                       position_ids, v_t_buf, x_out_buf, dt, prefix_len,
                       rope_cos_il, rope_sin_il,
                       planned_stage_descriptor);
}

std::vector<int64_t> rpu_wall_oss_action_resolve_stage_domain(
    int64_t handle, int64_t horizon, int64_t prefix_len,
    int64_t mask_kv_len, int64_t requested_chunk_size,
    bool loop_mode, int64_t num_steps, bool b_step_is_bias,
    bool have_te) {
    return WallOssActionStepRegistry::get(
               handle, "rpu_wall_oss_action_resolve_stage_domain")
        ->resolve_action_stage_domain(
            horizon, prefix_len, mask_kv_len, requested_chunk_size,
            loop_mode, num_steps, b_step_is_bias, have_te);
}

int64_t rpu_wall_oss_action_get_resolved_chunk_size(int64_t handle) {
    return WallOssActionStepRegistry::get(
               handle, "rpu_wall_oss_action_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

void rpu_wall_oss_action_enable_execution_reconfigure(int64_t handle) {
    WallOssActionStepRegistry::get(
        handle, "rpu_wall_oss_action_enable_execution_reconfigure")
        ->enable_execution_reconfigure_guard();
}

void rpu_wall_oss_action_stage_chunk_size(
    int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0,
                "Wall action execution transaction token must be positive");
    WallOssActionStepRegistry::get(
        handle, "rpu_wall_oss_action_stage_chunk_size")
        ->stage_execution_chunk_size(
            static_cast<uint64_t>(token), chunk_size);
}

void rpu_wall_oss_action_denoise_loop_forward(
    int64_t handle, const at::Tensor& x0_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const at::Tensor& b_all, const std::optional<at::Tensor>& te_all,
    const std::optional<at::Tensor>& attention_mask, const at::Tensor& position_ids,
    at::Tensor x_traj, at::Tensor v_traj, double dt, int64_t prefix_len, int64_t num_steps,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il,
    at::IntArrayRef planned_stage_descriptor)
{
    WallOssActionStepRegistry::get(handle, "rpu_wall_oss_action_denoise_loop_forward")
        ->denoise_loop_forward(x0_rpu, k_caches, v_caches, b_all, te_all, attention_mask,
                               position_ids, x_traj, v_traj, dt, prefix_len, num_steps,
                               rope_cos_il, rope_sin_il,
                               planned_stage_descriptor);
}
