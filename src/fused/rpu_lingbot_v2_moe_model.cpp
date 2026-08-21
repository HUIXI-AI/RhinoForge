// rpu_lingbot_v2_moe_model.cpp — LingBot2 (V2) sparse-MoE action-expert decoder.
// See rpu_lingbot_v2_moe_model.h for the scope + math contract.
//
// ═══════════════════════════════════════════════════════════════════════════════
// HOW THE STATIC (E, T) ROUTING-WEIGHT MATRIX IS BUILT
// ═══════════════════════════════════════════════════════════════════════════════
// The shipped topk_by_select_fp16 kernel returns exactly k values and uint16
// expert ids with stable original-index ordering on ties.  The backend widens
// those ids and scatters them into a static dense [E,T] mask in SPM:
//
//   1. Transpose the [T,E] router scores to [E,T] (rpu_launch_transpose_nchw_to_nhwc_spm).
//      This is THE enabling trick. With E as the OUTER (slowest) dimension:
//        (a) a per-token reduction over all E experts becomes a sequence of
//            contiguous buffer halvings -> log2(32) = 5 flat eltwise kernels,
//            instead of 31 strided ones; and
//        (b) expert e's routing-weight column w[:,e] becomes a CONTIGUOUS [T]
//            slice at byte offset e*Tp*2 — which is exactly the [N,1] operand
//            shape that rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel wants.
//   2. ids[t,:] = strict_top_k(scores_for_choice[t,:]) [T,k]
//      with the deterministic secondary key expert_index ascending.
//   3. mask = scatter(ids)                    [E,T], exactly k ones/token
//      w    = routing_scores * mask           (weights come from the UNBIASED scores)
//      w   /= tree_sum(w)                       (positive sigmoid => nonzero sum)
//      routed_scaling (4.0) is applied once by the scalar SPM multiply below.
//   Data-dependent VALUES are fine; data-dependent SHAPES are not. Nothing here
//   has a data-dependent shape, so the graph stays capturable/replayable.
//
// ═══════════════════════════════════════════════════════════════════════════════
// KNOWN DEVIATIONS / RESIDUAL RISKS — MUST be validated on-board before trusting
// ═══════════════════════════════════════════════════════════════════════════════
// [D1] ROUTER PRECISION. The reference path disables autocast around the router
//      because bf16/fp16 logits flip top-k on
//      near-equal scores; the hard gate is "must not change top-k indices".
//      The RPU SPM datapath is fp16 (every rpu_launch_*_spm_* kernel takes
//      c10::Half; there is no fp32 SPM eltwise/transpose). The router therefore
//      runs with fp32 ACCUMULATE (rpu_launch_linear_spm_to_spm_kernel = acc32,
//      NOT the acc16 variant used everywhere else) but fp16 STORAGE of the
//      logits/scores. This is the highest-precision router expressible with the
//      available kernels, and it is a REAL DEVIATION from the reference.
// [D2] TIE HANDLING — CLOSED. topk_by_select_fp16 chooses exactly k entries;
//      equal choice scores retain ascending expert-index order.  The dense SPM
//      scatter therefore has exactly k ones for every token, including an exact
//      4th/5th boundary tie or an all-equal row.
// [D3] all_reduce_sum_residual is invoked with residual == output ("moe_acc") to
//      accumulate experts in place. Elementwise-safe by inspection (out[i] =
//      sum_cores(in[i]) + residual[i], same index), but no existing model does
//      this; worth a board check.
// [D2] has an isolated host contract and an on-board conformance gate. [D1]
// remains the explicitly recorded precision limitation.
#include "rpu_lingbot_v2_moe_model.h"
#include "core/rpu_lingbot2_ring_collective.h"
#include "model_handle_registry.h"
#include "rpu_kernel_decls.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"
#include "core/rpu_spm_allocator.h"

#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <cmath>
#include <cstdint>
#include <cstdio>    // std::printf (debug router banner)
#include <cstdlib>   // std::getenv (debug router flag)
#include <cstring>
#include <limits>
#include <vector>

namespace v3 {

LingbotV2MoeExpertModel::LingbotV2MoeExpertModel()  = default;
LingbotV2MoeExpertModel::~LingbotV2MoeExpertModel() = default;

lingbot_v2_moe_internal::ExactSuffixProfileKind
LingbotV2MoeExpertModel::validate_exact_suffix_profile(
    int64_t prefix_len) const {
    using ProfileKind =
        lingbot_v2_moe_internal::ExactSuffixProfileKind;
    constexpr int64_t kPrefixLen = 225;
    constexpr int64_t kSuffixLen = 51;
    constexpr int64_t kExpectedLayers = 36;
    constexpr int64_t kPackedRows = 16'384;

    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "LingBot2 expert profile admission must run outside Graph capture");
    TORCH_CHECK(prefix_len == kPrefixLen,
                "LingBot2 expert profile requires prefix_len=225, got ",
                prefix_len);
    TORCH_CHECK(
        num_layers() == 36 && hidden_size() == 768 &&
            intermediate_size() == 768 && num_q_heads() == 32 &&
            num_kv_heads() == 8 && head_dim() == 128 &&
            has_qkv_bias_ && !has_qk_norm_ && use_silu_,
        "LingBot2 expert profile requires exact Qwen2.5 "
        "L36/H768/NQ32/NKV8/HD128/shared-I768 with QKV bias");
    TORCH_CHECK(
        num_experts_ == 32 && top_k_ == 4 && routed_inter_ == 512 &&
            chunk_size_ == kSuffixLen &&
            routed_scaling_ == c10::Half(4.0f),
        "LingBot2 expert profile requires E32/top4/routed-I512/"
        "chunk51/scaling4");
    TORCH_CHECK(
        denoise_unroll_ && num_steps_ == 10 && state_dim_ == 55 &&
            action_dim_ == 55 && state_dim_pad_ == 64 &&
            action_dim_pad_ == 64 && adarms_ && adarms_mutable_ &&
            adarms_unroll_ && !denoise_unroll_active_,
        "LingBot2 expert profile requires the exact ten-step FP16 "
        "denoise-unroll and indexed AdaRMS schedule");
    TORCH_CHECK(
        fp16_top4_ && !debug_dense_soft_router_ && !nvfp4_ && !debug_stages_ &&
            !debug_down_acc16_ && !debug_capture_gate_ &&
            !debug_capture_l0_ && !debug_capture_innorm_ &&
            !debug_cap_resid_ && !debug_routed_only_ &&
            !debug_dump_router_h_ && get_chunk_size_override() == 0,
        "LingBot2 expert profile rejects debug/chunk-override modes");

    const bool dense_fp16 =
        !grouped_experts_ && !packed_w8a16_ && packed_gate_.empty() &&
        packed_up_.empty() && packed_down_.empty() &&
        packed_gate_s_.empty() && packed_up_s_.empty() &&
        packed_down_s_.empty();

    TORCH_CHECK(
        static_cast<int64_t>(layer_weights_.size()) == kExpectedLayers,
        "LingBot2 exact suffix requires complete 36-layer base weights");
    auto check_base_role = [](const at::Tensor& weight,
                              const at::Tensor& scale,
                              bool quantized,
                              const char* name) {
        const at::ScalarType expected_dtype =
            quantized ? at::kChar : at::kHalf;
        TORCH_CHECK(
            weight.defined() && weight.scalar_type() == expected_dtype &&
                weight.device().type() == at::kPrivateUse1 &&
                weight.is_contiguous() && weight.dim() == 2,
            name, quantized
                ? " must be contiguous W8 kChar RPU weight"
                : " must be contiguous dense FP16 RPU weight");
        if (!quantized) {
            TORCH_CHECK(
                !scale.defined(), name,
                " dense FP16 profile rejects a bound quantization scale");
            return;
        }
        TORCH_CHECK(
            scale.defined() && scale.scalar_type() == at::kHalf &&
                scale.device().type() == at::kPrivateUse1 &&
                scale.is_contiguous() && scale.dim() == 1,
            name, " W8 scale must be contiguous per-channel FP16 RPU");
    };
    auto check_base_layers = [&](bool quantized) {
        for (int64_t layer = 0; layer < kExpectedLayers; ++layer) {
            const auto& weights = layer_weights_[layer];
            check_base_role(weights.q_w, weights.q_ws, quantized, "base_q");
            check_base_role(weights.k_w, weights.k_ws, quantized, "base_k");
            check_base_role(weights.v_w, weights.v_ws, quantized, "base_v");
            check_base_role(weights.o_w, weights.o_ws, quantized, "base_o");
            check_base_role(
                weights.gate_w, weights.gate_ws, quantized,
                "base_shared_gate");
            check_base_role(
                weights.up_w, weights.up_ws, quantized,
                "base_shared_up");
            check_base_role(
                weights.down_w, weights.down_ws, quantized,
                "base_shared_down");
        }
    };
    if (dense_fp16) {
        TORCH_CHECK(
            !denoise_twostage_,
            "LingBot2 DenseFp16 exact suffix keeps denoise two-stage off");
        check_base_layers(/*quantized=*/false);
        return ProfileKind::DenseFp16;
    }

    const bool grouped_w8a16 =
        grouped_experts_ && packed_w8a16_ &&
        !packed_gate_.empty() && packed_gate_[0].scalar_type() == at::kChar;
    const bool grouped_w4a16 =
        grouped_experts_ && packed_w8a16_ &&
        !packed_gate_.empty() && packed_gate_[0].scalar_type() == at::kByte;
    TORCH_CHECK(
        grouped_w8a16 != grouped_w4a16,
        "LingBot2 exact suffix accepts only DenseFp16, GroupedW8A16, "
        "or GroupedW4A16");
    const ProfileKind grouped_kind = grouped_w4a16
        ? ProfileKind::GroupedW4A16
        : ProfileKind::GroupedW8A16;
    check_base_layers(/*quantized=*/true);
    TORCH_CHECK(
        packed_gate_.size() == kExpectedLayers &&
            packed_up_.size() == kExpectedLayers &&
            packed_down_.size() == kExpectedLayers &&
            packed_gate_s_.size() == kExpectedLayers &&
            packed_up_s_.size() == kExpectedLayers &&
            packed_down_s_.size() == kExpectedLayers,
        "LingBot2 grouped quantized profile requires complete 36-layer packed weights "
        "and scales; partial binding is rejected");

    auto check_pack = [grouped_w4a16](
                          const at::Tensor& tensor, int64_t rows,
                          int64_t logical_cols, const char* name) {
        const at::ScalarType expected_dtype =
            grouped_w4a16 ? at::kByte : at::kChar;
        const int64_t storage_cols =
            grouped_w4a16 ? logical_cols / 2 : logical_cols;
        TORCH_CHECK(
            tensor.defined() && tensor.scalar_type() == expected_dtype,
            name, grouped_w4a16
                ? " must use packed int4 kByte storage"
                : " must use signed int8 kChar storage");
        TORCH_CHECK(
            tensor.device().type() == at::kPrivateUse1 && tensor.dim() == 2 &&
                tensor.is_contiguous() && tensor.size(0) == rows &&
                tensor.size(1) == storage_cols,
            name, " must be exact contiguous-layout RPU [", rows, ",",
            storage_cols,
            "], got ", tensor.sizes());
    };
    auto check_scale = [grouped_w4a16](
                           const at::Tensor& tensor, int64_t channels,
                           int64_t logical_k, int partition,
                           const char* name) {
        TORCH_CHECK(
            tensor.defined() && tensor.scalar_type() == at::kHalf &&
                tensor.device().type() == at::kPrivateUse1 &&
                tensor.is_contiguous(),
            name, " must be contiguous FP16 RPU");
        if (grouped_w4a16) {
            TORCH_CHECK(
                tensor.dim() == 2 &&
                    (tensor.size(0) == 32 || tensor.size(0) == 64 ||
                     tensor.size(0) == 128),
                name, " pgrp dim 0 must carry group_size 32/64/128, got ",
                tensor.sizes());
            const int64_t group_size = tensor.size(0);
            TORCH_CHECK(
                (partition == 1 && channels % NUM_CORES == 0) ||
                    (partition == 0 && logical_k % NUM_CORES == 0),
                name, " has dimensions incompatible with its partition");
            const int64_t local_k =
                partition == 1 ? logical_k : logical_k / NUM_CORES;
            const int64_t local_n =
                partition == 1 ? channels / NUM_CORES : channels;
            TORCH_CHECK(local_k % group_size == 0,
                        name, " local K must be divisible by group_size");
            const int64_t local_g = local_k / group_size;
            const int64_t expected = ((local_g + 3) / 4) *
                                     ((local_n + 63) / 64) * NUM_CORES * 4 * 64;
            TORCH_CHECK(
                tensor.numel() == expected,
                name, " controller-striped pgrp payload has ", tensor.numel(),
                " elements, expected ", expected);
            constexpr uint64_t kScaleAlignment = NUM_CORES * 512;
            TORCH_CHECK(
                ::rhino_lkn::RpuGetDevAddr(tensor.data_ptr<c10::Half>()) %
                        kScaleAlignment ==
                    0,
                name, " pgrp payload must be ", kScaleAlignment,
                "-byte aligned");
        } else {
            TORCH_CHECK(
                tensor.dim() == 1 && tensor.size(0) == channels,
                name, " must be per-channel FP16 RPU [", channels,
                "], got ", tensor.sizes());
        }
    };
    for (int64_t layer = 0; layer < kExpectedLayers; ++layer) {
        check_pack(packed_gate_[layer], kPackedRows, hidden_size(),
                   "packed_gate");
        check_pack(packed_up_[layer], kPackedRows, hidden_size(),
                   "packed_up");
        check_pack(packed_down_[layer], hidden_size(), kPackedRows,
                   "packed_down");
        check_scale(
            packed_gate_s_[layer], kPackedRows, hidden_size(), /*partition=*/1,
            "packed_gate_scale");
        check_scale(
            packed_up_s_[layer], kPackedRows, hidden_size(), /*partition=*/1,
            "packed_up_scale");
        check_scale(packed_down_s_[layer], hidden_size(), kPackedRows,
                    /*partition=*/0,
                    "packed_down_scale");
    }
    const char* row_chunk_env = std::getenv("RPU_L2_RCHUNK");
    TORCH_CHECK(
        row_chunk_env != nullptr &&
            std::strcmp(row_chunk_env, "1632") == 0,
        "LingBot2 grouped W8A16/W4A16 exact suffix requires "
        "RPU_L2_RCHUNK=1632");
    TORCH_CHECK(
        denoise_twostage_,
        "LingBot2 grouped W8A16/W4A16 exact suffix requires product-scoped "
        "denoise two-stage");
    return grouped_kind;
}

uint64_t LingbotV2MoeExpertModel::exact_suffix_profile_hash(
    lingbot_v2_moe_internal::ExactSuffixProfileKind kind) const {
    uint64_t hash = UINT64_C(1469598103934665603);
    auto mix = [&hash](uint64_t value) {
        hash ^= value;
        hash *= UINT64_C(1099511628211);
    };
    mix(UINT64_C(0x4c42324d4f455031));
    mix(static_cast<uint8_t>(kind));
    mix(static_cast<uint64_t>(num_layers()));
    mix(static_cast<uint64_t>(hidden_size()));
    mix(static_cast<uint64_t>(num_experts_));
    mix(static_cast<uint64_t>(top_k_));
    mix(static_cast<uint64_t>(routed_inter_));
    mix(static_cast<uint64_t>(chunk_size_));
    mix(static_cast<uint64_t>(num_steps_));
    mix(static_cast<uint64_t>(denoise_twostage_));
    TORCH_INTERNAL_ASSERT(hash != 0);
    return hash;
}

lingbot_v2_moe_internal::ExactSuffixProfileDescriptor
LingbotV2MoeExpertModel::describe_exact_suffix_spm_profile(
    int64_t prefix_len) {
    const auto kind = validate_exact_suffix_profile(prefix_len);
    LayoutContext layout;
    layout.chunk_size = 51;
    layout.kv_insert_chunk_size = 0;
    layout.max_kv_seq_len = prefix_len + 51;
    layout.num_layers = 36;
    layout.use_attn_mask = true;
    layout.is_causal = false;

    bool dry_prepared = false;
    auto cancel_dry_on_failure = c10::make_scope_exit([&] {
        if (dry_prepared) {
            cancel_spm_pipeline_component_for_cpu_contract();
        }
    });
    const SpmPipelineComponentLayout dry_layout =
        prepare_spm_pipeline_component_for_cpu_contract(layout);
    dry_prepared = true;
    const auto descriptor =
        lingbot_v2_moe_internal::ExactSuffixProfileDescriptor{
            kind, exact_suffix_profile_hash(kind), dry_layout.layout_hash,
            dry_layout.temporary_bytes};
    TORCH_CHECK(descriptor.profile_hash != 0,
                "LingBot2 expert profile hash must be nonzero");
    TORCH_CHECK(descriptor.layout_hash != 0,
                "LingBot2 expert layout hash must be nonzero");
    TORCH_CHECK(descriptor.temporary_bytes > 0,
                "LingBot2 expert temporary layout must be nonempty");
    cancel_spm_pipeline_component_for_cpu_contract();
    dry_prepared = false;
    cancel_dry_on_failure.release();
    return descriptor;
}

SpmPipelineComponentLayout
LingbotV2MoeExpertModel::prime_exact_suffix_spm_layout(
    int64_t prefix_len,
    const lingbot_v2_moe_internal::ExactSuffixProfileDescriptor& expected) {
    constexpr int64_t kPrefixLen = 225;
    constexpr int64_t kSuffixLen = 51;
    constexpr int64_t kCertifiedChunkCeiling = 64;
    constexpr size_t kExpectedPersistentBytes = 318'976;

    const auto kind = validate_exact_suffix_profile(prefix_len);
    TORCH_CHECK(
        expected.kind == kind && expected.profile_hash != 0 &&
            expected.profile_hash == exact_suffix_profile_hash(kind) &&
            expected.layout_hash != 0 && expected.temporary_bytes > 0,
        "LingBot2 expert profile changed after native preflight admission");

    set_chunk_envelope(
        /*max_kv_len=*/kPrefixLen + kSuffixLen,
        /*chunk=*/kCertifiedChunkCeiling);

    LayoutContext layout;
    layout.chunk_size = kSuffixLen;
    layout.kv_insert_chunk_size = 0;
    layout.max_kv_seq_len = kPrefixLen + kSuffixLen;
    layout.num_layers = 36;
    layout.use_attn_mask = true;
    layout.is_causal = false;

    const std::vector<BufferDecl> declarations = declare_buffers(layout);
    size_t persistent_bytes = 0;
    for (const BufferDecl& declaration : declarations) {
        if (declaration.alias_of != nullptr) continue;
        const size_t aligned =
            (static_cast<size_t>(declaration.size) + 255) & ~size_t{255};
        if (declaration.storage == StorageClass::Persistent) {
            persistent_bytes += aligned;
        } else if (declaration.storage == StorageClass::PersistentPerLayer) {
            TORCH_CHECK(declaration.per_layer >= 0,
                        "LingBot2 expert SPM prime found negative per_layer");
            persistent_bytes +=
                aligned * static_cast<size_t>(declaration.per_layer);
        }
    }
    TORCH_CHECK(
        persistent_bytes == kExpectedPersistentBytes,
        "LingBot2 expert persistent declaration drifted; expected ",
        kExpectedPersistentBytes, " bytes/core, got ", persistent_bytes);
    TORCH_CHECK(
        SPM_ALLOC.persistent_used() == 0,
        "LingBot2 expert SPM prime requires no ordinary persistent arena");

    const size_t super_before = SPM_ALLOC.super_persistent_used();
    SpmPipelineComponentLayout result =
        prepare_spm_pipeline_component(layout);
    TORCH_CHECK(
        result.temporary_bytes == expected.temporary_bytes &&
            result.layout_hash == expected.layout_hash,
        "LingBot2 expert live SPM layout drifted from native preflight; "
        "temporary expected/got=", expected.temporary_bytes, "/",
        result.temporary_bytes, ", hash expected/got=", expected.layout_hash,
        "/", result.layout_hash);
    const size_t super_after = SPM_ALLOC.super_persistent_used();
    TORCH_CHECK(
        SPM_ALLOC.persistent_used() == 0 &&
            super_after >= super_before &&
            super_after - super_before == kExpectedPersistentBytes,
        "LingBot2 expert cold prime must add exactly ",
        kExpectedPersistentBytes,
        " super-persistent bytes/core; before/after=", super_before, "/",
        super_after);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Controlled 10-step denoise configuration
// ─────────────────────────────────────────────────────────────────────────────
void LingbotV2MoeExpertModel::set_denoise_weights(
    const at::Tensor& state_w, const at::Tensor& state_b,
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias,
    const at::Tensor& time_all,
    const at::Tensor& adarms_is, const at::Tensor& adarms_ish,
    const at::Tensor& adarms_ps, const at::Tensor& adarms_psh,
    int64_t state_dim, int64_t action_dim, int64_t state_dim_pad,
    int64_t action_dim_pad, int64_t num_steps) {
    TORCH_CHECK(num_layers() > 0 && num_experts_ > 0,
                "set_denoise_weights must follow set_weights/set_moe_weights");
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "set_denoise_weights must run before the first forward");
    TORCH_CHECK(!denoise_unroll_, "set_denoise_weights may only be called once");
    TORCH_CHECK(state_dim > 0 && action_dim > 0 && num_steps > 0,
                "state_dim/action_dim/num_steps must be positive");
    TORCH_CHECK(state_dim <= state_dim_pad && action_dim <= action_dim_pad,
                "logical state/action dims must fit their padded dims");
    TORCH_CHECK(state_dim_pad % 16 == 0 && action_dim_pad % 16 == 0,
                "state_dim_pad/action_dim_pad must be multiples of 16");
    TORCH_CHECK(chunk_size_ > 1,
                "set_denoise_weights requires state+action suffix rows");

    const int64_t h = hidden_size();
    auto chk = [](const at::Tensor& t, int64_t n, const char* name) {
        TORCH_CHECK(t.defined() && t.scalar_type() == at::kHalf
                 && t.device().type() == at::kPrivateUse1 && t.is_contiguous()
                 && t.numel() == n,
                    name, " must be contiguous fp16 RPU with numel=", n);
    };
    chk(state_w, h * state_dim_pad, "state_w");
    chk(state_b, h, "state_b");
    chk(wc, h * action_dim_pad, "wc");
    chk(mo, h * h, "mo");
    chk(mo_bias, h, "mo_bias");
    chk(op, action_dim_pad * h, "op");
    chk(op_bias, action_dim_pad, "op_bias");
    chk(time_all, num_steps * h, "time_all");
    TORCH_CHECK(time_all.dim() == 2 && time_all.size(0) == num_steps
             && time_all.size(1) == h,
                "time_all must be [num_steps, hidden]");

    state_w_ = state_w;
    state_b_ = state_b;
    wc_ = wc;
    mo_ = mo;
    mo_bias_ = mo_bias;
    op_ = op;
    op_bias_ = op_bias;
    time_all_ = time_all;
    state_dim_ = state_dim;
    action_dim_ = action_dim;
    state_dim_pad_ = state_dim_pad;
    action_dim_pad_ = action_dim_pad;
    num_steps_ = num_steps;
    const char* denoise_twostage_env =
        std::getenv("RPU_LINGBOT2_DENOISE_TWOSTAGE");
    denoise_twostage_ = denoise_twostage_env != nullptr &&
        std::strcmp(denoise_twostage_env, "1") == 0;
    denoise_stage_ = at::empty(
        {1, chunk_size_, h},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    denoise_unroll_ = true;

    // This is mutually exclusive with bind_adarms_schedule(); the Python
    // controlled profile branches before either setter is called.
    set_adarms_unroll(adarms_is, adarms_ish, adarms_ps, adarms_psh);
    invalidate_model_state();  // Must remain the last statement.
}

// ─────────────────────────────────────────────────────────────────────────────
// set_moe_weights
// ─────────────────────────────────────────────────────────────────────────────
void LingbotV2MoeExpertModel::set_moe_weights(
    at::TensorList router_gate_w, at::TensorList router_bias,
    at::TensorList expert_gate_w, at::TensorList expert_up_w,
    at::TensorList expert_down_w,
    int64_t num_experts, int64_t top_k, int64_t routed_inter,
    double routed_scaling, int64_t chunk_size) {
    const int64_t nl = num_layers();
    const int64_t h  = hidden_size();
    TORCH_CHECK(nl > 0, "set_moe_weights must follow CausalDecoderModel::set_weights");
    TORCH_CHECK(num_experts > 0 && top_k > 0 && routed_inter > 0 && chunk_size > 0,
                "num_experts/top_k/routed_inter/chunk_size must be positive");
    // emit_tree_reduce halves the expert dimension log2(E) times.
    TORCH_CHECK((num_experts & (num_experts - 1)) == 0,
                "num_experts(", num_experts, ") must be a power of 2 (tree reduction)");
    TORCH_CHECK(top_k <= num_experts, "top_k(", top_k, ") > num_experts(", num_experts, ")");
    TORCH_CHECK(num_experts >= 16 && num_experts % 16 == 0,
                "num_experts(", num_experts,
                ") must be v16-aligned for backend MoE selection");
    TORCH_CHECK(Align(chunk_size, (int64_t)16) <= 128,
                "aligned chunk_size must be <= 128 for backend MoE selection");
    // Pitfall H: N is the FULL output dim and the kernel divides it by num_cores.
    TORCH_CHECK(routed_inter % NUM_CORES == 0,
                "routed_inter(", routed_inter, ") must be divisible by NUM_CORES=", NUM_CORES);
    // THD_VECTOR_SIZE=16 — the Nx1_NxC per-row broadcast needs c % 16 == 0.
    TORCH_CHECK((routed_inter / NUM_CORES) % 16 == 0,
                "routed_inter/NUM_CORES(", routed_inter / NUM_CORES, ") must be a multiple of 16");
    TORCH_CHECK(use_silu_, "LingbotV2MoeExpertModel requires SwiGLU (use_silu=true)");
    TORCH_CHECK((int64_t)router_gate_w.size() == nl && (int64_t)router_bias.size() == nl,
                "router_gate_w/router_bias must have size num_layers=", nl);
    TORCH_CHECK(num_experts <= std::numeric_limits<int64_t>::max() / nl,
                "num_layers*num_experts overflows int64");
    const int64_t expert_tensor_count = nl * num_experts;
    TORCH_CHECK((int64_t)expert_gate_w.size() == expert_tensor_count
             && (int64_t)expert_up_w.size()   == expert_tensor_count
             && (int64_t)expert_down_w.size() == expert_tensor_count,
                "expert_* lists must have size num_layers*num_experts=",
                expert_tensor_count);

    auto check_exact = [](const at::Tensor& t,
                          at::IntArrayRef expected_shape,
                          const char* name,
                          int64_t index) {
        int64_t expected_numel = 1;
        for (const int64_t dim : expected_shape) {
            TORCH_CHECK(
                dim > 0 &&
                    expected_numel <=
                        std::numeric_limits<int64_t>::max() / dim,
                name, "[", index, "] expected shape overflows int64: ",
                expected_shape);
            expected_numel *= dim;
        }
        TORCH_CHECK(t.defined(), name, "[", index, "] is undefined");
        TORCH_CHECK(t.scalar_type() == at::kHalf &&
                        t.device().type() == at::kPrivateUse1 &&
                        t.is_contiguous(),
                    name, "[", index,
                    "] must be a contiguous fp16 RPU tensor");
        TORCH_CHECK(t.sizes() == expected_shape &&
                        t.numel() == expected_numel,
                    name, "[", index, "] must have shape ", expected_shape,
                    " and numel=", expected_numel, "; got shape ", t.sizes(),
                    " and numel=", t.numel());
    };
    for (int64_t i = 0; i < nl; ++i) {
        check_exact(router_gate_w[i], {num_experts, h},
                    "router_gate_w", i);
        check_exact(router_bias[i], {num_experts}, "router_bias", i);
    }
    for (int64_t i = 0; i < expert_tensor_count; ++i) {
        check_exact(expert_gate_w[i], {routed_inter, h},
                    "expert_gate_w", i);
        check_exact(expert_up_w[i], {routed_inter, h},
                    "expert_up_w", i);
        check_exact(expert_down_w[i], {h, routed_inter},
                    "expert_down_w", i);
    }

    router_gate_.assign(router_gate_w.begin(), router_gate_w.end());
    router_bias_.assign(router_bias.begin(), router_bias.end());
    expert_gate_.assign(expert_gate_w.begin(), expert_gate_w.end());
    expert_up_.assign(expert_up_w.begin(), expert_up_w.end());
    expert_down_.assign(expert_down_w.begin(), expert_down_w.end());
    num_experts_    = num_experts;
    top_k_          = top_k;
    routed_inter_   = routed_inter;
    routed_scaling_ = c10::Half(static_cast<float>(routed_scaling));
    std::printf("[lingbot2] MOE set_weights: routed_scaling=%.4f top_k=%ld num_experts=%ld routed_inter=%ld\n",
                (double)(float)routed_scaling_, (long)top_k_, (long)num_experts_, (long)routed_inter_);
    std::fflush(stdout);
    chunk_size_     = chunk_size;
    // The transpose kernel requires C (= the token dim) to be v16-aligned.
    tp_rows_        = Align(chunk_size, (int64_t)16);
    (void)h;

    // Pad rows [chunk_size, tp_rows) of the router score buffer are filled with
    // 1.0 from this keepalive. Pad lanes are inert anyway (every router op is
    // either lane-local or reduces over E, never over T, so a pad lane can never
    // contaminate a real token), but filling keeps NaN out of the buffer.
    //
    // Why 1.0 and not 0.0: a zeroed pad row makes the pad column's tree-sum 0, so
    // scoresT/sum = 0/0 = NaN, which then has to be patched out by writing 1.0
    // into the [1,Tp] denominator's pad lanes. That patch is (Tp-seq_len)*2 bytes
    // -- 16-byte aligned ONLY when (Tp-seq_len) % 8 == 0, and it is not: the real
    // suffix is 51 rows and Tp is 64, so the patch was 26 bytes and
    // ddr_broadcast_spm_dma rejects that size. Seeding the pad rows
    // with 1.0 instead makes each pad column sum to E and divide to a finite 1/E,
    // so no denominator patch is needed at all. This write is
    // (Tp-seq_len)*E*2 = (Tp-seq_len)*64 bytes -- aligned for EVERY seq_len.
    //
    // Real tokens are bit-identical either way: the transpose is a pure
    // permutation and the tree-sum reduces each column independently, so pad rows
    // only ever reach pad columns. Those columns are never read -- the combine is
    // n=seq_len (emit_moe_mlp). Pad weights are therefore 1/E, not 0.
    ones_pad_ = at::ones({tp_rows_ * num_experts_},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    // topk_by_select carries a uint16 value beside every key.  Pre-encode the
    // final flattened [E,Tp] scatter destination (expert*Tp + token), avoiding
    // a variable-index [k,Tp] scatter shape that the shipped kernel does not
    // support. PyTorch FP16 is only the two-byte storage owner for DMA.
    at::Tensor expert_ids_cpu = at::empty(
        {tp_rows_, num_experts_},
        at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
    auto* expert_ids_raw = reinterpret_cast<uint16_t*>(
        expert_ids_cpu.data_ptr<c10::Half>());
    for (int64_t t = 0; t < tp_rows_; ++t) {
        for (int64_t e = 0; e < num_experts_; ++e) {
            expert_ids_raw[t * num_experts_ + e] =
                static_cast<uint16_t>(e * tp_rows_ + t);
        }
    }
    expert_scatter_ids_ = expert_ids_cpu.to(at::kPrivateUse1).contiguous();
    // Per-layer relay slot: the router runs on core 0 only (the transpose kernel
    // is core-0 only — src/ops/rpu_transpose.cpp enqueues with {0}), but the
    // per-expert Nx1_NxC weight broadcast needs the [E,Tp] weights on ALL 8
    // cores. One row PER LAYER (not one shared row) so the 36 per-layer
    // core0->DDR->all-cores relays inside a single graph have no WAR hazard.
    rw_stage_ = at::zeros({nl, num_experts_, tp_rows_},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    rpu_ddr_flush_force_sized(
        ones_pad_.data_ptr<c10::Half>(), ones_pad_.nbytes());
    rpu_ddr_flush_force_sized(
        expert_scatter_ids_.data_ptr<c10::Half>(),
        expert_scatter_ids_.nbytes());

    // ── DEBUG dense-soft router opt-in. Default OFF; the fp16 top-4 path below
    //    is the normal router. A strict hard-fail remains only when both paths
    //    are explicitly disabled.
    {
        const char* e = std::getenv("RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER");
        debug_dense_soft_router_ = (e != nullptr && e[0] == '1' && e[1] == '\0');
    }

    // ── strict fp16 top-4 router — default.  topk_by_select_fp16 returns exactly
    //    four ids; uint16->int32 + SPM scatter creates a dense static mask.
    //
    //    WHY IT IS THE DEFAULT: dense-soft bypasses top-k and activates all 32
    //    experts, while top-4 is the model's routing contract.
    //
    //    RPU_LINGBOT2_FP16_TOP4:
    //      unset -> ON, unless dense-soft was explicitly requested (so the pre-existing
    //               DEBUG_DENSE_SOFT_ROUTER=1 scripts keep their meaning)
    //      "1"   -> ON, and WINS over dense-soft so explicit top-4 selection
    //               cannot silently become a dense-soft run.
    //      "0"   -> OFF. With dense-soft also unset this leaves emit_router_select on its strict
    //               TORCH_CHECK(false) hard-fail, which is the intended way to ask for the
    //               (non-existent) fp32 router and find out loudly.
    {
        const char* e = std::getenv("RPU_LINGBOT2_FP16_TOP4");
        const bool explicit_on  = (e != nullptr && e[0] == '1' && e[1] == '\0');
        const bool explicit_off = (e != nullptr && e[0] == '0' && e[1] == '\0');
        fp16_top4_ = explicit_on || (!explicit_off && !debug_dense_soft_router_);
    }

    // ── DEBUG router-input (residual1) dump opt-in. SEPARATE env var; default OFF.
    //    When OFF nothing is allocated and emit_router_select emits no extra DMA, so the
    //    graph is byte-identical to a build without this feature.
    {
        const char* e = std::getenv("RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H");
        debug_dump_router_h_ = (e != nullptr && e[0] == '1' && e[1] == '\0');
    }
    if (debug_dump_router_h_) {
        // Fresh dedicated tensor, one slot PER LAYER => no layer overwrites another and
        // no existing DDR buffer is reused.
        h_stage_ = at::zeros({nl, tp_rows_, hidden_size()},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        std::printf("[lingbot2] DEBUG_DUMP_ROUTER_H=1 -> capturing router input residual1 [%ld, %ld, %ld]\n", (long)nl, (long)tp_rows_, (long)hidden_size());
    }
    // ── DEBUG all-layer residual-stream capture (layer_in + attn_resid). SEPARATE env var,
    //    default OFF => nothing allocated, no DMA emitted, graph byte-identical.
    {
        const char* e = std::getenv("RPU_L2_CAP_RESID");
        debug_cap_resid_ = (e != nullptr && e[0] == '1' && e[1] == '\0');
    }
    if (debug_cap_resid_) {
        lin_stage_ = at::zeros({nl, tp_rows_, hidden_size()},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        ar_stage_ = at::zeros({nl, tp_rows_, hidden_size()},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        std::printf("[lingbot2] RPU_L2_CAP_RESID=1 -> capturing layer_in + attn_resid [%ld, %ld, %ld]\n", (long)nl, (long)tp_rows_, (long)hidden_size());
    }
    // Print the active router once per set_weights so the selected path is explicit.
    if (fp16_top4_) {
        std::printf("LINGBOT2_FP16_TOP4_ROUTER\n");
        std::printf("STRICT_TOPK=%ld\n", (long)top_k_);
        std::printf("MASK_APPLIED_VIA_MOE_RWBC\n");
        std::fflush(stdout);
    } else if (debug_dense_soft_router_) {
        std::printf("DEBUG_FP16_DENSE_SOFT_ROUTER\n");
        std::printf("NO_TOPK\n");
        std::printf("ALL_32_EXPERTS_ACTIVE\n");
        std::printf("ACCURACY_UNVALIDATED\n");
        std::fflush(stdout);
    }
    {
        const char* e = std::getenv("RPU_L2_CAPTURE_L0");
        debug_capture_l0_ = (e != nullptr && e[0] == '1' && e[1] == '\0');
        const char* ero = std::getenv("RPU_L2_ROUTED_ONLY");
        debug_routed_only_ = (ero != nullptr && ero[0] == '1' && ero[1] == '\0');
        const char* est = std::getenv("RPU_L2_STAGES");
        debug_stages_ = (est != nullptr && est[0] == '1' && est[1] == '\0');
        const char* ea = std::getenv("RPU_L2_DOWN_ACC16");
        debug_down_acc16_ = (ea != nullptr && ea[0] == '1' && ea[1] == '\0');
        if (debug_stages_) {
            int64_t gw = num_experts_ * (routed_inter_ / NUM_CORES);
            scap_ = at::zeros({tp_rows_, gw}, at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
            ccap_ = at::zeros({tp_rows_, gw}, at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        }
        if (debug_capture_l0_)
            l0_out_ = at::zeros({tp_rows_, hidden_size()},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        // Opt-in capture of layer 0's post-AdaRMS activation. Same shape/dtype/device as
        // l0_out_; allocated only when asked for, so the default build is unchanged.
        const char* ein = std::getenv("RPU_L2_CAPTURE_INNORM");
        debug_capture_innorm_ = (ein != nullptr && ein[0] == '1' && ein[1] == '\0');
        if (debug_capture_innorm_)
            innorm_ = at::zeros({tp_rows_, hidden_size()},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        const char* eg = std::getenv("RPU_L2_CAPTURE_GATE");
        debug_capture_gate_ = (eg != nullptr && eg[0] == '1' && eg[1] == '\0');
        if (debug_capture_gate_) {
            gcap_ = at::zeros({tp_rows_, num_experts_ * (routed_inter_ / NUM_CORES)},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
            acap_ = at::zeros({tp_rows_, hidden_size()},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        }
    }
    invalidate_model_state();   // Must remain the last statement.
}

// ─────────────────────────────────────────────────────────────────────────────
// declare_buffers
//
// Pure function of LayoutContext + chunk-independent model config. The layout is
// hashed from LayoutContext only, so no
// buffer may be sized by anything routing-dependent — every MoE buffer below is
// sized for the STATIC worst case of all E experts). Side-effect-free/idempotent.
//
// Every MoE buffer is declared LayerWide over the MLP window [7, 8], i.e. the
// phase range of the dense MLP it replaces. Pitfall G audit (each buffer's true
// first-write .. last-read, all inside emit_moe_mlp):
//   moe_logits/scores/choice/scoresT/choiceT/work/rmax/mask/rw/rwbc : 7 .. 7
//   moe_gate/moe_up            : written 7, read 7            (per expert)
//   moe_down                   : written 7, read 7            (per expert)
//   moe_acc                    : written 7 (e=0 down_proj), read+written 7 (local adds),
//                                read 8 (final all-reduce). [R4] holds per-core PARTIALS,
//                                not the all-reduced value -- nothing else reads it.
//   residual1 (base, [1,8])    : read 7 (router/gate/up), written 8 (final reduce)
//   residual2 (base, aliases input_norm [1,8]) : read 8 ([R4] the single final reduce's
//                                residual; was read 7 as the e=0 seed. Still inside its
//                                declared [1,8] window and never written during 7-8.)
// Declaring the whole MoE set over [7,8] makes it mutually non-aliasing AND
// non-aliasing with residual1/input_norm, while still letting the allocator
// alias it onto the attention buffers (phases 2..5), which are dead by then.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<BufferDecl> LingbotV2MoeExpertModel::declare_buffers(const LayoutContext& ctx) {
    auto d = CausalDecoderModel::declare_buffers(ctx);
    if (num_experts_ <= 0) return d;   // not yet configured — base layout only

    const int64_t cs = ctx.chunk_size;
    const int64_t h  = hidden_size();
    const int64_t E  = num_experts_;
    const int64_t Tp = Align(cs, (int64_t)16);
    const int64_t I  = routed_inter_;
    constexpr int DW = 2;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
    using SC = StorageClass;
    constexpr BufferScope ALL = BufferScope::LayerWide;

    const int64_t te   = A(Tp * E * DW);        // [Tp,E] / [E,Tp] router slabs
    const int64_t half = A((E / 2) * Tp * DW);  // tree-reduction scratch (moe_sum)
    const int64_t tk16 = A(Tp * top_k_ * 2);     // fp16 key / raw uint16 ids
    const int64_t tk32 = A(Tp * top_k_ * 4);     // widened int32 ids for scatter
    const int64_t mlp  = A(cs * (I / NUM_CORES) * DW);
    const int64_t full = A(cs * h * DW);

    // Router (core 0 computes; the 1xC_NxC/Nx1_NxC launchers have no num_cores
    // parameter and always run broadcast on all 8 cores, so cores 1..7 compute
    // garbage into these slots from garbage inputs — inert, never read).
    // ── DEBUG dense-soft router scratch (sized from Tp/E ONLY — never from a
    //    routing result, so declare_buffers stays a pure function of
    //    LayoutContext + static model config; Pitfall G-3). Declared
    //    unconditionally so the SPM layout hash does not depend on the env
    //    flag (a flag flip must not silently alias onto a different layout).
    d.push_back({"moe_logits",  te,   7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_scores",  te,   7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_scoresT", te,   7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_sum",     half, 7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_rwbc",    te,   7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_work",    te,   7, 8, SC::Temp, 0, nullptr, ALL});  // strict top-k dense mask
    d.push_back({"moe_topk_keys", tk16, 7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_topk_ids",  tk16, 7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_topk_ids_i32",tk32,7, 8, SC::Temp, 0, nullptr, ALL});
    // Routed experts.
    d.push_back({"moe_gate",    mlp,  7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_up",      mlp,  7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_down",    full, 7, 8, SC::Temp, 0, nullptr, ALL});
    d.push_back({"moe_acc",     full, 7, 8, SC::Temp, 0, nullptr, ALL});
    // ── GROUPED-EXPERTS scratch (core-slice-interleaved). Added only when packed
    //    weights are bound; a run has a fixed opt-in flag so the layout hash is
    //    stable within the run and the baseline path is byte-unchanged.
    if (grouped_experts_) {
        const int64_t gpc = A(cs * (E * (I / NUM_CORES)) * DW);  // per-core [seq, E*Ic]
        // DIAG: Persistent (never aliased) to test whether Temp aliasing corrupts the big buffers.
        d.push_back({"moe_ggate", gpc, 7, 8, SC::Temp, 0, nullptr, ALL});
        d.push_back({"moe_gup",   gpc, 7, 8, SC::Temp, 0, nullptr, ALL});
        d.push_back({"moe_gsilu", gpc, 7, 8, SC::Temp, 0, nullptr, ALL});
        d.push_back({"moe_gscale",gpc, 7, 8, SC::Temp, 0, nullptr, ALL});
        d.push_back({"moe_wtm",   te,  7, 8, SC::Temp, 0, nullptr, ALL});
        d.push_back({"moe_wtmb",  te,  7, 8, SC::Temp, 0, nullptr, ALL});  // relay DEST (src!=dst)
    }

    // DEBUG-ONLY: never-aliased slot for the layer-0 post-AdaRMS activation. Declared ONLY
    // when RPU_L2_CAPTURE_INNORM=1, so with the flag off the SPM layout (and its hash) is
    // byte-identical to the baseline. Persistent over the whole layer window [0,8] so no
    // other buffer can be aliased onto it while the capture DMA is still draining — the
    // failure mode that made the first attempt read residual2's data instead.
    if (debug_capture_innorm_) {
        d.push_back({"innorm_dbg", full, 0, 8, SC::Persistent, 0, nullptr, ALL});
    }

    // Per-layer router correction bias [E] — PersistentPerLayer preload,
    // DMA'd once per BUILD (re-fires on invalidate_model_state). Callback emits
    // ONLY a DMA launch; the framework owns the batch capture context.
    {
        BufferDecl ids;
        ids.name      = "moe_expert_scatter_ids";
        ids.size      = te;
        ids.storage   = StorageClass::Persistent;
        ids.scope     = BufferScope::LayerWide;
        ids.preload_callback =
            [this](FusedModelBase&, int, uint32_t core0_addr) {
                rpu_launch_ddr_broadcast_spm_dma(
                    expert_scatter_ids_.data_ptr<c10::Half>(),
                    tp_rows_ * num_experts_, core0_addr,
                    /*num_cores=*/1);
            };
        d.push_back(ids);
    }
    {
        BufferDecl b;
        b.name      = "moe_rbias";
        b.size      = A(((E + 255) / 256) * 256 * DW);   // DMA-safe slot
        b.storage   = StorageClass::PersistentPerLayer;
        b.per_layer = num_layers();
        b.scope     = BufferScope::LayerWide;
        b.preload_callback =
            [this](FusedModelBase&, int L, uint32_t core0_addr) {
                rpu_launch_ddr_broadcast_spm_dma(
                    router_bias_[L].data_ptr<c10::Half>(), num_experts_, core0_addr);
            };
        d.push_back(b);
    }

    if (denoise_unroll_) {
        // Controlled pre/post hook scratch. x_t survives the complete
        // pre->36-layer->post body; the remaining buffers live only on their
        // producing side of the decoder and may alias outside those windows.
        const int64_t state = A(state_dim_pad_ * DW);
        const int64_t x = A(cs * action_dim_pad_ * DW);
        d.push_back({"denoise_x",        x,                 0, 9, SC::Temp, 0, nullptr, ALL});
        d.push_back({"denoise_state",    state,             0, 0, SC::Temp, 0, nullptr, ALL});
        d.push_back({"denoise_state_b",  A(h * DW),         0, 0, SC::Temp, 0, nullptr, ALL});
        d.push_back({"denoise_state_h",  A(h * DW),         0, 0, SC::Temp, 0, nullptr, ALL});
        d.push_back({"denoise_enc",      A(cs * h * DW),    0, 0, SC::Temp, 0, nullptr, ALL});
        d.push_back({"denoise_emb",      A(cs * h * DW),    0, 0, SC::Temp, 0, nullptr, ALL});
        d.push_back({"denoise_time",     A(h * DW),         0, 0, SC::Temp, 0, nullptr, ALL});
        d.push_back({"denoise_mo_b",     A(h * DW),         0, 0, SC::Temp, 0, nullptr, ALL});
        d.push_back({"denoise_v",        x,                 8, 9, SC::Temp, 0, nullptr, ALL});
        d.push_back({"denoise_op_b",     A(action_dim_pad_ * DW),
                                                              8, 9, SC::Temp, 0, nullptr, ALL});
    }
    return d;
}

ModelStaticConfig LingbotV2MoeExpertModel::static_config() {
    ModelStaticConfig cfg = CausalDecoderModel::static_config();
    if (denoise_unroll_) {
        cfg.pre_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
            &LingbotV2MoeExpertModel::emit_denoise_pre_layers);
        cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
            &LingbotV2MoeExpertModel::emit_denoise_post_layers);
        cfg.body_iterations = num_steps_;
        // The complete pre -> 36 layers -> post stream is replay-safe: every
        // per-call input is refreshed through a mutable base before
        // run_all_layers.  Opt this handle into the existing per-model replay
        // path instead of relying on the process-wide Wall-OSS switch.  The
        // full-body capability below is the matching permission for the
        // pre/post hooks and ten body iterations.
        cfg.fast_replay_skip_layer_loop = true;
        cfg.fast_replay_skip_full_body = true;
    }
    return cfg;
}

void LingbotV2MoeExpertModel::emit_denoise_pre_layers() {
    const int64_t bit = ctx().body_iter;
    const int64_t h = hidden_size();
    const int64_t cs = chunk_size_;

    if (bit == 0) {
        // Fresh caller state/noise are mutable graph inputs. State projection
        // is evaluated once; its FP16 output remains in the stable DDR stage.
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &denoise_state_src_base_, 0, state_dim_pad_,
            addr(0, "denoise_state"), /*num_cores=*/1);
        rpu_launch_ddr_broadcast_spm_dma(
            state_b_.data_ptr<c10::Half>(), h,
            addr(0, "denoise_state_b"), /*num_cores=*/1);
        rpu_launch_linear_spm_to_spm_kernel(
            addr(0, "denoise_state"), state_w_, addr(0, "denoise_state_h"),
            /*M=*/1, /*N=*/h, /*K=*/state_dim_pad_,
            /*partition=*/1, /*num_cores=*/1,
            addr(0, "denoise_state_b"), /*force_gemm=*/true);
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "denoise_state_h"),
            denoise_stage_.data_ptr<c10::Half>(), h);

        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &denoise_x0_src_base_, 0, cs * action_dim_pad_,
            addr(0, "denoise_x"), /*num_cores=*/1);
    }

    rpu_launch_ddr_broadcast_spm_dma(
        time_all_.data_ptr<c10::Half>() + bit * h, h,
        addr(0, "denoise_time"), /*num_cores=*/1);
    rpu_launch_ddr_broadcast_spm_dma(
        mo_bias_.data_ptr<c10::Half>(), h,
        addr(0, "denoise_mo_b"), /*num_cores=*/1);

    // Existing Linear kernel: FP16 input/weight/output with internal ACC32.
    rpu_launch_linear_spm_to_spm_kernel(
        addr(0, "denoise_x"), wc_, addr(0, "denoise_enc"),
        /*M=*/cs, /*N=*/h, /*K=*/action_dim_pad_,
        /*partition=*/1, /*num_cores=*/1,
        addr(0, "denoise_time"), /*force_gemm=*/true);
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "denoise_enc"), addr(0, "denoise_enc"),
        cs * h, ValuOpType::SILU, /*is_gelu=*/false, /*num_cores=*/1);
    rpu_launch_linear_spm_to_spm_kernel(
        addr(0, "denoise_enc"), mo_, addr(0, "denoise_emb"),
        /*M=*/cs, /*N=*/h, /*K=*/h,
        /*partition=*/1, /*num_cores=*/1,
        addr(0, "denoise_mo_b"), /*force_gemm=*/true);

    // Row 0 is the projected state written above. Only action rows change per
    // Euler step, so preserve row 0 and replace rows [1, cs).
    rpu_launch_spm_copy_ddr_dma(
        addr(0, "denoise_emb") + h * static_cast<int64_t>(sizeof(c10::Half)),
        denoise_stage_.data_ptr<c10::Half>() + h, (cs - 1) * h);
}

void LingbotV2MoeExpertModel::emit_denoise_post_layers() {
    const int64_t bit = ctx().body_iter;
    const int64_t h = hidden_size();
    const int64_t cs = chunk_size_;

    rpu_launch_ddr_broadcast_spm_dma(
        op_bias_.data_ptr<c10::Half>(), action_dim_pad_,
        addr(0, "denoise_op_b"), /*num_cores=*/1);
    rpu_launch_linear_spm_to_spm_kernel(
        addr(0, "residual1"), op_, addr(0, "denoise_v"),
        /*M=*/cs, /*N=*/action_dim_pad_, /*K=*/h,
        /*partition=*/1, /*num_cores=*/1,
        addr(0, "denoise_op_b"), /*force_gemm=*/true);

    // FP16 state between steps; dt is converted to FP16 and baked at BUILD.
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "denoise_x"), addr(0, "denoise_v"), addr(0, "denoise_x"),
        cs * action_dim_pad_, ValuOpType::ADD, denoise_dt_, /*num_cores=*/1);
    if (bit + 1 == num_steps_) {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "denoise_x"), &denoise_final_dst_base_,
            /*dst_offset_bytes=*/0, cs * action_dim_pad_);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_router_select — FP16-storage router with strict exactly-k selection.
//
// ROUTER CAPABILITY GAP
// A bit-identical CPU-reference router requires FP32 logits through selection.
// The current operator asset stores this path in FP16; a later cast cannot
// recover lost mantissa bits. The existing graph-aware selector provides
// exactly-k selection, stable tie-breaking, and correction bias after FP16
// storage, but near-boundary FP32 choices remain outside its contract.
//
// REQUIRED OP CONTRACT:
//   in : hidden [T,H] fp16 SPM, gate_w [E,H], corr_bias [E]
//   fp32 INTERNALLY, logits never materialising in fp16:
//     logits(fp32) -> sigmoid(fp32) -> +corr_bias -> STRICT top-k (exactly k)
//     -> gather weights from the UNBIASED sigmoid scores -> /(sum+1e-20)
//     -> *routed_scaling
//   out: (a) selected_indices [T,k] int32
//        (b) routing_weights  [T,k] fp32/fp16 (normalised and scaled)
//        (c) dense [E,Tp] scatter of (a)x(b)  <- REQUIRED TOO: the dense-einsum
//            combine needs it and no scatter kernel exists either.
//   tie-break: LOWEST INDEX WINS (strict `>` on an ascending scan).
//
// Explicitly disabling the FP16 route still fails loudly rather than pretending
// the unavailable full-FP32 contract exists.
// ─────────────────────────────────────────────────────────────────────────────
void LingbotV2MoeExpertModel::emit_router_select(int layer_idx, int64_t seq_len) {
    if (!debug_dense_soft_router_ && !fp16_top4_) {
        // Strict path: fail rather than silently approximating FP32 routing.
        TORCH_CHECK(false,
            "LingbotV2MoeExpertModel: the fp32 fused router-select op is NOT available "
            "in the current operator asset. The default FP16 backend_moe path provides exactly-k, stable "
            "tie-breaking, and selection-only correction bias, but it cannot reproduce "
            "near-boundary FP32 choices after values collapse to FP16. "
            "ACTION: enable RPU_LINGBOT2_FP16_TOP4=1, or obtain a matched asset "
            "with a validated full-FP32 router-select operator. "
            "See emit_router_select() for the full spec. "
            "DEBUG-ONLY BYPASS: RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER=1 selects a dense "
            "soft router (NO top-k, all 32 experts, ACCURACY_UNVALIDATED).");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DEBUG FP16 DENSE SOFT ROUTER — diagnostic only, NOT the official semantics.
    //
    //   logits  = h @ gate_w^T            (FP32 accumulate, FP16 output, core-0)
    //   scores  = sigmoid(logits)         (existing FP16 SPM eltwise, {19,0})
    //   w[e,t]  = scores[t,e] / sum_e' scores[t,e']     <- dense over ALL 32
    //   (routed_scaling 4.0 is folded into the per-expert combine alpha,
    //    rpu_lingbot_v2_moe_model.cpp emit_moe_mlp — NOT applied here.)
    //
    // This debug dense-soft router intentionally differs from the reference path:
    //   * NO top-k          — every expert is weighted, none selected.
    //   * fp16 output/sigmoid — the reference keeps the router in true fp32.
    //   * denominator       — sum over all 32, not over the top-4.
    //   * correction bias   — official adds it ONLY to the SELECTION key
    //                         (scores_for_choice) and gathers the weights
    //                         from the unbiased scores. This path performs
    //                         no selection, so the bias MUST NOT enter the
    //                         weights. router_bias_ / the "moe_rbias"
    //                         preload stay loaded and bound (interface kept), but
    //                         are intentionally not read here.
    //
    // Address convention:
    //   * rpu_launch_transpose_nchw_to_nhwc_spm takes an SPM **OFFSET**: it does
    //     SPM_ALLOC.addr(0, off) itself, and
    //     SPM_ALLOC::addr(core,off) = base_addr_[core] + off. Passing addr()
    //     would double-add the base. So the transpose gets addr_offset(name).value; every other kernel
    //     below gets the absolute addr().
    //   * The transpose is core-0 only
    //     and hardcodes core 0 in its own addr() call — consistent with running
    //     the whole router on core 0 and relaying the result to all 8 cores.
    // ═══════════════════════════════════════════════════════════════════════════
    const int64_t h  = hidden_size();
    const int64_t E  = num_experts_;
    const int64_t Tp = tp_rows_;
    constexpr int64_t DW = 2;

    // ── 0. Zero the whole [Tp,E] score slab from the keepalive, so the pad rows
    //       [seq_len, Tp) are exactly 0 (they feed the transpose + tree-sum).
    //       sigmoid(0)=0.5 would otherwise make pad lanes nonzero; zeroing AFTER
    //       sigmoid is what we want, so we zero moe_scores post-sigmoid below.

    // ── 1. logits = residual1 @ gate_w^T  -> [Tp, E], core-0 only.
    //    Pitfall H: N is the FULL output dim; the kernel divides by num_cores.
    //    router_gate_ was col-swizzled for one core.
    // ── 0. DEBUG-ONLY: capture the router's REAL input BEFORE any router math runs.
    //    Reads residual1, writes a private per-layer slot. Emits nothing when the opt-in
    //    env var is off, so the router math / weights / kernel order are unchanged.
    if (debug_dump_router_h_) {
        c10::Half* hstage = h_stage_.data_ptr<c10::Half>()
                          + (int64_t)layer_idx * tp_rows_ * h;
        rpu_launch_spm_copy_ddr_dma(addr(0, "residual1"), hstage, seq_len * h);
    }

    // Accumulate the router GEMM in FP32, then store FP16 logits. This cannot
    // recover the exact FP32-selection contract, but it avoids an unnecessary
    // ACC16 deviation before the already-documented FP16 storage boundary.
    rpu_launch_linear_spm_to_spm_kernel(
        addr(0, "residual1"), router_gate_[layer_idx], addr(0, "moe_logits"),
        seq_len, /*N=*/E, /*K=*/h, /*partition=*/1, /*num_cores=*/1);

    // ── 2. scores = sigmoid(logits) on the REAL tokens only -> [seq_len, E].
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "moe_logits"), addr(0, "moe_scores"),
        seq_len * E, ValuOpType::SIGMOID, /*is_gelu=*/false, /*num_cores=*/1);

    // ── 3. Fill the pad rows [seq_len, Tp) of moe_scores with 1.0. They are
    //       transposed and tree-summed together with the real rows; leaving
    //       sigmoid garbage (or stale SPM) there would put junk into pad columns
    //       of moe_rwbc. Pad columns are never read by the combine (n=seq_len),
    //       but junk/NaN there would still be a latent trap.
    //       1.0 (not 0.0) => each pad column sums to E and divides to a finite
    //       1/E, so the denominator needs no pad patch — see the ones_pad_ note in
    //       set_moe_weights for why that patch could not be 16-byte aligned.
    //       (Tp-seq_len)*E*2 bytes is a multiple of 64, so this DMA is always
    //       aligned. Real tokens are untouched: pad rows reach pad columns only.
    if (seq_len < Tp) {
        rpu_launch_ddr_broadcast_spm_dma(
            ones_pad_.data_ptr<c10::Half>(), (Tp - seq_len) * E,
            addr(0, "moe_scores") + seq_len * E * DW, /*num_cores=*/1);
    }

    // ── 4. scoresT = transpose(scores): [C=Tp, H=1, W=E] -> [H=1, W=E, C=Tp]
    //       i.e. [Tp,E] -> [E,Tp]. C (= the token dim) must be v16-aligned; that is
    //       exactly why tp_rows_ = Align(chunk_size,16) (set_moe_weights).
    //       ★ OFFSETS here, not addr(), because this kernel takes typed SPM offsets.
    rpu_launch_transpose_nchw_to_nhwc_spm(
        addr_offset("moe_scores").value, addr_offset("moe_scoresT").value,
        /*C=*/(int)Tp, /*H=*/1, /*W=*/(int)E);

    // ── 5T. strict fp16 top-k (RPU_LINGBOT2_FP16_TOP4).
    //        choice = unbiased sigmoid score + correction bias; exactly k ids are
    //        selected with lower expert index winning an exact tie.  The helper
    //        scatters a dense [E,Tp] mask, gathers UNBIASED scores, and normalises.
    //        moe_logits is reused as token-major choice after its logits are dead.
    if (fp16_top4_) {
        rpu_launch_backend_moe_select_fp16_spm(
            addr(0, "moe_scores"),
            addr(0, "moe_scoresT"),
            layer_addr(layer_idx, 0, "moe_rbias"),
            addr(0, "moe_expert_scatter_ids"),
            addr(0, "moe_logits"),
            addr(0, "moe_topk_keys"),
            addr(0, "moe_topk_ids"),
            addr(0, "moe_topk_ids_i32"),
            addr(0, "moe_work"),
            addr(0, "moe_sum"),
            addr(0, "moe_rwbc"),
            Tp, E, top_k_);
    }

    if (!fp16_top4_) {
    // ── 5. sum = tree_sum over E -> [1, Tp]. With E as the OUTER dim the reduction
    //       is log2(E) contiguous halvings. First halving reads moe_scoresT and
    //       writes moe_sum (preserving moe_scoresT for step 6); the rest are
    //       in-place on moe_sum.
    {
        int64_t half_rows = E / 2;
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "moe_scoresT"),
            addr(0, "moe_scoresT") + half_rows * Tp * DW,
            addr(0, "moe_sum"),
            half_rows * Tp, ValuOpType::ADD, c10::Half(1.0f), /*num_cores=*/1);
        for (half_rows /= 2; half_rows >= 1; half_rows /= 2) {
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "moe_sum"),
                addr(0, "moe_sum") + half_rows * Tp * DW,
                addr(0, "moe_sum"),
                half_rows * Tp, ValuOpType::ADD, c10::Half(1.0f), /*num_cores=*/1);
        }
    }

    // ── 6. rwbc = scoresT / sum  -> [E, Tp], row-sum over E == 1 per token.
    //       a = [1,C] broadcast operand = moe_sum [1,Tp];  b = [N,C] = scoresT.
    //       out = scale_a*a op scale_b*b with scale_a=1, scale_b=alpha
    //       so DIV would give sum/scoresT.
    //       is_bopa swaps the operand order for non-commutative ops (:298) =>
    //       out = scoresT / sum. Correct direction.
    //       No +1e-20 guard: sigmoid > 0 strictly, so a 32-term sum can never be
    //       0 (the official 1e-20 protects a top-k gather, which we do not do).
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
        addr(0, "moe_sum"), addr(0, "moe_scoresT"), addr(0, "moe_rwbc"),
        /*n=*/E, /*c=*/Tp, c10::Half(1.0f), ValuOpType::DIV, /*is_bopa=*/true);

    }  // end !fp16_top4_ (dense-soft /sum weights)

    // Apply routed_scaling_factor after score normalization. The grouped
    // combine does not apply this scalar, so scale the core-0 routing weights
    // before the relay; both expert paths then consume the same scaled values.
    if (routed_scaling_ != c10::Half(1.0f)) {
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            addr(0, "moe_rwbc"), routed_scaling_, addr(0, "moe_rwbc"),
            E * Tp, ValuOpType::MUL);
    }

    // ── 7. Relay core0 -> DDR -> all 8 cores: the router ran on core 0 only, but
    //       the per-expert Nx1_NxC combine needs moe_rwbc on every core.
    //       rw_stage_ is a stable registered tensor with one slot PER LAYER, so
    //       the 36 relays inside one graph have no WAR hazard => fixed DMA is safe.
    {
        c10::Half* stage = rw_stage_.data_ptr<c10::Half>() + (int64_t)layer_idx * E * Tp;
        rpu_launch_spm_copy_ddr_dma(addr(0, "moe_rwbc"), stage, E * Tp);
        rpu_launch_ddr_broadcast_spm_dma(stage, E * Tp, addr(0, "moe_rwbc"), NUM_CORES);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// emit_moe_mlp — replaces emit_mlp_pipeline (the dense MLP) for one layer.
//   in : residual1 = post-attention normed hidden [cs, h] (replicated, 8 cores)
//        residual2 = residual stream [cs, h]
//   out: residual1 = residual2 + y_routed + y_shared
// ─────────────────────────────────────────────────────────────────────────────
void LingbotV2MoeExpertModel::emit_moe_mlp(int layer_idx, int64_t seq_len) {
    // DIAG RPU_L2_BUFONLY: grouped buffers stay declared, but run the per-expert path.
    const bool bufonly = (std::getenv("RPU_L2_BUFONLY") != nullptr);
    if (grouped_experts_ && !bufonly) { emit_moe_mlp_grouped(layer_idx, seq_len); return; }
    const int64_t h  = hidden_size();
    const int64_t E  = num_experts_;
    const int64_t Tp = tp_rows_;
    const int64_t I  = routed_inter_;
    const int64_t S  = intermediate_size();   // shared-expert intermediate (padded)
    const int64_t Ic = I / NUM_CORES;
    const int64_t Sc = S / NUM_CORES;

    // ══ ROUTER / SELECT ══════════════════════════════════════════════════════
    // Produces "moe_rwbc" = the dense [E, Tp] routing-weight matrix consumed by
    // the expert combine below (w[e,t] = normalised x routed_scaling if expert e
    // is in token t's top-k, else 0).
    //
    // Exact top-k routing uses explicit selected IDs.
    emit_router_select(layer_idx, seq_len);

    // ══ ROUTED EXPERTS (dense-einsum: every expert, every token) ═════════════
    for (int64_t e = 0; e < E; ++e) {
        const int64_t wi = layer_idx * E + e;
        // gate_e = h @ gate_w_e.T   (col-partition; Pitfall H: N is the FULL dim)
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), expert_gate_[wi], addr(0, "moe_gate"),
            seq_len, /*N=*/I, /*K=*/h, /*partition=*/1, NUM_CORES);
        // up_e = h @ up_w_e.T
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), expert_up_[wi], addr(0, "moe_up"),
            seq_len, /*N=*/I, /*K=*/h, /*partition=*/1, NUM_CORES);
        // act = silu(gate) * up   — exactly SwiGLU (rhino llama_silu_mul)
        rpu_launch_silu_mul_spm_kernel(
            addr(0, "moe_gate"), addr(0, "moe_up"), addr(0, "moe_gate"),
            seq_len * Ic, NUM_CORES);
        // act *= routed_scaling * w[:,e]. down_e is linear and bias-free, so
        // scaling the ACTIVATION is identical to scaling the expert output —
        // and it is 8x cheaper (Ic vs h columns). w[:,e] is the contiguous [Tp]
        // slice at e*Tp; n=seq_len uses only the real tokens. alpha folds in 4.0.
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
            addr(0, "moe_rwbc") + e * Tp * DWIDTH, addr(0, "moe_gate"), addr(0, "moe_gate"),
            /*n=*/seq_len, /*c=*/Ic, routed_scaling_, ValuOpType::MUL, /*is_bopa=*/false);
        // down_e (row-partition -> per-core partial).
        // [R4] e==0 writes the partial STRAIGHT into moe_acc, seeding the accumulator
        // without a copy and without a zero-fill (it fully overwrites [cs,h]).
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "moe_gate"), expert_down_[wi],
            e == 0 ? addr(0, "moe_acc") : addr(0, "moe_down"),
            seq_len, /*N=*/h, /*K=*/I, /*partition=*/0, NUM_CORES);
        // [R4] LOCAL per-core partial accumulate -- no cross-core traffic.
        // moe_acc now holds each core's PARTIAL sum over experts; the single
        // all-reduce after the shared expert turns it into the true sum.
        if (e != 0) {
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "moe_acc"), addr(0, "moe_down"), addr(0, "moe_acc"),
                seq_len * h, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
        }
    }

    // ══ SHARED EXPERT (always-on, UNGATED — no shared_expert_gate) ═══════════
    // Bound through the base's dense gate_w/up_w/down_w with
    // intermediate_size == shared_inter_pad, so it reuses the base's
    // "gate"/"up"/"down" slots. Emitted by hand rather than via
    // emit_mlp_pipeline because the final reduce must add moe_acc (which already
    // carries y_routed + the residual stream), not residual2.
    const auto& lw = layer_weights_[layer_idx];
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
        seq_len, /*N=*/S, /*K=*/h, /*partition=*/1, NUM_CORES,
        /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.gate_ws);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), lw.up_w, addr(0, "up"),
        seq_len, /*N=*/S, /*K=*/h, /*partition=*/1, NUM_CORES,
        /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.up_ws);
    rpu_launch_silu_mul_spm_kernel(
        addr(0, "gate"), addr(0, "up"), addr(0, "gate"), seq_len * Sc, NUM_CORES);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "gate"), lw.down_w, addr(0, "down"),
        seq_len, /*N=*/h, /*K=*/S, /*partition=*/0, NUM_CORES,
        /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.down_ws);
    // [R4] Fold the shared expert's per-core partial into the SAME local accumulator...
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "moe_acc"), addr(0, "down"), addr(0, "moe_acc"),
        seq_len * h, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
    // ...then ONE all-reduce for the whole layer:
    //   residual1 = AllReduce_c(sum_e p_c(e) + p_c(shared)) + residual2
    // residual2 is still live here (declared [1,8]; only READ during phase 7-8), and
    // residual1 is written LAST, after every expert has read it as the MoE input.
    // input/residual/output are distinct, so the reduce may use the two-stage path.
    const bool force_fast_allreduce =
        denoise_twostage_ && denoise_unroll_active_;
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "moe_acc"), addr(0, "residual2"),
        addr(0, "residual1"), seq_len, h, NUM_CORES, NUM_CORES,
        /*force_twostage=*/force_fast_allreduce,
        /*force_chunk_v2=*/force_fast_allreduce);
    if (debug_capture_l0_ && layer_idx == 0)
        rpu_launch_spm_copy_ddr_dma(addr(0, "residual1"), l0_out_.data_ptr<c10::Half>(), seq_len * h);
}


// ─────────────────────────────────────────────────────────────────────────────
// emit_moe_mlp_grouped — CORE-SLICE-INTERLEAVED grouped experts.
//
// Packing (built in convert.py): core c holds slice-c (the c-th I/8 rows) of EVERY
// expert, so the per-core packed GEMM computes all E experts' slice-c in ONE launch.
// Because every core sees all E experts, the routing weight w[seq,E] is IDENTICAL on
// all cores (a plain broadcast) — this is what makes a single contiguous scale legal.
//   gate/up : packed [E*I, h], col-partition N=E*I  -> per-core [seq, E*Ic]
//   down    : packed [h, E*I], row-partition K=E*I   -> per-core PARTIAL [seq, h]
//             (the K=E*I contraction sums BOTH slices and experts; the final
//              all-reduce sums the 8 per-core partials => exact sum over all experts)
// Kernel count/layer ~= 19 vs the per-expert loop's ~176 (32 experts x 5-6 kernels).
// Numerically identical dense-soft math to emit_moe_mlp, just reassociated.
// ─────────────────────────────────────────────────────────────────────────────
void LingbotV2MoeExpertModel::emit_moe_mlp_grouped(int layer_idx, int64_t seq_len) {
    const int64_t h   = hidden_size();
    const int64_t E   = num_experts_;
    const int64_t Tp  = tp_rows_;
    const int64_t I   = routed_inter_;
    const int64_t S   = intermediate_size();
    const int64_t Sc  = S / NUM_CORES;
    const int64_t Ic  = I / NUM_CORES;      // per-core rows of one expert (64)
    const int64_t GpC = E * Ic;             // per-core packed width (E*Ic = 2048)
    const int64_t NI  = E * I;              // packed N/K (16384)

    // ── ROUTER: emit_router_select leaves the FINAL per-token routing weights in moe_rwbc
    //    [E,Tp] (both router variants), plus moe_scores[seq,E] + moe_sum[1,Tp] on core 0.
    //    Then build the TOKEN-MAJOR weights w[seq,E] and relay to all 8 cores (grouped
    //    experts need the full [seq,E] on every core).
    emit_router_select(layer_idx, seq_len);
    if (layer_idx == 0 && std::getenv("RPU_L2_ADDR")) {
        const int64_t te_ = Align(Tp * E * 2, (int64_t)256);
        const int64_t half_ = Align((E/2) * Tp * 2, (int64_t)256);
        const int64_t gpc_ = Align(seq_len * GpC * 2, (int64_t)256);
        const int64_t full_ = Align(seq_len * h * 2, (int64_t)256);
        std::printf("SPMADDR2 moe_logits=%u,%ld moe_scores=%u,%ld moe_scoresT=%u,%ld moe_sum=%u,%ld moe_rwbc=%u,%ld moe_wtm=%u,%ld moe_ggate=%u,%ld moe_gup=%u,%ld moe_gsilu=%u,%ld moe_gscale=%u,%ld moe_acc=%u,%ld residual1=%u,%ld residual2=%u,%ld\n",
            addr(0,"moe_logits"),te_, addr(0,"moe_scores"),te_, addr(0,"moe_scoresT"),te_,
            addr(0,"moe_sum"),half_, addr(0,"moe_rwbc"),te_, addr(0,"moe_wtm"),te_,
            addr(0,"moe_ggate"),gpc_, addr(0,"moe_gup"),gpc_, addr(0,"moe_gsilu"),gpc_,
            addr(0,"moe_gscale"),gpc_, addr(0,"moe_acc"),full_, addr(0,"residual1"),full_, addr(0,"residual2"),full_);
        std::fflush(stdout);
    }
    if (fp16_top4_) {
        // ★ TOP-4 MUST COME FROM moe_rwbc. emit_router_select's top-4 branch already produced
        //   the MASKED + renormalised weights in moe_rwbc [E,Tp] (mask = scoresT >= thr, then
        //   /S_top4). Rebuilding w from moe_scores here would silently DROP that mask: sigmoid
        //   is strictly > 0, so all 32 experts would keep a nonzero weight and the only residue
        //   of top-4 would be the denominator (S_top4 instead of S_all) — i.e. dense-soft times
        //   a per-token gain, with ZERO top-k sparsity and no error signal. So transpose the
        //   already-correct [E,Tp] weights into the token-major [Tp,E] layout the grouped path
        //   consumes. (Inverse of the moe_scores->moe_scoresT transpose above: there C=Tp,W=E;
        //   here C=E,W=Tp. Rows 0..seq_len-1 of the [Tp,E] result are the real tokens.)
        //   ★ OFFSETS here, not addr(), as required by the transpose launcher.
        rpu_launch_transpose_nchw_to_nhwc_spm(
            addr_offset("moe_rwbc").value, addr_offset("moe_wtm").value,
            /*C=*/(int)E, /*H=*/1, /*W=*/(int)Tp);
    } else {
        // dense-soft: w[t,e] = moe_scores[t,e] / moe_sum[t]  (Nx1_NxC: per-row broadcast of the
        //   sum; is_bopa=true selects b/a = scores/sum, matching the [E,Tp] rwbc divide direction)
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
            addr(0, "moe_sum"), addr(0, "moe_scores"), addr(0, "moe_wtm"),
            /*n=*/seq_len, /*c=*/E, c10::Half(1.0f), ValuOpType::DIV, /*is_bopa=*/true);
    }
    {
        c10::Half* stage = wtm_stage_.data_ptr<c10::Half>() + (int64_t)layer_idx * Tp * E;
        rpu_launch_spm_copy_ddr_dma(addr(0, "moe_wtm"), stage, seq_len * E);
        // Broadcast into a SEPARATE dest: with src==dst (moe_wtm->moe_wtm), core 0
        // is both the DMA source and a destination, so aliasing can corrupt its
        // copy. All cores read the routing weights from moe_wtmb.
        rpu_launch_ddr_broadcast_spm_dma(stage, seq_len * E, addr(0, "moe_wtmb"), NUM_CORES);
    }

    // ── GROUPED EXPERTS ──────────────────────────────────────────────────────
    // gate_all = residual1 @ gate_packed^T  (col-partition, N=E*I) -> per-core [seq, E*Ic]
    // Quantized packed weights carry W8 [N] or a controller-striped W4 pgrp
    // scale payload; when
    // the packed weights are fp16 these stay undefined and the launcher takes its fp16 path.
    const at::Tensor gs = packed_gate_s_.empty() ? at::Tensor() : packed_gate_s_[layer_idx];
    const at::Tensor us = packed_up_s_.empty()   ? at::Tensor() : packed_up_s_[layer_idx];
    const at::Tensor ds = packed_down_s_.empty() ? at::Tensor() : packed_down_s_[layer_idx];
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), packed_gate_[layer_idx], addr(0, "moe_ggate"),
        seq_len, /*N=*/NI, /*K=*/h, /*partition=*/1, NUM_CORES,
        /*bias_spm_addr=*/0u, /*force_gemm=*/false, gs);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), packed_up_[layer_idx], addr(0, "moe_gup"),
        seq_len, /*N=*/NI, /*K=*/h, /*partition=*/1, NUM_CORES,
        /*bias_spm_addr=*/0u, /*force_gemm=*/false, us);
    // act = silu(gate) * up over the whole per-core width.
    //
    // `llama_silu_mul` reads each block length as signed 16-bit, so the shared
    // launcher uses signed-safe 32512-element blocks and submits larger calls as
    // multiple blocks in one device grid.
    //
    // The per-expert path never reached the boundary: seq_len*Ic = 51*64 = 3264. This
    // Keep the outer host grouping fixed at 32512 unless explicitly overridden.
    {
        const int64_t total = seq_len * GpC;
        // DIAG RPU_L2_SCHUNK: override the outer host grouping (default 32512). The shared
        // launcher still enforces signed-safe device blocks for every positive value.
        const int64_t CHUNK = [] {
            const char* e = std::getenv("RPU_L2_SCHUNK");
            const int64_t v = (e != nullptr) ? (int64_t)std::atol(e) : 0;
            return v > 0 ? v : (int64_t)(256 * 127);   // "" -> atol==0 -> off+=0 hangs
        }();
        // debug_stages: write silu to a FRESH buffer (single-version -> reliable capture, user #8)
        const char* silu_out = debug_stages_ ? "moe_gsilu" : "moe_ggate";
        for (int64_t off = 0; off < total; off += CHUNK) {
            const int64_t nn = (total - off < CHUNK) ? (total - off) : CHUNK;
            rpu_launch_silu_mul_spm_kernel(
                addr(0, "moe_ggate") + off * DWIDTH, addr(0, "moe_gup") + off * DWIDTH,
                addr(0, silu_out) + off * DWIDTH, nn, NUM_CORES);
        }
        if (debug_stages_ && layer_idx == 0) {
            rpu_launch_spm_copy_ddr_dma(addr(0,"moe_wtm"), gcap_.data_ptr<c10::Half>(), Tp * E);  // CORE0 wtm
            rpu_launch_spm_copy_ddr_dma(addr(1,"moe_wtm"), scap_.data_ptr<c10::Half>(), Tp * E);  // CORE1 wtm
        }
    }
    if (debug_capture_gate_ && layer_idx == 0)
        rpu_launch_spm_copy_ddr_dma(addr(0, "moe_ggate"), gcap_.data_ptr<c10::Half>(), seq_len * (E * (I / NUM_CORES)));
    // act[t,e,i] *= w[t,e] * routed_scaling. Per-core view [seq,E,Ic] as [seq*E, Ic];
    //   a = moe_wtm (row t*E+e = w[t,e], IDENTICAL on all cores), contiguous.
    // act[t,e,i] *= w[t,e] * routed_scaling. The packed grouped binding requires one
    // Nx1_NxC launch over all seq_len*E rows: advancing a later host-side chunk would
    // not advance the packed B-operand inside the kernel. set_grouped_experts() therefore
    // rejects RCHUNK < chunk_size*E; the public quantized profiles set RCHUNK=1632.
    const char* scale_in  = debug_stages_ ? "moe_gsilu"  : "moe_ggate";
    const char* scale_out = debug_stages_ ? "moe_gscale" : "moe_ggate";
    {
        const int64_t nrows = seq_len * E;   // 1632
        // DIAG RPU_L2_RCHUNK: scale row-chunk size. The low default deliberately makes a
        // raw grouped opt-in fail loudly at bind time; a supported profile must select a
        // value large enough to keep this loop to one launch.
        const int64_t RCHUNK = [] {
            const char* e = std::getenv("RPU_L2_RCHUNK");
            const int64_t v = (e != nullptr) ? (int64_t)std::atol(e) : 0;
            return v > 0 ? v : (int64_t)256;          // "" -> atol==0 -> off+=0 hangs
        }();
        for (int64_t r = 0; r < nrows; r += RCHUNK) {
            const int64_t rn = (nrows - r < RCHUNK) ? (nrows - r) : RCHUNK;
            rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
                addr(0, "moe_wtmb") + r * DWIDTH,
                addr(0, scale_in) + r * Ic * DWIDTH,
                addr(0, scale_out) + r * Ic * DWIDTH,
                /*n=*/rn, /*c=*/Ic, routed_scaling_, ValuOpType::MUL, /*is_bopa=*/false);
        }
    }
    if (debug_stages_ && layer_idx == 0)
        rpu_launch_spm_copy_ddr_dma(addr(0,"moe_gscale"), ccap_.data_ptr<c10::Half>(), seq_len * GpC);
    // down_all = act @ down_packed^T (row-partition K=E*I) -> per-core PARTIAL into moe_acc.
    //   The contraction over E*I sums all experts+slices; moe_acc is fully overwritten
    //   (seed), like e==0 in the per-expert loop. ACC32 is required for this
    //   full E*I contraction; ACC16 does not preserve the reference result.
    if (debug_down_acc16_ || packed_w8a16_)
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, scale_out), packed_down_[layer_idx], addr(0, "moe_acc"),
            seq_len, /*N=*/h, /*K=*/NI, /*partition=*/0, NUM_CORES,
            /*bias_spm_addr=*/0u, /*force_gemm=*/false, ds);
    else
        rpu_launch_linear_spm_to_spm_kernel(   // acc32 = fp32-accumulate (default)
            addr(0, scale_out), packed_down_[layer_idx], addr(0, "moe_acc"),
            seq_len, /*N=*/h, /*K=*/NI, /*partition=*/0, NUM_CORES);
    if (debug_capture_gate_ && layer_idx == 0)
        rpu_launch_spm_copy_ddr_dma(addr(0, "moe_acc"), acap_.data_ptr<c10::Half>(), seq_len * h);

    // ── SHARED EXPERT (identical to emit_moe_mlp) folded into moe_acc, then ONE reduce ──
    const auto& lw = layer_weights_[layer_idx];
    if (debug_routed_only_) {
        // DIAG: layer output = pure routed contribution (zero shared + zero residual2),
        // so l0_out is CPU-verifiable against the full routed sum with no in-place capture.
        rpu_launch_memset_spm_multicore(addr(0, "residual2"), seq_len * h);
    } else {
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
        seq_len, /*N=*/S, /*K=*/h, /*partition=*/1, NUM_CORES, 0, false, lw.gate_ws);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "residual1"), lw.up_w, addr(0, "up"),
        seq_len, /*N=*/S, /*K=*/h, /*partition=*/1, NUM_CORES, 0, false, lw.up_ws);
    rpu_launch_silu_mul_spm_kernel(
        addr(0, "gate"), addr(0, "up"), addr(0, "gate"), seq_len * Sc, NUM_CORES);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "gate"), lw.down_w, addr(0, "down"),
        seq_len, /*N=*/h, /*K=*/S, /*partition=*/0, NUM_CORES, 0, false, lw.down_ws);
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "moe_acc"), addr(0, "down"), addr(0, "moe_acc"),
        seq_len * h, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
    }
    const bool force_fast_allreduce =
        denoise_twostage_ && denoise_unroll_active_;
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "moe_acc"), addr(0, "residual2"),
        addr(0, "residual1"), seq_len, h, NUM_CORES, NUM_CORES,
        /*force_twostage=*/force_fast_allreduce,
        /*force_chunk_v2=*/force_fast_allreduce);
    if (debug_capture_l0_ && layer_idx == 0)
        rpu_launch_spm_copy_ddr_dma(addr(0, "residual1"), l0_out_.data_ptr<c10::Half>(), seq_len * h);
}

// Bind per-layer packed (core-slice-interleaved) expert weights and enable the grouped
// path. Called from convert.py ONLY when RPU_LINGBOT2_GROUPED_EXPERTS=1, AFTER
// set_moe_weights. Ends with invalidate_model_state() (weights changed -> rebuild).
void LingbotV2MoeExpertModel::set_packed_expert_weights(
    at::TensorList gate_packed, at::TensorList up_packed, at::TensorList down_packed) {
    const int64_t nl = num_layers();
    TORCH_CHECK(num_experts_ > 0, "set_packed_expert_weights must follow set_moe_weights");
    TORCH_CHECK((int64_t)gate_packed.size() == nl && (int64_t)up_packed.size() == nl
             && (int64_t)down_packed.size() == nl,
                "packed weight lists must have size num_layers=", nl);
    const int64_t h  = hidden_size();
    const int64_t NI = num_experts_ * routed_inter_;
    // fp16 = the default packed path. kChar = W8A16 (int8 storage + a per-output-channel
    // fp16 scale bound separately by set_packed_expert_scales). Only the acc16 GEMM launcher
    // takes an int8 weight + scale, so the W8A16 path forces the down GEMM onto acc16 too.
    // fp16 = default. kChar = W8A16 int8 [r,c]. kByte = W4A16 nibble-packed int4 [r, c/2]
    // (quant/int4_pack.swizzle_pack_int4). All three route through the acc16 launcher, which
    // dispatches on the weight dtype; the W4/W8 paths force the down GEMM onto acc16.
    auto chk = [](const at::Tensor& t, int64_t r, int64_t c, const char* nm) {
        const bool int4 = t.defined() && t.scalar_type() == at::kByte;
        const int64_t c_eff = int4 ? c / 2 : c;
        TORCH_CHECK(t.defined()
                 && (t.scalar_type() == at::kHalf || t.scalar_type() == at::kChar
                     || t.scalar_type() == at::kByte)
                 && t.device().type() == at::kPrivateUse1 && t.dim() == 2
                 && t.size(0) == r && t.size(1) == c_eff
                 && t.is_contiguous(),
                    nm, " must be contiguous fp16/int8 [", r, ",", c,
                    "] or int4-packed [", r, ",", c/2, "]");
    };
    for (int64_t L = 0; L < nl; ++L) {
        chk(gate_packed[L], NI, h, "gate_packed");
        chk(up_packed[L],   NI, h, "up_packed");
        chk(down_packed[L], h, NI, "down_packed");
    }
    const char* grouped_env = std::getenv("RPU_LINGBOT2_GROUPED_EXPERTS");
    const bool grouped_requested =
        grouped_env != nullptr && grouped_env[0] == '1' && grouped_env[1] == '\0';
    if (grouped_requested) {
        const char* row_chunk_env = std::getenv("RPU_L2_RCHUNK");
        const int64_t row_chunk =
            row_chunk_env != nullptr ? (int64_t)std::atol(row_chunk_env) : 0;
        const int64_t required_rows = chunk_size_ * num_experts_;
        TORCH_CHECK(row_chunk >= required_rows,
            "LingBot2 grouped experts require one routing-scale call: set "
            "RPU_L2_RCHUNK >= ", required_rows, " (got ", row_chunk,
            "). The multi-call row-offset path is known numerically wrong; refusing "
            "to build it.");
    }
    packed_gate_.assign(gate_packed.begin(), gate_packed.end());
    packed_up_.assign(up_packed.begin(), up_packed.end());
    packed_down_.assign(down_packed.begin(), down_packed.end());
    // "quantized" = int8 OR int4; both need the down GEMM on acc16 and both bind scales.
    packed_w8a16_ = (gate_packed[0].scalar_type() == at::kChar
                  || gate_packed[0].scalar_type() == at::kByte);
    packed_gate_s_.clear(); packed_up_s_.clear(); packed_down_s_.clear();
    wtm_stage_ = at::zeros({nl, tp_rows_, num_experts_},
        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
    // Opt-in gate: activate the grouped path only when the env flag is set, so a
    // stray bind cannot silently change the default path (Gate A).
    grouped_experts_ = grouped_requested;
    if (grouped_experts_) {
        std::printf("LINGBOT2_GROUPED_EXPERTS\nCORE_SLICE_INTERLEAVED\nACCURACY_UNVALIDATED\n");
        std::fflush(stdout);
    }
    invalidate_model_state();   // Must remain the last statement.
}

// ─────────────────────────────────────────────────────────────────────────────
// build_layer_subgraph
//
// Structurally the CausalDecoderModel explicit-2D-mask (SEQUENTIAL) path — the
// same phase order, the same kernels, the same AdaRMS FiLM hooks — with the
// Phase 7-8 emit_mlp_pipeline call swapped for emit_moe_mlp. The base's
// bidirectional-no-mask (KV_FIRST) and fused-lm_head branches are NOT
// reproduced: the LingBot action expert always runs with an explicit 2D
// additive mask and never with a vocab head, so they are rejected loudly
// instead of being silently mis-emitted.
// ─────────────────────────────────────────────────────────────────────────────
void LingbotV2MoeExpertModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    TORCH_CHECK(num_experts_ > 0, "build_layer_subgraph before set_moe_weights");
    TORCH_CHECK(ctx().attention_mask.has_value() && !ctx().is_causal,
        "LingbotV2MoeExpertModel requires an explicit 2D additive mask "
        "(is_causal=false + attention_mask), matching the LingBot action expert.");
    TORCH_CHECK(!fuse_lm_head_, "LingbotV2MoeExpertModel does not support the fused lm_head");
    TORCH_CHECK(!has_mrope_, "LingbotV2MoeExpertModel expects plain 1D-RoPE (mrope_section=[])");
    TORCH_CHECK(deepstack_lang_layers_.empty(),
                "LingbotV2MoeExpertModel does not support DeepStack injection");
    TORCH_CHECK(chunk.len == chunk_size_,
        "chunk.len(", chunk.len, ") != set_moe_weights chunk_size(", chunk_size_, ")");

    const auto& lw = layer_weights_[layer_idx];
    const int64_t seq_len = chunk.len;
    const int64_t rope_base =
        (!has_mrope_ && rope_position_base_ >= 0) ? rope_position_base_ : ctx().position;
    const int64_t cos_sin_start = rope_base + chunk.offset;
    const int64_t h = hidden_size();
    const int64_t nq = num_q_heads(), nkv = num_kv_heads(), hd = head_dim();
    const int tp = attn_tp();
    const bool is_last_layer = (layer_idx == num_layers() - 1);

    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);
    }

    // ── DEBUG: capture the layer input (residual1) BEFORE any layer math, all layers. ──
    if (debug_cap_resid_) {
        c10::Half* p = lin_stage_.data_ptr<c10::Half>() + (int64_t)layer_idx * tp_rows_ * h;
        rpu_launch_spm_copy_ddr_dma(addr(0, "residual1"), p, seq_len * h);
    }

    // ── Phase 1: input RMSNorm (+ AdaRMS FiLM shift) ──
    emit_adarms_mut_refresh_input(layer_idx);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual1"), addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "norm_w"),   // adarms_: scale rides here
        seq_len, h, eps_);
    // DEBUG-ONLY: recompute the FiLM shift into a never-aliased slot BEFORE the real one.
    // Reads input_norm (written by the rmsnorm above), so the dependency on the normalisation
    // is explicit; writes innorm_dbg, which nothing else touches. The real shift below is
    // untouched and still reads the same input_norm, so the computation is unchanged.
    // Emits nothing unless RPU_L2_CAPTURE_INNORM=1.
    const bool cap_innorm = (debug_capture_innorm_ && layer_idx == 0 && adarms_);
    if (cap_innorm) {
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            layer_addr(layer_idx, 0, "input_shift"),
            addr(0, "input_norm"), addr(0, "innorm_dbg"),
            seq_len, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
    }
    if (adarms_) {
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            layer_addr(layer_idx, 0, "input_shift"),
            addr(0, "input_norm"), addr(0, "input_norm"),
            seq_len, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
    }
    if (cap_innorm)
        rpu_launch_spm_copy_ddr_dma(addr(0, "innorm_dbg"),
                                    innorm_.data_ptr<c10::Half>(), seq_len * h);

    // ── Phase 2: QKV linear (Qwen2.5 QKV bias fused via the per-layer SPM slot) ──
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "input_norm"), lw.q_w, addr(0, "q"),
        seq_len, nq * hd, h, /*partition=*/1, /*num_cores=*/tp,
        has_qkv_bias_ ? layer_addr(layer_idx, 0, "q_bias") : 0u,
        /*force_gemm=*/false, lw.q_ws);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "input_norm"), lw.k_w, addr(0, "k"),
        seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
        has_qkv_bias_ ? layer_addr(layer_idx, 0, "k_bias") : 0u,
        /*force_gemm=*/false, lw.k_ws);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "input_norm"), lw.v_w, addr(0, "v"),
        seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
        has_qkv_bias_ ? layer_addr(layer_idx, 0, "v_bias") : 0u,
        /*force_gemm=*/false, lw.v_ws);

    // ── Phase 3: RoPE (plain 1D; LingBot has no QK head-norm) ──
    const int64_t local_q_heads = nq / tp;
    const int64_t local_kv_heads = nkv / tp;
    TORCH_CHECK(!has_qk_norm_, "LingbotV2MoeExpertModel expects no QK head-norm (Qwen2.5)");
    // LingBot2's suffix is the fixed M=51/full-rotary case.  Use the M>1
    // partial_mrope tiling only here; the shared decoder's 1D-RoPE dispatch must
    // remain unchanged for other models.  cos_/sin_ are immutable model-owned
    // [max_seq, hd/2] tables and cos_sin_start preserves the compressed-prefix
    // RoPE base independently from the KV insertion offset.
    rpu_launch_partial_mrope_spm_kernel(
        addr(0, "q"), addr(0, "q"),
        cos_.data_ptr<c10::Half>(), sin_.data_ptr<c10::Half>(),
        cos_sin_start, seq_len, local_q_heads, hd,
        /*rotary_dim=*/hd, tp);
    rpu_launch_partial_mrope_spm_kernel(
        addr(0, "k"), addr(0, "k"),
        cos_.data_ptr<c10::Half>(), sin_.data_ptr<c10::Half>(),
        cos_sin_start, seq_len, local_kv_heads, hd,
        /*rotary_dim=*/hd, tp);

    // ── Phase 4: KV-cache insert + SDPA + O-proj (SDPA/KV-insert take typed
    //    SPM OFFSETS via addr_offset(), every other kernel takes addr()) ──
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    rpu_launch_insert_kcache_spm_unified(
        k_cache, ctx().position + chunk.offset, addr_offset("k").value, seq_len, nkv, hd, tp);
    rpu_launch_insert_vcache_spm_unified(
        v_cache, ctx().position + chunk.offset, addr_offset("v").value, seq_len, nkv, hd, tp);

    const int64_t kv_seq_len = chunk.kv_seq_len;
    const int mask_type = prepared_attn_mask_.mask_type;      // MASK_2D (4)
    const uint32_t mask_off = addr_offset("sdpa_mask").value;
    sdpa_dma_mask_to_spm(prepared_attn_mask_, mask_off, seq_len, kv_seq_len, tp);

    rpu_launch_sdpa_spm_dispatch(
        sdpa_kernel_, k_cache, v_cache, mask_type, c10::nullopt,
        addr_offset("q").value, addr_offset("output").value,
        addr_offset("sdpa_tmp").value, mask_off,
        seq_len, nq, nkv, hd, kv_seq_len, tp, NUM_CORES);

    rpu_launch_linear_spm_to_spm_acc16_kernel(
        addr(0, "output"), lw.o_w, addr(0, "oproj"),
        seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
        /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.o_ws);

    // ── Phase 5: attention reduce + residual ──
    const bool force_fast_allreduce =
        denoise_twostage_ && denoise_unroll_active_;
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "residual1"),
        addr(0, "residual2"), seq_len, h, tp, NUM_CORES,
        /*force_twostage=*/force_fast_allreduce,
        /*force_chunk_v2=*/force_fast_allreduce);

    // ── DEBUG: capture the post-attention residual (residual2 = layer_in + attn_out). ──
    if (debug_cap_resid_) {
        c10::Half* p = ar_stage_.data_ptr<c10::Half>() + (int64_t)layer_idx * tp_rows_ * h;
        rpu_launch_spm_copy_ddr_dma(addr(0, "residual2"), p, seq_len * h);
    }

    // ── Phase 6: post-attention RMSNorm (+ AdaRMS FiLM shift) ──
    emit_adarms_mut_refresh_post(layer_idx);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "residual2"), addr(0, "residual1"),
        layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, eps_);
    if (adarms_) {
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            layer_addr(layer_idx, 0, "post_shift"),
            addr(0, "residual1"), addr(0, "residual1"),
            seq_len, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
    }

    // ── Phase 7-8: ★ sparse-MoE MLP (replaces emit_mlp_pipeline) ──
    emit_moe_mlp(layer_idx, seq_len);

    // ── final_norm fuse (last layer only) ──
    if (is_last_layer) {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "residual1"),
            addr(0, "final_norm_w"), seq_len, h, eps_);
    }
    if (!ctx().output_to_spm && !denoise_unroll_) {
        emit_layer_output_dma(layer_idx, chunk);
    }
}

void LingbotV2MoeExpertModel::denoise_unroll_forward(
    const at::Tensor& x0_rpu, const at::Tensor& state_rpu,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const std::optional<at::Tensor>& attention_mask,
    at::Tensor& x_final, double dt, int64_t prefix_len,
    int64_t cos_sin_offset, int64_t num_steps) {
    TORCH_CHECK(denoise_unroll_,
                "denoise_unroll_forward before set_denoise_weights");
    TORCH_CHECK(num_steps == num_steps_,
                "num_steps(", num_steps, ") must match configured num_steps(",
                num_steps_, ")");
    TORCH_CHECK(prefix_len >= 0 && cos_sin_offset >= 0,
                "prefix_len/cos_sin_offset must be non-negative");
    const int64_t cs = chunk_size_;

    TORCH_CHECK(x0_rpu.dim() == 3 && x0_rpu.size(0) == 1
             && x0_rpu.size(1) == cs && x0_rpu.size(2) == action_dim_pad_
             && x0_rpu.scalar_type() == at::kHalf && x0_rpu.is_contiguous()
             && x0_rpu.device().type() == at::kPrivateUse1,
                "x0_rpu must be [1,", cs, ",", action_dim_pad_,
                "] contiguous fp16 RPU");
    TORCH_CHECK(state_rpu.dim() == 2 && state_rpu.size(0) == 1
             && state_rpu.size(1) == state_dim_pad_
             && state_rpu.scalar_type() == at::kHalf && state_rpu.is_contiguous()
             && state_rpu.device().type() == at::kPrivateUse1,
                "state_rpu must be [1,", state_dim_pad_,
                "] contiguous fp16 RPU");
    TORCH_CHECK(x_final.dim() == 3 && x_final.size(0) == 1
             && x_final.size(1) == cs && x_final.size(2) == action_dim_pad_
             && x_final.scalar_type() == at::kHalf && x_final.is_contiguous()
             && x_final.device().type() == at::kPrivateUse1,
                "x_final must be [1,", cs, ",", action_dim_pad_,
                "] contiguous fp16 RPU");
    TORCH_CHECK(attention_mask.has_value(),
                "denoise_unroll_forward requires the explicit suffix attention mask");
    TORCH_CHECK(dt != 0.0 && std::isfinite(dt),
                "dt must be a finite non-zero Euler coefficient");
    const c10::Half dt_half = c10::Half(static_cast<float>(dt));
    if (!denoise_dt_pinned_) {
        denoise_dt_ = dt_half;
        denoise_dt_pinned_ = true;
    } else {
        TORCH_CHECK(denoise_dt_ == dt_half,
                    "dt changed across replays (it is baked into the graph)");
    }

    if (!k_caches.empty()) {
        const at::Tensor& kc = k_caches[0];
        TORCH_CHECK(kc.dim() >= 6,
                    "denoise_unroll_forward: malformed k_cache");
        TORCH_CHECK(prefix_len + cs <= kc.size(1) * kc.size(5),
                    "prefix_len+suffix exceeds k_cache capacity");
    }

    denoise_x0_ref_ = x0_rpu;
    denoise_state_ref_ = state_rpu;
    denoise_final_ref_ = x_final;
    denoise_x0_src_base_ = ::rhino_lkn::RpuGetDevAddr(x0_rpu.data_ptr());
    denoise_state_src_base_ = ::rhino_lkn::RpuGetDevAddr(state_rpu.data_ptr());
    denoise_final_dst_base_ = ::rhino_lkn::RpuGetDevAddr(x_final.data_ptr());
    rpu_ddr_flush_force_sized(
        x0_rpu.data_ptr<c10::Half>(), x0_rpu.nbytes());
    rpu_ddr_flush_force_sized(
        state_rpu.data_ptr<c10::Half>(), state_rpu.nbytes());

    TORCH_CHECK(!denoise_unroll_active_,
                "denoise_unroll_forward does not support re-entry");
    denoise_unroll_active_ = true;
    auto clear_denoise_scope = c10::make_scope_exit([this] {
        denoise_unroll_active_ = false;
    });

    (void)CausalDecoderModel::forward(
        denoise_stage_, k_caches, v_caches, attention_mask,
        /*position=*/prefix_len, /*is_causal=*/false,
        /*position_ids=*/std::nullopt,
        /*deepstack_dense_visual_embeds=*/std::nullopt,
        /*rope_cos_il=*/std::nullopt, /*rope_sin_il=*/std::nullopt,
        cos_sin_offset);
}

}  // namespace v3

// =============================================================================
// Instance registry — ModelHandleRegistry<v3::LingbotV2MoeExpertModel>
// =============================================================================
using LingbotV2MoeRegistry = ModelHandleRegistry<v3::LingbotV2MoeExpertModel>;

namespace v3::lingbot_v2_moe_internal {

ExactSuffixProfileDescriptor describe_exact_suffix_spm_profile(
    int64_t handle, int64_t prefix_len) {
    auto* model = LingbotV2MoeRegistry::get(
        handle, "lingbot2_multiview_spm_z2_preflight");
    return model->describe_exact_suffix_spm_profile(prefix_len);
}

SpmPipelineComponentLayout prime_exact_suffix_spm_layout(
    int64_t handle, int64_t prefix_len,
    const ExactSuffixProfileDescriptor& expected) {
    auto* model = LingbotV2MoeRegistry::get(
        handle, "lingbot2_multiview_spm_z2_prepare");
    return model->prime_exact_suffix_spm_layout(prefix_len, expected);
}

}  // namespace v3::lingbot_v2_moe_internal

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================
int64_t rpu_lingbot_v2_moe_create() {
    return LingbotV2MoeRegistry::create();
}

void rpu_lingbot_v2_moe_destroy(int64_t handle) {
    LingbotV2MoeRegistry::destroy(handle, "rpu_lingbot_v2_moe_destroy");
}

// Mandatory setter order (V1 precedent): base set_weights -> set_moe_weights.
// The SHARED expert rides the base's dense gate_w/up_w/down_w lists with
// intermediate_size == shared_inter_pad (704 -> 768: a multiple of 128 for the
// 8-core col-swizzle; zero-padding is numerically exact for SwiGLU because
// silu(0)*0 == 0 under the EXP_INTER_PAD contract).
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
    double routed_scaling, int64_t chunk_size)
{
    auto* m = LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_set_weights");
    // Plain Qwen2 768-d decoder: no QK head-norm, QKV bias present, plain 1D-RoPE.
    m->CausalDecoderModel::set_weights(
        q_w, k_w, v_w, o_w,
        /*q_norm=*/{}, /*k_norm=*/{},
        input_norm, post_norm,
        shared_gate_w, shared_up_w, shared_down_w,
        cos, sin, final_norm_w,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, /*intermediate_size=*/shared_inter_pad,
        eps, /*use_silu=*/true,
        /*mrope_section=*/{}, /*deepstack_lang_layers=*/{},
        q_bias, k_bias, v_bias);
    m->set_moe_weights(router_gate_w, router_bias,
                       expert_gate_w, expert_up_w, expert_down_w,
                       num_experts, top_k, routed_inter, routed_scaling, chunk_size);
}

void rpu_lingbot_v2_moe_set_packed_weights(
    int64_t handle, at::TensorList gate_packed, at::TensorList up_packed,
    at::TensorList down_packed)
{
    LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_set_packed_weights")
        ->set_packed_expert_weights(gate_packed, up_packed, down_packed);
}

// DEBUG-ONLY packed-weight read-back. Returns the live DDR tensor the emit path
// actually hands to the GEMM launcher (packed_gate_/packed_up_/packed_down_), so a
// mismatch indicts the converter/swizzle/bind contract rather than the graph.
// Bind the scales that accompany grouped quantized weights. W8A16 uses a
// per-output-channel [N] scale. W4A16 pgrp uses the controller-striped latest
// ABI; dim 0 carries group_size metadata.
void v3::LingbotV2MoeExpertModel::set_packed_expert_scales(
    at::TensorList gate_scale, at::TensorList up_scale, at::TensorList down_scale) {
    const int64_t nl = num_layers();
    TORCH_CHECK(!packed_gate_.empty(), "set_packed_expert_scales must follow set_packed_expert_weights");
    TORCH_CHECK(packed_w8a16_, "set_packed_expert_scales: packed weights are fp16, not int8");
    TORCH_CHECK((int64_t)gate_scale.size() == nl && (int64_t)up_scale.size() == nl
             && (int64_t)down_scale.size() == nl, "scale lists must have size num_layers=", nl);
    const int64_t NI = num_experts_ * routed_inter_;
    const bool int4_pgrp = packed_gate_[0].scalar_type() == at::kByte;
    auto chk = [int4_pgrp](const at::Tensor& t, int64_t n, int64_t k,
                           int partition, const char* nm) {
        TORCH_CHECK(t.defined() && t.scalar_type() == at::kHalf
                 && t.device().type() == at::kPrivateUse1 && t.is_contiguous(),
                    nm, " must be contiguous fp16 RPU");
        if (int4_pgrp) {
            TORCH_CHECK(t.dim() == 2 &&
                            (t.size(0) == 32 || t.size(0) == 64 ||
                             t.size(0) == 128),
                        nm, " pgrp dim 0 must carry group_size 32/64/128, got ",
                        t.sizes());
            const int64_t group_size = t.size(0);
            TORCH_CHECK(k % NUM_CORES == 0 || partition == 1,
                        nm, " pgrp row K must be divisible by cores");
            TORCH_CHECK(n % NUM_CORES == 0 || partition == 0,
                        nm, " pgrp col N must be divisible by cores");
            const int64_t local_k = partition == 1 ? k : k / NUM_CORES;
            const int64_t local_n = partition == 1 ? n / NUM_CORES : n;
            TORCH_CHECK(local_k % group_size == 0,
                        nm, " local K must be divisible by group_size");
            const int64_t local_g = local_k / group_size;
            const int64_t expected = ((local_g + 3) / 4) *
                                     ((local_n + 63) / 64) * NUM_CORES * 4 * 64;
            TORCH_CHECK(t.numel() == expected,
                        nm, " controller-striped pgrp payload has ", t.numel(),
                        " elements, expected ", expected);
        } else {
            TORCH_CHECK(t.dim() == 1 && t.size(0) == n,
                        nm, " must be per-channel [N]=[", n, "], got ", t.sizes());
        }
    };
    for (int64_t L = 0; L < nl; ++L) {
        TORCH_CHECK((packed_gate_[L].scalar_type() == at::kByte) == int4_pgrp
                 && (packed_up_[L].scalar_type() == at::kByte) == int4_pgrp
                 && (packed_down_[L].scalar_type() == at::kByte) == int4_pgrp,
                    "packed expert quantization dtype must be consistent across layers and roles");
        chk(gate_scale[L], NI, hidden_size(), /*partition=*/1, "gate_scale");
        chk(up_scale[L], NI, hidden_size(), /*partition=*/1, "up_scale");
        chk(down_scale[L], hidden_size(), NI, /*partition=*/0, "down_scale");
    }
    auto retain_scale = [int4_pgrp](const at::Tensor& scale) {
        if (!int4_pgrp) return scale;
        constexpr int64_t kAlignmentBytes = NUM_CORES * 512;
        const uint64_t source_addr = RpuGetDevAddr(scale.data_ptr<c10::Half>());
        if (source_addr % kAlignmentBytes == 0) return scale;
        constexpr int64_t kAlignmentElements =
            kAlignmentBytes / static_cast<int64_t>(sizeof(c10::Half));
        auto storage = at::empty(
            {scale.numel() + kAlignmentElements}, scale.options());
        const uint64_t storage_addr =
            RpuGetDevAddr(storage.data_ptr<c10::Half>());
        const int64_t byte_offset = static_cast<int64_t>(
            (kAlignmentBytes - (storage_addr % kAlignmentBytes)) %
            kAlignmentBytes);
        TORCH_CHECK(byte_offset % static_cast<int64_t>(sizeof(c10::Half)) == 0,
                    "packed expert pgrp scale alignment is not FP16-addressable");
        auto aligned = storage
            .narrow(0, byte_offset / sizeof(c10::Half), scale.numel())
            .view(scale.sizes());
        aligned.copy_(scale);
        TORCH_CHECK(
            RpuGetDevAddr(aligned.data_ptr<c10::Half>()) % kAlignmentBytes == 0,
            "failed to align packed expert pgrp scale to ", kAlignmentBytes,
            " bytes");
        return aligned;
    };
    packed_gate_s_.clear();
    packed_up_s_.clear();
    packed_down_s_.clear();
    packed_gate_s_.reserve(nl);
    packed_up_s_.reserve(nl);
    packed_down_s_.reserve(nl);
    for (int64_t L = 0; L < nl; ++L) {
        packed_gate_s_.push_back(retain_scale(gate_scale[L]));
        packed_up_s_.push_back(retain_scale(up_scale[L]));
        packed_down_s_.push_back(retain_scale(down_scale[L]));
    }
    std::printf(int4_pgrp ? "LINGBOT2_GROUPED_W4A16_PGRP\n"
                          : "LINGBOT2_GROUPED_W8A16\n");
    std::fflush(stdout);
    invalidate_model_state();   // Must remain the last statement.
}

void rpu_lingbot_v2_moe_set_packed_scales(
    int64_t handle, at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale)
{
    LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_set_packed_scales")
        ->set_packed_expert_scales(gate_scale, up_scale, down_scale);
}

// Bind W8A16 scales for the attention + shared-expert linears. Additive: set_weights already
// stores those weights, and the emit path already passes lw.*_ws to the acc16 launcher, so this
// only fills the fields. Empty list for a role => that role stays fp16.
void v3::LingbotV2MoeExpertModel::set_base_scales(
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws, at::TensorList o_ws,
    at::TensorList gate_ws, at::TensorList up_ws, at::TensorList down_ws) {
    const int64_t nl = num_layers();
    TORCH_CHECK(!layer_weights_.empty(), "set_base_scales must follow set_weights");
    // A pointer-to-member would need LayerWeights' exact enclosing scope; a setter lambda is
    // simpler and scope-independent.
    auto chk = [&](at::TensorList src, const char* nm) {
        if (src.size() == 0) return false;
        TORCH_CHECK((int64_t)src.size() == nl, nm, " scale list must have size num_layers=", nl);
        for (int64_t L = 0; L < nl; ++L)
            TORCH_CHECK(src[L].defined() && src[L].scalar_type() == at::kHalf
                     && src[L].device().type() == at::kPrivateUse1,
                        nm, " scale must be fp16 RPU");
        return true;
    };
    const bool hq = chk(q_ws, "q"), hk = chk(k_ws, "k"), hv = chk(v_ws, "v"), ho = chk(o_ws, "o");
    const bool hg = chk(gate_ws, "shared_gate"), hu = chk(up_ws, "shared_up"),
               hd = chk(down_ws, "shared_down");
    for (int64_t L = 0; L < nl; ++L) {
        auto& lw = layer_weights_[L];
        if (hq) lw.q_ws = q_ws[L];
        if (hk) lw.k_ws = k_ws[L];
        if (hv) lw.v_ws = v_ws[L];
        if (ho) lw.o_ws = o_ws[L];
        if (hg) lw.gate_ws = gate_ws[L];
        if (hu) lw.up_ws = up_ws[L];
        if (hd) lw.down_ws = down_ws[L];
    }
    std::printf("LINGBOT2_BASE_W8A16\n"); std::fflush(stdout);
    invalidate_model_state();   // Must remain the last statement.
}

void rpu_lingbot_v2_moe_set_base_scales(
    int64_t handle, at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws, at::TensorList down_ws)
{
    LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_set_base_scales")
        ->set_base_scales(q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws);
}

at::Tensor v3::LingbotV2MoeExpertModel::debug_packed(int64_t which, int64_t layer) const {
    const std::vector<at::Tensor>* v =
        which == 0 ? &packed_gate_ : which == 1 ? &packed_up_ :
        which == 2 ? &packed_down_ : nullptr;
    TORCH_CHECK(v != nullptr, "debug_packed: which must be 0=gate, 1=up, 2=down, got ", which);
    TORCH_CHECK(!v->empty(), "debug_packed: no packed weights bound (grouped path never set up)");
    TORCH_CHECK(layer >= 0 && layer < (int64_t)v->size(),
                "debug_packed: layer ", layer, " out of range [0,", v->size(), ")");
    return (*v)[layer];
}

at::Tensor rpu_lingbot_v2_moe_debug_packed(int64_t handle, int64_t which, int64_t layer)
{
    // Default-OFF gate: without the flag this op throws, so it is unreachable from
    // the default path and cannot perturb it.
    const char* f = std::getenv("RPU_L2_DBG_PACKED");
    TORCH_CHECK(f != nullptr && f[0] == '1' && f[1] == '\0',
                "rpu_lingbot_v2_moe_debug_packed is DEBUG-ONLY: set RPU_L2_DBG_PACKED=1");
    return LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_packed")
        ->debug_packed(which, layer);
}

// REPLAY-safe per-Euler-step AdaRMS FiLM. The 10-step denoise loop is driven from
// Python: step 0 BUILDs the graph, steps 1..9 REPLAY it, with only the FiLM
// scale/shift contents refreshed per step (V1 default path).
void rpu_lingbot_v2_moe_set_adarms_step_mutable(
    int64_t handle,
    const at::Tensor& input_scale, const at::Tensor& input_shift,
    const at::Tensor& post_scale,  const at::Tensor& post_shift)
{
    LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_set_adarms_step_mutable")
        ->set_adarms_step_mutable(input_scale, input_shift, post_scale, post_shift);
}

// Bind the immutable Euler AdaRMS schedule once; forward selects the step.
void rpu_lingbot_v2_moe_bind_adarms_schedule(
    int64_t handle,
    const at::Tensor& input_scale, const at::Tensor& input_shift,
    const at::Tensor& post_scale,  const at::Tensor& post_shift)
{
    LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_bind_adarms_schedule")
        ->bind_adarms_schedule(input_scale, input_shift, post_scale, post_shift);
}

void rpu_lingbot_v2_moe_set_denoise_unroll(
    int64_t handle,
    const at::Tensor& state_w, const at::Tensor& state_b,
    const at::Tensor& wc, const at::Tensor& mo, const at::Tensor& mo_bias,
    const at::Tensor& op, const at::Tensor& op_bias,
    const at::Tensor& time_all,
    const at::Tensor& adarms_is, const at::Tensor& adarms_ish,
    const at::Tensor& adarms_ps, const at::Tensor& adarms_psh,
    int64_t state_dim, int64_t action_dim, int64_t state_dim_pad,
    int64_t action_dim_pad, int64_t num_steps) {
    LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_set_denoise_unroll")
        ->set_denoise_weights(
            state_w, state_b, wc, mo, mo_bias, op, op_bias, time_all,
            adarms_is, adarms_ish, adarms_ps, adarms_psh,
            state_dim, action_dim, state_dim_pad, action_dim_pad, num_steps);
}

void rpu_lingbot_v2_moe_denoise_unroll_forward(
    int64_t handle, const at::Tensor& x0_rpu, const at::Tensor& state_rpu,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const std::optional<at::Tensor>& attention_mask, at::Tensor x_final,
    double dt, int64_t prefix_len, int64_t cos_sin_offset,
    int64_t num_steps) {
    LingbotV2MoeRegistry::get(
        handle, "rpu_lingbot_v2_moe_denoise_unroll_forward")
        ->denoise_unroll_forward(
            x0_rpu, state_rpu, k_caches, v_caches, attention_mask, x_final,
            dt, prefix_len, cos_sin_offset, num_steps);
}

// cos_sin_offset decouples the RoPE position base from the KV-insert offset.
//
// WHY THIS IS REQUIRED (not an optimisation): the V2 action expert is rotated by
// the VLM's rotary_emb, and the suffix's RoPE positions continue from the
// *M-RoPE-compressed* prefix position, whereas the KV-insert offset is the
// *uncompressed* prefix length. With one 32x32-patch image plus 8 text tokens,
// the KV offset is S=264 while the RoPE base is p0=24. A single
// `position` cannot express both, so without this the action is silently rotated
// at the wrong positions.
//
//   position       -> KV-cache insert offset (unchanged)
//   cos_sin_offset -> RoPE table base;  -1 = "same as position" (back-compat)
//
// This uses CausalDecoderModel::forward's trailing cos_sin_offset parameter,
// which sets rope_position_base_ — already honoured by
// build_layer_subgraph's rope_base computation.
// DEBUG-ONLY: hand back the dense-soft router's device-computed [nl,E,Tp]
// routing weights so diagnostics can inspect shape, padding, and row sums on
// device data. Returns the live RPU-resident tensor; the caller copies to CPU.
at::Tensor rpu_lingbot_v2_moe_debug_rw_stage(int64_t handle)
{
    return LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_rw_stage")
        ->debug_rw_stage();
}

at::Tensor rpu_lingbot_v2_moe_debug_wtm_stage(int64_t handle)
{
    return LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_wtm_stage")
        ->debug_wtm_stage();
}

at::Tensor rpu_lingbot_v2_moe_debug_l0_out(int64_t handle)
{
    return LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_l0_out")->debug_l0_out();
}

at::Tensor rpu_lingbot_v2_moe_debug_innorm(int64_t handle)
{
    return LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_innorm")->debug_innorm();
}

at::Tensor rpu_lingbot_v2_moe_debug_gcap(int64_t handle)
{
    return LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_gcap")->debug_gcap();
}

at::Tensor rpu_lingbot_v2_moe_debug_acap(int64_t handle)
{
    return LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_acap")->debug_acap();
}

at::Tensor rpu_lingbot_v2_moe_debug_scap(int64_t handle){return LingbotV2MoeRegistry::get(handle,"s")->debug_scap();}
at::Tensor rpu_lingbot_v2_moe_debug_ccap(int64_t handle){return LingbotV2MoeRegistry::get(handle,"c")->debug_ccap();}

at::Tensor rpu_lingbot_v2_moe_debug_router_h(int64_t handle)
{
    at::Tensor t = LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_router_h")
        ->debug_router_h();
    TORCH_CHECK(t.defined(),
        "lingbot2: router-h capture is off. Set RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H=1 before "
        "building the model to populate it.");
    return t;
}

at::Tensor rpu_lingbot_v2_moe_debug_layer_in(int64_t handle)
{
    at::Tensor t = LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_layer_in")
        ->debug_layer_in();
    TORCH_CHECK(t.defined(),
        "lingbot2: layer_in capture is off. Set RPU_L2_CAP_RESID=1 before building the model.");
    return t;
}

at::Tensor rpu_lingbot_v2_moe_debug_attn_resid(int64_t handle)
{
    at::Tensor t = LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_debug_attn_resid")
        ->debug_attn_resid();
    TORCH_CHECK(t.defined(),
        "lingbot2: attn_resid capture is off. Set RPU_L2_CAP_RESID=1 before building the model.");
    return t;
}

at::Tensor rpu_lingbot_v2_moe_forward(
    int64_t handle, const at::Tensor& hidden_states,
    std::vector<at::Tensor> k_caches, std::vector<at::Tensor> v_caches,
    const std::optional<at::Tensor>& attention_mask, int64_t position,
    int64_t cos_sin_offset, int64_t adarms_schedule_step)
{
    auto* model = LingbotV2MoeRegistry::get(handle, "rpu_lingbot_v2_moe_forward");
    if (adarms_schedule_step >= 0) {
        model->select_adarms_schedule_step(adarms_schedule_step);
    } else {
        TORCH_CHECK(!model->has_bound_adarms_schedule(),
                    "lingbot_v2_moe_forward: a bound AdaRMS schedule requires "
                    "adarms_schedule_step >= 0");
    }
    return model->forward(hidden_states, k_caches, v_caches, attention_mask,
                          position, /*is_causal=*/false, /*position_ids=*/std::nullopt,
                          /*deepstack_dense_visual_embeds=*/std::nullopt,
                          /*rope_cos_il=*/std::nullopt, /*rope_sin_il=*/std::nullopt,
                          cos_sin_offset);
}
