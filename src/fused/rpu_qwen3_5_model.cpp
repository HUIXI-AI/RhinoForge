// rpu_qwen3_5_model.cpp — Qwen3.5 hybrid-attention fused model (v3).
// See rpu_qwen3_5_model.h for scope. Full-attention emission mirrors
// CausalDecoderModel::build_layer_subgraph (rpu_qwen3_model.cpp) for the
// non-M-RoPE, QK-norm path; GDN layers dispatch to build_gdn(decode) by chunk.len.
//
// ── FILE MAP (read order: main flow → mixers → buffers/config → setup → C-API) ──
//   MAIN FLOW
//     forward()               — Python entry's workhorse; drives run_all_layers
//     build_layer_subgraph()  — assembles ONE layer: input DMA → input_layernorm →
//                               token mixer → shared post_norm + MLP + output
//   TOKEN MIXERS (read normed "input_norm"; end with all_reduce that adds h_in residual)
//     build_full_attention()  — QK-norm + (partial M-)RoPE + SDPA + gated o_proj
//     build_gdn()             — unified GDN mixer; if(decode) splits recurrent (seq=1)
//                               vs chunked delta-rule (prefill, L>1); shared setup/proj/tail
//     emit_mlp_and_output()   — shared post-mixer tail (post_norm + MLP + final_norm + out)
//   SPM LAYOUT       declare_buffers() — full-attn + the active GDN decode/prefill set
//   GRAPH-PLAN HOOKS static_config / dynamic_config / subclass_chunk_size_valid
//   SETUP            set_weights() (store swizzled weights)
//   C-API (bottom)   rpu_qwen3_5_* free funcs — Python handle ↔ object bridges
#include "rpu_qwen3_5_model.h"
#include "rpu_qwen3_5_spm_z2.h"

#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_spm_buffers.h"
#include "rpu_spm_pipeline.h"
#include "rpu_helpers.h"
#include "rpu_profile.h"         // rpu_ddr_flush
#include "rpu_runtime_state.h"   // diagnostic-only hidden-state export

#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>     // std::sqrt (GDN qk_scale)
#include <limits>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace {

// PARTIAL_MROPE encodes DDR bases in 256-byte units (addr >> 8). The caching
// allocator guarantees only 32-byte alignment, so retain an aligned view when
// needed instead of silently truncating address bits in the launcher.
at::Tensor keep_256b_aligned_rpu_copy(const at::Tensor& src,
                                      const char* name) {
    TORCH_CHECK(src.defined() && src.device().type() == at::kPrivateUse1
                && src.scalar_type() == at::kHalf && src.is_contiguous(),
                name, " must be a contiguous FP16 RPU tensor");
    const uint64_t src_addr = RpuGetDevAddr(src.data_ptr<c10::Half>());
    if ((src_addr & 0xFF) == 0)
        return src;

    constexpr int64_t kAlignmentBytes = 256;
    constexpr int64_t kAlignmentElems =
        kAlignmentBytes / static_cast<int64_t>(sizeof(c10::Half));
    auto storage = at::empty({src.numel() + kAlignmentElems}, src.options());
    const uint64_t storage_addr =
        RpuGetDevAddr(storage.data_ptr<c10::Half>());
    const int64_t byte_offset = static_cast<int64_t>(
        (kAlignmentBytes - (storage_addr & (kAlignmentBytes - 1)))
        & (kAlignmentBytes - 1));
    TORCH_CHECK(byte_offset % static_cast<int64_t>(sizeof(c10::Half)) == 0,
                name, " allocator returned an address incompatible with FP16 alignment");
    auto aligned = storage.narrow(
        0, byte_offset / static_cast<int64_t>(sizeof(c10::Half)), src.numel()
    ).view(src.sizes());
    aligned.copy_(src);
    TORCH_CHECK((RpuGetDevAddr(aligned.data_ptr<c10::Half>()) & 0xFF) == 0,
                name, " failed to create a 256-byte aligned RPU copy");
    return aligned;
}

}  // namespace

namespace v3 {

Qwen3_5Model::Qwen3_5Model() = default;
Qwen3_5Model::~Qwen3_5Model() = default;

void Qwen3_5Model::set_chunk_size_cap(int64_t cap) {
    TORCH_CHECK(cap == 0 || cap >= 64,
                "Qwen3.5 text chunk cap must be 0 or at least 64, got ", cap);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 text chunk cap must be set before the first forward");
    chunk_size_cap_ = cap;
    set_chunk_size_override(0);
    invalidate_model_state();
}

void Qwen3_5Model::set_prefill_chunk_size(int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0
                    || (chunk_size >= 64 && chunk_size % 64 == 0),
                "Qwen3.5 text chunk size must be 0 (auto) or a positive "
                "multiple of 64, got ", chunk_size);
    // Prefill is a bounded raw-Graph one-shot, not a retained GraphCache entry.
    // It is therefore safe for the certification harness to switch this exact
    // request between independent cache-reset legs on one loaded model.  The
    // public execution configuration remains cold/per-instance; this internal
    // setter invalidates the FMB layout before the next one-shot build.
    set_chunk_size_override(chunk_size);
    invalidate_model_state();
}

void Qwen3_5Model::set_linear_acc32(bool enabled) {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 linear accumulation mode must be set before the first forward");
    linear_acc32_ = enabled;
    invalidate_model_state();
}

void Qwen3_5Model::set_fast_replay(bool enabled) {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 fast replay must be set before the first forward");
    fast_replay_enabled_ = enabled;
    invalidate_model_state();
}

void Qwen3_5Model::enable_action_mode() {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 action mode must be enabled before the first forward");
    TORCH_CHECK(num_layers() > 0,
                "Qwen3.5 action mode requires set_weights first");
    TORCH_CHECK(!has_gdn_,
                "Qwen3.5 action mode supports all-full-attention experts only");
    action_mode_ = true;
    action_prefix_lens_.assign(num_layers(), 0);
    set_chunk_size_override(0);
    invalidate_model_state();
}

void Qwen3_5Model::set_action_io_weights(
    const at::Tensor& input_w, const at::Tensor& input_b,
    const at::Tensor& output_w, const at::Tensor& output_b,
    int64_t action_dim, int64_t action_dim_pad, int64_t action_len) {
    TORCH_CHECK(action_mode_,
                "Qwen3.5 action I/O weights require action mode");
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 action I/O weights must be set before the first forward");
    TORCH_CHECK(action_dim > 0 && action_dim_pad >= action_dim
                && action_dim_pad % 16 == 0,
                "Qwen3.5 action dimensions must satisfy 0 < dim <= padded dim "
                "and padded dim must be a multiple of 16");
    TORCH_CHECK(action_len > 0 && action_len % 16 == 0,
                "Qwen3.5 action length must be a positive multiple of 16");
    auto check_rpu_half = [](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.defined()
                    && tensor.device().type() == at::kPrivateUse1
                    && tensor.scalar_type() == at::kHalf
                    && tensor.is_contiguous(),
                    name, " must be contiguous FP16 on RPU");
    };
    check_rpu_half(input_w, "Qwen3.5 action input weight");
    check_rpu_half(input_b, "Qwen3.5 action input bias");
    check_rpu_half(output_w, "Qwen3.5 action output weight");
    check_rpu_half(output_b, "Qwen3.5 action output bias");
    TORCH_CHECK(input_w.numel() == hidden_size() * action_dim_pad
                && input_b.numel() == hidden_size(),
                "Qwen3.5 action input projection shape mismatch");
    TORCH_CHECK(output_w.numel() == action_dim_pad * hidden_size()
                && output_b.numel() == action_dim_pad,
                "Qwen3.5 action output projection shape mismatch");

    action_input_w_ = input_w;
    action_input_b_ = input_b;
    action_output_w_ = output_w;
    action_output_b_ = output_b;
    action_dim_ = action_dim;
    action_dim_pad_ = action_dim_pad;
    action_len_ = action_len;
    action_loop_active_ = false;
    action_rtc_active_ = false;
    action_rtc_prefix_len_ = 0;
    action_rtc_prefix_ref_ = at::Tensor();
    action_rtc_prefix_live_base_ = 0;
    action_trace_active_ = false;
    action_trace_pre_action_ref_ = at::Tensor();
    action_trace_action_embed_ref_ = at::Tensor();
    action_trace_final_hidden_ref_ = at::Tensor();
    action_trace_velocity_ref_ = at::Tensor();
    action_trace_post_action_ref_ = at::Tensor();
    action_trace_pre_action_live_base_ = 0;
    action_trace_action_embed_live_base_ = 0;
    action_trace_final_hidden_live_base_ = 0;
    action_trace_velocity_live_base_ = 0;
    action_trace_post_action_live_base_ = 0;
    action_num_steps_ = 1;
    action_euler_scale_pinned_ = false;
    auto options = input_w.options();
    action_hidden_stage_ = at::empty(
        {1, action_len_, hidden_size()}, options);
    action_velocity_stage_ = at::empty(
        {1, action_len_, action_dim_pad_}, options);
    invalidate_model_state();
}

int64_t Qwen3_5Model::resolve_prefill_chunk_size(int64_t execution_len) {
    TORCH_CHECK(execution_len >= 64 && execution_len % 64 == 0,
                "Qwen3.5 prefill execution length must be a positive multiple of 64, got ",
                execution_len);
    return resolve_chunk_size_for_shape(
        execution_len, /*position=*/0, std::nullopt, /*is_causal=*/true);
}

SpmPipelineComponentLayout Qwen3_5Model::prepare_z2_layout(
    int64_t execution_len,
    int64_t real_len) {
    constexpr int64_t kCanaryExecutionLen = 128;
    constexpr int64_t kCanaryRealLen = 72;
    constexpr int64_t kCanaryHidden = 2048;
    TORCH_CHECK(!z2_bound_,
                "Qwen3.5 text Z2: cannot prepare while a lease is active");
    TORCH_CHECK(execution_len == kCanaryExecutionLen &&
                    real_len == kCanaryRealLen,
                "Qwen3.5 text Z2 canary accepts only execution/real=(128,72), got (",
                execution_len, ",", real_len, ")");
    TORCH_CHECK(hidden_size() == kCanaryHidden,
                "Qwen3.5 text Z2 canary accepts only the 2B hidden size 2048, got ",
                hidden_size());
    TORCH_CHECK(valid_prefill_len_ == real_len,
                "Qwen3.5 text Z2: valid prefill length must be set to ", real_len,
                " before prepare; got ", valid_prefill_len_);
    TORCH_CHECK(has_mrope_ && prefill_cos_.defined() && prefill_sin_.defined() &&
                    prefill_cos_.dim() == 2 &&
                    prefill_cos_.sizes() == prefill_sin_.sizes() &&
                    prefill_cos_.size(0) == execution_len &&
                    prefill_cos_.size(1) == rotary_dim_ / 2,
                "Qwen3.5 text Z2: stable prefill M-RoPE tables must be [128,",
                rotary_dim_ / 2, "] before prepare");

    const int64_t resolved = resolve_prefill_chunk_size(execution_len);
    LayoutContext layout;
    layout.chunk_size = resolved;
    layout.max_kv_seq_len = execution_len;
    layout.num_layers = num_layers();
    layout.use_attn_mask = false;
    layout.is_causal = true;
    auto prepared = prepare_spm_pipeline_component(layout);

    z2_prepared_execution_len_ = execution_len;
    z2_prepared_real_len_ = real_len;
    z2_prefill_cos_addr_ =
        ::rhino_lkn::RpuGetDevAddr(prefill_cos_.data_ptr<c10::Half>());
    z2_prefill_sin_addr_ =
        ::rhino_lkn::RpuGetDevAddr(prefill_sin_.data_ptr<c10::Half>());
    return prepared;
}

SpmDense2DSpec Qwen3_5Model::z2_storage_spec(int64_t execution_len) const {
    TORCH_CHECK(execution_len == 128,
                "Qwen3.5 text Z2 canary storage requires 128 rows, got ",
                execution_len);
    TORCH_CHECK(hidden_size() == 2048,
                "Qwen3.5 text Z2 canary storage requires hidden size 2048, got ",
                hidden_size());
    SpmDense2DSpec spec;
    spec.rows = execution_len;
    spec.cols = hidden_size();
    spec.validate();
    return spec;
}

void Qwen3_5Model::validate_z2_prefill_contract() const {
    TORCH_CHECK(z2_prepared_execution_len_ == 128 &&
                    z2_prepared_real_len_ == 72,
                "Qwen3.5 text Z2: fixed prefill layout was not prepared");
    TORCH_CHECK(valid_prefill_len_ == z2_prepared_real_len_,
                "Qwen3.5 text Z2: valid prefill length drifted after prepare");
    TORCH_CHECK(prefill_cos_.defined() && prefill_sin_.defined() &&
                    prefill_cos_.dim() == 2 &&
                    prefill_cos_.sizes() == prefill_sin_.sizes() &&
                    prefill_cos_.size(0) == z2_prepared_execution_len_ &&
                    prefill_cos_.size(1) == rotary_dim_ / 2,
                "Qwen3.5 text Z2: prefill M-RoPE shape drifted after prepare");
    TORCH_CHECK(
        ::rhino_lkn::RpuGetDevAddr(prefill_cos_.data_ptr<c10::Half>()) ==
                z2_prefill_cos_addr_ &&
            ::rhino_lkn::RpuGetDevAddr(prefill_sin_.data_ptr<c10::Half>()) ==
                z2_prefill_sin_addr_,
        "Qwen3.5 text Z2: prefill M-RoPE address drifted after prepare");
}

void Qwen3_5Model::adopt_z2_layout(
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    validate_z2_prefill_contract();
    TORCH_CHECK(!z2_bound_,
                "Qwen3.5 text Z2: binding is already active");
    adopt_spm_pipeline_component(lease, scratch);
}

void Qwen3_5Model::bind_z2_port(
    const SpmPipelineLease& lease,
    const SpmPortView& port) {
    validate_z2_prefill_contract();
    TORCH_CHECK(!z2_bound_,
                "Qwen3.5 text Z2: binding is already active");
    const auto expected = z2_storage_spec(z2_prepared_execution_len_);
    TORCH_CHECK(port.spec() == expected,
                "Qwen3.5 text Z2: full storage port spec mismatch");
    z2_input_port_addr_ = port.resolve_physical_addr(/*core=*/0, lease);
    z2_epoch_ = lease.epoch();
    z2_plan_hash_ = lease.plan_hash();
    z2_bound_ = true;
}

void Qwen3_5Model::validate_z2_layout(
    const SpmPipelineLease& lease) const {
    validate_z2_prefill_contract();
    TORCH_CHECK(z2_bound_, "Qwen3.5 text Z2: no active binding");
    TORCH_CHECK(z2_epoch_ == lease.epoch() &&
                    z2_plan_hash_ == lease.plan_hash(),
                "Qwen3.5 text Z2: stale lease binding");
    validate_spm_pipeline_component(lease);
}

void Qwen3_5Model::clear_z2_layout(uint64_t epoch, uint64_t plan_hash) {
    if (z2_bound_) {
        TORCH_CHECK(z2_epoch_ == epoch && z2_plan_hash_ == plan_hash,
                    "Qwen3.5 text Z2: stale clear token");
    }
    release_spm_pipeline_component(epoch, plan_hash);
    z2_input_port_addr_ = 0;
    z2_epoch_ = 0;
    z2_plan_hash_ = 0;
    z2_bound_ = false;
}

at::Tensor Qwen3_5Model::forward_action(
    const at::Tensor& hidden_states,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens) {
    TORCH_CHECK(action_mode_,
                "Qwen3.5 action forward requires enable_action_mode()");
    TORCH_CHECK(hidden_states.dim() == 3 && hidden_states.size(0) == 1,
                "Qwen3.5 action hidden_states must be [1, action_len, hidden]");
    const int64_t seq_len = hidden_states.size(1);
    TORCH_CHECK(seq_len > 0 && seq_len % 16 == 0,
                "Qwen3.5 action_len must be a positive multiple of 16, got ",
                seq_len);
    TORCH_CHECK(adaptive_mod.defined() && adaptive_mod.dim() == 2
                && adaptive_mod.size(0) == 2 * num_layers() + 1
                && adaptive_mod.size(1) == 3 * hidden_size(),
                "Qwen3.5 action adaptive_mod must be [",
                2 * num_layers() + 1, ", ", 3 * hidden_size(), "], got ",
                adaptive_mod.sizes());
    TORCH_CHECK(adaptive_mod.device().type() == at::kPrivateUse1
                && adaptive_mod.scalar_type() == at::kHalf
                && adaptive_mod.is_contiguous(),
                "Qwen3.5 action adaptive_mod must be contiguous FP16 on RPU");
    TORCH_CHECK(static_cast<int64_t>(prefix_lens.size()) == num_layers(),
                "Qwen3.5 action prefix_lens must have ", num_layers(),
                " entries, got ", prefix_lens.size());
    TORCH_CHECK(prefill_cos_.defined() && prefill_sin_.defined()
                && prefill_cos_.size(0) >= seq_len,
                "Qwen3.5 action requires set_prefill_rope with at least ",
                seq_len, " rows before forward");

    action_prefix_lens_.assign(prefix_lens.begin(), prefix_lens.end());
    int64_t max_prefix = 0;
    for (int64_t i = 0; i < num_layers(); ++i) {
        const int64_t prefix = action_prefix_lens_[i];
        TORCH_CHECK(prefix >= 0,
                    "Qwen3.5 action prefix_lens[", i, "] must be non-negative");
        TORCH_CHECK(i < static_cast<int64_t>(k_caches.size())
                    && k_caches[i].defined() && k_caches[i].dim() == 7,
                    "Qwen3.5 action K cache[", i, "] must be a defined 7D tensor");
        const int64_t capacity = k_caches[i].size(1) * 16;
        TORCH_CHECK(prefix + seq_len <= capacity,
                    "Qwen3.5 action cache capacity exceeded at layer ", i,
                    ": prefix=", prefix, " action_len=", seq_len,
                    " capacity=", capacity);
        max_prefix = std::max(max_prefix, prefix);
    }

    adaptive_mod_ref_ = adaptive_mod;
    adaptive_mod_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(adaptive_mod.data_ptr<c10::Half>());
    rpu_ddr_flush_force(adaptive_mod.data_ptr<c10::Half>());
    if (action_rtc_active_) {
        action_rtc_active_ = false;
        action_rtc_prefix_len_ = 0;
        action_rtc_prefix_ref_ = at::Tensor();
        action_rtc_prefix_live_base_ = 0;
        invalidate_model_state();
    }
    action_step_active_ = false;
    fast_replay_active_ = fast_replay_enabled_;

    // Bidirectional action self-attention requires the entire suffix to be
    // inserted before SDPA; force one chunk for this fixed-horizon forward.
    set_chunk_size_override(seq_len);
    return run_all_layers(
        hidden_states, k_caches, v_caches, std::nullopt,
        max_prefix, /*is_causal=*/false);
}

at::Tensor Qwen3_5Model::forward_action_step(
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor& action_out,
    double delta_t,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens,
    int64_t num_steps) {
    return forward_action_step_impl(
        action, action_keep_mask, action_out,
        /*action_prefix=*/nullptr, /*action_prefix_len=*/0,
        /*trace=*/nullptr,
        delta_t, adaptive_mod, k_caches, v_caches, prefix_lens, num_steps);
}

at::Tensor Qwen3_5Model::forward_action_rtc_step(
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor& action_out,
    const at::Tensor& action_prefix,
    int64_t action_prefix_len,
    double delta_t,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens,
    int64_t num_steps) {
    return forward_action_step_impl(
        action, action_keep_mask, action_out,
        &action_prefix, action_prefix_len,
        /*trace=*/nullptr,
        delta_t, adaptive_mod, k_caches, v_caches, prefix_lens, num_steps);
}

at::Tensor Qwen3_5Model::forward_action_rtc_trace(
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor& action_out,
    const at::Tensor& action_prefix,
    int64_t action_prefix_len,
    double delta_t,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens,
    at::Tensor& trace_pre_action,
    at::Tensor& trace_action_embed,
    at::Tensor& trace_final_hidden,
    at::Tensor& trace_velocity,
    at::Tensor& trace_post_action) {
    ActionRtcTraceOutputs trace{
        trace_pre_action,
        trace_action_embed,
        trace_final_hidden,
        trace_velocity,
        trace_post_action,
    };
    return forward_action_step_impl(
        action, action_keep_mask, action_out,
        &action_prefix, action_prefix_len, &trace,
        delta_t, adaptive_mod, k_caches, v_caches, prefix_lens,
        /*num_steps=*/10);
}

at::Tensor Qwen3_5Model::forward_action_step_impl(
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor& action_out,
    const at::Tensor* action_prefix,
    int64_t action_prefix_len,
    const ActionRtcTraceOutputs* trace,
    double delta_t,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens,
    int64_t num_steps) {
    TORCH_CHECK(action_mode_ && action_input_w_.defined(),
                "Qwen3.5 action step requires action mode and I/O weights");
    TORCH_CHECK(num_steps == 1 || num_steps == 10,
                "Qwen3.5 action supports one diagnostic step or the canonical "
                "10-step Euler loop, got ", num_steps);
    const bool loop_active = num_steps > 1;
    const bool rtc_active = action_prefix != nullptr;
    const bool trace_active = trace != nullptr;
    TORCH_CHECK(!rtc_active || loop_active,
                "Qwen3.5 RTC action requires the canonical unrolled loop");
    TORCH_CHECK(!trace_active || rtc_active,
                "Qwen3.5 action trace is available only for the RTC loop");
    if (action_loop_active_ != loop_active || action_num_steps_ != num_steps
            || action_rtc_active_ != rtc_active
            || action_rtc_prefix_len_ != action_prefix_len
            || action_trace_active_ != trace_active) {
        action_loop_active_ = loop_active;
        action_num_steps_ = num_steps;
        action_rtc_active_ = rtc_active;
        action_rtc_prefix_len_ = action_prefix_len;
        action_trace_active_ = trace_active;
        invalidate_model_state();
    }
    TORCH_CHECK(action.dim() == 3 && action.size(0) == 1
                && action.size(1) == action_len_
                && action.size(2) == action_dim_pad_
                && action.device().type() == at::kPrivateUse1
                && action.scalar_type() == at::kHalf
                && action.is_contiguous(),
                "Qwen3.5 action step input must be [1, ", action_len_, ", ",
                action_dim_pad_, "] contiguous FP16 on RPU");
    TORCH_CHECK(action_keep_mask.dim() == 1
                && action_keep_mask.size(0) == action_dim_pad_
                && action_keep_mask.device().type() == at::kPrivateUse1
                && action_keep_mask.scalar_type() == at::kHalf
                && action_keep_mask.is_contiguous(),
                "Qwen3.5 action keep mask must be [", action_dim_pad_,
                "] contiguous FP16 on RPU");
    TORCH_CHECK(action_out.sizes() == action.sizes()
                && action_out.device().type() == at::kPrivateUse1
                && action_out.scalar_type() == at::kHalf
                && action_out.is_contiguous(),
                "Qwen3.5 action step output must match the input shape and be "
                "contiguous FP16 on RPU");
    TORCH_CHECK(action_out.data_ptr() != action.data_ptr(),
                "Qwen3.5 action step requires distinct input/output buffers");
    if (rtc_active) {
        TORCH_CHECK(action_prefix_len > 0 && action_prefix_len < action_len_,
                    "Qwen3.5 RTC action prefix length must be in [1, ",
                    action_len_ - 1, "], got ", action_prefix_len);
        TORCH_CHECK(action_prefix->dim() == 3
                    && action_prefix->size(0) == 1
                    && action_prefix->size(1) == action_prefix_len
                    && action_prefix->size(2) == action_dim_pad_
                    && action_prefix->device().type() == at::kPrivateUse1
                    && action_prefix->scalar_type() == at::kHalf
                    && action_prefix->is_contiguous(),
                    "Qwen3.5 RTC action prefix must be [1, ",
                    action_prefix_len, ", ", action_dim_pad_,
                    "] contiguous FP16 on RPU");
        TORCH_CHECK(action_prefix->data_ptr() != action.data_ptr()
                    && action_prefix->data_ptr() != action_out.data_ptr(),
                    "Qwen3.5 RTC action prefix must use a distinct buffer");
    } else {
        TORCH_CHECK(action_prefix_len == 0,
                    "Qwen3.5 non-RTC action cannot carry a prefix length");
    }
    if (trace_active) {
        auto check_trace = [this, num_steps](
                               const at::Tensor& tensor,
                               const char* name,
                               int64_t width) {
            TORCH_CHECK(tensor.dim() == 4
                        && tensor.size(0) == num_steps
                        && tensor.size(1) == 1
                        && tensor.size(2) == action_len_
                        && tensor.size(3) == width
                        && tensor.device().type() == at::kPrivateUse1
                        && tensor.scalar_type() == at::kHalf
                        && tensor.is_contiguous(),
                        name, " must be [", num_steps, ", 1, ",
                        action_len_, ", ", width,
                        "] contiguous FP16 on RPU");
        };
        check_trace(trace->pre_action,
                    "Qwen3.5 RTC pre-action trace", action_dim_pad_);
        check_trace(trace->action_embed,
                    "Qwen3.5 RTC action-embed trace", hidden_size());
        check_trace(trace->final_hidden,
                    "Qwen3.5 RTC final-hidden trace", hidden_size());
        check_trace(trace->velocity,
                    "Qwen3.5 RTC velocity trace", action_dim_pad_);
        check_trace(trace->post_action,
                    "Qwen3.5 RTC post-action trace", action_dim_pad_);
        const std::array<const void*, 9> distinct_ptrs = {
            action.data_ptr(),
            action_keep_mask.data_ptr(),
            action_out.data_ptr(),
            action_prefix->data_ptr(),
            trace->pre_action.data_ptr(),
            trace->action_embed.data_ptr(),
            trace->final_hidden.data_ptr(),
            trace->velocity.data_ptr(),
            trace->post_action.data_ptr(),
        };
        for (size_t i = 0; i < distinct_ptrs.size(); ++i) {
            for (size_t j = i + 1; j < distinct_ptrs.size(); ++j) {
                TORCH_CHECK(
                    distinct_ptrs[i] != distinct_ptrs[j],
                    "Qwen3.5 RTC trace destinations and action inputs must "
                    "use distinct buffers");
            }
        }
    }
    TORCH_CHECK(delta_t > 0.0,
                "Qwen3.5 action Euler delta_t must be positive");
    const c10::Half euler_scale =
        c10::Half(-static_cast<float>(delta_t));
    if (!action_euler_scale_pinned_) {
        action_euler_scale_ = euler_scale;
        action_euler_scale_pinned_ = true;
    } else {
        TORCH_CHECK(action_euler_scale_ == euler_scale,
                    "Qwen3.5 action Euler delta_t changed after graph build");
    }
    const int64_t modulation_rows = 2 * num_layers() + 1;
    const int64_t modulation_width = 3 * hidden_size();
    const bool modulation_shape_ok = adaptive_mod.defined() &&
        ((!loop_active && !rtc_active && adaptive_mod.dim() == 2
          && adaptive_mod.size(0) == modulation_rows
          && adaptive_mod.size(1) == modulation_width)
         || (loop_active && !rtc_active && adaptive_mod.dim() == 3
             && adaptive_mod.size(0) == num_steps
             && adaptive_mod.size(1) == modulation_rows
             && adaptive_mod.size(2) == modulation_width)
         || (loop_active && rtc_active && adaptive_mod.dim() == 4
             && adaptive_mod.size(0) == num_steps
             && adaptive_mod.size(1) == modulation_rows
             && adaptive_mod.size(2) == 2
             && adaptive_mod.size(3) == modulation_width));
    TORCH_CHECK(modulation_shape_ok
                && adaptive_mod.device().type() == at::kPrivateUse1
                && adaptive_mod.scalar_type() == at::kHalf
                && adaptive_mod.is_contiguous(),
                "Qwen3.5 action modulation must be [", modulation_rows,
                ", ", modulation_width, "] for one step, [", num_steps,
                ", ", modulation_rows, ", ", modulation_width,
                "] for an unrolled loop, or [", num_steps, ", ",
                modulation_rows, ", 2, ", modulation_width,
                "] for RTC");
    TORCH_CHECK(static_cast<int64_t>(prefix_lens.size()) == num_layers(),
                "Qwen3.5 action step prefix_lens must have ", num_layers(),
                " entries, got ", prefix_lens.size());
    TORCH_CHECK(prefill_cos_.defined() && prefill_sin_.defined()
                && prefill_cos_.size(0) >= action_len_,
                "Qwen3.5 action step requires prefill RoPE with at least ",
                action_len_, " rows");

    action_prefix_lens_.assign(prefix_lens.begin(), prefix_lens.end());
    int64_t max_prefix = 0;
    for (int64_t i = 0; i < num_layers(); ++i) {
        const int64_t prefix = action_prefix_lens_[i];
        TORCH_CHECK(prefix >= 0,
                    "Qwen3.5 action step prefix_lens[", i,
                    "] must be non-negative");
        TORCH_CHECK(i < static_cast<int64_t>(k_caches.size())
                    && i < static_cast<int64_t>(v_caches.size())
                    && k_caches[i].defined() && k_caches[i].dim() == 7
                    && v_caches[i].defined() && v_caches[i].dim() == 7,
                    "Qwen3.5 action step K/V cache[", i,
                    "] must be defined 7D tensors");
        const int64_t capacity = k_caches[i].size(1) * 16;
        TORCH_CHECK(prefix + action_len_ <= capacity,
                    "Qwen3.5 action step cache capacity exceeded at layer ", i,
                    ": prefix=", prefix, " action_len=", action_len_,
                    " capacity=", capacity);
        max_prefix = std::max(max_prefix, prefix);
    }

    adaptive_mod_ref_ = adaptive_mod;
    adaptive_mod_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(adaptive_mod.data_ptr<c10::Half>());
    rpu_ddr_flush_force(adaptive_mod.data_ptr<c10::Half>());
    action_input_ref_ = action;
    action_input_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(action.data_ptr<c10::Half>());
    rpu_ddr_flush_force(action.data_ptr<c10::Half>());
    action_keep_mask_ref_ = action_keep_mask;
    action_keep_mask_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(action_keep_mask.data_ptr<c10::Half>());
    rpu_ddr_flush_force(action_keep_mask.data_ptr<c10::Half>());
    action_output_ref_ = action_out;
    action_output_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(action_out.data_ptr<c10::Half>());
    rpu_ddr_flush_force(action_out.data_ptr<c10::Half>());
    if (rtc_active) {
        action_rtc_prefix_ref_ = *action_prefix;
        action_rtc_prefix_live_base_ = ::rhino_lkn::RpuGetDevAddr(
            action_prefix->data_ptr<c10::Half>());
        rpu_ddr_flush_force(action_prefix->data_ptr<c10::Half>());
    } else {
        action_rtc_prefix_ref_ = at::Tensor();
        action_rtc_prefix_live_base_ = 0;
    }
    action_trace_active_ = trace_active;
    if (trace_active) {
        auto bind_trace = [](at::Tensor& ref, uint64_t& live_base,
                             at::Tensor& tensor) {
            ref = tensor;
            live_base = ::rhino_lkn::RpuGetDevAddr(
                tensor.data_ptr<c10::Half>());
            rpu_ddr_flush_force(tensor.data_ptr<c10::Half>());
        };
        bind_trace(
            action_trace_pre_action_ref_,
            action_trace_pre_action_live_base_, trace->pre_action);
        bind_trace(
            action_trace_action_embed_ref_,
            action_trace_action_embed_live_base_, trace->action_embed);
        bind_trace(
            action_trace_final_hidden_ref_,
            action_trace_final_hidden_live_base_, trace->final_hidden);
        bind_trace(
            action_trace_velocity_ref_,
            action_trace_velocity_live_base_, trace->velocity);
        bind_trace(
            action_trace_post_action_ref_,
            action_trace_post_action_live_base_, trace->post_action);
    } else {
        action_trace_pre_action_ref_ = at::Tensor();
        action_trace_action_embed_ref_ = at::Tensor();
        action_trace_final_hidden_ref_ = at::Tensor();
        action_trace_velocity_ref_ = at::Tensor();
        action_trace_post_action_ref_ = at::Tensor();
        action_trace_pre_action_live_base_ = 0;
        action_trace_action_embed_live_base_ = 0;
        action_trace_final_hidden_live_base_ = 0;
        action_trace_velocity_live_base_ = 0;
        action_trace_post_action_live_base_ = 0;
    }
    action_step_active_ = true;
    fast_replay_active_ = fast_replay_enabled_;
    set_chunk_size_override(action_len_);
    (void) run_all_layers(
        action_hidden_stage_, k_caches, v_caches, std::nullopt,
        max_prefix, /*is_causal=*/false);
    return action_velocity_stage_;
}

// ── forward: stash GDN state for the seam, then all-layers-once ──────────────
at::Tensor Qwen3_5Model::forward(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    std::vector<at::Tensor>& gdn_states,
    std::vector<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position, bool is_causal) {
    TORCH_CHECK(!z2_bound_,
                "Qwen3.5 text ordinary forward is unavailable while its Z2 lease is active");
    return forward_impl(hidden_states, k_caches, v_caches, gdn_states,
                        conv_states, attention_mask, position, is_causal);
}

at::Tensor Qwen3_5Model::forward_z2(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    std::vector<at::Tensor>& gdn_states,
    std::vector<at::Tensor>& conv_states,
    uint64_t epoch,
    uint64_t plan_hash) {
    validate_z2_prefill_contract();
    TORCH_CHECK(z2_bound_ && z2_epoch_ == epoch &&
                    z2_plan_hash_ == plan_hash,
                "Qwen3.5 text Z2 forward received a stale epoch/plan hash");
    TORCH_CHECK(hidden_states.defined() && hidden_states.dim() == 3 &&
                    hidden_states.size(0) == 1 &&
                    hidden_states.size(1) == z2_prepared_execution_len_ &&
                    hidden_states.size(2) == hidden_size() &&
                    hidden_states.device().type() == at::kPrivateUse1 &&
                    hidden_states.scalar_type() == at::kHalf &&
                    hidden_states.is_contiguous(),
                "Qwen3.5 text Z2 hidden must be contiguous RPU FP16 [1,128,2048]");
    return forward_impl(hidden_states, k_caches, v_caches, gdn_states,
                        conv_states, std::nullopt, /*position=*/0,
                        /*is_causal=*/true);
}

at::Tensor Qwen3_5Model::forward_impl(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    std::vector<at::Tensor>& gdn_states,
    std::vector<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position, bool is_causal) {
    const int64_t seq_len = hidden_states.size(1);
    TORCH_CHECK(seq_len == 1 || seq_len % 64 == 0,
                "Qwen3.5 text expects decode length 1 or a prefill execution "
                "length divisible by 64; use the Python adapter for ragged prefill");
    gdn_states_  = &gdn_states;
    conv_states_ = &conv_states;
    fast_replay_active_ = fast_replay_enabled_ && seq_len > 1;

    // Diagnostic-only per-layer hidden-state tap. The G0.5 adapter records
    // debug prefill in a cache separate from the production prefill graph, so
    // these fixed-address DMA destinations are present only in a debug BUILD.
    // Keep one physical-length allocation stable across that cache's REPLAY.
    if (get_debug_export() && !action_mode_ && seq_len > 1) {
        const int64_t batch_size = hidden_states.size(0);
        const int64_t h = hidden_size();
        TORCH_CHECK(batch_size == 1,
                    "Qwen3.5 fused prefill debug export requires batch_size=1");
        if (!per_layer_debug_buf_.defined()
            || per_layer_debug_buf_.size(0) != num_layers()
            || per_layer_debug_buf_.size(1) != batch_size
            || per_layer_debug_buf_.size(2) != seq_len
            || per_layer_debug_buf_.size(3) != h) {
            per_layer_debug_buf_ = at::empty(
                {num_layers(), batch_size, seq_len, h},
                at::TensorOptions()
                    .dtype(at::kHalf)
                    .device(at::kPrivateUse1));
        }
        if (!last_layer_decoder_debug_buf_.defined()
            || last_layer_decoder_debug_buf_.size(0) != batch_size
            || last_layer_decoder_debug_buf_.size(1) != seq_len
            || last_layer_decoder_debug_buf_.size(2) != h) {
            last_layer_decoder_debug_buf_ = at::empty(
                {batch_size, seq_len, h},
                at::TensorOptions()
                    .dtype(at::kHalf)
                    .device(at::kPrivateUse1));
        }
    }

    // Fast replay skips build_gdn(), so refresh every mutable cache base before
    // entering run_all_layers. Decode still walks the layer loop, but shares the
    // same safe prologue.
    for (int64_t layer_idx = 0; layer_idx < num_layers(); ++layer_idx) {
        if (layer_is_full_[layer_idx])
            continue;
        at::Tensor* st = gdn_state_for_layer(layer_idx);
        at::Tensor* cv = conv_state_for_layer(layer_idx);
        TORCH_CHECK(st && cv && st->defined() && cv->defined(),
                    "Qwen3.5 missing recurrent/conv state cache for layer ",
                    layer_idx);
        gdn_state_live_addr_[layer_idx] =
            ::rhino_lkn::RpuGetDevAddr(st->data_ptr<c10::Half>());
        conv_state_live_addr_[layer_idx] =
            ::rhino_lkn::RpuGetDevAddr(cv->data_ptr<c10::Half>());
        rpu_ddr_flush_force(st->data_ptr<c10::Half>());
        rpu_ddr_flush_force(cv->data_ptr<c10::Half>());
    }

    at::Tensor out = run_all_layers(hidden_states, k_caches, v_caches,
                                    attention_mask, position, is_causal);

    // PREFILL 的 resolved chunk_size 单独留一份。基类的 last_resolved_chunk_size_
    // 每次 compute_chunks 都覆写,而 decode 跑在 prefill 之后
    // 且 seq_len=1 → planner 固定挑 cs=16,于是事后从 Python 读回来的永远是 16,
    // 拿不到 prefill 真正用的值。vision 没有 decode 所以不存在这个问题。
    if (seq_len > 1)
        last_prefill_chunk_size_ = get_last_resolved_chunk_size();

    if (get_debug_export() && !action_mode_ && seq_len > 1
        && per_layer_debug_buf_.defined()
        && last_layer_decoder_debug_buf_.defined()) {
        // Publish views of the stable debug staging allocations. The getter is
        // the post-capture synchronization boundary; cloning here would add 24
        // more RPU allocations and can run before a BUILD scope executes.
        for (int64_t layer_idx = 0; layer_idx < num_layers(); ++layer_idx) {
            const std::string key =
                "L" + std::to_string(layer_idx) + "_layer_output";
            g_debug_tensors[key] = per_layer_debug_buf_[layer_idx];
        }
        const std::string last_prefix =
            "L" + std::to_string(num_layers() - 1);
        g_debug_tensors[last_prefix + "_decoder_output"] =
            last_layer_decoder_debug_buf_;
        g_debug_tensors[last_prefix + "_final_norm_output"] =
            per_layer_debug_buf_[num_layers() - 1];
    }
    gdn_states_  = nullptr;
    conv_states_ = nullptr;

    return out;
}

uint32_t Qwen3_5Model::layer_input_residual_addr(
    int layer_idx,
    const ChunkInfo& chunk) const {
    if (!z2_bound_ || layer_idx != 0) {
        return addr(0, "residual1");
    }
    TORCH_CHECK(z2_input_port_addr_ != 0,
                "Qwen3.5 text Z2: layer-0 input port is unset");
    TORCH_CHECK(chunk.offset >= 0 && chunk.len > 0 &&
                    chunk.offset <= z2_prepared_execution_len_ &&
                    chunk.len <= z2_prepared_execution_len_ - chunk.offset,
                "Qwen3.5 text Z2: layer-0 chunk escapes the prepared input port");
    const uint64_t byte_offset =
        static_cast<uint64_t>(chunk.offset) *
        static_cast<uint64_t>(hidden_size()) * sizeof(c10::Half);
    TORCH_CHECK(byte_offset <= std::numeric_limits<uint32_t>::max() -
                                   z2_input_port_addr_,
                "Qwen3.5 text Z2: layer-0 port address overflows uint32");
    return z2_input_port_addr_ + static_cast<uint32_t>(byte_offset);
}

void Qwen3_5Model::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    const int64_t h = hidden_size();
    const bool z2_layer0 = z2_bound_ && layer_idx == 0;
    const uint32_t input_residual =
        layer_input_residual_addr(layer_idx, chunk);

    // A layer = input DMA → input_layernorm → token mixer → post_norm + MLP + output.
    // ① input DMA: the first layer of the cross-layer group brings h_in into residual1.
    if (!ctx().input_in_spm && !z2_layer0) {
        emit_layer_input_dma(layer_idx, chunk);   // h_in → residual1
    }
    // ② input_layernorm (HF: at the DecoderLayer, NOT inside the mixer). residual1 keeps
    //    h_in (the residual); "input_norm" gets the normed input the mixer reads.
    if (action_mode_) {
        load_adaptive_mod_row(2 * layer_idx);
        apply_adaptive_norm(
            input_residual, addr(0, "input_norm"), chunk);
    } else {
        rpu_launch_rmsnorm_spm_kernel(
            input_residual, addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "norm_w"), chunk.len, h, eps_);
    }
    // ③ token mixer (reads "input_norm"; ends with an all_reduce that adds the h_in residual).
    if (layer_is_full_[layer_idx]) {
        // full-attn's all_reduce leaves the stream in residual2 (== input_norm) directly.
        build_full_attention(layer_idx, chunk);
    } else {
        // decode = recurrent single step (chunk.len<=1); prefill = chunked delta-rule (L>1).
        // GDN's all_reduce leaves the stream in residual2 (== input_norm), like full-attn.
        if (chunk.len <= 1) build_gdn(layer_idx, chunk, /*decode=*/true);
        else                build_gdn(layer_idx, chunk, /*decode=*/false);
    }
    // Both mixers leave the post-mixer stream in residual2 (== input_norm).
    // ④ shared post_norm + MLP + final_norm + output (stream in input_norm).
    emit_mlp_and_output(layer_idx, chunk);
}

void Qwen3_5Model::launch_linear(
    uint32_t input, const at::Tensor& weight, uint32_t output,
    int64_t m, int64_t n, int64_t k, int partition, int num_cores,
    uint32_t bias_spm_addr, bool force_gemm) {
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        input, weight, output, m, n, k, partition, num_cores,
        bias_spm_addr, force_gemm, /*scale=*/{},
        /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
        /*force_acc32=*/linear_acc32_);
}

void Qwen3_5Model::load_adaptive_mod_row(int64_t row) {
    TORCH_CHECK(action_mode_ && adaptive_mod_ref_.defined(),
                "Qwen3.5 action modulation is unavailable");
    const int64_t width = 3 * hidden_size();
    const int64_t variants = action_rtc_active_ ? 2 : 1;
    const int64_t rows = 2 * num_layers() + 1;
    const int64_t step_row = action_loop_active_
        ? (ctx().body_iter * rows + row) * variants
        : row * variants;
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &adaptive_mod_live_base_,
        step_row * width * static_cast<int64_t>(sizeof(c10::Half)),
        variants * width, addr(0, "adaptive_mod"), NUM_CORES);
}

void Qwen3_5Model::apply_adaptive_norm(
    uint32_t input, uint32_t output, const ChunkInfo& chunk) {
    const int64_t h = hidden_size();
    const int64_t clean_rows = action_rtc_active_
        ? std::clamp<int64_t>(
              action_rtc_prefix_len_ - chunk.offset, 0, chunk.len)
        : 0;
    auto emit_segment = [&](int64_t row_offset, int64_t row_count,
                            uint32_t modulation) {
        if (row_count == 0) return;
        const uint32_t data_offset = static_cast<uint32_t>(
            row_offset * h * static_cast<int64_t>(DWIDTH));
        const uint32_t shift = modulation + static_cast<uint32_t>(h * DWIDTH);
        rpu_launch_rmsnorm_spm_kernel(
            input + data_offset, output + data_offset,
            modulation, row_count, h, eps_);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            shift, output + data_offset, output + data_offset,
            row_count, h, c10::Half(1.0), ValuOpType::ADD,
            /*is_bopa=*/false);
    };
    if (!action_rtc_active_) {
        emit_segment(/*row_offset=*/0, chunk.len, addr(0, "adaptive_mod"));
        return;
    }
    emit_segment(/*row_offset=*/0, clean_rows, addr(0, "adaptive_mod"));
    emit_segment(
        clean_rows, chunk.len - clean_rows,
        addr(0, "adaptive_mod") + static_cast<uint32_t>(3 * h * DWIDTH));
}

void Qwen3_5Model::apply_adaptive_residual_gate(
    uint32_t output, uint32_t residual, const ChunkInfo& chunk) {
    const int64_t h = hidden_size();
    const int64_t clean_rows = action_rtc_active_
        ? std::clamp<int64_t>(
              action_rtc_prefix_len_ - chunk.offset, 0, chunk.len)
        : 0;
    auto emit_segment = [&](int64_t row_offset, int64_t row_count,
                            uint32_t modulation) {
        if (row_count == 0) return;
        const uint32_t data_offset = static_cast<uint32_t>(
            row_offset * h * static_cast<int64_t>(DWIDTH));
        const uint32_t gate =
            modulation + static_cast<uint32_t>(2 * h * DWIDTH);
        apply_raw_residual_gate(
            output + data_offset, residual + data_offset, gate, row_count);
    };
    if (!action_rtc_active_) {
        emit_segment(/*row_offset=*/0, chunk.len, addr(0, "adaptive_mod"));
        return;
    }
    emit_segment(/*row_offset=*/0, clean_rows, addr(0, "adaptive_mod"));
    emit_segment(
        clean_rows, chunk.len - clean_rows,
        addr(0, "adaptive_mod") + static_cast<uint32_t>(3 * h * DWIDTH));
}

void Qwen3_5Model::apply_raw_residual_gate(
    uint32_t output, uint32_t residual, uint32_t gate, int64_t seq_len) {
    const int64_t elems = seq_len * hidden_size();
    rpu_launch_eltwise_binary_spm_kernel(
        output, residual, output, elems,
        ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
        gate, output, output, seq_len, hidden_size(),
        c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_eltwise_binary_spm_kernel(
        output, residual, output, elems,
        ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
}

void Qwen3_5Model::emit_action_input_projection() {
    TORCH_CHECK(action_step_active_,
                "Qwen3.5 action input hook outside action-step forward");
    if (!action_loop_active_ || ctx().body_iter == 0) {
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &action_input_live_base_, /*src_offset_bytes=*/0,
            action_len_ * action_dim_pad_, addr(0, "action_input"),
            /*num_cores=*/1);
    }
    if (action_rtc_active_) {
        // Replay-safe prefix inpainting: the caller supplies a fresh compact
        // DDR tensor, and every Euler body restores its committed leading rows
        // before the input projection. The previous body may have updated those
        // rows in SPM, so loading only body 0 would change RTC semantics.
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &action_rtc_prefix_live_base_, /*src_offset_bytes=*/0,
            action_rtc_prefix_len_ * action_dim_pad_,
            addr(0, "action_input"), /*num_cores=*/1);
    }
    if (action_trace_active_) {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_input"),
            &action_trace_pre_action_live_base_,
            ctx().body_iter * action_len_ * action_dim_pad_ * DWIDTH,
            action_len_ * action_dim_pad_);
    }
    launch_linear(
        addr(0, "action_input"), action_input_w_,
        addr(0, "action_hidden"), action_len_, hidden_size(),
        action_dim_pad_, /*partition=*/1, /*num_cores=*/1,
        addr(0, "action_input_bias"));
    if (action_trace_active_) {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_hidden"),
            &action_trace_action_embed_live_base_,
            ctx().body_iter * action_len_ * hidden_size() * DWIDTH,
            action_len_ * hidden_size());
    }
    rpu_launch_spm_copy_ddr_dma(
        addr(0, "action_hidden"),
        action_hidden_stage_.data_ptr<c10::Half>(),
        action_len_ * hidden_size());
}

void Qwen3_5Model::emit_action_output_projection() {
    TORCH_CHECK(action_step_active_,
                "Qwen3.5 action output hook outside action-step forward");
    if (action_trace_active_) {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "residual1"),
            &action_trace_final_hidden_live_base_,
            ctx().body_iter * action_len_ * hidden_size() * DWIDTH,
            action_len_ * hidden_size());
    }
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &action_keep_mask_live_base_, /*src_offset_bytes=*/0,
        action_dim_pad_, addr(0, "action_keep_mask"),
        /*num_cores=*/1);
    launch_linear(
        addr(0, "residual1"), action_output_w_,
        addr(0, "action_velocity"), action_len_, action_dim_pad_,
        hidden_size(), /*partition=*/1, /*num_cores=*/1,
        addr(0, "action_output_bias"));
    if (action_trace_active_) {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_velocity"),
            &action_trace_velocity_live_base_,
            ctx().body_iter * action_len_ * action_dim_pad_ * DWIDTH,
            action_len_ * action_dim_pad_);
    }
    const bool final_step =
        !action_loop_active_ || ctx().body_iter + 1 == action_num_steps_;
    if (final_step) {
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "action_velocity"),
            action_velocity_stage_.data_ptr<c10::Half>(),
            action_len_ * action_dim_pad_);
    }
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "action_input"), addr(0, "action_velocity"),
        addr(0, "action_input"), action_len_ * action_dim_pad_,
        ValuOpType::ADD, action_euler_scale_, /*num_cores=*/1);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
        addr(0, "action_keep_mask"), addr(0, "action_input"),
        addr(0, "action_input"), action_len_, action_dim_pad_,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
    if (action_trace_active_) {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_input"),
            &action_trace_post_action_live_base_,
            ctx().body_iter * action_len_ * action_dim_pad_ * DWIDTH,
            action_len_ * action_dim_pad_);
    }
    if (final_step) {
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_input"), &action_output_live_base_,
            /*dst_offset_bytes=*/0, action_len_ * action_dim_pad_);
    }
}

// ── build_full_attention: Qwen3.5 full-attention path with QK norm, partial
//    M-RoPE and the attention output gate. Qwen3.5 has no DeepStack. ──────────
void Qwen3_5Model::build_full_attention(int layer_idx, const ChunkInfo& chunk) {
    const auto& lw = layer_weights_[layer_idx];
    int64_t seq_len = chunk.len;
    int64_t cos_sin_start =
        action_mode_ ? chunk.offset : ctx().position + chunk.offset;
    int64_t h = hidden_size();
    int64_t nq = num_q_heads();
    int64_t nkv = num_kv_heads();
    int64_t hd = head_dim();
    // The adapter replicates logical GQA KV heads to an effective NUM_CORES
    // width. One complete effective KV head lives on each active core;
    // attention-phase kernels run on that resolved tensor-parallel width and
    // virtual_num_cores=NUM_CORES keeps the KV-cache swizzle layout.
    // MLP / all-reduce still span NUM_CORES.
    int64_t tp = attn_tp();

    // input DMA + input_layernorm are done by build_layer_subgraph; the normed input is
    // in "input_norm", h_in stays in residual1 (for the all_reduce residual).
    // Phase 2: QKV linear (col-partition over tp cores)
    launch_linear(
        addr(0, "input_norm"), lw.q_w, addr(0, "q"), seq_len, nq * hd, h, 1, tp);
    launch_linear(
        addr(0, "input_norm"), lw.k_w, addr(0, "k"), seq_len, nkv * hd, h, 1, tp);
    launch_linear(
        addr(0, "input_norm"), lw.v_w, addr(0, "v"), seq_len, nkv * hd, h, 1, tp);
    // Gated-attention gate projection. Compute here while "input_norm" still
    // holds the Phase-1 input RMSNorm output (Phase 3 k_norm overwrites it).
    // Applied as sigmoid(gate) ⊙ attn_out after SDPA (Qwen3.5 attn_output_gate).
    launch_linear(
        addr(0, "input_norm"), lw.attn_gate_w, addr(0, "attn_gate"),
        seq_len, nq * hd, h, 1, tp);

    // Phase 3: QK RMSNorm + RoPE. M-RoPE (interleaved) + partial rotary when
    // has_mrope_ (Qwen3.5); the 1D RoPE branch stays for non-mrope models.
    int64_t local_q_heads = nq / tp;
    int64_t local_kv_dim  = nkv * hd / tp;
    // PREFILL (seq_len>1) uses the per-forward interleaved M-RoPE tables (B channel, set via
    // set_prefill_rope) when available; DECODE + the non-mrope rope branch use the static
    // position-lookup cos_/sin_ from set_weights. Empty prefill_cos_ (non-mrope models / not set)
    // ⇒ fall back to cos_ (text-only interleaved == arange, so identical there).
    const bool use_prefill_rope = (seq_len > 1) && has_mrope_
                                  && prefill_cos_.defined() && prefill_cos_.numel() > 0;
    c10::Half* cos_ptr = (use_prefill_rope ? prefill_cos_ : cos_).data_ptr<c10::Half>();
    c10::Half* sin_ptr = (use_prefill_rope ? prefill_sin_ : sin_).data_ptr<c10::Half>();
    // Decode after an image indexes the static cos_/sin_ at position + rope_delta
    // (HF compute_3d_position_ids). PREFILL uses prefill_cos_ which already bakes the
    // shifted per-token positions from get_rope_index, so the delta must NOT double-apply
    // there. cos_sin_start feeds only the rope kernels (KV-insert recomputes its own
    // ctx().position+chunk.offset), so shifting it here does not disturb the KV slot.
    if (!use_prefill_rope) cos_sin_start += mrope_pos_delta_;

    // Partial rotary, split by phase:
    //   DECODE  (seq_len==1) → partial_rope_1d  (grid NUM_ELT_PER_WRP=512, grid_y=num_tokens)
    //   PREFILL (seq_len >1) → partial_mrope    (grid 64,                 grid_y=ceil(/64))
    // Both read [*, rotary_dim_/2] cos/sin tables and rotate only the first
    // rotary_dim_ values (rotate-half within rotary_dim_). The operator writes
    // only that span, so input/output aliasing preserves the remaining head values.
    // Text-only tables coincide (T==H==W); multimodal prefill needs cos_for_mrope.
    auto emit_partial_rope = [&](uint32_t buf, int64_t heads) {
        if (seq_len <= 1)
            rpu_launch_partial_rope_1d_spm_kernel(
                buf, buf, cos_ptr, sin_ptr, cos_sin_start, seq_len, heads, hd, rotary_dim_, tp);
        else
            rpu_launch_partial_mrope_spm_kernel(
                buf, buf, cos_ptr, sin_ptr, cos_sin_start, seq_len, heads, hd, rotary_dim_, tp);
    };

    if (has_qk_norm_) {
        int64_t local_kv_heads = nkv / tp;
        // Q: norm + partial RoPE stay in-place because the partial kernels write
        // only the rotary span. The legacy full-RoPE path remains out-of-place.
        if (has_mrope_) {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q"), addr(0, "q"),
                layer_addr(layer_idx, 0, "q_norm_w"),
                seq_len * local_q_heads, hd, eps_);
            emit_partial_rope(addr(0, "q"), local_q_heads);
        } else {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q"), addr(0, "output"),
                layer_addr(layer_idx, 0, "q_norm_w"),
                seq_len * local_q_heads, hd, eps_);
            rpu_launch_rope_spm_kernel(
                addr(0, "output"), addr(0, "q"), cos_ptr, sin_ptr,
                seq_len, local_q_heads, hd, cos_sin_start);
        }
        // K: norm + rope (same as Q), ONLY when each core owns >=1 whole head.
        // No GQA weight replication: attn runs on tp = min(NUM_CORES, nkv) cores,
        // so local_kv_heads = nkv / tp == 1 (one whole KV head per active core).
        // The >0 guard is belt-and-suspenders for any nkv % tp != 0 edge case.
        if (local_kv_heads > 0) {
            if (has_mrope_) {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "k"), addr(0, "k"),
                    layer_addr(layer_idx, 0, "k_norm_w"),
                    seq_len * local_kv_heads, hd, eps_);
                emit_partial_rope(addr(0, "k"), local_kv_heads);
            } else {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "k"), addr(0, "input_norm"),
                    layer_addr(layer_idx, 0, "k_norm_w"),
                    seq_len * local_kv_heads, hd, eps_);
                rpu_launch_rope_spm_kernel(
                    addr(0, "input_norm"), addr(0, "k"), cos_ptr, sin_ptr,
                    seq_len, local_kv_heads, hd, cos_sin_start);
            }
        }
    } else {
        rpu_launch_rope_spm_kernel(
            addr(0, "q"), addr(0, "q"), cos_ptr, sin_ptr,
            seq_len, local_q_heads, hd, cos_sin_start);
        rpu_launch_rope_spm_kernel(
            addr(0, "k"), addr(0, "k"), cos_ptr, sin_ptr,
            seq_len, 1, local_kv_dim, cos_sin_start);
    }

    // Phase 4: KV-cache insert + SDPA + O_proj (offsets via addr_offset)
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    const int64_t kv_insert_pos =
        (action_mode_ ? action_prefix_lens_[layer_idx] : ctx().position)
        + chunk.offset;
    rpu_launch_insert_kcache_spm_unified(
        k_cache, kv_insert_pos, addr_offset("k").value,
        seq_len, nkv, hd, tp);
    rpu_launch_insert_vcache_spm_unified(
        v_cache, kv_insert_pos, addr_offset("v").value,
        seq_len, nkv, hd, tp);

    int64_t kv_seq_len = action_mode_
        ? action_prefix_lens_[layer_idx] + ctx().seq_len
        : chunk.kv_seq_len;
    bool sdpa_causal = ctx().is_causal && (seq_len > 1);
    // SDPA on tp cores; virtual_num_cores=NUM_CORES governs the KV-cache 7D swizzle.
    rpu_launch_sdpa_spm_dispatch(
        sdpa_kernel_, k_cache, v_cache, sdpa_causal ? 1 : 0, c10::nullopt,
        addr_offset("q").value, addr_offset("output").value,
        addr_offset("sdpa_tmp").value, 0,
        seq_len, nq, nkv, hd, kv_seq_len, tp, NUM_CORES);

    // Gated-attention output gate (Qwen3.5 attn_output_gate=true):
    //   output = sigmoid(gate) ⊙ output, before o_proj.
    {
        int64_t gate_elems = seq_len * (nq / tp) * hd;
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "attn_gate"), addr(0, "attn_gate"),
            gate_elems, ValuOpType::SIGMOID, /*is_gelu=*/false, tp);
        rpu_launch_eltwise_mul_spm_kernel(
            addr(0, "attn_gate"), addr(0, "output"), addr(0, "output"), gate_elems);
    }
    // O-proj row-partition over tp cores; reduce back to NUM_CORES below.
    launch_linear(
        addr(0, "output"), lw.o_w, addr(0, "oproj"), seq_len, h, nq * hd, 0, tp);

    // Phase 5: attention reduce + residual (tp inputs → NUM_CORES out)
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), layer_input_residual_addr(layer_idx, chunk),
        addr(0, "residual2"), seq_len, h,
        tp, NUM_CORES);
    if (action_mode_) {
        apply_adaptive_residual_gate(
            addr(0, "residual2"), addr(0, "residual1"), chunk);
    }

    // The all_reduce left the residual stream in residual2 (== input_norm); the shared
    // post_norm + MLP + final_norm + output tail runs back in build_layer_subgraph
    // (emit_mlp_and_output), same as the GDN path. build_full_attention is mixer-only.
}

// ── build_gdn: the unified Gated-DeltaNet token mixer. Shared setup/proj/tail + an
//    if(decode) split for the divergent conv + delta-rule core (decode = recurrent
//    single step; prefill = chunked delta-rule). Dispatched by build_layer_subgraph.
void Qwen3_5Model::build_gdn(int layer_idx, const ChunkInfo& chunk, bool decode) {
    const auto& lw = layer_weights_[layer_idx];
    const int64_t H = gdn_nvh_, Dk = gdn_dk_, Dv = gdn_dv_;
    const int64_t conv_dim = gdn_conv_dim_, Kc = gdn_kc_;
    const int64_t nkh = (conv_dim - H * Dv) / (2 * Dk), rep = H / nkh;
    const int64_t key_dim = nkh * Dk, value_dim = H * Dv, hidden = hidden_size();
    const int     nc = NUM_CORES;
    const int64_t Hc = H / nc, Hc_kv = nkh / nc, lval = value_dim / nc, lkey = key_dim / nc;
    const int64_t lconv = conv_dim / nc;                          // per-core conv channels
    const int64_t n_bg = ((Hc + 15) / 16) * 16, N_bg = n_bg * nc; // padded b/a width (16/core)
    const int64_t L = decode ? 1 : chunk.len;   // seq_len: decode 1 token / prefill chunk
    TORCH_CHECK(H % nc == 0 && conv_dim % nc == 0 && value_dim % nc == 0 && nkh % nc == 0,
                "build_gdn: H/conv_dim/value_dim/nkh must be divisible by 8");
    at::Tensor* st = gdn_state_for_layer(layer_idx);
    at::Tensor* cv = conv_state_for_layer(layer_idx);
    TORCH_CHECK(st && cv && st->defined() && cv->defined(),
                "build_gdn: missing recurrent/conv state cache for layer ", layer_idx);
    auto D = [&](const char* n) { return addr(0, n); };
    const bool initial_prefill_chunk =
        !decode && ctx().position == 0 && chunk.offset == 0;

    // Buffer selectors: decode writes its own gdn_* buffers; prefill writes the conv
    // xpad x-region (q/k at row qx, v at vx) so the halo sees [conv_state ; conv_in].
    const int64_t hist = Kc - 1;
    const uint32_t qx = (uint32_t)(hist * lkey * 2), vx = (uint32_t)(hist * lval * 2);
    const uint32_t q_out = decode ? D("gdn_q") : D("gdn_c_q_pad") + qx;
    const uint32_t k_out = decode ? D("gdn_k") : D("gdn_c_k_pad") + qx;
    const uint32_t v_out = decode ? D("gdn_v") : D("gdn_c_v_pad") + vx;
    const uint32_t z_out = decode ? D("gdn_z") : D("gdn_c_z");
    const uint32_t b_out = decode ? D("gdn_b") : D("gdn_c_b");
    const uint32_t a_out = decode ? D("gdn_a") : D("gdn_c_a");

    // ── DMA the M-independent weights/state into SPM. ──
    rpu_launch_ddr_broadcast_spm_dma(lw.gdn_norm_w.data_ptr<c10::Half>(),    Dv,      D("gdn_normw"), nc);
    rpu_launch_ddr_broadcast_spm_dma(gdn_zero_.data_ptr<c10::Half>(),        Dk * Dv, D("gdn_zero"),  nc);
    // gdn_Al: decode = RAW A_log (recurrent device-exps it); prefill = precomputed neg_exp_A.
    if (decode) {
        rpu_launch_ddr_scatter_spm_dma(lw.gdn_A_log.data_ptr<c10::Half>(), 8, 16, D("gdn_Al"), nc);
    } else {
        TORCH_CHECK(gdn_neg_exp_A_[layer_idx].defined(), "build_gdn: neg_exp_A missing for layer ", layer_idx);
        rpu_launch_ddr_scatter_spm_dma(gdn_neg_exp_A_[layer_idx].data_ptr<c10::Half>(), 8, 16, D("gdn_Al"), nc);
    }
    rpu_launch_ddr_scatter_spm_dma(lw.gdn_dt_bias.data_ptr<c10::Half>(), 8, 16, D("gdn_dt"), nc);
    if (initial_prefill_chunk) {
        rpu_launch_fill_spm_kernel(
            D("gdn_state"), Hc * Dk * Dv, c10::Half(0.0f), nc);
        rpu_launch_fill_spm_kernel(
            D("gdn_cs"), Kc * lconv, c10::Half(0.0f), nc);
    } else {
        rpu_launch_ddr_scatter_spm_dma_mutable(
            &gdn_state_live_addr_[layer_idx], 0, Hc * Dk * Dv,
            Hc * Dk * Dv * 2, D("gdn_state"), nc);
        rpu_launch_ddr_scatter_spm_dma_mutable(
            &conv_state_live_addr_[layer_idx], 0, Kc * lconv,
            Kc * lconv * 2, D("gdn_cs"), nc);
    }
    if (!decode) {  // prefill-only M-gen chunk masks (tril/strict, same on every core)
        rpu_launch_ddr_broadcast_spm_dma(gdn_tril_.data_ptr<c10::Half>(),   64 * 64, D("gdn_c_tril"),   nc);
        rpu_launch_ddr_broadcast_spm_dma(gdn_strict_.data_ptr<c10::Half>(), 64 * 64, D("gdn_c_strict"), nc);
    }

    // ── in_proj: q/k/v/z col-parallel + N_bg-padded b/a. input_layernorm is at the layer
    //    level → "input_norm"; decode writes its own buffers, prefill the conv xpad region. ──
    launch_linear(D("input_norm"), lw.gdn_q_w,    q_out, L, key_dim,   hidden, 1, nc, 0, true);
    launch_linear(D("input_norm"), lw.gdn_k_w,    k_out, L, key_dim,   hidden, 1, nc, 0, true);
    launch_linear(D("input_norm"), lw.gdn_v_w,    v_out, L, value_dim, hidden, 1, nc, 0, true);
    launch_linear(D("input_norm"), lw.gdn_in_z_w, z_out, L, value_dim, hidden, 1, nc, 0, true);
    launch_linear(D("input_norm"), lw.gdn_b_bg_w, b_out, L, N_bg, hidden, 1, nc, 0, true);
    launch_linear(D("input_norm"), lw.gdn_a_bg_w, a_out, L, N_bg, hidden, 1, nc, 0, true);

    // ── Divergent token mixer: decode = recurrent single step / prefill = chunked delta-rule.
    //    Both converge on the shared phase-6 tail below. ──
    if (decode) {
        // 2) causal conv1d update — per-path on the path-major conv-state window. gdn_cs is
        //    reinterpreted as [q:(Kc,lkey) | k:(Kc,lkey) | v:(Kc,lval)]; each path's conv is
        //    in-place (out = row Kc-1), then SiLU into its own activation buffer.
        const uint32_t cs_q = D("gdn_cs");
        const uint32_t cs_k = D("gdn_cs") + (uint32_t)(Kc * lkey * 2);
        const uint32_t cs_v = D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2);
        rpu_launch_fla_conv1d_spm_kernel(cs_q, D("gdn_q"), lw.gdn_conv_q_w.data_ptr<c10::Half>(), 1, lkey, Kc, nc);
        rpu_launch_fla_conv1d_spm_kernel(cs_k, D("gdn_k"), lw.gdn_conv_k_w.data_ptr<c10::Half>(), 1, lkey, Kc, nc);
        rpu_launch_fla_conv1d_spm_kernel(cs_v, D("gdn_v"), lw.gdn_conv_v_w.data_ptr<c10::Half>(), 1, lval, Kc, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_q + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_qa"), lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_k + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_ka"), lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_v + (uint32_t)((Kc - 1) * lval * 2), D("gdn_va"), lval, ValuOpType::SILU, false, nc);

        // 3) per-path q/k/v already separated by the proj (no fused split needed).
        const uint32_t a_q = D("gdn_qa");
        const uint32_t a_k = D("gdn_ka");
        const uint32_t a_v = D("gdn_va");

        // 3b) GQA expand: tile q/k from Hc_kv heads → Hc heads (repeat_interleave via
        // the [Hc_kv,1,Dk] middle-dim tile: head h → out 2h,2h+1). The recurrent
        // l2norms q/k internally and l2norm(copy)==copy(l2norm) per head, so tiling
        // first is exact. Skipped when rep==1 (non-GQA: a_q/a_k already have Hc heads).
        uint32_t rq = a_q, rk = a_k;
        if (rep > 1) {
            rpu_launch_tile_spm_kernel(a_q, D("gdn_q_rep"), Hc_kv, 1, Dk, 1, rep, 1, nc);
            rpu_launch_tile_spm_kernel(a_k, D("gdn_k_rep"), Hc_kv, 1, Dk, 1, rep, 1, nc);
            rq = D("gdn_q_rep");
            rk = D("gdn_k_rep");
        }

        // 4) l2norm + gating + recurrent delta-rule (Hc heads/core, 8-core SPMD). gdn_Al = RAW
        //    A_log (the recurrent device-exps it); consumes raw a/b/dt; state updated in place.
        rpu_emit_gdn_recurrent_multihead_seq(
            rq, rk, a_v, D("gdn_state"), D("gdn_p"), D("gdn_delta"), D("gdn_zero"),
            D("gdn_Al"), D("gdn_a"), D("gdn_b"), D("gdn_dt"),
            D("gdn_gexp"), D("gdn_beta"), D("gdn_gs"), D("gdn_out"),
            Hc, Dk, Dv, nc);
        // decode leaves the updated conv window in gdn_cs; the shared tail scatters it → cache.
    } else {

    // ── Route B (arbitrary prefill length): the adapter pads the whole prefill seq up to a multiple
    //    of 64, so L=chunk.len is ALWAYS 64-aligned and the chunk core runs on L directly (full-attn
    //    SDPA + GDN both need 64-alignment). Lv = the REAL (non-pad) tokens in this chunk =
    //    clamp(valid_prefill_len_ - chunk.offset, 0, L); it equals L for a non-tail chunk or when
    //    unset. conv/halo run on Lv (→ conv_state cso = last-3 REAL tokens), and rows [Lv:L] of
    //    k/β/g are zero-filled so the pad tokens contribute NOTHING to the carried recurrent_state
    //    (exact). Pad rows' output is garbage; the adapter slices it off the padded logits. ──
    TORCH_CHECK(L % 64 == 0, "build_gdn: prefill chunk.len (", L,
                ") must be a multiple of 64 (route B: adapter pads the prefill sequence)");
    int64_t Lv = L;
    if (valid_prefill_len_ > 0)
        Lv = std::max<int64_t>(0, std::min<int64_t>(L, valid_prefill_len_ - chunk.offset));
    TORCH_CHECK(Lv >= 1, "build_gdn: prefill chunk has 0 real tokens (Lv=0, chunk.offset=",
                chunk.offset, ", valid_prefill_len=", valid_prefill_len_, ") — bad chunk plan/padding");

    // ── Phase 2: causal conv1d (prefill, per-path) + SiLU → gdn_c_query/key/value. ──
    //   pad rows [0:hist] = prev-chunk conv state (fresh prefill = 0); conv writes IN PLACE into
    //   gdn_c_*_pad[hist:], then SiLU reads it → gdn_c_query/key/value (the conv-activated q/k/v).
    if (Lv == 1) {
        // Lv==1 edge (last chunk holds a single real token; real_len ≡ 1 mod cs, > cs): the prefill
        // conv kernel is L>1 only, so use the DECODE single-token conv. gdn_cs already holds the conv
        // history (loaded from the cache at setup, before the decode/prefill split); the token's
        // q/k/v projection sits at gdn_c_*_pad+qx/vx (proj wrote row hist). 1:1 with the decode
        // branch: the conv output lands in gdn_cs[Kc-1] (SiLU → gdn_c_*[0]) and gdn_cs is updated in
        // place to [h1,h2,x,out] so its [0:3] = last-3 real tokens; the shared tail scatters gdn_cs
        // to the cache. Rows [1:L] are zero-filled below, so the chunk core sees [silu(conv); 0…].
        const uint32_t cs_q = D("gdn_cs");
        const uint32_t cs_k = D("gdn_cs") + (uint32_t)(Kc * lkey * 2);
        const uint32_t cs_v = D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2);
        rpu_launch_fla_conv1d_spm_kernel(cs_q, D("gdn_c_q_pad") + qx, lw.gdn_conv_q_w.data_ptr<c10::Half>(), 1, lkey, Kc, nc);
        rpu_launch_fla_conv1d_spm_kernel(cs_k, D("gdn_c_k_pad") + qx, lw.gdn_conv_k_w.data_ptr<c10::Half>(), 1, lkey, Kc, nc);
        rpu_launch_fla_conv1d_spm_kernel(cs_v, D("gdn_c_v_pad") + vx, lw.gdn_conv_v_w.data_ptr<c10::Half>(), 1, lval, Kc, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_q + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_c_query"), lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_k + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_c_key"),   lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_v + (uint32_t)((Kc - 1) * lval * 2), D("gdn_c_value"), lval, ValuOpType::SILU, false, nc);
    } else {
        const int64_t N_seg = (Lv >= 32) ? 8 : 1, seg_len = Lv / N_seg;
        const int64_t seg_rem = Lv % N_seg;   // remainder-in-seg0 (matches kernel _seg_lo / launcher reg[22])
        // pad[0:hist] = prev-chunk conv carry, copied from gdn_cs (loaded from the DDR cache at
        // setup; fresh chunk 0 = 0 since the cache inits to 0). gdn_cs is path-major, SAME offsets
        // as decode's cs_q/cs_k/cs_v (q@+0, k@+Kc·lkey, v@+Kc·2lkey). The conv below writes the NEW
        // carry back into these gdn_cs regions (cso pointers) and the shared tail scatters gdn_cs →
        // cache, so conv_state carries across prefill chunks and into decode. (SPM→SPM MAX-copy.)
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_cs"),                               D("gdn_cs"),                               D("gdn_c_q_pad"), hist * lkey, ValuOpType::MAX, c10::Half(1.0f), nc);
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_cs") + (uint32_t)(Kc * lkey * 2),     D("gdn_cs") + (uint32_t)(Kc * lkey * 2),     D("gdn_c_k_pad"), hist * lkey, ValuOpType::MAX, c10::Half(1.0f), nc);
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2), D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2), D("gdn_c_v_pad"), hist * lval, ValuOpType::MAX, c10::Half(1.0f), nc);
        if (N_seg > 1) {  // halo[bz] = pad[_seg_lo(bz)]: the L%8 remainder goes ENTIRELY to seg0
            // (base=L/8, rem=L%8; kernel/launcher reg[22]=seg_rem), so the 8 halo starts are
            // {0, rem+base, rem+2base, …} — equidistant ONLY when rem==0. A single strided slice
            // cuts equidistant starts, so rem==0 keeps the original one-slice form; rem!=0 needs
            // TWO slices: seg0 = pad[0:hist] (conv history), seg1..N-1 = pad[rem+bz*base] (stride
            // base from row rem+base → halo rows [hist:]). The operator validates ragged L.
            // PER-CORE: slice is use_local=FALSE (reg0-3 are a GLOBAL per-core base+off addr),
            // so issue one single-core launch per core at addr(c,·), else cores
            // 1..nc-1's halo stays UNWRITTEN (conv reads NaN).
            auto emit_halo = [&](int c, const char* pad_n, const char* halo_n, int64_t l) {
                if (seg_rem == 0) {
                    rpu_launch_slice_spm_kernel(addr(c, pad_n), addr(c, halo_n), {N_seg, seg_len, l}, {N_seg, hist, l}, {0, 0, 0}, 1);
                } else {
                    rpu_launch_slice_spm_kernel(addr(c, pad_n), addr(c, halo_n), {1, seg_len, l}, {1, hist, l}, {0, 0, 0}, 1);
                    rpu_launch_slice_spm_kernel(addr(c, pad_n) + (uint32_t)((seg_rem + seg_len) * l * 2),
                                                addr(c, halo_n) + (uint32_t)(hist * l * 2),
                                                {N_seg - 1, seg_len, l}, {N_seg - 1, hist, l}, {0, 0, 0}, 1);
                }
            };
            for (int c = 0; c < nc; ++c) {
                emit_halo(c, "gdn_c_q_pad", "gdn_c_q_halo", lkey);
                emit_halo(c, "gdn_c_k_pad", "gdn_c_k_halo", lkey);
                emit_halo(c, "gdn_c_v_pad", "gdn_c_v_halo", lval);
            }
        }
        // cso (new conv carry = last hist inputs; only the top segment stores it) is written
        // straight into gdn_cs at the same path-major offsets decode uses → the shared tail's
        // gdn_cs→cache scatter carries it to the next chunk / decode step.
        rpu_launch_fla_conv1d_prefill_spm_kernel(D("gdn_c_q_pad"), D("gdn_cs"),                               D("gdn_c_q_halo"), lw.gdn_conv_q_w.data_ptr<c10::Half>(), 1, lkey, Kc, Lv, nc);
        rpu_launch_fla_conv1d_prefill_spm_kernel(D("gdn_c_k_pad"), D("gdn_cs") + (uint32_t)(Kc * lkey * 2),     D("gdn_c_k_halo"), lw.gdn_conv_k_w.data_ptr<c10::Half>(), 1, lkey, Kc, Lv, nc);
        rpu_launch_fla_conv1d_prefill_spm_kernel(D("gdn_c_v_pad"), D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2), D("gdn_c_v_halo"), lw.gdn_conv_v_w.data_ptr<c10::Half>(), 1, lval, Kc, Lv, nc);
        rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_q_pad") + qx, D("gdn_c_query"), L * lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_k_pad") + qx, D("gdn_c_key"),   L * lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_v_pad") + vx, D("gdn_c_value"), L * lval, ValuOpType::SILU, false, nc);
    }
    // ── Phases 3–5: prefill chunked delta-rule core. Falls through to the SHARED phase-6 tail
    //    below (gated-norm + out_proj + state/conv writeback + AllReduce residual), same as the
    //    decode branch — both leave the post-mixer stream in residual2 for build_layer_subgraph. ──
    const int64_t C = 64, N = L / C;            // GDN chunk size / sub-chunk count (L is 64-aligned)
    const int64_t vg_c = H / nc;                // beta/g heads/core (== Hc)
    // Per-core slices take global SPM addresses and do not rebase internally.
    // Launch once per core using pc_addr to select that core's resolved buffer.
    auto pc_addr = [&](uint32_t a0, int cc) -> uint32_t {
        return a0 - SPM_ALLOC.addr(0, 0) + SPM_ALLOC.addr(cc, 0);
    };
    auto slice_pc = [&](uint32_t s, uint32_t d, const std::vector<int64_t>& is,
                        const std::vector<int64_t>& os, const std::vector<int64_t>& b) {
        for (int cc = 0; cc < nc; ++cc)
            rpu_launch_slice_spm_kernel(pc_addr(s, cc), pc_addr(d, cc), is, os, b, 1);
    };

    // ===== Phase 3: prep (beta/g) + q/k l2norm + GQA tile =====
    slice_pc(D("gdn_c_b"), D("gdn_c_beta"), {L, n_bg}, {L, vg_c}, {0, 0});
    slice_pc(D("gdn_c_a"), D("gdn_c_g"),    {L, n_bg}, {L, vg_c}, {0, 0});
    rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_beta"), D("gdn_c_beta"), L * vg_c, ValuOpType::SIGMOID, false, nc);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_dt"), D("gdn_c_g"), D("gdn_c_g"), L, vg_c, c10::Half(1.0f), ValuOpType::ADD, false);  // +dt_bias
    rpu_launch_eltwise_unary_spm_kernel(
        D("gdn_c_g"), D("gdn_c_g"), L * vg_c, ValuOpType::SOFTPLUS,
        false, nc);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_Al"), D("gdn_c_g"), D("gdn_c_g"), L, vg_c, c10::Half(1.0f), ValuOpType::MUL, false);  // g *= neg_exp_A
    rpu_launch_l2norm_spm_kernel(D("gdn_c_query"), D("gdn_c_query"), L * Hc_kv, Dk, 1e-6, nc);
    rpu_launch_l2norm_spm_kernel(D("gdn_c_key"),   D("gdn_c_key"),   L * Hc_kv, Dk, 1e-6, nc);
    uint32_t rq = D("gdn_c_query"), rk = D("gdn_c_key");
    const uint32_t a_v = D("gdn_c_value");
    if (rep > 1) {
        rpu_launch_tile_spm_kernel(D("gdn_c_query"), D("gdn_c_q_rep"), L * Hc_kv, 1, Dk, 1, rep, 1, nc);
        rpu_launch_tile_spm_kernel(D("gdn_c_key"),   D("gdn_c_k_rep"), L * Hc_kv, 1, Dk, 1, rep, 1, nc);
        rq = D("gdn_c_q_rep");
        rk = D("gdn_c_k_rep");
    }

    // Zero the pad rows [Lv:L] of the chunk-core inputs (route B: the last chunk's [valid:L] are pad
    // tokens). k/β/g=0 makes the pad positions contribute NOTHING to the carried recurrent_state
    // (exact, not approx); q/v zeroed too so no NaN leaks through the chunk attn (their outputs get
    // sliced off the padded logits by the adapter).
    if (Lv < L) {
        const int64_t np = L - Lv;
        rpu_launch_fill_spm_kernel(rq              + (uint32_t)(Lv * Hc * Dk * 2), np * Hc * Dk, c10::Half(0.0f), nc);
        rpu_launch_fill_spm_kernel(rk              + (uint32_t)(Lv * Hc * Dk * 2), np * Hc * Dk, c10::Half(0.0f), nc);
        rpu_launch_fill_spm_kernel(a_v             + (uint32_t)(Lv * lval * 2),    np * lval,    c10::Half(0.0f), nc);
        rpu_launch_fill_spm_kernel(D("gdn_c_beta") + (uint32_t)(Lv * vg_c * 2),    np * vg_c,    c10::Half(0.0f), nc);
        rpu_launch_fill_spm_kernel(D("gdn_c_g")    + (uint32_t)(Lv * vg_c * 2),    np * vg_c,    c10::Half(0.0f), nc);
    }

    // ===== Phase 4: CHUNK CORE (M-gen) — N-major [N,Hc,C,*] =====
    const int64_t HN = Hc * N, HL = Hc * L, CC = C * C;
    const float   qk_scale = 1.0f / std::sqrt((float)Dk);
    // N-major setup transpose: token-major [N,C,Hc,Dl] -> [N,Hc,C,Dl] (= permute(0,2,1,3)),
    // realized as N independent 3D {1,0,2} transposes ([C,Hc,Dl]->[Hc,C,Dl]) — one per sub-chunk,
    // since there is no 4D SPM permute. Keeping N outermost turns every phase-5 sub-chunk read into
    // a pure base+offset view (no slice/insert). Phase 4's ops are all per-(h,n)-group batched ops
    // (HN batch), agnostic to (h,n) vs (n,h) ordering, so the result is bit-identical to head-major.
    auto permute_nmajor = [&](uint32_t in_base, uint32_t out_base, int64_t Dl) {
        for (int64_t ni = 0; ni < N; ++ni)
            rpu_launch_permute3d_spm_kernel(
                in_base  + (uint32_t)(ni * C  * Hc * Dl * 2),   // block ni: [C, Hc, Dl]
                out_base + (uint32_t)(ni * Hc * C  * Dl * 2),   // block ni: [Hc, C, Dl]
                C, Hc, Dl, {1, 0, 2}, nc);
    };
    permute_nmajor(rq,               D("gdn_c_q_h"),    Dk);
    permute_nmajor(rk,               D("gdn_c_k_h"),    Dk);
    permute_nmajor(a_v,              D("gdn_c_v_h"),    Dv);
    permute_nmajor(D("gdn_c_beta"),  D("gdn_c_beta_h"), 1);
    permute_nmajor(D("gdn_c_g"),     D("gdn_c_g_h"),    1);
    rpu_launch_eltwise_binary_scalar_spm_kernel(D("gdn_c_q_h"), c10::Half(qk_scale), D("gdn_c_q_h"), HL * Dk, ValuOpType::MUL);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_beta_h"), D("gdn_c_k_h"), D("gdn_c_kbeta"), HL, Dk, c10::Half(1.0f), ValuOpType::MUL, false);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_beta_h"), D("gdn_c_v_h"), D("gdn_c_vbeta"), HL, Dv, c10::Half(1.0f), ValuOpType::MUL, false);
    rpu_launch_cumsum_spm_kernel(D("gdn_c_g_h"), D("gdn_c_gcumsum"), D("gdn_c_ws"), HN, C, nc);
    rpu_launch_fill_spm_kernel(D("gdn_c_decay"), HN * CC, c10::Half(0.0f), nc);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_gcumsum"), D("gdn_c_decay"), D("gdn_c_decay"), HN * C, C, c10::Half(1.0f), ValuOpType::ADD, false);
    rpu_launch_eltwise_binary_Bx1xC_BxNxC_spm_kernel(D("gdn_c_gcumsum"), D("gdn_c_decay"), D("gdn_c_decay"), HN, C, C, c10::Half(1.0f), ValuOpType::SUB, true);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_c_tril"), D("gdn_c_decay"), D("gdn_c_decay"), HN, CC, c10::Half(1.0f), ValuOpType::MUL, false);
    rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_decay"), D("gdn_c_decay"), HN * CC, ValuOpType::EXP, false, nc);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_c_tril"), D("gdn_c_decay"), D("gdn_c_decay"), HN, CC, c10::Half(1.0f), ValuOpType::MUL, false);
    rpu_launch_bmm_spm_kernel(D("gdn_c_kbeta"), D("gdn_c_k_h"), D("gdn_c_attn"), C, C, Dk, HN, nc, BmmMode::RowByRow);
    rpu_launch_eltwise_binary_scalar_spm_kernel(D("gdn_c_attn"), c10::Half(-1.0f), D("gdn_c_attn"), HN * CC, ValuOpType::MUL);
    rpu_launch_eltwise_binary_spm_kernel(D("gdn_c_attn"), D("gdn_c_decay"), D("gdn_c_attn"), HN * CC, ValuOpType::MUL, c10::Half(1.0f), nc);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_c_strict"), D("gdn_c_attn"), D("gdn_c_attn"), HN, CC, c10::Half(1.0f), ValuOpType::MUL, false);
    rpu_launch_unit_tril_inv_spm_kernel(D("gdn_c_attn"), D("gdn_c_attn"), HN, nc);
    rpu_launch_bmm_spm_kernel(D("gdn_c_attn"), D("gdn_c_vbeta"), D("gdn_c_vnew"), C, Dv, C, HN, nc);
    rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_gcumsum"), D("gdn_c_gexp"), HL, ValuOpType::EXP, false, nc);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_gexp"), D("gdn_c_kbeta"), D("gdn_c_kbeta"), HL, Dk, c10::Half(1.0f), ValuOpType::MUL, false);
    rpu_launch_bmm_spm_kernel(D("gdn_c_attn"), D("gdn_c_kbeta"), D("gdn_c_kcd"), C, Dk, C, HN, nc);

    // ===== Phase 5: per-chunk recurrent loop. gdn_state [Hc,Dk,Dv] carried across i. =====
    // N-major layout ([N,Hc,C,*]) makes each sub-chunk i a CONTIGUOUS [Hc,C,*] block, so the
    // per-chunk operands are pure base+offset views into the phase-4 buffers — no slice/insert.
    // a_qc/a_kc/a_vc/a_gexpc are mutated in place below, but each gdn_c_*[i] region is touched
    // only within iteration i and dead afterwards, so aliasing the source is exact. Only glast
    // (last timestep, C-axis) still needs a real slice — orthogonal to the N reorder.
    const std::vector<int64_t> sh_gl{Hc, C}, sl_gl{Hc, 1};
    for (int64_t i = 0; i < N; ++i) {
        const std::vector<int64_t> bC{0, C - 1};
        const uint32_t a_qc    = D("gdn_c_q_h")     + (uint32_t)(i * Hc * C * Dk * 2);
        const uint32_t a_kc    = D("gdn_c_k_h")     + (uint32_t)(i * Hc * C * Dk * 2);
        const uint32_t a_vc    = D("gdn_c_vnew")    + (uint32_t)(i * Hc * C * Dv * 2);
        const uint32_t a_kcdc  = D("gdn_c_kcd")     + (uint32_t)(i * Hc * C * Dk * 2);
        const uint32_t a_decc  = D("gdn_c_decay")   + (uint32_t)(i * Hc * C * C  * 2);
        const uint32_t a_gc    = D("gdn_c_gcumsum") + (uint32_t)(i * Hc * C * 2);
        const uint32_t a_gexpc = D("gdn_c_gexp")    + (uint32_t)(i * Hc * C * 2);
        const uint32_t a_outc  = D("gdn_c_coreout") + (uint32_t)(i * Hc * C * Dv * 2);
        rpu_launch_bmm_spm_kernel(a_qc, a_kc, D("gdn_c_attnc"), C, C, Dk, Hc, nc, BmmMode::RowByRow);
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_c_attnc"), a_decc, D("gdn_c_attnc"), Hc * C * C, ValuOpType::MUL, c10::Half(1.0f), nc);
        rpu_launch_bmm_spm_kernel(a_kcdc, D("gdn_state"), a_outc, C, Dv, Dk, Hc, nc);
        rpu_launch_eltwise_binary_spm_kernel(a_vc, a_outc, a_vc, Hc * C * Dv, ValuOpType::SUB, c10::Half(1.0f), nc);
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(a_gexpc, a_qc, a_qc, Hc * C, Dk, c10::Half(1.0f), ValuOpType::MUL, false);
        rpu_launch_bmm_spm_kernel(a_qc, D("gdn_state"), a_outc, C, Dv, Dk, Hc, nc);
        rpu_launch_bmm_spm_kernel(D("gdn_c_attnc"), a_vc, D("gdn_c_recout"), C, Dv, C, Hc, nc);
        rpu_launch_eltwise_binary_spm_kernel(a_outc, D("gdn_c_recout"), a_outc, Hc * C * Dv, ValuOpType::ADD, c10::Half(1.0f), nc);
        slice_pc(a_gexpc, D("gdn_c_glast"), sh_gl, sl_gl, bC);
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_glast"), D("gdn_state"), D("gdn_state"), Hc, Dk * Dv, c10::Half(1.0f), ValuOpType::MUL, false);
        slice_pc(a_gc, D("gdn_c_glast"), sh_gl, sl_gl, bC);
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_glast"), a_gc, a_gexpc, Hc, C, c10::Half(1.0f), ValuOpType::SUB, false);
        rpu_launch_eltwise_unary_spm_kernel(a_gexpc, a_gexpc, Hc * C, ValuOpType::EXP, false, nc);
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(a_gexpc, a_kc, a_kc, Hc * C, Dk, c10::Half(1.0f), ValuOpType::MUL, false);
        rpu_launch_bmm_spm_kernel(a_kc, a_vc, D("gdn_c_recstate"), Dk, Dv, C, Hc, nc, BmmMode::ColByCol);
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_state"), D("gdn_c_recstate"), D("gdn_state"), Hc * Dk * Dv, ValuOpType::ADD, c10::Half(1.0f), nc);
        // a_outc == gdn_c_coreout[i] (written in place above) — no insert needed.
    }
    // final transpose: core_attn_out N-major [N,Hc,C,Dv] -> token-major [L,Hc,Dv] (= [N,C,Hc,Dv]),
    // N independent 3D {1,0,2} transposes ([Hc,C,Dv]->[C,Hc,Dv]), one per sub-chunk (no 4D permute).
    // Pad rows [Lv:L] carry zeroed output; the adapter slices them off the logits.
    for (int64_t ni = 0; ni < N; ++ni)
        rpu_launch_permute3d_spm_kernel(
            D("gdn_c_coreout") + (uint32_t)(ni * Hc * C * Dv * 2),   // block ni: [Hc, C, Dv]
            D("gdn_c_out_tok") + (uint32_t)(ni * C  * Hc * Dv * 2),  // block ni: [C, Hc, Dv]
            Hc, C, Dv, {1, 0, 2}, nc);
    }  // end prefill (chunked delta-rule)

    // ── Phase 6 tail (SHARED decode + prefill): RMSNormGated (norm-then-gate) + out_proj
    //    (row-parallel) + recurrent-state writeback + AllReduce residual → residual2.
    //    RMSNormGated applies rms_norm to the raw core_attn_out over Dv · norm_w,
    //    THEN · silu(z). core_attn_out is token-major [L,Hc,Dv] per core → M = L*Hc, N = Dv;
    //    decode's gdn_out [Hc,Dv] is already token-major (L=1). ──
    const uint32_t out_buf   = decode ? D("gdn_out")   : D("gdn_c_out_tok");
    const uint32_t gated_buf = decode ? D("gdn_gated") : D("gdn_c_gated");
    const uint32_t z_buf     = decode ? D("gdn_z")     : D("gdn_c_z");
    const uint32_t m_buf     = decode ? D("gdn_m")     : D("gdn_c_m");
    rpu_launch_rmsnorm_spm_kernel(out_buf, gated_buf, D("gdn_normw"), L * Hc, Dv, 1e-6);
    // gate = silu(z) · normed as TWO SEPARATE ops: EltwiseUnary applies
    // silu(z) in place, then EltwiseBinary multiplies. The fused llama_silu_mul
    // kernel applies SiLU to the opposite operand and is not equivalent here.
    rpu_launch_eltwise_unary_spm_kernel(z_buf, z_buf, L * lval, ValuOpType::SILU, false, nc);
    rpu_launch_eltwise_binary_spm_kernel(z_buf, gated_buf, gated_buf, L * lval, ValuOpType::MUL, c10::Half(1.0f), nc);
    // out_proj (row-parallel): gated [L,value_dim] @ gdn_out_w → per-core partial m [L,hidden].
    // FULL K = value_dim (launcher splits local_k = value_dim/tp).
    launch_linear(gated_buf, lw.gdn_out_w, m_buf, L, hidden, value_dim, 0, nc, 0, true);
    // state + conv-window writeback (SHARED decode + prefill): both leave the updated recurrent
    // state in gdn_state and the new conv carry in gdn_cs (prefill's cso is written into gdn_cs at
    // the path offsets), so one pair of scatters carries BOTH states across chunks and into decode.
    rpu_launch_spm_scatter_ddr_dma_mutable(D("gdn_state"), &gdn_state_live_addr_[layer_idx], 0, Hc * Dk * Dv, Hc * Dk * Dv * 2, nc);
    rpu_launch_spm_scatter_ddr_dma_mutable(D("gdn_cs"),    &conv_state_live_addr_[layer_idx], 0, Kc * lconv,    Kc * lconv * 2,    nc);
    // AllReduce the out_proj partials + h_in residual → post-mixer stream (== HF `mixer_out`).
    rpu_launch_all_reduce_sum_residual_kernel(
        m_buf, layer_input_residual_addr(layer_idx, chunk),
        D("residual2"), L, hidden);
    return;
}


at::Tensor* Qwen3_5Model::gdn_state_for_layer(int layer_idx) {
    if (!gdn_states_ || layer_idx < 0 ||
        layer_idx >= static_cast<int>(gdn_states_->size())) return nullptr;
    return &(*gdn_states_)[layer_idx];
}

at::Tensor* Qwen3_5Model::conv_state_for_layer(int layer_idx) {
    if (!conv_states_ || layer_idx < 0 ||
        layer_idx >= static_cast<int>(conv_states_->size())) return nullptr;
    return &(*conv_states_)[layer_idx];
}

// Post-mixer layer tail, shared by full-attn and GDN (HF: residual add is fused into
// the mixer's all_reduce; this is the post_attention_layernorm + MLP + residual half).
// CONTRACT: the residual stream (h_in + mixer_out) must be in "input_norm" (== the
// "residual2" alias) on entry. Emits: residual1 = post_norm(stream); MLP(+residual) →
// residual1; final_norm at the last layer; then output DMA.
void Qwen3_5Model::emit_mlp_and_output(int layer_idx, const ChunkInfo& chunk) {
    const auto& lw = layer_weights_[layer_idx];
    const int64_t h = hidden_size(), seq_len = chunk.len;
    const bool is_last_layer = (layer_idx == num_layers() - 1);
    // post-attention RMSNorm: residual1 = post_norm(stream); input_norm keeps the stream.
    if (action_mode_) {
        load_adaptive_mod_row(2 * layer_idx + 1);
        apply_adaptive_norm(
            addr(0, "input_norm"), addr(0, "residual1"), chunk);
    } else {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "input_norm"), addr(0, "residual1"),
            layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, eps_);
    }
    // MLP (SwiGLU) + reduce + residual → residual1.
    emit_mlp_pipeline(lw.gate_w, lw.up_w, lw.down_w, seq_len,
                      use_silu_ ? ActivationKind::SILU : ActivationKind::NONE,
                      {}, {}, {},
                      /*gate_nvfp4_ts_addr=*/0,
                      /*up_nvfp4_ts_addr=*/0,
                      /*down_nvfp4_ts_addr=*/0,
                      /*nvfp4_layer_id=*/0,
                      /*force_twostage_chunk_v2=*/false,
                      /*down_out_bf16=*/false,
                      /*residual_is_bf16=*/false,
                      /*acc32=*/linear_acc32_);
    if (action_mode_) {
        apply_adaptive_residual_gate(
            addr(0, "residual1"), addr(0, "residual2"), chunk);
    }
    // Split the last decoder-layer output from final norm for debug-only
    // localization. This fixed target belongs only to the independent debug
    // graph and remains stable across its same-shape REPLAY.
    if (is_last_layer && get_debug_export() && !action_mode_
        && ctx().seq_len > 1 && last_layer_decoder_debug_buf_.defined()) {
        c10::Half* decoder_base =
            last_layer_decoder_debug_buf_.data_ptr<c10::Half>()
            + chunk.offset * h;
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "residual1"), decoder_base, chunk.len * h);
    }
    // final_norm fuse at the last layer (Qwen3.5 last layer is full_attention).
    if (is_last_layer) {
        if (action_mode_) {
            load_adaptive_mod_row(2 * num_layers());
            apply_adaptive_norm(
                addr(0, "residual1"), addr(0, "residual1"), chunk);
        } else {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual1"), addr(0, "residual1"),
                addr(0, "final_norm_w"), seq_len, h, eps_);
        }
    }
    // Export the post-layer residual stream at physical prefill length. For
    // the last layer this deliberately runs after final_norm, matching the
    // localization harness's layer-23 CPU reference. Chunk offset fills the
    // corresponding rows of [layers,1,physical_seq,hidden].
    if (get_debug_export() && !action_mode_ && ctx().seq_len > 1
        && per_layer_debug_buf_.defined()) {
        c10::Half* layer_base = per_layer_debug_buf_.data_ptr<c10::Half>()
            + layer_idx * per_layer_debug_buf_.size(1)
                * per_layer_debug_buf_.size(2) * h
            + chunk.offset * h;
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "residual1"), layer_base, chunk.len * h);
    }
    if (!ctx().output_to_spm && !(action_step_active_ && is_last_layer)) {
        emit_layer_output_dma(layer_idx, chunk);
    }
}

// ── declare_buffers: union of full-attention and GDN buffer sets; per-layer
//    norm preloads are guarded so empty attention slots on GDN layers are not
//    read. ───────────────────────────────────────────────────────────────────
std::vector<BufferDecl> Qwen3_5Model::declare_buffers(const LayoutContext& ctx) {
    int64_t cs = ctx.chunk_size;
    int64_t h  = hidden_size();
    int64_t nq = num_q_heads();
    int64_t nkv = num_kv_heads();
    int64_t hd = head_dim();
    int64_t is_ = intermediate_size();
    int64_t nl  = num_layers();

    // num_kv_heads() is the adapter-expanded effective KV width, so tp resolves
    // to NUM_CORES and each core owns one whole effective KV head. q/k/v/output/
    // attn_gate are sized per active core; MLP also spans NUM_CORES.
    int64_t tp = attn_tp();
    int64_t local_q  = nq / tp;
    int64_t local_kv = nkv * hd / tp;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

    int64_t res = A(cs * h * DWIDTH);
    int64_t q   = A(cs * local_q * hd * DWIDTH);
    int64_t kv  = A(cs * local_kv * DWIDTH);
    int64_t mlp = A(cs * (is_ / NUM_CORES) * DWIDTH);
    int64_t nw  = A(h * DWIDTH);
    int64_t hnw = A(hd * DWIDTH);
    int64_t tmp = A(sdpa_compute_tmp_v16_size(make_sdpa_config(), cs) * 32);

    std::vector<BufferDecl> decls;
    decls.push_back({"residual1",  res,  1, 8, StorageClass::Temp, 0, nullptr});
    decls.push_back({"input_norm", res,  1, 8, StorageClass::Temp, 0, nullptr});
    decls.push_back({"residual2",  0,    0, 0, StorageClass::Temp, 0, "input_norm"});
    decls.push_back({"q",          q,    2, 5, StorageClass::Temp, 0, nullptr});
    decls.push_back({"k",          kv,   2, 4, StorageClass::Temp, 0, nullptr});
    decls.push_back({"v",          kv,   2, 4, StorageClass::Temp, 0, nullptr});
    decls.push_back({"output",     q,    3, 5, StorageClass::Temp, 0, nullptr});
    decls.push_back({"attn_gate",  q,    2, 5, StorageClass::Temp, 0, nullptr});
    decls.push_back({"oproj",      res,  5, 5, StorageClass::Temp, 0, nullptr});
    decls.push_back({"sdpa_tmp",   tmp,  5, 5, StorageClass::Temp, 0, nullptr});
    decls.push_back({"gate",       mlp,  7, 8, StorageClass::Temp, 0, nullptr});
    decls.push_back({"up",         mlp,  7, 7, StorageClass::Temp, 0, nullptr});
    decls.push_back({"down",       res,  7, 8, StorageClass::Temp, 0, nullptr});
    if (action_mode_) {
        decls.push_back({
            // RTC carries clean/noisy AdaLN triples side-by-side. Reserve both
            // halves for every action graph so toggling RTC never changes the
            // SPM manifest; non-RTC loads and reads only the first half.
            "adaptive_mod", A(6 * h * DWIDTH), 1, 8,
            StorageClass::Temp, 0, nullptr});
    }
    if (action_input_w_.defined()) {
        decls.push_back({
            "action_input", A(cs * action_dim_pad_ * DWIDTH), 0, 9,
            StorageClass::Temp, 0, nullptr});
        decls.push_back({
            "action_hidden", A(cs * h * DWIDTH), 0, 0,
            StorageClass::Temp, 0, nullptr});
        decls.push_back({
            "action_keep_mask", A(action_dim_pad_ * DWIDTH), 8, 9,
            StorageClass::Temp, 0, nullptr});
        decls.push_back({
            "action_velocity", A(cs * action_dim_pad_ * DWIDTH), 8, 9,
            StorageClass::Temp, 0, nullptr});
        {
            BufferDecl d;
            d.name = "action_input_bias";
            d.size = A(h * DWIDTH);
            d.storage = StorageClass::Persistent;
            d.preload_callback = [this](FusedModelBase&, int, uint32_t core0_addr) {
                rpu_launch_ddr_broadcast_spm_dma(
                    action_input_b_.data_ptr<c10::Half>(), hidden_size(),
                    core0_addr, /*num_cores=*/1);
            };
            decls.push_back(d);
        }
        {
            BufferDecl d;
            d.name = "action_output_bias";
            d.size = A(action_dim_pad_ * DWIDTH);
            d.storage = StorageClass::Persistent;
            d.preload_callback = [this](FusedModelBase&, int, uint32_t core0_addr) {
                rpu_launch_ddr_broadcast_spm_dma(
                    action_output_b_.data_ptr<c10::Half>(), action_dim_pad_,
                    core0_addr, /*num_cores=*/1);
            };
            decls.push_back(d);
        }
    }

    // Per-layer input_layernorm weight, preloaded for ALL layers (full + GDN now share
    // the unified input_norm channel; the GDN mixers read this persistent "norm_w"
    // instead of a per-forward gdn_innorm_w DMA).
    {
        BufferDecl d;
        d.name = "norm_w"; d.size = nw;
        d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl;
        d.preload_callback = [this](FusedModelBase&, int L, uint32_t core0_addr) {
            rpu_launch_ddr_broadcast_spm_dma(
                layer_weights_[L].input_norm_w.data_ptr<c10::Half>(),
                hidden_size(), core0_addr);
        };
        decls.push_back(d);
    }
    {
        BufferDecl d;
        d.name = "post_norm_w"; d.size = nw;
        d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl;
        d.preload_callback = [this](FusedModelBase&, int L, uint32_t core0_addr) {
            // both full and GDN layers do the standard post_norm + MLP half
            rpu_launch_ddr_broadcast_spm_dma(
                layer_weights_[L].post_norm_w.data_ptr<c10::Half>(),
                hidden_size(), core0_addr);
        };
        decls.push_back(d);
    }
    if (has_qk_norm_) {
        {
            BufferDecl d;
            d.name = "q_norm_w"; d.size = hnw;
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl;
            d.preload_callback = [this](FusedModelBase&, int L, uint32_t core0_addr) {
                if (!layer_is_full_[L]) return;
                rpu_launch_ddr_broadcast_spm_dma(
                    layer_weights_[L].q_norm_w.data_ptr<c10::Half>(),
                    head_dim(), core0_addr);
            };
            decls.push_back(d);
        }
        {
            BufferDecl d;
            d.name = "k_norm_w"; d.size = hnw;
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl;
            d.preload_callback = [this](FusedModelBase&, int L, uint32_t core0_addr) {
                if (!layer_is_full_[L]) return;
                rpu_launch_ddr_broadcast_spm_dma(
                    layer_weights_[L].k_norm_w.data_ptr<c10::Half>(),
                    head_dim(), core0_addr);
            };
            decls.push_back(d);
        }
    }
    {
        BufferDecl d;
        d.name = "final_norm_w"; d.size = nw;
        d.storage = StorageClass::Persistent; d.per_layer = 0;
        d.preload_callback = [this](FusedModelBase&, int, uint32_t core0_addr) {
            rpu_launch_ddr_broadcast_spm_dma(
                final_norm_w_.data_ptr<c10::Half>(), hidden_size(), core0_addr);
        };
        decls.push_back(d);
    }

    // ── GDN buffers. A forward runs exactly one mode: decode resolves below 64 rows;
    //    adapter-padded prefill resolves to a multiple of 64. Declare only the active
    //    set so an unused mode cannot inflate the planner's SPM peak. ──
    if (has_gdn_) {
        const int64_t H = gdn_nvh_, Dk = gdn_dk_, Dv = gdn_dv_;
        const int64_t conv_dim = gdn_conv_dim_, Kc = gdn_kc_, hidden = hidden_size();
        const int     nc = NUM_CORES;
        const int64_t Hc = H / nc, lconv = conv_dim / nc, lval = (H * Dv) / nc;
        const int64_t nkh = (conv_dim - H * Dv) / (2 * Dk), lkey = (nkh * Dk) / nc;
        const int64_t n_bg = ((Hc + 15) / 16) * 16, hist = Kc - 1;
        const int64_t CS = std::max<int64_t>(64, ctx.chunk_size), N = CS / 64;
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
        auto B = [&](const char* n, int64_t elems) -> BufferDecl {
            return {n, A(elems * DWIDTH), 1, 6, StorageClass::Temp, 0, nullptr};
        };
        // BP: phase-tagged LayerWide Temp. This model is SEQUENTIAL, not KV_FIRST;
        // the phase windows therefore share the layer-loop timeline with attention/MLP.
        auto BP = [&](const char* n, int64_t elems, int p0, int p1) -> BufferDecl {
            return {n, A(elems * DWIDTH), p0, p1, StorageClass::Temp, 0, nullptr};
        };
        // Setup/state buffers are shared by both GDN modes.
        decls.insert(decls.end(), {
            B("gdn_normw", Dv), B("gdn_zero", Dk * Dv),
            B("gdn_Al", 8), B("gdn_dt", 8),
            B("gdn_state", Hc * Dk * Dv), B("gdn_cs", Kc * lconv),
        });
        // ── GDN DECODE (mixer) buffers — 8-core, Hc heads/core ──
        if (ctx.chunk_size < 64) {
            decls.insert(decls.end(), {
                B("gdn_q", lkey), B("gdn_k", lkey), B("gdn_v", lval),       // per-path proj/conv input
                B("gdn_qa", lkey), B("gdn_ka", lkey), B("gdn_va", lval),    // per-path post-conv (silu)
                B("gdn_q_rep", Hc * Dk), B("gdn_k_rep", Hc * Dk),  // GQA-tiled q/k (Hc heads)
                B("gdn_z", lval), B("gdn_a", n_bg), B("gdn_b", n_bg),
                B("gdn_p", Hc * Dk * Dv), B("gdn_delta", Hc * Dv),
                B("gdn_gexp", Hc), B("gdn_beta", Hc), B("gdn_gs", Hc),
                B("gdn_out", Hc * Dv), B("gdn_gated", lval), B("gdn_m", hidden),
            });
        }
        // ── GDN PREFILL (chunk) buffers — phase-tagged for lifecycle aliasing. Phases:
        //   1 proj | 2 conv+prep+tile | 3 transpose | 4 M-gen | 5 recurrent | 6 tail.
        //   WITHOUT phases the L=256 set sums ~10MB and OOMs; WITH them peak ~3.7MB. The
        //   build_gdn prefill branch MUST NOT read a buffer after its phase_end. ──
        if (ctx.chunk_size >= 64) decls.insert(decls.end(), {
            // ---- phase 1: proj ----  (z spans to the tail; h_in stays in residual1;
            // input_layernorm is at the layer level, reading shared "input_norm".)
            BP("gdn_c_z",      CS * lval,   1, 6),
            BP("gdn_c_b",      CS * n_bg,   1, 3), BP("gdn_c_a", CS * n_bg, 1, 3),
            // ---- phase 2: per-path conv-prefill (pad = [hist+L, l*]) + silu ----
            BP("gdn_c_q_pad", (hist + CS) * lkey, 1, 2), BP("gdn_c_k_pad", (hist + CS) * lkey, 1, 2),
            BP("gdn_c_v_pad", (hist + CS) * lval, 1, 2),
            // conv carry (cso) is written straight into the LayerWide gdn_cs at path offsets — no
            // separate gdn_c_*_cso buffers (shared with decode; tail scatters gdn_cs → cache).
            BP("gdn_c_q_halo", 8 * hist * lkey, 2, 2), BP("gdn_c_k_halo", 8 * hist * lkey, 2, 2),
            BP("gdn_c_v_halo", 8 * hist * lval, 2, 2),
            BP("gdn_c_query", CS * lkey, 2, 4), BP("gdn_c_key", CS * lkey, 2, 4),
            BP("gdn_c_value", CS * lval, 2, 4),
            // ---- phase 3: prep (beta/g [L,vg_c=Hc]) + GQA tile ----
            BP("gdn_c_beta",  CS * Hc,  3, 4), BP("gdn_c_g", CS * Hc, 3, 4),
            BP("gdn_c_q_rep", CS * Hc * Dk, 3, 4), BP("gdn_c_k_rep", CS * Hc * Dk, 3, 4),
            // ---- phase 3: token->N-major transpose [N,Hc,C,*] (== [Hc*CS,*] total) ----
            BP("gdn_c_q_h",    Hc * CS * Dk, 3, 5), BP("gdn_c_k_h", Hc * CS * Dk, 3, 5),
            BP("gdn_c_v_h",    Hc * CS * Dv, 3, 4),
            BP("gdn_c_beta_h", Hc * CS, 3, 4), BP("gdn_c_g_h", Hc * CS, 3, 4),
            // ---- phase 4: M-gen [N,Hc,C,*] ----
            BP("gdn_c_kbeta",  Hc * CS * Dk, 4, 4), BP("gdn_c_vbeta", Hc * CS * Dv, 4, 4),
            BP("gdn_c_gcumsum", Hc * CS,        4, 5),
            BP("gdn_c_decay",   Hc * N * 64 * 64, 4, 5),
            BP("gdn_c_attn",    Hc * N * 64 * 64, 4, 4),
            BP("gdn_c_vnew",    Hc * CS * Dv,   4, 5),
            BP("gdn_c_kcd",     Hc * CS * Dk,   4, 5),
            BP("gdn_c_gexp",    Hc * CS,        4, 5),
            // Host-preloaded masks are loaded in phase 1 and read in phase 4, so
            // their scope spans [1,4]. M *= -1 and q *= qk_scale remain separate.
            BP("gdn_c_tril",   64 * 64, 1, 4), BP("gdn_c_strict", 64 * 64, 1, 4),
            BP("gdn_c_ws",     Hc * CS, 4, 4),           // cumsum workspace
            // ---- phase 5: recurrent per-chunk scratch [Hc,C,*] ----
            // q/k/v/kcd/decay/g/gexp/out use offset views into phase-4 storage.
            BP("gdn_c_attnc", Hc * 64 * 64, 5, 5),
            BP("gdn_c_recout", Hc * 64 * Dv, 5, 5),
            BP("gdn_c_recstate", Hc * Dk * Dv, 5, 5), BP("gdn_c_glast", Hc, 5, 5),
            BP("gdn_c_coreout", Hc * CS * Dv, 5, 6),
            // ---- phase 6: tail ----
            BP("gdn_c_out_tok", CS * Hc * Dv, 6, 6),
            BP("gdn_c_gated",   CS * lval,    6, 6),
            BP("gdn_c_m",       CS * hidden,  6, 6),
        });
    }
    return decls;
}


ModelStaticConfig Qwen3_5Model::static_config() {
    ModelStaticConfig cfg;
    cfg.num_layers = num_layers();
    cfg.cross_layer_batch_size = num_layers();   // single group (mirror Qwen3)
    cfg.fast_replay_skip_layer_loop = fast_replay_active_;
    if (action_step_active_) {
        cfg.pre_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
            &Qwen3_5Model::emit_action_input_projection);
        cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
            &Qwen3_5Model::emit_action_output_projection);
        cfg.body_iterations = action_loop_active_ ? action_num_steps_ : 1;
    }
    return cfg;
}

ModelDynamicConfig Qwen3_5Model::dynamic_config(const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    return cfg;
}

bool Qwen3_5Model::subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                             int64_t position) const {
    if (action_mode_) {
        // G0.5 action attention is bidirectional and unpadded. It must run as
        // one whole suffix chunk so every query sees every action key. Validate
        // both sparse layer families: shared VLM prefix and action-only.
        return cs >= seq_len
            && sdpa_is_valid_chunk_size(
                make_sdpa_config(/*mask=*/0), cs, seq_len, position)
            && sdpa_is_valid_chunk_size(
                make_sdpa_config(/*mask=*/0), cs, seq_len, /*position=*/0);
    }
    if (!sdpa_is_valid_chunk_size(make_sdpa_config(), cs, seq_len, position))
        return false;
    // Decode uses the recurrent single-token path and does not have the 64-row
    // chunk constraint. Prefill consists of C=64 GDN sub-chunks.
    if (seq_len <= 1 || !has_gdn_) return true;
    if (cs % 64 != 0) return false;

    // Certified envelope, checked per candidate. Unlike CausalDecoderModel this
    // class already closed the explicit-override route below (the cap check was
    // always in the VALIDITY hook, not the cap hook), so the envelope only has to
    // join it here. The deny-by-default half lives in subclass_chunk_size_cap.
    if (!chunk_within_envelope(cs)) return false;

    // Optional cold per-handle cap. The planner still chooses a fitting valid
    // value; this is not an exact chunk-size override.
    return chunk_size_cap_ <= 0 || cs <= chunk_size_cap_;
}

// Qwen3.5 had no cap hook; the auto scan was bounded only by the
// validity predicate above. The override route also needs a once-per-resolve
// gate so an undeclared handle fails loudly before SPM allocation on both
// planner entry points.
int64_t Qwen3_5Model::subclass_chunk_size_cap(int64_t seq_len,
                                              int64_t position) const {
    // G0.5 action mode is a fixed all-full-attention expert, not a Qwen3.5
    // text-prefill profile. Its action-length override is checked separately.
    if (action_mode_) return 0;
    const int64_t envelope_chunk =
        enforce_chunk_envelope(seq_len, position, "Qwen3_5Model");
    if (chunk_size_cap_ <= 0) return envelope_chunk;
    if (envelope_chunk <= 0) return chunk_size_cap_;
    return std::min(chunk_size_cap_, envelope_chunk);
}

// Per-forward prefill M-RoPE tables. Generic Qwen3.5 uses a fresh oneshot graph;
// fixed-profile retained graphs rely on same-shape copy_ preserving this address.
void Qwen3_5Model::set_prefill_rope(const at::Tensor& cos, const at::Tensor& sin) {
    TORCH_CHECK(cos.defined() && sin.defined() && cos.dim() == 2 && sin.dim() == 2,
                "set_prefill_rope: cos/sin must be 2D [N, rotary_dim/2]");
    TORCH_CHECK(cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf,
                "set_prefill_rope: cos/sin must be FP16");
    TORCH_CHECK(cos.device().type() == at::kPrivateUse1
                && sin.device().type() == at::kPrivateUse1,
                "set_prefill_rope: cos/sin must be on RPU device");
    TORCH_CHECK(cos.sizes() == sin.sizes(),
                "set_prefill_rope: cos/sin shapes must match");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
                "set_prefill_rope: cos/sin must be contiguous");
    if (prefill_cos_.defined() && prefill_cos_.sizes() == cos.sizes()) {
        // Retain the DDR addresses baked into an existing action GraphCache
        // entry. Text prefill also benefits from reusing the aligned backing.
        prefill_cos_.copy_(cos);
        prefill_sin_.copy_(sin);
        return;
    }
    prefill_cos_ = keep_256b_aligned_rpu_copy(cos, "set_prefill_rope: cos");
    prefill_sin_ = keep_256b_aligned_rpu_copy(sin, "set_prefill_rope: sin");
}

// ═══════════════════════════ SETUP (weights) ═══════════════════════════════

// ── set_weights: store full-layer weights; GDN slots stay empty ──────────────
void Qwen3_5Model::set_weights(
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
    at::TensorList gdn_b_bg_list, at::TensorList gdn_a_bg_list) {

    const int64_t N = static_cast<int64_t>(layer_is_full.size());
    TORCH_CHECK(N > 0, "qwen3_5 set_weights: layer_is_full must be non-empty");
    const auto check_list = [N](at::TensorList list, const char* name) {
        TORCH_CHECK(static_cast<int64_t>(list.size()) == N,
                    "qwen3_5 set_weights: ", name,
                    " must have num_layers=", N, " entries, got ", list.size());
    };
    check_list(q_w_list, "q_w_list");
    check_list(k_w_list, "k_w_list");
    check_list(v_w_list, "v_w_list");
    check_list(o_w_list, "o_w_list");
    check_list(q_norm_list, "q_norm_list");
    check_list(k_norm_list, "k_norm_list");
    check_list(attn_gate_list, "attn_gate_list");
    check_list(input_norm_list, "input_norm_list");
    check_list(post_norm_list, "post_norm_list");
    check_list(gate_list, "gate_list");
    check_list(up_list, "up_list");
    check_list(down_list, "down_list");
    check_list(gdn_in_z_list, "gdn_in_z_list");
    check_list(gdn_out_list, "gdn_out_list");
    check_list(gdn_A_log_list, "gdn_A_log_list");
    check_list(gdn_dt_bias_list, "gdn_dt_bias_list");
    check_list(gdn_norm_list, "gdn_norm_list");
    check_list(gdn_q_list, "gdn_q_list");
    check_list(gdn_k_list, "gdn_k_list");
    check_list(gdn_v_list, "gdn_v_list");
    check_list(gdn_cq_list, "gdn_cq_list");
    check_list(gdn_ck_list, "gdn_ck_list");
    check_list(gdn_cv_list, "gdn_cv_list");
    check_list(gdn_b_bg_list, "gdn_b_bg_list");
    check_list(gdn_a_bg_list, "gdn_a_bg_list");

    const auto check_tensor = [](const at::Tensor& tensor, const char* name,
                                 int64_t layer, int64_t dim) {
        TORCH_CHECK(tensor.defined() && tensor.numel() > 0,
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] must be non-empty");
        TORCH_CHECK(tensor.device().type() == at::kPrivateUse1
                    && tensor.scalar_type() == at::kHalf
                    && tensor.is_contiguous(),
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] must be a contiguous FP16 RPU tensor");
        TORCH_CHECK(tensor.dim() == dim,
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] must be ", dim, "D, got ", tensor.dim(), "D");
    };
    for (int64_t i = 0; i < N; ++i) {
        TORCH_CHECK(layer_is_full[i] == 0 || layer_is_full[i] == 1,
                    "qwen3_5 set_weights: layer_is_full[", i,
                    "] must be 0 or 1, got ", layer_is_full[i]);
        check_tensor(input_norm_list[i], "input_norm_list", i, 1);
        check_tensor(post_norm_list[i], "post_norm_list", i, 1);
        check_tensor(gate_list[i], "gate_list", i, 2);
        check_tensor(up_list[i], "up_list", i, 2);
        check_tensor(down_list[i], "down_list", i, 2);
        if (layer_is_full[i]) {
            check_tensor(q_w_list[i], "q_w_list", i, 2);
            check_tensor(k_w_list[i], "k_w_list", i, 2);
            check_tensor(v_w_list[i], "v_w_list", i, 2);
            check_tensor(o_w_list[i], "o_w_list", i, 2);
            check_tensor(q_norm_list[i], "q_norm_list", i, 1);
            check_tensor(k_norm_list[i], "k_norm_list", i, 1);
            check_tensor(attn_gate_list[i], "attn_gate_list", i, 2);
        } else {
            check_tensor(gdn_in_z_list[i], "gdn_in_z_list", i, 2);
            check_tensor(gdn_out_list[i], "gdn_out_list", i, 2);
            check_tensor(gdn_A_log_list[i], "gdn_A_log_list", i, 1);
            check_tensor(gdn_dt_bias_list[i], "gdn_dt_bias_list", i, 1);
            check_tensor(gdn_norm_list[i], "gdn_norm_list", i, 1);
            check_tensor(gdn_q_list[i], "gdn_q_list", i, 2);
            check_tensor(gdn_k_list[i], "gdn_k_list", i, 2);
            check_tensor(gdn_v_list[i], "gdn_v_list", i, 2);
            check_tensor(gdn_cq_list[i], "gdn_cq_list", i, 3);
            check_tensor(gdn_ck_list[i], "gdn_ck_list", i, 3);
            check_tensor(gdn_cv_list[i], "gdn_cv_list", i, 3);
            check_tensor(gdn_b_bg_list[i], "gdn_b_bg_list", i, 2);
            check_tensor(gdn_a_bg_list[i], "gdn_a_bg_list", i, 2);
        }
    }
    check_tensor(cos, "cos", -1, 2);
    check_tensor(sin, "sin", -1, 2);
    check_tensor(final_norm_w, "final_norm_w", -1, 1);
    TORCH_CHECK(cos.sizes() == sin.sizes(),
                "qwen3_5 set_weights: cos/sin shapes must match");

    has_qk_norm_ = q_norm_list.size() > 0;
    use_silu_    = use_silu;
    eps_         = eps;

    layer_is_full_.assign(layer_is_full.begin(), layer_is_full.end());
    has_gdn_ = std::any_of(
        layer_is_full_.begin(), layer_is_full_.end(),
        [](uint8_t is_full) { return !is_full; });
    action_mode_ = false;
    action_rtc_active_ = false;
    action_rtc_prefix_len_ = 0;
    action_rtc_prefix_ref_ = at::Tensor();
    action_rtc_prefix_live_base_ = 0;
    adaptive_mod_ref_ = at::Tensor();
    action_prefix_lens_.clear();
    layer_weights_.assign(N, LayerWeights{});
    for (int64_t i = 0; i < N; ++i) {
        auto& lw = layer_weights_[i];
        // Every layer (full + GDN) has input_layernorm (unified channel), post_norm, MLP.
        lw.input_norm_w = input_norm_list[i];
        lw.post_norm_w = post_norm_list[i];
        lw.gate_w = gate_list[i]; lw.up_w = up_list[i]; lw.down_w = down_list[i];
        if (!layer_is_full_[i]) {
            // GDN mixer weights (attention slots stay empty).
            lw.gdn_in_z_w    = gdn_in_z_list[i];
            lw.gdn_out_w     = gdn_out_list[i];
            lw.gdn_A_log     = gdn_A_log_list[i];
            lw.gdn_dt_bias   = gdn_dt_bias_list[i];
            lw.gdn_norm_w    = gdn_norm_list[i];
            // prefill (chunk) per-path weights (separate q/k/v proj + conv, N_bg b/a)
            if (gdn_q_list.size() > 0) {
                lw.gdn_q_w = gdn_q_list[i]; lw.gdn_k_w = gdn_k_list[i]; lw.gdn_v_w = gdn_v_list[i];
                lw.gdn_conv_q_w = gdn_cq_list[i]; lw.gdn_conv_k_w = gdn_ck_list[i]; lw.gdn_conv_v_w = gdn_cv_list[i];
                lw.gdn_b_bg_w = gdn_b_bg_list[i]; lw.gdn_a_bg_w = gdn_a_bg_list[i];
            }
            continue;
        }
        lw.q_w = q_w_list[i]; lw.k_w = k_w_list[i];
        lw.v_w = v_w_list[i]; lw.o_w = o_w_list[i];
        lw.q_norm_w    = has_qk_norm_ ? q_norm_list[i] : at::Tensor();
        lw.k_norm_w    = has_qk_norm_ ? k_norm_list[i] : at::Tensor();
        lw.attn_gate_w = attn_gate_list[i];
    }
    cos_ = keep_256b_aligned_rpu_copy(cos, "qwen3_5 set_weights: cos");
    sin_ = keep_256b_aligned_rpu_copy(sin, "qwen3_5 set_weights: sin");
    final_norm_w_ = final_norm_w;

    // GDN mixer dims + persistent DDR scratch (residual bridge + zeros).
    gdn_nvh_      = gdn_num_v_heads;
    gdn_dk_       = gdn_key_head_dim;
    gdn_dv_       = gdn_value_head_dim;
    gdn_conv_dim_ = gdn_conv_dim;
    gdn_kc_       = gdn_conv_kernel;
    gdn_value_dim_ = gdn_nvh_ * gdn_dv_;
    if (has_gdn_) {
        auto opt = final_norm_w.options();   // fp16 on RPU
        gdn_zero_ = at::zeros({gdn_dk_ * gdn_dv_}, opt);
        // prefill-chunk constant masks (C=64 fixed). tril keeps the diagonal (decay
        // self-score exp(0)=1); strict is the strict-lower (i>j). Both match rhino's
        // host-prepared constant masks (the -1 is a separate binary_scalar, not folded
        // here). Built on CPU then moved to RPU (host-side fill; values exact).
        {
            const int64_t C = 64;
            auto cpuf = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
            auto ones = at::ones({C, C}, cpuf);
            gdn_tril_   = at::tril(ones).reshape({C * C}).to(opt);      // i>=j -> 1
            gdn_strict_ = at::tril(ones, -1).reshape({C * C}).to(opt);  // i>j  -> 1
        }
        // per-layer LIVE state addrs for the mutable GDN state/conv DMAs. Sized ONCE
        // here (never resized) so the graph-recorded &..._live_addr_[L] stay stable;
        // build_gdn refreshes each entry per forward before emitting the mutable DMA.
        gdn_state_live_addr_.assign(N, 0);
        conv_state_live_addr_.assign(N, 0);
        // per-layer neg_exp_A = -exp(A_log) (host; chunk multiplies g by this).
        gdn_neg_exp_A_.assign(N, at::Tensor());
        for (int64_t i = 0; i < N; ++i) {
            if (layer_is_full_[i]) continue;
            const auto& al = layer_weights_[i].gdn_A_log;
            if (al.defined() && al.numel() > 0)
                gdn_neg_exp_A_[i] = at::exp(al.to(at::kCPU).to(at::kFloat)).mul(-1.0).to(opt);
        }
    }

    set_model_params(num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size);
    set_num_layers(N);

    // M-RoPE setup. mrope_section is [T,H,W] in rotary_dim/2 units (Qwen3.5:
    // [11,11,10], sum=32=rotary_dim/2), so rotary_dim=2*sum(mrope_section).
    // The host bakes the T/H/W interleaving into [N, rotary_dim/2] cos/sin
    // tables; the partial kernels receive rotary_dim and need no strobe masks
    // or position_ids.
    has_mrope_ = !mrope_section.empty();
    if (has_mrope_) {
        int64_t sec_sum = 0;
        for (auto s : mrope_section) sec_sum += s;
        rotary_dim_ = 2 * sec_sum;
    }

    invalidate_model_state();   // Must remain the last statement.
}

}  // namespace v3

// =============================================================================
// Instance registry + C API for TORCH_LIBRARY_IMPL (registered in rpu_backend.cpp)
// =============================================================================
using Qwen3_5Registry = ModelHandleRegistry<v3::Qwen3_5Model>;

namespace v3::qwen3_5_z2_internal {

SpmPipelineComponentLayout prepare_text(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len) {
    return Qwen3_5Registry::get(handle, "qwen3_5_z2_prepare_text")
        ->prepare_z2_layout(execution_len, real_len);
}

SpmDense2DSpec text_storage_spec(int64_t handle, int64_t execution_len) {
    return Qwen3_5Registry::get(handle, "qwen3_5_z2_text_storage_spec")
        ->z2_storage_spec(execution_len);
}

void adopt_text(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    Qwen3_5Registry::get(handle, "qwen3_5_z2_adopt_text")
        ->adopt_z2_layout(lease, scratch);
}

void bind_text(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmPortView& storage) {
    Qwen3_5Registry::get(handle, "qwen3_5_z2_bind_text")
        ->bind_z2_port(lease, storage);
}

void validate_text(int64_t handle, const SpmPipelineLease& lease) {
    Qwen3_5Registry::get(handle, "qwen3_5_z2_validate_text")
        ->validate_z2_layout(lease);
}

void clear_text(int64_t handle, uint64_t epoch, uint64_t plan_hash) {
    Qwen3_5Registry::get(handle, "qwen3_5_z2_clear_text")
        ->clear_z2_layout(epoch, plan_hash);
}

at::Tensor forward_text_z2(
    int64_t handle,
    const at::Tensor& hidden,
    at::TensorList k_caches,
    at::TensorList v_caches,
    at::TensorList gdn_states,
    at::TensorList conv_states,
    uint64_t epoch,
    uint64_t plan_hash) {
    validate_text_dispatch(handle, epoch, plan_hash);
    auto* model = Qwen3_5Registry::get(handle, "qwen3_5_z2_forward_text");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    std::vector<at::Tensor> gs(gdn_states.begin(), gdn_states.end());
    std::vector<at::Tensor> cs(conv_states.begin(), conv_states.end());
    return model->forward_z2(hidden, kc, vc, gs, cs, epoch, plan_hash);
}

}  // namespace v3::qwen3_5_z2_internal

int64_t rpu_qwen3_5_create() { return Qwen3_5Registry::create(); }
void    rpu_qwen3_5_destroy(int64_t handle) {
    v3::qwen3_5_z2_internal::check_text_destroy_allowed(handle);
    Qwen3_5Registry::destroy(handle, "rpu_qwen3_5_destroy");
}

// Per-handle resolved prefill chunk size. Return the recorded prefill value when
// available; before prefill, fall back to the latest base resolved size.
int64_t rpu_qwen3_5_get_resolved_chunk_size(int64_t handle) {
    auto* m = Qwen3_5Registry::get(handle, "rpu_qwen3_5_get_resolved_chunk_size");
    const int64_t prefill_cs = m->last_prefill_chunk_size();
    return prefill_cs > 0 ? prefill_cs : m->get_last_resolved_chunk_size();
}

void rpu_qwen3_5_set_chunk_size_cap(int64_t handle, int64_t cap) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_chunk_size_cap")
        ->set_chunk_size_cap(cap);
}

void rpu_qwen3_5_set_prefill_chunk_size(int64_t handle, int64_t chunk_size) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_prefill_chunk_size")
        ->set_prefill_chunk_size(chunk_size);
}

void rpu_qwen3_5_set_chunk_envelope(int64_t handle, int64_t max_kv_len,
                                    int64_t chunk) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_qwen3_5_set_linear_acc32(int64_t handle, bool enabled) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_linear_acc32")
        ->set_linear_acc32(enabled);
}

void rpu_qwen3_5_set_fast_replay(int64_t handle, bool enabled) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_fast_replay")
        ->set_fast_replay(enabled);
}

void rpu_qwen3_5_enable_action_mode(int64_t handle) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_enable_action_mode")
        ->enable_action_mode();
}

void rpu_qwen3_5_set_action_io_weights(
    int64_t handle,
    const at::Tensor& input_w, const at::Tensor& input_b,
    const at::Tensor& output_w, const at::Tensor& output_b,
    int64_t action_dim, int64_t action_dim_pad, int64_t action_len) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_action_io_weights")
        ->set_action_io_weights(
            input_w, input_b, output_w, output_b,
            action_dim, action_dim_pad, action_len);
}

int64_t rpu_qwen3_5_resolve_prefill_chunk_size(
    int64_t handle, int64_t execution_len) {
    return Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_resolve_prefill_chunk_size")
        ->resolve_prefill_chunk_size(execution_len);
}

void rpu_qwen3_5_set_weights(
    int64_t handle,
    at::TensorList q, at::TensorList k, at::TensorList v, at::TensorList o,
    at::TensorList qn, at::TensorList kn, at::TensorList ag,
    at::TensorList in_norm, at::TensorList post_norm,
    at::TensorList gate, at::TensorList up, at::TensorList down,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm,
    at::IntArrayRef layer_is_full,
    int64_t nqh, int64_t nkvh, int64_t hd, int64_t hs, int64_t is_,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::TensorList gdn_in_z, at::TensorList gdn_out, at::TensorList gdn_A_log,
    at::TensorList gdn_dt_bias, at::TensorList gdn_norm,
    int64_t gdn_nvh, int64_t gdn_dk, int64_t gdn_dv, int64_t gdn_conv_dim,
    int64_t gdn_conv_kernel,
    at::TensorList gdn_q, at::TensorList gdn_k, at::TensorList gdn_v,
    at::TensorList gdn_cq, at::TensorList gdn_ck, at::TensorList gdn_cv,
    at::TensorList gdn_b_bg, at::TensorList gdn_a_bg) {
    auto* m = Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_weights");
    m->set_weights(q, k, v, o, qn, kn, ag, in_norm, post_norm, gate, up, down,
                   cos, sin, final_norm, layer_is_full,
                   nqh, nkvh, hd, hs, is_, eps, use_silu, mrope_section,
                   gdn_in_z, gdn_out, gdn_A_log, gdn_dt_bias, gdn_norm,
                   gdn_nvh, gdn_dk, gdn_dv, gdn_conv_dim, gdn_conv_kernel,
                   gdn_q, gdn_k, gdn_v, gdn_cq, gdn_ck, gdn_cv, gdn_b_bg, gdn_a_bg);
}

void rpu_qwen3_5_set_prefill_rope(int64_t handle, const at::Tensor& cos, const at::Tensor& sin) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_prefill_rope")->set_prefill_rope(cos, sin);
}

void rpu_qwen3_5_set_valid_prefill_len(int64_t handle, int64_t n) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_valid_prefill_len")->set_valid_prefill_len(n);
}

void rpu_qwen3_5_set_mrope_position_delta(int64_t handle, int64_t d) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_mrope_position_delta")->set_mrope_position_delta(d);
}

at::Tensor rpu_qwen3_5_forward(
    int64_t handle, const at::Tensor& hidden,
    at::TensorList k_caches, at::TensorList v_caches, at::TensorList gdn_states,
    at::TensorList conv_states,
    const std::optional<at::Tensor>& mask, int64_t position, bool is_causal) {
    auto* m = Qwen3_5Registry::get(handle, "rpu_qwen3_5_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    std::vector<at::Tensor> gs(gdn_states.begin(), gdn_states.end());
    std::vector<at::Tensor> cs(conv_states.begin(), conv_states.end());
    return m->forward(hidden, kc, vc, gs, cs, mask, position, is_causal);
}

at::Tensor rpu_qwen3_5_action_forward(
    int64_t handle, const at::Tensor& hidden, const at::Tensor& adaptive_mod,
    at::TensorList k_caches, at::TensorList v_caches,
    at::IntArrayRef prefix_lens) {
    auto* m = Qwen3_5Registry::get(handle, "rpu_qwen3_5_action_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return m->forward_action(hidden, adaptive_mod, kc, vc, prefix_lens);
}

at::Tensor rpu_qwen3_5_action_step_forward(
    int64_t handle, const at::Tensor& action,
    const at::Tensor& action_keep_mask, at::Tensor action_out,
    double delta_t, const at::Tensor& adaptive_mod,
    at::TensorList k_caches, at::TensorList v_caches,
    at::IntArrayRef prefix_lens, int64_t num_steps) {
    auto* m = Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_action_step_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return m->forward_action_step(
        action, action_keep_mask, action_out, delta_t,
        adaptive_mod, kc, vc, prefix_lens, num_steps);
}

at::Tensor rpu_qwen3_5_action_rtc_step_forward(
    int64_t handle, const at::Tensor& action,
    const at::Tensor& action_keep_mask, at::Tensor action_out,
    const at::Tensor& action_prefix, int64_t action_prefix_len,
    double delta_t, const at::Tensor& adaptive_mod,
    at::TensorList k_caches, at::TensorList v_caches,
    at::IntArrayRef prefix_lens, int64_t num_steps) {
    auto* m = Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_action_rtc_step_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return m->forward_action_rtc_step(
        action, action_keep_mask, action_out,
        action_prefix, action_prefix_len, delta_t,
        adaptive_mod, kc, vc, prefix_lens, num_steps);
}

at::Tensor rpu_qwen3_5_action_rtc_trace_forward(
    int64_t handle, const at::Tensor& action,
    const at::Tensor& action_keep_mask, at::Tensor action_out,
    const at::Tensor& action_prefix, int64_t action_prefix_len,
    double delta_t, const at::Tensor& adaptive_mod,
    at::TensorList k_caches, at::TensorList v_caches,
    at::IntArrayRef prefix_lens,
    at::Tensor trace_pre_action,
    at::Tensor trace_action_embed,
    at::Tensor trace_final_hidden,
    at::Tensor trace_velocity,
    at::Tensor trace_post_action) {
    auto* m = Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_action_rtc_trace_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return m->forward_action_rtc_trace(
        action, action_keep_mask, action_out,
        action_prefix, action_prefix_len, delta_t,
        adaptive_mod, kc, vc, prefix_lens,
        trace_pre_action, trace_action_embed, trace_final_hidden,
        trace_velocity, trace_post_action);
}
