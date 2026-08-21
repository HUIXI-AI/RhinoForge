// rpu_internvla_nextdit_model.cpp — standalone InternVLA-N1 Lumina NextDiT core.
//
// Scope of the first correctness slice:
//   input x [1,32,384] + per-layer precomputed AdaRMS modulation + precomputed
//   cross K/V -> 12 exact Lumina blocks -> final continuous norm/projection
//   -> hidden [1,32,384].
//
// CPU-side adapter owns caption projection and timestep/caption embedding.
// Production inference fuses the action encoder/position table, action decoder,
// and the complete multi-step Euler loop.
//
// The unusual part is diffusers' `layer_norm_across_heads`: Q/K are normalized
// over the full 384 channels BEFORE reshape to [6,64].  The existing decoder
// QK-norm kernels are per-head RMSNorm and are not semantically reusable.  This
// implementation therefore does head-partitioned projection -> core-0 gather ->
// exact affine LayerNorm -> head-major layout conversion -> 6-core SDPA.

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_profile.h"
#include "rpu_runtime_state.h"

#include <ATen/record_function.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NEXTDIT_DEVICE_CORES 8
#define NEXTDIT_DWIDTH 2
#define NEXTDIT_ACTION_PAD 16
#define NEXTDIT_ACTION_SEQ 32
#define NEXTDIT_MAX_ACTION_BATCH 4
#define NEXTDIT_MAX_CROSS_SEQ 320

namespace v3 {

static bool nextdit_fold_norm_modulation_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_INTERNVLA_NEXTDIT_FOLD_NORM_MOD");
        cached = (e && std::string(e) != "" && std::string(e) != "0" &&
                  std::string(e) != "false" && std::string(e) != "False") ? 1 : 0;
    }
    return cached != 0;
}

class InternVLANextDiTModel : public FusedModelBase {
public:
    struct LayerWeights {
        at::Tensor self_q_w, self_k_w, self_v_w;
        at::Tensor cross_q_w, cross_k_w, cross_v_w, out_w;
        at::Tensor ff1_w, ff2_w, ff3_w;

        at::Tensor norm1_w, norm2_w, ffn_norm1_w, ffn_norm2_w;
        at::Tensor self_qn_w, self_qn_b, self_kn_w, self_kn_b;
        at::Tensor cross_qn_w, cross_qn_b, cross_kn_w, cross_kn_b;
        at::Tensor context_norm_w;
        at::Tensor cross_gate_heads;
    };

    void set_16b_sdpa(bool enabled) {
        if (use_16b_sdpa_ != enabled) {
            use_16b_sdpa_ = enabled;
            invalidate_model_state();
        }
    }

    void set_fused_qkq_weights(at::TensorList weights) {
        TORCH_CHECK(static_cast<int64_t>(weights.size()) == num_layers(),
                    "fused Q/K/cross-Q list must match num_layers");
        fused_qkq_weights_.clear();
        fused_qkq_weights_.reserve(weights.size());
        for (int64_t i = 0; i < static_cast<int64_t>(weights.size()); ++i) {
            const auto& weight = weights[i];
            TORCH_CHECK(weight.dim() == 2
                        && weight.size(0) == 3 * hidden_size()
                        && weight.size(1) == hidden_size()
                        && weight.scalar_type() == at::kHalf
                        && weight.device().type() == at::kPrivateUse1
                        && weight.is_contiguous(),
                        "fused Q/K/cross-Q weight[", i,
                        "] must be contiguous fp16 RPU [3H,H]");
            fused_qkq_weights_.push_back(weight);
        }
        use_fused_qkq_ = true;
        invalidate_model_state();
    }

    void set_step_prepare_weights(
        const at::Tensor& weight, const at::Tensor& bias,
        const at::Tensor& fold_norm_weight)
    {
        const int64_t h = hidden_size();
        const int64_t out_features = (num_layers() * 4 + 1) * h;
        TORCH_CHECK(weight.dim() == 2 && weight.numel() == out_features * h
                    && weight.scalar_type() == at::kHalf
                    && weight.device().type() == at::kPrivateUse1
                    && weight.is_contiguous(),
                    "step-prepare weight must be contiguous fp16 RPU [(4L+1)H,H]");
        TORCH_CHECK(bias.dim() == 1 && bias.numel() == out_features
                    && bias.scalar_type() == at::kHalf
                    && bias.device().type() == at::kPrivateUse1
                    && bias.is_contiguous(),
                    "step-prepare bias must be contiguous fp16 RPU [(4L+1)H]");
        TORCH_CHECK(fold_norm_weight.dim() == 3
                    && fold_norm_weight.sizes()
                        == at::IntArrayRef({4, num_layers(), h})
                    && fold_norm_weight.scalar_type() == at::kHalf
                    && fold_norm_weight.device().type() == at::kPrivateUse1
                    && fold_norm_weight.is_contiguous(),
                    "step-prepare fold weights must be contiguous fp16 RPU [4,L,H]");
        step_prepare_w_ = weight;
        step_prepare_b_ = bias;
        step_fold_norm_w_ = fold_norm_weight;
        have_step_prepare_weights_ = true;
        invalidate_model_state();
    }

    void set_weights(
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
        int64_t ff_intermediate, double eps, double final_norm_eps)
    {
        const int64_t nl = static_cast<int64_t>(self_q_w.size());
        TORCH_CHECK(nl > 0, "internvla_nextdit_set_weights: empty layers");
        TORCH_CHECK(num_heads == 6,
                    "internvla_nextdit first slice requires 6 heads, got ", num_heads);
        TORCH_CHECK(num_heads * head_dim == hidden_size,
                    "num_heads*head_dim must equal hidden_size");
        TORCH_CHECK(hidden_size == 384 && head_dim == 64
                        && (ff_intermediate == 1024 || ff_intermediate == 1536),
                    "first slice targets hidden=384/head_dim=64/ff in {1024,1536}");

        auto check = [&](const at::TensorList& xs, const char* name, int rank) {
            TORCH_CHECK(static_cast<int64_t>(xs.size()) == nl,
                        name, " list size ", xs.size(), " != ", nl);
            for (int64_t i = 0; i < nl; ++i) {
                TORCH_CHECK(xs[i].defined() && xs[i].dim() == rank
                            && xs[i].device().type() == at::kPrivateUse1
                            && xs[i].scalar_type() == at::kHalf
                            && xs[i].is_contiguous(),
                            name, "[", i, "] must be contiguous fp16 RPU rank ", rank);
            }
        };
        check(self_k_w, "self_k_w", 2); check(self_v_w, "self_v_w", 2);
        check(cross_q_w, "cross_q_w", 2); check(cross_k_w, "cross_k_w", 2);
        check(cross_v_w, "cross_v_w", 2); check(out_w, "out_w", 2);
        check(ff1_w, "ff1_w", 2); check(ff2_w, "ff2_w", 2); check(ff3_w, "ff3_w", 2);
        check(norm1_w, "norm1_w", 1); check(norm2_w, "norm2_w", 1);
        check(ffn_norm1_w, "ffn_norm1_w", 1); check(ffn_norm2_w, "ffn_norm2_w", 1);
        check(self_qn_w, "self_qn_w", 1); check(self_qn_b, "self_qn_b", 1);
        check(self_kn_w, "self_kn_w", 1); check(self_kn_b, "self_kn_b", 1);
        check(cross_qn_w, "cross_qn_w", 1); check(cross_qn_b, "cross_qn_b", 1);
        check(cross_kn_w, "cross_kn_w", 1); check(cross_kn_b, "cross_kn_b", 1);
        check(context_norm_w, "context_norm_w", 1);
        check(cross_gate_heads, "cross_gate_heads", 2);
        check(self_q_w, "self_q_w", 2);
        auto check_final = [hidden_size](const at::Tensor& tensor, const char* name,
                                         int rank, int64_t numel) {
            TORCH_CHECK(tensor.defined() && tensor.dim() == rank
                        && tensor.numel() == numel
                        && tensor.device().type() == at::kPrivateUse1
                        && tensor.scalar_type() == at::kHalf
                        && tensor.is_contiguous(),
                        name, " must be contiguous fp16 RPU with ", numel,
                        " elements and rank ", rank);
        };
        check_final(final_out_w, "final_out_w", 2, hidden_size * hidden_size);
        check_final(final_out_b, "final_out_b", 1, hidden_size);
        check_final(final_ln_w, "final_ln_w", 1, hidden_size);
        check_final(final_ln_b, "final_ln_b", 1, hidden_size);
        check_final(action_enc_w, "action_enc_w", 2,
                    hidden_size * NEXTDIT_ACTION_PAD);
        check_final(action_enc_b, "action_enc_b", 1, hidden_size);
        check_final(action_position, "action_position", 2, 32 * hidden_size);
        check_final(action_dec_w, "action_dec_w", 2,
                    NEXTDIT_ACTION_PAD * hidden_size);
        check_final(action_dec_b, "action_dec_b", 1, NEXTDIT_ACTION_PAD);

        set_model_params(num_heads, num_heads, head_dim, hidden_size, ff_intermediate);
        set_num_layers(nl);
        eps_ = eps;
        final_norm_eps_ = final_norm_eps;
        final_out_w_ = final_out_w;
        final_out_b_ = final_out_b;
        final_ln_w_ = final_ln_w;
        final_ln_b_ = final_ln_b;
        action_enc_w_ = action_enc_w;
        action_enc_b_ = action_enc_b;
        action_position_ = action_position;
        action_dec_w_ = action_dec_w;
        action_dec_b_ = action_dec_b;
        layers_.clear();
        layers_.reserve(nl);
        for (int64_t i = 0; i < nl; ++i) {
            layers_.push_back({
                self_q_w[i], self_k_w[i], self_v_w[i],
                cross_q_w[i], cross_k_w[i], cross_v_w[i], out_w[i],
                ff1_w[i], ff2_w[i], ff3_w[i],
                norm1_w[i], norm2_w[i], ffn_norm1_w[i], ffn_norm2_w[i],
                self_qn_w[i], self_qn_b[i], self_kn_w[i], self_kn_b[i],
                cross_qn_w[i], cross_qn_b[i], cross_kn_w[i], cross_kn_b[i],
                context_norm_w[i], cross_gate_heads[i],
            });
        }
        invalidate_model_state();
    }

    at::Tensor forward(
        const at::Tensor& x,
        const at::Tensor& modulation,
        const at::Tensor& final_scale,
        const at::Tensor& raw_actions,
        bool use_action_encoder,
        bool populate_cross_cache,
        bool denoise_unroll,
        bool prepare_modulation_on_rpu,
        at::ArrayRef<double> euler_steps,
        const at::Tensor& cross_context,
        at::TensorList self_k_caches,
        at::TensorList self_v_caches,
        at::TensorList cross_k_caches,
        at::TensorList cross_v_caches,
        const at::Tensor& layer_outputs,
        bool collect_layer_outputs)
    {
        RECORD_FUNCTION("internvla_nextdit_forward", {});
        const int64_t nl = num_layers();
        const int64_t h = hidden_size();
        TORCH_CHECK(x.dim() == 3 && x.size(0) == 1 && x.size(2) == h
                    && x.scalar_type() == at::kHalf && x.is_contiguous()
                    && x.device().type() == at::kPrivateUse1,
                    "x must be [1,T,H] contiguous fp16 RPU");
        const int64_t seq = x.size(1);
        const int64_t denoise_steps = denoise_unroll
            ? static_cast<int64_t>(euler_steps.size()) : 1;
        TORCH_CHECK(!denoise_unroll || use_action_encoder,
                    "denoise unroll requires the fused action encoder/decoder");
        TORCH_CHECK(!denoise_unroll || (denoise_steps > 0 && denoise_steps <= 10),
                    "denoise unroll steps must be in [1,10], got ", denoise_steps);
        TORCH_CHECK(!prepare_modulation_on_rpu || denoise_unroll,
                    "RPU step preparation requires denoise unroll");
        TORCH_CHECK(!prepare_modulation_on_rpu || have_step_prepare_weights_,
                    "RPU step preparation requested before installing its weights");
        const bool modulation_shape_ok = prepare_modulation_on_rpu
            ? modulation.sizes() == at::IntArrayRef({denoise_steps, h})
            : (denoise_unroll
                ? modulation.sizes() == at::IntArrayRef({denoise_steps, nl, 4, h})
                : modulation.sizes() == at::IntArrayRef({nl, 4, h}));
        TORCH_CHECK(modulation_shape_ok
                    && modulation.scalar_type() == at::kHalf
                    && modulation.device().type() == at::kPrivateUse1
                    && modulation.is_contiguous(),
                    "modulation must be [layers,4,H], [steps,layers,4,H], or "
                    "activated [steps,H] for RPU preparation; contiguous fp16 RPU");
        const bool final_scale_shape_ok = prepare_modulation_on_rpu || (denoise_unroll
            ? final_scale.sizes() == at::IntArrayRef({denoise_steps, h})
            : final_scale.sizes() == at::IntArrayRef({h}));
        TORCH_CHECK(final_scale_shape_ok
                    && final_scale.scalar_type() == at::kHalf
                    && final_scale.device().type() == at::kPrivateUse1
                    && final_scale.is_contiguous(),
                    "final_scale must be [H], or [steps,H] for unroll, "
                    "contiguous fp16 RPU");
        TORCH_CHECK(raw_actions.dim() == 3,
                    "raw_actions must be rank-3 [B,32,16]");
        const int64_t action_batch = use_action_encoder ? raw_actions.size(0) : 1;
        TORCH_CHECK(action_batch > 0 && action_batch <= NEXTDIT_MAX_ACTION_BATCH,
                    "RPU action graph batch must be in [1,4], got ", action_batch);
        const bool packed_action_batch = use_action_encoder && action_batch > 1
            && seq == action_batch * NEXTDIT_ACTION_SEQ;
        TORCH_CHECK(seq == NEXTDIT_ACTION_SEQ || packed_action_batch,
                    "x sequence must be 32, or B*32 for a packed action batch; got ",
                    seq, " with B=", action_batch);
        TORCH_CHECK(raw_actions.size(0) == action_batch
                    && raw_actions.size(1) == NEXTDIT_ACTION_SEQ
                    && raw_actions.size(2) == NEXTDIT_ACTION_PAD
                    && raw_actions.scalar_type() == at::kHalf
                    && raw_actions.device().type() == at::kPrivateUse1
                    && raw_actions.is_contiguous(),
                    "raw_actions must be [B,T,16] contiguous fp16 RPU");
        TORCH_CHECK(static_cast<int64_t>(self_k_caches.size()) == nl
                    && static_cast<int64_t>(self_v_caches.size()) == nl
                    && static_cast<int64_t>(cross_k_caches.size()) == nl
                    && static_cast<int64_t>(cross_v_caches.size()) == nl,
                    "all cache lists must equal num_layers");
        const int64_t required_probe_slots = collect_layer_outputs ? nl + 3 : 1;
        TORCH_CHECK(layer_outputs.dim() == 3
                    && layer_outputs.size(0) >= required_probe_slots
                    && layer_outputs.size(1) == seq && layer_outputs.size(2) == h
                    && layer_outputs.scalar_type() == at::kHalf
                    && layer_outputs.device().type() == at::kPrivateUse1
                    && layer_outputs.is_contiguous(),
                    "layer_outputs has invalid shape/dtype/device");

        if (collect_layer_outputs_ != collect_layer_outputs) {
            collect_layer_outputs_ = collect_layer_outputs;
            invalidate_model_state();
        }
        if (use_action_encoder_ != use_action_encoder) {
            use_action_encoder_ = use_action_encoder;
            invalidate_model_state();
        }
        if (action_batch_size_ != action_batch) {
            action_batch_size_ = action_batch;
            invalidate_model_state();
        }
        if (packed_action_batch_ != packed_action_batch) {
            packed_action_batch_ = packed_action_batch;
            invalidate_model_state();
        }
        std::vector<c10::Half> next_euler_steps;
        next_euler_steps.reserve(euler_steps.size());
        for (double step : euler_steps)
            next_euler_steps.emplace_back(static_cast<float>(step));
        if (denoise_unroll_ != denoise_unroll
            || denoise_steps_ != denoise_steps
            || euler_steps_ != next_euler_steps) {
            denoise_unroll_ = denoise_unroll;
            denoise_steps_ = denoise_steps;
            euler_steps_ = std::move(next_euler_steps);
            invalidate_model_state();
        }
        if (prepare_modulation_on_rpu_ != prepare_modulation_on_rpu) {
            prepare_modulation_on_rpu_ = prepare_modulation_on_rpu;
            invalidate_model_state();
        }
        populate_cross_cache_ = populate_cross_cache;

        TORCH_CHECK(cross_context.dim() == 3 && cross_context.size(0) == 1
                    && cross_context.size(2) == h
                    && cross_context.scalar_type() == at::kHalf
                    && cross_context.device().type() == at::kPrivateUse1
                    && cross_context.is_contiguous(),
                    "cross_context must be [1,S,H] contiguous fp16 RPU");
        const int64_t cross_seq = cross_context.size(1);
        TORCH_CHECK(cross_seq > 0 && cross_seq <= NEXTDIT_MAX_CROSS_SEQ,
                    "cross sequence must be in [1,320], got ", cross_seq);

        if (cross_seq_ != cross_seq) {
            cross_seq_ = cross_seq;
            invalidate_model_state();
        }
        set_chunk_size_override(seq);

        if (use_action_encoder_) {
            if (!action_output_.defined()
                || action_output_.sizes()
                    != at::IntArrayRef({NEXTDIT_MAX_ACTION_BATCH,
                                        NEXTDIT_ACTION_SEQ, NEXTDIT_ACTION_PAD})) {
                action_output_ = at::empty(
                    {NEXTDIT_MAX_ACTION_BATCH, NEXTDIT_ACTION_SEQ,
                     NEXTDIT_ACTION_PAD}, x.options());
                invalidate_model_state();
            }
            if (!action_stage_.defined()
                || action_stage_.sizes()
                    != at::IntArrayRef({NEXTDIT_MAX_ACTION_BATCH
                                            * NEXTDIT_ACTION_SEQ,
                                        h})) {
                action_stage_ = at::empty(
                    {NEXTDIT_MAX_ACTION_BATCH * NEXTDIT_ACTION_SEQ, h}, x.options());
                invalidate_model_state();
            }
        }

        modulation_ref_ = modulation;
        if (prepare_modulation_on_rpu_) {
            if (!modulation_stage_.defined()) {
                modulation_stage_ = at::empty(
                    {10, num_layers() * 4 + 1, h}, x.options());
                invalidate_model_state();
            }
            step_input_src_base_ = ::rhino_lkn::RpuGetDevAddr(modulation.data_ptr());
            rpu_ddr_flush_force(modulation.data_ptr<c10::Half>());
            modulation_src_base_ = ::rhino_lkn::RpuGetDevAddr(
                modulation_stage_.data_ptr());
            final_scale_src_base_ = modulation_src_base_;
        } else {
            modulation_src_base_ = ::rhino_lkn::RpuGetDevAddr(modulation.data_ptr());
            rpu_ddr_flush_force(modulation.data_ptr<c10::Half>());
            final_scale_ref_ = final_scale;
            final_scale_src_base_ = ::rhino_lkn::RpuGetDevAddr(final_scale.data_ptr());
            rpu_ddr_flush_force(final_scale.data_ptr<c10::Half>());
        }
        raw_actions_ref_ = raw_actions;
        raw_actions_src_base_ = ::rhino_lkn::RpuGetDevAddr(raw_actions.data_ptr());
        rpu_ddr_flush_force(raw_actions.data_ptr<c10::Half>());

        if (populate_cross_cache_) {
            cross_context_ref_ = cross_context;
            cross_context_src_base_ = ::rhino_lkn::RpuGetDevAddr(
                cross_context_ref_.data_ptr());
            rpu_ddr_flush_force(cross_context_ref_.data_ptr<c10::Half>());
        }
        cross_k_caches_.assign(cross_k_caches.begin(), cross_k_caches.end());
        cross_v_caches_.assign(cross_v_caches.begin(), cross_v_caches.end());
        layer_outputs_ref_ = layer_outputs;

        std::vector<at::Tensor> sk(self_k_caches.begin(), self_k_caches.end());
        std::vector<at::Tensor> sv(self_v_caches.begin(), self_v_caches.end());
        at::Tensor hidden = run_all_layers(x, sk, sv, std::nullopt, 0, false);
        return use_action_encoder_
            ? action_output_.narrow(0, 0, action_batch_size_)
            : hidden;
    }

protected:
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers = num_layers();
        cfg.cross_layer_batch_size = num_layers();
        cfg.body_iterations = denoise_unroll_
            ? denoise_steps_
            : (use_action_encoder_ && !packed_action_batch_ ? action_batch_size_ : 1);
        // All per-forward inputs are refreshed through mutable DMA bases in
        // forward(); outputs/caches use stable storage.  Warm REPLAY can launch
        // the already-built graph without re-walking its host op stream.
        cfg.fast_replay_skip_layer_loop = true;
        cfg.fast_replay_skip_preload = true;
        // Norms, gates, final biases/position and zero_full are read-only after
        // their first population. Subsequent graph signatures share the same
        // live model/SPM generation and do not need to bake those DMAs again.
        cfg.clean_recording_skip_preload_callbacks = true;
        return cfg;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan&) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
        return cfg;
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        const int64_t seq = ctx.chunk_size;
        const int64_t h = hidden_size();
        const int64_t hd = head_dim();
        const int64_t ff = intermediate_size();
        const int64_t tp = attn_tp();
        const int64_t local_ff = ff / NEXTDIT_DEVICE_CORES;
        auto A = [](int64_t bytes) { return Align(bytes, 256); };
        const int64_t full = A(seq * h * NEXTDIT_DWIDTH);
        const int64_t work_full = A(NEXTDIT_MAX_CROSS_SEQ * h * NEXTDIT_DWIDTH);
        const int64_t local_attn = A(
            NEXTDIT_MAX_CROSS_SEQ * hd * NEXTDIT_DWIDTH);
        const int64_t local_query = A(seq * hd * NEXTDIT_DWIDTH);
        const int64_t local_qkq = A(seq * 3 * hd * NEXTDIT_DWIDTH);
        const int64_t gathered_qkq = A(seq * 3 * h * NEXTDIT_DWIDTH);
        const int64_t local_mlp = A(seq * local_ff * NEXTDIT_DWIDTH);
        // The unrolled graph consumes one contiguous [steps,layers,4,H]
        // table.  Keep a full per-step layer slice in SPM so each body
        // iteration needs one broadcast DMA instead of one DMA per layer.
        const int64_t mod = A(num_layers() * 4 * h * NEXTDIT_DWIDTH);
        const int64_t raw_action = A(seq * NEXTDIT_ACTION_PAD * NEXTDIT_DWIDTH);
        const int64_t vec = A(h * NEXTDIT_DWIDTH);
        const int64_t gate = A(hd * NEXTDIT_DWIDTH);
        const int64_t tmp = A(sdpa_compute_tmp_v16_size(make_sdpa_config(), seq) * 32);
        constexpr BufferScope ALL = BufferScope::LayerWide;

        std::vector<BufferDecl> out = {
            {"residual1", full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"residual2", full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"norm_hidden", full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"attn_full", full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"attn_norm", full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"ffn_in", full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"ff_full", full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"ff_norm", full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"full_projected", work_full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"full_qknorm", work_full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"layout_headmajor", work_full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"q", local_query, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"k", local_attn, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"cross_q", local_query, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"qkq_local", local_qkq, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"qkq_gathered", gathered_qkq, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"v", local_attn, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"self_out", local_query, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"cross_out", local_query, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"mixed", local_query, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"ff_gate", local_mlp, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"ff_up", local_mlp, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"ff_partial", full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_tmp", tmp, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"mod", mod, 0, 0, StorageClass::Persistent, 0, nullptr, ALL},
            {"final_scale", vec, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"raw_action", raw_action, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"action_state", raw_action, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"cross_context", work_full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"cross_norm_hidden", work_full, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
        };

        if (prepare_modulation_on_rpu_) {
            const int64_t step_rows = 10;
            const int64_t step_out = (num_layers() * 4 + 1) * h;
            const int64_t step_local = step_out / NEXTDIT_DEVICE_CORES;
            out.push_back({"step_input", A(step_rows * h * NEXTDIT_DWIDTH),
                           0, 0, StorageClass::Temp, 0, nullptr, ALL});
            out.push_back({"step_local", A(step_rows * step_local * NEXTDIT_DWIDTH),
                           0, 0, StorageClass::Temp, 0, nullptr, ALL});
            out.push_back({"step_gathered", A(step_rows * step_out * NEXTDIT_DWIDTH),
                           0, 0, StorageClass::Temp, 0, nullptr, ALL});
            out.push_back({"step_full", A(step_rows * step_out * NEXTDIT_DWIDTH),
                           0, 0, StorageClass::Temp, 0, nullptr, ALL});

            BufferDecl step_bias;
            step_bias.name = "step_bias";
            step_bias.size = A(step_local * NEXTDIT_DWIDTH);
            step_bias.storage = StorageClass::Persistent;
            step_bias.scope = ALL;
            step_bias.preload_callback = [this, step_local](
                FusedModelBase&, int, uint32_t addr0) {
                rpu_launch_ddr_scatter_spm_dma(
                    step_prepare_b_.data_ptr<c10::Half>(), step_local,
                    step_local * NEXTDIT_DWIDTH, addr0, NEXTDIT_DEVICE_CORES);
            };
            out.push_back(step_bias);

            BufferDecl fold_weight;
            fold_weight.name = "step_fold_norm";
            fold_weight.size = A(4 * num_layers() * h * NEXTDIT_DWIDTH);
            fold_weight.storage = StorageClass::Persistent;
            fold_weight.scope = ALL;
            fold_weight.preload_callback = [this, h](
                FusedModelBase&, int, uint32_t addr0) {
                rpu_launch_ddr_broadcast_spm_dma(
                    step_fold_norm_w_.data_ptr<c10::Half>(),
                    4 * num_layers() * h, addr0, 1);
            };
            out.push_back(fold_weight);
        }

        // Cores 0..5 overwrite this scratch with attention projection
        // partials; cores 6..7 remain zero inputs for the 8-core two-stage
        // reduce. Persistent storage prevents lifetime aliasing from clobbering
        // those two zero slices between layers or graph replays.
        BufferDecl partial;
        partial.name = "full_partial";
        partial.size = work_full;
        partial.storage = StorageClass::Persistent;
        partial.scope = ALL;
        partial.preload_callback = [h](FusedModelBase&, int, uint32_t addr0) {
            rpu_launch_memset_spm_multicore(
                addr0, NEXTDIT_MAX_CROSS_SEQ * h);
        };
        out.push_back(partial);

        BufferDecl zero;
        zero.name = "zero_full";
        zero.size = work_full;
        zero.storage = StorageClass::Persistent;
        zero.scope = ALL;
        zero.preload_callback = [h](FusedModelBase&, int, uint32_t addr0) {
            rpu_launch_memset_spm_multicore(
                addr0, NEXTDIT_MAX_CROSS_SEQ * h);
        };
        out.push_back(zero);

        auto add_final = [&](const char* name, at::Tensor InternVLANextDiTModel::*field,
                             int cores) {
            BufferDecl d;
            d.name = name; d.size = vec; d.storage = StorageClass::Persistent; d.scope = ALL;
            d.preload_callback = [this, field, cores](FusedModelBase&, int, uint32_t addr0) {
                rpu_launch_ddr_broadcast_spm_dma(
                    (this->*field).data_ptr<c10::Half>(), hidden_size(), addr0, cores);
            };
            out.push_back(d);
        };
        add_final("final_ln_w", &InternVLANextDiTModel::final_ln_w_, NEXTDIT_DEVICE_CORES);
        add_final("final_ln_b", &InternVLANextDiTModel::final_ln_b_, NEXTDIT_DEVICE_CORES);
        add_final("final_out_b", &InternVLANextDiTModel::final_out_b_, 1);
        add_final("action_enc_b", &InternVLANextDiTModel::action_enc_b_, 1);

        BufferDecl action_dec_bias;
        action_dec_bias.name = "action_dec_b";
        action_dec_bias.size = A(NEXTDIT_ACTION_PAD * NEXTDIT_DWIDTH);
        action_dec_bias.storage = StorageClass::Persistent;
        action_dec_bias.scope = ALL;
        action_dec_bias.preload_callback = [this](FusedModelBase&, int, uint32_t addr0) {
            rpu_launch_ddr_broadcast_spm_dma(
                action_dec_b_.data_ptr<c10::Half>(), NEXTDIT_ACTION_PAD, addr0, 1);
        };
        out.push_back(action_dec_bias);

        BufferDecl action_pos;
        action_pos.name = "action_position";
        action_pos.size = A(NEXTDIT_ACTION_SEQ * h * NEXTDIT_DWIDTH);
        action_pos.storage = StorageClass::Persistent;
        action_pos.scope = ALL;
        action_pos.preload_callback = [this, h](FusedModelBase&, int, uint32_t addr0) {
            rpu_launch_ddr_broadcast_spm_dma(
                action_position_.data_ptr<c10::Half>(), NEXTDIT_ACTION_SEQ * h,
                addr0, 1);
        };
        out.push_back(action_pos);

        auto add_replicated = [&](const char* name, at::Tensor LayerWeights::*field, int cores) {
            BufferDecl d;
            d.name = name; d.size = vec; d.storage = StorageClass::PersistentPerLayer;
            d.per_layer = static_cast<int>(num_layers()); d.scope = ALL;
            d.preload_callback = [this, field, cores](FusedModelBase&, int layer, uint32_t addr0) {
                rpu_launch_ddr_broadcast_spm_dma(
                    (layers_[layer].*field).data_ptr<c10::Half>(), hidden_size(), addr0, cores);
            };
            out.push_back(d);
        };
        add_replicated("norm1_w", &LayerWeights::norm1_w, NEXTDIT_DEVICE_CORES);
        add_replicated("norm2_w", &LayerWeights::norm2_w, NEXTDIT_DEVICE_CORES);
        add_replicated("ffn_norm1_w", &LayerWeights::ffn_norm1_w, NEXTDIT_DEVICE_CORES);
        add_replicated("ffn_norm2_w", &LayerWeights::ffn_norm2_w, NEXTDIT_DEVICE_CORES);
        add_replicated("self_qn_w", &LayerWeights::self_qn_w, NEXTDIT_DEVICE_CORES);
        add_replicated("self_qn_b", &LayerWeights::self_qn_b, NEXTDIT_DEVICE_CORES);
        add_replicated("self_kn_w", &LayerWeights::self_kn_w, NEXTDIT_DEVICE_CORES);
        add_replicated("self_kn_b", &LayerWeights::self_kn_b, NEXTDIT_DEVICE_CORES);
        add_replicated("cross_qn_w", &LayerWeights::cross_qn_w, NEXTDIT_DEVICE_CORES);
        add_replicated("cross_qn_b", &LayerWeights::cross_qn_b, NEXTDIT_DEVICE_CORES);
        add_replicated("cross_kn_w", &LayerWeights::cross_kn_w, NEXTDIT_DEVICE_CORES);
        add_replicated("cross_kn_b", &LayerWeights::cross_kn_b, NEXTDIT_DEVICE_CORES);
        add_replicated("context_norm_w", &LayerWeights::context_norm_w,
                       NEXTDIT_DEVICE_CORES);

        BufferDecl cg;
        cg.name = "cross_gate"; cg.size = gate;
        cg.storage = StorageClass::PersistentPerLayer;
        cg.per_layer = static_cast<int>(num_layers()); cg.scope = ALL;
        cg.preload_callback = [this, hd, tp](FusedModelBase&, int layer, uint32_t addr0) {
            rpu_launch_ddr_scatter_spm_dma(
                layers_[layer].cross_gate_heads.data_ptr<c10::Half>(), hd,
                hd * NEXTDIT_DWIDTH, addr0, tp);
        };
        out.push_back(cg);
        return out;
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layers_[layer_idx];
        const int64_t seq = chunk.len;
        const int64_t h = hidden_size();
        const int64_t hd = head_dim();
        const int64_t tp = attn_tp();
        const int64_t ff = intermediate_size();
        const int64_t local_ff = ff / NEXTDIT_DEVICE_CORES;
        const int64_t full_elems = seq * h;
        const int64_t local_q_elems = seq * hd;

        if (layer_idx == 0 && use_action_encoder_) {
            uint32_t encoder_input = addr(0, "raw_action");
            if (denoise_unroll_) {
                encoder_input = addr(0, "action_state");
                if (ctx().body_iter == 0) {
                    rpu_launch_ddr_broadcast_spm_dma_mutable(
                        &raw_actions_src_base_, 0, seq * NEXTDIT_ACTION_PAD,
                        encoder_input, 1);
                }
            } else {
                const int64_t raw_offset = packed_action_batch_
                    ? 0
                    : ctx().body_iter * seq * NEXTDIT_ACTION_PAD * NEXTDIT_DWIDTH;
                rpu_launch_ddr_broadcast_spm_dma_mutable(
                    &raw_actions_src_base_, raw_offset,
                    seq * NEXTDIT_ACTION_PAD, encoder_input, 1);
            }
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                encoder_input, action_enc_w_, addr(0, "residual1"),
                seq, h, NEXTDIT_ACTION_PAD, /*partition=*/1, /*num_cores=*/1,
                addr(0, "action_enc_b"), false, at::Tensor{});
            if (packed_action_batch_) {
                const uint32_t trajectory_bytes = static_cast<uint32_t>(
                    NEXTDIT_ACTION_SEQ * h * NEXTDIT_DWIDTH);
                for (int64_t batch_idx = 0; batch_idx < action_batch_size_;
                     ++batch_idx) {
                    const uint32_t residual = addr(0, "residual1")
                        + static_cast<uint32_t>(batch_idx) * trajectory_bytes;
                    rpu_launch_eltwise_binary_spm_kernel(
                        residual, addr(0, "action_position"), residual,
                        NEXTDIT_ACTION_SEQ * h, ValuOpType::ADD,
                        c10::Half(1.0), 1);
                }
            } else {
                rpu_launch_eltwise_binary_spm_kernel(
                    addr(0, "residual1"), addr(0, "action_position"),
                    addr(0, "residual1"), seq * h, ValuOpType::ADD,
                    c10::Half(1.0), 1);
            }
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "residual1"), action_stage_.data_ptr<c10::Half>(), seq * h);
            rpu_launch_ddr_broadcast_spm_dma(
                action_stage_.data_ptr<c10::Half>(), seq * h,
                addr(0, "residual1"), NEXTDIT_DEVICE_CORES);
        } else if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        if (layer_idx == 0 && populate_cross_cache_ && ctx().body_iter == 0) {
            const int64_t cross_rows = ((cross_seq_ + 15) / 16) * 16;
            rpu_launch_memset_spm_multicore(
                addr(0, "cross_context"), cross_rows * h);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &cross_context_src_base_, 0, cross_seq_ * h,
                addr(0, "cross_context"), NEXTDIT_DEVICE_CORES);
        }

        if (prepare_modulation_on_rpu_
            && layer_idx == 0 && ctx().body_iter == 0) {
            prepare_modulation_table();
        }

        // [scale_msa(+1), tanh(gate_msa), scale_mlp(+1), tanh(gate_mlp)]
        const uint32_t mod_root = addr(0, "mod");
        const uint32_t mod_base = mod_root
            + static_cast<uint32_t>(denoise_unroll_ ? layer_idx : 0)
                * 4 * h * NEXTDIT_DWIDTH;
        if (denoise_unroll_ && layer_idx == 0) {
            const int64_t step_stride =
                (num_layers() * 4 + (prepare_modulation_on_rpu_ ? 1 : 0))
                * h * NEXTDIT_DWIDTH;
            const int64_t step_offset = ctx().body_iter * step_stride;
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &modulation_src_base_,
                step_offset, num_layers() * 4 * h,
                addr(0, "mod"), NEXTDIT_DEVICE_CORES);
        } else if (!denoise_unroll_ && ctx().body_iter == 0) {
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &modulation_src_base_, layer_idx * 4 * h * NEXTDIT_DWIDTH,
                4 * h, mod_base, NEXTDIT_DEVICE_CORES);
        }
        const uint32_t scale_msa = prepare_modulation_on_rpu_
            ? mod_root + layer_idx * h * NEXTDIT_DWIDTH : mod_base;
        const uint32_t gate_msa = prepare_modulation_on_rpu_
            ? mod_root + (num_layers() + layer_idx) * h * NEXTDIT_DWIDTH
            : scale_msa + h * NEXTDIT_DWIDTH;
        const uint32_t scale_mlp = prepare_modulation_on_rpu_
            ? mod_root + (2 * num_layers() + layer_idx) * h * NEXTDIT_DWIDTH
            : scale_msa + 2 * h * NEXTDIT_DWIDTH;
        const uint32_t gate_mlp = prepare_modulation_on_rpu_
            ? mod_root + (3 * num_layers() + layer_idx) * h * NEXTDIT_DWIDTH
            : scale_msa + 3 * h * NEXTDIT_DWIDTH;
        const bool fold_norm_modulation =
            denoise_unroll_ && nextdit_fold_norm_modulation_enabled();

        // LuminaRMSNormZero: RMSNorm(x) * (1 + scale_msa).
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "norm_hidden"),
            fold_norm_modulation ? scale_msa : layer_addr(layer_idx, 0, "norm1_w"),
            seq, h, eps_);
        if (!fold_norm_modulation) {
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                scale_msa, addr(0, "norm_hidden"), addr(0, "norm_hidden"),
                seq, h, c10::Half(1.0), ValuOpType::MUL, false);
        }
        // Self Q/K and cross-Q share the same normalized input.  The opt-in
        // fused projection performs one [3H,H] GEMM and one gather; each
        // projection still goes through its original full-H LayerNorm and
        // exact head-major scatter.
        if (use_fused_qkq_) {
            project_qkq_exact(
                fused_qkq_weights_[layer_idx], addr(0, "norm_hidden"), seq,
                layer_addr(layer_idx, 0, "self_qn_w"),
                layer_addr(layer_idx, 0, "self_qn_b"),
                layer_addr(layer_idx, 0, "self_kn_w"),
                layer_addr(layer_idx, 0, "self_kn_b"),
                layer_addr(layer_idx, 0, "cross_qn_w"),
                layer_addr(layer_idx, 0, "cross_qn_b"),
                addr(0, "q"), addr(0, "k"), addr(0, "cross_q"),
                layer_idx == 0 && collect_layer_outputs_ && get_debug_export());
        } else {
            project_qk_exact(lw.self_q_w,
                             layer_addr(layer_idx, 0, "self_qn_w"),
                             layer_addr(layer_idx, 0, "self_qn_b"),
                             addr(0, "norm_hidden"), seq, addr(0, "q"),
                             layer_idx == 0 && collect_layer_outputs_ && get_debug_export());
            project_qk_exact(lw.self_k_w,
                             layer_addr(layer_idx, 0, "self_kn_w"),
                             layer_addr(layer_idx, 0, "self_kn_b"),
                             addr(0, "norm_hidden"), seq, addr(0, "k"), false);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.self_v_w, addr(0, "v"),
            seq, h, h, 1, tp);

        auto& sk = (*ctx().k_caches)[layer_idx];
        auto& sv = (*ctx().v_caches)[layer_idx];
        const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
        if (packed_action_batch_) {
            rpu_launch_insert_kcache_spm_unified(
                sk, 0, addr_offset("k").value, seq,
                num_kv_heads(), hd, tp, /*cache_batch_offset_elems=*/0,
                /*allow_non8_v16=*/true);
            rpu_launch_insert_vcache_spm_unified(
                sv, 0, addr_offset("v").value, seq,
                num_kv_heads(), hd, tp, /*cache_batch_offset_elems=*/0,
                /*allow_non8_v16=*/true);
            rpu_launch_sdpa_spm_minibatch_kernel(
                sk, sv, 0, scale,
                addr_offset("q").value,
                addr_offset("self_out").value,
                addr_offset("sdpa_tmp").value, 0,
                seq, num_q_heads(), num_kv_heads(), hd,
                NEXTDIT_ACTION_SEQ, action_batch_size_,
                tp, NEXTDIT_DEVICE_CORES,
                /*tile_m_override=*/16,
                /*use_16b=*/use_16b_sdpa_);
        } else {
            rpu_launch_insert_kcache_spm_unified(
                sk, 0, addr_offset("k").value, seq, num_kv_heads(), hd, tp,
                /*cache_batch_offset_elems=*/0,
                /*allow_non8_v16=*/true);
            rpu_launch_insert_vcache_spm_unified(
                sv, 0, addr_offset("v").value, seq, num_kv_heads(), hd, tp,
                /*cache_batch_offset_elems=*/0,
                /*allow_non8_v16=*/true);
            rpu_launch_sdpa_spm_unified_kernel_v2(
                sk, sv, 0, scale, addr_offset("q").value,
                addr_offset("self_out").value, addr_offset("sdpa_tmp").value, 0,
                seq, num_q_heads(), num_kv_heads(), hd, seq,
                tp, NEXTDIT_DEVICE_CORES,
                /*cache_batch_offset_elems=*/0,
                /*use_16b=*/use_16b_sdpa_);
        }

        // Cross attention: Q is dynamic/exact-normalized. On a condition-cache
        // miss, project the shared caption to K/V on RPU and retain the caches.
        if (!use_fused_qkq_) {
            project_qk_exact(lw.cross_q_w,
                             layer_addr(layer_idx, 0, "cross_qn_w"),
                             layer_addr(layer_idx, 0, "cross_qn_b"),
                             addr(0, "norm_hidden"), seq, addr(0, "q"), false);
        }
        auto& ck = cross_k_caches_[layer_idx];
        auto& cv = cross_v_caches_[layer_idx];
        if (populate_cross_cache_ && ctx().body_iter == 0) {
            populate_cross_layer(layer_idx, lw);
        }
        if (packed_action_batch_ && cross_seq_ < seq) {
            const uint32_t query_bytes = static_cast<uint32_t>(
                NEXTDIT_ACTION_SEQ * hd * NEXTDIT_DWIDTH);
            for (int64_t batch_idx = 0; batch_idx < action_batch_size_;
                 ++batch_idx) {
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    ck, cv, 0, scale,
                    (use_fused_qkq_ ? addr_offset("cross_q").value
                                    : addr_offset("q").value)
                        + static_cast<uint32_t>(batch_idx) * query_bytes,
                    addr_offset("cross_out").value
                        + static_cast<uint32_t>(batch_idx) * query_bytes,
                    addr_offset("sdpa_tmp").value, 0,
                    NEXTDIT_ACTION_SEQ, num_q_heads(), num_kv_heads(), hd,
                    cross_seq_, tp, NEXTDIT_DEVICE_CORES);
            }
        } else {
            rpu_launch_sdpa_spm_unified_kernel_v2(
                ck, cv, 0, scale,
                use_fused_qkq_ ? addr_offset("cross_q").value
                               : addr_offset("q").value,
                addr_offset("cross_out").value, addr_offset("sdpa_tmp").value,
                0, seq, num_q_heads(), num_kv_heads(), hd, cross_seq_,
                tp, NEXTDIT_DEVICE_CORES,
                /*cache_batch_offset_elems=*/0,
                /*use_16b=*/use_16b_sdpa_);
        }

        // Per-head tanh gate is precomputed/expanded by Python to [6,64].
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            layer_addr(layer_idx, 0, "cross_gate"), addr(0, "cross_out"),
            addr(0, "cross_out"), seq, hd, c10::Half(1.0), ValuOpType::MUL, false);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "self_out"), addr(0, "cross_out"), addr(0, "mixed"),
            local_q_elems, ValuOpType::ADD, c10::Half(1.0), tp);

        // Shared output projection then norm2 + tanh(gate_msa) residual.
        // The projection writes partials on the six attention cores. The two
        // idle-core slices stay zero in persistent SPM, allowing the validated
        // 8-core two-stage reduce to consume all eight inputs safely.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "mixed"), lw.out_w, addr(0, "full_partial"),
            seq, h, h, 0, tp);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "full_partial"), addr(0, "zero_full"), addr(0, "attn_full"),
            seq, h, NEXTDIT_DEVICE_CORES, NEXTDIT_DEVICE_CORES);
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "attn_full"), addr(0, "attn_norm"),
            fold_norm_modulation ? gate_msa : layer_addr(layer_idx, 0, "norm2_w"),
            seq, h, eps_);
        if (!fold_norm_modulation) {
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                gate_msa, addr(0, "attn_norm"), addr(0, "attn_norm"),
                seq, h, c10::Half(1.0), ValuOpType::MUL, false);
        }
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "residual1"), addr(0, "attn_norm"), addr(0, "residual2"),
            full_elems, ValuOpType::ADD, c10::Half(1.0), NEXTDIT_DEVICE_CORES);

        // ffn_norm1 * (1+scale_mlp) -> Lumina SwiGLU -> ffn_norm2 -> gated residual.
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual2"), addr(0, "ffn_in"),
            fold_norm_modulation ? scale_mlp
                                 : layer_addr(layer_idx, 0, "ffn_norm1_w"),
            seq, h, eps_);
        if (!fold_norm_modulation) {
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                scale_mlp, addr(0, "ffn_in"), addr(0, "ffn_in"),
                seq, h, c10::Half(1.0), ValuOpType::MUL, false);
        }
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "ffn_in"), lw.ff1_w, addr(0, "ff_gate"),
            seq, ff, h, 1, NEXTDIT_DEVICE_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "ffn_in"), lw.ff3_w, addr(0, "ff_up"),
            seq, ff, h, 1, NEXTDIT_DEVICE_CORES);
        rpu_launch_silu_mul_spm_kernel(
            addr(0, "ff_gate"), addr(0, "ff_up"), addr(0, "ff_gate"),
            seq * local_ff, NEXTDIT_DEVICE_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "ff_gate"), lw.ff2_w, addr(0, "ff_partial"),
            seq, h, ff, 0, NEXTDIT_DEVICE_CORES);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "ff_partial"), addr(0, "zero_full"), addr(0, "ff_full"),
            seq, h, NEXTDIT_DEVICE_CORES, NEXTDIT_DEVICE_CORES);
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "ff_full"), addr(0, "ff_norm"),
            fold_norm_modulation ? gate_mlp
                                 : layer_addr(layer_idx, 0, "ffn_norm2_w"),
            seq, h, eps_);
        if (!fold_norm_modulation) {
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                gate_mlp, addr(0, "ff_norm"), addr(0, "ff_norm"),
                seq, h, c10::Half(1.0), ValuOpType::MUL, false);
        }
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "residual2"), addr(0, "ff_norm"), addr(0, "residual1"),
            full_elems, ValuOpType::ADD, c10::Half(1.0), NEXTDIT_DEVICE_CORES);

        // Correctness-slice probe: one full hidden snapshot per block.  The
        // Python runtime owns a stable RPU buffer and clones it to CPU before
        // the next replay, so this does not violate the stable-DDR rule.
        if (collect_layer_outputs_) {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "residual1"),
                layer_outputs_ref_.data_ptr<c10::Half>() + layer_idx * full_elems,
                full_elems);
        }

        if (layer_idx == num_layers() - 1) {
            if (denoise_unroll_ || ctx().body_iter == 0) {
                const int64_t scale_offset = prepare_modulation_on_rpu_
                    ? (ctx().body_iter * (num_layers() * 4 + 1)
                       + num_layers() * 4) * h * NEXTDIT_DWIDTH
                    : (denoise_unroll_
                        ? ctx().body_iter * h * NEXTDIT_DWIDTH : 0);
                rpu_launch_ddr_broadcast_spm_dma_mutable(
                    &final_scale_src_base_, scale_offset, h, addr(0, "final_scale"),
                    NEXTDIT_DEVICE_CORES);
            }
            rpu_launch_layernorm_spm_kernel(
                addr(0, "residual1"), addr(0, "norm_hidden"),
                addr(0, "final_ln_w"), addr(0, "final_ln_b"),
                seq, h, final_norm_eps_, false, 0, NEXTDIT_DEVICE_CORES);
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                addr(0, "final_scale"), addr(0, "norm_hidden"),
                addr(0, "norm_hidden"), seq, h, c10::Half(1.0),
                ValuOpType::MUL, false);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "norm_hidden"), final_out_w_, addr(0, "residual1"),
                seq, h, h, /*partition=*/1, /*num_cores=*/1,
                addr(0, "final_out_b"), false, at::Tensor{});
            if (use_action_encoder_) {
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    addr(0, "residual1"), action_dec_w_, addr(0, "raw_action"),
                    seq, NEXTDIT_ACTION_PAD, h, /*partition=*/1, /*num_cores=*/1,
                    addr(0, "action_dec_b"), false, at::Tensor{});
                if (denoise_unroll_) {
                    rpu_launch_eltwise_binary_spm_kernel(
                        addr(0, "action_state"), addr(0, "raw_action"),
                        addr(0, "action_state"), seq * NEXTDIT_ACTION_PAD,
                        ValuOpType::ADD, euler_steps_.at(ctx().body_iter), 1);
                    if (ctx().body_iter == denoise_steps_ - 1) {
                        rpu_launch_spm_copy_ddr_dma(
                            addr(0, "action_state"),
                            action_output_.data_ptr<c10::Half>(),
                            seq * NEXTDIT_ACTION_PAD);
                    }
                } else {
                    rpu_launch_spm_copy_ddr_dma(
                        addr(0, "raw_action"),
                        action_output_.data_ptr<c10::Half>()
                            + ctx().body_iter * seq * NEXTDIT_ACTION_PAD,
                        seq * NEXTDIT_ACTION_PAD);
                }
            }
        }

        if (!ctx().output_to_spm && !use_action_encoder_)
            emit_layer_output_dma(layer_idx, chunk);
    }

private:
    SdpaConfig make_sdpa_config() const {
        return {SdpaKernelType::FLASH_ATTN_SPM, head_dim(), num_q_heads(),
                num_kv_heads(), attn_tp(), 0};
    }

    void prepare_modulation_table() {
        const int64_t h = hidden_size();
        const int64_t nl = num_layers();
        const int64_t steps = denoise_steps_;
        const int64_t plane = nl * h;
        const int64_t out_features = (4 * nl + 1) * h;
        const int64_t local_out = out_features / NEXTDIT_DEVICE_CORES;
        TORCH_CHECK(steps > 0 && steps <= 10,
                    "RPU step preparation requires 1..10 steps");
        TORCH_CHECK(out_features % NEXTDIT_DEVICE_CORES == 0,
                    "step-prepare output must divide eight cores");

        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &step_input_src_base_, 0, steps * h,
            addr(0, "step_input"), NEXTDIT_DEVICE_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "step_input"), step_prepare_w_, addr(0, "step_local"),
            steps, out_features, h, /*partition=*/1, NEXTDIT_DEVICE_CORES,
            addr(0, "step_bias"));
        rpu_launch_spm_gather_spm_dma(
            addr(0, "step_local"), steps * local_out,
            addr(0, "step_gathered"), NEXTDIT_DEVICE_CORES);
        rpu_launch_transpose_bnc_to_nbc_spm(
            addr_offset("step_gathered").value,
            addr_offset("step_full").value,
            NEXTDIT_DEVICE_CORES, static_cast<int>(steps),
            static_cast<int>(local_out));

        const uint32_t full = addr(0, "step_full");
        const uint32_t fold = addr(0, "step_fold_norm");
        for (int64_t step = 0; step < steps; ++step) {
            const uint32_t row = full + static_cast<uint32_t>(
                step * out_features * NEXTDIT_DWIDTH);
            const uint32_t scale_msa = row;
            const uint32_t gate_msa = row + plane * NEXTDIT_DWIDTH;
            const uint32_t scale_mlp = row + 2 * plane * NEXTDIT_DWIDTH;
            const uint32_t gate_mlp = row + 3 * plane * NEXTDIT_DWIDTH;
            const uint32_t final_scale = row + 4 * plane * NEXTDIT_DWIDTH;

            rpu_launch_eltwise_binary_scalar_spm_kernel(
                scale_msa, c10::Half(1.0), scale_msa, plane, ValuOpType::ADD);
            rpu_launch_eltwise_unary_spm_kernel(
                gate_msa, gate_msa, plane, ValuOpType::TANH,
                /*is_gelu=*/false, /*num_cores=*/1);
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                scale_mlp, c10::Half(1.0), scale_mlp, plane, ValuOpType::ADD);
            rpu_launch_eltwise_unary_spm_kernel(
                gate_mlp, gate_mlp, plane, ValuOpType::TANH,
                /*is_gelu=*/false, /*num_cores=*/1);
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                final_scale, c10::Half(1.0), final_scale, h, ValuOpType::ADD);

            if (nextdit_fold_norm_modulation_enabled()) {
                for (int64_t channel = 0; channel < 4; ++channel) {
                    rpu_launch_eltwise_binary_spm_kernel(
                        row + channel * plane * NEXTDIT_DWIDTH,
                        fold + channel * plane * NEXTDIT_DWIDTH,
                        row + channel * plane * NEXTDIT_DWIDTH,
                        plane, ValuOpType::MUL, c10::Half(1.0), 1);
                }
            }
            rpu_launch_spm_copy_ddr_dma(
                row,
                modulation_stage_.data_ptr<c10::Half>()
                    + step * out_features,
                out_features);
        }
    }

    void populate_cross_layer(int layer_idx, const LayerWeights& lw) {
        const int64_t h = hidden_size();
        const int64_t hd = head_dim();
        const int64_t tp = attn_tp();
        const int64_t cross_rows = ((cross_seq_ + 15) / 16) * 16;
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "cross_context"), addr(0, "cross_norm_hidden"),
            layer_addr(layer_idx, 0, "context_norm_w"),
            cross_rows, h, eps_);
        project_qk_exact(
            lw.cross_k_w,
            layer_addr(layer_idx, 0, "cross_kn_w"),
            layer_addr(layer_idx, 0, "cross_kn_b"),
            addr(0, "cross_norm_hidden"), cross_rows, addr(0, "k"), false);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "cross_norm_hidden"), lw.cross_v_w, addr(0, "v"),
            cross_rows, h, h, 1, tp);
        rpu_launch_insert_kcache_spm_unified(
            cross_k_caches_[layer_idx], 0, addr_offset("k").value, cross_seq_,
            num_kv_heads(), hd, tp,
            /*cache_batch_offset_elems=*/0,
            /*allow_non8_v16=*/true, /*allow_hybrid_v16=*/true);
        rpu_launch_insert_vcache_spm_unified(
            cross_v_caches_[layer_idx], 0, addr_offset("v").value, cross_seq_,
            num_kv_heads(), hd, tp,
            /*cache_batch_offset_elems=*/0,
            /*allow_non8_v16=*/true, /*allow_hybrid_v16=*/true);
    }

    // Head-partitioned projection produces one [seq,hd] slice on each of the
    // six attention cores. Gather the slices to core 0, transpose to row-major
    // [seq,H] for the exact full-hidden LayerNorm, then scatter head-major
    // slices back to the attention cores.
    void project_qk_exact(const at::Tensor& head_swizzled_weight,
                          uint32_t gamma, uint32_t beta, uint32_t input_full,
                          int64_t seq, uint32_t dst_local, bool capture_debug)
    {
        const int64_t h = hidden_size();
        const int64_t nh = num_q_heads();
        const int64_t hd = head_dim();
        const int64_t tp = attn_tp();
        TORCH_CHECK(seq % 16 == 0, "Q/K layout transpose requires seq%16, got ", seq);

        rpu_launch_linear_spm_to_spm_acc16_kernel(
            input_full, head_swizzled_weight, dst_local,
            seq, h, h, 1, tp);
        rpu_launch_spm_gather_spm_dma(
            dst_local, seq * hd, addr(0, "layout_headmajor"), tp);
        rpu_launch_transpose_bnc_to_nbc_spm(
            addr_offset("layout_headmajor").value,
            addr_offset("full_projected").value,
            static_cast<int>(nh), static_cast<int>(seq), static_cast<int>(hd));
        if (capture_debug) {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "full_projected"),
                layer_outputs_ref_.data_ptr<c10::Half>()
                    + num_layers() * seq * h,
                seq * h);
        }
        rpu_launch_layernorm_spm_kernel(
            addr(0, "full_projected"), addr(0, "full_qknorm"),
            gamma, beta, seq, h, 1e-5, false, 0, 1);
        if (capture_debug) {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "full_qknorm"),
                layer_outputs_ref_.data_ptr<c10::Half>()
                    + (num_layers() + 1) * seq * h,
                seq * h);
        }

        // [seq,heads,hd] -> [heads,seq,hd] in one exact data-movement pass.
        rpu_launch_transpose_bnc_to_nbc_spm(
            addr_offset("full_qknorm").value,
            addr_offset("layout_headmajor").value,
            static_cast<int>(seq), static_cast<int>(nh), static_cast<int>(hd));
        if (capture_debug) {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "layout_headmajor"),
                layer_outputs_ref_.data_ptr<c10::Half>()
                    + (num_layers() + 2) * seq * h,
                seq * h);
        }
        rpu_launch_spm_scatter_spm_dma(
            addr(0, "layout_headmajor"), seq * hd,
            seq * hd * NEXTDIT_DWIDTH, dst_local, tp);
    }

    void project_qkq_exact(
        const at::Tensor& fused_weight, uint32_t input_full, int64_t seq,
        uint32_t q_gamma, uint32_t q_beta,
        uint32_t k_gamma, uint32_t k_beta,
        uint32_t cross_q_gamma, uint32_t cross_q_beta,
        uint32_t q_dst, uint32_t k_dst, uint32_t cross_q_dst,
        bool capture_debug)
    {
        const int64_t h = hidden_size();
        const int64_t nh = num_q_heads();
        const int64_t hd = head_dim();
        const int64_t tp = attn_tp();
        const int64_t cores_per_projection = tp / 3;
        TORCH_CHECK(tp == 6 && cores_per_projection == 2,
                    "fused Q/K/cross-Q requires six attention cores");

        rpu_launch_linear_spm_to_spm_acc16_kernel(
            input_full, fused_weight, addr(0, "qkq_local"),
            seq, 3 * h, h, /*partition=*/1, tp);
        rpu_launch_spm_gather_spm_dma(
            addr(0, "qkq_local"), seq * 3 * hd,
            addr(0, "qkq_gathered"), tp);

        const uint32_t gathered_off = addr_offset("qkq_gathered").value;
        const uint32_t projection_stride = static_cast<uint32_t>(
            cores_per_projection * seq * 3 * hd * NEXTDIT_DWIDTH);
        const uint32_t gammas[3] = {q_gamma, k_gamma, cross_q_gamma};
        const uint32_t betas[3] = {q_beta, k_beta, cross_q_beta};
        const uint32_t destinations[3] = {q_dst, k_dst, cross_q_dst};

        for (int projection = 0; projection < 3; ++projection) {
            rpu_launch_transpose_bnc_to_nbc_spm(
                gathered_off + projection * projection_stride,
                addr_offset("full_projected").value,
                static_cast<int>(cores_per_projection),
                static_cast<int>(seq), static_cast<int>(3 * hd));
            if (projection == 0 && capture_debug) {
                rpu_launch_spm_copy_ddr_dma(
                    addr(0, "full_projected"),
                    layer_outputs_ref_.data_ptr<c10::Half>()
                        + num_layers() * seq * h,
                    seq * h);
            }
            rpu_launch_layernorm_spm_kernel(
                addr(0, "full_projected"), addr(0, "full_qknorm"),
                gammas[projection], betas[projection], seq, h,
                1e-5, false, 0, 1);
            if (projection == 0 && capture_debug) {
                rpu_launch_spm_copy_ddr_dma(
                    addr(0, "full_qknorm"),
                    layer_outputs_ref_.data_ptr<c10::Half>()
                        + (num_layers() + 1) * seq * h,
                    seq * h);
            }
            rpu_launch_transpose_bnc_to_nbc_spm(
                addr_offset("full_qknorm").value,
                addr_offset("layout_headmajor").value,
                static_cast<int>(seq), static_cast<int>(nh),
                static_cast<int>(hd));
            if (projection == 0 && capture_debug) {
                rpu_launch_spm_copy_ddr_dma(
                    addr(0, "layout_headmajor"),
                    layer_outputs_ref_.data_ptr<c10::Half>()
                        + (num_layers() + 2) * seq * h,
                    seq * h);
            }
            rpu_launch_spm_scatter_spm_dma(
                addr(0, "layout_headmajor"), seq * hd,
                seq * hd * NEXTDIT_DWIDTH, destinations[projection], tp);
        }
    }

    std::vector<LayerWeights> layers_;
    double eps_ = 1e-5;
    double final_norm_eps_ = 1e-6;
    int64_t cross_seq_ = 0;

    at::Tensor final_out_w_, final_out_b_, final_ln_w_, final_ln_b_;
    at::Tensor action_enc_w_, action_enc_b_, action_position_;
    at::Tensor action_dec_w_, action_dec_b_;
    at::Tensor step_prepare_w_, step_prepare_b_, step_fold_norm_w_;
    bool have_step_prepare_weights_ = false;

    at::Tensor modulation_ref_;
    uint64_t modulation_src_base_ = 0;
    uint64_t step_input_src_base_ = 0;
    at::Tensor final_scale_ref_;
    uint64_t final_scale_src_base_ = 0;
    at::Tensor raw_actions_ref_;
    uint64_t raw_actions_src_base_ = 0;
    at::Tensor cross_context_ref_;
    uint64_t cross_context_src_base_ = 0;
    at::Tensor action_stage_;
    at::Tensor action_output_;
    at::Tensor modulation_stage_;
    bool use_action_encoder_ = false;
    int64_t action_batch_size_ = 1;
    bool packed_action_batch_ = false;
    bool denoise_unroll_ = false;
    bool prepare_modulation_on_rpu_ = false;
    int64_t denoise_steps_ = 1;
    std::vector<c10::Half> euler_steps_;
    bool populate_cross_cache_ = true;
    at::Tensor layer_outputs_ref_;
    bool collect_layer_outputs_ = false;
    bool use_16b_sdpa_ = false;
    std::vector<at::Tensor> cross_k_caches_, cross_v_caches_;
    std::vector<at::Tensor> fused_qkq_weights_;
    bool use_fused_qkq_ = false;
};

}  // namespace v3

using InternVLANextDiTRegistry = ModelHandleRegistry<v3::InternVLANextDiTModel>;

int64_t rpu_internvla_nextdit_create() {
    return InternVLANextDiTRegistry::create();
}

void rpu_internvla_nextdit_destroy(int64_t handle) {
    InternVLANextDiTRegistry::destroy(handle, "rpu_internvla_nextdit_destroy");
}

void rpu_internvla_nextdit_set_16b_sdpa(int64_t handle, bool enabled) {
    InternVLANextDiTRegistry::get(
        handle, "rpu_internvla_nextdit_set_16b_sdpa")->set_16b_sdpa(enabled);
}

void rpu_internvla_nextdit_set_fused_qkq_weights(
    int64_t handle, at::TensorList weights)
{
    InternVLANextDiTRegistry::get(
        handle, "rpu_internvla_nextdit_set_fused_qkq_weights")
        ->set_fused_qkq_weights(weights);
}

void rpu_internvla_nextdit_set_step_prepare_weights(
    int64_t handle, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& fold_norm_weight)
{
    InternVLANextDiTRegistry::get(
        handle, "rpu_internvla_nextdit_set_step_prepare_weights")
        ->set_step_prepare_weights(weight, bias, fold_norm_weight);
}

void rpu_internvla_nextdit_set_weights(
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
    int64_t ff_intermediate, double eps, double final_norm_eps)
{
    InternVLANextDiTRegistry::get(handle, "rpu_internvla_nextdit_set_weights")->set_weights(
        self_q_w, self_k_w, self_v_w, cross_q_w, cross_k_w, cross_v_w, out_w,
        ff1_w, ff2_w, ff3_w, norm1_w, norm2_w, ffn_norm1_w, ffn_norm2_w,
        self_qn_w, self_qn_b, self_kn_w, self_kn_b,
        cross_qn_w, cross_qn_b, cross_kn_w, cross_kn_b, context_norm_w,
        cross_gate_heads,
        final_out_w, final_out_b, final_ln_w, final_ln_b,
        action_enc_w, action_enc_b, action_position,
        action_dec_w, action_dec_b,
        num_heads, head_dim, hidden_size, ff_intermediate, eps, final_norm_eps);
}

at::Tensor rpu_internvla_nextdit_forward(
    int64_t handle, const at::Tensor& x, const at::Tensor& modulation,
    const at::Tensor& final_scale,
    const at::Tensor& raw_actions, bool use_action_encoder, bool populate_cross_cache,
    bool denoise_unroll, bool prepare_modulation_on_rpu,
    at::ArrayRef<double> euler_steps,
    const at::Tensor& cross_context,
    at::TensorList self_k_caches, at::TensorList self_v_caches,
    at::TensorList cross_k_caches, at::TensorList cross_v_caches,
    const at::Tensor& layer_outputs, bool collect_layer_outputs)
{
    return InternVLANextDiTRegistry::get(handle, "rpu_internvla_nextdit_forward")->forward(
        x, modulation, final_scale, raw_actions, use_action_encoder, populate_cross_cache,
        denoise_unroll, prepare_modulation_on_rpu, euler_steps,
        cross_context,
        self_k_caches, self_v_caches, cross_k_caches, cross_v_caches,
        layer_outputs, collect_layer_outputs);
}
