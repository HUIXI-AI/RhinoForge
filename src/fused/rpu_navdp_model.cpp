// rpu_navdp_model.cpp — InternVLA-N1 NavDP action-head fused subsystem (v3 FusedModelBase).
//
// NavDP = 16-layer nn.TransformerDecoder diffusion policy (System-1), PRE-LN (norm_first):
//   per layer:  x += self_attn( LN1(x) )            # causal over 32 action tokens
//               x += cross_attn( LN2(x), memory )   # 34 memory tokens (time+goal+rgbd)
//               x += FFN( LN3(x) )                   # 384->1536->384, GELU
//   d_model=384, nhead=8 (head_dim=48), ff=1536, memory_len=34, predict_size=32, action_dim=3.
//
// SCOPE: forward(tgt,memory) runs the 16 decoder layers on RPU for the host-driven path.
//   denoise_loop_forward additionally keeps input_embed, out_pos add, final LayerNorm,
//   action_head, and the fixed DDPM20 reverse loop in one device graph; Python prepares the
//   per-step conditioning, coefficients, and noise tables.
//
// Design/reuse: mirrors src/fused/rpu_gr00t_dit_model.cpp (build_self_attn_block /
//   build_cross_attn_block / finish_attn_and_ffn), swapping AdaLN -> plain affine LayerNorm
//   (learnable gamma/beta per norm) and adding a per-layer THREE-subblock structure.
// Validation boundary: full-hidden metrics are record-only; the exact profile is admitted by
// its final-action, action-MAE, performance, and Graph lifecycle gates.  Native handle creation
// requires the controlled numeric-blocked evaluation selector before creating a handle.
//
// set_weights ENDS with invalidate_model_state() (D-503).

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_helpers.h"
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace v3 {

// NAVDP_FIXED_KERNEL_BASIS: non-manifest launchers implement fixed NavDP
// normalization/activation, trajectory glue, or BufferDecl transport. They have
// no runtime candidate; alternatives must first become typed manifest routes.

namespace {

constexpr int64_t kNavdpSelfKvSite = 5578031918666539451LL;
constexpr int64_t kNavdpCrossKvSite = 5028686091549759246LL;
constexpr int64_t kNavdpSelfAttentionDdrSite = 5266334158747974487LL;
constexpr int64_t kNavdpSelfAttentionSpmSite = 7913923308750549684LL;
constexpr int64_t kNavdpCrossAttentionDdrSite = 4550112057379938325LL;
constexpr int64_t kNavdpCrossAttentionMemoryPrefixDdrRequired = 1LL << 8;
constexpr int64_t kNavdpLoopPreloadDmaSite = 5843232389164851264LL;
constexpr int64_t kNavdpLoopInitialStateDmaSite = 2768844808659448183LL;
constexpr int64_t kNavdpLoopInputLinearSite = 8311979808409279276LL;
constexpr int64_t kNavdpLoopMemoryDmaSite = 2471321219266394663LL;
constexpr int64_t kNavdpLoopOutputLinearSite = 5717796996455384990LL;
constexpr int64_t kNavdpLoopNoiseDmaSite = 1326351836944110747LL;
constexpr int64_t kNavdpLoopOutputDmaSite = 2600646422303964861LL;
constexpr int64_t kNavdpMemoryDmaSite = 2072388790201406506LL;
constexpr int64_t kNavdpSelfAllReduceSite = 2412992928182911522LL;
constexpr int64_t kNavdpFormerCrossAllReduceSite = 985603305552088294LL;
constexpr int64_t kNavdpCrossAllReduceSite = 81349817211229034LL;
constexpr int64_t kNavdpFormerFfAllReduceSite = 2628574193033020498LL;
constexpr int64_t kNavdpFfAllReduceSite = 8341276851067680877LL;
constexpr int64_t kNavdpProjectionLinearSite = 7289281567548044119LL;
constexpr int64_t kNavdpExecutionScheduleSite = 8388526521297846072LL;
constexpr int64_t kNavdpFormerScheduleSite = 1364822026573774829LL;
constexpr uint32_t kNavdpKvCapabilities =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16;

enum class NavdpDmaRoute : int64_t {
    DDR_TO_SPM = 1,
    SPM_TO_DDR = 2,
};

}  // namespace

class NavdpModel : public FusedModelBase {
private:
    static constexpr int64_t kNavdpLayers = 16;
    static constexpr int64_t kFormerLayers = 2;
    static constexpr int64_t kHidden = 384;
    static constexpr int64_t kNumHeads = 8;
    static constexpr int64_t kHeadDim = 48;
    static constexpr int64_t kNavdpFf = 1536;
    static constexpr int64_t kFormerFf = 2048;
    static constexpr int64_t kNavdpMemory = 34;
    static constexpr int64_t kFormerMemory = 1024;
    static constexpr int64_t kPredictSize = 32;
    static constexpr int64_t kActionDim = 3;
    static constexpr int64_t kActionDimPadded = 16;
    static constexpr int64_t kDenoiseSteps = 20;
    static constexpr double kEps = 1e-5;

    enum class ExecutionMode : uint8_t { Unset, Forward, Denoise };

    static void check_rpu_half(
        const at::Tensor& tensor, const char* name, at::IntArrayRef expected_sizes)
    {
        TORCH_CHECK(tensor.defined(), "navdp: ", name, " is undefined");
        TORCH_CHECK(tensor.sizes() == expected_sizes,
                    "navdp: ", name, " must have shape ", expected_sizes,
                    ", got ", tensor.sizes());
        TORCH_CHECK(tensor.scalar_type() == at::kHalf &&
                    tensor.device().type() == at::kPrivateUse1 &&
                    tensor.is_contiguous(),
                    "navdp: ", name, " must be fp16 contiguous RPU tensor");
    }

    void check_caches(
        const std::vector<at::Tensor>& k_caches,
        const std::vector<at::Tensor>& v_caches,
        int64_t batch) const
    {
        TORCH_CHECK(static_cast<int64_t>(k_caches.size()) == num_layers() &&
                    static_cast<int64_t>(v_caches.size()) == num_layers(),
                    "navdp: k/v cache lists must each have ", num_layers(),
                    " tensors, got ", k_caches.size(), " and ", v_caches.size());
        const int64_t min_blocks =
            (std::max(memory_len_, predict_size_) + 15) / 16;
        for (int64_t i = 0; i < num_layers(); ++i) {
            const auto& kc = k_caches[i];
            const auto& vc = v_caches[i];
            TORCH_CHECK(kc.defined() && vc.defined(),
                        "navdp: cache[", i, "] is undefined");
            TORCH_CHECK(kc.dim() == 7 && vc.dim() == 7 &&
                        kc.sizes() == vc.sizes() &&
                        kc.size(0) == batch && kc.size(1) >= min_blocks &&
                        kc.size(2) == 1 && kc.size(3) == 3 &&
                        kc.size(4) == 8 && kc.size(5) == 16 && kc.size(6) == 16,
                        "navdp: cache[", i,
                        "] must use 7D RPUCache topology [", batch,
                        ", >=", min_blocks, ", 1, 3, 8, 16, 16], got k=",
                        kc.sizes(), " v=", vc.sizes());
            TORCH_CHECK(kc.scalar_type() == at::kHalf &&
                        vc.scalar_type() == at::kHalf &&
                        kc.device().type() == at::kPrivateUse1 &&
                        vc.device().type() == at::kPrivateUse1 &&
                        kc.is_contiguous() && vc.is_contiguous(),
                        "navdp: cache[", i,
                        "] must be fp16 contiguous RPU tensors");
        }
    }

public:
    // Per-decoder-layer weights (fp16, swizzled Python-side). QKV are split from the packed
    // nn.MultiheadAttention in_proj on the adapter side (q/k/v col-partition; o row-partition).
    struct LayerWeights {
        at::Tensor sq_w, sk_w, sv_w, so_w;       // self-attn q/k/v/out weights
        at::Tensor sq_b, sk_b, sv_b, so_b;       // self-attn biases
        at::Tensor cq_w, ck_w, cv_w, co_w;       // cross-attn q/k/v/out weights
        at::Tensor cq_b, ck_b, cv_b, co_b;       // cross-attn biases
        at::Tensor ff1_w, ff1_b, ff2_w, ff2_b;   // FFN 384->1536->384
        at::Tensor n1_w, n1_b, n2_w, n2_b, n3_w, n3_b;  // LayerNorm gamma/beta (affine)
    };

    NavdpModel() = default;

    void set_weights(
        at::TensorList sq_w, at::TensorList sk_w, at::TensorList sv_w, at::TensorList so_w,
        at::TensorList sq_b, at::TensorList sk_b, at::TensorList sv_b, at::TensorList so_b,
        at::TensorList cq_w, at::TensorList ck_w, at::TensorList cv_w, at::TensorList co_w,
        at::TensorList cq_b, at::TensorList ck_b, at::TensorList cv_b, at::TensorList co_b,
        at::TensorList ff1_w, at::TensorList ff1_b, at::TensorList ff2_w, at::TensorList ff2_b,
        at::TensorList n1_w, at::TensorList n1_b, at::TensorList n2_w, at::TensorList n2_b,
        at::TensorList n3_w, at::TensorList n3_b,
        int64_t num_heads, int64_t head_dim, int64_t hidden_size,
        int64_t ff_inter, int64_t memory_len, int64_t predict_size,
        int64_t action_dim, double eps)
    {
        const int64_t N = static_cast<int64_t>(sq_w.size());
        TORCH_CHECK(!weights_set_, "navdp_set_weights: weights are already installed");
        const bool navdp_profile =
            !former_mode_ && N == kNavdpLayers && num_heads == kNumHeads &&
            head_dim == kHeadDim && hidden_size == kHidden &&
            ff_inter == kNavdpFf && memory_len == kNavdpMemory &&
            predict_size == kPredictSize && action_dim == kActionDim && eps == kEps;
        const bool former_profile =
            former_mode_ && N == kFormerLayers && num_heads == kNumHeads &&
            head_dim == kHeadDim && hidden_size == kHidden &&
            ff_inter == kFormerFf && memory_len == kFormerMemory &&
            predict_size == kPredictSize && action_dim == kActionDim && eps == kEps;
        TORCH_CHECK(navdp_profile || former_profile,
                    "navdp_set_weights: unsupported profile; expected NavDP "
                    "(L=16,H=384,NH=8,HD=48,FF=1536,M=34,Q=32,A=3,eps=1e-5) "
                    "or Former (L=2,H=384,NH=8,HD=48,FF=2048,M=1024,Q=32,A=3,eps=1e-5)");

        auto check_length = [N](at::TensorList list, const char* name) {
            TORCH_CHECK(static_cast<int64_t>(list.size()) == N,
                        "navdp_set_weights: ", name, " must have ", N,
                        " tensors, got ", list.size());
        };
        check_length(sq_w, "sq_w"); check_length(sk_w, "sk_w");
        check_length(sv_w, "sv_w"); check_length(so_w, "so_w");
        check_length(sq_b, "sq_b"); check_length(sk_b, "sk_b");
        check_length(sv_b, "sv_b"); check_length(so_b, "so_b");
        check_length(cq_w, "cq_w"); check_length(ck_w, "ck_w");
        check_length(cv_w, "cv_w"); check_length(co_w, "co_w");
        check_length(cq_b, "cq_b"); check_length(ck_b, "ck_b");
        check_length(cv_b, "cv_b"); check_length(co_b, "co_b");
        check_length(ff1_w, "ff1_w"); check_length(ff1_b, "ff1_b");
        check_length(ff2_w, "ff2_w"); check_length(ff2_b, "ff2_b");
        check_length(n1_w, "n1_w"); check_length(n1_b, "n1_b");
        check_length(n2_w, "n2_w"); check_length(n2_b, "n2_b");
        check_length(n3_w, "n3_w"); check_length(n3_b, "n3_b");

        auto check_list = [](at::TensorList list, const char* name,
                             at::IntArrayRef sizes) {
            for (int64_t i = 0; i < static_cast<int64_t>(list.size()); ++i) {
                TORCH_CHECK(list[i].defined(), "navdp_set_weights: ", name,
                            "[", i, "] is undefined");
                TORCH_CHECK(list[i].sizes() == sizes,
                            "navdp_set_weights: ", name, "[", i,
                            "] must have shape ", sizes, ", got ", list[i].sizes());
                TORCH_CHECK(list[i].scalar_type() == at::kHalf &&
                            list[i].device().type() == at::kPrivateUse1 &&
                            list[i].is_contiguous(),
                            "navdp_set_weights: ", name, "[", i,
                            "] must be fp16 contiguous RPU tensor");
            }
        };
        check_list(sq_w, "sq_w", {hidden_size, hidden_size});
        check_list(sk_w, "sk_w", {hidden_size, hidden_size});
        check_list(sv_w, "sv_w", {hidden_size, hidden_size});
        check_list(so_w, "so_w", {hidden_size, hidden_size});
        check_list(cq_w, "cq_w", {hidden_size, hidden_size});
        check_list(ck_w, "ck_w", {hidden_size, hidden_size});
        check_list(cv_w, "cv_w", {hidden_size, hidden_size});
        check_list(co_w, "co_w", {hidden_size, hidden_size});
        check_list(ff1_w, "ff1_w", {ff_inter, hidden_size});
        check_list(ff2_w, "ff2_w", {hidden_size, ff_inter});
        check_list(ff1_b, "ff1_b", {ff_inter});
        check_list(sq_b, "sq_b", {hidden_size});
        check_list(sk_b, "sk_b", {hidden_size});
        check_list(sv_b, "sv_b", {hidden_size});
        check_list(so_b, "so_b", {hidden_size});
        check_list(cq_b, "cq_b", {hidden_size});
        check_list(ck_b, "ck_b", {hidden_size});
        check_list(cv_b, "cv_b", {hidden_size});
        check_list(co_b, "co_b", {hidden_size});
        check_list(ff2_b, "ff2_b", {hidden_size});
        check_list(n1_w, "n1_w", {hidden_size});
        check_list(n1_b, "n1_b", {hidden_size});
        check_list(n2_w, "n2_w", {hidden_size});
        check_list(n2_b, "n2_b", {hidden_size});
        check_list(n3_w, "n3_w", {hidden_size});
        check_list(n3_b, "n3_b", {hidden_size});

        eps_ = eps; memory_len_ = memory_len; predict_size_ = predict_size; action_dim_ = action_dim;
        layer_weights_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            auto& lw = layer_weights_[i];
            lw.sq_w = sq_w[i]; lw.sk_w = sk_w[i]; lw.sv_w = sv_w[i]; lw.so_w = so_w[i];
            lw.sq_b = sq_b[i]; lw.sk_b = sk_b[i]; lw.sv_b = sv_b[i]; lw.so_b = so_b[i];
            lw.cq_w = cq_w[i]; lw.ck_w = ck_w[i]; lw.cv_w = cv_w[i]; lw.co_w = co_w[i];
            lw.cq_b = cq_b[i]; lw.ck_b = ck_b[i]; lw.cv_b = cv_b[i]; lw.co_b = co_b[i];
            lw.ff1_w = ff1_w[i]; lw.ff1_b = ff1_b[i]; lw.ff2_w = ff2_w[i]; lw.ff2_b = ff2_b[i];
            lw.n1_w = n1_w[i]; lw.n1_b = n1_b[i]; lw.n2_w = n2_w[i]; lw.n2_b = n2_b[i];
            lw.n3_w = n3_w[i]; lw.n3_b = n3_b[i];
        }
        set_model_params(num_heads, num_heads, head_dim, hidden_size, ff_inter);
        set_num_layers(N);
        weights_set_ = true;
        invalidate_model_state();  // D-503
    }

    // memory[memory_len, hidden] held per-forward (cross-attn K/V source, broadcast to SPM once).
    at::Tensor forward(
        const at::Tensor& tgt,               // [seq_q=predict_size, hidden]  (already input_embed'd host-side)
        const at::Tensor& memory,            // [memory_len, hidden]
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& causal_mask,
        at::IntArrayRef planned_stage_descriptor = {})
    {
        TORCH_CHECK(weights_set_, "navdp forward: weights are not installed");
        TORCH_CHECK(execution_mode_ != ExecutionMode::Denoise,
                    "navdp forward: handle is locked to denoise mode");
        TORCH_CHECK(!causal_mask.has_value(),
                    "navdp forward: causal_mask is unsupported");
        check_rpu_half(tgt, "tgt", {1, predict_size_, hidden_size()});
        check_rpu_half(memory, "memory", {1, memory_len_, hidden_size()});
        check_caches(k_caches, v_caches, /*batch=*/1);
        execution_mode_ = ExecutionMode::Forward;
        // Model-owned SEPARATE cross-attn caches (gr00t alternates block types → never reuses a
        // cache for self-then-cross in one layer; sharing it here corrupted the cross SDPA).
        if (static_cast<int64_t>(cross_kc_.size()) != num_layers()) {
            cross_kc_.clear(); cross_vc_.clear();
            // Cross-attn memory is SHARED across trajectories → keep the cross caches batch-1 even
            // when the self-attn caches are batch-B (narrow(0,0,1) is the whole tensor when B==1).
            for (auto& kc : k_caches) cross_kc_.push_back(at::zeros_like(kc.narrow(0, 0, 1)));
            for (auto& vc : v_caches) cross_vc_.push_back(at::zeros_like(vc.narrow(0, 0, 1)));
        }
        // Device DDR address (NOT the raw data_ptr) — the mutable broadcast's live_src_base must be
        // the RPU device address (no internal conversion, unlike the fixed variant). Using the raw
        // ptr was the cross-attn bug (mis-diagnosed as fp16). Mirrors gr00t vl_src_base_.
        mem_src_base_ = ::rhino_lkn::RpuGetDevAddr(memory.data_ptr<c10::Half>());
        mem_loaded_this_forward_ = false;
        const std::vector<ChunkInfo> input_chunks{
            {0, 0, predict_size_, predict_size_}};
        const std::vector<FmbExecutionSpan> spans{{0, predict_size_}};
        auto out = planned_stage_descriptor.empty()
            ? run_all_layers(
                  tgt, k_caches, v_caches, std::nullopt,
                  /*position=*/0, /*is_causal=*/!former_mode_,
                  input_chunks, spans)
            : run_all_layers(
                  tgt, k_caches, v_caches, std::nullopt,
                  /*position=*/0, /*is_causal=*/!former_mode_,
                  /*planned_chunk_size=*/0, planned_stage_descriptor);
        return out;
    }

    void validate_execution_admission(
        int64_t packed_rows, int64_t denoise_steps) const {

        TORCH_CHECK(weights_set_,
                    "navdp_resolve_stage_domain: weights are not installed");
        TORCH_CHECK(
            packed_rows > 0 && packed_rows % predict_size_ == 0,
            "navdp_resolve_stage_domain: packed_rows must be a positive "
            "multiple of predict_size=", predict_size_, ", got ", packed_rows);
        const int64_t requested_batch = packed_rows / predict_size_;
        const bool denoise_loop = denoise_steps != 0;
        TORCH_CHECK(
            denoise_loop ? denoise_steps == kDenoiseSteps : requested_batch == 1,
            "navdp_resolve_stage_domain: ordinary forward requires one "
            "trajectory and denoise requires exactly ", kDenoiseSteps,
            " steps");
        TORCH_CHECK(
            former_mode_ ? !denoise_loop : requested_batch <= 4,
            "navdp_resolve_stage_domain: unsupported trajectory batch ",
            requested_batch, former_mode_ ? " for Former" : " for NavDP");
        TORCH_CHECK(
            !(denoise_loop && execution_mode_ == ExecutionMode::Forward) &&
                !(!denoise_loop && execution_mode_ == ExecutionMode::Denoise),
            "navdp_resolve_stage_domain: requested lifecycle conflicts with "
            "the handle's latched execution mode");
        if (execution_mode_ == ExecutionMode::Denoise) {
            TORCH_CHECK(
                batch_ == requested_batch,
                "navdp_resolve_stage_domain: denoise batch differs from the "
                "latched handle batch");
        }
    }

    std::vector<int64_t> resolve_stage_domain(
        int64_t packed_rows, int64_t denoise_steps) {
        validate_execution_admission(packed_rows, denoise_steps);
        const int64_t requested_batch = packed_rows / predict_size_;
        const bool denoise_loop = denoise_steps != 0;
        const int64_t saved_batch = batch_;
        const bool saved_loop_mode = loop_mode_;
        const int64_t saved_num_steps = num_steps_;
        const int64_t saved_override = get_chunk_size_override();
        auto restore = c10::make_scope_exit([&] {
            batch_ = saved_batch;
            loop_mode_ = saved_loop_mode;
            num_steps_ = saved_num_steps;
            set_chunk_size_override(saved_override);
        });
        batch_ = requested_batch;
        loop_mode_ = denoise_loop;
        num_steps_ = denoise_loop ? denoise_steps : 1;
        set_chunk_size_override(0);

        const std::vector<ChunkInfo> input_chunks{
            {0, 0, packed_rows, packed_rows}};
        std::vector<FmbExecutionSpan> spans;
        spans.reserve(requested_batch);
        for (int64_t stream = 0; stream < requested_batch; ++stream) {
            spans.push_back(
                {stream * predict_size_, predict_size_, stream});
        }
        const FmbStageBoundaryPolicies boundary_policies{};
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                packed_rows, /*position=*/0,
                /*attention_mask=*/std::nullopt,
                /*is_causal=*/!former_mode_, input_chunks, spans,
                boundary_policies));
    }

protected:
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            batch_, loop_mode_,
            num_steps_);
    }

    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers = num_layers();
        cfg.cross_layer_batch_size = num_layers();
        // DDPM in-graph unroll: loop_mode_ makes run_all_layers loop the body num_steps_
        // times in ONE graph (ctx().body_iter advances). Default off → single-step identical.
        cfg.body_iterations = loop_mode_ ? num_steps_ : 1;
        if (loop_mode_) {
            cfg.pre_layers_fn  = reinterpret_cast<void (FusedModelBase::*)()>(
                &NavdpModel::emit_pre_layers_body);
            cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
                &NavdpModel::emit_post_layers_body);
        }
        return cfg;
    }

    bool subclass_chunk_size_valid(
        int64_t chunk_size, int64_t seq_len,
        int64_t position) const override {
        // Self-attention is block-diagonal in 32-row trajectories, while the
        // unrolled post hook consumes the complete packed residual from SPM.
        // A partial/misaligned execution chunk would cross a trajectory or
        // leave only the final chunk resident, so fail planning instead of
        // silently changing semantics.
        return position == 0 && seq_len > 0 &&
            seq_len % predict_size_ == 0 && chunk_size == seq_len;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        TORCH_CHECK(
            plan.num_chunks == 1 && plan.chunk_size == bps(),
            "NavDP requires one complete, trajectory-aligned compute chunk; "
            "got chunk_size=", plan.chunk_size,
            " num_chunks=", plan.num_chunks, " packed_rows=", bps());
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;  // 32x384 fp16 ~24KB
        cfg.attention_policy = spm_kv_by_mha_enabled_
            ? AttentionExecutionPolicy::AUTO
            : AttentionExecutionPolicy::DDR_KV;
        return cfg;
    }

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
        TORCH_CHECK(
            position == 0 && plan.compute.chunks.size() == 1,
            "NavDP COMPLETE descriptor requires one position-zero chunk");
        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = std::max(logical_len, memory_len_);
        manifest.kv_insert_physical_rows =
            std::max(physical_len, memory_len_);
        manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation = FmbLinearAccumulationPolicy::ACC16;

        const int64_t invocation = plan.compute.chunks.front().idx;
        const auto append = [&manifest, invocation](
                                FmbRouteFamily family, int64_t site_id,
                                int64_t selector, int64_t flags = 0,
                                std::vector<int64_t> arguments = {},
                                std::optional<int64_t> route_invocation =
                                    std::nullopt) {
            manifest.routes.push_back(
                {site_id, family, selector, flags, std::move(arguments),
                 route_invocation.value_or(invocation)});
        };
        const auto append_linear = [&append](int64_t site_id) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE));
        };
        const auto append_dma = [&append](
                                    int64_t site_id, NavdpDmaRoute route,
                                    std::vector<int64_t> arguments = {},
                                    std::optional<int64_t> route_invocation =
                                        std::nullopt) {
            append(FmbRouteFamily::MUTABLE_DMA, site_id,
                   static_cast<int64_t>(route), /*flags=*/0,
                   std::move(arguments), route_invocation);
        };
        const int64_t body_iterations = loop_mode_ ? num_steps_ : 1;
        append(
            FmbRouteFamily::GRAPH_SCHEDULE, kNavdpExecutionScheduleSite,
            loop_mode_ ? 2 : 1, /*flags=*/0,
            {loop_mode_ ? 1 : 0, num_steps_, batch_, body_iterations});
        append(
            FmbRouteFamily::GRAPH_SCHEDULE, kNavdpFormerScheduleSite,
            former_mode_ ? 2 : 1, /*flags=*/0,
            {former_mode_ ? 1 : 0,
             former_mode_ ? 0 : 1,  // non-causal / causal
             former_mode_ ? 1 : 0,  // post-LN / pre-LN
             former_mode_ ? 1 : 0});  // ReLU / GELU

        // Site 1 is trajectory-local self attention.  Its exact NavDP profile
        // can consume raw SPM K/V; Former and disabled-policy handles remain
        // DDR.  Site 2 is prefix/memory cross attention and is DDR_REQUIRED:
        // the existing ABI takes RPUCache tensors and has no raw-residency port.
        const AttentionExecutionPolicy self_policy =
            !former_mode_ && spm_kv_by_mha_enabled_ &&
                layout.attention_policy ==
                    AttentionExecutionPolicy::SPM_KV_BY_MHA
            ? AttentionExecutionPolicy::SPM_KV_BY_MHA
            : AttentionExecutionPolicy::DDR_KV;
        append(
            FmbRouteFamily::ATTENTION,
            self_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA
                ? kNavdpSelfAttentionSpmSite
                : kNavdpSelfAttentionDdrSite,
            static_cast<int64_t>(self_policy),
            /*flags=*/former_mode_ ? 0 : 1,
            {batch_, predict_size_, physical_len});
        append(
            FmbRouteFamily::ATTENTION, kNavdpCrossAttentionDdrSite,
            static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
            kNavdpCrossAttentionMemoryPrefixDdrRequired,
            {batch_, predict_size_, memory_len_});
        // Self keeps a DDR mirror even when attention consumes raw SPM K/V;
        // cross has no raw-residency route in the current launcher ABI.
        const KvInsertSegmentPlan self_kv_plan =
            resolve_kvinsert_plan_auto(
                kNavdpSelfKvSite, manifest.graph_lifecycle,
                /*position=*/0, predict_size_, predict_size_, NUM_CORES,
                num_kv_heads(), head_dim(), kNavdpKvCapabilities);
        const KvInsertRouteArguments self_kv_arguments =
            rpu_kvinsert_route_arguments(
                self_kv_plan, NUM_CORES, num_kv_heads(), head_dim());
        append(
            FmbRouteFamily::KV_INSERT, kNavdpSelfKvSite,
            static_cast<int64_t>(self_kv_plan.route()), /*flags=*/0,
            std::vector<int64_t>(self_kv_arguments.begin(),
                                 self_kv_arguments.end()));
        const KvInsertSegmentPlan cross_kv_plan =
            resolve_kvinsert_plan_auto(
                kNavdpCrossKvSite, manifest.graph_lifecycle,
                /*position=*/0, memory_len_, memory_len_, NUM_CORES,
                num_kv_heads(), head_dim(), kNavdpKvCapabilities);
        const KvInsertRouteArguments cross_kv_arguments =
            rpu_kvinsert_route_arguments(
                cross_kv_plan, NUM_CORES, num_kv_heads(), head_dim());
        append(
            FmbRouteFamily::KV_INSERT, kNavdpCrossKvSite,
            static_cast<int64_t>(cross_kv_plan.route()), /*flags=*/0,
            std::vector<int64_t>(cross_kv_arguments.begin(),
                                 cross_kv_arguments.end()));

        append_linear(kNavdpProjectionLinearSite);
        append(
            FmbRouteFamily::ALL_REDUCE, kNavdpSelfAllReduceSite,
            fmb_ring_all_reduce_route_selector(physical_len, hidden_size()));
        append(
            FmbRouteFamily::ALL_REDUCE,
            former_mode_ ? kNavdpFormerCrossAllReduceSite
                         : kNavdpCrossAllReduceSite,
            fmb_ring_all_reduce_route_selector(physical_len, hidden_size()));
        append(
            FmbRouteFamily::ALL_REDUCE,
            former_mode_ ? kNavdpFormerFfAllReduceSite
                         : kNavdpFfAllReduceSite,
            fmb_ring_all_reduce_route_selector(physical_len, hidden_size()));

        if (loop_mode_) {
            const int64_t action_dim_padded =
                ((action_dim_ + 15) / 16) * 16;
            const int64_t preload_sizes[] = {
                hidden_size(), action_dim_padded, hidden_size(),
                hidden_size(), predict_size_ * hidden_size()};
            for (int64_t i = 0; i < 5; ++i) {
                append_dma(
                    kNavdpLoopPreloadDmaSite, NavdpDmaRoute::DDR_TO_SPM,
                    {num_steps_, preload_sizes[i]}, i);
            }
            append_dma(
                kNavdpLoopInitialStateDmaSite, NavdpDmaRoute::DDR_TO_SPM,
                {batch_, predict_size_, action_dim_padded});
            append_linear(kNavdpLoopInputLinearSite);
            append_dma(
                kNavdpLoopMemoryDmaSite, NavdpDmaRoute::DDR_TO_SPM,
                {num_steps_, memory_len_, hidden_size()});
            append_linear(kNavdpLoopOutputLinearSite);
            append_dma(
                kNavdpLoopNoiseDmaSite, NavdpDmaRoute::DDR_TO_SPM,
                {num_steps_, batch_, predict_size_, action_dim_padded});
            append_dma(
                kNavdpLoopOutputDmaSite, NavdpDmaRoute::SPM_TO_DDR,
                {batch_, predict_size_, action_dim_padded});
        } else {
            append_dma(
                kNavdpMemoryDmaSite, NavdpDmaRoute::DDR_TO_SPM,
                {memory_len_, hidden_size()});
        }
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA);
        return manifest;
    }

    std::vector<FmbPhysicalExecutionManifest>
    physical_manifest_domain_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
        LayoutContext ddr_layout = layout;
        ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;
        std::vector<FmbPhysicalExecutionManifest> domain{
            physical_manifest_for_candidate(
                plan, ddr_layout, physical_len, logical_len, position)};

        LayoutContext raw_layout = layout;
        raw_layout.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        if (subclass_spm_kv_by_mha_eligible(
                plan, raw_layout, position)) {
            domain.push_back(physical_manifest_for_candidate(
                plan, raw_layout, physical_len, logical_len, position));
        }
        return domain;
    }

    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& /*manifest*/) const override {
        return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
    }

    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(0, memory_len_);
        h = detail::layout_mix(h, predict_size_);
        h = detail::layout_mix(h, action_dim_);
        h = detail::layout_mix(h, former_mode_ ? 1 : 0);
        h = detail::layout_mix(h, loop_mode_ ? 1 : 0);
        h = detail::layout_mix(h, num_steps_);
        h = detail::layout_mix(h, batch_);
        return h;
    }

    bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const override {
        if (!spm_kv_by_mha_enabled_ || former_mode_ || position != 0 ||
            !layout.is_causal || layout.use_attn_mask ||
            layout.batch_size != 1 ||
            layout.attention_policy !=
                AttentionExecutionPolicy::SPM_KV_BY_MHA ||
            plan.chunk_mode != ChunkMode::SEQUENTIAL ||
            plan.input.chunks.size() != 1 ||
            plan.qkv.chunks.size() != 1 ||
            plan.compute.chunks.size() != 1 || plan.spans.empty()) {
            return false;
        }

        const int64_t packed_rows = bps();
        if (packed_rows <= 0 || plan.compute.chunks.front().offset != 0 ||
            plan.compute.chunks.front().len != packed_rows ||
            plan.compute.chunks.front().kv_seq_len != packed_rows ||
            plan.qkv.chunks.front().len != packed_rows ||
            plan.input.chunks.front().len != packed_rows ||
            layout.chunk_size != packed_rows ||
            layout.max_kv_seq_len != packed_rows ||
            static_cast<int64_t>(plan.spans.size()) != batch_) {
            return false;
        }
        for (int64_t b = 0; b < batch_; ++b) {
            if (plan.spans[b].offset != b * predict_size_ ||
                plan.spans[b].len != predict_size_) {
                return false;
            }
        }
        return num_layers() == kNavdpLayers && hidden_size() == kHidden &&
            num_q_heads() == kNumHeads && num_kv_heads() == kNumHeads &&
            head_dim() == kHeadDim && intermediate_size() == kNavdpFf &&
            memory_len_ == kNavdpMemory && predict_size_ == kPredictSize &&
            action_dim_ == kActionDim &&
            sdpa_by_mha_spm_is_valid(
                batch_, predict_size_, predict_size_,
                num_q_heads(), num_kv_heads(), head_dim(), NUM_CORES,
                /*MASK_LTM=*/1);
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        const int64_t cs = ctx.chunk_size;
        const int64_t h  = hidden_size();
        const int64_t nq = num_q_heads();
        const int64_t hd = head_dim();
        const int64_t FF = intermediate_size();
        const int64_t S  = memory_len_;
        const int64_t local_qhd   = (nq * hd) / NUM_CORES;
        const int64_t local_inter = FF / NUM_CORES;
        const int64_t kv_seq = std::max<int64_t>(cs, S);
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        const int64_t res    = A(cs * h * DWIDTH);
        const int64_t qbuf   = A(cs * local_qhd * DWIDTH);
        const int64_t kvbuf  = A(kv_seq * local_qhd * DWIDTH);
        const int64_t outbuf = A(cs * local_qhd * DWIDTH);
        const int64_t mlp    = A(cs * local_inter * DWIDTH);
        const int64_t mem_sz = A(S * h * DWIDTH);
        const SdpaConfig cfg_self {SdpaKernelType::FLASH_ATTN_SPM, hd, nq, nq, static_cast<int>(attn_tp()), 1};  // causal
        const SdpaConfig cfg_cross{SdpaKernelType::FLASH_ATTN_SPM, hd, nq, nq, static_cast<int>(attn_tp()), 0};  // none
        int64_t tmp_self  = sdpa_compute_tmp_v16_size(cfg_self, cs);
        int64_t tmp_cross = sdpa_compute_tmp_v16_size(cfg_cross, cs);
        int64_t tmp = A(std::max(tmp_self, tmp_cross) * 32);
        if (ctx.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            // The raw self-attention V^T layout is
            // [trajectory,heads/core,head_dim,predict_size]. Cross-attention
            // remains on the durable DDR-cache path and reuses this workspace.
            tmp = std::max(tmp, A(cs * local_qhd * DWIDTH));
        }

        constexpr BufferScope ALL = BufferScope::LayerWide;
        std::vector<BufferDecl> v = {
            {"residual",    res,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"acc",         res,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"norm_hidden", res,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"q",           qbuf,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"k",           kvbuf,  0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"v",           kvbuf,  0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_out",    outbuf, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"oproj",       res,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_tmp",    tmp,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"ff_mid",      mlp,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"memory",      mem_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
        };

        const int nl = static_cast<int>(num_layers());
        // Per-layer LayerNorm gamma/beta ([h], broadcast to all cores).
        auto ln_vec = [&](const char* name, at::Tensor LayerWeights::* field) {
            BufferDecl d; d.name = name; d.size = A(h * DWIDTH);
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl; d.scope = ALL;
            d.preload_callback = [this, field](FusedModelBase&, int L, uint32_t a0) {
                rpu_launch_ddr_broadcast_spm_dma(
                    (layer_weights_[L].*field).data_ptr<c10::Half>(), hidden_size(), a0);
            };
            v.push_back(d);
        };
        ln_vec("n1w", &LayerWeights::n1_w); ln_vec("n1b", &LayerWeights::n1_b);
        ln_vec("n2w", &LayerWeights::n2_w); ln_vec("n2b", &LayerWeights::n2_b);
        ln_vec("n3w", &LayerWeights::n3_w); ln_vec("n3b", &LayerWeights::n3_b);

        // Per-layer linear biases: col-partition (q/k/v/ff1) + row-broadcast (o/ff2).
        auto col_bias = [&](const char* name, int64_t per_core, at::Tensor LayerWeights::* field) {
            BufferDecl d; d.name = name; d.size = A(per_core * DWIDTH);
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl; d.scope = ALL;
            d.preload_callback = [this, per_core, field](FusedModelBase&, int L, uint32_t a0) {
                rpu_launch_ddr_scatter_spm_dma(
                    (layer_weights_[L].*field).data_ptr<c10::Half>(),
                    per_core, per_core * DWIDTH, a0, NUM_CORES);
            };
            v.push_back(d);
        };
        auto row_bias = [&](const char* name, at::Tensor LayerWeights::* field) {
            BufferDecl d; d.name = name; d.size = A(h * DWIDTH);
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl; d.scope = ALL;
            d.preload_callback = [this, field](FusedModelBase&, int L, uint32_t a0) {
                rpu_launch_memset_spm_multicore(a0, hidden_size());
                rpu_launch_ddr_broadcast_spm_dma(
                    (layer_weights_[L].*field).data_ptr<c10::Half>(), hidden_size(), a0, /*num_cores=*/1);
            };
            v.push_back(d);
        };
        col_bias("sq_bias", local_qhd, &LayerWeights::sq_b);
        col_bias("sk_bias", local_qhd, &LayerWeights::sk_b);
        col_bias("sv_bias", local_qhd, &LayerWeights::sv_b);
        col_bias("cq_bias", local_qhd, &LayerWeights::cq_b);
        col_bias("ck_bias", local_qhd, &LayerWeights::ck_b);
        col_bias("cv_bias", local_qhd, &LayerWeights::cv_b);
        col_bias("ff1_bias", local_inter, &LayerWeights::ff1_b);
        row_bias("so_bias",  &LayerWeights::so_b);
        row_bias("co_bias",  &LayerWeights::co_b);
        row_bias("ff2_bias", &LayerWeights::ff2_b);

        if (loop_mode_) {
            const int64_t AD = ((action_dim_+15)/16)*16;   // pad action_dim→16 (alignment)
            const int64_t PS = bps();   // batch folded in: all loop buffers hold B trajectories (traj-major)
            // Denoise-loop private SPM (LayerWide → own slot, persists across body_iter).
            v.push_back(BufferDecl{"na_spm",   A(PS*AD*DWIDTH), 0,0, StorageClass::Temp,0,nullptr, ALL}); // trajectory state
            v.push_back(BufferDecl{"ae_spm",   A(PS*h*DWIDTH),  0,0, StorageClass::Temp,0,nullptr, ALL}); // input_embed out
            v.push_back(BufferDecl{"ln_spm",   A(PS*h*DWIDTH),  0,0, StorageClass::Temp,0,nullptr, ALL}); // final-LN out
            v.push_back(BufferDecl{"eps_spm",  A(PS*AD*DWIDTH), 0,0, StorageClass::Temp,0,nullptr, ALL}); // action_head out
            v.push_back(BufferDecl{"x0_spm",   A(PS*AD*DWIDTH), 0,0, StorageClass::Temp,0,nullptr, ALL}); // DDPM x0
            v.push_back(BufferDecl{"sigz_spm", A(PS*AD*DWIDTH), 0,0, StorageClass::Temp,0,nullptr, ALL}); // per-iter sig*z
            // Preloaded glue constants (broadcast once at BUILD; preload_callback is a NAMED field).
            auto add_pre = [&](const char* nm, int64_t nelem,
                               at::Tensor NavdpModel::* f,
                               int64_t route_invocation) {
                BufferDecl d; d.name = nm; d.size = A(nelem*DWIDTH);
                d.storage = StorageClass::Persistent; d.scope = ALL;
                d.preload_callback = [this, f, nelem, route_invocation](
                                         FusedModelBase&, int, uint32_t a0) {
                    this->ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        kNavdpLoopPreloadDmaSite,
                        static_cast<int64_t>(NavdpDmaRoute::DDR_TO_SPM),
                        /*resolved_flags=*/0, {num_steps_, nelem},
                        route_invocation);
                    rpu_launch_ddr_broadcast_spm_dma((this->*f).data_ptr<c10::Half>(), nelem, a0, NUM_CORES);
                };
                v.push_back(d);
            };
            add_pre("ie_b_spm", h, &NavdpModel::ie_b_, 0);
            add_pre("ah_b_spm", AD, &NavdpModel::ah_b_, 1);
            add_pre("flnw_spm", h, &NavdpModel::fln_w_, 2);
            add_pre("flnb_spm", h, &NavdpModel::fln_b_, 3);
            add_pre("outpos_spm", predict_size_*h, &NavdpModel::out_pos_, 4);  // [predict_size,h]; added per-traj block
        }
        return v;
    }

    void emit_pre_layers_body() {
        const int64_t h = hidden_size(), PS = bps(), S = memory_len_;   // PS = B*predict_size (traj-major)
        const int64_t PSc = predict_size_;   // per-trajectory rows (out_pos [PSc,h] shared across traj)
        const int64_t AD = ((action_dim_+15)/16)*16;   // padded action_dim (alignment)
        const int64_t i = ctx().body_iter;
        // [A1] body_iter 0: load init noise x0 → na_spm (mutable). iters 1+: na_spm holds the
        // post-hook's result (persists in its LayerWide slot).
        if (i == 0) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                kNavdpLoopInitialStateDmaSite,
                static_cast<int64_t>(NavdpDmaRoute::DDR_TO_SPM),
                /*resolved_flags=*/0, {batch_, predict_size_, AD});
            rpu_launch_ddr_broadcast_spm_dma_mutable(&x0_src_base_, 0, PS*AD, addr(0,"na_spm"), NUM_CORES);
        }
        // [A2] input_embed: na_spm[PS,AD] → ae_spm[PS,H] (single-core, K=AD) + bias; add out_pos.
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, kNavdpLoopInputLinearSite,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0);
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0,"na_spm"), ie_w_, addr(0,"ae_spm"),
            PS, h, AD, /*partition=*/1, /*num_cores=*/1, addr(0,"ie_b_spm"), at::Tensor());
        for (int64_t b = 0; b < batch_; ++b) {   // out_pos [PSc,h] added to each trajectory's ae block
            const uint32_t ob = static_cast<uint32_t>(b * PSc * h * DWIDTH);
            rpu_launch_eltwise_binary_spm_kernel(addr(0,"ae_spm")+ob, addr(0,"outpos_spm"), addr(0,"ae_spm")+ob,
                PSc*h, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
        }
        // [A3] memory row i → "memory" SPM (mutable, per-iter offset).
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA, kNavdpLoopMemoryDmaSite,
            static_cast<int64_t>(NavdpDmaRoute::DDR_TO_SPM),
            /*resolved_flags=*/0, {num_steps_, memory_len_, h});
        rpu_launch_ddr_broadcast_spm_dma_mutable(&mem_all_src_base_, i*S*h*DWIDTH, S*h, addr(0,"memory"), NUM_CORES);
        mem_loaded_this_forward_ = true;   // suppress the body's own memory broadcast
        // [A4] ae_spm → na_stage_ DDR (layer-0 reads it via FMB hidden_in_src_base_).
        rpu_launch_spm_copy_ddr_dma(addr(0,"ae_spm"), na_stage_.data_ptr<c10::Half>(), PS*h);
    }

    void emit_post_layers_body() {
        const int64_t h = hidden_size(), PS = bps();   // PS = B*predict_size (traj-major); affine is elementwise
        const int64_t AD = ((action_dim_+15)/16)*16;   // padded action_dim (alignment)
        const int64_t i = ctx().body_iter;
        const float* c = coeff_.data_ptr<float>() + i*4;
        const float A_=c[0], B_=c[1], C_=c[2], D_=c[3];
        // build_layer_subgraph left the final layer output in "residual".
        // [Z1] final-LN → ln_spm.
        layernorm(addr(0,"residual"), addr(0,"ln_spm"), addr(0,"flnw_spm"), addr(0,"flnb_spm"), PS, h);
        // [Z2] action_head: ln_spm[PS,H] → eps_spm[PS,AD] (single-core, K=H) + bias.
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, kNavdpLoopOutputLinearSite,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0);
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0,"ln_spm"), ah_w_, addr(0,"eps_spm"),
            PS, AD, h, /*partition=*/1, /*num_cores=*/1, addr(0,"ah_b_spm"), at::Tensor());
        // [Z3] DDPM affine: x0 = clip(A*na + B*eps, -1,1);  na = C*x0 + D*na;  na += sig*z (i<N-1).
        //   x0 = na + (B/A)*eps ; x0 *= A ; x0 = clip(x0, 1.0)
        rpu_launch_eltwise_binary_spm_kernel(addr(0,"na_spm"), addr(0,"eps_spm"), addr(0,"x0_spm"),
            PS*AD, ValuOpType::ADD, c10::Half(B_/A_), NUM_CORES);
        rpu_launch_eltwise_binary_scalar_spm_kernel(addr(0,"x0_spm"), c10::Half(A_), addr(0,"x0_spm"),
            PS*AD, ValuOpType::MUL);
        rpu_launch_eltwise_binary_scalar_spm_kernel(addr(0,"x0_spm"), c10::Half(1.0), addr(0,"x0_spm"),
            PS*AD, ValuOpType::MIN);   // clamp upper +1
        rpu_launch_eltwise_binary_scalar_spm_kernel(addr(0,"x0_spm"), c10::Half(-1.0), addr(0,"x0_spm"),
            PS*AD, ValuOpType::MAX);   // clamp lower -1  (CLIP scalar is one-sided → need both)
        //   na = x0 + (D/C)*na ; na *= C
        rpu_launch_eltwise_binary_spm_kernel(addr(0,"x0_spm"), addr(0,"na_spm"), addr(0,"na_spm"),
            PS*AD, ValuOpType::ADD, c10::Half(D_/C_), NUM_CORES);
        rpu_launch_eltwise_binary_scalar_spm_kernel(addr(0,"na_spm"), c10::Half(C_), addr(0,"na_spm"),
            PS*AD, ValuOpType::MUL);
        if (i < num_steps_ - 1) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, kNavdpLoopNoiseDmaSite,
                static_cast<int64_t>(NavdpDmaRoute::DDR_TO_SPM),
                /*resolved_flags=*/0,
                {num_steps_, batch_, predict_size_, AD});
            rpu_launch_ddr_broadcast_spm_dma_mutable(&sigz_src_base_, i*PS*AD*DWIDTH, PS*AD, addr(0,"sigz_spm"), NUM_CORES);
            rpu_launch_eltwise_binary_spm_kernel(addr(0,"na_spm"), addr(0,"sigz_spm"), addr(0,"na_spm"),
                PS*AD, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
        }
        // [Z4] last iter: na_spm → out DDR.
        if (i == num_steps_ - 1) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, kNavdpLoopOutputDmaSite,
                static_cast<int64_t>(NavdpDmaRoute::SPM_TO_DDR),
                /*resolved_flags=*/0, {batch_, predict_size_, AD});
            rpu_launch_spm_copy_ddr_dma_mutable(addr(0,"na_spm"), &out_dst_base_, 0, PS*AD);
        }
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        const int64_t seq = chunk.len;
        const int64_t h   = hidden_size();
        const int64_t nq  = num_q_heads();
        const int64_t hd  = head_dim();
        const int64_t FF  = intermediate_size();
        const int64_t S   = memory_len_;
        const int64_t li  = FF / NUM_CORES;
        const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
        // Trajectory batching: batch folded into seq (M=seq for all row-wise ops), but the two
        // attentions run per-trajectory (block-diagonal) over PSb-row slices + own KV-cache batch-slice.
        // B==1 (serial: seq==predict_size_, or seq not a multiple) → single narrow(0,0,1) iter,
        // byte-identical to the unbatched path. local_qhd = per-core q/k/v width.
        const int64_t local_qhd = (nq * hd) / NUM_CORES;
        const int64_t B    = (predict_size_ > 0 && seq > predict_size_ && seq % predict_size_ == 0)
                             ? seq / predict_size_ : 1;
        const int64_t PSb  = seq / B;
        const uint32_t qrow = static_cast<uint32_t>(PSb * local_qhd * DWIDTH);  // per-traj SPM row offset
        const int64_t body_iterations = loop_mode_ ? num_steps_ : 1;

        if (layer_idx == 0) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                kNavdpExecutionScheduleSite, loop_mode_ ? 2 : 1,
                /*resolved_flags=*/0,
                {loop_mode_ ? 1 : 0, num_steps_, batch_, body_iterations},
                chunk.idx);
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                kNavdpFormerScheduleSite, former_mode_ ? 2 : 1,
                /*resolved_flags=*/0,
                {former_mode_ ? 1 : 0,
                 former_mode_ ? 0 : 1,
                 former_mode_ ? 1 : 0,
                 former_mode_ ? 1 : 0},
                chunk.idx);
        }

        // Broadcast memory to SPM once per forward (cross-attn K/V input). MUTABLE variant: src is
        // rewritten per graph-replay from mem_src_base_ (a DEVICE addr, set in forward) → restores
        // graph-replay perf. (The earlier "fixed + fresh cache per forward" workaround is retired now
        // that mem_src_base_ is the correct device address, not the raw data_ptr.)
        if (layer_idx == 0 && chunk.idx == 0 && !mem_loaded_this_forward_) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, kNavdpMemoryDmaSite,
                static_cast<int64_t>(NavdpDmaRoute::DDR_TO_SPM),
                /*resolved_flags=*/0, {memory_len_, hidden_size()},
                chunk.idx);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                &mem_src_base_, /*src_offset_bytes=*/0, S * h, addr(0, "memory"), NUM_CORES);
            mem_loaded_this_forward_ = true;
        }
        if (!ctx().input_in_spm) emit_layer_input_dma(layer_idx, chunk, "residual");

        auto& kc = (*ctx().k_caches)[layer_idx];
        auto& vc = (*ctx().v_caches)[layer_idx];

        // former_mode_ = RGBD former_net perceiver: POST-LN (attn→add→norm), ReLU, non-causal self-attn.
        // navdp (false) = PRE-LN (norm→attn→add), GELU, causal self-attn — the unchanged `else` path.
        const int self_mask = former_mode_ ? 0 : 1;

        // ---- sub-block 1: self-attn ----
        if (!former_mode_)   // PRE-LN: normalize first, attn reads norm_hidden
            layernorm(addr(0, "residual"), addr(0, "norm_hidden"),
                      layer_addr(layer_idx, 0, "n1w"), layer_addr(layer_idx, 0, "n1b"), seq, h);
        const uint32_t sa_in = former_mode_ ? addr(0, "residual") : addr(0, "norm_hidden");
        proj(sa_in, lw.sq_w, addr(0, "q"), seq, nq * hd, h, 1, layer_addr(layer_idx, 0, "sq_bias"));
        proj(sa_in, lw.sk_w, addr(0, "k"), seq, nq * hd, h, 1, layer_addr(layer_idx, 0, "sk_bias"));
        proj(sa_in, lw.sv_w, addr(0, "v"), seq, nq * hd, h, 1, layer_addr(layer_idx, 0, "sv_bias"));
        const KvInsertSegmentPlan self_kv_plan = resolve_kv_insert_plan(
            kNavdpSelfKvSite, /*position=*/0, PSb);
        for (int64_t b = 0; b < B; ++b) {   // always mirror K/V to DDR for fallback/debug parity
            auto kcb = kc.narrow(0, b, 1);
            auto vcb = vc.narrow(0, b, 1);
            const uint32_t o = static_cast<uint32_t>(b) * qrow;
            rpu_launch_insert_kvcache_spm_unified_with_plan(
                kcb, vcb, addr_offset("k").value + o,
                addr_offset("v").value + o, nq, hd, NUM_CORES,
                /*k_cache_batch_offset_elems=*/0,
                /*v_cache_batch_offset_elems=*/0, /*spm_rows=*/0,
                self_kv_plan);
            if (ctx().attention_policy ==
                AttentionExecutionPolicy::DDR_KV) {
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION,
                    kNavdpSelfAttentionDdrSite,
                    static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                    /*resolved_flags=*/self_mask,
                    {batch_, predict_size_, seq}, chunk.idx);
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    kcb, vcb, self_mask, scale,
                    addr_offset("q").value + o, addr_offset("sdpa_out").value + o, addr_offset("sdpa_tmp").value, 0,
                    PSb, nq, nq, hd, /*kv_seq_len=*/PSb, NUM_CORES, NUM_CORES);
            }
        }
        if (ctx().attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            TORCH_CHECK(
                !former_mode_ && self_mask == 1 && PSb == kPredictSize &&
                    B == batch_,
                "NavDP raw-SPM self-attention escaped its exact profile");
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                kNavdpSelfAttentionSpmSite,
                static_cast<int64_t>(
                    AttentionExecutionPolicy::SPM_KV_BY_MHA),
                /*resolved_flags=*/self_mask,
                {batch_, predict_size_, seq}, chunk.idx);
            rpu_launch_v_transpose_spm(
                addr(0, "v"), addr(0, "sdpa_tmp"), B, PSb, nq, hd,
                NUM_CORES);
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q"), addr(0, "k"), addr(0, "sdpa_tmp"),
                addr(0, "sdpa_out"), /*mask_spm=*/0, self_mask, scale,
                B, PSb, PSb, nq, nq, hd, NUM_CORES);
        }
        proj(addr(0, "sdpa_out"), lw.so_w, addr(0, "oproj"), seq, h, nq * hd, 0, layer_addr(layer_idx, 0, "so_bias"));
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE, kNavdpSelfAllReduceSite,
            fmb_ring_all_reduce_route_selector(seq, h),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual"), addr(0, "acc"), seq, h, NUM_CORES, NUM_CORES);  // acc = reduce+x
        if (former_mode_)    // POST-LN: x = norm1(x + self_attn) → into "residual"
            layernorm(addr(0, "acc"), addr(0, "residual"),
                      layer_addr(layer_idx, 0, "n1w"), layer_addr(layer_idx, 0, "n1b"), seq, h);
        const uint32_t x2 = former_mode_ ? addr(0, "residual") : addr(0, "acc");  // current x for sub-block 2

        // ---- sub-block 2: cross-attn(memory) ----
        if (!former_mode_)
            layernorm(addr(0, "acc"), addr(0, "norm_hidden"),
                      layer_addr(layer_idx, 0, "n2w"), layer_addr(layer_idx, 0, "n2b"), seq, h);
        const uint32_t ca_in = former_mode_ ? x2 : addr(0, "norm_hidden");
        proj(ca_in, lw.cq_w, addr(0, "q"), seq, nq * hd, h, 1, layer_addr(layer_idx, 0, "cq_bias"));
        proj(addr(0, "memory"), lw.ck_w, addr(0, "k"), S, nq * hd, h, 1,
             layer_addr(layer_idx, 0, "ck_bias"));
        proj(addr(0, "memory"), lw.cv_w, addr(0, "v"), S, nq * hd, h, 1,
             layer_addr(layer_idx, 0, "cv_bias"));
        auto& kc_x = cross_kc_[layer_idx];
        auto& vc_x = cross_vc_[layer_idx];
        const KvInsertSegmentPlan cross_kv_plan = resolve_kv_insert_plan(
            kNavdpCrossKvSite, /*position=*/0, S);
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            kc_x, vc_x, addr_offset("k").value,
            addr_offset("v").value, nq, hd, NUM_CORES,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0, /*spm_rows=*/0,
            cross_kv_plan);
        for (int64_t b = 0; b < B; ++b) {   // shared memory K/V (projected once), per-trajectory queries
            const uint32_t o = static_cast<uint32_t>(b) * qrow;
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                kNavdpCrossAttentionDdrSite,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                kNavdpCrossAttentionMemoryPrefixDdrRequired,
                {batch_, predict_size_, memory_len_}, chunk.idx);
            rpu_launch_sdpa_spm_unified_kernel_v2(
                kc_x, vc_x, /*mask_type=*/0 /*none — attend all memory*/, scale,
                addr_offset("q").value + o, addr_offset("sdpa_out").value + o, addr_offset("sdpa_tmp").value, 0,
                PSb, nq, nq, hd, /*kv_seq_len=*/S, NUM_CORES, NUM_CORES);
        }
        proj(addr(0, "sdpa_out"), lw.co_w, addr(0, "oproj"), seq, h, nq * hd, 0, layer_addr(layer_idx, 0, "co_bias"));
        if (former_mode_) {   // POST-LN: x = norm2(x + cross_attn) → "residual"
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                kNavdpFormerCrossAllReduceSite,
                fmb_ring_all_reduce_route_selector(seq, h),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), x2, addr(0, "acc"), seq, h, NUM_CORES, NUM_CORES);
            layernorm(addr(0, "acc"), addr(0, "residual"),
                      layer_addr(layer_idx, 0, "n2w"), layer_addr(layer_idx, 0, "n2b"), seq, h);
        } else {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE, kNavdpCrossAllReduceSite,
                fmb_ring_all_reduce_route_selector(seq, h),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "acc"), addr(0, "residual"), seq, h, NUM_CORES, NUM_CORES);
        }
        // ---- sub-block 3: FFN (GELU/navdp | ReLU/former) ----  (both x now in "residual")
        if (!former_mode_)
            layernorm(addr(0, "residual"), addr(0, "norm_hidden"),
                      layer_addr(layer_idx, 0, "n3w"), layer_addr(layer_idx, 0, "n3b"), seq, h);
        const uint32_t ff_in = former_mode_ ? addr(0, "residual") : addr(0, "norm_hidden");
        proj(ff_in, lw.ff1_w, addr(0, "ff_mid"), seq, FF, h, 1, layer_addr(layer_idx, 0, "ff1_bias"));
        if (former_mode_)   // ReLU = max(x, 0)
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "ff_mid"), c10::Half(0.0), addr(0, "ff_mid"), seq * li, ValuOpType::MAX);
        else
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "ff_mid"), addr(0, "ff_mid"), seq * li,
                ValuOpType::ADD, GeluMode::ERF, NUM_CORES);
        proj(addr(0, "ff_mid"), lw.ff2_w, addr(0, "oproj"), seq, h, FF, 0, layer_addr(layer_idx, 0, "ff2_bias"));
        if (former_mode_) {   // POST-LN: x = norm3(x + ff) → "residual"
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                kNavdpFormerFfAllReduceSite,
                fmb_ring_all_reduce_route_selector(seq, h),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual"), addr(0, "acc"), seq, h, NUM_CORES, NUM_CORES);
            layernorm(addr(0, "acc"), addr(0, "residual"),
                      layer_addr(layer_idx, 0, "n3w"), layer_addr(layer_idx, 0, "n3b"), seq, h);
        } else {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE, kNavdpFfAllReduceSite,
                fmb_ring_all_reduce_route_selector(seq, h),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual"), addr(0, "acc"),
                seq, h, NUM_CORES, NUM_CORES);
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "acc"), c10::Half(0.0), addr(0, "residual"),
                seq * h, ValuOpType::ADD);
        }

        if (!ctx().output_to_spm) emit_layer_output_dma(layer_idx, chunk, "residual");
    }

private:
    KvInsertSegmentPlan resolve_kv_insert_plan(
        int64_t site_id, int64_t position, int64_t logical_rows) {
        const int64_t invocation = ctx().physical_route_invocation;
        const FmbRouteManifestEntry& route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, site_id, invocation);
        const KvInsertSegmentPlan plan = restore_kvinsert_plan(
            site_id, route.arguments, NUM_CORES, num_kv_heads(), head_dim());
        const KvInsertSegmentPlan shape_plan = rpu_resolve_kvinsert_segment_plan(
            position, logical_rows, logical_rows, NUM_CORES,
            num_kv_heads(), head_dim(), kNavdpKvCapabilities,
            plan.route());
        const KvInsertRouteArguments canonical =
            rpu_kvinsert_route_arguments(
                shape_plan, NUM_CORES, num_kv_heads(), head_dim());
        TORCH_CHECK(
            plan.logical_rows() == logical_rows &&
                plan.physical_rows() == logical_rows &&
                plan.segment_count() > 0 &&
                plan.segment(0).position == position &&
                route.arguments.size() == canonical.size() &&
                std::equal(canonical.begin(),
                           canonical.begin() + kKvInsertRouteTopologyWords,
                           route.arguments.begin()),
            "NavDP KV-insert descriptor does not match site ", site_id,
            " launch geometry");
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, site_id,
            static_cast<int64_t>(plan.route()), route.flags,
            route.arguments, invocation);
        return plan;
    }

    // affine LayerNorm: (x-mean)/std * gamma + beta.
    void layernorm(uint32_t src, uint32_t dst, uint32_t gamma, uint32_t beta, int64_t seq, int64_t h) {
        rpu_launch_layernorm_spm_kernel(src, dst, gamma, beta, seq, h, eps_, false, 0, NUM_CORES);
    }
    // linear (partition 1=col, 0=row) + bias.
    void proj(uint32_t in, const at::Tensor& w, uint32_t out, int64_t M, int64_t Nout, int64_t K,
              int64_t partition, uint32_t bias_addr) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, kNavdpProjectionLinearSite,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, ctx().physical_route_invocation);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in, w, out, M, Nout, K, partition, NUM_CORES, bias_addr, at::Tensor());
    }

    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.insert(identity.end(), {
            static_cast<int64_t>(former_mode_),
            static_cast<int64_t>(spm_kv_by_mha_enabled_),
            static_cast<int64_t>(weights_set_),
            static_cast<int64_t>(glue_ready_)});
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.sq_w, &weights.sk_w, &weights.sv_w, &weights.so_w,
                    &weights.sq_b, &weights.sk_b, &weights.sv_b, &weights.so_b,
                    &weights.cq_w, &weights.ck_w, &weights.cv_w, &weights.co_w,
                    &weights.cq_b, &weights.ck_b, &weights.cv_b, &weights.co_b,
                    &weights.ff1_w, &weights.ff1_b, &weights.ff2_w, &weights.ff2_b,
                    &weights.n1_w, &weights.n1_b, &weights.n2_w, &weights.n2_b,
                    &weights.n3_w, &weights.n3_b}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &ie_w_, &ie_b_, &ah_w_, &ah_b_,
                &fln_w_, &fln_b_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<at::Tensor> cross_kc_, cross_vc_;   // model-owned separate cross-attn caches
    double  eps_          = 1e-5;
    int64_t memory_len_   = 34;
    int64_t predict_size_ = 32;
    int64_t action_dim_   = 3;
    uint64_t mem_src_base_ = 0;
    bool mem_loaded_this_forward_ = false;
    // DDPM in-graph unroll (P2/P3). loop_mode_ off → single-step byte-identical.
    bool    loop_mode_ = false;
    int64_t num_steps_ = 1;
    int64_t batch_ = 1;                         // # trajectories batched into one unroll (folded into seq)
    int64_t bps() const { return batch_ * predict_size_; }
    // former_mode_: reuse this decoder as the RGBD former_net perceiver (2-layer TransformerDecoder,
    // POST-LN + ReLU + non-causal self-attn, memory_len=1024). Weights are structurally identical to
    // a NavDP layer. Guarded in build_layer_subgraph; navdp path (false) is the unchanged `else`.
    bool    former_mode_ = false;
    bool    spm_kv_by_mha_enabled_ = true;
    bool    weights_set_ = false;
    bool    glue_ready_ = false;
    ExecutionMode execution_mode_ = ExecutionMode::Unset;
public:
    void configure_cold_routes(bool spm_kv_by_mha) {
        TORCH_CHECK(!weights_set_ && execution_mode_ == ExecutionMode::Unset,
                    "navdp cold routes must be configured before weights or dispatch");
        spm_kv_by_mha_enabled_ = spm_kv_by_mha;
    }

    void set_former_mode(bool f) {
        if (former_mode_ == f) return;
        TORCH_CHECK(!weights_set_ && execution_mode_ == ExecutionMode::Unset,
                    "navdp_set_former_mode: mode must be selected before weights or dispatch");
        former_mode_ = f;
        invalidate_model_state();
    }
private:
    // Glue weights (model-wide, on-device fp16) folded into the unrolled graph.
    at::Tensor ie_w_, ie_b_, ah_w_, ah_b_, fln_w_, fln_b_;   // input_embed / action_head / final-LN
    at::Tensor out_pos_;                                     // [predict_size, H] positional add
    at::Tensor coeff_;                                       // [num_steps, 4] host fp32: A,B,C,D (baked per iter)
    at::Tensor sigz_all_;                                    // [num_steps, predict_size, action_dim] rpu fp16 (sig*z)
    at::Tensor na_stage_;                                    // DDR staging: pre-hook writes input_embed here; body reads
    uint64_t x0_src_base_ = 0, out_dst_base_ = 0;           // mutable: init-noise src, action out dst
    uint64_t mem_all_src_base_ = 0, sigz_src_base_ = 0;    // mutable: per-iter memory row, per-iter sig*z row

public:
    // Denoise glue weights (called once after set_weights). ie/ah/fln + out_pos, all fp16 rpu.
    void set_denoise_glue(at::Tensor ie_w, at::Tensor ie_b, at::Tensor ah_w, at::Tensor ah_b,
                          at::Tensor fln_w, at::Tensor fln_b, at::Tensor out_pos) {
        TORCH_CHECK(weights_set_, "navdp_set_denoise_glue: weights are not installed");
        TORCH_CHECK(!former_mode_, "navdp_set_denoise_glue: Former profile is unsupported");
        TORCH_CHECK(execution_mode_ == ExecutionMode::Unset,
                    "navdp_set_denoise_glue: glue must be installed before dispatch");
        TORCH_CHECK(!glue_ready_, "navdp_set_denoise_glue: glue is already installed");
        check_rpu_half(ie_w, "ie_w", {kHidden, kActionDimPadded});
        check_rpu_half(ie_b, "ie_b", {kHidden});
        check_rpu_half(ah_w, "ah_w", {kActionDimPadded, kHidden});
        check_rpu_half(ah_b, "ah_b", {kActionDimPadded});
        check_rpu_half(fln_w, "fln_w", {kHidden});
        check_rpu_half(fln_b, "fln_b", {kHidden});
        check_rpu_half(out_pos, "out_pos", {kPredictSize, kHidden});
        ie_w_ = ie_w; ie_b_ = ie_b; ah_w_ = ah_w; ah_b_ = ah_b;
        fln_w_ = fln_w; fln_b_ = fln_b; out_pos_ = out_pos;
        glue_ready_ = true;
        invalidate_model_state();
    }

    // In-graph DDPM unroll: x0[predict_size,action_dim] init noise, mem_all[num_steps,memory_len,H]
    // per-step conditioning, sigz[num_steps,predict_size,action_dim], coeff[num_steps,4]=A,B,C,D.
    // Runs the 20-step reverse diffusion for ONE trajectory in a single BUILD+REPLAY; writes the
    // final action into out[predict_size,action_dim].
    at::Tensor denoise_loop_forward(at::Tensor x0, at::Tensor mem_all, at::Tensor sigz,
                                    at::Tensor coeff, std::vector<at::Tensor>& k_caches,
                                    std::vector<at::Tensor>& v_caches, at::Tensor out,
                                    at::IntArrayRef planned_stage_descriptor = {}) {
        TORCH_CHECK(weights_set_, "navdp denoise: weights are not installed");
        TORCH_CHECK(!former_mode_, "navdp denoise: Former profile is unsupported");
        TORCH_CHECK(glue_ready_, "navdp denoise: denoise glue is not installed");
        TORCH_CHECK(execution_mode_ != ExecutionMode::Forward,
                    "navdp denoise: handle is locked to ordinary forward mode");

        TORCH_CHECK(x0.defined() && x0.dim() == 2 &&
                    x0.size(0) > 0 && x0.size(0) % kPredictSize == 0 &&
                    x0.size(1) == kActionDimPadded,
                    "navdp denoise: x0 must have shape [B*32, 16], got ", x0.sizes());
        TORCH_CHECK(x0.scalar_type() == at::kHalf &&
                    x0.device().type() == at::kPrivateUse1 && x0.is_contiguous(),
                    "navdp denoise: x0 must be fp16 contiguous RPU tensor");
        const int64_t B = x0.size(0) / kPredictSize;
        check_rpu_half(mem_all, "mem_all", {kDenoiseSteps, kNavdpMemory, kHidden});
        check_rpu_half(sigz, "sigz", {kDenoiseSteps, B * kPredictSize, kActionDimPadded});
        check_rpu_half(out, "out", {B * kPredictSize, kActionDimPadded});
        TORCH_CHECK(coeff.defined() && coeff.sizes() == at::IntArrayRef({kDenoiseSteps, 4}) &&
                    coeff.scalar_type() == at::kFloat &&
                    coeff.device().type() == at::kCPU && coeff.is_contiguous(),
                    "navdp denoise: coeff must be CPU fp32 contiguous [20, 4]");
        check_caches(k_caches, v_caches, B);

        if (execution_mode_ == ExecutionMode::Unset) {
            loop_mode_ = true;
            num_steps_ = kDenoiseSteps;
            batch_ = B;
            execution_mode_ = ExecutionMode::Denoise;
            // The public request passed the same read-only mode/batch admission
            // before planning; actual forward keeps its latch and safety checks.
            invalidate_model_state(/*planning_domain_changed=*/false);
        } else {
            TORCH_CHECK(loop_mode_ && num_steps_ == kDenoiseSteps && batch_ == B,
                        "navdp denoise: num_steps and batch are fixed after first dispatch; "
                        "expected steps=", num_steps_, " batch=", batch_,
                        ", got steps=", kDenoiseSteps, " batch=", B);
        }
        if (static_cast<int64_t>(cross_kc_.size()) != num_layers()) {
            cross_kc_.clear(); cross_vc_.clear();
            // Cross-attn memory is SHARED across trajectories → keep the cross caches batch-1 even
            // when the self-attn caches are batch-B (narrow(0,0,1) is the whole tensor when B==1).
            for (auto& kc : k_caches) cross_kc_.push_back(at::zeros_like(kc.narrow(0, 0, 1)));
            for (auto& vc : v_caches) cross_vc_.push_back(at::zeros_like(vc.narrow(0, 0, 1)));
        }
        if (!coeff_.defined()) {
            coeff_ = coeff.clone();                   // host fp32: baked per-iter
        } else {
            TORCH_CHECK(at::equal(coeff_, coeff),
                        "navdp denoise: DDPM20 coefficients are fixed after first dispatch");
        }
        sigz_all_ = sigz;                             // rpu fp16 keepalive
        x0_ref_ = x0; mem_all_ref_ = mem_all; out_ref_ = out;   // keepalives
        rpu_ddr_flush_force(x0.data_ptr<c10::Half>());
        rpu_ddr_flush_force(mem_all.data_ptr<c10::Half>());
        rpu_ddr_flush_force(sigz.data_ptr<c10::Half>());
        rpu_ddr_flush_force(out.data_ptr<c10::Half>());
        x0_src_base_      = ::rhino_lkn::RpuGetDevAddr(x0.data_ptr());
        mem_all_src_base_ = ::rhino_lkn::RpuGetDevAddr(mem_all.data_ptr());
        sigz_src_base_    = ::rhino_lkn::RpuGetDevAddr(sigz.data_ptr());
        out_dst_base_     = ::rhino_lkn::RpuGetDevAddr(out.data_ptr());
        if (!na_stage_.defined())   // stable across calls → graph caches
            na_stage_ = at::empty({1, bps(), hidden_size()},
                                  at::TensorOptions().dtype(at::kHalf).device(x0.device()));
        const int64_t packed_rows = bps();
        const std::vector<ChunkInfo> input_chunks{
            {0, 0, packed_rows, packed_rows}};
        std::vector<FmbExecutionSpan> spans;
        spans.reserve(batch_);
        for (int64_t b = 0; b < batch_; ++b) {
            spans.push_back({b * predict_size_, predict_size_, b});
        }
        const FmbStageBoundaryPolicies boundary_policies{};
        if (planned_stage_descriptor.empty()) {
            (void)run_all_layers(
                na_stage_, k_caches, v_caches, std::nullopt,
                /*position=*/0, /*is_causal=*/true, input_chunks, spans,
                boundary_policies);
        } else {
            (void)run_all_layers(
                na_stage_, k_caches, v_caches, std::nullopt,
                /*position=*/0, /*is_causal=*/true,
                /*planned_chunk_size=*/0, planned_stage_descriptor);
        }
        return out;
    }

private:
    at::Tensor x0_ref_, mem_all_ref_, out_ref_;   // keepalives across the synchronous forward
};

}  // namespace v3

// ---- instance registry + C launchers ----
using NavdpRegistry = ModelHandleRegistry<v3::NavdpModel>;

void rpu_navdp_validate_execution_admission(
        int64_t handle, int64_t packed_rows, int64_t denoise_steps) {
    NavdpRegistry::get(handle, "rpu_navdp_validate_execution_admission")
        ->validate_execution_admission(packed_rows, denoise_steps);
}

std::vector<int64_t> rpu_navdp_planner_cache_identity(int64_t handle) {
    return NavdpRegistry::get(handle, "rpu_navdp_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_navdp_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    NavdpRegistry::get(handle, "rpu_navdp_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_navdp_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    NavdpRegistry::get(handle, "rpu_navdp_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_navdp_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return NavdpRegistry::get(handle, "rpu_navdp_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_navdp_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return NavdpRegistry::get(handle, "rpu_navdp_kvinsert_cost_domain")
        ->kvinsert_cost_domain("navdp", descriptor);
}

std::string rpu_navdp_kvinsert_cost_catalog_sha256(int64_t handle) {
    return NavdpRegistry::get(
        handle, "rpu_navdp_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

int64_t rpu_navdp_create(bool allow_numeric_blocked, bool spm_kv_by_mha) {
    TORCH_CHECK(
        allow_numeric_blocked,
        "InternVLA-N1 RPU requires explicit controlled numeric-blocked admission");
    const int64_t handle = NavdpRegistry::create();
    NavdpRegistry::get(handle, "rpu_navdp_create")
        ->configure_cold_routes(spm_kv_by_mha);
    return handle;
}
void    rpu_navdp_destroy(int64_t handle) { NavdpRegistry::destroy(handle, "rpu_navdp_destroy"); }
void    rpu_navdp_set_former_mode(int64_t handle, bool former) {
    NavdpRegistry::get(handle, "rpu_navdp")->set_former_mode(former);
}

void rpu_navdp_set_weights(
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
    int64_t action_dim, double eps)
{
    NavdpRegistry::get(handle, "rpu_navdp")->set_weights(
        sq_w, sk_w, sv_w, so_w, sq_b, sk_b, sv_b, so_b,
        cq_w, ck_w, cv_w, co_w, cq_b, ck_b, cv_b, co_b,
        ff1_w, ff1_b, ff2_w, ff2_b, n1_w, n1_b, n2_w, n2_b, n3_w, n3_b,
        num_heads, head_dim, hidden_size, ff_inter, memory_len, predict_size, action_dim, eps);
}

at::Tensor rpu_navdp_forward(
    int64_t handle, const at::Tensor& tgt, const at::Tensor& memory,
    at::TensorList k_caches_list, at::TensorList v_caches_list,
    const std::optional<at::Tensor>& causal_mask,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return NavdpRegistry::get(handle, "rpu_navdp")->forward(
        tgt, memory, k_caches, v_caches, causal_mask,
        planned_stage_descriptor);
}

std::vector<int64_t> rpu_navdp_resolve_stage_domain(
    int64_t handle, int64_t packed_rows, int64_t denoise_steps) {
    return NavdpRegistry::get(
               handle, "rpu_navdp_resolve_stage_domain")
        ->resolve_stage_domain(packed_rows, denoise_steps);
}

int64_t rpu_navdp_get_resolved_chunk_size(int64_t handle) {
    return NavdpRegistry::get(
               handle, "rpu_navdp_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

void rpu_navdp_set_denoise_glue(
    int64_t handle, at::Tensor ie_w, at::Tensor ie_b, at::Tensor ah_w, at::Tensor ah_b,
    at::Tensor fln_w, at::Tensor fln_b, at::Tensor out_pos)
{
    NavdpRegistry::get(handle, "rpu_navdp")->set_denoise_glue(ie_w, ie_b, ah_w, ah_b, fln_w, fln_b, out_pos);
}

at::Tensor rpu_navdp_denoise_loop_forward(
    int64_t handle, const at::Tensor& x0, const at::Tensor& mem_all, const at::Tensor& sigz,
    const at::Tensor& coeff, at::TensorList k_caches_list,
    at::TensorList v_caches_list, const at::Tensor& out,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return NavdpRegistry::get(handle, "rpu_navdp")->denoise_loop_forward(
        x0, mem_all, sigz, coeff, k_caches, v_caches,
        const_cast<at::Tensor&>(out), planned_stage_descriptor);
}
