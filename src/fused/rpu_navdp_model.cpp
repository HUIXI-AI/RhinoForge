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
// set_weights ends with invalidate_model_state().

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_helpers.h"
#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace v3 {

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
        invalidate_model_state();  // Must remain last.
    }

    // memory[memory_len, hidden] held per-forward (cross-attn K/V source, broadcast to SPM once).
    at::Tensor forward(
        const at::Tensor& tgt,               // [seq_q=predict_size, hidden]  (already input_embed'd host-side)
        const at::Tensor& memory,            // [memory_len, hidden]
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& causal_mask)
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
        // Model-owned cross-attention caches remain separate from self-attention caches.
        if (static_cast<int64_t>(cross_kc_.size()) != num_layers()) {
            cross_kc_.clear(); cross_vc_.clear();
            // Cross-attn memory is SHARED across trajectories → keep the cross caches batch-1 even
            // when the self-attn caches are batch-B (narrow(0,0,1) is the whole tensor when B==1).
            for (auto& kc : k_caches) cross_kc_.push_back(at::zeros_like(kc.narrow(0, 0, 1)));
            for (auto& vc : v_caches) cross_vc_.push_back(at::zeros_like(vc.narrow(0, 0, 1)));
        }
        // Device DDR address (NOT the raw data_ptr) — the mutable broadcast's live_src_base must be
        // the RPU device address (no internal conversion, unlike the fixed variant).
        // Mirrors gr00t vl_src_base_.
        mem_src_base_ = ::rhino_lkn::RpuGetDevAddr(memory.data_ptr<c10::Half>());
        mem_loaded_this_forward_ = false;
        auto out = run_all_layers(tgt, k_caches, v_caches, std::nullopt, /*position=*/0, /*is_causal=*/true);
        return out;
    }

protected:
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

    ModelDynamicConfig dynamic_config(const ChunkPlan&) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;  // 32x384 fp16 ~24KB
        return cfg;
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
        const int64_t tmp = A(std::max(tmp_self, tmp_cross) * 32);

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
            auto add_pre = [&](const char* nm, int64_t nelem, at::Tensor NavdpModel::* f) {
                BufferDecl d; d.name = nm; d.size = A(nelem*DWIDTH);
                d.storage = StorageClass::Persistent; d.scope = ALL;
                d.preload_callback = [this, f, nelem](FusedModelBase&, int, uint32_t a0) {
                    rpu_launch_ddr_broadcast_spm_dma((this->*f).data_ptr<c10::Half>(), nelem, a0, NUM_CORES);
                };
                v.push_back(d);
            };
            add_pre("ie_b_spm",   h,    &NavdpModel::ie_b_);
            add_pre("ah_b_spm",   AD,   &NavdpModel::ah_b_);
            add_pre("flnw_spm",   h,    &NavdpModel::fln_w_);
            add_pre("flnb_spm",   h,    &NavdpModel::fln_b_);
            add_pre("outpos_spm", predict_size_*h, &NavdpModel::out_pos_);  // [predict_size,h]; added per-traj block
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
            rpu_launch_ddr_broadcast_spm_dma_mutable(&x0_src_base_, 0, PS*AD, addr(0,"na_spm"), NUM_CORES);
        }
        // [A2] input_embed: na_spm[PS,AD] → ae_spm[PS,H] (single-core, K=AD) + bias; add out_pos.
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0,"na_spm"), ie_w_, addr(0,"ae_spm"),
            PS, h, AD, /*partition=*/1, /*num_cores=*/1, addr(0,"ie_b_spm"), false, at::Tensor());
        for (int64_t b = 0; b < batch_; ++b) {   // out_pos [PSc,h] added to each trajectory's ae block
            const uint32_t ob = static_cast<uint32_t>(b * PSc * h * DWIDTH);
            rpu_launch_eltwise_binary_spm_kernel(addr(0,"ae_spm")+ob, addr(0,"outpos_spm"), addr(0,"ae_spm")+ob,
                PSc*h, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
        }
        // [A3] memory row i → "memory" SPM (mutable, per-iter offset).
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
        rpu_launch_linear_spm_to_spm_acc16_kernel(addr(0,"ln_spm"), ah_w_, addr(0,"eps_spm"),
            PS, AD, h, /*partition=*/1, /*num_cores=*/1, addr(0,"ah_b_spm"), false, at::Tensor());
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
            rpu_launch_ddr_broadcast_spm_dma_mutable(&sigz_src_base_, i*PS*AD*DWIDTH, PS*AD, addr(0,"sigz_spm"), NUM_CORES);
            rpu_launch_eltwise_binary_spm_kernel(addr(0,"na_spm"), addr(0,"sigz_spm"), addr(0,"na_spm"),
                PS*AD, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
        }
        // [Z4] last iter: na_spm → out DDR.
        if (i == num_steps_ - 1) {
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

        // Broadcast cross-attention memory once per forward. The mutable source
        // base is rebound from the device address on graph replay.
        if (layer_idx == 0 && chunk.idx == 0 && !mem_loaded_this_forward_) {
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
        for (int64_t b = 0; b < B; ++b) {   // block-diagonal self-attn: each trajectory's PSb tokens
            auto kcb = kc.narrow(0, b, 1);
            auto vcb = vc.narrow(0, b, 1);
            const uint32_t o = static_cast<uint32_t>(b) * qrow;
            rpu_launch_insert_kcache_spm_unified(kcb, 0, addr_offset("k").value + o, PSb, nq, hd, NUM_CORES);
            rpu_launch_insert_vcache_spm_unified(vcb, 0, addr_offset("v").value + o, PSb, nq, hd, NUM_CORES);
            rpu_launch_sdpa_spm_unified_kernel_v2(
                kcb, vcb, self_mask, scale,
                addr_offset("q").value + o, addr_offset("sdpa_out").value + o, addr_offset("sdpa_tmp").value, 0,
                PSb, nq, nq, hd, /*kv_seq_len=*/PSb, NUM_CORES, NUM_CORES);
        }
        proj(addr(0, "sdpa_out"), lw.so_w, addr(0, "oproj"), seq, h, nq * hd, 0, layer_addr(layer_idx, 0, "so_bias"));
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
        rpu_launch_insert_kcache_spm_unified(kc_x, 0, addr_offset("k").value, S, nq, hd, NUM_CORES);
        rpu_launch_insert_vcache_spm_unified(vc_x, 0, addr_offset("v").value, S, nq, hd, NUM_CORES);
        for (int64_t b = 0; b < B; ++b) {   // shared memory K/V (projected once), per-trajectory queries
            const uint32_t o = static_cast<uint32_t>(b) * qrow;
            rpu_launch_sdpa_spm_unified_kernel_v2(
                kc_x, vc_x, /*mask_type=*/0 /*none — attend all memory*/, scale,
                addr_offset("q").value + o, addr_offset("sdpa_out").value + o, addr_offset("sdpa_tmp").value, 0,
                PSb, nq, nq, hd, /*kv_seq_len=*/S, NUM_CORES, NUM_CORES);
        }
        proj(addr(0, "sdpa_out"), lw.co_w, addr(0, "oproj"), seq, h, nq * hd, 0, layer_addr(layer_idx, 0, "co_bias"));
        if (former_mode_) {   // POST-LN: x = norm2(x + cross_attn) → "residual"
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), x2, addr(0, "acc"), seq, h, NUM_CORES, NUM_CORES);
            layernorm(addr(0, "acc"), addr(0, "residual"),
                      layer_addr(layer_idx, 0, "n2w"), layer_addr(layer_idx, 0, "n2b"), seq, h);
        } else {
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
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual"), addr(0, "acc"), seq, h, NUM_CORES, NUM_CORES);
            layernorm(addr(0, "acc"), addr(0, "residual"),
                      layer_addr(layer_idx, 0, "n3w"), layer_addr(layer_idx, 0, "n3b"), seq, h);
        } else {
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual"), addr(0, "residual"), seq, h, NUM_CORES, NUM_CORES);
        }

        if (!ctx().output_to_spm) emit_layer_output_dma(layer_idx, chunk, "residual");
    }

private:
    // affine LayerNorm: (x-mean)/std * gamma + beta.
    void layernorm(uint32_t src, uint32_t dst, uint32_t gamma, uint32_t beta, int64_t seq, int64_t h) {
        rpu_launch_layernorm_spm_kernel(src, dst, gamma, beta, seq, h, eps_, false, 0, NUM_CORES);
    }
    // linear (partition 1=col, 0=row) + bias.
    void proj(uint32_t in, const at::Tensor& w, uint32_t out, int64_t M, int64_t Nout, int64_t K,
              int64_t partition, uint32_t bias_addr) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in, w, out, M, Nout, K, partition, NUM_CORES, bias_addr, false, at::Tensor());
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<at::Tensor> cross_kc_, cross_vc_;   // model-owned separate cross-attn caches
    double  eps_          = 1e-5;
    int64_t memory_len_   = 34;
    int64_t predict_size_ = 32;
    int64_t action_dim_   = 3;
    uint64_t mem_src_base_ = 0;
    bool mem_loaded_this_forward_ = false;
    // DDPM in-graph unroll. loop_mode_ off → single-step byte-identical.
    bool    loop_mode_ = false;
    int64_t num_steps_ = 1;
    int64_t batch_ = 1;                         // # trajectories batched into one unroll (folded into seq)
    int64_t bps() const { return batch_ * predict_size_; }
    // former_mode_: reuse this decoder as the RGBD former_net perceiver (2-layer TransformerDecoder,
    // POST-LN + ReLU + non-causal self-attn, memory_len=1024). Weights are structurally identical to
    // a NavDP layer. Guarded in build_layer_subgraph; navdp path (false) is the unchanged `else`.
    bool    former_mode_ = false;
    bool    weights_set_ = false;
    bool    glue_ready_ = false;
    ExecutionMode execution_mode_ = ExecutionMode::Unset;
public:
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
                                    std::vector<at::Tensor>& v_caches, at::Tensor out) {
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
            invalidate_model_state();
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
        (void) run_all_layers(na_stage_, k_caches, v_caches, std::nullopt, /*position=*/0, /*is_causal=*/true);
        return out;
    }

private:
    at::Tensor x0_ref_, mem_all_ref_, out_ref_;   // keepalives across the synchronous forward
};

}  // namespace v3

// ---- instance registry + C launchers ----
using NavdpRegistry = ModelHandleRegistry<v3::NavdpModel>;

int64_t rpu_navdp_create() {
    const auto exact_one = [](const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && value[0] == '1' && value[1] == '\0';
    };
    const auto unset_or_zero = [](const char* name) {
        const char* value = std::getenv(name);
        return value == nullptr || (value[0] == '0' && value[1] == '\0');
    };
    const bool exact_twostage = exact_one("RPU_INTERNVLA_N1_EXACT_TWOSTAGE");
    const bool legacy_single_stage =
        exact_one("RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED");
    TORCH_CHECK(
        exact_twostage || unset_or_zero("RPU_INTERNVLA_N1_EXACT_TWOSTAGE"),
        "RPU_INTERNVLA_N1_EXACT_TWOSTAGE accepts only literal 0 or 1");
    TORCH_CHECK(
        legacy_single_stage ||
            unset_or_zero("RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED"),
        "RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED accepts only literal 0 or 1");
    TORCH_CHECK(
        exact_twostage || legacy_single_stage,
        "InternVLA-N1 RPU requires exact RPU_INTERNVLA_N1_EXACT_TWOSTAGE=1 "
        "or the legacy single-stage controlled-evaluation selector");
    TORCH_CHECK(
        exact_twostage
            ? exact_one("RPU_ALLREDUCE_TWOSTAGE")
            : unset_or_zero("RPU_ALLREDUCE_TWOSTAGE"),
        "InternVLA-N1 exact profile requires RPU_ALLREDUCE_TWOSTAGE=1; "
        "legacy single-stage evaluation requires it unset or 0");
    TORCH_CHECK(unset_or_zero("RPU_ALLREDUCE_RING") &&
                    unset_or_zero("RPU_ALLREDUCE_CHUNK_V2"),
                "InternVLA-N1 requires ring and chunk-v2 all-reduce unset or 0");
    return NavdpRegistry::create();
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
    const std::optional<at::Tensor>& causal_mask)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return NavdpRegistry::get(handle, "rpu_navdp")->forward(tgt, memory, k_caches, v_caches, causal_mask);
}

void rpu_navdp_set_denoise_glue(
    int64_t handle, at::Tensor ie_w, at::Tensor ie_b, at::Tensor ah_w, at::Tensor ah_b,
    at::Tensor fln_w, at::Tensor fln_b, at::Tensor out_pos)
{
    NavdpRegistry::get(handle, "rpu_navdp")->set_denoise_glue(ie_w, ie_b, ah_w, ah_b, fln_w, fln_b, out_pos);
}

at::Tensor rpu_navdp_denoise_loop_forward(
    int64_t handle, const at::Tensor& x0, const at::Tensor& mem_all, const at::Tensor& sigz,
    const at::Tensor& coeff, at::TensorList k_caches_list, at::TensorList v_caches_list, const at::Tensor& out)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return NavdpRegistry::get(handle, "rpu_navdp")->denoise_loop_forward(
        x0, mem_all, sigz, coeff, k_caches, v_caches, const_cast<at::Tensor&>(out));
}
