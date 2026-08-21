// rpu_gr00t_vl_encoder_model.cpp — GR00T-N1.7 "全程 RPU" Component A
//                                   (v3 FusedModelBase subsystem)
//
// Fuses the host-fp32 `vlln + vl_self_attention` glue (runtime.py::_vl_embeds) on-device:
//
//   u = raw ⊙ inv_w_rms                         (per-channel scale; recovers hidden/rms(hidden))
//   x = LayerNorm(u, vlln.w, vlln.b, eps=1e-5)   (scale-invariance identity)
//   4× SelfAttentionTransformer block (plain LN, NO AdaLN, plain residuals):
//     n1 = LN(x, norm1)        ; q/k/v = linear(n1)+bias (col)
//     KV-insert@0 + bidirectional SDPA (MASK_NONE, scale 1/√64, nh=32, hd=64)
//     x += linear(attn, to_out)+bias (row)
//     n3 = LN(x, norm3)        ; ff = GELU(linear(n3, ff0)+bias) ; x += linear(ff, ff2)+bias (row)
//
// Structural twin of the DiT self-attn block (rpu_gr00t_dit_model.cpp) minus AdaLN/cross, and
// of the SigLIP encoder (rpu_siglip_model.cpp: KV-insert + SDPA MASK_NONE + GELU FFN + plain LN).
// GELU uses the reference's tanh approximation. Single chunk is REQUIRED so the bidirectional SDPA
// attends all S query positions (chunking would split attention) — enforced by the
// TORCH_CHECK(plan.num_chunks == 1) in dynamic_config. NOTE: set_chunk_size_override(0) does not
// provide it; 0 means AUTO.
//
// Fresh per-forward output via spm_copy_ddr_dma → allocate_tracked_output.
// set_weights ends with invalidate_model_state().
//
#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"
#include "rpu_gr00t_spm_z2.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

// W8A16 per-projection scale validation. Same four checks as the already-strong
// sites in rpu_siglip_model.cpp and rpu_pi05_denoise_step_model.cpp: list
// length, int8 weight dtype, 1D contiguous fp16 RPU scale, and
// scale.numel() == output channels. Checking only the list length is not enough —
// a wrong-length or wrong-dtype scale reaches PARALLEL_LINEAR_W8A16 as garbage
// per-channel multipliers, i.e. a silent numeric corruption rather than a crash.
// The `else` leg is the reverse hole: an int8 weight with no scale list is read
// back as fp16 by the kernel.
static void check_gr00t_vl_w8a16_scale_list(int64_t n_layers,
                                            const at::TensorList& weights,
                                            const at::TensorList& scales,
                                            const char* name) {
    const char* ctx = "gr00t_vl_encoder_set_weights";
    const bool quantized = !scales.empty();
    if (quantized) {
        TORCH_CHECK(static_cast<int64_t>(scales.size()) == n_layers,
                    ctx, ": ", name, "_scale.size()=", scales.size(),
                    " != num_layers=", n_layers);
    }
    for (int64_t i = 0; i < n_layers; ++i) {
        const at::Tensor& w = weights[i];
        if (quantized) {
            const at::Tensor& s = scales[i];
            TORCH_CHECK(w.scalar_type() == at::kChar,
                        ctx, ": W8A16 mode requires int8 ", name, "[", i,
                        "], got ", w.scalar_type());
            TORCH_CHECK(s.defined() && s.dim() == 1
                        && s.scalar_type() == at::kHalf
                        && s.device().type() == at::kPrivateUse1
                        && s.is_contiguous(),
                        ctx, ": ", name, "_scale[", i,
                        "] must be 1D contiguous fp16 RPU tensor");
            TORCH_CHECK(s.numel() == w.size(0),
                        ctx, ": ", name, "_scale[", i, "].numel()=", s.numel(),
                        " != output dim=", w.size(0));
        } else {
            TORCH_CHECK(w.scalar_type() != at::kChar,
                        ctx, ": int8 ", name, "[", i,
                        "] requires a non-empty scale list for this projection");
        }
    }
}

namespace v3 {

// =============================================================================
// Gr00tVlEncoderModel — v3::FusedModelBase subclass (GR00T vlln + vl_self_attn)
// =============================================================================
class Gr00tVlEncoderModel : public FusedModelBase {
public:
    // Per-block weights. All fp16; linears swizzled on the Python side
    // (q/k/v/ff0 col, to_out/ff2 row). norm gammas/betas are per-channel [hidden].
    struct LayerWeights {
        at::Tensor norm1_w, norm1_b;                   // LN1 affine
        at::Tensor to_q_w, to_k_w, to_v_w, to_out_w;   // attention projections
        at::Tensor to_q_b, to_k_b, to_v_b, to_out_b;   // attention biases
        at::Tensor norm3_w, norm3_b;                   // LN3 affine
        at::Tensor ff0_w, ff2_w;                       // FFN 2048->8192->2048
        at::Tensor ff0_b, ff2_b;                       // FFN biases
        // W8A16: per-output-channel fp16 scales for the 6 quantized GEMMs (undefined = fp16).
        at::Tensor to_q_ws, to_k_ws, to_v_ws, to_out_ws, ff0_ws, ff2_ws;
    };

    Gr00tVlEncoderModel() = default;

    void set_weights(
        at::TensorList norm1_w_list, at::TensorList norm1_b_list,
        at::TensorList to_q_list, at::TensorList to_q_b_list,
        at::TensorList to_k_list, at::TensorList to_k_b_list,
        at::TensorList to_v_list, at::TensorList to_v_b_list,
        at::TensorList to_out_list, at::TensorList to_out_b_list,
        at::TensorList norm3_w_list, at::TensorList norm3_b_list,
        at::TensorList ff0_list, at::TensorList ff0_b_list,
        at::TensorList ff2_list, at::TensorList ff2_b_list,
        const at::Tensor& vlln_w, const at::Tensor& vlln_b, const at::Tensor& inv_w_rms,
        int64_t num_heads, int64_t head_dim,
        int64_t hidden_size, int64_t ff_inter, double eps,
        at::TensorList to_q_scale_list = {}, at::TensorList to_k_scale_list = {},
        at::TensorList to_v_scale_list = {}, at::TensorList to_out_scale_list = {},
        at::TensorList ff0_scale_list = {}, at::TensorList ff2_scale_list = {})
    {
        int64_t N = static_cast<int64_t>(to_q_list.size());
        TORCH_CHECK(N > 0, "gr00t_vl_encoder_set_weights: empty weight lists");
        TORCH_CHECK(num_heads > 0 && head_dim > 0 && hidden_size > 0 && ff_inter > 0,
                    "gr00t_vl_encoder_set_weights: dim params must be positive");
        TORCH_CHECK(num_heads * head_dim == hidden_size,
                    "gr00t_vl_encoder_set_weights: num_heads*head_dim (", num_heads * head_dim,
                    ") != hidden_size (", hidden_size, ")");

        auto check_list = [&](const at::TensorList& l, const char* n, int64_t rank) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "gr00t_vl_encoder_set_weights: ", n, ".size()=", l.size(),
                        " != num_layers=", N);
            for (int64_t i = 0; i < N; ++i) {
                TORCH_CHECK(l[i].defined() && l[i].dim() == rank
                            && l[i].device().type() == at::kPrivateUse1
                            && l[i].is_contiguous(),
                            "gr00t_vl_encoder_set_weights: ", n, "[", i,
                            "] must be ", rank, "D contiguous RPU tensor");
            }
        };
        check_list(norm1_w_list, "norm1_w", 1); check_list(norm1_b_list, "norm1_b", 1);
        check_list(to_q_list,    "to_q",    2); check_list(to_q_b_list,  "to_q_b",  1);
        check_list(to_k_list,    "to_k",    2); check_list(to_k_b_list,  "to_k_b",  1);
        check_list(to_v_list,    "to_v",    2); check_list(to_v_b_list,  "to_v_b",  1);
        check_list(to_out_list,  "to_out",  2); check_list(to_out_b_list,"to_out_b",1);
        check_list(norm3_w_list, "norm3_w", 1); check_list(norm3_b_list, "norm3_b", 1);
        check_list(ff0_list,     "ff0",     2); check_list(ff0_b_list,   "ff0_b",   1);
        check_list(ff2_list,     "ff2",     2); check_list(ff2_b_list,   "ff2_b",   1);
        auto check_vec = [&](const at::Tensor& t, const char* n) {
            TORCH_CHECK(t.defined() && t.dim() == 1 && t.size(0) == hidden_size
                        && t.device().type() == at::kPrivateUse1 && t.is_contiguous(),
                        "gr00t_vl_encoder_set_weights: ", n, " must be 1-D [hidden] contiguous RPU tensor");
        };
        check_vec(vlln_w, "vlln_w"); check_vec(vlln_b, "vlln_b"); check_vec(inv_w_rms, "inv_w_rms");

        // No GQA; reuse base slots (intermediate_size = ff_inter).
        set_model_params(num_heads, num_heads, head_dim, hidden_size, ff_inter);
        set_num_layers(N);
        eps_ = eps;

        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; ++i) {
            layer_weights_.push_back({
                norm1_w_list[i], norm1_b_list[i],
                to_q_list[i], to_k_list[i], to_v_list[i], to_out_list[i],
                to_q_b_list[i], to_k_b_list[i], to_v_b_list[i], to_out_b_list[i],
                norm3_w_list[i], norm3_b_list[i],
                ff0_list[i], ff2_list[i], ff0_b_list[i], ff2_b_list[i],
            });
        }
        vlln_w_ = vlln_w; vlln_b_ = vlln_b; inv_w_rms_ = inv_w_rms;

        // W8A16: attach per-output-channel scales for the 6 quantized GEMMs (empty = fp16).
        check_gr00t_vl_w8a16_scale_list(N, to_q_list,   to_q_scale_list,   "to_q");
        check_gr00t_vl_w8a16_scale_list(N, to_k_list,   to_k_scale_list,   "to_k");
        check_gr00t_vl_w8a16_scale_list(N, to_v_list,   to_v_scale_list,   "to_v");
        check_gr00t_vl_w8a16_scale_list(N, to_out_list, to_out_scale_list, "to_out");
        check_gr00t_vl_w8a16_scale_list(N, ff0_list,    ff0_scale_list,    "ff0");
        check_gr00t_vl_w8a16_scale_list(N, ff2_list,    ff2_scale_list,    "ff2");
        if (!to_q_scale_list.empty()) {
            for (int64_t i = 0; i < N; ++i) {
                layer_weights_[i].to_q_ws   = to_q_scale_list[i];
                layer_weights_[i].to_k_ws   = to_k_scale_list[i];
                layer_weights_[i].to_v_ws   = to_v_scale_list[i];
                layer_weights_[i].to_out_ws = to_out_scale_list[i];
                layer_weights_[i].ff0_ws    = ff0_scale_list[i];
                layer_weights_[i].ff2_ws    = ff2_scale_list[i];
            }
        }

        invalidate_model_state();  // Last non-empty statement of set_weights.
    }

    // forward — raw [B,S,hidden] post-RMSNorm backbone output -> vl_embeds [B,S,hidden].
    at::Tensor forward(
        const at::Tensor& raw,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches)
    {
        RECORD_FUNCTION("gr00t_vl_encoder_forward", {});
        TORCH_CHECK(!z2_bound_,
                    "gr00t_vl_encoder_forward: Z2 lease is active; use the Z2 forward op");
        TORCH_CHECK(raw.defined() && raw.dim() == 3 && raw.size(-1) == hidden_size()
                    && raw.device().type() == at::kPrivateUse1 && raw.is_contiguous(),
                    "gr00t_vl_encoder_forward: raw must be [B,S,hidden] contiguous RPU tensor");

        // This path requires a single chunk. set_chunk_size_override(0) means
        // AUTO, i.e. let the
        // planner pick. Single-chunk is what the emission below REQUIRES, not
        // what this line arranges, and nothing checked it. The requirement is
        // enforced in dynamic_config(); this line just declines to pin a chunk
        // and lets the SPM planner choose (PLAN A.2).
        set_chunk_size_override(0);

        return run_all_layers(
            raw, k_caches, v_caches, /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false);
    }

    SpmPipelineComponentLayout prepare_z2_layout(int64_t seq_len) {
        TORCH_CHECK(seq_len > 0,
                    "gr00t_vl_encoder Z2: seq_len must be positive");
        TORCH_CHECK(!z2_bound_,
                    "gr00t_vl_encoder Z2: cannot prepare while a lease is active");
        LayoutContext layout;
        layout.chunk_size = seq_len;
        layout.max_kv_seq_len = seq_len;
        layout.num_layers = num_layers();
        layout.use_attn_mask = false;
        layout.is_causal = false;
        auto prepared = prepare_spm_pipeline_component(layout);
        z2_prepared_seq_len_ = seq_len;
        return prepared;
    }

    SpmDense2DSpec z2_port_spec(int64_t seq_len) const {
        TORCH_CHECK(seq_len > 0,
                    "gr00t_vl_encoder Z2: port seq_len must be positive");
        SpmDense2DSpec spec;
        spec.rows = seq_len;
        spec.cols = hidden_size();
        spec.validate();
        return spec;
    }

    void adopt_z2_layout(const SpmPipelineLease& lease,
                         const SpmTensorView& scratch) {
        TORCH_CHECK(z2_prepared_seq_len_ > 0,
                    "gr00t_vl_encoder Z2: prepare must precede begin");
        TORCH_CHECK(!z2_bound_,
                    "gr00t_vl_encoder Z2: binding is already active");
        adopt_spm_pipeline_component(lease, scratch);
    }

    void bind_z2_port(const SpmPipelineLease& lease,
                      const SpmPortView& port) {
        const auto expected = z2_port_spec(z2_prepared_seq_len_);
        TORCH_CHECK(port.spec() == expected,
                    "gr00t_vl_encoder Z2: typed port spec mismatch");
        z2_port_addr_ = port.resolve_physical_addr(/*core=*/0, lease);
        z2_epoch_ = lease.epoch();
        z2_plan_hash_ = lease.plan_hash();
        z2_bound_ = true;
    }

    void validate_z2_layout(const SpmPipelineLease& lease) const {
        TORCH_CHECK(z2_bound_,
                    "gr00t_vl_encoder Z2: no active binding");
        TORCH_CHECK(z2_epoch_ == lease.epoch() &&
                        z2_plan_hash_ == lease.plan_hash(),
                    "gr00t_vl_encoder Z2: stale lease binding");
        validate_spm_pipeline_component(lease);
    }

    void clear_z2_layout(uint64_t epoch, uint64_t plan_hash) {
        if (z2_bound_) {
            TORCH_CHECK(z2_epoch_ == epoch && z2_plan_hash_ == plan_hash,
                        "gr00t_vl_encoder Z2: stale clear token");
        }
        release_spm_pipeline_component(epoch, plan_hash);
        z2_bound_ = false;
        z2_port_addr_ = 0;
        z2_epoch_ = 0;
        z2_plan_hash_ = 0;
    }

    void unprepare_z2_layout(int64_t seq_len) {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "gr00t VL Z2 unprepare must run outside Graph capture");
        TORCH_CHECK(!z2_bound_ && z2_epoch_ == 0 &&
                        z2_plan_hash_ == 0 && z2_port_addr_ == 0,
                    "gr00t VL Z2 unprepare requires no active component lease");
        TORCH_CHECK(z2_prepared_seq_len_ == 0 ||
                        z2_prepared_seq_len_ == seq_len,
                    "gr00t VL Z2 unprepare received a stale sequence identity");
        // Keep the FMB cached layout, persistent allocations, and weights.
        // Only the coordinator-owned authority is being retired.
        z2_prepared_seq_len_ = 0;
    }

    void check_z2_destroy_allowed() const {
        TORCH_CHECK(!z2_bound_ && z2_prepared_seq_len_ == 0 &&
                        z2_epoch_ == 0 && z2_plan_hash_ == 0 &&
                        z2_port_addr_ == 0,
                    "cannot destroy the GR00T VL handle while its Z2 layout "
                    "is prepared or active; clear the outer GraphCache and "
                    "unprepare Z2 first");
    }

    void forward_z2(
        const at::Tensor& raw,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        uint64_t epoch,
        uint64_t plan_hash)
    {
        RECORD_FUNCTION("gr00t_vl_encoder_forward_z2", {});
        TORCH_CHECK(z2_bound_ && z2_epoch_ == epoch &&
                        z2_plan_hash_ == plan_hash,
                    "gr00t_vl_encoder_forward_z2: stale epoch/plan hash");
        TORCH_CHECK(raw.defined() && raw.dim() == 3 &&
                        raw.size(0) == 1 &&
                        raw.size(-2) == z2_prepared_seq_len_ &&
                        raw.size(-1) == hidden_size() &&
                        raw.scalar_type() == at::kHalf &&
                        raw.device().type() == at::kPrivateUse1 &&
                        raw.is_contiguous(),
                    "gr00t_vl_encoder_forward_z2: raw must match prepared [1,S,H] fp16 RPU layout");
        set_chunk_size_override(0);
        (void)run_all_layers(
            raw, k_caches, v_caches, /*mask=*/std::nullopt,
            /*position=*/0, /*is_causal=*/false);
    }

    void stage_z2_outer_fast_component(
        GraphKernelRegisterCensusGuard& guard,
        at::TensorList k_caches,
        at::TensorList v_caches) const {
        TORCH_CHECK(z2_bound_ && z2_epoch_ != 0 && z2_plan_hash_ != 0,
                    "gr00t VL Z2 outer-fast component is not bound");
        stage_spm_outer_fast_component(
            guard, k_caches, v_caches);
    }

    void bind_z2_outer_fast_input(
        GraphKernelRegisterCensusGuard& guard,
        const at::Tensor& raw) {
        TORCH_CHECK(raw.defined() && raw.dim() == 3 &&
                        raw.size(0) == 1 &&
                        raw.size(1) == z2_prepared_seq_len_ &&
                        raw.size(2) == hidden_size(),
                    "gr00t VL Z2 outer-fast raw shape drifted");
        bind_spm_outer_fast_input(guard, raw);
    }

protected:
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers             = num_layers();
        cfg.cross_layer_batch_size = num_layers();  // single group (SigLIP discipline)
        cfg.body_iterations        = 1;
        return cfg;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        // Hard-wired single-chunk, same as the three sibling towers.
        // build_layer_subgraph inserts K/V at a LITERAL position 0 (:370-372,
        // `insert_kcache(k_cache, 0, ...)`) and runs SDPA with a CHUNK-LOCAL
        // kv_seq_len (:378, `/*kv_seq_len=*/seq_len`). A >1-chunk plan would
        // therefore have chunk n overwrite chunk 0's K/V rows and each chunk
        // would attend only itself — bidirectional attention silently degrading
        // to per-chunk self-attention, with the right shapes and wrong numbers.
        //
        // The other three towers already guard this (rpu_dinov3_vision_model.cpp
        // and rpu_qwen25vl_vision_model.cpp; the qwen3_vl / qwen3_5 towers
        // instead thread chunk.offset through KV-insert and so need no guard).
        // This hook must inspect the plan rather than accepting it implicitly.
        TORCH_CHECK(plan.num_chunks == 1,
            "gr00t_vl_encoder requires a single chunk (got ", plan.num_chunks,
            ", chunk_size=", plan.chunk_size,
            "); multi-chunk would silently break cross-patch attention");
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;   // residual stream stays in SPM
        return cfg;
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        const int64_t cs = ctx.chunk_size;
        const int64_t h  = hidden_size();          // 2048
        const int64_t nq = num_q_heads();          // 32
        const int64_t hd = head_dim();             // 64
        const int64_t FF = intermediate_size();    // 8192
        const int64_t local_qhd   = (nq * hd) / NUM_CORES;   // 256
        const int64_t local_inter = FF / NUM_CORES;          // 1024
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        const int64_t res    = A(cs * h * DWIDTH);
        const int64_t qbuf   = A(cs * local_qhd * DWIDTH);
        const int64_t outbuf = A(cs * local_qhd * DWIDTH);
        const int64_t mlp    = A(cs * local_inter * DWIDTH);
        const int64_t ln_sz  = A(h * DWIDTH);

        int64_t tmp_self = sdpa_compute_tmp_v16_size(make_sdpa_config(0), cs);
        const int64_t tmp = A(tmp_self * 32);

        constexpr BufferScope ALL = BufferScope::LayerWide;
        std::vector<BufferDecl> v = {
            {"residual1",   res,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"residual2",   res,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"norm_hidden", res,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"q",           qbuf,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"k",           qbuf,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"v",           qbuf,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_out",    outbuf, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"oproj",       res,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_tmp",    tmp,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"ff_mid",      mlp,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
        };

        // ---- global Persistent: vlln gamma/beta + inv_w_rms (broadcast [hidden]) ----
        auto bcast_const = [&](const char* name, at::Tensor Gr00tVlEncoderModel::* field) {
            BufferDecl d; d.name = name; d.size = ln_sz;
            d.storage = StorageClass::Persistent; d.scope = ALL;
            d.preload_callback = [this, field](FusedModelBase&, int, uint32_t core0_addr) {
                rpu_launch_ddr_broadcast_spm_dma(
                    this->*field, /*src_offset_elements=*/0,
                    hidden_size(), core0_addr);
            };
            v.push_back(d);
        };
        bcast_const("vlln_w",    &Gr00tVlEncoderModel::vlln_w_);
        bcast_const("vlln_b",    &Gr00tVlEncoderModel::vlln_b_);
        bcast_const("inv_w_rms", &Gr00tVlEncoderModel::inv_w_rms_);

        // ---- per-layer Persistent: norm1/norm3 gamma/beta (broadcast [hidden]) ----
        const int nl = static_cast<int>(num_layers());
        auto ln_perlayer = [&](const char* name, at::Tensor LayerWeights::* field) {
            BufferDecl d; d.name = name; d.size = ln_sz;
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl; d.scope = ALL;
            d.preload_callback = [this, field](FusedModelBase&, int L, uint32_t core0_addr) {
                rpu_launch_ddr_broadcast_spm_dma(
                    layer_weights_[L].*field, /*src_offset_elements=*/0,
                    hidden_size(), core0_addr);
            };
            v.push_back(d);
        };
        ln_perlayer("norm1_w", &LayerWeights::norm1_w);
        ln_perlayer("norm1_b", &LayerWeights::norm1_b);
        ln_perlayer("norm3_w", &LayerWeights::norm3_w);
        ln_perlayer("norm3_b", &LayerWeights::norm3_b);

        // ---- per-layer linear biases (col: scatter per-core ; row: memset+core0) ----
        auto col_bias = [&](const char* name, int64_t per_core, at::Tensor LayerWeights::* field) {
            BufferDecl d; d.name = name; d.size = A(per_core * DWIDTH);
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl; d.scope = ALL;
            d.preload_callback = [this, per_core, field](FusedModelBase&, int L, uint32_t core0_addr) {
                rpu_launch_ddr_scatter_spm_dma(
                    layer_weights_[L].*field, /*src_offset_elements=*/0,
                    per_core, per_core * DWIDTH, core0_addr, NUM_CORES);
            };
            v.push_back(d);
        };
        auto row_bias = [&](const char* name, at::Tensor LayerWeights::* field) {
            BufferDecl d; d.name = name; d.size = A(h * DWIDTH);
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl; d.scope = ALL;
            d.preload_callback = [this, field](FusedModelBase&, int L, uint32_t core0_addr) {
                rpu_launch_memset_spm_multicore(core0_addr, hidden_size());
                rpu_launch_ddr_broadcast_spm_dma(
                    layer_weights_[L].*field, /*src_offset_elements=*/0,
                    hidden_size(), core0_addr, /*num_cores=*/1);
            };
            v.push_back(d);
        };
        col_bias("q_bias",   local_qhd,   &LayerWeights::to_q_b);
        col_bias("k_bias",   local_qhd,   &LayerWeights::to_k_b);
        col_bias("v_bias",   local_qhd,   &LayerWeights::to_v_b);
        col_bias("ff0_bias", local_inter, &LayerWeights::ff0_b);
        row_bias("out_bias", &LayerWeights::to_out_b);
        row_bias("ff2_bias", &LayerWeights::ff2_b);
        return v;
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        const int64_t seq_len     = chunk.len;
        const int64_t h           = hidden_size();
        const int64_t nq          = num_q_heads();
        const int64_t hd          = head_dim();
        const int64_t FF          = intermediate_size();
        const int64_t local_inter = FF / NUM_CORES;

        if (!ctx().input_in_spm) emit_layer_input_dma(layer_idx, chunk);

        // ---- vlln prefix (once, before block 0): residual1 = LN(raw ⊙ inv_w_rms, vlln) ----
        if (layer_idx == 0 && chunk.idx == 0) {
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                addr(0, "inv_w_rms"), addr(0, "residual1"), addr(0, "norm_hidden"),
                seq_len, h, c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
            rpu_launch_layernorm_spm_kernel(
                addr(0, "norm_hidden"), addr(0, "residual1"),
                addr(0, "vlln_w"), addr(0, "vlln_b"),
                seq_len, h, eps_, false, 0, NUM_CORES);
        }

        // ---- n1 = LN(x, norm1) -> norm_hidden ----
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual1"), addr(0, "norm_hidden"),
            layer_addr(layer_idx, 0, "norm1_w"), layer_addr(layer_idx, 0, "norm1_b"),
            seq_len, h, eps_, false, 0, NUM_CORES);

        // ---- q/k/v (col-partition) + bias ----
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "q_bias"), false, lw.to_q_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "k_bias"), false, lw.to_k_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "v_bias"), false, lw.to_v_ws);

        // ---- KV-insert@0 + bidirectional SDPA (MASK_NONE, scale 1/√64) ----
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        rpu_launch_insert_kcache_spm_unified(
            k_cache, 0, addr_offset("k").value, seq_len, nq, hd, NUM_CORES);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, 0, addr_offset("v").value, seq_len, nq, hd, NUM_CORES);
        const double attn_scale = 1.0 / std::sqrt(static_cast<double>(hd));
        rpu_launch_sdpa_spm_unified_kernel_v2(
            k_cache, v_cache, /*mask_type=*/0, attn_scale,
            addr_offset("q").value, addr_offset("sdpa_out").value,
            addr_offset("sdpa_tmp").value, /*sdpa_mask_off=*/0,
            seq_len, nq, nq, hd, /*kv_seq_len=*/seq_len, NUM_CORES, NUM_CORES);

        // ---- o-proj (row) + bias + PLAIN residual -> residual2 = x + attn ----
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.to_out_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, NUM_CORES, layer_addr(layer_idx, 0, "out_bias"), false, lw.to_out_ws);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
            seq_len, h, NUM_CORES, NUM_CORES);

        // ---- n3 = LN(x, norm3) -> norm_hidden ----
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual2"), addr(0, "norm_hidden"),
            layer_addr(layer_idx, 0, "norm3_w"), layer_addr(layer_idx, 0, "norm3_b"),
            seq_len, h, eps_, false, 0, NUM_CORES);

        // ---- FFN: ff0(col)+bias -> GELU(tanh-approx) -> ff2(row)+bias + PLAIN residual -> residual1 ----
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.ff0_w, addr(0, "ff_mid"),
            seq_len, FF, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "ff0_bias"), false, lw.ff0_ws);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "ff_mid"), addr(0, "ff_mid"),
            seq_len * local_inter, ValuOpType::ADD,
            GeluMode::TANH, NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "ff_mid"), lw.ff2_w, addr(0, "oproj"),
            seq_len, h, FF, 0, NUM_CORES, layer_addr(layer_idx, 0, "ff2_bias"), false, lw.ff2_ws);
        const bool z2_final = z2_bound_ &&
                              layer_idx == num_layers() - 1 &&
                              !ctx().output_to_spm;
        if (z2_final) {
            TORCH_CHECK(chunk.idx == 0 && chunk.offset == 0 &&
                            chunk.len == z2_prepared_seq_len_,
                        "gr00t_vl_encoder Z2 requires one full-sequence chunk");
        }
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual2"),
            z2_final ? z2_port_addr_ : addr(0, "residual1"),
            seq_len, h, NUM_CORES, NUM_CORES);

        // ---- output: framework writes residual1 -> output_tensor_ on the last layer
        // (output_to_spm=false there); intermediate layers keep residual1 SPM-resident.
        // Same path as the DiT (gate.multi_input-proven); caller copies immediately. ----
        if (!ctx().output_to_spm && !z2_final)
            emit_layer_output_dma(layer_idx, chunk, "residual1");
    }

    SdpaConfig make_sdpa_config(int mask = 0) const {
        return {sdpa_kernel_, head_dim(), num_q_heads(), num_q_heads(), attn_tp(), mask};
    }

private:
    std::vector<LayerWeights> layer_weights_;
    at::Tensor vlln_w_, vlln_b_, inv_w_rms_;
    double eps_ = 1e-5;
    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;
    bool z2_bound_ = false;
    int64_t z2_prepared_seq_len_ = 0;
    uint32_t z2_port_addr_ = 0;
    uint64_t z2_epoch_ = 0;
    uint64_t z2_plan_hash_ = 0;
};

}  // namespace v3

// =============================================================================
// Instance registry + public C API (mirror rpu_gr00t_dit_*).
// =============================================================================
using Gr00tVlEncoderRegistry = ModelHandleRegistry<v3::Gr00tVlEncoderModel>;

int64_t rpu_gr00t_vl_encoder_create() {
    return Gr00tVlEncoderRegistry::create();
}

void rpu_gr00t_vl_encoder_destroy(int64_t handle) {
    v3::gr00t_z2_internal::check_vl_destroy_allowed(handle);
    v3::gr00t_z2_internal::check_vl_component_destroy_allowed(handle);
    Gr00tVlEncoderRegistry::destroy(handle, "rpu_gr00t_vl_encoder_destroy");
}

void rpu_gr00t_vl_encoder_set_weights(
    int64_t handle,
    at::TensorList norm1_w_list, at::TensorList norm1_b_list,
    at::TensorList to_q_list, at::TensorList to_q_b_list,
    at::TensorList to_k_list, at::TensorList to_k_b_list,
    at::TensorList to_v_list, at::TensorList to_v_b_list,
    at::TensorList to_out_list, at::TensorList to_out_b_list,
    at::TensorList norm3_w_list, at::TensorList norm3_b_list,
    at::TensorList ff0_list, at::TensorList ff0_b_list,
    at::TensorList ff2_list, at::TensorList ff2_b_list,
    const at::Tensor& vlln_w, const at::Tensor& vlln_b, const at::Tensor& inv_w_rms,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t ff_inter, double eps)
{
    Gr00tVlEncoderRegistry::get(handle, "rpu_gr00t_vl_encoder")->set_weights(
        norm1_w_list, norm1_b_list, to_q_list, to_q_b_list, to_k_list, to_k_b_list,
        to_v_list, to_v_b_list, to_out_list, to_out_b_list, norm3_w_list, norm3_b_list,
        ff0_list, ff0_b_list, ff2_list, ff2_b_list,
        vlln_w, vlln_b, inv_w_rms, num_heads, head_dim, hidden_size, ff_inter, eps);
}

// W8A16 variant: required per-output-channel scale lists for the 6 quantized GEMMs (separate op
// because aten schemas can't default Tensor[] args; mirrors gr00t_dit_set_weights_w8a16).
void rpu_gr00t_vl_encoder_set_weights_w8a16(
    int64_t handle,
    at::TensorList norm1_w_list, at::TensorList norm1_b_list,
    at::TensorList to_q_list, at::TensorList to_q_b_list,
    at::TensorList to_k_list, at::TensorList to_k_b_list,
    at::TensorList to_v_list, at::TensorList to_v_b_list,
    at::TensorList to_out_list, at::TensorList to_out_b_list,
    at::TensorList norm3_w_list, at::TensorList norm3_b_list,
    at::TensorList ff0_list, at::TensorList ff0_b_list,
    at::TensorList ff2_list, at::TensorList ff2_b_list,
    const at::Tensor& vlln_w, const at::Tensor& vlln_b, const at::Tensor& inv_w_rms,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t ff_inter, double eps,
    at::TensorList to_q_scale_list, at::TensorList to_k_scale_list,
    at::TensorList to_v_scale_list, at::TensorList to_out_scale_list,
    at::TensorList ff0_scale_list, at::TensorList ff2_scale_list)
{
    Gr00tVlEncoderRegistry::get(handle, "rpu_gr00t_vl_encoder")->set_weights(
        norm1_w_list, norm1_b_list, to_q_list, to_q_b_list, to_k_list, to_k_b_list,
        to_v_list, to_v_b_list, to_out_list, to_out_b_list, norm3_w_list, norm3_b_list,
        ff0_list, ff0_b_list, ff2_list, ff2_b_list,
        vlln_w, vlln_b, inv_w_rms, num_heads, head_dim, hidden_size, ff_inter, eps,
        to_q_scale_list, to_k_scale_list, to_v_scale_list,
        to_out_scale_list, ff0_scale_list, ff2_scale_list);
}

at::Tensor rpu_gr00t_vl_encoder_forward(
    int64_t handle,
    const at::Tensor& raw,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gr00tVlEncoderRegistry::get(handle, "rpu_gr00t_vl_encoder")->forward(
        raw, k_caches, v_caches);
}

namespace v3::gr00t_z2_internal {

SpmPipelineComponentLayout prepare_vl_encoder(int64_t handle, int64_t seq_len) {
    return Gr00tVlEncoderRegistry::get(handle, "gr00t_spm_z2_prepare_vl")
        ->prepare_z2_layout(seq_len);
}

void unprepare_vl_encoder(int64_t handle, int64_t seq_len) {
    Gr00tVlEncoderRegistry::get(handle, "gr00t_spm_z2_unprepare_vl")
        ->unprepare_z2_layout(seq_len);
}

SpmDense2DSpec vl_encoder_port_spec(int64_t handle, int64_t seq_len) {
    return Gr00tVlEncoderRegistry::get(handle, "gr00t_spm_z2_vl_port_spec")
        ->z2_port_spec(seq_len);
}

void stage_vl_outer_fast_component(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches) {
    Gr00tVlEncoderRegistry::get(
        handle, "gr00t_spm_z2_stage_vl_outer_fast_component")
        ->stage_z2_outer_fast_component(guard, k_caches, v_caches);
}

void bind_vl_outer_fast_input(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& raw) {
    Gr00tVlEncoderRegistry::get(
        handle, "gr00t_spm_z2_bind_vl_outer_fast_input")
        ->bind_z2_outer_fast_input(guard, raw);
}

void adopt_vl_encoder(int64_t handle,
                      const SpmPipelineLease& lease,
                      const SpmTensorView& scratch) {
    Gr00tVlEncoderRegistry::get(handle, "gr00t_spm_z2_adopt_vl")
        ->adopt_z2_layout(lease, scratch);
}

void bind_vl_encoder(int64_t handle,
                     const SpmPipelineLease& lease,
                     const SpmPortView& port) {
    Gr00tVlEncoderRegistry::get(handle, "gr00t_spm_z2_bind_vl")
        ->bind_z2_port(lease, port);
}

void validate_vl_encoder(int64_t handle,
                         const SpmPipelineLease& lease) {
    Gr00tVlEncoderRegistry::get(handle, "gr00t_spm_z2_validate_vl")
        ->validate_z2_layout(lease);
}

void clear_vl_encoder(int64_t handle, uint64_t epoch, uint64_t plan_hash) {
    Gr00tVlEncoderRegistry::get(handle, "gr00t_spm_z2_clear_vl")
        ->clear_z2_layout(epoch, plan_hash);
}

void check_vl_component_destroy_allowed(int64_t handle) {
    Gr00tVlEncoderRegistry::get(
        handle, "gr00t_spm_z2_check_vl_component_destroy")
        ->check_z2_destroy_allowed();
}

}  // namespace v3::gr00t_z2_internal

void rpu_gr00t_vl_encoder_forward_z2(
    int64_t handle,
    const at::Tensor& raw,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t epoch,
    int64_t plan_hash)
{
    v3::gr00t_z2_internal::validate_vl_dispatch(
        handle, static_cast<uint64_t>(epoch),
        static_cast<uint64_t>(plan_hash));
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    Gr00tVlEncoderRegistry::get(handle, "rpu_gr00t_vl_encoder_z2")->forward_z2(
        raw, k_caches, v_caches,
        static_cast<uint64_t>(epoch), static_cast<uint64_t>(plan_hash));
}
