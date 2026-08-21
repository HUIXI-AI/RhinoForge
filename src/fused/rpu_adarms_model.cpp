// rpu_adarms_model.cpp — Pi0.5 Action Expert AdaRMS all-layers-once model
//                         (v3 FusedModelBase port)
//
// AdaRMSModel implements the flat v3 FusedModelBase contract with a flat-phase
// lifecycle, no registered callbacks, no persistent weights, and no post-graph.
//
// Key framework integration points:
//   - Replace `buf(name)` at SDPA + KV-insert call sites with
//     `addr_offset(name).value`, using the typed SpmOffset contract.
//   - Access model params via pimpl getters (num_layers() / hidden_size() /
//     num_q_heads() / num_kv_heads() / head_dim() / intermediate_size() /
//     attn_tp()).
//   - Class-member `cond_ref_` keeps the tensor alive for the synchronous
//     forward() (run_all_layers consumes the DDR pointer inside the batched
//     dispatch).
//   - set_weights ENDS with `invalidate_model_state();` as last non-empty
//     statement.
//
// Execution scope:
//   - Per-forward cond DMA (layer 0 / chunk 0 gate): `cond_loaded_this_forward_`
//     subclass flag toggled on the first build_layer_subgraph invocation.
//   - Per-layer GEMV (cond * dense_w + bias -> [scale|shift|gate]) at
//     chunk.idx == 0, result persists in SPM across that layer's remaining
//     chunks.
//   - (1+scale) preprocessing via scalar ADD to the live GEMV output SPM buffer.
//   - Gated residual 3-step: SUB full-size → 1xC->NxC MUL gate slice → ADD.
//   - Always SEQUENTIAL (causal autoregressive); flat phases (all LayerWide).
//   - No final_norm fusion (Pi0.5 flow-matching head runs on Python).
//   - cfg.cross_layer_batch_size = num_layers() (force single group).
//
// Design contract: FusedModelBase.

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"  // shared runtime state
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>
#include <algorithm>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace v3 {

namespace {

void check_adarms_scale_lists(
    int64_t n_layers,
    at::TensorList q_w, at::TensorList k_w, at::TensorList v_w,
    at::TensorList o_w, at::TensorList gate_w, at::TensorList up_w,
    at::TensorList down_w,
    at::TensorList q_ws, at::TensorList k_ws, at::TensorList v_ws,
    at::TensorList o_ws, at::TensorList gate_ws, at::TensorList up_ws,
    at::TensorList down_ws)
{
    const bool has_scale = !q_ws.empty();
    auto check_scale_list = [&](const at::TensorList& list, const char* name) {
        if (has_scale) {
            TORCH_CHECK(static_cast<int64_t>(list.size()) == n_layers,
                        "adarms_set_weights: ", name, ".size()=", list.size(),
                        " != num_layers=", n_layers);
        } else {
            TORCH_CHECK(list.empty(),
                        "adarms_set_weights: ", name,
                        " must be empty unless all quantized scale lists are provided");
        }
    };
    check_scale_list(k_ws,    "k_w_scale");
    check_scale_list(v_ws,    "v_w_scale");
    check_scale_list(o_ws,    "o_w_scale");
    check_scale_list(gate_ws, "gate_scale");
    check_scale_list(up_ws,   "up_scale");
    check_scale_list(down_ws, "down_scale");
    if (has_scale) {
        TORCH_CHECK(static_cast<int64_t>(q_ws.size()) == n_layers,
                    "adarms_set_weights: q_w_scale.size()=", q_ws.size(),
                    " != num_layers=", n_layers);
    }

    auto check_weight_dtype = [&](const at::Tensor& w,
                                  const at::Tensor& scale,
                                  const char* name,
                                  int64_t i) {
        TORCH_CHECK(w.device().type() == at::kPrivateUse1 && w.is_contiguous(),
                    "adarms_set_weights: ", name, "[", i,
                    "] must be contiguous RPU tensor");
        TORCH_CHECK(w.scalar_type() == at::kHalf || w.scalar_type() == at::kChar
                    || w.scalar_type() == at::kByte,
                    "adarms_set_weights: ", name, "[", i,
                    "] must be fp16 / int8 / packed-uint8, got ", w.scalar_type());
        if (has_scale) {
            TORCH_CHECK(w.scalar_type() == at::kChar || w.scalar_type() == at::kByte,
                        "adarms_set_weights: quantized mode requires int8(W8A16) or "
                        "uint8(packed-INT4) ", name, "[", i, "], got ", w.scalar_type());
            TORCH_CHECK(scale.defined() && scale.dim() == 1
                        && scale.scalar_type() == at::kHalf
                        && scale.device().type() == at::kPrivateUse1
                        && scale.is_contiguous(),
                        "adarms_set_weights: ", name, "_scale[", i,
                        "] must be 1D contiguous fp16 RPU tensor");
            TORCH_CHECK(scale.numel() == w.size(0),
                        "adarms_set_weights: ", name, "_scale[", i,
                        "].numel()=", scale.numel(),
                        " != output dim=", w.size(0));
        } else {
            TORCH_CHECK(w.scalar_type() != at::kChar && w.scalar_type() != at::kByte,
                        "adarms_set_weights: quantized ", name, "[", i,
                        "] requires scale lists");
        }
    };

    for (int64_t i = 0; i < n_layers; ++i) {
        check_weight_dtype(q_w[i],    has_scale ? q_ws[i]    : at::Tensor(), "q_w", i);
        check_weight_dtype(k_w[i],    has_scale ? k_ws[i]    : at::Tensor(), "k_w", i);
        check_weight_dtype(v_w[i],    has_scale ? v_ws[i]    : at::Tensor(), "v_w", i);
        check_weight_dtype(o_w[i],    has_scale ? o_ws[i]    : at::Tensor(), "o_w", i);
        check_weight_dtype(gate_w[i], has_scale ? gate_ws[i] : at::Tensor(), "gate_w", i);
        check_weight_dtype(up_w[i],   has_scale ? up_ws[i]   : at::Tensor(), "up_w", i);
        check_weight_dtype(down_w[i], has_scale ? down_ws[i] : at::Tensor(), "down_w", i);
    }
}

}  // namespace

// =============================================================================
// AdaRMSModel — v3::FusedModelBase subclass (Pi0.5 Action Expert)
// =============================================================================

class AdaRMSModel : public FusedModelBase {
public:
    // Per-layer weights. Dense weights are expanded+transformed on Python side
    // to [8*3*hidden, hidden] (col-partition swizzled). Dense biases are
    // [3*hidden] fp16.
    struct LayerWeights {
        // Attention
        at::Tensor q_w, k_w, v_w, o_w;
        // MLP
        at::Tensor gate_proj_w, up_proj_w, down_proj_w;
        at::Tensor q_ws, k_ws, v_ws, o_ws;
        at::Tensor gate_ws, up_ws, down_ws;
        // AdaRMS GEMV (per-layer, not shared)
        at::Tensor attn_dense_w;    // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor attn_dense_b;    // [3*hidden] fp16
        at::Tensor mlp_dense_w;     // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor mlp_dense_b;     // [3*hidden] fp16
    };

    AdaRMSModel() = default;

    // ========================================================================
    // set_weights (called by the Python converter)
    //
    // Layer-specific AdaRMS dense weights (attn & mlp) are per-layer.
    // cos/sin are globals computed on RPU device with kernel format [max_pos, head_dim/2].
    // eps is the RMSNorm epsilon (typically 1e-6).
    // ========================================================================
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
        at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
        const at::Tensor& cos, const at::Tensor& sin,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps,
        at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
        at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
        at::TensorList gate_scale_list, at::TensorList up_scale_list,
        at::TensorList down_scale_list)
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "adarms_set_weights: empty weight lists");
        TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0
                    && hidden_size > 0 && intermediate_size > 0,
                    "adarms_set_weights: dim params must be positive");
        TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                    "adarms_set_weights: num_q_heads (", num_q_heads,
                    ") must be divisible by num_kv_heads (", num_kv_heads, ")");

        // All 11 per-layer lists must have the same length
        auto check_list = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "adarms_set_weights: ", n, ".size()=", l.size(),
                        " != num_layers=", N);
        };
        check_list(k_w_list,          "k_w_list");
        check_list(v_w_list,          "v_w_list");
        check_list(o_w_list,          "o_w_list");
        check_list(gate_list,         "gate_list");
        check_list(up_list,           "up_list");
        check_list(down_list,         "down_list");
        check_list(attn_dense_w_list, "attn_dense_w_list");
        check_list(attn_dense_b_list, "attn_dense_b_list");
        check_list(mlp_dense_w_list,  "mlp_dense_w_list");
        check_list(mlp_dense_b_list,  "mlp_dense_b_list");
        check_adarms_scale_lists(
            N,
            q_w_list, k_w_list, v_w_list, o_w_list,
            gate_list, up_list, down_list,
            q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
            gate_scale_list, up_scale_list, down_scale_list);
        const bool has_scale = !q_w_scale_list.empty();

        // Global tensor checks
        TORCH_CHECK(cos.defined() && sin.defined(),
                    "adarms_set_weights: cos/sin must be defined");
        TORCH_CHECK(cos.dim() == 2,
                    "adarms_set_weights: cos must be 2D, got ", cos.dim(), "D");
        TORCH_CHECK(sin.sizes() == cos.sizes(),
                    "adarms_set_weights: sin.sizes() must equal cos.sizes()");
        TORCH_CHECK(cos.size(-1) == head_dim || cos.size(-1) == head_dim / 2,
                    "adarms_set_weights: cos last dim must be head_dim or head_dim/2");

        // Per-layer defined/rank checks (2D for linear weights, 1D for bias)
        auto check_all_defined_rank = [&](const at::TensorList& list,
                                          const char* name, int64_t expected_rank) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "adarms_set_weights: ", name, "[", i, "] is undefined");
                TORCH_CHECK(list[i].dim() == expected_rank,
                            "adarms_set_weights: ", name, "[", i, "] must be ",
                            expected_rank, "D, got ", list[i].dim(), "D");
            }
        };
        check_all_defined_rank(q_w_list,          "q_w_list",          2);
        check_all_defined_rank(k_w_list,          "k_w_list",          2);
        check_all_defined_rank(v_w_list,          "v_w_list",          2);
        check_all_defined_rank(o_w_list,          "o_w_list",          2);
        check_all_defined_rank(gate_list,         "gate_list",         2);
        check_all_defined_rank(up_list,           "up_list",           2);
        check_all_defined_rank(down_list,         "down_list",         2);
        // Dense weight is expanded+transformed [8*3*hidden, hidden] (2D); bias is [3*hidden] (1D)
        check_all_defined_rank(attn_dense_w_list, "attn_dense_w_list", 2);
        check_all_defined_rank(attn_dense_b_list, "attn_dense_b_list", 1);
        check_all_defined_rank(mlp_dense_w_list,  "mlp_dense_w_list",  2);
        check_all_defined_rank(mlp_dense_b_list,  "mlp_dense_b_list",  1);

        // Commit model params (pimpl: hidden/q/kv/head_dim/intermediate + attn_tp_)
        set_model_params(num_q_heads, num_kv_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;

        // AdaRMS-specific derived values (the v3 framework's set_model_params
        // already sets pimpl attn_tp_ = std::min(8, num_kv_heads), matching
        // v2's std::min(NUM_CORES, num_kv_heads)). Cache here for readability
        // inside the hot build_layer_subgraph path.
        local_q_heads_ = num_q_heads / attn_tp();
        local_kv_dim_  = num_kv_heads * head_dim / attn_tp();

        // Copy weights into layer state
        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                gate_list[i], up_list[i], down_list[i],
                has_scale ? q_w_scale_list[i] : at::Tensor(),
                has_scale ? k_w_scale_list[i] : at::Tensor(),
                has_scale ? v_w_scale_list[i] : at::Tensor(),
                has_scale ? o_w_scale_list[i] : at::Tensor(),
                has_scale ? gate_scale_list[i] : at::Tensor(),
                has_scale ? up_scale_list[i] : at::Tensor(),
                has_scale ? down_scale_list[i] : at::Tensor(),
                attn_dense_w_list[i], attn_dense_b_list[i],
                mlp_dense_w_list[i],  mlp_dense_b_list[i],
            });
        }
        cos_ = cos;
        sin_ = sin;

        invalidate_model_state();  // Must remain the last statement of set_weights.
    }

    void set_rope_position(int64_t position) {
        rope_position_ = position;
    }

    // ========================================================================
    // forward -- cond is an explicit per-call argument on the RPU device.
    //
    // Each forward = one Pi0.5 denoising step with a new cond.
    // The cache key excludes cond values; the REPLAY cursor patches the
    // cond DDR pointer on each subsequent denoise step.
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& hidden_states,
        const at::Tensor& cond,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal)
    {
        TORCH_CHECK(num_layers() > 0,
                    "AdaRMSModel::forward called before set_weights");
        TORCH_CHECK(!layer_weights_.empty(),
                    "AdaRMSModel::forward: internal state inconsistent");
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "AdaRMSModel::forward: hidden_states must be on RPU device");
        TORCH_CHECK(cond.defined() && cond.dim() == 1 &&
                    cond.size(0) == hidden_size(),
                    "AdaRMSModel::forward: cond must be 1D [hidden_size=",
                    hidden_size(), "], got dim=", (cond.defined() ? cond.dim() : -1),
                    " size=", (cond.defined() ? cond.size(0) : -1));
        TORCH_CHECK(cond.device().type() == at::kPrivateUse1,
                    "AdaRMSModel::forward: cond must be on RPU device");
        TORCH_CHECK(cond.scalar_type() == at::kHalf,
                    "AdaRMSModel::forward: cond must be fp16");
        TORCH_CHECK(position >= 0,
                    "AdaRMSModel::forward: position must be non-negative, got ", position);
        TORCH_CHECK(hidden_states.is_contiguous(),
                    "AdaRMSModel::forward: hidden_states must be contiguous "
                    "(downstream kernels use raw data_ptr())");
        TORCH_CHECK(cond.is_contiguous(),
                    "AdaRMSModel::forward: cond must be contiguous "
                    "(DMA source pointer assumes row-major layout)");

        // AdaRMS always auto-computes chunk_size (not affected by global override)
        set_chunk_size_override(0);

        // Reset per-forward cond state: a new denoise step is a new cond.
        // cond DMA/GEMV will re-emit into the graph on the first chunk of layer 0.
        // cond_ref_ keeps the tensor alive for the synchronous forward (batch
        // end executes before run_all_layers returns); the DDR pointer is
        // cursor-patched per REPLAY via mutable DMA + cond_src_base_.
        cond_loaded_this_forward_ = false;
        cond_ref_ = cond;
        // cond_src_base_ feeds rpu_launch_ddr_scatter_spm_dma_mutable in the
        // cond scatter site below (Phase 0). Set once per forward; REPLAY end()
        // patches each per-core DMA's src via Queue_t::update_dma_kernel from
        // this live storage. Per-forward flush mirrors fused_model_base's
        // hidden_in_src_base_ pattern (caller may have CPU-dirty lines).
        cond_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond.data_ptr());
        // The SDK enqueue path flushes its own uncached
        // descriptor buffers, while a caller tensor is a CACHED HostDDR_t that
        // only an explicit RpuDdrFlush ever cleans (see the same analysis in
        // graph_cpu_fallback_stub.cpp). `cond` is caller-supplied, exactly as
        // the comment above says. Keep the explicit flush aligned with the
        // equivalent rpu_qwenpi05_model.cpp path.
        rpu_ddr_flush_force(cond.data_ptr<c10::Half>());

        // Prepare the SDPA mask outside per-layer Phase 6. The mask is invariant
        // across all expert layers within one forward, so each layer only needs
        // the prepared DDR-to-SPM DMA. The single-chunk-per-forward
        // assumption (suffix_len <= chunk_size auto-pick) matches the existing
        // sdpa_prepare_mask shape contract: it asserts mask_2d.size(0)==seq_q,
        // which already requires seq_q == ctx().seq_len == chunk.len.
        //
        // Cache the prepared mask across forwards when the caller passes the
        // same attention_mask tensor, allowing a matching data_ptr to skip
        // preparation.
        // We hold a strong ref to the input tensor so its storage cannot be
        // reallocated to a different tensor while the cache is valid.
        const int64_t prep_seq_q = hidden_states.size(1);
        const int64_t prep_seq_k = position + prep_seq_q;
        const bool    prep_is_causal = is_causal && (prep_seq_q > 1);
        const bool    has_mask = attention_mask.has_value() && attention_mask->defined();
        const void*   in_ptr = has_mask ? attention_mask->data_ptr() : nullptr;
        const void*   cached_ptr = prepared_mask_input_ref_.defined()
                                   ? prepared_mask_input_ref_.data_ptr() : nullptr;
        const bool cache_hit =
            (cached_ptr               == in_ptr)        &&
            (prepared_mask_seq_q_     == prep_seq_q)    &&
            (prepared_mask_seq_k_     == prep_seq_k)    &&
            (prepared_mask_is_causal_ == prep_is_causal);
        if (!cache_hit) {
            prepared_mask_ = sdpa_prepare_mask(
                attention_mask, prep_is_causal, prep_seq_q, prep_seq_k,
                sdpa_stable_mask_cache());
            prepared_mask_input_ref_  = has_mask ? attention_mask.value() : at::Tensor();
            prepared_mask_seq_q_      = prep_seq_q;
            prepared_mask_seq_k_      = prep_seq_k;
            prepared_mask_is_causal_  = prep_is_causal;
        }

        return run_all_layers(hidden_states, k_caches, v_caches,
                              attention_mask, position, is_causal);
    }

protected:
    // ========================================================================
    // static_config -- graph-naive flat-phase subclass; no legacy graph slot.
    //
    // All four pointer-to-member
    // slots (preload_fn / kv_first_fn / kv_first_chunk_plan_fn / post_fn)
    // stay nullptr by default. Flat-phase baseline; no persistent weights.
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers       = num_layers();
        // Force single group — matches Gemma/SigLIP/QwenPi05/Qwen3 discipline.
        // Every subclass pins a single group and ignores the global runtime knob.
        cfg.cross_layer_batch_size = num_layers();
        // Callbacks are all null for AdaRMS (no preload, KV_FIRST, or post step).
        return cfg;
    }

    // ========================================================================
    // dynamic_config -- always SEQUENTIAL
    // ========================================================================
    ModelDynamicConfig dynamic_config(const ChunkPlan& /*plan*/) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;   // AdaRMS never uses KV_FIRST
        cfg.inter_layer_io = InterLayerIO::AUTO;
        return cfg;
    }

    // ========================================================================
    // declare_buffers -- flat phases, ALL LayerWide (matches v2 verbatim).
    //
    // AdaRMS adds 5 NEW SPM buffers on top of the decoder skeleton:
    //   - cond            [hidden_size]      -- DMA'd once per forward
    //   - attn_gemv       [3 * hidden_size]  -- scale|shift|gate for attn norm
    //   - mlp_gemv        [3 * hidden_size]  -- scale|shift|gate for mlp  norm
    //   - bias_temp       [3 * hidden_size]  -- temporary for bias DMA + add
    //   - gemv_partial    [3 * hidden_size]  -- per-core partial before all_reduce
    // All BufferScope::LayerWide (flat phases — aliasing deferred; the
    // v3 framework handles lifecycle aliasing via phase ranges, but AdaRMS
    // uses 0,0 to match v2's forward parity).
    //
    // No per-buffer preload hook is used; all buffers are StorageClass::Temp.
    // ========================================================================
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        int64_t cs = ctx.chunk_size;
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();

        int64_t local_q  = nq / attn_tp();
        int64_t local_kv = nkv * hd / attn_tp();
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        int64_t res   = A(cs * h * DWIDTH);
        int64_t q     = A(cs * local_q * hd * DWIDTH);
        int64_t kv    = A(cs * local_kv * DWIDTH);
        int64_t out   = A(cs * local_q * hd * DWIDTH);
        int64_t oproj = A(cs * h * DWIDTH);
        int64_t mlp   = A(cs * (is_ / NUM_CORES) * DWIDTH);

        // SDPA tmp and mask sizing
        SdpaConfig sdpa_cfg = make_sdpa_config(ctx.use_attn_mask ? 4 : 1);
        int64_t tmp = A(sdpa_compute_tmp_v16_size(sdpa_cfg, cs) * 32);
        int64_t mask_sz = ctx.use_attn_mask
            ? A(cs * CeilDiv(ctx.max_kv_seq_len, (int64_t)16) * 32)
            : 0;

        // AdaRMS extras (sized per-core; broadcast DMA means same addr on all cores)
        int64_t cond_sz     = A(h * DWIDTH);
        int64_t gemv_out_sz = A(3 * h * DWIDTH);

        constexpr BufferScope ALL = BufferScope::LayerWide;

        return {
            // Structural (mirrors Gemma, but flat phases 0,0)
            {"residual1",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"input_norm",   res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"residual2",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"q",            q,       0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"k",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"v",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"output",       out,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"oproj",        oproj,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_tmp",     tmp,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_mask",    mask_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"gate",         mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"up",           mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"down",         res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},

            // AdaRMS-specific (flat phases, LayerWide)
            {"cond",         cond_sz,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"attn_gemv",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"mlp_gemv",     gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"bias_temp",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"gemv_partial", gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
        };
    }

    // ========================================================================
    // build_layer_subgraph (SEQUENTIAL)
    //
    // Execution flow per layer:
    //
    //   Phase 0 (layer 0, chunk 0 ONLY): DMA cond DDR -> "cond" SPM (scatter)
    //   Phase A (every layer, chunk 0):  GEMV attn_dense_w * cond + attn_dense_b -> "attn_gemv"
    //                                    GEMV mlp_dense_w  * cond + mlp_dense_b  -> "mlp_gemv"
    //                                    +1.0 on first hidden_size slice (scale += 1)
    //                                    Result persists in SPM across chunks of this layer.
    //   Phase 1:  input DMA -> "residual1"
    //   Phase 2:  RMSNorm with (1+scale) weight  "residual1" -> "input_norm"
    //             1xC->NxC ADD shift (broadcast shift slice onto each row) in-place
    //   Phase 3:  QKV Linear (col partition, attn_tp cores)
    //   Phase 4:  RoPE on Q, K (in-place, use cos_/sin_)
    //   Phase 5:  insert_kcache/insert_vcache at absolute position
    //   Phase 6:  SDPA (causal or 2D mask)
    //   Phase 7:  O_proj (row partition, attn_tp cores) -> "oproj"
    //   Phase 8:  all_reduce_sum_residual: oproj + residual1 -> residual2
    //   Phase 9:  GATED attn residual:  residual2 = attn_gate * (residual2 - residual1) + residual1
    //             (3-step: SUB full-size, 1xC->NxC MUL with attn_gate slice, ADD full-size)
    //   Phase 10: post-norm with (1+scale_mlp): residual2 -> residual1
    //             1xC->NxC ADD shift_mlp in-place
    //   Phase 11: MLP (gate/up/GELU/mul/down) + reduce -> residual1 = reduce(down) + residual2
    //   Phase 13: GATED MLP residual: residual1 = mlp_gate * (residual1 - residual2) + residual2
    //   Phase 14: output DMA (unless ctx().output_to_spm)
    //
    // Pitfall 3 structural fix: SDPA + KV-insert use addr_offset(name).value
    // (typed SpmOffset, compile-time distinct from absolute addr() uint32_t).
    // ========================================================================
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t cos_sin_start = ctx().position + chunk.offset;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int tp = attn_tp();
        int64_t num_elems_full = seq_len * h;
        bool is_first_layer_first_chunk =
            (layer_idx == 0) && (chunk.idx == 0) && !cond_loaded_this_forward_;
        bool is_chunk_zero = (chunk.idx == 0);

        // --------------------------------------------------------------------
        // Phase 0: cond DMA (once per forward)
        // SCATTER cond_ref_ [hidden_size] fp16 -> "cond" so that each core c
        // holds cond[c*H/NUM_CORES : (c+1)*H/NUM_CORES] at its local SPM
        // offset. This matches the row-partition GEMM used in
        // adarms_gemv_to_spm (each core reads its own local_k=H/NUM_CORES
        // slice of cond at the same local offset).
        //
        // On REPLAY the DDR pointer is cursor-patched to the new cond tensor.
        // --------------------------------------------------------------------
        if (is_first_layer_first_chunk) {
            int64_t local_k           = h / NUM_CORES;
            int64_t core_stride_bytes = local_k * DWIDTH;
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &cond_src_base_,
                /*src_offset_bytes=*/0,
                /*elements_per_core=*/local_k,
                /*core_stride_bytes=*/core_stride_bytes,
                addr(0, "cond"),
                /*num_cores=*/NUM_CORES);
            cond_loaded_this_forward_ = true;
        }

        // --------------------------------------------------------------------
        // Phase A: per-layer GEMV at chunk.idx == 0
        // Computes [scale | shift | gate] from cond * dense_w + bias, with
        // +1.0 baked into the scale slice. Result persists in SPM for this
        // layer's remaining chunks.
        // --------------------------------------------------------------------
        if (is_chunk_zero) {
            adarms_gemv_to_spm(addr(0, "cond"),
                               lw.attn_dense_w, lw.attn_dense_b,
                               addr(0, "attn_gemv"),
                               addr(0, "bias_temp"),
                               addr(0, "gemv_partial"));
            adarms_gemv_to_spm(addr(0, "cond"),
                               lw.mlp_dense_w,  lw.mlp_dense_b,
                               addr(0, "mlp_gemv"),
                               addr(0, "bias_temp"),
                               addr(0, "gemv_partial"));
        }

        // GEMV slice offsets inside attn_gemv / mlp_gemv (computed from core 0
        // base address; broadcast DMA means every core shares the same offsets).
        uint32_t attn_scale = addr(0, "attn_gemv");
        uint32_t attn_shift = attn_scale + (uint32_t)(h * DWIDTH);
        uint32_t attn_gate  = attn_scale + (uint32_t)(h * DWIDTH * 2);
        uint32_t mlp_scale  = addr(0, "mlp_gemv");
        uint32_t mlp_shift  = mlp_scale + (uint32_t)(h * DWIDTH);
        uint32_t mlp_gate   = mlp_scale + (uint32_t)(h * DWIDTH * 2);

        // --------------------------------------------------------------------
        // Phase 1: layer input DMA -> "residual1"
        // Skip if previous layer left residual1 in SPM (SPM_RESIDENT mode).
        // --------------------------------------------------------------------
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);  // writes to "residual1"
        }

        // --------------------------------------------------------------------
        // Phase 2: AdaRMSNorm (attn side): residual1 -> input_norm
        // (1+scale) was applied during GEMV, so RMSNorm uses attn_scale directly.
        // Then add shift (broadcast 1xC onto NxC rows).
        // --------------------------------------------------------------------
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            attn_scale,
            seq_len, h, eps_);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            attn_shift, addr(0, "input_norm"), addr(0, "input_norm"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);

        // --------------------------------------------------------------------
        // Phase 3: QKV Linear (attn_tp cores, col partition)
        // --------------------------------------------------------------------
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h,
            /*partition=*/1, tp, /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.q_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp, /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.k_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp, /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.v_ws);

        // --------------------------------------------------------------------
        // Phase 4: RoPE (in-place, no QK RMSNorm for AdaRMS Gemma)
        // --------------------------------------------------------------------
        int64_t local_kv_heads = nkv / tp;
        c10::Half* cos_ptr = cos_.data_ptr<c10::Half>();
        c10::Half* sin_ptr = sin_.data_ptr<c10::Half>();
        // RoPE reads the LOGICAL position, the KV insert below the physical row
        // (see rope_position_). Equal only when the prefix has no pad rows.
        const int64_t rope_start = (rope_position_ >= 0)
            ? rope_position_ + chunk.offset : cos_sin_start;
        rpu_launch_rope_spm_kernel(
            addr(0, "q"), addr(0, "q"),
            cos_ptr, sin_ptr,
            seq_len, local_q_heads_, hd, rope_start, tp);
        rpu_launch_rope_spm_kernel(
            addr(0, "k"), addr(0, "k"),
            cos_ptr, sin_ptr,
            seq_len, local_kv_heads, hd, rope_start, tp);

        // --------------------------------------------------------------------
        // Phase 5: KV cache insert at absolute position.
        //
        // Pitfall 3 structural fix: KV-insert takes SPM OFFSETS, not absolute
        // addresses. addr_offset("name").value is compile-time distinct from
        // addr(core, "name").
        // --------------------------------------------------------------------
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        rpu_launch_insert_kcache_spm_unified(
            k_cache, cos_sin_start, addr_offset("k").value,
            seq_len, nkv, hd, tp);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, cos_sin_start, addr_offset("v").value,
            seq_len, nkv, hd, tp);

        // --------------------------------------------------------------------
        // Phase 6: SDPA (causal or 2D mask).
        //
        // Pitfall 3 structural fix: SDPA args (q / output / sdpa_tmp /
        // sdpa_mask) take SPM offsets via addr_offset(...).value.
        // --------------------------------------------------------------------
        int64_t kv_seq_len = ctx().position + ctx().seq_len;
        // The mask is prepared once at forward() entry.
        // Per-layer Phase 6 only DMAs the prepared DDR tensor → SPM.
        // mask_type was determined at prepare time (MASK_NONE/MASK_LTM/MASK_2D).
        sdpa_dma_mask_to_spm(prepared_mask_,
                             addr_offset("sdpa_mask").value,
                             seq_len, kv_seq_len, tp);
        int mask_type = prepared_mask_.mask_type;
        rpu_launch_sdpa_spm_unified_kernel_v2(
            k_cache, v_cache, mask_type, c10::nullopt,
            addr_offset("q").value,
            addr_offset("output").value,
            addr_offset("sdpa_tmp").value,
            addr_offset("sdpa_mask").value,
            seq_len, nq, nkv, hd,
            kv_seq_len, tp, NUM_CORES);

        // --------------------------------------------------------------------
        // Phase 7: O_proj (row partition, attn_tp cores)
        // --------------------------------------------------------------------
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "output"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd,
            /*partition=*/0, tp, /*bias_spm_addr=*/0, /*force_gemm=*/false, lw.o_ws);

        // --------------------------------------------------------------------
        // Phase 8: all_reduce_sum_residual (attn)
        // residual2 = reduce_sum(oproj, attn_tp cores) + residual1
        // Output goes to all NUM_CORES.
        // --------------------------------------------------------------------
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
            seq_len, h, tp, NUM_CORES);

        // --------------------------------------------------------------------
        // Phase 9: GATED attn residual
        // residual2 = attn_gate * (residual2 - residual1) + residual1
        // Implemented as 3-step: SUB full-size, 1xC->NxC MUL gate slice, ADD full-size
        // --------------------------------------------------------------------
        rpu_launch_eltwise_binary_spm_kernel(          // residual2 -= residual1
            addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
            num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(  // residual2 *= attn_gate (1xC broadcast)
            attn_gate, addr(0, "residual2"), addr(0, "residual2"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
        rpu_launch_eltwise_binary_spm_kernel(          // residual2 += residual1
            addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);

        // --------------------------------------------------------------------
        // Phase 10: Post-attention AdaRMSNorm (MLP side): residual2 -> residual1
        // (1+scale_mlp) baked into mlp_scale; then add shift_mlp.
        // --------------------------------------------------------------------
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual2"), addr(0, "residual1"),
            mlp_scale,
            seq_len, h, eps_);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            mlp_shift, addr(0, "residual1"), addr(0, "residual1"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::ADD, /*is_bopa=*/false);

        // --------------------------------------------------------------------
        // Phase 11: MLP pipeline
        // --------------------------------------------------------------------
        emit_mlp_pipeline(lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w,
                          seq_len, ActivationKind::GELU,
                          lw.gate_ws, lw.up_ws, lw.down_ws);

        // --------------------------------------------------------------------
        // Phase 13: GATED MLP residual
        // residual1 = mlp_gate * (residual1 - residual2) + residual2
        // After emit_mlp_pipeline: residual1 = reduce(down) + residual2
        // So (residual1 - residual2) = reduce(down), then gate it, then add back residual2.
        // --------------------------------------------------------------------
        rpu_launch_eltwise_binary_spm_kernel(          // residual1 -= residual2
            addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
            num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(  // residual1 *= mlp_gate (1xC broadcast)
            mlp_gate, addr(0, "residual1"), addr(0, "residual1"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
        rpu_launch_eltwise_binary_spm_kernel(          // residual1 += residual2
            addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);

        // --------------------------------------------------------------------
        // Phase 14: Output DMA (unless output stays in SPM for next layer)
        // --------------------------------------------------------------------
        if (!ctx().output_to_spm) {
            emit_layer_output_dma(layer_idx, chunk);  // from "residual1"
        }
    }

private:
    SdpaConfig make_sdpa_config(int mask = 1) const {
        return {sdpa_kernel_, head_dim(), num_q_heads(), num_kv_heads(),
                attn_tp(), mask};
    }

    // ========================================================================
    // GEMV helper: row-partition 8-core + fused all_reduce+bias.
    //
    // Weight is the natural [3H, H] tensor row-partition swizzled so each core
    // holds K/NUM_CORES=H/NUM_CORES columns of all 3H output rows. Layout of
    // the final output (on every core after reduce):
    //   [scale(hidden) | shift(hidden) | gate(hidden)]
    // The scale portion gets +1.0 in place so later RMSNorm sees (1+scale)
    // directly.
    //
    // Three distinct SPM buffers are required (all_reduce_sum_residual expects
    // input, residual, and output to be non-overlapping SPM regions):
    //   gemv_out_spm_addr    : final reduced+biased [3H] (broadcast to all cores)
    //   bias_temp_spm_addr   : DMA'd bias, used as the reduce-residual [3H]
    //   partial_spm_addr     : per-core partial [3H] from the row-partition GEMM
    // ========================================================================
    void adarms_gemv_to_spm(uint32_t cond_spm_addr,
                            const at::Tensor& dense_w,
                            const at::Tensor& dense_b,
                            uint32_t gemv_out_spm_addr,
                            uint32_t bias_temp_spm_addr,
                            uint32_t partial_spm_addr)
    {
        int64_t h = hidden_size();
        int64_t out_features = 3 * h;

        // Step 1: DMA bias -> bias_temp (replicated across all 8 cores, acts
        // as the residual input to the fused all_reduce+bias kernel).
        rpu_launch_ddr_broadcast_spm_dma(
            dense_b.data_ptr<c10::Half>(), out_features, bias_temp_spm_addr);
        // Step 2: Row-partition GEMV. partial_spm_addr holds the per-core
        // partial [3H] along the local_k = H/NUM_CORES split.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, /*num_cores=*/NUM_CORES,
            /*bias_spm_addr=*/0,
            /*force_gemm=*/false);
        // Step 3: Fused all_reduce + bias. reduce(partial) + bias -> gemv_out
        // broadcast to all output cores. Input and output MUST be distinct
        // SPM regions.
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
            /*M=*/1, /*N=*/out_features,
            /*input_num_cores=*/NUM_CORES, /*output_num_cores=*/NUM_CORES);
        // Step 4: (1+scale) -- add +1.0 only to the first hidden_size elements
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0),
            gemv_out_spm_addr, h, ValuOpType::ADD);
    }

    // ----- Model state -----
    std::vector<LayerWeights> layer_weights_;
    at::Tensor cos_, sin_;
    // Logical RoPE start for the action suffix, or -1 to use the physical row.
    //
    // `cos_sin_start` is one number doing two jobs: the RoPE table index and the
    // KV cache write row. The suffix must be WRITTEN at the physical row
    // (runtime.py resets the shared cache to prefix_pad_masks.shape[1], so it
    // lands past the prefix), but its RoPE position is the LOGICAL one the
    // reference uses: sum(prefix_pad_masks). Those differ by exactly the number
    // of pad rows in the prefix. The suffix itself never has pad rows
    // (suffix_pad_masks is all ones), so the positions are a contiguous range
    // and a scalar start is enough — no gathered table, unlike the prefix.
    //
    // Baked into the graph at BUILD, so callers MUST put it in the signature.
    int64_t rope_position_ = -1;
    double eps_ = 1e-6;

    // Derived dims (computed in set_weights, cached for hot path)
    int64_t local_q_heads_ = 0;
    int64_t local_kv_dim_  = 0;

    // Per-forward cond state (reset on every forward() entry).
    // cond_ref_ keeps the tensor alive until run_all_layers returns (the
    // batch end is synchronous inside run_all_layers); the DDR pointer is
    // cursor-patched per REPLAY via cond_src_base_ + mutable DMA.
    at::Tensor cond_ref_;
    uint64_t   cond_src_base_ = 0;
    bool cond_loaded_this_forward_ = false;

    // Per-forward prepared SDPA mask shared across layers.
    // sdpa_prepare_mask() does dim-reduce + fp16-cast + constant_pad_nd +
    // DDR copy; results invariant across all 18 expert layers within one
    // forward().
    // sdpa_dma_mask_to_spm() in the per-layer Phase 6 reads from this member;
    // member assignment also keeps DDR alive across the synchronous forward().
    PreparedMask prepared_mask_;

    // Cross-forward cache key for prepared_mask_.
    // forward() compares (input_data_ptr, seq_q, seq_k, is_causal) against
    // this cache; on hit, skips sdpa_prepare_mask entirely (5 → 1 per
    // inference). prepared_mask_input_ref_ is a strong ref to the caller's
    // last input tensor — keeps its storage pinned so a different tensor
    // cannot reuse the same data_ptr while our cache is valid.
    at::Tensor prepared_mask_input_ref_;
    int64_t    prepared_mask_seq_q_     = -1;
    int64_t    prepared_mask_seq_k_     = -1;
    bool       prepared_mask_is_causal_ = false;

    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;
};

}  // namespace v3

// =============================================================================
// Instance registry -- ModelHandleRegistry<v3::AdaRMSModel>.
// Handles are monotonic and are not reused; the Python adapter owns graph
// admission independently from this registry.
// =============================================================================

using AdaRMSRegistry = ModelHandleRegistry<v3::AdaRMSModel>;

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

int64_t rpu_adarms_create() {
    return AdaRMSRegistry::create();
}

void rpu_adarms_destroy(int64_t handle) {
    AdaRMSRegistry::destroy(handle, "rpu_adarms_destroy");
}

void rpu_adarms_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps,
    at::TensorList q_w_scale_list, at::TensorList k_w_scale_list,
    at::TensorList v_w_scale_list, at::TensorList o_w_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list)
{
    AdaRMSRegistry::get(handle, "rpu_adarms")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_list, up_list, down_list,
        attn_dense_w_list, attn_dense_b_list,
        mlp_dense_w_list,  mlp_dense_b_list,
        cos, sin,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps,
        q_w_scale_list, k_w_scale_list, v_w_scale_list, o_w_scale_list,
        gate_scale_list, up_scale_list, down_scale_list);
}

void rpu_adarms_set_rope_position(int64_t handle, int64_t position) {
    AdaRMSRegistry::get(handle, "rpu_adarms_set_rope_position")
        ->set_rope_position(position);
}

at::Tensor rpu_adarms_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal)
{
    // TensorList -> std::vector<at::Tensor> (shallow copy; tensors are refcounted)
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());

    return AdaRMSRegistry::get(handle, "rpu_adarms")->forward(
        hidden_states, cond, k_caches, v_caches,
        attention_mask, position, is_causal);
}
