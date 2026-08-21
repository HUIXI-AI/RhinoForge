// rpu_qwenpi05_model.cpp — QwenPI05 Action Expert all-layers-once model
//                           (v3 FusedModelBase port)
//
// QwenPI05 is structurally the closest analog of AdaRMS (flat-phase,
// SEQUENTIAL, per-forward cond + per-layer GEMV) PLUS q_norm/k_norm per-layer
// (Qwen3 pattern).
//
// Key differences vs AdaRMS:
//   - QK norm: per-head RMSNorm with [head_dim] weights (from Qwen3)
//   - RoPE: cos/sin passed per-forward (M-RoPE pre-computed on Python side)
//   - MLP activation: SiLU (SwiGLU) instead of GELU (GeGLU)
//   - Prefix KV: non-causal attention with VLM prefix cache; kv_insert_pos
//     = ctx().position + chunk.offset
//   - cos_sin_start = chunk.offset (pre-indexed M-RoPE, NOT ctx().position)
//   - Gated residual = SUB -> TANH -> 1xC->NxC MUL -> ADD (4 steps,
//     vs AdaRMS 3-step without TANH)
//
// Framework integration:
//   - Replace `buf(name)` at SDPA + KV-insert call sites with
//     `addr_offset(name).value`, using the typed SpmOffset contract.
//   - Access model params via pimpl getters.
//   - Class-member cond_ref_/cos_ref_/sin_ref_ assignments keep tensors alive for
//     the synchronous forward().
//   - set_weights ends with `invalidate_model_state();` as its last non-empty
//     statement.
//
// Execution scope:
//   - Per-forward cond DMA (layer 0 / chunk 0 gate) via cond_loaded_this_forward_.
//   - Per-layer GEMV (cond * dense_w + bias -> [scale|shift|gate]) at
//     chunk.idx == 0, result persists in SPM across chunks of this layer.
//   - (1+scale) preprocessing via scalar ADD to the live GEMV output SPM buffer.
//   - Gated residual 4-step: SUB full-size -> TANH gate slice -> 1xC->NxC MUL
//     with gate slice -> ADD full-size.
//   - Per-layer q_norm_w/k_norm_w DMA + RMSNorm before RoPE.
//   - Always SEQUENTIAL (no KV_FIRST); flat phases (all LayerWide).
//   - cfg.cross_layer_batch_size = num_layers() (force single group).
//   - Debug export flush machinery for per-core SPM dumps.
//
// Design contract: FusedModelBase.

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_helpers.h"
#include "rpu_spm_allocator.h"
#include "rpu_runtime_state.h"  // Shared runtime globals and accessors.

#include <ATen/ATen.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace v3 {

// =============================================================================
// QwenPI05Model — v3::FusedModelBase subclass (QwenPI05 Action Expert)
// =============================================================================

class QwenPI05Model : public FusedModelBase {
public:
    struct LayerWeights {
        // Attention
        at::Tensor q_w, k_w, v_w, o_w;
        // MLP
        at::Tensor gate_proj_w, up_proj_w, down_proj_w;
        // AdaRMS GEMV (per-layer)
        at::Tensor attn_dense_w;    // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor attn_dense_b;    // [3*hidden] fp16
        at::Tensor mlp_dense_w;     // [3*hidden, hidden] fp16, row-partition swizzle (K split 8 cores)
        at::Tensor mlp_dense_b;     // [3*hidden] fp16
        // QK norm (per-layer, new vs AdaRMS)
        at::Tensor q_norm_w;        // [head_dim] fp16
        at::Tensor k_norm_w;        // [head_dim] fp16
    };

    QwenPI05Model() = default;

    // ========================================================================
    // set_weights — accepts per-layer weights including QK norm
    //
    // Unlike AdaRMS, cos/sin are NOT stored here (they are per-forward).
    // QK norm weights (q_norm, k_norm) are per-layer [head_dim] fp16.
    // ========================================================================
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
        at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps)
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "qwenpi05_set_weights: empty weight lists");
        TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0
                    && hidden_size > 0 && intermediate_size > 0,
                    "qwenpi05_set_weights: dim params must be positive");
        TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                    "qwenpi05_set_weights: num_q_heads (", num_q_heads,
                    ") must be divisible by num_kv_heads (", num_kv_heads, ")");

        // All per-layer lists must have same length
        auto check_list = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "qwenpi05_set_weights: ", n, ".size()=", l.size(),
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
        check_list(q_norm_list,       "q_norm_list");
        check_list(k_norm_list,       "k_norm_list");

        // Per-layer defined/rank checks: 2D for linear weights, 1D for bias/norm
        auto check_rank = [&](const at::TensorList& list, const char* name,
                              int64_t expected_rank) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "qwenpi05_set_weights: ", name, "[", i, "] undefined");
                TORCH_CHECK(list[i].dim() == expected_rank,
                            "qwenpi05_set_weights: ", name, "[", i, "] must be ",
                            expected_rank, "D, got ", list[i].dim(), "D");
            }
        };
        check_rank(q_w_list,          "q_w_list",          2);
        check_rank(k_w_list,          "k_w_list",          2);
        check_rank(v_w_list,          "v_w_list",          2);
        check_rank(o_w_list,          "o_w_list",          2);
        check_rank(gate_list,         "gate_list",         2);
        check_rank(up_list,           "up_list",           2);
        check_rank(down_list,         "down_list",         2);
        check_rank(attn_dense_w_list, "attn_dense_w_list", 2);
        check_rank(attn_dense_b_list, "attn_dense_b_list", 1);
        check_rank(mlp_dense_w_list,  "mlp_dense_w_list",  2);
        check_rank(mlp_dense_b_list,  "mlp_dense_b_list",  1);
        check_rank(q_norm_list,       "q_norm_list",       1);
        check_rank(k_norm_list,       "k_norm_list",       1);

        // Commit model params (pimpl: hidden/q/kv/head_dim/intermediate + attn_tp_)
        set_model_params(num_q_heads, num_kv_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;

        // Cached derived dims (v3 framework's set_model_params already sets
        // pimpl attn_tp_ = std::min(8, num_kv_heads), matching v2's
        // std::min(NUM_CORES, num_kv_heads) — cache here for readability
        // inside the hot build_layer_subgraph path).
        local_q_heads_ = num_q_heads / attn_tp();
        local_kv_dim_  = num_kv_heads * head_dim / attn_tp();

        // Copy weights into layer state
        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                gate_list[i], up_list[i], down_list[i],
                attn_dense_w_list[i], attn_dense_b_list[i],
                mlp_dense_w_list[i],  mlp_dense_b_list[i],
                q_norm_list[i], k_norm_list[i],
            });
        }

        invalidate_model_state();  // Last non-empty statement of set_weights.
    }

    // ========================================================================
    // forward — cond + cos/sin are per-forward arguments
    //
    // Unlike AdaRMS which stores cos/sin in set_weights, QwenPI05 receives
    // cos/sin per-forward (M-RoPE pre-computed on Python side).
    // Three tensors kept alive via class members: cond_ref_, cos_ref_, sin_ref_.
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& hidden_states,
        const at::Tensor& cond,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& cos,
        const at::Tensor& sin,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal)
    {
        TORCH_CHECK(num_layers() > 0,
                    "QwenPI05Model::forward called before set_weights");
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "QwenPI05Model::forward: hidden_states must be on RPU device");
        TORCH_CHECK(hidden_states.is_contiguous(),
                    "QwenPI05Model::forward: hidden_states must be contiguous "
                    "(downstream kernels use raw data_ptr())");
        TORCH_CHECK(position >= 0,
                    "QwenPI05Model::forward: position must be non-negative, got ", position);

        // cond validation (mirrors AdaRMS:224-240 — cross-boundary tensors
        // consumed as raw c10::Half* in the graph MUST be validated at the
        // forward entry; shape checks alone are insufficient.)
        TORCH_CHECK(cond.defined() && cond.dim() == 1 &&
                    cond.size(0) == hidden_size(),
                    "QwenPI05Model::forward: cond must be 1D [hidden_size=",
                    hidden_size(), "], got dim=", (cond.defined() ? cond.dim() : -1),
                    " size=", (cond.defined() ? cond.size(0) : -1));
        TORCH_CHECK(cond.device().type() == at::kPrivateUse1,
                    "QwenPI05Model::forward: cond must be on RPU device");
        TORCH_CHECK(cond.scalar_type() == at::kHalf,
                    "QwenPI05Model::forward: cond must be fp16");
        TORCH_CHECK(cond.is_contiguous(),
                    "QwenPI05Model::forward: cond must be contiguous "
                    "(DMA source pointer assumes row-major layout)");

        // cos/sin validation — consumed as raw c10::Half* in rotary kernels.
        TORCH_CHECK(cos.defined() && sin.defined(),
                    "QwenPI05Model::forward: cos/sin must be defined");
        TORCH_CHECK(cos.device().type() == at::kPrivateUse1 &&
                    sin.device().type() == at::kPrivateUse1,
                    "QwenPI05Model::forward: cos/sin must be on RPU device");
        TORCH_CHECK(cos.scalar_type() == at::kHalf &&
                    sin.scalar_type() == at::kHalf,
                    "QwenPI05Model::forward: cos/sin must be fp16");
        TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
                    "QwenPI05Model::forward: cos/sin must be contiguous "
                    "(rotary kernel assumes row-major layout)");
        TORCH_CHECK(cos.sizes() == sin.sizes(),
                    "QwenPI05Model::forward: cos/sin shapes must match, got cos=",
                    cos.sizes(), " sin=", sin.sizes());
        // The rotary
        // kernel reads cos_ref_/sin_ref_ as raw c10::Half* starting at
        // `cos_sin_start = chunk.offset` for `seq_len` rows × head_dim/2
        // columns; the DDR table contract is [max_seq, head_dim/2].
        //
        // Only accept head_dim/2. Accepting either half- or full-dim is incorrect — the kernel
        // consumes raw data_ptr with stride = head_dim/2 fp16 elements per
        // row. A full-dim caller would still reach kernel dispatch (validation
        // passes) but misindex positions. The canonical fused producer always
        // returns [suffix_len, head_dim/2]. Any caller that wants to
        // pass full-dim must slice to half-dim before the call.
        TORCH_CHECK(cos.dim() == 2,
                    "QwenPI05Model::forward: cos must be 2D [max_pos, head_dim/2], got dim=",
                    cos.dim());
        TORCH_CHECK(sin.dim() == 2,
                    "QwenPI05Model::forward: sin must be 2D [max_pos, head_dim/2], got dim=",
                    sin.dim());
        TORCH_CHECK(cos.size(1) == head_dim() / 2,
                    "QwenPI05Model::forward: cos last-dim must equal head_dim/2=",
                    head_dim() / 2, " (kernel DDR table contract); got ",
                    cos.size(1),
                    ". If the caller has full-dim [max_pos, head_dim] tables, "
                    "slice to half-dim before .forward(): "
                    "cos_half = cos[..., :head_dim//2].contiguous().");
        TORCH_CHECK(cos.size(0) >= hidden_states.size(1),
                    "QwenPI05Model::forward: cos row count (", cos.size(0),
                    ") must cover input seq_len (", hidden_states.size(1),
                    ") — M-RoPE is pre-indexed by chunk.offset up to seq_len-1");

        // QwenPI05 always auto-computes chunk_size (not affected by global override)
        set_chunk_size_override(0);

        // Reset per-forward cond state: a new denoise step is a new cond.
        // cond_ref_/cos_ref_/sin_ref_ keep the tensors alive for the
        // synchronous forward (run_all_layers batch end executes before
        // return); cond DDR pointer cursor-patched per REPLAY via mutable DMA.
        cond_loaded_this_forward_ = false;
        cond_ref_ = cond;
        cos_ref_ = cos;
        sin_ref_ = sin;
        // cond_src_base_ feeds rpu_launch_ddr_scatter_spm_dma_mutable in the
        // cond scatter site below; REPLAY end() reads this live address. Per-
        // forward flush mirrors fused_model_base's hidden_in_src_base_ pattern
        // (caller may have CPU-dirty cache lines from an upstream op).
        cond_src_base_ = ::rhino_lkn::RpuGetDevAddr(cond.data_ptr());
        rpu_ddr_flush_force(cond.data_ptr<c10::Half>());

        return run_all_layers(hidden_states, k_caches, v_caches,
                              attention_mask, position, is_causal);
    }

    // ========================================================================
    // debug_flush_dumps — materialize per-core SPM reads after forward.
    // Called by Python via qwenpi05_debug_flush_dumps op after a debug-export
    // forward finishes, to populate g_debug_tensors.
    // ========================================================================
    void debug_flush_dumps() {
        for (const auto& req : debug_dump_requests_) {
            at::Tensor t = at::empty({req.num_cores, req.elems_per_core},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
            c10::Half* dst = t.data_ptr<c10::Half>();
            for (int c = 0; c < req.num_cores; ++c) {
                const void* src = SPM_ALLOC.cpu_ptr(c, req.buf_off);
                std::memcpy(dst + c * req.elems_per_core, src,
                            req.elems_per_core * sizeof(c10::Half));
            }
            g_debug_tensors[req.name] = t;
        }
        debug_dump_requests_.clear();
    }

protected:
    // ========================================================================
    // static_config — graph-naive flat-phase subclass; no legacy graph slot.
    //
    // QwenPI05 is a flat-phase, SEQUENTIAL subclass — like AdaRMS. All four
    // Optional pointer-to-member slots (preload_fn / kv_first_fn /
    // kv_first_chunk_plan_fn / post_fn) stay nullptr by default. No persistent
    // weights, no KV_FIRST, no post-graph.
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers       = num_layers();
        // Pin to a single fused graph regardless of the global
        // `g_cross_layer_batch_size` (which is shared with Qwen3 text
        // decode where callers sometimes set it to 7 / 12 / 36). For
        // QwenPI05 action expert we always want all layers in one graph
        // replay per DDIM step — setting it equal to num_layers() makes
        // `compute_num_groups` in FusedModelBase return 1 unconditionally.
        cfg.cross_layer_batch_size = num_layers();
        // Callbacks are all nullptr for QwenPI05 (no preload / KV_FIRST / post).
        return cfg;
    }

    // ========================================================================
    // dynamic_config — always SEQUENTIAL
    // ========================================================================
    ModelDynamicConfig dynamic_config(const ChunkPlan& /*plan*/) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::AUTO;
        return cfg;
    }

    // ========================================================================
    // declare_buffers — flat phases, ALL LayerWide
    //
    // Same as AdaRMS but adds q_norm_w and k_norm_w buffers for QK norm plus
    // debug-snapshot buffers allocated unconditionally so the graph layout
    // stays stable across debug/non-debug invocations.
    //
    // SPM budget (hidden=1024, intermediate=3072, heads=16, kv=8, head_dim=128):
    //   attn_tp = min(8, 8) = 8
    //   local_q_heads = 16/8 = 2
    //   local_kv_dim = 8*128/8 = 128
    //   residual ~ 2KB, q ~ 512B, kv ~ 256B, oproj ~ 2KB, mlp ~ 768B
    //   cond ~ 2KB, gemv ~ 6KB, q/k_norm_w ~ 256B each
    //   Total per core ~ 15KB << 8109 KB SPM budget
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

        // QK norm weights (new vs AdaRMS)
        int64_t norm_w_sz = A(hd * DWIDTH);

        constexpr BufferScope ALL = BufferScope::LayerWide;
        int64_t nl = num_layers();

        std::vector<BufferDecl> decls;

        // Structural (mirrors AdaRMS, flat phases 0,0)
        decls.push_back({"residual1",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"input_norm",   res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"residual2",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"q",            q,       0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"k",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"v",            kv,      0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"output",       out,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"oproj",        oproj,   0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"sdpa_tmp",     tmp,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"sdpa_mask",    mask_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"gate",         mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"up",           mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"down",         res,     0, 0, StorageClass::Temp, 0, nullptr, ALL});

        // AdaRMS-specific
        decls.push_back({"cond",         cond_sz,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"attn_gemv",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"mlp_gemv",     gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"bias_temp",    gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"gemv_partial", gemv_out_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});

        // DEBUG snapshot buffers (only used when get_debug_export() is true;
        // allocated unconditionally so the graph layout stays stable).
        // These capture Q/K at Phase 3 (prenorm) and Phase 3b (post-norm)
        // without conflicting with later phases that reuse q/k/output/input_norm.
        decls.push_back({"dbg_q_prenorm", q,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_k_prenorm", kv,    0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_q_norm",    q,     0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_k_norm",    kv,    0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_sdpa_out",  out,   0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_oproj_out", oproj, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        decls.push_back({"dbg_layer_out", res,   0, 0, StorageClass::Temp, 0, nullptr, ALL});

        // ─────────────────────────────────────────────────────────────────────
        // QK norm weights use PersistentPerLayer preload callbacks.
        // ─────────────────────────────────────────────────────────────────────
        {
            BufferDecl q_norm;
            q_norm.name      = "q_norm_w";
            q_norm.size      = norm_w_sz;
            q_norm.storage   = StorageClass::PersistentPerLayer;
            q_norm.per_layer = nl;
            q_norm.scope     = BufferScope::LayerWide;
            q_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        layer_weights_[L].q_norm_w.data_ptr<c10::Half>(),
                        head_dim(), core0_addr);
                };
            decls.push_back(q_norm);
        }
        {
            BufferDecl k_norm;
            k_norm.name      = "k_norm_w";
            k_norm.size      = norm_w_sz;
            k_norm.storage   = StorageClass::PersistentPerLayer;
            k_norm.per_layer = nl;
            k_norm.scope     = BufferScope::LayerWide;
            k_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        layer_weights_[L].k_norm_w.data_ptr<c10::Half>(),
                        head_dim(), core0_addr);
                };
            decls.push_back(k_norm);
        }

        return decls;
    }

    // ========================================================================
    // build_layer_subgraph — complete 16-operation per-layer pipeline
    //
    // Pipeline: cond DMA -> GEMV -> input DMA -> AdaRMSNorm(attn) -> QKV ->
    //   QKNorm -> RoPE -> KV insert -> SDPA -> OProj -> all_reduce_sum_residual
    //   -> gated attn residual(4-step) -> AdaRMSNorm(MLP) -> MLP(SiLU) ->
    //   gated MLP residual(4-step) -> output DMA
    //
    // Key differences from AdaRMS:
    //   - cos_sin_start = chunk.offset (pre-indexed M-RoPE, NOT ctx().position)
    //   - kv_insert_pos = ctx().position + chunk.offset (absolute cache position)
    //   - QK Norm before RoPE (Qwen3 pattern)
    //   - Gated residual = SUB -> TANH -> broadcast MUL -> ADD (4 steps, not 3)
    //   - ActivationKind::SILU (not GELU)
    //   - cos_ref_/sin_ref_ DDR pointers (not cos_/sin_)
    //
    // Pitfall 3 structural fix: SDPA + KV-insert use addr_offset(name).value
    // (typed SpmOffset, compile-time distinct from absolute addr() uint32_t).
    // ========================================================================
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t cos_sin_start = chunk.offset;  // pre-indexed by Python M-RoPE
        int64_t kv_insert_pos = ctx().position + chunk.offset;  // absolute cache pos
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
        //
        // Row-partition GEMV consumes the cond as a per-core K-slice:
        // core c needs cond[c*H/8 : (c+1)*H/8] at its SPM base offset.
        // The GEMV kernel does NOT stride into a replicated buffer; it
        // just reads `local_k = H/8` elements from `core_base + cond_addr`
        // on each core. So we SCATTER cond from DDR to per-core SPM
        // (core c reads cond[c*H/8:(c+1)*H/8] from DDR and writes to the
        // SAME local SPM offset on its own core).
        //
        // A broadcast DMA would make every core read cond[0:H/8]. Scatter is
        // required so core c receives cond[c*H/8:(c+1)*H/8].
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
        // Phase A: per-layer GEMV (chunk 0 only)
        // Computes [scale | shift | gate] from cond * dense_w + bias, with
        // +1.0 baked into the scale slice. Result persists in SPM for this
        // layer's remaining chunks.
        // --------------------------------------------------------------------
        if (is_chunk_zero) {
            adarms_gemv_to_spm(addr(0, "cond"),
                               lw.attn_dense_w, lw.attn_dense_b,
                               addr(0, "attn_gemv"), addr(0, "bias_temp"),
                               addr(0, "gemv_partial"));
            adarms_gemv_to_spm(addr(0, "cond"),
                               lw.mlp_dense_w, lw.mlp_dense_b,
                               addr(0, "mlp_gemv"), addr(0, "bias_temp"),
                               addr(0, "gemv_partial"));
        }

        // GEMV slice offsets (computed from core 0 base address; broadcast
        // DMA means every core shares the same offsets).
        uint32_t attn_scale = addr(0, "attn_gemv");
        uint32_t attn_shift = attn_scale + (uint32_t)(h * DWIDTH);
        uint32_t attn_gate  = attn_scale + (uint32_t)(h * DWIDTH * 2);
        uint32_t mlp_scale  = addr(0, "mlp_gemv");
        uint32_t mlp_shift  = mlp_scale + (uint32_t)(h * DWIDTH);
        uint32_t mlp_gate   = mlp_scale + (uint32_t)(h * DWIDTH * 2);

        // --------------------------------------------------------------------
        // Phase 1: layer input DMA
        // --------------------------------------------------------------------
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        // --------------------------------------------------------------------
        // Phase 2: AdaRMSNorm (attn): residual1 -> input_norm
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
        // Phase 3: QKV Linear (col partition)
        // --------------------------------------------------------------------
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h,
            /*partition=*/1, tp);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nkv * hd, h,
            /*partition=*/1, tp);

        // DEBUG EXPORT (layer 0): Phase 3 Q/K prenorm snapshot.
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            debug_dump_spm_per_core("qwenpi05_cond_L0",
                addr(0, "cond"), h, NUM_CORES,
                addr_offset("cond").value);
            debug_dump_spm_per_core("qwenpi05_qnormw_L0",
                addr(0, "q_norm_w"), hd, NUM_CORES,
                addr_offset("q_norm_w").value);
            debug_dump_spm_per_core("qwenpi05_residual1_L0",
                addr(0, "residual1"), seq_len * h, NUM_CORES,
                addr_offset("residual1").value);
            debug_dump_spm_per_core("qwenpi05_input_norm_L0",
                addr(0, "input_norm"), seq_len * h, NUM_CORES,
                addr_offset("input_norm").value);

            // In-graph copy: dbg_q_prenorm <- q
            int64_t q_elems_per_core = seq_len * local_q_heads_ * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "q"), c10::Half(0.0), addr(0, "dbg_q_prenorm"),
                q_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("qwenpi05_q_prenorm_L0",
                addr(0, "dbg_q_prenorm"),
                q_elems_per_core, tp,
                addr_offset("dbg_q_prenorm").value);

            int64_t local_kv_heads_dbg = nkv / tp;
            int64_t k_elems_per_core = seq_len * local_kv_heads_dbg * hd;
            // In-graph copy: dbg_k_prenorm <- k
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "k"), c10::Half(0.0), addr(0, "dbg_k_prenorm"),
                k_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("qwenpi05_k_prenorm_L0",
                addr(0, "dbg_k_prenorm"),
                k_elems_per_core, tp,
                addr_offset("dbg_k_prenorm").value);

            // V buffer — Phase 3 V linear output, never modified after Phase 5
            // V-cache insert.
            debug_dump_spm_per_core("qwenpi05_v_L0",
                addr(0, "v"),
                k_elems_per_core, tp,
                addr_offset("v").value);
        }

        // --------------------------------------------------------------------
        // Phase 3b: QK Norm (Qwen3 pattern: RMSNorm before RoPE)
        // --------------------------------------------------------------------
        int64_t local_kv_heads = nkv / tp;
        // q_norm_w / k_norm_w are PersistentPerLayer (see declare_buffers).
        // The .preload_callback DMA'd them at BUILD — we just read the slot.
        // Q norm: q -> output
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "q"), addr(0, "output"),
            layer_addr(layer_idx, 0, "q_norm_w"),
            seq_len * local_q_heads_, hd, eps_);
        // K norm: k -> input_norm
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "k"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "k_norm_w"),
            seq_len * local_kv_heads, hd, eps_);

        // DEBUG EXPORT (layer 0, chunk 0 only): snapshot Q norm and K norm
        // output into dedicated buffers BEFORE Phase 4 RoPE / Phase 6 SDPA
        // overwrite `q`/`output`/`input_norm`.
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            int64_t q_elems_per_core = seq_len * local_q_heads_ * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "output"), c10::Half(0.0), addr(0, "dbg_q_norm"),
                q_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("qwenpi05_q_norm_L0",
                addr(0, "dbg_q_norm"),
                q_elems_per_core, tp,
                addr_offset("dbg_q_norm").value);

            int64_t k_elems_per_core = seq_len * local_kv_heads * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "input_norm"), c10::Half(0.0), addr(0, "dbg_k_norm"),
                k_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("qwenpi05_k_norm_L0",
                addr(0, "dbg_k_norm"),
                k_elems_per_core, tp,
                addr_offset("dbg_k_norm").value);
        }

        // --------------------------------------------------------------------
        // Phase 4: RoPE (per-forward cos_ref_/sin_ref_, start=chunk.offset)
        // --------------------------------------------------------------------
        rpu_launch_rope_spm_kernel(
            addr(0, "output"), addr(0, "q"),
            cos_ref_.data_ptr<c10::Half>(), sin_ref_.data_ptr<c10::Half>(),
            seq_len, local_q_heads_, hd, cos_sin_start);
        rpu_launch_rope_spm_kernel(
            addr(0, "input_norm"), addr(0, "k"),
            cos_ref_.data_ptr<c10::Half>(), sin_ref_.data_ptr<c10::Half>(),
            seq_len, local_kv_heads, hd, cos_sin_start);

        // DEBUG EXPORT (layer 0, chunk 0 only): dump Q/K after RoPE
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            debug_dump_spm_per_core("qwenpi05_q_rope_L0",
                addr(0, "q"),
                seq_len * local_q_heads_ * hd, tp,
                addr_offset("q").value);
            debug_dump_spm_per_core("qwenpi05_k_rope_L0",
                addr(0, "k"),
                seq_len * local_kv_heads * hd, tp,
                addr_offset("k").value);
        }

        // --------------------------------------------------------------------
        // Phase 5: KV cache insert (absolute position, NOT cos_sin_start).
        //
        // Pitfall 3 structural fix: KV-insert takes SPM OFFSETS, not absolute
        // addresses. addr_offset("name").value is compile-time distinct from
        // addr(core, "name").
        // --------------------------------------------------------------------
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        rpu_launch_insert_kcache_spm_unified(
            k_cache, kv_insert_pos, addr_offset("k").value,
            seq_len, nkv, hd, tp);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, kv_insert_pos, addr_offset("v").value,
            seq_len, nkv, hd, tp);

        // --------------------------------------------------------------------
        // Phase 6: SDPA (2D mask from Python, non-causal for prefix KV).
        //
        // Pitfall 3 structural fix: SDPA args (q / output / sdpa_tmp /
        // sdpa_mask) take SPM offsets via addr_offset(...).value.
        // --------------------------------------------------------------------
        int64_t kv_seq_len = ctx().position + ctx().seq_len;
        bool sdpa_causal = ctx().is_causal && (seq_len > 1);
        int mask_type = sdpa_load_mask_to_spm(
            ctx().attention_mask, sdpa_causal,
            addr_offset("sdpa_mask").value, seq_len, kv_seq_len, tp,
            /*out_mask_ddr_ref=*/nullptr, sdpa_stable_mask_cache());
        rpu_launch_sdpa_spm_unified_kernel_v2(
            k_cache, v_cache, mask_type, c10::nullopt,
            addr_offset("q").value,
            addr_offset("output").value,
            addr_offset("sdpa_tmp").value,
            addr_offset("sdpa_mask").value,
            seq_len, nq, nkv, hd,
            kv_seq_len, tp, NUM_CORES);

        // DEBUG EXPORT: snapshot SDPA output before O_proj consumes it
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            int64_t q_elems_per_core = seq_len * local_q_heads_ * hd;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "output"), c10::Half(0.0), addr(0, "dbg_sdpa_out"),
                q_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("qwenpi05_sdpa_out_L0",
                addr(0, "dbg_sdpa_out"),
                q_elems_per_core, tp,
                addr_offset("dbg_sdpa_out").value);
        }

        // --------------------------------------------------------------------
        // Phase 7: O_proj (row partition)
        // --------------------------------------------------------------------
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "output"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd,
            /*partition=*/0, tp);

        // DEBUG EXPORT: snapshot O_proj per-core output before all_reduce
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            int64_t oproj_elems_per_core = seq_len * h;
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "oproj"), c10::Half(0.0), addr(0, "dbg_oproj_out"),
                oproj_elems_per_core, ValuOpType::ADD);
            debug_dump_spm_per_core("qwenpi05_oproj_out_L0",
                addr(0, "dbg_oproj_out"),
                oproj_elems_per_core, NUM_CORES,
                addr_offset("dbg_oproj_out").value);
        }

        // --------------------------------------------------------------------
        // Phase 8: all_reduce_sum_residual
        // residual2 = reduce_sum(oproj) + residual1
        // --------------------------------------------------------------------
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
            seq_len, h, tp, NUM_CORES);

        // --------------------------------------------------------------------
        // Phase 9: Gated attn residual (4-step SUB -> TANH -> MUL -> ADD)
        // After all_reduce_sum_residual: residual2 = reduce(oproj) + residual1
        // Goal: residual2 = residual1 + tanh(attn_gate) * reduce(oproj)
        //
        // SUB: residual2 = residual2 - residual1 = reduce(oproj)
        // --------------------------------------------------------------------
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
            num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
        // TANH: attn_gate = tanh(attn_gate) in-place [1,H]
        rpu_launch_eltwise_unary_spm_kernel(
            attn_gate, attn_gate,
            h, ValuOpType::TANH,
            /*is_gelu=*/false, /*num_cores=*/NUM_CORES);
        // MUL: residual2 = attn_gate [1,H] * residual2 [S,H]
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            attn_gate, addr(0, "residual2"), addr(0, "residual2"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
        // ADD: residual2 = residual2 + residual1 (gated result in residual2)
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);

        // --------------------------------------------------------------------
        // Phase 10: AdaRMSNorm (MLP): residual2 -> residual1
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
        // Phase 11: MLP pipeline (SiLU, NOT GELU)
        // emit_mlp_pipeline: reads residual1, writes residual1 = reduce(down) + residual2
        // --------------------------------------------------------------------
        emit_mlp_pipeline(lw.gate_proj_w, lw.up_proj_w, lw.down_proj_w,
                          seq_len, ActivationKind::SILU);

        // --------------------------------------------------------------------
        // Phase 13: Gated MLP residual (4-step SUB -> TANH -> MUL -> ADD)
        // After MLP: residual1 = reduce(down) + residual2
        // Goal: residual1 = residual2 + tanh(mlp_gate) * reduce(down)
        // --------------------------------------------------------------------
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
            num_elems_full, ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
        // TANH: mlp_gate = tanh(mlp_gate) in-place [1,H]
        rpu_launch_eltwise_unary_spm_kernel(
            mlp_gate, mlp_gate,
            h, ValuOpType::TANH,
            /*is_gelu=*/false, /*num_cores=*/NUM_CORES);
        // MUL: residual1 = mlp_gate [1,H] * residual1 [S,H]
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            mlp_gate, addr(0, "residual1"), addr(0, "residual1"),
            seq_len, h,
            c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
        // ADD: residual1 = residual1 + residual2 (final gated MLP result)
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0), NUM_CORES);

        // DEBUG EXPORT: snapshot per-layer final output (residual1)
        if (layer_idx == 0 && chunk.idx == 0 && get_debug_export()) {
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "residual1"), c10::Half(0.0), addr(0, "dbg_layer_out"),
                num_elems_full, ValuOpType::ADD);
            debug_dump_spm_per_core("qwenpi05_layer_out_L0",
                addr(0, "dbg_layer_out"),
                num_elems_full, NUM_CORES,
                addr_offset("dbg_layer_out").value);
        }

        // --------------------------------------------------------------------
        // Phase 14: Output DMA
        // --------------------------------------------------------------------
        if (!ctx().output_to_spm) {
            emit_layer_output_dma(layer_idx, chunk);
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
    // holds K/8=H/8 columns of all 3H output rows. Layout of the final output
    // (on every core after reduce): [scale(hidden) | shift(hidden) | gate(hidden)].
    // The scale portion gets +1.0 in place so later RMSNorm sees (1+scale)
    // directly.
    //
    // Three distinct SPM buffers are required (all_reduce_sum_residual expects
    // input, residual, and output to be non-overlapping SPM regions):
    //   gemv_out_spm_addr     : final reduced+biased [3H] (broadcast to all cores)
    //   bias_temp_spm_addr    : DMA'd bias, used as the reduce-residual [3H]
    //   partial_spm_addr      : per-core partial [3H] from the row-partition GEMV
    //
    // Steps:
    //   1. DMA bias broadcast -> bias_temp (same bias on all 8 cores)
    //   2. GEMV (row partition, 8 cores): cond * dense_w -> partial
    //      Each core computes a partial [3H] along its local_k=H/8 slice.
    //   3. all_reduce_sum_residual(partial, bias) -> gemv_out
    //      Broadcast full [3H] to all cores.
    //   4. scale_slice += 1.0 (only first hidden_size elements)
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
        // Step 2: Keep the unsupported legacy escape hatch on its established
        // GEMM route. The public fp16 wrapper now supports this local_k, but
        // QwenPI05 has no maintained model-level parity gate; switching it would
        // be an unverified compatibility change. partial_spm_addr holds the
        // per-core [3H] partial along the local_k = H/NUM_CORES split.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, /*num_cores=*/NUM_CORES,
            /*bias_spm_addr=*/0,
            /*force_gemm=*/true);
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

    // ========================================================================
    // debug_dump_spm_per_core — record metadata for a per-core SPM dump.
    //
    // The actual readback happens OUTSIDE the graph replay. We direct-map
    // per-core SPM via SPM_ALLOC.cpu_ptr(core, buf_off) and memcpy into an
    // at::Tensor. This avoids any kernel-level scatter and is a pure
    // post-execution operation.
    //
    // Call `debug_flush_dumps()` from the Python side after the forward
    // completes and before the next forward.
    // ========================================================================
    struct DebugDumpRequest {
        std::string name;
        uint32_t buf_off;
        int64_t elems_per_core;
        int num_cores;
    };
    std::vector<DebugDumpRequest> debug_dump_requests_;

    void debug_dump_spm_per_core(const std::string& name,
                                 uint32_t /*spm_core0_addr*/,
                                 int64_t elems_per_core,
                                 int num_cores,
                                 uint32_t buf_off)
    {
        if (!get_debug_export()) return;
        DebugDumpRequest req;
        req.name = name;
        req.buf_off = buf_off;
        req.elems_per_core = elems_per_core;
        req.num_cores = num_cores;
        debug_dump_requests_.push_back(req);
    }

    // ----- Model state -----
    std::vector<LayerWeights> layer_weights_;
    double eps_ = 1e-6;

    // Derived dims (computed in set_weights, cached for hot path)
    int64_t local_q_heads_ = 0;
    int64_t local_kv_dim_  = 0;

    // Per-forward tensor refs (kept alive for graph execution/replay via
    // class-member assignment; run_all_layers is synchronous so refcount
    // covers the full dispatch window). `cond_src_base_` is the live RPU
    // dev addr of cond_ref_'s data_ptr — feeds the mutable-DMA cond scatter
    // so REPLAY end() picks up the per-forward base from this storage.
    at::Tensor cond_ref_;
    uint64_t   cond_src_base_ = 0;
    at::Tensor cos_ref_;
    at::Tensor sin_ref_;
    bool cond_loaded_this_forward_ = false;

    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;
};

}  // namespace v3

// =============================================================================
// Instance registry — uses ModelHandleRegistry<v3::QwenPI05Model> template
// =============================================================================

using QwenPI05Registry = ModelHandleRegistry<v3::QwenPI05Model>;

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (file-scope, not namespaced)
// =============================================================================

int64_t rpu_qwenpi05_create() {
    return QwenPI05Registry::create();
}

void rpu_qwenpi05_destroy(int64_t handle) {
    QwenPI05Registry::destroy(handle, "rpu_qwenpi05_destroy");
}

void rpu_qwenpi05_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    at::TensorList attn_dense_w_list, at::TensorList attn_dense_b_list,
    at::TensorList mlp_dense_w_list,  at::TensorList mlp_dense_b_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps)
{
    QwenPI05Registry::get(handle, "rpu_qwenpi05")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        gate_list, up_list, down_list,
        attn_dense_w_list, attn_dense_b_list,
        mlp_dense_w_list,  mlp_dense_b_list,
        q_norm_list, k_norm_list,
        num_q_heads, num_kv_heads, head_dim,
        hidden_size, intermediate_size, eps);
}

at::Tensor rpu_qwenpi05_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal)
{
    // TensorList -> std::vector<at::Tensor> (shallow copy; tensors are refcounted)
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());

    return QwenPI05Registry::get(handle, "rpu_qwenpi05")->forward(
        hidden_states, cond, k_caches, v_caches,
        cos, sin, attention_mask, position, is_causal);
}

// Post-forward helper: materialize per-core SPM dumps requested during the
// graph build (only active when debug export is enabled). This MUST be
// called after a forward finishes; it reads SPM via direct CPU-mapped
// pointers and populates g_debug_tensors, which Python reads via
// rpu_backend.get_debug_tensor(name).
void rpu_qwenpi05_debug_flush_dumps(int64_t handle) {
    QwenPI05Registry::get(handle, "rpu_qwenpi05_debug_flush_dumps")->debug_flush_dumps();
}
