// rpu_gemma4_model.cpp — Gemma4 E4B text decoder, all-layers-once (v3 FusedModelBase)
//
// Gemma4 is a mixed-geometry fused subsystem: per-layer head_dim
// (sliding 256 / global 512), a four-norm decoder block, QK-RMSNorm, v_norm,
// sliding-window mask, cross-layer KV-share, and PLE side-input.
// Model constraints:
//   * RMSNorm is DIRECT-WEIGHT — norm preload callbacks DMA the raw weight and
//     do not apply an additive offset.
//   * SDPA qk_scale = 1.0 — NOT 1/sqrt(head_dim), NOT query_pre_attn_scalar.
//   * Per-layer head_dim: base set_model_params() gets head_dim_max (512) so
//     declare_buffers sizes attention buffers wide enough for both widths;
//     build_layer_subgraph passes each layer's REAL head_dim to rope/SDPA.
//
// Supported implementation:
//   * RoPE uses NeoX d/2 with half-width cos/sin tables.
//   * v_norm is an unweighted RMS over V with an all-ones weight.
//   * Sliding hd256 layers read only the aligned windowed KV range and apply a
//     compact MASK_2D. Decode uses a 16-token bucket-stable window so cached KV
//     offsets and mask DMA sizes remain stable. Global hd512 layers retain the
//     full-causal LTM path.
//   * PLE injects the per-layer side input and layer_scalar.
//   * Cross-layer KV sharing routes SDPA through geom.kv_source_layer.
//
// Framework contract: src/core/fused_model_base.h

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_spm_buffers.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_spm_allocator.h"
#include "rpu_runtime_state.h"
#include <c10/util/Half.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>   // -inf fill for the windowed mask
#include <utility>  // windowed_mask_cache_ key
#include "rpu_bounded_shape_map.h"

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace v3 {

// =============================================================================
// Gemma4Model — v3::FusedModelBase subclass (Gemma4 E4B text decoder)
// =============================================================================

class Gemma4Model : public FusedModelBase {
public:
    // Per-layer weights. k_w / v_w / k_norm_w are placeholders (undefined or a
    // dummy) for KV-shared layers (idx >= first_shared) and are never read by
    // build_layer_subgraph — those layers reuse the source layer's K/V slot.
    struct LayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor q_norm_w, k_norm_w;
        at::Tensor input_norm_w, post_attn_norm_w;
        at::Tensor pre_ff_norm_w, post_ff_norm_w;
        at::Tensor gate_w, up_w, down_w;
        // PLE: per_layer_input_gate / per_layer_projection (both
        // row-swizzled, num_cores=8) + post_per_layer_input_norm. Defined only
        // when has_ple_ (ple_*_list non-empty); else default (unused).
        at::Tensor ple_gate_w, ple_proj_w, ple_post_norm_w;
        double layer_scalar = 1.0;   // per-layer output scalar
    };

    // Per-layer mixed geometry, mirrored by adapters/gemma4/geometry.py.
    struct LayerGeom {
        int64_t head_dim;        // 256 sliding / 512 global
        int64_t num_kv_heads;    // 2 (E4B)
        int64_t kv_source_layer; // K/V slot SDPA reads (== idx if not shared)
        bool    is_global;
        bool    is_kv_shared;
    };

    Gemma4Model() = default;

    // ========================================================================
    // set_weights
    // ========================================================================
    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        at::TensorList input_norm_list, at::TensorList post_attn_norm_list,
        at::TensorList pre_ff_norm_list, at::TensorList post_ff_norm_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        const at::Tensor& cos_sliding, const at::Tensor& sin_sliding,
        const at::Tensor& cos_global,  const at::Tensor& sin_global,
        const at::Tensor& final_norm_w,
        at::TensorList ple_gate_list, at::TensorList ple_proj_list,
        at::TensorList ple_post_norm_list, const at::Tensor& layer_scalar,
        at::IntArrayRef layer_type_ids,
        at::IntArrayRef head_dim_per_layer,
        at::IntArrayRef num_kv_heads_per_layer,
        at::IntArrayRef kv_source_layer,
        at::IntArrayRef is_kv_shared,
        int64_t num_q_heads, int64_t num_kv_heads_max, int64_t head_dim_max,
        int64_t hidden_size, int64_t intermediate_size, int64_t ple_dim,
        double eps, double qk_scale, int64_t sliding_window)
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "gemma4_set_weights: empty weight lists");
        TORCH_CHECK(num_q_heads > 0 && num_kv_heads_max > 0 && head_dim_max > 0
                    && hidden_size > 0 && intermediate_size > 0,
                    "gemma4_set_weights: model dim params must be positive");
        TORCH_CHECK(num_q_heads % num_kv_heads_max == 0,
                    "gemma4_set_weights: num_q_heads must be divisible by num_kv_heads");

        auto check_len = [&](size_t sz, const char* name) {
            TORCH_CHECK(static_cast<int64_t>(sz) == N,
                        "gemma4_set_weights: ", name, ".size()=", sz,
                        " != num_layers=", N);
        };
        check_len(k_w_list.size(),  "k_w_list");
        check_len(v_w_list.size(),  "v_w_list");
        check_len(o_w_list.size(),  "o_w_list");
        check_len(q_norm_list.size(),        "q_norm_list");
        check_len(k_norm_list.size(),        "k_norm_list");
        check_len(input_norm_list.size(),    "input_norm_list");
        check_len(post_attn_norm_list.size(),"post_attn_norm_list");
        check_len(pre_ff_norm_list.size(),   "pre_ff_norm_list");
        check_len(post_ff_norm_list.size(),  "post_ff_norm_list");
        check_len(gate_list.size(), "gate_list");
        check_len(up_list.size(),   "up_list");
        check_len(down_list.size(), "down_list");
        check_len(layer_type_ids.size(),        "layer_type_ids");
        check_len(head_dim_per_layer.size(),    "head_dim_per_layer");
        check_len(num_kv_heads_per_layer.size(),"num_kv_heads_per_layer");
        check_len(kv_source_layer.size(),       "kv_source_layer");
        check_len(is_kv_shared.size(),          "is_kv_shared");

        // PLE is present iff ple_gate_list is full-length. When present, all
        // PLE lists and layer_scalar must be full-length.
        const bool has_ple = (ple_gate_list.size() == static_cast<size_t>(N));
        if (has_ple) {
            check_len(ple_proj_list.size(),      "ple_proj_list");
            check_len(ple_post_norm_list.size(), "ple_post_norm_list");
            TORCH_CHECK(ple_dim > 0,
                        "gemma4_set_weights: ple_dim must be > 0 when PLE present");
            TORCH_CHECK(layer_scalar.defined() && layer_scalar.dim() == 1
                        && layer_scalar.size(0) == N,
                        "gemma4_set_weights: layer_scalar must be 1D [num_layers]");
        }
        at::Tensor ls_cpu = has_ple
            ? layer_scalar.to(at::kCPU).to(at::kFloat).contiguous() : at::Tensor();

        TORCH_CHECK(cos_sliding.defined() && sin_sliding.defined()
                    && cos_global.defined() && sin_global.defined()
                    && final_norm_w.defined(),
                    "gemma4_set_weights: cos/sin (both)/final_norm_w must be defined");
        TORCH_CHECK(final_norm_w.dim() == 1 && final_norm_w.size(0) == hidden_size,
                    "gemma4_set_weights: final_norm_w must be 1D [hidden_size]");
        TORCH_CHECK(qk_scale > 0.0,
                    "gemma4_set_weights: qk_scale must be positive (Gemma4 = 1.0)");

        // Build geometry table + commit per-layer weights, validating the
        // mixed-width shapes. Shared layers skip k/v/k_norm shape checks.
        geom_.clear();   geom_.reserve(N);
        layer_weights_.clear();   layer_weights_.reserve(N);
        max_sliding_hd_ = 0;   // widest head_dim among sliding layers
        for (int64_t i = 0; i < N; i++) {
            const int64_t hd  = head_dim_per_layer[i];
            const int64_t nkv = num_kv_heads_per_layer[i];
            const bool shared = is_kv_shared[i] != 0;
            const bool is_global = layer_type_ids[i] != 0;
            TORCH_CHECK(layer_type_ids[i] == 0 || layer_type_ids[i] == 1,
                        "gemma4_set_weights: layer_type_ids[", i, "] must be 0/1");
            TORCH_CHECK(hd > 0 && nkv > 0, "gemma4_set_weights: bad per-layer geom @", i);
            TORCH_CHECK(kv_source_layer[i] >= 0 && kv_source_layer[i] < N,
                        "gemma4_set_weights: kv_source_layer[", i, "] out of range");
            TORCH_CHECK(!shared || !is_kv_shared[kv_source_layer[i]],
                        "gemma4_set_weights: layer ", i, " sources a shared layer");

            // q/o always present; validate widths from the per-layer head_dim.
            TORCH_CHECK(q_w_list[i].defined() && q_w_list[i].dim() == 2,
                        "gemma4_set_weights: q_w[", i, "] must be 2D");
            TORCH_CHECK(q_w_list[i].size(0) == num_q_heads * hd,
                        "gemma4_set_weights: q_w[", i, "].rows=", q_w_list[i].size(0),
                        " != num_q_heads*head_dim=", num_q_heads * hd);
            TORCH_CHECK(o_w_list[i].defined() && o_w_list[i].dim() == 2,
                        "gemma4_set_weights: o_w[", i, "] must be 2D");
            TORCH_CHECK(o_w_list[i].size(1) == num_q_heads * hd,
                        "gemma4_set_weights: o_w[", i, "].cols != num_q_heads*head_dim");
            TORCH_CHECK(q_norm_list[i].defined() && q_norm_list[i].dim() == 1
                        && q_norm_list[i].size(0) == hd,
                        "gemma4_set_weights: q_norm[", i, "] must be 1D [head_dim]");

            if (!shared) {
                TORCH_CHECK(k_w_list[i].defined() && k_w_list[i].size(0) == nkv * hd,
                            "gemma4_set_weights: k_w[", i, "] width mismatch");
                TORCH_CHECK(v_w_list[i].defined() && v_w_list[i].size(0) == nkv * hd,
                            "gemma4_set_weights: v_w[", i, "] width mismatch");
                TORCH_CHECK(k_norm_list[i].defined() && k_norm_list[i].size(0) == hd,
                            "gemma4_set_weights: k_norm[", i, "] must be [head_dim]");
            }

            geom_.push_back({hd, nkv, kv_source_layer[i], is_global, shared});
            if (!is_global) max_sliding_hd_ = std::max(max_sliding_hd_, hd);
            LayerWeights lw{};
            lw.q_w = q_w_list[i]; lw.k_w = k_w_list[i];
            lw.v_w = v_w_list[i]; lw.o_w = o_w_list[i];
            lw.q_norm_w = q_norm_list[i]; lw.k_norm_w = k_norm_list[i];
            lw.input_norm_w = input_norm_list[i];
            lw.post_attn_norm_w = post_attn_norm_list[i];
            lw.pre_ff_norm_w = pre_ff_norm_list[i];
            lw.post_ff_norm_w = post_ff_norm_list[i];
            lw.gate_w = gate_list[i]; lw.up_w = up_list[i]; lw.down_w = down_list[i];
            if (has_ple) {
                TORCH_CHECK(ple_gate_list[i].defined()
                            && ple_gate_list[i].size(0) == ple_dim
                            && ple_proj_list[i].defined()
                            && ple_proj_list[i].size(1) == ple_dim
                            && ple_post_norm_list[i].defined()
                            && ple_post_norm_list[i].size(0) == hidden_size,
                            "gemma4_set_weights: PLE weight shape mismatch @", i);
                lw.ple_gate_w = ple_gate_list[i];
                lw.ple_proj_w = ple_proj_list[i];
                lw.ple_post_norm_w = ple_post_norm_list[i];
                lw.layer_scalar = static_cast<double>(ls_cpu.data_ptr<float>()[i]);
            }
            layer_weights_.push_back(std::move(lw));
        }
        has_ple_ = has_ple;
        ple_dim_ = has_ple ? ple_dim : 0;

        // Commit base params with the MAX widths so declare_buffers (which reads
        // head_dim()/num_kv_heads()) sizes attention buffers for the widest layer.
        set_model_params(num_q_heads, num_kv_heads_max, head_dim_max,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;
        qk_scale_ = qk_scale;
        sliding_window_ = sliding_window;   // 0 disables windowing
        cos_sliding_ = cos_sliding; sin_sliding_ = sin_sliding;
        cos_global_  = cos_global;  sin_global_  = sin_global;
        final_norm_w_ = final_norm_w;

        // v_norm is an unweighted RMS. The
        // rmsnorm kernel always multiplies by a weight buffer, so an all-ones
        // weight (widest head_dim) yields the unweighted norm. One DDR constant
        // shared by every layer; the kernel reads only the leading `head_dim`.
        v_norm_ones_ = at::ones({head_dim_max},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));

        invalidate_model_state();  // Must remain the last non-empty statement.
    }

    // ========================================================================
    // forward — public entry (SEQUENTIAL causal; SDPA generates the causal mask)
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal,
        int64_t chunk_size,
        const std::optional<at::Tensor>& side_input)
    {
        TORCH_CHECK(num_layers() > 0, "Gemma4Model::forward before set_weights");
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "Gemma4Model::forward: hidden_states must be on RPU device");
        TORCH_CHECK(position >= 0, "Gemma4Model::forward: position must be >= 0");

        set_chunk_size_override(chunk_size > 0 ? chunk_size : 0);
        seq_len_ = hidden_states.size(1);

        // PLE side input is [num_layers, seq, ple_dim]. The per-layer
        // slice is DMA-broadcast per replay; its src ptr drifts per forward so the
        // mutable-DMA base is refreshed here (caller owns the flush).
        if (has_ple_) {
            TORCH_CHECK(side_input.has_value() && side_input->defined(),
                        "Gemma4Model::forward: side_input required when PLE present");
            // Core-major scatter layout [num_layers, NUM_CORES, seq, ple_dim/NUM_CORES];
            // only the total element count is contractual (read by per-core offset).
            TORCH_CHECK(side_input->numel() == num_layers() * seq_len_ * ple_dim_,
                        "Gemma4Model::forward: side_input numel must be "
                        "num_layers*seq*ple_dim, got ", side_input->numel());
            side_input_ = side_input->contiguous();
            rpu_ddr_flush_force(side_input_.data_ptr<c10::Half>());
            side_live_base_ = RpuGetDevAddr(side_input_.data_ptr<c10::Half>());
        }

        // Sliding layers window attention once total context exceeds
        // sliding_window_; global layers stay full causal. The base allocates
        // mask SPM only for an explicit non-causal mask, so preserve the real
        // causal intent and pass a 16-bucketed full-attend marker. The marker
        // content is not consumed; build_layer_subgraph uploads the per-layer
        // window mask. Shorter contexts keep the regular causal path.
        original_is_causal_ = is_causal;
        const int64_t total_kv = position + seq_len_;
        if (is_causal && sliding_window_ > 0 && total_kv > sliding_window_) {
            const int64_t kv_align = Align(total_kv, (int64_t)16);
            at::Tensor marker = at::zeros({seq_len_, kv_align},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
            return run_all_layers(hidden_states, k_caches, v_caches,
                                  std::optional<at::Tensor>(marker), position,
                                  /*is_causal=*/false);
        }
        return run_all_layers(hidden_states, k_caches, v_caches,
                              attention_mask, position, is_causal);
    }

protected:
    // ========================================================================
    // static_config — plain SEQUENTIAL causal decoder with no optional callbacks.
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers = num_layers();
        cfg.cross_layer_batch_size = num_layers();  // single group
        return cfg;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& /*plan*/) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::AUTO;
        return cfg;
    }

    // ========================================================================
    // subclass_chunk_size_valid — the auto chunk-size search must reject any cs
    // illegal for any SDPA mode build_layer_subgraph can emit. Longer sequences
    // can otherwise select a chunk that violates the flash grid constraint. The LTM modes (global hd512
    // + sliding hd256) run for EVERY build, windowed or not, so validate them
    // unconditionally; a windowed build additionally emits sliding hd256 MASK_2D.
    // Qwen3 applies the same unconditional validation.
    // ========================================================================
    bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                   int64_t position) const override {
        const int64_t hd_g = head_dim();      // widest (global) = 512
        const int64_t nq   = num_q_heads();
        const int64_t nkv  = num_kv_heads();
        const int tp       = attn_tp();
        auto ok = [&](int64_t hd_, int mt) {
            SdpaConfig c{sdpa_kernel_, hd_, nq, nkv, tp, mt};
            return sdpa_is_valid_chunk_size(c, cs, seq_len, position);
        };
        bool v = ok(hd_g, 1)                                   // global hd512 LTM (always)
            && (max_sliding_hd_ <= 0 || ok(max_sliding_hd_, 1));  // sliding hd256 LTM (always)
        if (ctx().attention_mask.has_value())                 // windowed: + sliding MASK_2D
            v = v && ok(max_sliding_hd_, 4);
        return v;
    }

    // ========================================================================
    // declare_buffers — SPM layout, sized for the WIDEST per-layer head_dim.
    // All temps are LayerWide; the framework
    // auto-shrinks chunk_size to fit SPM. Norm weights are direct-weight
    // (preload DMAs the raw weight without an additive offset).
    // ========================================================================
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        int64_t cs  = ctx.chunk_size;
        int64_t h   = hidden_size();
        int64_t nq  = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd  = head_dim();          // == head_dim_max (512) — widest layer
        int64_t is_ = intermediate_size();
        int tp = attn_tp();
        int64_t local_q  = nq / tp;
        int64_t local_kv = nkv * hd / tp;

        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        int64_t res   = A(cs * h * DWIDTH);
        int64_t q     = A(cs * local_q * hd * DWIDTH);
        int64_t kv    = A(cs * local_kv * DWIDTH);
        int64_t out   = A(cs * local_q * hd * DWIDTH);
        int64_t oproj = A(cs * h * DWIDTH);
        int64_t mlp   = A(cs * (is_ / NUM_CORES) * DWIDTH);

        // A windowed build emits a mix of SDPA modes (global hd512 LTM/NONE +
        // sliding hd256 MASK_2D/LTM), so sdpa_tmp must be the MAX over all of them.
        // The non-windowed build uses the single causal formula.
        int64_t tmp;
        if (ctx.use_attn_mask) {
            auto tmp_for = [&](int64_t hd_, int mt) {
                SdpaConfig c{sdpa_kernel_, hd_, nq, nkv, tp, mt};
                return sdpa_compute_tmp_v16_size(c, cs);
            };
            int64_t e = std::max({ tmp_for(hd, 1),                  // global hd512 LTM
                                   tmp_for(hd, 0),                  // global hd512 NONE (decode)
                                   tmp_for(max_sliding_hd_, 4),     // sliding hd256 MASK_2D
                                   tmp_for(max_sliding_hd_, 1) });  // sliding hd256 LTM (kv<=512)
            tmp = A(e * 32);
        } else {
            SdpaConfig sdpa_cfg{sdpa_kernel_, hd, nq, nkv, tp, 1};
            tmp = A(sdpa_compute_tmp_v16_size(sdpa_cfg, cs) * 32);
        }
        int64_t mask_sz = ctx.use_attn_mask
            ? A(cs * CeilDiv(ctx.max_kv_seq_len, (int64_t)16) * 32) : 0;

        int64_t norm_h  = A(h * DWIDTH);
        int64_t norm_hd = A(hd * DWIDTH);   // q/k norm sized for widest head_dim
        int nl = static_cast<int>(num_layers());

        constexpr BufferScope ALL = BufferScope::LayerWide;
        std::vector<BufferDecl> d;
        d.push_back({"residual1", res,   1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"input_norm",res,   1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"residual2", res,   1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"q",         q,     1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"k",         kv,    1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"v",         kv,    1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"output",    out,   1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"oproj",     oproj, 1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"sdpa_tmp",  tmp,   1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"sdpa_mask", mask_sz,1,13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"gate",      mlp,   1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"up",        mlp,   1, 13, StorageClass::Temp, 0, nullptr, ALL});
        d.push_back({"down",      res,   1, 13, StorageClass::Temp, 0, nullptr, ALL});

        // Direct-weight norm preloads (DMA raw weight; NO +1.0). q/k norm are
        // sized for the widest head_dim; shared layers skip k_norm (undefined).
        auto add_norm = [&](const char* name, int64_t sz,
                            at::Tensor LayerWeights::* member) {
            BufferDecl b;
            b.name = name; b.size = sz;
            b.storage = StorageClass::PersistentPerLayer; b.per_layer = nl;
            b.scope = BufferScope::LayerWide;
            b.preload_callback =
                [this, member](FusedModelBase&, int L, uint32_t core0_addr) {
                    const at::Tensor& w = layer_weights_[L].*member;
                    if (!w.defined()) return;   // shared-layer k_norm: nothing to load
                    rpu_launch_ddr_broadcast_spm_dma(
                        w.data_ptr<c10::Half>(), w.numel(), core0_addr);
                };
            d.push_back(b);
        };
        add_norm("input_norm_w",     norm_h,  &LayerWeights::input_norm_w);
        add_norm("post_attn_norm_w", norm_h,  &LayerWeights::post_attn_norm_w);
        add_norm("pre_ff_norm_w",    norm_h,  &LayerWeights::pre_ff_norm_w);
        add_norm("post_ff_norm_w",   norm_h,  &LayerWeights::post_ff_norm_w);
        add_norm("q_norm_w",         norm_hd, &LayerWeights::q_norm_w);
        add_norm("k_norm_w",         norm_hd, &LayerWeights::k_norm_w);

        {
            BufferDecl b;
            b.name = "final_norm_w"; b.size = norm_h;
            b.storage = StorageClass::Persistent; b.per_layer = 0;
            b.scope = BufferScope::LayerWide;
            b.preload_callback =
                [this](FusedModelBase&, int, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        final_norm_w_.data_ptr<c10::Half>(),
                        final_norm_w_.numel(), core0_addr);
                };
            d.push_back(b);
        }

        // Shared all-ones weight for the unweighted v_norm (sized for the
        // widest head_dim; per-layer rmsnorm reads only its leading head_dim).
        {
            BufferDecl b;
            b.name = "v_norm_ones"; b.size = norm_hd;
            b.storage = StorageClass::Persistent; b.per_layer = 0;
            b.scope = BufferScope::LayerWide;
            b.preload_callback =
                [this](FusedModelBase&, int, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        v_norm_ones_.data_ptr<c10::Half>(),
                        v_norm_ones_.numel(), core0_addr);
                };
            d.push_back(b);
        }

        // PLE temporaries and a zero residual for the residual-form all-reduce.
        if (has_ple_) {
            int64_t ple_buf = A(cs * ple_dim_ * DWIDTH);
            d.push_back({"ple_gate", ple_buf, 1, 13, StorageClass::Temp, 0, nullptr, ALL});
            d.push_back({"ple_side", ple_buf, 1, 13, StorageClass::Temp, 0, nullptr, ALL});
            // ple_zero: zeros residual fed to all_reduce_sum_residual (only a
            // residual-form all-reduce exists). MUST be Temp, NOT Persistent: it
            // is sized by chunk (cs*h), and a chunk-sized Persistent buffer would
            // change shape between the prefill (cs=S) and decode (cs=1) BUILDs and
            // trip the persistent-shape-stability check.
            // It is zeroed per-layer in build_layer_subgraph via a self-subtract.
            d.push_back({"ple_zero", res, 1, 13, StorageClass::Temp, 0, nullptr, ALL});

            // Per-layer PLE norm preload (direct-weight, like the 4 block norms).
            BufferDecl bn;
            bn.name = "ple_post_norm_w"; bn.size = norm_h;
            bn.storage = StorageClass::PersistentPerLayer; bn.per_layer = nl;
            bn.scope = BufferScope::LayerWide;
            bn.preload_callback = [this](FusedModelBase&, int L, uint32_t core0_addr) {
                const at::Tensor& w = layer_weights_[L].ple_post_norm_w;
                if (!w.defined()) return;
                rpu_launch_ddr_broadcast_spm_dma(
                    w.data_ptr<c10::Half>(), w.numel(), core0_addr);
            };
            d.push_back(bn);
        }
        return d;
    }

    // The three sizing inputs declare_buffers reads that the framework's params
    // hash cannot see. `max_sliding_hd_` sizes the Temp "sdpa_tmp" on the
    // windowed branch, and `ple_dim_` sizes the Temp "ple_gate"/"ple_side" — two
    // geometry tables sharing the same head_dim_max (512, set into the base by
    // set_weights) would otherwise hash identically with different sliding
    // widths. `has_ple_` only gates decls, and its PersistentPerLayer member
    // would already abort loudly, but it is the same decision so it belongs here.
    // All three are written once in set_weights before the first forward, so
    // folding them in is a no-op today; the point is that the layout can no
    // longer move without the hash moving.
    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(0, has_ple_ ? 1 : 0);
        h = detail::layout_mix(h, ple_dim_);
        h = detail::layout_mix(h, max_sliding_hd_);
        return h;
    }

    // ========================================================================
    // build_layer_subgraph — SEQUENTIAL per-layer body with per-layer geometry.
    // ========================================================================
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        const auto& g  = geom_[layer_idx];
        int64_t seq_len = chunk.len;
        int64_t cos_sin_start = ctx().position + chunk.offset;
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t hd = g.head_dim;            // PER-LAYER width (256 / 512)
        int64_t nkv = g.num_kv_heads;
        int tp = attn_tp();
        int64_t local_q_heads  = nq / tp;
        int64_t local_kv_heads = nkv / tp;
        bool is_last_layer = (layer_idx == num_layers() - 1);

        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        // Input RMSNorm (direct-weight).
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "input_norm_w"),
            seq_len, h, eps_);

        // Q projection, plus K/V projections for non-shared layers.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h, /*partition=*/1, /*num_cores=*/tp);
        if (!g.is_kv_shared) {
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.k_w, addr(0, "k"),
                seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.v_w, addr(0, "v"),
                seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp);
        }

        // QK-RMSNorm (direct-weight) and RoPE. Select the cos/sin table by
        // layer type and per-layer head_dim.
        c10::Half* cos_ptr = g.is_global ? cos_global_.data_ptr<c10::Half>()
                                         : cos_sliding_.data_ptr<c10::Half>();
        c10::Half* sin_ptr = g.is_global ? sin_global_.data_ptr<c10::Half>()
                                         : sin_sliding_.data_ptr<c10::Half>();
        // Q: q -> output (norm) -> q (rope)
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "q"), addr(0, "output"),
            layer_addr(layer_idx, 0, "q_norm_w"),
            seq_len * local_q_heads, hd, eps_);
        rpu_launch_rope_spm_kernel(
            addr(0, "output"), addr(0, "q"), cos_ptr, sin_ptr,
            seq_len, local_q_heads, hd, cos_sin_start, tp);

        if (!g.is_kv_shared) {
            // K: k -> input_norm (norm) -> k (rope)
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "k"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "k_norm_w"),
                seq_len * local_kv_heads, hd, eps_);
            rpu_launch_rope_spm_kernel(
                addr(0, "input_norm"), addr(0, "k"), cos_ptr, sin_ptr,
                seq_len, local_kv_heads, hd, cos_sin_start, tp);

            // v_norm is an unweighted RMS on V, without RoPE. In-place
            // v -> v with the all-ones weight; the kv insert reads addr("v").
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "v"), addr(0, "v"),
                addr(0, "v_norm_ones"),
                seq_len * local_kv_heads, hd, eps_);

            // KV-cache insert into THIS layer's own slot.
            auto& k_cache = (*ctx().k_caches)[layer_idx];
            auto& v_cache = (*ctx().v_caches)[layer_idx];
            rpu_launch_insert_kcache_spm_unified(
                k_cache, cos_sin_start, addr_offset("k").value,
                seq_len, nkv, hd, tp);
            rpu_launch_insert_vcache_spm_unified(
                v_cache, cos_sin_start, addr_offset("v").value,
                seq_len, nkv, hd, tp);
        }

        // SDPA reads the source layer's K/V slot, which is the current layer's
        // slot when K/V is not shared. qk_scale is 1.0.
        //
        // Per-(layer,chunk) mask routing: a sliding(hd256) layer whose chunk
        // sees total kv>512 uses an explicit windowed MASK_2D; global(hd512) layers
        // and short (kv<=512) chunks keep the legacy LTM/NONE causal path (window
        // not yet clipping == full causal). The fragile hd512 2D path NEVER fires.
        // ctx().is_causal is clobbered to false on the windowed forward (marker
        // mask), so the causal intent is read from original_is_causal_.
        auto& k_src = (*ctx().k_caches)[g.kv_source_layer];
        auto& v_src = (*ctx().v_caches)[g.kv_source_layer];
        int64_t kv_seq_len = chunk.kv_seq_len;
        int        mask_type;
        uint32_t   mask_off = 0u;
        at::Tensor k_use = k_src, v_use = v_src;   // may be window-offset views
        int64_t    sdpa_kv = kv_seq_len;
        if (g.is_global || sliding_window_ <= 0 || kv_seq_len <= sliding_window_) {
            mask_type = (original_is_causal_ && seq_len > 1) ? 1 : 0;  // LTM / NONE
        } else {
            // Bound the sliding-layer KV read to the window union
            // [win_start, kv) instead of reading all of [0, kv) and masking. The
            // chunk's earliest attended key is abs_start-(W-1) (abs_start = absolute
            // pos of row 0 = kv - seq_len); align DOWN to the 16-key swizzle chunk
            // so the read starts on a chunk boundary. Slicing dim 1 (sKeyVx, the
            // outermost-after-batch seq-chunk dim) offsets the cache base pointer;
            // the swizzle strides are geometry-only so the kernel reads the window
            // correctly. A small [seq_len, win_len] MASK_2D still applies the exact
            // per-query causal+window inside the read range — but with far fewer
            // masked leading tiles (the kernel's chunked-MASK_2D precision sink),
            // so decode + long-context windowed attention stays clean.
            mask_type = 4;
            int64_t win_start, win_len;
            if (seq_len == 1) {
                // Decode: bucket-stable window. The Python sig buckets the graph by
                // ceil(total_kv/16), so the (<=16) decode positions sharing a bucket
                // REPLAY one cached graph; the SDPA cache-base offset + mask DMA size
                // must therefore be constant across the bucket (only the mask CONTENT
                // varies, refreshed per replay via the stable slot). Derive both from
                // b=ceil(kv/16): win_start=16(b-1)-W is 16-aligned and <= every query's
                // window start in the bucket; win_len=16b-win_start covers up to the
                // bucket's max kv (extra keys are future -> masked). Constant per bucket.
                const int64_t b = (kv_seq_len + 15) / 16;
                win_start = std::max((int64_t)0, 16 * (b - 1) - sliding_window_);
                win_len   = 16 * b - win_start;
            } else {
                // Prefill: one deterministic forward per (S, position) -> exact window.
                const int64_t abs_start = kv_seq_len - seq_len;
                win_start = (std::max((int64_t)0, abs_start - (sliding_window_ - 1)) / 16) * 16;
                win_len   = kv_seq_len - win_start;
            }
            // Never read past the allocated KV capacity (extra keys mask out as future).
            const int64_t cap = k_src.size(1) * 16;
            win_len = std::min(win_len, cap - win_start);
            if (win_start > 0) {
                k_use = k_src.slice(1, win_start / 16);
                v_use = v_src.slice(1, win_start / 16);
            }
            sdpa_kv = win_len;
            mask_off = upload_windowed_mask(chunk, win_start, win_len);
        }
        rpu_launch_sdpa_spm_dispatch(
            sdpa_kernel_, k_use, v_use, mask_type,
            std::optional<double>(qk_scale_),
            addr_offset("q").value, addr_offset("output").value,
            addr_offset("sdpa_tmp").value, mask_off,
            seq_len, nq, nkv, hd, sdpa_kv, tp, NUM_CORES);

        // O proj (row partition over tp cores).
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "output"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp);

        // Post-attention normalization and residual block:
        //   residual2 = reduce(oproj) + residual1
        //   residual2 = residual2 - residual1            (isolate attn out)
        //   input_norm = post_attn_norm(residual2)
        //   residual2 = residual1 + input_norm
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
            seq_len, h, tp, NUM_CORES);
        rpu_launch_eltwise_sub_spm_kernel(
            addr(0, "residual2"), addr(0, "residual1"), addr(0, "residual2"),
            seq_len * h);
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual2"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "post_attn_norm_w"),
            seq_len, h, eps_);
        rpu_launch_eltwise_add_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"), addr(0, "residual2"),
            seq_len * h);

        // Pre/post-feedforward normalization and residual block:
        //   residual1 = pre_ff_norm(residual2)
        //   down      = down(gelu(gate(residual1)) * up(residual1))
        //   residual1 = reduce(down) + residual2
        //   residual1 = residual1 - residual2            (isolate mlp out)
        //   input_norm = post_ff_norm(residual1)
        //   residual1 = residual2 + input_norm
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual2"), addr(0, "residual1"),
            layer_addr(layer_idx, 0, "pre_ff_norm_w"),
            seq_len, h, eps_);
        int64_t mlp_elems = seq_len * (intermediate_size() / NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
            seq_len, intermediate_size(), h, 1);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "gate"), addr(0, "gate"),
            mlp_elems, ValuOpType::ADD, GeluMode::TANH);   // gelu_pytorch_tanh
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), lw.up_w, addr(0, "up"),
            seq_len, intermediate_size(), h, 1);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
            mlp_elems, ValuOpType::MUL, c10::Half(1.0));
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "gate"), lw.down_w, addr(0, "down"),
            seq_len, h, intermediate_size(), 0);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
            seq_len, h);
        rpu_launch_eltwise_sub_spm_kernel(
            addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
            seq_len * h);
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "post_ff_norm_w"),
            seq_len, h, eps_);
        rpu_launch_eltwise_add_spm_kernel(
            addr(0, "residual2"), addr(0, "input_norm"), addr(0, "residual1"),
            seq_len * h);

        // Per-layer side-input injection and layer_scalar. residual1
        // (replicated [seq,h]) holds the pre-PLE layer output. Both PLE linears
        // are row-partitioned -> all-reduce -> replicated, so the side-input only
        // broadcasts (no scatter) and residual1 stays replicated across cores.
        if (has_ple_) {
            const int64_t pd = ple_dim_;
            const int64_t pc = pd / NUM_CORES;   // per-core ple cols (gate col-partition)
            // Mirror the MLP col->row pattern: gate is COL-partitioned (core c
            // owns ple cols [c*pc, (c+1)*pc)), so the side-input must be SCATTERED
            // the same way (core c <- side[L][:, c*pc:(c+1)*pc]); proj is then
            // row-partitioned (consumes the partitioned gate) + all-reduce ->
            // replicated [seq,h]. side_input_ is [num_layers, NUM_CORES, seq, pc]
            // core-major; mutable since the src ptr drifts per forward.
            // chunk.offset selects rows within each core block, so this is also
            // correct for chunked prefill (core_stride spans the full core block).
            const int64_t side_off_bytes =
                (layer_idx * seq_len_ * pd + chunk.offset * pc) * DWIDTH;
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &side_live_base_, side_off_bytes, seq_len * pc,
                seq_len_ * pc * DWIDTH, addr(0, "ple_side"), NUM_CORES);

            // gate = per_layer_input_gate(residual1)  [h -> pd], COL-partition.
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "residual1"), lw.ple_gate_w, addr(0, "ple_gate"),
                seq_len, pd, h, /*partition=*/1, /*num_cores=*/NUM_CORES);
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "ple_gate"), addr(0, "ple_gate"),
                seq_len * pc, ValuOpType::ADD, GeluMode::TANH);   // gelu_pytorch_tanh
            rpu_launch_eltwise_mul_spm_kernel(
                addr(0, "ple_gate"), addr(0, "ple_side"), addr(0, "ple_gate"),
                seq_len * pc);

            // proj = per_layer_projection(gate)  [pd -> h], ROW-part + all-reduce
            // (with a zeros residual), then post_per_layer_input_norm, then residual
            // add. Zero the residual buffer in-place via self-subtract (residual1 is
            // replicated [seq,h] here) so no Persistent chunk-sized buffer is needed.
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "ple_gate"), lw.ple_proj_w, addr(0, "down"),
                seq_len, h, pd, /*partition=*/0, /*num_cores=*/NUM_CORES);
            rpu_launch_eltwise_sub_spm_kernel(
                addr(0, "residual1"), addr(0, "residual1"), addr(0, "ple_zero"),
                seq_len * h);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "down"), addr(0, "ple_zero"), addr(0, "residual2"),
                seq_len, h);
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual2"), addr(0, "residual2"),
                layer_addr(layer_idx, 0, "ple_post_norm_w"), seq_len, h, eps_);
            rpu_launch_eltwise_add_spm_kernel(
                addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
                seq_len * h);

            // Apply the per-layer output scalar.
            rpu_launch_eltwise_binary_scalar_spm_kernel(
                addr(0, "residual1"), c10::Half(lw.layer_scalar),
                addr(0, "residual1"), seq_len * h, ValuOpType::MUL);
        }

        // Last layer: final RMSNorm (Persistent slot).
        if (is_last_layer) {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual1"), addr(0, "residual1"),
                addr(0, "final_norm_w"), seq_len, h, eps_);
        }

        if (!ctx().output_to_spm) {
            emit_layer_output_dma(layer_idx, chunk);
        }
    }

private:
    // Build, memoize, and upload the windowed additive mask over
    // the OFFSET KV read range [win_start, win_start+win_len) for a sliding chunk;
    // returns the "sdpa_mask" SPM offset. abs_start = kv - qn is the absolute pos
    // of row 0 (chunk.offset is NOT absolute). Row q (abs = abs_start+q) attends
    // absolute keys (abs-(sliding-1), abs]; the mask is laid out over the read
    // window so column kk maps to absolute key win_start+kk. sdpa_prepare_mask
    // copies into a shape-keyed STABLE slot (same address, content overwritten),
    // so the baked DMA re-reads refreshed content every replay -> correct for
    // chunked prefill AND decode. Memo key (qn,kv) is valid since win_start/win_len
    // are determined by (qn,kv).
    uint32_t upload_windowed_mask(const ChunkInfo& chunk, int64_t win_start, int64_t win_len) {
        const int64_t qn = chunk.len, kv = chunk.kv_seq_len;
        const int64_t abs_start = kv - qn;
        auto key = std::make_pair(qn, kv);
        at::Tensor& host = windowed_mask_cache_.touch(key);
        if (!host.defined()) {
            host = at::zeros({qn, win_len},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
            const c10::Half NEG =
                static_cast<c10::Half>(-std::numeric_limits<float>::infinity());
            auto a = host.accessor<c10::Half, 2>();
            for (int64_t q = 0; q < qn; ++q) {
                const int64_t abs = abs_start + q;
                for (int64_t kk = 0; kk < win_len; ++kk) {
                    const int64_t k = win_start + kk;        // absolute key index
                    if (!(k <= abs && k > abs - sliding_window_)) a[q][kk] = NEG;
                }
            }
        }
        PreparedMask pm = sdpa_prepare_mask(c10::optional<at::Tensor>(host),
                                            /*is_causal=*/false, qn, win_len,
                                            sdpa_stable_mask_cache());
        uint32_t mask_off = addr_offset("sdpa_mask").value;
        sdpa_dma_mask_to_spm(pm, mask_off, qn, win_len, attn_tp());
        return mask_off;
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerGeom>    geom_;
    at::Tensor cos_sliding_, sin_sliding_, cos_global_, sin_global_;
    at::Tensor final_norm_w_;
    at::Tensor v_norm_ones_;        // all-ones weight for unweighted v_norm
    // PLE state.
    bool has_ple_ = false;
    int64_t ple_dim_ = 0;
    at::Tensor side_input_;          // per-forward [num_layers, seq, ple_dim] (kept alive)
    uint64_t side_live_base_ = 0;    // mutable-DMA base (RpuGetDevAddr of side_input_)
    double eps_ = 1e-6;
    double qk_scale_ = 1.0;     // Gemma4 self.scaling
    int64_t seq_len_ = 0;
    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;
    // Sliding-window mask state.
    int64_t sliding_window_ = 0;         // 512 (E4B); 0 disables windowing
    int64_t max_sliding_hd_ = 0;         // widest head_dim among sliding layers (256)
    bool    original_is_causal_ = true;  // true causal intent (ctx().is_causal clobbered on windowed path)
    // Bounded host mask memo by (qn, kv). Entries are CPU tensors whose content
    // is copied into stable shape-keyed RPU slots before DMA use.
    static constexpr size_t kMaxWindowedMasks = 256;
    rpu::BoundedShapeMap<std::pair<int64_t, int64_t>, at::Tensor, rpu::PairHash>
        windowed_mask_cache_{kMaxWindowedMasks, rpu::ShapeMapPolicy::Evict,
                             "Gemma4Model::windowed_mask_cache_"};
};

}  // namespace v3

// =============================================================================
// Instance registry + public C API
// =============================================================================

using Gemma4Registry = ModelHandleRegistry<v3::Gemma4Model>;

int64_t rpu_gemma4_create() {
    return Gemma4Registry::create();
}

void rpu_gemma4_destroy(int64_t handle) {
    Gemma4Registry::destroy(handle, "rpu_gemma4_destroy");
}

void rpu_gemma4_set_weights(
    int64_t handle,
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList input_norm_list, at::TensorList post_attn_norm_list,
    at::TensorList pre_ff_norm_list, at::TensorList post_ff_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos_sliding, const at::Tensor& sin_sliding,
    const at::Tensor& cos_global,  const at::Tensor& sin_global,
    const at::Tensor& final_norm_w,
    at::TensorList ple_gate_list, at::TensorList ple_proj_list,
    at::TensorList ple_post_norm_list, const at::Tensor& layer_scalar,
    at::IntArrayRef layer_type_ids,
    at::IntArrayRef head_dim_per_layer,
    at::IntArrayRef num_kv_heads_per_layer,
    at::IntArrayRef kv_source_layer,
    at::IntArrayRef is_kv_shared,
    int64_t num_q_heads, int64_t num_kv_heads_max, int64_t head_dim_max,
    int64_t hidden_size, int64_t intermediate_size, int64_t ple_dim,
    double eps, double qk_scale, int64_t sliding_window)
{
    Gemma4Registry::get(handle, "rpu_gemma4")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        q_norm_list, k_norm_list,
        input_norm_list, post_attn_norm_list, pre_ff_norm_list, post_ff_norm_list,
        gate_list, up_list, down_list,
        cos_sliding, sin_sliding, cos_global, sin_global, final_norm_w,
        ple_gate_list, ple_proj_list, ple_post_norm_list, layer_scalar,
        layer_type_ids, head_dim_per_layer, num_kv_heads_per_layer,
        kv_source_layer, is_kv_shared,
        num_q_heads, num_kv_heads_max, head_dim_max,
        hidden_size, intermediate_size, ple_dim, eps, qk_scale, sliding_window);
}

at::Tensor rpu_gemma4_forward(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal,
    int64_t chunk_size,
    const std::optional<at::Tensor>& side_input)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gemma4Registry::get(handle, "rpu_gemma4")->forward(
        hidden_states, k_caches, v_caches, attention_mask, position, is_causal,
        chunk_size, side_input);
}
