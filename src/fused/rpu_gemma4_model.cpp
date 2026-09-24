// rpu_gemma4_model.cpp — Gemma4 E4B text decoder using FusedModelBase.
//
// Per-layer head dimensions differ: sliding attention uses 256 and global
// attention uses 512. Buffer declarations use the maximum width; each layer
// passes its actual geometry to RoPE and SDPA.
// RMSNorm consumes direct weights, without Gemma2's +1 preprocessing.
// SDPA uses qk_scale=1.0. QK-RMSNorm and the four-norm residual structure follow
// the corresponding shared decoder mechanisms.
// Sliding layers use windowed KV slices and a window-local explicit mask;
// decode signatures keep the slice and mask size stable within a KV bucket.
// Global layers use full causal attention. PLE side inputs, layer_scalar and
// cross-layer KV sharing are bound through the model's per-layer geometry.
// Framework contract: src/core/fused_model_base.h.

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_spm_buffers.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_spm_allocator.h"
#include "rpu_spm_residency.h"
#include "rpu_runtime_state.h"
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>   // T4: -inf fill for the windowed mask
#include <utility>  // T4: windowed_mask_cache_ keyed by (chunk.len, chunk.kv_seq_len)
#include "rpu_bounded_shape_map.h"

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace v3 {

namespace {
// GEMMA4_FIXED_KERNEL_BASIS: RMSNorm and elementwise GELU/mul/add/sub/scalar
// launchers implement the E4B checkpoint's exact math and have no selector.
// Persistent norm-weight preload DMA is fixed by BufferDecl ownership. Only
// policy-bearing linear/rope/KV/attention/reduce/mutable-side-DMA sites route.
constexpr int64_t GEMMA4_Q_LINEAR_SITE = 8725399426442661120LL;
constexpr int64_t GEMMA4_K_LINEAR_SITE = 6916943715144097592LL;
constexpr int64_t GEMMA4_V_LINEAR_SITE = 3870327977122590662LL;
constexpr int64_t GEMMA4_Q_ROPE_SITE = 723594843514692113LL;
constexpr int64_t GEMMA4_K_ROPE_SITE = 5337083984893514264LL;
constexpr int64_t GEMMA4_KV_INSERT_SITE = 6562867400688913245LL;
constexpr int64_t GEMMA4_ATTENTION_SITE = 6581941412312031012LL;
constexpr int64_t GEMMA4_PREPARE_ALL_REDUCE_SITE = 5286999247906518060LL;
constexpr int64_t GEMMA4_O_LINEAR_SITE = 8703745226169185634LL;
constexpr int64_t GEMMA4_ATTENTION_ALL_REDUCE_SITE = 2468806415651175396LL;
constexpr int64_t GEMMA4_GATE_LINEAR_SITE = 728529537003975745LL;
constexpr int64_t GEMMA4_UP_LINEAR_SITE = 668941020522880346LL;
constexpr int64_t GEMMA4_DOWN_LINEAR_SITE = 146067009784919675LL;
constexpr int64_t GEMMA4_MLP_ALL_REDUCE_SITE = 4396400591359725542LL;
constexpr int64_t GEMMA4_PLE_DMA_SITE = 6582489299560935010LL;
constexpr int64_t GEMMA4_PLE_GATE_LINEAR_SITE = 7599842320145438531LL;
constexpr int64_t GEMMA4_PLE_PROJ_LINEAR_SITE = 7972588270602457603LL;
constexpr int64_t GEMMA4_PLE_ALL_REDUCE_SITE = 7105886685527167396LL;
constexpr int64_t GEMMA4_PLE_NORM_PRELOAD_SITE = 7312830814832177843LL;
constexpr int64_t GEMMA4_WINDOW_MASK_SCHEDULE_SITE = 858194843690279753LL;

enum class Gemma4RopeRoute : int64_t { ROPE_SPM = 1 };
enum class Gemma4AllReduceRoute : int64_t { PREPARE_RING_INPUT = 1 };
enum class Gemma4MutableDmaRoute : int64_t {
    DDR_SCATTER_TO_SPM = 1,
    DDR_BROADCAST_TO_SPM = 1,
};

int64_t gemma4_kv_invocation(int64_t chunk_idx, bool is_global) {
    return chunk_idx * 2 + (is_global ? 1 : 0);
}
}  // namespace

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
        // T6 PLE (E4B): per_layer_input_gate / per_layer_projection (both
        // row-swizzled, num_cores=8) + post_per_layer_input_norm. Defined only
        // when has_ple_ (ple_*_list non-empty); else default (unused).
        at::Tensor ple_gate_w, ple_proj_w, ple_post_norm_w;
        double layer_scalar = 1.0;   // per-layer output scalar (audit §6; != 1.0)
    };

    // Per-layer mixed geometry (the T1b geometry table; mirror of the Python
    // adapters/gemma4/geometry.py LayerGeom, kept C++-side like gemma2's
    // layer_type_ids_).
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

        // T6 PLE is opt-in: present iff ple_gate_list is non-empty (the T5
        // single-layer path binds without it). When present, all PLE lists +
        // layer_scalar must be full-length.
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
        max_sliding_hd_ = 0;   // T4: widest head_dim among sliding layers (256 E4B)
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
        sliding_window_ = sliding_window;   // T4: 0 disables (global-only / legacy)
        cos_sliding_ = cos_sliding; sin_sliding_ = sin_sliding;
        cos_global_  = cos_global;  sin_global_  = sin_global;
        final_norm_w_ = final_norm_w;

        // T3: v_norm is an UNWEIGHTED RMS (with_scale=False — audit §2). The
        // rmsnorm kernel always multiplies by a weight buffer, so an all-ones
        // weight (widest head_dim) yields the unweighted norm. One DDR constant
        // shared by every layer; the kernel reads only the leading `head_dim`.
        v_norm_ones_ = at::ones({head_dim_max},
            at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));

        invalidate_model_state();  // D-503: last non-empty statement
    }

    std::vector<int64_t> resolve_stage_domain(
        int64_t execution_len, int64_t planning_position,
        int64_t max_kv_len, int64_t requested_chunk_size) {
        detail::validate_fmb_planning_shape(
            execution_len, planning_position, "Gemma4 planner");
        TORCH_CHECK(
            cos_sliding_.defined() && cos_global_.defined(),
            "RPU_PLANNER_REJECT:CAPABILITY: Gemma4 planner requires "
            "initialized weights");
        TORCH_CHECK(
            max_kv_len >= planning_position + execution_len,
            "RPU_PLANNER_REJECT:EXACT_MISMATCH: Gemma4 max_kv_len must "
            "cover the planned execution");
        const int64_t rope_horizon = std::min(
            cos_sliding_.size(0), cos_global_.size(0));
        TORCH_CHECK(
            max_kv_len <= rope_horizon,
            "RPU_PLANNER_REJECT:CAPABILITY: Gemma4 execution exceeds the "
            "initialized RoPE horizon");
        TORCH_CHECK(
            requested_chunk_size == 0 ||
                (requested_chunk_size >= 16 &&
                 requested_chunk_size % 16 == 0),
            "RPU_PLANNER_REJECT:EXACT_MISMATCH: Gemma4 requested chunk "
            "size must be 0 or a positive multiple of 16, got ",
            requested_chunk_size);

        std::optional<at::Tensor> marker;
        bool planner_is_causal = true;
        if (sliding_window_ > 0 && max_kv_len > sliding_window_) {
            marker = at::empty(
                {1, Align(max_kv_len, static_cast<int64_t>(16))},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
            planner_is_causal = false;
        }
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                execution_len, planning_position, marker,
                planner_is_causal, requested_chunk_size));
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
        const std::optional<at::Tensor>& side_input,
        at::IntArrayRef planned_stage_descriptor)
    {
        TORCH_CHECK(num_layers() > 0, "Gemma4Model::forward before set_weights");
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "Gemma4Model::forward: hidden_states must be on RPU device");
        TORCH_CHECK(position >= 0, "Gemma4Model::forward: position must be >= 0");

        seq_len_ = hidden_states.size(1);
        detail::validate_fmb_planning_shape(
            seq_len_, position, "Gemma4Model::forward");
        TORCH_CHECK(
            chunk_size == 0 || planned_stage_descriptor.empty(),
            "RPU_PLANNER_REJECT:EXACT_MISMATCH: Gemma4 forward received "
            "both a scalar chunk and a stage-plan descriptor");
        TORCH_CHECK(
            !planned_stage_descriptor.empty(),
            "RPU_PLANNER_REJECT:CAPABILITY: Gemma4 production forward "
            "requires a COMPLETE native stage-plan descriptor");
        set_chunk_size_override(0);

        // dynamic_config also has planning-only callers. Only a real forward
        // may refresh the stable mask slots, including on fast/deep REPLAY.
        preparing_forward_masks_ = true;
        auto finish_mask_prepare = c10::make_scope_exit([&] {
            preparing_forward_masks_ = false;
        });

        // T6 PLE: per-layer side-input [num_layers, seq, ple_dim]. The per-layer
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

        // T4 sliding-window mask. The 35 sliding(hd256) layers must window their
        // attention once total context >512; global(hd512) layers stay full causal.
        // Per-(layer,chunk) routing lives in build_layer_subgraph and needs the
        // TRUE causal intent, but the base only sets use_attn_mask (allocates the
        // mask SPM buffer + relayout + folds max_kv_seq_len into the alloc-hash)
        // when it receives an explicit mask with is_causal=false. So: preserve
        // original_is_causal_ for per-layer routing, then drive the base with a
        // full-attend MARKER mask sized [seq_len, Align(total_kv,16)] (16-bucketed
        // so the hash + declare_buffers CeilDiv move in lockstep -> relayout only
        // every 16 tokens, WITHOUT editing the shared base hash). The marker
        // CONTENT is never consumed; dynamic_config refreshes one stable DDR
        // mask per chunk before any fast skip. A zero-growth single-chunk
        // schedule shares its SPM copy; multi-chunk retains per-layer copies.
        // total_kv<=512 keeps the
        // legacy causal path -> byte-identical to pre-T4.
        original_is_causal_ = is_causal;
        const int64_t total_kv = position + seq_len_;
        if (is_causal && sliding_window_ > 0 && total_kv > sliding_window_) {
            const int64_t kv_align = Align(total_kv, (int64_t)16);
            at::Tensor marker = at::zeros({seq_len_, kv_align},
                at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
            return run_all_layers(hidden_states, k_caches, v_caches,
                                  std::optional<at::Tensor>(marker), position,
                                  /*is_causal=*/false,
                                  /*planned_chunk_size=*/0,
                                  planned_stage_descriptor);
        }
        return run_all_layers(hidden_states, k_caches, v_caches,
                              attention_mask, position, is_causal,
                              /*planned_chunk_size=*/0,
                              planned_stage_descriptor);
    }

protected:
    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
        TORCH_CHECK(
            plan.chunk_mode == ChunkMode::SEQUENTIAL,
            "RPU_PLANNER_REJECT:CAPABILITY: Gemma4 descriptor requires "
            "SEQUENTIAL traversal");

        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = physical_len == 1
            ? std::min(cos_sliding_.size(0), cos_global_.size(0))
            : position + logical_len;
        manifest.kv_insert_physical_rows = physical_len;
        manifest.graph_lifecycle = FmbGraphLifecycle::RETAINED_CACHE;
        manifest.linear_accumulation = FmbLinearAccumulationPolicy::ACC16;

        auto append = [&](FmbRouteFamily family, int64_t site_id,
                          int64_t selector, int64_t flags = 0,
                          std::vector<int64_t> arguments = {},
                          int64_t invocation = 0) {
            manifest.routes.push_back({site_id, family, selector, flags,
                                       std::move(arguments), invocation});
        };
        bool mask_retained = false;
        (void)declare_buffers_for_mask_residency(layout, true, &mask_retained);
        // Default traversal is layer-outer. One shared SPM slot can retain only
        // a single chunk's mask; multi-chunk keeps every layer's original DMA.
        mask_retained = mask_retained && plan.compute.chunks.size() == 1;
        append(FmbRouteFamily::GRAPH_SCHEDULE, GEMMA4_WINDOW_MASK_SCHEDULE_SITE,
               mask_retained ? 2 : 1, 0, window_mask_schedule_arguments(plan));

        auto append_linear = [&](int64_t site_id, int64_t invocation) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(
                       FmbLinearRouteSelector::AUTO_TILE),
                   0, {}, invocation);
        };

        for (const ChunkInfo& chunk : plan.compute.chunks) {
            const int64_t invocation = chunk.idx;
            append_linear(GEMMA4_Q_LINEAR_SITE, invocation);
            append_linear(GEMMA4_K_LINEAR_SITE, invocation);
            append_linear(GEMMA4_V_LINEAR_SITE, invocation);
            append(FmbRouteFamily::ROPE, GEMMA4_Q_ROPE_SITE,
                   static_cast<int64_t>(Gemma4RopeRoute::ROPE_SPM),
                   0, {}, invocation);
            append(FmbRouteFamily::ROPE, GEMMA4_K_ROPE_SITE,
                   static_cast<int64_t>(Gemma4RopeRoute::ROPE_SPM),
                   0, {}, invocation);

            // The one source call handles both Gemma4 head widths. Keep one
            // immutable typed plan per (chunk, geometry), while all layers of
            // that geometry consume the same V3 invocation.
            for (const bool global : {false, true}) {
                auto it = std::find_if(
                    geom_.begin(), geom_.end(),
                    [global](const LayerGeom& geom) {
                        return !geom.is_kv_shared &&
                            geom.is_global == global;
                    });
                if (it == geom_.end()) continue;
                const bool dynamic_decode_position =
                    physical_len == 1 && position == 0;
                const KvInsertSegmentPlan kv_plan =
                    rpu_resolve_kvinsert_segment_plan_auto(
                        dynamic_decode_position
                            ? 0 : position + chunk.offset,
                        chunk.len, chunk.len, attn_tp(),
                        it->num_kv_heads, it->head_dim,
                        KV_INSERT_CAP_V2);
                const KvInsertRouteArguments arguments =
                    rpu_kvinsert_route_arguments(
                        kv_plan, attn_tp(), it->num_kv_heads,
                        it->head_dim);
                append(
                    FmbRouteFamily::KV_INSERT, GEMMA4_KV_INSERT_SITE,
                    static_cast<int64_t>(kv_plan.route()),
                    dynamic_decode_position
                        ? KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION : 0,
                    {arguments.begin(), arguments.end()},
                    gemma4_kv_invocation(chunk.idx, global));
            }

            // Sliding/global attention and cross-layer KV sharing all read
            // DDR cache views. There is intentionally no SPM-KV eligibility
            // branch for this mixed-geometry owner: DDR_KV is required.
            append(FmbRouteFamily::ATTENTION, GEMMA4_ATTENTION_SITE,
                   static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                   0, {}, invocation);
            append(FmbRouteFamily::ALL_REDUCE,
                   GEMMA4_PREPARE_ALL_REDUCE_SITE,
                   static_cast<int64_t>(
                       Gemma4AllReduceRoute::PREPARE_RING_INPUT),
                   0, {}, invocation);
            append_linear(GEMMA4_O_LINEAR_SITE, invocation);
            append(FmbRouteFamily::ALL_REDUCE,
                   GEMMA4_ATTENTION_ALL_REDUCE_SITE,
                   fmb_ring_all_reduce_route_selector(
                       chunk.len, hidden_size()),
                   0, {}, invocation);
            append_linear(GEMMA4_GATE_LINEAR_SITE, invocation);
            append_linear(GEMMA4_UP_LINEAR_SITE, invocation);
            append_linear(GEMMA4_DOWN_LINEAR_SITE, invocation);
            append(FmbRouteFamily::ALL_REDUCE,
                   GEMMA4_MLP_ALL_REDUCE_SITE,
                   fmb_ring_all_reduce_route_selector(
                       chunk.len, hidden_size()),
                   0, {}, invocation);
            if (has_ple_) {
                append(FmbRouteFamily::MUTABLE_DMA, GEMMA4_PLE_DMA_SITE,
                       static_cast<int64_t>(
                           Gemma4MutableDmaRoute::DDR_SCATTER_TO_SPM),
                       0, {}, invocation);
                append_linear(GEMMA4_PLE_GATE_LINEAR_SITE, invocation);
                append_linear(GEMMA4_PLE_PROJ_LINEAR_SITE, invocation);
                append(FmbRouteFamily::ALL_REDUCE,
                       GEMMA4_PLE_ALL_REDUCE_SITE,
                       fmb_ring_all_reduce_route_selector(
                           chunk.len, hidden_size()),
                       0, {}, invocation);
            }
        }
        if (has_ple_) {
            for (int64_t layer = 0; layer < num_layers(); ++layer) {
                const at::Tensor& weight =
                    layer_weights_[layer].ple_post_norm_w;
                append(
                    FmbRouteFamily::MUTABLE_DMA,
                    GEMMA4_PLE_NORM_PRELOAD_SITE,
                    static_cast<int64_t>(
                        Gemma4MutableDmaRoute::DDR_BROADCAST_TO_SPM),
                    0, {layer, weight.numel(), has_ple_ ? 1 : 0}, layer);
            }
        }
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA);
        return manifest;
    }

    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& /*manifest*/) const override {
        return {true, FmbGraphLifecycle::RETAINED_CACHE};
    }

    // ========================================================================
    // static_config — plain SEQUENTIAL causal decoder (no D-501 opt-ins).
    // ========================================================================
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers = num_layers();
        cfg.cross_layer_batch_size = num_layers();  // single group
        return cfg;
    }

    ModelDynamicConfig planning_dynamic_config(const ChunkPlan& /*plan*/) override {
        ModelDynamicConfig cfg;
        cfg.chunk_mode = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::AUTO;
        return cfg;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        auto cfg = planning_dynamic_config(plan);
        if (preparing_forward_masks_) prepare_windowed_masks(plan);
        return cfg;
    }

    // ========================================================================
    // subclass_chunk_size_valid — the auto chunk-size search must reject any cs
    // illegal for ANY SDPA mode build_layer_subgraph can emit. gemma4 was only
    // ever exercised at S~6 (single chunk), so this hook was absent and the
    // auto-pick landed on SDPA-illegal chunk sizes at longer S (repro: S=500 ->
    // seq_q=144 violates the flash grid constraint). The LTM modes (global hd512
    // + sliding hd256) run for EVERY build, windowed or not, so validate them
    // unconditionally; a windowed build additionally emits sliding hd256 MASK_2D.
    // Mirrors Qwen3 rpu_qwen3_model.h:1039 (which validates unconditionally too).
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
    // All temps are LayerWide (no aliasing) for the skeleton; the framework
    // auto-shrinks chunk_size to fit SPM. Norm weights are direct-weight
    // (preload DMAs the raw weight; NO gemma2 +1.0 — audit §1).
    // ========================================================================
    std::vector<BufferDecl> declare_buffers_for_mask_residency(
        const LayoutContext& ctx, bool allow_residency,
        bool* retained = nullptr) const {
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

        // T4: a windowed build emits a MIX of SDPA modes (global hd512 LTM/NONE +
        // sliding hd256 MASK_2D/LTM), so sdpa_tmp must be the MAX over all of them.
        // The non-windowed build keeps the EXACT pre-T4 single formula so the
        // <=512 layout stays byte-identical.
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

        // T3: shared all-ones weight for the unweighted v_norm (sized for the
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

        // T6 PLE temps + a zeros residual for the no-residual all-reduce.
        if (has_ple_) {
            int64_t ple_buf = A(cs * ple_dim_ * DWIDTH);
            d.push_back({"ple_gate", ple_buf, 1, 13, StorageClass::Temp, 0, nullptr, ALL});
            d.push_back({"ple_side", ple_buf, 1, 13, StorageClass::Temp, 0, nullptr, ALL});
            // ple_zero: zeros residual fed to all_reduce_sum_residual (only a
            // residual-form all-reduce exists). MUST be Temp, NOT Persistent: it
            // is sized by chunk (cs*h), and a chunk-sized Persistent buffer would
            // change shape between the prefill (cs=S) and decode (cs=1) BUILDs and
            // trip the persistent-shape-stability check (fused_model_base.cpp:351).
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
                this->ctx().consume_physical_route(
                    FmbRouteFamily::MUTABLE_DMA,
                    GEMMA4_PLE_NORM_PRELOAD_SITE,
                    static_cast<int64_t>(
                        Gemma4MutableDmaRoute::DDR_BROADCAST_TO_SPM),
                    /*resolved_flags=*/0,
                    {L, w.numel(), has_ple_ ? 1 : 0}, L);
                rpu_launch_ddr_broadcast_spm_dma(
                    w.data_ptr<c10::Half>(), w.numel(), core0_addr);
            };
            d.push_back(bn);
        }
        bool keep = false;
        if (allow_residency && ctx.use_attn_mask && sliding_window_ > 0 &&
            std::any_of(geom_.begin(), geom_.end(),
                        [](const LayerGeom& g) { return !g.is_global; })) {
            keep = detail::plan_forward_spm_residency(d, {"sdpa_mask"}).retained;
        }
        if (retained) *retained = keep;
        return d;
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        return declare_buffers_for_mask_residency(
            ctx, ctx.physical_manifest_fingerprint != 0);
    }

    std::vector<int64_t> window_mask_schedule_arguments(
        const FmbThreeStageChunkPlan& plan) const {
        const auto first = std::find_if(geom_.begin(), geom_.end(),
            [](const LayerGeom& g) { return !g.is_global; });
        return {sliding_window_, static_cast<int64_t>(first - geom_.begin()),
                static_cast<int64_t>(std::count_if(geom_.begin(), geom_.end(),
                    [](const LayerGeom& g) { return !g.is_global; })),
                plan.compute.plan.chunk_size,
                static_cast<int64_t>(plan.compute.chunks.size()),
                num_q_heads(), num_kv_heads(), head_dim(), attn_tp(), NUM_CORES};
    }

    bool window_mask_dma_for_layer(int layer_idx, const ChunkInfo& chunk) const {
        if (!ctx().has_complete_physical_manifest()) return true;
        const auto& route = ctx().find_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, GEMMA4_WINDOW_MASK_SCHEDULE_SITE);
        TORCH_CHECK(route.selector == 1 || route.selector == 2,
                    "Gemma4 mask requires per-layer or single-chunk retained schedule");
        const auto arguments = window_mask_schedule_arguments(ctx().stage_plan);
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, GEMMA4_WINDOW_MASK_SCHEDULE_SITE,
            route.selector, /*resolved_flags=*/0, arguments);
        if (route.selector == 1) return true;
        TORCH_CHECK(arguments[4] == 1 && arguments[2] > 0 && chunk.idx == 0,
                    "Gemma4 mask residency requires one chunk and a sliding layer");
        // A graph-position predicate, not a host loaded flag. The first sliding
        // layer's DMA executes again on every REPLAY after DDR mask refresh.
        return layer_idx == arguments[1];
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
        const bool load_window_mask = window_mask_dma_for_layer(layer_idx, chunk);
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

        // Phase 1: input RMSNorm (direct-weight).
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "input_norm_w"),
            seq_len, h, eps_);

        // Phase 2: Q proj (+ K/V proj for non-shared layers).
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GEMMA4_Q_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h, /*partition=*/1, /*num_cores=*/tp);
        if (!g.is_kv_shared) {
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, GEMMA4_K_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.k_w, addr(0, "k"),
                seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp);
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, GEMMA4_V_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "input_norm"), lw.v_w, addr(0, "v"),
                seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp);
        }

        // Phase 3: QK-RMSNorm (direct-weight) + RoPE. Dual cos/sin tables are
        // selected by layer type and per-layer head_dim. T2 verified the
        // kernel's NeoX d/2 pairing and half-width table parity against HF.
        c10::Half* cos_ptr = g.is_global ? cos_global_.data_ptr<c10::Half>()
                                         : cos_sliding_.data_ptr<c10::Half>();
        c10::Half* sin_ptr = g.is_global ? sin_global_.data_ptr<c10::Half>()
                                         : sin_sliding_.data_ptr<c10::Half>();
        // Q: q -> output (norm) -> q (rope)
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "q"), addr(0, "output"),
            layer_addr(layer_idx, 0, "q_norm_w"),
            seq_len * local_q_heads, hd, eps_);
        ctx().consume_physical_route(
            FmbRouteFamily::ROPE, GEMMA4_Q_ROPE_SITE,
            static_cast<int64_t>(Gemma4RopeRoute::ROPE_SPM),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_rope_spm_kernel(
            addr(0, "output"), addr(0, "q"), cos_ptr, sin_ptr,
            seq_len, local_q_heads, hd, cos_sin_start, tp);

        if (!g.is_kv_shared) {
            // K: k -> input_norm (norm) -> k (rope)
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "k"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "k_norm_w"),
                seq_len * local_kv_heads, hd, eps_);
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, GEMMA4_K_ROPE_SITE,
                static_cast<int64_t>(Gemma4RopeRoute::ROPE_SPM),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_rope_spm_kernel(
                addr(0, "input_norm"), addr(0, "k"), cos_ptr, sin_ptr,
                seq_len, local_kv_heads, hd, cos_sin_start, tp);

            // T3: v_norm — unweighted RMS on V (with_scale=False, no rope). In-place
            // v -> v with the all-ones weight; the kv insert reads addr("v").
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "v"), addr(0, "v"),
                addr(0, "v_norm_ones"),
                seq_len * local_kv_heads, hd, eps_);

            // KV-cache insert into THIS layer's own slot.
            auto& k_cache = (*ctx().k_caches)[layer_idx];
            auto& v_cache = (*ctx().v_caches)[layer_idx];
            const int64_t kv_invocation =
                gemma4_kv_invocation(chunk.idx, g.is_global);
            const FmbRouteManifestEntry& kv_route =
                ctx().find_physical_route(
                    FmbRouteFamily::KV_INSERT,
                    GEMMA4_KV_INSERT_SITE, kv_invocation);
            const KvInsertSegmentPlan template_plan =
                rpu_kvinsert_segment_plan_from_route_arguments(
                    kv_route.arguments, tp, nkv, hd);
            const bool dynamic_position =
                (kv_route.flags &
                 KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION) != 0;
            TORCH_CHECK(
                !dynamic_position ||
                    (seq_len == 1 &&
                     template_plan.segment(0).position == 0),
                "Gemma4 dynamic KV route requires a canonical "
                "single-token decode template");
            const KvInsertSegmentPlan kv_plan = dynamic_position
                ? rpu_rebase_kvinsert_segment_plan_position(
                      template_plan, cos_sin_start, tp, nkv, hd)
                : template_plan;
            TORCH_CHECK(
                kv_plan.logical_rows() == seq_len &&
                    kv_plan.physical_rows() == seq_len &&
                    kv_plan.segment(0).position == cos_sin_start,
                "Gemma4 KV descriptor geometry drift at invocation ",
                kv_invocation);
            ctx().consume_physical_route(
                FmbRouteFamily::KV_INSERT, GEMMA4_KV_INSERT_SITE,
                static_cast<int64_t>(template_plan.route()),
                kv_route.flags, kv_route.arguments, kv_invocation);
            rpu_launch_insert_kvcache_spm_unified_with_plan(
                k_cache, v_cache,
                addr_offset("k").value, addr_offset("v").value,
                nkv, hd, tp,
                /*k_cache_batch_offset_elems=*/0,
                /*v_cache_batch_offset_elems=*/0,
                /*spm_rows=*/0, kv_plan);
        }

        // Phase 4: SDPA over the SOURCE layer's K/V slot (== own slot when not
        // shared). qk_scale = 1.0 (audit §2).
        //
        // T4 per-(layer,chunk) mask routing: a sliding(hd256) layer whose chunk
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
        at::Tensor k_use = k_src, v_use = v_src;   // Approach-2: may be window-offset views
        int64_t    sdpa_kv = kv_seq_len;
        if (g.is_global || sliding_window_ <= 0 || kv_seq_len <= sliding_window_) {
            mask_type = (original_is_causal_ && seq_len > 1) ? 1 : 0;  // LTM / NONE
        } else {
            // Approach-2 (§9): bound the sliding-layer KV read to the window union
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
            const auto window = windowed_mask_range(chunk, k_src.size(1) * 16);
            const int64_t win_start = window.first, win_len = window.second;
            if (win_start > 0) {
                k_use = k_src.slice(1, win_start / 16);
                v_use = v_src.slice(1, win_start / 16);
            }
            sdpa_kv = win_len;
            mask_off = copy_windowed_mask_to_spm(chunk, win_start, win_len, load_window_mask);
        }
        ctx().consume_physical_attention_route(
            GEMMA4_ATTENTION_SITE, AttentionExecutionPolicy::DDR_KV,
            chunk.idx);
        rpu_launch_sdpa_spm_dispatch(
            sdpa_kernel_, k_use, v_use, mask_type,
            std::optional<double>(qk_scale_),
            addr_offset("q").value, addr_offset("output").value,
            addr_offset("sdpa_tmp").value, mask_off,
            seq_len, nq, nkv, hd, sdpa_kv, tp, NUM_CORES);

        // O proj (row partition over tp cores).
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            GEMMA4_PREPARE_ALL_REDUCE_SITE,
            static_cast<int64_t>(
                Gemma4AllReduceRoute::PREPARE_RING_INPUT),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_prepare_ring_all_reduce_input(addr(0, "oproj"), seq_len, h, tp);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GEMMA4_O_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "output"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp);

        // Phase 5: Gemma2 4-norm attention block (post_attention_layernorm):
        //   residual2 = reduce(oproj) + residual1
        //   residual2 = residual2 - residual1            (isolate attn out)
        //   input_norm = post_attn_norm(residual2)
        //   residual2 = residual1 + input_norm
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            GEMMA4_ATTENTION_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0, {}, chunk.idx);
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

        // Phase 6-8: Gemma2 4-norm MLP block (pre/post_feedforward_layernorm):
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
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GEMMA4_GATE_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
            seq_len, intermediate_size(), h, 1);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "gate"), addr(0, "gate"),
            mlp_elems, ValuOpType::ADD, GeluMode::TANH);   // gelu_pytorch_tanh
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GEMMA4_UP_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "residual1"), lw.up_w, addr(0, "up"),
            seq_len, intermediate_size(), h, 1);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
            mlp_elems, ValuOpType::MUL, c10::Half(1.0));
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GEMMA4_DOWN_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "gate"), lw.down_w, addr(0, "down"),
            seq_len, h, intermediate_size(), 0);
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            GEMMA4_MLP_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0, {}, chunk.idx);
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

        // T6 PLE: per-layer side-input injection (E4B) + layer_scalar. residual1
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
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, GEMMA4_PLE_DMA_SITE,
                static_cast<int64_t>(
                    Gemma4MutableDmaRoute::DDR_SCATTER_TO_SPM),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_ddr_scatter_spm_dma_mutable(
                &side_live_base_, side_off_bytes, seq_len * pc,
                seq_len_ * pc * DWIDTH, addr(0, "ple_side"), NUM_CORES);

            // gate = per_layer_input_gate(residual1)  [h -> pd], COL-partition.
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, GEMMA4_PLE_GATE_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
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
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, GEMMA4_PLE_PROJ_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "ple_gate"), lw.ple_proj_w, addr(0, "down"),
                seq_len, h, pd, /*partition=*/0, /*num_cores=*/NUM_CORES);
            rpu_launch_eltwise_sub_spm_kernel(
                addr(0, "residual1"), addr(0, "residual1"), addr(0, "ple_zero"),
                seq_len * h);
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                GEMMA4_PLE_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(seq_len, h),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "down"), addr(0, "ple_zero"), addr(0, "residual2"),
                seq_len, h);
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual2"), addr(0, "residual2"),
                layer_addr(layer_idx, 0, "ple_post_norm_w"), seq_len, h, eps_);
            rpu_launch_eltwise_add_spm_kernel(
                addr(0, "residual1"), addr(0, "residual2"), addr(0, "residual1"),
                seq_len * h);

            // layer output *= layer_scalar (audit §6; per-layer, != 1.0).
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
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{
            1, static_cast<int64_t>(sdpa_kernel_), static_cast<int64_t>(has_ple_),
            ple_dim_, sliding_window_};
        append_kvinsert_cost_scalar_identity(identity, eps_);
        append_kvinsert_cost_scalar_identity(identity, qk_scale_);
        identity.push_back(static_cast<int64_t>(geom_.size()));
        for (const auto& geometry : geom_) {
            identity.insert(identity.end(), {
                geometry.head_dim, geometry.num_kv_heads, geometry.kv_source_layer,
                static_cast<int64_t>(geometry.is_global),
                static_cast<int64_t>(geometry.is_kv_shared)});
        }
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            append_kvinsert_cost_scalar_identity(identity, weights.layer_scalar);
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.q_norm_w, &weights.k_norm_w,
                    &weights.input_norm_w, &weights.post_attn_norm_w,
                    &weights.pre_ff_norm_w, &weights.post_ff_norm_w,
                    &weights.gate_w, &weights.up_w, &weights.down_w,
                    &weights.ple_gate_w, &weights.ple_proj_w, &weights.ple_post_norm_w}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &cos_sliding_, &sin_sliding_, &cos_global_, &sin_global_,
                &final_norm_w_, &v_norm_ones_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    // Keep the decode window geometry stable for all positions in one 16-key
    // Graph bucket. Only the additive contents change at each step. Prefill
    // uses the union of the chunk's visible windows. Both prepare and the
    // layer consumer use this helper, including the allocated-capacity clamp.
    std::pair<int64_t, int64_t> windowed_mask_range(
        const ChunkInfo& chunk, int64_t capacity) const {
        int64_t start, length;
        if (chunk.len == 1) {
            const int64_t bucket = (chunk.kv_seq_len + 15) / 16;
            start = std::max(int64_t{0}, 16 * (bucket - 1) - sliding_window_);
            length = 16 * bucket - start;
        } else {
            const int64_t first_query = chunk.kv_seq_len - chunk.len;
            start = (std::max(int64_t{0}, first_query - (sliding_window_ - 1)) / 16) * 16;
            length = chunk.kv_seq_len - start;
        }
        return {start, std::min(length, capacity - start)};
    }

    struct WindowedMask {
        ChunkInfo chunk;
        int64_t start = 0;
        int64_t length = 0;
        PreparedMask prepared;
    };

    void prepare_windowed_masks(const ChunkPlan& plan) {
        windowed_masks_.clear();
        if (sliding_window_ <= 0 ||
            ctx().position + ctx().seq_len <= sliding_window_) return;

        // The public cache has a common capacity across all sliding source
        // layers. Require that contract before sharing their DDR mask content.
        int64_t capacity = -1;
        for (const auto& g : geom_) {
            if (g.is_global) continue;
            TORCH_CHECK(ctx().k_caches != nullptr && g.kv_source_layer >= 0 &&
                        static_cast<size_t>(g.kv_source_layer) < ctx().k_caches->size(),
                        "Gemma4 window mask requires its source KV cache");
            const auto& source = ctx().k_caches->at(g.kv_source_layer);
            TORCH_CHECK(source.dim() > 1, "Gemma4 window mask requires swizzled KV");
            const int64_t source_capacity = source.size(1) * 16;
            TORCH_CHECK(capacity < 0 || capacity == source_capacity,
                        "Gemma4 sliding source KV capacities must match");
            capacity = source_capacity;
        }
        if (capacity < 0) return;
        TORCH_CHECK(capacity >= ctx().position + ctx().seq_len,
                    "Gemma4 window mask exceeds the source KV capacity");
        TORCH_CHECK(plan.chunk_size > 0 && plan.num_chunks > 0 &&
                    (ctx().seq_len + plan.chunk_size - 1) / plan.chunk_size == plan.num_chunks,
                    "Gemma4 window masks require the resolved SEQUENTIAL chunk plan");
        windowed_masks_.resize(plan.num_chunks);
        // FMB validates the descriptor against this uniform SEQUENTIAL schedule
        // before dynamic_config. Preparation runs even when fast/deep REPLAY
        // skips build_layer_subgraph, and planning_dynamic_config is side-effect free.
        for (int64_t i = 0; i < plan.num_chunks; ++i) {
            const int64_t offset = i * plan.chunk_size;
            const int64_t length = std::min(plan.chunk_size, ctx().seq_len - offset);
            const ChunkInfo chunk{static_cast<int>(i), offset, length,
                                  ctx().position + offset + length};
            if (chunk.kv_seq_len <= sliding_window_) continue;
            auto& entry = windowed_masks_[i];
            entry.chunk = chunk;
            const auto window = windowed_mask_range(chunk, capacity);
            entry.start = window.first;
            entry.length = window.second;
            TORCH_CHECK(entry.length > 0, "Gemma4 window mask has an empty read range");
            auto& host = windowed_mask_cache_.touch(std::make_pair(chunk.len, chunk.kv_seq_len));
            if (!host.defined() || host.size(1) != entry.length) {
                // A caller may replace its cache with a different capacity.
                // Rebuild the CPU memo if that changes the final bucket clamp.
                host = at::zeros({chunk.len, entry.length},
                    at::TensorOptions().dtype(at::kHalf).device(at::kCPU));
                const c10::Half neg = static_cast<c10::Half>(
                    -std::numeric_limits<float>::infinity());
                auto values = host.accessor<c10::Half, 2>();
                for (int64_t q = 0; q < chunk.len; ++q) {
                    const int64_t absolute_query = chunk.kv_seq_len - chunk.len + q;
                    for (int64_t k = 0; k < entry.length; ++k) {
                        const int64_t absolute_key = entry.start + k;
                        if (!(absolute_key <= absolute_query &&
                              absolute_key > absolute_query - sliding_window_))
                            values[q][k] = neg;
                    }
                }
            }
            // Different chunks can have equal shapes but different content.
            // Their baked DMA sources must not alias while the Graph is pending.
            entry.prepared = sdpa_prepare_mask(
                c10::optional<at::Tensor>(host), /*is_causal=*/false,
                chunk.len, entry.length, sdpa_stable_mask_cache(),
                /*ordinal=*/chunk.idx);
        }
    }

    uint32_t copy_windowed_mask_to_spm(
        const ChunkInfo& chunk, int64_t start, int64_t length, bool load) {
        TORCH_CHECK(chunk.idx >= 0 &&
                    static_cast<size_t>(chunk.idx) < windowed_masks_.size(),
                    "Gemma4 window mask was not prepared for this chunk");
        const auto& entry = windowed_masks_[chunk.idx];
        TORCH_CHECK(entry.chunk.offset == chunk.offset && entry.chunk.len == chunk.len &&
                    entry.chunk.kv_seq_len == chunk.kv_seq_len &&
                    entry.start == start && entry.length == length,
                    "Gemma4 prepared window mask geometry drift");
        // This dedicated slot is read-only to SDPA. Global LTM/NONE layers
        // never write it; multiple chunks retain the original overwrite order.
        const uint32_t mask_off = addr_offset("sdpa_mask").value;
        if (load)
            sdpa_dma_mask_to_spm(entry.prepared, mask_off, chunk.len, length, attn_tp());
        return mask_off;
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerGeom>    geom_;
    at::Tensor cos_sliding_, sin_sliding_, cos_global_, sin_global_;
    at::Tensor final_norm_w_;
    at::Tensor v_norm_ones_;        // T3: all-ones weight for unweighted v_norm
    // T6 PLE state.
    bool has_ple_ = false;
    int64_t ple_dim_ = 0;
    at::Tensor side_input_;          // per-forward [num_layers, seq, ple_dim] (kept alive)
    uint64_t side_live_base_ = 0;    // mutable-DMA base (RpuGetDevAddr of side_input_)
    double eps_ = 1e-6;
    double qk_scale_ = 1.0;     // Gemma4 self.scaling == 1.0 (audit §2)
    int64_t seq_len_ = 0;
    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;
    // T4 sliding-window mask state.
    int64_t sliding_window_ = 0;         // 512 (E4B); 0 disables windowing
    int64_t max_sliding_hd_ = 0;         // widest head_dim among sliding layers (256)
    bool    original_is_causal_ = true;  // true causal intent (ctx().is_causal clobbered on windowed path)
    bool preparing_forward_masks_ = false;
    std::vector<WindowedMask> windowed_masks_;
    // Host mask memo by (qn, kv). The decode cursor advances every step, so this
    // cache must be bounded. CPU entries are safe to evict: sdpa_prepare_mask copies
    // their contents into separate stable RPU slots before DMA captures addresses.
    // A cache miss rebuilds the host mask in O(qn * win_len).
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

std::vector<int64_t> rpu_gemma4_planner_cache_identity(int64_t handle) {
    return Gemma4Registry::get(handle, "rpu_gemma4_planner_cache_identity")
        ->planner_cache_identity();
}

int64_t rpu_gemma4_create() {
    return Gemma4Registry::create();
}

void rpu_gemma4_destroy(int64_t handle) {
    Gemma4Registry::destroy(handle, "rpu_gemma4_destroy");
}

void rpu_gemma4_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    Gemma4Registry::get(handle, "rpu_gemma4_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
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

std::vector<int64_t> rpu_gemma4_resolve_stage_domain(
    int64_t handle, int64_t execution_len, int64_t planning_position,
    int64_t max_kv_len, int64_t requested_chunk_size)
{
    return Gemma4Registry::get(handle, "rpu_gemma4_resolve_stage_domain")
        ->resolve_stage_domain(
            execution_len, planning_position, max_kv_len,
            requested_chunk_size);
}

KvInsertCostDomainQuery rpu_gemma4_kvinsert_cost_domain(
    int64_t handle, at::IntArrayRef descriptor) {
    return Gemma4Registry::get(handle, "rpu_gemma4_kvinsert_cost_domain")
        ->kvinsert_cost_domain("gemma4", descriptor);
}

std::string rpu_gemma4_kvinsert_cost_catalog_sha256(int64_t handle) {
    return Gemma4Registry::get(handle, "rpu_gemma4_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
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
    const std::optional<at::Tensor>& side_input,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gemma4Registry::get(handle, "rpu_gemma4")->forward(
        hidden_states, k_caches, v_caches, attention_mask, position, is_causal,
        chunk_size, side_input, planned_stage_descriptor);
}
