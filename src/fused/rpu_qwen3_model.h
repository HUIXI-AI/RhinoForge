// rpu_qwen3_model.h — shared v3::CausalDecoderModel declaration and inline
// implementation. Specialized subclasses can inherit the Qwen-family decoder
// body; rpu_qwen3_model.cpp owns the registry and C-API free functions.
//
// The inline methods use unqualified ATen and launch helpers. Keep those
// using-directives scoped inside `namespace v3` so includers' global namespaces
// are unaffected.
#pragma once

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_spm_buffers.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <typeinfo>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <utility>
#include <string>

#include "rpu_runtime_state.h"  // shared debug and cross-layer runtime state

namespace v3 {

struct SpmDense2DSpec;
class SpmPortView;
class SpmPortId;

// Namespace-scoped using-directives preserve unqualified-name lookup without
// leaking into the global namespace.
using namespace at;
using namespace ::rhino_lkn;

// Shared constants are inline constexpr so all includers use one definition.
inline constexpr int NUM_CORES = 8;
inline constexpr int DWIDTH    = 2;

// Phase 2.5: lm_head 输出 SPM buffer 大小上限.
// declare_buffers 在 fuse_lm_head_ 状态未知时被框架调用 (ensure_allocated 在 set_lm_head
// 之前可能已经发生过), 所以我们用一个固定上限分配 logits buffer, 避免引入跨调用
// 顺序的 SPM 重分配复杂度。
//   Qwen3 系列 vocab 一致 = 151936 (0.6B/1.7B/4B/8B/14B/32B)
//   per-core: 151936/8 = 18992 halves ≈ 38 KB
//   上限设 200000 (per-core 25000 halves = 50 KB), 给未来更大 vocab 模型留余量
// 总 SPM 占用: 50 KB / core, 远小于 8109 KB / core 预算。
inline constexpr int64_t QWEN3_FUSED_LM_HEAD_MAX_VOCAB = 200000;

// Opt-in AdaRMS scale/shift broadcast. DDR slabs and reverse-allocated SPM slots
// must both be layer-contiguous and ascending. Default OFF.
inline bool read_adarms_fused_bcast_enabled() {
    const char* e = std::getenv("RPU_ADARMS_FUSED_BCAST");
    return e && std::string(e) != "" && std::string(e) != "0" &&
           std::string(e) != "false" && std::string(e) != "False";
}

// Qwen3-VL position_ids keepalive buffer capacity.
//
// M-RoPE reads `[seq_len, 3]` int32 positions from a stable DDR buffer keyed by
// absolute index `ctx().position + chunk.offset`. We pre-allocate a single
// max-sized buffer on set_weights and overwrite per-forward into the appropriate
// row range. Sized to match RPUCache's 8192-token capacity.
//
// 8192 × 3 × 4B = 96 KB (negligible). Buffer is allocated lazily in set_weights
// only when mrope_section is non-empty (Qwen3/Llama paths skip the cost).
inline constexpr int64_t QWEN3_MROPE_MAX_KEEPALIVE_SEQ = 8192;

// Cold per-handle A/B for the exact Qwen3 short-prefill raw-SPM attention
// profile. Unset and 1 admit AUTO; 0 pins the existing DDR-cache SDPA path.
// The value is snapshotted by CausalDecoderModel construction, never re-read
// while a graph-capable handle is live.
inline bool qwen3_spm_kv_by_mha_enabled() {
    const char* e = std::getenv("RPU_QWEN3_SPM_KV_BY_MHA");
    if (!e) return true;
    TORCH_CHECK(
        std::strcmp(e, "0") == 0 || std::strcmp(e, "1") == 0,
        "RPU_QWEN3_SPM_KV_BY_MHA accepts only 0 or 1, got ", e);
    return std::strcmp(e, "1") == 0;
}

// =============================================================================
// CausalDecoderModel — v3::FusedModelBase subclass (Qwen3 + Llama; future Phi/Mistral)
// =============================================================================

class CausalDecoderModel : public FusedModelBase {
public:
    struct LayerWeights {
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor q_norm_w, k_norm_w;
        at::Tensor input_norm_w, post_norm_w;
        at::Tensor gate_w, up_w, down_w;
        // Optional QKV bias (Qwen2.5 / Wall-OSS-0.5). Undefined when
        // has_qkv_bias_ == false (Qwen3 / Llama path).
        at::Tensor q_bias_w, k_bias_w, v_bias_w;
        at::Tensor q_ws, k_ws, v_ws, o_ws, gate_ws, up_ws, down_ws;
    };

    // Whether the loaded weights include explicit Q/K normalization.
    bool has_qk_norm_ = false;

    // Optional QKV bias presence (Qwen2.5 / Wall-OSS-0.5). All three bias lists
    // empty -> has_qkv_bias_=false (Qwen3 / Llama, unchanged). All three size==N
    // -> true: a persistent per-layer SPM bias buffer is scattered per col-partition
    // and its addr is passed to the Q/K/V linear launchers (no GEMM-side add).
    bool has_qkv_bias_ = false;

    // RhinoVLA fast-replay opt-in (per-model; flows into ModelStaticConfig,
    // honored by run_all_layers). Default false → qwen3/llama/wall_oss unaffected.
    bool fast_replay_skip_layer_loop_ = false;
    bool preload_replay_skip_ = false;
    // Batch decode: KV-cache slot for a batch==1 forward (batched prefill).
    // Stamped per forward; consumed by build_layer_subgraph.
    int64_t cache_batch_slot_ = 0;
    bool batch_decode_active_ = false;
    void set_fast_replay_skip_layer_loop(bool e) { fast_replay_skip_layer_loop_ = e; }
    void set_preload_replay_skip(bool e) { preload_replay_skip_ = e; }
    void set_configured_chunk_size_cap(int64_t cs) {
        TORCH_CHECK(cs == 0 || cs >= 16,
                    "configured chunk_size cap must be 0 or at least 16, got ", cs);
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "configured chunk_size cap must be set before the first forward");
        configured_chunk_size_cap_ = cs;
        set_chunk_size_override(0);
        invalidate_model_state();
    }
    void set_equal_two_prefill(bool enabled) {
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "equal-two prefill policy must be set before the first forward");
        equal_two_prefill_ = enabled;
        set_chunk_size_override(0);
        invalidate_model_state();
    }
    void set_linear_acc32(bool enabled) {
        TORCH_CHECK(
            get_last_resolved_chunk_size() == 0,
            "causal decoder linear accumulation mode must be set before the first forward");
        linear_acc32_ = enabled;
        invalidate_model_state();
    }

    CausalDecoderModel() = default;

    // Chunk validity depends on the planned forward's cache position plus its
    // execution length, so planning must use the actual starting position.
    int64_t resolve_prefill_chunk_size(int64_t execution_len,
                                       int64_t position) {
        // Dry planning and execution share the same per-handle chunk override.
        return resolve_chunk_size_for_shape(
            execution_len, position, std::nullopt,
            /*is_causal=*/true);
    }

    // ── Model state (set once via set_weights, invalidates graph cache) ──

    void set_weights(
        at::TensorList q_w_list, at::TensorList k_w_list,
        at::TensorList v_w_list, at::TensorList o_w_list,
        at::TensorList q_norm_list, at::TensorList k_norm_list,
        at::TensorList input_norm_list, at::TensorList post_norm_list,
        at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
        const at::Tensor& cos, const at::Tensor& sin,
        const at::Tensor& final_norm_w,
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps, bool use_silu,
        // Empty defaults preserve existing 1D-RoPE / no-injection behavior.
        // Optional M-RoPE and DeepStack inputs remain gated below.
        at::IntArrayRef mrope_section = {},
        at::IntArrayRef deepstack_lang_layers = {},
        // Wall-OSS-0.5 optional QKV bias. Empty defaults preserve the
        // existing no-bias path (Qwen3 / Llama). All three size==N -> biased.
        at::TensorList q_bias_list = {},
        at::TensorList k_bias_list = {},
        at::TensorList v_bias_list = {},
        at::TensorList q_w_scale_list = {},
        at::TensorList k_w_scale_list = {},
        at::TensorList v_w_scale_list = {},
        at::TensorList o_w_scale_list = {},
        at::TensorList gate_scale_list = {},
        at::TensorList up_scale_list = {},
        at::TensorList down_scale_list = {},
        at::Tensor q_nvfp4_tensor_scales = {},
        at::Tensor k_nvfp4_tensor_scales = {},
        at::Tensor v_nvfp4_tensor_scales = {},
        at::Tensor o_nvfp4_tensor_scales = {},
        at::Tensor gate_nvfp4_tensor_scales = {},
        at::Tensor up_nvfp4_tensor_scales = {},
        at::Tensor down_nvfp4_tensor_scales = {})
    {
        // ─────────────────────────────────────────────────────────────────────
        // Basic sanity
        // ─────────────────────────────────────────────────────────────────────
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "causal_decoder_set_weights: empty weight lists");
        TORCH_CHECK(num_q_heads > 0 && num_kv_heads > 0 && head_dim > 0
                    && hidden_size > 0 && intermediate_size > 0,
                    "causal_decoder_set_weights: model dim params must be positive "
                    "(num_q_heads=", num_q_heads, ", num_kv_heads=", num_kv_heads,
                    ", head_dim=", head_dim, ", hidden_size=", hidden_size,
                    ", intermediate_size=", intermediate_size, ")");
        // Note: Qwen3 permits num_q_heads * head_dim != hidden_size (q-projection
        // dim can exceed hidden_size, e.g. Qwen3-0.6B has 16*128=2048 vs 1024).
        // o_proj maps (num_q_heads*head_dim) → hidden_size.
        TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                    "causal_decoder_set_weights: num_q_heads (", num_q_heads,
                    ") must be divisible by num_kv_heads (", num_kv_heads,
                    ") — Qwen3 uses GQA");

        // ─────────────────────────────────────────────────────────────────────
        // Qwen3-VL DeepStack entry validation.
        //
        // Indices must reference valid layers, be distinct, and not exceed 3
        // (Qwen3-VL ships 3 deepstack mergers; more would require extending
        // the deepstack_dense_src_base_ array below).
        // ─────────────────────────────────────────────────────────────────────
        const bool has_deepstack = (deepstack_lang_layers.size() > 0);
        if (has_deepstack) {
            TORCH_CHECK(deepstack_lang_layers.size() <= 3,
                        "causal_decoder_set_weights: deepstack_lang_layers must have "
                        "at most 3 entries (Qwen3-VL deepstack merger count), got size=",
                        deepstack_lang_layers.size());
            for (size_t i = 0; i < deepstack_lang_layers.size(); ++i) {
                const int64_t L = deepstack_lang_layers[i];
                TORCH_CHECK(L >= 0 && L < N,
                            "causal_decoder_set_weights: deepstack_lang_layers[", i,
                            "]=", L, " must be in [0, num_layers=", N, ")");
                for (size_t j = i + 1; j < deepstack_lang_layers.size(); ++j) {
                    TORCH_CHECK(deepstack_lang_layers[j] != L,
                                "causal_decoder_set_weights: deepstack_lang_layers must "
                                "be distinct; entries ", i, " and ", j,
                                " both equal ", L);
                }
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // All 11 weight lists must have the same length
        // ─────────────────────────────────────────────────────────────────────
        auto check_list = [&](const at::TensorList& list, const char* name) {
            TORCH_CHECK(static_cast<int64_t>(list.size()) == N,
                        "causal_decoder_set_weights: ", name, ".size()=", list.size(),
                        " != num_layers=", N,
                        " (inferred from q_w_list.size()). "
                        "All per-layer weight lists must have identical length.");
        };
        check_list(k_w_list,       "k_w_list");
        check_list(v_w_list,       "v_w_list");
        check_list(o_w_list,       "o_w_list");
        // q_norm/k_norm length checks are
        // gated on non-empty input. Qwen3 (qk_norm_lists populated) -> both lists
        // size==N -> has_qk_norm_=true. Llama (empty lists from
        // _patch_decoder_common(qk_norm_lists=None)) -> has_qk_norm_=false.
        // Reject the asymmetric case (one list non-empty, the other empty) loudly.
        TORCH_CHECK(
            (q_norm_list.size() > 0) == (k_norm_list.size() > 0),
            "causal_decoder_set_weights: q_norm_list and k_norm_list must both be empty "
            "(no QK head-norm; e.g. Llama) OR both have size==num_layers (e.g. "
            "Qwen3). Got q_norm_list.size()=", q_norm_list.size(),
            " vs k_norm_list.size()=", k_norm_list.size(), ".");
        if (q_norm_list.size() > 0 || k_norm_list.size() > 0) {
            check_list(q_norm_list,    "q_norm_list");
            check_list(k_norm_list,    "k_norm_list");
            has_qk_norm_ = true;
        } else {
            has_qk_norm_ = false;
        }
        check_list(input_norm_list,"input_norm_list");
        check_list(post_norm_list, "post_norm_list");
        check_list(gate_list,      "gate_list");
        check_list(up_list,        "up_list");
        check_list(down_list,      "down_list");

        const bool has_scale = !q_w_scale_list.empty();
        auto check_optional_scale_list = [&](const at::TensorList& list,
                                             const char* name) {
            if (has_scale) {
                check_list(list, name);
            } else {
                TORCH_CHECK(list.empty(),
                            "causal_decoder_set_weights: ", name,
                            " must be empty unless all quantized scale lists are provided");
            }
        };
        check_optional_scale_list(k_w_scale_list,  "k_w_scale_list");
        check_optional_scale_list(v_w_scale_list,  "v_w_scale_list");
        check_optional_scale_list(o_w_scale_list,  "o_w_scale_list");
        check_optional_scale_list(gate_scale_list, "gate_scale_list");
        check_optional_scale_list(up_scale_list,   "up_scale_list");
        check_optional_scale_list(down_scale_list, "down_scale_list");
        const bool has_nvfp4 =
            has_scale && q_w_scale_list[0].scalar_type() == at::kByte;
        if (has_scale) {
            check_list(q_w_scale_list, "q_w_scale_list");
        }

        // ─────────────────────────────────────────────────────────────────────
        // Global tensor shape checks
        // ─────────────────────────────────────────────────────────────────────
        TORCH_CHECK(cos.defined() && sin.defined() && final_norm_w.defined(),
                    "causal_decoder_set_weights: cos/sin/final_norm_w must be defined tensors");
        // cos/sin: either [max_pos, head_dim] (HF expanded format) or [max_pos, head_dim/2]
        // (kernel format, half-dim since RoPE applies to pairs). Both are accepted.
        TORCH_CHECK(cos.dim() == 2,
                    "causal_decoder_set_weights: cos must be 2D, got ", cos.dim(), "D");
        TORCH_CHECK(cos.size(-1) == head_dim || cos.size(-1) == head_dim / 2,
                    "causal_decoder_set_weights: cos last dim must be head_dim (", head_dim,
                    ") or head_dim/2 (", head_dim / 2, "), got ", cos.size(-1));
        TORCH_CHECK(sin.sizes() == cos.sizes(),
                    "causal_decoder_set_weights: sin.sizes() must equal cos.sizes(), got ",
                    sin.sizes(), " vs ", cos.sizes());
        TORCH_CHECK(final_norm_w.dim() == 1 && final_norm_w.size(0) == hidden_size,
                    "causal_decoder_set_weights: final_norm_w must be 1D with size=hidden_size (",
                    hidden_size, "), got shape ", final_norm_w.sizes());

        // ─────────────────────────────────────────────────────────────────────
        // Per-layer defined/rank check (quick structural sanity)
        // Full weight-shape verification is subclass-internal detail; we just
        // make sure no tensor is undefined or wrong rank.
        // ─────────────────────────────────────────────────────────────────────
        auto check_all_defined_rank = [&](const at::TensorList& list,
                                          const char* name, int64_t expected_rank) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "causal_decoder_set_weights: ", name, "[", i, "] is undefined");
                TORCH_CHECK(list[i].dim() == expected_rank,
                            "causal_decoder_set_weights: ", name, "[", i, "] must be ",
                            expected_rank, "D, got ", list[i].dim(), "D");
            }
        };
        check_all_defined_rank(q_w_list,        "q_w_list",        2);
        check_all_defined_rank(k_w_list,        "k_w_list",        2);
        check_all_defined_rank(v_w_list,        "v_w_list",        2);
        check_all_defined_rank(o_w_list,        "o_w_list",        2);
        // Skip QK-norm rank checks when has_qk_norm_=false
        // (Llama path: q_norm_list and k_norm_list are intentionally empty).
        if (has_qk_norm_) {
            check_all_defined_rank(q_norm_list,     "q_norm_list",     1);
            check_all_defined_rank(k_norm_list,     "k_norm_list",     1);
        }
        check_all_defined_rank(input_norm_list, "input_norm_list", 1);
        check_all_defined_rank(post_norm_list,  "post_norm_list",  1);
        check_all_defined_rank(gate_list,       "gate_list",       2);
        check_all_defined_rank(up_list,         "up_list",         2);
        check_all_defined_rank(down_list,       "down_list",       2);
        if (has_scale) {
            auto check_quant_pair = [&](const at::TensorList& weights,
                                        const at::TensorList& scales,
                                        const char* name) {
                for (int64_t i = 0; i < N; ++i) {
                    TORCH_CHECK(scales[i].defined(),
                                "causal_decoder_set_weights: ", name,
                                " scale[", i, "] is undefined");
                    if (has_nvfp4 && scales[i].scalar_type() == at::kByte) {
                        TORCH_CHECK(weights[i].scalar_type() == at::kByte,
                                    "causal_decoder_set_weights: ", name,
                                    " NVFP4 weight[", i, "] must be packed uint8");
                        TORCH_CHECK(
                            scales[i].size(0) * 16 == weights[i].size(1) * 2
                                && scales[i].size(1) == weights[i].size(0),
                            "causal_decoder_set_weights: ", name,
                            " NVFP4 scale must be [K/16,N]; weight=",
                            weights[i].sizes(), " scale=", scales[i].sizes());
                        continue;
                    }
                    TORCH_CHECK(
                        scales[i].scalar_type() == at::kHalf,
                        "causal_decoder_set_weights: ", name, " scale[", i,
                        "] must be fp16 for W8/W4 or uint8 FP8 E4M3FN for "
                        "NVFP4, got ", scales[i].scalar_type());
                    if (weights[i].scalar_type() == at::kHalf) {
                        TORCH_CHECK(
                            !has_nvfp4,
                            "causal_decoder_set_weights: mixed NVFP4 ", name,
                            " weight[", i, "] must use the W8A16 int8 fallback, got FP16");
                        TORCH_CHECK(
                            scales[i].dim() == 1 &&
                                scales[i].size(0) == weights[i].size(0),
                            "causal_decoder_set_weights: FP16 ", name,
                            " placeholder scale[", i,
                            "] must be per-channel [N], got ", scales[i].sizes());
                    } else if (weights[i].scalar_type() == at::kChar) {
                        TORCH_CHECK(
                            scales[i].dim() == 1 &&
                                scales[i].size(0) == weights[i].size(0),
                            "causal_decoder_set_weights: W8A16 ", name,
                            " scale[", i, "] must be per-channel [N], got ",
                            scales[i].sizes());
                    } else {
                        TORCH_CHECK(
                            weights[i].scalar_type() == at::kByte,
                            "causal_decoder_set_weights: quantized ", name,
                            " weight[", i, "] must be int8 W8A16 or packed uint8 ",
                            "W4A16, got ", weights[i].scalar_type());
                        TORCH_CHECK(
                            scales[i].dim() == 2,
                            "causal_decoder_set_weights: packed W4A16 ", name,
                            " scale[", i, "] must use the controller-striped ",
                            "pgrp 2D payload, got ", scales[i].sizes());
                    }
                }
            };
            check_quant_pair(q_w_list,    q_w_scale_list,  "q");
            check_quant_pair(k_w_list,    k_w_scale_list,  "k");
            check_quant_pair(v_w_list,    v_w_scale_list,  "v");
            check_quant_pair(o_w_list,    o_w_scale_list,  "o");
            check_quant_pair(gate_list,   gate_scale_list, "gate");
            check_quant_pair(up_list,     up_scale_list,   "up");
            check_quant_pair(down_list,   down_scale_list, "down");
        }

        auto check_nvfp4_tensor_scales = [&](const at::Tensor& ts, const char* name) {
            if (!has_nvfp4) {
                TORCH_CHECK(!ts.defined(),
                            "causal_decoder_set_weights: ", name,
                            " is only valid with uint8 NVFP4 block scales");
                return;
            }
            TORCH_CHECK(ts.defined() && ts.dim() == 1 && ts.size(0) >= N,
                        "causal_decoder_set_weights: ", name,
                        " must be fp32 RPU [at_least_num_layers=", N, "], got ",
                        ts.defined() ? ts.sizes() : at::IntArrayRef{});
            TORCH_CHECK(ts.numel() % 8 == 0,
                        "causal_decoder_set_weights: ", name,
                        " must cover complete 32-byte vector-load blocks "
                        "(numel must be divisible by 8), got ", ts.numel());
            TORCH_CHECK(ts.scalar_type() == at::kFloat,
                        "causal_decoder_set_weights: ", name,
                        " must be fp32, got ", ts.scalar_type());
            TORCH_CHECK(ts.device().type() == c10::DeviceType::PrivateUse1 &&
                            ts.is_contiguous(),
                        "causal_decoder_set_weights: ", name,
                        " must be contiguous on RPU");
        };
        check_nvfp4_tensor_scales(q_nvfp4_tensor_scales, "q_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(k_nvfp4_tensor_scales, "k_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(v_nvfp4_tensor_scales, "v_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(o_nvfp4_tensor_scales, "o_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(gate_nvfp4_tensor_scales, "gate_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(up_nvfp4_tensor_scales, "up_nvfp4_tensor_scales");
        check_nvfp4_tensor_scales(down_nvfp4_tensor_scales, "down_nvfp4_tensor_scales");
        // ─────────────────────────────────────────────────────────────────────
        // Wall-OSS-0.5 optional QKV bias presence.
        // Reject the asymmetric case (some empty, some populated) loudly.
        // All three empty -> no bias (Qwen3 / Llama, unchanged).
        // All three size==N -> 1D per-layer bias vectors gathered into SPM.
        // ─────────────────────────────────────────────────────────────────────
        const bool qkv_bias_present = (q_bias_list.size() > 0);
        TORCH_CHECK(
            (q_bias_list.size() > 0) == (k_bias_list.size() > 0)
            && (k_bias_list.size() > 0) == (v_bias_list.size() > 0),
            "causal_decoder_set_weights: q/k/v_bias_list must all be empty "
            "(no QKV bias; e.g. Qwen3/Llama) OR all have size==num_layers "
            "(e.g. Qwen2.5/Wall-OSS). Got sizes q=", q_bias_list.size(),
            " k=", k_bias_list.size(), " v=", v_bias_list.size(), ".");
        if (qkv_bias_present) {
            check_list(q_bias_list, "q_bias_list");
            check_list(k_bias_list, "k_bias_list");
            check_list(v_bias_list, "v_bias_list");
            check_all_defined_rank(q_bias_list, "q_bias_list", 1);
            check_all_defined_rank(k_bias_list, "k_bias_list", 1);
            check_all_defined_rank(v_bias_list, "v_bias_list", 1);
            has_qkv_bias_ = true;
        } else {
            has_qkv_bias_ = false;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Commit state
        // ─────────────────────────────────────────────────────────────────────
        set_model_params(num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;
        use_silu_ = use_silu;

        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                // Undefined tensors when no qk-norm
                // (Llama path). build_layer_subgraph branches on has_qk_norm_
                // and never dereferences q_norm_w / k_norm_w in the false branch.
                has_qk_norm_ ? q_norm_list[i] : at::Tensor(),
                has_qk_norm_ ? k_norm_list[i] : at::Tensor(),
                input_norm_list[i], post_norm_list[i],
                gate_list[i], up_list[i], down_list[i],
                // Optional QKV bias (undefined when has_qkv_bias_==false).
                has_qkv_bias_ ? q_bias_list[i] : at::Tensor(),
                has_qkv_bias_ ? k_bias_list[i] : at::Tensor(),
                has_qkv_bias_ ? v_bias_list[i] : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(q_w_list[i], q_w_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(k_w_list[i], k_w_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(v_w_list[i], v_w_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(o_w_list[i], o_w_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(gate_list[i], gate_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(up_list[i], up_scale_list[i])
                          : at::Tensor(),
                has_scale ? rpu_retain_linear_quant_scale(down_list[i], down_scale_list[i])
                          : at::Tensor(),
            });
        }
        cos_ = cos;
        sin_ = sin;
        final_norm_w_ = final_norm_w;
        nvfp4_ = has_nvfp4;
        q_nvfp4_tensor_scales_ = q_nvfp4_tensor_scales;
        k_nvfp4_tensor_scales_ = k_nvfp4_tensor_scales;
        v_nvfp4_tensor_scales_ = v_nvfp4_tensor_scales;
        o_nvfp4_tensor_scales_ = o_nvfp4_tensor_scales;
        gate_nvfp4_tensor_scales_ = gate_nvfp4_tensor_scales;
        up_nvfp4_tensor_scales_ = up_nvfp4_tensor_scales;
        down_nvfp4_tensor_scales_ = down_nvfp4_tensor_scales;

        // ─────────────────────────────────────────────────────────────────────
        // Qwen3-VL M-RoPE setup.
        //
        // mrope_section empty → 1D RoPE path (Qwen3 / Llama transparent).
        // mrope_section non-empty → validate (3 axes, sum == head_dim/2, kernel-format
        //   cos/sin) and compute strobe masks + allocate position_ids keepalive.
        // ─────────────────────────────────────────────────────────────────────
        const bool has_mrope = (mrope_section.size() > 0);
        if (has_mrope) {
            // C1: mrope_section must be exactly 3 axes (T, H, W), all non-negative,
            // sum == head_dim/2 (rotary half-dim).
            TORCH_CHECK(mrope_section.size() == 3,
                        "causal_decoder_set_weights: mrope_section must have exactly "
                        "3 entries (T, H, W), got size=", mrope_section.size());
            for (size_t i = 0; i < 3; ++i) {
                TORCH_CHECK(mrope_section[i] >= 0,
                            "causal_decoder_set_weights: mrope_section[", i,
                            "] must be non-negative, got ", mrope_section[i]);
            }
            const int64_t mrope_sum = mrope_section[0] + mrope_section[1] + mrope_section[2];
            TORCH_CHECK(mrope_sum == head_dim / 2,
                        "causal_decoder_set_weights: sum(mrope_section)=", mrope_sum,
                        " must equal head_dim/2=", head_dim / 2);

            // C2: mrope kernel reads cos/sin as [max_seq, head_dim/2] strictly —
            // HF-expanded [max_seq, head_dim] is rejected (entry check C2).
            TORCH_CHECK(cos.size(-1) == head_dim / 2,
                        "causal_decoder_set_weights: mrope path requires cos.last_dim "
                        "== head_dim/2 (kernel format), got ", cos.size(-1),
                        " (head_dim=", head_dim, ").  Adapter must pass kernel-format "
                        "cos/sin via cos_sin= override.");

            mrope_section_.assign(mrope_section.begin(), mrope_section.end());
            mrope_strobe_masks_ = rpu_compute_mrope_strobe_masks(
                mrope_section_, static_cast<int32_t>(head_dim));

            // Lazy-allocate keepalive on RPU device. Stable across forwards;
            // forward() copies per-forward [seq_len, 3] position_ids into the
            // appropriate row range at index `ctx().position`.
            if (!position_ids_keepalive_.defined()
                || position_ids_keepalive_.size(0) != QWEN3_MROPE_MAX_KEEPALIVE_SEQ
                || position_ids_keepalive_.size(1) != 3
                || position_ids_keepalive_.device().type() != at::kPrivateUse1) {
                position_ids_keepalive_ = at::zeros(
                    {QWEN3_MROPE_MAX_KEEPALIVE_SEQ, 3},
                    at::TensorOptions().dtype(at::kInt).device(at::kPrivateUse1));
            } else {
                position_ids_keepalive_.zero_();
            }

            // partial_mrope keepalives [MAX_KEEPALIVE_SEQ, head_dim/2] fp16 —
            // populated per-forward only when the adapter supplies interleaved
            // cos/sin; idle (allocated, never written) for the legacy mrope path.
            const int64_t half = head_dim / 2;
            if (!cos_il_keepalive_.defined()
                || cos_il_keepalive_.size(0) != QWEN3_MROPE_MAX_KEEPALIVE_SEQ
                || cos_il_keepalive_.size(1) != half
                || cos_il_keepalive_.device().type() != at::kPrivateUse1) {
                auto il_opts = at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1);
                cos_il_keepalive_ = at::zeros({QWEN3_MROPE_MAX_KEEPALIVE_SEQ, half}, il_opts);
                sin_il_keepalive_ = at::zeros({QWEN3_MROPE_MAX_KEEPALIVE_SEQ, half}, il_opts);
            }
            // Invalidate the keepalive skip-cache: this block (re)allocated and/or
            // zero_()'d the keepalives, so a later same-src-ptr match must NOT skip
            // the refill (it would read a stale/zeroed buffer).
            ka_last_position_ = -1;
            ka_last_seq_ = -1;
            ka_last_pos_src_ = nullptr;
            ka_last_pos_src_version_ = -1;
            ka_last_cos_src_ = nullptr;
            ka_last_cos_src_version_ = -1;
            ka_last_sin_src_ = nullptr;
            ka_last_sin_src_version_ = -1;
            has_mrope_ = true;
        } else {
            has_mrope_ = false;
            mrope_section_.clear();
            mrope_strobe_masks_ = {};
            position_ids_keepalive_ = at::Tensor{};
        }

        // ─────────────────────────────────────────────────────────────────────
        // Qwen3-VL DeepStack state commit.
        //
        // Store the layer indices that receive visual_embed injection. The dense
        // src bases stay zero-initialized until the first forward — the mutable
        // DMA recorded in build_layer_subgraph will deref &deepstack_dense_src_base_
        // each REPLAY, picking up the live dev_addr that forward() writes from
        // caller-supplied tensors.
        // ─────────────────────────────────────────────────────────────────────
        if (has_deepstack) {
            deepstack_lang_layers_.assign(
                deepstack_lang_layers.begin(), deepstack_lang_layers.end());
        } else {
            deepstack_lang_layers_.clear();
        }
        deepstack_dense_src_base_.fill(0);
        deepstack_dense_refs_.clear();

        // AdaRMS FiLM state (LingBot-VLA expert) is established per set_adarms_*() call,
        // which always runs AFTER set_weights. Clear it here (symmetric with the mrope /
        // deepstack resets above) so a handle re-used as a PLAIN decoder after AdaRMS use
        // does not retain stale flags / shift buffers and keep adding FiLM shifts.
        adarms_ = false;
        adarms_mutable_ = false;
        adarms_unroll_ = false;
        adarms_schedule_select_ = false;
        adarms_schedule_step_ = -1;
        input_shift_.clear();
        post_shift_.clear();
        input_scale_ka_ = at::Tensor{};
        post_scale_ka_  = at::Tensor{};
        input_shift_ka_ = at::Tensor{};
        post_shift_ka_  = at::Tensor{};
        adarms_is_ = at::Tensor{};
        adarms_ish_ = at::Tensor{};
        adarms_ps_ = at::Tensor{};
        adarms_psh_ = at::Tensor{};
        adarms_is_schedule_base_ = adarms_ish_schedule_base_ = 0;
        adarms_ps_schedule_base_ = adarms_psh_schedule_base_ = 0;
        adarms_is_live_base_ = adarms_ish_live_base_ = 0;
        adarms_ps_live_base_ = adarms_psh_live_base_ = 0;

        invalidate_model_state();  // Last non-empty statement of set_weights.
    }

    // AdaRMS FiLM step-update (LingBot-VLA expert). The Euler denoise loop calls
    // this before each forward with the step's folded scale/shift. The scale rides
    // on the input/post norm-weight slots (rmsnorm uses it as the weight); the shift
    // is broadcast-added after each norm. All tensors [hidden] fp16 RPU. Must be
    // called AFTER set_weights (needs num_layers()/layer_weights_).
    void set_adarms_step(
        at::TensorList input_scales, at::TensorList input_shifts,
        at::TensorList post_scales,  at::TensorList post_shifts)
    {
        const int64_t N = num_layers();
        TORCH_CHECK(!layer_weights_.empty(), "set_adarms_step: call set_weights first");
        TORCH_CHECK(!adarms_schedule_select_,
                    "set_adarms_step cannot replace a bound AdaRMS schedule");
        TORCH_CHECK((int64_t)input_scales.size()==N && (int64_t)input_shifts.size()==N
                 && (int64_t)post_scales.size()==N  && (int64_t)post_shifts.size()==N,
                    "set_adarms_step: all 4 lists must have size num_layers=", N);
        const int64_t h = hidden_size();
        input_shift_.resize(N);
        post_shift_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            auto chk = [&](const at::Tensor& t, const char* nm){
                TORCH_CHECK(t.defined() && t.dim()==1 && t.size(0)==h
                         && t.scalar_type()==at::kHalf
                         && t.device().type()==at::kPrivateUse1,
                         "set_adarms_step: ", nm, "[", i, "] must be [", h, "] fp16 RPU");
            };
            chk(input_scales[i],"input_scales"); chk(input_shifts[i],"input_shifts");
            chk(post_scales[i], "post_scales");  chk(post_shifts[i], "post_shifts");
            layer_weights_[i].input_norm_w = input_scales[i];   // scale rides on norm_w
            layer_weights_[i].post_norm_w  = post_scales[i];
            input_shift_[i] = input_shifts[i];
            post_shift_[i]  = post_shifts[i];
        }
        adarms_ = true;
        adarms_mutable_ = false;  // plain build-per-step path: disable mutable/unroll so a
        adarms_unroll_  = false;  // handle downgrading from those modes doesn't keep the old path
        invalidate_model_state();  // Must remain last.
    }

    // set_adarms_step_mutable — REPLAY-safe AdaRMS for the denoise loop (LingBot-VLA).
    // The plain set_adarms_step above swaps the per-layer scale/shift tensors and
    // invalidate_model_state()s, so every Euler step rebuilds the expert graph.
    // This variant instead holds the scale/shift in STABLE-ADDR keepalives whose CONTENTS
    // are refreshed in place per step (no invalidate), and build_*_body emits a per-layer
    // graph-body broadcast DMA that re-loads them into the norm_w/shift SPM slots on every
    // REPLAY (emit_adarms_mut_refresh) — the same discipline as the partial_mrope cos/sin
    // keepalive. So the graph is BUILT once and REPLAYED for steps 1..N (fast-replay-safe).
    // Inputs are [num_layers, hidden] fp16 RPU (one contiguous tensor per role, not lists).
    void set_adarms_step_mutable(
        const at::Tensor& input_scale, const at::Tensor& input_shift,
        const at::Tensor& post_scale,  const at::Tensor& post_shift)
    {
        const int64_t N = num_layers();
        const int64_t h = hidden_size();
        TORCH_CHECK(!layer_weights_.empty(), "set_adarms_step_mutable: call set_weights first");
        TORCH_CHECK(!adarms_schedule_select_,
                    "set_adarms_step_mutable cannot replace a bound AdaRMS schedule");
        auto chk = [&](const at::Tensor& t, const char* nm){
            TORCH_CHECK(t.defined() && t.dim()==2 && t.size(0)==N && t.size(1)==h
                     && t.scalar_type()==at::kHalf && t.device().type()==at::kPrivateUse1,
                     "set_adarms_step_mutable: ", nm, " must be [", N, ",", h, "] fp16 RPU");
        };
        chk(input_scale,"input_scale"); chk(input_shift,"input_shift");
        chk(post_scale,"post_scale");   chk(post_shift,"post_shift");
        if (!adarms_mutable_) {
            // First step: allocate stable keepalives (own storage) + point the per-layer
            // scale/shift at [hidden] views of them, then BUILD once (emit the refresh DMAs).
            input_scale_ka_ = input_scale.clone();
            input_shift_ka_ = input_shift.clone();
            post_scale_ka_  = post_scale.clone();
            post_shift_ka_  = post_shift.clone();
            input_shift_.resize(N);
            post_shift_.resize(N);
            for (int64_t i = 0; i < N; ++i) {
                layer_weights_[i].input_norm_w = input_scale_ka_[i];  // [hidden] view, stable addr
                layer_weights_[i].post_norm_w  = post_scale_ka_[i];
                input_shift_[i] = input_shift_ka_[i];
                post_shift_[i]  = post_shift_ka_[i];
            }
            adarms_ = true;
            adarms_mutable_ = true;
            invalidate_model_state();  // build the mutable graph once
        } else {
            // Subsequent steps: refresh keepalive CONTENTS in place (addr stable) + flush.
            // No invalidate ⇒ the built graph REPLAYs; its body DMAs re-read these buffers.
            input_scale_ka_.copy_(input_scale);
            input_shift_ka_.copy_(input_shift);
            post_scale_ka_.copy_(post_scale);
            post_shift_ka_.copy_(post_shift);
            rpu_ddr_flush_force(input_scale_ka_.data_ptr<c10::Half>());
            rpu_ddr_flush_force(input_shift_ka_.data_ptr<c10::Half>());
            rpu_ddr_flush_force(post_scale_ka_.data_ptr<c10::Half>());
            rpu_ddr_flush_force(post_shift_ka_.data_ptr<c10::Half>());
        }
    }

    // set_adarms_unroll — bind the call-invariant per-step AdaRMS fold for a graph
    // denoise unroll. The four buffers are
    // [num_steps, num_layers, hidden] fp16 RPU (adapter's precomputed fold_all); the
    // per-body refresh (emit_adarms_mut_refresh_*) reads slice (body_iter*N + L) each
    // iteration, so one BUILD unrolls all num_steps steps with per-step FiLM. The
    // preload_callbacks + build-time layer_weights_ views point at step 0 (body_iter 0).
    // Must be called AFTER set_weights.
    void set_adarms_unroll(
        const at::Tensor& input_scale, const at::Tensor& input_shift,
        const at::Tensor& post_scale,  const at::Tensor& post_shift)
    {
        const int64_t N = num_layers();
        const int64_t h = hidden_size();
        TORCH_CHECK(!layer_weights_.empty(), "set_adarms_unroll: call set_weights first");
        TORCH_CHECK(!adarms_schedule_select_,
                    "set_adarms_unroll cannot replace a bound AdaRMS schedule");
        auto chk = [&](const at::Tensor& t, const char* nm){
            TORCH_CHECK(t.defined() && t.dim()==3 && t.size(1)==N && t.size(2)==h
                     && t.scalar_type()==at::kHalf && t.device().type()==at::kPrivateUse1,
                     "set_adarms_unroll: ", nm, " must be [num_steps,", N, ",", h, "] fp16 RPU");
        };
        chk(input_scale,"input_scale"); chk(input_shift,"input_shift");
        chk(post_scale,"post_scale");   chk(post_shift,"post_shift");
        TORCH_CHECK(input_scale.size(0)==input_shift.size(0)
                 && input_scale.size(0)==post_scale.size(0)
                 && input_scale.size(0)==post_shift.size(0),
                 "set_adarms_unroll: all 4 buffers must share num_steps");
        // Own the storage (stable addr across the synchronous forward + REPLAYs).
        adarms_is_  = input_scale.clone();
        adarms_ish_ = input_shift.clone();
        adarms_ps_  = post_scale.clone();
        adarms_psh_ = post_shift.clone();
        // Point the per-layer scale/shift at body_iter-0 [hidden] views so the
        // declare_buffers preload_callbacks (BUILD-time) load step 0; the per-body
        // refresh DMAs then re-read the correct body_iter slice on every iteration.
        input_shift_.resize(N);
        post_shift_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            layer_weights_[i].input_norm_w = adarms_is_[0][i];   // scale rides on norm_w
            layer_weights_[i].post_norm_w  = adarms_ps_[0][i];
            input_shift_[i] = adarms_ish_[0][i];
            post_shift_[i]  = adarms_psh_[0][i];
        }
        adarms_         = true;
        adarms_mutable_ = true;   // declare_buffers shift slots + build-body refresh + FiLM add
        adarms_unroll_  = true;   // refresh reads the body_iter slice (below)
        invalidate_model_state();
    }

    // LingBot2 host-glue optimization: bind the immutable full Euler AdaRMS schedule once,
    // then select a step by changing four stable member live-bases immediately before forward.
    // The captured graph records pointers to those member fields, so full fast replay can skip
    // the C++ op walk and still re-dereference the current step during queue preparation.
    // No schedule copy or flush is needed in the per-step path.
    void bind_adarms_schedule(
        const at::Tensor& input_scale, const at::Tensor& input_shift,
        const at::Tensor& post_scale,  const at::Tensor& post_shift)
    {
        const int64_t N = num_layers();
        const int64_t h = hidden_size();
        TORCH_CHECK(!layer_weights_.empty(), "bind_adarms_schedule: call set_weights first");
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "bind_adarms_schedule must run before the first forward");
        TORCH_CHECK(!adarms_schedule_select_,
                    "bind_adarms_schedule may only bind once per set_weights");
        TORCH_CHECK(!adarms_ && !adarms_mutable_ && !adarms_unroll_,
                    "bind_adarms_schedule cannot replace another AdaRMS mode");
        TORCH_CHECK(adarms_fused_bcast_enabled_,
                    "bind_adarms_schedule requires RPU_ADARMS_FUSED_BCAST=1");
        auto chk = [&](const at::Tensor& t, const char* nm){
            TORCH_CHECK(t.defined() && t.dim()==3 && t.size(0)>0
                     && t.size(1)==N && t.size(2)==h && t.is_contiguous()
                     && t.scalar_type()==at::kHalf && t.device().type()==at::kPrivateUse1,
                     "bind_adarms_schedule: ", nm,
                     " must be contiguous [num_steps,", N, ",", h, "] fp16 RPU");
        };
        chk(input_scale,"input_scale"); chk(input_shift,"input_shift");
        chk(post_scale,"post_scale");   chk(post_shift,"post_shift");
        const int64_t steps = input_scale.size(0);
        TORCH_CHECK(input_shift.size(0)==steps && post_scale.size(0)==steps
                 && post_shift.size(0)==steps,
                    "bind_adarms_schedule: all 4 buffers must share num_steps");

        // Direct tensor references keep the caller's RPU storage alive. Contents are immutable;
        // flush each complete slab once here, instead of four copy_ + four flushes per step.
        adarms_is_  = input_scale;
        adarms_ish_ = input_shift;
        adarms_ps_  = post_scale;
        adarms_psh_ = post_shift;
        rpu_ddr_flush_force_sized(adarms_is_.data_ptr<c10::Half>(), adarms_is_.nbytes());
        rpu_ddr_flush_force_sized(adarms_ish_.data_ptr<c10::Half>(), adarms_ish_.nbytes());
        rpu_ddr_flush_force_sized(adarms_ps_.data_ptr<c10::Half>(), adarms_ps_.nbytes());
        rpu_ddr_flush_force_sized(adarms_psh_.data_ptr<c10::Half>(), adarms_psh_.nbytes());
        adarms_is_schedule_base_ = ::rhino_lkn::RpuGetDevAddr(adarms_is_.data_ptr<c10::Half>());
        adarms_ish_schedule_base_ = ::rhino_lkn::RpuGetDevAddr(adarms_ish_.data_ptr<c10::Half>());
        adarms_ps_schedule_base_ = ::rhino_lkn::RpuGetDevAddr(adarms_ps_.data_ptr<c10::Half>());
        adarms_psh_schedule_base_ = ::rhino_lkn::RpuGetDevAddr(adarms_psh_.data_ptr<c10::Half>());

        input_shift_.resize(N);
        post_shift_.resize(N);
        for (int64_t i = 0; i < N; ++i) {
            layer_weights_[i].input_norm_w = adarms_is_[0][i];
            layer_weights_[i].post_norm_w  = adarms_ps_[0][i];
            input_shift_[i] = adarms_ish_[0][i];
            post_shift_[i]  = adarms_psh_[0][i];
        }
        adarms_ = true;
        adarms_mutable_ = true;
        adarms_unroll_ = false;
        adarms_schedule_select_ = true;
        select_adarms_schedule_step(0);
        invalidate_model_state();
    }

    void select_adarms_schedule_step(int64_t step) {
        TORCH_CHECK(adarms_schedule_select_,
                    "select_adarms_schedule_step requires bind_adarms_schedule");
        TORCH_CHECK(step >= 0 && step < adarms_is_.size(0),
                    "AdaRMS schedule step ", step, " out of range [0,", adarms_is_.size(0), ")");
        const uint64_t offset = static_cast<uint64_t>(step * num_layers() * hidden_size()
                                                      * sizeof(c10::Half));
        adarms_is_live_base_  = adarms_is_schedule_base_  + offset;
        adarms_ish_live_base_ = adarms_ish_schedule_base_ + offset;
        adarms_ps_live_base_  = adarms_ps_schedule_base_  + offset;
        adarms_psh_live_base_ = adarms_psh_schedule_base_ + offset;
        adarms_schedule_step_ = step;
    }

    bool has_bound_adarms_schedule() const { return adarms_schedule_select_; }

    // ── Public entry (called from TORCH_LIBRARY wrapper) ──

    at::Tensor forward(
        const at::Tensor& hidden_states,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        int64_t position,
        bool is_causal,
        // None defaults preserve existing 1D-RoPE / no-injection behavior.
        // Optional M-RoPE and DeepStack inputs remain gated below.
        const std::optional<at::Tensor>& position_ids = std::nullopt,
        const std::optional<std::vector<at::Tensor>>& deepstack_dense_visual_embeds = std::nullopt,
        // partial_mrope path: per-forward interleaved cos/sin [seq_len, head_dim/2]
        // fp16 RPU. When supplied (M-RoPE active), the M-RoPE launchers swap to
        // partial_mrope. None (default) preserves the llama_mrope_interleave path.
        const std::optional<at::Tensor>& rope_cos_il = std::nullopt,
        const std::optional<at::Tensor>& rope_sin_il = std::nullopt,
        // 1D-RoPE position base. -1 uses cos_sin_start =
        // ctx().position + chunk.offset). >=0 decouples the RoPE table index
        // from the cache-insert offset (see rope_position_base_ for rationale).
        int64_t cos_sin_offset = -1,
        // Batch decode: which KV-cache batch slot this (batch=1) forward reads
        // and writes. Batched PREFILL runs one call per sequence with
        // batch_slot = b; batched DECODE passes batch_slot = 0 and covers slots
        // 0..bs-1 internally. The cache base rides regs 8/9 of the KV-insert /
        // SDPA kernels, which `sync_mutable_params()` re-pushes on every REPLAY
        // exactly like `position` — so all B prefills REPLAY one graph rather
        // than building one graph per slot. That is why batch_slot must NOT
        // enter the GraphCache signature.
        int64_t batch_slot = 0,
        // Fail-closed public capability. Python sets this only for the plain
        // Qwen3 adapter; shared CausalDecoderModel users remain batch=1.
    bool allow_batch_decode = false)
    {
        // ─────────────────────────────────────────────────────────────────────
        // Model must be configured (set_weights called) before forward
        // ─────────────────────────────────────────────────────────────────────
        TORCH_CHECK(num_layers() > 0,
                    "CausalDecoderModel::forward called before set_weights. "
                    "Call torch.ops.rpu.causal_decoder_set_weights(...) first.");
        TORCH_CHECK(!layer_weights_.empty() && final_norm_w_.defined(),
                    "CausalDecoderModel::forward: internal state inconsistent, "
                    "set_weights must have been called successfully");

        // ─────────────────────────────────────────────────────────────────────
        // Shape / layout contract — run_all_layers does full hidden_states and
        // k/v cache checks (device / dtype / contiguous / per-layer defined).
        // Keep only the subclass-specific early checks that framework doesn't do.
        // ─────────────────────────────────────────────────────────────────────
        TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                    "CausalDecoderModel::forward: hidden_states must be on RPU device, got ",
                    hidden_states.device());
        TORCH_CHECK(position >= 0,
                    "CausalDecoderModel::forward: position must be non-negative, got ", position);

        // Batch decode: stamp the per-forward KV-cache slot (read by
        // build_layer_subgraph). A non-zero slot is only meaningful for the
        // one-sequence-at-a-time batched prefill; batched decode drives slots
        // itself from hidden_states.size(0).
        TORCH_CHECK(batch_slot >= 0,
                    "CausalDecoderModel::forward: batch_slot must be non-negative, got ",
                    batch_slot);
        TORCH_CHECK(batch_slot == 0 || hidden_states.size(0) == 1,
                    "CausalDecoderModel::forward: batch_slot > 0 selects a single "
                    "cache slot, so it requires hidden_states batch == 1; got batch=",
                    hidden_states.size(0), ", batch_slot=", batch_slot);
        const bool has_batched_cache =
            !k_caches.empty() && k_caches.front().defined() &&
            k_caches.front().dim() > 0 && k_caches.front().size(0) > 1;
        const bool batch_decode_request = hidden_states.size(0) > 1 ||
            batch_slot > 0 || has_batched_cache;
        TORCH_CHECK(!batch_decode_request || allow_batch_decode,
                    "CausalDecoderModel::forward: batched cache/decode is "
                    "enabled only by the plain Qwen3 adapter");
        TORCH_CHECK(!batch_decode_request ||
                        (!fast_replay_skip_layer_loop_ &&
                         !preload_replay_skip_),
                    "CausalDecoderModel::forward: batched cache/decode is "
                    "incompatible with fast-replay host op-stream skipping");
        const bool saved_batch_decode_active = batch_decode_active_;
        batch_decode_active_ = batch_decode_request;
        auto restore_batch_decode_active = c10::make_scope_exit(
            [&] { batch_decode_active_ = saved_batch_decode_active; });
        cache_batch_slot_ = batch_slot;

        // Stamp the per-forward 1D-RoPE position base (read by
        // build_layer_subgraph at cos_sin_start). M-RoPE couples cos_sin_start to
        // the keepalive fill row (ctx().position), so reject the override there.
        TORCH_CHECK(cos_sin_offset < 0 || !has_mrope_,
                    "CausalDecoderModel::forward: cos_sin_offset (1D-RoPE position "
                    "base override) is not supported with M-RoPE active; got "
                    "cos_sin_offset=", cos_sin_offset);
        rope_position_base_ = cos_sin_offset;

        // ─────────────────────────────────────────────────────────────────────
        // Attention mask modes:
        //   - is_causal=true:  原 LTM 路径。attention_mask 可传 (HF 默认 4D causal
        //                      mask) 但被 base 静默忽略 (use_explicit_mask=false)。
        //   - is_causal=false: 必须带显式 2D additive mask。base 把它
        //                      归一成 use_explicit_mask
        //                      → ctx().attention_mask；动作去噪 (任意长 prefix,
        //                      rectangular [H, P+H]) 走 MASK_2D。
        // ─────────────────────────────────────────────────────────────────────
        //   - is_causal=false + 无 mask: 故意的全双向 MASK_NONE。双向
        //     prefill 走这条:build_layer_subgraph 的
        //     sdpa_causal = is_causal && (seq_len>1) =
        //     false -> mask_type=0 (MASK_NONE),与 text_decode 的 seq=1 路是同一条
        //     kernel 分支 (c10::nullopt mask, mask_off=0)。故不再硬拒 is_causal=false 无
        //     mask;调用方对"双向无 mask vs 漏传 mask"负责。

        // ─────────────────────────────────────────────────────────────────────
        // Qwen3-VL M-RoPE forward-entry handling.
        //
        // Contract:
        //   1. set_weights allocated position_ids_keepalive_ at
        //      [MAX_KEEPALIVE_SEQ, 3] int32 RPU DDR (stable across forwards).
        //   2. Caller passes per-forward position_ids [seq_len, 3] int32 RPU;
        //      we copy it into the keepalive at row `ctx().position` and flush.
        //   3. build_layer_subgraph dispatches rpu_launch_mrope_spm_kernel with
        //      pos_offset = ctx().position + chunk.offset (absolute index into
        //      the keepalive — NOT a per-forward offset).
        // ─────────────────────────────────────────────────────────────────────
        const int64_t seq_len_in = hidden_states.size(1);

        // Bound the 1D-RoPE table override independently of KV-cache capacity.
        // Chunks partition [0, seq_len), so the highest table row is the selected
        // base plus seq_len_in. -1 selects the automatic position-based path.
        TORCH_CHECK(cos_sin_offset >= -1,
                    "CausalDecoderModel::forward: cos_sin_offset must be >= -1 "
                    "(-1 = legacy/auto), got ", cos_sin_offset);
        if (cos_sin_offset >= 0) {
            TORCH_CHECK(cos_.defined() && sin_.defined(),
                        "CausalDecoderModel::forward: cos_sin_offset override "
                        "requires RoPE tables to be set (call set_weights first)");
            TORCH_CHECK(seq_len_in <= cos_.size(0)
                            && cos_sin_offset <= cos_.size(0) - seq_len_in
                            && seq_len_in <= sin_.size(0)
                            && cos_sin_offset <= sin_.size(0) - seq_len_in,
                        "CausalDecoderModel::forward: cos_sin_offset overflows "
                        "RoPE table: cos_sin_offset=", cos_sin_offset,
                        " + seq_len=", seq_len_in, " exceeds table rows (cos=",
                        cos_.size(0), ", sin=", sin_.size(0),
                        "). Caller compressed the RoPE position base incorrectly.");
        } else if (!has_mrope_) {
            // The automatic path uses position + chunk.offset and therefore
            // requires the tables to cover position + seq_len_in.
            TORCH_CHECK(cos_.defined() && sin_.defined(),
                        "CausalDecoderModel::forward: RoPE tables must be set "
                        "(call set_weights first)");
            TORCH_CHECK(seq_len_in <= cos_.size(0)
                            && position <= cos_.size(0) - seq_len_in
                            && seq_len_in <= sin_.size(0)
                            && position <= sin_.size(0) - seq_len_in,
                        "CausalDecoderModel::forward: position overflows the RoPE "
                        "table: position=", position, " + seq_len=", seq_len_in,
                        " exceeds table rows (cos=", cos_.size(0), ", sin=",
                        sin_.size(0), "). The table must cover every reachable "
                        "position — see _install_causal_decoder_forward.");
        }

        if (has_mrope_) {
            TORCH_CHECK(position_ids.has_value(),
                        "CausalDecoderModel::forward: position_ids is required "
                        "when set_weights was called with non-empty mrope_section "
                        "(M-RoPE active).");
            const at::Tensor& pos = *position_ids;
            TORCH_CHECK(pos.dim() == 2,
                        "CausalDecoderModel::forward: position_ids must be 2D "
                        "[seq_len, 3], got ", pos.dim(), "D");
            TORCH_CHECK(pos.size(0) == seq_len_in,
                        "CausalDecoderModel::forward: position_ids.size(0)=",
                        pos.size(0), " must equal seq_len=", seq_len_in);
            TORCH_CHECK(pos.size(1) == 3,
                        "CausalDecoderModel::forward: position_ids.size(1) must be 3, got ",
                        pos.size(1));
            TORCH_CHECK(pos.scalar_type() == at::kInt,
                        "CausalDecoderModel::forward: position_ids must be int32, got ",
                        pos.scalar_type());
            TORCH_CHECK(pos.is_contiguous(),
                        "CausalDecoderModel::forward: position_ids must be contiguous");
            TORCH_CHECK(pos.device().type() == at::kPrivateUse1,
                        "CausalDecoderModel::forward: position_ids must be on RPU device, got ",
                        pos.device());
            TORCH_CHECK(position + seq_len_in <= QWEN3_MROPE_MAX_KEEPALIVE_SEQ,
                        "CausalDecoderModel::forward: position+seq_len (",
                        position, "+", seq_len_in, ") exceeds keepalive capacity (",
                        QWEN3_MROPE_MAX_KEEPALIVE_SEQ,
                        "). Bump QWEN3_MROPE_MAX_KEEPALIVE_SEQ in rpu_qwen3_model.cpp.");
            TORCH_CHECK(position_ids_keepalive_.defined(),
                        "CausalDecoderModel::forward: position_ids_keepalive_ undefined "
                        "(set_weights with non-empty mrope_section must precede forward).");
            // Skip copy/flush when the same source pointer and content version are
            // reused at the same (position, seq). Version-disabled inference tensors
            // use pointer identity. Flush only the rows written; set_weights
            // invalidates this cache after keepalive allocation or reset.
            const bool ka_geom_same =
                (position == ka_last_position_ && seq_len_in == ka_last_seq_);
            const auto& pos_version_counter =
                pos.unsafeGetTensorImpl()->version_counter();
            const int64_t pos_src_version = pos_version_counter.enabled()
                ? static_cast<int64_t>(pos_version_counter.current_version())
                : -1;
            const bool pos_content_changed =
                pos_src_version >= 0 &&
                pos_src_version != ka_last_pos_src_version_;
            if (!ka_geom_same || pos.data_ptr() != ka_last_pos_src_ ||
                pos_content_changed) {
                position_ids_keepalive_.narrow(0, position, seq_len_in).copy_(pos);
                rpu_ddr_flush_force_sized(
                    position_ids_keepalive_.narrow(0, position, seq_len_in).data_ptr<int32_t>(),
                    static_cast<size_t>(seq_len_in) * 3 * sizeof(int32_t));
                ka_last_pos_src_ = pos.data_ptr();
                ka_last_pos_src_version_ = pos_src_version;
            }

            // partial_mrope: when the adapter supplies host-baked interleaved
            // cos/sin, refresh the stable keepalive for this forward. This runs in
            // the prologue (before run_all_layers / the FAST_REPLAY body-skip), so
            // the BUILD-baked keepalive addr stays valid while contents update —
            // the same stable-address discipline as position_ids.
            partial_mrope_active_ = rope_cos_il.has_value();
            if (partial_mrope_active_) {
                TORCH_CHECK(rope_sin_il.has_value(),
                            "CausalDecoderModel::forward: rope_cos_il supplied without rope_sin_il");
                const at::Tensor& cil = *rope_cos_il;
                const at::Tensor& sil = *rope_sin_il;
                const int64_t half = cos_il_keepalive_.size(1);
                TORCH_CHECK(cil.dim() == 2 && cil.size(0) == seq_len_in
                            && cil.size(1) == half && sil.sizes() == cil.sizes(),
                            "CausalDecoderModel::forward: rope_cos_il/sin_il must be "
                            "[seq_len, head_dim/2]=[", seq_len_in, ",", half, "], got cos ",
                            cil.sizes(), " sin ", sil.sizes());
                TORCH_CHECK(cil.scalar_type() == at::kHalf && sil.scalar_type() == at::kHalf,
                            "CausalDecoderModel::forward: rope_cos_il/sin_il must be fp16");
                TORCH_CHECK(cil.is_contiguous() && sil.is_contiguous()
                            && cil.device().type() == at::kPrivateUse1
                            && sil.device().type() == at::kPrivateUse1,
                            "CausalDecoderModel::forward: rope_cos_il/sin_il must be "
                            "contiguous on RPU");
                const size_t il_bytes =
                    static_cast<size_t>(seq_len_in) * half * sizeof(c10::Half);
                const auto& cos_version_counter =
                    cil.unsafeGetTensorImpl()->version_counter();
                const int64_t cos_src_version = cos_version_counter.enabled()
                    ? static_cast<int64_t>(cos_version_counter.current_version())
                    : -1;
                const bool cos_content_changed =
                    cos_src_version >= 0 &&
                    cos_src_version != ka_last_cos_src_version_;
                if (!ka_geom_same || cil.data_ptr() != ka_last_cos_src_ ||
                    cos_content_changed) {
                    cos_il_keepalive_.narrow(0, position, seq_len_in).copy_(cil);
                    rpu_ddr_flush_force_sized(
                        cos_il_keepalive_.narrow(0, position, seq_len_in).data_ptr<c10::Half>(),
                        il_bytes);
                    ka_last_cos_src_ = cil.data_ptr();
                    ka_last_cos_src_version_ = cos_src_version;
                }
                const auto& sin_version_counter =
                    sil.unsafeGetTensorImpl()->version_counter();
                const int64_t sin_src_version = sin_version_counter.enabled()
                    ? static_cast<int64_t>(sin_version_counter.current_version())
                    : -1;
                const bool sin_content_changed =
                    sin_src_version >= 0 &&
                    sin_src_version != ka_last_sin_src_version_;
                if (!ka_geom_same || sil.data_ptr() != ka_last_sin_src_ ||
                    sin_content_changed) {
                    sin_il_keepalive_.narrow(0, position, seq_len_in).copy_(sil);
                    rpu_ddr_flush_force_sized(
                        sin_il_keepalive_.narrow(0, position, seq_len_in).data_ptr<c10::Half>(),
                        il_bytes);
                    ka_last_sin_src_ = sil.data_ptr();
                    ka_last_sin_src_version_ = sin_src_version;
                }
            }
            // Commit the geometry for the next forward's skip check (after all three
            // sources are handled for this position/seq).
            ka_last_position_ = position;
            ka_last_seq_ = seq_len_in;
        } else {
            partial_mrope_active_ = false;
        }
        // If position_ids is passed without M-RoPE active, silently ignore
        // (HF callers always supply position_ids; 1D-RoPE path doesn't need them).

        // ─────────────────────────────────────────────────────────────────────
        // Qwen3-VL DeepStack forward-entry handling.
        //
        // Lifecycle (mirrors AdaRMS cond_src_base_ / cond_ref_ template):
        //   1. set_weights stamped deepstack_lang_layers_; build_layer_subgraph
        //      will emit a mutable DMA reading &deepstack_dense_src_base_[i] for
        //      each layer in that list (one src slot per merger index).
        //   2. Caller-supplied dense tensors are held in deepstack_dense_refs_
        //      for the synchronous forward (batch end runs before forward
        //      returns) so their storage cannot be reused by the caching
        //      allocator mid-graph.
        //   3. Per-replay, the mutable DMA cursor-patches the live DDR address.
        //   4. rpu_ddr_flush_force on each dense tensor ensures CPU-side writes
        //      (Python scatter-fill at adapter level) are visible to the RPU
        //      DMA engine.
        // ─────────────────────────────────────────────────────────────────────
        if (!deepstack_lang_layers_.empty()) {
            TORCH_CHECK(deepstack_dense_visual_embeds.has_value(),
                        "CausalDecoderModel::forward: deepstack_dense_visual_embeds "
                        "is required when set_weights was called with non-empty "
                        "deepstack_lang_layers (got None).");
            const auto& embeds = *deepstack_dense_visual_embeds;
            TORCH_CHECK(embeds.size() == deepstack_lang_layers_.size(),
                        "CausalDecoderModel::forward: deepstack_dense_visual_embeds "
                        "must have length == deepstack_lang_layers (",
                        deepstack_lang_layers_.size(), "), got ", embeds.size());
            const int64_t h_local = hidden_size();
            deepstack_dense_refs_.clear();
            deepstack_dense_refs_.reserve(embeds.size());
            for (size_t i = 0; i < embeds.size(); ++i) {
                const at::Tensor& d = embeds[i];
                TORCH_CHECK(d.defined(),
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] is undefined");
                TORCH_CHECK(d.dim() == 2,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] must be 2D [seq_len, hidden], got ", d.dim(), "D");
                TORCH_CHECK(d.size(0) >= seq_len_in,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "].size(0)=", d.size(0),
                            " must be >= seq_len=", seq_len_in,
                            " (max src_offset_bytes baked into BUILD-time graph is "
                            "(seq_len - chunk.len) * hidden * 2)");
                TORCH_CHECK(d.size(1) == h_local,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "].size(1)=", d.size(1),
                            " must equal hidden_size=", h_local);
                TORCH_CHECK(d.scalar_type() == at::kHalf,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] must be fp16, got ", d.scalar_type());
                TORCH_CHECK(d.is_contiguous(),
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] must be contiguous");
                TORCH_CHECK(d.device().type() == at::kPrivateUse1,
                            "CausalDecoderModel::forward: deepstack_dense_visual_embeds[",
                            i, "] must be on RPU device, got ", d.device());
                deepstack_dense_refs_.push_back(d);
                deepstack_dense_src_base_[i] =
                    ::rhino_lkn::RpuGetDevAddr(d.data_ptr<c10::Half>());
                rpu_ddr_flush_force(d.data_ptr<c10::Half>());
            }
        } else if (deepstack_dense_visual_embeds.has_value()
                   && !deepstack_dense_visual_embeds->empty()) {
            // DeepStack disabled in set_weights; caller passed dense tensors
            // anyway. Silently ignore (same pattern as position_ids on 1D-RoPE
            // path) — avoids breaking Qwen3-VL callers that build dense embeds
            // unconditionally and let the C++ side decide.
        }

        const bool decode_fused_lm_head = fuse_lm_head_ && seq_len_in == 1;
        if (decode_fused_lm_head) {
            TORCH_CHECK(vocab_size_ > 0 && vocab_size_ % NUM_CORES == 0,
                        "CausalDecoderModel::forward: fuse_lm_head requires "
                        "vocab_size_=", vocab_size_, " > 0 and divisible by ",
                        NUM_CORES);
            // Batch decode: one logits row per sequence, freshly allocated
            // every forward so callers may accumulate the returned tensor.
            lm_head_logits_out_ = at::empty(
                {hidden_states.size(0), 1, vocab_size_}, hidden_states.options());
            lm_head_logits_dst_base_ =
                ::rhino_lkn::RpuGetDevAddr(lm_head_logits_out_.data_ptr<c10::Half>());
        } else {
            lm_head_logits_out_ = at::Tensor{};
            lm_head_logits_dst_base_ = 0;
        }

        // Only KV_FIRST without a mask consumes Q staging. Its per-shape slot is
        // keyed by {seq_len, width} and never reallocated, preserving fixed-DMA
        // addresses across Graph replay and weight-shape changes.
        if (!is_causal && !attention_mask.has_value()) {
            const int64_t q_ddr_local_dim = (num_q_heads() / attn_tp()) * head_dim();
            const std::pair<int64_t, int64_t> q_key{seq_len_in, q_ddr_local_dim};
            if (q_ddr_slots_.find(q_key) == q_ddr_slots_.end()) {
                TORCH_CHECK(q_ddr_slots_.size() < 128,
                            "CausalDecoderModel: q_ddr_slots_ exceeded 128 distinct "
                            "(seq_len,width) KV_FIRST shapes on one handle — likely a "
                            "runaway shape loop (per-shape Q staging is never freed for "
                            "replay-address stability; normal use sees a handful)");
                q_ddr_slots_.emplace(q_key, at::empty(
                    {static_cast<int64_t>(NUM_CORES), seq_len_in, q_ddr_local_dim},
                    at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1)));
            }
        }
        seq_len_ = seq_len_in;

        // 透传给 run_all_layers (入口做 batch==1 / dim==3 / contiguous / hidden_size 检查)
        at::Tensor result = run_all_layers(hidden_states, k_caches, v_caches,
                                           attention_mask, position, is_causal);

        // ★ Phase 2.5 decode-only fused lm_head 后处理:
        //   build_layer_subgraph 在最后一层追加 lm_head GEMM,随后用 graph-aware
        //   mutable scatter DMA 将 8 个 core 的 SPM shard 写入 lm_head_logits_out_.
        //   返回的 logits tensor 因此由 GraphCache end() 统一 BUILD/REPLAY 写入,
        //   不再在 graph scope 内提前 host memcpy 读取尚未执行的 SPM 内容.
        //
        //   条件: fuse_lm_head_ && decode (seq_len=1)
        //   prefill (seq_len>1) 不走这条路 — 走正常 hidden→Python lm_head 路径.
        if (decode_fused_lm_head) {
            rpu_ddr_flush(lm_head_logits_out_.data_ptr<c10::Half>());
            return lm_head_logits_out_;
        }

        return result;
    }

    // ── Phase 2.5: fused lm_head 入口 (per-instance state, called from C API) ──

    void set_lm_head(
        const at::Tensor& lm_head_weight,
        const std::optional<at::Tensor>& lm_head_scale = std::nullopt) {
        TORCH_CHECK(lm_head_weight.defined(),
                    "set_lm_head: lm_head_weight must be defined");
        TORCH_CHECK(lm_head_weight.dim() == 2,
                    "set_lm_head: lm_head_weight must be 2D [vocab, hidden], got ",
                    lm_head_weight.dim(), "D");
        TORCH_CHECK(lm_head_weight.size(1) == hidden_size(),
                    "set_lm_head: lm_head_weight.size(1)=", lm_head_weight.size(1),
                    " must match hidden_size=", hidden_size());
        TORCH_CHECK(lm_head_weight.size(0) > 0
                    && lm_head_weight.size(0) % NUM_CORES == 0,
                    "set_lm_head: vocab_size=", lm_head_weight.size(0),
                    " must be > 0 and divisible by NUM_CORES=", NUM_CORES);
        TORCH_CHECK(lm_head_weight.size(0) <= QWEN3_FUSED_LM_HEAD_MAX_VOCAB,
                    "set_lm_head: vocab_size=", lm_head_weight.size(0),
                    " exceeds QWEN3_FUSED_LM_HEAD_MAX_VOCAB=",
                    QWEN3_FUSED_LM_HEAD_MAX_VOCAB,
                    ". Bump the constant in rpu_qwen3_model.cpp if a larger vocab "
                    "model is being added (also re-check the 8109 KB/core SPM budget).");
        const bool lm_head_is_w8a16 = lm_head_weight.scalar_type() == at::kChar;
        TORCH_CHECK(lm_head_weight.scalar_type() == at::kHalf || lm_head_is_w8a16,
                    "set_lm_head: lm_head_weight must be FP16 (half) or int8, got ",
                    lm_head_weight.scalar_type());
        TORCH_CHECK(lm_head_weight.device().type() == at::kPrivateUse1,
                    "set_lm_head: lm_head_weight must be on RPU device, got ",
                    lm_head_weight.device());
        if (lm_head_is_w8a16) {
            TORCH_CHECK(lm_head_scale.has_value() && lm_head_scale->defined(),
                        "set_lm_head: int8 lm_head_weight requires "
                        "lm_head_scale");
            TORCH_CHECK(lm_head_scale->dim() == 1,
                        "set_lm_head: lm_head_scale must be 1D [vocab], got ",
                        lm_head_scale->dim(), "D");
            TORCH_CHECK(lm_head_scale->numel() == lm_head_weight.size(0),
                        "set_lm_head: lm_head_scale.numel()=",
                        lm_head_scale->numel(), " must match vocab_size=",
                        lm_head_weight.size(0));
            TORCH_CHECK(lm_head_scale->scalar_type() == at::kHalf,
                        "set_lm_head: lm_head_scale must be FP16 (half), got ",
                        lm_head_scale->scalar_type());
            TORCH_CHECK(lm_head_scale->device().type() == at::kPrivateUse1,
                        "set_lm_head: lm_head_scale must be on RPU device, got ",
                        lm_head_scale->device());
            lm_head_w_scale_ = *lm_head_scale;
        } else {
            lm_head_w_scale_ = at::Tensor{};
        }
        lm_head_w_ = lm_head_weight;
        vocab_size_ = lm_head_weight.size(0);
        fuse_lm_head_ = true;
        invalidate_model_state();  // Last non-empty statement of set_lm_head.
    }

    void clear_lm_head() {
        lm_head_w_ = at::Tensor{};
        lm_head_w_scale_ = at::Tensor{};
        vocab_size_ = 0;
        fuse_lm_head_ = false;
        invalidate_model_state();  // Last non-empty statement of clear_lm_head.
    }

protected:
    // ═══════════════════════════════════════════════════════════════════════
    // declare_buffers
    //
    // Norm weights use per-layer persistent slots and final_norm_w uses a global
    // persistent slot. Preload callbacks refresh them on BUILD or model-state
    // changes, keeping weight DMA outside the chunk loop.
    // ═══════════════════════════════════════════════════════════════════════

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        // KV_FIRST mode is selected (dynamic_config) for every bidirectional
        // no-mask forward, regardless of chunk count. Key the KV_FIRST buffer
        // layout on that SAME condition — NOT on kv_insert_chunk_size>0 — so q_kv
        // is declared whenever emit_kv_first_body runs, including the single-chunk
        // case where the framework leaves kv_insert_chunk_size==0 (KV-insert chunk
        // == compute chunk); otherwise emit would reference an undeclared q_kv.
        // Derive kv_first_layout from the LayoutContext
        // parameter (`ctx`), NOT from the runtime InferenceContext (this->ctx()).
        // declare_buffers MUST be a pure function of the hashed LayoutContext —
        // run_all_layers threads is_causal into layout_ctx and the hash folds in
        // the same derived bit (compute_params_hash_impl), so a handle that
        // switches causal↔bidirectional re-allocates instead of reusing the wrong
        // layout. q_kv is KvInsert-scope and aliases with the Compute buffers, so
        // both the auto-chunk probe estimate and the final allocation stay ≈ the
        // non-KV_FIRST footprint.
        const bool kv_first_layout = !ctx.is_causal && !ctx.use_attn_mask;
        int64_t comp_cs = ctx.chunk_size;
        int64_t kv_cs = ctx.effective_kv_cs();
        int64_t wide_cs = std::max(comp_cs, kv_cs);
        // Batch decode: B sequences' rows are packed into the GEMM M dim, so
        // every ROW-PARALLEL slot (residual/q/k/v/mlp/down/lm_head_out) holds
        // B * chunk rows. sdpa_tmp deliberately does NOT scale: SDPA is emitted
        // once per sequence at seq_q = chunk, so its scratch stays per-sequence.
        // batch_size == 1 (everything but Qwen3 batch decode) → all sizes
        // byte-identical to before.
        int64_t bs = ctx.batch_size > 0 ? ctx.batch_size : 1;
        int64_t h  = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int64_t is_ = intermediate_size();
        int64_t nl  = num_layers();

        // attn_tp() == min(NUM_CORES, nkv). For nkv >= NUM_CORES (Qwen3/Llama)
        // this is NUM_CORES (no-op); for nkv < NUM_CORES (Wall-OSS nkv=2) the
        // whole attention phase runs tensor-parallel across attn_tp() cores.
        int tp = attn_tp();
        int64_t local_q = nq / tp;
        int64_t local_kv = nkv * hd / tp;

        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
        // DMA-safe per-core bias slot (round element count up to a 256 multiple,
        // mirrors the vision encoder's bias buffers).
        auto dma_safe = [&A](int64_t elems) -> int64_t {
            return A(((elems + 255) / 256) * 256 * DWIDTH);
        };
        int64_t q_bias_sz  = dma_safe(local_q * hd);   // (nq/tp)*hd per core
        int64_t kv_bias_sz = dma_safe(local_kv);       // nkv*hd/tp per core

        int64_t res = A(bs * wide_cs * h * DWIDTH);
        int64_t q   = A(bs * comp_cs * local_q * hd * DWIDTH);
        // KV_FIRST Q-split (mirrors Gemma q_kv/q_comp): Phase-1 Q lives in its own
        // KvInsert-scope slot sized at the KV-insert chunk, aliasing with the
        // Compute-scope buffers in the disjoint Phase-2 window. Without this split,
        // LayerWide q/output lose attention↔MLP phase aliasing and can exceed the
        // KV_FIRST SPM budget. Non-KV-FIRST omits q_kv and
        // comp_scope==LayerWide → the SEQUENTIAL layout stays byte-identical.
        int64_t q_kv = A(bs * kv_cs * local_q * hd * DWIDTH);
        int64_t kv  = A(bs * (kv_first_layout ? kv_cs : comp_cs) * local_kv * DWIDTH);
        int64_t mlp = A(bs * comp_cs * (is_ / NUM_CORES) * DWIDTH);
        int64_t down_sz = A(bs * comp_cs * h * DWIDTH);
        int64_t nw  = A(h * DWIDTH);
        int64_t hnw = A(hd * DWIDTH);

        // sdpa_tmp must be sized for the mask type that will ACTUALLY run.
        //   - use_attn_mask → MASK_2D (4): Wall-OSS action-denoise rectangular mask.
        //   - KV_FIRST bidirectional → MASK_NONE (0): is_causal=false runs mask_type=0
        //     (see the no-mask SDPA dispatch in build_layer_subgraph: `sdpa_causal ? 1 : 0`).
        //   - causal PREFILL (comp_cs>1) → LTM (1): the dispatch picks mask 1 for
        //     every chunk of len>1 (build_layer_subgraph: sdpa_causal=is_causal&&seq_len>1).
        //   - causal DECODE (comp_cs<=1) → MASK_NONE (0): a single-token chunk has
        //     seq_len==1 so the SAME dispatch picks mask 0, NOT LTM. (comp_cs here is
        //     ctx.chunk_size = alloc_ctx.chunk_size = chunks[0].len = min(chunk_size,
        //     seq_len) (fused_model_base.cpp run_all_layers); for seq_len==1 decode that
        //     is 1 even though compute_chunks' auto-scan minimum is 16 — so this branch
        //     is LIVE for text_decode, NOT dead.)
        // LTM imposes the extra tk%tm==0 constraint that MASK_NONE does NOT →
        // choose_tile_k picks
        // tile_k_LTM ≤ tile_k_NONE → sizing a path with mask=1 can UNDER-allocate vs
        // a real mask=0 launch (SDPA writes past sdpa_tmp, corrupting adjacent SPM).
        // Match temporary sizing to dispatch: MASK_NONE for KV_FIRST and
        // single-token causal decode, LTM for causal prefill.
        const int sdpa_tmp_mask =
            ctx.use_attn_mask ? 4 : ((kv_first_layout || comp_cs <= 1) ? 0 : 1);
        int64_t tmp = A(sdpa_compute_tmp_v16_size(
            make_sdpa_config(sdpa_tmp_mask), comp_cs) * 32);
        const bool spm_kv_by_mha =
            ctx.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        if (spm_kv_by_mha) {
            // The raw-SPM kernel consumes V transposed as
            // [batch,nKVHeadPerCore,headDim,seq]. Reuse sdpa_tmp for that
            // one-shot value, but size it for the complete tensor.
            tmp = std::max(tmp, A(bs * comp_cs * local_kv * DWIDTH));
        }

        // Explicit 2D mask SPM slot. Rows = chunk (cs), cols = max KV
        // (= mask->size(-1) = P+H), padded to
        // v16. Only declared on the explicit-mask path so the no-mask SPM layout
        // existing no-mask model paths stay byte-identical.
        int64_t mask_sz = ctx.use_attn_mask
            ? A(comp_cs * CeilDiv(ctx.max_kv_seq_len, (int64_t)16) * 32)
            : 0;

        std::vector<BufferDecl> decls;
        const BufferScope attn_kv_scope =
            kv_first_layout ? BufferScope::KvInsert : BufferScope::LayerWide;
        const BufferScope comp_scope =
            kv_first_layout ? BufferScope::Compute : BufferScope::LayerWide;

        // Structural (alive all phases)
        decls.push_back({"residual1",  res,      1, 8, StorageClass::Temp, 0, nullptr});
        decls.push_back({"input_norm", res, 1, 8, StorageClass::Temp, 0, nullptr});
        decls.push_back({"residual2",  0,    0, 0, StorageClass::Temp, 0, "input_norm"});

        // Attention. KV_FIRST: q (Phase-2 reload) and output (SDPA out) are
        // Compute-scope so they alias with the MLP buffers; Phase-1 Q uses the
        // KvInsert-scope q_kv slot (declared only in KV_FIRST). Non-KV_FIRST:
        // comp_scope==LayerWide and q_kv is omitted → byte-identical layout.
        // NOTE: q_kv is keyed on kv_first_layout (= !ctx.is_causal && !ctx.use_attn_mask;
        // Use kv_first_layout, not the old kv_insert_chunk_size>0 condition; see the
        // derivation at the top of this method), so it is declared for EVERY bidirectional
        // no-mask forward, INCLUDING the single-chunk case where kv_insert_chunk_size==0.
        // Raw V -> V^T and by-MHA happen while K/V are still live. Phase 4
        // prevents sdpa_tmp from aliasing its V input; the DDR path retains its
        // established phase-5 layout byte-for-byte.
        const int sdpa_tmp_phase = spm_kv_by_mha ? 4 : 5;
        decls.push_back({"q",          q,    2, 5,
                         StorageClass::Temp, 0, nullptr, comp_scope});
        if (kv_first_layout) {
            decls.push_back({"q_kv",   q_kv, 2, 4, StorageClass::Temp, 0, nullptr, attn_kv_scope});
        }
        decls.push_back({"k",          kv,   2, 4,
                         StorageClass::Temp, 0, nullptr, attn_kv_scope});
        decls.push_back({"v",          kv,   2, 4,
                         StorageClass::Temp, 0, nullptr, attn_kv_scope});
        decls.push_back({"output",     q,    3, 5,
                         StorageClass::Temp, 0, nullptr, comp_scope});
        decls.push_back({"oproj",      res, 5, 5,
                         StorageClass::Temp, 0, nullptr, comp_scope});
        decls.push_back({"sdpa_tmp",   tmp, sdpa_tmp_phase, sdpa_tmp_phase,
                         StorageClass::Temp, 0, nullptr, comp_scope});

        // Explicit 2D mask slot (phase 3..5 — conservative window covering the
        // SDPA consumer so it cannot alias with sdpa_tmp). Conditional
        // so no-mask forwards keep the original layout.
        if (ctx.use_attn_mask) {
            decls.push_back({"sdpa_mask", mask_sz, 3, 5, StorageClass::Temp, 0, nullptr, comp_scope});
        }

        // MLP (phase 7-8, aliased with attention).
        decls.push_back({"gate",       mlp,      7, 8,
                         StorageClass::Temp, 0, nullptr, comp_scope});
        decls.push_back({"up",         mlp,      7, 7, StorageClass::Temp, 0, nullptr, comp_scope});
        decls.push_back({"down",       down_sz, 7, 8,
                         StorageClass::Temp, 0, nullptr, comp_scope});

        // ★ Phase 2.5: decode-only fused lm_head output buffer.
        //   Used at last layer when fuse_lm_head_=true && seq_len=1.
        //   Phase (8, 9): first_step=8 forces a different SPM offset from
        //   residual1 (1-8); without that, the aliasing algorithm would
        //   reuse residual1's slot and the lm_head GEMM would overwrite its
        //   own input. See SPM-aliasing notes.
        //   Batch decode: the fused GEMM runs M = bs, so this holds bs rows of
        //   the col-partitioned vocab slice.
        decls.push_back({"lm_head_out",
                         A(bs * (QWEN3_FUSED_LM_HEAD_MAX_VOCAB / NUM_CORES) * DWIDTH),
                         8, 9, StorageClass::Temp, 0, nullptr});

        // Qwen3-VL DeepStack scratch buffer.
        //   Only allocated when the user opted into DeepStack injection via
        //   set_weights(deepstack_lang_layers=...). Lifetime is a one-tick
        //   window after the MLP residual (phase 8) and before final_norm
        //   (phase 9) — the framework may alias this slot with anything whose
        //   lifetime range doesn't overlap [8, 8].
        //
        // The per-instance cost is chunk_size × hidden_size × element_size
        // per core.
        if (!deepstack_lang_layers_.empty()) {
            decls.push_back({"deepstack_scratch",
                             res,  // chunk_size × hidden × DWIDTH (same as residual1)
                             8, 8, StorageClass::Temp, 0, nullptr});
        }

        // ─────────────────────────────────────────────────────────────────────
        // Norm-weight preloads — PersistentPerLayer + .preload_callback.
        // Callbacks emit only DMA launches into the caller-owned capture.
        // Captured-by-this layer_weights_[L] tensors / final_norm_w_ are
        // stable across forwards because set_weights writes them once and
        // invalidates the graph cache via model_state_gen.
        // ─────────────────────────────────────────────────────────────────────
        // B1: only the replay-safe AdaRMS path re-DMAs these every step, so only it
        // needs the DDR-matching ascending layout. Gated on adarms_mutable_ ⇒ every
        // other decoder (and plain build-per-step AdaRMS) keeps the original
        // descending addresses, byte-identical.
        const bool adarms_fused =
            adarms_mutable_ && adarms_fused_bcast_enabled_;

        {
            BufferDecl input_norm;
            input_norm.name      = "norm_w";
            input_norm.size      = nw;
            input_norm.storage   = StorageClass::PersistentPerLayer;
            input_norm.per_layer = nl;
            input_norm.scope     = BufferScope::LayerWide;
            input_norm.reverse_layer_alloc = adarms_fused;
            input_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    const at::Tensor& weight =
                        layer_weights_[L].input_norm_w;
                    rpu_launch_ddr_broadcast_spm_dma(
                        weight.data_ptr<c10::Half>(), hidden_size(),
                        core0_addr);
                };
            decls.push_back(input_norm);
        }

        {
            BufferDecl post_norm;
            post_norm.name      = "post_norm_w";
            post_norm.size      = nw;
            post_norm.storage   = StorageClass::PersistentPerLayer;
            post_norm.per_layer = nl;
            post_norm.scope     = BufferScope::LayerWide;
            post_norm.reverse_layer_alloc = adarms_fused;
            post_norm.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    const at::Tensor& weight =
                        layer_weights_[L].post_norm_w;
                    rpu_launch_ddr_broadcast_spm_dma(
                        weight.data_ptr<c10::Half>(), hidden_size(),
                        core0_addr);
                };
            decls.push_back(post_norm);
        }

        // AdaRMS FiLM shift vectors (per-layer [hidden]); only when adarms_.
        // Broadcast-added after each RMSNorm in build_layer_subgraph.
        if (adarms_) {
            BufferDecl in_shift;
            in_shift.name      = "input_shift";
            in_shift.size      = nw;
            in_shift.storage   = StorageClass::PersistentPerLayer;
            in_shift.per_layer = nl;
            in_shift.scope     = BufferScope::LayerWide;
            in_shift.reverse_layer_alloc = adarms_fused;
            in_shift.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        input_shift_[L].data_ptr<c10::Half>(),
                        hidden_size(), core0_addr);
                };
            decls.push_back(in_shift);

            BufferDecl po_shift;
            po_shift.name      = "post_shift";
            po_shift.size      = nw;
            po_shift.storage   = StorageClass::PersistentPerLayer;
            po_shift.per_layer = nl;
            po_shift.scope     = BufferScope::LayerWide;
            po_shift.reverse_layer_alloc = adarms_fused;
            po_shift.preload_callback =
                [this](FusedModelBase&, int L, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        post_shift_[L].data_ptr<c10::Half>(),
                        hidden_size(), core0_addr);
                };
            decls.push_back(po_shift);
        }

        // q_norm_w / k_norm_w only present when has_qk_norm_ (Qwen3 yes,
        // Llama no — set in set_weights based on input list presence).
        if (has_qk_norm_) {
            {
                BufferDecl q_norm;
                q_norm.name      = "q_norm_w";
                q_norm.size      = hnw;
                q_norm.storage   = StorageClass::PersistentPerLayer;
                q_norm.per_layer = nl;
                q_norm.scope     = BufferScope::LayerWide;
                q_norm.preload_callback =
                    [this](FusedModelBase&, int L, uint32_t core0_addr) {
                        const at::Tensor& weight =
                            layer_weights_[L].q_norm_w;
                        rpu_launch_ddr_broadcast_spm_dma(
                            weight.data_ptr<c10::Half>(), head_dim(),
                            core0_addr);
                    };
                decls.push_back(q_norm);
            }
            {
                BufferDecl k_norm;
                k_norm.name      = "k_norm_w";
                k_norm.size      = hnw;
                k_norm.storage   = StorageClass::PersistentPerLayer;
                k_norm.per_layer = nl;
                k_norm.scope     = BufferScope::LayerWide;
                k_norm.preload_callback =
                    [this](FusedModelBase&, int L, uint32_t core0_addr) {
                        const at::Tensor& weight =
                            layer_weights_[L].k_norm_w;
                        rpu_launch_ddr_broadcast_spm_dma(
                            weight.data_ptr<c10::Half>(), head_dim(),
                            core0_addr);
                    };
                decls.push_back(k_norm);
            }
        }

        // q_bias / k_bias / v_bias only present when has_qkv_bias_ (Qwen2.5 /
        // Wall-OSS). Each is a persistent per-layer SPM slot scattered per
        // col-partition (core c gets the contiguous output-channel slice that
        // the col-swizzled weight produces on core c — bias is NOT swizzled,
        // matching the verified vision-encoder layout). GQA: Q slot is
        // (nq/tp)*hd per core, K/V slots are nkv*hd/tp per core (different
        // sizes — vision assumes nkv==nq and cannot be copied verbatim).
        if (has_qkv_bias_) {
            auto add_bias = [&](const char* name, int64_t sz,
                                at::Tensor LayerWeights::* member) {
                BufferDecl b;
                b.name      = name;
                b.size      = sz;
                b.storage   = StorageClass::PersistentPerLayer;
                b.per_layer = nl;
                b.scope     = BufferScope::LayerWide;
                b.preload_callback =
                    [this, member](FusedModelBase&, int L, uint32_t core0_addr) {
                        int tp_ = attn_tp();
                        const at::Tensor& bw = layer_weights_[L].*member;
                        int64_t per_core = bw.numel() / tp_;
                        rpu_launch_ddr_scatter_spm_dma(
                            bw.data_ptr<c10::Half>(),
                            per_core, per_core * DWIDTH,
                            core0_addr, tp_);
                    };
                decls.push_back(b);
            };
            add_bias("q_bias", q_bias_sz,  &LayerWeights::q_bias_w);
            add_bias("k_bias", kv_bias_sz, &LayerWeights::k_bias_w);
            add_bias("v_bias", kv_bias_sz, &LayerWeights::v_bias_w);
        }

        // Canonical NVFP4 ACC16 consumes FP32 tensor scales from SPM:
        // params 20/21 carry this projection-family array address and param 23
        // carries layer_id. These model-wide arrays are persistent so graph
        // replay cannot observe out-of-band SPM contents.
        if (nvfp4_) {
            auto add_nvfp4_ts =
                [&](const char* name, at::Tensor CausalDecoderModel::* member) {
                    BufferDecl ts;
                    ts.name       = name;
                    const int64_t scale_elems = (this->*member).numel();
                    ts.size       = A(scale_elems * static_cast<int64_t>(sizeof(float)));
                    ts.storage    = StorageClass::Persistent;
                    ts.per_layer  = 0;
                    ts.scope      = BufferScope::LayerWide;
                    ts.preload_callback =
                        [this, member](FusedModelBase&, int, uint32_t core0_addr) {
                            auto* src = reinterpret_cast<c10::Half*>(
                                (this->*member).data_ptr<float>());
                            rpu_launch_ddr_broadcast_spm_dma(
                                src, 2 * (this->*member).numel(), core0_addr);
                        };
                    decls.push_back(ts);
                };
            add_nvfp4_ts("nvfp4_q_ts",    &CausalDecoderModel::q_nvfp4_tensor_scales_);
            add_nvfp4_ts("nvfp4_k_ts",    &CausalDecoderModel::k_nvfp4_tensor_scales_);
            add_nvfp4_ts("nvfp4_v_ts",    &CausalDecoderModel::v_nvfp4_tensor_scales_);
            add_nvfp4_ts("nvfp4_o_ts",    &CausalDecoderModel::o_nvfp4_tensor_scales_);
            add_nvfp4_ts("nvfp4_gate_ts", &CausalDecoderModel::gate_nvfp4_tensor_scales_);
            add_nvfp4_ts("nvfp4_up_ts",   &CausalDecoderModel::up_nvfp4_tensor_scales_);
            add_nvfp4_ts("nvfp4_down_ts", &CausalDecoderModel::down_nvfp4_tensor_scales_);
        }

        // ★ final_norm fuse: single Persistent slot. The same final_norm_w_
        //   tensor is read at the last layer's RMSNorm. Persistent (not
        //   PersistentPerLayer) because it's one weight, not 28.
        {
            BufferDecl final_norm;
            final_norm.name     = "final_norm_w";
            final_norm.size     = nw;
            final_norm.storage  = StorageClass::Persistent;
            final_norm.per_layer = 0;
            final_norm.scope    = BufferScope::LayerWide;
            final_norm.preload_callback =
                [this](FusedModelBase&, int /*layer*/, uint32_t core0_addr) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        final_norm_w_.data_ptr<c10::Half>(), hidden_size(),
                        core0_addr);
                };
            decls.push_back(final_norm);
        }

        return decls;
    }

    // The DeepStack input to declare_buffers above is invisible to the generic
    // params hash and can move Temp offsets. The other uncovered gates here
    // (adarms_, has_qk_norm_, has_qkv_bias_, nvfp4_ and the nvfp4 scale numels)
    // all guard Persistent/PersistentPerLayer decls, which the persistent-shape
    // check in ensure_allocated already turns into a hard abort — folding those
    // in would convert a loud error into a silent re-layout, which is the wrong
    // direction. Only emptiness matters; no size depends on the layer indices.
    // Subclasses that append their own decls chain onto this value.
    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(
            0, deepstack_lang_layers_.empty() ? 0 : 1);
        return h;
    }

    bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const override {
        if (!qwen3_spm_kv_by_mha_enabled_ ||
            typeid(*this) != typeid(CausalDecoderModel) ||
            layout.attention_policy !=
                AttentionExecutionPolicy::SPM_KV_BY_MHA ||
            position != 0 || !layout.is_causal || layout.use_attn_mask ||
            layout.batch_size != 1 ||
            plan.chunk_mode != ChunkMode::SEQUENTIAL ||
            plan.input.chunks.size() != 1 ||
            plan.qkv.chunks.size() != 1 ||
            plan.compute.chunks.size() != 1 || plan.spans.size() != 1) {
            return false;
        }

        const ChunkInfo& chunk = plan.compute.chunks.front();
        const int64_t seq = chunk.len;
        if ((seq != 16 && seq != 32 && seq != 64 && seq != 128) ||
            chunk.offset != 0 || chunk.kv_seq_len != seq ||
            plan.qkv.chunks.front().len != seq ||
            plan.input.chunks.front().len != seq ||
            plan.spans.front().offset != 0 || plan.spans.front().len != seq ||
            layout.chunk_size != seq || layout.max_kv_seq_len != seq) {
            return false;
        }

        const bool exact_dense_profile =
            num_q_heads() == 16 && num_kv_heads() == 8 &&
            head_dim() == 128 && num_layers() == 28 &&
            ((hidden_size() == 1024 && intermediate_size() == 3072) ||
             (hidden_size() == 2048 && intermediate_size() == 6144));
        if (!exact_dense_profile || layer_weights_.size() != 28 ||
            !use_silu_ || !has_qk_norm_ || has_qkv_bias_ ||
            nvfp4_ ||
            has_mrope_ || partial_mrope_active_ ||
            !deepstack_lang_layers_.empty() || adarms_ || adarms_mutable_ ||
            adarms_unroll_ || adarms_schedule_select_ ||
            rope_position_base_ >= 0 || batch_decode_active_ ||
            sdpa_kernel_ != SdpaKernelType::FLASH_ATTN_SPM) {
            return false;
        }

        return std::all_of(
            layer_weights_.begin(), layer_weights_.end(),
            [](const LayerWeights& layer) {
                return layer.q_w.scalar_type() == at::kHalf &&
                    layer.k_w.scalar_type() == at::kHalf &&
                    layer.v_w.scalar_type() == at::kHalf &&
                    layer.o_w.scalar_type() == at::kHalf &&
                    layer.gate_w.scalar_type() == at::kHalf &&
                    layer.up_w.scalar_type() == at::kHalf &&
                    layer.down_w.scalar_type() == at::kHalf;
            });
    }

    // ═══════════════════════════════════════════════════════════════════════
    // static/dynamic config
    // ═══════════════════════════════════════════════════════════════════════

    SpmFmbTraversalCapability spm_fmb_traversal_capability() const override {
        return SpmFmbTraversalCapability::Unsealed;
    }

    SpmFmbPreloadCapability spm_fmb_preload_capability() const override {
        return SpmFmbPreloadCapability::Unsealed;
    }

    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        // Python GraphSignature/GraphCache owns admission; C++ run_all_layers
        // only emits the op stream. Qwen3 lm_head 融合仍在 decode
        // (seq_len=1) 时由 build_layer_subgraph 末尾追加 lm_head GEMM kernel。
        cfg.num_layers = num_layers();
        // Force single group: cross_layer_batch_size = num_layers().
        //
        // The global
        // runtime knob `g_cross_layer_batch_size` (default 12, set via
        // `torch.rpu.set_cross_layer_batch_size`) leaks across subclasses
        // and is not part of this model's contract. Every v3 subclass pins a
        // single group. If per-Qwen3 multi-group is needed later, wire an
        // explicit per-subclass knob — do not re-enable the global one.
        //
        cfg.cross_layer_batch_size = num_layers();
        cfg.kv_first_fn = static_cast<void(FusedModelBase::*)(int, const ChunkInfo&)>(
                              &CausalDecoderModel::emit_kv_first_body);
        cfg.kv_first_chunk_plan_fn =
            static_cast<ChunkPlan(FusedModelBase::*)(const ChunkPlan&)>(
                &CausalDecoderModel::plan_kv_first_chunks);
        // RhinoVLA fast-replay opt-in (set via causal_decoder_set_fast_replay).
        cfg.fast_replay_skip_layer_loop = fast_replay_skip_layer_loop_;
        // preload-skip is only safe alongside skip_layer_loop (the cursor is set
        // absolutely past the preload region); gate them together so a lone
        // preload-skip can't desync the REPLAY cursor.
        cfg.fast_replay_skip_preload =
            preload_replay_skip_ && fast_replay_skip_layer_loop_;
        cfg.batch_decode_active = batch_decode_active_;
        return cfg;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        ModelDynamicConfig cfg;
        cfg.attention_policy = qwen3_spm_kv_by_mha_enabled_
            ? AttentionExecutionPolicy::AUTO
            : AttentionExecutionPolicy::DDR_KV;
        // Per-forward runtime hook (NOT cached): selects the execution chunk mode
        // from the SAME (is_causal, attention_mask) that run_all_layers threads
        // into layout_ctx. This is safe because the resulting
        // KV_FIRST layout is now part of the allocation identity — declare_buffers
        // derives kv_first_layout from the hashed LayoutContext and the hash folds
        // in the derived bit + kv_insert_chunk_size (compute_params_hash_impl).
        if (!ctx().is_causal && !ctx().attention_mask.has_value()) {
            cfg.chunk_mode = ChunkMode::KV_FIRST;
        } else {
            cfg.chunk_mode = ChunkMode::SEQUENTIAL;
        }
        cfg.inter_layer_io = InterLayerIO::AUTO;     // chunks==1 → SPM_RESIDENT, 否则 → DDR_PINGPONG

        // Prepare the explicit 2D mask once per forward. The base has set ctx_
        // (attention_mask/position/seq_len) and computed `plan`
        // before this call. The action-denoise rectangular [H, P+H] mask only
        // works as one upload per layer when there is a single chunk, so assert
        // it (action horizon H <= chunk_size; the chunk validator is mask-aware so
        // cs>=H is reachable). sdpa_prepare_mask copies the (CPU) mask to a stable
        // RPU DDR slot and returns mask_type=MASK_2D(4).
        if (ctx().attention_mask.has_value()) {
            TORCH_CHECK(plan.num_chunks == 1,
                        "CausalDecoderModel: explicit 2D mask path requires a "
                        "single chunk (got num_chunks=", plan.num_chunks,
                        "). Action horizon must be <= resolved chunk_size.");
            const int64_t seq_q = ctx().seq_len;
            const int64_t seq_k = ctx().position + ctx().seq_len;   // P + H
            // Skip the re-prepare (CPU→RPU upload + stable-slot copy) when the
            // caller passes the SAME (host-memoized) mask at the same dims — the
            // Wall-OSS action mask is _action_mask_cached (stable ptr across frames
            // for a fixed prefix), and this instance is the slot's only same-shape
            // user between forwards. Mirrors the vision #23 last_window_mask_ptr_
            // cache. A different ptr / seq → re-prepare (correct for a length change).
            const void* m_ptr = ctx().attention_mask->data_ptr();
            if (m_ptr != last_attn_mask_ptr_ || seq_q != last_attn_seq_q_
                || seq_k != last_attn_seq_k_) {
                prepared_attn_mask_ = sdpa_prepare_mask(
                    ctx().attention_mask, /*is_causal=*/false, seq_q, seq_k,
                    sdpa_stable_mask_cache());
                last_attn_mask_ptr_ = m_ptr;
                last_attn_seq_q_ = seq_q;
                last_attn_seq_k_ = seq_k;
            }
        }
        return cfg;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // subclass_chunk_size_valid — gate auto chunk_size search on SDPA tile
    // constraints.
    //
    // The framework's auto-pick must check SDPA legality in addition to SPM
    // budget. This hook enforces tile-product, VLM and mask-tiling constraints
    // during the search.
    //
    // sdpa_is_valid_chunk_size is the same function the SDPA kernel-launch path
    // uses to validate its config — so
    // by definition any cs we approve here will not be rejected downstream.
    // ═══════════════════════════════════════════════════════════════════════
    bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len, int64_t position) const override {
        // Qwen/Llama-family causal prefill kernels require a final physical chunk
        // of at least one v16 (for example, 513=512+1 and 2049=512*4+1 are unsafe).
        // Reject that candidate here so both auto and exact
        // planning share the same correctness gate; an off-grid logical length
        // remains valid when another 16-aligned chunk size leaves a safe tail;
        // a short input that fits in one physical chunk has no tail launch.
        const int64_t tail = seq_len % cs;
        if (seq_len > cs && tail > 0 && tail < 16) return false;
        // Apply the certified envelope to every candidate, including overrides.
        if (seq_len > 1 && !chunk_within_envelope(cs)) return false;
        // Apply the per-handle cap to overrides as well; equal_two_prefill_ has a
        // stricter rule below.
        if (seq_len > 1 && configured_chunk_size_cap_ > 0 && !equal_two_prefill_
            && cs > configured_chunk_size_cap_) {
            return false;
        }
        if (equal_two_prefill_ && configured_chunk_size_cap_ > 0
            && position == 0 && seq_len > configured_chunk_size_cap_
            && cs != seq_len / 2) {
            return false;
        }
        // Validate against the mask type that will run. The LTM accumulated-query
        // checks (sQryAcc%16 / %sQry) gate chunk selection and
        // would reject an arbitrary-position action chunk before MASK_2D even
        // runs; with mask=4 those checks are skipped so cs>=H is reachable.
        // 无显式 mask 时一律按 LTM(1) 校验 chunk-size(保守):LTM tile 约束(sQryAcc%16 等)比
        // MASK_NONE 更严，使 auto-scan 选择更小的 chunk，从而满足 SPM 临时区预算。
        // LTM-valid chunk 必然 NONE-runnable,故 launch 仍按 is_causal=false 跑 MASK_NONE,结果正确。
        // KV_FIRST bidirectional (is_causal=false, no mask): the kernel runs
        // MASK_NONE (attn_mask_type=0), which does NOT enforce the
        // position-dependent LTM sQryAcc%16 / %sQry checks, which are gated on
        // attn_mask_type==1. At a non-16-aligned prefix position those
        // checks reject EVERY cs (prefill frame≥1, e.g. position=1566 →
        // sQryAcc=1566, 1566%16=14≠0) although the MASK_NONE kernel runs fine.
        // Validate the position-INDEPENDENT LTM tile/VLM/sync + C3 constraints at
        // position=0 (sQryAcc=0 → the real-position acc check is skipped): yields
        // the same position-0 valid set, so no SPM over-pick, and unblocks
        // multi-frame prefill (position = f·frame_len). The SPM budget check in
        // compute_chunks_impl independently caps cs against actual free space.
        if (!ctx().is_causal && !ctx().attention_mask.has_value()) {
            return sdpa_is_valid_chunk_size(
                make_sdpa_config(1), cs, seq_len, /*position=*/0);
        }
        int mask = ctx().attention_mask.has_value() ? 4 : 1;
        return sdpa_is_valid_chunk_size(make_sdpa_config(mask), cs, seq_len, position);
    }

    int64_t subclass_chunk_size_cap(int64_t seq_len, int64_t position) const override {
        // Hard half of the certified envelope. Called once per resolve, and
        // — this is what makes it a complete gate — called UNCONDITIONALLY by both
        // planner entry points before compute_chunks_impl branches on the override.
        // So the deny lands on the override route too, before any SPM allocation.
        const int64_t envelope_chunk =
            enforce_chunk_envelope(seq_len, position, "CausalDecoderModel");
        // Tighter of the two ceilings wins; 0 means "no ceiling from this source".
        auto tighten = [](int64_t a, int64_t b) {
            if (a <= 0) return b;
            if (b <= 0) return a;
            return std::min(a, b);
        };
        if (equal_two_prefill_ && configured_chunk_size_cap_ > 0
            && position == 0 && seq_len > configured_chunk_size_cap_) {
            TORCH_CHECK(
                seq_len % 32 == 0,
                "equal-two prefill requires a 32-aligned execution length, got ",
                seq_len);
            TORCH_CHECK(
                seq_len <= 2 * configured_chunk_size_cap_,
                "equal-two prefill exceeds the validated two-chunk range: seq_len=",
                seq_len, ", chunk_size_cap=", configured_chunk_size_cap_,
                ", max_seq_len=", 2 * configured_chunk_size_cap_);
            // seq_len/2 is already <= configured_chunk_size_cap_ by the check
            // above, but it must also respect the certified envelope.
            TORCH_CHECK(
                envelope_chunk <= 0 || seq_len / 2 <= envelope_chunk,
                "equal-two prefill would need chunk=", seq_len / 2,
                ", above the certified envelope chunk=", envelope_chunk,
                " (seq_len=", seq_len, ").");
            return seq_len / 2;
        }
        return tighten(configured_chunk_size_cap_, envelope_chunk);
    }


    ChunkPlan plan_kv_first_chunks(const ChunkPlan& compute_plan) {
        // Return the
        // compute plan unchanged — the KV-insert chunk == the compute chunk.
        //
        // An independent KV-insert chunk would need an exact footprint check. The framework's temp estimate
        // (estimate_temporary_total: layer_wide + max(kvin,comp)) under-counts
        // the KV_FIRST KvInsert/Compute aliased layout. compute_chunks_impl
        // already validates comp_cs against the real KV_FIRST declare_buffers footprint (is_causal=false →
        // kv_first_layout at kv_cs==comp_cs), so the compute plan provably fits.
        //
        // KV-insert chunk size is NUMERICALLY NEUTRAL (Phase-1 KV staging only;
        // Phase-2 attention reads the full KV cache), so this preserves the
        // output bit-for-bit. A separate KV-insert batching path would need a
        // footprint model that matches alloc_temporary_aliased exactly.
        // kv_insert_chunk_size (== 0 here) is still
        // folded into compute_params_hash_impl so any future upgrade re-keys the
        // allocation instead of aliasing a stale layout.
        return compute_plan;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Shared QK RoPE dispatch for sequential and KV_FIRST paths. It selects 1D,
    // M-RoPE or partial M-RoPE from model state and dereferences only the active
    // path's keepalive inputs. rotary_dim == dim at all current sites.
    // ─────────────────────────────────────────────────────────────────────────
    void emit_qk_rope(const char* in_slot, const char* out_slot,
                      int64_t n_heads, int64_t dim, int64_t seq_len,
        int64_t cos_sin_start, int tp) {
        if (has_mrope_ && partial_mrope_active_) {
            rpu_launch_partial_mrope_spm_kernel(
                addr(0, in_slot), addr(0, out_slot),
                cos_il_keepalive_.data_ptr<c10::Half>(),
                sin_il_keepalive_.data_ptr<c10::Half>(),
                cos_sin_start, seq_len, n_heads, dim,
                /*rotary_dim=*/dim, tp);
        } else if (has_mrope_) {
            rpu_launch_mrope_spm_kernel(
                addr(0, in_slot), addr(0, out_slot),
                cos_.data_ptr<c10::Half>(), sin_.data_ptr<c10::Half>(),
                position_ids_keepalive_.data_ptr<int32_t>(), cos_sin_start,
                mrope_strobe_masks_, seq_len, n_heads, dim, tp);
        } else {
            rpu_launch_rope_spm_kernel(
                addr(0, in_slot), addr(0, out_slot),
                cos_.data_ptr<c10::Half>(), sin_.data_ptr<c10::Half>(),
                seq_len, n_heads, dim, cos_sin_start);
        }
    }

    // Replay-safe AdaRMS (set_adarms_step_mutable): before each RMSNorm, re-load this
    // layer's scale (norm_w slot) and shift (input_shift/post_shift slot) into SPM from the
    // stable keepalive. Recorded into the graph body ⇒ re-executes on every REPLAY, picking
    // up the per-step contents the adapter wrote. No-op unless adarms_mutable_ (so every
    // other decoder — and the plain build-per-step AdaRMS — is unaffected).
    // B1: one broadcast DMA fills all `N` layers of a per-layer [hidden] SPM slot.
    // Requires the slot's layers to be ASCENDING and gap-free — guaranteed by
    // BufferDecl::reverse_layer_alloc, VERIFIED here rather than assumed: if the
    // layout is ever the other way round (flag mismatch, decl edit, allocator
    // change) a silent write would scramble scale/shift across layers and only show
    // up as a wrong action. TORCH_CHECK turns that into a loud build-time failure.
    // BUILD-time only (not per REPLAY), so the N address lookups are free.
    uint32_t validate_adarms_stack_layout(int64_t N, int64_t h, const char* slot) {
        TORCH_CHECK(N >= 2, "validate_adarms_stack_layout: needs >=2 layers, got ", N);
        const uint32_t a0 = layer_addr(0, 0, slot);
        // SpmAllocator::ALIGN == 256; spelled out rather
        // than pulled from a helper so the stride here cannot drift from the decl's
        // `nw = A(h*DWIDTH)` without the check below firing.
        const uint32_t stride = (uint32_t)(((h * DWIDTH + 255) / 256) * 256);
        for (int64_t L = 1; L < N; ++L) {
            TORCH_CHECK(layer_addr((int)L, 0, slot) == a0 + (uint32_t)L * stride,
                "validate_adarms_stack_layout: SPM slot '", slot, "' is not ascending "
                "contiguous at layer ", L,
                ". RPU_ADARMS_FUSED_BCAST requires reverse_layer_alloc on this decl.");
        }
        return a0;
    }

    void emit_adarms_stack_bcast(c10::Half* ddr, int64_t N, int64_t h, const char* slot) {
        rpu_launch_ddr_broadcast_spm_dma(
            ddr, N * h, validate_adarms_stack_layout(N, h, slot));
    }

    void emit_adarms_stack_bcast_mutable(
        const uint64_t* live_base, int64_t N, int64_t h, const char* slot)
    {
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            live_base, /*src_offset_bytes=*/0, N * h,
            validate_adarms_stack_layout(N, h, slot), NUM_CORES);
    }

    void emit_adarms_mut_refresh_input(int64_t L) {
        if (!adarms_mutable_) return;  // guarded ⇒ input_shift_[L] is resized (safe)
        const int64_t h = hidden_size();
        if (adarms_fused_bcast_enabled_) {
            // Whole 36-layer stack in one DMA per role, emitted once at layer 0.
            // Emitting at L==0 (instead of hoisting out of the layer loop) keeps the
            // build structure untouched: the DMA still lands in the graph body ahead
            // of every layer's norm, so all layers see their data before use.
            if (L != 0) return;
            const int64_t N = num_layers();
            if (adarms_schedule_select_) {
                emit_adarms_stack_bcast_mutable(
                    &adarms_is_live_base_, N, h, "norm_w");
                emit_adarms_stack_bcast_mutable(
                    &adarms_ish_live_base_, N, h, "input_shift");
                return;
            }
            emit_adarms_stack_bcast(adarms_unroll_
                ? adarms_is_.data_ptr<c10::Half>() + ctx().body_iter * N * h
                : input_scale_ka_.data_ptr<c10::Half>(), N, h, "norm_w");
            emit_adarms_stack_bcast(adarms_unroll_
                ? adarms_ish_.data_ptr<c10::Half>() + ctx().body_iter * N * h
                : input_shift_ka_.data_ptr<c10::Half>(), N, h, "input_shift");
            return;
        }
        // In-graph unroll: read this body_iter's (scale, shift) slice from the
        // [num_steps, num_layers, hidden] fold buffers. Single-step (adarms_unroll_
        // false) keeps the mutable-keepalive src ⇒ byte-identical.
        c10::Half* scale_ptr = adarms_unroll_
            ? adarms_is_.data_ptr<c10::Half>() + (ctx().body_iter * num_layers() + L) * h
            : layer_weights_[L].input_norm_w.data_ptr<c10::Half>();
        c10::Half* shift_ptr = adarms_unroll_
            ? adarms_ish_.data_ptr<c10::Half>() + (ctx().body_iter * num_layers() + L) * h
            : input_shift_[L].data_ptr<c10::Half>();
        rpu_launch_ddr_broadcast_spm_dma(scale_ptr, h, layer_addr((int)L, 0, "norm_w"));
        rpu_launch_ddr_broadcast_spm_dma(shift_ptr, h, layer_addr((int)L, 0, "input_shift"));
    }
    void emit_adarms_mut_refresh_post(int64_t L) {
        if (!adarms_mutable_) return;
        const int64_t h = hidden_size();
        if (adarms_fused_bcast_enabled_) {
            if (L != 0) return;
            const int64_t N = num_layers();
            if (adarms_schedule_select_) {
                emit_adarms_stack_bcast_mutable(
                    &adarms_ps_live_base_, N, h, "post_norm_w");
                emit_adarms_stack_bcast_mutable(
                    &adarms_psh_live_base_, N, h, "post_shift");
                return;
            }
            emit_adarms_stack_bcast(adarms_unroll_
                ? adarms_ps_.data_ptr<c10::Half>() + ctx().body_iter * N * h
                : post_scale_ka_.data_ptr<c10::Half>(), N, h, "post_norm_w");
            emit_adarms_stack_bcast(adarms_unroll_
                ? adarms_psh_.data_ptr<c10::Half>() + ctx().body_iter * N * h
                : post_shift_ka_.data_ptr<c10::Half>(), N, h, "post_shift");
            return;
        }
        c10::Half* scale_ptr = adarms_unroll_
            ? adarms_ps_.data_ptr<c10::Half>() + (ctx().body_iter * num_layers() + L) * h
            : layer_weights_[L].post_norm_w.data_ptr<c10::Half>();
        c10::Half* shift_ptr = adarms_unroll_
            ? adarms_psh_.data_ptr<c10::Half>() + (ctx().body_iter * num_layers() + L) * h
            : post_shift_[L].data_ptr<c10::Half>();
        rpu_launch_ddr_broadcast_spm_dma(scale_ptr, h, layer_addr((int)L, 0, "post_norm_w"));
        rpu_launch_ddr_broadcast_spm_dma(shift_ptr, h, layer_addr((int)L, 0, "post_shift"));
    }

    void emit_kv_first_body(int layer_idx, const ChunkInfo& chunk) {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        const int64_t rope_base =
            (!has_mrope_ && rope_position_base_ >= 0) ? rope_position_base_ : ctx().position;
        int64_t cos_sin_start = rope_base + chunk.offset;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        int tp = attn_tp();
        const uint32_t input_residual = addr(0, "residual1");

        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        emit_adarms_mut_refresh_input(layer_idx);  // replay-safe AdaRMS: refresh scale+shift SPM
        rpu_launch_rmsnorm_spm_kernel(
            input_residual, addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "norm_w"),   // adarms_: scale rides here
            seq_len, h, eps_);
        if (adarms_) {  // AdaRMS FiLM: input_norm += input_shift (LingBot-VLA expert)
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                layer_addr(layer_idx, 0, "input_shift"),
                addr(0, "input_norm"), addr(0, "input_norm"),
                seq_len, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
        }

        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q_kv"),
            seq_len, nq * hd, h, /*partition=*/1, /*num_cores=*/tp,
            qkv_fused_bias_addr(layer_idx, "q_bias"), lw.q_ws,
            nvfp4_scale_addr("nvfp4_q_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);
        emit_nvfp4_qkv_bias(
            layer_idx, "q_bias", "q_kv", seq_len, nq * hd / tp);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
            qkv_fused_bias_addr(layer_idx, "k_bias"), lw.k_ws,
            nvfp4_scale_addr("nvfp4_k_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);
        emit_nvfp4_qkv_bias(
            layer_idx, "k_bias", "k", seq_len, nkv * hd / tp);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
            qkv_fused_bias_addr(layer_idx, "v_bias"), lw.v_ws,
            nvfp4_scale_addr("nvfp4_v_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);
        emit_nvfp4_qkv_bias(
            layer_idx, "v_bias", "v", seq_len, nkv * hd / tp);

        // Q-split: Phase-1 Q lives in the KvInsert-scope "q_kv" slot. The QK-norm
        // intermediate reuses "input_norm" as scratch (free after the q/k/v
        // linears above consumed the LN output, and before the k-norm below
        // overwrites it in sequence) — this avoids a Compute-scope "output"
        // write inside the KvInsert subgraph. local_q_heads*hd <= h, so the
        // LayerWide input_norm slot is always large enough.
        int64_t local_q_heads = nq / tp;
        int64_t local_kv_dim  = nkv * hd / tp;
        bool has_qk_head_norm = has_qk_norm_;
        // RoPE 经 emit_qk_rope 单一真相源：自动含 partial_mrope 叶，
        // 与 build_layer_subgraph 同枝 → KV_FIRST 不再漏抄 RoPE 分支。Q-split:Phase-1 Q
        // 在 KvInsert-scope "q_kv";QK-norm 中间值复用 "input_norm" 作 scratch。
        if (has_qk_head_norm) {
            int64_t local_kv_heads = nkv / tp;
            // Q: q_kv → input_norm (norm) → q_kv (RoPE)
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q_kv"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "q_norm_w"),
                seq_len * local_q_heads, hd, eps_);
            emit_qk_rope("input_norm", "q_kv", local_q_heads, hd, seq_len, cos_sin_start, tp);
            // K: k → input_norm (norm) → k (RoPE)
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "k"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "k_norm_w"),
                seq_len * local_kv_heads, hd, eps_);
            emit_qk_rope("input_norm", "k", local_kv_heads, hd, seq_len, cos_sin_start, tp);
        } else {
            emit_qk_rope("q_kv", "q_kv", local_q_heads, hd, seq_len, cos_sin_start, tp);
            emit_qk_rope("k", "k", 1, local_kv_dim, seq_len, cos_sin_start, tp);
        }

        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        rpu_launch_insert_kcache_spm_unified(
            k_cache, ctx().position + chunk.offset, addr_offset("k").value,
            seq_len, nkv, hd, tp);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, ctx().position + chunk.offset, addr_offset("v").value,
            seq_len, nkv, hd, tp);

        TORCH_CHECK(
            q_ddr_slots_.count({seq_len_, (num_q_heads() / attn_tp()) * head_dim()}) == 1,
            "CausalDecoderModel: Q staging slot not allocated in KV_FIRST mode");
        const at::Tensor& q_slot = q_ddr_slot();
        int64_t q_local_elems = seq_len * local_q_heads * hd;
        c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
        int64_t q_row_stride = local_q_heads * hd;
        int64_t q_elem_offset = chunk.offset * q_row_stride;
        int64_t q_core_stride_bytes = q_slot.size(1) * q_row_stride * DWIDTH;
        rpu_launch_spm_scatter_ddr_dma(
            addr(0, "q_kv"), q_ddr_base + q_elem_offset,
            q_local_elems, q_core_stride_bytes,
            /*num_cores=*/tp);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // build_layer_subgraph (8 phases + final_norm fuse)
    //
    // Body verbatim from v2 per-layer compute, with two mechanical edits:
    //   - SDPA + KV-insert call sites use typed SPM offsets via addr_offset(name).value
    //     (Pitfall 3 structural fix — compile-time distinct from addr() absolutes).
    //   - Protected model-param members become pimpl getters (num_layers() etc.).
    // ═══════════════════════════════════════════════════════════════════════

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        int64_t seq_len = chunk.len;
        // RoPE table index. The default (rope_position_base_ < 0) and the
        // M-RoPE keepalive both index from ctx().position; the 1D override
        // decouples the RoPE position base from the cache-insert offset (KV-insert
        // below stays on ctx().position). chunk.offset advances within the new
        // contiguous block.
        const int64_t rope_base =
            (!has_mrope_ && rope_position_base_ >= 0) ? rope_position_base_ : ctx().position;
        int64_t cos_sin_start = rope_base + chunk.offset;
        int64_t L = num_layers();
        bool is_last_layer = (layer_idx == L - 1);
        // A one-row tail of a multi-token prefill is still prefill.  Decode-only
        // work must follow the whole request length, not this chunk's length.
        const bool decode_forward = ctx().seq_len == 1;
        int64_t h = hidden_size();
        int64_t nq = num_q_heads();
        int64_t nkv = num_kv_heads();
        int64_t hd = head_dim();
        // Attention tensor-parallel factor: min(NUM_CORES, nkv). NUM_CORES for
        // nkv>=8 (Qwen3/Llama no-op), 2 for Wall-OSS nkv=2. MLP stays NUM_CORES.
        int tp = attn_tp();
        const uint32_t input_residual = addr(0, "residual1");

        if (is_last_layer && fuse_lm_head_ && decode_forward) {
            TORCH_CHECK(lm_head_logits_out_.defined()
                            && lm_head_logits_dst_base_ != 0,
                        "CausalDecoderModel: fused decode lm_head output was "
                        "not allocated for the whole forward");
        }

        // ── Batch decode ─────────────────────────────────────────────────────
        // `bs` independent sequences share this layer. Their rows are packed
        // batch-major into the GEMM M dimension, so every ROW-PARALLEL op
        // (norm / q,k,v / o_proj / all-reduce / MLP / final norm / lm_head)
        // runs ONE launch over `rows` and reads each weight exactly once —
        // that amortization is the entire point of batching a decode step.
        //
        // Only three things are per-sequence, because each sequence owns KV
        // cache slot b: KV-insert, SDPA, and the fused-lm_head output DMA.
        // Those are emitted in a loop over b below.
        //
        // ⚠️ `torch.rpu.set_ddr_flush(True)` makes each KV insert flush its whole
        // cache tensor, so per-step work becomes O(total KV-cache bytes) and grows
        // with batch. The flag defaults to false (g_rpu_ddr_flush_enabled).
        //
        // Not a bug in this loop: flush_boundary_ptrs() sort+uniques the
        // pointers, so the bs identical flushes emitted below collapse to one.
        // The residual cost is inherent to flushing a B-times-larger cache.
        //
        // RoPE is neither: run_all_layers admits bs > 1 only at seq_len == 1,
        // so all `bs` rows share ONE position. The rope kernel indexes cos/sin
        // by ROW (cos[cos_sin_start + row]) and is flat across heads, so
        // presenting the [bs][heads][hd] SPM region as seq_len=1 with
        // bs*heads "heads" gives every row the same position in a single
        // launch — same memory, no extra kernels. See emit_qk_rope callers.
        const int64_t bs = ctx().batch_size;
        const int64_t rows = bs * seq_len;
        if (bs > 1) {
            TORCH_CHECK(seq_len == 1,
                        "CausalDecoderModel: batch decode requires chunk.len == 1, got ",
                        seq_len);
            TORCH_CHECK(ctx().is_causal && !ctx().attention_mask.has_value(),
                        "CausalDecoderModel: batch decode requires the causal "
                        "no-explicit-mask path");
        }

        if (!ctx().is_causal && !ctx().attention_mask.has_value()) {
            // KV_FIRST Phase-2 needs local_q_heads here; the sequential path
            // declares its own copy later at the Phase-3 QK-norm site (this
            // branch always returns, so the two declarations never collide).
            int64_t local_q_heads = nq / tp;
            if (!ctx().input_in_spm) {
                emit_layer_input_dma(layer_idx, chunk);
            }

            int64_t q_local_elems = seq_len * local_q_heads * hd;
            const at::Tensor& q_slot = q_ddr_slot();
            c10::Half* q_ddr_base = q_slot.data_ptr<c10::Half>();
            int64_t q_row_stride = local_q_heads * hd;
            int64_t q_elem_offset = chunk.offset * q_row_stride;
            int64_t q_core_stride = q_slot.size(1) * q_row_stride * DWIDTH;
            rpu_launch_ddr_scatter_spm_dma(
                q_ddr_base + q_elem_offset,
                /*elements_per_core=*/q_local_elems,
                /*core_stride_bytes=*/q_core_stride,
                addr(0, "q"),
                /*num_cores=*/tp);

            int64_t total_kv_seq_len = ctx().position + seq_len_;
            auto& k_cache = (*ctx().k_caches)[layer_idx];
            auto& v_cache = (*ctx().v_caches)[layer_idx];
            rpu_launch_sdpa_spm_dispatch(
                sdpa_kernel_,
                k_cache, v_cache, 0 /*MASK_NONE*/, c10::nullopt,
                addr_offset("q").value,
                addr_offset("output").value,
                addr_offset("sdpa_tmp").value, 0,
                seq_len, nq, nkv, hd, total_kv_seq_len, tp, NUM_CORES);

            rpu_prepare_ring_all_reduce_input(
                addr(0, "oproj"), seq_len, h, tp);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "output"), lw.o_w, addr(0, "oproj"),
                seq_len, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
                /*bias_spm_addr=*/0, lw.o_ws,
                nvfp4_scale_addr("nvfp4_o_ts"), (uint16_t)layer_idx,
                /*force_acc32=*/linear_acc32_);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), input_residual,
                addr(0, "residual2"), seq_len, h, tp, NUM_CORES);
            emit_adarms_mut_refresh_post(layer_idx);  // replay-safe AdaRMS: refresh scale+shift SPM
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual2"), addr(0, "residual1"),
                layer_addr(layer_idx, 0, "post_norm_w"),   // adarms_: scale rides here
                seq_len, h, eps_);
            if (adarms_) {  // AdaRMS FiLM: residual1 (normed) += post_shift
                rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                    layer_addr(layer_idx, 0, "post_shift"),
                    addr(0, "residual1"), addr(0, "residual1"),
                    seq_len, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
            }
            emit_mlp_pipeline(
                lw.gate_w, lw.up_w, lw.down_w, seq_len,
                use_silu_ ? ActivationKind::SILU : ActivationKind::NONE,
                lw.gate_ws, lw.up_ws, lw.down_ws,
                nvfp4_scale_addr("nvfp4_gate_ts"),
                nvfp4_scale_addr("nvfp4_up_ts"),
                nvfp4_scale_addr("nvfp4_down_ts"),
                (uint16_t)layer_idx,
                /*acc32=*/linear_acc32_);

            if (!deepstack_lang_layers_.empty()) {
                auto it = std::find(deepstack_lang_layers_.begin(),
                                    deepstack_lang_layers_.end(),
                                    static_cast<int64_t>(layer_idx));
                if (it != deepstack_lang_layers_.end()) {
                    const int idx_in_list = static_cast<int>(
                        std::distance(deepstack_lang_layers_.begin(), it));
                    rpu_launch_ddr_broadcast_spm_dma_mutable(
                        &deepstack_dense_src_base_[idx_in_list],
                        /*src_offset_bytes=*/chunk.offset * h * DWIDTH,
                        /*num_elements=*/chunk.len * h,
                        addr(0, "deepstack_scratch"),
                        NUM_CORES);
                    rpu_launch_eltwise_binary_spm_kernel(
                        addr(0, "residual1"),
                        addr(0, "deepstack_scratch"),
                        addr(0, "residual1"),
                        chunk.len * h,
                        ValuOpType::ADD,
                        c10::Half(1.0f),
                        NUM_CORES);
                }
            }

            if (is_last_layer) {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "residual1"), addr(0, "residual1"),
                    addr(0, "final_norm_w"),
                    seq_len, h, eps_);
            }

            if (is_last_layer && fuse_lm_head_ && decode_forward) {
                // The shared fused-lm-head path uses the generated auto-tile
                // family for FP16 and W8A16, including decode at M=1.
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    addr(0, "residual1"), lm_head_w_, addr(0, "lm_head_out"),
                    /*M=*/1, /*N=*/vocab_size_, /*K=*/h,
                    /*partition=*/1, /*num_cores=*/NUM_CORES,
                    /*bias_spm_addr=*/0,
                    lm_head_w_scale_,
                    /*nvfp4_tensor_scale_spm_addr=*/0,
                    /*nvfp4_layer_id=*/0,
                    /*force_acc32=*/linear_acc32_);
                const int64_t local_n = vocab_size_ / NUM_CORES;
                rpu_launch_spm_scatter_ddr_dma_mutable(
                    addr(0, "lm_head_out"),
                    &lm_head_logits_dst_base_,
                    /*dst_offset_bytes=*/0,
                    /*elements_per_core=*/local_n,
                    /*core_stride_bytes=*/local_n * DWIDTH,
                    NUM_CORES);
            } else if (!ctx().output_to_spm) {
                emit_layer_output_dma(layer_idx, chunk);
            }
            return;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Input DMA (if input is not already in SPM) — framework-internal detail:
        //   layer 0 reads ctx().hidden_states (request tensor);
        //   layer > 0 reads the DDR chain buffer (framework owns ping-pong);
        //   ctx().input_in_spm == true skips DMA (intra-group SPM-resident).
        // ─────────────────────────────────────────────────────────────────────
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 1: Input RMSNorm
        //
        // norm_w is PersistentPerLayer with a .preload_callback that DMAs
        // layer_weights_[L].input_norm_w into SPM once per BUILD (re-fires on
        // weights_dirty_ / model_state_gen advance). Read the per-layer slot
        // via layer_addr(layer_idx, 0, "norm_w") — see declare_buffers.
        // ─────────────────────────────────────────────────────────────────────
        emit_adarms_mut_refresh_input(layer_idx);  // replay-safe AdaRMS: refresh scale+shift SPM
        rpu_launch_rmsnorm_spm_kernel(
            input_residual, addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "norm_w"),   // adarms_: scale rides here
            rows, h, eps_);
        if (adarms_) {  // AdaRMS FiLM: input_norm += input_shift (LingBot-VLA expert)
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                layer_addr(layer_idx, 0, "input_shift"),
                addr(0, "input_norm"), addr(0, "input_norm"),
                rows, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 2: QKV Linear
        // ─────────────────────────────────────────────────────────────────────
        // QKV col-partition over attn_tp() cores. Optional bias (Qwen2.5/Wall-OSS)
        // is fused for FP16/W8/W4. NVFP4 uses the same persistent SPM bias slots
        // but adds them separately to avoid the fused-bias replay hazard.
        // Batch decode: M = rows, so the q/k/v weights stream from DDR once for
        // all `bs` sequences instead of once per sequence.
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            rows, nq * hd, h, /*partition=*/1, /*num_cores=*/tp,
            qkv_fused_bias_addr(layer_idx, "q_bias"), lw.q_ws,
            nvfp4_scale_addr("nvfp4_q_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);
        emit_nvfp4_qkv_bias(
            layer_idx, "q_bias", "q", rows, nq * hd / tp);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            rows, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
            qkv_fused_bias_addr(layer_idx, "k_bias"), lw.k_ws,
            nvfp4_scale_addr("nvfp4_k_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);
        emit_nvfp4_qkv_bias(
            layer_idx, "k_bias", "k", rows, nkv * hd / tp);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            rows, nkv * hd, h, /*partition=*/1, /*num_cores=*/tp,
            qkv_fused_bias_addr(layer_idx, "v_bias"), lw.v_ws,
            nvfp4_scale_addr("nvfp4_v_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);
        emit_nvfp4_qkv_bias(
            layer_idx, "v_bias", "v", rows, nkv * hd / tp);

        // ─────────────────────────────────────────────────────────────────────
        // Phase 3: QK RMSNorm + RoPE
        // ─────────────────────────────────────────────────────────────────────
        int64_t local_q_heads = nq / tp;
        int64_t local_kv_dim  = nkv * hd / tp;
        // QK head normalization is an explicit model property set with weights.
        bool has_qk_head_norm = has_qk_norm_;

        // RoPE 表指针 / strobe masks / partial-mrope keepalive 现由 emit_qk_rope 内部从成员
        // 按 has_mrope_/partial_mrope_active_ 派生，此处不再展开。

        // Phase 06.1: op-boundary cos/sin parity dump.
        // Captures the EXACT cos/sin slice the rope kernel is about to consume,
        // plus rope_meta tuple proving the kernel was invoked with the expected
        // (ctx().position, chunk.offset, cos_sin_start, seq_len, layer_idx, chunk_idx).
        // Gated behind get_debug_export(); no default-path runtime impact.
        if (get_debug_export()) {
            // Partial-M-RoPE consumes the interleaved cos_il/sin_il keepalives;
            // the debug dump follows the same active-path selection as emit_qk_rope.
            const bool partial_dump = has_mrope_ && partial_mrope_active_;
            const at::Tensor& cos_src = partial_dump ? cos_il_keepalive_ : cos_;
            const at::Tensor& sin_src = partial_dump ? sin_il_keepalive_ : sin_;
            int64_t cos_dim = cos_src.size(-1);  // head_dim or head_dim/2 (kernel format)
            auto cos_slice = cos_src.narrow(0, cos_sin_start, seq_len).contiguous().clone();
            auto sin_slice = sin_src.narrow(0, cos_sin_start, seq_len).contiguous().clone();
            int64_t chunk_idx = chunk.idx;
            std::string cos_key = "L" + std::to_string(layer_idx) + "_cos_at_kernel_input_chunk" + std::to_string(chunk_idx);
            std::string sin_key = "L" + std::to_string(layer_idx) + "_sin_at_kernel_input_chunk" + std::to_string(chunk_idx);
            std::string meta_key = "L" + std::to_string(layer_idx) + "_rope_meta_chunk" + std::to_string(chunk_idx);
            g_debug_tensors[cos_key] = cos_slice;
            g_debug_tensors[sin_key] = sin_slice;
            auto meta = at::tensor({(double)ctx().position, (double)chunk.offset,
                                    (double)cos_sin_start, (double)seq_len,
                                    (double)layer_idx, (double)chunk_idx,
                                    (double)cos_dim, (double)(partial_dump ? 1 : 0)},
                                   at::TensorOptions().dtype(at::kDouble));
            g_debug_tensors[meta_key] = meta;
        }

        // RoPE 经 emit_qk_rope 单一真相源：has_mrope_ ×
        // partial_mrope_active_ → 1D-RoPE / M-RoPE / partial-M-RoPE 三变体,与
        // emit_kv_first_body 同枝(共享 cos/sin 表 + cos_sin_start = ctx().position +
        // chunk.offset 语义)。新增 RoPE 模式只改 helper 一处,两路自动同步。
        if (has_qk_head_norm) {
            // q_norm_w / k_norm_w are PersistentPerLayer (see declare_buffers).
            // The .preload_callback DMA'd them into SPM at BUILD; we just read.
            int64_t local_kv_heads = nkv / tp;

            // Q: q_buf → output_buf (norm) → q_buf (RoPE)
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q"), addr(0, "output"),
                layer_addr(layer_idx, 0, "q_norm_w"),
                rows * local_q_heads, hd, eps_);
            emit_qk_rope("output", "q", bs * local_q_heads, hd, seq_len, cos_sin_start, tp);

            // K: k_buf → input_norm_buf (norm) → k_buf (RoPE)
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "k"), addr(0, "input_norm"),
                layer_addr(layer_idx, 0, "k_norm_w"),
                rows * local_kv_heads, hd, eps_);
            emit_qk_rope("input_norm", "k", bs * local_kv_heads, hd, seq_len, cos_sin_start, tp);
        } else {
            // No QK head norm, in-place RoPE
            emit_qk_rope("q", "q", bs * local_q_heads, hd, seq_len, cos_sin_start, tp);
            emit_qk_rope("k", "k", bs, local_kv_dim, seq_len, cos_sin_start, tp);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 4: KV cache insert + SDPA + O_proj
        //
        // Pitfall 3 structural fix: SDPA and KV-insert need SPM OFFSETS
        // (not absolute addresses). Use addr_offset("name").value — the typed
        // SpmOffset can't be confused with the absolute addr() return value.
        // ─────────────────────────────────────────────────────────────────────
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];

        // Batch decode strides. SPM buffer sizes in declare_buffers are byte
        // counts, so addr_offset() values are byte offsets: sequence b's rows
        // start one per-sequence block in. The caches are [batch, ...]
        // contiguous, so slot b starts numel/batch elements in.
        const int64_t k_spm_stride = seq_len * local_kv_dim * DWIDTH;
        const int64_t q_spm_stride = seq_len * local_q_heads * hd * DWIDTH;
        // Slot base: batched decode owns slots 0..bs-1 (cache_batch_slot_ == 0);
        // batched prefill runs bs == 1 against slot cache_batch_slot_.
        const int64_t slot0 = cache_batch_slot_;
        const bool multi_slot = (bs > 1 || slot0 > 0);
        const int64_t k_cache_batch_stride =
            multi_slot ? k_cache.numel() / k_cache.size(0) : 0;
        const int64_t v_cache_batch_stride =
            multi_slot ? v_cache.numel() / v_cache.size(0) : 0;
        if (multi_slot) {
            TORCH_CHECK(k_cache.size(0) >= slot0 + bs && v_cache.size(0) >= slot0 + bs,
                        "CausalDecoderModel: batch decode needs KV caches with "
                        "batch dim >= ", slot0 + bs, ", got k=", k_cache.size(0),
                        " v=", v_cache.size(0));
            // SDPA takes ONE cache offset for both K and V. K and V are the same
            // size in RPUCache (sKeyVx == sValVx; the last two swizzle dims are
            // transposed but equal in product), so one stride is correct — assert
            // it rather than assume it.
            TORCH_CHECK(k_cache_batch_stride == v_cache_batch_stride,
                        "CausalDecoderModel: batch decode assumes equal K/V batch "
                        "strides, got k=", k_cache_batch_stride,
                        " v=", v_cache_batch_stride);
        }

        // Each sequence owns cache slot b, so KV-insert is per-sequence. All
        // `bs` sequences sit at the SAME position (equal-length batch), so the
        // insert position is identical across b — only the cache base moves.
        for (int64_t b = 0; b < bs; ++b) {
            rpu_launch_insert_kcache_spm_unified(
                k_cache, ctx().position + chunk.offset,
                addr_offset("k").value + (uint32_t)(b * k_spm_stride),
                seq_len, nkv, hd, tp, (slot0 + b) * k_cache_batch_stride);
            rpu_launch_insert_vcache_spm_unified(
                v_cache, ctx().position + chunk.offset,
                addr_offset("v").value + (uint32_t)(b * k_spm_stride),
                seq_len, nkv, hd, tp, (slot0 + b) * v_cache_batch_stride);
        }

        int64_t kv_seq_len = chunk.kv_seq_len;

        // Explicit 2D mask path vs LTM/NONE. When base reports an
        // explicit mask (is_causal=false + mask), use the MASK_2D type prepared
        // in dynamic_config and upload the mask to SPM right before SDPA. The
        // sdpa_mask Temp slot is phase-aliased, so it must be re-uploaded every
        // layer. Single
        // chunk is asserted in dynamic_config, so seq_q=seq_len and
        // seq_k=kv_seq_len (= P+H) are constant across the (only) chunk.
        int mask_type;
        uint32_t mask_off = 0;
        if (ctx().attention_mask.has_value()) {
            mask_type = prepared_attn_mask_.mask_type;          // MASK_2D (4)
            mask_off  = addr_offset("sdpa_mask").value;
            sdpa_dma_mask_to_spm(prepared_attn_mask_, mask_off,
                                 seq_len, kv_seq_len, tp);
        } else {
            bool sdpa_causal = ctx().is_causal && (seq_len > 1);
            mask_type = sdpa_causal ? 1 : 0;                    // MASK_LTM / MASK_NONE
        }

        // SDPA runs on attn_tp() cores while NUM_CORES preserves the full-core
        // KV-cache layout.
        if (ctx().attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            TORCH_CHECK(
                bs == 1 && ctx().position == 0 && chunk.offset == 0 &&
                    seq_len == kv_seq_len && mask_type == 1 &&
                    tp == NUM_CORES,
                "CausalDecoderModel raw-SPM by-MHA escaped its exact short "
                "Qwen3 prefill contract");
            rpu_launch_v_transpose_spm(
                addr(0, "v"), addr(0, "sdpa_tmp"),
                /*batch=*/1, kv_seq_len, nkv, hd, tp);
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q"), addr(0, "k"), addr(0, "sdpa_tmp"),
                addr(0, "output"), /*mask_spm=*/0, mask_type,
                1.0 / std::sqrt(static_cast<double>(hd)),
                /*batch=*/1, seq_len, kv_seq_len, nq, nkv, hd, tp);
        } else {
            // Batch decode: one launch per sequence, each reading only its own
            // cache slot. The launches safely reuse the one sdpa_tmp scratch.
            for (int64_t b = 0; b < bs; ++b) {
                rpu_launch_sdpa_spm_dispatch(
                    sdpa_kernel_,
                    k_cache, v_cache, mask_type, c10::nullopt,
                    addr_offset("q").value + (uint32_t)(b * q_spm_stride),
                    addr_offset("output").value + (uint32_t)(b * q_spm_stride),
                    addr_offset("sdpa_tmp").value, mask_off,
                    seq_len, nq, nkv, hd, kv_seq_len, tp, NUM_CORES,
                    (slot0 + b) * k_cache_batch_stride);
            }
        }

        // O-proj row-partition over attn_tp() cores; reduce back to NUM_CORES below.
        rpu_prepare_ring_all_reduce_input(
            addr(0, "oproj"), rows, h, tp);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "output"), lw.o_w, addr(0, "oproj"),
            rows, h, nq * hd, /*partition=*/0, /*num_cores=*/tp,
            /*bias_spm_addr=*/0, lw.o_ws,
            nvfp4_scale_addr("nvfp4_o_ts"), (uint16_t)layer_idx,
            /*force_acc32=*/linear_acc32_);

        // ─────────────────────────────────────────────────────────────────────
        // Phase 5: Attention reduce + residual (attn_tp() inputs -> NUM_CORES out)
        // ─────────────────────────────────────────────────────────────────────
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), input_residual,
            addr(0, "residual2"), rows, h, tp, NUM_CORES);

        // ─────────────────────────────────────────────────────────────────────
        // Phase 6: Post-attention RMSNorm (post_norm_w is PersistentPerLayer).
        // ─────────────────────────────────────────────────────────────────────
        emit_adarms_mut_refresh_post(layer_idx);  // replay-safe AdaRMS: refresh scale+shift SPM
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual2"), addr(0, "residual1"),
            layer_addr(layer_idx, 0, "post_norm_w"),   // adarms_: scale rides here
            rows, h, eps_);
        if (adarms_) {  // AdaRMS FiLM: residual1 (normed) += post_shift
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                layer_addr(layer_idx, 0, "post_shift"),
                addr(0, "residual1"), addr(0, "residual1"),
                rows, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Phase 7-8: MLP (gate, up, silu?, mul, down) + reduce + residual → residual1
        // ─────────────────────────────────────────────────────────────────────
        emit_mlp_pipeline(
            lw.gate_w, lw.up_w, lw.down_w, rows,
            use_silu_ ? ActivationKind::SILU : ActivationKind::NONE,
            lw.gate_ws, lw.up_ws, lw.down_ws,
            nvfp4_scale_addr("nvfp4_gate_ts"),
            nvfp4_scale_addr("nvfp4_up_ts"),
            nvfp4_scale_addr("nvfp4_down_ts"),
            (uint16_t)layer_idx,
            /*acc32=*/linear_acc32_);

        // ─────────────────────────────────────────────────────────────────────
        // Phase 8.5: DeepStack visual_embed inject
        //
        // When layer_idx ∈ deepstack_lang_layers_, add a dense visual_embed
        // tensor to residual1 (which currently holds the post-MLP hidden state).
        // The dense tensor is supplied per-forward by the Python adapter and
        // patched into the cached graph via mutable DMA (cursor-patch only the
        // src base; the [chunk.offset → spm_dst] mapping is BUILD-baked).
        //
        // For text-only / decode forwards the caller passes a zero buffer (view
        // into a persistent zero keepalive [MAX_KEEPALIVE_SEQ, hidden]) so the
        // add reduces to a no-op while keeping a single GraphCache slot
        // The buffer MUST be sized to MAX_KEEPALIVE_SEQ because src_offset_bytes
        // is baked at BUILD; chunk N > 0 reads past
        // the per-forward seq_len region).
        //
        // Injection must precede final_norm fuse: the mergers' outputs are
        // added at HF Qwen3VLTextModel BEFORE the model.norm call —
        // re-ordering corrupts the post-norm hidden state.
        // ─────────────────────────────────────────────────────────────────────
        if (!deepstack_lang_layers_.empty()) {
            auto it = std::find(deepstack_lang_layers_.begin(),
                                deepstack_lang_layers_.end(),
                                static_cast<int64_t>(layer_idx));
            if (it != deepstack_lang_layers_.end()) {
                const int idx_in_list = static_cast<int>(
                    std::distance(deepstack_lang_layers_.begin(), it));
                rpu_launch_ddr_broadcast_spm_dma_mutable(
                    &deepstack_dense_src_base_[idx_in_list],
                    /*src_offset_bytes=*/chunk_row_offset_(chunk) * h * DWIDTH,
                    /*num_elements=*/rows * h,
                    addr(0, "deepstack_scratch"),
                    NUM_CORES);
                rpu_launch_eltwise_binary_spm_kernel(
                    addr(0, "residual1"),
                    addr(0, "deepstack_scratch"),
                    addr(0, "residual1"),
                    rows * h,
                    ValuOpType::ADD,
                    c10::Half(1.0f),
                    NUM_CORES);
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // ★ final_norm fuse (only at the very last layer)
        //   HF Qwen3Model.forward calls self.norm(hidden_states) before returning.
        //   We fuse it into this kernel to avoid a DDR round-trip.
        //   In-place: residual1 → final_norm_w RMSNorm → residual1
        //
        //   final_norm 始终在此处 fuse (prefill 和 decode 都做).
        //   decode (seq_len=1) + fuse_lm_head_=true 时, 后面的 Output DMA 段会
        //   追加 lm_head GEMM 并跳过 spm2ddr, 实现全程 SPM 的 fused decode.
        // ─────────────────────────────────────────────────────────────────────
        if (is_last_layer) {
            // final_norm_w is Persistent (single-slot, see declare_buffers) —
            // .preload_callback DMA'd it at BUILD; addr(0, ...) reads the slot.
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual1"), addr(0, "residual1"),
                addr(0, "final_norm_w"), rows, h, eps_);
        }

        // ─────────────────────────────────────────────────────────────────────
        // Output DMA / decode fused lm_head
        // ─────────────────────────────────────────────────────────────────────
        if (is_last_layer && fuse_lm_head_ && decode_forward) {
            // Decode-only fused lm_head.
            //
            //   此时 residual1 已经是 normed hidden (final_norm fuse 输出).
            //   追加一个 col-partition GEMM 到当前 main graph batch:
            //     input  = residual1 (normed, M=1, K=hidden_size, 在 SPM)
            //     weight = lm_head_w_ (DDR, col-partition swizzled)
            //     output = lm_head_out (SPM, vocab_size/8 per core)
            //
            //   随后用 mutable scatter DMA 将每个 core 的 vocab shard 写到
            //   lm_head_logits_out_ 的连续 DDR 区间。这个 DMA 节点跟 lm_head
            //   GEMM 一起进入 GraphCache BUILD/REPLAY,避免 graph scope 内
            //   host memcpy 读未执行的 SPM。
            //
            //   lm-head 无条件走 generated auto-tile；W8A16 默认 ACC16，
            //   fp16 由 per-handle 策略或 RPU_LINEAR_ACC32=1 切到 ACC32。
            //
            //   SPM 峰值: lm_head_out = MAX_VOCAB/8 * 2B ≈ 50KB/core
            //   (Phase 9, 与 attention Phase 2-5 / MLP Phase 7-8 不重叠)
            //   Batch decode: M = rows, so the ~150k-row lm_head weight streams
            //   from DDR ONCE for all `bs` sequences — the single biggest weight
            //   read of a decode step.
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                addr(0, "residual1"),      // normed hidden, M=rows, in SPM
                lm_head_w_,                // DDR weight [vocab, hidden]
                addr(0, "lm_head_out"),    // SPM output per core, [rows][local_n]
                /*M=*/rows,
                /*N=*/vocab_size_,
                /*K=*/h,
                /*partition=*/1,           // col partition
                /*num_cores=*/NUM_CORES,
                /*bias_spm_addr=*/0,
                lm_head_w_scale_,
                /*nvfp4_tensor_scale_spm_addr=*/0,
                /*nvfp4_layer_id=*/0,
                /*force_acc32=*/linear_acc32_);
            const int64_t local_n = vocab_size_ / NUM_CORES;
            // The logits tensor is [bs, 1, vocab]; core c owns vocab slice
            // [c*local_n, (c+1)*local_n). SPM holds [rows][local_n] per core, so
            // one scatter per row moves that row's 8 shards into its vocab span.
            // bs == 1 requires one scatter call.
            for (int64_t b = 0; b < bs; ++b) {
                rpu_launch_spm_scatter_ddr_dma_mutable(
                    addr(0, "lm_head_out") + (uint32_t)(b * local_n * DWIDTH),
                    &lm_head_logits_dst_base_,
                    /*dst_offset_bytes=*/b * vocab_size_ * DWIDTH,
                    /*elements_per_core=*/local_n,
                    /*core_stride_bytes=*/local_n * DWIDTH,
                    NUM_CORES);
            }
        } else if (!ctx().output_to_spm) {
            // 正常路径: spm2ddr 写回 output_tensor_ (或 chain buffer)
            emit_layer_output_dma(layer_idx, chunk);
        }
    }

protected:  // LingBot2 sparse-MoE subclass reuses the decoder's layer state.
    uint32_t nvfp4_scale_addr(const char* name) const {
        return nvfp4_ ? addr(0, name) : 0;
    }

    uint32_t qkv_fused_bias_addr(int layer_idx, const char* name) const {
        return has_qkv_bias_ && !nvfp4_
            ? layer_addr(layer_idx, 0, name)
            : 0u;
    }

    void emit_nvfp4_qkv_bias(
        int layer_idx, const char* bias_name, const char* output_name,
        int64_t seq_len, int64_t local_n)
    {
        if (!nvfp4_ || !has_qkv_bias_) return;
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            layer_addr(layer_idx, 0, bias_name),
            addr(0, output_name),
            addr(0, output_name),
            seq_len, local_n, c10::Half(1.0f),
            ValuOpType::ADD, /*is_bopa=*/false);
    }

    SdpaConfig make_sdpa_config(int mask = 1) const {
        // attn_tp() == NUM_CORES for nkv>=8 (no-op); attn_tp() for Wall-OSS nkv=2.
        // Drives sdpa_tmp sizing + chunk-validity consistently with the SDPA call.
        return {sdpa_kernel_, head_dim(), num_q_heads(), num_kv_heads(), attn_tp(), mask};
    }

    // ── Model state (stored in C++, set via set_weights) ──
    std::vector<LayerWeights> layer_weights_;
    at::Tensor cos_, sin_;
    at::Tensor final_norm_w_;
    double eps_ = 1e-6;
    bool use_silu_ = true;
    bool nvfp4_ = false;
    bool qwen3_spm_kv_by_mha_enabled_ =
        qwen3_spm_kv_by_mha_enabled();
    at::Tensor q_nvfp4_tensor_scales_, k_nvfp4_tensor_scales_;
    at::Tensor v_nvfp4_tensor_scales_, o_nvfp4_tensor_scales_;
    at::Tensor gate_nvfp4_tensor_scales_, up_nvfp4_tensor_scales_;
    at::Tensor down_nvfp4_tensor_scales_;

    // ── AdaRMS FiLM (LingBot-VLA action expert) ──
    // When adarms_ (set via set_adarms_step), the per-layer input/post RMSNorm
    // becomes FiLM: rmsnorm(x)·scale + shift. The SCALE rides on the existing
    // input_norm_w / post_norm_w slots (caller passes folded scale=(1+γ)·weight
    // as those); this adds only the per-layer SHIFT + a broadcast shift-add after
    // each norm. Default off ⇒ plain RMS, byte-identical for all existing decoders.
    bool adarms_ = false;
    std::vector<at::Tensor> input_shift_, post_shift_;  // per-layer [hidden] fp16 RPU
    // Replay-safe AdaRMS (set_adarms_step_mutable): stable-addr [num_layers, hidden]
    // keepalives. Contents refreshed per Euler step; the per-layer scale/shift SPM slots
    // are re-DMA'd from these on every REPLAY via emit_adarms_mut_refresh (build-body op),
    // so the expert graph is built once and replayed for steps 1..N. Default off ⇒ the
    // plain build-per-step path above is unchanged (byte-identical for every other decoder).
    bool adarms_mutable_ = false;
    // Cold per native decoder handle: this flag also chooses the Persistent
    // per-layer allocation direction, so the emit path must never re-read a
    // potentially changed process environment.
    bool adarms_fused_bcast_enabled_ = read_adarms_fused_bcast_enabled();
    at::Tensor input_scale_ka_, post_scale_ka_, input_shift_ka_, post_shift_ka_;
    // In-graph denoise-unroll AdaRMS. When
    // adarms_unroll_, the per-body refresh DMAs read this body_iter's (scale, shift)
    // slice from stable [num_steps, num_layers, hidden] buffers at offset
    // (body_iter*num_layers + L)*hidden — so ONE graph unrolls all num_steps Euler
    // steps with per-step FiLM. Requires adarms_mutable_ (declare_buffers + FiLM add).
    // Default off ⇒ emit_adarms_mut_refresh_* keeps the mutable-keepalive path
    // byte-identical. Set via set_adarms_unroll (contents are call-invariant).
    bool adarms_unroll_ = false;
    at::Tensor adarms_is_, adarms_ish_, adarms_ps_, adarms_psh_;  // [num_steps, num_layers, hidden] fp16 RPU
    // Host-bounce-free schedule selection (LingBot2 controlled path). The four schedule
    // tensors above own immutable [num_steps,num_layers,hidden] storage. Captured mutable
    // DMAs retain pointers to the live-base MEMBERS below; select_adarms_schedule_step only
    // advances their values. DDR->SPM refresh remains unchanged.
    bool adarms_schedule_select_ = false;
    int64_t adarms_schedule_step_ = -1;
    uint64_t adarms_is_schedule_base_ = 0, adarms_ish_schedule_base_ = 0;
    uint64_t adarms_ps_schedule_base_ = 0, adarms_psh_schedule_base_ = 0;
    uint64_t adarms_is_live_base_ = 0, adarms_ish_live_base_ = 0;
    uint64_t adarms_ps_live_base_ = 0, adarms_psh_live_base_ = 0;

    // ── Phase 2.5: fused lm_head 状态 (per-instance, set via set_lm_head) ──
    at::Tensor lm_head_w_;        // [vocab_size, hidden_size] fp16/int8, RPU, col-partition swizzled
    at::Tensor lm_head_w_scale_;  // [vocab_size] fp16 scale for int8 lm_head_w_
    bool fuse_lm_head_ = false;   // decode (seq_len=1) 时追加 lm_head GEMM 到 main graph batch
    int64_t vocab_size_ = 0;      // lm_head_w_.size(0)
    at::Tensor lm_head_logits_out_;   // [1, 1, vocab_size] graph-written decode logits
    uint64_t lm_head_logits_dst_base_ = 0;  // mutable DMA destination base
    // Adapter-owned per-handle auto-planner ceiling. The process-global
    // override still wins when nonzero and remains an exact chunk request.
    int64_t configured_chunk_size_cap_ = 0;
    // Wall TP8 prefill policy: one chunk while seq<=cap, otherwise exactly two
    // equal chunks. Build-time immutable; Python pads the execution length to 32.
    bool equal_two_prefill_ = false;
    bool linear_acc32_ = false;

    // ── 1D-RoPE position-base override (cache-insert ↔ RoPE decouple) ──
    //
    // Some multimodal callers use a RoPE position base distinct from the
    // cache-insert offset. The generic 1D path couples both to `ctx().position`
    // (cos_sin_start = ctx().position + chunk.offset), which is correct only
    // when the two coincide. Set per-forward from the
    // cos_sin_offset arg; -1 = legacy "use ctx().position". Honored ONLY on the
    // 1D-RoPE path (!has_mrope_): the M-RoPE keepalive is filled at row
    // `ctx().position` (forward():545), so its cos_sin_start must stay coupled.
    int64_t rope_position_base_ = -1;

    // ── Qwen3-VL M-RoPE state ──
    bool has_mrope_ = false;
    std::vector<int32_t> mrope_section_;                            // [T, H, W] half-dim split
    std::array<std::array<uint16_t, 16>, 3> mrope_strobe_masks_{};  // T/H/W × WARP_SIZE strobe
    at::Tensor position_ids_keepalive_;                             // [MAX_KEEPALIVE_SEQ, 3] int32 RPU DDR

    // ── partial_mrope path: host-baked interleaved cos/sin ──
    // Stable-address keepalives are refreshed in the forward prologue before
    // Graph replay. partial_mrope_active_ selects them for the current forward.
    bool partial_mrope_active_ = false;
    at::Tensor cos_il_keepalive_;                                   // [MAX_KEEPALIVE_SEQ, hd/2] fp16 RPU DDR
    at::Tensor sin_il_keepalive_;                                   // [MAX_KEEPALIVE_SEQ, hd/2] fp16 RPU DDR
    // Skip keepalive copy/flush for the same source pointer/version and geometry.
    // Version-disabled inference tensors use pointer identity. set_weights
    // invalidates the cache after keepalive allocation or reset.
    int64_t ka_last_position_ = -1;
    int64_t ka_last_seq_ = -1;
    const void* ka_last_pos_src_ = nullptr;
    int64_t ka_last_pos_src_version_ = -1;
    const void* ka_last_cos_src_ = nullptr;
    int64_t ka_last_cos_src_version_ = -1;
    const void* ka_last_sin_src_ = nullptr;
    int64_t ka_last_sin_src_version_ = -1;

    // ── Qwen3-VL DeepStack state ──
    //
    // deepstack_lang_layers_   — text-decoder layers (in order) that receive
    //                            a visual_embed eltwise add (Qwen3-VL 2B/4B:
    //                            [0, 1, 2]).
    // deepstack_dense_src_base_ — live DDR addresses cursor-patched by mutable
    //                             DMA at each REPLAY. Updated by forward() from
    //                             the caller-supplied dense visual_embeds[i].
    // deepstack_dense_refs_    — keeps the caller's dense visual_embeds[i]
    //                            tensors alive across the synchronous forward
    //                            (mirrors AdaRMS cond_ref_ and keeps DMA inputs
    //                            alive for the synchronous captured forward).
    std::vector<int64_t> deepstack_lang_layers_;
    std::array<uint64_t, 3> deepstack_dense_src_base_{};
    std::vector<at::Tensor> deepstack_dense_refs_;

    // ── Wall-OSS explicit 2D mask state ──
    //
    // Set once per forward in dynamic_config(plan) when base reports an explicit
    // mask (ctx().attention_mask.has_value(), i.e. is_causal=false + mask given).
    // Cached across the per-layer build_layer_subgraph calls of a single forward;
    // ignored on no-mask forwards (ctx().attention_mask empty → LTM/NONE path).
    // mask_type=MASK_2D(4) for the action-denoise rectangular [H, P+H] mask.
    PreparedMask prepared_attn_mask_;
    // sdpa_prepare_mask cache (mirrors vision #23): the explicit 2D mask is
    // host-memoized → stable ptr across frames for a fixed prefix, but
    // sdpa_prepare_mask re-uploads it [seq_q,seq_k] CPU→RPU + stable-slot copy
    // every forward. Skip when (ptr, seq_q, seq_k) are unchanged.
    const void* last_attn_mask_ptr_ = nullptr;
    int64_t last_attn_seq_q_ = -1;
    int64_t last_attn_seq_k_ = -1;

    // ── KV_FIRST bidirectional no-mask state ──
    // Q staging keyed by {seq_len, q_ddr_local_dim} (= (num_q_heads()/attn_tp())*
    // head_dim()) — stable per-shape slots, NEVER reallocated, so an already-captured
    // graph's FIXED scatter DMA address stays valid across REPLAY even after another
    // shape's forward. Mirrors SigLIP q_ddr_slots_ / temp_ddr_slots_. Bounded by the handle's
    // distinct shape set (a handful in practice); a generous cap (alloc site) catches
    // pathological runaway loudly — per-shape slots can't be freed without breaking
    // the replay-address stability they exist to provide.
    std::map<std::pair<int64_t, int64_t>, at::Tensor> q_ddr_slots_;
    at::Tensor& q_ddr_slot() {
        return q_ddr_slots_.at({seq_len_, (num_q_heads() / attn_tp()) * head_dim()});
    }
    int64_t seq_len_ = 0;

    // ── Kernel selection (matches existing Qwen3 v2) ──
    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;
};

}  // namespace v3
