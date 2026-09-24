// rpu_dinov3_vision_model.cpp — DINOv3 ViT Vision Encoder (P4.1 + P4.2)
//
// 12-block (B/16) / 24-block (L/16) ViT encoder. Architecture vs Qwen3-VL:
//   - Q/K/V are separate Linears (no fused QKV) with asymmetric bias
//     (q/v/o have bias; k bias=None per HF config.key_bias=False).
//   - Explicit CLS + 4 register tokens prepended at the sequence start;
//     2D RoPE applies to patch tokens ONLY — the Q/K SPM addresses fed to
//     `rpu_launch_rope_2d_ddr_kernel` skip the first 5 special tokens.
//   - LayerScale γ_attn / γ_mlp element-wise mul after attn-out and mlp-out
//     remains a runtime multiply: folding small scales into FP16 weights can
//     underflow and change the result.
//   - Final LayerNorm + CLS slice as pooler (no merger, no DeepStack).
//
// Patch embed (Conv2d k=16 s=16) runs OUTSIDE the fused graph on CPU fp16
// (qwen3_vl F31 pattern). Python adapter prepends CLS + 4 register tokens
// and passes a `[1, num_tokens=1+R+num_patches, hidden]` fp16 RPU tensor to
// `forward()`. ViT-B/16 224×224 → num_tokens = 5 + 196 = 201.
//
// =============================================================================
// P4.1: create() / destroy() (skeleton + handle wiring).
// P4.2: set_weights() / set_rope_tables() / position_idx_keepalive() — weight
//       storage + RoPE table registration + per-forward int16 keepalive.
// P4.3: declare_buffers / emit_preload_weights / build_layer_subgraph
//       (real fused kernel sequence + SPM layout).
// P4.4: Python forward dispatch + numeric verify vs CPU fp32.
// =============================================================================

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"
#include "rpu_spm_allocator.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <string>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

// Position-table keepalive cap. ViT-B/16 224×224 → 196 patches; ViT-L/16
// same. 4096 covers up to 64×64 grid (1024×1024 image) — far beyond the
// 224×224 first-port profile.
constexpr int64_t DINOV3_VISION_MAX_KEEPALIVE_SEQ = 4096;

// Number of leading special tokens (CLS + register tokens) at the start of
// every DINOv3 sequence. Layer enforces num_register_tokens == 4 in the
// adapter (DINOv3Adapter._check_profile); the per-block forward keeps these
// 5 tokens out of the 2D RoPE pass and propagates them through SDPA + LN
// unchanged. If a future profile relaxes the register-token count, this
// constant becomes a config-driven int64_t.
constexpr int64_t DINOV3_VISION_NUM_SPECIAL_TOKENS = 5;

// Stable physical-route identities. The remaining launchers implement fixed
// DINO block math, debug export, or unconditional BufferDecl dataflow.
constexpr int64_t DINO_LN1_SITE = 8453738534831930450LL;
constexpr int64_t DINO_LN2_SITE = 4849048609893578846LL;
constexpr int64_t DINO_FINAL_NORM_SITE = 5760176277303677138LL;
constexpr int64_t DINO_LINEAR_ACC32_SITE = 8178669907035785042LL;
constexpr int64_t DINO_LINEAR_ACC16_SITE = 3984475134489774403LL;
constexpr int64_t DINO_Q_ROPE_DDR_SITE = 1673103556271315851LL;
constexpr int64_t DINO_K_ROPE_DDR_SITE = 2575381403489921854LL;
constexpr int64_t DINO_KV_INSERT_SITE = 2479886838587274388LL;
constexpr int64_t DINO_ATTENTION_32B_SITE = 6446246286887995950LL;
constexpr int64_t DINO_ATTENTION_16B_SITE = 818476026266528054LL;
constexpr int64_t DINO_ATTENTION_RAW_SPM_SITE = 5876524946555003440LL;
constexpr int64_t DINO_HYBRID_ATTENTION_SITE = 4579214973313800409LL;
constexpr int64_t DINO_HYBRID_SDPA_DMA_SITE = 1441805410418777326LL;
constexpr int64_t DINO_ATTN_ALL_REDUCE_SITE = 7147251202314566361LL;
constexpr int64_t DINO_MLP_ALL_REDUCE_SITE = 8822678331829260402LL;
constexpr int64_t DINO_OUTPUT_MLP_ALL_REDUCE_SITE = 2783645739028647889LL;
constexpr int64_t DINO_FINAL_NORM_GAMMA_PRELOAD_SITE = 8288718030370119634LL;
constexpr int64_t DINO_FINAL_NORM_BETA_PRELOAD_SITE = 3186088323150764618LL;
constexpr int64_t DINO_OUTPUT_MLP_BIAS1_PRELOAD_SITE = 4266926617532228252LL;
constexpr int64_t DINO_OUTPUT_MLP_BIAS2_PRELOAD_SITE = 4739230270905830658LL;
constexpr int64_t DINO_ATTENTION_FLAG_USE_16B = 1LL << 0;
constexpr int64_t DINO_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED = 1LL << 1;
constexpr int64_t DINO_HYBRID_FLAG_CPU_SDPA_DDR_REQUIRED = 1LL << 0;

constexpr uint32_t DINO_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
    KV_INSERT_CAP_HYBRID2 | KV_INSERT_CAP_HYBRID3;
// KV-insert remains a durable DDR mirror even when attention consumes the
// same freshly-produced K/V from SPM.  This flag is not an attention-residency
// decision and must not make the raw-SPM route look DDR-ineligible.
constexpr int64_t DINO_KV_FLAG_DDR_MIRROR = 1;

enum class DinoVisionRopeRoute : int64_t {
    ROPE_2D_DDR = 2,
};

enum class DinoVisionMutableDmaRoute : int64_t {
    HYBRID_SDPA_DDR_SCATTER_TO_SPM = 1,
    PRELOAD_DDR_BROADCAST_TO_SPM_ALL_CORES = 2,
    PRELOAD_DDR_SCATTER_TO_SPM = 3,
    PRELOAD_DDR_BROADCAST_TO_SPM_CORE0 = 4,
};

namespace v3 {

// DINOV3_VISION_FIXED_KERNEL_BASIS: remaining launches are fixed math/transport kernels.

class DINOv3VisionModel : public FusedModelBase {
public:
    struct LayerWeights {
        // [out, in] fp16 DDR, swizzled per partition (Python adapter handles):
        //   q/k/v/up — col-partition, o/down — row-partition.
        at::Tensor q_w, k_w, v_w, o_w;
        at::Tensor up_w, down_w;
    };

    struct LayerBiasNorm {
        // LayerNorm γ/β for the two pre-block norms.
        at::Tensor ln1_w, ln1_b, ln2_w, ln2_b;
        // Attention biases. DINOv3 has q_bias=True, k_bias=False (==None),
        // v_bias=True, o_bias=True. k_b is intentionally absent from the
        // setter and from this struct — emit_preload_weights memsets the
        // SPM k_bias slot to zero in P4.3.
        at::Tensor q_b, v_b, o_b;
        // MLP biases (DINOv3 mlp_bias=True by default).
        at::Tensor up_b, down_b;
        // LayerScale γ vectors (P1.3: keep-as-runtime-mul, do not fold).
        at::Tensor gamma_attn, gamma_mlp;
    };

    DINOv3VisionModel() = default;

    int64_t resolve_vision_chunk_size(int64_t num_tokens) {
        return resolve_chunk_size_for_shape(
            num_tokens, /*position=*/0, std::nullopt, /*is_causal=*/false);
    }

    std::vector<int64_t> resolve_stage_domain(int64_t num_tokens) {
        TORCH_CHECK(num_layers() > 0 && has_rope_,
                    "RPU_PLANNER_REJECT:CAPABILITY: DINO Vision weights/RoPE "
                    "are not initialized");
        TORCH_CHECK(num_tokens > DINOV3_VISION_NUM_SPECIAL_TOKENS &&
                        num_tokens <= DINOV3_VISION_MAX_KEEPALIVE_SEQ +
                            DINOV3_VISION_NUM_SPECIAL_TOKENS,
                    "RPU_PLANNER_REJECT:CAPABILITY: invalid DINO Vision "
                    "token count ", num_tokens);
        const std::vector<ChunkInfo> input_chunks{
            {0, 0, num_tokens, num_tokens}};
        const std::vector<FmbExecutionSpan> spans{
            {0, num_tokens, 0}};
        const FmbStageBoundaryPolicies boundary_policies{};
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                num_tokens, /*position=*/0,
                /*attention_mask=*/std::nullopt, /*is_causal=*/false,
                input_chunks, spans, boundary_policies,
                get_chunk_size_override(), /*logical_len=*/num_tokens));
    }

    // ========================================================================
    // set_weights — per-layer weights, biases, LN, γ vectors.
    //
    // Inputs: 17 TensorLists, all sized exactly N (= num_layers); five scalar
    // dim params. K bias is intentionally NOT a list — HF DINOv3
    // (config.key_bias = False) constructs `k_proj` without bias. Python
    // adapter MUST swizzle all 6 weight tensors before calling.
    //
    // Per-layer preconditions:
    //   - All tensors fp16 on PrivateUse1 (RPU), contiguous.
    //   - q_w/k_w/v_w/o_w shape [hidden, hidden].
    //   - up_w shape [intermediate, hidden]; down_w [hidden, intermediate].
    //   - ln1_w/ln1_b/ln2_w/ln2_b/q_b/v_b/o_b/up_b/down_b/gamma_attn/gamma_mlp
    //     all 1D `[hidden]` (or up_b `[intermediate]`).
    // ========================================================================
    void set_weights(
        at::TensorList q_w_list,  at::TensorList k_w_list,
        at::TensorList v_w_list,  at::TensorList o_w_list,
        at::TensorList up_w_list, at::TensorList down_w_list,
        at::TensorList ln1_w_list, at::TensorList ln1_b_list,
        at::TensorList ln2_w_list, at::TensorList ln2_b_list,
        at::TensorList q_b_list,  at::TensorList v_b_list, at::TensorList o_b_list,
        at::TensorList up_b_list, at::TensorList down_b_list,
        at::TensorList gamma_attn_list, at::TensorList gamma_mlp_list,
        int64_t num_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size,
        double eps)
    {
        int64_t N = static_cast<int64_t>(q_w_list.size());
        TORCH_CHECK(N > 0, "dinov3_vision_set_weights: empty weight lists");
        TORCH_CHECK(num_heads > 0 && head_dim > 0
                    && hidden_size > 0 && intermediate_size > 0,
                    "dinov3_vision_set_weights: dim params must be positive");

        auto check_list_size = [&](const at::TensorList& l, const char* n) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "dinov3_vision_set_weights: ", n,
                        ".size()=", l.size(), " != num_layers=", N);
        };
        check_list_size(k_w_list,    "k_w_list");
        check_list_size(v_w_list,    "v_w_list");
        check_list_size(o_w_list,    "o_w_list");
        check_list_size(up_w_list,   "up_w_list");
        check_list_size(down_w_list, "down_w_list");
        check_list_size(ln1_w_list,  "ln1_w_list");
        check_list_size(ln1_b_list,  "ln1_b_list");
        check_list_size(ln2_w_list,  "ln2_w_list");
        check_list_size(ln2_b_list,  "ln2_b_list");
        check_list_size(q_b_list,    "q_b_list");
        check_list_size(v_b_list,    "v_b_list");
        check_list_size(o_b_list,    "o_b_list");
        check_list_size(up_b_list,   "up_b_list");
        check_list_size(down_b_list, "down_b_list");
        check_list_size(gamma_attn_list, "gamma_attn_list");
        check_list_size(gamma_mlp_list,  "gamma_mlp_list");

        auto check_rank = [&](const at::TensorList& list, const char* name, int64_t r) {
            for (int64_t i = 0; i < N; i++) {
                TORCH_CHECK(list[i].defined(),
                            "dinov3_vision_set_weights: ", name, "[", i, "] undefined");
                TORCH_CHECK(list[i].dim() == r,
                            "dinov3_vision_set_weights: ", name, "[", i, "] must be ",
                            r, "D, got ", list[i].dim(), "D");
                TORCH_CHECK(list[i].scalar_type() == at::kHalf,
                            "dinov3_vision_set_weights: ", name, "[", i, "] must be fp16");
                TORCH_CHECK(list[i].device().type() == at::kPrivateUse1,
                            "dinov3_vision_set_weights: ", name, "[", i, "] must be on RPU device");
                TORCH_CHECK(list[i].is_contiguous(),
                            "dinov3_vision_set_weights: ", name, "[", i, "] must be contiguous");
            }
        };
        check_rank(q_w_list,    "q_w_list",    2);
        check_rank(k_w_list,    "k_w_list",    2);
        check_rank(v_w_list,    "v_w_list",    2);
        check_rank(o_w_list,    "o_w_list",    2);
        check_rank(up_w_list,   "up_w_list",   2);
        check_rank(down_w_list, "down_w_list", 2);
        check_rank(ln1_w_list,  "ln1_w_list",  1);
        check_rank(ln1_b_list,  "ln1_b_list",  1);
        check_rank(ln2_w_list,  "ln2_w_list",  1);
        check_rank(ln2_b_list,  "ln2_b_list",  1);
        check_rank(q_b_list,    "q_b_list",    1);
        check_rank(v_b_list,    "v_b_list",    1);
        check_rank(o_b_list,    "o_b_list",    1);
        check_rank(up_b_list,   "up_b_list",   1);
        check_rank(down_b_list, "down_b_list", 1);
        check_rank(gamma_attn_list, "gamma_attn_list", 1);
        check_rank(gamma_mlp_list,  "gamma_mlp_list",  1);

        // Per-core attention slice must be exact (no NUM_CORES-dropped heads).
        // ViT-L/16 (16 heads) maps cleanly; ViT-B/16 (12 heads) must be
        // pre-padded to 16 heads by the Python adapter (P4.4). Enforced here
        // to fail fast at set_weights, not later in the kernel sequence.
        TORCH_CHECK(num_heads % NUM_CORES == 0,
                    "dinov3_vision_set_weights: num_heads=", num_heads,
                    " must be divisible by NUM_CORES=", NUM_CORES,
                    " — ViT-B 12-head must be zero-padded to 16 before this call");
        TORCH_CHECK(intermediate_size % NUM_CORES == 0,
                    "dinov3_vision_set_weights: intermediate_size=", intermediate_size,
                    " must be divisible by NUM_CORES=", NUM_CORES);
        TORCH_CHECK(head_dim % 4 == 0,
                    "dinov3_vision_set_weights: head_dim=", head_dim,
                    " must be divisible by 4 (rope_2d_ddr kernel constraint)");

        // Vision encoder is MHA (no GQA): num_kv_heads == num_q_heads.
        set_model_params(num_heads, num_heads, head_dim,
                         hidden_size, intermediate_size);
        set_num_layers(N);
        eps_ = eps;
        // DINOv3 ViT-B/L head_dim=64, no SigLIP-style 72→80 padding.
        orig_head_dim_ = head_dim;

        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_weights_.push_back({
                q_w_list[i], k_w_list[i], v_w_list[i], o_w_list[i],
                up_w_list[i], down_w_list[i],
            });
        }

        layer_bias_norm_.clear();
        layer_bias_norm_.reserve(N);
        for (int64_t i = 0; i < N; i++) {
            layer_bias_norm_.push_back({
                ln1_w_list[i], ln1_b_list[i],
                ln2_w_list[i], ln2_b_list[i],
                q_b_list[i], v_b_list[i], o_b_list[i],
                up_b_list[i], down_b_list[i],
                gamma_attn_list[i], gamma_mlp_list[i],
            });
        }

        invalidate_model_state();
    }

    // ========================================================================
    // set_rope_tables — register FreqCos / FreqSin DDR static tables.
    //
    // Tables shape: `[max_hw, head_dim/4]` fp16 RPU. The rope_2d_ddr kernel
    // indexes the SAME table by both row_idx and col_idx (DINOv3 normalized
    // patch centers map symmetrically across H/W). 256B alignment required —
    // the Python adapter must pad alloc if natural size doesn't align.
    // Lazy-allocs `position_idx_keepalive_` once on first call.
    // ========================================================================
    void set_rope_tables(const at::Tensor& freq_cos, const at::Tensor& freq_sin)
    {
        TORCH_CHECK(freq_cos.defined() && freq_sin.defined(),
                    "dinov3_vision_set_rope: freq_cos / freq_sin must be defined");
        TORCH_CHECK(freq_cos.dim() == 2 && freq_sin.dim() == 2,
                    "dinov3_vision_set_rope: freq_cos/sin must be 2D, got ",
                    freq_cos.dim(), "/", freq_sin.dim(), "D");
        TORCH_CHECK(freq_cos.sizes() == freq_sin.sizes(),
                    "dinov3_vision_set_rope: freq_cos / freq_sin shape mismatch");
        TORCH_CHECK(freq_cos.scalar_type() == at::kHalf
                    && freq_sin.scalar_type() == at::kHalf,
                    "dinov3_vision_set_rope: freq_cos / freq_sin must be fp16");
        TORCH_CHECK(freq_cos.device().type() == at::kPrivateUse1
                    && freq_sin.device().type() == at::kPrivateUse1,
                    "dinov3_vision_set_rope: freq_cos / freq_sin must be on RPU device");
        TORCH_CHECK(freq_cos.is_contiguous() && freq_sin.is_contiguous(),
                    "dinov3_vision_set_rope: freq_cos / freq_sin must be contiguous");

        freq_cos_ = freq_cos;
        freq_sin_ = freq_sin;
        max_hw_ = freq_cos.size(0);

        // Lazy-alloc on first call; reuse afterwards (sizes never change).
        if (!position_idx_keepalive_.defined()) {
            position_idx_keepalive_ = at::empty(
                {DINOV3_VISION_MAX_KEEPALIVE_SEQ, 2},
                at::TensorOptions().dtype(at::kShort).device(at::kPrivateUse1));
            position_idx_keepalive_.zero_();
        }

        has_rope_ = true;
        invalidate_model_state();
    }

    int64_t max_keepalive_seq() const { return DINOV3_VISION_MAX_KEEPALIVE_SEQ; }

    // Expose the keepalive base for adapter-side per-forward `.copy_()`.
    at::Tensor& position_idx_keepalive() { return position_idx_keepalive_; }

    void set_final_norm(const at::Tensor& weight, const at::Tensor& bias) {
        TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 &&
                    weight.numel() == hidden_size() && bias.numel() == hidden_size(),
                    "dinov3 final norm tensors must be [hidden]");
        for (const auto* t : {&weight, &bias}) {
            TORCH_CHECK(t->scalar_type() == at::kHalf &&
                        t->device().type() == at::kPrivateUse1 && t->is_contiguous(),
                        "dinov3 final norm tensors must be contiguous fp16 RPU");
        }
        final_norm_w_ = weight;
        final_norm_b_ = bias;
        has_final_norm_ = true;
        invalidate_model_state();
    }

    void set_output_mlp(
        const at::Tensor& first_weight, const at::Tensor& first_bias,
        const at::Tensor& second_weight, const at::Tensor& second_bias)
    {
        const int64_t h = hidden_size();
        for (const auto* weight : {&first_weight, &second_weight}) {
            TORCH_CHECK(weight->dim() == 2 && weight->numel() == h * h
                        && weight->scalar_type() == at::kHalf
                        && weight->device().type() == at::kPrivateUse1
                        && weight->is_contiguous(),
                        "DINO output-MLP weights must be contiguous fp16 RPU [H,H]");
        }
        for (const auto* bias : {&first_bias, &second_bias}) {
            TORCH_CHECK(bias->dim() == 1 && bias->numel() == h
                        && bias->scalar_type() == at::kHalf
                        && bias->device().type() == at::kPrivateUse1
                        && bias->is_contiguous(),
                        "DINO output-MLP biases must be contiguous fp16 RPU [H]");
        }
        output_mlp_w1_ = first_weight;
        output_mlp_b1_ = first_bias;
        output_mlp_w2_ = second_weight;
        output_mlp_b2_ = second_bias;
        has_output_mlp_ = true;
        invalidate_model_state();
    }

    // ========================================================================
    // forward — drive the N-block encoder. Mirrors qwen3vl_vision_forward.
    //
    // Input  : [1, num_tokens, hidden_size] fp16 RPU. Python adapter must
    //          have already: (a) folded patch_embed (CPU fp16 Conv2d→Linear),
    //          (b) prepended CLS + register tokens, (c) written the per-patch
    //          (row, col) int16 pairs to slots [0..num_patches-1] of the
    //          keepalive returned by position_idx_keepalive().
    //          num_tokens == DINOV3_VISION_NUM_SPECIAL_TOKENS + num_patches.
    // Output : [1, num_tokens, hidden_size] fp16 RPU (final block hidden
    //          state, BEFORE the model's final LayerNorm and CLS slice —
    //          those happen in Python).
    // ========================================================================
    at::Tensor forward(
        const at::Tensor& input,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        int64_t num_tokens_in,
        at::IntArrayRef planned_stage_descriptor)
    {
        TORCH_CHECK(!planned_stage_descriptor.empty(),
                    "DINO Vision production forward requires its native "
                    "A6 stage descriptor");
        TORCH_CHECK(num_layers() > 0,
                    "DINOv3VisionModel::forward: set_weights must be called first");
        TORCH_CHECK(has_rope_,
                    "DINOv3VisionModel::forward: set_rope_tables must be called first");
        TORCH_CHECK(input.device().type() == at::kPrivateUse1,
                    "DINOv3VisionModel::forward: input must be on RPU device");
        TORCH_CHECK(input.is_contiguous(),
                    "DINOv3VisionModel::forward: input must be contiguous");
        TORCH_CHECK(input.dim() == 3,
                    "DINOv3VisionModel::forward: input must be 3D [1,N,H], got ",
                    input.dim(), "D");
        TORCH_CHECK(input.size(0) == 1,
                    "DINOv3VisionModel::forward: batch must be 1, got ", input.size(0));
        TORCH_CHECK(input.size(1) == num_tokens_in,
                    "DINOv3VisionModel::forward: input.size(1)=", input.size(1),
                    " must match num_tokens=", num_tokens_in);
        TORCH_CHECK(num_tokens_in > DINOV3_VISION_NUM_SPECIAL_TOKENS
                    && num_tokens_in <= DINOV3_VISION_MAX_KEEPALIVE_SEQ
                                        + DINOV3_VISION_NUM_SPECIAL_TOKENS,
                    "DINOv3VisionModel::forward: num_tokens=", num_tokens_in,
                    " must lie in (", DINOV3_VISION_NUM_SPECIAL_TOKENS, ", ",
                    DINOV3_VISION_MAX_KEEPALIVE_SEQ
                      + DINOV3_VISION_NUM_SPECIAL_TOKENS, "]");

        // Flush per-forward position_idx keepalive. Python adapter just wrote
        // [num_patches, 2] int16 (row, col) into the keepalive base; without
        // an explicit flush the CPU-side write may sit in the caching
        // allocator's write buffer while the rope_2d kernel's DDR DMA reads
        // stale (zero or prior-forward) data — silently collapsing 2D RoPE
        // to identity. Same pattern as qwen3vl_vision_model::forward.
        rpu_ddr_flush_force(position_idx_keepalive_.data_ptr<int16_t>());

        // Vision encoder is single-chunk, bidirectional (no causal mask).
        // The adapter may cold-pin the one-chunk 16-aligned capacity through
        // rpu_execution. dynamic_config rejects any value that would actually
        // split the logical token sequence.

        // [DEBUG] Per-layer hidden-state probe. Stable across REPLAY since the
        // data_ptr is baked into the cached graph's spm→ddr DMAs (gemma2 P4
        // pattern). Only allocated when set_debug_export(true). Toggling debug
        // export off→on at a later forward triggers a graph rebuild via
        // reset_graph_cache() (called from set_debug_export()).
        if (get_debug_export()) {
            int64_t N  = num_layers();
            int64_t bs = input.size(0);
            int64_t st = num_tokens_in;
            int64_t h  = hidden_size();
            if (!per_layer_debug_buf_.defined() ||
                per_layer_debug_buf_.size(0) != N ||
                per_layer_debug_buf_.size(1) != bs ||
                per_layer_debug_buf_.size(2) != st ||
                per_layer_debug_buf_.size(3) != h) {
                per_layer_debug_buf_ = at::empty(
                    {N, bs, st, h},
                    at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
            }
            // Phase-level layer-0 probes (broadcast-replicated SPM, read core 0).
            // Shape [1, num_tokens, hidden] each.
            auto alloc_phase = [&](at::Tensor& buf) {
                if (!buf.defined() ||
                    buf.size(0) != bs || buf.size(1) != st || buf.size(2) != h) {
                    buf = at::empty({bs, st, h},
                        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
                }
            };
            alloc_phase(post_ln1_debug_buf_);
            alloc_phase(post_oproj_ar_debug_buf_);
            alloc_phase(post_attn_block_debug_buf_);
            alloc_phase(post_ln2_debug_buf_);
            alloc_phase(post_down_ar_debug_buf_);
            // Layer-0 MLP-internal col/row-partition probes (per-core scatter).
            // up_proj output is col-partition [seq, local_inter]; down_proj
            // output is row-partition [seq, h] (full hidden, partial along K).
            int64_t local_inter = intermediate_size() / NUM_CORES;
            auto alloc_per_core = [&](at::Tensor& buf, int64_t per_core_dim) {
                if (!buf.defined() ||
                    buf.size(0) != NUM_CORES ||
                    buf.size(1) != st || buf.size(2) != per_core_dim) {
                    buf = at::empty({(int64_t)NUM_CORES, st, per_core_dim},
                        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
                }
            };
            alloc_per_core(post_up_proj_debug_buf_,    local_inter);
            alloc_per_core(post_down_partial_debug_buf_, h);
            // Col-partition per-layer probes (per-core SPM "q"/"k"/"v"/"sdpa_out"
            // gathered to DDR via spm_scatter_ddr_dma). Shape
            // [num_layers, NUM_CORES=8, seq_len, local_q_dim] — extended from
            // layer-0-only so hybrid-CPU-SDPA driver can dump each layer's
            // q/k/v after prior layers have been SDPA-corrected.
            int64_t local_q_dim = (num_q_heads() / NUM_CORES) * head_dim();
            auto alloc_qkv = [&](at::Tensor& buf) {
                if (!buf.defined() ||
                    buf.size(0) != N ||
                    buf.size(1) != NUM_CORES ||
                    buf.size(2) != st || buf.size(3) != local_q_dim) {
                    buf = at::empty({N, (int64_t)NUM_CORES, st, local_q_dim},
                        at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
                }
            };
            alloc_qkv(post_q_debug_buf_);
            alloc_qkv(post_k_debug_buf_);
            alloc_qkv(post_v_debug_buf_);
            alloc_qkv(post_sdpa_debug_buf_);
        }

        at::Tensor result = run_all_layers(
            input, k_caches, v_caches,
            /*mask=*/std::nullopt, /*position=*/0, /*is_causal=*/false,
            /*planned_chunk_size=*/0, planned_stage_descriptor);

        // [DEBUG] Publish per-layer outputs into g_debug_tensors for Python-side
        // get_debug_tensor("L{L}_layer_output") after the capture executes.
        if (get_debug_export() && per_layer_debug_buf_.defined()) {
            int64_t N = num_layers();
            // Capture has not executed the SPM->DDR probes yet. Publish views;
            // get_debug_tensor owns the post-capture CPU coherency boundary.
            for (int64_t L = 0; L < N; ++L) {
                std::string key = "L" + std::to_string(L) + "_layer_output";
                g_debug_tensors[key] = per_layer_debug_buf_[L];
            }
        }
        // Layer-0 phase-level probes.
        auto publish_phase = [&](const at::Tensor& buf, const char* key) {
            if (buf.defined()) {
                g_debug_tensors[key] = buf;
            }
        };
        if (get_debug_export()) {
            publish_phase(post_ln1_debug_buf_, "L0_post_ln1");
            publish_phase(post_oproj_ar_debug_buf_, "L0_post_oproj_ar");
            publish_phase(post_attn_block_debug_buf_, "L0_post_attn_block");
            publish_phase(post_ln2_debug_buf_, "L0_post_ln2");
            publish_phase(post_up_proj_debug_buf_, "L0_post_up_proj");
            publish_phase(post_down_partial_debug_buf_, "L0_post_down_partial");
            publish_phase(post_down_ar_debug_buf_, "L0_post_down_ar");
            // Per-layer Q/K/V/SDPA col-partition probes: split [N, 8, seq, local_q_dim]
            // into N separate `L{L}_post_q` etc keys so Python can consume per-layer.
            auto publish_per_layer_qkv = [&](const at::Tensor& buf, const char* basename) {
                if (!buf.defined()) return;
                int64_t N_local = buf.size(0);
                for (int64_t L = 0; L < N_local; ++L) {
                    std::string key = "L" + std::to_string(L) + "_post_" + basename;
                    g_debug_tensors[key] = buf[L];
                }
            };
            publish_per_layer_qkv(post_q_debug_buf_,    "q");
            publish_per_layer_qkv(post_k_debug_buf_,    "k");
            publish_per_layer_qkv(post_v_debug_buf_,    "v");
            publish_per_layer_qkv(post_sdpa_debug_buf_, "sdpa");
        }

        return result;
    }

protected:
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers = num_layers();
        cfg.preload_fn = static_cast<void(FusedModelBase::*)()>(
                             &DINOv3VisionModel::emit_preload_weights);
        // Single group — bidirectional vision encoder, same constraint as
        // SigLIP / qwen3_vl (multi-group REPLAY non-determinism mitigation).
        cfg.cross_layer_batch_size = num_layers();
        return cfg;
    }

    bool subclass_chunk_size_valid(
        int64_t chunk_size, int64_t seq_len,
        int64_t /*position*/) const override {
        // Filter the candidate domain before dynamic_config/manifest building;
        // invalid small candidates must not abort AUTO or a later exact match.
        return chunk_size >= seq_len;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        // This vision encoder is hard-wired single-chunk: build_layer_subgraph uses a
        // literal position=0 KV-insert and pos_offset=0 2D-RoPE with a CHUNK-LOCAL
        // seq_len, so a >1-chunk plan would have chunk n overwrite chunk 0's K/V rows
        // and silently degrade bidirectional attention to per-chunk self-attention.
        // Fail loudly instead. (Same guard as rpu_qwen25vl_vision_model.cpp:589; the
        // qwen3_vl / qwen3_5 towers instead thread chunk.offset through KV-insert.)
        TORCH_CHECK(plan.num_chunks == 1,
            "dinov3_vision requires a single chunk (got ", plan.num_chunks,
            "); multi-chunk would silently break cross-patch attention");
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;
        // RAW_SPM is descriptor-owned: legacy/no-descriptor entry points keep
        // the historical DDR route and cannot silently select a physical ABI.
        cfg.attention_policy = ctx().has_complete_physical_manifest()
            ? AttentionExecutionPolicy::AUTO
            : AttentionExecutionPolicy::DDR_KV;
        return cfg;
    }

    void consume_manifest_route(
        FmbRouteFamily family, int64_t site_id, int64_t selector,
        int64_t invocation = 0) {
        if (!ctx().has_complete_physical_manifest()) return;
        ctx().consume_physical_route(
            family, site_id, selector, /*resolved_flags=*/0,
            /*resolved_arguments=*/{}, invocation);
    }

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout, int64_t physical_len,
        int64_t logical_len, int64_t position) const override {
        TORCH_CHECK(plan.compute.chunks.size() == 1,
                    "DINO Vision COMPLETE descriptor requires one compute chunk");
        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = position + logical_len;
        manifest.kv_insert_physical_rows = physical_len;
        manifest.graph_lifecycle = FmbGraphLifecycle::RETAINED_CACHE;
        manifest.linear_accumulation = use_acc32_
            ? FmbLinearAccumulationPolicy::ACC32
            : FmbLinearAccumulationPolicy::ACC16;

        auto append = [&](FmbRouteFamily family, int64_t site_id,
                          int64_t selector,
                          std::vector<int64_t> arguments = {},
                          int64_t invocation = 0) {
            manifest.routes.push_back({
                site_id, family, selector, /*flags=*/0,
                std::move(arguments), invocation});
        };
        const ChunkInfo& chunk = plan.compute.chunks.front();
        LayoutContext spm_layout = layout;
        spm_layout.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const bool raw_spm_eligible = subclass_spm_kv_by_mha_eligible(
            plan, spm_layout, position);
        const bool raw_spm = layout.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        TORCH_CHECK(
            !raw_spm || raw_spm_eligible,
            "RPU_PLANNER_REJECT:CAPABILITY: DINO raw-SPM attention was "
            "selected outside the exact ViT-B full-K/V profile");
        append(
            FmbRouteFamily::LINEAR,
            use_acc32_ ? DINO_LINEAR_ACC32_SITE : DINO_LINEAR_ACC16_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            {use_acc32_ ? 1 : 0});
        const int64_t norm_route = standalone_compatibility_ ? 2 : 1;
        append(FmbRouteFamily::NORMALIZATION, DINO_LN1_SITE, norm_route);
        append(FmbRouteFamily::NORMALIZATION, DINO_LN2_SITE, norm_route);
        if (has_final_norm_) {
            append(FmbRouteFamily::NORMALIZATION, DINO_FINAL_NORM_SITE, norm_route);
        }
        if (!identity_rope_) {
            append(FmbRouteFamily::ROPE, DINO_Q_ROPE_DDR_SITE,
                   static_cast<int64_t>(DinoVisionRopeRoute::ROPE_2D_DDR),
                   {identity_rope_ ? 1 : 0,
                    chunk.len - DINOV3_VISION_NUM_SPECIAL_TOKENS,
                    num_q_heads() / NUM_CORES, head_dim(), head_dim(),
                    NUM_CORES});
            append(FmbRouteFamily::ROPE, DINO_K_ROPE_DDR_SITE,
                   static_cast<int64_t>(DinoVisionRopeRoute::ROPE_2D_DDR),
                   {identity_rope_ ? 1 : 0,
                    chunk.len - DINOV3_VISION_NUM_SPECIAL_TOKENS,
                    num_q_heads() / NUM_CORES, head_dim(), head_dim(),
                    NUM_CORES});
        }
        const KvInsertSegmentPlan kv_plan =
            resolve_kvinsert_plan_auto(
                DINO_KV_INSERT_SITE, manifest.graph_lifecycle,
                position + chunk.offset, chunk.len, chunk.len, NUM_CORES,
                num_q_heads(), head_dim(), DINO_KV_CAPABILITIES);
        const KvInsertRouteArguments kv_arguments =
            rpu_kvinsert_route_arguments(
                kv_plan, NUM_CORES, num_q_heads(), head_dim());
        manifest.routes.push_back({
            DINO_KV_INSERT_SITE, FmbRouteFamily::KV_INSERT,
            static_cast<int64_t>(kv_plan.route()),
            DINO_KV_FLAG_DDR_MIRROR,
            {kv_arguments.begin(), kv_arguments.end()}, chunk.idx});
        manifest.kv_insert_physical_rows = std::max(
            manifest.kv_insert_physical_rows, kv_plan.physical_rows());

        for (int64_t layer = 0; layer < num_layers(); ++layer) {
            const bool hybrid = layer < hybrid_layer_count_ &&
                hybrid_cpu_sdpa_out_.defined();
            const int64_t per_core_qkv_elems = chunk.len *
                (num_q_heads() / NUM_CORES) * head_dim();
            if (hybrid) {
                manifest.routes.push_back({
                    DINO_HYBRID_ATTENTION_SITE,
                    FmbRouteFamily::ATTENTION,
                    static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                    DINO_HYBRID_FLAG_CPU_SDPA_DDR_REQUIRED,
                    {hybrid_layer_count_, chunk.len, num_q_heads(),
                     head_dim(), NUM_CORES}, layer});
                manifest.routes.push_back({
                    DINO_HYBRID_SDPA_DMA_SITE,
                    FmbRouteFamily::MUTABLE_DMA,
                    static_cast<int64_t>(DinoVisionMutableDmaRoute::
                        HYBRID_SDPA_DDR_SCATTER_TO_SPM),
                    DINO_HYBRID_FLAG_CPU_SDPA_DDR_REQUIRED,
                    {hybrid_layer_count_, per_core_qkv_elems,
                     per_core_qkv_elems * DWIDTH, NUM_CORES}, layer});
            } else {
                const int64_t ddr_flags =
                    (use_16b_sdpa_ ? DINO_ATTENTION_FLAG_USE_16B : 0) |
                    (!raw_spm_eligible
                         ? DINO_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED : 0);
                manifest.routes.push_back({
                    raw_spm
                        ? DINO_ATTENTION_RAW_SPM_SITE
                        : (use_16b_sdpa_ ? DINO_ATTENTION_16B_SITE
                                        : DINO_ATTENTION_32B_SITE),
                    FmbRouteFamily::ATTENTION,
                    static_cast<int64_t>(
                        raw_spm
                            ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                            : AttentionExecutionPolicy::DDR_KV),
                    raw_spm ? 0 : ddr_flags,
                    {use_16b_sdpa_ ? 1 : 0, hybrid_layer_count_,
                     chunk.len, num_q_heads(), head_dim(), NUM_CORES}, layer});
            }
        }
        append(
            FmbRouteFamily::ALL_REDUCE, DINO_ATTN_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(chunk.len, hidden_size()));
        append(
            FmbRouteFamily::ALL_REDUCE, DINO_MLP_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(chunk.len, hidden_size()));
        if (has_output_mlp_) {
            append(
                FmbRouteFamily::ALL_REDUCE,
                DINO_OUTPUT_MLP_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(
                    chunk.len, hidden_size()));
        }
        if (has_final_norm_) {
            append(
                FmbRouteFamily::MUTABLE_DMA,
                DINO_FINAL_NORM_GAMMA_PRELOAD_SITE,
                static_cast<int64_t>(DinoVisionMutableDmaRoute::
                    PRELOAD_DDR_BROADCAST_TO_SPM_ALL_CORES),
                {has_final_norm_ ? 1 : 0, hidden_size(), NUM_CORES});
            append(
                FmbRouteFamily::MUTABLE_DMA,
                DINO_FINAL_NORM_BETA_PRELOAD_SITE,
                static_cast<int64_t>(DinoVisionMutableDmaRoute::
                    PRELOAD_DDR_BROADCAST_TO_SPM_ALL_CORES),
                {has_final_norm_ ? 1 : 0, hidden_size(), NUM_CORES});
        }
        if (has_output_mlp_) {
            append(
                FmbRouteFamily::MUTABLE_DMA,
                DINO_OUTPUT_MLP_BIAS1_PRELOAD_SITE,
                static_cast<int64_t>(DinoVisionMutableDmaRoute::
                    PRELOAD_DDR_SCATTER_TO_SPM),
                {has_output_mlp_ ? 1 : 0, hidden_size() / NUM_CORES,
                 hidden_size() / NUM_CORES * DWIDTH, NUM_CORES});
            append(
                FmbRouteFamily::MUTABLE_DMA,
                DINO_OUTPUT_MLP_BIAS2_PRELOAD_SITE,
                static_cast<int64_t>(DinoVisionMutableDmaRoute::
                    PRELOAD_DDR_BROADCAST_TO_SPM_CORE0),
                {has_output_mlp_ ? 1 : 0, hidden_size(), 1});
        }
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA);
        std::sort(
            manifest.routes.begin(), manifest.routes.end(),
            [](const FmbRouteManifestEntry& lhs,
               const FmbRouteManifestEntry& rhs) {
                return std::make_tuple(
                           static_cast<int64_t>(lhs.family), lhs.site_id,
                           lhs.invocation) <
                    std::make_tuple(
                           static_cast<int64_t>(rhs.family), rhs.site_id,
                           rhs.invocation);
            });
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

        LayoutContext spm_layout = layout;
        spm_layout.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        if (subclass_spm_kv_by_mha_eligible(
                plan, spm_layout, position)) {
            domain.push_back(physical_manifest_for_candidate(
                plan, spm_layout, physical_len, logical_len, position));
        }
        return domain;
    }

    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& /*manifest*/) const override {
        return {true, FmbGraphLifecycle::RETAINED_CACHE};
    }

    bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const override {
        if (position != 0 || layout.is_causal || layout.use_attn_mask ||
            layout.batch_size != 1 ||
            layout.attention_policy !=
                AttentionExecutionPolicy::SPM_KV_BY_MHA ||
            plan.chunk_mode != ChunkMode::SEQUENTIAL ||
            plan.input.chunks.size() != 1 ||
            plan.qkv.chunks.size() != 1 ||
            plan.compute.chunks.size() != 1 || plan.spans.size() != 1 ||
            use_16b_sdpa_) {
            return false;
        }
        const ChunkInfo& chunk = plan.compute.chunks.front();
        const int64_t active_hybrid_layers =
            hybrid_cpu_sdpa_out_.defined()
            ? std::min<int64_t>(hybrid_layer_count_, num_layers()) : 0;
        return chunk.offset == 0 && chunk.len == 201 &&
            chunk.kv_seq_len == 201 &&
            plan.input.chunks.front().offset == 0 &&
            plan.input.chunks.front().len == 201 &&
            plan.qkv.chunks.front().offset == 0 &&
            plan.qkv.chunks.front().len == 201 &&
            plan.spans.front().offset == 0 && plan.spans.front().len == 201 &&
            num_layers() == 12 && layer_weights_.size() == 12 &&
            active_hybrid_layers < num_layers() &&
            hidden_size() == 768 && intermediate_size() == 3072 &&
            num_q_heads() == 16 && num_kv_heads() == 16 &&
            head_dim() == 64 && orig_head_dim_ == 64 &&
            sdpa_by_mha_spm_is_valid(
                /*batch=*/1, /*seq_q=*/201, /*seq_k=*/201,
                num_q_heads(), num_kv_heads(), head_dim(), NUM_CORES,
                /*MASK_NONE=*/0);
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        const int64_t cs   = ctx.chunk_size;        // full seq for vision
        const int64_t h    = hidden_size();
        const int64_t nq   = num_q_heads();
        const int64_t hd   = head_dim();
        const int64_t is_  = intermediate_size();

        const int64_t local_q_dim = (nq / NUM_CORES) * hd;
        const int64_t local_inter = is_ / NUM_CORES;
        const int64_t local_hidden = h / NUM_CORES;
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        const int64_t res   = A(cs * h * DWIDTH);
        const bool raw_spm = ctx.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const int64_t qkv_rows = raw_spm ? Align(cs, int64_t{16}) : cs;
        const int64_t qkv   = A(qkv_rows * local_q_dim * DWIDTH);
        const int64_t up_sz = A(cs * local_inter * DWIDTH);

        SdpaConfig sdpa_cfg{SdpaKernelType::FLASH_ATTN_SPM,
                            hd, /*nq=*/nq, /*nkv=*/nq,
                            /*cores=*/NUM_CORES, /*mask=*/0};
        const SdpaTiling t = sdpa_compute_tiling(sdpa_cfg, cs);
        const int64_t nkv_per_core = CeilDiv(nq, (int64_t)NUM_CORES);
        int64_t sdpa_tmp =
            A(t.tile_n_v16 * t.tile_k * nkv_per_core * CeilDiv(cs, t.tile_m) * 32);
        if (raw_spm) {
            // Raw V is transposed into a disjoint [H/core, D, PAD16(S)] slot.
            sdpa_tmp = std::max(
                sdpa_tmp, A(qkv_rows * local_q_dim * DWIDTH));
        }

        auto dma_safe = [&](int64_t elems) -> int64_t {
            const int64_t dma_elems = ((elems + 255) / 256) * 256;
            return A(dma_elems * DWIDTH);
        };
        const int64_t norm_w_sz   = dma_safe(h);
        const int64_t q_bias_sz   = dma_safe(local_q_dim);
        const int64_t up_bias_sz  = dma_safe(local_inter);
        const int64_t full_bias_sz = dma_safe(h);
        const int64_t gamma_sz    = dma_safe(h);

        const int nl = static_cast<int>(num_layers());

        std::vector<BufferDecl> decls = {
            // Temps. Phase numbering (mirrors qwen3vl with γ-mul split to
            // run AFTER AllReduce — see P4.5 finding 2026-05-21):
            //   1: LN1 (writes input_norm)
            //   2: Q/K/V Linear (writes q,k,v) + in-place RoPE-2D
            //   3: KV-cache insert + SDPA (writes sdpa_out)
            //   4: O Linear → AllReduce (no residual) → γ_attn × NxC → +residual1
            //      (writes input_norm = residual1 + γ⊙W_o⊙attn)
            //   5: LN2 (writes oproj)
            //   6: up_proj + GELU (writes up_buf)
            //   7: down_proj → AllReduce → γ_mlp × NxC → +input_norm
            //      (writes residual1 = input_norm + γ⊙W_d⊙mlp)
            {"residual1",   res,       1, 7, StorageClass::Temp, 0, nullptr},
            {"input_norm",  res,       1, 7, StorageClass::Temp, 0, nullptr},
            {"oproj",       res,       4, 7, StorageClass::Temp, 0, nullptr},
            // Pre-zeroed dummy residual for AllReduce. The kernel always adds
            // a residual term; passing this all-zero buffer turns it into a
            // plain Σ_cores sum. Persistent (shared across all layers) with
            // a memset preload that runs once during emit_preload_weights.
            // P4.5 root cause: γ × per-core partial pushes intermediate values
            // into fp16 subnormal range (P1.3 γ_attn abs_min = 3.69e-6 ≪
            // fp16 smallest-normal 6.1e-5); AllReduce-then-multiply preserves
            // precision because the sum lands in normal-fp16 range first.
            {"zero_residual", res,     4, 7, StorageClass::Persistent, 0, nullptr},

            // q/k get rotated in place by rope_2d; v unchanged. Phases 2-3.
            {"q",           qkv,       2, 3, StorageClass::Temp, 0, nullptr},
            {"k",           qkv,       2, 3, StorageClass::Temp, 0, nullptr},
            {"v",           qkv,       2, 3, StorageClass::Temp, 0, nullptr},
            {"sdpa_out",    qkv,       3, 7, StorageClass::Temp, 0, nullptr},
            {"sdpa_tmp",    sdpa_tmp,  3, 3, StorageClass::Temp, 0, nullptr},
            {"up_buf",      up_sz,     6, 7, StorageClass::Temp, 0, nullptr},

            // Persistent per-layer (preloaded once via emit_preload_weights).
            {"ln1_gamma",   norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"ln1_beta",    norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"ln2_gamma",   norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"ln2_beta",    norm_w_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"q_bias",      q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            // k_bias is preload-zeroed (DINOv3 k_proj has bias=None). The
            // Q/K/V Linear kernels each ADD their bias slot; zero bias is
            // numerically equivalent to bias=None.
            {"k_bias",      q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"v_bias",      q_bias_sz,    0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"up_bias",     up_bias_sz,   0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"o_bias",      full_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"down_bias",   full_bias_sz, 0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            // LayerScale γ vectors live in persistent SPM (broadcast to all
            // cores) so the eltwise 1xC_NxC kernel can read [1, hidden] from
            // any core during phase 4 / 7. P1.3 decision: NOT folded into
            // o_proj / down_proj weights (fp16 subnormal risk).
            {"gamma_attn",  gamma_sz,     0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
            {"gamma_mlp",   gamma_sz,     0, 0, StorageClass::PersistentPerLayer, nl, nullptr},
        };
        if (has_final_norm_) {
            decls.push_back({"final_norm_gamma", norm_w_sz, 0, 0,
                             StorageClass::Persistent, 0, nullptr});
            decls.push_back({"final_norm_beta", norm_w_sz, 0, 0,
                             StorageClass::Persistent, 0, nullptr});
        }
        if (has_output_mlp_) {
            // The output MLP runs after the transformer's phase-7 tail.  Keep
            // its input, col-parallel hidden, and row-parallel partial in
            // distinct buffers: the phase-1..7 transformer temps are allowed
            // to alias once their declared lifetimes end.
            decls.push_back({"output_mlp_io", res, 7, 10,
                             StorageClass::Temp, 0, nullptr});
            decls.push_back({"output_mlp_partial", res, 7, 10,
                             StorageClass::Temp, 0, nullptr});
            decls.push_back({"output_mlp_hidden",
                             A(cs * local_hidden * DWIDTH), 7, 10,
                             StorageClass::Temp, 0, nullptr});
            decls.push_back({"output_mlp_b1", dma_safe(local_hidden), 0, 0,
                             StorageClass::Persistent, 0, nullptr});
            decls.push_back({"output_mlp_b2", norm_w_sz, 0, 0,
                             StorageClass::Persistent, 0, nullptr});
        }
        return decls;
    }

    void emit_preload_weights() {
        const int64_t h = hidden_size();
        const int64_t local_q_dim = (num_q_heads() / NUM_CORES) * head_dim();
        const int64_t local_inter = intermediate_size() / NUM_CORES;

        // Loop 1: 8-core DMAs for all layers (memset broadcast-bias slots,
        // broadcast norm γ/β and γ_attn/γ_mlp, scatter col-partition biases).
        for (int64_t L = 0; L < num_layers(); ++L) {
            const auto& bn = layer_bias_norm_[L];

            // Memset o_bias / down_bias slots across all 8 cores; the real
            // bias is broadcast-DMAd to core 0 only in Loop 2 below (so the
            // bias contribution comes in once during AllReduce-sum-residual,
            // not 8× — same pattern as qwen3_vl).
            rpu_launch_memset_spm_multicore(layer_addr(L, 0, "o_bias"), h);
            rpu_launch_memset_spm_multicore(layer_addr(L, 0, "down_bias"), h);

            // k_bias slot is permanently zeroed (DINOv3 k_proj bias=None).
            // DINOv2 also uses this zero slot: its fused-QKV key bias adds a
            // query-independent row constant to QK^T, which softmax cancels.
            rpu_launch_memset_spm_multicore(layer_addr(L, 0, "k_bias"),
                                             local_q_dim);

            // LayerNorm γ/β: broadcast to every core.
            rpu_launch_ddr_broadcast_spm_dma(
                bn.ln1_w.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "ln1_gamma"));
            rpu_launch_ddr_broadcast_spm_dma(
                bn.ln1_b.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "ln1_beta"));
            rpu_launch_ddr_broadcast_spm_dma(
                bn.ln2_w.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "ln2_gamma"));
            rpu_launch_ddr_broadcast_spm_dma(
                bn.ln2_b.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "ln2_beta"));

            // LayerScale γ_attn / γ_mlp: broadcast to every core so the
            // eltwise 1xC_NxC kernel in phase 4 / 7 can read [1, hidden]
            // from each core's local SPM.
            rpu_launch_ddr_broadcast_spm_dma(
                bn.gamma_attn.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "gamma_attn"));
            rpu_launch_ddr_broadcast_spm_dma(
                bn.gamma_mlp.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "gamma_mlp"));

            // Col-partition biases: scatter to per-core slots.
            rpu_launch_ddr_scatter_spm_dma(
                bn.q_b.data_ptr<c10::Half>(),
                local_q_dim, local_q_dim * DWIDTH,
                layer_addr(L, 0, "q_bias"),
                /*num_cores=*/NUM_CORES);
            rpu_launch_ddr_scatter_spm_dma(
                bn.v_b.data_ptr<c10::Half>(),
                local_q_dim, local_q_dim * DWIDTH,
                layer_addr(L, 0, "v_bias"),
                /*num_cores=*/NUM_CORES);
            rpu_launch_ddr_scatter_spm_dma(
                bn.up_b.data_ptr<c10::Half>(),
                local_inter, local_inter * DWIDTH,
                layer_addr(L, 0, "up_bias"),
                /*num_cores=*/NUM_CORES);
        }

        if (has_final_norm_) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                DINO_FINAL_NORM_GAMMA_PRELOAD_SITE,
                static_cast<int64_t>(DinoVisionMutableDmaRoute::
                    PRELOAD_DDR_BROADCAST_TO_SPM_ALL_CORES),
                /*resolved_flags=*/0,
                {has_final_norm_ ? 1 : 0, h, NUM_CORES});
            rpu_launch_ddr_broadcast_spm_dma(
                final_norm_w_.data_ptr<c10::Half>(), h,
                addr(0, "final_norm_gamma"));
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                DINO_FINAL_NORM_BETA_PRELOAD_SITE,
                static_cast<int64_t>(DinoVisionMutableDmaRoute::
                    PRELOAD_DDR_BROADCAST_TO_SPM_ALL_CORES),
                /*resolved_flags=*/0,
                {has_final_norm_ ? 1 : 0, h, NUM_CORES});
            rpu_launch_ddr_broadcast_spm_dma(
                final_norm_b_.data_ptr<c10::Half>(), h,
                addr(0, "final_norm_beta"));
        }
        if (has_output_mlp_) {
            rpu_launch_memset_spm_multicore(addr(0, "output_mlp_b2"), h);
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                DINO_OUTPUT_MLP_BIAS1_PRELOAD_SITE,
                static_cast<int64_t>(DinoVisionMutableDmaRoute::
                    PRELOAD_DDR_SCATTER_TO_SPM),
                /*resolved_flags=*/0,
                {has_output_mlp_ ? 1 : 0, h / NUM_CORES,
                 h / NUM_CORES * DWIDTH, NUM_CORES});
            rpu_launch_ddr_scatter_spm_dma(
                output_mlp_b1_.data_ptr<c10::Half>(), h / NUM_CORES,
                h / NUM_CORES * DWIDTH, addr(0, "output_mlp_b1"), NUM_CORES);
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                DINO_OUTPUT_MLP_BIAS2_PRELOAD_SITE,
                static_cast<int64_t>(DinoVisionMutableDmaRoute::
                    PRELOAD_DDR_BROADCAST_TO_SPM_CORE0),
                /*resolved_flags=*/0,
                {has_output_mlp_ ? 1 : 0, h, 1});
            rpu_launch_ddr_broadcast_spm_dma(
                output_mlp_b2_.data_ptr<c10::Half>(), h,
                addr(0, "output_mlp_b2"), 1);
        }

        // Loop 2: 1-core DMAs for row-partition biases (o_b / down_b apply
        // once in the AllReduce-sum, hence broadcast-DMA only on core 0).
        for (int64_t L = 0; L < num_layers(); ++L) {
            const auto& bn = layer_bias_norm_[L];
            rpu_launch_ddr_broadcast_spm_dma(
                bn.o_b.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "o_bias"), /*num_cores=*/1);
            rpu_launch_ddr_broadcast_spm_dma(
                bn.down_b.data_ptr<c10::Half>(), h,
                layer_addr(L, 0, "down_bias"), /*num_cores=*/1);
        }
    }

    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const auto& lw = layer_weights_[layer_idx];
        const int64_t seq_len = chunk.len;        // total tokens = 5 + num_patches
        const int64_t h  = hidden_size();
        const int64_t nq = num_q_heads();
        const int64_t hd = head_dim();
        const int64_t is_ = intermediate_size();
        const int64_t local_inter = is_ / NUM_CORES;
        const int64_t local_q_dim = (nq / NUM_CORES) * hd;

        const int64_t num_patches = seq_len - DINOV3_VISION_NUM_SPECIAL_TOKENS;
        const int64_t skip_bytes =
            DINOV3_VISION_NUM_SPECIAL_TOKENS * local_q_dim * DWIDTH;
        auto launch_linear = [&](uint32_t input, const at::Tensor& weight,
                                 uint32_t output, int64_t M, int64_t N,
                                 int64_t K, int partition, uint32_t bias) {
            if (use_acc32_) {
                ctx().consume_physical_route(
                    FmbRouteFamily::LINEAR, DINO_LINEAR_ACC32_SITE,
                    static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                    /*resolved_flags=*/0, {use_acc32_ ? 1 : 0});
                rpu_launch_linear_spm_to_spm_kernel(
                    input, weight, output, M, N, K, partition, NUM_CORES, bias);
            } else {
                ctx().consume_physical_route(
                    FmbRouteFamily::LINEAR, DINO_LINEAR_ACC16_SITE,
                    static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                    /*resolved_flags=*/0, {use_acc32_ ? 1 : 0});
                rpu_launch_linear_spm_to_spm_acc16_kernel(
                    input, weight, output, M, N, K, partition, NUM_CORES, bias);
            }
        };

        // The cold controlled DINOv2-S profile owns this math route. The
        // COMPLETE descriptor must agree before dispatch, including REPLAY.
        const int64_t norm_route = standalone_compatibility_ ? 2 : 1;
        const auto rpu_launch_layernorm_selected_spm_kernel = standalone_compatibility_
            ? rpu_launch_layernorm_spm_kernel
            : rpu_launch_layernorm_bf16_spm_kernel;

        // Phase 1: DDR→SPM input DMA + LayerNorm1.
        if (!ctx().input_in_spm) {
            emit_layer_input_dma(layer_idx, chunk);
        }

        // One-time-per-forward zero-init of the dummy AllReduce residual.
        // Placed inside layer 0's subgraph so the size (= seq_len * h) is known
        // and the memset runs exactly once per REPLAY. Persistent storage means
        // subsequent layers (and later forwards) inherit the zeroes — AllReduce
        // only reads the residual region, never writes it.
        if (layer_idx == 0) {
            rpu_launch_memset_spm_multicore(addr(0, "zero_residual"),
                                             seq_len * h);
        }

        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION, DINO_LN1_SITE, norm_route, 0, {});
        rpu_launch_layernorm_selected_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "ln1_gamma"), layer_addr(layer_idx, 0, "ln1_beta"),
            seq_len, h, eps_, false, 0, NUM_CORES);

        // [DEBUG] Layer-0 post-LN1 probe — broadcast-replicated, read core 0.
        if (layer_idx == 0 && get_debug_export() && post_ln1_debug_buf_.defined()) {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "input_norm"),
                post_ln1_debug_buf_.data_ptr<c10::Half>(),
                seq_len * h);
        }

        // Phase 2: Q / K / V Linear with bias (col-partition).
        // k_bias slot is preload-zeroed (DINOv3 k_proj bias=None).
        const int64_t per_core_qkv_elems = seq_len * local_q_dim;
        const int64_t per_layer_qkv_elems = NUM_CORES * per_core_qkv_elems;
        launch_linear(
            addr(0, "input_norm"), lw.q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1,
            layer_addr(layer_idx, 0, "q_bias"));
        launch_linear(
            addr(0, "input_norm"), lw.k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1,
            layer_addr(layer_idx, 0, "k_bias"));
        launch_linear(
            addr(0, "input_norm"), lw.v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1,
            layer_addr(layer_idx, 0, "v_bias"));

        // Phase 2.5: 2D RoPE on Q and K — patch tokens only.
        // CLS + 4 register tokens (the first DINOV3_VISION_NUM_SPECIAL_TOKENS
        // entries per core) stay unrotated. The kernel reads `num_patches`
        // entries from position_idx[0..num_patches-1] and rotates `num_patches`
        // tokens starting at `addr(0, "q") + skip_bytes` (i.e., the patch slice
        // within the per-core Q SPM buffer). Same for K.
        if (num_patches > 0 && !identity_rope_) {
            const int64_t local_heads = nq / NUM_CORES;
            const int64_t head_dim_pad = hd;  // head_dim=64, no padding.
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, DINO_Q_ROPE_DDR_SITE,
                static_cast<int64_t>(DinoVisionRopeRoute::ROPE_2D_DDR),
                /*resolved_flags=*/0,
                {identity_rope_ ? 1 : 0, num_patches, local_heads, hd,
                 head_dim_pad, NUM_CORES});
            rpu_launch_rope_2d_ddr_kernel(
                addr(0, "q") + skip_bytes, addr(0, "q") + skip_bytes,
                freq_cos_.data_ptr<c10::Half>(),
                freq_sin_.data_ptr<c10::Half>(),
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/0,
                num_patches,
                local_heads, hd, head_dim_pad, NUM_CORES);
            ctx().consume_physical_route(
                FmbRouteFamily::ROPE, DINO_K_ROPE_DDR_SITE,
                static_cast<int64_t>(DinoVisionRopeRoute::ROPE_2D_DDR),
                /*resolved_flags=*/0,
                {identity_rope_ ? 1 : 0, num_patches, local_heads, hd,
                 head_dim_pad, NUM_CORES});
            rpu_launch_rope_2d_ddr_kernel(
                addr(0, "k") + skip_bytes, addr(0, "k") + skip_bytes,
                freq_cos_.data_ptr<c10::Half>(),
                freq_sin_.data_ptr<c10::Half>(),
                position_idx_keepalive_.data_ptr<int16_t>(),
                /*pos_offset=*/0,
                num_patches,
                local_heads, hd, head_dim_pad, NUM_CORES);
        }

        // [DEBUG] Per-layer post-Q/K/V probe (post-RoPE). Gather 8 cores' SPM
        // slice into a per-layer offset in the [N, 8, seq, local_q_dim] buf.
        if (get_debug_export() && post_q_debug_buf_.defined()) {
            c10::Half* q_base = post_q_debug_buf_.data_ptr<c10::Half>()
                                + layer_idx * per_layer_qkv_elems;
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "q"), q_base,
                per_core_qkv_elems, per_core_qkv_elems * (int64_t)DWIDTH, NUM_CORES);
            c10::Half* k_base = post_k_debug_buf_.data_ptr<c10::Half>()
                                + layer_idx * per_layer_qkv_elems;
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "k"), k_base,
                per_core_qkv_elems, per_core_qkv_elems * (int64_t)DWIDTH, NUM_CORES);
            c10::Half* v_base = post_v_debug_buf_.data_ptr<c10::Half>()
                                + layer_idx * per_layer_qkv_elems;
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "v"), v_base,
                per_core_qkv_elems, per_core_qkv_elems * (int64_t)DWIDTH, NUM_CORES);
        }

        // Phase 3: KV cache insert (position=0 always for vision) + bidir SDPA.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const auto& kv_route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, DINO_KV_INSERT_SITE, chunk.idx);
        const KvInsertSegmentPlan kv_plan =
            restore_kvinsert_plan(
                DINO_KV_INSERT_SITE, kv_route.arguments, NUM_CORES, nq, hd);
        TORCH_CHECK(
            kv_plan.logical_rows() == seq_len &&
                kv_plan.physical_rows() == seq_len &&
                kv_plan.segment(0).position == chunk.offset,
            "DINO Vision KV descriptor geometry drift");
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, DINO_KV_INSERT_SITE,
            static_cast<int64_t>(kv_plan.route()),
            DINO_KV_FLAG_DDR_MIRROR, kv_route.arguments, chunk.idx);
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache,
            addr_offset("k").value, addr_offset("v").value,
            nq, hd, NUM_CORES,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            /*spm_rows=*/0, kv_plan);

        // Hybrid CPU-SDPA branch: layer_idx < hybrid_layer_count_ → skip the
        // fp16 RPU SDPA kernel (which overflows on ViT-S Q·K^T) and instead
        // DMA the externally-computed sdpa_out (per-core layout
        // [8, seq, local_q_dim] held inside the [N, 8, seq, local_q_dim] DDR
        // staging) into SPM 'sdpa_out'. KV cache insert above is still emitted
        // so the cache is in a defined state, but the cache content is not
        // read by anything in this branch.
        if (layer_idx < hybrid_layer_count_ && hybrid_cpu_sdpa_out_.defined()) {
            c10::Half* src = hybrid_cpu_sdpa_out_.data_ptr<c10::Half>()
                             + layer_idx * per_layer_qkv_elems;
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                DINO_HYBRID_ATTENTION_SITE,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                DINO_HYBRID_FLAG_CPU_SDPA_DDR_REQUIRED,
                {hybrid_layer_count_, seq_len, nq, hd, NUM_CORES},
                layer_idx);
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                DINO_HYBRID_SDPA_DMA_SITE,
                static_cast<int64_t>(DinoVisionMutableDmaRoute::
                    HYBRID_SDPA_DDR_SCATTER_TO_SPM),
                DINO_HYBRID_FLAG_CPU_SDPA_DDR_REQUIRED,
                {hybrid_layer_count_, per_core_qkv_elems,
                 per_core_qkv_elems * DWIDTH, NUM_CORES}, layer_idx);
            rpu_launch_ddr_scatter_spm_dma(
                src, per_core_qkv_elems,
                per_core_qkv_elems * (int64_t)DWIDTH,
                addr(0, "sdpa_out"), NUM_CORES);
        } else {
            const double attn_scale =
                1.0 / std::sqrt(static_cast<double>(orig_head_dim_));
            // Select the FP16 bank variant per model. This preserves each
            // profile's accumulation route; the retired BF16 kernel is not used.
            if (ctx().attention_policy ==
                AttentionExecutionPolicy::SPM_KV_BY_MHA) {
                TORCH_CHECK(
                    !use_16b_sdpa_ && seq_len == 201,
                    "DINO raw-SPM attention escaped its exact ViT-B profile");
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION,
                    DINO_ATTENTION_RAW_SPM_SITE,
                    static_cast<int64_t>(
                        AttentionExecutionPolicy::SPM_KV_BY_MHA),
                    /*resolved_flags=*/0,
                    {0, hybrid_layer_count_, seq_len, nq, hd, NUM_CORES},
                    layer_idx);
                rpu_launch_v_transpose_spm(
                    addr(0, "v"), addr(0, "sdpa_tmp"),
                    /*batch=*/1, seq_len, nq, hd, NUM_CORES);
                rpu_launch_sdpa_by_mha_spm(
                    addr(0, "q"), addr(0, "k"), addr(0, "sdpa_tmp"),
                    addr(0, "sdpa_out"), /*mask_spm=*/0,
                    /*MASK_NONE=*/0, attn_scale, /*batch=*/1,
                    seq_len, seq_len, nq, nq, hd, NUM_CORES);
            } else if (use_16b_sdpa_) {
                const int64_t flags = DINO_ATTENTION_FLAG_USE_16B |
                    DINO_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED;
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION, DINO_ATTENTION_16B_SITE,
                    static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                    flags,
                    {use_16b_sdpa_ ? 1 : 0, hybrid_layer_count_, seq_len,
                     nq, hd, NUM_CORES},
                    layer_idx);
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    k_cache, v_cache,
                    0 /*MASK_NONE*/, attn_scale,
                    addr_offset("q").value,
                    addr_offset("sdpa_out").value,
                    addr_offset("sdpa_tmp").value, 0,
                    seq_len, nq, nq, hd,
                    seq_len, NUM_CORES, NUM_CORES,
                    /*cache_batch_offset_elems=*/0,
                    /*use_16b=*/true);
            } else {
                LayoutContext spm_layout;
                spm_layout.attention_policy =
                    AttentionExecutionPolicy::SPM_KV_BY_MHA;
                spm_layout.batch_size = ctx().batch_size;
                spm_layout.is_causal = ctx().is_causal;
                spm_layout.use_attn_mask = ctx().attention_mask.has_value();
                const bool raw_profile = subclass_spm_kv_by_mha_eligible(
                    ctx().stage_plan, spm_layout, ctx().position);
                ctx().consume_physical_route(
                    FmbRouteFamily::ATTENTION, DINO_ATTENTION_32B_SITE,
                    static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                    raw_profile
                        ? 0
                        : DINO_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                    {0, hybrid_layer_count_, seq_len, nq, hd, NUM_CORES},
                    layer_idx);
                rpu_launch_sdpa_spm_unified_kernel_v2(
                    k_cache, v_cache,
                    0 /*MASK_NONE*/, attn_scale,
                    addr_offset("q").value,
                    addr_offset("sdpa_out").value,
                    addr_offset("sdpa_tmp").value, 0,
                    seq_len, nq, nq, hd,
                    seq_len, NUM_CORES, NUM_CORES,
                    /*cache_batch_offset_elems=*/0,
                    /*use_16b=*/false);
            }
        }

        // [DEBUG] Per-layer post-SDPA probe.
        if (get_debug_export() && post_sdpa_debug_buf_.defined()) {
            c10::Half* sdpa_base = post_sdpa_debug_buf_.data_ptr<c10::Half>()
                                   + layer_idx * per_layer_qkv_elems;
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "sdpa_out"), sdpa_base,
                per_core_qkv_elems, per_core_qkv_elems * (int64_t)DWIDTH, NUM_CORES);
        }

        // Phase 4: O_proj (row-partition, with bias)
        //          → AllReduce (no residual; sum across cores into input_norm)
        //          → γ_attn × NxC mul (in-place on input_norm = γ ⊙ o_full)
        //          → eltwise ADD residual1 (writes input_norm = pre-LN2 residual).
        //
        // P4.5 root cause (2026-05-21 debug session): the original sequence
        // (per-core γ × partial → AllReduce + Residual) was math-equivalent in
        // real arithmetic but pushed intermediate values into fp16 subnormal
        // range when γ entries fell below 6.1e-5 (P1.3 γ_attn abs_min = 3.7e-6).
        // 8 subnormal partials summed → lossy. AllReduce-then-multiply keeps
        // the sum in normal fp16 before the one γ scaling, matching HF math.
        const int64_t num_elems_full = seq_len * h;
        launch_linear(
            addr(0, "sdpa_out"), lw.o_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0,
            layer_addr(layer_idx, 0, "o_bias"));
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE, DINO_ATTN_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h));
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "zero_residual"), addr(0, "input_norm"),
            seq_len, h, NUM_CORES, NUM_CORES);
        // [DEBUG] Layer-0 post-OProj-AR probe — raw attn output (= attn(LN1(x)))
        // before LayerScale1 / residual. Reading 'input_norm' here is exactly
        // the AllReduce sum of per-core O_proj partials (broadcast).
        if (layer_idx == 0 && get_debug_export() && post_oproj_ar_debug_buf_.defined()) {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "input_norm"),
                post_oproj_ar_debug_buf_.data_ptr<c10::Half>(),
                seq_len * h);
        }
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            layer_addr(layer_idx, 0, "gamma_attn"),
            addr(0, "input_norm"), addr(0, "input_norm"),
            seq_len, h, c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "input_norm"), addr(0, "residual1"), addr(0, "input_norm"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0f), NUM_CORES);

        // [DEBUG] Layer-0 post-attn-block probe (input_norm = γ_attn·AR(Oproj)+residual1).
        if (layer_idx == 0 && get_debug_export() && post_attn_block_debug_buf_.defined()) {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "input_norm"),
                post_attn_block_debug_buf_.data_ptr<c10::Half>(),
                seq_len * h);
        }

        // Phase 5: LayerNorm2 on the post-attn residual.
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION, DINO_LN2_SITE, norm_route, 0, {});
        rpu_launch_layernorm_selected_spm_kernel(
            addr(0, "input_norm"), addr(0, "oproj"),
            layer_addr(layer_idx, 0, "ln2_gamma"), layer_addr(layer_idx, 0, "ln2_beta"),
            seq_len, h, eps_, false, 0, NUM_CORES);

        // [DEBUG] Layer-0 post-LN2 probe — broadcast-replicated, read core 0.
        if (layer_idx == 0 && get_debug_export() && post_ln2_debug_buf_.defined()) {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "oproj"),
                post_ln2_debug_buf_.data_ptr<c10::Half>(),
                seq_len * h);
        }

        // Phase 6: up_proj (col-partition, with bias) + GELU.
        // HF DINOv3 default hidden_act = "gelu" (exact, NOT tanh-approx) per
        // configuration_dinov3_vit.py, so select the explicit erf formula.
        launch_linear(
            addr(0, "oproj"), lw.up_w, addr(0, "up_buf"),
            seq_len, is_, h, 1,
            layer_addr(layer_idx, 0, "up_bias"));
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "up_buf"), addr(0, "up_buf"),
            seq_len * local_inter, ValuOpType::ADD, GeluMode::ERF, NUM_CORES);

        // [DEBUG] Layer-0 post-up_proj+GELU probe (col-partition scatter).
        if (layer_idx == 0 && get_debug_export() && post_up_proj_debug_buf_.defined()) {
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "up_buf"),
                post_up_proj_debug_buf_.data_ptr<c10::Half>(),
                seq_len * local_inter,
                seq_len * local_inter * (int64_t)DWIDTH, NUM_CORES);
        }

        // Phase 7: down_proj (row-partition, with bias)
        //          → AllReduce (no residual; sum across cores into residual1)
        //          → γ_mlp × NxC mul (in-place on residual1 = γ ⊙ mlp_full)
        //          → eltwise ADD input_norm (writes residual1 = pre-block state
        //            for next layer's LN1 under SPM_RESIDENT mode).
        // Same fp16-subnormal reasoning as Phase 4 (P4.5 finding).
        launch_linear(
            addr(0, "up_buf"), lw.down_w, addr(0, "oproj"),
            seq_len, h, is_, 0,
            layer_addr(layer_idx, 0, "down_bias"));

        // [DEBUG] Layer-0 post-down_proj per-core partial probe (BEFORE AR).
        // Each core holds the full [seq, h] but partial-sum over its slice of
        // the K=intermediate dim. Gather all 8 cores' partials so Python can
        // distinguish per-core overflow vs AR-sum overflow.
        if (layer_idx == 0 && get_debug_export() && post_down_partial_debug_buf_.defined()) {
            rpu_launch_spm_scatter_ddr_dma(
                addr(0, "oproj"),
                post_down_partial_debug_buf_.data_ptr<c10::Half>(),
                seq_len * h,
                seq_len * h * (int64_t)DWIDTH, NUM_CORES);
        }

        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE, DINO_MLP_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h));
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "zero_residual"), addr(0, "residual1"),
            seq_len, h, NUM_CORES, NUM_CORES);

        // [DEBUG] Layer-0 post-AR probe (broadcast residual1 after AllReduce,
        // BEFORE γ_mlp multiply and BEFORE the final residual add).
        if (layer_idx == 0 && get_debug_export() && post_down_ar_debug_buf_.defined()) {
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "residual1"),
                post_down_ar_debug_buf_.data_ptr<c10::Half>(),
                seq_len * h);
        }

        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            layer_addr(layer_idx, 0, "gamma_mlp"),
            addr(0, "residual1"), addr(0, "residual1"),
            seq_len, h, c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "residual1"), addr(0, "input_norm"), addr(0, "residual1"),
            num_elems_full, ValuOpType::ADD, c10::Half(1.0f), NUM_CORES);

        // [DEBUG] Per-layer hidden-state SPM→DDR dump (gemma2 pattern). Gated
        // on get_debug_export() at graph BUILD time. The DMA target pointer
        // is baked into the cached graph; toggling debug export later
        // requires reset_graph_cache() (which set_debug_export() does).
        // residual1 holds the full layer output (post-MLP-block residual)
        // broadcast-replicated across all 8 cores; reading core 0's slot
        // is sufficient. DMA channel 0 lives on stream 0 (core 0's compute
        // stream), giving natural read-after-write serialization with the
        // upstream residual1 write without a redundant multi-core barrier.
        if (get_debug_export() && per_layer_debug_buf_.defined()) {
            const int64_t bs = per_layer_debug_buf_.size(1);  // == 1 for vision
            const int64_t st = per_layer_debug_buf_.size(2);
            c10::Half* layer_base = per_layer_debug_buf_.data_ptr<c10::Half>()
                + layer_idx * bs * st * h
                + chunk.offset * h;
            rpu_launch_spm_copy_ddr_dma(
                addr(0, "residual1"),
                layer_base,
                chunk.len * h);
        }

        // Output DMA for non-SPM-resident path (last layer or framework
        // override). Framework's run_all_layers writes residual1 →
        // output_tensor_ via the standard tail; this branch keeps parity
        // with qwen3_vl_vision_model.cpp:626-637.
        const bool apply_final_norm =
            has_final_norm_ && layer_idx == static_cast<int>(num_layers()) - 1;
        if (apply_final_norm) {
            ctx().consume_physical_route(
                FmbRouteFamily::NORMALIZATION, DINO_FINAL_NORM_SITE, norm_route, 0, {});
            rpu_launch_layernorm_selected_spm_kernel(
                addr(0, "residual1"),
                addr(0, has_output_mlp_ ? "output_mlp_io" : "input_norm"),
                addr(0, "final_norm_gamma"), addr(0, "final_norm_beta"),
                seq_len, h, eps_, false, 0, NUM_CORES);
        }
        const bool apply_output_mlp =
            has_output_mlp_ && layer_idx == static_cast<int>(num_layers()) - 1;
        if (apply_output_mlp) {
            const uint32_t mlp_input = apply_final_norm
                ? addr(0, "output_mlp_io") : addr(0, "residual1");
            launch_linear(
                mlp_input, output_mlp_w1_, addr(0, "output_mlp_hidden"),
                seq_len, h, h, /*partition=*/1, addr(0, "output_mlp_b1"));
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "output_mlp_hidden"), addr(0, "output_mlp_hidden"),
                seq_len * (h / NUM_CORES), ValuOpType::ADD,
                GeluMode::ERF, NUM_CORES);
            launch_linear(
                addr(0, "output_mlp_hidden"), output_mlp_w2_,
                addr(0, "output_mlp_partial"),
                seq_len, h, h, /*partition=*/0, addr(0, "output_mlp_b2"));
            consume_manifest_route(
                FmbRouteFamily::ALL_REDUCE,
                DINO_OUTPUT_MLP_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(seq_len, h));
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "output_mlp_partial"), addr(0, "zero_residual"),
                addr(0, "output_mlp_io"), seq_len, h,
                NUM_CORES, NUM_CORES);
        }
        if (!ctx().output_to_spm) {
            emit_layer_output_dma(
                layer_idx, chunk,
                apply_output_mlp ? "output_mlp_io"
                    : (apply_final_norm ? "input_norm" : "residual1"));
        }
    }

private:
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.insert(identity.end(), {
            static_cast<int64_t>(has_rope_),
            static_cast<int64_t>(has_final_norm_),
            static_cast<int64_t>(has_output_mlp_),
            static_cast<int64_t>(use_acc32_),
            static_cast<int64_t>(use_16b_sdpa_),
            static_cast<int64_t>(standalone_compatibility_),
            static_cast<int64_t>(identity_rope_)});
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.q_w, &weights.k_w, &weights.v_w, &weights.o_w,
                    &weights.up_w, &weights.down_w}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        identity.push_back(static_cast<int64_t>(layer_bias_norm_.size()));
        for (const auto& weights : layer_bias_norm_) {
            for (const auto* tensor : {
                    &weights.ln1_w, &weights.ln1_b, &weights.ln2_w, &weights.ln2_b,
                    &weights.q_b, &weights.v_b, &weights.o_b, &weights.up_b,
                    &weights.down_b, &weights.gamma_attn, &weights.gamma_mlp}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &final_norm_w_, &final_norm_b_, &output_mlp_w1_, &output_mlp_b1_,
                &output_mlp_w2_, &output_mlp_b2_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    std::vector<LayerWeights> layer_weights_;
    std::vector<LayerBiasNorm> layer_bias_norm_;

    // 2D RoPE state
    at::Tensor freq_cos_;
    at::Tensor freq_sin_;
    at::Tensor position_idx_keepalive_;
    int64_t max_hw_ = 0;
    bool has_rope_ = false;

    // Optional DINOv2-S final LayerNorm and caption projection.
    at::Tensor final_norm_w_, final_norm_b_;
    at::Tensor output_mlp_w1_, output_mlp_b1_, output_mlp_w2_, output_mlp_b2_;
    bool has_final_norm_ = false;
    bool has_output_mlp_ = false;

    // Model config
    double eps_ = 1e-5;       // DINOv3 layer_norm_eps default
    int64_t orig_head_dim_ = 0;

    // [DEBUG] Per-layer hidden-state DDR staging. Allocated lazily in
    // forward() when get_debug_export() is true. Shape [N, bs, seq_len, h].
    // Pointer baked into the BUILD-time spm→ddr DMA via gemma2 pattern.
    at::Tensor per_layer_debug_buf_;
    // Layer-0 phase-level probes (broadcast-replicated SPM buffers).
    at::Tensor post_ln1_debug_buf_;          // 'input_norm' after LN1
    at::Tensor post_oproj_ar_debug_buf_;     // 'input_norm' after AllReduce (raw attn out)
    at::Tensor post_attn_block_debug_buf_;   // 'input_norm' after Oproj+AR+γ+residual1
    at::Tensor post_ln2_debug_buf_;          // 'oproj' after LN2
    at::Tensor post_down_ar_debug_buf_;      // 'residual1' after MLP AllReduce (pre γ_mlp)
    // Layer-0 MLP-internal per-core probes (col/row-partition scatter).
    at::Tensor post_up_proj_debug_buf_;      // 'up_buf' after up_proj + GELU (col-partition)
    at::Tensor post_down_partial_debug_buf_; // 'oproj' down_proj per-core partial (pre-AR)
    // Col-partition layer-0 probes: shape [NUM_CORES=8, seq_len, local_q_dim].
    at::Tensor post_q_debug_buf_;            // 'q' after RoPE (per-layer)
    at::Tensor post_k_debug_buf_;            // 'k' after RoPE (per-layer)
    at::Tensor post_v_debug_buf_;            // 'v' after V linear (per-layer)
    at::Tensor post_sdpa_debug_buf_;         // 'sdpa_out' (per-layer)

public:
    // ========================================================================
    // Hybrid CPU-SDPA fallback (verification / fp16 SDPA overflow workaround).
    //
    // When `hybrid_layer_count` > 0, layers [0, hybrid_layer_count) skip the
    // RPU FLASH_ATTN_SPM kernel and instead pull externally-computed sdpa_out
    // from the per-core scattered DDR tensor `hybrid_cpu_sdpa_out`. Used by
    // diagnostics to isolate FP16 attention overflow from the remaining layers.
    //
    // Shape contract:
    //   hybrid_cpu_sdpa_out: [num_layers, NUM_CORES=8, num_tokens, local_q_dim]
    //     fp16 on PrivateUse1 (RPU), contiguous. Per-core layout matches the
    //     dump produced by post_sdpa_debug_buf_ (head-major: core c holds head
    //     c when local_q_heads=1, or heads {c*lqh .. c*lqh+lqh-1} otherwise).
    //
    // Toggling invalidates the cached graph (BUILD-time decision on which
    // kernel to emit for SDPA).
    // ========================================================================
    void set_hybrid_cpu_sdpa(const at::Tensor& sdpa_out, int64_t hybrid_layer_count) {
        TORCH_CHECK(hybrid_layer_count >= 0,
                    "set_hybrid_cpu_sdpa: hybrid_layer_count must be >= 0, got ",
                    hybrid_layer_count);
        if (hybrid_layer_count > 0) {
            TORCH_CHECK(sdpa_out.defined(),
                        "set_hybrid_cpu_sdpa: sdpa_out must be defined when "
                        "hybrid_layer_count > 0");
            TORCH_CHECK(sdpa_out.dim() == 4,
                        "set_hybrid_cpu_sdpa: sdpa_out must be 4D "
                        "[N, NUM_CORES, num_tokens, local_q_dim], got ",
                        sdpa_out.dim(), "D");
            TORCH_CHECK(sdpa_out.scalar_type() == at::kHalf,
                        "set_hybrid_cpu_sdpa: sdpa_out must be fp16");
            TORCH_CHECK(sdpa_out.device().type() == at::kPrivateUse1,
                        "set_hybrid_cpu_sdpa: sdpa_out must be on RPU device");
            TORCH_CHECK(sdpa_out.is_contiguous(),
                        "set_hybrid_cpu_sdpa: sdpa_out must be contiguous");
        }
        hybrid_cpu_sdpa_out_ = sdpa_out;
        hybrid_layer_count_ = hybrid_layer_count;
        invalidate_model_state();
    }

private:
    // Hybrid CPU-SDPA injection state (see set_hybrid_cpu_sdpa above).
    at::Tensor hybrid_cpu_sdpa_out_;
    int64_t hybrid_layer_count_ = 0;

public:
    void set_standalone_compatibility(bool enabled) {
        TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                    "DINO standalone compatibility must be set before the first forward");
        TORCH_CHECK(!enabled ||
                        (num_layers() == 12 && hidden_size() == 384 &&
                         intermediate_size() == 1536 && num_q_heads() == 8 &&
                         head_dim() == 64 && identity_rope_ && use_acc32_ &&
                         use_16b_sdpa_),
                    "DINO standalone compatibility requires the controlled DINOv2-S profile");
        standalone_compatibility_ = enabled;
        invalidate_model_state();
    }

    void set_acc32(bool enabled) {
        TORCH_CHECK(!standalone_compatibility_ || enabled,
                    "DINO standalone compatibility precision/RoPE profile is immutable");
        if (use_acc32_ != enabled) {
            use_acc32_ = enabled;
            invalidate_model_state();
        }
    }

    void set_16b_sdpa(bool enabled) {
        TORCH_CHECK(!standalone_compatibility_ || enabled,
                    "DINO standalone compatibility precision/RoPE profile is immutable");
        if (use_16b_sdpa_ != enabled) {
            use_16b_sdpa_ = enabled;
            invalidate_model_state();
        }
    }

    void set_identity_rope(bool enabled) {
        TORCH_CHECK(!standalone_compatibility_ || enabled,
                    "DINO standalone compatibility precision/RoPE profile is immutable");
        if (identity_rope_ != enabled) {
            identity_rope_ = enabled;
            invalidate_model_state();
        }
    }

private:
    bool standalone_compatibility_ = false;
    bool use_acc32_ = false;
    bool use_16b_sdpa_ = false;
    bool identity_rope_ = false;
};

}  // namespace v3

using DINOv3VisionRegistry = ModelHandleRegistry<v3::DINOv3VisionModel>;

std::vector<int64_t> rpu_dinov3_vision_planner_cache_identity(int64_t handle) {
    return DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_dinov3_vision_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_dinov3_vision_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_dinov3_vision_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision_kvinsert_cost_domain")
        ->kvinsert_cost_domain("dinov3_vision", descriptor);
}

std::string rpu_dinov3_vision_kvinsert_cost_catalog_sha256(int64_t handle) {
    return DINOv3VisionRegistry::get(
        handle, "rpu_dinov3_vision_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

// =============================================================================
// Public C API for TORCH_LIBRARY_IMPL wrappers (P4.1 exposes only create/destroy)
// =============================================================================

int64_t rpu_dinov3_vision_create() {
    return DINOv3VisionRegistry::create();
}

void rpu_dinov3_vision_destroy(int64_t handle) {
    DINOv3VisionRegistry::destroy(handle, "rpu_dinov3_vision_destroy");
}

void rpu_dinov3_vision_set_weights(
    int64_t handle,
    at::TensorList q_w_list,  at::TensorList k_w_list,
    at::TensorList v_w_list,  at::TensorList o_w_list,
    at::TensorList up_w_list, at::TensorList down_w_list,
    at::TensorList ln1_w_list, at::TensorList ln1_b_list,
    at::TensorList ln2_w_list, at::TensorList ln2_b_list,
    at::TensorList q_b_list,  at::TensorList v_b_list, at::TensorList o_b_list,
    at::TensorList up_b_list, at::TensorList down_b_list,
    at::TensorList gamma_attn_list, at::TensorList gamma_mlp_list,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps)
{
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")->set_weights(
        q_w_list, k_w_list, v_w_list, o_w_list,
        up_w_list, down_w_list,
        ln1_w_list, ln1_b_list, ln2_w_list, ln2_b_list,
        q_b_list, v_b_list, o_b_list,
        up_b_list, down_b_list,
        gamma_attn_list, gamma_mlp_list,
        num_heads, head_dim, hidden_size, intermediate_size, eps);
}

void rpu_dinov3_vision_set_rope(
    int64_t handle,
    const at::Tensor& freq_cos,
    const at::Tensor& freq_sin)
{
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->set_rope_tables(freq_cos, freq_sin);
}

at::Tensor rpu_dinov3_vision_position_idx_keepalive(int64_t handle) {
    return DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->position_idx_keepalive();
}

void rpu_dinov3_vision_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision_set_chunk_envelope")
        ->set_chunk_envelope(std::min(max_kv_len,
            DINOV3_VISION_MAX_KEEPALIVE_SEQ + DINOV3_VISION_NUM_SPECIAL_TOKENS), chunk);
}

void rpu_dinov3_vision_set_chunk_size(int64_t handle, int64_t chunk_size) {
    auto model = DINOv3VisionRegistry::get(
        handle, "rpu_dinov3_vision_set_chunk_size");
    TORCH_CHECK(model->get_last_resolved_chunk_size() == 0,
                "DINOv3 vision chunk size must be set before the first forward");
    model->set_control_chunk_size_override(chunk_size, "rpu_dinov3_vision_set_chunk_size");
}

void rpu_dinov3_vision_stage_chunk_size(
        int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0,
                "DINOv3 vision hot-reconfigure token must be positive");
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision_stage_chunk_size")
        ->stage_control_chunk_size_override(
            static_cast<uint64_t>(token), chunk_size,
            "rpu_dinov3_vision_stage_chunk_size");
}

int64_t rpu_dinov3_vision_get_chunk_size_override(int64_t handle) {
    return DINOv3VisionRegistry::get(
        handle, "rpu_dinov3_vision_get_chunk_size_override")
        ->get_chunk_size_override();
}

int64_t rpu_dinov3_vision_resolve_chunk_size(
    int64_t handle, int64_t num_tokens) {
    return DINOv3VisionRegistry::get(
        handle, "rpu_dinov3_vision_resolve_chunk_size")
        ->resolve_vision_chunk_size(num_tokens);
}

std::vector<int64_t> rpu_dinov3_vision_resolve_stage_domain(
    int64_t handle, int64_t num_tokens) {
    return DINOv3VisionRegistry::get(
        handle, "rpu_dinov3_vision_resolve_stage_domain")
        ->resolve_stage_domain(num_tokens);
}

int64_t rpu_dinov3_vision_get_resolved_chunk_size(int64_t handle) {
    return DINOv3VisionRegistry::get(
        handle, "rpu_dinov3_vision_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

at::Tensor rpu_dinov3_vision_forward(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    int64_t num_tokens,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->forward(input, k_caches, v_caches, num_tokens,
                  planned_stage_descriptor);
}

void rpu_dinov3_vision_set_hybrid_cpu_sdpa(
    int64_t handle,
    const at::Tensor& sdpa_out,
    int64_t hybrid_layer_count)
{
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->set_hybrid_cpu_sdpa(sdpa_out, hybrid_layer_count);
}

void rpu_dinov3_vision_set_acc32(int64_t handle, bool enabled) {
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->set_acc32(enabled);
}

void rpu_dinov3_vision_set_16b_sdpa(int64_t handle, bool enabled) {
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->set_16b_sdpa(enabled);
}

void rpu_dinov3_vision_set_standalone_compatibility(int64_t handle, bool enabled) {
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->set_standalone_compatibility(enabled);
}

void rpu_dinov3_vision_set_identity_rope(int64_t handle, bool enabled) {
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->set_identity_rope(enabled);
}

void rpu_dinov3_vision_set_final_norm(
    int64_t handle,
    const at::Tensor& weight,
    const at::Tensor& bias)
{
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->set_final_norm(weight, bias);
}

void rpu_dinov3_vision_set_output_mlp(
    int64_t handle,
    const at::Tensor& first_weight, const at::Tensor& first_bias,
    const at::Tensor& second_weight, const at::Tensor& second_bias)
{
    DINOv3VisionRegistry::get(handle, "rpu_dinov3_vision")
        ->set_output_mlp(first_weight, first_bias, second_weight, second_bias);
}
