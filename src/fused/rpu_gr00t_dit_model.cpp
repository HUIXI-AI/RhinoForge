// rpu_gr00t_dit_model.cpp — GR00T-N1.7 flow-matching DiT action head.
//
// 32-layer AlternateVLDiT with explicit block_is_cross per-layer flags:
// self-attention operates on state/action tokens; cross-attention projects
// external vl_embeds through to_k/to_v and applies image/text subset masks.
// Both norm1 paths use timestep-conditioned AdaLayerNorm.
// pre_layers_fn encodes embodiments; post_layers_fn applies norm_out, proj_out,
// the decoder and Euler integration. body_iterations controls the step count.
// cond = SiLU(temb) is prepared on the host. All DiT linears carry bias.
// set_weights ends with invalidate_model_state().

#include "fused_model_base.h"
#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_eltwise.h"
#include "rpu_helpers.h"
#include "rpu_runtime_state.h"
#include "rpu_gr00t_spm_z2.h"
#include "rpu_spm_residency.h"
#include "execution_coordinator.h"
#include <ATen/record_function.h>
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <string>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

static inline int64_t gr00t_pad16(int64_t n) { return ((n + 15) / 16) * 16; }

// W8A16 per-projection scale validation. Same four checks as the already-strong
// sites (rpu_siglip_model.cpp:227, rpu_pi05_denoise_step_model.cpp:68): list
// length, int8 weight dtype, 1D contiguous fp16 RPU scale, and
// scale.numel() == output channels. Checking only the list length is not enough —
// a wrong-length or wrong-dtype scale reaches the generated W8A16 family as garbage
// per-channel multipliers, i.e. a silent numeric corruption rather than a crash.
// The `else` leg is the reverse hole: an int8 weight with no scale list is read
// back as fp16 by the kernel.
static void check_gr00t_w8a16_scale_list(int64_t n_layers,
                                         const at::TensorList& weights,
                                         const at::TensorList& scales,
                                         const char* ctx, const char* name) {
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

namespace {

// Stable descriptor identities. GR00T_DIT_FIXED_KERNEL_BASIS: every other
// launch is fixed checkpoint math (normalization, activation, residual/Euler),
// unconditional parameter preload or graph output transfer.
// Denoise-conditioned table/bias preloads and the other named sites are routes.
constexpr int64_t GR00T_DIT_MASK_SCHEDULE_SITE = 4273136855374514935LL;
enum class Gr00tDiTMaskSchedule : int64_t {
    UPLOAD_EACH_CROSS_BLOCK = 1,
    RETAIN_TEXT_AND_IMAGE = 2,
};

constexpr int64_t GR00T_DIT_SC_LINEAR_SITE = 275862543721218402LL;
constexpr int64_t GR00T_DIT_COND_DMA_SITE = 830444116670174423LL;
constexpr int64_t GR00T_DIT_VL_DMA_SITE = 4588528469086136179LL;
constexpr int64_t GR00T_DIT_SELF_Q_LINEAR_SITE = 7449164786571286334LL;
constexpr int64_t GR00T_DIT_SELF_K_LINEAR_SITE = 5242062524892150705LL;
constexpr int64_t GR00T_DIT_SELF_V_LINEAR_SITE = 4911590865261992265LL;
constexpr int64_t GR00T_DIT_SELF_KV_INSERT_SITE = 943879439154617039LL;
constexpr int64_t GR00T_DIT_SELF_RAW_ATTENTION_SITE = 7562350367563329289LL;
constexpr int64_t GR00T_DIT_SELF_DDR_ATTENTION_SITE = 7949922424679309799LL;
constexpr int64_t GR00T_DIT_CROSS_Q_LINEAR_SITE = 5217244039428376528LL;
constexpr int64_t GR00T_DIT_CROSS_K_LINEAR_SITE = 5905192764435173745LL;
constexpr int64_t GR00T_DIT_CROSS_V_LINEAR_SITE = 3966455849160875409LL;
constexpr int64_t GR00T_DIT_CROSS_KV_INSERT_SITE = 803980681430785592LL;
constexpr int64_t GR00T_DIT_CROSS_RAW_ATTENTION_SITE = 744437513311785286LL;
constexpr int64_t GR00T_DIT_CROSS_DDR_ATTENTION_SITE = 4477641737292138302LL;
constexpr int64_t GR00T_DIT_O_LINEAR_SITE = 2494940213169617053LL;
constexpr int64_t GR00T_DIT_ATTN_ALL_REDUCE_SITE = 5952600852836303488LL;
constexpr int64_t GR00T_DIT_FF1_LINEAR_SITE = 3924273869232540867LL;
constexpr int64_t GR00T_DIT_FF2_LINEAR_SITE = 8734709123000487057LL;
constexpr int64_t GR00T_DIT_FF_ALL_REDUCE_SITE = 3949863280593130248LL;
constexpr int64_t GR00T_DIT_ACTION_DMA_SITE = 1249920613318509402LL;
constexpr int64_t GR00T_DIT_STATE_DMA_SITE = 4514082781221755405LL;
constexpr int64_t GR00T_DIT_OUTPUT_DMA_SITE = 5092355175424429012LL;
constexpr int64_t GR00T_DIT_ADALN_LINEAR_SITE = 1621027378373485012LL;
constexpr int64_t GR00T_DIT_ADALN_ALL_REDUCE_SITE = 3384661119353053555LL;
constexpr int64_t GR00T_DIT_DENOISE_TABLE_PRELOAD_SITE =
    1729441700529011334LL;
constexpr int64_t GR00T_DIT_DENOISE_BIAS_PRELOAD_SITE =
    941780559855581702LL;

constexpr uint32_t GR00T_DIT_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 | KV_INSERT_CAP_PAD16;
// Both self and cross raw routes consume freshly-produced SPM K/V, but retain
// the existing DDR cache write as a durable fallback/cache ABI mirror.
constexpr int64_t GR00T_DIT_KV_FLAG_DDR_MIRROR = 1;
constexpr int64_t GR00T_DIT_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED = 1LL << 0;

enum class Gr00tDiTMutableDmaRoute : int64_t {
    DDR_SCATTER_TO_SPM = 1,
    DDR_BROADCAST_TO_SPM = 2,
    SPM_TO_DDR = 3,
};

}  // namespace

// =============================================================================
// Gr00tDiTModel — v3::FusedModelBase subclass (GR00T flow-matching DiT head)
// =============================================================================
class Gr00tDiTModel : public FusedModelBase {
public:
    // Per-DiT-block weights. All fp16, swizzled on the Python side. For cross blocks
    // to_k/to_v are [hidden, cross_dim=2048] (project vl_embeds); self blocks [hidden,hidden].
    struct LayerWeights {
        at::Tensor to_q_w, to_k_w, to_v_w, to_out_w;   // attention projections
        at::Tensor to_q_b, to_k_b, to_v_b, to_out_b;   // attention biases (all present)
        at::Tensor ff1_w, ff2_w;                       // FFN 1536->6144->1536
        at::Tensor ff1_b, ff2_b;                       // FFN biases
        at::Tensor adaln_w, adaln_b;                   // norm1 modulation dense [2*hidden, hidden]
        bool       is_cross = false;                   // cross-attn (vl_embeds K/V) vs self-attn
        // W8A16: per-output-channel fp16 scales for the 6 quantized GEMMs (undefined = fp16 path).
        at::Tensor to_q_ws, to_k_ws, to_v_ws, to_out_ws, ff1_ws, ff2_ws;
        at::Tensor adaln_ws;                           // AdaLN modulation GEMV scale (M=1 row, DMA-bound)
    };

    Gr00tDiTModel() = default;

    // ------------------------------------------------------------------ //
    // set_weights — per-layer DiT block weights + per-layer block type + global dims.
    // ------------------------------------------------------------------ //
    void set_weights(
        at::TensorList to_q_list, at::TensorList to_k_list,
        at::TensorList to_v_list, at::TensorList to_out_list,
        at::TensorList ff1_list, at::TensorList ff2_list,
        at::TensorList adaln_w_list, at::TensorList adaln_b_list,
        at::TensorList to_q_b_list, at::TensorList to_k_b_list,
        at::TensorList to_v_b_list, at::TensorList to_out_b_list,
        at::TensorList ff1_b_list, at::TensorList ff2_b_list,
        at::IntArrayRef block_is_cross,
        int64_t num_heads, int64_t head_dim,
        int64_t hidden_size, int64_t ff_inter, int64_t cross_dim,
        double eps,
        at::TensorList to_q_scale_list = {}, at::TensorList to_k_scale_list = {},
        at::TensorList to_v_scale_list = {}, at::TensorList to_out_scale_list = {},
        at::TensorList ff1_scale_list = {}, at::TensorList ff2_scale_list = {},
        at::TensorList adaln_scale_list = {})
    {
        int64_t N = static_cast<int64_t>(to_q_list.size());
        TORCH_CHECK(N > 0, "gr00t_dit_set_weights: empty weight lists");
        TORCH_CHECK(num_heads > 0 && head_dim > 0 && hidden_size > 0 && ff_inter > 0 && cross_dim > 0,
                    "gr00t_dit_set_weights: dim params must be positive");
        TORCH_CHECK(num_heads * head_dim == hidden_size,
                    "gr00t_dit_set_weights: num_heads*head_dim (", num_heads * head_dim,
                    ") != hidden_size (", hidden_size, ")");
        TORCH_CHECK(static_cast<int64_t>(block_is_cross.size()) == N,
                    "gr00t_dit_set_weights: block_is_cross.size()=", block_is_cross.size(),
                    " != num_layers=", N);

        auto check_list = [&](const at::TensorList& l, const char* n, int64_t rank) {
            TORCH_CHECK(static_cast<int64_t>(l.size()) == N,
                        "gr00t_dit_set_weights: ", n, ".size()=", l.size(),
                        " != num_layers=", N);
            for (int64_t i = 0; i < N; ++i) {
                TORCH_CHECK(l[i].defined() && l[i].dim() == rank
                            && l[i].device().type() == at::kPrivateUse1
                            && l[i].is_contiguous(),
                            "gr00t_dit_set_weights: ", n, "[", i,
                            "] must be ", rank, "D contiguous RPU tensor");
            }
        };
        check_list(to_k_list,   "to_k",   2);
        check_list(to_v_list,   "to_v",   2);
        check_list(to_out_list, "to_out", 2);
        check_list(ff1_list,    "ff1",    2);
        check_list(ff2_list,    "ff2",    2);
        check_list(adaln_w_list,"adaln_w",2);
        check_list(adaln_b_list,"adaln_b",1);
        check_list(to_q_list,   "to_q",   2);
        check_list(to_q_b_list,   "to_q_b",   1);
        check_list(to_k_b_list,   "to_k_b",   1);
        check_list(to_v_b_list,   "to_v_b",   1);
        check_list(to_out_b_list, "to_out_b", 1);
        check_list(ff1_b_list,    "ff1_b",    1);
        check_list(ff2_b_list,    "ff2_b",    1);

        // DiT: no GQA (num_kv_heads == num_q_heads), no separate intermediate for
        // the attention path; reuse the base param slots (intermediate_size = ff_inter).
        set_model_params(num_heads, num_heads, head_dim, hidden_size, ff_inter);
        set_num_layers(N);
        eps_ = eps;
        cross_dim_ = cross_dim;

        layer_weights_.clear();
        layer_weights_.reserve(N);
        for (int64_t i = 0; i < N; ++i) {
            layer_weights_.push_back({
                to_q_list[i], to_k_list[i], to_v_list[i], to_out_list[i],
                to_q_b_list[i], to_k_b_list[i], to_v_b_list[i], to_out_b_list[i],
                ff1_list[i], ff2_list[i],
                ff1_b_list[i], ff2_b_list[i],
                adaln_w_list[i], adaln_b_list[i],
                block_is_cross[i] != 0,
            });
        }

        // W8A16: attach per-output-channel scales for the 6 quantized GEMMs (q/k/v/out/ff1/ff2).
        // When the scale lists are empty the GEMMs stay fp16 (the rpu_linear kernel keys on the
        // weight dtype + a defined scale). The per-embodiment encoders / AdaLN GEMV / glue stay fp16.
        {
            const char* c = "gr00t_dit_set_weights";
            check_gr00t_w8a16_scale_list(N, to_q_list,   to_q_scale_list,   c, "to_q");
            check_gr00t_w8a16_scale_list(N, to_k_list,   to_k_scale_list,   c, "to_k");
            check_gr00t_w8a16_scale_list(N, to_v_list,   to_v_scale_list,   c, "to_v");
            check_gr00t_w8a16_scale_list(N, to_out_list, to_out_scale_list, c, "to_out");
            check_gr00t_w8a16_scale_list(N, ff1_list,    ff1_scale_list,    c, "ff1");
            check_gr00t_w8a16_scale_list(N, ff2_list,    ff2_scale_list,    c, "ff2");
            check_gr00t_w8a16_scale_list(N, adaln_w_list, adaln_scale_list, c, "adaln_w");
        }
        if (!to_q_scale_list.empty()) {
            for (int64_t i = 0; i < N; ++i) {
                layer_weights_[i].to_q_ws   = to_q_scale_list[i];
                layer_weights_[i].to_k_ws   = to_k_scale_list[i];
                layer_weights_[i].to_v_ws   = to_v_scale_list[i];
                layer_weights_[i].to_out_ws = to_out_scale_list[i];
                layer_weights_[i].ff1_ws    = ff1_scale_list[i];
                layer_weights_[i].ff2_ws    = ff2_scale_list[i];
            }
        }
        // AdaLN GEMV scale (separate list — may be int8'd independently of the 6 block GEMMs).
        if (!adaln_scale_list.empty()) {
            for (int64_t i = 0; i < N; ++i) layer_weights_[i].adaln_ws = adaln_scale_list[i];
        }

        // norm3 = LayerNorm(affine=False): gamma=1, beta=0 as [hidden] SPM vectors.
        auto dev = to_q_list[0].device();
        ln_ones_  = at::ones ({hidden_size}, at::TensorOptions().dtype(at::kHalf)).to(dev);
        ln_zeros_ = at::zeros({hidden_size}, at::TensorOptions().dtype(at::kHalf)).to(dev);

        invalidate_model_state();  // D-503: last non-empty statement of set_weights
    }

    // ------------------------------------------------------------------ //
    // set_denoise_weights — Phase B glue (state/action encoders, norm_out, decoder)
    // + baked cond/tau/pos tables. Folds runtime.py's per-step host-fp32 glue on-device.
    // All weights single-core col-swizzled ([out,in], TP=1) at emb=20; padded 132→144 where
    // noted. Call AFTER set_weights (32 DiT blocks). See build_denoise_step.py for prep/design.
    // ------------------------------------------------------------------ //
    void set_denoise_weights(
        const at::Tensor& se1_w, const at::Tensor& se1_b,      // state_encoder layer1 (K-pad 144)
        const at::Tensor& se2_w, const at::Tensor& se2_b,      // state_encoder layer2
        const at::Tensor& ae1_w, const at::Tensor& ae1_b,      // action_encoder W1 (K-pad 144)
        const at::Tensor& ae2a_w, const at::Tensor& ae2t_w,    // action_encoder W2 (a-part / tau-part)
        const at::Tensor& ae2_b,
        const at::Tensor& ae3_w, const at::Tensor& ae3_b,      // action_encoder W3
        const at::Tensor& dec1_w, const at::Tensor& dec1_b,    // action_decoder layer1
        const at::Tensor& dec2_w, const at::Tensor& dec2_b,    // action_decoder layer2 (N-pad 144)
        const at::Tensor& po1_w, const at::Tensor& po1_b,      // proj_out_1 (norm_out shift|scale)
        const at::Tensor& po2_w, const at::Tensor& po2_b,      // proj_out_2
        const at::Tensor& cond_table, const at::Tensor& tau_table, const at::Tensor& pos_table,
        int64_t action_dim, int64_t action_dim_pad, int64_t num_steps, double dt,
        const at::Tensor& se2_scale = {}, const at::Tensor& ae2a_scale = {},
        const at::Tensor& ae2t_scale = {}, const at::Tensor& ae3_scale = {},
        const at::Tensor& dec1_scale = {}, const at::Tensor& dec2_scale = {},
        const at::Tensor& po1_scale = {}, const at::Tensor& po2_scale = {})
    {
        TORCH_CHECK(num_layers() > 0, "set_denoise_weights must follow set_weights (32 DiT blocks)");
        TORCH_CHECK(action_dim_pad % 16 == 0, "action_dim_pad must be %16");
        TORCH_CHECK(cond_table.size(0) == num_steps && cond_table.size(1) == hidden_size(),
                    "cond_table must be [num_steps, hidden]");
        TORCH_CHECK(tau_table.size(0) == num_steps && tau_table.size(-1) == hidden_size(),
                    "tau_table must be [num_steps, AH, hidden]");
        se1_w_ = se1_w; se1_b_ = se1_b; se2_w_ = se2_w; se2_b_ = se2_b;
        ae1_w_ = ae1_w; ae1_b_ = ae1_b; ae2a_w_ = ae2a_w; ae2t_w_ = ae2t_w; ae2_b_ = ae2_b;
        ae3_w_ = ae3_w; ae3_b_ = ae3_b;
        dec1_w_ = dec1_w; dec1_b_ = dec1_b; dec2_w_ = dec2_w; dec2_b_ = dec2_b;
        po1_w_ = po1_w; po1_b_ = po1_b; po2_w_ = po2_w; po2_b_ = po2_b;
        se2_ws_ = rpu_retain_linear_quant_scale(se2_w, se2_scale);
        ae2a_ws_ = rpu_retain_linear_quant_scale(ae2a_w, ae2a_scale);
        ae2t_ws_ = rpu_retain_linear_quant_scale(ae2t_w, ae2t_scale);
        ae3_ws_ = rpu_retain_linear_quant_scale(ae3_w, ae3_scale);
        dec1_ws_ = rpu_retain_linear_quant_scale(dec1_w, dec1_scale);
        dec2_ws_ = rpu_retain_linear_quant_scale(dec2_w, dec2_scale);
        po1_ws_ = rpu_retain_linear_quant_scale(po1_w, po1_scale);
        po2_ws_ = rpu_retain_linear_quant_scale(po2_w, po2_scale);
        cond_table_t_ = cond_table; tau_table_ = tau_table; pos_table_ = pos_table;
        action_dim_ = action_dim; action_dim_pad_ = action_dim_pad;
        num_steps_ = num_steps; dt_ = c10::Half(static_cast<float>(dt));
        action_horizon_ = tau_table.size(1);                   // 40
        // Derive inner dims from the un-swizzled bias sizes (robust to arch changes).
        enc_mid_       = se1_b.size(0);                         // state_encoder layer1 out (1024)
        dec_mid_       = dec1_b.size(0);                        // action_decoder layer1 out (1024)
        model_out_dim_ = po2_b.size(0);                        // proj_out_2 out (1024)
        denoise_ready_ = true;
        invalidate_model_state();  // D-503
    }

    void set_cache_envelope(int64_t max_kv_len, int64_t chunk) {
        TORCH_CHECK(num_layers() > 0 && max_kv_len > 0,
                    "gr00t_dit_set_chunk_envelope: installed DiT weights and positive cache required");
        set_chunk_envelope(std::max(max_kv_len, denoise_ready_ ? action_horizon_ + 1 : 0), chunk);
    }

    // Single-core SPM Linear helper (col-partition; bias from SPM addr).
    // M=1 and larger shapes use the shared generator-selected auto-tile family.
    void sc_lin(uint32_t in_a, const at::Tensor& w, uint32_t out_a,
                int64_t M, int64_t N, int64_t K, uint32_t bias_a = 0,
                const at::Tensor& scale = {}) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_SC_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in_a, w, out_a, M, N, K, /*partition=*/1, /*num_cores=*/1,
            bias_a, scale);
    }

    // ------------------------------------------------------------------ //
    // forward — drive the DiT once (one denoise step). sa_embs [B,T,hidden]; cond
    // [hidden] = SiLU(temb) host-side; vl_embeds [B,S,cross_dim] (cross K/V source,
    // constant across steps); cross_mask [T,S] fp16 additive (0 attend / -inf mask).
    // ------------------------------------------------------------------ //
    at::Tensor forward(
        const at::Tensor& sa_embs,
        const at::Tensor& cond,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const std::optional<at::Tensor>& attention_mask,
        const std::optional<at::Tensor>& vl_embeds,
        const std::optional<at::Tensor>& cross_mask,
        const std::optional<at::Tensor>& cross_mask_img,
        at::IntArrayRef planned_stage_descriptor = {})
    {
        RECORD_FUNCTION("gr00t_dit_forward", {});
        TORCH_CHECK(!z2_bound_,
                    "gr00t_dit_forward: Z2 lease is active; use the Z2 unroll op");
        TORCH_CHECK(cond.defined() && cond.dim() == 1
                    && cond.size(0) == hidden_size()
                    && cond.device().type() == at::kPrivateUse1
                    && cond.is_contiguous(),
                    "gr00t_dit_forward: cond must be 1-D [hidden_size] contiguous RPU tensor");

        // Per-forward cond state (mirrors AdaRMS): scatter re-emits on each BUILD; the
        // mutable-DMA src is cursor-patched per REPLAY from cond_src_base_.
        cond_ref_                 = cond;
        cond_src_base_            = ::rhino_lkn::RpuGetDevAddr(cond.data_ptr());
        cond_loaded_this_forward_ = false;

        // Per-forward vl_embeds state (cross-attn K/V source; broadcast to SPM once,
        // projected per cross-block by its own to_k/to_v).
        const int64_t seq_q = sa_embs.size(-2);
        if (vl_embeds.has_value() && vl_embeds->defined()) {
            const at::Tensor& vl = vl_embeds.value();
            TORCH_CHECK(vl.size(-1) == cross_dim_ && vl.device().type() == at::kPrivateUse1
                        && vl.is_contiguous(),
                        "gr00t_dit_forward: vl_embeds must be [..,S,cross_dim] contiguous RPU tensor");
            vl_ref_      = vl;
            vl_seq_      = vl.size(-2);
            vl_src_base_ = ::rhino_lkn::RpuGetDevAddr(vl.data_ptr());
            vl_loaded_this_forward_ = false;
            TORCH_CHECK(cross_mask.has_value() && cross_mask->defined(),
                        "gr00t_dit_forward: cross_mask required when vl_embeds is given");
            // MASK_2D: host [seq_q, vl_seq_] fp16 (0 attend / -inf mask). cross_mask = TEXT
            // subset (cross blocks L%4==0); cross_mask_img = IMAGE subset (L%4!=0).
            // Keep distinct ordinals for text/image: equal shape does not
            // imply equal visibility, and both addresses are retained by Graph.
            prepared_cross_mask_ = prep_mask_into(cross_mask, seq_q, text_mask_slot_, /*ordinal=*/1);
            if (cross_mask_img.has_value() && cross_mask_img->defined())
                prepared_cross_mask_img_ = prep_mask_into(cross_mask_img, seq_q, image_mask_slot_, /*ordinal=*/2);
            else
                prepared_cross_mask_img_ = prepared_cross_mask_;
        } else {
            vl_seq_ = 0;
        }

        if (denoise_mode_ || loop_mode_) {
            denoise_mode_ = false;
            loop_mode_ = false;
            invalidate_model_state(/*planning_domain_changed=*/false);
        }
        return run_all_layers(sa_embs, k_caches, v_caches, attention_mask,
                              /*position=*/0, /*is_causal=*/false,
                              /*planned_chunk_size=*/0, planned_stage_descriptor);
    }

    // ------------------------------------------------------------------ //
    // step_forward — Phase B fused denoise step. Folds the per-step glue on-device:
    //   actions/state -> [pre] encoders -> sa_embs(emb_stage_) -> 32 DiT blocks ->
    //   [post] norm_out -> decoder -> in-graph Euler -> actions_next [1,40,144].
    // step_idx selects baked cond/tau. Caller crops the [.. ,:132] public output.
    // ------------------------------------------------------------------ //
    at::Tensor step_forward(
        const at::Tensor& actions_pad,                   // [1,AH,action_dim_pad] fp16 (zero-pad)
        const at::Tensor& state_pad,                     // [1,1,action_dim_pad] fp16 (zero-pad)
        int64_t step_idx,
        std::vector<at::Tensor>& k_caches, std::vector<at::Tensor>& v_caches,
        const at::Tensor& vl_embeds,
        const at::Tensor& tmask, const at::Tensor& imask,
        at::IntArrayRef planned_stage_descriptor = {})
    {
        RECORD_FUNCTION("gr00t_denoise_step_forward", {});
        TORCH_CHECK(!z2_bound_,
                    "gr00t_denoise_step_forward: Z2 lease is active; use the Z2 unroll op");
        TORCH_CHECK(denoise_ready_, "step_forward before set_denoise_weights");
        TORCH_CHECK(actions_pad.dim() == 3 && actions_pad.size(1) == action_horizon_
                    && actions_pad.size(2) == action_dim_pad_ && actions_pad.scalar_type() == at::kHalf
                    && actions_pad.is_contiguous() && actions_pad.device().type() == at::kPrivateUse1,
                    "actions_pad must be [1,AH,action_dim_pad] fp16 contig RPU");
        TORCH_CHECK(step_idx >= 0 && step_idx < num_steps_, "step_idx out of range");
        // Enter single-step denoise mode (rebuild if switching from plain/unroll path).
        if (!denoise_mode_ || loop_mode_) { denoise_mode_ = true; loop_mode_ = false; invalidate_model_state(/*planning_domain_changed=*/false); }
        step_idx_ = step_idx;

        // Per-forward mutable-DMA bases (held as named refs across the synchronous forward).
        actions_ref_ = actions_pad; actions_src_base_ = ::rhino_lkn::RpuGetDevAddr(actions_pad.data_ptr());
        state_ref_   = state_pad;   state_src_base_   = ::rhino_lkn::RpuGetDevAddr(state_pad.data_ptr());

        if (!emb_stage_.defined() || emb_stage_.size(1) != action_horizon_ + 1)
            emb_stage_ = at::empty({1, action_horizon_ + 1, hidden_size()},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        at::Tensor out = allocate_tracked_output({1, action_horizon_, action_dim_pad_});
        x_out_ref_ = out;
        x_out_dst_base_ = ::rhino_lkn::RpuGetDevAddr(out.data_ptr());

        // vl_embeds + text/image MASK_2D — same machinery as forward() (cross-attn K/V).
        cond_loaded_this_forward_ = false;            // pre-hook scatters cond from the table
        const int64_t seq_q = action_horizon_ + 1;    // 41 (state + AH actions)
        TORCH_CHECK(vl_embeds.size(-1) == cross_dim_ && vl_embeds.device().type() == at::kPrivateUse1
                    && vl_embeds.is_contiguous(), "vl_embeds must be [..,S,cross_dim] contig RPU");
        vl_ref_ = vl_embeds; vl_seq_ = vl_embeds.size(-2);
        vl_src_base_ = ::rhino_lkn::RpuGetDevAddr(vl_embeds.data_ptr());
        vl_loaded_this_forward_ = false;
        prepared_cross_mask_     = prep_mask_into(tmask, seq_q, text_mask_slot_, /*ordinal=*/1);
        prepared_cross_mask_img_ = prep_mask_into(imask, seq_q, image_mask_slot_, /*ordinal=*/2);

        run_all_layers(emb_stage_, k_caches, v_caches, std::nullopt,
                       /*position=*/0, /*is_causal=*/false,
                       /*planned_chunk_size=*/0, planned_stage_descriptor);
        rpu_ddr_flush(out.data_ptr<c10::Half>());
        return out;                                   // [1,AH,action_dim_pad]; caller crops :action_dim
    }

    // ------------------------------------------------------------------ //
    // unroll_forward — Phase C: run the full num_steps Euler denoise in ONE graph
    // (body_iterations = num_steps_). pre/post hooks fire per iteration with ctx().body_iter
    // advancing 0..num_steps-1: iter 0 loads noise + state_features; each iter selects
    // cond[iter]/tau[iter], runs encoder→32 blocks→norm_out→decoder→in-graph Euler. x_t_spm
    // persists across iters (the post-hook writes x_out every iter → final = last iter's x_t).
    // ------------------------------------------------------------------ //
    at::Tensor unroll_forward(
        const at::Tensor& noise_pad,                     // [1,AH,action_dim_pad] fp16 (zero-pad)
        const at::Tensor& state_pad,                     // [1,1,action_dim_pad] fp16 (zero-pad)
        std::vector<at::Tensor>& k_caches, std::vector<at::Tensor>& v_caches,
        const at::Tensor& vl_embeds, const at::Tensor& tmask,
        const at::Tensor& imask,
        at::IntArrayRef planned_stage_descriptor = {})
    {
        RECORD_FUNCTION("gr00t_denoise_unroll_forward", {});
        TORCH_CHECK(!z2_bound_,
                    "gr00t_denoise_unroll_forward: Z2 lease is active; use the Z2 unroll op");
        TORCH_CHECK(denoise_ready_, "unroll_forward before set_denoise_weights");
        TORCH_CHECK(noise_pad.dim() == 3 && noise_pad.size(1) == action_horizon_
                    && noise_pad.size(2) == action_dim_pad_ && noise_pad.scalar_type() == at::kHalf
                    && noise_pad.is_contiguous() && noise_pad.device().type() == at::kPrivateUse1,
                    "noise_pad must be [1,AH,action_dim_pad] fp16 contig RPU");
        // Node-count backstop (wall_oss precedent): num_steps bodies must stay well below the
        // 32768 batch-item cap. GR00T body ≈ 32 blocks + glue ≈ ~360 nodes → 4×≈1.4k. Loud-fail margin.
        TORCH_CHECK(num_steps_ * 2000 < 32000, "unroll num_steps too large for the kernel-batch cap");
        // Enter unroll mode (rebuild on transition from plain / step form).
        if (!denoise_mode_ || !loop_mode_) { denoise_mode_ = true; loop_mode_ = true; invalidate_model_state(/*planning_domain_changed=*/false); }

        actions_ref_ = noise_pad; actions_src_base_ = ::rhino_lkn::RpuGetDevAddr(noise_pad.data_ptr());
        state_ref_   = state_pad; state_src_base_   = ::rhino_lkn::RpuGetDevAddr(state_pad.data_ptr());
        if (!emb_stage_.defined() || emb_stage_.size(1) != action_horizon_ + 1)
            emb_stage_ = at::empty({1, action_horizon_ + 1, hidden_size()},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        at::Tensor out = allocate_tracked_output({1, action_horizon_, action_dim_pad_});
        x_out_ref_ = out;
        x_out_dst_base_ = ::rhino_lkn::RpuGetDevAddr(out.data_ptr());

        cond_loaded_this_forward_ = false;
        const int64_t seq_q = action_horizon_ + 1;
        TORCH_CHECK(vl_embeds.size(-1) == cross_dim_ && vl_embeds.device().type() == at::kPrivateUse1
                    && vl_embeds.is_contiguous(), "vl_embeds must be [..,S,cross_dim] contig RPU");
        vl_ref_ = vl_embeds; vl_seq_ = vl_embeds.size(-2);
        vl_src_base_ = ::rhino_lkn::RpuGetDevAddr(vl_embeds.data_ptr());
        vl_loaded_this_forward_ = false;
        prepared_cross_mask_     = prep_mask_into(tmask, seq_q, text_mask_slot_, /*ordinal=*/1);
        prepared_cross_mask_img_ = prep_mask_into(imask, seq_q, image_mask_slot_, /*ordinal=*/2);

        run_all_layers(
            emb_stage_, k_caches, v_caches, std::nullopt,
            /*position=*/0, /*is_causal=*/false,
            /*planned_chunk_size=*/0, planned_stage_descriptor);
        rpu_ddr_flush(out.data_ptr<c10::Half>());
        return out;                                   // final x_t after num_steps Euler steps
    }

    std::vector<int64_t> resolve_denoise_stage_domain(
        int64_t action_rows, int64_t cross_seq_len, int64_t step_idx) {
        TORCH_CHECK(denoise_ready_,
                    "gr00t_denoise_resolve_stage_domain: set_denoise_weights "
                    "must precede planning");
        TORCH_CHECK(!z2_bound_ && z2_prepared_seq_len_ == 0,
                    "gr00t_denoise_resolve_stage_domain: Z2 layout is "
                    "prepared or active");
        TORCH_CHECK(action_rows == action_horizon_ + 1,
                    "gr00t_denoise_resolve_stage_domain: action_rows must "
                    "match the fixed state+action geometry");
        TORCH_CHECK(cross_seq_len > 0,
                    "gr00t_denoise_resolve_stage_domain: cross_seq_len must "
                    "be positive");
        TORCH_CHECK(step_idx >= -1 && step_idx < num_steps_,
                    "gr00t_denoise_resolve_stage_domain: step_idx out of range");
        return resolve_dit_stage_domain(action_rows, cross_seq_len,
                                        std::nullopt, true, step_idx);
    }

    std::vector<int64_t> resolve_dit_stage_domain(
        int64_t rows, int64_t cross_seq_len,
        const std::optional<at::Tensor>& attention_mask,
        bool denoise = false, int64_t step_idx = -1) {
        RpuExecutionCleanupGuard planning("gr00t_dit_resolve_stage_domain");
        TORCH_CHECK(!z2_bound_ && z2_prepared_seq_len_ == 0,
                    "gr00t_dit_resolve_stage_domain: Z2 layout is prepared or active");
        TORCH_CHECK(cross_seq_len >= 0,
                    "gr00t_dit_resolve_stage_domain: negative cross KV rows");
        const bool saved_denoise_mode = denoise_mode_;
        const bool saved_loop_mode = loop_mode_;
        const int64_t saved_vl_seq = vl_seq_;
        const int64_t saved_step = step_idx_;
        auto restore = c10::make_scope_exit([&] {
            denoise_mode_ = saved_denoise_mode;
            loop_mode_ = saved_loop_mode;
            vl_seq_ = saved_vl_seq;
            step_idx_ = saved_step;
        });
        denoise_mode_ = denoise;
        loop_mode_ = denoise && step_idx == -1;
        step_idx_ = step_idx < 0 ? 0 : step_idx;
        vl_seq_ = cross_seq_len;
        return encode_fmb_prefill_stage_domain(
            resolve_prefill_stage_domain_for_shape(
                rows, /*position=*/0, attention_mask,
                /*is_causal=*/false, /*requested_chunk_size=*/0,
                /*logical_len=*/0, /*planning_chunk_size_override=*/0));
    }

    SpmPipelineComponentLayout prepare_z2_layout(int64_t seq_len) {
        TORCH_CHECK(denoise_ready_,
                    "gr00t denoise Z2: set_denoise_weights must precede prepare");
        TORCH_CHECK(seq_len > 0,
                    "gr00t denoise Z2: seq_len must be positive");
        TORCH_CHECK(!z2_bound_,
                    "gr00t denoise Z2: cannot prepare while a lease is active");
        if (!denoise_mode_ || !loop_mode_) {
            denoise_mode_ = true;
            loop_mode_ = true;
            invalidate_model_state();
        }
        vl_seq_ = seq_len;
        LayoutContext layout;
        layout.chunk_size = action_horizon_ + 1;
        layout.max_kv_seq_len = action_horizon_ + 1;
        layout.num_layers = num_layers();
        layout.use_attn_mask = false;
        layout.is_causal = false;
        auto prepared = prepare_spm_pipeline_component(
            layout, compose_fmb_default_three_stage_chunk_plan(
                        layout, action_horizon_ + 1, /*position=*/0,
                        ChunkMode::SEQUENTIAL));
        z2_prepared_seq_len_ = seq_len;
        return prepared;
    }

    SpmDense2DSpec z2_port_spec(int64_t seq_len) const {
        TORCH_CHECK(seq_len > 0,
                    "gr00t denoise Z2: port seq_len must be positive");
        SpmDense2DSpec spec;
        spec.rows = seq_len;
        spec.cols = cross_dim_;
        spec.validate();
        return spec;
    }

    void prepare_z2_masks(const at::Tensor& tmask,
                          const at::Tensor& imask,
                          uint64_t plan_hash) {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "gr00t denoise Z2 mask preparation must run outside Graph");
        TORCH_CHECK(!z2_bound_ && z2_prepared_seq_len_ > 0,
                    "gr00t denoise Z2 masks require a prepared inactive layout");
        TORCH_CHECK(z2_masks_consumed_generation_ != nullptr &&
                        (!z2_masks_prepared_ ||
                         z2_masks_prepared_generation_ <=
                             *z2_masks_consumed_generation_),
                    "gr00t denoise Z2 refuses to overwrite an unconsumed "
                    "prepared mask generation");
        // Invalidate any previously prepared token before either dedicated
        // slot is modified.  If the second copy fails, no old generation can
        // authorize a mixed new-text/old-image pair.
        z2_masks_prepared_ = false;
        z2_masks_plan_hash_ = 0;
        const int64_t seq_q = action_horizon_ + 1;
        vl_seq_ = z2_prepared_seq_len_;
        PreparedMask prepared_text =
            prep_mask_into(tmask, seq_q, text_mask_slot_, /*ordinal=*/1);
        PreparedMask prepared_image =
            prep_mask_into(imask, seq_q, image_mask_slot_, /*ordinal=*/2);
        TORCH_CHECK(z2_mask_generation_ != nullptr &&
                        *z2_mask_generation_ !=
                            std::numeric_limits<uint64_t>::max(),
                    "gr00t denoise Z2 mask generation exhausted");
        prepared_cross_mask_ = std::move(prepared_text);
        prepared_cross_mask_img_ = std::move(prepared_image);
        ++*z2_mask_generation_;
        z2_masks_prepared_generation_ = *z2_mask_generation_;
        z2_masks_plan_hash_ = plan_hash;
        z2_masks_prepared_ = true;
    }

    void adopt_z2_layout(const SpmPipelineLease& lease,
                         const SpmTensorView& scratch) {
        TORCH_CHECK(z2_prepared_seq_len_ > 0,
                    "gr00t denoise Z2: prepare must precede begin");
        TORCH_CHECK(!z2_bound_,
                    "gr00t denoise Z2: binding is already active");
        adopt_spm_pipeline_component(lease, scratch);
    }

    void bind_z2_port(const SpmPipelineLease& lease,
                      const SpmPortView& port) {
        const auto expected = z2_port_spec(z2_prepared_seq_len_);
        TORCH_CHECK(port.spec() == expected,
                    "gr00t denoise Z2: typed port spec mismatch");
        z2_port_addr_ = port.resolve_physical_addr(/*core=*/0, lease);
        z2_epoch_ = lease.epoch();
        z2_plan_hash_ = lease.plan_hash();
        z2_bound_ = true;
    }

    void validate_z2_layout(const SpmPipelineLease& lease) const {
        TORCH_CHECK(z2_bound_, "gr00t denoise Z2: no active binding");
        TORCH_CHECK(z2_epoch_ == lease.epoch() &&
                        z2_plan_hash_ == lease.plan_hash(),
                    "gr00t denoise Z2: stale lease binding");
        validate_spm_pipeline_component(lease);
    }

    void clear_z2_layout(uint64_t epoch, uint64_t plan_hash) {
        if (z2_bound_) {
            TORCH_CHECK(z2_epoch_ == epoch && z2_plan_hash_ == plan_hash,
                        "gr00t denoise Z2: stale clear token");
        }
        release_spm_pipeline_component(epoch, plan_hash);
        z2_bound_ = false;
        z2_port_addr_ = 0;
        z2_epoch_ = 0;
        z2_plan_hash_ = 0;
        z2_masks_prepared_ = false;
        z2_masks_plan_hash_ = 0;
    }

    void unprepare_z2_layout(int64_t seq_len) {
        TORCH_CHECK(!RpuKernelGraph::has_active(),
                    "gr00t denoise Z2 unprepare must run outside Graph capture");
        TORCH_CHECK(!z2_bound_ && z2_epoch_ == 0 &&
                        z2_plan_hash_ == 0 && z2_port_addr_ == 0,
                    "gr00t denoise Z2 unprepare requires no active component lease");
        TORCH_CHECK(z2_prepared_seq_len_ == 0 ||
                        z2_prepared_seq_len_ == seq_len,
                    "gr00t denoise Z2 unprepare received a stale sequence identity");

        // Retire only the Z2 authority.  The FMB cached layout, persistent SPM
        // allocations, model weights, and stable mask slots remain reusable.
        z2_prepared_seq_len_ = 0;
        z2_masks_prepared_ = false;
        z2_masks_plan_hash_ = 0;
        z2_masks_prepared_generation_ = 0;
        prepared_cross_mask_ = PreparedMask{};
        prepared_cross_mask_img_ = PreparedMask{};
        TORCH_INTERNAL_ASSERT(z2_mask_generation_ != nullptr &&
                              z2_masks_consumed_generation_ != nullptr);
        *z2_masks_consumed_generation_ = *z2_mask_generation_;
    }

    void check_z2_destroy_allowed() const {
        TORCH_CHECK(!z2_bound_ && z2_prepared_seq_len_ == 0 &&
                        z2_epoch_ == 0 && z2_plan_hash_ == 0 &&
                        z2_port_addr_ == 0 && !z2_masks_prepared_ &&
                        z2_masks_plan_hash_ == 0 &&
                        z2_masks_prepared_generation_ == 0,
                    "cannot destroy the GR00T denoise handle while its Z2 "
                    "layout or masks are prepared or active; clear the outer "
                    "GraphCache and unprepare Z2 first");
    }

    at::Tensor unroll_forward_z2(
        const at::Tensor& noise_pad,
        const at::Tensor& state_pad,
        std::vector<at::Tensor>& k_caches,
        std::vector<at::Tensor>& v_caches,
        const at::Tensor& tmask,
        const at::Tensor& imask,
        uint64_t epoch,
        uint64_t plan_hash,
        bool use_prepared_masks = false)
    {
        RECORD_FUNCTION("gr00t_denoise_unroll_forward_z2", {});
        TORCH_CHECK(z2_bound_ && z2_epoch_ == epoch &&
                        z2_plan_hash_ == plan_hash,
                    "gr00t_denoise_unroll_forward_z2: stale epoch/plan hash");
        TORCH_CHECK(denoise_ready_ && denoise_mode_ && loop_mode_,
                    "gr00t_denoise_unroll_forward_z2: denoise layout was not prepared");
        TORCH_CHECK(noise_pad.dim() == 3 &&
                        noise_pad.size(0) == 1 &&
                        noise_pad.size(1) == action_horizon_ &&
                        noise_pad.size(2) == action_dim_pad_ &&
                        noise_pad.scalar_type() == at::kHalf &&
                        noise_pad.is_contiguous() &&
                        noise_pad.device().type() == at::kPrivateUse1,
                    "noise_pad must be [1,AH,action_dim_pad] fp16 contig RPU");
        TORCH_CHECK(state_pad.defined() && state_pad.dim() == 3 &&
                        state_pad.size(0) == 1 && state_pad.size(1) == 1 &&
                        state_pad.size(2) == action_dim_pad_ &&
                        state_pad.scalar_type() == at::kHalf &&
                        state_pad.is_contiguous() &&
                        state_pad.device().type() == at::kPrivateUse1,
                    "state_pad must be [1,1,action_dim_pad] fp16 contig RPU");
        TORCH_CHECK(num_steps_ * 2000 < 32000,
                    "unroll num_steps too large for the kernel-batch cap");

        actions_ref_ = noise_pad;
        actions_src_base_ = ::rhino_lkn::RpuGetDevAddr(noise_pad.data_ptr());
        state_ref_ = state_pad;
        state_src_base_ = ::rhino_lkn::RpuGetDevAddr(state_pad.data_ptr());
        if (!emb_stage_.defined() || emb_stage_.size(1) != action_horizon_ + 1) {
            emb_stage_ = at::empty(
                {1, action_horizon_ + 1, hidden_size()},
                at::TensorOptions().dtype(at::kHalf).device(at::kPrivateUse1));
        }
        at::Tensor out =
            allocate_tracked_output({1, action_horizon_, action_dim_pad_});
        x_out_ref_ = out;
        x_out_dst_base_ = ::rhino_lkn::RpuGetDevAddr(out.data_ptr());

        cond_loaded_this_forward_ = false;
        vl_seq_ = z2_prepared_seq_len_;
        vl_loaded_this_forward_ = true;
        if (use_prepared_masks) {
            TORCH_CHECK(z2_masks_prepared_ &&
                            z2_masks_plan_hash_ == plan_hash &&
                            z2_mask_generation_ != nullptr &&
                            *z2_mask_generation_ ==
                                z2_masks_prepared_generation_ &&
                            z2_masks_prepared_generation_ >
                                *z2_masks_consumed_generation_,
                        "gr00t denoise Z2: prepared masks are missing or stale");
        } else {
            const int64_t seq_q = action_horizon_ + 1;
            prepared_cross_mask_ = prep_mask_into(tmask, seq_q, text_mask_slot_, /*ordinal=*/1);
            prepared_cross_mask_img_ =
                prep_mask_into(imask, seq_q, image_mask_slot_, /*ordinal=*/2);
        }

        run_all_layers(emb_stage_, k_caches, v_caches, std::nullopt,
                       /*position=*/0, /*is_causal=*/false);
        rpu_ddr_flush(out.data_ptr<c10::Half>());
        return out;
    }

    void stage_z2_outer_fast_component(
        GraphKernelRegisterCensusGuard& guard,
        at::TensorList k_caches,
        at::TensorList v_caches) const {
        TORCH_CHECK(z2_bound_ && z2_epoch_ != 0 && z2_plan_hash_ != 0,
                    "gr00t denoise Z2 outer-fast component is not bound");
        stage_spm_outer_fast_component(
            guard, k_caches, v_caches);
    }

    void stage_z2_outer_fast_invocation_authority(
        GraphKernelRegisterCensusGuard& guard,
        uint64_t plan_hash) const {
        TORCH_CHECK(z2_masks_prepared_ &&
                        z2_masks_plan_hash_ == plan_hash &&
                        z2_mask_generation_ != nullptr &&
                        *z2_mask_generation_ ==
                            z2_masks_prepared_generation_ &&
                        z2_masks_prepared_generation_ >
                            *z2_masks_consumed_generation_ &&
                        prepared_cross_mask_.mask_type == 4 &&
                        prepared_cross_mask_img_.mask_type == 4 &&
                        prepared_cross_mask_.ddr_tensor.defined() &&
                        prepared_cross_mask_img_.ddr_tensor.defined() &&
                        text_mask_slot_.defined() &&
                        image_mask_slot_.defined() &&
                        prepared_cross_mask_.ddr_tensor
                                .unsafeGetTensorImpl() ==
                            text_mask_slot_.unsafeGetTensorImpl() &&
                        prepared_cross_mask_img_.ddr_tensor
                                .unsafeGetTensorImpl() ==
                            image_mask_slot_.unsafeGetTensorImpl(),
                    "gr00t denoise Z2 mask invocation authority is stale");
        guard.stage_outer_fast_invocation_authority(
            z2_mask_generation_, z2_masks_consumed_generation_,
            z2_masks_prepared_generation_);
        guard.stage_outer_fast_invocation_visibility(
            prepared_cross_mask_.ddr_tensor);
        guard.stage_outer_fast_invocation_visibility(
            prepared_cross_mask_img_.ddr_tensor);
    }

    at::Tensor prepare_z2_outer_fast_replay(
        GraphKernelRegisterCensusGuard& guard,
        const at::Tensor& noise_pad,
        const at::Tensor& state_pad,
        uint64_t epoch,
        uint64_t plan_hash) {
        TORCH_CHECK(z2_bound_ && z2_epoch_ == epoch &&
                        z2_plan_hash_ == plan_hash && denoise_ready_ &&
                        denoise_mode_ && loop_mode_,
                    "gr00t denoise outer-fast replay has stale model/lease "
                    "state");
        TORCH_CHECK(noise_pad.defined() && noise_pad.dim() == 3 &&
                        noise_pad.size(0) == 1 &&
                        noise_pad.size(1) == action_horizon_ &&
                        noise_pad.size(2) == action_dim_pad_ &&
                        noise_pad.scalar_type() == at::kHalf &&
                        noise_pad.device().type() == at::kPrivateUse1 &&
                        noise_pad.is_contiguous(),
                    "gr00t denoise outer-fast noise shape/storage drifted");
        TORCH_CHECK(state_pad.defined() && state_pad.dim() == 3 &&
                        state_pad.size(0) == 1 && state_pad.size(1) == 1 &&
                        state_pad.size(2) == action_dim_pad_ &&
                        state_pad.scalar_type() == at::kHalf &&
                        state_pad.device().type() == at::kPrivateUse1 &&
                        state_pad.is_contiguous(),
                    "gr00t denoise outer-fast state shape/storage drifted");
        TORCH_CHECK(z2_masks_prepared_ &&
                        z2_masks_plan_hash_ == plan_hash &&
                        z2_mask_generation_ != nullptr &&
                        *z2_mask_generation_ ==
                            z2_masks_prepared_generation_ &&
                        z2_masks_prepared_generation_ >
                            *z2_masks_consumed_generation_ &&
                        prepared_cross_mask_.mask_type == 4 &&
                        prepared_cross_mask_img_.mask_type == 4 &&
                        text_mask_slot_.defined() &&
                        image_mask_slot_.defined() &&
                        text_mask_slot_.dim() == 2 &&
                        image_mask_slot_.dim() == 2 &&
                        text_mask_slot_.size(0) == action_horizon_ + 1 &&
                        image_mask_slot_.size(0) == action_horizon_ + 1 &&
                        text_mask_slot_.size(1) == 96 &&
                        image_mask_slot_.size(1) == 96 &&
                        prepared_cross_mask_.ddr_tensor.defined() &&
                        prepared_cross_mask_img_.ddr_tensor.defined() &&
                        prepared_cross_mask_.ddr_tensor.unsafeGetTensorImpl() ==
                            text_mask_slot_.unsafeGetTensorImpl() &&
                        prepared_cross_mask_img_.ddr_tensor
                                .unsafeGetTensorImpl() ==
                            image_mask_slot_.unsafeGetTensorImpl(),
                    "gr00t denoise outer-fast prepared masks are stale");
        TORCH_CHECK(emb_stage_.defined() && emb_stage_.dim() == 3 &&
                        emb_stage_.size(0) == 1 &&
                        emb_stage_.size(1) == action_horizon_ + 1 &&
                        emb_stage_.size(2) == hidden_size() &&
                        emb_stage_.scalar_type() == at::kHalf &&
                        emb_stage_.device().type() == at::kPrivateUse1 &&
                        emb_stage_.is_contiguous(),
                    "gr00t denoise outer-fast emb_stage owner drifted");

        // Keep the previous destination alive until this allocation finishes;
        // Graph's mutable-destination seal then rejects TensorImpl/StorageImpl
        // reuse against the last successful invocation.
        at::Tensor out =
            allocate_tracked_output({1, action_horizon_, action_dim_pad_});
        guard.bind_fast_mutable_dma(
            actions_src_base_, noise_pad, GraphOuterFastDmaSide::Source,
            /*byte_begin=*/0, noise_pad.nbytes());
        guard.bind_fast_mutable_dma(
            state_src_base_, state_pad, GraphOuterFastDmaSide::Source,
            /*byte_begin=*/0, state_pad.nbytes());
        bind_spm_outer_fast_input(guard, emb_stage_);
        guard.bind_fast_mutable_dma(
            x_out_dst_base_, out, GraphOuterFastDmaSide::Destination,
            /*byte_begin=*/0, out.nbytes());
        return out;
    }

protected:
    KvCostLayoutScope capture_kvinsert_cost_layout_scope() override {
        return capture_kvinsert_cost_layout_fields(
            denoise_mode_, loop_mode_,
            vl_seq_, step_idx_);
    }

    static std::vector<int64_t> adaln_route_arguments(
        bool denoise, bool loop, int64_t vl_seq, int64_t step_idx) {
        std::vector<int64_t> args{denoise ? 1 : 0, vl_seq};
        if (denoise && !loop) args.push_back(step_idx);
        return args;
    }

    // ---- static / dynamic config ----
    ModelStaticConfig static_config() override {
        ModelStaticConfig cfg;
        cfg.num_layers = num_layers();
        cfg.cross_layer_batch_size = num_layers();  // single group (AdaRMS/SigLIP discipline)
        // Denoise mode (Phase B/C): fold the per-step glue into pre/post hooks; unroll repeats
        // the body num_steps_ times (Phase C). Plain DiT path leaves hooks null + 1 iteration.
        if (denoise_mode_) {
            cfg.pre_layers_fn  = reinterpret_cast<void (FusedModelBase::*)()>(
                &Gr00tDiTModel::emit_pre_layers_body);
            cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
                &Gr00tDiTModel::emit_post_layers_body);
            cfg.body_iterations = loop_mode_ ? num_steps_ : 1;
        } else {
            cfg.body_iterations = 1;                 // per-step REPLAY first; unroll later
        }
        return cfg;
    }

    ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override {
        TORCH_CHECK(plan.num_chunks == 1,
            "gr00t_dit requires a single chunk (got ", plan.num_chunks,
            ", chunk_size=", plan.chunk_size,
            "); multi-chunk would overwrite self-attention K/V at position 0");
        ModelDynamicConfig cfg;
        cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
        cfg.inter_layer_io = InterLayerIO::SPM_RESIDENT;   // 41x1536 fp16 ~126KB, fits SPM
        // RAW_SPM is descriptor-owned: legacy/no-descriptor entry points keep
        // the historical DDR route and cannot silently select a physical ABI.
        cfg.attention_policy = ctx().has_complete_physical_manifest()
            ? AttentionExecutionPolicy::AUTO
            : AttentionExecutionPolicy::DDR_KV;
        return cfg;
    }

    bool subclass_chunk_size_valid(
        int64_t chunk_size, int64_t seq_len,
        int64_t /*position*/) const override {
        return chunk_size >= seq_len;
    }

    FmbPhysicalExecutionManifest physical_manifest_for_candidate(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position) const override {
        const bool request_raw = layout.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        return physical_manifest_for_attention_routes(
            plan, layout, physical_len, logical_len, position,
            request_raw && self_raw_spm_eligible(plan, layout, position),
            request_raw && cross_raw_spm_eligible(plan, layout, position));
    }

    FmbPhysicalExecutionManifest physical_manifest_for_attention_routes(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position, bool self_raw_spm,
        bool cross_raw_spm) const {
        TORCH_CHECK(
            position == 0 && plan.compute.chunks.size() == 1,
            "GR00T DiT COMPLETE descriptor requires one position-zero chunk");
        const ChunkInfo& chunk = plan.compute.chunks.front();
        LayoutContext spm_layout = layout;
        spm_layout.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        TORCH_CHECK(
            !self_raw_spm ||
                self_raw_spm_eligible(plan, spm_layout, position),
            "RPU_PLANNER_REJECT:CAPABILITY: GR00T DiT self raw-SPM "
            "attention was selected outside its exact full-K/V profile");
        TORCH_CHECK(
            !cross_raw_spm ||
                cross_raw_spm_eligible(plan, spm_layout, position),
            "RPU_PLANNER_REJECT:CAPABILITY: GR00T DiT cross raw-SPM "
            "attention was selected outside its exact full-K/V profile");
        const bool has_self = std::any_of(
            layer_weights_.begin(), layer_weights_.end(),
            [](const LayerWeights& layer) { return !layer.is_cross; });
        const bool has_cross = std::any_of(
            layer_weights_.begin(), layer_weights_.end(),
            [](const LayerWeights& layer) { return layer.is_cross; });
        TORCH_CHECK(!has_cross || vl_seq_ > 0,
                    "GR00T DiT cross blocks require planned cross KV rows");

        FmbPhysicalExecutionManifest manifest;
        manifest.state = FmbPhysicalManifestState::COMPLETE;
        manifest.logical_length = logical_len;
        manifest.physical_length = physical_len;
        manifest.execution_padding_rows = physical_len - logical_len;
        manifest.kv_logical_length = std::max(logical_len, vl_seq_);
        manifest.kv_insert_physical_rows = std::max(
            gr00t_pad16(chunk.len),
            has_cross ? gr00t_pad16(vl_seq_) : int64_t{0});
        manifest.graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
        manifest.linear_accumulation = FmbLinearAccumulationPolicy::ACC16;

        const int64_t invocation = chunk.idx;
        const auto append = [&](FmbRouteFamily family, int64_t site_id,
                                int64_t selector,
                                std::vector<int64_t> arguments = {}) {
            manifest.routes.push_back({
                site_id, family, selector, /*flags=*/0,
                std::move(arguments), invocation});
        };
        const auto append_linear = [&](int64_t site_id) {
            append(FmbRouteFamily::LINEAR, site_id,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE));
        };
        const auto append_kv = [&](int64_t site_id, int64_t rows) {
            const KvInsertSegmentPlan kv_plan =
                resolve_kvinsert_plan_auto(
                    site_id, manifest.graph_lifecycle,
                    /*position=*/0, rows, gr00t_pad16(rows),
                    attn_tp(), num_kv_heads(), head_dim(),
                    GR00T_DIT_KV_CAPABILITIES);
            const KvInsertRouteArguments arguments =
                rpu_kvinsert_route_arguments(
                    kv_plan, attn_tp(), num_kv_heads(), head_dim());
            manifest.routes.push_back({
                site_id, FmbRouteFamily::KV_INSERT,
                static_cast<int64_t>(kv_plan.route()),
                GR00T_DIT_KV_FLAG_DDR_MIRROR,
                {arguments.begin(), arguments.end()}, invocation});
        };

        if (has_cross) {
            // Mint the schedule for a future COMPLETE allocation. The initial
            // capacity-only probe has no fingerprint and retains the baseline.
            LayoutContext mask_layout = layout;
            mask_layout.attention_policy = self_raw_spm || cross_raw_spm
                ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                : AttentionExecutionPolicy::DDR_KV;
            append(FmbRouteFamily::GRAPH_SCHEDULE, GR00T_DIT_MASK_SCHEDULE_SITE,
                   mask_schedule_for_layout(mask_layout, /*complete=*/true),
                   mask_schedule_arguments(chunk.len));
        }

        if (!denoise_mode_) {
            append(FmbRouteFamily::MUTABLE_DMA, GR00T_DIT_COND_DMA_SITE,
                   static_cast<int64_t>(
                       Gr00tDiTMutableDmaRoute::DDR_SCATTER_TO_SPM));
        }
        if (vl_seq_ > 0 && !z2_bound_) {
            append(FmbRouteFamily::MUTABLE_DMA, GR00T_DIT_VL_DMA_SITE,
                   static_cast<int64_t>(
                       Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        }
        append(
            FmbRouteFamily::LINEAR, GR00T_DIT_ADALN_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            adaln_route_arguments(denoise_mode_, loop_mode_, vl_seq_, step_idx_));
        append(FmbRouteFamily::ALL_REDUCE,
               GR00T_DIT_ADALN_ALL_REDUCE_SITE,
               fmb_ring_all_reduce_route_selector(
                   /*rows=*/1, /*cols=*/2 * hidden_size()));
        if (has_self) {
            append_linear(GR00T_DIT_SELF_Q_LINEAR_SITE);
            append_linear(GR00T_DIT_SELF_K_LINEAR_SITE);
            append_linear(GR00T_DIT_SELF_V_LINEAR_SITE);
            append_kv(GR00T_DIT_SELF_KV_INSERT_SITE, chunk.len);
            manifest.routes.push_back({
                self_raw_spm
                    ? GR00T_DIT_SELF_RAW_ATTENTION_SITE
                    : GR00T_DIT_SELF_DDR_ATTENTION_SITE,
                FmbRouteFamily::ATTENTION,
                static_cast<int64_t>(
                    self_raw_spm
                        ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                        : AttentionExecutionPolicy::DDR_KV),
                self_raw_spm || self_raw_spm_eligible(
                    plan, spm_layout, position)
                    ? 0 : GR00T_DIT_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                {}, invocation});
        }
        if (has_cross) {
            append_linear(GR00T_DIT_CROSS_Q_LINEAR_SITE);
            append_linear(GR00T_DIT_CROSS_K_LINEAR_SITE);
            append_linear(GR00T_DIT_CROSS_V_LINEAR_SITE);
            append_kv(GR00T_DIT_CROSS_KV_INSERT_SITE, vl_seq_);
            manifest.routes.push_back({
                cross_raw_spm
                    ? GR00T_DIT_CROSS_RAW_ATTENTION_SITE
                    : GR00T_DIT_CROSS_DDR_ATTENTION_SITE,
                FmbRouteFamily::ATTENTION,
                static_cast<int64_t>(
                    cross_raw_spm
                        ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                        : AttentionExecutionPolicy::DDR_KV),
                cross_raw_spm || cross_raw_spm_eligible(
                    plan, spm_layout, position)
                    ? 0 : GR00T_DIT_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                {}, invocation});
        }
        append_linear(GR00T_DIT_O_LINEAR_SITE);
        append(FmbRouteFamily::ALL_REDUCE,
               GR00T_DIT_ATTN_ALL_REDUCE_SITE,
               fmb_ring_all_reduce_route_selector(chunk.len, hidden_size()));
        append_linear(GR00T_DIT_FF1_LINEAR_SITE);
        append_linear(GR00T_DIT_FF2_LINEAR_SITE);
        append(FmbRouteFamily::ALL_REDUCE,
               GR00T_DIT_FF_ALL_REDUCE_SITE,
               fmb_ring_all_reduce_route_selector(chunk.len, hidden_size()));
        if (denoise_ready_) {
            const int64_t action_horizon = action_horizon_;
            const int64_t hidden = hidden_size();
            const int64_t selector = static_cast<int64_t>(
                Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM);
            const auto append_preload = [&](int64_t site_id,
                                            int64_t route_invocation,
                                            int64_t elements) {
                manifest.routes.push_back({
                    site_id, FmbRouteFamily::MUTABLE_DMA, selector,
                    /*flags=*/0, {denoise_ready_ ? 1 : 0, num_steps_, elements},
                    route_invocation});
            };
            append_preload(
                GR00T_DIT_DENOISE_TABLE_PRELOAD_SITE, 0,
                num_steps_ * action_horizon * hidden);
            append_preload(
                GR00T_DIT_DENOISE_TABLE_PRELOAD_SITE, 1,
                action_horizon * hidden);
            const int64_t bias_elements[] = {
                enc_mid_, hidden, hidden, hidden, hidden, dec_mid_,
                action_dim_pad_, 2 * hidden, model_out_dim_};
            for (int64_t index = 0; index < 9; ++index) {
                append_preload(
                    GR00T_DIT_DENOISE_BIAS_PRELOAD_SITE, index,
                    bias_elements[index]);
            }
        }
        if (denoise_mode_) {
            append(FmbRouteFamily::MUTABLE_DMA,
                   GR00T_DIT_ACTION_DMA_SITE,
                   static_cast<int64_t>(
                       Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM));
            append(FmbRouteFamily::MUTABLE_DMA,
                   GR00T_DIT_STATE_DMA_SITE,
                   static_cast<int64_t>(
                       Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM));
            append_linear(GR00T_DIT_SC_LINEAR_SITE);
            append(FmbRouteFamily::MUTABLE_DMA,
                   GR00T_DIT_OUTPUT_DMA_SITE,
                   static_cast<int64_t>(
                       Gr00tDiTMutableDmaRoute::SPM_TO_DDR));
        }
        if (!z2_bound_) {
            append_fmb_shared_runtime_routes(
                manifest, plan, hidden_size(),
                FMB_SHARED_LAYER_INPUT_DMA);
        }
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
        LayoutContext spm_layout = layout;
        spm_layout.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const bool self_raw = self_raw_spm_eligible(
            plan, spm_layout, position);
        const bool cross_raw = cross_raw_spm_eligible(
            plan, spm_layout, position);
        std::vector<FmbPhysicalExecutionManifest> domain;
        domain.push_back(physical_manifest_for_attention_routes(
            plan, layout, physical_len, logical_len, position,
            /*self_raw_spm=*/false, /*cross_raw_spm=*/false));
        if (self_raw) {
            domain.push_back(physical_manifest_for_attention_routes(
                plan, layout, physical_len, logical_len, position,
                /*self_raw_spm=*/true, /*cross_raw_spm=*/false));
        }
        if (cross_raw) {
            domain.push_back(physical_manifest_for_attention_routes(
                plan, layout, physical_len, logical_len, position,
                /*self_raw_spm=*/false, /*cross_raw_spm=*/true));
        }
        if (self_raw && cross_raw) {
            domain.push_back(physical_manifest_for_attention_routes(
                plan, layout, physical_len, logical_len, position,
                /*self_raw_spm=*/true, /*cross_raw_spm=*/true));
        }
        return domain;
    }

    FmbPhysicalManifestForwardCapability
    physical_manifest_forward_capability(
        const FmbPhysicalExecutionManifest& /*manifest*/) const override {
        return {true, FmbGraphLifecycle::COMPOSITE_CHILD};
    }

    bool raw_spm_common_profile_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const {
        if (z2_bound_ || z2_prepared_seq_len_ != 0 || position != 0 ||
            layout.is_causal || layout.use_attn_mask ||
            layout.batch_size != 1 ||
            layout.attention_policy !=
                AttentionExecutionPolicy::SPM_KV_BY_MHA ||
            plan.chunk_mode != ChunkMode::SEQUENTIAL ||
            plan.input.chunks.size() != 1 ||
            plan.qkv.chunks.size() != 1 ||
            plan.compute.chunks.size() != 1 || plan.spans.size() != 1) {
            return false;
        }
        const ChunkInfo& chunk = plan.compute.chunks.front();
        if (chunk.offset != 0 || chunk.len != 41 ||
            chunk.kv_seq_len != 41 ||
            plan.input.chunks.front().offset != 0 ||
            plan.input.chunks.front().len != 41 ||
            plan.qkv.chunks.front().offset != 0 ||
            plan.qkv.chunks.front().len != 41 ||
            plan.spans.front().offset != 0 || plan.spans.front().len != 41 ||
            num_layers() != 32 || layer_weights_.size() != 32 ||
            hidden_size() != 1536 || intermediate_size() != 6144 ||
            num_q_heads() != 32 || num_kv_heads() != 32 ||
            head_dim() != 48 || cross_dim_ != 2048) {
            return false;
        }
        return true;
    }

    bool self_raw_spm_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const {
        if (!raw_spm_common_profile_eligible(plan, layout, position)) {
            return false;
        }
        const bool has_self = std::any_of(
            layer_weights_.begin(), layer_weights_.end(),
            [](const LayerWeights& layer) { return !layer.is_cross; });
        const int64_t seq = plan.compute.chunks.front().len;
        return has_self && sdpa_by_mha_spm_is_valid(
            /*batch=*/1, seq, seq,
            num_q_heads(), num_kv_heads(), head_dim(), NUM_CORES,
            /*MASK_NONE=*/0);
    }

    bool cross_raw_spm_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const {
        if (!raw_spm_common_profile_eligible(plan, layout, position)) {
            return false;
        }
        const bool has_cross = std::any_of(
            layer_weights_.begin(), layer_weights_.end(),
            [](const LayerWeights& layer) { return layer.is_cross; });
        const int64_t seq = plan.compute.chunks.front().len;
        return has_cross && vl_seq_ >= seq && sdpa_by_mha_spm_is_valid(
            /*batch=*/1, seq, vl_seq_,
            num_q_heads(), num_kv_heads(), head_dim(), NUM_CORES,
            /*MASK_2D=*/4);
    }

    bool subclass_spm_kv_by_mha_eligible(
        const FmbThreeStageChunkPlan& plan,
        const LayoutContext& layout,
        int64_t position) const override {
        return self_raw_spm_eligible(plan, layout, position) ||
            cross_raw_spm_eligible(plan, layout, position);
    }

    // ---- SPM manifest (design §3). All LayerWide; activations tiny. ----
    std::vector<BufferDecl> baseline_buffer_declarations(const LayoutContext& ctx) const {
        const int64_t cs = ctx.chunk_size;
        const int64_t h  = hidden_size();          // 1536
        const int64_t nq = num_q_heads();          // 32
        const int64_t hd = head_dim();             // 48
        const int64_t FF = intermediate_size();    // 6144
        // The planner freezes PAD16 V16 at position zero for both self and
        // cross KV; reserve that exact physical source geometry.
        const int64_t kv_seq = gr00t_pad16(std::max<int64_t>(cs, vl_seq_));

        const int64_t local_qhd   = (nq * hd) / NUM_CORES;   // per-core attn width (192)
        const int64_t local_inter = FF / NUM_CORES;          // per-core FFN width (768)
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

        const int64_t res     = A(cs * h * DWIDTH);
        const int64_t qbuf    = A(cs * local_qhd * DWIDTH);       // q: seq_q tokens
        const int64_t kvbuf   = A(kv_seq * local_qhd * DWIDTH);   // k/v: up to S tokens
        const int64_t outbuf  = A(cs * local_qhd * DWIDTH);       // SDPA out: seq_q tokens
        const int64_t mlp     = A(cs * local_inter * DWIDTH);
        const int64_t adaln   = A(2 * h * DWIDTH);
        const int64_t cond_sz = A(h * DWIDTH);
        const int64_t ln_sz   = A(h * DWIDTH);

        // SDPA scratch: size for the worst of self (MASK_NONE) and cross (MASK_2D).
        int64_t tmp_self  = sdpa_compute_tmp_v16_size(make_sdpa_config(0), cs);
        int64_t tmp_cross = (vl_seq_ > 0) ? sdpa_compute_tmp_v16_size(make_sdpa_config(4), cs) : 0;
        int64_t tmp = A(std::max(tmp_self, tmp_cross) * 32);
        if (ctx.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            // V transpose is [H/core,D,PAD16(S)] for the larger of self/cross.
            tmp = std::max(tmp, A(kv_seq * local_qhd * DWIDTH));
        }

        constexpr BufferScope ALL = BufferScope::LayerWide;
        std::vector<BufferDecl> v = {
            {"residual1",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"residual2",    res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"norm_hidden",  res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"q",            qbuf,    0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"k",            kvbuf,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"v",            kvbuf,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_out",     outbuf,  0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"oproj",        res,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"sdpa_tmp",     tmp,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"ff_mid",       mlp,     0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"adaln",        adaln,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"bias_temp",    adaln,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"gemv_partial", adaln,   0, 0, StorageClass::Temp, 0, nullptr, ALL},
            {"cond",         cond_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL},
        };
        // Cross-attn buffers (only when vl_embeds is in play).
        if (vl_seq_ > 0) {
            const int64_t vl_sz   = A(vl_seq_ * cross_dim_ * DWIDTH);                 // replicated [S,2048]
            const int64_t mask_sz = A(cs * CeilDiv(vl_seq_, (int64_t)16) * 32);       // MASK_2D [seq_q,S]
            v.push_back({"vl_embeds", vl_sz,   0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"sdpa_mask", mask_sz, 0, 0, StorageClass::Temp, 0, nullptr, ALL});
        }

        // norm3 affine=False gamma/beta — Persistent, preloaded from members.
        auto ln_const = [&](const char* name, at::Tensor Gr00tDiTModel::* field) {
            BufferDecl d; d.name = name; d.size = ln_sz;
            d.storage = StorageClass::Persistent; d.scope = ALL;
            d.preload_callback = [this, field](FusedModelBase&, int, uint32_t core0_addr) {
                rpu_launch_ddr_broadcast_spm_dma(
                    this->*field, /*src_offset_elements=*/0,
                    hidden_size(), core0_addr);
            };
            v.push_back(d);
        };
        ln_const("ln_ones",  &Gr00tDiTModel::ln_ones_);
        ln_const("ln_zeros", &Gr00tDiTModel::ln_zeros_);

        // ---- Per-layer linear biases (all DiT linears carry bias) ----
        const int nl = static_cast<int>(num_layers());
        auto col_bias = [&](const char* name, int64_t per_core,
                            at::Tensor LayerWeights::* field) {
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
        col_bias("ff1_bias", local_inter, &LayerWeights::ff1_b);
        row_bias("out_bias", &LayerWeights::to_out_b);
        row_bias("ff2_bias", &LayerWeights::ff2_b);

        // ---- Phase B/C compute scratch follows the active execution mode. ----
        if (denoise_mode_) {
            const int64_t AH = action_horizon_, SQ = AH + 1, NP = action_dim_pad_;
            const int64_t EM = enc_mid_, DM = dec_mid_, MO = model_out_dim_;
            // compute temps (single-core core0; distinct slots via phase 0,0).
            v.push_back({"x_t_spm",      A(AH * NP * DWIDTH),  0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"state_feat",   A(h * DWIDTH),        0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"cond_full",    A(h * DWIDTH),        0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"se_mid",       A(EM * DWIDTH),       0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"enc0",         A(AH * h * DWIDTH),   0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"enc1",         A(AH * h * DWIDTH),   0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"enc2",         A(AH * h * DWIDTH),   0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"so",           A(2 * h * DWIDTH),    0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"normed",       A(SQ * h * DWIDTH),   0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"model_out",    A(SQ * MO * DWIDTH),  0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"dec_mid",      A(SQ * DM * DWIDTH),  0, 0, StorageClass::Temp, 0, nullptr, ALL});
            v.push_back({"pred",         A(SQ * NP * DWIDTH),  0, 0, StorageClass::Temp, 0, nullptr, ALL});
        }
        // Installed Persistent weights keep one immutable allocation/preload
        // identity while this same handle switches plain / step / unroll.
        if (denoise_ready_) {
            const int64_t AH = action_horizon_, NP = action_dim_pad_;
            const int64_t EM = enc_mid_, DM = dec_mid_, MO = model_out_dim_;
            // tables broadcast to core 0 (read by single-core encoder GEMMs / adds).
            auto tbl = [&](const char* name, int64_t elems,
                           at::Tensor Gr00tDiTModel::* fld,
                           int64_t route_invocation) {
                BufferDecl d; d.name = name; d.size = A(elems * DWIDTH);
                d.storage = StorageClass::Persistent; d.scope = ALL;
                d.preload_callback = [this, elems, fld, route_invocation](
                    FusedModelBase&, int, uint32_t a0) {
                    this->ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        GR00T_DIT_DENOISE_TABLE_PRELOAD_SITE,
                        static_cast<int64_t>(
                            Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                        /*resolved_flags=*/0,
                        {denoise_ready_ ? 1 : 0, num_steps_, elems},
                        route_invocation);
                    rpu_launch_ddr_broadcast_spm_dma(
                        this->*fld, /*src_offset_elements=*/0, elems, a0,
                        /*num_cores=*/1);
                };
                v.push_back(d);
            };
            tbl("tau_table", num_steps_ * AH * h,
                &Gr00tDiTModel::tau_table_, /*route_invocation=*/0);
            tbl("pos_table", AH * h, &Gr00tDiTModel::pos_table_,
                /*route_invocation=*/1);
            // glue biases broadcast to core 0 (single-core GEMM bias_spm_addr).
            auto dbias = [&](const char* name, int64_t elems,
                             at::Tensor Gr00tDiTModel::* fld,
                             int64_t route_invocation) {
                BufferDecl d; d.name = name; d.size = A(elems * DWIDTH);
                d.storage = StorageClass::Persistent; d.scope = ALL;
                d.preload_callback = [this, elems, fld, route_invocation](
                    FusedModelBase&, int, uint32_t a0) {
                    this->ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        GR00T_DIT_DENOISE_BIAS_PRELOAD_SITE,
                        static_cast<int64_t>(
                            Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                        /*resolved_flags=*/0,
                        {denoise_ready_ ? 1 : 0, num_steps_, elems},
                        route_invocation);
                    rpu_launch_ddr_broadcast_spm_dma(
                        this->*fld, /*src_offset_elements=*/0, elems, a0,
                        /*num_cores=*/1);
                };
                v.push_back(d);
            };
            dbias("se1_bias", EM, &Gr00tDiTModel::se1_b_, 0);
            dbias("se2_bias", h,  &Gr00tDiTModel::se2_b_, 1);
            dbias("ae1_bias", h,  &Gr00tDiTModel::ae1_b_, 2);
            dbias("ae2_bias", h,  &Gr00tDiTModel::ae2_b_, 3);
            dbias("ae3_bias", h,  &Gr00tDiTModel::ae3_b_, 4);
            dbias("dec1_bias", DM, &Gr00tDiTModel::dec1_b_, 5);
            dbias("dec2_bias", NP, &Gr00tDiTModel::dec2_b_, 6);
            dbias("po1_bias", 2 * h, &Gr00tDiTModel::po1_b_, 7);
            dbias("po2_bias", MO,  &Gr00tDiTModel::po2_b_, 8);
        }
        return v;
    }

    int first_cross_mask_layer(bool image) const {
        for (size_t i = 0; i < layer_weights_.size(); ++i) {
            if (layer_weights_[i].is_cross && ((i % 4 != 0) == image))
                return static_cast<int>(i);
        }
        return -1;
    }

    std::vector<BufferDecl> planned_buffer_declarations(
        const LayoutContext& layout, bool complete, bool* retained = nullptr) const {
        auto baseline = baseline_buffer_declarations(layout);
        if (retained) *retained = false;
        // Ordinary COMPLETE single-chunk public DiT geometry only. Z2 has an
        // independently sealed lease/register contract: its prepare has no
        // COMPLETE fingerprint and its prepared/bound executions stay baseline.
        if (!complete || z2_bound_ || z2_prepared_seq_len_ != 0 ||
            layout.chunk_size != 41 || layout.batch_size != 1 ||
            layout.is_causal || layout.use_attn_mask ||
            hidden_size() != 1536 || intermediate_size() != 6144 ||
            num_q_heads() != 32 || head_dim() != 48 || cross_dim_ != 2048 ||
            vl_seq_ <= 0 || vl_seq_ > 96 ||
            first_cross_mask_layer(false) < 0 || first_cross_mask_layer(true) < 0)
            return baseline;

        // All other scratch stays conservatively live through hooks and every
        // layer. Only AdaLN's input temporaries and the later Q/K operands have
        // disjoint proven windows: AdaLN consumes bias/partial before Q/K
        // projection; Q/K are dead after SDPA and absent from FFN/pre/post hooks.
        auto candidate = baseline;
        BufferDecl image_mask;
        int found = 0;
        for (auto& decl : candidate) {
            if (decl.storage != StorageClass::Temp || decl.alias_of) continue;
            if (decl.scope != BufferScope::LayerWide ||
                decl.phase_start != 0 || decl.phase_end != 0)
                return baseline;
            decl.phase_start = 0;
            decl.phase_end = 8;
            const std::string name = decl.name ? decl.name : "";
            if (name == "bias_temp" || name == "gemv_partial") {
                decl.phase_start = decl.phase_end = 1;
                ++found;
            } else if (name == "q" || name == "k") {
                decl.phase_start = 2;
                decl.phase_end = 3;
                ++found;
            } else if (name == "sdpa_mask") {
                image_mask = decl;
                image_mask.name = "sdpa_mask_image";
                ++found;
            }
        }
        if (found != 5) return baseline;
        candidate.push_back(std::move(image_mask));
        const auto plan = detail::plan_forward_spm_residency(
            candidate, {"sdpa_mask", "sdpa_mask_image"});
        // The helper compares the extended candidate to itself; also compare
        // with the ORIGINAL complete declarations, before adding a second mask.
        if (!plan.retained || plan.retained_peak_bytes >
                detail::estimate_temporary_total(baseline))
            return baseline;
        if (retained) *retained = true;
        return candidate;
    }

    std::vector<BufferDecl> declare_buffers(const LayoutContext& layout) override {
        return planned_buffer_declarations(
            layout, /*complete=*/layout.physical_manifest_fingerprint != 0);
    }

    int64_t mask_schedule_for_layout(const LayoutContext& layout, bool complete) const {
        bool retained = false;
        planned_buffer_declarations(layout, complete, &retained);
        return static_cast<int64_t>(retained
            ? Gr00tDiTMaskSchedule::RETAIN_TEXT_AND_IMAGE
            : Gr00tDiTMaskSchedule::UPLOAD_EACH_CROSS_BLOCK);
    }

    std::vector<int64_t> mask_schedule_arguments(int64_t rows) const {
        return {rows, vl_seq_, denoise_mode_ ? 1 : 0, loop_mode_ ? 1 : 0,
                loop_mode_ ? num_steps_ : 1, num_layers(),
                first_cross_mask_layer(false) + 1, first_cross_mask_layer(true) + 1};
    }

    // `vl_seq_` is the VLM prefix length and it is re-set on EVERY forward
    // (forward, step_forward, unroll_forward). It sizes non-aliased Temp decls
    // above — "k", "v", "sdpa_tmp" — and gates the existence of "vl_embeds" and
    // "sdpa_mask". Meanwhile every input the framework's params hash CAN see is
    // pinned for the handle: the denoise unroll always runs
    // seq_len = action_horizon+1 with no explicit mask, and use_attn_mask==false
    // is exactly the case where max_kv_seq_len is left OUT of the hash. So a
    // prompt-length change moved the SPM layout with the hash frozen. The
    // adapter's own GraphSignature already carries S (adapters/gr00t/runtime.py),
    // so a new prefix does trigger a fresh BUILD — which was being emitted
    // against offsets sized for the previous prefix. This is the one subclass in
    // the tree where the gap is reachable from an ordinary request.
    // `denoise_mode_` and `cross_dim_` gate/size the same block and flip at most
    // once each; folded in for completeness.
    int64_t subclass_layout_hash() const override {
        int64_t h = detail::layout_mix(0, vl_seq_);
        h = detail::layout_mix(h, denoise_mode_ ? 1 : 0);
        h = detail::layout_mix(h, cross_dim_);
        return h;
    }

    // ---- per-block compute: dispatch on explicit per-layer block type ----
    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {
        const int64_t h = hidden_size();

        // Phase 0: cond scatter + (if cross used anywhere) vl_embeds broadcast — once
        // per forward, on the FIRST layer/chunk0 (block type agnostic).
        const bool is_first = (layer_idx == 0) && (chunk.idx == 0);
        if (is_first && ctx().body_iter == 0 &&
            ctx().has_complete_physical_manifest() &&
            (first_cross_mask_layer(false) >= 0 || first_cross_mask_layer(true) >= 0)) {
            LayoutContext layout;
            layout.chunk_size = ctx().stage_plan.compute.chunks.front().len;
            layout.max_kv_seq_len = ctx().seq_len;
            layout.num_layers = num_layers();
            layout.use_attn_mask = ctx().attention_mask.has_value();
            layout.is_causal = ctx().is_causal;
            layout.batch_size = ctx().batch_size;
            layout.attention_policy = ctx().attention_policy;
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, GR00T_DIT_MASK_SCHEDULE_SITE,
                mask_schedule_for_layout(layout, /*complete=*/true),
                /*resolved_flags=*/0, mask_schedule_arguments(chunk.len), chunk.idx);
        }

        // Plain DiT path: cond is the forward input arg, scattered here. Denoise path
        // (Phase B/C): cond comes from the baked table, scattered by the pre-hook → skip.
        if (is_first && !denoise_mode_ && !cond_loaded_this_forward_) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, GR00T_DIT_COND_DMA_SITE,
                static_cast<int64_t>(
                    Gr00tDiTMutableDmaRoute::DDR_SCATTER_TO_SPM),
                /*resolved_flags=*/0, {}, chunk.idx);
            const int64_t local_k = h / NUM_CORES;
            rpu_launch_ddr_scatter_spm_dma_mutable(
                cond_src_base_, cond_ref_, /*src_offset_bytes=*/0,
                /*elements_per_core=*/local_k, /*core_stride_bytes=*/local_k * DWIDTH,
                addr(0, "cond"), NUM_CORES);
            cond_loaded_this_forward_ = true;
        }
        if (is_first && vl_seq_ > 0 && !vl_loaded_this_forward_ && !z2_bound_) {
            // vl_embeds replicated to all cores (linear input reads full cross_dim).
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA, GR00T_DIT_VL_DMA_SITE,
                static_cast<int64_t>(
                    Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                vl_src_base_, vl_ref_, /*src_offset_bytes=*/0,
                vl_seq_ * cross_dim_, addr(0, "vl_embeds"), NUM_CORES);
            vl_loaded_this_forward_ = true;
        }

        if (layer_weights_[layer_idx].is_cross) build_cross_attn_block(layer_idx, chunk);
        else                                    build_self_attn_block(layer_idx, chunk);
    }

    KvInsertSegmentPlan consume_kv_insert_plan(
        int64_t site_id, int64_t true_seq, int64_t invocation) {
        const auto& route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, site_id, invocation);
        const KvInsertSegmentPlan plan =
            restore_kvinsert_plan(
                site_id, route.arguments, attn_tp(), num_kv_heads(), head_dim());
        TORCH_CHECK(plan.segment(0).position == 0 &&
                        plan.logical_rows() == true_seq &&
                        plan.physical_rows() == gr00t_pad16(true_seq),
                    "GR00T DiT KV descriptor geometry drift");
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, site_id,
            static_cast<int64_t>(plan.route()),
            GR00T_DIT_KV_FLAG_DDR_MIRROR,
            route.arguments, invocation);
        if (plan.physical_rows() == true_seq) return plan;
        const int64_t local_qhd = (num_q_heads() * head_dim()) / NUM_CORES;
        const int64_t pad_elems =
            (plan.physical_rows() - true_seq) * local_qhd;
        const uint32_t off = static_cast<uint32_t>(true_seq * local_qhd * DWIDTH);
        rpu_launch_memset_spm_multicore(addr(0, "k") + off, pad_elems);
        rpu_launch_memset_spm_multicore(addr(0, "v") + off, pad_elems);
        return plan;
    }

    // ====================== SELF-ATTN BLOCK ======================
    void build_self_attn_block(int layer_idx, const ChunkInfo& chunk) {
        const auto& lw = layer_weights_[layer_idx];
        const int64_t seq_len     = chunk.len;
        const int64_t h           = hidden_size();
        const int64_t nq          = num_q_heads();
        const int64_t hd          = head_dim();
        const int64_t FF          = intermediate_size();
        const int64_t local_inter = FF / NUM_CORES;

        if (chunk.idx == 0) {
            dit_adaln_gemv(addr(0, "cond"), lw.adaln_w, lw.adaln_b,
                           addr(0, "adaln"), addr(0, "bias_temp"), addr(0, "gemv_partial"), lw.adaln_ws);
        }
        const uint32_t scale = addr(0, "adaln");
        const uint32_t shift = scale + static_cast<uint32_t>(h * DWIDTH);

        if (!ctx().input_in_spm) emit_layer_input_dma(layer_idx, chunk);

        // AdaLayerNorm(norm1) = (1+scale)*LN(x)+shift -> norm_hidden (one LN call)
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual1"), addr(0, "norm_hidden"),
            scale, shift, seq_len, h, eps_, false, 0, NUM_CORES);

        // QKV (col-partition) + bias
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_SELF_Q_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "q_bias"), lw.to_q_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_SELF_K_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "k_bias"), lw.to_k_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_SELF_V_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "v_bias"), lw.to_v_ws);

        // KV-insert(pos 0) + self-SDPA (non-causal, MASK_NONE). v16-pad: insert pad16(seq_len)
        // (zeroed tail) so the unified insert uses v16; SDPA below still reads kv_seq_len=seq_len.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const KvInsertSegmentPlan kv_plan = consume_kv_insert_plan(
            GR00T_DIT_SELF_KV_INSERT_SITE, seq_len, chunk.idx);
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache,
            addr_offset("k").value, addr_offset("v").value,
            nq, hd, NUM_CORES,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            /*spm_rows=*/kv_plan.physical_rows(), kv_plan);
        const double attn_scale = 1.0 / std::sqrt(static_cast<double>(hd));
        const bool raw_attention = ctx().attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const int64_t attention_site = raw_attention
            ? GR00T_DIT_SELF_RAW_ATTENTION_SITE
            : GR00T_DIT_SELF_DDR_ATTENTION_SITE;
        const AttentionExecutionPolicy attention_policy =
            ctx().physical_attention_policy_for_site(
                attention_site, chunk.idx);
        if (attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                GR00T_DIT_SELF_RAW_ATTENTION_SITE,
                static_cast<int64_t>(
                    AttentionExecutionPolicy::SPM_KV_BY_MHA),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_v_transpose_spm(
                addr(0, "v"), addr(0, "sdpa_tmp"),
                /*batch=*/1, seq_len, nq, hd, NUM_CORES);
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q"), addr(0, "k"), addr(0, "sdpa_tmp"),
                addr(0, "sdpa_out"), /*mask_spm=*/0,
                /*MASK_NONE=*/0, attn_scale, /*batch=*/1,
                seq_len, seq_len, nq, nq, hd, NUM_CORES);
        } else {
            LayoutContext spm_layout;
            spm_layout.attention_policy =
                AttentionExecutionPolicy::SPM_KV_BY_MHA;
            spm_layout.batch_size = ctx().batch_size;
            spm_layout.is_causal = ctx().is_causal;
            spm_layout.use_attn_mask = ctx().attention_mask.has_value();
            const bool raw_profile = self_raw_spm_eligible(
                ctx().stage_plan, spm_layout, ctx().position);
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                GR00T_DIT_SELF_DDR_ATTENTION_SITE,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                raw_profile
                    ? 0
                    : GR00T_DIT_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                {}, chunk.idx);
            rpu_launch_sdpa_spm_unified_kernel_v2(
                k_cache, v_cache, /*mask_type=*/0, attn_scale,
                addr_offset("q").value, addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value, /*sdpa_mask_off=*/0,
                seq_len, nq, nq, hd, /*kv_seq_len=*/seq_len,
                NUM_CORES, NUM_CORES);
        }

        finish_attn_and_ffn(layer_idx, chunk, lw, seq_len, h, FF, local_inter);
    }

    // ====================== CROSS-ATTN BLOCK (P4) ======================
    void build_cross_attn_block(int layer_idx, const ChunkInfo& chunk) {
        const auto& lw = layer_weights_[layer_idx];
        const int64_t seq_len     = chunk.len;          // query tokens (41)
        const int64_t S           = vl_seq_;            // external K/V tokens
        const int64_t h           = hidden_size();
        const int64_t nq          = num_q_heads();
        const int64_t hd          = head_dim();
        const int64_t FF          = intermediate_size();
        const int64_t local_inter = FF / NUM_CORES;
        TORCH_CHECK(S > 0, "gr00t_dit cross-attn block needs vl_embeds (vl_seq_>0)");

        if (chunk.idx == 0) {
            dit_adaln_gemv(addr(0, "cond"), lw.adaln_w, lw.adaln_b,
                           addr(0, "adaln"), addr(0, "bias_temp"), addr(0, "gemv_partial"), lw.adaln_ws);
        }
        const uint32_t scale = addr(0, "adaln");
        const uint32_t shift = scale + static_cast<uint32_t>(h * DWIDTH);

        if (!ctx().input_in_spm) emit_layer_input_dma(layer_idx, chunk);

        // AdaLayerNorm(norm1) -> norm_hidden (over the seq_len query tokens)
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual1"), addr(0, "norm_hidden"),
            scale, shift, seq_len, h, eps_, false, 0, NUM_CORES);

        // Q from norm_hidden (col-partition) + bias
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_CROSS_Q_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "q_bias"), lw.to_q_ws);

        // K/V from vl_embeds: linear([S,cross_dim], to_k/to_v[hidden,cross_dim]) -> [S,hidden]
        const uint32_t vl_input_addr =
            z2_bound_ ? z2_port_addr_ : addr(0, "vl_embeds");
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_CROSS_K_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            vl_input_addr, lw.to_k_w, addr(0, "k"),
            S, nq * hd, cross_dim_, 1, NUM_CORES, layer_addr(layer_idx, 0, "k_bias"), lw.to_k_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_CROSS_V_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            vl_input_addr, lw.to_v_w, addr(0, "v"),
            S, nq * hd, cross_dim_, 1, NUM_CORES, layer_addr(layer_idx, 0, "v_bias"), lw.to_v_ws);

        // Insert S external K/V at pos 0, then cross-SDPA (seq_q=41, kv=S, MASK_2D). v16-pad:
        // insert pad16(S) (zeroed tail) for the v16 insert; SDPA below still reads kv_seq_len=S.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const KvInsertSegmentPlan kv_plan = consume_kv_insert_plan(
            GR00T_DIT_CROSS_KV_INSERT_SITE, S, chunk.idx);
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache,
            addr_offset("k").value, addr_offset("v").value,
            nq, hd, NUM_CORES,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            /*spm_rows=*/kv_plan.physical_rows(), kv_plan);
        // GR00T mask alternation (dit.py:391): cross blocks with global idx%4==0 attend
        // the TEXT subset, else the IMAGE subset (attend_text_every_n_blocks=2).
        const bool image_mask = layer_idx % 4 != 0;
        const PreparedMask& pm = image_mask ? prepared_cross_mask_img_
                                           : prepared_cross_mask_;
        const bool retain_masks = ctx().has_complete_physical_manifest() &&
            ctx().find_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, GR00T_DIT_MASK_SCHEDULE_SITE,
                chunk.idx).selector ==
                static_cast<int64_t>(Gr00tDiTMaskSchedule::RETAIN_TEXT_AND_IMAGE);
        const char* mask_slot = retain_masks && image_mask
            ? "sdpa_mask_image" : "sdpa_mask";
        // Initialization is a fixed Graph position, never a host-loaded flag.
        // Each replay refreshes both stable ordinal DDR sources once; later
        // layers and denoise bodies use the separately retained SPM operands.
        if (!retain_masks || (ctx().body_iter == 0 &&
                              layer_idx == first_cross_mask_layer(image_mask))) {
            sdpa_dma_mask_to_spm(pm, addr_offset(mask_slot).value, seq_len, S, NUM_CORES);
        }
        const double attn_scale = 1.0 / std::sqrt(static_cast<double>(hd));
        const bool raw_attention = ctx().attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const int64_t attention_site = raw_attention
            ? GR00T_DIT_CROSS_RAW_ATTENTION_SITE
            : GR00T_DIT_CROSS_DDR_ATTENTION_SITE;
        const AttentionExecutionPolicy attention_policy =
            ctx().physical_attention_policy_for_site(
                attention_site, chunk.idx);
        if (attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            TORCH_CHECK(pm.mask_type == 4,
                        "GR00T DiT raw-SPM cross attention requires MASK_2D");
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                GR00T_DIT_CROSS_RAW_ATTENTION_SITE,
                static_cast<int64_t>(
                    AttentionExecutionPolicy::SPM_KV_BY_MHA),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_v_transpose_spm(
                addr(0, "v"), addr(0, "sdpa_tmp"),
                /*batch=*/1, S, nq, hd, NUM_CORES);
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q"), addr(0, "k"), addr(0, "sdpa_tmp"),
                addr(0, "sdpa_out"), addr(0, mask_slot), pm.mask_type,
                attn_scale, /*batch=*/1, seq_len, S,
                nq, nq, hd, NUM_CORES);
        } else {
            LayoutContext spm_layout;
            spm_layout.attention_policy =
                AttentionExecutionPolicy::SPM_KV_BY_MHA;
            spm_layout.batch_size = ctx().batch_size;
            spm_layout.is_causal = ctx().is_causal;
            spm_layout.use_attn_mask = ctx().attention_mask.has_value();
            const bool raw_profile = cross_raw_spm_eligible(
                ctx().stage_plan, spm_layout, ctx().position);
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                GR00T_DIT_CROSS_DDR_ATTENTION_SITE,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                raw_profile
                    ? 0
                    : GR00T_DIT_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED,
                {}, chunk.idx);
            rpu_launch_sdpa_spm_unified_kernel_v2(
                k_cache, v_cache, /*mask_type=*/pm.mask_type, attn_scale,
                addr_offset("q").value, addr_offset("sdpa_out").value,
                addr_offset("sdpa_tmp").value,
                addr_offset(mask_slot).value,
                seq_len, nq, nq, hd, /*kv_seq_len=*/S,
                NUM_CORES, NUM_CORES);
        }

        finish_attn_and_ffn(layer_idx, chunk, lw, seq_len, h, FF, local_inter);
    }

    // ---- shared tail: o-proj + plain residual + norm3 + GELU-FFN + plain residual ----
    void finish_attn_and_ffn(int layer_idx, const ChunkInfo& chunk, const LayerWeights& lw,
                             int64_t seq_len, int64_t h, int64_t FF, int64_t local_inter) {
        const int64_t nq = num_q_heads();
        const int64_t hd = head_dim();

        // o_proj (row-partition) + bias + PLAIN residual (no gate)
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_O_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.to_out_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, NUM_CORES, layer_addr(layer_idx, 0, "out_bias"), lw.to_out_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            GR00T_DIT_ATTN_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
            seq_len, h, NUM_CORES, NUM_CORES);

        // norm3 = LayerNorm(affine=False) -> norm_hidden
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual2"), addr(0, "norm_hidden"),
            addr(0, "ln_ones"), addr(0, "ln_zeros"),
            seq_len, h, 1e-5, false, 0, NUM_CORES);

        // FFN: ff1(col)+bias -> GELU(tanh-approx) -> ff2(row)+bias + PLAIN residual
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_FF1_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.ff1_w, addr(0, "ff_mid"),
            seq_len, FF, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "ff1_bias"), lw.ff1_ws);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "ff_mid"), addr(0, "ff_mid"),
            seq_len * local_inter, ValuOpType::ADD,
            GeluMode::TANH, NUM_CORES);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_FF2_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "ff_mid"), lw.ff2_w, addr(0, "oproj"),
            seq_len, h, FF, 0, NUM_CORES, layer_addr(layer_idx, 0, "ff2_bias"), lw.ff2_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE, GR00T_DIT_FF_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(seq_len, h),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual2"), addr(0, "residual1"),
            seq_len, h, NUM_CORES, NUM_CORES);

        if (!ctx().output_to_spm) emit_layer_output_dma(layer_idx, chunk);
    }

    // ---- AdaLN modulation GEMV: cond * dense_w + bias -> [scale|shift] (2*h) ----
    void dit_adaln_gemv(uint32_t cond_spm_addr,
                        const at::Tensor& dense_w, const at::Tensor& dense_b,
                        uint32_t gemv_out_spm_addr,
                        uint32_t bias_temp_spm_addr,
                        uint32_t partial_spm_addr,
                        const at::Tensor& scale = {})
    {
        const int64_t h = hidden_size();
        const int64_t out_features = 2 * h;
        rpu_launch_ddr_broadcast_spm_dma(
            dense_b, /*src_offset_elements=*/0,
            out_features, bias_temp_spm_addr);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_ADALN_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            adaln_route_arguments(denoise_mode_, loop_mode_, vl_seq_, step_idx_));
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, NUM_CORES, /*bias_spm_addr=*/0, scale);
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            GR00T_DIT_ADALN_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(
                /*rows=*/1, /*cols=*/out_features),
            /*resolved_flags=*/0);
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
            /*M=*/1, /*N=*/out_features, NUM_CORES, NUM_CORES);
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0), gemv_out_spm_addr, h, ValuOpType::ADD);
    }

    // ============== Phase B/C — pre-hook: encoders -> sa_embs (emb_stage_) + cond ==============
    // All single-core (core 0); the cond scatter (8-core) feeds the multi-core DiT adaln.
    // step = step_idx_ (step form) or ctx().body_iter (unroll). See build_denoise_step.py design.
    void emit_pre_layers_body() {
        const int64_t h = hidden_size();
        const int64_t AH = action_horizon_, NP = action_dim_pad_, EM = enc_mid_;
        const int64_t bit = ctx().body_iter;
        const int64_t step = loop_mode_ ? bit : step_idx_;
        const int64_t cond_off = step * h;
        const uint32_t tau_step = addr(0, "tau_table") + static_cast<uint32_t>(step * AH * h * DWIDTH);
        const int64_t local_k = h / NUM_CORES;

        // cond[step]: 8-core scatter -> "cond" (DiT adaln) + core0 broadcast -> "cond_full" (norm_out)
        rpu_launch_ddr_scatter_spm_dma(
            cond_table_t_, cond_off, local_k, local_k * DWIDTH,
            addr(0, "cond"), NUM_CORES);
        rpu_launch_ddr_broadcast_spm_dma(
            cond_table_t_, cond_off, h, addr(0, "cond_full"),
            /*num_cores=*/1);

        if (bit == 0) {
            // x_t (actions, zero-padded [AH,NP]) -> x_t_spm (mutable; iter-0 only — iters 1..N-1
            // read the prior post-hook's in-SPM Euler result).
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                GR00T_DIT_ACTION_DMA_SITE,
                static_cast<int64_t>(
                    Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                actions_src_base_, actions_ref_, 0, AH * NP,
                addr(0, "x_t_spm"), /*num_cores=*/1);
            // state encoder (M=1): se_mid = relu(state @ se1 + b1); state_feat = se_mid @ se2 + b2
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                GR00T_DIT_STATE_DMA_SITE,
                static_cast<int64_t>(
                    Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0);
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                state_src_base_, state_ref_, 0, NP, addr(0, "enc0"), 1);
            sc_lin(addr(0, "enc0"), se1_w_, addr(0, "se_mid"), 1, EM, NP, addr(0, "se1_bias"));
            rpu_launch_eltwise_binary_scalar_spm_kernel(addr(0, "se_mid"), c10::Half(0.0),
                                                        addr(0, "se_mid"), EM, ValuOpType::MAX);
            sc_lin(addr(0, "se_mid"), se2_w_, addr(0, "state_feat"), 1, h, EM, addr(0, "se2_bias"), se2_ws_);
        }

        // action encoder: a_emb = x_t @ ae1 + b1 ; xx = silu(a_emb@W2a + b2 + tau@W2t) ; af = xx@W3 + b3 + pos
        sc_lin(addr(0, "x_t_spm"), ae1_w_, addr(0, "enc0"), AH, h, NP, addr(0, "ae1_bias"));
        sc_lin(addr(0, "enc0"), ae2a_w_, addr(0, "enc1"), AH, h, h, addr(0, "ae2_bias"), ae2a_ws_);   // + b2 (folded)
        sc_lin(tau_step,        ae2t_w_, addr(0, "enc2"), AH, h, h, /*bias=*/0, ae2t_ws_);
        rpu_launch_eltwise_binary_spm_kernel(addr(0, "enc1"), addr(0, "enc2"), addr(0, "enc1"),
                                             AH * h, ValuOpType::ADD, c10::Half(1.0), 1);
        rpu_launch_eltwise_unary_spm_kernel(addr(0, "enc1"), addr(0, "enc1"),
                                            AH * h, ValuOpType::SILU, /*is_gelu=*/false, 1);
        sc_lin(addr(0, "enc1"), ae3_w_, addr(0, "enc0"), AH, h, h, addr(0, "ae3_bias"), ae3_ws_);
        rpu_launch_eltwise_binary_spm_kernel(addr(0, "enc0"), addr(0, "pos_table"), addr(0, "enc0"),
                                             AH * h, ValuOpType::ADD, c10::Half(1.0), 1);

        // sa_embs -> emb_stage_ (token-dim cat: row0 = state_feat, rows 1..AH = af)
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "state_feat"), emb_stage_,
            /*dst_offset_elements=*/0, h);
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "enc0"), emb_stage_,
            /*dst_offset_elements=*/h, AH * h);
    }

    // ============== Phase B/C — post-hook: norm_out -> decoder -> in-graph Euler ==============
    void emit_post_layers_body() {
        const int64_t h = hidden_size();
        const int64_t AH = action_horizon_, SQ = AH + 1, NP = action_dim_pad_;
        const int64_t DM = dec_mid_, MO = model_out_dim_;

        // norm_out (DISTINCT from DiT AdaLN): so = cond_full @ po1 + b1 ; shift=so[:h], scale=so[h:].
        // Fold the modulate into ONE LN: gamma=(1+scale), beta=shift, eps=1e-6, affine via so slices.
        sc_lin(addr(0, "cond_full"), po1_w_, addr(0, "so"), 1, 2 * h, h, addr(0, "po1_bias"), po1_ws_);
        const uint32_t shift_a = addr(0, "so");
        const uint32_t scale_a = addr(0, "so") + static_cast<uint32_t>(h * DWIDTH);
        rpu_launch_eltwise_binary_scalar_spm_kernel(scale_a, c10::Half(1.0), scale_a, h, ValuOpType::ADD);
        rpu_launch_layernorm_spm_kernel(addr(0, "residual1"), addr(0, "normed"),
                                        scale_a, shift_a, SQ, h, 1e-6, false, 0, /*num_cores=*/1);
        sc_lin(addr(0, "normed"), po2_w_, addr(0, "model_out"), SQ, MO, h, addr(0, "po2_bias"), po2_ws_);

        // action_decoder: d1 = relu(model_out @ dec1 + b1) ; pred = d1 @ dec2 + b2 (N-pad 144)
        sc_lin(addr(0, "model_out"), dec1_w_, addr(0, "dec_mid"), SQ, DM, MO, addr(0, "dec1_bias"), dec1_ws_);
        rpu_launch_eltwise_binary_scalar_spm_kernel(addr(0, "dec_mid"), c10::Half(0.0),
                                                    addr(0, "dec_mid"), SQ * DM, ValuOpType::MAX);
        sc_lin(addr(0, "dec_mid"), dec2_w_, addr(0, "pred"), SQ, NP, DM, addr(0, "dec2_bias"), dec2_ws_);

        // in-graph Euler: x_t_spm[AH,NP] += dt · pred[1:SQ] (skip row0 = state token; pad cols stay 0)
        const uint32_t pred_act = addr(0, "pred") + static_cast<uint32_t>(NP * DWIDTH);
        rpu_launch_eltwise_binary_spm_kernel(addr(0, "x_t_spm"), pred_act, addr(0, "x_t_spm"),
                                             AH * NP, ValuOpType::ADD, dt_, /*num_cores=*/1);

        // output: x_t_spm -> x_out (mutable; fresh tracked tensor per forward)
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA, GR00T_DIT_OUTPUT_DMA_SITE,
            static_cast<int64_t>(Gr00tDiTMutableDmaRoute::SPM_TO_DDR),
            /*resolved_flags=*/0);
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "x_t_spm"), x_out_dst_base_, x_out_ref_,
            /*dst_offset_bytes=*/0, AH * NP);
    }

    SdpaConfig make_sdpa_config(int mask = 0) const {
        return {sdpa_kernel_, head_dim(), num_q_heads(), num_q_heads(),
                attn_tp(), mask};
    }

    // Text/image masks can have identical shapes but different contents. The
    // shared cache's ordinal gives each operand its own stable DDR address,
    // so preparing image never overwrites text and needs no second DDR copy.
    PreparedMask prep_mask_into(const std::optional<at::Tensor>& m, int64_t seq_q,
                                at::Tensor& slot, int64_t ordinal) {
        PreparedMask pm = sdpa_prepare_mask(
            m, /*is_causal=*/false, seq_q, vl_seq_,
            sdpa_stable_mask_cache(), ordinal);
        slot = pm.mask_type == 4 ? pm.ddr_tensor : at::Tensor{};
        return pm;
    }

private:
    std::vector<int64_t> kvinsert_cost_weight_identity() const override {
        if (layer_weights_.empty()) return {};
        std::vector<int64_t> identity{1};
        append_kvinsert_cost_scalar_identity(identity, eps_);
        identity.insert(identity.end(), {
            static_cast<int64_t>(denoise_ready_)});
        identity.push_back(static_cast<int64_t>(layer_weights_.size()));
        for (const auto& weights : layer_weights_) {
            for (const auto* tensor : {
                    &weights.to_q_w, &weights.to_k_w, &weights.to_v_w, &weights.to_out_w,
                    &weights.to_q_b, &weights.to_k_b, &weights.to_v_b, &weights.to_out_b,
                    &weights.ff1_w, &weights.ff2_w, &weights.ff1_b, &weights.ff2_b,
                    &weights.adaln_w, &weights.adaln_b, &weights.to_q_ws, &weights.to_k_ws,
                    &weights.to_v_ws, &weights.to_out_ws, &weights.ff1_ws, &weights.ff2_ws,
                    &weights.adaln_ws}) {
                append_kvinsert_cost_tensor_identity(identity, *tensor);
            }
        }
        for (const auto* tensor : {
                &se1_w_, &se1_b_, &se2_w_, &se2_b_,
                &ae1_w_, &ae1_b_, &ae2a_w_, &ae2t_w_,
                &ae2_b_, &ae3_w_, &ae3_b_, &dec1_w_,
                &dec1_b_, &dec2_w_, &dec2_b_, &po1_w_,
                &po1_b_, &po2_w_, &po2_b_, &se2_ws_,
                &ae2a_ws_, &ae2t_ws_, &ae3_ws_, &dec1_ws_,
                &dec2_ws_, &po1_ws_, &po2_ws_}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        return identity;
    }

    std::vector<LayerWeights> layer_weights_;
    double eps_ = 1e-5;
    int64_t cross_dim_ = 2048;
    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;

    // norm3 affine=False constants (gamma=1, beta=0), DDR-resident on rpu.
    at::Tensor ln_ones_, ln_zeros_;

    // Per-forward cond (= SiLU(temb)) state (mirrors AdaRMS).
    at::Tensor cond_ref_;
    uint64_t   cond_src_base_ = 0;
    bool       cond_loaded_this_forward_ = false;

    // Per-forward vl_embeds (cross-attn K/V source) + prepared MASK_2D.
    at::Tensor   vl_ref_;
    uint64_t     vl_src_base_ = 0;
    int64_t      vl_seq_ = 0;
    bool         vl_loaded_this_forward_ = false;
    PreparedMask prepared_cross_mask_;       // text subset (L%4==0)
    PreparedMask prepared_cross_mask_img_;   // image subset (L%4!=0)
    at::Tensor   text_mask_slot_;            // dedicated DDR slots (avoid shared-slot alias)
    at::Tensor   image_mask_slot_;

    // ---- Phase B/C denoise glue (set via set_denoise_weights) ----
    bool    denoise_ready_ = false;          // set_denoise_weights called
    bool    denoise_mode_  = false;          // step_forward path active (vs plain gr00t_dit_forward)
    bool    loop_mode_     = false;          // Phase C in-graph unroll (body_iterations = num_steps_)
    int64_t step_idx_      = 0;              // step form: which baked cond/tau
    int64_t num_steps_     = 4;
    int64_t action_dim_     = 132, action_dim_pad_ = 144, action_horizon_ = 40;
    int64_t enc_mid_ = 1024, dec_mid_ = 1024, model_out_dim_ = 1024;
    c10::Half dt_ = c10::Half(0.25f);
    at::Tensor se1_w_, se1_b_, se2_w_, se2_b_;
    at::Tensor ae1_w_, ae1_b_, ae2a_w_, ae2t_w_, ae2_b_, ae3_w_, ae3_b_;
    at::Tensor dec1_w_, dec1_b_, dec2_w_, dec2_b_;
    at::Tensor po1_w_, po1_b_, po2_w_, po2_b_;
    // W8A16 (opt-in): per-output-channel fp16 scales for the 8 K%32==0 glue GEMMs (undefined = fp16).
    // se1/ae1 stay fp16 (K=144 fails the int8 col-swizzle's K%32 alignment).
    at::Tensor se2_ws_, ae2a_ws_, ae2t_ws_, ae3_ws_, dec1_ws_, dec2_ws_, po1_ws_, po2_ws_;
    at::Tensor cond_table_t_, tau_table_, pos_table_;
    at::Tensor emb_stage_;                   // stable DDR staging (sa_embs -> layer-0 input)
    at::Tensor actions_ref_, state_ref_;     // mutable-DMA keepalives (held across forward)
    at::Tensor x_out_ref_;                   // fresh mutable-dst owner; prior held through allocation
    uint64_t   actions_src_base_ = 0, state_src_base_ = 0, x_out_dst_base_ = 0;
    bool z2_bound_ = false;
    int64_t z2_prepared_seq_len_ = 0;
    uint32_t z2_port_addr_ = 0;
    uint64_t z2_epoch_ = 0;
    uint64_t z2_plan_hash_ = 0;
    bool z2_masks_prepared_ = false;
    uint64_t z2_masks_plan_hash_ = 0;
    std::shared_ptr<uint64_t> z2_mask_generation_ =
        std::make_shared<uint64_t>(0);
    uint64_t z2_masks_prepared_generation_ = 0;
    std::shared_ptr<uint64_t> z2_masks_consumed_generation_ =
        std::make_shared<uint64_t>(0);
};

}  // namespace v3

// =============================================================================
// Instance registry + public C API (mirror rpu_adarms_*).
// =============================================================================
using Gr00tDiTRegistry = ModelHandleRegistry<v3::Gr00tDiTModel>;

std::vector<int64_t> rpu_gr00t_denoise_planner_cache_identity(int64_t handle) {
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_denoise_planner_cache_identity")
        ->planner_cache_identity();
}

std::vector<int64_t> rpu_gr00t_dit_planner_cache_identity(int64_t handle) {
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_gr00t_dit_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk) {
    Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_set_chunk_envelope")
        ->set_cache_envelope(max_kv_len, chunk);
}

void rpu_gr00t_dit_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_gr00t_dit_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_gr00t_dit_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_kvinsert_cost_domain")
        ->kvinsert_cost_domain("gr00t_dit", descriptor);
}

std::string rpu_gr00t_dit_kvinsert_cost_catalog_sha256(int64_t handle) {
    return Gr00tDiTRegistry::get(
        handle, "rpu_gr00t_dit_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

int64_t rpu_gr00t_dit_create() {
    return Gr00tDiTRegistry::create();
}

void rpu_gr00t_dit_destroy(int64_t handle) {
    v3::gr00t_z2_internal::check_denoise_destroy_allowed(handle);
    v3::gr00t_z2_internal::check_denoise_component_destroy_allowed(handle);
    Gr00tDiTRegistry::destroy(handle, "rpu_gr00t_dit_destroy");
}

void rpu_gr00t_dit_set_weights(
    int64_t handle,
    at::TensorList to_q_list, at::TensorList to_k_list,
    at::TensorList to_v_list, at::TensorList to_out_list,
    at::TensorList ff1_list, at::TensorList ff2_list,
    at::TensorList adaln_w_list, at::TensorList adaln_b_list,
    at::TensorList to_q_b_list, at::TensorList to_k_b_list,
    at::TensorList to_v_b_list, at::TensorList to_out_b_list,
    at::TensorList ff1_b_list, at::TensorList ff2_b_list,
    at::IntArrayRef block_is_cross,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t ff_inter, int64_t cross_dim,
    double eps)
{
    // fp16 path: set_weights' scale params default to empty.
    Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit")->set_weights(
        to_q_list, to_k_list, to_v_list, to_out_list,
        ff1_list, ff2_list, adaln_w_list, adaln_b_list,
        to_q_b_list, to_k_b_list, to_v_b_list, to_out_b_list,
        ff1_b_list, ff2_b_list,
        block_is_cross,
        num_heads, head_dim, hidden_size, ff_inter, cross_dim, eps);
}

// W8A16 variant: required per-output-channel scale lists for the 6 quantized GEMMs (separate op
// because aten schemas can't default Tensor[] args; mirrors causal_decoder_set_weights_w8a16).
void rpu_gr00t_dit_set_weights_w8a16(
    int64_t handle,
    at::TensorList to_q_list, at::TensorList to_k_list,
    at::TensorList to_v_list, at::TensorList to_out_list,
    at::TensorList ff1_list, at::TensorList ff2_list,
    at::TensorList adaln_w_list, at::TensorList adaln_b_list,
    at::TensorList to_q_b_list, at::TensorList to_k_b_list,
    at::TensorList to_v_b_list, at::TensorList to_out_b_list,
    at::TensorList ff1_b_list, at::TensorList ff2_b_list,
    at::IntArrayRef block_is_cross,
    int64_t num_heads, int64_t head_dim,
    int64_t hidden_size, int64_t ff_inter, int64_t cross_dim,
    double eps,
    at::TensorList to_q_scale_list, at::TensorList to_k_scale_list,
    at::TensorList to_v_scale_list, at::TensorList to_out_scale_list,
    at::TensorList ff1_scale_list, at::TensorList ff2_scale_list,
    at::TensorList adaln_scale_list)
{
    Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit")->set_weights(
        to_q_list, to_k_list, to_v_list, to_out_list,
        ff1_list, ff2_list, adaln_w_list, adaln_b_list,
        to_q_b_list, to_k_b_list, to_v_b_list, to_out_b_list,
        ff1_b_list, ff2_b_list,
        block_is_cross,
        num_heads, head_dim, hidden_size, ff_inter, cross_dim, eps,
        to_q_scale_list, to_k_scale_list, to_v_scale_list,
        to_out_scale_list, ff1_scale_list, ff2_scale_list, adaln_scale_list);
}

at::Tensor rpu_gr00t_dit_forward(
    int64_t handle,
    const at::Tensor& sa_embs,
    const at::Tensor& cond,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const std::optional<at::Tensor>& attention_mask,
    const std::optional<at::Tensor>& vl_embeds,
    const std::optional<at::Tensor>& cross_mask,
    const std::optional<at::Tensor>& cross_mask_img,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit")->forward(
        sa_embs, cond, k_caches, v_caches, attention_mask, vl_embeds,
        cross_mask, cross_mask_img, planned_stage_descriptor);
}

std::vector<int64_t> rpu_gr00t_dit_resolve_stage_domain(
        int64_t handle, int64_t rows, int64_t cross_seq_len,
        const std::optional<at::Tensor>& attention_mask) {
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_resolve_stage_domain")
        ->resolve_dit_stage_domain(rows, cross_seq_len, attention_mask);
}

// ---- Phase B: gr00t_denoise (extends the DiT handle with the per-step glue) ----
void rpu_gr00t_denoise_set_weights(
    int64_t handle,
    const at::Tensor& se1_w, const at::Tensor& se1_b,
    const at::Tensor& se2_w, const at::Tensor& se2_b,
    const at::Tensor& ae1_w, const at::Tensor& ae1_b,
    const at::Tensor& ae2a_w, const at::Tensor& ae2t_w, const at::Tensor& ae2_b,
    const at::Tensor& ae3_w, const at::Tensor& ae3_b,
    const at::Tensor& dec1_w, const at::Tensor& dec1_b,
    const at::Tensor& dec2_w, const at::Tensor& dec2_b,
    const at::Tensor& po1_w, const at::Tensor& po1_b,
    const at::Tensor& po2_w, const at::Tensor& po2_b,
    const at::Tensor& cond_table, const at::Tensor& tau_table, const at::Tensor& pos_table,
    int64_t action_dim, int64_t action_dim_pad, int64_t num_steps, double dt,
    const std::optional<at::Tensor>& se2_scale, const std::optional<at::Tensor>& ae2a_scale,
    const std::optional<at::Tensor>& ae2t_scale, const std::optional<at::Tensor>& ae3_scale,
    const std::optional<at::Tensor>& dec1_scale, const std::optional<at::Tensor>& dec2_scale,
    const std::optional<at::Tensor>& po1_scale, const std::optional<at::Tensor>& po2_scale)
{
    auto u = [](const std::optional<at::Tensor>& t) { return t.has_value() ? *t : at::Tensor(); };
    Gr00tDiTRegistry::get(handle, "rpu_gr00t_denoise")->set_denoise_weights(
        se1_w, se1_b, se2_w, se2_b, ae1_w, ae1_b, ae2a_w, ae2t_w, ae2_b, ae3_w, ae3_b,
        dec1_w, dec1_b, dec2_w, dec2_b, po1_w, po1_b, po2_w, po2_b,
        cond_table, tau_table, pos_table, action_dim, action_dim_pad, num_steps, dt,
        u(se2_scale), u(ae2a_scale), u(ae2t_scale), u(ae3_scale),
        u(dec1_scale), u(dec2_scale), u(po1_scale), u(po2_scale));
}

at::Tensor rpu_gr00t_denoise_step_forward(
    int64_t handle,
    const at::Tensor& actions, const at::Tensor& state, int64_t step_idx,
    at::TensorList k_caches_list, at::TensorList v_caches_list,
    const at::Tensor& vl_embeds, const at::Tensor& tmask, const at::Tensor& imask,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_denoise")->step_forward(
        actions, state, step_idx, k_caches, v_caches, vl_embeds, tmask, imask,
        planned_stage_descriptor);
}

at::Tensor rpu_gr00t_denoise_unroll_forward(
    int64_t handle,
    const at::Tensor& noise, const at::Tensor& state,
    at::TensorList k_caches_list, at::TensorList v_caches_list,
    const at::Tensor& vl_embeds, const at::Tensor& tmask,
    const at::Tensor& imask,
    at::IntArrayRef planned_stage_descriptor)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_denoise")->unroll_forward(
        noise, state, k_caches, v_caches, vl_embeds, tmask, imask,
        planned_stage_descriptor);
}

std::vector<int64_t> rpu_gr00t_denoise_resolve_stage_domain(
        int64_t handle, int64_t action_rows, int64_t cross_seq_len,
        int64_t step_idx) {
    return Gr00tDiTRegistry::get(
               handle, "rpu_gr00t_denoise_resolve_stage_domain")
        ->resolve_denoise_stage_domain(action_rows, cross_seq_len, step_idx);
}

int64_t rpu_gr00t_denoise_get_resolved_chunk_size(int64_t handle) {
    return Gr00tDiTRegistry::get(
               handle, "rpu_gr00t_denoise_get_resolved_chunk_size")
        ->get_last_resolved_chunk_size();
}

namespace v3::gr00t_z2_internal {

SpmPipelineComponentLayout prepare_denoise(int64_t handle, int64_t seq_len) {
    return Gr00tDiTRegistry::get(handle, "gr00t_spm_z2_prepare_denoise")
        ->prepare_z2_layout(seq_len);
}

void unprepare_denoise(int64_t handle, int64_t seq_len) {
    Gr00tDiTRegistry::get(handle, "gr00t_spm_z2_unprepare_denoise")
        ->unprepare_z2_layout(seq_len);
}

SpmDense2DSpec denoise_port_spec(int64_t handle, int64_t seq_len) {
    return Gr00tDiTRegistry::get(handle, "gr00t_spm_z2_denoise_port_spec")
        ->z2_port_spec(seq_len);
}

void stage_denoise_outer_fast_component(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches) {
    Gr00tDiTRegistry::get(
        handle, "gr00t_spm_z2_stage_denoise_outer_fast_component")
        ->stage_z2_outer_fast_component(guard, k_caches, v_caches);
}

void stage_denoise_outer_fast_invocation_authority(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    uint64_t plan_hash) {
    Gr00tDiTRegistry::get(
        handle,
        "gr00t_spm_z2_stage_denoise_outer_fast_invocation_authority")
        ->stage_z2_outer_fast_invocation_authority(guard, plan_hash);
}

at::Tensor prepare_denoise_outer_fast_replay(
    int64_t handle,
    GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& noise,
    const at::Tensor& state,
    uint64_t epoch,
    uint64_t plan_hash) {
    return Gr00tDiTRegistry::get(
               handle, "gr00t_spm_z2_prepare_denoise_outer_fast_replay")
        ->prepare_z2_outer_fast_replay(
            guard, noise, state, epoch, plan_hash);
}

void prepare_denoise_masks(int64_t handle,
                           const at::Tensor& tmask,
                           const at::Tensor& imask,
                           uint64_t plan_hash) {
    Gr00tDiTRegistry::get(handle, "gr00t_spm_z2_prepare_denoise_masks")
        ->prepare_z2_masks(tmask, imask, plan_hash);
}

at::Tensor forward_denoise_prepared_masks(
    int64_t handle,
    const at::Tensor& noise,
    const at::Tensor& state,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    uint64_t epoch,
    uint64_t plan_hash) {
    validate_denoise_dispatch(handle, epoch, plan_hash);
    std::vector<at::Tensor> k_caches(
        k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(
        v_caches_list.begin(), v_caches_list.end());
    return Gr00tDiTRegistry::get(
               handle, "gr00t_spm_z2_denoise_prepared_masks")
        ->unroll_forward_z2(
            noise, state, k_caches, v_caches,
            /*tmask=*/at::Tensor{}, /*imask=*/at::Tensor{},
            epoch, plan_hash, /*use_prepared_masks=*/true);
}

void adopt_denoise(int64_t handle,
                   const SpmPipelineLease& lease,
                   const SpmTensorView& scratch) {
    Gr00tDiTRegistry::get(handle, "gr00t_spm_z2_adopt_denoise")
        ->adopt_z2_layout(lease, scratch);
}

void bind_denoise(int64_t handle,
                   const SpmPipelineLease& lease,
                   const SpmPortView& port) {
    Gr00tDiTRegistry::get(handle, "gr00t_spm_z2_bind_denoise")
        ->bind_z2_port(lease, port);
}

void validate_denoise(int64_t handle,
                      const SpmPipelineLease& lease) {
    Gr00tDiTRegistry::get(handle, "gr00t_spm_z2_validate_denoise")
        ->validate_z2_layout(lease);
}

void clear_denoise(int64_t handle, uint64_t epoch, uint64_t plan_hash) {
    Gr00tDiTRegistry::get(handle, "gr00t_spm_z2_clear_denoise")
        ->clear_z2_layout(epoch, plan_hash);
}

void check_denoise_component_destroy_allowed(int64_t handle) {
    Gr00tDiTRegistry::get(
        handle, "gr00t_spm_z2_check_denoise_component_destroy")
        ->check_z2_destroy_allowed();
}

}  // namespace v3::gr00t_z2_internal

at::Tensor rpu_gr00t_denoise_unroll_forward_z2(
    int64_t handle,
    const at::Tensor& noise,
    const at::Tensor& state,
    at::TensorList k_caches_list,
    at::TensorList v_caches_list,
    const at::Tensor& tmask,
    const at::Tensor& imask,
    int64_t epoch,
    int64_t plan_hash)
{
    v3::gr00t_z2_internal::validate_denoise_dispatch(
        handle, static_cast<uint64_t>(epoch),
        static_cast<uint64_t>(plan_hash));
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_denoise_z2")
        ->unroll_forward_z2(
            noise, state, k_caches, v_caches, tmask, imask,
            static_cast<uint64_t>(epoch), static_cast<uint64_t>(plan_hash));
}
