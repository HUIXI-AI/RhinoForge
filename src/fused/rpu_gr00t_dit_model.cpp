// rpu_gr00t_dit_model.cpp — GR00T-N1.7 flow-matching DiT action head
//                            (v3 FusedModelBase subsystem, "New fused subsystem")
//
// 32-layer AlternateVLDiT:
//   - self-attn block over the 41 state+action tokens and
//     cross-attn block (Q=41 tokens, K/V = vl_embeds projected by this block's to_k/to_v,
//     2048->1536, MASK_2D image/text subset). Both norm1 = AdaLayerNorm (timestep-cond).
//   - pre_layers_fn = per-embodiment encoders -> sa_embs; post_layers_fn =
//     norm_out + proj_out + decoder + Euler integrate; body_iterations = #steps.
//
// Current path:
//   Self-attn block: AdaLN -> self-SDPA(41,MASK_NONE) -> plain resid -> LN3 -> GELU-FFN.
//   Cross-attn block: AdaLN -> Q from norm_hidden, K/V = linear(vl_embeds,
//        to_k/to_v[1536,2048]) inserted into the per-layer cache -> cross-SDPA(seq_q=41,
//        kv_seq_len=S, MASK_2D) -> plain resid -> LN3 -> GELU-FFN. Block type is an EXPLICIT
//        per-layer flag (block_is_cross), NOT layer-parity (so single-block tests pick type).
//        The current interface accepts one cross_mask.
//
// Structural lineage: SigLIP encoder (KV-insert + SDPA + plain GELU FFN) + AdaRMS modulation
// GEMV. cond = SiLU(temb) host-side (mirrors AdaRMS). All DiT linears carry bias.
//
// set_weights ends with invalidate_model_state().

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
#include <cstdlib>   // std::getenv (gr00t_kvpad16_enabled) — don't rely on a transitive include
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

// RPU_GR00T_KVPAD16 — default OFF. Pad the denoise KV-insert sequence to the
// next 16 so the v16 path is used when sequence and position are aligned. The
// padded rows are zeroed and SDPA reads only kv_seq_len=true_seq.
// Mirrors the Pi05 RPU_PI05_KVINSERT_PAD16 path.
static bool gr00t_kvpad16_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* e = std::getenv("RPU_GR00T_KVPAD16");
        cached = (e && (e[0] == '1' || e[0] == 't' || e[0] == 'T')) ? 1 : 0;
    }
    return cached != 0;
}
static inline int64_t gr00t_pad16(int64_t n) { return ((n + 15) / 16) * 16; }

// W8A16 per-projection scale validation. The same four checks apply at every
// call site: list
// length, int8 weight dtype, 1D contiguous fp16 RPU scale, and
// scale.numel() == output channels. Checking only the list length is not enough —
// a wrong-length or wrong-dtype scale reaches PARALLEL_LINEAR_W8A16 as garbage
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

        invalidate_model_state();  // Last non-empty statement of set_weights.
    }

    // ------------------------------------------------------------------ //
    // set_denoise_weights — Phase B glue (state/action encoders, norm_out, decoder)
    // + baked cond/tau/pos tables. Folds runtime.py's per-step host-fp32 glue on-device.
    // All weights single-core col-swizzled ([out,in], TP=1) at emb=20; padded 132→144 where
    // noted. Call AFTER set_weights (32 DiT blocks).
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
        se2_ws_ = se2_scale; ae2a_ws_ = ae2a_scale; ae2t_ws_ = ae2t_scale; ae3_ws_ = ae3_scale;
        dec1_ws_ = dec1_scale; dec2_ws_ = dec2_scale; po1_ws_ = po1_scale; po2_ws_ = po2_scale;
        cond_table_t_ = cond_table; tau_table_ = tau_table; pos_table_ = pos_table;
        action_dim_ = action_dim; action_dim_pad_ = action_dim_pad;
        num_steps_ = num_steps; dt_ = c10::Half(static_cast<float>(dt));
        action_horizon_ = tau_table.size(1);                   // 40
        // Derive inner dims from the un-swizzled bias sizes (robust to arch changes).
        enc_mid_       = se1_b.size(0);                         // state_encoder layer1 out (1024)
        dec_mid_       = dec1_b.size(0);                        // action_decoder layer1 out (1024)
        model_out_dim_ = po2_b.size(0);                        // proj_out_2 out (1024)
        denoise_ready_ = true;
        invalidate_model_state();  // Must remain last.
    }

    // Single-core SPM Linear helper (col-partition; bias from SPM addr).
    // M=1 and larger shapes use the shared shape-selected auto-tile family.
    void sc_lin(uint32_t in_a, const at::Tensor& w, uint32_t out_a,
                int64_t M, int64_t N, int64_t K, uint32_t bias_a = 0,
                const at::Tensor& scale = {}) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            in_a, w, out_a, M, N, K, /*partition=*/1, /*num_cores=*/1,
            bias_a, /*force_gemm=*/false, scale);
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
        const std::optional<at::Tensor>& cross_mask_img)
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
            // ⚠️ sdpa_prepare_mask writes a stable slot keyed by (seq_q, seq_k_aligned), so two
            // same-shape masks alias and the second clobbers the first. Copy each into its own
            // dedicated member slot (alloc-once + copy_; stable
            // across REPLAY for the denoise loop).
            prepared_cross_mask_ = prep_mask_into(cross_mask, seq_q, text_mask_slot_);
            if (cross_mask_img.has_value() && cross_mask_img->defined())
                prepared_cross_mask_img_ = prep_mask_into(cross_mask_img, seq_q, image_mask_slot_);
            else
                prepared_cross_mask_img_ = prepared_cross_mask_;
        } else {
            vl_seq_ = 0;
        }

        return run_all_layers(sa_embs, k_caches, v_caches, attention_mask,
                              /*position=*/0, /*is_causal=*/false);
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
        const at::Tensor& tmask, const at::Tensor& imask)
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
        if (!denoise_mode_ || loop_mode_) { denoise_mode_ = true; loop_mode_ = false; invalidate_model_state(); }
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
        prepared_cross_mask_     = prep_mask_into(tmask, seq_q, text_mask_slot_);
        prepared_cross_mask_img_ = prep_mask_into(imask, seq_q, image_mask_slot_);

        run_all_layers(emb_stage_, k_caches, v_caches, std::nullopt, /*position=*/0, /*is_causal=*/false);
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
        const at::Tensor& vl_embeds, const at::Tensor& tmask, const at::Tensor& imask)
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
        if (!denoise_mode_ || !loop_mode_) { denoise_mode_ = true; loop_mode_ = true; invalidate_model_state(); }

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
        prepared_cross_mask_     = prep_mask_into(tmask, seq_q, text_mask_slot_);
        prepared_cross_mask_img_ = prep_mask_into(imask, seq_q, image_mask_slot_);

        run_all_layers(emb_stage_, k_caches, v_caches, std::nullopt, /*position=*/0, /*is_causal=*/false);
        rpu_ddr_flush(out.data_ptr<c10::Half>());
        return out;                                   // final x_t after num_steps Euler steps
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
        auto prepared = prepare_spm_pipeline_component(layout);
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
            prep_mask_into(tmask, seq_q, text_mask_slot_);
        PreparedMask prepared_image =
            prep_mask_into(imask, seq_q, image_mask_slot_);
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
            prepared_cross_mask_ = prep_mask_into(tmask, seq_q, text_mask_slot_);
            prepared_cross_mask_img_ =
                prep_mask_into(imask, seq_q, image_mask_slot_);
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
        return cfg;
    }

    // ---- SPM manifest (design §3). All LayerWide; activations tiny. ----
    std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override {
        const int64_t cs = ctx.chunk_size;
        const int64_t h  = hidden_size();          // 1536
        const int64_t nq = num_q_heads();          // 32
        const int64_t hd = head_dim();             // 48
        const int64_t FF = intermediate_size();    // 6144
        int64_t kv_seq = std::max<int64_t>(cs, vl_seq_);         // cross K/V hold S>cs tokens
        if (gr00t_kvpad16_enabled()) kv_seq = gr00t_pad16(kv_seq);  // room for the v16 padded insert

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
        const int64_t tmp = A(std::max(tmp_self, tmp_cross) * 32);

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

        // ---- Phase B/C denoise glue buffers (only in denoise mode) ----
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
            // tables broadcast to core 0 (read by single-core encoder GEMMs / adds).
            auto tbl = [&](const char* name, int64_t elems, at::Tensor Gr00tDiTModel::* fld) {
                BufferDecl d; d.name = name; d.size = A(elems * DWIDTH);
                d.storage = StorageClass::Persistent; d.scope = ALL;
                d.preload_callback = [this, elems, fld](FusedModelBase&, int, uint32_t a0) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        this->*fld, /*src_offset_elements=*/0, elems, a0,
                        /*num_cores=*/1);
                };
                v.push_back(d);
            };
            tbl("tau_table", num_steps_ * AH * h, &Gr00tDiTModel::tau_table_);
            tbl("pos_table", AH * h,              &Gr00tDiTModel::pos_table_);
            // glue biases broadcast to core 0 (single-core GEMM bias_spm_addr).
            auto dbias = [&](const char* name, int64_t elems, at::Tensor Gr00tDiTModel::* fld) {
                BufferDecl d; d.name = name; d.size = A(elems * DWIDTH);
                d.storage = StorageClass::Persistent; d.scope = ALL;
                d.preload_callback = [this, elems, fld](FusedModelBase&, int, uint32_t a0) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        this->*fld, /*src_offset_elements=*/0, elems, a0,
                        /*num_cores=*/1);
                };
                v.push_back(d);
            };
            dbias("se1_bias", EM, &Gr00tDiTModel::se1_b_);
            dbias("se2_bias", h,  &Gr00tDiTModel::se2_b_);
            dbias("ae1_bias", h,  &Gr00tDiTModel::ae1_b_);
            dbias("ae2_bias", h,  &Gr00tDiTModel::ae2_b_);
            dbias("ae3_bias", h,  &Gr00tDiTModel::ae3_b_);
            dbias("dec1_bias", DM, &Gr00tDiTModel::dec1_b_);
            dbias("dec2_bias", NP, &Gr00tDiTModel::dec2_b_);
            dbias("po1_bias", 2 * h, &Gr00tDiTModel::po1_b_);
            dbias("po2_bias", MO,  &Gr00tDiTModel::po2_b_);
        }
        return v;
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
        // Plain DiT path: cond is the forward input arg, scattered here. Denoise path
        // (Phase B/C): cond comes from the baked table, scattered by the pre-hook → skip.
        if (is_first && !denoise_mode_ && !cond_loaded_this_forward_) {
            const int64_t local_k = h / NUM_CORES;
            rpu_launch_ddr_scatter_spm_dma_mutable(
                cond_src_base_, cond_ref_, /*src_offset_bytes=*/0,
                /*elements_per_core=*/local_k, /*core_stride_bytes=*/local_k * DWIDTH,
                addr(0, "cond"), NUM_CORES);
            cond_loaded_this_forward_ = true;
        }
        if (is_first && vl_seq_ > 0 && !vl_loaded_this_forward_ && !z2_bound_) {
            // vl_embeds replicated to all cores (linear input reads full cross_dim).
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                vl_src_base_, vl_ref_, /*src_offset_bytes=*/0,
                vl_seq_ * cross_dim_, addr(0, "vl_embeds"), NUM_CORES);
            vl_loaded_this_forward_ = true;
        }

        if (layer_weights_[layer_idx].is_cross) build_cross_attn_block(layer_idx, chunk);
        else                                    build_self_attn_block(layer_idx, chunk);
    }

    // v16 KV-insert pad (RPU_GR00T_KVPAD16): zero the [true_seq, pad16) rows of the "k"/"v" SPM
    // source, then return pad16(true_seq) so the unified insert engages v16. The SDPA reads only
    // kv_seq_len=true_seq → the padded (zeroed) cache rows are ignored. Returns true_seq if disabled
    // or already 16-aligned.
    int64_t kv_insert_seq(int64_t true_seq) {
        if (!gr00t_kvpad16_enabled()) return true_seq;
        const int64_t padded = gr00t_pad16(true_seq);
        if (padded == true_seq) return true_seq;
        const int64_t local_qhd = (num_q_heads() * head_dim()) / NUM_CORES;
        const int64_t pad_elems = (padded - true_seq) * local_qhd;
        const uint32_t off = static_cast<uint32_t>(true_seq * local_qhd * DWIDTH);
        rpu_launch_memset_spm_multicore(addr(0, "k") + off, pad_elems);
        rpu_launch_memset_spm_multicore(addr(0, "v") + off, pad_elems);
        return padded;
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
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "q_bias"), false, lw.to_q_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "k_bias"), false, lw.to_k_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "v_bias"), false, lw.to_v_ws);

        // KV-insert(pos 0) + self-SDPA (non-causal, MASK_NONE). v16-pad: insert pad16(seq_len)
        // (zeroed tail) so the unified insert uses v16; SDPA below still reads kv_seq_len=seq_len.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const int64_t ins_seq = kv_insert_seq(seq_len);
        rpu_launch_insert_kcache_spm_unified(
            k_cache, 0, addr_offset("k").value, ins_seq, nq, hd, NUM_CORES);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, 0, addr_offset("v").value, ins_seq, nq, hd, NUM_CORES);
        const double attn_scale = 1.0 / std::sqrt(static_cast<double>(hd));
        rpu_launch_sdpa_spm_unified_kernel_v2(
            k_cache, v_cache, /*mask_type=*/0, attn_scale,
            addr_offset("q").value, addr_offset("sdpa_out").value,
            addr_offset("sdpa_tmp").value, /*sdpa_mask_off=*/0,
            seq_len, nq, nq, hd, /*kv_seq_len=*/seq_len, NUM_CORES, NUM_CORES);

        finish_attn_and_ffn(layer_idx, chunk, lw, seq_len, h, FF, local_inter);
    }

    // ====================== CROSS-ATTN BLOCK ======================
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
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "q_bias"), false, lw.to_q_ws);

        // K/V from vl_embeds: linear([S,cross_dim], to_k/to_v[hidden,cross_dim]) -> [S,hidden]
        const uint32_t vl_input_addr =
            z2_bound_ ? z2_port_addr_ : addr(0, "vl_embeds");
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            vl_input_addr, lw.to_k_w, addr(0, "k"),
            S, nq * hd, cross_dim_, 1, NUM_CORES, layer_addr(layer_idx, 0, "k_bias"), false, lw.to_k_ws);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            vl_input_addr, lw.to_v_w, addr(0, "v"),
            S, nq * hd, cross_dim_, 1, NUM_CORES, layer_addr(layer_idx, 0, "v_bias"), false, lw.to_v_ws);

        // Insert S external K/V at pos 0, then cross-SDPA (seq_q=41, kv=S, MASK_2D). v16-pad:
        // insert pad16(S) (zeroed tail) for the v16 insert; SDPA below still reads kv_seq_len=S.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const int64_t ins_seq = kv_insert_seq(S);
        rpu_launch_insert_kcache_spm_unified(
            k_cache, 0, addr_offset("k").value, ins_seq, nq, hd, NUM_CORES);
        rpu_launch_insert_vcache_spm_unified(
            v_cache, 0, addr_offset("v").value, ins_seq, nq, hd, NUM_CORES);
        // GR00T mask alternation: cross blocks with global idx%4==0 attend
        // the TEXT subset, else the IMAGE subset (attend_text_every_n_blocks=2).
        const PreparedMask& pm = (layer_idx % 4 == 0) ? prepared_cross_mask_
                                                      : prepared_cross_mask_img_;
        sdpa_dma_mask_to_spm(pm, addr_offset("sdpa_mask").value, seq_len, S, NUM_CORES);
        const double attn_scale = 1.0 / std::sqrt(static_cast<double>(hd));
        rpu_launch_sdpa_spm_unified_kernel_v2(
            k_cache, v_cache, /*mask_type=*/pm.mask_type, attn_scale,
            addr_offset("q").value, addr_offset("sdpa_out").value,
            addr_offset("sdpa_tmp").value, addr_offset("sdpa_mask").value,
            seq_len, nq, nq, hd, /*kv_seq_len=*/S, NUM_CORES, NUM_CORES);

        finish_attn_and_ffn(layer_idx, chunk, lw, seq_len, h, FF, local_inter);
    }

    // ---- shared tail: o-proj + plain residual + norm3 + GELU-FFN + plain residual ----
    void finish_attn_and_ffn(int layer_idx, const ChunkInfo& chunk, const LayerWeights& lw,
                             int64_t seq_len, int64_t h, int64_t FF, int64_t local_inter) {
        const int64_t nq = num_q_heads();
        const int64_t hd = head_dim();

        // o_proj (row-partition) + bias + PLAIN residual (no gate)
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.to_out_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, NUM_CORES, layer_addr(layer_idx, 0, "out_bias"), false, lw.to_out_ws);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
            seq_len, h, NUM_CORES, NUM_CORES);

        // norm3 = LayerNorm(affine=False) -> norm_hidden
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual2"), addr(0, "norm_hidden"),
            addr(0, "ln_ones"), addr(0, "ln_zeros"),
            seq_len, h, 1e-5, false, 0, NUM_CORES);

        // FFN: ff1(col)+bias -> GELU(tanh-approx) -> ff2(row)+bias + PLAIN residual
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.ff1_w, addr(0, "ff_mid"),
            seq_len, FF, h, 1, NUM_CORES, layer_addr(layer_idx, 0, "ff1_bias"), false, lw.ff1_ws);
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "ff_mid"), addr(0, "ff_mid"),
            seq_len * local_inter, ValuOpType::ADD,
            GeluMode::TANH, NUM_CORES);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "ff_mid"), lw.ff2_w, addr(0, "oproj"),
            seq_len, h, FF, 0, NUM_CORES, layer_addr(layer_idx, 0, "ff2_bias"), false, lw.ff2_ws);
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
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, NUM_CORES, /*bias_spm_addr=*/0, /*force_gemm=*/false, scale);
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
            /*M=*/1, /*N=*/out_features, NUM_CORES, NUM_CORES);
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0), gemv_out_spm_addr, h, ValuOpType::ADD);
    }

    // ============== Phase B/C — pre-hook: encoders -> sa_embs (emb_stage_) + cond ==============
    // All single-core (core 0); the cond scatter (8-core) feeds the multi-core DiT adaln.
    // step = step_idx_ (step form) or ctx().body_iter (unroll).
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
            rpu_launch_ddr_broadcast_spm_dma_mutable(
                actions_src_base_, actions_ref_, 0, AH * NP,
                addr(0, "x_t_spm"), /*num_cores=*/1);
            // state encoder (M=1): se_mid = relu(state @ se1 + b1); state_feat = se_mid @ se2 + b2
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
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "x_t_spm"), x_out_dst_base_, x_out_ref_,
            /*dst_offset_bytes=*/0, AH * NP);
    }

    SdpaConfig make_sdpa_config(int mask = 0) const {
        return {sdpa_kernel_, head_dim(), num_q_heads(), num_q_heads(),
                attn_tp(), mask};
    }

    // Prepare a MASK_2D into a DEDICATED member slot (sdpa_prepare_mask's shared
    // stable slot is keyed by (seq_q,seq_k) and would alias two same-shape masks).
    PreparedMask prep_mask_into(const std::optional<at::Tensor>& m, int64_t seq_q,
                                at::Tensor& slot) {
        PreparedMask pm = sdpa_prepare_mask(
            m, /*is_causal=*/false, seq_q, vl_seq_,
            sdpa_stable_mask_cache());
        if (pm.mask_type == 4) {   // MASK_2D — copy out of the shared slot
            if (!slot.defined() || slot.sizes() != pm.ddr_tensor.sizes())
                slot = at::empty_like(pm.ddr_tensor);
            slot.copy_(pm.ddr_tensor);
            pm.ddr_tensor = slot;
        }
        return pm;
    }

private:
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
    const std::optional<at::Tensor>& cross_mask_img)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit")->forward(
        sa_embs, cond, k_caches, v_caches, attention_mask, vl_embeds, cross_mask, cross_mask_img);
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
    const at::Tensor& vl_embeds, const at::Tensor& tmask, const at::Tensor& imask)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_denoise")->step_forward(
        actions, state, step_idx, k_caches, v_caches, vl_embeds, tmask, imask);
}

at::Tensor rpu_gr00t_denoise_unroll_forward(
    int64_t handle,
    const at::Tensor& noise, const at::Tensor& state,
    at::TensorList k_caches_list, at::TensorList v_caches_list,
    const at::Tensor& vl_embeds, const at::Tensor& tmask, const at::Tensor& imask)
{
    std::vector<at::Tensor> k_caches(k_caches_list.begin(), k_caches_list.end());
    std::vector<at::Tensor> v_caches(v_caches_list.begin(), v_caches_list.end());
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_denoise")->unroll_forward(
        noise, state, k_caches, v_caches, vl_embeds, tmask, imask);
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
