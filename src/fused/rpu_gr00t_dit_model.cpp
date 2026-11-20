#include "rpu_specialized_linear_contracts.h"
// rpu_gr00t_dit_model.cpp — GR00T-N1.7 flow-matching DiT action head
//                            (v3 FusedModelBase subsystem, "New fused subsystem")
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
#include <c10/core/InferenceMode.h>
#include <c10/util/Half.h>
#include <c10/util/ScopeExit.h>
#include <array>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <limits>
#include <memory>
#include <tuple>
#include <string>
#include <vector>

using namespace at;
using namespace ::rhino_lkn;

#define DWIDTH 2

static inline int64_t gr00t_pad16(int64_t n) { return ((n + 15) / 16) * 16; }

// W8A16 per-projection scale validation. Same four checks as the already-strong
// sites (rpu_siglip_model.cpp:227, rpu_pi05_denoise_step_model.cpp:68): list
// length, int8 weight dtype, 1D contiguous fp16 device scale, and
// scale.numel() == output channels. Checking only the list length is not enough —
// a wrong-length or wrong-dtype scale reaches the generated W8A16 family as garbage
// per-channel multipliers, i.e. a silent numeric corruption rather than a crash.
// The `else` leg is the reverse hole: an int8 weight with no scale list is read
// back as fp16 by the kernel.
//
// W4A16 (RPU_GR00T_W4A16) block projections arrive as G32 pgrp-packed uint8
// [N,K/2] with the controller-striped 2D fp16 scale. `w4_partition` is the
// Linear partition the forward launches them with (1 column, 0 row); -1 means
// the projection has no W4 route. The launcher re-checks the same ABI on every
// call; this check fails at installation instead of on the first forward.
static void check_gr00t_w4a16_projection(const at::Tensor& w, const at::Tensor& s,
                                         int w4_partition, int64_t cores,
                                         const char* ctx, const char* name, int64_t i) {
    const int64_t n = w.size(0), k = w.size(1) * 2;
    const bool col = w4_partition == 1;
    TORCH_CHECK(cores > 0 && n % (col ? 16 * cores : 16) == 0 &&
                    k % (col ? 64 : 64 * cores) == 0,
                ctx, ": packed W4 ", name, "[", i, "] [", n, ",", k,
                "] does not split into 16-aligned N and 64-aligned K over ", cores,
                col ? " column" : " row", " lanes");
    const int64_t local_g = (col ? k : k / cores) / 32;
    const int64_t local_n = col ? n / cores : n;
    const int64_t expected = cores * ((local_g + 3) / 4) * ((local_n + 63) / 64) * 256;
    TORCH_CHECK(s.defined() && s.dim() == 2 && s.size(0) == 32 &&
                    s.scalar_type() == at::kHalf && s.device().type() == at::kPrivateUse1 &&
                    s.is_contiguous() && s.numel() == expected,
                ctx, ": packed W4 ", name, "_scale[", i, "] must be the contiguous fp16 RPU "
                "G32 pgrp payload of ", expected, " elements, got ", s.sizes());
}

static void check_gr00t_w8a16_scale_list(int64_t n_layers,
                                         const at::TensorList& weights,
                                         const at::TensorList& scales,
                                         const char* ctx, const char* name,
                                         int w4_partition = -1, int64_t cores = 0) {
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
            if (w4_partition >= 0 && w.scalar_type() == at::kByte) {
                check_gr00t_w4a16_projection(w, s, w4_partition, cores, ctx, name, i);
                continue;
            }
            TORCH_CHECK(w.scalar_type() == at::kChar,
                        ctx, ": quantized ", name, "[", i, "] must be int8",
                        w4_partition >= 0 ? " or packed uint8 W4" : "",
                        ", got ", w.scalar_type());
            TORCH_CHECK(s.defined() && s.dim() == 1
                        && s.scalar_type() == at::kHalf
                        && s.device().type() == at::kPrivateUse1
                        && s.is_contiguous(),
                        ctx, ": ", name, "_scale[", i,
                        "] must be 1D contiguous fp16 device tensor");
            TORCH_CHECK(s.numel() == w.size(0),
                        ctx, ": ", name, "_scale[", i, "].numel()=", s.numel(),
                        " != output dim=", w.size(0));
        } else {
            TORCH_CHECK(w.scalar_type() != at::kChar && w.scalar_type() != at::kByte,
                        ctx, ": quantized ", name, "[", i,
                        "] requires a non-empty scale list for this projection");
        }
    }
}

namespace v3 {

namespace {

// Stable descriptor identities. GR00T_DIT_FIXED_KERNEL_BASIS: every other
// launch is fixed checkpoint math (normalization, activation, W4 row bias,
// residual/Euler), unconditional parameter preload or graph output transfer.
// Denoise-conditioned table/bias preloads and the other named sites are routes.
constexpr int64_t GR00T_DIT_MASK_SCHEDULE_SITE = 4273136855374514935LL;
constexpr int64_t GR00T_DIT_ADALN_TABLE_SCHEDULE_SITE = 4251372394969008273LL;
constexpr int64_t GR00T_DIT_ADALN_TABLE_DMA_SITE = 6410548948878176467LL;
constexpr int64_t GR00T_DIT_TIME_TABLE_SCHEDULE_SITE = 5760993072901643056LL;
constexpr int64_t GR00T_DIT_TIME_TABLE_DMA_SITE = 6818841147164026446LL;
constexpr int64_t GR00T_DIT_FF1_RESIDENT_SCHEDULE_SITE = 4042616317377446460LL;
constexpr int64_t GR00T_DIT_FF1_RESIDENT_DMA_SITE = 719158113318470140LL;
constexpr int64_t GR00T_DIT_FF1_RESIDENT_LINEAR_SITE = 3484526589911248235LL;
constexpr int64_t GR00T_DIT_FF1_RESIDENT_BYTES = 1179648;
enum class Gr00tDiTMaskSchedule : int64_t {
    UPLOAD_EACH_CROSS_BLOCK = 1,
    RETAIN_TEXT_AND_IMAGE = 2,
};

constexpr int64_t GR00T_DIT_CROSS_V_TRANSPOSE_SCHEDULE_SITE = 5771321946979058017LL;
constexpr int64_t GR00T_DIT_CROSS_KV_SCHEDULE_SITE = 3879408591400702310LL;
enum class Gr00tDiTCrossKvSchedule : int64_t {
    PROJECT_EACH_BODY = 1,
    REUSE_FIRST_BODY_DDR = 2,
    REUSE_FIRST_BODY_SPM = 3,
};

bool gr00t_cross_kv_reuse_opt_in() {
    const char* value = std::getenv("RPU_GR00T_CROSS_KV_REUSE");
    if (value == nullptr || *value == '\0' || std::string(value) == "0") return false;
    TORCH_CHECK(std::string(value) == "1", "RPU_GR00T_CROSS_KV_REUSE must be 0 or 1");
    return true;
}

bool gr00t_kv_spm_only_opt_in() {
    const char* value = std::getenv("RPU_GR00T_KV_SPM_ONLY");
    if (value == nullptr || *value == '\0' || std::string(value) == "0") return false;
    TORCH_CHECK(std::string(value) == "1", "RPU_GR00T_KV_SPM_ONLY must be 0 or 1");
    return true;
}

bool gr00t_cross_v_transpose_reuse_opt_in() {
    const char* value = std::getenv("RPU_GR00T_CROSS_V_TRANSPOSE_REUSE");
    if (value == nullptr || *value == '\0' || std::string(value) == "0") return false;
    TORCH_CHECK(std::string(value) == "1", "RPU_GR00T_CROSS_V_TRANSPOSE_REUSE must be 0 or 1");
    return true;
}

bool gr00t_self_qkv_multibase_opt_in() {
    const char* value = std::getenv("RPU_GR00T_SELF_QKV_MULTIBASE");
    if (value == nullptr || *value == '\0' || std::string(value) == "0") return false;
    TORCH_CHECK(std::string(value) == "1", "RPU_GR00T_SELF_QKV_MULTIBASE must be 0 or 1");
    return true;
}

bool gr00t_ring_xor3_opt_in() {
    const char* value = std::getenv("RPU_GR00T_RING_XOR3");
    if (value == nullptr || *value == '\0' || std::string(value) == "0") return false;
    TORCH_CHECK(std::string(value) == "1", "RPU_GR00T_RING_XOR3 must be 0 or 1");
    return true;
}

bool gr00t_ff1_resident_w8_opt_in() {
    const char* value = std::getenv("RPU_GR00T_FF1_RESIDENT_W8");
    if (value == nullptr || *value == '\0' || std::string(value) == "0") return false;
    TORCH_CHECK(std::string(value) == "1", "RPU_GR00T_FF1_RESIDENT_W8 must be 0 or 1");
    return true;
}

constexpr int64_t GR00T_DIT_SC_LINEAR_SITE = 275862543721218402LL;
constexpr int64_t GR00T_DIT_COND_DMA_SITE = 830444116670174423LL;
constexpr int64_t GR00T_DIT_VL_DMA_SITE = 4588528469086136179LL;
constexpr int64_t GR00T_DIT_SELF_QKV_MULTIBASE_SITE = 7389640756804652104LL;
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
constexpr int64_t GR00T_DIT_ATTN_RING_XOR3_SITE = 7023018689830422266LL;
constexpr int64_t GR00T_DIT_FF_RING_XOR3_SITE = 7869571541108622222LL;
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
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 | KV_INSERT_CAP_PAD16 |
    KV_INSERT_CAP_NON8_TP_V16;
// Raw attention retains its DDR mirror unless the exact cold policy selects
// descriptor-owned SPM-only K/V for this invocation.
constexpr int64_t GR00T_DIT_KV_FLAG_DDR_MIRROR = 1;
constexpr int64_t GR00T_DIT_KV_FLAG_RAW_SPM_ONLY = 2;
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

    Gr00tDiTModel()
        : cross_kv_reuse_opt_in_(gr00t_cross_kv_reuse_opt_in()),
          kv_spm_only_opt_in_(gr00t_kv_spm_only_opt_in()),
          cross_v_transpose_reuse_opt_in_(gr00t_cross_v_transpose_reuse_opt_in()),
          ff1_resident_w8_opt_in_(gr00t_ff1_resident_w8_opt_in()),
          self_qkv_multibase_opt_in_(gr00t_self_qkv_multibase_opt_in()),
          ring_xor3_opt_in_(gr00t_ring_xor3_opt_in()) {}

    // Cold execution topology. Must precede set_weights: the Python adapter
    // swizzles every DiT Linear for this count. Eight keeps the historical
    // layout and every TP8 fast path. Four admits only the exact GR00T N1.7
    // DiT (32 blocks, 32 heads x D48, H1536, FF6144, cross 2048): 8 heads and
    // 1536 FF channels per core, DDR KV with eight physical stripes. Raw-SPM
    // attention, cross-KV reuse, the time/AdaLN tables, resident FF1, packed
    // QKV, the XOR3 ring, SPM-only KV and Z2 stay eight-core only.
    void set_execution_cores(int64_t cores) {
        TORCH_CHECK(cores == 4 || cores == 8,
                    "GR00T DiT execution cores must be 4 or 8");
        TORCH_CHECK(layer_weights_.empty() && !denoise_ready_ && !z2_bound_ &&
                        z2_prepared_seq_len_ == 0,
                    "GR00T DiT execution cores must precede weights");
        set_execution_core_count(static_cast<int>(cores));
    }

    std::vector<int64_t> execution_topology() const {
        TORCH_CHECK(!layer_weights_.empty(),
                    "GR00T DiT topology requires installed weights");
        return {num_cores(), attn_tp(), mlp_tp(),
                DecoderExecutionTopology::physical_kv_cores,
                num_q_heads(), intermediate_size()};
    }

    DecoderExecutionTopology resolve_model_execution_topology(
            int64_t nq, int64_t nkv, int64_t hd, int64_t h,
            int64_t intermediate) const override {
        if (num_cores() == 8)
            return FusedModelBase::resolve_model_execution_topology(
                nq, nkv, hd, h, intermediate);
        TORCH_CHECK(num_cores() == 4 && nq == 32 && nkv == 32 && hd == 48 &&
                        h == 1536 && intermediate == 6144,
                    "no admitted reduced-core GR00T DiT profile: 4 cores "
                    "require 32 heads, D48, H1536 and FF6144");
        return {4, 4, 4};
    }

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
        unroll_mask_publications_ = {};
        check_ff1_resident_setter_state();
        check_self_qkv_setter_state();
        check_ring_xor3_setter_state();
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
                            "] must be ", rank, "D contiguous device tensor");
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

        TORCH_CHECK(num_cores() == 8 || (N == 32 && cross_dim == 2048),
                    "reduced-core GR00T DiT requires 32 blocks and cross_dim 2048");
        // DiT: no GQA (num_kv_heads == num_q_heads), no separate intermediate for
        // the attention path; reuse the base param slots (intermediate_size = ff_inter).
        set_model_params(num_heads, num_heads, head_dim, hidden_size, ff_inter);
        set_num_layers(N);
        eps_ = eps;
        cross_dim_ = cross_dim;

        layer_weights_.clear();
        self_qkv_multibase_ready_ = false;
        ring_xor3_ready_ = false;
        clear_adaln_table();
        clear_time_tables();
        clear_ff1_resident_weight();
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
        // W4A16 carries G32 pgrp-packed weights and striped scales through the same lists.
        // When the scale lists are empty the GEMMs stay fp16 (the rpu_linear kernel keys on the
        // weight dtype + a defined scale). The AdaLN GEMV has its own int8-only scale list.
        {
            const char* c = "gr00t_dit_set_weights";
            const int64_t cores = num_cores();
            // One precision for the six block projections of every layer: the scale
            // fan-out below and the W8-only fused admissions read it from one tensor.
            for (const at::TensorList* list : {&to_q_list, &to_k_list, &to_v_list,
                                               &to_out_list, &ff1_list, &ff2_list}) {
                for (int64_t i = 0; i < N; ++i) {
                    TORCH_CHECK((*list)[i].scalar_type() == to_q_list[0].scalar_type(),
                                "gr00t_dit_set_weights: block projections must share one precision, got ",
                                (*list)[i].scalar_type(), " in layer ", i, " and ",
                                to_q_list[0].scalar_type(), " for to_q[0]");
                }
            }
            check_gr00t_w8a16_scale_list(N, to_q_list,   to_q_scale_list,   c, "to_q",   1, cores);
            check_gr00t_w8a16_scale_list(N, to_k_list,   to_k_scale_list,   c, "to_k",   1, cores);
            check_gr00t_w8a16_scale_list(N, to_v_list,   to_v_scale_list,   c, "to_v",   1, cores);
            check_gr00t_w8a16_scale_list(N, to_out_list, to_out_scale_list, c, "to_out", 0, cores);
            check_gr00t_w8a16_scale_list(N, ff1_list,    ff1_scale_list,    c, "ff1",    1, cores);
            check_gr00t_w8a16_scale_list(N, ff2_list,    ff2_scale_list,    c, "ff2",    0, cores);
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

        install_ff1_resident_weight();
        install_self_qkv_multibase();
        install_ring_xor3();
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
        unroll_mask_publications_ = {};
        check_ff1_resident_setter_state();
        check_self_qkv_setter_state();
        check_ring_xor3_setter_state();
        TORCH_CHECK(num_layers() > 0, "set_denoise_weights must follow set_weights (32 DiT blocks)");
        TORCH_CHECK(action_dim_pad % 16 == 0, "action_dim_pad must be %16");
        TORCH_CHECK(cond_table.size(0) == num_steps && cond_table.size(1) == hidden_size(),
                    "cond_table must be [num_steps, hidden]");
        TORCH_CHECK(tau_table.size(0) == num_steps && tau_table.size(-1) == hidden_size(),
                    "tau_table must be [num_steps, AH, hidden]");
        clear_adaln_table();
        clear_time_tables();
        clear_ff1_resident_weight();
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
        install_ff1_resident_weight();
        invalidate_model_state();  // D-503
    }

    void prepare_adaln_table() {
        RpuExecutionCoordinator::require_graph_quiescent("gr00t_dit_prepare_adaln_table");
        RpuExecutionCoordinator::require_no_physical_claim("gr00t_dit_prepare_adaln_table");
        auto execution = RpuExecutionCoordinator::enter_physical("gr00t_dit_prepare_adaln_table");
        auto release = c10::make_scope_exit([&] {
            RpuExecutionCoordinator::exit_physical_noexcept(execution);
        });
        check_adaln_prepare_state();
        std::vector<at::Tensor> weights, biases, scales;
        for (const auto& layer : layer_weights_) {
            weights.push_back(layer.adaln_w);
            biases.push_back(layer.adaln_b);
            scales.push_back(layer.adaln_ws);
        }
        // Own the schedule bytes as well as its derived table. Later callers
        // cannot mutate the original cond tensor behind the cold table.
        auto cond = cond_table_t_.clone();
        auto table = rpu_gr00t_precompute_adaln(cond, weights, biases, scales, attn_tp());
        cond_table_t_ = std::move(cond);
        adaln_table_ = std::move(table);
        adaln_table_src_base_ = ::rhino_lkn::RpuGetDevAddr(adaln_table_.data_ptr());
        ++adaln_table_generation_;
        for (const auto& layer : layer_weights_) {
            for (const auto* tensor : {&layer.adaln_w, &layer.adaln_b, &layer.adaln_ws}) {
                if (!tensor->defined()) continue;
                const auto& version = tensor->unsafeGetTensorImpl()->version_counter();
                adaln_sources_.push_back({*tensor, static_cast<int64_t>(version.current_version())});
            }
        }
        invalidate_model_state();
    }

    void prepare_time_tables() {
        RpuExecutionCoordinator::require_graph_quiescent("gr00t_dit_prepare_time_tables");
        RpuExecutionCoordinator::require_no_physical_claim("gr00t_dit_prepare_time_tables");
        auto execution = RpuExecutionCoordinator::enter_physical("gr00t_dit_prepare_time_tables");
        auto release = c10::make_scope_exit([&] {
            RpuExecutionCoordinator::exit_physical_noexcept(execution);
        });
        check_time_prepare_state();
        auto cond = cond_table_t_.clone();
        auto tau = tau_table_.clone();
        auto tables = rpu_gr00t_precompute_time_glue(
            tau, cond, ae2t_w_, ae2t_ws_, po1_w_, po1_b_, po1_ws_);
        // Keep raw owned schedule inputs for plain/step/Z2 fallback.
        cond_table_t_ = std::move(cond);
        tau_table_ = std::move(tau);
        time_tau_table_ = std::move(std::get<0>(tables));
        time_po1_table_ = std::move(std::get<1>(tables));
        time_tau_src_base_ = ::rhino_lkn::RpuGetDevAddr(time_tau_table_.data_ptr());
        time_po1_src_base_ = ::rhino_lkn::RpuGetDevAddr(time_po1_table_.data_ptr());
        ++time_table_generation_;
        for (const auto* tensor : {&ae2t_w_, &ae2t_ws_, &po1_w_, &po1_b_, &po1_ws_}) {
            if (!tensor->defined()) continue;
            const auto& version = tensor->unsafeGetTensorImpl()->version_counter();
            time_sources_.push_back({*tensor, static_cast<int64_t>(version.current_version())});
        }
        invalidate_model_state();
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
        begin_adaln_inference();
        begin_time_inference();
        begin_ff1_resident_inference();
        TORCH_CHECK(!z2_bound_,
                    "gr00t_dit_forward: Z2 lease is active; use the Z2 unroll op");
        TORCH_CHECK(cond.defined() && cond.dim() == 1
                    && cond.size(0) == hidden_size()
                    && cond.device().type() == at::kPrivateUse1
                    && cond.is_contiguous(),
                    "gr00t_dit_forward: cond must be 1-D [hidden_size] contiguous device tensor");

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
                        "gr00t_dit_forward: vl_embeds must be [..,S,cross_dim] contiguous device tensor");
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
        begin_adaln_inference();
        begin_time_inference();
        begin_ff1_resident_inference();
        TORCH_CHECK(!z2_bound_,
                    "gr00t_denoise_step_forward: Z2 lease is active; use the Z2 unroll op");
        TORCH_CHECK(denoise_ready_, "step_forward before set_denoise_weights");
        TORCH_CHECK(actions_pad.dim() == 3 && actions_pad.size(1) == action_horizon_
                    && actions_pad.size(2) == action_dim_pad_ && actions_pad.scalar_type() == at::kHalf
                    && actions_pad.is_contiguous() && actions_pad.device().type() == at::kPrivateUse1,
                    "actions_pad must be [1,AH,action_dim_pad] fp16 contig device");
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
                    && vl_embeds.is_contiguous(), "vl_embeds must be [..,S,cross_dim] contig device");
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
        begin_adaln_inference();
        begin_time_inference();
        begin_ff1_resident_inference();
        TORCH_CHECK(!z2_bound_,
                    "gr00t_denoise_unroll_forward: Z2 lease is active; use the Z2 unroll op");
        TORCH_CHECK(denoise_ready_, "unroll_forward before set_denoise_weights");
        TORCH_CHECK(noise_pad.dim() == 3 && noise_pad.size(1) == action_horizon_
                    && noise_pad.size(2) == action_dim_pad_ && noise_pad.scalar_type() == at::kHalf
                    && noise_pad.is_contiguous() && noise_pad.device().type() == at::kPrivateUse1,
                    "noise_pad must be [1,AH,action_dim_pad] fp16 contig device");
        // Node-count backstop (wall_oss precedent): num_steps bodies must stay well below the
        // 65536 batch-item cap. GR00T body ≈ 32 blocks + glue ≈ ~360 nodes → 4×≈1.4k. Loud-fail margin.
        TORCH_CHECK(num_steps_ * 2000 < 65536, "unroll num_steps too large for the kernel-batch cap");
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
                    && vl_embeds.is_contiguous(), "vl_embeds must be [..,S,cross_dim] contig device");
        vl_ref_ = vl_embeds; vl_seq_ = vl_embeds.size(-2);
        vl_src_base_ = ::rhino_lkn::RpuGetDevAddr(vl_embeds.data_ptr());
        vl_loaded_this_forward_ = false;
        prepare_unroll_masks(tmask, imask, seq_q);

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
        TORCH_CHECK(num_cores() == 8,
                    "gr00t denoise Z2 is the eight-core profile only");
        if (!denoise_mode_ || !loop_mode_) {
            denoise_mode_ = true;
            loop_mode_ = true;
            invalidate_model_state();
        }
        vl_seq_ = seq_len;
        // Freeze ordinary-versus-Z2 route eligibility before creating the
        // COMPLETE child descriptor. Failed preparation restores the old state.
        const int64_t previous_seq_len = z2_prepared_seq_len_;
        z2_prepared_seq_len_ = seq_len;
        bool prepared_committed = false;
        auto rollback_prepared = c10::make_scope_exit([&] {
            if (!prepared_committed) z2_prepared_seq_len_ = previous_seq_len;
        });
        LayoutContext layout;
        layout.chunk_size = action_horizon_ + 1;
        layout.max_kv_seq_len = action_horizon_ + 1;
        layout.num_layers = num_layers();
        layout.use_attn_mask = false;
        layout.is_causal = false;
        const auto stage_plan = compose_fmb_default_three_stage_chunk_plan(
            layout, action_horizon_ + 1, /*position=*/0, ChunkMode::SEQUENTIAL);
        const auto manifest = physical_manifest_for_candidate(
            stage_plan, layout, action_horizon_ + 1, action_horizon_ + 1, /*position=*/0);
        const int64_t chunk_capacity = ((layout.chunk_size + 15) / 16) * 16;
        auto descriptor = encode_fmb_prefill_stage_candidate({
            stage_plan.input.plan.chunk_size, chunk_capacity,
            chunk_capacity, stage_plan, manifest});
        const auto candidate = prepare_stage_candidate(descriptor);
        TORCH_CHECK(candidate->candidate().physical_manifest.state ==
                        FmbPhysicalManifestState::COMPLETE &&
                        candidate->candidate().physical_manifest.graph_lifecycle ==
                            FmbGraphLifecycle::COMPOSITE_CHILD,
                    "GR00T denoise Z2 requires COMPLETE child preparation");
        layout.stage_plan_fingerprint = candidate->stage_fingerprint();
        layout = bind_physical_layout_context(layout, manifest);
        auto prepared = prepare_spm_pipeline_component(layout, stage_plan);
        z2_stage_descriptor_ = std::move(descriptor);
        prepared_committed = true;
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
        z2_stage_descriptor_.clear();
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
        begin_adaln_inference();
        begin_time_inference();
        begin_ff1_resident_inference();
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
                    "noise_pad must be [1,AH,action_dim_pad] fp16 contig device");
        TORCH_CHECK(state_pad.defined() && state_pad.dim() == 3 &&
                        state_pad.size(0) == 1 && state_pad.size(1) == 1 &&
                        state_pad.size(2) == action_dim_pad_ &&
                        state_pad.scalar_type() == at::kHalf &&
                        state_pad.is_contiguous() &&
                        state_pad.device().type() == at::kPrivateUse1,
                    "state_pad must be [1,1,action_dim_pad] fp16 contig device");
        TORCH_CHECK(num_steps_ * 2000 < 65536,
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
                       /*position=*/0, /*is_causal=*/false,
                       /*planned_chunk_size=*/0, z2_stage_descriptor_);
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
        const LayoutContext& candidate_layout,
        int64_t physical_len, int64_t logical_len,
        int64_t position, bool self_raw_spm,
        bool cross_raw_spm) const {
        LayoutContext layout = candidate_layout;
        layout.attention_policy = self_raw_spm || cross_raw_spm
            ? AttentionExecutionPolicy::SPM_KV_BY_MHA
            : AttentionExecutionPolicy::DDR_KV;
        layout.forward_operand_residency = ff1_residency_for_candidate(layout);
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
        const auto append_kv = [&](int64_t site_id, int64_t rows, bool raw_attention) {
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
                raw_attention && kv_spm_only_profile(chunk.len, layout.batch_size)
                    ? GR00T_DIT_KV_FLAG_RAW_SPM_ONLY : GR00T_DIT_KV_FLAG_DDR_MIRROR,
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
            if (cross_v_transpose_reuse_opt_in_) {
                append(FmbRouteFamily::GRAPH_SCHEDULE,
                       GR00T_DIT_CROSS_V_TRANSPOSE_SCHEDULE_SITE,
                       cross_v_transpose_profile(chunk.len, layout.batch_size, cross_raw_spm) ? 2 : 1,
                       cross_v_transpose_arguments(chunk.len, layout.batch_size, cross_raw_spm));
            }
            if (cross_kv_reuse_opt_in_) {
                append(FmbRouteFamily::GRAPH_SCHEDULE,
                       GR00T_DIT_CROSS_KV_SCHEDULE_SITE,
                       cross_kv_schedule(chunk.len, layout.batch_size, cross_raw_spm),
                       cross_kv_schedule_arguments(chunk.len, layout.batch_size));
            }
        }

        if (!denoise_mode_) {
            append(FmbRouteFamily::MUTABLE_DMA, GR00T_DIT_COND_DMA_SITE,
                   static_cast<int64_t>(
                       Gr00tDiTMutableDmaRoute::DDR_SCATTER_TO_SPM));
        }
        if (vl_seq_ > 0 && !z2_bound_ && z2_prepared_seq_len_ == 0) {
            append(FmbRouteFamily::MUTABLE_DMA, GR00T_DIT_VL_DMA_SITE,
                   static_cast<int64_t>(
                       Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        }
        const bool time_tables = time_table_profile(chunk.len, layout.batch_size);
        if (time_tau_table_.defined() && denoise_mode_) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, GR00T_DIT_TIME_TABLE_SCHEDULE_SITE,
                   time_tables ? 2 : 1, time_table_arguments(chunk.len, layout.batch_size));
        }
        if (time_tables) {
            for (int64_t invocation = 0; invocation < 2 * num_steps_; ++invocation) {
                manifest.routes.push_back({GR00T_DIT_TIME_TABLE_DMA_SITE,
                    FmbRouteFamily::MUTABLE_DMA,
                    static_cast<int64_t>(Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM), 0,
                    time_table_arguments(chunk.len, layout.batch_size), invocation});
            }
        }
        const bool adaln_table = adaln_table_profile(chunk.len, layout.batch_size);
        if (adaln_table_.defined()) {
            append(FmbRouteFamily::GRAPH_SCHEDULE, GR00T_DIT_ADALN_TABLE_SCHEDULE_SITE,
                   adaln_table ? 2 : 1, adaln_table_arguments(chunk.len, layout.batch_size));
        }
        if (adaln_table) {
            for (int64_t body = 0; body < num_steps_; ++body) {
                manifest.routes.push_back({GR00T_DIT_ADALN_TABLE_DMA_SITE,
                    FmbRouteFamily::MUTABLE_DMA,
                    static_cast<int64_t>(Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM), 0,
                    adaln_table_arguments(chunk.len, layout.batch_size), body});
            }
        } else {
            append(FmbRouteFamily::LINEAR, GR00T_DIT_ADALN_LINEAR_SITE,
                   static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                   adaln_route_arguments(denoise_mode_, loop_mode_, vl_seq_, step_idx_));
            append(FmbRouteFamily::ALL_REDUCE, GR00T_DIT_ADALN_ALL_REDUCE_SITE,
                   fmb_ring_all_reduce_route_selector(/*rows=*/1, /*cols=*/2 * hidden_size(),
                                                      num_cores()));
        }
        if (has_self) {
            if (self_qkv_multibase_profile(chunk.len, layout.batch_size)) {
                append(FmbRouteFamily::LINEAR, GR00T_DIT_SELF_QKV_MULTIBASE_SITE,
                       /*MULTIBASE_M592N32K128=*/2,
                       self_qkv_multibase_arguments(chunk.len, layout.batch_size));
            } else {
                append_linear(GR00T_DIT_SELF_Q_LINEAR_SITE);
                append_linear(GR00T_DIT_SELF_K_LINEAR_SITE);
                append_linear(GR00T_DIT_SELF_V_LINEAR_SITE);
            }
            append_kv(GR00T_DIT_SELF_KV_INSERT_SITE, chunk.len, self_raw_spm);
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
            append_kv(GR00T_DIT_CROSS_KV_INSERT_SITE, vl_seq_, cross_raw_spm);
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
        if (ring_xor3_profile(chunk.len, layout.batch_size)) {
            append(FmbRouteFamily::ALL_REDUCE, GR00T_DIT_ATTN_RING_XOR3_SITE,
                   /*EXACT_XOR3=*/2,
                   ring_xor3_arguments(chunk.len, layout.batch_size, /*ff=*/false));
        } else {
            append(FmbRouteFamily::ALL_REDUCE,
                   GR00T_DIT_ATTN_ALL_REDUCE_SITE,
                   fmb_ring_all_reduce_route_selector(chunk.len, hidden_size(), num_cores()));
        }
        append_linear(GR00T_DIT_FF1_LINEAR_SITE);
        if (ff1_resident_w8_opt_in_ && denoise_mode_) {
            const bool resident = layout.forward_operand_residency ==
                FmbForwardOperandResidency::FORWARD;
            append(FmbRouteFamily::GRAPH_SCHEDULE, GR00T_DIT_FF1_RESIDENT_SCHEDULE_SITE,
                   resident ? 2 : 1, ff1_resident_arguments(chunk.len, layout.batch_size));
            if (resident) {
                append(FmbRouteFamily::MUTABLE_DMA, GR00T_DIT_FF1_RESIDENT_DMA_SITE,
                       static_cast<int64_t>(Gr00tDiTMutableDmaRoute::DDR_SCATTER_TO_SPM),
                       ff1_resident_arguments(chunk.len, layout.batch_size));
                append(FmbRouteFamily::LINEAR, GR00T_DIT_FF1_RESIDENT_LINEAR_SITE,
                       /*SPM_WEIGHT_M512N48K128=*/2,
                       ff1_resident_arguments(chunk.len, layout.batch_size));
            }
        }
        append_linear(GR00T_DIT_FF2_LINEAR_SITE);
        if (ring_xor3_profile(chunk.len, layout.batch_size)) {
            append(FmbRouteFamily::ALL_REDUCE, GR00T_DIT_FF_RING_XOR3_SITE,
                   /*EXACT_XOR3=*/2,
                   ring_xor3_arguments(chunk.len, layout.batch_size, /*ff=*/true));
        } else {
            append(FmbRouteFamily::ALL_REDUCE,
                   GR00T_DIT_FF_ALL_REDUCE_SITE,
                   fmb_ring_all_reduce_route_selector(chunk.len, hidden_size(), num_cores()));
        }
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
        // The cross-attention bridge is SPM-resident, while each body's
        // generated embeddings still enter layer zero through its input DMA.
        append_fmb_shared_runtime_routes(
            manifest, plan, hidden_size(),
            FMB_SHARED_LAYER_INPUT_DMA, /*compute_row_multiplier=*/1,
            num_cores(), mlp_tp());
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
        if ((num_cores() != 4 && num_cores() != 8) || z2_bound_ || z2_prepared_seq_len_ != 0 ||
            position != 0 || layout.is_causal || layout.use_attn_mask ||
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
            num_q_heads(), num_kv_heads(), head_dim(), num_cores(),
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
        if (num_cores() == 4 && cross_kv_reuse_profile(seq, layout.batch_size) &&
            2 * cross_kv_bank_bytes() > (num_steps_ <= 4 ? 4718592 : 3932160)) {
            return false;
        }
        return has_cross && vl_seq_ >= seq && sdpa_by_mha_spm_is_valid(
            /*batch=*/1, seq, vl_seq_,
            num_q_heads(), num_kv_heads(), head_dim(), num_cores(),
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

        const int64_t local_qhd   = (nq * hd) / num_cores();  // per-core attn width (192 at TP8)
        const int64_t local_inter = FF / num_cores();         // per-core FFN width (768 at TP8)
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
        append_cross_kv_buffers(v, ctx);
        if (adaln_table_profile(ctx.chunk_size, ctx.batch_size)) {
            v.push_back({"adaln_body_table", adaln_table_body_bytes(),
                         0, 0, StorageClass::Temp, 0, nullptr, ALL});
        }
        if (ctx.forward_operand_residency == FmbForwardOperandResidency::FORWARD) {
            TORCH_CHECK(ctx.chunk_size >= 41 && ff1_resident_profile(41, ctx.batch_size),
                        "GR00T resident FF1 layout requires eligible installed weights");
            // Live from body-0 preload through all admitted bodies. Never rely on
            // a previous request's Temp contents or allocate outside this plan.
            v.push_back({"ff1_resident_w8", GR00T_DIT_FF1_RESIDENT_BYTES,
                         0, 0, StorageClass::Temp, 0, nullptr, ALL});
        }
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
                    hidden_size(), core0_addr, num_cores());
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
                    per_core, per_core * DWIDTH, core0_addr, num_cores());
            };
            v.push_back(d);
        };
        auto row_bias = [&](const char* name, at::Tensor LayerWeights::* field) {
            BufferDecl d; d.name = name; d.size = A(h * DWIDTH);
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl; d.scope = ALL;
            d.preload_callback = [this, field](FusedModelBase&, int L, uint32_t core0_addr) {
                rpu_launch_memset_spm_multicore(core0_addr, hidden_size(), num_cores());
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
        // Prepared AdaLN table paths never read legacy AdaLN scratch.
        // Its extra 6144-byte phase-1 slot, together with bias/partial, can
        // cover a second mask without increasing the peak. Longer prefixes can
        // also share disjoint pre/post glue scratch below. The original upload
        // schedule remains the fallback if the actual no-growth proof fails.
        const bool long_mask_profile = vl_seq_ > 96 &&
            adaln_table_profile(layout.chunk_size, layout.batch_size);
        // Ordinary COMPLETE single-chunk public DiT geometry only. Z2 has an
        // independently sealed lease/register contract: its preparation and
        // prepared/bound executions retain the baseline scratch schedule.
        if (!complete || z2_bound_ || z2_prepared_seq_len_ != 0 ||
            layout.chunk_size != 41 || layout.batch_size != 1 ||
            layout.is_causal || layout.use_attn_mask ||
            hidden_size() != 1536 || intermediate_size() != 6144 ||
            num_q_heads() != 32 || head_dim() != 48 || cross_dim_ != 2048 ||
            vl_seq_ <= 0 || (vl_seq_ > 96 && !long_mask_profile) ||
            first_cross_mask_layer(false) < 0 || first_cross_mask_layer(true) < 0)
            return baseline;

        // Root scratch stays conservatively live through hooks and every
        // layer. Only AdaLN's input temporaries and the later Q/K operands have
        // disjoint proven windows: AdaLN consumes bias/partial before Q/K
        // projection; Q/K are dead after SDPA and absent from FFN/pre/post hooks.
        auto candidate = baseline;
        BufferDecl image_mask;
        BufferDecl* encoder_scratch = nullptr;
        BufferDecl* output_norm_scratch = nullptr;
        int found = 0;
        for (auto& decl : candidate) {
            if (decl.storage != StorageClass::Temp || decl.alias_of) continue;
            if (decl.scope != BufferScope::LayerWide ||
                decl.phase_start != 0 || decl.phase_end != 0)
                return baseline;
            decl.phase_start = 0;
            decl.phase_end = 8;
            const std::string name = decl.name ? decl.name : "";
            if (name == "enc0") encoder_scratch = &decl;
            if (name == "normed") output_norm_scratch = &decl;
            if (name == "bias_temp" || name == "gemv_partial" ||
                (long_mask_profile && name == "adaln")) {
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
        if (found != (long_mask_profile ? 6 : 5)) return baseline;
        if (long_mask_profile && (vl_seq_ > 224 || (num_cores() == 4 && vl_seq_ > 96))) {
            // enc0's last pre-hook use copies the action embeddings to DDR
            // emb_stage_. normed is first written by the post-hook LayerNorm,
            // after every DiT layer, and consumed there by po2. Neither value
            // crosses a body boundary. Keep their shared root live through all
            // hooks/layers so both retained masks and cross-KV/FF1 stay disjoint.
            // Short profiles and independently sealed Z2 layouts stay unchanged.
            if (!encoder_scratch || !output_norm_scratch ||
                encoder_scratch->size <= 0 ||
                encoder_scratch->size > output_norm_scratch->size)
                return baseline;
            encoder_scratch->alias_of = "normed";
        }
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

    bool kv_spm_only_profile(int64_t rows, int64_t batch) const {
        return kv_spm_only_opt_in_ && denoise_ready_ && denoise_mode_ && loop_mode_ &&
            (num_steps_ == 4 || num_steps_ == 10) && action_horizon_ == 40 && rows == 41 && batch == 1 &&
            num_layers() == 32 && hidden_size() == 1536 && intermediate_size() == 6144 &&
            num_q_heads() == 32 && num_kv_heads() == 32 && head_dim() == 48 &&
            cross_dim_ == 2048 && (attn_tp() == 4 || attn_tp() == 8) && !z2_bound_ && z2_prepared_seq_len_ == 0;
    }

    int64_t kv_insert_flags(int64_t site_id, int64_t invocation) {
        if (!ctx().has_complete_physical_manifest() ||
            !kv_spm_only_profile(ctx().seq_len, ctx().batch_size))
            return GR00T_DIT_KV_FLAG_DDR_MIRROR;
        TORCH_CHECK(site_id == GR00T_DIT_SELF_KV_INSERT_SITE ||
                        site_id == GR00T_DIT_CROSS_KV_INSERT_SITE,
                    "GR00T SPM-only KV requires a self or cross insert site");
        const int64_t raw_site = site_id == GR00T_DIT_SELF_KV_INSERT_SITE
            ? GR00T_DIT_SELF_RAW_ATTENTION_SITE : GR00T_DIT_CROSS_RAW_ATTENTION_SITE;
        for (const auto& route : ctx().physical_manifest().routes) {
            if (route.family == FmbRouteFamily::ATTENTION &&
                route.site_id == raw_site && route.invocation == invocation) {
                TORCH_CHECK(ctx().physical_attention_policy_for_site(raw_site, invocation) ==
                                AttentionExecutionPolicy::SPM_KV_BY_MHA,
                            "GR00T SPM-only KV raw attention selector drift");
                return GR00T_DIT_KV_FLAG_RAW_SPM_ONLY;
            }
        }
        return GR00T_DIT_KV_FLAG_DDR_MIRROR;
    }

    void check_self_qkv_setter_state() const {
        if (!self_qkv_multibase_opt_in_) return;
        TORCH_CHECK(!RpuKernelGraph::has_active() && !inference_started_ &&
                        get_last_resolved_chunk_size() == 0 && !z2_bound_ &&
                        z2_prepared_seq_len_ == 0,
                    "GR00T self-QKV policy must be installed before any forward or Z2 prepare");
    }

    void install_self_qkv_multibase() {
        check_self_qkv_setter_state();
        self_qkv_multibase_ready_ = false;
        if (!self_qkv_multibase_opt_in_) return;
        if (num_layers() != 32 || hidden_size() != 1536 || intermediate_size() != 6144 ||
            num_q_heads() != 32 || num_kv_heads() != 32 || head_dim() != 48 ||
            attn_tp() != 8 || cross_dim_ != 2048) return;
        bool has_self = false;
        for (const auto& layer : layer_weights_) {
            if (layer.is_cross) continue;
            has_self = true;
            for (const auto* weight : {&layer.to_q_w, &layer.to_k_w, &layer.to_v_w}) {
                if (!weight->defined() || weight->scalar_type() != at::kChar ||
                    weight->device().type() != at::kPrivateUse1 || !weight->is_contiguous() ||
                    weight->dim() != 2 || weight->size(0) != 1536 || weight->size(1) != 1536)
                    return;
            }
            for (const auto* tensor : {&layer.to_q_b, &layer.to_k_b, &layer.to_v_b,
                                      &layer.to_q_ws, &layer.to_k_ws, &layer.to_v_ws}) {
                if (!tensor->defined() || tensor->scalar_type() != at::kHalf ||
                    tensor->device().type() != at::kPrivateUse1 || !tensor->is_contiguous() ||
                    tensor->dim() != 1 || tensor->size(0) != 1536) return;
            }
        }
        if (!has_self) return;
        TORCH_CHECK(KernelCache::instance().has_loaded(
                        KernelId::GR00T_QKV_MULTIBASE_W8A16_M592N32K128),
                    "GR00T self-QKV multibase requires its optional expansion payload");
        self_qkv_multibase_ready_ = true;
    }

    bool self_qkv_multibase_profile(int64_t rows, int64_t batch) const {
        return self_qkv_multibase_opt_in_ && self_qkv_multibase_ready_ &&
            denoise_ready_ && denoise_mode_ && loop_mode_ &&
            (num_steps_ == 4 || num_steps_ == 10) && vl_seq_ > 0 &&
            action_horizon_ == 40 && num_layers() == 32 && rows == 41 && batch == 1 &&
            hidden_size() == 1536 && intermediate_size() == 6144 &&
            num_q_heads() == 32 && num_kv_heads() == 32 && head_dim() == 48 &&
            cross_dim_ == 2048 && attn_tp() == 8 && !z2_bound_ && z2_prepared_seq_len_ == 0;
    }

    std::vector<int64_t> self_qkv_multibase_arguments(int64_t rows, int64_t batch) const {
        const auto c = rpu_specialized_linear::contract(KernelId::GR00T_QKV_MULTIBASE_W8A16_M592N32K128);
        std::vector<int64_t> args{rows, batch, vl_seq_, num_steps_, num_layers(), hidden_size(),
            intermediate_size(), num_q_heads(), num_kv_heads(), head_dim(), attn_tp(), cross_dim_,
            action_horizon_, denoise_mode_ ? 1 : 0, loop_mode_ ? 1 : 0,
            self_qkv_multibase_opt_in_ ? 1 : 0, self_qkv_multibase_ready_ ? 1 : 0,
            /*local_N=*/192, /*bias=*/1, /*tile=*/c.m, c.n, c.k, /*grid=*/c.gx, c.gy, c.gz, /*ABI=*/2,
            2, 3, 20, 21, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47};
        for (const auto& layer : layer_weights_) args.push_back(layer.is_cross ? 1 : 0);
        return args;
    }

    bool emit_self_qkv_multibase(int layer_idx, const ChunkInfo& chunk, const LayerWeights& lw) {
        if (!ctx().has_complete_physical_manifest() ||
            !self_qkv_multibase_profile(chunk.len, ctx().batch_size) ||
            layer_idx < 0 || layer_idx >= num_layers() || lw.is_cross) return false;
        ctx().consume_physical_route(FmbRouteFamily::LINEAR, GR00T_DIT_SELF_QKV_MULTIBASE_SITE,
            /*MULTIBASE_M592N32K128=*/2, 0,
            self_qkv_multibase_arguments(chunk.len, ctx().batch_size), chunk.idx);
        rpu_launch_gr00t_qkv_multibase_w8_kernel(
            addr(0, "norm_hidden"), lw.to_q_w, lw.to_k_w, lw.to_v_w,
            addr(0, "q"), addr(0, "k"), addr(0, "v"),
            layer_addr(layer_idx, 0, "q_bias"), layer_addr(layer_idx, 0, "k_bias"),
            layer_addr(layer_idx, 0, "v_bias"), lw.to_q_ws, lw.to_k_ws, lw.to_v_ws);
        return true;
    }

    void check_ring_xor3_setter_state() const {
        if (!ring_xor3_opt_in_) return;
        TORCH_CHECK(!RpuKernelGraph::has_active() && !inference_started_ &&
                        get_last_resolved_chunk_size() == 0 && !z2_bound_ &&
                        z2_prepared_seq_len_ == 0,
                    "GR00T XOR3 ring policy must be installed before any forward or Z2 prepare");
    }

    void install_ring_xor3() {
        check_ring_xor3_setter_state();
        ring_xor3_ready_ = false;
        if (!ring_xor3_opt_in_) return;
        if (num_layers() != 32 || layer_weights_.size() != 32 ||
            hidden_size() != 1536 || intermediate_size() != 6144 ||
            num_q_heads() != 32 || num_kv_heads() != 32 || head_dim() != 48 ||
            attn_tp() != 8 || cross_dim_ != 2048) return;
        // Both generated ACC16 producer families retire FP16 partial sums.
        // The XOR3 payload only consumes those three SPM buffers; require a
        // uniform producer precision and the corresponding original scales.
        const auto precision = layer_weights_.front().to_out_w.scalar_type();
        if (precision != at::kChar && precision != at::kHalf) return;
        const bool w8 = precision == at::kChar;
        for (const auto& layer : layer_weights_) {
            for (const auto* weight : {&layer.to_out_w, &layer.ff2_w}) {
                const int64_t k = weight == &layer.to_out_w ? 1536 : 6144;
                if (!weight->defined() || weight->scalar_type() != precision ||
                    weight->device().type() != at::kPrivateUse1 || !weight->is_contiguous() ||
                    weight->dim() != 2 || weight->size(0) != 1536 || weight->size(1) != k)
                    return;
            }
            for (const auto* tensor : {&layer.to_out_b, &layer.ff2_b}) {
                if (!tensor->defined() || tensor->scalar_type() != at::kHalf ||
                    tensor->device().type() != at::kPrivateUse1 || !tensor->is_contiguous() ||
                    tensor->dim() != 1 || tensor->size(0) != 1536) return;
            }
            for (const auto* scale : {&layer.to_out_ws, &layer.ff2_ws}) {
                if (!w8) {
                    if (scale->defined()) return;
                } else if (!scale->defined() || scale->scalar_type() != at::kHalf ||
                           scale->device().type() != at::kPrivateUse1 || !scale->is_contiguous() ||
                           scale->dim() != 1 || scale->size(0) != 1536) return;
            }
        }
        TORCH_CHECK(KernelCache::instance().has_loaded(
                        KernelId::GR00T_ALL_REDUCE_RESIDUAL_XOR3_M41N1536),
                    "GR00T XOR3 residual ring requires its optional expansion payload");
        ring_xor3_ready_ = true;
    }

    bool ring_xor3_profile(int64_t rows, int64_t batch) const {
        return ring_xor3_opt_in_ && ring_xor3_ready_ &&
            denoise_ready_ && denoise_mode_ && loop_mode_ &&
            (num_steps_ == 4 || num_steps_ == 10) && vl_seq_ > 0 &&
            action_horizon_ == 40 && num_layers() == 32 && rows == 41 && batch == 1 &&
            hidden_size() == 1536 && intermediate_size() == 6144 &&
            num_q_heads() == 32 && num_kv_heads() == 32 && head_dim() == 48 &&
            cross_dim_ == 2048 && attn_tp() == 8 && !z2_bound_ && z2_prepared_seq_len_ == 0;
    }

    std::vector<int64_t> ring_xor3_arguments(int64_t rows, int64_t batch, bool ff) const {
        return {/*ABI=*/1, /*residual2->1 vs residual1->2=*/ff ? 1 : 0,
            rows, batch, vl_seq_, num_steps_, num_layers(), hidden_size(), intermediate_size(),
            num_q_heads(), num_kv_heads(), head_dim(), attn_tp(), cross_dim_, action_horizon_,
            denoise_mode_ ? 1 : 0, loop_mode_ ? 1 : 0,
            ring_xor3_opt_in_ ? 1 : 0, ring_xor3_ready_ ? 1 : 0,
            /*FP16=*/16, /*original paced reduction=*/2,
            /*chunk elements/bytes/alignment=*/7872, 15744, 128,
            /*full matrix bytes=*/125952, /*rounds=*/1, /*round capacity=*/63488,
            /*actual grid=*/8, 1, 1, /*XOR gather stages=*/3, /*epochs=*/12,
            /*full input/residual/output layouts=*/1, 1, 1,
            /*all disjoint=*/1, /*DDR residual=*/0,
            /*invocations per ordinary request/site=*/num_steps_ * num_layers()};
    }

    bool emit_attn_ring_xor3(const ChunkInfo& chunk) {
        if (!ctx().has_complete_physical_manifest() ||
            !ring_xor3_profile(chunk.len, ctx().batch_size)) return false;
        ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE, GR00T_DIT_ATTN_RING_XOR3_SITE,
            /*EXACT_XOR3=*/2, 0,
            ring_xor3_arguments(chunk.len, ctx().batch_size, /*ff=*/false), chunk.idx);
        rpu_launch_gr00t_all_reduce_residual_xor3_kernel(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"));
        return true;
    }

    bool emit_ff_ring_xor3(const ChunkInfo& chunk) {
        if (!ctx().has_complete_physical_manifest() ||
            !ring_xor3_profile(chunk.len, ctx().batch_size)) return false;
        ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE, GR00T_DIT_FF_RING_XOR3_SITE,
            /*EXACT_XOR3=*/2, 0,
            ring_xor3_arguments(chunk.len, ctx().batch_size, /*ff=*/true), chunk.idx);
        rpu_launch_gr00t_all_reduce_residual_xor3_kernel(
            addr(0, "oproj"), addr(0, "residual2"), addr(0, "residual1"));
        return true;
    }

    void check_ff1_resident_setter_state() const {
        if (!ff1_resident_w8_opt_in_) return;
        TORCH_CHECK(!RpuKernelGraph::has_active() && !inference_started_ &&
                        get_last_resolved_chunk_size() == 0 && !z2_bound_ &&
                        z2_prepared_seq_len_ == 0,
                    "GR00T resident FF1 weights must be installed before any forward or Z2 prepare");
    }

    void clear_ff1_resident_weight() {
        ff1_resident_weight_ = at::Tensor{};
        ff1_resident_source_ = at::Tensor{};
        ff1_resident_src_base_ = 0;
        ff1_resident_source_version_ = 0;
    }

    void install_ff1_resident_weight() {
        if (!ff1_resident_w8_opt_in_ || !denoise_ready_ || num_layers() != 32 ||
            hidden_size() != 1536 || intermediate_size() != 6144 ||
            num_q_heads() != 32 || num_kv_heads() != 32 || head_dim() != 48 ||
            attn_tp() != 8 || cross_dim_ != 2048 || action_horizon_ != 40 ||
            (num_steps_ != 4 && num_steps_ != 10) || layer_weights_[0].ff1_w.scalar_type() != at::kChar)
            return;
        check_ff1_resident_setter_state();
        RpuExecutionCoordinator::require_graph_quiescent("gr00t_install_ff1_resident_w8");
        RpuExecutionCoordinator::require_no_physical_claim("gr00t_install_ff1_resident_w8");
        auto execution = RpuExecutionCoordinator::enter_physical("gr00t_install_ff1_resident_w8");
        auto release = c10::make_scope_exit([&] {
            RpuExecutionCoordinator::exit_physical_noexcept(execution);
        });
        const auto& layer = layer_weights_[0];
        const auto& version = layer.ff1_w.unsafeGetTensorImpl()->version_counter();
        TORCH_CHECK(version.enabled(), "GR00T resident FF1 requires a versioned weight owner");
        const int64_t source_version = static_cast<int64_t>(version.current_version());
        for (const auto* tensor : {&layer.ff1_b, &layer.ff1_ws}) {
            TORCH_CHECK(tensor->defined() && tensor->scalar_type() == at::kHalf &&
                            tensor->device().type() == at::kPrivateUse1 &&
                            tensor->is_contiguous() && tensor->dim() == 1 && tensor->size(0) == 6144,
                        "GR00T resident FF1 requires original fp16 bias and scale [6144]");
        }
        TORCH_CHECK(KernelCache::instance().has_loaded(
                        KernelId::GR00T_FF1_RESIDENT_W8A16_M512N48K128),
                    "GR00T resident FF1 requires its optional expansion payload");
        auto packed = rpu_pack_gr00t_ff1_resident_w8(layer.ff1_w);
        TORCH_CHECK(version.enabled() && static_cast<int64_t>(version.current_version()) == source_version,
                    "GR00T resident FF1 source changed during cold installation");
        ff1_resident_weight_ = std::move(packed);
        ff1_resident_src_base_ = ::rhino_lkn::RpuGetDevAddr(ff1_resident_weight_.data_ptr());
        ff1_resident_source_ = layer.ff1_w;
        ff1_resident_source_version_ = source_version;
        ++ff1_resident_generation_;
    }

    void begin_ff1_resident_inference() {
        if (!ff1_resident_source_.defined()) return;
        const auto& version = ff1_resident_source_.unsafeGetTensorImpl()->version_counter();
        TORCH_CHECK(version.enabled() &&
                        static_cast<int64_t>(version.current_version()) == ff1_resident_source_version_,
                    "GR00T resident FF1 source was mutated after cold installation");
    }

    bool ff1_resident_profile(int64_t rows, int64_t batch) const {
        return ff1_resident_w8_opt_in_ && ff1_resident_weight_.defined() &&
            denoise_ready_ && denoise_mode_ && loop_mode_ &&
            (num_steps_ == 4 || num_steps_ == 10) && vl_seq_ > 0 &&
            action_horizon_ == 40 && num_layers() == 32 && rows == 41 && batch == 1 &&
            hidden_size() == 1536 && intermediate_size() == 6144 &&
            num_q_heads() == 32 && num_kv_heads() == 32 && head_dim() == 48 &&
            cross_dim_ == 2048 && attn_tp() == 8 && !z2_bound_ && z2_prepared_seq_len_ == 0;
    }

    std::vector<int64_t> ff1_resident_arguments(int64_t rows, int64_t batch) const {
        const auto c = rpu_specialized_linear::contract(KernelId::GR00T_FF1_RESIDENT_W8A16_M512N48K128);
        return {rows, batch, vl_seq_, num_steps_, num_layers(), hidden_size(), intermediate_size(),
                num_q_heads(), num_kv_heads(), head_dim(), attn_tp(), action_horizon_,
                denoise_mode_ ? 1 : 0, loop_mode_ ? 1 : 0, ff1_resident_w8_opt_in_ ? 1 : 0,
                ff1_resident_weight_.defined() ? 1 : 0, ff1_resident_generation_,
                /*layer=*/0, GR00T_DIT_FF1_RESIDENT_BYTES, /*bias=*/1, c.m, c.n, c.k, c.gx, c.gy, c.gz, /*ABI=*/2};
    }

    FmbForwardOperandResidency ff1_residency_for_candidate(
        const LayoutContext& layout) const {
        // A COMPLETE selection is immutable, including capacity fallback. Never
        // consult live allocator occupancy while reconstructing/replaying it.
        if (layout.forward_operand_residency != FmbForwardOperandResidency::UNSPECIFIED) {
            TORCH_CHECK(layout.forward_operand_residency == FmbForwardOperandResidency::PER_LAYER ||
                            (layout.forward_operand_residency == FmbForwardOperandResidency::FORWARD &&
                             layout.chunk_size >= 41 && ff1_resident_profile(41, layout.batch_size)),
                        "GR00T invalid bound resident FF1 selection");
            return layout.forward_operand_residency;
        }
        if (!ff1_resident_profile(layout.chunk_size, layout.batch_size))
            return FmbForwardOperandResidency::PER_LAYER;
        const auto fits = [&](LayoutContext candidate) {
            candidate.forward_operand_residency = FmbForwardOperandResidency::FORWARD;
            return declared_spm_layout_fits(
                planned_buffer_declarations(candidate, /*complete=*/true));
        };
        if (!fits(layout)) return FmbForwardOperandResidency::PER_LAYER;
        if (layout.planning_chunk_capacity > 0 &&
                layout.planning_chunk_capacity != layout.chunk_size) {
            auto capacity = layout;
            capacity.chunk_size = layout.planning_chunk_capacity;
            if (!fits(capacity)) return FmbForwardOperandResidency::PER_LAYER;
        }
        return FmbForwardOperandResidency::FORWARD;
    }

    FmbForwardOperandResidency physical_forward_operand_residency(
        const FmbPhysicalExecutionManifest& manifest) const override {
        if (manifest.state != FmbPhysicalManifestState::COMPLETE ||
                !ff1_resident_w8_opt_in_ || !denoise_mode_)
            return FmbForwardOperandResidency::UNSPECIFIED;
        const auto matches = [](const auto& route) {
            return route.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                route.site_id == GR00T_DIT_FF1_RESIDENT_SCHEDULE_SITE;
        };
        const auto route = std::find_if(manifest.routes.begin(), manifest.routes.end(), matches);
        TORCH_CHECK(route != manifest.routes.end() &&
                        std::count_if(manifest.routes.begin(), manifest.routes.end(), matches) == 1 &&
                        route->flags == 0 && route->invocation == 0 &&
                        (route->selector == 1 || route->selector == 2),
                    "GR00T requires one valid resident FF1 schedule");
        const auto& args = route->arguments;
        // Guard the two indexed fields; full equality below binds the current
        // generated tile/grid/ABI and rejects truncated or retired schedules.
        TORCH_CHECK(args.size() >= 2 && args[0] == manifest.physical_length && args[1] > 0 &&
                        args == ff1_resident_arguments(args[0], args[1]) &&
                        (route->selector == 1 || ff1_resident_profile(args[0], args[1])),
                    "GR00T resident FF1 schedule differs from its installed owner");
        return route->selector == 2 ? FmbForwardOperandResidency::FORWARD
                                    : FmbForwardOperandResidency::PER_LAYER;
    }

    bool ff1_resident_selected() const {
        if (!ctx().has_complete_physical_manifest() || !ff1_resident_w8_opt_in_ || !denoise_mode_)
            return false;
        return physical_forward_operand_residency(ctx().physical_manifest()) ==
            FmbForwardOperandResidency::FORWARD;
    }

    void load_ff1_resident_weight() {
        if (!ff1_resident_w8_opt_in_ || !ctx().has_complete_physical_manifest()) return;
        const bool active = ff1_resident_selected();
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            GR00T_DIT_FF1_RESIDENT_SCHEDULE_SITE, active ? 2 : 1, 0,
            ff1_resident_arguments(ctx().seq_len, ctx().batch_size), 0);
        if (!active) return;
        TORCH_CHECK(ctx().body_iter >= 0 && ctx().body_iter < num_steps_,
                    "GR00T resident FF1 body out of bounds");
        if (ctx().body_iter != 0) return;
        ctx().consume_physical_route(FmbRouteFamily::MUTABLE_DMA,
            GR00T_DIT_FF1_RESIDENT_DMA_SITE,
            static_cast<int64_t>(Gr00tDiTMutableDmaRoute::DDR_SCATTER_TO_SPM), 0,
            ff1_resident_arguments(ctx().seq_len, ctx().batch_size), 0);
        rpu_launch_gr00t_ff1_resident_w8_preload(
            ff1_resident_src_base_, ff1_resident_weight_, addr(0, "ff1_resident_w8"));
    }

    void check_time_prepare_state() const {
        TORCH_CHECK(!RpuKernelGraph::has_active() && !inference_started_ &&
                        get_last_resolved_chunk_size() == 0 && !z2_bound_ &&
                        z2_prepared_seq_len_ == 0 && !time_tau_table_.defined(),
                    "GR00T time tables must be prepared once before any forward or Z2 prepare");
        TORCH_CHECK(denoise_ready_ && (num_steps_ == 4 || num_steps_ == 10) && num_layers() == 32 &&
                        hidden_size() == 1536 && intermediate_size() == 6144 &&
                        num_q_heads() == 32 && num_kv_heads() == 32 && head_dim() == 48 &&
                        (attn_tp() == 4 || attn_tp() == 8) && action_horizon_ == 40 && cross_dim_ == 2048,
                    "GR00T time tables require the installed four- or ten-step TP4/TP8 public DiT profile");
        for (const auto* source : {&ae2t_w_, &ae2t_ws_, &po1_w_, &po1_b_, &po1_ws_}) {
            TORCH_CHECK(!source->defined() || source->unsafeGetTensorImpl()->version_counter().enabled(),
                        "GR00T time preparation requires versioned weight/bias/scale owners");
        }
    }

    void clear_time_tables() {
        time_tau_table_ = at::Tensor{};
        time_po1_table_ = at::Tensor{};
        time_tau_src_base_ = time_po1_src_base_ = 0;
        time_sources_.clear();
    }

    void begin_time_inference() {
        for (const auto& source : time_sources_) {
            const auto& version = source.first.unsafeGetTensorImpl()->version_counter();
            TORCH_CHECK(version.enabled() &&
                            static_cast<int64_t>(version.current_version()) == source.second,
                        "GR00T time table source was mutated after cold preparation");
        }
    }

    bool time_table_profile(int64_t rows, int64_t batch) const {
        return time_tau_table_.defined() && time_po1_table_.defined() &&
            denoise_ready_ && denoise_mode_ && loop_mode_ && action_horizon_ == 40 &&
            (num_steps_ == 4 || num_steps_ == 10) && num_layers() == 32 && rows == 41 && batch == 1 &&
            hidden_size() == 1536 && intermediate_size() == 6144 &&
            num_q_heads() == 32 && num_kv_heads() == 32 && head_dim() == 48 &&
            cross_dim_ == 2048 && (attn_tp() == 4 || attn_tp() == 8) && !z2_bound_ && z2_prepared_seq_len_ == 0;
    }

    std::vector<int64_t> time_table_arguments(int64_t rows, int64_t batch) const {
        return {rows, batch, num_steps_, num_layers(), hidden_size(), intermediate_size(),
                num_q_heads(), num_kv_heads(), head_dim(), attn_tp(), action_horizon_,
                denoise_mode_ ? 1 : 0, loop_mode_ ? 1 : 0, time_table_generation_,
                action_horizon_ * hidden_size() * DWIDTH, 2 * hidden_size() * DWIDTH};
    }

    bool time_tables_active() {
        const bool active = ctx().has_complete_physical_manifest() &&
            time_table_profile(ctx().seq_len, ctx().batch_size);
        if (time_tau_table_.defined() && ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                GR00T_DIT_TIME_TABLE_SCHEDULE_SITE, active ? 2 : 1, 0,
                time_table_arguments(ctx().seq_len, ctx().batch_size), 0);
        }
        return active;
    }

    void check_adaln_prepare_state() const {
        TORCH_CHECK(!RpuKernelGraph::has_active() && !inference_started_ &&
                        get_last_resolved_chunk_size() == 0 &&
                        !z2_bound_ && z2_prepared_seq_len_ == 0 && !adaln_table_.defined(),
                    "GR00T AdaLN table must be prepared once before any forward or Z2 prepare");
        TORCH_CHECK(denoise_ready_ && (num_steps_ == 4 || num_steps_ == 10) && num_layers() == 32 &&
                        hidden_size() == 1536 && intermediate_size() == 6144 &&
                        num_q_heads() == 32 && num_kv_heads() == 32 && head_dim() == 48 &&
                        (attn_tp() == 4 || attn_tp() == 8) && action_horizon_ == 40 && cross_dim_ == 2048,
                    "GR00T AdaLN table requires the installed four- or ten-step TP4/TP8 public DiT profile");
        for (const auto& layer : layer_weights_) {
            for (const auto* source : {&layer.adaln_w, &layer.adaln_b, &layer.adaln_ws}) {
                TORCH_CHECK(!source->defined() || source->unsafeGetTensorImpl()->version_counter().enabled(),
                            "GR00T AdaLN cold preparation requires versioned weight/bias/scale owners");
            }
        }
    }

    void clear_adaln_table() {
        adaln_table_ = at::Tensor{};
        adaln_table_src_base_ = 0;
        adaln_sources_.clear();
    }

    void begin_adaln_inference() {
        // Monotonic: even a rejected/failed forward closes the cold API.
        inference_started_ = true;
        for (const auto& source : adaln_sources_) {
            const auto& version = source.first.unsafeGetTensorImpl()->version_counter();
            TORCH_CHECK(version.enabled() &&
                            static_cast<int64_t>(version.current_version()) == source.second,
                        "GR00T AdaLN table source was mutated after cold preparation");
        }
    }

    bool adaln_table_profile(int64_t rows, int64_t batch) const {
        return adaln_table_.defined() && denoise_ready_ && denoise_mode_ && loop_mode_ &&
            (num_steps_ == 4 || num_steps_ == 10) && num_layers() == 32 && rows == 41 && batch == 1 &&
            hidden_size() == 1536 && intermediate_size() == 6144 &&
            num_q_heads() == 32 && num_kv_heads() == 32 && head_dim() == 48 &&
            cross_dim_ == 2048 && (attn_tp() == 4 || attn_tp() == 8) &&
            !z2_bound_ && z2_prepared_seq_len_ == 0;
    }

    int64_t adaln_table_body_bytes() const {
        return num_layers() * 2 * hidden_size() * DWIDTH;
    }

    std::vector<int64_t> adaln_table_arguments(int64_t rows, int64_t batch) const {
        return {rows, batch, num_steps_, num_layers(), hidden_size(), intermediate_size(),
                num_q_heads(), num_kv_heads(), head_dim(), attn_tp(),
                denoise_mode_ ? 1 : 0, loop_mode_ ? 1 : 0,
                adaln_table_generation_, adaln_table_body_bytes()};
    }

    uint32_t adaln_layer_scale(int layer_idx, const ChunkInfo& chunk) {
        const bool table = ctx().has_complete_physical_manifest() &&
            adaln_table_profile(chunk.len, ctx().batch_size);
        if (adaln_table_.defined() && ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                GR00T_DIT_ADALN_TABLE_SCHEDULE_SITE, table ? 2 : 1, 0,
                adaln_table_arguments(chunk.len, ctx().batch_size), chunk.idx);
        }
        if (table) {
            TORCH_CHECK(layer_idx >= 0 && layer_idx < num_layers() &&
                            ctx().body_iter >= 0 && ctx().body_iter < num_steps_,
                        "GR00T AdaLN table layer/body out of bounds");
            return addr(0, "adaln_body_table") +
                static_cast<uint32_t>(layer_idx * 2 * hidden_size() * DWIDTH);
        }
        const auto& lw = layer_weights_[layer_idx];
        if (chunk.idx == 0) {
            dit_adaln_gemv(addr(0, "cond"), lw.adaln_w, lw.adaln_b,
                           addr(0, "adaln"), addr(0, "bias_temp"), addr(0, "gemv_partial"), lw.adaln_ws);
        }
        return addr(0, "adaln");
    }

    bool cross_kv_reuse_profile(int64_t rows, int64_t batch) const {
        return cross_kv_reuse_opt_in_ && (num_cores() == 4 || num_cores() == 8) && denoise_ready_ &&
            denoise_mode_ && loop_mode_ && (num_steps_ == 4 || num_steps_ == 10) &&
            !z2_bound_ && z2_prepared_seq_len_ == 0 &&
            batch == 1 && rows == 41 && num_layers() == 32 &&
            hidden_size() == 1536 && intermediate_size() == 6144 &&
            num_q_heads() == 32 && num_kv_heads() == 32 && head_dim() == 48 &&
            cross_dim_ == 2048 && vl_seq_ > 0;
    }

    int64_t cross_kv_schedule(int64_t rows, int64_t batch, bool cross_raw) const {
        if (!cross_kv_reuse_profile(rows, batch))
            return static_cast<int64_t>(Gr00tDiTCrossKvSchedule::PROJECT_EACH_BODY);
        return static_cast<int64_t>(cross_raw
            ? Gr00tDiTCrossKvSchedule::REUSE_FIRST_BODY_SPM
            : Gr00tDiTCrossKvSchedule::REUSE_FIRST_BODY_DDR);
    }

    int64_t cross_kv_bank_stride() const {
        return Align(gr00t_pad16(vl_seq_) *
            (num_kv_heads() * head_dim() / attn_tp()) * DWIDTH, int64_t{256});
    }

    int64_t cross_kv_bank_bytes() const {
        return std::count_if(layer_weights_.begin(), layer_weights_.end(),
            [](const LayerWeights& layer) { return layer.is_cross; }) * cross_kv_bank_stride();
    }

    uint32_t cross_kv_bank_offset(int layer_idx) const {
        TORCH_CHECK(layer_idx >= 0 && layer_idx < num_layers() &&
                    layer_weights_[layer_idx].is_cross,
                    "GR00T cross-KV bank requires a cross layer");
        const int64_t ordinal = std::count_if(
            layer_weights_.begin(), layer_weights_.begin() + layer_idx,
            [](const LayerWeights& layer) { return layer.is_cross; });
        const int64_t offset = ordinal * cross_kv_bank_stride();
        TORCH_CHECK(offset >= 0 && offset + cross_kv_bank_stride() <= cross_kv_bank_bytes() &&
                    cross_kv_bank_bytes() <= SpmAllocator::SPM_USABLE,
                    "GR00T cross-KV bank extent is invalid");
        return static_cast<uint32_t>(offset);
    }

    void append_cross_kv_buffers(std::vector<BufferDecl>& buffers, const LayoutContext& layout) const {
        if (!cross_kv_reuse_profile(layout.chunk_size, layout.batch_size) ||
            layout.attention_policy != AttentionExecutionPolicy::SPM_KV_BY_MHA) return;
        const int64_t bytes = cross_kv_bank_bytes();
        if (bytes == 0 || (num_cores() == 4 &&
                           2 * bytes > (num_steps_ <= 4 ? 4718592 : 3932160))) return;
        // Overall raw layout includes mixed self-raw/cross-DDR candidates.
        // Budget both banks there too; all layers/bodies and hooks must preserve them.
        // The mask residency planner extends these conservative 0..0 lifetimes
        // with every other full-lifetime scratch buffer to 0..8.
        buffers.push_back({"cross_k_bank", bytes, 0, 0, StorageClass::Temp, 0, nullptr, BufferScope::LayerWide});
        buffers.push_back({"cross_v_bank", bytes, 0, 0, StorageClass::Temp, 0, nullptr, BufferScope::LayerWide});
    }

    std::vector<int64_t> cross_kv_schedule_arguments(int64_t rows, int64_t batch) const {
        std::vector<int64_t> args{
            denoise_mode_ ? 1 : 0, loop_mode_ ? 1 : 0,
            num_steps_, num_layers(), rows, vl_seq_, batch,
            hidden_size(), intermediate_size(), cross_dim_,
            num_q_heads(), num_kv_heads(), head_dim(),
            cross_kv_bank_stride(), cross_kv_bank_bytes()};
        for (const auto& layer : layer_weights_) args.push_back(layer.is_cross ? 1 : 0);
        return args;
    }

    bool cross_v_transpose_profile(int64_t rows, int64_t batch, bool cross_raw) const {
        return cross_v_transpose_reuse_opt_in_ && cross_raw &&
            cross_kv_reuse_profile(rows, batch) && (attn_tp() == 4 || attn_tp() == 8);
    }

    std::vector<int64_t> cross_v_transpose_arguments(int64_t rows, int64_t batch, bool cross_raw) const {
        auto args = cross_kv_schedule_arguments(rows, batch);
        args.insert(args.end(), {cross_raw ? 1 : 0, cross_v_transpose_reuse_opt_in_ ? 1 : 0,
                                attn_tp(), gr00t_pad16(vl_seq_)});
        return args;
    }

    bool cross_v_transpose_reused(const ChunkInfo& chunk, AttentionExecutionPolicy policy) const {
        const bool raw = policy == AttentionExecutionPolicy::SPM_KV_BY_MHA;
        const bool active = ctx().has_complete_physical_manifest() &&
            cross_v_transpose_profile(chunk.len, ctx().batch_size, raw);
        if (cross_v_transpose_reuse_opt_in_ && ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
                GR00T_DIT_CROSS_V_TRANSPOSE_SCHEDULE_SITE, active ? 2 : 1, 0,
                cross_v_transpose_arguments(chunk.len, ctx().batch_size, raw), chunk.idx);
        }
        return active;
    }

    bool cross_kv_projection_needed(
        const ChunkInfo& chunk, AttentionExecutionPolicy attention_policy) const {
        if (!cross_kv_reuse_opt_in_ || !ctx().has_complete_physical_manifest()) return true;
        const int64_t schedule = cross_kv_schedule(
            chunk.len, ctx().batch_size,
            attention_policy != AttentionExecutionPolicy::DDR_KV);
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, GR00T_DIT_CROSS_KV_SCHEDULE_SITE,
            schedule, /*resolved_flags=*/0,
            cross_kv_schedule_arguments(chunk.len, ctx().batch_size), chunk.idx);
        // A fixed graph position refreshes every layer on EVERY request/replay.
        // Never use a host "loaded" flag: vl_embeds changes between observations.
        return schedule == static_cast<int64_t>(Gr00tDiTCrossKvSchedule::PROJECT_EACH_BODY)
            || ctx().body_iter == 0;
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
        if (cross_kv_reuse_opt_in_) {
            h = detail::layout_mix(h, loop_mode_ ? 1 : 0);
            h = detail::layout_mix(h, num_steps_);
        }
        if (kv_spm_only_opt_in_) {
            h = detail::layout_mix(h, kv_spm_only_opt_in_ ? 1 : 0);
            h = detail::layout_mix(h, loop_mode_ ? 1 : 0);
            h = detail::layout_mix(h, num_steps_);
        }
        if (cross_v_transpose_reuse_opt_in_) {
            h = detail::layout_mix(h, GR00T_DIT_CROSS_V_TRANSPOSE_SCHEDULE_SITE);
            h = detail::layout_mix(h, loop_mode_ ? 1 : 0);
            h = detail::layout_mix(h, num_steps_);
        }
        if (adaln_table_.defined()) {
            h = detail::layout_mix(h, adaln_table_generation_);
            h = detail::layout_mix(h, adaln_table_body_bytes());
            h = detail::layout_mix(h, loop_mode_ ? 1 : 0);
            h = detail::layout_mix(h, num_steps_);
        }
        if (time_tau_table_.defined()) {
            h = detail::layout_mix(h, time_table_generation_);
            h = detail::layout_mix(h, loop_mode_ ? 1 : 0);
            h = detail::layout_mix(h, num_steps_);
        }
        if (self_qkv_multibase_opt_in_) {
            h = detail::layout_mix(h, GR00T_DIT_SELF_QKV_MULTIBASE_SITE);
            h = detail::layout_mix(h, self_qkv_multibase_ready_ ? 1 : 0);
            h = detail::layout_mix(h, loop_mode_ ? 1 : 0);
            h = detail::layout_mix(h, num_steps_);
        }
        if (ring_xor3_opt_in_) {
            h = detail::layout_mix(h, GR00T_DIT_ATTN_RING_XOR3_SITE);
            h = detail::layout_mix(h, GR00T_DIT_FF_RING_XOR3_SITE);
            h = detail::layout_mix(h, ring_xor3_ready_ ? 1 : 0);
            h = detail::layout_mix(h, loop_mode_ ? 1 : 0);
            h = detail::layout_mix(h, num_steps_);
        }
        if (ff1_resident_w8_opt_in_) {
            h = detail::layout_mix(h, GR00T_DIT_FF1_RESIDENT_SCHEDULE_SITE);
            h = detail::layout_mix(h, ff1_resident_generation_);
            h = detail::layout_mix(h, ff1_resident_weight_.defined() ? 1 : 0);
            h = detail::layout_mix(h, loop_mode_ ? 1 : 0);
            h = detail::layout_mix(h, num_steps_);
        }
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
            layout.forward_operand_residency =
                physical_forward_operand_residency(ctx().physical_manifest());
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
            const int64_t local_k = h / num_cores();
            rpu_launch_ddr_scatter_spm_dma_mutable(
                cond_src_base_, cond_ref_, /*src_offset_bytes=*/0,
                /*elements_per_core=*/local_k, /*core_stride_bytes=*/local_k * DWIDTH,
                addr(0, "cond"), num_cores());
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
                vl_seq_ * cross_dim_, addr(0, "vl_embeds"), num_cores());
            vl_loaded_this_forward_ = true;
        }

        if (layer_weights_[layer_idx].is_cross) build_cross_attn_block(layer_idx, chunk);
        else                                    build_self_attn_block(layer_idx, chunk);
    }

    KvInsertSegmentPlan consume_kv_insert_plan(
        int64_t site_id, int64_t true_seq, int64_t invocation,
        uint32_t k_spm = 0, uint32_t v_spm = 0) {
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
            kv_insert_flags(site_id, invocation),
            route.arguments, invocation);
        if (plan.physical_rows() == true_seq) return plan;
        const int64_t local_qhd = (num_q_heads() * head_dim()) / num_cores();
        const int64_t pad_elems =
            (plan.physical_rows() - true_seq) * local_qhd;
        const uint32_t off = static_cast<uint32_t>(true_seq * local_qhd * DWIDTH);
        rpu_launch_memset_spm_multicore((k_spm ? k_spm : addr(0, "k")) + off, pad_elems, num_cores());
        rpu_launch_memset_spm_multicore((v_spm ? v_spm : addr(0, "v")) + off, pad_elems, num_cores());
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
        const int64_t local_inter = FF / num_cores();

        const uint32_t scale = adaln_layer_scale(layer_idx, chunk);
        const uint32_t shift = scale + static_cast<uint32_t>(h * DWIDTH);

        if (!ctx().input_in_spm) emit_layer_input_dma(layer_idx, chunk);

        // AdaLayerNorm(norm1) = (1+scale)*LN(x)+shift -> norm_hidden (one LN call)
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual1"), addr(0, "norm_hidden"),
            scale, shift, seq_len, h, eps_, false, 0, num_cores());

        // QKV (col-partition) + bias
        if (!emit_self_qkv_multibase(layer_idx, chunk, lw)) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_SELF_Q_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, num_cores(), layer_addr(layer_idx, 0, "q_bias"), lw.to_q_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_SELF_K_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_k_w, addr(0, "k"),
            seq_len, nq * hd, h, 1, num_cores(), layer_addr(layer_idx, 0, "k_bias"), lw.to_k_ws);
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_SELF_V_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_v_w, addr(0, "v"),
            seq_len, nq * hd, h, 1, num_cores(), layer_addr(layer_idx, 0, "v_bias"), lw.to_v_ws);
        }

        // KV-insert(pos 0) + self-SDPA (non-causal, MASK_NONE). v16-pad: insert pad16(seq_len)
        // (zeroed tail) so the unified insert uses v16; SDPA below still reads kv_seq_len=seq_len.
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];
        const KvInsertSegmentPlan kv_plan = consume_kv_insert_plan(
            GR00T_DIT_SELF_KV_INSERT_SITE, seq_len, chunk.idx);
        if (kv_insert_flags(GR00T_DIT_SELF_KV_INSERT_SITE, chunk.idx) == GR00T_DIT_KV_FLAG_DDR_MIRROR) {
            rpu_launch_insert_kvcache_spm_unified_with_plan(
                k_cache, v_cache,
                addr_offset("k").value, addr_offset("v").value,
                nq, hd, num_cores(),
                /*k_cache_batch_offset_elems=*/0,
                /*v_cache_batch_offset_elems=*/0,
                /*spm_rows=*/kv_plan.physical_rows(), kv_plan);
        }
        const double attn_scale = 1.0 / std::sqrt(static_cast<double>(hd));
        // The aggregate layout can also represent self DDR plus cross raw.
        int64_t attention_site = GR00T_DIT_SELF_DDR_ATTENTION_SITE;
        if (ctx().has_complete_physical_manifest()) {
            for (const auto& route : ctx().physical_manifest().routes) {
                if (route.family == FmbRouteFamily::ATTENTION &&
                    route.invocation == chunk.idx &&
                    route.site_id == GR00T_DIT_SELF_RAW_ATTENTION_SITE) {
                    attention_site = route.site_id;
                }
            }
        }
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
                /*batch=*/1, seq_len, nq, hd, num_cores());
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q"), addr(0, "k"), addr(0, "sdpa_tmp"),
                addr(0, "sdpa_out"), /*mask_spm=*/0,
                /*MASK_NONE=*/0, attn_scale, /*batch=*/1,
                seq_len, seq_len, nq, nq, hd, num_cores());
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
                num_cores(), DecoderExecutionTopology::physical_kv_cores);
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
        const int64_t local_inter = FF / num_cores();
        TORCH_CHECK(S > 0, "gr00t_dit cross-attn block needs vl_embeds (vl_seq_>0)");

        // Layout policy is a union: self raw-SPM can coexist with cross DDR.
        // Resolve this consumer's actual route before deciding whether K/V is reusable.
        int64_t attention_site = GR00T_DIT_CROSS_DDR_ATTENTION_SITE;
        if (ctx().has_complete_physical_manifest()) {
            for (const auto& route : ctx().physical_manifest().routes) {
                if (route.family == FmbRouteFamily::ATTENTION &&
                    route.invocation == chunk.idx &&
                    route.site_id == GR00T_DIT_CROSS_RAW_ATTENTION_SITE) {
                    attention_site = route.site_id;
                }
            }
        }
        const AttentionExecutionPolicy attention_policy =
            ctx().physical_attention_policy_for_site(attention_site, chunk.idx);
        const bool raw_bank = ctx().has_complete_physical_manifest() &&
            attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA &&
            cross_kv_schedule(seq_len, ctx().batch_size, /*cross_raw=*/true) ==
                static_cast<int64_t>(Gr00tDiTCrossKvSchedule::REUSE_FIRST_BODY_SPM);
        const uint32_t bank_offset = raw_bank ? cross_kv_bank_offset(layer_idx) : 0;
        const char* k_slot = raw_bank ? "cross_k_bank" : "k";
        const bool transposed_bank = cross_v_transpose_reused(chunk, attention_policy);
        TORCH_CHECK(!transposed_bank || raw_bank, "GR00T transposed V requires a raw cross-KV bank");
        // Body0 projects into ordinary V scratch, then writes the transposed bank.
        // Each bank has pad16(S)*local_heads*head_dim halfs in either layout.
        const char* v_slot = raw_bank && !transposed_bank ? "cross_v_bank" : "v";
        const uint32_t v_offset = raw_bank && !transposed_bank ? bank_offset : 0;
        const uint32_t k_spm = addr(0, k_slot) + bank_offset;
        const uint32_t v_spm = addr(0, v_slot) + v_offset;
        auto& k_cache = (*ctx().k_caches)[layer_idx];
        auto& v_cache = (*ctx().v_caches)[layer_idx];

        const uint32_t scale = adaln_layer_scale(layer_idx, chunk);
        const uint32_t shift = scale + static_cast<uint32_t>(h * DWIDTH);

        if (!ctx().input_in_spm) emit_layer_input_dma(layer_idx, chunk);

        // AdaLayerNorm(norm1) -> norm_hidden (over the seq_len query tokens)
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual1"), addr(0, "norm_hidden"),
            scale, shift, seq_len, h, eps_, false, 0, num_cores());

        // Q from norm_hidden (col-partition) + bias
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_CROSS_Q_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.to_q_w, addr(0, "q"),
            seq_len, nq * hd, h, 1, num_cores(), layer_addr(layer_idx, 0, "q_bias"), lw.to_q_ws);

        // K/V from vl_embeds: linear([S,cross_dim], to_k/to_v[hidden,cross_dim]) -> [S,hidden]
        if (cross_kv_projection_needed(chunk, attention_policy)) {
            const uint32_t vl_input_addr =
                z2_bound_ ? z2_port_addr_ : addr(0, "vl_embeds");
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, GR00T_DIT_CROSS_K_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                vl_input_addr, lw.to_k_w, k_spm,
                S, nq * hd, cross_dim_, 1, num_cores(), layer_addr(layer_idx, 0, "k_bias"), lw.to_k_ws);
            ctx().consume_physical_route(
                FmbRouteFamily::LINEAR, GR00T_DIT_CROSS_V_LINEAR_SITE,
                static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_linear_spm_to_spm_acc16_kernel(
                vl_input_addr, lw.to_v_w, v_spm,
                S, nq * hd, cross_dim_, 1, num_cores(), layer_addr(layer_idx, 0, "v_bias"), lw.to_v_ws);

            // Insert S external K/V at pos 0, then cross-SDPA (seq_q=41, kv=S, MASK_2D). v16-pad:
            // insert pad16(S) (zeroed tail) for the v16 insert; SDPA below still reads kv_seq_len=S.
            const KvInsertSegmentPlan kv_plan = consume_kv_insert_plan(
                GR00T_DIT_CROSS_KV_INSERT_SITE, S, chunk.idx, k_spm, v_spm);
            if (kv_insert_flags(GR00T_DIT_CROSS_KV_INSERT_SITE, chunk.idx) == GR00T_DIT_KV_FLAG_DDR_MIRROR) {
                rpu_launch_insert_kvcache_spm_unified_with_plan(
                    k_cache, v_cache,
                    addr_offset(k_slot).value + bank_offset, addr_offset(v_slot).value + v_offset,
                    nq, hd, num_cores(),
                    /*k_cache_batch_offset_elems=*/0,
                    /*v_cache_batch_offset_elems=*/0,
                    /*spm_rows=*/kv_plan.physical_rows(), kv_plan);
            }
        }
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
            sdpa_dma_mask_to_spm(pm, addr_offset(mask_slot).value, seq_len, S, num_cores());
        }
        const double attn_scale = 1.0 / std::sqrt(static_cast<double>(hd));
        if (attention_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            TORCH_CHECK(pm.mask_type == 4,
                        "GR00T DiT raw-SPM cross attention requires MASK_2D");
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                GR00T_DIT_CROSS_RAW_ATTENTION_SITE,
                static_cast<int64_t>(
                    AttentionExecutionPolicy::SPM_KV_BY_MHA),
                /*resolved_flags=*/0, {}, chunk.idx);
            const uint32_t transposed_v = transposed_bank
                ? addr(0, "cross_v_bank") + bank_offset : addr(0, "sdpa_tmp");
            TORCH_CHECK(!transposed_bank || (ctx().body_iter >= 0 && ctx().body_iter < num_steps_),
                        "GR00T transposed V body out of bounds");
            // Fixed body0 refreshes the bank on EVERY inference/Graph replay.
            if (!transposed_bank || ctx().body_iter == 0) {
                rpu_launch_v_transpose_spm(
                    v_spm, transposed_v, /*batch=*/1, S, nq, hd, num_cores());
            }
            rpu_launch_sdpa_by_mha_spm(
                addr(0, "q"), k_spm, transposed_v,
                addr(0, "sdpa_out"), addr(0, mask_slot), pm.mask_type,
                attn_scale, /*batch=*/1, seq_len, S,
                nq, nq, hd, num_cores());
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
                num_cores(), DecoderExecutionTopology::physical_kv_cores);
        }

        finish_attn_and_ffn(layer_idx, chunk, lw, seq_len, h, FF, local_inter);
    }

    // Packed W4 row Linear takes no in-kernel bias. The row bias buffer holds the
    // bias on core 0 and zeros elsewhere, so adding it to every core's partial
    // before the reduction is the same sum the int8/fp16 kernel makes.
    void add_row_bias_partial(uint32_t bias_addr, int64_t seq_len, int64_t h) {
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            bias_addr, addr(0, "oproj"), addr(0, "oproj"),
            seq_len, h, c10::Half(1.0f), ValuOpType::ADD, /*is_bopa=*/false, num_cores());
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
        const bool w4_o = lw.to_out_w.scalar_type() == at::kByte;
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "sdpa_out"), lw.to_out_w, addr(0, "oproj"),
            seq_len, h, nq * hd, 0, num_cores(),
            w4_o ? 0 : layer_addr(layer_idx, 0, "out_bias"), lw.to_out_ws);
        if (w4_o) add_row_bias_partial(layer_addr(layer_idx, 0, "out_bias"), seq_len, h);
        if (!emit_attn_ring_xor3(chunk)) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE,
                GR00T_DIT_ATTN_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(seq_len, h, num_cores()),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
                seq_len, h, num_cores(), num_cores());
        }

        // norm3 = LayerNorm(affine=False) -> norm_hidden
        rpu_launch_layernorm_spm_kernel(
            addr(0, "residual2"), addr(0, "norm_hidden"),
            addr(0, "ln_ones"), addr(0, "ln_zeros"),
            seq_len, h, 1e-5, false, 0, num_cores());

        // FFN: ff1(col)+bias -> GELU(tanh-approx) -> ff2(row)+bias + PLAIN residual
        if (layer_idx == 0 && ctx().has_complete_physical_manifest() &&
            ff1_resident_selected()) {
            ctx().consume_physical_route(FmbRouteFamily::LINEAR,
                GR00T_DIT_FF1_RESIDENT_LINEAR_SITE, /*SPM_WEIGHT_M512N48K128=*/2, 0,
                ff1_resident_arguments(seq_len, ctx().batch_size), chunk.idx);
            rpu_launch_gr00t_ff1_resident_w8_kernel(
                addr(0, "norm_hidden"), addr(0, "ff1_resident_w8"), addr(0, "ff_mid"),
                layer_addr(layer_idx, 0, "ff1_bias"), lw.ff1_ws);
        } else {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_FF1_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "norm_hidden"), lw.ff1_w, addr(0, "ff_mid"),
            seq_len, FF, h, 1, num_cores(), layer_addr(layer_idx, 0, "ff1_bias"), lw.ff1_ws);
        }
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "ff_mid"), addr(0, "ff_mid"),
            seq_len * local_inter, ValuOpType::ADD,
            GeluMode::TANH, num_cores());
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_FF2_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0, {}, chunk.idx);
        const bool w4_ff2 = lw.ff2_w.scalar_type() == at::kByte;
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            addr(0, "ff_mid"), lw.ff2_w, addr(0, "oproj"),
            seq_len, h, FF, 0, num_cores(),
            w4_ff2 ? 0 : layer_addr(layer_idx, 0, "ff2_bias"), lw.ff2_ws);
        if (w4_ff2) add_row_bias_partial(layer_addr(layer_idx, 0, "ff2_bias"), seq_len, h);
        if (!emit_ff_ring_xor3(chunk)) {
            ctx().consume_physical_route(
                FmbRouteFamily::ALL_REDUCE, GR00T_DIT_FF_ALL_REDUCE_SITE,
                fmb_ring_all_reduce_route_selector(seq_len, h, num_cores()),
                /*resolved_flags=*/0, {}, chunk.idx);
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "oproj"), addr(0, "residual2"), addr(0, "residual1"),
                seq_len, h, num_cores(), num_cores());
        }

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
            out_features, bias_temp_spm_addr, num_cores());
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, GR00T_DIT_ADALN_LINEAR_SITE,
            static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE),
            /*resolved_flags=*/0,
            adaln_route_arguments(denoise_mode_, loop_mode_, vl_seq_, step_idx_));
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            cond_spm_addr, dense_w, partial_spm_addr,
            /*M=*/1, /*N=*/out_features, /*K=*/h,
            /*partition=*/0, num_cores(), /*bias_spm_addr=*/0, scale);
        ctx().consume_physical_route(
            FmbRouteFamily::ALL_REDUCE,
            GR00T_DIT_ADALN_ALL_REDUCE_SITE,
            fmb_ring_all_reduce_route_selector(
                /*rows=*/1, /*cols=*/out_features, num_cores()),
            /*resolved_flags=*/0);
        rpu_launch_all_reduce_sum_residual_kernel(
            partial_spm_addr, bias_temp_spm_addr, gemv_out_spm_addr,
            /*M=*/1, /*N=*/out_features, num_cores(), num_cores());
        rpu_launch_eltwise_binary_scalar_spm_kernel(
            gemv_out_spm_addr, c10::Half(1.0), gemv_out_spm_addr, h, ValuOpType::ADD,
            num_cores());
    }

    bool load_adaln_body_table() {
        if (!ctx().has_complete_physical_manifest() ||
            !adaln_table_profile(ctx().seq_len, ctx().batch_size)) return false;
        const int64_t body = ctx().body_iter;
        TORCH_CHECK(body >= 0 && body < num_steps_, "GR00T AdaLN table body out of bounds");
        ctx().consume_physical_route(FmbRouteFamily::MUTABLE_DMA,
            GR00T_DIT_ADALN_TABLE_DMA_SITE,
            static_cast<int64_t>(Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM), 0,
            adaln_table_arguments(ctx().seq_len, ctx().batch_size), body);
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            adaln_table_src_base_, adaln_table_, body * adaln_table_body_bytes(),
            adaln_table_body_bytes() / DWIDTH, addr(0, "adaln_body_table"), num_cores());
        return true;
    }

    bool load_time_table(bool tau) {
        if (!ctx().has_complete_physical_manifest() ||
            !time_table_profile(ctx().seq_len, ctx().batch_size)) return false;
        const int64_t body = ctx().body_iter;
        TORCH_CHECK(body >= 0 && body < num_steps_, "GR00T time table body out of bounds");
        const int64_t elements = (tau ? action_horizon_ : 2) * hidden_size();
        ctx().consume_physical_route(FmbRouteFamily::MUTABLE_DMA,
            GR00T_DIT_TIME_TABLE_DMA_SITE,
            static_cast<int64_t>(Gr00tDiTMutableDmaRoute::DDR_BROADCAST_TO_SPM), 0,
            time_table_arguments(ctx().seq_len, ctx().batch_size), body * 2 + (tau ? 0 : 1));
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            tau ? time_tau_src_base_ : time_po1_src_base_,
            tau ? time_tau_table_ : time_po1_table_, body * elements * DWIDTH,
            elements, addr(0, tau ? "enc2" : "so"), /*num_cores=*/1);
        return true;
    }

    // ============== Phase B/C — pre-hook: encoders -> sa_embs (emb_stage_) + cond ==============
    // All single-core (core 0); the cond scatter (8-core) feeds the multi-core DiT adaln.
    // step = step_idx_ (step form) or ctx().body_iter (unroll). See build_denoise_step.py design.
    void emit_pre_layers_body() {
        load_ff1_resident_weight();
        const int64_t h = hidden_size();
        const int64_t AH = action_horizon_, NP = action_dim_pad_, EM = enc_mid_;
        const int64_t bit = ctx().body_iter;
        const int64_t step = loop_mode_ ? bit : step_idx_;
        const int64_t cond_off = step * h;
        const uint32_t tau_step = addr(0, "tau_table") + static_cast<uint32_t>(step * AH * h * DWIDTH);
        const int64_t local_k = h / num_cores();
        const bool time_tables = time_tables_active();

        // cond[step]: 8-core scatter -> "cond" (DiT adaln) + core0 broadcast -> "cond_full" (norm_out)
        if (!load_adaln_body_table()) {
            rpu_launch_ddr_scatter_spm_dma(
                cond_table_t_, cond_off, local_k, local_k * DWIDTH,
                addr(0, "cond"), num_cores());
        }
        if (!time_tables) {
            rpu_launch_ddr_broadcast_spm_dma(
                cond_table_t_, cond_off, h, addr(0, "cond_full"),
                /*num_cores=*/1);
        }

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
                                                        addr(0, "se_mid"), EM, ValuOpType::MAX,
                                                        num_cores());
            sc_lin(addr(0, "se_mid"), se2_w_, addr(0, "state_feat"), 1, h, EM, addr(0, "se2_bias"), se2_ws_);
        }

        // action encoder: a_emb = x_t @ ae1 + b1 ; xx = silu(a_emb@W2a + b2 + tau@W2t) ; af = xx@W3 + b3 + pos
        sc_lin(addr(0, "x_t_spm"), ae1_w_, addr(0, "enc0"), AH, h, NP, addr(0, "ae1_bias"));
        sc_lin(addr(0, "enc0"), ae2a_w_, addr(0, "enc1"), AH, h, h, addr(0, "ae2_bias"), ae2a_ws_);   // + b2 (folded)
        if (time_tables) load_time_table(/*tau=*/true);
        else sc_lin(tau_step, ae2t_w_, addr(0, "enc2"), AH, h, h, /*bias=*/0, ae2t_ws_);
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
        const uint32_t shift_a = addr(0, "so");
        const uint32_t scale_a = addr(0, "so") + static_cast<uint32_t>(h * DWIDTH);
        if (!load_time_table(/*tau=*/false)) {
            sc_lin(addr(0, "cond_full"), po1_w_, addr(0, "so"), 1, 2 * h, h, addr(0, "po1_bias"), po1_ws_);
            rpu_launch_eltwise_binary_scalar_spm_kernel(scale_a, c10::Half(1.0), scale_a, h, ValuOpType::ADD,
                                                        num_cores());
        }
        rpu_launch_layernorm_spm_kernel(addr(0, "residual1"), addr(0, "normed"),
                                        scale_a, shift_a, SQ, h, 1e-6, false, 0, /*num_cores=*/1);
        sc_lin(addr(0, "normed"), po2_w_, addr(0, "model_out"), SQ, MO, h, addr(0, "po2_bias"), po2_ws_);

        // action_decoder: d1 = relu(model_out @ dec1 + b1) ; pred = d1 @ dec2 + b2 (N-pad 144)
        sc_lin(addr(0, "model_out"), dec1_w_, addr(0, "dec_mid"), SQ, DM, MO, addr(0, "dec1_bias"), dec1_ws_);
        rpu_launch_eltwise_binary_scalar_spm_kernel(addr(0, "dec_mid"), c10::Half(0.0),
                                                    addr(0, "dec_mid"), SQ * DM, ValuOpType::MAX,
                                                    num_cores());
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
        unroll_mask_publications_ = {};
        PreparedMask pm = sdpa_prepare_mask(
            m, /*is_causal=*/false, seq_q, vl_seq_,
            sdpa_stable_mask_cache(), ordinal);
        slot = pm.mask_type == 4 ? pm.ddr_tensor : at::Tensor{};
        return pm;
    }

private:
    // Only ordinary unroll may retain completed CPU publications. Every other
    // cross-mask writer uses prep_mask_into and revokes both private receipts.
    struct Gr00tMaskSpan {
        const void* tensor_impl = nullptr;
        const void* storage_impl = nullptr;
        uintptr_t base = 0, data = 0;
        size_t storage_bytes = 0, bytes = 0;
        int64_t offset = 0, rows = 0, cols = 0, stride0 = 0, stride1 = 0;
        bool snapshot(const at::Tensor& tensor) {
            if (!tensor.defined() || tensor.device().type() != at::kPrivateUse1 ||
                tensor.dim() != 2 || tensor.scalar_type() != at::kHalf ||
                !tensor.is_contiguous() || tensor.nbytes() == 0)
                return false;
            const auto storage = tensor.storage();
            *this = {tensor.unsafeGetTensorImpl(), storage.unsafeGetStorageImpl(),
                reinterpret_cast<uintptr_t>(storage.data_ptr().get()),
                reinterpret_cast<uintptr_t>(tensor.data_ptr()), storage.nbytes(), tensor.nbytes(),
                tensor.storage_offset(), tensor.size(0), tensor.size(1),
                tensor.stride(0), tensor.stride(1)};
            const auto max = std::numeric_limits<uintptr_t>::max();
            return base != 0 && data >= base && offset >= 0 &&
                storage_bytes <= max - base && bytes <= max - data &&
                data + bytes <= base + storage_bytes &&
                static_cast<uint64_t>(offset) <= storage_bytes / sizeof(c10::Half) &&
                data == base + static_cast<size_t>(offset) * sizeof(c10::Half);
        }
        bool operator==(const Gr00tMaskSpan& other) const {
            return tensor_impl == other.tensor_impl && storage_impl == other.storage_impl &&
                base == other.base && data == other.data && storage_bytes == other.storage_bytes &&
                bytes == other.bytes && offset == other.offset && rows == other.rows &&
                cols == other.cols && stride0 == other.stride0 && stride1 == other.stride1;
        }
        bool disjoint(uintptr_t other_base, size_t other_bytes) const {
            const auto max = std::numeric_limits<uintptr_t>::max();
            return other_base != 0 && other_bytes <= max - other_base &&
                (base + storage_bytes <= other_base ||
                 other_base + other_bytes <= base);
        }
    };
    struct Gr00tMaskPublication {
        PreparedMask mask;
        Gr00tMaskSpan span;
        uint64_t generation = 0;
        int64_t logical_k = 0;
        bool published = false;
    };
    std::array<Gr00tMaskPublication, 2> unroll_mask_publications_;

    static bool unroll_mask_matches(const at::Tensor& source,
                                    const Gr00tMaskPublication& receipt) {
        const auto* input = source.data_ptr<c10::Half>();
        const auto* target = receipt.mask.ddr_tensor.data_ptr<c10::Half>();
        const c10::Half masked(-std::numeric_limits<float>::infinity());
        for (int64_t row = 0; row < receipt.span.rows; ++row) {
            if (std::memcmp(input + row * receipt.logical_k,
                    target + row * receipt.span.cols,
                    static_cast<size_t>(receipt.logical_k) * sizeof(c10::Half)) != 0)
                return false;
            for (int64_t col = receipt.logical_k; col < receipt.span.cols; ++col)
                if (std::memcmp(target + row * receipt.span.cols + col,
                        &masked, sizeof(c10::Half)) != 0)
                    return false;
        }
        return true;
    }

    void prepare_unroll_masks(const at::Tensor& tmask, const at::Tensor& imask,
                              int64_t seq_q) {
        try {
            const std::array<const at::Tensor*, 2> sources{&tmask, &imask};
            std::array<uintptr_t, 2> source_base{};
            std::array<size_t, 2> source_bytes{};
            const auto maximum = std::numeric_limits<uintptr_t>::max();
            const uint64_t generation = installed_model_state_generation();
            bool cacheable = g_rpu_ddr_flush_force_enabled;
            // Inspect BOTH original backings before either slot can hit or write.
            // A narrowed CPU alias may overlap a private owner only in its tail.
            for (size_t i = 0; cacheable && i < sources.size(); ++i) {
                const auto& source = *sources[i];
                cacheable = source.defined() && source.device().is_cpu() &&
                    source.layout() == at::kStrided && source.dim() == 2 &&
                    source.scalar_type() == at::kHalf && source.is_contiguous() &&
                    !source.requires_grad() && !source.is_neg() && !source.is_conj() &&
                    source.size(0) == seq_q &&
                    source.size(1) == vl_seq_ && source.nbytes() != 0;
                if (!cacheable) break;
                const auto storage = source.storage();
                source_base[i] = reinterpret_cast<uintptr_t>(storage.data_ptr().get());
                source_bytes[i] = storage.nbytes();
                const auto data = reinterpret_cast<uintptr_t>(source.data_ptr());
                const auto offset = source.storage_offset();
                cacheable = source_base[i] != 0 && source_bytes[i] != 0 &&
                    source_bytes[i] <= maximum - source_base[i] &&
                    data >= source_base[i] && source.nbytes() <= maximum - data &&
                    data + source.nbytes() <= source_base[i] + source_bytes[i] &&
                    offset >= 0 && static_cast<uint64_t>(offset) <= source_bytes[i] / sizeof(c10::Half) &&
                    data == source_base[i] + static_cast<size_t>(offset) * sizeof(c10::Half);
            }
            for (const auto& receipt : unroll_mask_publications_) {
                if (!cacheable || !receipt.published) continue;
                Gr00tMaskSpan current;
                cacheable = receipt.generation == generation &&
                    current.snapshot(receipt.mask.ddr_tensor) && current == receipt.span &&
                    current.disjoint(source_base[0], source_bytes[0]) &&
                    current.disjoint(source_base[1], source_bytes[1]);
            }
            if (!cacheable) unroll_mask_publications_ = {};
            std::array<PreparedMask*, 2> prepared{&prepared_cross_mask_, &prepared_cross_mask_img_};
            std::array<at::Tensor*, 2> slots{&text_mask_slot_, &image_mask_slot_};
            for (size_t i = 0; i < sources.size(); ++i) {
                auto& receipt = unroll_mask_publications_[i];
                if (cacheable && receipt.published && receipt.logical_k == vl_seq_ &&
                    receipt.span.rows == seq_q &&
                    receipt.span.cols == Align(vl_seq_, 16) &&
                    (c10::InferenceMode::is_enabled() || !receipt.mask.ddr_tensor.is_inference()) &&
                    unroll_mask_matches(*sources[i], receipt)) {
                    *prepared[i] = receipt.mask;
                    *slots[i] = receipt.mask.ddr_tensor;
                    continue;
                }
                receipt.published = false;
                // Keep shared preparation, public device acquisition and copy_ error
                // behavior unchanged. A failed second copy revokes the whole pair.
                *prepared[i] = sdpa_prepare_mask(*sources[i], false, seq_q, vl_seq_,
                                                sdpa_stable_mask_cache(), i + 1);
                *slots[i] = prepared[i]->mask_type == 4 ? prepared[i]->ddr_tensor : at::Tensor{};
                if (!cacheable) continue;
                Gr00tMaskSpan span;
                bool valid = prepared[i]->mask_type == 4 && span.snapshot(*slots[i]) &&
                    span.rows == seq_q && span.cols == Align(vl_seq_, 16) &&
                    span.disjoint(source_base[0], source_bytes[0]) &&
                    span.disjoint(source_base[1], source_bytes[1]);
                const auto& other = unroll_mask_publications_[1 - i];
                if (valid && other.published)
                    valid = span.disjoint(other.span.base, other.span.storage_bytes);
                if (!valid || !g_rpu_ddr_flush_force_enabled ||
                    installed_model_state_generation() != generation) {
                    unroll_mask_publications_ = {};
                    cacheable = false;
                } else {
                    receipt = {*prepared[i], span, generation, vl_seq_, true};
                }
            }
        } catch (...) {
            unroll_mask_publications_ = {};
            throw;
        }
    }

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
        if (adaln_table_.defined()) {
            identity.insert(identity.end(), {adaln_table_generation_, num_steps_,
                                             adaln_table_body_bytes()});
            append_kvinsert_cost_tensor_identity(identity, cond_table_t_);
            append_kvinsert_cost_tensor_identity(identity, adaln_table_);
        }
        if (time_tau_table_.defined()) {
            identity.insert(identity.end(), {time_table_generation_, num_steps_, action_horizon_});
            for (const auto* tensor : {&cond_table_t_, &tau_table_, &time_tau_table_, &time_po1_table_})
                append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        if (self_qkv_multibase_opt_in_) {
            identity.insert(identity.end(), {GR00T_DIT_SELF_QKV_MULTIBASE_SITE,
                                             self_qkv_multibase_ready_ ? 1 : 0});
            for (const auto& layer : layer_weights_) identity.push_back(layer.is_cross ? 1 : 0);
        }
        if (ring_xor3_opt_in_) {
            identity.insert(identity.end(), {GR00T_DIT_ATTN_RING_XOR3_SITE,
                GR00T_DIT_FF_RING_XOR3_SITE, ring_xor3_ready_ ? 1 : 0});
        }
        if (ff1_resident_weight_.defined()) {
            identity.insert(identity.end(), {ff1_resident_generation_, GR00T_DIT_FF1_RESIDENT_BYTES});
            append_kvinsert_cost_tensor_identity(identity, ff1_resident_weight_);
        }
        return identity;
    }

    std::vector<LayerWeights> layer_weights_;
    bool self_qkv_multibase_ready_ = false;
    bool ring_xor3_ready_ = false;
    at::Tensor ff1_resident_weight_, ff1_resident_source_;
    uint64_t ff1_resident_src_base_ = 0;
    int64_t ff1_resident_generation_ = 0, ff1_resident_source_version_ = 0;
    at::Tensor time_tau_table_, time_po1_table_;
    uint64_t time_tau_src_base_ = 0, time_po1_src_base_ = 0;
    int64_t time_table_generation_ = 0;
    std::vector<std::pair<at::Tensor, int64_t>> time_sources_;
    at::Tensor adaln_table_;
    uint64_t adaln_table_src_base_ = 0;
    int64_t adaln_table_generation_ = 0;
    bool inference_started_ = false;
    std::vector<std::pair<at::Tensor, int64_t>> adaln_sources_;
    double eps_ = 1e-5;
    int64_t cross_dim_ = 2048;
    SdpaKernelType sdpa_kernel_ = SdpaKernelType::FLASH_ATTN_SPM;

    // norm3 affine=False constants (gamma=1, beta=0), DDR-resident on device.
    at::Tensor ln_ones_, ln_zeros_;

    // Per-forward cond (= SiLU(temb)) state (mirrors AdaRMS).
    at::Tensor cond_ref_;
    uint64_t   cond_src_base_ = 0;
    bool       cond_loaded_this_forward_ = false;

    // Per-forward vl_embeds (cross-attn K/V source) + prepared MASK_2D.
    const bool cross_kv_reuse_opt_in_;        // cold handle policy; never reread during REPLAY
    const bool kv_spm_only_opt_in_;
    const bool cross_v_transpose_reuse_opt_in_;
    const bool ff1_resident_w8_opt_in_;
    const bool self_qkv_multibase_opt_in_;
    const bool ring_xor3_opt_in_;
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
    std::vector<int64_t> z2_stage_descriptor_;
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

void rpu_gr00t_dit_prepare_time_tables(int64_t handle) {
    Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_prepare_time_tables")->prepare_time_tables();
}

void rpu_gr00t_dit_prepare_adaln_table(int64_t handle) {
    Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_prepare_adaln_table")->prepare_adaln_table();
}

void rpu_gr00t_dit_set_execution_core_count(int64_t handle, int64_t cores) {
    Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_set_execution_core_count")
        ->set_execution_cores(cores);
}

std::vector<int64_t> rpu_gr00t_dit_get_execution_topology(int64_t handle) {
    return Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_get_execution_topology")
        ->execution_topology();
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

void rpu_gr00t_dit_prepare_persistent_spm(int64_t handle, int64_t execution_len) {
    Gr00tDiTRegistry::get(handle, "rpu_gr00t_dit_prepare_persistent_spm")
        ->prepare_persistent_spm(execution_len, 0, false);
}
