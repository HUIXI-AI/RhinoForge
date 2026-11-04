// rpu_qwen3_5_model.cpp — Qwen3.5 hybrid-attention fused model (v3).
// See rpu_qwen3_5_model.h for scope. Full-attention emission mirrors
// CausalDecoderModel::build_layer_subgraph (rpu_qwen3_model.cpp) for the
// non-M-RoPE, QK-norm path; GDN layers dispatch to build_gdn(decode) by chunk.len.
//
// ── FILE MAP (read order: main flow → mixers → buffers/config → setup → C-API) ──
//   MAIN FLOW
//     forward()               — Python entry's workhorse; drives run_all_layers
//     build_layer_subgraph()  — assembles ONE layer: input DMA → input_layernorm →
//                               token mixer → shared post_norm + MLP + output
//   TOKEN MIXERS (read normed "input_norm"; end with all_reduce that adds h_in residual)
//     build_full_attention()  — QK-norm + (partial M-)RoPE + SDPA + gated o_proj
//     build_gdn()             — unified GDN mixer; if(decode) splits recurrent (seq=1)
//                               vs chunked delta-rule (prefill, L>1); shared setup/proj/tail
//     emit_mlp_and_output()   — shared post-mixer tail (post_norm + MLP + final_norm + out)
//   SPM LAYOUT       declare_buffers() — full-attn + the active GDN decode/prefill set
//   GRAPH-PLAN HOOKS static_config / dynamic_config / subclass_chunk_size_valid
//   SETUP            set_weights() (store swizzled weights)
//   C-API (bottom)   rpu_qwen3_5_* free funcs — Python handle ↔ object bridges
#include "rpu_qwen3_5_model.h"
#include "rpu_qwen3_5_spm_z2.h"
#include "qwen3_5_execution_topology.h"

#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_spm_buffers.h"
#include "rpu_spm_residency.h"
#include "rpu_spm_pipeline.h"
#include "rpu_helpers.h"
#include "rpu_profile.h"         // rpu_ddr_flush
#include "rpu_linear_tiling.h"
#include "rpu_pi05_owner_norm.h"
#include "rpu_runtime_state.h"   // diagnostic-only hidden-state export
#include "execution_coordinator.h"

#include <c10/util/ScopeExit.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>     // std::sqrt (GDN qk_scale)
#include <cstring>
#include <limits>
#include <tuple>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace {

constexpr uint32_t QWEN35_TEXT_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
    KV_INSERT_CAP_NON8_TP_V16;
constexpr int64_t QWEN35_TEXT_KV_REASON_DDR_REQUIRED = 1;

enum Qwen35TextAttentionRouteFlags : int64_t {
    QWEN35_TEXT_ATTN_PREFIX_HISTORY_DDR_REQUIRED = 1LL << 0,
    QWEN35_TEXT_ATTN_MULTI_CHUNK_DDR_REQUIRED = 1LL << 1,
    QWEN35_TEXT_ATTN_RAW_KERNEL_INCOMPATIBLE_DDR_REQUIRED = 1LL << 2,
    QWEN35_TEXT_ATTN_ACTION_PREFIX_DDR_REQUIRED = 1LL << 3,
    QWEN35_TEXT_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED = 1LL << 4,
};

constexpr int64_t QWEN35_TEXT_LINEAR_SITE = 4115846627077743284LL;
constexpr int64_t QWEN35_ACTION_NVFP4_LINEAR_SITE = 1171762392203719440LL;
constexpr int64_t QWEN35_TEXT_FULL_PREPARE_ALL_REDUCE_SITE =
    4219478719568483423LL;
constexpr int64_t QWEN35_TEXT_FULL_ALL_REDUCE_SITE =
    2379894461900596701LL;
constexpr int64_t QWEN35_TEXT_GDN_ALL_REDUCE_SITE =
    5015834406872099990LL;
constexpr int64_t QWEN35_TEXT_MLP_ALL_REDUCE_SITE =
    700438019929736867LL;
constexpr int64_t QWEN35_TEXT_CORE_TOPOLOGY_SITE = 3215916181030544856LL;
constexpr int64_t QWEN35_TEXT_GDN_PREPARE_ALL_REDUCE_SITE = 6288021177778164369LL;
constexpr int64_t QWEN35_TEXT_GDN_MASK_RESIDENCY_SITE = 120356759196541611LL;
constexpr int64_t QWEN35_TEXT_GDN_LOCAL_SLICE_SITE = 4624574716066886237LL;
constexpr int64_t QWEN35_TEXT_OWNER_RMS_SITE = 8722488789555919585LL;
constexpr int64_t QWEN35_TEXT_GDN_OWNER_RMS_SITE = 7871266237092441430LL;
constexpr int64_t QWEN35_TEXT_MLP_TAIL_SITE = 8475656447499929675LL;

constexpr int64_t QWEN35_TEXT_PARTIAL_ROPE_1D_SITE =
    2169018622841064289LL;
constexpr int64_t QWEN35_TEXT_PARTIAL_MROPE_SITE =
    6715828002838469508LL;
constexpr int64_t QWEN35_TEXT_Q_NORM_ROPE_SITE =
    3390471367801678782LL;
constexpr int64_t QWEN35_TEXT_K_NORM_ROPE_SITE =
    3432209414188589631LL;
constexpr int64_t QWEN35_TEXT_Q_ROPE_SITE = 7236156864873926LL;
constexpr int64_t QWEN35_TEXT_K_ROPE_SITE = 7698060160224136988LL;

constexpr int64_t QWEN35_TEXT_ADAPTIVE_MOD_DMA_SITE =
    4776555215188706607LL;
constexpr int64_t QWEN35_ACTION_FUSED_NORM_SHIFT_SITE = 2620485673566908366LL;
constexpr int64_t QWEN35_ACTION_RMS_NORM_SHIFT_SITE = 1358207896742669459LL;
constexpr int64_t QWEN35_ACTION_RESIDUAL_GATE_SITE = 2460629872091638979LL;
constexpr int64_t QWEN35_ACTION_RTC_RESIDUAL_GATE_SITE = 8612470743072654560LL;
constexpr int64_t QWEN35_ACTION_TAIL_SITE = 7035949248365271364LL;
constexpr int64_t QWEN35_ACTION_MLP_TAIL_SITE = 4368884220398897667LL;
constexpr int64_t QWEN35_ACTION_RING_RTC_RESIDUAL_SITE = 7969833234331291786LL;
constexpr int64_t QWEN35_ACTION_GATEUP_RESIDENT_SITE = 2095553279890566356LL;
constexpr int64_t QWEN35_ACTION_GATEUP_PRELOAD_SITE = 4846322196319892473LL;
constexpr int64_t QWEN35_ACTION_KV_OWNER_SITE = 9072654722328257913LL;
constexpr int64_t QWEN35_ACTION_Q_GATE_SITE = 2511688968361667267LL;
constexpr int64_t QWEN35_ACTION_GATEUP_PAIR_SITE = 5257909631264440265LL;
constexpr int64_t QWEN35_ACTION_GATEUP_RESIDENT_W8_SITE = 3829831243603816656LL;
constexpr int64_t QWEN35_ACTION_GATEUP_PRELOAD_W8_SITE = 6555739373369272234LL;
constexpr int64_t QWEN35_ACTION_O_RESIDENT_W8_SITE = 4430871007764841339LL;
constexpr int64_t QWEN35_ACTION_O_PRELOAD_W8_SITE = 4233487996477166511LL;
constexpr int64_t QWEN35_ACTION_O_RESIDENT_NVFP4_SITE = 5103834779006160382LL;
constexpr int64_t QWEN35_ACTION_O_PRELOAD_NVFP4_SITE = 4699291274527758253LL;
constexpr int64_t QWEN35_ACTION_DOWN_RESIDENT_NVFP4_SITE = 5936546264671002449LL;
constexpr int64_t QWEN35_ACTION_DOWN_PRELOAD_NVFP4_SITE = 1330477162992561100LL;
constexpr int64_t QWEN35_ACTION_DOWN_PARTIAL_W8_SITE = 7974993165105200463LL;
constexpr int64_t QWEN35_ACTION_DOWN_PARTIAL_PRELOAD_W8_SITE = 7768637646996913103LL;
constexpr int64_t QWEN35_ACTION_KV_OWNER_W8_SITE = 73490380550116103LL;
constexpr int64_t QWEN35_ACTION_GATE_KV_OWNER_W8_SITE = 5392460504564407201LL;
constexpr int64_t QWEN35_ACTION_ATTN_SIGMOID_MUL_SITE = 8546526474721653363LL;
constexpr int64_t QWEN35_ACTION_QK_NORM_ROPE_GROUP4_SITE = 696118694548312598LL;
constexpr int64_t QWEN35_ACTION_Q_GATE_W8_SITE = 5855704986459519851LL;
constexpr int64_t QWEN35_ACTION_QKV_GATE_W8_SITE = 2447844358394284528LL;
constexpr int64_t QWEN35_ACTION_QKV_GATE_OWNER_ONLY_W8_SITE = 558172469151339648LL;
constexpr int64_t QWEN35_ACTION_K_OWNER_NORM_ROPE_SITE = 2641642310566033775LL;
constexpr int64_t QWEN35_ACTION_OWNER_CACHE_INSERT_SITE = 8281105153491195213LL;
constexpr int64_t QWEN35_ACTION_OWNER_CACHE_JOINT_SITE = 7450172011194584301LL;
constexpr int64_t QWEN35_ACTION_GATEUP_PAIR_W8_SITE = 4700085927040894734LL;
constexpr int64_t QWEN35_ACTION_GATEUP_PAIR_NVFP4_SITE = 4571887477213075828LL;
constexpr int64_t QWEN35_ACTION_GATEUP_SWIGLU_NVFP4_SITE = 1871355692309183287LL;
constexpr int64_t QWEN35_ACTION_GATEUP_PAIR_FP16_ACC16_SITE = 4279327644766908377LL;
constexpr int64_t QWEN35_TEXT_GATEUP_PAIR_FP16_ACC16_SITE = 2576448788894560734LL;
constexpr int64_t QWEN35_ACTION_O_RESIDENT_NVFP4_ACC32_SITE = 4187530286635807349LL;
constexpr int64_t QWEN35_ACTION_DOWN_RESIDENT_NVFP4_ACC32_SITE = 8057254336740368708LL;
constexpr int64_t QWEN35_ACTION_GATEUP_PAIR_NVFP4_ACC32_SITE = 9073505328675940699LL;
constexpr int64_t QWEN35_ACTION_GATEUP_SWIGLU_NVFP4_ACC32_SITE = 4752878381648738023LL;
constexpr int64_t QWEN35_ACTION_RESIDENT_MATRIX_BYTES = 1048576;
constexpr int64_t QWEN35_ACTION_RESIDENT_BUNDLE_BYTES = 2097152;
constexpr int64_t QWEN35_ACTION_RESIDENT_LAYERS = 3;
constexpr int64_t QWEN35_ACTION_RESIDENT_W8_LAYERS = 6;
constexpr int64_t QWEN35_ACTION_RESIDENT_TOTAL_BYTES =
    QWEN35_ACTION_RESIDENT_LAYERS * QWEN35_ACTION_RESIDENT_BUNDLE_BYTES;
constexpr int64_t QWEN35_TEXT_ACTION_INPUT_DMA_SITE =
    5753123845172482927LL;
constexpr int64_t QWEN35_TEXT_ACTION_RTC_PREFIX_DMA_SITE =
    3125805963720679503LL;
constexpr int64_t QWEN35_TEXT_ACTION_TRACE_PRE_DMA_SITE =
    5950445703812717097LL;
constexpr int64_t QWEN35_TEXT_ACTION_TRACE_EMBED_DMA_SITE =
    530244801951758262LL;
constexpr int64_t QWEN35_TEXT_ACTION_TRACE_HIDDEN_DMA_SITE =
    7825855580192819414LL;
constexpr int64_t QWEN35_TEXT_ACTION_KEEP_MASK_DMA_SITE =
    6395936005365706696LL;
constexpr int64_t QWEN35_TEXT_ACTION_TRACE_VELOCITY_DMA_SITE =
    1593989881490609202LL;
constexpr int64_t QWEN35_TEXT_ACTION_TRACE_POST_DMA_SITE =
    4453412602072100184LL;
constexpr int64_t QWEN35_TEXT_ACTION_OUTPUT_DMA_SITE =
    3683811421906687568LL;
constexpr int64_t QWEN35_TEXT_ACTION_INPUT_BIAS_PRELOAD_SITE =
    1610850368048667879LL;
constexpr int64_t QWEN35_TEXT_ACTION_OUTPUT_BIAS_PRELOAD_SITE =
    3726963074905085368LL;
constexpr int64_t QWEN35_TEXT_Q_NORM_PRELOAD_SITE =
    1967194233859833524LL;
constexpr int64_t QWEN35_TEXT_K_NORM_PRELOAD_SITE =
    8085492494885672497LL;
constexpr int64_t QWEN35_TEXT_ACTION_SCHEDULE_SITE =
    4167953566644294997LL;
constexpr int64_t QWEN35_TEXT_QK_NORM_SCHEDULE_SITE =
    7233132720578852326LL;
constexpr int64_t QWEN35_TEXT_SILU_ROUND_MUL_SITE = 7739768473980724746LL;
constexpr int64_t QWEN35_TEXT_MLP_SILU_SITE = 5544299293668623348LL;
constexpr int64_t QWEN35_TEXT_HIGH_PRECISION_SITE = 2088803336706670510LL;
constexpr int64_t QWEN35_ACTION_HIGH_PRECISION_SITE = 6744413876730691692LL;
constexpr int64_t QWEN35_TEXT_GATEUP_PAIR_SITE = 751944759910871833LL;
constexpr int64_t QWEN35_TEXT_GATEUP_PAIR_W8_SITE = 267678052960427182LL;
constexpr int64_t QWEN35_ACTION_SILU_ROUND_MUL_SITE = 3269136659581603685LL;
constexpr int64_t QWEN35_ACTION_SILU_HIGH_MUL_SITE = 5790657297908870019LL;
constexpr int64_t QWEN35_ACTION_NEWTON_NORM_SHIFT_SITE = 5482108007151016838LL;

constexpr int64_t QWEN35_TEXT_GDN_STATE_LOAD_DMA_SITE =
    3058759894390714248LL;
constexpr int64_t QWEN35_TEXT_GDN_CONV_DECODE_LOAD_DMA_SITE =
    4006260985833968847LL;
constexpr int64_t QWEN35_TEXT_GDN_CONV_PREFILL_LOAD_DMA_SITE =
    2783544813801395793LL;
constexpr int64_t QWEN35_TEXT_GDN_CONV_PREFILL_STORE_DMA_SITE =
    2740211351482404404LL;
constexpr int64_t QWEN35_TEXT_GDN_STATE_PREFILL_LOAD_DMA_SITE =
    5713030811346189917LL;
constexpr int64_t QWEN35_TEXT_GDN_STATE_PREFILL_STORE_DMA_SITE =
    416809216438236659LL;
constexpr int64_t QWEN35_TEXT_GDN_STATE_DECODE_STORE_DMA_SITE =
    222275318630817083LL;
constexpr int64_t QWEN35_TEXT_GDN_CONV_DECODE_STORE_DMA_SITE =
    1343355373711461289LL;

constexpr int64_t QWEN35_ACTION_ROUTE_IO = 1LL << 0;
constexpr int64_t QWEN35_ACTION_ROUTE_RTC = 1LL << 1;
constexpr int64_t QWEN35_ACTION_ROUTE_TRACE = 1LL << 2;
constexpr int64_t QWEN35_ACTION_ROUTE_MASK =
    QWEN35_ACTION_ROUTE_IO | QWEN35_ACTION_ROUTE_RTC |
    QWEN35_ACTION_ROUTE_TRACE;

enum class Qwen35TextAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

enum class Qwen35TextRopeRoute : int64_t {
    PARTIAL_ROPE_1D = 1,
    PARTIAL_MROPE = 2,
    ROPE_SPM = 3,
};

enum class Qwen35TextMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM = 1,
    DDR_SCATTER_TO_SPM = 2,
    SPM_COPY_TO_DDR = 3,
    SPM_SCATTER_TO_DDR = 4,
    DDR_BROADCAST_MOD_BODY = 5,
};

constexpr int64_t qwen35_text_rope_route(Qwen35TextRopeRoute route) {
    return static_cast<int64_t>(route);
}

constexpr int64_t qwen35_text_dma_route(Qwen35TextMutableDmaRoute route) {
    return static_cast<int64_t>(route);
}

// PARTIAL_MROPE encodes DDR bases in 256-byte units (addr >> 8). The caching
// allocator guarantees only 32-byte alignment, so retain an aligned view when
// needed instead of silently truncating address bits in the launcher.
at::Tensor keep_256b_aligned_rpu_copy(const at::Tensor& src,
                                      const char* name) {
    TORCH_CHECK(src.defined() && src.device().type() == at::kPrivateUse1
                && src.scalar_type() == at::kHalf && src.is_contiguous(),
                name, " must be a contiguous FP16 device tensor");
    const uint64_t src_addr = RpuGetDevAddr(src.data_ptr<c10::Half>());
    if ((src_addr & 0xFF) == 0)
        return src;

    constexpr int64_t kAlignmentBytes = 256;
    constexpr int64_t kAlignmentElems =
        kAlignmentBytes / static_cast<int64_t>(sizeof(c10::Half));
    auto storage = at::empty({src.numel() + kAlignmentElems}, src.options());
    const uint64_t storage_addr =
        RpuGetDevAddr(storage.data_ptr<c10::Half>());
    const int64_t byte_offset = static_cast<int64_t>(
        (kAlignmentBytes - (storage_addr & (kAlignmentBytes - 1)))
        & (kAlignmentBytes - 1));
    TORCH_CHECK(byte_offset % static_cast<int64_t>(sizeof(c10::Half)) == 0,
                name, " allocator returned an address incompatible with FP16 alignment");
    auto aligned = storage.narrow(
        0, byte_offset / static_cast<int64_t>(sizeof(c10::Half)), src.numel()
    ).view(src.sizes());
    aligned.copy_(src);
    TORCH_CHECK((RpuGetDevAddr(aligned.data_ptr<c10::Half>()) & 0xFF) == 0,
                name, " failed to create a 256-byte aligned device copy");
    return aligned;
}

}  // namespace

namespace v3 {

// QWEN35_TEXT_FIXED_KERNEL_BASIS: non-manifest launchers implement fixed GDN,
// normalization/activation, cache bookkeeping, or BufferDecl transport. Any
// selectable physical implementation must first become a typed manifest route.

Qwen3_5Model::Qwen3_5Model() = default;
Qwen3_5Model::~Qwen3_5Model() = default;

void Qwen3_5Model::set_execution_cores(int64_t cores) {
    TORCH_CHECK(cores == 4 || cores == 6 || cores == 8,
                "Qwen3.5 execution cores must be 4, 6 or 8");
    TORCH_CHECK(layer_weights_.empty() && !z2_bound_ && !action_mode_,
                "Qwen3.5 execution cores must be bound before weights");
    set_execution_core_count(static_cast<int>(cores));
}

std::vector<int64_t> Qwen3_5Model::execution_topology() const {
    TORCH_CHECK(num_layers() > 0,
                "Qwen3.5 topology requires installed model weights");
    return {num_cores(), attn_tp(), mlp_tp(), lm_head_tp(), NUM_CORES};
}

DecoderExecutionTopology Qwen3_5Model::resolve_model_execution_topology(
        int64_t nq, int64_t nkv, int64_t hd, int64_t h, int64_t intermediate) const {
    if (num_cores() == 8)
        return FusedModelBase::resolve_model_execution_topology(
            nq, nkv, hd, h, intermediate);
    return resolve_qwen35_reduced_execution_topology(
        num_cores(), nq, nkv, hd, h, intermediate);
}

std::vector<int64_t> Qwen3_5Model::cold_topology_arguments() const {
    return {1, num_cores(), attn_tp(), gdn_tp(), mlp_tp(), lm_head_tp(), NUM_CORES};
}

int64_t Qwen3_5Model::subclass_layout_hash() const {
    // Precision selects different generated arithmetic and specialized tiles.
    int64_t hash = detail::layout_mix(0, linear_acc32_);
    if (high_precision_) {
        hash = detail::layout_mix(hash, high_precision_site());
        for (const auto value : high_precision_arguments())
            hash = detail::layout_mix(hash, value);
    }
    if (action_nvfp4_.enabled()) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_NVFP4_LINEAR_SITE);
        for (int family = 0; family < 5; ++family) {
            for (const auto value : action_nvfp4_arguments(family))
                hash = detail::layout_mix(hash, value);
            std::vector<int64_t> owner;
            append_kvinsert_cost_tensor_identity(owner, action_nvfp4_.values[family]);
            for (const auto value : owner) hash = detail::layout_mix(hash, value);
        }
    }
    if (num_cores() != 8) {
        for (const auto value : cold_topology_arguments())
            hash = detail::layout_mix(hash, value);
    }
    hash = detail::layout_mix(hash, action_mod_body_fp16_);
    if (use_action_mod_body()) {
        hash = detail::layout_mix(hash, 49 * 6 * 1024 * DWIDTH);
    }
    // Bind optional arithmetic schedules without reading a previous forward's
    // RTC prefix. The runtime split is an input to the fixed M32 row selector.
    if (action_mode_ && action_rtc_norm_shift_available_) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_FUSED_NORM_SHIFT_SITE);
        hash = detail::layout_mix(hash, 3);
    }
    if (high_precision_ && action_mode_) {
        hash = detail::layout_mix(hash, action_newton_norm_shift_available_ ? 1 : 0);
        if (use_action_newton_norm_shift(action_len_)) {
            hash = detail::layout_mix(hash, QWEN35_ACTION_NEWTON_NORM_SHIFT_SITE);
            for (const auto value : action_newton_norm_shift_arguments())
                hash = detail::layout_mix(hash, value);
        }
    }
    if (action_mode_ && action_ring_rtc_residual_available_) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_RING_RTC_RESIDUAL_SITE);
        for (auto value : action_ring_rtc_residual_arguments())
            hash = detail::layout_mix(hash, value);
    }
    if (action_mode_ && action_rtc_sub_gated_residual_available_) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_RTC_RESIDUAL_GATE_SITE);
        hash = detail::layout_mix(hash, 4);
    }
    if (action_mode_ && action_sub_gated_residual_available_) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_RESIDUAL_GATE_SITE);
        hash = detail::layout_mix(hash, 3);
    }
    if (use_g05_gdn_local_slice(256)) {
        hash = detail::layout_mix(hash, QWEN35_TEXT_GDN_LOCAL_SLICE_SITE);
        for (const auto value : g05_gdn_local_slice_arguments(256))
            hash = detail::layout_mix(hash, value);
    }
    if (use_text_gateup_pair(256)) {
        hash = detail::layout_mix(hash, text_gateup_pair_site());
        hash = detail::layout_mix(hash, text_gateup_pair_w8_ ? 2 : 1);
    }
    if (!action_mode_ && text_silu_round_mul_available_) {
        hash=detail::layout_mix(hash,QWEN35_TEXT_SILU_ROUND_MUL_SITE);
        for (const auto value:text_silu_round_mul_arguments()) hash=detail::layout_mix(hash,value);
        for (const auto& lw:layer_weights_) {
            std::vector<int64_t> owners;
            for (const auto* owner:{&lw.gate_w,&lw.up_w,&lw.down_w,&lw.gate_ws,&lw.up_ws,&lw.down_ws})
                append_kvinsert_cost_tensor_identity(owners,*owner);
            for (const auto value:owners) hash=detail::layout_mix(hash,value);
        }
    }
    if (!action_mode_ && text_mlp_tail_available_) {
        hash = detail::layout_mix(hash,QWEN35_TEXT_MLP_TAIL_SITE);
        for (const auto value:text_mlp_tail_arguments()) hash=detail::layout_mix(hash,value);
        for (const auto& w:layer_weights_) {
            std::vector<int64_t> owners;
            for (const auto* t:{&w.input_norm_w,&w.post_norm_w,&w.gate_w,&w.up_w,&w.down_w,
                    &w.gate_ws,&w.up_ws,&w.down_ws,&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.attn_gate_w,
                    &w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.attn_gate_ws,&w.gdn_q_w,&w.gdn_k_w,
                    &w.gdn_v_w,&w.gdn_in_z_w,&w.gdn_out_w,&w.gdn_q_ws,&w.gdn_k_ws,
                    &w.gdn_v_ws,&w.gdn_in_z_ws,&w.gdn_out_ws,&w.gdn_b_bg_w,&w.gdn_a_bg_w,
                    &w.gdn_b_bg_ws,&w.gdn_a_bg_ws}) append_kvinsert_cost_tensor_identity(owners,*t);
            for (const auto value:owners) hash=detail::layout_mix(hash,value);
        }
    }
    if (!action_mode_ && text_owner_rms_available_) {
        hash = detail::layout_mix(hash, QWEN35_TEXT_OWNER_RMS_SITE);
        for (const auto value : text_owner_rms_arguments())
            hash = detail::layout_mix(hash, value);
        for (size_t layer = 3; layer < layer_weights_.size(); layer += 4) {
            std::vector<int64_t> owner;
            append_kvinsert_cost_tensor_identity(owner, layer_weights_[layer].post_norm_w);
            for (const auto value : owner) hash = detail::layout_mix(hash, value);
        }
    }
    if (!action_mode_ && text_gdn_owner_rms_available_) {
        hash = detail::layout_mix(hash, QWEN35_TEXT_GDN_OWNER_RMS_SITE);
        for (const auto value : text_gdn_owner_rms_arguments())
            hash = detail::layout_mix(hash, value);
        for (size_t layer = 0; layer < layer_weights_.size(); ++layer) {
            if (layer_is_full_[layer]) continue;
            std::vector<int64_t> owner;
            append_kvinsert_cost_tensor_identity(owner, layer_weights_[layer].post_norm_w);
            for (const auto value : owner) hash = detail::layout_mix(hash, value);
        }
    }
    if (use_action_gateup_resident(action_len_)) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_GATEUP_RESIDENT_SITE);
        hash = detail::layout_mix(hash, 2);
        hash = detail::layout_mix(hash, action_gateup_resident_layers());
        hash = detail::layout_mix(hash, QWEN35_ACTION_RESIDENT_TOTAL_BYTES);
        hash = detail::layout_mix(hash, action_gateup_resident_w8_);
    }
    if (use_action_nvfp4_o_resident(action_len_)) {
        hash = detail::layout_mix(hash, action_nvfp4_o_resident_site());
        for (const auto value : action_nvfp4_o_resident_arguments(0, -1))
            hash = detail::layout_mix(hash, value);
        for (const auto* owner : {&action_nvfp4_o_resident_.first, &action_nvfp4_o_resident_.second}) {
            std::vector<int64_t> identity;
            append_kvinsert_cost_tensor_identity(identity, *owner);
            for (const auto value : identity) hash = detail::layout_mix(hash, value);
        }
    }
    if (use_action_nvfp4_down_resident(action_len_,0)) {
        hash = detail::layout_mix(hash, action_nvfp4_down_resident_site());
        for (const auto value : action_nvfp4_down_resident_arguments(0, -1))
            hash = detail::layout_mix(hash, value);
        for (const auto* owner : {&action_nvfp4_down_resident_.first, &action_nvfp4_down_resident_.second}) {
            std::vector<int64_t> identity;
            append_kvinsert_cost_tensor_identity(identity, *owner);
            for (const auto value : identity) hash = detail::layout_mix(hash, value);
        }
    }
    if (use_action_o_resident(action_len_)) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_O_RESIDENT_W8_SITE);
        for (const auto value : action_o_resident_arguments(0, false))
            hash = detail::layout_mix(hash, value);
    }
    if (use_action_down_partial(action_len_,0)) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_DOWN_PARTIAL_W8_SITE);
        for (const auto value : action_down_partial_arguments(false))
            hash = detail::layout_mix(hash, value);
    }
    if (action_mode_ && action_tail_available_) {
        hash = detail::layout_mix(hash,QWEN35_ACTION_TAIL_SITE);
        for (const auto value : action_tail_arguments()) hash = detail::layout_mix(hash,value);
    }
    if (action_mode_ && action_mlp_tail_available_) {
        hash = detail::layout_mix(hash,QWEN35_ACTION_MLP_TAIL_SITE);
        for (const auto value : action_mlp_tail_arguments()) hash = detail::layout_mix(hash,value);
    }
    if (action_mode_ && action_attn_sigmoid_mul_available_) {
        hash = detail::layout_mix(hash,QWEN35_ACTION_ATTN_SIGMOID_MUL_SITE);
        for (const auto value : action_attn_sigmoid_mul_arguments())
            hash = detail::layout_mix(hash,value);
    }
    if (use_action_nvfp4_swiglu(action_len_)) {
        hash = detail::layout_mix(hash, action_nvfp4_swiglu_site());
        for (const auto value : action_nvfp4_swiglu_arguments())
            hash = detail::layout_mix(hash, value);
    } else if (use_action_gateup_pair(action_len_)) {
        hash = detail::layout_mix(hash, action_gateup_pair_site());
        hash = detail::layout_mix(hash, 1);
        if (action_gateup_pair_nvfp4_)
            for (const auto value : action_gateup_pair_arguments())
                hash = detail::layout_mix(hash, value);
    }
    if (use_action_silu_round_mul(action_len_) && !use_action_nvfp4_swiglu(action_len_)) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_SILU_ROUND_MUL_SITE);
        hash = detail::layout_mix(hash, 1);
    }
    if (high_precision_ && action_mode_) {
        hash = detail::layout_mix(hash, action_silu_high_mul_available_ ? 1 : 0);
        if (use_action_silu_high_mul(action_len_)) {
            hash = detail::layout_mix(hash, QWEN35_ACTION_SILU_HIGH_MUL_SITE);
            for (const auto value : action_silu_high_mul_arguments())
                hash = detail::layout_mix(hash, value);
        }
    }
    if (action_nvfp4_qk_norm_rope_available_ && use_action_nvfp4(action_len_)) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_QK_NORM_ROPE_GROUP4_SITE);
        for (const auto value : action_nvfp4_qk_norm_rope_arguments())
            hash = detail::layout_mix(hash, value);
    }
    if (action_gate_kv_owner_available_ && use_action_nvfp4(action_len_)) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_GATE_KV_OWNER_W8_SITE);
        for (const auto value : action_gate_kv_owner_arguments(0))
            hash = detail::layout_mix(hash, value);
    }
    if (action_owner_cache_joint_available_ && use_action_nvfp4(action_len_)) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_OWNER_CACHE_JOINT_SITE);
        for (const auto value : action_owner_cache_joint_abi())
            hash = detail::layout_mix(hash, value);
    }
    // This capability changes transport, not the declared full K/V roots.
    // Position is already part of the candidate descriptor; never hash stale
    // per-layer prefix state from the previous call into a dry query.
    if (action_owner_cache_available_ && (use_action_qkv_gate(action_len_) ||
            (use_action_nvfp4(action_len_) && use_action_kv_owner(action_len_)))) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_OWNER_CACHE_INSERT_SITE);
        for (const int64_t value : {3,161,449,16,1,0,245,181})
            hash = detail::layout_mix(hash,value);
        for (const auto value : action_k_owner_norm_rope_arguments())
            hash = detail::layout_mix(hash, value);
    }
    if (use_action_qkv_gate(action_len_)) {
        hash = detail::layout_mix(hash, QWEN35_ACTION_QKV_GATE_W8_SITE);
        for (const auto value : action_qkv_gate_arguments())
            hash = detail::layout_mix(hash, value);
    } else {
        if (use_action_q_gate(action_len_)) {
            hash = detail::layout_mix(hash, action_q_gate_w8_ ? QWEN35_ACTION_Q_GATE_W8_SITE : QWEN35_ACTION_Q_GATE_SITE);
            hash = detail::layout_mix(hash, 1);
        }
        if (use_action_kv_owner(action_len_)) {
            hash = detail::layout_mix(hash, action_kv_owner_w8_ ? QWEN35_ACTION_KV_OWNER_W8_SITE : QWEN35_ACTION_KV_OWNER_SITE);
            hash = detail::layout_mix(hash, 1);
        }
    }
    return hash;
}

int64_t Qwen3_5Model::text_ring_route(int64_t rows, int64_t cols) const {
    return fmb_ring_all_reduce_route_selector(rows, cols, num_cores());
}

bool Qwen3_5Model::use_g05_gdn_local_slice(int64_t rows) const {
    // Same BASE copy arithmetic, admitted only for the retained G05 Text
    // profile. Each physical chunk remains within the measured C64..C256
    // domain, including tails of longer logical prefixes.
    if (action_mode_ || z2_bound_ || get_debug_export() ||
        !fast_replay_enabled_ || !retained_prefill_graph_ || !has_gdn_ ||
        num_cores() != 8 || gdn_tp() != 8 || num_layers() != 24 ||
        hidden_size() != 2048 || intermediate_size() != 6144 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 ||
        gdn_nvh_ != 16 || gdn_dk_ != 128 || gdn_dv_ != 128 ||
        gdn_conv_dim_ != 6144 || gdn_kc_ != 4 ||
        layer_is_full_.size() != 24 || rows < 64 || rows > 256 || rows % 64)
        return false;
    for (size_t layer = 0; layer < layer_is_full_.size(); ++layer)
        if (bool(layer_is_full_[layer]) != ((layer + 1) % 4 == 0)) return false;
    return true;
}

std::vector<int64_t> Qwen3_5Model::g05_gdn_local_slice_arguments(int64_t rows) const {
    // Version, chunk rows, GDN64, per-core heads/padded columns, TP,
    // calls/layer, launches/call, copy width, hardware-local address ABI.
    return {1, rows, 64, 2, 16, 8, 2 + 2 * (rows / 64), 1, 2,
            0x40000000, 0x1000000};
}

void Qwen3_5Model::set_chunk_size_cap(int64_t cap) {
    TORCH_CHECK(cap == 0 || cap >= 64,
                "Qwen3.5 text chunk cap must be 0 or at least 64, got ", cap);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 text chunk cap must be set before the first forward");
    chunk_size_cap_ = cap;
    set_chunk_size_override(0);
    invalidate_model_state();
}

void Qwen3_5Model::set_prefill_chunk_size(int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0
                    || (chunk_size >= 64 && chunk_size % 64 == 0),
                "Qwen3.5 text chunk size must be 0 (auto) or a positive "
                "multiple of 64, got ", chunk_size);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 text chunk size must be set before the first forward; "
                "use the execution-reconfigure transaction for a live handle");
    set_chunk_size_override(chunk_size);
    invalidate_model_state();
}

void Qwen3_5Model::stage_prefill_execution_controls(
        uint64_t token, int64_t cap, int64_t chunk_size,
        int64_t max_kv_len, int64_t envelope_chunk) {
    TORCH_CHECK(cap == 0 || cap >= 64,
                "Qwen3.5 text chunk cap must be 0 or at least 64, got ", cap);
    TORCH_CHECK(chunk_size == 0 ||
                    (chunk_size >= 64 && chunk_size % 64 == 0),
                "Qwen3.5 text chunk size must be 0 (auto) or a positive "
                "multiple of 64, got ", chunk_size);
    const int64_t old_cap = chunk_size_cap_;
    stage_execution_controls(
        token,
        chunk_size,
        ChunkEnvelope{max_kv_len, envelope_chunk},
        [this, cap] { chunk_size_cap_ = cap; },
        [this, old_cap] { chunk_size_cap_ = old_cap; },
        "Qwen3_5Model::stage_prefill_execution_controls");
}

void Qwen3_5Model::set_linear_acc32(bool enabled) {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 linear accumulation mode must be set before the first forward");
    TORCH_CHECK(!action_mode_ || enabled == linear_acc32_,
                "Qwen3.5 action accumulation must be bound before enabling action mode");
    TORCH_CHECK(!action_nvfp4_.enabled() || enabled == linear_acc32_,
                "G05 NVFP4 accumulation must be bound before installing weights");
    linear_acc32_ = enabled;
    prepare_text_gateup_pair();
    prepare_text_owner_rms();
    invalidate_model_state();
}

void Qwen3_5Model::set_high_precision(bool enabled) {
    constexpr const char* operation = "qwen3_5_set_high_precision";
    RpuExecutionCoordinator::require_graph_quiescent(operation);
    RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
    TORCH_CHECK(!high_precision_bound_ && layer_weights_.empty() &&
                    !action_mode_ && !z2_bound_ &&
                    get_last_resolved_chunk_size() == 0,
                "G05 math precision must be bound once before weights and dispatch");
    // Reject an incomplete optional REF before publishing the cold policy.
    if (enabled) rpu_require_high_precision_math_kernels();
    high_precision_ = enabled;
    high_precision_bound_ = true;
    invalidate_model_state();
}

RpuUnaryPrecision Qwen3_5Model::unary_precision() const {
    return high_precision_ ? RpuUnaryPrecision::HIGH : RpuUnaryPrecision::BASE;
}

RpuRmsNormSpmRoute Qwen3_5Model::rmsnorm_route(int64_t rows, int64_t cols) const {
    return high_precision_ ? rpu_resolve_high_precision_rmsnorm_spm_route(rows, cols)
                           : RpuRmsNormSpmRoute::BASE;
}

int64_t Qwen3_5Model::high_precision_site() const {
    return action_mode_ ? QWEN35_ACTION_HIGH_PRECISION_SITE
                        : QWEN35_TEXT_HIGH_PRECISION_SITE;
}

std::vector<int64_t> Qwen3_5Model::high_precision_arguments() const {
    // Version, HIGH math/role, exact model geometry, GDN geometry, cores,
    // independent Linear accumulation and enabled operator families. FP16
    // storage/retirement and the original GELU formula remain unchanged.
    return {1, high_precision_ ? 1 : 0, action_mode_ ? 1 : 0, hidden_size(), intermediate_size(),
            num_layers(), num_q_heads(), num_kv_heads(), head_dim(),
            gdn_nvh_, gdn_dk_, gdn_dv_, num_cores(), linear_acc32_ ? 32 : 16,
            use_silu_ ? 1 : 0, has_qk_norm_ ? 1 : 0, has_gdn_ ? 1 : 0};
}

void Qwen3_5Model::consume_high_precision_math() {
    TORCH_CHECK(high_precision_ && ctx().has_complete_physical_manifest(),
                "G05 HIGH math requires its cold COMPLETE schedule");
    if (!action_mode_) {
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_TEXT_HIGH_PRECISION_SITE,
            /*selector=*/1, /*flags=*/0, high_precision_arguments());
    } else {
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_ACTION_HIGH_PRECISION_SITE,
            /*selector=*/1, /*flags=*/0, high_precision_arguments());
    }
}

void Qwen3_5Model::set_fast_replay(bool enabled) {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 fast replay must be set before the first forward");
    fast_replay_enabled_ = enabled;
    prepare_text_gateup_pair();
    prepare_text_owner_rms();
    invalidate_model_state();
}

void Qwen3_5Model::enable_action_mode() {
    TORCH_CHECK(num_cores() == 8, "Qwen3.5 action mode requires eight cores");
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 action mode must be enabled before the first forward");
    TORCH_CHECK(num_layers() > 0,
                "Qwen3.5 action mode requires set_weights first");
    TORCH_CHECK(!has_gdn_,
                "Qwen3.5 action mode supports all-full-attention experts only");
    action_mode_ = true;
    prepare_action_newton_norm_shift();
    // Bind the optional Pi05 payload once, before planning/capture. It retains
    // the baseline RMSNorm FP16 rounding before adding the FP16 shift.
    action_norm_shift_available_ = KernelCache::instance().has_loaded(
        KernelId::PI05_ADARMS_NORM_SHIFT_H1024) && !high_precision_;
    action_gated_residual_available_ = KernelCache::instance().has_loaded(
        KernelId::PI05_GATED_RESIDUAL_H1024);
    action_rtc_norm_shift_available_ = KernelCache::instance().has_loaded(
        KernelId::G05_RTC_ADARMS_NORM_SHIFT_M32H1024) && !high_precision_;
    action_ring_rtc_residual_available_ = KernelCache::instance().has_loaded(
        KernelId::G05_RING_RTC_SUB_GATED_RESIDUAL_M32H1024);
    action_rtc_sub_gated_residual_available_ = KernelCache::instance().has_loaded(
        KernelId::G05_RTC_SUB_GATED_RESIDUAL_M32H1024);
    action_sub_gated_residual_available_ = KernelCache::instance().has_loaded(
        KernelId::G05_SUB_GATED_RESIDUAL_H1024);
    action_prefix_lens_.assign(num_layers(), 0);
    set_chunk_size_override(0);
    invalidate_model_state();
}

void Qwen3_5Model::set_action_io_weights(
    const at::Tensor& input_w, const at::Tensor& input_b,
    const at::Tensor& output_w, const at::Tensor& output_b,
    int64_t action_dim, int64_t action_dim_pad, int64_t action_len) {
    TORCH_CHECK(action_mode_,
                "Qwen3.5 action I/O weights require action mode");
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 action I/O weights must be set before the first forward");
    TORCH_CHECK(action_dim > 0 && action_dim_pad >= action_dim
                && action_dim_pad % 16 == 0,
                "Qwen3.5 action dimensions must satisfy 0 < dim <= padded dim "
                "and padded dim must be a multiple of 16");
    TORCH_CHECK(!action_nvfp4_.enabled() || (action_len == 32 && action_dim == 27 && action_dim_pad == 32),
                "G05 NVFP4 Action requires horizon32 and 27/32 I/O geometry");
    TORCH_CHECK(action_len > 0 && action_len % 16 == 0,
                "Qwen3.5 action length must be a positive multiple of 16");
    auto check_rpu_half = [](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.defined()
                    && tensor.device().type() == at::kPrivateUse1
                    && tensor.scalar_type() == at::kHalf
                    && tensor.is_contiguous(),
                    name, " must be contiguous FP16 on device");
    };
    check_rpu_half(input_w, "Qwen3.5 action input weight");
    check_rpu_half(input_b, "Qwen3.5 action input bias");
    check_rpu_half(output_w, "Qwen3.5 action output weight");
    check_rpu_half(output_b, "Qwen3.5 action output bias");
    TORCH_CHECK(input_w.numel() == hidden_size() * action_dim_pad
                && input_b.numel() == hidden_size(),
                "Qwen3.5 action input projection shape mismatch");
    TORCH_CHECK(output_w.numel() == action_dim_pad * hidden_size()
                && output_b.numel() == action_dim_pad,
                "Qwen3.5 action output projection shape mismatch");

    action_input_w_ = input_w;
    action_input_b_ = input_b;
    action_output_w_ = output_w;
    action_output_b_ = output_b;
    action_dim_ = action_dim;
    action_dim_pad_ = action_dim_pad;
    action_len_ = action_len;
    action_loop_active_ = false;
    action_rtc_active_ = false;
    action_rtc_prefix_len_ = 0;
    action_rtc_prefix_ref_ = at::Tensor();
    action_rtc_prefix_live_base_ = 0;
    action_trace_active_ = false;
    action_trace_pre_action_ref_ = at::Tensor();
    action_trace_action_embed_ref_ = at::Tensor();
    action_trace_final_hidden_ref_ = at::Tensor();
    action_trace_velocity_ref_ = at::Tensor();
    action_trace_post_action_ref_ = at::Tensor();
    action_trace_pre_action_live_base_ = 0;
    action_trace_action_embed_live_base_ = 0;
    action_trace_final_hidden_live_base_ = 0;
    action_trace_velocity_live_base_ = 0;
    action_trace_post_action_live_base_ = 0;
    action_num_steps_ = 1;
    action_euler_scale_pinned_ = false;
    auto options = input_w.options();
    action_hidden_stage_ = at::empty(
        {1, action_len_, hidden_size()}, options);
    action_velocity_stage_ = at::empty(
        {1, action_len_, action_dim_pad_}, options);
    // Prove the DDR Gate/Up replacement before choosing the shared six-MiB
    // residency budget. No discarded Gate/Up owner needs to be packed first.
    prepare_action_gateup_pair();
    prepare_action_silu_round_mul();
    prepare_action_silu_high_mul();
    prepare_action_nvfp4_swiglu();
    prepare_action_attn_sigmoid_mul();
    prepare_action_o_resident();
    prepare_action_nvfp4_o_resident();
    prepare_action_nvfp4_down_resident();
    prepare_action_gateup_resident();
    prepare_action_down_partial();
    prepare_action_kv_owner();
    prepare_action_q_gate();
    prepare_action_qkv_gate();
    prepare_action_owner_cache();
    prepare_action_gate_kv_owner();
    prepare_action_nvfp4_qk_norm_rope();
    prepare_action_owner_cache_joint();
    prepare_action_tail();
    prepare_action_mlp_tail();
    invalidate_model_state();
}

int64_t Qwen3_5Model::resolve_prefill_chunk_size(int64_t execution_len) {
    TORCH_CHECK(execution_len >= 64 && execution_len % 64 == 0,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 prefill execution "
                "length must be a positive multiple of 64, got ",
                execution_len);
    return resolve_chunk_size_for_shape(
        execution_len, /*position=*/0, std::nullopt, /*is_causal=*/true);
}

std::vector<int64_t> Qwen3_5Model::resolve_prefill_stage_domain(
    int64_t execution_len, int64_t logical_len,
    int64_t planning_chunk_size_override) {
    TORCH_CHECK(planning_chunk_size_override <= 0 ||
                    planning_chunk_size_override % 64 == 0,
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Qwen3.5 prefill "
                "planning chunk must be a multiple of 64");
    TORCH_CHECK(execution_len >= 64 && execution_len % 64 == 0,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 prefill execution "
                "length must be a positive "
                "multiple of 64, got ", execution_len);
    TORCH_CHECK(
        logical_len > 1 && logical_len <= execution_len,
        "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 prefill logical length "
        "must be in [2, execution length], got ", logical_len,
        " for execution length ", execution_len);
    return encode_fmb_prefill_stage_domain(
        resolve_prefill_stage_domain_for_shape(
            execution_len, /*position=*/0, std::nullopt,
            /*is_causal=*/true, /*requested_chunk_size=*/0, logical_len,
            planning_chunk_size_override));
}

std::vector<int64_t> Qwen3_5Model::resolve_decode_stage_descriptor() {
    auto candidates = resolve_prefill_stage_domain_for_shape(
        /*seq_len=*/1, /*position=*/0, std::nullopt,
        /*is_causal=*/true);
    TORCH_CHECK(
        candidates.size() == 1,
        "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 fixed decode must resolve "
        "exactly one native stage candidate, got ", candidates.size());
    return encode_fmb_prefill_stage_candidate(candidates.front());
}

std::vector<int64_t> Qwen3_5Model::resolve_action_stage_domain(
    int64_t requested_action_len, int64_t max_prefix_len,
    int64_t requested_chunk_size, bool include_action_io,
    int64_t action_route_flags, int64_t graph_lifecycle, int64_t num_steps) {
    TORCH_CHECK(action_mode_ && action_len_ > 0,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 action planner "
                "requires an installed action-mode handle");
    TORCH_CHECK(requested_action_len == action_len_,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 action planner "
                "requires the configured horizon ", action_len_, ", got ",
                requested_action_len);
    TORCH_CHECK(max_prefix_len >= 0,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 action prefix "
                "length must be non-negative, got ", max_prefix_len);
    TORCH_CHECK(requested_chunk_size == 0 ||
                    requested_chunk_size == action_len_,
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Qwen3.5 action chunk "
                "must be AUTO or exact ", action_len_, ", got ",
                requested_chunk_size);
    TORCH_CHECK(
        action_route_flags >= 0 &&
            (action_route_flags & ~QWEN35_ACTION_ROUTE_MASK) == 0 &&
            include_action_io ==
                ((action_route_flags & QWEN35_ACTION_ROUTE_IO) != 0) &&
            ((action_route_flags & QWEN35_ACTION_ROUTE_RTC) == 0 ||
             include_action_io) &&
            ((action_route_flags & QWEN35_ACTION_ROUTE_TRACE) == 0 ||
             (action_route_flags & QWEN35_ACTION_ROUTE_RTC) != 0),
        "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 action route flags do not "
        "describe a canonical base/IO/RTC/trace variant");
    const auto lifecycle = static_cast<FmbGraphLifecycle>(graph_lifecycle);
    TORCH_CHECK(
        lifecycle == FmbGraphLifecycle::BOUNDED_ONESHOT ||
            lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD,
        "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 action lifecycle must be "
        "BOUNDED_ONESHOT or COMPOSITE_CHILD, got ", graph_lifecycle);
    TORCH_CHECK(
        (num_steps == 1 || num_steps == 10) &&
            (include_action_io || num_steps == 1) &&
            ((action_route_flags & QWEN35_ACTION_ROUTE_RTC) == 0 || num_steps == 10) &&
            (lifecycle != FmbGraphLifecycle::BOUNDED_ONESHOT || num_steps == 1),
        "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 action schedule must be a "
        "single hidden/IO step or a retained 10-step IO/RTC loop");
    const bool saved_include_io = action_descriptor_include_io_;
    const int64_t saved_route_flags = action_descriptor_route_flags_;
    const FmbGraphLifecycle saved_lifecycle =
        action_descriptor_graph_lifecycle_;
    const bool saved_step = action_step_active_;
    const bool saved_loop = action_loop_active_;
    const bool saved_rtc = action_rtc_active_;
    const bool saved_trace = action_trace_active_;
    const int64_t saved_steps = action_num_steps_;
    action_descriptor_include_io_ = include_action_io;
    action_descriptor_route_flags_ = action_route_flags;
    action_descriptor_graph_lifecycle_ = lifecycle;
    // Query the requested first-forward schedule, never the previous call's
    // hidden/step/unroll state. These fields also affect temporary SPM layout.
    action_step_active_ = include_action_io;
    action_loop_active_ = num_steps > 1;
    action_num_steps_ = num_steps;
    action_rtc_active_ = (action_route_flags & QWEN35_ACTION_ROUTE_RTC) != 0;
    action_trace_active_ = (action_route_flags & QWEN35_ACTION_ROUTE_TRACE) != 0;
    auto restore = c10::make_scope_exit(
        [&] {
            action_descriptor_include_io_ = saved_include_io;
            action_descriptor_route_flags_ = saved_route_flags;
            action_descriptor_graph_lifecycle_ = saved_lifecycle;
            action_step_active_ = saved_step;
            action_loop_active_ = saved_loop;
            action_num_steps_ = saved_steps;
            action_rtc_active_ = saved_rtc;
            action_trace_active_ = saved_trace;
        });
    return encode_fmb_prefill_stage_domain(
        resolve_prefill_stage_domain_for_shape(
            action_len_, max_prefix_len, std::nullopt,
            /*is_causal=*/false, requested_chunk_size,
            /*logical_len=*/action_len_));
}

void Qwen3_5Model::set_action_chunk_size(int64_t chunk_size) {
    TORCH_CHECK(action_mode_ && action_len_ > 0,
                "Qwen3.5 action chunk control requires action mode and I/O "
                "weights");
    TORCH_CHECK(chunk_size == 0 || chunk_size == action_len_,
                "Qwen3.5 action chunk size must be AUTO (0) or exact ",
                action_len_, ", got ", chunk_size);
    set_control_chunk_size_override(
        chunk_size, "Qwen3_5Model::set_action_chunk_size");
}

void Qwen3_5Model::stage_action_chunk_size(
        uint64_t token, int64_t chunk_size) {
    TORCH_CHECK(action_mode_ && action_len_ > 0,
                "Qwen3.5 action chunk control requires action mode and I/O "
                "weights");
    TORCH_CHECK(chunk_size == 0 || chunk_size == action_len_,
                "Qwen3.5 action chunk size must be AUTO (0) or exact ",
                action_len_, ", got ", chunk_size);
    stage_control_chunk_size_override(
        token, chunk_size, "Qwen3_5Model::stage_action_chunk_size");
}

void Qwen3_5Model::enable_execution_reconfigure() {
    enable_execution_reconfigure_guard();
}

void Qwen3_5Model::set_retained_prefill_graph(bool enabled) {
    TORCH_CHECK(
        get_last_resolved_chunk_size() == 0,
        "Qwen3.5 prefill Graph lifecycle must be fixed before the first "
        "forward");
    retained_prefill_graph_ = enabled;
    invalidate_model_state();
}

SpmPipelineComponentLayout Qwen3_5Model::prepare_z2_layout(
    int64_t execution_len,
    int64_t real_len) {
    TORCH_CHECK(num_cores() == 8, "Qwen3.5 Z2 canary requires eight cores");
    constexpr int64_t kCanaryExecutionLen = 128;
    constexpr int64_t kCanaryRealLen = 72;
    constexpr int64_t kCanaryHidden = 2048;
    TORCH_CHECK(!z2_bound_,
                "Qwen3.5 text Z2: cannot prepare while a lease is active");
    TORCH_CHECK(execution_len == kCanaryExecutionLen &&
                    real_len == kCanaryRealLen,
                "Qwen3.5 text Z2 canary accepts only execution/real=(128,72), got (",
                execution_len, ",", real_len, ")");
    TORCH_CHECK(hidden_size() == kCanaryHidden,
                "Qwen3.5 text Z2 canary accepts only the 2B hidden size 2048, got ",
                hidden_size());
    TORCH_CHECK(valid_prefill_len_ == real_len,
                "Qwen3.5 text Z2: valid prefill length must be set to ", real_len,
                " before prepare; got ", valid_prefill_len_);
    TORCH_CHECK(has_mrope_ && prefill_cos_.defined() && prefill_sin_.defined() &&
                    prefill_cos_.dim() == 2 &&
                    prefill_cos_.sizes() == prefill_sin_.sizes() &&
                    prefill_cos_.size(0) == execution_len &&
                    prefill_cos_.size(1) == rotary_dim_ / 2,
                "Qwen3.5 text Z2: stable prefill M-RoPE tables must be [128,",
                rotary_dim_ / 2, "] before prepare");

    const int64_t resolved = resolve_prefill_chunk_size(execution_len);
    LayoutContext layout;
    layout.chunk_size = resolved;
    layout.max_kv_seq_len = execution_len;
    layout.num_layers = num_layers();
    layout.use_attn_mask = false;
    layout.is_causal = true;
    auto prepared = prepare_spm_pipeline_component(
        layout, compose_fmb_default_three_stage_chunk_plan(
                    layout, execution_len, /*position=*/0,
                    ChunkMode::SEQUENTIAL));

    z2_prepared_execution_len_ = execution_len;
    z2_prepared_real_len_ = real_len;
    z2_prefill_cos_addr_ =
        ::rhino_lkn::RpuGetDevAddr(prefill_cos_.data_ptr<c10::Half>());
    z2_prefill_sin_addr_ =
        ::rhino_lkn::RpuGetDevAddr(prefill_sin_.data_ptr<c10::Half>());
    return prepared;
}

SpmDense2DSpec Qwen3_5Model::z2_storage_spec(int64_t execution_len) const {
    TORCH_CHECK(execution_len == 128,
                "Qwen3.5 text Z2 canary storage requires 128 rows, got ",
                execution_len);
    TORCH_CHECK(hidden_size() == 2048,
                "Qwen3.5 text Z2 canary storage requires hidden size 2048, got ",
                hidden_size());
    SpmDense2DSpec spec;
    spec.rows = execution_len;
    spec.cols = hidden_size();
    spec.validate();
    return spec;
}

void Qwen3_5Model::validate_z2_prefill_contract() const {
    TORCH_CHECK(num_cores() == 8, "Qwen3.5 Z2 canary requires eight cores");
    TORCH_CHECK(z2_prepared_execution_len_ == 128 &&
                    z2_prepared_real_len_ == 72,
                "Qwen3.5 text Z2: fixed prefill layout was not prepared");
    TORCH_CHECK(valid_prefill_len_ == z2_prepared_real_len_,
                "Qwen3.5 text Z2: valid prefill length drifted after prepare");
    TORCH_CHECK(prefill_cos_.defined() && prefill_sin_.defined() &&
                    prefill_cos_.dim() == 2 &&
                    prefill_cos_.sizes() == prefill_sin_.sizes() &&
                    prefill_cos_.size(0) == z2_prepared_execution_len_ &&
                    prefill_cos_.size(1) == rotary_dim_ / 2,
                "Qwen3.5 text Z2: prefill M-RoPE shape drifted after prepare");
    TORCH_CHECK(
        ::rhino_lkn::RpuGetDevAddr(prefill_cos_.data_ptr<c10::Half>()) ==
                z2_prefill_cos_addr_ &&
            ::rhino_lkn::RpuGetDevAddr(prefill_sin_.data_ptr<c10::Half>()) ==
                z2_prefill_sin_addr_,
        "Qwen3.5 text Z2: prefill M-RoPE address drifted after prepare");
}

void Qwen3_5Model::adopt_z2_layout(
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    validate_z2_prefill_contract();
    TORCH_CHECK(!z2_bound_,
                "Qwen3.5 text Z2: binding is already active");
    adopt_spm_pipeline_component(lease, scratch);
}

void Qwen3_5Model::bind_z2_port(
    const SpmPipelineLease& lease,
    const SpmPortView& port) {
    validate_z2_prefill_contract();
    TORCH_CHECK(!z2_bound_,
                "Qwen3.5 text Z2: binding is already active");
    const auto expected = z2_storage_spec(z2_prepared_execution_len_);
    TORCH_CHECK(port.spec() == expected,
                "Qwen3.5 text Z2: full storage port spec mismatch");
    z2_input_port_addr_ = port.resolve_physical_addr(/*core=*/0, lease);
    z2_epoch_ = lease.epoch();
    z2_plan_hash_ = lease.plan_hash();
    z2_bound_ = true;
}

void Qwen3_5Model::validate_z2_layout(
    const SpmPipelineLease& lease) const {
    validate_z2_prefill_contract();
    TORCH_CHECK(z2_bound_, "Qwen3.5 text Z2: no active binding");
    TORCH_CHECK(z2_epoch_ == lease.epoch() &&
                    z2_plan_hash_ == lease.plan_hash(),
                "Qwen3.5 text Z2: stale lease binding");
    validate_spm_pipeline_component(lease);
}

void Qwen3_5Model::clear_z2_layout(uint64_t epoch, uint64_t plan_hash) {
    if (z2_bound_) {
        TORCH_CHECK(z2_epoch_ == epoch && z2_plan_hash_ == plan_hash,
                    "Qwen3.5 text Z2: stale clear token");
    }
    release_spm_pipeline_component(epoch, plan_hash);
    z2_input_port_addr_ = 0;
    z2_epoch_ = 0;
    z2_plan_hash_ = 0;
    z2_bound_ = false;
}

at::Tensor Qwen3_5Model::forward_action(
    const at::Tensor& hidden_states,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens,
    at::IntArrayRef planned_stage_descriptor) {
    TORCH_CHECK(action_mode_,
                "Qwen3.5 action forward requires enable_action_mode()");
    TORCH_CHECK(hidden_states.dim() == 3 && hidden_states.size(0) == 1,
                "Qwen3.5 action hidden_states must be [1, action_len, hidden]");
    const int64_t seq_len = hidden_states.size(1);
    TORCH_CHECK(seq_len > 0 && seq_len % 16 == 0,
                "Qwen3.5 action_len must be a positive multiple of 16, got ",
                seq_len);
    TORCH_CHECK(adaptive_mod.defined() && adaptive_mod.dim() == 2
                && adaptive_mod.size(0) == 2 * num_layers() + 1
                && adaptive_mod.size(1) == 3 * hidden_size(),
                "Qwen3.5 action adaptive_mod must be [",
                2 * num_layers() + 1, ", ", 3 * hidden_size(), "], got ",
                adaptive_mod.sizes());
    TORCH_CHECK(adaptive_mod.device().type() == at::kPrivateUse1
                && adaptive_mod.scalar_type() == at::kHalf
                && adaptive_mod.is_contiguous(),
                "Qwen3.5 action adaptive_mod must be contiguous FP16 on device");
    TORCH_CHECK(static_cast<int64_t>(prefix_lens.size()) == num_layers(),
                "Qwen3.5 action prefix_lens must have ", num_layers(),
                " entries, got ", prefix_lens.size());
    TORCH_CHECK(prefill_cos_.defined() && prefill_sin_.defined()
                && prefill_cos_.size(0) >= seq_len,
                "Qwen3.5 action requires set_prefill_rope with at least ",
                seq_len, " rows before forward");

    action_prefix_lens_.assign(prefix_lens.begin(), prefix_lens.end());
    int64_t max_prefix = 0;
    for (int64_t i = 0; i < num_layers(); ++i) {
        const int64_t prefix = action_prefix_lens_[i];
        TORCH_CHECK(prefix >= 0,
                    "Qwen3.5 action prefix_lens[", i, "] must be non-negative");
        TORCH_CHECK(i < static_cast<int64_t>(k_caches.size())
                    && k_caches[i].defined() && k_caches[i].dim() == 7,
                    "Qwen3.5 action K cache[", i, "] must be a defined 7D tensor");
        const int64_t capacity = k_caches[i].size(1) * 16;
        TORCH_CHECK(prefix + seq_len <= capacity,
                    "Qwen3.5 action cache capacity exceeded at layer ", i,
                    ": prefix=", prefix, " action_len=", seq_len,
                    " capacity=", capacity);
        max_prefix = std::max(max_prefix, prefix);
    }

    adaptive_mod_ref_ = adaptive_mod;
    adaptive_mod_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(adaptive_mod.data_ptr<c10::Half>());
    rpu_ddr_flush_force(adaptive_mod.data_ptr<c10::Half>());
    if (action_step_active_ || action_loop_active_ || action_num_steps_ != 1 ||
            action_rtc_active_ || action_trace_active_) {
        action_step_active_ = false;
        action_loop_active_ = false;
        action_num_steps_ = 1;
        action_trace_active_ = false;
        action_rtc_active_ = false;
        action_rtc_prefix_len_ = 0;
        action_rtc_prefix_ref_ = at::Tensor();
        action_rtc_prefix_live_base_ = 0;
        invalidate_model_state(/*planning_domain_changed=*/false);
    }
    action_step_active_ = false;
    fast_replay_active_ = fast_replay_enabled_;

    TORCH_CHECK(!planned_stage_descriptor.empty(),
                "Qwen3.5 action forward requires one complete planner "
                "stage descriptor");
    return run_all_layers(
        hidden_states, k_caches, v_caches, std::nullopt,
        max_prefix, /*is_causal=*/false,
        /*planned_chunk_size=*/0, planned_stage_descriptor);
}

at::Tensor Qwen3_5Model::forward_action_step(
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor& action_out,
    double delta_t,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens,
    int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor) {
    return forward_action_step_impl(
        action, action_keep_mask, action_out,
        /*action_prefix=*/nullptr, /*action_prefix_len=*/0,
        /*trace=*/nullptr,
        delta_t, adaptive_mod, k_caches, v_caches, prefix_lens, num_steps,
        planned_stage_descriptor);
}

at::Tensor Qwen3_5Model::forward_action_rtc_step(
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor& action_out,
    const at::Tensor& action_prefix,
    int64_t action_prefix_len,
    double delta_t,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens,
    int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor) {
    return forward_action_step_impl(
        action, action_keep_mask, action_out,
        &action_prefix, action_prefix_len,
        /*trace=*/nullptr,
        delta_t, adaptive_mod, k_caches, v_caches, prefix_lens, num_steps,
        planned_stage_descriptor);
}

at::Tensor Qwen3_5Model::forward_action_rtc_trace(
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor& action_out,
    const at::Tensor& action_prefix,
    int64_t action_prefix_len,
    double delta_t,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens,
    at::Tensor& trace_pre_action,
    at::Tensor& trace_action_embed,
    at::Tensor& trace_final_hidden,
    at::Tensor& trace_velocity,
    at::Tensor& trace_post_action,
    at::IntArrayRef planned_stage_descriptor) {
    ActionRtcTraceOutputs trace{
        trace_pre_action,
        trace_action_embed,
        trace_final_hidden,
        trace_velocity,
        trace_post_action,
    };
    return forward_action_step_impl(
        action, action_keep_mask, action_out,
        &action_prefix, action_prefix_len, &trace,
        delta_t, adaptive_mod, k_caches, v_caches, prefix_lens,
        /*num_steps=*/10, planned_stage_descriptor);
}

at::Tensor Qwen3_5Model::forward_action_step_impl(
    const at::Tensor& action,
    const at::Tensor& action_keep_mask,
    at::Tensor& action_out,
    const at::Tensor* action_prefix,
    int64_t action_prefix_len,
    const ActionRtcTraceOutputs* trace,
    double delta_t,
    const at::Tensor& adaptive_mod,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    at::IntArrayRef prefix_lens,
    int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor) {
    TORCH_CHECK(action_mode_ && action_input_w_.defined(),
                "Qwen3.5 action step requires action mode and I/O weights");
    TORCH_CHECK(num_steps == 1 || num_steps == 10,
                "Qwen3.5 action supports one diagnostic step or the canonical "
                "10-step Euler loop, got ", num_steps);
    const bool loop_active = num_steps > 1;
    const bool rtc_active = action_prefix != nullptr;
    const bool trace_active = trace != nullptr;
    TORCH_CHECK(!rtc_active || loop_active,
                "Qwen3.5 RTC action requires the canonical unrolled loop");
    TORCH_CHECK(!trace_active || rtc_active,
                "Qwen3.5 action trace is available only for the RTC loop");
    const bool cpu_action = action.device().is_cpu();
    const bool cpu_prefix = rtc_active && action_prefix->device().is_cpu();
    const bool private_cpu_loop = loop_active && !trace_active
        && !planned_stage_descriptor.empty()
        && action_len_ == 32 && action_dim_pad_ == 32;
    TORCH_CHECK(!cpu_action || (private_cpu_loop
                && action.sizes() == at::IntArrayRef({1, 32, 32})
                && action.scalar_type() == at::kHalf && action.is_contiguous()
                && !action.requires_grad() && !action.is_neg() && !action.is_conj()),
                "Qwen3.5 CPU action requires plain contiguous FP16 [1,32,32] "
                "for the canonical non-trace ten-step loop");
    TORCH_CHECK(!cpu_prefix || (private_cpu_loop
                && action_prefix_len > 0 && action_prefix_len < action_len_
                && action_prefix->sizes() == at::IntArrayRef({1, action_prefix_len, 32})
                && action_prefix->scalar_type() == at::kHalf
                && action_prefix->is_contiguous() && !action_prefix->requires_grad()
                && !action_prefix->is_neg() && !action_prefix->is_conj()),
                "Qwen3.5 CPU RTC prefix requires plain contiguous FP16 "
                "[1,prefix_len,32] for the canonical non-trace ten-step loop");
    if (!action_step_active_ || action_loop_active_ != loop_active || action_num_steps_ != num_steps
            || action_rtc_active_ != rtc_active
            || action_rtc_prefix_len_ != action_prefix_len
            || action_trace_active_ != trace_active) {
        action_loop_active_ = loop_active;
        action_num_steps_ = num_steps;
        action_rtc_active_ = rtc_active;
        action_rtc_prefix_len_ = action_prefix_len;
        action_trace_active_ = trace_active;
        invalidate_model_state(/*planning_domain_changed=*/false);
    }
    TORCH_CHECK(action.dim() == 3 && action.size(0) == 1
                && action.size(1) == action_len_
                && action.size(2) == action_dim_pad_
                && (action.device().type() == at::kPrivateUse1 || cpu_action)
                && action.scalar_type() == at::kHalf
                && action.is_contiguous(),
                "Qwen3.5 action step input must be [1, ", action_len_, ", ",
                action_dim_pad_, "] contiguous FP16 on CPU or device");
    TORCH_CHECK(action_keep_mask.dim() == 1
                && action_keep_mask.size(0) == action_dim_pad_
                && action_keep_mask.device().type() == at::kPrivateUse1
                && action_keep_mask.scalar_type() == at::kHalf
                && action_keep_mask.is_contiguous(),
                "Qwen3.5 action keep mask must be [", action_dim_pad_,
                "] contiguous FP16 on device");
    TORCH_CHECK(action_out.sizes() == action.sizes()
                && action_out.device().type() == at::kPrivateUse1
                && action_out.scalar_type() == at::kHalf
                && action_out.is_contiguous(),
                "Qwen3.5 action step output must match the input shape and be "
                "contiguous FP16 on device");
    TORCH_CHECK(action_out.data_ptr() != action.data_ptr(),
                "Qwen3.5 action step requires distinct input/output buffers");
    if (rtc_active) {
        TORCH_CHECK(action_prefix_len > 0 && action_prefix_len < action_len_,
                    "Qwen3.5 RTC action prefix length must be in [1, ",
                    action_len_ - 1, "], got ", action_prefix_len);
        TORCH_CHECK(action_prefix->dim() == 3
                    && action_prefix->size(0) == 1
                    && action_prefix->size(1) == action_prefix_len
                    && action_prefix->size(2) == action_dim_pad_
                    && (action_prefix->device().type() == at::kPrivateUse1 || cpu_prefix)
                    && action_prefix->scalar_type() == at::kHalf
                    && action_prefix->is_contiguous(),
                    "Qwen3.5 RTC action prefix must be [1, ",
                    action_prefix_len, ", ", action_dim_pad_,
                    "] contiguous FP16 on CPU or device");
        TORCH_CHECK(action_prefix->data_ptr() != action.data_ptr()
                    && action_prefix->data_ptr() != action_out.data_ptr(),
                    "Qwen3.5 RTC action prefix must use a distinct buffer");
    } else {
        TORCH_CHECK(action_prefix_len == 0,
                    "Qwen3.5 non-RTC action cannot carry a prefix length");
    }
    if (trace_active) {
        auto check_trace = [this, num_steps](
                               const at::Tensor& tensor,
                               const char* name,
                               int64_t width) {
            TORCH_CHECK(tensor.dim() == 4
                        && tensor.size(0) == num_steps
                        && tensor.size(1) == 1
                        && tensor.size(2) == action_len_
                        && tensor.size(3) == width
                        && tensor.device().type() == at::kPrivateUse1
                        && tensor.scalar_type() == at::kHalf
                        && tensor.is_contiguous(),
                        name, " must be [", num_steps, ", 1, ",
                        action_len_, ", ", width,
                        "] contiguous FP16 on device");
        };
        check_trace(trace->pre_action,
                    "Qwen3.5 RTC pre-action trace", action_dim_pad_);
        check_trace(trace->action_embed,
                    "Qwen3.5 RTC action-embed trace", hidden_size());
        check_trace(trace->final_hidden,
                    "Qwen3.5 RTC final-hidden trace", hidden_size());
        check_trace(trace->velocity,
                    "Qwen3.5 RTC velocity trace", action_dim_pad_);
        check_trace(trace->post_action,
                    "Qwen3.5 RTC post-action trace", action_dim_pad_);
        const std::array<const void*, 9> distinct_ptrs = {
            action.data_ptr(),
            action_keep_mask.data_ptr(),
            action_out.data_ptr(),
            action_prefix->data_ptr(),
            trace->pre_action.data_ptr(),
            trace->action_embed.data_ptr(),
            trace->final_hidden.data_ptr(),
            trace->velocity.data_ptr(),
            trace->post_action.data_ptr(),
        };
        for (size_t i = 0; i < distinct_ptrs.size(); ++i) {
            for (size_t j = i + 1; j < distinct_ptrs.size(); ++j) {
                TORCH_CHECK(
                    distinct_ptrs[i] != distinct_ptrs[j],
                    "Qwen3.5 RTC trace destinations and action inputs must "
                    "use distinct buffers");
            }
        }
    }
    TORCH_CHECK(delta_t > 0.0,
                "Qwen3.5 action Euler delta_t must be positive");
    const c10::Half euler_scale =
        c10::Half(-static_cast<float>(delta_t));
    if (!action_euler_scale_pinned_) {
        action_euler_scale_ = euler_scale;
        action_euler_scale_pinned_ = true;
    } else {
        TORCH_CHECK(action_euler_scale_ == euler_scale,
                    "Qwen3.5 action Euler delta_t changed after graph build");
    }
    const int64_t modulation_rows = 2 * num_layers() + 1;
    const int64_t modulation_width = 3 * hidden_size();
    const bool modulation_shape_ok = adaptive_mod.defined() &&
        ((!loop_active && !rtc_active && adaptive_mod.dim() == 2
          && adaptive_mod.size(0) == modulation_rows
          && adaptive_mod.size(1) == modulation_width)
         || (loop_active && !rtc_active && adaptive_mod.dim() == 3
             && adaptive_mod.size(0) == num_steps
             && adaptive_mod.size(1) == modulation_rows
             && adaptive_mod.size(2) == modulation_width)
         || (loop_active && rtc_active && adaptive_mod.dim() == 4
             && adaptive_mod.size(0) == num_steps
             && adaptive_mod.size(1) == modulation_rows
             && adaptive_mod.size(2) == 2
             && adaptive_mod.size(3) == modulation_width));
    TORCH_CHECK(modulation_shape_ok
                && adaptive_mod.device().type() == at::kPrivateUse1
                && adaptive_mod.scalar_type() == at::kHalf
                && adaptive_mod.is_contiguous(),
                "Qwen3.5 action modulation must be [", modulation_rows,
                ", ", modulation_width, "] for one step, [", num_steps,
                ", ", modulation_rows, ", ", modulation_width,
                "] for an unrolled loop, or [", num_steps, ", ",
                modulation_rows, ", 2, ", modulation_width,
                "] for RTC");
    TORCH_CHECK(static_cast<int64_t>(prefix_lens.size()) == num_layers(),
                "Qwen3.5 action step prefix_lens must have ", num_layers(),
                " entries, got ", prefix_lens.size());
    TORCH_CHECK(prefill_cos_.defined() && prefill_sin_.defined()
                && prefill_cos_.size(0) >= action_len_,
                "Qwen3.5 action step requires prefill RoPE with at least ",
                action_len_, " rows");

    action_prefix_lens_.assign(prefix_lens.begin(), prefix_lens.end());
    int64_t max_prefix = 0;
    for (int64_t i = 0; i < num_layers(); ++i) {
        const int64_t prefix = action_prefix_lens_[i];
        TORCH_CHECK(prefix >= 0,
                    "Qwen3.5 action step prefix_lens[", i,
                    "] must be non-negative");
        TORCH_CHECK(i < static_cast<int64_t>(k_caches.size())
                    && i < static_cast<int64_t>(v_caches.size())
                    && k_caches[i].defined() && k_caches[i].dim() == 7
                    && v_caches[i].defined() && v_caches[i].dim() == 7,
                    "Qwen3.5 action step K/V cache[", i,
                    "] must be defined 7D tensors");
        const int64_t capacity = k_caches[i].size(1) * 16;
        TORCH_CHECK(prefix + action_len_ <= capacity,
                    "Qwen3.5 action step cache capacity exceeded at layer ", i,
                    ": prefix=", prefix, " action_len=", action_len_,
                    " capacity=", capacity);
        max_prefix = std::max(max_prefix, prefix);
    }

    // The private Python loop supplies CPU action/prefix. Copy all current
    // bytes every call; public device aliases keep their original publication path.
    at::Tensor action_rpu = action;
    at::Tensor prefix_rpu = rtc_active ? *action_prefix : at::Tensor();
    if (cpu_action) {
        action_rpu = at::empty(action.sizes(),
            at::TensorOptions().dtype(at::kHalf).device(action_out.device()));
        std::memcpy(action_rpu.data_ptr(), action.const_data_ptr(), action.nbytes());
    }
    if (cpu_prefix) {
        prefix_rpu = at::empty(action_prefix->sizes(),
            at::TensorOptions().dtype(at::kHalf).device(action_out.device()));
        std::memcpy(prefix_rpu.data_ptr(), action_prefix->const_data_ptr(), action_prefix->nbytes());
    }

    adaptive_mod_ref_ = adaptive_mod;
    adaptive_mod_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(adaptive_mod.data_ptr<c10::Half>());
    action_input_ref_ = action_rpu;
    action_input_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(action_rpu.data_ptr<c10::Half>());
    action_keep_mask_ref_ = action_keep_mask;
    action_keep_mask_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(action_keep_mask.data_ptr<c10::Half>());
    action_output_ref_ = action_out;
    action_output_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(action_out.data_ptr<c10::Half>());
    if (rtc_active) {
        action_rtc_prefix_ref_ = prefix_rpu;
        action_rtc_prefix_live_base_ = ::rhino_lkn::RpuGetDevAddr(
            prefix_rpu.data_ptr<c10::Half>());
    } else {
        action_rtc_prefix_ref_ = at::Tensor();
        action_rtc_prefix_live_base_ = 0;
    }
    action_trace_active_ = trace_active;
    if (trace_active) {
        auto bind_trace = [](at::Tensor& ref, uint64_t& live_base,
                             at::Tensor& tensor) {
            ref = tensor;
            live_base = ::rhino_lkn::RpuGetDevAddr(
                tensor.data_ptr<c10::Half>());
        };
        bind_trace(
            action_trace_pre_action_ref_,
            action_trace_pre_action_live_base_, trace->pre_action);
        bind_trace(
            action_trace_action_embed_ref_,
            action_trace_action_embed_live_base_, trace->action_embed);
        bind_trace(
            action_trace_final_hidden_ref_,
            action_trace_final_hidden_live_base_, trace->final_hidden);
        bind_trace(
            action_trace_velocity_ref_,
            action_trace_velocity_live_base_, trace->velocity);
        bind_trace(
            action_trace_post_action_ref_,
            action_trace_post_action_live_base_, trace->post_action);
    } else {
        action_trace_pre_action_ref_ = at::Tensor();
        action_trace_action_embed_ref_ = at::Tensor();
        action_trace_final_hidden_ref_ = at::Tensor();
        action_trace_velocity_ref_ = at::Tensor();
        action_trace_post_action_ref_ = at::Tensor();
        action_trace_pre_action_live_base_ = 0;
        action_trace_action_embed_live_base_ = 0;
        action_trace_final_hidden_live_base_ = 0;
        action_trace_velocity_live_base_ = 0;
        action_trace_post_action_live_base_ = 0;
    }
    bool batch_published = false;
    if (g_rpu_ddr_flush_force_enabled) {
        std::array<void*, 10> flush_ptrs{};
        std::array<size_t, 10> flush_sizes{};
        size_t flush_count = 0;
        const auto collect_flush = [&](const at::Tensor& tensor, bool private_copy = false) {
            void* ptr = tensor.data_ptr<c10::Half>();
            auto* owner = ::rhino_lkn::RpuGetHostddr(ptr);
            if (private_copy) {
                const uintptr_t base = owner
                    ? reinterpret_cast<uintptr_t>(owner->get_cpu_ptr()) : 0;
                const size_t owner_bytes = owner ? owner->get_memory_size() : 0;
                const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
                const size_t bytes = tensor.nbytes();
                const uintptr_t limit = std::numeric_limits<uintptr_t>::max();
                bool bounded = owner && base != 0 && bytes > 0 && owner_bytes > 0
                    && owner_bytes <= limit - base && address >= base
                    && address <= limit - 63 && bytes <= limit - address - 63;
                if (bounded) {
                    const size_t offset = address - base;
                    bounded = offset <= owner_bytes && bytes <= owner_bytes - offset
                        && (address & ~uintptr_t(63)) >= base
                        && ((address + bytes + 63) & ~uintptr_t(63)) <= base + owner_bytes;
                }
                if (bounded) {
                    flush_ptrs[flush_count] = ptr;
                    flush_sizes[flush_count++] = bytes;
                } else {
                    rpu_ddr_flush_force_sized(ptr, bytes);
                }
                return;
            }
            if (owner) {
                const uintptr_t base = reinterpret_cast<uintptr_t>(owner->get_cpu_ptr());
                const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
                const size_t size = owner->get_memory_size();
                const uintptr_t limit = std::numeric_limits<uintptr_t>::max();
                if (size > 0 && base <= address && size <= limit - base &&
                    address - base < size && base + size <= limit - 63) {
                    // Preserve the scalar SDK allocation-tail range, including
                    // dirty aliases beyond this Tensor's logical view.
                    flush_ptrs[flush_count] = ptr;
                    flush_sizes[flush_count++] = size - (address - base);
                    return;
                }
            }
            rpu_ddr_flush_force(ptr);
        };
        collect_flush(adaptive_mod);
        collect_flush(action_rpu, cpu_action);
        collect_flush(action_keep_mask);
        collect_flush(action_out);
        if (rtc_active)
            collect_flush(prefix_rpu, cpu_prefix);
        if (trace_active) {
            collect_flush(trace->pre_action);
            collect_flush(trace->action_embed);
            collect_flush(trace->final_hidden);
            collect_flush(trace->velocity);
            collect_flush(trace->post_action);
        }
        if (flush_count > 0) {
            ::rhino_lkn::RpuDdrFlushBatch(
                flush_ptrs.data(), flush_sizes.data(), flush_count);
#if defined(__aarch64__)
            // The SDK Batch ends with dsb ish; retain the original large-tail
            // scalar flush's system-wide completion before device DMA.
            __asm__ volatile("dsb sy" ::: "memory");
#endif
            batch_published = true;
        }
    }
    if (cpu_action || cpu_prefix) {
#if defined(__aarch64__)
        // Complete current CPU copies before mutable device DMA, including
        // scalar fallback or the explicitly disabled force-publication mode.
        if (!batch_published) __asm__ volatile("dsb sy" ::: "memory");
#endif
        if (RpuKernelGraph::has_active()) {
            if (cpu_action) RpuKernelGraph::active().keep_alive(action_input_ref_);
            if (cpu_prefix) RpuKernelGraph::active().keep_alive(action_rtc_prefix_ref_);
        }
    }
    action_step_active_ = true;
    fast_replay_active_ = fast_replay_enabled_;
    TORCH_CHECK(!planned_stage_descriptor.empty(),
                "Qwen3.5 action step requires one complete planner stage "
                "descriptor");
    (void) run_all_layers(
        action_hidden_stage_, k_caches, v_caches, std::nullopt,
        max_prefix, /*is_causal=*/false,
        /*planned_chunk_size=*/0, planned_stage_descriptor);
    return action_velocity_stage_;
}

at::Tensor Qwen3_5Model::prepare_decode_cpu_input(const at::Tensor& hidden_states) {
    // Used by one complete public serialized decode/capture at a time.
    // This receipt lock does not make raw native forward reentrant.
    // Read the entire current CPU value, even on an eventual hit. The receipt
    // describes private copied bytes, never a public Tensor version/address.
    std::vector<uint8_t> current_bytes(hidden_states.nbytes());
    std::memcpy(current_bytes.data(), hidden_states.data_ptr(), current_bytes.size());
    std::lock_guard<std::mutex> lock(decode_cpu_input_mutex_);
    const uint64_t generation = installed_model_state_generation();
    const bool force = rpu_get_ddr_flush_force();
    if (force && decode_cpu_input_published_ &&
        decode_cpu_input_generation_ == generation &&
        decode_cpu_input_owner_.defined() &&
        decode_cpu_input_owner_.sizes() == hidden_states.sizes() &&
        decode_cpu_input_bytes_.size() == current_bytes.size() &&
        std::memcmp(decode_cpu_input_bytes_.data(), current_bytes.data(),
                    current_bytes.size()) == 0) {
        return decode_cpu_input_owner_;
    }
    // Invalidate before allocation/copy/publication, so an exception cannot
    // certify an unpublished owner. Never overwrite the previous DDR owner.
    decode_cpu_input_published_ = false;
    auto next_owner = at::empty(
        hidden_states.sizes(), hidden_states.options().device(at::kPrivateUse1));
    std::memcpy(next_owner.data_ptr(), current_bytes.data(), current_bytes.size());
    rpu_ddr_flush_force_sized(next_owner.data_ptr(), current_bytes.size());
    if (force) {
#if defined(__aarch64__)
        // Complete CPU publication before certifying this private DDR owner.
        __asm__ volatile("dsb sy" ::: "memory");
#endif
        decode_cpu_input_owner_ = next_owner;
        decode_cpu_input_bytes_ = std::move(current_bytes);
        decode_cpu_input_generation_ = generation;
        decode_cpu_input_published_ = true;
    } else {
        decode_cpu_input_owner_ = at::Tensor{};
        decode_cpu_input_bytes_.clear();
    }
    // FMB's original mutable layer-0 DMA binds this local owner and keep_alive
    // retains it through deferred execution. This helper never exposes it.
    return next_owner;
}

// ── forward: stash GDN state for the seam, then all-layers-once ──────────────
at::Tensor Qwen3_5Model::forward(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    std::vector<at::Tensor>& gdn_states,
    std::vector<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position, bool is_causal,
    int64_t planned_chunk_size,
    at::IntArrayRef planned_stage_descriptor) {
    TORCH_CHECK(!z2_bound_,
                "Qwen3.5 text ordinary forward is unavailable while its Z2 lease is active");
    TORCH_CHECK(
        planned_chunk_size == 0 && !planned_stage_descriptor.empty(),
        "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 ordinary text forward "
        "requires one complete planner stage descriptor");
    if (hidden_states.device().is_cpu()) {
        TORCH_CHECK(hidden_states.defined() && hidden_states.dim() == 3 &&
                        hidden_states.size(0) == 1 && hidden_states.size(1) == 1 &&
                        hidden_states.size(2) == hidden_size() &&
                        hidden_states.scalar_type() == at::kHalf &&
                        hidden_states.is_contiguous() && !hidden_states.requires_grad() &&
                        position > 0 && is_causal && !action_mode_ &&
                        mrope_pos_delta_ == 0 && !attention_mask.has_value(),
                    "Qwen3.5 CPU ingress requires ordinary causal text decode "
                    "with contiguous CPU FP16 [1,1,H], position > 0, no mask/grad");
        auto private_hidden = prepare_decode_cpu_input(hidden_states);
        return forward_impl(private_hidden, k_caches, v_caches, gdn_states,
                            conv_states, attention_mask, position, is_causal,
                            planned_chunk_size, planned_stage_descriptor);
    }
    return forward_impl(hidden_states, k_caches, v_caches, gdn_states,
                        conv_states, attention_mask, position, is_causal,
                        planned_chunk_size, planned_stage_descriptor);
}

at::Tensor Qwen3_5Model::forward_z2(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    std::vector<at::Tensor>& gdn_states,
    std::vector<at::Tensor>& conv_states,
    uint64_t epoch,
    uint64_t plan_hash) {
    validate_z2_prefill_contract();
    TORCH_CHECK(z2_bound_ && z2_epoch_ == epoch &&
                    z2_plan_hash_ == plan_hash,
                "Qwen3.5 text Z2 forward received a stale epoch/plan hash");
    TORCH_CHECK(hidden_states.defined() && hidden_states.dim() == 3 &&
                    hidden_states.size(0) == 1 &&
                    hidden_states.size(1) == z2_prepared_execution_len_ &&
                    hidden_states.size(2) == hidden_size() &&
                    hidden_states.device().type() == at::kPrivateUse1 &&
                    hidden_states.scalar_type() == at::kHalf &&
                    hidden_states.is_contiguous(),
                "Qwen3.5 text Z2 hidden must be contiguous device FP16 [1,128,2048]");
    return forward_impl(hidden_states, k_caches, v_caches, gdn_states,
                        conv_states, std::nullopt, /*position=*/0,
                        /*is_causal=*/true, /*planned_chunk_size=*/0,
                        /*planned_stage_descriptor=*/{});
}

at::Tensor Qwen3_5Model::forward_impl(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    std::vector<at::Tensor>& gdn_states,
    std::vector<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position, bool is_causal,
    int64_t planned_chunk_size,
    at::IntArrayRef planned_stage_descriptor) {
    if (num_cores() != 8) {
        TORCH_CHECK(hidden_states.defined() && hidden_states.dim() == 3 &&
                        hidden_states.size(0) == 1 &&
                        hidden_states.size(2) == hidden_size() &&
                        hidden_states.scalar_type() == at::kHalf &&
                        hidden_states.device().type() == at::kPrivateUse1 &&
                        hidden_states.is_contiguous() && is_causal,
                    "Qwen3.5 reduced execution requires causal B1 FP16 hidden states");
        TORCH_CHECK(gdn_states.size() == static_cast<size_t>(num_layers()) &&
                        conv_states.size() == static_cast<size_t>(num_layers()),
                    "Qwen3.5 reduced execution requires one cache slot per layer");
    }
    TORCH_CHECK(!action_nvfp4_.enabled(),
                "G05 NVFP4 weights are Action-only; use the Action entrypoint");
    const int64_t seq_len = hidden_states.size(1);
    TORCH_CHECK(seq_len == 1 || seq_len % 64 == 0,
                "Qwen3.5 text expects decode length 1 or a prefill execution "
                "length divisible by 64; use the Python adapter for ragged prefill");
    // Validate the installed capacity before any cache state or fixed-DMA
    // buffers can change. The Action entrypoints own their separate geometry.
    (void)subclass_chunk_size_cap(seq_len, position);
    if (!action_mode_) {
        const bool use_prefill_rope = seq_len > 1 && has_mrope_ &&
            prefill_cos_.defined() && prefill_cos_.numel() > 0;
        const auto& rope_cos = use_prefill_rope ? prefill_cos_ : cos_;
        const auto& rope_sin = use_prefill_rope ? prefill_sin_ : sin_;
        TORCH_CHECK(
            rope_cos.defined() && rope_sin.defined() &&
                rope_cos.dim() == 2 && rope_cos.sizes() == rope_sin.sizes() &&
                seq_len <= rope_cos.size(0) && position <= rope_cos.size(0) - seq_len,
            "Qwen3.5 request exceeds the installed rotary table capacity");
        // Per-token prefill tables already encode M-RoPE positions. Compare
        // static decode offsets before adding them to avoid integer overflow.
        TORCH_CHECK(
            use_prefill_rope ||
                (mrope_pos_delta_ >= -position &&
                 mrope_pos_delta_ <= rope_cos.size(0) - seq_len - position),
            "Qwen3.5 rotary position delta exceeds the installed table capacity");
    }
    gdn_states_  = &gdn_states;
    conv_states_ = &conv_states;
    fast_replay_active_ = fast_replay_enabled_ && seq_len > 1;

    // Diagnostic-only per-layer hidden-state tap. The G0.5 adapter records
    // debug prefill in a cache separate from the production prefill graph, so
    // these fixed-address DMA destinations are present only in a debug BUILD.
    // Keep one physical-length allocation stable across that cache's REPLAY.
    if (get_debug_export() && !action_mode_ && seq_len > 1) {
        const int64_t batch_size = hidden_states.size(0);
        const int64_t h = hidden_size();
        TORCH_CHECK(batch_size == 1,
                    "Qwen3.5 fused prefill debug export requires batch_size=1");
        if (!per_layer_debug_buf_.defined()
            || per_layer_debug_buf_.size(0) != num_layers()
            || per_layer_debug_buf_.size(1) != batch_size
            || per_layer_debug_buf_.size(2) != seq_len
            || per_layer_debug_buf_.size(3) != h) {
            per_layer_debug_buf_ = at::empty(
                {num_layers(), batch_size, seq_len, h},
                at::TensorOptions()
                    .dtype(at::kHalf)
                    .device(at::kPrivateUse1));
        }
        if (!last_layer_decoder_debug_buf_.defined()
            || last_layer_decoder_debug_buf_.size(0) != batch_size
            || last_layer_decoder_debug_buf_.size(1) != seq_len
            || last_layer_decoder_debug_buf_.size(2) != h) {
            last_layer_decoder_debug_buf_ = at::empty(
                {batch_size, seq_len, h},
                at::TensorOptions()
                    .dtype(at::kHalf)
                    .device(at::kPrivateUse1));
        }
    }

    // Fast replay skips build_gdn(), so refresh every mutable cache base before
    // entering run_all_layers. Only fast replay needs the unconditional CPU or device
    // boundary flush; ordinary text prefill/decode retain the configurable
    // intra-device flush used by build_gdn().
    std::vector<void*> cache_flush_ptrs;
    std::vector<size_t> cache_flush_sizes;
    if (fast_replay_active_ && g_rpu_ddr_flush_force_enabled) {
        cache_flush_ptrs.reserve(2 * num_layers());
        cache_flush_sizes.reserve(2 * num_layers());
    }
    const auto collect_cache_flush = [&](const at::Tensor& tensor) {
        void* ptr = tensor.data_ptr<c10::Half>();
        auto* owner = ::rhino_lkn::RpuGetHostddr(ptr);
        if (owner) {
            const uintptr_t base = reinterpret_cast<uintptr_t>(owner->get_cpu_ptr());
            const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
            const size_t size = owner->get_memory_size();
            const uintptr_t limit = std::numeric_limits<uintptr_t>::max();
            if (size > 0 && base <= address && size <= limit - base &&
                address - base < size && base + size <= limit - 63) {
                // Match unsized SDK flush: the allocation tail, including
                // backing storage beyond this Tensor's logical view.
                cache_flush_ptrs.push_back(ptr);
                cache_flush_sizes.push_back(size - (address - base));
                return;
            }
        }
        // Keep the original SDK lookup/fallback for an unresolvable range.
        rpu_ddr_flush_force(ptr);
    };
    for (int64_t layer_idx = 0; layer_idx < num_layers(); ++layer_idx) {
        if (layer_is_full_[layer_idx])
            continue;
        at::Tensor* st = gdn_state_for_layer(layer_idx);
        at::Tensor* cv = conv_state_for_layer(layer_idx);
        TORCH_CHECK(st && cv && st->defined() && cv->defined(),
                    "Qwen3.5 missing recurrent/conv state cache for layer ",
                    layer_idx);
        if (num_cores() != 8) {
            const auto valid_state = [](const at::Tensor& tensor) {
                return tensor.scalar_type() == at::kHalf &&
                       tensor.device().type() == at::kPrivateUse1 &&
                       tensor.is_contiguous();
            };
            TORCH_CHECK(valid_state(*st) && valid_state(*cv) &&
                            st->sizes() == at::IntArrayRef({gdn_nvh_, gdn_dk_, gdn_dv_}) &&
                            cv->sizes() == at::IntArrayRef({gdn_tp(), gdn_kc_,
                                                          gdn_conv_dim_ / gdn_tp()}),
                        "Qwen3.5 recurrent/conv cache layout does not match the cold GDN core count");
        }
        gdn_state_live_addr_[layer_idx] =
            ::rhino_lkn::RpuGetDevAddr(st->data_ptr<c10::Half>());
        conv_state_live_addr_[layer_idx] =
            ::rhino_lkn::RpuGetDevAddr(cv->data_ptr<c10::Half>());
        if (fast_replay_active_) {
            if (g_rpu_ddr_flush_force_enabled) {
                collect_cache_flush(*st);
                collect_cache_flush(*cv);
            }
        } else {
            rpu_ddr_flush(st->data_ptr<c10::Half>());
            rpu_ddr_flush(cv->data_ptr<c10::Half>());
        }
    }
    if (g_rpu_ddr_flush_force_enabled && !cache_flush_ptrs.empty()) {
        ::rhino_lkn::RpuDdrFlushBatch(cache_flush_ptrs.data(),
                                    cache_flush_sizes.data(), cache_flush_ptrs.size());
#if defined(__aarch64__)
        // SDK batch ends with dsb ish; retain the old large-tail flush's
        // system-wide completion before the device consumes these caches.
        __asm__ volatile("dsb sy" ::: "memory");
#endif
    }

    at::Tensor out = run_all_layers(hidden_states, k_caches, v_caches,
                                    attention_mask, position, is_causal,
                                    planned_chunk_size,
                                    planned_stage_descriptor);

    // PREFILL 的 resolved chunk_size 单独留一份。基类的 last_resolved_chunk_size_
    // (fused_model_base.cpp:546) 每次 compute_chunks 都覆写,而 decode 跑在 prefill 之后
    // 且 seq_len=1 → planner 固定挑 cs=16,于是事后从 Python 读回来的永远是 16,
    // 拿不到 prefill 真正用的值。vision 没有 decode 所以不存在这个问题。
    if (seq_len > 1)
        last_prefill_chunk_size_ = get_last_resolved_chunk_size();

    if (get_debug_export() && !action_mode_ && seq_len > 1
        && per_layer_debug_buf_.defined()
        && last_layer_decoder_debug_buf_.defined()) {
        // Publish views of the stable debug staging allocations. The getter is
        // the post-capture synchronization boundary; cloning here would add 24
        // more device allocations and can run before a BUILD scope executes.
        for (int64_t layer_idx = 0; layer_idx < num_layers(); ++layer_idx) {
            const std::string key =
                "L" + std::to_string(layer_idx) + "_layer_output";
            g_debug_tensors[key] = per_layer_debug_buf_[layer_idx];
        }
        const std::string last_prefix =
            "L" + std::to_string(num_layers() - 1);
        g_debug_tensors[last_prefix + "_decoder_output"] =
            last_layer_decoder_debug_buf_;
        g_debug_tensors[last_prefix + "_final_norm_output"] =
            per_layer_debug_buf_[num_layers() - 1];
    }
    gdn_states_  = nullptr;
    conv_states_ = nullptr;

    return out;
}

uint32_t Qwen3_5Model::layer_input_residual_addr(
    int layer_idx,
    const ChunkInfo& chunk) const {
    if (!z2_bound_ || layer_idx != 0) {
        return addr(0, "residual1");
    }
    TORCH_CHECK(z2_input_port_addr_ != 0,
                "Qwen3.5 text Z2: layer-0 input port is unset");
    TORCH_CHECK(chunk.offset >= 0 && chunk.len > 0 &&
                    chunk.offset <= z2_prepared_execution_len_ &&
                    chunk.len <= z2_prepared_execution_len_ - chunk.offset,
                "Qwen3.5 text Z2: layer-0 chunk escapes the prepared input port");
    const uint64_t byte_offset =
        static_cast<uint64_t>(chunk.offset) *
        static_cast<uint64_t>(hidden_size()) * sizeof(c10::Half);
    TORCH_CHECK(byte_offset <= std::numeric_limits<uint32_t>::max() -
                                   z2_input_port_addr_,
                "Qwen3.5 text Z2: layer-0 port address overflows uint32");
    return z2_input_port_addr_ + static_cast<uint32_t>(byte_offset);
}

void Qwen3_5Model::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    const int64_t h = hidden_size();
    const bool z2_layer0 = z2_bound_ && layer_idx == 0;
    const uint32_t input_residual =
        layer_input_residual_addr(layer_idx, chunk);
    TORCH_CHECK(!high_precision_ || ctx().has_complete_physical_manifest(),
                "G05 HIGH math requires a COMPLETE descriptor before emission");

    if (ctx().has_complete_physical_manifest() &&
        layer_idx == 0 && chunk.idx == 0) {
        if (num_cores() != 8) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_TEXT_CORE_TOPOLOGY_SITE,
                /*selector=*/1, /*flags=*/0, cold_topology_arguments());
        }
        if (high_precision_) {
            consume_high_precision_math();
        }
        const int64_t action_body_iterations =
            action_step_active_ && action_loop_active_ ? action_num_steps_ : 1;
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE,
            QWEN35_TEXT_ACTION_SCHEDULE_SITE,
            action_mode_ ? 2 : 1, /*resolved_flags=*/0,
            {action_mode_ ? 1 : 0, action_input_w_.defined() ? 1 : 0,
             action_loop_active_ ? 1 : 0, action_rtc_active_ ? 1 : 0,
             action_step_active_ ? 1 : 0, action_body_iterations,
             action_len_, action_dim_pad_}, chunk.idx);
        const int64_t full_layer_count = std::count_if(
            layer_is_full_.begin(), layer_is_full_.end(),
            [](int value) { return value != 0; });
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE,
            QWEN35_TEXT_QK_NORM_SCHEDULE_SITE,
            has_qk_norm_ ? 2 : 1, /*resolved_flags=*/0,
            {has_qk_norm_ ? 1 : 0, has_mrope_ ? 1 : 0,
             head_dim(), attn_tp(), full_layer_count}, chunk.idx);
    }

    // A layer = input DMA → input_layernorm → token mixer → post_norm + MLP + output.
    // ① input DMA: the first layer of the cross-layer group brings h_in into residual1.
    if (!ctx().input_in_spm && !z2_layer0) {
        emit_layer_input_dma(layer_idx, chunk);   // h_in → residual1
    }
    // ② input_layernorm (HF: at the DecoderLayer, NOT inside the mixer). residual1 keeps
    //    h_in (the residual); "input_norm" gets the normed input the mixer reads.
    //    Exact interior Text carry has only owner raw R and already-normalized N.
    if (action_mode_) {
        load_adaptive_mod_row(2 * layer_idx);
        // Always advance the row for the current attention residual gate.
        // Only the matching prior same-group MLP tail supplied this norm.
        if (!action_mlp_tail_handoff(layer_idx,chunk,false)) {
            apply_adaptive_norm(
                input_residual, addr(0, "input_norm"), chunk);
        }
    } else if (text_mlp_tail_handoff(layer_idx,chunk,false)) {
        // Only the successful previous callback in this same SPM group emits
        // this normalized N. Validate its COMPLETE schedule before any mixer.
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            QWEN35_TEXT_MLP_TAIL_SITE,1,0,text_mlp_tail_arguments(ctx().physical_manifest().logical_length),chunk.idx);
    } else {
        rpu_launch_rmsnorm_spm_kernel(
            input_residual, addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "norm_w"), chunk.len, h, eps_,
            rmsnorm_route(chunk.len, h), num_cores());
    }
    // This layer-local result cannot survive an abort or leak into the next layer.
    bool text_owner_norm_emitted = false;
    bool action_norm_emitted = false;
    // ③ token mixer (reads "input_norm"; ends with an all_reduce that adds the h_in residual).
    if (layer_is_full_[layer_idx]) {
        // Ordinary full-attention returns a full raw stream in input_norm.
        // The exact owner-RMS route instead keeps owner raw there and full
        // normalized in residual1, reported only through this layer-local flag.
        build_full_attention(layer_idx, chunk, &text_owner_norm_emitted, &action_norm_emitted);
    } else {
        // decode = recurrent single step (chunk.len<=1); prefill = chunked delta-rule (L>1).
        // GDN uses the same full-raw fallback or original-owner raw ABI;
        // only this exact prefill producer may publish the local norm flag.
        if (chunk.len <= 1) build_gdn(layer_idx, chunk, /*decode=*/true);
        else                build_gdn(layer_idx, chunk, /*decode=*/false,
                                      &text_owner_norm_emitted);
    }
    // ④ shared post_norm + MLP + final_norm + output. Only the successful
    // owner producer skips postnorm; all fallbacks keep the full raw ABI.
    emit_mlp_and_output(layer_idx, chunk, text_owner_norm_emitted, action_norm_emitted);
}

bool Qwen3_5Model::use_action_nvfp4(int64_t rows) const {
    // set_weights proved every family and the exact L24/H1024 profile before
    // publishing these owners. The precision is cold and cannot change later.
    return action_nvfp4_.enabled() && action_mode_ &&
        !z2_bound_ && rows == 32 && num_cores() == 8;
}

std::vector<int64_t> Qwen3_5Model::action_nvfp4_arguments(int family) const {
    TORCH_CHECK(action_nvfp4_.enabled() && family >= 0 && family < 5,
                "G05 NVFP4 requires a valid installed projection family");
    const int64_t n = G05Nvfp4Tables::output_sizes[family];
    const int64_t k = G05Nvfp4Tables::input_sizes[family];
    const int partition = G05Nvfp4Tables::partitions[family];
    const int64_t local_n = partition ? n / 8 : n;
    const int64_t local_k = partition ? k : k / 8;
    const auto tile = linear_acc32_
        ? rpu_pl_tiling::TilePick{partition && n == 4096 ? 48 : 64,
                                  partition && n == 4096 ? 384 : 320}
        : rpu_pl_tiling::select_tile_nvfp4_acc16(32, local_n, local_k);
    // Version, explicit family, layer-domain, M/N/K, TP layout, actual accumulation,
    // block16/striped_v2, 24 FP32 scales/256-byte slab, actual generated tile.
    return {1, action_nvfp4_.enabled() ? 1 : 0,
            family, 24, 32, n, k, partition, 8, linear_acc32_ ? 32 : 16, 16, 2, 24, 256,
            local_n, local_k, tile.m_tile, tile.n_tile, 128};
}

void Qwen3_5Model::launch_linear(
    int64_t chunk_idx,
    uint32_t input, const at::Tensor& weight, uint32_t output,
    int64_t m, int64_t n, int64_t k, int partition, int num_cores,
    uint32_t bias_spm_addr, const at::Tensor& scale,
    int nvfp4_family, int layer, const at::Tensor& negative_zero) {
    if (action_nvfp4_.enabled() && nvfp4_family >= 0) {
        TORCH_CHECK(use_action_nvfp4(m) && nvfp4_family < 5 &&
                        layer >= 0 && layer < 24 && num_cores == 8 && bias_spm_addr == 0 &&
                        n == G05Nvfp4Tables::output_sizes[nvfp4_family] &&
                        k == G05Nvfp4Tables::input_sizes[nvfp4_family] &&
                        partition == G05Nvfp4Tables::partitions[nvfp4_family] &&
                        ctx().has_complete_physical_manifest(),
                    "G05 NVFP4 launch requires exact M32/L24/TP8 family and COMPLETE route");
        const auto& owner = layer_weights_[layer];
        const std::array<const at::Tensor*, 5> weights{
            &owner.q_w, &owner.o_w, &owner.gate_w, &owner.up_w, &owner.down_w};
        const std::array<const at::Tensor*, 5> scales{
            &owner.q_ws, &owner.o_ws, &owner.gate_ws, &owner.up_ws, &owner.down_ws};
        TORCH_CHECK(weight.is_same(*weights[nvfp4_family]) &&
                        scale.is_same(*scales[nvfp4_family]),
                    "G05 NVFP4 projection family/layer owner mismatch");
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, QWEN35_ACTION_NVFP4_LINEAR_SITE,
            static_cast<int64_t>(linear_acc32_ ? FmbLinearRouteSelector::G05_NVFP4_V2_ACC32
                : FmbLinearRouteSelector::G05_NVFP4_V2_ACC16), 0,
                action_nvfp4_arguments(nvfp4_family), chunk_idx * 5 + nvfp4_family);
        rpu_launch_g05_nvfp4_v2_kernel(
            input, weight, output, m, n, k, partition, num_cores, scale,
            addr(0, G05Nvfp4Tables::names[nvfp4_family]), static_cast<uint16_t>(layer), linear_acc32_);
        return;
    }
    TORCH_CHECK(!(weight.scalar_type() == at::kByte && scale.defined() &&
                      scale.scalar_type() == at::kByte),
                "G05 NVFP4 must use an explicit projection family/layer route");
    const FmbLinearRouteSelector selector =
        linear_route_selector(weight, m);
    const int64_t invocation = linear_invocation(chunk_idx, selector);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, QWEN35_TEXT_LINEAR_SITE,
            static_cast<int64_t>(selector), /*resolved_flags=*/0,
            /*resolved_arguments=*/{}, invocation);
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        input, weight, output, m, n, k, partition, num_cores,
        bias_spm_addr, scale,
        /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
        /*force_acc32=*/linear_acc32_,
        /*prefer_gemv=*/selector == FmbLinearRouteSelector::GEMV,
        /*qkv_planar_direct=*/false, /*prefer_row_weight_reuse=*/false,
        negative_zero);
}

void Qwen3_5Model::consume_manifest_route(
    FmbRouteFamily family, int64_t site_id,
    int64_t selector, int64_t invocation) {
    if (!ctx().has_complete_physical_manifest()) return;
    ctx().consume_physical_route(
        family, site_id, selector, /*resolved_flags=*/0,
        /*resolved_arguments=*/{}, invocation);
}

bool Qwen3_5Model::use_action_mod_body() const {
    // Match the G05 ten-step expert. The cold stage query binds the same loop
    // state as forward before asking the allocator for this larger table.
    return action_mode_ && action_step_active_ && action_loop_active_ &&
        action_num_steps_ == 10 && action_len_ == 32 &&
        (linear_acc32_ || action_mod_body_w8_ || action_mod_body_fp16_ || use_action_nvfp4(action_len_)) &&
        !has_gdn_ && num_layers() == 24 && hidden_size() == 1024 && num_cores() == 8;
}

bool Qwen3_5Model::has_text_fp16_projection_owners() const {
    // Cold-only proof for the exact G05 retained Text owner-row ABI. Every
    // projection bordering the full/GDN mixer and MLP is genuine FP16; no
    // quantization scale can survive this admission. No weights are repacked.
    if (action_mode_ || !fast_replay_enabled_ || !use_silu_ || !has_gdn_ ||
        eps_ != 1e-6 || num_layers() != 24 || hidden_size() != 2048 ||
        intermediate_size() != 6144 || num_cores() != 8 || attn_tp() != 8 ||
        gdn_tp() != 8 || mlp_tp() != 8 || num_q_heads() != 8 ||
        num_kv_heads() != 8 || head_dim() != 256 || layer_weights_.size() != 24 ||
        layer_is_full_.size() != 24 || gdn_nvh_ != 16 || gdn_dk_ != 128 ||
        gdn_dv_ != 128 || gdn_conv_dim_ != 6144 || gdn_kc_ != 4 ||
        !layer_weights_.front().gate_w.defined() ||
        layer_weights_.front().gate_w.scalar_type() != at::kHalf) return false;
    const auto device = layer_weights_.front().gate_w.device();
    const auto tensor = [&](const at::Tensor& t,int64_t n,int64_t k) {
        return t.defined() && t.scalar_type() == at::kHalf &&
            t.device().type() == c10::DeviceType::PrivateUse1 && t.device() == device &&
            t.is_contiguous() && t.storage_offset() == 0 &&
            (k == 0 ? (t.dim() == 1 && t.numel() == n)
                    : (t.dim() == 2 && t.size(0) == n && t.size(1) == k));
    };
    const auto linear = [&](const at::Tensor& w,const at::Tensor& scale,
                            int64_t n,int64_t k) {
        return tensor(w,n,k) && (!scale.defined() || scale.numel() == 0);
    };
    for (size_t layer=0;layer<24;++layer) {
        const auto& w=layer_weights_[layer];
        if (bool(layer_is_full_[layer]) != ((layer+1)%4 == 0) ||
            !tensor(w.input_norm_w,2048,0) || !tensor(w.post_norm_w,2048,0) ||
            !linear(w.gate_w,w.gate_ws,6144,2048) ||
            !linear(w.up_w,w.up_ws,6144,2048) ||
            !linear(w.down_w,w.down_ws,2048,6144)) return false;
        if (layer_is_full_[layer]) {
            if (!linear(w.q_w,w.q_ws,2048,2048) ||
                !linear(w.k_w,w.k_ws,2048,2048) ||
                !linear(w.v_w,w.v_ws,2048,2048) ||
                !linear(w.o_w,w.o_ws,2048,2048) ||
                !linear(w.attn_gate_w,w.attn_gate_ws,2048,2048)) return false;
        } else {
            if (!linear(w.gdn_q_w,w.gdn_q_ws,2048,2048) ||
                !linear(w.gdn_k_w,w.gdn_k_ws,2048,2048) ||
                !linear(w.gdn_v_w,w.gdn_v_ws,2048,2048) ||
                !linear(w.gdn_in_z_w,w.gdn_in_z_ws,2048,2048) ||
                !linear(w.gdn_out_w,w.gdn_out_ws,2048,2048) ||
                !linear(w.gdn_b_bg_w,w.gdn_b_bg_ws,128,2048) ||
                !linear(w.gdn_a_bg_w,w.gdn_a_bg_ws,128,2048)) return false;
        }
    }
    return true;
}

void Qwen3_5Model::prepare_text_owner_rms() {
    text_mlp_tail_available_ = false;
    prepare_text_gdn_owner_rms();
    text_owner_rms_available_ = false;
    if (high_precision_ || action_mode_ || !fast_replay_enabled_ || !use_silu_ ||
        !has_gdn_ || eps_ != 1e-6 || num_layers() != 24 || hidden_size() != 2048 ||
        intermediate_size() != 6144 || num_cores() != 8 || attn_tp() != 8 ||
        gdn_tp() != 8 || mlp_tp() != 8 || num_q_heads() != 8 ||
        num_kv_heads() != 8 || head_dim() != 256 || layer_weights_.size() != 24 ||
        layer_is_full_.size() != 24) return;
    const bool fp16 = has_text_fp16_projection_owners();
    for (size_t layer = 0; layer < 24; ++layer) {
        if (bool(layer_is_full_[layer]) != ((layer+1)%4 == 0)) return;
        if (!layer_is_full_[layer]) continue;
        const auto& weights = layer_weights_[layer];
        if (!fp16) for (const auto* w : {&weights.q_w,&weights.k_w,&weights.v_w,&weights.o_w,
                &weights.attn_gate_w,&weights.gate_w,&weights.up_w,&weights.down_w}) {
            if (!w->defined() || w->scalar_type() != at::kChar ||
                w->device().type() != c10::DeviceType::PrivateUse1) return;
        }
        const auto& norm = weights.post_norm_w;
        if (!norm.defined() || norm.scalar_type() != at::kHalf ||
            norm.device() != weights.o_w.device() || !norm.is_contiguous() ||
            norm.storage_offset() != 0 || norm.dim() != 1 || norm.numel() != 2048)
            return;
    }
    text_owner_rms_available_ = KernelCache::instance().has_loaded(
        KernelId::G05_TEXT_RING_OWNER_RMS_M256H2048);
    prepare_text_mlp_tail();
}

bool Qwen3_5Model::use_text_owner_rms(
    const FmbThreeStageChunkPlan& plan, int64_t physical_len, int64_t logical_len,
    int64_t position, int64_t batch) const {
    if (!text_owner_rms_available_ || action_mode_ || z2_bound_ || get_debug_export() ||
        !fast_replay_enabled_ || !retained_prefill_graph_ ||
        physical_len != 256 || !(logical_len == 245 || (logical_len >= 193 && logical_len <= 256 && logical_len % 16 == 1)) || position != 0 || batch != 1 ||
        plan.qkv.chunks.size() != 1 || plan.compute.chunks.size() != 1) return false;
    const auto& qkv = plan.qkv.chunks.front();
    const auto& compute = plan.compute.chunks.front();
    return qkv.idx == 0 && qkv.offset == 0 && qkv.len == 256 &&
        compute.idx == 0 && compute.offset == 0 && compute.len == 256;
}

std::vector<int64_t> Qwen3_5Model::text_owner_rms_arguments(int64_t logical_len) const {
    TORCH_CHECK((logical_len == 245 || (logical_len >= 193 && logical_len <= 256 && logical_len % 16 == 1)),"G05 Text fused route requires P245 or single E256 P%16=1 in [193,256]");
    // Version, exact M/H/TP, owner elements/rounds/rows, round split,
    // logical prefix, six full-attention layers, ACC bits, norm FP16 eps,
    // original-position raw ABI, exact residual=normalized alias, cold payload.
    return {2,256,2048,8,65536,2,32,63488,2048,logical_len,6,linear_acc32_ ? 32 : 16,17,1,1,
            text_owner_rms_available_ ? 1 : 0,193,256,16,1,245};
}

void Qwen3_5Model::launch_text_owner_rms(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(ctx().has_complete_physical_manifest() &&
        use_text_owner_rms(ctx().stage_plan,ctx().seq_len,
            ctx().physical_manifest().logical_length,ctx().position,ctx().batch_size) &&
        layer >= 0 && layer < 24 && layer_is_full_[layer] &&
        chunk.idx == 0 && chunk.offset == 0 && chunk.len == 256,
        "G05 Text owner RMS escaped its exact six-layer COMPLETE profile");
    ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,QWEN35_TEXT_OWNER_RMS_SITE,
        1,0,text_owner_rms_arguments(ctx().physical_manifest().logical_length),chunk.idx);
    rpu_launch_g05_text_owner_rms_m256h2048_kernel(
        addr(0,"oproj"),addr(0,"residual1"),addr(0,"input_norm"),
        layer_addr(layer,0,"post_norm_w"));
}

void Qwen3_5Model::prepare_text_gdn_owner_rms() {
    text_gdn_owner_rms_available_ = false;
    // Independent from the six full-attention capability: a GDN owner may
    // fail this narrower proof without disabling the existing R28 route.
    if (high_precision_ || action_mode_ || !fast_replay_enabled_ || !use_silu_ ||
        !has_gdn_ || eps_ != 1e-6 || num_layers() != 24 || hidden_size() != 2048 ||
        intermediate_size() != 6144 || num_cores() != 8 || attn_tp() != 8 ||
        gdn_tp() != 8 || mlp_tp() != 8 || num_q_heads() != 8 ||
        num_kv_heads() != 8 || head_dim() != 256 || layer_weights_.size() != 24 ||
        layer_is_full_.size() != 24 || gdn_nvh_ != 16 || gdn_dk_ != 128 ||
        gdn_dv_ != 128 || gdn_conv_dim_ != 6144 || gdn_kc_ != 4) return;
    const bool fp16 = has_text_fp16_projection_owners();
    for (size_t layer = 0; layer < 24; ++layer) {
        if (bool(layer_is_full_[layer]) != ((layer+1)%4 == 0)) return;
        if (layer_is_full_[layer]) continue;
        const auto& weights = layer_weights_[layer];
        // The FP16 branch has the full24-layer proof; retain the narrower
        // existing four-family W8 proof independently.
        // GDN scalar b/a projections intentionally keep their original FP16.
        const std::array<const at::Tensor*,4> projections{
            &weights.gdn_out_w,&weights.gate_w,&weights.up_w,&weights.down_w};
        const std::array<const at::Tensor*,4> scales{
            &weights.gdn_out_ws,&weights.gate_ws,&weights.up_ws,&weights.down_ws};
        constexpr int64_t n[] = {2048,6144,6144,2048};
        constexpr int64_t k[] = {2048,2048,2048,6144};
        if (!fp16) for (size_t family = 0; family < projections.size(); ++family) {
            const auto& weight = *projections[family];
            const auto& scale = *scales[family];
            if (!weight.defined() || weight.scalar_type() != at::kChar ||
                weight.device().type() != c10::DeviceType::PrivateUse1 ||
                weight.device() != weights.gdn_out_w.device() ||
                !weight.is_contiguous() || weight.storage_offset() != 0 ||
                weight.dim() != 2 || weight.size(0) != n[family] ||
                weight.size(1) != k[family] || !scale.defined() ||
                scale.scalar_type() != at::kHalf || scale.device() != weight.device() ||
                !scale.is_contiguous() || scale.storage_offset() != 0 ||
                scale.dim() != 1 || scale.numel() != n[family]) return;
        }
        const auto& norm = weights.post_norm_w;
        if (!norm.defined() || norm.scalar_type() != at::kHalf ||
            norm.device() != weights.gdn_out_w.device() || !norm.is_contiguous() ||
            norm.storage_offset() != 0 || norm.dim() != 1 || norm.numel() != 2048)
            return;
    }
    text_gdn_owner_rms_available_ = KernelCache::instance().has_loaded(
        KernelId::G05_TEXT_RING_OWNER_RMS_M256H2048);
}

bool Qwen3_5Model::use_text_gdn_owner_rms(
    const FmbThreeStageChunkPlan& plan, int64_t physical_len, int64_t logical_len,
    int64_t position, int64_t batch) const {
    if (!text_gdn_owner_rms_available_ || action_mode_ || z2_bound_ || get_debug_export() ||
        !fast_replay_enabled_ || !retained_prefill_graph_ ||
        physical_len != 256 || !(logical_len == 245 || (logical_len >= 193 && logical_len <= 256 && logical_len % 16 == 1)) || position != 0 || batch != 1 ||
        plan.qkv.chunks.size() != 1 || plan.compute.chunks.size() != 1) return false;
    const auto& qkv = plan.qkv.chunks.front();
    const auto& compute = plan.compute.chunks.front();
    return qkv.idx == 0 && qkv.offset == 0 && qkv.len == 256 &&
        compute.idx == 0 && compute.offset == 0 && compute.len == 256;
}

std::vector<int64_t> Qwen3_5Model::text_gdn_owner_rms_arguments(int64_t logical_len) const {
    TORCH_CHECK((logical_len == 245 || (logical_len >= 193 && logical_len <= 256 && logical_len % 16 == 1)),"G05 Text fused route requires P245 or single E256 P%16=1 in [193,256]");
    // Same paced/base math and original owner-row ABI as the six FA layers,
    // but a distinct 18-layer producer with a dead 128 KiB RAW-prefix alias.
    return {2,256,2048,8,65536,2,32,63488,2048,logical_len,18,linear_acc32_ ? 32 : 16,17,1,1,
            text_gdn_owner_rms_available_ ? 1 : 0,16,128,128,6144,4,131072,20,1,193,256,16,1,245};
}

void Qwen3_5Model::launch_text_gdn_owner_rms(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(ctx().has_complete_physical_manifest() &&
        use_text_gdn_owner_rms(ctx().stage_plan,ctx().seq_len,
            ctx().physical_manifest().logical_length,ctx().position,ctx().batch_size) &&
        layer >= 0 && layer < 24 && !layer_is_full_[layer] &&
        chunk.idx == 0 && chunk.offset == 0 && chunk.len == 256,
        "G05 Text GDN owner RMS escaped its exact eighteen-layer COMPLETE profile");
    ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,QWEN35_TEXT_GDN_OWNER_RMS_SITE,
        1,0,text_gdn_owner_rms_arguments(ctx().physical_manifest().logical_length),chunk.idx);
    // gdn_c_out_tok aliases the first 128 KiB of input_norm. The preceding
    // W8 out-projection has finished reading it, just as before the old ring.
    // The wrapper checks all four live full spans are aligned and disjoint.
    rpu_launch_g05_text_owner_rms_m256h2048_kernel(
        addr(0,"gdn_c_m"),addr(0,"residual1"),addr(0,"input_norm"),
        layer_addr(layer,0,"post_norm_w"));
}

void Qwen3_5Model::prepare_text_mlp_tail() {
    text_mlp_tail_available_ = false;
    // Require the exact retained all24-layer owner ABI already proved at the
    // two mixer tails; a failure here leaves both existing capabilities intact.
    if (!text_owner_rms_available_ || !text_gdn_owner_rms_available_ ||
        action_mode_ || !fast_replay_enabled_ || eps_ != 1e-6 ||
        num_layers() != 24 || hidden_size() != 2048 || intermediate_size() != 6144 ||
        num_cores() != 8 || attn_tp() != 8 || gdn_tp() != 8 || mlp_tp() != 8 ||
        layer_weights_.size() != 24 || layer_is_full_.size() != 24) return;
    if (has_text_fp16_projection_owners()) {
        // Same generated FP16 output roots and BASE owner RMS arithmetic as
        // the W8 path; the producer/consumer handoff remains exact P245/C256.
        text_mlp_tail_available_ = KernelCache::instance().has_loaded(
            KernelId::G05_TEXT_RING_OWNER_RMS_M256H2048);
        return;
    }
    const auto device = layer_weights_.front().down_w.device();
    const auto tensor = [&](const at::Tensor& t,at::ScalarType dtype,
                            int64_t n,int64_t k) {
        return t.defined() && t.scalar_type() == dtype &&
            t.device().type() == c10::DeviceType::PrivateUse1 && t.device() == device &&
            t.is_contiguous() && t.storage_offset() == 0 &&
            (k == 0 ? (t.dim() == 1 && t.numel() == n)
                    : (t.dim() == 2 && t.size(0) == n && t.size(1) == k));
    };
    const auto linear = [&](const at::Tensor& w,const at::Tensor& s,int64_t n,int64_t k) {
        return tensor(w,at::kChar,n,k) && tensor(s,at::kHalf,n,0);
    };
    for (size_t layer=0;layer<24;++layer) {
        const auto& w=layer_weights_[layer];
        if (bool(layer_is_full_[layer]) != ((layer+1)%4 == 0) ||
            !tensor(w.input_norm_w,at::kHalf,2048,0) ||
            !tensor(w.post_norm_w,at::kHalf,2048,0) ||
            !linear(w.gate_w,w.gate_ws,6144,2048) ||
            !linear(w.up_w,w.up_ws,6144,2048) ||
            !linear(w.down_w,w.down_ws,2048,6144)) return;
        if (layer_is_full_[layer]) {
            if (!linear(w.q_w,w.q_ws,2048,2048) ||
                !linear(w.k_w,w.k_ws,2048,2048) ||
                !linear(w.v_w,w.v_ws,2048,2048) ||
                !linear(w.o_w,w.o_ws,2048,2048) ||
                !linear(w.attn_gate_w,w.attn_gate_ws,2048,2048)) return;
        } else {
            if (!linear(w.gdn_q_w,w.gdn_q_ws,2048,2048) ||
                !linear(w.gdn_k_w,w.gdn_k_ws,2048,2048) ||
                !linear(w.gdn_v_w,w.gdn_v_ws,2048,2048) ||
                !linear(w.gdn_in_z_w,w.gdn_in_z_ws,2048,2048) ||
                !linear(w.gdn_out_w,w.gdn_out_ws,2048,2048) ||
                !tensor(w.gdn_b_bg_w,at::kHalf,128,2048) ||
                !tensor(w.gdn_a_bg_w,at::kHalf,128,2048) ||
                (w.gdn_b_bg_ws.defined() && w.gdn_b_bg_ws.numel() != 0) ||
                (w.gdn_a_bg_ws.defined() && w.gdn_a_bg_ws.numel() != 0)) return;
        }
    }
    text_mlp_tail_available_ = KernelCache::instance().has_loaded(
        KernelId::G05_TEXT_RING_OWNER_RMS_M256H2048);
}

bool Qwen3_5Model::use_text_mlp_tail(
    const FmbThreeStageChunkPlan& plan,int64_t physical_len,int64_t logical_len,
    int64_t position,int64_t batch) const {
    if (!text_mlp_tail_available_ ||
        !use_text_owner_rms(plan,physical_len,logical_len,position,batch) ||
        !use_text_gdn_owner_rms(plan,physical_len,logical_len,position,batch) ||
        plan.chunk_mode != ChunkMode::SEQUENTIAL || plan.input.chunks.size() != 1 ||
        plan.spans.size() != 1) return false;
    const auto& input=plan.input.chunks.front();const auto& span=plan.spans.front();
    return input.idx == 0 && input.offset == 0 && input.len == 256 &&
        span.offset == 0 && span.len == 256 && span.group_id <= 0;
}

bool Qwen3_5Model::text_mlp_tail_handoff(
    int layer,const ChunkInfo& chunk,bool producer) {
    if (!ctx().has_complete_physical_manifest() ||
        !use_text_mlp_tail(ctx().stage_plan,ctx().seq_len,
            ctx().physical_manifest().logical_length,ctx().position,ctx().batch_size) ||
        ctx().body_iter != 0 || chunk.idx != 0 || chunk.offset != 0 || chunk.len != 256)
        return false;
    // static_config fixes one24-layer group and one Text body. The actual
    // sequential FMB traversal installs both flags; a throwing callback never
    // visits its successor. No readiness member survives abort/REPLAY.
    return producer ? (layer >= 0 && layer < 23 && ctx().output_to_spm)
                    : (layer > 0 && layer < 24 && ctx().input_in_spm);
}

std::vector<int64_t> Qwen3_5Model::text_mlp_tail_arguments(int64_t logical_len) const {
    TORCH_CHECK((logical_len == 245 || (logical_len >= 193 && logical_len <= 256 && logical_len % 16 == 1)),"G05 Text fused route requires P245 or single E256 P%16=1 in [193,256]");
    // Version,M/H/I,TP/ACC,logical/P/B,one24-layergroup/body,producer0..22,
    // consumer1..23,next gamma,full roots/ownerrows/rounds,RAW=R/norm=N,
    // same-group flags,BASE eps and immutable cold capability.
    return {2,256,2048,6144,8,linear_acc32_ ? 32 : 16,logical_len,0,1,24,1,0,22,1,23,1,
            1048576,32,65536,2,63488,2048,1,1,1,1,
            static_cast<int64_t>(c10::Half(eps_).x),text_mlp_tail_available_?1:0,193,256,16,1,245};
}

void Qwen3_5Model::launch_text_mlp_tail(int layer,const ChunkInfo& chunk) {
    TORCH_CHECK(text_mlp_tail_handoff(layer,chunk,true),
                "G05 Text MLP tail escaped its same-group COMPLETE next input RMS");
    ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
        QWEN35_TEXT_MLP_ALL_REDUCE_SITE,text_ring_route(256,2048),0,{},chunk.idx);
    ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
        QWEN35_TEXT_MLP_TAIL_SITE,1,0,text_mlp_tail_arguments(ctx().physical_manifest().logical_length),chunk.idx);
    // Down has retired its product read from R. The existing wrapper checks
    // full disjoint roots and gamma; R keeps only original owner32 raw rows,
    // N receives full normalized input for the next GDN/full-attention layer.
    rpu_launch_g05_text_owner_rms_m256h2048_kernel(
        addr(0,"down"),addr(0,"input_norm"),addr(0,"residual1"),
        layer_addr(layer+1,0,"norm_w"));
}

void Qwen3_5Model::prepare_text_gateup_pair() {
    text_silu_round_mul_available_ = false;
    text_gateup_pair_available_ = false;
    text_gateup_pair_w8_ = false;
    // G05 binds accumulation and fast replay during cold installation. Recompute at
    // all three cold setters; public text does not enable this private replay
    // scope. The optional payload never widens the arithmetic/profile proof.
    if (action_mode_ || !fast_replay_enabled_ || !use_silu_ ||
        !has_gdn_ || num_layers() != 24 || hidden_size() != 2048 ||
        intermediate_size() != 6144 || num_cores() != 8 || attn_tp() != 8 ||
        gdn_tp() != 8 || mlp_tp() != 8 || num_q_heads() != 8 ||
        num_kv_heads() != 8 || head_dim() != 256 || layer_weights_.size() != 24 ||
        layer_is_full_.size() != 24) return;
    const bool w8 = layer_weights_.front().gate_w.defined() &&
        layer_weights_.front().gate_w.scalar_type() == at::kChar;
    const auto id = w8
        ? (linear_acc32_ ? KernelId::G05_TEXT_GATE_UP_W8A16_ACC32_M256N768K2048
                         : KernelId::G05_TEXT_GATE_UP_W8A16_ACC16_M256N768K2048)
        : (linear_acc32_ ? KernelId::G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048
                         : KernelId::G05_TEXT_GATE_UP_FP16_ACC16_M256N768K2048);
    if (!KernelCache::instance().has_loaded(id)) return;
    for (size_t layer = 0; layer < layer_weights_.size(); ++layer) {
        if (bool(layer_is_full_[layer]) != ((layer + 1) % 4 == 0)) return;
        const auto& weights = layer_weights_[layer];
        for (const auto* weight : {&weights.gate_w, &weights.up_w}) {
            if (!weight->defined() || weight->scalar_type() != (w8 ? at::kChar : at::kHalf) ||
                weight->device().type() != c10::DeviceType::PrivateUse1 ||
                !weight->is_contiguous() || (!w8 && !linear_acc32_ && weight->storage_offset() != 0) ||
                weight->dim() != 2 ||
                weight->size(0) != 6144 || weight->size(1) != 2048) return;
        }
        for (const auto* scale : {&weights.gate_ws, &weights.up_ws}) {
            if (w8) {
                if (!scale->defined() || scale->scalar_type() != at::kHalf ||
                    scale->device().type() != c10::DeviceType::PrivateUse1 ||
                    !scale->is_contiguous() || scale->dim() != 1 ||
                    scale->numel() != 6144) return;
            } else if (scale->defined() && scale->numel() != 0) return;
        }
    }
    text_gateup_pair_w8_ = w8;
    text_gateup_pair_available_ = true;
    prepare_text_silu_round_mul();
}

bool Qwen3_5Model::use_text_gateup_pair(int64_t rows) const {
    return text_gateup_pair_available_ && !action_mode_ && !z2_bound_ &&
        !get_debug_export() && retained_prefill_graph_ && rows == 256;
}

int64_t Qwen3_5Model::text_gateup_pair_site() const {
    if (text_gateup_pair_w8_) return QWEN35_TEXT_GATEUP_PAIR_W8_SITE;
    return linear_acc32_ ? QWEN35_TEXT_GATEUP_PAIR_SITE
                         : QWEN35_TEXT_GATEUP_PAIR_FP16_ACC16_SITE;
}

std::vector<int64_t> Qwen3_5Model::text_gateup_pair_arguments() const {
    // Version, M/local N/K, TP, ACC bits, parent M/N/K, blocks per projection,
    // total grid, input/output row bytes, independent weight owners and layers.
    if (text_gateup_pair_w8_)
        return {2,256,768,2048,8,linear_acc32_ ? 32 : 16,linear_acc32_ ? 288 : 400,80,128,10,20,4096,1536,4,24,8,6144};
    return {1,256,768,2048,8,linear_acc32_ ? 32 : 16,
            linear_acc32_ ? 352 : 432,linear_acc32_ ? 64 : 80,128,
            linear_acc32_ ? 12 : 10,linear_acc32_ ? 24 : 20,4096,1536,2,24};
}

void Qwen3_5Model::launch_text_gateup_pair(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_text_gateup_pair(chunk.len) && layer >= 0 && layer < num_layers() &&
                    ctx().has_complete_physical_manifest(),
                "G05 Text Gate/Up requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::LINEAR, text_gateup_pair_w8_ ? QWEN35_TEXT_GATEUP_PAIR_W8_SITE
            : (linear_acc32_ ? QWEN35_TEXT_GATEUP_PAIR_SITE
                             : QWEN35_TEXT_GATEUP_PAIR_FP16_ACC16_SITE), 1, 0,
        text_gateup_pair_arguments(), chunk.idx);
    const auto& weights = layer_weights_[layer];
    if (text_gateup_pair_w8_) {
        rpu_launch_g05_text_gate_up_linear_w8a16_acc32_kernel(
            addr(0, "residual1"), weights.gate_w, weights.up_w,
            addr(0, "gate"), addr(0, "up"), weights.gate_ws, weights.up_ws, linear_acc32_);
    } else if (linear_acc32_) {
        rpu_launch_g05_text_gate_up_linear_fp16_acc32_kernel(
            addr(0, "residual1"), weights.gate_w, weights.up_w,
            addr(0, "gate"), addr(0, "up"));
    } else {
        rpu_launch_g05_text_gate_up_linear_fp16_acc16_kernel(
            addr(0, "residual1"), weights.gate_w, weights.up_w,
            addr(0, "gate"), addr(0, "up"));
    }
}

void Qwen3_5Model::prepare_text_silu_round_mul() {
    text_silu_round_mul_available_ = false;
    // Recomputed after the pair at each cold setter/install. Retained/debug
    // policy is checked by the shared planning/runtime predicate below.
    if (high_precision_ || !text_gateup_pair_available_ || action_mode_ ||
        !fast_replay_enabled_ || !use_silu_ || !has_gdn_ ||
        num_layers() != 24 || hidden_size() != 2048 || intermediate_size() != 6144 ||
        num_cores() != 8 || attn_tp() != 8 || mlp_tp() != 8 || gdn_tp() != 8 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 ||
        layer_weights_.size() != 24 || layer_is_full_.size() != 24) return;
    if (!text_gateup_pair_w8_) {
        if (!has_text_fp16_projection_owners()) return;
        // Both selected-mode Gate/Up projections have retired FP16 results.
        // This payload preserves the original BASE SiLU FP16 rounding and MUL.
        text_silu_round_mul_available_=KernelCache::instance().has_loaded(
            KernelId::G05_SILU_ROUND_MUL_M32N512);
        return;
    }
    for (size_t layer=0;layer<24;++layer) {
        if (bool(layer_is_full_[layer]) != ((layer+1)%4==0)) return;
        const auto& lw=layer_weights_[layer];
        const std::array<const at::Tensor*,3> weights{&lw.gate_w,&lw.up_w,&lw.down_w};
        const std::array<const at::Tensor*,3> scales{&lw.gate_ws,&lw.up_ws,&lw.down_ws};
        constexpr int64_t n[]={6144,6144,2048},k[]={2048,2048,6144};
        for (size_t family=0;family<3;++family) {
            const auto& w=*weights[family];const auto& scale=*scales[family];
            if (!w.defined() || w.scalar_type()!=at::kChar ||
                w.device().type()!=c10::DeviceType::PrivateUse1 ||
                w.device()!=layer_weights_.front().gate_w.device() ||
                !w.is_contiguous() || w.storage_offset()!=0 || w.dim()!=2 ||
                w.size(0)!=n[family] || w.size(1)!=k[family] ||
                !scale.defined() || scale.scalar_type()!=at::kHalf ||
                scale.device()!=w.device() || !scale.is_contiguous() ||
                scale.storage_offset()!=0 || scale.dim()!=1 || scale.numel()!=n[family]) return;
        }
    }
    text_silu_round_mul_available_=KernelCache::instance().has_loaded(
        KernelId::G05_SILU_ROUND_MUL_M32N512);
}

bool Qwen3_5Model::use_text_silu_round_mul(
    const FmbThreeStageChunkPlan& plan,int64_t physical_len,int64_t logical_len,
    int64_t position,int64_t batch) const {
    if (!text_silu_round_mul_available_ ||
        !fast_replay_enabled_ || !use_silu_ || !use_text_gateup_pair(physical_len) ||
        physical_len!=256 || !(logical_len == 245 || (logical_len >= 193 && logical_len <= 256 && logical_len % 16 == 1)) || position!=0 || batch!=1 ||
        plan.qkv.chunks.size()!=1 || plan.compute.chunks.size()!=1) return false;
    const auto& q=plan.qkv.chunks.front();const auto& c=plan.compute.chunks.front();
    return q.idx==0 && q.offset==0 && q.len==256 && c.idx==0 && c.offset==0 && c.len==256;
}

bool Qwen3_5Model::use_text_silu_round_mul(int64_t rows) {
    return ctx().has_complete_physical_manifest() && rows==ctx().seq_len &&
        use_text_silu_round_mul(ctx().stage_plan,ctx().seq_len,
            ctx().physical_manifest().logical_length,ctx().position,ctx().batch_size);
}

std::vector<int64_t> Qwen3_5Model::text_silu_round_mul_arguments(int64_t logical_len) const {
    TORCH_CHECK((logical_len == 245 || (logical_len >= 193 && logical_len <= 256 && logical_len % 16 == 1)),"G05 Text fused route requires P245 or single E256 P%16=1 in [193,256]");
    // Version,M,H,I,localN,TP,ACC,elements,normal,last,grid,stride,BASE op,
    // full R bound,Gate/Up spans,product aliases R,dead Gate,24 layers,P245,
    // completed pair/cold capability and shared nondebug/retained policy.
    return {2,256,2048,6144,768,8,linear_acc32_ ? 32 : 16,196608,25600,17408,8,51200,134,
            1048576,393216,1,1,24,logical_len,text_gateup_pair_available_?1:0,
            text_silu_round_mul_available_?1:0,get_debug_export()?1:0,
            retained_prefill_graph_?1:0,193,256,16,1,245};
}

void Qwen3_5Model::launch_text_silu_round_mul(int layer,const ChunkInfo& chunk) {
    TORCH_CHECK(use_text_silu_round_mul(chunk.len) && layer>=0 && layer<24 &&
        chunk.idx==0 && chunk.offset==0 && chunk.len==256,
        "G05 Text rounded SiLU/MUL escaped its exact COMPLETE paired profile");
    ctx().consume_physical_route(FmbRouteFamily::ACTIVATION,QWEN35_TEXT_SILU_ROUND_MUL_SITE,
        1,0,text_silu_round_mul_arguments(ctx().physical_manifest().logical_length),chunk.idx);
    rpu_launch_g05_text_silu_round_mul_spm_kernel(
        addr(0,"gate"),addr(0,"up"),addr(0,"residual1"));
}

void Qwen3_5Model::prepare_action_attn_sigmoid_mul() {
    action_attn_sigmoid_mul_available_ = false;
    // Installed NVFP4 tables already prove the mixed Q/O/MLP and W8 K/V/gate
    // owners. This optional elementwise schedule does not replace any owner.
    if (!use_action_nvfp4(action_len_) || has_gdn_ || num_layers() != 24 ||
        hidden_size() != 1024 || intermediate_size() != 4096 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 ||
        attn_tp() != 8 || mlp_tp() != 8 || layer_is_full_.size() != 24 ||
        std::any_of(layer_is_full_.begin(), layer_is_full_.end(),
                    [](uint8_t full) { return !full; })) return;
    action_attn_sigmoid_mul_available_ = KernelCache::instance().has_loaded(
        KernelId::G05_ATTN_SIGMOID_ROUND_MUL_M32D256);
}

bool Qwen3_5Model::use_action_attn_sigmoid_mul(int64_t rows) const {
    return action_attn_sigmoid_mul_available_ && use_action_nvfp4(rows) &&
        action_step_active_ && action_loop_active_ && action_num_steps_ == 10 &&
        !get_debug_export();
}

std::vector<int64_t> Qwen3_5Model::action_attn_sigmoid_mul_arguments() const {
    // Version,M,D,TP,elements,block/grid/stride,full roots,original modes,
    // absolute-SPM GP bases,unit FP16 scales,zero clips,retained outputs,L/steps.
    return {1,32,256,8,8192,1024,8,2048,16384,19,0,0x2692,0x0492,
            4,6,12,0x3c00,0,1,2,24,10,
            action_attn_sigmoid_mul_available_ ? 1 : 0};
}

void Qwen3_5Model::launch_action_attn_sigmoid_mul(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_attn_sigmoid_mul(chunk.len) &&
                    ctx().has_complete_physical_manifest() &&
                    layer >= 0 && layer < 24 && layer_is_full_[layer] &&
                    chunk.idx == 0 && chunk.offset == 0 && chunk.len == 32,
                "G05 attention Sigmoid/Mul requires its exact NVFP4 COMPLETE route");
    ctx().consume_physical_route(FmbRouteFamily::ACTIVATION,
        QWEN35_ACTION_ATTN_SIGMOID_MUL_SITE,1,0,
        action_attn_sigmoid_mul_arguments(),chunk.idx);
    rpu_launch_g05_attn_sigmoid_round_mul_spm_kernel(
        addr(0,"attn_gate"),addr(0,"output"));
}

void Qwen3_5Model::prepare_action_nvfp4_swiglu() {
    // Exact pair admission already proves every original mixed-NV Gate/Up owner.
    // Keep old pair + rounded SiLU/Mul available as the cold REF fallback.
    action_nvfp4_swiglu_available_ = action_gateup_pair_nvfp4_ &&
        action_gateup_pair_available_ && action_silu_round_mul_available_ &&
        use_action_nvfp4(action_len_) &&
        KernelCache::instance().has_loaded(
            linear_acc32_ ? KernelId::G05_GATE_UP_SWIGLU_NVFP4_ACC32_M32N512K1024_N128
                          : KernelId::G05_GATE_UP_SWIGLU_NVFP4_ACC16_M32N512K1024_N128);
}

bool Qwen3_5Model::use_action_nvfp4_swiglu(int64_t rows) const {
    return action_nvfp4_swiglu_available_ && action_gateup_pair_nvfp4_ &&
        use_action_nvfp4(rows) && use_action_silu_round_mul(rows);
}

int64_t Qwen3_5Model::action_nvfp4_swiglu_site() const {
    return linear_acc32_ ? QWEN35_ACTION_GATEUP_SWIGLU_NVFP4_ACC32_SITE
                         : QWEN35_ACTION_GATEUP_SWIGLU_NVFP4_SITE;
}

std::vector<int64_t> Qwen3_5Model::action_nvfp4_swiglu_arguments() const {
    // Exact pair geometry, four DDR owners/five SPM roots, BASE SiLU/Mul modes,
    // eight-warp barrier/four epilogue warps, zeroed core-counter pairs, Up not stored.
    return {1,32,512,1024,8,linear_acc32_ ? 32 : 16,176,128,128,4,8,2048,1024,4,24,
            96,256,2,20,32,34,24,36,4,38,134,0x3e92,0x0492,
            1,8,4,128,130,0,0,0,action_nvfp4_swiglu_available_ ? 1 : 0};
}

void Qwen3_5Model::launch_action_nvfp4_swiglu(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_nvfp4_swiglu(chunk.len) && layer >= 0 && layer < 24 &&
                    ctx().has_complete_physical_manifest() &&
                    !use_action_gateup_resident(chunk.len),
                "G05 NVFP4 SwiGLU requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,
        linear_acc32_ ? QWEN35_ACTION_GATEUP_SWIGLU_NVFP4_ACC32_SITE
                     : QWEN35_ACTION_GATEUP_SWIGLU_NVFP4_SITE,
        static_cast<int64_t>(linear_acc32_
            ? FmbLinearRouteSelector::G05_NVFP4_SWIGLU_ACC32
            : FmbLinearRouteSelector::G05_NVFP4_SWIGLU_ACC16),0,
        action_nvfp4_swiglu_arguments(),chunk.idx);
    const auto& lw = layer_weights_[layer];
    if (linear_acc32_) {
        rpu_launch_g05_gate_up_swiglu_nvfp4_acc32_kernel(
            addr(0,"residual1"),lw.gate_w,lw.up_w,addr(0,"gate"),addr(0,"up"),
            lw.gate_ws,lw.up_ws,addr(0,G05Nvfp4Tables::names[2]),
            addr(0,G05Nvfp4Tables::names[3]),static_cast<uint16_t>(layer));
    } else {
        rpu_launch_g05_gate_up_swiglu_nvfp4_acc16_kernel(
            addr(0,"residual1"),lw.gate_w,lw.up_w,addr(0,"gate"),addr(0,"up"),
            lw.gate_ws,lw.up_ws,addr(0,G05Nvfp4Tables::names[2]),
            addr(0,G05Nvfp4Tables::names[3]),static_cast<uint16_t>(layer));
    }
}

void Qwen3_5Model::prepare_action_silu_round_mul() {
    // Pair admission already proved the exact profile and every Gate/Up owner.
    action_silu_round_mul_available_ = !high_precision_ && action_gateup_pair_available_ &&
        KernelCache::instance().has_loaded(KernelId::G05_SILU_ROUND_MUL_M32N512);
}

bool Qwen3_5Model::use_action_silu_round_mul(int64_t rows) const {
    return action_silu_round_mul_available_ && use_silu_ && use_action_gateup_pair(rows);
}

std::vector<int64_t> Qwen3_5Model::action_silu_round_mul_arguments() const {
    // Version, rows/local width, TP, elements, block elements, grid, BASE SiLU,
    // resident layer count. Resident layers keep their original SiLU/MUL order.
    return {1,32,512,8,16384,2048,8,134,
            use_action_gateup_resident(32) ? action_gateup_resident_layers() : 0};
}

void Qwen3_5Model::launch_action_silu_round_mul(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_silu_round_mul(chunk.len) &&
                    layer >= 0 && layer < num_layers() &&
                    ctx().has_complete_physical_manifest() &&
                    !(layer >= 1 && layer <= action_gateup_resident_layers() &&
                      use_action_gateup_resident(chunk.len)),
                "G05 rounded SiLU/Mul requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::ACTIVATION,QWEN35_ACTION_SILU_ROUND_MUL_SITE,1,0,
        action_silu_round_mul_arguments(),chunk.idx);
    rpu_launch_g05_silu_round_mul_spm_kernel(addr(0,"gate"),addr(0,"up"));
}

void Qwen3_5Model::prepare_action_silu_high_mul() {
    // The existing paired projection admission proves the exact G05 Action
    // geometry and independent FP16/W8/NVFP4 weight and scale owners. Sample
    // the optional HIGH payload once, before planning or Graph capture.
    action_silu_high_mul_available_ = high_precision_ && action_mode_ &&
        use_silu_ && !has_gdn_ && action_gateup_pair_available_ &&
        KernelCache::instance().has_loaded(KernelId::RHINOVLA_SILU_HIGH_MUL);
}

bool Qwen3_5Model::use_action_silu_high_mul(int64_t rows) const {
    return high_precision_ && action_silu_high_mul_available_ && use_silu_ &&
        use_action_gateup_pair(rows);
}

std::vector<int64_t> Qwen3_5Model::action_silu_high_mul_arguments() const {
    // Version, M/H/local-I/TP, HIGH parent block/grid/tail, HIGH/custom SiLU,
    // parent input/final SiLU modes, ordinary MUL, both FP16 retirements,
    // independent Linear accumulation, layer count and resident exclusions.
    return {1,32,1024,512,8,16384,15872,2,512,1,134,0x3e92,0x3c92,0x0492,
            2,2,linear_acc32_ ? 32 : 16,24,
            use_action_gateup_resident(32) ? action_gateup_resident_layers() : 0,
            action_silu_high_mul_available_ ? 1 : 0};
}

void Qwen3_5Model::launch_action_silu_high_mul(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_silu_high_mul(chunk.len) &&
                    layer >= 0 && layer < num_layers() &&
                    ctx().has_complete_physical_manifest() &&
                    !(layer >= 1 && layer <= action_gateup_resident_layers() &&
                      use_action_gateup_resident(chunk.len)),
                "G05 HIGH SiLU/Mul requires its exact nonresident COMPLETE descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::ACTIVATION,QWEN35_ACTION_SILU_HIGH_MUL_SITE,1,0,
        action_silu_high_mul_arguments(),chunk.idx);
    // HIGH SiLU retires to FP16 before the unchanged FP16 MUL. The shared
    // payload preserves both boundaries and the 15872-element parent tail.
    rpu_launch_rhinovla_silu_high_mul_spm_kernel(
        addr(0,"gate"),addr(0,"up"),addr(0,"gate"),16384,mlp_tp());
}

void Qwen3_5Model::prepare_action_gateup_pair() {
    action_gateup_pair_available_ = false;
    action_gateup_pair_w8_ = false;
    action_gateup_pair_nvfp4_ = false;
    if (!action_mode_ || !use_silu_ || has_gdn_ || action_len_ != 32 ||
        num_layers() != 24 || hidden_size() != 1024 || intermediate_size() != 4096 ||
        num_cores() != 8 || attn_tp() != 8 || mlp_tp() != 8 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 ||
        layer_weights_.size() != 24 || layer_is_full_.size() != 24 ||
        !std::all_of(layer_is_full_.begin(), layer_is_full_.end(),
                     [](uint8_t full) { return full != 0; })) return;
    if (action_nvfp4_.enabled()) {
        if (!KernelCache::instance().has_loaded(
                linear_acc32_ ? KernelId::G05_GATE_UP_PAIRED_NVFP4_ACC32_M32N512K1024_N128
                              : KernelId::G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128)) return;
        // The installed five-family tables and full Action profile were proved
        // in set_weights; retain independent owners and recheck pair geometry.
        for (const auto& lw : layer_weights_) {
            for (const auto* weight : {&lw.gate_w, &lw.up_w}) {
                if (!weight->defined() || weight->scalar_type() != at::kByte ||
                    weight->device().type() != c10::DeviceType::PrivateUse1 ||
                    !weight->is_contiguous() || weight->storage_offset() != 0 || weight->dim() != 2 ||
                    weight->size(0) != 4096 || weight->size(1) != 512) return;
            }
            for (const auto* scale : {&lw.gate_ws, &lw.up_ws}) {
                if (!scale->defined() || scale->scalar_type() != at::kByte ||
                    scale->device().type() != c10::DeviceType::PrivateUse1 ||
                    !scale->is_contiguous() || scale->storage_offset() != 0 || scale->dim() != 2 ||
                    scale->size(0) != 64 || scale->size(1) != 4096) return;
            }
        }
        action_gateup_pair_nvfp4_ = true;
        action_gateup_pair_available_ = true;
        return;
    }
    const bool w8 = layer_weights_[0].gate_w.defined() &&
        layer_weights_[0].gate_w.scalar_type() == at::kChar;
    if (!KernelCache::instance().has_loaded(w8
            ? (linear_acc32_ ? KernelId::G05_GATE_UP_W8A16_ACC32_M32N512K1024
                             : KernelId::G05_GATE_UP_W8A16_ACC16_M32N512K1024)
            : (linear_acc32_ ? KernelId::G05_GATE_UP_FP16_ACC32_M32N512K1024
                             : KernelId::G05_GATE_UP_FP16_ACC16_M32N512K1024))) return;
    // Independent installed TP8 weight owners; no concatenation or repacking.
    for (const auto& lw : layer_weights_) {
        for (const auto* weight : {&lw.gate_w, &lw.up_w}) {
            if (!weight->defined() || weight->scalar_type() != (w8 ? at::kChar : at::kHalf) ||
                weight->device().type() != c10::DeviceType::PrivateUse1 ||
                !weight->is_contiguous() || (!w8 && !linear_acc32_ && weight->storage_offset() != 0) ||
                weight->dim() != 2 ||
                weight->size(0) != 4096 || weight->size(1) != 1024) return;
        }
    }
    if (w8) {
        for (const auto& lw : layer_weights_) {
            for (const auto* scale : {&lw.gate_ws, &lw.up_ws}) {
                if (!scale->defined() || scale->scalar_type() != at::kHalf ||
                    scale->device().type() != c10::DeviceType::PrivateUse1 ||
                    !scale->is_contiguous() || scale->dim() != 1 || scale->size(0) != 4096) return;
            }
        }
    }
    if (!w8 && !linear_acc32_) {
        for (const auto& lw : layer_weights_)
            for (const auto* scale : {&lw.gate_ws, &lw.up_ws})
                if (scale->defined() && scale->numel() != 0) return;
    }
    action_gateup_pair_w8_ = w8;
    action_gateup_pair_available_ = true;
}

bool Qwen3_5Model::use_action_gateup_pair(int64_t rows) const {
    return action_gateup_pair_available_ && use_action_mod_body() && rows == 32 &&
        (!action_gateup_pair_nvfp4_ || use_action_nvfp4(rows));
}

int64_t Qwen3_5Model::action_gateup_pair_site() const {
    if (action_gateup_pair_nvfp4_)
        return linear_acc32_ ? QWEN35_ACTION_GATEUP_PAIR_NVFP4_ACC32_SITE
                             : QWEN35_ACTION_GATEUP_PAIR_NVFP4_SITE;
    if (action_gateup_pair_w8_) return QWEN35_ACTION_GATEUP_PAIR_W8_SITE;
    return linear_acc32_ ? QWEN35_ACTION_GATEUP_PAIR_SITE
                         : QWEN35_ACTION_GATEUP_PAIR_FP16_ACC16_SITE;
}

std::vector<int64_t> Qwen3_5Model::action_gateup_pair_arguments() const {
    // Version, M/local N/K, TP, ACC bits, parent tile, blocks per projection,
    // combined grid, byte row strides, Gate/Up DDR pairs, resident layer count.
    if (action_gateup_pair_nvfp4_)
        return {3,32,512,1024,8,linear_acc32_ ? 32 : 16,176,128,128,4,8,2048,1024,4,24,
                96,256,20,32,34,24,36,4,38,1,action_gateup_pair_nvfp4_ ? 1 : 0};
    if (action_gateup_pair_w8_) return {1,32,512,1024,8,linear_acc32_ ? 32 : 16,linear_acc32_ ? 240 : 352,96,128,6,12,2048,1024,2,24,
        use_action_gateup_resident(32) ? action_gateup_resident_layers() : 0,20,28,1};
    return {1,32,512,1024,8,linear_acc32_ ? 32 : 16,linear_acc32_ ? 304 : 432,80,128,7,14,2048,1024,2,24,
            use_action_gateup_resident(32) ? action_gateup_resident_layers() : 0};
}

void Qwen3_5Model::launch_action_gateup_pair(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_gateup_pair(chunk.len) && layer >= 0 && layer < num_layers() &&
                    ctx().has_complete_physical_manifest() &&
                    !(layer >= 1 && layer <= action_gateup_resident_layers() &&
                      use_action_gateup_resident(chunk.len)),
                "G05 Gate/Up requires its exact COMPLETE descriptor");
    if (action_gateup_pair_nvfp4_) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, linear_acc32_ ? QWEN35_ACTION_GATEUP_PAIR_NVFP4_ACC32_SITE
                         : QWEN35_ACTION_GATEUP_PAIR_NVFP4_SITE,
            static_cast<int64_t>(linear_acc32_
                ? FmbLinearRouteSelector::G05_NVFP4_PAIRED_ACC32
                : FmbLinearRouteSelector::G05_NVFP4_PAIRED_ACC16), 0,
            action_gateup_pair_arguments(), chunk.idx);
        const auto& lw = layer_weights_[layer];
        if (linear_acc32_) {
            rpu_launch_g05_gate_up_paired_nvfp4_acc32_kernel(
                addr(0,"residual1"), lw.gate_w, lw.up_w, addr(0,"gate"), addr(0,"up"),
                lw.gate_ws, lw.up_ws, addr(0,G05Nvfp4Tables::names[2]),
                addr(0,G05Nvfp4Tables::names[3]), static_cast<uint16_t>(layer));
        } else {
            rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel(
                addr(0,"residual1"), lw.gate_w, lw.up_w, addr(0,"gate"), addr(0,"up"),
                lw.gate_ws, lw.up_ws, addr(0,G05Nvfp4Tables::names[2]),
                addr(0,G05Nvfp4Tables::names[3]), static_cast<uint16_t>(layer));
        }
        return;
    }
    ctx().consume_physical_route(
        FmbRouteFamily::LINEAR,action_gateup_pair_w8_ ? QWEN35_ACTION_GATEUP_PAIR_W8_SITE
            : (linear_acc32_ ? QWEN35_ACTION_GATEUP_PAIR_SITE
                             : QWEN35_ACTION_GATEUP_PAIR_FP16_ACC16_SITE),1,0,
        action_gateup_pair_arguments(),chunk.idx);
    const auto& lw = layer_weights_[layer];
    if (action_gateup_pair_w8_) {
        rpu_launch_g05_gate_up_linear_w8a16_acc32_kernel(addr(0,"residual1"),lw.gate_w,lw.up_w,lw.gate_ws,lw.up_ws,addr(0,"gate"),addr(0,"up"),linear_acc32_);
    } else if (linear_acc32_) {
        rpu_launch_g05_gate_up_linear_fp16_acc32_kernel(
            addr(0,"residual1"),lw.gate_w,lw.up_w,addr(0,"gate"),addr(0,"up"));
    } else {
        rpu_launch_g05_gate_up_linear_fp16_acc16_kernel(
            addr(0,"residual1"),lw.gate_w,lw.up_w,addr(0,"gate"),addr(0,"up"));
    }
}

void Qwen3_5Model::prepare_action_q_gate() {
    action_q_gate_available_ = false;
    action_q_gate_w8_ = false;
    if (!action_mode_ || has_gdn_ || action_len_ != 32 ||
        num_layers() != 24 || hidden_size() != 1024 || intermediate_size() != 4096 ||
        num_cores() != 8 || attn_tp() != 8 || mlp_tp() != 8 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 ||
        layer_weights_.size() != 24 || layer_is_full_.size() != 24 ||
        !std::all_of(layer_is_full_.begin(), layer_is_full_.end(),
                     [](uint8_t full) { return full != 0; })) return;
    const bool w8 = layer_weights_[0].q_w.defined() &&
        layer_weights_[0].q_w.scalar_type() == at::kChar;
    if (!linear_acc32_ && !w8) return;
    if (!KernelCache::instance().has_loaded(w8
            ? (linear_acc32_ ? KernelId::G05_Q_GATE_W8A16_ACC32_M32N256K1024
                             : KernelId::G05_Q_GATE_W8A16_ACC16_M32N256K1024)
            : KernelId::G05_Q_GATE_FP16_ACC32_M32N256K1024)) return;
    // Independent installed TP8 weight owners; no concatenation or repacking.
    for (const auto& lw : layer_weights_) {
        for (const auto* weight : {&lw.q_w, &lw.attn_gate_w}) {
            if (!weight->defined() || weight->scalar_type() != (w8 ? at::kChar : at::kHalf) ||
                weight->device().type() != c10::DeviceType::PrivateUse1 ||
                !weight->is_contiguous() || weight->dim() != 2 ||
                weight->size(0) != 2048 || weight->size(1) != 1024) return;
        }
    }
    if (w8) {
        for (const auto& lw : layer_weights_) {
            for (const auto* scale : {&lw.q_ws, &lw.attn_gate_ws}) {
                if (!scale->defined() || scale->scalar_type() != at::kHalf ||
                    scale->device().type() != c10::DeviceType::PrivateUse1 ||
                    !scale->is_contiguous() || scale->dim() != 1 || scale->size(0) != 2048) return;
            }
        }
    }
    action_q_gate_w8_ = w8;
    action_q_gate_available_ = true;
}

bool Qwen3_5Model::use_action_q_gate(int64_t rows) const {
    return action_q_gate_available_ && use_action_mod_body() && rows == 32;
}

std::vector<int64_t> Qwen3_5Model::action_q_gate_arguments() const {
    // Version, M/local N/K, TP, ACC bits, parent tile, blocks per projection,
    // combined grid, input/output row strides, Q/gate weight register pairs.
    if (action_q_gate_w8_) return {1,32,256,1024,8,linear_acc32_ ? 32 : 16,linear_acc32_ ? 192 : 288,128,128,2,4,2048,512,2,24,20,28,1};
    return {1,32,256,1024,8,32,240,112,128,3,6,2048,512,2,24};
}

void Qwen3_5Model::launch_action_q_gate(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_q_gate(chunk.len) && layer >= 0 && layer < num_layers() &&
                    ctx().has_complete_physical_manifest(),
                "G05 Q/gate requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::LINEAR,action_q_gate_w8_ ? QWEN35_ACTION_Q_GATE_W8_SITE : QWEN35_ACTION_Q_GATE_SITE,1,0,
        action_q_gate_arguments(),chunk.idx);
    const auto& lw = layer_weights_[layer];
    if (action_q_gate_w8_) {
        rpu_launch_g05_q_gate_linear_w8a16_acc32_kernel(addr(0,"input_norm"),lw.q_w,lw.attn_gate_w,lw.q_ws,lw.attn_gate_ws,addr(0,"q"),addr(0,"attn_gate"),linear_acc32_);
    } else {
    rpu_launch_g05_q_gate_linear_fp16_acc32_kernel(
        addr(0,"input_norm"),lw.q_w,lw.attn_gate_w,addr(0,"q"),addr(0,"attn_gate"));
    }
}

void Qwen3_5Model::prepare_action_kv_owner() {
    action_kv_owner_available_ = false;
    action_kv_owner_w8_ = false;
    if (!action_mode_ || has_gdn_ || action_len_ != 32 ||
        num_layers() != 24 || hidden_size() != 1024 || intermediate_size() != 4096 ||
        num_cores() != 8 || attn_tp() != 8 || mlp_tp() != 8 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 ||
        layer_weights_.size() != 24 || layer_is_full_.size() != 24 ||
        !std::all_of(layer_is_full_.begin(), layer_is_full_.end(),
                     [](uint8_t full) { return full != 0; })) return;
    const bool w8 = layer_weights_[0].k_w.defined() &&
        layer_weights_[0].k_w.scalar_type() == at::kChar;
    if (!linear_acc32_ && !w8) return;
    if (!KernelCache::instance().has_loaded(w8
            ? (linear_acc32_ ? KernelId::G05_KV_OWNER_W8A16_ACC32_M32N256K1024
                             : KernelId::G05_KV_OWNER_W8A16_ACC16_M32N256K1024)
            : KernelId::G05_KV_OWNER_FP16_ACC32_M32N256K1024) ||
        !KernelCache::instance().has_loaded(KernelId::G05_KV_OWNER_FANOUT_M32D256)) return;
    // Metadata only exposes the expanded KV8 width. Prove the actual immutable
    // installed bytes before replacing four copies by one owner computation.
    for (const auto& lw : layer_weights_) {
        if (w8) {
            if (!rpu_g05_kv_weight_is_replicated_w8a16(lw.k_w, lw.k_ws) ||
                !rpu_g05_kv_weight_is_replicated_w8a16(lw.v_w, lw.v_ws)) return;
        } else if (!rpu_g05_kv_weight_is_replicated_fp16(lw.k_w) ||
                   !rpu_g05_kv_weight_is_replicated_fp16(lw.v_w)) return;
    }
    action_kv_owner_w8_ = w8;
    action_kv_owner_available_ = true;
}

bool Qwen3_5Model::use_action_kv_owner(int64_t rows) const {
    return action_kv_owner_available_ && use_action_mod_body() && rows == 32;
}

std::vector<int64_t> Qwen3_5Model::action_kv_owner_arguments() const {
    // Version, M/K, logical/physical KV heads, D, TP, tile M/N/K,
    // K owners, V owners, fanout extent. All values are nonnegative.
    if (action_kv_owner_w8_) return {1,32,1024,2,8,256,8,linear_acc32_ ? 192 : 288,128,128,0,4,2,6,16384,20,28,1};
    return {1,32,1024,2,8,256,8,240,112,128,0,4,2,6,16384};
}

void Qwen3_5Model::launch_action_kv_owner(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_kv_owner(chunk.len) && layer >= 0 && layer < num_layers() &&
                    ctx().has_complete_physical_manifest(),
                "G05 KV owner requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(
        FmbRouteFamily::LINEAR,action_kv_owner_w8_ ? QWEN35_ACTION_KV_OWNER_W8_SITE : QWEN35_ACTION_KV_OWNER_SITE,1,0,
        action_kv_owner_arguments(),chunk.idx);
    const auto& lw = layer_weights_[layer];
    if (action_kv_owner_w8_) {
        rpu_launch_g05_kv_owner_linear_w8a16_acc32_kernel(addr(0,"input_norm"),lw.k_w,lw.v_w,lw.k_ws,lw.v_ws,addr(0,"k"),addr(0,"v"),linear_acc32_);
    } else {
    rpu_launch_g05_kv_owner_linear_fp16_acc32_kernel(
        addr(0,"input_norm"),lw.k_w,lw.v_w,addr(0,"k"),addr(0,"v"));
    }
}

void Qwen3_5Model::prepare_action_qkv_gate() {
    // Reuse the all-layer Q/gate metadata and immutable KV replica proofs.
    // Only the optional W8 combined schedule changes; old REFs keep both routes.
    action_qkv_gate_available_ = action_q_gate_available_ && action_q_gate_w8_ &&
        action_kv_owner_available_ && action_kv_owner_w8_ &&
        KernelCache::instance().has_loaded(linear_acc32_
            ? KernelId::G05_QKV_GATE_OWNER_W8A16_ACC32_M32N256K1024
            : KernelId::G05_QKV_GATE_OWNER_W8A16_ACC16_M32N256K1024) &&
        KernelCache::instance().has_loaded(KernelId::G05_KV_OWNER_FANOUT_M32D256);
}

bool Qwen3_5Model::use_action_qkv_gate(int64_t rows) const {
    return action_qkv_gate_available_ && use_action_q_gate(rows) && use_action_kv_owner(rows);
}

std::vector<int64_t> Qwen3_5Model::action_qkv_gate_arguments() const {
    // Version, M/local N/K, TP/ACC, parent tile, blocks/grid, row bytes,
    // Q/gate/K/V weight and scale GP bases, replicas/logical KV, owner cores,
    // fanout bytes and W8 dtype mode. Four independent output roots stay live.
    return {1,32,256,1024,8,linear_acc32_ ? 32 : 16,linear_acc32_ ? 192 : 288,128,128,2,6,2048,512,
            2,24,30,36,20,28,34,40,4,2,0,4,2,6,16384,1};
}

void Qwen3_5Model::launch_action_qkv_gate(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_qkv_gate(chunk.len) && layer >= 0 && layer < num_layers() &&
                    ctx().has_complete_physical_manifest(),
                "G05 QKV/gate requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,
        QWEN35_ACTION_QKV_GATE_W8_SITE,1,0,action_qkv_gate_arguments(),chunk.idx);
    const auto& lw = layer_weights_[layer];
    rpu_launch_g05_qkv_gate_owner_linear_w8a16_acc32_kernel(
        addr(0,"input_norm"),lw.q_w,lw.attn_gate_w,lw.k_w,lw.v_w,
        lw.q_ws,lw.attn_gate_ws,lw.k_ws,lw.v_ws,
        addr(0,"q"),addr(0,"attn_gate"),addr(0,"k"),addr(0,"v"),linear_acc32_);
}

void Qwen3_5Model::prepare_action_nvfp4_qk_norm_rope() {
    action_nvfp4_qk_norm_rope_available_=false;
    if (!use_action_nvfp4(action_len_) || !action_owner_cache_available_ ||
        !has_qk_norm_ || !has_mrope_ || rotary_dim_!=64 || eps_!=1e-6 ||
        num_q_heads()!=8 || num_kv_heads()!=8 || head_dim()!=256 || attn_tp()!=8 ||
        layer_weights_.size()!=24) return;
    for (auto id:{KernelId::G05_Q_NORM_PARTIAL_ROPE_M32D256R64_GROUP4,
                 KernelId::G05_K_OWNER_NORM_PARTIAL_ROPE_M32D256R64_GROUP4})
        if (!KernelCache::instance().has_loaded(id)) return;
    for (const auto& lw:layer_weights_)
        for (const auto* norm:{&lw.q_norm_w,&lw.k_norm_w})
            if (!norm->defined() || norm->device().type()!=c10::DeviceType::PrivateUse1 ||
                norm->scalar_type()!=at::kHalf || !norm->is_contiguous() ||
                norm->dim()!=1 || norm->size(0)!=256) return;
    action_nvfp4_qk_norm_rope_available_=true;
}

bool Qwen3_5Model::use_action_nvfp4_qk_norm_rope(int64_t rows,int64_t prefix) const {
    return action_nvfp4_qk_norm_rope_available_ && use_action_nvfp4(rows) &&
        use_action_owner_cache(rows,prefix);
}

std::vector<int64_t> Qwen3_5Model::action_nvfp4_qk_norm_rope_arguments() const {
    // Q/K remain separate kernels and semantic frequency owners. Position and
    // table values are live operands, validated independently by the launcher.
    return {1,action_nvfp4_qk_norm_rope_available_?1:0,32,256,64,8,4,8,32,
            1,17,0x1c00,512,16384,0,4,16,18,20,22,23,24,25,11,11,10,24,2};
}

void Qwen3_5Model::launch_action_nvfp4_qk_norm_rope(
    int layer,const ChunkInfo& chunk,const at::Tensor& query_cos,
    const at::Tensor& query_sin,int64_t query_position) {
    TORCH_CHECK(use_action_nvfp4_qk_norm_rope(chunk.len,ctx().position) &&
                    chunk.idx==0 && chunk.offset==0 && layer>=0 && layer<24 &&
                    ctx().has_complete_physical_manifest(),
                "G05 group4 Q/K norm/RoPE requires exact COMPLETE owner-cache route");
    ctx().consume_physical_route(FmbRouteFamily::ROPE,
        QWEN35_ACTION_QK_NORM_ROPE_GROUP4_SITE,1,0,
        action_nvfp4_qk_norm_rope_arguments(),chunk.idx);
    rpu_launch_g05_qk_norm_rope_group4_spm_kernel(
        addr(0,"q"),addr(0,"k"),layer_addr(layer,0,"q_norm_w"),layer_addr(layer,0,"k_norm_w"),
        query_cos,query_sin,query_position,prefill_cos_,prefill_sin_,chunk.offset);
}

void Qwen3_5Model::prepare_action_gate_kv_owner() {
    action_gate_kv_owner_available_=false;
    // This joint producer has ACC16 arithmetic; ACC32 keeps owner K/V plus gate.
    if (linear_acc32_ || !use_action_nvfp4(action_len_) || !action_owner_cache_available_ ||
        !action_kv_owner_w8_ || layer_weights_.size()!=24 ||
        !KernelCache::instance().has_loaded(
            KernelId::G05_GATE_KV_OWNER_W8A16_ACC16_M32N256K1024_N128)) return;
    // Reuse immutable K/V replica contents, but independently prove the new
    // launcher's full six-owner ABI. Older owner routes also accept aligned
    // views and read aliases, which must retain their original safe fallback.
    for (const auto& lw:layer_weights_) {
        const auto& gate=lw.attn_gate_w;
        for (const auto* w:{&gate,&lw.k_w,&lw.v_w}) {
            if (!w->defined() || w->scalar_type()!=at::kChar ||
                w->device().type()!=c10::DeviceType::PrivateUse1 || w->device()!=gate.device() ||
                !w->is_contiguous() || w->storage_offset()!=0 || w->dim()!=2 ||
                w->size(0)!=2048 || w->size(1)!=1024) return;
        }
        for (const auto* scale:{&lw.attn_gate_ws,&lw.k_ws,&lw.v_ws}) {
            if (!scale->defined() || scale->scalar_type()!=at::kHalf ||
                scale->device()!=gate.device() || !scale->is_contiguous() ||
                scale->storage_offset()!=0 || scale->dim()!=1 || scale->size(0)!=2048) return;
        }
        const std::array<const at::Tensor*,6> owners{{
            &gate,&lw.k_w,&lw.v_w,&lw.attn_gate_ws,&lw.k_ws,&lw.v_ws}};
        std::array<uint64_t,6> addresses{};
        for (size_t i=0;i<owners.size();++i) {
            const uint64_t address=RpuGetDevAddr(owners[i]->data_ptr());
            constexpr uint64_t limit=UINT64_C(1)<<40;
            if (!address || address%256 || address>=limit || owners[i]->nbytes()>limit-address) return;
            for (size_t j=0;j<i;++j)
                if (!(address+owners[i]->nbytes()<=addresses[j] ||
                      addresses[j]+owners[j]->nbytes()<=address)) return;
            addresses[i]=address;
        }
    }
    action_gate_kv_owner_available_=true;
}

bool Qwen3_5Model::use_action_gate_kv_owner(int64_t rows,int64_t prefix) const {
    return action_gate_kv_owner_available_ && use_action_nvfp4(rows) &&
        use_action_owner_cache(rows,prefix);
}

std::vector<int64_t> Qwen3_5Model::action_gate_kv_owner_arguments(int64_t prefix) const {
    TORCH_CHECK(prefix==0 || prefix==245 || prefix==181 || (prefix>=161 && prefix<=449 && prefix%16==1),"G05 gate/KV owner requires bounded P0/P181/P245/P%16=1");
    // Version/cold capability, M/localN/K/globalN, TP/ACC16, parent/grid,
    // independent weight/scale/output bases, replica domain and zero fanout.
    return {3,action_gate_kv_owner_available_?1:0,32,256,1024,2048,8,16,
            288,128,128,2,4,2048,512,2,24,30,20,28,34,4,26,32,
            4,2,0,4,2,6,0,1,24,0,245,prefix,161,449,16,1,181};
}

void Qwen3_5Model::launch_action_gate_kv_owner(int layer,const ChunkInfo& chunk) {
    const int64_t prefix=ctx().position;
    TORCH_CHECK(use_action_gate_kv_owner(chunk.len,prefix) &&
                    chunk.idx==0 && chunk.offset==0 && layer>=0 && layer<24 &&
                    ctx().has_complete_physical_manifest(),
                "G05 gate/KV owner requires its exact COMPLETE owner-cache descriptor");
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,
        QWEN35_ACTION_GATE_KV_OWNER_W8_SITE,1,0,
        action_gate_kv_owner_arguments(prefix),chunk.idx);
    const auto& lw=layer_weights_[layer];
    rpu_launch_g05_gate_kv_owner_only_w8a16_acc16_kernel(
        addr(0,"input_norm"),lw.attn_gate_w,lw.k_w,lw.v_w,
        lw.attn_gate_ws,lw.k_ws,lw.v_ws,addr(0,"attn_gate"),addr(0,"k"),addr(0,"v"));
}

void Qwen3_5Model::prepare_action_owner_cache() {
    action_owner_cache_available_ = false;
    // Reuse the installed W8 QKV metadata and immutable KV replica proofs.
    // Shared K norm parameters are broadcast from one retained owner per layer.
    const bool mixed_owner = use_action_nvfp4(action_len_) &&
        action_kv_owner_available_ && action_kv_owner_w8_;
    if (high_precision_ || (!action_qkv_gate_available_ && !mixed_owner) || !has_qk_norm_ || !has_mrope_ ||
        rotary_dim_ != 64 || eps_ != 1e-6 || layer_weights_.size() != 24) return;
    for (const auto id : {
            KernelId::G05_K_OWNER_RMS_NORM_M32D256,
            KernelId::G05_K_OWNER_PARTIAL_MROPE_M32D256R64,
            KernelId::G05_OWNER_INSERT_KCACHE_MULTIWARP,
            KernelId::G05_OWNER_INSERT_KCACHE_MULTIWARP_V16,
            KernelId::G05_OWNER_INSERT_VCACHE_MULTIWARP,
            KernelId::G05_OWNER_INSERT_VCACHE_MULTIWARP_V16}) {
        if (!KernelCache::instance().has_loaded(id)) return;
    }
    for (const auto& lw : layer_weights_) {
        const auto& norm = lw.k_norm_w;
        if (!norm.defined() || norm.device().type() != c10::DeviceType::PrivateUse1 ||
            norm.scalar_type() != at::kHalf || !norm.is_contiguous() ||
            norm.dim() != 1 || norm.size(0) != 256) return;
    }
    action_owner_cache_available_ = true;
}

bool Qwen3_5Model::use_action_owner_cache(int64_t rows, int64_t max_prefix) const {
    // Both dry planning and forward pass their current global prefix position.
    // The per-layer positions are checked by the consumed frozen KV plan.
    return action_owner_cache_available_ &&
        (use_action_qkv_gate(rows) || (use_action_nvfp4(rows) && use_action_kv_owner(rows))) &&
        (max_prefix == 0 || max_prefix == 245 || max_prefix == 181 || (max_prefix >= 161 && max_prefix <= 449 && max_prefix % 16 == 1));
}

std::vector<int64_t> Qwen3_5Model::action_qkv_gate_owner_only_arguments() const {
    // Original joint projection ABI, with zero fanout bytes at its own site.
    return {1,32,256,1024,8,linear_acc32_ ? 32 : 16,linear_acc32_ ? 192 : 288,128,128,2,6,2048,512,
            2,24,30,36,20,28,34,40,4,2,0,4,2,6,0,1};
}

std::vector<int64_t> Qwen3_5Model::action_k_owner_norm_rope_arguments() const {
    // Version/M/D/rotary/launch cores/active owners, BASE norm grid, RoPE
    // grid, norm bytes/K bytes, BASE selector, FP16 epsilon bits, table GP.
    return {1,32,256,64,8,0,4,32,1,512,16384,
            static_cast<int64_t>(RpuRmsNormSpmRoute::BASE),17,0,2};
}

std::vector<int64_t> Qwen3_5Model::action_owner_cache_arguments(
    const KvInsertSegmentPlan& plan) const {
    rpu_validate_kvinsert_segment_plan(plan,8,8,256);
    const auto part_is = [&](size_t index, KvInsertKernel kernel,
                             int64_t position, int64_t offset, int64_t rows) {
        const auto& part = plan.segment(index);
        return part.kernel == kernel && part.position == position &&
            part.token_offset == offset && part.rows == rows;
    };
    const bool p0 = plan.route() == KvInsertRoute::ALIGNED_V16 &&
        plan.segment_count() == 1 && part_is(0,KvInsertKernel::V16,0,0,32);
    const bool p245 = plan.route() == KvInsertRoute::HYBRID3 &&
        plan.segment_count() == 3 && part_is(0,KvInsertKernel::V2,245,0,11) &&
        part_is(1,KvInsertKernel::V16,256,11,16) &&
        part_is(2,KvInsertKernel::V2,272,27,5);
    const int64_t prefix = plan.segment(0).position;
    const bool p1 = prefix >= 161 && prefix <= 449 && prefix % 16 == 1 &&
        plan.route() == KvInsertRoute::HYBRID3 && plan.segment_count() == 3 &&
        part_is(0,KvInsertKernel::V2,prefix,0,15) &&
        part_is(1,KvInsertKernel::V16,prefix+15,15,16) &&
        part_is(2,KvInsertKernel::V2,prefix+31,31,1);
    const bool p181 = plan.route() == KvInsertRoute::HYBRID3 &&
        plan.segment_count() == 3 && part_is(0,KvInsertKernel::V2,181,0,11) &&
        part_is(1,KvInsertKernel::V16,192,11,16) &&
        part_is(2,KvInsertKernel::V2,208,27,5);
    TORCH_CHECK(plan.logical_rows() == 32 && plan.physical_rows() == 32 &&
                    (p0 || p245 || p1 || p181),
                "G05 owner cache requires exact unpadded bounded P0/P181/P245/P%16=1 M32 segment plan");
    // Transport schedule only: the original KV_INSERT site remains the
    // sole authority for the standard KV22 plan and native cost domain.
    // Inner cache ABI and source owners. Capacity is validated against the
    // actual full DDR owner; it does not change these strides or row writes.
    std::vector<int64_t> args{1,32,256,8,2,0,4,2,6,16384,512,1,16,8,16,16};
    const auto segment_args = rpu_kvinsert_route_arguments(plan,8,8,256);
    args.insert(args.end(),segment_args.begin(),segment_args.end());
    return args;
}

void Qwen3_5Model::launch_action_qkv_gate_owner_only(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_owner_cache(chunk.len,ctx().position) && chunk.offset == 0 &&
                    layer >= 0 && layer < num_layers() && ctx().has_complete_physical_manifest(),
                "G05 owner-only QKV/gate requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,
        QWEN35_ACTION_QKV_GATE_OWNER_ONLY_W8_SITE,1,0,
        action_qkv_gate_owner_only_arguments(),chunk.idx);
    const auto& lw = layer_weights_[layer];
    rpu_launch_g05_qkv_gate_owner_only_linear_w8a16_acc32_kernel(
        addr(0,"input_norm"),lw.q_w,lw.attn_gate_w,lw.k_w,lw.v_w,
        lw.q_ws,lw.attn_gate_ws,lw.k_ws,lw.v_ws,
        addr(0,"q"),addr(0,"attn_gate"),addr(0,"k"),addr(0,"v"),linear_acc32_);
}

void Qwen3_5Model::launch_action_k_owner_norm_rope(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_owner_cache(chunk.len,ctx().position) && chunk.offset == 0 &&
                    layer >= 0 && layer < num_layers() && ctx().has_complete_physical_manifest(),
                "G05 K owner norm/RoPE requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(FmbRouteFamily::ROPE,
        QWEN35_ACTION_K_OWNER_NORM_ROPE_SITE,1,0,
        action_k_owner_norm_rope_arguments(),chunk.idx);
    rpu_launch_g05_k_owner_norm_rope_spm_kernel(
        addr(0,"k"),layer_addr(layer,0,"k_norm_w"),
        prefill_cos_,prefill_sin_,chunk.offset);
}

void Qwen3_5Model::prepare_action_owner_cache_joint() {
    action_owner_cache_joint_available_ = false;
    if (!use_action_nvfp4(action_len_) || !action_owner_cache_available_) return;
    for (auto id : {KernelId::G05_OWNER_INSERT_KV_CACHE_M32D256_P0,
                    KernelId::G05_OWNER_INSERT_KV_CACHE_M32D256_P245})
        if (!KernelCache::instance().has_loaded(id)) return;
    action_owner_cache_joint_available_ = true;
}

bool Qwen3_5Model::use_action_owner_cache_joint(int64_t rows, int64_t prefix) const {
    return action_owner_cache_joint_available_ && use_action_nvfp4(rows) &&
        (prefix == 0 || prefix == 245) && use_action_owner_cache(rows,prefix);
}

std::vector<int64_t> Qwen3_5Model::action_owner_cache_joint_abi() const {
    // Version/cold capability/FP16 cache element bits, M/D/TP, two families, P0/P245 grids,
    // source owners, independent SPM/DDR GP pairs and unchanged cache layout.
    return {1,action_owner_cache_joint_available_?1:0,16,32,256,8,2,2,17,
            0,4,2,6,6,10,8,12,1,16,8,16,16,16384};
}

std::vector<int64_t> Qwen3_5Model::action_owner_cache_joint_arguments(
    const KvInsertSegmentPlan& plan) const {
    auto arguments = action_owner_cache_joint_abi();
    const auto original = action_owner_cache_arguments(plan);
    arguments.insert(arguments.end(),original.begin(),original.end());
    return arguments;
}

void Qwen3_5Model::launch_action_owner_cache_insert(
    int layer, const ChunkInfo& chunk, const KvInsertSegmentPlan& plan) {
    TORCH_CHECK(use_action_owner_cache(chunk.len,ctx().position) && chunk.offset == 0 &&
                    layer >= 0 && layer < num_layers() && ctx().has_complete_physical_manifest(),
                "G05 owner cache insert requires its exact COMPLETE descriptor");
    const int64_t position = action_prefix_lens_[layer];
    TORCH_CHECK(plan.segment(0).position == position,
                "G05 owner cache live prefix differs from its immutable plan");
    if (use_action_owner_cache_joint(chunk.len,plan.segment(0).position)) {
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            QWEN35_ACTION_OWNER_CACHE_JOINT_SITE,1,0,
            action_owner_cache_joint_arguments(plan),position > 0 ? 1 : 0);
        rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(
            (*ctx().k_caches)[layer],(*ctx().v_caches)[layer],
            addr_offset("k").value,addr_offset("v").value,plan);
        return;
    }
    ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
        QWEN35_ACTION_OWNER_CACHE_INSERT_SITE,1,0,
        action_owner_cache_arguments(plan),position > 0 ? 1 : 0);
    rpu_launch_g05_owner_insert_kvcache_spm_with_plan(
        (*ctx().k_caches)[layer],(*ctx().v_caches)[layer],
        addr_offset("k").value,addr_offset("v").value,plan);
}

void Qwen3_5Model::prepare_action_nvfp4_o_resident() {
    action_nvfp4_o_resident_available_ = false;
    action_nvfp4_o_resident_ = {};
    // Optional, exact mixed profile. No fallback tile override or FP16/W8 owner changes.
    if (!use_action_nvfp4(action_len_) || !action_gateup_pair_nvfp4_ ||
        !action_gateup_pair_available_ || (!high_precision_ && !action_silu_round_mul_available_) ||
        !KernelCache::instance().has_loaded(
            linear_acc32_ ? KernelId::G05_O_RESIDENT_NVFP4_ACC32_M32N1024K256_N80
                          : KernelId::G05_O_RESIDENT_NVFP4_ACC16_M32N1024K256_N80)) return;
    TORCH_CHECK(layer_weights_.size() == 24 && action_nvfp4_.values.size() == 5,
                "G05 resident NVFP4 O requires the complete cold Action owner");
    std::vector<at::Tensor> weights, scales;
    weights.reserve(24); scales.reserve(24);
    for (const auto& lw : layer_weights_) {
        weights.push_back(lw.o_w); scales.push_back(lw.o_ws);
    }
    auto candidate = rpu_pack_g05_o_resident_nvfp4(weights, scales);
    action_nvfp4_o_resident_ = std::move(candidate);
    action_nvfp4_o_resident_available_ = true;
}

bool Qwen3_5Model::use_action_nvfp4_o_resident(int64_t rows) const {
    return action_nvfp4_o_resident_available_ && use_action_nvfp4(rows) &&
        action_nvfp4_o_resident_.first.defined() && action_nvfp4_o_resident_.second.defined() &&
        use_action_mod_body();
}

int64_t Qwen3_5Model::action_nvfp4_o_resident_site() const {
    return linear_acc32_ ? QWEN35_ACTION_O_RESIDENT_NVFP4_ACC32_SITE
                         : QWEN35_ACTION_O_RESIDENT_NVFP4_SITE;
}

std::vector<int64_t> Qwen3_5Model::action_nvfp4_o_resident_arguments(
    int64_t layer, int64_t preload) const {
    TORCH_CHECK(layer >= 0 && layer < 24 && preload >= -1 && preload < 2 &&
                    (preload == -1 || layer == 0), "G05 resident NVFP4 O route index invalid");
    // Version, exact profile/mode, parent/grid, FP4 and FP8 layer/pool bytes,
    // absolute-SPM encodings/offsets, original tensor-scale ABI and body0 preload.
    return {1,layer,24,10,32,1024,2048,256,8,linear_acc32_ ? 32 : 16,272,80,128,13,
            131072,16384,24*131072,24*16384,layer*131072,layer*16384,
            // Physical route arguments are nonnegative: compute=0, FP4=1, FP8=2.
            2,20,24,27,2048,0,preload+1,96,action_nvfp4_o_resident_available_ ? 1 : 0};
}

void Qwen3_5Model::preload_action_nvfp4_o_resident() {
    if (!use_action_nvfp4_o_resident(action_len_)) return;
    TORCH_CHECK(ctx().has_complete_physical_manifest() && ctx().body_iter >= 0 &&
                    ctx().body_iter < action_num_steps_, "G05 NVFP4 O preload requires COMPLETE body ownership");
    if (ctx().body_iter != 0) return;
    for (int64_t family = 0; family < 2; ++family)
        ctx().consume_physical_route(FmbRouteFamily::MUTABLE_DMA,
            QWEN35_ACTION_O_PRELOAD_NVFP4_SITE,
            qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),0,
            action_nvfp4_o_resident_arguments(0,family),family);
    rpu_launch_g05_o_resident_nvfp4_preload(
        action_nvfp4_o_resident_.first,addr(0,"action_nvfp4_o_weight"),0);
    rpu_launch_g05_o_resident_nvfp4_preload(
        action_nvfp4_o_resident_.second,addr(0,"action_nvfp4_o_scale"),1);
}

void Qwen3_5Model::launch_action_nvfp4_o_resident(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_nvfp4_o_resident(chunk.len) && chunk.idx == 0 && chunk.offset == 0 &&
                    layer >= 0 && layer < 24 && ctx().has_complete_physical_manifest(),
                "G05 NVFP4 O resident requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,linear_acc32_ ? QWEN35_ACTION_O_RESIDENT_NVFP4_ACC32_SITE
                     : QWEN35_ACTION_O_RESIDENT_NVFP4_SITE,
        static_cast<int64_t>(linear_acc32_
            ? FmbLinearRouteSelector::G05_NVFP4_RESIDENT_ACC32
            : FmbLinearRouteSelector::G05_NVFP4_RESIDENT_ACC16),0,
        action_nvfp4_o_resident_arguments(layer,-1),layer);
    if (linear_acc32_) {
        rpu_launch_g05_o_resident_nvfp4_acc32_kernel(addr(0,"output"),
            addr(0,"action_nvfp4_o_weight") + static_cast<uint32_t>(layer*131072),addr(0,"oproj"),
            addr(0,"action_nvfp4_o_scale") + static_cast<uint32_t>(layer*16384),
            addr(0,G05Nvfp4Tables::names[1]),static_cast<uint16_t>(layer));
    } else {
        rpu_launch_g05_o_resident_nvfp4_acc16_kernel(addr(0,"output"),
            addr(0,"action_nvfp4_o_weight") + static_cast<uint32_t>(layer*131072),addr(0,"oproj"),
            addr(0,"action_nvfp4_o_scale") + static_cast<uint32_t>(layer*16384),
            addr(0,G05Nvfp4Tables::names[1]),static_cast<uint16_t>(layer));
    }
}

void Qwen3_5Model::prepare_action_nvfp4_down_resident() {
    action_nvfp4_down_resident_available_ = false;
    action_nvfp4_down_resident_ = {};
    // The joint capacity proof replaces the old W8 O/partial-Down roots with
    // all 24 compressed O layers and the first ten compressed Down layers.
    if (!action_nvfp4_o_resident_available_ || !use_action_nvfp4(action_len_) ||
        !KernelCache::instance().has_loaded(
            linear_acc32_ ? KernelId::G05_DOWN_RESIDENT_NVFP4_ACC32_M32N1024K512_N64
                          : KernelId::G05_DOWN_RESIDENT_NVFP4_ACC16_M32N1024K512_N64)) return;
    TORCH_CHECK(layer_weights_.size() == 24 && action_nvfp4_.values.size() == 5,
                "G05 resident NVFP4 Down requires the complete cold Action owner");
    std::vector<at::Tensor> weights, scales;
    weights.reserve(10); scales.reserve(10);
    for (int layer = 0; layer < 10; ++layer) {
        const auto& lw = layer_weights_[layer];
        weights.push_back(lw.down_w); scales.push_back(lw.down_ws);
    }
    auto candidate = rpu_pack_g05_down_resident_nvfp4(weights, scales);
    action_nvfp4_down_resident_ = std::move(candidate);
    action_nvfp4_down_resident_available_ = true;
}

bool Qwen3_5Model::use_action_nvfp4_down_resident(int64_t rows, int64_t layer) const {
    return layer >= 0 && layer < 10 && action_nvfp4_down_resident_available_ &&
        use_action_nvfp4_o_resident(rows) &&
        action_nvfp4_down_resident_.first.defined() && action_nvfp4_down_resident_.second.defined() &&
        use_action_mod_body();
}

int64_t Qwen3_5Model::action_nvfp4_down_resident_site() const {
    return linear_acc32_ ? QWEN35_ACTION_DOWN_RESIDENT_NVFP4_ACC32_SITE
                         : QWEN35_ACTION_DOWN_RESIDENT_NVFP4_SITE;
}

std::vector<int64_t> Qwen3_5Model::action_nvfp4_down_resident_arguments(
    int64_t layer, int64_t preload) const {
    TORCH_CHECK(layer >= 0 && layer < 10 && preload >= -1 && preload < 2 &&
                    (preload == -1 || layer == 0), "G05 resident NVFP4 Down route index invalid");
    // Version, family4, exact profile/mode, parent/grid, FP4 and FP8 layer/pool bytes,
    // absolute-SPM encodings/offsets, original tensor-scale ABI and body0 preload.
    return {1,4,layer,24,10,10,32,1024,4096,512,8,linear_acc32_ ? 32 : 16,320,64,128,16,
            262144,32768,10*262144,10*32768,layer*262144,layer*32768,
            // Physical route arguments are nonnegative: compute=0, FP4=1, FP8=2.
            2,20,24,27,2048,0,preload+1,96,action_nvfp4_down_resident_available_ ? 1 : 0};
}

void Qwen3_5Model::preload_action_nvfp4_down_resident() {
    if (!use_action_nvfp4_down_resident(action_len_,0)) return;
    TORCH_CHECK(ctx().has_complete_physical_manifest() && ctx().body_iter >= 0 &&
                    ctx().body_iter < action_num_steps_, "G05 NVFP4 Down preload requires COMPLETE body ownership");
    if (ctx().body_iter != 0) return;
    for (int64_t family = 0; family < 2; ++family)
        ctx().consume_physical_route(FmbRouteFamily::MUTABLE_DMA,
            QWEN35_ACTION_DOWN_PRELOAD_NVFP4_SITE,
            qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),0,
            action_nvfp4_down_resident_arguments(0,family),family);
    rpu_launch_g05_down_resident_nvfp4_preload(
        action_nvfp4_down_resident_.first,addr(0,"action_nvfp4_down_weight"),0);
    rpu_launch_g05_down_resident_nvfp4_preload(
        action_nvfp4_down_resident_.second,addr(0,"action_nvfp4_down_scale"),1);
}

void Qwen3_5Model::launch_action_nvfp4_down_resident(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_nvfp4_down_resident(chunk.len,layer) && chunk.idx == 0 && chunk.offset == 0 &&
                    layer >= 0 && layer < 10 && ctx().has_complete_physical_manifest(),
                "G05 NVFP4 Down resident requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,linear_acc32_ ? QWEN35_ACTION_DOWN_RESIDENT_NVFP4_ACC32_SITE
                     : QWEN35_ACTION_DOWN_RESIDENT_NVFP4_SITE,
        static_cast<int64_t>(linear_acc32_
            ? FmbLinearRouteSelector::G05_NVFP4_RESIDENT_ACC32
            : FmbLinearRouteSelector::G05_NVFP4_RESIDENT_ACC16),0,
        action_nvfp4_down_resident_arguments(layer,-1),layer);
    if (linear_acc32_) {
        rpu_launch_g05_down_resident_nvfp4_acc32_kernel(addr(0,"gate"),
            addr(0,"action_nvfp4_down_weight") + static_cast<uint32_t>(layer*262144),addr(0,"down"),
            addr(0,"action_nvfp4_down_scale") + static_cast<uint32_t>(layer*32768),
            addr(0,G05Nvfp4Tables::names[4]),static_cast<uint16_t>(layer));
    } else {
        rpu_launch_g05_down_resident_nvfp4_acc16_kernel(addr(0,"gate"),
            addr(0,"action_nvfp4_down_weight") + static_cast<uint32_t>(layer*262144),addr(0,"down"),
            addr(0,"action_nvfp4_down_scale") + static_cast<uint32_t>(layer*32768),
            addr(0,G05Nvfp4Tables::names[4]),static_cast<uint16_t>(layer));
    }
}

void Qwen3_5Model::prepare_action_o_resident() {
    action_o_resident_available_ = false;
    action_o_resident_n128_ = false;
    action_o_resident_.clear();
    // The paired Gate/Up proof supplies the exact all-full Action geometry and
    // its replacement MLP route. O admission is cold and ACC16-only.
    if (linear_acc32_ || !action_gateup_pair_available_ || !action_gateup_pair_w8_ ||
        (!high_precision_ && !action_silu_round_mul_available_) ||
        !KernelCache::instance().has_loaded(KernelId::G05_O_RESIDENT_W8A16_ACC16_M32N1024K256)) return;
    for (const auto& lw : layer_weights_) {
        for (const auto* weight : {&lw.q_w, &lw.k_w, &lw.v_w, &lw.o_w,
                                  &lw.attn_gate_w, &lw.gate_w, &lw.up_w, &lw.down_w}) {
            if (!weight->defined() || weight->scalar_type() != at::kChar) return;
        }
        const auto& weight = lw.o_w;
        const auto& scale = lw.o_ws;
        if (weight.device().type() != c10::DeviceType::PrivateUse1 ||
            weight.device() != layer_weights_.front().o_w.device() ||
            !weight.is_contiguous() || weight.dim() != 2 ||
            weight.size(0) != 1024 || weight.size(1) != 2048 || weight.storage_offset() != 0 ||
            !scale.defined() || scale.device().type() != c10::DeviceType::PrivateUse1 ||
            scale.scalar_type() != at::kHalf || !scale.is_contiguous() ||
            scale.dim() != 1 || scale.size(0) != 1024 || scale.storage_offset() != 0) return;
        for (const auto* owner : {&weight, &scale}) {
            const uint64_t address = RpuGetDevAddr(owner->data_ptr());
            constexpr uint64_t limit = UINT64_C(1) << 40;
            if (address == 0 || address % 256 != 0 || address >= limit ||
                owner->nbytes() > limit - address) return;
        }
    }
    std::vector<at::Tensor> owners;
    owners.reserve(6);
    for (int64_t first = 0; first < 24; first += 4) {
        std::vector<at::Tensor> weights;
        for (int64_t layer = first; layer < first + 4; ++layer)
            weights.push_back(layer_weights_[layer].o_w);
        owners.push_back(rpu_pack_g05_o_resident_w8a16(weights));
    }
    action_o_resident_ = std::move(owners);
    action_o_resident_available_ = true;
    action_o_resident_n128_ = KernelCache::instance().has_loaded(
        KernelId::G05_O_RESIDENT_W8A16_ACC16_M32N1024K256_N128);
}

bool Qwen3_5Model::use_action_o_resident(int64_t rows) const {
    return action_o_resident_available_ && !linear_acc32_ &&
        action_o_resident_.size() == 6 &&
        std::all_of(action_o_resident_.begin(), action_o_resident_.end(),
                    [](const at::Tensor& owner) { return owner.defined(); }) &&
        use_action_mod_body() && rows == 32;
}

std::vector<int64_t> Qwen3_5Model::action_o_resident_arguments(
    int64_t layer, bool preload) const {
    TORCH_CHECK(layer >= 0 && layer < 24 && (!preload || layer % 4 == 0),
                "G05 resident O layer/bundle is invalid");
    // Version, layer/bundles/steps, global and local geometry, accumulation,
    // packed byte strides, exact generated parent/grid, operation and SPM
    // offset; original scale GP, weight-core stride and scale-core stride.
    return {1,layer,24,4,6,10,32,1024,2048,256,8,16,
            262144,1048576,QWEN35_ACTION_RESIDENT_TOTAL_BYTES,
            action_o_resident_n128_ ? 288 : 448,action_o_resident_n128_ ? 128 : 64,
            128,action_o_resident_n128_ ? 8 : 16,preload ? 1 : 0,layer * 262144,20,15,8,22,0};
}

void Qwen3_5Model::preload_action_o_resident() {
    if (!use_action_o_resident(action_len_)) return;
    TORCH_CHECK(ctx().has_complete_physical_manifest() &&
                    ctx().body_iter >= 0 && ctx().body_iter < action_num_steps_,
                "G05 resident O requires its COMPLETE loop descriptor");
    if (ctx().body_iter != 0) return;
    for (int64_t bundle = 0; bundle < 6; ++bundle) {
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA,QWEN35_ACTION_O_PRELOAD_W8_SITE,
            qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
            0,action_o_resident_arguments(bundle * 4,true),bundle);
    }
    for (int64_t bundle = 0; bundle < 6; ++bundle) {
        rpu_launch_g05_gateup_resident_w8a16_preload(
            action_o_resident_[bundle],addr(0,"action_o_resident") +
                static_cast<uint32_t>(bundle * 1048576));
    }
}

void Qwen3_5Model::launch_action_o_resident(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_o_resident(chunk.len) && chunk.idx == 0 && chunk.offset == 0 &&
                    layer >= 0 && layer < 24 && ctx().has_complete_physical_manifest(),
                "G05 resident O requires its exact COMPLETE descriptor");
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,
        QWEN35_ACTION_O_RESIDENT_W8_SITE,6,0,
        action_o_resident_arguments(layer,false),layer);
    rpu_launch_g05_o_resident_w8a16_acc16_kernel(
        addr(0,"output"),addr(0,"action_o_resident") +
            static_cast<uint32_t>(layer * 262144),addr(0,"oproj"),
        layer_weights_[layer].o_ws,action_o_resident_n128_);
}

void Qwen3_5Model::prepare_action_down_partial() {
    action_down_partial_available_ = false;
    action_down_partial_ = at::Tensor();
    // Extra headroom only: keep all 24 O layers and bind exactly layer0's
    // first672 output columns. Ten-body eligibility remains the O profile's.
    if (linear_acc32_ || !action_o_resident_available_ ||
        !KernelCache::instance().has_loaded(
            KernelId::G05_DOWN_PARTIAL672_W8A16_ACC16_M32N1024K512)) return;
    const auto& weight = layer_weights_.front().down_w;
    const auto& scale = layer_weights_.front().down_ws;
    if (!weight.defined() || weight.scalar_type() != at::kChar ||
        weight.device().type() != c10::DeviceType::PrivateUse1 ||
        !weight.is_contiguous() || weight.dim() != 2 ||
        weight.size(0) != 1024 || weight.size(1) != 4096 || weight.storage_offset() != 0 ||
        !scale.defined() || scale.scalar_type() != at::kHalf ||
        scale.device() != weight.device() || !scale.is_contiguous() ||
        scale.dim() != 1 || scale.size(0) != 1024 || scale.storage_offset() != 0) return;
    constexpr uint64_t limit = UINT64_C(1) << 40;
    for (const auto* owner : {&weight, &scale}) {
        const uint64_t address = RpuGetDevAddr(owner->data_ptr());
        if (address == 0 || address % 256 != 0 || address >= limit ||
            owner->nbytes() > limit - address) return;
    }
    const uint64_t weight_address = RpuGetDevAddr(weight.data_ptr());
    const uint64_t scale_address = RpuGetDevAddr(scale.data_ptr());
    if (!(weight_address + weight.nbytes() <= scale_address ||
          scale_address + scale.nbytes() <= weight_address)) return;
    action_down_partial_ = rpu_pack_g05_down_partial_w8a16(weight);
    action_down_partial_available_ = true;
}

bool Qwen3_5Model::use_action_down_partial(int64_t rows, int64_t layer) const {
    return layer == 0 && action_down_partial_available_ &&
        action_down_partial_.defined() && use_action_o_resident(rows);
}

std::vector<int64_t> Qwen3_5Model::action_down_partial_arguments(bool preload) const {
    // Version/layer/total-layers/bodies, full and local shape, TP/ACC,
    // resident channels/blocks/bytes, unchanged parent tile/grid, operation,
    // SPM/DDR weight GP, original scale GP, weight stride and row partition.
    return {1,0,24,10,32,1024,4096,512,8,16,672,14,344064,
            512,48,128,22,preload ? 1 : 0,24,2,20,15,8,22,0};
}

void Qwen3_5Model::preload_action_down_partial() {
    if (!use_action_down_partial(action_len_,0)) return;
    TORCH_CHECK(ctx().has_complete_physical_manifest() &&
                    ctx().body_iter >= 0 && ctx().body_iter < action_num_steps_,
                "G05 partial Down requires its COMPLETE ten-body descriptor");
    if (ctx().body_iter != 0) return;
    ctx().consume_physical_route(
        FmbRouteFamily::MUTABLE_DMA,QWEN35_ACTION_DOWN_PARTIAL_PRELOAD_W8_SITE,
        qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
        0,action_down_partial_arguments(true),0);
    rpu_launch_g05_down_partial_w8a16_preload(
        action_down_partial_,addr(0,"action_down_partial"));
}

void Qwen3_5Model::launch_action_down_partial(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_down_partial(chunk.len,layer) &&
                    chunk.idx == 0 && chunk.offset == 0 && ctx().has_complete_physical_manifest(),
                "G05 partial Down requires its exact layer0 COMPLETE descriptor");
    ctx().consume_physical_route(FmbRouteFamily::LINEAR,
        QWEN35_ACTION_DOWN_PARTIAL_W8_SITE,6,0,action_down_partial_arguments(false),0);
    rpu_launch_g05_down_partial_w8a16_acc16_kernel(
        addr(0,"gate"),layer_weights_[0].down_w,addr(0,"action_down_partial"),
        addr(0,"down"),layer_weights_[0].down_ws);
}

void Qwen3_5Model::prepare_action_gateup_resident() {
    action_mod_body_w8_ = false;
    action_mod_body_fp16_ = false;
    action_gateup_resident_available_ = false;
    action_gateup_resident_w8_ = false;
    action_gateup_resident_ = {};
    // Installation precedes the first action-step/loop state. Bind eligibility
    // and immutable packed ownership here; no repacking or payload lookup in
    // planning, BUILD or REPLAY.
    if (!action_mode_ || has_gdn_ || action_len_ != 32 ||
        num_layers() != 24 || hidden_size() != 1024 || intermediate_size() != 4096 ||
        num_cores() != 8 || mlp_tp() != 8 || attn_tp() != 8 ||
        num_q_heads() != 8 || num_kv_heads() != 8 || head_dim() != 256 ||
        !use_silu_ || layer_weights_.size() != 24 || layer_is_full_.size() != 24 ||
        !std::all_of(layer_is_full_.begin(), layer_is_full_.end(),
                     [](uint8_t full) { return full != 0; })) return;
    const bool w8 = layer_weights_[0].gate_w.defined() &&
        layer_weights_[0].gate_w.scalar_type() == at::kChar;
    for (const auto& lw : layer_weights_) {
        for (const auto* weight : {&lw.q_w, &lw.k_w, &lw.v_w, &lw.o_w,
                                  &lw.attn_gate_w, &lw.gate_w, &lw.up_w, &lw.down_w}) {
            if (!weight->defined() || weight->scalar_type() != (w8 ? at::kChar : at::kHalf)) return;
        }
    }
    // This precision-independent DMA capability does not depend on an optional
    // resident payload. All layer projection dtypes were proved above.
    action_mod_body_w8_ = w8;
    action_mod_body_fp16_ = !w8;
    if (action_o_resident_available_) return;
    if (!linear_acc32_ && !w8) return;
    if (w8) {
        for (const auto& lw : layer_weights_) {
            for (const auto* scale : {&lw.gate_ws, &lw.up_ws}) {
                if (!scale->defined() || scale->scalar_type() != at::kHalf ||
                    scale->device().type() != c10::DeviceType::PrivateUse1 ||
                    !scale->is_contiguous() || scale->dim() != 1 || scale->size(0) != 4096) return;
            }
        }
    }
    if (!KernelCache::instance().has_loaded(w8
            ? (linear_acc32_ ? KernelId::G05_GATEUP_RESIDENT_W8A16_ACC32_M240N96K128
                             : KernelId::G05_GATEUP_RESIDENT_W8A16_ACC16_M352N96K128)
            : KernelId::G05_GATEUP_RESIDENT_FP16_ACC32_M304N80K128)) return;
    const int64_t resident_layers = w8 ? QWEN35_ACTION_RESIDENT_W8_LAYERS
                                      : QWEN35_ACTION_RESIDENT_LAYERS;
    action_gateup_resident_.resize(resident_layers);
    for (int64_t slot = 0; slot < resident_layers; ++slot) {
        const auto& weights = layer_weights_[slot + 1];
        action_gateup_resident_[slot] = w8
            ? rpu_pack_g05_gateup_resident_w8a16(weights.gate_w, weights.up_w)
            : rpu_pack_g05_gateup_resident_fp16(weights.gate_w, weights.up_w);
    }
    action_gateup_resident_w8_ = w8;
    action_gateup_resident_available_ = true;
}

int64_t Qwen3_5Model::action_gateup_resident_layers() const {
    if (action_o_resident_available_) return 0;
    // W8 doubles the retained layers within the original FP16 six-MiB root.
    // This cold precision identity also binds pair/activation exclusions.
    return action_gateup_resident_w8_ ? QWEN35_ACTION_RESIDENT_W8_LAYERS
                                      : QWEN35_ACTION_RESIDENT_LAYERS;
}

bool Qwen3_5Model::use_action_gateup_resident(int64_t rows) const {
    return action_gateup_resident_available_ &&
        static_cast<int64_t>(action_gateup_resident_.size()) == action_gateup_resident_layers() &&
        std::all_of(action_gateup_resident_.begin(), action_gateup_resident_.end(),
                    [](const at::Tensor& owner) { return owner.defined(); }) &&
        use_action_mod_body() && intermediate_size() == 4096 && mlp_tp() == 8 && rows == 32;
}

std::vector<int64_t> Qwen3_5Model::action_gateup_resident_arguments(
        int64_t layer, int64_t projection) const {
    TORCH_CHECK(layer >= 1 && layer <= action_gateup_resident_layers() &&
                    projection >= 0 && projection <= 2,
                "G05 resident Gate/Up layer/projection is invalid");
    // Version, layer, pair count, steps, M/N/K, TP, matrix/bundle/total bytes, tile M/N/K,
    // projection (2 denotes the pair preload), offset within each core.
    if (action_gateup_resident_w8_) {
        std::vector<int64_t> args{linear_acc32_ ? 4 : 5, layer, action_gateup_resident_layers(), 10, 32, 4096, 1024, 8,
            524288,1048576,QWEN35_ACTION_RESIDENT_TOTAL_BYTES,linear_acc32_ ? 240 : 352,96,128,projection,
            (layer - 1) * 1048576 + (projection == 2 ? 0 : projection * 524288),20,1};
        if (!linear_acc32_) args.push_back(16);
        return args;
    }
    return {2, layer, action_gateup_resident_layers(), 10, 32, 4096, 1024, 8,
            QWEN35_ACTION_RESIDENT_MATRIX_BYTES, QWEN35_ACTION_RESIDENT_BUNDLE_BYTES,
            QWEN35_ACTION_RESIDENT_TOTAL_BYTES,
            304, 80, 128, projection,
            (layer - 1) * QWEN35_ACTION_RESIDENT_BUNDLE_BYTES +
                (projection == 2 ? 0 : projection * QWEN35_ACTION_RESIDENT_MATRIX_BYTES)};
}

void Qwen3_5Model::preload_action_gateup_resident() {
    if (!use_action_gateup_resident(action_len_)) return;
    TORCH_CHECK(ctx().has_complete_physical_manifest() &&
                    ctx().body_iter >= 0 && ctx().body_iter < action_num_steps_,
                "G05 resident Gate/Up requires its COMPLETE loop descriptor");
    if (ctx().body_iter != 0) return;
    // Reject a stale member before enqueueing any of this group's transfers.
    for (int64_t slot = 0; slot < action_gateup_resident_layers(); ++slot) {
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA, action_gateup_resident_w8_ ? QWEN35_ACTION_GATEUP_PRELOAD_W8_SITE : QWEN35_ACTION_GATEUP_PRELOAD_SITE,
            qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
            0, action_gateup_resident_arguments(slot + 1, 2), slot);
    }
    for (int64_t slot = 0; slot < action_gateup_resident_layers(); ++slot) {
        if (action_gateup_resident_w8_) {
            rpu_launch_g05_gateup_resident_w8a16_preload(
                action_gateup_resident_[slot], addr(0,"action_gateup_resident") +
                    static_cast<uint32_t>(slot * 1048576));
        } else {
        rpu_launch_ddr_scatter_spm_dma(
            action_gateup_resident_[slot], 0,
            QWEN35_ACTION_RESIDENT_BUNDLE_BYTES / DWIDTH,
            QWEN35_ACTION_RESIDENT_BUNDLE_BYTES, addr(0, "action_gateup_resident") +
                static_cast<uint32_t>(slot * QWEN35_ACTION_RESIDENT_BUNDLE_BYTES), NUM_CORES);
        }
    }
}

void Qwen3_5Model::launch_action_gateup_resident(
        int64_t layer, int64_t projection, uint32_t output) {
    TORCH_CHECK(use_action_gateup_resident(32) &&
                    layer >= 1 && layer <= action_gateup_resident_layers() &&
                    projection >= 0 && projection < 2 &&
                    ctx().has_complete_physical_manifest(),
                "G05 resident Gate/Up requires its COMPLETE Linear descriptor");
    const int64_t invocation = (layer - 1) * 2 + projection;
    ctx().consume_physical_route(
        FmbRouteFamily::LINEAR, action_gateup_resident_w8_ ? QWEN35_ACTION_GATEUP_RESIDENT_W8_SITE : QWEN35_ACTION_GATEUP_RESIDENT_SITE,
        /*SPM W8/FP16; accumulation bound by manifest=*/action_gateup_resident_w8_ ? 6 : 5, 0,
        action_gateup_resident_arguments(layer, projection), invocation);
    if (action_gateup_resident_w8_) {
        const auto& weights = layer_weights_[layer];
        rpu_launch_g05_gateup_resident_w8a16_kernel(
            addr(0,"residual1"),addr(0,"action_gateup_resident") +
                static_cast<uint32_t>(invocation * 524288),output,
            projection == 0 ? weights.gate_ws : weights.up_ws, linear_acc32_);
    } else {
    rpu_launch_g05_gateup_resident_fp16_kernel(
        addr(0, "residual1"), addr(0, "action_gateup_resident") +
            static_cast<uint32_t>(invocation * QWEN35_ACTION_RESIDENT_MATRIX_BYTES), output);
    }
}

std::vector<int64_t> Qwen3_5Model::action_mod_body_arguments(int64_t body) const {
    const int64_t rows = 2 * num_layers() + 1;
    const int64_t variants = action_rtc_active_ ? 2 : 1;
    const int64_t width = 3 * hidden_size();
    return {rows, variants, width, num_cores(), body * rows * variants * width * DWIDTH};
}

void Qwen3_5Model::load_adaptive_mod_row(int64_t row) {
    TORCH_CHECK(action_mode_ && adaptive_mod_ref_.defined(),
                "Qwen3.5 action modulation is unavailable");
    const int64_t width = 3 * hidden_size();
    const int64_t variants = action_rtc_active_ ? 2 : 1;
    const int64_t rows = 2 * num_layers() + 1;
    TORCH_CHECK(row >= 0 && row < rows, "Qwen3.5 action modulation row out of range");
    adaptive_mod_row_offset_ = 0;
    int64_t step_row = action_loop_active_
        ? (ctx().body_iter * rows + row) * variants
        : row * variants;
    int64_t elements = variants * width;
    if (use_action_mod_body()) {
        TORCH_CHECK(ctx().has_complete_physical_manifest() &&
                    ctx().body_iter >= 0 && ctx().body_iter < action_num_steps_,
                    "Qwen3.5 modulation body requires its COMPLETE loop descriptor");
        adaptive_mod_row_offset_ = static_cast<uint32_t>(row * elements * DWIDTH);
        // Layer zero is the first table consumer in every body. This slot is
        // live across all 49 norm/gate consumers and replaced by the next body.
        if (row != 0) return;
        ctx().consume_physical_route(
            FmbRouteFamily::MUTABLE_DMA, QWEN35_TEXT_ADAPTIVE_MOD_DMA_SITE,
            qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_BROADCAST_MOD_BODY),
            0, action_mod_body_arguments(ctx().body_iter), ctx().body_iter);
        elements *= rows;
    } else {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ADAPTIVE_MOD_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
    }
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &adaptive_mod_live_base_,
        step_row * width * static_cast<int64_t>(sizeof(c10::Half)),
        elements, addr(0, "adaptive_mod"), NUM_CORES);
}

void Qwen3_5Model::prepare_action_newton_norm_shift() {
    // Optional scalar Newton epilogue; the existing HIGH admission has proved
    // this exact Action geometry. Vector Newton parents stay decomposed.
    action_newton_norm_shift_available_ = high_precision_ && action_mode_ &&
        !has_gdn_ && hidden_size() == 1024 && intermediate_size() == 4096 &&
        num_layers() == 24 && num_q_heads() == 8 && num_kv_heads() == 8 &&
        head_dim() == 256 && num_cores() == 8 && attn_tp() == 8 && mlp_tp() == 8 &&
        std::isfinite(eps_) && eps_ == 1e-6 &&
        KernelCache::instance().has_loaded(KernelId::RHINOVLA_NEWTON_NORM_SHIFT_H1024);
}

bool Qwen3_5Model::use_action_newton_norm_shift(int64_t rows) const {
    return high_precision_ && action_newton_norm_shift_available_ &&
        action_mode_ && action_rtc_active_ && use_action_mod_body() && rows == 32;
}

std::vector<int64_t> Qwen3_5Model::action_newton_norm_shift_arguments() const {
    // Conditional policy, not a full-M32 fusion: exact geometry/ACC/eps,
    // clean/noisy variants and row strides, live clean-domain, scalar-only
    // rule with zero skip, original V16/V32 routes, and both FP16 retirements.
    return {1,32,1024,8,24,linear_acc32_ ? 32 : 16,
            static_cast<int64_t>(c10::Half(eps_).x),2,2048,6144,0,32,1,1,16,32,
            static_cast<int64_t>(RpuRmsNormSpmRoute::NEWTON),
            static_cast<int64_t>(RpuRmsNormSpmRoute::NEWTON_V16_GRID8),
            static_cast<int64_t>(RpuRmsNormSpmRoute::NEWTON_V32_GRID8),
            2,2,0x4492,0x0092,action_newton_norm_shift_available_ ? 1 : 0};
}

void Qwen3_5Model::apply_action_newton_norm_shift(
    uint32_t input, uint32_t output, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_newton_norm_shift(chunk.len) &&
                    ctx().has_complete_physical_manifest(),
                "G05 conditional Newton norm/shift requires its exact RTC COMPLETE policy");
    ctx().consume_physical_route(
        FmbRouteFamily::NORMALIZATION,QWEN35_ACTION_NEWTON_NORM_SHIFT_SITE,1,0,
        action_newton_norm_shift_arguments(),chunk.idx);
    const int64_t h = hidden_size();
    const uint32_t modulation = addr(0,"adaptive_mod") + adaptive_mod_row_offset_;
    const int64_t clean = std::clamp<int64_t>(
        action_rtc_prefix_len_ - chunk.offset,0,chunk.len);
    auto emit_segment = [&](int64_t offset, int64_t rows, uint32_t scale) {
        if (rows == 0) return;
        const uint32_t data_offset = static_cast<uint32_t>(offset * h * DWIDTH);
        const uint32_t shift = scale + static_cast<uint32_t>(h * DWIDTH);
        const auto route = rmsnorm_route(rows,h);
        if (route == RpuRmsNormSpmRoute::NEWTON) {
            // The shared payload preserves scalar Newton's final FP16 result
            // before the original FP16 ADD with shift as its first operand.
            rpu_launch_rhinovla_newton_norm_shift_spm_kernel(
                input + data_offset,output + data_offset,scale,shift,rows,eps_,num_cores());
        } else {
            // In particular M16/M32 still use their original vector parents;
            // never substitute the scalar kernel solely to remove an ADD.
            rpu_launch_rmsnorm_spm_kernel(
                input + data_offset,output + data_offset,scale,rows,h,eps_,route,num_cores());
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                shift,output + data_offset,output + data_offset,
                rows,h,c10::Half(1.0),ValuOpType::ADD,/*is_bopa=*/false);
        }
    };
    emit_segment(0,clean,modulation);
    emit_segment(clean,chunk.len - clean,modulation + static_cast<uint32_t>(3 * h * DWIDTH));
}

int64_t Qwen3_5Model::action_norm_route(int64_t rows) const {
    // HIGH has its own cold math authority. BASE norm+shift payloads cannot
    // be reused; the optional scalar-only policy is handled separately.
    if (high_precision_) return 1;
    if (!action_mode_ || hidden_size() != 1024 || num_cores() != 8 ||
        rows < 1 || rows > 64 || !std::isfinite(eps_) || eps_ <= 0 || eps_ > 1.0)
        return 1;
    if (action_rtc_norm_shift_available_ && action_rtc_active_ && rows == 32)
        return 3;
    return action_norm_shift_available_ ? 2 : 1;
}

std::vector<int64_t> Qwen3_5Model::action_norm_arguments(int64_t rows) const {
    if (action_norm_route(rows) == 3) {
        // M/H/cores, ABI version, clean/noisy variants, modulation byte stride,
        // accepted runtime split domain. Never bind stale prefix state in dry.
        return {rows, hidden_size(), num_cores(), 1, 2,
                3 * hidden_size() * DWIDTH, 0, 32};
    }
    return {rows, hidden_size(), num_cores()};
}

void Qwen3_5Model::apply_adaptive_norm(
    uint32_t input, uint32_t output, const ChunkInfo& chunk) {
    const int64_t h = hidden_size();
    const uint32_t modulation = addr(0, "adaptive_mod") + adaptive_mod_row_offset_;
    const int64_t norm_route = action_norm_route(chunk.len);
    TORCH_CHECK(ctx().has_complete_physical_manifest(),
                "Qwen3.5 action normalization requires a COMPLETE descriptor");
    if (use_action_newton_norm_shift(chunk.len)) {
        apply_action_newton_norm_shift(input,output,chunk);
        return;
    }
    const int64_t clean_rows = action_rtc_active_
        ? std::clamp<int64_t>(
              action_rtc_prefix_len_ - chunk.offset, 0, chunk.len)
        : 0;
    if (norm_route == 3) {
        TORCH_CHECK(clean_rows >= 0 && clean_rows <= 32,
                    "G05 RTC normalization split must be within [0,32]");
        ctx().consume_physical_route(
            FmbRouteFamily::NORMALIZATION, QWEN35_ACTION_FUSED_NORM_SHIFT_SITE,
            norm_route, /*resolved_flags=*/0,
            action_norm_arguments(chunk.len), chunk.idx);
        const uint32_t noisy = modulation + static_cast<uint32_t>(3 * h * DWIDTH);
        rpu_launch_g05_rtc_adarms_norm_shift_spm_kernel(
            input, output, modulation, modulation + static_cast<uint32_t>(h * DWIDTH),
            noisy, noisy + static_cast<uint32_t>(h * DWIDTH),
            clean_rows, eps_, num_cores());
        return;
    }
    auto emit_segment = [&](int64_t row_offset, int64_t row_count,
                            uint32_t modulation) {
        if (row_count == 0) return;
        const uint32_t data_offset = static_cast<uint32_t>(
            row_offset * h * static_cast<int64_t>(DWIDTH));
        const uint32_t shift = modulation + static_cast<uint32_t>(h * DWIDTH);
        if (norm_route == 2) {
            ctx().consume_physical_route(
                FmbRouteFamily::NORMALIZATION, QWEN35_ACTION_FUSED_NORM_SHIFT_SITE,
                norm_route, /*resolved_flags=*/0,
                {chunk.len, h, num_cores()}, chunk.idx);
            rpu_launch_pi05_adarms_norm_shift_spm_kernel(
                input + data_offset, output + data_offset,
                modulation, shift, row_count, eps_, num_cores());
        } else {
            ctx().consume_physical_route(
                FmbRouteFamily::NORMALIZATION, QWEN35_ACTION_RMS_NORM_SHIFT_SITE,
                norm_route, /*resolved_flags=*/0,
                {chunk.len, h, num_cores()}, chunk.idx);
            rpu_launch_rmsnorm_spm_kernel(
                input + data_offset, output + data_offset,
                modulation, row_count, h, eps_, rmsnorm_route(row_count, h), num_cores());
            rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
                shift, output + data_offset, output + data_offset,
                row_count, h, c10::Half(1.0), ValuOpType::ADD,
                /*is_bopa=*/false);
        }
    };
    if (!action_rtc_active_) {
        emit_segment(/*row_offset=*/0, chunk.len, modulation);
        return;
    }
    emit_segment(/*row_offset=*/0, clean_rows, modulation);
    emit_segment(
        clean_rows, chunk.len - clean_rows,
        modulation + static_cast<uint32_t>(3 * h * DWIDTH));
}

void Qwen3_5Model::prepare_action_tail() {
    action_tail_available_ = false;
    if (!use_action_nvfp4(action_len_) || has_gdn_ || eps_ != 1e-6 ||
        !use_silu_ || num_layers() != 24 || hidden_size() != 1024 ||
        intermediate_size() != 4096 || num_q_heads() != 8 || num_kv_heads() != 8 ||
        head_dim() != 256 || attn_tp() != 8 || mlp_tp() != 8 ||
        layer_is_full_.size() != 24 || !action_ring_rtc_residual_available_ ||
        !action_rtc_norm_shift_available_ ||
        std::any_of(layer_is_full_.begin(),layer_is_full_.end(),
                    [](uint8_t full) { return !full; })) return;
    action_tail_available_ = KernelCache::instance().has_loaded(
        KernelId::G05_RING_RTC_THEN_NORM_SHIFT_M32H1024);
}

bool Qwen3_5Model::use_action_tail(int64_t rows) const {
    return action_tail_available_ && use_action_nvfp4(rows) &&
        use_action_ring_rtc_residual(rows) && use_action_mod_body() &&
        action_rtc_active_ && action_norm_route(rows) == 3 &&
        eps_ == 1e-6 && !get_debug_export();
}

std::vector<int64_t> Qwen3_5Model::action_tail_arguments() const {
    // Version/M/H/TP/source Linear accumulation, full raw/norm, original ring + norm modes,
    // GP norm rebase32, four row waves, explicit counter0 low/high zero,
    // full49-row two-variant modulation body and clean split domain.
    return {1,32,1024,8,linear_acc32_ ? 32 : 16,65536,65536,1,3,32,4,0,128,129,0,
            49,2,3072,12288,602112,0,32,24,10,
            static_cast<int64_t>(c10::Half(eps_).x),action_tail_available_?1:0};
}

void Qwen3_5Model::launch_action_tail(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_tail(chunk.len) && ctx().has_complete_physical_manifest() &&
                    layer >= 0 && layer < 24 && layer_is_full_[layer] &&
                    chunk.idx == 0 && chunk.offset == 0 && chunk.len == 32 &&
                    ctx().body_iter >= 0 && ctx().body_iter < 10 &&
                    adaptive_mod_row_offset_ == uint32_t(2 * layer * 12288),
                "G05 Action tail escaped its same-layer COMPLETE modulation body");
    // forward_action_step_impl validates live [10,49,2,3072] FP16 backing,
    // and layer0 loads this complete body through the original typed DMA.
    const uint64_t body = addr(0,"adaptive_mod");
    const uint64_t base = SPM_ALLOC.addr(0,0);
    TORCH_CHECK(body >= base && body % 256 == 0 &&
                    body - base + 602112 <= SpmAllocator::SPM_USABLE,
                "G05 Action tail full modulation body exceeds SPM");
    ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
        QWEN35_TEXT_FULL_ALL_REDUCE_SITE,text_ring_route(32,1024),0,{},chunk.idx);
    ctx().consume_physical_route(FmbRouteFamily::NORMALIZATION,
        QWEN35_ACTION_FUSED_NORM_SHIFT_SITE,3,0,action_norm_arguments(32),chunk.idx);
    ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
        QWEN35_ACTION_TAIL_SITE,1,0,action_tail_arguments(),chunk.idx);
    const uint32_t current = static_cast<uint32_t>(body) + adaptive_mod_row_offset_;
    const uint32_t next = current + 12288;
    const int64_t clean = std::clamp<int64_t>(action_rtc_prefix_len_,0,32);
    rpu_launch_g05_action_tail_spm_kernel(
        addr(0,"oproj"),addr(0,"residual1"),addr(0,"residual2"),
        current+4096,current+10240,next,next+2048,next+6144,next+8192,clean,eps_);
}

void Qwen3_5Model::prepare_action_mlp_tail() {
    // The same optional R30 payload and cold exact owner/math proof. Keep a
    // separate capability because this schedule carries norm across a layer.
    action_mlp_tail_available_ = action_tail_available_;
}

bool Qwen3_5Model::use_action_mlp_tail(
    const FmbThreeStageChunkPlan& plan, int64_t rows) const {
    if (!action_mlp_tail_available_ || !use_action_tail(rows) || rows != 32 ||
        plan.qkv.chunks.size() != 1 || plan.compute.chunks.size() != 1) return false;
    const auto& q = plan.qkv.chunks.front();
    const auto& c = plan.compute.chunks.front();
    return q.idx == 0 && q.offset == 0 && q.len == 32 &&
        c.idx == 0 && c.offset == 0 && c.len == 32;
}

bool Qwen3_5Model::action_mlp_tail_handoff(
    int layer, const ChunkInfo& chunk, bool producer) {
    if (!ctx().has_complete_physical_manifest() ||
        !use_action_mlp_tail(ctx().stage_plan,ctx().seq_len) ||
        ctx().batch_size != 1 || ctx().body_iter < 0 || ctx().body_iter >= 10 ||
        chunk.idx != 0 || chunk.offset != 0 || chunk.len != 32) return false;
    // FMB installs these paired flags from the same sequential SPM traversal:
    // producer(l).output_to_spm == consumer(l+1).input_in_spm within a group.
    // No readiness state survives a failed callback, body boundary or REPLAY.
    return producer ? (layer >= 0 && layer < 23 && ctx().output_to_spm)
                    : (layer > 0 && layer < 24 && ctx().input_in_spm);
}

std::vector<int64_t> Qwen3_5Model::action_mlp_tail_arguments() const {
    // Exact M/H/TP/ACC, full raw+norm, old skip aliases next input_norm,
    // producer/consumer layer domains, current row2*l+1 -> next row2*l+2,
    // full49-row RTC body, one M32 chunk, same-group SPM flags and counters.
    return {1,32,1024,8,linear_acc32_ ? 32 : 16,65536,65536,1,0,22,1,23,2,1,2,2,
            49,2,3072,12288,602112,10,1,32,1,1,128,129,0,
            static_cast<int64_t>(c10::Half(eps_).x),action_mlp_tail_available_?1:0};
}

void Qwen3_5Model::launch_action_mlp_tail(int layer, const ChunkInfo& chunk) {
    TORCH_CHECK(action_mlp_tail_handoff(layer,chunk,true) &&
        adaptive_mod_row_offset_ == uint32_t((2 * layer + 1) * 12288) &&
        addr(0,"residual2") == addr(0,"input_norm"),
        "G05 MLP tail escaped its same-group COMPLETE successor norm");
    const uint64_t body = addr(0,"adaptive_mod");
    const uint64_t base = SPM_ALLOC.addr(0,0);
    TORCH_CHECK(body >= base && body % 256 == 0 &&
        body - base + 602112 <= SpmAllocator::SPM_USABLE,
        "G05 MLP tail full modulation body exceeds SPM");
    ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE,
        QWEN35_TEXT_MLP_ALL_REDUCE_SITE,text_ring_route(32,1024),0,{},chunk.idx);
    ctx().consume_physical_route(FmbRouteFamily::NORMALIZATION,
        QWEN35_ACTION_FUSED_NORM_SHIFT_SITE,3,0,action_norm_arguments(32),chunk.idx);
    ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
        QWEN35_ACTION_MLP_TAIL_SITE,1,0,action_mlp_tail_arguments(),chunk.idx);
    const uint32_t current = static_cast<uint32_t>(body) + adaptive_mod_row_offset_;
    const uint32_t next = current + 12288;
    const int64_t clean = std::clamp<int64_t>(action_rtc_prefix_len_,0,32);
    // Original ring first retires full raw residual1. The wrapper's norm then
    // overwrites only its dead skip, which is next layer's input_norm root.
    rpu_launch_g05_action_tail_spm_kernel(
        addr(0,"down"),addr(0,"residual2"),addr(0,"residual1"),
        current+4096,current+10240,next,next+2048,next+6144,next+8192,clean,eps_);
}

bool Qwen3_5Model::use_action_ring_rtc_residual(int64_t rows) const {
    // Exact cold-proved FP16/W8/mixed NVFP4 ten-step Action. Modulation is broadcast,
    // and residual1/2 are fully replicated by the preceding original gather.
    return action_ring_rtc_residual_available_ && !z2_bound_ &&
        (action_mod_body_w8_ || action_mod_body_fp16_ || use_action_nvfp4(rows)) &&
        use_action_mod_body() && rows == 32 && attn_tp() == 8 && mlp_tp() == 8;
}

std::vector<int64_t> Qwen3_5Model::action_ring_rtc_residual_arguments() const {
    // Version, M/H/TP, source Linear mode, original nopace ring, one round,
    // owner rows, blocks, FP16 ADD/SUB/MUL/ADD retirements, replicated operands,
    // legal RTC split domain and clean/noisy gate-row stride.
    return {1,32,1024,8,linear_acc32_ ? 32 : 16,1,1,4,8,4,1,0,32,3*1024*DWIDTH};
}

void Qwen3_5Model::launch_action_ring_rtc_residual(
    uint32_t input, uint32_t residual, uint32_t output,
    int64_t ring_site, const ChunkInfo& chunk) {
    TORCH_CHECK(use_action_ring_rtc_residual(chunk.len) && chunk.offset == 0 &&
                    (ring_site == QWEN35_TEXT_FULL_ALL_REDUCE_SITE ||
                     ring_site == QWEN35_TEXT_MLP_ALL_REDUCE_SITE) &&
                    ctx().has_complete_physical_manifest(),
                "G05 ring RTC residual requires its exact COMPLETE Action profile");
    const bool attention = ring_site == QWEN35_TEXT_FULL_ALL_REDUCE_SITE;
    TORCH_CHECK(input == addr(0, attention ? "oproj" : "down") &&
                    residual == addr(0, attention ? "residual1" : "residual2") &&
                    output == addr(0, attention ? "residual2" : "residual1"),
                "G05 ring RTC residual requires the model-owned replicated residual roots");
    ctx().consume_physical_route(FmbRouteFamily::ALL_REDUCE, ring_site,
        text_ring_route(chunk.len, hidden_size()), 0, {}, chunk.idx);
    const uint32_t modulation = addr(0, "adaptive_mod") + adaptive_mod_row_offset_;
    const uint32_t clean_gate = modulation + 2 * 1024 * DWIDTH;
    const uint32_t noisy_gate = action_rtc_active_ ? clean_gate + 3 * 1024 * DWIDTH : clean_gate;
    const int64_t clean_rows = action_rtc_active_
        ? std::clamp<int64_t>(action_rtc_prefix_len_ - chunk.offset, 0, chunk.len) : 0;
    ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
        QWEN35_ACTION_RING_RTC_RESIDUAL_SITE, 1, 0,
        action_ring_rtc_residual_arguments(), chunk.idx);
    rpu_launch_g05_ring_rtc_residual_spm_kernel(
        input, residual, clean_gate, noisy_gate, output, clean_rows, num_cores());
}

int64_t Qwen3_5Model::action_residual_route(int64_t rows) const {
    if (!action_mode_ || hidden_size() != 1024 || num_cores() != 8 ||
        rows < 1 || rows > 64) return 1;
    if (action_rtc_sub_gated_residual_available_ && action_rtc_active_ && rows == 32) return 4;
    if (action_sub_gated_residual_available_) return 3;
    return action_gated_residual_available_ ? 2 : 1;
}

void Qwen3_5Model::apply_adaptive_residual_gate(
    uint32_t output, uint32_t residual, const ChunkInfo& chunk) {
    const int64_t h = hidden_size();
    const uint32_t modulation = addr(0, "adaptive_mod") + adaptive_mod_row_offset_;
    const int64_t route = action_residual_route(chunk.len);
    TORCH_CHECK(ctx().has_complete_physical_manifest(),
                "Qwen3.5 action residual gate requires a COMPLETE descriptor");
    const int64_t clean_rows = action_rtc_active_
        ? std::clamp<int64_t>(
              action_rtc_prefix_len_ - chunk.offset, 0, chunk.len)
        : 0;
    if (route == 4) {
        // Dry planning binds the split domain; each replay supplies the live prefix.
        ctx().consume_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,
            QWEN35_ACTION_RTC_RESIDUAL_GATE_SITE,4,0,
            {chunk.len,h,num_cores(),1,2,3*h*DWIDTH,0,32,4},chunk.idx);
        const uint32_t clean_gate = modulation + static_cast<uint32_t>(2*h*DWIDTH);
        rpu_launch_g05_rtc_sub_gated_residual_spm_kernel(
            output,residual,clean_gate,clean_gate + static_cast<uint32_t>(3*h*DWIDTH),
            output,clean_rows,num_cores());
        return;
    }
    auto emit_segment = [&](int64_t row_offset, int64_t row_count,
                            uint32_t modulation) {
        if (row_count == 0) return;
        const uint32_t data_offset = static_cast<uint32_t>(
            row_offset * h * static_cast<int64_t>(DWIDTH));
        const uint32_t gate =
            modulation + static_cast<uint32_t>(2 * h * DWIDTH);
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_ACTION_RESIDUAL_GATE_SITE,
            route, /*resolved_flags=*/0, {chunk.len, h, num_cores()}, chunk.idx);
        apply_raw_residual_gate(
            output + data_offset, residual + data_offset, gate, row_count, route);
    };
    if (!action_rtc_active_) {
        emit_segment(/*row_offset=*/0, chunk.len, modulation);
        return;
    }
    emit_segment(/*row_offset=*/0, clean_rows, modulation);
    emit_segment(
        clean_rows, chunk.len - clean_rows,
        modulation + static_cast<uint32_t>(3 * h * DWIDTH));
}

void Qwen3_5Model::apply_raw_residual_gate(
    uint32_t output, uint32_t residual, uint32_t gate, int64_t seq_len,
    int64_t route) {
    if (route == 3) {
        // The new payload preserves the separate FP16 SUB and MUL retirement
        // points before the original residual ADD; it needs no SPM scratch.
        rpu_launch_g05_sub_gated_residual_spm_kernel(
            output, residual, gate, output, seq_len, num_cores());
        return;
    }
    const int64_t elems = seq_len * hidden_size();
    rpu_launch_eltwise_binary_spm_kernel(
        output, residual, output, elems,
        ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
    if (route == 2) {
        // Keep the preceding FP16 subtraction and the rounded FP16 product:
        // residual + round_fp16(gate * round_fp16(output - residual)).
        rpu_launch_pi05_gated_residual_spm_kernel(
            output, residual, gate, output, seq_len, num_cores());
        return;
    }
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
        gate, output, output, seq_len, hidden_size(),
        c10::Half(1.0), ValuOpType::MUL, /*is_bopa=*/false);
    rpu_launch_eltwise_binary_spm_kernel(
        output, residual, output, elems,
        ValuOpType::ADD, c10::Half(1.0), NUM_CORES);
}

void Qwen3_5Model::emit_action_input_projection() {
    TORCH_CHECK(action_step_active_,
                "Qwen3.5 action input hook outside action-step forward");
    preload_action_gateup_resident();
    preload_action_o_resident();
    preload_action_nvfp4_o_resident();
    preload_action_nvfp4_down_resident();
    preload_action_down_partial();
    if (!action_loop_active_ || ctx().body_iter == 0) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ACTION_INPUT_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &action_input_live_base_, /*src_offset_bytes=*/0,
            action_len_ * action_dim_pad_, addr(0, "action_input"),
            /*num_cores=*/1);
    }
    if (action_rtc_active_) {
        // Replay-safe prefix inpainting: the caller supplies a fresh compact
        // DDR tensor, and every Euler body restores its committed leading rows
        // before the input projection. The previous body may have updated those
        // rows in SPM, so loading only body 0 would change RTC semantics.
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ACTION_RTC_PREFIX_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            &action_rtc_prefix_live_base_, /*src_offset_bytes=*/0,
            action_rtc_prefix_len_ * action_dim_pad_,
            addr(0, "action_input"), /*num_cores=*/1);
    }
    if (action_trace_active_) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ACTION_TRACE_PRE_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::SPM_COPY_TO_DDR));
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_input"),
            &action_trace_pre_action_live_base_,
            ctx().body_iter * action_len_ * action_dim_pad_ * DWIDTH,
            action_len_ * action_dim_pad_);
    }
    launch_linear(
        /*chunk_idx=*/0,
        addr(0, "action_input"), action_input_w_,
        addr(0, "action_hidden"), action_len_, hidden_size(),
        action_dim_pad_, /*partition=*/1, /*num_cores=*/1,
        addr(0, "action_input_bias"));
    if (action_trace_active_) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ACTION_TRACE_EMBED_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::SPM_COPY_TO_DDR));
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_hidden"),
            &action_trace_action_embed_live_base_,
            ctx().body_iter * action_len_ * hidden_size() * DWIDTH,
            action_len_ * hidden_size());
    }
    rpu_launch_spm_copy_ddr_dma(
        addr(0, "action_hidden"),
        action_hidden_stage_.data_ptr<c10::Half>(),
        action_len_ * hidden_size());
}

void Qwen3_5Model::emit_action_output_projection() {
    TORCH_CHECK(action_step_active_,
                "Qwen3.5 action output hook outside action-step forward");
    if (action_trace_active_) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ACTION_TRACE_HIDDEN_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::SPM_COPY_TO_DDR));
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "residual1"),
            &action_trace_final_hidden_live_base_,
            ctx().body_iter * action_len_ * hidden_size() * DWIDTH,
            action_len_ * hidden_size());
    }
    consume_manifest_route(
        FmbRouteFamily::MUTABLE_DMA,
        QWEN35_TEXT_ACTION_KEEP_MASK_DMA_SITE,
        qwen35_text_dma_route(
            Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &action_keep_mask_live_base_, /*src_offset_bytes=*/0,
        action_dim_pad_, addr(0, "action_keep_mask"),
        /*num_cores=*/1);
    launch_linear(
        /*chunk_idx=*/0,
        addr(0, "residual1"), action_output_w_,
        addr(0, "action_velocity"), action_len_, action_dim_pad_,
        hidden_size(), /*partition=*/1, /*num_cores=*/1,
        addr(0, "action_output_bias"));
    if (action_trace_active_) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ACTION_TRACE_VELOCITY_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::SPM_COPY_TO_DDR));
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_velocity"),
            &action_trace_velocity_live_base_,
            ctx().body_iter * action_len_ * action_dim_pad_ * DWIDTH,
            action_len_ * action_dim_pad_);
    }
    const bool final_step =
        !action_loop_active_ || ctx().body_iter + 1 == action_num_steps_;
    if (final_step) {
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "action_velocity"),
            action_velocity_stage_.data_ptr<c10::Half>(),
            action_len_ * action_dim_pad_);
    }
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "action_input"), addr(0, "action_velocity"),
        addr(0, "action_input"), action_len_ * action_dim_pad_,
        ValuOpType::ADD, action_euler_scale_, /*num_cores=*/1);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
        addr(0, "action_keep_mask"), addr(0, "action_input"),
        addr(0, "action_input"), action_len_, action_dim_pad_,
        c10::Half(1.0f), ValuOpType::MUL, /*is_bopa=*/false);
    if (action_trace_active_) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ACTION_TRACE_POST_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::SPM_COPY_TO_DDR));
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_input"),
            &action_trace_post_action_live_base_,
            ctx().body_iter * action_len_ * action_dim_pad_ * DWIDTH,
            action_len_ * action_dim_pad_);
    }
    if (final_step) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ACTION_OUTPUT_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::SPM_COPY_TO_DDR));
        rpu_launch_spm_copy_ddr_dma_mutable(
            addr(0, "action_input"), &action_output_live_base_,
            /*dst_offset_bytes=*/0, action_len_ * action_dim_pad_);
    }
}

// ── build_full_attention: Qwen3.5 full-attention path with QK norm, partial
//    M-RoPE and the attention output gate. Qwen3.5 has no DeepStack. ──────────
void Qwen3_5Model::build_full_attention(int layer_idx, const ChunkInfo& chunk,
                                      bool* text_owner_norm_emitted, bool* action_norm_emitted) {
    const auto& lw = layer_weights_[layer_idx];
    int64_t seq_len = chunk.len;
    int64_t cos_sin_start =
        action_mode_ ? chunk.offset : ctx().position + chunk.offset;
    int64_t h = hidden_size();
    int64_t nq = num_q_heads();
    int64_t nkv = num_kv_heads();
    int64_t hd = head_dim();
    // The adapter replicates logical GQA KV heads to the selected attention
    // width. One complete effective KV head lives on each active core.
    // DDR cache storage keeps the physical eight-controller layout, while
    // mixer reductions broadcast to the selected owner core domain.
    int64_t tp = attn_tp();
    // Preserve legacy eight-core auxiliary launches for non-text owners.
    const int auxiliary_cores = num_cores() == 8 ? NUM_CORES : static_cast<int>(tp);

    // input DMA + input_layernorm are done by build_layer_subgraph; the normed input is
    // in "input_norm", h_in stays in residual1 (for the all_reduce residual).
    // Phase 2: QKV linear (col-partition over tp cores)
    const bool owner_cache = use_action_owner_cache(seq_len,ctx().position);
    if (use_action_gate_kv_owner(seq_len,ctx().position)) {
        launch_linear(chunk.idx, addr(0,"input_norm"), lw.q_w, addr(0,"q"),
            seq_len,nq*hd,h,1,tp,0,lw.q_ws,0,layer_idx);
        launch_action_gate_kv_owner(layer_idx,chunk);
    } else if (owner_cache && !use_action_nvfp4(seq_len)) {
        launch_action_qkv_gate_owner_only(layer_idx, chunk);
    } else if (use_action_qkv_gate(seq_len)) {
        launch_action_qkv_gate(layer_idx, chunk);
    } else {
        const bool q_gate_pair = use_action_q_gate(seq_len);
        if (q_gate_pair) {
            launch_action_q_gate(layer_idx, chunk);
        } else {
            launch_linear(
                chunk.idx,
                addr(0, "input_norm"), lw.q_w, addr(0, "q"), seq_len, nq * hd, h,
                1, tp, 0, lw.q_ws, 0, layer_idx);
        }
        if (use_action_kv_owner(seq_len)) {
            launch_action_kv_owner(layer_idx, chunk);
        } else {
            launch_linear(
                chunk.idx,
                addr(0, "input_norm"), lw.k_w, addr(0, "k"), seq_len, nkv * hd, h,
                1, tp, 0, lw.k_ws);
            launch_linear(
                chunk.idx,
                addr(0, "input_norm"), lw.v_w, addr(0, "v"), seq_len, nkv * hd, h,
                1, tp, 0, lw.v_ws);
        }
        // Gated-attention gate projection. Compute here while "input_norm" still
        // holds the Phase-1 input RMSNorm output (Phase 3 k_norm overwrites it).
        // Applied as sigmoid(gate) ⊙ attn_out after SDPA (Qwen3.5 attn_output_gate).
        if (!q_gate_pair) {
            launch_linear(
                chunk.idx,
                addr(0, "input_norm"), lw.attn_gate_w, addr(0, "attn_gate"),
                seq_len, nq * hd, h, 1, tp, 0, lw.attn_gate_ws);
        }
    }

    // Phase 3: QK RMSNorm + RoPE. M-RoPE (interleaved) + partial rotary when
    // has_mrope_ (Qwen3.5); the 1D RoPE branch stays for non-mrope models.
    int64_t local_q_heads = nq / tp;
    int64_t local_kv_dim  = nkv * hd / tp;
    // PREFILL (seq_len>1) uses the per-forward interleaved M-RoPE tables (B channel, set via
    // set_prefill_rope) when available; DECODE + the non-mrope rope branch use the static
    // position-lookup cos_/sin_ from set_weights. Empty prefill_cos_ (non-mrope models / not set)
    // ⇒ fall back to cos_ (text-only interleaved == arange, so identical there).
    const bool use_prefill_rope = (seq_len > 1) && has_mrope_
                                  && prefill_cos_.defined() && prefill_cos_.numel() > 0;
    c10::Half* cos_ptr = (use_prefill_rope ? prefill_cos_ : cos_).data_ptr<c10::Half>();
    c10::Half* sin_ptr = (use_prefill_rope ? prefill_sin_ : sin_).data_ptr<c10::Half>();
    // W2: DECODE after an image indexes the static cos_/sin_ at position + rope_delta
    // (HF compute_3d_position_ids). PREFILL uses prefill_cos_ which already bakes the
    // shifted per-token positions from get_rope_index, so the delta must NOT double-apply
    // there. cos_sin_start feeds only the rope kernels (KV-insert recomputes its own
    // ctx().position+chunk.offset), so shifting it here does not disturb the KV slot.
    if (!use_prefill_rope) cos_sin_start += mrope_pos_delta_;

    auto emit_partial_rope = [&](uint32_t buf, int64_t heads) {
        if (seq_len <= 1) {
            consume_manifest_route(
                FmbRouteFamily::ROPE,
                QWEN35_TEXT_PARTIAL_ROPE_1D_SITE,
                qwen35_text_rope_route(
                    Qwen35TextRopeRoute::PARTIAL_ROPE_1D),
                chunk.idx);
            rpu_launch_partial_rope_1d_spm_kernel(
                buf, buf, cos_ptr, sin_ptr, cos_sin_start, seq_len, heads, hd, rotary_dim_, tp);
        } else {
            consume_manifest_route(
                FmbRouteFamily::ROPE,
                QWEN35_TEXT_PARTIAL_MROPE_SITE,
                qwen35_text_rope_route(
                    Qwen35TextRopeRoute::PARTIAL_MROPE),
                chunk.idx);
            rpu_launch_partial_mrope_spm_kernel(
                buf, buf, cos_ptr, sin_ptr, cos_sin_start, seq_len, heads, hd, rotary_dim_, tp);
        }
    };

    if (use_action_nvfp4_qk_norm_rope(seq_len,ctx().position)) {
        launch_action_nvfp4_qk_norm_rope(layer_idx,chunk,
            use_prefill_rope ? prefill_cos_ : cos_,
            use_prefill_rope ? prefill_sin_ : sin_,cos_sin_start);
    } else if (has_qk_norm_) {
        int64_t local_kv_heads = nkv / tp;
        // Q: norm + partial RoPE stay in-place because the partial kernels write
        // only the rotary span. The legacy full-RoPE path remains out-of-place.
        if (has_mrope_) {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q"), addr(0, "q"),
                layer_addr(layer_idx, 0, "q_norm_w"),
                seq_len * local_q_heads, hd, eps_, rmsnorm_route(seq_len * local_q_heads, hd), auxiliary_cores);
            emit_partial_rope(addr(0, "q"), local_q_heads);
        } else {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q"), addr(0, "output"),
                layer_addr(layer_idx, 0, "q_norm_w"),
                seq_len * local_q_heads, hd, eps_, rmsnorm_route(seq_len * local_q_heads, hd), auxiliary_cores);
            consume_manifest_route(
                FmbRouteFamily::ROPE,
                QWEN35_TEXT_Q_NORM_ROPE_SITE,
                qwen35_text_rope_route(Qwen35TextRopeRoute::ROPE_SPM),
                chunk.idx);
            rpu_launch_rope_spm_kernel(
                addr(0, "output"), addr(0, "q"), cos_ptr, sin_ptr,
                seq_len, local_q_heads, hd, cos_sin_start, auxiliary_cores);
        }
        // K: norm + rope (same as Q), ONLY when each core owns >=1 whole head.
        // No GQA weight replication: attn runs on tp = min(NUM_CORES, nkv) cores,
        // so local_kv_heads = nkv / tp == 1 (one whole KV head per active core).
        // The >0 guard is belt-and-suspenders for any nkv % tp != 0 edge case.
        if (owner_cache) {
            launch_action_k_owner_norm_rope(layer_idx, chunk);
        } else if (local_kv_heads > 0) {
            if (has_mrope_) {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "k"), addr(0, "k"),
                    layer_addr(layer_idx, 0, "k_norm_w"),
                    seq_len * local_kv_heads, hd, eps_, rmsnorm_route(seq_len * local_kv_heads, hd), auxiliary_cores);
                emit_partial_rope(addr(0, "k"), local_kv_heads);
            } else {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "k"), addr(0, "input_norm"),
                    layer_addr(layer_idx, 0, "k_norm_w"),
                    seq_len * local_kv_heads, hd, eps_, rmsnorm_route(seq_len * local_kv_heads, hd), auxiliary_cores);
                consume_manifest_route(
                    FmbRouteFamily::ROPE,
                    QWEN35_TEXT_K_NORM_ROPE_SITE,
                    qwen35_text_rope_route(Qwen35TextRopeRoute::ROPE_SPM),
                    chunk.idx);
                rpu_launch_rope_spm_kernel(
                    addr(0, "input_norm"), addr(0, "k"), cos_ptr, sin_ptr,
                    seq_len, local_kv_heads, hd, cos_sin_start, auxiliary_cores);
            }
        }
    } else {
        consume_manifest_route(
            FmbRouteFamily::ROPE, QWEN35_TEXT_Q_ROPE_SITE,
            qwen35_text_rope_route(Qwen35TextRopeRoute::ROPE_SPM),
            chunk.idx);
        rpu_launch_rope_spm_kernel(
            addr(0, "q"), addr(0, "q"), cos_ptr, sin_ptr,
            seq_len, local_q_heads, hd, cos_sin_start, auxiliary_cores);
        consume_manifest_route(
            FmbRouteFamily::ROPE, QWEN35_TEXT_K_ROPE_SITE,
            qwen35_text_rope_route(Qwen35TextRopeRoute::ROPE_SPM),
            chunk.idx);
        rpu_launch_rope_spm_kernel(
            addr(0, "k"), addr(0, "k"), cos_ptr, sin_ptr,
            seq_len, 1, local_kv_dim, cos_sin_start, auxiliary_cores);
    }

    // Phase 4: KV-cache insert + SDPA + O_proj (P3: offsets via addr_offset)
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    const int64_t kv_insert_pos =
        (action_mode_ ? action_prefix_lens_[layer_idx] : ctx().position)
        + chunk.offset;
    const KvInsertSegmentPlan kv_plan = [&] {
        if (!ctx().has_complete_physical_manifest()) {
            TORCH_CHECK(
                z2_bound_,
                "Qwen3.5 production KV insert requires a COMPLETE physical "
                "descriptor");
            return rpu_resolve_kvinsert_segment_plan_auto(
                kv_insert_pos, seq_len, seq_len, tp, nkv, hd,
                QWEN35_TEXT_KV_CAPABILITIES);
        }
        const int64_t invocation = action_mode_
            ? (action_prefix_lens_[layer_idx] > 0 ? 1 : 0)
            : chunk.idx;
        const auto& kv_route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, kFullAttentionKvInsertSite,
            invocation);
        const KvInsertSegmentPlan template_plan =
            restore_kvinsert_plan(
                kFullAttentionKvInsertSite, kv_route.arguments, tp, nkv, hd);
        const bool dynamic_position =
            (kv_route.flags & KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION) != 0;
        TORCH_CHECK(
            !dynamic_position ||
                (!action_mode_ && seq_len == 1 &&
                 template_plan.segment(0).position == 0),
            "Qwen3.5 dynamic KV position is only valid for a canonical "
            "single-token text decode template");
        KvInsertSegmentPlan resolved_plan = dynamic_position
            ? rpu_rebase_kvinsert_segment_plan_position(
                  template_plan, kv_insert_pos, tp, nkv, hd)
            : template_plan;
        TORCH_CHECK(
            resolved_plan.logical_rows() == seq_len &&
                resolved_plan.physical_rows() == seq_len &&
                resolved_plan.segment(0).position == kv_insert_pos,
            "Qwen3.5 KV descriptor geometry drift at layer ", layer_idx,
            ", invocation ", invocation);
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, kFullAttentionKvInsertSite,
            static_cast<int64_t>(template_plan.route()), kv_route.flags,
            kv_route.arguments, invocation);
        return resolved_plan;
    }();
    if (owner_cache) {
        launch_action_owner_cache_insert(layer_idx,chunk,kv_plan);
    } else {
        rpu_launch_insert_kvcache_spm_unified_with_plan(
            k_cache, v_cache,
            addr_offset("k").value, addr_offset("v").value,
            nkv, hd, tp,
            /*k_cache_batch_offset_elems=*/0,
            /*v_cache_batch_offset_elems=*/0,
            // Action has no padded physical tail: the frozen plan's physical rows
            // equal its logical horizon. Z2 also has no padded SPM tail.
            /*spm_rows=*/0, kv_plan);
    }

    int64_t kv_seq_len = action_mode_
        ? action_prefix_lens_[layer_idx] + ctx().seq_len
        : chunk.kv_seq_len;
    bool sdpa_causal = ctx().is_causal && (seq_len > 1);
    const bool raw_spm = ctx().attention_policy ==
        AttentionExecutionPolicy::SPM_KV_BY_MHA;
    const int64_t attention_site = raw_spm
        ? kFullAttentionRawSpmSite : kFullAttentionSite;
    const int64_t attention_invocation = action_mode_ ? 0 : chunk.idx;
    const FmbRouteManifestEntry* attention_route = nullptr;
    std::vector<int64_t> attention_arguments;
    if (ctx().has_complete_physical_manifest()) {
        attention_route = &ctx().find_physical_route(
            FmbRouteFamily::ATTENTION, attention_site,
            attention_invocation);
        attention_arguments = action_mode_ || seq_len == 1
            ? std::vector<int64_t>{seq_len}
            : std::vector<int64_t>{
                  /*batch=*/1, seq_len, kv_seq_len, nq, nkv, hd, tp,
                  sdpa_causal ? 1 : 0};
        TORCH_CHECK(
            (raw_spm && attention_route->flags == 0) ||
                (!raw_spm && attention_route->flags != 0),
            "Qwen3.5 attention descriptor lacks its exact RAW_SPM or "
            "DDR_REQUIRED reason");
    }
    if (raw_spm) {
        if (attention_route != nullptr) {
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION,
                kFullAttentionRawSpmSite,
                static_cast<int64_t>(
                    AttentionExecutionPolicy::SPM_KV_BY_MHA),
                attention_route->flags, attention_arguments,
                attention_invocation);
        }
        TORCH_CHECK(
            !action_mode_ && sdpa_causal &&
                sdpa_by_mha_spm_is_valid(
                    /*batch=*/1, seq_len, kv_seq_len, nq, nkv, hd, tp,
                    /*MASK_LTM=*/1),
            "Qwen3.5 RAW_SPM attention escaped its exact P0 single-chunk "
            "capability");
        rpu_launch_v_transpose_spm(
            addr(0, "v"), addr(0, "sdpa_tmp"),
            /*batch=*/1, kv_seq_len, nkv, hd, tp);
        rpu_launch_sdpa_by_mha_spm(
            addr(0, "q"), addr(0, "k"), addr(0, "sdpa_tmp"),
            addr(0, "output"), /*mask_spm=*/0, /*MASK_LTM=*/1,
            1.0 / std::sqrt(static_cast<double>(hd)),
            /*batch=*/1, seq_len, kv_seq_len, nq, nkv, hd, tp);
    } else {
        if (attention_route != nullptr) {
            ctx().consume_physical_route(
                FmbRouteFamily::ATTENTION, kFullAttentionSite,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                attention_route->flags, attention_arguments,
                attention_invocation);
        }
        // DDR SDPA on tp cores; virtual_num_cores=NUM_CORES governs the
        // KV-cache 7-D swizzle.
        rpu_launch_sdpa_spm_dispatch(
            sdpa_kernel_, k_cache, v_cache, sdpa_causal ? 1 : 0,
            c10::nullopt,
            addr_offset("q").value, addr_offset("output").value,
            addr_offset("sdpa_tmp").value, 0,
            seq_len, nq, nkv, hd, kv_seq_len, tp, NUM_CORES);
    }

    // Gated-attention output gate (Qwen3.5 attn_output_gate=true):
    //   output = sigmoid(gate) ⊙ output, before o_proj.
    if (use_action_attn_sigmoid_mul(seq_len)) {
        launch_action_attn_sigmoid_mul(layer_idx,chunk);
    } else {
        int64_t gate_elems = seq_len * (nq / tp) * hd;
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "attn_gate"), addr(0, "attn_gate"),
            gate_elems, ValuOpType::SIGMOID, /*is_gelu=*/false, tp);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "attn_gate"), addr(0, "output"), addr(0, "output"), gate_elems, ValuOpType::MUL, c10::Half(1.0f), auxiliary_cores);
    }
    // O-proj row-partition over tp cores; reduce to the owner below.
    consume_manifest_route(
        FmbRouteFamily::ALL_REDUCE,
        QWEN35_TEXT_FULL_PREPARE_ALL_REDUCE_SITE,
        static_cast<int64_t>(
            Qwen35TextAllReduceRoute::PREPARE_RING_INPUT),
        chunk.idx);
    rpu_prepare_ring_all_reduce_input(
        addr(0, "oproj"), seq_len, h, tp, num_cores());
    if (use_action_nvfp4_o_resident(seq_len)) {
        launch_action_nvfp4_o_resident(layer_idx, chunk);
    } else if (use_action_o_resident(seq_len)) {
        launch_action_o_resident(layer_idx, chunk);
    } else {
        launch_linear(
            chunk.idx,
            addr(0, "output"), lw.o_w, addr(0, "oproj"), seq_len, h, nq * hd,
            0, tp, 0, lw.o_ws, 1, layer_idx);
    }

    // Phase 5: exact Action tail keeps full raw and full normalized outputs.
    if (action_norm_emitted && use_action_tail(seq_len)) {
        launch_action_tail(layer_idx,chunk);
        *action_norm_emitted = true;
    } else if (text_owner_norm_emitted && ctx().has_complete_physical_manifest() &&
        use_text_owner_rms(ctx().stage_plan,ctx().seq_len,
            ctx().physical_manifest().logical_length,ctx().position,ctx().batch_size)) {
        launch_text_owner_rms(layer_idx,chunk);
        // Publish only after COMPLETE consumption and successful producer launch.
        *text_owner_norm_emitted = true;
    } else if (use_action_ring_rtc_residual(seq_len)) {
        launch_action_ring_rtc_residual(
            addr(0, "oproj"), addr(0, "residual1"), addr(0, "residual2"),
            QWEN35_TEXT_FULL_ALL_REDUCE_SITE, chunk);
    } else {
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_TEXT_FULL_ALL_REDUCE_SITE,
            text_ring_route(seq_len, h), chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "oproj"), layer_input_residual_addr(layer_idx, chunk),
            addr(0, "residual2"), seq_len, h,
            tp, num_cores());
        if (action_mode_) {
            apply_adaptive_residual_gate(
                addr(0, "residual2"), addr(0, "residual1"), chunk);
        }
    }

    // Ordinary all_reduce leaves full raw in input_norm. The selected owner
    // path leaves only original owner32 raw rows there and full normalized in
    // residual1; its local output flag tells the shared tail to skip postnorm.
}

// ── build_gdn: the unified Gated-DeltaNet token mixer. Shared setup/proj/tail + an
//    if(decode) split for the divergent conv + delta-rule core (decode = recurrent
//    single step; prefill = chunked delta-rule). Dispatched by build_layer_subgraph.
void Qwen3_5Model::build_gdn(int layer_idx, const ChunkInfo& chunk, bool decode,
                            bool* text_owner_norm_emitted) {
    const auto& lw = layer_weights_[layer_idx];
    const int64_t H = gdn_nvh_, Dk = gdn_dk_, Dv = gdn_dv_;
    const int64_t conv_dim = gdn_conv_dim_, Kc = gdn_kc_;
    const int64_t nkh = (conv_dim - H * Dv) / (2 * Dk), rep = H / nkh;
    const int64_t key_dim = nkh * Dk, value_dim = H * Dv, hidden = hidden_size();
    const int     nc = gdn_tp();
    const int64_t Hc = H / nc, Hc_kv = nkh / nc, lval = value_dim / nc, lkey = key_dim / nc;
    const int64_t lconv = conv_dim / nc;                          // per-core conv channels
    const int64_t n_bg = ((Hc + 15) / 16) * 16, N_bg = n_bg * nc; // padded b/a width (16/core)
    const int64_t L = decode ? 1 : chunk.len;   // seq_len: decode 1 token / prefill chunk
    TORCH_CHECK(H % nc == 0 && conv_dim % nc == 0 && value_dim % nc == 0 && nkh % nc == 0,
                "build_gdn: H/conv_dim/value_dim/nkh must divide the GDN core count");
    at::Tensor* st = gdn_state_for_layer(layer_idx);
    at::Tensor* cv = conv_state_for_layer(layer_idx);
    TORCH_CHECK(st && cv && st->defined() && cv->defined(),
                "build_gdn: missing recurrent/conv state cache for layer ", layer_idx);
    auto D = [&](const char* n) { return addr(0, n); };
    const bool initial_prefill_chunk =
        !decode && ctx().position == 0 && chunk.offset == 0;

    // Buffer selectors: decode writes its own gdn_* buffers; prefill writes the conv
    // xpad x-region (q/k at row qx, v at vx) so the halo sees [conv_state ; conv_in].
    const int64_t hist = Kc - 1;
    const uint32_t qx = (uint32_t)(hist * lkey * 2), vx = (uint32_t)(hist * lval * 2);
    const uint32_t q_out = decode ? D("gdn_q") : D("gdn_c_q_pad") + qx;
    const uint32_t k_out = decode ? D("gdn_k") : D("gdn_c_k_pad") + qx;
    const uint32_t v_out = decode ? D("gdn_v") : D("gdn_c_v_pad") + vx;
    const uint32_t z_out = decode ? D("gdn_z") : D("gdn_c_z");
    const uint32_t b_out = decode ? D("gdn_b") : D("gdn_c_b");
    const uint32_t a_out = decode ? D("gdn_a") : D("gdn_c_a");

    // ── DMA the M-independent weights/state into SPM. ──
    rpu_launch_ddr_broadcast_spm_dma(lw.gdn_norm_w.data_ptr<c10::Half>(),    Dv,      D("gdn_normw"), nc);
    if (decode)
        rpu_launch_ddr_broadcast_spm_dma(
            gdn_zero_.data_ptr<c10::Half>(), Dk * Dv, D("gdn_zero"), nc);
    // gdn_Al: decode = RAW A_log (recurrent device-exps it); prefill = precomputed neg_exp_A.
    const int64_t scalar_slots = qwen35_gdn_scalar_slots(H, nc);
    if (decode) {
        rpu_launch_ddr_scatter_spm_dma(lw.gdn_A_log.data_ptr<c10::Half>(), scalar_slots, scalar_slots * DWIDTH, D("gdn_Al"), nc);
    } else {
        TORCH_CHECK(gdn_neg_exp_A_[layer_idx].defined(), "build_gdn: neg_exp_A missing for layer ", layer_idx);
        rpu_launch_ddr_scatter_spm_dma(gdn_neg_exp_A_[layer_idx].data_ptr<c10::Half>(), scalar_slots, scalar_slots * DWIDTH, D("gdn_Al"), nc);
    }
    rpu_launch_ddr_scatter_spm_dma(lw.gdn_dt_bias.data_ptr<c10::Half>(), scalar_slots, scalar_slots * DWIDTH, D("gdn_dt"), nc);
    if (decode) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_GDN_STATE_LOAD_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
            chunk.idx);
        rpu_launch_ddr_scatter_spm_dma_mutable(
            &gdn_state_live_addr_[layer_idx], 0, Hc * Dk * Dv,
            Hc * Dk * Dv * 2, D("gdn_state"), nc);
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_GDN_CONV_DECODE_LOAD_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
            chunk.idx);
        rpu_launch_ddr_scatter_spm_dma_mutable(
            &conv_state_live_addr_[layer_idx], 0, Kc * lconv,
            Kc * lconv * 2, D("gdn_cs"), nc);
    } else if (initial_prefill_chunk) {
        rpu_launch_fill_spm_kernel(
            D("gdn_cs"), Kc * lconv, c10::Half(0.0f), nc);
    } else {
        // Prefill consumes the conv carry immediately in phase 2. The recurrent
        // state is loaded only at phase 16, after earlier scratch is dead.
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_GDN_CONV_PREFILL_LOAD_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
            chunk.idx);
        rpu_launch_ddr_scatter_spm_dma_mutable(
            &conv_state_live_addr_[layer_idx], 0, Kc * lconv,
            Kc * lconv * 2, D("gdn_cs"), nc);
    }
    bool load_chunk_masks = !decode;
    if (!decode && ctx().has_complete_physical_manifest() && !action_mode_ && !z2_bound_) {
        const auto& route = ctx().find_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_TEXT_GDN_MASK_RESIDENCY_SITE);
        TORCH_CHECK(route.selector == 1 || route.selector == 2,
                    "GDN mask residency requires per-layer or forward-retained schedule");
        const auto arguments = gdn_mask_schedule_arguments(ctx().stage_plan);
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_TEXT_GDN_MASK_RESIDENCY_SITE,
            route.selector, /*resolved_flags=*/0, arguments);
        // This is an execution-position predicate, never a host 'already loaded'
        // cache: every Graph execution records/replays its first pair of DMAs.
        load_chunk_masks = route.selector == 1 ||
            (layer_idx == arguments[3] && chunk.idx == 0);
    }
    if (load_chunk_masks) {  // fixed 64x64 masks; all GDN consumers only read them
        rpu_launch_ddr_broadcast_spm_dma(gdn_tril_.data_ptr<c10::Half>(),   64 * 64, D("gdn_c_tril"),   nc);
        rpu_launch_ddr_broadcast_spm_dma(gdn_strict_.data_ptr<c10::Half>(), 64 * 64, D("gdn_c_strict"), nc);
    }

    // ── in_proj: q/k/v/z col-parallel + N_bg-padded b/a. input_layernorm is at the layer
    //    level → "input_norm"; decode writes its own buffers, prefill the conv xpad region. ──
    launch_linear(chunk.idx,
                  D("input_norm"), lw.gdn_q_w, q_out, L, key_dim, hidden,
                  1, nc, 0, lw.gdn_q_ws);
    launch_linear(chunk.idx,
                  D("input_norm"), lw.gdn_k_w, k_out, L, key_dim, hidden,
                  1, nc, 0, lw.gdn_k_ws);
    launch_linear(chunk.idx,
                  D("input_norm"), lw.gdn_v_w, v_out, L, value_dim, hidden,
                  1, nc, 0, lw.gdn_v_ws);
    launch_linear(chunk.idx,
                  D("input_norm"), lw.gdn_in_z_w, z_out, L, value_dim, hidden,
                  1, nc, 0, lw.gdn_in_z_ws);
    launch_linear(chunk.idx,
                  D("input_norm"), lw.gdn_b_bg_w, b_out, L, N_bg, hidden,
                  1, nc, 0, lw.gdn_b_bg_ws);
    launch_linear(chunk.idx,
                  D("input_norm"), lw.gdn_a_bg_w, a_out, L, N_bg, hidden,
                  1, nc, 0, lw.gdn_a_bg_ws);

    // ── Divergent token mixer: decode = recurrent single step / prefill = chunked delta-rule.
    //    Both converge on the shared phase-6 tail below. ──
    if (decode) {
        // 2) causal conv1d update — per-path on the path-major conv-state window. gdn_cs is
        //    reinterpreted as [q:(Kc,lkey) | k:(Kc,lkey) | v:(Kc,lval)]; each path's conv is
        //    in-place (out = row Kc-1), then SiLU into its own activation buffer.
        const uint32_t cs_q = D("gdn_cs");
        const uint32_t cs_k = D("gdn_cs") + (uint32_t)(Kc * lkey * 2);
        const uint32_t cs_v = D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2);
        rpu_launch_fla_conv1d_spm_kernel(cs_q, D("gdn_q"), lw.gdn_conv_q_w.data_ptr<c10::Half>(), 1, lkey, Kc, nc);
        rpu_launch_fla_conv1d_spm_kernel(cs_k, D("gdn_k"), lw.gdn_conv_k_w.data_ptr<c10::Half>(), 1, lkey, Kc, nc);
        rpu_launch_fla_conv1d_spm_kernel(cs_v, D("gdn_v"), lw.gdn_conv_v_w.data_ptr<c10::Half>(), 1, lval, Kc, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_q + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_qa"), lkey, ValuOpType::SILU, false, nc, unary_precision());
        rpu_launch_eltwise_unary_spm_kernel(cs_k + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_ka"), lkey, ValuOpType::SILU, false, nc, unary_precision());
        rpu_launch_eltwise_unary_spm_kernel(cs_v + (uint32_t)((Kc - 1) * lval * 2), D("gdn_va"), lval, ValuOpType::SILU, false, nc, unary_precision());

        // 3) per-path q/k/v already separated by the proj (no fused split needed).
        const uint32_t a_q = D("gdn_qa");
        const uint32_t a_k = D("gdn_ka");
        const uint32_t a_v = D("gdn_va");

        // 3b) GQA expand: tile q/k from Hc_kv heads → Hc heads (repeat_interleave via
        // the [Hc_kv,1,Dk] middle-dim tile: head h → out 2h,2h+1). The recurrent
        // l2norms q/k internally and l2norm(copy)==copy(l2norm) per head, so tiling
        // first is exact. Skipped when rep==1 (non-GQA: a_q/a_k already have Hc heads).
        uint32_t rq = a_q, rk = a_k;
        if (rep > 1) {
            rpu_launch_tile_spm_kernel(a_q, D("gdn_q_rep"), Hc_kv, 1, Dk, 1, rep, 1, nc);
            rpu_launch_tile_spm_kernel(a_k, D("gdn_k_rep"), Hc_kv, 1, Dk, 1, rep, 1, nc);
            rq = D("gdn_q_rep");
            rk = D("gdn_k_rep");
        }

        // 4) l2norm + gating + recurrent delta-rule (Hc heads/core). gdn_Al = RAW
        //    A_log (the recurrent device-exps it); consumes raw a/b/dt; state updated in place.
        rpu_emit_gdn_recurrent_multihead_seq(
            rq, rk, a_v, D("gdn_state"), D("gdn_p"), D("gdn_scratch"),
            D("gdn_delta"), D("gdn_zero"),
            D("gdn_Al"), D("gdn_a"), D("gdn_b"), D("gdn_dt"),
            D("gdn_gexp"), D("gdn_beta"), D("gdn_gs"), D("gdn_out"),
            Hc, Dk, Dv, nc);
        // decode leaves the updated conv window in gdn_cs; the shared tail scatters it → cache.
    } else {

    // ── Route B (arbitrary prefill length): the adapter pads the whole prefill seq up to a multiple
    //    of 64, so L=chunk.len is ALWAYS 64-aligned and the chunk core runs on L directly (full-attn
    //    SDPA + GDN both need 64-alignment). Lv = the REAL (non-pad) tokens in this chunk =
    //    clamp(valid_prefill_len_ - chunk.offset, 0, L); it equals L for a non-tail chunk or when
    //    unset. conv/halo run on Lv (→ conv_state cso = last-3 REAL tokens), and rows [Lv:L] of
    //    k/β/g are zero-filled so the pad tokens contribute NOTHING to the carried recurrent_state
    //    (exact). Pad rows' output is garbage; the adapter slices it off the padded logits. ──
    TORCH_CHECK(L % 64 == 0, "build_gdn: prefill chunk.len (", L,
                ") must be a multiple of 64 (route B: adapter pads the prefill sequence)");
    int64_t Lv = L;
    if (valid_prefill_len_ > 0)
        Lv = std::max<int64_t>(0, std::min<int64_t>(L, valid_prefill_len_ - chunk.offset));
    TORCH_CHECK(Lv >= 1, "build_gdn: prefill chunk has 0 real tokens (Lv=0, chunk.offset=",
                chunk.offset, ", valid_prefill_len=", valid_prefill_len_, ") — bad chunk plan/padding");

    // ── Phase 2: causal conv1d (prefill, per-path) + SiLU → gdn_c_query/key/value. ──
    //   pad rows [0:hist] = prev-chunk conv state (fresh prefill = 0). The
    //   no-inplace kernel leaves pad intact and writes activated inputs to
    //   gdn_c_query/key/value through the following SiLU.
    if (Lv == 1) {
        // Lv==1 edge (last chunk holds a single real token; real_len ≡ 1 mod cs, > cs): the prefill
        // conv kernel is L>1 only, so use the DECODE single-token conv. gdn_cs already holds the conv
        // history (loaded from the cache at setup, before the decode/prefill split); the token's
        // q/k/v projection sits at gdn_c_*_pad+qx/vx (proj wrote row hist). 1:1 with the decode
        // branch: the conv output lands in gdn_cs[Kc-1] (SiLU → gdn_c_*[0]) and gdn_cs is updated in
        // place to [h1,h2,x,out] so its [0:3] = last-3 real tokens; the shared tail scatters gdn_cs
        // to the cache. Rows [1:L] are zero-filled below, so the chunk core sees [silu(conv); 0…].
        const uint32_t cs_q = D("gdn_cs");
        const uint32_t cs_k = D("gdn_cs") + (uint32_t)(Kc * lkey * 2);
        const uint32_t cs_v = D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2);
        rpu_launch_fla_conv1d_spm_kernel(cs_q, D("gdn_c_q_pad") + qx, lw.gdn_conv_q_w.data_ptr<c10::Half>(), 1, lkey, Kc, nc);
        rpu_launch_fla_conv1d_spm_kernel(cs_k, D("gdn_c_k_pad") + qx, lw.gdn_conv_k_w.data_ptr<c10::Half>(), 1, lkey, Kc, nc);
        rpu_launch_fla_conv1d_spm_kernel(cs_v, D("gdn_c_v_pad") + vx, lw.gdn_conv_v_w.data_ptr<c10::Half>(), 1, lval, Kc, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_q + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_c_query"), lkey, ValuOpType::SILU, false, nc, unary_precision());
        rpu_launch_eltwise_unary_spm_kernel(cs_k + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_c_key"),   lkey, ValuOpType::SILU, false, nc, unary_precision());
        rpu_launch_eltwise_unary_spm_kernel(cs_v + (uint32_t)((Kc - 1) * lval * 2), D("gdn_c_value"), lval, ValuOpType::SILU, false, nc, unary_precision());
    } else {
        // pad[0:hist] = prev-chunk conv carry, copied from gdn_cs (loaded from the DDR cache at
        // setup; fresh chunk 0 = 0 since the cache inits to 0). gdn_cs is path-major, SAME offsets
        // as decode's cs_q/cs_k/cs_v (q@+0, k@+Kc·lkey, v@+Kc·2lkey). The conv below writes the NEW
        // carry back into these gdn_cs regions (cso pointers) and the shared tail scatters gdn_cs →
        // cache, so conv_state carries across prefill chunks and into decode. (SPM→SPM MAX-copy.)
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_cs"),                               D("gdn_cs"),                               D("gdn_c_q_pad"), hist * lkey, ValuOpType::MAX, c10::Half(1.0f), nc);
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_cs") + (uint32_t)(Kc * lkey * 2),     D("gdn_cs") + (uint32_t)(Kc * lkey * 2),     D("gdn_c_k_pad"), hist * lkey, ValuOpType::MAX, c10::Half(1.0f), nc);
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2), D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2), D("gdn_c_v_pad"), hist * lval, ValuOpType::MAX, c10::Half(1.0f), nc);
        // cso (new conv carry = last hist inputs; only the top segment stores it) is written
        // straight into gdn_cs at the same path-major offsets decode uses → the shared tail's
        // gdn_cs→cache scatter carries it to the next chunk / decode step.
        rpu_launch_fla_conv1d_prefill_noinplace_spm_kernel(D("gdn_c_q_pad"), D("gdn_c_query"), D("gdn_cs"),                               lw.gdn_conv_q_w.data_ptr<c10::Half>(), 1, lkey, Kc, Lv, nc);
        rpu_launch_fla_conv1d_prefill_noinplace_spm_kernel(D("gdn_c_k_pad"), D("gdn_c_key"),   D("gdn_cs") + (uint32_t)(Kc * lkey * 2),     lw.gdn_conv_k_w.data_ptr<c10::Half>(), 1, lkey, Kc, Lv, nc);
        rpu_launch_fla_conv1d_prefill_noinplace_spm_kernel(D("gdn_c_v_pad"), D("gdn_c_value"), D("gdn_cs") + (uint32_t)(Kc * 2 * lkey * 2), lw.gdn_conv_v_w.data_ptr<c10::Half>(), 1, lval, Kc, Lv, nc);
        rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_query"), D("gdn_c_query"), L * lkey, ValuOpType::SILU, false, nc, unary_precision());
        rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_key"),   D("gdn_c_key"),   L * lkey, ValuOpType::SILU, false, nc, unary_precision());
        rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_value"), D("gdn_c_value"), L * lval, ValuOpType::SILU, false, nc, unary_precision());
    }
    // The conv carry is complete after phase 2 and is not read again in prefill.
    consume_manifest_route(
        FmbRouteFamily::MUTABLE_DMA,
        QWEN35_TEXT_GDN_CONV_PREFILL_STORE_DMA_SITE,
        qwen35_text_dma_route(
            Qwen35TextMutableDmaRoute::SPM_SCATTER_TO_DDR),
        chunk.idx);
    rpu_launch_spm_scatter_ddr_dma_mutable(
        D("gdn_cs"), &conv_state_live_addr_[layer_idx], 0,
        Kc * lconv, Kc * lconv * 2, nc);
    // ── Phases 3–17: prefill chunked delta-rule core. Falls through to the shared tail
    //    below (gated-norm + out_proj + state/conv writeback + AllReduce residual), same as the
    //    decode branch — both leave the post-mixer stream in residual2 for build_layer_subgraph. ──
    const int64_t C = 64, N = L / C;            // GDN chunk size / sub-chunk count (L is 64-aligned)
    const int64_t vg_c = H / nc;                // beta/g heads/core (== Hc)
    const uint32_t q_rep_buf = D("gdn_c_q_rep");
    const uint32_t k_rep_buf = D("gdn_c_k_rep");
    const uint32_t q_h_buf = D("gdn_c_q_h");
    const uint32_t decay_buf = D("gdn_c_decay");
    // G05 uses the BASE payload's validated local-address mode: one launch
    // applies this same slice independently to each core. Other profiles keep
    // the original eight global-address launches with their original rebasing.
    const bool g05_local_slice = use_g05_gdn_local_slice(L);
    if (g05_local_slice) {
        TORCH_CHECK(ctx().has_complete_physical_manifest(),
                    "G05 local GDN slices require a COMPLETE physical schedule");
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_TEXT_GDN_LOCAL_SLICE_SITE,
            1, 0, g05_gdn_local_slice_arguments(L), chunk.idx);
    }
    auto pc_addr = [&](uint32_t a0, int cc) -> uint32_t {
        return a0 - SPM_ALLOC.addr(0, 0) + SPM_ALLOC.addr(cc, 0);
    };
    auto slice_pc = [&](uint32_t s, uint32_t d, const std::vector<int64_t>& is,
                        const std::vector<int64_t>& os, const std::vector<int64_t>& b) {
        const int launches = g05_local_slice ? 1 : nc;
        for (int cc = 0; cc < launches; ++cc)
            rpu_launch_slice_spm_kernel(
                g05_local_slice ? s : pc_addr(s, cc),
                g05_local_slice ? d : pc_addr(d, cc),
                is, os, b, g05_local_slice ? nc : 1, g05_local_slice);
    };

    // ===== Phase 3: prep (beta/g) + q/k l2norm + GQA tile =====
    slice_pc(D("gdn_c_b"), D("gdn_c_beta"), {L, n_bg}, {L, vg_c}, {0, 0});
    slice_pc(D("gdn_c_a"), D("gdn_c_g"),    {L, n_bg}, {L, vg_c}, {0, 0});
    rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_beta"), D("gdn_c_beta"), L * vg_c, ValuOpType::SIGMOID, false, nc);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_dt"), D("gdn_c_g"), D("gdn_c_g"), L, vg_c, c10::Half(1.0f), ValuOpType::ADD, false, nc);  // +dt_bias
    rpu_launch_eltwise_unary_spm_kernel(
        D("gdn_c_g"), D("gdn_c_g"), L * vg_c, ValuOpType::SOFTPLUS,
        false, nc);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_Al"), D("gdn_c_g"), D("gdn_c_g"), L, vg_c, c10::Half(1.0f), ValuOpType::MUL, false, nc);  // g *= neg_exp_A
    rpu_launch_l2norm_spm_kernel(D("gdn_c_query"), D("gdn_c_query"), L * Hc_kv, Dk, 1e-6, nc);
    rpu_launch_l2norm_spm_kernel(D("gdn_c_key"),   D("gdn_c_key"),   L * Hc_kv, Dk, 1e-6, nc);
    uint32_t rq = D("gdn_c_query"), rk = D("gdn_c_key");
    const uint32_t a_v = D("gdn_c_value");
    if (rep > 1) {
        rpu_launch_tile_spm_kernel(D("gdn_c_query"), q_rep_buf, L * Hc_kv, 1, Dk, 1, rep, 1, nc);
        rpu_launch_tile_spm_kernel(D("gdn_c_key"),   k_rep_buf, L * Hc_kv, 1, Dk, 1, rep, 1, nc);
        rq = q_rep_buf;
        rk = k_rep_buf;
    }

    // Zero the pad rows [Lv:L] of the chunk-core inputs (route B: the last chunk's [valid:L] are pad
    // tokens). k/β/g=0 makes the pad positions contribute NOTHING to the carried recurrent_state
    // (exact, not approx); q/v zeroed too so no NaN leaks through the chunk attn (their outputs get
    // sliced off the padded logits by the adapter).
    if (Lv < L) {
        const int64_t np = L - Lv;
        rpu_launch_fill_spm_kernel(rq              + (uint32_t)(Lv * Hc * Dk * 2), np * Hc * Dk, c10::Half(0.0f), nc);
        rpu_launch_fill_spm_kernel(rk              + (uint32_t)(Lv * Hc * Dk * 2), np * Hc * Dk, c10::Half(0.0f), nc);
        rpu_launch_fill_spm_kernel(a_v             + (uint32_t)(Lv * lval * 2),    np * lval,    c10::Half(0.0f), nc);
        rpu_launch_fill_spm_kernel(D("gdn_c_beta") + (uint32_t)(Lv * vg_c * 2),    np * vg_c,    c10::Half(0.0f), nc);
        rpu_launch_fill_spm_kernel(D("gdn_c_g")    + (uint32_t)(Lv * vg_c * 2),    np * vg_c,    c10::Half(0.0f), nc);
    }

    // ===== Phases 4-8: q/k/v/beta/g transpose — N-major [N,Hc,C,*] =====
    const int64_t HN = Hc * N, HL = Hc * L, CC = C * C;
    const float   qk_scale = 1.0f / std::sqrt((float)Dk);
    // N-major setup transpose: token-major [N,C,Hc,Dl] -> [N,Hc,C,Dl] (= permute(0,2,1,3)),
    // realized as N independent 3D {1,0,2} transposes ([C,Hc,Dl]->[Hc,C,Dl]) — one per sub-chunk,
    // since there is no 4D SPM permute. Keeping N outermost turns every phase-5 sub-chunk read into
    // a pure base+offset view (no slice/insert). Phase 4's ops are all per-(h,n)-group batched ops
    // (HN batch), agnostic to (h,n) vs (n,h) ordering, so the result is bit-identical to head-major.
    auto permute_nmajor = [&](uint32_t in_base, uint32_t out_base, int64_t Dl) {
        for (int64_t ni = 0; ni < N; ++ni)
            rpu_launch_permute3d_spm_kernel(
                in_base  + (uint32_t)(ni * C  * Hc * Dl * 2),   // block ni: [C, Hc, Dl]
                out_base + (uint32_t)(ni * Hc * C  * Dl * 2),   // block ni: [Hc, C, Dl]
                C, Hc, Dl, {1, 0, 2}, nc);
    };
    permute_nmajor(rq,               q_h_buf,             Dk);
    permute_nmajor(rk,               D("gdn_c_k_h"),    Dk);
    permute_nmajor(a_v,              D("gdn_c_v_h"),    Dv);
    permute_nmajor(D("gdn_c_beta"),  D("gdn_c_beta_h"), 1);
    permute_nmajor(D("gdn_c_g"),     D("gdn_c_g_h"),    1);
    // ===== Phases 9-12: q scale, beta*k, beta*v, cumsum =====
    rpu_launch_eltwise_binary_scalar_spm_kernel(q_h_buf, c10::Half(qk_scale), q_h_buf, HL * Dk, ValuOpType::MUL, nc);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_beta_h"), D("gdn_c_k_h"), D("gdn_c_kbeta"), HL, Dk, c10::Half(1.0f), ValuOpType::MUL, false, nc);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_beta_h"), D("gdn_c_v_h"), D("gdn_c_v_h"), HL, Dv, c10::Half(1.0f), ValuOpType::MUL, false, nc);
    rpu_launch_cumsum_spm_kernel(D("gdn_c_g_h"), D("gdn_c_gcumsum"), D("gdn_c_ws"), HN, C, nc);

    // ===== Phase 13: decay matrix =====
    rpu_launch_fill_spm_kernel(decay_buf, HN * CC, c10::Half(0.0f), nc);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_gcumsum"), decay_buf, decay_buf, HN * C, C, c10::Half(1.0f), ValuOpType::ADD, false, nc);
    rpu_launch_eltwise_binary_Bx1xC_BxNxC_spm_kernel(D("gdn_c_gcumsum"), decay_buf, decay_buf, HN, C, C, c10::Half(1.0f), ValuOpType::SUB, true, nc);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_c_tril"), decay_buf, decay_buf, HN, CC, c10::Half(1.0f), ValuOpType::MUL, false, nc);
    rpu_launch_eltwise_unary_spm_kernel(decay_buf, decay_buf, HN * CC, ValuOpType::EXP, false, nc);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_c_tril"), decay_buf, decay_buf, HN, CC, c10::Half(1.0f), ValuOpType::MUL, false, nc);
    rpu_launch_bmm_spm_kernel(D("gdn_c_kbeta"), D("gdn_c_k_h"), D("gdn_c_attn"), C, C, Dk, HN, nc, BmmMode::RowByRow);
    rpu_launch_eltwise_binary_scalar_spm_kernel(D("gdn_c_attn"), c10::Half(-1.0f), D("gdn_c_attn"), HN * CC, ValuOpType::MUL, nc);
    rpu_launch_eltwise_binary_spm_kernel(D("gdn_c_attn"), decay_buf, D("gdn_c_attn"), HN * CC, ValuOpType::MUL, c10::Half(1.0f), nc);
    rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(D("gdn_c_strict"), D("gdn_c_attn"), D("gdn_c_attn"), HN, CC, c10::Half(1.0f), ValuOpType::MUL, false, nc);
    rpu_launch_unit_tril_inv_spm_kernel(D("gdn_c_attn"), D("gdn_c_attn"), HN, nc);
    rpu_launch_bmm_spm_kernel(D("gdn_c_attn"), D("gdn_c_v_h"), D("gdn_c_vnew"), C, Dv, C, HN, nc);

    // ===== Phase 15: recurrent-state coefficients =====
    rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_gcumsum"), D("gdn_c_gexp"), HL, ValuOpType::EXP, false, nc);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_gexp"), D("gdn_c_kbeta"), D("gdn_c_kbeta"), HL, Dk, c10::Half(1.0f), ValuOpType::MUL, false, nc);
    rpu_launch_bmm_spm_kernel(D("gdn_c_attn"), D("gdn_c_kbeta"), D("gdn_c_kcd"), C, Dk, C, HN, nc);

    // ===== Phase 16: per-chunk recurrent loop. gdn_state [Hc,Dk,Dv] carried across i. =====
    // The state is consumed only by this loop, so load or initialize it after
    // the earlier phase scratch is dead.
    if (initial_prefill_chunk) {
        rpu_launch_fill_spm_kernel(
            D("gdn_state"), Hc * Dk * Dv, c10::Half(0.0f), nc);
    } else {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_GDN_STATE_PREFILL_LOAD_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
            chunk.idx);
        rpu_launch_ddr_scatter_spm_dma_mutable(
            &gdn_state_live_addr_[layer_idx], 0, Hc * Dk * Dv,
            Hc * Dk * Dv * 2, D("gdn_state"), nc);
    }
    // N-major layout ([N,Hc,C,*]) makes each sub-chunk i a CONTIGUOUS [Hc,C,*] block, so the
    // per-chunk operands are pure base+offset views into the phase-4 buffers — no slice/insert.
    // a_qc/a_kc/a_vc/a_gexpc are mutated in place below, but each gdn_c_*[i] region is touched
    // only within iteration i and dead afterwards, so aliasing the source is exact. Only glast
    // (last timestep, C-axis) still needs a real slice — orthogonal to the N reorder.
    const std::vector<int64_t> sh_gl{Hc, C}, sl_gl{Hc, 1};
    for (int64_t i = 0; i < N; ++i) {
        const std::vector<int64_t> bC{0, C - 1};
        const uint32_t a_qc    = D("gdn_c_q_h")     + (uint32_t)(i * Hc * C * Dk * 2);
        const uint32_t a_kc    = D("gdn_c_k_h")     + (uint32_t)(i * Hc * C * Dk * 2);
        const uint32_t a_vc    = D("gdn_c_vnew")    + (uint32_t)(i * Hc * C * Dv * 2);
        const uint32_t a_kcdc  = D("gdn_c_kcd")     + (uint32_t)(i * Hc * C * Dk * 2);
        const uint32_t a_decc  = D("gdn_c_decay")   + (uint32_t)(i * Hc * C * C  * 2);
        const uint32_t a_gc    = D("gdn_c_gcumsum") + (uint32_t)(i * Hc * C * 2);
        const uint32_t a_gexpc = D("gdn_c_gexp")    + (uint32_t)(i * Hc * C * 2);
        const uint32_t a_outc  = D("gdn_c_coreout") + (uint32_t)(i * Hc * C * Dv * 2);
        rpu_launch_bmm_spm_kernel(a_qc, a_kc, D("gdn_c_attnc"), C, C, Dk, Hc, nc, BmmMode::RowByRow);
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_c_attnc"), a_decc, D("gdn_c_attnc"), Hc * C * C, ValuOpType::MUL, c10::Half(1.0f), nc);
        rpu_launch_bmm_spm_kernel(a_kcdc, D("gdn_state"), a_outc, C, Dv, Dk, Hc, nc);
        rpu_launch_eltwise_binary_spm_kernel(a_vc, a_outc, a_vc, Hc * C * Dv, ValuOpType::SUB, c10::Half(1.0f), nc);
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(a_gexpc, a_qc, a_qc, Hc * C, Dk, c10::Half(1.0f), ValuOpType::MUL, false, nc);
        rpu_launch_bmm_spm_kernel(a_qc, D("gdn_state"), a_outc, C, Dv, Dk, Hc, nc);
        rpu_launch_bmm_spm_kernel(D("gdn_c_attnc"), a_vc, D("gdn_c_loop_b"), C, Dv, C, Hc, nc);
        rpu_launch_eltwise_binary_spm_kernel(a_outc, D("gdn_c_loop_b"), a_outc, Hc * C * Dv, ValuOpType::ADD, c10::Half(1.0f), nc);
        slice_pc(a_gexpc, D("gdn_c_glast"), sh_gl, sl_gl, bC);
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_glast"), D("gdn_state"), D("gdn_state"), Hc, Dk * Dv, c10::Half(1.0f), ValuOpType::MUL, false, nc);
        slice_pc(a_gc, D("gdn_c_glast"), sh_gl, sl_gl, bC);
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(D("gdn_c_glast"), a_gc, a_gexpc, Hc, C, c10::Half(1.0f), ValuOpType::SUB, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(a_gexpc, a_gexpc, Hc * C, ValuOpType::EXP, false, nc);
        rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(a_gexpc, a_kc, a_kc, Hc * C, Dk, c10::Half(1.0f), ValuOpType::MUL, false, nc);
        rpu_launch_bmm_spm_kernel(a_kc, a_vc, D("gdn_c_loop_b"), Dk, Dv, C, Hc, nc, BmmMode::ColByCol);
        rpu_launch_eltwise_binary_spm_kernel(D("gdn_state"), D("gdn_c_loop_b"), D("gdn_state"), Hc * Dk * Dv, ValuOpType::ADD, c10::Half(1.0f), nc);
        // a_outc == gdn_c_coreout[i] (written in place above) — no insert needed.
    }
    consume_manifest_route(
        FmbRouteFamily::MUTABLE_DMA,
        QWEN35_TEXT_GDN_STATE_PREFILL_STORE_DMA_SITE,
        qwen35_text_dma_route(
            Qwen35TextMutableDmaRoute::SPM_SCATTER_TO_DDR),
        chunk.idx);
    rpu_launch_spm_scatter_ddr_dma_mutable(
        D("gdn_state"), &gdn_state_live_addr_[layer_idx], 0,
        Hc * Dk * Dv, Hc * Dk * Dv * 2, nc);
    // ===== Phase 17: final transpose =====
    // core_attn_out N-major [N,Hc,C,Dv] -> token-major [L,Hc,Dv] (= [N,C,Hc,Dv]),
    // N independent 3D {1,0,2} transposes ([Hc,C,Dv]->[C,Hc,Dv]), one per sub-chunk (no 4D permute).
    // Pad rows [Lv:L] carry zeroed output; the adapter slices them off the logits.
    for (int64_t ni = 0; ni < N; ++ni)
        rpu_launch_permute3d_spm_kernel(
            D("gdn_c_coreout") + (uint32_t)(ni * Hc * C * Dv * 2),   // block ni: [Hc, C, Dv]
            D("gdn_c_out_tok") + (uint32_t)(ni * C  * Hc * Dv * 2),  // block ni: [C, Hc, Dv]
            Hc, C, Dv, {1, 0, 2}, nc);
    }  // end prefill (chunked delta-rule)

    // ── Phases 18-20 tail (SHARED decode + prefill): RMSNormGated + out_proj
    //    (row-parallel) + recurrent-state writeback + AllReduce residual → residual2. 1:1 with
    //    modeling_qwen3_5.py:553-558. RMSNormGated is norm-THEN-gate
    //    operation: rms_norm the raw core_attn_out over Dv · norm_w,
    //    THEN · silu(z). core_attn_out is token-major [L,Hc,Dv] per core → M = L*Hc, N = Dv;
    //    decode's gdn_out [Hc,Dv] is already token-major (L=1). ──
    const uint32_t out_buf   = decode ? D("gdn_out")   : D("gdn_c_out_tok");
    const uint32_t gated_buf = decode ? D("gdn_gated") : out_buf;
    const uint32_t z_buf     = z_out;
    const uint32_t m_buf     = decode ? D("gdn_m")     : D("gdn_c_m");
    if (decode && !high_precision_) {
        rpu_launch_qwen3_5_rms_norm_gated(
            out_buf, z_buf, D("gdn_normw"), gated_buf, Hc, Dv, 1e-6, nc);
    } else {
        rpu_launch_rmsnorm_spm_kernel(
            out_buf, gated_buf, D("gdn_normw"), L * Hc, Dv, 1e-6, rmsnorm_route(L * Hc, Dv), nc);
        // Keep the proven prefill formula: gate = silu(z) * normed.
        rpu_launch_eltwise_unary_spm_kernel(
            z_buf, z_buf, L * lval, ValuOpType::SILU, false, nc, unary_precision());
        rpu_launch_eltwise_binary_spm_kernel(
            z_buf, gated_buf, gated_buf, L * lval, ValuOpType::MUL,
            c10::Half(1.0f), nc);
    }
    // out_proj (row-parallel): gated [L,value_dim] @ gdn_out_w → per-core partial m [L,hidden].
    // FULL K = value_dim (launcher splits local_k = value_dim/tp).
    if (nc != num_cores()) {
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_TEXT_GDN_PREPARE_ALL_REDUCE_SITE,
            static_cast<int64_t>(Qwen35TextAllReduceRoute::PREPARE_RING_INPUT),
            chunk.idx);
        rpu_prepare_ring_all_reduce_input(m_buf, L, hidden, nc, num_cores());
    }
    launch_linear(chunk.idx,
                  gated_buf, lw.gdn_out_w, m_buf, L, hidden, value_dim,
                  0, nc, 0, lw.gdn_out_ws);
    // Prefill persisted both states before their SPM slots were reused. Decode
    // keeps them live to this shared tail.
    if (decode) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_GDN_STATE_DECODE_STORE_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::SPM_SCATTER_TO_DDR),
            chunk.idx);
        rpu_launch_spm_scatter_ddr_dma_mutable(D("gdn_state"), &gdn_state_live_addr_[layer_idx], 0, Hc * Dk * Dv, Hc * Dk * Dv * 2, nc);
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_GDN_CONV_DECODE_STORE_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::SPM_SCATTER_TO_DDR),
            chunk.idx);
        rpu_launch_spm_scatter_ddr_dma_mutable(D("gdn_cs"),    &conv_state_live_addr_[layer_idx], 0, Kc * lconv,    Kc * lconv * 2,    nc);
    }
    // out_proj has consumed gated_buf, whose prefill prefix aliases RAW.
    // As in the old ring, RAW can now replace that dead input. The owner path
    // retains original owner rows there and gathers full normalized residual1.
    if (!decode && text_owner_norm_emitted && ctx().has_complete_physical_manifest() &&
        use_text_gdn_owner_rms(ctx().stage_plan,ctx().seq_len,
            ctx().physical_manifest().logical_length,ctx().position,ctx().batch_size)) {
        launch_text_gdn_owner_rms(layer_idx,chunk);
        *text_owner_norm_emitted = true;
    } else {
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_TEXT_GDN_ALL_REDUCE_SITE,
            text_ring_route(L, hidden), chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            m_buf, layer_input_residual_addr(layer_idx, chunk),
            D("residual2"), L, hidden, nc, num_cores());
    }
    return;
}


at::Tensor* Qwen3_5Model::gdn_state_for_layer(int layer_idx) {
    if (!gdn_states_ || layer_idx < 0 ||
        layer_idx >= static_cast<int>(gdn_states_->size())) return nullptr;
    return &(*gdn_states_)[layer_idx];
}

at::Tensor* Qwen3_5Model::conv_state_for_layer(int layer_idx) {
    if (!conv_states_ || layer_idx < 0 ||
        layer_idx >= static_cast<int>(conv_states_->size())) return nullptr;
    return &(*conv_states_)[layer_idx];
}

// Post-mixer layer tail, shared by full-attn and GDN (HF: residual add is fused into
// the mixer's all_reduce; this is the post_attention_layernorm + MLP + residual half).
// CONTRACT: ordinary input_norm (== residual2) contains the full raw stream.
// With this layer's owner flag it contains only original owner32 raw rows and
// residual1 already holds full normalized. The unchanged MLP ring consumes
// the matching owner residual rows. The exact interior Text tail instead
// keeps owner raw in residual1 and next normalized input in input_norm.
// Last-layer full residual, final norm and output DMA remain unchanged.
void Qwen3_5Model::emit_mlp_and_output(int layer_idx, const ChunkInfo& chunk,
                                     bool text_owner_norm_emitted, bool action_norm_emitted) {
    const auto& lw = layer_weights_[layer_idx];
    const int64_t h = hidden_size(), seq_len = chunk.len;
    const bool is_last_layer = (layer_idx == num_layers() - 1);
    // post-attention RMSNorm: residual1 = post_norm(stream); input_norm keeps the stream.
    TORCH_CHECK(!(text_owner_norm_emitted && action_norm_emitted),
                "Qwen3.5 norm producers cannot mix Text and Action flags");
    if (action_norm_emitted) {
        TORCH_CHECK(use_action_tail(chunk.len) && ctx().has_complete_physical_manifest() &&
                        layer_idx >= 0 && layer_idx < 24 && layer_is_full_[layer_idx] &&
                        chunk.idx == 0 && chunk.offset == 0 && chunk.len == 32,
                    "G05 Action postnorm skip lacks its same-layer tail producer");
        // Preserve the modulation-row update for the later MLP residual gate.
        load_adaptive_mod_row(2 * layer_idx + 1);
    } else if (text_owner_norm_emitted) {
        TORCH_CHECK(ctx().has_complete_physical_manifest() &&
            layer_idx >= 0 && layer_idx < 24 &&
            (layer_is_full_[layer_idx]
                ? use_text_owner_rms(ctx().stage_plan,ctx().seq_len,
                    ctx().physical_manifest().logical_length,ctx().position,ctx().batch_size)
                : use_text_gdn_owner_rms(ctx().stage_plan,ctx().seq_len,
                    ctx().physical_manifest().logical_length,ctx().position,ctx().batch_size)) &&
            chunk.idx == 0 && chunk.offset == 0 && chunk.len == 256,
            "G05 Text postnorm skip lacks its same-layer owner RMS producer");
        // residual1 already contains the full normalized result; input_norm
        // keeps only its original owner32 rows for the unchanged MLP ring.
    } else if (action_mode_) {
        load_adaptive_mod_row(2 * layer_idx + 1);
        apply_adaptive_norm(
            addr(0, "input_norm"), addr(0, "residual1"), chunk);
    } else {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "input_norm"), addr(0, "residual1"),
            layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, eps_, rmsnorm_route(seq_len, h), num_cores());
    }
    // MLP (SwiGLU) + reduce + residual → residual1. Standard text can reuse
    // residual1 for gate*up; action keeps that product in gate so residual1
    // retains the established adaptive-normalization lifetime.
    const int64_t mlp_elems =
        seq_len * (intermediate_size() / mlp_tp());
    const uint32_t product_buf =
        action_mode_ ? addr(0, "gate") : addr(0, "residual1");
    const uint32_t residual_buf =
        action_mode_ ? addr(0, "residual2") : addr(0, "input_norm");
    const at::Tensor gate_scale = lw.gate_ws;
    const at::Tensor up_scale = lw.up_ws;
    const at::Tensor down_scale = lw.down_ws;
    const bool resident_gateup = layer_idx >= 1 && layer_idx <= action_gateup_resident_layers() &&
        use_action_gateup_resident(seq_len);
    const bool paired_gateup = !resident_gateup && use_action_gateup_pair(seq_len);
    const bool text_paired_gateup = use_text_gateup_pair(seq_len);
    const bool text_fused_silu_mul = text_paired_gateup && use_text_silu_round_mul(seq_len);
    const bool fused_silu_mul = paired_gateup && use_action_silu_round_mul(seq_len);
    const bool high_silu_mul = paired_gateup && use_action_silu_high_mul(seq_len);
    const bool fused_gateup = paired_gateup && use_action_nvfp4_swiglu(seq_len);
    if (resident_gateup) {
        launch_action_gateup_resident(layer_idx, 0, addr(0, "gate"));
    } else if (fused_gateup) {
        launch_action_nvfp4_swiglu(layer_idx, chunk);
    } else if (paired_gateup) {
        // Up reads the same post-norm input; compute it before SiLU while
        // preserving separate rounded FP16 outputs and the original product.
        launch_action_gateup_pair(layer_idx, chunk);
    } else if (text_paired_gateup) {
        // Both projections read residual1 before the original SiLU/MUL. Text
        // still writes its product into residual1 after both inputs retire.
        launch_text_gateup_pair(layer_idx, chunk);
    } else {
        launch_linear(
            chunk.idx,
            addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
            seq_len, intermediate_size(), h, 1, mlp_tp(), 0, gate_scale, 2, layer_idx, lw.gate_negative_zero);
    }
    if (use_silu_ && !fused_silu_mul && !text_fused_silu_mul && !high_silu_mul) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ACTIVATION, QWEN35_TEXT_MLP_SILU_SITE,
                /*resolved_selector=*/1, /*resolved_flags=*/0,
                {use_silu_ ? 1 : 0, mlp_elems,
                 static_cast<int64_t>(ValuOpType::SILU)}, chunk.idx);
        }
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "gate"), addr(0, "gate"), mlp_elems,
            ValuOpType::SILU, false, mlp_tp(), unary_precision());
    }
    if (resident_gateup) {
        launch_action_gateup_resident(layer_idx, 1, addr(0, "up"));
    } else if (!paired_gateup && !text_paired_gateup) {
        launch_linear(
            chunk.idx,
            addr(0, "residual1"), lw.up_w, addr(0, "up"),
            seq_len, intermediate_size(), h, 1, mlp_tp(), 0, up_scale, 3, layer_idx, lw.up_negative_zero);
    }
    if (!fused_gateup) {
        if (text_fused_silu_mul) {
            // Both projections have retired before overwriting their R input.
            // Activated Gate is dead; phase31 Down overwrites Gate/Up storage.
            launch_text_silu_round_mul(layer_idx,chunk);
        } else if (fused_silu_mul) {
            launch_action_silu_round_mul(layer_idx, chunk);
        } else if (high_silu_mul) {
            launch_action_silu_high_mul(layer_idx, chunk);
        } else {
            rpu_launch_eltwise_binary_spm_kernel(
                addr(0, "gate"), addr(0, "up"), product_buf,
                mlp_elems, ValuOpType::MUL, c10::Half(1.0), mlp_tp());
        }
    }
    if (use_action_nvfp4_down_resident(seq_len,layer_idx)) {
        launch_action_nvfp4_down_resident(layer_idx,chunk);
    } else if (use_action_down_partial(seq_len,layer_idx)) {
        launch_action_down_partial(layer_idx,chunk);
    } else {
        launch_linear(
            chunk.idx,
            product_buf, lw.down_w, addr(0, "down"),
            seq_len, h, intermediate_size(), 0, mlp_tp(), 0, down_scale, 4, layer_idx, lw.down_negative_zero);
    }
    if (action_mlp_tail_handoff(layer_idx,chunk,true)) {
        launch_action_mlp_tail(layer_idx,chunk);
    } else if (text_mlp_tail_handoff(layer_idx,chunk,true)) {
        launch_text_mlp_tail(layer_idx,chunk);
    } else if (use_action_ring_rtc_residual(seq_len)) {
        launch_action_ring_rtc_residual(
            addr(0, "down"), residual_buf, addr(0, "residual1"),
            QWEN35_TEXT_MLP_ALL_REDUCE_SITE, chunk);
    } else {
        consume_manifest_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_TEXT_MLP_ALL_REDUCE_SITE,
            text_ring_route(seq_len, h), chunk.idx);
        rpu_launch_all_reduce_sum_residual_kernel(
            addr(0, "down"), residual_buf, addr(0, "residual1"),
            seq_len, h, mlp_tp(), num_cores());
        if (action_mode_) {
            apply_adaptive_residual_gate(
                addr(0, "residual1"), addr(0, "residual2"), chunk);
        }
    }
    // Split the last decoder-layer output from final norm for debug-only
    // localization. This fixed target belongs only to the independent debug
    // graph and remains stable across its same-shape REPLAY.
    if (is_last_layer && get_debug_export() && !action_mode_
        && ctx().seq_len > 1 && last_layer_decoder_debug_buf_.defined()) {
        c10::Half* decoder_base =
            last_layer_decoder_debug_buf_.data_ptr<c10::Half>()
            + chunk.offset * h;
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "residual1"), decoder_base, chunk.len * h);
    }
    // final_norm fuse at the last layer (Qwen3.5 last layer is full_attention).
    if (is_last_layer) {
        if (action_mode_) {
            load_adaptive_mod_row(2 * num_layers());
            apply_adaptive_norm(
                addr(0, "residual1"), addr(0, "residual1"), chunk);
        } else {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "residual1"), addr(0, "residual1"),
                addr(0, "final_norm_w"), seq_len, h, eps_, rmsnorm_route(seq_len, h), num_cores());
        }
    }
    // Export the post-layer residual stream at physical prefill length. For
    // the last layer this deliberately runs after final_norm, matching the
    // localization harness's layer-23 CPU reference. Chunk offset fills the
    // corresponding rows of [layers,1,physical_seq,hidden].
    if (get_debug_export() && !action_mode_ && ctx().seq_len > 1
        && per_layer_debug_buf_.defined()) {
        c10::Half* layer_base = per_layer_debug_buf_.data_ptr<c10::Half>()
            + layer_idx * per_layer_debug_buf_.size(1)
                * per_layer_debug_buf_.size(2) * h
            + chunk.offset * h;
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "residual1"), layer_base, chunk.len * h);
    }
    if (!ctx().output_to_spm && !(action_step_active_ && is_last_layer)) {
        emit_layer_output_dma(layer_idx, chunk);
    }
}

// ── declare_buffers: union of full-attention and GDN buffer sets; per-layer
//    norm preloads are guarded so empty attention slots on GDN layers are not
//    read. ───────────────────────────────────────────────────────────────────
std::vector<BufferDecl> Qwen3_5Model::declare_legacy_buffers(
    const LayoutContext& ctx) const {
    int64_t cs = ctx.chunk_size;
    int64_t h  = hidden_size();
    int64_t nq = num_q_heads();
    int64_t nkv = num_kv_heads();
    int64_t hd = head_dim();
    int64_t is_ = intermediate_size();
    int64_t nl  = num_layers();

    // Attention buffers use the adapter-expanded effective KV width and the
    // selected attention TP. MLP buffers use their independent selected TP.
    int64_t tp = attn_tp();
    int64_t local_q  = nq / tp;
    int64_t local_kv = nkv * hd / tp;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

    int64_t res = A(cs * h * DWIDTH);
    int64_t q   = A(cs * local_q * hd * DWIDTH);
    const int64_t raw_kv_rows = Align(cs, static_cast<int64_t>(16));
    int64_t kv  = A((ctx.attention_policy ==
                         AttentionExecutionPolicy::SPM_KV_BY_MHA
                     ? raw_kv_rows : cs) * local_kv * DWIDTH);
    int64_t mlp = A(cs * (is_ / mlp_tp()) * DWIDTH);
    int64_t mlp_down = res;
    int64_t oproj = res;
    int64_t nw  = A(h * DWIDTH);
    int64_t hnw = A(hd * DWIDTH);
    int64_t tmp = A(sdpa_compute_tmp_v16_size(make_sdpa_config(), cs) * 32);
    if (ctx.attention_policy ==
        AttentionExecutionPolicy::SPM_KV_BY_MHA) {
        tmp = std::max(tmp, A(raw_kv_rows * local_kv * DWIDTH));
    }

    const char* narrow_input_alias = action_mode_ ? nullptr : "input_norm";
    if (!action_mode_) {
        TORCH_CHECK(q <= res,
                    "Qwen3.5 attention output exceeds input_norm alias: output=",
                    q, ", input_norm=", res);
    }

    // GDN occupies phases 1..20, full attention 22..27 and the shared MLP
    // 30..31. The paths are mutually exclusive by layer, so disjoint local
    // phase ranges let main's LayerWide planner reuse their scratch without
    // abusing KV_FIRST-only BufferScope values.
    std::vector<BufferDecl> decls;
    action_nvfp4_.declare(decls, num_cores());
    decls.push_back({"residual1",  res,  1, 31, StorageClass::Temp, 0, nullptr});
    decls.push_back({"input_norm", res,  1, 31, StorageClass::Temp, 0, nullptr});
    decls.push_back({"residual2",  0,    0, 0, StorageClass::Temp, 0, "input_norm"});
    decls.push_back({"q",          q,    22, 25, StorageClass::Temp, 0, nullptr});
    const int attention_input_end = ctx.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA
        ? 25 : 24;
    decls.push_back({"k",          kv,   22, attention_input_end,
                     StorageClass::Temp, 0, nullptr});
    decls.push_back({"v",          kv,   22, attention_input_end,
                     StorageClass::Temp, 0, nullptr});
    decls.push_back({"output",     q,    25, 27, StorageClass::Temp, 0, narrow_input_alias});
    decls.push_back({"attn_gate",  q,    22, 26, StorageClass::Temp, 0, nullptr});
    decls.push_back({"oproj",      oproj, 27, 27, StorageClass::Temp, 0, nullptr});
    decls.push_back({"sdpa_tmp",   tmp,  25, 25, StorageClass::Temp, 0, nullptr});
    // Text mode consumes gate at phase 30 and writes the elementwise product
    // into residual1. Action mode keeps the generic FMB path, whose phase-31
    // down projection still reads the product from gate.
    decls.push_back({"gate",       mlp,  30, action_mode_ ? 31 : 30,
                     StorageClass::Temp, 0, nullptr});
    decls.push_back({"up",         mlp,  30, 30, StorageClass::Temp, 0, nullptr});
    decls.push_back({"down",       mlp_down, 31, 31, StorageClass::Temp, 0, nullptr});
    if (use_action_gateup_resident(action_len_)) {
        // The body-0 preload must survive every layer and each later body's
        // phase-0 input hook, not merely the MLP's phase 30.
        decls.push_back({"action_gateup_resident", QWEN35_ACTION_RESIDENT_TOTAL_BYTES,
                         0, 31, StorageClass::Temp, 0, nullptr, BufferScope::LayerWide});
    }
    if (use_action_nvfp4_o_resident(action_len_)) {
        // Body-0 preload survives every layer, phase0, and all ten FM bodies.
        decls.push_back({"action_nvfp4_o_weight",24 * 131072,
                        0,31,StorageClass::Temp,0,nullptr,BufferScope::LayerWide});
        decls.push_back({"action_nvfp4_o_scale",24 * 16384,
                        0,31,StorageClass::Temp,0,nullptr,BufferScope::LayerWide});
    }
    if (use_action_nvfp4_down_resident(action_len_,0)) {
        decls.push_back({"action_nvfp4_down_weight",10 * 262144,
                        0,31,StorageClass::Temp,0,nullptr,BufferScope::LayerWide});
        decls.push_back({"action_nvfp4_down_scale",10 * 32768,
                        0,31,StorageClass::Temp,0,nullptr,BufferScope::LayerWide});
    }
    if (use_action_o_resident(action_len_)) {
        decls.push_back({"action_o_resident",QWEN35_ACTION_RESIDENT_TOTAL_BYTES,
                        0,31,StorageClass::Temp,0,nullptr,BufferScope::LayerWide});
    }
    if (action_mode_) {
        const int64_t modulation_rows = use_action_mod_body() ? 2 * num_layers() + 1 : 1;
        decls.push_back({
            // RTC carries clean/noisy AdaLN triples side-by-side. Reserve both
            // halves for every action graph so toggling RTC never changes the
            // SPM manifest; non-RTC loads and reads only the first half.
            // LayerWide phase 1..31 prevents aliasing with any layer consumer
            // or post-body hook. A phase-0 input may reuse it before row 0 DMA.
            "adaptive_mod", A(modulation_rows * 6 * h * DWIDTH), 1, 31,
            StorageClass::Temp, 0, nullptr});
    }
    if (action_input_w_.defined()) {
        decls.push_back({
            "action_input", A(cs * action_dim_pad_ * DWIDTH), 0, 9,
            StorageClass::Temp, 0, nullptr});
        decls.push_back({
            "action_hidden", A(cs * h * DWIDTH), 0, 0,
            StorageClass::Temp, 0, nullptr});
        decls.push_back({
            "action_keep_mask", A(action_dim_pad_ * DWIDTH), 8, 9,
            StorageClass::Temp, 0, nullptr});
        decls.push_back({
            "action_velocity", A(cs * action_dim_pad_ * DWIDTH), 8, 9,
            StorageClass::Temp, 0, nullptr});
        {
            BufferDecl d;
            d.name = "action_input_bias";
            d.size = A(h * DWIDTH);
            d.storage = StorageClass::Persistent;
            d.preload_callback = [this](FusedModelBase&, int, uint32_t core0_addr) {
                if (this->ctx().has_complete_physical_manifest()) {
                    this->ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        QWEN35_TEXT_ACTION_INPUT_BIAS_PRELOAD_SITE,
                        qwen35_text_dma_route(
                            Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                        /*resolved_flags=*/0, {hidden_size(), 1});
                }
                rpu_launch_ddr_broadcast_spm_dma(
                    action_input_b_.data_ptr<c10::Half>(), hidden_size(),
                    core0_addr, /*num_cores=*/1);
            };
            decls.push_back(d);
        }
        {
            BufferDecl d;
            d.name = "action_output_bias";
            d.size = A(action_dim_pad_ * DWIDTH);
            d.storage = StorageClass::Persistent;
            d.preload_callback = [this](FusedModelBase&, int, uint32_t core0_addr) {
                if (this->ctx().has_complete_physical_manifest()) {
                    this->ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        QWEN35_TEXT_ACTION_OUTPUT_BIAS_PRELOAD_SITE,
                        qwen35_text_dma_route(
                            Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                        /*resolved_flags=*/0, {action_dim_pad_, 1});
                }
                rpu_launch_ddr_broadcast_spm_dma(
                    action_output_b_.data_ptr<c10::Half>(), action_dim_pad_,
                    core0_addr, /*num_cores=*/1);
            };
            decls.push_back(d);
        }
    }

    // Per-layer input_layernorm weight, preloaded for ALL layers (full + GDN now share
    // the unified input_norm channel; the GDN mixers read this persistent "norm_w"
    // instead of a per-forward gdn_innorm_w DMA).
    {
        BufferDecl d;
        d.name = "norm_w"; d.size = nw;
        d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl;
        d.preload_callback = [this](FusedModelBase&, int L, uint32_t core0_addr) {
            rpu_launch_ddr_broadcast_spm_dma(
                layer_weights_[L].input_norm_w.data_ptr<c10::Half>(),
                hidden_size(), core0_addr, num_cores());
        };
        decls.push_back(d);
    }
    {
        BufferDecl d;
        d.name = "post_norm_w"; d.size = nw;
        d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl;
        d.preload_callback = [this](FusedModelBase&, int L, uint32_t core0_addr) {
            // both full and GDN layers do the standard post_norm + MLP half
            rpu_launch_ddr_broadcast_spm_dma(
                layer_weights_[L].post_norm_w.data_ptr<c10::Half>(),
                hidden_size(), core0_addr, num_cores());
        };
        decls.push_back(d);
    }
    if (has_qk_norm_) {
        {
            BufferDecl d;
            d.name = "q_norm_w"; d.size = hnw;
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl;
            d.preload_callback = [this](FusedModelBase&, int L, uint32_t core0_addr) {
                if (!layer_is_full_[L]) return;
                if (this->ctx().has_complete_physical_manifest()) {
                    this->ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        QWEN35_TEXT_Q_NORM_PRELOAD_SITE,
                        qwen35_text_dma_route(
                            Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                        /*resolved_flags=*/0,
                        {L, layer_is_full_[L] ? 1 : 0, head_dim(), 1}, L);
                }
                rpu_launch_ddr_broadcast_spm_dma(
                    layer_weights_[L].q_norm_w.data_ptr<c10::Half>(),
                    head_dim(), core0_addr, num_cores() == 8 ? NUM_CORES : attn_tp());
            };
            decls.push_back(d);
        }
        {
            BufferDecl d;
            d.name = "k_norm_w"; d.size = hnw;
            d.storage = StorageClass::PersistentPerLayer; d.per_layer = nl;
            d.preload_callback = [this](FusedModelBase&, int L, uint32_t core0_addr) {
                if (!layer_is_full_[L]) return;
                if (this->ctx().has_complete_physical_manifest()) {
                    this->ctx().consume_physical_route(
                        FmbRouteFamily::MUTABLE_DMA,
                        QWEN35_TEXT_K_NORM_PRELOAD_SITE,
                        qwen35_text_dma_route(
                            Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                        /*resolved_flags=*/0,
                        {L, layer_is_full_[L] ? 1 : 0, head_dim(), 1}, L);
                }
                rpu_launch_ddr_broadcast_spm_dma(
                    layer_weights_[L].k_norm_w.data_ptr<c10::Half>(),
                    head_dim(), core0_addr, num_cores() == 8 ? NUM_CORES : attn_tp());
            };
            decls.push_back(d);
        }
    }
    {
        BufferDecl d;
        d.name = "final_norm_w"; d.size = nw;
        d.storage = StorageClass::Persistent; d.per_layer = 0;
        d.preload_callback = [this](FusedModelBase&, int, uint32_t core0_addr) {
            rpu_launch_ddr_broadcast_spm_dma(
                final_norm_w_.data_ptr<c10::Half>(), hidden_size(), core0_addr, num_cores());
        };
        decls.push_back(d);
    }

    if (use_action_down_partial(action_len_,0)) {
        decls.push_back({"action_down_partial",344064,
            0,31,StorageClass::Temp,0,nullptr,BufferScope::LayerWide});
    }

    // ── GDN buffers. A forward runs exactly one mode: decode resolves below 64 rows;
    //    adapter-padded prefill resolves to a multiple of 64. Declare only the active
    //    set so an unused mode cannot inflate the planner's SPM peak. ──
    if (has_gdn_) {
        const int64_t H = gdn_nvh_, Dk = gdn_dk_, Dv = gdn_dv_;
        const int64_t conv_dim = gdn_conv_dim_, Kc = gdn_kc_, hidden = hidden_size();
        const int     nc = gdn_tp();
        const int64_t Hc = H / nc, lconv = conv_dim / nc, lval = (H * Dv) / nc;
        const int64_t nkh = (conv_dim - H * Dv) / (2 * Dk), lkey = (nkh * Dk) / nc;
        const int64_t n_bg = ((Hc + 15) / 16) * 16, hist = Kc - 1;
        const int64_t scalar_slots = qwen35_gdn_scalar_slots(H, nc);
        const int64_t CS = std::max<int64_t>(64, ctx.chunk_size), N = CS / 64;
        auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };
        auto B = [&](const char* n, int64_t elems) -> BufferDecl {
            return {n, A(elems * DWIDTH), 1, 6, StorageClass::Temp, 0, nullptr};
        };
        auto BP = [&](const char* n, int64_t elems, int p0, int p1,
                      const char* alias = nullptr) -> BufferDecl {
            return {n, A(elems * DWIDTH), p0, p1,
                    StorageClass::Temp, 0, alias};
        };
        // Small setup weights are shared by both GDN modes. In prefill the
        // normalization weight is loaded during setup but consumed after the
        // recurrent phase, so its declared lifetime must span phase 18. A_log
        // and dt_bias are consumed while forming g at phase 3.
        if (ctx.chunk_size < 64) {
            decls.insert(decls.end(), {
                B("gdn_normw", Dv),
                B("gdn_Al", scalar_slots), B("gdn_dt", scalar_slots),
            });
        } else {
            decls.insert(decls.end(), {
                BP("gdn_normw", Dv, 1, 18),
                BP("gdn_Al", scalar_slots, 1, 3), BP("gdn_dt", scalar_slots, 1, 3),
            });
        }
        if (ctx.chunk_size < 64) {
            decls.push_back(B("gdn_state", Hc * Dk * Dv));
            decls.push_back(B("gdn_cs", Kc * lconv));
            decls.push_back(B("gdn_zero", Dk * Dv));
        }
        // ── GDN DECODE (mixer) buffers — selected cores, Hc heads/core ──
        if (ctx.chunk_size < 64) {
            decls.insert(decls.end(), {
                B("gdn_q", lkey), B("gdn_k", lkey), B("gdn_v", lval),       // per-path proj/conv input
                B("gdn_qa", lkey), B("gdn_ka", lkey), B("gdn_va", lval),    // per-path post-conv (silu)
                B("gdn_q_rep", Hc * Dk), B("gdn_k_rep", Hc * Dk),  // GQA-tiled q/k (Hc heads)
                B("gdn_z", lval), B("gdn_a", n_bg), B("gdn_b", n_bg),
                B("gdn_p", Hc * Dk * Dv), B("gdn_scratch", Hc * Dk * Dv),
                B("gdn_delta", Hc * Dv),
                B("gdn_gexp", Hc), B("gdn_beta", Hc), B("gdn_gs", Hc),
                B("gdn_out", Hc * Dv), B("gdn_gated", lval), B("gdn_m", hidden),
            });
        }
        if (ctx.chunk_size >= 64 && !action_mode_) {
            const int64_t out_tok_bytes = A(CS * Hc * Dv * DWIDTH);
            TORCH_CHECK(out_tok_bytes <= res,
                        "Qwen3.5 GDN out_tok exceeds input_norm alias: out_tok=",
                        out_tok_bytes, ", input_norm=", res);
        }
        // ── GDN PREFILL (chunk) buffers. Phase numbers follow the actual
        // producer/consumer order so dead intermediates can share SPM. Declaration
        // order among equal starts is intentional: the planner is stable first-fit.
        // WITHOUT phases the L=256 set sums ~10MB and OOMs.
        //   build_gdn prefill branch MUST NOT read a buffer after its phase_end. ──
        if (ctx.chunk_size >= 64) decls.insert(decls.end(), {
            BP("gdn_c_z",      CS * lval, 1, 19),
            BP("gdn_c_a",      CS * n_bg, 1, 3),
            BP("gdn_c_v_pad", (hist + CS) * lval, 1, 2),
            BP("gdn_c_strict", 64 * 64, 1, 14),
            BP("gdn_c_tril", 64 * 64, 1, 13),
            BP("gdn_c_k_pad", (hist + CS) * lkey, 1, 2),
            BP("gdn_c_b",      CS * n_bg, 1, 3),
            BP("gdn_cs", Kc * lconv, 1, 2),
            BP("gdn_c_q_pad", (hist + CS) * lkey, 1, 2),
            BP("gdn_c_key", CS * lkey, 2, 5),
            BP("gdn_c_query", CS * lkey, 2, 4),
            BP("gdn_c_value", CS * lval, 2, 6),
            BP("gdn_c_q_rep", CS * Hc * Dk, 3, 4),
            BP("gdn_c_beta", CS * Hc, 3, 7),
            BP("gdn_c_k_rep", CS * Hc * Dk, 3, 5),
            BP("gdn_c_g", CS * Hc, 3, 8),
            BP("gdn_c_q_h", Hc * CS * Dk, 4, 16),
            BP("gdn_c_k_h", Hc * CS * Dk, 5, 16),
            BP("gdn_c_v_h", Hc * CS * Dv, 6, 14),
            BP("gdn_c_beta_h", Hc * CS, 7, 11),
            BP("gdn_c_g_h", Hc * CS, 8, 12),
            BP("gdn_c_kbeta", Hc * CS * Dk, 10, 15),
            BP("gdn_c_gcumsum", Hc * CS, 12, 16),
            BP("gdn_c_decay", Hc * N * 64 * 64, 13, 16),
            BP("gdn_c_vnew", Hc * CS * Dv, 14, 16),
            BP("gdn_c_attn", Hc * N * 64 * 64, 14, 15),
            BP("gdn_c_kcd", Hc * CS * Dk, 15, 16),
            BP("gdn_c_gexp", Hc * CS, 15, 16),
            BP("gdn_c_ws", 64 * 17 * 16, 12, 12),
            BP("gdn_c_coreout", Hc * CS * Dv, 16, 17),
            BP("gdn_c_attnc", Hc * 64 * 64, 16, 16),
            BP("gdn_state", Hc * Dk * Dv, 16, 16),
            BP("gdn_c_glast", Hc, 16, 16),
            BP("gdn_c_loop_b", Hc * Dk * Dv, 16, 16),
            BP("gdn_c_out_tok", CS * Hc * Dv, 17, 20,
               narrow_input_alias),
            BP("gdn_c_m", CS * hidden, 20, 20),
        });
    }
    return decls;
}

std::vector<BufferDecl> Qwen3_5Model::declare_buffers(const LayoutContext& layout) {
    auto declarations = declare_legacy_buffers(layout);
    switch (layout.forward_operand_residency) {
        case FmbForwardOperandResidency::UNSPECIFIED:
        case FmbForwardOperandResidency::PER_LAYER:
            return declarations;
        case FmbForwardOperandResidency::FORWARD: {
            TORCH_CHECK(!action_mode_ && !z2_bound_ && has_gdn_ &&
                            layout.chunk_size >= 64,
                        "Qwen3.5 text retained masks require GDN prefill");
            const auto trial = detail::make_forward_spm_residency_candidate(
                declarations, {"gdn_c_tril", "gdn_c_strict"});
            TORCH_CHECK(trial.valid,
                        "Qwen3.5 text mask declaration cannot retain operands");
            return trial.declarations;
        }
    }
    TORCH_CHECK(false, "Qwen3.5 text unknown forward operand residency");
}

FmbForwardOperandResidency Qwen3_5Model::gdn_mask_residency_for_candidate(
    const LayoutContext& layout) const {
    // A descriptor-bound layout is immutable: do not reselect from live free
    // space during Graph reconstruction, allocation, or replay.
    if (layout.forward_operand_residency !=
            FmbForwardOperandResidency::UNSPECIFIED) {
        TORCH_CHECK(layout.forward_operand_residency ==
                        FmbForwardOperandResidency::PER_LAYER ||
                    layout.forward_operand_residency ==
                        FmbForwardOperandResidency::FORWARD,
                    "Qwen3.5 text unknown bound mask residency");
        return layout.forward_operand_residency;
    }
    if (action_mode_ || z2_bound_ || !has_gdn_ || layout.chunk_size < 64)
        return FmbForwardOperandResidency::PER_LAYER;
    const auto fits = [&](const LayoutContext& candidate_layout) {
        const auto trial = detail::make_forward_spm_residency_candidate(
            declare_legacy_buffers(candidate_layout),
            {"gdn_c_tril", "gdn_c_strict"});
        return trial.valid && declared_spm_layout_fits(trial.declarations);
    };
    if (!fits(layout)) return FmbForwardOperandResidency::PER_LAYER;
    if (layout.planning_chunk_capacity > 0 &&
            layout.planning_chunk_capacity != layout.chunk_size) {
        auto capacity_layout = layout;
        capacity_layout.chunk_size = layout.planning_chunk_capacity;
        if (!fits(capacity_layout)) return FmbForwardOperandResidency::PER_LAYER;
    }
    return FmbForwardOperandResidency::FORWARD;
}

std::vector<int64_t> Qwen3_5Model::gdn_mask_schedule_arguments(
    const FmbThreeStageChunkPlan& plan) const {
    const auto first = std::find(layer_is_full_.begin(), layer_is_full_.end(), 0);
    return {64, 64, 2,
            static_cast<int64_t>(first - layer_is_full_.begin()),
            static_cast<int64_t>(std::count(
                layer_is_full_.begin(), layer_is_full_.end(), 0)),
            plan.compute.plan.chunk_size,
            static_cast<int64_t>(plan.compute.chunks.size()),
            gdn_tp(), num_cores()};
}


FmbForwardOperandResidency Qwen3_5Model::physical_forward_operand_residency(
    const FmbPhysicalExecutionManifest& manifest) const {
    if (manifest.state != FmbPhysicalManifestState::COMPLETE ||
            manifest.physical_length == 1 || !has_gdn_ || action_mode_ || z2_bound_)
        return FmbForwardOperandResidency::UNSPECIFIED;
    const auto route = std::find_if(
        manifest.routes.begin(), manifest.routes.end(), [](const auto& row) {
            return row.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                row.site_id == QWEN35_TEXT_GDN_MASK_RESIDENCY_SITE;
        });
    TORCH_CHECK(route != manifest.routes.end() && route->flags == 0 &&
                    route->invocation == 0 &&
                    (route->selector == 1 || route->selector == 2),
                "Qwen3.5 text prefill requires its bound GDN mask schedule");
    TORCH_CHECK(std::count_if(manifest.routes.begin(), manifest.routes.end(),
                    [](const auto& row) {
                        return row.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                            row.site_id == QWEN35_TEXT_GDN_MASK_RESIDENCY_SITE;
                    }) == 1,
                "Qwen3.5 text requires one GDN mask schedule");
    const auto& args = route->arguments;
    const auto first = std::find(layer_is_full_.begin(), layer_is_full_.end(), 0);
    TORCH_CHECK(args.size() == 9 && args[0] == 64 && args[1] == 64 &&
                    args[2] == 2 &&
                    args[3] == first - layer_is_full_.begin() &&
                    args[4] == std::count(layer_is_full_.begin(), layer_is_full_.end(), 0) &&
                    args[5] >= 64 && args[5] % 64 == 0 &&
                    args[6] == CeilDiv(manifest.physical_length, args[5]) &&
                    args[7] == gdn_tp() && args[8] == num_cores(),
                "Qwen3.5 text GDN mask geometry differs from its owner");
    return route->selector == 2 ? FmbForwardOperandResidency::FORWARD
                                : FmbForwardOperandResidency::PER_LAYER;
}



ModelStaticConfig Qwen3_5Model::static_config() {
    ModelStaticConfig cfg;
    cfg.num_layers = num_layers();
    cfg.cross_layer_batch_size = num_layers();   // single group (mirror Qwen3)
    cfg.fast_replay_skip_layer_loop = fast_replay_active_;
    if (action_step_active_) {
        cfg.pre_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
            &Qwen3_5Model::emit_action_input_projection);
        cfg.post_layers_fn = reinterpret_cast<void (FusedModelBase::*)()>(
            &Qwen3_5Model::emit_action_output_projection);
        cfg.body_iterations = action_loop_active_ ? action_num_steps_ : 1;
    }
    return cfg;
}

ModelDynamicConfig Qwen3_5Model::dynamic_config(const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    // COMPLETE per-site routes decide between the bounded P0 single-chunk
    // raw-SPM handoff and explicit DDR_REQUIRED fallbacks.
    cfg.attention_policy = AttentionExecutionPolicy::AUTO;
    return cfg;
}

bool Qwen3_5Model::subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                             int64_t position) const {
    if (action_nvfp4_.enabled() && (!action_mode_ || cs != 32 || seq_len != 32)) return false;
    if (action_mode_) {
        // G0.5 action attention is bidirectional and unpadded. It must run as
        // one whole suffix chunk so every query sees every action key. Validate
        // both sparse layer families: shared VLM prefix and action-only.
        return cs >= seq_len
            && sdpa_is_valid_chunk_size(
                make_sdpa_config(/*mask=*/0), cs, seq_len, position)
            && sdpa_is_valid_chunk_size(
                make_sdpa_config(/*mask=*/0), cs, seq_len, /*position=*/0);
    }
    if (!sdpa_is_valid_chunk_size(make_sdpa_config(), cs, seq_len, position))
        return false;
    // Decode uses the recurrent single-token path and does not have the 64-row
    // chunk constraint. Prefill consists of C=64 GDN sub-chunks.
    if (seq_len <= 1 || !has_gdn_) return true;
    if (cs % 64 != 0) return false;

    // MR-A: certified envelope, per candidate. Unlike CausalDecoderModel this
    // class already closed the explicit-override route below (the cap check was
    // always in the VALIDITY hook, not the cap hook), so the envelope only has to
    // join it here. The deny-by-default half lives in subclass_chunk_size_cap.
    if (!chunk_within_envelope(cs)) return false;

    // The cold cap bounds AUTO enumeration only. Exact controls still pass the
    // kernel and certified-envelope predicates above, but do not inherit that
    // search heuristic.
    return true;
}

// MR-A: Qwen3.5 had no cap hook at all — the auto scan was bounded only by the
// validity predicate above. The override route reached compute_chunks_impl:499
// without any once-per-resolve gate, so there was nowhere to fail an undeclared
// handle loudly. This adds that gate; it runs before SPM allocation on both
// planner entry points (fused_model_base.cpp:615 dry, :1024 forward).
int64_t Qwen3_5Model::subclass_chunk_size_cap(int64_t seq_len,
                                              int64_t position) const {
    // G0.5 action mode is a fixed all-full-attention expert, not a Qwen3.5
    // text-prefill profile. Its action-length override is checked separately.
    if (action_mode_) return 0;
    constexpr int64_t max_kv_rows = int64_t{UINT16_MAX} * 16;
    TORCH_CHECK(
        seq_len > 0 && position >= 0 && seq_len <= max_kv_rows &&
            position <= max_kv_rows - seq_len,
        "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 attention key length "
        "exceeds the positive kernel register domain");
    const int64_t envelope_chunk =
        enforce_chunk_envelope(seq_len, position, "Qwen3_5Model");
    if (chunk_size_cap_ <= 0) return envelope_chunk;
    if (envelope_chunk <= 0) return chunk_size_cap_;
    return std::min(chunk_size_cap_, envelope_chunk);
}

bool Qwen3_5Model::subclass_spm_kv_by_mha_eligible(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t position) const {
    if (z2_bound_ || action_mode_ ||
        layout.attention_policy !=
            AttentionExecutionPolicy::SPM_KV_BY_MHA ||
        position != 0 || layout.batch_size != 1 || !layout.is_causal ||
        layout.use_attn_mask || plan.chunk_mode != ChunkMode::SEQUENTIAL ||
        plan.input.chunks.size() != 1 || plan.qkv.chunks.size() != 1 ||
        plan.compute.chunks.size() != 1 || plan.spans.size() != 1) {
        return false;
    }

    const ChunkInfo& input = plan.input.chunks.front();
    const ChunkInfo& qkv = plan.qkv.chunks.front();
    const ChunkInfo& compute = plan.compute.chunks.front();
    const FmbExecutionSpan& span = plan.spans.front();
    const int64_t seq = compute.len;
    if (seq <= 1 || input.offset != 0 || qkv.offset != 0 ||
        compute.offset != 0 ||
        span.offset != 0 || input.len != seq || qkv.len != seq ||
        span.len != seq || compute.kv_seq_len != seq ||
        layout.chunk_size != seq || layout.max_kv_seq_len != seq) {
        return false;
    }
    if (std::none_of(
            layer_is_full_.begin(), layer_is_full_.end(),
            [](int value) { return value != 0; })) {
        return false;
    }
    return sdpa_by_mha_spm_is_valid(
        /*batch=*/1, seq, seq, num_q_heads(), num_kv_heads(), head_dim(),
        attn_tp(), /*MASK_LTM=*/1);
}

std::vector<FmbPhysicalExecutionManifest>
Qwen3_5Model::physical_manifest_domain_for_candidate(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len,
    int64_t position) const {
    LayoutContext ddr_layout = layout;
    ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;
    std::vector<FmbPhysicalExecutionManifest> domain{
        physical_manifest_for_candidate(
            plan, ddr_layout, physical_len, logical_len, position)};

    LayoutContext raw_layout = layout;
    raw_layout.attention_policy =
        AttentionExecutionPolicy::SPM_KV_BY_MHA;
    if (subclass_spm_kv_by_mha_eligible(plan, raw_layout, position)) {
        domain.push_back(physical_manifest_for_candidate(
            plan, raw_layout, physical_len, logical_len, position));
    }
    return domain;
}

FmbLinearRouteSelector Qwen3_5Model::linear_route_selector(
    const at::Tensor& weight, int64_t m) const {
    const bool acc32 = linear_acc32_;
    // The new legacy 27B profile has only AUTO_TILE provenance. Its COMPLETE
    // descriptor must not inherit the smaller profiles' unmeasured GEMV route.
    const bool legacy_27b_geometry =
        num_layers() == 64 && hidden_size() == 5120 &&
        intermediate_size() == 17408 && num_q_heads() == 24 &&
        head_dim() == 256;
    if (m == 1 && !acc32 && !legacy_27b_geometry &&
        (weight.scalar_type() == at::kHalf ||
         weight.scalar_type() == at::kChar)) {
        return FmbLinearRouteSelector::GEMV;
    }
    return FmbLinearRouteSelector::AUTO_TILE;
}

FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len, int64_t position) const {
    // Z2 has its own physical coordinator. Action is a regular composite child
    // and consumes the same descriptor path as text below.
    if (z2_bound_) return {};

    FmbPhysicalExecutionManifest manifest;
    manifest.state = FmbPhysicalManifestState::COMPLETE;
    manifest.logical_length = logical_len;
    manifest.physical_length = physical_len;
    manifest.execution_padding_rows = physical_len - logical_len;
    manifest.kv_logical_length = physical_len == 1
        ? cos_.size(0) : position + logical_len;
    manifest.kv_insert_physical_rows = physical_len;
    manifest.graph_lifecycle = action_mode_
        ? action_descriptor_graph_lifecycle_
        : physical_len == 1 || retained_prefill_graph_
            ? FmbGraphLifecycle::RETAINED_CACHE
            : FmbGraphLifecycle::BOUNDED_ONESHOT;
    manifest.linear_accumulation =
        linear_acc32_
        ? FmbLinearAccumulationPolicy::ACC32
        : FmbLinearAccumulationPolicy::ACC16;
    if (num_cores() != 8) {
        manifest.routes.push_back({
            QWEN35_TEXT_CORE_TOPOLOGY_SITE, FmbRouteFamily::GRAPH_SCHEDULE,
            /*selector=*/1, /*flags=*/0, cold_topology_arguments()});
    }

    const bool has_full_attention = std::any_of(
        layer_is_full_.begin(), layer_is_full_.end(),
        [](int value) { return value != 0; });
    const bool has_gdn = std::any_of(
        layer_is_full_.begin(), layer_is_full_.end(),
        [](int value) { return value == 0; });
    const bool raw_spm = has_full_attention &&
        layout.attention_policy ==
            AttentionExecutionPolicy::SPM_KV_BY_MHA &&
        subclass_spm_kv_by_mha_eligible(plan, layout, position);
    auto attention_arguments = [&](int64_t seq_q, int64_t seq_k,
                                   int64_t mask_type) {
        return std::vector<int64_t>{
            /*batch=*/1, seq_q, seq_k, num_q_heads(), num_kv_heads(),
            head_dim(), attn_tp(), mask_type};
    };
    auto ddr_attention_reason = [&] {
        if (action_mode_)
            return QWEN35_TEXT_ATTN_ACTION_PREFIX_DDR_REQUIRED;
        if (position != 0)
            return QWEN35_TEXT_ATTN_PREFIX_HISTORY_DDR_REQUIRED;
        if (plan.qkv.chunks.size() != 1 ||
            plan.compute.chunks.size() != 1)
            return QWEN35_TEXT_ATTN_MULTI_CHUNK_DDR_REQUIRED;
        LayoutContext raw_probe = layout;
        raw_probe.attention_policy =
            AttentionExecutionPolicy::SPM_KV_BY_MHA;
        return subclass_spm_kv_by_mha_eligible(
                   plan, raw_probe, position)
            ? QWEN35_TEXT_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED
            : QWEN35_TEXT_ATTN_RAW_KERNEL_INCOMPATIBLE_DDR_REQUIRED;
    };
    if (action_mode_ && has_full_attention) {
        manifest.routes.push_back({
            kFullAttentionSite, FmbRouteFamily::ATTENTION,
            static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
            ddr_attention_reason(),
            {physical_len}});
    } else if (has_full_attention) {
        for (const ChunkInfo& chunk : plan.compute.chunks) {
            manifest.routes.push_back({
                raw_spm ? kFullAttentionRawSpmSite : kFullAttentionSite,
                FmbRouteFamily::ATTENTION,
                static_cast<int64_t>(
                    raw_spm ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                            : AttentionExecutionPolicy::DDR_KV),
                raw_spm ? 0 : ddr_attention_reason(),
                physical_len == 1
                    ? std::vector<int64_t>{1}
                    : attention_arguments(
                          chunk.len, chunk.kv_seq_len, /*MASK_LTM=*/1),
                /*invocation=*/chunk.idx});
        }
    }
    if (!action_mode_ && has_gdn_ && physical_len > 1) {
        const auto residency = gdn_mask_residency_for_candidate(layout);
        manifest.routes.push_back({
            QWEN35_TEXT_GDN_MASK_RESIDENCY_SITE, FmbRouteFamily::GRAPH_SCHEDULE,
            residency == FmbForwardOperandResidency::FORWARD ? 2 : 1,
            /*flags=*/0, gdn_mask_schedule_arguments(plan)});
    }
    auto append_route = [&](FmbRouteFamily family, int64_t site_id,
                            int64_t selector, int64_t invocation = 0) {
        manifest.routes.push_back({
            site_id, family, selector, /*flags=*/0, {}, invocation});
    };
    auto append_linear = [&](FmbLinearRouteSelector selector,
                             int64_t chunk_idx) {
        manifest.routes.push_back({
            QWEN35_TEXT_LINEAR_SITE, FmbRouteFamily::LINEAR,
            static_cast<int64_t>(selector), /*flags=*/0, {},
            linear_invocation(chunk_idx, selector)});
    };
    const int64_t action_body_iterations =
        action_step_active_ && action_loop_active_ ? action_num_steps_ : 1;
    manifest.routes.push_back({
        QWEN35_TEXT_ACTION_SCHEDULE_SITE, FmbRouteFamily::GRAPH_SCHEDULE,
        action_mode_ ? 2 : 1, /*flags=*/0,
        {action_mode_ ? 1 : 0, action_input_w_.defined() ? 1 : 0,
         action_loop_active_ ? 1 : 0, action_rtc_active_ ? 1 : 0,
         action_step_active_ ? 1 : 0, action_body_iterations,
         action_len_, action_dim_pad_}});
    if (high_precision_) {
        manifest.routes.push_back({high_precision_site(),
            FmbRouteFamily::GRAPH_SCHEDULE, 1, /*flags=*/0,
            high_precision_arguments()});
    }
    const int64_t full_layer_count = std::count_if(
        layer_is_full_.begin(), layer_is_full_.end(),
        [](int value) { return value != 0; });
    manifest.routes.push_back({
        QWEN35_TEXT_QK_NORM_SCHEDULE_SITE,
        FmbRouteFamily::GRAPH_SCHEDULE,
        has_qk_norm_ ? 2 : 1, /*flags=*/0,
        {has_qk_norm_ ? 1 : 0, has_mrope_ ? 1 : 0,
         head_dim(), attn_tp(), full_layer_count}});
    if (action_input_w_.defined()) {
        manifest.routes.push_back({
            QWEN35_TEXT_ACTION_INPUT_BIAS_PRELOAD_SITE,
            FmbRouteFamily::MUTABLE_DMA,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM),
            /*flags=*/0, {hidden_size(), 1}});
        manifest.routes.push_back({
            QWEN35_TEXT_ACTION_OUTPUT_BIAS_PRELOAD_SITE,
            FmbRouteFamily::MUTABLE_DMA,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM),
            /*flags=*/0, {action_dim_pad_, 1}});
    }
    if (has_qk_norm_) {
        for (int64_t layer = 0; layer < num_layers(); ++layer) {
            if (!layer_is_full_[layer]) continue;
            for (const int64_t site_id : {
                     QWEN35_TEXT_Q_NORM_PRELOAD_SITE,
                     QWEN35_TEXT_K_NORM_PRELOAD_SITE}) {
                manifest.routes.push_back({
                    site_id, FmbRouteFamily::MUTABLE_DMA,
                    qwen35_text_dma_route(
                        Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                    /*flags=*/0, {layer, 1, head_dim(), 1}, layer});
            }
        }
    }
    for (const ChunkInfo& chunk : plan.compute.chunks) {
        if (use_action_attn_sigmoid_mul(chunk.len)) {
            manifest.routes.push_back({QWEN35_ACTION_ATTN_SIGMOID_MUL_SITE,
                FmbRouteFamily::ACTIVATION,1,0,action_attn_sigmoid_mul_arguments(),chunk.idx});
        }
        if (use_g05_gdn_local_slice(chunk.len)) {
            manifest.routes.push_back({
                QWEN35_TEXT_GDN_LOCAL_SLICE_SITE, FmbRouteFamily::GRAPH_SCHEDULE,
                1, 0, g05_gdn_local_slice_arguments(chunk.len), chunk.idx});
        }
        if (use_text_gateup_pair(chunk.len)) {
            manifest.routes.push_back({
                text_gateup_pair_site(),
                FmbRouteFamily::LINEAR, 1, 0, text_gateup_pair_arguments(), chunk.idx});
        }
        if (use_action_gateup_pair(chunk.len)) {
            if (use_action_nvfp4_swiglu(chunk.len)) {
                manifest.routes.push_back({action_nvfp4_swiglu_site(),
                    FmbRouteFamily::LINEAR,
                    static_cast<int64_t>(linear_acc32_
                        ? FmbLinearRouteSelector::G05_NVFP4_SWIGLU_ACC32
                        : FmbLinearRouteSelector::G05_NVFP4_SWIGLU_ACC16),
                    0,action_nvfp4_swiglu_arguments(),chunk.idx});
            } else manifest.routes.push_back({
                action_gateup_pair_site(),
                FmbRouteFamily::LINEAR,
                action_gateup_pair_nvfp4_ ? static_cast<int64_t>(linear_acc32_
                    ? FmbLinearRouteSelector::G05_NVFP4_PAIRED_ACC32
                    : FmbLinearRouteSelector::G05_NVFP4_PAIRED_ACC16) : 1,
                0,action_gateup_pair_arguments(),chunk.idx});
        }
        if (use_text_silu_round_mul(plan,physical_len,logical_len,position,layout.batch_size)) {
            manifest.routes.push_back({QWEN35_TEXT_SILU_ROUND_MUL_SITE,
                FmbRouteFamily::ACTIVATION,1,0,text_silu_round_mul_arguments(logical_len),chunk.idx});
        }
        if (use_action_silu_round_mul(chunk.len)) {
            if (!use_action_nvfp4_swiglu(chunk.len)) manifest.routes.push_back({
                QWEN35_ACTION_SILU_ROUND_MUL_SITE,FmbRouteFamily::ACTIVATION,
                1,0,action_silu_round_mul_arguments(),chunk.idx});
        }
        if (use_action_silu_high_mul(chunk.len)) {
            manifest.routes.push_back({QWEN35_ACTION_SILU_HIGH_MUL_SITE,
                FmbRouteFamily::ACTIVATION,1,0,
                action_silu_high_mul_arguments(),chunk.idx});
        }
        if (use_action_owner_cache(chunk.len,position)) {
            if (use_action_gate_kv_owner(chunk.len,position)) {
                manifest.routes.push_back({QWEN35_ACTION_GATE_KV_OWNER_W8_SITE,
                    FmbRouteFamily::LINEAR,1,0,action_gate_kv_owner_arguments(position),chunk.idx});
            } else if (use_action_nvfp4(chunk.len)) {
                // Mixed Q uses NVFP4; W8 K/V retain their owner producer and fanout.
                manifest.routes.push_back({QWEN35_ACTION_KV_OWNER_W8_SITE,
                    FmbRouteFamily::LINEAR,1,0,action_kv_owner_arguments(),chunk.idx});
            } else {
                manifest.routes.push_back({QWEN35_ACTION_QKV_GATE_OWNER_ONLY_W8_SITE,
                    FmbRouteFamily::LINEAR,1,0,action_qkv_gate_owner_only_arguments(),chunk.idx});
            }
            if (use_action_nvfp4_qk_norm_rope(chunk.len,position)) {
                manifest.routes.push_back({QWEN35_ACTION_QK_NORM_ROPE_GROUP4_SITE,
                    FmbRouteFamily::ROPE,1,0,action_nvfp4_qk_norm_rope_arguments(),chunk.idx});
            } else manifest.routes.push_back({QWEN35_ACTION_K_OWNER_NORM_ROPE_SITE,
                FmbRouteFamily::ROPE,1,0,action_k_owner_norm_rope_arguments(),chunk.idx});
        } else if (use_action_qkv_gate(chunk.len)) {
            manifest.routes.push_back({QWEN35_ACTION_QKV_GATE_W8_SITE,
                FmbRouteFamily::LINEAR,1,0,action_qkv_gate_arguments(),chunk.idx});
        } else {
            if (use_action_q_gate(chunk.len)) {
                manifest.routes.push_back({
                    action_q_gate_w8_ ? QWEN35_ACTION_Q_GATE_W8_SITE : QWEN35_ACTION_Q_GATE_SITE,FmbRouteFamily::LINEAR,
                    1,0,action_q_gate_arguments(),chunk.idx});
            }
            if (use_action_kv_owner(chunk.len)) {
                manifest.routes.push_back({
                    action_kv_owner_w8_ ? QWEN35_ACTION_KV_OWNER_W8_SITE : QWEN35_ACTION_KV_OWNER_SITE,FmbRouteFamily::LINEAR,
                    1,0,action_kv_owner_arguments(),chunk.idx});
            }
        }
        if (use_action_gateup_resident(chunk.len)) {
            for (int64_t layer = 1; layer <= action_gateup_resident_layers(); ++layer) {
                for (int64_t projection = 0; projection < 2; ++projection) {
                    manifest.routes.push_back({
                        action_gateup_resident_w8_ ? QWEN35_ACTION_GATEUP_RESIDENT_W8_SITE : QWEN35_ACTION_GATEUP_RESIDENT_SITE, FmbRouteFamily::LINEAR,
                        action_gateup_resident_w8_ ? 6 : 5, 0, action_gateup_resident_arguments(layer, projection),
                        (layer - 1) * 2 + projection});
                }
            }
        }
        if (use_action_down_partial(chunk.len,0)) {
            manifest.routes.push_back({QWEN35_ACTION_DOWN_PARTIAL_W8_SITE,
                FmbRouteFamily::LINEAR,6,0,action_down_partial_arguments(false),0});
        }
        if (use_action_nvfp4_o_resident(chunk.len)) {
            for (int64_t layer = 0; layer < 24; ++layer) {
                manifest.routes.push_back({action_nvfp4_o_resident_site(),
                    FmbRouteFamily::LINEAR,
                    static_cast<int64_t>(linear_acc32_
                        ? FmbLinearRouteSelector::G05_NVFP4_RESIDENT_ACC32
                        : FmbLinearRouteSelector::G05_NVFP4_RESIDENT_ACC16),0,
                    action_nvfp4_o_resident_arguments(layer,-1),layer});
            }
        }
        if (use_action_nvfp4_down_resident(chunk.len,0)) {
            for (int64_t layer = 0; layer < 10; ++layer) {
                manifest.routes.push_back({action_nvfp4_down_resident_site(),
                    FmbRouteFamily::LINEAR,
                    static_cast<int64_t>(linear_acc32_
                        ? FmbLinearRouteSelector::G05_NVFP4_RESIDENT_ACC32
                        : FmbLinearRouteSelector::G05_NVFP4_RESIDENT_ACC16),0,
                    action_nvfp4_down_resident_arguments(layer,-1),layer});
            }
        }
        if (use_action_o_resident(chunk.len)) {
            for (int64_t layer = 0; layer < 24; ++layer) {
                manifest.routes.push_back({QWEN35_ACTION_O_RESIDENT_W8_SITE,
                    FmbRouteFamily::LINEAR,6,0,
                    action_o_resident_arguments(layer,false),layer});
            }
        }
        if (action_mode_) {
            const int64_t norm_route = action_norm_route(chunk.len);
            if (use_action_newton_norm_shift(chunk.len)) {
                manifest.routes.push_back({QWEN35_ACTION_NEWTON_NORM_SHIFT_SITE,
                    FmbRouteFamily::NORMALIZATION,1,0,
                    action_newton_norm_shift_arguments(),chunk.idx});
            } else manifest.routes.push_back({
                norm_route >= 2 ? QWEN35_ACTION_FUSED_NORM_SHIFT_SITE
                                : QWEN35_ACTION_RMS_NORM_SHIFT_SITE,
                FmbRouteFamily::NORMALIZATION, norm_route, /*flags=*/0,
                action_norm_arguments(chunk.len), chunk.idx});
            if (use_action_tail(chunk.len)) {
                manifest.routes.push_back({QWEN35_ACTION_TAIL_SITE,
                    FmbRouteFamily::GRAPH_SCHEDULE,1,0,action_tail_arguments(),chunk.idx});
            }
            if (use_action_ring_rtc_residual(chunk.len)) {
                manifest.routes.push_back({QWEN35_ACTION_RING_RTC_RESIDUAL_SITE,
                    FmbRouteFamily::GRAPH_SCHEDULE,1,0,
                    action_ring_rtc_residual_arguments(),chunk.idx});
            } else if (action_residual_route(chunk.len) == 4) {
                manifest.routes.push_back({QWEN35_ACTION_RTC_RESIDUAL_GATE_SITE,
                    FmbRouteFamily::GRAPH_SCHEDULE,4,0,
                    {chunk.len,hidden_size(),num_cores(),1,2,3*hidden_size()*DWIDTH,0,32,4},chunk.idx});
            } else {
                manifest.routes.push_back({
                    QWEN35_ACTION_RESIDUAL_GATE_SITE, FmbRouteFamily::GRAPH_SCHEDULE,
                    action_residual_route(chunk.len), /*flags=*/0,
                    {chunk.len, hidden_size(), num_cores()}, chunk.idx});
            }
        }
        if (use_text_mlp_tail(plan,physical_len,logical_len,position,layout.batch_size)) {
            manifest.routes.push_back({QWEN35_TEXT_MLP_TAIL_SITE,
                FmbRouteFamily::GRAPH_SCHEDULE,1,0,text_mlp_tail_arguments(logical_len),chunk.idx});
        }
        if (use_action_mlp_tail(plan,physical_len)) {
            manifest.routes.push_back({QWEN35_ACTION_MLP_TAIL_SITE,
                FmbRouteFamily::GRAPH_SCHEDULE,1,0,action_mlp_tail_arguments(),chunk.idx});
        }
        if (use_action_nvfp4(chunk.len)) {
            for (int family = 0; family < 5; ++family) {
                if (family == 1 && use_action_nvfp4_o_resident(chunk.len)) continue;
                if ((family == 2 || family == 3) && use_action_gateup_pair(chunk.len)) continue;
                manifest.routes.push_back({QWEN35_ACTION_NVFP4_LINEAR_SITE,
                    FmbRouteFamily::LINEAR,
                    static_cast<int64_t>(linear_acc32_ ? FmbLinearRouteSelector::G05_NVFP4_V2_ACC32
                        : FmbLinearRouteSelector::G05_NVFP4_V2_ACC16),
                        0, action_nvfp4_arguments(family), chunk.idx * 5 + family});
            }
        }
        bool needs_auto_tile = false;
        bool needs_gemv = false;
        auto observe_linear = [&](const at::Tensor& weight,
                                  int64_t rows = -1) {
            if (action_nvfp4_.enabled() && weight.scalar_type() == at::kByte) return;
            const FmbLinearRouteSelector selector =
                linear_route_selector(
                    weight, rows >= 0 ? rows : chunk.len);
            TORCH_INTERNAL_ASSERT(
                selector == FmbLinearRouteSelector::AUTO_TILE ||
                    selector == FmbLinearRouteSelector::GEMV,
                "Qwen3.5 text resolved an unsupported LINEAR route");
            needs_auto_tile |=
                selector == FmbLinearRouteSelector::AUTO_TILE;
            needs_gemv |= selector == FmbLinearRouteSelector::GEMV;
        };
        for (int64_t layer_idx = 0; layer_idx < num_layers(); ++layer_idx) {
            const auto& lw = layer_weights_[layer_idx];
            if (layer_is_full_[layer_idx]) {
                observe_linear(lw.q_w);
                if (!use_action_gate_kv_owner(chunk.len,position)) {
                    observe_linear(lw.k_w);
                    observe_linear(lw.v_w);
                    observe_linear(lw.attn_gate_w);
                }
                observe_linear(lw.o_w);
            } else {
                observe_linear(lw.gdn_q_w);
                observe_linear(lw.gdn_k_w);
                observe_linear(lw.gdn_v_w);
                observe_linear(lw.gdn_in_z_w);
                observe_linear(lw.gdn_b_bg_w);
                observe_linear(lw.gdn_a_bg_w);
                observe_linear(lw.gdn_out_w);
            }
            observe_linear(lw.gate_w);
            observe_linear(lw.up_w);
            observe_linear(lw.down_w);
        }
        if (action_mode_ && action_descriptor_include_io_ && chunk.idx == 0) {
            observe_linear(action_input_w_, physical_len);
            observe_linear(action_output_w_, physical_len);
        }
        if (needs_auto_tile) {
            append_linear(FmbLinearRouteSelector::AUTO_TILE, chunk.idx);
        }
        if (needs_gemv) {
            append_linear(FmbLinearRouteSelector::GEMV, chunk.idx);
        }
        if (use_silu_ && (!use_action_silu_round_mul(chunk.len) ||
                          use_action_gateup_resident(chunk.len)) &&
            (!use_action_silu_high_mul(chunk.len) || use_action_gateup_resident(chunk.len)) &&
            !use_text_silu_round_mul(plan,physical_len,logical_len,position,layout.batch_size)) {
            manifest.routes.push_back({
                QWEN35_TEXT_MLP_SILU_SITE, FmbRouteFamily::ACTIVATION,
                /*selector=*/1, /*flags=*/0,
                {use_silu_ ? 1 : 0,
                 chunk.len * (intermediate_size() / mlp_tp()),
                 static_cast<int64_t>(ValuOpType::SILU)}, chunk.idx});
        }
    }

    for (const ChunkInfo& chunk : plan.compute.chunks) {
        if (has_full_attention) {
            if (use_action_nvfp4_qk_norm_rope(chunk.len,position)) {
                // The dedicated group4 route above replaces both Q/K RoPE leaves.
            } else if (has_qk_norm_ && has_mrope_) {
                append_route(
                    FmbRouteFamily::ROPE,
                    chunk.len <= 1
                        ? QWEN35_TEXT_PARTIAL_ROPE_1D_SITE
                        : QWEN35_TEXT_PARTIAL_MROPE_SITE,
                    qwen35_text_rope_route(
                        chunk.len <= 1
                            ? Qwen35TextRopeRoute::PARTIAL_ROPE_1D
                            : Qwen35TextRopeRoute::PARTIAL_MROPE),
                    chunk.idx);
            } else if (has_qk_norm_) {
                append_route(
                    FmbRouteFamily::ROPE,
                    QWEN35_TEXT_Q_NORM_ROPE_SITE,
                    qwen35_text_rope_route(
                        Qwen35TextRopeRoute::ROPE_SPM),
                    chunk.idx);
                append_route(
                    FmbRouteFamily::ROPE,
                    QWEN35_TEXT_K_NORM_ROPE_SITE,
                    qwen35_text_rope_route(
                        Qwen35TextRopeRoute::ROPE_SPM),
                    chunk.idx);
            } else {
                append_route(
                    FmbRouteFamily::ROPE, QWEN35_TEXT_Q_ROPE_SITE,
                    qwen35_text_rope_route(
                        Qwen35TextRopeRoute::ROPE_SPM),
                    chunk.idx);
                append_route(
                    FmbRouteFamily::ROPE, QWEN35_TEXT_K_ROPE_SITE,
                    qwen35_text_rope_route(
                        Qwen35TextRopeRoute::ROPE_SPM),
                    chunk.idx);
            }
            append_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN35_TEXT_FULL_PREPARE_ALL_REDUCE_SITE,
                static_cast<int64_t>(
                    Qwen35TextAllReduceRoute::PREPARE_RING_INPUT),
                chunk.idx);
            if (use_text_owner_rms(plan,physical_len,logical_len,position,layout.batch_size)) {
                manifest.routes.push_back({QWEN35_TEXT_OWNER_RMS_SITE,
                    FmbRouteFamily::ALL_REDUCE,1,0,text_owner_rms_arguments(logical_len),chunk.idx});
            } else {
                append_route(
                    FmbRouteFamily::ALL_REDUCE,
                    QWEN35_TEXT_FULL_ALL_REDUCE_SITE,
                    text_ring_route(chunk.len, hidden_size()),
                    chunk.idx);
            }
        }
        if (has_gdn) {
            if (gdn_tp() != num_cores()) {
                append_route(
                    FmbRouteFamily::ALL_REDUCE,
                    QWEN35_TEXT_GDN_PREPARE_ALL_REDUCE_SITE,
                    static_cast<int64_t>(Qwen35TextAllReduceRoute::PREPARE_RING_INPUT),
                    chunk.idx);
            }
            const bool decode = chunk.len <= 1;
            const bool initial_prefill_chunk =
                !decode && position == 0 && chunk.offset == 0;
            if (decode) {
                for (const int64_t site_id : {
                         QWEN35_TEXT_GDN_STATE_LOAD_DMA_SITE,
                         QWEN35_TEXT_GDN_CONV_DECODE_LOAD_DMA_SITE}) {
                    append_route(
                        FmbRouteFamily::MUTABLE_DMA, site_id,
                        qwen35_text_dma_route(
                            Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
                        chunk.idx);
                }
                for (const int64_t site_id : {
                         QWEN35_TEXT_GDN_STATE_DECODE_STORE_DMA_SITE,
                         QWEN35_TEXT_GDN_CONV_DECODE_STORE_DMA_SITE}) {
                    append_route(
                        FmbRouteFamily::MUTABLE_DMA, site_id,
                        qwen35_text_dma_route(
                            Qwen35TextMutableDmaRoute::SPM_SCATTER_TO_DDR),
                        chunk.idx);
                }
            } else {
                if (!initial_prefill_chunk) {
                    for (const int64_t site_id : {
                             QWEN35_TEXT_GDN_CONV_PREFILL_LOAD_DMA_SITE,
                             QWEN35_TEXT_GDN_STATE_PREFILL_LOAD_DMA_SITE}) {
                        append_route(
                            FmbRouteFamily::MUTABLE_DMA, site_id,
                            qwen35_text_dma_route(
                                Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
                            chunk.idx);
                    }
                }
                for (const int64_t site_id : {
                         QWEN35_TEXT_GDN_CONV_PREFILL_STORE_DMA_SITE,
                         QWEN35_TEXT_GDN_STATE_PREFILL_STORE_DMA_SITE}) {
                    append_route(
                        FmbRouteFamily::MUTABLE_DMA, site_id,
                        qwen35_text_dma_route(
                            Qwen35TextMutableDmaRoute::SPM_SCATTER_TO_DDR),
                        chunk.idx);
                }
            }
            if (use_text_gdn_owner_rms(plan,physical_len,logical_len,position,layout.batch_size)) {
                manifest.routes.push_back({QWEN35_TEXT_GDN_OWNER_RMS_SITE,
                    FmbRouteFamily::ALL_REDUCE,1,0,text_gdn_owner_rms_arguments(logical_len),chunk.idx});
            } else {
                append_route(
                    FmbRouteFamily::ALL_REDUCE,
                    QWEN35_TEXT_GDN_ALL_REDUCE_SITE,
                    text_ring_route(chunk.len, hidden_size()),
                    chunk.idx);
            }
        }
        append_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_TEXT_MLP_ALL_REDUCE_SITE,
            text_ring_route(chunk.len, hidden_size()),
            chunk.idx);
    }

    if (action_mode_) {
        if (use_action_down_partial(physical_len,0)) {
            manifest.routes.push_back({QWEN35_ACTION_DOWN_PARTIAL_PRELOAD_W8_SITE,
                FmbRouteFamily::MUTABLE_DMA,
                qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
                0,action_down_partial_arguments(true),0});
        }
        if (use_action_nvfp4_o_resident(physical_len)) {
            for (int64_t family = 0; family < 2; ++family) {
                manifest.routes.push_back({QWEN35_ACTION_O_PRELOAD_NVFP4_SITE,
                    FmbRouteFamily::MUTABLE_DMA,
                    qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),0,
                    action_nvfp4_o_resident_arguments(0,family),family});
            }
        }
        if (use_action_nvfp4_down_resident(physical_len,0)) {
            for (int64_t family = 0; family < 2; ++family) {
                manifest.routes.push_back({QWEN35_ACTION_DOWN_PRELOAD_NVFP4_SITE,
                    FmbRouteFamily::MUTABLE_DMA,
                    qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),0,
                    action_nvfp4_down_resident_arguments(0,family),family});
            }
        }
        if (use_action_o_resident(physical_len)) {
            for (int64_t bundle = 0; bundle < 6; ++bundle) {
                manifest.routes.push_back({QWEN35_ACTION_O_PRELOAD_W8_SITE,
                    FmbRouteFamily::MUTABLE_DMA,
                    qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
                    0,action_o_resident_arguments(bundle * 4,true),bundle});
            }
        }
        if (use_action_gateup_resident(physical_len)) {
            for (int64_t slot = 0; slot < action_gateup_resident_layers(); ++slot) {
                manifest.routes.push_back({
                    action_gateup_resident_w8_ ? QWEN35_ACTION_GATEUP_PRELOAD_W8_SITE : QWEN35_ACTION_GATEUP_PRELOAD_SITE, FmbRouteFamily::MUTABLE_DMA,
                    qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_SCATTER_TO_SPM),
                    0, action_gateup_resident_arguments(slot + 1, 2), slot});
            }
        }
        if (use_action_mod_body()) {
            for (int64_t body = 0; body < action_num_steps_; ++body) {
                manifest.routes.push_back({
                    QWEN35_TEXT_ADAPTIVE_MOD_DMA_SITE, FmbRouteFamily::MUTABLE_DMA,
                    qwen35_text_dma_route(Qwen35TextMutableDmaRoute::DDR_BROADCAST_MOD_BODY),
                    0, action_mod_body_arguments(body), body});
            }
        } else {
            append_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_TEXT_ADAPTIVE_MOD_DMA_SITE,
                qwen35_text_dma_route(
                    Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        }
        if ((action_descriptor_route_flags_ & QWEN35_ACTION_ROUTE_IO) != 0) {
            append_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_TEXT_ACTION_INPUT_DMA_SITE,
                qwen35_text_dma_route(
                    Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
            append_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_TEXT_ACTION_KEEP_MASK_DMA_SITE,
                qwen35_text_dma_route(
                    Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
            append_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_TEXT_ACTION_OUTPUT_DMA_SITE,
                qwen35_text_dma_route(
                    Qwen35TextMutableDmaRoute::SPM_COPY_TO_DDR));
        }
        if ((action_descriptor_route_flags_ & QWEN35_ACTION_ROUTE_RTC) != 0) {
            append_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_TEXT_ACTION_RTC_PREFIX_DMA_SITE,
                qwen35_text_dma_route(
                    Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
        }
        if ((action_descriptor_route_flags_ & QWEN35_ACTION_ROUTE_TRACE) != 0) {
            for (const int64_t site_id : {
                     QWEN35_TEXT_ACTION_TRACE_PRE_DMA_SITE,
                     QWEN35_TEXT_ACTION_TRACE_EMBED_DMA_SITE,
                     QWEN35_TEXT_ACTION_TRACE_HIDDEN_DMA_SITE,
                     QWEN35_TEXT_ACTION_TRACE_VELOCITY_DMA_SITE,
                     QWEN35_TEXT_ACTION_TRACE_POST_DMA_SITE}) {
                append_route(
                    FmbRouteFamily::MUTABLE_DMA, site_id,
                    qwen35_text_dma_route(
                        Qwen35TextMutableDmaRoute::SPM_COPY_TO_DDR));
            }
        }
    }
    if (action_mode_ && has_full_attention) {
        constexpr uint32_t kv_caps = KV_INSERT_CAP_V2 |
            KV_INSERT_CAP_V16 | KV_INSERT_CAP_HYBRID2 |
            KV_INSERT_CAP_HYBRID3;
        auto append_kv_route = [&](int64_t invocation,
                                   int64_t insert_position) {
            const KvInsertSegmentPlan kv_plan =
                resolve_kvinsert_plan_auto(
                    kFullAttentionKvInsertSite, manifest.graph_lifecycle,
                    insert_position, physical_len, physical_len, attn_tp(),
                    num_kv_heads(), head_dim(), kv_caps);
            const KvInsertRouteArguments arguments =
                rpu_kvinsert_route_arguments(
                    kv_plan, attn_tp(), num_kv_heads(), head_dim());
            manifest.routes.push_back({
                kFullAttentionKvInsertSite, FmbRouteFamily::KV_INSERT,
                static_cast<int64_t>(kv_plan.route()), /*flags=*/0,
                {arguments.begin(), arguments.end()}, invocation});
            if (use_action_owner_cache(physical_len,position) &&
                use_action_owner_cache_joint(physical_len,insert_position)) {
                manifest.routes.push_back({
                    QWEN35_ACTION_OWNER_CACHE_JOINT_SITE,FmbRouteFamily::GRAPH_SCHEDULE,
                    1,0,action_owner_cache_joint_arguments(kv_plan),invocation});
            } else if (use_action_owner_cache(physical_len,position)) {
                manifest.routes.push_back({
                    QWEN35_ACTION_OWNER_CACHE_INSERT_SITE,FmbRouteFamily::GRAPH_SCHEDULE,
                    1,0,action_owner_cache_arguments(kv_plan),invocation});
            }
        };
        // One native call site has two action geometries. Invocation 0 is the
        // local position-zero plan; invocation 1 is the shared-prefix plan.
        append_kv_route(/*invocation=*/0, /*insert_position=*/0);
        if (position > 0) {
            append_kv_route(/*invocation=*/1, position);
        }
    } else if (has_full_attention) {
        for (const ChunkInfo& chunk : plan.qkv.chunks) {
            const bool dynamic_decode_position = physical_len == 1;
            const int64_t insert_position = dynamic_decode_position
                ? 0 : position + chunk.offset;
            const KvInsertSegmentPlan kv_plan = dynamic_decode_position
                ? rpu_resolve_kvinsert_segment_plan(
                      insert_position, chunk.len, chunk.len, attn_tp(),
                      num_kv_heads(), head_dim(), KV_INSERT_CAP_V2,
                      KvInsertRoute::V2)
                : resolve_kvinsert_plan_auto(
                      kFullAttentionKvInsertSite, manifest.graph_lifecycle,
                      insert_position, chunk.len, chunk.len, attn_tp(),
                      num_kv_heads(), head_dim(),
                      QWEN35_TEXT_KV_CAPABILITIES);
            const KvInsertRouteArguments arguments =
                rpu_kvinsert_route_arguments(
                    kv_plan, attn_tp(), num_kv_heads(), head_dim());
            manifest.routes.push_back({
                kFullAttentionKvInsertSite, FmbRouteFamily::KV_INSERT,
                static_cast<int64_t>(kv_plan.route()),
                QWEN35_TEXT_KV_REASON_DDR_REQUIRED |
                    (dynamic_decode_position
                         ? KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION : 0),
                {arguments.begin(), arguments.end()},
                /*invocation=*/chunk.idx});
        }
    }
    append_fmb_shared_runtime_routes(
        manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA,
        /*compute_row_multiplier=*/1,
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

FmbPhysicalManifestForwardCapability
Qwen3_5Model::physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& manifest) const {
    if (z2_bound_) return {};
    if (action_mode_) {
        TORCH_CHECK(
            manifest.graph_lifecycle == FmbGraphLifecycle::BOUNDED_ONESHOT ||
                manifest.graph_lifecycle == FmbGraphLifecycle::COMPOSITE_CHILD,
            "Qwen3.5 action descriptor carries an unsupported Graph lifecycle");
        // The generic KV cost oracle may replace a primary KV22 with an EXACT
        // candidate. This transport has its own frozen copy of that plan and
        // only admits the two measured schedules. Reject drift before layout
        // publication or Graph construction, rather than at the native writer.
        for (const auto& route : manifest.routes) {
            const bool joint = route.site_id == QWEN35_ACTION_OWNER_CACHE_JOINT_SITE;
            if (!joint && route.site_id != QWEN35_ACTION_OWNER_CACHE_INSERT_SITE) continue;
            TORCH_CHECK(route.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                            route.selector == 1 && route.flags == 0 &&
                            (route.invocation == 0 || route.invocation == 1),
                        "G05 owner cache transport descriptor is invalid");
            const auto plan = restore_kvinsert_plan_from_manifest(
                manifest,kFullAttentionKvInsertSite,route.invocation,8,8,256);
            TORCH_CHECK(joint == use_action_owner_cache_joint(
                            plan.physical_rows(),plan.segment(0).position),
                        "G05 owner cache transport differs from its actual frozen prefix/cold joint capability");
            TORCH_CHECK(route.arguments == (joint ? action_owner_cache_joint_arguments(plan)
                                                 : action_owner_cache_arguments(plan)),
                        "G05 owner cache transport differs from its primary KV plan");
        }
        return {true, manifest.graph_lifecycle};
    }
    const FmbGraphLifecycle expected = manifest.logical_length == 1 ||
            retained_prefill_graph_
        ? FmbGraphLifecycle::RETAINED_CACHE
        : FmbGraphLifecycle::BOUNDED_ONESHOT;
    return {true, expected};
}

// Per-forward prefill M-RoPE tables. Generic Qwen3.5 uses a fresh oneshot graph;
// fixed-profile retained graphs rely on same-shape copy_ preserving this address.
void Qwen3_5Model::set_prefill_rope(const at::Tensor& cos, const at::Tensor& sin) {
    TORCH_CHECK(cos.defined() && sin.defined() && cos.dim() == 2 && sin.dim() == 2,
                "set_prefill_rope: cos/sin must be 2D [N, rotary_dim/2]");
    TORCH_CHECK(cos.scalar_type() == at::kHalf && sin.scalar_type() == at::kHalf,
                "set_prefill_rope: cos/sin must be FP16");
    TORCH_CHECK(cos.device().type() == at::kPrivateUse1
                && sin.device().type() == at::kPrivateUse1,
                "set_prefill_rope: cos/sin must be on device");
    TORCH_CHECK(cos.sizes() == sin.sizes(),
                "set_prefill_rope: cos/sin shapes must match");
    TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous(),
                "set_prefill_rope: cos/sin must be contiguous");
    if (prefill_cos_.defined() && prefill_cos_.sizes() == cos.sizes()) {
        // Retain the DDR addresses baked into an existing action GraphCache
        // entry. Text prefill also benefits from reusing the aligned backing.
        prefill_cos_.copy_(cos);
        prefill_sin_.copy_(sin);
        return;
    }
    prefill_cos_ = keep_256b_aligned_rpu_copy(cos, "set_prefill_rope: cos");
    prefill_sin_ = keep_256b_aligned_rpu_copy(sin, "set_prefill_rope: sin");
}

// ═══════════════════════════ SETUP (weights) ═══════════════════════════════

// ── set_weights: store full-layer weights; GDN slots stay empty ──────────────
void Qwen3_5Model::set_weights(
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList attn_gate_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList gate_list, at::TensorList up_list, at::TensorList down_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    at::IntArrayRef layer_is_full,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::TensorList gdn_in_z_list, at::TensorList gdn_out_list,
    at::TensorList gdn_A_log_list, at::TensorList gdn_dt_bias_list,
    at::TensorList gdn_norm_list,
    int64_t gdn_num_v_heads, int64_t gdn_key_head_dim,
    int64_t gdn_value_head_dim, int64_t gdn_conv_dim, int64_t gdn_conv_kernel,
    at::TensorList gdn_q_list, at::TensorList gdn_k_list, at::TensorList gdn_v_list,
    at::TensorList gdn_cq_list, at::TensorList gdn_ck_list, at::TensorList gdn_cv_list,
    at::TensorList gdn_b_bg_list, at::TensorList gdn_a_bg_list,
    at::TensorList q_scale_list, at::TensorList k_scale_list,
    at::TensorList v_scale_list, at::TensorList o_scale_list,
    at::TensorList attn_gate_scale_list,
    at::TensorList gate_scale_list, at::TensorList up_scale_list,
    at::TensorList down_scale_list,
    at::TensorList gdn_in_z_scale_list, at::TensorList gdn_out_scale_list,
    at::TensorList gdn_q_scale_list, at::TensorList gdn_k_scale_list,
    at::TensorList gdn_v_scale_list, at::TensorList gdn_b_bg_scale_list,
    at::TensorList gdn_a_bg_scale_list, at::TensorList nvfp4_tensor_scales,
    at::TensorList gate_negative_zero_list, at::TensorList up_negative_zero_list,
    at::TensorList down_negative_zero_list) {

    const int64_t N = static_cast<int64_t>(layer_is_full.size());
    TORCH_CHECK(N > 0, "qwen3_5 set_weights: layer_is_full must be non-empty");
    if (high_precision_) {
        const bool action_geometry = hidden_size == 1024 && intermediate_size == 4096 &&
            std::all_of(layer_is_full.begin(), layer_is_full.end(),
                        [](int64_t full) { return full == 1; });
        bool text_geometry = hidden_size == 2048 && intermediate_size == 6144 &&
            gdn_num_v_heads == 16 && gdn_key_head_dim == 128 &&
            gdn_value_head_dim == 128 && gdn_conv_dim == 6144 && gdn_conv_kernel == 4;
        for (int64_t layer = 0; text_geometry && layer < N; ++layer)
            text_geometry = layer_is_full[layer] == ((layer + 1) % 4 == 0 ? 1 : 0);
        TORCH_CHECK(num_cores() == 8 && N == 24 && num_q_heads == 8 &&
                        num_kv_heads == 8 && head_dim == 256 && eps == 1e-6 &&
                        use_silu && mrope_section == at::IntArrayRef({11, 11, 10}) &&
                        (action_geometry || text_geometry),
                    "G05 HIGH math requires exact TP8 Text or Action geometry");
        rpu_require_high_precision_math_kernels();
    }
    const bool reduced = num_cores() != 8;
    const bool dense27 = qwen35_legacy27_geometry(
        num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size);
    const bool reduced_legacy27 = reduced && dense27;
    // Admission and set_model_params keep logical geometry. Only the prepared
    // gate/up/down tensors carry cold MLP padding (0.8B MLP6: 3584 -> 3648).
    const int64_t prepared_intermediate_size =
        decoder_mlp_intermediate_size(intermediate_size, num_cores());
    if (reduced || dense27) {
        const int expected_layers = dense27 ? 64 : qwen35_reduced_profile_num_layers(
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size);
        const int64_t expected_value_heads =
            dense27 ? 48 : (hidden_size == 1024 || hidden_size == 2048) ? 16 : 32;
        TORCH_CHECK(expected_layers != 0 && N == expected_layers &&
                        !action_mode_ && !z2_bound_ && use_silu && eps == 1e-6 &&
                        mrope_section == at::IntArrayRef({11, 11, 10}) &&
                        gdn_num_v_heads == expected_value_heads &&
                        gdn_key_head_dim == 128 && gdn_value_head_dim == 128 &&
                        gdn_conv_kernel == 4 &&
                        gdn_conv_dim == (32 + expected_value_heads) * 128,
                    "Qwen3.5 execution requires an exact dense FP16, 2B W8 or admitted 27B quantized text profile");
        for (int64_t i = 0; i < N; ++i) {
            TORCH_CHECK(layer_is_full[i] == ((i + 1) % 4 == 0 ? 1 : 0),
                        "Qwen3.5 reduced execution requires the 3-GDN/1-full layer schedule");
        }
    }
    const bool uint4_mlp = !gate_negative_zero_list.empty() ||
        !up_negative_zero_list.empty() || !down_negative_zero_list.empty();
    if (uint4_mlp) {
        const bool exact_geometry =
            (N == 24 && hidden_size == 2048 && intermediate_size == 6144 && num_q_heads == 8) ||
            (N == 32 && hidden_size == 2560 && intermediate_size == 9216 && num_q_heads == 16) ||
            (N == 32 && hidden_size == 4096 && intermediate_size == 12288 && num_q_heads == 16);
        const int64_t value_heads = hidden_size == 2048 ? 16 : 32;
        TORCH_CHECK(num_cores() == 8 && nvfp4_tensor_scales.empty() && exact_geometry &&
                        !action_mode_ && !z2_bound_ && num_kv_heads == 8 && head_dim == 256 &&
                        use_silu && eps == 1e-6 && mrope_section == at::IntArrayRef({11, 11, 10}) &&
                        gdn_num_v_heads == value_heads && gdn_key_head_dim == 128 &&
                        gdn_value_head_dim == 128 && gdn_conv_kernel == 4 &&
                        gdn_conv_dim == (32 + value_heads) * 128,
                    "QuantTrio UINT4 MLP requires exact dense 2B/4B/9B TP8 text without NVFP4");
        for (int64_t layer = 0; layer < N; ++layer) {
            TORCH_CHECK(layer_is_full[layer] == ((layer + 1) % 4 == 0 ? 1 : 0),
                        "QuantTrio UINT4 MLP requires the 3-GDN/1-full layer schedule");
        }
        TORCH_CHECK(gate_negative_zero_list.size() == static_cast<size_t>(N) &&
                        up_negative_zero_list.size() == static_cast<size_t>(N) &&
                        down_negative_zero_list.size() == static_cast<size_t>(N),
                    "UINT4 MLP requires complete gate/up/down negative_zero lists");
    }
    const auto check_list = [N](at::TensorList list, const char* name) {
        TORCH_CHECK(static_cast<int64_t>(list.size()) == N,
                    "qwen3_5 set_weights: ", name,
                    " must have num_layers=", N, " entries, got ", list.size());
    };
    check_list(q_w_list, "q_w_list");
    check_list(k_w_list, "k_w_list");
    check_list(v_w_list, "v_w_list");
    check_list(o_w_list, "o_w_list");
    check_list(q_norm_list, "q_norm_list");
    check_list(k_norm_list, "k_norm_list");
    check_list(attn_gate_list, "attn_gate_list");
    check_list(input_norm_list, "input_norm_list");
    check_list(post_norm_list, "post_norm_list");
    check_list(gate_list, "gate_list");
    check_list(up_list, "up_list");
    check_list(down_list, "down_list");
    check_list(gdn_in_z_list, "gdn_in_z_list");
    check_list(gdn_out_list, "gdn_out_list");
    check_list(gdn_A_log_list, "gdn_A_log_list");
    check_list(gdn_dt_bias_list, "gdn_dt_bias_list");
    check_list(gdn_norm_list, "gdn_norm_list");
    check_list(gdn_q_list, "gdn_q_list");
    check_list(gdn_k_list, "gdn_k_list");
    check_list(gdn_v_list, "gdn_v_list");
    check_list(gdn_cq_list, "gdn_cq_list");
    check_list(gdn_ck_list, "gdn_ck_list");
    check_list(gdn_cv_list, "gdn_cv_list");
    check_list(gdn_b_bg_list, "gdn_b_bg_list");
    check_list(gdn_a_bg_list, "gdn_a_bg_list");
    check_list(q_scale_list, "q_scale_list");
    check_list(k_scale_list, "k_scale_list");
    check_list(v_scale_list, "v_scale_list");
    check_list(o_scale_list, "o_scale_list");
    check_list(attn_gate_scale_list, "attn_gate_scale_list");
    check_list(gate_scale_list, "gate_scale_list");
    check_list(up_scale_list, "up_scale_list");
    check_list(down_scale_list, "down_scale_list");
    check_list(gdn_in_z_scale_list, "gdn_in_z_scale_list");
    check_list(gdn_out_scale_list, "gdn_out_scale_list");
    check_list(gdn_q_scale_list, "gdn_q_scale_list");
    check_list(gdn_k_scale_list, "gdn_k_scale_list");
    check_list(gdn_v_scale_list, "gdn_v_scale_list");
    check_list(gdn_b_bg_scale_list, "gdn_b_bg_scale_list");
    check_list(gdn_a_bg_scale_list, "gdn_a_bg_scale_list");

    const bool nvfp4 = !nvfp4_tensor_scales.empty();
    G05Nvfp4Tables candidate_nvfp4;
    if (nvfp4) {
        TORCH_CHECK(!reduced && !z2_bound_ && N == 24 &&
                        num_q_heads == 8 && num_kv_heads == 8 && head_dim == 256 &&
                        hidden_size == 1024 && intermediate_size == 4096 &&
                        use_silu && eps == 1e-6 &&
                        mrope_section == at::IntArrayRef({11, 11, 10}) &&
                        cos.dim() == 2 && cos.size(1) == 32 &&
                        std::all_of(layer_is_full.begin(), layer_is_full.end(),
                            [](int64_t full) { return full == 1; }),
                    "G05 NVFP4 weights require exact full L24/H1024/I4096/Q8/KV8/D256 geometry");
        candidate_nvfp4.install(nvfp4_tensor_scales);
        for (int family = 0; family < 5; ++family) {
            const bool col = G05Nvfp4Tables::partitions[family] == 1;
            const auto tile = rpu_pl_tiling::select_tile_nvfp4_acc16(
                32, G05Nvfp4Tables::output_sizes[family] / (col ? 8 : 1),
                G05Nvfp4Tables::input_sizes[family] / (col ? 1 : 8));
            const auto kernel = linear_acc32_
                ? (col && G05Nvfp4Tables::output_sizes[family] == 4096
                    ? KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48
                    : KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64)
                : static_cast<KernelId>(
                    static_cast<int>(KernelId::PL_AT_NVFP4_ACC16_M176N128) + (128 - tile.n_tile) / 16);
            TORCH_CHECK(KernelCache::instance().has_loaded(kernel),
                        "G05 NVFP4 requires the selected genuine ",
                        linear_acc32_ ? "ACC32" : "ACC16", " expansion payload");
        }
    }

    const auto check_tensor = [](const at::Tensor& tensor, const char* name,
                                 int64_t layer, int64_t dim) {
        TORCH_CHECK(tensor.defined() && tensor.numel() > 0,
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] must be non-empty");
        TORCH_CHECK(tensor.device().type() == at::kPrivateUse1
                    && tensor.scalar_type() == at::kHalf
                    && tensor.is_contiguous(),
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] must be a contiguous FP16 device tensor");
        TORCH_CHECK(tensor.dim() == dim,
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] must be ", dim, "D, got ", tensor.dim(), "D");
    };
    // Exact 2B W8 main/GDN profile; b/a scalar projections remain FP16.
    // Inspect only after list cardinalities have been validated above.
    const bool reduced_w8_2b = reduced && !reduced_legacy27 &&
        hidden_size == 2048 && gate_list[0].defined() &&
        gate_list[0].scalar_type() == at::kChar;
    // Geometry is shared by the historical mixed bundle and the official
    // uniform checkpoint. Determine precision only after validating list sizes,
    // then enforce the selected policy for every projection in every layer.
    const bool dense27_w8 = dense27 && gdn_q_list[0].defined() &&
        gdn_q_list[0].scalar_type() == at::kChar;
    const bool dense27_w4 = dense27 && gdn_q_list[0].defined() &&
        gdn_q_list[0].scalar_type() == at::kByte;
    TORCH_CHECK(!dense27_w4 || !reduced,
                "official Qwen3.5-27B W4 group32 requires eight cores");
    const auto check_linear = [reduced, dense27, dense27_w8, dense27_w4, reduced_w8_2b, nvfp4, uint4_mlp](const at::Tensor& weight,
                                 const at::Tensor& scale,
                                 const char* name, int64_t layer) {
        TORCH_CHECK(weight.defined() && weight.numel() > 0 && weight.dim() == 2,
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] weight must be a non-empty 2D tensor");
        const std::string role(name);
        if (nvfp4) {
            const int family = role == "q" ? 0 : role == "o" ? 1 :
                role == "gate" ? 2 : role == "up" ? 3 : role == "down" ? 4 : -1;
            if (family >= 0) {
                const int64_t n = G05Nvfp4Tables::output_sizes[family];
                const int64_t k = G05Nvfp4Tables::input_sizes[family];
                TORCH_CHECK(weight.scalar_type() == at::kByte &&
                                weight.sizes() == at::IntArrayRef({n, k / 2}) &&
                                scale.defined() && scale.scalar_type() == at::kByte &&
                                scale.device().type() == at::kPrivateUse1 &&
                                scale.is_contiguous() &&
                                scale.sizes() == at::IntArrayRef({k / 16, n}),
                            "G05 NVFP4 requires packed uint8 [N,K/2] and striped FP8 [K/16,N] for ", name);
                const auto cpu = scale.cpu();
                const auto* values = cpu.data_ptr<uint8_t>();
                for (int64_t element = 0; element < cpu.numel(); ++element)
                    TORCH_CHECK((values[element] & 0x7f) != 0x7f,
                                "G05 NVFP4 block scale contains FP8 NaN");
            } else {
                TORCH_CHECK((role == "k" || role == "v" || role == "attn_gate") &&
                                weight.scalar_type() == at::kChar &&
                                weight.sizes() == at::IntArrayRef({2048, 1024}),
                            "G05 NVFP4 preserves W8 K/V/attention-gate owners");
            }
        }
        if (uint4_mlp && role != "gate" && role != "up" && role != "down") {
            TORCH_CHECK(weight.scalar_type() == at::kHalf,
                        "QuantTrio AWQ only quantizes MLP projections; ", name,
                        "[", layer, "] must remain FP16");
        }
        const bool legacy_main = dense27 && !dense27_w8 && !dense27_w4 &&
            role.rfind("gdn_", 0) != 0;
        const bool w8_2b_projection = reduced_w8_2b &&
            role != "gdn_b_bg" && role != "gdn_a_bg";
        const auto expected_dtype = dense27_w4 ? at::kByte :
            (dense27_w8 || legacy_main || w8_2b_projection ? at::kChar : at::kHalf);
        TORCH_CHECK((!reduced && !dense27) || weight.scalar_type() == expected_dtype,
                    "Qwen3.5 reduced projection dtype must match exact FP16, "
                    "2B W8 with FP16 GDN b/a, official 27B uniform W8/W4, "
                    "or legacy W8-main FP16-GDN policy");
        TORCH_CHECK(weight.device().type() == at::kPrivateUse1
                    && weight.is_contiguous()
                    && (weight.scalar_type() == at::kHalf
                        || weight.scalar_type() == at::kChar
                        || weight.scalar_type() == at::kByte),
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] weight must be contiguous FP16, int8, or packed uint8 on device");
        if (weight.scalar_type() == at::kChar) {
            TORCH_CHECK(scale.defined() && scale.dim() == 1
                        && scale.numel() == weight.size(0)
                        && scale.scalar_type() == at::kHalf
                        && scale.device().type() == at::kPrivateUse1
                        && scale.is_contiguous(),
                        "qwen3_5 set_weights: W8A16 ", name, "[", layer,
                        "] requires contiguous FP16 device scale [N=",
                        weight.size(0), "]");
        } else if (weight.scalar_type() == at::kByte && nvfp4) {
            // Exact family shape and FP8 scale validation above; never W4 ABI.
        } else if (weight.scalar_type() == at::kByte) {
            TORCH_CHECK(!dense27_w4 || (scale.defined() && scale.dim() == 2 && scale.size(0) == 32),
                        "official Qwen3.5-27B W4 requires group32 scales");
            if (dense27_w4) {
                const bool row = role == "down" || role == "o" || role == "gdn_out";
                const int64_t local_n = weight.size(0) / (row ? 1 : 8);
                const int64_t local_g = (weight.size(1) * 2 / 32) / (row ? 8 : 1);
                const int64_t scale_elements = ((local_g + 3) / 4) *
                    ((local_n + 63) / 64) * 8 * 4 * 64;
                TORCH_CHECK(scale.numel() == scale_elements,
                            "official Qwen3.5-27B W4 scale has incorrect controller padding for ", name);
            }
            TORCH_CHECK(scale.defined() && scale.dim() == 2
                        && (scale.size(0) == 32 || scale.size(0) == 64
                            || scale.size(0) == 128)
                        && scale.numel() > 0
                        && scale.scalar_type() == at::kHalf
                        && scale.device().type() == at::kPrivateUse1
                        && scale.is_contiguous(),
                        "qwen3_5 set_weights: packed W4A16 ", name, "[", layer,
                        "] requires current-main controller-striped FP16 scale "
                        "[group_size, physical_elements/group_size], got ",
                        scale.defined() ? scale.sizes() : at::IntArrayRef{});
        } else {
            TORCH_CHECK(!scale.defined() || scale.numel() == 0,
                        "qwen3_5 set_weights: FP16 ", name, "[", layer,
                        "] must not carry a quant scale");
        }
    };
    for (int64_t i = 0; i < N; ++i) {
        TORCH_CHECK(layer_is_full[i] == 0 || layer_is_full[i] == 1,
                    "qwen3_5 set_weights: layer_is_full[", i,
                    "] must be 0 or 1, got ", layer_is_full[i]);
        if (nvfp4) {
            TORCH_CHECK(input_norm_list[i].defined() && post_norm_list[i].defined() &&
                            q_norm_list[i].defined() && k_norm_list[i].defined() &&
                            input_norm_list[i].sizes() == at::IntArrayRef({1024}) &&
                            post_norm_list[i].sizes() == at::IntArrayRef({1024}) &&
                            q_norm_list[i].sizes() == at::IntArrayRef({256}) &&
                            k_norm_list[i].sizes() == at::IntArrayRef({256}),
                        "G05 NVFP4 requires H1024 and D256 normalization owners");
        }
        if (reduced || dense27) {
            const auto shape = [i](const at::Tensor& tensor,
                                    at::IntArrayRef expected, const char* name) {
                std::vector<int64_t> physical(expected.begin(), expected.end());
                if (physical.size() == 2 && tensor.scalar_type() == at::kByte)
                    physical[1] /= 2;
                TORCH_CHECK(tensor.defined() && tensor.sizes() == at::IntArrayRef(physical),
                            "Qwen3.5 reduced ", name, " shape mismatch at layer ", i);
            };
            shape(input_norm_list[i], {hidden_size}, "input norm");
            shape(post_norm_list[i], {hidden_size}, "post norm");
            shape(gate_list[i], {prepared_intermediate_size, hidden_size}, "MLP gate");
            shape(up_list[i], {prepared_intermediate_size, hidden_size}, "MLP up");
            shape(down_list[i], {hidden_size, prepared_intermediate_size}, "MLP down");
            if (layer_is_full[i]) {
                shape(q_w_list[i], {num_q_heads * head_dim, hidden_size}, "Q");
                shape(k_w_list[i], {num_kv_heads * head_dim, hidden_size}, "K");
                shape(v_w_list[i], {num_kv_heads * head_dim, hidden_size}, "V");
                shape(o_w_list[i], {hidden_size, num_q_heads * head_dim}, "O");
                shape(attn_gate_list[i], {num_q_heads * head_dim, hidden_size}, "attention gate");
                shape(q_norm_list[i], {head_dim}, "Q norm");
                shape(k_norm_list[i], {head_dim}, "K norm");
            } else {
                const int64_t gc = gdn_tp();
                const int64_t key = 16 * gdn_key_head_dim;
                const int64_t value = gdn_num_v_heads * gdn_value_head_dim;
                shape(gdn_q_list[i], {key, hidden_size}, "GDN Q");
                shape(gdn_k_list[i], {key, hidden_size}, "GDN K");
                shape(gdn_v_list[i], {value, hidden_size}, "GDN V");
                shape(gdn_in_z_list[i], {value, hidden_size}, "GDN Z");
                shape(gdn_out_list[i], {hidden_size, value}, "GDN output");
                shape(gdn_b_bg_list[i], {16 * gc, hidden_size}, "GDN beta");
                shape(gdn_a_bg_list[i], {16 * gc, hidden_size}, "GDN decay");
                const int64_t scalar_slots = qwen35_gdn_scalar_slots(gdn_num_v_heads, gc);
                shape(gdn_A_log_list[i], {scalar_slots * gc}, "GDN A_log");
                shape(gdn_dt_bias_list[i], {scalar_slots * gc}, "GDN dt_bias");
                shape(gdn_norm_list[i], {gdn_value_head_dim}, "GDN norm");
                shape(gdn_cq_list[i], {gc, gdn_conv_kernel, key / gc}, "GDN conv Q");
                shape(gdn_ck_list[i], {gc, gdn_conv_kernel, key / gc}, "GDN conv K");
                shape(gdn_cv_list[i], {gc, gdn_conv_kernel, value / gc}, "GDN conv V");
            }
        }
        check_tensor(input_norm_list[i], "input_norm_list", i, 1);
        check_tensor(post_norm_list[i], "post_norm_list", i, 1);
        check_linear(gate_list[i], gate_scale_list[i], "gate", i);
        check_linear(up_list[i], up_scale_list[i], "up", i);
        check_linear(down_list[i], down_scale_list[i], "down", i);
        if (layer_is_full[i]) {
            check_linear(q_w_list[i], q_scale_list[i], "q", i);
            check_linear(k_w_list[i], k_scale_list[i], "k", i);
            check_linear(v_w_list[i], v_scale_list[i], "v", i);
            check_linear(o_w_list[i], o_scale_list[i], "o", i);
            check_tensor(q_norm_list[i], "q_norm_list", i, 1);
            check_tensor(k_norm_list[i], "k_norm_list", i, 1);
            check_linear(attn_gate_list[i], attn_gate_scale_list[i],
                         "attn_gate", i);
        } else {
            check_linear(gdn_in_z_list[i], gdn_in_z_scale_list[i],
                         "gdn_in_z", i);
            check_linear(gdn_out_list[i], gdn_out_scale_list[i],
                         "gdn_out", i);
            check_tensor(gdn_A_log_list[i], "gdn_A_log_list", i, 1);
            check_tensor(gdn_dt_bias_list[i], "gdn_dt_bias_list", i, 1);
            check_tensor(gdn_norm_list[i], "gdn_norm_list", i, 1);
            check_linear(gdn_q_list[i], gdn_q_scale_list[i], "gdn_q", i);
            check_linear(gdn_k_list[i], gdn_k_scale_list[i], "gdn_k", i);
            check_linear(gdn_v_list[i], gdn_v_scale_list[i], "gdn_v", i);
            check_tensor(gdn_cq_list[i], "gdn_cq_list", i, 3);
            check_tensor(gdn_ck_list[i], "gdn_ck_list", i, 3);
            check_tensor(gdn_cv_list[i], "gdn_cv_list", i, 3);
            check_linear(gdn_b_bg_list[i], gdn_b_bg_scale_list[i],
                         "gdn_b_bg", i);
            check_linear(gdn_a_bg_list[i], gdn_a_bg_scale_list[i],
                         "gdn_a_bg", i);
        }
    }
    check_tensor(cos, "cos", -1, 2);
    check_tensor(sin, "sin", -1, 2);
    check_tensor(final_norm_w, "final_norm_w", -1, 1);
    TORCH_CHECK(!nvfp4 || final_norm_w.sizes() == at::IntArrayRef({1024}),
                "G05 NVFP4 requires final normalization [1024]");
    TORCH_CHECK(cos.sizes() == sin.sizes(),
                "qwen3_5 set_weights: cos/sin shapes must match");
    TORCH_CHECK((!reduced && !dense27) || (cos.size(1) == 32 && final_norm_w.numel() == hidden_size),
                "Qwen3.5 reduced RoPE/final norm geometry mismatch");

    // Validate and align every zero owner before publishing any new weights.
    // Empty first-layer placeholders may be ordinary CPU tensors, like scales.
    const auto prepare_zeros = [&](at::TensorList weights, at::TensorList scales,
                                   at::TensorList zeros, const char* role,
                                   int64_t out_features, int64_t in_features) {
        std::vector<at::Tensor> owners(N);
        if (!uint4_mlp) return owners;
        for (int64_t layer = 0; layer < N; ++layer) {
            const auto& zero = zeros[layer];
            const bool present = zero.defined() && zero.numel() != 0;
            if (layer == 0) {
                TORCH_CHECK(!present && weights[layer].scalar_type() == at::kHalf &&
                                weights[layer].sizes() == at::IntArrayRef({out_features, in_features}),
                            "QuantTrio AWQ layer 0 MLP must remain FP16 without zero: ", role);
                continue;
            }
            const auto& weight = weights[layer];
            const auto& scale = scales[layer];
            const bool row = std::string(role) == "down";
            const int64_t local_n = row ? out_features : out_features / num_cores();
            const int64_t local_k = row ? in_features / num_cores() : in_features;
            const int64_t metadata_elements = ((local_k / 128 + 3) / 4) *
                ((local_n + 63) / 64) * num_cores() * 4 * 64;
            TORCH_CHECK(local_k % 128 == 0 && local_n % 16 == 0,
                        "QuantTrio AWQ projection does not preserve TP8 G128 geometry");
            TORCH_CHECK(present && weight.scalar_type() == at::kByte &&
                            weight.sizes() == at::IntArrayRef({out_features, in_features / 2}) &&
                            scale.defined() && scale.dim() == 2 && scale.size(0) == 128 &&
                            scale.numel() == metadata_elements &&
                            zero.device() == weight.device() && zero.scalar_type() == at::kHalf &&
                            zero.is_contiguous() && zero.sizes() == scale.sizes(),
                        "QuantTrio AWQ ", role, "[", layer,
                        "] requires packed UINT4, G128 FP16 scale and matching device FP16 negative_zero");
            owners[layer] = rpu_retain_linear_quant_scale(weight, zero);
        }
        return owners;
    };
    auto gate_zero_owners = prepare_zeros(gate_list, gate_scale_list,
        gate_negative_zero_list, "gate", prepared_intermediate_size, hidden_size);
    auto up_zero_owners = prepare_zeros(up_list, up_scale_list,
        up_negative_zero_list, "up", prepared_intermediate_size, hidden_size);
    auto down_zero_owners = prepare_zeros(down_list, down_scale_list,
        down_negative_zero_list, "down", hidden_size, prepared_intermediate_size);

    action_nvfp4_ = std::move(candidate_nvfp4);
    action_nvfp4_o_resident_available_ = false;
    action_nvfp4_o_resident_ = {};
    action_nvfp4_down_resident_available_ = false;
    action_nvfp4_down_resident_ = {};
    has_qk_norm_ = q_norm_list.size() > 0;
    use_silu_    = use_silu;
    eps_         = eps;

    layer_is_full_.assign(layer_is_full.begin(), layer_is_full.end());
    has_gdn_ = std::any_of(
        layer_is_full_.begin(), layer_is_full_.end(),
        [](uint8_t is_full) { return !is_full; });
    action_mode_ = false;
    action_norm_shift_available_ = false;
    action_newton_norm_shift_available_ = false;
    action_gated_residual_available_ = false;
    action_rtc_norm_shift_available_ = false;
    action_sub_gated_residual_available_ = false;
    action_rtc_sub_gated_residual_available_ = false;
    action_ring_rtc_residual_available_ = false;
    action_mod_body_w8_ = false;
    action_mod_body_fp16_ = false;
    action_gateup_resident_available_ = false;
    action_gateup_resident_ = {};
    action_o_resident_available_ = false;
    action_o_resident_n128_ = false;
    action_o_resident_.clear();
    action_down_partial_available_ = false;
    action_down_partial_ = at::Tensor();
    action_gateup_resident_w8_ = false;
    action_kv_owner_w8_ = false;
    action_q_gate_w8_ = false;
    action_gateup_pair_w8_ = false;
    action_gateup_pair_nvfp4_ = false;
    action_rtc_active_ = false;
    action_kv_owner_available_ = false;
    action_q_gate_available_ = false;
    action_qkv_gate_available_ = false;
    action_owner_cache_available_ = false;
    action_owner_cache_joint_available_ = false;
    action_gate_kv_owner_available_ = false;
    action_nvfp4_qk_norm_rope_available_ = false;
    action_gateup_pair_available_ = false;
    action_silu_round_mul_available_ = false;
    action_silu_high_mul_available_ = false;
    action_nvfp4_swiglu_available_ = false;
    action_attn_sigmoid_mul_available_ = false;
    action_tail_available_ = false;
    action_mlp_tail_available_ = false;
    text_mlp_tail_available_ = false;
    text_silu_round_mul_available_ = false;
    text_gateup_pair_available_ = false;
    text_gateup_pair_w8_ = false;
    text_gdn_owner_rms_available_ = false;
    action_rtc_prefix_len_ = 0;
    action_rtc_prefix_ref_ = at::Tensor();
    action_rtc_prefix_live_base_ = 0;
    adaptive_mod_ref_ = at::Tensor();
    adaptive_mod_row_offset_ = 0;
    action_prefix_lens_.clear();
    layer_weights_.assign(N, LayerWeights{});
    for (int64_t i = 0; i < N; ++i) {
        auto& lw = layer_weights_[i];
        // Every layer (full + GDN) has input_layernorm (unified channel), post_norm, MLP.
        lw.input_norm_w = input_norm_list[i];
        lw.post_norm_w = post_norm_list[i];
        lw.gate_w = gate_list[i]; lw.up_w = up_list[i]; lw.down_w = down_list[i];
        lw.gate_negative_zero = std::move(gate_zero_owners[i]);
        lw.up_negative_zero = std::move(up_zero_owners[i]);
        lw.down_negative_zero = std::move(down_zero_owners[i]);
        lw.gate_ws = rpu_retain_linear_quant_scale(
            gate_list[i], gate_scale_list[i]);
        lw.up_ws = rpu_retain_linear_quant_scale(
            up_list[i], up_scale_list[i]);
        lw.down_ws = rpu_retain_linear_quant_scale(
            down_list[i], down_scale_list[i]);
        if (!layer_is_full_[i]) {
            // GDN mixer weights (attention slots stay empty).
            lw.gdn_in_z_w    = gdn_in_z_list[i];
            lw.gdn_out_w     = gdn_out_list[i];
            lw.gdn_A_log     = gdn_A_log_list[i];
            lw.gdn_dt_bias   = gdn_dt_bias_list[i];
            lw.gdn_norm_w    = gdn_norm_list[i];
            lw.gdn_in_z_ws = rpu_retain_linear_quant_scale(
                gdn_in_z_list[i], gdn_in_z_scale_list[i]);
            lw.gdn_out_ws = rpu_retain_linear_quant_scale(
                gdn_out_list[i], gdn_out_scale_list[i]);
            // prefill (chunk) per-path weights (separate q/k/v proj + conv, N_bg b/a)
            if (gdn_q_list.size() > 0) {
                lw.gdn_q_w = gdn_q_list[i]; lw.gdn_k_w = gdn_k_list[i]; lw.gdn_v_w = gdn_v_list[i];
                lw.gdn_conv_q_w = gdn_cq_list[i]; lw.gdn_conv_k_w = gdn_ck_list[i]; lw.gdn_conv_v_w = gdn_cv_list[i];
                lw.gdn_b_bg_w = gdn_b_bg_list[i]; lw.gdn_a_bg_w = gdn_a_bg_list[i];
                lw.gdn_q_ws = rpu_retain_linear_quant_scale(
                    gdn_q_list[i], gdn_q_scale_list[i]);
                lw.gdn_k_ws = rpu_retain_linear_quant_scale(
                    gdn_k_list[i], gdn_k_scale_list[i]);
                lw.gdn_v_ws = rpu_retain_linear_quant_scale(
                    gdn_v_list[i], gdn_v_scale_list[i]);
                lw.gdn_b_bg_ws = rpu_retain_linear_quant_scale(
                    gdn_b_bg_list[i], gdn_b_bg_scale_list[i]);
                lw.gdn_a_bg_ws = rpu_retain_linear_quant_scale(
                    gdn_a_bg_list[i], gdn_a_bg_scale_list[i]);
            }
            continue;
        }
        lw.q_w = q_w_list[i]; lw.k_w = k_w_list[i];
        lw.v_w = v_w_list[i]; lw.o_w = o_w_list[i];
        lw.q_ws = rpu_retain_linear_quant_scale(
            q_w_list[i], q_scale_list[i]);
        lw.k_ws = rpu_retain_linear_quant_scale(
            k_w_list[i], k_scale_list[i]);
        lw.v_ws = rpu_retain_linear_quant_scale(
            v_w_list[i], v_scale_list[i]);
        lw.o_ws = rpu_retain_linear_quant_scale(
            o_w_list[i], o_scale_list[i]);
        lw.q_norm_w    = has_qk_norm_ ? q_norm_list[i] : at::Tensor();
        lw.k_norm_w    = has_qk_norm_ ? k_norm_list[i] : at::Tensor();
        lw.attn_gate_w = attn_gate_list[i];
        lw.attn_gate_ws = rpu_retain_linear_quant_scale(
            attn_gate_list[i], attn_gate_scale_list[i]);
    }
    cos_ = keep_256b_aligned_rpu_copy(cos, "qwen3_5 set_weights: cos");
    sin_ = keep_256b_aligned_rpu_copy(sin, "qwen3_5 set_weights: sin");
    final_norm_w_ = final_norm_w;

    // GDN mixer dims + persistent DDR scratch (residual bridge + zeros).
    gdn_nvh_      = gdn_num_v_heads;
    gdn_dk_       = gdn_key_head_dim;
    gdn_dv_       = gdn_value_head_dim;
    gdn_conv_dim_ = gdn_conv_dim;
    gdn_kc_       = gdn_conv_kernel;
    gdn_value_dim_ = gdn_nvh_ * gdn_dv_;
    if (has_gdn_) {
        auto opt = final_norm_w.options();   // fp16 on device
        gdn_zero_ = at::zeros({gdn_dk_ * gdn_dv_}, opt);
        // prefill-chunk constant masks (C=64 fixed). tril keeps the diagonal (decay
        // self-score exp(0)=1); strict is the strict-lower (i>j). Both match rhino's
        // host-prepared constant masks (the -1 is a separate binary_scalar, not folded
        // here). Built on CPU then moved to device (host-side fill; values exact).
        {
            const int64_t C = 64;
            auto cpuf = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
            auto ones = at::ones({C, C}, cpuf);
            gdn_tril_   = at::tril(ones).reshape({C * C}).to(opt);      // i>=j -> 1
            gdn_strict_ = at::tril(ones, -1).reshape({C * C}).to(opt);  // i>j  -> 1
        }
        // per-layer LIVE state addrs for the mutable GDN state/conv DMAs. Sized ONCE
        // here (never resized) so the graph-recorded &..._live_addr_[L] stay stable;
        // build_gdn refreshes each entry per forward before emitting the mutable DMA.
        gdn_state_live_addr_.assign(N, 0);
        conv_state_live_addr_.assign(N, 0);
        // per-layer neg_exp_A = -exp(A_log) (host; chunk multiplies g by this).
        gdn_neg_exp_A_.assign(N, at::Tensor());
        for (int64_t i = 0; i < N; ++i) {
            if (layer_is_full_[i]) continue;
            const auto& al = layer_weights_[i].gdn_A_log;
            if (al.defined() && al.numel() > 0)
                gdn_neg_exp_A_[i] = at::exp(al.to(at::kCPU).to(at::kFloat)).mul(-1.0).to(opt);
        }
    }

    set_model_params(num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size);
    set_num_layers(N);

    // M-RoPE setup. mrope_section is [T,H,W] in rotary_dim/2 units (Qwen3.5:
    // [11,11,10], sum=32=rotary_dim/2), so rotary_dim=2*sum(mrope_section).
    // The host bakes the T/H/W interleaving into [N, rotary_dim/2] cos/sin
    // tables; the partial kernels receive rotary_dim and need no strobe masks
    // or position_ids.
    has_mrope_ = !mrope_section.empty();
    if (has_mrope_) {
        int64_t sec_sum = 0;
        for (auto s : mrope_section) sec_sum += s;
        rotary_dim_ = 2 * sec_sum;
    }

    prepare_text_gateup_pair();
    prepare_text_owner_rms();
    invalidate_model_state();   // D-503: MUST be the last statement.
}

}  // namespace v3

// =============================================================================
// Instance registry + C API for TORCH_LIBRARY_IMPL (registered in rpu_backend.cpp)
// =============================================================================
using Qwen3_5Registry = ModelHandleRegistry<v3::Qwen3_5Model>;

void rpu_qwen3_5_prepare_persistent_spm(int64_t handle, int64_t execution_len) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_prepare_persistent_spm")
        ->prepare_persistent_spm(execution_len, 0, true);
}

std::vector<int64_t> rpu_qwen3_5_planner_cache_identity(int64_t handle) {
    return Qwen3_5Registry::get(handle, "rpu_qwen3_5_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_qwen3_5_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen3_5_kvinsert_exact_candidate(
    int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
    int64_t invocation, int64_t route) {
    return Qwen3_5Registry::get(handle, "rpu_qwen3_5_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_qwen3_5_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return Qwen3_5Registry::get(handle, "rpu_qwen3_5_kvinsert_cost_domain")
        ->kvinsert_cost_domain("qwen3_5", descriptor);
}

std::string rpu_qwen3_5_kvinsert_cost_catalog_sha256(int64_t handle) {
    return Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

namespace v3::qwen3_5_z2_internal {

SpmPipelineComponentLayout prepare_text(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len) {
    return Qwen3_5Registry::get(handle, "qwen3_5_z2_prepare_text")
        ->prepare_z2_layout(execution_len, real_len);
}

SpmDense2DSpec text_storage_spec(int64_t handle, int64_t execution_len) {
    return Qwen3_5Registry::get(handle, "qwen3_5_z2_text_storage_spec")
        ->z2_storage_spec(execution_len);
}

void adopt_text(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    Qwen3_5Registry::get(handle, "qwen3_5_z2_adopt_text")
        ->adopt_z2_layout(lease, scratch);
}

void bind_text(
    int64_t handle,
    const SpmPipelineLease& lease,
    const SpmPortView& storage) {
    Qwen3_5Registry::get(handle, "qwen3_5_z2_bind_text")
        ->bind_z2_port(lease, storage);
}

void validate_text(int64_t handle, const SpmPipelineLease& lease) {
    Qwen3_5Registry::get(handle, "qwen3_5_z2_validate_text")
        ->validate_z2_layout(lease);
}

void clear_text(int64_t handle, uint64_t epoch, uint64_t plan_hash) {
    Qwen3_5Registry::get(handle, "qwen3_5_z2_clear_text")
        ->clear_z2_layout(epoch, plan_hash);
}

at::Tensor forward_text_z2(
    int64_t handle,
    const at::Tensor& hidden,
    at::TensorList k_caches,
    at::TensorList v_caches,
    at::TensorList gdn_states,
    at::TensorList conv_states,
    uint64_t epoch,
    uint64_t plan_hash) {
    validate_text_dispatch(handle, epoch, plan_hash);
    auto* model = Qwen3_5Registry::get(handle, "qwen3_5_z2_forward_text");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    std::vector<at::Tensor> gs(gdn_states.begin(), gdn_states.end());
    std::vector<at::Tensor> cs(conv_states.begin(), conv_states.end());
    return model->forward_z2(hidden, kc, vc, gs, cs, epoch, plan_hash);
}

}  // namespace v3::qwen3_5_z2_internal

int64_t rpu_qwen3_5_create() { return Qwen3_5Registry::create(); }

void rpu_qwen3_5_set_execution_cores(int64_t handle, int64_t num_cores) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_execution_cores")
        ->set_execution_cores(num_cores);
}

std::vector<int64_t> rpu_qwen3_5_get_execution_topology(int64_t handle) {
    return Qwen3_5Registry::get(handle, "rpu_qwen3_5_get_execution_topology")
        ->execution_topology();
}
void    rpu_qwen3_5_destroy(int64_t handle) {
    v3::qwen3_5_z2_internal::check_text_destroy_allowed(handle);
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_destroy")
        ->check_execution_reconfigure_destroy_allowed(
            "rpu_qwen3_5_destroy");
    Qwen3_5Registry::destroy(handle, "rpu_qwen3_5_destroy");
}

// Per-handle SPM-budget-resolved prefill chunk_size (0 before any forward).
// Read from Python after a prefill forward to observe the multi-chunk split.
//
// 返回的是 **PREFILL** 的 cs。基类的 last_resolved_chunk_size_ 每次 compute_chunks 都覆写,
// 而 decode(seq_len=1)必然跑在 prefill 之后且固定选 cs=16 —— 直接读基类那份的话,凡是
// "prefill + decode 都跑过才来读"的调用方(如 perf harness 在 warmup 之后读)拿到的永远是
// 16,与 prefill 实际用的值无关。这里优先返回 forward 里单独记下的 prefill 那份;
// 尚未跑过 prefill 时(=0)退回基类值,保持只跑 prefill 的老调用方(parity 测试)行为不变。
int64_t rpu_qwen3_5_get_resolved_chunk_size(int64_t handle) {
    auto* m = Qwen3_5Registry::get(handle, "rpu_qwen3_5_get_resolved_chunk_size");
    const int64_t prefill_cs = m->last_prefill_chunk_size();
    return prefill_cs > 0 ? prefill_cs : m->get_last_resolved_chunk_size();
}

void rpu_qwen3_5_set_chunk_size_cap(int64_t handle, int64_t cap) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_chunk_size_cap")
        ->set_chunk_size_cap(cap);
}

void rpu_qwen3_5_set_prefill_chunk_size(int64_t handle, int64_t chunk_size) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_prefill_chunk_size")
        ->set_prefill_chunk_size(chunk_size);
}

void rpu_qwen3_5_set_chunk_envelope(int64_t handle, int64_t max_kv_len,
                                    int64_t chunk) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_qwen3_5_stage_prefill_execution_controls(
        int64_t handle, int64_t token, int64_t cap, int64_t chunk_size,
        int64_t max_kv_len, int64_t envelope_chunk) {
    TORCH_CHECK(token > 0, "Qwen3.5 hot-reconfigure token must be positive");
    Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_stage_prefill_execution_controls")
        ->stage_prefill_execution_controls(
            static_cast<uint64_t>(token), cap, chunk_size,
            max_kv_len, envelope_chunk);
}

void rpu_qwen3_5_set_linear_acc32(int64_t handle, bool enabled) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_linear_acc32")
        ->set_linear_acc32(enabled);
}

void rpu_qwen3_5_set_high_precision(int64_t handle, bool enabled) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_high_precision")
        ->set_high_precision(enabled);
}

void rpu_qwen3_5_set_fast_replay(int64_t handle, bool enabled) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_fast_replay")
        ->set_fast_replay(enabled);
}

void rpu_qwen3_5_set_retained_prefill_graph(
        int64_t handle, bool enabled) {
    Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_set_retained_prefill_graph")
        ->set_retained_prefill_graph(enabled);
}

void rpu_qwen3_5_enable_action_mode(int64_t handle) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_enable_action_mode")
        ->enable_action_mode();
}

void rpu_qwen3_5_set_action_chunk_size(
        int64_t handle, int64_t chunk_size) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_action_chunk_size")
        ->set_action_chunk_size(chunk_size);
}

void rpu_qwen3_5_stage_action_chunk_size(
        int64_t handle, int64_t token, int64_t chunk_size) {
    TORCH_CHECK(token > 0, "Qwen3.5 action hot-reconfigure token must be positive");
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_stage_action_chunk_size")
        ->stage_action_chunk_size(static_cast<uint64_t>(token), chunk_size);
}

void rpu_qwen3_5_enable_execution_reconfigure(int64_t handle) {
    Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_enable_execution_reconfigure")
        ->enable_execution_reconfigure();
}

void rpu_qwen3_5_set_action_io_weights(
    int64_t handle,
    const at::Tensor& input_w, const at::Tensor& input_b,
    const at::Tensor& output_w, const at::Tensor& output_b,
    int64_t action_dim, int64_t action_dim_pad, int64_t action_len) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_action_io_weights")
        ->set_action_io_weights(
            input_w, input_b, output_w, output_b,
            action_dim, action_dim_pad, action_len);
}

int64_t rpu_qwen3_5_resolve_prefill_chunk_size(
    int64_t handle, int64_t execution_len) {
    return Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_resolve_prefill_chunk_size")
        ->resolve_prefill_chunk_size(execution_len);
}

std::vector<int64_t> rpu_qwen3_5_resolve_prefill_stage_domain(
    int64_t handle, int64_t execution_len, int64_t logical_len,
    int64_t planning_chunk_size_override) {
    return Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_resolve_prefill_stage_domain")
        ->resolve_prefill_stage_domain(
            execution_len, logical_len, planning_chunk_size_override);
}

std::vector<int64_t> rpu_qwen3_5_resolve_decode_stage_descriptor(
        int64_t handle) {
    return Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_resolve_decode_stage_descriptor")
        ->resolve_decode_stage_descriptor();
}

std::vector<int64_t> rpu_qwen3_5_resolve_action_stage_domain(
        int64_t handle, int64_t action_len, int64_t max_prefix_len,
        int64_t requested_chunk_size, bool include_action_io,
        int64_t action_route_flags, int64_t graph_lifecycle, int64_t num_steps) {
    return Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_resolve_action_stage_domain")
        ->resolve_action_stage_domain(
            action_len, max_prefix_len, requested_chunk_size,
            include_action_io, action_route_flags, graph_lifecycle, num_steps);
}

void rpu_qwen3_5_set_weights(
    int64_t handle,
    at::TensorList q, at::TensorList k, at::TensorList v, at::TensorList o,
    at::TensorList qn, at::TensorList kn, at::TensorList ag,
    at::TensorList in_norm, at::TensorList post_norm,
    at::TensorList gate, at::TensorList up, at::TensorList down,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm,
    at::IntArrayRef layer_is_full,
    int64_t nqh, int64_t nkvh, int64_t hd, int64_t hs, int64_t is_,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::TensorList gdn_in_z, at::TensorList gdn_out, at::TensorList gdn_A_log,
    at::TensorList gdn_dt_bias, at::TensorList gdn_norm,
    int64_t gdn_nvh, int64_t gdn_dk, int64_t gdn_dv, int64_t gdn_conv_dim,
    int64_t gdn_conv_kernel,
    at::TensorList gdn_q, at::TensorList gdn_k, at::TensorList gdn_v,
    at::TensorList gdn_cq, at::TensorList gdn_ck, at::TensorList gdn_cv,
    at::TensorList gdn_b_bg, at::TensorList gdn_a_bg,
    at::TensorList q_scale, at::TensorList k_scale,
    at::TensorList v_scale, at::TensorList o_scale,
    at::TensorList attn_gate_scale,
    at::TensorList gate_scale, at::TensorList up_scale,
    at::TensorList down_scale,
    at::TensorList gdn_in_z_scale, at::TensorList gdn_out_scale,
    at::TensorList gdn_q_scale, at::TensorList gdn_k_scale,
    at::TensorList gdn_v_scale, at::TensorList gdn_b_bg_scale,
    at::TensorList gdn_a_bg_scale,
    const std::optional<std::vector<at::Tensor>>& nvfp4_tensor_scales,
    const std::optional<std::vector<at::Tensor>>& gate_negative_zero,
    const std::optional<std::vector<at::Tensor>>& up_negative_zero,
    const std::optional<std::vector<at::Tensor>>& down_negative_zero) {
    auto* m = Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_weights");
    m->set_weights(q, k, v, o, qn, kn, ag, in_norm, post_norm, gate, up, down,
                   cos, sin, final_norm, layer_is_full,
                   nqh, nkvh, hd, hs, is_, eps, use_silu, mrope_section,
                   gdn_in_z, gdn_out, gdn_A_log, gdn_dt_bias, gdn_norm,
                   gdn_nvh, gdn_dk, gdn_dv, gdn_conv_dim, gdn_conv_kernel,
                   gdn_q, gdn_k, gdn_v, gdn_cq, gdn_ck, gdn_cv,
                   gdn_b_bg, gdn_a_bg,
                   q_scale, k_scale, v_scale, o_scale, attn_gate_scale,
                   gate_scale, up_scale, down_scale,
                   gdn_in_z_scale, gdn_out_scale,
                   gdn_q_scale, gdn_k_scale, gdn_v_scale,
                   gdn_b_bg_scale, gdn_a_bg_scale,
                   nvfp4_tensor_scales.value_or(std::vector<at::Tensor>{}),
                   gate_negative_zero.value_or(std::vector<at::Tensor>{}),
                   up_negative_zero.value_or(std::vector<at::Tensor>{}),
                   down_negative_zero.value_or(std::vector<at::Tensor>{}));
}

void rpu_qwen3_5_set_prefill_rope(int64_t handle, const at::Tensor& cos, const at::Tensor& sin) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_prefill_rope")->set_prefill_rope(cos, sin);
}

void rpu_qwen3_5_set_valid_prefill_len(int64_t handle, int64_t n) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_valid_prefill_len")->set_valid_prefill_len(n);
}

void rpu_qwen3_5_set_mrope_position_delta(int64_t handle, int64_t d) {
    Qwen3_5Registry::get(handle, "rpu_qwen3_5_set_mrope_position_delta")->set_mrope_position_delta(d);
}

at::Tensor rpu_qwen3_5_forward(
    int64_t handle, const at::Tensor& hidden,
    at::TensorList k_caches, at::TensorList v_caches, at::TensorList gdn_states,
    at::TensorList conv_states,
    const std::optional<at::Tensor>& mask, int64_t position, bool is_causal,
    int64_t planned_chunk_size,
    at::IntArrayRef planned_stage_descriptor) {
    auto* m = Qwen3_5Registry::get(handle, "rpu_qwen3_5_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    std::vector<at::Tensor> gs(gdn_states.begin(), gdn_states.end());
    std::vector<at::Tensor> cs(conv_states.begin(), conv_states.end());
    return m->forward(
        hidden, kc, vc, gs, cs, mask, position, is_causal,
        planned_chunk_size, planned_stage_descriptor);
}

at::Tensor rpu_qwen3_5_action_forward(
    int64_t handle, const at::Tensor& hidden, const at::Tensor& adaptive_mod,
    at::TensorList k_caches, at::TensorList v_caches,
    at::IntArrayRef prefix_lens,
    at::IntArrayRef planned_stage_descriptor) {
    auto* m = Qwen3_5Registry::get(handle, "rpu_qwen3_5_action_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return m->forward_action(
        hidden, adaptive_mod, kc, vc, prefix_lens,
        planned_stage_descriptor);
}

at::Tensor rpu_qwen3_5_action_step_forward(
    int64_t handle, const at::Tensor& action,
    const at::Tensor& action_keep_mask, at::Tensor action_out,
    double delta_t, const at::Tensor& adaptive_mod,
    at::TensorList k_caches, at::TensorList v_caches,
    at::IntArrayRef prefix_lens, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor) {
    auto* m = Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_action_step_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return m->forward_action_step(
        action, action_keep_mask, action_out, delta_t,
        adaptive_mod, kc, vc, prefix_lens, num_steps,
        planned_stage_descriptor);
}

at::Tensor rpu_qwen3_5_action_rtc_step_forward(
    int64_t handle, const at::Tensor& action,
    const at::Tensor& action_keep_mask, at::Tensor action_out,
    const at::Tensor& action_prefix, int64_t action_prefix_len,
    double delta_t, const at::Tensor& adaptive_mod,
    at::TensorList k_caches, at::TensorList v_caches,
    at::IntArrayRef prefix_lens, int64_t num_steps,
    at::IntArrayRef planned_stage_descriptor) {
    auto* m = Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_action_rtc_step_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return m->forward_action_rtc_step(
        action, action_keep_mask, action_out,
        action_prefix, action_prefix_len, delta_t,
        adaptive_mod, kc, vc, prefix_lens, num_steps,
        planned_stage_descriptor);
}

at::Tensor rpu_qwen3_5_action_rtc_trace_forward(
    int64_t handle, const at::Tensor& action,
    const at::Tensor& action_keep_mask, at::Tensor action_out,
    const at::Tensor& action_prefix, int64_t action_prefix_len,
    double delta_t, const at::Tensor& adaptive_mod,
    at::TensorList k_caches, at::TensorList v_caches,
    at::IntArrayRef prefix_lens,
    at::Tensor trace_pre_action,
    at::Tensor trace_action_embed,
    at::Tensor trace_final_hidden,
    at::Tensor trace_velocity,
    at::Tensor trace_post_action,
    at::IntArrayRef planned_stage_descriptor) {
    auto* m = Qwen3_5Registry::get(
        handle, "rpu_qwen3_5_action_rtc_trace_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    return m->forward_action_rtc_trace(
        action, action_keep_mask, action_out,
        action_prefix, action_prefix_len, delta_t,
        adaptive_mod, kc, vc, prefix_lens,
        trace_pre_action, trace_action_embed, trace_final_hidden,
        trace_velocity, trace_post_action, planned_stage_descriptor);
}
