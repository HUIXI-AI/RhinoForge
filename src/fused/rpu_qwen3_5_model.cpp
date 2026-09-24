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
#include "rpu_runtime_state.h"   // diagnostic-only hidden-state export

#include <c10/util/ScopeExit.h>
#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>     // std::sqrt (GDN qk_scale)
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
constexpr int64_t QWEN35_TEXT_MLP_SILU_SITE = 5544299293668623348LL;

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
                name, " must be a contiguous FP16 RPU tensor");
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
                name, " failed to create a 256-byte aligned RPU copy");
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
    if (num_cores() == 8) return 0;
    int64_t hash = 0;
    for (const auto value : cold_topology_arguments())
        hash = detail::layout_mix(hash, value);
    return hash;
}

int64_t Qwen3_5Model::text_ring_route(int64_t rows, int64_t cols) const {
    return fmb_ring_all_reduce_route_selector(rows, cols, num_cores());
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
    linear_acc32_ = enabled;
    invalidate_model_state();
}

void Qwen3_5Model::set_fast_replay(bool enabled) {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 fast replay must be set before the first forward");
    fast_replay_enabled_ = enabled;
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
    TORCH_CHECK(action_len > 0 && action_len % 16 == 0,
                "Qwen3.5 action length must be a positive multiple of 16");
    auto check_rpu_half = [](const at::Tensor& tensor, const char* name) {
        TORCH_CHECK(tensor.defined()
                    && tensor.device().type() == at::kPrivateUse1
                    && tensor.scalar_type() == at::kHalf
                    && tensor.is_contiguous(),
                    name, " must be contiguous FP16 on RPU");
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
                "Qwen3.5 action adaptive_mod must be contiguous FP16 on RPU");
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
                && action.device().type() == at::kPrivateUse1
                && action.scalar_type() == at::kHalf
                && action.is_contiguous(),
                "Qwen3.5 action step input must be [1, ", action_len_, ", ",
                action_dim_pad_, "] contiguous FP16 on RPU");
    TORCH_CHECK(action_keep_mask.dim() == 1
                && action_keep_mask.size(0) == action_dim_pad_
                && action_keep_mask.device().type() == at::kPrivateUse1
                && action_keep_mask.scalar_type() == at::kHalf
                && action_keep_mask.is_contiguous(),
                "Qwen3.5 action keep mask must be [", action_dim_pad_,
                "] contiguous FP16 on RPU");
    TORCH_CHECK(action_out.sizes() == action.sizes()
                && action_out.device().type() == at::kPrivateUse1
                && action_out.scalar_type() == at::kHalf
                && action_out.is_contiguous(),
                "Qwen3.5 action step output must match the input shape and be "
                "contiguous FP16 on RPU");
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
                    && action_prefix->device().type() == at::kPrivateUse1
                    && action_prefix->scalar_type() == at::kHalf
                    && action_prefix->is_contiguous(),
                    "Qwen3.5 RTC action prefix must be [1, ",
                    action_prefix_len, ", ", action_dim_pad_,
                    "] contiguous FP16 on RPU");
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
                        "] contiguous FP16 on RPU");
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

    adaptive_mod_ref_ = adaptive_mod;
    adaptive_mod_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(adaptive_mod.data_ptr<c10::Half>());
    rpu_ddr_flush_force(adaptive_mod.data_ptr<c10::Half>());
    action_input_ref_ = action;
    action_input_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(action.data_ptr<c10::Half>());
    rpu_ddr_flush_force(action.data_ptr<c10::Half>());
    action_keep_mask_ref_ = action_keep_mask;
    action_keep_mask_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(action_keep_mask.data_ptr<c10::Half>());
    rpu_ddr_flush_force(action_keep_mask.data_ptr<c10::Half>());
    action_output_ref_ = action_out;
    action_output_live_base_ =
        ::rhino_lkn::RpuGetDevAddr(action_out.data_ptr<c10::Half>());
    rpu_ddr_flush_force(action_out.data_ptr<c10::Half>());
    if (rtc_active) {
        action_rtc_prefix_ref_ = *action_prefix;
        action_rtc_prefix_live_base_ = ::rhino_lkn::RpuGetDevAddr(
            action_prefix->data_ptr<c10::Half>());
        rpu_ddr_flush_force(action_prefix->data_ptr<c10::Half>());
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
            rpu_ddr_flush_force(tensor.data_ptr<c10::Half>());
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
                "Qwen3.5 text Z2 hidden must be contiguous RPU FP16 [1,128,2048]");
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
    const int64_t seq_len = hidden_states.size(1);
    TORCH_CHECK(seq_len == 1 || seq_len % 64 == 0,
                "Qwen3.5 text expects decode length 1 or a prefill execution "
                "length divisible by 64; use the Python adapter for ragged prefill");
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
    // entering run_all_layers. Only fast replay needs the unconditional CPU/RPU
    // boundary flush; ordinary text prefill/decode retain the configurable
    // intra-RPU flush used by build_gdn().
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
            rpu_ddr_flush_force(st->data_ptr<c10::Half>());
            rpu_ddr_flush_force(cv->data_ptr<c10::Half>());
        } else {
            rpu_ddr_flush(st->data_ptr<c10::Half>());
            rpu_ddr_flush(cv->data_ptr<c10::Half>());
        }
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
        // more RPU allocations and can run before a BUILD scope executes.
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

    if (ctx().has_complete_physical_manifest() &&
        layer_idx == 0 && chunk.idx == 0) {
        if (num_cores() != 8) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_TEXT_CORE_TOPOLOGY_SITE,
                /*selector=*/1, /*flags=*/0, cold_topology_arguments());
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
    if (action_mode_) {
        load_adaptive_mod_row(2 * layer_idx);
        apply_adaptive_norm(
            input_residual, addr(0, "input_norm"), chunk);
    } else {
        rpu_launch_rmsnorm_spm_kernel(
            input_residual, addr(0, "input_norm"),
            layer_addr(layer_idx, 0, "norm_w"), chunk.len, h, eps_,
            RpuRmsNormSpmRoute::BASE, num_cores());
    }
    // ③ token mixer (reads "input_norm"; ends with an all_reduce that adds the h_in residual).
    if (layer_is_full_[layer_idx]) {
        // full-attn's all_reduce leaves the stream in residual2 (== input_norm) directly.
        build_full_attention(layer_idx, chunk);
    } else {
        // decode = recurrent single step (chunk.len<=1); prefill = chunked delta-rule (L>1).
        // GDN's all_reduce leaves the stream in residual2 (== input_norm), like full-attn.
        if (chunk.len <= 1) build_gdn(layer_idx, chunk, /*decode=*/true);
        else                build_gdn(layer_idx, chunk, /*decode=*/false);
    }
    // Both mixers leave the post-mixer stream in residual2 (== input_norm).
    // ④ shared post_norm + MLP + final_norm + output (stream in input_norm).
    emit_mlp_and_output(layer_idx, chunk);
}

void Qwen3_5Model::launch_linear(
    int64_t chunk_idx,
    uint32_t input, const at::Tensor& weight, uint32_t output,
    int64_t m, int64_t n, int64_t k, int partition, int num_cores,
    uint32_t bias_spm_addr, const at::Tensor& scale) {
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
        /*prefer_gemv=*/selector == FmbLinearRouteSelector::GEMV);
}

void Qwen3_5Model::consume_manifest_route(
    FmbRouteFamily family, int64_t site_id,
    int64_t selector, int64_t invocation) {
    if (!ctx().has_complete_physical_manifest()) return;
    ctx().consume_physical_route(
        family, site_id, selector, /*resolved_flags=*/0,
        /*resolved_arguments=*/{}, invocation);
}

void Qwen3_5Model::load_adaptive_mod_row(int64_t row) {
    TORCH_CHECK(action_mode_ && adaptive_mod_ref_.defined(),
                "Qwen3.5 action modulation is unavailable");
    const int64_t width = 3 * hidden_size();
    const int64_t variants = action_rtc_active_ ? 2 : 1;
    const int64_t rows = 2 * num_layers() + 1;
    const int64_t step_row = action_loop_active_
        ? (ctx().body_iter * rows + row) * variants
        : row * variants;
    consume_manifest_route(
        FmbRouteFamily::MUTABLE_DMA,
        QWEN35_TEXT_ADAPTIVE_MOD_DMA_SITE,
        qwen35_text_dma_route(
            Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        &adaptive_mod_live_base_,
        step_row * width * static_cast<int64_t>(sizeof(c10::Half)),
        variants * width, addr(0, "adaptive_mod"), NUM_CORES);
}

void Qwen3_5Model::apply_adaptive_norm(
    uint32_t input, uint32_t output, const ChunkInfo& chunk) {
    const int64_t h = hidden_size();
    const int64_t clean_rows = action_rtc_active_
        ? std::clamp<int64_t>(
              action_rtc_prefix_len_ - chunk.offset, 0, chunk.len)
        : 0;
    auto emit_segment = [&](int64_t row_offset, int64_t row_count,
                            uint32_t modulation) {
        if (row_count == 0) return;
        const uint32_t data_offset = static_cast<uint32_t>(
            row_offset * h * static_cast<int64_t>(DWIDTH));
        const uint32_t shift = modulation + static_cast<uint32_t>(h * DWIDTH);
        rpu_launch_rmsnorm_spm_kernel(
            input + data_offset, output + data_offset,
            modulation, row_count, h, eps_);
        rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(
            shift, output + data_offset, output + data_offset,
            row_count, h, c10::Half(1.0), ValuOpType::ADD,
            /*is_bopa=*/false);
    };
    if (!action_rtc_active_) {
        emit_segment(/*row_offset=*/0, chunk.len, addr(0, "adaptive_mod"));
        return;
    }
    emit_segment(/*row_offset=*/0, clean_rows, addr(0, "adaptive_mod"));
    emit_segment(
        clean_rows, chunk.len - clean_rows,
        addr(0, "adaptive_mod") + static_cast<uint32_t>(3 * h * DWIDTH));
}

void Qwen3_5Model::apply_adaptive_residual_gate(
    uint32_t output, uint32_t residual, const ChunkInfo& chunk) {
    const int64_t h = hidden_size();
    const int64_t clean_rows = action_rtc_active_
        ? std::clamp<int64_t>(
              action_rtc_prefix_len_ - chunk.offset, 0, chunk.len)
        : 0;
    auto emit_segment = [&](int64_t row_offset, int64_t row_count,
                            uint32_t modulation) {
        if (row_count == 0) return;
        const uint32_t data_offset = static_cast<uint32_t>(
            row_offset * h * static_cast<int64_t>(DWIDTH));
        const uint32_t gate =
            modulation + static_cast<uint32_t>(2 * h * DWIDTH);
        apply_raw_residual_gate(
            output + data_offset, residual + data_offset, gate, row_count);
    };
    if (!action_rtc_active_) {
        emit_segment(/*row_offset=*/0, chunk.len, addr(0, "adaptive_mod"));
        return;
    }
    emit_segment(/*row_offset=*/0, clean_rows, addr(0, "adaptive_mod"));
    emit_segment(
        clean_rows, chunk.len - clean_rows,
        addr(0, "adaptive_mod") + static_cast<uint32_t>(3 * h * DWIDTH));
}

void Qwen3_5Model::apply_raw_residual_gate(
    uint32_t output, uint32_t residual, uint32_t gate, int64_t seq_len) {
    const int64_t elems = seq_len * hidden_size();
    rpu_launch_eltwise_binary_spm_kernel(
        output, residual, output, elems,
        ValuOpType::SUB, c10::Half(1.0), NUM_CORES);
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
void Qwen3_5Model::build_full_attention(int layer_idx, const ChunkInfo& chunk) {
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
    launch_linear(
        chunk.idx,
        addr(0, "input_norm"), lw.q_w, addr(0, "q"), seq_len, nq * hd, h,
        1, tp, 0, lw.q_ws);
    launch_linear(
        chunk.idx,
        addr(0, "input_norm"), lw.k_w, addr(0, "k"), seq_len, nkv * hd, h,
        1, tp, 0, lw.k_ws);
    launch_linear(
        chunk.idx,
        addr(0, "input_norm"), lw.v_w, addr(0, "v"), seq_len, nkv * hd, h,
        1, tp, 0, lw.v_ws);
    // Gated-attention gate projection. Compute here while "input_norm" still
    // holds the Phase-1 input RMSNorm output (Phase 3 k_norm overwrites it).
    // Applied as sigmoid(gate) ⊙ attn_out after SDPA (Qwen3.5 attn_output_gate).
    launch_linear(
        chunk.idx,
        addr(0, "input_norm"), lw.attn_gate_w, addr(0, "attn_gate"),
        seq_len, nq * hd, h, 1, tp, 0, lw.attn_gate_ws);

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

    if (has_qk_norm_) {
        int64_t local_kv_heads = nkv / tp;
        // Q: norm + partial RoPE stay in-place because the partial kernels write
        // only the rotary span. The legacy full-RoPE path remains out-of-place.
        if (has_mrope_) {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q"), addr(0, "q"),
                layer_addr(layer_idx, 0, "q_norm_w"),
                seq_len * local_q_heads, hd, eps_, RpuRmsNormSpmRoute::BASE, auxiliary_cores);
            emit_partial_rope(addr(0, "q"), local_q_heads);
        } else {
            rpu_launch_rmsnorm_spm_kernel(
                addr(0, "q"), addr(0, "output"),
                layer_addr(layer_idx, 0, "q_norm_w"),
                seq_len * local_q_heads, hd, eps_, RpuRmsNormSpmRoute::BASE, auxiliary_cores);
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
        if (local_kv_heads > 0) {
            if (has_mrope_) {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "k"), addr(0, "k"),
                    layer_addr(layer_idx, 0, "k_norm_w"),
                    seq_len * local_kv_heads, hd, eps_, RpuRmsNormSpmRoute::BASE, auxiliary_cores);
                emit_partial_rope(addr(0, "k"), local_kv_heads);
            } else {
                rpu_launch_rmsnorm_spm_kernel(
                    addr(0, "k"), addr(0, "input_norm"),
                    layer_addr(layer_idx, 0, "k_norm_w"),
                    seq_len * local_kv_heads, hd, eps_, RpuRmsNormSpmRoute::BASE, auxiliary_cores);
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
    rpu_launch_insert_kvcache_spm_unified_with_plan(
        k_cache, v_cache,
        addr_offset("k").value, addr_offset("v").value,
        nkv, hd, tp,
        /*k_cache_batch_offset_elems=*/0,
        /*v_cache_batch_offset_elems=*/0,
        // Action has no padded physical tail: the frozen plan's physical rows
        // equal its logical horizon. Z2 also has no padded SPM tail.
        /*spm_rows=*/0, kv_plan);

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
    {
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
    launch_linear(
        chunk.idx,
        addr(0, "output"), lw.o_w, addr(0, "oproj"), seq_len, h, nq * hd,
        0, tp, 0, lw.o_ws);

    // Phase 5: attention reduce + residual (tp inputs → owner outputs)
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

    // The all_reduce left the residual stream in residual2 (== input_norm); the shared
    // post_norm + MLP + final_norm + output tail runs back in build_layer_subgraph
    // (emit_mlp_and_output), same as the GDN path. build_full_attention is mixer-only.
}

// ── build_gdn: the unified Gated-DeltaNet token mixer. Shared setup/proj/tail + an
//    if(decode) split for the divergent conv + delta-rule core (decode = recurrent
//    single step; prefill = chunked delta-rule). Dispatched by build_layer_subgraph.
void Qwen3_5Model::build_gdn(int layer_idx, const ChunkInfo& chunk, bool decode) {
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
        rpu_launch_eltwise_unary_spm_kernel(cs_q + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_qa"), lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_k + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_ka"), lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_v + (uint32_t)((Kc - 1) * lval * 2), D("gdn_va"), lval, ValuOpType::SILU, false, nc);

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
        rpu_launch_eltwise_unary_spm_kernel(cs_q + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_c_query"), lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_k + (uint32_t)((Kc - 1) * lkey * 2), D("gdn_c_key"),   lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(cs_v + (uint32_t)((Kc - 1) * lval * 2), D("gdn_c_value"), lval, ValuOpType::SILU, false, nc);
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
        rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_query"), D("gdn_c_query"), L * lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_key"),   D("gdn_c_key"),   L * lkey, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_unary_spm_kernel(D("gdn_c_value"), D("gdn_c_value"), L * lval, ValuOpType::SILU, false, nc);
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

    auto pc_addr = [&](uint32_t a0, int cc) -> uint32_t {
        return a0 - SPM_ALLOC.addr(0, 0) + SPM_ALLOC.addr(cc, 0);
    };
    auto slice_pc = [&](uint32_t s, uint32_t d, const std::vector<int64_t>& is,
                        const std::vector<int64_t>& os, const std::vector<int64_t>& b) {
        for (int cc = 0; cc < nc; ++cc)
            rpu_launch_slice_spm_kernel(pc_addr(s, cc), pc_addr(d, cc), is, os, b, 1);
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


    const uint32_t out_buf   = decode ? D("gdn_out")   : D("gdn_c_out_tok");
    const uint32_t gated_buf = decode ? D("gdn_gated") : out_buf;
    const uint32_t z_buf     = z_out;
    const uint32_t m_buf     = decode ? D("gdn_m")     : D("gdn_c_m");
    if (decode) {
        rpu_launch_qwen3_5_rms_norm_gated(
            out_buf, z_buf, D("gdn_normw"), gated_buf, Hc, Dv, 1e-6, nc);
    } else {
        rpu_launch_rmsnorm_spm_kernel(
            out_buf, gated_buf, D("gdn_normw"), L * Hc, Dv, 1e-6, RpuRmsNormSpmRoute::BASE, nc);
        // Keep the proven prefill formula: gate = silu(z) * normed.
        rpu_launch_eltwise_unary_spm_kernel(
            z_buf, z_buf, L * lval, ValuOpType::SILU, false, nc);
        rpu_launch_eltwise_binary_spm_kernel(
            z_buf, gated_buf, gated_buf, L * lval, ValuOpType::MUL,
            c10::Half(1.0f), nc);
    }
    // out_proj (row-parallel): gated [L,value_dim] @ gdn_out_w → per-core partial m [L,hidden].
    // FULL K = value_dim (launcher splits local_k = value_dim/tp). See [[emitter-local-vs-graphop-full-dims]].
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
    // AllReduce the out_proj partials + h_in residual → post-mixer stream (== HF `mixer_out`).
    consume_manifest_route(
        FmbRouteFamily::ALL_REDUCE,
        QWEN35_TEXT_GDN_ALL_REDUCE_SITE,
        text_ring_route(L, hidden), chunk.idx);
    rpu_launch_all_reduce_sum_residual_kernel(
        m_buf, layer_input_residual_addr(layer_idx, chunk),
        D("residual2"), L, hidden, nc, num_cores());
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
// CONTRACT: the residual stream (h_in + mixer_out) must be in "input_norm" (== the
// "residual2" alias) on entry. Emits: residual1 = post_norm(stream); MLP(+residual) →
// residual1; final_norm at the last layer; then output DMA.
void Qwen3_5Model::emit_mlp_and_output(int layer_idx, const ChunkInfo& chunk) {
    const auto& lw = layer_weights_[layer_idx];
    const int64_t h = hidden_size(), seq_len = chunk.len;
    const bool is_last_layer = (layer_idx == num_layers() - 1);
    // post-attention RMSNorm: residual1 = post_norm(stream); input_norm keeps the stream.
    if (action_mode_) {
        load_adaptive_mod_row(2 * layer_idx + 1);
        apply_adaptive_norm(
            addr(0, "input_norm"), addr(0, "residual1"), chunk);
    } else {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "input_norm"), addr(0, "residual1"),
            layer_addr(layer_idx, 0, "post_norm_w"), seq_len, h, eps_, RpuRmsNormSpmRoute::BASE, num_cores());
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
    const at::Tensor gate_scale = action_mode_ ? at::Tensor{} : lw.gate_ws;
    const at::Tensor up_scale = action_mode_ ? at::Tensor{} : lw.up_ws;
    const at::Tensor down_scale = action_mode_ ? at::Tensor{} : lw.down_ws;
    launch_linear(
        chunk.idx,
        addr(0, "residual1"), lw.gate_w, addr(0, "gate"),
        seq_len, intermediate_size(), h, 1, mlp_tp(), 0, gate_scale);
    if (use_silu_) {
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::ACTIVATION, QWEN35_TEXT_MLP_SILU_SITE,
                /*resolved_selector=*/1, /*resolved_flags=*/0,
                {use_silu_ ? 1 : 0, mlp_elems,
                 static_cast<int64_t>(ValuOpType::SILU)}, chunk.idx);
        }
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "gate"), addr(0, "gate"), mlp_elems,
            ValuOpType::SILU, false, mlp_tp());
    }
    launch_linear(
        chunk.idx,
        addr(0, "residual1"), lw.up_w, addr(0, "up"),
        seq_len, intermediate_size(), h, 1, mlp_tp(), 0, up_scale);
    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "gate"), addr(0, "up"), product_buf,
        mlp_elems, ValuOpType::MUL, c10::Half(1.0), mlp_tp());
    launch_linear(
        chunk.idx,
        product_buf, lw.down_w, addr(0, "down"),
        seq_len, h, intermediate_size(), 0, mlp_tp(), 0, down_scale);
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
                addr(0, "final_norm_w"), seq_len, h, eps_, RpuRmsNormSpmRoute::BASE, num_cores());
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
    if (action_mode_) {
        decls.push_back({
            // RTC carries clean/noisy AdaLN triples side-by-side. Reserve both
            // halves for every action graph so toggling RTC never changes the
            // SPM manifest; non-RTC loads and reads only the first half.
            "adaptive_mod", A(6 * h * DWIDTH), 1, 31,
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
    // descriptor must not inherit the smaller profiles' GEMV route without separate admission.
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
        bool needs_auto_tile = false;
        bool needs_gemv = false;
        auto observe_linear = [&](const at::Tensor& weight,
                                  int64_t rows = -1) {
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
                observe_linear(lw.k_w);
                observe_linear(lw.v_w);
                observe_linear(lw.attn_gate_w);
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
        if (use_silu_) {
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
            if (has_qk_norm_ && has_mrope_) {
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
            append_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN35_TEXT_FULL_ALL_REDUCE_SITE,
                text_ring_route(chunk.len, hidden_size()),
                chunk.idx);
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
            append_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN35_TEXT_GDN_ALL_REDUCE_SITE,
                text_ring_route(chunk.len, hidden_size()),
                chunk.idx);
        }
        append_route(
            FmbRouteFamily::ALL_REDUCE,
            QWEN35_TEXT_MLP_ALL_REDUCE_SITE,
            text_ring_route(chunk.len, hidden_size()),
            chunk.idx);
    }

    if (action_mode_) {
        append_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_TEXT_ADAPTIVE_MOD_DMA_SITE,
            qwen35_text_dma_route(
                Qwen35TextMutableDmaRoute::DDR_BROADCAST_TO_SPM));
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
                "set_prefill_rope: cos/sin must be on RPU device");
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
    at::TensorList gdn_a_bg_scale_list) {

    const int64_t N = static_cast<int64_t>(layer_is_full.size());
    TORCH_CHECK(N > 0, "qwen3_5 set_weights: layer_is_full must be non-empty");
    const bool reduced = num_cores() != 8;
    const bool reduced_legacy27 = reduced && qwen35_legacy27_geometry(
        num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size);
    // Admission and set_model_params keep logical geometry. Only the prepared
    // gate/up/down tensors carry cold MLP padding (0.8B MLP6: 3584 -> 3648).
    const int64_t prepared_intermediate_size =
        decoder_mlp_intermediate_size(intermediate_size, num_cores());
    if (reduced) {
        const int expected_layers = qwen35_reduced_profile_num_layers(
            num_q_heads, num_kv_heads, head_dim, hidden_size, intermediate_size);
        const int64_t expected_value_heads =
            reduced_legacy27 ? 48 : (hidden_size == 1024 || hidden_size == 2048) ? 16 : 32;
        TORCH_CHECK(expected_layers != 0 && N == expected_layers &&
                        !action_mode_ && !z2_bound_ && use_silu && eps == 1e-6 &&
                        mrope_section == at::IntArrayRef({11, 11, 10}) &&
                        gdn_num_v_heads == expected_value_heads &&
                        gdn_key_head_dim == 128 && gdn_value_head_dim == 128 &&
                        gdn_conv_kernel == 4 &&
                        gdn_conv_dim == (32 + expected_value_heads) * 128,
                    "Qwen3.5 reduced execution requires an exact dense FP16, 2B W8 or legacy27 W8 text profile");
        for (int64_t i = 0; i < N; ++i) {
            TORCH_CHECK(layer_is_full[i] == ((i + 1) % 4 == 0 ? 1 : 0),
                        "Qwen3.5 reduced execution requires the 3-GDN/1-full layer schedule");
        }
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

    const auto check_tensor = [](const at::Tensor& tensor, const char* name,
                                 int64_t layer, int64_t dim) {
        TORCH_CHECK(tensor.defined() && tensor.numel() > 0,
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] must be non-empty");
        TORCH_CHECK(tensor.device().type() == at::kPrivateUse1
                    && tensor.scalar_type() == at::kHalf
                    && tensor.is_contiguous(),
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] must be a contiguous FP16 RPU tensor");
        TORCH_CHECK(tensor.dim() == dim,
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] must be ", dim, "D, got ", tensor.dim(), "D");
    };
    // Exact 2B W8 main/GDN profile; b/a scalar projections remain FP16.
    // Inspect only after list cardinalities have been validated above.
    const bool reduced_w8_2b = reduced && !reduced_legacy27 &&
        hidden_size == 2048 && gate_list[0].defined() &&
        gate_list[0].scalar_type() == at::kChar;
    const auto check_linear = [reduced, reduced_legacy27, reduced_w8_2b](const at::Tensor& weight,
                                 const at::Tensor& scale,
                                 const char* name, int64_t layer) {
        TORCH_CHECK(weight.defined() && weight.numel() > 0 && weight.dim() == 2,
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] weight must be a non-empty 2D tensor");
        const std::string role(name);
        const bool legacy_main = reduced_legacy27 && role.rfind("gdn_", 0) != 0;
        const bool w8_2b_projection = reduced_w8_2b &&
            role != "gdn_b_bg" && role != "gdn_a_bg";
        TORCH_CHECK(!reduced || weight.scalar_type() ==
                        (legacy_main || w8_2b_projection ? at::kChar : at::kHalf),
                    "Qwen3.5 reduced projection dtype must match exact FP16, "
                    "2B W8 with FP16 GDN b/a, or legacy W8-main FP16-GDN policy");
        TORCH_CHECK(weight.device().type() == at::kPrivateUse1
                    && weight.is_contiguous()
                    && (weight.scalar_type() == at::kHalf
                        || weight.scalar_type() == at::kChar
                        || weight.scalar_type() == at::kByte),
                    "qwen3_5 set_weights: ", name, "[", layer,
                    "] weight must be contiguous FP16, int8, or packed uint8 on RPU");
        if (weight.scalar_type() == at::kChar) {
            TORCH_CHECK(scale.defined() && scale.dim() == 1
                        && scale.numel() == weight.size(0)
                        && scale.scalar_type() == at::kHalf
                        && scale.device().type() == at::kPrivateUse1
                        && scale.is_contiguous(),
                        "qwen3_5 set_weights: W8A16 ", name, "[", layer,
                        "] requires contiguous FP16 RPU scale [N=",
                        weight.size(0), "]");
        } else if (weight.scalar_type() == at::kByte) {
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
        if (reduced) {
            const auto shape = [i](const at::Tensor& tensor,
                                    at::IntArrayRef expected, const char* name) {
                TORCH_CHECK(tensor.defined() && tensor.sizes() == expected,
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
    TORCH_CHECK(cos.sizes() == sin.sizes(),
                "qwen3_5 set_weights: cos/sin shapes must match");
    TORCH_CHECK(!reduced || (cos.size(1) == 32 && final_norm_w.numel() == hidden_size),
                "Qwen3.5 reduced RoPE/final norm geometry mismatch");

    has_qk_norm_ = q_norm_list.size() > 0;
    use_silu_    = use_silu;
    eps_         = eps;

    layer_is_full_.assign(layer_is_full.begin(), layer_is_full.end());
    has_gdn_ = std::any_of(
        layer_is_full_.begin(), layer_is_full_.end(),
        [](uint8_t is_full) { return !is_full; });
    action_mode_ = false;
    action_rtc_active_ = false;
    action_rtc_prefix_len_ = 0;
    action_rtc_prefix_ref_ = at::Tensor();
    action_rtc_prefix_live_base_ = 0;
    adaptive_mod_ref_ = at::Tensor();
    action_prefix_lens_.clear();
    layer_weights_.assign(N, LayerWeights{});
    for (int64_t i = 0; i < N; ++i) {
        auto& lw = layer_weights_[i];
        // Every layer (full + GDN) has input_layernorm (unified channel), post_norm, MLP.
        lw.input_norm_w = input_norm_list[i];
        lw.post_norm_w = post_norm_list[i];
        lw.gate_w = gate_list[i]; lw.up_w = up_list[i]; lw.down_w = down_list[i];
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
        auto opt = final_norm_w.options();   // fp16 on RPU
        gdn_zero_ = at::zeros({gdn_dk_ * gdn_dv_}, opt);
        // prefill-chunk constant masks (C=64 fixed). tril keeps the diagonal (decay
        // self-score exp(0)=1); strict is the strict-lower (i>j). Both match rhino's
        // host-prepared constant masks (the -1 is a separate binary_scalar, not folded
        // here). Built on CPU then moved to RPU (host-side fill; values exact).
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
    at::TensorList gdn_a_bg_scale) {
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
                   gdn_b_bg_scale, gdn_a_bg_scale);
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
