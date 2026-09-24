// rpu_qwen3_5_moe_model.cpp — independent Qwen3.5 sparse-MoE fused model.
// See rpu_qwen3_5_moe_model.h for scope. Full-attention emission mirrors
// CausalDecoderModel::build_layer_subgraph (rpu_qwen3_model.cpp) for the
// non-M-RoPE, QK-norm path; GDN layers dispatch to build_gdn(decode) by chunk.len.
//
// ── FILE MAP (read order: main flow → mixers → buffers/config → setup → C-API) ──
//   MAIN FLOW
//     forward()               — Python entry's workhorse; drives run_all_layers
//     build_layer_subgraph()  — assembles ONE layer: input DMA → input_layernorm →
//                               token mixer → post_norm + sparse MoE + output
//   TOKEN MIXERS (read normed "input_norm"; end with all_reduce that adds h_in residual)
//     build_full_attention()  — QK-norm + (partial M-)RoPE + SDPA + gated o_proj
//     build_gdn()             — unified GDN mixer; if(decode) splits recurrent (seq=1)
//                               vs chunked delta-rule (prefill, L>1); shared setup/proj/tail
//     emit_mlp_and_output()   — post-norm + router/shared/routed/merge helpers
//   SPM LAYOUT       declare_buffers() — attention/GDN plus sparse-MoE temporaries
//   GRAPH-PLAN HOOKS static_config / dynamic_config / subclass_chunk_size_valid
//   SETUP            set_weights() (store swizzled weights)
//   C-API (bottom)   rpu_qwen3_5_moe_* free funcs — Python handle ↔ object bridges
#include "rpu_qwen3_5_moe_model.h"

#include "model_handle_registry.h"
#include "rpu_ops.h"
#include "rpu_spm_buffers.h"
#include "rpu_spm_residency.h"
#include "rpu_helpers.h"
#include "rpu_profile.h"         // rpu_ddr_flush

#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>     // std::sqrt (GDN qk_scale)
#include <limits>
#include <stdexcept>
#include <tuple>

using namespace at;
using namespace ::rhino_lkn;

#define NUM_CORES 8
#define DWIDTH 2

namespace {

constexpr int kNumCores = NUM_CORES;
// Only the post-mixer MLP is sliced; Attention/GDN keep the outer chunk.
constexpr int64_t kMoeChunkSize = 128;

template <class Fn>
void for_each_moe_slice(const v3::ChunkInfo& chunk, Fn&& emit) {
    for (int64_t row = 0; row < chunk.len; row += kMoeChunkSize) {
        // Token starts within this forward distinguish slices across outer
        // chunks. Linear routes also distinguish this domain from the mixer.
        emit(row, std::min(kMoeChunkSize, chunk.len - row), chunk.offset + row);
    }
}

constexpr uint32_t QWEN35_MOE_KV_CAPABILITIES =
    KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16;
constexpr int64_t QWEN35_MOE_KV_REASON_DDR_REQUIRED = 1;
constexpr int64_t QWEN35_MOE_ATTN_DDR_REQUIRED = 1;

constexpr int64_t QWEN35_MOE_GDN_MASK_RESIDENCY_SITE = 4606644302213524820LL;
constexpr int64_t QWEN35_MOE_ATTENTION_SITE = 4947544041507852223LL;
constexpr int64_t QWEN35_MOE_LINEAR_SITE = 2506358524225803894LL;
constexpr int64_t QWEN35_MOE_GROUPED_FP8_SITE = 1444084512072972229LL;
constexpr int64_t QWEN35_MOE_FULL_PREPARE_ALL_REDUCE_SITE =
    548848499648069557LL;
constexpr int64_t QWEN35_MOE_FULL_ALL_REDUCE_SITE =
    4164414102147778666LL;
constexpr int64_t QWEN35_MOE_GDN_ALL_REDUCE_SITE =
    386092204056374248LL;
constexpr int64_t QWEN35_MOE_ALL_REDUCE_SITE = 2673422434034478182LL;
constexpr int64_t QWEN35_MOE_PARTIAL_ROPE_1D_SITE = 8158010487592746687LL;
constexpr int64_t QWEN35_MOE_PARTIAL_MROPE_SITE = 4428095955208568309LL;
constexpr int64_t QWEN35_MOE_KV_INSERT_SITE = 1342453738351445632LL;
constexpr int64_t QWEN35_MOE_Q_NORM_PRELOAD_SITE = 8709779403018824296LL;
constexpr int64_t QWEN35_MOE_K_NORM_PRELOAD_SITE = 6881982945286419444LL;
constexpr int64_t QWEN35_MOE_GDN_STATE_DECODE_LOAD_SITE =
    3595760404038313015LL;
constexpr int64_t QWEN35_MOE_GDN_CONV_DECODE_LOAD_SITE =
    5303998550722101434LL;
constexpr int64_t QWEN35_MOE_GDN_STATE_DECODE_STORE_SITE =
    8813227130097969675LL;
constexpr int64_t QWEN35_MOE_GDN_CONV_DECODE_STORE_SITE =
    3832262806628594865LL;
constexpr int64_t QWEN35_MOE_GDN_STATE_PREFILL_LOAD_SITE =
    7276468846054720425LL;
constexpr int64_t QWEN35_MOE_GDN_CONV_PREFILL_LOAD_SITE =
    2393559597590066234LL;
constexpr int64_t QWEN35_MOE_GDN_STATE_PREFILL_STORE_SITE =
    1930073973136784520LL;
constexpr int64_t QWEN35_MOE_GDN_CONV_PREFILL_STORE_SITE =
    5987041279149103130LL;
constexpr int64_t QWEN35_MOE_SHARED_SILU_SITE = 5790502580595072857LL;
constexpr int64_t QWEN35_MOE_SHARED_SIGMOID_SITE = 4065797117506920453LL;
constexpr int64_t QWEN35_MOE_ROUTED_SILU_SITE = 4050586624170153493LL;
constexpr int64_t QWEN35_MOE_EXACT_PROFILE_SITE = 2504589813057384578LL;
constexpr int64_t QWEN35_MOE_ROUTER_SCHEDULE_SITE = 586450017484669802LL;
constexpr int64_t QWEN35_MOE_ROUTER_ALL_GATHER_SITE = 845855607644119815LL;
constexpr int64_t QWEN35_MOE_CORE_TOPOLOGY_SITE = 8975316895652603198LL;
constexpr int64_t QWEN35_MOE_ROUTER_BRIDGE_STORE_SITE =
    3225143782393096355LL;
constexpr int64_t QWEN35_MOE_ROUTER_BRIDGE_LOAD_SITE =
    2526999425649687054LL;
constexpr int64_t QWEN35_MOE_GDN_PREPARE_ALL_REDUCE_SITE =
    4400682760413035627LL;

enum class Qwen35MoeAllReduceRoute : int64_t {
    PREPARE_RING_INPUT = 3,
};

enum class Qwen35MoeRopeRoute : int64_t {
    PARTIAL_ROPE_1D = 1,
    PARTIAL_MROPE = 2,
};

enum class Qwen35MoeMutableDmaRoute : int64_t {
    DDR_BROADCAST_TO_SPM = 1,
    DDR_SCATTER_TO_SPM = 2,
    SPM_CORE0_TO_DDR = 3,
    SPM_SCATTER_TO_DDR = 4,
};

enum class Qwen35MoeActivationRoute : int64_t {
    SILU_MUL = 1,
    SIGMOID = 2,
};

constexpr int64_t qwen35_moe_dma_route(Qwen35MoeMutableDmaRoute route) {
    return static_cast<int64_t>(route);
}

// GDN uses phases 1..20 and attention 22..27. MoE follows either mixer.
// MoE phases follow individual producers/consumers; they do not split execution.
constexpr int kMoePhaseRouterLinear = 30;
constexpr int kMoePhaseRouterGather = 31;
constexpr int kMoePhaseRouterSoftmax = 32;
constexpr int kMoePhaseTopK = 33;
constexpr int kMoePhaseSort = 34;
constexpr int kMoePhaseBincount = 35;
constexpr int kMoePhaseProbNorm = 36;
constexpr int kMoePhaseSharedGate = 37;
constexpr int kMoePhaseSharedUp = 38;
constexpr int kMoePhaseSharedAct = 39;
constexpr int kMoePhaseSharedDown = 40;
constexpr int kMoePhaseSharedAlpha = 41;
constexpr int kMoePhaseSharedAlphaSlice = 42;
// Phase 43 applies sigmoid in place; alpha remains live through SharedWeight.
constexpr int kMoePhaseSharedWeight = 44;
constexpr int kMoePhaseGatherIndex = 45;
constexpr int kMoePhaseGather = 46;
constexpr int kMoePhaseRoutedGate = 47;
constexpr int kMoePhaseRoutedUp = 48;
constexpr int kMoePhaseRoutedAct = 49;
constexpr int kMoePhaseRoutedDown = 50;
constexpr int kMoePhaseWeight = 51;
constexpr int kMoePhaseScatter = 52;
constexpr int kMoePhaseReduce = 53;

int64_t aligned_bytes(int64_t elements, int64_t element_bytes = 2) {
    TORCH_CHECK(elements >= 0 &&
                    elements <= std::numeric_limits<int64_t>::max() /
                                    element_bytes,
                "qwen3_5_moe: buffer size overflow");
    return Align(elements * element_bytes, 256);
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

// QWEN35_MOE_TEXT_FIXED_KERNEL_BASIS: every launcher is bound to one of the
// exact TP4/TP6/TP8 owners below. Any alternate implementation must first
// become a typed COMPLETE-manifest route.

void Qwen3_5MoeStaticConfig::validate(int64_t hidden_size) const {
    TORCH_CHECK(hidden_size == 2048,
                "qwen3_5_moe: exact 35B-A3B profile requires hidden_size=2048, got ",
                hidden_size);
    TORCH_CHECK(tensor_parallel == 4 || tensor_parallel == 6 ||
                    tensor_parallel == kNumCores,
                "qwen3_5_moe: exact profile requires tp=4, tp=6, or tp=8, got ",
                tensor_parallel);
    TORCH_CHECK(num_experts == 256 && top_k == 8 &&
                    routed_intermediate == 512 && shared_intermediate == 512,
                "qwen3_5_moe: exact 35B-A3B profile requires "
                "E256/top8/routed512/shared512");
    TORCH_CHECK(normalize_topk,
                "qwen3_5_moe: first graph requires normalized top-k routing");
}

Qwen3_5MoeModel::Qwen3_5MoeModel() = default;
Qwen3_5MoeModel::~Qwen3_5MoeModel() = default;

void Qwen3_5MoeModel::set_execution_cores(int64_t cores) {
    TORCH_CHECK(
        cores == 4 || cores == 6 || cores == 8,
        "Qwen3.5-MoE execution cores must be 4, 6, or 8");
    TORCH_CHECK(
        layer_weights_.empty() && moe_layer_weights_.empty() &&
            !expert_ids_.defined() && !token_ids_.defined() &&
            !router_bridge_.defined(),
        "Qwen3.5-MoE execution cores must be bound before weights");
    set_execution_core_count(static_cast<int>(cores));
}

std::vector<int64_t> Qwen3_5MoeModel::execution_topology() const {
    TORCH_CHECK(
        num_layers() > 0,
        "Qwen3.5-MoE topology requires installed model weights");
    return {num_cores(), attn_tp(), mlp_tp(), lm_head_tp(), NUM_CORES};
}

std::vector<int64_t> Qwen3_5MoeModel::execution_topology_v2() const {
    TORCH_CHECK(
        num_layers() > 0,
        "Qwen3.5-MoE topology-v2 requires installed model weights");
    // Stable schema-v2 order:
    // version, root, attention, GDN, router, routing, routed, shared, head,
    // physical KV stripes.
    return {2, num_cores(), attn_tp(), gdn_tp(), router_tp(), routing_tp(),
            routed_expert_tp(), shared_expert_tp(), lm_head_tp(), NUM_CORES};
}

DecoderExecutionTopology Qwen3_5MoeModel::resolve_model_execution_topology(
        int64_t nq, int64_t nkv, int64_t hd, int64_t h,
        int64_t intermediate) const {
    if (num_cores() == 8) {
        return FusedModelBase::resolve_model_execution_topology(
            nq, nkv, hd, h, intermediate);
    }
    if ((num_cores() != 4 && num_cores() != 6) ||
        nq != 16 || nkv != 4 || hd != 256 ||
        h != 2048 || intermediate != 512) {
        throw std::invalid_argument(
            "no admitted reduced-core Qwen3.5-35B-A3B MoE template");
    }
    return {num_cores(), 4, num_cores()};
}

std::vector<int64_t> Qwen3_5MoeModel::cold_topology_arguments() const {
    if (num_cores() != 6) {
        // Preserve the existing TP4 identity and the TP8 default.
        return {1, num_cores(), attn_tp(), gdn_tp(), mlp_tp(),
                lm_head_tp(), NUM_CORES};
    }
    // Internal cold identity construction is also safe before public topology
    // attestation becomes legal; do not couple it to the installed-weight gate.
    return {2, num_cores(), attn_tp(), gdn_tp(), router_tp(), routing_tp(),
            routed_expert_tp(), shared_expert_tp(), lm_head_tp(), NUM_CORES};
}

std::vector<int64_t> Qwen3_5MoeModel::exact_profile_arguments() const {
    if (num_cores() != 6) {
        return {num_layers(), hidden_size(), num_q_heads(), num_kv_heads(),
                head_dim(), moe_.num_experts, moe_.top_k,
                moe_.routed_intermediate, moe_.shared_intermediate,
                moe_.tensor_parallel, gdn_nvh_, gdn_dk_, gdn_dv_,
                gdn_conv_dim_, gdn_kc_, has_qk_norm_ ? 1 : 0};
    }
    return {2, num_layers(), hidden_size(), num_q_heads(), num_kv_heads(),
            head_dim(), moe_.num_experts, moe_.top_k,
            moe_.routed_intermediate, physical_routed_intermediate(),
            moe_.shared_intermediate, physical_shared_intermediate(),
            moe_.tensor_parallel, gdn_nvh_, gdn_dk_, gdn_dv_,
            gdn_conv_dim_, gdn_kc_, has_qk_norm_ ? 1 : 0,
            router_bridge_.defined() ? 1 : 0,
            router_bridge_.defined() ? router_bridge_.size(0) : 0};
}

int64_t Qwen3_5MoeModel::text_ring_route(
        int64_t rows, int64_t cols) const {
    return fmb_ring_all_reduce_route_selector(rows, cols, num_cores());
}

void Qwen3_5MoeModel::set_chunk_size_cap(int64_t cap) {
    TORCH_CHECK(cap == 0 || cap >= 64,
                "Qwen3.5 text chunk cap must be 0 or at least 64, got ", cap);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 text chunk cap must be set before the first forward");
    chunk_size_cap_ = cap;
    set_chunk_size_override(0);
    invalidate_model_state();
}

void Qwen3_5MoeModel::set_prefill_chunk_size(int64_t chunk_size) {
    TORCH_CHECK(chunk_size == 0
                    || (chunk_size >= 64 && chunk_size % 64 == 0),
                "Qwen3.5 text chunk size must be 0 (auto) or a positive "
                "multiple of 64, got ", chunk_size);
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 MoE text chunk size must be set before the first "
                "forward");
    set_chunk_size_override(chunk_size);
    invalidate_model_state();
}

void Qwen3_5MoeModel::set_linear_acc32(bool enabled) {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 linear accumulation mode must be set before the first forward");
    linear_acc32_ = enabled;
    invalidate_model_state();
}

void Qwen3_5MoeModel::set_fast_replay(bool enabled) {
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "Qwen3.5 fast replay must be set before the first forward");
    fast_replay_enabled_ = enabled;
    invalidate_model_state();
}

void Qwen3_5MoeModel::set_retained_prefill_graph(bool enabled) {
    TORCH_CHECK(
        get_last_resolved_chunk_size() == 0,
        "Qwen3.5 MoE prefill Graph lifecycle must be fixed before the first "
        "forward");
    retained_prefill_graph_ = enabled;
    invalidate_model_state();
}

int64_t Qwen3_5MoeModel::resolve_prefill_chunk_size(int64_t execution_len) {
    TORCH_CHECK(execution_len >= 64 && execution_len % 64 == 0,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 MoE prefill execution "
                "length must be a positive multiple of 64, got ",
                execution_len);
    return resolve_chunk_size_for_shape(
        execution_len, /*position=*/0, std::nullopt, /*is_causal=*/true);
}

std::vector<int64_t> Qwen3_5MoeModel::resolve_prefill_stage_domain(
    int64_t execution_len, int64_t logical_len,
    int64_t planning_chunk_size_override) {
    TORCH_CHECK(planning_chunk_size_override <= 0 ||
                    planning_chunk_size_override % 64 == 0,
                "RPU_PLANNER_REJECT:EXACT_MISMATCH: Qwen3.5 MoE prefill "
                "planning chunk must be a multiple of 64");
    TORCH_CHECK(execution_len >= 64 && execution_len % 64 == 0,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 MoE prefill execution "
                "length must be a positive multiple of 64, got ", execution_len);
    TORCH_CHECK(logical_len > 1 && logical_len <= execution_len,
                "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 MoE prefill logical "
                "length must be in [2, execution length], got ", logical_len,
                " for execution length ", execution_len);
    return encode_fmb_prefill_stage_domain(
        resolve_prefill_stage_domain_for_shape(
            execution_len, /*position=*/0, std::nullopt,
            /*is_causal=*/true, /*requested_chunk_size=*/0, logical_len,
            planning_chunk_size_override));
}

std::vector<int64_t> Qwen3_5MoeModel::resolve_decode_stage_descriptor() {
    auto candidates = resolve_prefill_stage_domain_for_shape(
        /*seq_len=*/1, /*position=*/0, std::nullopt,
        /*is_causal=*/true);
    TORCH_CHECK(
        candidates.size() == 1,
        "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 MoE fixed decode must "
        "resolve exactly one native stage candidate, got ", candidates.size());
    return encode_fmb_prefill_stage_candidate(candidates.front());
}

// ── forward: stash GDN state for the seam, then all-layers-once ──────────────
at::Tensor Qwen3_5MoeModel::forward(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    std::vector<at::Tensor>& gdn_states,
    std::vector<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position, bool is_causal,
    int64_t planned_chunk_size,
    at::IntArrayRef planned_stage_descriptor) {
    TORCH_CHECK(
        planned_chunk_size == 0 && !planned_stage_descriptor.empty(),
        "RPU_PLANNER_REJECT:CAPABILITY: Qwen3.5 MoE text forward requires "
        "one complete planner stage descriptor");
    return forward_impl(hidden_states, k_caches, v_caches, gdn_states,
                        conv_states, attention_mask, position, is_causal,
                        planned_chunk_size, planned_stage_descriptor);
}

at::Tensor Qwen3_5MoeModel::forward_impl(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    std::vector<at::Tensor>& gdn_states,
    std::vector<at::Tensor>& conv_states,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position, bool is_causal,
    int64_t planned_chunk_size,
    at::IntArrayRef planned_stage_descriptor) {
    TORCH_CHECK(hidden_states.defined() && hidden_states.dim() == 3 &&
                    hidden_states.size(0) == 1 &&
                    hidden_states.size(2) == hidden_size() &&
                    hidden_states.scalar_type() == at::kHalf &&
                    hidden_states.device().type() == at::kPrivateUse1 &&
                    hidden_states.is_contiguous() && is_causal,
                "Qwen3.5 MoE execution requires causal B1 contiguous FP16 "
                "hidden states on RPU");
    const int64_t seq_len = hidden_states.size(1);
    if (num_cores() != 8) {
        TORCH_CHECK(
            gdn_states.size() == static_cast<size_t>(num_layers()) &&
                conv_states.size() == static_cast<size_t>(num_layers()),
            "Qwen3.5-MoE reduced execution requires one GDN cache slot per layer");
    }
    TORCH_CHECK(seq_len == 1 || seq_len % 64 == 0,
                "Qwen3.5 text expects decode length 1 or a prefill execution "
                "length divisible by 64; use the Python adapter for ragged prefill");
    gdn_states_  = &gdn_states;
    conv_states_ = &conv_states;
    fast_replay_active_ = fast_replay_enabled_ && seq_len > 1;

    // Fast replay skips build_gdn(), so refresh every mutable cache base before
    // entering run_all_layers. Decode still walks the layer loop, but shares the
    // same safe prologue.
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
            TORCH_CHECK(
                valid_state(*st) && valid_state(*cv) &&
                    st->sizes() == at::IntArrayRef(
                        {gdn_nvh_, gdn_dk_, gdn_dv_}) &&
                    cv->sizes() == at::IntArrayRef(
                        {gdn_tp(), gdn_kc_, gdn_conv_dim_ / gdn_tp()}),
                "Qwen3.5-MoE recurrent/conv cache layout does not match the "
                "cold GDN core count");
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

    gdn_states_  = nullptr;
    conv_states_ = nullptr;

    return out;
}

void Qwen3_5MoeModel::build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) {
    const int64_t h = hidden_size();
    if (ctx().has_complete_physical_manifest() && layer_idx == 0) {
        if (num_cores() != 8 && chunk.idx == 0) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE,
                QWEN35_MOE_CORE_TOPOLOGY_SITE,
                /*selector=*/num_cores() == 6 ? 2 : 1,
                /*resolved_flags=*/0,
                cold_topology_arguments());
        }
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_MOE_EXACT_PROFILE_SITE,
            /*selector=*/num_cores() == 6 ? 2 : 1,
            /*resolved_flags=*/0, exact_profile_arguments(), chunk.idx);
    }
    const uint32_t input_residual = addr(0, "residual1");

    // A layer = input DMA → input_layernorm → token mixer → post_norm + MLP + output.
    // ① input DMA: the first layer of the cross-layer group brings h_in into residual1.
    if (!ctx().input_in_spm) {
        emit_layer_input_dma(layer_idx, chunk);   // h_in → residual1
    }
    // ② input_layernorm: residual1 keeps h_in while input_norm feeds the mixer.
    rpu_launch_rmsnorm_spm_kernel(
        input_residual, addr(0, "input_norm"),
        layer_addr(layer_idx, 0, "norm_w"), chunk.len, h, eps_,
        RpuRmsNormSpmRoute::BASE, num_cores());
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

FmbLinearRouteSelector Qwen3_5MoeModel::linear_route_selector(
    const at::Tensor& weight, int64_t rows) const {
    if (rows == 1 && !linear_acc32_ && weight.scalar_type() == at::kHalf)
        return FmbLinearRouteSelector::GEMV;
    return FmbLinearRouteSelector::AUTO_TILE;
}

void Qwen3_5MoeModel::launch_linear(
    int64_t chunk_idx,
    uint32_t input, const at::Tensor& weight, uint32_t output,
    int64_t m, int64_t n, int64_t k, int partition, int num_cores,
    uint32_t bias_spm_addr, const at::Tensor& scale, bool mlp_slice) {
    const FmbLinearRouteSelector selector = linear_route_selector(weight, m);
    const int64_t invocation = linear_invocation(chunk_idx, selector, mlp_slice);
    consume_manifest_route(
        FmbRouteFamily::LINEAR, QWEN35_MOE_LINEAR_SITE,
        static_cast<int64_t>(selector), invocation);
    rpu_launch_linear_spm_to_spm_acc16_kernel(
        input, weight, output, m, n, k, partition, num_cores,
        bias_spm_addr, scale,
        /*nvfp4_tensor_scale_spm_addr=*/0, /*nvfp4_layer_id=*/0,
        /*force_acc32=*/linear_acc32_,
        /*prefer_gemv=*/selector == FmbLinearRouteSelector::GEMV);
}

void Qwen3_5MoeModel::launch_grouped_linear(
    int64_t chunk_idx,
    uint32_t input, const at::Tensor& weight,
    uint32_t m_sizes, const at::Tensor& scale,
    uint32_t output, int64_t n, int64_t k,
    int64_t num_experts, bool is_col_parallel,
    int num_cores, int64_t weight_mode) {
    const int64_t invocation = chunk_idx * 2 + (is_col_parallel ? 0 : 1);
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::LINEAR, QWEN35_MOE_GROUPED_FP8_SITE,
            weight_mode, /*resolved_flags=*/0,
            {n, k, num_experts, is_col_parallel ? 1 : 0, num_cores,
             linear_acc32_ ? 1 : 0},
            invocation);
    }
    rpu_launch_grouped_parallel_linear_spm_kernel(
        input, weight, m_sizes, optional_scale(scale), output, /*bias=*/0,
        n, k, num_experts, is_col_parallel, num_cores, weight_mode, linear_acc32_);
}

void Qwen3_5MoeModel::consume_manifest_route(
    FmbRouteFamily family, int64_t site_id,
    int64_t selector, int64_t invocation) {
    if (!ctx().has_complete_physical_manifest()) return;
    ctx().consume_physical_route(
        family, site_id, selector, /*resolved_flags=*/0,
        /*resolved_arguments=*/{}, invocation);
}

void Qwen3_5MoeModel::build_full_attention(int layer_idx, const ChunkInfo& chunk) {
    const auto& lw = layer_weights_[layer_idx];
    int64_t seq_len = chunk.len;
    int64_t cos_sin_start = ctx().position + chunk.offset;
    int64_t h = hidden_size();
    int64_t nq = num_q_heads();
    int64_t nkv = num_kv_heads();
    int64_t hd = head_dim();
    // The adapter replicates logical GQA KV heads to the selected attention
    // width. One complete effective KV head lives on each active core;
    // attention-phase kernels run on that resolved tensor-parallel width and
    // virtual_num_cores=NUM_CORES keeps the KV-cache swizzle layout.
    // MoE / residual ownership follows the selected model core domain.
    int64_t tp = attn_tp();
    const int auxiliary_cores =
        num_cores() == 8 ? NUM_CORES : static_cast<int>(tp);

    // input DMA + input_layernorm are done by build_layer_subgraph; the normed input is
    // in "input_norm", h_in stays in residual1 (for the all_reduce residual).
    // Phase 2: QKV linear (col-partition over tp cores)
    launch_linear(
        chunk.idx,
        addr(0, "input_norm"), lw.q_w, addr(0, "q"), seq_len, nq * hd, h, 1, tp,
        /*bias=*/0, lw.q_scale);
    launch_linear(
        chunk.idx,
        addr(0, "input_norm"), lw.k_w, addr(0, "k"), seq_len, nkv * hd, h, 1, tp,
        /*bias=*/0, lw.k_scale);
    launch_linear(
        chunk.idx,
        addr(0, "input_norm"), lw.v_w, addr(0, "v"), seq_len, nkv * hd, h, 1, tp,
        /*bias=*/0, lw.v_scale);
    // Gated-attention gate projection. Compute here while "input_norm" still
    // holds the Phase-1 input RMSNorm output (Phase 3 k_norm overwrites it).
    // Applied as sigmoid(gate) ⊙ attn_out after SDPA (Qwen3.5 attn_output_gate).
    launch_linear(
        chunk.idx,
        addr(0, "input_norm"), lw.attn_gate_w, addr(0, "attn_gate"),
        seq_len, nq * hd, h, 1, tp, /*bias=*/0, lw.attn_gate_scale);

    // Phase 3: QK RMSNorm + RoPE. M-RoPE (interleaved) + partial rotary when
    // has_mrope_ (Qwen3.5); the 1D RoPE branch stays for non-mrope models.
    int64_t local_q_heads = nq / tp;
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
                FmbRouteFamily::ROPE, QWEN35_MOE_PARTIAL_ROPE_1D_SITE,
                static_cast<int64_t>(Qwen35MoeRopeRoute::PARTIAL_ROPE_1D),
                chunk.idx);
            rpu_launch_partial_rope_1d_spm_kernel(
                buf, buf, cos_ptr, sin_ptr, cos_sin_start, seq_len, heads, hd, rotary_dim_, tp);
        } else {
            consume_manifest_route(
                FmbRouteFamily::ROPE, QWEN35_MOE_PARTIAL_MROPE_SITE,
                static_cast<int64_t>(Qwen35MoeRopeRoute::PARTIAL_MROPE),
                chunk.idx);
            rpu_launch_partial_mrope_spm_kernel(
                buf, buf, cos_ptr, sin_ptr, cos_sin_start, seq_len, heads, hd, rotary_dim_, tp);
        }
    };

    TORCH_INTERNAL_ASSERT(has_qk_norm_ && has_mrope_);
    const int64_t local_kv_heads = nkv / tp;
    TORCH_INTERNAL_ASSERT(local_kv_heads == 1);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "q"), addr(0, "q"),
        layer_addr(layer_idx, 0, "q_norm_w"),
        seq_len * local_q_heads, hd, eps_,
        RpuRmsNormSpmRoute::BASE, auxiliary_cores);
    emit_partial_rope(addr(0, "q"), local_q_heads);
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "k"), addr(0, "k"),
        layer_addr(layer_idx, 0, "k_norm_w"),
        seq_len * local_kv_heads, hd, eps_,
        RpuRmsNormSpmRoute::BASE, auxiliary_cores);
    emit_partial_rope(addr(0, "k"), local_kv_heads);

    // Phase 4: KV-cache insert + SDPA + O_proj (P3: offsets via addr_offset)
    auto& k_cache = (*ctx().k_caches)[layer_idx];
    auto& v_cache = (*ctx().v_caches)[layer_idx];
    const int64_t kv_insert_pos = ctx().position + chunk.offset;
    const KvInsertSegmentPlan kv_plan = [&] {
        TORCH_CHECK(
            ctx().has_complete_physical_manifest(),
            "Qwen3.5 MoE KV insert requires a COMPLETE physical descriptor");
        const auto& route = ctx().find_physical_route(
            FmbRouteFamily::KV_INSERT, QWEN35_MOE_KV_INSERT_SITE,
            chunk.idx);
        const KvInsertSegmentPlan template_plan = restore_kvinsert_plan(
            QWEN35_MOE_KV_INSERT_SITE, route.arguments, tp, nkv, hd);
        const bool dynamic_position =
            (route.flags & KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION) != 0;
        TORCH_CHECK(
            !dynamic_position ||
                (seq_len == 1 && template_plan.segment(0).position == 0),
            "Qwen3.5 MoE dynamic KV position is valid only for the canonical "
            "single-token decode template");
        KvInsertSegmentPlan resolved = dynamic_position
            ? rpu_rebase_kvinsert_segment_plan_position(
                  template_plan, kv_insert_pos, tp, nkv, hd)
            : template_plan;
        TORCH_CHECK(
            resolved.logical_rows() == seq_len &&
                resolved.physical_rows() == seq_len &&
                resolved.segment(0).position == kv_insert_pos,
            "Qwen3.5 MoE KV descriptor geometry drift at layer ", layer_idx,
            ", chunk ", chunk.idx);
        ctx().consume_physical_route(
            FmbRouteFamily::KV_INSERT, QWEN35_MOE_KV_INSERT_SITE,
            static_cast<int64_t>(template_plan.route()), route.flags,
            route.arguments, chunk.idx);
        return resolved;
    }();
    rpu_launch_insert_kvcache_spm_unified_with_plan(
        k_cache, v_cache, addr_offset("k").value, addr_offset("v").value,
        nkv, hd, tp, /*k_cache_batch_offset_elems=*/0,
        /*v_cache_batch_offset_elems=*/0, /*spm_rows=*/0, kv_plan);

    int64_t kv_seq_len = chunk.kv_seq_len;
    bool sdpa_causal = ctx().is_causal && (seq_len > 1);
    if (ctx().has_complete_physical_manifest()) {
        const auto& attention_route = ctx().find_physical_route(
            FmbRouteFamily::ATTENTION, QWEN35_MOE_ATTENTION_SITE,
            chunk.idx);
        const std::vector<int64_t> arguments = seq_len == 1
            ? std::vector<int64_t>{1}
            : std::vector<int64_t>{
                  /*batch=*/1, seq_len, kv_seq_len, nq, nkv, hd, tp,
                  sdpa_causal ? 1 : 0};
        TORCH_CHECK(
            attention_route.flags == QWEN35_MOE_ATTN_DDR_REQUIRED,
            "Qwen3.5 MoE attention descriptor lacks its exact DDR_REQUIRED "
            "reason");
        ctx().consume_physical_route(
            FmbRouteFamily::ATTENTION, QWEN35_MOE_ATTENTION_SITE,
            static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
            attention_route.flags, arguments, chunk.idx);
    }
    // SDPA on tp cores; virtual_num_cores=NUM_CORES governs the KV-cache 7D swizzle.
    rpu_launch_sdpa_spm_dispatch(
        sdpa_kernel_, k_cache, v_cache, sdpa_causal ? 1 : 0, c10::nullopt,
        addr_offset("q").value, addr_offset("output").value,
        addr_offset("sdpa_tmp").value, 0,
        seq_len, nq, nkv, hd, kv_seq_len, tp, NUM_CORES);

    // Gated-attention output gate (Qwen3.5 attn_output_gate=true):
    //   output = sigmoid(gate) ⊙ output, before o_proj.
    {
        int64_t gate_elems = seq_len * (nq / tp) * hd;
        rpu_launch_eltwise_unary_spm_kernel(
            addr(0, "attn_gate"), addr(0, "attn_gate"),
            gate_elems, ValuOpType::SIGMOID, /*is_gelu=*/false, tp);
        rpu_launch_eltwise_binary_spm_kernel(
            addr(0, "attn_gate"), addr(0, "output"), addr(0, "output"),
            gate_elems, ValuOpType::MUL, c10::Half(1.0f), auxiliary_cores);
    }
    // O-proj row-partition over tp cores; reduce back to the model owner below.
    consume_manifest_route(
        FmbRouteFamily::ALL_REDUCE,
        QWEN35_MOE_FULL_PREPARE_ALL_REDUCE_SITE,
        static_cast<int64_t>(Qwen35MoeAllReduceRoute::PREPARE_RING_INPUT),
        chunk.idx);
    rpu_prepare_ring_all_reduce_input(
        addr(0, "oproj"), seq_len, h, tp, num_cores());
    launch_linear(
        chunk.idx,
        addr(0, "output"), lw.o_w, addr(0, "oproj"), seq_len, h, nq * hd, 0, tp,
        /*bias=*/0, lw.o_scale);

    // Phase 5: attention reduce + residual (tp inputs → owner outputs)
    consume_manifest_route(
        FmbRouteFamily::ALL_REDUCE, QWEN35_MOE_FULL_ALL_REDUCE_SITE,
        text_ring_route(seq_len, h), chunk.idx);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "oproj"), addr(0, "residual1"),
        addr(0, "residual2"), seq_len, h,
        tp, num_cores());

    // The all_reduce left the residual stream in residual2 (== input_norm); the shared
    // post_norm + MLP + final_norm + output tail runs back in build_layer_subgraph
    // (emit_mlp_and_output), same as the GDN path. build_full_attention is mixer-only.
}

// ── build_gdn: the unified Gated-DeltaNet token mixer. Shared setup/proj/tail + an
//    if(decode) split for the divergent conv + delta-rule core (decode = recurrent
//    single step; prefill = chunked delta-rule). Dispatched by build_layer_subgraph.
void Qwen3_5MoeModel::build_gdn(int layer_idx, const ChunkInfo& chunk, bool decode) {
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
    if (decode) {
        rpu_launch_ddr_scatter_spm_dma(lw.gdn_A_log.data_ptr<c10::Half>(), 8, 16, D("gdn_Al"), nc);
    } else {
        TORCH_CHECK(gdn_neg_exp_A_[layer_idx].defined(), "build_gdn: neg_exp_A missing for layer ", layer_idx);
        rpu_launch_ddr_scatter_spm_dma(gdn_neg_exp_A_[layer_idx].data_ptr<c10::Half>(), 8, 16, D("gdn_Al"), nc);
    }
    rpu_launch_ddr_scatter_spm_dma(lw.gdn_dt_bias.data_ptr<c10::Half>(), 8, 16, D("gdn_dt"), nc);
    if (decode) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_MOE_GDN_STATE_DECODE_LOAD_SITE,
            qwen35_moe_dma_route(
                Qwen35MoeMutableDmaRoute::DDR_SCATTER_TO_SPM), chunk.idx);
        rpu_launch_ddr_scatter_spm_dma_mutable(
            &gdn_state_live_addr_[layer_idx], 0, Hc * Dk * Dv,
            Hc * Dk * Dv * 2, D("gdn_state"), nc);
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_MOE_GDN_CONV_DECODE_LOAD_SITE,
            qwen35_moe_dma_route(
                Qwen35MoeMutableDmaRoute::DDR_SCATTER_TO_SPM), chunk.idx);
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
            QWEN35_MOE_GDN_CONV_PREFILL_LOAD_SITE,
            qwen35_moe_dma_route(
                Qwen35MoeMutableDmaRoute::DDR_SCATTER_TO_SPM), chunk.idx);
        rpu_launch_ddr_scatter_spm_dma_mutable(
            &conv_state_live_addr_[layer_idx], 0, Kc * lconv,
            Kc * lconv * 2, D("gdn_cs"), nc);
    }
    bool load_chunk_masks = !decode;
    if (!decode && ctx().has_complete_physical_manifest()) {
        const auto& route = ctx().find_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_MOE_GDN_MASK_RESIDENCY_SITE);
        TORCH_CHECK(route.selector == 1 || route.selector == 2,
                    "Qwen3.5 MoE GDN mask schedule is invalid");
        const auto arguments = gdn_mask_schedule_arguments(ctx().stage_plan);
        ctx().consume_physical_route(
            FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_MOE_GDN_MASK_RESIDENCY_SITE,
            route.selector, /*resolved_flags=*/0, arguments);
        // Re-executed by every retained Graph. Retention spans this forward's
        // layer/chunk traversal, never an intervening owner or a later replay.
        load_chunk_masks = route.selector == 1 ||
            (layer_idx == arguments[3] && chunk.idx == 0);
    }
    if (load_chunk_masks) {  // fixed read-only 64x64 operands, identical on all cores
        rpu_launch_ddr_broadcast_spm_dma(gdn_tril_.data_ptr<c10::Half>(),   64 * 64, D("gdn_c_tril"),   nc);
        rpu_launch_ddr_broadcast_spm_dma(gdn_strict_.data_ptr<c10::Half>(), 64 * 64, D("gdn_c_strict"), nc);
    }

    // ── in_proj: q/k/v/z col-parallel + N_bg-padded b/a. input_layernorm is at the layer
    //    level → "input_norm"; decode writes its own buffers, prefill the conv xpad region. ──
    launch_linear(chunk.idx, D("input_norm"), lw.gdn_q_w, q_out, L, key_dim, hidden,
                  1, nc, /*bias=*/0, lw.gdn_q_scale);
    launch_linear(chunk.idx, D("input_norm"), lw.gdn_k_w, k_out, L, key_dim, hidden,
                  1, nc, /*bias=*/0, lw.gdn_k_scale);
    launch_linear(chunk.idx, D("input_norm"), lw.gdn_v_w, v_out, L, value_dim, hidden,
                  1, nc, /*bias=*/0, lw.gdn_v_scale);
    launch_linear(chunk.idx, D("input_norm"), lw.gdn_in_z_w, z_out, L, value_dim,
                  hidden, 1, nc, /*bias=*/0, lw.gdn_in_z_scale);
    launch_linear(chunk.idx, D("input_norm"), lw.gdn_b_bg_w, b_out, L, N_bg, hidden, 1, nc);
    launch_linear(chunk.idx, D("input_norm"), lw.gdn_a_bg_w, a_out, L, N_bg, hidden, 1, nc);

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

        // 4) l2norm + gating + recurrent delta-rule (Hc heads/core, 8-core SPMD). gdn_Al = RAW
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
        QWEN35_MOE_GDN_CONV_PREFILL_STORE_SITE,
        qwen35_moe_dma_route(
            Qwen35MoeMutableDmaRoute::SPM_SCATTER_TO_DDR), chunk.idx);
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
            QWEN35_MOE_GDN_STATE_PREFILL_LOAD_SITE,
            qwen35_moe_dma_route(
                Qwen35MoeMutableDmaRoute::DDR_SCATTER_TO_SPM), chunk.idx);
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
        QWEN35_MOE_GDN_STATE_PREFILL_STORE_SITE,
        qwen35_moe_dma_route(
            Qwen35MoeMutableDmaRoute::SPM_SCATTER_TO_DDR), chunk.idx);
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
            out_buf, gated_buf, D("gdn_normw"), L * Hc, Dv, 1e-6,
            RpuRmsNormSpmRoute::BASE, nc);
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
            QWEN35_MOE_GDN_PREPARE_ALL_REDUCE_SITE,
            static_cast<int64_t>(Qwen35MoeAllReduceRoute::PREPARE_RING_INPUT),
            chunk.idx);
        rpu_prepare_ring_all_reduce_input(
            m_buf, L, hidden, nc, num_cores());
    }
    launch_linear(chunk.idx, gated_buf, lw.gdn_out_w, m_buf, L, hidden, value_dim,
                  0, nc, 0, lw.gdn_out_scale);
    // Prefill persisted both states before their SPM slots were reused. Decode
    // keeps them live to this shared tail.
    if (decode) {
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_MOE_GDN_STATE_DECODE_STORE_SITE,
            qwen35_moe_dma_route(
                Qwen35MoeMutableDmaRoute::SPM_SCATTER_TO_DDR), chunk.idx);
        rpu_launch_spm_scatter_ddr_dma_mutable(D("gdn_state"), &gdn_state_live_addr_[layer_idx], 0, Hc * Dk * Dv, Hc * Dk * Dv * 2, nc);
        consume_manifest_route(
            FmbRouteFamily::MUTABLE_DMA,
            QWEN35_MOE_GDN_CONV_DECODE_STORE_SITE,
            qwen35_moe_dma_route(
                Qwen35MoeMutableDmaRoute::SPM_SCATTER_TO_DDR), chunk.idx);
        rpu_launch_spm_scatter_ddr_dma_mutable(D("gdn_cs"),    &conv_state_live_addr_[layer_idx], 0, Kc * lconv,    Kc * lconv * 2,    nc);
    }
    // AllReduce the out_proj partials + h_in residual → post-mixer stream (== HF `mixer_out`).
    consume_manifest_route(
        FmbRouteFamily::ALL_REDUCE, QWEN35_MOE_GDN_ALL_REDUCE_SITE,
        text_ring_route(L, hidden), chunk.idx);
    rpu_launch_all_reduce_sum_residual_kernel(
        m_buf, addr(0, "residual1"),
        D("residual2"), L, hidden, nc, num_cores());
    return;
}


at::Tensor* Qwen3_5MoeModel::gdn_state_for_layer(int layer_idx) {
    if (!gdn_states_ || layer_idx < 0 ||
        layer_idx >= static_cast<int>(gdn_states_->size())) return nullptr;
    return &(*gdn_states_)[layer_idx];
}

at::Tensor* Qwen3_5MoeModel::conv_state_for_layer(int layer_idx) {
    if (!conv_states_ || layer_idx < 0 ||
        layer_idx >= static_cast<int>(conv_states_->size())) return nullptr;
    return &(*conv_states_)[layer_idx];
}

// Post-mixer layer tail, shared by full-attn and GDN (HF: residual add is fused into
// the mixer's all_reduce; this is the post_attention_layernorm + MLP + residual half).
// CONTRACT: the residual stream (h_in + mixer_out) must be in "input_norm" (== the
// "residual2" alias) on entry. Emits: residual1 = post_norm(stream); MLP(+residual) →
// residual1; final_norm at the last layer; then output DMA.
void Qwen3_5MoeModel::emit_mlp_and_output(
    int layer_idx, const ChunkInfo& chunk) {
    TORCH_CHECK(
        static_cast<int64_t>(moe_layer_weights_.size()) == num_layers() &&
            expert_ids_.defined() && token_ids_.defined(),
        "qwen3_5_moe: independent MoE weights/route tables are not bound");
    const int64_t h = hidden_size();
    const bool is_last_layer = layer_idx == num_layers() - 1;

    // Mixer residual remains in input_norm; post-normed MoE input goes to residual1.
    rpu_launch_rmsnorm_spm_kernel(
        addr(0, "input_norm"), addr(0, "residual1"),
        layer_addr(layer_idx, 0, "post_norm_w"), chunk.len, h, eps_,
        RpuRmsNormSpmRoute::BASE, num_cores());

    // Finish the complete MLP and residual update before reusing its scratch.
    // Routing IDs stay slice-local; only the full-chunk streams advance.
    for_each_moe_slice(chunk, [&](int64_t row, int64_t t, int64_t invocation) {
        if (ctx().has_complete_physical_manifest() && layer_idx == 0) {
            ctx().consume_physical_route(
                FmbRouteFamily::GRAPH_SCHEDULE, QWEN35_MOE_ROUTER_SCHEDULE_SITE,
                t == 1 ? 1 : 2, /*resolved_flags=*/0,
                {t, t * moe_.top_k, moe_.num_experts, moe_.top_k,
                 chunk.idx, row, chunk.len}, invocation);
        }
        const uint32_t byte_offset = static_cast<uint32_t>(row * h * DWIDTH);
        const uint32_t input_output = addr(0, "residual1") + byte_offset;
        const uint32_t residual = addr(0, "input_norm") + byte_offset;
        emit_router_and_shuffle(layer_idx, invocation, t, input_output);
        emit_shared_expert(layer_idx, invocation, t, input_output);
        emit_routed_experts(layer_idx, invocation, t, input_output);
        emit_moe_merge_and_residual(invocation, t, residual, input_output);
    });

    if (is_last_layer) {
        rpu_launch_rmsnorm_spm_kernel(
            addr(0, "residual1"), addr(0, "residual1"),
            addr(0, "final_norm_w"), chunk.len, h, eps_,
            RpuRmsNormSpmRoute::BASE, num_cores());
    }

    if (!ctx().output_to_spm) {
        emit_layer_output_dma(layer_idx, chunk);
    }
}

// ── declare_buffers: union of full-attention and GDN buffer sets; per-layer
//    norm preloads are guarded so empty attention slots on GDN layers are not
//    read. ───────────────────────────────────────────────────────────────────
std::vector<BufferDecl> Qwen3_5MoeModel::declare_legacy_buffers(const LayoutContext& ctx) const {
    int64_t cs = ctx.chunk_size;
    int64_t h  = hidden_size();
    int64_t nq = num_q_heads();
    int64_t nkv = num_kv_heads();
    int64_t hd = head_dim();
    int64_t nl  = num_layers();
    moe_.validate(h);
    const int64_t e = moe_.num_experts;
    const int64_t k_moe = moe_.top_k;
    const int64_t moe_cs = std::min(cs, kMoeChunkSize);
    const int64_t r = moe_cs * k_moe;
    const int64_t local_e = e / router_tp();
    const int64_t local_ri =
        physical_routed_intermediate() / routed_expert_tp();
    const int64_t local_si =
        physical_shared_intermediate() / shared_expert_tp();
    TORCH_CHECK(
        moe_.tensor_parallel == routed_expert_tp() &&
            routing_tp() == routed_expert_tp() &&
            shared_expert_tp() == routed_expert_tp() &&
            routed_expert_tp() == mlp_tp() &&
            e % router_tp() == 0 &&
            physical_routed_intermediate() % routed_expert_tp() == 0 &&
            physical_shared_intermediate() % shared_expert_tp() == 0,
        "qwen3_5_moe: mixed MoE owners or physical intermediate drifted");

    // num_kv_heads() is the adapter-expanded effective KV width, so tp resolves
    // to the selected attention owner and each active core owns one whole
    // effective KV head. q/k/v/output/attn_gate and MoE temporaries are sized
    // per active core.
    int64_t tp = attn_tp();
    int64_t local_q  = nq / tp;
    int64_t local_kv = nkv * hd / tp;
    auto A = [](int64_t bytes) -> int64_t { return Align(bytes, 256); };

    int64_t res = A(cs * h * DWIDTH);
    int64_t q   = A(cs * local_q * hd * DWIDTH);
    int64_t kv  = A(cs * local_kv * DWIDTH);
    int64_t nw  = A(h * DWIDTH);
    int64_t hnw = A(hd * DWIDTH);
    int64_t tmp = A(sdpa_compute_tmp_v16_size(make_sdpa_config(), cs) * 32);

    const char* narrow_input_alias = "input_norm";

    std::vector<BufferDecl> decls;
    decls.push_back({"residual1",  res,  1, kMoePhaseReduce, StorageClass::Temp, 0, nullptr});
    decls.push_back({"input_norm", res,  1, kMoePhaseReduce, StorageClass::Temp, 0, nullptr});
    decls.push_back({"residual2",  0,    0, 0, StorageClass::Temp, 0, "input_norm"});
    decls.push_back({"q",          q,    22, 25, StorageClass::Temp, 0, nullptr});
    decls.push_back({"k",          kv,   22, 24, StorageClass::Temp, 0, nullptr});
    decls.push_back({"v",          kv,   22, 24, StorageClass::Temp, 0, nullptr});
    decls.push_back({"output",     q,    25, 27, StorageClass::Temp, 0, nullptr});
    decls.push_back({"attn_gate",  q,    22, 26, StorageClass::Temp, 0, nullptr});
    decls.push_back({"oproj",      res,  27, 27, StorageClass::Temp, 0, nullptr});
    decls.push_back({"sdpa_tmp",   tmp,  25, 25, StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_router_local", aligned_bytes(moe_cs * local_e),
                       kMoePhaseRouterLinear, kMoePhaseRouterGather,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_router_logits", aligned_bytes(moe_cs * e),
                       kMoePhaseRouterGather, kMoePhaseRouterSoftmax,
                       StorageClass::Temp, 0, nullptr});
    if (num_cores() == 6) {
        // Keep the route6 receiver distinct from the router4 all-gather source.
        // This is the same proven SPM arrangement as the bridge primitive and
        // avoids making core0 both the prior source and the broadcast target at
        // one address. BufferDecl allocates the receiver on cores 0..5.
        decls.push_back({"moe_route_logits", aligned_bytes(moe_cs * e),
                           kMoePhaseRouterGather, kMoePhaseRouterSoftmax,
                           StorageClass::Temp, 0, nullptr});
    }
    decls.push_back({"moe_router_probs", aligned_bytes(moe_cs * e),
                       kMoePhaseRouterSoftmax, kMoePhaseWeight,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_route_scores", aligned_bytes(r),
                       kMoePhaseTopK, kMoePhaseProbNorm,
                       StorageClass::Temp, 0, nullptr});
    // FP16 keys become U16 in-place in BinCount; both have two-byte storage.
    decls.push_back({"moe_route_expert_ids", aligned_bytes(r),
                       kMoePhaseTopK, kMoePhaseWeight,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_route_token_ids", aligned_bytes(r),
                       kMoePhaseSort, kMoePhaseScatter,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_m_sizes", aligned_bytes(e),
                       kMoePhaseBincount, kMoePhaseRoutedDown,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_route_indices_i32", aligned_bytes(r, 4),
                       kMoePhaseGatherIndex, kMoePhaseGather,
                       StorageClass::Temp, 0, nullptr});

    decls.push_back({"moe_shared_gate", aligned_bytes(moe_cs * local_si),
                       kMoePhaseSharedGate, kMoePhaseSharedAct,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_shared_up", aligned_bytes(moe_cs * local_si),
                       kMoePhaseSharedUp, kMoePhaseSharedAct,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_shared_act", aligned_bytes(moe_cs * local_si),
                       kMoePhaseSharedAct, kMoePhaseSharedDown,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_shared_alpha_pad", aligned_bytes(moe_cs * 16),
                       kMoePhaseSharedAlpha, kMoePhaseSharedAlphaSlice,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_shared_alpha", aligned_bytes(moe_cs),
                       kMoePhaseSharedAlphaSlice, kMoePhaseSharedWeight,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_partial", aligned_bytes(moe_cs * h),
                       kMoePhaseSharedDown, kMoePhaseReduce,
                       StorageClass::Temp, 0, nullptr});

    decls.push_back({"moe_shuffled_input", aligned_bytes(r * h),
                       kMoePhaseGather, kMoePhaseScatter,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_routed_gate", aligned_bytes(r * local_ri),
                       kMoePhaseRoutedGate, kMoePhaseRoutedAct,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_routed_up", aligned_bytes(r * local_ri),
                       kMoePhaseRoutedUp, kMoePhaseRoutedAct,
                       StorageClass::Temp, 0, nullptr});
    decls.push_back({"moe_routed_act", aligned_bytes(r * local_ri),
                       kMoePhaseRoutedAct, kMoePhaseRoutedDown,
                       StorageClass::Temp, 0, nullptr});
    // Down starts only after both grouped Gate/Up have released the shuffled
    // input, so routed output may reuse that exact [R,H] allocation.
    // Its physical root remains live through the output's final Scatter read.
    decls.push_back({"moe_routed_output", 0, 0, 0, StorageClass::Temp, 0,
                       "moe_shuffled_input"});


    const int64_t route_capacity = expert_ids_.size(0);
    TORCH_CHECK(route_capacity > 0 && token_ids_.size(0) == route_capacity,
                "qwen3_5_moe: route-table capacity mismatch");
    const int64_t expert_table_elems = route_capacity * e;
    const int64_t token_table_elems = route_capacity * k_moe;
    {
        BufferDecl d;
        d.name = "moe_expert_ids";
        d.size = aligned_bytes(expert_table_elems);
        d.storage = StorageClass::Persistent;
        d.preload_callback =
            [this, expert_table_elems,
             nc = routing_tp()](
                FusedModelBase&, int, uint32_t core0_addr) {
                TORCH_CHECK(expert_ids_.numel() == expert_table_elems,
                            "qwen3_5_moe: expert_ids changed after buffer declaration");
                rpu_launch_ddr_broadcast_spm_dma(
                    expert_ids_, /*src_offset_elements=*/0, expert_table_elems,
                    core0_addr, nc);
            };
        decls.push_back(d);
    }
    {
        BufferDecl d;
        d.name = "moe_token_ids";
        d.size = aligned_bytes(token_table_elems);
        d.storage = StorageClass::Persistent;
        d.preload_callback =
            [this, token_table_elems,
             nc = routing_tp()](
                FusedModelBase&, int, uint32_t core0_addr) {
                TORCH_CHECK(token_ids_.numel() == token_table_elems,
                            "qwen3_5_moe: token_ids changed after buffer declaration");
                // The fixed DMA is byte-oriented but its historical API uses
                // Half* for every two-byte payload. token_ids is raw U16/int16.
                rpu_launch_ddr_broadcast_spm_dma(
                    reinterpret_cast<c10::Half*>(token_ids_.data_ptr<int16_t>()),
                    token_table_elems, core0_addr, nc);
            };
        decls.push_back(d);
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
            // both full and GDN layers do the standard post_norm + sparse-MoE half
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
                        QWEN35_MOE_Q_NORM_PRELOAD_SITE,
                        qwen35_moe_dma_route(
                            Qwen35MoeMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                        /*resolved_flags=*/0,
                        {L, 1, head_dim(), 1}, L);
                }
                rpu_launch_ddr_broadcast_spm_dma(
                    layer_weights_[L].q_norm_w.data_ptr<c10::Half>(),
                    head_dim(), core0_addr,
                    num_cores() == 8 ? NUM_CORES : attn_tp());
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
                        QWEN35_MOE_K_NORM_PRELOAD_SITE,
                        qwen35_moe_dma_route(
                            Qwen35MoeMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                        /*resolved_flags=*/0,
                        {L, 1, head_dim(), 1}, L);
                }
                rpu_launch_ddr_broadcast_spm_dma(
                    layer_weights_[L].k_norm_w.data_ptr<c10::Half>(),
                    head_dim(), core0_addr,
                    num_cores() == 8 ? NUM_CORES : attn_tp());
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
                final_norm_w_.data_ptr<c10::Half>(), hidden_size(), core0_addr,
                num_cores());
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
                B("gdn_Al", 8), B("gdn_dt", 8),
            });
        } else {
            decls.insert(decls.end(), {
                BP("gdn_normw", Dv, 1, 18),
                BP("gdn_Al", 8, 1, 3), BP("gdn_dt", 8, 1, 3),
            });
        }
        if (ctx.chunk_size < 64) {
            decls.push_back(B("gdn_state", Hc * Dk * Dv));
            decls.push_back(B("gdn_cs", Kc * lconv));
            decls.push_back(B("gdn_zero", Dk * Dv));
        }
        // ── GDN DECODE (mixer) buffers — active-core, Hc heads/core ──
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
        if (ctx.chunk_size >= 64) {
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


std::vector<BufferDecl> Qwen3_5MoeModel::declare_buffers(const LayoutContext& layout) {
    auto declarations = declare_legacy_buffers(layout);
    switch (layout.forward_operand_residency) {
        case FmbForwardOperandResidency::UNSPECIFIED:
        case FmbForwardOperandResidency::PER_LAYER:
            return declarations;
        case FmbForwardOperandResidency::FORWARD: {
            TORCH_CHECK(has_gdn_ && layout.chunk_size >= 64,
                        "Qwen3.5 MoE retained masks require GDN prefill");
            const auto trial = detail::make_forward_spm_residency_candidate(
                declarations, {"gdn_c_tril", "gdn_c_strict"});
            TORCH_CHECK(trial.valid,
                        "Qwen3.5 MoE mask declaration cannot retain operands");
            return trial.declarations;
        }
    }
    TORCH_CHECK(false, "Qwen3.5 MoE unknown forward operand residency");
}

FmbForwardOperandResidency Qwen3_5MoeModel::gdn_mask_residency_for_candidate(
    const LayoutContext& layout) const {
    // A descriptor-bound layout is immutable: do not reselect from live free
    // space during Graph reconstruction, allocation, or replay.
    if (layout.forward_operand_residency !=
            FmbForwardOperandResidency::UNSPECIFIED) {
        TORCH_CHECK(layout.forward_operand_residency ==
                        FmbForwardOperandResidency::PER_LAYER ||
                    layout.forward_operand_residency ==
                        FmbForwardOperandResidency::FORWARD,
                    "Qwen3.5 MoE unknown bound mask residency");
        return layout.forward_operand_residency;
    }
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

std::vector<int64_t> Qwen3_5MoeModel::gdn_mask_schedule_arguments(
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

FmbForwardOperandResidency Qwen3_5MoeModel::physical_forward_operand_residency(
    const FmbPhysicalExecutionManifest& manifest) const {
    if (manifest.state != FmbPhysicalManifestState::COMPLETE ||
            manifest.physical_length == 1 || !has_gdn_)
        return FmbForwardOperandResidency::UNSPECIFIED;
    const auto route = std::find_if(
        manifest.routes.begin(), manifest.routes.end(), [](const auto& row) {
            return row.family == FmbRouteFamily::GRAPH_SCHEDULE &&
                row.site_id == QWEN35_MOE_GDN_MASK_RESIDENCY_SITE;
        });
    TORCH_CHECK(route != manifest.routes.end() && route->flags == 0 &&
                    route->invocation == 0 &&
                    (route->selector == 1 || route->selector == 2),
                "Qwen3.5 MoE prefill requires its bound GDN mask schedule");
    const auto& args = route->arguments;
    const auto first = std::find(layer_is_full_.begin(), layer_is_full_.end(), 0);
    TORCH_CHECK(args.size() == 9 && args[0] == 64 && args[1] == 64 &&
                    args[2] == 2 &&
                    args[3] == first - layer_is_full_.begin() &&
                    args[4] == std::count(layer_is_full_.begin(), layer_is_full_.end(), 0) &&
                    args[5] >= 64 && args[5] % 64 == 0 &&
                    args[6] == CeilDiv(manifest.physical_length, args[5]) &&
                    args[7] == gdn_tp() && args[8] == num_cores(),
                "Qwen3.5 MoE GDN mask geometry differs from its owner");
    return route->selector == 2 ? FmbForwardOperandResidency::FORWARD
                                : FmbForwardOperandResidency::PER_LAYER;
}


int64_t Qwen3_5MoeModel::subclass_layout_hash() const {
    int64_t hash = 0;
    if (num_cores() != 8) {
        for (const int64_t value : cold_topology_arguments())
            hash = detail::layout_mix(hash, value);
    }
    for (const int64_t value : {
             moe_.num_experts, moe_.top_k, moe_.routed_intermediate,
             moe_.shared_intermediate, moe_.tensor_parallel,
             gdn_nvh_, gdn_dk_, gdn_dv_, gdn_conv_dim_, gdn_kc_,
             static_cast<int64_t>(linear_acc32_),
             static_cast<int64_t>(retained_prefill_graph_)}) {
        hash = detail::layout_mix(hash, value);
    }
    if (num_cores() == 6) {
        hash = detail::layout_mix(hash, physical_routed_intermediate());
        hash = detail::layout_mix(hash, physical_shared_intermediate());
        hash = detail::layout_mix(
            hash, router_bridge_.defined() ? router_bridge_.size(0) : 0);
    }
    for (const uint8_t full : layer_is_full_)
        hash = detail::layout_mix(hash, full);
    return hash;
}

std::vector<int64_t> Qwen3_5MoeModel::kvinsert_cost_weight_identity() const {
    if (layer_weights_.empty()) return {};
    std::vector<int64_t> identity{1};
    if (num_cores() != 8) {
        const auto topology = cold_topology_arguments();
        identity.insert(identity.end(), topology.begin(), topology.end());
    }
    identity.insert(identity.end(), {
        moe_.num_experts, moe_.top_k, moe_.routed_intermediate,
        moe_.shared_intermediate, moe_.tensor_parallel,
        static_cast<int64_t>(linear_acc32_),
        static_cast<int64_t>(retained_prefill_graph_)});
    if (num_cores() == 6) {
        identity.push_back(physical_routed_intermediate());
        identity.push_back(physical_shared_intermediate());
    }
    append_kvinsert_cost_scalar_identity(identity, eps_);
    for (size_t i = 0; i < layer_weights_.size(); ++i) {
        identity.push_back(layer_is_full_[i]);
        const auto& lw = layer_weights_[i];
        for (const auto* tensor : {
                 &lw.q_w, &lw.k_w, &lw.v_w, &lw.o_w,
                 &lw.q_scale, &lw.k_scale, &lw.v_scale, &lw.o_scale,
                 &lw.q_norm_w, &lw.k_norm_w, &lw.attn_gate_w,
                 &lw.attn_gate_scale, &lw.input_norm_w, &lw.post_norm_w,
                 &lw.gdn_in_z_w, &lw.gdn_out_w, &lw.gdn_A_log,
                 &lw.gdn_dt_bias, &lw.gdn_norm_w, &lw.gdn_in_z_scale,
                 &lw.gdn_out_scale, &lw.gdn_q_w, &lw.gdn_k_w,
                 &lw.gdn_v_w, &lw.gdn_conv_q_w, &lw.gdn_conv_k_w,
                 &lw.gdn_conv_v_w, &lw.gdn_q_scale, &lw.gdn_k_scale,
                 &lw.gdn_v_scale, &lw.gdn_b_bg_w, &lw.gdn_a_bg_w}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
        const auto& mw = moe_layer_weights_[i];
        identity.push_back(mw.routed_weight_mode);
        for (const auto* tensor : {
                 &mw.router_w, &mw.routed_gate_w, &mw.routed_up_w,
                 &mw.routed_down_w, &mw.routed_gate_scale,
                 &mw.routed_up_scale, &mw.routed_down_scale,
                 &mw.shared_gate_w, &mw.shared_up_w, &mw.shared_down_w,
                 &mw.shared_scalar_gate_w}) {
            append_kvinsert_cost_tensor_identity(identity, *tensor);
        }
    }
    for (const auto* tensor : {
             &cos_, &sin_, &final_norm_w_, &expert_ids_, &token_ids_}) {
        append_kvinsert_cost_tensor_identity(identity, *tensor);
    }
    if (router_bridge_.defined())
        append_kvinsert_cost_tensor_identity(identity, router_bridge_);
    return identity;
}

ModelStaticConfig Qwen3_5MoeModel::static_config() {
    ModelStaticConfig cfg;
    cfg.num_layers = num_layers();
    cfg.cross_layer_batch_size = num_layers();   // single group (mirror Qwen3)
    cfg.fast_replay_skip_layer_loop = fast_replay_active_;
    return cfg;
}

ModelDynamicConfig Qwen3_5MoeModel::dynamic_config(const ChunkPlan& /*plan*/) {
    ModelDynamicConfig cfg;
    cfg.chunk_mode     = ChunkMode::SEQUENTIAL;
    cfg.inter_layer_io = InterLayerIO::AUTO;
    cfg.attention_policy = AttentionExecutionPolicy::AUTO;
    return cfg;
}

bool Qwen3_5MoeModel::subclass_chunk_size_valid(int64_t cs, int64_t seq_len,
                                             int64_t position) const {
    if (!sdpa_is_valid_chunk_size(make_sdpa_config(), cs, seq_len, position))
        return false;
    const int64_t moe_cs = std::min(cs, kMoeChunkSize);
    if ((expert_ids_.defined() && moe_cs > expert_ids_.size(0)) ||
        (token_ids_.defined() && moe_cs > token_ids_.size(0)))
        return false;

    if (seq_len > 1 && moe_cs * moe_.top_k > 4096) return false;
    // Decode uses the recurrent single-token path and does not have the 64-row
    // chunk constraint. Prefill consists of C=64 GDN sub-chunks.
    if (seq_len <= 1 || !has_gdn_) return true;
    if (cs % 64 != 0) return false;

    // MR-A: certified envelope, per candidate. Unlike CausalDecoderModel this
    // class already closed the explicit-override route below (the cap check was
    // always in the VALIDITY hook, not the cap hook), so the envelope only has to
    // join it here. The deny-by-default half lives in subclass_chunk_size_cap.
    if (!chunk_within_envelope(cs)) return false;

    return true;
}

// MR-A: Qwen3.5 had no cap hook at all — the auto scan was bounded only by the
// validity predicate above. The override route reached compute_chunks_impl:499
// without any once-per-resolve gate, so there was nowhere to fail an undeclared
// handle loudly. This adds that gate; it runs before SPM allocation on both
// planner entry points (fused_model_base.cpp:615 dry, :1024 forward).
int64_t Qwen3_5MoeModel::subclass_chunk_size_cap(int64_t seq_len,
                                              int64_t position) const {
    const int64_t envelope_chunk =
        enforce_chunk_envelope(seq_len, position, "Qwen3_5MoeModel");
    if (chunk_size_cap_ <= 0) return envelope_chunk;
    if (envelope_chunk <= 0) return chunk_size_cap_;
    return std::min(chunk_size_cap_, envelope_chunk);
}

bool Qwen3_5MoeModel::subclass_spm_kv_by_mha_eligible(
    const FmbThreeStageChunkPlan&, const LayoutContext&, int64_t) const {
    // The exact 35B sparse profile needs the MoE temporary arena throughout
    // the layer body. Its first port therefore owns the explicit DDR-KV path.
    return false;
}

std::vector<FmbPhysicalExecutionManifest>
Qwen3_5MoeModel::physical_manifest_domain_for_candidate(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len,
    int64_t position) const {
    LayoutContext ddr_layout = layout;
    ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;
    return {physical_manifest_for_candidate(
        plan, ddr_layout, physical_len, logical_len, position)};
}

FmbPhysicalExecutionManifest
Qwen3_5MoeModel::physical_manifest_for_candidate(
    const FmbThreeStageChunkPlan& plan,
    const LayoutContext& layout,
    int64_t physical_len, int64_t logical_len,
    int64_t position) const {
    TORCH_CHECK(
        layout.attention_policy == AttentionExecutionPolicy::DDR_KV,
        "Qwen3.5 MoE COMPLETE manifest supports only DDR-KV attention");

    FmbPhysicalExecutionManifest manifest;
    manifest.state = FmbPhysicalManifestState::COMPLETE;
    manifest.logical_length = logical_len;
    manifest.physical_length = physical_len;
    manifest.execution_padding_rows = physical_len - logical_len;
    manifest.kv_logical_length = physical_len == 1
        ? cos_.size(0) : position + logical_len;
    manifest.kv_insert_physical_rows = physical_len;
    manifest.graph_lifecycle = physical_len == 1 || retained_prefill_graph_
        ? FmbGraphLifecycle::RETAINED_CACHE
        : FmbGraphLifecycle::BOUNDED_ONESHOT;
    manifest.linear_accumulation = linear_acc32_
        ? FmbLinearAccumulationPolicy::ACC32
        : FmbLinearAccumulationPolicy::ACC16;
    if (num_cores() != 8) {
        manifest.routes.push_back({
            QWEN35_MOE_CORE_TOPOLOGY_SITE,
            FmbRouteFamily::GRAPH_SCHEDULE,
            /*selector=*/num_cores() == 6 ? 2 : 1,
            /*flags=*/0, cold_topology_arguments()});
    }

    if (has_gdn_ && physical_len > 1) {
        const auto residency = gdn_mask_residency_for_candidate(layout);
        manifest.routes.push_back({
            QWEN35_MOE_GDN_MASK_RESIDENCY_SITE, FmbRouteFamily::GRAPH_SCHEDULE,
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
        append_route(
            FmbRouteFamily::LINEAR, QWEN35_MOE_LINEAR_SITE,
            static_cast<int64_t>(selector),
            linear_invocation(chunk_idx, selector));
    };

    for (int64_t layer = 0; layer < num_layers(); ++layer) {
        if (!layer_is_full_[layer]) continue;
        for (const int64_t site : {
                 QWEN35_MOE_Q_NORM_PRELOAD_SITE,
                 QWEN35_MOE_K_NORM_PRELOAD_SITE}) {
            manifest.routes.push_back({
                site, FmbRouteFamily::MUTABLE_DMA,
                qwen35_moe_dma_route(
                    Qwen35MoeMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*flags=*/0, {layer, 1, head_dim(), 1}, layer});
        }
    }

    const int64_t full_layers = std::count_if(
        layer_is_full_.begin(), layer_is_full_.end(),
        [](uint8_t value) { return value != 0; });
    const int64_t gdn_layers = num_layers() - full_layers;
    for (const ChunkInfo& chunk : plan.compute.chunks) {
        const bool decode = chunk.len == 1;
        manifest.routes.push_back({
            QWEN35_MOE_EXACT_PROFILE_SITE, FmbRouteFamily::GRAPH_SCHEDULE,
            /*selector=*/num_cores() == 6 ? 2 : 1,
            /*flags=*/0, exact_profile_arguments(), chunk.idx});

        bool needs_auto_tile = false;
        bool needs_gemv = false;
        auto observe_linear = [&](const at::Tensor& weight) {
            const FmbLinearRouteSelector selector =
                linear_route_selector(weight, chunk.len);
            needs_auto_tile |= selector == FmbLinearRouteSelector::AUTO_TILE;
            needs_gemv |= selector == FmbLinearRouteSelector::GEMV;
        };
        for (int64_t layer = 0; layer < num_layers(); ++layer) {
            const auto& lw = layer_weights_[layer];
            if (layer_is_full_[layer]) {
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
        }
        if (needs_auto_tile)
            append_linear(FmbLinearRouteSelector::AUTO_TILE, chunk.idx);
        if (needs_gemv)
            append_linear(FmbLinearRouteSelector::GEMV, chunk.idx);

        append_mlp_manifest_routes(manifest, chunk);

        if (full_layers > 0) {
            manifest.routes.push_back({
                QWEN35_MOE_ATTENTION_SITE, FmbRouteFamily::ATTENTION,
                static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV),
                QWEN35_MOE_ATTN_DDR_REQUIRED,
                decode ? std::vector<int64_t>{1}
                       : std::vector<int64_t>{
                             1, chunk.len, chunk.kv_seq_len,
                             num_q_heads(), num_kv_heads(), head_dim(),
                             attn_tp(), 1},
                chunk.idx});
            append_route(
                FmbRouteFamily::ROPE,
                decode ? QWEN35_MOE_PARTIAL_ROPE_1D_SITE
                       : QWEN35_MOE_PARTIAL_MROPE_SITE,
                static_cast<int64_t>(
                    decode ? Qwen35MoeRopeRoute::PARTIAL_ROPE_1D
                           : Qwen35MoeRopeRoute::PARTIAL_MROPE),
                chunk.idx);
            append_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN35_MOE_FULL_PREPARE_ALL_REDUCE_SITE,
                static_cast<int64_t>(
                    Qwen35MoeAllReduceRoute::PREPARE_RING_INPUT),
                chunk.idx);
            append_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN35_MOE_FULL_ALL_REDUCE_SITE,
                text_ring_route(chunk.len, hidden_size()), chunk.idx);
        }
        if (gdn_layers > 0) {
            if (gdn_tp() != num_cores()) {
                append_route(
                    FmbRouteFamily::ALL_REDUCE,
                    QWEN35_MOE_GDN_PREPARE_ALL_REDUCE_SITE,
                    static_cast<int64_t>(
                        Qwen35MoeAllReduceRoute::PREPARE_RING_INPUT),
                    chunk.idx);
            }
            const bool initial_prefill =
                !decode && position == 0 && chunk.offset == 0;
            if (decode) {
                for (const int64_t site : {
                         QWEN35_MOE_GDN_STATE_DECODE_LOAD_SITE,
                         QWEN35_MOE_GDN_CONV_DECODE_LOAD_SITE}) {
                    append_route(
                        FmbRouteFamily::MUTABLE_DMA, site,
                        qwen35_moe_dma_route(
                            Qwen35MoeMutableDmaRoute::DDR_SCATTER_TO_SPM),
                        chunk.idx);
                }
                for (const int64_t site : {
                         QWEN35_MOE_GDN_STATE_DECODE_STORE_SITE,
                         QWEN35_MOE_GDN_CONV_DECODE_STORE_SITE}) {
                    append_route(
                        FmbRouteFamily::MUTABLE_DMA, site,
                        qwen35_moe_dma_route(
                            Qwen35MoeMutableDmaRoute::SPM_SCATTER_TO_DDR),
                        chunk.idx);
                }
            } else {
                if (!initial_prefill) {
                    for (const int64_t site : {
                             QWEN35_MOE_GDN_CONV_PREFILL_LOAD_SITE,
                             QWEN35_MOE_GDN_STATE_PREFILL_LOAD_SITE}) {
                        append_route(
                            FmbRouteFamily::MUTABLE_DMA, site,
                            qwen35_moe_dma_route(
                                Qwen35MoeMutableDmaRoute::DDR_SCATTER_TO_SPM),
                            chunk.idx);
                    }
                }
                for (const int64_t site : {
                         QWEN35_MOE_GDN_CONV_PREFILL_STORE_SITE,
                         QWEN35_MOE_GDN_STATE_PREFILL_STORE_SITE}) {
                    append_route(
                        FmbRouteFamily::MUTABLE_DMA, site,
                        qwen35_moe_dma_route(
                            Qwen35MoeMutableDmaRoute::SPM_SCATTER_TO_DDR),
                        chunk.idx);
                }
            }
            append_route(
                FmbRouteFamily::ALL_REDUCE,
                QWEN35_MOE_GDN_ALL_REDUCE_SITE,
                text_ring_route(chunk.len, hidden_size()), chunk.idx);
        }
    }

    if (full_layers > 0) {
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
                      QWEN35_MOE_KV_INSERT_SITE, manifest.graph_lifecycle,
                      insert_position, chunk.len, chunk.len, attn_tp(),
                      num_kv_heads(), head_dim(),
                      QWEN35_MOE_KV_CAPABILITIES);
            const KvInsertRouteArguments arguments =
                rpu_kvinsert_route_arguments(
                    kv_plan, attn_tp(), num_kv_heads(), head_dim());
            manifest.routes.push_back({
                QWEN35_MOE_KV_INSERT_SITE, FmbRouteFamily::KV_INSERT,
                static_cast<int64_t>(kv_plan.route()),
                QWEN35_MOE_KV_REASON_DDR_REQUIRED |
                    (dynamic_decode_position
                         ? KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION : 0),
                {arguments.begin(), arguments.end()}, chunk.idx});
        }
    }
    append_fmb_shared_runtime_routes(
        manifest, plan, hidden_size(), FMB_SHARED_LAYER_INPUT_DMA,
        /*compute_row_multiplier=*/1, num_cores(), mlp_tp());
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

void Qwen3_5MoeModel::append_mlp_manifest_routes(
    FmbPhysicalExecutionManifest& manifest, const ChunkInfo& chunk) const {
    const int64_t grouped_mode = moe_layer_weights_.empty()
        ? 5 : moe_layer_weights_.front().routed_weight_mode;
    const int64_t local_i =
        physical_routed_intermediate() / routed_expert_tp();
    const int nc = routed_expert_tp();
    const int router_nc = router_tp();

    auto append_route = [&](FmbRouteFamily family, int64_t site_id,
                            int64_t selector, int64_t invocation) {
        manifest.routes.push_back({
            site_id, family, selector, /*flags=*/0, {}, invocation});
    };
    for_each_moe_slice(chunk, [&](int64_t row, int64_t t, int64_t invocation) {
        manifest.routes.push_back({
            QWEN35_MOE_ROUTER_SCHEDULE_SITE,
            FmbRouteFamily::GRAPH_SCHEDULE,
            t == 1 ? 1 : 2, /*flags=*/0,
            {t, t * moe_.top_k, moe_.num_experts,
             moe_.top_k, chunk.idx, row, chunk.len}, invocation});
        bool needs_auto_tile = false;
        bool needs_gemv = false;
        auto observe_linear = [&](const at::Tensor& weight) {
            const auto selector = linear_route_selector(weight, t);
            needs_auto_tile |= selector == FmbLinearRouteSelector::AUTO_TILE;
            needs_gemv |= selector == FmbLinearRouteSelector::GEMV;
        };
        for (const auto& mw : moe_layer_weights_) {
            observe_linear(mw.router_w);
            observe_linear(mw.shared_gate_w);
            observe_linear(mw.shared_up_w);
            observe_linear(mw.shared_down_w);
            observe_linear(mw.shared_scalar_gate_w);
        }
        for (auto selector : {FmbLinearRouteSelector::AUTO_TILE,
                              FmbLinearRouteSelector::GEMV}) {
            if ((selector == FmbLinearRouteSelector::AUTO_TILE && needs_auto_tile) ||
                (selector == FmbLinearRouteSelector::GEMV && needs_gemv)) {
                append_route(
                    FmbRouteFamily::LINEAR, QWEN35_MOE_LINEAR_SITE,
                    static_cast<int64_t>(selector),
                    linear_invocation(invocation, selector, /*mlp_slice=*/true));
            }
        }
        manifest.routes.push_back({
            QWEN35_MOE_GROUPED_FP8_SITE, FmbRouteFamily::LINEAR,
            grouped_mode, /*flags=*/0,
            {local_i, hidden_size(), moe_.num_experts, 1, nc, linear_acc32_ ? 1 : 0},
            invocation * 2});
        manifest.routes.push_back({
            QWEN35_MOE_GROUPED_FP8_SITE, FmbRouteFamily::LINEAR,
            grouped_mode, /*flags=*/0,
            {hidden_size(), local_i, moe_.num_experts, 0, nc, linear_acc32_ ? 1 : 0},
            invocation * 2 + 1});

        const RpuAllGatherSchedule gather_schedule =
            rpu_resolve_all_gather_schedule(
                moe_.num_experts / router_nc, sizeof(c10::Half));
        manifest.routes.push_back({
            QWEN35_MOE_ROUTER_ALL_GATHER_SITE, FmbRouteFamily::COLLECTIVE,
            static_cast<int64_t>(gather_schedule), /*flags=*/0,
            {t, moe_.num_experts / router_nc,
             static_cast<int64_t>(sizeof(c10::Half)), router_nc}, invocation});
        if (num_cores() == 6) {
            manifest.routes.push_back({
                QWEN35_MOE_ROUTER_BRIDGE_STORE_SITE,
                FmbRouteFamily::MUTABLE_DMA,
                qwen35_moe_dma_route(
                    Qwen35MoeMutableDmaRoute::SPM_CORE0_TO_DDR),
                /*flags=*/0,
                {t, moe_.num_experts, router_nc, routing_tp(),
                 router_bridge_.size(0)},
                invocation});
            manifest.routes.push_back({
                QWEN35_MOE_ROUTER_BRIDGE_LOAD_SITE,
                FmbRouteFamily::MUTABLE_DMA,
                qwen35_moe_dma_route(
                    Qwen35MoeMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*flags=*/0,
                {t, moe_.num_experts, router_nc, routing_tp(),
                 router_bridge_.size(0)},
                invocation});
        }
        append_route(
            FmbRouteFamily::ACTIVATION, QWEN35_MOE_SHARED_SILU_SITE,
            static_cast<int64_t>(Qwen35MoeActivationRoute::SILU_MUL),
            invocation * 4 + 1);
        append_route(
            FmbRouteFamily::ACTIVATION, QWEN35_MOE_SHARED_SIGMOID_SITE,
            static_cast<int64_t>(Qwen35MoeActivationRoute::SIGMOID),
            invocation * 4 + 2);
        append_route(
            FmbRouteFamily::ACTIVATION, QWEN35_MOE_ROUTED_SILU_SITE,
            static_cast<int64_t>(Qwen35MoeActivationRoute::SILU_MUL),
            invocation * 4 + 1);

        append_route(
            FmbRouteFamily::ALL_REDUCE, QWEN35_MOE_ALL_REDUCE_SITE,
            text_ring_route(t, hidden_size()), invocation);
    });
}

FmbPhysicalManifestForwardCapability
Qwen3_5MoeModel::physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& manifest) const {
    const FmbGraphLifecycle expected = manifest.logical_length == 1 ||
            retained_prefill_graph_
        ? FmbGraphLifecycle::RETAINED_CACHE
        : FmbGraphLifecycle::BOUNDED_ONESHOT;
    return {true, expected};
}

// Per-forward prefill M-RoPE tables. Generic Qwen3.5 uses a fresh oneshot graph;
// fixed-profile retained graphs rely on same-shape copy_ preserving this address.
void Qwen3_5MoeModel::set_prefill_rope(const at::Tensor& cos, const at::Tensor& sin) {
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
void Qwen3_5MoeModel::set_weights(
    at::TensorList q_w_list, at::TensorList k_w_list,
    at::TensorList v_w_list, at::TensorList o_w_list,
    at::TensorList q_norm_list, at::TensorList k_norm_list,
    at::TensorList attn_gate_list,
    at::TensorList q_scale_list, at::TensorList k_scale_list,
    at::TensorList v_scale_list, at::TensorList o_scale_list,
    at::TensorList attn_gate_scale_list,
    at::TensorList input_norm_list, at::TensorList post_norm_list,
    at::TensorList router_list,
    at::TensorList routed_gate_list, at::TensorList routed_up_list,
    at::TensorList routed_down_list,
    at::TensorList routed_gate_scale_list,
    at::TensorList routed_up_scale_list,
    at::TensorList routed_down_scale_list,
    at::TensorList shared_gate_list, at::TensorList shared_up_list,
    at::TensorList shared_down_list,
    at::TensorList shared_scalar_gate_list,
    const at::Tensor& cos, const at::Tensor& sin,
    const at::Tensor& final_norm_w,
    const at::Tensor& expert_ids, const at::Tensor& token_ids,
    at::IntArrayRef layer_is_full,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size,
    int64_t num_experts, int64_t top_k,
    int64_t routed_intermediate, int64_t shared_intermediate,
    int64_t tensor_parallel, int64_t routed_weight_mode,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::TensorList gdn_in_z_list, at::TensorList gdn_out_list,
    at::TensorList gdn_in_z_scale_list,
    at::TensorList gdn_out_scale_list,
    at::TensorList gdn_A_log_list, at::TensorList gdn_dt_bias_list,
    at::TensorList gdn_norm_list,
    int64_t gdn_num_v_heads, int64_t gdn_key_head_dim,
    int64_t gdn_value_head_dim, int64_t gdn_conv_dim, int64_t gdn_conv_kernel,
    at::TensorList gdn_q_list, at::TensorList gdn_k_list, at::TensorList gdn_v_list,
    at::TensorList gdn_q_scale_list, at::TensorList gdn_k_scale_list,
    at::TensorList gdn_v_scale_list,
    at::TensorList gdn_cq_list, at::TensorList gdn_ck_list, at::TensorList gdn_cv_list,
    at::TensorList gdn_b_bg_list, at::TensorList gdn_a_bg_list) {

    const int64_t N = static_cast<int64_t>(layer_is_full.size());
    const int64_t effective_kv_heads = num_cores() == 8 ? 8 : 4;
    const int64_t physical_routed_intermediate =
        num_cores() == 6 ? 576 : routed_intermediate;
    const int64_t physical_shared_intermediate =
        num_cores() == 6 ? 576 : shared_intermediate;
    TORCH_CHECK(
        (num_cores() == 4 || num_cores() == 6 || num_cores() == 8) &&
            N == 40 &&
            num_q_heads == 16 && num_kv_heads == effective_kv_heads &&
            head_dim == 256 && hidden_size == 2048 &&
            num_experts == 256 && top_k == 8 &&
            routed_intermediate == 512 && shared_intermediate == 512 &&
            tensor_parallel == num_cores() && routed_weight_mode == 5 &&
            gdn_num_v_heads == 32 && gdn_key_head_dim == 128 &&
            gdn_value_head_dim == 128 && gdn_conv_dim == 8192 &&
            gdn_conv_kernel == 4,
        "RPU_MODEL_REJECT:EXACT_PROFILE: qwen3_5_moe supports only the "
        "Qwen3.5-35B-A3B TP4/mixed-TP6/TP8 mixed-E4M3 profiles "
        "(L40/H2048/Q16/effective-KV=4 for reduced profiles/HD256/"
        "E256/top8/logical-I512/logical-shared512, "
        "GDN V32/Dk128/Dv128/conv8192/K4)");
    TORCH_CHECK(
        mrope_section.size() == 3 && mrope_section[0] == 11 &&
            mrope_section[1] == 11 && mrope_section[2] == 10,
        "RPU_MODEL_REJECT:EXACT_PROFILE: qwen3_5_moe requires "
        "mrope_section=[11,11,10]");
    for (int64_t layer = 0; layer < N; ++layer) {
        const int64_t expected_full = (layer % 4 == 3) ? 1 : 0;
        TORCH_CHECK(
            layer_is_full[layer] == expected_full,
            "RPU_MODEL_REJECT:EXACT_PROFILE: qwen3_5_moe layer schedule "
            "requires every fourth layer to use full attention; layer ",
            layer, " got ", layer_is_full[layer]);
    }
    const auto check_list = [N](at::TensorList list, const char* name) {
        TORCH_CHECK(static_cast<int64_t>(list.size()) == N,
                    "qwen3_5_moe set_weights: ", name,
                    " must have num_layers=", N, " entries, got ", list.size());
    };
    check_list(q_w_list, "q_w_list");
    check_list(k_w_list, "k_w_list");
    check_list(v_w_list, "v_w_list");
    check_list(o_w_list, "o_w_list");
    check_list(q_norm_list, "q_norm_list");
    check_list(k_norm_list, "k_norm_list");
    check_list(attn_gate_list, "attn_gate_list");
    check_list(q_scale_list, "q_scale_list");
    check_list(k_scale_list, "k_scale_list");
    check_list(v_scale_list, "v_scale_list");
    check_list(o_scale_list, "o_scale_list");
    check_list(attn_gate_scale_list, "attn_gate_scale_list");
    check_list(input_norm_list, "input_norm_list");
    check_list(post_norm_list, "post_norm_list");
    check_list(router_list, "router_list");
    check_list(routed_gate_list, "routed_gate_list");
    check_list(routed_up_list, "routed_up_list");
    check_list(routed_down_list, "routed_down_list");
    check_list(routed_gate_scale_list, "routed_gate_scale_list");
    check_list(routed_up_scale_list, "routed_up_scale_list");
    check_list(routed_down_scale_list, "routed_down_scale_list");
    check_list(shared_gate_list, "shared_gate_list");
    check_list(shared_up_list, "shared_up_list");
    check_list(shared_down_list, "shared_down_list");
    check_list(shared_scalar_gate_list, "shared_scalar_gate_list");
    check_list(gdn_in_z_list, "gdn_in_z_list");
    check_list(gdn_out_list, "gdn_out_list");
    check_list(gdn_in_z_scale_list, "gdn_in_z_scale_list");
    check_list(gdn_out_scale_list, "gdn_out_scale_list");
    check_list(gdn_A_log_list, "gdn_A_log_list");
    check_list(gdn_dt_bias_list, "gdn_dt_bias_list");
    check_list(gdn_norm_list, "gdn_norm_list");
    check_list(gdn_q_list, "gdn_q_list");
    check_list(gdn_k_list, "gdn_k_list");
    check_list(gdn_v_list, "gdn_v_list");
    check_list(gdn_q_scale_list, "gdn_q_scale_list");
    check_list(gdn_k_scale_list, "gdn_k_scale_list");
    check_list(gdn_v_scale_list, "gdn_v_scale_list");
    check_list(gdn_cq_list, "gdn_cq_list");
    check_list(gdn_ck_list, "gdn_ck_list");
    check_list(gdn_cv_list, "gdn_cv_list");
    check_list(gdn_b_bg_list, "gdn_b_bg_list");
    check_list(gdn_a_bg_list, "gdn_a_bg_list");

    Qwen3_5MoeStaticConfig next_moe{
        num_experts, top_k, routed_intermediate, shared_intermediate,
        tensor_parallel, /*normalize_topk=*/true};
    next_moe.validate(hidden_size);
    TORCH_CHECK(use_silu,
                "qwen3_5_moe set_weights: Qwen3.5 routed/shared experts require SiLU");

    const auto check_tensor = [](const at::Tensor& tensor, const char* name,
                                 int64_t layer, int64_t dim) {
        TORCH_CHECK(tensor.defined() && tensor.numel() > 0,
                    "qwen3_5_moe set_weights: ", name, "[", layer,
                    "] must be non-empty");
        TORCH_CHECK(tensor.device().type() == at::kPrivateUse1
                    && tensor.scalar_type() == at::kHalf
                    && tensor.is_contiguous(),
                    "qwen3_5_moe set_weights: ", name, "[", layer,
                    "] must be a contiguous FP16 RPU tensor");
        TORCH_CHECK(tensor.dim() == dim,
                    "qwen3_5_moe set_weights: ", name, "[", layer,
                    "] must be ", dim, "D, got ", tensor.dim(), "D");
    };
    const auto check_bytes = [](const at::Tensor& tensor, const char* name,
                                int64_t layer, int64_t expected) {
        TORCH_CHECK(tensor.nbytes() == expected,
                    "qwen3_5_moe set_weights: ", name, "[", layer,
                    "] has ", tensor.nbytes(), " bytes, expected ", expected);
    };
    const auto check_shape = [](const at::Tensor& tensor,
                                at::IntArrayRef expected,
                                const char* name, int64_t layer) {
        TORCH_CHECK(
            tensor.sizes() == expected,
            "qwen3_5_moe set_weights: ", name, "[", layer,
            "] shape ", tensor.sizes(), " does not match ", expected);
    };
    const auto check_weight_alignment = [](const at::Tensor& tensor,
                                           const char* name, int64_t layer) {
        TORCH_CHECK((RpuGetDevAddr(tensor.data_ptr()) & 0xFF) == 0,
                    "qwen3_5_moe set_weights: ", name, "[", layer,
                    "] DDR address must be 256-byte aligned");
    };
    const auto check_projection = [&](const at::Tensor& weight,
                                      const at::Tensor& scale,
                                      const char* name, int64_t layer,
                                      int64_t out_features,
                                      int64_t in_features) {
        TORCH_CHECK(weight.defined() && weight.numel() > 0 &&
                        weight.device().type() == at::kPrivateUse1 &&
                        weight.is_contiguous() && weight.dim() == 2 &&
                        weight.size(0) == out_features &&
                        weight.size(1) == in_features,
                    "qwen3_5_moe set_weights: ", name, "[", layer,
                    "] must be contiguous RPU [", out_features, ",",
                    in_features, "], got ", weight.sizes());
        TORCH_CHECK(weight.scalar_type() == at::kByte,
                    "qwen3_5_moe set_weights: ", name, "[", layer,
                    "] must use full-byte uint8 E4M3 storage");
        check_weight_alignment(weight, name, layer);
        check_tensor(scale, name, layer, 1);
        TORCH_CHECK(scale.numel() == out_features,
                    "qwen3_5_moe set_weights: FP8 ", name, "[", layer,
                    "] scale must have ", out_features, " elements, got ",
                    scale.numel());
        check_bytes(weight, name, layer, out_features * in_features);
        check_weight_alignment(scale, name, layer);
    };

    const int64_t resolved_weight_mode = routed_weight_mode;
    TORCH_CHECK(resolved_weight_mode == 5,
                "qwen3_5_moe set_weights: routed_weight_mode must be "
                "5 (full-byte FP8 E4M3)");
    const auto routed_dtype = at::kByte;
    const int64_t routed_dwidth = 1;
    const int64_t routed_weight_bytes =
        num_experts * hidden_size * physical_routed_intermediate *
        routed_dwidth;

    for (int64_t i = 0; i < N; ++i) {
        TORCH_CHECK(layer_is_full[i] == 0 || layer_is_full[i] == 1,
                    "qwen3_5_moe set_weights: layer_is_full[", i,
                    "] must be 0 or 1, got ", layer_is_full[i]);
        check_tensor(input_norm_list[i], "input_norm_list", i, 1);
        check_tensor(post_norm_list[i], "post_norm_list", i, 1);
        check_tensor(router_list[i], "router_list", i, 2);
        check_tensor(shared_gate_list[i], "shared_gate_list", i, 2);
        check_tensor(shared_up_list[i], "shared_up_list", i, 2);
        check_tensor(shared_down_list[i], "shared_down_list", i, 2);
        check_tensor(shared_scalar_gate_list[i],
                     "shared_scalar_gate_list", i, 2);
        check_shape(input_norm_list[i], {hidden_size}, "input_norm_list", i);
        check_shape(post_norm_list[i], {hidden_size}, "post_norm_list", i);
        check_shape(router_list[i], {num_experts, hidden_size},
                    "router_list", i);
        check_shape(shared_gate_list[i],
                    {physical_shared_intermediate, hidden_size},
                    "shared_gate_list", i);
        check_shape(shared_up_list[i],
                    {physical_shared_intermediate, hidden_size},
                    "shared_up_list", i);
        check_shape(shared_down_list[i],
                    {hidden_size, physical_shared_intermediate},
                    "shared_down_list", i);
        check_shape(shared_scalar_gate_list[i],
                    {16 * shared_expert_tp(), hidden_size},
                    "shared_scalar_gate_list", i);
        check_bytes(router_list[i], "router_list", i,
                    hidden_size * num_experts * 2);
        check_bytes(shared_gate_list[i], "shared_gate_list", i,
                    hidden_size * physical_shared_intermediate * 2);
        check_bytes(shared_up_list[i], "shared_up_list", i,
                    hidden_size * physical_shared_intermediate * 2);
        check_bytes(shared_down_list[i], "shared_down_list", i,
                    hidden_size * physical_shared_intermediate * 2);
        check_bytes(shared_scalar_gate_list[i],
                    "shared_scalar_gate_list", i,
                    16 * shared_expert_tp() * hidden_size * 2);
        check_weight_alignment(router_list[i], "router_list", i);
        check_weight_alignment(shared_gate_list[i], "shared_gate_list", i);
        check_weight_alignment(shared_up_list[i], "shared_up_list", i);
        check_weight_alignment(shared_down_list[i], "shared_down_list", i);
        check_weight_alignment(shared_scalar_gate_list[i],
                               "shared_scalar_gate_list", i);

        const std::array<std::pair<const at::Tensor*, const char*>, 3>
            routed_weights{{
                {&routed_gate_list[i], "routed_gate_list"},
                {&routed_up_list[i], "routed_up_list"},
                {&routed_down_list[i], "routed_down_list"}}};
        for (const auto& item : routed_weights) {
            const auto& tensor = *item.first;
            TORCH_CHECK(tensor.defined() && tensor.numel() > 0 &&
                            tensor.device().type() == at::kPrivateUse1 &&
                            tensor.scalar_type() == routed_dtype &&
                            tensor.is_contiguous(),
                        "qwen3_5_moe set_weights: ", item.second, "[", i,
                        "] has the wrong storage for routed_weight_mode=",
                        resolved_weight_mode);
            check_bytes(tensor, item.second, i, routed_weight_bytes);
            check_shape(
                tensor,
                {num_experts, hidden_size * physical_routed_intermediate},
                item.second, i);
            check_weight_alignment(tensor, item.second, i);
        }

        const std::array<std::pair<const at::Tensor*, const char*>, 3>
            routed_scales{{
                {&routed_gate_scale_list[i], "routed_gate_scale_list"},
                {&routed_up_scale_list[i], "routed_up_scale_list"},
                {&routed_down_scale_list[i], "routed_down_scale_list"}}};
        const std::array<int64_t, 3> scale_bytes{{
            num_experts * physical_routed_intermediate * 2,
            num_experts * physical_routed_intermediate * 2,
            num_experts * hidden_size * 2}};
        for (size_t j = 0; j < routed_scales.size(); ++j) {
            const auto& tensor = *routed_scales[j].first;
            check_tensor(tensor, routed_scales[j].second, i, 2);
            check_bytes(tensor, routed_scales[j].second, i, scale_bytes[j]);
            check_shape(
                tensor,
                {num_experts,
                 j == 2 ? hidden_size : physical_routed_intermediate},
                routed_scales[j].second, i);
            check_weight_alignment(tensor, routed_scales[j].second, i);
        }
        if (layer_is_full[i]) {
            check_projection(q_w_list[i], q_scale_list[i], "q_w_list", i,
                             num_q_heads * head_dim, hidden_size);
            check_projection(k_w_list[i], k_scale_list[i], "k_w_list", i,
                             num_kv_heads * head_dim, hidden_size);
            check_projection(v_w_list[i], v_scale_list[i], "v_w_list", i,
                             num_kv_heads * head_dim, hidden_size);
            check_projection(o_w_list[i], o_scale_list[i], "o_w_list", i,
                             hidden_size, num_q_heads * head_dim);
            check_projection(attn_gate_list[i], attn_gate_scale_list[i],
                             "attn_gate_list", i,
                             num_q_heads * head_dim, hidden_size);
            check_tensor(q_norm_list[i], "q_norm_list", i, 1);
            check_tensor(k_norm_list[i], "k_norm_list", i, 1);
            check_shape(q_norm_list[i], {head_dim}, "q_norm_list", i);
            check_shape(k_norm_list[i], {head_dim}, "k_norm_list", i);
        } else {
            const int64_t gdn_value_dim =
                gdn_num_v_heads * gdn_value_head_dim;
            const int64_t gdn_key_dim =
                (gdn_conv_dim - gdn_value_dim) / 2;
            check_projection(gdn_in_z_list[i], gdn_in_z_scale_list[i],
                             "gdn_in_z_list", i, gdn_value_dim, hidden_size);
            check_projection(gdn_out_list[i], gdn_out_scale_list[i],
                             "gdn_out_list", i, hidden_size, gdn_value_dim);
            check_tensor(gdn_A_log_list[i], "gdn_A_log_list", i, 1);
            check_tensor(gdn_dt_bias_list[i], "gdn_dt_bias_list", i, 1);
            check_tensor(gdn_norm_list[i], "gdn_norm_list", i, 1);
            check_projection(gdn_q_list[i], gdn_q_scale_list[i],
                             "gdn_q_list", i, gdn_key_dim, hidden_size);
            check_projection(gdn_k_list[i], gdn_k_scale_list[i],
                             "gdn_k_list", i, gdn_key_dim, hidden_size);
            check_projection(gdn_v_list[i], gdn_v_scale_list[i],
                             "gdn_v_list", i, gdn_value_dim, hidden_size);
            check_tensor(gdn_cq_list[i], "gdn_cq_list", i, 3);
            check_tensor(gdn_ck_list[i], "gdn_ck_list", i, 3);
            check_tensor(gdn_cv_list[i], "gdn_cv_list", i, 3);
            check_tensor(gdn_b_bg_list[i], "gdn_b_bg_list", i, 2);
            check_tensor(gdn_a_bg_list[i], "gdn_a_bg_list", i, 2);
            const int64_t gc = gdn_tp();
            check_shape(gdn_A_log_list[i], {8 * gc}, "gdn_A_log_list", i);
            check_shape(gdn_dt_bias_list[i], {8 * gc}, "gdn_dt_bias_list", i);
            check_shape(gdn_norm_list[i], {gdn_value_head_dim},
                        "gdn_norm_list", i);
            check_shape(gdn_cq_list[i],
                        {gc, gdn_conv_kernel, gdn_key_dim / gc},
                        "gdn_cq_list", i);
            check_shape(gdn_ck_list[i],
                        {gc, gdn_conv_kernel, gdn_key_dim / gc},
                        "gdn_ck_list", i);
            check_shape(gdn_cv_list[i],
                        {gc, gdn_conv_kernel, gdn_value_dim / gc},
                        "gdn_cv_list", i);
            check_shape(gdn_b_bg_list[i], {16 * gc, hidden_size},
                        "gdn_b_bg_list", i);
            check_shape(gdn_a_bg_list[i], {16 * gc, hidden_size},
                        "gdn_a_bg_list", i);
        }
    }
    check_tensor(cos, "cos", -1, 2);
    check_tensor(sin, "sin", -1, 2);
    check_tensor(final_norm_w, "final_norm_w", -1, 1);
    TORCH_CHECK(cos.sizes() == sin.sizes(),
                "qwen3_5_moe set_weights: cos/sin shapes must match");
    TORCH_CHECK(cos.size(1) == 32 && final_norm_w.numel() == hidden_size,
                "qwen3_5_moe set_weights: RoPE/final norm geometry mismatch");
    TORCH_CHECK(expert_ids.defined() && expert_ids.device().type() == at::kPrivateUse1 &&
                    expert_ids.scalar_type() == at::kHalf &&
                    expert_ids.is_contiguous() && expert_ids.dim() == 2 &&
                    expert_ids.size(0) > 0 && expert_ids.size(1) == num_experts,
                "qwen3_5_moe set_weights: expert_ids must be contiguous FP16 RPU "
                "[max_chunk,num_experts]");
    TORCH_CHECK(token_ids.defined() && token_ids.device().type() == at::kPrivateUse1 &&
                    token_ids.scalar_type() == at::kShort &&
                    token_ids.is_contiguous() && token_ids.dim() == 2 &&
                    token_ids.size(0) == expert_ids.size(0) &&
                    token_ids.size(1) == top_k,
                "qwen3_5_moe set_weights: token_ids must be contiguous raw-U16 "
                "RPU [max_chunk,top_k] with the same max_chunk as expert_ids");

    at::Tensor next_router_bridge;
    if (num_cores() == 6) {
        // One stable model-owned DDR allocation is shared by every layer and
        // chunk. Graph nodes capture this owner once; only the active [T,E]
        // prefix is overwritten by the router4 -> route6 bridge.
        next_router_bridge = at::empty(
            {expert_ids.size(0), num_experts}, expert_ids.options());
        TORCH_CHECK(
            next_router_bridge.scalar_type() == at::kHalf &&
                next_router_bridge.device().type() == at::kPrivateUse1 &&
                next_router_bridge.is_contiguous(),
            "qwen3_5_moe set_weights: failed to allocate the TP6 router bridge");
    }

    // Bind the exact profile's development admission in the native owner.
    // Python retains the same (8192, 4096/top_k) bounds, but that state alone
    // cannot authorize the base planner. The route storage can only narrow
    // the chunk bound. The 8192-row bound describes physical prefill execution:
    // do not clip it to the static decode rotary table's logical capacity,
    // because prefill uses dynamic rotary tables and padded KV storage.
    // The public preflight checks the installed logical KV/rotary capacity.
    // Every candidate still passes the native SPM and
    // kernel predicates; this declaration is not a hardware certificate.
    // The cold-only check must precede any retained-weight mutation.
    set_chunk_envelope(
        /*max_kv_len=*/8192,
        std::min<int64_t>(4096 / top_k, expert_ids.size(0)));

    has_qk_norm_ = q_norm_list.size() > 0;
    use_silu_    = use_silu;
    eps_         = eps;

    layer_is_full_.assign(layer_is_full.begin(), layer_is_full.end());
    has_gdn_ = std::any_of(
        layer_is_full_.begin(), layer_is_full_.end(),
        [](uint8_t is_full) { return !is_full; });
    moe_ = next_moe;
    expert_ids_ = expert_ids;
    token_ids_ = token_ids;
    router_bridge_ = next_router_bridge;
    moe_layer_weights_.assign(N, MoeLayerWeights{});
    layer_weights_.assign(N, LayerWeights{});
    for (int64_t i = 0; i < N; ++i) {
        auto& lw = layer_weights_[i];
        auto& mw = moe_layer_weights_[i];
        // Every layer (full + GDN) has input/post norms and an independent MoE tail.
        lw.input_norm_w = input_norm_list[i];
        lw.post_norm_w = post_norm_list[i];
        mw.router_w = router_list[i];
        mw.routed_gate_w = routed_gate_list[i];
        mw.routed_up_w = routed_up_list[i];
        mw.routed_down_w = routed_down_list[i];
        mw.routed_gate_scale = routed_gate_scale_list[i];
        mw.routed_up_scale = routed_up_scale_list[i];
        mw.routed_down_scale = routed_down_scale_list[i];
        mw.routed_weight_mode = resolved_weight_mode;
        mw.shared_gate_w = shared_gate_list[i];
        mw.shared_up_w = shared_up_list[i];
        mw.shared_down_w = shared_down_list[i];
        mw.shared_scalar_gate_w = shared_scalar_gate_list[i];
        if (!layer_is_full_[i]) {
            // GDN mixer weights (attention slots stay empty).
            lw.gdn_in_z_w    = gdn_in_z_list[i];
            lw.gdn_out_w     = gdn_out_list[i];
            lw.gdn_in_z_scale = gdn_in_z_scale_list[i];
            lw.gdn_out_scale = gdn_out_scale_list[i];
            lw.gdn_A_log     = gdn_A_log_list[i];
            lw.gdn_dt_bias   = gdn_dt_bias_list[i];
            lw.gdn_norm_w    = gdn_norm_list[i];
            // prefill (chunk) per-path weights (separate q/k/v proj + conv, N_bg b/a)
            if (gdn_q_list.size() > 0) {
                lw.gdn_q_w = gdn_q_list[i]; lw.gdn_k_w = gdn_k_list[i]; lw.gdn_v_w = gdn_v_list[i];
                lw.gdn_q_scale = gdn_q_scale_list[i];
                lw.gdn_k_scale = gdn_k_scale_list[i];
                lw.gdn_v_scale = gdn_v_scale_list[i];
                lw.gdn_conv_q_w = gdn_cq_list[i]; lw.gdn_conv_k_w = gdn_ck_list[i]; lw.gdn_conv_v_w = gdn_cv_list[i];
                lw.gdn_b_bg_w = gdn_b_bg_list[i]; lw.gdn_a_bg_w = gdn_a_bg_list[i];
            }
            continue;
        }
        lw.q_w = q_w_list[i]; lw.k_w = k_w_list[i];
        lw.v_w = v_w_list[i]; lw.o_w = o_w_list[i];
        lw.q_scale = q_scale_list[i]; lw.k_scale = k_scale_list[i];
        lw.v_scale = v_scale_list[i]; lw.o_scale = o_scale_list[i];
        lw.q_norm_w    = has_qk_norm_ ? q_norm_list[i] : at::Tensor();
        lw.k_norm_w    = has_qk_norm_ ? k_norm_list[i] : at::Tensor();
        lw.attn_gate_w = attn_gate_list[i];
        lw.attn_gate_scale = attn_gate_scale_list[i];
    }
    cos_ = keep_256b_aligned_rpu_copy(cos, "qwen3_5_moe set_weights: cos");
    sin_ = keep_256b_aligned_rpu_copy(sin, "qwen3_5_moe set_weights: sin");
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

    // FusedModelBase retains one intermediate dimension for generic planning;
    // the sparse model binds it to the routed expert width and never emits the
    // dense MLP pipeline.
    set_model_params(num_q_heads, num_kv_heads, head_dim, hidden_size,
                     routed_intermediate);
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

c10::optional<at::Tensor> Qwen3_5MoeModel::optional_scale(
    const at::Tensor& scale) const {
    if (!scale.defined() || scale.numel() == 0) return c10::nullopt;
    return scale;
}

void Qwen3_5MoeModel::emit_router_and_shuffle(
    int layer_idx, int64_t invocation, int64_t t, uint32_t input_spm_addr) {
    const auto& lw = moe_layer_weights_.at(layer_idx);
    const int64_t e = moe_.num_experts;
    const int64_t k = moe_.top_k;
    const int64_t r = t * k;
    const int router_nc = router_tp();
    const int route_nc = routing_tp();

    // Router col-linear -> full logits -> probabilities.
    launch_linear(
        invocation,
        input_spm_addr, lw.router_w, addr(0, "moe_router_local"),
        t, e, hidden_size(), /*partition=col=*/1, router_nc,
        /*bias_spm_addr=*/0, {}, /*mlp_slice=*/true);
    const RpuAllGatherSchedule gather_schedule =
        rpu_resolve_all_gather_schedule(e / router_nc, sizeof(c10::Half));
    if (ctx().has_complete_physical_manifest()) {
        ctx().consume_physical_route(
            FmbRouteFamily::COLLECTIVE, QWEN35_MOE_ROUTER_ALL_GATHER_SITE,
            static_cast<int64_t>(gather_schedule), /*resolved_flags=*/0,
            {t, e / router_nc, static_cast<int64_t>(sizeof(c10::Half)),
             router_nc},
            invocation);
    }
    rpu_launch_all_gather_spm_kernel(
        addr(0, "moe_router_local"), addr(0, "moe_router_logits"),
        t, e / router_nc, sizeof(c10::Half), router_nc,
        gather_schedule);

    if (router_nc != route_nc) {
        TORCH_INTERNAL_ASSERT(
            num_cores() == 6 && router_nc == 4 && route_nc == 6);
        TORCH_CHECK(
            router_bridge_.defined() && router_bridge_.dim() == 2 &&
                router_bridge_.size(0) >= t && router_bridge_.size(1) == e,
            "qwen3_5_moe: TP6 router bridge capacity is not installed");
        const std::vector<int64_t> bridge_arguments{
            t, e, router_nc, route_nc, router_bridge_.size(0)};
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_MOE_ROUTER_BRIDGE_STORE_SITE,
                qwen35_moe_dma_route(
                    Qwen35MoeMutableDmaRoute::SPM_CORE0_TO_DDR),
                /*resolved_flags=*/0, bridge_arguments, invocation);
        }
        rpu_launch_spm_copy_ddr_dma(
            addr(0, "moe_router_logits"), router_bridge_,
            /*dst_offset_elements=*/0, t * e);
        if (ctx().has_complete_physical_manifest()) {
            ctx().consume_physical_route(
                FmbRouteFamily::MUTABLE_DMA,
                QWEN35_MOE_ROUTER_BRIDGE_LOAD_SITE,
                qwen35_moe_dma_route(
                    Qwen35MoeMutableDmaRoute::DDR_BROADCAST_TO_SPM),
                /*resolved_flags=*/0, bridge_arguments, invocation);
        }
        rpu_launch_ddr_broadcast_spm_dma(
            router_bridge_, /*src_offset_elements=*/0, t * e,
            addr(0, "moe_route_logits"), route_nc);
    }
    const uint32_t route_logits = router_nc == route_nc
        ? addr(0, "moe_router_logits")
        : addr(0, "moe_route_logits");
    rpu_launch_softmax_c16_spm_kernel(
        route_logits, addr(0, "moe_router_probs"),
        t, e, route_nc);

    // Step 1 shared by prefill/decode: select top-K scores and carry FP16
    // expert IDs.  cvt_key=false matches runtime's no_cvt_key=false call.
    rpu_launch_moe_topk_scalar_fp16_spm_kernel(
        addr(0, "moe_router_probs"), addr(0, "moe_expert_ids"),
        addr(0, "moe_route_scores"), addr(0, "moe_route_expert_ids"),
        t, e, k, /*descending=*/true, /*cvt_key=*/false, route_nc);

    if (t == 1) {
        // Runtime decode step 2: sort all K selected expert keys ascending,
        // carrying scores in-place.  token IDs remain the persistent zeros.
        rpu_launch_moe_topk_scalar_fp16_spm_kernel(
            addr(0, "moe_route_expert_ids"), addr(0, "moe_route_scores"),
            addr(0, "moe_route_expert_ids"), addr(0, "moe_route_scores"),
            1, k, k, /*descending=*/false, /*cvt_key=*/false, route_nc);
    } else {
        // Runtime prefill step 2: sort all R=T*K tasks by expert ID and carry
        // the static token ID.  R>10 for every supported prefill chunk.
        rpu_launch_moe_topk_by_kvsort_fp16_spm_kernel(
            addr(0, "moe_route_expert_ids"), addr(0, "moe_token_ids"),
            addr(0, "moe_route_expert_ids"),
            addr(0, "moe_route_token_ids"),
            /*workspace_key=*/0, /*workspace_value=*/0,
            1, r, r, /*descending=*/false, route_nc);
    }

    // Runtime step 3: sorted FP16 expert keys -> in-place U16 IDs + m_sizes.
    rpu_launch_moe_bincount_scalar_spm_kernel(
        addr(0, "moe_route_expert_ids"),
        addr(0, "moe_route_expert_ids"), addr(0, "moe_m_sizes"),
        r, e, /*cvt_key=*/true, /*output_cvt_key=*/true, route_nc);

    if (t == 1) {
        // Decode token IDs are all zero and do not participate in the sort.
        rpu_launch_fill_spm_kernel(
            addr(0, "moe_route_token_ids"), k, c10::Half(0.0f), route_nc);
    }

    // Sum of top-K probabilities is order invariant.  Decode carries the
    // runtime-sorted scores in-place; prefill keeps the original TopK order.
    rpu_launch_moe_topk_prob_norm_fp16_spm_kernel(
        addr(0, "moe_router_probs"), addr(0, "moe_route_scores"),
        addr(0, "moe_router_probs"), t, e, k, route_nc);
}

void Qwen3_5MoeModel::emit_shared_expert(
    int layer_idx, int64_t invocation, int64_t t, uint32_t input_spm_addr) {
    const auto& lw = moe_layer_weights_.at(layer_idx);
    const int64_t h = hidden_size();
    const int64_t si = physical_shared_intermediate();
    const int nc = shared_expert_tp();

    launch_linear(
        invocation,
        input_spm_addr, lw.shared_gate_w,
        addr(0, "moe_shared_gate"), t, si, h,
        /*partition=col=*/1, nc,
        /*bias_spm_addr=*/0, {}, /*mlp_slice=*/true);
    launch_linear(
        invocation,
        input_spm_addr, lw.shared_up_w,
        addr(0, "moe_shared_up"), t, si, h,
        /*partition=col=*/1, nc,
        /*bias_spm_addr=*/0, {}, /*mlp_slice=*/true);
    consume_manifest_route(
        FmbRouteFamily::ACTIVATION, QWEN35_MOE_SHARED_SILU_SITE,
        static_cast<int64_t>(Qwen35MoeActivationRoute::SILU_MUL),
        invocation * 4 + 1);
    rpu_launch_silu_mul_spm_kernel(
        addr(0, "moe_shared_gate"), addr(0, "moe_shared_up"),
        addr(0, "moe_shared_act"), t * (si / nc), nc);
    launch_linear(
        invocation,
        addr(0, "moe_shared_act"), lw.shared_down_w,
        addr(0, "moe_partial"), t, h, si,
        /*partition=row=*/0, nc,
        /*bias_spm_addr=*/0, {}, /*mlp_slice=*/true);

    // Ordinary col-parallel Linear requires local_n=16 alignment.  Weight is
    // [16*tp,H] with identical rows, so each core emits [T,16] identical
    // scalars; slice lane 0 to the contiguous [T] broadcast vector.
    launch_linear(
        invocation,
        input_spm_addr, lw.shared_scalar_gate_w,
        addr(0, "moe_shared_alpha_pad"), t, 16 * nc, h,
        /*partition=col=*/1, nc,
        /*bias_spm_addr=*/0, {}, /*mlp_slice=*/true);
    for (int core = 0; core < nc; ++core) {
        rpu_launch_slice_spm_kernel(
            addr(core, "moe_shared_alpha_pad"),
            addr(core, "moe_shared_alpha"),
            {t, 16}, {t, 1}, {0, 0}, /*num_cores=*/1);
    }
    consume_manifest_route(
        FmbRouteFamily::ACTIVATION, QWEN35_MOE_SHARED_SIGMOID_SITE,
        static_cast<int64_t>(Qwen35MoeActivationRoute::SIGMOID),
        invocation * 4 + 2);
    rpu_launch_eltwise_unary_spm_kernel(
        addr(0, "moe_shared_alpha"), addr(0, "moe_shared_alpha"), t,
        ValuOpType::SIGMOID, GeluMode::NONE, nc);
    rpu_launch_eltwise_binary_Nx1_NxC_spm_kernel(
        addr(0, "moe_shared_alpha"), addr(0, "moe_partial"),
        addr(0, "moe_partial"), t, h, c10::Half(1.0), ValuOpType::MUL,
        /*is_bopa=*/false, nc);
}

void Qwen3_5MoeModel::emit_routed_experts(
    int layer_idx, int64_t invocation, int64_t t, uint32_t input_spm_addr) {
    const auto& lw = moe_layer_weights_.at(layer_idx);
    const int64_t h = hidden_size();
    const int64_t e = moe_.num_experts;
    const int64_t r = t * moe_.top_k;
    const int64_t local_i =
        physical_routed_intermediate() / routed_expert_tp();
    const int nc = routed_expert_tp();

    rpu_launch_cast_uint16_int32_spm_kernel(
        addr(0, "moe_route_token_ids"), addr(0, "moe_route_indices_i32"),
        r, nc);
    rpu_launch_gather_fp16_spm_kernel(
        input_spm_addr, addr(0, "moe_route_indices_i32"),
        addr(0, "moe_shuffled_input"), t, h, /*axis=*/0, r, nc);

    launch_grouped_linear(
        invocation,
        addr(0, "moe_shuffled_input"), lw.routed_gate_w,
        addr(0, "moe_m_sizes"), lw.routed_gate_scale,
        addr(0, "moe_routed_gate"),
        local_i, h, e, /*is_col_parallel=*/true, nc,
        lw.routed_weight_mode);
    launch_grouped_linear(
        invocation,
        addr(0, "moe_shuffled_input"), lw.routed_up_w,
        addr(0, "moe_m_sizes"), lw.routed_up_scale,
        addr(0, "moe_routed_up"),
        local_i, h, e, /*is_col_parallel=*/true, nc,
        lw.routed_weight_mode);
    consume_manifest_route(
        FmbRouteFamily::ACTIVATION, QWEN35_MOE_ROUTED_SILU_SITE,
        static_cast<int64_t>(Qwen35MoeActivationRoute::SILU_MUL),
        invocation * 4 + 1);
    rpu_launch_silu_mul_spm_kernel(
        addr(0, "moe_routed_gate"), addr(0, "moe_routed_up"),
        addr(0, "moe_routed_act"), r * local_i, nc);
    launch_grouped_linear(
        invocation,
        addr(0, "moe_routed_act"), lw.routed_down_w,
        addr(0, "moe_m_sizes"), lw.routed_down_scale,
        addr(0, "moe_routed_output"),
        h, local_i, e, /*is_col_parallel=*/false, nc,
        lw.routed_weight_mode);
    rpu_launch_moe_gather_mul_v2_fp16_spm_kernel(
        addr(0, "moe_routed_output"), addr(0, "moe_router_probs"),
        addr(0, "moe_route_token_ids"), addr(0, "moe_route_expert_ids"),
        addr(0, "moe_routed_output"), t, e, r, h, nc);
}

void Qwen3_5MoeModel::emit_moe_merge_and_residual(
    int64_t invocation, int64_t t,
    uint32_t residual_spm_addr, uint32_t output_spm_addr) {
    const int64_t h = hidden_size();
    const int64_t r = t * moe_.top_k;
    const int nc = routing_tp();

    // moe_partial already contains the gated shared-expert partial.
    rpu_launch_scatternd_add_u16_fp16_spm_kernel(
        addr(0, "moe_partial"), addr(0, "moe_route_token_ids"),
        addr(0, "moe_routed_output"), t, r, h, nc);
    consume_manifest_route(
        FmbRouteFamily::ALL_REDUCE, QWEN35_MOE_ALL_REDUCE_SITE,
        text_ring_route(t, h), invocation);
    rpu_launch_all_reduce_sum_residual_kernel(
        addr(0, "moe_partial"), residual_spm_addr,
        output_spm_addr, t, h, nc, num_cores());
}

}  // namespace v3

// =============================================================================
// Instance registry + C API for TORCH_LIBRARY_IMPL (registered in rpu_backend.cpp)
// =============================================================================
using Qwen3_5MoeRegistry = ModelHandleRegistry<v3::Qwen3_5MoeModel>;

std::vector<int64_t> rpu_qwen3_5_moe_planner_cache_identity(int64_t handle) {
    return Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_planner_cache_identity")
        ->planner_cache_identity();
}

void rpu_qwen3_5_moe_bind_kvinsert_costs(
        int64_t handle, at::IntArrayRef identity,
        const std::string& catalog_sha256, at::IntArrayRef certificate_rows) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_bind_kvinsert_costs")
        ->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows);
}

std::string rpu_qwen3_5_moe_kvinsert_cost_catalog_sha256(int64_t handle) {
    return Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_kvinsert_cost_catalog_sha256")
        ->kvinsert_cost_catalog_sha256();
}

std::tuple<std::vector<int64_t>, int64_t, int64_t>
rpu_qwen3_5_moe_kvinsert_exact_candidate(
        int64_t handle, at::IntArrayRef descriptor, int64_t site_id,
        int64_t invocation, int64_t route) {
    return Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_kvinsert_exact_candidate")
        ->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route);
}

KvInsertCostDomainQuery rpu_qwen3_5_moe_kvinsert_cost_domain(
        int64_t handle, at::IntArrayRef descriptor) {
    return Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_kvinsert_cost_domain")
        ->kvinsert_cost_domain("qwen3_5_moe", descriptor);
}

int64_t rpu_qwen3_5_moe_create() { return Qwen3_5MoeRegistry::create(); }

void rpu_qwen3_5_moe_set_execution_cores(
        int64_t handle, int64_t num_cores) {
    Qwen3_5MoeRegistry::get(
        handle, "rpu_qwen3_5_moe_set_execution_cores")
        ->set_execution_cores(num_cores);
}

std::vector<int64_t> rpu_qwen3_5_moe_get_execution_topology(
        int64_t handle) {
    return Qwen3_5MoeRegistry::get(
        handle, "rpu_qwen3_5_moe_get_execution_topology")
        ->execution_topology();
}

std::vector<int64_t> rpu_qwen3_5_moe_get_execution_topology_v2(
        int64_t handle) {
    return Qwen3_5MoeRegistry::get(
        handle, "rpu_qwen3_5_moe_get_execution_topology_v2")
        ->execution_topology_v2();
}

void    rpu_qwen3_5_moe_destroy(int64_t handle) {
    Qwen3_5MoeRegistry::destroy(handle, "rpu_qwen3_5_moe_destroy");
}

void rpu_qwen3_5_moe_prepare_persistent_spm(int64_t handle, int64_t execution_len) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_prepare_persistent_spm")
        ->prepare_persistent_spm(execution_len, 0, true);
}

// Per-handle SPM-budget-resolved prefill chunk_size (0 before any forward).
// Read from Python after a prefill forward to observe the multi-chunk split.
//
// 返回的是 **PREFILL** 的 cs。基类的 last_resolved_chunk_size_ 每次 compute_chunks 都覆写,
// 而 decode(seq_len=1)必然跑在 prefill 之后且固定选 cs=16 —— 直接读基类那份的话,凡是
// "prefill + decode 都跑过才来读"的调用方(如 perf harness 在 warmup 之后读)拿到的永远是
// 16,与 prefill 实际用的值无关。这里优先返回 forward 里单独记下的 prefill 那份;
// 尚未跑过 prefill 时(=0)退回基类值,保持只跑 prefill 的老调用方(parity 测试)行为不变。
int64_t rpu_qwen3_5_moe_get_resolved_chunk_size(int64_t handle) {
    auto* m = Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_get_resolved_chunk_size");
    const int64_t prefill_cs = m->last_prefill_chunk_size();
    return prefill_cs > 0 ? prefill_cs : m->get_last_resolved_chunk_size();
}

void rpu_qwen3_5_moe_set_chunk_size_cap(int64_t handle, int64_t cap) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_set_chunk_size_cap")
        ->set_chunk_size_cap(cap);
}

void rpu_qwen3_5_moe_set_prefill_chunk_size(int64_t handle, int64_t chunk_size) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_set_prefill_chunk_size")
        ->set_prefill_chunk_size(chunk_size);
}

void rpu_qwen3_5_moe_set_chunk_envelope(int64_t handle, int64_t max_kv_len,
                                    int64_t chunk) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_set_chunk_envelope")
        ->set_chunk_envelope(max_kv_len, chunk);
}

void rpu_qwen3_5_moe_set_linear_acc32(int64_t handle, bool enabled) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_set_linear_acc32")
        ->set_linear_acc32(enabled);
}

void rpu_qwen3_5_moe_set_fast_replay(int64_t handle, bool enabled) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_set_fast_replay")
        ->set_fast_replay(enabled);
}

void rpu_qwen3_5_moe_set_retained_prefill_graph(
        int64_t handle, bool enabled) {
    Qwen3_5MoeRegistry::get(
        handle, "rpu_qwen3_5_moe_set_retained_prefill_graph")
        ->set_retained_prefill_graph(enabled);
}

int64_t rpu_qwen3_5_moe_resolve_prefill_chunk_size(
    int64_t handle, int64_t execution_len) {
    return Qwen3_5MoeRegistry::get(
        handle, "rpu_qwen3_5_moe_resolve_prefill_chunk_size")
        ->resolve_prefill_chunk_size(execution_len);
}

std::vector<int64_t> rpu_qwen3_5_moe_resolve_prefill_stage_domain(
    int64_t handle, int64_t execution_len, int64_t logical_len,
    int64_t planning_chunk_size_override) {
    return Qwen3_5MoeRegistry::get(
        handle, "rpu_qwen3_5_moe_resolve_prefill_stage_domain")
        ->resolve_prefill_stage_domain(
            execution_len, logical_len, planning_chunk_size_override);
}

std::vector<int64_t> rpu_qwen3_5_moe_resolve_decode_stage_descriptor(
        int64_t handle) {
    return Qwen3_5MoeRegistry::get(
        handle, "rpu_qwen3_5_moe_resolve_decode_stage_descriptor")
        ->resolve_decode_stage_descriptor();
}

void rpu_qwen3_5_moe_set_weights(
    int64_t handle,
    at::TensorList q, at::TensorList k, at::TensorList v, at::TensorList o,
    at::TensorList qn, at::TensorList kn, at::TensorList ag,
    at::TensorList packed_scale,
    at::TensorList in_norm, at::TensorList post_norm,
    at::TensorList router,
    at::TensorList routed_gate, at::TensorList routed_up,
    at::TensorList routed_down,
    at::TensorList shared_gate, at::TensorList shared_up,
    at::TensorList shared_down, at::TensorList shared_scalar_gate,
    const at::Tensor& cos, const at::Tensor& sin, const at::Tensor& final_norm,
    const at::Tensor& expert_ids, const at::Tensor& token_ids,
    at::IntArrayRef layer_is_full,
    int64_t nqh, int64_t nkvh, int64_t hd, int64_t hs,
    int64_t num_experts, int64_t top_k, int64_t routed_intermediate,
    int64_t shared_intermediate, int64_t tensor_parallel,
    int64_t routed_weight_mode,
    double eps, bool use_silu,
    at::IntArrayRef mrope_section,
    at::TensorList gdn_in_z, at::TensorList gdn_out,
    at::TensorList gdn_A_log,
    at::TensorList gdn_dt_bias, at::TensorList gdn_norm,
    int64_t gdn_nvh, int64_t gdn_dk, int64_t gdn_dv, int64_t gdn_conv_dim,
    int64_t gdn_conv_kernel,
    at::TensorList gdn_q, at::TensorList gdn_k, at::TensorList gdn_v,
    at::TensorList gdn_cq, at::TensorList gdn_ck, at::TensorList gdn_cv,
    at::TensorList gdn_b_bg, at::TensorList gdn_a_bg) {
    // Fixed group-major scale layout supplied by the Python adapter:
    // q,k,v,o,attn_gate,routed_gate,routed_up,routed_down,
    // gdn_in_z,gdn_out,gdn_q,gdn_k,gdn_v. Each group has N entries.
    constexpr int64_t kScaleGroups = 13;
    const int64_t n = static_cast<int64_t>(layer_is_full.size());
    TORCH_CHECK(
        n > 0 && static_cast<int64_t>(packed_scale.size()) == kScaleGroups * n,
        "qwen3_5_moe_set_weights: packed_scale must contain ",
        kScaleGroups, " groups x ", n, " layers, got ", packed_scale.size());
    const auto scale_group = [&](int64_t group) {
        return packed_scale.slice(
            static_cast<size_t>(group * n), static_cast<size_t>(n));
    };
    auto* m = Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_set_weights");
    m->set_weights(
                   q, k, v, o, qn, kn, ag,
                   scale_group(0), scale_group(1), scale_group(2),
                   scale_group(3), scale_group(4),
                   in_norm, post_norm,
                   router, routed_gate, routed_up, routed_down,
                   scale_group(5), scale_group(6), scale_group(7),
                   shared_gate, shared_up, shared_down, shared_scalar_gate,
                   cos, sin, final_norm, expert_ids, token_ids, layer_is_full,
                   nqh, nkvh, hd, hs, num_experts, top_k,
                   routed_intermediate, shared_intermediate, tensor_parallel,
                   routed_weight_mode, eps, use_silu, mrope_section,
                   gdn_in_z, gdn_out, scale_group(8), scale_group(9),
                   gdn_A_log, gdn_dt_bias, gdn_norm,
                   gdn_nvh, gdn_dk, gdn_dv, gdn_conv_dim, gdn_conv_kernel,
                   gdn_q, gdn_k, gdn_v,
                   scale_group(10), scale_group(11), scale_group(12),
                   gdn_cq, gdn_ck, gdn_cv, gdn_b_bg, gdn_a_bg);
}

void rpu_qwen3_5_moe_set_prefill_rope(int64_t handle, const at::Tensor& cos, const at::Tensor& sin) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_set_prefill_rope")->set_prefill_rope(cos, sin);
}

void rpu_qwen3_5_moe_set_valid_prefill_len(int64_t handle, int64_t n) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_set_valid_prefill_len")->set_valid_prefill_len(n);
}

void rpu_qwen3_5_moe_set_mrope_position_delta(int64_t handle, int64_t d) {
    Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_set_mrope_position_delta")->set_mrope_position_delta(d);
}

at::Tensor rpu_qwen3_5_moe_forward(
    int64_t handle, const at::Tensor& hidden,
    at::TensorList k_caches, at::TensorList v_caches, at::TensorList gdn_states,
    at::TensorList conv_states,
    const std::optional<at::Tensor>& mask, int64_t position, bool is_causal,
    int64_t planned_chunk_size,
    at::IntArrayRef planned_stage_descriptor) {
    auto* m = Qwen3_5MoeRegistry::get(handle, "rpu_qwen3_5_moe_forward");
    std::vector<at::Tensor> kc(k_caches.begin(), k_caches.end());
    std::vector<at::Tensor> vc(v_caches.begin(), v_caches.end());
    std::vector<at::Tensor> gs(gdn_states.begin(), gdn_states.end());
    std::vector<at::Tensor> cs(conv_states.begin(), conv_states.end());
    return m->forward(
        hidden, kc, vc, gs, cs, mask, position, is_causal,
        planned_chunk_size, planned_stage_descriptor);
}
