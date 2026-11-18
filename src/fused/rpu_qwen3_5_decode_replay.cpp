// Bounded owner-local replay of the exact Qwen3.5 0.8B text decode body.
// The contract is the one in rpu_qwen3vl_decode_replay.cpp: the ordinary forward
// and the Graph runtime remain the authority for allocation, dynamic DMA and
// COMPLETE routes. Only proven static emission is omitted, and the Graph patch
// API refreshes every position-dependent kernel field.
//
// QWEN35_DECODE_REPLAY_FIXED_KERNEL_BASIS: the candidate head's hidden-row
// broadcast and per-core output scatter are fixed transport DMA. Its Linear is
// a typed launcher and stays a separately registered planner site.
#include "rpu_qwen3_5_model.h"

#include "rpu_helpers.h"
#include "rpu_ops.h"
#include "rpu_profile.h"
#include "rpu_runtime_state.h"
#include "rpu_spm_buffers.h"
#include "rpu_spm_pipeline.h"
#include "rpu_spm_residency.h"
#include "rpu_linear_tiling.h"

#include <algorithm>
#include <array>
#include <optional>
#include <set>
#include <string>
#include <typeinfo>

using namespace at;
using namespace ::rhino_lkn;

// Uses the unqualified runtime names the directives above bring into scope.
#include "rpu_decode_replay_common.h"

namespace v3 {
namespace {

using decode_replay::DecodeGraphKey;
using decode_replay::DecodeOwner;
using decode_replay::graph_key;
using decode_replay::tensor_shape;

struct DecodeKernelSlot { size_t kernel, node; std::string name; bool all_cores; };

// Names whose registers hold only cold geometry, SPM offsets and fixed owner
// addresses at seq_len 1. Any other name keeps the ordinary body.
bool static_decode_kernel(const std::string& name) {
    static const std::set<std::string> names{
        "llama_rms_norm", "llm_fla_conv1d_w16a16_acc16", "unary", "unary_softplus",
        "binary_sameshape", "binary_scalar", "binary_Nx1_NxC_v16_batch", "l2norm",
        "qwen3_5_mul_reduce_rows", "qwen3_5_rank1_fma", "qwen3_5_rms_norm_gated",
        "llm_all_reduce_residual_nopace"};
    if (names.count(name)) return true;
    // Ordinary FP16/W8 tiles only; grouped and W4 cannot satisfy this by prefix.
    using rpu_pl_tiling::WeightType;
    for (const auto& candidate : rpu_pl_tiling::detail::parallel_candidates) {
        if ((candidate.weight == WeightType::Fp16 || candidate.weight == WeightType::Int8) &&
            name == candidate.name) return true;
    }
    return false;
}

// Per full layer, in stream order: Q RoPE, K norm, K RoPE, K insert, V insert,
// attention. The registers below are every field that differs between steps,
// enumerated by interposing the launch library's setters on real requests.
constexpr size_t kPatchesPerFullLayer = 16;

bool bind_decode_patches(const std::vector<DecodeKernelSlot>& kernels, int64_t full_layers,
                         std::vector<RegisterPatch>& patches) {
    std::set<size_t> indices;
    for (size_t i = 0; i < kernels.size(); ++i) {
        if (!indices.insert(kernels[i].kernel).second ||
            (i && kernels[i - 1].node >= kernels[i].node)) return false;
    }
    patches.clear();
    patches.reserve(full_layers * kPatchesPerFullLayer);
    int phase = 0;
    int64_t completed = 0;
    const auto add = [&](const DecodeKernelSlot& slot, std::initializer_list<uint32_t> regs) {
        for (uint32_t reg : regs) patches.push_back({slot.kernel, reg, 0, 2});
    };
    for (const auto& slot : kernels) {
        if (slot.name == "partial_rope_1d") {
            if (phase > 1 || !slot.all_cores) return false;
            add(slot, {8u, 9u});
            ++phase;
        } else if (slot.name == "llama_insert_kcache_multiwarp") {
            if (phase != 2 || !slot.all_cores) return false;
            add(slot, {0u, 1u});
            ++phase;
        } else if (slot.name == "llama_insert_vcache_multiwarp") {
            if (phase != 3 || !slot.all_cores) return false;
            add(slot, {0u, 1u});
            ++phase;
        } else if (slot.name == "llm_fp16_32b_prefill_flash_attn_univ_dp") {
            if (phase != 4 || !slot.all_cores) return false;
            add(slot, {0u, 1u, 4u, 5u, 16u, 17u, 29u, 31u});
            phase = 0;
            ++completed;
        } else if (phase > 1 || !static_decode_kernel(slot.name)) {
            return false;
        }
    }
    return phase == 0 && completed == full_layers &&
        patches.size() == size_t(full_layers * kPatchesPerFullLayer);
}

void update_decode_patches(std::vector<RegisterPatch>& patches, int64_t position,
                           int64_t rope_position) {
    TORCH_CHECK(!patches.empty() && patches.size() % kPatchesPerFullLayer == 0 &&
                    position >= 0 && rope_position >= 0,
                "unsealed Qwen3.5 decode patch recipe");
    const uint32_t p = position, s = p + 1, rope = rope_position;
    const std::array<uint16_t, kPatchesPerFullLayer> values{
        uint16_t(rope), uint16_t(rope >> 16), uint16_t(rope), uint16_t(rope >> 16),
        uint16_t(p), uint16_t(p >> 16), uint16_t(p), uint16_t(p >> 16),
        uint16_t(s), uint16_t(s >> 16), uint16_t((s + 15) / 16), uint16_t((p + 15) / 16),
        uint16_t(p), uint16_t(p >> 16), uint16_t((s + 15) / 16), uint16_t((s + 15) / 16)};
    for (size_t i = 0; i < patches.size(); ++i)
        patches[i].value = values[i % kPatchesPerFullLayer];
}

// One [1, blocks, ...] physical layout on every full layer; GDN slots stay empty.
std::optional<int64_t> decode_cache_capacity(at::TensorList k, at::TensorList v,
                                             const std::vector<uint8_t>& is_full) {
    if (k.size() != is_full.size() || v.size() != is_full.size()) return std::nullopt;
    std::optional<std::vector<int64_t>> shape;
    for (size_t layer = 0; layer < is_full.size(); ++layer) {
        if (!is_full[layer]) continue;
        for (const at::Tensor* t : {&k[layer], &v[layer]}) {
            if (!t->defined() || t->device().type() != at::kPrivateUse1 ||
                t->scalar_type() != at::kHalf || t->layout() != c10::Layout::Strided ||
                !t->is_contiguous() || t->dim() != 7 || t->size(0) != 1 || t->size(1) <= 0)
                return std::nullopt;
            if (shape && !t->sizes().equals(*shape)) return std::nullopt;
            if (!shape) shape = t->sizes().vec();
        }
    }
    if (!shape) return std::nullopt;
    return (*shape)[1] * 16;
}

} // namespace

struct Qwen3_5Model::CheckedDecodeReplayState {
    struct Recipe {
        DecodeGraphKey key;
        GraphOpStreamStamp begin{}, body{}, end{};
        uint64_t model_generation = 0, topology = 0, layout = 0;
        std::vector<int64_t> descriptor;
        std::vector<DecodeOwner> owners;
        std::vector<RegisterPatch> patches;
        std::array<uint32_t, 8> bases{};
        // Persistent norms are not part of the temporary COMPLETE layout hash.
        std::vector<uint32_t> norms;
        size_t temporary = 0;
        int64_t cache_capacity = 0;
    };
    std::unique_ptr<Recipe> recipe;
    bool active = false, rebind = false, body_seen = false, skipped = false;
    GraphOpStreamStamp begin{}, body{};
    DecodeGraphKey key;
    uint64_t generation = 0, layout = 0;
    int64_t position = 0, cache_capacity = 0;
    at::TensorList k, v;
    at::IntArrayRef descriptor;
    // Diagnostics: bound recipes, skipped bodies, ordinary bodies inside the scope,
    // and the step at which the last rejected binding gave up (0 = none).
    int64_t binds = 0, skips = 0, ordinary = 0, reject = 0;

    // Every DDR operand the skipped stream baked. Unused slots are undefined or
    // empty by the installer's contract. Recurrent state and KV change content
    // through the device, so only their identity is frozen.
    template<class F> static bool visit_owners(Qwen3_5Model& model, at::TensorList k,
                                               at::TensorList v, F fn) {
        bool ok = true;
        const auto visit = [&](const at::Tensor& t, bool fixed) {
            if (ok && t.defined() && t.numel() > 0) ok = fn(t, fixed);
        };
        for (int layer = 0; layer < model.num_layers(); ++layer) {
            const auto& w = model.layer_weights_[layer];
            for (const auto* t : {
                    &w.input_norm_w, &w.post_norm_w, &w.gate_w, &w.up_w, &w.down_w,
                    &w.gate_ws, &w.up_ws, &w.down_ws, &w.gate_negative_zero,
                    &w.up_negative_zero, &w.down_negative_zero, &w.q_w, &w.k_w, &w.v_w,
                    &w.o_w, &w.q_ws, &w.k_ws, &w.v_ws, &w.o_ws, &w.attn_gate_w,
                    &w.attn_gate_ws, &w.q_norm_w, &w.k_norm_w, &w.gdn_in_z_w, &w.gdn_out_w,
                    &w.gdn_A_log, &w.gdn_dt_bias, &w.gdn_norm_w, &w.gdn_q_w, &w.gdn_k_w,
                    &w.gdn_v_w, &w.gdn_conv_q_w, &w.gdn_conv_k_w, &w.gdn_conv_v_w,
                    &w.gdn_b_bg_w, &w.gdn_a_bg_w, &w.gdn_in_z_ws, &w.gdn_out_ws, &w.gdn_q_ws,
                    &w.gdn_k_ws, &w.gdn_v_ws, &w.gdn_b_bg_ws, &w.gdn_a_bg_ws})
                visit(*t, true);
            if (model.layer_is_full_[layer]) {
                visit(k[layer], false);
                visit(v[layer], false);
            } else {
                visit(model.gdn_neg_exp_A_[layer], true);
                const auto* state = model.gdn_state_for_layer(layer);
                const auto* conv = model.conv_state_for_layer(layer);
                if (!state || !conv || !state->defined() || !conv->defined()) return false;
                visit(*state, false);
                visit(*conv, false);
            }
        }
        for (const auto* t : {&model.cos_, &model.sin_, &model.final_norm_w_, &model.gdn_zero_,
                              &model.decode_head_w_})
            visit(*t, true);
        visit(model.decode_head_out_, false);
        return ok;
    }

    bool owners_match(Qwen3_5Model& model) const {
        size_t index = 0;
        return visit_owners(model, k, v, [&](const at::Tensor& t, bool) {
            return index < recipe->owners.size() && recipe->owners[index++].matches(t);
        }) && index == recipe->owners.size();
    }

    static std::vector<uint32_t> norm_addresses(Qwen3_5Model& model) {
        std::vector<uint32_t> result;
        result.reserve(model.num_layers() * 4 + 1);
        for (int layer = 0; layer < model.num_layers(); ++layer) {
            result.push_back(model.layer_addr(layer, 0, "norm_w"));
            result.push_back(model.layer_addr(layer, 0, "post_norm_w"));
            if (model.layer_is_full_[layer]) {
                result.push_back(model.layer_addr(layer, 0, "q_norm_w"));
                result.push_back(model.layer_addr(layer, 0, "k_norm_w"));
            }
        }
        result.push_back(model.addr(0, "final_norm_w"));
        return result;
    }
};

bool Qwen3_5Model::begin_checked_decode_replay(
    const at::Tensor& hidden, at::TensorList k, at::TensorList v,
    const std::optional<at::Tensor>& mask, int64_t position, bool is_causal,
    at::IntArrayRef descriptor) {
    if (typeid(*this) != typeid(Qwen3_5Model) || !RpuKernelGraph::has_active() ||
        // The exact Qwen3.5 0.8B text profile, the only one whose kernel stream and
        // dynamic fields were enumerated on hardware.
        num_layers() != 24 || hidden_size() != 1024 || intermediate_size() != 3584 ||
        num_q_heads() != 8 || head_dim() != 256 || !has_gdn_ || !has_mrope_ || !has_qk_norm_ ||
        num_cores() != 8 || attn_tp() != 8 || mlp_tp() != 8 ||
        sdpa_kernel_ != SdpaKernelType::FLASH_ATTN_SPM ||
        action_mode_ || action_step_active_ || action_nvfp4_.enabled() || z2_bound_ ||
        high_precision_ || get_debug_export() || fast_replay_active_ ||
        // The skipped launchers' gated flushes are not replicated.
        g_rpu_ddr_flush_enabled ||
        !is_causal || mask.has_value() || descriptor.empty() || position <= 0 ||
        layer_weights_.size() != size_t(num_layers()) ||
        layer_is_full_.size() != size_t(num_layers()) ||
        !tensor_shape(hidden, at::kHalf, {1, 1, hidden_size()}) ||
        hidden.device().type() != at::kPrivateUse1 ||
        !cos_.defined() || !sin_.defined()) return false;
    const auto full_layers = std::count_if(layer_is_full_.begin(), layer_is_full_.end(),
                                           [](uint8_t full) { return full != 0; });
    const auto cache_capacity = decode_cache_capacity(k, v, layer_is_full_);
    const int64_t rope_position = position + mrope_pos_delta_;
    // Out-of-range requests keep the ordinary launchers and their own errors.
    if (full_layers != 6 || !cache_capacity || position >= *cache_capacity ||
        rope_position < 0 || rope_position >= cos_.size(0) ||
        rope_position >= sin_.size(0)) return false;
    auto& graph = RpuKernelGraph::active();
    if ((graph.state() != RpuKernelGraph::State::RECORDING &&
         graph.state() != RpuKernelGraph::State::REPLAYING) ||
        graph.kernel_register_census_active() || graph.runtime_policy().fmb_fast_replay ||
        graph.runtime_policy().fmb_deep_fast_replay) return false;
    if (!checked_decode_replay_state_)
        checked_decode_replay_state_ = std::make_shared<CheckedDecodeReplayState>();
    auto& state = *checked_decode_replay_state_;
    TORCH_CHECK(!state.active, "nested Qwen3.5 checked decode invocation");
    state.begin = graph.op_stream_stamp();
    // External prefixes/suffixes are not silently swallowed by a full-stream skip.
    if (state.begin.position != 0) return false;
    state.active = true; state.body_seen = false; state.skipped = false;
    state.key = graph_key(graph, state.begin);
    state.generation = installed_model_state_generation();
    state.position = position; state.k = k; state.v = v; state.descriptor = descriptor;
    state.cache_capacity = *cache_capacity;
    state.rebind = graph.state() == RpuKernelGraph::State::RECORDING || !state.recipe ||
        !(state.recipe->key == state.key) || state.recipe->model_generation != state.generation ||
        !descriptor.equals(state.recipe->descriptor) || !state.owners_match(*this);
    return true;
}

bool Qwen3_5Model::try_checked_layer_body_replay() {
    if (!checked_decode_replay_state_ || !checked_decode_replay_state_->active) return false;
    auto& state = *checked_decode_replay_state_;
    auto& graph = RpuKernelGraph::active();
    const auto& chunks = ctx().stage_plan.compute.chunks;
    // Bind the resolved route, and the one chunk whose fields the patches encode.
    if (!ctx().has_complete_physical_manifest() ||
        ctx().attention_policy != AttentionExecutionPolicy::DDR_KV || ctx().seq_len != 1 ||
        ctx().batch_size != 1 || ctx().position != state.position ||
        ctx().stage_plan.chunk_mode != ChunkMode::SEQUENTIAL || chunks.size() != 1 ||
        chunks.front().offset != 0 || chunks.front().len != 1 ||
        chunks.front().kv_seq_len != state.position + 1) return false;
    state.body = graph.op_stream_stamp();
    TORCH_CHECK(graph_key(graph, state.body) == state.key &&
                    state.generation == installed_model_state_generation(),
                "Qwen3.5 checked replay graph changed before its body");
    state.layout = checked_layer_body_replay_layout_hash();
    if (!state.layout) return false;
    state.body_seen = true;
    if (state.rebind) return false;
    auto& recipe = *state.recipe;
    const auto topology = graph.build_topology_hash();
    TORCH_CHECK(graph.state() == RpuKernelGraph::State::REPLAYING &&
        graph.is_op_stream_stamp_current(recipe.begin) &&
        graph.is_op_stream_stamp_current(recipe.body) && graph.is_op_stream_stamp_current(recipe.end) &&
        recipe.begin.position == 0 && state.body.position == recipe.body.position &&
        recipe.end.position == state.body.extent && recipe.end.extent == state.body.extent &&
        recipe.layout == state.layout && recipe.cache_capacity == state.cache_capacity && topology &&
        (!recipe.topology || recipe.topology == topology),
        "Qwen3.5 checked replay graph/stamp/topology/layout proof changed");
    for (int core = 0; core < 8; ++core)
        TORCH_CHECK(recipe.bases[core] == SPM_ALLOC.addr(core, 0),
                    "Qwen3.5 checked replay absolute SPM base changed");
    TORCH_CHECK(recipe.temporary == SPM_ALLOC.temporary_mark() &&
                    recipe.norms == CheckedDecodeReplayState::norm_addresses(*this),
                "Qwen3.5 checked replay temporary/persistent SPM layout changed");
    recipe.topology = topology; // First REPLAY seals the committed BUILD hash.
    RECORD_FUNCTION("rpu_qwen3_5::checked_decode_replay", {});
    update_decode_patches(recipe.patches, state.position, state.position + mrope_pos_delta_);
    // No fallback is allowed after this boundary. The existing Graph API owns
    // typed/census/semantic validation and register patch application.
    graph.skip_op_stream_with_patches(recipe.patches, {});
    state.skipped = true;
    ++state.skips;
    return true;
}

void Qwen3_5Model::finish_checked_decode_replay() {
    if (!checked_decode_replay_state_ || !checked_decode_replay_state_->active) return;
    auto& state = *checked_decode_replay_state_;
    auto& graph = RpuKernelGraph::active();
    TORCH_CHECK(state.generation == installed_model_state_generation(),
                "Qwen3.5 checked replay model changed during forward");
    const auto end = graph.op_stream_stamp();
    TORCH_CHECK(graph_key(graph, end) == state.key,
                "Qwen3.5 checked replay graph changed during forward");
    if (state.skipped) {
        TORCH_CHECK(end.position == state.recipe->end.position && end.position == end.extent,
                    "Qwen3.5 checked replay did not consume the exact body extent");
    } else {
        ++state.ordinary;
        if (state.body_seen && state.rebind && end.position == end.extent) {
            auto next = std::make_unique<CheckedDecodeReplayState::Recipe>();
            std::vector<DecodeKernelSlot> kernels;
            const auto plan = graph.replay_plan();
            bool admissible = plan.steps && !graph.kernel_register_census_active();
            if (!admissible) state.reject = 1;
            if (admissible) for (size_t i = 0; i < plan.steps->size(); ++i) {
                const auto& node = plan.steps->at(i);
                if (node.kind == GraphNodeKind::Kernel) {
                    const auto& kernel = node.as_kernel();
                    if (kernel.register_census) { admissible = false; state.reject = 2; break; }
                    kernels.push_back({kernel.kernel_idx, i, kernel.kernel_id
                        ? KERNEL_ID_NAMES[static_cast<size_t>(*kernel.kernel_id)] : kernel.kernel_name,
                        kernel.core_ids == std::vector<uint8_t>({0, 1, 2, 3, 4, 5, 6, 7})});
                } else if (node.kind == GraphNodeKind::Dma) {
                    const auto& dma = node.as_dma();
                    if (dma.semantic_endpoint_id || dma.semantic_spm_peer_id) {
                        admissible = false; state.reject = 3; break;
                    }
                } else if (node.kind != GraphNodeKind::Barrier && node.kind != GraphNodeKind::Branch) {
                    admissible = false; state.reject = 4; break;
                }
            }
            if (admissible && !bind_decode_patches(
                    kernels, std::count_if(layer_is_full_.begin(), layer_is_full_.end(),
                                           [](uint8_t full) { return full != 0; }),
                    next->patches)) {
                admissible = false; state.reject = 5;
            }
            if (admissible) {
                next->owners.reserve(num_layers() * 32);
                admissible = CheckedDecodeReplayState::visit_owners(*this, state.k, state.v,
                    [&](const at::Tensor& t, bool fixed) {
                        DecodeOwner owner;
                        if (!owner.freeze(t, fixed)) return false;
                        next->owners.push_back(std::move(owner));
                        return true;
                    });
                if (!admissible) state.reject = 100 + int64_t(next->owners.size());
            }
            if (admissible) {
                next->key = state.key; next->begin = state.begin;
                next->body = state.body; next->end = end;
                next->model_generation = state.generation;
                next->descriptor.assign(state.descriptor.begin(), state.descriptor.end());
                next->layout = state.layout;
                next->cache_capacity = state.cache_capacity;
                next->topology = graph.build_topology_hash();
                next->norms = CheckedDecodeReplayState::norm_addresses(*this);
                next->temporary = SPM_ALLOC.temporary_mark();
                for (int core = 0; core < 8; ++core) next->bases[core] = SPM_ALLOC.addr(core, 0);
                // Retire old strong owners only after the ordinary emission has
                // rebound every static DDR operand of the current graph.
                state.recipe = std::move(next);
                ++state.binds;
            }
        }
    }
    state.active = false;
    state.k = {}; state.v = {}; state.descriptor = {}; state.cache_capacity = 0;
}

void Qwen3_5Model::cancel_checked_decode_replay() noexcept {
    if (!checked_decode_replay_state_) return;
    auto& state = *checked_decode_replay_state_;
    // A failed invocation cannot leave a partially patched recipe reusable.
    if (state.active) state.recipe.reset();
    state.active = false; state.k = {}; state.v = {}; state.descriptor = {}; state.cache_capacity = 0;
}

bool Qwen3_5Model::checked_decode_replay_skipped() const {
    return checked_decode_replay_state_ && checked_decode_replay_state_->active &&
        checked_decode_replay_state_->skipped;
}

void Qwen3_5Model::set_decode_candidate_head(const at::Tensor& weight, const at::Tensor& output) {
    const int64_t n = weight.defined() && weight.dim() == 2 ? weight.size(0) : 0;
    TORCH_CHECK(num_cores() == 8 && hidden_size() == 1024,
                "Qwen3.5 decode candidate head requires 8 cores and H1024");
    TORCH_CHECK(n > 0 && n % (16 * 8) == 0 && weight.size(1) == hidden_size() &&
                    weight.device().type() == at::kPrivateUse1 &&
                    weight.scalar_type() == at::kHalf && weight.is_contiguous(),
                "Qwen3.5 decode candidate head requires a col-swizzled FP16 [N, 1024] weight");
    TORCH_CHECK(tensor_shape(output, at::kHalf, {1, n}) &&
                    output.device().type() == at::kPrivateUse1,
                "Qwen3.5 decode candidate head requires a contiguous FP16 [1, N] output");
    decode_head_w_ = weight;
    decode_head_out_ = output;
    invalidate_model_state();
}

// One row, the same three stages as the eager single-row Linear: broadcast the
// hidden, the FP16 ACC32 auto-tile, and core-major scatter to DDR. The output is
// one owned tensor, so these nodes need no per-replay field.
void Qwen3_5Model::emit_decode_candidate_head(const at::Tensor& hidden) {
    TORCH_CHECK(graph_dma::active() && tensor_shape(hidden, at::kHalf, {1, 1, hidden_size()}),
                "Qwen3.5 decode candidate head belongs to the captured decode stream");
    constexpr int cores = 8;
    constexpr size_t dwidth = sizeof(c10::Half);
    const int64_t n = decode_head_w_.size(0), k = hidden_size();
    const size_t local_n = static_cast<size_t>(n / cores);
    SpmAllocator::ScopedTemporary spm_scope;
    const uint32_t x = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(k * dwidth));
    const uint32_t y = SPM_ALLOC.addr(0, SPM_ALLOC.alloc_temporary(local_n * dwidth));
    rpu_launch_ddr_broadcast_spm_dma(hidden.data_ptr<c10::Half>(), k, x, cores);
    rpu_launch_linear_spm_to_spm_kernel(x, decode_head_w_, y, 1, n, k, /*partition=*/1, cores, 0,
                                        static_cast<uint32_t>(k * dwidth));
    rpu_launch_spm_scatter_ddr_dma(y, decode_head_out_.data_ptr<c10::Half>(), local_n,
                                   local_n * dwidth, cores);
}

std::vector<int64_t> Qwen3_5Model::checked_decode_replay_counters() const {
    if (!checked_decode_replay_state_) return {0, 0, 0, 0};
    const auto& state = *checked_decode_replay_state_;
    return {state.binds, state.skips, state.ordinary, state.reject};
}

} // namespace v3
