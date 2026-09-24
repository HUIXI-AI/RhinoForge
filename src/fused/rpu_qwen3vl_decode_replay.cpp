// Bounded owner-local replay of the admitted Qwen3-VL M1 bodies.
// The ordinary forward/preload and Graph runtime remain the authority for
// dynamic DMA, allocation and COMPLETE routes. Only proven static emission is
// omitted; the existing Graph patch API refreshes every dynamic kernel field.
#include "rpu_qwen3_model.h"
#include <set>
#include "../ops/rpu_linear_tiling.h"

namespace v3 {
namespace {

struct DecodeOwner {
    at::Tensor tensor; // retains storage even if the caller replaces its cache
    c10::TensorImpl* impl = nullptr;
    c10::StorageImpl* storage = nullptr;
    at::ScalarType dtype = at::ScalarType::Undefined;
    c10::Device device{c10::DeviceType::CPU};
    std::vector<int64_t> shape, strides;
    int64_t offset = 0;
    uint32_t version = 0;
    uint64_t address = 0;
    size_t storage_bytes = 0;
    bool immutable = false;

    bool freeze(const at::Tensor& t, bool fixed) {
        if (!t.defined() || t.device().type() != at::kPrivateUse1 ||
            t.layout() != c10::Layout::Strided || !t.is_contiguous()) return false;
        const auto& counter = t.unsafeGetTensorImpl()->version_counter();
        // Versionless inference weights are still supported by the ordinary
        // body; they cannot prove an immutable replay binding.
        if (fixed && !counter.enabled()) return false;
        tensor = t; impl = t.unsafeGetTensorImpl();
        storage = t.storage().unsafeGetStorageImpl();
        dtype = t.scalar_type(); device = t.device();
        shape = t.sizes().vec(); strides = t.strides().vec();
        offset = t.storage_offset(); storage_bytes = t.storage().nbytes();
        address = RpuGetDevAddr(t.data_ptr()); immutable = fixed;
        version = fixed ? counter.current_version() : 0;
        return address && address < (1ULL << 40) && address % 256 == 0;
    }

    bool matches(const at::Tensor& t) const {
        if (!t.defined() || t.unsafeGetTensorImpl() != impl ||
            t.layout() != c10::Layout::Strided || t.scalar_type() != dtype ||
            t.device() != device || t.storage().unsafeGetStorageImpl() != storage ||
            t.storage().nbytes() != storage_bytes || t.storage_offset() != offset ||
            !t.sizes().equals(shape) || !t.strides().equals(strides) ||
            RpuGetDevAddr(t.data_ptr()) != address) return false;
        const auto& counter = t.unsafeGetTensorImpl()->version_counter();
        return !immutable || (counter.enabled() && counter.current_version() == version);
    }
};

struct DecodeGraphKey {
    const RpuKernelGraph* graph = nullptr;
    uint64_t lifetime = 0, generation = 0, signature = 0, segment = 0;
    bool operator==(const DecodeGraphKey& other) const {
        return graph == other.graph && lifetime == other.lifetime &&
            generation == other.generation && signature == other.signature &&
            segment == other.segment;
    }
};

DecodeGraphKey graph_key(RpuKernelGraph& graph, const GraphOpStreamStamp& stamp) {
    return {&graph, graph.debug_stats().graph_lifetime_id, stamp.build_generation,
            stamp.signature_identity, stamp.signature_segment_key};
}

struct DecodeKernelSlot { size_t kernel, node; std::string name; };

// Physical packed-cache rows, independent of the cold prefill envelope. The
// public runner separately checks RPUCache's logical (possibly unaligned) limit.
constexpr int64_t kCheckedDecodeMaxCacheRows = 4352;

std::optional<int64_t> decode_cache_capacity(at::TensorList k, at::TensorList v, int layers) {
    if ((layers != 28 && layers != 36) || k.size() != size_t(layers) || v.size() != size_t(layers) || !k[0].defined() || k[0].dim() != 7)
        return std::nullopt;
    const int64_t blocks = k[0].size(1);
    if (blocks <= 0 || blocks > kCheckedDecodeMaxCacheRows / 16)
        return std::nullopt;
    const auto valid = [blocks](const at::Tensor& t) {
        if (!t.defined() || t.device().type() != at::kPrivateUse1 ||
            t.scalar_type() != at::kHalf || t.layout() != c10::Layout::Strided ||
            !t.is_contiguous()) return false;
        return t.sizes().equals({1, blocks, 1, 8, 8, 16, 16});
    };
    for (int layer = 0; layer < layers; ++layer)
        if (!valid(k[layer]) || !valid(v[layer])) return std::nullopt;
    return blocks * 16;
}

// Cold geometry/dtype proof chooses a grammar, not a new kernel route.
// The actual recorded stream must match in full before any emission is skipped.
struct DecodeProfile {
    int layers = 36;
    at::ScalarType dtype = at::kChar;
    bool acc32 = false, gemv = true, packed_qkv = true, fused_mlp = true;
};

bool linear_name(const std::string& name, const DecodeProfile& profile, bool head) {
    if (!head && profile.gemv)
        return name == (profile.dtype == at::kHalf ? "llama_gemv" : "llama_gemv_wint8");
    using namespace rpu_pl_tiling;
    for (int n : {128, 112, 96, 80, 64, 48, 32}) {
        std::string expected;
        if (profile.dtype == at::kByte)
            expected = profile.acc32
                ? autotile_kernel_name_int4_acc32(mtile_int4_acc32(n), n)
                : autotile_kernel_name_int4(mtile_int4(n), n);
        else {
            const bool half = profile.dtype == at::kHalf;
            expected = profile.acc32
                ? autotile_kernel_name_acc32(half, half ? mtile_fp16_acc32(n) : mtile_w8a16_acc32(n), n)
                : autotile_kernel_name(half, half ? mtile_w16(n) : mtile_w8a16(n), n);
        }
        if (name == expected) return true;
    }
    return false;
}

bool bind_decode_patches(const std::vector<DecodeKernelSlot>& kernels,
                         std::vector<RegisterPatch>& patches,
                         const DecodeProfile& profile) {
    size_t cursor = 0;
    std::set<size_t> indices;
    for (size_t i = 0; i < kernels.size(); ++i) {
        if (!indices.insert(kernels[i].kernel).second ||
            (i && kernels[i - 1].node >= kernels[i].node)) return false;
    }
    const auto norm = [&]() {
        if (cursor == kernels.size()) return false;
        const auto& n = kernels[cursor].name;
        if (n != "llama_rms_norm" && n != "llama_rms_norm_v16" &&
            n != "llama_rms_norm_v32") return false;
        ++cursor; return true;
    };
    const auto take = [&](const char* name) {
        if (cursor == kernels.size() || kernels[cursor].name != name) return false;
        ++cursor; return true;
    };
    const auto linear = [&]() {
        if (cursor == kernels.size() || !linear_name(kernels[cursor].name, profile, false)) return false;
        ++cursor; return true;
    };
    const auto position_patch = [&]() {
        for (uint32_t reg : {0u, 1u})
            patches.push_back({kernels[cursor - 1].kernel, reg, 0, 2});
    };
    patches.clear(); patches.reserve(profile.layers * 16);
    std::optional<int> qk_kind;
    std::optional<bool> combined_o;
    for (int layer = 0; layer < profile.layers; ++layer) {
        if (!norm() || !linear()) return false;
        if (!profile.packed_qkv && (!linear() || !linear())) return false;
        if (cursor == kernels.size()) return false;
        const auto& qk = kernels[cursor].name;
        const int kind = qk == "qwen3vl_qk_norm_mrope_kv_insert_d128" ? 2 :
            qk == "qwen3vl_qk_norm_mrope_d128" ? 1 : 0;
        if (qk_kind && *qk_kind != kind) return false;
        qk_kind = kind;
        if (kind) {
            if (profile.layers != 36) return false;
            ++cursor; position_patch();
        } else {
            if (!norm() || !norm()) return false;
            for (int axis = 0; axis < 2; ++axis) {
                if (!take("llama_mrope_interleave")) return false;
                position_patch();
            }
        }
        if (kind != 2) {
            for (const auto* name : {"llama_insert_kcache_multiwarp",
                                    "llama_insert_vcache_multiwarp"}) {
                if (!take(name)) return false;
                position_patch();
            }
        }
        if (!take("llm_fp16_32b_prefill_flash_attn_univ_dp")) return false;
        for (uint32_t reg : {0u, 1u, 4u, 5u, 16u, 17u, 29u, 31u})
            patches.push_back({kernels[cursor - 1].kernel, reg, 0, 2});
        if (cursor == kernels.size()) return false;
        const bool fused_o = kernels[cursor].name == "qwen3vl_o_ring_norm_m1_h2560_w8";
        if (combined_o && *combined_o != fused_o) return false;
        combined_o = fused_o;
        if (fused_o) {
            if (profile.layers != 36 || profile.dtype != at::kChar || !profile.gemv ||
                !take("qwen3vl_o_ring_norm_m1_h2560_w8")) return false;
        } else if (!linear() || !take("llm_all_reduce_residual_nopace") || !norm()) return false;
        if (profile.fused_mlp) {
            if (!take("qwen3vl_gateup_swiglu_gemv")) return false;
        } else if (profile.layers == 36 && profile.dtype == at::kChar && profile.gemv) {
            if (!linear() || !linear() || !take("silu_mul")) return false;
        } else if (!linear() || !take("unary") || !linear() || !take("binary_sameshape")) return false;
        if (!linear() || !take("llm_all_reduce_residual_nopace") ||
            (layer < 3 && !take("binary_sameshape"))) return false;
    }
    return norm() && cursor + 1 == kernels.size() && linear_name(kernels[cursor].name, profile, true);
}

void update_decode_patches(std::vector<RegisterPatch>& patches, int64_t position,
                           int64_t cache_capacity, int layers) {
    TORCH_CHECK((layers == 28 || layers == 36) &&
                    (patches.size() == size_t(layers * 10) || patches.size() == size_t(layers * 14) ||
                     patches.size() == size_t(layers * 16)) &&
                    cache_capacity > 0 && cache_capacity <= kCheckedDecodeMaxCacheRows &&
                    cache_capacity % 16 == 0 && position >= 0 && position < cache_capacity,
                "unsealed Qwen3-VL decode patch recipe");
    const uint32_t p = position, s = p + 1;
    const size_t stride = patches.size() / layers;
    const size_t position_fields = stride - 8;
    const std::array<uint16_t, 8> attention_values{
        uint16_t(s), uint16_t(s >> 16), uint16_t((s + 15) / 16),
        uint16_t((p + 15) / 16), uint16_t(p), uint16_t(p >> 16),
        uint16_t((s + 15) / 16), uint16_t((s + 15) / 16)};
    for (int layer = 0; layer < layers; ++layer) {
        for (size_t i = 0; i < position_fields; i += 2) {
            patches[layer * stride + i].value = uint16_t(p);
            patches[layer * stride + i + 1].value = uint16_t(p >> 16);
        }
        for (size_t i = 0; i < attention_values.size(); ++i)
            patches[layer * stride + position_fields + i].value = attention_values[i];
    }
}

bool tensor_shape(const at::Tensor& t, at::ScalarType dtype, at::IntArrayRef shape) {
    return t.defined() && t.scalar_type() == dtype && t.sizes().equals(shape);
}

bool readable_elements(const at::Tensor& tensor, uint64_t count) {
    const auto offset = tensor.storage_offset();
    const auto width = tensor.element_size();
    const auto bytes = tensor.storage().nbytes();
    return offset >= 0 && static_cast<uint64_t>(offset) <= bytes / width &&
        count <= bytes / width - static_cast<uint64_t>(offset);
}

void check_decode_position(const at::Tensor& positions, const at::Tensor& cos,
                           int64_t position, int64_t cache_capacity) {
    TORCH_CHECK(position >= 0 && position < cache_capacity && position < cos.size(0) &&
                    position < positions.size(0),
                "Qwen3-VL checked decode position exceeds physical cache/logical RoPE bounds");
    TORCH_CHECK(readable_elements(positions, static_cast<uint64_t>(position) * 3 + 8),
                "Qwen3-VL checked decode position lacks 32-byte readable backing");
    const auto* thw = positions.data_ptr<int32_t>() + position * 3;
    for (int axis = 0; axis < 3; ++axis)
        TORCH_CHECK(thw[axis] >= 0 && thw[axis] < cos.size(0),
                    "Qwen3-VL checked decode T/H/W exceeds logical RoPE bounds");
}

} // namespace

struct CausalDecoderModel::Qwen3VlDecodeReplayState {
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

    static DecodeProfile profile(const CausalDecoderModel& model) {
        DecodeProfile p;
        p.layers = model.num_layers(); p.dtype = model.layer_weights_.front().q_w.scalar_type();
        p.acc32 = model.linear_acc32_; p.gemv = model.qwen3vl_decode_gemv(1);
        p.packed_qkv = p.gemv && model.qwen3vl_4b_packed_qkv_weights_;
        p.fused_mlp = p.gemv && model.qwen3vl_4b_decode_gateup_fused_;
        return p;
    }

    template<class F> static bool visit_owners(CausalDecoderModel& model,
                                               at::TensorList k, at::TensorList v, F fn) {
        const auto p = profile(model);
        for (int l = 0; l < p.layers; ++l) {
            const auto& w = model.layer_weights_[l];
            if (p.packed_qkv) {
                if (!fn(w.qkv_w, true) || !fn(w.qkv_ws, true)) return false;
            } else {
                for (const auto* t : {&w.q_w, &w.k_w, &w.v_w})
                    if (!fn(*t, true)) return false;
                if (p.dtype != at::kHalf)
                    for (const auto* t : {&w.q_ws, &w.k_ws, &w.v_ws})
                        if (!fn(*t, true)) return false;
            }
            for (const auto* t : {&w.o_w, &w.gate_w, &w.up_w, &w.down_w,
                    &w.input_norm_w, &w.post_norm_w, &w.q_norm_w, &w.k_norm_w})
                if (!fn(*t, true)) return false;
            if (p.dtype != at::kHalf)
                for (const auto* t : {&w.o_ws, &w.gate_ws, &w.up_ws, &w.down_ws})
                    if (!fn(*t, true)) return false;
            if (!fn(k[l], false) || !fn(v[l], false)) return false;
        }
        for (const auto* t : {&model.cos_, &model.sin_, &model.final_norm_w_, &model.lm_head_w_})
            if (!fn(*t, true)) return false;
        if (p.packed_qkv && !fn(model.qwen3vl_4b_qkv_scale_bank_, true)) return false;
        if (p.dtype != at::kHalf && !fn(model.lm_head_w_scale_, true)) return false;
        return fn(model.position_ids_keepalive_, false);
    }

    static bool projection_shape(const at::Tensor& w, const at::Tensor& scale,
                                  at::ScalarType dtype, int64_t n, int64_t k, bool row) {
        if (!tensor_shape(w, dtype, {n, dtype == at::kByte ? k / 2 : k})) return false;
        if (dtype == at::kHalf) return !scale.defined();
        if (dtype == at::kChar) return tensor_shape(scale, at::kHalf, {n});
        if (dtype != at::kByte || !scale.defined() || scale.scalar_type() != at::kHalf ||
            scale.dim() != 2 || scale.size(0) != 32) return false;
        const int64_t local_n = row ? n : n / 8, local_k = row ? k / 8 : k;
        return scale.numel() == ((local_k / 32 + 3) / 4) * ((local_n + 63) / 64) * 8 * 4 * 64;
    }

    static bool owner_shapes(CausalDecoderModel& model) {
        const auto p = profile(model);
        const int64_t h = model.hidden_size(), i = model.intermediate_size();
        const int64_t q = model.num_q_heads() * 128;
        for (const auto& w : model.layer_weights_) {
            if (p.packed_qkv) {
                if (!tensor_shape(w.qkv_w, at::kChar, {6144, 2560}) ||
                    !tensor_shape(w.qkv_ws, at::kHalf, {6144})) return false;
            } else if (!projection_shape(w.q_w, w.q_ws, p.dtype, q, h, false) ||
                       !projection_shape(w.k_w, w.k_ws, p.dtype, 1024, h, false) ||
                       !projection_shape(w.v_w, w.v_ws, p.dtype, 1024, h, false)) return false;
            if (!projection_shape(w.o_w, w.o_ws, p.dtype, h, q, true) ||
                !projection_shape(w.gate_w, w.gate_ws, p.dtype, i, h, false) ||
                !projection_shape(w.up_w, w.up_ws, p.dtype, i, h, false) ||
                !projection_shape(w.down_w, w.down_ws, p.dtype, h, i, true) ||
                !tensor_shape(w.input_norm_w, at::kHalf, {h}) ||
                !tensor_shape(w.post_norm_w, at::kHalf, {h}) ||
                !tensor_shape(w.q_norm_w, at::kHalf, {128}) ||
                !tensor_shape(w.k_norm_w, at::kHalf, {128})) return false;
            if (p.fused_mlp && (!readable_elements(w.gate_ws, i + 256) ||
                                !readable_elements(w.up_ws, i + 256))) return false;
            if (p.gemv && model.qwen3vl_4b_decode_o_ring_norm_fused_ &&
                !readable_elements(w.o_ws, h + 256)) return false;
        }
        return tensor_shape(model.final_norm_w_, at::kHalf, {h}) &&
            (!p.packed_qkv || tensor_shape(model.qwen3vl_4b_qkv_scale_bank_, at::kHalf, {36, 6144})) &&
            // Ordinary 2B/8B use an exact-size THW table; only the bounded
            // cache envelope is reachable. 4B retains its fused-kernel tail proof.
            readable_elements(model.position_ids_keepalive_,
                ((h == 2560 ? 8192 : kCheckedDecodeMaxCacheRows) - 1) * 3 + 8);
    }

    bool owners_match(CausalDecoderModel& model) const {
        size_t index = 0;
        return visit_owners(model, k, v, [&](const at::Tensor& t, bool) {
            return index < recipe->owners.size() && recipe->owners[index++].matches(t);
        }) && index == recipe->owners.size();
    }

    static auto norm_addresses(CausalDecoderModel& model) {
        std::vector<uint32_t> result(model.num_layers() * 4 + 1);
        size_t i = 0;
        for (int l = 0; l < model.num_layers(); ++l)
            for (const auto* name : {"norm_w", "post_norm_w", "q_norm_w", "k_norm_w"})
                result[i++] = model.layer_addr(l, 0, name);
        result[i] = model.addr(0, "final_norm_w");
        return result;
    }

    static void visibility(CausalDecoderModel& model, at::TensorList k, at::TensorList v) {
        for (size_t l = 0; l < model.num_layers(); ++l) {
            const auto& w = model.layer_weights_[l];
            // Preserve the skipped launchers' gated flushes on their actual
            // native operands, including the persistent norm preloads. Sized
            // preamble/output flushes still execute in the ordinary caller.
            for (const auto* t : {&w.input_norm_w, &w.post_norm_w,
                                  &w.q_norm_w, &w.k_norm_w})
                rpu_ddr_flush(t->data_ptr());
            if (!profile(model).packed_qkv) {
                if (profile(model).dtype != at::kHalf)
                    for (const auto* t : {&w.q_ws, &w.k_ws, &w.v_ws}) rpu_ddr_flush(t->data_ptr());
                for (const auto* t : {&w.q_w, &w.k_w, &w.v_w})
                    rpu_ddr_flush(t->data_ptr());
            } else {
                rpu_ddr_flush(w.qkv_w.data_ptr()); rpu_ddr_flush(w.qkv_ws.data_ptr());
            }
            rpu_ddr_flush(model.cos_.data_ptr()); rpu_ddr_flush(model.sin_.data_ptr());
            rpu_ddr_flush(k[l].data_ptr()); rpu_ddr_flush(k[l].data_ptr());
            rpu_ddr_flush(v[l].data_ptr()); rpu_ddr_flush(v[l].data_ptr());
            rpu_ddr_flush(k[l].data_ptr()); rpu_ddr_flush(v[l].data_ptr());
            for (const auto* t : {&w.o_w, &w.gate_w, &w.up_w, &w.down_w})
                rpu_ddr_flush(t->data_ptr());
            if (profile(model).dtype != at::kHalf)
                for (const auto* t : {&w.o_ws, &w.gate_ws, &w.up_ws, &w.down_ws})
                    rpu_ddr_flush(t->data_ptr());
        }
        rpu_ddr_flush(model.final_norm_w_.data_ptr());
        rpu_ddr_flush(model.lm_head_w_.data_ptr());
        if (profile(model).dtype != at::kHalf) rpu_ddr_flush(model.lm_head_w_scale_.data_ptr());
    }
};

bool CausalDecoderModel::begin_qwen3vl_decode_replay(
    const at::Tensor& hidden, at::TensorList k, at::TensorList v,
    const std::optional<at::Tensor>& mask, int64_t position,
    bool is_causal, at::IntArrayRef descriptor) {
    // All controls are existing cold native fields, never public profile-vector
    // offsets, a changed Graph policy, or a second model configuration axis.
    // The existing exact 8B profile keeps ordinary ACC32 auto-tile arithmetic.
    // Do not reuse the resident-prefill capability: that would also admit its
    // unmeasured ACC16 GEMV route and change more than host replay emission.
    const bool exact_8b = num_layers() == 36 && hidden_size() == 4096 &&
        intermediate_size() == 12288 && num_q_heads() == 32 && linear_acc32_ &&
        !qwen3vl_decode_gemv(1) && !layer_weights_.empty() &&
        layer_weights_.front().q_w.defined() &&
        layer_weights_.front().q_w.scalar_type() == at::kHalf;
    const bool ordinary = qwen3vl_ordinary_resident_prefill_profile_ || qwen3vl_4b_w8_profile_ || exact_8b;
    const bool geometry = (num_layers() == 28 && hidden_size() == 2048 && intermediate_size() == 6144 && num_q_heads() == 16) ||
        (num_layers() == 36 && hidden_size() == 2560 && intermediate_size() == 9728 && num_q_heads() == 32) || exact_8b;
    const bool envelope = exact_8b
        ? (chunk_envelope().max_kv_len == 320 || chunk_envelope().max_kv_len == 4176) &&
          chunk_envelope().chunk == 128
        : (chunk_envelope().max_kv_len == 2048 || chunk_envelope().max_kv_len == 4176) &&
          chunk_envelope().chunk == (num_layers() == 28 ? 320 : 256);
    if (typeid(*this) != typeid(CausalDecoderModel) || !RpuKernelGraph::has_active() ||
        !ordinary || !geometry || !is_causal || mask.has_value() || partial_mrope_active_ ||
        adarms_ || planning_mode_ != CausalDecoderPlanningMode::ORDINARY ||
        qwen3vl_pooler_z1_dispatch_ || qwen3vl_multiview_text_composite_dispatch_ ||
        !has_qk_norm_ || !has_mrope_ || num_kv_heads() != 8 || head_dim() != 128 ||
        layer_weights_.size() != size_t(num_layers()) ||
        num_cores() != 8 || attn_tp() != 8 || mlp_tp() != 8 || lm_head_tp() != 8 ||
        !fuse_lm_head_ || vocab_size_ != 151936 || lm_head_exact_candidates_ || reuse_lm_head_output_ ||
        !use_silu_ || nvfp4_ || has_qkv_bias_ || norm_slab_active() ||
        equal_two_prefill_ || configured_chunk_size_cap_ ||
        adarms_mutable_ || adarms_unroll_ || adarms_schedule_select_ || adarms_fused_bcast_enabled_ ||
        sdpa_kernel_ != SdpaKernelType::FLASH_ATTN_SPM ||
        mrope_section_ != std::vector<int32_t>({24, 20, 20}) ||
        deepstack_lang_layers_ != std::vector<int64_t>({0, 1, 2}) ||
        // Preserve each named prefill envelope; actual KV owners bound decode.
        !envelope ||
        get_debug_export() || descriptor.empty() || k.size() != size_t(num_layers()) || v.size() != size_t(num_layers()) ||
        !tensor_shape(hidden, at::kHalf, {1, 1, hidden_size()})) return false;
    auto& graph = RpuKernelGraph::active();
    if ((graph.state() != RpuKernelGraph::State::RECORDING &&
         graph.state() != RpuKernelGraph::State::REPLAYING) ||
        graph.kernel_register_census_active() || graph.runtime_policy().fmb_fast_replay ||
        graph.runtime_policy().fmb_deep_fast_replay) return false;
    const auto profile = Qwen3VlDecodeReplayState::profile(*this);
    // Quantized ACC32 is not yet admitted to this specialized replay path;
    // retain the generic recorded-Graph execution for those profiles.
    if (profile.acc32 && profile.dtype != at::kHalf) return false;
    if (!Qwen3VlDecodeReplayState::projection_shape(lm_head_w_, lm_head_w_scale_,
            profile.dtype, 151936, hidden_size(), false) ||
        !tensor_shape(position_ids_keepalive_, at::kInt, {8192, 3}) ||
        !cos_.defined() || cos_.dim() != 2 || cos_.size(1) != 64 ||
        cos_.scalar_type() != at::kHalf || !tensor_shape(sin_, at::kHalf, cos_.sizes())) return false;
    const auto cache_capacity = decode_cache_capacity(k, v, num_layers());
    if (!cache_capacity) return false;
    check_decode_position(position_ids_keepalive_, cos_, position, *cache_capacity);
    if (!qwen3vl_decode_replay_state_)
        qwen3vl_decode_replay_state_ = std::make_shared<Qwen3VlDecodeReplayState>();
    auto& state = *qwen3vl_decode_replay_state_;
    TORCH_CHECK(!state.active, "nested Qwen3-VL checked decode invocation");
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

bool CausalDecoderModel::try_checked_layer_body_replay() {
    if (!qwen3vl_decode_replay_state_ || !qwen3vl_decode_replay_state_->active) return false;
    auto& state = *qwen3vl_decode_replay_state_;
    auto& graph = RpuKernelGraph::active();
    // AUTO capability can be enabled while the actual M1 plan uses DDR KV.
    // Bind the resolved route, not the broader cold capability switch.
    if (!ctx().has_complete_physical_manifest() ||
        ctx().attention_policy != AttentionExecutionPolicy::DDR_KV || ctx().seq_len != 1 ||
        ctx().batch_size != 1 || ctx().position != state.position ||
        ctx().stage_plan.chunk_mode != ChunkMode::SEQUENTIAL ||
        ctx().stage_plan.compute.chunks.size() != 1) return false;
    state.body = graph.op_stream_stamp();
    TORCH_CHECK(graph_key(graph, state.body) == state.key &&
                    state.generation == installed_model_state_generation(),
                "Qwen3-VL checked replay graph changed before its body");
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
        "Qwen3-VL checked replay graph/stamp/topology/layout proof changed");
    for (int core = 0; core < 8; ++core)
        TORCH_CHECK(recipe.bases[core] == SPM_ALLOC.addr(core, 0),
                    "Qwen3-VL checked replay absolute SPM base changed");
    TORCH_CHECK(recipe.temporary == SPM_ALLOC.temporary_mark() &&
                    recipe.norms == Qwen3VlDecodeReplayState::norm_addresses(*this),
                "Qwen3-VL checked replay temporary/persistent SPM layout changed");
    recipe.topology = topology; // First REPLAY seals the committed BUILD hash.
    RECORD_FUNCTION("rpu_causal::checked_decode_replay", {});
    update_decode_patches(recipe.patches, state.position, recipe.cache_capacity, num_layers());
    Qwen3VlDecodeReplayState::visibility(*this, state.k, state.v);
    // No fallback is allowed after this boundary. The existing Graph API owns
    // typed/census/semantic validation and register patch application.
    graph.skip_op_stream_with_patches(recipe.patches, {});
    state.skipped = true;
    return true;
}

void CausalDecoderModel::finish_qwen3vl_decode_replay() {
    if (!qwen3vl_decode_replay_state_ || !qwen3vl_decode_replay_state_->active) return;
    auto& state = *qwen3vl_decode_replay_state_;
    auto& graph = RpuKernelGraph::active();
    TORCH_CHECK(state.generation == installed_model_state_generation(),
                "Qwen3-VL checked replay model changed during forward");
    const auto end = graph.op_stream_stamp();
    TORCH_CHECK(graph_key(graph, end) == state.key,
                "Qwen3-VL checked replay graph changed during forward");
    if (state.skipped) {
        TORCH_CHECK(end.position == state.recipe->end.position && end.position == end.extent,
                    "Qwen3-VL checked replay did not consume the exact body extent");
    } else if (state.body_seen && state.rebind && end.position == end.extent) {
        auto next = std::make_unique<Qwen3VlDecodeReplayState::Recipe>();
        std::vector<DecodeKernelSlot> kernels;
        const auto plan = graph.replay_plan();
        bool admissible = plan.steps && !graph.kernel_register_census_active();
        if (admissible) for (size_t i = 0; i < plan.steps->size(); ++i) {
            const auto& node = plan.steps->at(i);
            if (node.kind == GraphNodeKind::Kernel) {
                const auto& kernel = node.as_kernel();
                if (kernel.register_census || kernel.core_ids != std::vector<uint8_t>({0,1,2,3,4,5,6,7})) { admissible = false; break; }
                kernels.push_back({kernel.kernel_idx, i, kernel.kernel_id
                    ? KERNEL_ID_NAMES[static_cast<size_t>(*kernel.kernel_id)] : kernel.kernel_name});
            } else if (node.kind == GraphNodeKind::Dma) {
                const auto& dma = node.as_dma();
                if (dma.semantic_endpoint_id || dma.semantic_spm_peer_id) { admissible = false; break; }
            } else if (node.kind != GraphNodeKind::Barrier && node.kind != GraphNodeKind::Branch) {
                admissible = false; break;
            }
        }
        if (admissible && Qwen3VlDecodeReplayState::owner_shapes(*this) &&
            bind_decode_patches(kernels, next->patches,
                                Qwen3VlDecodeReplayState::profile(*this))) {
            next->owners.reserve(num_layers() * 20 + 7);
            admissible = Qwen3VlDecodeReplayState::visit_owners(*this, state.k, state.v,
                [&](const at::Tensor& t, bool fixed) {
                    DecodeOwner owner;
                    if (!owner.freeze(t, fixed)) return false;
                    next->owners.push_back(std::move(owner)); return true;
                });
            if (admissible) {
                next->key = state.key; next->begin = state.begin;
                next->body = state.body; next->end = end;
                next->model_generation = state.generation;
                next->descriptor.assign(state.descriptor.begin(), state.descriptor.end());
                next->layout = state.layout;
                next->cache_capacity = state.cache_capacity;
                next->topology = graph.build_topology_hash();
                next->norms = Qwen3VlDecodeReplayState::norm_addresses(*this);
                next->temporary = SPM_ALLOC.temporary_mark();
                for (int core = 0; core < 8; ++core) next->bases[core] = SPM_ALLOC.addr(core, 0);
                // Retire old strong owners only after the ordinary emission has
                // rebound every static DDR operand of the current graph.
                state.recipe = std::move(next);
            }
        }
    }
    state.active = false;
    state.k = {}; state.v = {}; state.descriptor = {}; state.cache_capacity = 0;
}

void CausalDecoderModel::cancel_qwen3vl_decode_replay() noexcept {
    if (!qwen3vl_decode_replay_state_) return;
    auto& state = *qwen3vl_decode_replay_state_;
    // A failed invocation cannot leave a partially patched recipe reusable.
    if (state.active) state.recipe.reset();
    state.active = false; state.k = {}; state.v = {}; state.descriptor = {}; state.cache_capacity = 0;
}

} // namespace v3
