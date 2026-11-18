// Bounded owner-local replay of admitted Qwen3-VL and TTS Predictor M1 bodies.
// The ordinary forward/preload and Graph runtime remain the authority for
// dynamic DMA, allocation and COMPLETE routes. Only proven static emission is
// omitted; the existing Graph patch API refreshes every dynamic kernel field.
#include "rpu_qwen3_model.h"
#include "rpu_decode_replay_common.h"
#include <set>
#include "rpu_linear_tiling.h"

namespace v3 {
namespace {

using decode_replay::DecodeGraphKey;
using decode_replay::DecodeOwner;
using decode_replay::graph_key;
using decode_replay::tensor_shape;

struct DecodeKernelSlot { size_t kernel, node; std::string name; };

// Physical packed-cache rows, independent of the cold prefill envelope. The
// public runner separately checks RPUCache's logical (possibly unaligned) limit.
constexpr int64_t kCheckedDecodeMaxKeyRows = int64_t{UINT16_MAX} * 16;

std::optional<int64_t> decode_cache_capacity(at::TensorList k, at::TensorList v, int layers) {
    if ((layers != 5 && layers != 28 && layers != 36) || k.size() != size_t(layers) || v.size() != size_t(layers) || !k[0].defined() || k[0].dim() != 7)
        return std::nullopt;
    const int64_t blocks = k[0].size(1);
    if (blocks <= 0 || blocks > std::numeric_limits<int64_t>::max() / 16)
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
    bool tts_predictor = false;
    // A fused head's weight dtype need not match the decoder layers'. It does on
    // Qwen3-VL, where head_dtype resolves to dtype, and does not on the TTS
    // Predictor, whose FP16 layers carry an int8 head.
    at::ScalarType head_dtype = at::kChar;
    bool head_bank = false;
};

bool linear_name(const std::string& name, const DecodeProfile& profile, bool head) {
    if (!head && profile.gemv)
        return name == (profile.dtype == at::kHalf ? "llama_gemv" : "llama_gemv_wint8");
    // A fused head's weight dtype need not match the decoder layers'.
    const at::ScalarType dtype = head ? profile.head_dtype : profile.dtype;
    if (dtype != at::kHalf && dtype != at::kChar && dtype != at::kByte)
        return false;
    using namespace rpu_pl_tiling;
    const auto weight = dtype == at::kByte ? WeightType::Int4 :
        dtype == at::kHalf ? WeightType::Fp16 : WeightType::Int8;
    for (const auto& candidate : rpu_pl_tiling::detail::parallel_candidates) {
        if (candidate.weight == weight && candidate.acc32 == profile.acc32 &&
            name == candidate.name) return true;
    }
    return false;
}

bool bind_decode_patches(const std::vector<DecodeKernelSlot>& kernels,
                         std::vector<RegisterPatch>& patches,
                         std::vector<RegisterPatch>& head_patches,
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
    head_patches.clear();
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
                if (!take(profile.tts_predictor ? "llama_rope" : "llama_mrope_interleave")) return false;
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
        const bool fused_o = kernels[cursor].name == "qwen3vl_o_ring_norm_w8_m1_h2560_v2";
        if (combined_o && *combined_o != fused_o) return false;
        combined_o = fused_o;
        if (fused_o) {
            if (profile.layers != 36 || profile.dtype != at::kChar || !profile.gemv ||
                !take("qwen3vl_o_ring_norm_w8_m1_h2560_v2")) return false;
        } else if (!linear() || !take("llm_all_reduce_residual_nopace") || !norm()) return false;
        if (profile.fused_mlp) {
            if (!take("qwen3vl_gateup_swiglu_w8_m1_v2")) return false;
        } else if (profile.tts_predictor && profile.dtype == at::kHalf && !profile.acc32) {
            if (!linear() || !linear() || !take("llama_silu_mul")) return false;
        } else if (profile.layers == 36 && profile.dtype == at::kChar && profile.gemv) {
            if (!linear() || !linear() || !take("silu_mul")) return false;
        } else if (!linear() || !take("unary") || !linear() || !take("binary_sameshape")) return false;
        if (!linear() || !take("llm_all_reduce_residual_nopace") ||
            (!profile.tts_predictor && layer < 3 && !take("binary_sameshape"))) return false;
    }
    if (!norm()) return false;
    if (profile.head_bank) {
        // The bank's selected head is the one trailing projection. Its weight
        // and scale addresses live in registers 2/3 and 20/21 as `addr >> 8`
        // u16 pairs (set_parallel_linear_regs), so a selection reaches this
        // retained stream as four patches rather than a rebuild.
        if (cursor + 1 != kernels.size() ||
            !linear_name(kernels[cursor].name, profile, true)) return false;
        for (uint32_t reg : {2u, 3u, 20u, 21u})
            head_patches.push_back({kernels[cursor].kernel, reg, 0, 2});
        return true;
    }
    return profile.tts_predictor ? cursor == kernels.size() :
        cursor + 1 == kernels.size() && linear_name(kernels[cursor].name, profile, true);
}

void update_decode_patches(std::vector<RegisterPatch>& patches, int64_t position,
                           int64_t cache_capacity, int layers) {
    TORCH_CHECK((layers == 5 || layers == 28 || layers == 36) &&
                    (patches.size() == size_t(layers * 10) || patches.size() == size_t(layers * 14) ||
                     patches.size() == size_t(layers * 16)) &&
                    cache_capacity > 0 && cache_capacity % 16 == 0 &&
                    position >= 0 && position < cache_capacity &&
                    position < kCheckedDecodeMaxKeyRows,
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

void update_head_patches(std::vector<RegisterPatch>& patches,
                         const at::Tensor& weight, const at::Tensor& scale) {
    TORCH_CHECK(patches.size() == 4, "unsealed fused head patch recipe");
    TORCH_CHECK(weight.defined() && scale.defined(),
                "fused head patch requires an installed head");
    const uint64_t w_addr = RpuGetDevAddr(weight.data_ptr());
    const uint64_t s_addr = RpuGetDevAddr(scale.data_ptr());
    // The ABI carries both as `addr >> 8` in a u16 pair, so the low 8 bits have
    // nowhere to go. The recording launcher shifts without checking; a patch is
    // the one place that can still refuse an address it would silently truncate.
    TORCH_CHECK((w_addr & 0xff) == 0 && (s_addr & 0xff) == 0,
                "fused head payload is not 256-byte aligned");
    const uint64_t w = w_addr >> 8, s = s_addr >> 8;
    TORCH_CHECK(w <= UINT32_MAX && s <= UINT32_MAX,
                "fused head address exceeds the 32-bit register ABI");
    patches[0].value = uint16_t(w & 0xffff);
    patches[1].value = uint16_t((w >> 16) & 0xffff);
    patches[2].value = uint16_t(s & 0xffff);
    patches[3].value = uint16_t((s >> 16) & 0xffff);
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
    TORCH_CHECK(position >= 0 && position < cache_capacity && position < kCheckedDecodeMaxKeyRows && position < cos.size(0) &&
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
        // Kept out of `patches` so update_decode_patches' layer-strided seal
        // stays exact; empty unless a head bank is installed.
        std::vector<RegisterPatch> head_patches;
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
        p.tts_predictor = model.num_layers() == 5;
        p.head_bank = model.lm_head_bank_index_ >= 0;
        p.head_dtype = model.lm_head_w_.defined()
            ? model.lm_head_w_.scalar_type() : p.dtype;
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
        for (const auto* t : {&model.cos_, &model.sin_, &model.final_norm_w_})
            if (!fn(*t, true)) return false;
        if (!p.tts_predictor && !fn(model.lm_head_w_, true)) return false;
        if (p.head_bank)
            for (size_t i = 0; i < model.lm_head_bank_w_.size(); ++i)
                if (!fn(model.lm_head_bank_w_[i], true) ||
                    !fn(model.lm_head_bank_scale_[i], true)) return false;
        if (p.packed_qkv && !fn(model.qwen3vl_4b_qkv_scale_bank_, true)) return false;
        if (!p.tts_predictor && p.dtype != at::kHalf && !fn(model.lm_head_w_scale_, true)) return false;
        return p.tts_predictor || fn(model.position_ids_keepalive_, false);
    }

    static bool projection_shape(const at::Tensor& w, const at::Tensor& scale,
                                  at::ScalarType dtype, int64_t n, int64_t k, bool row) {
        int64_t physical_n = n;
        if (dtype == at::kByte) {
            if (n <= 0 || k <= 0 || (row ? k % 8 : n % 8)) return false;
            const int64_t local_n = row ? n : n / 8, local_k = row ? k / 8 : k;
            if (local_n % 16 || local_k % 32) return false;
            // New W4 storage pads each TP shard to N32; n remains the logical
            // output width (including the unpadded vocabulary for the head).
            physical_n = ((local_n + 31) / 32) * 32 * (row ? 1 : 8);
        }
        if (!tensor_shape(w, dtype, {physical_n, dtype == at::kByte ? k / 2 : k})) return false;
        if (dtype == at::kHalf) return !scale.defined();
        if (dtype == at::kChar) return tensor_shape(scale, at::kHalf, {n});
        if (dtype != at::kByte || !scale.defined() || scale.scalar_type() != at::kHalf ||
            scale.dim() != 2 || scale.size(0) != 32) return false;
        const int64_t local_n = row ? n : n / 8, local_k = row ? k / 8 : k;
        // Scale storage still uses controller-striped N64/G4 blocks.
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
            // Every installed THW owner retains initialized final-row overfetch.
            (p.tts_predictor || readable_elements(model.position_ids_keepalive_,
                (uint64_t(model.position_ids_keepalive_.size(0)) - 1) * 3 + 8));
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
        if (!profile(model).tts_predictor) rpu_ddr_flush(model.lm_head_w_.data_ptr());
        if (!profile(model).tts_predictor && profile(model).dtype != at::kHalf)
            rpu_ddr_flush(model.lm_head_w_scale_.data_ptr());
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
    // The Predictor admits either no head at all (the head stays an eager op in
    // Python) or a bank-installed int8 head, whose selected entry is patched into
    // the retained stream per step. Nothing else may carry a fused head here.
    const bool tts_head_bank = lm_head_bank_index_ >= 0 && fuse_lm_head_ &&
        lm_head_w_.defined() && lm_head_w_scale_.defined() &&
        lm_head_w_.scalar_type() == at::kChar &&
        lm_head_w_scale_.scalar_type() == at::kHalf &&
        !fuse_prefill_last_lm_head_ && vocab_size_ > 0 &&
        vocab_size_ % lm_head_tp() == 0;
    const bool tts_headless = !fuse_lm_head_ && !lm_head_w_.defined() &&
        !lm_head_w_scale_.defined() && lm_head_bank_index_ < 0;
    const bool tts_predictor = fast_replay_skip_layer_loop_ && num_layers() == 5 &&
        hidden_size() == 1024 && intermediate_size() == 3072 && num_q_heads() == 16 &&
        !has_mrope_ && (tts_headless || tts_head_bank) &&
        deepstack_lang_layers_.empty() && rope_position_base_ < 0 &&
        !layer_weights_.empty() && layer_weights_.front().q_w.defined() &&
        (layer_weights_.front().q_w.scalar_type() == at::kHalf ||
         layer_weights_.front().q_w.scalar_type() == at::kChar ||
         layer_weights_.front().q_w.scalar_type() == at::kByte);
    const bool ordinary = qwen3vl_ordinary_resident_prefill_profile_ || qwen3vl_4b_w8_profile_ || exact_8b || tts_predictor;
    const bool geometry = (num_layers() == 28 && hidden_size() == 2048 && intermediate_size() == 6144 && num_q_heads() == 16) ||
        (num_layers() == 36 && hidden_size() == 2560 && intermediate_size() == 9728 && num_q_heads() == 32) || exact_8b || tts_predictor;
    const bool envelope = tts_predictor || (chunk_envelope().max_kv_len > 0 &&
        chunk_envelope().chunk == (exact_8b ? 128 : num_layers() == 28 ? 320 : 256));
    if (typeid(*this) != typeid(CausalDecoderModel) || !RpuKernelGraph::has_active() ||
        !ordinary || !geometry || !is_causal || mask.has_value() || partial_mrope_active_ ||
        decoder_mixed_precision_ || adarms_ || planning_mode_ != CausalDecoderPlanningMode::ORDINARY ||
        qwen3vl_pooler_z1_dispatch_ || qwen3vl_multiview_text_composite_dispatch_ ||
        !has_qk_norm_ || (!tts_predictor && !has_mrope_) || num_kv_heads() != 8 || head_dim() != 128 ||
        layer_weights_.size() != size_t(num_layers()) ||
        num_cores() != 8 || attn_tp() != 8 || mlp_tp() != 8 || lm_head_tp() != 8 ||
        (!tts_predictor && (!fuse_lm_head_ || vocab_size_ != 151936)) || lm_head_exact_candidates_ || reuse_lm_head_output_ ||
        !use_silu_ || nvfp4_ || has_qkv_bias_ || norm_slab_active() ||
        equal_two_prefill_ || configured_chunk_size_cap_ ||
        adarms_mutable_ || adarms_unroll_ || adarms_schedule_select_ || adarms_fused_bcast_enabled_ ||
        wall_prefill_staged_mlp_ || wall_prefill_mixed_requested_ || wall_prefill_staged_requested_ ||
        sdpa_kernel_ != SdpaKernelType::FLASH_ATTN_SPM ||
        (!tts_predictor && (mrope_section_ != std::vector<int32_t>({24, 20, 20}) ||
         deepstack_lang_layers_ != std::vector<int64_t>({0, 1, 2}))) ||
        // Preserve per-profile chunk policy; actual owners bound decode length.
        !envelope ||
        get_debug_export() || descriptor.empty() || k.size() != size_t(num_layers()) || v.size() != size_t(num_layers()) ||
        !tensor_shape(hidden, at::kHalf, {1, 1, hidden_size()})) return false;
    auto& graph = RpuKernelGraph::active();
    if ((graph.state() != RpuKernelGraph::State::RECORDING &&
         graph.state() != RpuKernelGraph::State::REPLAYING) ||
        graph.kernel_register_census_active() || graph.runtime_policy().fmb_fast_replay ||
        graph.runtime_policy().fmb_deep_fast_replay) return false;
    const auto profile = Qwen3VlDecodeReplayState::profile(*this);
    // The exact TTS Predictor uses the checked tiled grammar for both ACC modes.
    // Other quantized ACC32 profiles retain generic recorded-Graph execution.
    if (!tts_predictor && profile.acc32 && profile.dtype != at::kHalf) return false;
    if ((!tts_predictor && (!Qwen3VlDecodeReplayState::projection_shape(lm_head_w_, lm_head_w_scale_,
            profile.dtype, 151936, hidden_size(), false) ||
        !cos_.defined() || cos_.dim() != 2 || cos_.size(0) <= 0 ||
        !position_ids_keepalive_.defined() || position_ids_keepalive_.dim() != 2 ||
        position_ids_keepalive_.size(0) <= 0 || position_ids_keepalive_.size(0) > cos_.size(0) ||
        !tensor_shape(position_ids_keepalive_, at::kInt, {position_ids_keepalive_.size(0), 3}))) ||
        !cos_.defined() || cos_.dim() != 2 || cos_.size(1) != 64 ||
        cos_.scalar_type() != at::kHalf || !tensor_shape(sin_, at::kHalf, cos_.sizes())) return false;
    if (profile.head_bank &&
        !Qwen3VlDecodeReplayState::projection_shape(lm_head_w_, lm_head_w_scale_,
            at::kChar, vocab_size_, hidden_size(), false)) return false;
    const auto cache_capacity = decode_cache_capacity(k, v, num_layers());
    if (!cache_capacity) return false;
    if (tts_predictor) {
        if (*cache_capacity != 32 || cos_.size(0) != 32) return false;
        TORCH_CHECK(position >= 0 && position < 32, "TTS Predictor checked decode position exceeds cache");
    } else {
        check_decode_position(position_ids_keepalive_, cos_, position, *cache_capacity);
    }
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
    if (!recipe.head_patches.empty())
        update_head_patches(recipe.head_patches, lm_head_w_, lm_head_w_scale_);
    Qwen3VlDecodeReplayState::visibility(*this, state.k, state.v);
    // No fallback is allowed after this boundary. The existing Graph API owns
    // typed/census/semantic validation and register patch application.
    if (recipe.head_patches.empty()) {
        graph.skip_op_stream_with_patches(recipe.patches, {});
    } else {
        std::vector<RegisterPatch> applied = recipe.patches;
        applied.insert(applied.end(), recipe.head_patches.begin(),
                       recipe.head_patches.end());
        graph.skip_op_stream_with_patches(applied, {});
    }
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
            bind_decode_patches(kernels, next->patches, next->head_patches,
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
