"""Small-image Vision admission and reused payload tail bounds (no device)."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
VISION = (ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text()


def _body(marker, source=VISION):
    begin = source.index(marker)
    brace = source.index("{", begin)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[begin:end]


def test_small_image_native_admission_rejects_other_layouts(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native admission check")
    # Execute the actual C++ predicate with metadata-only tensors. In particular,
    # do not accidentally admit unaligned 3x324 K/V, old I3456 FP16 storage, or
    # the mixed block-17 Down owner used by the larger-image profile.
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    cpp = tmp_path / "small_vision.cpp"
    cpp.write_text(
        """
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>
#define TORCH_CHECK(condition, ...) assert(condition)
constexpr int64_t SDPA_MAX_VLM_SIZE = 400;
template<typename T> T CeilDiv(T n, T d) { return (n + d - 1) / d; }
"""
        + _body("inline uint32_t GetMhaAttnUnivVlmNeed(", helpers)
        + _body("inline bool sdpa_by_mha_spm_is_valid(", helpers)
        + decls[decls.index("enum RpuRmsNormCapability"):
                decls.index("enum class RpuMixedRmsNormSpmRoute")]
        + _body("inline bool rpu_rmsnorm_capability_valid(", decls)
        + _body("inline RpuRmsNormSpmRoute rpu_resolve_rmsnorm_spm_route(", decls)
        + """
namespace at {
enum Dtype { kHalf, kChar, kOther };
struct Tensor {
    Dtype dtype = kHalf;
    bool present = false;
    Dtype scalar_type() const { return dtype; }
    bool defined() const { return present; }
};
}
enum class VisionSchedule { SISC, MISC };
enum { QWEN25_WINDOW_DDR_ATTN_SITE, QWEN25_MISC_DDR_ATTN_SITE,
       QWEN25_STANDARD_DDR_ATTN_SITE };
constexpr int64_t NUM_CORES = 8, QWEN25VL_VISION_SDPA_QUERY_CHUNK = 144,
                  QWEN25VL_VISION_MAX_KEEPALIVE_SEQ = 4096;
struct ChunkInfo { int64_t idx, len; };
struct FmbThreeStageChunkPlan {
    struct Compute { std::vector<ChunkInfo> chunks{{0, 768}}; } compute;
};
struct Model {
    struct Sorting { bool vision_row_reuse = true, silu_mul = true; uint32_t fp16_transfer_mask = 1; } wall_sorting_;
    bool linear_acc32_ = false;
    bool generic = true, mixed_precision_ = false, per_window_sdpa_enabled_ = false, z1_bound_ = false;
    int64_t layers = 32, hidden = 1280, heads = 16, dim = 80, intermediate = 3584;
    int64_t max_keepalive_seq_ = 4096;
    int64_t image_batch_count_ = 3, current_num_patches_ = 768, valid_patches_per_image_ = 0;
    bool window_mask_present_ = true;
    uint32_t small_rmsnorm_capability_ = RPU_RMSNORM_CAP_BASE;
    bool small_owner_ring_norm_available_ = false;
    bool small_down_owner_norm_available_ = false;
    double eps_ = 1e-6;
    at::Tensor merger_ln_q_w_{at::kHalf, true};
    VisionSchedule schedule_ = VisionSchedule::MISC;
    std::vector<int64_t> fullatt_block_indexes_{7, 15, 23, 31}, cu_window_;
    bool debug = false;
    bool get_debug_export() const { return debug; }
    // This legacy fixture models an expansion without the optional R47 payload.
    // Its real admission and COMPLETE route are tested in test_wall_base_vision_qk_rope.py.
    bool small_image_qk_rope_enabled(int64_t) const { return false; }
    bool small_image_ragged_sdpa_enabled(int64_t) const { return false; }
    bool is_generic() const { return generic; }
    bool merger_active() const { return true; }
    int64_t num_layers() const { return layers; }
    int64_t hidden_size() const { return hidden; }
    int64_t num_q_heads() const { return heads; }
    int64_t num_kv_heads() const { return heads; }
    int64_t head_dim() const { return dim; }
    int64_t intermediate_size() const { return intermediate; }
"""
        + _body("    struct LayerWeights {") + ";\n"
        + _body("    struct LayerBiasNorm {") + ";\n"
        + "std::vector<LayerWeights> layer_weights_ = std::vector<LayerWeights>(32);\n"
        + "std::vector<LayerBiasNorm> layer_bias_norm_ = std::vector<LayerBiasNorm>(32);\n"
        + _body("    bool padded_pair_enabled() const")
        + _body("    std::vector<int64_t> sorting_schedule_arguments() const")
        + _body("    bool small_image_row_weight_outer_enabled(int64_t rows) const")
        + _body("    bool small_image_fp16_row_profile(int64_t rows) const")
        + _body("    bool small_image_gate_up_enabled(int64_t rows) const")
        + _body("    bool small_image_native_window_shape() const")
        + _body("    bool small_image_native_window_enabled() const")
        + _body("    bool small_image_single_raw_query_enabled() const")
        + _body("    bool small_image_raw_kv_only_enabled() const")
        + _body("    bool small_image_resident_mask_enabled() const")
        + _body("    RpuRmsNormSpmRoute small_image_rmsnorm_route(int64_t rows) const")
        + _body("    bool small_image_owner_ring_norm_enabled(int64_t rows) const")
        + _body("    std::vector<int64_t> small_image_owner_ring_norm_arguments(int64_t rows) const")
        + _body("    bool small_image_down_owner_norm_enabled(int64_t rows) const")
        + _body("    int64_t small_image_schedule_flags() const")
        + _body("    struct GenericAttentionGeometry {") + ";\n"
        + _body("    bool generic_attention_layer_is_full(int64_t layer) const")
        + _body("    int64_t generic_shared_attention_mask_domain() const")
        + _body("    std::vector<GenericAttentionGeometry>\n    generic_attention_geometries(")
        + _body("    std::vector<int64_t> generic_attention_arguments(")
        + _body("    bool generic_attention_raw_valid(")
        + _body("    bool generic_attention_site_raw_valid(")
        + """
};
int main() {
    for (const bool fp16 : {false, true}) {
        Model m;
        m.wall_sorting_.fp16_transfer_mask = fp16 ? 1 : 0;
        for (auto &w : m.layer_weights_) {
            for (auto *t : {&w.q_w, &w.k_w, &w.v_w, &w.o_w,
                           &w.gate_w, &w.up_w, &w.down_w}) {
                t->dtype = fp16 ? at::kHalf : at::kChar; t->present = true;
            }
            w.o_ws.present = w.down_ws.present = !fp16;
            w.gate_ws.present = w.up_ws.present = !fp16;
        }
        assert(m.small_image_row_weight_outer_enabled(768));
        assert(!m.small_image_gate_up_enabled(768));
        assert(m.small_image_resident_mask_enabled());
        assert(m.small_image_schedule_flags() == 1);
        auto norm = m;
        norm.small_rmsnorm_capability_ = RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V32;
        for (auto& bn : norm.layer_bias_norm_)
            bn.norm1_w.present = bn.norm2_w.present = true;
        assert(norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::V32_GRID8);
        assert(norm.small_image_schedule_flags() == (1 | 128));
        for (const int64_t wrong_rows : {0, 256, 324, 336, 576, 648, 672, 972, 1728})
            assert(norm.small_image_rmsnorm_route(wrong_rows) == RpuRmsNormSpmRoute::BASE);
        auto invalid_norm = norm;
        invalid_norm.small_rmsnorm_capability_ = RPU_RMSNORM_CAP_BASE;
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.wall_sorting_.silu_mul = false;
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.generic = false;
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.debug = true;
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.mixed_precision_ = true;
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.z1_bound_ = true;
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.layer_bias_norm_.pop_back();
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.layer_bias_norm_[17].norm1_w.dtype = at::kOther;
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.merger_ln_q_w_.dtype = at::kOther;
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.layer_weights_.clear();
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.layer_weights_[31].down_w.dtype = at::kOther;
        assert(invalid_norm.small_image_rmsnorm_route(768) == RpuRmsNormSpmRoute::BASE);
        // Execute the canonical payload's eight workgroup, four-row loop
        // envelope: physical M672 is 21 groups/warp and M768 is 24. No
        // rounding, tail split, SPM padding or additional scratch is used.
        for (const int64_t rows : {672, 768}) {
            const int64_t reg3 = rows / 8;
            assert(reg3 % 4 == 0 && 8 * (reg3 / 4) * 4 == rows);
            assert(7 * reg3 + (reg3 / 4 - 1) * 4 + 3 == rows - 1);
            assert(rows * 1280 * 2 == (rows == 672 ? 1720320 : 1966080));
        }
        const auto dense_sites = m.generic_attention_geometries({});
        assert(dense_sites.size() == 3);
        for (const auto &site : dense_sites)
            assert(site.batch == 1 && site.seq_q == 256 && site.seq_k == 256 && site.mask_domain == 17);
        auto native = m;
        native.cu_window_ = {0, 64, 128, 192, 256};
        assert(native.small_image_native_window_enabled());
        assert(!native.small_image_resident_mask_enabled());
        assert(native.small_image_raw_kv_only_enabled());
        assert(native.small_image_schedule_flags() == 6);
        native.debug = true;
        assert(!native.small_image_raw_kv_only_enabled());
        assert(native.small_image_schedule_flags() == 2);
        native.debug = false;
        const auto sites = native.generic_attention_geometries({});
        assert(sites.size() == 2);
        assert(sites[0].site_id == QWEN25_WINDOW_DDR_ATTN_SITE && sites[0].batch == 12 && sites[0].seq_k == 64);
        assert(sites[1].site_id == QWEN25_MISC_DDR_ATTN_SITE && sites[1].batch == 3 && sites[1].seq_k == 256);
        for (const auto &site : sites) {
            assert(site.invocation == 0 && site.seq_q == site.seq_k && site.mask_domain == 1);
            assert(site.batch * site.seq_k == 768 && site.seq_k % 16 == 0);
            // Each core owns two heads. Both layouts cover exactly the packed
            // Q/K/V/output and transpose allocation, with no stride padding.
            assert(site.batch * site.seq_k * 2 * 80 * 2 == 245760);
            assert(native.generic_attention_arguments(site) == std::vector<int64_t>(
                {site.batch, site.seq_q, site.seq_k, 16, 16, 80, 8, 1,
                 static_cast<int64_t>(VisionSchedule::MISC), 0}));
        }
        for (int64_t layer = 0; layer < 32; ++layer)
            assert(native.generic_attention_layer_is_full(layer) == ((layer + 1) % 8 == 0));
        // Two real 224 images use the same four-window layout per image,
        // with exactly 512 packed rows and no third-image placeholders.
        auto two = norm;
        two.image_batch_count_ = 2;
        two.current_num_patches_ = 512;
        two.cu_window_ = {0, 64, 128, 192, 256};
        assert(two.small_image_row_weight_outer_enabled(512));
        assert(!two.small_image_row_weight_outer_enabled(768));
        assert(!two.small_image_gate_up_enabled(512));
        assert(two.small_image_native_window_enabled());
        assert(two.small_image_raw_kv_only_enabled());
        assert(two.small_image_rmsnorm_route(512) == RpuRmsNormSpmRoute::V32_GRID8);
        assert(!two.small_image_owner_ring_norm_enabled(512));
        assert(!two.small_image_down_owner_norm_enabled(512));
        FmbThreeStageChunkPlan two_plan;
        two_plan.compute.chunks = {{0, 512}};
        const auto two_sites = two.generic_attention_geometries(two_plan);
        assert(two_sites.size() == 2);
        assert(two_sites[0].batch == 8 && two_sites[0].seq_q == 64);
        assert(two_sites[1].batch == 2 && two_sites[1].seq_q == 256);
        for (const auto& site : two_sites) {
            assert(site.batch * site.seq_k == 512 && site.mask_domain == 1);
            assert(sdpa_by_mha_spm_is_valid(site.batch, site.seq_q, site.seq_k,
                                          16, 16, 80, 8, 0));
            assert(two.generic_attention_arguments(site) == std::vector<int64_t>(
                {site.batch, site.seq_q, site.seq_k, 16, 16, 80, 8, 1,
                 static_cast<int64_t>(VisionSchedule::MISC), 0}));
        }
        two.cu_window_.clear();
        assert(!two.small_image_native_window_enabled());
        assert(two.small_image_resident_mask_enabled());
        two.current_num_patches_ = 768;
        assert(!two.small_image_row_weight_outer_enabled(512));
        assert(!two.small_image_resident_mask_enabled());
        two.image_batch_count_ = 4; two.current_num_patches_ = 1024;
        assert(!two.small_image_native_window_shape());
        assert(!two.small_image_row_weight_outer_enabled(1024));
        native.cu_window_ = {0, 128, 256};
        assert(!native.small_image_native_window_enabled());
        assert(!native.small_image_raw_kv_only_enabled());
        for (const int64_t rows : {256, 324, 336, 512, 648, 972, 1728})
            assert(!m.small_image_row_weight_outer_enabled(rows));
        auto bad = m; bad.current_num_patches_ = 972;
        assert(!bad.small_image_row_weight_outer_enabled(324));
        assert(!bad.small_image_resident_mask_enabled());
        bad = m; bad.layer_weights_.clear();
        assert(!bad.small_image_row_weight_outer_enabled(768));
        assert(!bad.small_image_native_window_enabled());
        assert(!bad.small_image_resident_mask_enabled());
        bad = m; bad.layer_weights_[0].q_w.dtype = at::kOther;
        assert(!bad.small_image_row_weight_outer_enabled(768));
        assert(!bad.small_image_resident_mask_enabled());
        bad = m; bad.window_mask_present_ = false;
        assert(!bad.small_image_native_window_shape());
        assert(!bad.small_image_resident_mask_enabled());
        bad = m; bad.fullatt_block_indexes_ = {7, 15, 23};
        assert(!bad.small_image_native_window_shape());
        assert(!bad.small_image_resident_mask_enabled());
        bad = m; bad.intermediate = 3456;
        assert(!bad.small_image_row_weight_outer_enabled(768));
        bad = m; bad.mixed_precision_ = true;
        assert(!bad.small_image_row_weight_outer_enabled(768));
        bad = m; bad.wall_sorting_.vision_row_reuse = false;
        assert(!bad.small_image_row_weight_outer_enabled(768));
        assert(!bad.small_image_gate_up_enabled(768));
        assert(bad.small_image_resident_mask_enabled());
        bad = m; bad.layer_weights_[17].down_w.dtype = fp16 ? at::kChar : at::kHalf;
        assert(!bad.small_image_row_weight_outer_enabled(768));
        bad = m; bad.layer_weights_[31].o_ws.present = fp16;
        assert(!bad.small_image_row_weight_outer_enabled(768));
        if (fp16) {
            bad = m; bad.wall_sorting_.fp16_transfer_mask = 0;
            assert(!bad.small_image_row_weight_outer_enabled(768));
        }
        m.image_batch_count_ = 1; m.current_num_patches_ = 324;
        assert(m.small_image_row_weight_outer_enabled(324));
        assert(m.small_image_gate_up_enabled(324) == !fp16);
        assert(!m.small_image_row_weight_outer_enabled(336));
        assert(m.small_image_resident_mask_enabled());
        assert(!m.small_image_native_window_shape());
        assert(m.small_image_raw_kv_only_enabled());
        assert(m.small_image_schedule_flags() == 5);
        m.schedule_ = VisionSchedule::SISC;
        FmbThreeStageChunkPlan single;
        single.compute.chunks = {{0, 324}};
        assert(m.small_image_single_raw_query_enabled());
        assert(m.small_image_schedule_flags() == 37);
        const auto whole = m.generic_attention_geometries(single);
        assert(whole.size() == 1);
        const auto& site = whole.front();
        assert(site.site_id == QWEN25_STANDARD_DDR_ATTN_SITE);
        assert(site.invocation == 0 && site.batch == 1);
        assert(site.seq_q == 324 && site.seq_k == 324 && site.mask_domain == 17);
        assert(m.generic_attention_arguments(site) == std::vector<int64_t>(
            {1, 324, 324, 16, 16, 80, 8, 17, static_cast<int64_t>(VisionSchedule::SISC), 0}));
        assert(m.generic_attention_raw_valid(site));
        assert(m.generic_attention_site_raw_valid(single, site.site_id));
        for (const int mask : {0, 4})
            assert(sdpa_by_mha_spm_is_valid(1, 324, 324, 16, 16, 80, 8, mask));
        assert(!sdpa_by_mha_spm_is_valid(1, 324, 324, 16, 16, 80, 8, 1));
        assert(!sdpa_by_mha_spm_is_valid(2, 324, 324, 16, 16, 80, 8, 0));
        assert(GetMhaAttnUnivVlmNeed(5, 5, 16, 5, 5, 5, 0) == 213);
        assert(GetMhaAttnUnivVlmNeed(5, 5, 16, 5, 5, 5, 4) == 268);
        // The real raw tile spans 80/80/80/80/4 query rows, with no new
        // SPM scratch. The old mask covers all real rows with a 336-key stride.
        assert(CeilDiv(int64_t{324}, int64_t{80}) == 5);
        assert(324 - 4 * 80 == 4);
        assert(324 * 2 * 80 * 2 <= 107520);
        assert(CeilDiv(int64_t{324}, int64_t{16}) * 16 * 2 * 80 * 2 == 107520);
        assert(324 * 336 * 2 <= 217856);
        bad = m; bad.debug = true;
        assert(!bad.small_image_single_raw_query_enabled());
        const auto strips = bad.generic_attention_geometries(single);
        assert(strips.size() == 3);
        for (int i = 0; i < 3; ++i) {
            assert(strips[i].site_id == QWEN25_STANDARD_DDR_ATTN_SITE);
            assert(strips[i].invocation == i && strips[i].batch == 1);
            assert(strips[i].seq_q == (i == 2 ? 36 : 144));
            assert(strips[i].seq_k == 324 && strips[i].mask_domain == 17);
        }
        bad = m; bad.schedule_ = VisionSchedule::MISC;
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.per_window_sdpa_enabled_ = true;
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.window_mask_present_ = false;
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.fullatt_block_indexes_ = {7, 15, 23};
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.layer_weights_[17].down_w.dtype = fp16 ? at::kChar : at::kHalf;
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.layer_weights_.clear();
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.debug = true;
        assert(!bad.small_image_raw_kv_only_enabled());
        assert(!bad.small_image_single_raw_query_enabled());
        assert(bad.small_image_schedule_flags() == 1);
        bad = m; bad.current_num_patches_ = 336;
        assert(!bad.small_image_raw_kv_only_enabled());
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.image_batch_count_ = 3;
        assert(!bad.small_image_raw_kv_only_enabled());
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.mixed_precision_ = true;
        assert(!bad.small_image_raw_kv_only_enabled());
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.z1_bound_ = true;
        assert(!bad.small_image_raw_kv_only_enabled());
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.intermediate = 3456;
        bad.wall_sorting_.vision_row_reuse = false;
        bad.wall_sorting_.fp16_transfer_mask = 6;
        assert(bad.small_image_resident_mask_enabled() == fp16);
        assert(bad.small_image_raw_kv_only_enabled() == fp16);
        assert(bad.small_image_single_raw_query_enabled() == fp16);
        assert(bad.small_image_schedule_flags() == (fp16 ? 37 : 0));
        bad = m; bad.wall_sorting_.silu_mul = false;
        assert(!bad.small_image_gate_up_enabled(324));
        assert(!bad.small_image_resident_mask_enabled());
        assert(!bad.small_image_raw_kv_only_enabled());
        assert(!bad.small_image_single_raw_query_enabled());
        bad = m; bad.wall_sorting_.vision_row_reuse = false;
        assert(!bad.small_image_gate_up_enabled(324));
        bad = m; bad.layer_weights_[0].gate_ws.present = false;
        assert(!bad.small_image_gate_up_enabled(324));
        bad = m; bad.layer_weights_[31].up_ws.present = false;
        assert(!bad.small_image_gate_up_enabled(324));
        auto pair = m;
        pair.image_batch_count_ = 2; pair.current_num_patches_ = 672;
        pair.schedule_ = VisionSchedule::MISC;
        pair.valid_patches_per_image_ = 324;
        assert(pair.padded_pair_enabled());
        assert(pair.small_image_raw_kv_only_enabled());
        assert(pair.small_image_resident_mask_enabled());
        assert(pair.small_image_schedule_flags() == 69);
        norm.image_batch_count_ = 2; norm.current_num_patches_ = 672;
        norm.valid_patches_per_image_ = 324;
        assert(norm.small_image_rmsnorm_route(672) == RpuRmsNormSpmRoute::V32_GRID8);
        assert(norm.small_image_rmsnorm_route(648) == RpuRmsNormSpmRoute::BASE);
        assert(norm.small_image_schedule_flags() == (69 | 128));
        assert(!norm.small_image_owner_ring_norm_enabled(672));
        auto fused = norm; fused.small_owner_ring_norm_available_ = true;
        assert(fused.small_image_owner_ring_norm_enabled(672));
        assert(fused.small_image_schedule_flags() == (69 | 128 | 256));
        assert(fused.small_image_owner_ring_norm_arguments(672) ==
            std::vector<int64_t>({1,672,1280,8,84,12,1,1}));
        for (int64_t wrong : {324,336,648,768})
            assert(!fused.small_image_owner_ring_norm_enabled(wrong));
        assert(!fused.small_image_down_owner_norm_enabled(672));
        auto down = fused; down.small_down_owner_norm_available_ = true;
        assert(down.small_image_down_owner_norm_enabled(672));
        assert(down.small_image_schedule_flags() == (69 | 128 | 768));
        if (fp16) {
            // Every installed layer/projection must be defined Half and have no
            // quantization scales. A mixed or incomplete last layer cannot
            // inherit an activation-only payload from the first layer's dtype.
            for (int layer=0; layer<32; ++layer) for (int field=0; field<7; ++field) {
                for (int mutation=0; mutation<3; ++mutation) {
                    auto bad=down; auto& w=bad.layer_weights_[layer];
                    std::vector<at::Tensor*> weights{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w};
                    std::vector<at::Tensor*> scales{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws};
                    if(mutation==0) weights[field]->present=false;
                    if(mutation==1) weights[field]->dtype=at::kOther;
                    if(mutation==2) scales[field]->present=true;
                    assert(!bad.small_image_owner_ring_norm_enabled(672));
                    assert(!bad.small_image_down_owner_norm_enabled(672));
                }
            }
            for (int mutation=0;mutation<4;++mutation) {
                auto bad=down;
                if(mutation==0) bad.wall_sorting_.vision_row_reuse=false;
                if(mutation==1) bad.wall_sorting_.fp16_transfer_mask=6;
                if(mutation==2) bad.intermediate=3456;
                if(mutation==3) bad.wall_sorting_.silu_mul=false;
                assert(!bad.small_image_owner_ring_norm_enabled(672));
                assert(!bad.small_image_down_owner_norm_enabled(672));
            }
        }
        auto down_off = down; down_off.small_owner_ring_norm_available_ = false;
        assert(!down_off.small_image_down_owner_norm_enabled(672));
        for (int64_t wrong : {324,336,648,768})
            assert(!down.small_image_down_owner_norm_enabled(wrong));
        auto off = fused; off.debug = true;
        assert(!off.small_image_owner_ring_norm_enabled(672));
        off = fused; off.eps_ = 1e-5;
        assert(!off.small_image_owner_ring_norm_enabled(672));
        off = fused; off.small_rmsnorm_capability_ = RPU_RMSNORM_CAP_BASE;
        assert(!off.small_image_owner_ring_norm_enabled(672));
        off = fused; off.layer_weights_[31].down_w.dtype = at::kOther;
        assert(!off.small_image_owner_ring_norm_enabled(672));
        invalid_norm = norm; invalid_norm.valid_patches_per_image_ = 0;
        assert(invalid_norm.small_image_rmsnorm_route(672) == RpuRmsNormSpmRoute::BASE);
        invalid_norm = norm; invalid_norm.current_num_patches_ = 324;
        invalid_norm.image_batch_count_ = 1; invalid_norm.valid_patches_per_image_ = 0;
        assert(invalid_norm.small_image_rmsnorm_route(324) == RpuRmsNormSpmRoute::BASE);
        assert(pair.sorting_schedule_arguments() == std::vector<int64_t>({fp16 ? 1 : 0, 1, 1, 324, 336, 2, 16}));
        assert(pair.small_image_row_weight_outer_enabled(672));
        assert(pair.small_image_gate_up_enabled(672) == !fp16);
        FmbThreeStageChunkPlan pair_plan; pair_plan.compute.chunks = {{0,672}};
        const auto pair_sites = pair.generic_attention_geometries(pair_plan);
        assert(pair_sites.size()==2);
        for (int i=0; i<2; ++i) {
            assert(pair_sites[i].batch==1 && pair_sites[i].seq_q==336 && pair_sites[i].seq_k==336);
            assert(pair_sites[i].invocation==i && pair_sites[i].mask_domain==16);
            assert(pair.generic_attention_raw_valid(pair_sites[i]));
        }
        auto ordinary_pair = pair; ordinary_pair.valid_patches_per_image_=0;
        assert(!ordinary_pair.padded_pair_enabled() && !ordinary_pair.small_image_raw_kv_only_enabled());
        auto rejected = pair; rejected.debug=true; assert(!rejected.padded_pair_enabled());
        rejected=pair; rejected.current_num_patches_=648; assert(!rejected.padded_pair_enabled());
        rejected=pair; rejected.valid_patches_per_image_=320; assert(!rejected.padded_pair_enabled());
        rejected=pair; rejected.image_batch_count_=3; assert(!rejected.padded_pair_enabled());
        rejected=pair; rejected.wall_sorting_.silu_mul=false; assert(!rejected.padded_pair_enabled());
        rejected=pair; rejected.mixed_precision_=true; assert(!rejected.padded_pair_enabled());
        rejected=pair; rejected.layer_weights_[31].down_w.dtype=at::kOther; assert(!rejected.padded_pair_enabled());
        if (fp16) {
            norm.intermediate = 3456; norm.wall_sorting_.vision_row_reuse = false;
            norm.wall_sorting_.fp16_transfer_mask = 6;
            assert(norm.small_image_rmsnorm_route(672) == RpuRmsNormSpmRoute::V32_GRID8);
            pair.intermediate=3456; pair.wall_sorting_.vision_row_reuse=false;
            pair.wall_sorting_.fp16_transfer_mask=6;
            assert(pair.padded_pair_enabled() && pair.small_image_schedule_flags()==69);
            assert(!pair.small_image_row_weight_outer_enabled(672) && !pair.small_image_gate_up_enabled(672));
        }
    }
}
"""
    )
    binary = tmp_path / "small_vision"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(cpp), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_single_raw_query_keeps_complete_authority_and_mask_lifetime():
    domain = _body("    physical_manifest_domain_for_candidate(")
    controlled = domain[domain.index("if (small_image_native_window_enabled()"):
                        domain.index("LayoutContext ddr_layout")]
    assert "small_image_raw_kv_only_enabled()" in controlled
    assert "subclass_spm_kv_by_mha_eligible(plan, raw_layout, position)" in controlled
    assert "!sites.empty() && std::all_of(" in controlled
    assert "generic_attention_site_raw_valid(plan, site.site_id)" in controlled
    declarations = _body("    std::vector<BufferDecl> declare_buffers(")
    assert "if (window_mask_present_ && !small_image_native_window_enabled())" in declarations
    assert "resident_mask ? 1 : 3, resident_mask ? 9 : 4" in declarations
    assert "? Align(cs, int64_t{16}) : cs" in declarations
    assert "small_image_single_raw_query_enabled" not in declarations
    producer = _body("    std::vector<GenericAttentionGeometry>\n    generic_attention_geometries(")
    consumer = _body("    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override")
    for body in (producer, consumer):
        assert "small_image_single_raw_query_enabled()" in body
    assert "const int64_t query_chunk = single_raw_query" in consumer
    assert "TORCH_CHECK(!single_raw_query ||" in consumer
    assert "single-query M324 Vision requires its planned raw-SPM route" in consumer
    assert "site, mask_type," in consumer and "QWEN25_STANDARD_RAW_ATTN_SITE" in consumer
    assert 'q_begin * mask_row_elems * DWIDTH' in consumer
    flags = _body("    int64_t small_image_schedule_flags() const")
    assert "small_image_native_window_enabled() ? 2 : 0" in flags
    assert "small_image_single_raw_query_enabled() ? 32 : 0" in flags
    assert "small_image_schedule_flags()" in _body("    int64_t subclass_layout_hash() const override")


def test_small_vector_norm_uses_physical_rows_and_separate_complete_receipts():
    setter = _body("    void set_execution_routes(")
    assert "rmsnorm_capability_ = rmsnorm_contract.capability" in setter
    assert "exact MIMC, mixed" in setter
    assert "small_rmsnorm_capability_" not in setter
    snapshot = VISION[VISION.index("        small_rmsnorm_capability_ ="):
                      VISION.index("    void set_per_window_sdpa(")]
    assert "wall_sorting_.silu_mul" in snapshot
    assert "!mixed_precision_enabled_ && !exact_mimc_enabled_" in snapshot
    assert "!layer_group_enabled_" in snapshot
    assert "rpu_snapshot_rmsnorm_spm_contract(" in snapshot
    assert '"wall_base_rmsnorm_v32_bitcopy"' in snapshot
    assert "KernelCache::instance().get_kernel(" in snapshot
    assert "RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V32" in snapshot
    assert "invalidate_model_state()" in snapshot
    helper = _body("    RpuRmsNormSpmRoute small_image_rmsnorm_route(")
    assert "rpu_snapshot_rmsnorm_spm_contract" not in helper
    consume = _body("    RpuRmsNormSpmRoute consume_small_image_rmsnorm_route(")
    assert "ctx().has_complete_physical_manifest()" in consume
    assert "FmbRouteFamily::NORMALIZATION" in consume
    assert "{rows, hidden_size(), static_cast<int64_t>(small_rmsnorm_capability_)}" in consume
    producer = _body("    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    assert "small_image_rmsnorm_route(chunk.len)" in producer
    assert "small_image_rmsnorm_route(physical_len)" in producer
    assert "{physical_len, hidden_size()," in producer
    for name in ("QWEN25_SMALL_INPUT_RMSNORM_SITE", "QWEN25_SMALL_POST_RMSNORM_SITE",
                 "QWEN25_SMALL_MERGER_RMSNORM_SITE"):
        assert name in producer
    consumer = _body("    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override")
    assert "QWEN25_SMALL_INPUT_RMSNORM_SITE, seq_len, chunk.idx" in consumer
    assert "QWEN25_SMALL_POST_RMSNORM_SITE, seq_len, chunk.idx" in consumer
    assert "QWEN25_SMALL_MERGER_RMSNORM_SITE, seq, invocation" in VISION
    assert "RpuRmsNormSpmRoute::BASE ? 128 : 0" in _body(
        "    int64_t small_image_schedule_flags() const")
    # Withdrawn single-image window and activation experiments stay absent.
    assert "small_image_single_native_window" not in VISION
    assert "small_image_uniform_cold_shape" not in VISION
    assert "small_image_unpadded_silu_mul_enabled" not in VISION
    assert "(mixed_precision_ || small_image_row_weight_outer_enabled(seq_len));" in consumer


def test_small_row_route_retains_typed_dma_and_complete_descriptor():
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    begin = source.index("void rpu_launch_wall_vision_row_weight_outer_w8a16_spm_kernel(")
    end = source.index("void rpu_launch_linear_spm_to_spm_acc16_kernel(", begin)
    launcher = " ".join(source[begin:end].split())
    assert "sorting.vision_row_reuse && (M == 512 || M == 768 || M == 324 || (M == 672 && sorting.silu_mul))" in launcher
    assert "(M == 1728 || small_vision_rows) && N == 1280 && (K == 1280 || K == 3584)" in launcher
    assert "GraphDdrRegisterRole::Int8ModelWeight" in launcher
    assert "GraphDdrRegisterRole::PerChannelScale" in launcher
    assert "check_spm(input_spm_addr, input_bytes)" in launcher
    assert "check_spm(output_spm_addr, output_bytes)" in launcher
    assert "input_spm_addr + input_bytes <= output_spm_addr" in launcher
    assert "queue->set_broadcast_mode(true)" in launcher
    producer = _body("    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    consumer = _body("    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override")
    assert "small_image_acc16_row_projection_enabled(chunk.len)" in producer
    assert "small_image_row_weight_outer_enabled(seq_len)" in consumer
    assert "Qwen25VisionLinearRoute::WALL_ROW_FULL_LOCAL_K" in producer
    assert "Qwen25VisionLinearRoute::WALL_ROW_FULL_LOCAL_K" in consumer
    for site in ("QWEN25_O_LINEAR_SITE", "QWEN25_DOWN_LINEAR_SITE"):
        assert site in producer and site in consumer


def test_single_base_gate_up_admits_exact_rows_and_dispatches_complete_tiles(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for native launcher admission")
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    begin = source.index("void rpu_launch_wall_vision_gate_up_swiglu_w8a16_spm_kernel(")
    end = source.index("void rpu_launch_wall_qkv_w8a16_multibase_spm_kernel(", begin)
    launcher = source[begin:end]
    guard_start = launcher.index("  const bool small_image =")
    guard_end = launcher.index("  TORCH_CHECK(input_spm_addr", guard_start)
    guard = launcher[guard_start:guard_end]
    dispatch = launcher[launcher.rindex("  if (multibase) {"):launcher.rindex("}")]
    cpp = tmp_path / "gate_up_rows.cpp"
    cpp.write_text("""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("reject"); } while (0)
struct Sorting { bool vision_row_reuse, silu_mul; };
bool admitted(int64_t rows, bool fp16, bool multibase, Sorting sorting) {
    try {
""" + guard + """
        TORCH_CHECK(!fp16 || multibase);
        return true;
    } catch (const std::runtime_error&) { return false; }
}
int64_t CeilDiv(int64_t n, int64_t d) { return (n + d - 1) / d; }
struct Queue {
    int64_t y = 0;
    void enqueu_kernel(int&, std::array<uint16_t, 3> grid, const std::vector<int>&) {
        assert(grid[0] == 5 && grid[2] == 1); y = grid[1];
    }
};
int64_t grid_y(int64_t rows, bool multibase) {
    Queue recorded; auto* queue = &recorded;
    int dummy_kernel = 0; auto* kernel = &dummy_kernel;
    std::vector<int> cores;
""" + dispatch + """
    return recorded.y;
}
int main() {
    for (const int64_t rows : {324}) {
        assert(admitted(rows, false, false, {true, true}));
        assert(!admitted(rows, true, false, {true, true}));
        assert(!admitted(rows, true, true, {true, true}));
        assert(!admitted(rows, false, true, {true, true}));
        assert(!admitted(rows, false, false, {false, true}));
        assert(!admitted(rows, false, false, {true, false}));
    }
    for (const int64_t rows : {0, 256, 320, 336, 352, 384, 704, 768, 972, 1728})
        assert(!admitted(rows, false, false, {true, true}));
    assert(admitted(576, false, false, {false, false}));
    assert(admitted(576, false, true, {false, false}));
    assert(admitted(576, true, true, {false, false}));
    assert(grid_y(576, true) == 3);
    assert(admitted(672, false, false, {true, true}));
    assert(!admitted(672, false, false, {true, false}));
    assert(!admitted(672, false, true, {true, true}));
    for (const int64_t rows : {324, 576, 672}) {
        // Actual launcher grid against the audited M352 payload's origin and
        // tail bounds. An extra block causes unsigned row-count underflow;
        // a missing block leaves part of the Gate/Up result unwritten.
        const int64_t grid = grid_y(rows, false);
        assert(grid == (rows == 324 ? 1 : 2));
        int64_t covered = 0, tail = 0;
        for (int64_t block = 0; block < grid; ++block) {
            assert(block * 352 < rows);
            tail = std::min(int64_t{352}, rows - block * 352);
            covered += tail;
        }
        assert(covered == rows);
        assert(tail == (rows == 324 ? 324 : rows == 672 ? 320 : 224));
    }
}
""")
    binary = tmp_path / "gate_up_rows"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(cpp), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)

    compact = " ".join(launcher.split())
    assert "const int64_t kM = rows;" in compact
    assert "static_cast<uint32_t>(kM), static_cast<uint32_t>(kNPerCore)" in compact
    assert "{5, static_cast<uint16_t>(CeilDiv(rows, int64_t{352})), 1}" in compact
    assert "queue->enqueu_kernel(*kernel, {5, 3, 1}, cores)" in compact
    assert "check_range(input_spm_addr, input_bytes)" in compact
    assert "check_range(gate_output_spm_addr, output_bytes)" in compact
    assert "input_spm_addr + input_bytes <= gate_output_spm_addr" in compact
    assert "int64_t rows = 576" in (ROOT / "src/core/rpu_kernel_decls.h").read_text()


def test_small_gate_up_route_has_matching_plan_and_emission():
    producer = _body("    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    consumer = _body("    void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override")
    assert "small_image_acc16_gate_up_projection_enabled(chunk.len)" in producer
    assert "small_image_gate_up_arguments(chunk.len)" in producer
    assert "small_image_acc16_gate_up_projection_enabled(seq_len)" in consumer
    assert "chunk.idx, small_image_gate_up_arguments(seq_len)" in consumer
    for body in (producer, consumer):
        assert "Qwen25VisionLinearRoute::WALL_SINGLE_BASE_GATE_UP_SWIGLU" in body
    assert "0, 0, 0, 0, wall_sorting_, seq_len" in consumer


def test_owner_ring_norm_complete_consumer_rejects_stale_descriptor(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ is required for the native contract check')
    cpp = tmp_path / 'owner_norm.cpp'
    cpp.write_text('''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("contract"); } while(0)
constexpr int64_t NUM_CORES = 8;
enum class FmbRouteFamily { ALL_REDUCE };
'''
        + VISION[VISION.index('constexpr int64_t QWEN25_SMALL_OWNER_RING_NORM_SITE'):
                 VISION.index(';', VISION.index('constexpr int64_t QWEN25_SMALL_OWNER_RING_NORM_SITE'))+1]
        + _body('enum class Qwen25VisionAllReduceRoute') + ';\n'
        + '''
struct Model {
    bool enabled = true, complete = true;
    int64_t expected_invocation = 0, calls = 0;
    std::vector<int64_t> expected{1,672,1280,8,84,12,1,1};
    Model& ctx() { return *this; }
    bool has_complete_physical_manifest() const { return complete; }
    bool small_image_owner_ring_norm_enabled(int64_t rows) const { return enabled && rows == 672; }
    int64_t hidden_size() const { return 1280; }
    void consume_physical_route(FmbRouteFamily family, int64_t site,
        int64_t selector, int64_t flags, std::vector<int64_t> args, int64_t invocation) {
        TORCH_CHECK(family == FmbRouteFamily::ALL_REDUCE &&
                    site == QWEN25_SMALL_OWNER_RING_NORM_SITE &&
                    selector == static_cast<int64_t>(Qwen25VisionAllReduceRoute::OWNER_RING_NORM_M672) &&
                    flags == 0 && args == expected && invocation == expected_invocation);
        ++calls;
    }
'''
        + _body('    std::vector<int64_t> small_image_owner_ring_norm_arguments(')
        + _body('    void consume_small_image_owner_ring_norm(')
        + '''
};
int main() {
    Model good;
    for (int layer = 0; layer < 32; ++layer) good.consume_small_image_owner_ring_norm(672, 0);
    assert(good.calls == 32);
    auto rejects = [](Model m, int64_t rows=672, int64_t invocation=0) {
        bool caught = false;
        try { m.consume_small_image_owner_ring_norm(rows, invocation); }
        catch (const std::runtime_error&) { caught = true; }
        assert(caught && m.calls == 0);
    };
    Model bad; bad.complete = false; rejects(bad);
    bad = Model{}; bad.enabled = false; rejects(bad);
    for (int i=0; i<8; ++i) {
        bad = Model{}; ++bad.expected[i]; rejects(bad);
    }
    rejects(Model{}, 648); rejects(Model{}, 672, 1);
}
''')
    binary = tmp_path / 'owner_norm'
    subprocess.run([compiler, '-std=c++17', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_down_next_norm_rejects_wrong_layer_plan_and_handoff(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ is required for the native contract check')
    header = (ROOT / 'src/core/fused_model_base.h').read_text()
    cpp = tmp_path / 'down_norm.cpp'
    cpp.write_text('''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("contract"); } while(0)
constexpr int64_t NUM_CORES = 8;
enum class ChunkMode { SEQUENTIAL, KV_FIRST };
enum class FmbRouteFamily { ALL_REDUCE };
'''
        + header[header.index('struct ChunkInfo {'):header.index('// Historical SISC/SIMC')]
        + _body('enum class Qwen25VisionAllReduceRoute') + ';\n'
        + VISION[VISION.index('constexpr int64_t QWEN25_SMALL_DOWN_OWNER_NORM_SITE'):
                 VISION.index(';', VISION.index('constexpr int64_t QWEN25_SMALL_DOWN_OWNER_NORM_SITE'))+1]
        + '''
struct Model {
    bool enabled=true, complete=true, input_in_spm=false, output_to_spm=true;
    int calls=0;
    FmbThreeStageChunkPlan stage_plan;
    struct Manifest { int64_t logical_length=648, physical_length=672, execution_padding_rows=24; } manifest;
    int64_t expected_layer=0;
    std::vector<int64_t> expected{1,672,1280,8,84,12,2,1,0,1};
    Model() {
        for (auto* stage : {&stage_plan.input, &stage_plan.qkv, &stage_plan.compute}) {
            stage->plan={672,1}; stage->chunks={{0,0,672,672}};
        }
        stage_plan.spans={{0,336,0},{336,336,1}};
    }
    Model& ctx() { return *this; }
    bool has_complete_physical_manifest() const { return complete; }
    const Manifest& physical_manifest() const { return manifest; }
    bool small_image_down_owner_norm_enabled(int64_t rows) const { return enabled && rows==672; }
    int64_t hidden_size() const { return 1280; }
    void consume_physical_route(FmbRouteFamily family, int64_t site, int64_t selector,
        int64_t flags, const std::vector<int64_t>& args, int64_t invocation) {
        TORCH_CHECK(family==FmbRouteFamily::ALL_REDUCE && site==QWEN25_SMALL_DOWN_OWNER_NORM_SITE &&
            selector==static_cast<int64_t>(Qwen25VisionAllReduceRoute::OWNER_DOWN_RING_NORM_M672) &&
            flags==0 && args==expected && invocation==expected_layer);
        ++calls;
    }
'''
        + _body('    void validate_small_image_down_owner_plan(')
        + _body('    void validate_small_image_down_owner_context(')
        + _body('    std::vector<int64_t> small_image_down_owner_norm_arguments(')
        + _body('    void consume_small_image_down_owner_norm(')
        + '''
};
int main() {
    ChunkInfo chunk{0,0,672,672};
    Model good;
    for (int layer=0; layer<32; ++layer) {
        good.input_in_spm=layer>0; good.output_to_spm=layer<31;
        good.validate_small_image_down_owner_context(layer,chunk);
        if (layer<31) {
            good.expected_layer=layer; good.expected[8]=layer; good.expected[9]=layer+1;
            good.consume_small_image_down_owner_norm(672,layer);
        }
    }
    assert(good.calls==31);
    auto rejects=[](auto action) {
        bool caught=false; try { action(); } catch (const std::runtime_error&) { caught=true; }
        assert(caught);
    };
    auto reject_context=[&](Model bad, int64_t layer=0, ChunkInfo c={0,0,672,672}) {
        rejects([&] { bad.validate_small_image_down_owner_context(layer,c); });
        assert(bad.calls==0);
    };
    Model bad; bad.complete=false; reject_context(bad);
    bad=Model{}; bad.enabled=false; reject_context(bad);
    bad=Model{}; bad.input_in_spm=true; reject_context(bad);
    bad=Model{}; bad.output_to_spm=false; reject_context(bad);
    reject_context(Model{},1); reject_context(Model{},-1); reject_context(Model{},32);
    bad=Model{}; bad.input_in_spm=true; reject_context(bad,31);
    bad=Model{}; --bad.manifest.logical_length; reject_context(bad);
    bad=Model{}; --bad.manifest.physical_length; reject_context(bad);
    bad=Model{}; --bad.manifest.execution_padding_rows; reject_context(bad);
    for (int field=0;field<4;++field) {
        auto c=chunk;
        if(field==0) ++c.idx; if(field==1) ++c.offset;
        if(field==2) --c.len; if(field==3) --c.kv_seq_len;
        reject_context(Model{},0,c);
    }
    bad=Model{}; bad.stage_plan.chunk_mode=ChunkMode::KV_FIRST; reject_context(bad);
    bad=Model{}; bad.stage_plan.spans.pop_back(); reject_context(bad);
    for(int i=0;i<2;++i) for(int field=0;field<3;++field) {
        bad=Model{}; auto& span=bad.stage_plan.spans[i];
        if(field==0) ++span.offset; if(field==1) --span.len; if(field==2) ++span.group_id;
        reject_context(bad);
    }
    for(int i=0;i<3;++i) for(int field=0;field<7;++field) {
        bad=Model{}; auto* stage=i==0?&bad.stage_plan.input:i==1?&bad.stage_plan.qkv:&bad.stage_plan.compute;
        if(field==0) --stage->plan.chunk_size; if(field==1) ++stage->plan.num_chunks;
        if(field==2) ++stage->chunks[0].idx; if(field==3) ++stage->chunks[0].offset;
        if(field==4) --stage->chunks[0].len; if(field==5) --stage->chunks[0].kv_seq_len;
        if(field==6) stage->chunks.push_back(chunk);
        reject_context(bad);
    }
    // Bind both producer and consumer gamma, not merely the shape.
    for(int i=0;i<10;++i) {
        bad=Model{}; ++bad.expected[i];
        rejects([&] {bad.consume_small_image_down_owner_norm(672,0);}); assert(bad.calls==0);
    }
    bad=Model{}; bad.expected_layer=1;
    rejects([&] {bad.consume_small_image_down_owner_norm(672,0);});
    bad=Model{}; bad.complete=false; rejects([&] {bad.consume_small_image_down_owner_norm(672,0);});
    bad=Model{}; bad.enabled=false; rejects([&] {bad.consume_small_image_down_owner_norm(672,0);});
    bad=Model{};
    for(int layer : {-1,31,32}) rejects([&] {bad.consume_small_image_down_owner_norm(672,layer);});
    rejects([&] {bad.consume_small_image_down_owner_norm(648,0);});
    // A supplied valid cold plan must not be replaced by stale context state.
    bad=Model{}; bad.stage_plan.compute.chunks.clear();
    bad.validate_small_image_down_owner_plan(Model{}.stage_plan);
}
''')
    binary = tmp_path / 'down_norm'
    subprocess.run([compiler, '-std=c++17', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
