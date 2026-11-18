"""Execute production row-weight-reuse admission, manifest and dispatch on host."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _block(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def ordinary_linear_reuse_executable(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    base = (ROOT / "src/core/fused_model_base.h").read_text()
    planning = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    helpers = "\n".join(_block(decoder, marker) for marker in (
        "    bool ordinary_row_weight_reuse_plan_enabled(",
        "    std::vector<int64_t> ordinary_o_route_arguments(",
        "    bool consume_ordinary_o_route(",
        "    void consume_causal_route(",
        "    std::array<int64_t, 6> cold_topology_arguments() const",
        "    std::vector<int64_t> bind_cold_topology_arguments(",
        "    bool qwen3vl_decode_gemv(",
        "    CausalDecoderLinearRoute decoder_projection_route(",
    ))
    planner_if = _block(decoder, "if (ordinary_row_weight_reuse_plan_enabled(")
    after_if = decoder[decoder.index(planner_if) + len(planner_if):]
    planner = planner_if + " " + _block(after_if, "else {")
    manifest_start = decoder.index("    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    append_start = decoder.index("        auto append =", manifest_start)
    appender = decoder[append_start:decoder.index("        auto append_attention =", append_start)]
    appender += _block(decoder[manifest_start:], "        auto append_linear =") + ";"
    consumer = _block(planning, "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(")
    consumer = consumer.replace("FmbPhysicalManifestConsumer::consume_route", "consume_physical_route")
    constants = re.search(r"inline constexpr int64_t CAUSAL_DECODER_O_SITE\s*=\s*\d+LL;", decoder).group(0)
    enums = "\n".join((
        _block(base, "enum class FmbLinearRouteSelector : int") + ";",
        _block(decoder, "enum class CausalDecoderLinearRoute : int64_t") + ";",
        _block(decoder, "enum class CausalDecoderPlanningMode : int64_t") + ";",
    ))
    dispatch = re.search(
        r"            const bool reuse_o_weight = consume_ordinary_o_route\(\s*"
        r"CAUSAL_DECODER_O_SITE,\s*rows,\s*chunk.idx\);"
        r"\s*rpu_launch_linear_spm_to_spm_acc16_kernel\([\s\S]*?\);", decoder
    ).group(0)
    program = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>
#include "src/ops/rpu_linear_tiling.h"
#include "src/ops/rpu_pi_wall_tiling.h"
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("route rejected"); } while (0)
constexpr int NUM_CORES = 8;
namespace at {
enum ScalarType { kHalf, kChar, kFloat };
struct Tensor {
    ScalarType dtype = kHalf;
    bool present = true;
    ScalarType scalar_type() const { return dtype; }
    bool defined() const { return present; }
};
struct IntArrayRef : std::vector<int64_t> {
    using std::vector<int64_t>::vector;
    IntArrayRef(const std::vector<int64_t>& value) : std::vector<int64_t>(value) {}
    template<size_t N> IntArrayRef(const std::array<int64_t, N>& value)
        : std::vector<int64_t>(value.begin(), value.end()) {}
};
}
enum class ChunkMode { SEQUENTIAL, KV_FIRST };
enum class FmbRouteFamily { LINEAR, ALL_REDUCE, MUTABLE_DMA, ACTIVATION };
enum class KernelId : uint16_t {};
KernelId pi_wall_kernel_id(uint16_t ordinal) { return static_cast<KernelId>(ordinal); }
struct KernelCache {
    bool available = true;
    static KernelCache& instance() { static KernelCache value; return value; }
    bool has_loaded(KernelId) const { return available; }
};
struct ChunkInfo { int64_t idx, offset, len; };
struct Stage { std::vector<ChunkInfo> chunks; };
struct FmbThreeStageChunkPlan { ChunkMode chunk_mode = ChunkMode::SEQUENTIAL; Stage compute; };
struct LayoutContext { bool is_causal = true; int64_t batch_size = 1; };
struct FmbRouteManifestEntry {
    int64_t site_id; FmbRouteFamily family; int64_t selector, flags;
    std::vector<int64_t> arguments; int64_t invocation;
};
struct Manifest { std::vector<FmbRouteManifestEntry> routes; };
''' + enums + constants + r'''
struct Context {
    bool complete = true, is_causal = true;
    int64_t batch_size = 1;
    FmbThreeStageChunkPlan stage_plan;
    Manifest manifest_;
    const auto& manifest() const { return manifest_; }
    mutable std::vector<int> consumed_;
    bool has_complete_physical_manifest() const { return complete; }
    const FmbRouteManifestEntry& find_route(FmbRouteFamily family, int64_t site, int64_t invocation) const {
        for (const auto& entry : manifest_.routes)
            if (entry.family == family && entry.site_id == site && entry.invocation == invocation) return entry;
        throw std::runtime_error("missing route");
    }
''' + consumer + r'''
};
struct Launch { int64_t rows, n, k; int partition, cores; bool reuse, gemv; };
std::vector<Launch> launched;
void rpu_launch_linear_spm_to_spm_acc16_kernel(
    uint32_t, const at::Tensor&, uint32_t, int64_t m, int64_t n, int64_t k,
    int partition, int cores, uint32_t, const at::Tensor&, uint32_t, uint16_t,
    bool, bool prefer_gemv, bool, bool reuse) {
    launched.push_back({m, n, k, partition, cores, reuse, prefer_gemv});
}
class CausalDecoderModel {
public:
    virtual ~CausalDecoderModel() = default;
    struct LayerWeights { at::Tensor o_w{at::kChar}, o_ws; };
    int cores = 8, tp = 8;
    int64_t h = 1024, q_heads = 16, hd = 128;
    int64_t seq_len_ = 512;
 bool qwen3vl_fp16_decode_profile_=false;
 bool qwen3vl_4b_w8_profile_ = false;
    bool linear_acc32_ = false, has_mrope_ = false, has_qkv_bias_ = false;
    bool nvfp4_ = false, decoder_mixed_precision_ = false;
    bool wall_prefill_staged_mlp_ = false, decode_gate_up_fusion_ = false, adarms_ = false;
    bool pre_layers_residual1_ready_ = false, wall_z1_dispatch_ = false;
    bool qwen3vl_pooler_z1_dispatch_ = false, qwen3vl_multiview_text_composite_dispatch_ = false;
    std::vector<int> deepstack_lang_layers_;
    std::vector<LayerWeights> layer_weights_{3};
    CausalDecoderPlanningMode planning_mode_ = CausalDecoderPlanningMode::ORDINARY;
    Context context;
    int num_cores() const { return cores; }
    int attn_tp() const { return tp; }
    int mlp_tp() const { return cores; }
    int lm_head_tp() const { return cores; }
    int64_t hidden_size() const { return h; }
    int64_t num_q_heads() const { return q_heads; }
    int64_t head_dim() const { return hd; }
    const Context& ctx() const { return context; }
    uint32_t addr(int, const char*) const { return 256; }
    uint32_t nvfp4_scale_addr(const char*) const { return 0; }
''' + helpers + r'''
    Manifest plan_for_test(const FmbThreeStageChunkPlan& plan, const LayoutContext& layout = {}) const {
        const int64_t physical_len = plan.compute.chunks.empty() ? 0 :
            plan.compute.chunks.back().offset + plan.compute.chunks.back().len;
        Manifest manifest;
''' + appender + planner + r'''
        return manifest;
    }
    void bind(Manifest manifest, const FmbThreeStageChunkPlan& plan, const LayoutContext& layout = {}) {
        context.manifest_ = std::move(manifest);
        context.consumed_.assign(context.manifest_.routes.size(), 0);
        context.stage_plan = plan;
        seq_len_ = plan.compute.chunks.empty() ? 0 :
            plan.compute.chunks.back().offset + plan.compute.chunks.back().len;
        context.is_causal = layout.is_causal;
        context.batch_size = layout.batch_size;
    }
    void dispatch(const ChunkInfo& chunk) {
        const int layer_idx = 0;
        const auto& lw = layer_weights_.at(0);
        const int64_t rows = chunk.len, nq = num_q_heads();
''' + dispatch + r'''
    }
};
class DerivedDecoder : public CausalDecoderModel {};
template<class F> void must_reject(F fn) {
    launched.clear();
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && launched.empty());
}
void routes() {
    for (int64_t tail : {1, 16, 17, 64, 256, 511, 512}) {
        CausalDecoderModel model;
        FmbThreeStageChunkPlan plan;
        plan.compute.chunks = {{0, 0, 512}, {1, 512, tail}};
        const auto manifest = model.plan_for_test(plan);
        assert(manifest.routes.size() == 2);
        model.bind(manifest, plan);
        for (const auto& chunk : plan.compute.chunks) {
            const bool reuse = chunk.len == 512;
            const auto& route = manifest.routes[chunk.idx];
            assert(route.site_id == CAUSAL_DECODER_O_SITE && route.family == FmbRouteFamily::LINEAR);
            assert(route.selector == (reuse ? 5 : 1) && route.flags == 0 && route.invocation == chunk.idx);
            std::vector<int64_t> expected{chunk.len, 1024, 2048, 0, 8};
            if (reuse) {
                const auto tile = rpu_pi_wall::select(rpu_pi_wall::Operation::RowWeightReuse,
                    rpu_pl_tiling::WeightType::Int8, false, 512, 1024, 256);
                expected.insert(expected.end(), {2, tile.tile.m_tile, tile.tile.n_tile,
                    tile.tile.k_tile, tile.grid_x, tile.grid_y});
            }
            assert(route.arguments == expected);
            launched.clear();
            model.dispatch(chunk);
            assert(launched.size() == 1 && launched[0].reuse == reuse);
            assert(!launched[0].gemv);
            assert(launched[0].rows == chunk.len && launched[0].n == 1024 && launched[0].k == 2048);
            assert(launched[0].partition == 0 && launched[0].cores == 8);
            assert(model.context.consumed_[chunk.idx] == 1);
        }
    }
    // The merged coarse O path must retain the exact decode GEMV selector;
    // neither planner nor runtime may reinterpret it as row-weight reuse.
    CausalDecoderModel decode;
    decode.qwen3vl_4b_w8_profile_ = true;
    decode.has_mrope_ = true; decode.h = 2560; decode.q_heads = 32;
    FmbThreeStageChunkPlan single;
    single.compute.chunks = {{0, 0, 1}};
    const auto manifest = decode.plan_for_test(single);
    assert(manifest.routes.size() == 1 && manifest.routes[0].selector == 2);
    decode.bind(manifest, single); launched.clear();
    decode.dispatch(single.compute.chunks[0]);
    assert(launched.size() == 1 && launched[0].gemv && !launched[0].reuse);
    auto wrong = manifest; wrong.routes[0].selector = 1;
    decode.bind(wrong, single);
    must_reject([&] { decode.dispatch(single.compute.chunks[0]); });
}
void rejection() {
    CausalDecoderModel original;
    FmbThreeStageChunkPlan plan;
    plan.compute.chunks = {{0, 0, 512}, {1, 512, 17}};
    const auto manifest = original.plan_for_test(plan);
    original.bind(manifest, plan);
    must_reject([&] {
        original.consume_ordinary_o_route(CAUSAL_DECODER_O_SITE + 1, 512, 0);
    });
    for (const auto& chunk : plan.compute.chunks) {
        for (int fault = 0; fault < 16; ++fault) {
            CausalDecoderModel model;
            model.bind(manifest, plan);
            auto& route = model.context.manifest_.routes[chunk.idx];
            if (fault == 0) route.selector = route.selector == 5 ? 1 : 5;
            if (fault == 1) route.flags = 1;
            if (fault >= 2 && fault < 11) {
                const size_t arg = fault - 2;
                if (arg >= route.arguments.size()) continue;
                ++route.arguments[arg];
            }
            if (fault == 11) route.invocation += 2;
            if (fault == 12) route.arguments.pop_back();
            if (fault == 13) model.context.manifest_.routes.erase(model.context.manifest_.routes.begin() + chunk.idx);
            if (fault == 14) model.context.complete = false;
            if (fault == 15) route.selector = static_cast<int64_t>(FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32);
            must_reject([&] { model.dispatch(chunk); });
        }
    }
    // Availability/dtype must not silently reinterpret a selected per-chunk
    // descriptor as the old coarse AUTO_TILE route.
    for (int fault = 0; fault < 2; ++fault) {
        CausalDecoderModel model;
        model.bind(manifest, plan);
        if (fault == 0) KernelCache::instance().available = false;
        if (fault == 1) model.layer_weights_[1].o_ws.present = false;
        must_reject([&] { model.dispatch(plan.compute.chunks[0]); });
        KernelCache::instance().available = true;
    }
}
void coarse_fallback(CausalDecoderModel& model, const FmbThreeStageChunkPlan& plan,
                     const LayoutContext& layout = {}, bool can_dispatch = true) {
    assert(!model.ordinary_row_weight_reuse_plan_enabled(plan, layout.is_causal, layout.batch_size));
    const auto manifest = model.plan_for_test(plan, layout);
    assert(manifest.routes.size() == 1);
    const auto& route = manifest.routes[0];
    assert(route.selector == 1 && route.invocation == 0);
    assert(route.arguments.empty() == (model.cores == 8));
    model.bind(manifest, plan, layout);
    // Site identity is mandatory even when this owner retains coarse AUTO_TILE.
    must_reject([&] {
        model.consume_ordinary_o_route(CAUSAL_DECODER_O_SITE + 1, 512, 0);
    });
    if (can_dispatch) for (const auto& chunk : plan.compute.chunks) {
        launched.clear();
        model.dispatch(chunk);
        assert(launched.size() == 1 && !launched[0].reuse);
        assert(model.context.consumed_[0] == 1);
    }
}
void fallback() {
    FmbThreeStageChunkPlan plan;
    plan.compute.chunks = {{0, 0, 512}, {1, 512, 17}};
    for (int mode = 0; mode < 26; ++mode) {
        CausalDecoderModel model;
        LayoutContext layout;
        FmbThreeStageChunkPlan current = plan;
        if (mode == 0) KernelCache::instance().available = false;
        if (mode == 1) model.layer_weights_[1].o_w.dtype = at::kHalf;
        if (mode == 2) model.layer_weights_[1].o_w.present = false;
        if (mode == 3) model.layer_weights_[1].o_ws.present = false;
        if (mode == 4) model.layer_weights_[1].o_ws.dtype = at::kFloat;
        if (mode == 5) model.cores = model.tp = 4;
        if (mode == 6) model.cores = model.tp = 6;
        if (mode == 7) model.tp = 4;
        if (mode == 8) model.linear_acc32_ = true;
        if (mode == 9) model.has_mrope_ = true;
        if (mode == 10) model.has_qkv_bias_ = true;
        if (mode == 11) model.nvfp4_ = true;
        if (mode == 12) model.decoder_mixed_precision_ = true;
        if (mode == 13) model.wall_prefill_staged_mlp_ = true;
        if (mode == 14) model.decode_gate_up_fusion_ = true;
        if (mode == 15) model.adarms_ = true;
        if (mode == 16) model.deepstack_lang_layers_ = {1};
        if (mode == 17) model.pre_layers_residual1_ready_ = true;
        if (mode == 18) model.planning_mode_ = CausalDecoderPlanningMode::WALL_Z1;
        if (mode == 19) model.wall_z1_dispatch_ = true;
        if (mode == 20) model.qwen3vl_pooler_z1_dispatch_ = true;
        if (mode == 21) model.qwen3vl_multiview_text_composite_dispatch_ = true;
        if (mode == 22) model.layer_weights_.clear();
        if (mode == 23) layout.is_causal = false;
        if (mode == 24) layout.batch_size = 2;
        if (mode == 25) current.chunk_mode = ChunkMode::KV_FIRST;
        coarse_fallback(model, current, layout, mode != 22);
        KernelCache::instance().available = true;
    }
    DerivedDecoder derived;
    coarse_fallback(derived, plan);
    for (int mode = 0; mode < 4; ++mode) {
        CausalDecoderModel model;
        FmbThreeStageChunkPlan current = plan;
        if (mode == 0) current.compute.chunks = {{0, 0, 256}, {1, 256, 17}};
        if (mode == 1) current.compute.chunks.clear();
        if (mode == 2) model.h = 2048;
        if (mode == 3) model.q_heads = 8;
        coarse_fallback(model, current);
    }
    assert(!rpu_pl_tiling::supports_w8a16_row_weight_reuse(512, 1024, 2048, 1, 8));
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode(argv[1]);
    if (mode == "routes") routes();
    else if (mode == "rejection") rejection();
    else if (mode == "fallback") fallback();
    else return 1;
}
'''
    directory = tmp_path_factory.mktemp("ordinary_linear_reuse")
    source = directory / "ordinary_linear_reuse.cpp"
    source.write_text(program)
    executable = directory / "ordinary_linear_reuse"
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(ROOT), str(source), "-o", str(executable)], check=True
    )
    return executable


@pytest.mark.parametrize("case", ["routes", "rejection", "fallback"])
def test_ordinary_linear_reuse_production_routes(ordinary_linear_reuse_executable, case):
    subprocess.run([str(ordinary_linear_reuse_executable), case], check=True)
