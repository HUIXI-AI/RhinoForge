"""Execute ordinary SwiGLU admission, planning and dispatch from production C++."""
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
def ordinary_silu_executable(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    decoder = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    plan_source = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    helpers = "\n".join(_block(decoder, marker) for marker in (
        "    bool ordinary_silu_mul_route_enabled() const",
        "    bool use_planned_silu_mul() const",
    ))
    # Wall prefill siblings share this admission block; ordinary profiles bind
    # those cold capabilities off while executing the actual ordinary predicate.
    planner = _block(decoder, "        if (!qwen_c512 && !shared_prefill_silu && (")
    consumer = _block(plan_source, "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(")
    consumer = consumer.replace("FmbPhysicalManifestConsumer::consume_route", "consume_physical_route")
    # This signature has a default `{}` argument before its body brace.
    shared_consumer = re.search(
        r"void consume_shared_runtime_route\([\s\S]*?\n\}", base
    ).group(0)
    # The HIGH fusion owns the preceding branch; this probe executes only the
    # existing ordinary SiLU/Mul route and retains its independent admission.
    fused_dispatch = _block(base, "else if (fuse_silu_mul) {").removeprefix("else ")
    wall_gate = re.search(r"    const bool wall_action_fuse_gate_up =[\s\S]*?;", base).group(0)
    constant = re.search(r"inline constexpr int64_t FMB_SHARED_MLP_SILU_MUL_SITE\s*=\s*\d+LL;", header).group(0)
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
#include "rpu_wall_sorting_config.h"
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("route rejected"); } while (0)
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
    IntArrayRef& operator=(const std::array<int64_t, 3>& value) {
        assign(value.begin(), value.end()); return *this;
    }
};
}
enum class ChunkMode { SEQUENTIAL, KV_FIRST };
enum class FmbRouteFamily { ACTIVATION, LINEAR };
enum class ActivationKind { NONE, SILU, GELU };
enum class CausalDecoderPlanningMode { ORDINARY, WALL_Z1, QWEN3VL_POOLER_Z1, QWEN3VL_MULTIVIEW_TEXT };
struct ChunkInfo { int64_t idx, offset, len; };
struct Stage { std::vector<ChunkInfo> chunks; };
struct Plan { ChunkMode chunk_mode = ChunkMode::SEQUENTIAL; Stage compute; };
struct LayoutContext { bool is_causal = true; int64_t batch_size = 1; };
struct FmbRouteManifestEntry {
    FmbRouteFamily family;
    int64_t site_id, selector, flags;
    std::vector<int64_t> arguments;
    int64_t invocation;
};
struct Manifest { std::vector<FmbRouteManifestEntry> routes; };
''' + constant + r'''
struct InferenceContext {
    bool complete = true, is_causal = true;
    int64_t batch_size = 1, physical_route_invocation = 0;
    Plan stage_plan;
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
''' + shared_consumer + r'''
struct Launch { uint32_t x, y, out; int64_t elements; int cores; };
std::vector<Launch> launched;
void rpu_launch_silu_mul_spm_kernel(uint32_t x, uint32_t y, uint32_t out, int64_t elements, int cores) {
    launched.push_back({x, y, out, elements, cores});
}
class CausalDecoderModel {
public:
    virtual ~CausalDecoderModel() = default;
    struct LayerWeights { at::Tensor gate_w, up_w, down_w; };
    int cores = 8, mlp_cores = 8;
    int64_t intermediate = 3072;
    bool use_silu_ = true, linear_acc32_ = false, has_mrope_ = false;
    bool has_qkv_bias_ = false, nvfp4_ = false, decoder_mixed_precision_ = false;
    bool wall_prefill_staged_mlp_ = false, decode_gate_up_fusion_ = false, adarms_ = false;
    bool pre_layers_residual1_ready_ = false, wall_z1_dispatch_ = false;
    bool qwen3vl_pooler_z1_dispatch_ = false, qwen3vl_multiview_text_composite_dispatch_ = false;
    std::vector<int> deepstack_lang_layers_;
    std::vector<LayerWeights> layer_weights_{3};
    CausalDecoderPlanningMode planning_mode_ = CausalDecoderPlanningMode::ORDINARY;
    InferenceContext context;
    int num_cores() const { return cores; }
    int mlp_tp() const { return mlp_cores; }
    int64_t intermediate_size() const { return intermediate; }
    const InferenceContext& ctx() const { return context; }
''' + helpers + r'''
    Manifest plan_for_test(const Plan& plan, const LayoutContext& layout) const {
        Manifest manifest;
        const bool qwen_c512 = false, shared_prefill_silu = false;
        const bool wall_small_prefill_fused = false, wall_fp16_long_prefill_fused = false;
        auto append = [&](FmbRouteFamily family, int64_t site, int64_t selector, int64_t flags,
                          std::vector<int64_t> arguments, int64_t invocation) {
            manifest.routes.push_back({family, site, selector, flags, std::move(arguments), invocation});
        };
''' + planner + r'''
        return manifest;
    }
    void bind(Manifest manifest, const Plan& plan, const LayoutContext& layout = {}) {
        context.manifest_ = std::move(manifest);
        context.consumed_.assign(context.manifest_.routes.size(), 0);
        context.stage_plan = plan;
        context.is_causal = layout.is_causal;
        context.batch_size = layout.batch_size;
    }
};
class DerivedDecoder : public CausalDecoderModel {};
struct Pipeline {
    struct Impl { InferenceContext ctx_; int64_t intermediate_size_ = 2048, hidden_size_ = 1024; } impl;
    Impl* pimpl_ = &impl;
    uint32_t addr(int, const char* name) { return std::string(name) == "gate" ? 256 : 8192; }
    void dispatch(bool fuse_silu_mul, bool bind_silu_mul_route, int64_t seq_len,
                  int kNumCores, ActivationKind act = ActivationKind::SILU) {
        const int64_t elems = seq_len * (pimpl_->intermediate_size_ / kNumCores);
''' + fused_dispatch + r'''
    }
    bool wall_gate(bool fuse_wall_gate_up, bool fuse_silu_mul, int64_t seq_len,
                   bool acc32, at::Tensor gate_w, at::Tensor up_w,
                   at::Tensor gate_ws = {}, at::Tensor up_ws = {},
                   ActivationKind act = ActivationKind::SILU,
                   const WallSortingConfig& sorting = {}) {
''' + wall_gate + r'''
        return wall_action_fuse_gate_up;
    }
};
template<class F> void must_reject(F fn) {
    launched.clear();
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && launched.empty());
}
void routes() {
    for (int64_t intermediate : {3072, 6144, 8192, 12288}) {
        CausalDecoderModel model;
        model.intermediate = intermediate;
        for (int64_t rows : {1, 3, 15, 16, 17, 31, 32, 58, 64, 128, 314, 320}) {
            Plan plan;
            plan.compute.chunks = {{0, 0, 64}, {1, 64, rows}};
            const auto manifest = model.plan_for_test(plan, {});
            assert(manifest.routes.size() == plan.compute.chunks.size());
            model.bind(manifest, plan);
            assert(model.ordinary_silu_mul_route_enabled() && model.use_planned_silu_mul());
            for (const auto& chunk : plan.compute.chunks) {
                const auto& route = manifest.routes[chunk.idx];
                assert(route.family == FmbRouteFamily::ACTIVATION);
                assert(route.site_id == FMB_SHARED_MLP_SILU_MUL_SITE && route.selector == 1 && route.flags == 0);
                assert(route.invocation == chunk.idx);
                assert((route.arguments == std::vector<int64_t>{chunk.len, intermediate / 8, 8}));
                Pipeline pipeline;
                pipeline.impl.ctx_ = model.context;
                pipeline.impl.ctx_.physical_route_invocation = chunk.idx;
                pipeline.impl.intermediate_size_ = intermediate;
                launched.clear();
                pipeline.dispatch(model.use_planned_silu_mul(), true, chunk.len, 8);
                assert(launched.size() == 1);
                assert(launched[0].elements == chunk.len * intermediate / 8 && launched[0].cores == 8);
                assert(launched[0].x == launched[0].out && launched[0].x != launched[0].y);
                assert(pipeline.impl.ctx_.consumed_[chunk.idx] == 1);
            }
        }
    }
}
void rejection() {
    CausalDecoderModel model;
    Plan plan;
    plan.compute.chunks = {{0, 0, 64}, {1, 64, 17}};
    const auto manifest = model.plan_for_test(plan, {});
    model.bind(manifest, plan);
    for (int fault = 0; fault < 9; ++fault) {
        Pipeline pipeline;
        pipeline.impl.ctx_ = model.context;
        pipeline.impl.ctx_.physical_route_invocation = 1;
        pipeline.impl.intermediate_size_ = model.intermediate;
        auto& route = pipeline.impl.ctx_.manifest_.routes[1];
        if (fault == 0) route.selector = 2;
        if (fault == 1) route.flags = 1;
        if (fault >= 2 && fault <= 4) ++route.arguments[fault - 2];
        if (fault == 5) route.arguments.pop_back();
        if (fault == 6) route.invocation = 2;
        if (fault == 7) route.family = FmbRouteFamily::LINEAR;
        if (fault == 8) pipeline.impl.ctx_.manifest_.routes.pop_back();
        must_reject([&] { pipeline.dispatch(true, true, 17, 8); });
    }
    Pipeline pipeline;
    must_reject([&] { pipeline.dispatch(true, false, 17, 8, ActivationKind::GELU); });
}
void fallback() {
    Plan plan;
    plan.compute.chunks = {{0, 0, 17}};
    for (int mode = 0; mode < 24; ++mode) {
        CausalDecoderModel model;
        LayoutContext layout;
        if (mode == 22) model.intermediate = 3080;
        if (mode == 23) model.intermediate = 0;
        if (mode == 0) model.cores = 4;
        if (mode == 1) model.cores = 6;
        if (mode == 2) model.mlp_cores = 4;
        if (mode == 3) model.linear_acc32_ = true;
        if (mode == 4) model.use_silu_ = false;
        if (mode == 5) model.has_mrope_ = true;
        if (mode == 6) model.has_qkv_bias_ = true;
        if (mode == 7) model.nvfp4_ = true;
        if (mode == 8) model.decoder_mixed_precision_ = true;
        if (mode == 9) model.wall_prefill_staged_mlp_ = true;
        if (mode == 10) model.decode_gate_up_fusion_ = true;
        if (mode == 11) model.adarms_ = true;
        if (mode == 12) model.deepstack_lang_layers_ = {1};
        if (mode == 13) model.pre_layers_residual1_ready_ = true;
        if (mode == 14) model.planning_mode_ = CausalDecoderPlanningMode::WALL_Z1;
        if (mode == 15) model.wall_z1_dispatch_ = true;
        if (mode == 16) model.qwen3vl_pooler_z1_dispatch_ = true;
        if (mode == 17) model.qwen3vl_multiview_text_composite_dispatch_ = true;
        if (mode == 18) model.layer_weights_.clear();
        if (mode == 19) model.layer_weights_[1].gate_w.dtype = at::kChar;
        if (mode == 20) model.layer_weights_[1].up_w.dtype = at::kChar;
        if (mode == 21) model.layer_weights_[1].down_w.dtype = at::kFloat;
        assert(!model.ordinary_silu_mul_route_enabled());
        const auto manifest = model.plan_for_test(plan, layout);
        assert(manifest.routes.empty());
        model.bind(manifest, plan, layout);
        assert(!model.use_planned_silu_mul());
        Pipeline pipeline;
        launched.clear();
        pipeline.dispatch(model.use_planned_silu_mul(), false, 17, 8);
        assert(launched.empty());
    }
    DerivedDecoder derived;
    assert(!derived.ordinary_silu_mul_route_enabled());
    assert(derived.plan_for_test(plan, {}).routes.empty());
    for (int mode = 0; mode < 4; ++mode) {
        CausalDecoderModel model;
        LayoutContext layout;
        Plan alternate = plan;
        if (mode == 0) layout.is_causal = false;
        if (mode == 1) layout.batch_size = 2;
        if (mode == 2) alternate.chunk_mode = ChunkMode::KV_FIRST;
        auto manifest = model.plan_for_test(alternate, layout);
        assert(manifest.routes.empty() == (mode != 3));
        model.bind(manifest, alternate, layout);
        if (mode == 3) model.context.complete = false;
        assert(!model.use_planned_silu_mul());
    }
}
void isolation() {
    Pipeline pipeline;
    at::Tensor quantized; quantized.dtype = at::kChar;
    // Even the exact historical shape must not select the Wall producer
    // merely because the independent elementwise fusion is enabled.
    assert(!pipeline.wall_gate(false, true, 20, false, quantized, quantized));
    assert(pipeline.wall_gate(true, true, 20, false, quantized, quantized));
    assert(!pipeline.wall_gate(true, false, 20, false, quantized, quantized));
    assert(!pipeline.wall_gate(true, true, 21, false, quantized, quantized));
    assert(!pipeline.wall_gate(true, true, 20, true, quantized, quantized));
    assert(!pipeline.wall_gate(true, true, 20, false, {}, {}));
    at::Tensor missing; missing.present = false;
    assert(!pipeline.wall_gate(true, true, 20, false, quantized, quantized, missing));
    // An explicitly unbound historical fusion remains independent of the new
    // ordinary descriptor route and therefore needs no new route receipt.
    launched.clear();
    pipeline.dispatch(true, false, 20, 8);
    assert(launched.size() == 1);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string mode(argv[1]);
    if (mode == "routes") routes();
    else if (mode == "rejection") rejection();
    else if (mode == "fallback") fallback();
    else if (mode == "isolation") isolation();
    else return 1;
}
'''
    directory = tmp_path_factory.mktemp("ordinary_silu")
    source = directory / "ordinary_silu.cpp"
    source.write_text(program)
    executable = directory / "ordinary_silu"
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "src/core"), str(source), "-o", str(executable)], check=True)
    return executable


@pytest.mark.parametrize("case", ["routes", "rejection", "fallback", "isolation"])
def test_ordinary_silu_mul_production_routes(ordinary_silu_executable, case):
    subprocess.run([str(ordinary_silu_executable), case], check=True)
