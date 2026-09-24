"""Execute production Qwen RMS planning/dispatch on host-only C++ doubles."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _block(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def qwen_rms_executable(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    root = tmp_path_factory.mktemp("qwen8_rms")
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    resolver = declarations[declarations.index("enum RpuRmsNormCapability"):
                            declarations.index("RpuRmsNormSpmContract rpu_snapshot_rmsnorm_spm_contract(")]
    planner = _block(source, "        if (ordinary_rmsnorm_route_enabled() && layout.is_causal &&")
    helpers = "\n".join(_block(source, marker) for marker in (
        "    bool ordinary_rmsnorm_route_enabled() const",
        "    RpuRmsNormSpmRoute consume_ordinary_rmsnorm_route(",
        "    RpuRmsNormSpmRoute consume_decoder_rmsnorm_route(",
        "    RpuRmsNormSpmRoute consume_delivery_rmsnorm_route(",
    ))
    consumer_source = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    consumer = _block(consumer_source, "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(")
    consumer = consumer.replace("FmbPhysicalManifestConsumer::consume_route", "consume_physical_route")
    constants = "\n".join(re.search(rf"inline constexpr int64_t {name}\s*=\s*\d+LL;", source).group(0)
                          for name in ("CAUSAL_DECODER_ORDINARY_RMSNORM_SITE",
                                       "RHINOVLA_TEXT_HIGH_RMSNORM_SITE",
                                       "RHINOVLA_TEXT_HIGH_ACTIVATION_SITE",
                                       "QWEN3VL_DELIVERY_RMS_SITE"))
    calls = re.findall(r"rpu_launch_rmsnorm_spm_kernel\([\s\S]*?\);", source)
    calls = [call for call in calls if "consume_ordinary_rmsnorm_route(" in call or
             re.search(r"consume_decoder_rmsnorm_route\(rows,\s*h,\s*chunk.idx\)", call)]
    assert len(calls) == 5  # input, Q, K, post, final; execute actual launch arguments.
    dispatch = "\n".join(f"if (site == {index}) {{ {call} return; }}" for index, call in enumerate(calls))
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>
#define TORCH_INTERNAL_ASSERT(condition) assert(condition)
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("route rejected"); } while (0)
enum class ChunkMode { SEQUENTIAL, KV_FIRST };
enum class FmbRouteFamily { NORMALIZATION, LINEAR, GRAPH_SCHEDULE };
struct ChunkInfo { int64_t idx, offset, len, kv_seq_len; };
struct Stage { std::vector<ChunkInfo> chunks; };
struct FmbThreeStageChunkPlan { ChunkMode chunk_mode = ChunkMode::SEQUENTIAL; Stage compute, qkv; };
struct LayoutContext { bool is_causal = true; int64_t batch_size = 1; };
struct FmbRouteManifestEntry { FmbRouteFamily family; int64_t site_id, selector, flags; std::vector<int64_t> arguments; int64_t invocation; };
struct Manifest { std::vector<FmbRouteManifestEntry> routes; };
namespace at { using IntArrayRef = std::vector<int64_t>; }
''' + constants + resolver + r'''
struct Context {
    bool is_causal = true, complete = true;
    int64_t batch_size = 1;
    FmbThreeStageChunkPlan stage_plan;
    Manifest manifest_;
    mutable std::vector<int> consumed_;
    const Manifest& manifest() const { return manifest_; }
    bool has_complete_physical_manifest() const { return complete; }
    const FmbRouteManifestEntry& find_route(FmbRouteFamily family, int64_t site, int64_t invocation) const {
        for (const auto& entry : manifest_.routes)
            if (entry.family == family && entry.site_id == site && entry.invocation == invocation) return entry;
        throw std::runtime_error("missing route");
    }
''' + consumer + r'''
};
struct Launch { uint32_t input, output, gamma; int64_t rows, cols; double eps; RpuRmsNormSpmRoute route; };
std::vector<Launch> launched;
void rpu_launch_rmsnorm_spm_kernel(uint32_t input, uint32_t output, uint32_t gamma,
                                 int64_t rows, int64_t cols, double eps, RpuRmsNormSpmRoute route,
                                 int cores = 8) {
    assert(cores == 8);
    launched.push_back({input, output, gamma, rows, cols, eps, route});
}
class CausalDecoderModel {
public:
    virtual ~CausalDecoderModel() = default;
    int64_t h = 4096, intermediate = 12288, layers = 36, q = 32, kv = 8, hd = 128, tp = 8;
    bool has_qk_norm_ = true, has_qkv_bias_ = false, use_silu_ = true, nvfp4_ = false;
    bool adarms_ = false, qwen3vl_2b_w8_profile_ = false;
    bool rhinovla_high_precision_ = false, rhinovla_high_precision_fusions_ = false;
    bool qwen3vl4b_fused_qk = false;
    bool has_mrope_ = false;
    std::vector<int> deepstack_lang_layers_;
    RpuRmsNormSpmContract ordinary_rmsnorm_spm_contract_{
        RPU_RMSNORM_CAP_GENERIC_DIRECT_SPM};
    double eps_ = 1e-6;
    Context context;
    int64_t hidden_size() const { return h; }
    int64_t intermediate_size() const { return intermediate; }
    int64_t num_layers() const { return layers; }
    int64_t num_q_heads() const { return q; }
    int64_t num_kv_heads() const { return kv; }
    int64_t head_dim() const { return hd; }
    int64_t attn_tp() const { return tp; }
    int64_t num_cores() const { return 8; }
    int64_t mlp_tp() const { return 8; }
    const Context& ctx() const { return context; }
    uint32_t addr(int, const char* name) const { return static_cast<uint32_t>(std::string(name).back()); }
    uint32_t norm_input_addr(int, int) const { return 10; }
    uint32_t norm_q_addr(int, int) const { return 20; }
    uint32_t norm_k_addr(int, int) const { return 30; }
    uint32_t norm_post_addr(int, int) const { return 40; }
    uint32_t norm_final_addr(int) const { return 50; }
''' + helpers + r'''
    Manifest plan_for_test(const FmbThreeStageChunkPlan& plan, const LayoutContext& layout) const {
        Manifest result;
        auto append = [&](FmbRouteFamily family, int64_t site, int64_t selector, int64_t flags,
                          std::vector<int64_t> args, int64_t invocation) {
            result.routes.push_back({family, site, selector, flags, std::move(args), invocation});
        };
''' + planner + r'''
        return result;
    }
    void bind(Manifest manifest, const LayoutContext& layout = {},
              ChunkMode chunk_mode = ChunkMode::SEQUENTIAL) {
        context.manifest_ = std::move(manifest);
        context.consumed_.assign(context.manifest_.routes.size(), 0);
        context.is_causal = layout.is_causal;
        context.batch_size = layout.batch_size;
        context.stage_plan.chunk_mode = chunk_mode;
    }
    void dispatch(int site, const ChunkInfo& chunk) {
        const int layer_idx = 0;
        const int64_t rows = chunk.len, local_q_heads = q / tp, local_kv_heads = kv / tp;
        const uint32_t input_residual = 1;
''' + dispatch + r'''
        throw std::runtime_error("unknown launch site");
    }
};
class DerivedDecoder : public CausalDecoderModel {};

template<class F> void must_reject(F fn) {
    launched.clear();
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && launched.empty());
}

void routes_for(CausalDecoderModel& model) {
    LayoutContext layout;
    for (int64_t rows : {1, 4, 15, 16, 17, 58, 64, 128, 160, 256, 314, 320, 512}) {
        FmbThreeStageChunkPlan plan;
        // The native route is owned by each compute chunk, independent of
        // QKV partition and physical/logical total (e.g. VL P314 ->320).
        plan.qkv.chunks = {{0, 0, rows + 16, rows + 16}};
        plan.compute.chunks = {{0, 0, 16, 16}, {1, 16, rows, rows + 16}};
        const auto manifest = model.plan_for_test(plan, layout);
        assert(manifest.routes.size() == 6);
        model.bind(manifest);
        for (const auto& chunk : plan.compute.chunks) {
            for (int site = 0; site < 5; ++site) {
                launched.clear();
                model.dispatch(site, chunk);
                assert(launched.size() == 1);
                const auto& got = launched[0];
                const int64_t norm_rows = chunk.len * (site == 1 ? model.q / model.tp :
                                                       site == 2 ? model.kv / model.tp : 1);
                const int64_t cols = site == 1 || site == 2 ? model.hd : model.h;
                assert(got.rows == norm_rows && got.cols == cols && got.eps == model.eps_);
                assert(got.gamma == static_cast<uint32_t>((site + 1) * 10));
                const auto expected =
                    cols <= 8192 && norm_rows % 32 == 0
                        ? RpuRmsNormSpmRoute::V32_GRID8
                        : cols <= 8192 && norm_rows % 16 == 0
                            ? RpuRmsNormSpmRoute::V16_GRID8
                            : RpuRmsNormSpmRoute::BASE;
                assert(got.route == expected);
                if (site == 4) assert(got.input == got.output); // final RMS stays in place.
                const int64_t role = site == 1 ? 1 : site == 2 ? 2 : 0;
                for (int corruption = 0; corruption < 8; ++corruption) {
                    model.bind(manifest);
                    for (auto& entry : model.context.manifest_.routes) {
                        if (entry.invocation != 3 * chunk.idx + role) continue;
                        if (corruption == 0) entry.selector = -1;
                        if (corruption == 1) entry.flags = 1;
                        if (corruption == 2) ++entry.arguments[0];
                        if (corruption == 3) ++entry.arguments[1];
                        if (corruption == 4) ++entry.arguments[2];
                        if (corruption == 5) ++entry.site_id;
                        if (corruption == 6) entry.family = FmbRouteFamily::LINEAR;
                        if (corruption == 7) entry.invocation += 100;
                    }
                    must_reject([&] { model.dispatch(site, chunk); });
                }
                model.bind(manifest);
                model.context.complete = false;
                must_reject([&] { model.dispatch(site, chunk); });
                model.context.complete = true;
            }
        }
    }
    for (int64_t rows : {0, -16}) {
        FmbThreeStageChunkPlan plan;
        plan.compute.chunks = {{0, 0, rows, rows}};
        must_reject([&] { model.plan_for_test(plan, layout); });
    }
}

void routes() {
    for (int profile = 0; profile < 8; ++profile) {
        CausalDecoderModel model;
        if (profile == 1 || profile == 5) {
            model.h = 2048; model.intermediate = 6144; model.layers = 28; model.q = 16;
        }
        if (profile == 2 || profile == 3) {
            model.h = 2560; model.intermediate = 9728; model.layers = 36; model.q = 32;
        }
        if (profile == 3 || profile == 4 || profile == 5) {
            model.has_mrope_ = true; model.deepstack_lang_layers_ = {0, 1, 2};
        }
        if (profile == 6) {
            // Qwen3-0.6B: previously excluded by an exact model whitelist.
            model.h = 1024; model.intermediate = 3072;
            model.layers = 28; model.q = 16;
        }
        if (profile == 7) {
            // Unknown ordinary geometry proves route admission is operator-level.
            model.h = 3072; model.intermediate = 8192;
            model.layers = 24; model.q = 24;
        }
        routes_for(model); // Known Qwen/VL sizes plus generic ordinary geometry.
    }
}

void isolation() {
    FmbThreeStageChunkPlan plan;
    plan.compute.chunks = {{0, 0, 128, 128}};

    // Q/K norms are site-local. A decoder without them still routes the
    // input/post/final direct-SPM norm through the generic contract.
    {
        CausalDecoderModel model;
        model.has_qk_norm_ = false;
        const auto manifest = model.plan_for_test(plan, {});
        assert(manifest.routes.size() == 1);
        assert(manifest.routes[0].invocation == 0);
        model.bind(manifest);
        for (int site : {0, 3, 4}) {
            launched.clear();
            model.dispatch(site, plan.compute.chunks[0]);
            assert(launched.size() == 1);
            assert(launched[0].route == RpuRmsNormSpmRoute::V32_GRID8);
        }
    }

    // A runtime payload without vector kernels snapshots BASE and publishes no
    // ordinary RMS route authority.
    {
        CausalDecoderModel model;
        model.ordinary_rmsnorm_spm_contract_.capability =
            RPU_RMSNORM_CAP_BASE;
        assert(!model.ordinary_rmsnorm_route_enabled());
        assert(model.plan_for_test(plan, {}).routes.empty());
        model.bind({});
        model.context.complete = false;
        for (int role = 0; role < 3; ++role) {
            assert(model.consume_ordinary_rmsnorm_route(
                128, role == 0 ? model.h : model.hd, 0, role) ==
                RpuRmsNormSpmRoute::BASE);
        }
    }

    // Loaded capability controls the widest route independently of model size.
    for (uint32_t cap : {
             RPU_RMSNORM_CAP_GENERIC_DIRECT_SPM,
             RPU_RMSNORM_CAP_BASE | RPU_RMSNORM_CAP_V32,
             RPU_RMSNORM_CAP_ALL}) {
        CausalDecoderModel model;
        model.h = 1024; model.intermediate = 3072;
        model.layers = 28; model.q = 16; // Qwen3-0.6B
        model.ordinary_rmsnorm_spm_contract_.capability = cap;
        FmbThreeStageChunkPlan p;
        p.compute.chunks = {{0, 0, 16, 16}};
        const auto manifest = model.plan_for_test(p, {});
        assert(manifest.routes.size() == 3);
        for (const auto& entry : manifest.routes) {
            const int role = static_cast<int>(entry.invocation);
            const auto expected =
                (cap & RPU_RMSNORM_CAP_V32) && entry.arguments[0] % 32 == 0
                    ? RpuRmsNormSpmRoute::V32_GRID8
                    : (cap & RPU_RMSNORM_CAP_V16) &&
                            entry.arguments[0] % 16 == 0
                        ? RpuRmsNormSpmRoute::V16_GRID8
                        : RpuRmsNormSpmRoute::BASE;
            assert(entry.selector == static_cast<int64_t>(expected));
            assert(entry.arguments[2] == cap);
            assert(role >= 0 && role <= 2);
        }
    }

    // Width is shape eligibility, not capability: only the full-row site falls
    // back when C exceeds the vector payload limit.
    {
        CausalDecoderModel model;
        model.h = 8193;
        FmbThreeStageChunkPlan p;
        p.compute.chunks = {{0, 0, 32, 32}};
        const auto manifest = model.plan_for_test(p, {});
        assert(manifest.routes.size() == 3);
        assert(manifest.routes[0].selector ==
               static_cast<int64_t>(RpuRmsNormSpmRoute::BASE));
        assert(manifest.routes[1].selector ==
               static_cast<int64_t>(RpuRmsNormSpmRoute::V32_GRID8));
        assert(manifest.routes[2].selector ==
               static_cast<int64_t>(RpuRmsNormSpmRoute::V32_GRID8));
    }

    // Different numerical/dtype routes and incompatible traversals stay
    // isolated from the ordinary contract.
    for (int mode = 0; mode < 4; ++mode) {
        CausalDecoderModel model;
        LayoutContext layout;
        if (mode == 0) model.adarms_ = true;
        if (mode == 1) model.qwen3vl_2b_w8_profile_ = true;
        if (mode == 2) layout.is_causal = false;
        if (mode == 3) layout.batch_size = 2;
        assert(model.plan_for_test(plan, layout).routes.empty());
    }

    DerivedDecoder derived;
    assert(derived.ordinary_rmsnorm_route_enabled());
    assert(derived.plan_for_test(plan, {}).routes.size() == 3);

    CausalDecoderModel model;
    plan.chunk_mode = ChunkMode::KV_FIRST;
    const auto kv_first_manifest = model.plan_for_test(plan, {});
    assert(kv_first_manifest.routes.empty());
    model.bind(kv_first_manifest, {}, ChunkMode::KV_FIRST);
    model.context.complete = false;
    for (int role = 0; role < 3; ++role) {
        assert(model.consume_ordinary_rmsnorm_route(
            128, role == 0 ? model.h : model.hd, 0, role) ==
            RpuRmsNormSpmRoute::BASE);
    }
    plan.chunk_mode = ChunkMode::SEQUENTIAL;

    // Existing exact2B delivery keeps V64 prefill and BASE decode, including
    // Q/K launch routing. It must never consume the ordinary site.
    model.qwen3vl_2b_w8_profile_ = true;
    model.h = 2048; model.intermediate = 6144; model.layers = 28; model.q = 16;
    for (int64_t rows : {1, 192}) {
        Manifest manifest;
        for (int role = 0; role < 3; ++role) {
            int64_t n = rows * (role == 1 ? 2 : 1);
            auto route = n % 64 == 0 ? RpuRmsNormSpmRoute::QWEN3VL_V64 : RpuRmsNormSpmRoute::BASE;
            manifest.routes.push_back({FmbRouteFamily::NORMALIZATION, QWEN3VL_DELIVERY_RMS_SITE,
                                      static_cast<int64_t>(route), 0, {n, role == 0 ? 2048 : 128}, role});
        }
        model.bind(manifest);
        for (int site = 0; site < 5; ++site) {
            launched.clear();
            model.dispatch(site, {0, 0, rows, rows});
            assert(launched.size() == 1);
            assert(launched[0].route == (rows == 192 ? RpuRmsNormSpmRoute::QWEN3VL_V64 : RpuRmsNormSpmRoute::BASE));
        }
    }
}
int main(int argc, char** argv) {
    assert(argc == 2);
    if (std::string(argv[1]) == "routes") routes();
    else if (std::string(argv[1]) == "isolation") isolation();
    else return 1;
}
'''
    path = root / "qwen_rms.cpp"
    path.write_text(program)
    executable = root / "qwen_rms"
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(executable)], check=True)
    return executable


def test_qwen_rms_manifest_dispatch_and_stale_descriptor_rejection(qwen_rms_executable):
    subprocess.run([str(qwen_rms_executable), "routes"], check=True)


def test_qwen_rms_isolation_and_existing_delivery_routes(qwen_rms_executable):
    subprocess.run([str(qwen_rms_executable), "isolation"], check=True)
