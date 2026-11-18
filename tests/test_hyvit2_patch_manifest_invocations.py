"""Host-only HYViT2 patch admission, manifest and device retirement regressions."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(
    os.environ.get(
        "RHINOFORGE_TEST_ROOT",
        Path(__file__).resolve().parents[1],
    )
)
SOURCE = ROOT / "src/fused/rpu_hyvit2_vision_model.cpp"


def _balanced_block(source: str, marker: str, *, trailing_semicolon: bool = False) -> str:
    begin = source.index(marker)
    opening = (
        source.index(") {", begin) + 2
        if "const auto append" in marker else source.index("{", begin)
    )
    depth = 0
    for cursor in range(opening, len(source)):
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
            if depth == 0:
                end = cursor + 1
                if trailing_semicolon:
                    assert source[end] == ";"
                    end += 1
                return source[begin:end]
    raise AssertionError(f"unterminated block: {marker}")


def test_three_image_external_patch_routes_use_invocation_not_flags(tmp_path: Path) -> None:
    """Execute the production append block for the packed S588/s240 profile."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host-native manifest test")

    source = SOURCE.read_text(encoding="utf-8")
    append_lambda = _balanced_block(
        source,
        "        const auto append = [&](FmbRouteFamily family, int64_t site_id,",
        trailing_semicolon=True,
    )
    patch_prologue = _balanced_block(
        source,
        "        if (planning_external_patch_prologue_) {",
    )
    patch_constants = "\n".join(
        re.findall(
            r"constexpr int64_t HYVIT2_PATCH_(?:INPUT_DMA|LINEAR|ALL_GATHER|POSITION_DMA|OUTPUT_DMA)_SITE\s*=\s*[^;]+;",
            source,
        )
    )
    assert patch_constants.count("constexpr int64_t") == 5
    patch_route_enum = _balanced_block(
        source,
        "enum class HyViT2PatchMutableDmaRoute : int64_t",
        trailing_semicolon=True,
    )

    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
#define DWIDTH 2

enum class FmbRouteFamily : int64_t {
    ATTENTION = 1, LINEAR = 2, ALL_REDUCE = 3, ROPE = 4, KV_INSERT = 5,
    MUTABLE_DMA = 6, NORMALIZATION = 7, ACTIVATION = 8,
    GRAPH_SCHEDULE = 9, COLLECTIVE = 10,
};
enum class FmbLinearRouteSelector : int64_t { AUTO_TILE = 1 };
__PATCH_ROUTE_ENUM__
__PATCH_CONSTANTS__

using RpuAllGatherSchedule = int64_t;
RpuAllGatherSchedule rpu_resolve_all_gather_schedule(int64_t, int64_t) {
    return 1;
}

struct FmbExecutionSpan {
    int64_t offset;
    int64_t len;
    int64_t group_id;
};
struct FmbThreeStageChunkPlan {
    std::vector<FmbExecutionSpan> spans;
};
struct FmbRouteManifestEntry {
    int64_t site_id;
    FmbRouteFamily family;
    int64_t selector;
    int64_t flags;
    std::vector<int64_t> arguments;
    int64_t invocation;
};
struct FmbPhysicalExecutionManifest {
    std::vector<FmbRouteManifestEntry> routes;
};

struct Harness {
    bool planning_external_patch_prologue_ = true;
    bool has_patch_emb_ = true;
    int64_t image_batch_count_ = 3;
    int64_t pe_kh_ = 16;
    int64_t pe_kw_ = 16;
    int64_t pe_cin_orig_ = 3;
    int64_t pe_cin_padded_ = 16;
    int64_t pe_cout_ = 1152;
    int64_t pe_strideh_ = 16;
    int64_t pe_stridew_ = 16;
    int64_t patch_embed_cores_ = 8;

    FmbPhysicalExecutionManifest build(const FmbThreeStageChunkPlan& plan) const {
        FmbPhysicalExecutionManifest manifest;
__APPEND_LAMBDA__
__PATCH_PROLOGUE__
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
};

int main() {
    const FmbThreeStageChunkPlan plan{{
        {0, 196, 0}, {196, 196, 1}, {392, 196, 2},
    }};
    const auto manifest = Harness{}.build(plan);
    assert(manifest.routes.size() == 15);

    const std::vector<int64_t> sites{
        HYVIT2_PATCH_INPUT_DMA_SITE,
        HYVIT2_PATCH_LINEAR_SITE,
        HYVIT2_PATCH_ALL_GATHER_SITE,
        HYVIT2_PATCH_POSITION_DMA_SITE,
        HYVIT2_PATCH_OUTPUT_DMA_SITE,
    };
    std::set<std::tuple<int64_t, int64_t, int64_t>> identities;
    for (const auto& route : manifest.routes) {
        assert(route.flags == 0);
        assert(identities.emplace(
            static_cast<int64_t>(route.family), route.site_id,
            route.invocation).second);
    }
    for (const int64_t site : sites) {
        std::vector<int64_t> invocations;
        for (const auto& route : manifest.routes) {
            if (route.site_id == site) invocations.push_back(route.invocation);
        }
        assert((invocations == std::vector<int64_t>{0, 1, 2}));
    }

    std::vector<int64_t> output_offsets;
    for (const auto& route : manifest.routes) {
        if (route.site_id == HYVIT2_PATCH_OUTPUT_DMA_SITE) {
            output_offsets.push_back(route.arguments.at(0));
        }
    }
    assert((output_offsets == std::vector<int64_t>{0, 451584, 903168}));
}
'''
    for key, value in {
        "PATCH_ROUTE_ENUM": patch_route_enum,
        "PATCH_CONSTANTS": patch_constants,
        "APPEND_LAMBDA": append_lambda,
        "PATCH_PROLOGUE": patch_prologue,
    }.items():
        program = program.replace(f"__{key}__", value)

    cpp = tmp_path / "hyvit2-manifest-invocations.cpp"
    binary = tmp_path / "hyvit2-manifest-invocations"
    cpp.write_text(program, encoding="utf-8")
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run(
        [str(binary)], capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stderr


def test_patch_geometry_preserves_batch_semantics_and_rejects_bad_inputs(tmp_path):
    """Execute the native preflight and setter with allocation-free metadata."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host-native geometry test")
    source = SOURCE.read_text(encoding="utf-8")
    preflight = _balanced_block(source, "    int64_t validate_patch_inputs(")
    setter = _balanced_block(source, "    void set_patch_emb_params(")
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
#define DWIDTH 2
constexpr int64_t kMaxPackedImages = 8;
namespace at {
constexpr int kHalf = 1, kPrivateUse1 = 2;
struct Tensor {
    std::vector<int64_t> shape;
    int dtype = kHalf, device_type = kPrivateUse1;
    bool contiguous = true;
    bool defined() const { return !shape.empty(); }
    int64_t dim() const { return shape.size(); }
    int64_t size(int i) const { return shape.at(i); }
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    int scalar_type() const { return dtype; }
    struct Device { int value; int type() const { return value; } };
    Device device() const { return {device_type}; }
    bool is_contiguous() const { return contiguous; }
};
using TensorList = std::vector<Tensor>;
}
struct Model {
    bool has_patch_emb_ = false;
    int64_t patch_kernel_size_ = 0, patch_stride_ = 0;
    int64_t pe_kh_ = 0, pe_kw_ = 0, pe_strideh_ = 0, pe_stridew_ = 0;
    int64_t pe_cin_orig_ = 3, pe_cin_padded_ = 0, pe_cout_ = 0;
    int patch_embed_cores_ = 8;
    at::Tensor patch_emb_weight_, patch_emb_pos_emb_;
__SETTER__
__PREFLIGHT__
};
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
int main() {
    Model model;
    const at::Tensor weight{{1152, 16 * 16 * 16}};
    const at::Tensor positions{{1, 196, 1152}};
    const at::Tensor image{{1, 3, 224, 224}};
    model.set_patch_emb_params(weight, positions, 16, 16);
    // Single tensor B=N and N list entries must pack the same number of rows.
    assert(model.validate_patch_inputs({{{3, 3, 224, 224}}}, true) == 588);
    assert(model.validate_patch_inputs({image, image, image}, false) == 588);
    assert(model.validate_patch_inputs({{{8, 3, 224, 224}}}, true) == 1568);
    rejects([&] { model.validate_patch_inputs({{{2, 3, 224, 224}}}, false); });
    rejects([&] { model.validate_patch_inputs({image, {{1, 3, 240, 240}}}, false); });
    rejects([&] { model.validate_patch_inputs({{{1, 1, 224, 224}}}, true); });
    rejects([&] { model.validate_patch_inputs({{{9, 3, 224, 224}}}, true); });
    rejects([&] { model.validate_patch_inputs(std::vector<at::Tensor>(9, image), false); });
    rejects([&] { model.validate_patch_inputs({}, false); });
    rejects([&] { model.validate_patch_inputs({{{1, 3, 224, 225}}}, true); });
    rejects([&] { model.validate_patch_inputs({{{1, 3, 16, 0}}}, true); });
    // Width is the permute stride hazard; a bottom height tail is valid.
    assert(model.validate_patch_inputs({{{1, 3, 225, 224}}}, true) == 196);
    auto invalid = image;
    invalid.dtype = 0;
    rejects([&] { model.validate_patch_inputs({invalid}, true); });
    invalid = image;
    invalid.device_type = 0;
    rejects([&] { model.validate_patch_inputs({invalid}, true); });
    invalid = image;
    invalid.contiguous = false;
    rejects([&] { model.validate_patch_inputs({invalid}, true); });
    model.set_patch_emb_params(weight, {{1, 195, 1152}}, 16, 16);
    rejects([&] { model.validate_patch_inputs({image}, true); });
    model.set_patch_emb_params(weight, positions, 16, 16);
    // A failed setter must leave the previous valid configuration intact.
    rejects([&] { model.set_patch_emb_params(weight, positions, 0, 16); });
    assert(model.pe_kh_ == 16 && model.has_patch_emb_);
    rejects([&] { model.set_patch_emb_params({{1152, 16 * 16 * 4}}, positions, 16, 16); });
    assert(model.pe_cin_padded_ == 16);
    // Single-core im2col supports width tails and overlapping patches.
    model.patch_embed_cores_ = 1;
    assert(model.validate_patch_inputs({{{1, 3, 224, 225}}}, true) == 196);
    model.set_patch_emb_params(weight, {{1, 729, 1152}}, 16, 8);
    assert(model.validate_patch_inputs({image}, true) == 729);
    model.patch_embed_cores_ = 8;
    rejects([&] { model.validate_patch_inputs({image}, true); });
}
'''.replace("__SETTER__", setter).replace("__PREFLIGHT__", preflight)
    cpp = tmp_path / "hyvit2-patch-geometry.cpp"
    binary = tmp_path / "hyvit2-patch-geometry"
    cpp.write_text(program, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)


def test_complete_patch_encoder_retires_every_stream_before_arena_reuse(tmp_path):
    """Execute the native boundary and allocator; SDK timing needs a board run."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host-native retirement test")
    source = SOURCE.read_text(encoding="utf-8")
    eligible = _balanced_block(source, "    bool patch_encoder_device_fence_eligible(")
    retirement = _balanced_block(source, "    void retire_patch_temporaries(")
    reset = _balanced_block(
        (ROOT / "src/core/rpu_spm_allocator.cpp").read_text(),
        "void SpmAllocator::reset_temporary()",
    )
    program = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
#define NUM_CORES 8
enum class FmbPhysicalManifestState {UNSPECIFIED, COMPLETE};
enum class FmbGraphLifecycle {UNSPECIFIED, COMPOSITE_CHILD};
enum class FmbLinearAccumulationPolicy {ACC16, ACC32};
struct Manifest {
    FmbPhysicalManifestState state = FmbPhysicalManifestState::COMPLETE;
    FmbGraphLifecycle graph_lifecycle = FmbGraphLifecycle::COMPOSITE_CHILD;
    FmbLinearAccumulationPolicy linear_accumulation = FmbLinearAccumulationPolicy::ACC16;
    int64_t logical_length=392,physical_length=392,execution_padding_rows=0,kv_logical_length=392;
};
struct Context {
    bool complete=true;
    mutable int reads=0;
    Manifest manifest;
    bool has_complete_physical_manifest() const {++reads;return complete;}
    const Manifest&physical_manifest()const{++reads;return manifest;}
};
struct GraphSignature {};
struct RpuKernelGraph {
    enum class State {RECORDING,REPLAYING,PASSTHROUGH,BUILT};
    State current=State::RECORDING;
    std::vector<std::pair<int,int>> events;
    int branches=0;
    static RpuKernelGraph&active();
    State state()const{return current;}
    void record_barrier(uint8_t self,uint8_t target){events.push_back({self,target});}
    void record_branch(uint64_t key,GraphSignature,const std::string&label){
        assert(key==0x5349474c49505f52ull&&label=="hyvit2_patch_embed_spm_reset");
        ++branches;events.push_back({-1,-1});
    }
} graph;
RpuKernelGraph&RpuKernelGraph::active(){return graph;}
struct SpmAllocator {
    bool pipeline_arena_locked_=false;
    int64_t pipeline_arena_epoch_=23,t_end_=456789,generation_=5;
    int64_t persistent_generation_=17,p_start_=6000000,sp_floor_=6000000;
    void reset_temporary();
} allocator;
#define SPM_ALLOC allocator
__RESET__
struct Harness {
    Context context;
    bool has_patch_emb_=true;
    int64_t patch_embed_cores_=8,pe_cin_orig_=3,pe_cin_padded_=16,pe_cout_=1152;
    int64_t pe_kh_=16,pe_kw_=16,pe_strideh_=16,pe_stridew_=16,orig_head_dim_=72;
    int num_cores()const{return 8;}
    int64_t layers=27,h=1152,inter=4352,q=16,kv=16,hd=80;
    std::vector<int>layer_weights_=std::vector<int>(27);
    const Context&ctx()const{return context;}
    int64_t num_layers()const{return layers;}
    int64_t hidden_size()const{return h;}
    int64_t intermediate_size()const{return inter;}
    int64_t num_q_heads()const{return q;}
    int64_t num_kv_heads()const{return kv;}
    int64_t head_dim()const{return hd;}
__ELIGIBLE__
__RETIREMENT__
};
void init(RpuKernelGraph::State state=RpuKernelGraph::State::RECORDING){
    graph=RpuKernelGraph{};graph.current=state;allocator=SpmAllocator{};
}
void check_reset(){
    assert(allocator.t_end_==0&&allocator.generation_==6);
    assert(allocator.persistent_generation_==17&&allocator.p_start_==6000000&&allocator.sp_floor_==6000000);
}
void check_fence(){
    assert(graph.branches==0&&graph.events.size()==8);
    std::array<bool,8> outstanding;outstanding.fill(true);
    for(int i=0;i<8;++i){
        assert((graph.events[i]==std::make_pair(0,2*i)));
        // SYNC offset=0 retires all preceding STARTs on its target stream.
        outstanding[graph.events[i].second/2]=false;
    }
    for(bool pending:outstanding)assert(!pending);
    check_reset();
}
void check_branch(){
    assert(graph.branches==1&&graph.events.size()==1);
    assert((graph.events[0]==std::make_pair(-1,-1)));check_reset();
}
int main(){
    for(auto state:{RpuKernelGraph::State::RECORDING,RpuKernelGraph::State::REPLAYING}){
        for(int count:{2,3}){
            init(state);Harness h;
            h.context.manifest.logical_length=h.context.manifest.physical_length=
                h.context.manifest.kv_logical_length=count*196;
            h.retire_patch_temporaries(true,count,224,224,196);check_fence();
        }
        // A valid descriptor alone cannot admit a standalone patch call.
        init(state);Harness h;
        h.retire_patch_temporaries(false,2,224,224,196);check_branch();
        assert(h.context.reads==0);
        for(auto args:std::vector<std::array<int64_t,4>>{
                {1,224,224,196},{4,224,224,196},{2,225,224,196},
                {2,224,225,196},{2,224,224,195}}){
            init(state);Harness h;
            h.retire_patch_temporaries(true,args[0],args[1],args[2],args[3]);check_branch();
        }
        const std::vector<std::function<void(Harness&)>> faults{
            [](auto&h){h.context.complete=false;},
            [](auto&h){h.has_patch_emb_=false;},
            [](auto&h){h.patch_embed_cores_=1;},
            [](auto&h){h.pe_cin_orig_=4;},[](auto&h){h.pe_cin_padded_=32;},
            [](auto&h){h.pe_cout_=1280;},[](auto&h){h.pe_kh_=8;},
            [](auto&h){h.pe_kw_=8;},[](auto&h){h.pe_strideh_=8;},
            [](auto&h){h.pe_stridew_=8;},[](auto&h){h.layers=26;},
            [](auto&h){h.layer_weights_.pop_back();},[](auto&h){h.h=1280;},
            [](auto&h){h.inter=4304;},[](auto&h){h.q=8;},[](auto&h){h.kv=8;},
            [](auto&h){h.hd=72;},[](auto&h){h.orig_head_dim_=80;},
            [](auto&h){h.context.manifest.state=FmbPhysicalManifestState::UNSPECIFIED;},
            [](auto&h){h.context.manifest.graph_lifecycle=FmbGraphLifecycle::UNSPECIFIED;},
            [](auto&h){h.context.manifest.linear_accumulation=FmbLinearAccumulationPolicy::ACC32;},
            [](auto&h){h.context.manifest.logical_length=588;},
            [](auto&h){h.context.manifest.physical_length=400;},
            [](auto&h){h.context.manifest.execution_padding_rows=8;},
            [](auto&h){h.context.manifest.kv_logical_length=400;},
        };
        for(const auto&fault:faults){
            init(state);Harness h;fault(h);
            h.retire_patch_temporaries(true,2,224,224,196);check_branch();
        }
        // The new fence cannot grant authority to reset a locked physical arena.
        for(bool prologue:{false,true}){
            init(state);allocator.pipeline_arena_locked_=true;Harness h;bool rejected=false;
            try{h.retire_patch_temporaries(prologue,2,224,224,196);}
            catch(const std::runtime_error&){rejected=true;}
            assert(rejected&&allocator.t_end_==456789&&allocator.generation_==5);
        }
    }
    for(auto state:{RpuKernelGraph::State::PASSTHROUGH,RpuKernelGraph::State::BUILT}){
        init(state);Harness h;h.retire_patch_temporaries(true,2,224,224,196);
        assert(graph.events.empty()&&h.context.reads==0);check_reset();
    }
}
'''.replace("__ELIGIBLE__", eligible).replace("__RETIREMENT__", retirement).replace("__RESET__", reset)
    # The loop closes all image output DMAs before the arena retirement call.
    packed = _balanced_block(source, "    at::Tensor run_packed_patch_embed_core(")
    assert packed.index("fused_patch_embedding(") < packed.index("retire_patch_temporaries(")
    assert retirement.index("graph.record_barrier(") < retirement.index("SPM_ALLOC.reset_temporary()")
    cpp, binary = tmp_path / "hyvit2-retirement.cpp", tmp_path / "hyvit2-retirement"
    cpp.write_text(program, encoding="utf-8")
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", str(cpp), "-o", str(binary)],
        capture_output=True, text=True, timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
