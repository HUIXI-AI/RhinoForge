"""Actual C++ cold profile/geometry on CPU doubles, not model/HW evidence."""
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _body(source, signature):
    start = source.index(signature)
    begin = source.index("{", start)
    end = source.index("\n}", begin) + 2
    return source[begin:end]


def test_g05_k1_native_fast_replay_keeps_build_and_other_vision_profiles(tmp_path):
    """Run the actual owner policy and FMB skip branch on CPU graph doubles."""
    import ast

    adapter = ast.parse((ROOT / "python/rpu_backend/adapters/qwen3_5/vision.py").read_text())
    precision = next(node for node in adapter.body
                     if isinstance(node, ast.FunctionDef) and node.name == "_qwen35_vision_precision")
    namespace = {}
    exec(compile(ast.Module(body=[precision], type_ignores=[]), "vision.py", "exec"), namespace)
    resolve_precision = namespace["_qwen35_vision_precision"]
    assert resolve_precision(None, certified_g05=True) == (True, False)
    assert resolve_precision({"vision": {"linear_acc32": True}}) == (True, True)
    assert resolve_precision(None) == (False, False)

    source = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    skip = base[base.index("    const bool fast_skip = !checked_body_replay &&"):]
    skip = skip[:skip.index("    const FmbExecutionTraversalSpec execution_spec{")]
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
struct ChunkInfo {};
struct ChunkPlan {};
struct FusedModelBase {};
struct ModelStaticConfig {
    int num_layers=0, cross_layer_batch_size=0, body_iterations=1;
    bool fast_replay_skip_layer_loop=false, fast_replay_skip_preload=false;
    bool fast_replay_bake_post_fn=false, batch_decode_active=false;
    bool replay_reemit_layer_body=false;
    void (FusedModelBase::*preload_fn)()=nullptr;
    void (FusedModelBase::*pre_layers_fn)()=nullptr;
    void (FusedModelBase::*post_layers_fn)()=nullptr;
    void (FusedModelBase::*post_fn)()=nullptr;
    void (FusedModelBase::*kv_first_fn)(int,const ChunkInfo&)=nullptr;
    ChunkPlan (FusedModelBase::*kv_first_chunk_plan_fn)(const ChunkPlan&)=nullptr;
};
struct Qwen3_5VisionModel : FusedModelBase {
    int cores=8, attn=8, mlp=8, layers=24, hidden=1024, intermediate=4096;
    int q=16, kv=16, hd=64, orig_head_dim_=64, patch_dim_=1536;
    int merger_hidden_=4096, out_hidden_size_=2048;
    int current_temporal_num_frames_=1, current_camera_batch_count_=1;
    int current_output_patches_=256;
    double eps_=1e-6;
    bool linear_acc32_=true, gelu_erf_ultra_=false, moe_precision_=false,g05_w8a16_=false,g05_qkv_spm_available_=false;
    bool has_temporal_=false, has_rope_=true, has_step0_=true, has_merger_=true;
    bool z2_bound_=false, debug=false, current_compact_step0_=false;
    int num_cores() const { return cores; }
    int attn_tp() const { return attn; }
    int mlp_tp() const { return mlp; }
    int num_layers() const { return layers; }
    int hidden_size() const { return hidden; }
    int intermediate_size() const { return intermediate; }
    int num_q_heads() const { return q; }
    int num_kv_heads() const { return kv; }
    int head_dim() const { return hd; }
    bool get_debug_export() const { return debug; }
    bool is_g05_k1_b3_profile() const { return false; }
    bool is_g05_k1_camera_group_profile() const { return false; }
    void emit_preload_weights() {}
    void emit_step0() {}
    void emit_merger() {}
    void emit_kv_first_body(int,const ChunkInfo&) {}
    ChunkPlan plan_kv_first_chunks(const ChunkPlan&) { return {}; }
    bool is_g05_k1_fast_replay_profile() const __PROFILE__
    ModelStaticConfig static_config() __CONFIG__
};
struct RpuKernelGraph {
    enum class State { RECORDING, REPLAYING, PASSTHROUGH };
    State current=State::RECORDING;
    int skip_stream=0, skip_body=0;
    bool has_post_fn_cursor() const { return false; }
    State state() const { return current; }
    void skip_op_stream_for_fast_replay() { ++skip_stream; }
    void skip_layer_body_for_fast_replay() { ++skip_body; }
    static bool has_active();
    static RpuKernelGraph& active();
};
RpuKernelGraph graph;
bool active_graph=true;
bool RpuKernelGraph::has_active() { return active_graph; }
RpuKernelGraph& RpuKernelGraph::active() { return graph; }
struct Context {
    int inherited=0;
    bool has_complete_physical_manifest() const { return true; }
    void inherit_physical_routes_from_matching_graph_replay() { ++inherited; }
};
struct ReplayPatch { int kernel_id=0, reg_lo=0, value=0, expected_count=0; };
enum class KernelId { unused };
void patch_replay_kernel_u32(KernelId,int,int,int) { assert(false); }
struct Impl { uint64_t pipeline_lease_epoch_=0; Context ctx_;
    std::vector<ReplayPatch> replay_kernel_u32_patches_; } impl;
struct RuntimePolicy { bool fmb_fast_replay=false; };
int actual_body_iterations(const ModelStaticConfig& static_cfg,
                           bool matching_physical_manifest_replay_branch=true) {
    Impl* pimpl_=&impl;
    const RuntimePolicy* graph_runtime_policy=nullptr;
    const bool checked_body_replay=false, callback_yield_replay_run=false;
    __SKIP__
    return body_iters;
}
int main() {
    Qwen3_5VisionModel v;
    const auto cfg=v.static_config();
    assert(v.is_g05_k1_fast_replay_profile());
    assert(cfg.fast_replay_skip_layer_loop && cfg.fast_replay_skip_preload);
    assert(!cfg.fast_replay_bake_post_fn && cfg.post_fn==nullptr);
    assert(cfg.pre_layers_fn==static_cast<void(FusedModelBase::*)()>(&Qwen3_5VisionModel::emit_step0));
    assert(cfg.post_layers_fn==static_cast<void(FusedModelBase::*)()>(&Qwen3_5VisionModel::emit_merger));
    auto excluded=[&](auto alter) {
        auto other=v; alter(other); const auto c=other.static_config();
        assert(!c.fast_replay_skip_layer_loop && !c.fast_replay_skip_preload);
    };
    excluded([](auto& x){ x.linear_acc32_=false; });
    excluded([](auto& x){ x.gelu_erf_ultra_=true; });
    excluded([](auto& x){ x.moe_precision_=true; });
    excluded([](auto& x){ x.cores=4; });
    excluded([](auto& x){ x.attn=4; });
    excluded([](auto& x){ x.mlp=4; });
    excluded([](auto& x){ x.layers=27; });
    excluded([](auto& x){ x.hidden=1152; });
    excluded([](auto& x){ x.intermediate=4352; });
    excluded([](auto& x){ x.q=12; });
    excluded([](auto& x){ x.kv=8; });
    excluded([](auto& x){ x.hd=80; });
    excluded([](auto& x){ x.orig_head_dim_=72; });
    excluded([](auto& x){ x.eps_=1e-5; });
    excluded([](auto& x){ x.has_rope_=false; });
    excluded([](auto& x){ x.has_step0_=false; });
    excluded([](auto& x){ x.patch_dim_=768; });
    excluded([](auto& x){ x.has_merger_=false; });
    excluded([](auto& x){ x.merger_hidden_=4608; });
    excluded([](auto& x){ x.out_hidden_size_=2560; });
    excluded([](auto& x){ x.z2_bound_=true; });
    excluded([](auto& x){ x.debug=true; });
    excluded([](auto& x){ x.current_temporal_num_frames_=6; });
    excluded([](auto& x){ x.current_camera_batch_count_=3; });
    excluded([](auto& x){ x.current_compact_step0_=true; });
    excluded([](auto& x){ x.current_output_patches_=128; });
    auto temporal=v; temporal.has_temporal_=true;
    assert(!temporal.is_g05_k1_fast_replay_profile());
    assert(temporal.static_config().fast_replay_skip_layer_loop);
    assert(temporal.static_config().fast_replay_skip_preload);

    // The production FMB branch must still emit BUILD and immediate work.
    assert(actual_body_iterations(cfg)==1 && impl.ctx_.inherited==0);
    graph.current=RpuKernelGraph::State::PASSTHROUGH;
    assert(actual_body_iterations(cfg)==1 && graph.skip_stream==0);
    graph.current=RpuKernelGraph::State::REPLAYING;
    active_graph=false;
    assert(actual_body_iterations(cfg)==1);
    active_graph=true;
    assert(actual_body_iterations(cfg)==0 && graph.skip_stream==1);
    assert(impl.ctx_.inherited==1 && graph.skip_body==0);
    bool rejected=false;
    try { actual_body_iterations(cfg,false); } catch(const std::exception&) { rejected=true; }
    assert(rejected && impl.ctx_.inherited==1);
    impl.pipeline_lease_epoch_=1;
    assert(actual_body_iterations(cfg)==1);
    impl.pipeline_lease_epoch_=0;
    auto ordinary=v; ordinary.linear_acc32_=false;
    assert(actual_body_iterations(ordinary.static_config())==1);
    auto w8=ordinary;w8.g05_w8a16_=true;
    assert(w8.is_g05_k1_fast_replay_profile());
    assert(w8.static_config().fast_replay_skip_layer_loop&&w8.static_config().fast_replay_skip_preload);
    assert(actual_body_iterations(w8.static_config())==0);
    auto fp16=ordinary;fp16.g05_qkv_spm_available_=true;
    assert(fp16.is_g05_k1_fast_replay_profile());
    assert(fp16.static_config().fast_replay_skip_layer_loop&&fp16.static_config().fast_replay_skip_preload);
    assert(actual_body_iterations(fp16.static_config())==0);
    graph.current=RpuKernelGraph::State::RECORDING;
    assert(actual_body_iterations(w8.static_config())==1);
    assert(actual_body_iterations(fp16.static_config())==1);
}
'''
    for marker, value in (
        ("__PROFILE__", _body(source, "bool Qwen3_5VisionModel::is_g05_k1_fast_replay_profile()")),
        ("__CONFIG__", _body(source, "ModelStaticConfig Qwen3_5VisionModel::static_config()")),
        ("__SKIP__", skip),
    ):
        cpp = cpp.replace(marker, value)
    path, binary = tmp_path / "g05_replay.cpp", tmp_path / "g05_replay"
    path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                               str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr


def test_vision_ring_preparation_uses_actual_owner_budget(tmp_path):
    """The actual Vision call must not inherit the shared helper's TP8 default."""
    source = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    layer = _body(source, "void Qwen3_5VisionModel::build_layer_subgraph(")
    call, = re.findall(r"rpu_prepare_ring_all_reduce_input\([^;]+;", layer)
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    declaration = re.search(
        r"void rpu_prepare_ring_all_reduce_input\([^;]+;", declarations).group()
    shared = (ROOT / "src/core/rpu_runtime_extras.cpp").read_text()
    graph = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>
template<class... A> std::string message(A... args) {
    std::ostringstream out; (out << ... << args); return out.str();
}
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(message(__VA_ARGS__)); } while(false)
namespace c10 { struct Half { uint16_t value=0; explicit Half(float) {} }; }
struct SpmAllocator { static constexpr uint64_t ALIGN=256; };
constexpr int kRingCoreNum=8;
struct GraphRuntimePolicy { int execution_core_count=8; } policy;
uint8_t validated_kernel_core_extent(const GraphRuntimePolicy& policy,
    const std::vector<uint8_t>& core_ids) __CORE_GUARD__
int fills=0, fill_cores=0;
void rpu_launch_fill_spm_kernel(uint32_t, int64_t, c10::Half, int cores) {
    std::vector<uint8_t> ids;
    for (int core=0; core<cores; ++core) ids.push_back(static_cast<uint8_t>(core));
    validated_kernel_core_extent(policy, ids);
    ++fills; fill_cores=cores;
}
__PREPARE_DECLARATION__
void rpu_prepare_ring_all_reduce_input(uint32_t input_spm, int64_t M, int64_t N,
    int input_num_cores, int output_num_cores) __PREPARE_BODY__
struct VisionCaller {
    int owner;
    int num_cores() const { return owner; }
    uint32_t addr(int, const char*) const { return 0x40001000; }
    void prepare(int tp, int64_t seq_len, int64_t h) { __VISION_CALL__ }
};
int main() {
    // Real 2B/4B and padded 9B widths, including a compute tail. Equal
    // producer/consumer domains must not emit an unnecessary fill kernel.
    for (int owner : {4,8}) {
        policy.execution_core_count=owner;
        for (int64_t width : {768,1024,1152}) {
            for (int64_t rows : {624,600}) {
                VisionCaller{owner}.prepare(owner, rows, width);
                assert(fills==0);
            }
        }
    }
    // A legacy owner8 / attention4 partial producer still needs the full
    // consumer-domain initialization before the row-parallel write.
    policy.execution_core_count=8;
    VisionCaller{8}.prepare(4,624,1024);
    assert(fills==1 && fill_cores==8);
    // The actual Graph authority must still reject an explicit foreign
    // consumer domain. Do not repair this regression by relaxing the guard.
    policy.execution_core_count=4;
    bool rejected=false;
    try { rpu_prepare_ring_all_reduce_input(0x40001000,624,1024,4,8); }
    catch (const std::runtime_error& error) {
        rejected=std::string(error.what()).find("Graph kernel core count exceeds execution_core_count")!=std::string::npos;
    }
    assert(rejected && fills==1);
}
'''
    for marker, value in (
        ("__CORE_GUARD__", _body(graph, "uint8_t validated_kernel_core_extent(")),
        ("__PREPARE_DECLARATION__", declaration),
        ("__PREPARE_BODY__", _body(shared, "void rpu_prepare_ring_all_reduce_input(")),
        ("__VISION_CALL__", call),
    ):
        cpp = cpp.replace(marker, value)
    source_path, binary = tmp_path / "vision_ring.cpp", tmp_path / "vision_ring"
    source_path.write_text(cpp)
    compiled = subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source_path), "-o", str(binary)],
        capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    execution = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert execution.returncode == 0, execution.stderr


def test_actual_qwen35_cold_profile_and_temporal_admission(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_qwen3_5_vision_model.h").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    constants = source[source.index("constexpr int64_t G05_MAX_CAMERA_BATCH"):
                       source.index("// STEP 0 row-chunk.")]
    constructor = ("{}" if "Qwen3_5VisionModel() = default;" in header else
                   _body(source, "Qwen3_5VisionModel::Qwen3_5VisionModel()"))
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
__CONSTANTS__
namespace at {
constexpr int kPrivateUse1=1, kHalf=2;
struct Tensor {
    int device_type=1, dtype=2; bool contiguous=true;
    int64_t rows=6, width=1024, rank=2;
    struct Device { int value; int type() const { return value; } };
    Device device() const { return {device_type}; }
    int scalar_type() const { return dtype; }
    bool is_contiguous() const { return contiguous; }
    bool defined() const { return false; }
    int64_t dim() const { return rank; }
    int64_t size(int axis) const { return axis ? width : rows; }
};
}
struct FusedModelBase {
    struct ChunkEnvelope { int64_t max_kv_len=0, chunk=0; };
    struct Impl {
        ChunkEnvelope chunk_envelope_;
        std::vector<int> kv_cost_candidates_;
        std::optional<int> kv_cost_planning_context_;
        int kv_cost_layout_scope_=0;
        std::string kv_cost_snapshot_reason_;
        int64_t planner_cache_epoch_=0;
        std::vector<int> prepared_stage_candidates_, prepared_spm_costs_;
    } impl;
    Impl* pimpl_=&impl;
    int64_t layers=24, hidden=1024, heads=16, dimension=64, last=0;
    int cores=8;
    int num_cores() const { return cores; }
    int64_t num_layers() const { return layers; }
    int64_t hidden_size() const { return hidden; }
    int64_t intermediate_size() const { return 4096; }
    int64_t num_q_heads() const { return heads; }
    int64_t num_kv_heads() const { return heads; }
    int64_t head_dim() const { return dimension; }
    int64_t attn_tp() const { return 8; }
    int64_t get_last_resolved_chunk_size() const { return last; }
    std::vector<int64_t> kvinsert_cost_weight_identity() const { return {1}; }
    const ChunkEnvelope& chunk_envelope() const { return impl.chunk_envelope_; }
    void set_chunk_envelope(int64_t max_kv_len, int64_t chunk);
    void invalidate_planner_cache() __INVALIDATE_PLANNER__
    std::vector<int64_t> installed_profile_identity() const;
};
void FusedModelBase::set_chunk_envelope(int64_t max_kv_len, int64_t chunk) __SETTER__
std::vector<int64_t> FusedModelBase::installed_profile_identity() const __PROFILE__
struct Qwen3_5VisionModel : FusedModelBase {
    at::Tensor temporal_pe_, position_idx_keepalive_,q_ddr_buf_,hidden_buf_,merged_buf_;
    int64_t temporal_num_frames_=0, temporal_patches_per_frame_=0;
    bool has_temporal_=false, has_step0_=true, high_precision_=false;
    int64_t patch_dim_=1536, current_temporal_num_frames_=1, current_camera_batch_count_=1;
    bool current_compact_step0_=false;
    int64_t current_output_patches_=0, generation=0;
    int64_t cache_capacity_=0, keepalive_capacity_=0;
    Qwen3_5VisionModel();
    void configure_cache_capacity(int64_t);
    void set_temporal(const at::Tensor&, int64_t, int64_t);
    void configure_execution_geometry(int64_t, int64_t, int64_t, bool);
    void invalidate_model_state() { ++generation; }
    bool g05_k1_b3_available() const { return false; }
    bool g05_k1_camera_group_available(int64_t) const { return false; }
};
Qwen3_5VisionModel::Qwen3_5VisionModel() __CTOR__
void Qwen3_5VisionModel::configure_cache_capacity(int64_t max_seq_len) __CAPACITY__
void Qwen3_5VisionModel::set_temporal(const at::Tensor& temporal_pe,
    int64_t num_frames, int64_t patches_per_frame) __TEMPORAL__
void Qwen3_5VisionModel::configure_execution_geometry(int64_t execution_len,
    int64_t temporal_num_frames, int64_t camera_batch_count, bool compact_input) __GEOMETRY__
template<class F> void reject(F body) {
    bool rejected=false;
    try { body(); } catch (const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
void profile(const Qwen3_5VisionModel& model) {
    for (int64_t value : model.installed_profile_identity()) std::cout << value << ' ';
    std::cout << '\n';
}
int main() {
    Qwen3_5VisionModel generic;
    assert(generic.chunk_envelope().max_kv_len == 4096);
    assert(generic.chunk_envelope().chunk == 0);
    const auto before=generic.installed_profile_identity();
    generic.configure_execution_geometry(4096,1,1,false);
    reject([&] { generic.configure_execution_geometry(4097,1,1,false); });
    reject([&] { generic.configure_execution_geometry(4608,6,3,false); });
    assert(generic.installed_profile_identity()==before && generic.generation==0);
    profile(generic);
    // Existing tensor/profile guards must run before publishing temporal capacity.
    for (int fault=0; fault<12; ++fault) {
        Qwen3_5VisionModel bad;
        at::Tensor tensor;
        int frames=6, patches=256;
        if (fault==0) bad.last=16;
        if (fault==1) bad.layers=23;
        if (fault==2) bad.hidden=512;
        if (fault==3) frames=5;
        if (fault==4) patches=128;
        if (fault==5) tensor.device_type=0;
        if (fault==6) tensor.dtype=0;
        if (fault==7) tensor.contiguous=false;
        if (fault==8) tensor.rank=3;
        if (fault==9) tensor.rows=5;
        if (fault==10) tensor.width=1023;
        if (fault==11) bad.cores=4;
        const auto original=bad.installed_profile_identity();
        reject([&] { bad.set_temporal(tensor,frames,patches); });
        assert(bad.installed_profile_identity()==original && !bad.has_temporal_ && bad.generation==0);
    }
    generic.impl.kv_cost_candidates_={11};
    generic.impl.kv_cost_planning_context_=3;
    generic.impl.kv_cost_layout_scope_=5;
    generic.impl.kv_cost_snapshot_reason_="READY";
    generic.set_temporal(at::Tensor{},6,256);
    assert(generic.chunk_envelope().max_kv_len==G05_TEMPORAL_MAX_SEQ);
    assert(generic.chunk_envelope().chunk==0 && generic.generation==1);
    assert(generic.impl.kv_cost_candidates_.empty() && !generic.impl.kv_cost_planning_context_);
    assert(generic.impl.kv_cost_layout_scope_==0 &&
        generic.impl.kv_cost_snapshot_reason_=="NATIVE_STAGE_ORACLE_REQUIRED");
    generic.configure_execution_geometry(4608,6,3,true);
    assert(generic.current_output_patches_==768);
    generic.configure_execution_geometry(1536,6,1,false);
    reject([&] { generic.configure_execution_geometry(4608,1,1,false); });
    reject([&] { generic.configure_execution_geometry(4609,6,3,false); });
    reject([&] { generic.configure_execution_geometry(4096,6,3,false); });
    assert(generic.chunk_envelope().max_kv_len==4608);
    profile(generic);
    Qwen3_5VisionModel wide;
    wide.layers=0;
    wide.configure_cache_capacity(32768);
    wide.layers=24;
    wide.configure_execution_geometry(16384,1,1,false);
    wide.configure_execution_geometry(32768,1,1,false);
    reject([&] { wide.configure_execution_geometry(32769,1,1,false); });
    wide.set_temporal(at::Tensor{},6,256);
    assert(wide.chunk_envelope().max_kv_len==32768);
    wide.configure_execution_geometry(4608,6,3,true);
    Qwen3_5VisionModel huge;
    huge.layers=0;
    huge.configure_cache_capacity(2097152);
    huge.layers=24;
    huge.configure_execution_geometry(1048560,1,1,false);
    reject([&] { huge.configure_execution_geometry(1048576,1,1,false); });
    Qwen3_5VisionModel reduced;
    reduced.cores=4;
    reduced.configure_execution_geometry(4096,1,1,false);
    reject([&] { reduced.configure_execution_geometry(1536,6,1,false); });
    reject([&] { reduced.configure_execution_geometry(4608,6,3,true); });
    assert(reduced.current_temporal_num_frames_==1 &&
           reduced.current_camera_batch_count_==1 && !reduced.current_compact_step0_);
}
'''
    for marker, value in (("__CONSTANTS__", constants), ("__CTOR__", constructor),
        ("__CAPACITY__", _body(source, "void Qwen3_5VisionModel::configure_cache_capacity(")),
        ("__SETTER__", _body(base, "void FusedModelBase::set_chunk_envelope(")),
        ("__INVALIDATE_PLANNER__", _body(base, "void FusedModelBase::invalidate_planner_cache()")),
        ("__PROFILE__", _body(base, "std::vector<int64_t> FusedModelBase::installed_profile_identity()")),
        ("__TEMPORAL__", _body(source, "void Qwen3_5VisionModel::set_temporal(")),
        ("__GEOMETRY__", _body(source, "void Qwen3_5VisionModel::configure_execution_geometry("))):
        cpp = cpp.replace(marker, value)
    source_path, binary = tmp_path / "profile.cpp", tmp_path / "profile"
    source_path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
        str(source_path), "-o", str(binary)], capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    execution = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert execution.returncode == 0, execution.stderr
    profiles = [[int(value) for value in line.split()] for line in execution.stdout.splitlines()]
    assert [words[8:10] for words in profiles] == [[4096, 0], [4608, 0]]


def test_existing_python_cache_and_native_finer_bounds_stay_independent():
    source = (ROOT / "python/rpu_backend/adapters/qwen3_5/vision.py").read_text()
    assert "or int(max_seq_len) <= 0" in source
    assert re.search(r"4608 if int\(temporal_num_frames\) > 1 else 256\*int\(camera_batch_count\) if int\(camera_batch_count\) in \(2,3\) else min\(\s*65535 \* 16,", source)
    assert '"max_seq_len", logical_len)' in source
    native = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    setter = _body(native, "void Qwen3_5VisionModel::set_chunk_size_cap(")
    exact = _body(native, "void Qwen3_5VisionModel::set_prefill_chunk_size(")
    hot = _body(native, "void Qwen3_5VisionModel::stage_prefill_execution_controls(")
    assert all("set_chunk_envelope" not in body for body in (setter, exact, hot))


def test_actual_native_topology_is_cold_and_matches_vision_partition(tmp_path):
    """Compile the production admission methods; no backend or board is loaded."""
    source = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <numeric>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "execution_topology.h"
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
using v3::DecoderExecutionTopology;
using v3::resolve_decoder_execution_topology;
struct FusedModelBase {
    struct Impl { int num_cores_=8; int64_t hidden_size_=0; bool valid_=false; } impl;
    Impl* pimpl_=&impl;
    int64_t layers=0, last=0;
    int attn=8, mlp=8;
    int num_cores() const { return impl.num_cores_; }
    int attn_tp() const { return attn; }
    int mlp_tp() const { return mlp; }
    int64_t num_layers() const { return layers; }
    int64_t get_last_resolved_chunk_size() const { return last; }
    void set_execution_core_count(int num_cores) __BASE_SETTER__
    DecoderExecutionTopology resolve_model_execution_topology(
        int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
        int64_t hidden_size, int64_t intermediate_size) const __BASE_RESOLVE__
};
struct Qwen3_5VisionModel : FusedModelBase {
    bool linear_acc32_=false, gelu_erf_ultra_=false, has_temporal_=false, z2_bound_=false;
    bool moe_precision_=false, high_precision_=false;
    std::vector<int> layer_weights_;
    void invalidate_model_state() {}
    void configure_execution_cores(int64_t cores) __CORES__
    std::vector<int64_t> execution_topology() const __TOPOLOGY__
    DecoderExecutionTopology resolve_model_execution_topology(
        int64_t q, int64_t kv, int64_t d, int64_t h, int64_t intermediate) const __RESOLVE__
    void set_linear_acc32(bool enabled) __ACC32__
    void install_geometry(int count, int heads, int dim, int hidden, int intermediate, double epsilon) {
        std::vector<int> q_w_list(count);
        const int64_t num_heads=heads, head_dim=dim, hidden_size=hidden, intermediate_size=intermediate;
        const double eps=epsilon;
        __ADMISSION__
    }
};
template<class F> void reject(F body) {
    bool rejected=false;
    try { body(); } catch (const std::exception&) { rejected=true; }
    assert(rejected);
}
int main() {
    Qwen3_5VisionModel v;
    for(int n : {0,1,2,3,5,6,7,9,256}) reject([&]{ v.configure_execution_cores(n); });
    assert(v.num_cores()==8);
    // Use the real FMB default resolver: its legacy min(8, KV) contract must
    // remain unchanged for other models. Vision uses the same whole-head
    // partition as its existing Python swizzle and native QKV launchers.
    assert(v.FusedModelBase::resolve_model_execution_topology(12,12,64,768,3072).attention_tp==8);
    for (int heads : {12,16}) {
        const auto t=v.resolve_model_execution_topology(heads,heads,64,heads*64,heads*256);
        assert(t.num_cores==8 && t.attention_tp==std::gcd(heads,8) && t.mlp_tp==8);
        v.layers=12; v.attn=t.attention_tp; v.mlp=t.mlp_tp;
        assert((v.execution_topology()==std::vector<int64_t>{8,std::gcd(heads,8),8,8}));
    }
    v.layers=0;
    reject([&]{ v.execution_topology(); });
    v.configure_execution_cores(4);
    for(bool acc32 : {false,true}) for(const auto& geometry : std::vector<std::vector<int>>{{12,12,64,768,3072}, {24,16,64,1024,4096}, {27,16,72,1152,4304}}) {
        v.set_linear_acc32(acc32);
        v.install_geometry(geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],1e-6);
        for(int axis=0;axis<5;++axis) {
            auto bad=geometry; ++bad[axis];
            reject([&]{ v.install_geometry(bad[0],bad[1],bad[2],bad[3],bad[4],1e-6); });
        }
        reject([&]{v.install_geometry(geometry[0],geometry[1],geometry[2],geometry[3],geometry[4],1e-5);});
    }
    const auto small=v.resolve_model_execution_topology(12,12,64,768,3072);
    assert(small.num_cores==4 && small.attention_tp==4 && small.mlp_tp==4);
    reject([&]{v.resolve_model_execution_topology(12,8,64,768,3072);});
    reject([&]{v.resolve_model_execution_topology(12,12,80,768,3072);});
    reject([&]{v.resolve_model_execution_topology(12,12,64,768,3328);});
    const auto regular=v.resolve_model_execution_topology(16,16,64,1024,4096);
    const auto padded=v.resolve_model_execution_topology(16,16,80,1152,4352);
    assert(regular.num_cores==4 && padded.attention_tp==4 && padded.mlp_tp==4);
    reject([&]{v.resolve_model_execution_topology(16,16,72,1152,4304);});
    reject([&]{v.resolve_model_execution_topology(16,8,64,1024,4096);});
    v.set_linear_acc32(true);
    assert(v.linear_acc32_);
    v.layers=24; v.attn=v.mlp=4;
    reject([&]{v.set_linear_acc32(false);});
    assert(v.linear_acc32_);
    assert((v.execution_topology()==std::vector<int64_t>{4,4,4,8}));
    v.impl.hidden_size_=1024;
    reject([&]{v.configure_execution_cores(8);});
    assert(v.num_cores()==4);
    Qwen3_5VisionModel acc;
    acc.set_linear_acc32(true);
    acc.configure_execution_cores(4);
    assert(acc.num_cores()==4 && acc.linear_acc32_);
    acc.install_geometry(24,16,64,1024,4096,1e-6);
    acc.configure_execution_cores(8);
    assert(acc.resolve_model_execution_topology(16,16,64,1024,4096).attention_tp==8);
    assert(acc.resolve_model_execution_topology(12,12,64,768,3072).attention_tp==4);
    acc.install_geometry(12,12,64,768,3072,1e-6);
    for(int fault=0;fault<3;++fault) {
        Qwen3_5VisionModel busy;
        if(fault==0)busy.has_temporal_=true;
        if(fault==1)busy.z2_bound_=true;
        if(fault==2)busy.last=16;
        reject([&]{busy.configure_execution_cores(4);});
        assert(busy.num_cores()==8);
    }
}
'''
    admission = _body(source, "void Qwen3_5VisionModel::set_weights(")
    admission = admission[1:admission.index("    const int64_t padded_head_dim")]
    for marker, value in (
        ("__BASE_SETTER__", _body(base, "void FusedModelBase::set_execution_core_count(")),
        ("__BASE_RESOLVE__", _body(base, "DecoderExecutionTopology FusedModelBase::resolve_model_execution_topology(")),
        ("__CORES__", _body(source, "void Qwen3_5VisionModel::configure_execution_cores(")),
        ("__TOPOLOGY__", _body(source, "std::vector<int64_t> Qwen3_5VisionModel::execution_topology()")),
        ("__RESOLVE__", _body(source, "DecoderExecutionTopology Qwen3_5VisionModel::resolve_model_execution_topology(")),
        ("__ACC32__", _body(source, "void Qwen3_5VisionModel::set_linear_acc32(")),
        ("__ADMISSION__", admission),
    ):
        cpp = cpp.replace(marker, value)
    path, binary = tmp_path / "cores.cpp", tmp_path / "cores"
    path.write_text(cpp)
    result = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                             "-I", str(ROOT / "src/core"), str(path), "-o", str(binary)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr


def test_actual_native_merger_admits_exact_profiles_before_installation(tmp_path):
    """Execute the entire production setter with metadata-only tensor doubles.

    This covers the separate merger admission that rejected the real 0.8B
    owner4 load after the tower had passed its own profile checks. Tensor
    storage/swizzle contents and model numerics require the hardware test.
    """
    source = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    merge_unit = re.search(
        r"constexpr int64_t QWEN3_5_SPATIAL_MERGE_UNIT\s*=\s*\d+;", source
    ).group()
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
template<class... A> std::string message(A... args) {
    std::ostringstream out; (out << ... << args); return out.str();
}
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(message(__VA_ARGS__)); } while(false)
__MERGE_UNIT__
namespace at {
constexpr int kPrivateUse1=1, kHalf=2;
struct Tensor {
    std::vector<int64_t> shape;
    bool present=false, contiguous=true;
    int dtype=kHalf, device_type=kPrivateUse1;
    Tensor() = default;
    Tensor(std::initializer_list<int64_t> dimensions) : shape(dimensions), present(true) {}
    struct Device { int value; int type() const { return value; } };
    Device device() const { return {device_type}; }
    bool defined() const { return present; }
    int scalar_type() const { return dtype; }
    bool is_contiguous() const { return contiguous; }
    int64_t dim() const { return static_cast<int64_t>(shape.size()); }
    int64_t size(int axis) const { return shape.at(axis); }
};
}
struct Qwen3_5VisionModel {
    int owner=8;
    int64_t hidden=0, out_hidden_size_=0, merger_hidden_=0, generation=0;
    bool has_merger_=false, moe_precision_=false;
    at::Tensor merger_fc1_w_, merger_fc1_b_, merger_fc2_w_, merger_fc2_b_;
    at::Tensor merger_norm_w_, merger_norm_b_;
    int num_cores() const { return owner; }
    int64_t hidden_size() const { return hidden; }
    void invalidate_model_state() { ++generation; }
    void set_merger(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t);
};
void Qwen3_5VisionModel::set_merger(
    const at::Tensor& fc1_w, const at::Tensor& fc1_b,
    const at::Tensor& fc2_w, const at::Tensor& fc2_b,
    const at::Tensor& norm_w, const at::Tensor& norm_b,
    int64_t out_hidden_size) __SET_MERGER__
using Weights = std::array<at::Tensor,6>;
Weights weights(int64_t h, int64_t out) {
    return {at::Tensor{h*4,h*4}, at::Tensor{h*4}, at::Tensor{out,h*4},
            at::Tensor{out}, at::Tensor{h}, at::Tensor{h}};
}
void install(Qwen3_5VisionModel& model, const Weights& w, int64_t out) {
    model.set_merger(w[0],w[1],w[2],w[3],w[4],w[5],out);
}
void reject(int owner, int64_t h, int64_t out, const Weights& w,
            const std::string& diagnostic) {
    Qwen3_5VisionModel model;
    model.owner=owner; model.hidden=h;
    bool rejected=false;
    try { install(model,w,out); }
    catch (const std::runtime_error& error) {
        rejected=std::string(error.what()).find(diagnostic)!=std::string::npos;
    }
    assert(rejected);
    // Every rejection must precede tensor retention and state publication.
    assert(!model.has_merger_ && model.generation==0);
    assert(model.out_hidden_size_==0 && model.merger_hidden_==0);
    assert(!model.merger_fc1_w_.defined() && !model.merger_fc1_b_.defined());
    assert(!model.merger_fc2_w_.defined() && !model.merger_fc2_b_.defined());
    assert(!model.merger_norm_w_.defined() && !model.merger_norm_b_.defined());
}
int main() {
    // All admitted exact reduced widths also keep working on the default
    // owner8 path. Pairs are 0.8B, 2B, 4B, 35B-A3B, 9B and official 27B.
    for (int owner : {4,8}) {
        for (const auto& profile : std::vector<std::array<int64_t,2>>{
                {768,1024}, {1024,2048}, {1024,2560},
                {1152,2048}, {1152,4096}, {1152,5120}}) {
            const int64_t h=profile[0], out=profile[1];
            Qwen3_5VisionModel model;
            model.owner=owner; model.hidden=h;
            const auto w=weights(h,out);
            install(model,w,out);
            assert(model.has_merger_ && model.generation==1);
            assert(model.out_hidden_size_==out && model.merger_hidden_==h*4);
            assert(model.merger_fc1_w_.shape==w[0].shape);
            assert(model.merger_fc1_b_.shape==w[1].shape);
            assert(model.merger_fc2_w_.shape==w[2].shape);
            assert(model.merger_fc2_b_.shape==w[3].shape);
            assert(model.merger_norm_w_.shape==w[4].shape);
            assert(model.merger_norm_b_.shape==w[5].shape);
            // Packed tensors retain logical [out,in] metadata. Corrupt each
            // axis/bias/norm independently; none may partially install.
            for (size_t tensor=0; tensor<w.size(); ++tensor) {
                for (size_t axis=0; axis<w[tensor].shape.size(); ++axis) {
                    auto bad=w; ++bad[tensor].shape[axis];
                    reject(owner,h,out,bad,"must be [");
                }
            }
        }
    }
    // Reduced admission stays exact even for otherwise aligned tensors.
    for (const auto& profile : std::vector<std::array<int64_t,2>>{
            {768,2048}, {768,2560}, {768,4096}, {1024,1024},
            {1024,4096}, {1152,1024}, {1152,2560}, {1152,3072},
            {896,1024}, {1024,5120}, {1152,5136}}) {
        reject(4,profile[0],profile[1],weights(profile[0],profile[1]),
               "exact 0.8B/2B/4B/9B/27B/35B-A3B output width");
        // These are valid on the established generic owner8 contract.
        Qwen3_5VisionModel legacy;
        legacy.hidden=profile[0];
        install(legacy,weights(profile[0],profile[1]),profile[1]);
        assert(legacy.has_merger_ && legacy.generation==1);
    }
    // Actual tensor/device and swizzle-shape boundaries remain enforced.
    for (size_t tensor=0; tensor<6; ++tensor) {
        for (int fault=0; fault<5; ++fault) {
            auto bad=weights(768,1024);
            std::string diagnostic;
            if(fault==0) { bad[tensor].present=false; diagnostic="defined"; }
            if(fault==1) { bad[tensor].dtype=0; diagnostic="fp16"; }
            if(fault==2) { bad[tensor].device_type=0; diagnostic="on device"; }
            if(fault==3) { bad[tensor].contiguous=false; diagnostic="contiguous"; }
            if(fault==4) { bad[tensor].shape.push_back(1); diagnostic="D, got"; }
            reject(4,768,1024,bad,diagnostic);
        }
    }
    reject(4,0,1024,weights(768,1024),"call set_weights() first");
    reject(4,768,0,weights(768,1024),"must be positive");
    reject(8,767,1024,weights(767,1024),"divisible by 16*tp");
    reject(8,768,1025,weights(768,1025),"divisible by 16");
}
'''
    cpp = cpp.replace("__MERGE_UNIT__", merge_unit).replace(
        "__SET_MERGER__", _body(source, "void Qwen3_5VisionModel::set_merger("))
    path, binary = tmp_path / "merger.cpp", tmp_path / "merger"
    path.write_text(cpp)
    compiled = subprocess.run(
        ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(binary)],
        capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr
