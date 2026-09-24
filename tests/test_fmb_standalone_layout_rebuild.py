"""Execute the real FMB allocation state machine with a host-only SPM arena.

Only device initialization and unrelated owner bookkeeping are replaced. The
production caller, allocation paths, allocator arithmetic/aliasing and complete
layout hash are compiled from the checkout, so no RPU library/device is loaded.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _definition(source, prefix):
    start = source.index(prefix)
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda m: " " * len(m[0]), source)
    brace = masked.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def _host_program(core_source):
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    impl = (ROOT / "src/core/fused_model_base_impl.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    # Use the actual production argument expression, not a test reimplementation
    # of the COMPLETE/expected-layout/lease admission condition.
    run = _definition(core_source, "at::Tensor FusedModelBase::run_all_layers_impl(")
    call_start = run.index("const bool standalone_complete_layout =")
    allocation = run.index("ensure_allocated_impl(", call_start)
    call = run[call_start:run.index(";", allocation) + 1]
    types = "\n".join(_definition(header, prefix) + ";" for prefix in (
        "enum class StorageClass", "enum class BufferScope", "struct BufferDecl",
        "enum class FmbForwardOperandResidency", "struct LayoutContext", "struct SpmPipelineComponentLayout"))
    owner = _definition(impl, "struct OwnedPipelineDecl") + ";"
    support = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("native contract rejected"); } while (0)
enum class AttentionExecutionPolicy {DDR_KV, SPM_KV_BY_MHA};
enum class FmbRopeTableResidency {UNSPECIFIED};
class FusedModelBase;
'''
    support += types
    support += "\nnamespace detail {" + _definition(header, "inline int64_t layout_mix(") + "}\n"
    support += r'''
class SpmAllocator {
public:
    static constexpr size_t ALIGN=256, SPM_USABLE=8*1024*1024;
    struct AllocRequest {int64_t bytes;int first_step,last_step,scope=0;};
    struct AliasedPlan {std::vector<uint32_t> offsets;size_t peak_bytes=0;};
    static size_t align_up(size_t n){return (n+ALIGN-1)&~(ALIGN-1);}
    bool initialized_=true,pipeline_arena_locked_=false;
    uint64_t pipeline_arena_epoch_=0,generation_=0,persistent_generation_=0;
    size_t t_end_=0,sp_floor_=SPM_USABLE,p_start_=SPM_USABLE;
    bool is_initialized()const{return initialized_;}
    void init(){throw std::runtime_error("host test must never initialize the device");}
    uint64_t generation()const{return generation_;}
    uint64_t persistent_generation()const{return persistent_generation_;}
    size_t persistent_used()const{return sp_floor_-p_start_;}
    uint32_t alloc_temporary(size_t);
    uint32_t alloc_super_persistent(size_t);
    static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);
    std::vector<uint32_t> alloc_temporary_aliased(const std::vector<AllocRequest>&);
    void reset_temporary();
    void reset_all();
} arena;
#define SPM_ALLOC arena
'''
    for prefix in (
        "uint32_t SpmAllocator::alloc_temporary(",
        "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(",
        "std::vector<uint32_t> SpmAllocator::alloc_temporary_aliased(",
        "uint32_t SpmAllocator::alloc_super_persistent(",
        "void SpmAllocator::reset_temporary(", "void SpmAllocator::reset_all(",
    ):
        support += _definition(allocator, prefix) + "\n"
    support += r'''
class FusedModelBase {
public:
    struct Impl {
        __OWNER__
        struct {bool valid=false;} pipeline_manifest_snapshot_;
        struct {
            bool complete=true;
            bool has_complete_physical_manifest()const{return complete;}
        } ctx_;
        uint64_t pipeline_lease_epoch_=0;
        int64_t num_q_heads_=8,num_kv_heads_=8,head_dim_=64;
        int64_t hidden_size_=1024,intermediate_size_=4096;
        int64_t cached_params_hash_=0,persistent_layout_hash_=0;
        bool valid_=false,allocation_declaration_valid_=false;
        bool persistent_allocated_=false,pipeline_occurrence_schedule_present_=false;
        bool weights_dirty_=false,preload_callbacks_dirty_=false;
        uint64_t allocation_declaration_hash_=0,alloc_gen_=0,cached_persistent_gen_=0;
        uint64_t persistent_layout_version_=0,manifest_generation=0;
        size_t persistent_usage_=0,temp_per_layer_usage_=0;
        std::vector<OwnedPipelineDecl> allocation_declarations_;
        std::unordered_map<std::string,uint32_t> offsets_,persistent_offsets_;
        std::vector<std::unordered_map<std::string,uint32_t>> per_layer_offsets_,persistent_per_layer_offsets_;
    };
};
using OwnedPipelineDecl=FusedModelBase::Impl::OwnedPipelineDecl;
// No sealed pipeline is built by this allocation-only test. Bookkeeping has
// no effect on arena arithmetic, declaration identity or layout hashes.
void retire_pipeline_manifest(FusedModelBase::Impl& s){s.pipeline_manifest_snapshot_.valid=false;}
void advance_pipeline_manifest_generation(FusedModelBase::Impl& s){++s.manifest_generation;}
constexpr uint64_t kFmbFnvOffset=0xcbf29ce484222325ULL,kFmbFnvPrime=0x100000001b3ULL;
'''.replace("__OWNER__", owner)
    for prefix in (
        "uint64_t fmb_hash_u64(", "uint64_t fmb_hash_string(", "uint64_t fmb_nonzero_hash(",
        "static int64_t compute_params_hash_impl(", "static int64_t compute_persistent_hash_impl(",
        "static size_t align_spm_bytes(", "static uint64_t pipeline_layout_hash_impl(",
        "static std::vector<OwnedPipelineDecl> own_pipeline_decls(",
        "static uint64_t owned_pipeline_decl_hash(", "static bool same_owned_pipeline_declarations(",
        "static void allocate_temp_aliased_impl(", "static void resolve_aliases_impl(",
        "static void allocate_persistent_impl(", "static void ensure_allocated_impl(",
        "struct CpuTemporaryLayoutPlan", "CpuTemporaryLayoutPlan plan_cpu_temporary_layout(",
        "SpmPipelineComponentLayout current_temporary_layout_identity(",
    ):
        support += _definition(core_source, prefix) + (";\n" if prefix.startswith("struct ") else "\n")
    support += r'''
void production_allocation(FusedModelBase::Impl& state,const std::vector<BufferDecl>& decls,
                           const LayoutContext& alloc_ctx,uint64_t expected_layout_hash) {
    auto* pimpl_=&state;
    auto estimate_fn=[](const LayoutContext&){return int64_t{0};};
    auto subclass_layout_hash=[](){return int64_t{0};};
    __CALL__
}
SpmPipelineComponentLayout planned(const FusedModelBase::Impl& s,
                                  const std::vector<BufferDecl>& decls,const LayoutContext& ctx) {
    const auto p=plan_cpu_temporary_layout(decls);
    const auto h=compute_params_hash_impl(ctx,s.num_q_heads_,s.num_kv_heads_,s.head_dim_,
                                        s.hidden_size_,s.intermediate_size_,0);
    return {p.required_extent,pipeline_layout_hash_impl(decls,h,p.required_extent,p.offsets,p.per_layer_offsets)};
}
SpmPipelineComponentLayout actual(const FusedModelBase::Impl& s,const std::vector<BufferDecl>& decls) {
    return current_temporary_layout_identity(decls,s.cached_params_hash_,s.offsets_,s.per_layer_offsets_);
}
std::vector<BufferDecl> buffers() {
    return {{"norm",256,0,0,StorageClass::PersistentPerLayer,2},
            {"saved",512,0,3,StorageClass::TempPerLayer,2},
            {"q",2048,0,1,StorageClass::Temp},
            {"out",4096,2,3,StorageClass::Temp},
            {"q_alias",2048,0,1,StorageClass::Temp,0,"q"}};
}
LayoutContext geometry(int64_t chunk) {
    LayoutContext c;c.chunk_size=chunk;c.num_layers=2;
    c.stage_plan_fingerprint=55;c.physical_manifest_fingerprint=91;return c;
}
int alternating_owners() {
    arena=SpmAllocator{};const auto decls=buffers();
    FusedModelBase::Impl vision,text;
    const auto vision_ctx=geometry(624);const auto expected=planned(vision,decls,vision_ctx);
    uint64_t stable_persistent=0;
    std::unordered_map<std::string,uint32_t> first_offsets;
    std::vector<std::unordered_map<std::string,uint32_t>> first_persistent;
    for(int generation=0;generation<6;++generation) {
        const auto before=arena.generation();
        production_allocation(vision,decls,vision_ctx,expected.layout_hash);
        const auto current=actual(vision,decls);
        if(current.layout_hash!=expected.layout_hash || current.temporary_bytes!=expected.temporary_bytes) {
            std::cerr<<"alternating owner generation "<<generation<<" expected extent="<<expected.temporary_bytes
                     <<" actual="<<current.temporary_bytes<<"; allocated layout differs\n";
            return 86;
        }
        if(generation==0){first_offsets=vision.offsets_;first_persistent=vision.persistent_per_layer_offsets_;}
        else {
            assert(vision.offsets_==first_offsets && vision.persistent_per_layer_offsets_==first_persistent);
            assert(arena.generation()==before+1); // Exactly one reset, no Path1 double reset.
            assert(arena.persistent_generation()==stable_persistent);
        }
        // Successful Vision exits before the independent Text owner starts.
        arena.reset_temporary();
        const auto text_ctx=geometry(1+generation); // Prefill/decode changes Text's layout.
        const auto text_expected=planned(text,decls,text_ctx);
        production_allocation(text,decls,text_ctx,text_expected.layout_hash);
        assert(arena.t_end_>0 && actual(text,decls).layout_hash==text_expected.layout_hash);
        stable_persistent=arena.persistent_generation();
    }
    std::cout<<"six alternating same-shape Vision calls retain their complete layout\n";
    return 0;
}
void protected_scopes() {
    const auto decls=buffers();const auto ctx=geometry(624);
    // COMPLETE standalone owners retain placement even without an explicit hash.
    // Non-COMPLETE and active lease/composite owners keep their existing scopes.
    for(int excluded=0;excluded<3;++excluded) {
        arena=SpmAllocator{};FusedModelBase::Impl s;
        const auto expected=planned(s,decls,ctx);
        production_allocation(s,decls,ctx,expected.layout_hash);
        const auto old_offset=s.offsets_.at("q");
        arena.reset_temporary();arena.alloc_temporary(8192);
        if(excluded==1)s.ctx_.complete=false;
        if(excluded==2)s.pipeline_lease_epoch_=9;
        const auto before=arena.generation();
        production_allocation(s,decls,ctx,excluded==0?0:expected.layout_hash);
        if (excluded==0) {
            assert(arena.generation()>before && s.offsets_.at("q")==old_offset);
            assert(actual(s,decls).layout_hash==expected.layout_hash);
        } else {
            assert(arena.generation()==before && s.offsets_.at("q")==old_offset+8192);
        }
    }
    // With a physical lease locked, a stale generation must fail closed;
    // valid leased Path3 may reuse exactly the already reserved placement.
    arena=SpmAllocator{};FusedModelBase::Impl leased;
    const auto expected=planned(leased,decls,ctx);
    production_allocation(leased,decls,ctx,expected.layout_hash);
    leased.pipeline_lease_epoch_=12;arena.pipeline_arena_locked_=true;
    const auto gen=arena.generation();const auto waterline=arena.t_end_;
    production_allocation(leased,decls,ctx,expected.layout_hash);
    assert(arena.generation()==gen && arena.t_end_==waterline);
    ++arena.generation_;
    bool rejected=false;
    try{production_allocation(leased,decls,ctx,expected.layout_hash);}
    catch(const std::runtime_error&){rejected=true;}
    assert(rejected && arena.generation()==gen+1 && arena.t_end_==waterline);
    // Ordinary warm Path3 must not reset or advance allocation identity either.
    arena=SpmAllocator{};FusedModelBase::Impl warm;
    production_allocation(warm,decls,ctx,expected.layout_hash);
    const auto warm_gen=arena.generation();const auto warm_end=arena.t_end_;
    const auto manifest_gen=warm.manifest_generation;
    production_allocation(warm,decls,ctx,expected.layout_hash);
    assert(arena.generation()==warm_gen && arena.t_end_==warm_end && warm.manifest_generation==manifest_gen);
    std::cout<<"implicit COMPLETE layout, non-COMPLETE/lease exclusions and warm Path3 preserved\n";
}
int main(){protected_scopes();return alternating_owners();}
'''.replace("__CALL__", call)
    return support


def _run_host(tmp_path, core_source):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    cpp, exe = tmp_path / "fmb_allocation.cpp", tmp_path / "fmb_allocation"
    cpp.write_text(_host_program(core_source))
    built = subprocess.run([compiler, "-std=c++17", "-O0", "-g", str(cpp), "-o", str(exe)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    return subprocess.run([str(exe)], capture_output=True, text=True)


def test_standalone_complete_layout_rebuild_preserves_offsets_and_other_scopes(tmp_path):
    result = _run_host(tmp_path, (ROOT / "src/core/fused_model_base.cpp").read_text())
    assert result.returncode == 0, result.stdout + result.stderr
