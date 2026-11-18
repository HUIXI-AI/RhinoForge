"""Execute the native NVFP4 producer admission against stale/cross-format routes."""
import importlib
from pathlib import Path
import json
import re
import runpy
import subprocess

from test_siglip_attention_dry_live_contract import _function

ROOT = Path(__file__).resolve().parents[1]


def test_nvfp4_geglu_rejects_incomplete_stale_or_cross_format_descriptor(tmp_path):
    source = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_pi05_denoise_step_model.h").read_text()
    policy = _function(header, "    FmbLinearAccumulationPolicy linear_accumulation_policy(")
    names = ("pi05_nvfp4_geglu_m50_arguments", "use_pi05_nvfp4_geglu_m50",
             "validate_pi05_nvfp4_geglu_m50_manifest")
    methods = [_function(source, "Pi05DenoiseStepModel::" + name + "(") for name in names]
    bodies = [method[method.index("{"):] for method in methods]
    selection = "\n".join(_function(source, marker) for marker in (
        "bool pi05_nvfp4_acc16_geglu_double_buffered(", "bool pi05_nvfp4_acc16_geglu_tile_scale("))
    constants = "\n".join(re.findall(
        r"constexpr int64_t PI05_DENOISE_(?:NVFP4_GEGLU_\w+|W4_GEGLU_M50_\w+_SITE|GEGLU_M50_\w+_SITE)\s*=.*?;",
        source, re.S))
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
enum class KernelId { PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128,
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128,
    PI05_NVFP4_GEGLU_DB,
    PI05_NVFP4_GEGLU_DB_TSCALE,
    PI05_NVFP4_GEGLU_DB_SSPM,
    PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM,
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M64N64K128_DB_SSPM_SPLIT,
    PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM_SPLIT };
// The fused AdaRMS norm payload's and the split siblings' REF probes (rpu_kernel_decls.h).
bool norm_payload_loaded=false;
bool pi05_adarms_geglu_fused(int64_t rows){return rows>8&&rows<=16&&norm_payload_loaded;}
bool split_loaded[2]={false,false};
bool pi05_nvfp4_geglu_split(bool norm){return split_loaded[norm];}
struct KernelCache {
    bool loaded=true, double_buffered=true, tile_scale=false;
    static KernelCache& instance() { static KernelCache cache;return cache; }
    bool has_loaded(KernelId id) const {
        if (id==KernelId::PI05_NVFP4_GEGLU_DB) return double_buffered;
        if (id==KernelId::PI05_NVFP4_GEGLU_DB_TSCALE) return tile_scale;
        assert(id==KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128);
        return loaded;
    }
};
enum class FmbPhysicalManifestState { UNSPECIFIED, COMPLETE };
enum class FmbGraphLifecycle { UNSPECIFIED, COMPOSITE_CHILD };
enum class FmbLinearAccumulationPolicy { ACC16, ACC32, MIXED_BY_SITE };
enum class FmbRouteFamily { LINEAR, GRAPH_SCHEDULE };
// Shared action-horizon helpers, defined beside the routes under test.
bool pi05_action_rows_admitted(int64_t rows){return rows==50||rows==10;}
int64_t pi05_kv_insert_rows(int64_t rows){return rows==10?16:64;}
__SELECTION__
__CONSTANTS__
struct Route { int64_t site_id; FmbRouteFamily family; int64_t selector, flags, invocation;
    std::vector<int64_t> arguments; };
struct FmbPhysicalExecutionManifest {
    FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
    int64_t logical_length=50,physical_length=50,execution_padding_rows=0,
            kv_logical_length=850,kv_insert_physical_rows=64;
    FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
    FmbLinearAccumulationPolicy linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
    std::vector<Route> routes;
};
struct Model {
    bool linear_acc32_=false;
__POLICY__
    int64_t planned_prefix_len_=800, planned_num_steps_=10, chunk_size_=50;
    int cores=8,attention_cores=8,mlp_cores=8;
    int num_cores() const {return cores;} int attn_tp() const {return attention_cores;}
    int mlp_tp() const {return mlp_cores;}
    bool pi05_nvfp4_geglu_m50_opt_in_=true, owners=true, continuation=true;
    struct { bool value=true; bool enabled() const { return value; } } nvfp4_;
    struct { bool present=false; bool defined() const { return present; } } nvfp4_spm_scale_;
    bool pi05_nvfp4_weight_profile_admitted() const { return owners; }
    bool use_pi05_xor3_gated_mlp(int64_t rows) const { return continuation && rows==50; }
    std::vector<int64_t> pi05_w4_fastpath_arguments() const { return {2,16,8,32}; }
    void validate_pi05_w4_fastpath_manifest(const FmbPhysicalExecutionManifest&) const { TORCH_CHECK(owners); }
    void validate_pi05_xor3_gated_mlp_manifest(const FmbPhysicalExecutionManifest&) const { TORCH_CHECK(continuation); }
    std::vector<int64_t> pi05_nvfp4_geglu_m50_arguments() const __ARGS__
    bool use_pi05_nvfp4_geglu_m50(int64_t rows) const __USE__
    void validate_pi05_nvfp4_geglu_m50_manifest(const FmbPhysicalExecutionManifest& manifest) const __VALIDATE__
};
int main() {
    Model model;
    FmbPhysicalExecutionManifest valid;
    valid.routes={{PI05_DENOISE_NVFP4_GEGLU_POLICY_SITE,FmbRouteFamily::GRAPH_SCHEDULE,1,0,0,
                   model.pi05_nvfp4_geglu_m50_arguments()},
                  {PI05_DENOISE_NVFP4_GEGLU_LINEAR_SITE,FmbRouteFamily::LINEAR,
                   PI05_DENOISE_NVFP4_GEGLU_ACC16_M320N64K128,0,0,model.pi05_nvfp4_geglu_m50_arguments()}};
    model.validate_pi05_nvfp4_geglu_m50_manifest(valid);
    // The double-buffered M64 payload is planned whenever the REF carries it.
    assert(valid.routes[1].arguments[2]==static_cast<int64_t>(
        KernelId::PI05_NVFP4_GEGLU_DB));
    assert(valid.routes[1].arguments[8]==64);
    auto reject=[&](const FmbPhysicalExecutionManifest& m) {
        try { model.validate_pi05_nvfp4_geglu_m50_manifest(m);assert(false); }
        catch (const std::runtime_error&) {}
    };
    for (int mutation=0;mutation<24;++mutation) {
        auto bad=valid;
        switch(mutation) {
          case 0: bad.state=FmbPhysicalManifestState::UNSPECIFIED;break;
          case 1: bad.physical_length=64;break;
          case 2: bad.logical_length=49;break;
          case 3: bad.execution_padding_rows=14;break;
          case 4: bad.kv_logical_length=849;break;
          case 5: bad.kv_insert_physical_rows=50;break;
          case 6: bad.graph_lifecycle=FmbGraphLifecycle::UNSPECIFIED;break;
          case 7: bad.linear_accumulation=FmbLinearAccumulationPolicy::ACC32;break;
          case 8: bad.routes.erase(bad.routes.begin());break;
          case 9: bad.routes.pop_back();break;
          case 10: bad.routes.push_back(bad.routes[0]);break;
          case 11: bad.routes.push_back(bad.routes[1]);break;
          case 12: bad.routes[1].site_id=123;break;
          case 13: bad.routes[1].selector=1;break;
          case 14: bad.routes[1].flags=1;break;
          case 15: bad.routes[1].invocation=1;break;
          case 16: bad.routes[0].arguments[0]=99;break;
          case 17: bad.routes[1].arguments[28]=128;break; // INT4 K128 cannot replace NVFP4 K16
          case 18: bad.routes.push_back({PI05_DENOISE_W4_GEGLU_M50_LINEAR_SITE,FmbRouteFamily::LINEAR,1,0,0,{}});break;
          case 19: bad.routes.push_back({PI05_DENOISE_GEGLU_M50_LINEAR_SITE,FmbRouteFamily::LINEAR,1,0,0,{}});break;
          case 20: bad.routes[1].selector=PI05_DENOISE_NVFP4_GEGLU_M320N64K128;break;
          case 21: bad.routes[1].arguments[0]=1;break; // Retired ACC32 argument ABI.
          case 22: bad.routes[1].arguments[2]=static_cast<int64_t>(KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128);break;
          case 23: bad.routes[1].arguments[34]=32;break; // ACC32 producer arithmetic.

        }
        reject(bad);
    }
    model.owners=false;reject(valid);model.owners=true;
    model.linear_acc32_=true;
    assert(model.linear_accumulation_policy()==FmbLinearAccumulationPolicy::ACC32);
    reject(valid);
    auto acc32=valid;acc32.linear_accumulation=FmbLinearAccumulationPolicy::ACC32;
    reject(acc32); // ACC32 remains on decomposed projections.
    model.linear_acc32_=false;
    // Without the double-buffered payload the M320 one is planned, and a
    // descriptor planned for either payload is stale for the other.
    auto& cache=KernelCache::instance();
    cache.double_buffered=false;reject(valid);
    auto m320=valid;
    for (auto& route : m320.routes) route.arguments=model.pi05_nvfp4_geglu_m50_arguments();
    assert(m320.routes[1].arguments[2]==static_cast<int64_t>(
        KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128));
    assert(m320.routes[1].arguments[8]==320);
    model.validate_pi05_nvfp4_geglu_m50_manifest(m320);
    cache.loaded=false;reject(m320);  // neither payload: no fusion
    cache.double_buffered=true;reject(m320);
    model.validate_pi05_nvfp4_geglu_m50_manifest(valid);  // the M64 payload alone suffices
    cache.loaded=true;
    // The tile-scale sibling is planned when the REF carries it beside the
    // double-buffered payload, also with tile M 64; either descriptor is stale
    // for the other, and without the double-buffered payload it is never planned.
    cache.tile_scale=true;reject(valid);
    auto tiled=valid;
    for (auto& route : tiled.routes) route.arguments=model.pi05_nvfp4_geglu_m50_arguments();
    assert(tiled.routes[1].arguments[2]==static_cast<int64_t>(
        KernelId::PI05_NVFP4_GEGLU_DB_TSCALE));
    assert(tiled.routes[1].arguments[8]==64);
    model.validate_pi05_nvfp4_geglu_m50_manifest(tiled);
    // An installed SPM-scale slab source adds its sibling, which the layers
    // under N take; a descriptor from either side is stale for the other.
    model.nvfp4_spm_scale_.present=true;reject(tiled);
    auto spm_scaled=tiled;
    for (auto& route : spm_scaled.routes) route.arguments=model.pi05_nvfp4_geglu_m50_arguments();
    assert(spm_scaled.routes[1].arguments[2]==tiled.routes[1].arguments[2]);
    assert(spm_scaled.routes[1].arguments.size()==tiled.routes[1].arguments.size()+1);
    model.validate_pi05_nvfp4_geglu_m50_manifest(spm_scaled);
    // The fused norm payload is named only at 9..16 rows, where those layers run it.
    norm_payload_loaded=true;model.validate_pi05_nvfp4_geglu_m50_manifest(spm_scaled);
    model.chunk_size_=10;
    const auto m10=model.pi05_nvfp4_geglu_m50_arguments();
    // Right after the SPM-scale sibling, before the fast-path profile.
    const size_t fused_at=m10.size()-model.pi05_w4_fastpath_arguments().size()-1;
    assert(m10[fused_at]==static_cast<int64_t>(
        KernelId::PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM));
    assert(m10[fused_at-1]==static_cast<int64_t>(
        KernelId::PI05_NVFP4_GEGLU_DB_SSPM));
    assert(m10.size()==spm_scaled.routes[1].arguments.size()+1);
    // The split siblings take those two places, each when the REF has it; a
    // descriptor from either side is stale for the other.
    for (bool norm : {false,true}) {
        split_loaded[norm]=true;
        const auto split=model.pi05_nvfp4_geglu_m50_arguments();
        assert(split.size()==m10.size());
        for (size_t i=0;i<m10.size();++i) {
            if (i==fused_at-1 && !norm) assert(split[i]==static_cast<int64_t>(
                KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M64N64K128_DB_SSPM_SPLIT));
            else if (i==fused_at && norm) assert(split[i]==static_cast<int64_t>(
                KernelId::PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM_SPLIT));
            else assert(split[i]==m10[i]);
        }
        split_loaded[norm]=false;
    }
    split_loaded[0]=true;
    model.chunk_size_=50;reject(spm_scaled);
    auto split50=spm_scaled;
    for (auto& route : split50.routes) route.arguments=model.pi05_nvfp4_geglu_m50_arguments();
    model.validate_pi05_nvfp4_geglu_m50_manifest(split50);
    split_loaded[0]=false;reject(split50);
    model.chunk_size_=10;
    norm_payload_loaded=false;
    assert(model.pi05_nvfp4_geglu_m50_arguments().size()==m10.size()-1);
    model.chunk_size_=50;
    model.nvfp4_spm_scale_.present=false;reject(spm_scaled);
    model.validate_pi05_nvfp4_geglu_m50_manifest(tiled);
    cache.double_buffered=false;reject(tiled);model.validate_pi05_nvfp4_geglu_m50_manifest(m320);
    cache.double_buffered=true;cache.tile_scale=false;reject(tiled);
    for (int* domain : {&model.cores,&model.attention_cores,&model.mlp_cores}) {
        for (int reduced : {4,6}) {*domain=reduced;reject(valid);}
        *domain=8;
    }
    for (int steps : {1,4,9,11}) {model.planned_num_steps_=steps;reject(valid);}
    model.planned_num_steps_=10;
    model.continuation=false;reject(valid);model.continuation=true;
    model.nvfp4_.value=false;reject(valid);model.nvfp4_.value=true;
    model.pi05_nvfp4_geglu_m50_opt_in_=false;reject(valid);
    auto empty=valid;empty.routes.clear();model.validate_pi05_nvfp4_geglu_m50_manifest(empty);
    model.pi05_nvfp4_geglu_m50_opt_in_=true;
    model.planned_prefix_len_=544;
    // The Action shape stays M50/PAD64, but a P800 logical-KV descriptor
    // cannot be reused for a true two-camera prefix.
    reject(valid);
    valid.kv_logical_length=594;
    model.validate_pi05_nvfp4_geglu_m50_manifest(valid);
    valid.kv_logical_length=608;reject(valid); // physical insertion end is not attention length
    for (int prefix : {672,928}) {
        model.planned_prefix_len_=prefix;
        reject(valid); // A previous profile's KV length cannot be reused.
        valid.kv_logical_length=prefix+50;
        model.validate_pi05_nvfp4_geglu_m50_manifest(valid);
    }
    for (int prefix : {0,543,576,608,640,799,801,832,864,896}) {
        model.planned_prefix_len_=prefix;
        valid.kv_logical_length=prefix+50;
        reject(valid); // The new producer is limited to actual two/three-image T32/T160.
    }
}
'''
    cpp = cpp.replace("__POLICY__", policy)
    for key, value in zip(("ARGS", "USE", "VALIDATE"), bodies):
        cpp = cpp.replace("__" + key + "__", value)
    cpp = cpp.replace("__CONSTANTS__", constants).replace("__SELECTION__", selection)
    path, binary = tmp_path / "nvfp4-policy.cpp", tmp_path / "nvfp4-policy"
    path.write_text(cpp)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr


def test_nvfp4_geglu_launchers_keep_distinct_ids_and_typed_ddr_abi(tmp_path):
    """Execute both production launchers through their real shared ABI writer."""
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    functions = "\n".join(_function(source, marker) for marker in (
        "static void launch_pi05_denoise_gate_up_geglu_nvfp4_m50(",
        "void rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_m50_kernel(",
        "void rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel("))
    model = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    selection = "\n".join(_function(model, marker) for marker in (
        "bool pi05_nvfp4_acc16_geglu_double_buffered(", "bool pi05_nvfp4_acc16_geglu_tile_scale("))
    cpp = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
constexpr int NUM_CORES=8;
// Shared horizon predicate; defined in rpu_pi05_denoise_step_model.cpp and
// declared in rpu_kernel_decls.h, which this harness replaces.
bool pi05_action_rows_admitted(int64_t rows){return rows==50||rows==10;}
// The fused norm payload's REF probe and operands, also from rpu_kernel_decls.h.
bool norm_payload_loaded=true;
bool pi05_adarms_geglu_fused(int64_t rows){return rows>8&&rows<=16&&norm_payload_loaded;}
struct Pi05AdaRmsNormOperands {uint32_t input=0,scale=0,shift=0;double eps=0;};
// Two bytes; a recognizable stand-in for the FP16 bits.
namespace c10 {struct Half {uint16_t x;explicit Half(float v):x(uint16_t(std::lround(v*1e6))){}};}
namespace at {
constexpr int kPrivateUse1=1,kByte=2;
using IntArrayRef=std::vector<int64_t>;
struct Tensor {
    uint64_t address;IntArrayRef shape;int dtype=kByte;bool present=true;
    bool defined()const{return present;}
    struct Device {int type()const{return kPrivateUse1;}
        bool operator==(const Device&)const{return true;}};
    Device device()const{return {};}
    int scalar_type()const{return dtype;}
    bool is_contiguous()const{return true;}
    int storage_offset()const{return 0;}
    const IntArrayRef& sizes()const{return shape;}
    int64_t nbytes()const{return shape[0]*shape[1];}
    void* data_ptr()const{return reinterpret_cast<void*>(address);}
};
}
struct SpmAllocator {
    static constexpr uint64_t SPM_USABLE=8387584;
    bool is_initialized()const{return true;}
    uint32_t addr(int,int)const{return 0x100000;}
} allocator;
#define SPM_ALLOC allocator
int flushed=0;
void rpu_ddr_flush(void*){++flushed;}
uint64_t RpuGetDevAddr(void* ptr){return reinterpret_cast<uint64_t>(ptr);}
enum class KernelId {PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128,
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128,
    PI05_NVFP4_GEGLU_DB,
    PI05_NVFP4_GEGLU_DB_TSCALE,
    PI05_NVFP4_GEGLU_DB_SSPM,
    PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM,
    PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M64N64K128_DB_SSPM_SPLIT,
    PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM_SPLIT};
// The split siblings' REF probe, from rpu_kernel_decls.h.
bool split_loaded[2]={false,false};
bool pi05_nvfp4_geglu_split(bool norm){return split_loaded[norm];}
bool is_split(KernelId id){
    return id==KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M64N64K128_DB_SSPM_SPLIT ||
        id==KernelId::PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM_SPLIT;
}
struct KernelCache {
    bool double_buffered=true,tile_scale=false;
    static KernelCache& instance(){static KernelCache cache;return cache;}
    bool has_loaded(KernelId id)const{
        if(id==KernelId::PI05_NVFP4_GEGLU_DB_TSCALE)return tile_scale;
        assert(id==KernelId::PI05_NVFP4_GEGLU_DB);
        return double_buffered;
    }
};
__SELECTION__
enum class GraphDdrRegisterRole {PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct Abi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
    GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {Abi abi;at::Tensor owner;};
std::vector<GraphDdrRegisterOperandSpec> staged;
KernelId staged_id,looked_up_id;
struct Kernel {
    std::map<int,uint16_t> regs;
    std::vector<int> full_word_writes;
    void reset_regs(){regs.clear();full_word_writes.clear();}
    void set_regs(int reg,uint16_t value){regs[reg]=value;}
    void set_regs(int reg,uint32_t value){
        assert(reg%2==0);full_word_writes.push_back(reg);
        regs[reg]=value&0xffff;regs[reg+1]=value>>16;
    }
} kernel;
struct Writer {
    void write(Kernel& k) {
        for(const auto& operand:staged) {
            const auto encoded=operand.owner.address>>8;
            k.set_regs(operand.abi.lo,static_cast<uint16_t>(encoded&0xffff));
            k.set_regs(operand.abi.hi,static_cast<uint16_t>(encoded>>16));
        }
    }
};
struct RpuKernelGraph {
    static RpuKernelGraph& active(){static RpuKernelGraph graph;return graph;}
    Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> operands) {
        staged_id=id;staged=std::move(operands);return {};
    }
};
Kernel* get_kernel(KernelId id){looked_up_id=id;return &kernel;}
#define GET_KERNEL(id) get_kernel(id)
struct Queue {
    bool broadcast=false;int launches=0;
    void set_broadcast_mode(bool value){broadcast=value;}
    void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
        assert(broadcast && staged_id==looked_up_id);
        // The split siblings run sixteen blocks, every other payload eight.
        assert(grid==std::vector<uint16_t>({uint16_t(is_split(staged_id)?16:8),1,1}));
        assert(kernel.regs.at(64)==grid[0]);
        assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
        ++launches;
    }
} queue;
Queue* get_queue(int cores){assert(cores==8);queue.broadcast=false;return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
// The declaration's defaults, as in rpu_kernel_decls.h.
void rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(
    uint32_t, const at::Tensor&, const at::Tensor&, uint32_t, const at::Tensor&, const at::Tensor&,
    uint32_t, uint32_t, uint16_t, int64_t, const at::Tensor&, const at::Tensor&,
    uint32_t = 0, uint32_t = 0, const Pi05AdaRmsNormOperands* = nullptr);
__FUNCTIONS__
int main(){
    const at::Tensor gate{0x2000000,{4096,512}},up{0x2400000,{4096,512}},
        gate_scale{0x2800000,{64,4096}},up_scale{0x2900000,{64,4096}},
        gate_tile_scale{0x2a00000,{64,4096}},up_tile_scale{0x2b00000,{64,4096}};
    const auto u32=[](int reg){return uint32_t(kernel.regs.at(reg))|(uint32_t(kernel.regs.at(reg+1))<<16);};
    // Both action horizons go through the same payload: M is a register.
    for(bool tile_scale:{false,true})
    for(bool double_buffered:{true,false})
    for(int64_t rows:{int64_t{50},int64_t{10}})
    for(bool acc16:{false,true,false,true}) {
        KernelCache::instance().double_buffered=double_buffered;
        KernelCache::instance().tile_scale=tile_scale;
        const auto launch=[&](uint32_t input,const at::Tensor& g,const at::Tensor& u,uint32_t out,
                              const at::Tensor& gs,const at::Tensor& us,uint32_t gts,uint32_t uts,
                              uint16_t layer,int64_t m){
            if(acc16) rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(
                input,g,u,out,gs,us,gts,uts,layer,m,gate_tile_scale,up_tile_scale,0,0);
            else rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_m50_kernel(
                input,g,u,out,gs,us,gts,uts,layer,m);
        };
        staged.clear();flushed=0;
        launch(0x100000,gate,up,0x120000,gate_scale,up_scale,0x130000,0x130100,17,rows);
        // ACC16 takes the double-buffered payload when the REF carries it and
        // its tile-scale sibling, reading the companions, when it carries both.
        const bool tiled=acc16 && double_buffered && tile_scale;
        assert(staged_id==(!acc16 ? KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128
                           : tiled ? KernelId::PI05_NVFP4_GEGLU_DB_TSCALE
                           : double_buffered ? KernelId::PI05_NVFP4_GEGLU_DB
                                             : KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M320N64K128));
        assert(staged.size()==4 && flushed==4);
        const int registers[]={2,20,32,34};
        const at::Tensor* owners[]={&gate,tiled ? &gate_tile_scale : &gate_scale,
                                    &up,tiled ? &up_tile_scale : &up_scale};
        for(size_t i=0;i<staged.size();++i){
            const auto& operand=staged[i];
            assert(operand.abi.role==(i%2 ? GraphDdrRegisterRole::Fp8BlockScale
                                          : GraphDdrRegisterRole::PackedFp4ModelWeight));
            assert(operand.abi.access==GraphDdrRegisterAccess::Read);
            assert(operand.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
            assert(operand.abi.lo==registers[i] && operand.abi.hi==registers[i]+1);
            assert(operand.owner.address==owners[i]->address);
            assert(u32(registers[i])==owners[i]->address>>8);
        }
        assert(u32(0)==0x100000 && u32(4)==0x120000 && u32(6)==0);
        assert(kernel.regs.at(8)==rows && u32(10)==512 && kernel.regs.at(12)==1024);
        assert(u32(24)==0x130000 && u32(36)==0x130100 && kernel.regs.at(27)==17);
        if(acc16){
            // Every launch initializes both complete core counter words. A
            // halfword-only reset leaves stale high bits from earlier kernels.
            assert(kernel.full_word_writes==std::vector<int>({128,130}));
            assert(u32(128)==0 && u32(130)==0);
        }else{
            assert(kernel.full_word_writes.empty());
            for(int reg=128;reg<=255;++reg)assert(kernel.regs.count(reg)==0);
        }
        const auto launches=queue.launches;
        for(int mutation=0;mutation<5;++mutation){
            auto bad=gate;
            if(mutation==0) bad.dtype=0;
            if(mutation==1) bad.shape={4096,1024};
            bool rejected=false;
            // mutation 4: an unadmitted action horizon, rejected like a bad owner.
            try{launch(0x100000,bad,up,mutation==2?0x100000:0x120000,
                gate_scale,up_scale,0x130000,0x130100,mutation==3?18:0,
                mutation==4?32:rows);}
            catch(const std::runtime_error&){rejected=true;}
            assert(rejected && queue.launches==launches);
        }
        if(tiled){
            // The tile-scale payload never falls back to the striped scales.
            auto missing=up_tile_scale;missing.present=false;
            bool rejected=false;
            try{rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(
                0x100000,gate,up,0x120000,gate_scale,up_scale,0x130000,0x130100,17,rows,
                gate_tile_scale,missing,0,0);}
            catch(const std::runtime_error&){rejected=true;}
            assert(rejected && queue.launches==launches);
        }
    }
    assert(queue.launches==32);
    // SPM scales: only the weights stay DDR operands, 20/21 and 34/35 hold the
    // two slabs, and the slabs are checked like the other SPM operands.
    KernelCache::instance().double_buffered=true;
    KernelCache::instance().tile_scale=true;
    for(int64_t rows:{int64_t{50},int64_t{10}}){
        const auto launch=[&](uint32_t gate_slab,uint32_t up_slab){
            rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(
                0x100000,gate,up,0x120000,gate_scale,up_scale,0x130000,0x130100,17,rows,
                gate_tile_scale,up_tile_scale,gate_slab,up_slab);
        };
        launch(0x140000,0x148000);
        assert(staged_id==KernelId::PI05_NVFP4_GEGLU_DB_SSPM);
        assert(staged.size()==2 && staged[0].abi.lo==2 && staged[0].owner.address==gate.address &&
               staged[1].abi.lo==32 && staged[1].owner.address==up.address);
        assert(u32(2)==gate.address>>8 && u32(32)==up.address>>8);
        assert(u32(20)==0x140000 && u32(34)==0x148000);
        assert(kernel.full_word_writes==std::vector<int>({128,130}));
        const auto launches=queue.launches;
        const std::pair<uint32_t,uint32_t> bad[]={
            {0x140000,0},{0,0x148000},{0x100000,0x148000},{0x140000,0x144000},
            {0x140001,0x148000},{0x900000,0x148000}};
        for(const auto& [gate_slab,up_slab]:bad){
            bool rejected=false;
            try{launch(gate_slab,up_slab);}catch(const std::runtime_error&){rejected=true;}
            assert(rejected && queue.launches==launches);
        }
        KernelCache::instance().tile_scale=false;
        bool rejected=false;
        try{launch(0x140000,0x148000);}catch(const std::runtime_error&){rejected=true;}
        assert(rejected && queue.launches==launches);
        KernelCache::instance().tile_scale=true;
    }
    assert(queue.launches==34);
    // The AdaRMS norm folded in (M <= 16): the SPM-scale image plus the
    // row-pair norm's register image at 40..54, writing the GeGLU's input
    // rows; the norm's operands are checked with the other slabs.
    const Pi05AdaRmsNormOperands norm{0x150000,0x158000,0x158800,1e-6};
    const auto fused=[&](int64_t rows,const Pi05AdaRmsNormOperands* n,uint32_t slab=0x140000){
        rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(
            0x100000,gate,up,0x120000,gate_scale,up_scale,0x130000,0x130100,17,rows,
            gate_tile_scale,up_tile_scale,slab,slab?0x148000:0,n);
    };
    fused(10,nullptr);
    const auto sspm=kernel.regs;
    fused(10,&norm);
    assert(staged_id==KernelId::PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM);
    assert(staged.size()==2 && kernel.full_word_writes==std::vector<int>({128,130}));
    auto image=kernel.regs;
    assert(image.at(40)==1024 && image.at(41)==c10::Half(1.0f/1024).x &&
           image.at(42)==2048 && image.at(43)==0 && image.at(50)==c10::Half(1e-6f).x &&
           image.at(54)==10 && !image.count(51));
    assert(u32(44)==norm.input && u32(46)==0x100000 && u32(48)==norm.scale && u32(52)==norm.shift);
    for(int reg=38;reg<64;++reg){assert(!sspm.count(reg));image.erase(reg);}
    assert(image==sspm);
    const auto launches=queue.launches;
    const auto rejects=[&](auto f){
        bool rejected=false;
        try{f();}catch(const std::runtime_error&){rejected=true;}
        assert(rejected && queue.launches==launches);
    };
    // M50, no SPM scales, a REF without the payload.
    rejects([&]{fused(50,&norm);});
    rejects([&]{fused(10,&norm,0);});
    norm_payload_loaded=false;
    rejects([&]{fused(10,&norm);});
    norm_payload_loaded=true;
    // The norm's rows on the GeGLU input or output, its scale inside its rows,
    // a misaligned shift, eps outside (0,1].
    for(const auto& bad:{Pi05AdaRmsNormOperands{0x100000,0x158000,0x158800,1e-6},
                         Pi05AdaRmsNormOperands{0x120000,0x158000,0x158800,1e-6},
                         Pi05AdaRmsNormOperands{0x150000,0x150800,0x158800,1e-6},
                         Pi05AdaRmsNormOperands{0x150000,0x158000,0x158810,1e-6},
                         Pi05AdaRmsNormOperands{0x150000,0x158000,0x158800,0.0},
                         Pi05AdaRmsNormOperands{0x150000,0x158000,0x158800,NAN}})
        rejects([&]{fused(10,&bad);});
    // The payload and the operands come together.
    rejects([&]{launch_pi05_denoise_gate_up_geglu_nvfp4_m50(
        0x100000,gate,up,0x120000,gate_tile_scale,up_tile_scale,0x130000,0x130100,17,
        KernelId::PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM,10,
        0x140000,0x148000,nullptr);});
    rejects([&]{launch_pi05_denoise_gate_up_geglu_nvfp4_m50(
        0x100000,gate,up,0x120000,gate_tile_scale,up_tile_scale,0x130000,0x130100,17,
        KernelId::PI05_NVFP4_GEGLU_DB_SSPM,10,
        0x140000,0x148000,&norm);});
    assert(queue.launches==36);
    // With a split sibling loaded, its SPM-scale payload launches it instead:
    // the same image but p64 = 16, grid 16; the other payloads are unchanged.
    for(bool norm_split:{false,true}){
        split_loaded[norm_split]=true;
        fused(10,nullptr);
        assert(staged_id==(norm_split ? KernelId::PI05_NVFP4_GEGLU_DB_SSPM
                           : KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_ACC16_M64N64K128_DB_SSPM_SPLIT));
        auto plain=kernel.regs;
        if(!norm_split){assert(plain.at(64)==16);plain[64]=8;}
        assert(plain==sspm);
        fused(10,&norm);
        assert(staged_id==(norm_split ? KernelId::PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM_SPLIT
                           : KernelId::PI05_DENOISE_ADARMS_GATE_UP_GEGLU_NVFP4_ACC16_M16N64K128_DB_SSPM));
        auto with_norm=kernel.regs;
        if(norm_split){assert(with_norm.at(64)==16);with_norm[64]=8;}
        for(int reg=38;reg<64;++reg)with_norm.erase(reg);
        assert(with_norm==sspm && kernel.full_word_writes==std::vector<int>({128,130}));
        // The tile-scale payload has no split sibling.
        rpu_launch_pi05_denoise_gate_up_geglu_nvfp4_acc16_m50_kernel(
            0x100000,gate,up,0x120000,gate_scale,up_scale,0x130000,0x130100,17,50,
            gate_tile_scale,up_tile_scale,0,0);
        assert(staged_id==KernelId::PI05_NVFP4_GEGLU_DB_TSCALE);
        split_loaded[norm_split]=false;
    }
    assert(queue.launches==42);
}
'''.replace("__FUNCTIONS__", functions).replace("__SELECTION__", selection)
    path, binary = tmp_path / "nvfp4-launchers.cpp", tmp_path / "nvfp4-launchers"
    path.write_text(cpp)
    built = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", str(path), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


def test_nvfp4_cold_policy_preserves_uniform_arithmetic_with_optimized_flags(tmp_path):
    source = (ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_pi05_denoise_step_model.h").read_text()
    constructor = _function(source, "Pi05DenoiseStepModel::Pi05DenoiseStepModel(")
    constructor = constructor.replace("Pi05DenoiseStepModel::Pi05DenoiseStepModel", "Model")
    fields = "\n".join(re.findall(r"    const (?:bool|int64_t) \w+_;", header))
    policy = _function(header, "    FmbLinearAccumulationPolicy linear_accumulation_policy(")
    profile = vars(importlib.import_module("examples.runners.pi05.optimized"))["profile_environment"](
        "w4a16", prefill_w8a8=False, vision_owner_ln=True, cameras=3)
    environment = ",\n".join("{" + json.dumps(key) + ", " + str(value == "1").lower() + "}"
                             for key, value in profile.items())
    code = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
enum class FmbLinearAccumulationPolicy { ACC16, ACC32, MIXED_BY_SITE };
std::map<std::string, bool> environment = {__ENVIRONMENT__};
bool pi05_kernel_opt_in(const char* name) { return environment[name]; }
int64_t pi05_gateup_bundle_layers() { return 0; }
struct Model {
__FIELDS__
__CONSTRUCTOR__
__POLICY__
};
int main() {
    assert(!environment["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"]);
    // A legacy inherited opt-in cannot re-enable the retired mixed stack.
    environment["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"] = true;
    assert(environment["RPU_PI05_DENOISE_W4_FASTPATH"]);
    for (bool new_flag : {false,true}) for (bool acc32 : {false, true}) {
        environment["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] = new_flag;
        Model model(acc32);
        assert(model.linear_accumulation_policy() == (acc32
            ? FmbLinearAccumulationPolicy::ACC32 : FmbLinearAccumulationPolicy::ACC16));
        // Only the new ACC16 flag enables GeGLU; inherited legacy flags cannot
        // reopen the mixed Gate/Up32 + KV16 stack or change ACC32 projections.
        assert(model.pi05_nvfp4_geglu_m50_opt_in_ == (new_flag && !acc32));
        assert(model.pi05_w4_fastpath_opt_in_ == !acc32);
        assert(model.pi05_kv1_pair_owner_opt_in_ == !acc32);
        assert(model.pi05_xor3_gated_mlp_opt_in_ == !acc32);
        assert(!model.pi05_geglu_m50_opt_in_ && !model.pi05_w4_geglu_m50_opt_in_);
    }
}
'''
    for key, value in (("ENVIRONMENT", environment), ("FIELDS", fields),
                       ("CONSTRUCTOR", constructor), ("POLICY", policy)):
        code = code.replace("__" + key + "__", value)
    path, binary = tmp_path / "nvfp4-cold.cpp", tmp_path / "nvfp4-cold"
    path.write_text(code)
    compiled = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", str(path), "-o", str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert run.returncode == 0, run.stderr
