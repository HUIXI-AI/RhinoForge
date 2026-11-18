"""Host execution probe of actual G05 local-slice register emission and bounds."""
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[1]
definition = runpy.run_path(str(ROOT / "tests/test_qwen35_gdn_mask_residency.py"))["definition"]


def test_g05_slice_local_registers_and_fail_closed_boundaries(tmp_path):
    slice_src = (ROOT / "src/ops/rpu_slice.cpp").read_text()
    code = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#define TORCH_CHECK(c, ...) do {if(!(c)) throw std::runtime_error("rejected");} while(0)
namespace c10 {struct Half {uint16_t value;};}
template<class A,class B> int64_t CeilDiv(A a,B b){return (a+b-1)/b;}
constexpr int V16_NUM=16, MAX_FP16_VLM_V16_PER_THD=64;
struct SpmAllocator {
    static constexpr int NUM_CORES=8;
    static constexpr size_t SPM_USABLE=0x7ffc00;
    bool initialized=true;
    std::array<uint32_t,8> origins;
    SpmAllocator(){for(int c=0;c<8;++c)origins[c]=0x40000400+c*0x1000000;}
    bool is_initialized()const{return initialized;}
    uint32_t addr(int c,uint32_t off)const{return origins[c]+off;}
} SPM_ALLOC;
enum class GraphKernelNoDdrProof {SliceSpm};
struct Kernel_t {
    std::array<uint16_t,96> r{};std::string name;
    void set_regs(int i,uint16_t v){r.at(i)=v;}
    uint32_t pair(int i)const{return r[i]|(uint32_t(r[i+1])<<16);}
};
std::unordered_map<uint32_t,uint16_t> memory;
int staged=0,lookups=0,launches=0,last_core_count=0,active_cores=8;
struct RpuKernelGraph {
    Kernel_t kernel;
    static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
    void stage_kernel_no_ddr(const std::string&,GraphKernelNoDdrProof){++staged;}
    Kernel_t* get_kernel_reset(const std::string& n){++lookups;kernel={};kernel.name=n;return &kernel;}
};
struct Queue {
    bool broadcast=false;
    void set_broadcast_mode(bool x){broadcast=x;}
    void enqueu_kernel(Kernel_t& k,std::array<uint16_t,3> grid,const std::vector<uint8_t>& cores){
        assert(broadcast);++launches;last_core_count=cores.size();
        assert(grid[0]==k.r[64]&&grid[1]==k.r[65]&&grid[2]==k.r[66]);
        for(int core:cores){
            auto physical=[&](uint32_t a){return a<0x40000000?a+0x40000000+core*0x1000000:a;};
            const uint32_t src=physical(k.pair(0)),dst=physical(k.pair(2));
            auto copy=[&](size_t s,size_t d){memory[dst+d]=memory.at(src+s);};
            if(k.name=="slice_last_dim_axis"){
                assert(k.r[4]==0&&k.r[5]==0);
                for(uint32_t o=0;o<k.pair(24);++o)
                    for(uint32_t m=0;m<k.pair(18);++m)
                        for(uint32_t c=0;c<k.r[7];++c)
                            copy(o*k.pair(20)+m*k.pair(12)+c*2,
                                 o*k.pair(22)+m*k.pair(14)+c*2);
            }else if(k.name=="slice_single_axis"){
                assert(k.r[4]==0&&k.r[5]==0);
                for(uint32_t o=0;o<grid[2];++o)
                    for(uint32_t m=0;m<k.r[7];++m)
                        for(uint32_t c=0;c<k.pair(16);++c)
                            copy(o*k.pair(12)+m*k.pair(8)+c*2,
                                 o*k.pair(14)+m*k.pair(10)+c*2);
            }else{assert(false);
            }
        }
    }
} queue;
#define GET_QUEUE(n) (&queue)
void rpu_launch_slice_spm_kernel(uint32_t,uint32_t,const std::vector<int64_t>&,
    const std::vector<int64_t>&,const std::vector<int64_t>&,int,bool=false);
'''
    for marker in (
        "static inline int64_t slice_shp_prod(",
        "static size_t slice_checked_multiply(",
        "static size_t slice_checked_add(",
        "static void validate_slice_spm_interval(",
        "static uint32_t slice_per_core_local_address(",
        "void rpu_launch_slice_spm_kernel(",
    ):
        code += "\n" + definition(slice_src, marker)
    code += r'''

size_t prod(const std::vector<int64_t>& shape){size_t n=1;for(auto x:shape)n*=x;return n;}
void seed(uint32_t s,size_t n){memory.clear();for(int c=0;c<active_cores;++c)for(size_t i=0;i<n;++i)memory[s+c*0x1000000+i*2]=uint16_t(c*733+i*19);}
std::vector<uint16_t> snapshot(uint32_t d,size_t n){std::vector<uint16_t> r;for(int c=0;c<active_cores;++c)for(size_t i=0;i<n;++i)r.push_back(memory.at(d+c*0x1000000+i*2));return r;}
int main(){
    const uint32_t s=SPM_ALLOC.addr(0,0xfe00),d=SPM_ALLOC.addr(0,0x180000);
    const std::vector<std::vector<int64_t>> inputs={{64,16},{128,16},{192,16},{256,16},{2,64},{4,64},{6,64},{8,64},{12,64}};
    const std::vector<std::vector<int64_t>> outputs={{64,2},{128,2},{192,2},{256,2},{2,1},{4,1},{6,1},{8,1},{12,1}};
    const std::vector<std::vector<int64_t>> begins={{0,0},{0,0},{0,0},{0,0},{0,63},{0,63},{0,63},{0,63},{0,63}};
    for(int nc:{4,8})for(size_t i=0;i<inputs.size();++i){
        active_cores=nc;
        seed(s,prod(inputs[i]));launches=0;
        for(int c=0;c<nc;++c)
            rpu_launch_slice_spm_kernel(s+c*0x1000000,d+c*0x1000000,inputs[i],outputs[i],begins[i],1);
        assert(launches==nc&&last_core_count==1);auto expected=snapshot(d,prod(outputs[i]));
        seed(s,prod(inputs[i]));launches=0;
        rpu_launch_slice_spm_kernel(s,d,inputs[i],outputs[i],begins[i],nc,true);
        assert(launches==1&&last_core_count==nc);
        assert(snapshot(d,prod(outputs[i]))==expected);
        assert(RpuKernelGraph::active().kernel.pair(0)==0x10200+begins[i].back()*2);
        assert(RpuKernelGraph::active().kernel.pair(2)==0x180400);
        for(size_t core=0;core<size_t(nc);++core)for(size_t row=0;row<size_t(outputs[i][0]);++row)
            for(size_t col=0;col<size_t(outputs[i][1]);++col)
                assert(memory.at(d+core*0x1000000+(row*outputs[i][1]+col)*2)==
                       uint16_t(core*733+(row*inputs[i][1]+begins[i][1]+col)*19));
    }
    for(int arm=0;arm<9;++arm){
        const int before=staged,gets=lookups,calls=launches;
        auto input=s,output=d;int cores=8;
        std::vector<int64_t> shape{64,16},out{64,2},begin{0,0};
        if(arm==0)input=SPM_ALLOC.addr(1,0x2000);
        if(arm==1)output=SPM_ALLOC.addr(0,SpmAllocator::SPM_USABLE-2);
        if(arm==2)SPM_ALLOC.origins[7]+=0x100;
        if(arm==3)cores=9;
        if(arm==4)SPM_ALLOC.initialized=false;
        if(arm==5)begin[1]=15;
        if(arm==6)shape[0]=std::numeric_limits<int64_t>::max();
        if(arm==7)input=SPM_ALLOC.addr(0,SpmAllocator::SPM_USABLE-1024);
        if(arm==8)output=SPM_ALLOC.addr(1,0x2000);
        try{rpu_launch_slice_spm_kernel(input,output,shape,out,begin,cores,true);assert(false);}
        catch(const std::runtime_error&){}
        assert(staged==before&&lookups==gets&&launches==calls);
        if(arm==2)SPM_ALLOC.origins[7]-=0x100;
        SPM_ALLOC.initialized=true;
    }
}
'''
    cpp, binary = tmp_path / "g05_gdn_slice.cpp", tmp_path / "g05_gdn_slice"
    cpp.write_text(code)
    build = subprocess.run(["g++", "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
                           capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr


def test_g05_actual_local_slice_profile_complete_schedule_and_identity(tmp_path):
    from test_g05_vision_owner_model import _block
    from test_g05_q_gate_native import _compile_run
    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_qwen3_5_model.h").read_text()
    methods = "\n".join(
        declaration + _block(source, stem)
        for declaration, stem in (
            ("bool use_g05_gdn_local_slice(int64_t rows)const", "bool Qwen3_5Model::use_g05_gdn_local_slice("),
            ("std::vector<int64_t>g05_gdn_local_slice_arguments(int64_t rows)const", "std::vector<int64_t> Qwen3_5Model::g05_gdn_local_slice_arguments("),
        )
    )
    start = source.index("    const bool g05_local_slice =")
    caller = source[start:source.index("    // ===== Phase 3:", start)]
    manifest = _block(source, "if (use_g05_gdn_local_slice(chunk.len))")
    layout = _block(source, "if (use_g05_gdn_local_slice(256))")
    cost = _block(header, "if (use_g05_gdn_local_slice(256))")
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
constexpr int64_t QWEN35_TEXT_GDN_LOCAL_SLICE_SITE=4624574716066886237LL;
namespace detail{int64_t layout_mix(int64_t x,int64_t y){return uint64_t(x)*131+uint64_t(y);}}
struct Spm{uint32_t addr(int core,int off)const{return 0x40000400u+core*0x1000000u+off;}}SPM_ALLOC;
enum class FmbRouteFamily{GRAPH_SCHEDULE};
struct ChunkInfo{int64_t len=256,idx=2;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
struct Call{uint32_t src,dst;std::vector<int64_t>is,os,begin;int cores;bool local;};
std::vector<Call>calls;
void rpu_launch_slice_spm_kernel(uint32_t s,uint32_t d,const std::vector<int64_t>&is,const std::vector<int64_t>&os,const std::vector<int64_t>&b,int n,bool local=false){calls.push_back({s,d,is,os,b,n,local});}
struct Model{
 bool action_mode_=false,z2_bound_=false,debug=false,fast_replay_enabled_=true,retained_prefill_graph_=true,has_gdn_=true,complete=true;
 int cores=8,layers=24,h=2048,inter=6144,nq=8,nkv=8,hd=256;
 int64_t gdn_nvh_=16,gdn_dk_=128,gdn_dv_=128,gdn_conv_dim_=6144,gdn_kc_=4;
 std::vector<int>layer_is_full_=std::vector<int>(24);
 std::vector<Route>routes;int consumed=0;
 Model(){for(int i=0;i<24;++i)layer_is_full_[i]=((i+1)%4==0);}
 bool get_debug_export()const{return debug;}int num_cores()const{return cores;}int gdn_tp()const{return cores==8?8:4;}
 int num_layers()const{return layers;}int hidden_size()const{return h;}int intermediate_size()const{return inter;}
 int num_q_heads()const{return nq;}int num_kv_heads()const{return nkv;}int head_dim()const{return hd;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t invocation){
   TORCH_CHECK(routes.size()==1);const auto&r=routes[0];
   TORCH_CHECK(r.site==site&&r.family==f&&r.selector==selector&&r.flags==flags&&r.args==args&&r.invocation==invocation);++consumed;
 }
 __METHODS__
 void plan(ChunkInfo chunk={}){struct{std::vector<Route>routes;}manifest;if(use_g05_gdn_local_slice(chunk.len))__MANIFEST__ routes=manifest.routes;}
 int64_t layout_hash()const{int64_t hash=0;if(use_g05_gdn_local_slice(256))__LAYOUT__ return hash;}
 std::vector<int64_t>cost_identity()const{std::vector<int64_t>identity{1};if(use_g05_gdn_local_slice(256))__COST__ return identity;}
 void slice(ChunkInfo chunk,bool glast=false){int64_t L=chunk.len;int nc=cores;__CALLER__
   slice_pc(SPM_ALLOC.addr(0,0x10000),SPM_ALLOC.addr(0,0x20000),glast?std::vector<int64_t>{2,64}:std::vector<int64_t>{L,16},glast?std::vector<int64_t>{2,1}:std::vector<int64_t>{L,2},glast?std::vector<int64_t>{0,63}:std::vector<int64_t>{0,0});
 }
};
template<class F>void reject(F f){auto n=calls.size();try{f();assert(false);}catch(const std::runtime_error&){}assert(calls.size()==n);}
int main(){
 Model m;auto hash=m.layout_hash();auto identity=m.cost_identity();assert(hash&&identity.size()>1);
 for(int rows:{64,128,192,256})for(bool glast:{false,true}){
   ChunkInfo chunk{rows,3};m.plan(chunk);assert(m.routes.size()==1);auto&r=m.routes[0];
   assert(r.args==std::vector<int64_t>({1,rows,64,2,16,8,2+2*(rows/64),1,2,0x40000000,0x1000000}));
   calls.clear();m.slice(chunk,glast);assert(calls.size()==1&&calls[0].cores==8&&calls[0].local);
   assert(calls[0].src==SPM_ALLOC.addr(0,0x10000)&&calls[0].dst==SPM_ALLOC.addr(0,0x20000));
   for(size_t i=0;i<r.args.size();++i){auto x=m;++x.routes[0].args[i];reject([&]{x.slice(chunk,glast);});}
   for(int bad=0;bad<5;++bad){auto x=m;
   if(bad==0)x.complete=false;
   if(bad==1)++x.routes[0].selector;
   if(bad==2)++x.routes[0].flags;
   if(bad==3)++x.routes[0].invocation;
   if(bad==4)x.routes.clear();
   reject([&]{x.slice(chunk,glast);});}
 }
 for(int bad=0;bad<12;++bad){auto x=m;
   if(bad==0)x.action_mode_=true;
   if(bad==1)x.z2_bound_=true;
   if(bad==2)x.debug=true;
   if(bad==3)x.fast_replay_enabled_=false;
   if(bad==4)x.retained_prefill_graph_=false;
   if(bad==5)x.cores=4;
   if(bad==6)x.h=1024;
   if(bad==7)x.gdn_nvh_=32;
   if(bad==8)x.gdn_dk_=64;
   if(bad==9)x.gdn_conv_dim_=4096;
   if(bad==10)x.layer_is_full_[0]=1;
   if(bad==11)x.layers=28;
   x.plan();assert(x.routes.empty()&&x.layout_hash()!=hash&&x.cost_identity()!=identity);
   calls.clear();x.slice({});assert(calls.size()==size_t(x.cores));
   for(int core=0;core<x.cores;++core)assert(!calls[core].local&&calls[core].cores==1&&calls[core].src==SPM_ALLOC.addr(core,0x10000)&&calls[core].dst==SPM_ALLOC.addr(core,0x20000));
 }
 for(int rows:{1,32,63,65,255,257,320})assert(!m.use_g05_gdn_local_slice(rows));
}
'''
    replacements = {"__METHODS__": methods, "__MANIFEST__": manifest, "__CALLER__": caller,
                    "__LAYOUT__": layout, "__COST__": cost}
    for marker, replacement in replacements.items():
        cpp = cpp.replace(marker, replacement)
    _compile_run(tmp_path, "g05_gdn_local_schedule", cpp)
