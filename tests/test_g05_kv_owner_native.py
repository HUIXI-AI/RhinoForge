"""Compile actual cold admission/descriptor bodies without loading an device model."""
from pathlib import Path
import subprocess

from pi_wall_host_catalog import with_actual_linear_contracts
from test_siglip_cold_profile import _body
from test_g05_vision_owner_model import _block


def test_kv_owner_cold_proof_fallback_and_fail_before_launch(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    methods = [
        ("void prepare_action_kv_owner()", "void Qwen3_5Model::prepare_action_kv_owner("),
        ("bool use_action_kv_owner(int64_t rows) const", "bool Qwen3_5Model::use_action_kv_owner("),
        ("bool use_action_mod_body() const", "bool Qwen3_5Model::use_action_mod_body("),
        ("std::vector<int64_t> action_kv_owner_arguments() const", "std::vector<int64_t> Qwen3_5Model::action_kv_owner_arguments("),
        ("void launch_action_kv_owner(int layer,const ChunkInfo& chunk)", "void Qwen3_5Model::launch_action_kv_owner("),
    ]
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
enum class KernelId {G05_KV_OWNER_FP16_ACC32_M32N256K1024,G05_KV_OWNER_FANOUT_M32D256,G05_KV_OWNER_W8A16_ACC32_M32N256K1024,G05_KV_OWNER_W8A16_ACC16_M32N256K1024};
struct KernelCache {
    bool linear=true,fanout=true;
    static KernelCache& instance(){static KernelCache k;return k;}
    bool has_loaded(KernelId id){return id!=KernelId::G05_KV_OWNER_FANOUT_M32D256?linear:fanout;}
};
namespace at{constexpr int kChar=3;}
struct Weight {bool replicated=true;int dtype=1;bool defined()const{return true;}int scalar_type()const{return dtype;}};
struct Weights {Weight k_w,v_w,k_ws,v_ws;};
int proofs=0,launches=0;
bool rpu_g05_kv_weight_is_replicated_fp16(const Weight& w){++proofs;return w.replicated;}
void rpu_launch_g05_kv_owner_linear_fp16_acc32_kernel(uint32_t,const Weight&,const Weight&,uint32_t,uint32_t){++launches;}
bool rpu_g05_kv_weight_is_replicated_w8a16(const Weight&w,const Weight&s){++proofs;return w.dtype==at::kChar&&w.replicated&&s.replicated;}
std::vector<bool>linear_modes;
void rpu_launch_g05_kv_owner_linear_w8a16_acc32_kernel(uint32_t,const Weight&w,const Weight&v,const Weight&s,const Weight&vs,uint32_t,uint32_t,bool linear_acc32=true){linear_modes.push_back(linear_acc32);assert(w.dtype==at::kChar&&v.dtype==at::kChar&&s.replicated&&vs.replicated);++launches;}
enum class FmbRouteFamily{LINEAR};
constexpr int64_t QWEN35_ACTION_KV_OWNER_SITE=9072654722328257913LL,QWEN35_ACTION_KV_OWNER_W8_SITE=73490380550116103LL;
struct ChunkInfo{int64_t len=32,idx=0;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> args;int64_t invocation;};
struct Model {
 bool use_action_nvfp4(int64_t)const{return false;}

 bool action_mod_body_fp16_=false,action_mod_body_w8_=false;
 bool action_gateup_resident_w8_=false,action_kv_owner_w8_=false,action_q_gate_w8_=false,action_gateup_pair_w8_=false,text_gateup_pair_w8_=false;
    bool action_mode_=true,linear_acc32_=true,has_gdn_=false,action_kv_owner_available_=false;
    bool action_step_active_=false,action_loop_active_=false,complete=true;
    int64_t action_len_=32,action_num_steps_=1,layers=24,h=1024,intermediate=4096,cores=8,heads=8,kv=8,hd=256,tp=8;
    std::vector<Weights> layer_weights_{24};
    std::vector<uint8_t> layer_is_full_=std::vector<uint8_t>(24,1);
    std::vector<Route> routes;
    int64_t num_layers()const{return layers;} int64_t hidden_size()const{return h;}
    int64_t intermediate_size()const{return intermediate;} int64_t num_cores()const{return cores;}
    int64_t attn_tp()const{return tp;} int64_t mlp_tp()const{return tp;}
    int64_t num_q_heads()const{return heads;} int64_t num_kv_heads()const{return kv;}
    int64_t head_dim()const{return hd;}
    Model& ctx(){return *this;} bool has_complete_physical_manifest()const{return complete;}
    uint32_t addr(int,const char*)const{return 0;}
    void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation){
        auto it=std::find_if(routes.begin(),routes.end(),[&](auto& r){return r.site==site&&r.invocation==invocation;});
        TORCH_CHECK(it!=routes.end()&&it->family==family&&it->selector==selector&&it->flags==flags&&it->args==args);
    }
    __METHODS__
    void plan(){struct {std::vector<Route> routes;} manifest;ChunkInfo chunk;
        if(use_action_kv_owner(chunk.len)) __MANIFEST__
        routes=manifest.routes;
    }
};
template<class F>void reject(F f){int before=launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(launches==before);}
int main(){
    Model m;m.prepare_action_kv_owner();assert(m.action_kv_owner_available_&&proofs==48);
    assert(!m.use_action_kv_owner(32)); // cold prime uses original K/V
    m.action_step_active_=m.action_loop_active_=true;m.action_num_steps_=10;
    m.plan();assert(m.routes.size()==1&&m.use_action_kv_owner(32)&&!m.use_action_kv_owner(31));
    assert(m.routes[0].args==std::vector<int64_t>({2,32,1024,2,8,256,8,80,80,512,0,4,2,6,16384,20,34,2,2,32,4,36,4,1}));
    for(int layer=0;layer<24;++layer)m.launch_action_kv_owner(layer,{});
    assert(launches==24&&proofs==48); // no weight scans in BUILD/REPLAY
    reject([&]{m.launch_action_kv_owner(24,{});});
    reject([&]{m.launch_action_kv_owner(0,{31,0});});
    m.complete=false;reject([&]{m.launch_action_kv_owner(0,{});});m.complete=true;
    m.routes[0].args[12]=0;reject([&]{m.launch_action_kv_owner(0,{});});
    m.layer_weights_.back().v_w.replicated=false;m.prepare_action_kv_owner();
    assert(!m.action_kv_owner_available_);m.plan();assert(m.routes.empty());
    m.layer_weights_.back().v_w.replicated=true;
    for(bool linear:{false,true}){KernelCache::instance().linear=linear;KernelCache::instance().fanout=!linear;
        m.prepare_action_kv_owner();assert(!m.action_kv_owner_available_);}
    KernelCache::instance().linear=KernelCache::instance().fanout=true;
    m.linear_acc32_=false;m.prepare_action_kv_owner();assert(!m.action_kv_owner_available_);
    m.linear_acc32_=true;m.layer_is_full_[23]=0;m.prepare_action_kv_owner();assert(!m.action_kv_owner_available_);
    m.layer_is_full_[23]=1;m.heads=16;m.prepare_action_kv_owner();assert(!m.action_kv_owner_available_);
    m.heads=8;m.prepare_action_kv_owner();assert(m.action_kv_owner_available_);
 for(auto&w:m.layer_weights_)w.k_w.dtype=w.v_w.dtype=at::kChar;
 m.prepare_action_kv_owner();assert(m.action_kv_owner_available_&&m.action_kv_owner_w8_);m.plan();
 assert(m.routes[0].site==QWEN35_ACTION_KV_OWNER_W8_SITE&&m.routes[0].args==std::vector<int64_t>({2,32,1024,2,8,256,8,96,48,512,0,4,2,6,16384,20,34,1,2,32,4,36,6,1}));
 const int w8proofs=proofs;
 for(int layer=0;layer<24;++layer)m.launch_action_kv_owner(layer,{});
 assert(proofs==w8proofs);m.routes[0].args[16]=20;reject([&]{m.launch_action_kv_owner(0,{});});
 m.plan();
 {auto x=m;x.linear_acc32_=false;x.action_mod_body_w8_=true;x.prepare_action_kv_owner();x.plan();assert(x.use_action_kv_owner(32)&&x.routes[0].args[7]==48);auto valid=x.routes;x.routes=m.routes;reject([&]{x.launch_action_kv_owner(0,{});});x.routes=valid;linear_modes.clear();x.launch_action_kv_owner(23,{});assert(linear_modes==std::vector<bool>({false}));}
 m.layer_weights_.back().v_ws.replicated=false;m.prepare_action_kv_owner();assert(!m.action_kv_owner_available_&&!m.action_kv_owner_w8_);m.plan();assert(m.routes.empty());
 m.layer_weights_.back().v_ws.replicated=true;m.layer_weights_.back().k_w.dtype=1;m.prepare_action_kv_owner();assert(!m.action_kv_owner_available_);

}
'''
    cpp = cpp.replace("__METHODS__", "\n".join(decl + _body(source, marker) for decl, marker in methods))
    cpp = cpp.replace("__MANIFEST__", _block(source, "if (use_action_kv_owner(chunk.len))"))
    path = tmp_path / "owner.cpp"
    path.write_text(with_actual_linear_contracts(cpp))
    exe = tmp_path / "owner"
    subprocess.run(["c++", "-std=c++17", "-O0", "-I", str(root / "src/ops"), "-I", str(root / "src/core"), str(path), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_actual_kv_owner_launcher_preserves_pending_kernel_and_register_abi(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/ops/rpu_linear.cpp").read_text()
    body = _body(source, "void rpu_launch_g05_kv_owner_linear_fp16_acc32_kernel(")
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10 {using Half=uint16_t;enum class DeviceType{PrivateUse1};}
namespace at {
constexpr int kHalf=1,kChar=3;
struct Tensor {
    uint64_t address;
    bool defined()const{return true;}
    int scalar_type()const{return kHalf;}
    struct Device {c10::DeviceType type()const{return c10::DeviceType::PrivateUse1;}};
    Device device()const{return {};}
    bool is_contiguous()const{return true;}
    int dim()const{return 2;}
    int64_t size(int axis)const{return axis==0?2048:1024;}
    template<class T>T* data_ptr()const{return reinterpret_cast<T*>(address);}
};
}
struct SpmAllocator {
    static constexpr uint32_t SPM_USABLE=8303616;
    bool is_initialized()const{return true;}
    uint32_t addr(int core,int offset)const{assert(core==0);return 0x20000000+offset;}
} allocator;
#define SPM_ALLOC allocator
enum class KernelId{G05_KV_OWNER_FP16_ACC32_M32N256K1024,G05_KV_OWNER_FANOUT_M32D256};
constexpr auto OWNER=KernelId::G05_KV_OWNER_FP16_ACC32_M32N256K1024;
constexpr auto FANOUT=KernelId::G05_KV_OWNER_FANOUT_M32D256;
struct KernelCache {
    bool owner=true,fanout=true;
    static KernelCache& instance(){static KernelCache cache;return cache;}
    bool has_loaded(KernelId id)const{return id==OWNER?owner:fanout;}
};
enum class GraphDdrRegisterRole{ModelWeight};
enum class GraphDdrRegisterAccess{Read};
enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {
    GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
    GraphDdrRegisterEncoding encoding;int lo,hi;
};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor owner;};
struct Kernel {
    KernelId id;
    std::map<int,uint16_t> regs;
    void reset_regs(){regs.clear();}
    void set_regs(int index,uint16_t value){regs[index]=value;}
    void set_regs(int,uint32_t)=delete; // Catch accidental overlapping full-word writes.
};
struct Submission {
    KernelId id;
    std::vector<uint16_t> grid;
    std::map<int,uint16_t> regs;
    std::vector<GraphDdrRegisterOperandSpec> operands;
};
struct Writer {void write(Kernel&);};
struct RpuKernelGraph {
    std::optional<KernelId> pending,staged;
    std::vector<GraphDdrRegisterOperandSpec> operands;
    std::vector<Submission> submissions;
    Kernel owner{OWNER,{}},fanout{FANOUT,{}};
    bool written=false;
    int stages=0,lookups=0;
    static RpuKernelGraph& active(){static RpuKernelGraph graph;return graph;}
    Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> specs){
        assert(!pending&&!staged&&id==OWNER&&specs.size()==2);
        staged=id;operands=std::move(specs);written=false;++stages;return {};
    }
    Kernel* get(KernelId id){
        // Recording and replay both have one upcoming kernel: a second GET
        // cannot replace that identity before its submission consumes it.
        assert(!pending&&KernelCache::instance().has_loaded(id));
        if(id==OWNER)assert(staged==OWNER);
        else assert(!staged);
        pending=id;++lookups;return id==OWNER?&owner:&fanout;
    }
};
void Writer::write(Kernel& kernel){
    auto& g=RpuKernelGraph::active();
    assert(g.staged==OWNER&&g.pending==OWNER&&kernel.id==OWNER&&!g.written);
    for(const auto& operand:g.operands){
        const auto encoded=operand.owner.address>>8;
        kernel.set_regs(operand.abi.lo,static_cast<uint16_t>(encoded));
        kernel.set_regs(operand.abi.hi,static_cast<uint16_t>(encoded>>16));
    }
    g.written=true;
}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
std::vector<uint64_t> flushed;
void rpu_ddr_flush(c10::Half* pointer){flushed.push_back(reinterpret_cast<uint64_t>(pointer));}
struct Queue {
    bool broadcast=false;
    void set_broadcast_mode(bool value){broadcast=value;}
    void enqueu_kernel(Kernel& kernel,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
        auto& g=RpuKernelGraph::active();
        assert(broadcast&&g.pending==kernel.id);
        assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
        if(kernel.id==OWNER)assert(g.staged==OWNER&&g.written);
        else assert(!g.staged&&!g.written&&g.operands.empty());
        g.submissions.push_back({kernel.id,grid,kernel.regs,g.operands});
        g.pending.reset();g.staged.reset();g.operands.clear();g.written=false;
    }
} queue;
Queue* get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
void launch(uint32_t input,const at::Tensor& key_weight,const at::Tensor& value_weight,
            uint32_t key_output,uint32_t value_output) __BODY__
uint32_t pair(const Submission& s,int reg){return uint32_t(s.regs.at(reg))|(uint32_t(s.regs.at(reg+1))<<16);}
template<class F>void reject_before_launch(F run){
    auto& g=RpuKernelGraph::active();
    const auto submitted=g.submissions.size(),flushes=flushed.size();
    const int stages=g.stages,lookups=g.lookups;
    try{run();assert(false);}catch(const std::runtime_error&){}
    assert(g.submissions.size()==submitted&&flushed.size()==flushes);
    assert(g.stages==stages&&g.lookups==lookups&&!g.pending&&!g.staged);
}
int main(){
    auto& g=RpuKernelGraph::active();
    for(uint32_t pass=0;pass<2;++pass){
        const uint32_t delta=pass*0x40000;
        const uint32_t input=0x20020000+delta,k=0x20040000+delta,v=0x20050000+delta;
        const at::Tensor kw{0x1234567800ULL+pass*0x400000},vw{0x2345678900ULL+pass*0x400000};
        launch(input,kw,vw,k,v);
        assert(g.submissions.size()==(pass+1)*2&&!g.pending&&!g.staged);
        const auto& owner=g.submissions[pass*2];const auto& fanout=g.submissions[pass*2+1];
        assert(owner.id==OWNER&&owner.grid==std::vector<uint16_t>({4,1,1}));
        assert(fanout.id==FANOUT&&fanout.grid==std::vector<uint16_t>({1,1,1}));
        assert(pair(owner,0)==input&&pair(owner,4)==k&&pair(owner,36)==v);
        assert(pair(owner,2)==kw.address>>8&&pair(owner,32)==vw.address>>8);
        assert(pair(owner,6)==0&&owner.regs.at(8)==1&&owner.regs.at(9)==32&&pair(owner,10)==256);
        assert(owner.regs.at(12)==1024&&owner.regs.at(13)==64&&owner.regs.at(14)==0);
        assert(owner.regs.at(15)==8&&pair(owner,16)==2048&&pair(owner,18)==512);
        assert(pair(owner,20)==0&&pair(owner,22)==0&&pair(owner,24)==0&&owner.regs.at(26)==2&&owner.regs.at(27)==2);
        assert(pair(fanout,0)==k-allocator.addr(0,0)&&pair(fanout,2)==v-allocator.addr(0,0));
        assert(owner.operands.size()==2&&fanout.operands.empty());
        for(int i=0;i<2;++i){
            const auto& operand=owner.operands[i];
            assert(operand.abi.role==GraphDdrRegisterRole::ModelWeight);
            assert(operand.abi.access==GraphDdrRegisterAccess::Read);
            assert(operand.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
            assert(operand.abi.lo==(i?32:2)&&operand.abi.hi==(i?33:3));
            assert(operand.owner.address==(i?vw.address:kw.address));
        }
        assert(flushed[pass*2]==kw.address&&flushed[pass*2+1]==vw.address);
    }
    const at::Tensor kw{0x100000000ULL},vw{0x200000000ULL};
    auto& cache=KernelCache::instance();
    for(bool owner:{false,true}){
        cache.owner=owner;cache.fanout=!owner;
        reject_before_launch([&]{launch(0x20020000,kw,vw,0x20040000,0x20050000);});
    }
    cache.owner=cache.fanout=true;
    reject_before_launch([&]{launch(0x20020000,kw,vw,0x20020000,0x20050000);});
    reject_before_launch([&]{launch(0x20020000,kw,vw,0x20040000,0x20040000);});
    assert(g.stages==2&&g.lookups==4&&g.submissions.size()==4);
}
'''.replace("__BODY__", body)
    path = tmp_path / "launcher.cpp"
    path.write_text(with_actual_linear_contracts(cpp))
    exe = tmp_path / "launcher"
    subprocess.run([
        "c++", "-std=c++17", "-O0", "-I", str(root / "src/ops"), "-I", str(root / "src/core"),
        str(path), "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)
