"""Execute native GR00T resident-FF1 policy, replay preload and physical ABI on host."""
from pathlib import Path
from pi_wall_host_catalog import with_actual_linear_contracts
import re
import shutil
import subprocess

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/fused/rpu_gr00t_dit_model.cpp"
LINEAR = ROOT / "src/ops/rpu_linear.cpp"


def _run(tmp_path, source):
    cpp, exe = tmp_path / "ff1.cpp", tmp_path / "ff1"
    cpp.write_text(with_actual_linear_contracts(source))
    result = subprocess.run([shutil.which("g++"), "-std=c++17", "-O1", "-I", str(ROOT),
                             str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout)


COMMON = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <initializer_list>
#include <memory>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do{if(!(c))throw std::runtime_error(#c);}while(0)
template<class F>void rejects(F f){try{f();assert(false);}catch(const std::runtime_error&) {}}
namespace c10 {using Half=uint16_t;enum class DeviceType{PrivateUse1,CPU};
template<class F>struct Exit{F f;~Exit(){f();}};template<class F>Exit<F>make_scope_exit(F f){return {f};}}
namespace at {
enum Type{kChar,kHalf};constexpr auto kPrivateUse1=c10::DeviceType::PrivateUse1;
struct Device{c10::DeviceType value=kPrivateUse1;c10::DeviceType type()const{return value;}};
struct Version{bool valid=true;int64_t value=0;bool enabled()const{return valid;}int64_t current_version()const{return value;}};
struct Impl{Version version;Version&version_counter(){return version;}};
struct Tensor{
 std::shared_ptr<std::vector<uint8_t>>bytes;std::shared_ptr<Impl>impl=std::make_shared<Impl>();
 std::vector<int64_t>shape;Type dtype=kChar;bool contiguous_=true;Device dev;
 Tensor()=default;
 Tensor(std::initializer_list<int64_t>s,Type t):shape(s),dtype(t){bytes=std::make_shared<std::vector<uint8_t>>(count(shape)*(t==kHalf?2:1));}
 static int64_t count(const std::vector<int64_t>&s){return std::accumulate(s.begin(),s.end(),int64_t{1},std::multiplies<int64_t>());}
 bool defined()const{return bool(bytes);}Type scalar_type()const{return dtype;}Device device()const{return dev;}
 bool is_contiguous()const{return contiguous_;}int64_t dim()const{return shape.size();}int64_t size(int i)const{return shape.at(i);}
 size_t nbytes()const{return bytes?bytes->size():0;}Impl*unsafeGetTensorImpl()const{return impl.get();}
 template<class T=void>T*data_ptr()const{return reinterpret_cast<T*>(bytes->data());}
 Tensor cpu()const{return *this;}Tensor to(Device)const{Tensor x=*this;x.bytes=std::make_shared<std::vector<uint8_t>>(*bytes);return x;}
 Tensor view(std::initializer_list<int64_t>s)const{assert(count(s)==count(shape));Tensor x=*this;x.shape=s;return x;}
 Tensor contiguous()const{return *this;}
 Tensor permute(std::initializer_list<int64_t>order)const{
  assert(shape.size()==4);std::vector<int64_t>o(order);Tensor x=*this;for(int d=0;d<4;++d)x.shape[d]=shape[o[d]];
  x.bytes=std::make_shared<std::vector<uint8_t>>(nbytes());
  for(int64_t index=0;index<count(x.shape);++index){int64_t rest=index;std::array<int64_t,4>v{};
   for(int d=3;d>=0;--d){v[o[d]]=rest%x.shape[d];rest/=x.shape[d];}
   int64_t input=((v[0]*shape[1]+v[1])*shape[2]+v[2])*shape[3]+v[3];(*x.bytes)[index]=(*bytes)[input];}return x;
 }
};}
// The device boundary maps CPU owners to stable aligned 40-bit RPU addresses.
namespace rhino_lkn{uint64_t RpuGetDevAddr(void*p){
 static std::map<void*,uint64_t> addresses;
 auto [it,added]=addresses.emplace(p,UINT64_C(0x100000000)+addresses.size()*UINT64_C(0x1000000));
 (void)added;return it->second;
}}
using rhino_lkn::RpuGetDevAddr;
struct RpuExecutionCoordinator{
 inline static int token=0;inline static bool busy=false;
 static void require_graph_quiescent(const char*){TORCH_CHECK(!busy);}
 static void require_no_physical_claim(const char*){TORCH_CHECK(!token);}
 static int enter_physical(const char*){TORCH_CHECK(!token);return token=1;}
 static void exit_physical_noexcept(int){token=0;}
 static int current_physical_token_if_owned(const char*){return token;}
};
void rpu_ddr_flush_force(void*){}void rpu_ddr_flush(void*){}
'''


def test_ff1_resident_native_lifecycle_routes_and_replay(tmp_path):
    source = MODEL.read_text()
    generated = (ROOT / "src/ops/rpu_specialized_linear_contracts.h").read_text()
    kernel_ids = sorted(set(re.findall(r"KernelId::(\w+)", generated)))
    generated = generated[generated.index("namespace rpu_specialized_linear {"):]
    methods = "\n".join(_definition(source, marker) for marker in (
        "void check_ff1_resident_setter_state(", "void clear_ff1_resident_weight(",
        "void install_ff1_resident_weight(", "void begin_ff1_resident_inference(",
        "bool ff1_resident_profile(", "std::vector<int64_t> ff1_resident_arguments(",
        "FmbForwardOperandResidency physical_forward_operand_residency(",
        "bool ff1_resident_selected(",
        "void load_ff1_resident_weight(",
    )).replace(" override", "")
    constants = "\n".join(re.findall(r"constexpr int64_t GR00T_DIT_FF1_RESIDENT_\w+ = .*?;", source))
    ff1 = _definition(source, "void finish_attn_and_ffn(")
    ff1 = ff1[ff1.index("        if (layer_idx == 0"):ff1.index("        rpu_launch_eltwise_unary_spm_kernel(")]
    cpp = COMMON + r'''
#define NUM_CORES 8
__CONSTANTS__
__ENV__
__CURRENT_CONTRACT__
struct KernelCache{inline static bool loaded=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId){return loaded;}};
struct RpuKernelGraph{inline static bool active=false;static bool has_active(){return active;}};
int pack_calls=0,preloads=0,resident_calls=0,ddr_calls=0,spm_value=-1,request_value=7;
bool mutate_during_pack=false;
at::Tensor rpu_pack_gr00t_ff1_resident_w8(const at::Tensor&w){
 assert(RpuExecutionCoordinator::token==1);++pack_calls;
 if(mutate_during_pack)++w.impl->version.value;
 return at::Tensor({8,1179648},at::kChar);
}
void rpu_launch_gr00t_ff1_resident_w8_preload(uint64_t&base,const at::Tensor&t,uint32_t dst){
 assert(base==RpuGetDevAddr(t.data_ptr())&&dst==0x10200000);++preloads;spm_value=(*t.bytes)[0];
}
void rpu_launch_gr00t_ff1_resident_w8_kernel(uint32_t a,uint32_t w,uint32_t o,uint32_t b,const at::Tensor&s){
 assert(a==0x10000000&&w==0x10200000&&o==0x10020000&&b==0x10700000&&s.size(0)==6144);
 assert(spm_value==request_value);++resident_calls;
}
template<class...T>void rpu_launch_linear_spm_to_spm_acc16_kernel(T...){++ddr_calls;}
enum class FmbRouteFamily{GRAPH_SCHEDULE,MUTABLE_DMA,LINEAR};
enum class FmbForwardOperandResidency{UNSPECIFIED,PER_LAYER,FORWARD};
enum class FmbPhysicalManifestState{UNSPECIFIED,COMPLETE};
struct Route{FmbRouteFamily family;int64_t site_id,selector,flags;std::vector<int64_t>arguments;int64_t invocation;};
struct FmbPhysicalExecutionManifest{FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;int64_t physical_length=41;std::vector<Route>routes;};
enum class Gr00tDiTMutableDmaRoute{DDR_SCATTER_TO_SPM=1};enum class FmbLinearRouteSelector{AUTO_TILE=1};
constexpr int64_t GR00T_DIT_FF1_LINEAR_SITE=1;
struct ChunkInfo{int64_t len=41,idx=0;};
struct Context{
 bool complete=true;int64_t seq_len=41,batch_size=1,body_iter=0,selector=2;int routes=0;
 int schedule_count=1;int64_t flags=0,invocation=0,physical_length=41;
 std::vector<int64_t>args;bool has_complete_physical_manifest()const{return complete;}
 FmbPhysicalExecutionManifest physical_manifest()const{
  FmbPhysicalExecutionManifest m;m.physical_length=physical_length;
  for(int i=0;i<schedule_count;++i)m.routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,GR00T_DIT_FF1_RESIDENT_SCHEDULE_SITE,selector,flags,args,invocation});return m;
 }
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selected,int64_t flags,const std::vector<int64_t>&arguments,int64_t invocation){
  assert(flags==0&&invocation==0);++routes;
  if(site==GR00T_DIT_FF1_LINEAR_SITE){assert(arguments.empty());return;}
  TORCH_CHECK(arguments==args);
  if(family==FmbRouteFamily::GRAPH_SCHEDULE)TORCH_CHECK(selected==selector);
  else if(family==FmbRouteFamily::LINEAR)TORCH_CHECK(selected==2);
  else TORCH_CHECK(selected==1);
 }
};
struct Model{
 bool ff1_resident_w8_opt_in_=gr00t_ff1_resident_w8_opt_in(),denoise_ready_=true,denoise_mode_=true,loop_mode_=true,inference_started_=false,z2_bound_=false;
 int64_t resolved=0,z2_prepared_seq_len_=0,num_steps_=4,action_horizon_=40,vl_seq_=82,cross_dim_=2048;
 int64_t h=1536,ff=6144,q=32,kv=32,hd=48,tp=8,layers=32;
 at::Tensor ff1_resident_weight_,ff1_resident_source_;uint64_t ff1_resident_src_base_=0;
 int64_t ff1_resident_generation_=0,ff1_resident_source_version_=0;
 struct LayerWeights{at::Tensor ff1_w{{6144,1536},at::kChar},ff1_b{{6144},at::kHalf},ff1_ws{{6144},at::kHalf};};
 std::vector<LayerWeights>layer_weights_=std::vector<LayerWeights>(1);Context context;
 int64_t get_last_resolved_chunk_size()const{return resolved;}int64_t num_layers()const{return layers;}
 int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return ff;}
 int64_t num_q_heads()const{return q;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}int64_t attn_tp()const{return tp;}int num_cores()const{return NUM_CORES;}
 Context&ctx(){return context;}const Context&ctx()const{return context;}
 uint32_t addr(int,const char*n){std::string s(n);return s=="ff1_resident_w8"?0x10200000:s=="norm_hidden"?0x10000000:0x10020000;}
 uint32_t layer_addr(int layer,int,const char*n){assert(layer>=0&&layer<32&&std::string(n)=="ff1_bias");return 0x10700000+layer*1536;}
 __METHODS__
 void record(bool capacity_fits=true){context.args=ff1_resident_arguments(context.seq_len,context.batch_size);context.physical_length=context.seq_len;context.selector=capacity_fits&&ff1_resident_profile(context.seq_len,context.batch_size)?2:1;}
 void emit(int layer_idx){ChunkInfo chunk;const auto&lw=layer_weights_[0];int64_t seq_len=41,h=1536,FF=6144;__FF1__}
};
int main(){
 unsetenv("RPU_GR00T_FF1_RESIDENT_W8");assert(!gr00t_ff1_resident_w8_opt_in());
 for(auto s:{"","0"}){setenv("RPU_GR00T_FF1_RESIDENT_W8",s,1);assert(!gr00t_ff1_resident_w8_opt_in());}
 setenv("RPU_GR00T_FF1_RESIDENT_W8","true",1);rejects([]{gr00t_ff1_resident_w8_opt_in();});
 setenv("RPU_GR00T_FF1_RESIDENT_W8","1",1);Model m;setenv("RPU_GR00T_FF1_RESIDENT_W8","0",1);assert(m.ff1_resident_w8_opt_in_);
 m.install_ff1_resident_weight();assert(pack_calls==1&&m.ff1_resident_generation_==1&&RpuExecutionCoordinator::token==0);m.record();
 // COMPLETE arguments bind the actual generated tile/grid and current ABI.
 const auto contract=rpu_specialized_linear::contract(KernelId::GR00T_FF1_RESIDENT_W8A16_M512N48K128);
 assert(m.context.args.size()==27);
 assert(std::vector<int64_t>(m.context.args.begin()+20,m.context.args.end())==
        std::vector<int64_t>({contract.m,contract.n,contract.k,contract.gx,contract.gy,contract.gz,2}));
 // Retired 23-word schedules and malformed lengths must fail before DMA/kernel.
 for(size_t n:{size_t(0),size_t(1),size_t(23),size_t(26),size_t(28)}){
  Model x=m;x.context.args.resize(n);const int loads=preloads,kernels=resident_calls;
  rejects([&]{x.load_ff1_resident_weight();});rejects([&]{x.emit(0);});
  assert(preloads==loads&&resident_calls==kernels);
 }
 for(int request=0;request<3;++request){spm_value=-1;request_value=7+request;(*m.ff1_resident_weight_.bytes)[0]=request_value;m.begin_ff1_resident_inference();
  for(int body=0;body<4;++body){m.context.body_iter=body;m.load_ff1_resident_weight();for(int layer=0;layer<32;++layer)m.emit(layer);}}
 assert(preloads==3&&resident_calls==12&&ddr_calls==372);
 // The ten-step owner is installed cold, then each request reuses exactly
 // one body-0 preload for layer zero. Changed packed bytes prove replay refresh.
 setenv("RPU_GR00T_FF1_RESIDENT_W8","1",1);Model ten;setenv("RPU_GR00T_FF1_RESIDENT_W8","0",1);
 ten.num_steps_=10;ten.vl_seq_=169;ten.install_ff1_resident_weight();ten.record();
 assert(ten.ff1_resident_profile(41,1)&&ten.ff1_resident_generation_==1);
 for(int request=0;request<3;++request){spm_value=-1;request_value=31+request;
  (*ten.ff1_resident_weight_.bytes)[0]=request_value;ten.begin_ff1_resident_inference();
  const int before_load=preloads,before_resident=resident_calls,before_ddr=ddr_calls;
  for(int body=0;body<10;++body){ten.context.body_iter=body;ten.load_ff1_resident_weight();
   assert(preloads==before_load+1);
   for(int layer=0;layer<32;++layer)ten.emit(layer);
   assert(resident_calls==before_resident+body+1&&ddr_calls==before_ddr+(body+1)*31);
  }
 }
 for(int body:{-1,10}){ten.context.body_iter=body;const int before=preloads;
  rejects([&]{ten.load_ff1_resident_weight();});assert(preloads==before);}
 ten.context.body_iter=0;
 // Both directions of 4/82 <-> 10/169 COMPLETE drift fail before DMA/kernel.
 for(bool old_owner:{false,true}){Model x=old_owner?m:ten;x.context.body_iter=0;
  x.context.args=old_owner?ten.context.args:m.context.args;
  const int loads=preloads,kernels=resident_calls;
  rejects([&]{x.load_ff1_resident_weight();});rejects([&]{x.emit(0);});
  assert(preloads==loads&&resident_calls==kernels);
 }
 for(size_t field=0;field<ten.context.args.size();++field){Model x=ten;++x.context.args[field];
  const int loads=preloads,kernels=resident_calls;
  rejects([&]{x.load_ff1_resident_weight();});rejects([&]{x.emit(0);});
  assert(preloads==loads&&resident_calls==kernels);
 }
 {Model x=ten;x.context.selector=3;const int loads=preloads;
  rejects([&]{x.load_ff1_resident_weight();});assert(preloads==loads);}
 // Eligible geometry can still bind the ordinary DDR path on cold budget failure.
 for(auto profile:{std::pair{1,4},std::pair{256,4},std::pair{97,10},std::pair{214,10},std::pair{256,10},std::pair{193,4},std::pair{225,4},std::pair{289,4},std::pair{238,4},std::pair{270,4},std::pair{334,4}}){
  Model x=ten;x.vl_seq_=profile.first;x.num_steps_=profile.second;x.context.body_iter=0;
  assert(x.ff1_resident_profile(41,1));x.record(false);const int loads=preloads,kernels=resident_calls,ddr=ddr_calls;
  assert(!x.ff1_resident_selected());x.load_ff1_resident_weight();x.emit(0);
  assert(preloads==loads&&resident_calls==kernels&&ddr_calls==ddr+1);
 }
 for(int bad=0;bad<5;++bad){Model x=ten;switch(bad){case 0:x.context.schedule_count=0;break;case 1:x.context.schedule_count=2;break;case 2:x.context.flags=1;break;case 3:x.context.invocation=1;break;case 4:x.context.physical_length=48;break;}
  const int loads=preloads,kernels=resident_calls;rejects([&]{x.load_ff1_resident_weight();});rejects([&]{x.emit(0);});assert(preloads==loads&&resident_calls==kernels);}
 for(auto profile:{std::pair{0,4},std::pair{-1,10},std::pair{82,3},std::pair{169,5}}){
  Model x=ten;x.vl_seq_=profile.first;x.num_steps_=profile.second;
  assert(!x.ff1_resident_profile(41,1));const int loads=preloads,kernels=resident_calls,ddr=ddr_calls;
  rejects([&]{x.load_ff1_resident_weight();});x.record();x.load_ff1_resident_weight();x.emit(0);
  assert(preloads==loads&&resident_calls==kernels&&ddr_calls==ddr+1);
 }
 ++ten.ff1_resident_source_.impl->version.value;
 {const int loads=preloads,kernels=resident_calls;rejects([&]{ten.begin_ff1_resident_inference();});assert(preloads==loads&&resident_calls==kernels);}
 --ten.ff1_resident_source_.impl->version.value;
 ten.ff1_resident_source_.impl->version.valid=false;rejects([&]{ten.begin_ff1_resident_inference();});ten.ff1_resident_source_.impl->version.valid=true;
 Model mutating=ten;mutating.clear_ff1_resident_weight();mutate_during_pack=true;
 rejects([&]{mutating.install_ff1_resident_weight();});mutate_during_pack=false;
 assert(!mutating.ff1_resident_weight_.defined()&&RpuExecutionCoordinator::token==0);
 for(int body:{-1,4}){m.context.body_iter=body;rejects([&]{m.load_ff1_resident_weight();});}m.context.body_iter=0;
 auto oldargs=m.context.args;++m.ff1_resident_generation_;rejects([&]{m.load_ff1_resident_weight();});--m.ff1_resident_generation_;
 m.context.selector=3;rejects([&]{m.load_ff1_resident_weight();});m.context.selector=2;
 for(int drift=0;drift<12;++drift){Model x=m;switch(drift){case 0:x.loop_mode_=false;break;case 1:x.denoise_mode_=false;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:x.vl_seq_=0;break;case 5:x.context.seq_len=40;break;case 6:x.context.batch_size=2;break;case 7:x.num_steps_=3;break;case 8:x.tp=4;break;case 9:x.h=1024;break;case 10:x.ff1_resident_w8_opt_in_=false;break;case 11:x.ff1_resident_weight_=at::Tensor{};break;}
  assert(!x.ff1_resident_profile(x.context.seq_len,x.context.batch_size));x.record();int count=preloads;x.load_ff1_resident_weight();assert(preloads==count);}
 m.context.complete=false;int count=preloads;m.load_ff1_resident_weight();m.emit(0);assert(preloads==count);m.context.complete=true;
 ++m.ff1_resident_source_.impl->version.value;rejects([&]{m.begin_ff1_resident_inference();});--m.ff1_resident_source_.impl->version.value;
 m.ff1_resident_source_.impl->version.valid=false;rejects([&]{m.begin_ff1_resident_inference();});m.ff1_resident_source_.impl->version.valid=true;
 for(int bad=0;bad<7;++bad){Model x=m;switch(bad){case 0:x.inference_started_=true;break;case 1:x.resolved=41;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:RpuKernelGraph::active=true;break;case 5:RpuExecutionCoordinator::busy=true;break;case 6:RpuExecutionCoordinator::token=1;break;}rejects([&]{x.install_ff1_resident_weight();});RpuKernelGraph::active=false;RpuExecutionCoordinator::busy=false;RpuExecutionCoordinator::token=0;}
 KernelCache::loaded=false;rejects([&]{m.install_ff1_resident_weight();});assert(!RpuExecutionCoordinator::token);KernelCache::loaded=true;
 m.layer_weights_[0].ff1_w.impl->version.valid=false;rejects([&]{m.install_ff1_resident_weight();});m.layer_weights_[0].ff1_w.impl->version.valid=true;
 mutate_during_pack=true;rejects([&]{m.install_ff1_resident_weight();});mutate_during_pack=false;assert(!RpuExecutionCoordinator::token);
 m.clear_ff1_resident_weight();assert(!m.ff1_resident_weight_.defined()&&!m.ff1_resident_source_.defined()&&!m.ff1_resident_src_base_);m.install_ff1_resident_weight();assert(m.ff1_resident_generation_==2);
 Model off;KernelCache::loaded=false;off.install_ff1_resident_weight();assert(!off.ff1_resident_weight_.defined());
 Model fp=m;fp.clear_ff1_resident_weight();fp.layer_weights_[0].ff1_w.dtype=at::kHalf;fp.install_ff1_resident_weight();assert(!fp.ff1_resident_weight_.defined());
}
'''
    cpp = (cpp.replace("__CURRENT_CONTRACT__", "enum class KernelId{" + ",".join(kernel_ids) + "};\n" + generated)
           .replace("__CONSTANTS__", constants)
           .replace("__ENV__", _definition(source, "bool gr00t_ff1_resident_w8_opt_in("))
           .replace("__METHODS__", methods).replace("__FF1__", ff1))
    _run(tmp_path, cpp)


def test_ff1_resident_packing_preload_and_bias_register_abi(tmp_path):
    source = LINEAR.read_text()
    functions = "\n".join(_definition(source, marker) for marker in (
        "void check_gr00t_ff1_resident_spm(", "at::Tensor rpu_pack_gr00t_ff1_resident_w8(",
        "void rpu_launch_gr00t_ff1_resident_w8_preload(", "void rpu_launch_gr00t_ff1_resident_w8_kernel(",
    ))
    cpp = COMMON + r'''
#include "src/ops/rpu_linear_tiling.h"
constexpr size_t kGr00tFf1ResidentW8PerCore=1179648;
struct SpmAllocator{static constexpr size_t SPM_USABLE=8387584;bool init=true;bool is_initialized(){return init;}uint32_t addr(int,int){return 0x10000000;}}SPM_ALLOC;
enum class KernelId{GR00T_FF1_RESIDENT_W8A16_M512N48K128};
struct Kernel_t{std::array<uint16_t,32>r{};void reset_regs(){r.fill(0);}void set_regs(int i,uint16_t v){r.at(i)=v;}}kernel;
struct Queue{bool broadcast=false;int launches=0;void set_broadcast_mode(bool v){broadcast=v;}
 void enqueu_kernel(Kernel_t&,std::vector<uint16_t>grid,std::initializer_list<int>cores){assert(broadcast&&grid==std::vector<uint16_t>({10,1,1})&&cores.size()==8);++launches;}}queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
enum class GraphDdrRegisterRole{PerChannelScale};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Writer{at::Tensor scale;void write(Kernel_t&k){rpu_pl_tiling::set_u32_reg_pair(k,20,RpuGetDevAddr(scale.data_ptr())>>8);}};
struct RpuKernelGraph{inline static bool active_=false;int keepalives=0;static bool has_active(){return active_;}static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor&){++keepalives;}
 Writer stage_kernel_ddr_registers(KernelId,const std::vector<GraphDdrRegisterOperandSpec>&ops){assert(ops.size()==1&&ops[0].abi.lo==20&&ops[0].abi.hi==21);return {ops[0].tensor};}};
enum class GraphOuterFastDmaSide{Source};
namespace graph_dma {bool enabled=true;int census=0;bool active(){return enabled;}
void stage_census_mutable_burst(uint64_t base,const at::Tensor&t,GraphOuterFastDmaSide,int64_t offset,size_t bytes,int cores){assert(base==RpuGetDevAddr(t.data_ptr())&&offset==0&&bytes==9437184&&cores==8);++census;}}
int dmas=0;void rpu_launch_ddr_scatter_spm_dma_mutable(uint64_t*,int64_t off,size_t halves,size_t stride,uint32_t dst,int cores){assert(off==0&&halves==589824&&stride==1179648&&dst==0x10200000&&cores==8);++dmas;}
__FUNCTIONS__
int main(){
 at::Tensor installed({6144,1536},at::kChar);for(size_t i=0;i<installed.nbytes();++i)(*installed.bytes)[i]=uint8_t((i/512*131+i%512*17+i/128512)%251);
 rejects([&]{rpu_pack_gr00t_ff1_resident_w8(installed);});RpuExecutionCoordinator::token=1;
 RpuKernelGraph::active_=true;rejects([&]{rpu_pack_gr00t_ff1_resident_w8(installed);});RpuKernelGraph::active_=false;
 auto packed=rpu_pack_gr00t_ff1_resident_w8(installed);assert(packed.size(0)==8&&packed.size(1)==1179648&&packed.bytes!=installed.bytes);
 for(size_t c=0;c<8;++c)for(size_t tile=0;tile<48*48;++tile)for(size_t b=0;b<512;++b)assert((*packed.bytes)[c*1179648+tile*512+b]==(*installed.bytes)[(tile*8+c)*512+b]);
 for(int bad=0;bad<4;++bad){auto w=installed;switch(bad){case 0:w.dtype=at::kHalf;break;case 1:w.shape[1]=1024;break;case 2:w.contiguous_=false;break;case 3:w.dev.value=c10::DeviceType::CPU;break;}rejects([&]{rpu_pack_gr00t_ff1_resident_w8(w);});}
 uint64_t base=RpuGetDevAddr(packed.data_ptr());rpu_launch_gr00t_ff1_resident_w8_preload(base,packed,0x10200000);assert(dmas==1&&graph_dma::census==1&&RpuKernelGraph::active().keepalives==1);
 ++base;rejects([&]{rpu_launch_gr00t_ff1_resident_w8_preload(base,packed,0x10200000);});--base;
 graph_dma::enabled=false;rejects([&]{rpu_launch_gr00t_ff1_resident_w8_preload(base,packed,0x10200000);});graph_dma::enabled=true;
 at::Tensor scale({6144},at::kHalf);uint32_t a=0x10000000,w=0x10200000,o=0x10020000,b=0x10700000;
 rpu_launch_gr00t_ff1_resident_w8_kernel(a,w,o,b,scale);assert(queue.launches==1);
 Kernel_t original;rpu_pl_tiling::set_parallel_linear_regs(original,{a,0,o,b,41,768,1536,1,8,3072,1536,RpuGetDevAddr(scale.data_ptr()),1,1});
 for(int i=0;i<32;++i)if(i!=2&&i!=3)assert(kernel.r[i]==original.r[i]);assert((uint32_t(kernel.r[3])<<16|kernel.r[2])==w);
 auto good=scale;for(int bad=0;bad<4;++bad){scale=good;switch(bad){case 0:scale.dtype=at::kChar;break;case 1:scale.shape[0]=6143;break;case 2:scale.contiguous_=false;break;case 3:scale.dev.value=c10::DeviceType::CPU;break;}rejects([&]{rpu_launch_gr00t_ff1_resident_w8_kernel(a,w,o,b,scale);});}scale=good;
 for(auto wrong:{a-256,w+1,0x10000000u+uint32_t(SpmAllocator::SPM_USABLE)-1179648u+256u})rejects([&]{rpu_launch_gr00t_ff1_resident_w8_kernel(a,wrong,o,b,scale);});
 rejects([&]{rpu_launch_gr00t_ff1_resident_w8_kernel(a,w,a,b,scale);});rejects([&]{rpu_launch_gr00t_ff1_resident_w8_kernel(a,w,o,w,scale);});
 SPM_ALLOC.init=false;rejects([&]{rpu_launch_gr00t_ff1_resident_w8_kernel(a,w,o,b,scale);});
 assert(queue.launches==1);
}
'''.replace("__FUNCTIONS__", functions)
    _run(tmp_path, cpp)


def test_ff1_resident_is_declared_and_invalidated_by_real_callers():
    source = MODEL.read_text()
    for name in ("set_weights", "set_denoise_weights"):
        setter = _definition(source.replace("= {}", "= at::Tensor()"), f"void {name}(")
        assert setter.index("check_ff1_resident_setter_state()") < setter.index("clear_ff1_resident_weight()")
        assert setter.index("clear_ff1_resident_weight()") < setter.index("install_ff1_resident_weight()") < setter.index("invalidate_model_state()")
    for name in ("forward", "step_forward", "unroll_forward", "unroll_forward_z2"):
        body = _definition(source.replace("= {}", "= at::IntArrayRef()"), f"at::Tensor {name}(")
        assert body.index("begin_ff1_resident_inference()") < body.index("TORCH_CHECK(")
    pre = _definition(source, "void emit_pre_layers_body(")
    assert pre.index("load_ff1_resident_weight()") < pre.index("time_tables_active()")
    declare = _definition(source, "std::vector<BufferDecl> baseline_buffer_declarations(")
    assert '"ff1_resident_w8", GR00T_DIT_FF1_RESIDENT_BYTES' in declare
    assert "ctx.forward_operand_residency == FmbForwardOperandResidency::FORWARD" in declare
    assert "ff1_resident_profile(41, ctx.batch_size)" in declare
    assert '0, 0, StorageClass::Temp, 0, nullptr, ALL' in declare
    layout = _definition(source, "int64_t subclass_layout_hash(")
    for field in ("ff1_resident_w8_opt_in_", "ff1_resident_generation_", "ff1_resident_weight_.defined()", "loop_mode_", "num_steps_"):
        assert field in layout
    manifest = _definition(source, "FmbPhysicalExecutionManifest physical_manifest_for_attention_routes(")
    for family in ("SCHEDULE", "DMA", "LINEAR"):
        assert f"GR00T_DIT_FF1_RESIDENT_{family}_SITE" in manifest
    assert "append_linear(GR00T_DIT_FF1_LINEAR_SITE)" in manifest
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    assert cache.index("GR00T_FF1_RESIDENT_W8A16_M512N48K128") > cache.index("UNARY_GELU_TANH_HIGH_PRECISION")
    assert '"gr00t_ff1_resident_w8a16_v2"' in cache
    assert '"gr00t_ff1_resident_w8a16_v2"' in (ROOT / "src/core/rpu_kernel_cache.inc").read_text()


def test_ff1_resident_actual_spm_declarations_fit_v6_and_do_not_alias(tmp_path):
    from test_gr00t_mask_residency import _program

    source = MODEL.read_text()
    core = (ROOT / "src/core/fused_model_base.cpp").read_text()
    allocator_h = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    cpp = _program(full_declarations=True)
    cross = "\n".join(_definition(source, marker) for marker in (
        "bool cross_kv_reuse_profile(", "int64_t cross_kv_bank_stride(",
        "int64_t cross_kv_bank_bytes(", "void append_cross_kv_buffers(",
    ))
    cpp = cpp.replace(
        "void append_cross_kv_buffers(std::vector<BufferDecl>&,const LayoutContext&)const{}",
        "using LayerWeights=Layer; bool cross_kv_reuse_opt_in_=true,denoise_ready_=true;\n"
        "int64_t num_kv_heads()const{return nq;}\n" + cross)
    cpp = cpp.replace(
        "bool ff1_resident_profile(int64_t,int64_t)const{return false;}",
        "bool ff1_resident_w8_opt_in_=false;\n"
        "struct Owner{bool defined()const{return true;}}ff1_resident_weight_,adaln_table_;\n" +
        _definition(source, "bool ff1_resident_profile("))
    cpp = cpp.replace("bool adaln_table_profile(int64_t,int64_t)const{return false;}",
                      _definition(source, "bool adaln_table_profile("))
    constants = "\n".join(re.search(r"static constexpr size_t " + name + r"\s*=[^;]+;", allocator_h)[0]
                          for name in ("SPM_TOTAL", "SPM_RESERVED", "SPM_USABLE", "SPM_PLANNING_BUDGET"))
    cpp = cpp.replace("class SpmAllocator{public:", "class SpmAllocator{public:\n" + constants + r'''
 inline static int reads=0;
 inline static int64_t free_bytes=SPM_PLANNING_BUDGET;
 int64_t free_space()const{++reads;return free_bytes;}
 int64_t temporary_used()const{++reads;return 0;}
''')
    cpp = cpp.replace("namespace v3{class FusedModelBase{};", r'''
SpmAllocator SPM_ALLOC;
struct RpuKernelGraph{inline static bool active=false;static bool has_active(){return active;}};
namespace v3{class FusedModelBase{};
''')
    budget = _definition(core, "struct FixedSpmOverhead") + ";\n"
    budget += "\n".join(_definition(core, marker) for marker in (
        "static FixedSpmOverhead estimate_fixed_overhead(",
        "static int64_t available_temporary_spm_after_reset(",
        "static bool final_chunk_layout_fits(",
    ))
    cpp = cpp.replace("class Gr00tDiTModel{public:", budget + "\nclass Gr00tDiTModel{public:\nint64_t minimum_eager_spm_bytes()const{return 0;}")
    methods = _definition(source, "FmbForwardOperandResidency ff1_residency_for_candidate(")
    methods += "\n" + _definition(core, "bool FusedModelBase::declared_spm_layout_fits(").replace("FusedModelBase::", "")
    cpp = cpp.replace(" SdpaConfig make_sdpa_config(", r'''
 struct Impl{bool persistent_allocated_=false;}impl_;const Impl*pimpl_=&impl_;
''' + methods + "\n SdpaConfig make_sdpa_config(")
    cpp = cpp[:cpp.index("int main(){using namespace v3;Gr00tDiTModel model;")] + r'''
int main(){using namespace v3;
 constexpr auto forward=FmbForwardOperandResidency::FORWARD, per_layer=FmbForwardOperandResidency::PER_LAYER;
 constexpr auto unspecified=FmbForwardOperandResidency::UNSPECIFIED;
 auto has_weight=[](const auto&decl){return std::any_of(decl.begin(),decl.end(),[](const auto&d){return std::string(d.name)=="ff1_resident_w8";});};
 for(auto profile:{std::pair{4,82},std::pair{10,130},std::pair{10,169},std::pair{10,214},std::pair{4,193},std::pair{4,225},std::pair{4,289},std::pair{4,238},std::pair{4,270},std::pair{4,334},std::pair{10,334}})for(bool raw:{false,true}){
  Gr00tDiTModel model;model.denoise_mode_=model.loop_mode_=true;model.num_steps_=profile.first;model.vl_seq_=profile.second;
  model.ff1_resident_w8_opt_in_=true;
  LayoutContext layout;layout.chunk_size=41;layout.batch_size=1;layout.is_causal=false;
  layout.attention_policy=raw?AttentionExecutionPolicy::SPM_KV_BY_MHA:AttentionExecutionPolicy::DDR_KV;layout.physical_manifest_fingerprint=123;
  auto trial=layout;trial.forward_operand_residency=forward;
  const auto selected=model.declare_buffers(trial), baseline=model.declare_buffers(layout);
  const auto peak=detail::estimate_temporary_total(selected), before=detail::estimate_temporary_total(baseline);
  const auto fixed=estimate_fixed_overhead(selected);
  assert(has_weight(selected)&&!has_weight(baseline)&&peak-before==1179648);
  assert(fixed.persistent>0&&fixed.persistent==estimate_fixed_overhead(baseline).persistent);
  std::vector<SpmAllocator::AllocRequest>requests;size_t resident=0;
  for(const auto&d:selected)if(d.storage==StorageClass::Temp){
   if(std::string(d.name)=="ff1_resident_w8"){resident=requests.size();assert(d.size==1179648&&d.phase_start==0&&(d.phase_end==0||d.phase_end==8)&&d.scope==BufferScope::LayerWide);}
   requests.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});
  }
  const auto plan=SpmAllocator::plan_temporary_aliased(requests);
  for(size_t i=0;i<requests.size();++i)if(i!=resident)assert(plan.offsets[i]+requests[i].bytes<=plan.offsets[resident]||plan.offsets[resident]+1179648<=plan.offsets[i]);
  // Retain production Persistent declarations. External subsystem occupancy
  // is the existing v6/v16 recorded whole-pipeline total minus this owner.
  const int64_t recorded_persistent=profile.first==4?1586688:2323968;
  assert(recorded_persistent>=fixed.persistent);
  constexpr int64_t ceiling=8303616,reserve=65536;
  const int64_t held=recorded_persistent-fixed.persistent;
  SpmAllocator::free_bytes=ceiling-held;SpmAllocator::reads=0;
  // This is also the real FMB manifest-domain filter: large raw banks must
  // leave a feasible original DDR candidate, not publish an oversized route.
  const bool baseline_fits=model.declared_spm_layout_fits(baseline);
  assert(baseline_fits==(!raw||(profile.second!=289&&profile.second!=334)));
  const auto actual=model.ff1_residency_for_candidate(layout);assert(SpmAllocator::reads>0);
  assert(actual==(peak+recorded_persistent+reserve<=ceiling?forward:per_layer));
  auto capacity=trial;capacity.chunk_size=48;
  const auto capacity_decl=model.declare_buffers(capacity);
  const auto capacity_peak=detail::estimate_temporary_total(capacity_decl);
  assert(has_weight(capacity_decl));
  auto with_capacity=layout;with_capacity.planning_chunk_capacity=48;
  const auto padded=model.ff1_residency_for_candidate(with_capacity);
  assert(padded==((std::max(peak,capacity_peak)+recorded_persistent+reserve<=ceiling)?forward:per_layer));
  std::cout<<"cold-fit,"<<profile.first<<","<<profile.second<<","<<raw<<","<<fixed.persistent<<","<<peak<<","<<capacity_peak<<","<<int(actual)<<","<<int(padded)<<"\n";
  // Available-byte boundary is exercised independently of the static ceiling.
  if(peak+fixed.persistent+reserve<=ceiling){
   SpmAllocator::free_bytes=peak+fixed.persistent+reserve;
   assert(model.ff1_residency_for_candidate(layout)==forward);
   --SpmAllocator::free_bytes;assert(model.ff1_residency_for_candidate(layout)==per_layer);
  }
  // Once bound, both decisions are immutable even in Graph with no free SPM.
  SpmAllocator::free_bytes=0;RpuKernelGraph::active=true;
  for(auto selection:{forward,per_layer}){
   auto bound=layout;bound.forward_operand_residency=selection;const int reads=SpmAllocator::reads;
   assert(model.ff1_residency_for_candidate(bound)==selection);
   assert(has_weight(model.declare_buffers(bound))==(selection==forward));
   assert(SpmAllocator::reads==reads);
  }
  bool rejected=false;try{model.ff1_residency_for_candidate(layout);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
  RpuKernelGraph::active=false;
  model.ff1_resident_w8_opt_in_=false;rejected=false;
  try{model.declare_buffers(trial);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
  rejected=false;try{model.ff1_residency_for_candidate(trial);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
  const int reads=SpmAllocator::reads;assert(model.ff1_residency_for_candidate(layout)==per_layer&&SpmAllocator::reads==reads);
 }
}
'''
    _run(tmp_path, cpp)
