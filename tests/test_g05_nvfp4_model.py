"""Exercise actual G05 mixed NVFP4 dispatch and persistent scale-table boundaries."""
from pathlib import Path
from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body

ROOT = Path(__file__).resolve().parents[1]


def test_nvfp4_family_layer_routes_and_scale_table_rejection(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    tables = (ROOT / 'src/fused/g05_nvfp4.h').read_text()
    methods = [
        ('void set_linear_acc32(bool enabled)', 'void Qwen3_5Model::set_linear_acc32('),
        ('bool use_action_nvfp4(int64_t rows)const', 'bool Qwen3_5Model::use_action_nvfp4('),
        ('std::vector<int64_t> action_nvfp4_arguments(int family)const', 'std::vector<int64_t> Qwen3_5Model::action_nvfp4_arguments('),
        ('bool use_action_mod_body()const', 'bool Qwen3_5Model::use_action_mod_body('),
        ('void launch_linear(int64_t chunk_idx,uint32_t input,const at::Tensor& weight,uint32_t output,int64_t m,int64_t n,int64_t k,int partition,int num_cores,uint32_t bias_spm_addr,const at::Tensor& scale,int nvfp4_family,int layer,const at::Tensor& negative_zero={})', 'void Qwen3_5Model::launch_linear('),
    ]
    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "rpu_linear_tiling.h"
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10 { struct Half {float value;explicit Half(float x):value(x<0.0000000298f?0:x>65504?INFINITY:x){}operator float()const{return value;}}; }
namespace at {
constexpr int kFloat=1,kByte=2,kChar=3,kHalf=4,kPrivateUse1=5;
struct Tensor {
 int id=0,dtype=kByte,rank=1,location=kPrivateUse1;bool contiguous=true;int64_t count=24;
 std::vector<float> data=std::vector<float>(24,1.0f);
 bool defined()const{return id!=0;}int scalar_type()const{return dtype;}bool is_contiguous()const{return contiguous;}
 struct Device {int d;int type()const{return d;}};Device device()const{return {location};}
 int dim()const{return rank;}int64_t numel()const{return count;}Tensor cpu()const{return *this;}
 template<class T>T*data_ptr()const{return reinterpret_cast<T*>(const_cast<float*>(data.data()));}
 bool is_same(const Tensor& x)const{return id==x.id;}
};
using TensorList=std::vector<Tensor>;
}
enum class StorageClass{Persistent};enum class BufferScope{LayerWide};struct FusedModelBase{};
struct BufferDecl {const char*name;int64_t bytes,first,last;StorageClass storage;int zero;const char*alias;BufferScope scope;std::function<void(FusedModelBase&,int,uint32_t)> init;};
int preloads=0;void rpu_launch_ddr_broadcast_spm_dma(c10::Half*,int64_t count,uint32_t destination,int cores){assert(count==48&&destination%256==0&&cores==8);++preloads;}
__TABLES__
// The dense-text NVFP4 owner is uninstalled in this probe (enabled() false, no tables);
// launch_linear only needs its family count and slab names to compile.
struct Qwen35TextNvfp4Tables {enum Family:int{COUNT=13};static constexpr const char*names[COUNT]={};std::vector<at::Tensor>values;bool enabled()const{return false;}};
enum class FmbLinearRouteSelector{AUTO_TILE=1,GEMV=2,G05_NVFP4_V2_ACC16=8,G05_NVFP4_V2_ACC32=12};enum class FmbRouteFamily{LINEAR};
constexpr int64_t QWEN35_ACTION_NVFP4_LINEAR_SITE=1171762392203719440LL,QWEN35_TEXT_LINEAR_SITE=4115846627077743284LL;
struct Weights {at::Tensor q_w,o_w,gate_w,up_w,down_w,q_ws,o_ws,gate_ws,up_ws,down_ws;
 // Inert text-NVFP4 identity owners scanned by launch_linear; id 0 keeps them undefined.
 at::Tensor k_w,v_w,attn_gate_w,gdn_q_w,gdn_k_w,gdn_v_w,gdn_in_z_w,gdn_out_w,k_ws,v_ws,attn_gate_ws,gdn_q_ws,gdn_k_ws,gdn_v_ws,gdn_in_z_ws,gdn_out_ws;};
struct Route {int64_t site,selector,invocation;std::vector<int64_t>args;};
struct Launch {int family,layer;uint32_t table;bool acc32;};std::vector<Launch> launches;int legacy=0;
void rpu_launch_g05_nvfp4_v2_kernel(uint32_t,const at::Tensor&w,uint32_t,int64_t rows,int64_t n,int64_t k,int partition,int cores,const at::Tensor&s,uint32_t ts,uint16_t layer,bool acc32){
 const int family=w.id%10;assert(rows==32&&cores==8&&layer==w.id/10-1&&s.id==w.id+1000);
 assert(n==G05Nvfp4Tables::output_sizes[family]&&k==G05Nvfp4Tables::input_sizes[family]&&partition==G05Nvfp4Tables::partitions[family]);launches.push_back({family,layer,ts,acc32});
}
template<class...T>void rpu_launch_linear_spm_to_spm_acc16_kernel(T...){++legacy;}
struct Model {
 G05Nvfp4Tables action_nvfp4_;Qwen35TextNvfp4Tables text_nvfp4_;bool action_mode_=true,linear_acc32_=false,z2_bound_=false,complete=true;
 bool action_step_active_=true,action_loop_active_=true,action_mod_body_fp16_=false,action_mod_body_w8_=false,has_gdn_=false;
 int64_t action_num_steps_=10,action_len_=32;int cores=8;
 std::vector<Weights>layer_weights_{24};std::vector<Route>routes;int consumed=0;
 int invalidations=0;int64_t last=0;int64_t get_last_resolved_chunk_size()const{return last;}void prepare_text_gateup_pair(){}void prepare_text_owner_rms(){}void invalidate_model_state(){++invalidations;}
 int num_cores()const{return cores;}int attn_tp()const{return cores;}int mlp_tp()const{return cores;}int num_layers()const{return 24;}int hidden_size()const{return 1024;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 uint32_t addr(int core,const char*name)const{assert(core==0);for(int i=0;i<5;++i)if(std::string(name)==G05Nvfp4Tables::names[i])return 0x40010000+i*256;throw std::runtime_error("bad slab");}
 FmbLinearRouteSelector linear_route_selector(const at::Tensor&,int64_t)const{return FmbLinearRouteSelector::AUTO_TILE;}
 int64_t linear_invocation(int64_t chunk,FmbLinearRouteSelector sel)const{return chunk*8+int(sel);}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,const std::vector<int64_t>&args,int64_t invocation){assert(f==FmbRouteFamily::LINEAR&&flags==0);for(auto&r:routes)if(r.site==site&&r.invocation==invocation){TORCH_CHECK(r.selector==selector&&r.args==args);++consumed;return;}throw std::runtime_error("missing manifest");}
 __METHODS__
};
template<class F>void rejects(Model&m,F f){auto before=launches.size();int c=m.consumed,l=legacy;try{f();assert(false);}catch(const std::runtime_error&){}assert(launches.size()==before&&m.consumed==c&&legacy==l);}
int main(){
 Model m;std::vector<at::Tensor>ts(5);for(int i=0;i<5;++i){ts[i].id=500+i;ts[i].dtype=at::kFloat;}m.action_nvfp4_.install(ts);
 // Installed NV payload availability and arithmetic mode are immutable even before action enable.
 m.action_mode_=false;rejects(m,[&]{m.set_linear_acc32(true);});assert(!m.linear_acc32_&&m.invalidations==0);
 m.set_linear_acc32(false);assert(m.invalidations==1);m.action_mode_=true;
 assert(m.use_action_mod_body()&&!m.action_mod_body_w8_);m.action_num_steps_=1;assert(!m.use_action_mod_body());m.action_num_steps_=10;
 std::vector<BufferDecl>decls;m.action_nvfp4_.declare(decls,8);assert(decls.size()==5);FusedModelBase base;for(auto&d:decls){assert(d.bytes==256&&d.storage==StorageClass::Persistent);d.init(base,0,0x40010000);}assert(preloads==5);
 for(int layer=0;layer<24;++layer){auto&w=m.layer_weights_[layer];std::array<at::Tensor*,5>weights{&w.q_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w},scales{&w.q_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws};for(int f=0;f<5;++f){weights[f]->id=(layer+1)*10+f;scales[f]->id=weights[f]->id+1000;}}
 for(int f=0;f<5;++f)m.routes.push_back({QWEN35_ACTION_NVFP4_LINEAR_SITE,8,10+f,m.action_nvfp4_arguments(f)});
 auto run=[&](int family,int layer,int64_t rows=32){auto&w=m.layer_weights_[layer>=0&&layer<24?layer:0];std::array<at::Tensor*,5>weights{&w.q_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w},scales{&w.q_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws};int safe=family>=0&&family<5?family:0;m.launch_linear(2,0x40020000,*weights[safe],0x40030000,rows,G05Nvfp4Tables::output_sizes[safe],G05Nvfp4Tables::input_sizes[safe],G05Nvfp4Tables::partitions[safe],8,0,*scales[safe],family,layer);};
 for(int layer=0;layer<24;++layer)for(int f=0;f<5;++f)run(f,layer);
 assert(launches.size()==120&&m.consumed==120&&legacy==0);for(auto&l:launches)assert(l.table==0x40010000+uint32_t(l.family)*256);
 for(int f=0;f<5;++f){for(size_t arg=0;arg<m.routes[f].args.size();++arg){++m.routes[f].args[arg];rejects(m,[&]{run(f,0);});--m.routes[f].args[arg];}}
 rejects(m,[&]{run(5,0);});rejects(m,[&]{run(0,-1);});rejects(m,[&]{run(0,24);});rejects(m,[&]{run(0,0,31);});
 m.complete=false;rejects(m,[&]{run(0,0);});m.complete=true;m.linear_acc32_=true;rejects(m,[&]{run(0,0);});m.linear_acc32_=false;m.action_mode_=false;rejects(m,[&]{run(0,0);});m.action_mode_=true;
 auto wrong=m.layer_weights_[0].q_w;wrong.id=999;rejects(m,[&]{m.launch_linear(2,0,wrong,0,32,2048,1024,1,8,0,m.layer_weights_[0].q_ws,0,0);});
 // The launch lane count must be the model's own cold budget.
 rejects(m,[&]{m.launch_linear(2,0,m.layer_weights_[0].q_w,0,32,2048,1024,1,4,0,m.layer_weights_[0].q_ws,0,0);});
 auto before=m.action_nvfp4_.values;for(int fault=0;fault<10;++fault){auto bad=ts;if(fault==0)bad.pop_back();if(fault==1)bad[0].dtype=at::kHalf;if(fault==2)bad[0].location=0;if(fault==3)bad[0].contiguous=false;if(fault==4)bad[0].count=18;if(fault==5)bad[0].rank=2;if(fault==6)bad[0].data[23]=NAN;if(fault==7)bad[0].data[23]=0;if(fault==8)bad[0].data[23]=1e-30f;if(fault==9)bad[0].data[23]=1e9f;rejects(m,[&]{m.action_nvfp4_.install(bad);});assert(m.action_nvfp4_.values[0].id==before[0].id);}
 // Replanning ACC32 selects its actual mode and Pi05 tile, never the ACC16 route.
 m.linear_acc32_=true;m.routes.clear();
 for(int f=0;f<5;++f){auto args=m.action_nvfp4_arguments(f);assert(args[9]==32);assert(args[16]==(f==2||f==3?384:320)&&args[17]==(f==2||f==3?48:64));m.routes.push_back({QWEN35_ACTION_NVFP4_LINEAR_SITE,12,10+f,args});}
 const auto base_count=launches.size();for(int layer=0;layer<24;++layer)for(int f=0;f<5;++f)run(f,layer);
 assert(launches.size()==base_count+120);for(size_t i=base_count;i<launches.size();++i)assert(launches[i].acc32);
 m.routes[0].selector=8;rejects(m,[&]{run(0,0);});m.routes[0].selector=12;
 m.routes[0].args[9]=16;rejects(m,[&]{run(0,0);});m.routes[0].args[9]=32;
 m.linear_acc32_=false;rejects(m,[&]{run(0,0);});
 // Old W8 callers retain the ordinary route and do not require tensor tables.
 m.action_nvfp4_.values.clear();m.routes={{QWEN35_TEXT_LINEAR_SITE,1,17,{}}};at::Tensor w,scale;w.dtype=at::kChar;scale.dtype=at::kHalf;m.launch_linear(2,0,w,0,32,2048,1024,1,8,0,scale,0,0);assert(legacy==1);
}
'''
    cpp = cpp.replace('__TABLES__', 'struct G05Nvfp4Tables ' + _body(tables, 'struct G05Nvfp4Tables') + ';')
    cpp = cpp.replace('__METHODS__', '\n'.join(signature + _body(source, marker) for signature, marker in methods))
    _compile_run(tmp_path, 'g05_nvfp4_model', cpp)
