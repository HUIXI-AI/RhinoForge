"""Exercise GR00T's actual XOR3 residual ring cold policy and full FFN continuation."""
from pathlib import Path
import re
import shutil
import subprocess

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/fused/rpu_gr00t_dit_model.cpp"


def test_ring_xor3_model_admission_descriptor_and_lifecycle(tmp_path):
    source = MODEL.read_text()
    methods = "\n".join(_definition(source, marker) for marker in (
        "void check_ring_xor3_setter_state(", "void install_ring_xor3(",
        "bool ring_xor3_profile(", "std::vector<int64_t> ring_xor3_arguments(",
        "bool emit_attn_ring_xor3(", "bool emit_ff_ring_xor3(", "void finish_attn_and_ffn(",
    ))
    constants = "\n".join(re.findall(
        r"constexpr int64_t GR00T_DIT_(?:ATTN_RING_XOR3|FF_RING_XOR3|ATTN_ALL_REDUCE|O_LINEAR|FF1_LINEAR|FF1_RESIDENT_LINEAR|FF2_LINEAR|FF_ALL_REDUCE)_SITE = .*?;", source))
    manifest = _definition(source, "FmbPhysicalExecutionManifest physical_manifest_for_attention_routes(")
    start = manifest.index("        if (ring_xor3_profile(")
    registration = manifest[start:manifest.index("        append_linear(GR00T_DIT_FF1_LINEAR_SITE)", start)]
    start = manifest.index("        if (ring_xor3_profile(", manifest.index("        append_linear(GR00T_DIT_FF2_LINEAR_SITE)"))
    registration += manifest[start:manifest.index("        if (denoise_ready_)", start)]
    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(c, ...) do{if(!(c))throw std::runtime_error(#c);}while(0)
#define NUM_CORES 8
template<class F>void rejects(F f){try{f();assert(false);}catch(const std::runtime_error&){}}
namespace at {
enum Type{kChar,kHalf,kByte};constexpr int kPrivateUse1=1;
struct Device{int value=1;int type()const{return value;}};
struct Tensor{int id=0;bool live=true,contiguous=true;Type dtype=kChar;Device dev;std::vector<int64_t>shape{1536,1536};
 bool defined()const{return live;}bool is_contiguous()const{return contiguous;}Type scalar_type()const{return dtype;}
 Device device()const{return dev;}int64_t dim()const{return shape.size();}int64_t size(int n)const{return shape.at(n);}};
}
__CONSTANTS__
__READER__
enum class KernelId{GR00T_ALL_REDUCE_RESIDUAL_XOR3_M41N1536};
struct KernelCache{inline static bool loaded=true;inline static int reads=0;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId){++reads;return loaded;}};
struct RpuKernelGraph{inline static bool active=false;static bool has_active(){return active;}};
enum class FmbRouteFamily{LINEAR,ALL_REDUCE};enum class FmbLinearRouteSelector{AUTO_TILE=1};
enum class ValuOpType{ADD};enum class GeluMode{TANH};
struct ChunkInfo{int64_t len=41,idx=0;};struct LayoutContext{int64_t batch_size=1;};
struct Route{FmbRouteFamily family;int64_t site,selector,flags,invocation;std::vector<int64_t>args;};
std::vector<std::string> events;
struct Context{
 bool complete=true,output_to_spm=true;int64_t batch_size=1;std::vector<Route>routes;
 bool has_complete_physical_manifest()const{return complete;}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,const std::vector<int64_t>&args,int64_t invocation){
  if(!complete)return;
  TORCH_CHECK(std::all_of(args.begin(),args.end(),[](int64_t a){return a>=0;}));
  auto found=std::find_if(routes.begin(),routes.end(),[&](const Route&r){return r.family==family&&r.site==site&&r.invocation==invocation;});
  TORCH_CHECK(found!=routes.end()&&found->selector==selector&&found->flags==flags&&found->args==args);
  if(site==GR00T_DIT_ATTN_RING_XOR3_SITE)events.push_back("consume-attn");
  if(site==GR00T_DIT_FF_RING_XOR3_SITE)events.push_back("consume-ff");
 }
};
int64_t fmb_ring_all_reduce_route_selector(int64_t rows,int64_t cols,int=8){return ((rows*cols+7)/8+15)/16*16*2<11*1024?1:2;}
constexpr uint32_t O=0x10000000,R1=0x10020000,R2=0x10040000,NORM=0x10060000;
constexpr uint32_t GAMMA=0x10080000,BETA=0x10081000,SDPA=0x10082000,MID=0x10090000;
int fused=0,ordinary_ring=0,ordinary_ln=0,expected_layer=0,expected_rows=41,producer=0;
int last_raw=0,last_norm=0;std::map<uint32_t,int>spm;
void rpu_launch_gr00t_all_reduce_residual_xor3_kernel(uint32_t in,uint32_t residual,uint32_t out){
 assert(in==O&&!events.empty());
 if(residual==R1){assert(out==R2&&events.back()=="consume-attn");spm[out]=spm[in]+spm[residual];last_raw=spm[out];events.push_back("xor3-attn");}
 else{assert(residual==R2&&out==R1&&events.back()=="consume-ff"&&spm[R2]==last_raw);spm[out]=spm[in]+spm[residual];events.push_back("xor3-ff");}++fused;
}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t in,uint32_t residual,uint32_t out,int64_t rows,int64_t cols,int a,int b){
 assert(in==O&&rows==expected_rows&&cols==1536&&a==8&&b==8);
 if(residual==R1){assert(out==R2);spm[out]=spm[in]+spm[residual];last_raw=spm[out];events.push_back("ring");}
 else{assert(residual==R2&&out==R1&&spm[R2]==last_raw);spm[out]=spm[in]+spm[residual];events.push_back("ff-ring");}++ordinary_ring;
}
void rpu_launch_layernorm_spm_kernel(uint32_t in,uint32_t out,uint32_t gamma,uint32_t beta,int64_t rows,int64_t cols,double eps,bool skip,uint32_t skip_addr,int cores){
 assert(in==R2&&out==NORM&&gamma==GAMMA&&beta==BETA&&rows==expected_rows&&cols==1536&&eps==1e-5&&!skip&&skip_addr==0&&cores==8);
 assert(spm[in]==last_raw);last_norm=last_raw+99;spm[out]=last_norm;events.push_back("ln");++ordinary_ln;
}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,const at::Tensor&w,uint32_t out,int64_t rows,int64_t n,int64_t k,int part,int cores,uint32_t bias,const at::Tensor&scale){
 assert(rows==expected_rows&&cores==8);const int id=expected_layer*10;
 if(w.id==1000+id){assert(in==SDPA&&out==O&&n==1536&&k==1536&&part==0&&bias==0x10600000+expected_layer*1536&&scale.id==3000+id);spm[out]=++producer;events.push_back("o");}
 else if(w.id==1001+id){assert(in==NORM&&out==MID&&n==6144&&k==1536&&part==1&&bias==0x10600000+expected_layer*1536+512&&scale.id==3001+id);assert(spm[in]==last_norm);spm[out]=spm[in]+3;events.push_back("ff1");}
 else {assert(w.id==1002+id&&in==MID&&out==O&&n==1536&&k==6144&&part==0&&bias==0x10600000+expected_layer*1536+1024&&scale.id==3002+id);spm[out]=spm[in]+5;events.push_back("ff2");}
}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t in,uint32_t out,int64_t elements,ValuOpType op,GeluMode gelu,int cores){assert(in==MID&&out==MID&&elements==expected_rows*768&&op==ValuOpType::ADD&&gelu==GeluMode::TANH&&cores==8);spm[out]+=7;events.push_back("gelu");}
void rpu_launch_gr00t_ff1_resident_w8_kernel(uint32_t,uint32_t,uint32_t,uint32_t,const at::Tensor&){assert(false);}
struct Model{
 __LAYER__
 std::vector<LayerWeights>layer_weights_=std::vector<LayerWeights>(32);
 bool ring_xor3_opt_in_=gr00t_ring_xor3_opt_in(),ring_xor3_ready_=false;
 bool inference_started_=false,z2_bound_=false,denoise_ready_=true,denoise_mode_=true,loop_mode_=true;
 int64_t resolved=0,z2_prepared_seq_len_=0,num_steps_=4,action_horizon_=40,vl_seq_=82,cross_dim_=2048;
 int64_t h=1536,ff=6144,q=32,kv=32,hd=48,tp=8,layers=32;double eps_=1e-5;Context context;
 Model(){for(int layer=0;layer<32;++layer){auto&x=layer_weights_[layer];x.is_cross=layer%2==0;
  int projection=0;for(auto*w:{&x.to_out_w,&x.ff1_w,&x.ff2_w})w->id=1000+layer*10+projection++;
  x.ff2_w.shape={1536,6144};
  for(auto*b:{&x.to_out_b,&x.ff2_b}){b->dtype=at::kHalf;b->shape={1536};}
  projection=0;for(auto*s:{&x.to_out_ws,&x.ff1_ws,&x.ff2_ws}){s->id=3000+layer*10+projection++;s->dtype=at::kHalf;s->shape={1536};}}
 }
 int64_t get_last_resolved_chunk_size()const{return resolved;}int64_t num_layers()const{return layers;}
 int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return ff;}
 int64_t num_q_heads()const{return q;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}int64_t attn_tp()const{return tp;}int num_cores()const{return NUM_CORES;}
 Context&ctx(){return context;}
 uint32_t addr(int,const char*n){std::string s(n);if(s=="oproj")return O;if(s=="residual1")return R1;if(s=="residual2")return R2;if(s=="norm_hidden")return NORM;if(s=="ln_ones")return GAMMA;if(s=="ln_zeros")return BETA;if(s=="sdpa_out")return SDPA;assert(s=="ff_mid");return MID;}
 uint32_t layer_addr(int layer,int,const char*n){std::string s(n);int offset=s=="out_bias"?0:s=="ff1_bias"?512:1024;return 0x10600000+layer*1536+offset;}
 bool ff1_resident_selected()const{return false;}
 std::vector<int64_t>ff1_resident_arguments(int64_t,int64_t)const{return {};}
 void emit_layer_output_dma(int,const ChunkInfo&){assert(false);}
 void add_row_bias_partial(uint32_t,int64_t,int64_t){assert(false);}
 __METHODS__
 void record(const ChunkInfo&chunk){LayoutContext layout{context.batch_size};context.routes.clear();
  auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,std::vector<int64_t>args=std::vector<int64_t>{}){
   assert(std::all_of(args.begin(),args.end(),[](int64_t a){return a>=0;}));context.routes.push_back({family,site,selector,0,chunk.idx,std::move(args)});};
  __REGISTER__
  for(auto site:{GR00T_DIT_O_LINEAR_SITE,GR00T_DIT_FF1_LINEAR_SITE,GR00T_DIT_FF2_LINEAR_SITE})append(FmbRouteFamily::LINEAR,site,1);
 }
 void finish(int layer,const ChunkInfo&chunk){expected_layer=layer;expected_rows=chunk.len;events.clear();spm[R1]=layer+100;finish_attn_and_ffn(layer,chunk,layer_weights_[layer],chunk.len,1536,6144,768);}
};
int main(){
 unsetenv("RPU_GR00T_RING_XOR3");setenv("RPU_GR00T_SELF_QKV_MULTIBASE","1",1);assert(!gr00t_ring_xor3_opt_in());
 for(auto value:{"","0"}){setenv("RPU_GR00T_RING_XOR3",value,1);assert(!gr00t_ring_xor3_opt_in());}
 for(auto value:{"true","2","-1","01"," 1","1 "}){setenv("RPU_GR00T_RING_XOR3",value,1);rejects([]{gr00t_ring_xor3_opt_in();});}
 setenv("RPU_GR00T_RING_XOR3","0",1);Model off;KernelCache::loaded=false;int reads=KernelCache::reads;off.install_ring_xor3();assert(!off.ring_xor3_ready_&&KernelCache::reads==reads);
 ChunkInfo chunk;off.record(chunk);assert(off.context.routes[0].site==GR00T_DIT_ATTN_ALL_REDUCE_SITE&&off.context.routes[0].selector==2);
 off.finish(0,chunk);assert((events==std::vector<std::string>{"o","ring","ln","ff1","gelu","ff2","ff-ring"}));assert(fused==0&&ordinary_ring==2&&ordinary_ln==1);
 setenv("RPU_GR00T_RING_XOR3","1",1);Model m;setenv("RPU_GR00T_RING_XOR3","0",1);assert(m.ring_xor3_opt_in_);
 rejects([&]{m.install_ring_xor3();});assert(!m.ring_xor3_ready_);KernelCache::loaded=true;m.install_ring_xor3();assert(m.ring_xor3_ready_);m.record(chunk);
 auto&route=m.context.routes[0];assert(route.site==GR00T_DIT_ATTN_RING_XOR3_SITE&&route.selector==2);
 // Each independent site binds role, full FP16 geometry, XOR3 gather and 12 epochs.
 const std::vector<int64_t>geometry{16,2,7872,15744,128,125952,1,63488,8,1,1,3,12,1,1,1,1,0,128};
 assert(std::vector<int64_t>(route.args.begin()+19,route.args.end())==geometry);
 assert(route.args[1]==0&&m.context.routes[1].site==GR00T_DIT_FF_RING_XOR3_SITE&&m.context.routes[1].args[1]==1);
 assert(std::vector<int64_t>(m.context.routes[1].args.begin()+19,m.context.routes[1].args.end())==geometry);
 reads=KernelCache::reads;KernelCache::loaded=false;
 for(int request=0;request<3;++request)for(int body=0;body<4;++body)for(int layer=0;layer<32;++layer){
  m.finish(layer,chunk);assert((events==std::vector<std::string>{"o","consume-attn","xor3-attn","ln","ff1","gelu","ff2","consume-ff","xor3-ff"}));
  assert(spm[R1]==2*last_raw+99+3+7+5&&spm[R2]==last_raw&&spm[NORM]==last_norm);
 }
 assert(fused==3*4*32*2&&ordinary_ring==2&&ordinary_ln==1+3*4*32&&KernelCache::reads==reads);
 for(auto admitted:{std::pair{4,1},std::pair{4,256},std::pair{4,193},std::pair{4,225},std::pair{4,289},std::pair{4,238},std::pair{4,270},std::pair{4,334},std::pair{10,334},std::pair{10,1},std::pair{10,97},std::pair{10,169},std::pair{10,214},std::pair{10,256}}){
 const int64_t steps=admitted.first,seq=admitted.second;
 Model ten=m;ten.num_steps_=steps;ten.vl_seq_=seq;assert(ten.ring_xor3_profile(41,1));
 for(bool ff:{false,true}){const int calls=fused;events.clear();rejects([&]{if(ff)ten.emit_ff_ring_xor3(chunk);else ten.emit_attn_ring_xor3(chunk);});assert(fused==calls&&events.empty());}
 ten.record(chunk);for(const auto&r:ten.context.routes)if(r.family==FmbRouteFamily::ALL_REDUCE){assert(r.args[4]==seq&&r.args[5]==steps&&r.args.back()==ten.num_steps_*ten.num_layers());}
 for(int request=0;request<3;++request)for(int body=0;body<ten.num_steps_;++body)for(int layer=0;layer<32;++layer){
  const int calls=fused;ten.finish(layer,chunk);assert(fused==calls+2);
  assert((events==std::vector<std::string>{"o","consume-attn","xor3-attn","ln","ff1","gelu","ff2","consume-ff","xor3-ff"}));
  assert(spm[R1]==2*last_raw+99+3+7+5&&spm[R2]==last_raw&&spm[NORM]==last_norm);}
 Model back=ten;back.num_steps_=4;back.vl_seq_=82;
 for(bool ff:{false,true}){const int calls=fused;events.clear();rejects([&]{if(ff)back.emit_ff_ring_xor3(chunk);else back.emit_attn_ring_xor3(chunk);});assert(fused==calls&&events.empty());}
 Model stale=ten;stale.vl_seq_=seq==169?214:169;
 for(bool ff:{false,true}){const int calls=fused;events.clear();rejects([&]{if(ff)stale.emit_ff_ring_xor3(chunk);else stale.emit_attn_ring_xor3(chunk);});assert(fused==calls&&events.empty());}
 for(auto profile:{std::pair{4,0},std::pair{4,-1},std::pair{10,0},std::pair{10,-1},std::pair{3,214},std::pair{5,214}}){
  Model x=ten;x.num_steps_=profile.first;x.vl_seq_=profile.second;
  assert(!x.ring_xor3_profile(41,1));x.record(chunk);assert(x.context.routes[0].site==GR00T_DIT_ATTN_ALL_REDUCE_SITE);
  const int calls=fused;x.finish(1,chunk);assert(fused==calls&&events[1]=="ring"&&events[2]=="ln");}
 }

 // Uniform FP16 uses the same full-size SPM ring and unchanged ACC16 producers.
 KernelCache::loaded=true;Model half=m;
 for(auto&layer:half.layer_weights_){layer.to_out_w.dtype=layer.ff2_w.dtype=at::kHalf;
  layer.to_out_ws.live=layer.ff2_ws.live=false;}
 half.install_ring_xor3();assert(half.ring_xor3_ready_);
 for(int seq:{193,225,289,238,270,334}){half.vl_seq_=seq;half.record(chunk);
  for(int request=0;request<3;++request)for(int body=0;body<4;++body)for(int layer=0;layer<32;++layer){
   const int before=fused;half.finish(layer,chunk);assert(fused==before+2);}}
 for(int bad=0;bad<4;++bad){Model x=half;switch(bad){
  case 0:x.layer_weights_[7].ff2_w.dtype=at::kChar;break;
  case 1:x.layer_weights_[9].to_out_ws.live=true;break;
  case 2:x.layer_weights_[11].to_out_b.live=false;break;
  case 3:x.layer_weights_[13].ff2_w.shape={1536,1536};break;}
  x.install_ring_xor3();assert(!x.ring_xor3_ready_);}
 KernelCache::loaded=false;rejects([&]{half.install_ring_xor3();});assert(!half.ring_xor3_ready_);KernelCache::loaded=true;
 // Reject every public argument and both site identities before either associated launch.
 const size_t nargs=m.context.routes[0].args.size();
 for(int site=0;site<2;++site)for(size_t i=0;i<nargs+5;++i){auto old=m.context.routes;auto&r=m.context.routes[site];
  if(i<nargs)++r.args[i];else switch(i-nargs){case 0:r.flags=1;break;case 1:r.selector=1;break;case 2:++r.invocation;break;case 3:r.family=FmbRouteFamily::LINEAR;break;case 4:r.site=site?GR00T_DIT_FF_ALL_REDUCE_SITE:GR00T_DIT_ATTN_ALL_REDUCE_SITE;break;}
  const int calls=fused;events.clear();rejects([&]{if(site)m.emit_ff_ring_xor3(chunk);else m.emit_attn_ring_xor3(chunk);});assert(fused==calls&&events.empty());m.context.routes=old;
 }
 // Mode and profile changes mint the original route; stale fused manifests must fail closed.
 for(int drift=0;drift<20;++drift){Model x=m;ChunkInfo c=chunk;switch(drift){case 0:x.loop_mode_=false;break;case 1:x.denoise_mode_=false;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:x.vl_seq_=0;break;case 5:c.len=40;break;case 6:x.context.batch_size=2;break;case 7:x.num_steps_=3;break;case 8:x.tp=4;break;case 9:x.ff=4096;break;case 10:x.ring_xor3_opt_in_=false;break;case 11:x.ring_xor3_ready_=false;break;case 12:x.denoise_ready_=false;break;case 13:x.action_horizon_=39;break;case 14:x.hd=64;break;case 15:x.layers=31;break;case 16:x.h=1024;break;case 17:x.q=16;break;case 18:x.kv=16;break;case 19:x.cross_dim_=1536;break;}
  assert(!x.ring_xor3_profile(c.len,x.context.batch_size));const int calls=fused;
  if(drift<14||drift==15){rejects([&]{x.finish(1,c);});assert(fused==calls);}
  else assert(!x.emit_attn_ring_xor3(c)&&!x.emit_ff_ring_xor3(c));
  x.record(c);assert(x.context.routes[0].site==GR00T_DIT_ATTN_ALL_REDUCE_SITE);
  if(drift<14||drift==15){x.finish(1,c);assert(fused==calls&&events[1]=="ring"&&events[2]=="ln");}
 }
 m.context.complete=false;int old=fused;m.finish(1,chunk);assert(fused==old&&events[1]=="ring"&&events[2]=="ln");m.context.complete=true;
 for(int bad=0;bad<5;++bad){Model x=m;switch(bad){case 0:x.inference_started_=true;break;case 1:x.resolved=41;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:RpuKernelGraph::active=true;break;}
  rejects([&]{x.check_ring_xor3_setter_state();});rejects([&]{x.install_ring_xor3();});assert(x.ring_xor3_ready_);RpuKernelGraph::active=false;}
 // Cold admission checks both self/cross layers and both original row producers.
 for(int layer:{0,1,30,31})for(int operand=0;operand<6;++operand)for(int bad=0;bad<6;++bad){Model x=m;auto&l=x.layer_weights_[layer];std::array<at::Tensor*,6>t{&l.to_out_w,&l.ff2_w,&l.to_out_b,&l.to_out_ws,&l.ff2_b,&l.ff2_ws};
  switch(bad){case 0:t[operand]->live=false;break;case 1:t[operand]->dtype=operand<2?at::kHalf:at::kChar;break;case 2:t[operand]->dev.value=0;break;case 3:t[operand]->contiguous=false;break;case 4:t[operand]->shape[0]=1535;break;case 5:t[operand]->shape={};break;}
  reads=KernelCache::reads;x.install_ring_xor3();assert(!x.ring_xor3_ready_&&KernelCache::reads==reads);x.record(chunk);assert(x.context.routes[0].site==GR00T_DIT_ATTN_ALL_REDUCE_SITE);
 }
 Model missing_layer=m;missing_layer.layer_weights_.pop_back();missing_layer.install_ring_xor3();assert(!missing_layer.ring_xor3_ready_);
 // Replacing cold sources with valid operands reinstalls from the current payload identity.
 KernelCache::loaded=true;m.install_ring_xor3();assert(m.ring_xor3_ready_);
}
'''
    cpp = (cpp.replace("__CONSTANTS__", constants)
           .replace("__READER__", _definition(source, "bool gr00t_ring_xor3_opt_in("))
           .replace("__LAYER__", _definition(source, "struct LayerWeights") + ";")
           .replace("__METHODS__", methods).replace("__REGISTER__", registration))
    path, exe = tmp_path / "ring_model.cpp", tmp_path / "ring_model"
    path.write_text(cpp)
    result = subprocess.run([shutil.which("g++"), "-std=c++17", "-O1", str(path), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_ring_xor3_preserves_projection_norm_ffn_and_spm_contract():
    source = MODEL.read_text()
    setters = source.replace("= {}", "= at::Tensor()")
    for name in ("set_weights", "set_denoise_weights"):
        setter = _definition(setters, f"void {name}(")
        assert setter.index("check_ring_xor3_setter_state()") < setter.index("TORCH_CHECK(")
    install = _definition(setters, "void set_weights(")
    assert install.index("ring_xor3_ready_ = false") < install.index("install_ring_xor3()") < install.index("invalidate_model_state()")
    for name in ("emit_attn_ring_xor3", "emit_ff_ring_xor3"):
        emitter = _definition(source, f"bool {name}(")
        assert emitter.index("consume_physical_route(") < emitter.index("rpu_launch_gr00t_all_reduce_residual_xor3_kernel(")
    finish = _definition(source, "void finish_attn_and_ffn(")
    # Original O arithmetic, LayerNorm and both FF1 routes remain with full residual2.
    assert finish.count("rpu_launch_linear_spm_to_spm_acc16_kernel(") == 3
    assert finish.index("lw.to_out_ws") < finish.index("emit_attn_ring_xor3(") < finish.index("ff1_resident_selected(")
    assert 'addr(0, "oproj"), addr(0, "residual2"), addr(0, "residual1")' in finish
    assert 'addr(0, "norm_hidden"), lw.ff1_w' in finish
    assert 'seq_len, h, 1e-5, false, 0, num_cores()' in finish
    assert "ring_xor3" not in _definition(source, "std::vector<BufferDecl> baseline_buffer_declarations(")
    assert "ring_xor3" not in _definition(source, "std::vector<BufferDecl> declare_buffers(")
    layout = _definition(source, "int64_t subclass_layout_hash(")
    owner = _definition(source, "std::vector<int64_t> kvinsert_cost_weight_identity(")
    for identity in (layout, owner):
        assert "GR00T_DIT_ATTN_RING_XOR3_SITE" in identity and "GR00T_DIT_FF_RING_XOR3_SITE" in identity and "ring_xor3_ready_" in identity
    assert "&weights.to_out_w" in owner and "&weights.to_out_ws" in owner and "&weights.to_out_b" in owner
    assert "&weights.ff2_w" in owner and "&weights.ff2_ws" in owner and "&weights.ff2_b" in owner
    assert finish.index("lw.ff2_ws") < finish.index("emit_ff_ring_xor3(")
    assert "const bool ring_xor3_opt_in_;" in source
