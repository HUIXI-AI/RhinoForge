"""Run GR00T's actual cold self-QKV admission, descriptor and emitter on host."""
from pathlib import Path
from pi_wall_host_catalog import with_actual_linear_contracts
import re
import shutil
import subprocess

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/fused/rpu_gr00t_dit_model.cpp"


def test_qkv_multibase_native_model_admission_descriptor_and_operands(tmp_path):
    source = MODEL.read_text()
    methods = "\n".join(_definition(source, marker) for marker in (
        "void check_self_qkv_setter_state(", "void install_self_qkv_multibase(",
        "bool self_qkv_multibase_profile(", "std::vector<int64_t> self_qkv_multibase_arguments(",
        "bool emit_self_qkv_multibase(",
    ))
    constants = "\n".join(re.findall(
        r"constexpr int64_t GR00T_DIT_SELF_(?:QKV_MULTIBASE|Q_LINEAR|K_LINEAR|V_LINEAR)_SITE = .*?;", source))
    self_block = _definition(source, "void build_self_attn_block(")
    projections = self_block[self_block.index("        // QKV (col-partition)"):self_block.index("        // KV-insert(pos 0)")]
    manifest = _definition(source, "FmbPhysicalExecutionManifest physical_manifest_for_attention_routes(")
    begin = manifest.index("            if (self_qkv_multibase_profile(")
    registration = manifest[begin:manifest.index("            append_kv(GR00T_DIT_SELF_KV_INSERT_SITE", begin)]
    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do{if(!(c))throw std::runtime_error(#c);}while(0)
#define NUM_CORES 8
template<class F>void rejects(F f){try{f();assert(false);}catch(const std::runtime_error&){}}
namespace at {
enum Type{kChar,kHalf};constexpr int kPrivateUse1=1;
struct Device{int value=1;int type()const{return value;}};
struct Tensor{int id=0;bool live=true,contiguous=true;Type dtype=kChar;Device dev;std::vector<int64_t>shape{1536,1536};
 bool defined()const{return live;}bool is_contiguous()const{return contiguous;}Type scalar_type()const{return dtype;}
 Device device()const{return dev;}int64_t dim()const{return shape.size();}int64_t size(int n)const{return shape.at(n);}};
}
__CONSTANTS__
__ENV__
enum class KernelId{GR00T_QKV_MULTIBASE_W8A16_M592N32K128};
struct KernelCache{inline static bool loaded=true;inline static int reads=0;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId){++reads;return loaded;}};
struct RpuKernelGraph{inline static bool active=false;static bool has_active(){return active;}};
enum class FmbRouteFamily{LINEAR};enum class FmbLinearRouteSelector{AUTO_TILE=1};
struct ChunkInfo{int64_t len=41,idx=0;};struct LayoutContext{int64_t batch_size=1;};
struct Route{int64_t site,selector,flags,invocation;std::vector<int64_t>args;};
struct Context{
 bool complete=true;int64_t batch_size=1;std::vector<Route>routes;int consumed=0;
 bool has_complete_physical_manifest()const{return complete;}
 void consume_physical_route(FmbRouteFamily,int64_t site,int64_t selector,int64_t flags,const std::vector<int64_t>&args,int64_t invocation){
  if(!complete)return;
  TORCH_CHECK(std::all_of(args.begin(),args.end(),[](int64_t a){return a>=0;}));
  auto found=std::find_if(routes.begin(),routes.end(),[&](const Route&r){return r.site==site&&r.invocation==invocation;});
  TORCH_CHECK(found!=routes.end()&&found->selector==selector&&found->flags==flags&&found->args==args);++consumed;
 }
};
int fused=0,ordinary=0,expected_layer=1;
void rpu_launch_gr00t_qkv_multibase_w8_kernel(uint32_t in,const at::Tensor&q,const at::Tensor&k,const at::Tensor&v,
 uint32_t qo,uint32_t ko,uint32_t vo,uint32_t qb,uint32_t kb,uint32_t vb,const at::Tensor&qs,const at::Tensor&ks,const at::Tensor&vs){
 const int layer=expected_layer;assert(in==0x10000000&&qo==0x10020000&&ko==0x10030000&&vo==0x10040000);
 assert(q.id==1000+layer*10&&k.id==1001+layer*10&&v.id==1002+layer*10);
 assert(qs.id==3000+layer*10&&ks.id==3001+layer*10&&vs.id==3002+layer*10);
 assert(qb==0x10600000+layer*1536&&kb==qb+512&&vb==qb+1024);++fused;
}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,const at::Tensor&w,uint32_t out,
 int64_t rows,int64_t n,int64_t k,int part,int cores,uint32_t bias,const at::Tensor&scale){
 int which=ordinary%3;assert(in==0x10000000&&out==0x10020000+which*65536);
 assert(w.id==1000+expected_layer*10+which&&scale.id==3000+expected_layer*10+which);
 assert(rows>0&&n==1536&&k==1536&&part==1&&cores==8&&bias==0x10600000+expected_layer*1536+which*512);++ordinary;
}
struct Model{
 __LAYER__
 std::vector<LayerWeights>layer_weights_=std::vector<LayerWeights>(32);
 bool self_qkv_multibase_opt_in_=gr00t_self_qkv_multibase_opt_in(),self_qkv_multibase_ready_=false;
 bool inference_started_=false,z2_bound_=false,denoise_ready_=true,denoise_mode_=true,loop_mode_=true;
 int64_t resolved=0,z2_prepared_seq_len_=0,num_steps_=4,action_horizon_=40,vl_seq_=82,cross_dim_=2048;
 int64_t h=1536,ff=6144,q=32,kv=32,hd=48,tp=8,layers=32;Context context;
 Model(){for(int layer=0;layer<32;++layer){auto&x=layer_weights_[layer];x.is_cross=layer%2==0;
  int projection=0;for(auto*w:{&x.to_q_w,&x.to_k_w,&x.to_v_w}){w->id=1000+layer*10+projection++;if(x.is_cross){w->dtype=at::kHalf;w->shape={1536,2048};}}
  projection=0;for(auto*b:{&x.to_q_b,&x.to_k_b,&x.to_v_b}){b->id=2000+layer*10+projection++;b->dtype=at::kHalf;b->shape={1536};}
  projection=0;for(auto*s:{&x.to_q_ws,&x.to_k_ws,&x.to_v_ws}){s->id=3000+layer*10+projection++;s->dtype=at::kHalf;s->shape={1536};}}
 }
 int64_t get_last_resolved_chunk_size()const{return resolved;}int64_t num_layers()const{return layers;}
 int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return ff;}
 int64_t num_q_heads()const{return q;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}int64_t attn_tp()const{return tp;}int num_cores()const{return NUM_CORES;}
 Context&ctx(){return context;}
 uint32_t addr(int,const char*n){std::string s(n);if(s=="norm_hidden")return 0x10000000;if(s=="q")return 0x10020000;if(s=="k")return 0x10030000;assert(s=="v");return 0x10040000;}
 uint32_t layer_addr(int layer,int,const char*n){std::string s(n);int offset=s=="q_bias"?0:s=="k_bias"?512:1024;return 0x10600000+layer*1536+offset;}
 __METHODS__
 void record(const ChunkInfo&chunk){LayoutContext layout{context.batch_size};context.routes.clear();
  auto append=[&](FmbRouteFamily,int64_t site,int64_t selector,std::vector<int64_t>args=std::vector<int64_t>{}){
   assert(std::all_of(args.begin(),args.end(),[](int64_t a){return a>=0;}));context.routes.push_back({site,selector,0,chunk.idx,std::move(args)});};
  auto append_linear=[&](int64_t site){append(FmbRouteFamily::LINEAR,site,static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE));};
  __REGISTER__
 }
 void project(int layer_idx,const ChunkInfo&chunk){const auto&lw=layer_weights_[layer_idx];int64_t seq_len=chunk.len,nq=32,hd=48,h=1536;
  __PROJECTIONS__
 }
};
int main(){
 unsetenv("RPU_GR00T_SELF_QKV_MULTIBASE");assert(!gr00t_self_qkv_multibase_opt_in());
 for(auto value:{"","0"}){setenv("RPU_GR00T_SELF_QKV_MULTIBASE",value,1);assert(!gr00t_self_qkv_multibase_opt_in());}
 for(auto value:{"true","2","01"," 1","1 "}){setenv("RPU_GR00T_SELF_QKV_MULTIBASE",value,1);rejects([]{gr00t_self_qkv_multibase_opt_in();});}
 setenv("RPU_GR00T_SELF_QKV_MULTIBASE","0",1);Model off;KernelCache::loaded=false;int reads=KernelCache::reads;off.install_self_qkv_multibase();assert(!off.self_qkv_multibase_ready_&&KernelCache::reads==reads);
 ChunkInfo chunk;off.record(chunk);assert(off.context.routes.size()==3);off.project(1,chunk);assert(ordinary==3&&fused==0);
 setenv("RPU_GR00T_SELF_QKV_MULTIBASE","1",1);Model m;setenv("RPU_GR00T_SELF_QKV_MULTIBASE","0",1);assert(m.self_qkv_multibase_opt_in_);
 rejects([&]{m.install_self_qkv_multibase();});assert(!m.self_qkv_multibase_ready_);KernelCache::loaded=true;m.install_self_qkv_multibase();assert(m.self_qkv_multibase_ready_);m.record(chunk);assert(m.context.routes.size()==1&&m.context.routes[0].site==GR00T_DIT_SELF_QKV_MULTIBASE_SITE);
 // The physical descriptor must advertise the actual x-only 18-block grid.
 assert(m.context.routes[0].args[22]==18 && m.context.routes[0].args[23]==1 && m.context.routes[0].args[24]==1);
 for(int request=0;request<3;++request)for(int body=0;body<4;++body)for(int layer=0;layer<32;++layer){
  expected_layer=layer;if(layer%2==0){assert(!m.emit_self_qkv_multibase(layer,chunk,m.layer_weights_[layer]));}else m.project(layer,chunk);}
 assert(fused==3*4*16&&ordinary==3);expected_layer=1;
 // Both ten-step profiles need their own descriptor, including S169 vs S214.
 for(auto admitted:{std::pair{4,1},std::pair{4,256},std::pair{4,193},std::pair{4,225},std::pair{4,289},std::pair{4,238},std::pair{4,270},std::pair{4,334},std::pair{10,334},std::pair{10,1},std::pair{10,97},std::pair{10,169},std::pair{10,214},std::pair{10,256}}){
 const int64_t steps=admitted.first,seq=admitted.second;
 Model ten=m;ten.num_steps_=steps;ten.vl_seq_=seq;assert(ten.self_qkv_multibase_profile(41,1));
 int before=fused;rejects([&]{ten.project(1,chunk);});assert(fused==before);
 ten.record(chunk);assert(ten.context.routes.size()==1);
 assert(ten.context.routes[0].args[2]==seq&&ten.context.routes[0].args[3]==steps);
 for(int request=0;request<3;++request)for(int body=0;body<ten.num_steps_;++body)for(int layer=1;layer<32;layer+=2){
  expected_layer=layer;const int calls=fused;ten.project(layer,chunk);assert(fused==calls+1);}
 expected_layer=1;Model back=ten;back.num_steps_=4;back.vl_seq_=82;
 before=fused;rejects([&]{back.project(1,chunk);});assert(fused==before);
 Model stale=ten;stale.vl_seq_=seq==169?214:169;before=fused;
 rejects([&]{stale.project(1,chunk);});assert(fused==before);
 for(auto profile:{std::pair{4,0},std::pair{4,-1},std::pair{10,0},std::pair{10,-1},std::pair{3,214},std::pair{5,214}}){
  Model x=ten;x.num_steps_=profile.first;x.vl_seq_=profile.second;
  assert(!x.self_qkv_multibase_profile(41,1));x.record(chunk);assert(x.context.routes.size()==3);
  const int calls=fused;x.project(1,chunk);assert(fused==calls);}
 }

 for(int drift=0;drift<4;++drift){auto old=m.context.routes;switch(drift){case 0:m.context.routes[0].flags=1;break;case 1:m.context.routes[0].selector=1;break;case 2:++m.context.routes[0].invocation;break;case 3:++m.context.routes[0].args[0];break;}
  int launches=fused;rejects([&]{m.project(1,chunk);});assert(fused==launches);m.context.routes=old;}
 for(int drift=0;drift<14;++drift){Model x=m;ChunkInfo c=chunk;switch(drift){case 0:x.loop_mode_=false;break;case 1:x.denoise_mode_=false;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:x.vl_seq_=0;break;case 5:c.len=40;break;case 6:x.context.batch_size=2;break;case 7:x.num_steps_=3;break;case 8:x.tp=4;break;case 9:x.ff=4096;break;case 10:x.self_qkv_multibase_opt_in_=false;break;case 11:x.self_qkv_multibase_ready_=false;break;case 12:x.denoise_ready_=false;break;case 13:x.action_horizon_=39;break;}
  assert(!x.self_qkv_multibase_profile(c.len,x.context.batch_size));x.record(c);assert(x.context.routes.size()==3);int old=fused;x.project(1,c);assert(fused==old);}
 m.context.complete=false;int old=fused;m.project(1,chunk);assert(fused==old);m.context.complete=true;
 for(int bad=0;bad<5;++bad){Model x=m;switch(bad){case 0:x.inference_started_=true;break;case 1:x.resolved=41;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:RpuKernelGraph::active=true;break;}
  rejects([&]{x.check_self_qkv_setter_state();});rejects([&]{x.install_self_qkv_multibase();});assert(x.self_qkv_multibase_ready_);RpuKernelGraph::active=false;}
 // Every self layer and all nine original operands participate in cold admission.
 for(int layer:{1,31})for(int operand=0;operand<9;++operand)for(int bad=0;bad<5;++bad){Model x=m;auto&l=x.layer_weights_[layer];std::array<at::Tensor*,9>t{&l.to_q_w,&l.to_k_w,&l.to_v_w,&l.to_q_b,&l.to_k_b,&l.to_v_b,&l.to_q_ws,&l.to_k_ws,&l.to_v_ws};
  switch(bad){case 0:t[operand]->live=false;break;case 1:t[operand]->dtype=operand<3?at::kHalf:at::kChar;break;case 2:t[operand]->dev.value=0;break;case 3:t[operand]->contiguous=false;break;case 4:t[operand]->shape[0]=1535;break;}
  int reads=KernelCache::reads;x.install_self_qkv_multibase();assert(!x.self_qkv_multibase_ready_&&KernelCache::reads==reads);}
 Model all_cross=m;for(auto&l:all_cross.layer_weights_)l.is_cross=true;reads=KernelCache::reads;all_cross.install_self_qkv_multibase();assert(!all_cross.self_qkv_multibase_ready_&&KernelCache::reads==reads);
 Model fallback=m;fallback.layer_weights_[1].to_q_w.dtype=at::kHalf;fallback.install_self_qkv_multibase();fallback.record(chunk);assert(fallback.context.routes.size()==3);
 auto oldargs=m.context.routes[0].args;m.layer_weights_[0].is_cross=false;rejects([&]{m.project(1,chunk);});m.layer_weights_[0].is_cross=true;
}
'''
    cpp = (cpp.replace("__CONSTANTS__", constants)
           .replace("__ENV__", _definition(source, "bool gr00t_self_qkv_multibase_opt_in("))
           .replace("__LAYER__", _definition(source, "struct LayerWeights") + ";")
           .replace("__METHODS__", methods).replace("__REGISTER__", registration)
           .replace("__PROJECTIONS__", projections))
    path, exe = tmp_path / "qkv.cpp", tmp_path / "qkv"
    path.write_text(with_actual_linear_contracts(cpp))
    result = subprocess.run([shutil.which("g++"), "-std=c++17", "-O1", str(path), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_qkv_multibase_cold_binding_and_original_kv_continuation():
    source = MODEL.read_text()
    setters = source.replace("= {}", "= at::Tensor()")
    for name in ("set_weights", "set_denoise_weights"):
        setter = _definition(setters, f"void {name}(")
        assert setter.index("check_self_qkv_setter_state()") < setter.index("TORCH_CHECK(")
    install = _definition(setters, "void set_weights(")
    assert install.index("self_qkv_multibase_ready_ = false") < install.index("install_self_qkv_multibase()") < install.index("invalidate_model_state()")
    emitter = _definition(source, "bool emit_self_qkv_multibase(")
    assert emitter.index("consume_physical_route(") < emitter.index("rpu_launch_gr00t_qkv_multibase_w8_kernel(")
    body = _definition(source, "void build_self_attn_block(")
    assert body.index("emit_self_qkv_multibase(") < body.index("consume_kv_insert_plan(") < body.index("rpu_launch_v_transpose_spm(")
    assert body.count("rpu_launch_linear_spm_to_spm_acc16_kernel(") == 3
    assert "emit_self_qkv_multibase(" not in _definition(source, "void build_cross_attn_block(")
    assert "self_qkv_multibase" not in _definition(source, "std::vector<BufferDecl> baseline_buffer_declarations(")
    layout = _definition(source, "int64_t subclass_layout_hash(")
    assert "GR00T_DIT_SELF_QKV_MULTIBASE_SITE" in layout and "self_qkv_multibase_ready_" in layout
    # No derived tensor or packed weight: original owners remain in model identity.
    owner = _definition(source, "std::vector<int64_t> kvinsert_cost_weight_identity(")
    assert "&weights.to_q_w" in owner and "&weights.to_k_ws" in owner and "&weights.to_v_b" in owner
    assert "GR00T_DIT_SELF_QKV_MULTIBASE_SITE" in owner
