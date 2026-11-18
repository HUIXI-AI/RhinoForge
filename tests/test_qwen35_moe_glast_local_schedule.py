"""Actual MoE glast policy/route/caller snippets under a CPU SDK recorder.

The recorder uses logical core addresses; physical allocation/remapping is covered
only by separately recorded hardware diagnostics, not this facade.
"""
from pathlib import Path
import importlib
import json

ROOT = Path(__file__).resolve().parents[1]


def test_actual_moe_glast_routes_and_resident_invocations(tmp_path):
    from test_g05_vision_owner_model import _block
    from test_g05_q_gate_native import _compile_run
    source = (ROOT / 'src/fused/rpu_qwen3_5_moe_model.cpp').read_text()
    header = (ROOT / 'src/fused/rpu_qwen3_5_moe_model.h').read_text()
    names = [('bool', 'has_gdn_glast_local_capability', ''),
             ('std::vector<int64_t>', 'gdn_glast_local_identity', ''),
             ('bool', 'use_gdn_glast_local', 'int64_t rows'),
             ('std::vector<int64_t>', 'gdn_glast_local_arguments', 'int64_t rows')]
    methods = '\n'.join(f'{ret} {name}({args})const' + _block(source, f'{ret} Qwen3_5MoeModel::{name}(')
                        for ret, name, args in names)
    a = source.index('    auto pc_addr =')
    original = source[a:source.index('    // ===== Phase 3:', a)]
    a = source.index('    const bool glast_local =')
    glast = source[a:source.index('    // ===== Phase 16:', a)]
    beta = '\n'.join(l for l in source.splitlines() if 'slice_pc(D("gdn_c_' in l)
    emits = '\n'.join(l for l in source.splitlines() if 'slice_glast(a_g' in l)
    assert len(emits.splitlines()) == 2
    manifest = 'if (use_gdn_glast_local(chunk.len))' + _block(source, 'if (use_gdn_glast_local(chunk.len))')
    layout = 'if (has_gdn_glast_local_capability())' + _block(source, 'if (has_gdn_glast_local_capability())')
    cost_source = source[source.index('std::vector<int64_t> Qwen3_5MoeModel::kvinsert_cost_weight_identity()'):]
    cost = 'if (has_gdn_glast_local_capability())' + _block(cost_source, 'if (has_gdn_glast_local_capability())')
    # Extract the production expansion of outer windows into actual mixer
    # calls. The resident caller and manifest must consume the same chunks.
    a = source.index('    std::vector<ChunkInfo> mixer_chunks;')
    b = source.index('    for (const ChunkInfo& chunk : mixer_chunks)', a)
    expansion = source[a:b]
    assert source.index('if (use_gdn_glast_local(chunk.len))', b) < source.index('bool needs_auto_tile', b)
    resident_method = _block(source, 'std::vector<ChunkInfo> Qwen3_5MoeModel::planned_resident_mixer_chunks(')
    resident_caller = _block(source, 'void Qwen3_5MoeModel::build_resident_layer(')
    assert 'for (const auto& sub : planned_resident_mixer_chunks(chunk))' in resident_caller
    assert 'build_gdn(layer_idx, sub, /*decode=*/false)' in resident_caller
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
constexpr int64_t QWEN35_MOE_EXACT_PROFILE_SITE=2504589813057384578LL;
constexpr int64_t QWEN35_MOE_GDN_GLAST_LOCAL_SITE=8383641522579396593LL;
namespace detail{uint64_t layout_mix(uint64_t x,int64_t y){return x*131+uint64_t(y);}}
struct Spm{uint32_t addr(int c,int o)const{return 0x40000400u+c*0x1000000u+o;}}SPM_ALLOC;
enum class FmbRouteFamily{GRAPH_SCHEDULE};
struct ChunkInfo{int64_t idx,offset,len,kv_seq_len; ChunkInfo(int64_t n,int64_t i):idx(i),offset(0),len(n),kv_seq_len(n){} ChunkInfo(int64_t i,int64_t o,int64_t n,int64_t k):idx(i),offset(o),len(n),kv_seq_len(k){}};
struct MixerCall{int64_t offset,rows,absolute_start;};
struct Window{std::vector<MixerCall>calls;};
struct ResidentPlan{std::vector<Window>windows;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
struct Call{uint32_t src,dst;std::vector<int64_t>is,os,b;int nc;bool local;};
std::vector<Call> calls;
void rpu_launch_slice_spm_kernel(uint32_t s,uint32_t d,const std::vector<int64_t>&is,const std::vector<int64_t>&os,const std::vector<int64_t>&b,int nc,bool local=false){calls.push_back({s,d,is,os,b,nc,local});}
struct Model {
 bool action_mode_=false,z2_bound_=false,debug=false,fast_replay_enabled_=false,retained_prefill_graph_=false,has_gdn_=true,complete=true;
 int cores=4,layers=64,h=5120,inter=17408,nq=24,nkv=4,hd=256;
 int64_t gdn_nvh_=48,gdn_dk_=128,gdn_dv_=128,gdn_conv_dim_=10240,gdn_kc_=4;
 std::vector<int>layer_is_full_=std::vector<int>(64);std::vector<Route>routes;int consumed=0;
 bool get_debug_export()const{return debug;}int num_cores()const{return cores;}int gdn_tp()const{return cores==6?4:cores;}
 int num_layers()const{return layers;}int hidden_size()const{return h;}int intermediate_size()const{return inter;}
 int num_q_heads()const{return nq;}int num_kv_heads()const{return nkv;}int head_dim()const{return hd;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t sel,int64_t flags,const std::vector<int64_t>&args,int64_t inv){
  TORCH_CHECK(routes.size()==1);auto&r=routes[0];TORCH_CHECK(r.family==f&&r.site==site&&r.selector==sel&&r.flags==flags&&r.args==args&&r.invocation==inv);++consumed;
 }
 bool resident_prefill_=false; ResidentPlan resident;
 const ResidentPlan&resident_plan_for_outer(const ChunkInfo&)const{return resident;}
 std::vector<ChunkInfo>planned_resident_mixer_chunks(const ChunkInfo&outer)const __RESIDENT__
 std::vector<int64_t>exact_profile_arguments()const{return {};}
 template<class M> void append_mlp_manifest_routes(M&,const ChunkInfo&)const{}
 __METHODS__
 void plan(ChunkInfo chunk){struct{std::vector<Route>routes;}manifest;__MANIFEST__ routes=manifest.routes;}
 void full_plan(std::vector<ChunkInfo>outer_chunks,int64_t physical_len){
  struct{struct{std::vector<ChunkInfo>chunks;}compute;}plan;plan.compute.chunks=outer_chunks;
  struct{std::vector<Route>routes;}manifest;__EXPANSION__
  for(const auto&chunk:mixer_chunks){__MANIFEST__}
  routes.clear();for(const auto&r:manifest.routes)if(r.site==QWEN35_MOE_GDN_GLAST_LOCAL_SITE)routes.push_back(r);
 }
 uint64_t layout_hash()const{uint64_t hash=0;__LAYOUT__ return hash;}
 std::vector<int64_t>cost_identity()const{std::vector<int64_t>identity{1};__COST__ return identity;}
 void slice(ChunkInfo chunk){int64_t L=chunk.len,Hc=gdn_nvh_/gdn_tp(),C=64,N=L/C,n_bg=16,vg_c=Hc;int nc=gdn_tp();
  auto D=[&](const char* name){std::string n(name);return SPM_ALLOC.addr(0,n=="gdn_c_glast"?0x180000:0x10000);};
  __ORIGINAL__
  __BETA__
  __GLAST__
  const std::vector<int64_t> sh_gl{Hc,C},sl_gl{Hc,1},bC{0,C-1};
  for(int64_t i=0;i<N;++i){
   const uint32_t a_gexpc=SPM_ALLOC.addr(0,0x20000)+uint32_t(i*Hc*C*2);
   const uint32_t a_gc=SPM_ALLOC.addr(0,0x40000)+uint32_t(i*Hc*C*2);
   __EMITS__
  }
 }
};
template<class F> void reject(F f){try{f();assert(false);}catch(const std::runtime_error&){} }
int main(){
 for(int cores:{4,6,8})for(int heads:{16,32,48,64})for(int rows:{64,128,192,256,320,512})for(bool debug:{false,true}){
  Model m;m.cores=cores;m.gdn_nvh_=heads;m.debug=debug;ChunkInfo c{rows,7};m.plan(c);
  assert(m.routes.size()==1&&m.routes[0].site==QWEN35_MOE_GDN_GLAST_LOCAL_SITE);
  int nc=m.gdn_tp(),hc=heads/nc;assert(m.routes[0].args==std::vector<int64_t>({1,rows,64,hc,nc,2,128,2,63,1,2*(rows/64)}));
  auto key=m.layout_hash();auto cost=m.cost_identity();assert(key&&cost.size()>1);m.plan({64,0});assert(m.layout_hash()==key&&m.cost_identity()==cost);m.plan(c);
  calls.clear();m.slice(c);assert(m.consumed==1);assert(calls.size()==size_t(2*nc+2*(rows/64)));
  for(int i=0;i<2*nc;++i)assert(!calls[i].local&&calls[i].nc==1);
  for(int i=0;i<2*(rows/64);++i){auto&x=calls[2*nc+i];assert(x.local&&x.nc==nc&&x.is==std::vector<int64_t>({hc,64})&&x.os==std::vector<int64_t>({hc,1})&&x.b==std::vector<int64_t>({0,63}));assert(x.src==SPM_ALLOC.addr(0,i%2?0x40000:0x20000)+uint32_t((i/2)*hc*64*2));assert(x.dst==SPM_ALLOC.addr(0,0x180000));}
  for(size_t i=0;i<m.routes[0].args.size();++i){auto bad=m;++bad.routes[0].args[i];calls.clear();reject([&]{bad.slice(c);});assert(calls.size()==size_t(2*nc));}
  for(int i=0;i<6;++i){auto bad=m;if(i==0)bad.complete=false;if(i==1)bad.routes.clear();if(i==2)++bad.routes[0].invocation;if(i==3)++bad.routes[0].selector;if(i==4)++bad.routes[0].flags;if(i==5)++bad.routes[0].site;calls.clear();reject([&]{bad.slice(c);});assert(calls.size()==size_t(2*nc));}
 }
 Model m;for(int n:{0,1,32,63,65,127}){m.plan({n,0});assert(m.routes.empty()&&!m.use_gdn_glast_local(n));}
 for(int bad=0;bad<5;++bad){Model x;if(bad==0)x.cores=2;if(bad==1)x.cores=3;if(bad==2)x.has_gdn_=false;if(bad==3)x.gdn_nvh_=0;if(bad==4)x.gdn_nvh_=49;assert(!x.has_gdn_glast_local_capability());x.plan({64,0});assert(x.routes.empty());}
 for(int cores:{4,6,8}){
  Model r;r.cores=cores;r.resident_prefill_=true;
  r.resident.windows={{{{0,128,0},{128,64,128},{192,64,192}}},{{{256,192,256},{448,64,448}}}};
  auto cold=r.layout_hash();auto cost=r.cost_identity();
  r.full_plan({ChunkInfo(0,0,256,256),ChunkInfo(1,256,256,512)},512);
  assert(r.routes.size()==5);
  const int64_t ids[]={0,2,3,4,7},rows[]={128,64,64,192,64};
  auto planned=r.routes;
  for(int i=0;i<5;++i){assert(planned[i].invocation==ids[i]&&planned[i].args[1]==rows[i]);r.routes={planned[i]};calls.clear();r.slice({rows[i],ids[i]});assert(calls.size()==size_t(2*r.gdn_tp()+2*(rows[i]/64)));}
  assert(r.layout_hash()==cold&&r.cost_identity()==cost);
  r.resident_prefill_=false;r.full_plan({ChunkInfo(0,0,256,256),ChunkInfo(1,256,256,512)},512);
  assert(r.routes.size()==2&&r.routes[0].invocation==0&&r.routes[1].invocation==1&&r.routes[1].args[1]==256);
  r.full_plan({ChunkInfo(1,0)},1);assert(r.routes.empty());
 }

}
'''
    for marker, value in {'__METHODS__': methods, '__MANIFEST__': manifest,
                          '__LAYOUT__': layout, '__COST__': cost,
                          '__ORIGINAL__': original, '__BETA__': beta,
                          '__GLAST__': glast, '__EMITS__': emits, '__RESIDENT__': resident_method, '__EXPANSION__': expansion}.items():
        cpp = cpp.replace(marker, value)
    _compile_run(tmp_path, 'moe_glast_complete_schedule', cpp)


