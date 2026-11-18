"""Actual dense glast policy/route/caller snippets under a CPU SDK recorder.

The recorder uses logical core addresses; physical allocation/remapping is covered
only by separately recorded hardware diagnostics, not this facade.
"""
from pathlib import Path
import importlib
import json

ROOT = Path(__file__).resolve().parents[1]


def test_actual_glast_routing_complete_identity_and_mutations(tmp_path):
    from test_g05_vision_owner_model import _block
    from test_g05_q_gate_native import _compile_run
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    header = (ROOT / 'src/fused/rpu_qwen3_5_model.h').read_text()
    names = [('bool', 'use_g05_gdn_local_slice', 'int64_t rows'),
             ('std::vector<int64_t>', 'g05_gdn_local_slice_arguments', 'int64_t rows'),
             ('bool', 'has_text_gdn_glast_local_capability', ''),
             ('std::vector<int64_t>', 'text_gdn_glast_local_identity', ''),
             ('bool', 'use_text_gdn_glast_local', 'int64_t rows'),
             ('std::vector<int64_t>', 'text_gdn_glast_local_arguments', 'int64_t rows')]
    methods = '\n'.join(f'{ret} {name}({args})const' + _block(source, f'{ret} Qwen3_5Model::{name}(')
                        for ret, name, args in names)
    a = source.index('    const bool g05_local_slice =')
    original = source[a:source.index('    // ===== Phase 3:', a)]
    a = source.index('    const bool glast_local =')
    glast = source[a:source.index('    // ===== Phase 16:', a)]
    beta = '\n'.join(l for l in source.splitlines() if 'slice_pc(D("gdn_c_' in l)
    emits = '\n'.join(l for l in source.splitlines() if 'slice_glast(a_g' in l)
    assert len(emits.splitlines()) == 2
    manifest = '\n'.join(f'if ({name}(chunk.len))' + _block(source, f'if ({name}(chunk.len))')
                         for name in ['use_g05_gdn_local_slice', 'use_text_gdn_glast_local'])
    layout = '\n'.join(f'if ({cond})' + _block(source, f'if ({cond})') for cond in
                       ['use_g05_gdn_local_slice(256)', 'has_text_gdn_glast_local_capability()'])
    cost = '\n'.join(f'if ({cond})' + _block(header, f'if ({cond})') for cond in
                     ['use_g05_gdn_local_slice(256)', 'has_text_gdn_glast_local_capability()'])
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
constexpr int64_t QWEN35_TEXT_GDN_LOCAL_SLICE_SITE=4624574716066886237LL;
constexpr int64_t QWEN35_TEXT_GDN_GLAST_LOCAL_SITE=5486472411379814366LL;
namespace detail{uint64_t layout_mix(uint64_t x,int64_t y){return x*131+uint64_t(y);}}
struct Spm{uint32_t addr(int c,int o)const{return 0x40000400u+c*0x1000000u+o;}}SPM_ALLOC;
enum class FmbRouteFamily{GRAPH_SCHEDULE};
struct ChunkInfo{int64_t len,idx;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
struct Call{uint32_t src,dst;std::vector<int64_t>is,os,b;int nc;bool local;};
std::vector<Call> calls;
void rpu_launch_slice_spm_kernel(uint32_t s,uint32_t d,const std::vector<int64_t>&is,const std::vector<int64_t>&os,const std::vector<int64_t>&b,int nc,bool local=false){calls.push_back({s,d,is,os,b,nc,local});}
struct Model {
 bool action_mode_=false,z2_bound_=false,debug=false,fast_replay_enabled_=false,retained_prefill_graph_=false,has_gdn_=true,complete=true;
 int cores=4,layers=64,h=5120,inter=17408,nq=24,nkv=4,hd=256;
 int64_t gdn_nvh_=48,gdn_dk_=128,gdn_dv_=128,gdn_conv_dim_=10240,gdn_kc_=4;
 std::vector<int>layer_is_full_=std::vector<int>(64);std::vector<Route>routes;int consumed=0;
 bool get_debug_export()const{return debug;}int num_cores()const{return cores;}int gdn_tp()const{return cores==8?8:4;}
 int num_layers()const{return layers;}int hidden_size()const{return h;}int intermediate_size()const{return inter;}
 int num_q_heads()const{return nq;}int num_kv_heads()const{return nkv;}int head_dim()const{return hd;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t sel,int64_t flags,const std::vector<int64_t>&args,int64_t inv){
  TORCH_CHECK(routes.size()==1);auto&r=routes[0];TORCH_CHECK(r.family==f&&r.site==site&&r.selector==sel&&r.flags==flags&&r.args==args&&r.invocation==inv);++consumed;
 }
 __METHODS__
 void plan(ChunkInfo chunk){struct{std::vector<Route>routes;}manifest;__MANIFEST__ routes=manifest.routes;}
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
  assert(m.routes.size()==1&&m.routes[0].site==QWEN35_TEXT_GDN_GLAST_LOCAL_SITE);
  int nc=m.gdn_tp(),hc=heads/nc;assert(m.routes[0].args==std::vector<int64_t>({1,rows,64,hc,nc,2,128,2,63,1,2*(rows/64)}));
  auto key=m.layout_hash();auto cost=m.cost_identity();assert(key&&cost.size()>1);m.plan({64,0});assert(m.layout_hash()==key&&m.cost_identity()==cost);m.plan(c);
  calls.clear();m.slice(c);assert(m.consumed==1);assert(calls.size()==size_t(2*nc+2*(rows/64)));
  for(int i=0;i<2*nc;++i)assert(!calls[i].local&&calls[i].nc==1);
  for(int i=0;i<2*(rows/64);++i){auto&x=calls[2*nc+i];assert(x.local&&x.nc==nc&&x.is==std::vector<int64_t>({hc,64})&&x.os==std::vector<int64_t>({hc,1})&&x.b==std::vector<int64_t>({0,63}));assert(x.src==SPM_ALLOC.addr(0,i%2?0x40000:0x20000)+uint32_t((i/2)*hc*64*2));assert(x.dst==SPM_ALLOC.addr(0,0x180000));}
  for(size_t i=0;i<m.routes[0].args.size();++i){auto bad=m;++bad.routes[0].args[i];calls.clear();reject([&]{bad.slice(c);});assert(calls.size()==size_t(2*nc));}
  for(int i=0;i<6;++i){auto bad=m;if(i==0)bad.complete=false;if(i==1)bad.routes.clear();if(i==2)++bad.routes[0].invocation;if(i==3)++bad.routes[0].selector;if(i==4)++bad.routes[0].flags;if(i==5)++bad.routes[0].site;calls.clear();reject([&]{bad.slice(c);});assert(calls.size()==size_t(2*nc));}
 }
 Model m;for(int n:{0,1,32,63,65,127}){m.plan({n,0});assert(m.routes.empty()&&!m.use_text_gdn_glast_local(n));}
 for(int bad=0;bad<5;++bad){Model x;if(bad==0)x.action_mode_=true;if(bad==1)x.z2_bound_=true;if(bad==2)x.has_gdn_=false;if(bad==3)x.gdn_nvh_=0;if(bad==4)x.gdn_nvh_=49;assert(!x.has_text_gdn_glast_local_capability());x.plan({64,0});assert(x.routes.empty());}
 Model g;g.cores=8;g.layers=24;g.h=2048;g.inter=6144;g.nq=g.nkv=8;g.gdn_nvh_=16;g.gdn_conv_dim_=6144;g.fast_replay_enabled_=g.retained_prefill_graph_=true;g.layer_is_full_=std::vector<int>(24);for(int i=0;i<24;++i)g.layer_is_full_[i]=(i+1)%4==0;
 g.plan({256,1});assert(g.routes.size()==1&&g.routes[0].site==QWEN35_TEXT_GDN_LOCAL_SLICE_SITE);calls.clear();g.slice({256,1});assert(g.consumed==1&&calls.size()==10);for(auto&x:calls)assert(x.local&&x.nc==8);
 g.debug=true;g.consumed=0;g.plan({256,2});assert(g.routes.size()==1&&g.routes[0].site==QWEN35_TEXT_GDN_GLAST_LOCAL_SITE);calls.clear();g.slice({256,2});assert(g.consumed==1&&calls.size()==24);for(int i=0;i<16;++i)assert(!calls[i].local);for(int i=16;i<24;++i)assert(calls[i].local);
}
'''
    for marker, value in {'__METHODS__': methods, '__MANIFEST__': manifest,
                          '__LAYOUT__': layout, '__COST__': cost,
                          '__ORIGINAL__': original, '__BETA__': beta,
                          '__GLAST__': glast, '__EMITS__': emits}.items():
        cpp = cpp.replace(marker, value)
    _compile_run(tmp_path, 'dense_glast_complete_schedule', cpp)


