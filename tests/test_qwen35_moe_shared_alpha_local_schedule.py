"""Extract actual shared-alpha route and launch code into the existing CPU recorder.

This verifies scheduling contracts, never hardware ordinal mapping or EME repair.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def test_actual_shared_alpha_routes_and_local_launches(tmp_path):
    from test_g05_vision_owner_model import _block
    from test_g05_q_gate_native import _compile_run
    source = (ROOT / "src/fused/rpu_qwen3_5_moe_model.cpp").read_text()
    constants = "\n".join(re.findall(r"constexpr int64_t QWEN35_MOE_(?:RESIDENT_)?SHARED_ALPHA_LOCAL_SITE = \d+LL;", source))
    helpers = "\n".join(signature + _block(source, signature) for signature in (
        "std::vector<int64_t> shared_alpha_local_identity(int nc)",
        "std::vector<int64_t> shared_alpha_local_arguments(int64_t rows, int nc)"))
    parts = {}
    for name, method in [("ordinary", "emit_shared_expert"), ("resident", "emit_resident_mlp")]:
        body = _block(source, "void Qwen3_5MoeModel::" + method + "(")
        start = body.index('    TORCH_CHECK(ctx().has_complete_physical_manifest(),\n                "Qwen3.5 MoE scalar slice')
        stop = body.index("    consume_manifest_route(", start)
        parts[name] = body[start:stop]
        assert parts[name].count("rpu_launch_slice_spm_kernel(") == 1
        assert "for (" not in parts[name] and "addr(core" not in parts[name]
    body = _block(source, "void Qwen3_5MoeModel::append_mlp_manifest_routes(")
    start = body.index("        manifest.routes.push_back({\n            QWEN35_MOE_SHARED_ALPHA_LOCAL_SITE")
    parts["ordinary_manifest"] = body[start:body.index("        append_route(", start)]
    body = _block(source, "void Qwen3_5MoeModel::append_resident_mlp_manifest_routes(")
    start = body.index("    append(FmbRouteFamily::GRAPH_SCHEDULE,\n        QWEN35_MOE_RESIDENT_SHARED_ALPHA_LOCAL_SITE")
    parts["resident_manifest"] = body[start:body.index("    append(FmbRouteFamily::ACTIVATION", start)]
    body = _block(source, "int64_t Qwen3_5MoeModel::subclass_layout_hash(")
    parts["layout"] = body[body.index("    for (const auto value : shared_alpha_local_identity"):body.index("    if (has_gdn_glast_local_capability())")]
    body = _block(source, "std::vector<int64_t> Qwen3_5MoeModel::kvinsert_cost_weight_identity(")
    parts["cost"] = body[body.index("    const auto alpha_schedule"):body.index("    if (has_gdn_glast_local_capability())")]
    cpp = r"""
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
__constants__
__helpers__
namespace detail { uint64_t layout_mix(uint64_t x,int64_t y){return x*131+uint64_t(y);} }
enum class FmbRouteFamily { GRAPH_SCHEDULE };
struct Route {int64_t site; FmbRouteFamily family; int64_t selector,flags; std::vector<int64_t> args; int64_t invocation;};
struct Call {uint32_t source,destination;std::vector<int64_t> in,out,begin;int nc;bool local;};
std::vector<Call> calls;
void rpu_launch_slice_spm_kernel(uint32_t s,uint32_t d,const std::vector<int64_t>&i,const std::vector<int64_t>&o,const std::vector<int64_t>&b,int n,bool l=false){calls.push_back({s,d,i,o,b,n,l});}
struct Model {
 int nc=6; bool complete=true; int consumed=0; std::vector<Route> routes;
 int shared_expert_tp()const{return nc;}
 uint32_t addr(int c,const char*n)const{return 0x40000400u+c*0x1000000u+(std::string(n)=="moe_shared_alpha_pad"?0x1000:0x5000);}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,const std::vector<int64_t>&args,int64_t inv){
  TORCH_CHECK(routes.size()==1);const auto&r=routes[0];
  TORCH_CHECK(r.family==f&&r.site==site&&r.selector==selector&&r.flags==flags&&r.args==args&&r.invocation==inv);++consumed;
 }
 void plan(int64_t rows,int64_t offset,bool resident){
  struct {std::vector<Route>routes;}manifest;int64_t t=rows,w=rows,invocation=offset,shared_inv=offset+8192;
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,std::vector<int64_t>args,int64_t inv){manifest.routes.push_back({site,f,selector,0,args,inv});};
  if(resident){__resident_manifest__}else{__ordinary_manifest__}routes=manifest.routes;
 }
 void emit(int64_t rows,int64_t offset,bool resident){
  int64_t w=rows,t=rows,invocation=offset,shared_inv=offset+8192;
  if(resident){__resident__}else{__ordinary__}
 }
 uint64_t layout()const{uint64_t hash=0;__layout__ return hash;}
 std::vector<int64_t>cost()const{std::vector<int64_t>identity{1};__cost__ return identity;}
};
template<class F>void reject(F f){bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}assert(rejected);}
int main(){
 for(bool resident:{false,true})for(int nc:{4,6,8})for(int rows:{1,2,64,128,192,256,512})for(int offset:{0,128,512}){
  if(resident&&(nc!=8||rows<64))continue;
  Model m;m.nc=nc;auto cold=m.layout();auto cost=m.cost();m.plan(rows,offset,resident);
  assert(m.layout()==cold&&m.cost()==cost&&cold);
  assert(m.routes.size()==1);auto&r=m.routes[0];
  assert(r.site==(resident?QWEN35_MOE_RESIDENT_SHARED_ALPHA_LOCAL_SITE:QWEN35_MOE_SHARED_ALPHA_LOCAL_SITE));
  assert(r.args==std::vector<int64_t>({1,rows,16,1,nc,2,32,2,0,1}));
  assert(r.invocation==offset+(resident?8192:0));
  calls.clear();m.emit(rows,offset,resident);assert(m.consumed==1&&calls.size()==1);
  auto&c=calls[0];assert(c.local&&c.nc==nc&&c.source==m.addr(0,"moe_shared_alpha_pad")&&c.destination==m.addr(0,"moe_shared_alpha"));
  assert(c.in==std::vector<int64_t>({rows,16})&&c.out==std::vector<int64_t>({rows,1})&&c.begin==std::vector<int64_t>({0,0}));
  for(size_t k=0;k<r.args.size();++k){auto b=m;++b.routes[0].args[k];calls.clear();reject([&]{b.emit(rows,offset,resident);});assert(calls.empty());}
  for(int k=0;k<6;++k){auto b=m;if(k==0)b.complete=false;if(k==1)b.routes.clear();if(k==2)++b.routes[0].site;if(k==3)++b.routes[0].selector;if(k==4)++b.routes[0].flags;if(k==5)++b.routes[0].invocation;calls.clear();reject([&]{b.emit(rows,offset,resident);});assert(calls.empty());}
 }
 for(int nc:{0,1,2,3,5,7,9})reject([&]{shared_alpha_local_arguments(64,nc);});
 for(int rows:{0,-1})reject([&]{shared_alpha_local_arguments(rows,6);});
 assert(shared_alpha_local_identity(4)!=shared_alpha_local_identity(6));
 assert(shared_alpha_local_identity(6)!=shared_alpha_local_identity(8));
}
"""
    for key, value in dict(parts, constants=constants, helpers=helpers).items():
        cpp = cpp.replace("__" + key + "__", value)
    _compile_run(tmp_path, "moe_shared_alpha_local", cpp)


