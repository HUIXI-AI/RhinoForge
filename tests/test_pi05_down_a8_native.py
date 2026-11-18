"""Run the actual Pi Down A8 layout, emitters and COMPLETE route checks."""
from pathlib import Path
import re

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def test_actual_down_a8_uses_retired_glu_roots_and_preserves_carry(tmp_path, monkeypatch):
    import test_pi05_t128_two_pair_layout as layout_fixture
    import torch

    # Share the production declaration/allocator fixture without intercepting
    # a test execution or duplicating the allocation algorithm.
    code = layout_fixture._declaration_fixture_code()
    model = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    methods = "\n".join(_function(model, "    " + marker) for marker in (
        "int64_t pi05_down_pair_linear_selector(",
        "std::vector<int64_t> pi05_down_a8_quant_arguments(",
        "std::array<uint32_t, 4> pi05_down_a8_workspace(",
        "void emit_pi05_down_a8_linear(",
        "void pi05_down_pair_validate_routes(",
    ))
    builder = _function(model, "    FmbPhysicalExecutionManifest physical_manifest_for_candidate(")
    builder = builder[builder.index("        append(FmbRouteFamily::GRAPH_SCHEDULE, PI05_OWNER_NORM_POLICY_SITE,"):]
    builder = builder[:builder.index("        const bool gateup_pair =")]
    finish = _function(model, "    void emit_pi05_o_pair_finish(")
    tail = finish[finish.rindex("        for(int half=0;half!=2;++half)"):finish.rindex("\n    }")]
    sites = set(re.findall(r"\b(?:PI05_\w+_SITE|GEMMA_\w+_SITE|FMB_SHARED_\w+_SITE)\b", methods + builder + tail))
    for site in sorted(sites):
        if site.startswith("FMB_SHARED_") or re.search(rf"constexpr int64_t {site}\s*=", code):
            continue
        definition = re.search(rf"^constexpr int64_t {site}\s*=\s*[^;]+;", model, re.M)
        assert definition is not None, site
        code = definition[0] + "\n" + code
    # Constants need int64_t before the original generated includes.
    code = '#include <cstdint>\n#include "ops/rpu_linear_w8a8_contract.h"\n' + code
    owner_start = code.index("struct Owner {")
    prefix, owner = code[:owner_start], code[owner_start:]
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    dma_start = decls.index("struct DeferredDmaChannels {")
    prefix += decls[dma_start:decls.index("};", dma_start) + 2] + "\n"
    prefix += r'''
struct Context {
 FmbPhysicalExecutionManifest manifest;
 std::vector<FmbRouteManifestEntry> consumed;
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,
     int64_t flags,std::vector<int64_t> args,int64_t invocation=0){
  const auto found=std::find_if(manifest.routes.begin(),manifest.routes.end(),[&](const auto&r){return r.site_id==site&&r.invocation==invocation&&r.family==family&&r.selector==selector&&r.flags==flags&&r.arguments==args;});
  TORCH_CHECK(found!=manifest.routes.end(),"missing actual route");consumed.push_back(*found);
 }
};
struct MockSpm {uint32_t addr(int,int)const{return 256;}} mock_spm;
#undef SPM_ALLOC
#define SPM_ALLOC mock_spm
uint64_t C,B,Q,SX,Y0,Y1,O0,O1,RAW;
const at::Tensor *expected_w=nullptr,*expected_sw=nullptr;
std::vector<uint8_t> memory;
std::vector<int> operations;
void put(uint64_t p,uint64_t n,uint8_t v){assert(p+n<=memory.size());std::fill_n(memory.begin()+p,n,v);}
void has(uint64_t p,uint64_t n,uint8_t v){assert(p+n<=memory.size());for(uint64_t i=0;i<n;++i)assert(memory[p+i]==v);}
void rpu_launch_dynamic_quantize_w8a8_spm_kernel(uint32_t x,uint32_t q,uint32_t sx,int64_t m,int64_t k,int cores){
 const int half=x==Y0?0:1;assert(x==(half?Y1:Y0)&&m==int64_t(C)&&k==2048&&cores==8);
 assert(q==Q+half*B/2&&sx==SX+half*C*2);has(x,B,half?20:10);has(RAW,B/4,30);
 put(q,B/2,half?3:1);put(sx,C*2,half?4:2);operations.push_back(half);
}
void rpu_launch_linear_w8a8_spm_kernel(uint32_t q,uint32_t sx,const at::Tensor&w,const at::Tensor&sw,uint32_t y,int64_t m,int64_t n,int64_t k,int part,int cores){
 assert(q==Q&&sx==SX&&y==Y0&&m==int64_t(2*C)&&n==2048&&k==16384&&part==0&&cores==8);
 assert(&w==expected_w&&&sw==expected_sw);has(Q,B/2,1);has(Q+B/2,B/2,3);has(SX,C*2,2);has(SX+C*2,C*2,4);
 has(RAW,B/4,30);put(Y0,B,40);put(Y1,B,41);operations.push_back(2);
}
void rpu_launch_pi05_compact_residual_spm_kernel(uint32_t partial,uint32_t residual,uint32_t output,int64_t rows){
 const int half=partial==Y0?0:1;assert(partial==(half?Y1:Y0)&&residual==RAW+(half?0:B/8)&&output==(half?O1:O0)&&rows==int64_t(C));
 has(partial,B,half?41:40);has(residual,B/8,30);if(!half)has(Y1,B,41);
 put(output,B,half?51:50);operations.push_back(3+half*2);
}
void rpu_launch_rmsnorm_spm_kernel(uint32_t x,uint32_t y,uint32_t,int64_t m,int64_t n,double,int,int cores){assert(x==y&&m==int64_t(C)&&n==2048&&cores==8);operations.push_back(9);}
'''
    owner = owner.replace("at::Tensor input_norm_w,post_attn_norm_w;", "at::Tensor input_norm_w,post_attn_norm_w,down_proj_w,down_ws;")
    owner = owner.replace("return 0x100+offsets.at(name);", 'if(std::string(name)=="final_norm_w")return 0x800000;const auto it=addr_overrides.find(name);if(it!=addr_overrides.end())return it->second;return 0x100+offsets.at(name);')
    end = owner.rindex("};")
    owner = owner[:end] + r'''
 bool pi05_prefill_down_pair_=true;
 double eps_=1e-6;
 std::unordered_map<std::string,uint32_t> addr_overrides;
 Context context;Context&ctx(){return context;}
 uint32_t consume_rmsnorm_route(int64_t,const ChunkInfo&)const{return 0;}
 bool carried_=false;
 DeferredDmaChannels pi05_o_pair_output_dma_;
 int pi05_o_pair_carried_layer_=-1;
 void emit_layer_output_dma(int,const ChunkInfo& chunk,const char* name,uint32_t offset=0,
                            DeferredDmaChannels* deferred=nullptr,bool replicated=false){
  assert(offset==0 && !(carried_ && chunk.idx==1));
  assert(deferred==(carried_ ? &pi05_o_pair_output_dma_ : nullptr));
  assert(replicated==carried_);
  assert(addr(0,name)==(chunk.idx?O1:O0));has(addr(0,name),B,chunk.idx?51:50);
  if(deferred)deferred->used=0xfe;
  operations.push_back(4+chunk.idx*2);
 }
''' + methods + r'''
 FmbPhysicalExecutionManifest actual_down_manifest(){
  FmbPhysicalExecutionManifest manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;
  manifest.logical_length=manifest.physical_length=manifest.kv_logical_length=pi05_prefix_rows();
  const bool paired_down=true,owner_norm=true,o_pair=true;
  const auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation=0){manifest.routes.push_back({site,f,selector,flags,std::move(args),invocation});};
''' + builder + r'''
  for(const auto& route:pi05_o_pair_routes())manifest.routes.push_back(route);
  return manifest;
 }
 void actual_consumers(int layer_idx){
  const uint32_t s0=addr(0,"residual1"),s1=addr(0,"pi05_pair_s1"),s2=addr(0,"pi05_pair_s2"),s3=addr(0,"pi05_pair_s3");
  const uint32_t c0=addr(0,"pi05_pair_residual"),c1=c0+pi05_pair_owner_bytes();const bool gateup_pair=true,carried=carried_;
''' + tail + r'''
 }
};
template<class F>void rejects(F fn){bool failed=false;try{fn();}catch(const std::exception&){failed=true;}assert(failed);}
int main(){
 for(int64_t rows:{272,400})for(bool panel:{false,true}){
  if(panel&&rows!=272)continue;Owner owner;owner.pi05_pair_rows_=rows;owner.pi05_prefill_a8_=true;owner.pi05_prefill_a8_panel512_=panel;
  LayoutContext lc;lc.chunk_size=lc.kv_insert_chunk_size=rows;lc.max_kv_seq_len=2*rows;lc.num_layers=18;lc.use_attn_mask=true;
  const auto decls=owner.declare_buffers(lc);const auto plan=plan_cpu_temporary_layout(decls);owner.offsets=plan.offsets;
  assert(plan.required_extent==size_t(rows==400?7630592:panel?6032640:5049600));
  assert(plan.required_extent+656640<=SpmAllocator::SPM_PLANNING_BUDGET/256*256);
  C=rows;B=rows*4096;Q=owner.addr(0,"residual1");SX=owner.addr(0,"pi05_pair_s1")+B;Y0=owner.addr(0,"pi05_pair_s2");Y1=owner.addr(0,"pi05_pair_s3");O0=owner.addr(0,"pi05_pair_s1");O1=Q;RAW=owner.addr(0,"pi05_pair_residual");
  assert(Y1==Y0+B);memory.assign(256+plan.required_extent,0);
  owner.context.manifest=owner.actual_down_manifest();owner.pi05_down_pair_validate_routes(owner.context.manifest);
  for(int layer:{0,17})for(bool carried:{false,true}){
   if(carried&&layer==17)continue;
   owner.carried_=carried;owner.pi05_o_pair_output_dma_={};owner.pi05_o_pair_carried_layer_=-1;
   put(Y0,B,10);put(Y1,B,20);put(RAW,B/4,30);operations.clear();owner.context.consumed.clear();
   expected_w=&owner.layer_weights_[layer].down_proj_w;expected_sw=&owner.layer_weights_[layer].down_ws;
   owner.emit_pi05_down_a8_linear(layer);owner.actual_consumers(layer);
   assert(operations==(layer==17?std::vector<int>{0,1,2,3,9,4,5,9,6}:carried?std::vector<int>{0,1,2,3,4,5}:std::vector<int>{0,1,2,3,4,5,6}));
   has(Q,B,51);has(RAW,B/4,30);
   // Half 1 may complete while half 0 DMA still reads replicated S1.
   // Preserve every byte of that source until the next KV body's join.
   has(O0,B,50);
   assert(owner.pi05_o_pair_output_dma_.used==(carried?0xfe:0));
   assert(owner.pi05_o_pair_carried_layer_==(carried?layer:-1));
  }
  const auto valid=owner.context.manifest;
  for(int kind:{0,1,2,3}){auto bad=valid;if(kind==0){for(auto&r:bad.routes)if(r.site_id==PI05_DOWN_PAIR_LINEAR_SITE)r.selector=1;}
   if(kind==1){bad.routes.erase(std::remove_if(bad.routes.begin(),bad.routes.end(),[](const auto&r){return r.site_id==PI05_DOWN_A8_QUANT_SITE&&r.invocation==1;}),bad.routes.end());}
   if(kind==2){for(auto&r:bad.routes)if(r.site_id==PI05_DOWN_A8_QUANT_SITE&&r.invocation==1)r.arguments[9]+=32;}
   if(kind==3){for(auto&r:bad.routes)if(r.site_id==PI05_DOWN_PAIR_LINEAR_SITE)r.arguments[7]=0;}
   rejects([&]{owner.pi05_down_pair_validate_routes(bad);});}
  const std::vector<std::string> roots{"residual1","pi05_pair_s1","pi05_pair_s2","pi05_pair_s3","pi05_pair_residual"};
  const auto reject_before_quant=[&]{operations.clear();rejects([&]{owner.emit_pi05_down_a8_linear(0);});assert(operations.empty());owner.addr_overrides.clear();};
  // Disconnected Y roots, every full-root boundary and every pairwise live
  // overlap must fail before the first quantizer can write Q or SX.
  owner.addr_overrides["pi05_pair_s3"]=Y1+256;reject_before_quant();
  for(const auto& root:roots){
   owner.addr_overrides[root]=256+SpmAllocator::SPM_USABLE-256;reject_before_quant();
   owner.addr_overrides[root]=owner.addr(0,root.c_str())+1;reject_before_quant();
  }
  for(size_t i=0;i<roots.size();++i)for(size_t j=i+1;j<roots.size();++j){
   owner.addr_overrides[roots[j]]=owner.addr(0,roots[i].c_str());reject_before_quant();
  }
  owner.pi05_prefill_a8_=false;assert(!owner.pi05_down_a8_enabled()&&owner.pi05_down_pair_linear_selector()==1);rejects([&]{owner.pi05_down_pair_validate_routes(valid);});
 }
 for(int rows:{288,304,320,416,432,448,464}){Owner owner;owner.pi05_pair_rows_=rows;owner.pi05_prefill_a8_=true;assert(!owner.pi05_down_a8_enabled());}
}
'''
    _run_native(tmp_path, prefix + owner, torch,
                extra_sources=(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",))
