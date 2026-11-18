"""Native GR00T VL policy/manifest dispatch with real KV plans and FMB consumer."""
from pathlib import Path
import re

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_gr00t_vl_encoder_model.cpp"


def test_vl_kv_spm_only_actual_manifest_dispatch_and_cache_preservation(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = SOURCE.read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    methods = "\n".join(_function(source, marker).replace(" override", "") for marker in (
        "    FmbPhysicalExecutionManifest physical_manifest_for_candidate(",
        "    std::vector<FmbPhysicalExecutionManifest>\n    physical_manifest_domain_for_candidate(",
        "    bool subclass_spm_kv_by_mha_eligible(", "    bool kv_spm_only_profile(",
        "    AttentionExecutionPolicy attention_policy_for_chunk(",
        "    int64_t kv_insert_flags(", "    int64_t subclass_layout_hash(",
    ))
    reader = _function(source, "bool gr00t_vl_kv_spm_only_opt_in(")
    constructor = _function(source, "    Gr00tVlEncoderModel()")
    constructor = constructor.replace("Gr00tVlEncoderModel", "Owner")
    constants = "\n".join(re.findall(r"constexpr (?:int64_t|uint32_t) GR00T_VL_\w+ =[^;]+;", source))
    capability = "\n".join(_function(helpers, marker) for marker in (
        "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline bool sdpa_by_mha_spm_is_valid(",
    ))
    vlm_limit, = re.findall(r"^#define SDPA_MAX_VLM_SIZE .+$", helpers, re.M)
    block = _function(source, "    void build_layer_subgraph(")
    dispatch = block[block.index("        // ---- KV-insert@0"):
                     block.index("        // ---- o-proj")]
    cpp = r'''
#include "core/fused_model_base.h"
#include "core/rpu_profile.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using namespace v3;
#define NUM_CORES 8
#define DWIDTH 2
__VLM_LIMIT__
__CAPABILITY__
__CONSTANTS__
__READER__
struct HostContext {
 int64_t seq_len=82,batch_size=1,position=0;bool is_causal=false;
 std::optional<int> attention_mask;
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
 FmbThreeStageChunkPlan stage_plan;
 FmbPhysicalManifestConsumer consumer;
 std::vector<int>*k_caches=nullptr,*v_caches=nullptr;
 bool has_complete_physical_manifest()const{return consumer.complete();}
 const auto&physical_manifest()const{return consumer.manifest();}
 const auto&find_physical_route(FmbRouteFamily family,int64_t site,int64_t invocation)const{
  return consumer.find_route(family,site,invocation);
 }
 const auto&consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,
                                  at::IntArrayRef args,int64_t invocation)const{
  return consumer.consume_route(family,site,selector,flags,args,invocation);
 }
 auto physical_attention_policy_for_site(int64_t site,int64_t invocation)const{
  return consumer.attention_policy_for_site(site,invocation);
 }
};
int writes=0,transposes=0,raw_attentions=0,ddr_attentions=0,memsets=0;
std::vector<uint16_t> k_spm(96*256),v_spm(96*256);
void rpu_launch_memset_spm_multicore(uint32_t,int64_t){++memsets;}
void rpu_launch_insert_kvcache_spm_unified_with_plan(
 int&k,int&v,uint32_t ka,uint32_t va,int64_t nq,int64_t hd,int cores,
 int64_t ko,int64_t vo,int64_t rows,const KvInsertSegmentPlan&plan){
 assert(ka==0x20000&&va==0x40000&&nq==32&&hd==64&&cores==8&&ko==0&&vo==0&&rows==0);
 assert(plan.logical_rows()==plan.physical_rows()&&plan.segment(0).position==0);
 // The production mirror kernels read SPM and write DDR only.
 ++writes;++k;++v;
}
void rpu_launch_v_transpose_spm(uint32_t input,uint32_t output,int64_t batch,int64_t seq,
                              int64_t nq,int64_t hd,int cores){
 assert(input==0x10040000&&output==0x10060000&&batch==1&&seq>0&&nq==32&&hd==64&&cores==8);++transposes;
}
void rpu_launch_sdpa_by_mha_spm(uint32_t q,uint32_t k,uint32_t v,uint32_t out,uint32_t mask,
 int mask_kind,double scale,int64_t batch,int64_t sq,int64_t sk,int64_t nq,int64_t nk,int64_t hd,int cores){
 assert(q==0x10010000&&k==0x10020000&&v==0x10060000&&out==0x10080000&&mask==0&&mask_kind==0);
 assert(scale==.125&&batch==1&&sq==sk&&nq==32&&nk==32&&hd==64&&cores==8);++raw_attentions;
}
void rpu_launch_sdpa_spm_unified_kernel_v2(int&k,int&v,int mask_kind,double scale,
 uint32_t q,uint32_t out,uint32_t tmp,uint32_t mask,int64_t seq,int64_t nq,int64_t nk,int64_t hd,
 int64_t kv_seq,int qc,int kc){
 assert(k==v&&k>0&&mask_kind==0&&scale==.125&&q==0x10000&&out==0x80000&&tmp==0x60000&&mask==0);
 assert(seq==kv_seq&&nq==32&&nk==32&&hd==64&&qc==8&&kc==8);++ddr_attentions;
}
namespace v3 {
struct Owner {
 const bool kv_spm_only_opt_in_;
 bool z2_bound_=false;int64_t z2_prepared_seq_len_=0;
 int64_t layers=4,h=2048,ff=8192,q=32,kv=32,hd=64,tp=8,cores=NUM_CORES;
 std::vector<int> layer_weights_=std::vector<int>(4);
 std::vector<int> k_cache=std::vector<int>(4),v_cache=std::vector<int>(4);
 HostContext context;
 __CONSTRUCTOR__
 HostContext&ctx(){return context;}
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return h;}
 int64_t intermediate_size()const{return ff;}int64_t num_q_heads()const{return q;}
 int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}int attn_tp()const{return tp;}int num_cores()const{return cores;}int mlp_tp()const{return cores;}
 KvInsertSegmentPlan resolve_kvinsert_plan_auto(int64_t,FmbGraphLifecycle,int64_t p,int64_t l,
     int64_t physical,int cores,int64_t heads,int64_t dim,uint32_t caps)const{
  return rpu_resolve_kvinsert_segment_plan_auto(p,l,physical,cores,heads,dim,caps);
 }
 KvInsertSegmentPlan restore_kvinsert_plan(int64_t,at::IntArrayRef args,int cores,int64_t heads,int64_t dim)const{
  return rpu_kvinsert_segment_plan_from_route_arguments(args,cores,heads,dim);
 }
 uint32_t addr(int,const char*n)const{
  const std::string name=n;return name=="q"?0x10010000:name=="k"?0x10020000:name=="v"?0x10040000:
   name=="sdpa_tmp"?0x10060000:0x10080000;
 }
 struct Offset{uint32_t value;};Offset addr_offset(const char*n)const{return {addr(0,n)-0x10000000};}
 __METHODS__
 LayoutContext layout(bool raw)const{LayoutContext result;result.chunk_size=((context.seq_len+15)/16)*16;result.batch_size=context.batch_size;
  result.is_causal=context.is_causal;result.use_attn_mask=context.attention_mask.has_value();
  result.attention_policy=raw?AttentionExecutionPolicy::SPM_KV_BY_MHA:AttentionExecutionPolicy::DDR_KV;return result;}
 void plan(int64_t seq=82){context.seq_len=seq;context.position=0;
  context.stage_plan=compose_fmb_three_stage_chunk_plan({{0,0,seq,seq}},{{0,0,seq,seq}},
      {{0,0,seq,seq}},{{0,seq,-1}},ChunkMode::SEQUENTIAL);}
 auto manifest(bool raw)const{return physical_manifest_for_candidate(context.stage_plan,layout(raw),context.seq_len,context.seq_len,0);}
 void record(bool raw){context.k_caches=&k_cache;context.v_caches=&v_cache;context.consumer.reset(manifest(raw));
  // Deliberately oppose the actual route: union layout cannot choose the writer.
  context.attention_policy=raw?AttentionExecutionPolicy::DDR_KV:AttentionExecutionPolicy::SPM_KV_BY_MHA;}
 void run(int layer_idx=0){const ChunkInfo&chunk=context.stage_plan.compute.chunks.front();
  const int64_t seq_len=chunk.len,nq=num_q_heads(),hd=head_dim();
  __DISPATCH__
 }
};
}
template<class M>auto kv_route(M&manifest){
 return std::find_if(manifest.routes.begin(),manifest.routes.end(),[](const auto&r){return r.site_id==GR00T_VL_KV_INSERT_SITE;});}
template<class M>auto attn_route(M&manifest){
 return std::find_if(manifest.routes.begin(),manifest.routes.end(),[](const auto&r){return r.family==FmbRouteFamily::ATTENTION;});}
template<class F>void rejected(F call){const auto before=std::make_tuple(writes,transposes,raw_attentions,ddr_attentions,memsets);
 bool caught=false;try{call();}catch(const c10::Error&){caught=true;}
 assert(caught&&before==std::make_tuple(writes,transposes,raw_attentions,ddr_attentions,memsets));}
int main(){
 unsetenv("RPU_GR00T_VL_KV_SPM_ONLY");Owner off;off.plan();assert(!off.kv_spm_only_opt_in_&&off.subclass_layout_hash()==0);
 for(const char*s:{"","0"}){setenv("RPU_GR00T_VL_KV_SPM_ONLY",s,1);Owner x;assert(!x.kv_spm_only_opt_in_);}
 for(const char*s:{"true","2","-1"," 1","1 "}){setenv("RPU_GR00T_VL_KV_SPM_ONLY",s,1);rejected([]{Owner x;});}
 setenv("RPU_GR00T_VL_KV_SPM_ONLY","1",1);Owner on;on.plan();unsetenv("RPU_GR00T_VL_KV_SPM_ONLY");
 assert(on.kv_spm_only_opt_in_&&!off.kv_spm_only_opt_in_&&on.subclass_layout_hash()!=0);
 for(int64_t seq:{1,82,97,169,214,256,193,225,289,238,270,334})for(Owner*owner:{&off,&on})for(bool raw:{false,true}){
  off.plan(seq);on.plan(seq);k_spm.resize(((seq+15)/16)*16*256);v_spm.resize(k_spm.size());
  if(seq==169)assert(owner->layout(raw).chunk_size==176&&k_spm.size()==176*256);
  const auto domain=owner->physical_manifest_domain_for_candidate(owner->context.stage_plan,owner->layout(raw),seq,seq,0);
  assert(domain.size()==2);auto selected=domain.at(raw?1:0);
  assert(kv_route(selected)->flags==(raw&&owner->kv_spm_only_opt_in_?2:1));
  assert(selected.kv_logical_length==seq&&selected.kv_insert_physical_rows==seq);
  auto base=off.manifest(raw);assert(kv_route(selected)->arguments==kv_route(base)->arguments);
  for(int replay=0;replay<3;++replay){owner->record(raw);
   for(int layer=0;layer<4;++layer){for(size_t i=0;i<k_spm.size();++i){k_spm[i]=uint16_t(i+1+replay);v_spm[i]=uint16_t(i+7+layer);}
    const auto k_before=k_spm,v_before=v_spm;const int kw=owner->k_cache[layer],vw=owner->v_cache[layer];
    const int w=writes,t=transposes,r=raw_attentions,d=ddr_attentions;owner->run(layer);
    const int mirror=raw&&owner->kv_spm_only_opt_in_?0:1;
    assert(writes==w+mirror&&owner->k_cache[layer]==kw+mirror&&owner->v_cache[layer]==vw+mirror);
    assert(transposes==t+raw&&raw_attentions==r+raw&&ddr_attentions==d+!raw);
    // Preserve the original producer/transpose handling of each padded slab:
    // 82/96, 169/176 or 214/224; logical rows and existing tails gain no clear.
    assert(k_spm==k_before&&v_spm==v_before&&memsets==0);
   }
  }
 }
 for(int64_t seq:{81,96,128,168,170,213,215}){on.plan(seq);on.record(true);assert(kv_route(on.context.consumer.manifest())->flags==2);}
 // Nine by-MHA query tiles exceed the actual kernel sync capability;
 // candidate generation retains DDR, independently of later SPM admission.
 on.plan(641);const auto outside=on.physical_manifest_domain_for_candidate(on.context.stage_plan,on.layout(false),641,641,0);
 assert(outside.size()==1&&kv_route(outside.front())->flags==1);rejected([&]{on.record(true);});
 on.record(false);const int previous_writes=writes;on.run();assert(writes==previous_writes+1);
 // Supported lengths require their own consumed KV geometry, including stride176.
 for(auto lengths:{std::pair{82,214},std::pair{214,82},std::pair{169,214},std::pair{214,169}}){on.plan(lengths.first);on.record(true);
  const auto stale=on.context.consumer.manifest();on.plan(lengths.second);on.context.consumer.reset(stale);rejected([&]{on.run();});}
 on.plan();
 {Owner m4=on;m4.cores=4;m4.tp=4;assert(m4.kv_spm_only_profile(m4.context.stage_plan,m4.layout(true),0));}
 for(int bad=0;bad<14;++bad){Owner x=on;switch(bad){case 0:x.z2_bound_=true;break;case 1:x.z2_prepared_seq_len_=82;break;
  case 2:x.tp=4;break;case 3:x.layers=3;break;case 4:x.layer_weights_.resize(3);break;case 5:x.h=1536;break;
  case 6:x.ff=6144;break;case 7:x.q=16;break;case 8:x.kv=16;break;case 9:x.hd=48;break;
  case 10:x.context.batch_size=2;break;case 11:x.context.is_causal=true;break;case 12:x.context.attention_mask=1;break;
  case 13:x.context.stage_plan.qkv.chunks[0].len=81;break;}
  assert(!x.kv_spm_only_profile(x.context.stage_plan,x.layout(true),0));
  auto ddr=x.manifest(false);assert(kv_route(ddr)->flags==1);
 }
 assert(!on.kv_spm_only_profile(on.context.stage_plan,on.layout(true),1));
 for(bool raw:{false,true})for(int bad=0;bad<7;++bad){on.record(raw);auto corrupted=on.context.consumer.manifest();
  switch(bad){case 0:kv_route(corrupted)->flags=raw?1:2;break;
   case 1:kv_route(corrupted)->arguments[2]=81;break;
   case 2:kv_route(corrupted)->arguments[3]=96;break;
   case 3:kv_route(corrupted)->arguments[6]=1;break;
   case 4:attn_route(corrupted)->selector=static_cast<int64_t>(raw?AttentionExecutionPolicy::DDR_KV:AttentionExecutionPolicy::SPM_KV_BY_MHA);break;
   case 5:corrupted.routes.erase(attn_route(corrupted));break;
   case 6:corrupted.routes.push_back(*attn_route(corrupted));break;}
  on.context.consumer.reset(corrupted);rejected([&]{on.run();});
 }
 on.record(true);auto wrong_invocation=on.context.consumer.manifest();attn_route(wrong_invocation)->invocation=1;
 on.context.consumer.reset(wrong_invocation);rejected([&]{on.run();});
 on.record(true);on.context.seq_len=81;rejected([&]{on.run();});on.context.seq_len=82;
 FmbPhysicalExecutionManifest absent;on.context.consumer.reset(absent);
 assert(on.kv_insert_flags(on.context.stage_plan.compute.chunks.front(),AttentionExecutionPolicy::SPM_KV_BY_MHA)==1);
 std::cout<<"GR00T VL KV SPM-only policy/descriptor/cache contract: PASS\n";
}
'''
    for marker, value in {"__VLM_LIMIT__": vlm_limit, "__CAPABILITY__": capability,
                          "__CONSTANTS__": constants, "__READER__": reader,
                          "__CONSTRUCTOR__": constructor, "__METHODS__": methods,
                          "__DISPATCH__": dispatch}.items():
        cpp = cpp.replace(marker, value)
    result = _run_native(tmp_path, cpp, torch, extra_sources=(
        ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",
        ROOT / "src/ops/rpu_kvinsert_segment_plan.cpp",
    ))
    assert "GR00T VL KV SPM-only policy/descriptor/cache contract: PASS" in result


def test_vl_kv_spm_only_cold_identity_and_original_spm_contract():
    source = SOURCE.read_text()
    assert "const bool kv_spm_only_opt_in_;" in source
    identity = _function(source, "    std::vector<int64_t> kvinsert_cost_weight_identity(")
    layout_hash = _function(source, "    int64_t subclass_layout_hash(")
    for body in (identity, layout_hash):
        assert "kv_spm_only_opt_in_" in body and "GR00T_VL_KV_FLAG_RAW_SPM_ONLY" in body
    block = _function(source, "    void build_layer_subgraph(")
    assert block.index("attention_policy_for_chunk(") < block.index("restore_kvinsert_plan(")
    assert block.index("restore_kvinsert_plan(") < block.index("const int64_t kv_flags")
    assert block.index("const int64_t kv_flags") < block.index("rpu_launch_insert_kvcache_spm_unified_with_plan(")
    declarations = _function(source, "    std::vector<BufferDecl> declare_buffers(")
    assert "raw_spm ? Align(cs, int64_t{16}) : cs" in declarations
    assert 'row_bias("out_bias"' in declarations and 'row_bias("ff2_bias"' in declarations
    assert "rpu_launch_memset_spm_multicore(core0_addr, hidden_size(), num_cores())" in declarations
    # Omitting DDR-only writers cannot remove an SPM tail clear. The preserved
    # preload memset calls initialize row biases and are unrelated to K/V tails.
    assert "rpu_launch_memset_spm_multicore(" not in block
