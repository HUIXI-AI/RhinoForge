"""Execute GR00T's actual KV flag consumer, padding and write guards on host."""
from pathlib import Path
import re

from test_gr00t_adaln_table import _run
from test_siglip_attention_dry_live_contract import _function

SOURCE = Path(__file__).resolve().parents[1] / "src/fused/rpu_gr00t_dit_model.cpp"


def test_kv_spm_only_uses_actual_attention_and_preserves_tail_and_ddr_fallback(tmp_path):
    source = SOURCE.read_text()
    methods = "\n".join(_function(source, marker) for marker in (
        "    bool kv_spm_only_profile(", "    int64_t kv_insert_flags(",
        "    KvInsertSegmentPlan consume_kv_insert_plan(",
    ))
    constants = "\n".join(re.findall(r"constexpr int64_t GR00T_DIT_(?:(?:SELF|CROSS)_(?:KV_INSERT|RAW_ATTENTION|DDR_ATTENTION)_SITE|KV_FLAG_\w+) = \d+(?:LL)?;", source))
    reader = _function(source, "bool gr00t_kv_spm_only_opt_in()")
    self_block = _function(source, "    void build_self_attn_block(")
    cross_block = _function(source, "    void build_cross_attn_block(")
    self_guard = _function(self_block, "        if (kv_insert_flags(")
    cross_guard = _function(cross_block, "            if (kv_insert_flags(")
    selection = self_block[self_block.index("        int64_t attention_site ="):self_block.index("        if (attention_policy ==")]
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do{if(!(ok))throw std::runtime_error(#ok);}while(false)
#define DWIDTH 2
#define NUM_CORES 8
__CONSTANTS__
__READER__
int64_t gr00t_pad16(int64_t n){return (n+15)/16*16;}
enum class FmbRouteFamily{ATTENTION,KV_INSERT};
enum class AttentionExecutionPolicy:int64_t{DDR_KV=1,SPM_KV_BY_MHA=2};
struct ChunkInfo{int64_t len=41,idx=0;};
struct KvInsertSegmentPlan{
 int64_t logical,physical,position;struct Segment{int64_t position;};
 int64_t logical_rows()const{return logical;}int64_t physical_rows()const{return physical;}
 int64_t route()const{return 16;}Segment segment(int)const{return {position};}
};
KvInsertSegmentPlan restore_kvinsert_plan(int64_t,const std::vector<int64_t>&a,int64_t tp,int64_t nq,int64_t hd){assert((tp==4||tp==8)&&nq==32&&hd==48);return {a[0],a[1],a[2]};}
struct Route{FmbRouteFamily family;int64_t site_id,invocation,flags,selector;std::vector<int64_t>arguments;};
struct Context{
 bool complete=true;int64_t seq_len=41,batch_size=1,body_iter=0;std::vector<Route>routes;int consumed=0;
 // Deliberately keep the union layout raw while a particular site is DDR.
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
 bool has_complete_physical_manifest()const{return complete;}const Context&physical_manifest()const{return *this;}
 const Route&find_physical_route(FmbRouteFamily family,int64_t site,int64_t invocation)const{
  for(const auto&r:routes)if(r.family==family&&r.site_id==site&&r.invocation==invocation)return r;throw std::runtime_error("missing route");
 }
 AttentionExecutionPolicy physical_attention_policy_for_site(int64_t site,int64_t invocation)const{return static_cast<AttentionExecutionPolicy>(find_physical_route(FmbRouteFamily::ATTENTION,site,invocation).selector);}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,const std::vector<int64_t>&args,int64_t invocation){
  const auto&r=find_physical_route(family,site,invocation);TORCH_CHECK(r.selector==selector&&r.flags==flags&&r.arguments==args);++consumed;
 }
};
std::vector<std::pair<uint32_t,int64_t>>zeros;int writes=0;
void rpu_launch_memset_spm_multicore(uint32_t address,int64_t elements,int=8){zeros.push_back({address,elements});}
void rpu_launch_insert_kvcache_spm_unified_with_plan(int&k,int&v,uint32_t ka,uint32_t va,int64_t nq,int64_t hd,int cores,int64_t ko,int64_t vo,int64_t rows,const KvInsertSegmentPlan&plan){
 assert(nq==32&&hd==48&&cores==8&&ko==0&&vo==0&&rows==plan.physical_rows()&&zeros.size()>=2);assert(ka!=va);++writes;++k;++v;
}
struct Model{
 const bool kv_spm_only_opt_in_=gr00t_kv_spm_only_opt_in();
 bool denoise_ready_=true,denoise_mode_=true,loop_mode_=true,z2_bound_=false;
 int64_t z2_prepared_seq_len_=0,num_steps_=4,action_horizon_=40,cross_dim_=2048,vl_seq_=82;
 int64_t layers=32,h=1536,ff=6144,q=32,kv=32,hd=48,tp=8;
 Context context;int self_k=0,self_v=0,cross_k=0,cross_v=0;
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return ff;}
 int64_t num_q_heads()const{return q;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}int64_t attn_tp()const{return tp;}int num_cores()const{return NUM_CORES;}
 Context&ctx(){return context;}uint32_t addr(int,const char*n)const{return std::string(n)=="k"?0x10020000:0x10040000;}
 struct Offset{uint32_t value;};Offset addr_offset(const char*n)const{return {std::string(n)=="k"?0x20000:std::string(n)=="v"?0x40000:std::string(n)=="cross_k_bank"?0x60000u:0x80000u};}
 __METHODS__
 void record(bool self_raw,bool cross_raw){
  context.routes.clear();
  for(bool cross:{false,true}){const bool raw=cross?cross_raw:self_raw;const int64_t n=cross?vl_seq_:41;
   context.routes.push_back({FmbRouteFamily::ATTENTION,cross?(raw?GR00T_DIT_CROSS_RAW_ATTENTION_SITE:GR00T_DIT_CROSS_DDR_ATTENTION_SITE):(raw?GR00T_DIT_SELF_RAW_ATTENTION_SITE:GR00T_DIT_SELF_DDR_ATTENTION_SITE),0,0,raw?2:1,{}});
   context.routes.push_back({FmbRouteFamily::KV_INSERT,cross?GR00T_DIT_CROSS_KV_INSERT_SITE:GR00T_DIT_SELF_KV_INSERT_SITE,0,raw&&context.complete&&kv_spm_only_profile(context.seq_len,context.batch_size)?2:1,16,{n,gr00t_pad16(n),0}});
  }
 }
 AttentionExecutionPolicy self_attention(){ChunkInfo chunk;__SELECTION__ return attention_policy;}
 void self(){ChunkInfo chunk;int&k_cache=self_k;int&v_cache=self_v;const int64_t nq=32,hd=48;
  const auto kv_plan=consume_kv_insert_plan(GR00T_DIT_SELF_KV_INSERT_SITE,41,chunk.idx);
  __SELF_GUARD__
 }
 void cross(){ChunkInfo chunk;int&k_cache=cross_k;int&v_cache=cross_v;const int64_t nq=32,hd=48;
  const char*k_slot="cross_k_bank";const char*v_slot="cross_v_bank";const uint32_t bank_offset=1536,v_offset=bank_offset;
  const auto kv_plan=consume_kv_insert_plan(GR00T_DIT_CROSS_KV_INSERT_SITE,vl_seq_,chunk.idx,0x10060000+bank_offset,0x10080000+bank_offset);
  __CROSS_GUARD__
 }
};
int main(){
 unsetenv("RPU_GR00T_KV_SPM_ONLY");setenv("RPU_GR00T_CROSS_KV_REUSE","1",1);Model off;off.record(true,true);
 assert(off.kv_insert_flags(GR00T_DIT_SELF_KV_INSERT_SITE,0)==1);off.self();off.cross();assert(writes==2);
 for(const char*value:{"","0"}){setenv("RPU_GR00T_KV_SPM_ONLY",value,1);Model x;assert(!x.kv_spm_only_opt_in_);}
 for(const char*value:{"true","2","-1"," 1"}){setenv("RPU_GR00T_KV_SPM_ONLY",value,1);try{Model x;assert(false);}catch(const std::runtime_error&){} }
 setenv("RPU_GR00T_KV_SPM_ONLY","1",1);Model model;unsetenv("RPU_GR00T_KV_SPM_ONLY");assert(model.kv_spm_only_opt_in_&&!off.kv_spm_only_opt_in_);
 for(auto profile:{std::pair{4,82},std::pair{10,214}})for(bool self_raw:{false,true})for(bool cross_raw:{false,true}){
  model.num_steps_=profile.first;model.vl_seq_=profile.second;const int64_t S=model.vl_seq_;
  model.record(self_raw,cross_raw);const int before=writes;
  assert(model.self_attention()==(self_raw?AttentionExecutionPolicy::SPM_KV_BY_MHA:AttentionExecutionPolicy::DDR_KV));
  for(int replay=0;replay<3;++replay)for(int body=0;body<model.num_steps_;++body){model.context.body_iter=body;zeros.clear();model.self();model.cross();
   assert(zeros.size()==4&&zeros[0]==std::make_pair(uint32_t(0x10020000+41*192*2),int64_t(7*192)));
   assert(zeros[1].first==0x10040000+41*192*2&&zeros[2].first==0x10060000+1536+S*192*2&&zeros[2].second==(gr00t_pad16(S)-S)*192&&zeros[3].first==0x10080000+1536+S*192*2);
  }
  assert(writes-before==3*model.num_steps_*((self_raw?0:1)+(cross_raw?0:1)));
 }
 // A cross-KV descriptor for the other prefix cannot reach tail clearing or DDR writes.
 for(auto lengths:{std::pair{82,214},std::pair{214,82}}){model.vl_seq_=lengths.first;model.record(true,true);model.vl_seq_=lengths.second;
  const int before=writes;zeros.clear();try{model.cross();assert(false);}catch(const std::runtime_error&){}assert(writes==before&&zeros.empty());}
 model.num_steps_=4;model.vl_seq_=82;
 model.record(true,true);const int old=model.self_k;model.self();assert(model.self_k==old);
 model.record(false,false);model.self();model.cross();assert(model.self_k==old+1&&model.self_k==model.self_v&&model.cross_k==model.cross_v);
 for(int bad=0;bad<10;++bad){Model x=model;switch(bad){case 0:x.loop_mode_=false;break;case 1:x.denoise_mode_=false;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:x.context.seq_len=48;break;case 5:x.context.complete=false;break;case 6:x.action_horizon_=39;break;case 7:x.context.batch_size=2;break;case 8:x.tp=2;break;case 9:x.num_steps_=3;break;}x.record(true,true);assert(x.kv_insert_flags(GR00T_DIT_SELF_KV_INSERT_SITE,0)==1&&x.kv_insert_flags(GR00T_DIT_CROSS_KV_INSERT_SITE,0)==1);}
 {Model m4=model;m4.tp=4;m4.record(true,true);assert(m4.kv_insert_flags(GR00T_DIT_SELF_KV_INSERT_SITE,0)==2&&m4.kv_insert_flags(GR00T_DIT_CROSS_KV_INSERT_SITE,0)==2);}
 for(int bad=0;bad<6;++bad){Model x=model;x.record(true,true);auto&route=x.context.routes[1];switch(bad){case 0:route.flags=1;break;case 1:route.arguments[0]=40;break;case 2:route.arguments[1]=64;break;case 3:route.arguments[2]=1;break;case 4:x.context.routes[0].selector=1;break;case 5:route.invocation=1;break;}
  const int before=writes;zeros.clear();try{x.self();assert(false);}catch(const std::runtime_error&){}assert(writes==before&&zeros.empty());
 }
}
'''.replace("__CONSTANTS__", constants).replace("__READER__", reader).replace("__METHODS__", methods).replace("__SELF_GUARD__", self_guard).replace("__CROSS_GUARD__", cross_guard).replace("__SELECTION__", selection)
    _run(tmp_path, cpp)


def test_kv_spm_only_keeps_manifest_geometry_and_independent_cold_policy():
    source = SOURCE.read_text()
    manifest = _function(source, "    FmbPhysicalExecutionManifest physical_manifest_for_attention_routes(")
    assert "append_kv(GR00T_DIT_SELF_KV_INSERT_SITE, chunk.len, self_raw_spm)" in manifest
    assert "append_kv(GR00T_DIT_CROSS_KV_INSERT_SITE, vl_seq_, cross_raw_spm)" in manifest
    assert "raw_attention && kv_spm_only_profile(chunk.len, layout.batch_size)" in manifest
    consume = _function(source, "    KvInsertSegmentPlan consume_kv_insert_plan(")
    assert consume.index("restore_kvinsert_plan(") < consume.index("plan.logical_rows() == true_seq") < consume.index("ctx().consume_physical_route(") < consume.index("rpu_launch_memset_spm_multicore(")
    assert "kv_insert_flags(site_id, invocation)" in consume
    profile = _function(source, "    bool kv_spm_only_profile(")
    assert "cross_kv_reuse_opt_in_" not in profile and "direct_spm" not in profile
    assert "const bool kv_spm_only_opt_in_;" in source
    layout = _function(source, "    int64_t subclass_layout_hash(")
    assert "kv_spm_only_opt_in_ ? 1 : 0" in layout
    cross = _function(source, "    void build_cross_attn_block(")
    assert cross.index("cross_kv_projection_needed(") < cross.index("consume_kv_insert_plan(") < cross.index("if (kv_insert_flags(")
