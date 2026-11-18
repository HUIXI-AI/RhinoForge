"""Native policy, operand and transpose-guard execution with emulated SPM."""
from pathlib import Path
import re

from test_gr00t_adaln_table import _run
from test_siglip_attention_dry_live_contract import _function

SOURCE = Path(__file__).resolve().parents[1] / "src/fused/rpu_gr00t_dit_model.cpp"


def test_cross_v_transpose_reuse_refreshes_bank_and_keeps_ddr_independent(tmp_path):
    source = SOURCE.read_text()
    methods = "\n".join(_function(source, marker) for marker in (
        "    bool cross_kv_reuse_profile(", "    int64_t cross_kv_schedule(",
        "    int64_t cross_kv_bank_stride(", "    int64_t cross_kv_bank_bytes(",
        "    uint32_t cross_kv_bank_offset(", "    void append_cross_kv_buffers(",
        "    std::vector<int64_t> cross_kv_schedule_arguments(", "    bool cross_kv_projection_needed(",
        "    bool cross_v_transpose_profile(", "    std::vector<int64_t> cross_v_transpose_arguments(",
        "    bool cross_v_transpose_reused(", "    bool kv_spm_only_profile(", "    int64_t kv_insert_flags(",
    ))
    constants = "\n".join(re.findall(r"constexpr int64_t GR00T_DIT_(?:CROSS_(?:KV|V_TRANSPOSE)_SCHEDULE_SITE|(?:SELF|CROSS)_(?:KV_INSERT|RAW_ATTENTION|DDR_ATTENTION)_SITE|KV_FLAG_\w+) = \d+(?:LL)?;", source))
    constants += "\n" + re.search(r"enum class Gr00tDiTCrossKvSchedule.*?\n};", source, re.S)[0]
    readers = "\n".join(_function(source, f"bool {name}()") for name in (
        "gr00t_cross_kv_reuse_opt_in", "gr00t_kv_spm_only_opt_in", "gr00t_cross_v_transpose_reuse_opt_in"))
    cross = _function(source, "    void build_cross_attn_block(")
    operands = cross[cross.index("        const bool raw_bank ="):cross.index("        auto& k_cache =")]
    transpose = cross[cross.index("            const uint32_t transposed_v ="):cross.index("            rpu_launch_sdpa_by_mha_spm(")]
    mirror = _function(cross, "            if (kv_insert_flags(")
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do{if(!(ok))throw std::runtime_error(#ok);}while(false)
#define DWIDTH 2
#define NUM_CORES 8
int64_t Align(int64_t n,int64_t a){return (n+a-1)/a*a;}int64_t gr00t_pad16(int64_t n){return Align(n,16);}
struct SpmAllocator{static constexpr int64_t SPM_USABLE=8387584;};
enum class AttentionExecutionPolicy:int64_t{DDR_KV=1,SPM_KV_BY_MHA=2};
enum class FmbRouteFamily{GRAPH_SCHEDULE,ATTENTION};
enum class StorageClass{Temp};enum class BufferScope{LayerWide};
struct BufferDecl{const char*name;int64_t size;int begin,end;StorageClass storage;int per_layer;void*preload;BufferScope scope;};
struct LayoutContext{int64_t chunk_size=41,batch_size=1;AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;};
__CONSTANTS__
__READERS__
struct ChunkInfo{int64_t len=41,idx=0;};
struct Route{FmbRouteFamily family;int64_t site_id,invocation,selector;std::vector<int64_t>arguments;};
struct Context{
 bool complete=true;int64_t batch_size=1,body_iter=0,seq_len=41;std::vector<Route>routes;mutable int consumed=0;
 bool has_complete_physical_manifest()const{return complete;}const Context&physical_manifest()const{return *this;}
 const Route&find(FmbRouteFamily f,int64_t site,int64_t invocation)const{for(const auto&r:routes)if(r.family==f&&r.site_id==site&&r.invocation==invocation)return r;throw std::runtime_error("missing route");}
 AttentionExecutionPolicy physical_attention_policy_for_site(int64_t site,int64_t invocation)const{return static_cast<AttentionExecutionPolicy>(find(FmbRouteFamily::ATTENTION,site,invocation).selector);}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selection,int64_t flags,const std::vector<int64_t>&args,int64_t invocation)const{
  const auto&r=find(f,site,invocation);TORCH_CHECK(flags==0&&selection==r.selector&&args==r.arguments);++consumed;
 }
};
std::map<uint32_t,std::vector<int>>spm;int transposes=0,mirrors=0,projections=0;uint32_t expected_v=0,expected_k=0;
struct Plan{int64_t rows;int64_t physical_rows()const{return rows;}};
void rpu_launch_insert_kvcache_spm_unified_with_plan(int&,int&,uint32_t k,uint32_t v,int64_t nq,int64_t hd,int cores,int64_t ko,int64_t vo,int64_t rows,const Plan&p){
 assert(k+0x10000000==expected_k&&v+0x10000000==expected_v&&nq==32&&hd==48&&cores==8&&ko==0&&vo==0&&rows==p.rows);assert(spm.count(expected_v));++mirrors;
}
void rpu_launch_v_transpose_spm(uint32_t input,uint32_t output,int64_t batch,int64_t s,int64_t nq,int64_t d,int cores){
 const int64_t pad=gr00t_pad16(s);assert(input==expected_v&&input!=output&&batch==1&&nq==32&&d==48&&cores==8&&spm[input].size()==pad*192);
 auto&out=spm[output];out.resize(pad*192);for(int h=0;h<4;++h)for(int c=0;c<48;++c)for(int r=0;r<pad;++r)out[(h*48+c)*pad+r]=spm[input][r*192+h*48+c];++transposes;
}
struct Model{
 const bool cross_kv_reuse_opt_in_=gr00t_cross_kv_reuse_opt_in(),kv_spm_only_opt_in_=gr00t_kv_spm_only_opt_in(),cross_v_transpose_reuse_opt_in_=gr00t_cross_v_transpose_reuse_opt_in();
 bool denoise_ready_=true,denoise_mode_=true,loop_mode_=true,z2_bound_=false;
 int64_t num_steps_=4,z2_prepared_seq_len_=0,cross_dim_=2048,vl_seq_=82,action_horizon_=40;
 int64_t layers=32,h=1536,ff=6144,q=32,kv=32,hd=48,tp=8;
 struct LayerWeights{bool is_cross;};std::vector<LayerWeights>layer_weights_=std::vector<LayerWeights>(32);
 Model(){for(int l=0;l<32;++l)layer_weights_[l].is_cross=l%2==0;}
 Context context;Context&ctx(){return context;}const Context&ctx()const{return context;}
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return ff;}
 int64_t num_q_heads()const{return q;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}int64_t attn_tp()const{return tp;}int num_cores()const{return NUM_CORES;}
 uint32_t addr(int,const char*n)const{const std::string name=n;return name=="cross_k_bank"?0x10200000:name=="cross_v_bank"?0x10400000:name=="k"?0x10020000:name=="v"?0x10040000:0x10060000;}
 struct Offset{uint32_t value;};Offset addr_offset(const char*n)const{return {addr(0,n)-0x10000000};}
 __METHODS__
 void record(const ChunkInfo&chunk,bool raw=true){context.routes={
  {FmbRouteFamily::GRAPH_SCHEDULE,GR00T_DIT_CROSS_KV_SCHEDULE_SITE,0,cross_kv_schedule(chunk.len,context.batch_size,raw),cross_kv_schedule_arguments(chunk.len,context.batch_size)},
  {FmbRouteFamily::GRAPH_SCHEDULE,GR00T_DIT_CROSS_V_TRANSPOSE_SCHEDULE_SITE,0,cross_v_transpose_profile(chunk.len,context.batch_size,raw)?2:1,cross_v_transpose_arguments(chunk.len,context.batch_size,raw)},
  {FmbRouteFamily::ATTENTION,raw?GR00T_DIT_CROSS_RAW_ATTENTION_SITE:GR00T_DIT_CROSS_DDR_ATTENTION_SITE,0,raw?2:1,{}}};}
 uint32_t run(const ChunkInfo&chunk,int layer_idx,AttentionExecutionPolicy attention_policy,int request){
  const int64_t seq_len=chunk.len,S=vl_seq_,nq=32,hd=48;
  __OPERANDS__
  expected_v=v_spm;expected_k=k_spm;
  if(transposed_bank){assert(v_spm==addr(0,"v")&&v_offset==0&&raw_bank);}
  if(cross_kv_projection_needed(chunk,attention_policy)){
   auto&v=spm[v_spm];v.resize(gr00t_pad16(S)*192);for(int r=0;r<gr00t_pad16(S);++r)for(int c=0;c<192;++c)v[r*192+c]=r<S?request*100000+layer_idx*1000+r*192+c:0;++projections;
   int k_cache=0,v_cache=0;const Plan kv_plan{gr00t_pad16(S)};
   __MIRROR__
  }
  if(attention_policy==AttentionExecutionPolicy::DDR_KV)return 0;
  __TRANSPOSE__
  if(transposed_bank)assert(transposed_v==addr(0,"cross_v_bank")+bank_offset&&transposed_v+cross_kv_bank_stride()<=addr(0,"cross_v_bank")+cross_kv_bank_bytes());
  return transposed_v;
 }
};
int main(){
 const auto raw=AttentionExecutionPolicy::SPM_KV_BY_MHA,ddr=AttentionExecutionPolicy::DDR_KV;ChunkInfo chunk;
 setenv("RPU_GR00T_CROSS_KV_REUSE","1",1);unsetenv("RPU_GR00T_CROSS_V_TRANSPOSE_REUSE");Model off;assert(!off.cross_v_transpose_reuse_opt_in_);
 for(const char*value:{"","0"}){setenv("RPU_GR00T_CROSS_V_TRANSPOSE_REUSE",value,1);Model x;assert(!x.cross_v_transpose_reuse_opt_in_);}
 for(const char*value:{"2","true"," 1"}){setenv("RPU_GR00T_CROSS_V_TRANSPOSE_REUSE",value,1);try{Model x;assert(false);}catch(const std::runtime_error&){} }
 setenv("RPU_GR00T_CROSS_V_TRANSPOSE_REUSE","1",1);
 for(bool only_spm:{false,true}){
  setenv("RPU_GR00T_KV_SPM_ONLY",only_spm?"1":"0",1);Model m;setenv("RPU_GR00T_CROSS_V_TRANSPOSE_REUSE","0",1);assert(m.cross_v_transpose_reuse_opt_in_);setenv("RPU_GR00T_CROSS_V_TRANSPOSE_REUSE","1",1);
  for(auto profile:{std::pair{4,82},std::pair{4,148},std::pair{4,256},std::pair{4,193},std::pair{4,225},std::pair{4,289},std::pair{4,238},std::pair{4,270},std::pair{4,334},std::pair{10,334},std::pair{10,1},std::pair{10,97},std::pair{10,256},std::pair{10,169},std::pair{10,214}}){
   m.num_steps_=profile.first;const int64_t S=profile.second;m.vl_seq_=S;m.record(chunk);spm.clear();transposes=mirrors=projections=0;
   LayoutContext layout;std::vector<BufferDecl>banks;m.append_cross_kv_buffers(banks,layout);
   assert(banks.size()==2&&banks[0].size==16*gr00t_pad16(S)*192*2&&banks[1].size==banks[0].size);
   if(S==169)assert(gr00t_pad16(S)==176&&m.cross_kv_bank_stride()==67584&&m.cross_kv_bank_bytes()==1081344);
   if(S==214)assert(m.cross_kv_bank_stride()==86016&&m.cross_kv_bank_bytes()==1376256);
   for(int request=0;request<3;++request)for(int body=0;body<m.num_steps_;++body){m.context.body_iter=body;for(int layer=0;layer<32;layer+=2){
    if(body>0)spm[m.addr(0,"v")].assign(gr00t_pad16(S)*192,-1);
    const auto out=m.run(chunk,layer,raw,request);const auto&v=spm[out];const auto pad=gr00t_pad16(S);
    for(int c=0;c<192;++c)for(int r=0;r<pad;++r)assert(v[c*pad+r]==(r<S?request*100000+layer*1000+r*192+c:0));
   }}
   assert(transposes==48&&projections==48&&mirrors==(only_spm?0:48));
   assert(m.context.body_iter==m.num_steps_-1);
   for(int64_t body:{int64_t{-1},m.num_steps_}){m.context.body_iter=body;
    const int before=transposes;try{m.run(chunk,0,raw,0);assert(false);}catch(const std::runtime_error&){}assert(transposes==before);}
   m.context.body_iter=0;
  }
  for(auto lengths:{std::pair{169,214},std::pair{214,169}}){
   m.num_steps_=10;m.vl_seq_=lengths.first;m.record(chunk);m.vl_seq_=lengths.second;
   const int before=transposes;try{m.cross_v_transpose_reused(chunk,raw);assert(false);}catch(const std::runtime_error&){}assert(transposes==before);
  }
  m.num_steps_=4;m.vl_seq_=82;m.record(chunk,false);transposes=mirrors=projections=0;
  for(int body=0;body<4;++body){m.context.body_iter=body;for(int layer=0;layer<32;layer+=2)assert(m.run(chunk,layer,ddr,7)==0);}
  assert(transposes==0&&mirrors==16&&projections==16);
 }
 // VT-off keeps the raw bank and transposes it per body; KV-only remains independent.
 off.record(chunk);transposes=projections=0;for(int body=0;body<4;++body){off.context.body_iter=body;for(int l=0;l<32;l+=2)assert(off.run(chunk,l,raw,9)==off.addr(0,"sdpa_tmp"));}assert(transposes==64&&projections==16);
 // Without cross-KV reuse there is no bank to retain, even when VT opt-in is set.
 setenv("RPU_GR00T_CROSS_KV_REUSE","0",1);Model unbanked;unbanked.record(chunk);transposes=projections=0;
 for(int body=0;body<4;++body){unbanked.context.body_iter=body;unbanked.run(chunk,0,raw,10);}assert(transposes==4&&projections==4);
 setenv("RPU_GR00T_CROSS_KV_REUSE","1",1);Model m;m.record(chunk);
 {Model m4=m;m4.tp=4;m4.record(chunk);assert(m4.cross_v_transpose_reused(chunk,raw));}
 for(int bad=0;bad<11;++bad){Model x=m;switch(bad){case 0:x.vl_seq_=148;break;case 1:x.num_steps_=3;break;case 2:x.layers=31;break;case 3:x.tp=2;break;case 4:x.context.batch_size=2;break;case 5:x.layer_weights_[0].is_cross=false;break;case 6:x.context.routes[1].selector=1;break;case 7:x.context.routes[1].invocation=1;break;case 8:x.num_steps_=10;break;case 9:x.vl_seq_=214;break;case 10:x.num_steps_=10;x.vl_seq_=214;break;}
  const int before=transposes;try{x.cross_v_transpose_reused(chunk,raw);assert(false);}catch(const std::runtime_error&){}assert(transposes==before);
 }
 for(int bad=0;bad<12;++bad){Model x=m;switch(bad){case 0:x.loop_mode_=false;break;case 1:x.denoise_mode_=false;break;case 2:x.z2_bound_=true;break;case 3:x.z2_prepared_seq_len_=82;break;case 4:x.context.batch_size=2;break;case 5:x.tp=2;break;case 6:x.vl_seq_=0;break;case 7:x.context.complete=false;break;case 8:x.num_steps_=3;break;case 9:x.num_steps_=10;x.vl_seq_=0;break;case 10:x.num_steps_=10;x.vl_seq_=-1;break;case 11:x.num_steps_=5;break;}x.record(chunk);assert(!x.cross_v_transpose_reused(chunk,raw));}
 for(int body:{-1,4}){m.context.body_iter=body;const int before=transposes;try{m.run(chunk,0,raw,0);assert(false);}catch(const std::runtime_error&){}assert(transposes==before);}
}
'''.replace("__CONSTANTS__", constants).replace("__READERS__", readers).replace("__METHODS__", methods).replace("__OPERANDS__", operands).replace("__TRANSPOSE__", transpose).replace("__MIRROR__", mirror)
    _run(tmp_path, cpp)


def test_cross_v_transpose_reuse_binds_layout_and_keeps_writer_before_transpose():
    source = SOURCE.read_text()
    cross = _function(source, "    void build_cross_attn_block(")
    assert cross.index("consume_kv_insert_plan(") < cross.index("rpu_launch_insert_kvcache_spm_unified_with_plan(") < cross.index("rpu_launch_v_transpose_spm(")
    assert 'addr_offset(v_slot).value + v_offset' in cross
    assert 'addr(0, "q"), k_spm, transposed_v' in cross
    assert 'consume_kv_insert_plan(\n                GR00T_DIT_CROSS_KV_INSERT_SITE, S, chunk.idx, k_spm, v_spm)' in cross
    layout = _function(source, "    int64_t subclass_layout_hash(")
    assert "GR00T_DIT_CROSS_V_TRANSPOSE_SCHEDULE_SITE" in layout
    manifest = _function(source, "    FmbPhysicalExecutionManifest physical_manifest_for_attention_routes(")
    assert "cross_v_transpose_profile(chunk.len, layout.batch_size, cross_raw_spm) ? 2 : 1" in manifest
    assert "cross_v_transpose_arguments(chunk.len, layout.batch_size, cross_raw_spm)" in manifest
    banks = _function(source, "    void append_cross_kv_buffers(")
    assert "cross_v_transpose" not in banks
