"""Execute real owner-cache cold/manifest/caller methods on the shared CPU fixture."""
from pathlib import Path
import re

from test_g05_q_gate_native import _compile_run
from test_g05_qkv_gate_native import _OWNER_CACHE_STUBS, _qkv_gate_model_cpp
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def test_owner_cache_cold_admission_routes_and_real_caller_fallbacks(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    attention = _block(source, 'void Qwen3_5Model::build_full_attention(')
    manifest = _block(source, 'FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate(')
    methods = [
        ('void prepare_action_owner_cache()', 'void Qwen3_5Model::prepare_action_owner_cache('),
        ('bool use_action_owner_cache(int64_t rows,int64_t max_prefix)const', 'bool Qwen3_5Model::use_action_owner_cache('),
        ('std::vector<int64_t>action_qkv_gate_owner_only_arguments()const', 'std::vector<int64_t> Qwen3_5Model::action_qkv_gate_owner_only_arguments('),
        ('std::vector<int64_t>action_k_owner_norm_rope_arguments()const', 'std::vector<int64_t> Qwen3_5Model::action_k_owner_norm_rope_arguments('),
        ('std::vector<int64_t>action_owner_cache_arguments(const KvInsertSegmentPlan&plan)const', 'std::vector<int64_t> Qwen3_5Model::action_owner_cache_arguments('),
        ('void launch_action_qkv_gate_owner_only(int layer,const ChunkInfo&chunk)', 'void Qwen3_5Model::launch_action_qkv_gate_owner_only('),
        ('void launch_action_k_owner_norm_rope(int layer,const ChunkInfo&chunk)', 'void Qwen3_5Model::launch_action_k_owner_norm_rope('),
        ('void launch_action_owner_cache_insert(int layer,const ChunkInfo&chunk,const KvInsertSegmentPlan&plan)', 'void Qwen3_5Model::launch_action_owner_cache_insert('),
        ('FmbPhysicalManifestForwardCapability physical_manifest_forward_capability(const FmbPhysicalExecutionManifest&manifest)const', 'Qwen3_5Model::physical_manifest_forward_capability('),
    ]
    cpp = _qkv_gate_model_cpp().split('int main(){', 1)[0]
    cpp = '#include <limits>\n#include <tuple>\n' + cpp
    cpp = cpp.replace('FmbRouteFamily{LINEAR,ROPE,KV_INSERT,OTHER}', 'FmbRouteFamily{LINEAR,ROPE,KV_INSERT,GRAPH_SCHEDULE,OTHER}')
    cpp = cpp.replace(_OWNER_CACHE_STUBS, '\n'.join(decl + _block(source, stem) for decl, stem in methods))
    ids = re.findall(r'KernelId::(G05_(?:K_OWNER|OWNER_INSERT)_\w+)', _block(source, 'void Qwen3_5Model::prepare_action_owner_cache('))
    assert len(ids) == 6
    cpp = cpp.replace('enum class KernelId{', 'enum class KernelId{' + ','.join(ids) + ',')
    cpp = cpp.replace('bool q=true,kv=true,fanout=true,joint=true;', 'bool q=true,kv=true,fanout=true,joint=true;int missing_owner=-1;')
    owner_ids_arr = ','.join(f'KernelId::{name}' for name in ids)
    cpp = cpp.replace('bool has_loaded(KernelId id){++queries;', f'bool has_loaded(KernelId id){{++queries;const std::array<KernelId,6> kOwnerIds{{{owner_ids_arr}}};for(int idx=0;idx<6;++idx)if(id==kOwnerIds[idx])return idx!=missing_owner;')
    cpp = cpp.replace('namespace c10{enum', 'namespace c10{using Half=uint16_t;enum')
    cpp = cpp.replace('int dim()const{return rank;}', 'int64_t numel()const{return rank==1?rows:rows*cols;}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(owner);}int dim()const{return rank;}')
    support = r'''
enum class RpuRmsNormSpmRoute:int64_t{BASE=1};
enum class KvInsertRoute:int64_t{V2=1,ALIGNED_V16=2,HYBRID2=4,HYBRID3=5};
enum class KvInsertKernel:int64_t{V2=1,V16=2};
struct Segment{KvInsertKernel kernel;int64_t position,token_offset,rows;};
struct KvInsertSegmentPlan{
 int64_t logical=32,physical=32,selection_state=2,mask=31;std::vector<Segment>parts;KvInsertRoute route_id;
 explicit KvInsertSegmentPlan(int64_t p=0){if(p==0){parts={{KvInsertKernel::V16,0,0,32}};route_id=KvInsertRoute::ALIGNED_V16;}else{parts={{KvInsertKernel::V2,p,0,11},{KvInsertKernel::V16,p+11,11,16},{KvInsertKernel::V2,p+27,27,5}};route_id=KvInsertRoute::HYBRID3;}}
 int64_t logical_rows()const{return logical;}int64_t physical_rows()const{return physical;}
 const Segment&segment(size_t i)const{return parts.at(i);}size_t segment_count()const{return parts.size();}
 KvInsertRoute route()const{return route_id;}
};
using KvInsertRouteArguments=std::vector<int64_t>;
void rpu_validate_kvinsert_segment_plan(const KvInsertSegmentPlan&p,int c,int h,int d){TORCH_CHECK(c==8&&h==8&&d==256&&p.logical>0&&p.physical>=p.logical&&!p.parts.empty());}
KvInsertRouteArguments rpu_kvinsert_route_arguments(const KvInsertSegmentPlan&p,int c,int h,int d){
 rpu_validate_kvinsert_segment_plan(p,c,h,d);std::vector<int64_t>x{2,int64_t(p.route()),p.logical,p.physical,int64_t(p.segment_count())};
 for(const auto&s:p.parts)x.insert(x.end(),{int64_t(s.kernel),s.position,s.token_offset,s.rows});
 x.resize(17);x.insert(x.end(),{p.selection_state,p.mask,100,123,0});return x;
}
KvInsertSegmentPlan restore_plan(const std::vector<int64_t>&x){assert(x.size()==22);KvInsertSegmentPlan p(x.at(6));p.logical=x.at(2);p.physical=x.at(3);p.route_id=KvInsertRoute(x.at(1));p.selection_state=x.at(17);p.mask=x.at(18);p.parts.clear();for(int64_t i=0;i<x.at(4);++i)p.parts.push_back({KvInsertKernel(x.at(5+i*4)),x.at(6+i*4),x.at(7+i*4),x.at(8+i*4)});return p;}
KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan_auto(int64_t p,int64_t l,int64_t ph,int,int,int,uint32_t){KvInsertSegmentPlan x(p);x.logical=l;x.physical=ph;return x;}
KvInsertSegmentPlan rpu_rebase_kvinsert_segment_plan_position(const KvInsertSegmentPlan&p,int64_t pos,int,int,int){auto x=p;const int64_t delta=pos-p.segment(0).position;for(auto&s:x.parts)s.position+=delta;return x;}
constexpr uint32_t KV_INSERT_CAP_V2=1,KV_INSERT_CAP_V16=2,KV_INSERT_CAP_HYBRID2=8,KV_INSERT_CAP_HYBRID3=16,QWEN35_TEXT_KV_CAPABILITIES=27;
constexpr int64_t KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION=1LL<<16,kFullAttentionKvInsertSite=8503568375190929130LL;
enum class Qwen35TextRopeRoute{ROPE_SPM=1};int64_t qwen35_text_rope_route(Qwen35TextRopeRoute r){return int64_t(r);}
std::vector<int64_t>consumed;
enum class FmbPhysicalManifestState{INCOMPLETE,COMPLETE};
enum class FmbGraphLifecycle:int64_t{RETAINED_CACHE=1,BOUNDED_ONESHOT=2,COMPOSITE_CHILD=4};
struct CostRoute{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>arguments;int64_t invocation;};
struct FmbPhysicalExecutionManifest{FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;std::vector<CostRoute>routes;int64_t logical_length=32;};
using CostManifest=FmbPhysicalExecutionManifest;
struct FmbPhysicalManifestForwardCapability{bool supported=false;FmbGraphLifecycle lifecycle=FmbGraphLifecycle::BOUNDED_ONESHOT;};
'''
    cpp = cpp.replace('struct Model{', support + r'''
void rpu_launch_g05_qkv_gate_owner_only_linear_w8a16_acc32_kernel(uint32_t input,const at::Tensor&q,const at::Tensor&g,const at::Tensor&k,const at::Tensor&v,const at::Tensor&qs,const at::Tensor&gs,const at::Tensor&ks,const at::Tensor&vs,uint32_t qo,uint32_t go,uint32_t ko,uint32_t vo,bool linear_acc32=true){rpu_launch_g05_qkv_gate_owner_linear_w8a16_acc32_kernel(input,q,g,k,v,qs,gs,ks,vs,qo,go,ko,vo,linear_acc32);launches.back()="owner-linear";}
void rpu_launch_g05_k_owner_norm_rope_spm_kernel(uint32_t k,uint32_t gamma,const at::Tensor&cos,const at::Tensor&sin,int64_t pos){assert(k==0x28000&&gamma>=0x50000&&cos.owner==500&&sin.owner==501&&pos==0);launches.push_back("owner-norm-rope");}
void rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(at::Tensor&,at::Tensor&,uint32_t,uint32_t,const KvInsertSegmentPlan&){assert(false);}
void rpu_launch_g05_owner_insert_kvcache_spm_with_plan(at::Tensor&k,at::Tensor&v,uint32_t ko,uint32_t vo,const KvInsertSegmentPlan&p){assert(k.owner+100==v.owner&&ko==0x28000&&vo==0x2c000&&(p.segment(0).position==0||p.segment(0).position==245));launches.push_back("owner-cache");}
void rpu_launch_rmsnorm_spm_kernel(uint32_t input,uint32_t output,uint32_t,int64_t rows,int64_t dim,double eps,RpuRmsNormSpmRoute route,int cores){assert(input==output&&rows==32&&dim==256&&eps==1e-6&&route==RpuRmsNormSpmRoute::BASE&&cores==8);launches.push_back(input==0x20000?"q-norm":"k-norm");}
template<class...T>void rpu_launch_rope_spm_kernel(T...){assert(false);}
void rpu_launch_insert_kvcache_spm_unified_with_plan(at::Tensor&,at::Tensor&,uint32_t,uint32_t,int64_t,int64_t,int64_t,int,int,int,const KvInsertSegmentPlan&){launches.push_back("cache-original");}
struct Model{
 int64_t rotary_dim_=64;bool z2_bound_=false,retained_prefill_graph_=false,catalog_v2=false;
 std::vector<int64_t>action_prefix_lens_=std::vector<int64_t>(24);
 at::Tensor prefill_cos_,prefill_sin_;std::vector<at::Tensor>cache_k{24},cache_v{24};
 std::vector<at::Tensor>*k_caches=&cache_k,*v_caches=&cache_v;
 void owner_fixture(){for(int i=0;i<24;++i){auto&n=layer_weights_[i].k_norm_w;n.rank=1;n.rows=256;n.owner=1000+i;action_prefix_lens_[i]=(i%4==3)?245:0;cache_k[i].owner=3000+i;cache_v[i].owner=3100+i;}prefill_cos_.owner=500;prefill_sin_.owner=501;prefill_cos_.rows=prefill_sin_.rows=64;prefill_cos_.cols=prefill_sin_.cols=32;}
 uint32_t layer_addr(int layer,int,const char*)const{return 0x50000+layer*512;}
 struct Offset{uint32_t value;};Offset addr_offset(const char*n)const{return {addr(0,n)};}
 const Route&find_physical_route(FmbRouteFamily f,int64_t site,int64_t inv)const{auto i=std::find_if(routes.begin(),routes.end(),[&](const auto&r){return r.family==f&&r.site==site&&r.invocation==inv;});TORCH_CHECK(i!=routes.end());return *i;}
 KvInsertSegmentPlan restore_kvinsert_plan(int64_t,const std::vector<int64_t>&x,int64_t,int64_t,int64_t)const{return restore_plan(x);}
 KvInsertSegmentPlan restore_kvinsert_plan_from_manifest(const FmbPhysicalExecutionManifest&m,int64_t site,int64_t invocation,int,int,int)const{auto i=std::find_if(m.routes.begin(),m.routes.end(),[&](const auto&r){return r.family==FmbRouteFamily::KV_INSERT&&r.site_id==site&&r.invocation==invocation;});TORCH_CHECK(i!=m.routes.end());return restore_plan(i->arguments);}
 KvInsertSegmentPlan resolve_kvinsert_plan_auto(int64_t,int,int64_t pos,int64_t l,int64_t ph,int64_t c,int64_t h,int64_t d,uint32_t cap)const{auto p=rpu_resolve_kvinsert_segment_plan_auto(pos,l,ph,c,h,d,cap);if(catalog_v2){p.route_id=KvInsertRoute::V2;p.parts={{KvInsertKernel::V2,pos,0,32}};}return p;}
 void consume_manifest_route(FmbRouteFamily,int64_t,int64_t,int64_t){assert(false);}
''', 1)
    # Context field spelling follows the production route record.
    cpp = cpp.replace('std::vector<int64_t>args;int64_t invocation;', 'std::vector<int64_t>arguments;int64_t invocation;').replace('.args', '.arguments').replace('->args', '->arguments')
    cpp = cpp.replace('it->arguments==args);}', 'it->arguments==args);consumed.push_back(site);}')
    extra_constants = '\n'.join(re.findall(r'constexpr int64_t QWEN35_TEXT_(?:Q_NORM_ROPE_SITE|K_NORM_ROPE_SITE|Q_ROPE_SITE|K_ROPE_SITE)\s*=.*?;', source, re.S))
    cpp = cpp.replace('struct Model{', extra_constants + '\nstruct Model{', 1)
    cache_manifest = _block(manifest[manifest.rindex('    if (action_mode_ && has_full_attention)'):], 'if (action_mode_ && has_full_attention)')
    phase3 = _block(attention, 'if (has_qk_norm_)')
    phase4 = attention[attention.index('    auto& k_cache ='):attention.index('    int64_t kv_seq_len =')]
    extras = r'''
 void append_cache_plan(int64_t physical_len,int64_t position){struct{std::vector<Route>routes;int graph_lifecycle=4;}manifest;manifest.routes=routes;__CACHE_MANIFEST__ routes=manifest.routes;}
 void norm_stage(int layer_idx,ChunkInfo chunk={}){const bool owner_cache=use_action_owner_cache(chunk.len,ctx().position);int64_t seq_len=chunk.len,nkv=8,tp=8,hd=256,local_q_heads=1,cos_sin_start=0;int auxiliary_cores=8;c10::Half*cos_ptr=nullptr,*sin_ptr=nullptr;auto emit_partial_rope=[&](uint32_t buf,int64_t heads){assert(heads==1);launches.push_back(buf==0x20000?"q-rope":"k-rope");};if(has_qk_norm_)__PHASE3__}
 void insert_stage(int layer_idx,ChunkInfo chunk={}){const bool owner_cache=use_action_owner_cache(chunk.len,ctx().position);int64_t seq_len=chunk.len,nkv=8,tp=8,hd=256;__PHASE4__}
 void full_plan(int64_t dry_position){const auto old_position=position;position=dry_position;plan();position=old_position;append_cache_plan(32,dry_position);}
 void chain(int layer){phase2(layer);norm_stage(layer);insert_stage(layer);}
'''.replace('__CACHE_MANIFEST__', cache_manifest).replace('__PHASE3__', phase3).replace('__PHASE4__', phase4)
    cpp = cpp.replace(' void loop(){', extras + '\n void loop(){', 1)
    # Exercise the actual generic cost-domain filter, not a copied family predicate.
    generic = (ROOT / 'src/core/fused_model_base.cpp').read_text()
    implementation = (ROOT / 'src/core/fused_model_base_impl.h').read_text()
    begin = implementation.index('    struct KvCostObservation {')
    cost_structs = implementation[begin:implementation.index('    // A scoped pointer', begin)]
    cpp += r'''
struct LayoutContext{};
struct FusedModelBase{struct Impl{__COST_STRUCTS__};};
struct FmbPrefillStageCandidate{CostManifest physical_manifest;int64_t qkv_chunk_size=32;};
std::vector<int64_t>encode_fmb_prefill_stage_candidate(const FmbPrefillStageCandidate&){return {32};}
constexpr size_t kKvInsertRouteArgumentWords=22;
static FusedModelBase::Impl::KvCostCandidate make_kvinsert_cost_candidate(const FmbPrefillStageCandidate&candidate,const std::vector<FusedModelBase::Impl::KvCostObservation>&observations)__COST_BODY__
'''.replace('\n+', '\n').replace('__COST_STRUCTS__', cost_structs).replace(
        '__COST_BODY__', _block(generic, 'static FusedModelBase::Impl::KvCostCandidate make_kvinsert_cost_candidate('))
    cpp += r'''
int main(){
 auto&cache=KernelCache::instance();Model m;m.owner_fixture();m.install();assert(m.action_owner_cache_available_&&!m.use_action_owner_cache(32,245));
 {auto x=m;x.high_precision_=true;x.prepare_action_owner_cache();assert(!x.action_owner_cache_available_&&x.action_qkv_gate_available_);}
 m.loop();m.full_plan(245);assert(m.use_action_owner_cache(32,0)&&m.use_action_owner_cache(32,245)&&!m.use_action_owner_cache(32,244)&&!m.use_action_owner_cache(31,245));
 const int64_t linear=558172469151339648LL,norm=2641642310566033775LL,insert=8281105153491195213LL;
 auto has=[](const Model&x,int64_t site){return std::any_of(x.routes.begin(),x.routes.end(),[&](const auto&r){return r.site==site;});};
 assert(has(m,linear)&&has(m,norm)&&has(m,insert)&&has(m,kFullAttentionKvInsertSite)&&!has(m,QWEN35_ACTION_QKV_GATE_W8_SITE));
 assert(m.find_physical_route(FmbRouteFamily::LINEAR,linear,2).arguments[27]==0);
 // Only the original primary KV22 enters the generic native cost domain.
 FmbPrefillStageCandidate candidate;std::vector<FusedModelBase::Impl::KvCostObservation>observations;
 for(const auto&r:m.routes){candidate.physical_manifest.routes.push_back({r.site,r.family,r.selector,r.flags,r.arguments,r.invocation});
  if(r.family==FmbRouteFamily::KV_INSERT){assert(r.site==kFullAttentionKvInsertSite&&r.arguments.size()==22&&r.arguments[0]==2);observations.push_back({r.site,4,8,8,256,27,r.arguments});}
  if(r.site==insert)assert(r.family==FmbRouteFamily::GRAPH_SCHEDULE&&r.arguments.size()==38);
 }
 const auto cost=make_kvinsert_cost_candidate(candidate,observations);assert(cost.reason=="READY"&&cost.rows.size()==2&&observations.size()==2);
 for(const auto&row:cost.rows)assert(row[0]==kFullAttentionKvInsertSite&&(row[3]==0||row[3]==245));
 auto wrong_family=candidate;for(auto&r:wrong_family.physical_manifest.routes)if(r.site_id==insert)r.family=FmbRouteFamily::KV_INSERT;
 const auto rejected_cost=make_kvinsert_cost_candidate(wrong_family,observations);assert(rejected_cost.rows.empty()&&rejected_cost.reason=="NATIVE_KV_DOMAIN_PRODUCER_UNAVAILABLE");
 // The existing capability hook must close the dry/EXACT boundary, before
 // either layout publication or a native launch can accept an impossible plan.
 assert(m.physical_manifest_forward_capability(candidate.physical_manifest).supported);
 {auto x=m;x.catalog_v2=true;reject([&]{x.full_plan(245);});}
 for(int64_t pos:{0,245}){
  KvInsertSegmentPlan v2(pos);v2.route_id=KvInsertRoute::V2;v2.parts={{KvInsertKernel::V2,pos,0,32}};
  reject([&]{m.action_owner_cache_arguments(v2);});
  auto alternate=candidate.physical_manifest;for(auto&r:alternate.routes)if(r.site_id==kFullAttentionKvInsertSite&&r.invocation==(pos>0?1:0)){r.selector=1;r.arguments=rpu_kvinsert_route_arguments(v2,8,8,256);}
  reject([&]{m.physical_manifest_forward_capability(alternate);});
  auto exact=candidate.physical_manifest;for(auto&r:exact.routes)if(r.site_id==kFullAttentionKvInsertSite&&r.invocation==(pos>0?1:0))r.arguments[17]=1;
  reject([&]{m.physical_manifest_forward_capability(exact);});
 }
 for(size_t part=0;part<3;++part)for(int field=0;field<4;++field){KvInsertSegmentPlan x(245);auto&s=x.parts[part];if(field==0)s.kernel=s.kernel==KvInsertKernel::V2?KvInsertKernel::V16:KvInsertKernel::V2;if(field==1)++s.position;if(field==2)++s.token_offset;if(field==3)++s.rows;reject([&]{m.action_owner_cache_arguments(x);});}
 for(int field=0;field<4;++field){KvInsertSegmentPlan x(0);if(field==0)x.route_id=KvInsertRoute::V2;if(field==1)x.parts.push_back(x.parts[0]);if(field==2)++x.logical;if(field==3)++x.physical;reject([&]{m.action_owner_cache_arguments(x);});}
 for(int field=0;field<6;++field){auto x=candidate.physical_manifest;for(auto&r:x.routes)if(r.site_id==insert&&r.invocation==1){if(field==0)r.family=FmbRouteFamily::KV_INSERT;if(field==1)++r.selector;if(field==2)++r.flags;if(field==3)++r.invocation;if(field==4)++r.arguments[0];}if(field==5)x.routes.erase(std::remove_if(x.routes.begin(),x.routes.end(),[](const auto&r){return r.site_id==kFullAttentionKvInsertSite&&r.invocation==1;}),x.routes.end());reject([&]{m.physical_manifest_forward_capability(x);});}
 auto identity=m.kvinsert_cost_weight_identity();auto hash=m.subclass_layout_hash();const auto queries=cache.queries;const auto proofs=replica_proofs;
 for(int layer=0;layer<24;++layer){launches.clear();consumed.clear();m.chain(layer);assert(launches==std::vector<std::string>({"owner-linear","q-norm","q-rope","owner-norm-rope","owner-cache"}));for(auto site:{linear,norm,insert,kFullAttentionKvInsertSite})assert(std::count(consumed.begin(),consumed.end(),site)==1);}
 assert(cache.queries==queries&&replica_proofs==proofs);
 // Every scalar of each new COMPLETE route rejects before its own first launch.
 for(int kind=0;kind<3;++kind){const auto site=kind==0?linear:kind==1?norm:insert;const int64_t invocation=kind==2?1:2;const auto&base=m.find_physical_route(kind==0?FmbRouteFamily::LINEAR:kind==1?FmbRouteFamily::ROPE:FmbRouteFamily::GRAPH_SCHEDULE,site,invocation);
  auto run=[&](Model&x){if(kind==0)x.launch_action_qkv_gate_owner_only(23,{});else if(kind==1)x.launch_action_k_owner_norm_rope(23,{});else x.launch_action_owner_cache_insert(23,{},KvInsertSegmentPlan(245));};
  for(size_t i=0;i<base.arguments.size();++i){auto x=m;for(auto&r:x.routes)if(r.site==site&&r.invocation==invocation)++r.arguments[i];reject([&]{run(x);});}
  for(int fault=0;fault<8;++fault){auto x=m;for(auto&r:x.routes)if(r.site==site&&r.invocation==invocation){if(fault==0)++r.site;if(fault==1)r.family=FmbRouteFamily::OTHER;if(fault==2)++r.selector;if(fault==3)++r.flags;if(fault==4)++r.invocation;if(fault==5)r.arguments.pop_back();}if(fault==6)x.routes.clear();if(fault==7)x.complete=false;reject([&]{run(x);});}
 }
 for(auto bad:std::vector<ChunkInfo>{{31,2,0},{32,2,1},{32,3,0}}){reject([&]{m.launch_action_qkv_gate_owner_only(0,bad);});reject([&]{m.launch_action_k_owner_norm_rope(0,bad);});}
 reject([&]{m.launch_action_qkv_gate_owner_only(-1,{});});reject([&]{m.launch_action_k_owner_norm_rope(24,{});});reject([&]{m.launch_action_owner_cache_insert(23,{},KvInsertSegmentPlan(0));});
 {auto x=m;++x.action_prefix_lens_[23];reject([&]{x.insert_stage(23);});}
 {auto x=m;for(auto&r:x.routes)if(r.site==kFullAttentionKvInsertSite&&r.invocation==1)r.arguments[3]=48;reject([&]{x.insert_stage(23);});}
 // Dry queries consume the passed position, never stale per-layer prefix state.
 {auto x=m;std::fill(x.action_prefix_lens_.begin(),x.action_prefix_lens_.end(),333);x.position=999;x.full_plan(245);assert(has(x,linear)&&has(x,insert));assert(x.subclass_layout_hash()==hash);x.full_plan(244);assert(!has(x,linear)&&!has(x,insert)&&has(x,QWEN35_ACTION_QKV_GATE_W8_SITE));x.full_plan(0);assert(has(x,linear)&&has(x,insert));}
 for(int missing=0;missing<6;++missing){cache.missing_owner=missing;auto x=m;x.install();assert(!x.action_owner_cache_available_&&x.action_qkv_gate_available_);x.full_plan(245);assert(!has(x,linear)&&!has(x,norm)&&!has(x,insert)&&has(x,QWEN35_ACTION_QKV_GATE_W8_SITE));launches.clear();x.chain(23);assert(launches==std::vector<std::string>({"joint","q-norm","q-rope","k-norm","k-rope","cache-original"}));assert(x.kvinsert_cost_weight_identity()!=identity&&x.subclass_layout_hash()!=hash);}cache.missing_owner=-1;
 for(int field=0;field<6;++field){auto x=m;auto&n=x.layer_weights_.back().k_norm_w;if(field==0)n.present=false;if(field==1)n.location=c10::DeviceType::CPU;if(field==2)n.dtype=at::kChar;if(field==3)n.contiguous=false;if(field==4)n.rank=2;if(field==5)n.rows=128;x.prepare_action_owner_cache();assert(!x.action_owner_cache_available_);}
 for(int field=0;field<6;++field){auto x=m;if(field==0)x.has_qk_norm_=false;if(field==1)x.has_mrope_=false;if(field==2)x.rotary_dim_=256;if(field==3)x.eps_=1e-5;if(field==4)x.layer_weights_.pop_back();if(field==5)x.action_qkv_gate_available_=false;x.prepare_action_owner_cache();assert(!x.action_owner_cache_available_);}
 for(int field=0;field<5;++field){auto x=m;if(field==0)x.action_num_steps_=1;if(field==1)x.action_loop_active_=false;if(field==2)x.action_step_active_=false;if(field==3)x.action_len_=16;if(field==4)x.position=244;x.full_plan(x.position);assert(!has(x,linear)&&!has(x,insert));launches.clear();x.phase2(0);assert(launches.front()!=("owner-linear"));}
 Model fp16(false);fp16.owner_fixture();fp16.install();fp16.loop();fp16.full_plan(245);assert(!fp16.action_owner_cache_available_&&!has(fp16,linear));launches.clear();fp16.phase2(0);assert(launches==std::vector<std::string>({"qgate-fp16","kv-fp16"}));
 {auto x=m;++x.layer_weights_.back().k_norm_w.owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 // Accumulation changes only the projection; BASE K math and KV22 remain shared.
 {Model x;x.linear_acc32_=false;x.owner_fixture();x.install();x.loop();x.full_plan(245);assert(x.action_owner_cache_available_&&x.action_qkv_gate_owner_only_arguments()[5]==16&&x.action_qkv_gate_owner_only_arguments()[6]==48&&x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=identity);
  const auto valid=x.routes;x.routes=m.routes;reject([&]{x.phase2(23);});x.routes=valid;
  assert(x.action_k_owner_norm_rope_arguments()==m.action_k_owner_norm_rope_arguments()&&x.action_owner_cache_arguments(KvInsertSegmentPlan(245))==m.action_owner_cache_arguments(KvInsertSegmentPlan(245)));
  for(int layer=0;layer<24;++layer){linear_modes.clear();launches.clear();x.chain(layer);assert(launches==std::vector<std::string>({"owner-linear","q-norm","q-rope","owner-norm-rope","owner-cache"})&&linear_modes==std::vector<bool>({false}));}
  cache.acc16_available=false;x.install();x.full_plan(245);assert(!x.action_owner_cache_available_&&!has(x,linear)&&x.action_mod_body_w8_);linear_modes.clear();launches.clear();x.chain(23);assert(launches==std::vector<std::string>({"q","k","v","gate","q-norm","q-rope","k-norm","k-rope","cache-original"})&&linear_modes.empty());cache.acc16_available=true;
 }
 m.reset();assert(!m.action_owner_cache_available_&&!m.action_qkv_gate_available_);m.action_mode_=true;m.install();m.loop();assert(m.action_owner_cache_available_&&m.subclass_layout_hash()==hash);
}
'''
    # Also exercise the real mixed-profile predicate and its separate W8 KV
    # producer; packed Q/O/MLP must never enter the old joint W8 launcher.
    cpp = cpp.replace('bool enabled()const{return false;}',
                      'bool enabled()const{return !values.empty();}')
    cpp = cpp.replace('bool use_action_nvfp4(int64_t)const{return false;}',
                      'bool use_action_nvfp4(int64_t rows)const' +
                      _block(source, 'bool Qwen3_5Model::use_action_nvfp4('))
    cpp = cpp.replace('constexpr int kHalf=1,kChar=3;', 'constexpr int kHalf=1,kChar=3,kByte=4;')
    extra = r'''
 // Mixed NVFP4 Q/O/MLP keeps independent W8 KV-owner+fanout and original
 // owner norm/cache paths, while the joint W8 producer is absent.
 {Model x;x.linear_acc32_=false;x.action_nvfp4_.values={1,2,3,4,5};x.owner_fixture();
  for(auto&w:x.layer_weights_)for(auto*t:{&w.q_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})t->dtype=at::kByte;
  x.install();x.loop();x.full_plan(245);
  assert(x.action_owner_cache_available_&&!x.action_qkv_gate_available_&&!x.action_mod_body_w8_);
  assert(has(x,QWEN35_ACTION_KV_OWNER_W8_SITE)&&has(x,norm)&&has(x,insert)&&!has(x,linear)&&!has(x,QWEN35_ACTION_QKV_GATE_W8_SITE));
  launches.clear();linear_modes.clear();x.chain(23);
  assert(launches==std::vector<std::string>({"q","kv-w8","gate","q-norm","q-rope","owner-norm-rope","owner-cache"}));
  assert(linear_modes==std::vector<bool>({false}));
  x.full_plan(244);assert(!has(x,norm)&&!has(x,insert)&&has(x,QWEN35_ACTION_KV_OWNER_W8_SITE));
  cache.missing_owner=0;x.prepare_action_owner_cache();assert(!x.action_owner_cache_available_&&x.action_kv_owner_available_);cache.missing_owner=-1;
 }
'''
    end = cpp.rindex('\n}')
    cpp = cpp[:end] + extra + cpp[end:]
    _compile_run(tmp_path, 'owner_cache_model', cpp)
