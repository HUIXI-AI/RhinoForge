"""Actual typed joint writer and cold COMPLETE contracts."""
from pathlib import Path

from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_joint_cache_native_two_typed_owners_complete_spans_and_plans(tmp_path, monkeypatch):
    import test_g05_owner_cache_native as owner
    captured = []
    monkeypatch.setattr(owner, '_compile_run', lambda p, n, cpp: captured.append(cpp))
    owner.test_owner_norm_rope_and_segmented_cache_actual_wrappers(tmp_path)
    cpp = captured[0].split('int main(){', 1)[0]
    cpp += r'''
int main(){
 auto&g=RpuKernelGraph::active();
 constexpr auto P0=KernelId::G05_OWNER_INSERT_KV_CACHE_M32D256_P0;
 constexpr auto P245=KernelId::G05_OWNER_INSERT_KV_CACHE_M32D256_P245;
 at::Tensor kc{0x1240000000,{1,136,1,16,8,16,16}},vc{0x1250000000,{1,136,1,16,8,16,16}};
 const KvInsertSegmentPlan p0{KvInsertRoute::ALIGNED_V16,{{KvInsertKernel::V16,0,0,32}}};
 const KvInsertSegmentPlan p245{KvInsertRoute::HYBRID3,{{KvInsertKernel::V2,245,0,11},{KvInsertKernel::V16,256,11,16},{KvInsertKernel::V2,272,27,5}}};
 auto launch=[&](const auto&p){rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(kc,vc,0x10000,0x20000,p);};
 for(int pass=0;pass<2;++pass)for(const auto&plan:{p0,p245}){
  kc.address+=256;vc.address+=512;const auto before=g.nodes.size();launch(plan);
  assert(g.nodes.size()==before+1);const auto&s=g.nodes.back();const bool prefix=plan.segment(0).position!=0;
  assert(s.id==(prefix?P245:P0)&&s.grid==std::vector<uint16_t>({uint16_t(prefix?17:2),2,1}));
  assert(pair(s,0)==uint32_t(prefix?245:0)&&s.regs.at(2)==1&&s.regs.at(3)==256&&s.regs.at(4)==32&&s.regs.at(21)==8);
  assert(pair(s,6)==0x10000&&pair(s,10)==0x20000&&pair(s,8)==kc.address>>8&&pair(s,12)==vc.address>>8);
  assert(s.operands.size()==2);
  for(int kind=0;kind<2;++kind){const auto&o=s.operands[kind];assert(o.abi.role==(kind?GraphDdrRegisterRole::ValueCache:GraphDdrRegisterRole::KeyCache)&&o.abi.access==GraphDdrRegisterAccess::Write&&o.abi.lo==(kind?12:8)&&o.abi.hi==(kind?13:9));assert(o.owner.address==(kind?vc.address:kc.address));}
 }
 // Both full DDR owners must be admitted before the first staging/flush.
 for(int which=0;which<2;++which)for(int fault=0;fault<11;++fault){auto a=kc,b=vc;auto&t=which?b:a;
  if(fault==0)t.present=false;
  if(fault==1)t.location=c10::DeviceType::CPU;
  if(fault==2)t.dtype=2;
  if(fault==3)t.contiguous=false;
  if(fault==4)t.offset=128;
  if(fault==5)t.shape[1]=17;
  if(fault==6)t.shape[4]=2;
  if(fault==7)t.address+=1;
  if(fault==8)t.address=0;
  if(fault==9)t.address=(UINT64_C(1)<<40)-256;
  if(fault==10)t.shape.pop_back();
  reject([&]{rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(a,b,0x10000,0x20000,p245);});
 }
 reject([&]{rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(kc,kc,0x10000,0x20000,p0);});
 for(int which=0;which<2;++which)for(auto offset:{0x10001u,SpmAllocator::SPM_USABLE-256,0x40010000u})
  reject([&]{rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(kc,vc,which?0x10000:offset,which?offset:0x20000,p0);});
 reject([&]{rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(kc,vc,0x10000,0x10100,p0);});
 for(int fault=0;fault<8;++fault){auto p=p245;
  if(fault==0)p.logical=31;
  if(fault==1)p.physical=48;
  if(fault==2)p.parts[1].token_offset=12;
  if(fault==3)p.parts[2].rows=4;
  if(fault==4)p.parts[0].position=246;
  if(fault==5)p.kind=KvInsertRoute::V2;
  if(fault==6)p.parts.pop_back();
  if(fault==7)p.parts[1].kernel=KvInsertKernel::V2;
  reject([&]{launch(p);});
 }
 for(const auto&p:{p0,p245}){auto id=p.segment(0).position?P245:P0;KernelCache::instance().missing.insert(id);reject([&]{launch(p);});KernelCache::instance().missing.clear();}
 g.active_flag=false;reject([&]{launch(p0);});g.active_flag=true;
 allocator.initialized=false;reject([&]{launch(p245);});allocator.initialized=true;
 // Capacity is a real owner extent, not fixed S136 and never padded logical rows.
 kc.shape[1]=vc.shape[1]=18;launch(p245);
 kc.shape[1]=vc.shape[1]=2;launch(p0);reject([&]{launch(p245);});
 rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(kc,vc,0x10000,SpmAllocator::SPM_USABLE-16384,p0);
}
'''
    _compile_run(tmp_path, 'g05_joint_cache_actual', cpp)


def test_joint_cache_cold_complete_kv22_identity_and_fallbacks(tmp_path, monkeypatch):
    import test_g05_owner_cache_model as owner
    from test_g05_vision_owner_model import _block
    captured = []
    monkeypatch.setattr(owner, '_compile_run', lambda p, n, cpp: captured.append(cpp))
    owner.test_owner_cache_cold_admission_routes_and_real_caller_fallbacks(tmp_path)
    cpp = captured[0].split('int main(){', 1)[0]
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    stubs = ''' bool action_owner_cache_joint_available_=false;
 void prepare_action_owner_cache_joint(){}
 bool use_action_owner_cache_joint(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_owner_cache_joint_abi()const{return {};}
 template<class...T>std::vector<int64_t> action_owner_cache_joint_arguments(T...)const{return {};}
'''
    methods = [
        ('void prepare_action_owner_cache_joint()', 'void Qwen3_5Model::prepare_action_owner_cache_joint('),
        ('bool use_action_owner_cache_joint(int64_t rows,int64_t prefix)const', 'bool Qwen3_5Model::use_action_owner_cache_joint('),
        ('std::vector<int64_t> action_owner_cache_joint_abi()const', 'std::vector<int64_t> Qwen3_5Model::action_owner_cache_joint_abi('),
        ('std::vector<int64_t> action_owner_cache_joint_arguments(const KvInsertSegmentPlan&plan)const', 'std::vector<int64_t> Qwen3_5Model::action_owner_cache_joint_arguments('),
    ]
    assert stubs in cpp
    cpp = cpp.replace(stubs, ' bool action_owner_cache_joint_available_=false;\n' + '\n'.join(d + _block(source, n) for d, n in methods))
    ids = ['G05_OWNER_INSERT_KV_CACHE_M32D256_P0', 'G05_OWNER_INSERT_KV_CACHE_M32D256_P245']
    cpp = cpp.replace('enum class KernelId{', 'enum class KernelId{' + ','.join(ids) + ',')
    cpp = cpp.replace('struct KernelCache{', 'struct KernelCache{bool p0=true,p245=true;')
    cpp = cpp.replace('bool has_loaded(KernelId id){++queries;', 'bool has_loaded(KernelId id){++queries;if(id==KernelId::' + ids[0] + ')return p0;if(id==KernelId::' + ids[1] + ')return p245;\n ')
    cpp = cpp.replace('std::vector<int64_t>action_nvfp4_arguments(int)const{assert(false);return {};}', 'std::vector<int64_t>action_nvfp4_arguments(int family)const{return {family};}')
    old = 'void rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(at::Tensor&,at::Tensor&,uint32_t,uint32_t,const KvInsertSegmentPlan&){assert(false);}'
    new = '''void rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(at::Tensor&k,at::Tensor&v,uint32_t ko,uint32_t vo,const KvInsertSegmentPlan&p){assert(k.owner+100==v.owner&&ko==0x28000&&vo==0x2c000&&(p.segment(0).position==0||p.segment(0).position==245));launches.push_back("joint-cache");}'''
    assert old in cpp
    cpp = cpp.replace(old, new)
    fixture = '''
 void joint_fixture(){linear_acc32_=false;action_nvfp4_.values={1,2,3,4,5};owner_fixture();
  for(auto&w:layer_weights_)for(auto*t:{&w.q_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})t->dtype=at::kByte;
  install();loop();full_plan(245);
 }
'''
    cpp = cpp.replace(' void loop(){', fixture + '\n void loop(){', 1)
    cpp += r'''
int main(){
 Model m;m.joint_fixture();auto&cache=KernelCache::instance();
 const auto site=QWEN35_ACTION_OWNER_CACHE_JOINT_SITE,oldsite=QWEN35_ACTION_OWNER_CACHE_INSERT_SITE;
 auto has=[](const Model&x,int64_t id){return std::any_of(x.routes.begin(),x.routes.end(),[&](const auto&r){return r.site==id;});};
 assert(m.action_owner_cache_joint_available_&&m.use_action_owner_cache_joint(32,0)&&m.use_action_owner_cache_joint(32,245));
 assert(!m.use_action_owner_cache_joint(31,245)&&!m.use_action_owner_cache_joint(32,244));
 assert(has(m,site)&&!has(m,oldsite));
 auto identity=m.kvinsert_cost_weight_identity();auto hash=m.subclass_layout_hash();
 FmbPrefillStageCandidate candidate;std::vector<FusedModelBase::Impl::KvCostObservation>observations;
 for(const auto&r:m.routes){candidate.physical_manifest.routes.push_back({r.site,r.family,r.selector,r.flags,r.arguments,r.invocation});if(r.family==FmbRouteFamily::KV_INSERT)observations.push_back({r.site,4,8,8,256,27,r.arguments});}
 assert(m.physical_manifest_forward_capability(candidate.physical_manifest).supported);
 const auto cost=make_kvinsert_cost_candidate(candidate,observations);assert(cost.reason=="READY"&&cost.rows.size()==2&&observations.size()==2);
 for(const auto&r:cost.rows)assert(r[0]==kFullAttentionKvInsertSite);
 const auto query_snapshot=cache.queries,proof_snapshot=replica_proofs;
 for(int layer:{0,23}){launches.clear();consumed.clear();m.insert_stage(layer);assert(launches==std::vector<std::string>({"joint-cache"}));assert(std::count(consumed.begin(),consumed.end(),site)==1&&std::count(consumed.begin(),consumed.end(),kFullAttentionKvInsertSite)==1);}
 assert(cache.queries==query_snapshot&&replica_proofs==proof_snapshot);
 // Every new ABI and frozen original KV22 argument is checked before writing.
 for(int64_t invocation:{0,1}){
  auto base=m.find_physical_route(FmbRouteFamily::GRAPH_SCHEDULE,site,invocation);
  for(auto a:base.arguments)assert(a>=0);
  for(size_t arg=0;arg<base.arguments.size();++arg){auto x=m;for(auto&r:x.routes)if(r.site==site&&r.invocation==invocation)++r.arguments[arg];reject([&]{x.insert_stage(invocation?23:0);});
   auto altered=candidate.physical_manifest;for(auto&r:altered.routes)if(r.site_id==site&&r.invocation==invocation)++r.arguments[arg];reject([&]{m.physical_manifest_forward_capability(altered);});}
  for(int fault=0;fault<8;++fault){auto x=m;for(auto&r:x.routes)if(r.site==site&&r.invocation==invocation){if(fault==0)++r.site;if(fault==1)r.family=FmbRouteFamily::OTHER;if(fault==2)++r.selector;if(fault==3)++r.flags;if(fault==4)r.invocation+=2;if(fault==5)r.arguments.pop_back();}if(fault==6)x.routes.clear();if(fault==7)x.complete=false;reject([&]{x.insert_stage(invocation?23:0);});}
 }
 // Cost-oracle alternate V2, missing primary KV22 and old/new mismatches fail cold.
 for(int fault=0;fault<4;++fault){auto alternate=candidate.physical_manifest;
  if(fault==0)for(auto&r:alternate.routes)if(r.site_id==kFullAttentionKvInsertSite&&r.invocation==1){KvInsertSegmentPlan p(245);p.route_id=KvInsertRoute::V2;p.parts={{KvInsertKernel::V2,245,0,32}};r.selector=1;r.arguments=rpu_kvinsert_route_arguments(p,8,8,256);}
  if(fault==1)alternate.routes.erase(std::remove_if(alternate.routes.begin(),alternate.routes.end(),[](const auto&r){return r.site_id==kFullAttentionKvInsertSite&&r.invocation==1;}),alternate.routes.end());
  if(fault==2)for(auto&r:alternate.routes)if(r.site_id==site)r.site_id=oldsite;
  if(fault==3)for(auto&r:alternate.routes)if(r.site_id==site)r.family=FmbRouteFamily::KV_INSERT;
  reject([&]{m.physical_manifest_forward_capability(alternate);});
 }
 for(int which=0;which<2;++which){cache.p0=which!=0;cache.p245=which!=1;auto x=m;x.install();x.full_plan(245);
  assert(!x.action_owner_cache_joint_available_&&!has(x,site)&&has(x,oldsite));launches.clear();x.insert_stage(23);assert(launches==std::vector<std::string>({"owner-cache"}));assert(x.kvinsert_cost_weight_identity()!=identity&&x.subclass_layout_hash()!=hash);
  reject([&]{x.physical_manifest_forward_capability(candidate.physical_manifest);});
 }cache.p0=cache.p245=true;
 {auto x=m;x.catalog_v2=true;reject([&]{x.full_plan(245);});}
 {auto x=m;x.action_prefix_lens_[23]=244;reject([&]{x.insert_stage(23);});}
 for(auto bad:std::vector<ChunkInfo>{{31,2,0},{32,2,1}})reject([&]{m.launch_action_owner_cache_insert(23,bad,KvInsertSegmentPlan(245));});
 reject([&]{m.launch_action_owner_cache_insert(-1,{},KvInsertSegmentPlan(0));});reject([&]{m.launch_action_owner_cache_insert(24,{},KvInsertSegmentPlan(245));});
 // Cache writes remain FP16 under both Linear accumulation modes.
 {auto x=m;x.linear_acc32_=true;x.prepare_action_owner_cache_joint();assert(x.action_owner_cache_joint_available_&&x.action_owner_cache_joint_abi()[2]==16);x.full_plan(245);launches.clear();x.insert_stage(23);assert(launches==std::vector<std::string>({"joint-cache"}));}
 for(int fault=1;fault<4;++fault){auto x=m;
  if(fault==0)x.linear_acc32_=true;
  if(fault==1)x.action_owner_cache_available_=false;
  if(fault==2)x.action_len_=16;
  if(fault==3)x.action_nvfp4_.values.clear();
  x.prepare_action_owner_cache_joint();assert(!x.action_owner_cache_joint_available_);
 }
 {auto x=m;x.position=244;x.action_prefix_lens_[23]=244;x.full_plan(244);assert(!has(x,site)&&!has(x,oldsite));launches.clear();x.insert_stage(23);assert(launches==std::vector<std::string>({"cache-original"}));}
 {Model w8;w8.owner_fixture();w8.install();w8.loop();w8.full_plan(245);assert(!w8.action_owner_cache_joint_available_&&has(w8,oldsite));launches.clear();w8.insert_stage(23);assert(launches==std::vector<std::string>({"owner-cache"}));}
 {Model fp16(false);fp16.owner_fixture();fp16.install();fp16.loop();assert(!fp16.action_owner_cache_joint_available_);}
 m.reset();assert(!m.action_owner_cache_joint_available_);m.action_mode_=true;m.joint_fixture();assert(m.action_owner_cache_joint_available_&&m.subclass_layout_hash()==hash);
}
'''
    cpp = cpp.replace(';if(', ';\n if(').replace(';return kv', ';\n return kv').replace(';reject([', ';\n reject([')
    cpp = cpp.replace('throw std::runtime_error("rejected")', 'throw std::runtime_error(std::string("rejected at ")+std::to_string(__LINE__))')
    _compile_run(tmp_path, 'g05_joint_cache_model', cpp)
