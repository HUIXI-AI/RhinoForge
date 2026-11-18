"""Execute the real group4 wrapper and model bodies without device hardware."""
from pathlib import Path

from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]
GROUP_STUBS = """ bool action_nvfp4_qk_norm_rope_available_=false;
 bool use_action_nvfp4_qk_norm_rope(int64_t,int64_t)const{return false;}
 std::vector<int64_t> action_nvfp4_qk_norm_rope_arguments()const{return {};}
 void prepare_action_nvfp4_qk_norm_rope(){}
 template<class...T>void launch_action_nvfp4_qk_norm_rope(T...){assert(false);}
"""


def test_group4_native_wrapper_contract(tmp_path, monkeypatch):
    import test_g05_owner_cache_native as owner
    captured = []
    monkeypatch.setattr(owner, "_compile_run", lambda p, n, cpp: captured.append(cpp))
    owner.test_owner_norm_rope_and_segmented_cache_actual_wrappers(tmp_path)
    cpp = captured[0].split("int main(){", 1)[0]
    cpp += r'''
int main(){
 auto&g=RpuKernelGraph::active();
 constexpr auto Q4=KernelId::G05_Q_NORM_PARTIAL_ROPE_M32D256R64_GROUP4;
 constexpr auto K4=KernelId::G05_K_OWNER_NORM_PARTIAL_ROPE_M32D256R64_GROUP4;
 const uint32_t q=0x40010000,k=0x40020000,qn=0x40030000,kn=0x40040000;
 at::Tensor qc{0x1234000000,{300,32}},qs{0x1235000000,{300,32}},kc{0x1236000000,{300,32}},ks{0x1237000000,{300,32}};
 auto launch=[&]{rpu_launch_g05_qk_norm_rope_group4_spm_kernel(q,k,qn,kn,qc,qs,245,kc,ks,0);};
 for(int pass=0;pass<2;++pass){qc.address+=256;qs.address+=512;kc.address+=768;ks.address+=1024;
  auto begin=g.nodes.size();int64_t qp=pass?0:245,kp=pass?245:0;
  rpu_launch_g05_qk_norm_rope_group4_spm_kernel(q,k,qn,kn,qc,qs,qp,kc,ks,kp);
  assert(g.nodes.size()==begin+2);
  for(int family=0;family<2;++family){const auto&s=g.nodes[begin+family];
   assert(s.id==(family?K4:Q4)&&s.grid==std::vector<uint16_t>({8,1,1}));
   assert(pair(s,4)==(family?k:q)&&pair(s,6)==(family?k:q)&&pair(s,8)==(family?kn:qn));
   assert(s.regs.at(0)==256&&s.regs.at(1)==0x1c00&&s.regs.at(2)==512&&s.regs.at(3)==0&&s.regs.at(10)==0x11);
   assert(pair(s,16)==(family?kc.address:qc.address)>>8&&pair(s,18)==(family?ks.address:qs.address)>>8);
   assert(pair(s,20)==uint32_t(family?kp:qp)&&s.regs.at(22)==1&&s.regs.at(23)==4&&s.regs.at(24)==64&&s.regs.at(25)==1);
   assert(s.operands.size()==2);
   for(int i=0;i<2;++i){const auto&o=s.operands[i];assert(o.abi.role==GraphDdrRegisterRole::SemanticInput&&o.abi.access==GraphDdrRegisterAccess::Read&&o.abi.lo==16+2*i&&o.abi.hi==17+2*i);}
  }
 }
 // Distinct Q/K semantics may legally share immutable table owners.
 rpu_launch_g05_qk_norm_rope_group4_spm_kernel(q,k,qn,kn,qc,qs,0,qc,qs,245);
 for(auto id:{Q4,K4}){KernelCache::instance().missing.insert(id);reject(launch);KernelCache::instance().missing.clear();}
 g.active_flag=false;reject(launch);g.active_flag=true;allocator.initialized=false;reject(launch);allocator.initialized=true;
 for(int which=0;which<4;++which)for(int fault=0;fault<11;++fault){auto a=qc,b=qs,c=kc,d=ks;at::Tensor*owners[]={&a,&b,&c,&d};auto&t=*owners[which];
  if(fault==0)t.present=false;
  if(fault==1)t.location=c10::DeviceType::CPU;
  if(fault==2)t.dtype=2;
  if(fault==3)t.contiguous=false;
  if(fault==4)t.offset=128;
  if(fault==5)t.shape={31,32};
  if(fault==6)t.shape={300,31};
  if(fault==7)t.address+=1;
  if(fault==8)t.address=0;
  if(fault==9)t.address=(UINT64_C(1)<<40)-256;
  if(fault==10)t.shape={300,32,1};
  reject([&]{rpu_launch_g05_qk_norm_rope_group4_spm_kernel(q,k,qn,kn,a,b,0,c,d,0);});
 }
 // Even invalid K input is rejected before Q can stage a graph leaf.
 for(int family=0;family<2;++family){
  for(int64_t pos:{int64_t{-1},int64_t{269},int64_t{4294967296}})
   reject([&]{rpu_launch_g05_qk_norm_rope_group4_spm_kernel(q,k,qn,kn,qc,qs,family?0:pos,kc,ks,family?pos:0);});
  reject([&]{rpu_launch_g05_qk_norm_rope_group4_spm_kernel(q,k,qn,kn,qc,family?qs:qc,0,kc,family?kc:ks,0);});
 }
 for(int which=0;which<4;++which)for(auto bad:{q+1,0x3fffff00u,0x407fff00u}){uint32_t a[]={q,k,qn,kn};a[which]=bad;reject([&]{rpu_launch_g05_qk_norm_rope_group4_spm_kernel(a[0],a[1],a[2],a[3],qc,qs,0,kc,ks,0);});}
 for(int i=0;i<4;++i)for(int j=i+1;j<4;++j){uint32_t a[]={q,k,qn,kn};a[j]=a[i];reject([&]{rpu_launch_g05_qk_norm_rope_group4_spm_kernel(a[0],a[1],a[2],a[3],qc,qs,0,kc,ks,0);});}
 // Last logical row extent is valid, independent of allocated memory padding.
 qc.shape=qs.shape={277,32};kc.shape=ks.shape={32,32};launch();
}
'''
    _compile_run(tmp_path, "g05_group4_native", cpp)


def test_group4_cold_complete_and_fallbacks(tmp_path, monkeypatch):
    import test_g05_owner_cache_model as owner
    captured = []
    monkeypatch.setattr(owner, "_compile_run", lambda p, n, cpp: captured.append(cpp))
    owner.test_owner_cache_cold_admission_routes_and_real_caller_fallbacks(tmp_path)
    cpp = captured[0].split("int main(){", 1)[0]
    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    methods = [
        ("void prepare_action_nvfp4_qk_norm_rope()", "void Qwen3_5Model::prepare_action_nvfp4_qk_norm_rope("),
        ("bool use_action_nvfp4_qk_norm_rope(int64_t rows,int64_t prefix)const", "bool Qwen3_5Model::use_action_nvfp4_qk_norm_rope("),
        ("std::vector<int64_t> action_nvfp4_qk_norm_rope_arguments()const", "std::vector<int64_t> Qwen3_5Model::action_nvfp4_qk_norm_rope_arguments("),
        ("void launch_action_nvfp4_qk_norm_rope(int layer,const ChunkInfo&chunk,const at::Tensor&query_cos,const at::Tensor&query_sin,int64_t query_position)", "void Qwen3_5Model::launch_action_nvfp4_qk_norm_rope("),
    ]
    assert GROUP_STUBS in cpp
    cpp = cpp.replace(GROUP_STUBS, "bool action_nvfp4_qk_norm_rope_available_=false;\n" + "\n".join(d + _block(source, s) for d, s in methods))
    ids = ["G05_Q_NORM_PARTIAL_ROPE_M32D256R64_GROUP4", "G05_K_OWNER_NORM_PARTIAL_ROPE_M32D256R64_GROUP4"]
    cpp = cpp.replace("enum class KernelId{", "enum class KernelId{" + ",".join(ids) + ",")
    cpp = cpp.replace("struct KernelCache{", "struct KernelCache{bool q4=true,k4=true;")
    cpp = cpp.replace("bool has_loaded(KernelId id){++queries;", "bool has_loaded(KernelId id){++queries;if(id==KernelId::" + ids[0] + ")return q4;if(id==KernelId::" + ids[1] + ")return k4;\n ")
    cpp = cpp.replace("idx=2,offset=0", "idx=0,offset=0")
    cpp = cpp.replace("std::vector<int64_t>action_nvfp4_arguments(int)const{assert(false);return {};}", "std::vector<int64_t>action_nvfp4_arguments(int family)const{return {family};}")
    support = r'''
std::vector<int64_t>group4_operands;
void rpu_launch_g05_qk_norm_rope_group4_spm_kernel(uint32_t q,uint32_t k,uint32_t qn,uint32_t kn,const at::Tensor&qc,const at::Tensor&qs,int64_t qp,const at::Tensor&kc,const at::Tensor&ks,int64_t kp){
 assert(q==0x20000&&k==0x28000&&qn>=0x50000&&kn>=0x50000);
 group4_operands={qc.owner,qs.owner,qp,kc.owner,ks.owner,kp};launches.push_back("group4-q-k");}
'''
    cpp = cpp.replace("struct Model{", support + "struct Model{", 1)
    attention = _block(source, "void Qwen3_5Model::build_full_attention(")
    branch = "if (use_action_nvfp4_qk_norm_rope(seq_len,ctx().position))" + _block(attention, "if (use_action_nvfp4_qk_norm_rope(seq_len,ctx().position))")
    # Execute the actual caller branch. Existing norm_stage runs the unchanged
    # fallback body from the same source, including the K-owner continuation.
    caller = r'''
 void group_stage(int layer_idx,bool use_prefill_rope,int64_t cos_sin_start,ChunkInfo chunk={}){
  int64_t seq_len=chunk.len;__BRANCH__ else {norm_stage(layer_idx,chunk);}
 }
 at::Tensor cos_,sin_;
 void group_fixture(){linear_acc32_=false;action_nvfp4_.values={1,2,3,4,5};owner_fixture();
  for(auto&w:layer_weights_){for(auto*t:{&w.q_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})t->dtype=at::kByte;w.q_norm_w.rank=1;w.q_norm_w.rows=256;}
  cos_.owner=700;sin_.owner=701;install();loop();full_plan(245);
 }
'''.replace("__BRANCH__", branch)
    cpp = cpp.replace(" void loop(){", caller + "\n void loop(){", 1)
    cpp += r'''
int main(){
 Model m;m.group_fixture();auto&cache=KernelCache::instance();
 const auto site=QWEN35_ACTION_QK_NORM_ROPE_GROUP4_SITE;
 auto has=[](const Model&x,int64_t id){return std::any_of(x.routes.begin(),x.routes.end(),[&](const auto&r){return r.site==id;});};
 assert(m.action_nvfp4_qk_norm_rope_available_&&m.use_action_nvfp4_qk_norm_rope(32,0)&&m.use_action_nvfp4_qk_norm_rope(32,245));
 assert(!m.use_action_nvfp4_qk_norm_rope(31,245)&&!m.use_action_nvfp4_qk_norm_rope(32,244));
 assert(has(m,site)&&!has(m,QWEN35_ACTION_K_OWNER_NORM_ROPE_SITE));
 const auto args=m.action_nvfp4_qk_norm_rope_arguments();for(auto a:args)assert(a>=0);
 auto identity=m.kvinsert_cost_weight_identity();auto hash=m.subclass_layout_hash();
 FmbPrefillStageCandidate candidate;std::vector<FusedModelBase::Impl::KvCostObservation>observations;
 for(const auto&r:m.routes){candidate.physical_manifest.routes.push_back({r.site,r.family,r.selector,r.flags,r.arguments,r.invocation});if(r.family==FmbRouteFamily::KV_INSERT)observations.push_back({r.site,4,8,8,256,27,r.arguments});}
 const auto cost=make_kvinsert_cost_candidate(candidate,observations);assert(cost.reason=="READY"&&cost.rows.size()==2);
 auto metadata0=metadata,queries=cache.queries,proofs=replica_proofs;
 for(int layer:{0,23})for(int64_t prefix:{0,245}){m.position=prefix;m.full_plan(prefix);
  launches.clear();m.group_stage(layer,true,0);assert(launches==std::vector<std::string>({"group4-q-k"}));
  assert(group4_operands==std::vector<int64_t>({500,501,0,500,501,0}));
  launches.clear();m.group_stage(layer,false,245);assert(launches==std::vector<std::string>({"group4-q-k"}));
  assert(group4_operands==std::vector<int64_t>({700,701,245,500,501,0}));
 }
 assert(metadata==metadata0&&cache.queries==queries&&replica_proofs==proofs);
 // Live frequency ownership/position is not a stale descriptor address.
 m.cos_.owner=900;m.sin_.owner=901;m.group_stage(23,false,17);
 assert(group4_operands==std::vector<int64_t>({900,901,17,500,501,0})&&m.action_nvfp4_qk_norm_rope_arguments()==args);
 for(size_t i=0;i<args.size();++i){auto x=m;for(auto&r:x.routes)if(r.site==site)++r.arguments[i];reject([&]{x.group_stage(0,true,0);});}
 for(int fault=0;fault<8;++fault){auto x=m;for(auto&r:x.routes)if(r.site==site){if(fault==0)++r.site;if(fault==1)r.family=FmbRouteFamily::OTHER;if(fault==2)++r.selector;if(fault==3)++r.flags;if(fault==4)++r.invocation;if(fault==5)r.arguments.pop_back();}if(fault==6)x.routes.clear();if(fault==7)x.complete=false;reject([&]{x.group_stage(0,true,0);});}
 for(auto bad:std::vector<ChunkInfo>{{31,0,0},{32,1,0},{32,0,1}})reject([&]{m.launch_action_nvfp4_qk_norm_rope(0,bad,m.cos_,m.sin_,0);});
 reject([&]{m.launch_action_nvfp4_qk_norm_rope(-1,{},m.cos_,m.sin_,0);});reject([&]{m.launch_action_nvfp4_qk_norm_rope(24,{},m.cos_,m.sin_,0);});
 // Both optional payload omissions fall back before publication.
 for(int which=0;which<2;++which){cache.q4=which!=0;cache.k4=which!=1;auto x=m;x.install();x.full_plan(245);
  assert(!x.action_nvfp4_qk_norm_rope_available_&&!has(x,site)&&has(x,QWEN35_ACTION_K_OWNER_NORM_ROPE_SITE));
  launches.clear();x.group_stage(23,true,0);assert(launches==std::vector<std::string>({"q-norm","q-rope","owner-norm-rope"}));
  assert(x.kvinsert_cost_weight_identity()!=identity&&x.subclass_layout_hash()!=hash);
 }cache.q4=cache.k4=true;
 for(int family=0;family<2;++family)for(int fault=0;fault<6;++fault){auto x=m;auto&n=family?x.layer_weights_.back().k_norm_w:x.layer_weights_.back().q_norm_w;
  if(fault==0)n.present=false;
  if(fault==1)n.location=c10::DeviceType::CPU;
  if(fault==2)n.dtype=at::kChar;
  if(fault==3)n.contiguous=false;
  if(fault==4)n.rank=2;
  if(fault==5)n.rows=128;
  x.prepare_action_nvfp4_qk_norm_rope();assert(!x.action_nvfp4_qk_norm_rope_available_);
 }
 for(int fault=0;fault<10;++fault){auto x=m;
  if(fault==0)x.has_qk_norm_=false;
  if(fault==1)x.has_mrope_=false;
  if(fault==2)x.rotary_dim_=256;
  if(fault==3)x.eps_=1e-5;
  if(fault==4)x.heads=16;
  if(fault==5)x.kv=2;
  if(fault==6)x.hd=128;
  if(fault==7)x.tp=4;
  if(fault==8)x.layer_weights_.pop_back();
  if(fault==9)x.action_owner_cache_available_=false;
  x.prepare_action_nvfp4_qk_norm_rope();assert(!x.action_nvfp4_qk_norm_rope_available_);
 }
 {auto x=m;x.linear_acc32_=true;x.prepare_action_nvfp4_qk_norm_rope();assert(x.action_nvfp4_qk_norm_rope_available_);x.full_plan(245);launches.clear();x.group_stage(23,false,17);assert(launches==std::vector<std::string>({"group4-q-k"}));}
 {Model w8;w8.owner_fixture();w8.install();w8.loop();assert(!w8.action_nvfp4_qk_norm_rope_available_);}
 {Model fp16(false);fp16.owner_fixture();fp16.install();fp16.loop();assert(!fp16.action_nvfp4_qk_norm_rope_available_);}
 {auto x=m;++x.layer_weights_.back().q_norm_w.owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 {auto x=m;++x.layer_weights_.back().k_norm_w.owner;assert(x.kvinsert_cost_weight_identity()!=identity);}
 m.reset();assert(!m.action_nvfp4_qk_norm_rope_available_);m.action_mode_=true;m.group_fixture();assert(m.action_nvfp4_qk_norm_rope_available_&&m.subclass_layout_hash()==hash);
}
'''
    cpp = cpp.replace(";if(", ";\n if(").replace(";return kv", ";\n return kv").replace(";reject([", ";\n reject([")
    _compile_run(tmp_path, "g05_group4_model", cpp)
