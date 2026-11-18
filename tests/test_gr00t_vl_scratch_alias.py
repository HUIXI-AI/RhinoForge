"""Long-prefix VL scratch declarations, real allocator, and Z2 isolation."""
from pathlib import Path
import re
import shutil
import subprocess

from test_fmb_standalone_layout_rebuild import _definition
from test_gr00t_mask_residency import _program

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_gr00t_vl_encoder_model.cpp"


def test_vl_long_prefix_alias_real_allocator_and_z2_prepare(tmp_path):
    source = SOURCE.read_text()
    # Reuse the actual tiling, BufferDecl and allocator extraction, without the
    # DiT fixture. Preserve every production VL declaration and size.
    cpp = _program(full_declarations=True).split("class Gr00tDiTModel{public:")[0]
    declaration = _definition(source, "std::vector<BufferDecl> declare_buffers(")
    fields = sorted(set(re.findall(r"&Gr00tVlEncoderModel::(\w+)", declaration)))
    layer_fields = sorted(set(re.findall(r"&LayerWeights::(\w+)", declaration)))
    while "d.preload_callback =" in declaration:
        callback = _definition(declaration, "d.preload_callback =")
        declaration = declaration.replace(callback, "d.preload_callback=nullptr", 1)
    cpp += r'''
}
#include <memory>
namespace v3 {
namespace c10 {
template<class F>struct ScopeExit{F f;~ScopeExit(){f();}};
template<class F>ScopeExit<F>make_scope_exit(F f){return {f};}
}
struct SpmPipelineComponentLayout{int id=1;};
enum class ChunkMode{SEQUENTIAL};
// The real prepare_z2_layout transaction below consumes these boundary doubles.
// Codec/cache internals have separate tests; retain the actual allocator here.
struct StagePlan{struct Stage{struct{int64_t chunk_size;}plan;}input;};
StagePlan compose_fmb_default_three_stage_chunk_plan(const LayoutContext& l,int64_t seq,int64_t pos,ChunkMode){
 assert(l.chunk_size==seq&&pos==0);return {{{seq}}};
}
enum class FmbPhysicalManifestState{COMPLETE,INCOMPLETE};
enum class FmbGraphLifecycle{COMPOSITE_CHILD,RETAINED_CACHE};
struct Manifest{
 FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
 FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
 int64_t rows;
};
struct Candidate{int64_t input_chunk_size,qkv_chunk_size,compute_chunk_size;StagePlan stage_plan;Manifest physical_manifest;};
std::map<int64_t,Candidate> encoded_candidates;
std::vector<int64_t> encode_fmb_prefill_stage_candidate(const Candidate& c){
 assert(c.input_chunk_size==c.physical_manifest.rows);
 assert(c.qkv_chunk_size==Align(c.input_chunk_size,16)&&c.compute_chunk_size==c.qkv_chunk_size);
 const int64_t id=encoded_candidates.size()+1;encoded_candidates.emplace(id,c);return {id};
}
struct Prepared{Candidate value;const Candidate&candidate()const{return value;}uint64_t stage_fingerprint()const{return 17;}};
class Gr00tVlEncoderModel{public:
 bool z2_layout_preparing_=false,z2_bound_=false,prepare_failure=false;
 int candidate_failure=0,component_calls=0;std::vector<int64_t>z2_stage_descriptor_;
 int64_t z2_prepared_seq_len_=0,h=2048,nq=32,nkv=32,hd=64,ff=8192,layers=4,tp=8;
 int64_t hidden_size()const{return h;}int64_t num_q_heads()const{return nq;}
 int64_t num_kv_heads()const{return nkv;}int64_t head_dim()const{return hd;}
 int64_t intermediate_size()const{return ff;}int64_t num_layers()const{return layers;}
 int attn_tp()const{return tp;}int num_cores()const{return 8;}
 SdpaConfig make_sdpa_config(int mask)const{return {SdpaKernelType::FLASH_ATTN_SPM,hd,nq,nq,8,mask};}
 __FIELDS__
 struct LayerWeights{__LAYER_FIELDS__};std::vector<LayerWeights>layer_weights_=std::vector<LayerWeights>(4);
 __DECLARATION__
 Manifest physical_manifest_for_candidate(const StagePlan& p,const LayoutContext& l,int64_t logical,int64_t physical,int64_t pos){
  assert(z2_layout_preparing_&&z2_prepared_seq_len_==logical);
  assert(p.input.plan.chunk_size==logical&&l.chunk_size==logical&&logical==physical&&pos==0);
  for(const auto&d:declare_buffers(l))assert(d.alias_of==nullptr);
  TORCH_CHECK(candidate_failure!=1);return {FmbPhysicalManifestState::COMPLETE,FmbGraphLifecycle::COMPOSITE_CHILD,logical};
 }
 std::shared_ptr<Prepared> prepare_stage_candidate(const std::vector<int64_t>&descriptor){
  TORCH_CHECK(candidate_failure!=2);auto value=encoded_candidates.at(descriptor.at(0));
  if(candidate_failure==3)value.physical_manifest.state=FmbPhysicalManifestState::INCOMPLETE;
  if(candidate_failure==4)value.physical_manifest.graph_lifecycle=FmbGraphLifecycle::RETAINED_CACHE;
  return std::make_shared<Prepared>(Prepared{value});
 }
 LayoutContext bind_physical_layout_context(LayoutContext l,const Manifest& manifest){
  assert(l.stage_plan_fingerprint==17&&l.chunk_size==manifest.rows);
  TORCH_CHECK(candidate_failure!=5);l.physical_manifest_fingerprint=31;return l;
 }
 SpmPipelineComponentLayout prepare_spm_pipeline_component(const LayoutContext& layout,const StagePlan&plan){
  ++component_calls;
  assert(z2_layout_preparing_&&layout.stage_plan_fingerprint==17&&layout.physical_manifest_fingerprint==31);
  assert(plan.input.plan.chunk_size==layout.chunk_size);
  for(const auto&d:declare_buffers(layout))assert(d.alias_of==nullptr);
  TORCH_CHECK(!prepare_failure);return {};
 }
 __PREPARE__
};
bool aliased(const std::vector<BufferDecl>&d){
 for(const auto&b:d)if(std::string(b.name)=="oproj")return b.alias_of&&std::string(b.alias_of)=="norm_hidden";
 assert(false);return false;
}
int64_t fixed(const std::vector<BufferDecl>&d){int64_t n=0;for(const auto&b:d){
 if(b.storage==StorageClass::Persistent)n+=b.size;
 if(b.storage==StorageClass::PersistentPerLayer)n+=b.size*b.per_layer;}return n;}
template<class F>void rejects(F f){try{f();assert(false);}catch(const std::runtime_error&){}}
}
int main(){using namespace v3;
 Gr00tVlEncoderModel model;int64_t fixed_bytes=-1;
 constexpr int64_t four_step_available=8303616-1586688-65536;
 // T128/160/192/224/256: 256px P=T+140/206; 224px P=T+110/161.
 // Retain the original short profiles and the declared capacity ceiling.
 for(int seq:{193,225,238,256,268,270,289,300,302,321,332,334,353,364,366,384,385,396,398,417,430,462,512})
 for(bool raw:{false,true})for(bool capacity:{false,true}){
  LayoutContext layout;layout.chunk_size=capacity?Align(seq,16):seq;layout.max_kv_seq_len=seq;
  layout.batch_size=1;layout.is_causal=false;layout.physical_manifest_fingerprint=123;
  layout.attention_policy=raw?AttentionExecutionPolicy::SPM_KV_BY_MHA:AttentionExecutionPolicy::DDR_KV;
  const auto decl=model.declare_buffers(layout);const auto peak=detail::estimate_temporary_total(decl);
  const int64_t cs=layout.chunk_size,res=Align(cs*2048*2,256);
  assert(aliased(decl)==(cs>256));
  if(fixed_bytes<0)fixed_bytes=fixed(decl);assert(fixed_bytes==fixed(decl));
  auto baseline=decl,previous=decl;
  for(auto&d:baseline){d.alias_of=nullptr;d.phase_start=d.phase_end=0;}
  for(auto&d:previous){if(std::string(d.name)!="oproj")d.alias_of=nullptr;d.phase_start=d.phase_end=0;}
  assert(detail::estimate_temporary_total(baseline)-detail::estimate_temporary_total(previous)==(cs>256?res:0));
  assert(peak<=four_step_available);
  if(cs>256)assert(peak<detail::estimate_temporary_total(previous));
  if(seq>=384)assert(detail::estimate_temporary_total(previous)>four_step_available);
  // Dry planning and COMPLETE forward planning must reserve the same roots.
  auto dry=layout;dry.physical_manifest_fingerprint=0;
  assert(detail::estimate_temporary_total(model.declare_buffers(dry))==peak);
  std::vector<SpmAllocator::AllocRequest>req;std::vector<std::string>names;
  std::map<std::string,int64_t>sizes,offsets;
  for(const auto&d:decl)if(d.storage==StorageClass::Temp){sizes[d.name]=d.size;
   if(!d.alias_of){names.push_back(d.name);req.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});}}
  const auto plan=SpmAllocator::plan_temporary_aliased(req);assert(int64_t(plan.peak_bytes)==peak);
  for(size_t i=0;i<req.size();++i)offsets[names[i]]=plan.offsets[i];
  for(const auto&d:decl)if(d.alias_of){assert(sizes[d.name]<=sizes[d.alias_of]);offsets[d.name]=offsets.at(d.alias_of);}
  if(cs>256){assert(offsets["norm_hidden"]==offsets["oproj"]);
   assert(offsets["ff_mid"]==offsets["residual1"]);}
  for(auto group:{std::vector<std::string>{"oproj","residual1","residual2"},
                 std::vector<std::string>{"residual1","norm_hidden","q","k","v","sdpa_tmp","sdpa_out"},
                 std::vector<std::string>{"norm_hidden","ff_mid","residual2"}})
   for(size_t i=0;i<group.size();++i)for(size_t j=i+1;j<group.size();++j){auto a=group[i],b=group[j];
    assert(offsets[a]+sizes[a]<=offsets[b]||offsets[b]+sizes[b]<=offsets[a]);}
  // Replay the layer's read/write lifetimes in the real allocator's storage.
  // The maximum requested prefix exercises both raw and DDR layouts and both
  // exact-row and rounded-capacity declarations, across all four layers.
  if(seq==417||seq==462){
   std::vector<unsigned char>spm(peak,0);std::map<std::string,unsigned char>value;unsigned char stamp=0;
   auto write=[&](const std::string&name){value[name]=++stamp;
    std::fill(spm.begin()+offsets[name],spm.begin()+offsets[name]+sizes[name],stamp);};
   auto read=[&](const std::string&name){assert(value.count(name));
    assert(std::all_of(spm.begin()+offsets[name],spm.begin()+offsets[name]+sizes[name],
      [&](unsigned char byte){return byte==value[name];}));};
   write("residual1");
   for(int layer=0;layer<4;++layer){
    if(layer==0){read("residual1");write("norm_hidden");read("norm_hidden");write("residual1");}
    read("residual1");write("norm_hidden");
    for(auto name:{"q","k","v"}){read("norm_hidden");write(name);}
    read("v");write("sdpa_tmp");
    for(auto name:{"q","k","v","sdpa_tmp"})read(name);write("sdpa_out");
    read("sdpa_out");write("oproj");
    read("oproj");read("residual1");write("residual2");
    read("residual2");write("norm_hidden");
    read("norm_hidden");write("ff_mid");read("ff_mid");write("ff_mid");
    read("ff_mid");write("oproj");
    read("oproj");read("residual2");write("residual1");
   }
   read("residual1");
  }
  std::cout<<"vl-layout,"<<seq<<","<<cs<<","<<raw<<","<<fixed(decl)<<","<<peak
           <<","<<detail::estimate_temporary_total(previous)<<"\n";
 }
 LayoutContext long_layout;long_layout.chunk_size=334;long_layout.batch_size=1;long_layout.is_causal=false;
 assert(aliased(model.declare_buffers(long_layout)));
 // Every unsupported mode keeps the historical independent storage.
 for(int bad=0;bad<10;++bad){auto x=model;auto l=long_layout;switch(bad){
  case 0:x.z2_layout_preparing_=true;break;case 1:x.z2_bound_=true;break;
  case 2:x.z2_prepared_seq_len_=334;break;case 3:l.batch_size=2;break;
  case 4:l.is_causal=true;break;case 5:l.use_attn_mask=true;break;
  case 6:x.layers=3;break;case 7:x.tp=4;break;case 8:x.nkv=16;break;case 9:l.chunk_size=256;break;}
  assert(!aliased(x.declare_buffers(l)));}
 for(int seq:{82,334}){auto x=model;x.prepare_z2_layout(seq);
  assert(!x.z2_layout_preparing_&&x.z2_prepared_seq_len_==seq&&x.component_calls==1);
  assert(encoded_candidates.at(x.z2_stage_descriptor_.at(0)).physical_manifest.rows==seq);
  assert(!aliased(x.declare_buffers(long_layout)));}
 // Any failure before component admission rolls back both cold state and the
 // previous descriptor; COMPLETE and COMPOSITE_CHILD are independent guards.
 for(int failure=1;failure<=5;++failure)for(bool was_preparing:{false,true}){
  auto x=model;x.prepare_z2_layout(82);const auto descriptor=x.z2_stage_descriptor_;
  x.z2_layout_preparing_=was_preparing;x.candidate_failure=failure;
  rejects([&]{x.prepare_z2_layout(334);});
  assert(x.z2_layout_preparing_==was_preparing&&x.z2_prepared_seq_len_==82);
  assert(x.z2_stage_descriptor_==descriptor&&x.component_calls==1);
 }
 {auto x=model;x.prepare_z2_layout(82);const auto descriptor=x.z2_stage_descriptor_;
  x.prepare_failure=true;rejects([&]{x.prepare_z2_layout(334);});
  assert(!x.z2_layout_preparing_&&x.z2_prepared_seq_len_==82&&x.z2_stage_descriptor_==descriptor);}
 model.prepare_failure=true;rejects([&]{model.prepare_z2_layout(334);});
 assert(!model.z2_layout_preparing_&&model.z2_prepared_seq_len_==0);
 assert(aliased(model.declare_buffers(long_layout)));
}
'''
    cpp = (cpp.replace("__FIELDS__", "at::Tensor " + ",".join(fields) + ";")
           .replace("__LAYER_FIELDS__", "at::Tensor " + ",".join(layer_fields) + ";")
           .replace("__DECLARATION__", declaration.replace(" override", ""))
           .replace("__PREPARE__", _definition(source, "SpmPipelineComponentLayout prepare_z2_layout(")))
    path, exe = tmp_path / "vl_layout.cpp", tmp_path / "vl_layout"
    path.write_text(cpp)
    result = subprocess.run([shutil.which("g++"), "-std=c++17", "-O1", str(path), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout)


def test_vl_alias_matches_actual_projection_and_ring_lifetimes():
    block = _definition(SOURCE.read_text(), "void build_layer_subgraph(")
    # Anchor the allocator simulation to the actual layer stream, including
    # both attention branches, their last reads before residual2 is written,
    # and GELU/FF2's last reads before residual1 is reused by the final ring.
    # Moving any of these calls must re-establish the lifetime proof.
    calls = re.findall(r"(rpu_launch_\w+)\((.*?)\);", block, re.S)
    accesses = []
    scratch = {
        "norm_hidden", "oproj", "residual1", "residual2", "ff_mid",
        "q", "k", "v", "sdpa_tmp", "sdpa_out",
    }
    for name, args in calls:
        operands = re.findall(r'addr(?:_offset)?\((?:0,\s*)?"(\w+)"\)', args)
        if scratch.intersection(operands):
            accesses.append((name, operands))
    assert accesses == [
        ("rpu_launch_eltwise_binary_1xC_NxC_spm_kernel", ["inv_w_rms", "residual1", "norm_hidden"]),
        ("rpu_launch_layernorm_spm_kernel", ["norm_hidden", "residual1", "vlln_w", "vlln_b"]),
        ("rpu_launch_layernorm_spm_kernel", ["residual1", "norm_hidden"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["norm_hidden", "q"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["norm_hidden", "k"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["norm_hidden", "v"]),
        ("rpu_launch_insert_kvcache_spm_unified_with_plan", ["k", "v"]),
        ("rpu_launch_v_transpose_spm", ["v", "sdpa_tmp"]),
        ("rpu_launch_sdpa_by_mha_spm", ["q", "k", "sdpa_tmp", "sdpa_out"]),
        ("rpu_launch_sdpa_spm_unified_kernel_v2", ["q", "sdpa_out", "sdpa_tmp"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["sdpa_out", "oproj"]),
        ("rpu_launch_all_reduce_sum_residual_kernel", ["oproj", "residual1", "residual2"]),
        ("rpu_launch_layernorm_spm_kernel", ["residual2", "norm_hidden"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["norm_hidden", "ff_mid"]),
        ("rpu_launch_eltwise_unary_spm_kernel", ["ff_mid", "ff_mid"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["ff_mid", "oproj"]),
        ("rpu_launch_all_reduce_sum_residual_kernel", ["oproj", "residual2", "residual1"]),
    ]
