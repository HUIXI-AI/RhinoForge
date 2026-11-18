"""HALO mask residency capacity, route admission and per-graph initialization."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition
from _halo_mask_host import _program as _temporary_program

ROOT = Path(__file__).resolve().parents[1]


def _without_callbacks(source):
    pattern = re.compile(r"\.preload_callback\s*=\s*")
    offset = 0
    while match := pattern.search(source, offset):
        start = match.end()
        assert source[start] == "["
        body = _definition(source[start:], "[")
        source = source[:start] + "{}" + source[start + len(body):]
        offset = start + 2
    return source


def _program():
    base = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    halo = (ROOT / "src/fused/rpu_halo_action_expert_model.cpp").read_text()
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    allocator_h = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    p = '#include "src/fused/rpu_qwen3_w8a8_contract.h"\n#include "src/fused/rpu_qwen3_c192_contract.h"\n' + _temporary_program()
    constants = allocator_h[allocator_h.index("    static constexpr size_t SPM_TOTAL"):
                            allocator_h.index("    static constexpr size_t ALIGN")]
    p = p.replace("class SpmAllocator {public:", "class SpmAllocator {public:\n" + constants)
    p = p.replace("static AliasedPlan plan_temporary_aliased", r'''
    size_t held=0,temporary=0;
    size_t free_space()const{return SPM_USABLE-held-temporary;}
    size_t temporary_used()const{return temporary;}
    static AliasedPlan plan_temporary_aliased''')
    p += r'''
#define TORCH_INTERNAL_ASSERT(c) TORCH_CHECK(c)
namespace at {struct Tensor {int64_t numel()const{return 0;}};}
class CausalDecoderModel {public:
 int64_t h=1536,nq=12,nkv=2,hd=128,inter=8960,layers=28;
 bool wall_prefill_staged_mlp_=false,decode_gate_up_fusion_=false;
 bool prefill_packed_gate_up_strided_=false;
 bool lingbot_folded_rmsnorm_=false,lingbot_prefill_kv_only_=false;
 // These capabilities are outside the HALO profile fixed by this fixture.
 bool qwen3vl_4b_packed_decode_qkv(int64_t,bool,bool,int64_t)const{return false;}
 bool lingbot_prefill_deepstack_reuses_down(const LayoutContext&)const{return false;}
 bool qwen3vl_multiview_text_dry_prepared_=false,qwen3vl_multiview_text_composite_prepared_=false;
 int qwen3vl_pooler_z1_retained_deepstack_count_=0;
 std::vector<int> deepstack_lang_layers_;
 bool adarms_mutable_=false,adarms_fused_bcast_enabled_=false,adarms_=false;
 bool has_qk_norm_=true,has_qkv_bias_=true,nvfp4_=false;
 bool qwen3_prefill_w8a8_=false,qwen3_c192_=false,qwen3_c256_=false,qwen3_c512_=false;
 int64_t qwen3_inplace_capacity()const{return 0;}
 struct LayerWeights {at::Tensor q_bias_w,k_bias_w,v_bias_w;};
 at::Tensor q_nvfp4_tensor_scales_,k_nvfp4_tensor_scales_,v_nvfp4_tensor_scales_,
 o_nvfp4_tensor_scales_,gate_nvfp4_tensor_scales_,up_nvfp4_tensor_scales_,down_nvfp4_tensor_scales_;
 bool norm_slab_active()const{return false;} int64_t norm_slab_size_bytes()const{return 0;}
 int64_t hidden_size()const{return h;} int64_t num_q_heads()const{return nq;}
 int64_t num_kv_heads()const{return nkv;} int64_t head_dim()const{return hd;}
 int64_t intermediate_size()const{return inter;} int64_t num_layers()const{return layers;}
 int attn_tp()const{return std::min(int(nkv),8);} int mlp_tp()const{return 8;}
 int lm_head_tp()const{return 8;}
 SdpaConfig make_sdpa_config(int m)const{return {SdpaKernelType::FLASH_ATTN_SPM,hd,nq,nkv,attn_tp(),m,0};}
'''
    p += _without_callbacks(_definition(base, "std::vector<BufferDecl> causal_baseline_buffer_declarations("))
    p += r'''
 std::vector<BufferDecl> declare_buffers(const LayoutContext&l){return causal_baseline_buffer_declarations(l);}
};
class HaloActionExpertModel:public CausalDecoderModel {public:
 int64_t moe_chunk_size_=18;
 bool persistent_allocated=false;
 struct MoeLayerWeights {at::Tensor input_norm_w,post_norm_w,q_norm_w,k_norm_w,q_bias,k_bias,v_bias;};
 std::vector<BufferDecl> declare_buffers(const LayoutContext&);
 std::vector<BufferDecl> baseline_buffer_declarations(const LayoutContext&)const;
 FmbForwardOperandResidency mask_residency_for_candidate(const LayoutContext&)const;
 bool declared_spm_layout_fits(const std::vector<BufferDecl>&)const;
};
'''
    for method in ("baseline_buffer_declarations", "declare_buffers"):
        p += _without_callbacks(_definition(halo, f"std::vector<BufferDecl> HaloActionExpertModel::{method}("))
    p += _definition(halo, "FmbForwardOperandResidency\nHaloActionExpertModel::mask_residency_for_candidate(")
    p += "\nSpmAllocator fake_occupancy;\n#define SPM_ALLOC fake_occupancy\n"
    for marker in ("struct FixedSpmOverhead", "static FixedSpmOverhead estimate_fixed_overhead(",
                   "static int64_t available_temporary_spm_after_reset(", "static bool final_chunk_layout_fits("):
        p += _definition(fmb, marker) + (";\n" if marker.startswith("struct") else "\n")
    p += r'''
bool HaloActionExpertModel::declared_spm_layout_fits(const std::vector<BufferDecl>&d)const{
 return final_chunk_layout_fits(d,persistent_allocated);
}
void verify_mask_not_overwritten(const std::vector<BufferDecl>&d) {
 std::vector<SpmAllocator::AllocRequest> r;size_t mask=0;
 for(auto&b:d)if(b.storage==StorageClass::Temp&&!b.alias_of){
  if(std::string(b.name)=="sdpa_mask")mask=r.size();
  r.push_back({b.size,b.phase_start,b.phase_end,int(b.scope)});
 }
 auto p=SpmAllocator::plan_temporary_aliased(r);
 size_t lo=p.offsets[mask],hi=lo+r[mask].bytes;
 std::vector<unsigned char> spm(p.peak_bytes,0);
 std::fill(spm.begin()+lo,spm.begin()+hi,0xa5);
 for(int layer=0;layer<28;++layer)for(size_t i=0;i<r.size();++i)if(i!=mask){
  size_t a=p.offsets[i],b=a+r[i].bytes;assert(b<=lo||a>=hi);
  std::fill(spm.begin()+a,spm.begin()+b,layer+1);
 }
 assert(std::all_of(spm.begin()+lo,spm.begin()+hi,[](auto x){return x==0xa5;}));
}
void run(int64_t rows,int64_t kv){
 HaloActionExpertModel m;LayoutContext l;l.chunk_size=rows;l.max_kv_seq_len=kv;
 l.use_attn_mask=true;l.is_causal=false;l.num_layers=28;
 auto d=m.declare_buffers(l);l.forward_operand_residency=FmbForwardOperandResidency::FORWARD;
 auto candidate=m.declare_buffers(l);int64_t mask_bytes=0;
 for(auto&b:candidate)if(std::string(b.name)=="sdpa_mask")mask_bytes=b.size;
 auto f=estimate_fixed_overhead(d),fc=estimate_fixed_overhead(candidate);
 auto old=detail::estimate_temporary_total(d),next=detail::estimate_temporary_total(candidate);
 assert(f.total()==fc.total()&&f.temp_per_layer==0&&next>old);
 auto no_growth=d;assert(!detail::plan_forward_spm_residency(no_growth,{"sdpa_mask"}).retained);
 verify_mask_not_overwritten(candidate);
 // Existing FMB arithmetic must agree before/after allocating our own fixed
 // buffers, and must subtract every other owner's held bytes in both states.
 auto maximum_other=SpmAllocator::SPM_USABLE-65536-f.total()-next;
 assert(maximum_other%256==0);
 for(int64_t other:{int64_t(0),int64_t(maximum_other),int64_t(maximum_other+256)}){
  for(bool warm:{false,true}){
   fake_occupancy.held=other+(warm?f.persistent:0);fake_occupancy.temporary=4096;
   auto available=available_temporary_spm_after_reset(candidate,warm);
   bool candidate_fits=final_chunk_layout_fits(candidate,warm);
   bool baseline_fits=final_chunk_layout_fits(d,warm);
   assert(candidate_fits==(other<=maximum_other));assert(baseline_fits);
   std::cout<<"capacity,"<<rows<<","<<kv<<","<<other<<","<<warm<<","<<available
            <<","<<baseline_fits<<","<<candidate_fits<<"\n";
  }
 }
 fake_occupancy.held=maximum_other+(next-old)+256;fake_occupancy.temporary=0;
 assert(!final_chunk_layout_fits(d,false)&&!final_chunk_layout_fits(candidate,false));
 if(rows==18){
  auto cold=l;cold.forward_operand_residency=FmbForwardOperandResidency::UNSPECIFIED;
  cold.planning_chunk_capacity=32;auto capacity=cold;capacity.chunk_size=32;
  capacity.forward_operand_residency=FmbForwardOperandResidency::FORWARD;
  const auto capacity_peak=detail::estimate_temporary_total(m.declare_buffers(capacity));
  const int64_t limit=SpmAllocator::SPM_USABLE-65536-f.total()-capacity_peak;
  for(int64_t other:{int64_t(0),limit,limit+256})for(bool warm:{false,true}){
   m.persistent_allocated=warm;fake_occupancy.held=other+(warm?f.persistent:0);
   auto choice=m.mask_residency_for_candidate(cold);
   assert(choice==(other<=limit?FmbForwardOperandResidency::FORWARD:FmbForwardOperandResidency::PER_LAYER));
   auto bound=cold;bound.forward_operand_residency=choice;
   assert(final_chunk_layout_fits(m.declare_buffers(bound),warm));
   bound.chunk_size=32;assert(final_chunk_layout_fits(m.declare_buffers(bound),warm));
   std::cout<<"dual-admission,"<<kv<<","<<other<<","<<warm<<","<<int(choice)<<"\n";
  }
  fake_occupancy.held=0;m.persistent_allocated=false;
  auto invalid=cold;invalid.planning_chunk_capacity=0;
  assert(m.mask_residency_for_candidate(invalid)==FmbForwardOperandResidency::PER_LAYER);
  invalid=cold;invalid.is_causal=true;
  assert(m.mask_residency_for_candidate(invalid)==FmbForwardOperandResidency::PER_LAYER);
  invalid=cold;invalid.batch_size=2;
  assert(m.mask_residency_for_candidate(invalid)==FmbForwardOperandResidency::PER_LAYER);
  invalid=cold;invalid.use_attn_mask=false;
  assert(m.mask_residency_for_candidate(invalid)==FmbForwardOperandResidency::PER_LAYER);
  auto bound=cold;bound.forward_operand_residency=FmbForwardOperandResidency::FORWARD;
  fake_occupancy.held=limit+256;
  assert(m.mask_residency_for_candidate(bound)==FmbForwardOperandResidency::FORWARD);
  auto probe=bound;probe.chunk_size=32;
  assert(!final_chunk_layout_fits(m.declare_buffers(probe),false));
  bound.forward_operand_residency=FmbForwardOperandResidency::PER_LAYER;fake_occupancy.held=0;
  assert(m.mask_residency_for_candidate(bound)==FmbForwardOperandResidency::PER_LAYER);
  bound.forward_operand_residency=static_cast<FmbForwardOperandResidency>(3);
  bool rejected=false;try{m.mask_residency_for_candidate(bound);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);rejected=false;
  try{m.declare_buffers(bound);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 }
 std::cout<<"layout,"<<rows<<","<<kv<<","<<f.persistent<<","<<f.temp_per_layer<<","<<old
          <<","<<next<<","<<mask_bytes<<","<<f.total()+next<<","<<maximum_other<<"\n";
 if(rows==18&&kv==4724)for(auto&b:d)if(b.storage!=StorageClass::Temp&&!b.alias_of)
  std::cout<<"fixed,"<<b.name<<","<<b.size<<","<<b.per_layer<<"\n";
}
} // namespace v3
int main(){
 for(auto rows:{18,32})for(auto kv:{4724,8192})v3::run(rows,kv);
}
'''
    return p


def test_halo_complete_declarations_and_physical_capacity_boundary(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "halo.cpp", tmp_path / "halo"
    cpp.write_text(_program())
    build = subprocess.run([compiler, "-std=c++17", "-O0", "-I", str(ROOT), str(cpp), "-o", str(binary)], text=True, capture_output=True)
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout)
    assert result.stdout.count("layout,") == 4
    assert result.stdout.count("capacity,") == 24


def test_halo_descriptor_controls_every_graph_mask_initialization(tmp_path):
    source = (ROOT / "src/fused/rpu_halo_action_expert_model.cpp").read_text()
    body = _definition(source, "void HaloActionExpertModel::build_layer_subgraph(")
    begin = body.index("    bool upload_mask = true;")
    end = body.index("    if (ctx().has_complete_physical_manifest()) {", begin + 55)
    # The second COMPLETE block is the existing attention-route consumer.
    while "if (upload_mask)" not in body[begin:end]:
        end = body.index("    if (ctx().has_complete_physical_manifest()) {", end + 1)
    emit = body[begin:end]
    program = r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c,...) do{if(!(c))throw std::runtime_error("guard");}while(0)
enum class FmbForwardOperandResidency{UNSPECIFIED=0,PER_LAYER=1,FORWARD=2};
enum class FmbPhysicalManifestState{NONE,COMPLETE};
enum class FmbRouteFamily{GRAPH_SCHEDULE};
struct FmbRouteManifestEntry{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>arguments;int64_t invocation=0;};
struct FmbPhysicalExecutionManifest{FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;std::vector<FmbRouteManifestEntry>routes;};
constexpr int64_t HALO_ACTION_MASK_SPM_SCHEDULE_SITE=3893593376471794725LL;
int uploads=0;
void sdpa_dma_mask_to_spm(int,int,int,int,int){++uploads;}
struct Context{
 bool complete=true;int64_t batch_size=1,body_iter=0;
 FmbRouteManifestEntry route{HALO_ACTION_MASK_SPM_SCHEDULE_SITE,FmbRouteFamily::GRAPH_SCHEDULE,2,0,{1,18,4724,2,28,18}};
 bool has_complete_physical_manifest()const{return complete;}
 const FmbRouteManifestEntry&find_physical_route(FmbRouteFamily family,int64_t site)const{
  TORCH_CHECK(route.site_id==site&&route.family==family);return route;
 }
 void consume_physical_route(FmbRouteFamily,int64_t,int64_t selector,int64_t flags,std::initializer_list<int64_t>a)const{
  TORCH_CHECK(selector==route.selector&&flags==route.flags&&std::vector<int64_t>(a)==route.arguments);
 }
};
class HaloActionExpertModel{public:
 int64_t moe_chunk_size_=18;Context context;
 int attn_tp()const{return 2;}int num_layers()const{return 28;}
 Context&ctx(){return context;}
 FmbForwardOperandResidency physical_forward_operand_residency(const FmbPhysicalExecutionManifest&)const;
 void emit(int layer_idx){int seq_len=18,tp=2,mask_off=0,prepared_attn_mask_=0;
  struct{int kv_seq_len=4724;}chunk;
'''+emit+'}\n};\n'
    program += _definition(source, "FmbForwardOperandResidency\nHaloActionExpertModel::physical_forward_operand_residency(")
    program += r'''
int main(){
 HaloActionExpertModel m;FmbPhysicalExecutionManifest manifest;
 for(int choice:{1,2}){
  m.context.route.selector=choice;manifest.routes={m.context.route};
  assert(int(m.physical_forward_operand_residency(manifest))==choice);
  for(int graph=0;graph<4;++graph){uploads=0;
   for(int layer=0;layer<28;++layer)m.emit(layer);
   assert(uploads==(choice==1?28:1));
  }
 }
 m.context.complete=false;uploads=0;for(int layer=0;layer<28;++layer)m.emit(layer);assert(uploads==28);
 m.context.complete=true;
 auto good=m.context.route;
 for(int field=0;field<11;++field){
  auto bad=good;
  if(field==0)bad.selector=3;else if(field==1)bad.flags=1;else if(field==2)bad.invocation=1;
  else if(field==3)bad.arguments.pop_back();else if(field==4)++bad.site_id;
  else if(field==7)bad.arguments[2]=17;else ++bad.arguments[field-5];
  manifest.routes={bad};bool rejected=false;
  try{m.physical_forward_operand_residency(manifest);}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);
 }
 manifest.routes={good,good};bool rejected=false;
 try{m.physical_forward_operand_residency(manifest);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 for(int field=0;field<8;++field){m.context.route=good;
  if(field==0)m.context.route.selector=3;else if(field==1)m.context.route.flags=1;else ++m.context.route.arguments[field-2];
  uploads=0;rejected=false;try{m.emit(0);}catch(const std::runtime_error&){rejected=true;}assert(rejected&&uploads==0);
 }
}
'''
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "route.cpp", tmp_path / "route"
    cpp.write_text(program)
    build = subprocess.run([compiler,"-std=c++17",str(cpp),"-o",str(binary)],text=True,capture_output=True)
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)],text=True,capture_output=True)
    assert result.returncode == 0, result.stderr
