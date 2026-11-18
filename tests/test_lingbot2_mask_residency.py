"""LingBot2 retained-mask admission uses real declarations and allocator budgets."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_lingbot_mask_residency import _program
from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp"

def strip_callbacks(text):
    while match := re.search(r"\w+\.preload_callback\s*=", text):
        block = _definition(text[match.start():], match.group())
        end = match.start() + len(block)
        assert text[end] == ";"
        text = text[:match.start()] + text[end + 1:]
    return text


def _full_program():
    text = _program()
    text = text.replace(" bool qwen3vl_4b_packed_decode_qkv(int64_t,bool,bool,int64_t)const{return false;}\n", "", 1)
    # Existing probe carries the complete actual temporary declarations. Add
    # the actual persistent portion instead of estimating it independently.
    base = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    marker = "std::vector<BufferDecl> causal_baseline_buffer_declarations("
    full = strip_callbacks(_definition(base, marker))
    full = full.replace(marker, "std::vector<BufferDecl> CausalDecoderModel::causal_baseline_buffer_declarations(", 1)
    old = _definition(text, "std::vector<BufferDecl> CausalDecoderModel::causal_baseline_buffer_declarations(")
    text = text.replace(old, full, 1)
    text = text.replace("using namespace v3;", "namespace at {struct Tensor {int64_t numel() const{return 0;}};}\nusing namespace v3;", 1)
    # Keep the real packed-QKV lifetime predicate; LingBot2 does not own the
    # Qwen3-VL profile or packed weights, so its existing layout stays ordinary.
    routing = "\n".join(_definition(base, marker) for marker in (
        "bool qwen3vl_decode_gemv(", "bool qwen3vl_4b_packed_decode_qkv("))
    # The planning-mode enum, `planning_mode_` and the real DeepStack predicate
    # already come from `_program()`; do not declare them a second time.
    text = text.replace("class CausalDecoderModel {public:",
        "class CausalDecoderModel {public:\n" + routing + r'''
 bool qwen3vl_4b_w8_profile_=false,qwen3vl_4b_packed_qkv_weights_=false;
 bool qwen3vl_fp16_decode_profile_=false;
 bool linear_acc32_=false,decoder_mixed_precision_=false;
 struct LayerWeights {at::Tensor q_bias_w,k_bias_w,v_bias_w;};
 bool adarms_mutable_=true,adarms_fused_bcast_enabled_=false,adarms_=true,nvfp4_=false;
 bool norm_slab_active()const{return false;} int64_t norm_slab_size_bytes()const{return 0;}
 at::Tensor q_nvfp4_tensor_scales_,k_nvfp4_tensor_scales_,v_nvfp4_tensor_scales_,o_nvfp4_tensor_scales_;
 at::Tensor gate_nvfp4_tensor_scales_,up_nvfp4_tensor_scales_,down_nvfp4_tensor_scales_;
''', 1)
    text = text.replace("BufferScope scope=BufferScope::LayerWide;", "BufferScope scope=BufferScope::LayerWide; bool reverse_layer_alloc=false;", 1)
    text = text.replace("static constexpr size_t ALIGN=256;", r'''
static constexpr size_t ALIGN=256;
static constexpr size_t SPM_USABLE=8387584;
static constexpr size_t SPM_PLANNING_BUDGET=SPM_USABLE*99/100;
size_t held=0,temporary=0;
size_t free_space()const{return SPM_USABLE-held-temporary;}
size_t temporary_used()const{return temporary;}
''', 1)
    insert = text.index("namespace v3 {")
    text = text[:insert] + "SpmAllocator test_allocator;\n#define SPM_ALLOC test_allocator\n" + text[insert:]
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    budget = _definition(fmb, "struct FixedSpmOverhead") + ";\n"
    for token in ("static FixedSpmOverhead estimate_fixed_overhead(",
                  "static int64_t available_temporary_spm_after_reset(",
                  "static bool final_chunk_layout_fits("):
        budget += _definition(fmb, token) + "\n"
    return text[:text.index("int main(){")] + budget


def _selection_program():
    source = SOURCE.read_text()
    text = _full_program()
    text = text.replace("struct LayoutContext {", """
enum class FmbForwardOperandResidency {UNSPECIFIED=0,PER_LAYER=1,FORWARD=2};
enum class FmbRouteFamily {GRAPH_SCHEDULE};
struct FmbRouteManifestEntry {
 int64_t site_id;FmbRouteFamily family;int64_t selector,flags;
 std::vector<int64_t> arguments;int64_t invocation;
};
struct FmbPhysicalExecutionManifest {
 int64_t physical_length=51,kv_logical_length=276;FmbRouteManifestEntry route;
};
struct FmbPhysicalManifestConsumer {
 FmbPhysicalExecutionManifest data;
 explicit FmbPhysicalManifestConsumer(const FmbPhysicalExecutionManifest& m):data(m){}
 const FmbRouteManifestEntry& find_route(FmbRouteFamily,int64_t site)const{
  TORCH_CHECK(site==data.route.site_id);return data.route;
 }
};
struct ExactDescriptor {
 int kind=1;uint64_t profile_hash=123,layout_hash=1;size_t temporary_bytes=1;
 FmbForwardOperandResidency mask_residency=FmbForwardOperandResidency::FORWARD;
 uint64_t model_state_generation=1;
};
struct ChunkInfo {int idx=0;};
struct Context {
 bool complete=true;int64_t body_iter=0;FmbRouteManifestEntry route;int consumed=0;
 bool has_complete_physical_manifest()const{return complete;}
 const FmbRouteManifestEntry& find_physical_route(FmbRouteFamily,int64_t site,int64_t i)const {
  TORCH_CHECK(site==route.site_id&&i==route.invocation);return route;
 }
 void consume_physical_route(FmbRouteFamily,int64_t site,int64_t selector,int64_t flags,
                            std::vector<int64_t> arguments,int64_t invocation) {
  TORCH_CHECK(site==route.site_id&&selector==route.selector&&flags==route.flags&&
              arguments==route.arguments&&invocation==route.invocation);++consumed;
 }
};
static bool final_chunk_layout_fits(const std::vector<BufferDecl>&,bool,int64_t);
int dmas=0,source_value=1,spm_value=0;
void sdpa_dma_mask_to_spm(int,uint32_t,int64_t,int64_t,int){++dmas;spm_value=source_value;}
constexpr int64_t L2_MASK_SPM_SCHEDULE=2878184572497809152LL;
struct LayoutContext {
 int64_t planning_chunk_capacity=0;
 FmbForwardOperandResidency forward_operand_residency=FmbForwardOperandResidency::UNSPECIFIED;
""", 1)
    text = text.replace("class LingbotV2MoeExpertModel:public CausalDecoderModel {public:", """
class LingbotV2MoeExpertModel:public CausalDecoderModel {public:
 int64_t chunk_size_=51,num_steps_=10;uint64_t generation=1;
 bool persistent_allocated=false;mutable int budget_queries=0;
 int num_cores()const{return 8;}
 uint64_t installed_model_state_generation()const{return generation;}
 int validate_exact_suffix_profile(int64_t prefix)const{TORCH_CHECK(prefix==225);return 1;}
 uint64_t exact_suffix_profile_hash(int kind)const{return kind*123;}
 std::optional<ExactDescriptor> exact_suffix_mask_authority_;
 bool declared_spm_layout_fits(const std::vector<BufferDecl>& d)const {
  ++budget_queries;return final_chunk_layout_fits(d,persistent_allocated,0);
 }
 FmbRouteManifestEntry mask_residency_route_for_layout(const LayoutContext&,bool=true)const;
 FmbForwardOperandResidency physical_forward_operand_residency(const FmbPhysicalExecutionManifest&)const;
 void emit_action_mask_spm(int,const ChunkInfo&,int64_t,int64_t,int,uint32_t);
 int prepared_attn_mask_=0;Context state;Context& ctx(){return state;}
 void forward(){for(int b=0;b<(denoise_unroll_?num_steps_:1);++b){state.body_iter=b;
  for(int l=0;l<num_layers();++l)emit_action_mask_spm(l,ChunkInfo{},51,276,8,1024);}}
""", 1)
    for token in ("FmbForwardOperandResidency mask_residency_mode(",
                  "FmbRouteManifestEntry LingbotV2MoeExpertModel::mask_residency_route_for_layout(",
                  "FmbForwardOperandResidency\nLingbotV2MoeExpertModel::physical_forward_operand_residency(",
                  "std::vector<BufferDecl> LingbotV2MoeExpertModel::declare_buffers(",
                  "void LingbotV2MoeExpertModel::emit_action_mask_spm("):
        text += _definition(source, token) + "\n"
    return text


def _run(tmp_path, program):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "native.cpp", tmp_path / "native"
    cpp.write_text(program)
    build = subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    print(run.stdout)


def test_real_budget_capacity_bound_layout_and_typed_replay(tmp_path):
    _run(tmp_path, _selection_program() + r'''
template<class Fn>void reject(Fn f){bool threw=false;try{f();}catch(const std::runtime_error&){threw=true;}assert(threw);}
int main(){
 LingbotV2MoeExpertModel m;LayoutContext l;l.planning_chunk_capacity=64;
 for(bool grouped:{false,true})for(bool unroll:{false,true}){
  m.grouped_experts_=grouped;m.denoise_unroll_=unroll;
  test_allocator.held=774144;test_allocator.temporary=500000;
  auto route=m.mask_residency_route_for_layout(l);assert(route.selector==2);
  auto old=m.declare_buffers(l);assert(estimate_fixed_overhead(old).persistent==318976);
  auto actual=detail::make_forward_spm_residency_candidate(old,{"sdpa_mask"});
  LayoutContext c=l;c.chunk_size=64;
  auto capacity=detail::make_forward_spm_residency_candidate(m.moe_baseline_buffer_declarations(c),{"sdpa_mask"});
  l.forward_operand_residency=FmbForwardOperandResidency::FORWARD;
  auto d=m.declare_buffers(l);auto fixed=estimate_fixed_overhead(d);
  assert(fixed.persistent==318976&&fixed.temp_per_layer==0);
  assert(detail::estimate_temporary_total(d)==actual.retained_peak_bytes);
  assert(actual.retained_peak_bytes-actual.original_peak_bytes==29440);
  auto req=requests(d);auto plan=SpmAllocator::plan_temporary_aliased(req);
  size_t mi=0,ri=0;for(auto&x:d)if(x.storage==StorageClass::Temp&&!x.alias_of){
   if(std::string(x.name)=="sdpa_mask")mi=ri;++ri;}
  size_t lo=plan.offsets[mi],hi=lo+req[mi].bytes;
  for(size_t i=0;i<req.size();++i)if(i!=mi)assert(plan.offsets[i]+req[i].bytes<=lo||hi<=plan.offsets[i]);
  std::cout<<grouped<<","<<unroll<<","<<actual.original_peak_bytes<<","<<actual.retained_peak_bytes
           <<","<<capacity.original_peak_bytes<<","<<capacity.retained_peak_bytes<<"\n";
  // The chosen layout is a pure mode input, even when later occupancy changes.
  m.persistent_allocated=true;test_allocator.held=SpmAllocator::SPM_USABLE-65536-actual.original_peak_bytes;
  assert(final_chunk_layout_fits(old,true)&&!final_chunk_layout_fits(d,true));
  auto again=m.declare_buffers(l);assert(detail::estimate_temporary_total(again)==actual.retained_peak_bytes);
  int queries=m.budget_queries;
  assert(m.mask_residency_route_for_layout(l).selector==2&&m.budget_queries==queries);
  l.forward_operand_residency=FmbForwardOperandResidency::UNSPECIFIED;
  assert(m.mask_residency_route_for_layout(l).selector==1);
  // Actual M51 fits, but the configurable C64 retained layout does not.
  test_allocator.held=SpmAllocator::SPM_USABLE-65536-capacity.original_peak_bytes;
  assert(final_chunk_layout_fits(d,true));
  assert(m.mask_residency_route_for_layout(l).selector==1);
  l.forward_operand_residency=FmbForwardOperandResidency::UNSPECIFIED;m.persistent_allocated=false;
 }
 m.grouped_experts_=true;m.denoise_unroll_=true;test_allocator.held=774144;
 l.max_kv_seq_len=65536;auto extreme=m.mask_residency_route_for_layout(l);
 assert(extreme.selector==1);l.max_kv_seq_len=276;
 // Z2 pins the selected mode despite new occupancy; FMB must fail capacity,
 // never silently reselect. All inherited invalidations advance generation.
 m.exact_suffix_mask_authority_=ExactDescriptor{};
 test_allocator.held=8000000;assert(m.mask_residency_route_for_layout(l).selector==2);
 ++m.generation;reject([&]{m.mask_residency_route_for_layout(l);});
 assert(m.mask_residency_route_for_layout(l,false).selector==1);--m.generation;
 l.max_kv_seq_len=277;reject([&]{m.mask_residency_route_for_layout(l);});l.max_kv_seq_len=276;
 m.exact_suffix_mask_authority_.reset();test_allocator.held=774144;
 m.state.route=m.mask_residency_route_for_layout(l);dmas=0;source_value=3;m.forward();
 FmbPhysicalExecutionManifest manifest;manifest.route=m.state.route;
 assert(m.physical_forward_operand_residency(manifest)==FmbForwardOperandResidency::FORWARD);
 m.exact_suffix_mask_authority_=ExactDescriptor{};++m.generation;
 reject([&]{m.physical_forward_operand_residency(manifest);});--m.generation;
 manifest.route.selector=1;reject([&]{m.physical_forward_operand_residency(manifest);});
 manifest.route=m.state.route;m.exact_suffix_mask_authority_.reset();
 ++manifest.kv_logical_length;reject([&]{m.physical_forward_operand_residency(manifest);});
 assert(dmas==1&&spm_value==3&&m.state.consumed==1);
 source_value=7;m.forward();assert(dmas==2&&spm_value==7&&m.state.consumed==2);
 auto saved=m.state.route;
 for(size_t i=0;i<6;++i){m.state.route=saved;++m.state.route.arguments[i];
  int before=dmas;reject([&]{m.forward();});assert(dmas==before);}
 m.state.route=saved;m.state.route.flags=1;reject([&]{m.forward();});
 m.state.route=saved;m.state.route.selector=3;reject([&]{m.forward();});
 m.state.route=saved;m.state.route.arguments.pop_back();reject([&]{m.forward();});
 m.state.route=saved;m.state.route.selector=1;dmas=0;m.forward();assert(dmas==360);
 m.state.complete=false;dmas=0;m.forward();assert(dmas==360);
 l.forward_operand_residency=static_cast<FmbForwardOperandResidency>(9);reject([&]{m.declare_buffers(l);});
}
''')


def test_z2_mode_is_bound_before_dry_and_live_layout_and_after_success():
    source = SOURCE.read_text()
    describe = _definition(source, "LingbotV2MoeExpertModel::describe_exact_suffix_spm_profile(")
    prime = _definition(source, "LingbotV2MoeExpertModel::prime_exact_suffix_spm_layout(")
    assert "honor_exact_authority=*/false" in describe
    assert describe.index("layout.forward_operand_residency =") < describe.index("prepare_spm_pipeline_component_for_cpu_contract(")
    assert prime.index("layout.forward_operand_residency = expected.mask_residency") < prime.index("declare_buffers(layout)")
    assert "expected.model_state_generation == installed_model_state_generation()" in prime
    assert "mask_residency_route_for_layout(" not in prime
    assert prime.index("super_after - super_before == kExpectedPersistentBytes") < prime.index("exact_suffix_mask_authority_ = expected")
    emitter = _definition(source, "void LingbotV2MoeExpertModel::emit_action_mask_spm(")
    assert "declared_spm_layout_fits" not in emitter
    assert "make_forward_spm_residency_candidate" not in emitter
