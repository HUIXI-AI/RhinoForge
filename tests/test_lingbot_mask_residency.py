"""LingBot mask residency: actual declarations, allocator, and shared typed emitter."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp"


def _program():
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    allocator_header = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    base = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    source = SOURCE.read_text()
    residency = (ROOT / "src/core/rpu_spm_residency.h").read_text()
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("native rejected"); } while(0)
#define TORCH_INTERNAL_ASSERT TORCH_CHECK
#define SDPA_MAX_VLM_SIZE 400
#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
#define DWIDTH 2
#define NUM_CORES 8
constexpr int64_t QWEN3_FUSED_LM_HEAD_MAX_VOCAB=200000;
template<class T> T CeilDiv(T a,T b) {return (a+b-1)/b;}
template<class T,class U> T Align(T a,U b) {return CeilDiv(a,static_cast<T>(b))*b;}
'''
    for marker in ("struct SdpaTiling", "enum class SdpaKernelType", "struct SdpaConfig"):
        program += _definition(helpers, marker) + ";\n"
    for marker in (
        "inline int64_t sdpa_tile_m_v16_initial(",
        "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline std::optional<int64_t> choose_tile_k_aggressive(",
        "inline std::optional<int64_t> choose_tile_k_conservative(",
        "inline std::optional<int64_t> choose_tile_k(",
        "inline bool sdpa_try_compute_tiling(",
        "inline SdpaTiling sdpa_compute_tiling(",
        "inline int64_t sdpa_compute_tmp_v16_size(",
    ):
        program += _definition(helpers, marker) + "\n"
    program += '''class SpmAllocator {public:
static constexpr size_t ALIGN=256;
static size_t align_up(size_t n) {return (n+ALIGN-1)&~(ALIGN-1);}
'''
    for marker in ("struct AllocRequest", "struct AliasedPlan"):
        program += _definition(allocator_header, marker) + ";\n"
    program += "static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);};\n"
    program += _definition(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    program += r'''
namespace v3 {
enum class StorageClass {Temp,TempPerLayer,Persistent,PersistentPerLayer};
enum class BufferScope {LayerWide,KvInsert,Compute,OutsideLayerLoop};
struct BufferDecl {
 const char* name=nullptr; int64_t size=0; int phase_start=0,phase_end=0;
 StorageClass storage=StorageClass::Temp;int per_layer=0;const char* alias_of=nullptr;
 BufferScope scope=BufferScope::LayerWide;
};
namespace detail {int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}
}
'''
    program += "namespace v3 {\n" + _definition(
        (ROOT / "src/core/fused_model_base.cpp").read_text(),
        "int64_t detail::estimate_temporary_total(") + "\n}\n"
    # Compile the actual shared helper; avoid importing Torch/device headers.
    program += re.sub(r'^\s*#(?:include|pragma)[^\n]*\n', '', residency, flags=re.M)
    program += r'''
using namespace v3;
enum class AttentionExecutionPolicy {DDR_KV,SPM_KV_BY_MHA};
'''
    program += _definition(base, "enum class CausalDecoderPlanningMode") + ";\n"
    program += r'''
struct LayoutContext {
 int64_t chunk_size=51,max_kv_seq_len=276,num_layers=36,kv_insert_chunk_size=0,batch_size=1;
 bool use_attn_mask=true,is_causal=false;
 AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::DDR_KV;
 int64_t effective_kv_cs()const{return kv_insert_chunk_size?kv_insert_chunk_size:chunk_size;}
};
class CausalDecoderModel {public:
 bool v1=false;
 bool qwen3vl_4b_packed_decode_qkv(int64_t,bool,bool,int64_t)const{return false;}
 bool lingbot_folded_rmsnorm_=false,lingbot_prefill_kv_only_=false,lingbot_prefill_mode_bound_=false;
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 bool wall_prefill_staged_mlp_=false,decode_gate_up_fusion_=false;
 bool prefill_packed_gate_up_strided_=false,qwen3vl_multiview_text_dry_prepared_=false;
 bool qwen3vl_multiview_text_composite_prepared_=false,has_qk_norm_=false,has_qkv_bias_=true;
 int qwen3vl_pooler_z1_retained_deepstack_count_=0;
 std::vector<int> deepstack_lang_layers_;
 SdpaKernelType sdpa_kernel_=SdpaKernelType::FLASH_ATTN_SPM;
 int64_t hidden_size()const{return 768;} int64_t intermediate_size()const{return v1?2816:768;}
 int64_t num_q_heads()const{return v1?16:32;} int64_t num_kv_heads()const{return v1?2:8;}
 int64_t head_dim()const{return 128;} int64_t num_layers()const{return 36;}
 int attn_tp()const{return v1?2:8;} int mlp_tp()const{return 8;} int lm_head_tp()const{return 8;}
 std::vector<BufferDecl> causal_baseline_buffer_declarations(const LayoutContext& ctx);
 bool explicit_mask_residency_requested_=false;
 virtual std::vector<BufferDecl> explicit_mask_baseline_buffer_declarations(const LayoutContext& ctx){return causal_baseline_buffer_declarations(ctx);}
 std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx);
'''
    program += _definition(base, "SdpaConfig make_sdpa_config(") + "\n"
    program += _definition(base, "bool lingbot_prefill_deepstack_reuses_down(") + "\n};\n"
    program += _definition(base, "std::vector<BufferDecl> declare_buffers(").replace(
        "std::vector<BufferDecl> declare_buffers(",
        "std::vector<BufferDecl> CausalDecoderModel::declare_buffers(", 1).replace(" override", "") + "\n"
    # All base Temp declarations precede persistent norm preloads. Assert that
    # boundary instead of maintaining a second handwritten model layout.
    marker = "std::vector<BufferDecl> causal_baseline_buffer_declarations("
    declaration = _definition(base, marker)
    boundary = declaration.index("// Norm-weight preloads")
    prefix, persistent = declaration[:boundary], declaration[boundary:]
    assert "StorageClass::Temp" not in persistent
    prefix = prefix[:prefix.rfind("// ─")]
    prefix = prefix.replace(marker, "std::vector<BufferDecl> CausalDecoderModel::causal_baseline_buffer_declarations(", 1)
    program += prefix + "return decls;\n}\n"
    program += r'''
struct FixtureTensor {int64_t elements=0;int64_t numel()const{return elements;}};
class LingbotV2MoeExpertModel:public CausalDecoderModel {public:
 bool fused_router_=false,routed_gemm_=false,sparse_experts_=false,sparse_inverse_combine_=false;
 bool sparse_route_compact_=false,denoise_tp8_encoder_=false;
 FixtureTensor fused_expert_ids_,sparse_expert_ids_,sparse_token_ids_,denoise_mo_bias_rows_;
 FixtureTensor expert_scatter_ids_{2048};
 int64_t num_experts_=32,top_k_=4,routed_inter_=512,tp_rows_=64;
 int64_t state_dim_pad_=64,action_dim_pad_=64;
 bool grouped_experts_=true,denoise_unroll_=true,debug_capture_innorm_=false;
 std::vector<BufferDecl> moe_baseline_buffer_declarations(const LayoutContext& ctx);
 std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx);
};
'''
    moe = _definition(source, "LingbotV2MoeExpertModel::moe_baseline_buffer_declarations(")
    moe = moe.replace("CausalDecoderModel::declare_buffers(ctx)",
                      "CausalDecoderModel::causal_baseline_buffer_declarations(ctx)")
    # Persistent callbacks emit DMA only; they do not affect temporary layout.
    while match := re.search(r"\w+\.preload_callback\s*=", moe):
        callback = _definition(moe[match.start():], match.group())
        end = match.start() + len(callback)
        assert moe[end] == ";"
        moe = moe[:match.start()] + moe[end + 1:]
    program += "std::vector<BufferDecl>\n" + moe + "\n"

    program += r'''
std::vector<SpmAllocator::AllocRequest> requests(const std::vector<BufferDecl>& decls) {
 std::vector<SpmAllocator::AllocRequest> r;
 for(auto& d:decls)if(d.storage==StorageClass::Temp && !d.alias_of)
  r.push_back({d.size,d.phase_start,d.phase_end,static_cast<int>(d.scope)});
 return r;
}
int main(){
 LingbotV2MoeExpertModel m;
 for(bool grouped:{false,true})for(bool unroll:{false,true})for(int64_t prefix:{192,225,512,973}){
  m.grouped_experts_=grouped;m.denoise_unroll_=unroll;
  LayoutContext l;l.max_kv_seq_len=prefix+51;
  auto old=m.moe_baseline_buffer_declarations(l),decls=old;
  auto p=detail::plan_forward_spm_residency(decls,{"sdpa_mask"});
  assert(old.size()==decls.size());
  for(size_t i=0;i<old.size();++i){
   assert(decls[i].size==old[i].size && decls[i].storage==old[i].storage);
  }
  auto req=requests(decls);auto plan=SpmAllocator::plan_temporary_aliased(req);
  auto oldreq=requests(old);auto oldplan=SpmAllocator::plan_temporary_aliased(oldreq);
  assert(plan.peak_bytes<=oldplan.peak_bytes);
  size_t mask=0,idx=0;
  for(auto& d:decls)if(d.storage==StorageClass::Temp && !d.alias_of){
   if(std::string(d.name)=="sdpa_mask")mask=idx;++idx;
  }
  if(p.retained){
   size_t begin=plan.offsets[mask],end=begin+req[mask].bytes;
   std::vector<unsigned char> spm(plan.peak_bytes,0);
   std::fill(spm.begin()+begin,spm.begin()+end,0x5a);
   for(int body=0;body<10;++body)for(size_t i=0;i<req.size();++i)if(i!=mask){
    size_t lo=plan.offsets[i],hi=lo+req[i].bytes;
    assert(hi<=begin||end<=lo);
    std::fill(spm.begin()+lo,spm.begin()+hi,static_cast<unsigned char>(body+1));
   }
   assert(std::all_of(spm.begin()+begin,spm.begin()+end,[](auto x){return x==0x5a;}));
  }
  std::cout<<grouped<<","<<unroll<<","<<prefix<<","<<p.retained<<","<<p.original_peak_bytes<<","<<p.retained_peak_bytes<<"\n";
 }
}
'''
    return program


def test_production_temp_declarations_admit_without_growth_or_alias(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "lingbot2_mask_lifetime.cpp"
    binary = tmp_path / "lingbot2_mask_lifetime"
    source.write_text(_program())
    built = subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    rows = result.stdout.strip().splitlines()
    assert len(rows) == 16
    print("grouped,unroll,prefix,retained,original_peak,candidate_peak\n" + result.stdout)


def test_actual_schedule_emitter_preserves_fallback_replay_and_rejects_drift(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    emitter = _definition(source, "void emit_explicit_mask_spm(")
    enum = _definition(source, "enum class CausalMaskSpmSchedule") + ";\n"
    program = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do {if(!(c))throw std::runtime_error("native rejected");}while(0)
enum class FmbRouteFamily {GRAPH_SCHEDULE};
constexpr int64_t CAUSAL_DECODER_MASK_SPM_SCHEDULE_SITE=301270362681343383LL;
struct ChunkInfo {int idx=0;};
struct Route {int64_t selector=1,flags=0;std::vector<int64_t> arguments;};
struct Context {
 bool complete=true;int64_t body_iter=0;Route route;int consumed=0;
 bool has_complete_physical_manifest()const{return complete;}
 const Route& find_physical_route(FmbRouteFamily,int64_t,int64_t invocation)const{
  TORCH_CHECK(invocation==0);return route;
 }
 void consume_physical_route(FmbRouteFamily,int64_t,int64_t selector,int64_t flags,
                             const std::vector<int64_t>& arguments,int64_t invocation){
  TORCH_CHECK(invocation==0 && selector==route.selector && flags==route.flags && arguments==route.arguments);
  ++consumed;
 }
};
int dmas=0,source_value=1,spm_value=0;
void sdpa_dma_mask_to_spm(int,uint32_t,int64_t,int64_t,int){++dmas;spm_value=source_value;}
'''
    program += enum
    program += r'''
struct Model {
 Context state;bool denoise_unroll_=true;int64_t num_steps_=3;int prepared_attn_mask_=0;
 bool explicit_mask_residency_requested_=true,keep_explicit_mask_in_spm_=false;
 int64_t explicit_mask_body_iterations()const{return denoise_unroll_?num_steps_:1;}
 Context& ctx(){return state;}
 int num_cores()const{return 8;}int num_layers()const{return 2;}

'''
    program += emitter
    program += r'''
 void layer(int layer_idx){emit_explicit_mask_spm(layer_idx,ChunkInfo{},51,276,2,1024);}
 void forward(){for(int body=0;body<(denoise_unroll_?num_steps_:1);++body){
  state.body_iter=body;for(int layer_idx=0;layer_idx<num_layers();++layer_idx)layer(layer_idx);
 }}
};
int main(){
 for(bool unroll:{false,true})for(bool retained:{false,true}){
  Model m;m.denoise_unroll_=unroll;
  m.state.route={retained?2:1,0,{51,276,2,8,unroll?3:1,2,100000,retained?100000:129440}};
  int per_forward=retained?1:(unroll?6:2);
  dmas=0;source_value=13;m.forward();assert(dmas==per_forward && spm_value==13);
  source_value=29;m.forward();assert(dmas==2*per_forward && spm_value==29);
  assert(m.state.consumed==2);
 }
 for(int malformed=0;malformed<6;++malformed){
  Model m;m.state.route={2,0,{51,276,2,8,3,2,100000,100000}};
  if(malformed==0)m.state.route.selector=7;
  if(malformed==1)m.state.route.arguments[0]=52;
  if(malformed==2)m.state.route.arguments[4]=10;
  if(malformed==3)m.state.route.arguments[7]=100256;
  if(malformed==4)m.state.route.arguments.pop_back();
  if(malformed==5)m.state.route.flags=1;
  dmas=0;bool rejected=false;try{m.forward();}catch(const std::runtime_error&){rejected=true;}
  assert(rejected && dmas==0);
 }
 Model legacy;legacy.state.complete=false;dmas=0;legacy.forward();assert(dmas==6);
 Model ordinary;ordinary.explicit_mask_residency_requested_=false;
 dmas=0;ordinary.forward();assert(dmas==6 && ordinary.state.consumed==0);
 Model wall;wall.explicit_mask_residency_requested_=false;wall.keep_explicit_mask_in_spm_=true;
 dmas=0;wall.forward();assert(dmas==1 && wall.state.consumed==0);
}
'''
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "lingbot2_mask_schedule.cpp", tmp_path / "lingbot2_mask_schedule"
    cpp.write_text(program)
    built = subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_v1_actual_plain_and_unroll_declarations_zero_growth(tmp_path):
    # Reuse the exact base declarations, SDPA sizing and allocator above. V1's
    # actual subclass appends its own pre/post declarations to that baseline.
    program = _program().split("int main(){", 1)[0]
    program += r'''
class LingbotDenoiseModel:public CausalDecoderModel {public:
 int64_t k_comb_pad_=80,action_dim_pad_=80;
 std::vector<BufferDecl> explicit_mask_baseline_buffer_declarations(const LayoutContext& ctx);
};
'''
    program += _definition(
        (ROOT / "src/fused/rpu_lingbot_denoise_model.cpp").read_text(),
        "std::vector<BufferDecl> LingbotDenoiseModel::explicit_mask_baseline_buffer_declarations(")
    program += r'''
int main(){
 CausalDecoderModel plain;plain.v1=true;plain.explicit_mask_residency_requested_=true;
 LingbotDenoiseModel unroll;unroll.v1=true;unroll.explicit_mask_residency_requested_=true;
 for(bool fused:{false,true})for(int64_t prefix:{75,180,192,205,225,512,973}){
  LayoutContext l;l.max_kv_seq_len=prefix+51;
  auto old=fused?unroll.explicit_mask_baseline_buffer_declarations(l):plain.causal_baseline_buffer_declarations(l);
  auto candidate=old;auto p=detail::plan_forward_spm_residency(candidate,{"sdpa_mask"});
  auto actual=fused?unroll.declare_buffers(l):plain.declare_buffers(l);
  assert(actual.size()==candidate.size());
  for(size_t i=0;i<actual.size();++i){
   assert(actual[i].size==old[i].size&&actual[i].storage==old[i].storage);
   assert(actual[i].phase_start==candidate[i].phase_start&&actual[i].phase_end==candidate[i].phase_end);
  }
  auto req=requests(candidate);auto plan=SpmAllocator::plan_temporary_aliased(req);
  auto oldreq=requests(old);auto oldplan=SpmAllocator::plan_temporary_aliased(oldreq);
  assert(plan.peak_bytes<=oldplan.peak_bytes);
  size_t mask=0,idx=0;
  for(auto& d:candidate)if(d.storage==StorageClass::Temp&&!d.alias_of){
   if(std::string(d.name)=="sdpa_mask")mask=idx;++idx;
  }
  if(p.retained){
   size_t begin=plan.offsets[mask],end=begin+req[mask].bytes;
   std::vector<unsigned char> spm(plan.peak_bytes,0);
   std::fill(spm.begin()+begin,spm.begin()+end,0x5a);
   for(int body=0;body<10;++body)for(size_t i=0;i<req.size();++i)if(i!=mask){
    size_t lo=plan.offsets[i],hi=lo+req[i].bytes;
    assert(hi<=begin||end<=lo);
    std::fill(spm.begin()+lo,spm.begin()+hi,static_cast<unsigned char>(body+1));
   }
   assert(std::all_of(spm.begin()+begin,spm.begin()+end,[](auto x){return x==0x5a;}));
  }
  std::cout<<fused<<","<<prefix<<","<<p.retained<<","<<p.original_peak_bytes<<","<<p.retained_peak_bytes<<"\n";
 }
}
'''
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source, binary = tmp_path / "lingbot1_mask_lifetime.cpp", tmp_path / "lingbot1_mask_lifetime"
    source.write_text(program)
    built = subprocess.run([compiler, "-std=c++17", "-O0", str(source), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    assert len(result.stdout.strip().splitlines()) == 14
    print("unroll,prefix,retained,original_peak,candidate_peak\n" + result.stdout)


def test_actual_cold_setter_rejects_reconfigure_and_implicit_derived_opt_in(tmp_path):
    setter = _definition((ROOT / "src/fused/rpu_qwen3_model.h").read_text(),
                         "void set_explicit_mask_spm_residency(")
    program = r'''#include <cassert>
#include <stdexcept>
#include <typeinfo>
#define TORCH_CHECK(c, ...) do {if(!(c))throw std::runtime_error("rejected");}while(0)
struct CausalDecoderModel {
 virtual ~CausalDecoderModel()=default;
 bool explicit_mask_residency_requested_=false,explicit_mask_residency_configured_=false;
 int layers=0,chunk=0,invalidations=0;
 int num_layers()const{return layers;}int get_last_resolved_chunk_size()const{return chunk;}
 void invalidate_model_state(){++invalidations;}
''' + setter + r'''
};
struct Derived:CausalDecoderModel {};
int main(){
 for(bool enable:{false,true}){
  CausalDecoderModel m;m.set_explicit_mask_spm_residency(enable);
  assert(m.explicit_mask_residency_requested_==enable && m.invalidations==1);
  bool caught=false;try{m.set_explicit_mask_spm_residency(!enable);}catch(...){caught=true;}
  assert(caught&&m.explicit_mask_residency_requested_==enable&&m.invalidations==1);
 }
 for(int late=0;late<2;++late){
  CausalDecoderModel m;if(late)m.chunk=64;else m.layers=2;
  bool caught=false;try{m.set_explicit_mask_spm_residency(true);}catch(...){caught=true;}
  assert(caught&&!m.explicit_mask_residency_requested_&&m.invalidations==0);
 }
 Derived d;bool caught=false;try{d.set_explicit_mask_spm_residency(true);}catch(...){caught=true;}
 assert(caught&&!d.explicit_mask_residency_requested_&&d.invalidations==0);
}
'''
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source, binary = tmp_path / "cold.cpp", tmp_path / "cold"
    source.write_text(program)
    built = subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
