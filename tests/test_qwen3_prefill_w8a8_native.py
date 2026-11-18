"""Execute Qwen's real declaration/manifest code without device computation."""
from pathlib import Path
import re
import subprocess

from test_fmb_standalone_layout_rebuild import _definition
from test_qwen3vl_vision_raw_spm import _program

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/fused/rpu_qwen3_model.h"


def _run(tmp_path, code):
    source, binary = tmp_path / "qwen_a8.cpp", tmp_path / "qwen_a8"
    source.write_text(code)
    result = subprocess.run(["c++", "-std=c++17", "-O0", "-I", str(ROOT / "src"),
                             str(source), "-o", str(binary)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _prefix():
    return (_program().split("int main(){", 1)[0] +
            '\n#include "fused/rpu_qwen3_w8a8_contract.h"\n'
            '#include "fused/rpu_qwen3_c192_contract.h"\n'
            'namespace qwen3_w8a8=v3::qwen3_w8a8;namespace qwen3_c192=v3::qwen3_c192;namespace qwen3_c512=v3::qwen3_c512;\n')


def test_qwen3_w8a8_actual_declarations_fit_and_protect_call_operands(tmp_path):
    """Actual first-fit offsets, budget and every quant/PL live range at C128."""
    model = MODEL.read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    code = _prefix()
    code = code.replace("class FusedModelBase {\npublic:", "class FusedModelBase {\npublic:\n struct Impl{bool persistent_allocated_=false;};")
    constants = allocator[allocator.index("    static constexpr size_t SPM_TOTAL"):
                          allocator.index("    static constexpr int    NUM_CORES")]
    code = code.replace("class SpmAllocator {\npublic:", "class SpmAllocator {\npublic:\n" + constants +
                        "size_t free_space()const{return SPM_USABLE;}size_t temporary_used()const{return 0;}\n")
    code += "\nnamespace detail {" + _definition(header, "struct FmbPreparedSpmCost {") + ";}\n"
    for marker in ("struct FixedSpmOverhead", "static FixedSpmOverhead estimate_fixed_overhead(",
                   "int64_t detail::estimate_temporary_total(",
                   "int64_t detail::FmbPreparedSpmCost::available(", "bool detail::FmbPreparedSpmCost::fits("):
        definition = _definition(base, marker)
        if marker.startswith("int64_t detail::estimate_temporary_total"):
            code += "namespace detail {int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}\n"
        code += definition + (";" if marker.startswith("struct ") else "") + "\n"
    code += "SpmAllocator host_arena;\n#define SPM_ALLOC host_arena\n"
    for marker in ("static int64_t available_temporary_spm_after_reset(",
                   "static bool final_chunk_layout_fits(", "struct ChunkSizeProbe {",
                   "static std::vector<ChunkSizeProbe> enumerate_chunk_size_domain_impl("):
        code += _definition(base, marker) + (";" if marker.startswith("struct ") else "") + "\n"
    body = _definition(model, "std::vector<BufferDecl> causal_baseline_buffer_declarations(")
    # Preload closures are unrelated device transport; every declaration remains.
    while match := re.search(r"\b\w+\.preload_callback\s*=", body):
        block = _definition(body[match.start():], match[0])
        end = body.index(";", match.start() + len(block))
        body = body[:match.start()] + body[end + 1:]
    code += r'''
constexpr int64_t QWEN3_FUSED_LM_HEAD_MAX_VOCAB=200000;
struct TextOwner {
 using CausalDecoderModel=TextOwner;
 bool qwen3_c256_=false,qwen3_c512_=false;
 bool qwen3_c192_=false;
 bool qwen3_prefill_w8a8_=false;
 bool equal_two_prefill_=false,decoder_mixed_precision_=false;
 int64_t configured_chunk_size_cap_=0;
 struct Context{std::optional<Tensor> attention_mask;bool is_causal=true;int64_t batch_size=1;}context;
 const Context& ctx()const{return context;}
 struct Envelope{int64_t max_kv_len=512,chunk=192;};Envelope chunk_envelope()const{return {512,qwen3_c512_?512:qwen3_c256_?256:192};}
 bool chunk_within_envelope(int64_t c)const{return c<=chunk_envelope().chunk;}
 int64_t enforce_chunk_envelope(int64_t,int64_t,const char*)const{return chunk_envelope().chunk;}
 void require_qwen3_32b_w8_profile()const{}void require_qwen3_prefill_w8a8_profile()const{}
 FusedModelBase::Impl state;FusedModelBase::Impl* pimpl_=&state;
 int64_t minimum_eager_spm_bytes()const{return 0;}
 bool wall_prefill_staged_mlp_=false,decode_gate_up_fusion_=false,prefill_packed_gate_up_strided_=false;
 bool qwen3vl_multiview_text_dry_prepared_=false,qwen3vl_multiview_text_composite_prepared_=false;
 bool adarms_mutable_=false,adarms_fused_bcast_enabled_=false,adarms_=false;
 bool lingbot_folded_rmsnorm_=false,lingbot_prefill_kv_only_=false;
 bool has_qk_norm_=true,has_qkv_bias_=false,nvfp4_=false;
 int64_t qwen3vl_pooler_z1_retained_deepstack_count_=0;
 std::vector<int64_t> deepstack_lang_layers_{};
 Tensor q_nvfp4_tensor_scales_,k_nvfp4_tensor_scales_,v_nvfp4_tensor_scales_,o_nvfp4_tensor_scales_,gate_nvfp4_tensor_scales_,up_nvfp4_tensor_scales_,down_nvfp4_tensor_scales_;
 struct LayerWeights {Tensor q_bias_w,k_bias_w,v_bias_w;};
 static constexpr int NUM_CORES=8;
 int64_t hidden_size()const{return 5120;} int64_t intermediate_size()const{return 25600;}
 int64_t num_layers()const{return 64;} int64_t num_q_heads()const{return 64;}
 int64_t num_kv_heads()const{return 8;} int64_t head_dim()const{return 128;}
 int attn_tp()const{return 8;} int mlp_tp()const{return 8;} int lm_head_tp()const{return 8;}
 bool norm_slab_active()const{return false;} int64_t norm_slab_size_bytes()const{return 0;}
 bool lingbot_prefill_deepstack_reuses_down(const LayoutContext&)const{return false;}
 bool qwen3vl_4b_packed_decode_qkv(int64_t,bool,bool,int64_t)const{return false;}
 SdpaKernelType sdpa_kernel_=SdpaKernelType::FLASH_ATTN_SPM;
''' + _definition(model, "int64_t qwen3_inplace_capacity(") + _definition(model, "SdpaConfig make_sdpa_config(") + body
    for marker in ("bool subclass_chunk_size_valid(", "int64_t subclass_chunk_size_cap("):
        code += _definition(model, marker).replace(" override", "") + "\n"
    resolver = _definition(base, "FusedModelBase::resolve_prefill_stage_domain_for_shape_impl(")
    # Run the actual caller's base LayoutContext and capacity enumeration,
    # then its actual capacity -> physical-first-chunk layout conversion.
    probe = resolver[resolver.index("    LayoutContext base_layout;"):
                     resolver.index("    std::vector<FmbPrefillStageCandidate> result;")]
    final_layout = resolver[resolver.index("        LayoutContext candidate_layout = base_layout;"):
                            resolver.index("        FmbThreeStageChunkPlan stage_plan =")]
    persistent = _definition(base, "void FusedModelBase::prepare_persistent_spm(")
    seed = persistent[persistent.index("    LayoutContext layout;"):
                      persistent.index("    const int64_t persistent_hash =")]
    code += r'''
 std::vector<BufferDecl> declare_buffers(const LayoutContext& c){return causal_baseline_buffer_declarations(c);}
 std::vector<LayoutContext> actual_domain_probe(int64_t seq_len,int64_t position){
  const bool use_explicit_mask=false,is_causal=true;
  const std::optional<Tensor> attention_mask=std::nullopt;
  const struct{int64_t num_layers=64;}static_cfg;
  const int64_t requested_chunk_size=0,effective_chunk_override=chunk_envelope().chunk;
''' + probe + r'''
  std::vector<LayoutContext> layouts;
  for(const auto& probe:probes){if(!probe.valid||!probe.fits)continue;
   const int64_t compute_chunk_size=probe.chunk_size;
   const ChunkPlan compute_plan{std::min(compute_chunk_size,seq_len),1+(seq_len-1)/compute_chunk_size};
   const ChunkPlan qkv_plan=compute_plan;
   const struct{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;}dynamic_cfg;
''' + final_layout + r'''
   declare_buffers(candidate_layout);layouts.push_back(candidate_layout);
  }return layouts;
 }
 std::vector<BufferDecl> actual_persistent_seed(int64_t execution_len){
  const int64_t position=0,mask_kv_len=0;const bool is_causal=true;
  struct{int64_t num_layers_=64;}installed;auto*pimpl_=&installed;
''' + seed + r'''
  return decls;
 }
};
'''
    code += r'''
int main(){
 TextOwner owner;
 for(int64_t capacity:{192,256,512})for(bool a8:{false,true}){
  if(capacity==512&&!a8)continue;owner.qwen3_c512_=capacity==512;
  owner.qwen3_c192_=capacity==192;owner.qwen3_c256_=capacity==256;owner.qwen3_prefill_w8a8_=a8;
  // The initial decode resolve probes C16 for actual row1. Later decode
  // positions use the same capacity probe; none may select C192 scratch.
  for(int64_t position:{0,512,513}){const auto layouts=owner.actual_domain_probe(1,position);assert(layouts.size()==1&&layouts[0].chunk_size==1&&layouts[0].planning_chunk_capacity==16);}
  const auto prefill=owner.actual_domain_probe(512,0);assert(prefill.size()==1&&prefill[0].chunk_size==capacity);
  const auto seeded=owner.actual_persistent_seed(512);const auto fixed=estimate_fixed_overhead(seeded);assert(fixed.persistent==1353728);
  for(int c:{16,32,64,128,160,176,192,208,256})if(c!=capacity)assert(!owner.subclass_chunk_size_valid(c,512,0));
  LayoutContext final_decode;final_decode.chunk_size=1;final_decode.max_kv_seq_len=513;final_decode.num_layers=64;final_decode.is_causal=true;
  auto cold=owner.causal_baseline_buffer_declarations(final_decode);owner.qwen3_c192_=false;owner.qwen3_c256_=false;owner.qwen3_c512_=false;auto ordinary=owner.causal_baseline_buffer_declarations(final_decode);
  assert(cold.size()==ordinary.size());for(size_t i=0;i<cold.size();++i)assert(std::string(cold[i].name)==ordinary[i].name&&cold[i].size==ordinary[i].size&&cold[i].phase_start==ordinary[i].phase_start&&cold[i].phase_end==ordinary[i].phase_end);
 }

 {
  owner.qwen3_c192_=owner.qwen3_c256_=false;owner.qwen3_c512_=true;owner.qwen3_prefill_w8a8_=true;
  LayoutContext lc;lc.chunk_size=lc.kv_insert_chunk_size=512;lc.num_layers=64;lc.max_kv_seq_len=512;lc.is_causal=true;lc.batch_size=1;lc.attention_policy=AttentionExecutionPolicy::DDR_KV;
  const auto ds=owner.causal_baseline_buffer_declarations(lc);const auto fixed=estimate_fixed_overhead(ds);const auto temp=detail::estimate_temporary_total(ds);
  assert(fixed.persistent==1353728&&temp==6882304);
  const int64_t budget=SpmAllocator::SPM_PLANNING_BUDGET&~(int64_t(SpmAllocator::ALIGN)-1);
  assert(budget-fixed.persistent-temp==67584&&budget-fixed.persistent-temp>=65536);
  std::vector<SpmAllocator::AllocRequest> req;std::vector<const BufferDecl*> roots;
  for(const auto&d:ds)if(d.storage==StorageClass::Temp&&!d.alias_of){req.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});roots.push_back(&d);}
  const auto plan=SpmAllocator::plan_temporary_aliased(req);std::map<std::string,uint64_t>a,b;
  for(size_t i=0;i<roots.size();++i){a[roots[i]->name]=plan.offsets[i];b[roots[i]->name]=roots[i]->size;}
  for(const auto&d:ds)if(d.alias_of){assert(std::string(d.name)=="a8_mlp"&&std::string(d.alias_of)=="oproj");a[d.name]=a.at(d.alias_of);b[d.name]=d.size;}
  assert(a.at("oproj")==1024&&a.at("oproj_suffix")==2622464);
  assert(qwen3_c512::oproj_regions_valid(a.at("oproj"),a.at("oproj_suffix"),a.at("a8_mlp"),0,SpmAllocator::SPM_USABLE));
  assert(b.at("oproj")==2621440&&b.at("oproj_suffix")==2621440&&b.at("a8_mlp")==2621440);
  for(const auto&d:ds){if(std::string(d.name)=="oproj")assert(d.phase_start==3&&d.phase_end==9);if(std::string(d.name)=="oproj_suffix")assert(d.phase_start==6&&d.phase_end==7);}
  for(int part=0;part<4;++part){const uint64_t consumed_end=a.at("oproj")+(part+1)*128*5120*2;assert(a.at("a8_mlp")+(part+1)*128*5120<=consumed_end);}
  const auto sep=[&](const char*x,const char*y){assert(a[x]+b[x]<=a[y]||a[y]+b[y]<=a[x]);};
  for(const char* n:{"residual2","input_norm","mlp_norm","up","norm2_input_tile"})assert(!b.count(n));
  for(const auto pair:std::array<std::pair<const char*,const char*>,33>{{
   {"residual1","a8_qkv"},{"residual1","a8_qkv_sx"},{"a8_qkv","a8_qkv_sx"},
   {"a8_qkv","q"},{"a8_qkv","k"},{"a8_qkv","v"},{"q","q_norm_scratch"},{"k","k_norm_scratch"},
   {"q","output"},{"q","sdpa_tmp"},{"output","sdpa_tmp"},{"output","a8_o"},{"output","a8_o_sx"},
   {"a8_o","oproj"},{"a8_o_sx","oproj"},{"oproj","o_residual_tile"},
   {"a8_o","oproj_suffix"},{"a8_o_sx","oproj_suffix"},{"oproj_suffix","o_residual_tile"},
   {"o_residual_tile","a8_mlp_sx"},{"a8_mlp","gate"},{"a8_mlp_sx","gate"},
   {"gate","a8_down"},{"gate","a8_down_sx"},{"a8_down","down"},{"a8_down_sx","down"},{"down","down_residual_tile"},
   {"o_compact_residual","oproj"},{"o_compact_residual","oproj_suffix"},
   {"o_compact_residual","o_residual_tile"},{"o_compact_residual","a8_mlp_sx"},
   {"down_compact_residual","down"},{"down_compact_residual","down_residual_tile"}
  }})sep(pair.first,pair.second);
  assert(b.at("o_compact_residual")==163840&&b.at("down_compact_residual")==163840);
  assert(qwen3_c512::residual_regions_valid(a.at("oproj"),a.at("o_compact_residual"),a.at("o_residual_tile"),0,SpmAllocator::SPM_USABLE));
  assert(qwen3_c512::residual_regions_valid(a.at("down"),a.at("down_compact_residual"),a.at("down_residual_tile"),0,SpmAllocator::SPM_USABLE));
  assert(b["residual1"]==128*5120*2&&b["a8_qkv"]==512*5120&&b["a8_down"]==512*3200&&b["down"]==512*5120*2);
  assert(sdpa_call_valid(owner.make_sdpa_config(1),512,0));
  owner.qwen3_prefill_w8a8_=false;bool bad=false;try{owner.causal_baseline_buffer_declarations(lc);}catch(const std::exception&){bad=true;}assert(bad);
  owner.qwen3_c512_=false;
  std::cout<<"C512 production A8 temp="<<temp<<" headroom="<<budget-fixed.persistent-temp<<"\n";
 }
 for(int64_t c:{1,16,32,64,128,192,256}){
  owner.qwen3_c192_=c==192;owner.qwen3_c256_=c==256;
  LayoutContext lc;lc.chunk_size=lc.kv_insert_chunk_size=c;lc.num_layers=64;
  lc.max_kv_seq_len=512;lc.is_causal=true;lc.batch_size=1;lc.attention_policy=AttentionExecutionPolicy::DDR_KV;
  owner.qwen3_prefill_w8a8_=false;auto off=owner.causal_baseline_buffer_declarations(lc);
  owner.qwen3_prefill_w8a8_=true;auto on=owner.causal_baseline_buffer_declarations(lc);
  if(c==1){assert(on.size()==off.size());for(size_t i=0;i<on.size();++i){assert(on[i].size==off[i].size&&on[i].phase_start==off[i].phase_start&&on[i].phase_end==off[i].phase_end);}continue;}
  if(c==192){assert(!sdpa_is_valid_chunk_size(owner.make_sdpa_config(1),c,512,0));for(auto r:std::array<std::pair<int64_t,int64_t>,3>{{{192,0},{192,192},{128,384}}})assert(sdpa_call_valid(owner.make_sdpa_config(1),r.first,r.second));}
  else assert(sdpa_is_valid_chunk_size(owner.make_sdpa_config(1),c,512,0));
  const auto fixed=estimate_fixed_overhead(on);const auto temp=detail::estimate_temporary_total(on);
  const detail::FmbPreparedSpmCost cost{fixed.persistent,fixed.temp_per_layer,temp};
  assert(cost.fits(SpmAllocator::SPM_USABLE,false,SpmAllocator::SPM_PLANNING_BUDGET));
  assert(fixed.persistent==1353728);
  std::vector<SpmAllocator::AllocRequest> reqs;std::vector<const BufferDecl*> roots;
  for(const auto&d:on)if(d.storage==StorageClass::Temp&&!d.alias_of){reqs.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});roots.push_back(&d);}
  const auto plan=SpmAllocator::plan_temporary_aliased(reqs);
  std::map<std::string,uint32_t> addr;std::map<std::string,int64_t> bytes;
  for(size_t i=0;i<roots.size();++i){addr[roots[i]->name]=0x10000000+plan.offsets[i];bytes[roots[i]->name]=roots[i]->size;}
  const auto disjoint=[&](const char*a,int64_t an,const char*b,int64_t bn){assert(an<=bytes.at(a)&&bn<=bytes.at(b));assert(uint64_t(addr[a])+an<=addr[b]||uint64_t(addr[b])+bn<=addr[a]);};
  const char* src[]{c>=128?"residual1":"input_norm","output",c>=192?"residual1":c==128?"input_norm":"residual1","gate"};
  const char* q[]{"a8_qkv","a8_o","a8_mlp","a8_down"};
  const char* sx[]{"a8_qkv_sx","a8_o_sx","a8_mlp_sx","a8_down_sx"};
  for(int r=0;r<4;++r){const auto k=qwen3_w8a8::kQuantK[r];disjoint(src[r],c*k*2,q[r],c*k);disjoint(src[r],c*k*2,sx[r],c*2);disjoint(q[r],c*k,sx[r],c*2);}
  if(c==128||c==192){disjoint("up",c*3200*2,"a8_down",c*3200);disjoint("up",c*3200*2,"a8_down_sx",c*2);disjoint("gate",c*3200*2,"up",c*3200*2);}
  const char* y[]{"q","k","v","oproj","gate","up","down"};const int roles[]{0,0,0,1,2,2,3};
  for(int p=0;p<7;++p){if(c==256&&p==5){disjoint("a8_mlp",c*5120,"gate",c*3200*2);continue;}auto s=qwen3_w8a8::shape(c,static_cast<qwen3_w8a8::Projection>(p));auto span=rpu_pl_w8a8::spans(s);const auto role=roles[p];disjoint(q[role],span.x,y[p],span.y);disjoint(sx[role],span.sx,y[p],span.y);auto descriptor=rpu_pl_w8a8::make_descriptor(s,{addr[q[role]],addr[y[p]],0,addr[sx[role]]},0x10000000,SpmAllocator::SPM_USABLE);assert(descriptor.ordinal<29);}
  if(c>=192){
   assert(bytes.at("sdpa_tmp")== (c==256?131072:65536)&&bytes.at("k_norm_scratch")==c*128*2);
   if(c==256)assert(!bytes.count("up"));
   assert(!bytes.count("input_norm")&&!bytes.count("mlp_norm"));
   for(const auto&d:on)if(std::string(d.name)=="residual2")assert(std::string(d.alias_of)=="residual1");
   const auto f=estimate_fixed_overhead(off);const auto a16=detail::estimate_temporary_total(off);
   const detail::FmbPreparedSpmCost a16cost{f.persistent,f.temp_per_layer,a16};
   assert(a16cost.fits(SpmAllocator::SPM_USABLE,false,SpmAllocator::SPM_PLANNING_BUDGET));
   // Real A16 roots and planner offsets: norm inputs, raw residual and partials remain disjoint.
   std::vector<SpmAllocator::AllocRequest> ar;std::vector<const BufferDecl*> ad;
   for(const auto&d:off)if(d.storage==StorageClass::Temp&&!d.alias_of){ar.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});ad.push_back(&d);}
   const auto ap=SpmAllocator::plan_temporary_aliased(ar);std::map<std::string,uint64_t> aa,ab;
   for(size_t i=0;i<ad.size();++i){aa[ad[i]->name]=ap.offsets[i];ab[ad[i]->name]=ad[i]->size;}
   const auto sep=[&](const char*a,const char*b){assert(aa[a]+ab[a]<=aa[b]||aa[b]+ab[b]<=aa[a]);};
   for(const char*n:{"input_norm","mlp_norm","k_norm_scratch","q","k","v","output","oproj","gate","up","down","lm_head_out"})if(ab.count(n))sep("residual1",n);
   sep("input_norm","q");sep("input_norm","k");sep("input_norm","v");sep("mlp_norm","gate");if(c!=256)sep("mlp_norm","up");sep("gate","down");sep("k","k_norm_scratch");
   if(c==256){
    assert(a16==6881280&&temp==6488064);
    for(const auto&d:off)if(std::string(d.name)=="lm_head_out")assert(d.phase_start==9&&d.phase_end==9);
    const int64_t budget=SpmAllocator::SPM_PLANNING_BUDGET&~(int64_t(SpmAllocator::ALIGN)-1);
    const int64_t eager_row_bytes=(5120+151936/8)*2+3*SpmAllocator::ALIGN;
    assert(eager_row_bytes==48992);
    assert(budget-f.persistent-f.temp_per_layer-a16==68608);
    assert(budget-f.persistent-f.temp_per_layer-a16>=qwen3_c192::kC256EagerHeadReserve);
    assert(budget-fixed.persistent-fixed.temp_per_layer-temp>=qwen3_c192::kC256EagerHeadReserve);
    assert(budget-f.persistent-f.temp_per_layer-a16>eager_row_bytes);
    // Reproduce the real decoder-fit/head-OOM layout without changing any
    // operand: only the unnecessarily early head-output lifetime differs.
    auto stale=off;for(auto&d:stale)if(std::string(d.name)=="lm_head_out")d.phase_start=8;
    assert(detail::estimate_temporary_total(stale)==6931456);
    assert(budget-f.persistent-f.temp_per_layer-detail::estimate_temporary_total(stale)==18432);
    assert(budget-f.persistent-f.temp_per_layer-detail::estimate_temporary_total(stale)<=eager_row_bytes);
   }
   if(c==192)assert(a16==6389760&&temp==6022144);
   std::cout<<"C"<<c<<" production A16 temp="<<a16<<" A8 temp="<<temp<<" available="<<cost.available(SpmAllocator::SPM_USABLE,false,SpmAllocator::SPM_PLANNING_BUDGET)<<"\n";
  }
  if(c==128){assert(detail::estimate_temporary_total(off)==5570560);std::cout<<"C128 actual A8 temp="<<temp<<" available="<<cost.available(SpmAllocator::SPM_USABLE,false,SpmAllocator::SPM_PLANNING_BUDGET)<<"\n";}
 }
}
'''
    print(_run(tmp_path, code))


def test_qwen3_w8a8_actual_manifest_and_plan_reject_stale_routes(tmp_path):
    model = MODEL.read_text()
    code = _prefix()
    # The stage resolver also applies this generic validator after the owner
    # builder. Owner-only equality checks previously missed its nonnegative
    # argument contract (the actual board failure was clamp_min == -127).
    code = ('#include <sstream>\n' + code).replace(
        '#define TORCH_CHECK(c,...) do {if(!(c))throw std::runtime_error("native contract rejected");}while(0)',
        'template<class... T> std::string contract_error(T&&... values){std::ostringstream out;(out << ... << values);return out.str();}\n'
        '#define TORCH_CHECK(c,...) do {if(!(c))throw std::runtime_error(contract_error(__VA_ARGS__));}while(0)')
    planner = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    code += _definition(planner, "void validate_fmb_physical_manifest(")
    code += r'''
constexpr int64_t CAUSAL_DECODER_Q_SITE=403971716841024546LL;
constexpr int64_t CAUSAL_DECODER_K_SITE=4795854722057375216LL;
constexpr int64_t CAUSAL_DECODER_V_SITE=6513089886571723927LL;
constexpr int64_t CAUSAL_DECODER_O_SITE=4,CAUSAL_DECODER_REDUCE_SITE=5;
enum class CausalDecoderLinearRoute {AUTO_TILE=1};
int64_t causal_ring_route(int64_t,int64_t){return 2;}
struct Owner {
 bool qwen3_prefill_w8a8_=true,qwen3_c192_=false,qwen3_c256_=false,qwen3_c512_=false;
'''
    for marker in ("int64_t qwen3_inplace_capacity(", "bool qwen3_c192_plan_valid(", "bool qwen3_c512_manifest_valid(", "bool qwen3_c192_manifest_valid(", "std::array<int64_t, 7> qwen3_prefill_w8a8_linear_sites(",
                   "bool qwen3_prefill_w8a8_manifest_valid(", "bool qwen3_prefill_w8a8_plan_valid("):
        code += _definition(model, marker)
    code += r'''
};
FmbPhysicalExecutionManifest manifest(Owner& owner,int64_t physical,int64_t cs){
 FmbPhysicalExecutionManifest m;m.state=FmbPhysicalManifestState::COMPLETE;m.physical_length=m.logical_length=m.kv_logical_length=physical;
 m.kv_insert_physical_rows=physical;m.graph_lifecycle=FmbGraphLifecycle::RETAINED_CACHE;
 m.linear_accumulation=physical>1?FmbLinearAccumulationPolicy::MIXED_BY_SITE:FmbLinearAccumulationPolicy::ACC16;
 m.routes.push_back({qwen3_w8a8::kPolicySite,FmbRouteFamily::GRAPH_SCHEDULE,1,0,{qwen3_w8a8::kPolicyVersion,physical,512,owner.qwen3_c512_?512:owner.qwen3_c256_?256:owner.qwen3_c192_?192:128,8,physical>1?7:0,physical>1?4:0},0});
 if(owner.qwen3_inplace_capacity())m.routes.push_back({owner.qwen3_c512_?qwen3_c512::kPolicySite:qwen3_c192::kPolicySite,FmbRouteFamily::GRAPH_SCHEDULE,1,0,owner.qwen3_c512_?qwen3_c512::policy_arguments(physical):qwen3_c192::policy_arguments(physical,owner.qwen3_prefill_w8a8_,owner.qwen3_inplace_capacity()),0});
 if(physical==1)return m;
 if(owner.qwen3_c512_){for(auto role:qwen3_c512::kDmaRoles)m.routes.push_back({qwen3_c512::kDmaSite,FmbRouteFamily::MUTABLE_DMA,qwen3_c512::dma_selector(role),0,qwen3_c512::dma_arguments(role),static_cast<int64_t>(role)});m.routes.push_back({qwen3_c512::kFinalNormSite,FmbRouteFamily::NORMALIZATION,1,0,qwen3_c512::final_norm_arguments(),0});}
 for(int64_t off=0,idx=0;off<physical;off+=cs,++idx){auto rows=std::min(cs,physical-off);
  for(int r=0;r<4;++r)m.routes.push_back({qwen3_w8a8::kQuantSites[r],qwen3_w8a8::uses_rms_quant(rows,r)?FmbRouteFamily::NORMALIZATION:FmbRouteFamily::ACTIVATION,qwen3_w8a8::quant_selector(rows,r),0,qwen3_w8a8::quant_arguments(rows,r),idx});
  for(int p=0;p<7;++p)m.routes.push_back({owner.qwen3_prefill_w8a8_linear_sites()[p],FmbRouteFamily::LINEAR,qwen3_w8a8::linear_selector(rows,static_cast<qwen3_w8a8::Projection>(p)),0,qwen3_w8a8::linear_arguments(rows,static_cast<qwen3_w8a8::Projection>(p)),idx});
  if(owner.qwen3_inplace_capacity()){
   m.routes.push_back({qwen3_c192::kDownReduceSite,FmbRouteFamily::ALL_REDUCE,owner.qwen3_c512_?qwen3_c512::kRingSelector:qwen3_c192::kRingSelector,0,owner.qwen3_c512_?qwen3_c512::ring_arguments():qwen3_c192::ring_arguments(rows),idx});
   m.routes.push_back({CAUSAL_DECODER_REDUCE_SITE,FmbRouteFamily::ALL_REDUCE,owner.qwen3_c512_?qwen3_c512::kRingSelector:qwen3_c192::kRingSelector,0,owner.qwen3_c512_?qwen3_c512::ring_arguments():qwen3_c192::ring_arguments(rows),idx});
  }else m.routes.push_back({qwen3_w8a8::kReduceSite,FmbRouteFamily::ALL_REDUCE,2,0,{1,rows,5120,8},idx});
 }
 std::sort(m.routes.begin(),m.routes.end(),[](const auto&a,const auto&b){return std::make_tuple(int64_t(a.family),a.site_id,a.invocation)<std::make_tuple(int64_t(b.family),b.site_id,b.invocation);});
 return m;
}
int main(){Owner owner;
 for(int64_t cs:{16,32,64,128}){
  auto m=manifest(owner,512,cs);assert(owner.qwen3_prefill_w8a8_manifest_valid(m));
  validate_fmb_physical_manifest(m,512,0);
  for(size_t i=0;i<m.routes.size();++i)if(std::find(qwen3_w8a8::kQuantSites.begin(),qwen3_w8a8::kQuantSites.end(),m.routes[i].site_id)!=qwen3_w8a8::kQuantSites.end()){
   assert(m.routes[i].arguments[7]==0x57f0&&m.routes[i].arguments[8]==0xd7f0);
   auto stale=m;stale.routes[i].arguments[8]=-127;
   assert(!owner.qwen3_prefill_w8a8_manifest_valid(stale));
   bool rejected=false;try{validate_fmb_physical_manifest(stale,512,0);}catch(const std::runtime_error&e){rejected=std::string(e.what()).find("physical manifest route argument is negative")!=std::string::npos;}assert(rejected);
   stale=m;--stale.routes[i].arguments[0];assert(!owner.qwen3_prefill_w8a8_manifest_valid(stale));
  }
  for(int p=0;p<7;++p){
   const auto a=qwen3_w8a8::linear_arguments(cs,static_cast<qwen3_w8a8::Projection>(p));
   const int ordinal=cs==128?(p==4||p==5?3:p==6?1:0):0;
   assert(a[8]==ordinal);
   if(ordinal==3)assert(a[9]==192&&a[10]==64&&a[11]==384&&a[19]==288);
   if(ordinal==1)assert(a[9]==144&&a[10]==48&&a[11]==512&&a[19]==246);
  }
  if(cs==128){
   int fused=0;for(const auto&r:m.routes)if(r.family==FmbRouteFamily::NORMALIZATION){++fused;assert(r.selector==2&&r.arguments[10]==6&&r.arguments[11]==0x11&&r.arguments[12]==0xa66);}
   assert(fused==8);
   auto stale=m;for(auto&r:stale.routes)if(r.site_id==qwen3_w8a8::kPolicySite)r.arguments[0]=2;assert(!owner.qwen3_prefill_w8a8_manifest_valid(stale));
   for(size_t i=0;i<m.routes.size();++i)if(m.routes[i].family==FmbRouteFamily::NORMALIZATION){
    stale=m;stale.routes[i].family=FmbRouteFamily::ACTIVATION;stale.routes[i].selector=1;stale.routes[i].arguments={1,128,5120,8,16,8,16,127,-127,1};assert(!owner.qwen3_prefill_w8a8_manifest_valid(stale));
   }
   // A descriptor captured before measured tile selection must not replay
   // ordinal0 for Gate/Up/Down under the new physical plan identity.
   for(size_t i=0;i<m.routes.size();++i)if(m.routes[i].family==FmbRouteFamily::LINEAR&&m.routes[i].arguments[8]!=0){
    stale=m;auto&a=stale.routes[i].arguments;a[8]=0;a[9]=128;a[10]=64;a[11]=128;a[19]=112;
    assert(!owner.qwen3_prefill_w8a8_manifest_valid(stale));
   }
  }else for(const auto&r:m.routes)assert(r.family!=FmbRouteFamily::NORMALIZATION);
  for(size_t i=0;i<m.routes.size();++i){auto bad=m;bad.routes.erase(bad.routes.begin()+i);assert(!owner.qwen3_prefill_w8a8_manifest_valid(bad));bad=m;bad.routes.push_back(bad.routes[i]);assert(!owner.qwen3_prefill_w8a8_manifest_valid(bad));
   for(size_t a=0;a<m.routes[i].arguments.size();++a){bad=m;++bad.routes[i].arguments[a];assert(!owner.qwen3_prefill_w8a8_manifest_valid(bad));}
   bad=m;++bad.routes[i].selector;assert(!owner.qwen3_prefill_w8a8_manifest_valid(bad));bad=m;bad.routes[i].flags=1;assert(!owner.qwen3_prefill_w8a8_manifest_valid(bad));
  }
  owner.qwen3_prefill_w8a8_=false;assert(!owner.qwen3_prefill_w8a8_manifest_valid(m));owner.qwen3_prefill_w8a8_=true;
  LayoutContext l;l.is_causal=true;l.batch_size=1;l.attention_policy=AttentionExecutionPolicy::DDR_KV;
  FmbThreeStageChunkPlan p;p.chunk_mode=ChunkMode::SEQUENTIAL;for(int64_t off=0,idx=0;off<512;off+=cs,++idx)p.compute.chunks.push_back({idx,off,cs,off+cs});
  assert(owner.qwen3_prefill_w8a8_plan_valid(p,l,512,0));assert(!owner.qwen3_prefill_w8a8_plan_valid(p,l,512,1));assert(!owner.qwen3_prefill_w8a8_plan_valid(p,l,528,0));
  l.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;assert(!owner.qwen3_prefill_w8a8_plan_valid(p,l,512,0));l.attention_policy=AttentionExecutionPolicy::DDR_KV;
  p.compute.chunks.back().offset++;assert(!owner.qwen3_prefill_w8a8_plan_valid(p,l,512,0));
 }
 {
  owner.qwen3_c192_=true;
  auto m=manifest(owner,512,192);assert(owner.qwen3_c192_manifest_valid(m)&&owner.qwen3_prefill_w8a8_manifest_valid(m));
  validate_fmb_physical_manifest(m,512,0);
  // Both full chunks use the measured one-M-loop C192 tile. The final
  // 128 rows retain their independently measured C128 selection.
  for(size_t i=0;i<m.routes.size();++i)if(m.routes[i].family==FmbRouteFamily::LINEAR){
   const auto&a=m.routes[i].arguments;
   if(a[1]==192){
    assert(a[8]==3&&a[9]==192&&a[10]==64&&a[11]==384&&a[19]==288);
    auto stale=m;auto&old=stale.routes[i].arguments;
    old[8]=0;old[9]=128;old[10]=64;old[11]=128;old[19]=112;
    assert(!owner.qwen3_prefill_w8a8_manifest_valid(stale));
   }else{
    assert(a[1]==128);
    const bool gate_up=a[2]==25600;
    const bool down=a[3]==25600;
    assert(a[8]==(gate_up?3:down?1:0));
   }
  }
  LayoutContext l;l.chunk_size=l.kv_insert_chunk_size=192;l.is_causal=true;l.batch_size=1;l.attention_policy=AttentionExecutionPolicy::DDR_KV;
  FmbThreeStageChunkPlan p;p.chunk_mode=ChunkMode::SEQUENTIAL;p.compute.chunks={{0,0,192,192},{1,192,192,384},{2,384,128,512}};p.input.chunks={{0,0,512,512}};p.qkv.chunks=p.compute.chunks;
  assert(owner.qwen3_prefill_w8a8_plan_valid(p,l,512,0));p.qkv.chunks.back().len=192;assert(!owner.qwen3_prefill_w8a8_plan_valid(p,l,512,0));
  for(size_t i=0;i<m.routes.size();++i){auto bad=m;bad.routes.erase(bad.routes.begin()+i);assert(!owner.qwen3_c192_manifest_valid(bad)||!owner.qwen3_prefill_w8a8_manifest_valid(bad));bad=m;bad.routes.push_back(bad.routes[i]);assert(!owner.qwen3_c192_manifest_valid(bad)||!owner.qwen3_prefill_w8a8_manifest_valid(bad));
   for(size_t a=0;a<m.routes[i].arguments.size();++a){bad=m;++bad.routes[i].arguments[a];assert(!owner.qwen3_c192_manifest_valid(bad)||!owner.qwen3_prefill_w8a8_manifest_valid(bad));}}
  owner.qwen3_c192_=false;assert(!owner.qwen3_c192_manifest_valid(m)&&!owner.qwen3_prefill_w8a8_manifest_valid(m));
  owner.qwen3_c192_=true;owner.qwen3_prefill_w8a8_=false;
  m.routes.erase(std::remove_if(m.routes.begin(),m.routes.end(),[](const auto&r){return r.site_id==qwen3_w8a8::kPolicySite||r.selector==201&&r.family==FmbRouteFamily::LINEAR||std::find(qwen3_w8a8::kQuantSites.begin(),qwen3_w8a8::kQuantSites.end(),r.site_id)!=qwen3_w8a8::kQuantSites.end();}),m.routes.end());
  for(auto&r:m.routes)if(r.site_id==qwen3_c192::kPolicySite)r.arguments=qwen3_c192::policy_arguments(512,false);
  for(int64_t i=0;i<3;++i)m.routes.push_back({qwen3_c192::kMlpSite,FmbRouteFamily::LINEAR,1,0,{1,i==2?128:192,5120,25600,8},i});
  m.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
  assert(owner.qwen3_c192_manifest_valid(m)&&owner.qwen3_prefill_w8a8_manifest_valid(m));
  auto bad=m;bad.routes.back().arguments[2]++;assert(!owner.qwen3_c192_manifest_valid(bad));
  owner.qwen3_c192_=false;owner.qwen3_prefill_w8a8_=true;
 }

 {
  owner.qwen3_c256_=true;owner.qwen3_prefill_w8a8_=true;
  auto m=manifest(owner,512,256);
  assert(owner.qwen3_c192_manifest_valid(m)&&owner.qwen3_prefill_w8a8_manifest_valid(m));
  validate_fmb_physical_manifest(m,512,0);
  for(size_t i=0;i<m.routes.size();++i)if(m.routes[i].site_id==qwen3_c192::kPolicySite){
   assert(m.routes[i].arguments[0]==3&&m.routes[i].arguments[9]==9&&m.routes[i].arguments[10]==65536);
   auto stale=m;stale.routes[i].arguments[0]=2;stale.routes[i].arguments.resize(9);
   assert(!owner.qwen3_c192_manifest_valid(stale));
  }
  LayoutContext l;l.chunk_size=l.kv_insert_chunk_size=256;l.is_causal=true;l.batch_size=1;l.attention_policy=AttentionExecutionPolicy::DDR_KV;
  FmbThreeStageChunkPlan plan;plan.chunk_mode=ChunkMode::SEQUENTIAL;plan.input.chunks={{0,0,512,512}};plan.compute.chunks={{0,0,256,256},{1,256,256,512}};plan.qkv.chunks=plan.compute.chunks;
  assert(owner.qwen3_prefill_w8a8_plan_valid(plan,l,512,0));
  auto badplan=plan;badplan.compute.chunks={{0,0,128,128},{1,128,128,256},{2,256,128,384},{3,384,128,512}};
  assert(!owner.qwen3_prefill_w8a8_plan_valid(badplan,l,512,0));
  for(size_t i=0;i<m.routes.size();++i){auto bad=m;bad.routes.erase(bad.routes.begin()+i);assert(!owner.qwen3_c192_manifest_valid(bad)||!owner.qwen3_prefill_w8a8_manifest_valid(bad));
   bad=m;bad.routes.push_back(bad.routes[i]);assert(!owner.qwen3_c192_manifest_valid(bad)||!owner.qwen3_prefill_w8a8_manifest_valid(bad));
   for(size_t j=0;j<m.routes[i].arguments.size();++j){bad=m;++bad.routes[i].arguments[j];assert(!owner.qwen3_c192_manifest_valid(bad)||!owner.qwen3_prefill_w8a8_manifest_valid(bad));}}
  for(size_t i=0;i<m.routes.size();++i)if(m.routes[i].site_id==qwen3_w8a8::kUpSite){
   const auto&r=m.routes[i];assert(r.selector==202&&r.arguments==qwen3_c192::up_mul_arguments(true));
   assert(r.arguments[8]==272&&r.arguments[9]==64&&r.arguments[10]==384);
   auto stale=m;stale.routes[i].arguments[8]=512;stale.routes[i].arguments[10]=128;
   assert(!owner.qwen3_prefill_w8a8_manifest_valid(stale));
  }
  owner.qwen3_c256_=false;owner.qwen3_c192_=true;assert(!owner.qwen3_c192_manifest_valid(m));owner.qwen3_c192_=false;owner.qwen3_c256_=true;
  owner.qwen3_prefill_w8a8_=false;
  m.routes.erase(std::remove_if(m.routes.begin(),m.routes.end(),[](const auto&r){return r.site_id==qwen3_w8a8::kPolicySite||r.family==FmbRouteFamily::LINEAR||std::find(qwen3_w8a8::kQuantSites.begin(),qwen3_w8a8::kQuantSites.end(),r.site_id)!=qwen3_w8a8::kQuantSites.end();}),m.routes.end());
  for(auto&r:m.routes)if(r.site_id==qwen3_c192::kPolicySite)r.arguments=qwen3_c192::policy_arguments(512,false,256);
  for(int64_t i=0;i<2;++i){m.routes.push_back({qwen3_c192::kMlpSite,FmbRouteFamily::LINEAR,1,0,{1,256,5120,25600,8},i});m.routes.push_back({qwen3_c192::kUpMulSite,FmbRouteFamily::LINEAR,202,0,qwen3_c192::up_mul_arguments(false),i});}
  m.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
  std::sort(m.routes.begin(),m.routes.end(),[](const auto&a,const auto&b){return std::make_tuple(int64_t(a.family),a.site_id,a.invocation)<std::make_tuple(int64_t(b.family),b.site_id,b.invocation);});
  validate_fmb_physical_manifest(m,512,0);
  assert(owner.qwen3_c192_manifest_valid(m)&&owner.qwen3_prefill_w8a8_manifest_valid(m));
  for(size_t i=0;i<m.routes.size();++i)if(m.routes[i].site_id==qwen3_c192::kUpMulSite){auto bad=m;bad.routes.erase(bad.routes.begin()+i);assert(!owner.qwen3_c192_manifest_valid(bad));for(size_t j=0;j<m.routes[i].arguments.size();++j){bad=m;++bad.routes[i].arguments[j];assert(!owner.qwen3_c192_manifest_valid(bad));}}
  owner.qwen3_c256_=false;owner.qwen3_prefill_w8a8_=true;
 }

 {
  owner.qwen3_c512_=true;owner.qwen3_prefill_w8a8_=true;
  auto m=manifest(owner,512,512);assert(owner.qwen3_c192_manifest_valid(m)&&owner.qwen3_prefill_w8a8_manifest_valid(m));validate_fmb_physical_manifest(m,512,0);
  LayoutContext l;l.chunk_size=l.kv_insert_chunk_size=512;l.is_causal=true;l.batch_size=1;l.attention_policy=AttentionExecutionPolicy::DDR_KV;
  FmbThreeStageChunkPlan plan;plan.chunk_mode=ChunkMode::SEQUENTIAL;plan.input.chunks=plan.qkv.chunks=plan.compute.chunks={{0,0,512,512}};
  assert(owner.qwen3_prefill_w8a8_plan_valid(plan,l,512,0));auto split=plan;split.compute.chunks={{0,0,256,256},{1,256,256,512}};assert(!owner.qwen3_prefill_w8a8_plan_valid(split,l,512,0));
  for(size_t i=0;i<m.routes.size();++i){auto bad=m;bad.routes.erase(bad.routes.begin()+i);assert(!owner.qwen3_c192_manifest_valid(bad)||!owner.qwen3_prefill_w8a8_manifest_valid(bad));bad=m;bad.routes.push_back(bad.routes[i]);assert(!owner.qwen3_c192_manifest_valid(bad)||!owner.qwen3_prefill_w8a8_manifest_valid(bad));for(size_t j=0;j<m.routes[i].arguments.size();++j){bad=m;++bad.routes[i].arguments[j];assert(!owner.qwen3_c192_manifest_valid(bad)||!owner.qwen3_prefill_w8a8_manifest_valid(bad));}}
  auto bad=m;bad.logical_length=511;assert(!owner.qwen3_c192_manifest_valid(bad));
  bad=m;bad.routes.push_back({qwen3_c512::kDmaSite,FmbRouteFamily::MUTABLE_DMA,1,0,qwen3_c512::dma_arguments(qwen3_c512::DmaRole::PostNorm),3});assert(!owner.qwen3_c192_manifest_valid(bad));
  bad=m;for(auto&r:bad.routes)if(r.site_id==qwen3_c512::kPolicySite)r.arguments[0]=2;assert(!owner.qwen3_c192_manifest_valid(bad));
  bad=m;for(auto&r:bad.routes)if(r.site_id==qwen3_w8a8::kQuantSites[2])r.arguments[0]=5;assert(!owner.qwen3_prefill_w8a8_manifest_valid(bad));
  const auto decode=manifest(owner,1,1);assert(owner.qwen3_c192_manifest_valid(decode)&&owner.qwen3_prefill_w8a8_manifest_valid(decode));
  bad=m;for(auto&r:bad.routes)if(r.site_id==qwen3_c512::kDmaSite&&r.invocation==1)r.selector=1;assert(!owner.qwen3_c192_manifest_valid(bad));
  bad=m;for(auto&r:bad.routes)if(r.site_id==qwen3_c512::kDmaSite&&r.invocation==4)r.arguments.back()=0;assert(!owner.qwen3_c192_manifest_valid(bad));
  bad=m;for(auto&r:bad.routes)if(r.family==FmbRouteFamily::ALL_REDUCE)r.selector=203;assert(!owner.qwen3_c192_manifest_valid(bad));
  owner.qwen3_c512_=false;assert(!owner.qwen3_c192_manifest_valid(m)&&!owner.qwen3_c192_manifest_valid(decode));
 }
 auto decode=manifest(owner,1,1);assert(owner.qwen3_prefill_w8a8_manifest_valid(decode));owner.qwen3_prefill_w8a8_=false;assert(!owner.qwen3_prefill_w8a8_manifest_valid(decode));decode.routes.clear();assert(owner.qwen3_prefill_w8a8_manifest_valid(decode));owner.qwen3_prefill_w8a8_=true;assert(!owner.qwen3_prefill_w8a8_manifest_valid(decode));
}
'''
    _run(tmp_path, code)


def test_qwen3_w8a8_actual_cold_profile_and_mlp_emission(tmp_path):
    """Real cold admission and launch call sites; device math is not simulated."""
    model = MODEL.read_text()
    code = _prefix().replace("class SpmAllocator {\npublic:", "class SpmAllocator {\npublic:\nstatic constexpr uint64_t SPM_USABLE=8387584;\n") + _definition(model, "enum class CausalDecoderPlanningMode") + ";\n"
    code += r'''
constexpr int64_t CAUSAL_DECODER_Q_SITE=1,CAUSAL_DECODER_K_SITE=2,CAUSAL_DECODER_V_SITE=3,CAUSAL_DECODER_O_SITE=4,FMB_SHARED_MLP_SILU_MUL_SITE=5,CAUSAL_DECODER_REDUCE_SITE=6;
bool operator==(Tensor::Device a,Tensor::Device b){return a.kind==b.kind;}
enum class KernelId { LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM, DYNAMIC_QUANTIZE_W8A8, RMS_NORM_QUANTIZE_W8A8_H5120_M128, RMS_NORM_QUANTIZE_W8A8_H5120_M192, RMS_NORM_QUANTIZE_W8A8_H5120_M256, MUL_QUANTIZE_W8A8_H3200_M128, MUL_QUANTIZE_W8A8_H3200_M192, QWEN3_UP_MUL_W8A16_M320N112K128, QWEN3_UP_MUL_W8A8_M272N64K384, QWEN3_UP_MUL_W8A8_M512N64K128,
#define RPU_W8A8_CANDIDATE(id,name,ordinal,m,n,k,lk,flags) id,
#include "ops/rpu_linear_w8a8_candidates.inc"
#undef RPU_W8A8_CANDIDATE
};
struct KernelCache {bool loaded=true,rms_loaded=true;KernelId missing=static_cast<KernelId>(-1);static KernelCache& instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{return loaded&&id!=missing&&(rms_loaded||id!=KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M128);}};
enum class RpuRmsNormSpmRoute:int64_t {BASE=1,V32_GRID8=6};
struct NormContract{RpuRmsNormSpmRoute route=RpuRmsNormSpmRoute::V32_GRID8;RpuRmsNormSpmRoute resolve(int64_t m,int64_t k)const{assert((m==128||m==192||m==256)&&k==5120);return route;}};
struct RpuExecutionCoordinator {static void require_graph_quiescent(const char*){}static void check_current_thread_execution_allowed(const char*){}};
struct RpuKernelGraph {static bool has_active(){return false;}};
enum class CausalDecoderLinearRoute {AUTO_TILE=1};
enum class ValuOpType {SILU,MUL}; enum class GeluMode {NONE};
struct Call {char kind;uint32_t x,sx,y;int64_t m,n,k;int partition;};
std::vector<Call> calls;
struct MockSpm {uint32_t addr(int,int)const{return 0x10000000;}} host_spm;
#define SPM_ALLOC host_spm
void rpu_launch_dynamic_quantize_w8a8_spm_kernel(uint32_t x,uint32_t q,uint32_t sx,int64_t m,int64_t k,int cores){assert(cores==8);calls.push_back({'Q',x,sx,q,m,0,k,1});}
void rpu_launch_rms_norm_quantize_w8a8_spm_kernel(uint32_t x,uint32_t q,uint32_t sx,uint32_t gamma,int64_t m,double eps,int cores){assert(cores==8&&(m==128||m==192||m==256)&&eps==1e-6&&gamma==0x70000000);calls.push_back({'N',x,sx,q,m,0,5120,1});}
void rpu_launch_linear_w8a8_spm_kernel(uint32_t x,uint32_t sx,const Tensor&w,const Tensor&sw,uint32_t y,int64_t m,int64_t n,int64_t k,int partition,int cores){assert(cores==8&&w.id==sw.id);calls.push_back({'L',x,sx,y,m,n,k,partition});}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t x,const Tensor&w,uint32_t y,int64_t m,int64_t n,int64_t k,int partition,int cores,uint32_t bias,const Tensor&sw){assert(cores==8&&bias==0&&w.id==sw.id);calls.push_back({'L',x,0,y,m,n,k,partition});}
void rpu_launch_qwen3_up_mul_w8a16_spm_kernel(uint32_t x,const Tensor&w,const Tensor&sw,uint32_t gate,int64_t m){assert(m==256&&w.id==5&&sw.id==5);calls.push_back({'U',x,0,gate,m,25600,5120,1});}
void rpu_launch_qwen3_up_mul_w8a8_spm_kernel(uint32_t x,uint32_t sx,const Tensor&w,const Tensor&sw,uint32_t gate,int64_t m){assert((m==256||m==512)&&w.id==5&&sw.id==5);calls.push_back({'U',x,sx,gate,m,25600,5120,1});}
void rpu_launch_mul_quantize_w8a8_spm_kernel(uint32_t gate,uint32_t up,uint32_t q,uint32_t sx,int64_t m,int64_t k,int cores){assert((m==128||m==192)&&k==3200&&cores==8&&gate!=up);calls.push_back({'X',gate,sx,q,m,0,k,1});}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t x,uint32_t y,int64_t n,ValuOpType v,GeluMode,int cores){assert(v==ValuOpType::SILU&&cores==8);calls.push_back({'S',x,0,y,n,0,0,0});}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t x,uint32_t up,uint32_t y,int64_t n,ValuOpType v,c10::Half,int cores){assert(v==ValuOpType::MUL&&cores==8);calls.push_back({'M',x,up,y,n,0,0,0});}
void rpu_launch_silu_mul_spm_kernel(uint32_t x,uint32_t up,uint32_t y,int64_t n,int cores){assert(cores==8);calls.push_back({'F',x,up,y,n,0,0,0});}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t x,uint32_t residual,uint32_t y,int64_t m,int64_t n,int input_cores,int output_cores){assert(input_cores==8&&output_cores==8);calls.push_back({'R',x,residual,y,m,n,0,0});}
void rpu_launch_qwen3_c512_all_reduce_residual_local_spm_kernel(uint32_t x,uint32_t compact,uint32_t y){assert(x!=compact&&compact!=y&&x!=y);calls.push_back({'R',x,compact,y,128,5120,0,2});}
void rpu_launch_qwen3_c192_all_reduce_residual_inplace_kernel(uint32_t x,uint32_t r,int64_t m){assert(m==192||m==128||m==256);calls.push_back({'R',x,r,r,m,5120,0,0});}
int64_t causal_ring_route(int64_t,int64_t){return 2;}
void rpu_launch_rmsnorm_spm_kernel(uint32_t x,uint32_t y,uint32_t gamma,int64_t m,int64_t k,double eps,RpuRmsNormSpmRoute route,int cores){assert(x==y&&m==128&&k==5120&&eps==1e-6&&cores==8&&route==RpuRmsNormSpmRoute::V32_GRID8);calls.push_back({'Z',x,gamma,y,m,0,k,0});}
struct Context {bool input_in_spm=false,output_to_spm=false;std::vector<FmbRouteManifestEntry> consumed;void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation=0){consumed.push_back({site,family,selector,flags,std::move(args),invocation});}};
class CausalDecoderModel {
public:
 virtual ~CausalDecoderModel()=default;
 struct LayerWeights {Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws,input_norm_w,post_norm_w;};
 std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(64);
 bool qwen3_c192_=false,qwen3_c192_bound_=false,qwen3_c256_=false,qwen3_c256_bound_=false,qwen3_c512_=false,qwen3_c512_bound_=false;
 struct ChunkEnvelope {int64_t max_kv_len=512,chunk=128;} envelope;
 const ChunkEnvelope& chunk_envelope()const{return envelope;}
 bool qwen3_prefill_w8a8_=false,qwen3_prefill_w8a8_bound_=false;
 bool has_qk_norm_=true,use_silu_=true,linear_acc32_=false;
 bool has_mrope_=false,has_qkv_bias_=false,nvfp4_=false,adarms_=false,decoder_mixed_precision_=false;
 bool decode_gate_up_fusion_=false,wall_prefill_staged_mlp_=false,lingbot_prefill_mode_bound_=false,rhinovla_high_precision_=false;
 // The real installer enables this capability by default; it is not a resolved SPM-KV route.
 bool qwen3_spm_kv_by_mha_enabled_=true,pre_layers_residual1_ready_=false,equal_two_prefill_=false;
 bool wall_z1_dispatch_=false,qwen3vl_pooler_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
 bool fuse_prefill_last_lm_head_=false,debug=false;bool get_debug_export()const{return debug;}
 bool silu_fused=false;double eps_=1e-6;int cores=8,invalidations=0,last_chunk=0;
 std::vector<int64_t> deepstack_lang_layers_{};
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 int num_cores()const{return cores;}int attn_tp()const{return cores;}int mlp_tp()const{return cores;}
 int64_t num_layers()const{return 64;}int64_t hidden_size()const{return 5120;}int64_t intermediate_size()const{return 25600;}
 int64_t num_q_heads()const{return 64;}int64_t num_kv_heads()const{return 8;}int64_t head_dim()const{return 128;}
 bool wall_sorting_route_enabled()const{return false;}int get_last_resolved_chunk_size()const{return last_chunk;}
 void invalidate_model_state(){++invalidations;}bool use_planned_silu_mul()const{return silu_fused;}
 Context context;Context&ctx(){return context;}
 NormContract ordinary_rmsnorm_spm_contract_;int norm_consumptions=0;
 bool ordinary_rmsnorm_route_enabled()const{return ordinary_rmsnorm_spm_contract_.route==RpuRmsNormSpmRoute::V32_GRID8;}
 RpuRmsNormSpmRoute consume_ordinary_rmsnorm_route(int64_t m,int64_t k,int64_t idx,int64_t role){assert((m==128||m==192||m==256)&&k==5120&&idx==2&&role==0);++norm_consumptions;return ordinary_rmsnorm_spm_contract_.route;}
 int bad_layout=0;std::vector<std::string> resolved;
 uint32_t addr(int,const char* name){resolved.push_back(name);if(qwen3_c512_){const std::map<std::string,uint32_t> offsets{{"oproj",1024},{"oproj_suffix",2622464},{"a8_mlp",1024},{"a8_mlp_sx",0},{"a8_o",5243904},{"a8_o_sx",0},{"o_residual_tile",5243904},{"o_compact_residual",6554624},{"down",1639424},{"down_residual_tile",0},{"down_compact_residual",1310720}};auto it=offsets.find(name);if(it!=offsets.end()){if(bad_layout==3&&std::string(name)=="o_compact_residual")return 0x10000000+1024;if(bad_layout==4&&std::string(name)=="down_compact_residual")return 0x10000000;return 0x10000000+it->second+((bad_layout==1&&std::string(name)=="oproj_suffix")||(bad_layout==2&&std::string(name)=="a8_mlp")?256:0);}}
 if(qwen3_inplace_capacity()&&std::string(name)=="residual2")name="residual1";static const std::vector<std::string> names{"a8_qkv","a8_qkv_sx","a8_o","a8_o_sx","a8_mlp","a8_mlp_sx","a8_down","a8_down_sx","input_norm","output","residual1","residual2","q","k","v","oproj","gate","up","down","mlp_norm","o_residual_tile","norm2_input_tile","down_residual_tile"};auto it=std::find(names.begin(),names.end(),name);assert(it!=names.end());return 0x10000000+uint32_t(it-names.begin())*0x10000;}

 uint32_t norm_input_addr(int,int)const{return 0x70000000;}uint32_t norm_post_addr(int,int)const{return 0x70000000;}uint32_t norm_final_addr(int)const{return 0x70000000;}
 void emit_layer_input_dma(int layer,const ChunkInfo&c,const char*dst="residual1"){assert(layer>=0&&layer<64);calls.push_back({'I',uint32_t(c.offset),uint32_t(layer),addr(0,dst),c.len,0,5120,0});}
 void emit_layer_residual_shard_load(int layer,const ChunkInfo&c,const char*dst,bool scratch){assert(c.len==128&&layer>=0&&layer<64);calls.push_back({scratch?'d':'i',uint32_t(c.offset),uint32_t(layer),addr(0,dst),c.len,16,5120,0});}
 void emit_layer_chain_scratch_load(int layer,const ChunkInfo&c,const char*dst){assert(layer>=0&&layer<64);calls.push_back({'D',uint32_t(c.offset),uint32_t(layer),addr(0,dst),c.len,0,5120,0});}
 void emit_layer_chain_scratch_store(int layer,const ChunkInfo&c,const char*src){calls.push_back({'T',addr(0,src),uint32_t(layer),uint32_t(c.offset),c.len,0,5120,0});}
 void emit_layer_output_dma(int layer,const ChunkInfo&c,const char*src){calls.push_back({'O',addr(0,src),uint32_t(layer),uint32_t(c.offset),c.len,0,5120,0});}
 void install_metadata(){for(auto&layer:layer_weights_){Tensor* weights[]{&layer.q_w,&layer.k_w,&layer.v_w,&layer.o_w,&layer.gate_w,&layer.up_w,&layer.down_w};Tensor* scales[]{&layer.q_ws,&layer.k_ws,&layer.v_ws,&layer.o_ws,&layer.gate_ws,&layer.up_ws,&layer.down_ws};for(int p=0;p<7;++p){weights[p]->dtype=at::kChar;weights[p]->shape={qwen3_w8a8::kGlobalN[p],qwen3_w8a8::kGlobalK[p]};scales[p]->dtype=at::kHalf;scales[p]->shape={qwen3_w8a8::kGlobalN[p]};weights[p]->id=scales[p]->id=p;}for(auto*norm:{&layer.input_norm_w,&layer.post_norm_w}){norm->dtype=at::kHalf;norm->shape={5120};}}}
'''
    for marker in ("int64_t qwen3_inplace_capacity(", "KernelId qwen3_prefill_w8a8_linear_id(",
                   "void require_qwen3_32b_w8_profile(", "void require_qwen3_prefill_w8a8_profile(", "void set_qwen3_c192(", "void set_qwen3_c256(", "void set_qwen3_c512(", "void set_prefill_w8a8(",
                   "void set_linear_acc32(", "std::array<int64_t, 7> qwen3_prefill_w8a8_linear_sites(",
                   "void consume_qwen3_c512_dma(", "uint32_t qwen3_c512_oproj_base(", "void emit_qwen3_c512_rms_quant(", "void emit_qwen3_c512_reduce(", "void emit_qwen3_prefill_w8a8_quant(", "void emit_qwen3_prefill_w8a8_rms_quant(", "void emit_qwen3_prefill_w8a8_linear(",
                   "void emit_qwen3_c192_down_reduce(", "void emit_qwen3_c192_a16_mlp(", "void emit_qwen3_prefill_w8a8_mlp("):
        code += _definition(model, marker).replace('#include "../ops/', '#include "ops/')
    ingress = model[model.index("        // Input DMA (if input is not already in SPM)"):]
    ingress = ingress[ingress.index("        if (qwen3vl_multiview_text_composite_dispatch_"):]
    ingress = ingress[:ingress.index("\n\n        // ─")]
    code += "void emit_qwen3vl_multiview_text_complement_ingress(int,const ChunkInfo&,uint32_t){assert(false);}"
    code += "void actual_common_input(int layer_idx,const ChunkInfo&chunk){const bool qwen_c512=qwen3_c512_;const struct{bool external=false;}z1_input;const uint32_t input_residual=addr(0,\"residual1\");" + ingress + "}"
    code += r'''
};
struct Derived: CausalDecoderModel{};
template<class F>void rejected(F&& f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}assert(bad);}
int main(){
 {CausalDecoderModel x;x.install_metadata();rejected([&]{x.set_qwen3_c192(true);});assert(!x.qwen3_c192_bound_);x.envelope.chunk=192;x.set_qwen3_c192(true);assert(x.qwen3_c192_);rejected([&]{x.set_qwen3_c192(false);});rejected([&]{x.set_linear_acc32(true);});x.set_prefill_w8a8(true);assert(x.qwen3_prefill_w8a8_);}
 {CausalDecoderModel x;x.install_metadata();x.envelope.chunk=192;rejected([&]{x.set_prefill_w8a8(true);});x.linear_acc32_=true;rejected([&]{x.set_qwen3_c192(true);});}
 {
  CausalDecoderModel x;x.install_metadata();x.envelope.chunk=192;x.set_qwen3_c192(true);
  for(int64_t m:{192,128})for(bool fused:{false,true}){
   calls.clear();x.context.consumed.clear();x.silu_fused=fused;ChunkInfo c{2,384,m,384+m};
   x.emit_qwen3_c192_a16_mlp(x.layer_weights_[0],c);
   std::string order;for(const auto&call:calls)order+=call.kind;assert(order==(fused?"LLFLR":"LSLMLR"));
   assert(calls[0].x==x.addr(0,"mlp_norm")&&calls[fused?1:2].x==x.addr(0,"mlp_norm"));
   assert(calls.back().sx==x.addr(0,"residual1")&&calls.back().y==calls.back().sx);
   assert(x.context.consumed.size()==size_t(fused?3:2));
  }
  x.set_prefill_w8a8(true);
  for(int64_t m:{192,128}){
   calls.clear();x.context.consumed.clear();x.silu_fused=false;ChunkInfo c{2,384,m,384+m};
   rejected([&]{x.emit_qwen3_c192_a16_mlp(x.layer_weights_[0],c);});assert(calls.empty());
   x.emit_qwen3_prefill_w8a8_rms_quant(c,2,x.addr(0,"residual1"),0x70000000);
   x.emit_qwen3_prefill_w8a8_mlp(x.layer_weights_[0],c,true);
   std::string order;for(const auto&call:calls)order+=call.kind;assert(order=="NLSLXLR");
   assert(calls.front().x==calls.back().y&&calls.back().sx==calls.back().y);
   const auto&r=x.context.consumed.back();assert(r.site_id==qwen3_c192::kDownReduceSite&&r.selector==201&&r.arguments==qwen3_c192::ring_arguments(m));
  }
 }

 {
  CausalDecoderModel x;x.install_metadata();rejected([&]{x.set_qwen3_c256(true);});assert(!x.qwen3_c256_bound_);
  x.envelope.chunk=256;x.set_qwen3_c256(true);assert(x.qwen3_c256_&&!x.qwen3_c192_);
  rejected([&]{x.set_qwen3_c192(true);});rejected([&]{x.set_qwen3_c256(false);});rejected([&]{x.set_linear_acc32(true);});
  ChunkInfo c{2,0,256,256};calls.clear();x.context.consumed.clear();
  x.emit_qwen3_c192_a16_mlp(x.layer_weights_[0],c);
  std::string order;for(const auto&call:calls)order+=call.kind;assert(order=="LSULR");
  assert(calls[0].y==calls[2].y&&calls[3].x==calls[2].y&&calls[2].m==256);
  assert(calls[2].x==x.addr(0,"mlp_norm"));
  assert(x.context.consumed.size()==3&&x.context.consumed[1].arguments==qwen3_c192::up_mul_arguments(false));
  x.set_prefill_w8a8(true);calls.clear();x.context.consumed.clear();
  x.emit_qwen3_prefill_w8a8_rms_quant(c,2,x.addr(0,"residual1"),0x70000000);
  x.emit_qwen3_prefill_w8a8_mlp(x.layer_weights_[0],c,true);
  order.clear();for(const auto&call:calls)order+=call.kind;assert(order=="NLSUQLR");
  assert(calls[1].y==calls[3].y&&calls[4].x==calls[3].y&&calls[3].m==256);
  assert(calls[1].x==calls[3].x&&calls[1].sx==calls[3].sx);
  for(const auto&call:calls)assert(call.y!=x.addr(0,"up"));
  assert(x.context.consumed[2].selector==202&&x.context.consumed[2].arguments==qwen3_c192::up_mul_arguments(true));
 }

 {
  CausalDecoderModel x;x.install_metadata();x.envelope.chunk=512;x.set_qwen3_c512(true);
  rejected([&]{x.set_prefill_w8a8(false);});assert(!x.qwen3_prefill_w8a8_bound_);x.set_prefill_w8a8(true);
  rejected([&]{x.set_qwen3_c256(true);});rejected([&]{x.set_linear_acc32(true);});
  ChunkInfo c{0,0,512,512};calls.clear();x.actual_common_input(0,c);assert(calls.empty());
  x.qwen3_c512_=false;x.actual_common_input(0,c);assert(calls.size()==1&&calls[0].m==512);x.qwen3_c512_=true;
  for(int layer:{0,1,63}){
   calls.clear();x.context.consumed.clear();x.emit_qwen3_c512_rms_quant(layer,c,0);
   assert(calls.size()==8);for(int i=0;i<4;++i){assert(calls[2*i].kind=='I'&&calls[2*i].x==uint32_t(i*128)&&calls[2*i].m==128);assert(calls[2*i+1].kind=='N'&&calls[2*i+1].y==x.addr(0,"a8_qkv")+i*128*5120&&calls[2*i+1].sx==x.addr(0,"a8_qkv_sx")+i*256);}
   calls.clear();x.context.consumed.clear();x.emit_qwen3_c512_reduce(layer,c,false);assert(calls.size()==16);
   for(int i=0;i<4;++i){assert(calls[4*i].kind=='i'&&calls[4*i+1].kind=='R'&&calls[4*i+2].kind=='N'&&calls[4*i+3].kind=='T');assert(calls[4*i+1].x==x.addr(0,"oproj")+i*128*5120*2&&calls[4*i+1].sx==x.addr(0,"o_compact_residual")&&calls[4*i+1].y==x.addr(0,"o_residual_tile")&&calls[4*i].y==calls[4*i+1].sx&&calls[4*i].n==16);assert(calls[4*i+2].x==calls[4*i+1].y&&calls[4*i+2].y==x.addr(0,"a8_mlp")+i*128*5120&&calls[4*i+2].sx==x.addr(0,"a8_mlp_sx")+i*256);assert(calls[4*i+2].y+128*5120<=calls[4*i+1].x+128*5120*2);assert(calls[4*i+3].x==calls[4*i+1].y);}
   int post=0;for(const auto&r:x.context.consumed){if(r.site_id==qwen3_w8a8::kQuantSites[2])++post;assert(r.site_id!=qwen3_c512::kDmaSite||r.invocation!=3);}assert(post==1);
   calls.clear();rejected([&]{x.emit_qwen3_c512_rms_quant(layer,c,2);});assert(calls.empty());
   calls.clear();x.emit_qwen3_prefill_w8a8_mlp(x.layer_weights_[layer],c,true,layer);
   std::string order;for(const auto&call:calls)order+=call.kind;assert(order==std::string("LSUQL")+(layer==63?"dRZOdRZOdRZOdRZO":"dROdROdROdRO"));
   assert(calls[2].m==512&&calls[2].y==calls[0].y&&calls[3].x==calls[2].y&&calls[4].m==512);
   for(const auto&call:calls)assert(call.y!=x.addr(0,"up"));
  }
  calls.clear();rejected([&]{x.emit_qwen3_c512_rms_quant(0,{1,128,384,512},0);});assert(calls.empty());
  for(int bad:{1,2}){x.bad_layout=bad;calls.clear();rejected([&]{x.emit_qwen3_c512_reduce(0,c,false);});assert(calls.empty());rejected([&]{x.emit_qwen3_prefill_w8a8_linear(c,qwen3_w8a8::Projection::O,x.layer_weights_[0].o_w,x.layer_weights_[0].o_ws,"oproj");});assert(calls.empty());}x.bad_layout=0;
  for(int bad:{3,4}){x.bad_layout=bad;calls.clear();rejected([&]{x.emit_qwen3_c512_reduce(0,c,bad==4);});assert(calls.empty());}x.bad_layout=0;
  calls.clear();x.resolved.clear();x.emit_qwen3_prefill_w8a8_linear(c,qwen3_w8a8::Projection::O,x.layer_weights_[0].o_w,x.layer_weights_[0].o_ws,"oproj");assert(calls.size()==1&&calls[0].y==x.addr(0,"oproj"));assert(std::find(x.resolved.begin(),x.resolved.end(),"oproj_suffix")!=x.resolved.end());
  calls.clear();x.context.input_in_spm=true;rejected([&]{x.emit_qwen3_c512_reduce(0,c,false);});assert(calls.empty());
 }
 for(int badmode:{0,1,2,3}){CausalDecoderModel x;x.install_metadata();x.envelope.chunk=512;if(badmode==0)x.fuse_prefill_last_lm_head_=true;if(badmode==1)x.debug=true;if(badmode==2)KernelCache::instance().missing=KernelId::QWEN3_UP_MUL_W8A8_M512N64K128;if(badmode==3)KernelCache::instance().missing=KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM;rejected([&]{x.set_qwen3_c512(true);});assert(!x.qwen3_c512_bound_);KernelCache::instance().missing=static_cast<KernelId>(-1);}
 CausalDecoderModel owner;owner.install_metadata();owner.set_prefill_w8a8(true);assert(owner.qwen3_prefill_w8a8_&&owner.invalidations==1&&owner.qwen3_spm_kv_by_mha_enabled_);
 {CausalDecoderModel off;off.install_metadata();off.qwen3_spm_kv_by_mha_enabled_=false;off.set_prefill_w8a8(true);assert(off.qwen3_prefill_w8a8_&&!off.qwen3_spm_kv_by_mha_enabled_);}
 rejected([&]{owner.set_prefill_w8a8(false);});rejected([&]{owner.set_prefill_w8a8(true);});rejected([&]{owner.set_linear_acc32(true);});assert(!owner.linear_acc32_);
 for(auto member:{&CausalDecoderModel::has_mrope_,&CausalDecoderModel::has_qkv_bias_,&CausalDecoderModel::nvfp4_,&CausalDecoderModel::adarms_,&CausalDecoderModel::decoder_mixed_precision_,&CausalDecoderModel::linear_acc32_}){CausalDecoderModel bad;bad.install_metadata();bad.*member=true;rejected([&]{bad.set_prefill_w8a8(true);});assert(!bad.qwen3_prefill_w8a8_bound_&&bad.invalidations==0);}
 for(int cores:{4,6}){CausalDecoderModel bad;bad.install_metadata();bad.cores=cores;rejected([&]{bad.set_prefill_w8a8(true);});}
 {Derived bad;bad.install_metadata();rejected([&]{bad.set_prefill_w8a8(true);});}
 {CausalDecoderModel bad;bad.install_metadata();bad.layer_weights_.back().down_w.dtype=at::kHalf;rejected([&]{bad.set_prefill_w8a8(true);});bad.install_metadata();bad.layer_weights_[31].q_ws.shape={8191};rejected([&]{bad.set_prefill_w8a8(true);});}
 {CausalDecoderModel bad;bad.install_metadata();KernelCache::instance().loaded=false;rejected([&]{bad.set_prefill_w8a8(true);});KernelCache::instance().loaded=true;bad.last_chunk=128;rejected([&]{bad.set_prefill_w8a8(true);});}
 {CausalDecoderModel bad;bad.install_metadata();KernelCache::instance().rms_loaded=false;rejected([&]{bad.set_prefill_w8a8(true);});KernelCache::instance().rms_loaded=true;assert(!bad.qwen3_prefill_w8a8_bound_);}

 {CausalDecoderModel bad;bad.install_metadata();KernelCache::instance().missing=KernelId::MUL_QUANTIZE_W8A8_H3200_M128;rejected([&]{bad.set_prefill_w8a8(true);});assert(!bad.qwen3_prefill_w8a8_bound_);KernelCache::instance().missing=static_cast<KernelId>(-1);}
 {CausalDecoderModel bad;bad.install_metadata();bad.envelope.chunk=256;KernelCache::instance().missing=KernelId::QWEN3_UP_MUL_W8A16_M320N112K128;rejected([&]{bad.set_qwen3_c256(true);});assert(!bad.qwen3_c256_bound_);KernelCache::instance().missing=static_cast<KernelId>(-1);}
 for(auto missing:{KernelId::QWEN3_UP_MUL_W8A8_M272N64K384,KernelId::RMS_NORM_QUANTIZE_W8A8_H5120_M256}){CausalDecoderModel bad;bad.install_metadata();bad.envelope.chunk=256;bad.set_qwen3_c256(true);KernelCache::instance().missing=missing;rejected([&]{bad.set_prefill_w8a8(true);});assert(!bad.qwen3_prefill_w8a8_bound_);KernelCache::instance().missing=static_cast<KernelId>(-1);}
 {CausalDecoderModel bad;bad.install_metadata();bad.ordinary_rmsnorm_spm_contract_.route=RpuRmsNormSpmRoute::BASE;rejected([&]{bad.set_prefill_w8a8(true);});}
 {CausalDecoderModel bad;bad.install_metadata();bad.layer_weights_.back().input_norm_w.dtype=at::kChar;rejected([&]{bad.set_prefill_w8a8(true);});bad.install_metadata();bad.layer_weights_[31].post_norm_w.shape={5119};rejected([&]{bad.set_prefill_w8a8(true);});}
 for(int64_t m:{16,32,64,128})for(bool fused:{false,true}){
  calls.clear();owner.context.consumed.clear();owner.norm_consumptions=0;owner.silu_fused=fused;ChunkInfo chunk{2,256,m,256+m};
  const bool rms=m==128;
  rejected([&]{owner.emit_qwen3_prefill_w8a8_mlp(owner.layer_weights_[0],chunk,!rms);});assert(calls.empty());
  if(rms)owner.emit_qwen3_prefill_w8a8_rms_quant(chunk,2,owner.addr(0,"residual2"),0x70000000);
  owner.emit_qwen3_prefill_w8a8_mlp(owner.layer_weights_[0],chunk,rms);
  std::string order;for(const auto&call:calls)order+=call.kind;assert(order==(rms?"NLSLXLR":std::string("Q")+(fused?"LLFQLR":"LSLMQLR")));
  assert(owner.norm_consumptions==int(rms));
  assert(calls.front().x==owner.addr(0,rms?"residual2":"residual1")&&calls.front().k==5120);
  const auto&gate=calls[1];const auto&up=calls[fused&&!rms?2:3];assert(gate.x==up.x&&gate.sx==up.sx&&gate.n==25600&&gate.k==5120&&gate.partition==1);
  const auto&quant=calls[calls.size()-3];assert(quant.x==owner.addr(0,"gate")&&quant.k==3200);
  const auto&down=calls[calls.size()-2];assert(down.x==quant.y&&down.sx==quant.sx&&down.n==5120&&down.k==25600&&down.partition==0);
  const auto&ring=calls.back();assert(ring.x==owner.addr(0,"down")&&ring.sx==owner.addr(0,"residual2")&&ring.y==owner.addr(0,"residual1"));
  assert(owner.context.consumed.size()==size_t(fused&&!rms?7:6));for(const auto&r:owner.context.consumed)assert(r.invocation==2);
 }
 for(size_t role:{0u,2u}){
  calls.clear();owner.context.consumed.clear();ChunkInfo c{2,256,128,384};
  rejected([&]{owner.emit_qwen3_prefill_w8a8_quant(c,role,0x60000000);});assert(calls.empty());
  owner.emit_qwen3_prefill_w8a8_rms_quant(c,role,0x60000000,0x70000000);
  assert(calls.size()==1&&calls[0].kind=='N'&&calls[0].x==0x60000000&&calls[0].y==owner.addr(0,role==0?"a8_qkv":"a8_mlp"));
  const auto&r=owner.context.consumed.back();assert(r.family==FmbRouteFamily::NORMALIZATION&&r.selector==2&&r.arguments==qwen3_w8a8::quant_arguments(128,role));
 }
 for(size_t role:{1u,3u,4u}){calls.clear();rejected([&]{owner.emit_qwen3_prefill_w8a8_rms_quant({2,0,128,128},role,0x60000000,0x70000000);});assert(calls.empty());}
 calls.clear();owner.ordinary_rmsnorm_spm_contract_.route=RpuRmsNormSpmRoute::BASE;rejected([&]{owner.emit_qwen3_prefill_w8a8_rms_quant({2,0,128,128},0,0x60000000,0x70000000);});assert(calls.empty());
}
'''
    _run(tmp_path, code)


def test_actual_qwen_cap_rejections_allow_padding_search_and_exact_c192(tmp_path):
    """Actual native cap errors flow through the production Python search facade."""
    from test_causal_prefill_padding import _load_function
    model = MODEL.read_text()
    code = _prefix() + r'''
#include <sstream>
#undef TORCH_CHECK
template<class... T>void check(bool ok,T&&...parts){if(!ok){std::ostringstream out;(out<<...<<parts);throw std::runtime_error(out.str());}}
#define TORCH_CHECK(ok,...) check(bool(ok),__VA_ARGS__)
struct Owner {
 bool qwen3_c192_=false,qwen3_c256_=false,qwen3_c512_=false,qwen3_prefill_w8a8_=true,equal_two_prefill_=false,decoder_mixed_precision_=false;
 int64_t configured_chunk_size_cap_=0;
 struct Context {bool is_causal=true;int batch_size=1;std::optional<int> attention_mask;}context;
 const Context&ctx()const{return context;}
 struct Envelope{int64_t max_kv_len=512,chunk=128;}envelope;
 const Envelope&chunk_envelope()const{return envelope;}
 bool chunk_within_envelope(int64_t cs)const{return cs<=envelope.chunk;}
 int64_t enforce_chunk_envelope(int64_t,int64_t,const char*)const{return envelope.chunk;}
 void require_qwen3_32b_w8_profile()const{} void require_qwen3_prefill_w8a8_profile()const{}
 int64_t num_q_heads()const{return 64;}int64_t num_kv_heads()const{return 8;}int64_t head_dim()const{return 128;}
 int64_t hidden_size()const{return 5120;}int64_t intermediate_size()const{return 25600;}
 bool has_qk_norm_=true,has_qkv_bias_=false;
 int attn_tp()const{return 8;}SdpaKernelType sdpa_kernel_=SdpaKernelType::FLASH_ATTN_SPM;
'''
    for marker in ("int64_t qwen3_inplace_capacity(", "SdpaConfig make_sdpa_config(", "bool subclass_chunk_size_valid(",
                   "int64_t subclass_chunk_size_cap("):
        code += _definition(model, marker).replace(" const override", " const")
    code += r'''
};
int main(){
 for(int mode=0;mode<5;++mode){Owner o;o.qwen3_c192_=mode==1||mode==2;o.qwen3_c256_=mode>=3;o.qwen3_prefill_w8a8_=mode!=1&&mode!=3;o.envelope.chunk=mode>=3?256:mode?192:128;
  assert(o.subclass_chunk_size_valid(o.envelope.chunk,512,0));
  if(mode){for(int c:{16,32,64,128,160,176,192,208,256})if(c!=o.envelope.chunk)assert(!o.subclass_chunk_size_valid(c,512,0));assert(!o.subclass_chunk_size_valid(192,512,1));assert(!o.subclass_chunk_size_valid(192,496,0));}
  assert(o.subclass_chunk_size_valid(16,1,513));
  for(int n=512;n<=576;++n){try{auto cap=o.subclass_chunk_size_cap(n,0);std::cout<<mode<<"|"<<n<<"|"<<cap<<"\n";}catch(const std::exception&e){std::cout<<mode<<"|"<<n<<"|!"<<e.what()<<"\n";}}
 }
}
'''
    output = _run(tmp_path, code)
    native = {(int(mode), int(rows)): value for mode, rows, value in
              (line.split("|", 2) for line in output.splitlines())}
    _, plan = _load_function("plan_bounded_prefill_execution")
    for mode, chunk in ((0, 128), (1, 192), (2, 192), (3, 256), (4, 256)):
        calls = []
        def domain(rows):
            calls.append(rows)
            value = native[mode, rows]
            if value.startswith("!"):
                raise RuntimeError(value[1:])
            assert int(value) == chunk
            return ((chunk, chunk, chunk),)
        assert plan(512, 8192, 64, alignment=1, exact_chunk_size=chunk,
                    resolve_stage_domain=domain) == (512, chunk)
        assert calls == list(range(512, 577))
        assert all(native[mode, n].startswith("!RPU_PLANNER_REJECT:CAPABILITY:")
                   for n in range(513, 577))
