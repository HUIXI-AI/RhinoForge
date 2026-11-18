"""Truncated GR00T text scratch: native declarations, allocator and stream order."""
from pathlib import Path
import shutil
import subprocess

from test_fmb_standalone_layout_rebuild import _definition
from test_hy_halo_mask_residency import _program as _layout_program

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/fused/rpu_qwen3_model.h"


def test_actual_gr00t_text_layout_and_layer_lifetimes(tmp_path):
    source = HEADER.read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    program = _layout_program().split("class CausalDecoderModel {public:", 1)[0]
    program = program.replace("#include <algorithm>", "#include <algorithm>\n#include <array>\n#include <map>")
    program += "\n#define TORCH_INTERNAL_ASSERT TORCH_CHECK\n"
    for marker in ("static inline bool sdpa_call_valid(", "inline bool sdpa_is_valid_chunk_size("):
        program += _definition(helpers, marker) + "\n"
    program += r'''
enum class CausalDecoderPlanningMode {ORDINARY,Z1};
class CausalDecoderModel {public:
 int64_t h=2048,inter=6144,nq=16,nkv=8,hd=128,layers=16,cores=8;
 int64_t hidden_size()const{return h;}int64_t intermediate_size()const{return inter;}
 int64_t num_q_heads()const{return nq;}int64_t num_kv_heads()const{return nkv;}
 int64_t head_dim()const{return hd;}int64_t num_layers()const{return layers;}
 int num_cores()const{return cores;}int attn_tp()const{return 8;}int mlp_tp()const{return 8;}int lm_head_tp()const{return 8;}
 bool has_mrope_=true,use_silu_=true,has_qk_norm_=true;
 bool has_qkv_bias_=false,fuse_lm_head_=false,nvfp4_=false,decoder_mixed_precision_=false;
 bool adarms_=false,adarms_mutable_=false,decode_gate_up_fusion_=false,wall_prefill_staged_mlp_=false;
 bool lingbot_prefill_mode_bound_=false,rhinovla_high_precision_=false,pre_layers_residual1_ready_=false;
 bool wall_z1_dispatch_=false,qwen3vl_pooler_z1_dispatch_=false;
 bool qwen3vl_multiview_text_dry_prepared_=false,qwen3vl_multiview_text_composite_prepared_=false;
 bool qwen3vl_multiview_text_composite_dispatch_=false,prefill_packed_gate_up_strided_=false;
 bool explicit_mask_residency_requested_=false;
 int64_t qwen3vl_pooler_z1_prepared_execution_len_=0,qwen3vl_pooler_z1_retained_deepstack_count_=0;
 std::vector<int32_t>mrope_section_{24,20,20};std::vector<int64_t>deepstack_lang_layers_{0,1,2};
 CausalDecoderPlanningMode planning_mode_=CausalDecoderPlanningMode::ORDINARY;
 SdpaKernelType sdpa_kernel_=SdpaKernelType::FLASH_ATTN_SPM;
 bool lingbot_prefill_deepstack_reuses_down(const LayoutContext&)const{return false;}
 bool qwen3vl_4b_packed_decode_qkv(int64_t,bool,bool,int64_t)const{return false;}
 SdpaConfig make_sdpa_config(int mask)const{return {sdpa_kernel_,hd,nq,nkv,8,mask,0};}
'''
    declaration = _definition(source, "std::vector<BufferDecl> causal_baseline_buffer_declarations(")
    boundary = declaration.index("        // Norm-weight preloads")
    # All production Temp slots are retained; device-only Persistent callbacks
    # cannot affect this allocator proof. The optimizer only edits these slots.
    assert "StorageClass::Temp" not in declaration[boundary:]
    program += declaration[:boundary] + "return decls;\n}\n"
    for marker in ("bool gr00t_long_prefill_scratch_profile(",
                   "void apply_gr00t_long_prefill_scratch_alias(",
                   "virtual std::vector<BufferDecl> explicit_mask_baseline_buffer_declarations(",
                   "std::vector<BufferDecl> declare_buffers("):
        program += _definition(source, marker).replace(" override", "") + "\n"
    program += r'''
};
bool changed(const std::vector<BufferDecl>&d){for(auto&b:d)if(std::string(b.name)=="deepstack_scratch")return b.alias_of;return false;}
struct Storage {
 std::map<std::string,int64_t>sizes,offsets;
 std::vector<unsigned char>bytes;std::map<std::string,unsigned char>value;unsigned char stamp=0;
 explicit Storage(const std::vector<BufferDecl>&d){std::vector<SpmAllocator::AllocRequest>r;std::vector<std::string>names;
  for(auto&b:d)if(b.storage==StorageClass::Temp){sizes[b.name]=b.size;if(!b.alias_of){names.push_back(b.name);r.push_back({b.size,b.phase_start,b.phase_end,int(b.scope)});}}
  auto plan=SpmAllocator::plan_temporary_aliased(r);bytes.resize(plan.peak_bytes);
  for(size_t i=0;i<r.size();++i)offsets[names[i]]=plan.offsets[i];
  for(auto&b:d)if(b.alias_of){offsets[b.name]=offsets.at(b.alias_of);if(!sizes[b.name])sizes[b.name]=sizes.at(b.alias_of);}
 }
 void disjoint(std::initializer_list<const char*> group){for(auto a:group)for(auto b:group)if(std::string(a)!=b)
  assert(offsets[a]+sizes[a]<=offsets[b]||offsets[b]+sizes[b]<=offsets[a]);}
 void write(const std::string&n){value[n]=++stamp;std::fill(bytes.begin()+offsets[n],bytes.begin()+offsets[n]+sizes[n],stamp);}
 void read(const std::string&n){assert(value.count(n));const auto expected=value[n];assert(std::all_of(bytes.begin()+offsets[n],bytes.begin()+offsets[n]+sizes[n],[&](auto b){return b==expected;}));}
 void replay(){
  // Independent repeated chunks and every truncated decoder layer rebuild all
  // overwritten values. DeepStack runs only at the actual layers 0/1/2.
  for(int chunk=0;chunk<2;++chunk){write("residual1");for(int layer=0;layer<16;++layer){
   read("residual1");write("input_norm");
   for(auto n:{"q","k","v"}){read("input_norm");write(n);}
   read("q");write("output");read("k");write("input_norm");
   read("output");read("input_norm");write("q");write("k");read("k");read("v");
   read("q");write("sdpa_tmp");write("output");read("q");read("sdpa_tmp");
   write("oproj");read("output");write("oproj");
   read("oproj");read("residual1");write("residual2");
   read("residual2");write("residual1");
   read("residual1");write("gate");read("gate");write("gate");
   read("residual1");write("up");read("up");read("gate");write("gate");
   read("gate");write("down");read("down");read("residual2");write("residual1");
   if(layer<3){write("deepstack_scratch");read("deepstack_scratch");read("residual1");write("residual1");}
  }read("residual1");}
 }
};
}
int main(){using namespace v3;CausalDecoderModel m;constexpr int64_t budget=8303616-1586688-65536;
 // Include both 224/256 two/three-image T128..256 profiles, including P238
 // below the alias threshold and the legacy P270/P289 compatibility cases.
 for(int p:{238,268,270,289,300,302,321,332,334,353,364,366,385,396,398,417,430,462})for(bool rounded:{false,true}){
  LayoutContext l;l.chunk_size=rounded?Align(p,16):p;l.max_kv_seq_len=p;l.num_layers=16;l.is_causal=true;
  auto baseline=m.causal_baseline_buffer_declarations(l),candidate=m.declare_buffers(l);
  assert(changed(candidate)==(l.chunk_size>256));assert(detail::estimate_temporary_total(candidate)<=budget);
  const auto before=detail::estimate_temporary_total(baseline),after=detail::estimate_temporary_total(candidate);
  assert(l.chunk_size>256?after<before:after==before);
  assert(sdpa_is_valid_chunk_size(m.make_sdpa_config(1),Align(p,16),p,0));
  l.physical_manifest_fingerprint=123;auto complete=m.declare_buffers(l);
  assert(detail::estimate_temporary_total(complete)==detail::estimate_temporary_total(candidate));
  Storage s(candidate),c(complete);assert(s.offsets==c.offsets);
  s.disjoint({"residual1","input_norm","q","k","v"});
  s.disjoint({"residual1","input_norm","q","k","v","output"});
  s.disjoint({"residual1","q","output","sdpa_tmp"});
  s.disjoint({"residual1","output","oproj"});
  s.disjoint({"residual1","oproj","residual2"});
  s.disjoint({"residual1","residual2","gate","up"});
  s.disjoint({"residual1","residual2","gate","down"});
  if(l.chunk_size>256)assert(s.offsets["down"]==s.offsets["deepstack_scratch"]);
  if(p==417||p==462)s.replay();
  if(p==462)assert(int64_t(s.bytes.size())==(rounded?6464512:6436864));
 }
 // The actual SDPA legality rejects all smaller chunks at logical P462.
 for(int c=16;c<=464;c+=16){const int tail=462%c;
  const bool valid=!(462>c&&tail>0&&tail<16)&&sdpa_is_valid_chunk_size(m.make_sdpa_config(1),c,462,0);
  assert(valid==(c==464));}
 LayoutContext l;l.chunk_size=464;l.max_kv_seq_len=462;l.num_layers=16;l.is_causal=true;
 // max_kv_seq_len is deliberately not an alias selector: no-mask generic
 // allocation keys omit it. Same chunk/profile must retain the same layout.
 auto same=m.declare_buffers(l);l.max_kv_seq_len=480;assert(Storage(same).offsets==Storage(m.declare_buffers(l)).offsets);
 for(int fault=0;fault<23;++fault){auto x=m;auto bad=l;switch(fault){
  case 0:x.layers=28;break;case 1:x.h=2560;break;case 2:x.cores=4;break;
  case 3:x.fuse_lm_head_=true;break;case 4:x.decode_gate_up_fusion_=true;break;
  case 5:x.qwen3vl_pooler_z1_dispatch_=true;break;case 6:x.qwen3vl_pooler_z1_prepared_execution_len_=128;break;
  case 7:x.qwen3vl_pooler_z1_retained_deepstack_count_=3;break;
  case 8:x.qwen3vl_multiview_text_dry_prepared_=true;break;case 9:x.qwen3vl_multiview_text_composite_prepared_=true;break;
  case 10:x.qwen3vl_multiview_text_composite_dispatch_=true;break;case 11:x.planning_mode_=CausalDecoderPlanningMode::Z1;break;
  case 12:x.deepstack_lang_layers_={0};break;case 13:x.mrope_section_={16,24,24};break;
  case 14:x.nvfp4_=true;break;case 15:x.adarms_=true;break;case 16:x.decoder_mixed_precision_=true;break;
  case 17:bad.is_causal=false;break;case 18:bad.use_attn_mask=true;break;case 19:bad.batch_size=2;break;
  case 20:bad.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;break;
  case 21:bad.chunk_size=256;break;case 22:bad.chunk_size=480;break;}
  assert(!changed(x.declare_buffers(bad)));
 }
 // The actual baseline preserves every declaration; only four Temp slots change.
 auto baseline=m.causal_baseline_buffer_declarations(l),candidate=m.declare_buffers(l);
 for(size_t i=0;i<baseline.size();++i){const auto&a=baseline[i];const auto&b=candidate[i];
  if(std::string(a.name)=="q"||std::string(a.name)=="sdpa_tmp"||std::string(a.name)=="down"||std::string(a.name)=="deepstack_scratch")continue;
  assert(a.size==b.size&&a.phase_start==b.phase_start&&a.phase_end==b.phase_end&&a.alias_of==b.alias_of&&a.scope==b.scope);}
 std::cout<<"PASS C464=6464512 P462=6436864\n";
}
'''
    cpp, exe = tmp_path / "text_layout.cpp", tmp_path / "text_layout"
    cpp.write_text(program)
    result = subprocess.run([shutil.which("g++"), "-std=c++17", "-O1", str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout


def test_alias_lifetimes_follow_actual_native_stream_and_layout_identity():
    source = HEADER.read_text()
    body = _definition(source, "void build_layer_subgraph(")
    # The last ordinary SDPA branch completes before O-proj preparation/writes.
    sdpa = body.rindex("rpu_launch_sdpa_spm_dispatch(")
    prepare = body.index("rpu_prepare_ring_all_reduce_input(", sdpa)
    oproj = body.index("rpu_launch_linear_spm_to_spm_acc16_kernel(", prepare)
    mlp = body.index("emit_mlp_pipeline(", oproj)
    deepstack = body.index("rpu_launch_ddr_broadcast_spm_dma_mutable(", mlp)
    add = body.index("rpu_launch_eltwise_binary_spm_kernel(", deepstack)
    assert sdpa < prepare < oproj < mlp < deepstack < add
    assert 'addr_offset("q")' in body[sdpa:prepare]
    assert 'addr_offset("sdpa_tmp")' in body[sdpa:prepare]
    assert 'addr(0, "output")' in body[oproj:mlp]
    assert 'addr(0, "q")' not in body[prepare:]
    assert 'addr_offset("q")' not in body[prepare:]
    assert 'addr(0, "deepstack_scratch")' in body[mlp:deepstack]
    mlp_body = _definition((ROOT / "src/core/fused_model_base.cpp").read_text(), "void FusedModelBase::emit_mlp_pipeline(")
    # Both ordinary MUL and fused SiLU/MUL consume Up before Down's first write;
    # the final ordinary residual ring consumes Down before returning to caller.
    up = mlp_body.index('up_w, addr(0, "up")')
    fused_mul = mlp_body.index("rpu_launch_silu_mul_spm_kernel(", up)
    mul = mlp_body.index("rpu_launch_eltwise_binary_spm_kernel(", fused_mul)
    down = mlp_body.index('down_w, addr(0, "down")', mul)
    ring = mlp_body.rindex("rpu_launch_all_reduce_sum_residual_kernel(")
    assert up < fused_mul < mul < down < ring
    assert 'addr(0, "down"), residual, addr(0, "residual1")' in mlp_body[ring:]
    identity = _definition(source, "int64_t subclass_layout_hash(")
    assert "gr00t_long_prefill_scratch_profile()" in identity
    alias = _definition(source, "void apply_gr00t_long_prefill_scratch_alias(")
    assert "max_kv_seq_len" not in alias
    baseline = _definition(source, "virtual std::vector<BufferDecl> explicit_mask_baseline_buffer_declarations(")
    assert "apply_gr00t_long_prefill_scratch_alias(declarations, ctx)" in baseline
