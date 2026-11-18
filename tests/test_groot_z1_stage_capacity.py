"""Real prepare, forward policy scope and decoder/SDPA capability, without a device."""
from pathlib import Path
import shutil
import subprocess

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def _decoder_forward(source):
    # The default IntArrayRef {} precedes the function body; it is not a
    # definition brace. Normalize only that default for the shared extractor.
    source = source.replace("planned_stage_descriptor = {},", "planned_stage_descriptor = at::IntArrayRef(),")
    return _definition(source, "at::Tensor forward(")


def test_z1_prepare_preserves_auto_and_explicit_override_authority(tmp_path):
    model = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    source = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.h").read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    core = (ROOT / "src/core/fused_model_base.cpp").read_text()
    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>
#include <utility>
namespace c10 {
template<class F> struct ScopeExit { F fn; ~ScopeExit(){fn();} };
template<class F> ScopeExit<F> make_scope_exit(F fn){return {std::move(fn)};}
}
#define TORCH_CHECK(c, ...) do{if(!(c))throw std::runtime_error("guard");}while(0)
#define SDPA_MAX_VLM_SIZE 400
#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
template<class T>T CeilDiv(T a,T b){return(a+b-1)/b;}
'''
    for marker in ("struct SdpaTiling", "enum class SdpaKernelType", "struct SdpaConfig"):
        cpp += _definition(helpers, marker) + ";\n"
    for marker in (
        "inline int64_t sdpa_tile_m_v16_initial(", "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline std::optional<int64_t> choose_tile_k_aggressive(",
        "inline std::optional<int64_t> choose_tile_k_conservative(",
        "inline std::optional<int64_t> choose_tile_k(", "inline bool sdpa_try_compute_tiling(",
        "inline SdpaTiling sdpa_compute_tiling(", "inline int64_t sdpa_compute_tmp_v16_size(",
        "static inline bool sdpa_call_valid(", "inline bool sdpa_is_valid_chunk_size(",
    ):
        cpp += _definition(helpers, marker) + "\n"
    cpp += _definition(core, "static int64_t legacy_chunk_override_for_exact_request(") + "\n"
    cpp += r'''
namespace at {using IntArrayRef=const std::vector<int64_t>&;}
struct RpuKernelGraph {static bool has_active(){return false;}};
using SpmPipelineComponentLayout=int;
struct ChunkEnvelope {int64_t chunk=0;};
class CausalDecoderModel {public:
 static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN=128;
 static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_REAL_LEN=72;
 static constexpr int64_t QWEN3VL_POOLER_Z1_GROOT_REAL_LEN=82;
 static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE=128;
 bool qwen3vl_multiview_text_dry_prepared_=false,qwen3vl_multiview_text_composite_prepared_=false;
 bool qwen3vl_pooler_z1_active_=false,qwen3vl_pooler_z1_bound_=false,qwen3vl_pooler_z1_dispatch_=false;
 int64_t qwen3vl_pooler_z1_prepared_execution_len_=0,qwen3vl_pooler_z1_retained_deepstack_count_=0;
 int64_t wall_z1_prepared_execution_len_=0;bool wall_z1_active_=false;
 bool wall_z1_dispatch_=false,qwen3vl_multiview_text_composite_dispatch_=false;
 int64_t wall_z1_prepared_chunk_size_=96,qwen3vl_pooler_z1_prepared_chunk_size_=128;
 int override_writes=0;
 bool qwen3vl_pooler_z1_retained_deepstack_bindings_empty()const{return true;}
 // Tensor/profile validation and allocation are independent of this policy bug.
 void validate_qwen3vl_pooler_z1_model_profile(int64_t,int64_t)const{}
 int64_t legacy_override=0,configured_chunk_size_cap_=0;
 bool equal_two_prefill_=false,decoder_mixed_precision_=false;
 bool has_qk_norm_=true,has_qkv_bias_=false;
 struct Context{std::optional<int> attention_mask;bool is_causal=true;} context;
 const Context&ctx()const{return context;}
 ChunkEnvelope envelope;
 const ChunkEnvelope&chunk_envelope()const{return envelope;}
 int64_t get_chunk_size_override()const{return legacy_override;}
 void set_chunk_size_override(int64_t value){legacy_override=value;++override_writes;}
 int64_t hidden_size()const{return 2048;}int64_t intermediate_size()const{return 6144;}
 int64_t num_q_heads()const{return 16;}int64_t num_kv_heads()const{return 8;}
 int64_t head_dim()const{return 128;}int attn_tp()const{return 8;}
 SdpaKernelType sdpa_kernel_=SdpaKernelType::FLASH_ATTN_SPM;
'''
    for text, marker in (
        (base, "bool chunk_within_envelope("),
        (model, "SdpaConfig make_sdpa_config("),
        (model, "bool subclass_chunk_size_valid("),
    ):
        cpp += _definition(text, marker).replace(" override", "") + "\n"
    prepare = _definition(source, "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(")
    # Execute the real preflight and all policy writes before SPM allocation.
    # No device tensors/allocator are needed to reproduce the original reject.
    prefix = prepare[:prepare.index("    LayoutContext layout;")]
    cpp += "SpmPipelineComponentLayout " + prefix.replace("CausalDecoderModel::", "").replace("bool partial_mrope)", "bool partial_mrope = false)") + "(void)partial_mrope; return 0;}\n"
    # Keep the actual forward scope alive while executing the real FMB exact
    # checks. Calling the validity predicate only after scope exit missed the
    # header's temporary C128 write in the original regression test.
    forward = _decoder_forward(model)
    start = forward.index("const int64_t saved_chunk_size_override =")
    end = forward.index("// R-Phase 1 (Qwen3-VL): M-RoPE forward-entry handling", start)
    cpp += "template<class Body> void forward_scope(Body body){\n" + forward[start:end] + "\nbody();}\n"
    exact = _definition(core, "at::Tensor FusedModelBase::run_all_layers_impl(")
    start = exact.index("const int64_t effective_legacy_override =")
    end = exact.index("// Validate only the exact immutable winner.", start)
    override_check = exact[start:end].replace("pimpl_->chunk_size_override_", "legacy_override")
    start = exact.index("TORCH_CHECK(\n            valid_fn(candidate.compute_chunk_size)", end)
    end = exact.index(";", start) + 1
    cpp += "void check_exact_stage(){\n" + r'''
 const int64_t seq_len=128;
 struct {int64_t compute_chunk_size=128;} candidate;
 auto valid_fn=[this](int64_t cs){return subclass_chunk_size_valid(cs,128,0);};
''' + override_check + exact[start:end] + "\n}\n};\n"
    cpp += r'''
int main(){
 CausalDecoderModel m;
 assert(sdpa_is_valid_chunk_size(m.make_sdpa_config(1),128,128,0));
 assert(m.subclass_chunk_size_valid(128,128,0));
 m.prepare_qwen3vl_pooler_z1_layout(128,82,3,{});
 assert(m.get_chunk_size_override()==0);
 assert(m.subclass_chunk_size_valid(128,128,0));
 m.qwen3vl_pooler_z1_dispatch_=true;
 m.forward_scope([&]{assert(m.get_chunk_size_override()==0);m.check_exact_stage();});
 assert(m.get_chunk_size_override()==0&&m.override_writes==0);
 bool failed=false;
 try{m.forward_scope([&]{m.check_exact_stage();throw std::runtime_error("downstream");});}
 catch(const std::runtime_error&){failed=true;}
 assert(failed&&m.get_chunk_size_override()==0&&m.override_writes==0);
 // AUTO-only public override rejection survives; private prepare never clears
 // or replaces an explicitly selected policy, even when it conflicts.
 for(int cs:{16,64,128,256}){
  CausalDecoderModel x;x.legacy_override=cs;
  x.prepare_qwen3vl_pooler_z1_layout(128,82,3,{});
  assert(x.get_chunk_size_override()==cs);
  assert(!x.subclass_chunk_size_valid(128,128,0));
  x.qwen3vl_pooler_z1_dispatch_=true;
  bool failed=false;
  try{x.forward_scope([&]{assert(x.get_chunk_size_override()==cs);x.check_exact_stage();});}
  catch(const std::runtime_error&){failed=true;}
  assert(failed&&x.get_chunk_size_override()==cs&&x.override_writes==0);
 }
 // Wall and multiview retain their existing scoped write AND restoration
 // on success/exception. Ordinary forwarding never writes the saved policy.
 for(int mode:{0,1,2}) for(bool fail:{false,true}){
  CausalDecoderModel x;x.legacy_override=64;
  x.wall_z1_dispatch_=mode==1;
  x.qwen3vl_multiview_text_composite_dispatch_=mode==2;
  const int64_t expected=mode==1?96:mode==2?0:64;
  bool failed=false;
  try{x.forward_scope([&]{assert(x.get_chunk_size_override()==expected);
       if(fail)throw std::runtime_error("downstream");});}
  catch(const std::runtime_error&){failed=true;}
  assert(failed==fail&&x.get_chunk_size_override()==64);
  assert(x.override_writes==(mode==0?0:2));
 }
 CausalDecoderModel ordinary;ordinary.envelope.chunk=320;
 ordinary.prepare_qwen3vl_pooler_z1_layout(128,72,0,{});
 assert(ordinary.get_chunk_size_override()==0&&ordinary.subclass_chunk_size_valid(128,128,0));
 assert(!ordinary.subclass_chunk_size_valid(336,336,0));
 assert(!m.subclass_chunk_size_valid(15,128,0));
 assert(!m.subclass_chunk_size_valid(128,128,1));
 assert(!m.subclass_chunk_size_valid(128,513,0));
 m.configured_chunk_size_cap_=64;assert(!m.subclass_chunk_size_valid(128,128,0));
 bool rejected=false;try{CausalDecoderModel bad;bad.prepare_qwen3vl_pooler_z1_layout(128,148,3,{});}catch(...){rejected=true;}assert(rejected);
}
'''
    code, exe = tmp_path / "z1_capacity.cpp", tmp_path / "z1_capacity"
    code.write_text(cpp)
    subprocess.run([shutil.which("g++") or "g++", "-std=c++17", "-O0", str(code), "-o", str(exe)], check=True, capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)


def test_z1_forward_and_retirement_keep_descriptor_authority():
    source = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text()
    prepare = _definition(source, "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(")
    descriptor = _definition(source, "std::vector<int64_t> CausalDecoderModel::qwen3vl_pooler_z1_stage_descriptor(")
    prime = _definition(source, "void CausalDecoderModel::prime_qwen3vl_pooler_z1_inputs(")
    forward = _definition(source, "at::Tensor CausalDecoderModel::forward_qwen3vl_pooler_z1(")
    unprepare = _definition(source, "void CausalDecoderModel::unprepare_qwen3vl_pooler_z1_layout()")
    header = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    dispatch = _decoder_forward(header)
    assert "planned_chunk_size == 0 && !planned_stage_descriptor.empty()" in dispatch
    assert "planned_chunk_size,\n                                    planned_stage_descriptor)" in dispatch
    assert "candidate.compute_chunk_size == 128" in prepare
    assert "FmbGraphLifecycle::COMPOSITE_CHILD" in prepare
    assert "encode_causal_physical_stage_descriptor(" in descriptor
    assert "qwen3vl_pooler_z1_stage_descriptor_ = stage_descriptor" in prime
    assert "/*planned_chunk_size=*/0," in forward
    assert "qwen3vl_pooler_z1_stage_descriptor_);" in forward
    assert "get_last_resolved_chunk_size() ==" in forward
    for body in (prepare, prime, forward, unprepare):
        assert "set_chunk_size_override(" not in body
    assert "qwen3vl_pooler_z1_prepared_chunk_size_ = 0" in unprepare
