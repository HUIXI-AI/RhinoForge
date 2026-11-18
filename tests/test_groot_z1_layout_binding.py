"""Exercise Z1 descriptor layout binding against the actual FMB allocator."""
from pathlib import Path
import shutil
import subprocess

from test_fmb_standalone_layout_rebuild import _definition, _host_program

ROOT = Path(__file__).resolve().parents[1]


def _program(source, *, old_default=False):
    core = (ROOT / "src/core/fused_model_base.cpp").read_text()
    cpp = _host_program(core).split("int main(){", 1)[0]
    cpp = cpp.replace('throw std::runtime_error("native contract rejected");',
                      'throw std::runtime_error(#__VA_ARGS__);')
    cpp = cpp.replace("#include <map>", "#include <map>\n#include <memory>\n#include <optional>")
    cpp = cpp.replace("enum class FmbRopeTableResidency {UNSPECIFIED};",
                      "enum class FmbRopeTableResidency {UNSPECIFIED, DDR, SPM};")
    # Keep descriptor minting and tensor I/O outside this allocation-only test.
    # Distinct complete fixture manifests stand for the two native RoPE modes;
    # the model's binding, real params hash and real allocation paths are run.
    cpp += r'''
namespace at {using IntArrayRef=const std::vector<int64_t>&;}
struct RpuKernelGraph {static bool has_active(){return false;}};
struct RpuExecutionCoordinator {static void require_no_physical_claim(const char*){
 TORCH_CHECK(!arena.pipeline_arena_locked_);}};
enum class FmbPhysicalManifestState {COMPLETE};
enum class FmbGraphLifecycle {COMPOSITE_CHILD};
enum class CausalDecoderPlanningMode {QWEN3VL_POOLER_Z1};
enum class ChunkMode {SEQUENTIAL};
enum class FmbRouteFamily {ROPE};
constexpr int64_t CAUSAL_DECODER_PARTIAL_MROPE_TENSOR_SITE=22;
struct FmbThreeStageChunkPlan {
 struct {struct {int64_t chunk_size=128;} plan;} input;
};
FmbThreeStageChunkPlan compose_fmb_default_three_stage_chunk_plan(
 const LayoutContext&,int64_t,int64_t,ChunkMode){return {};}
uint64_t fmb_three_stage_chunk_plan_fingerprint(const FmbThreeStageChunkPlan&){return 55;}
struct FmbPhysicalExecutionManifest {
 FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
 FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
 int64_t logical_length=82,physical_length=128;
 struct Route{FmbRouteFamily family;int64_t site_id;};std::vector<Route> routes;
 uint64_t fingerprint=0;
 FmbRopeTableResidency rope=FmbRopeTableResidency::DDR;
 AttentionExecutionPolicy attention=AttentionExecutionPolicy::DDR_KV;
 FmbForwardOperandResidency operands=FmbForwardOperandResidency::PER_LAYER;
};
uint64_t fmb_physical_manifest_fingerprint(const FmbPhysicalExecutionManifest&m){return m.fingerprint;}
FmbRopeTableResidency fmb_rope_table_residency(const FmbPhysicalExecutionManifest&m){return m.rope;}
FmbForwardOperandResidency physical_forward_operand_residency(const FmbPhysicalExecutionManifest&m){return m.operands;}
struct FmbPhysicalManifestConsumer {
 const FmbPhysicalExecutionManifest&m;
 explicit FmbPhysicalManifestConsumer(const FmbPhysicalExecutionManifest&v):m(v){}
 std::optional<AttentionExecutionPolicy> attention_layout_policy()const{return m.attention;}
};
struct Prepared {
 struct Candidate {int64_t input_chunk_size=128,qkv_chunk_size=128,compute_chunk_size=128;
  FmbThreeStageChunkPlan stage_plan;FmbPhysicalExecutionManifest physical_manifest;} value;
 const Candidate&candidate()const{return value;}
 uint64_t stage_fingerprint()const{return 55;}
};
class Model {public:
 static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_CHUNK_SIZE=128;
 static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN=128;
 static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_REAL_LEN=72;
 static constexpr int64_t QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES=1966080;
 FusedModelBase::Impl state;
 bool qwen3vl_pooler_z1_dispatch_=false;
 int64_t qwen3vl_pooler_z1_prepared_real_len_=82,qwen3vl_pooler_z1_retained_deepstack_count_=3;
 std::vector<int64_t> qwen3vl_pooler_z1_selected_stage_descriptor_,qwen3vl_pooler_z1_layout_stage_descriptor_;
 LayoutContext prepared_context;int prepare_calls=0;
 int64_t num_layers()const{return 16;}
 void validate_qwen3vl_pooler_z1_contract()const{}
 void kvinsert_cost_domain(const char*,at::IntArrayRef d){TORCH_CHECK(d.size()==3);}
 std::vector<int64_t> kvinsert_cost_owner_prefix(at::IntArrayRef d)const{
  TORCH_CHECK(d.size()==3);return {INT64_C(0x51565a3154434f53),d[0],d[1],d[2]};}
 void bind_kvinsert_cost_owner_prefix(at::IntArrayRef d,at::IntArrayRef p){TORCH_CHECK(kvinsert_cost_owner_prefix(d)==p);}
 std::vector<int64_t> encode_causal_physical_stage_descriptor(const FmbThreeStageChunkPlan&,
  const LayoutContext&,int64_t,int64_t,int64_t,int64_t physical,int64_t logical,int64_t,int64_t mode,
  FmbGraphLifecycle,CausalDecoderPlanningMode){assert(physical==128);return {logical,3,mode};}
 std::shared_ptr<Prepared> prepare_stage_candidate(at::IntArrayRef d){
  TORCH_CHECK(d.size()==3);auto p=std::make_shared<Prepared>();auto&m=p->value.physical_manifest;
  m.logical_length=d[0];m.fingerprint=1000+d[0]*2+d[2];
  m.routes={{FmbRouteFamily::ROPE,d[2]?CAUSAL_DECODER_PARTIAL_MROPE_TENSOR_SITE:21}};
  m.rope=d[2]?FmbRopeTableResidency::SPM:FmbRopeTableResidency::DDR;return p;
 }
 std::vector<BufferDecl> declarations()const{
  return {{"scratch",QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES,0,8,StorageClass::Temp}};}
 SpmPipelineComponentLayout prepare_spm_pipeline_component(LayoutContext c,const FmbThreeStageChunkPlan&p){
  c.stage_plan_fingerprint=fmb_three_stage_chunk_plan_fingerprint(p);
  auto estimate=[](const LayoutContext&){return int64_t{0};};
  ensure_allocated_impl(state,declarations(),c,estimate,0,true,false);
  prepared_context=c;++prepare_calls;return actual(state,declarations());
 }
'''
    cpp += _definition(core, "LayoutContext FusedModelBase::bind_physical_layout_context(").replace("FusedModelBase::", "") + "\n"
    prepare = _definition(source, "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(")
    start = prepare.index("    LayoutContext layout;")
    end = prepare.index("    // declare_buffers()", start)
    binding = prepare[start:end]
    if old_default:
        binding = binding.replace("layout = bind_physical_layout_context(layout, manifest);",
                                  "if (!selected_descriptor.empty()) layout = bind_physical_layout_context(layout, manifest);")
    cpp += "void cold(int64_t execution_len,int64_t real_len,int64_t retained_deepstack_count,at::IntArrayRef selected_descriptor,bool partial_mrope){\n"
    cpp += "qwen3vl_pooler_z1_prepared_real_len_=real_len;\n" + binding
    cpp += "prepare_spm_pipeline_component(layout,stage_plan);qwen3vl_pooler_z1_layout_stage_descriptor_=prepared_descriptor;qwen3vl_pooler_z1_selected_stage_descriptor_=selected_descriptor;}\n"
    getter = _definition(source, "std::vector<int64_t> CausalDecoderModel::qwen3vl_pooler_z1_stage_descriptor(")
    cpp += getter.replace("CausalDecoderModel::", "") + "\n"
    prime = _definition(source, "void CausalDecoderModel::prime_qwen3vl_pooler_z1_inputs(")
    start = prime.index("    const auto stage_descriptor =")
    end = prime.index("    const uint64_t position_ids_addr =", start)
    cpp += "void prime(bool partial_mrope){\n" + prime[start:end] + "}\n"
    cpp += r'''
 void live(bool partial){
  auto p=prepare_stage_candidate(qwen3vl_pooler_z1_stage_descriptor(partial));
  LayoutContext c;c.chunk_size=128;c.max_kv_seq_len=128;c.num_layers=16;c.is_causal=true;
  c.stage_plan_fingerprint=p->stage_fingerprint();c=bind_physical_layout_context(c,p->candidate().physical_manifest);
  production_allocation(state,declarations(),c,0);
  assert(compute_params_hash_impl(c,8,8,64,1024,4096,0)==
         compute_params_hash_impl(prepared_context,8,8,64,1024,4096,0));
 }
};
void leased_replay(Model&m,bool partial){
 // Model the existing acquire/adopt bookkeeping; run the actual allocator
 // with its physical lock set, so any unexpected reset/allocation is rejected.
 arena.reset_temporary();arena.alloc_temporary(4718592);
 arena.pipeline_arena_locked_=true;arena.pipeline_arena_epoch_=1;
 m.state.pipeline_lease_epoch_=1;m.state.alloc_gen_=arena.generation();
 const auto generation=arena.generation();const auto end=arena.t_end_;
 m.live(partial);m.live(partial);
 assert(arena.generation()==generation&&arena.t_end_==end);
 arena.pipeline_arena_locked_=false;m.state.pipeline_lease_epoch_=0;
}
int main(){
 for(bool selected:{false,true}){
  arena=SpmAllocator{};Model m;
  m.cold(128,82,3,selected?std::vector<int64_t>{82,3,0}:std::vector<int64_t>{},false);
  m.prime(false);assert(m.prepare_calls==1);leased_replay(m,false);
  assert(m.prepared_context.physical_manifest_fingerprint!=0);
  assert(m.prepared_context.rope_table_residency==FmbRopeTableResidency::DDR);
  assert(m.prepared_context.forward_operand_residency==FmbForwardOperandResidency::PER_LAYER);
 }
 arena=SpmAllocator{};Model partial;partial.cold(128,72,3,{},false);
 const auto before=actual(partial.state,partial.declarations());
 partial.prime(true);const auto after=actual(partial.state,partial.declarations());
 assert(before.temporary_bytes==1966080&&after.temporary_bytes==before.temporary_bytes);
 assert(before.layout_hash!=after.layout_hash);assert(partial.prepare_calls==2);
 leased_replay(partial,true);partial.prime(true);assert(partial.prepare_calls==2);
 partial.prime(false);assert(partial.prepare_calls==3);leased_replay(partial,false);
 arena.pipeline_arena_locked_=true;const auto layout=partial.qwen3vl_pooler_z1_layout_stage_descriptor_;
 bool rejected=false;try{partial.prime(true);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&partial.prepare_calls==3&&partial.qwen3vl_pooler_z1_layout_stage_descriptor_==layout);
 arena.pipeline_arena_locked_=false;
 arena=SpmAllocator{};Model fixed;fixed.cold(128,72,3,{72,3,0},false);
 rejected=false;try{fixed.prime(true);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected&&fixed.prepare_calls==1);
 std::cout<<"default/selected layouts survive leased replay; partial rebinding stays cold and exact\n";
}
'''
    return cpp


def test_default_selected_and_partial_layouts_survive_leased_replay(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text()
    for old_default in (False, True):
        cpp, exe = tmp_path / f"layout_{old_default}.cpp", tmp_path / f"layout_{old_default}"
        cpp.write_text(_program(source, old_default=old_default))
        built = subprocess.run([shutil.which("c++") or "c++", "-std=c++17", "-O0", str(cpp), "-o", str(exe)], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        result = subprocess.run([str(exe)], capture_output=True, text=True)
        if old_default:
            assert result.returncode != 0, "old default binding unexpectedly admitted"
            # The current allocator rejects the stale binding before any reset.
            assert "s.pipeline_lease_epoch_ == 0 || (!params_changed && !gen_changed && !persistent_invalidated)" in result.stderr
        else:
            assert result.returncode == 0, result.stdout + result.stderr


def test_partial_prime_precedes_physical_acquire_and_parent_plan_uses_fixed_extent():
    source = (ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.cpp").read_text()
    begin = _definition(source, "int64_t rpu_qwen3vl_pooler_spm_z1_begin(")
    assert begin.index("prime_text_inputs(") < begin.index("s.text_stage_descriptor =") < begin.index("SpmPipelineLease::acquire_physical(") < begin.index("adopt_text(")
    prepare = _definition(source, "int64_t rpu_qwen3vl_pooler_spm_z1_prepare(")
    plan = prepare[prepare.index("const std::vector<v3::SpmScratchDecl> scratch ="):prepare.index("const size_t expected_peak")]
    assert "text_layout.temporary_bytes" in plan and "layout_hash" not in plan
    assert "scratch, retained," in plan and "persistent_top_bytes, SpmAllocator::SPM_PLANNING_BUDGET" in plan
    assert "s.plan_hash = plan.hash()" in prepare
    digest = _definition(source, "uint64_t physical_binding_digest(")
    assert "&s.vision_stage_descriptor, &s.text_stage_descriptor" in digest
    model = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text()
    unprepare = _definition(model, "void CausalDecoderModel::unprepare_qwen3vl_pooler_z1_layout()")
    rollback = _definition(model, "void CausalDecoderModel::rollback_qwen3vl_pooler_z1_inputs()")
    assert "qwen3vl_pooler_z1_layout_stage_descriptor_.clear()" in unprepare
    assert "qwen3vl_pooler_z1_layout_stage_descriptor_.clear()" not in rollback
