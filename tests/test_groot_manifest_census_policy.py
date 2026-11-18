"""Host-only execution of GR00T's exact COMPLETE metadata quota and checkpoints."""
import ast
from pathlib import Path
import re
import shutil
import subprocess

from test_siglip_attention_dry_live_contract import _function

ROOT = Path(__file__).resolve().parents[1]
POOLER = 'src/fused/rpu_qwen3vl_pooler_spm_z1.cpp'


def _frozen_policy_source():
    outer = (ROOT / POOLER).read_text()
    wrapper = _function(outer, 'const GraphKernelRegisterPolicy& groot_n17_pooler_ds3_register_policy()')
    assert 'return rpu_pl_census::policy("groot_n17_pooler_ds3_register_policy");' in wrapper
    generated = (ROOT / 'src/fused/rpu_parallel_linear_register_policies.h').read_text()
    return _function(generated, 'inline const GraphKernelRegisterPolicy& groot_n17_pooler_ds3_register_policy()')


def test_groot_complete_metadata_quota_and_order_fail_closed(tmp_path):
    outer = (ROOT / POOLER).read_text()
    header = (ROOT / 'src/graph/graph_runtime.h').read_text()
    census = (ROOT / 'src/graph/graph_register_census.cpp').read_text()
    policy = _frozen_policy_source()
    stats = policy[policy.index('    p.expected_node_count ='):policy.index('    return p;')].replace('p.', 'value.')
    declarations = '\n'.join(_function(header, marker) + ';' for marker in (
        'struct GraphKernelRegisterCensusStats {',
        'struct GraphKernelRegisterPhaseExpectation {',
        'struct GraphKernelRegisterPolicy {',
        'enum class GraphNodeKind :',
    ))
    helpers = '\n'.join(_function(census, marker) for marker in (
        'uint64_t fnv_byte(', 'uint64_t fnv_u64(', 'uint64_t fnv_string(',
        'GraphKernelRegisterCensusStats stats_delta(', 'void check_stats_counts(',
    ))
    methods = '\n'.join(_function(census, marker).replace('RpuKernelGraph::', 'Probe::') for marker in (
        'void RpuKernelGraph::observe_physical_manifest_branch_for_census(',
        'void RpuKernelGraph::observe_kernel_register_node(',
        'void RpuKernelGraph::checkpoint_kernel_register_census(',
    ))
    source = r'''
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(c, ...) do { if(!(c)) throw std::runtime_error("rejected"); } while(0)
#define TORCH_INTERNAL_ASSERT(c, ...) TORCH_CHECK(c)
constexpr uint64_t kFnvOffset=14695981039346656037ULL,kFnvPrime=1099511628211ULL;
using GraphKernelRegisterRule=int;
''' + declarations + '\n' + helpers + r'''
GraphKernelRegisterPolicy policy() {
 GraphKernelRegisterPolicy value;
''' + stats + r'''
 return value;
}
struct Probe {
 bool kernel_register_census_active_=true,kernel_register_census_complete_=false;
 bool kernel_register_census_poisoned_=false;
 std::optional<int> pending_kernel_register_census_;
 size_t kernel_register_census_next_phase_=0;
 GraphKernelRegisterPolicy kernel_register_census_policy_=policy();
 GraphKernelRegisterCensusStats kernel_register_census_stats_,kernel_register_census_phase_begin_stats_;
 Probe(size_t phase) :kernel_register_census_next_phase_(phase) {
  kernel_register_census_stats_.ordered_node_kind_hash=kFnvOffset;
  kernel_register_census_stats_.ordered_kernel_hash=kFnvOffset;
  kernel_register_census_phase_begin_stats_=kernel_register_census_stats_;
 }
 void observe_physical_manifest_branch_for_census();
 void observe_kernel_register_node(GraphNodeKind,const std::string*);
 void checkpoint_kernel_register_census(const std::string&);
 void execution(GraphNodeKind kind) {
  static const std::string name="fixture";
  if(kind==GraphNodeKind::Kernel) ++kernel_register_census_stats_.kernel_count;
  observe_kernel_register_node(kind,&name);
 }
 void text_body() {
  // Frozen C128/16-layer text stream: 4 broadcasts per layer and final norm;
  // each broadcast has 7 barriers, 8 DMA nodes and 7 barriers. Then 324
  // kernels (including retained DeepStack/final norm), followed by output DMA.
  for(int repeat=0;repeat<65;++repeat) {
   for(int i=0;i<7;++i) execution(GraphNodeKind::Barrier);
   for(int i=0;i<8;++i) execution(GraphNodeKind::Dma);
   for(int i=0;i<7;++i) execution(GraphNodeKind::Barrier);
  }
  for(int i=0;i<324;++i) execution(GraphNodeKind::Kernel);
  execution(GraphNodeKind::Dma);
  // These tests isolate metadata. Existing register/operand partitions and
  // kernel-name identity are separately frozen by the production policy.
  const auto& e=kernel_register_census_policy_.phases[2].stats;
  auto& s=kernel_register_census_stats_;
  s.typed_kernel_count=e.typed_kernel_count;s.no_ddr_kernel_count=e.no_ddr_kernel_count;
  s.operand_count=e.operand_count;s.weight_operand_count=e.weight_operand_count;
  s.key_cache_operand_count=e.key_cache_operand_count;s.value_cache_operand_count=e.value_cache_operand_count;
  s.semantic_input_operand_count=e.semantic_input_operand_count;
  kernel_register_census_phase_begin_stats_.ordered_kernel_hash=e.ordered_kernel_hash;
 }
};
''' + methods + r'''
template<class F> void rejects(F f) {bool hit=false;try{f();}catch(const std::runtime_error&){hit=true;}assert(hit);}
int main() {
 const auto p=policy();
 assert(p.expected_node_count==8983);
 assert(p.expected_node_count-p.expected_kernel_count-p.expected_dma_count-p.expected_barrier_count==2);
 assert(p.expected_ordered_node_kind_hash==0xcdb6ff6477aff07fULL);
 for(size_t phase=0;phase<3;++phase) {
  const auto& s=p.phases[phase].stats;
  assert(s.node_count-s.kernel_count-s.dma_count-s.barrier_count==(phase==1?0:1));
  Probe probe(phase);
  if(phase!=1) probe.observe_physical_manifest_branch_for_census();
  const auto before=probe.kernel_register_census_stats_;
  rejects([&]{probe.observe_physical_manifest_branch_for_census();});
  assert(probe.kernel_register_census_stats_.node_count==before.node_count);
  assert(probe.kernel_register_census_stats_.ordered_node_kind_hash==before.ordered_node_kind_hash);
 }
 // Original zero metadata quota reproduces the production failure.
 {Probe q(0);q.kernel_register_census_policy_.expected_node_count-=2;
  --q.kernel_register_census_policy_.phases[0].stats.node_count;
  rejects([&]{q.observe_physical_manifest_branch_for_census();});}
 {Probe q(2);q.observe_physical_manifest_branch_for_census();q.text_body();
  assert(q.kernel_register_census_phase_begin_stats_.ordered_node_kind_hash==0xb6536f40e168a946ULL);
  q.checkpoint_kernel_register_census("text");assert(q.kernel_register_census_next_phase_==3);}
 // Missing metadata, same count in the wrong position, and a foreign phase
 // cannot be repaired by giving the total policy two slots.
 {Probe q(2);q.text_body();rejects([&]{q.checkpoint_kernel_register_census("text");});}
 {Probe q(2);q.text_body();q.observe_physical_manifest_branch_for_census();
  rejects([&]{q.checkpoint_kernel_register_census("text");});}
 {Probe q(2);q.observe_physical_manifest_branch_for_census();q.text_body();
  rejects([&]{q.checkpoint_kernel_register_census("vision");});}
 {Probe q(0);rejects([&]{q.observe_kernel_register_node(GraphNodeKind::Branch,nullptr);});}
 {Probe q(0);q.pending_kernel_register_census_=1;rejects([&]{q.observe_physical_manifest_branch_for_census();});}
 {Probe q(0);q.kernel_register_census_poisoned_=true;rejects([&]{q.observe_physical_manifest_branch_for_census();});}
 {Probe q(2);q.observe_physical_manifest_branch_for_census();q.text_body();
  ++q.kernel_register_census_stats_.dma_count;rejects([&]{q.checkpoint_kernel_register_census("text");});}
}
'''
    compiler = shutil.which('c++')
    assert compiler
    cpp, binary = tmp_path / 'groot_census.cpp', tmp_path / 'groot_census'
    cpp.write_text(source)
    result = subprocess.run([compiler,'-std=c++17','-O0',str(cpp),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_groot_policy_matches_production_canary_and_complete_emitter_chain():
    outer = (ROOT / POOLER).read_text()
    policy = _frozen_policy_source()
    expected = {
        'register_census_node_count': 8983,
        'register_census_dma_count': 3028,
        'register_census_barrier_count': 5166,
        'register_census_kernel_count': 787,
        'register_census_typed_kernel_count': 464,
        'register_census_no_ddr_kernel_count': 323,
        'register_census_policy_fingerprint': 0xF9482C3CB3B372D3,
        'register_census_operand_count': 568,
        'register_census_weight_operand_count': 328,
        'register_census_key_cache_operand_count': 80,
        'register_census_value_cache_operand_count': 80,
        'register_census_semantic_input_operand_count': 80,
        'register_census_ordered_node_kind_hash': 0xCDB6FF6477AFF07F,
        'register_census_ordered_kernel_hash': 0x4C4250F519B23648,
    }
    for name, field in [('fingerprint','policy_fingerprint'),('expected_node_count','node_count'),
                        ('expected_dma_count','dma_count'),('expected_barrier_count','barrier_count'),
                        ('expected_kernel_count','kernel_count'),
                        ('expected_ordered_node_kind_hash','ordered_node_kind_hash'),
                        ('expected_ordered_kernel_hash','ordered_kernel_hash')]:
        match = re.search(r'p\.'+name+r'\s*=\s*(?:UINT64_C\()?((?:0x)?[0-9a-fA-F]+)',policy)
        assert match
        assert int(match[1],0)==expected['register_census_'+field]
    pipeline = _function(outer,'at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward(')
    assert pipeline.count('::forward_vision_z1(')==1 and pipeline.count('::forward_text_z1(')==1
    assert pipeline.index('::forward_vision_z1(')<pipeline.index('checkpoint("vision")')<pipeline.index('checkpoint("handoff")')<pipeline.index('::forward_text_z1(')
    vision=(ROOT/'src/fused/rpu_qwen3vl_vision_model.cpp').read_text()
    text=(ROOT/'src/fused/rpu_qwen3_model.cpp').read_text()
    assert 'native_composite_stage_descriptor_' in _function(vision,'    void forward_pooler_z1(')
    assert 'planned_stage_descriptor);' in _function(vision,'    at::Tensor forward_impl(')
    assert 'qwen3vl_pooler_z1_stage_descriptor_);' in _function(text,'at::Tensor CausalDecoderModel::forward_qwen3vl_pooler_z1(')
    base=(ROOT/'src/core/fused_model_base.cpp').read_text()
    begin=base.index('    bool defer_physical_manifest_branch = false;')
    end=base.index('    validate_fmb_chunk_mode(static_cfg, dyn_cfg',begin)
    path=base[begin:end]
    assert path.count('physical_graph.record_physical_manifest_branch(')==1
    assert path.index('auto dyn_cfg = dynamic_config(plan);')<path.index('physical_graph.record_physical_manifest_branch(')
    assert 'has_complete_physical_manifest()' in path


def test_groot_exact_vector_norm_rule_and_current_head_order_fail_closed(tmp_path):
    """Execute the real policy/lookup; use the captured bounded Text grammar."""
    outer = (ROOT / POOLER).read_text()
    graph = (ROOT / 'src/graph/graph_runtime.h').read_text()
    census = (ROOT / 'src/graph/graph_register_census.cpp').read_text()
    cache = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    enum = re.search(r'enum class KernelId : uint16_t \{.*?\n\};', cache, re.S).group()
    names = re.search(r'static constexpr const char\* KERNEL_ID_NAMES\[\] = \{.*?\n\};', cache, re.S).group()
    enums = ('GraphDdrRegisterRole', 'GraphDdrRegisterAccess', 'GraphDdrRegisterEncoding',
             'GraphKernelRegisterRuleKind', 'GraphKernelNoDdrProof')
    structs = ('GraphDdrRegisterAbi', 'GraphKernelRegisterRule',
               'GraphKernelRegisterCensusStats', 'GraphKernelRegisterPhaseExpectation',
               'GraphKernelRegisterPolicy')
    declarations = '\n'.join(re.search(r'enum class '+n+r' : uint8_t \{.*?\n\};', graph, re.S).group() for n in enums)
    declarations += '\n'+'\n'.join(re.search(r'struct '+n+r' \{.*?\n\};', graph, re.S).group() for n in structs)
    generated = (ROOT / 'src/fused/rpu_parallel_linear_register_policies.h').read_text()
    helpers = _function(generated, 'inline GraphKernelRegisterRule rule(') + '\n' + _frozen_policy_source()
    helpers += '\n'+'\n'.join(_function(census, marker) for marker in (
        'uint64_t fnv_byte(', 'uint64_t fnv_u64(', 'uint64_t fnv_string(',
    ))
    lookup = _function(census, 'const GraphKernelRegisterRule& RpuKernelGraph::kernel_register_rule_for(\n        KernelId id) const').replace('RpuKernelGraph::', 'Probe::')
    source = r'''
#include <array>
#include <cstring>
#include <iterator>
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do { if(!(c)) throw std::runtime_error("rejected"); } while(0)
constexpr uint64_t kFnvOffset=14695981039346656037ULL,kFnvPrime=1099511628211ULL;
''' + enum + '\n' + names + '\n' + declarations + '\n' + helpers + r'''
struct Probe {
 GraphKernelRegisterPolicy kernel_register_census_policy_=groot_n17_pooler_ds3_register_policy();
 const GraphKernelRegisterRule& kernel_register_rule_for(KernelId id) const;
};
''' + lookup + r'''
template<class F> void rejects(F f) {bool hit=false;try{f();}catch(const std::runtime_error&){hit=true;}assert(hit);}
uint64_t text_hash(uint64_t hash, bool old_scalar=false, bool old_head_order=false) {
 // Exact capture: 16 20-kernel Text bodies, DeepStack after layers 0/1/2,
 // and one final norm. The native emitter performs both head norms before
 // emit_qk_rope_pair. This fixture contains no addresses or learned tolerance.
 using K=KernelId;
 const K norm=old_scalar?K::RMS_NORM_BF16_SPM:K::RMS_NORM_SPM_V32;
 std::array<K,20> layer={norm,
  K::PL_U_FP16_ACC16_M480N128K64,K::PL_U_FP16_ACC16_M1040N32K64,K::PL_U_FP16_ACC16_M1040N32K64,
  norm,norm,K::MROPE,K::MROPE,K::LLAMA_INSERT_KCACHE_V16,K::LLAMA_INSERT_VCACHE_V16,
  K::SDPA_FLASH_ATTN_SPM,K::PL_U_FP16_ACC16_M208N96K256,K::LLM_ALL_REDUCE_RESIDUAL_RING_PACED,
  norm,K::PL_U_FP16_ACC16_M240N80K256,K::UNARY_SPM,K::PL_U_FP16_ACC16_M240N80K256,
  K::BINARY_SAMESHAPE,K::PL_U_FP16_ACC16_M192N112K256,K::LLM_ALL_REDUCE_RESIDUAL_RING_PACED};
 if(old_head_order)std::swap(layer[5],layer[6]);
 for(int i=0;i<16;++i){
  for(const auto id:layer)hash=fnv_string(hash,KERNEL_ID_NAMES[static_cast<size_t>(id)]);
  if(i<3)hash=fnv_string(hash,KERNEL_ID_NAMES[static_cast<size_t>(K::BINARY_SAMESHAPE)]);
 }
 return fnv_string(hash,KERNEL_ID_NAMES[static_cast<size_t>(norm)]);
}
int main(){
 Probe p;
 const auto& policy=p.kernel_register_census_policy_;
 const auto& norm=p.kernel_register_rule_for(KernelId::RMS_NORM_SPM_V32);
 assert(norm.kind==GraphKernelRegisterRuleKind::IntrinsicNoDdr&&norm.operands.empty()&&norm.expected_count==65);
 assert(policy.rules.size()==26&&policy.expected_operand_count==568&&policy.expected_no_ddr_kernel_count==323);
 // Neither a different vector width nor the historical scalar route is admitted.
 rejects([&]{p.kernel_register_rule_for(KernelId::RMS_NORM_SPM_V16);});
 rejects([&]{p.kernel_register_rule_for(KernelId::RMS_NORM_BF16_SPM);});
 rejects([&]{p.kernel_register_rule_for(KernelId::PL_U_FP16_ACC16_M1584N32K32);});
 size_t linear_count=0,linear_rules=0;
 for(const auto& rule:policy.rules){
  if(!rule.kernel_id||*rule.kernel_id<KernelId::PL_AUTOTILE_BEGIN||
     *rule.kernel_id>=KernelId::GROUPED_PL_AUTOTILE_BEGIN)continue;
  const auto& a=rule.operands.at(0);
  assert(rule.kind==GraphKernelRegisterRuleKind::TypedDdr&&rule.operands.size()==1);
  assert(a.role==GraphDdrRegisterRole::ModelWeight&&a.access==GraphDdrRegisterAccess::Read);
  assert(a.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&a.reg_lo==2&&a.reg_hi==3);
  linear_count+=rule.expected_count;++linear_rules;
 }
 assert(linear_rules==11&&linear_count==264);
 const auto text=policy.phases.at(2).stats.ordered_kernel_hash;
 assert(text_hash(kFnvOffset)==text&&text==0x14e3319572e84b45ULL);
 assert(text_hash(kFnvOffset,true)!=text&&text_hash(kFnvOffset,false,true)!=text);
 assert(text_hash(kFnvOffset,true,true)==0x8039fbfd24b8f625ULL);
 auto prefix=policy.phases.at(0).stats.ordered_kernel_hash;
 prefix=fnv_string(prefix,"fill");prefix=fnv_string(prefix,"slice_single_axis");prefix=fnv_string(prefix,"slice_single_axis");
 assert(text_hash(prefix)==policy.expected_ordered_kernel_hash);
 assert(text_hash(prefix,true,true)==0x94f18c1f2c3fcb10ULL);
 for(const auto& rule:policy.rules){
  const auto name=rule.kernel_id?KERNEL_ID_NAMES[static_cast<size_t>(*rule.kernel_id)]:rule.kernel_name;
  std::cout<<name<<" "<<rule.expected_count<<"\n";
 }
 // Keeping the former policy's rule set reproduces the exact KernelId173 failure.
 for(auto& rule:p.kernel_register_census_policy_.rules)
  if(rule.kernel_id==KernelId::RMS_NORM_SPM_V32)rule.kernel_id=KernelId::RMS_NORM_BF16_SPM;
 rejects([&]{p.kernel_register_rule_for(KernelId::RMS_NORM_SPM_V32);});
}
'''
    compiler = shutil.which('c++')
    assert compiler
    cpp, binary = tmp_path/'groot_exact_rule.cpp', tmp_path/'groot_exact_rule'
    cpp.write_text(source)
    result = subprocess.run([compiler, '-std=c++17', '-O0', '-I', str(ROOT/'src/core'), '-I', str(ROOT/'src/ops'),
                             str(cpp), '-o', str(binary)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout+result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    expected = {
        'parallel_linear_m1040n32k64': 32,
        'parallel_linear_m544n48k128': 24,
        'parallel_linear_m480n64k128': 72,
        'parallel_linear_m432n80k128': 24,
        'parallel_linear_m240n80k256': 32,
        'parallel_linear_m96n80k512': 4,
        'parallel_linear_m208n96k256': 16,
        'parallel_linear_m80n96k512': 4,
        'parallel_linear_m192n112k256': 16,
        'parallel_linear_m480n128k64': 16,
        'parallel_linear_m336n128k128': 24,
        'llama_rms_norm_v32': 65,
        'rope_2d_spm': 48,
        'llama_mrope_interleave': 32,
        'llama_insert_kcache_multiwarp_v16': 40,
        'llama_insert_vcache_multiwarp_v16': 40,
        'llm_fp16_32b_prefill_flash_attn_univ_dp': 40,
        'llm_all_reduce_residual': 84,
        'memset_spm_multi_core_v2': 56,
        'layer_norm': 52,
        'binary_sameshape': 19,
        'unary': 16,
        'unary_gelu_tanh': 24,
        'unary_gelu_erf': 4,
        'fill': 1,
        'slice_single_axis': 2,
    }
    actual = {name: int(count) for name, count in (line.split() for line in result.stdout.splitlines())}
    assert actual == expected
    # Tie the captured grammar to the real ordinary head-norm emission branch.
    header = (ROOT/'src/fused/rpu_qwen3_model.h').read_text()
    start = header.index('} else if (has_qk_head_norm) {', header.index('emit_qwen3vl_4b_decode_qk_norm_mrope(layer_idx'))
    body = header[start:header.index('// Phase 4: KV cache insert', start)]
    assert body.index('norm_q_addr(layer_idx, 0)') < body.index('norm_k_addr(layer_idx, 0)') < body.index('emit_qk_rope_pair(')
