"""CPU execution of the bounded LingBot2 COMPLETE-marker policy transform."""
from pathlib import Path
import re
import shutil
import subprocess

from test_native_kvinsert_cost_domain import _body

ROOT = Path(__file__).resolve().parents[1]
OUTER = ROOT / 'src/fused/rpu_qwen3vl_multiview_spm_z2.cpp'


def test_actual_policy_transform_preserves_execution_rules_and_rejects_drift(tmp_path):
    native = OUTER.read_text()
    graph = (ROOT / 'src/graph/graph_runtime.h').read_text()
    body = _body(native, 'lingbot2_register_policy_with_manifest_branches')
    declarations = []
    for name in ('GraphKernelRegisterCensusStats', 'GraphKernelRegisterPhaseExpectation',
                 'GraphKernelRegisterPolicy'):
        declarations.append(re.search(r'struct ' + name + r' \{.*?\n\};', graph, re.S).group())
    kinds = re.search(r'enum class GraphNodeKind : uint8_t \{.*?\n\};', graph, re.S).group()
    fmb_header = (ROOT / 'src/core/fused_model_base.h').read_text()
    fmb_source = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    manifest_types = []
    for name in ('FmbPhysicalManifestState', 'FmbGraphLifecycle',
                 'FmbLinearAccumulationPolicy', 'FmbRouteFamily'):
        manifest_types.append(re.search(r'enum class ' + name + r' : int \{.*?\n\};', fmb_header, re.S).group())
    for name in ('FmbRouteManifestEntry', 'FmbPhysicalExecutionManifest'):
        manifest_types.append(re.search(r'struct ' + name + r' \{.*?\n\};', fmb_header, re.S).group())
    manifest_api = '\n'.join(manifest_types) + '\n'
    manifest_api += 'uint64_t mix_u64(uint64_t hash, uint64_t value) ' + _body(fmb_source, 'mix_u64')
    manifest_api += '\nuint64_t physical_manifest_fingerprint_impl(const FmbPhysicalExecutionManifest& manifest) ' + _body(fmb_source, 'physical_manifest_fingerprint_impl')
    manifest_api += '\nuint64_t fmb_physical_manifest_fingerprint(const FmbPhysicalExecutionManifest& manifest) ' + _body(fmb_source, 'fmb_physical_manifest_fingerprint')
    # Execute the actual bootstrap key expressions through the prepared
    # candidate's real accessor and the real manifest fingerprint function.
    prepared_fingerprint = _body(fmb_header, 'uint64_t manifest_fingerprint')
    key_begin = native.index('const uint64_t vision_key =', native.index('RpuKernelGraph bootstrap_graph'))
    key_end = native.index('s.manifest_register_policy.emplace(', key_begin)
    key_expressions = native[key_begin:key_end]
    harness = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
constexpr size_t kOccurrenceCount = 4;
// Opaque register rules are copied, never inspected/re-authored by this function.
using GraphKernelRegisterRule = std::vector<int>;
__DECLARATIONS__
__KINDS__
namespace v3 { __MANIFEST_API__ }
struct GraphSignature {
 int marker = 0;
 bool operator==(const GraphSignature& rhs) const { return marker == rhs.marker; }
};
struct BranchNodeData {uint64_t branch_key=0; GraphSignature branch_signature; std::string label;};
struct GraphNode {
 GraphNodeKind kind; BranchNodeData branch;
 const BranchNodeData& as_branch() const { return branch; }
};
GraphKernelRegisterPolicy lingbot2_register_policy_with_manifest_branches(
 const GraphKernelRegisterPolicy& baseline, const std::vector<GraphNode>& nodes,
 const std::array<uint64_t,kOccurrenceCount>& manifest_keys) __BODY__
uint64_t hash(const std::vector<GraphNode>& nodes, bool strip=false) {
 uint64_t h=14695981039346656037ULL;
 for(const auto& n:nodes) if(!strip || n.kind!=GraphNodeKind::Branch)
  h=(h^static_cast<uint8_t>(n.kind))*1099511628211ULL;
 return h;
}
template<class F> void rejects(F f) {
 bool threw=false;try{f();}catch(const std::runtime_error&){threw=true;}assert(threw);
}
int main() {
 const auto K=GraphNodeKind::Kernel,D=GraphNodeKind::Dma,B=GraphNodeKind::Barrier;
 const auto M=GraphNodeKind::Branch;
 const std::string label="FMB complete physical manifest";
 struct Candidate {v3::FmbPhysicalExecutionManifest physical_manifest;};
 Candidate vision,text;
 vision.physical_manifest.state=text.physical_manifest.state=v3::FmbPhysicalManifestState::COMPLETE;
 vision.physical_manifest.logical_length=vision.physical_manifest.physical_length=256;
 text.physical_manifest.logical_length=text.physical_manifest.physical_length=225;
 struct Prepared {
  uint64_t manifest_fingerprint_;
  explicit Prepared(const Candidate& candidate)
   : manifest_fingerprint_(v3::fmb_physical_manifest_fingerprint(candidate.physical_manifest)) {}
  uint64_t manifest_fingerprint() const __PREPARED_FINGERPRINT__
 };
 const Prepared vision_prepared(vision), text_prepared(text);
 const auto* prepared_vision=&vision_prepared;
 const auto* prepared_text=&text_prepared;
 __KEY_EXPRESSIONS__
 assert(vision_key!=0 && text_key!=0 && vision_key!=text_key);
 assert(vision_key==v3::fmb_physical_manifest_fingerprint(vision.physical_manifest));
 assert(text_key==v3::fmb_physical_manifest_fingerprint(text.physical_manifest));
 auto changed=text.physical_manifest;changed.physical_length++;
 assert(v3::fmb_physical_manifest_fingerprint(changed)!=text_key);
 std::vector<GraphNode> stream={
  {M,{vision_key,{},label}},{D,{}},{B,{}},{M,{vision_key,{},label}},{K,{}},
  {D,{}},{M,{vision_key,{},label}},{K,{}},{B,{}},{M,{text_key,{},label}}};
 const std::array<uint64_t,4> keys={vision_key,vision_key,vision_key,text_key};
 GraphKernelRegisterPolicy old;
 old.name="original";old.fingerprint=0x4c423257385a3242ULL;
 old.rules={{1,2,3},{4,8,20,21}};
 old.expected_node_count=6;old.expected_dma_count=2;old.expected_barrier_count=2;
 old.expected_kernel_count=2;old.expected_typed_kernel_count=1;old.expected_no_ddr_kernel_count=1;
 old.expected_operand_count=7;old.expected_weight_operand_count=2;
 old.expected_key_cache_operand_count=1;old.expected_value_cache_operand_count=1;
 old.expected_semantic_input_operand_count=3;old.expected_ordered_kernel_hash=123456;
 old.expected_ordered_node_kind_hash=hash(stream,true);
 GraphKernelRegisterCensusStats stats;
 stats.node_count=6;stats.dma_count=2;stats.barrier_count=2;stats.kernel_count=2;
 stats.typed_kernel_count=1;stats.no_ddr_kernel_count=1;stats.operand_count=7;
 stats.weight_operand_count=2;stats.key_cache_operand_count=1;stats.value_cache_operand_count=1;
 stats.semantic_input_operand_count=3;stats.ordered_kernel_hash=123456;
 stats.ordered_node_kind_hash=old.expected_ordered_node_kind_hash;
 old.phases.push_back({"outer",stats});
 const auto out=lingbot2_register_policy_with_manifest_branches(old,stream,keys);
 assert(out.name=="lingbot2_multiview_w8_n32k32_register_census_v1_manifest");
 assert(out.fingerprint==(old.fingerprint ^ 0x464d424d41524b34ULL) && out.fingerprint!=old.fingerprint);
 assert(out.expected_node_count==10 && old.expected_node_count==6);
 assert(out.expected_ordered_node_kind_hash==hash(stream));
 assert(out.phases.size()==1 && out.phases[0].name==old.phases[0].name);
 assert(out.phases[0].stats.node_count==10);
 assert(out.phases[0].stats.ordered_node_kind_hash==hash(stream));
 assert(out.rules==old.rules);
 // Every execution quantity/hash and corresponding phase field survives.
#define SAME(name) assert(out.expected_##name==old.expected_##name); \
 assert(out.phases[0].stats.name==old.phases[0].stats.name)
 SAME(dma_count);SAME(barrier_count);SAME(kernel_count);SAME(typed_kernel_count);
 SAME(no_ddr_kernel_count);SAME(operand_count);SAME(weight_operand_count);
 SAME(key_cache_operand_count);SAME(value_cache_operand_count);
 SAME(semantic_input_operand_count);SAME(ordered_kernel_hash);
#undef SAME
 auto check=[&](const std::vector<GraphNode>& nodes) {
  lingbot2_register_policy_with_manifest_branches(old,nodes,keys);
 };
 auto bad=stream;bad[0].branch.branch_key=33;rejects([&]{check(bad);});
 bad=stream;std::swap(bad[0].branch,bad[9].branch);rejects([&]{check(bad);});
 bad=stream;bad[0].branch.label="foreign";rejects([&]{check(bad);});
 bad=stream;bad[0].branch.branch_signature.marker=1;rejects([&]{check(bad);});
 bad=stream;bad.push_back(stream[0]);rejects([&]{check(bad);});
 bad=stream;bad.push_back({D,{}});rejects([&]{check(bad);});
 bad=stream;bad.erase(bad.begin());rejects([&]{check(bad);});
 bad=stream;bad[4].kind=D;rejects([&]{check(bad);});
 bad=stream;std::swap(bad[1],bad[2]);rejects([&]{check(bad);});
 bad=stream;bad[1].kind=GraphNodeKind::HostCallback;rejects([&]{check(bad);});
 auto no_key=keys;no_key[0]=0;
 rejects([&]{lingbot2_register_policy_with_manifest_branches(old,stream,no_key);});
 auto bad_policy=old;bad_policy.phases[0].stats.dma_count++;
 rejects([&]{lingbot2_register_policy_with_manifest_branches(bad_policy,stream,keys);});
 assert(old.name=="original" && old.phases[0].stats.node_count==6);
}
'''.replace('__DECLARATIONS__', '\n'.join(declarations)).replace('__KINDS__', kinds).replace('__BODY__', body).replace('__MANIFEST_API__', manifest_api).replace('__KEY_EXPRESSIONS__', key_expressions).replace('__PREPARED_FINGERPRINT__', prepared_fingerprint)
    source, binary = tmp_path / 'manifest_policy.cpp', tmp_path / 'manifest_policy'
    source.write_text(harness)
    result = subprocess.run([shutil.which('g++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                             str(source), '-o', str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(binary)], check=True)


def test_bootstrap_seal_is_only_policy_source_and_retained_paths_require_it():
    native = OUTER.read_text()
    # A successful run_build returns only after Graph.end and sealed occurrence
    # coverage; the bootstrap object still exists when its exact nodes are read.
    build = _body(native, 'run_build')
    assert build.index('graph.end();') < build.index('coordinator.seal_build(graph)')
    start = native.index('RpuKernelGraph bootstrap_graph(graph_runtime_policy);')
    end = native.index('// Sealing before B', start)
    bootstrap = native[start:end]
    assert bootstrap.index('run_build(') < bootstrap.index('bootstrap_graph.replay_plan()')
    assert bootstrap.index('lingbot2_register_policy_with_manifest_branches(') < bootstrap.index('++s.bootstrap_build_count;')
    assert 'lingbot2_multiview_w8_register_policy(), *plan.steps' in bootstrap
    assert bootstrap.count('.prepare_stage_candidate(') == 2
    assert 'vision_key = prepared_vision->manifest_fingerprint();' in bootstrap
    assert 'text_key = prepared_text->manifest_fingerprint();' in bootstrap
    assert '{vision_key, vision_key, vision_key, text_key}' in bootstrap
    assert '.physical_manifest.fingerprint' not in bootstrap
    for path in (build, _body(native, 'run_replay')):
        assert 's.manifest_register_policy.has_value()' in path
        assert '*s.manifest_register_policy' in path
    assert 's.manifest_register_policy.reset();' in _body(native, 'reset_identity')
    assert native.count('manifest_register_policy.emplace(') == 1
