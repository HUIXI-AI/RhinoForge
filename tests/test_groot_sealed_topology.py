"""CPU execution of GR00T census completion and sealed full-Graph identity."""
from pathlib import Path
import re
import shutil
import subprocess

from test_siglip_attention_dry_live_contract import _function

ROOT = Path(__file__).resolve().parents[1]


def test_actual_census_complete_hash_and_model_seal_lifecycle(tmp_path):
    runtime = (ROOT/'src/graph/graph_runtime.cpp').read_text()
    census = (ROOT/'src/graph/graph_register_census.cpp').read_text()
    header = (ROOT/'src/graph/graph_runtime.h').read_text()
    model = (ROOT/'src/fused/rpu_qwen3vl_pooler_spm_z1.cpp').read_text()
    declarations = '\n'.join(re.search(r'struct '+name+r' \{.*?\n\};', header, re.S).group()
        for name in ('GraphKernelRegisterCensusStats', 'GraphKernelRegisterPhaseExpectation', 'GraphKernelRegisterPolicy'))
    hash_class = _function(runtime, 'class StableGraphHasher {')+';'
    complete = _function(census, 'void RpuKernelGraph::complete_kernel_register_census()')
    hash_method = _function(runtime, 'uint64_t RpuKernelGraph::topology_hash_for_window(')
    built_hash = _function(runtime, 'uint64_t RpuKernelGraph::build_topology_hash() const')
    helpers = '\n'.join(_function(census, marker) for marker in (
        'GraphKernelRegisterCensusStats stats_delta(', 'void check_stats_counts('))
    seal = _function(model, 'void seal_recorded_ranges_topology_locked(')
    # Execute the real publication block reached by complete(), including DMA
    # owner annotations. Tensor/owner resolution is the only sealing endpoint
    # replaced by a small prepared owner fixture; no SDK or Tensor is imported.
    commit = _function(census, 'void RpuKernelGraph::commit_kernel_register_outer_fast_prepared(')
    publish = commit[commit.index('    auto& prepared ='):commit.index('    boundary_flush_ptrs_.swap(')]
    node_hash = _function(runtime, 'void hash_graph_node_topology(')
    dma_case = node_hash[node_hash.index('    case GraphNodeKind::Dma: {'):node_hash.index('    case GraphNodeKind::Barrier: {')]
    source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do{if(!(c))throw std::runtime_error("rejected");}while(0)
#define TORCH_INTERNAL_ASSERT(c, ...) TORCH_CHECK(c)
struct GraphKernelRegisterRule{size_t expected_count=0;};
'''+declarations+'\n'+hash_class+'\n'+helpers+r'''
enum class GraphNodeKind:uint8_t{Kernel=0,Branch=5,Dma=8,Barrier=9};
struct Dma{
 uint64_t bytes=32,semantic_endpoint_id=0,semantic_spm_peer_id=0;
 bool semantic_canonical_member_emission=false,outer_fast_owner_bound=false;
 uint8_t variant=1,channel=0,outer_fast_ddr_side=1;
 int64_t live_offset=0;size_t outer_fast_owner_ordinal=0,outer_fast_owner_byte_offset=0;
 uint64_t outer_fast_owner_topology_id=0;
};
struct Kernel{std::optional<int>register_census=1;};
struct Node{GraphNodeKind kind;Dma dma;Kernel kernel;
 const Kernel&as_kernel()const{return kernel;}const Dma&as_dma()const{return dma;}};
void hash_graph_node_topology(StableGraphHasher&hash,const Node&node,uint8_t dma_topology_schema,bool outer_fast_schema){
 hash.u8(static_cast<uint8_t>(node.kind));
 switch(node.kind){
'''+dma_case+r'''
 default:break;
 }
}
struct Stamp{size_t position=0;uint64_t generation=1,signature=7;};
struct Digest{size_t begin=0,end=0,node_count=0;uint64_t kind_mask=0,topology_hash=0;std::vector<GraphNodeKind>node_kinds;};
struct Build{bool staged=false;uint64_t physical_digest=0,topology_digest=0;};
struct Annotation{Dma*node;uint8_t side=1;size_t owner_ordinal=0;uint64_t owner_topology_id=17;size_t owner_byte_offset=0;};
struct Prepared{std::optional<Build>build_state;std::vector<Annotation>node_annotations;};
struct RpuKernelGraph{
 enum class State{PASSTHROUGH,RECORDING,BUILT,REPLAYING};
 State state_=State::RECORDING;bool replayable_=true,has_built_signature_=true;
 mutable uint64_t build_topology_hash_=0;uint64_t generation=1,signature=7;
 std::vector<Node>nodes_;
 bool kernel_register_census_active_=true,kernel_register_census_complete_=false,kernel_register_census_poisoned_=false;
 std::optional<int>pending_kernel_register_census_;
 size_t kernel_register_census_next_phase_=2,kernel_register_census_node_begin_=0,cursor_=0;
 size_t kernel_register_host_reemitted_node_count_=0,kernel_register_host_reemitted_kernel_count_=0;
 bool kernel_register_census_invocation_armed_=false,has_kernel_register_census_nodes_=false;
 GraphKernelRegisterPolicy kernel_register_census_policy_;
 GraphKernelRegisterCensusStats kernel_register_census_stats_,kernel_register_census_phase_begin_stats_;
 std::vector<size_t>kernel_register_census_rule_counts_;
 uint64_t kernel_register_census_policy_digest_=13,kernel_register_census_plan_hash_=19;
 Build kernel_register_outer_fast_build_state_;
 std::optional<Prepared>kernel_register_outer_fast_prepared_;
 struct Current{State state;size_t position;};
 Current op_stream_stamp()const{return {state_,state_==State::REPLAYING?cursor_:nodes_.size()};}
 uint64_t semantic_spm_producer_yield_id_digest_for_window(size_t,size_t)const{return 0;}
 uint64_t topology_hash_for_window(size_t,size_t)const;
 uint64_t build_topology_hash()const;
 // Stamp and node-container fixture: the real model seal below enforces the
 // unchanged full window. Native stamp semantics are separately covered by
 // existing graph/source and recorded-window tests.
 Digest digest_op_stream_window(Stamp begin,Stamp end)const{
  TORCH_CHECK(state_==State::RECORDING&&begin.generation==generation&&end.generation==generation&&begin.signature==signature&&end.signature==signature);
  TORCH_CHECK(begin.position<=end.position&&end.position<=nodes_.size());
  Digest d;d.begin=begin.position;d.end=end.position;d.node_count=d.end-d.begin;
  for(size_t i=d.begin;i<d.end;++i){d.node_kinds.push_back(nodes_[i].kind);d.kind_mask|=1ULL<<static_cast<uint8_t>(nodes_[i].kind);}
  d.topology_hash=topology_hash_for_window(d.begin,d.end);return d;
 }
 void build_kernel_register_retained_catalog(){}
 void complete_kernel_register_outer_fast_ordinary(){
  Prepared p;p.build_state=Build{true,23,29};
  p.node_annotations.push_back(Annotation{&nodes_[2].dma});
  kernel_register_outer_fast_prepared_=std::move(p);
'''+publish+r'''
 }
 void complete_kernel_register_census();
};
'''+hash_method+'\n'+built_hash+'\n'+complete+r'''
struct Ranges{std::array<Stamp,4>boundaries;std::array<Digest,3>windows;Digest whole;};
struct Qwen3VlPoolerZ1State{Ranges recorded_ranges;};
void validate_recorded_ranges_locked(const Qwen3VlPoolerZ1State&s,RpuKernelGraph&g){
 TORCH_CHECK(s.recorded_ranges.whole.topology_hash==g.build_topology_hash());
 for(auto stamp:s.recorded_ranges.boundaries)TORCH_CHECK(stamp.generation==g.generation&&stamp.signature==g.signature);
}
'''+seal+r'''
template<class F>void rejects(F f){bool hit=false;try{f();}catch(const std::runtime_error&){hit=true;}assert(hit);}
void setup(RpuKernelGraph&g,Qwen3VlPoolerZ1State&s){
 using K=GraphNodeKind;
 g.nodes_={{K::Branch},{K::Kernel},{K::Dma},{K::Barrier},{K::Dma},{K::Kernel},{K::Branch},{K::Kernel},{K::Dma}};
 auto&p=g.kernel_register_census_policy_;p.fingerprint=11;
 p.expected_node_count=9;p.expected_kernel_count=3;p.expected_dma_count=3;p.expected_barrier_count=1;p.expected_no_ddr_kernel_count=3;
 p.expected_ordered_node_kind_hash=31;p.expected_ordered_kernel_hash=37;
 auto&a=g.kernel_register_census_stats_;a.node_count=9;a.kernel_count=3;a.dma_count=3;a.barrier_count=1;a.no_ddr_kernel_count=3;a.ordered_node_kind_hash=31;a.ordered_kernel_hash=37;
 auto&b=g.kernel_register_census_phase_begin_stats_;b.node_count=6;b.kernel_count=2;b.dma_count=2;b.barrier_count=1;b.no_ddr_kernel_count=2;b.ordered_node_kind_hash=41;b.ordered_kernel_hash=43;
 GraphKernelRegisterCensusStats final;final.node_count=3;final.kernel_count=1;final.dma_count=1;final.no_ddr_kernel_count=1;final.ordered_node_kind_hash=41;final.ordered_kernel_hash=43;
 p.phases={{"vision",{}},{"handoff",{}},{"text",final}};
 auto&r=s.recorded_ranges;r.boundaries={Stamp{0},Stamp{4},Stamp{6},Stamp{9}};
 for(size_t i=0;i<3;++i)r.windows[i]=g.digest_op_stream_window(r.boundaries[i],r.boundaries[i+1]);
 r.whole=g.digest_op_stream_window(r.boundaries.front(),r.boundaries.back());
}
int main(){
 {RpuKernelGraph g;Qwen3VlPoolerZ1State s;setup(g,s);const auto old=s.recorded_ranges.whole.topology_hash;
  g.complete_kernel_register_census();assert(g.kernel_register_census_complete_&&g.kernel_register_outer_fast_build_state_.staged&&g.nodes_[2].dma.outer_fast_owner_bound);
  g.state_=RpuKernelGraph::State::BUILT;assert(old!=g.build_topology_hash());
  rejects([&]{validate_recorded_ranges_locked(s,g);});}
 {RpuKernelGraph g;Qwen3VlPoolerZ1State s;setup(g,s);
  seal_recorded_ranges_topology_locked(s,g);g.complete_kernel_register_census();g.state_=RpuKernelGraph::State::BUILT;
  rejects([&]{validate_recorded_ranges_locked(s,g);});}
 {RpuKernelGraph g;Qwen3VlPoolerZ1State s;setup(g,s);const auto before=s.recorded_ranges;
  g.complete_kernel_register_census();seal_recorded_ranges_topology_locked(s,g);
  assert(before.whole.topology_hash!=s.recorded_ranges.whole.topology_hash);
  for(size_t i=0;i<3;++i){assert(before.windows[i].topology_hash==s.recorded_ranges.windows[i].topology_hash);assert(before.windows[i].node_kinds==s.recorded_ranges.windows[i].node_kinds);assert(before.boundaries[i].position==s.recorded_ranges.boundaries[i].position);}
  g.state_=RpuKernelGraph::State::BUILT;validate_recorded_ranges_locked(s,g);
  g.state_=RpuKernelGraph::State::REPLAYING;g.cursor_=g.nodes_.size();const auto hash=s.recorded_ranges.whole.topology_hash;
  seal_recorded_ranges_topology_locked(s,g);assert(s.recorded_ranges.whole.topology_hash==hash);
  s.recorded_ranges.whole.topology_hash^=1;rejects([&]{seal_recorded_ranges_topology_locked(s,g);});}
 for(int fault=0;fault<6;++fault){RpuKernelGraph g;Qwen3VlPoolerZ1State s;setup(g,s);g.complete_kernel_register_census();
  if(fault==0)g.nodes_.push_back({GraphNodeKind::Kernel});
  if(fault==1)g.nodes_[1].kind=GraphNodeKind::Dma;
  if(fault==2)++g.generation;
  if(fault==3)++g.signature;
  if(fault==4)--s.recorded_ranges.boundaries.back().position;
  if(fault==5)s.recorded_ranges.whole.topology_hash=0;
  rejects([&]{seal_recorded_ranges_topology_locked(s,g);});}
 {RpuKernelGraph g;Qwen3VlPoolerZ1State s;setup(g,s);g.pending_kernel_register_census_=1;
  rejects([&]{g.complete_kernel_register_census();});assert(!g.kernel_register_outer_fast_build_state_.staged);}
}
'''
    compiler=shutil.which('c++');assert compiler
    cpp,binary=tmp_path/'sealed.cpp',tmp_path/'sealed';cpp.write_text(source)
    result=subprocess.run([compiler,'-std=c++17','-O0',str(cpp),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    pipeline=_function(model,'at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward(')
    assert pipeline.index('register_census->complete()') < pipeline.index('seal_recorded_ranges_topology_locked(')
    body=_function(model,'void seal_recorded_ranges_topology_locked(')
    assert 'ranges.windows' not in body
