"""GR00T-only COMPLETE marker/range checks, with no RPU/native backend import."""
import ast
import copy
from pathlib import Path
import shutil
import subprocess
import textwrap
from types import SimpleNamespace

import pytest
from test_siglip_attention_dry_live_contract import _function

ROOT=Path(__file__).resolve().parents[1]


def test_native_exact_marker_authentication_and_export(tmp_path):
    outer=(ROOT/'src/fused/rpu_qwen3vl_pooler_spm_z1.cpp').read_text()
    graph=(ROOT/'src/graph/graph_runtime.h').read_text()
    funcs='\n'.join(_function(outer,marker) for marker in (
        'void validate_groot_component_window_kinds(',
        'void validate_groot_component_window(',
        'uint64_t groot_component_manifest_fingerprint(',
    ))
    export=outer[outer.index('std::vector<std::vector<int64_t>> rpu_qwen3vl_pooler_spm_z1_recorded_ranges('):]
    export_loop=_function(export,'    for (size_t phase = 0; phase < ranges.windows.size(); ++phase) {')
    branch=_function(graph,'struct BranchNodeData {')+';'
    kinds=_function(graph,'enum class GraphNodeKind :')+';'
    source=r'''
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do{if(!(c))throw std::runtime_error("rejected");}while(0)
#define TORCH_INTERNAL_ASSERT(c, ...) TORCH_CHECK(c)
struct GraphSignature {int value=0;bool operator==(const GraphSignature& x)const{return value==x.value;}};
'''+branch+'\n'+kinds+r'''
using GraphNodeKindMask=uint64_t;
GraphNodeKindMask graph_node_kind_bit(GraphNodeKind k){return 1ULL<<static_cast<uint8_t>(k);}
struct GraphOpStreamWindowDigest {
 size_t begin=0,end=0,node_count=0;GraphNodeKindMask kind_mask=0;uint64_t topology_hash=99;
 std::vector<GraphNodeKind> node_kinds;
};
struct GraphNode {GraphNodeKind kind;BranchNodeData branch;
 const BranchNodeData&as_branch()const{return branch;}};
namespace v3 {
 enum class FmbPhysicalManifestState{UNSPECIFIED,COMPLETE};
 enum class FmbGraphLifecycle{ORDINARY,COMPOSITE_CHILD};
 struct Manifest {FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
 FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;};
 struct Candidate {Manifest physical_manifest;};
 struct Prepared {Candidate stored;uint64_t key;
 const Candidate&candidate()const{return stored;}
 uint64_t manifest_fingerprint()const{return key;}};
 struct Owner {std::vector<int64_t> descriptor;std::shared_ptr<Prepared> prepared;
 auto prepare_stage_candidate(const std::vector<int64_t>&d){TORCH_CHECK(d==descriptor);return prepared;}};
 namespace qwen3vl_pooler_z1_internal {
 Owner vision{{101},std::make_shared<Prepared>(Prepared{{},71})};
 Owner text{{202},std::make_shared<Prepared>(Prepared{{},72})};
 Owner&multiview_vision_owner(int64_t h){TORCH_CHECK(h==11);return vision;}
 Owner&multiview_text_owner(int64_t h){TORCH_CHECK(h==12);return text;}
 }
}
struct Qwen3VlPoolerZ1State {int64_t vision_handle=11,text_handle=12;
 std::vector<int64_t>vision_stage_descriptor{101},text_stage_descriptor{202};};
'''+funcs+r'''
GraphOpStreamWindowDigest window(const std::vector<GraphNode>&n,size_t a,size_t b){
 GraphOpStreamWindowDigest w;w.begin=a;w.end=b;w.node_count=b-a;
 for(size_t i=a;i<b;++i){w.node_kinds.push_back(n[i].kind);w.kind_mask|=graph_node_kind_bit(n[i].kind);}return w;
}
template<class F>void rejects(F f){bool hit=false;try{f();}catch(const std::runtime_error&){hit=true;}assert(hit);}
int64_t encode_u64(uint64_t v){return static_cast<int64_t>(v);}
int main(){
 const auto K=GraphNodeKind::Kernel,D=GraphNodeKind::Dma,B=GraphNodeKind::Barrier,M=GraphNodeKind::Branch;
 const std::string label=BranchNodeData::physical_manifest_label();
 const std::vector<GraphNode> nodes={{M,{71,{},label}},{K,{}},{D,{}},{B,{}},{D,{}},{K,{}},{M,{72,{},label}},{K,{}},{D,{}}};
 Qwen3VlPoolerZ1State s;
 assert(groot_component_manifest_fingerprint(s,0)==71);
 assert(groot_component_manifest_fingerprint(s,1)==0);
 assert(groot_component_manifest_fingerprint(s,2)==72);
 auto v=window(nodes,0,4),h=window(nodes,4,6),t=window(nodes,6,9);
 validate_groot_component_window(v,nodes,0,groot_component_manifest_fingerprint(s,0));
 validate_groot_component_window(h,nodes,1,0);
 validate_groot_component_window(t,nodes,2,groot_component_manifest_fingerprint(s,2));
 for(int fault=0;fault<9;++fault){auto bad=nodes;
  if(fault==0)bad[0].branch.branch_key=72;
  if(fault==1)bad[0].branch.label="ordinary branch";
  if(fault==2)bad[0].branch.branch_signature.value=1;
  if(fault==3)bad[0].branch.branch_key=0;
  if(fault==4)std::swap(bad[0],bad[1]);
  if(fault==5)bad[1]=bad[0];
  if(fault==6)bad[0]={K,{}};
  if(fault==7)bad[2]={GraphNodeKind::HostCallback,{}};
  if(fault==8)bad[2]={GraphNodeKind::ChildGraph,{}};
  rejects([&]{validate_groot_component_window(window(bad,0,4),bad,0,71);});
 }
 rejects([&]{validate_groot_component_window(v,nodes,1,0);});
 rejects([&]{validate_groot_component_window(h,nodes,0,71);});
 rejects([&]{validate_groot_component_window(t,nodes,2,71);});
 auto bad=v;bad.kind_mask^=1;rejects([&]{validate_groot_component_window(bad,nodes,0,71);});
 bad=v;++bad.end;rejects([&]{validate_groot_component_window(bad,nodes,0,71);});
 bad=v;bad.node_kinds[1]=D;rejects([&]{validate_groot_component_window(bad,nodes,0,71);});
 auto stale=s;stale.vision_stage_descriptor={303};rejects([&]{groot_component_manifest_fingerprint(stale,0);});
 v3::qwen3vl_pooler_z1_internal::vision.prepared->stored.physical_manifest.state=v3::FmbPhysicalManifestState::UNSPECIFIED;
 rejects([&]{groot_component_manifest_fingerprint(s,0);});
 struct {std::vector<GraphOpStreamWindowDigest>windows;}ranges{{v,h,t}};
 std::vector<std::vector<int64_t>>rows;
'''+export_loop+r'''
 assert(rows.size()==3&&rows[0][1]==0&&rows[0][2]==4&&rows[0][8]==5);
 assert(rows[0][5]==1&&rows[0][6]==1&&rows[0][7]==1);
 assert(rows[1][1]==4&&rows[1][2]==6&&rows[2][1]==6&&rows[2][8]==5);
}
'''
    compiler=shutil.which('c++');assert compiler
    cpp,binary=tmp_path/'groot_ranges.cpp',tmp_path/'groot_ranges';cpp.write_text(source)
    built=subprocess.run([compiler,'-std=c++17','-O0',str(cpp),'-o',str(binary)],capture_output=True,text=True,timeout=60)
    assert built.returncode==0,built.stdout+built.stderr
    ran=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    assert ran.returncode==0,ran.stdout+ran.stderr
    # Both BUILD admission and retained REPLAY revalidate actual nodes and the
    # current prepared descriptor. Export only consumes those private windows.
    for marker in ['void checkpoint_recorded_range_locked(', 'void validate_recorded_ranges_locked(']:
        body=_function(outer,marker)
        assert 'validate_groot_component_window(' in body
        assert 'groot_component_manifest_fingerprint(' in body and 'graph.replay_plan()' in body


def _adapter_case(*, inside=False, phase_kinds=None):
    source=(ROOT/'python/rpu_backend/adapters/gr00t/runtime.py').read_text()
    tree=ast.parse(source)
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_qwen_z1_recorded_ranges')
    actual=textwrap.dedent(ast.get_source_segment(source,method))
    phase_kinds=phase_kinds or ((5,0,8,9),(8,0),(5,0,8))
    kinds=sum(phase_kinds,())
    rows=[[1,3],[11,12,13,21,22,23,0,24,len(kinds)]]
    start=0
    for phase,values in enumerate(phase_kinds,1):
        rows.append([phase,start,start+len(values),sum(1<<k for k in set(values)),100+phase,
                     values.count(0),values.count(8),values.count(9),*values]);start+=len(values)
    spans=((1,9),) if inside else ((1,6),(7,9))
    segments=[SimpleNamespace(segment_id=i,start_idx=a,end_idx=b,kernel_count=kinds[a:b].count(0),
        dma_count=kinds[a:b].count(8),barrier_count=kinds[a:b].count(9),manifest_metadata_count=kinds[a:b].count(5))
        for i,(a,b) in enumerate(spans)]
    stats=SimpleNamespace(kernel_count=kinds.count(0),dma_count=kinds.count(8),barrier_count=kinds.count(9),
                          segment_count=len(segments),segment_census=segments)
    graph=SimpleNamespace(built_signature_identity=lambda:23,built_signature=lambda:SimpleNamespace(segment_key=0),
                          build_topology_hash=lambda:24,graph_size=lambda:len(kinds),debug_stats=lambda:stats)
    owner=SimpleNamespace(_vision=SimpleNamespace(_rpu_vision_handle=11),_text=SimpleNamespace(_rpu_decoder_handle=12),
                          _qwen_z1_plan_hash=13,_qwen_z1_graph_identity=lambda _graph:(21,22))
    scope={'torch':SimpleNamespace(ops=SimpleNamespace(rpu=SimpleNamespace(qwen3vl_pooler_spm_z1_recorded_ranges=lambda *args:rows)))}
    exec(actual,scope)
    return lambda:scope['_qwen_z1_recorded_ranges'](owner,graph),rows,stats


@pytest.mark.parametrize('inside',[False,True])
def test_adapter_keeps_exact_coordinates_and_authenticated_metadata_gaps(inside):
    run,rows,_=_adapter_case(inside=inside)
    assert run()==tuple(tuple(row) for row in rows)
    assert sum(len(row[8:]) for row in rows[2:])==9


@pytest.mark.parametrize('fault',['missing','extra','wrong_position','wrong_phase','callback','gap_work','gap_count','segment_metadata','segment_overlap','range_mask','range_total'])
def test_adapter_rejects_marker_and_sdk_gap_drift(fault):
    phases=((5,0,8,9),(8,0),(5,0,8))
    if fault=='missing':phases=((0,0,8,9),(8,0),(5,0,8))
    if fault=='extra':phases=((5,5,8,9),(8,0),(5,0,8))
    if fault=='wrong_position':phases=((0,5,8,9),(8,0),(5,0,8))
    if fault=='wrong_phase':phases=((5,0,8,9),(5,0),(5,0,8))
    if fault=='callback':phases=((5,0,6,9),(8,0),(5,0,8))
    run,rows,stats=_adapter_case(phase_kinds=phases)
    if fault=='gap_work':
        # Reconcile segment counts to ensure skipped device work fails the gap
        # check rather than a simple aggregate/packet-count mismatch.
        stats.segment_census[0].start_idx=2
        stats.segment_census[0].kernel_count-=1
    if fault=='gap_count':stats.segment_census[1].start_idx=6
    if fault=='segment_metadata':stats.segment_census[0].manifest_metadata_count=1
    if fault=='segment_overlap':stats.segment_census[1].start_idx=5
    if fault=='range_mask':rows[2][3]^=1
    if fault=='range_total':rows[1][-1]-=1
    with pytest.raises(RuntimeError):run()
