"""Execute the native range gate against valid COMPLETE and foreign markers."""
from pathlib import Path
import ast
from collections import Counter
import json
import subprocess

import pytest

from test_native_kvinsert_cost_domain import _body

ROOT = Path(__file__).resolve().parents[1]



def test_native_batch_window_accepts_only_exact_complete_prologue(tmp_path):
    native = (ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.cpp").read_text()
    kinds = _body(native, "validate_groot_component_window_kinds")
    body = _body(native, "validate_groot_component_window")
    source = r"""
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
enum class GraphNodeKind { Kernel=0, Branch=5, HostCallback=6, Dma=8, Barrier=9 };
uint64_t graph_node_kind_bit(GraphNodeKind k) {return uint64_t(1)<<unsigned(k);}
struct GraphSignature {int key=0; bool operator==(const GraphSignature& x)const{return key==x.key;}};
struct BranchNodeData {
 uint64_t branch_key=0; GraphSignature branch_signature; std::string label;
 bool is_physical_manifest_metadata() const {
  return branch_key!=0 && label=="FMB complete physical manifest" &&
   branch_signature==GraphSignature{};
 }
};
struct GraphNode {
 GraphNodeKind kind; BranchNodeData branch;
 const BranchNodeData& as_branch()const{return branch;}
};
struct GraphOpStreamWindowDigest {
 size_t begin=0,end=0,node_count=0; uint64_t kind_mask=0;
 std::vector<GraphNodeKind> node_kinds;
};
using GraphNodeKindMask = uint64_t;
void validate_groot_component_window_kinds(
 const GraphOpStreamWindowDigest& window,size_t phase) __KINDS__
void validate_groot_component_window(const GraphOpStreamWindowDigest& window,
 const std::vector<GraphNode>& nodes,size_t phase,uint64_t manifest_fingerprint) __BODY__
GraphOpStreamWindowDigest digest(const std::vector<GraphNode>& nodes) {
 GraphOpStreamWindowDigest w;w.end=w.node_count=nodes.size();
 for(auto&n:nodes){w.node_kinds.push_back(n.kind);w.kind_mask|=graph_node_kind_bit(n.kind);}
 return w;
}
template<class F> void rejects(F f){bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}assert(failed);}
int main(){
 const auto M=GraphNodeKind::Branch,K=GraphNodeKind::Kernel,D=GraphNodeKind::Dma;
 const GraphNode marker{M,{123,{},"FMB complete physical manifest"}};
 std::vector<GraphNode> good{marker,{K,{}},{D,{}}};
 validate_groot_component_window(digest(good),good,0,123);
 validate_groot_component_window(digest(good),good,2,123);
 rejects([&]{validate_groot_component_window(digest(good),good,3,123);});
 rejects([&]{validate_groot_component_window(digest(good),good,1,123);});
 std::vector<GraphNode> handoff{{D,{}},{K,{}}};
 validate_groot_component_window(digest(handoff),handoff,1,0);
 rejects([&]{validate_groot_component_window(digest(good),good,0,0);});
 rejects([&]{validate_groot_component_window(digest(good),good,0,124);});
 rejects([&]{validate_groot_component_window(digest(handoff),handoff,1,123);});
 auto bad=good;bad[0].branch.label="ordinary execution branch";
 rejects([&]{validate_groot_component_window(digest(bad),bad,0,123);});
 bad=good;bad[0].branch.branch_signature.key=1;
 rejects([&]{validate_groot_component_window(digest(bad),bad,0,123);});
 bad=good;std::swap(bad[0],bad[1]);
 rejects([&]{validate_groot_component_window(digest(bad),bad,0,123);});
 bad=good;bad.push_back(marker);
 rejects([&]{validate_groot_component_window(digest(bad),bad,0,123);});
 bad=good;bad[1].kind=GraphNodeKind::HostCallback;
 rejects([&]{validate_groot_component_window(digest(bad),bad,0,123);});
 auto w=digest(good);w.node_kinds[1]=D;
 rejects([&]{validate_groot_component_window(w,good,0,123);});
}
""".replace("__KINDS__", kinds).replace("__BODY__", body)
    cpp = tmp_path / "markers.cpp"
    exe = tmp_path / "markers"
    cpp.write_text(source)
    subprocess.run(["c++", "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
    checkpoint = _body(native, "checkpoint_recorded_range_locked")
    assert "validate_groot_component_window(" in checkpoint
    assert "groot_component_manifest_fingerprint(s, boundary - 1)" in checkpoint
    authority = _body(native, "groot_component_manifest_fingerprint")
    assert "FmbPhysicalManifestState::COMPLETE" in authority
    assert "FmbGraphLifecycle::COMPOSITE_CHILD" in authority
    assert "prepared->manifest_fingerprint()" in authority
    assert "s.vision_stage_descriptor : s.text_stage_descriptor" in authority
