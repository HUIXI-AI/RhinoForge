"""CPU execution of authenticated manifest metadata capture and census."""
from pathlib import Path

from test_siglip_attention_dry_live_contract import _function, _run_native


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_branch_keeps_exact_census_and_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    header = (ROOT / "src/graph/graph_runtime.h").read_text()
    infra = (ROOT / "src/graph/graph_infra.h").read_text()
    child = (ROOT / "src/graph/graph_runtime_child.cpp").read_text()
    capture = (ROOT / "src/graph/graph_runtime_capture.cpp").read_text()
    census = (ROOT / "src/graph/graph_register_census.cpp").read_text()
    definitions = "\n".join(_function(source, marker) + ";" for source, marker in (
        (infra, "struct GraphSignature {"),
        (header, "struct BranchNodeData {"),
        (header, "enum class GraphNodeKind :"),
        (header, "struct GraphKernelRegisterCensusStats {"),
        (header, "struct GraphKernelRegisterPhaseExpectation {"),
    ))
    functions = "\n".join(_function(census, marker) for marker in (
        "uint64_t fnv_byte(", "uint64_t fnv_u64(", "uint64_t fnv_string(",
        "GraphKernelRegisterCensusStats stats_delta(", "void check_stats_counts(",
    ))
    methods = "\n".join(_function(source, marker).replace("RpuKernelGraph::", "Probe::")
                        for source, marker in (
        (child, "void RpuKernelGraph::record_branch("),
        (child, "void RpuKernelGraph::record_physical_manifest_branch("),
        (census, "void RpuKernelGraph::observe_physical_manifest_branch_for_census("),
        (census, "void RpuKernelGraph::observe_kernel_register_node("),
    ))
    methods += "\n" + "\n".join(
        "template <typename ND>\n" + _function(capture, marker).replace("RpuKernelGraph::", "Probe::")
        for marker in ("bool RpuKernelGraph::capture_replay_child_skip_data(",
                       "void RpuKernelGraph::capture_data(")
    )
    policy = _function(census, "void validate_policy(")
    totals = policy[policy.index("    TORCH_CHECK(policy.expected_kernel_count"):
                    policy.index("    TORCH_CHECK(policy.expected_weight_operand_count")]
    phases = policy[policy.index("        TORCH_CHECK(stats.node_count"):
                    policy.index("        phase_totals.node_count +=")]
    source = r'''
#include <c10/util/Exception.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>
constexpr uint64_t kFnvOffset=14695981039346656037ULL, kFnvPrime=1099511628211ULL;
''' + definitions + "\n" + functions + r'''
struct GraphNode {
    GraphNodeKind kind;
    std::variant<BranchNodeData> data;
    const BranchNodeData& as_branch() const { return std::get<BranchNodeData>(data); }
    BranchNodeData& as_branch() { return std::get<BranchNodeData>(data); }
};
struct MemcpyNodeData {};
struct MemsetNodeData {};
enum class CopyKind { DDR_TO_DDR };
template<class ND> struct NodeKindOf;
template<> struct NodeKindOf<BranchNodeData> { static constexpr auto value=GraphNodeKind::Branch; };
std::string format_hex(uint64_t n) { return std::to_string(n); }
struct Policy {
    size_t expected_node_count=3, expected_kernel_count=1;
    size_t expected_dma_count=0, expected_barrier_count=0;
    size_t expected_typed_kernel_count=0, expected_no_ddr_kernel_count=1;
    uint64_t expected_ordered_node_kind_hash=1, expected_ordered_kernel_hash=1;
    std::vector<GraphKernelRegisterPhaseExpectation> phases;
};
void validate_shape(const Policy& policy) {
''' + totals + r'''
    for(const auto& phase:policy.phases) { const auto& stats=phase.stats;
''' + phases + r'''
    }
}
struct Probe {
    enum class State { RECORDING, REPLAYING, PASSTHROUGH, BUILT };
    State state_=State::RECORDING;
    bool require_single_segment_=false;
    bool kernel_register_census_active_=true, has_kernel_register_census_nodes_=false;
    bool kernel_register_census_complete_=false, kernel_register_census_poisoned_=false;
    std::optional<int> pending_kernel_register_census_;
    struct { bool active=false; size_t child_cursor=0; } replay_child_skip_;
    struct { Probe* child=nullptr; } child_graph;
    struct { int execution_core_count=8; } runtime_policy_;
    bool canonical_dma_build_trace_active_=false, canonical_dma_callback_active_=false;
    bool dma_burst_open=false;
    static inline Probe* current=nullptr;
    static bool has_active() { return current!=nullptr; }
    static Probe& active() { return *current; }
    std::vector<GraphNode> nodes_;
    size_t cursor_=0, kernel_register_census_next_phase_=0, ordinary_capture_calls=0;
    Policy kernel_register_census_policy_;
    GraphKernelRegisterCensusStats kernel_register_census_stats_;
    GraphKernelRegisterCensusStats kernel_register_census_phase_begin_stats_;
    Probe() {
        current=this; reset_stats();
        auto expected=kernel_register_census_stats_;
        expected.node_count=3; expected.kernel_count=1; expected.no_ddr_kernel_count=1;
        kernel_register_census_policy_.phases={{"full",expected}};
    }
    void reset_stats() {
        kernel_register_census_stats_={}; kernel_register_census_phase_begin_stats_={};
        kernel_register_census_stats_.ordered_node_kind_hash=kFnvOffset;
        kernel_register_census_stats_.ordered_kernel_hash=kFnvOffset;
        kernel_register_census_phase_begin_stats_=kernel_register_census_stats_;
    }
    void check_kernel_register_dma_burst_closed(const char*) const {
        TORCH_CHECK(!dma_burst_open,"open DMA burst");
    }
    auto& enter_replay_child_skip(const char*) {
        assert(child_graph.child); replay_child_skip_.active=true; return child_graph;
    }
    void finish_replay_child_skip_if_complete() {
        if(replay_child_skip_.child_cursor==child_graph.child->nodes_.size()) {
            replay_child_skip_.active=false; ++cursor_;
        }
    }
    template<class T> void capture_data(T data);
    template<class T> bool capture_replay_child_skip_data(T data);
    void resolve_recording_input_deps(int,const std::vector<uint64_t>&) { assert(false); }
    void track_recording_writer_dev_addr(uint64_t,int) { assert(false); }
    void record_branch(uint64_t,GraphSignature={},std::string={});
    void record_physical_manifest_branch(uint64_t);
    void observe_physical_manifest_branch_for_census();
    void observe_kernel_register_node(GraphNodeKind,const std::string*);
    void kernel() {
        const std::string name="test_kernel";
        ++kernel_register_census_stats_.kernel_count;
        ++kernel_register_census_stats_.no_ddr_kernel_count;
        observe_kernel_register_node(GraphNodeKind::Kernel,&name);
        if(state_==State::RECORDING) nodes_.push_back({GraphNodeKind::Kernel,{}});
        else { assert(nodes_.at(cursor_).kind==GraphNodeKind::Kernel); ++cursor_; }
    }
};
''' + methods + r'''
template<class F> void rejects(F&& f,const char* text) {
    bool rejected=false;
    try { f(); } catch(const c10::Error& error) {
        rejected=std::string(error.what()).find(text)!=std::string::npos;
        if(!rejected) std::cerr<<error.what()<<"\n";
    }
    assert(rejected);
}
int main() {
    Probe p; validate_shape(p.kernel_register_census_policy_);
    p.record_physical_manifest_branch(71); p.kernel(); p.record_physical_manifest_branch(72);
    assert(p.nodes_.size()==3 && p.kernel_register_census_stats_.node_count==3);
    assert(p.kernel_register_census_stats_.kernel_count==1 && p.kernel_register_census_stats_.dma_count==0);
    auto expected_hash=fnv_byte(fnv_byte(fnv_byte(kFnvOffset,5),0),5);
    assert(p.kernel_register_census_stats_.ordered_node_kind_hash==expected_hash);
    const auto built_stats=p.kernel_register_census_stats_;
    assert(p.ordinary_capture_calls==0);
    rejects([&]{p.record_physical_manifest_branch(73);},"metadata quota");
    assert(p.nodes_.size()==3 && p.kernel_register_census_stats_.node_count==3);
    rejects([&]{p.record_branch(72,{},"FMB complete physical manifest");},"forbidden");
    for(int replay=0;replay<3;++replay) {
        p.state_=Probe::State::REPLAYING; p.cursor_=0; p.reset_stats();
        p.record_physical_manifest_branch(71); p.kernel();
        rejects([&]{check_stats_counts(p.kernel_register_census_stats_,built_stats,"missing branch");},"census");
        p.record_physical_manifest_branch(72);
        check_stats_counts(p.kernel_register_census_stats_,built_stats,"full");
        assert(p.cursor_==p.nodes_.size());
        assert(p.kernel_register_census_stats_.ordered_node_kind_hash==expected_hash);
    }
    for(int kind=0;kind<8;++kind) {
        Probe::current=&p; p.state_=Probe::State::REPLAYING; p.cursor_=0; p.reset_stats();
        auto saved=p.nodes_[0];
        if(kind==0) p.nodes_[0].as_branch().branch_key=99;
        if(kind==1) p.nodes_[0].as_branch().label="untrusted";
        if(kind==2) p.nodes_[0].as_branch().branch_signature.flags=1;
        if(kind==3) p.nodes_[0].kind=GraphNodeKind::Dma;
        if(kind==4) p.pending_kernel_register_census_=1;
        if(kind==5) p.kernel_register_census_poisoned_=true;
        if(kind==6) p.dma_burst_open=true;
        if(kind==7) p.kernel_register_census_complete_=true;
        rejects([&]{p.record_physical_manifest_branch(71);},
                kind<3?"identity mismatch":kind==3?"node mismatch":kind==6?"open DMA":"unpoisoned");
        assert(p.cursor_==0 && p.kernel_register_census_stats_.node_count==0);
        p.nodes_[0]=saved; p.pending_kernel_register_census_.reset();
        p.kernel_register_census_poisoned_=false; p.kernel_register_census_complete_=false; p.dma_burst_open=false;
    }
    rejects([&]{p.record_physical_manifest_branch(0);},"nonzero");
    Probe::current=nullptr;
    rejects([&]{p.record_physical_manifest_branch(71);},"unpoisoned");
    Probe::current=&p;
    p.kernel_register_census_policy_.phases[0].stats.node_count=1;
    rejects([&]{p.record_physical_manifest_branch(71);},"metadata quota");
    p.kernel_register_census_policy_.expected_node_count=0;
    rejects([&]{validate_shape(p.kernel_register_census_policy_);},"incomplete exact census");
    // Ordinary Branch cannot exploit the fixed label, even with a metadata quota.
    { Probe q; rejects([&]{q.record_branch(7,{},"FMB complete physical manifest");},"forbidden");
      q.kernel_register_census_active_=false; q.record_physical_manifest_branch(9);
      assert(q.nodes_.size()==1 && q.ordinary_capture_calls==0);
      q.state_=Probe::State::REPLAYING;
      q.record_physical_manifest_branch(9); assert(q.cursor_==1);
      q.cursor_=0;
      rejects([&]{q.record_physical_manifest_branch(10);},"identity mismatch");
      rejects([&]{q.record_branch(9,{},"ordinary");},"private replay API");
      BranchNodeData fake; fake.branch_key=9; fake.label=BranchNodeData::physical_manifest_label();
      rejects([&]{q.capture_data(fake);},"reserved");
      q.state_=Probe::State::RECORDING;
      rejects([&]{q.capture_data(fake);},"reserved");
      rejects([&]{q.record_branch(9,{},fake.label);},"reserved"); }
    // The private metadata path still replays inside a promoted child window.
    for(int bad=0;bad<4;++bad) {
        Probe child; child.kernel_register_census_active_=false;
        child.record_physical_manifest_branch(41);
        child.record_physical_manifest_branch(42);
        Probe parent; parent.kernel_register_census_active_=false;
        parent.nodes_.push_back({GraphNodeKind::ChildGraph,{}});
        parent.child_graph.child=&child; parent.state_=Probe::State::REPLAYING;
        if(bad==1) child.nodes_[0].as_branch().branch_key=99;
        if(bad==2) child.nodes_[0].as_branch().label="ordinary";
        if(bad==3) child.nodes_[0].as_branch().branch_signature.flags=1;
        if(bad) {
            rejects([&]{parent.record_physical_manifest_branch(41);},
                    bad==1?"branch-key mismatch":"identity mismatch");
            assert(parent.cursor_==0 && parent.replay_child_skip_.child_cursor==0);
        } else {
            parent.record_physical_manifest_branch(41);
            assert(parent.cursor_==0 && parent.replay_child_skip_.child_cursor==1);
            parent.record_physical_manifest_branch(42);
            assert(parent.cursor_==1 && !parent.replay_child_skip_.active);
        }
    }
    // Generic census observation still rejects Branch and all opaque node kinds.
    for(auto kind:{GraphNodeKind::Branch,GraphNodeKind::Memcpy,GraphNodeKind::Memset,
                   GraphNodeKind::ChildGraph,GraphNodeKind::HostCallback,GraphNodeKind::Tier3Oneshot}) {
        Probe q; rejects([&]{q.observe_kernel_register_node(kind,nullptr);},"rejects Graph node kind");
    }
    std::cout<<"authenticated manifest Branch census: PASS\n";
}
'''
    assert "authenticated manifest Branch census: PASS" in _run_native(tmp_path, source, torch)


def test_only_private_fmb_entry_admits_metadata_and_full_coverage_remains():
    header = (ROOT / "src/graph/graph_runtime.h").read_text()
    runtime_class = header[header.index("class RpuKernelGraph {\npublic:"):]
    assert runtime_class.index("private:") < runtime_class.index("void record_physical_manifest_branch(")
    assert "friend class v3::FusedModelBase;" in runtime_class
    capture = (ROOT / "src/graph/graph_runtime_capture.cpp").read_text()
    assert "typed DDR-register census admits only Graph-visible " in capture
    census = (ROOT / "src/graph/graph_register_census.cpp").read_text()
    for marker in ("void RpuKernelGraph::complete_kernel_register_census(",
                   "void RpuKernelGraph::verify_kernel_register_end_preflight("):
        assert "kernel_register_census_stats_.node_count == nodes_.size()" in _function(census, marker)
