"""Exercise the production invocation snapshot without a device/runtime."""
from pathlib import Path
import subprocess


def test_dma_snapshots_keep_each_invocation_and_replay_slot(tmp_path):
    source = (Path(__file__).resolve().parents[1] /
              "src/graph/graph_runtime_capture.cpp").read_text()
    source = source[source.index("void RpuKernelGraph::snapshot_legacy_dma_bases("):
                    source.index("void RpuKernelGraph::record_barrier(")]
    harness = r'''
#include <cassert>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok, ...) if (!(ok)) throw std::runtime_error("rejected")
enum class GraphNodeKind {Dma, Kernel};
struct DmaNodeData {
 enum class Variant {Fixed, MutableSrc, MutableDst};
 Variant variant=Variant::MutableSrc;
 uint64_t semantic_endpoint_id=0; bool outer_fast_owner_bound=false;
 const uint64_t* live_base=nullptr;
};
struct Node {
 GraphNodeKind kind=GraphNodeKind::Dma; DmaNodeData data;
 DmaNodeData& as_dma(){return data;}
};
struct GraphOpStreamStamp {int state;size_t position;};
struct RpuKernelGraph {
 struct State {static constexpr int RECORDING=1, REPLAYING=2;};
 int state_=State::RECORDING;std::vector<Node> nodes_;size_t cursor=0;bool current=true;
 GraphOpStreamStamp op_stream_stamp(){return {state_,state_==1?nodes_.size():cursor};}
 bool is_op_stream_stamp_current(const GraphOpStreamStamp&){return current;}
 void snapshot_legacy_dma_bases(const GraphOpStreamStamp&,std::deque<uint64_t>&);
};
'''
    checks = r'''
int main(){
 RpuKernelGraph g;uint64_t shared=100;
 g.nodes_.push_back({GraphNodeKind::Dma,{DmaNodeData::Variant::MutableSrc,0,false,&shared}});
 std::deque<uint64_t> first,second;
 g.snapshot_legacy_dma_bases({1,0},first);auto* stable=g.nodes_[0].data.live_base;
 shared=200;g.nodes_.push_back({GraphNodeKind::Dma,{DmaNodeData::Variant::MutableDst,0,false,&shared}});
 g.snapshot_legacy_dma_bases({1,1},second);
 assert(*g.nodes_[0].data.live_base==100 && *g.nodes_[1].data.live_base==200);
 g.state_=2;g.cursor=1;shared=300;g.nodes_[0].data.live_base=&shared;
 g.snapshot_legacy_dma_bases({2,0},first);
 assert(g.nodes_[0].data.live_base==stable && *stable==300 && second[0]==200);
 g.nodes_[0].data.semantic_endpoint_id=9;
 bool rejected=false;try{g.snapshot_legacy_dma_bases({2,0},first);}catch(...){rejected=true;}
 assert(rejected && *stable==300);
 g.nodes_[0].data.semantic_endpoint_id=0;g.current=false;rejected=false;
 try{g.snapshot_legacy_dma_bases({2,0},first);}catch(...){rejected=true;}assert(rejected);
 g.current=true;std::deque<uint64_t> missing;rejected=false;
 try{g.snapshot_legacy_dma_bases({2,0},missing);}catch(...){rejected=true;}assert(rejected);
}
'''
    cpp = tmp_path / "snapshot.cpp"
    cpp.write_text(harness + source + checks)
    exe = tmp_path / "snapshot"
    subprocess.run(["c++", "-std=c++17", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
