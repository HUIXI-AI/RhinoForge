"""Exercise the production prefill owner witness with real ATen ownership.

The device address and active Graph boundary are emulated; all owner matching,
prepare/commit and invalidation logic comes from the native model header.
"""

from pathlib import Path
import subprocess

import torch

def _method(source, marker):
    start = source.index(marker)
    brace = source.index("{", start)
    depth, end = 1, brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


ROOT = Path(__file__).resolve().parents[1]


def test_prefill_witness_is_bound_to_the_exact_graph_capture(tmp_path):
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    start = source.index("    struct FastReplayOwner {")
    witness = source[start:source.index("    // Batch decode: KV-cache slot", start)]
    witness += _method(source, "void set_fast_replay_skip_layer_loop(")
    assert witness.count("c10::DeviceType::PrivateUse1") == 1
    witness = witness.replace("c10::DeviceType::PrivateUse1", "c10::DeviceType::CPU")
    program = r'''
#include <ATen/ATen.h>
#include <cassert>
#include <vector>
namespace rhino_lkn {
uint64_t RpuGetDevAddr(const void* p) { return reinterpret_cast<uintptr_t>(p); }
}
struct GraphOpStreamStamp;
struct RpuKernelGraph {
 enum class State { PASSTHROUGH, RECORDING, BUILT, REPLAYING };
 inline static RpuKernelGraph* current = nullptr;
 State mode = State::RECORDING;
 uint64_t lifetime = 1, generation = 1, signature = 42, segment = 7;
 size_t cursor = 0, extent = 0;
 bool valid = true;
 static bool has_active() { return current != nullptr; }
 static RpuKernelGraph& active() { assert(current); return *current; }
 State state() const { return mode; }
 bool replayable() const { return valid; }
 struct Stats { uint64_t graph_lifetime_id; };
 Stats debug_stats() const { return {lifetime}; }
 GraphOpStreamStamp op_stream_stamp() const;
 bool is_op_stream_stamp_current(const GraphOpStreamStamp&) const;
};
struct GraphOpStreamStamp {
 const RpuKernelGraph* graph = nullptr;
 RpuKernelGraph::State state = RpuKernelGraph::State::PASSTHROUGH;
 uint64_t build_generation = 0, signature_identity = 0, signature_segment_key = 0;
 size_t position = 0, extent = 0;
};
GraphOpStreamStamp RpuKernelGraph::op_stream_stamp() const {
 assert(current == this);
 assert(mode == State::RECORDING || mode == State::REPLAYING);
 return {this, mode, generation, signature, segment,
         mode == State::RECORDING ? extent : cursor, extent};
}
bool RpuKernelGraph::is_op_stream_stamp_current(const GraphOpStreamStamp& s) const {
 return s.graph == this && s.build_generation != 0 && s.build_generation == generation &&
        valid && mode != State::PASSTHROUGH && s.signature_identity == signature &&
        s.signature_segment_key == segment && s.position <= extent && s.extent <= extent;
}
struct Model {
 bool fast_replay_skip_layer_loop_ = true;
 uint64_t model_generation = 1;
 uint64_t installed_model_state_generation() const { return model_generation; }
 void invalidate_model_state() { ++model_generation; }
 __WITNESS__
};
using Cache = std::vector<at::Tensor>;
Cache cache() { return {at::zeros({1, 2, 16}, at::kHalf), at::zeros({1, 2, 16}, at::kHalf)}; }
bool prepare(Model& m, RpuKernelGraph& graph, const Cache& k, const Cache& v, bool candidate = true) {
 RpuKernelGraph::current = &graph;
 graph.cursor = 0;
 m.prepare_fast_replay_prefill_owners(k, v, candidate);
 return m.fast_replay_owner_bound_;
}
void commit(Model& m, RpuKernelGraph& graph, const Cache& k, const Cache& v) {
 graph.extent = 100;
 graph.cursor = 100;
 m.commit_fast_replay_prefill_owners(k, v);
 graph.mode = RpuKernelGraph::State::REPLAYING;
}
int main() {
 Model m;
 RpuKernelGraph a, b;
 b.lifetime = 2;
 b.signature = 43;
 Cache xk = cache(), xv = cache(), yk = cache(), yv = cache();
 assert(!prepare(m, a, xk, xv)); commit(m, a, xk, xv);
 assert(prepare(m, a, xk, xv)); commit(m, a, xk, xv);
 assert(!prepare(m, b, xk, xv)); commit(m, b, xk, xv);
 // The old model-global witness allowed B/Y to skip although B still held X.
 assert(!prepare(m, a, yk, yv)); commit(m, a, yk, yv);
 assert(!prepare(m, b, yk, yv)); commit(m, b, yk, yv);
 for (int replay = 0; replay < 3; ++replay) {
  assert(prepare(m, b, yk, yv)); commit(m, b, yk, yv);
 }
 // Alternating already-built graphs remains safe with the bounded one-entry witness.
 assert(!prepare(m, a, yk, yv)); commit(m, a, yk, yv);
 assert(prepare(m, a, yk, yv));
 // Every identity component, including same-address Graph lifetime ABA, matters.
 for (auto member : {&RpuKernelGraph::lifetime, &RpuKernelGraph::generation,
                     &RpuKernelGraph::signature, &RpuKernelGraph::segment}) {
  ++(a.*member);
  assert(!prepare(m, a, yk, yv)); commit(m, a, yk, yv);
  assert(prepare(m, a, yk, yv));
 }
 ++m.model_generation;
 assert(!prepare(m, a, yk, yv)); commit(m, a, yk, yv);
 assert(prepare(m, a, yk, yv));
 a.cursor = 10;
 m.prepare_fast_replay_prefill_owners(yk, yv, true);
 assert(!m.fast_replay_owner_bound_); // same Graph, different model-call window
 assert(!prepare(m, a, yk, yv, false));
 m.commit_fast_replay_prefill_owners(xk, xv);
 assert(prepare(m, a, yk, yv));
 // Owner replacement and metadata changes require normal emission.
 for (bool replace_k : {false, true}) {
  auto k = yk, v = yv;
  (replace_k ? k : v)[1] = cache()[0];
  assert(!prepare(m, a, k, v)); commit(m, a, k, v);
  assert(prepare(m, a, k, v));
 }
 auto k = yk;
 assert(!prepare(m, a, k, yv)); commit(m, a, k, yv);
 k[0].transpose_(1, 2);
 assert(!prepare(m, a, k, yv));
 k[0].transpose_(1, 2);
 // Failed/aborted or moved graph scopes must never publish a new owner proof.
 assert(!prepare(m, a, xk, xv));
 ++a.generation;
 m.commit_fast_replay_prefill_owners(xk, xv);
 assert(!prepare(m, a, xk, xv)); commit(m, a, xk, xv);
 assert(!prepare(m, a, yk, yv));
 RpuKernelGraph::current = &b;
 m.commit_fast_replay_prefill_owners(yk, yv);
 assert(!prepare(m, b, yk, yv)); commit(m, b, yk, yv);
 assert(!prepare(m, b, xk, xv));
 ++m.model_generation;
 m.commit_fast_replay_prefill_owners(xk, xv);
 assert(!prepare(m, b, xk, xv)); commit(m, b, xk, xv);
 for (auto mode : {RpuKernelGraph::State::PASSTHROUGH, RpuKernelGraph::State::BUILT,
                   RpuKernelGraph::State::RECORDING}) {
  b.mode = mode;
  assert(!prepare(m, b, xk, xv));
 }
 b.mode = RpuKernelGraph::State::REPLAYING;
 b.valid = false;
 assert(!prepare(m, b, xk, xv));
 m.commit_fast_replay_prefill_owners(xk, xv);
 b.valid = true;
 RpuKernelGraph::current = nullptr;
 m.prepare_fast_replay_prefill_owners(xk, xv, true);
 assert(!m.fast_replay_prefill_candidate_ && !m.fast_replay_owner_bound_);
 assert(!prepare(m, b, {}, {}));
 assert(!prepare(m, b, xk, {}));
 // Strong ownership prevents TensorImpl/storage-address reuse after caller replacement.
 int deleted = 0;
 {
  Model retained;
  RpuKernelGraph g;
  Cache k_owned{at::from_blob(new float[16], {16}, [&](void* p) {
    delete[] static_cast<float*>(p); ++deleted;
   }, at::TensorOptions().dtype(at::kFloat))};
  Cache v_owned{at::zeros({16}, at::kFloat)};
  assert(!prepare(retained, g, k_owned, v_owned)); commit(retained, g, k_owned, v_owned);
  k_owned.clear();
  assert(deleted == 0);
  retained.set_fast_replay_skip_layer_loop(false);
  assert(deleted == 1);
  assert(!retained.fast_replay_owner_bound_ && !retained.fast_replay_bound_stamp_.graph);
 }
}
'''.replace("__WITNESS__", witness)
    cpp, exe = tmp_path / "prefill_owners.cpp", tmp_path / "prefill_owners"
    cpp.write_text(program)
    torch_root = Path(torch.__file__).resolve().parent
    subprocess.run([
        "g++", "-std=c++17", "-O0",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        "-I", str(torch_root / "include"), str(cpp),
        "-L", str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
        f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True)
