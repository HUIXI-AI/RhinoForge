"""Eager col-partition DDR Linear: one SDK submission per call.

`rpu_launch_linear_ddr_kernel` records its eager col-partition steps in a
private, signature-less Graph scope (EagerLinearBatch) so they run as one
one-shot batch. A scope left open after a failure would make every later eager
call record into a dead Graph, so the failure paths are compiled here against
a stand-in Graph; the device behaviour is checked on the board.
"""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
LINEAR = (ROOT / "src/ops/rpu_linear.cpp").read_text(encoding="utf-8")


GRAPH_H = (ROOT / "src/graph/graph_runtime.h").read_text(encoding="utf-8")
GRAPH_CPP = (ROOT / "src/graph/graph_runtime.cpp").read_text(encoding="utf-8")


def _batch_class():
    h_start = GRAPH_H.index("class ScopedEagerAtomicBatch {")
    h_decl = GRAPH_H[h_start:GRAPH_H.index("\n};", h_start) + 3]
    cpp_start = GRAPH_CPP.index("ScopedEagerAtomicBatch::ScopedEagerAtomicBatch(")
    return h_decl + "\n" + GRAPH_CPP[cpp_start:] + "\nusing EagerLinearBatch = ScopedEagerAtomicBatch;\n"


def _launcher():
    start = LINEAR.index("void rpu_launch_linear_ddr_kernel(")
    return LINEAR[start:LINEAR.index("\n}\n", start)]


def test_batch_scope_is_eager_col_only_and_submits_after_the_loop():
    body = _launcher()
    guard = body.index("EagerLinearBatch eager_batch(")
    assert (
        "partition != 0 && !RpuKernelGraph::has_active(),\n"
        "      num_cores);"
    ) in body
    # SPM temporaries are scoped before the batch, so they outlive end().
    assert body.index("SpmAllocator::ScopedTemporary spm_scope;") < guard
    # col_scatter / col_scatter_rows are chosen from the eager state, before recording starts.
    assert body.index("bool col_scatter = false;") < guard
    assert body.index("bool col_scatter_rows = false;") < guard
    # Every device step of the call is inside the scope.
    assert guard < body.index("uint32_t b_addr = 0;")
    loop = body.index("  for (size_t row0 = 0; row0 < m; row0 += m_chunk) {")
    submit = body.index("    eager_batch.submit();")
    assert guard < loop < submit
    # Immediate fixed write-backs post-flush; keep that after the batch.
    assert "rpu_ddr_flush(yp + row0 * n);" in body[submit:]


def test_batch_scope_aborts_on_every_failure_path(tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    program = r'''
#include <array>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#define NUM_CORES 8
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("check"); } while (0)
struct LknBatchCfg { uint32_t max_entries = 65536; };
LknBatchCfg rpu_lkn_batch_config_at_load() { return {}; }
struct GraphRuntimePolicy { int execution_core_count = 8; };
std::vector<std::string> events;
bool fail_end = false, fail_abort = false;
struct RpuKernelGraph {
  GraphRuntimePolicy policy; bool open = false;
  explicit RpuKernelGraph(GraphRuntimePolicy p) : policy(p) {}
  size_t recorded_node_count() const noexcept { return 0; }
  void begin() { assert(!open); open = true; events.push_back("begin" + std::to_string(policy.execution_core_count)); }
  void end() { assert(open); events.push_back("end"); if (fail_end) throw std::runtime_error("end"); open = false; }
  void abort() { events.push_back("abort"); open = false; if (fail_abort) throw std::runtime_error("abort"); }
};
''' + _batch_class() + r'''
int main() {
  { EagerLinearBatch off(false, 4); assert(!off.active()); }
  assert(events.empty());

  { EagerLinearBatch b(true, 4); assert(b.active()); b.submit(); assert(!b.active()); }
  assert((events == std::vector<std::string>{"begin4", "end"}));

  events.clear();
  try { EagerLinearBatch b(true, 6); throw std::logic_error("step"); }
  catch (const std::logic_error&) {}
  assert((events == std::vector<std::string>{"begin6", "abort"}));

  events.clear(); fail_end = true; fail_abort = true;
  try { EagerLinearBatch b(true, 8); b.submit(); assert(false); }
  catch (const std::runtime_error& e) { assert(std::string(e.what()) == "end"); }
  assert((events == std::vector<std::string>{"begin8", "end", "abort"}));

  events.clear(); fail_end = false; fail_abort = false;
  { EagerLinearBatch b(true, 4); b.submit(); }
  assert((events == std::vector<std::string>{"begin4", "end"}));
  return 0;
}
'''
    src = tmp_path / "eager_batch.cpp"
    exe = tmp_path / "eager_batch"
    src.write_text(program, encoding="utf-8")
    subprocess.run([compiler, "-std=c++17", "-O0", str(src), "-o", str(exe)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(exe)], check=True, capture_output=True, text=True)
