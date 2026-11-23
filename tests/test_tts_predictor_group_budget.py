"""Check the exact Predictor budget without loading the device backend."""
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_predictor_budget_is_owned_and_preserves_other_limits(tmp_path):
    source = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    infra = (ROOT / "src/graph/graph_infra.h").read_text()
    policy = infra.split("struct GraphRuntimePolicy {", 1)[1].split("\n};", 1)[0]
    budgets = source[source.index("struct SegmentResourceBudget {"):
                     source.index("static SegmentResourceUse segment_node_resources(")]
    cpp = tmp_path / "budget.cpp"
    cpp.write_text(r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
enum class GraphHostOpDeferMode {Auto};
enum class GraphBatchSnapshotMode {Auto};
''' + "struct GraphRuntimePolicy {" + policy + "\n};\n" + budgets + r'''
int main() {
 GraphRuntimePolicy p;
 const std::string name="qwen3_tts_predictor_group";
 assert(segment_resource_budget(name,p).entries==8192);
 p.tts_predictor_group_15=true;
 auto b=segment_resource_budget(name,p);
 assert(b.entries==16384 && b.kd_bytes==(4u<<20) && b.instr_bytes==(32u<<20));
 assert(segment_resource_budget("qwen3_tts_talker",p).entries==8192);
 p.execution_core_count=4;
 assert(segment_resource_budget(name,p).entries==4096);
 p.execution_core_count=8;p.lkn_max_batch_entries=12000;
 p.lkn_kd_buf_mb=2;p.lkn_instr_buf_mb=8;
 b=segment_resource_budget(name,p);
 assert(b.entries==6000 && b.kd_bytes==(1u<<20) && b.instr_bytes==(4u<<20));
}
''')
    exe = tmp_path / "budget"
    subprocess.run(["g++", "-std=c++17", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
