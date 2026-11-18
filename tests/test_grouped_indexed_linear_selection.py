"""Host geometry-policy regressions using artificial costs, never tuning data."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_qwen35_moe_core_native import _definition

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def selection_runner(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host geometry selector requires g++")
    directory = tmp_path_factory.mktemp("generic_selection")
    # Explicit costs exercise ranking/ties and dimension bounds; none originate
    # from hardware, and none are compiled into the production measurement table.
    entries = []

    def entry(shape, ranked, *, indexed=True, dtype="uint4", kt=256):
        entries.append(dict(family="indexed" if indexed else "weighted_scatter_add",
                            dtype=dtype, shape=shape, k_tile=kt, ranked=ranked))

    entry([256, 2048, 256, 64, 2048, 8, 128],
          [[64, 32, 256, 1.0], [256, 32, 256, 1.0], [64, 64, 256, .7]])
    entry([256, 2048, 256, 64, 2048, 4, 128], [[256, 32, 256, .01]])
    entry([256, 2048, 256, 64, 2048, 8, 64], [[256, 32, 256, .01]])
    entry([256, 2048, 256, 64, 2048, 8, 0],
          [[64, 64, 256, .01], [256, 32, 256, .02]], dtype="int8")
    entry([128, 128, 1, 64, 2048, 8, 128],
          [[64, 32, 256, .8], [256, 32, 256, 1.0]])
    entry([128, 1024, 128, 96, 2048, 8, 128], [[64, 64, 256, 1.0]])
    entry([256, 2048, 256, 128, 2048, 8, 128], [[256, 32, 256, .1]])
    for n in (224, 256):
        entry([256, 2048, 256, n, 2048, 8, 128], [[256, 32, 256, 1.0]])
    entry([256, 2048, 256, 2048, 64, 8, 64], [[192, 256, 64, .5]],
          indexed=False, kt=64)
    entry([256, 2048, 256, 2048, 128, 8, 64], [[224, 64, 256, .5]], indexed=False)
    data = dict(schema_version=1, entries=entries)
    (directory / "artificial_table.json").write_text(json.dumps(data, indent=2))
    declarations, rows = [], []
    for index, row in enumerate(entries):
        ranked = ",".join("{" + ",".join(map(str, values)) + "}" for values in row["ranked"])
        declarations.append(f"const gil::RankedTile ranks{index}[] = {{{ranked}}};")
        shape = ",".join(map(str, row["shape"]))
        rows.append("{" + f'{str(row["family"] == "indexed").lower()},'
                    f'gil::WeightType::{"Uint4" if row["dtype"] == "uint4" else "Int8"},'
                    + "{" + shape + "},"
                    + f'{row["k_tile"]},ranks{index},{len(row["ranked"])}' + "}")
    declarations.append("const gil::Measurement table[] = {" + ",".join(rows) + "};")
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    begin = cache.index("enum class KernelId :")
    enum = cache[begin:cache.index("\n};", begin) + 3]
    header = (ROOT / "src/ops/rpu_grouped_indexed_linear.h").read_text()
    types = header.split("namespace rpu_grouped_indexed_linear {", 1)[1].split(
        "} // namespace rpu_grouped_indexed_linear", 1)[0]
    production = (ROOT / "src/ops/rpu_grouped_indexed_linear.cpp").read_text()
    definitions = _definition(production, "const gil::Tile kGilTiles[]").replace("gil::", "") + ";\n"
    definitions += "\n".join(_definition(production, marker) for marker in (
        "void validate_shape(", "Selection canonical(", "SelectionResult pick("))
    code = r'''
#include <algorithm>
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>
// Match c10's bare-if macro; do/while would hide unbraced if/else defects.
#define TORCH_CHECK(c,...) if (!(c)) { throw std::runtime_error("rejected"); }
__ENUM__
namespace rpu_grouped_indexed_linear {
__TYPES__
__DEFINITIONS__
}
namespace gil=rpu_grouped_indexed_linear;
__ROWS__
int unavailable_mode=0;
bool available(KernelId id){
 if(unavailable_mode==2)return false;
 return unavailable_mode!=1||(id!=KernelId::GPL_INDEXED_W4_M64N64K256&&
                              id!=KernelId::GPL_INDEXED_W8_M64N64K256);
}
int main(){int indexed,dtype;gil::Shape s{};
 while(std::cin>>indexed>>dtype>>s.T>>s.R>>s.E>>s.N>>s.K>>s.TP>>s.Q>>unavailable_mode){
  s.indexed=indexed;s.weight_type=static_cast<gil::WeightType>(dtype);s.col_parallel=indexed;
  auto selected=gil::pick(s,table,sizeof(table)/sizeof(table[0]),available);
  const auto t=selected.selection;
  std::cout<<t.m<<' '<<t.n<<' '<<t.k<<' '<<t.expert_partitions<<' '<<int(selected.source)<<'\n';
 }
}
'''.replace("__ENUM__", enum).replace("__TYPES__", types).replace(
        "__DEFINITIONS__", definitions).replace("__ROWS__", "\n".join(declarations))
    cpp, executable = directory / "selector.cpp", directory / "selector"
    cpp.write_text(code)
    compiled = subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"),
                               str(cpp), "-o", str(executable)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr

    def run(shapes):
        text = "\n".join(" ".join(map(str, shape)) for shape in shapes) + "\n"
        result = subprocess.run([str(executable)], input=text, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        return [tuple(map(int, line.split())) for line in result.stdout.splitlines()]

    return run


def test_exact_missing_candidate_tie_and_after_selection_expert_partition(selection_runner):
    # Columns: indexed,dtype,T,R,E,physicalN,physicalK,TP,Q,unavailable_mode.
    shapes = [
        (1, 8, 256, 2048, 256, 64, 2048, 8, 128, 0),
        (1, 8, 256, 2048, 256, 64, 2048, 8, 128, 1),
        (1, 8, 256, 2048, 256, 64, 2048, 8, 128, 2),
        (1, 8, 256, 2048, 256, 224, 2048, 8, 128, 0),
        (1, 8, 256, 2048, 256, 256, 2048, 8, 128, 0),
    ]
    assert selection_runner(shapes) == [
        (64, 64, 256, 8, 1), (256, 32, 256, 8, 1),
        (256, 32, 256, 8, 0), (256, 32, 256, 8, 1), (256, 32, 256, 1, 1),
    ]


def test_neighbor_same_nk_priority_scaling_and_bounded_fallback(selection_runner):
    shapes = [
        (1, 8, 257, 2048, 256, 64, 2048, 8, 128, 0),
        # M64 wins at the reference R128 but its extra M pass at R256 loses.
        (1, 8, 256, 256, 1, 64, 2048, 8, 128, 0),
        # Same N/K wins over a closer measured row with a different N.
        (1, 8, 256, 2048, 256, 96, 2048, 8, 128, 0),
        (1, 8, 2048, 2048, 256, 64, 2048, 8, 128, 0),
        (1, 8, 256, 2048, 256, 64, 2048, 2, 128, 0),
        (1, 8, 256, 2048, 256, 64, 2048, 8, 32, 0),
        (0, 8, 256, 2048, 256, 2048, 64, 8, 64, 0),
        (0, 8, 256, 2048, 256, 2048, 96, 8, 32, 0),
    ]
    assert selection_runner(shapes) == [
        (64, 64, 256, 8, 2), (256, 32, 256, 8, 2), (64, 64, 256, 8, 2),
        (256, 32, 256, 8, 0), (256, 32, 256, 8, 0), (256, 32, 256, 8, 0),
        (192, 256, 64, 1, 1), (224, 64, 256, 1, 0),
    ]


def test_e4m3_borrows_int8_policy_without_claiming_fp8_measurements(selection_runner):
    # E4M3 temporarily uses INT8's exact/neighbor geometry policy. These fixture
    # costs are artificial INT8 rows, not E4M3 measurements. Launch tests
    # separately verify mode 5 and raw-U8 owner provenance remain unchanged.
    assert selection_runner([
        (1, 1, 256, 2048, 256, 64, 2048, 8, 0, 0),
        (1, 5, 256, 2048, 256, 64, 2048, 8, 0, 0),
        (1, 5, 257, 2048, 256, 64, 2048, 8, 0, 0),
        (1, 5, 256, 2048, 256, 64, 2048, 8, 0, 1),
        (1, 5, 256, 2048, 256, 64, 2048, 8, 0, 2),
        (1, 5, 2048, 2048, 256, 64, 2048, 8, 0, 0),
        (0, 5, 256, 2048, 256, 2048, 64, 8, 0, 0),
        (0, 5, 256, 2048, 256, 2048, 288, 8, 0, 0),
    ]) == [
        (64, 64, 256, 8, 1),
        (64, 64, 256, 8, 1),
        (64, 64, 256, 8, 2),
        (256, 32, 256, 8, 1),
        (256, 32, 256, 8, 0),
        (256, 32, 256, 8, 0),
        (256, 128, 64, 1, 0),
        (224, 64, 256, 1, 0),
    ]
