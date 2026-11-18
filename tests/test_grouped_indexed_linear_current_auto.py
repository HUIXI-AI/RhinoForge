"""Actual C++ selection versus frozen upstream decisions, with no RPU calls."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_qwen35_moe_core_native import _definition

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def current_selector(tmp_path_factory):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("current_fused_selector")
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    enum = _definition(cache, "enum class KernelId :") + ";"
    header = (ROOT / "src/ops/rpu_grouped_indexed_linear.h").read_text()
    types = header.split("namespace rpu_grouped_indexed_linear {", 1)[1].split(
        "} // namespace rpu_grouped_indexed_linear", 1)[0]
    source = (ROOT / "src/ops/rpu_grouped_indexed_linear.cpp").read_text()
    definitions = _definition(source, "const gil::Tile kGilTiles[]").replace("gil::", "") + ";\n"
    definitions += "\n".join(_definition(source, marker) for marker in (
        "void validate_shape(", "Selection canonical(", "const Tile &selected_tile(",
        "SelectionResult pick(", "SelectionResult pick_current_auto("))
    measurements = (ROOT / "src/core/rpu_grouped_indexed_linear_measurements.inc").read_text()
    code = r'''
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"
#include <array>
#include <climits>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#define TORCH_CHECK(c,...) if (!(c)) { throw std::runtime_error("rejected"); }
__ENUM__
namespace rpu_grouped_indexed_linear {
__TYPES__
__DEFINITIONS__
}
namespace gil=rpu_grouped_indexed_linear;
__MEASUREMENTS__
int installation=0;
bool available(KernelId id) {
 if(installation==2)return false;
 if(installation==3)return id==KernelId::GPL_INDEXED_W4_M64N32K768;
 return installation!=1 || id<=KernelId::PI05_NVFP4_GEGLU_DB_TSCALE;
}
int main() {
 // Every new identity follows every old one, including the trailing Pi0.5 IDs.
#define GIL(indexed,dtype,id,symbol,m,n,k) static_assert(KernelId::id>KernelId::PI05_NVFP4_GEGLU_DB_TSCALE);
#include "rpu_grouped_indexed_linear_current_tiles.inc"
#undef GIL
#define GIL(indexed,dtype,id,symbol,m,n,k) static_assert(KernelId::id<KernelId::PI05_KV_OWNER_W8_DB);
#include "rpu_grouped_indexed_linear_tiles.inc"
#undef GIL
 int indexed,mode,policy;gil::Shape s{};
 while(std::cin>>indexed>>mode>>s.T>>s.R>>s.E>>s.N>>s.K>>s.Q>>s.TP>>installation>>policy) {
  s.indexed=indexed;s.weight_type=static_cast<gil::WeightType>(mode);s.col_parallel=indexed;
  try {
   auto chosen=policy ? gil::pick_current_auto(s,available)
                      : gil::pick(s,kGilMeasurements,kGilMeasurementCount,available);
   const auto t=chosen.selection;
   const auto& payload=gil::selected_tile(s,t);
   if(!available(payload.id))throw std::runtime_error("not installed");
   std::cout<<t.m<<' '<<t.n<<' '<<t.k<<' '<<t.expert_partitions<<'\n';
  } catch(const std::exception&) { std::cout<<"rejected\n"; }
 }
}
'''.replace("__ENUM__", enum).replace("__TYPES__", types).replace(
        "__DEFINITIONS__", definitions).replace("__MEASUREMENTS__", measurements)
    cpp, executable = directory / "select.cpp", directory / "select"
    cpp.write_text(code)
    subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"),
                    str(cpp), "-o", str(executable)], check=True)

    def run(rows):
        raw = "\n".join(" ".join(map(str, row)) for row in rows) + "\n"
        result = subprocess.run([str(executable)], input=raw, text=True, capture_output=True, check=True)
        return [None if row == "rejected" else list(map(int, row.split()))
                for row in result.stdout.splitlines()]
    return run


def test_current_auto_matches_upstream_with_installed_filtering(current_selector):
    fixture = json.loads((ROOT / "tests/fixtures/grouped_indexed_linear_current_auto.json").read_text())
    rows = [[int(c["indexed"]), c["mode"], c["T"], c["R"], c["E"], c["N"], c["K"],
             c["Q"], c["TP"], int(c["installation"] == "retained"), 1] for c in fixture["cases"]]
    assert current_selector(rows) == [c.get("expected") for c in fixture["cases"]]


def test_current_auto_no_available_payload_and_n16_base_guard(current_selector):
    indexed = [1, 8, 64, 512, 256, 64, 2048, 128, 8]
    assert current_selector([indexed + [2, 1], indexed + [3, 1]]) == [None, [64, 32, 768, 8]]
    boundary = [1, 2, 1, 1, 1, 2**20, 32, 0, 1]
    assert current_selector([boundary + [0, 1]])[0] is not None
    boundary[5] += 16
    assert current_selector([boundary + [0, 1], boundary + [0, 0]]) == [None, None]


def test_existing_model_policy_retains_measured_schedule(current_selector):
    rows, expected = [], []
    for mode in (8, 5):
        for t in (64, 128, 256, 384, 512, 768):
            for indexed in (1, 0):
                rows.append([indexed, mode, t, min(t * 8, 2048) if indexed else t * 8,
                             256, 64 if indexed else 2048, 2048 if indexed else 64,
                             (128 if indexed else 64) if mode == 8 else 0, 8, 0, 0])
                if mode == 8 and t >= 256:
                    expected.append([224, 64, 256, 8] if indexed else [224, 256, 64, 1])
                else:
                    expected.append([256, 32, 256, 8] if indexed else [256, 128, 64, 1])
    assert current_selector(rows) == expected
