"""Frozen seven-profile census, generated C++ and cache-name closure."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("census_codegen",
    ROOT / "scripts/generate_parallel_linear_register_policies.py")
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


def data():
    return json.loads((ROOT / "tests/fixtures/parallel_linear_register_census.v1.json").read_text())


def test_generated_policies_are_exact_frozen_artifact():
    assert GEN.render(data()) == (ROOT / "src/fused/rpu_parallel_linear_register_policies.h").read_text()


@pytest.mark.parametrize("mutation", ("count", "operand", "phase", "duplicate", "dynamic"))
def test_generator_rejects_invalid_authority(mutation):
    profile = deepcopy(next(iter(data()["profiles"].values())))
    if mutation == "count":
        profile["rules"][0]["count"] += 1
    elif mutation == "operand":
        next(r for r in profile["rules"] if r["operands"])["operands"][0][3] = 3
    elif mutation == "phase":
        profile["phases"][0]["kernel_count"] += 1
    elif mutation == "duplicate":
        profile["rules"].append(deepcopy(profile["rules"][0]))
    else:
        profile["rules"][0]["dynamic"] = not profile["rules"][0]["dynamic"]
    with pytest.raises(AssertionError):
        GEN.validate_profile(profile)


def test_generated_native_policies_bind_actual_cache_and_role_counts(tmp_path):
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    start = cache.index("static constexpr const char* KERNEL_ID_NAMES[] = {")
    stop = cache.index("\n};", start) + 3
    table = subprocess.check_output(["c++", "-E", "-P", "-x", "c++",
        "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"), "-"],
        input=cache[start:stop].encode()).decode()
    graph = (ROOT / "src/graph/graph_runtime.h").read_text()
    declarations = []
    for kind, names in (
        ("enum class", ("GraphDdrRegisterRole", "GraphDdrRegisterAccess", "GraphDdrRegisterEncoding",
                        "GraphKernelRegisterRuleKind", "GraphKernelNoDdrProof")),
        ("struct", ("GraphDdrRegisterAbi", "GraphKernelRegisterRule", "GraphKernelRegisterCensusStats",
                    "GraphKernelRegisterPhaseExpectation", "GraphKernelRegisterPolicy")),
    ):
        for name in names:
            declarations.append(re.search(rf"{kind} {name}(?: : \w+)? \{{.*?\n\}};", graph, re.S).group())
    code = """
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do {if(!(c)) throw std::runtime_error("rejected");}while(0)
enum class KernelId {};
"""
    code += "\n".join(declarations) + "\n" + table + "\n"
    header = GEN.render(data())
    code += "\n".join(line for line in header.splitlines() if not line.startswith(("#include", "#pragma")))
    code += "\nint main() {\n"
    for name, profile in data()["profiles"].items():
        code += f'{{ const auto& p=rpu_pl_census::policy("{name}");\n'
        for field in GEN.FIELDS:
            code += f"assert(p.expected_{field}==UINT64_C({profile['total'][field]}));\n"
        code += """
size_t count=0,operands=0,typed=0;
for(const auto&r:p.rules) {
 count+=r.expected_count;operands+=r.expected_count*r.operands.size();
 typed+=r.operands.empty()?0:r.expected_count;
 assert(r.kernel_id.has_value()!=(!r.kernel_name.empty()));
 if(r.kernel_id) assert(size_t(*r.kernel_id)<std::size(KERNEL_ID_NAMES));
}
assert(count==p.expected_kernel_count && operands==p.expected_operand_count && typed==p.expected_typed_kernel_count);
}\n"""
    code += 'bool rejected=false;try{rpu_pl_census::policy("unknown");}catch(const std::runtime_error&){rejected=true;}assert(rejected);\n}\n'
    cpp, binary = tmp_path / "policies.cpp", tmp_path / "policies"
    cpp.write_text(code)
    subprocess.run(["c++", "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
