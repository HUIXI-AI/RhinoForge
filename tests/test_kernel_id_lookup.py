"""Run real host lookup expressions across the former 8-bit ID boundary."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_kernel_cache_and_graph_names_do_not_truncate_ids(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    begin = header.index("enum class KernelId")
    enum = header[begin : header.index("};", begin) + 2]
    begin = header.index("static constexpr const char* KERNEL_ID_NAMES[]")
    names = header[begin : header.index("};", begin) + 2]
    cache_get = _function(header, "inline ::rhino_lkn::Kernel_t* get(KernelId id)")
    program_get = _function(header, "inline ::rhino_lkn::Program_t* get_program(KernelId id)")
    graph_lookups = []
    for filename in ("graph_runtime_execute.cpp", "graph_runtime_child.cpp",
                     "graph_register_census.cpp"):
        source = (ROOT / "src/graph" / filename).read_text()
        graph_lookups.extend(re.findall(
            r"KERNEL_ID_NAMES\[\s*static_cast<[^>]+>\((?:id|\*node.kernel_id|\*rule.kernel_id)\)\]",
            source,
        ))
    assert graph_lookups, "missing graph name lookup expressions"
    checks = "\n".join(
        f"assert(std::string({lookup}) == KERNEL_ID_NAMES[index]);"
        for lookup in graph_lookups
    )
    harness = r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include "rpu_pi_wall_tiling.h"
''' + enum + "\n" + names + r'''
namespace rhino_lkn { struct Kernel_t {}; struct Program_t {}; }
struct Cache {
    rhino_lkn::Kernel_t* fast_cache_[static_cast<size_t>(KernelId::_COUNT)]{};
    rhino_lkn::Program_t program;
    std::string requested;
    rhino_lkn::Program_t* get_program(const std::string& name) {
        requested = name;
        return &program;
    }
''' + cache_get + "\n" + program_get + r'''
};
int main() {
    constexpr size_t count = static_cast<size_t>(KernelId::_COUNT);
    static_assert(count == sizeof(KERNEL_ID_NAMES) / sizeof(KERNEL_ID_NAMES[0]));
    static_assert(count > 256, "this regression exercises IDs beyond one byte");
    rhino_lkn::Kernel_t kernels[count];
    Cache cache;
    for (size_t index = 0; index < count; ++index) cache.fast_cache_[index] = &kernels[index];
    for (size_t index = 0; index < count; ++index) {
        const auto id = static_cast<KernelId>(index);
        struct { std::optional<KernelId> kernel_id; } node{id}, rule{id};
        assert(cache.get(id) == &kernels[index]);
        assert(cache.get_program(id) == &cache.program);
        assert(cache.requested == KERNEL_ID_NAMES[index]);
''' + checks + r'''
    }
}
'''
    cpp, executable = tmp_path / "kernel_ids.cpp", tmp_path / "kernel_ids"
    cpp.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"), str(cpp), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


def test_generated_and_named_capabilities_remain_independent(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    enum = _function(header, "enum class KernelId") + ";"
    availability = _function(header, "class KernelPayloadAvailability final") + ";"
    harness = r'''
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include "rpu_pi_wall_tiling.h"
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
''' + enum + "\n" + availability + r'''
int main() {
    KernelPayloadAvailability payload;
    const std::array<KernelId, 7> optional{
        KernelId::PL_U_FP16_ACC16_M1584N32K32,
        KernelId::PI_WALL_QKV_W8A16_ACC16_M464N64K128_256X128X128,
        KernelId::GROUPED_PL_W8_ACC32_N48_DEVICE_M,
        KernelId::PI05_ADARMS_NORM_SHIFT_H1024,
        KernelId::PI05_GATED_RESIDUAL_H1024,
        KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64,
        KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48};
    payload.publish(false, false);
    for (const auto id : optional) assert(!payload.has_loaded(id));
    // Generated tiles/device-M use the loaded-ID set; named capabilities keep
    // their independent flags. Publishing either must not enable its neighbours.
    for (size_t selected = 0; selected < optional.size(); ++selected) {
        KernelPayloadAvailability::ExtendedPayloads loaded;
        if (selected < 3)
            loaded.set(static_cast<size_t>(optional[selected]) -
                       KernelPayloadAvailability::kExtendedBegin);
        payload.publish(false, false, false, false, false, false, loaded,
                        false, false, false, false, false, false, false, false,
                        selected == 3, selected == 4, selected == 5, selected == 6);
        for (size_t i = 0; i < optional.size(); ++i)
            assert(payload.has_loaded(optional[i]) == (i == selected));
    }
    payload.publish(false, false);
    for (const auto id : optional) assert(!payload.has_loaded(id));
    payload.withdraw();
    bool rejected = false;
    try { payload.has_loaded(optional[0]); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
'''
    cpp, executable = tmp_path / "capabilities.cpp", tmp_path / "capabilities"
    cpp.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "src/core"), "-I", str(ROOT / "src/ops"), str(cpp), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
