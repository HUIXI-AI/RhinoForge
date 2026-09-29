"""Exercise actual loader and snapshots with opaque dummy assets and a Launch stub."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()


@pytest.fixture(scope="module")
def loader(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler is unavailable")
    directory = tmp_path_factory.mktemp("two-ref-loader")
    # Compile the real path resolution, preload selection, manifest validation,
    # snapshot loading, lazy lookup and profile preflight. Only Launch and the
    # unrelated tensor/fast-pointer publication sections are replaced.
    paths = CACHE[CACHE.index("static std::string get_kernel_lib_path()"):
                  CACHE.index("static void print_tensor_info")]
    impl = CACHE[CACHE.index("static bool is_optional_kernel"):
                 CACHE.index("    // 填充快速缓存数组")]
    impl += "    initialized_ = true;\n}\n"
    impl += CACHE[CACHE.index("void KernelCache::require_names"):
                  CACHE.index("void KernelCache::populate_fast_cache")]
    impl += CACHE[CACHE.index("::rhino_lkn::Kernel_t* KernelCache::get_kernel"):
                  CACHE.index("::rhino_lkn::Program_t* KernelCache::get_program")]
    binding = (ROOT / "src/core/rpu_pybind.inc").read_text()
    roles_binding = binding[binding.index("    py::dict oplib_roles;"):
                            binding.index('    result["oplib_roles"]')]
    source = directory / "loader.cpp"
    source.write_text(r'''
#include "rpu_kernel_manifest.h"
#include "rpu_oplib_snapshot.h"
#include "rpu_linear_tiling.h"
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>
template<class... Args> void check(bool ok, Args&&... args) {
    if (!ok) { std::ostringstream os; (os << ... << args); throw std::runtime_error(os.str()); }
}
#define TORCH_CHECK(...) check(__VA_ARGS__)
bool log_at(int) { return false; }
std::unordered_map<std::string,std::string> loaded;
namespace rhino_lkn {
struct Program_t {
    int32_t create_with_binary_file(const std::string& path, const char* name) {
        std::ifstream file(path, std::ios::binary);
        loaded[name] = std::string(std::istreambuf_iterator<char>(file), {});
        const char* fail = std::getenv("FAIL_KERNEL");
        return fail && std::string(fail) == name ? 1 : 0;
    }
};
struct Kernel_t { Kernel_t(Program_t&, const char*) {} };
}
struct CachedKernel {
    std::unique_ptr<rhino_lkn::Program_t> program;
    std::unique_ptr<rhino_lkn::Kernel_t> kernel;
};
struct KernelCache {
    bool initialized_ = false;
    mutable std::mutex loaded_oplib_paths_mutex_;
    std::set<std::string> loaded_oplib_paths_;
    std::unordered_map<std::string,std::unordered_set<std::string>> oplib_symbols_;
    std::unordered_map<std::string,std::shared_ptr<OplibSnapshot>> oplib_snapshots_;
    std::unordered_map<std::string,CachedKernel> kernels_;
    const std::unordered_set<std::string>& kernel_manifest_names(const std::string&);
    std::shared_ptr<OplibSnapshot> oplib_snapshot(const std::string&);
    CachedKernel load_kernel_from_oplib(const std::string&,const std::string&);
    std::vector<std::string> loaded_oplib_paths() const;
    std::vector<std::pair<std::string,std::string>> loaded_oplib_bytes() const;
    void initialize();
    void require_names(const std::vector<std::string>&);
    rhino_lkn::Kernel_t* get_kernel(const std::string&);
};
''' + paths + impl + '''
namespace py { using dict = std::unordered_map<std::string,std::string>; }
py::dict identity_roles() {
''' + roles_binding + '''
    return oplib_roles;
}
''' + r'''
int main(int argc, char** argv) {
    try {
        if (std::string(argv[1]) == "inventory") {
            for (const auto& entry : KERNEL_LIST)
                std::cout << (entry.second == KERNEL_LIB_PATH_EXPANSION() ? "expansion:" : "main:")
                          << entry.first << "\n";
            for (const auto& name : rpu_pl_tiling::autotile_kernel_names())
                std::cout << (name.rfind("parallel_linear_wint4a16_pgrp_",0) == 0 ? "expansion:" : "main:")
                          << name << "\n";
            return 0;
        }
        KernelCache cache;
        cache.initialize();
        const auto paths = cache.loaded_oplib_paths();
        TORCH_CHECK(paths.size() == 2, "expected exactly two assets");
        const auto roles = identity_roles();
        TORCH_CHECK(roles.size() == 2 && roles.at("main") == FROZEN_KERNEL_LIB_PATH() &&
                    roles.at("expansion") == FROZEN_KERNEL_LIB_PATH_EXPANSION(),
                    "runtime identity must report main/expansion roles");
        const std::set<std::string> role_paths = {roles.at("main"), roles.at("expansion")};
        TORCH_CHECK(role_paths == std::set<std::string>(paths.begin(), paths.end()),
                    "runtime identity disagrees with loaded assets");
        for (const auto& name : rpu_pl_tiling::autotile_kernel_names()) {
            const bool w4 = name.rfind("parallel_linear_wint4a16_pgrp_",0) == 0;
            TORCH_CHECK(loaded.at(name) == (w4 ? "expansion" : "main"), "wrong autotile ABI role");
        }
        if (std::string(argv[1]) == "replace") {
            const auto before = cache.loaded_oplib_bytes();
            for (const auto& entry : before) {
                std::ofstream(entry.first) << "replaced";
                std::filesystem::remove(entry.first + ".kernels");
            }
            setenv("RPU_KERNEL_LIB_PATH", "/nonexistent/third.ref", 1);
            setenv("RPU_KERNEL_LIB_PATH_EXPANSION", "/nonexistent/fourth.ref", 1);
            TORCH_CHECK(cache.get_kernel("lazy_main") != nullptr, "lazy main missing");
            TORCH_CHECK(cache.get_kernel("lazy_expansion") != nullptr, "lazy expansion missing");
            TORCH_CHECK(loaded.at("lazy_main") == "main", "main snapshot drift");
            TORCH_CHECK(loaded.at("lazy_expansion") == "expansion", "expansion snapshot drift");
            TORCH_CHECK(before == cache.loaded_oplib_bytes(), "identity drift");
        } else if (argc > 2) {
            cache.require_names({argv[2]});
        }
        TORCH_CHECK(cache.get_kernel("third_asset_only") == nullptr, "unexpected third asset");
        std::cout << "PASS";
    } catch (const std::exception& e) {
        std::cerr << e.what() << " loaded=" << loaded.size();
        return 2;
    }
}
''')
    executable = directory / "loader"
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "src/core"),
                    "-I", str(ROOT / "src/ops"), str(source), "-ldl", "-pthread",
                    "-o", str(executable)], check=True, capture_output=True, text=True)
    env = os.environ.copy()
    env.pop("RPU_KERNEL_LIB_PATH", None)
    env.pop("RPU_KERNEL_LIB_PATH_EXPANSION", None)
    inventory = subprocess.check_output([str(executable), "inventory"], env=env, text=True)
    roles = {"main": set(), "expansion": set()}
    for line in inventory.splitlines():
        role, name = line.split(":")
        roles[role].add(name)
    mixed = CACHE.split("kMixedNames[] = {", 1)[1].split("};", 1)[0]
    roles["expansion"].update(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', mixed))
    roles["main"].add("lazy_main")
    roles["expansion"].add("lazy_expansion")
    # Main ships a different W4 ABI under these names: it must never win.
    roles["main"].update(n for n in roles["expansion"] if n.startswith("parallel_linear_wint4a16_pgrp_"))
    return executable, roles, env


def run_loader(loader, tmp_path, *, edit=None, command="check", required=None,
               same=False, swap=False, default_expansion=False, fail=None):
    executable, inventory, env = loader
    env = env.copy()
    roles = {role: set(names) for role, names in inventory.items()}
    if edit:
        edit(roles)
    paths = {"main": tmp_path / "rhinoOpLib_current.ref",
             "expansion": tmp_path / "rhinoExpansionOpLib_current.ref"}
    for role, path in paths.items():
        path.write_text(role)
        Path(str(path) + ".kernels").write_text(
            f"rhinoforge-kernels-v1\nasset-size={len(role)}\n" +
            "\n".join(sorted(roles[role])) + "\n")
    env["RPU_KERNEL_LIB_PATH"] = str(paths["expansion" if swap else "main"])
    env["RPU_KERNEL_LIB_PATH_EXPANSION"] = str(paths["main" if same or swap else "expansion"])
    if default_expansion:
        del env["RPU_KERNEL_LIB_PATH_EXPANSION"]
    if fail:
        env["FAIL_KERNEL"] = fail
    args = [str(executable), command]
    if required:
        args.append(required)
    return subprocess.run(args, env=env, capture_output=True, text=True, check=False)


def test_two_roles_and_default_sibling_path(loader, tmp_path):
    result = run_loader(loader, tmp_path, default_expansion=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "PASS"


@pytest.mark.parametrize("same,swap,error", [(True, False, "same payload"),
                                              (False, True, "role mismatch")])
def test_role_errors_precede_program_loading(loader, tmp_path, same, swap, error):
    result = run_loader(loader, tmp_path, same=same, swap=swap)
    assert result.returncode == 2
    assert error in result.stderr
    assert "loaded=0" in result.stderr


def test_signed_w4_cannot_fall_back_to_main(loader, tmp_path):
    result = run_loader(loader, tmp_path, edit=lambda roles: roles["expansion"].difference_update(
        n for n in list(roles["expansion"]) if n.startswith("parallel_linear_wint4a16_pgrp_")))
    assert result.returncode == 2
    assert "autotile ref mismatch" in result.stderr
    assert "expansion:parallel_linear_wint4a16_pgrp_" in result.stderr


@pytest.mark.parametrize("fail_load", [False, True])
def test_optional_profile_kernel_fails_preflight(loader, tmp_path, fail_load):
    name = "pi05_prefill_o_weight_outer_fp16_c336x2_m128n80k128"
    result = run_loader(loader, tmp_path, required=name,
                        fail=name if fail_load else None,
                        edit=None if fail_load else lambda roles: roles["expansion"].remove(name))
    assert result.returncode == 2
    assert "before loading weights" in result.stderr
    assert ("could not be loaded" if fail_load else "requires operator") in result.stderr


def test_optional_missing_does_not_block_other_profiles(loader, tmp_path):
    result = run_loader(loader, tmp_path, edit=lambda roles: roles["expansion"].remove(
        "pi05_prefill_o_weight_outer_fp16_c336x2_m128n80k128"))
    assert result.returncode == 0, result.stderr


def test_two_payload_snapshots_and_paths_stay_frozen(loader, tmp_path):
    result = run_loader(loader, tmp_path, command="replace")
    assert result.returncode == 0, result.stderr
