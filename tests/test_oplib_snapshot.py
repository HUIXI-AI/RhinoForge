"""Compile the real snapshot and load/publication bodies without Torch or SDK."""
from pathlib import Path
import shutil
import hashlib
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_sealed_snapshot_and_success_only_program_publication(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
    begin = source.index("std::shared_ptr<OplibSnapshot> KernelCache::oplib_snapshot")
    end = source.index("void KernelCache::initialize()", begin)
    actual_bodies = source[begin:end]
    # Substitute only compiled audit metadata for a synthetic reviewed library.
    # The actual hash/load/publication/lookup bodies have no runtime trust setter.
    metadata = (ROOT / "src/core/rpu_graph_program_contracts.h").read_text()
    fixture_digest = hashlib.sha256(b"loaded-ref-A").hexdigest()
    metadata = re.sub(
        r"inline constexpr std::array<Entry, \d+> kAuditedPrograms = .*?;",
        'inline constexpr std::array<Entry, 1> kAuditedPrograms = {{{"' +
        fixture_digest + '", "first", 3}}};', metadata, flags=re.DOTALL,
    )
    metadata_path = tmp_path / "src/core/rpu_graph_program_contracts.h"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(metadata)

    harness = r'''
#include <cassert>
#include <atomic>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>
std::string mutate_source;
ssize_t snapshot_test_pread(int fd, void* out, size_t size, off_t offset) {
    const auto result = ::pread(fd, out, size, offset);
    if (!mutate_source.empty()) {
        std::ofstream(mutate_source, std::ios::binary) << "different-size-mutated-source";
        mutate_source.clear();
    }
    return result;
}
#define pread snapshot_test_pread
#include "src/core/rpu_oplib_snapshot.h"
#include "src/core/rpu_sha256.h"
#include "src/core/rpu_graph_program_contracts.h"
#undef pread
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("load failed"); } while (0)
std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
bool fail_program = false, fail_kernel = false;
namespace rhino_lkn {
struct Program_t {
    std::string bytes;
    int create_with_binary_file(const std::string& path, const char*) {
        bytes = read_file(path);
        return fail_program ? -1 : 0;
    }
};
struct Kernel_t {
    Kernel_t(Program_t&, const char*) { if (fail_kernel) throw std::runtime_error("kernel failed"); }
};
}
struct CachedKernel {
    std::unique_ptr<rhino_lkn::Program_t> program;
    std::unique_ptr<rhino_lkn::Kernel_t> kernel;
    uint32_t graph_snapshot_contracts = 0;
};
struct KernelCache {
    const std::set<std::string>& kernel_manifest_names(const std::string&) const {
        static const std::set<std::string> names={"first", "second", "broken", "third"};
        return names;
    }
    bool initialized_ = true;
    mutable std::mutex loaded_oplib_paths_mutex_;
    std::set<std::string> loaded_oplib_paths_;
    std::unordered_map<std::string, std::shared_ptr<OplibSnapshot>> oplib_snapshots_;
    std::unordered_map<std::string, std::string> oplib_sha256_;
    std::unordered_map<std::string, CachedKernel> kernels_;
    std::shared_ptr<const std::unordered_map<std::string, uint32_t>> graph_snapshot_contracts_;
    uint32_t graph_snapshot_program_contracts(const std::string&) const;
    std::shared_ptr<const std::unordered_map<std::string, uint32_t>>
        graph_snapshot_contracts_snapshot() const;
    std::shared_ptr<OplibSnapshot> oplib_snapshot(const std::string&);
    CachedKernel load_kernel_from_oplib(const std::string&, const std::string&);
    std::vector<std::string> loaded_oplib_paths() const;
    std::vector<std::pair<std::string, std::string>> loaded_oplib_bytes() const;
    enum class KernelIntent { kRequired, kOptional, kPending, kRejected };
    struct KernelResolution {
        std::string name;
        std::string role;
        KernelIntent intent;
        bool resolved;
    };
    std::vector<KernelResolution> kernel_resolution_;
    std::vector<KernelResolution> kernel_resolution() const;
};
''' + actual_bodies + r'''
int main(int argc, char** argv) {
    assert(argc == 2);
    using rpu_payload_identity::sha256;
    assert(sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(sha256("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    assert(sha256(std::string(1000000, 'a')) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    using namespace rpu_graph_program_contracts;
    const std::array<Entry, 2> reviewed = {{{
        "expected-ref", "first", kInstructionArenaReadOnly | kPositionIndependentPrograms}, {
        "expected-ref", "second", kInstructionArenaReadOnly}}};
    assert(detail::lookup("expected-ref", "first", reviewed) == 3);
    assert(detail::lookup("expected-ref", "second", reviewed) == 1);
    assert(detail::lookup("different-ref", "first", reviewed) == 0);
    assert(detail::lookup("expected-ref", "unknown", reviewed) == 0);
    assert(audited_contracts(sha256("loaded-ref-A"), "first") == 3);
    assert(audited_contracts(sha256("loaded-ref-B"), "first") == 0);

    const std::string path = std::string(argv[1]) + "/source.ref";
    std::ofstream(path, std::ios::binary) << "loaded-ref-A";
    KernelCache cache;
    auto first = cache.load_kernel_from_oplib(path, "first");
    assert(first.program->bytes == "loaded-ref-A");
    assert(first.graph_snapshot_contracts == 3);
    assert(cache.oplib_sha256_.at(path) == sha256("loaded-ref-A"));
    assert(cache.graph_snapshot_program_contracts("first") == 0);
    std::ofstream(path, std::ios::binary) << "current-ref-B";
    auto second = cache.load_kernel_from_oplib(path, "second");
    assert(second.program->bytes == "loaded-ref-A");
    assert(second.graph_snapshot_contracts == 0);
    assert(cache.oplib_sha256_.at(path) != sha256(read_file(path)));
    // Query reads one immutable publication, never mutable cache/file storage.
    cache.kernels_.emplace("first", std::move(first));
    assert(cache.graph_snapshot_contracts_snapshot()->at("first") == 3);
    std::atomic_store(&cache.graph_snapshot_contracts_, cache.graph_snapshot_contracts_snapshot());
    assert(cache.graph_snapshot_program_contracts("first") == 3);
    assert(cache.graph_snapshot_program_contracts("unknown") == 0);
    std::atomic_store(&cache.graph_snapshot_contracts_,
                     std::shared_ptr<const std::unordered_map<std::string, uint32_t>>{});
    assert(cache.graph_snapshot_program_contracts("first") == 0);
    assert(read_file(path) == "current-ref-B");
    auto receipt = cache.loaded_oplib_bytes();
    assert(receipt.size() == 1 && receipt[0].first == path && receipt[0].second == "loaded-ref-A");
    const auto snapshot = cache.oplib_snapshot(path);
    const int fd = ::open(snapshot->path().c_str(), O_RDWR);
    assert(fd >= 0);
    assert(::write(fd, "X", 1) == -1 && errno == EPERM);
    assert(::ftruncate(fd, 0) == -1 && errno == EPERM);
    assert(::fcntl(fd, F_ADD_SEALS, F_SEAL_FUTURE_WRITE) == -1 && errno == EPERM);
    ::close(fd);
    assert(::unlink(path.c_str()) == 0);
    assert(cache.load_kernel_from_oplib(path, "third").program->bytes == "loaded-ref-A");

    std::ofstream(path, std::ios::binary) << "failure-input";
    KernelCache failed;
    fail_program = true;
    try { failed.load_kernel_from_oplib(path, "fail"); assert(false); }
    catch (const std::runtime_error&) {}
    assert(failed.loaded_oplib_paths().empty() && failed.loaded_oplib_bytes().empty());
    assert(!failed.graph_snapshot_contracts_);
    fail_program = false;
    fail_kernel = true;
    try { failed.load_kernel_from_oplib(path, "fail"); assert(false); }
    catch (const std::runtime_error&) {}
    assert(failed.loaded_oplib_paths().empty());
    fail_kernel = false;
    failed.initialized_ = false;
    try { failed.loaded_oplib_bytes(); assert(false); }
    catch (const std::runtime_error&) {}

    mutate_source = path;
    try { OplibSnapshot changed(path); assert(false); }
    catch (const std::runtime_error&) {}
    std::ofstream(path, std::ios::binary).close();
    try { OplibSnapshot empty(path); assert(false); }
    catch (const std::runtime_error&) {}
    const std::string alias = path + ".link";
    assert(::symlink(path.c_str(), alias.c_str()) == 0);
    try { OplibSnapshot symlink(alias); assert(false); }
    catch (const std::runtime_error&) {}
    const int large = ::open(path.c_str(), O_WRONLY);
    assert(large >= 0 && ::ftruncate(large, 257 * 1024 * 1024) == 0);
    ::close(large);
    try { OplibSnapshot oversized(path); assert(false); }
    catch (const std::runtime_error&) {}
    ::unlink(path.c_str());
    std::ofstream(path, std::ios::binary) << "RAII";
    std::string closed_path;
    { OplibSnapshot temporary(path); closed_path = temporary.path(); }
    assert(::access(closed_path.c_str(), F_OK) == -1);
}
'''
    cpp, executable = tmp_path / "snapshot.cpp", tmp_path / "snapshot"
    cpp.write_text(harness)
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                    "-I", str(tmp_path), "-I", str(ROOT), str(cpp), "-o", str(executable)], check=True)
    subprocess.run([str(executable), str(tmp_path)], check=True)


def test_program_lifetime_and_symbol_preflight_uses_public_manifest():
    header = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    source = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
    clear_start = header.index("void clear()")
    clear = header[clear_start : header.index("private:", clear_start)]
    assert clear.index("&graph_snapshot_contracts_") < clear.index("kernels_.clear()")
    assert clear.index("kernels_.clear()") < clear.index("oplib_snapshots_.clear()")
    assert "oplib_sha256_.clear()" in clear
    symbols = source[source.index("KernelCache::kernel_manifest_names") :
                     source.index("static std::string canonical_payload_path")]
    assert "rpu_kernel_manifest::load(lib_path)" in symbols
    assert "oplib_symbols_.find(lib_path)" in symbols
    load = source[source.index("CachedKernel KernelCache::load_kernel_from_oplib"):source.index("void KernelCache::initialize()")]
    assert load.index("kernel_manifest_names(path)") < load.index("oplib_snapshot(path)")
    assert "std::ifstream" not in symbols


def test_w8a8_replay_programs_keep_reviewed_contracts_for_published_digests():
    import json
    import re
    deps = json.loads((ROOT / "release/runtime-dependencies.json").read_text())["artifacts"]
    digests = {deps["rhino_ops_main"]["sha256"], deps["rhino_ops_expansion"]["sha256"]}
    metadata = (ROOT / "src/core/rpu_graph_program_contracts.h").read_text()
    bindings = {
        (digest, name): int(contracts)
        for digest, name, contracts in re.findall(
            r'\{"([a-f0-9]{64})", "([A-Za-z0-9_]+)", (\d+)\}', metadata
        )
        if digest in digests
    }
    linear_catalog = (ROOT / "src/ops/rpu_linear_w8a8_candidates.inc").read_text()
    expected = set(re.findall(r'"(parallel_linear_w8a8_[A-Za-z0-9_]+)"', linear_catalog))
    expected.update({
        "dynamic_quantize_w8a8", "layer_norm_quantize_w8a8_h1152",
        "gelu_tanh_quantize_w8a8_h544",
        "pi_geglu_a8_paired_i32_c272_m48n32k2048_p512",
        "qwen3_up_mul_w8a16_m320n112k128",
        "qwen3_up_mul_w8a8_m272n64k384", "qwen3_up_mul_w8a8_m512n64k128",
    })
    expected.update(f"pi05_vision_owner_layernorm_quantize_w8a8_m{m}n1152" for m in (512, 768))
    expected.update(f"rms_norm_quantize_w8a8_h5120_m{m}" for m in (128, 192, 256))
    expected.update(f"mul_quantize_w8a8_h3200_m{m}" for m in (128, 192))
    admitted = {name for (_digest, name), contracts in bindings.items() if contracts == 3}
    assert expected <= admitted
