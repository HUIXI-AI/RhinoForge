"""Board-free lifecycle tests for RMSNorm payload capability snapshots."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
KERNEL_CACHE_HEADER = (ROOT / "src/core/rpu_kernel_cache.h").read_text(
    encoding="utf-8"
)
KERNEL_CACHE_SOURCE = (ROOT / "src/core/rpu_kernel_cache.inc").read_text(
    encoding="utf-8"
)


def _block(source: str, marker: str) -> str:
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def _compile_and_run(tmp_path: Path, name: str, program: str) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source = tmp_path / f"{name}.cpp"
    executable = tmp_path / name
    source.write_text(program, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, timeout=30)


def test_kernel_payload_availability_is_atomic_and_withdrawn_before_clear(
    tmp_path: Path,
) -> None:
    availability = _block(
        KERNEL_CACHE_HEADER, "class KernelPayloadAvailability final"
    )
    cache_clear = _block(KERNEL_CACHE_HEADER, "void clear()")
    initialize = _block(KERNEL_CACHE_SOURCE, "void KernelCache::initialize()")

    assert "std::shared_ptr<const Snapshot> state_" in availability
    assert "std::atomic_load_explicit(&state_, std::memory_order_acquire)" in availability
    assert "std::atomic_store_explicit(&state_, next, std::memory_order_release)" in availability
    assert "std::shared_ptr<const Snapshot>{}" in _block(availability, "void withdraw()")
    assert cache_clear.index("payload_availability_.withdraw()") < cache_clear.index(
        "kernels_.clear()"
    )
    assert initialize.index("populate_fast_cache();") < initialize.index(
        "payload_availability_.publish("
    )

    program = (
        r'''
#include <atomic>
#include <bitset>
#include <memory>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
'''
        + _block(KERNEL_CACHE_HEADER, "enum class KernelId : uint16_t")
        + ";\n"
        + availability
        + ";\n"
        + r'''
int main() {
    KernelPayloadAvailability availability;
    bool rejected = false;
    try { (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V16); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    availability.publish(true, false);
    assert(availability.has_loaded(KernelId::RMS_NORM_SPM_V16));
    assert(!availability.has_loaded(KernelId::RMS_NORM_SPM_V32));
    rejected = false;
    try { (void)availability.has_loaded(KernelId::BINARY_SAMESHAPE); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    std::atomic<bool> start{false};
    std::atomic<bool> done{false};
    std::atomic<int> ready{0};
    std::atomic<int> reads{0};
    std::vector<std::thread> readers;
    for (int thread = 0; thread < 4; ++thread) {
        readers.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {}
            (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V16);
            (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V32);
            reads.fetch_add(1, std::memory_order_relaxed);
            ready.fetch_add(1, std::memory_order_release);
            while (!done.load(std::memory_order_acquire)) {
                try {
                    (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V16);
                    (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V32);
                    reads.fetch_add(1, std::memory_order_relaxed);
                } catch (const std::runtime_error&) {
                    // Withdrawal is the fail-closed state during teardown.
                }
            }
        });
    }
    std::thread publisher([&] {
        while (!start.load(std::memory_order_acquire)) {}
        while (ready.load(std::memory_order_acquire) != 4) {}
        for (int i = 0; i < 100000; ++i) {
            availability.withdraw();
            availability.publish((i & 1) != 0, (i & 2) != 0);
        }
        done.store(true, std::memory_order_release);
    });
    start.store(true, std::memory_order_release);
    publisher.join();
    for (auto& reader : readers) reader.join();
    assert(reads.load(std::memory_order_relaxed) > 0);

    availability.withdraw();
    rejected = false;
    try { (void)availability.has_loaded(KernelId::RMS_NORM_SPM_V32); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
'''
    )
    _compile_and_run(tmp_path, "payload_publication", program)
