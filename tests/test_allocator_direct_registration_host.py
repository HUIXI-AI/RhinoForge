"""Exercise the production direct HostDDR allocation transaction on the host."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/core/rpu_tensor_ops.inc"
ALLOCATOR_SOURCE = ROOT / "src/core/rpu_caching_allocator.cpp"
BACKEND_SOURCE = ROOT / "src/core/rpu_backend.cpp"


def _extract_balanced_function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unterminated production function: {signature}")


@pytest.fixture(scope="module")
def direct_allocation_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")

    production = SOURCE.read_text(encoding="utf-8")
    transaction = _extract_balanced_function(
        production, "static void* rpu_allocate_registered_ddr(",
    )
    direct_free = _extract_balanced_function(
        ALLOCATOR_SOURCE.read_text(encoding="utf-8"),
        "void rpu_free_registered_ddr(",
    )
    harness = r'''
#include "src/core/rpu_allocator_policy.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <stdexcept>

int allocations = 0;
int registrations = 0;
int unregistrations = 0;
int frees = 0;
bool return_null = false;
bool reject_registration = false;
void* storage = reinterpret_cast<void*>(uintptr_t{0x12340000});

namespace rhino_lkn {
void* RpuDdrAlloc(size_t) {
  ++allocations;
  return return_null ? nullptr : storage;
}
void RpuDdrFree(void* ptr) {
  assert(ptr == storage);
  ++frees;
}
}  // namespace rhino_lkn

void rpu_register_ddr_dma_allocation(void* ptr, size_t) {
  assert(ptr == storage);
  ++registrations;
  if (reject_registration) throw std::runtime_error("registry rejected");
}

void rpu_unregister_ddr_dma_allocation(void* ptr) {
  assert(ptr == storage);
  ++unregistrations;
}

TRANSACTION
DIRECT_FREE

int main() {
  assert(rpu_allocate_registered_ddr(4096) == storage);
  assert(allocations == 1 && registrations == 1 && frees == 0);

  reject_registration = true;
  try {
    (void)rpu_allocate_registered_ddr(4096);
    assert(false);
  } catch (const std::runtime_error&) {}
  assert(allocations == 2 && registrations == 2 && frees == 1);

  return_null = true;
  try {
    (void)rpu_allocate_registered_ddr(4096);
    assert(false);
  } catch (const std::bad_alloc&) {}
  assert(allocations == 3 && registrations == 2 && frees == 1);

  rpu_free_registered_ddr(storage);
  assert(unregistrations == 1 && frees == 2);

  rpu_tensor_ddr_lifecycle().begin_shutdown();
  rpu_free_registered_ddr(storage);
  assert(unregistrations == 2 && frees == 2);
  rpu_free_registered_ddr(nullptr);
  assert(unregistrations == 2 && frees == 2);
}
'''.replace("TRANSACTION", transaction).replace("DIRECT_FREE", direct_free)

    directory = tmp_path_factory.mktemp("allocator-direct-registration-host")
    cpp = directory / "direct_registration.cpp"
    binary = directory / "direct_registration"
    cpp.write_text(harness, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-pthread",
         "-I", str(ROOT), str(cpp), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return binary


def test_registration_failure_releases_physical_allocation(
    direct_allocation_binary,
):
    subprocess.run([str(direct_allocation_binary)], check=True, timeout=5)


def test_tensor_allocation_and_shutdown_use_the_process_lifecycle_gate() -> None:
    tensor_ops = SOURCE.read_text(encoding="utf-8")
    caching_allocator = ALLOCATOR_SOURCE.read_text(encoding="utf-8")
    backend = BACKEND_SOURCE.read_text(encoding="utf-8")

    allocation = _extract_balanced_function(
        tensor_ops, "at::DataPtr allocate(size_t nbytes) override",
    )
    gate = allocation.index("rpu_tensor_ddr_lifecycle().run_allocation(")
    assert gate < allocation.index(
        "RPUCachingAllocatorAdapter::get().allocate(nbytes)", gate,
    )
    assert gate < allocation.index("rpu_allocate_registered_ddr(nbytes)", gate)

    direct_free = _extract_balanced_function(
        caching_allocator, "void rpu_free_registered_ddr(",
    )
    unregister = direct_free.index("rpu_unregister_ddr_dma_allocation(cpu_base)")
    guarded_free = direct_free.index(
        "rpu_tensor_ddr_lifecycle().run_free_if_live", unregister,
    )
    assert guarded_free < direct_free.index("RpuDdrFree(cpu_base)", guarded_free)

    shutdown = _extract_balanced_function(backend, "void rpu_shutdown()")
    state = shutdown.index(
        "g_rpu_backend_lifecycle = RpuBackendLifecycleState::kShutdown"
    )
    begin_shutdown = shutdown.index(
        "rpu_tensor_ddr_lifecycle().begin_shutdown()", state,
    )
    graph_invalidation = shutdown.index(
        "invalidate_registered_rpu_kernel_graphs()", begin_shutdown,
    )
    caching_mark = shutdown.index(
        "RPUCachingAllocator::get().mark_shutdown()", graph_invalidation,
    )
    ddr_shutdown = shutdown.index("RpuDdrShutdown()", caching_mark)
    assert state < begin_shutdown < graph_invalidation < caching_mark < ddr_shutdown

    empty_cache = _extract_balanced_function(
        caching_allocator, "void RPUCachingAllocator::emptyCache()",
    )
    assert "RpuDdrFree" in empty_cache
    assert "rpu_tensor_ddr_lifecycle" not in empty_cache
