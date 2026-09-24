"""Exercise the production DDR logical-range registry with a host SDK double.

The test extracts the registry and resolver implementation verbatim.  It does
not import the backend, allocate an RPU buffer, or require a board.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
ALLOCATOR = ROOT / "src/core/rpu_caching_allocator.cpp"
KERNEL_CACHE = ROOT / "src/core/rpu_kernel_cache.h"
GRAPH_EXECUTE = ROOT / "src/graph/graph_runtime_execute.cpp"


@pytest.fixture(scope="module")
def logical_registry_host_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")

    production = ALLOCATOR.read_text(encoding="utf-8")
    begin = production.index("namespace {\n\nstruct ManagedDdrSegment")
    end = production.index("\nnamespace rpu {", begin)
    registry = production[begin:end]

    harness = r'''
#include "src/core/rpu_allocator_policy.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#define TORCH_CHECK(condition, ...) do { \
    if (!(condition)) throw std::runtime_error("contract rejection"); \
} while (0)

namespace rhino_lkn {
class Buffer_t {
public:
    Buffer_t(uintptr_t cpu, uint64_t device, size_t bytes)
        : cpu_(cpu), device_(device), bytes_(bytes) {}
    void* get_cpu_ptr() { return reinterpret_cast<void*>(cpu_); }
    uint64_t get_rpu_addr() const { return device_; }
    size_t get_memory_size() const { return bytes_; }
private:
    uintptr_t cpu_;
    uint64_t device_;
    size_t bytes_;
};
using HostDDR_t = Buffer_t;

std::vector<HostDDR_t*> owners;
int get_hostddr_calls = 0;
int fail_get_hostddr_call = 0;
HostDDR_t* RpuGetHostddr(void* ptr) {
    ++get_hostddr_calls;
    if (get_hostddr_calls == fail_get_hostddr_call) return nullptr;
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    for (auto* owner : owners) {
        const auto base = reinterpret_cast<uintptr_t>(owner->get_cpu_ptr());
        if (address >= base && address - base < owner->get_memory_size())
            return owner;
    }
    return nullptr;
}
uint64_t RpuGetDevAddr(void* ptr) {
    auto* owner = RpuGetHostddr(ptr);
    if (!owner) return 0;
    return owner->get_rpu_addr() +
        (reinterpret_cast<uintptr_t>(ptr) -
         reinterpret_cast<uintptr_t>(owner->get_cpu_ptr()));
}
void RpuDdrFree(void*) {}
}  // namespace rhino_lkn

struct RpuDmaEndpoint {
    rhino_lkn::Buffer_t* owner = nullptr;
    size_t offset = 0;
    uint64_t allocation_id = 0;

    explicit operator bool() const { return owner != nullptr; }
};

struct RpuDeviceDmaEndpointRequest {
    uint64_t device_addr = 0;
    size_t bytes = 0;
    const char* where = nullptr;
};

struct RpuCpuDmaEndpointRequest {
    const void* cpu_ptr = nullptr;
    size_t bytes = 0;
    const char* where = nullptr;
};

struct RpuDmaAllocationWitness {
    uint64_t allocation_id = 0;
    uint64_t expected_device_addr = 0;
    size_t bytes = 0;
    const char* where = nullptr;
};

class RpuDmaSubmissionLease final {
public:
    RpuDmaSubmissionLease() noexcept = default;
    ~RpuDmaSubmissionLease();
    RpuDmaSubmissionLease(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease& operator=(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease(RpuDmaSubmissionLease&&) noexcept;
    RpuDmaSubmissionLease& operator=(RpuDmaSubmissionLease&&) noexcept;
    explicit operator bool() const noexcept { return active_; }
private:
    explicit RpuDmaSubmissionLease(bool active) noexcept : active_(active) {}
    void release() noexcept;
    bool active_ = false;
    friend RpuDmaSubmissionLease rpu_resolve_device_dma_endpoints(
        const RpuDeviceDmaEndpointRequest*, size_t, RpuDmaEndpoint*);
    friend RpuDmaSubmissionLease rpu_resolve_cpu_dma_endpoints(
        const RpuCpuDmaEndpointRequest*, size_t, RpuDmaEndpoint*);
    friend RpuDmaSubmissionLease
    rpu_validate_live_dma_allocation_witnesses(
        const RpuDmaAllocationWitness*, size_t);
};

struct SpmAllocator { static constexpr int NUM_CORES = 8; };
struct FakeSpmAllocator {
    bool initialized = false;
    int root_calls = 0;
    std::array<rhino_lkn::Buffer_t*, SpmAllocator::NUM_CORES> roots{};
    bool is_initialized() const { return initialized; }
    rhino_lkn::Buffer_t* root_buffer(int core) {
        ++root_calls;
        return roots.at(static_cast<size_t>(core));
    }
} SPM_ALLOC;

REGISTRY

template <typename Fn>
void expect_rejection(Fn&& fn) {
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}

int main() {
    static_assert(!std::is_copy_constructible_v<RpuDmaSubmissionLease>);
    static_assert(!std::is_copy_assignable_v<RpuDmaSubmissionLease>);
    static_assert(std::is_nothrow_move_constructible_v<RpuDmaSubmissionLease>);
    static_assert(std::is_nothrow_move_assignable_v<RpuDmaSubmissionLease>);
    constexpr uintptr_t cpu = 0x10000000;
    constexpr uint64_t device = 0x80000000;
    constexpr size_t segment_bytes = 2 * 1024 * 1024;
    rhino_lkn::HostDDR_t packed(cpu, device, segment_bytes);
    rhino_lkn::owners.push_back(&packed);

    rpu_register_ddr_dma_segment(reinterpret_cast<void*>(cpu), segment_bytes);
    const uint64_t first = rpu_register_ddr_dma_logical_allocation(
        reinterpret_cast<void*>(cpu), 1024);
    const uint64_t second = rpu_register_ddr_dma_logical_allocation(
        reinterpret_cast<void*>(cpu + 4096), 2048);
    assert(first != 0 && second > first);
    assert(g_logical_ddr_by_cpu.size() == 2);
    assert(g_logical_ddr_by_device.size() == 2);
    assert(g_logical_ddr_by_allocation_id.size() == 2);

    // A failure at the third index rolls back the first two insertions. Force
    // an allocation-id collision to exercise the otherwise exceptional path.
    g_next_logical_ddr_allocation_id = first;
    expect_rejection([&] {
        (void)rpu_register_ddr_dma_logical_allocation(
            reinterpret_cast<void*>(cpu + 8192), 512);
    });
    assert(g_logical_ddr_by_cpu.size() == 2);
    assert(g_logical_ddr_by_device.size() == 2);
    assert(g_logical_ddr_by_allocation_id.size() == 2);
    assert(g_logical_ddr_by_cpu.count(cpu + 8192) == 0);
    assert(g_logical_ddr_by_device.count(device + 8192) == 0);
    g_next_logical_ddr_allocation_id = second + 1;

    // Exhaustion must reject before touching any of the three logical maps.
    g_next_logical_ddr_allocation_id = std::numeric_limits<uint64_t>::max();
    expect_rejection([&] {
        (void)rpu_register_ddr_dma_logical_allocation(
            reinterpret_cast<void*>(cpu + 8192), 512);
    });
    assert(g_logical_ddr_by_cpu.size() == 2);
    assert(g_logical_ddr_by_device.size() == 2);
    assert(g_logical_ddr_by_allocation_id.size() == 2);
    g_next_logical_ddr_allocation_id = second + 1;

    // Interior views are valid only through the end of their owning Storage.
    auto endpoint = rpu_resolve_cpu_dma_endpoint(
        reinterpret_cast<void*>(cpu + 100), 924, "interior CPU view");
    assert(endpoint.owner == &packed && endpoint.offset == 100 &&
           endpoint.allocation_id == first);
    endpoint = rpu_resolve_device_dma_endpoint(
        device + 4096 + 512, 1536, "interior device view");
    assert(endpoint.owner == &packed && endpoint.offset == 4096 + 512 &&
           endpoint.allocation_id == second);

    // Mixed DDR/SPM batches resolve every endpoint before publishing any
    // caller-visible result. SPM retains allocation identity zero.
    constexpr uintptr_t spm_cpu = 0x30000000;
    constexpr uint64_t spm_device = 0x70000000;
    rhino_lkn::Buffer_t spm(spm_cpu, spm_device, 4096);
    SPM_ALLOC.initialized = true;
    SPM_ALLOC.roots.fill(&spm);
    const RpuDeviceDmaEndpointRequest mixed_requests[] = {
        {device + 64, 128, "batch DDR first"},
        {spm_device + 32, 64, "batch SPM"},
        {device + 4096 + 100, 256, "batch DDR second"},
    };
    RpuDmaEndpoint mixed[3];
    rpu_resolve_device_dma_endpoints(mixed_requests, 3, mixed);
    assert(mixed[0].owner == &packed && mixed[0].offset == 64 &&
           mixed[0].allocation_id == first);
    assert(mixed[1].owner == &spm && mixed[1].offset == 32 &&
           mixed[1].allocation_id == 0);
    assert(mixed[2].owner == &packed && mixed[2].offset == 4096 + 100 &&
           mixed[2].allocation_id == second);
    const auto scalar_spm = rpu_resolve_device_dma_endpoint(
        spm_device + 128, 32, "scalar delegates to batch");
    assert(scalar_spm.owner == &spm && scalar_spm.offset == 128 &&
           scalar_spm.allocation_id == 0);

    // A late resolution failure is transactional: earlier results are not
    // published. The malformed-input pass also precedes all SPM inspection.
    const RpuDeviceDmaEndpointRequest late_failure[] = {
        {device + 32, 32, "valid batch prefix"},
        {device + 2048, 16, "invalid batch suffix"},
    };
    RpuDmaEndpoint unchanged[] = {{&spm, 77, 91}, {&spm, 88, 92}};
    expect_rejection([&] {
        rpu_resolve_device_dma_endpoints(late_failure, 2, unchanged);
    });
    assert(unchanged[0].owner == &spm && unchanged[0].offset == 77 &&
           unchanged[0].allocation_id == 91);
    assert(unchanged[1].owner == &spm && unchanged[1].offset == 88 &&
           unchanged[1].allocation_id == 92);
    const int root_calls_before_malformed = SPM_ALLOC.root_calls;
    const RpuDeviceDmaEndpointRequest malformed[] = {
        {device, 16, "valid malformed prefix"},
        {std::numeric_limits<uint64_t>::max(), 2, "overflow suffix"},
    };
    expect_rejection([&] {
        rpu_resolve_device_dma_endpoints(malformed, 2, unchanged);
    });
    assert(SPM_ALLOC.root_calls == root_calls_before_malformed);

    // Device batches, CPU batches, and witness validation each acquire one
    // global DDR submission lease. SPM-only batches acquire none. Logical
    // retirement waits for every lease while releasing the registry mutex.
    const uint64_t leased = rpu_register_ddr_dma_logical_allocation(
        reinterpret_cast<void*>(cpu + 8192), 512);
    const RpuDeviceDmaEndpointRequest leased_device_request[] = {
        {device + 8192, 512, "leased device endpoint"}};
    RpuDmaEndpoint leased_device_endpoint[1];
    auto device_lease = rpu_resolve_device_dma_endpoints(
        leased_device_request, 1, leased_device_endpoint);
    assert(device_lease && g_active_ddr_submission_leases == 1);
    const RpuCpuDmaEndpointRequest leased_cpu_request[] = {
        {reinterpret_cast<void*>(cpu + 8192), 512,
         "leased CPU endpoint"}};
    RpuDmaEndpoint leased_cpu_endpoint[1];
    auto cpu_lease = rpu_resolve_cpu_dma_endpoints(
        leased_cpu_request, 1, leased_cpu_endpoint);
    assert(cpu_lease && g_active_ddr_submission_leases == 2);
    const RpuDmaAllocationWitness leased_witness[] = {
        {leased, device + 8192, 512, "leased witness"}};
    auto witness_lease =
        rpu_validate_live_dma_allocation_witnesses(leased_witness, 1);
    assert(witness_lease && g_active_ddr_submission_leases == 3);
    const RpuDeviceDmaEndpointRequest spm_only_request[] = {
        {spm_device, 16, "SPM-only lease bypass"}};
    RpuDmaEndpoint spm_only_endpoint[1];
    auto spm_only_lease = rpu_resolve_device_dma_endpoints(
        spm_only_request, 1, spm_only_endpoint);
    assert(!spm_only_lease && g_active_ddr_submission_leases == 3);

    auto moved_device_lease = std::move(device_lease);
    assert(!device_lease && moved_device_lease);
    bool unregister_started = false;
    bool unregister_done = false;
    bool unregister_result = false;
    std::mutex test_mutex;
    std::condition_variable test_cv;
    std::thread unregister_thread([&] {
        {
            std::lock_guard<std::mutex> guard(test_mutex);
            unregister_started = true;
        }
        test_cv.notify_one();
        const bool result = rpu_unregister_ddr_dma_logical_allocation(
            reinterpret_cast<void*>(cpu + 8192), leased);
        {
            std::lock_guard<std::mutex> guard(test_mutex);
            unregister_result = result;
            unregister_done = true;
        }
        test_cv.notify_one();
    });
    {
        std::unique_lock<std::mutex> guard(test_mutex);
        test_cv.wait(guard, [&] { return unregister_started; });
        assert(!test_cv.wait_for(
            guard, std::chrono::milliseconds(20),
            [&] { return unregister_done; }));
    }
    cpu_lease = {};
    witness_lease = {};
    assert(g_active_ddr_submission_leases == 1);
    {
        std::unique_lock<std::mutex> guard(test_mutex);
        assert(!test_cv.wait_for(
            guard, std::chrono::milliseconds(20),
            [&] { return unregister_done; }));
    }
    moved_device_lease = {};
    {
        std::unique_lock<std::mutex> guard(test_mutex);
        assert(test_cv.wait_for(
            guard, std::chrono::seconds(1),
            [&] { return unregister_done; }));
    }
    unregister_thread.join();
    assert(unregister_result && g_active_ddr_submission_leases == 0);
    expect_rejection([&] {
        (void)rpu_resolve_device_dma_endpoint(
            device + 8192, 16, "retired leased allocation");
    });

    const RpuDmaAllocationWitness live_witnesses[] = {
        {first, device + 128, 896, "first live allocation"},
        {0, 0, 0, nullptr},
        {second, device + 4096 + 1024, 1024, "second live allocation"},
    };
    rpu_validate_live_dma_allocation_witnesses(live_witnesses, 3);
    const RpuDmaAllocationWitness wrong_owner[] = {
        {first, device, 16, "valid witness prefix"},
        {first, device + 4096, 16, "wrong logical owner suffix"},
    };
    expect_rejection([&] {
        rpu_validate_live_dma_allocation_witnesses(wrong_owner, 2);
    });

    expect_rejection([&] {
        (void)rpu_resolve_cpu_dma_endpoint(
            reinterpret_cast<void*>(cpu + 1023), 2, "logical overrun");
    });
    expect_rejection([&] {
        (void)rpu_resolve_device_dma_endpoint(
            device + 2048, 16, "unallocated segment gap");
    });
    expect_rejection([&] {
        (void)rpu_register_ddr_dma_logical_allocation(
            reinterpret_cast<void*>(cpu + 512), 1024);
    });
    expect_rejection([&] {
        rpu_unregister_ddr_dma_segment(reinterpret_cast<void*>(cpu));
    });

    // A wrong generation cannot retire the current logical owner.
    assert(!rpu_unregister_ddr_dma_logical_allocation(
        reinterpret_cast<void*>(cpu), first + 99));
    assert(g_logical_ddr_by_cpu.size() == 2);
    assert(g_logical_ddr_by_device.size() == 2);
    assert(g_logical_ddr_by_allocation_id.size() == 2);
    assert(rpu_resolve_device_dma_endpoint(device, 16, "still live")
               .allocation_id == first);
    assert(rpu_unregister_ddr_dma_logical_allocation(
        reinterpret_cast<void*>(cpu), first));
    expect_rejection([&] {
        (void)rpu_resolve_device_dma_endpoint(device, 16, "freed logical");
    });

    // Reuse has the same addresses but a distinct monotonic identity.
    const uint64_t reused = rpu_register_ddr_dma_logical_allocation(
        reinterpret_cast<void*>(cpu), 1024);
    assert(reused > second && reused != first);
    assert(rpu_resolve_device_dma_endpoint(device, 16, "reused logical")
               .allocation_id == reused);
    const RpuDmaAllocationWitness stale_after_aba[] = {
        {first, device, 16, "released generation"},
        {reused, device, 16, "new generation"},
    };
    expect_rejection([&] {
        rpu_validate_live_dma_allocation_witnesses(stale_after_aba, 2);
    });
    const RpuDmaAllocationWitness reused_live[] = {
        {reused, device + 1000, 24, "reused live generation"},
    };
    rpu_validate_live_dma_allocation_witnesses(reused_live, 1);
    assert(rpu_unregister_ddr_dma_logical_allocation(
        reinterpret_cast<void*>(cpu), reused));
    assert(rpu_unregister_ddr_dma_logical_allocation(
        reinterpret_cast<void*>(cpu + 4096), second));
    assert(g_logical_ddr_by_cpu.empty());
    assert(g_logical_ddr_by_device.empty());
    assert(g_logical_ddr_by_allocation_id.empty());
    rpu_unregister_ddr_dma_segment(reinterpret_cast<void*>(cpu));
    SPM_ALLOC.initialized = false;

    // The unchanged direct-allocator API performs the segment and logical
    // registration as one transaction.
    constexpr uintptr_t direct_cpu = 0x20000000;
    constexpr uint64_t direct_device = 0x90000000;
    rhino_lkn::HostDDR_t direct(direct_cpu, direct_device, 4096);
    rhino_lkn::owners.push_back(&direct);
    rpu_register_ddr_dma_allocation(
        reinterpret_cast<void*>(direct_cpu), 1536);
    const auto direct_endpoint = rpu_resolve_device_dma_endpoint(
        direct_device + 128, 1408, "direct allocation");
    assert(direct_endpoint.owner == &direct && direct_endpoint.offset == 128 &&
           direct_endpoint.allocation_id > reused);
    expect_rejection([&] {
        (void)rpu_resolve_device_dma_endpoint(
            direct_device + 1535, 2, "direct logical overrun");
    });
    rpu_unregister_ddr_dma_allocation(reinterpret_cast<void*>(direct_cpu));
    expect_rejection([&] {
        (void)rpu_resolve_device_dma_endpoint(
            direct_device, 16, "unregistered direct allocation");
    });

    // A malformed physical range is rejected before either segment index is
    // modified, avoiding wraparound at the end of the device address space.
    constexpr uintptr_t overflow_cpu = 0x40000000;
    constexpr uint64_t overflow_device =
        std::numeric_limits<uint64_t>::max() - 7;
    rhino_lkn::HostDDR_t overflow_owner(overflow_cpu, overflow_device, 16);
    rhino_lkn::owners.push_back(&overflow_owner);
    expect_rejection([&] {
        rpu_register_ddr_dma_segment(
            reinterpret_cast<void*>(overflow_cpu), 16);
    });
    assert(g_managed_ddr_segments_by_cpu.empty());
    assert(g_managed_ddr_segments_by_device.empty());

    // If the logical half of the legacy direct transaction fails, its newly
    // inserted physical segment must be rolled back as well.
    constexpr uintptr_t rollback_cpu = 0x50000000;
    constexpr uint64_t rollback_device = 0xa0000000;
    rhino_lkn::HostDDR_t rollback_owner(rollback_cpu, rollback_device, 4096);
    rhino_lkn::owners.push_back(&rollback_owner);
    rhino_lkn::fail_get_hostddr_call = rhino_lkn::get_hostddr_calls + 2;
    expect_rejection([&] {
        rpu_register_ddr_dma_allocation(
            reinterpret_cast<void*>(rollback_cpu), 1024);
    });
    rhino_lkn::fail_get_hostddr_call = 0;
    assert(g_managed_ddr_segments_by_cpu.empty());
    assert(g_managed_ddr_segments_by_device.empty());
    assert(g_logical_ddr_by_cpu.empty());
    assert(g_logical_ddr_by_device.empty());
    assert(g_logical_ddr_by_allocation_id.empty());

    rpu_resolve_device_dma_endpoints(nullptr, 0, nullptr);
    rpu_resolve_cpu_dma_endpoints(nullptr, 0, nullptr);
    rpu_validate_live_dma_allocation_witnesses(nullptr, 0);
}
'''.replace("REGISTRY", registry)

    directory = tmp_path_factory.mktemp("allocator-logical-registry-host")
    cpp = directory / "logical_registry.cpp"
    binary = directory / "logical_registry"
    cpp.write_text(harness, encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-pthread",
         "-I", str(ROOT), str(cpp), "-o", str(binary)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return binary


def test_live_logical_ranges_and_reuse_identity(logical_registry_host_binary):
    subprocess.run([str(logical_registry_host_binary)], check=True, timeout=5)


def test_cache_hooks_register_and_retire_logical_storage():
    source = ALLOCATOR.read_text(encoding="utf-8")
    assert "rpu_register_ddr_dma_segment(ptr, alloc_size);" in source
    assert "rpu_register_ddr_dma_logical_allocation(block->ptr, orig_size)" in source
    assert "rpu_unregister_ddr_dma_logical_allocation(\n            ptr, block->allocation_id)" in source
    assert "rpu_unregister_ddr_dma_segment(block->ptr);" in source


def test_batch_apis_take_one_registry_lock_and_scalar_delegates():
    source = ALLOCATOR.read_text(encoding="utf-8")

    def function(signature: str, next_signature: str) -> str:
        begin = source.index(signature)
        end = source.index(next_signature, begin)
        return source[begin:end]

    cpu_batch = function(
        "RpuDmaSubmissionLease rpu_resolve_cpu_dma_endpoints(",
        "RpuDmaSubmissionLease rpu_resolve_device_dma_endpoints(",
    )
    batch = function(
        "RpuDmaSubmissionLease rpu_resolve_device_dma_endpoints(",
        "RpuDmaEndpoint rpu_resolve_device_dma_endpoint(",
    )
    scalar = function(
        "RpuDmaEndpoint rpu_resolve_device_dma_endpoint(",
        "RpuDmaSubmissionLease rpu_validate_live_dma_allocation_witnesses(",
    )
    witnesses = function(
        "RpuDmaSubmissionLease rpu_validate_live_dma_allocation_witnesses(",
        "\nnamespace rpu {",
    )
    lock = "std::lock_guard<std::mutex> lock(g_managed_ddr_mutex);"
    assert cpu_batch.count(lock) == 1
    assert batch.count(lock) == 1
    assert witnesses.count(lock) == 1
    assert "rpu_resolve_device_dma_endpoints(\n        &request, /*count=*/1, &endpoint);" in scalar
    unregister = function(
        "bool rpu_unregister_ddr_dma_logical_allocation(",
        "void rpu_register_ddr_dma_allocation(",
    )
    assert "g_managed_ddr_lease_cv.wait(lock" in unregister
    assert "std::unique_lock<std::mutex> lock(g_managed_ddr_mutex);" in unregister
    assert "unordered_map" not in batch


def test_non_graph_dma_chokepoints_retain_atomic_endpoint_leases():
    cache = KERNEL_CACHE.read_text(encoding="utf-8")
    checked = cache[
        cache.index("inline void rpu_add_dma_checked(::rhino_lkn::Queue_t& q,\n                                uint64_t") :
        cache.index("inline void rpu_add_dma_mutable_checked(",
                    cache.index("inline void rpu_add_dma_checked(::rhino_lkn::Queue_t& q,\n                                uint64_t"))
    ]
    mutable = cache[
        cache.index("inline void rpu_add_dma_mutable_checked(::rhino_lkn::Queue_t& q,\n                                        uint64_t") :
        cache.index("// Same dropped-return-code treatment",
                    cache.index("inline void rpu_add_dma_mutable_checked(::rhino_lkn::Queue_t& q,\n                                        uint64_t"))
    ]
    for helper in (checked, mutable):
        assert "RpuDeviceDmaEndpointRequest requests[]" in helper
        assert "rpu_resolve_device_dma_endpoints(requests, 2, endpoints)" in helper
        assert "rpu_resolve_device_dma_endpoint(" not in helper
        assert helper.index("submission_lease") < helper.rindex("rpu_add_dma_")

    execute = GRAPH_EXECUTE.read_text(encoding="utf-8")
    dispatch = execute[
        execute.index("case CopyKind::DDR_TO_DDR:") :
        execute.index("case CopyKind::DDR_TO_SPM:",
                      execute.index("case CopyKind::DDR_TO_DDR:"))
    ]
    assert "execute_ddr_copy(m);" in dispatch
    ddr_copy = execute[
        execute.index("void RpuKernelGraph::execute_ddr_copy(") :
        execute.index("void RpuKernelGraph::execute_data_node(")
    ]
    assert "RpuCpuDmaEndpointRequest requests[]" in ddr_copy
    assert "rpu_resolve_cpu_dma_endpoints(requests, 2, endpoints)" in ddr_copy
    assert "rpu_resolve_cpu_dma_endpoint(" not in ddr_copy
    assert ddr_copy.index("submission_lease") < ddr_copy.index("rpu_add_dma_checked(")
    assert ddr_copy.index("rpu_add_dma_checked(") < ddr_copy.index("queue.enqueu_batch(")
    assert ddr_copy.index("queue.enqueu_batch(") < ddr_copy.index("queue.discard_completed_batch(")
    assert "data_dma_queue_.reset();" in ddr_copy
