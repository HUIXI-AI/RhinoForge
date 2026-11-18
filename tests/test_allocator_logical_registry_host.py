"""Exercise the production DDR logical-range registry with a host SDK double.

The test extracts the registry and resolver implementation verbatim.  It does
not import the backend, allocate an device buffer, or require a board.
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
ENDPOINT_HEADER = ROOT / "src/core/rpu_dma_endpoint.h"


@pytest.fixture(scope="module")
def logical_registry_host_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")

    production = ALLOCATOR.read_text(encoding="utf-8")
    begin = production.index("namespace {\n\nstruct ManagedDdrSegment")
    end = production.index("\nnamespace rpu {", begin)
    registry = production[begin:end]
    endpoint_header = ENDPOINT_HEADER.read_text(encoding="utf-8").replace(
        '#include "rhino_launch_buffer.h"', "").replace("#pragma once", "")
    graph_source = GRAPH_EXECUTE.read_text(encoding="utf-8")
    coalesce_start = graph_source.index("void coalesce_fixed_dma_allocation_witnesses(")
    coalesce_end = graph_source.index("\nuint8_t validated_kernel_core_extent(", coalesce_start)
    coalesce = graph_source[coalesce_start:coalesce_end]

    harness = r'''
#include "src/core/rpu_allocator_policy.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

// This executable owns the allocation fault injector. It is disabled except
// around one registration attempt, and never runs in the lease-wait thread.
thread_local std::ptrdiff_t allocations_before_failure = -1;
void* operator new(std::size_t bytes) {
    if (allocations_before_failure == 0) {
        allocations_before_failure = -1;
        throw std::bad_alloc();
    }
    if (allocations_before_failure > 0) --allocations_before_failure;
    if (void* result = std::malloc(bytes ? bytes : 1)) return result;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete[](void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { std::free(ptr); }

#define TORCH_CHECK(condition, ...) do { \
    if (!(condition)) throw std::runtime_error("contract rejection"); \
} while (0)

namespace rhino_lkn {
class Buffer_t {
public:
    Buffer_t(uintptr_t cpu, uint64_t device, size_t bytes)
        : cpu_(cpu), device_(device), bytes_(bytes) {}
    void* get_cpu_ptr() { return reinterpret_cast<void*>(cpu_); }
    uint64_t get_rpu_addr() const { ++address_reads; return device_; }
    size_t get_memory_size() const { ++size_reads; return bytes_; }
    mutable int address_reads = 0;
    mutable int size_reads = 0;
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

ENDPOINT_TYPES

// Async replay end (graph/graph_async_end.h): no Graph submission is pending in
// this harness, so frees are never deferred and the deferred list never drains.
bool rpu_graph_async_end_pending() noexcept { return false; }
namespace rpu {
struct RPUCachingAllocator {
    static RPUCachingAllocator& get() { std::abort(); }
    void free(void*) { std::abort(); }
};
}  // namespace rpu

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
COALESCE_WITNESSES

template <typename Fn>
void expect_rejection(Fn&& fn) {
    bool rejected = false;
    try { fn(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}

template <typename Index>
void check_address_index_growth() {
    Index index;
    const LogicalDdrAllocation original{
        reinterpret_cast<void*>(0x1000), 0x8000, 32, nullptr, 0, 1};
    auto first = index.emplace(1024, original).first;
    const auto* first_record = &first->second;
    // Grow well beyond the default hash bucket count, retaining an ordered
    // iterator/reference while insertion and hash rehash take place.
    for (uint64_t i = 1; i < 1024; ++i) {
        auto value = original;
        value.allocation_id = i + 1;
        assert(index.emplace(1024 + 64 * i, value).second);
    }
    const Index& read_only = index;
    assert(read_only.size() == 1024 && !read_only.empty());
    assert(&read_only.find(1024)->second == first_record);
    assert(first == index.begin());
    for (uint64_t i = 0; i < 1024; ++i) {
        const auto key = 1024 + 64 * i;
        auto exact = read_only.find(key);
        assert(exact != read_only.end() && exact->second.allocation_id == i + 1);
        assert(read_only.count(key) == 1);
        assert(read_only.lower_bound(key) == exact);
        assert(std::prev(read_only.upper_bound(key + 31)) == exact);
        assert(read_only.find(key + 1) == read_only.end());
    }
    auto replacement = original;
    replacement.allocation_id = 99999;
    const auto duplicate = index.emplace(1024, replacement);
    assert(!duplicate.second && &duplicate.first->second == first_record);
    assert(same_logical_ddr_allocation(duplicate.first->second, original));
    assert(index.size() == 1024);
    // Erasure must use the immutable map key, not a possibly damaged payload.
    first->second.cpu_base = reinterpret_cast<void*>(0xdead0000);
    first->second.device_base = 0xbeef0000;
    auto next = index.erase(first);
    assert(next == index.begin() && next->first == 1088);
    assert(index.find(1024) == index.end() && index.count(1024) == 0);
    assert(index.erase(1088) == 1 && index.erase(1088) == 0);
    assert(index.lower_bound(1024)->first == 1152);
    index.clear();
    assert(index.empty() && index.begin() == index.end());
    for (uint64_t i = 0; i < 1024; ++i)
        assert(index.find(1024 + 64 * i) == index.end());
    assert(index.emplace(1024, replacement).second);
    assert(index.find(1024)->second.allocation_id == 99999);
}

int main() {
    static_assert(!std::is_copy_constructible_v<RpuDmaSubmissionLease>);
    static_assert(!std::is_copy_assignable_v<RpuDmaSubmissionLease>);
    static_assert(std::is_nothrow_move_constructible_v<RpuDmaSubmissionLease>);
    static_assert(std::is_nothrow_move_assignable_v<RpuDmaSubmissionLease>);
    static_assert(!std::is_copy_constructible_v<RpuResolvedDeviceDmaBatch>);
    static_assert(!std::is_copy_assignable_v<RpuResolvedDeviceDmaBatch>);
    static_assert(std::is_nothrow_move_constructible_v<RpuResolvedDeviceDmaBatch>);
    static_assert(std::is_nothrow_move_assignable_v<RpuResolvedDeviceDmaBatch>);
    static_assert(std::is_same_v<decltype(std::declval<RpuResolvedDeviceDmaBatch&>().endpoints()),
                                 const std::vector<RpuDmaEndpoint>&>);
    allocations_before_failure = 0;
    auto empty_owned = rpu_resolve_device_dma_endpoints_owned(nullptr, 0);
    assert(empty_owned.endpoints().empty() && !empty_owned.lease());
    assert(allocations_before_failure == 0);
    allocations_before_failure = -1;
    check_address_index_growth<decltype(g_logical_ddr_by_cpu)>();
    check_address_index_growth<decltype(g_logical_ddr_by_device)>();
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

    // Fail each allocation in real registration, including an exact-index
    // insertion after its ordered node exists and later outer-index inserts.
    // Each failed attempt must preserve the two existing logical allocations.
    bool registration_completed = false;
    int allocation_failures = 0;
    for (int budget = 0; budget < 16; ++budget) {
        uint64_t attempted_id = 0;
        allocations_before_failure = budget;
        try {
            attempted_id = rpu_register_ddr_dma_logical_allocation(
                reinterpret_cast<void*>(cpu + 16384), 512);
            allocations_before_failure = -1;
        } catch (const std::bad_alloc&) {
            allocations_before_failure = -1;
            ++allocation_failures;
        }
        if (attempted_id != 0) {
            assert(rpu_unregister_ddr_dma_logical_allocation(
                reinterpret_cast<void*>(cpu + 16384), attempted_id));
            registration_completed = true;
        }
        assert(g_logical_ddr_by_cpu.size() == 2);
        assert(g_logical_ddr_by_device.size() == 2);
        assert(g_logical_ddr_by_allocation_id.size() == 2);
        assert(g_logical_ddr_by_cpu.find(cpu + 16384) == g_logical_ddr_by_cpu.end());
        assert(g_logical_ddr_by_device.find(device + 16384) == g_logical_ddr_by_device.end());
        assert(g_logical_ddr_by_cpu.lower_bound(cpu + 16384) == g_logical_ddr_by_cpu.end());
        assert(g_logical_ddr_by_device.lower_bound(device + 16384) == g_logical_ddr_by_device.end());
        assert(g_active_ddr_submission_leases == 0);
        if (registration_completed) break;
    }
    assert(registration_completed && allocation_failures > 0);

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
    const int mixed_root_calls = SPM_ALLOC.root_calls;
    rpu_resolve_device_dma_endpoints(mixed_requests, 3, mixed);
    assert(SPM_ALLOC.root_calls == mixed_root_calls + SpmAllocator::NUM_CORES);
    assert(spm.address_reads == SpmAllocator::NUM_CORES &&
           spm.size_reads == SpmAllocator::NUM_CORES);
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

    // The owned path publishes exactly one private vector, with the same live
    // DDR/SPM authority and no second publication copy/allocation.
    {
        allocations_before_failure = 1;
        auto owned = rpu_resolve_device_dma_endpoints_owned(mixed_requests, 3);
        assert(allocations_before_failure == 0);
        allocations_before_failure = -1;
        assert(owned.lease() && g_active_ddr_submission_leases == 1);
        assert(owned.endpoints().size() == 3);
        for (size_t i = 0; i < 3; ++i) {
            assert(owned.endpoints()[i].owner == mixed[i].owner);
            assert(owned.endpoints()[i].offset == mixed[i].offset);
            assert(owned.endpoints()[i].allocation_id == mixed[i].allocation_id);
        }
        auto moved = std::move(owned);
        assert(!owned.lease() && moved.lease());
        RpuResolvedDeviceDmaBatch assigned;
        assigned = std::move(moved);
        assert(!moved.lease() && assigned.lease());
        assert(g_active_ddr_submission_leases == 1);
        auto replacement = rpu_resolve_device_dma_endpoints_owned(mixed_requests, 3);
        assert(g_active_ddr_submission_leases == 2);
        assigned = std::move(replacement);
        assert(!replacement.lease() && assigned.lease());
        assert(g_active_ddr_submission_leases == 1);

        // Late failure cannot replace an existing result or leak its partial
        // fresh lease; envelope failure must precede every SPM inspection.
        const RpuDeviceDmaEndpointRequest bad_owned[] = {
            mixed_requests[0], {device + 2048, 16, "owned late gap"}};
        expect_rejection([&] {
            assigned = rpu_resolve_device_dma_endpoints_owned(bad_owned, 2);
        });
        assert(assigned.endpoints().size() == 3 && assigned.lease());
        assert(assigned.endpoints()[0].allocation_id == first);
        assert(assigned.endpoints()[1].owner == &spm);
        assert(g_active_ddr_submission_leases == 1);
        const int roots_before_bad_envelope = SPM_ALLOC.root_calls;
        const RpuDeviceDmaEndpointRequest bad_envelope[] = {
            mixed_requests[1], {UINT64_MAX, 2, "owned envelope overflow"}};
        expect_rejection([&] {
            assigned = rpu_resolve_device_dma_endpoints_owned(bad_envelope, 2);
        });
        assert(SPM_ALLOC.root_calls == roots_before_bad_envelope);
        assert(assigned.endpoints().size() == 3 && assigned.lease());
        assert(g_active_ddr_submission_leases == 1);

        // The existing fault injector proves allocation failure happens before
        // SPM/DDR inspection and cannot publish or replace an active lease.
        allocations_before_failure = 0;
        bool allocation_failed = false;
        const int roots_before_oom = SPM_ALLOC.root_calls;
        try { assigned = rpu_resolve_device_dma_endpoints_owned(mixed_requests, 3); }
        catch (const std::bad_alloc&) { allocation_failed = true; }
        allocations_before_failure = -1;
        assert(allocation_failed && SPM_ALLOC.root_calls == roots_before_oom);
        assert(assigned.lease() && assigned.endpoints().size() == 3);
        assert(g_active_ddr_submission_leases == 1);
    }
    assert(g_active_ddr_submission_leases == 0);
    {
        const RpuDeviceDmaEndpointRequest spm_request{
            spm_device + 4095, 1, "owned SPM exact end"};
        auto owned_spm = rpu_resolve_device_dma_endpoints_owned(&spm_request, 1);
        assert(owned_spm.endpoints().size() == 1 && !owned_spm.lease());
        assert(owned_spm.endpoints()[0].owner == &spm &&
               owned_spm.endpoints()[0].offset == 4095 &&
               owned_spm.endpoints()[0].allocation_id == 0);
        const RpuDeviceDmaEndpointRequest ddr_request{
            device + 992, 32, "legacy count1 exact end"};
        RpuDmaEndpoint scalar_output;
        allocations_before_failure = 0;
        auto scalar_lease = rpu_resolve_device_dma_endpoints(&ddr_request, 1, &scalar_output);
        assert(allocations_before_failure == 0 && scalar_lease);
        assert(scalar_output.owner == &packed && scalar_output.offset == 992);
        const auto scalar = rpu_resolve_device_dma_endpoint(
            spm_device + 128, 32, "legacy scalar no allocation");
        assert(allocations_before_failure == 0 && scalar.owner == &spm);
        allocations_before_failure = -1;
    }
    assert(g_active_ddr_submission_leases == 0);

    // A single batch may reuse a DDR allocation across an intervening SPM
    // endpoint, switch owners, then return to the first logical allocation.
    // Exact ends are legal; neither owning-segment gaps nor range overruns are.
    const RpuDeviceDmaEndpointRequest repeated_requests[] = {
        {device + 64, 128, "repeated first allocation"},
        {spm_device + 4095, 1, "SPM exact end"},
        {device + 992, 32, "same allocation exact end"},
        {device + 4096, 32, "next logical allocation"},
        {device + 4096 + 2016, 32, "second allocation exact end"},
        {device + 8, 16, "return to first allocation"},
    };
    RpuDmaEndpoint repeated[6];
    const int repeated_root_calls = SPM_ALLOC.root_calls;
    const int repeated_address_reads = spm.address_reads;
    const int repeated_size_reads = spm.size_reads;
    auto repeated_lease = rpu_resolve_device_dma_endpoints(repeated_requests, 6, repeated);
    assert(repeated_lease && g_active_ddr_submission_leases == 1);
    assert(SPM_ALLOC.root_calls == repeated_root_calls + SpmAllocator::NUM_CORES);
    assert(spm.address_reads == repeated_address_reads + SpmAllocator::NUM_CORES);
    assert(spm.size_reads == repeated_size_reads + SpmAllocator::NUM_CORES);
    assert(repeated[0].allocation_id == first && repeated[0].offset == 64);
    assert(repeated[1].owner == &spm && repeated[1].offset == 4095);
    assert(repeated[2].allocation_id == first && repeated[2].offset == 992);
    assert(repeated[3].allocation_id == second && repeated[3].offset == 4096);
    assert(repeated[4].allocation_id == second && repeated[4].offset == 6112);
    assert(repeated[5].allocation_id == first && repeated[5].offset == 8);
    repeated_lease = {};
    assert(g_active_ddr_submission_leases == 0);
    for (const auto invalid : {
            RpuDeviceDmaEndpointRequest{device + 1023, 2, "last allocation overrun"},
            RpuDeviceDmaEndpointRequest{device + 1024, 1, "last allocation end gap"},
            RpuDeviceDmaEndpointRequest{spm_device + 4095, 2, "cached SPM overrun"}}) {
        const RpuDeviceDmaEndpointRequest requests[] = {
            {device + 8, 16, "valid repeated prefix"}, invalid};
        RpuDmaEndpoint output[] = {{&spm, 77, 91}, {&spm, 88, 92}};
        expect_rejection([&] { rpu_resolve_device_dma_endpoints(requests, 2, output); });
        assert(output[0].offset == 77 && output[0].allocation_id == 91);
        assert(output[1].offset == 88 && output[1].allocation_id == 92);
        assert(g_active_ddr_submission_leases == 0);
    }

    // Root inspection remains lazy and per call. A request satisfied by core
    // zero must not inspect an unrelated missing later core; a DDR request
    // that reaches that core still rejects without publishing earlier results.
    SPM_ALLOC.roots[7] = nullptr;
    const RpuDeviceDmaEndpointRequest early_spm[] = {
        {spm_device + 16, 32, "early SPM core"},
        {spm_device + 64, 32, "same early SPM core"}};
    RpuDmaEndpoint early_output[2];
    const int early_root_calls = SPM_ALLOC.root_calls;
    assert(!rpu_resolve_device_dma_endpoints(early_spm, 2, early_output));
    assert(SPM_ALLOC.root_calls == early_root_calls + 1);
    const RpuDeviceDmaEndpointRequest missing_root[] = {
        early_spm[0], {device + 16, 32, "later missing root"}};
    RpuDmaEndpoint missing_output[] = {{&spm, 77, 91}, {&spm, 88, 92}};
    expect_rejection([&] { rpu_resolve_device_dma_endpoints(missing_root, 2, missing_output); });
    assert(missing_output[0].allocation_id == 91 && missing_output[1].allocation_id == 92);
    assert(g_active_ddr_submission_leases == 0);
    SPM_ALLOC.roots.fill(&spm);

    // A changed root between batches cannot inherit the earlier snapshot.
    rhino_lkn::Buffer_t replacement_spm(spm_cpu + 8192, spm_device + 8192, 2048);
    SPM_ALLOC.roots.fill(&replacement_spm);
    expect_rejection([&] {
        rpu_resolve_device_dma_endpoints(early_spm, 2, missing_output);
    });
    assert(missing_output[0].allocation_id == 91 && missing_output[1].allocation_id == 92);
    auto replaced_endpoint = rpu_resolve_device_dma_endpoint(
        spm_device + 8192 + 2047, 1, "replacement root exact end");
    assert(replaced_endpoint.owner == &replacement_spm && replaced_endpoint.offset == 2047);
    SPM_ALLOC.initialized = false;
    const int disabled_root_calls = SPM_ALLOC.root_calls;
    expect_rejection([&] {
        rpu_resolve_device_dma_endpoint(spm_device + 8192, 16, "disabled SPM root");
    });
    assert(SPM_ALLOC.root_calls == disabled_root_calls);
    SPM_ALLOC.initialized = true;
    SPM_ALLOC.roots.fill(&spm);

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
    auto owned_device = rpu_resolve_device_dma_endpoints_owned(leased_device_request, 1);
    assert(owned_device.lease() && g_active_ddr_submission_leases == 4);
    auto moved_owned_device = std::move(owned_device);
    assert(!owned_device.lease() && moved_owned_device.lease());

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
    moved_device_lease = {};
    assert(g_active_ddr_submission_leases == 1);
    assert(moved_owned_device.lease());
    {
        std::unique_lock<std::mutex> guard(test_mutex);
        assert(!test_cv.wait_for(
            guard, std::chrono::milliseconds(20),
            [&] { return unregister_done; }));
    }
    moved_owned_device = {};
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

    // Exercise the real cold Graph coalescer against the real live registry.
    // Nonconsecutive ranges and exact duplicates of one allocation collapse;
    // two logical allocations within one DDR segment stay independent.
    std::vector<RpuDmaAllocationWitness> combined_witnesses{
        {first, device + 512, 128, "middle"},
        {first, device + 512, 128, "duplicate before new ID"},
        {second, device + 4096 + 1024, 16, "second middle"},
        {first, device, 16, "head"},
        {first, device + 1000, 24, "tail"},
        {second, device + 4096, 2048, "second whole"},
    };
    coalesce_fixed_dma_allocation_witnesses(combined_witnesses);
    assert(combined_witnesses.size() == 2);
    assert(combined_witnesses[0].allocation_id == first);
    assert(combined_witnesses[0].expected_device_addr == device);
    assert(combined_witnesses[0].bytes == 1024);
    assert(combined_witnesses[1].allocation_id == second);
    assert(combined_witnesses[1].expected_device_addr == device + 4096);
    assert(combined_witnesses[1].bytes == 2048);
    {
        auto lease = rpu_validate_live_dma_allocation_witnesses(
            combined_witnesses.data(), combined_witnesses.size());
        assert(lease);
    }
    std::vector<RpuDmaAllocationWitness> out_of_bounds{
        {first, device, 16, "valid prefix"},
        {first, device + 1023, 2, "invalid suffix"},
    };
    coalesce_fixed_dma_allocation_witnesses(out_of_bounds);
    expect_rejection([&] {
        rpu_validate_live_dma_allocation_witnesses(
            out_of_bounds.data(), out_of_bounds.size());
    });
    const uint64_t max_address = std::numeric_limits<uint64_t>::max();
    std::vector<RpuDmaAllocationWitness> address_limit{
        {first, max_address - 15, 8, "high head"},
        {first, max_address - 3, 4, "last byte"},
    };
    coalesce_fixed_dma_allocation_witnesses(address_limit);
    assert(address_limit.size() == 1 && address_limit[0].bytes == 16);
    assert(address_limit[0].expected_device_addr == max_address - 15);
    std::vector<RpuDmaAllocationWitness> overflow{
        {first, max_address - 7, 9, "overflow"},
    };
    expect_rejection([&] { coalesce_fixed_dma_allocation_witnesses(overflow); });

    const RpuDmaAllocationWitness live_witnesses[] = {
        {first, device + 128, 896, "first live allocation"},
        {0, 0, 0, nullptr},
        {second, device + 4096 + 1024, 1024, "second live allocation"},
    };
    rpu_validate_live_dma_allocation_witnesses(live_witnesses, 3);
    // A valid prefix must not publish a lease when a later allocation has a
    // missing address index or any one of the six fields disagrees with ID.
    const RpuDmaAllocationWitness consistency_witnesses[] = {
        {second, device + 4096, 32, "valid consistency prefix"},
        {first, device, 32, "damaged consistency suffix"},
    };
    auto check_index_consistency = [&](auto& index, auto key) {
        const auto saved = index.find(key)->second;
        assert(index.erase(key) == 1);
        assert(index.find(key) == index.end());
        expect_rejection([&] {
            rpu_validate_live_dma_allocation_witnesses(consistency_witnesses, 2);
        });
        assert(g_active_ddr_submission_leases == 0);
        assert(index.emplace(key, saved).second);
        for (int field = 0; field < 6; ++field) {
            auto damaged = saved;
            switch (field) {
                case 0: damaged.cpu_base = reinterpret_cast<void*>(cpu + 16); break;
                case 1: damaged.device_base += 16; break;
                case 2: --damaged.bytes; break;
                case 3: damaged.owner = &spm; break;
                case 4: ++damaged.owner_offset; break;
                case 5: ++damaged.allocation_id; break;
            }
            index.find(key)->second = damaged;
            expect_rejection([&] {
                rpu_validate_live_dma_allocation_witnesses(consistency_witnesses, 2);
            });
            assert(g_active_ddr_submission_leases == 0);
            index.find(key)->second = saved;
        }
        {
            auto lease = rpu_validate_live_dma_allocation_witnesses(
                consistency_witnesses, 2);
            assert(lease && g_active_ddr_submission_leases == 1);
        }
        assert(g_active_ddr_submission_leases == 0);
    };
    check_index_consistency(g_logical_ddr_by_cpu, cpu);
    check_index_consistency(g_logical_ddr_by_device, device);
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
    expect_rejection([&] {
        rpu_validate_live_dma_allocation_witnesses(
            combined_witnesses.data(), combined_witnesses.size());
    });
    assert(rpu_resolve_device_dma_endpoint(device, 16, "reused logical")
               .allocation_id == reused);
    const RpuDeviceDmaEndpointRequest reused_batch[] = {
        {device + 16, 16, "reused allocation first range"},
        {device + 1000, 24, "reused allocation second range"}};
    RpuDmaEndpoint reused_endpoints[2];
    rpu_resolve_device_dma_endpoints(reused_batch, 2, reused_endpoints);
    assert(reused_endpoints[0].allocation_id == reused &&
           reused_endpoints[1].allocation_id == reused);
    const RpuDmaAllocationWitness stale_after_aba[] = {
        {first, device, 16, "released generation"},
        {reused, device, 16, "new generation"},
    };
    expect_rejection([&] {
        rpu_validate_live_dma_allocation_witnesses(stale_after_aba, 2);
    });
    assert(g_active_ddr_submission_leases == 0);
    const RpuDmaAllocationWitness reused_live[] = {
        {reused, device + 1000, 24, "reused live generation"},
    };
    rpu_validate_live_dma_allocation_witnesses(reused_live, 1);
    assert(g_active_ddr_submission_leases == 0);
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
'''.replace("ENDPOINT_TYPES", endpoint_header).replace("REGISTRY", registry).replace("COALESCE_WITNESSES", coalesce)

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
        "struct RpuDeviceDmaEndpointResolver {",
    )
    batch = function(
        "struct RpuDeviceDmaEndpointResolver {",
        "RpuDmaSubmissionLease rpu_resolve_device_dma_endpoints(",
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
    legacy = function(
        "RpuDmaSubmissionLease rpu_resolve_device_dma_endpoints(",
        "RpuResolvedDeviceDmaBatch rpu_resolve_device_dma_endpoints_owned(",
    )
    owned = function(
        "RpuResolvedDeviceDmaBatch rpu_resolve_device_dma_endpoints_owned(",
        "RpuDmaEndpoint rpu_resolve_device_dma_endpoint(",
    )
    for entrypoint in (legacy, owned):
        assert "RpuDeviceDmaEndpointResolver::validate_envelope(" in entrypoint
        assert "RpuDeviceDmaEndpointResolver::resolve(" in entrypoint
        assert lock not in entrypoint
    assert "std::copy" not in owned
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
        cache.index("inline void rpu_add_kernel_mutable_checked(",
                    cache.index("inline void rpu_add_dma_mutable_checked(::rhino_lkn::Queue_t& q,\n                                        uint64_t"))
    ]
    for helper in (checked, mutable):
        assert "RpuDeviceDmaEndpointRequest requests[]" in helper
        assert "rpu_resolve_device_dma_endpoints(requests, 2, endpoints)" in helper
        assert "rpu_resolve_device_dma_endpoint(" not in helper
        assert helper.index("submission_lease") < helper.rindex("rpu_add_dma_")

    execute = GRAPH_EXECUTE.read_text(encoding="utf-8")
    ddr_copy = execute[
        execute.index("void RpuKernelGraph::execute_ddr_copy(") :
        execute.index("void RpuKernelGraph::execute_data_node(",
                      execute.index("void RpuKernelGraph::execute_ddr_copy("))
    ]
    assert "RpuCpuDmaEndpointRequest requests[]" in ddr_copy
    assert "rpu_resolve_cpu_dma_endpoints(requests, 2, endpoints)" in ddr_copy
    assert "rpu_resolve_cpu_dma_endpoint(" not in ddr_copy
    assert ddr_copy.index("submission_lease") < ddr_copy.index("rpu_add_dma_checked(")
    assert ddr_copy.index("rpu_add_dma_checked(") < ddr_copy.index("build_batch()")
    assert ddr_copy.index("build_batch()") < ddr_copy.index("enqueu_batch(/*wait_finish=*/true)")
    assert ddr_copy.index("enqueu_batch(/*wait_finish=*/true)") < ddr_copy.index("discard_completed_batch()")
    assert "data_dma_queue_.reset();" in ddr_copy
