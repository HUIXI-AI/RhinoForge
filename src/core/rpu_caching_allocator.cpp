// rpu_caching_allocator.cpp
// RPU Caching Allocator Implementation

#include "rpu_caching_allocator.h"
#include "rpu_allocator_policy.h"
#include "rpu_ops.h"
#include "rpu_dma_endpoint.h"
#include "rpu_spm_allocator.h"

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

struct ManagedDdrSegment {
    void* cpu_base = nullptr;
    uint64_t device_base = 0;
    size_t bytes = 0;
    ::rhino_lkn::Buffer_t* owner = nullptr;
};

struct LogicalDdrAllocation {
    void* cpu_base = nullptr;
    uint64_t device_base = 0;
    size_t bytes = 0;
    ::rhino_lkn::Buffer_t* owner = nullptr;
    size_t owner_offset = 0;
    uint64_t allocation_id = 0;
};

// Range resolution needs ordered addresses; replay witnesses need exact bases.
// Keep both access paths in one container so removing a range also retires its
// exact entry. Callers hold the registry mutex while using returned iterators.
template <typename Address>
class LogicalAddressIndex {
    using Ordered = std::map<Address, LogicalDdrAllocation>;
public:
    using iterator = typename Ordered::iterator;
    using const_iterator = typename Ordered::const_iterator;

    LogicalAddressIndex() = default;
    LogicalAddressIndex(const LogicalAddressIndex&) = delete;
    LogicalAddressIndex& operator=(const LogicalAddressIndex&) = delete;
    LogicalAddressIndex(LogicalAddressIndex&&) = delete;
    LogicalAddressIndex& operator=(LogicalAddressIndex&&) = delete;

    iterator begin() { return ordered_.begin(); }
    const_iterator begin() const { return ordered_.begin(); }
    iterator end() { return ordered_.end(); }
    const_iterator end() const { return ordered_.end(); }
    iterator lower_bound(Address key) { return ordered_.lower_bound(key); }
    const_iterator lower_bound(Address key) const { return ordered_.lower_bound(key); }
    iterator upper_bound(Address key) { return ordered_.upper_bound(key); }
    const_iterator upper_bound(Address key) const { return ordered_.upper_bound(key); }
    size_t size() const { return ordered_.size(); }
    bool empty() const { return ordered_.empty(); }
    size_t count(Address key) const { return exact_.count(key); }

    iterator find(Address key) {
        const auto found = exact_.find(key);
        return found == exact_.end() ? ordered_.end() : found->second;
    }
    const_iterator find(Address key) const {
        const auto found = exact_.find(key);
        return found == exact_.end() ? ordered_.end() : const_iterator(found->second);
    }
    std::pair<iterator, bool> emplace(Address key, const LogicalDdrAllocation& value) {
        const auto inserted = ordered_.emplace(key, value);
        if (!inserted.second) return inserted;
        try {
            TORCH_CHECK(exact_.emplace(key, inserted.first).second,
                        "RPU DDR allocator: duplicate logical address index");
        } catch (...) {
            ordered_.erase(inserted.first);
            throw;
        }
        return inserted;
    }
    iterator erase(iterator position) {
        exact_.erase(position->first);
        return ordered_.erase(position);
    }
    size_t erase(Address key) {
        const auto position = find(key);
        if (position == end()) return 0;
        erase(position);
        return 1;
    }
    void clear() {
        exact_.clear();
        ordered_.clear();
    }

private:
    Ordered ordered_;
    std::unordered_map<Address, iterator> exact_;
};

std::mutex g_managed_ddr_mutex;
std::condition_variable g_managed_ddr_lease_cv;
size_t g_active_ddr_submission_leases = 0;
std::map<uintptr_t, ManagedDdrSegment> g_managed_ddr_segments_by_cpu;
std::map<uint64_t, ManagedDdrSegment> g_managed_ddr_segments_by_device;
LogicalAddressIndex<uintptr_t> g_logical_ddr_by_cpu;
LogicalAddressIndex<uint64_t> g_logical_ddr_by_device;
std::unordered_map<uint64_t, LogicalDdrAllocation> g_logical_ddr_by_allocation_id;
uint64_t g_next_logical_ddr_allocation_id = 1;

bool cpu_range_fits_address_space(uintptr_t base, size_t bytes) {
    return bytes > 0 &&
        static_cast<uint64_t>(bytes - 1) <=
            std::numeric_limits<uintptr_t>::max() - base;
}

bool device_range_fits_address_space(uint64_t base, size_t bytes) {
    return bytes > 0 &&
        static_cast<uint64_t>(bytes - 1) <=
            std::numeric_limits<uint64_t>::max() - base;
}

bool cpu_range_contains(uintptr_t base, size_t capacity,
                        uintptr_t current, size_t bytes) {
    if (current < base) return false;
    const uintptr_t delta = current - base;
    return delta <= capacity && bytes <= capacity - static_cast<size_t>(delta);
}

bool device_range_contains(uint64_t base, size_t capacity,
                           uint64_t current, size_t bytes) {
    if (current < base) return false;
    const uint64_t delta = current - base;
    return delta <= capacity && bytes <= capacity - static_cast<size_t>(delta);
}

template <typename Map, typename Address>
bool interval_is_available(const Map& allocations, Address base, size_t bytes) {
    const auto next = allocations.lower_bound(base);
    if (next != allocations.end()) {
        if (next->first == base || next->first - base < bytes) return false;
    }
    if (next != allocations.begin()) {
        const auto previous = std::prev(next);
        if (base - previous->first < previous->second.bytes) return false;
    }
    return true;
}

const ManagedDdrSegment* find_cpu_segment_locked(
        uintptr_t current, size_t bytes) {
    auto it = g_managed_ddr_segments_by_cpu.upper_bound(current);
    if (it == g_managed_ddr_segments_by_cpu.begin()) return nullptr;
    --it;
    return cpu_range_contains(it->first, it->second.bytes, current, bytes)
        ? &it->second : nullptr;
}

const LogicalDdrAllocation* find_cpu_logical_locked(
        uintptr_t current, size_t bytes) {
    auto it = g_logical_ddr_by_cpu.upper_bound(current);
    if (it == g_logical_ddr_by_cpu.begin()) return nullptr;
    --it;
    return cpu_range_contains(it->first, it->second.bytes, current, bytes)
        ? &it->second : nullptr;
}

const LogicalDdrAllocation* find_device_logical_locked(
        uint64_t current, size_t bytes) {
    auto it = g_logical_ddr_by_device.upper_bound(current);
    if (it == g_logical_ddr_by_device.begin()) return nullptr;
    --it;
    return device_range_contains(it->first, it->second.bytes, current, bytes)
        ? &it->second : nullptr;
}

bool same_managed_ddr_segment(
        const ManagedDdrSegment& lhs, const ManagedDdrSegment& rhs) {
    return lhs.cpu_base == rhs.cpu_base &&
        lhs.device_base == rhs.device_base && lhs.bytes == rhs.bytes &&
        lhs.owner == rhs.owner;
}

bool same_logical_ddr_allocation(
        const LogicalDdrAllocation& lhs,
        const LogicalDdrAllocation& rhs) {
    return lhs.cpu_base == rhs.cpu_base &&
        lhs.device_base == rhs.device_base && lhs.bytes == rhs.bytes &&
        lhs.owner == rhs.owner && lhs.owner_offset == rhs.owner_offset &&
        lhs.allocation_id == rhs.allocation_id;
}

}  // namespace

RpuDmaSubmissionLease::~RpuDmaSubmissionLease() {
    release();
}

RpuDmaSubmissionLease::RpuDmaSubmissionLease(
        RpuDmaSubmissionLease&& other) noexcept
    : active_(other.active_) {
    other.active_ = false;
}

RpuDmaSubmissionLease& RpuDmaSubmissionLease::operator=(
        RpuDmaSubmissionLease&& other) noexcept {
    if (this == &other) return *this;
    release();
    active_ = other.active_;
    other.active_ = false;
    return *this;
}

void RpuDmaSubmissionLease::release() noexcept {
    if (!active_) return;
    {
        std::lock_guard<std::mutex> lock(g_managed_ddr_mutex);
        if (g_active_ddr_submission_leases == 0) {
            // A move-only lease can release exactly once.  Keep a corrupted
            // counter fail-closed without throwing from a destructor.
            active_ = false;
            return;
        }
        --g_active_ddr_submission_leases;
        active_ = false;
    }
    g_managed_ddr_lease_cv.notify_all();
}

void rpu_register_ddr_dma_segment(void* cpu_base, size_t bytes) {
    TORCH_CHECK(cpu_base != nullptr && bytes > 0,
                "RPU DDR allocator: cannot register an empty HostDDR segment");
    auto* owner = ::rhino_lkn::RpuGetHostddr(cpu_base);
    TORCH_CHECK(owner != nullptr,
                "RPU DDR allocator: RpuDdrAlloc returned an unregistered "
                "HostDDR allocation");
    const uint64_t device_base = owner->get_rpu_addr();
    const size_t owner_bytes = owner->get_memory_size();
    const uintptr_t cpu_address = reinterpret_cast<uintptr_t>(cpu_base);
    TORCH_CHECK(device_base != 0 && owner->get_cpu_ptr() == cpu_base &&
                    bytes <= owner_bytes &&
                    cpu_range_fits_address_space(cpu_address, owner_bytes) &&
                    device_range_fits_address_space(device_base, owner_bytes),
                "RPU DDR allocator: inconsistent managed DDR allocation");
    const ManagedDdrSegment segment{
        cpu_base, device_base, owner_bytes, owner};
    std::lock_guard<std::mutex> lock(g_managed_ddr_mutex);
    TORCH_CHECK(interval_is_available(
                    g_managed_ddr_segments_by_cpu, cpu_address, owner_bytes) &&
                    interval_is_available(
                        g_managed_ddr_segments_by_device,
                        device_base, owner_bytes),
                "RPU DDR allocator: overlapping managed HostDDR segment");
    const bool cpu_inserted =
        g_managed_ddr_segments_by_cpu.emplace(cpu_address, segment).second;
    TORCH_CHECK(cpu_inserted,
                "RPU DDR allocator: duplicate managed DDR CPU base");
    try {
        const bool device_inserted =
            g_managed_ddr_segments_by_device.emplace(device_base, segment).second;
        TORCH_CHECK(device_inserted,
                    "RPU DDR allocator: duplicate managed DDR device base");
    } catch (...) {
        g_managed_ddr_segments_by_cpu.erase(cpu_address);
        throw;
    }
}

void rpu_unregister_ddr_dma_segment(void* cpu_base) {
    if (cpu_base == nullptr) return;
    const uintptr_t cpu_address = reinterpret_cast<uintptr_t>(cpu_base);
    std::lock_guard<std::mutex> lock(g_managed_ddr_mutex);
    const auto segment_it = g_managed_ddr_segments_by_cpu.find(cpu_address);
    if (segment_it == g_managed_ddr_segments_by_cpu.end()) return;
    const auto& segment = segment_it->second;
    auto logical_it = g_logical_ddr_by_cpu.lower_bound(cpu_address);
    TORCH_CHECK(
        logical_it == g_logical_ddr_by_cpu.end() ||
            !cpu_range_contains(cpu_address, segment.bytes,
                                logical_it->first, /*bytes=*/1),
        "RPU DDR allocator: cannot release a HostDDR segment containing a "
        "live logical allocation");
    const auto device_it =
        g_managed_ddr_segments_by_device.find(segment.device_base);
    TORCH_CHECK(device_it != g_managed_ddr_segments_by_device.end() &&
                    same_managed_ddr_segment(segment, device_it->second),
                "RPU DDR allocator: managed HostDDR segment indexes disagree");
    g_managed_ddr_segments_by_device.erase(device_it);
    g_managed_ddr_segments_by_cpu.erase(segment_it);
}

uint64_t rpu_register_ddr_dma_logical_allocation(
        void* cpu_base, size_t bytes) {
    TORCH_CHECK(cpu_base != nullptr && bytes > 0,
                "RPU DDR allocator: cannot register an empty logical allocation");
    auto* owner = ::rhino_lkn::RpuGetHostddr(cpu_base);
    TORCH_CHECK(owner != nullptr,
                "RPU DDR allocator: logical allocation has no HostDDR owner");
    const uintptr_t cpu_address = reinterpret_cast<uintptr_t>(cpu_base);
    const uintptr_t owner_cpu =
        reinterpret_cast<uintptr_t>(owner->get_cpu_ptr());
    TORCH_CHECK(cpu_range_contains(owner_cpu, owner->get_memory_size(),
                                   cpu_address, bytes),
                "RPU DDR allocator: logical allocation exceeds its HostDDR owner");
    const size_t owner_offset = static_cast<size_t>(cpu_address - owner_cpu);
    TORCH_CHECK(owner->get_rpu_addr() <=
                    std::numeric_limits<uint64_t>::max() - owner_offset,
                "RPU DDR allocator: logical device address overflows");
    const uint64_t device_base = owner->get_rpu_addr() + owner_offset;

    std::lock_guard<std::mutex> lock(g_managed_ddr_mutex);
    const auto* segment = find_cpu_segment_locked(cpu_address, bytes);
    TORCH_CHECK(segment != nullptr && segment->owner == owner &&
                    segment->device_base <= device_base &&
                    device_base - segment->device_base == owner_offset,
                "RPU DDR allocator: logical allocation is not backed by its "
                "registered HostDDR segment");
    TORCH_CHECK(interval_is_available(
                    g_logical_ddr_by_cpu, cpu_address, bytes) &&
                    interval_is_available(
                        g_logical_ddr_by_device, device_base, bytes),
                "RPU DDR allocator: overlapping live logical allocation");
    TORCH_CHECK(g_next_logical_ddr_allocation_id != 0 &&
                    g_next_logical_ddr_allocation_id !=
                        std::numeric_limits<uint64_t>::max(),
                "RPU DDR allocator: logical allocation identity exhausted");
    const uint64_t allocation_id = g_next_logical_ddr_allocation_id++;
    const LogicalDdrAllocation allocation{
        cpu_base, device_base, bytes, owner, owner_offset, allocation_id};
    bool cpu_inserted = false;
    bool device_inserted = false;
    bool id_inserted = false;
    try {
        cpu_inserted =
            g_logical_ddr_by_cpu.emplace(cpu_address, allocation).second;
        TORCH_CHECK(cpu_inserted,
                    "RPU DDR allocator: duplicate logical DDR CPU base");
        device_inserted =
            g_logical_ddr_by_device.emplace(device_base, allocation).second;
        TORCH_CHECK(device_inserted,
                    "RPU DDR allocator: duplicate logical DDR device base");
        id_inserted = g_logical_ddr_by_allocation_id.emplace(
            allocation_id, allocation).second;
        TORCH_CHECK(id_inserted,
                    "RPU DDR allocator: duplicate logical allocation identity");
    } catch (...) {
        if (id_inserted)
            g_logical_ddr_by_allocation_id.erase(allocation_id);
        if (device_inserted) g_logical_ddr_by_device.erase(device_base);
        if (cpu_inserted) g_logical_ddr_by_cpu.erase(cpu_address);
        throw;
    }
    return allocation_id;
}

bool rpu_unregister_ddr_dma_logical_allocation(
        void* cpu_base, uint64_t allocation_id) noexcept {
    if (cpu_base == nullptr) return false;
    try {
        const uintptr_t cpu_address = reinterpret_cast<uintptr_t>(cpu_base);
        std::unique_lock<std::mutex> lock(g_managed_ddr_mutex);
        auto cpu_it = g_logical_ddr_by_cpu.find(cpu_address);
        if (cpu_it == g_logical_ddr_by_cpu.end() ||
            (allocation_id != 0 &&
             cpu_it->second.allocation_id != allocation_id)) {
            return false;
        }
        const uint64_t expected_allocation_id =
            cpu_it->second.allocation_id;

        // Queue_t consumes raw Buffer_t pointers after endpoint resolution.
        // Wait without holding the registry mutex until every synchronous
        // submission lease has retired; only then may a direct allocation be
        // freed or a caching block become reusable at the same address.
        g_managed_ddr_lease_cv.wait(lock, [] {
            return g_active_ddr_submission_leases == 0;
        });

        // The condition-variable wait released the mutex. Revalidate all three
        // indexes before erasing so teardown remains transactional even if an
        // unrelated allocation changed while this deleter slept.
        cpu_it = g_logical_ddr_by_cpu.find(cpu_address);
        if (cpu_it == g_logical_ddr_by_cpu.end() ||
            cpu_it->second.allocation_id != expected_allocation_id) {
            return false;
        }
        const uint64_t device_base = cpu_it->second.device_base;
        const auto device_it = g_logical_ddr_by_device.find(device_base);
        if (device_it == g_logical_ddr_by_device.end() ||
            !same_logical_ddr_allocation(cpu_it->second, device_it->second)) {
            return false;
        }
        const auto id_it =
            g_logical_ddr_by_allocation_id.find(expected_allocation_id);
        if (id_it == g_logical_ddr_by_allocation_id.end() ||
            !same_logical_ddr_allocation(cpu_it->second, id_it->second)) {
            return false;
        }
        g_logical_ddr_by_allocation_id.erase(id_it);
        g_logical_ddr_by_device.erase(device_it);
        g_logical_ddr_by_cpu.erase(cpu_it);
        return true;
    } catch (...) {
        // Tensor deleters are noexcept.  A synchronization failure leaves the
        // allocation registered and therefore quarantined rather than risking
        // reuse while a Queue may still hold its Launch owner.
        return false;
    }
}

void rpu_register_ddr_dma_allocation(void* cpu_base, size_t bytes) {
    rpu_register_ddr_dma_segment(cpu_base, bytes);
    try {
        (void)rpu_register_ddr_dma_logical_allocation(cpu_base, bytes);
    } catch (...) {
        rpu_unregister_ddr_dma_segment(cpu_base);
        throw;
    }
}

void rpu_unregister_ddr_dma_allocation(void* cpu_base) {
    if (cpu_base == nullptr) return;
    (void)rpu_unregister_ddr_dma_logical_allocation(
        cpu_base, /*allocation_id=*/0);
    rpu_unregister_ddr_dma_segment(cpu_base);
}

void rpu_free_registered_ddr(void* cpu_base) noexcept {
    if (cpu_base == nullptr) return;
    try {
        // Registry identity is local process state and remains valid after the
        // SDK manager has shut down, so retire it for every late deleter.
        rpu_unregister_ddr_dma_allocation(cpu_base);
    } catch (...) {
        // Never free a physical segment whose registry teardown failed: it may
        // still contain a live logical allocation. Tensor deleters cannot
        // propagate an exception during object destruction.
        return;
    }
    (void)rpu_tensor_ddr_lifecycle().run_free_if_live([cpu_base] {
        ::rhino_lkn::RpuDdrFree(cpu_base);
    });
}

RpuDmaEndpoint rpu_resolve_cpu_dma_endpoint(
        const void* cpu_ptr, size_t bytes, const char* where) {
    TORCH_CHECK(cpu_ptr != nullptr && bytes > 0,
                where, ": null DDR pointer or empty DMA range");
    const uintptr_t current = reinterpret_cast<uintptr_t>(cpu_ptr);
    std::lock_guard<std::mutex> lock(g_managed_ddr_mutex);
    const auto* allocation = find_cpu_logical_locked(current, bytes);
    TORCH_CHECK(allocation != nullptr,
                where, ": CPU DDR DMA range is outside every live logical "
                "allocation");
    const size_t logical_offset =
        static_cast<size_t>(current -
                            reinterpret_cast<uintptr_t>(allocation->cpu_base));
    return RpuDmaEndpoint{
        allocation->owner, allocation->owner_offset + logical_offset,
        allocation->allocation_id};
}

RpuDmaSubmissionLease rpu_resolve_cpu_dma_endpoints(
        const RpuCpuDmaEndpointRequest* requests,
        size_t count,
        RpuDmaEndpoint* endpoints) {
    TORCH_CHECK(count == 0 || (requests != nullptr && endpoints != nullptr),
                "rpu_resolve_cpu_dma_endpoints: null batch storage");
    if (count == 0) return {};

    for (size_t i = 0; i < count; ++i) {
        TORCH_CHECK(requests[i].where != nullptr,
                    "rpu_resolve_cpu_dma_endpoints: request ", i,
                    " has no diagnostic context");
        TORCH_CHECK(requests[i].cpu_ptr != nullptr && requests[i].bytes > 0,
                    requests[i].where,
                    ": null DDR pointer or empty DMA range");
        TORCH_CHECK(cpu_range_fits_address_space(
                        reinterpret_cast<uintptr_t>(requests[i].cpu_ptr),
                        requests[i].bytes),
                    requests[i].where,
                    ": CPU DDR DMA range overflows the address space");
    }

    RpuDmaEndpoint scalar_resolved;
    std::vector<RpuDmaEndpoint> batch_resolved;
    RpuDmaEndpoint* resolved = &scalar_resolved;
    if (count > 1) {
        batch_resolved.resize(count);
        resolved = batch_resolved.data();
    }

    std::lock_guard<std::mutex> lock(g_managed_ddr_mutex);
    for (size_t i = 0; i < count; ++i) {
        const uintptr_t current =
            reinterpret_cast<uintptr_t>(requests[i].cpu_ptr);
        const auto* allocation =
            find_cpu_logical_locked(current, requests[i].bytes);
        TORCH_CHECK(allocation != nullptr,
                    requests[i].where,
                    ": CPU DDR DMA range is outside every live logical "
                    "allocation");
        const size_t logical_offset = static_cast<size_t>(
            current - reinterpret_cast<uintptr_t>(allocation->cpu_base));
        resolved[i] = RpuDmaEndpoint{
            allocation->owner,
            allocation->owner_offset + logical_offset,
            allocation->allocation_id};
    }
    TORCH_CHECK(
        g_active_ddr_submission_leases !=
            std::numeric_limits<size_t>::max(),
        "RPU DDR allocator: DMA submission lease count exhausted");
    ++g_active_ddr_submission_leases;
    std::copy(resolved, resolved + count, endpoints);
    return RpuDmaSubmissionLease(true);
}

RpuDmaSubmissionLease rpu_resolve_device_dma_endpoints(
        const RpuDeviceDmaEndpointRequest* requests,
        size_t count,
        RpuDmaEndpoint* endpoints) {
    TORCH_CHECK(count == 0 || (requests != nullptr && endpoints != nullptr),
                "rpu_resolve_device_dma_endpoints: null batch storage");
    if (count == 0) return {};

    // Validate the complete request envelope before consulting either address
    // space. This also keeps caller-owned output unchanged on malformed input.
    for (size_t i = 0; i < count; ++i) {
        TORCH_CHECK(requests[i].where != nullptr,
                    "rpu_resolve_device_dma_endpoints: request ", i,
                    " has no diagnostic context");
        TORCH_CHECK(requests[i].device_addr != 0 && requests[i].bytes > 0,
                    requests[i].where,
                    ": zero device address or empty DMA range");
        TORCH_CHECK(device_range_fits_address_space(
                        requests[i].device_addr, requests[i].bytes),
                    requests[i].where,
                    ": device DMA range overflows the address space");
    }

    // Keep the compatibility scalar wrapper allocation-free while retaining a
    // private publication buffer for a real batch.
    RpuDmaEndpoint scalar_resolved;
    std::vector<RpuDmaEndpoint> batch_resolved;
    RpuDmaEndpoint* resolved = &scalar_resolved;
    if (count > 1) {
        batch_resolved.resize(count);
        resolved = batch_resolved.data();
    }
    // Root Buffer metadata is immutable during one resolution batch. Load a
    // core lazily so an endpoint that resolves on an earlier core keeps the
    // existing validation order. Never retain roots across calls/lifetimes.
    struct SpmRootSnapshot {
        ::rhino_lkn::Buffer_t* root = nullptr;
        uint64_t base = 0;
        size_t bytes = 0;
    };
    SpmRootSnapshot spm_roots[SpmAllocator::NUM_CORES];
    bool has_ddr_request = false;
    const bool spm_initialized = SPM_ALLOC.is_initialized();
    for (size_t i = 0; i < count; ++i) {
        if (spm_initialized) {
            for (int core = 0; core < SpmAllocator::NUM_CORES; ++core) {
                auto& snapshot = spm_roots[core];
                if (snapshot.root == nullptr) {
                    snapshot.root = SPM_ALLOC.root_buffer(core);
                    TORCH_CHECK(
                        snapshot.root != nullptr,
                        requests[i].where,
                        ": initialized SPM allocator has no root for core ", core);
                    snapshot.base = snapshot.root->get_rpu_addr();
                    snapshot.bytes = snapshot.root->get_memory_size();
                }
                if (requests[i].device_addr >= snapshot.base) {
                    const uint64_t delta = requests[i].device_addr - snapshot.base;
                    if (delta <= snapshot.bytes &&
                        requests[i].bytes <= snapshot.bytes - delta) {
                        resolved[i] = RpuDmaEndpoint{
                            snapshot.root, static_cast<size_t>(delta),
                            /*allocation_id=*/0};
                        break;
                    }
                }
            }
        }
        has_ddr_request = has_ddr_request || !resolved[i];
    }

    if (has_ddr_request) {
        std::lock_guard<std::mutex> lock(g_managed_ddr_mutex);
        // Registry ranges are disjoint and cannot change under this lock.
        // Repeated controller ranges inside the last allocation therefore
        // resolve to exactly the same owner as another map lookup would.
        const LogicalDdrAllocation* previous_allocation = nullptr;
        for (size_t i = 0; i < count; ++i) {
            if (resolved[i]) continue;
            const auto* allocation = previous_allocation;
            if (allocation == nullptr || !device_range_contains(
                    allocation->device_base, allocation->bytes,
                    requests[i].device_addr, requests[i].bytes)) {
                allocation = find_device_logical_locked(
                    requests[i].device_addr, requests[i].bytes);
            }
            TORCH_CHECK(
                allocation != nullptr,
                requests[i].where,
                ": device DDR DMA range is outside every live logical "
                "allocation or SPM_ALLOC root");
            previous_allocation = allocation;
            const size_t logical_offset = static_cast<size_t>(
                requests[i].device_addr - allocation->device_base);
            resolved[i] = RpuDmaEndpoint{
                allocation->owner,
                allocation->owner_offset + logical_offset,
                allocation->allocation_id};
        }
        TORCH_CHECK(
            g_active_ddr_submission_leases !=
                std::numeric_limits<size_t>::max(),
            "RPU DDR allocator: DMA submission lease count exhausted");
        ++g_active_ddr_submission_leases;
        std::copy(resolved, resolved + count, endpoints);
        return RpuDmaSubmissionLease(true);
    }

    std::copy(resolved, resolved + count, endpoints);
    return {};
}

RpuDmaEndpoint rpu_resolve_device_dma_endpoint(
        uint64_t device_addr, size_t bytes, const char* where) {
    const RpuDeviceDmaEndpointRequest request{device_addr, bytes, where};
    RpuDmaEndpoint endpoint;
    // Compatibility scalar queries expose endpoint metadata only. Queue paths
    // that dereference Buffer_t across add/build/submit use the batch API and
    // retain its returned lease explicitly.
    (void)rpu_resolve_device_dma_endpoints(
        &request, /*count=*/1, &endpoint);
    return endpoint;
}

RpuDmaSubmissionLease rpu_validate_live_dma_allocation_witnesses(
        const RpuDmaAllocationWitness* witnesses,
        size_t count) {
    TORCH_CHECK(count == 0 || witnesses != nullptr,
                "rpu_validate_live_dma_allocation_witnesses: null batch");
    if (count == 0) return {};

    bool has_ddr_witness = false;
    for (size_t i = 0; i < count; ++i) {
        if (witnesses[i].allocation_id == 0) continue;
        has_ddr_witness = true;
        TORCH_CHECK(witnesses[i].where != nullptr,
                    "rpu_validate_live_dma_allocation_witnesses: witness ", i,
                    " has no diagnostic context");
        TORCH_CHECK(witnesses[i].expected_device_addr != 0 &&
                        witnesses[i].bytes > 0,
                    witnesses[i].where,
                    ": invalid live DDR allocation witness range");
        TORCH_CHECK(device_range_fits_address_space(
                        witnesses[i].expected_device_addr,
                        witnesses[i].bytes),
                    witnesses[i].where,
                    ": live DDR allocation witness range overflows the address "
                    "space");
    }
    if (!has_ddr_witness) return {};

    std::lock_guard<std::mutex> lock(g_managed_ddr_mutex);
    // Controller-striped transfers commonly repeat one logical allocation for
    // several ranges. Its identity and registry indexes cannot change while
    // this lock is held. Reuse that lookup within this batch only; each range
    // is still checked, and the next submission always revalidates lifetime.
    const LogicalDdrAllocation* previous_allocation = nullptr;
    for (size_t i = 0; i < count; ++i) {
        const auto& witness = witnesses[i];
        if (witness.allocation_id == 0) continue;
        if (previous_allocation == nullptr ||
            previous_allocation->allocation_id != witness.allocation_id) {
            const auto id_it =
                g_logical_ddr_by_allocation_id.find(witness.allocation_id);
            TORCH_CHECK(id_it != g_logical_ddr_by_allocation_id.end(),
                        witness.where,
                        ": logical DDR allocation identity is no longer live");
            const auto& allocation = id_it->second;
            const auto cpu_it = g_logical_ddr_by_cpu.find(
                reinterpret_cast<uintptr_t>(allocation.cpu_base));
            const auto device_it =
                g_logical_ddr_by_device.find(allocation.device_base);
            TORCH_CHECK(
                cpu_it != g_logical_ddr_by_cpu.end() &&
                    device_it != g_logical_ddr_by_device.end() &&
                    same_logical_ddr_allocation(
                        allocation, cpu_it->second) &&
                    same_logical_ddr_allocation(
                        allocation, device_it->second),
                witness.where,
                ": logical DDR allocation registry indexes disagree");
            previous_allocation = &allocation;
        }
        const auto& allocation = *previous_allocation;
        TORCH_CHECK(
            device_range_contains(
                allocation.device_base, allocation.bytes,
                witness.expected_device_addr, witness.bytes),
            witness.where,
            ": logical DDR allocation identity does not own the expected "
            "device range");
    }
    TORCH_CHECK(
        g_active_ddr_submission_leases !=
            std::numeric_limits<size_t>::max(),
        "RPU DDR allocator: DMA submission lease count exhausted");
    ++g_active_ddr_submission_leases;
    return RpuDmaSubmissionLease(true);
}

namespace rpu {

// ============================================================================
// Block Comparators
// ============================================================================

bool BlockComparatorSize::operator()(const Block* a, const Block* b) const {
    // First compare by size
    if (a->size != b->size) {
        return a->size < b->size;
    }
    // Then by address for deterministic ordering
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

bool BlockComparatorAddress::operator()(const Block* a, const Block* b) const {
    return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
}

// ============================================================================
// RPUCachingAllocator Implementation
// ============================================================================

RPUCachingAllocator::RPUCachingAllocator()
    : small_blocks_(/*small=*/true),
      large_blocks_(/*small=*/false) {
}

RPUCachingAllocator::~RPUCachingAllocator() {
    // Note: We intentionally don't call emptyCache() here.
    // During program shutdown, the order of static object destruction
    // is undefined, and calling RpuDdrFree at this point may cause
    // issues if the RPU runtime has already been deinitialized.
    // The OS will reclaim all memory when the process exits anyway.
}

size_t RPUCachingAllocator::round_size(size_t size) {
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    }
    // Round up to multiple of kMinBlockSize
    // Since kMinBlockSize is a multiple of kAlignment (32 bytes),
    // all returned sizes are guaranteed to be 32-byte aligned,
    // which is required by CopyMemory DMA transfers
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
}

size_t RPUCachingAllocator::get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
        // Small allocations: use small buffer size
        return kSmallBuffer;
    }
    // Bound HostDDR reservation for live medium-sized model weights instead
    // of reserving a 20 MiB segment for a request just above 1 MiB.
    return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
}

BlockPool& RPUCachingAllocator::get_pool(size_t size) {
    if (size <= kSmallSize) {
        return small_blocks_;
    }
    return large_blocks_;
}

Block* RPUCachingAllocator::get_free_block(size_t size, BlockPool& pool) {
    // Create a search key block
    Block search_key(size);

    // Find the smallest block that can satisfy the request
    auto it = pool.blocks.lower_bound(&search_key);

    if (it == pool.blocks.end()) {
        return nullptr;
    }

    Block* block = *it;

    // For large requests, don't return an oversized block if max_split_size would prevent splitting
    if (size < kMaxSplitSize && block->size >= kMaxSplitSize) {
        return nullptr;
    }

    // Remove from pool
    pool.erase(block);
    if (block->is_split()) {
        stats_.inactive_split_bytes.decrease(block->size);
    }

    return block;
}

// O(1) fast path for size-class allocations
Block* RPUCachingAllocator::get_free_block_fast(int size_class) {
    Block* block = size_class_lists_[size_class];
    if (block) {
        // Pop from head of list
        size_class_lists_[size_class] = block->next;
        block->next = nullptr;
        size_class_counts_[size_class]--;
    }
    return block;
}

// O(1) fast path for returning blocks to size-class lists
void RPUCachingAllocator::free_block_fast(Block* block, int size_class) {
    // Push to head of list
    block->next = size_class_lists_[size_class];
    block->prev = nullptr;  // Not used in size-class lists
    size_class_lists_[size_class] = block;
    size_class_counts_[size_class]++;
}

Block* RPUCachingAllocator::alloc_block(size_t size, BlockPool& pool) {
    size_t alloc_size = get_allocation_size(size);

    // Allocate from system
    void* ptr = rhino_lkn::RpuDdrAlloc(alloc_size);

    if (!ptr) {
        // Allocation failed
        stats_.num_ooms++;
        return nullptr;
    }

    // Verify that RpuDdrAlloc returns 32-byte aligned memory
    // This is required for CopyMemory DMA transfers
    if (reinterpret_cast<uintptr_t>(ptr) % kAlignment != 0) {
        if (log_at(1)) {
            std::cerr << "[RPUCachingAllocator] Error: RpuDdrAlloc returned unaligned pointer "
                      << ptr << " (alignment: " << (reinterpret_cast<uintptr_t>(ptr) % kAlignment)
                      << ", required: " << kAlignment << ")" << std::endl;
        }
        rhino_lkn::RpuDdrFree(ptr);
        stats_.num_ooms++;
        return nullptr;
    }

    // Create new block
    Block* block = new Block(ptr, alloc_size, &pool);
    try {
        rpu_register_ddr_dma_segment(ptr, alloc_size);
    } catch (...) {
        delete block;
        ::rhino_lkn::RpuDdrFree(ptr);
        throw;
    }

    // Update statistics
    total_allocated_memory_ += alloc_size;
    stats_.reserved_bytes.increase(alloc_size);
    stats_.segment.increase(1);
    stats_.num_device_alloc++;

    return block;
}

bool RPUCachingAllocator::should_split(const Block* block, size_t size) const {
    size_t remaining = block->size - size;

    if (block->pool->is_small) {
        // Small pool: split if there's at least kMinBlockSize remaining
        return remaining >= kMinBlockSize;
    } else {
        // Large pool: split if request is smaller than max_split_size
        // and remaining is larger than kSmallSize
        return (size < kMaxSplitSize) && (remaining > kSmallSize);
    }
}

void* RPUCachingAllocator::malloc(size_t orig_size) {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    if (shutdown_called_) {
        throw std::runtime_error("RPU caching allocator is shut down");
    }

    // Handle zero-size allocation
    if (orig_size == 0) {
        return nullptr;
    }

    // Round up the size
    size_t size = round_size(orig_size);

    // Try size-class fast path first (O(1) for sizes <= 1MB)
    int size_class = size_to_class(size);
    Block* block = nullptr;

    if (size_class >= 0) {
        // Use size class - round up to class size for exact match
        size = class_to_size(size_class);
        block = get_free_block_fast(size_class);
    }

    // Fallback to pool-based allocation if fast path didn't find a block
    if (!block) {
        // Get the appropriate pool
        BlockPool& pool = get_pool(size);

        // Try to get a free block from cache
        block = get_free_block(size, pool);

        if (!block) {
            // No suitable cached block, try to allocate a new one
            block = alloc_block(size, pool);

            if (!block) {
                // First allocation attempt failed, try to release cached blocks and retry
                stats_.num_alloc_retries++;

                // Release cached blocks
                release_blocks(small_blocks_);
                release_blocks(large_blocks_);

                // Retry allocation
                block = alloc_block(size, pool);

                if (!block) {
                    // Still failed, OOM
                    throw std::bad_alloc();
                }
            }
        }

        // Split the block if needed (only for pool-allocated blocks)
        if (should_split(block, size)) {
            // Create a new block for the remainder
            Block* remaining = new Block(
                static_cast<char*>(block->ptr) + size,
                block->size - size,
                &pool
            );

            // Link the blocks
            remaining->prev = block;
            remaining->next = block->next;
            if (block->next) {
                block->next->prev = remaining;
            }
            block->next = remaining;
            block->size = size;

            // Add remainder to pool
            pool.insert(remaining);

            // Update statistics for inactive split
            stats_.inactive_split_bytes.increase(remaining->size);
        }
    }

    // Mark as allocated
    block->allocated = true;
    block->requested_size = orig_size;

    // Track the block
    active_blocks_.insert(block);
    ptr_to_block_[block->ptr] = block;

    // Update statistics
    stats_.allocation.increase(1);
    stats_.allocated_bytes.increase(block->size);
    stats_.active_bytes.increase(block->size);

    try {
        // Register the caller-visible Storage interval, not the rounded block or
        // its containing HostDDR segment.  This is the DMA correctness boundary
        // and the identity Graph replay uses to reject same-address reuse.
        block->allocation_id =
            rpu_register_ddr_dma_logical_allocation(block->ptr, orig_size);
    } catch (...) {
        // Restore allocator metadata before propagating an invariant/registry
        // failure.  The recursive mutex makes this rollback use the one normal
        // free path without exposing the block to a caller.
        free(block->ptr);
        throw;
    }

    // Final alignment check - this should never fail if our logic is correct
    assert(reinterpret_cast<uintptr_t>(block->ptr) % kAlignment == 0 &&
           "Allocated block is not 32-byte aligned!");

    return block->ptr;
}

void RPUCachingAllocator::free(void* ptr) {
    if (!ptr) {
        return;
    }

    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Shutdown 后 (rpu_shutdown 调过 mark_shutdown), 底层 DDR 已经被
    // RpuDdrShutdown 全清, 这里再 free 已没有意义。直接返回, 避免 Python
    // 解释器析构 tensor 时触发的 "Attempting to free unknown pointer" 噪音。
    if (shutdown_called_) {
        return;
    }

    // Find the block
    auto it = ptr_to_block_.find(ptr);
    if (it == ptr_to_block_.end()) {
        if (log_at(2)) {
            std::cerr << "[RPUCachingAllocator] Warning: Attempting to free unknown pointer "
                      << ptr << std::endl;
        }
        return;
    }

    Block* block = it->second;

    if (!block->allocated) {
        if (log_at(2)) {
            std::cerr << "[RPUCachingAllocator] Warning: Double free detected for pointer "
                      << ptr << std::endl;
        }
        return;
    }

    if (block->allocation_id != 0 &&
        !rpu_unregister_ddr_dma_logical_allocation(
            ptr, block->allocation_id)) {
        // Quarantine the block instead of returning an allocation whose live
        // registry identity could still name this address.  Reuse would turn a
        // bookkeeping defect into silent cross-tensor DMA corruption.
        std::cerr << "[RPUCachingAllocator] Error: logical allocation identity "
                  << "mismatch while freeing ptr=" << ptr
                  << " allocation_id=" << block->allocation_id << std::endl;
        return;
    }

    // Mark as not allocated
    block->allocated = false;
    block->allocation_id = 0;
    block->requested_size = 0;

    // Remove from pointer map - block is no longer active
    ptr_to_block_.erase(it);

    // Update statistics
    stats_.allocation.decrease(1);
    stats_.allocated_bytes.decrease(block->size);
    stats_.active_bytes.decrease(block->size);

    // Try size-class fast path: if block matches a size class and is not split
    int size_class = size_to_class(block->size);
    if (size_class >= 0 && block->size == class_to_size(size_class) &&
        !block->is_split()) {
        // Fast path: O(1) return to size-class free list
        active_blocks_.erase(block);
        free_block_fast(block, size_class);
        return;
    }

    // Fallback: use pool-based free with merge logic
    free_block(block);
}

void RPUCachingAllocator::free_block(Block* block) {
    BlockPool& pool = *block->pool;

    // Try to merge with adjacent blocks
    size_t original_size = block->size;
    int64_t net_change_inactive_split = 0;

    // Try to merge with previous block
    if (block->prev && !block->prev->allocated) {
        size_t merged_size = try_merge_blocks(block, block->prev, pool);
        if (merged_size > 0) {
            net_change_inactive_split -= static_cast<int64_t>(merged_size);
        }
    }

    // Try to merge with next block
    if (block->next && !block->next->allocated) {
        size_t merged_size = try_merge_blocks(block, block->next, pool);
        if (merged_size > 0) {
            net_change_inactive_split -= static_cast<int64_t>(merged_size);
        }
    }

    // Remove from active blocks
    active_blocks_.erase(block);

    // Add to pool
    pool.insert(block);

    // Update inactive split statistics
    if (block->is_split()) {
        net_change_inactive_split += static_cast<int64_t>(block->size);
    }

    if (net_change_inactive_split > 0) {
        stats_.inactive_split_bytes.increase(static_cast<size_t>(net_change_inactive_split));
    } else if (net_change_inactive_split < 0) {
        stats_.inactive_split_bytes.decrease(static_cast<size_t>(-net_change_inactive_split));
    }
}

size_t RPUCachingAllocator::try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    // Can't merge if src is null or allocated
    if (!src || src->allocated) {
        return 0;
    }

    // Remove src from pool
    size_t erased = pool.erase(src);
    if (erased == 0) {
        // src was not in the pool (shouldn't happen for free blocks)
        return 0;
    }

    size_t merged_size = src->size;

    if (dst->prev == src) {
        // [src][dst] -> merge src into dst
        dst->ptr = src->ptr;
        dst->prev = src->prev;
        if (dst->prev) {
            dst->prev->next = dst;
        }
    } else if (dst->next == src) {
        // [dst][src] -> merge src into dst
        dst->next = src->next;
        if (dst->next) {
            dst->next->prev = dst;
        }
    } else {
        // Not adjacent, shouldn't happen
        pool.insert(src);  // Put it back
        return 0;
    }

    dst->size += merged_size;

    // Note: ptr_to_block_ entries are removed when blocks are freed,
    // so we don't need to update it here during merge

    delete src;

    return merged_size;
}

void RPUCachingAllocator::release_block(Block* block) {
    // Sanity check: only release blocks that are not split
    // This ensures we're releasing the original allocated pointer
    if (block->is_split()) {
        if (log_at(1)) {
            std::cerr << "[RPUCachingAllocator] Error: Attempting to release split block "
                      << "ptr=" << block->ptr << " size=" << block->size
                      << " prev=" << block->prev << " next=" << block->next
                      << std::endl;
        }
        return;
    }

    // Release memory back to system
    rpu_unregister_ddr_dma_segment(block->ptr);
    ::rhino_lkn::RpuDdrFree(block->ptr);

    // Update statistics
    total_allocated_memory_ -= block->size;
    stats_.reserved_bytes.decrease(block->size);
    stats_.segment.decrease(1);
    stats_.num_device_free++;

    // Remove from pool
    block->pool->erase(block);

    // Note: ptr_to_block_ entries are only for active blocks,
    // freed blocks are not in the map

    delete block;
}

void RPUCachingAllocator::release_blocks(BlockPool& pool) {
    // Release all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
        Block* block = *it;
        ++it;

        // Only release blocks that are not split
        if (!block->is_split()) {
            release_block(block);
        }
    }
}

void RPUCachingAllocator::emptyCache() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    // Release all blocks in size-class lists
    for (size_t i = 0; i < kNumSizeClasses; i++) {
        Block* block = size_class_lists_[i];
        while (block) {
            Block* next = block->next;
            // Release memory back to system
            rpu_unregister_ddr_dma_segment(block->ptr);
            ::rhino_lkn::RpuDdrFree(block->ptr);
            total_allocated_memory_ -= block->size;
            stats_.reserved_bytes.decrease(block->size);
            stats_.segment.decrease(1);
            stats_.num_device_free++;
            delete block;
            block = next;
        }
        size_class_lists_[i] = nullptr;
        size_class_counts_[i] = 0;
    }

    // Release all cached blocks from pools
    release_blocks(small_blocks_);
    release_blocks(large_blocks_);
}

void RPUCachingAllocator::mark_shutdown() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    shutdown_called_ = true;
}

DeviceStats RPUCachingAllocator::getStats() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return stats_;
}

void RPUCachingAllocator::resetPeakStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    stats_.allocation.reset_peak();
    stats_.reserved_bytes.reset_peak();
    stats_.allocated_bytes.reset_peak();
    stats_.active_bytes.reset_peak();
    stats_.segment.reset_peak();
    stats_.inactive_split_bytes.reset_peak();
}

void RPUCachingAllocator::resetAccumulatedStats() {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    stats_.allocation.reset_accumulated();
    stats_.reserved_bytes.reset_accumulated();
    stats_.allocated_bytes.reset_accumulated();
    stats_.active_bytes.reset_accumulated();
    stats_.segment.reset_accumulated();
    stats_.inactive_split_bytes.reset_accumulated();

    stats_.num_alloc_retries = 0;
    stats_.num_ooms = 0;
}

size_t RPUCachingAllocator::getLargestAvailableBlock() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    size_t largest = 0;

    // Check size-class lists (iterate from largest to smallest)
    for (int i = kNumSizeClasses - 1; i >= 0; i--) {
        if (size_class_lists_[i] != nullptr) {
            largest = std::max(largest, kSizeClasses[i]);
            break;  // Found the largest available size class
        }
    }

    // Check small blocks
    if (!small_blocks_.blocks.empty()) {
        auto it = small_blocks_.blocks.rbegin();
        largest = std::max(largest, (*it)->size);
    }

    // Check large blocks
    if (!large_blocks_.blocks.empty()) {
        auto it = large_blocks_.blocks.rbegin();
        largest = std::max(largest, (*it)->size);
    }

    return largest;
}

size_t RPUCachingAllocator::getTotalAllocatedMemory() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    return total_allocated_memory_;
}

size_t RPUCachingAllocator::getTotalCachedMemory() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    size_t cached = 0;

    // Count size-class cached memory
    for (size_t i = 0; i < kNumSizeClasses; i++) {
        cached += size_class_counts_[i] * kSizeClasses[i];
    }

    // Count pool-based cached memory
    for (const auto* block : small_blocks_.blocks) {
        cached += block->size;
    }

    for (const auto* block : large_blocks_.blocks) {
        cached += block->size;
    }

    return cached;
}

int64_t RPUCachingAllocator::getCachedSegmentCount() const {
    std::lock_guard<std::recursive_mutex> lock(mutex_);

    int64_t count = 0;
    for (size_t i = 0; i < kNumSizeClasses; ++i) {
        count += static_cast<int64_t>(size_class_counts_[i]);
    }
    for (const auto* block : small_blocks_.blocks) {
        count += !block->is_split();
    }
    for (const auto* block : large_blocks_.blocks) {
        count += !block->is_split();
    }
    return count;
}

RPUCachingAllocator& RPUCachingAllocator::get() {
    static RPUCachingAllocator instance;
    return instance;
}

// ============================================================================
// RPUCachingAllocatorAdapter Implementation
// ============================================================================

at::DataPtr RPUCachingAllocatorAdapter::allocate(size_t nbytes) {
    // Handle empty tensor allocation
    if (nbytes == 0) {
        return {nullptr, nullptr, nullptr, at::Device(c10::DeviceType::PrivateUse1, 0)};
    }

    void* ptr = RPUCachingAllocator::get().malloc(nbytes);

    if (!ptr) {
        throw std::bad_alloc();
    }

    // Create custom deleter that uses our caching allocator
    auto deleter = [](void* p) {
        RPUCachingAllocator::get().free(p);
    };

    return {ptr, ptr, deleter, at::Device(c10::DeviceType::PrivateUse1, 0)};
}

void RPUCachingAllocatorAdapter::copy_data(void* dest, const void* src, std::size_t count) const {
    memcpy(dest, src, count);
}

RPUCachingAllocatorAdapter& RPUCachingAllocatorAdapter::get() {
    static RPUCachingAllocatorAdapter instance;
    return instance;
}

// ============================================================================
// Global Functions
// ============================================================================

RPUCachingAllocator& get_caching_allocator() {
    return RPUCachingAllocator::get();
}

void empty_cache() {
    RPUCachingAllocator::get().emptyCache();
}

DeviceStats get_memory_stats() {
    return RPUCachingAllocator::get().getStats();
}

void reset_peak_memory_stats() {
    RPUCachingAllocator::get().resetPeakStats();
}

void reset_accumulated_memory_stats() {
    RPUCachingAllocator::get().resetAccumulatedStats();
}

} // namespace rpu
