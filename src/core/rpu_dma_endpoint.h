#pragma once

#include <cstddef>
#include <cstdint>

#include "rhino_launch_buffer.h"

// A production DMA endpoint is always one Launch-managed allocation plus a
// byte offset inside that allocation.  Device addresses remain useful as
// graph topology/diagnostic values, but must be resolved before they reach a
// Release Queue_t or standalone CopyMemory call.
struct RpuDmaEndpoint {
    ::rhino_lkn::Buffer_t* owner = nullptr;
    size_t offset = 0;
    // Zero identifies SPM and other non-DDR endpoints.  Every live PyTorch DDR
    // storage receives a process-monotonic nonzero identity, independent of
    // its physical HostDDR segment.  Graph replay uses this witness to reject
    // allocator ABA (a freed logical block reused at the same device address).
    uint64_t allocation_id = 0;

    explicit operator bool() const { return owner != nullptr; }
};

struct RpuDeviceDmaEndpointRequest;
struct RpuCpuDmaEndpointRequest;
struct RpuDmaAllocationWitness;

// A synchronous Queue submission may outlive the registry lookup that resolved
// its raw Launch Buffer_t endpoints.  This move-only lease keeps every live DDR
// logical allocation registered until the matching Queue_t submission has
// completed.  It deliberately does not retain SPM roots, whose lifetime is
// already covered by the process Graph/cleanup ownership contract.
class RpuDmaSubmissionLease final {
public:
    RpuDmaSubmissionLease() noexcept = default;
    ~RpuDmaSubmissionLease();

    RpuDmaSubmissionLease(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease& operator=(const RpuDmaSubmissionLease&) = delete;
    RpuDmaSubmissionLease(RpuDmaSubmissionLease&& other) noexcept;
    RpuDmaSubmissionLease& operator=(RpuDmaSubmissionLease&& other) noexcept;

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

// Resolve a set of graph device ranges as one allocator-registry transaction.
// The implementation validates every request before consulting SPM or DDR,
// holds the managed-DDR registry mutex at most once, and publishes no partial
// results if a later request is stale or outside a live allocation.
struct RpuDeviceDmaEndpointRequest {
    uint64_t device_addr = 0;
    size_t bytes = 0;
    const char* where = nullptr;
};

RpuDmaSubmissionLease rpu_resolve_device_dma_endpoints(
    const RpuDeviceDmaEndpointRequest* requests,
    size_t count,
    RpuDmaEndpoint* endpoints);

// Resolve CPU pointers as one registry transaction and retain their Launch
// owners across a synchronous standalone CopyMemory call.
struct RpuCpuDmaEndpointRequest {
    const void* cpu_ptr = nullptr;
    size_t bytes = 0;
    const char* where = nullptr;
};

RpuDmaSubmissionLease rpu_resolve_cpu_dma_endpoints(
    const RpuCpuDmaEndpointRequest* requests,
    size_t count,
    RpuDmaEndpoint* endpoints);

// A prepared Graph records the identity as well as the numeric address of a
// logical DDR allocation.  Validate a set of those witnesses under one
// registry lock before mutating or submitting a retained Queue_t.  Identity
// zero denotes an SPM or other non-DDR endpoint and is deliberately skipped.
struct RpuDmaAllocationWitness {
    uint64_t allocation_id = 0;
    uint64_t expected_device_addr = 0;
    size_t bytes = 0;
    const char* where = nullptr;
};

RpuDmaSubmissionLease rpu_validate_live_dma_allocation_witnesses(
    const RpuDmaAllocationWitness* witnesses,
    size_t count);

// Resolve a live CPU pointer (the preferred DDR path) or a legacy graph device
// address to a managed owner.  The device-address resolver consults only the
// parent backend's live DDR allocation registry and SPM_ALLOC's existing root
// buffers; it never constructs borrowed/shadow Buffer_t objects.
RpuDmaEndpoint rpu_resolve_cpu_dma_endpoint(
    const void* cpu_ptr, size_t bytes, const char* where);
RpuDmaEndpoint rpu_resolve_device_dma_endpoint(
    uint64_t device_addr, size_t bytes, const char* where);

// Keep the reverse device-address index complete for both the direct PyTorch
// allocator and the optional caching allocator.  The legacy pair below is the
// direct-allocation transaction: register/unregister both the physical segment
// and its one logical PyTorch storage.
void rpu_register_ddr_dma_allocation(void* cpu_base, size_t bytes);
void rpu_unregister_ddr_dma_allocation(void* cpu_base);
void rpu_free_registered_ddr(void* cpu_base) noexcept;

// A caching allocation has one physical HostDDR segment containing one or more
// independently-live logical PyTorch storages.  Segment registration owns only
// the Launch reverse mapping; DMA resolution is admitted exclusively through
// the logical registration and its requested storage byte range.
void rpu_register_ddr_dma_segment(void* cpu_base, size_t bytes);
void rpu_unregister_ddr_dma_segment(void* cpu_base);
uint64_t rpu_register_ddr_dma_logical_allocation(
    void* cpu_base, size_t bytes);
bool rpu_unregister_ddr_dma_logical_allocation(
    void* cpu_base, uint64_t allocation_id) noexcept;
