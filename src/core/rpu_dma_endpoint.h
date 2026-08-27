#pragma once

#include <cstddef>
#include <cstdint>

#include "rhino_launch_buffer.h"

// A DMA endpoint is a Launch-managed allocation plus a byte offset inside it.
// Raw addresses are resolved before reaching a Release queue or copy call.
struct RpuDmaEndpoint {
    ::rhino_lkn::Buffer_t* owner = nullptr;
    size_t offset = 0;

    explicit operator bool() const { return owner != nullptr; }
};

RpuDmaEndpoint rpu_resolve_cpu_dma_endpoint(
    const void* cpu_ptr, size_t bytes, const char* where);
RpuDmaEndpoint rpu_resolve_device_dma_endpoint(
    uint64_t device_addr, size_t bytes, const char* where);

void rpu_register_ddr_dma_allocation(void* cpu_base, size_t bytes);
void rpu_unregister_ddr_dma_allocation(void* cpu_base);
void rpu_free_registered_ddr(void* cpu_base);
