// Owner and Graph identity shared by the checked decode body replays. Include
// after the model header, which brings ATen, the Graph runtime and RpuGetDevAddr.
#pragma once

#include <cstdint>
#include <vector>

namespace v3 {
namespace decode_replay {

struct DecodeOwner {
    at::Tensor tensor; // retains storage even if the caller replaces its cache
    c10::TensorImpl* impl = nullptr;
    c10::StorageImpl* storage = nullptr;
    at::ScalarType dtype = at::ScalarType::Undefined;
    c10::Device device{c10::DeviceType::CPU};
    std::vector<int64_t> shape, strides;
    int64_t offset = 0;
    uint32_t version = 0;
    uint64_t address = 0;
    size_t storage_bytes = 0;
    bool immutable = false;

    bool freeze(const at::Tensor& t, bool fixed) {
        if (!t.defined() || t.device().type() != at::kPrivateUse1 ||
            t.layout() != c10::Layout::Strided || !t.is_contiguous()) return false;
        const auto& counter = t.unsafeGetTensorImpl()->version_counter();
        // Versionless inference weights are still supported by the ordinary
        // body; they cannot prove an immutable replay binding.
        if (fixed && !counter.enabled()) return false;
        tensor = t; impl = t.unsafeGetTensorImpl();
        storage = t.storage().unsafeGetStorageImpl();
        dtype = t.scalar_type(); device = t.device();
        shape = t.sizes().vec(); strides = t.strides().vec();
        offset = t.storage_offset(); storage_bytes = t.storage().nbytes();
        address = RpuGetDevAddr(t.data_ptr()); immutable = fixed;
        version = fixed ? counter.current_version() : 0;
        return address && address < (1ULL << 40) && address % 256 == 0;
    }

    bool matches(const at::Tensor& t) const {
        if (!t.defined() || t.unsafeGetTensorImpl() != impl ||
            t.layout() != c10::Layout::Strided || t.scalar_type() != dtype ||
            t.device() != device || t.storage().unsafeGetStorageImpl() != storage ||
            t.storage().nbytes() != storage_bytes || t.storage_offset() != offset ||
            !t.sizes().equals(shape) || !t.strides().equals(strides) ||
            RpuGetDevAddr(t.data_ptr()) != address) return false;
        const auto& counter = t.unsafeGetTensorImpl()->version_counter();
        return !immutable || (counter.enabled() && counter.current_version() == version);
    }
};

struct DecodeGraphKey {
    const RpuKernelGraph* graph = nullptr;
    uint64_t lifetime = 0, generation = 0, signature = 0, segment = 0;
    bool operator==(const DecodeGraphKey& other) const {
        return graph == other.graph && lifetime == other.lifetime &&
            generation == other.generation && signature == other.signature &&
            segment == other.segment;
    }
};

inline DecodeGraphKey graph_key(RpuKernelGraph& graph, const GraphOpStreamStamp& stamp) {
    return {&graph, graph.debug_stats().graph_lifetime_id, stamp.build_generation,
            stamp.signature_identity, stamp.signature_segment_key};
}

inline bool tensor_shape(const at::Tensor& t, at::ScalarType dtype, at::IntArrayRef shape) {
    return t.defined() && t.scalar_type() == dtype && t.sizes().equals(shape);
}

} // namespace decode_replay
} // namespace v3
