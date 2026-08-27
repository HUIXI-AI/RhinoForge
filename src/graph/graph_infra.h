// graph_infra.h — shared graph infrastructure types.
//
// This header contains the low-level building blocks used by graph capture
// and replay but does not execute graphs by itself:
//   - GraphSignature and layered signature diagnostics
//   - RpuGraphCache and default cache entry ownership
//   - graph-owned resource descriptors and SPM buffer pools
//   - HostCallback tier classification declarations
//
// RpuKernelGraph runtime code lives in graph_runtime.h/cpp and depends on this
// file; infra deliberately only forward-declares RpuKernelGraph to keep the
// dependency direction one-way.
#pragma once

#include "rhino_launch_buffer.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/stack.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

class RpuKernelGraph;

// =============================================================================
// GraphSignature — multi-graph cache lookup key
// =============================================================================

struct GraphSignature {
    // Stable id for the op, model phase, or layer group. Python callers usually
    // FNV1a-hash strings to int64_t before crossing the pybind boundary.
    int64_t op_id = 0;

    // Human-readable diagnostic label mirrored from Python
    // (`rpu_backend.GraphSignature(op_id="pi05_siglip_compute")`). NOT part of
    // operator==/Hash/empty() — only used for log lines + diagnostic strings.
    // Empty is allowed and means "no label" (older callers / no-backend mode).
    std::string op_id_str;

    // Tensor shape and dtype payload. Callers own the ordering convention.
    std::vector<int64_t> shapes;
    std::vector<int64_t> dyn_dims;
    std::vector<uint8_t> dtypes;

    // Small feature flags such as causal / SPM mode / flush mode.
    uint64_t flags = 0;

    // Optional control-flow and segment discriminators. Zero preserves the
    // pre-branch/pre-segment default signature behavior.
    uint64_t branch_key = 0;
    uint64_t segment_key = 0;

    bool operator==(const GraphSignature& other) const {
        return op_id == other.op_id &&
               flags == other.flags &&
               branch_key == other.branch_key &&
               segment_key == other.segment_key &&
               shapes == other.shapes &&
               dyn_dims == other.dyn_dims &&
               dtypes == other.dtypes;
    }
    bool operator!=(const GraphSignature& other) const {
        return !(*this == other);
    }

    bool empty() const {
        return op_id == 0 && flags == 0 &&
               branch_key == 0 && segment_key == 0 &&
               shapes.empty() && dyn_dims.empty() && dtypes.empty();
    }

    std::string to_string() const {
        std::string s = "GraphSignature(op_id=" + std::to_string(op_id);
        s += ",flags=0x";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%llx",
                      static_cast<unsigned long long>(flags));
        s += buf;
        s += ",branch=0x";
        std::snprintf(buf, sizeof(buf), "%llx",
                      static_cast<unsigned long long>(branch_key));
        s += buf;
        s += ",segment=0x";
        std::snprintf(buf, sizeof(buf), "%llx",
                      static_cast<unsigned long long>(segment_key));
        s += buf;
        s += ",shapes=[";
        for (size_t i = 0; i < shapes.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(shapes[i]);
        }
        s += "],dyn_dims=[";
        for (size_t i = 0; i < dyn_dims.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(dyn_dims[i]);
        }
        s += "],dtypes=[";
        for (size_t i = 0; i < dtypes.size(); ++i) {
            if (i) s += ",";
            s += std::to_string(static_cast<int>(dtypes[i]));
        }
        s += "])";
        return s;
    }
};

struct GraphSignatureHash {
    size_t operator()(const GraphSignature& sig) const noexcept {
        auto mix = [](size_t seed, size_t v) {
            return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
        };
        size_t h = std::hash<int64_t>{}(sig.op_id);
        h = mix(h, std::hash<uint64_t>{}(sig.flags));
        h = mix(h, std::hash<uint64_t>{}(sig.branch_key));
        h = mix(h, std::hash<uint64_t>{}(sig.segment_key));
        for (auto v : sig.shapes)   h = mix(h, std::hash<int64_t>{}(v));
        for (auto v : sig.dyn_dims) h = mix(h, std::hash<int64_t>{}(v));
        for (auto v : sig.dtypes)   h = mix(h, std::hash<uint8_t>{}(v));
        return h;
    }
};

// =============================================================================
// LayeredSignature — diagnostic view of which signature layer diverged
// =============================================================================

enum class SignatureLayer : uint8_t {
    GmId = 0,
    TensorStruct,
    ShapeBucket,
    DtypeLayout,
    ScalarFlags,
    BranchKey,
    SegmentKey,
    kCount,
};

inline const char* signature_layer_name(SignatureLayer layer) {
    switch (layer) {
    case SignatureLayer::GmId:         return "GmId";
    case SignatureLayer::TensorStruct: return "TensorStruct";
    case SignatureLayer::ShapeBucket:  return "ShapeBucket";
    case SignatureLayer::DtypeLayout:  return "DtypeLayout";
    case SignatureLayer::ScalarFlags:  return "ScalarFlags";
    case SignatureLayer::BranchKey:    return "BranchKey";
    case SignatureLayer::SegmentKey:   return "SegmentKey";
    case SignatureLayer::kCount:       return "kCount";
    }
    return "?";
}

struct SignatureLayerHash {
    SignatureLayer layer;
    uint64_t hash = 0;
};

struct LayeredSignature {
    std::array<SignatureLayerHash,
               static_cast<size_t>(SignatureLayer::kCount)> layers;

    LayeredSignature();

    uint64_t layer_hash(SignatureLayer layer) const;
    bool equals_up_to(const LayeredSignature& other,
                      SignatureLayer up_to_inclusive) const;
    SignatureLayer first_diverge(const LayeredSignature& other) const;
    std::string to_string() const;

    LayeredSignature with_layer_hash(SignatureLayer layer, uint64_t hash) const;
};

namespace graph_signature_tree_detail {
inline constexpr size_t layer_index(SignatureLayer layer) {
    return static_cast<size_t>(layer);
}

inline uint64_t mix_hash(uint64_t seed, uint64_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

template <typename T>
inline uint64_t hash_value(const T& value) {
    return static_cast<uint64_t>(std::hash<T>{}(value));
}

inline std::string hex_u64(uint64_t value) {
    std::ostringstream os;
    os << "0x" << std::hex << value;
    return os.str();
}
}  // namespace graph_signature_tree_detail

inline LayeredSignature::LayeredSignature() {
    for (size_t i = 0; i < layers.size(); ++i) {
        layers[i].layer = static_cast<SignatureLayer>(i);
        layers[i].hash = 0;
    }
}

inline uint64_t LayeredSignature::layer_hash(SignatureLayer layer) const {
    const size_t idx = graph_signature_tree_detail::layer_index(layer);
    return idx < layers.size() ? layers[idx].hash : 0;
}

inline bool LayeredSignature::equals_up_to(
        const LayeredSignature& other,
        SignatureLayer up_to_inclusive) const {
    const size_t last = graph_signature_tree_detail::layer_index(up_to_inclusive);
    if (last >= layers.size()) {
        return first_diverge(other) == SignatureLayer::kCount;
    }
    for (size_t i = 0; i <= last; ++i) {
        if (layers[i].hash != other.layers[i].hash) {
            return false;
        }
    }
    return true;
}

inline SignatureLayer LayeredSignature::first_diverge(
        const LayeredSignature& other) const {
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].hash != other.layers[i].hash) {
            return static_cast<SignatureLayer>(i);
        }
    }
    return SignatureLayer::kCount;
}

inline std::string LayeredSignature::to_string() const {
    std::ostringstream os;
    for (size_t i = 0; i < layers.size(); ++i) {
        for (size_t indent = 0; indent < i; ++indent) {
            os << "  ";
        }
        os << signature_layer_name(layers[i].layer)
           << "(" << graph_signature_tree_detail::hex_u64(layers[i].hash)
           << ")";
        if (i + 1 < layers.size()) {
            os << "\n";
        }
    }
    return os.str();
}

inline LayeredSignature LayeredSignature::with_layer_hash(
        SignatureLayer layer, uint64_t hash) const {
    LayeredSignature out = *this;
    const size_t idx = graph_signature_tree_detail::layer_index(layer);
    if (idx < out.layers.size()) {
        out.layers[idx].hash = hash;
    }
    return out;
}

inline LayeredSignature layered_from(const GraphSignature& sig) {
    using namespace graph_signature_tree_detail;

    LayeredSignature out;

    out.layers[layer_index(SignatureLayer::GmId)].hash =
        hash_value(sig.op_id);

    uint64_t tensor_struct = 0;
    tensor_struct = mix_hash(tensor_struct, hash_value(sig.shapes.size()));
    tensor_struct = mix_hash(tensor_struct, hash_value(sig.dyn_dims.size()));
    tensor_struct = mix_hash(tensor_struct, hash_value(sig.dtypes.size()));
    out.layers[layer_index(SignatureLayer::TensorStruct)].hash = tensor_struct;

    uint64_t shape_bucket = 0;
    for (int64_t v : sig.shapes) {
        shape_bucket = mix_hash(shape_bucket, hash_value(v));
    }
    for (int64_t v : sig.dyn_dims) {
        shape_bucket = mix_hash(shape_bucket, hash_value(v));
    }
    out.layers[layer_index(SignatureLayer::ShapeBucket)].hash = shape_bucket;

    uint64_t dtype_layout = 0;
    for (uint8_t v : sig.dtypes) {
        dtype_layout = mix_hash(dtype_layout, hash_value(v));
    }
    out.layers[layer_index(SignatureLayer::DtypeLayout)].hash = dtype_layout;

    out.layers[layer_index(SignatureLayer::ScalarFlags)].hash =
        hash_value(sig.flags);

    out.layers[layer_index(SignatureLayer::BranchKey)].hash = sig.branch_key;
    out.layers[layer_index(SignatureLayer::SegmentKey)].hash = sig.segment_key;

    return out;
}

// =============================================================================
// GraphOwnedResource — owner-tree metadata for graph-owned allocations
// =============================================================================

enum class ResourceKind : uint8_t {
    StableOutput,
    HostTemp,
    SpmSlot,
    ZeroCopyStorageHolder,
    ChildGraphRef,
    TransientTier3Buffer,
};

enum class LifetimeState : uint8_t {
    Alive,
    Released,
};

constexpr int kInvalidNodeId = -1;
constexpr int32_t kInvalidPoolSlotId = -1;

struct GraphOwnedResource {
    ResourceKind kind = ResourceKind::HostTemp;
    int owner_node_id = kInvalidNodeId;
    void* ptr = nullptr;
    uint64_t dev_addr = 0;
    size_t nbytes = 0;
    LifetimeState state = LifetimeState::Alive;
    uint64_t allocator_block_id = 0;
    int32_t pool_slot_id = kInvalidPoolSlotId;
    int last_write_node = kInvalidNodeId;
    int last_read_node = kInvalidNodeId;

    GraphOwnedResource() = default;

    GraphOwnedResource(ResourceKind k, int owner, void* host_ptr,
                       uint64_t dev, size_t bytes)
        : kind(k), owner_node_id(owner), ptr(host_ptr), dev_addr(dev),
          nbytes(bytes) {}

    bool is_alive() const { return state == LifetimeState::Alive; }
    void mark_released() { state = LifetimeState::Released; }

    bool covers_host_range(const void* base, size_t len) const {
        if (!ptr || !base) return false;
        const auto* lo = static_cast<const uint8_t*>(ptr);
        const auto* hi = lo + nbytes;
        const auto* qlo = static_cast<const uint8_t*>(base);
        const auto* qhi = qlo + len;
        return qlo >= lo && qhi <= hi && qhi >= qlo;
    }

    bool covers_dev_range(uint64_t base, size_t len) const {
        if (dev_addr == 0) return false;
        return base >= dev_addr && (base + len) <= (dev_addr + nbytes)
               && (base + len) >= base;
    }

    bool contains_host_addr(uintptr_t addr) const {
        if (!ptr) return false;
        const uintptr_t lo = reinterpret_cast<uintptr_t>(ptr);
        return addr >= lo && addr < lo + nbytes;
    }

    bool contains_dev_addr(uint64_t addr) const {
        if (dev_addr == 0) return false;
        return addr >= dev_addr && addr < dev_addr + nbytes;
    }
};

inline const char* resource_kind_name(ResourceKind k) {
    switch (k) {
    case ResourceKind::StableOutput:           return "StableOutput";
    case ResourceKind::HostTemp:               return "HostTemp";
    case ResourceKind::SpmSlot:                return "SpmSlot";
    case ResourceKind::ZeroCopyStorageHolder:  return "ZeroCopyStorageHolder";
    case ResourceKind::ChildGraphRef:          return "ChildGraphRef";
    case ResourceKind::TransientTier3Buffer:   return "TransientTier3Buffer";
    }
    return "Unknown";
}

inline const char* lifetime_state_name(LifetimeState s) {
    switch (s) {
    case LifetimeState::Alive:    return "Alive";
    case LifetimeState::Released: return "Released";
    }
    return "Unknown";
}

// =============================================================================
// GraphBufferPool — graph-managed SPM/DDR slots with stable replay addresses
// =============================================================================

template <typename BufferType>
class GraphBufferPool {
public:
    struct Slot {
        std::unique_ptr<BufferType> buffer;
        uint32_t slot_id;
    };

    GraphBufferPool() = default;
    virtual ~GraphBufferPool() = default;

    GraphBufferPool(const GraphBufferPool&) = delete;
    GraphBufferPool& operator=(const GraphBufferPool&) = delete;

    void reset_cursor() { cursor_ = 0; }

    void clear() {
        slots_.clear();
        ptr_to_slot_.clear();
        cursor_ = 0;
    }

    size_t size() const { return slots_.size(); }
    size_t cursor() const { return cursor_; }

    int32_t find_slot(const void* cpu_ptr) const {
        if (cpu_ptr == nullptr) return -1;
        auto it = ptr_to_slot_.find(const_cast<void*>(cpu_ptr));
        return it == ptr_to_slot_.end() ? -1 : it->second;
    }

    Slot& slot(int32_t slot_id) {
        assert(slot_id >= 0 && static_cast<size_t>(slot_id) < slots_.size() &&
               "GraphBufferPool: slot_id out of range");
        return slots_[slot_id];
    }

protected:
    Slot& create_slot(std::unique_ptr<BufferType> buf) {
        int32_t id = static_cast<int32_t>(slots_.size());
        void* cpu_ptr = buf->get_cpu_ptr();
        slots_.push_back({std::move(buf), static_cast<uint32_t>(id)});
        if (cpu_ptr != nullptr) {
            ptr_to_slot_[cpu_ptr] = id;
        }
        cursor_ = slots_.size();
        return slots_.back();
    }

    Slot& next_slot() {
        assert(cursor_ < slots_.size() &&
               "GraphBufferPool: replay cursor overflow");
        return slots_[cursor_++];
    }

    std::vector<Slot> slots_;
    std::unordered_map<void*, int32_t> ptr_to_slot_;
    size_t cursor_ = 0;
};

class GraphLocalSpmPool : public GraphBufferPool<rhino_lkn::LocalSPM_t> {
public:
    Slot& acquire_recording(size_t bytes, rhino_lkn::MemFlag flag,
                            rhino_lkn::MemStride stride, uint64_t alignment,
                            int core_id) {
        auto buf = std::make_unique<rhino_lkn::LocalSPM_t>(
            bytes, flag, stride, alignment, core_id);
        return create_slot(std::move(buf));
    }

    Slot& acquire_replaying() { return next_slot(); }
};

class GraphGlobalSpmPool : public GraphBufferPool<rhino_lkn::GlobalSPM_t> {
public:
    Slot& acquire_recording(size_t bytes, rhino_lkn::MemFlag flag,
                            rhino_lkn::MemStride stride, uint64_t alignment,
                            int core_id) {
        auto buf = std::make_unique<rhino_lkn::GlobalSPM_t>(
            bytes, flag, stride, alignment, core_id);
        return create_slot(std::move(buf));
    }

    Slot& acquire_replaying() { return next_slot(); }
};

class GraphHostDdrPool : public GraphBufferPool<rhino_lkn::HostDDR_t> {
public:
    Slot& acquire_recording(size_t bytes, rhino_lkn::MemFlag flag,
                            rhino_lkn::MemStride stride, uint64_t alignment) {
        auto buf = std::make_unique<rhino_lkn::HostDDR_t>(
            bytes, flag, stride, alignment);
        return create_slot(std::move(buf));
    }

    Slot& acquire_replaying() { return next_slot(); }
};

// =============================================================================
// HostCallback tiering and zero-copy CPU fallback declarations
// =============================================================================

enum class HostCallbackTier : uint8_t {
    Tier1 = 1,
    Tier2 = 2,
    Tier3 = 3,
    Reject = 0,
};

HostCallbackTier classify_host_op(const c10::OperatorHandle& op);
bool is_data_dependent_shape(const c10::OperatorName& name);
bool is_rng_op(const c10::OperatorName& name);

inline bool is_mutable_output_arg(const c10::FunctionSchema& schema,
                                  size_t arg_idx) {
    if (arg_idx >= schema.arguments().size()) return false;
    const auto& arg = schema.arguments()[arg_idx];
    const c10::AliasInfo* alias = arg.alias_info();
    return alias != nullptr && alias->isWrite();
}

inline bool is_inplace_self_mutation(const c10::FunctionSchema& schema,
                                     size_t arg_idx) {
    if (!is_mutable_output_arg(schema, arg_idx)) return false;
    return schema.arguments()[arg_idx].name() == "self";
}

inline bool all_returns_are_tensors(const c10::FunctionSchema& schema) {
    for (const auto& ret : schema.returns()) {
        if (!ret.type()->isSubtypeOf(*c10::TensorType::get())) {
            return false;
        }
    }
    return !schema.returns().empty();
}

std::vector<size_t> collect_output_arg_indices(const c10::FunctionSchema& schema);

void run_cpu_fallback_zerocopy(const c10::OperatorHandle& op,
                               torch::jit::Stack* stack,
                               bool alias_output_args);

// =============================================================================
// RpuGraphCache — signature to BUILT RpuKernelGraph container
// =============================================================================

struct RpuGraphCacheEntry {
    std::shared_ptr<RpuKernelGraph> graph;
    GraphSignature signature;

    size_t built_kernel_count = 0;
    size_t built_segment_count = 0;
    size_t built_local_spm_slot_count = 0;
    bool built_once = false;
    size_t replay_count = 0;
    size_t recapture_count = 0;
};

class RpuGraphCache {
public:
    RpuGraphCache() = default;
    explicit RpuGraphCache(size_t max_entries) : max_entries_(max_entries) {}

    RpuKernelGraph& get_or_create(const GraphSignature& sig);
    RpuKernelGraph* lookup(const GraphSignature& sig);
    std::shared_ptr<RpuKernelGraph> lookup_shared(const GraphSignature& sig);

    void evict(const GraphSignature& sig);
    void clear();

    size_t size() const { return entries_.size(); }
    size_t max_entries() const { return max_entries_; }

    void touch_entry(const GraphSignature& sig, bool replay);

    struct Snapshot {
        GraphSignature signature;
        size_t kernel_count;
        size_t segment_count;
        size_t local_spm_slot_count;
        size_t replay_count;
        size_t recapture_count;
        std::string non_replayable_reason;
        size_t data_node_count = 0;
        size_t boundary_flush_count = 0;
        size_t prepared_segment_hit_total = 0;
        size_t prepared_segment_miss_total = 0;
        size_t hw_batch_submit_total = 0;
        size_t host_callback_tier1_count = 0;
        size_t host_callback_tier2_count = 0;
        size_t host_callback_tier3_count = 0;
        size_t tier3_oneshot_count = 0;
        bool register_census_present = false;
        bool register_census_build_committed = false;
        uint64_t register_census_policy_fingerprint = 0;
        uint64_t register_census_policy_digest = 0;
        uint64_t register_census_plan_hash = 0;
        uint64_t register_census_live_epoch = 0;
        size_t register_census_node_count = 0;
        size_t register_census_dma_count = 0;
        size_t register_census_barrier_count = 0;
        size_t register_census_kernel_count = 0;
        size_t register_census_typed_kernel_count = 0;
        size_t register_census_no_ddr_kernel_count = 0;
        size_t register_census_operand_count = 0;
        size_t register_census_weight_operand_count = 0;
        size_t register_census_key_cache_operand_count = 0;
        size_t register_census_value_cache_operand_count = 0;
        size_t register_census_semantic_input_operand_count = 0;
        uint64_t register_census_ordered_node_kind_hash = 0;
        uint64_t register_census_ordered_kernel_hash = 0;
        bool register_census_outer_fast_hit = false;
        size_t register_census_outer_fast_retained_validated = 0;
        size_t register_census_outer_fast_mutable_slot_count = 0;
        size_t register_census_outer_fast_mutable_occurrence_count = 0;
        size_t register_census_host_reemitted_node_count = 0;
        size_t register_census_host_reemitted_kernel_count = 0;
    };
    std::vector<Snapshot> snapshot() const;

    bool cache_invariant_ok() const;
    std::map<int64_t, size_t> debug_bucket_counts() const;
    std::map<int64_t, std::map<uint64_t, size_t>> debug_branch_counts() const;
    std::string dump_signature_tree() const;
    std::string explain_miss(const GraphSignature& sig) const;

private:
    void add_to_layered_index(const GraphSignature& sig);
    void remove_from_layered_index(const GraphSignature& sig);

    std::unordered_map<GraphSignature, RpuGraphCacheEntry, GraphSignatureHash> entries_;
    std::unordered_map<int64_t, std::vector<GraphSignature>> by_gm_id_;
    std::unordered_map<int64_t, std::map<uint64_t, std::vector<GraphSignature>>>
        by_gm_branch_;
    size_t max_entries_ = std::numeric_limits<size_t>::max();
};

RpuGraphCache& default_graph_cache();
