// graph_runtime.cpp — RpuKernelGraph active stack, lifecycle, flushes, and resources.
//
// This file owns the runtime state transitions (PASSTHROUGH / RECORDING /
// REPLAYING / BUILT), thread-local active graph stack, graph resource cleanup,
// and boundary flush bookkeeping. Kernel execution and replay dispatch are in
// graph_runtime_execute.cpp.
#include "graph/graph_runtime.h"

#include <ATen/record_function.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

// =============================================================================
// 静态成员初始化
// =============================================================================

// A1 — active 栈化:nested begin 允许,同一 graph 重入仍被拒。
thread_local std::vector<RpuKernelGraph*> RpuKernelGraph::active_stack_;
thread_local RpuKernelGraph RpuKernelGraph::fallback_graph_;

namespace {

struct ProcessExecutionState {
    std::mutex mutex;
    std::thread::id owner_tid;
    uint64_t generation = 0;
    size_t graph_depth = 0;
    bool physical_active = false;
    size_t cleanup_depth = 0;
    bool shutting_down = false;
    std::vector<uint64_t> graph_tokens;
    uint64_t physical_token = 0;
    std::vector<uint64_t> cleanup_tokens;
};

enum class RetainedGraphLifetimeState : uint8_t {
    Live = 1,
    Invalidated = 2,
    Evicted = 3,
};

enum class RetainedPhysicalArenaBindingState : uint8_t {
    Bound = 1,
    Parked = 2,
};

struct RetainedPhysicalArenaRecord {
    uint64_t token = 0;
    uint64_t graph_lifetime_id = 0;
    uint64_t graph_build_generation = 0;
    uint64_t graph_signature_identity = 0;
    uint64_t graph_topology_hash = 0;
    uint64_t physical_execution_token = 0;
    uint64_t lease_epoch = 0;
    uint64_t plan_hash = 0;
    uint64_t allocator_generation = 0;
    uint64_t persistent_generation = 0;
    size_t arena_base = 0;
    size_t arena_end = 0;
    size_t reserved_top_bytes = 0;
    std::weak_ptr<uint64_t> lease_generation;
    uint64_t expected_lease_generation = 0;
    std::thread::id owner_thread;
    RetainedGraphLifetimeState graph_state =
        RetainedGraphLifetimeState::Live;
    RetainedPhysicalArenaBindingState binding_state =
        RetainedPhysicalArenaBindingState::Bound;
};

struct ProvisionalPhysicalGraphBuildRecord {
    uint64_t graph_lifetime_id = 0;
    uint64_t graph_build_generation = 0;
    uint64_t graph_signature_identity = 0;
    uint64_t graph_topology_hash = 0;
    uint64_t physical_execution_token = 0;
    std::thread::id owner_thread;
};

struct RetainedPhysicalArenaRegistry {
    std::mutex mutex;
    uint64_t next_token = 0;
    std::unordered_map<uint64_t, RetainedPhysicalArenaRecord> records;
    std::unordered_map<uint64_t, uint64_t> token_by_graph_lifetime;
    std::unordered_map<uint64_t, ProvisionalPhysicalGraphBuildRecord>
        provisional_by_graph_lifetime;
};

ProcessExecutionState& process_execution_state() {
    // Process lifetime avoids static-destruction ordering against Python Graph
    // objects and the backend atexit hook.
    static auto* state = new ProcessExecutionState();
    return *state;
}

RetainedPhysicalArenaRegistry& retained_physical_arena_registry() {
    // Like the registered-Graph registry, this survives Python/static teardown.
    // An accidentally dropped live guard must remain fail-closed rather than
    // disappearing because of C++ static destruction order.
    static auto* registry = new RetainedPhysicalArenaRegistry();
    return *registry;
}

uint64_t next_graph_lifetime_id() {
    static auto* counter = new std::atomic<uint64_t>(0);
    uint64_t observed = counter->load(std::memory_order_relaxed);
    for (;;) {
        TORCH_CHECK(observed != std::numeric_limits<uint64_t>::max(),
                    "RpuKernelGraph lifetime identity exhausted");
        const uint64_t next = observed + 1;
        if (counter->compare_exchange_weak(
                observed, next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return next;
        }
    }
}

uint64_t current_owned_physical_execution_token(
        const char* operation, bool require_graph_quiescent) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "retained physical Graph operation requires a name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(!state.shutting_down && state.cleanup_depth == 0 &&
                    state.physical_active && state.physical_token != 0 &&
                    state.owner_tid == current,
                operation,
                ": retained physical Graph requires the current thread's "
                "live physical arena claim");
    TORCH_CHECK(!require_graph_quiescent || state.graph_depth == 0,
                operation,
                ": retained physical Graph operation must run outside every "
                "Graph scope");
    return state.physical_token;
}

bool has_active_retained_physical_arena_guards() {
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (!registry.provisional_by_graph_lifetime.empty()) return true;
    return std::any_of(
        registry.records.begin(), registry.records.end(),
        [](const auto& item) {
            return item.second.binding_state ==
                RetainedPhysicalArenaBindingState::Bound;
        });
}

void require_no_active_retained_physical_arena_guards(
        const char* operation) {
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const bool bound = std::any_of(
        registry.records.begin(), registry.records.end(),
        [](const auto& item) {
            return item.second.binding_state ==
                RetainedPhysicalArenaBindingState::Bound;
        });
    TORCH_CHECK(
        registry.provisional_by_graph_lifetime.empty() && !bound,
        operation,
        ": an unarmed physical Graph BUILD or retained guard is still "
        "bound; invalidate the provisional Graph, park every live retained "
        "Graph, or invalidate the armed Graph and retire its guard before "
        "releasing the physical arena");
}

void register_provisional_physical_graph_build(
        uint64_t graph_lifetime_id,
        uint64_t graph_build_generation,
        uint64_t graph_signature_identity,
        uint64_t graph_topology_hash,
        uint64_t physical_execution_token) {
    TORCH_CHECK(graph_lifetime_id != 0 && graph_build_generation != 0 &&
                    graph_signature_identity != 0 &&
                    graph_topology_hash != 0 &&
                    physical_execution_token != 0,
                "RpuKernelGraph: provisional physical BUILD identity fields "
                "must all be nonzero");
    const uint64_t live_token = current_owned_physical_execution_token(
        "RpuKernelGraph physical BUILD commit",
        /*require_graph_quiescent=*/false);
    TORCH_CHECK(live_token == physical_execution_token,
                "RpuKernelGraph: physical BUILD claim changed before commit");

    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    TORCH_CHECK(
        registry.token_by_graph_lifetime.count(graph_lifetime_id) == 0 &&
            registry.provisional_by_graph_lifetime.count(
                graph_lifetime_id) == 0,
        "RpuKernelGraph: this Graph lifetime already has physical BUILD "
        "authority");
    ProvisionalPhysicalGraphBuildRecord record;
    record.graph_lifetime_id = graph_lifetime_id;
    record.graph_build_generation = graph_build_generation;
    record.graph_signature_identity = graph_signature_identity;
    record.graph_topology_hash = graph_topology_hash;
    record.physical_execution_token = physical_execution_token;
    record.owner_thread = std::this_thread::get_id();
    registry.provisional_by_graph_lifetime.emplace(
        graph_lifetime_id, std::move(record));
}

bool graph_has_provisional_physical_build(uint64_t graph_lifetime_id) {
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    return registry.provisional_by_graph_lifetime.count(
        graph_lifetime_id) != 0;
}

uint64_t bound_physical_execution_token_for_graph(
        uint64_t graph_lifetime_id) {
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto provisional = registry.provisional_by_graph_lifetime.find(
        graph_lifetime_id);
    const auto guarded = registry.token_by_graph_lifetime.find(
        graph_lifetime_id);
    TORCH_INTERNAL_ASSERT(
        provisional == registry.provisional_by_graph_lifetime.end() ||
            guarded == registry.token_by_graph_lifetime.end());
    if (provisional != registry.provisional_by_graph_lifetime.end()) {
        return provisional->second.physical_execution_token;
    }
    if (guarded == registry.token_by_graph_lifetime.end()) return 0;
    const auto record = registry.records.find(guarded->second);
    TORCH_INTERNAL_ASSERT(record != registry.records.end());
    return record->second.binding_state ==
            RetainedPhysicalArenaBindingState::Bound
        ? record->second.physical_execution_token
        : 0;
}

uint64_t register_retained_physical_arena_guard(
        uint64_t graph_lifetime_id,
        uint64_t graph_build_generation,
        uint64_t graph_signature_identity,
        uint64_t graph_topology_hash,
        uint64_t physical_execution_token,
        uint64_t lease_epoch,
        uint64_t plan_hash,
        uint64_t allocator_generation,
        uint64_t persistent_generation,
        size_t arena_base,
        size_t arena_end,
        size_t reserved_top_bytes,
        const std::weak_ptr<uint64_t>& lease_generation,
        uint64_t expected_lease_generation) {
    const std::shared_ptr<uint64_t> live_lease_generation =
        lease_generation.lock();
    TORCH_CHECK(graph_lifetime_id != 0 && graph_build_generation != 0 &&
                    graph_signature_identity != 0 && graph_topology_hash != 0 &&
                    physical_execution_token != 0 && lease_epoch != 0 &&
                    plan_hash != 0 && allocator_generation != 0 &&
                    arena_end > arena_base &&
                    live_lease_generation != nullptr &&
                    expected_lease_generation != 0 &&
                    *live_lease_generation == expected_lease_generation,
                "RpuKernelGraph: retained physical arena identity fields must "
                "describe one live opaque lease authority");
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto provisional = registry.provisional_by_graph_lifetime.find(
        graph_lifetime_id);
    TORCH_CHECK(
        provisional != registry.provisional_by_graph_lifetime.end(),
        "RpuKernelGraph: retained physical arm requires this exact "
        "provisional physical BUILD");
    const auto& pending = provisional->second;
    TORCH_CHECK(
        pending.graph_build_generation == graph_build_generation &&
            pending.graph_signature_identity == graph_signature_identity &&
            pending.graph_topology_hash == graph_topology_hash &&
            pending.physical_execution_token == physical_execution_token &&
            pending.owner_thread == std::this_thread::get_id(),
        "RpuKernelGraph: retained physical arm is stale, foreign, or from a "
        "different physical BUILD claim");
    TORCH_CHECK(
        registry.token_by_graph_lifetime.count(graph_lifetime_id) == 0,
        "RpuKernelGraph: this Graph lifetime already has a retained physical "
        "arena guard");
    for (const auto& item : registry.records) {
        const auto& existing = item.second;
        if (existing.binding_state ==
            RetainedPhysicalArenaBindingState::Parked) {
            continue;
        }
        const std::shared_ptr<uint64_t> existing_lease_generation =
            existing.lease_generation.lock();
        TORCH_CHECK(
            existing.physical_execution_token == physical_execution_token &&
                existing.lease_epoch == lease_epoch &&
                existing.plan_hash == plan_hash &&
                existing.allocator_generation == allocator_generation &&
                existing.persistent_generation == persistent_generation &&
                existing.arena_base == arena_base &&
                existing.arena_end == arena_end &&
                existing.reserved_top_bytes == reserved_top_bytes &&
                existing.expected_lease_generation ==
                    expected_lease_generation &&
                existing_lease_generation == live_lease_generation,
            "RpuKernelGraph: all retained Graphs under one physical claim must "
            "bind the same lease epoch/plan/allocator generation");
    }
    TORCH_CHECK(registry.next_token != std::numeric_limits<uint64_t>::max(),
                "retained physical Graph guard token exhausted");
    const uint64_t token = ++registry.next_token;
    RetainedPhysicalArenaRecord record;
    record.token = token;
    record.graph_lifetime_id = graph_lifetime_id;
    record.graph_build_generation = graph_build_generation;
    record.graph_signature_identity = graph_signature_identity;
    record.graph_topology_hash = graph_topology_hash;
    record.physical_execution_token = physical_execution_token;
    record.lease_epoch = lease_epoch;
    record.plan_hash = plan_hash;
    record.allocator_generation = allocator_generation;
    record.persistent_generation = persistent_generation;
    record.arena_base = arena_base;
    record.arena_end = arena_end;
    record.reserved_top_bytes = reserved_top_bytes;
    record.lease_generation = lease_generation;
    record.expected_lease_generation = expected_lease_generation;
    record.owner_thread = std::this_thread::get_id();
    record.binding_state = RetainedPhysicalArenaBindingState::Bound;
    const auto record_insert =
        registry.records.emplace(token, std::move(record));
    TORCH_INTERNAL_ASSERT(record_insert.second);
    try {
        const auto graph_insert =
            registry.token_by_graph_lifetime.emplace(
                graph_lifetime_id, token);
        TORCH_INTERNAL_ASSERT(graph_insert.second);
    } catch (...) {
        registry.records.erase(token);
        throw;
    }
    registry.provisional_by_graph_lifetime.erase(provisional);
    return token;
}

void validate_retained_physical_graph_replay(
        uint64_t graph_lifetime_id,
        uint64_t graph_build_generation,
        uint64_t graph_signature_identity,
        uint64_t graph_topology_hash,
        uint64_t physical_execution_token,
        bool exact_built_hit) {
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto graph_it =
        registry.token_by_graph_lifetime.find(graph_lifetime_id);
    if (graph_it == registry.token_by_graph_lifetime.end()) return;
    const auto record_it = registry.records.find(graph_it->second);
    TORCH_INTERNAL_ASSERT(record_it != registry.records.end());
    const auto& record = record_it->second;
    const std::shared_ptr<uint64_t> live_lease_generation =
        record.lease_generation.lock();
    TORCH_CHECK(
        record.binding_state == RetainedPhysicalArenaBindingState::Bound &&
            record.graph_state == RetainedGraphLifetimeState::Live &&
            exact_built_hit && live_lease_generation != nullptr &&
            record.expected_lease_generation != 0 &&
            *live_lease_generation == record.expected_lease_generation,
        "RpuKernelGraph: a retained physical Graph cannot recapture or rebuild; "
        "its lease authority must remain live; invalidate it, retire its "
        "physical guard, then build a new identity");
    TORCH_CHECK(
        record.owner_thread == std::this_thread::get_id() &&
            record.physical_execution_token == physical_execution_token &&
            record.graph_build_generation == graph_build_generation &&
            record.graph_signature_identity == graph_signature_identity &&
            record.graph_topology_hash == graph_topology_hash,
        "RpuKernelGraph: retained physical Graph identity is stale or foreign");
}

void mark_retained_physical_graph_invalidated(uint64_t graph_lifetime_id) {
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.provisional_by_graph_lifetime.erase(graph_lifetime_id);
    const auto graph_it =
        registry.token_by_graph_lifetime.find(graph_lifetime_id);
    if (graph_it == registry.token_by_graph_lifetime.end()) return;
    auto record_it = registry.records.find(graph_it->second);
    TORCH_INTERNAL_ASSERT(record_it != registry.records.end());
    if (record_it->second.graph_state == RetainedGraphLifetimeState::Live) {
        record_it->second.graph_state =
            RetainedGraphLifetimeState::Invalidated;
    }
}

void mark_retained_physical_graph_evicted_noexcept(
        uint64_t graph_lifetime_id) noexcept {
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.provisional_by_graph_lifetime.erase(graph_lifetime_id);
    const auto graph_it =
        registry.token_by_graph_lifetime.find(graph_lifetime_id);
    if (graph_it == registry.token_by_graph_lifetime.end()) return;
    const auto record_it = registry.records.find(graph_it->second);
    if (record_it == registry.records.end()) return;
    if (record_it->second.graph_state == RetainedGraphLifetimeState::Live) {
        record_it->second.graph_state = RetainedGraphLifetimeState::Evicted;
    }
}

bool graph_has_retained_physical_arena_guard(uint64_t graph_lifetime_id) {
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    return registry.token_by_graph_lifetime.count(graph_lifetime_id) != 0;
}

uint64_t next_execution_token_locked(ProcessExecutionState& state) {
    TORCH_CHECK(state.generation != std::numeric_limits<uint64_t>::max(),
                "RPU execution coordinator generation exhausted");
    return ++state.generation;
}

void clear_execution_owner_if_idle_locked(ProcessExecutionState& state) {
    if (state.graph_depth == 0 && !state.physical_active &&
        state.cleanup_depth == 0) {
        state.owner_tid = std::thread::id{};
    }
}

class GraphClaimRollback final {
public:
    explicit GraphClaimRollback(RpuExecutionCoordinator::Claim& claim)
        : claim_(&claim) {}
    ~GraphClaimRollback() {
        if (claim_ != nullptr) {
            RpuExecutionCoordinator::exit_graph_noexcept(*claim_);
        }
    }
    void dismiss() noexcept { claim_ = nullptr; }

private:
    RpuExecutionCoordinator::Claim* claim_;
};

// BUILD stamps may outlive the graph object that issued them. A per-object
// counter therefore permits an address-reuse ABA: a freshly constructed Graph
// at the same address could issue generation 1 again. Allocate process-unique
// BUILD-attempt nonces instead. Relaxed ordering is sufficient because the
// counter carries identity only; Graph state synchronization remains external.
std::atomic<uint64_t> g_build_attempt_nonce{0};

uint64_t next_build_attempt_nonce() {
    uint64_t observed = g_build_attempt_nonce.load(std::memory_order_relaxed);
    for (;;) {
        TORCH_CHECK(observed != std::numeric_limits<uint64_t>::max(),
                    "RpuKernelGraph BUILD attempt nonce exhausted");
        const uint64_t next = observed + 1;
        if (g_build_attempt_nonce.compare_exchange_weak(
                observed, next,
                std::memory_order_relaxed,
                std::memory_order_relaxed)) {
            return next;
        }
    }
}

class StableGraphHasher {
public:
    void u8(uint8_t value) {
        hash_ ^= value;
        hash_ *= UINT64_C(0x100000001b3);
    }

    void boolean(bool value) { u8(value ? 1 : 0); }

    void u64(uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            u8(static_cast<uint8_t>((value >> shift) & UINT64_C(0xff)));
        }
    }

    void i64(int64_t value) { u64(static_cast<uint64_t>(value)); }

    void string(std::string_view value) {
        u64(static_cast<uint64_t>(value.size()));
        for (unsigned char ch : value) u8(ch);
    }

    uint64_t finish() const { return hash_ == 0 ? 1 : hash_; }

private:
    uint64_t hash_ = UINT64_C(0xcbf29ce484222325);
};

template <typename T>
void hash_integral_vector(StableGraphHasher& hash,
                          const std::vector<T>& values) {
    hash.u64(static_cast<uint64_t>(values.size()));
    for (T value : values) {
        if constexpr (std::is_signed_v<T>) {
            hash.i64(static_cast<int64_t>(value));
        } else {
            hash.u64(static_cast<uint64_t>(value));
        }
    }
}

void hash_graph_signature(StableGraphHasher& hash,
                          const GraphSignature& signature) {
    // op_id_str is diagnostic-only and intentionally absent from equality.
    hash.i64(signature.op_id);
    hash_integral_vector(hash, signature.shapes);
    hash_integral_vector(hash, signature.dyn_dims);
    hash_integral_vector(hash, signature.dtypes);
    hash.u64(signature.flags);
    hash.u64(signature.branch_key);
    hash.u64(signature.segment_key);
}

uint64_t stable_graph_signature_identity(const GraphSignature& signature) {
    StableGraphHasher hash;
    hash.u64(UINT64_C(0x4752415048534947));  // "GRAPHSIG"
    hash_graph_signature(hash, signature);
    return hash.finish();
}

void hash_tensor_topology(StableGraphHasher& hash, const at::Tensor& tensor) {
    hash.boolean(tensor.defined());
    if (!tensor.defined()) return;
    hash.u64(static_cast<uint64_t>(tensor.scalar_type()));
    hash.u64(static_cast<uint64_t>(tensor.device().type()));
    hash.i64(static_cast<int64_t>(tensor.device().index()));
    hash.u64(static_cast<uint64_t>(tensor.layout()));
    hash.u64(static_cast<uint64_t>(tensor.dim()));
    for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
        hash.i64(tensor.size(dim));
    }
    if (tensor.layout() == at::kStrided) {
        hash.i64(tensor.storage_offset());
        for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
            hash.i64(tensor.stride(dim));
        }
    }
}

void hash_ivalue_topology(StableGraphHasher& hash,
                          const c10::IValue& value,
                          size_t depth = 0) {
    // Tensor payloads and arbitrary objects may carry host/device pointers. The
    // scalar/list/tuple forms that replay validates by value are serialized
    // recursively; unknown pointer-bearing forms are rejected instead of being
    // under-hashed by tag alone.
    const std::string tag = value.tagKind();
    hash.string(tag);
    TORCH_CHECK(depth < 32,
                "Graph topology IValue nesting exceeds 32 at tag '", tag,
                "'");
    if (value.isNone()) return;
    if (value.isTensor()) {
        TORCH_CHECK(!value.toTensor().defined(),
                    "Graph topology hash accepts only undefined Tensor "
                    "placeholders in args_template");
        return;
    }

    if (value.isInt()) {
        // ScalarType/Layout/MemoryFormat/QScheme are represented by IValue's
        // pointer-free Int tag and therefore share this exact-value path.
        hash.i64(value.toInt());
        return;
    }
    if (value.isBool()) {
        hash.boolean(value.toBool());
        return;
    }
    if (value.isDouble()) {
        uint64_t bits = 0;
        const double scalar = value.toDouble();
        static_assert(sizeof(bits) == sizeof(scalar));
        std::memcpy(&bits, &scalar, sizeof(bits));
        hash.u64(bits);
        return;
    }
    if (value.isComplexDouble()) {
        const c10::complex<double> scalar = value.toComplexDouble();
        const double real = scalar.real();
        const double imag = scalar.imag();
        uint64_t real_bits = 0;
        uint64_t imag_bits = 0;
        std::memcpy(&real_bits, &real, sizeof(real_bits));
        std::memcpy(&imag_bits, &imag, sizeof(imag_bits));
        hash.u64(real_bits);
        hash.u64(imag_bits);
        return;
    }
    if (value.isUnsigned()) {
        hash.u64(value.toUInt());
        return;
    }
    if (value.isString()) {
        hash.string(value.toStringRef());
        return;
    }
    if (value.isDevice()) {
        const c10::Device device = value.toDevice();
        hash.u64(static_cast<uint64_t>(device.type()));
        hash.i64(static_cast<int64_t>(device.index()));
        return;
    }
    if (value.isIntList()) {
        const std::vector<int64_t> values = value.toIntVector();
        hash_integral_vector(hash, values);
        return;
    }
    if (value.isBoolList()) {
        const c10::List<bool> values = value.toBoolList();
        hash.u64(static_cast<uint64_t>(values.size()));
        for (size_t index = 0; index < values.size(); ++index) {
            hash.boolean(values.get(index));
        }
        return;
    }
    if (value.isDoubleList()) {
        const std::vector<double> values = value.toDoubleVector();
        hash.u64(static_cast<uint64_t>(values.size()));
        for (double scalar : values) {
            uint64_t bits = 0;
            std::memcpy(&bits, &scalar, sizeof(bits));
            hash.u64(bits);
        }
        return;
    }
    if (value.isTuple()) {
        const auto& values = value.toTupleRef().elements();
        hash.u64(static_cast<uint64_t>(values.size()));
        for (const c10::IValue& item : values) {
            hash_ivalue_topology(hash, item, depth + 1);
        }
        return;
    }
    if (value.isList()) {
        const auto values = value.toListRef();
        hash.u64(static_cast<uint64_t>(values.size()));
        for (const c10::IValue& item : values) {
            hash_ivalue_topology(hash, item, depth + 1);
        }
        return;
    }
    TORCH_CHECK(false,
                "Graph topology hash rejects unsupported replay-equality "
                "IValue tag '", tag,
                "' (object/dict/future/RRef/generator/symbolic and other "
                "pointer-bearing forms require a typed topology contract)");
}

void hash_operator_identity(StableGraphHasher& hash,
                            const std::optional<c10::OperatorHandle>& op) {
    hash.boolean(op.has_value());
    if (!op.has_value()) return;
    const auto& name = op->operator_name();
    hash.string(name.name);
    hash.string(name.overload_name);
}

void hash_register_patches(StableGraphHasher& hash,
                           const std::vector<RegisterPatch>& patches) {
    hash.u64(static_cast<uint64_t>(patches.size()));
    for (const RegisterPatch& patch : patches) {
        hash.u64(static_cast<uint64_t>(patch.kernel_idx));
        hash.u64(patch.reg_idx);
        hash.u8(patch.width);
        hash.u64(patch.stride);
        // width=8 is explicitly an address+stride patch. Exclude its device
        // address while retaining scalar patch values for widths 1/2/4.
        if (patch.width != 8) hash.u64(patch.value);
    }
}

void hash_data_patches(StableGraphHasher& hash,
                       const std::vector<DataPatch>& patches) {
    hash.u64(static_cast<uint64_t>(patches.size()));
    for (const DataPatch& patch : patches) {
        hash.u64(static_cast<uint64_t>(patch.node_idx));
        hash.u8(patch.field);
        // ptr and dev_addr are replay-time locations, not topology.
    }
}

void hash_graph_node_topology(StableGraphHasher& hash,
                              const GraphNode& node,
                              uint8_t dma_topology_schema,
                              bool outer_fast_schema) {
    hash.u8(static_cast<uint8_t>(node.kind));
    switch (node.kind) {
    case GraphNodeKind::Kernel: {
        const auto& kernel = node.as_kernel();
        // Kernel register payload lives in Kernel_t rather than GraphNode and
        // may contain device pointers. This hash deliberately stops at the
        // structural launcher identity; typed register/endpoint provenance is
        // a separate physical-admission gate.
        hash.boolean(kernel.kernel_id.has_value());
        if (kernel.kernel_id.has_value()) {
            hash.u64(static_cast<uint64_t>(*kernel.kernel_id));
        }
        hash.u64(static_cast<uint64_t>(kernel.kernel_idx));
        hash.string(kernel.kernel_name);
        hash_integral_vector(hash, kernel.grid_dims);
        hash_integral_vector(hash, kernel.core_ids);
        hash.boolean(kernel.queue_state.broadcast_mode);
        hash.boolean(kernel.queue_state.flush_icache);
        if (dma_topology_schema >= 5) {
            hash.boolean(kernel.register_census.has_value());
            if (kernel.register_census.has_value()) {
                const auto& census = *kernel.register_census;
                hash.u8(static_cast<uint8_t>(census.kind));
                hash.u64(static_cast<uint64_t>(census.policy_rule_index));
                hash.u64(static_cast<uint64_t>(census.operands.size()));
                for (const auto& operand : census.operands) {
                    hash.u8(static_cast<uint8_t>(operand.abi.role));
                    hash.u8(static_cast<uint8_t>(operand.abi.access));
                    hash.u8(static_cast<uint8_t>(operand.abi.encoding));
                    hash.u64(operand.abi.reg_lo);
                    hash.u64(operand.abi.reg_hi);
                    hash.u64(operand.tensor_topology_id);
                    hash.u64(static_cast<uint64_t>(operand.tensor_nbytes));
                    hash.u64(static_cast<uint64_t>(operand.storage_nbytes));
                }
            }
        }
        break;
    }
    case GraphNodeKind::Memcpy: {
        const auto& copy = node.as_memcpy();
        hash.u64(static_cast<uint64_t>(copy.bytes));
        hash.u8(static_cast<uint8_t>(copy.kind));
        hash.i64(copy.dst_pool_slot);
        hash.i64(copy.src_pool_slot);
        hash.i64(copy.dma_channel);
        break;
    }
    case GraphNodeKind::Memset: {
        const auto& fill = std::get<MemsetNodeData>(node.data);
        hash.u8(fill.value);
        hash.u64(static_cast<uint64_t>(fill.bytes));
        break;
    }
    case GraphNodeKind::ChildGraph: {
        const auto& child = node.as_child_graph();
        hash.boolean(static_cast<bool>(child.child));
        if (child.child) {
            hash.u64(child.child->build_topology_hash());
            hash.u64(static_cast<uint64_t>(child.child->graph_size()));
        }
        hash_graph_signature(hash, child.child_signature);
        hash_register_patches(hash, child.input_patches);
        hash_data_patches(hash, child.data_patches);
        break;
    }
    case GraphNodeKind::Branch: {
        const auto& branch = node.as_branch();
        hash.u64(branch.branch_key);
        hash_graph_signature(hash, branch.branch_signature);
        // label is diagnostic-only and is not replay-validated.
        break;
    }
    case GraphNodeKind::HostCallback: {
        const auto& callback = node.as_host_callback();
        hash_operator_identity(hash, callback.op_handle);
        hash.u8(static_cast<uint8_t>(callback.tier));
        hash.u64(static_cast<uint64_t>(callback.args_template.size()));
        for (const c10::IValue& value : callback.args_template) {
            hash_ivalue_topology(hash, value);
        }
        hash_integral_vector(hash, callback.tensor_arg_indices);
        hash_integral_vector(hash, callback.output_arg_indices);
        hash.u64(static_cast<uint64_t>(callback.output_pool.size()));
        for (const at::Tensor& tensor : callback.output_pool) {
            hash_tensor_topology(hash, tensor);
        }
        hash_integral_vector(hash, callback.output_pool_bytes);
        // output_pool_dev_addrs and live_tensor_args contain runtime locations.
        break;
    }
    case GraphNodeKind::Tier3Oneshot: {
        const auto& oneshot = node.as_tier3_oneshot();
        hash_operator_identity(hash, oneshot.op_handle);
        hash.u64(static_cast<uint64_t>(oneshot.args_template.size()));
        for (const c10::IValue& value : oneshot.args_template) {
            hash_ivalue_topology(hash, value);
        }
        hash_integral_vector(hash, oneshot.tensor_arg_indices);
        hash_integral_vector(hash, oneshot.input_dep_node_ids);
        hash_integral_vector(hash, oneshot.output_dep_node_ids);
        hash_integral_vector(hash, oneshot.transient_resource_ids);
        // last_output_dev_addrs/live tensors/keepalives are runtime locations.
        break;
    }
    case GraphNodeKind::Dma: {
        const auto& dma = node.as_dma();
        hash.u8(static_cast<uint8_t>(dma.variant));
        hash.u64(static_cast<uint64_t>(dma.bytes));
        hash.u8(dma.channel);
        hash.i64(dma.live_offset);
        if (dma_topology_schema >= 2) {
            hash.u64(dma.semantic_endpoint_id);
        }
        if (dma_topology_schema >= 3) {
            hash.boolean(dma.semantic_canonical_member_emission);
        }
        if (dma_topology_schema >= 4) {
            hash.u64(dma.semantic_spm_peer_id);
        }
        if (outer_fast_schema) {
            hash.boolean(dma.outer_fast_owner_bound);
            if (dma.outer_fast_owner_bound) {
                hash.u8(static_cast<uint8_t>(dma.outer_fast_ddr_side));
                hash.u64(static_cast<uint64_t>(
                    dma.outer_fast_owner_ordinal));
                hash.u64(dma.outer_fast_owner_topology_id);
                hash.u64(static_cast<uint64_t>(
                    dma.outer_fast_owner_byte_offset));
            }
        }
        // src/dst_addr and live_base are runtime device/host locations.
        break;
    }
    case GraphNodeKind::Barrier: {
        const auto& barrier = node.as_barrier_node();
        hash.u8(barrier.self_stream);
        hash.u8(barrier.target_stream);
        break;
    }
    }
}

struct RegisteredGraphRegistry {
    std::mutex mutex;
    std::vector<std::weak_ptr<RpuKernelGraph>> graphs;
};

RegisteredGraphRegistry& registered_graph_registry() {
    // Process-lifetime registry: a deliberately leaked owner avoids static
    // destruction ordering against Python-held Graph objects at interpreter
    // shutdown. Entries are weak and therefore do not extend graph lifetime.
    static auto* registry = new RegisteredGraphRegistry();
    return *registry;
}

}  // namespace

RpuRetainedPhysicalArenaGuard::RpuRetainedPhysicalArenaGuard(
        uint64_t token,
        uint64_t graph_lifetime_id,
        uint64_t graph_build_generation,
        uint64_t graph_signature_identity,
        uint64_t graph_topology_hash,
        uint64_t physical_execution_token,
        uint64_t lease_epoch,
        uint64_t plan_hash,
        uint64_t allocator_generation,
        uint64_t persistent_generation,
        size_t arena_base,
        size_t arena_end,
        size_t reserved_top_bytes,
        const std::weak_ptr<uint64_t>& lease_generation,
        uint64_t expected_lease_generation,
        std::thread::id owner_thread)
    : token_(token),
      graph_lifetime_id_(graph_lifetime_id),
      graph_build_generation_(graph_build_generation),
      graph_signature_identity_(graph_signature_identity),
      graph_topology_hash_(graph_topology_hash),
      physical_execution_token_(physical_execution_token),
      lease_epoch_(lease_epoch),
      plan_hash_(plan_hash),
      allocator_generation_(allocator_generation),
      persistent_generation_(persistent_generation),
      arena_base_(arena_base),
      arena_end_(arena_end),
      reserved_top_bytes_(reserved_top_bytes),
      lease_generation_(lease_generation),
      expected_lease_generation_(expected_lease_generation),
      owner_thread_(owner_thread),
      active_(true) {}

RpuRetainedPhysicalArenaGuard::RpuRetainedPhysicalArenaGuard(
        RpuRetainedPhysicalArenaGuard&& other) noexcept
    : token_(other.token_),
      graph_lifetime_id_(other.graph_lifetime_id_),
      graph_build_generation_(other.graph_build_generation_),
      graph_signature_identity_(other.graph_signature_identity_),
      graph_topology_hash_(other.graph_topology_hash_),
      physical_execution_token_(other.physical_execution_token_),
      lease_epoch_(other.lease_epoch_),
      plan_hash_(other.plan_hash_),
      allocator_generation_(other.allocator_generation_),
      persistent_generation_(other.persistent_generation_),
      arena_base_(other.arena_base_),
      arena_end_(other.arena_end_),
      reserved_top_bytes_(other.reserved_top_bytes_),
      lease_generation_(other.lease_generation_),
      expected_lease_generation_(other.expected_lease_generation_),
      owner_thread_(other.owner_thread_),
      active_(other.active_) {
    other.clear();
}

void RpuRetainedPhysicalArenaGuard::clear() noexcept {
    token_ = 0;
    graph_lifetime_id_ = 0;
    graph_build_generation_ = 0;
    graph_signature_identity_ = 0;
    graph_topology_hash_ = 0;
    physical_execution_token_ = 0;
    lease_epoch_ = 0;
    plan_hash_ = 0;
    allocator_generation_ = 0;
    persistent_generation_ = 0;
    arena_base_ = 0;
    arena_end_ = 0;
    reserved_top_bytes_ = 0;
    lease_generation_.reset();
    expected_lease_generation_ = 0;
    owner_thread_ = std::thread::id{};
    active_ = false;
}

void RpuRetainedPhysicalArenaGuard::park(
        const RpuPhysicalArenaAuthority& authority) {
    TORCH_CHECK(active_ && physical_execution_token_ != 0,
                "RpuRetainedPhysicalArenaGuard::park: guard is not bound");
    const uint64_t live_physical_token =
        current_owned_physical_execution_token(
            "RpuRetainedPhysicalArenaGuard::park",
            /*require_graph_quiescent=*/true);
    const std::shared_ptr<uint64_t> authority_generation =
        authority.lease_generation_.lock();
    const std::shared_ptr<uint64_t> guard_generation =
        lease_generation_.lock();
    TORCH_CHECK(
        owner_thread_ == std::this_thread::get_id() &&
            authority.owner_thread_ == owner_thread_ &&
            live_physical_token == physical_execution_token_ &&
            authority.physical_execution_token_ == physical_execution_token_ &&
            authority.lease_epoch_ == lease_epoch_ &&
            authority.plan_hash_ == plan_hash_ &&
            authority.allocator_generation_ == allocator_generation_ &&
            authority.persistent_generation_ == persistent_generation_ &&
            authority.arena_base_ == arena_base_ &&
            authority.arena_end_ == arena_end_ &&
            authority.reserved_top_bytes_ == reserved_top_bytes_ &&
            authority.expected_lease_generation_ ==
                expected_lease_generation_ &&
            authority_generation != nullptr &&
            authority_generation == guard_generation &&
            *authority_generation == expected_lease_generation_,
        "RpuRetainedPhysicalArenaGuard::park: authority is stale or does not "
        "match the bound physical lease");

    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto record_it = registry.records.find(token_);
    TORCH_CHECK(record_it != registry.records.end(),
                "RpuRetainedPhysicalArenaGuard::park: guard token is stale");
    auto& record = record_it->second;
    const std::shared_ptr<uint64_t> record_generation =
        record.lease_generation.lock();
    TORCH_CHECK(
        record.binding_state == RetainedPhysicalArenaBindingState::Bound &&
            record.graph_state == RetainedGraphLifetimeState::Live &&
            record.token == token_ &&
            record.graph_lifetime_id == graph_lifetime_id_ &&
            record.graph_build_generation == graph_build_generation_ &&
            record.graph_signature_identity == graph_signature_identity_ &&
            record.graph_topology_hash == graph_topology_hash_ &&
            record.physical_execution_token == physical_execution_token_ &&
            record.lease_epoch == lease_epoch_ &&
            record.plan_hash == plan_hash_ &&
            record.allocator_generation == allocator_generation_ &&
            record.persistent_generation == persistent_generation_ &&
            record.arena_base == arena_base_ &&
            record.arena_end == arena_end_ &&
            record.reserved_top_bytes == reserved_top_bytes_ &&
            record.expected_lease_generation ==
                expected_lease_generation_ &&
            record_generation == authority_generation &&
            record.owner_thread == owner_thread_,
        "RpuRetainedPhysicalArenaGuard::park: bound Graph/physical identity "
        "is stale");

    record.binding_state = RetainedPhysicalArenaBindingState::Parked;
    record.physical_execution_token = 0;
    record.lease_epoch = 0;
    record.allocator_generation = 0;
    record.lease_generation.reset();
    record.expected_lease_generation = 0;
    physical_execution_token_ = 0;
    lease_epoch_ = 0;
    allocator_generation_ = 0;
    lease_generation_.reset();
    expected_lease_generation_ = 0;
}

void RpuRetainedPhysicalArenaGuard::rebind(
        const RpuPhysicalArenaAuthority& authority) {
    TORCH_CHECK(active_ && physical_execution_token_ == 0,
                "RpuRetainedPhysicalArenaGuard::rebind: guard is not parked");
    const uint64_t live_physical_token =
        current_owned_physical_execution_token(
            "RpuRetainedPhysicalArenaGuard::rebind",
            /*require_graph_quiescent=*/true);
    const std::shared_ptr<uint64_t> authority_generation =
        authority.lease_generation_.lock();
    TORCH_CHECK(
        owner_thread_ == std::this_thread::get_id() &&
            authority.owner_thread_ == owner_thread_ &&
            authority.physical_execution_token_ == live_physical_token &&
            authority.lease_epoch_ != 0 &&
            authority.plan_hash_ == plan_hash_ &&
            authority.allocator_generation_ != 0 &&
            authority.persistent_generation_ == persistent_generation_ &&
            authority.arena_base_ == arena_base_ &&
            authority.arena_end_ == arena_end_ &&
            authority.reserved_top_bytes_ == reserved_top_bytes_ &&
            authority.expected_lease_generation_ != 0 &&
            authority_generation != nullptr &&
            *authority_generation == authority.expected_lease_generation_,
        "RpuRetainedPhysicalArenaGuard::rebind: new authority changed the "
        "retained plan, persistent placement, or arena extent");

    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto record_it = registry.records.find(token_);
    TORCH_CHECK(record_it != registry.records.end(),
                "RpuRetainedPhysicalArenaGuard::rebind: guard token is stale");
    auto& record = record_it->second;
    TORCH_CHECK(
        record.binding_state == RetainedPhysicalArenaBindingState::Parked &&
            record.graph_state == RetainedGraphLifetimeState::Live &&
            record.token == token_ &&
            record.graph_lifetime_id == graph_lifetime_id_ &&
            record.graph_build_generation == graph_build_generation_ &&
            record.graph_signature_identity == graph_signature_identity_ &&
            record.graph_topology_hash == graph_topology_hash_ &&
            record.physical_execution_token == 0 &&
            record.lease_epoch == 0 && record.plan_hash == plan_hash_ &&
            record.allocator_generation == 0 &&
            record.persistent_generation == persistent_generation_ &&
            record.arena_base == arena_base_ &&
            record.arena_end == arena_end_ &&
            record.reserved_top_bytes == reserved_top_bytes_ &&
            record.lease_generation.expired() &&
            record.expected_lease_generation == 0 &&
            record.owner_thread == owner_thread_,
        "RpuRetainedPhysicalArenaGuard::rebind: parked Graph identity is stale");
    for (const auto& item : registry.records) {
        const auto& existing = item.second;
        if (existing.token == token_ ||
            existing.binding_state ==
                RetainedPhysicalArenaBindingState::Parked) {
            continue;
        }
        const std::shared_ptr<uint64_t> existing_generation =
            existing.lease_generation.lock();
        TORCH_CHECK(
            existing.physical_execution_token == live_physical_token &&
                existing.lease_epoch == authority.lease_epoch_ &&
                existing.plan_hash == authority.plan_hash_ &&
                existing.allocator_generation ==
                    authority.allocator_generation_ &&
                existing.persistent_generation ==
                    authority.persistent_generation_ &&
                existing.arena_base == authority.arena_base_ &&
                existing.arena_end == authority.arena_end_ &&
                existing.reserved_top_bytes ==
                    authority.reserved_top_bytes_ &&
                existing.expected_lease_generation ==
                    authority.expected_lease_generation_ &&
                existing_generation == authority_generation,
            "RpuRetainedPhysicalArenaGuard::rebind: another retained Graph "
            "is bound to a different physical lease");
    }

    record.binding_state = RetainedPhysicalArenaBindingState::Bound;
    record.physical_execution_token = live_physical_token;
    record.lease_epoch = authority.lease_epoch_;
    record.allocator_generation = authority.allocator_generation_;
    record.lease_generation = authority.lease_generation_;
    record.expected_lease_generation = authority.expected_lease_generation_;
    physical_execution_token_ = live_physical_token;
    lease_epoch_ = authority.lease_epoch_;
    allocator_generation_ = authority.allocator_generation_;
    lease_generation_ = authority.lease_generation_;
    expected_lease_generation_ = authority.expected_lease_generation_;
}

void RpuRetainedPhysicalArenaGuard::retire() {
    if (!active_) return;
    TORCH_CHECK(owner_thread_ == std::this_thread::get_id(),
                "RpuRetainedPhysicalArenaGuard::retire: owner thread is "
                "stale or foreign");
    std::unique_ptr<RpuExecutionCleanupGuard> parked_cleanup;
    if (physical_execution_token_ != 0) {
        const uint64_t live_physical_token =
            current_owned_physical_execution_token(
                "RpuRetainedPhysicalArenaGuard::retire",
                /*require_graph_quiescent=*/true);
        TORCH_CHECK(live_physical_token == physical_execution_token_,
                    "RpuRetainedPhysicalArenaGuard::retire: physical token "
                    "is stale or foreign");
    } else {
        parked_cleanup = std::make_unique<RpuExecutionCleanupGuard>(
            "RpuRetainedPhysicalArenaGuard::retire parked");
    }

    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto record_it = registry.records.find(token_);
    TORCH_CHECK(record_it != registry.records.end(),
                "RpuRetainedPhysicalArenaGuard::retire: guard token is stale");
    const auto& record = record_it->second;
    const bool bound = physical_execution_token_ != 0;
    const std::shared_ptr<uint64_t> live_lease_generation =
        record.lease_generation.lock();
    TORCH_CHECK(
        record.token == token_ &&
            record.graph_lifetime_id == graph_lifetime_id_ &&
            record.graph_build_generation == graph_build_generation_ &&
            record.graph_signature_identity == graph_signature_identity_ &&
            record.graph_topology_hash == graph_topology_hash_ &&
            record.binding_state ==
                (bound ? RetainedPhysicalArenaBindingState::Bound
                       : RetainedPhysicalArenaBindingState::Parked) &&
            record.physical_execution_token == physical_execution_token_ &&
            record.lease_epoch == lease_epoch_ &&
            record.plan_hash == plan_hash_ &&
            record.allocator_generation == allocator_generation_ &&
            record.persistent_generation == persistent_generation_ &&
            record.arena_base == arena_base_ &&
            record.arena_end == arena_end_ &&
            record.reserved_top_bytes == reserved_top_bytes_ &&
            record.expected_lease_generation ==
                expected_lease_generation_ &&
            (bound ? (live_lease_generation != nullptr &&
                      *live_lease_generation == expected_lease_generation_)
                   : (live_lease_generation == nullptr &&
                      expected_lease_generation_ == 0)) &&
            record.owner_thread == owner_thread_,
        "RpuRetainedPhysicalArenaGuard::retire: retained Graph/physical "
        "identity is stale");
    TORCH_CHECK(
        record.graph_state != RetainedGraphLifetimeState::Live,
        "RpuRetainedPhysicalArenaGuard::retire: retained Graph must be "
        "invalidated or evicted before guard retirement");
    const auto graph_it =
        registry.token_by_graph_lifetime.find(graph_lifetime_id_);
    TORCH_INTERNAL_ASSERT(graph_it !=
                              registry.token_by_graph_lifetime.end() &&
                          graph_it->second == token_);
    registry.token_by_graph_lifetime.erase(graph_it);
    registry.records.erase(record_it);
    clear();
}

void RpuRetainedPhysicalArenaGuard::require_release_allowed(
        uint64_t lease_epoch,
        uint64_t plan_hash,
        uint64_t allocator_generation,
        const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "retained physical arena release check requires a name");
    TORCH_CHECK(lease_epoch != 0 && plan_hash != 0 &&
                    allocator_generation != 0,
                operation,
                ": physical lease epoch/plan/allocator generation must be "
                "nonzero");
    auto& registry = retained_physical_arena_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    TORCH_CHECK(
        registry.provisional_by_graph_lifetime.empty(), operation,
        ": an unarmed physical Graph BUILD is still active; invalidate it "
        "or arm its retained guard before physical release");
    for (const auto& item : registry.records) {
        const auto& record = item.second;
        if (record.binding_state ==
            RetainedPhysicalArenaBindingState::Parked) {
            continue;
        }
        TORCH_CHECK(
            record.lease_epoch == lease_epoch &&
                record.plan_hash == plan_hash &&
                record.allocator_generation == allocator_generation,
            operation,
            ": active retained Graph belongs to a different physical arena "
            "identity");
    }
    const bool bound = std::any_of(
        registry.records.begin(), registry.records.end(),
        [](const auto& item) {
            return item.second.binding_state ==
                RetainedPhysicalArenaBindingState::Bound;
        });
    TORCH_CHECK(!bound, operation,
                ": retained physical Graph guard is still bound; park it or "
                "invalidate/evict the Graph and retire its guard before "
                "physical release");
}

RpuKernelGraph::LifetimeRetirementSentinel::~LifetimeRetirementSentinel() {
    mark_retained_physical_graph_evicted_noexcept(lifetime_id);
}

RpuKernelGraph::RpuKernelGraph()
    : lifetime_retirement_(next_graph_lifetime_id()),
      graph_lifetime_id_(lifetime_retirement_.lifetime_id) {}

RpuKernelGraph::~RpuKernelGraph() = default;

RpuExecutionCoordinator::Claim RpuExecutionCoordinator::enter_graph(
        const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "Graph execution claim requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.cleanup_depth == 0 && !state.shutting_down,
                operation, ": Graph execution is forbidden during process "
                "runtime cleanup");
    if (state.owner_tid == std::thread::id{}) {
        state.owner_tid = current;
    } else {
        TORCH_CHECK(state.owner_tid == current,
                    operation, ": RPU execution is owned by another thread");
        TORCH_INTERNAL_ASSERT(state.graph_depth > 0 || state.physical_active);
    }
    const uint64_t token = next_execution_token_locked(state);
    state.graph_tokens.push_back(token);
    state.graph_depth = state.graph_tokens.size();
    return Claim(ClaimKind::Graph, token, current);
}

void RpuExecutionCoordinator::validate_graph(
        const Claim& claim, const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "Graph claim validation requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(claim.kind_ == ClaimKind::Graph && claim.valid() &&
                    claim.owner_thread_ == current &&
                    state.owner_tid == current &&
                    state.cleanup_depth == 0 &&
                    state.graph_depth == state.graph_tokens.size() &&
                    !state.graph_tokens.empty() &&
                    state.graph_tokens.back() == claim.token_,
                operation,
                ": Graph scope owner/thread/token is stale or non-top");
}

void RpuExecutionCoordinator::exit_graph(
        Claim& claim, const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "Graph claim exit requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(claim.kind_ == ClaimKind::Graph && claim.valid() &&
                    claim.owner_thread_ == current &&
                    state.owner_tid == current &&
                    state.cleanup_depth == 0 &&
                    state.graph_depth == state.graph_tokens.size() &&
                    !state.graph_tokens.empty() &&
                    state.graph_tokens.back() == claim.token_,
                operation,
                ": Graph scope owner/thread/token is stale or non-top");
    state.graph_tokens.pop_back();
    state.graph_depth = state.graph_tokens.size();
    claim.clear();
    clear_execution_owner_if_idle_locked(state);
}

void RpuExecutionCoordinator::exit_graph_noexcept(
        Claim& claim) noexcept {
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (claim.kind_ != ClaimKind::Graph || !claim.valid() ||
        claim.owner_thread_ != current || state.owner_tid != current ||
        state.cleanup_depth != 0 || state.graph_tokens.empty() ||
        state.graph_tokens.back() != claim.token_) {
        return;
    }
    state.graph_tokens.pop_back();
    state.graph_depth = state.graph_tokens.size();
    claim.clear();
    clear_execution_owner_if_idle_locked(state);
}

RpuExecutionCoordinator::Claim RpuExecutionCoordinator::enter_physical(
        const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "physical execution claim requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.graph_depth == 0 && !state.physical_active &&
                    state.cleanup_depth == 0 && !state.shutting_down,
                operation,
                ": physical arena acquisition requires a process-quiescent "
                "Graph/physical/cleanup state");
    TORCH_CHECK(state.owner_tid == std::thread::id{},
                operation, ": process execution owner is not idle");
    state.owner_tid = current;
    const uint64_t token = next_execution_token_locked(state);
    state.physical_active = true;
    state.physical_token = token;
    return Claim(ClaimKind::Physical, token, current);
}

void RpuExecutionCoordinator::validate_physical(
        const Claim& claim, const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "physical claim validation requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(claim.kind_ == ClaimKind::Physical && claim.valid() &&
                    claim.owner_thread_ == current &&
                    state.owner_tid == current && state.physical_active &&
                    state.physical_token == claim.token_ &&
                    state.cleanup_depth == 0,
                operation,
                ": physical arena owner/thread/token is stale or foreign");
}

uint64_t RpuExecutionCoordinator::current_physical_token_if_owned(
        const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "physical token query requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (!state.physical_active) return 0;
    TORCH_CHECK(!state.shutting_down && state.cleanup_depth == 0 &&
                    state.physical_token != 0 &&
                    state.owner_tid == current,
                operation,
                ": physical arena claim is stale or owned by another thread");
    return state.physical_token;
}

void RpuExecutionCoordinator::require_no_physical_claim(
        const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "physical exclusion check requires an operation name");
    auto& state = process_execution_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(!state.physical_active,
                operation,
                ": prepared ChildGraph extraction/direct replay is "
                "forbidden while a physical arena claim is active");
}

void RpuExecutionCoordinator::require_graph_quiescent(
        const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "Graph-quiescent check requires an operation name");
    auto& state = process_execution_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.graph_depth == 0,
                operation, ": operation requires no process-active Graph");
    require_no_active_retained_physical_arena_guards(operation);
}

void RpuExecutionCoordinator::exit_physical(
        Claim& claim, const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "physical claim exit requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(claim.kind_ == ClaimKind::Physical && claim.valid() &&
                    claim.owner_thread_ == current &&
                    state.owner_tid == current && state.physical_active &&
                    state.physical_token == claim.token_ &&
                    state.graph_depth == 0 && state.cleanup_depth == 0,
                operation,
                ": physical arena owner/thread/token is stale, foreign, or "
                "has an active Graph");
    require_no_active_retained_physical_arena_guards(operation);
    state.physical_active = false;
    state.physical_token = 0;
    claim.clear();
    clear_execution_owner_if_idle_locked(state);
}

bool RpuExecutionCoordinator::physical_exit_allowed_noexcept(
        const Claim& claim) noexcept {
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    return claim.kind_ == ClaimKind::Physical && claim.valid() &&
           claim.owner_thread_ == current && state.owner_tid == current &&
           state.physical_active && state.physical_token == claim.token_ &&
           state.graph_depth == 0 && state.cleanup_depth == 0 &&
           !has_active_retained_physical_arena_guards();
}

void RpuExecutionCoordinator::exit_physical_noexcept(
        Claim& claim) noexcept {
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (claim.kind_ != ClaimKind::Physical || !claim.valid() ||
        claim.owner_thread_ != current || state.owner_tid != current ||
        !state.physical_active || state.physical_token != claim.token_ ||
        state.graph_depth != 0 || state.cleanup_depth != 0 ||
        has_active_retained_physical_arena_guards()) {
        return;
    }
    state.physical_active = false;
    state.physical_token = 0;
    claim.clear();
    clear_execution_owner_if_idle_locked(state);
}

RpuExecutionCoordinator::Claim RpuExecutionCoordinator::enter_cleanup(
        const char* operation,
        bool shutting_down,
        const char* owned_graph_diagnostic) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "runtime cleanup claim requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.graph_depth != 0 && state.owner_tid == current &&
        owned_graph_diagnostic != nullptr) {
        TORCH_CHECK(false, owned_graph_diagnostic);
    }
    TORCH_CHECK(state.graph_depth == 0 && !state.physical_active,
                operation,
                ": runtime cleanup requires no process-active Graph or "
                "physical arena");
    if (state.cleanup_depth == 0) {
        TORCH_CHECK(!state.shutting_down,
                    operation, ": runtime is permanently shut down");
        TORCH_CHECK(state.owner_tid == std::thread::id{},
                    operation, ": process execution owner is not idle");
        state.owner_tid = current;
    } else {
        TORCH_CHECK(state.owner_tid == current,
                    operation, ": runtime cleanup is owned by another thread");
    }
    const uint64_t token = next_execution_token_locked(state);
    state.cleanup_tokens.push_back(token);
    state.cleanup_depth = state.cleanup_tokens.size();
    state.shutting_down = state.shutting_down || shutting_down;
    return Claim(ClaimKind::Cleanup, token, current);
}

void RpuExecutionCoordinator::exit_cleanup(
        Claim& claim, const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "runtime cleanup exit requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(claim.kind_ == ClaimKind::Cleanup && claim.valid() &&
                    claim.owner_thread_ == current &&
                    state.owner_tid == current && state.graph_depth == 0 &&
                    !state.physical_active &&
                    state.cleanup_depth == state.cleanup_tokens.size() &&
                    !state.cleanup_tokens.empty() &&
                    state.cleanup_tokens.back() == claim.token_,
                operation,
                ": runtime cleanup owner/thread/token is stale or non-top");
    state.cleanup_tokens.pop_back();
    state.cleanup_depth = state.cleanup_tokens.size();
    claim.clear();
    clear_execution_owner_if_idle_locked(state);
}

void RpuExecutionCoordinator::exit_cleanup_noexcept(
        Claim& claim) noexcept {
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    if (claim.kind_ != ClaimKind::Cleanup || !claim.valid() ||
        claim.owner_thread_ != current || state.owner_tid != current ||
        state.graph_depth != 0 || state.physical_active ||
        state.cleanup_tokens.empty() ||
        state.cleanup_tokens.back() != claim.token_) {
        return;
    }
    state.cleanup_tokens.pop_back();
    state.cleanup_depth = state.cleanup_tokens.size();
    claim.clear();
    clear_execution_owner_if_idle_locked(state);
}

RpuExecutionCoordinator::Claim
RpuExecutionCoordinator::enter_allocator_mutation(
        const char* operation, bool& graph_claim_active) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "allocator mutation admission requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.cleanup_depth == 0 && !state.shutting_down,
                operation, ": allocator mutation is forbidden during "
                "runtime cleanup");
    TORCH_CHECK(!state.physical_active,
                operation, ": allocator mutation requires no process-active "
                "physical arena");
    if (state.graph_depth != 0) {
        TORCH_CHECK(state.owner_tid == current,
                    operation, ": RPU execution is owned by another thread");
        graph_claim_active = true;
        return Claim{};
    }

    TORCH_CHECK(state.owner_tid == std::thread::id{},
                operation, ": process execution owner is not idle");
    state.owner_tid = current;
    const uint64_t token = next_execution_token_locked(state);
    state.cleanup_tokens.push_back(token);
    state.cleanup_depth = state.cleanup_tokens.size();
    graph_claim_active = false;
    return Claim(ClaimKind::Cleanup, token, current);
}

RpuExecutionCoordinator::Claim
RpuExecutionCoordinator::enter_graph_invalidation(
        const char* operation,
        uint64_t expected_physical_token,
        bool& physical_claim_active) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "Graph invalidation admission requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.graph_depth == 0,
                operation,
                ": Graph invalidation is forbidden while a Graph scope is "
                "active; use abort for scope cleanup");

    if (state.physical_active) {
        TORCH_CHECK(expected_physical_token != 0 &&
                        expected_physical_token == state.physical_token &&
                        state.cleanup_depth == 0 &&
                        !state.shutting_down &&
                        state.owner_tid == current,
                    operation,
                    ": physical Graph invalidation requires the exact owning "
                    "live physical arena claim");
        physical_claim_active = true;
        return Claim{};
    }

    if (state.cleanup_depth == 0) {
        TORCH_CHECK(!state.shutting_down,
                    operation, ": runtime is permanently shut down");
        TORCH_CHECK(state.owner_tid == std::thread::id{},
                    operation, ": process execution owner is not idle");
        state.owner_tid = current;
    } else {
        TORCH_CHECK(state.owner_tid == current,
                    operation, ": runtime cleanup is owned by another thread");
    }
    const uint64_t token = next_execution_token_locked(state);
    state.cleanup_tokens.push_back(token);
    state.cleanup_depth = state.cleanup_tokens.size();
    physical_claim_active = false;
    return Claim(ClaimKind::Cleanup, token, current);
}

void RpuExecutionCoordinator::check_current_thread_execution_allowed(
        const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "execution ownership check requires an operation name");
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    TORCH_CHECK(state.cleanup_depth == 0 && !state.shutting_down,
                operation, ": execution is forbidden during runtime cleanup");
    TORCH_CHECK((state.graph_depth == 0 && !state.physical_active) ||
                    state.owner_tid == current,
                operation, ": RPU execution is owned by another thread");
}

bool RpuExecutionCoordinator::current_thread_execution_allowed_noexcept()
        noexcept {
    auto& state = process_execution_state();
    const std::thread::id current = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.cleanup_depth == 0 && !state.shutting_down &&
        ((state.graph_depth == 0 && !state.physical_active) ||
         state.owner_tid == current);
}

bool RpuExecutionCoordinator::has_graph_scope() {
    auto& state = process_execution_state();
    std::lock_guard<std::mutex> lock(state.mutex);
    return state.graph_depth != 0;
}

RpuExecutionCleanupGuard::RpuExecutionCleanupGuard(
        const char* operation,
        bool shutting_down,
        const char* owned_graph_diagnostic)
    : claim_(RpuExecutionCoordinator::enter_cleanup(
          operation, shutting_down, owned_graph_diagnostic)) {}

RpuExecutionCleanupGuard::~RpuExecutionCleanupGuard() {
    RpuExecutionCoordinator::exit_cleanup_noexcept(claim_);
}

RpuExecutionAllocatorMutationGuard::RpuExecutionAllocatorMutationGuard(
        const char* operation)
    : cleanup_claim_(RpuExecutionCoordinator::enter_allocator_mutation(
          operation, graph_claim_active_)) {}

RpuExecutionAllocatorMutationGuard::~RpuExecutionAllocatorMutationGuard() {
    if (cleanup_claim_.valid()) {
        RpuExecutionCoordinator::exit_cleanup_noexcept(cleanup_claim_);
    }
}

RpuExecutionGraphInvalidationGuard::RpuExecutionGraphInvalidationGuard(
        const char* operation, uint64_t expected_physical_token)
    : cleanup_claim_(RpuExecutionCoordinator::enter_graph_invalidation(
          operation, expected_physical_token, physical_claim_active_)) {}

RpuExecutionGraphInvalidationGuard::~RpuExecutionGraphInvalidationGuard() {
    if (cleanup_claim_.valid()) {
        RpuExecutionCoordinator::exit_cleanup_noexcept(cleanup_claim_);
    }
}

std::shared_ptr<RpuKernelGraph> make_registered_rpu_kernel_graph() {
    auto graph = std::make_shared<RpuKernelGraph>();
    auto& registry = registered_graph_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.graphs.emplace_back(graph);
    return graph;
}

void invalidate_registered_rpu_kernel_graphs() {
    RpuExecutionCleanupGuard cleanup(
        "reset_graph_cache registered-Graph invalidation");
    std::vector<std::shared_ptr<RpuKernelGraph>> live;
    auto& registry = registered_graph_registry();
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        std::vector<std::weak_ptr<RpuKernelGraph>> retained;
        retained.reserve(registry.graphs.size());
        live.reserve(registry.graphs.size());
        for (const auto& weak : registry.graphs) {
            if (auto graph = weak.lock()) {
                retained.emplace_back(graph);
                live.emplace_back(std::move(graph));
            }
        }
        registry.graphs.swap(retained);
    }

    // Public reset is a between-forwards operation. Reject an open scope
    // before mutating any graph so the operation is all-or-nothing.
    for (const auto& graph : live) {
        const auto state = graph->state();
        TORCH_CHECK(
            state != RpuKernelGraph::State::RECORDING &&
                state != RpuKernelGraph::State::REPLAYING,
            "reset_graph_cache() cannot run while a graph scope is active");
    }
    for (const auto& graph : live) {
        graph->invalidate();
    }
}

// =============================================================================
// 活跃图管理
// =============================================================================

RpuKernelGraph& RpuKernelGraph::active() {
    return active_stack_.empty() ? fallback_graph_ : *active_stack_.back();
}

bool RpuKernelGraph::has_active() {
    return !active_stack_.empty();
}

void RpuKernelGraph::check_direct_immediate_dma_allowed(
        const char* operation) {
    TORCH_CHECK(operation != nullptr && *operation != '\0',
                "direct/immediate DMA admission requires an operation name");
    RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
    if (active_stack_.empty()) return;

    const RpuKernelGraph* graph =
        active_stack_.size() == 1 ? active_stack_.back() : nullptr;
    const bool plain_downgraded_passthrough =
        graph != nullptr && graph->scope_open_ &&
        graph->state_ == State::PASSTHROUGH &&
        !graph->replayable_ &&
        !graph->pending_kernel_id_.has_value() &&
        !graph->pending_kernel_name_.has_value() &&
        !graph->has_semantic_dma_nodes_ &&
        !graph->semantic_dma_replay_owner_armed_ &&
        !graph->has_semantic_spm_peer_nodes_ &&
        !graph->semantic_spm_peer_replay_armed_ &&
        !graph->pending_semantic_spm_producer_yield_.has_value() &&
        !graph->has_semantic_spm_producer_yield_nodes_ &&
        !graph->semantic_spm_producer_yield_replay_armed_ &&
        !graph->semantic_spm_producer_yield_poisoned_ &&
        graph->semantic_dma_owner_generation_.expired() &&
        graph->semantic_dma_expected_owner_generation_ == 0 &&
        !graph->canonical_dma_callback_active_ &&
        !graph->canonical_dma_poisoned_ &&
        !graph->canonical_dma_build_trace_active_ &&
        !graph->canonical_dma_build_trace_complete_ &&
        !graph->canonical_dma_burst_permit_.active &&
        !graph->kernel_register_census_active_ &&
        !graph->has_kernel_register_census_nodes_ &&
        !graph->kernel_register_census_complete_ &&
        !graph->kernel_register_census_poisoned_ &&
        !graph->kernel_register_census_invocation_armed_ &&
        !graph->pending_kernel_register_census_.has_value();
    TORCH_CHECK(
        plain_downgraded_passthrough,
        operation,
        ": direct/immediate DMA is forbidden with an active Graph unless "
        "the current-thread stack contains exactly one plain downgraded "
        "PASSTHROUGH scope; depth=",
        active_stack_.size(),
        " top_state=",
        static_cast<int>(active_stack_.back()->state_));
}

bool RpuKernelGraph::is_on_active_stack(const RpuKernelGraph& g) {
    for (const RpuKernelGraph* p : active_stack_) {
        if (p == &g) return true;
    }
    return false;
}

RpuRetainedPhysicalArenaGuard
RpuKernelGraph::arm_retained_physical_arena(
        const RpuPhysicalArenaAuthority& authority) {
    const uint64_t physical_execution_token =
        current_owned_physical_execution_token(
            "RpuKernelGraph::arm_retained_physical_arena",
            /*require_graph_quiescent=*/true);
    TORCH_CHECK(
        authority.owner_thread_ == std::this_thread::get_id() &&
            authority.physical_execution_token_ == physical_execution_token,
        "RpuKernelGraph::arm_retained_physical_arena requires an authority "
        "minted by this exact physical owner/claim");
    TORCH_CHECK(!scope_open_ && state_ == State::BUILT && replayable_ &&
                    has_built_signature_ && build_generation_ != 0 &&
                    !nodes_.empty(),
                "RpuKernelGraph::arm_retained_physical_arena requires a "
                "non-empty replayable signed BUILT Graph outside capture");
    TORCH_CHECK(
        built_physical_execution_token_ != 0 &&
            built_physical_execution_token_ == physical_execution_token,
        "RpuKernelGraph::arm_retained_physical_arena requires a Graph built "
        "by this exact physical claim");
    TORCH_CHECK(
        !ever_extracted_child_window_ && !ever_embedded_as_child_,
        "RpuKernelGraph::arm_retained_physical_arena rejects a Graph "
        "lifetime that detached or was embedded as a prepared child");
    for (const GraphNode& node : nodes_) {
        TORCH_CHECK(
            node.kind != GraphNodeKind::ChildGraph,
            "RpuKernelGraph::arm_retained_physical_arena rejects retained "
            "roots containing ChildGraph nodes");
    }
    const uint64_t signature_identity = built_signature_identity();
    const uint64_t topology_hash = build_topology_hash();
    TORCH_CHECK(topology_hash != 0,
                "RpuKernelGraph::arm_retained_physical_arena requires a "
                "sealed nonzero BUILD topology");
    const uint64_t token = register_retained_physical_arena_guard(
        graph_lifetime_id_, build_generation_, signature_identity,
        topology_hash, physical_execution_token, authority.lease_epoch_,
        authority.plan_hash_, authority.allocator_generation_,
        authority.persistent_generation_, authority.arena_base_,
        authority.arena_end_, authority.reserved_top_bytes_,
        authority.lease_generation_, authority.expected_lease_generation_);
    return RpuRetainedPhysicalArenaGuard(
        token, graph_lifetime_id_, build_generation_, signature_identity,
        topology_hash, physical_execution_token, authority.lease_epoch_,
        authority.plan_hash_, authority.allocator_generation_,
        authority.persistent_generation_, authority.arena_base_,
        authority.arena_end_, authority.reserved_top_bytes_,
        authority.lease_generation_, authority.expected_lease_generation_,
        std::this_thread::get_id());
}

void RpuKernelGraph::request_retained_physical_arena_build() {
    TORCH_CHECK(
        scope_open_ && state_ == State::RECORDING &&
            execution_owner_thread_ == std::this_thread::get_id() &&
            recording_physical_execution_token_ != 0,
        "RpuKernelGraph: retained physical build opt-in requires the active "
        "signed RECORDING scope under this thread's physical claim");
    TORCH_CHECK(
        !retained_physical_build_requested_,
        "RpuKernelGraph: retained physical build was requested twice");
    retained_physical_build_requested_ = true;
    topology_hash_requested_ = true;
}

bool RpuKernelGraph::has_retained_physical_arena_guard() const {
    return graph_has_retained_physical_arena_guard(graph_lifetime_id_);
}

bool RpuKernelGraph::has_provisional_physical_build() const {
    return ::graph_has_provisional_physical_build(graph_lifetime_id_);
}

uint64_t RpuKernelGraph::topology_hash_for_window(size_t begin,
                                                   size_t end) const {
    TORCH_INTERNAL_ASSERT(begin <= end && end <= nodes_.size());
    uint8_t dma_topology_schema = 1;
    for (size_t index = begin; index < end; ++index) {
        if (nodes_[index].kind == GraphNodeKind::Kernel &&
            nodes_[index].as_kernel().register_census.has_value()) {
            dma_topology_schema = 5;
            continue;
        }
        if (nodes_[index].kind == GraphNodeKind::Dma &&
            nodes_[index].as_dma().semantic_endpoint_id != 0) {
            if (nodes_[index].as_dma().semantic_spm_peer_id != 0) {
                dma_topology_schema = std::max<uint8_t>(
                    dma_topology_schema, 4);
                continue;
            }
            dma_topology_schema = std::max<uint8_t>(
                dma_topology_schema,
                nodes_[index].as_dma().semantic_canonical_member_emission
                    ? 3 : 2);
        }
    }
    StableGraphHasher hash;
    hash.u64(UINT64_C(0x4752415048544f50));  // "GRAPHTOP"
    // Legacy-only windows retain schema 1 byte-for-byte.  A window containing
    // a v11 semantic DMA uses schema 2 and writes an ID for every occurrence
    // (including zero for unrelated legacy nodes in a mixed window).  A v12
    // canonical-member occurrence upgrades only that window to schema 3 and
    // additionally hashes its wrapper provenance bit.  A v13 typed SPM peer
    // upgrades to schema 4 and writes an opaque peer ID for every DMA (zero for
    // unrelated nodes). Schema 5 is opt-in only for a typed register census;
    // it binds the stable policy/physical plan and pointer-free per-Kernel
    // operand ABI/topology. Schemas 1-4 remain byte-for-byte frozen.
    hash.u64(dma_topology_schema);
    hash.u64(static_cast<uint64_t>(end - begin));
    if (dma_topology_schema >= 5) {
        hash.u64(kernel_register_census_policy_.fingerprint);
        hash.u64(kernel_register_census_policy_digest_);
        hash.u64(kernel_register_census_plan_hash_);
    }
    const bool outer_fast_schema = dma_topology_schema >= 5 &&
        kernel_register_outer_fast_build_state_.staged;
    if (outer_fast_schema) {
        hash.u64(UINT64_C(0x4f55544641535431));  // "OUTFAST1"
        hash.u64(kernel_register_outer_fast_build_state_.physical_digest);
        hash.u64(kernel_register_outer_fast_build_state_.topology_digest);
    }
    for (size_t index = begin; index < end; ++index) {
        hash.u64(static_cast<uint64_t>(index - begin));
        hash_graph_node_topology(
            hash, nodes_[index], dma_topology_schema, outer_fast_schema);
    }
    const uint64_t producer_yield_digest =
        semantic_spm_producer_yield_id_digest_for_window(begin, end);
    if (producer_yield_digest != 0) {
        // Conditional extension: an unmarked window performs no extra hash
        // writes and therefore preserves every existing schema/hash byte-for-
        // byte.  Marked windows bind the ordered terminal-writer sidecar under
        // an independent domain without changing DMA schemas 1--5.
        hash.u64(UINT64_C(0x53504d594c443031));  // "SPMYLD01"
        hash.u64(producer_yield_digest);
    }
    return hash.finish();
}

uint64_t RpuKernelGraph::
semantic_spm_producer_yield_id_digest_for_window(
        size_t begin, size_t end) const {
    TORCH_INTERNAL_ASSERT(begin <= end && end <= nodes_.size());
    size_t count = 0;
    for (size_t index = begin; index < end; ++index) {
        if (nodes_[index].kind == GraphNodeKind::Kernel &&
            nodes_[index].as_kernel().semantic_spm_producer_yield.has_value()) {
            ++count;
        }
    }
    if (count == 0) return 0;

    StableGraphHasher hash;
    hash.u64(UINT64_C(0x53504d5944494431));  // "SPMYDID1"
    hash.u64(static_cast<uint64_t>(count));
    for (size_t index = begin; index < end; ++index) {
        if (nodes_[index].kind != GraphNodeKind::Kernel) continue;
        const auto& marker =
            nodes_[index].as_kernel().semantic_spm_producer_yield;
        if (!marker.has_value()) continue;
        hash.u64(static_cast<uint64_t>(index - begin));
        hash.u64(marker->semantic_yield_id);
        hash.u8(marker->writer_kind);
    }
    return hash.finish();
}

std::vector<GraphSemanticSpmProducerYieldOccurrence>
RpuKernelGraph::semantic_spm_producer_yields_for_window(
        size_t begin, size_t end) const {
    TORCH_INTERNAL_ASSERT(begin <= end && end <= nodes_.size());
    std::vector<GraphSemanticSpmProducerYieldOccurrence> occurrences;
    for (size_t index = begin; index < end; ++index) {
        if (nodes_[index].kind != GraphNodeKind::Kernel) continue;
        const auto& marker =
            nodes_[index].as_kernel().semantic_spm_producer_yield;
        if (!marker.has_value()) continue;
        GraphSemanticSpmProducerYieldOccurrence occurrence;
        occurrence.relative_node_index = index - begin;
        occurrence.semantic_yield_id = marker->semantic_yield_id;
        occurrence.writer_kind = marker->writer_kind;
        occurrences.push_back(std::move(occurrence));
    }
    return occurrences;
}

GraphNodeKindMask RpuKernelGraph::node_kind_mask_for_window(
        size_t begin, size_t end) const {
    TORCH_INTERNAL_ASSERT(begin <= end && end <= nodes_.size());
    GraphNodeKindMask mask = 0;
    for (size_t index = begin; index < end; ++index) {
        mask |= graph_node_kind_bit(nodes_[index].kind);
    }
    return mask;
}

GraphOpStreamStamp RpuKernelGraph::op_stream_stamp() const {
    TORCH_CHECK(scope_open_ && !active_stack_.empty() &&
                    active_stack_.back() == this,
                "op_stream_stamp requires this graph to own the active scope");
    TORCH_CHECK(!replay_child_skip_.active,
                "op_stream_stamp cannot describe a partially consumed "
                "ChildGraph window");

    const GraphSignature* signature = nullptr;
    size_t position = 0;
    switch (state_) {
    case State::RECORDING:
        topology_hash_requested_ = true;
        signature = pending_signature_ ? &*pending_signature_ : nullptr;
        position = nodes_.size();
        break;
    case State::REPLAYING:
        signature = has_built_signature_ ? &built_signature_ : nullptr;
        position = cursor_;
        break;
    case State::PASSTHROUGH:
    case State::BUILT:
        TORCH_CHECK(false,
                    "op_stream_stamp requires RECORDING or REPLAYING; state=",
                    static_cast<int>(state_));
    }

    GraphOpStreamStamp stamp;
    stamp.graph = this;
    stamp.state = state_;
    stamp.build_generation = build_generation_;
    stamp.signature_identity = signature == nullptr
        ? 0 : stable_graph_signature_identity(*signature);
    stamp.signature_segment_key = signature == nullptr
        ? 0 : signature->segment_key;
    stamp.position = position;
    stamp.extent = nodes_.size();
    return stamp;
}

uint64_t RpuKernelGraph::build_topology_hash() const {
    if (build_topology_hash_ != 0) return build_topology_hash_;
    if ((state_ == State::BUILT || state_ == State::REPLAYING) && replayable_ &&
        has_built_signature_ && !nodes_.empty()) {
        build_topology_hash_ = topology_hash_for_window(0, nodes_.size());
    }
    return build_topology_hash_;
}

GraphNodeKindMask RpuKernelGraph::build_node_kind_mask() const {
    TORCH_CHECK((state_ == State::BUILT || state_ == State::REPLAYING) &&
                    replayable_,
                "build_node_kind_mask requires a replayable BUILT or "
                "REPLAYING graph; state=", static_cast<int>(state_),
                " replayable=", replayable_);
    return node_kind_mask_for_window(0, nodes_.size());
}

uint64_t RpuKernelGraph::built_signature_identity() const {
    TORCH_CHECK(has_built_signature_ &&
                    (state_ == State::BUILT || state_ == State::REPLAYING),
                "built_signature_identity requires BUILT or REPLAYING with a "
                "committed signature");
    return stable_graph_signature_identity(built_signature_);
}

bool RpuKernelGraph::is_op_stream_stamp_current(
        const GraphOpStreamStamp& stamp) const {
    if (stamp.graph != this || stamp.build_generation == 0 ||
        stamp.build_generation != build_generation_ || !replayable_ ||
        state_ == State::PASSTHROUGH) {
        return false;
    }

    const GraphSignature* signature = nullptr;
    if (state_ == State::RECORDING) {
        signature = pending_signature_ ? &*pending_signature_ : nullptr;
    } else if (has_built_signature_) {
        signature = &built_signature_;
    }
    const uint64_t identity = signature == nullptr
        ? 0 : stable_graph_signature_identity(*signature);
    const uint64_t segment_key = signature == nullptr
        ? 0 : signature->segment_key;
    return stamp.signature_identity == identity &&
           stamp.signature_segment_key == segment_key &&
           stamp.position <= nodes_.size() && stamp.extent <= nodes_.size();
}

GraphOpStreamWindowDigest RpuKernelGraph::digest_op_stream_window(
        const GraphOpStreamStamp& begin,
        const GraphOpStreamStamp& end) const {
    TORCH_CHECK(state_ == State::RECORDING && replayable_,
                "digest_op_stream_window requires a live replayable RECORDING");
    topology_hash_requested_ = true;
    TORCH_CHECK(begin.graph == this && end.graph == this,
                "digest_op_stream_window: stamp graph mismatch");
    TORCH_CHECK(begin.state == State::RECORDING &&
                    end.state == State::RECORDING,
                "digest_op_stream_window accepts only RECORDING stamps");
    TORCH_CHECK(begin.build_generation == build_generation_ &&
                    end.build_generation == build_generation_ &&
                    begin.build_generation != 0,
                "digest_op_stream_window: stale BUILD generation");
    TORCH_CHECK(begin.signature_identity == end.signature_identity &&
                    begin.signature_segment_key ==
                        end.signature_segment_key,
                "digest_op_stream_window: signature drift");
    TORCH_CHECK(begin.extent == begin.position && end.extent == end.position,
                "digest_op_stream_window: malformed RECORDING stamp");
    TORCH_CHECK(begin.position <= end.position &&
                    end.position <= nodes_.size(),
                "digest_op_stream_window: invalid node range [",
                begin.position, ",", end.position,
                ") for graph_size=", nodes_.size());

    const GraphOpStreamStamp current = op_stream_stamp();
    TORCH_CHECK(end.signature_identity == current.signature_identity &&
                    end.signature_segment_key ==
                        current.signature_segment_key,
                "digest_op_stream_window: current signature drift");

    GraphOpStreamWindowDigest digest;
    digest.begin = begin.position;
    digest.end = end.position;
    digest.node_count = end.position - begin.position;
    digest.kind_mask = node_kind_mask_for_window(begin.position, end.position);
    digest.topology_hash = topology_hash_for_window(begin.position, end.position);
    digest.semantic_spm_producer_yield_id_digest =
        semantic_spm_producer_yield_id_digest_for_window(
            begin.position, end.position);
    for (size_t index = begin.position; index < end.position; ++index) {
        if (nodes_[index].kind == GraphNodeKind::Kernel) {
            const auto& marker =
                nodes_[index].as_kernel().semantic_spm_producer_yield;
            if (marker.has_value()) {
                GraphSemanticSpmProducerYieldOccurrence occurrence;
                occurrence.relative_node_index = index - begin.position;
                occurrence.semantic_yield_id = marker->semantic_yield_id;
                occurrence.writer_kind = marker->writer_kind;
                digest.semantic_spm_producer_yields.push_back(
                    std::move(occurrence));
            }
        }
        if (nodes_[index].kind != GraphNodeKind::Dma) continue;
        const auto& dma = nodes_[index].as_dma();
        if (dma.semantic_endpoint_id == 0) {
            ++digest.legacy_dma_count;
            continue;
        }
        GraphSemanticDmaOccurrence occurrence;
        occurrence.relative_node_index = index - begin.position;
        occurrence.endpoint_id = dma.semantic_endpoint_id;
        occurrence.variant = static_cast<uint8_t>(dma.variant);
        occurrence.src_addr = dma.src_addr;
        occurrence.dst_addr = dma.dst_addr;
        occurrence.bytes = dma.bytes;
        occurrence.channel = dma.channel;
        occurrence.live_offset = dma.live_offset;
        occurrence.live_base = dma.live_base;
        occurrence.owner_generation = dma.semantic_owner_generation;
        occurrence.expected_owner_generation =
            dma.semantic_expected_owner_generation;
        occurrence.canonical_member_emission =
            dma.semantic_canonical_member_emission;
        occurrence.spm_peer_id = dma.semantic_spm_peer_id;
        digest.semantic_dmas.push_back(std::move(occurrence));
    }
    return digest;
}

bool RpuKernelGraph::has_semantic_dma_outside_window(
        size_t begin, size_t end) const {
    TORCH_CHECK(begin <= end && end <= nodes_.size(),
                "has_semantic_dma_outside_window: invalid node range [",
                begin, ",", end, ") for graph_size=", nodes_.size());
    for (size_t index = 0; index < nodes_.size(); ++index) {
        if (index >= begin && index < end) continue;
        if (nodes_[index].kind == GraphNodeKind::Dma &&
            nodes_[index].as_dma().semantic_endpoint_id != 0) {
            return true;
        }
    }
    return false;
}

bool RpuKernelGraph::has_semantic_spm_producer_yield_marker_nodes() const {
    if (has_semantic_spm_producer_yield_nodes_) return true;
    std::vector<const RpuKernelGraph*> pending{this};
    std::unordered_set<const RpuKernelGraph*> visited;
    while (!pending.empty()) {
        const RpuKernelGraph* graph = pending.back();
        pending.pop_back();
        if (graph == nullptr || !visited.insert(graph).second) continue;
        for (const GraphNode& node : graph->nodes_) {
            if (node.kind == GraphNodeKind::Kernel &&
                node.as_kernel().semantic_spm_producer_yield.has_value()) {
                return true;
            }
            if (node.kind == GraphNodeKind::ChildGraph) {
                const auto& child = node.as_child_graph().child;
                if (child) pending.push_back(child.get());
            }
        }
    }
    return false;
}

void RpuKernelGraph::stage_semantic_spm_producer_yield_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation,
        uint64_t semantic_yield_id,
        uint8_t writer_kind) {
    TORCH_CHECK(scope_open_ && !active_stack_.empty() &&
                    active_stack_.back() == this &&
                    (state_ == State::RECORDING ||
                     state_ == State::REPLAYING),
                "semantic SPM producer yield staging requires this graph's "
                "active RECORDING or REPLAYING scope");
    TORCH_CHECK(!semantic_spm_producer_yield_poisoned_ &&
                    !pending_semantic_spm_producer_yield_.has_value(),
                "semantic SPM producer yield is poisoned or already pending");
    TORCH_CHECK(!pending_kernel_id_.has_value() &&
                    !pending_kernel_name_.has_value(),
                "semantic SPM producer yield must be staged before the next "
                "Graph kernel lookup");
    TORCH_CHECK(owner_generation != nullptr && semantic_yield_id != 0 &&
                    expected_owner_generation != 0 &&
                    *owner_generation == expected_owner_generation,
                "semantic SPM producer yield owner is stale or its ID is zero");

    const std::shared_ptr<const uint64_t> bound_owner =
        semantic_spm_producer_yield_owner_generation_.lock();
    if (state_ == State::RECORDING) {
        // A terminal-writer marker is part of the sealed BUILD topology.
        // Prevent post-BUILD child-window rewriting from burying or rebasing
        // it without a fresh generation/digest.
        topology_hash_requested_ = true;
        if (bound_owner == nullptr) {
            TORCH_CHECK(!has_semantic_spm_producer_yield_nodes_,
                        "semantic SPM producer yield BUILD owner expired");
            semantic_spm_producer_yield_owner_generation_ = owner_generation;
            semantic_spm_producer_yield_expected_owner_generation_ =
                expected_owner_generation;
        } else {
            TORCH_CHECK(
                bound_owner.get() == owner_generation.get() &&
                    semantic_spm_producer_yield_expected_owner_generation_ ==
                        expected_owner_generation,
                "semantic SPM producer yield BUILD owner changed");
        }
    } else {
        TORCH_CHECK(has_semantic_spm_producer_yield_nodes_ &&
                        semantic_spm_producer_yield_replay_armed_ &&
                        semantic_spm_producer_yield_replay_arm_id_digest_ ==
                            semantic_spm_producer_yield_id_digest_ &&
                        bound_owner != nullptr &&
                        bound_owner.get() == owner_generation.get() &&
                        semantic_spm_producer_yield_expected_owner_generation_ ==
                            expected_owner_generation,
                    "semantic SPM producer yield REPLAY is unarmed or foreign");
    }

    PendingSemanticSpmProducerYield pending;
    pending.semantic_yield_id = semantic_yield_id;
    pending.writer_kind = writer_kind;
    pending.owner_generation = owner_generation;
    pending.expected_owner_generation = expected_owner_generation;
    pending_semantic_spm_producer_yield_.emplace(std::move(pending));
}

void RpuKernelGraph::cancel_semantic_spm_producer_yield_for_fmb() noexcept {
    if (pending_semantic_spm_producer_yield_.has_value()) {
        semantic_spm_producer_yield_poisoned_ = true;
    }
    pending_semantic_spm_producer_yield_.reset();
}

void RpuKernelGraph::poison_semantic_spm_producer_yield_for_fmb() noexcept {
    if (scope_open_ &&
        (state_ == State::RECORDING || state_ == State::REPLAYING)) {
        semantic_spm_producer_yield_poisoned_ = true;
    }
    pending_semantic_spm_producer_yield_.reset();
}

void RpuKernelGraph::arm_semantic_spm_producer_yield_replay_for_fmb(
        const std::shared_ptr<uint64_t>& owner_generation,
        uint64_t expected_owner_generation,
        uint64_t semantic_yield_id_digest) {
    const std::shared_ptr<const uint64_t> bound_owner =
        semantic_spm_producer_yield_owner_generation_.lock();
    TORCH_CHECK(scope_open_ && !active_stack_.empty() &&
                    active_stack_.back() == this &&
                    state_ == State::REPLAYING && cursor_ == 0,
                "semantic SPM producer yield arm requires REPLAYING cursor 0");
    TORCH_CHECK(!semantic_spm_producer_yield_poisoned_ &&
                    !semantic_spm_producer_yield_replay_armed_ &&
                    !pending_semantic_spm_producer_yield_.has_value(),
                "semantic SPM producer yield arm is duplicate or poisoned");
    TORCH_CHECK(has_semantic_spm_producer_yield_nodes_ &&
                    has_semantic_spm_producer_yield_marker_nodes() &&
                    semantic_spm_producer_yield_id_digest_ != 0 &&
                    semantic_yield_id_digest ==
                        semantic_spm_producer_yield_id_digest_,
                "semantic SPM producer yield ID digest is absent or stale");
    TORCH_CHECK(owner_generation != nullptr && bound_owner != nullptr &&
                    bound_owner.get() == owner_generation.get() &&
                    expected_owner_generation != 0 &&
                    semantic_spm_producer_yield_expected_owner_generation_ ==
                        expected_owner_generation &&
                    *owner_generation == expected_owner_generation,
                "semantic SPM producer yield replay owner is stale or foreign");
    semantic_spm_producer_yield_replay_arm_id_digest_ =
        semantic_yield_id_digest;
    semantic_spm_producer_yield_replay_armed_ = true;
}

uint64_t RpuKernelGraph::
semantic_spm_producer_yield_id_digest_for_fmb() const {
    TORCH_CHECK((state_ == State::BUILT || state_ == State::REPLAYING) &&
                    has_semantic_spm_producer_yield_nodes_ &&
                    semantic_spm_producer_yield_id_digest_ != 0,
                "semantic SPM producer yield digest requires a marked BUILT "
                "graph");
    return semantic_spm_producer_yield_id_digest_;
}

void RpuKernelGraph::verify_semantic_spm_producer_yield_replay_arm(
        const char* operation) const {
    if (!has_semantic_spm_producer_yield_marker_nodes()) return;
    const std::shared_ptr<const uint64_t> owner =
        semantic_spm_producer_yield_owner_generation_.lock();
    TORCH_CHECK(
        has_semantic_spm_producer_yield_nodes_ &&
            !semantic_spm_producer_yield_poisoned_ &&
            semantic_spm_producer_yield_replay_armed_ &&
            semantic_spm_producer_yield_replay_arm_id_digest_ != 0 &&
            semantic_spm_producer_yield_replay_arm_id_digest_ ==
                semantic_spm_producer_yield_id_digest_ &&
            owner != nullptr &&
            semantic_spm_producer_yield_expected_owner_generation_ != 0 &&
            *owner ==
                semantic_spm_producer_yield_expected_owner_generation_,
        operation,
        ": semantic SPM producer-yield Graph is stale or unarmed");
}

void RpuKernelGraph::verify_semantic_spm_producer_yield_end_preflight() {
    TORCH_CHECK(!semantic_spm_producer_yield_poisoned_ &&
                    !pending_semantic_spm_producer_yield_.has_value(),
                "RpuKernelGraph: semantic SPM producer yield is poisoned or "
                "pending before Graph execution");
    const bool observed = has_semantic_spm_producer_yield_marker_nodes();
    TORCH_CHECK(observed == has_semantic_spm_producer_yield_nodes_,
                "RpuKernelGraph: semantic SPM producer yield marker state "
                "drifted from the retained Graph");
    if (!observed) return;

    const std::shared_ptr<const uint64_t> owner =
        semantic_spm_producer_yield_owner_generation_.lock();
    TORCH_CHECK(replayable_ && owner != nullptr &&
                    semantic_spm_producer_yield_expected_owner_generation_ !=
                        0 &&
                    *owner ==
                        semantic_spm_producer_yield_expected_owner_generation_,
                "RpuKernelGraph: semantic SPM producer-yield Graph owner is "
                "stale or the Graph is non-replayable");
    const uint64_t digest =
        semantic_spm_producer_yield_id_digest_for_window(0, nodes_.size());
    TORCH_INTERNAL_ASSERT(digest != 0);
    if (state_ == State::RECORDING) {
        TORCH_CHECK(pending_signature_.has_value() &&
                        !semantic_spm_producer_yield_replay_armed_,
                    "RpuKernelGraph: producer-yield BUILD must remain signed "
                    "and cannot carry a replay arm");
        semantic_spm_producer_yield_id_digest_ = digest;
        return;
    }
    TORCH_CHECK(state_ == State::REPLAYING &&
                    digest == semantic_spm_producer_yield_id_digest_,
                "RpuKernelGraph: semantic SPM producer-yield Graph topology "
                "drifted before REPLAY");
    verify_semantic_spm_producer_yield_replay_arm("RpuKernelGraph::end");
}

void RpuKernelGraph::verify_composite_fmb_end_preflight() const {
    using Phase = CompositeFmbInvocationState::Phase;
    const auto& composite = composite_fmb_invocation_;
    if (composite.phase == Phase::None) return;
    const GraphOpStreamStamp stamp = op_stream_stamp();
    const bool build_complete =
        state_ == State::RECORDING &&
        composite.phase == Phase::BuildComplete;
    const bool replay_complete =
        state_ == State::REPLAYING &&
        composite.phase == Phase::ReplayComplete;
    TORCH_CHECK(
        (build_complete || replay_complete) &&
            composite.coordinator != nullptr &&
            composite.identity != 0 &&
            composite.build_generation == build_generation_ &&
            composite.signature_identity == stamp.signature_identity &&
            composite.signature_segment_key ==
                stamp.signature_segment_key &&
            composite.expected_occurrences > 0 &&
            composite.completed_occurrences ==
                composite.expected_occurrences &&
            composite.expected_producer_yields > 0 &&
            composite.completed_producer_yields ==
                composite.expected_producer_yields &&
            composite.next_outer_begin == nodes_.size() &&
            stamp.position == nodes_.size() &&
            (state_ != State::REPLAYING || cursor_ == nodes_.size()),
        "RpuKernelGraph: composite FMB invocation is incomplete, poisoned, "
        "or has an unclaimed Graph suffix before execution");
}

// =============================================================================
// begin / begin(sig) / begin_impl
// =============================================================================
//
// A2 — signature-driven admission:
//   begin(sig)   : 命中 BUILT+sig → REPLAYING;否则进 RECORDING + pending_signature_=sig
//   begin()      : 兼容形态,无 sig,任何状态都进 RECORDING(不命中 admission;BUILT
//                  会被 invalidate 重录)。
//   end()        : 若 RECORDING 落地成 BUILT 且 pending_signature_ 有值,则把它
//                  写入 built_signature_,设 has_built_signature_=true。
//
// 逻辑骨架由 begin_impl 持有,两个公开入口仅决定是否传 sig。

void RpuKernelGraph::begin() {
    begin_impl(std::nullopt);
}

void RpuKernelGraph::begin(const GraphSignature& sig) {
    begin_impl(sig);
}

void RpuKernelGraph::check_foreign_graph_execution_allowed(
        const char* operation,
        bool allow_managed_semantic_yield_scope_entry) const {
    RpuExecutionCoordinator::check_current_thread_execution_allowed(operation);
    TORCH_CHECK(
        allow_managed_semantic_yield_scope_entry ||
            state_ != State::BUILT ||
            !has_semantic_spm_producer_yield_marker_nodes(),
        operation,
        ": direct/baked replay of a semantic SPM producer-yield Graph is "
        "forbidden; enter signed REPLAY and arm the owning FMB first");
    if (active_stack_.empty() || active_stack_.back() == this) {
        return;
    }
    const RpuKernelGraph* outer = active_stack_.back();
    TORCH_CHECK(
        !outer->has_semantic_dma_nodes_ &&
            !outer->has_semantic_spm_producer_yield_nodes_ &&
            !outer->pending_semantic_spm_producer_yield_.has_value() &&
            !outer->semantic_spm_producer_yield_poisoned_ &&
            !outer->canonical_dma_build_trace_active_ &&
            !outer->canonical_dma_callback_active_ &&
            !outer->canonical_dma_burst_permit_.active &&
            !outer->canonical_dma_poisoned_ &&
            !outer->kernel_register_census_active_ &&
            !outer->has_kernel_register_census_nodes_ &&
            !outer->kernel_register_census_complete_ &&
            !outer->kernel_register_census_poisoned_ &&
            !outer->pending_kernel_register_census_.has_value(),
        operation,
        " is forbidden inside an active canonical/semantic/typed-register "
        "Graph scope");
}

void RpuKernelGraph::begin_impl(const std::optional<GraphSignature>& sig) {
    auto execution_claim = RpuExecutionCoordinator::enter_graph(
        "RpuKernelGraph::begin");
    GraphClaimRollback execution_rollback(execution_claim);

    TORCH_CHECK(!scope_open_,
                "RpuKernelGraph: begin() called twice without end()");
    // begin_impl itself owns signed-hit versus fresh-recapture admission.
    // Permit entry here so an exact hit can transition to REPLAYING and arm,
    // while a miss can invalidate the old marked topology before RECORDING.
    // Direct kernel/queue/child execution still uses the default false gate.
    check_foreign_graph_execution_allowed(
        "RpuKernelGraph: nested Graph begin",
        /*allow_managed_semantic_yield_scope_entry=*/true);
    // A1 — 允许 nested begin(不同 graph 实例),只拦截同一 graph 重入。
    TORCH_CHECK(std::find(active_stack_.begin(), active_stack_.end(), this)
                    == active_stack_.end(),
                "RpuKernelGraph: begin() called on a graph already on the active stack");
    constexpr size_t kMaxActiveDepth = 8;
    TORCH_CHECK(active_stack_.size() < kMaxActiveDepth,
                "RpuKernelGraph: active-stack depth exceeded ", kMaxActiveDepth);
    TORCH_INTERNAL_ASSERT(
        (!execution_claim_.has_value() || !execution_claim_->valid()) &&
        execution_owner_thread_ == std::thread::id{});
    execution_claim_.reset();

    // 命中 REPLAYING 的前提:(1) BUILT、(2) replayable、(3) 已有 built signature、
    // (4) 本次给了 sig 且与 built_signature_ 相等。无 sig 的 begin() 永远走 RECORDING。
    const bool hit_built =
        state_ == State::BUILT && replayable_ && has_built_signature_ &&
        sig.has_value() && *sig == built_signature_;
    const uint64_t entry_physical_token =
        RpuExecutionCoordinator::current_physical_token_if_owned(
            "RpuKernelGraph::begin physical ownership preflight");
    TORCH_CHECK(
        !graph_has_provisional_physical_build(graph_lifetime_id_),
        "RpuKernelGraph: an exact physical BUILD must be armed or "
        "invalidated before replay or recapture");
    const bool retained_physical_graph =
        graph_has_retained_physical_arena_guard(graph_lifetime_id_);
    if (retained_physical_graph) {
        const uint64_t physical_execution_token =
            current_owned_physical_execution_token(
                "RpuKernelGraph::begin retained physical replay",
                /*require_graph_quiescent=*/false);
        validate_retained_physical_graph_replay(
            graph_lifetime_id_, build_generation_,
            hit_built ? built_signature_identity() : 0,
            hit_built ? build_topology_hash() : 0,
            physical_execution_token, hit_built);
    }
    if (hit_built && has_semantic_dma_nodes_) {
        // A semantic owner arm is scope-local and non-transferable: begin()
        // always clears it, then the owning FMB token must arm this active
        // REPLAY before any typed emission/skip/end can execute the graph.
        semantic_dma_replay_owner_armed_ = false;
        semantic_spm_peer_replay_armed_ = false;
        cancel_canonical_dma_callback_for_fmb();
        canonical_dma_poisoned_ = false;
    }
    if (hit_built && has_semantic_spm_producer_yield_nodes_) {
        pending_semantic_spm_producer_yield_.reset();
        semantic_spm_producer_yield_replay_armed_ = false;
        semantic_spm_producer_yield_replay_arm_id_digest_ = 0;
        semantic_spm_producer_yield_poisoned_ = false;
    }
    replay_child_skip_ = ReplayChildSkipState{};
    if (state_ == State::PASSTHROUGH) {
        clear_kernel_register_build_state();
    } else if (hit_built) {
        clear_kernel_register_invocation_state();
    }

    switch (state_) {
    case State::PASSTHROUGH:
        kernels_.clear();
        nodes_.clear();
        segments_.clear();
        tensor_refs_.clear();
        boundary_flush_ptrs_.clear();
        // A fresh capture clears both admission tables. Persistent resources
        // use the owner tree and follow the same mark-released/clear sequence
        // as invalidate and abort.
        for (auto& r : host_callback_persistent_owners_) r.mark_released();
        host_callback_persistent_owners_.clear();
        host_callback_live_tier1_out_ranges_.clear();
        // ZeroCopyHolderGuard should remove zero-copy holders before
        // RECORDING. Clear any exception-path residue defensively.
        if (!host_callback_zero_copy_holders_.empty()) {
            for (auto& r : host_callback_zero_copy_holders_) r.mark_released();
            host_callback_zero_copy_holders_.clear();
        }
        // Re-entering RECORDING must clear the previous pool slots and reverse
        // lookup so an old CPU pointer cannot select a stale LocalSPM_t.
        local_spm_pool_.clear();
        global_spm_pool_.clear();
        // RECORDING dependency inference binds dev_addr-to-writer mappings to
        // the current capture; clear them to prevent stale dependency edges.
        recording_dev_addr_writer_.clear();
        // Tier3 transient buffers are not reused across captures because op
        // shape or dtype may change. Mark them released before clearing.
        for (auto& r : tier3_oneshot_transient_owners_) r.mark_released();
        tier3_oneshot_transient_owners_.clear();
        tier3_oneshot_transient_tensors_.clear();
        last_stats_ = {};
        pending_kernel_id_.reset();
        pending_kernel_name_.reset();
        replayable_ = true;
        non_replayable_reason_.clear();
        cursor_ = 0;
        // post_fn fast-replay: fresh capture re-records the post_fn-start
        // cursor via mark_post_fn_cursor; clear the stale flag here too.
        has_post_fn_cursor_ = false;
        pending_signature_ = sig;          // nullopt 时本轮 end() 不落 BUILT
        state_ = State::RECORDING;
        break;

    case State::BUILT:
        if (hit_built) {
            cursor_ = 0;
            // 不复用 capture 期旧 ptr;REPLAYING 期由 record_boundary_flush
            // 的调用点重新登记本轮当前的 ptr 值。
            boundary_flush_ptrs_.clear();
            // REPLAYING consumes the same LocalSPM_t in RECORDING order. Reset
            // only the cursor; slots and reverse lookup remain stable.
            local_spm_pool_.reset_cursor();
            global_spm_pool_.reset_cursor();
            // A new REPLAY keeps persistent stable storage but clears live
            // Tier1 .out arguments, which are registered again for this step.
            host_callback_live_tier1_out_ranges_.clear();
            state_ = State::REPLAYING;
        } else {
            // sig miss(或本次无 sig)→ 丢弃旧 BUILT,重新 RECORDING。
            // RpuGraphCache 命中 entry.graph 时这条分支不应触发(否则等于同一 entry
            // 的 sig 不一致,A2 起记 recapture_count)。
            invalidate_impl(/*from_abort=*/false);
            replayable_ = true;
            pending_signature_ = sig;
            state_ = State::RECORDING;
        }
        break;

    default:
        TORCH_CHECK(false,
                    "RpuKernelGraph: begin() called in invalid state ",
                    static_cast<int>(state_));
    }

    if (state_ == State::RECORDING) {
        recording_physical_execution_token_ = entry_physical_token;
        semantic_dma_owner_generation_.reset();
        semantic_dma_expected_owner_generation_ = 0;
        has_semantic_dma_nodes_ = false;
        semantic_dma_replay_owner_armed_ = false;
        has_semantic_spm_peer_nodes_ = false;
        semantic_spm_peer_replay_armed_ = false;
        pending_semantic_spm_producer_yield_.reset();
        semantic_spm_producer_yield_owner_generation_.reset();
        semantic_spm_producer_yield_expected_owner_generation_ = 0;
        semantic_spm_producer_yield_id_digest_ = 0;
        semantic_spm_producer_yield_replay_arm_id_digest_ = 0;
        has_semantic_spm_producer_yield_nodes_ = false;
        semantic_spm_producer_yield_replay_armed_ = false;
        semantic_spm_producer_yield_poisoned_ = false;
        composite_fmb_invocation_ = {};
        cancel_canonical_dma_callback_for_fmb();
        canonical_dma_poisoned_ = false;
        canonical_dma_build_trace_active_ = false;
        canonical_dma_build_trace_complete_ = false;
        canonical_dma_build_trace_profile_hash_ = 0;
        canonical_dma_build_trace_node_begin_ = 0;
        canonical_dma_build_trace_node_end_ = 0;
        op_stream_fully_skipped_ = false;
        build_generation_ = next_build_attempt_nonce();
        build_topology_hash_ = 0;
        topology_hash_requested_ = false;
        retained_physical_build_requested_ = false;
        const char* value = std::getenv("RPU_FASTREPLAY_SKIP_SYNC");
        fast_replay_skip_sync_ =
            value != nullptr && *value != '\0' && value[0] != '0';
    }

    // Push may allocate, so do it while the local process claim still has a
    // rollback guard and before changing the SPM defer depth.  Everything
    // after push is non-throwing bookkeeping.
    active_stack_.push_back(this);
    SPM_POOL.enter_defer_scope();
    execution_owner_thread_ = execution_claim.owner_thread();
    execution_claim_.emplace(std::move(execution_claim));
    scope_open_ = true;
    execution_rollback.dismiss();
}

// =============================================================================
// end
// =============================================================================

bool RpuKernelGraph::force_oneshot_on_replay_enabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY");
        return value != nullptr && *value != '\0' && *value != '0';
    }();
    return enabled;
}

void RpuKernelGraph::end() {
    RpuExecutionCoordinator::check_current_thread_execution_allowed(
        "RpuKernelGraph::end");
    TORCH_CHECK(scope_open_ && execution_claim_.has_value() &&
                    execution_claim_->valid(),
                "RpuKernelGraph: end() without matching begin()");
    RpuExecutionCoordinator::validate_graph(
        *execution_claim_, "RpuKernelGraph::end");
    TORCH_CHECK(!active_stack_.empty() && active_stack_.back() == this,
                "RpuKernelGraph: end() without matching begin() or non-top of "
                "active stack");
    verify_composite_fmb_end_preflight();
    verify_semantic_spm_producer_yield_end_preflight();
    TORCH_CHECK(!canonical_dma_poisoned_ &&
                    !canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active,
                "RpuKernelGraph: canonical FMB DMA scope is poisoned or "
                "remains open before Graph execution");
    TORCH_CHECK(!has_semantic_dma_nodes_ || replayable_,
                "RpuKernelGraph: semantic FMB DMA Graph cannot downgrade to "
                "non-replayable one-shot before sealing");
    if (canonical_dma_build_trace_active_) {
        const GraphSignature* signature =
            state_ == State::RECORDING && pending_signature_.has_value()
            ? &*pending_signature_ : nullptr;
        TORCH_CHECK(
            canonical_dma_build_trace_complete_ &&
                state_ == State::RECORDING && signature != nullptr &&
                signature->segment_key ==
                    canonical_dma_build_trace_profile_hash_ &&
                canonical_dma_build_trace_node_begin_ <=
                    canonical_dma_build_trace_node_end_ &&
                canonical_dma_build_trace_node_end_ == nodes_.size(),
            "RpuKernelGraph: canonical FMB DMA BUILD trace is incomplete or "
            "has an unclaimed Graph suffix before execution");
    }
    if (has_semantic_dma_nodes_) {
        const std::shared_ptr<const uint64_t> owner =
            semantic_dma_owner_generation_.lock();
        TORCH_CHECK(
            owner != nullptr &&
                semantic_dma_expected_owner_generation_ != 0 &&
                *owner == semantic_dma_expected_owner_generation_ &&
                (state_ != State::REPLAYING ||
                 semantic_dma_replay_owner_armed_),
            "RpuKernelGraph: semantic DMA owner is stale or unarmed before "
            "Graph execution");
    }
    TORCH_CHECK(
        !has_semantic_spm_peer_nodes_ ||
            (has_semantic_dma_nodes_ &&
             (state_ != State::REPLAYING ||
              semantic_spm_peer_replay_armed_)),
        "RpuKernelGraph: typed SPM peer is unarmed before Graph execution");

    // Schema-5 is fail-closed before topology hashing, boundary DDR flush,
    // Queue_t preparation, or any SDK submission.
    verify_kernel_register_end_preflight();

    switch (state_) {
    case State::RECORDING:
        if (nodes_.empty()) {
            // 空 capture（整个 scope 没提交任何 kernel）→ 直接落回 PASSTHROUGH
            pending_signature_.reset();
            recording_physical_execution_token_ = 0;
            built_physical_execution_token_ = 0;
            state_ = State::PASSTHROUGH;
        } else if (replayable_ && pending_signature_.has_value()) {
            const uint64_t topology_hash =
                (topology_hash_requested_ ||
                 has_kernel_register_census_nodes_ ||
                 retained_physical_build_requested_)
                ? topology_hash_for_window(0, nodes_.size()) : 0;
            // Replayable 路径：对 boundary ptr 做 pre + post 双 flush。
            //
            // 仅靠 post-batch flush 无法覆盖本轮 kernel 之前就必须对 RPU 可见
            // 的 DDR 输入，例如：
            //   - CPU fallback / host tensor op 刚写完的 hidden_states
            //   - 每步新生成的 cos/sin / mask / cache metadata tensor
            // data node（capture_data 的 memcpy/memset）仍会在
            // execute_graph_for_recording 内按原顺序穿插到下游 kernel 前，
            // 但 boundary_flush_ptrs_ 里登记的「纯 flush 请求」必须在 batch
            // 前先落一次，才能保持 eager 语义。
            flush_boundary_ptrs();
            flush_kernel_register_outer_fast_force_visibility();
            verify_kernel_register_outer_fast_launch_preflight();
            execute_graph_for_recording();
            flush_boundary_ptrs();
            flush_kernel_register_outer_fast_force_post_visibility();
            mark_kernel_register_outer_fast_execution_success();
            boundary_flush_ptrs_.clear();
            built_signature_ = *pending_signature_;
            has_built_signature_ = true;
            build_topology_hash_ = topology_hash;
            pending_signature_.reset();
            state_ = State::BUILT;
            built_physical_execution_token_ =
                retained_physical_build_requested_
                ? recording_physical_execution_token_ : 0;
            recording_physical_execution_token_ = 0;
            retained_physical_build_requested_ = false;
            if (built_physical_execution_token_ != 0) {
                register_provisional_physical_graph_build(
                    graph_lifetime_id_, build_generation_,
                    built_signature_identity(), build_topology_hash_,
                    built_physical_execution_token_);
            }
            if (has_kernel_register_census_nodes_) {
                kernel_register_census_build_committed_ = true;
                if (kernel_register_outer_fast_build_state_.staged) {
                    kernel_register_outer_fast_build_state_.committed = true;
                }
            }
        } else {
            // Non-replayable（sync_point 打断 / 无 sig / raw kernel 标记）
            // → 一次性执行 + 丢弃 segments_。同样需要 pre + post 双 flush，
            // 保持与 eager 路径一致的可见性语义。无 sig 的 begin() 永远走这条
            // 路径(pending_signature_ 是 nullopt),end 后回到 PASSTHROUGH。
            flush_boundary_ptrs();
            execute_graph_oneshot();
            flush_boundary_ptrs();
            boundary_flush_ptrs_.clear();
            pending_signature_.reset();
            recording_physical_execution_token_ = 0;
            built_physical_execution_token_ = 0;
            retained_physical_build_requested_ = false;
            state_ = State::PASSTHROUGH;
        }
        break;

    case State::REPLAYING: {
        TORCH_CHECK(cursor_ == nodes_.size(),
                    "RpuKernelGraph: replay incomplete: cursor=", cursor_,
                    " expected=", nodes_.size());
        // Trace ranges:把 REPLAYING 期 end() 收尾拆成 pre-flush /
        // execute / post-flush / stats 四段,Chrome tracing / Perfetto 里直接
        // 搜 `rpu_graph::*`。无语义改变。
        RECORD_FUNCTION("rpu_graph::end_REPLAYING", {});
        // REPLAYING 同样需要 pre + post 双 flush。
        // 原因：虽然 data node 不会在 REPLAYING 期 eager 写 DDR，但本轮
        // forward 里仍可能有 CPU fallback / host tensor op 先产出 DDR 输入，
        // 对应的 rpu_ddr_flush(ptr) 会登记到 boundary_flush_ptrs_。这些输入
        // 必须在 launch_prepared_batch() 前对 RPU 可见。
        {
            RECORD_FUNCTION("rpu_graph::flush_pre", {});
            flush_boundary_ptrs();
        }
        flush_kernel_register_outer_fast_force_visibility();
        verify_kernel_register_outer_fast_launch_preflight();
        // 诊断开关 RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY=1 强制走 oneshot 路径
        // (每步 build_segments + prepare_kernel_mutable + launch), 不复用
        // prepared_wq，用于隔离 prepared-queue 的 mutable-kernel 同步行为。
        //
        // This diagnostic changes behavior and must remain disabled for
        // production profiles: REPLAYING falls back to one-shot execution and
        // loses prepared-queue reuse. Always unset it after diagnosis.
        {
            if (force_oneshot_on_replay_enabled()) {
                RECORD_FUNCTION("rpu_graph::execute_replay_force_oneshot", {});
                segments_.clear();          // 丢弃 RECORDING 期 prepared_wq
                build_segments_from_nodes();
                execute_graph_oneshot();
                segments_.clear();
            } else {
                execute_graph_for_replaying();
            }
        }
        {
            RECORD_FUNCTION("rpu_graph::flush_post", {});
            flush_boundary_ptrs();
            flush_kernel_register_outer_fast_force_post_visibility();
        }
        mark_kernel_register_outer_fast_execution_success();
        boundary_flush_ptrs_.clear();
        state_ = State::BUILT;
        semantic_dma_replay_owner_armed_ = false;
        semantic_spm_peer_replay_armed_ = false;
        semantic_spm_producer_yield_replay_armed_ = false;
        semantic_spm_producer_yield_replay_arm_id_digest_ = 0;
        break;
    }

    case State::PASSTHROUGH:
        // sync_point() / raw-kernel 降级中途把 RECORDING 打断到 PASSTHROUGH。
        // 前缀已在 sync_point / enqueue 降级路径里落地执行（含 flush），
        // 这里只做收尾清理。
        recording_physical_execution_token_ = 0;
        built_physical_execution_token_ = 0;
        retained_physical_build_requested_ = false;
        break;

    default:
        TORCH_CHECK(false,
                    "RpuKernelGraph: end() called in invalid state ",
                    static_cast<int>(state_));
    }

    // keep_alive() 的引用在这里（执行之后）才释放，以覆盖 graph 的完整执行
    // 生命周期。该锚点不做 flush；相干性由 op 侧的
    // rpu_ddr_flush_force(执行前)和 boundary flush 机制负责。
    tensor_refs_.clear();
    pending_kernel_id_.reset();
    pending_kernel_name_.reset();
    if (has_kernel_register_census_nodes_) {
        last_stats_.register_census_present = true;
        last_stats_.register_census_build_committed =
            kernel_register_census_build_committed_;
        last_stats_.register_census_policy_fingerprint =
            kernel_register_census_policy_.fingerprint;
        last_stats_.register_census_policy_digest =
            kernel_register_census_policy_digest_;
        last_stats_.register_census_plan_hash =
            kernel_register_census_plan_hash_;
        last_stats_.register_census_live_epoch =
            kernel_register_census_live_epoch_;
        last_stats_.register_census = kernel_register_census_stats_;
        last_stats_.register_census_outer_fast_hit =
            kernel_register_outer_fast_hit_;
        last_stats_.register_census_outer_fast_retained_validated =
            kernel_register_outer_fast_retained_validated_;
        last_stats_.register_census_outer_fast_mutable_slot_count =
            kernel_register_outer_fast_mutable_slot_count_;
        last_stats_.register_census_outer_fast_mutable_occurrence_count =
            kernel_register_outer_fast_mutable_occurrence_count_;
        last_stats_.register_census_host_reemitted_node_count =
            kernel_register_host_reemitted_node_count_;
        last_stats_.register_census_host_reemitted_kernel_count =
            kernel_register_host_reemitted_kernel_count_;
    }
    clear_kernel_register_invocation_state();
    replay_child_skip_ = ReplayChildSkipState{};
    canonical_dma_build_trace_active_ = false;
    canonical_dma_build_trace_complete_ = false;
    canonical_dma_build_trace_profile_hash_ = 0;
    canonical_dma_build_trace_node_begin_ = 0;
    canonical_dma_build_trace_node_end_ = 0;
    // Keep the process claim until every local Graph/SPM write is complete.
    // This makes the coordinator unlock the happens-before boundary for a
    // later sequential thread; no Graph field is mutated after exit_graph().
    SPM_POOL.exit_defer_scope();
    active_stack_.pop_back();
    scope_open_ = false;
    execution_owner_thread_ = std::thread::id{};
    RpuExecutionCoordinator::exit_graph(
        *execution_claim_, "RpuKernelGraph::end");
}

// =============================================================================
// abort
// =============================================================================

void RpuKernelGraph::abort() {
    // 已经 end/abort 过或从未 begin:保持幂等 no-op。打开的 scope 则先验证
    // process claim；foreign thread 即使没有这条线程的 TLS stack 也必须拒绝，
    // 不能把 owner 的 claim 静默遗留。
    RpuExecutionCoordinator::check_current_thread_execution_allowed(
        "RpuKernelGraph::abort");
    if (!scope_open_) {
        return;
    }
    TORCH_CHECK(execution_claim_.has_value() && execution_claim_->valid(),
                "RpuKernelGraph: open scope has no process execution claim");
    RpuExecutionCoordinator::validate_graph(
        *execution_claim_, "RpuKernelGraph::abort");
    auto it = std::find(active_stack_.begin(), active_stack_.end(), this);
    TORCH_CHECK(it != active_stack_.end(),
                "RpuKernelGraph: abort() owner thread has no matching active "
                "stack entry");
    // A1 — 严格要求只能 abort 栈顶。非-top abort 是嵌套调用错乱,直接抛出
    // 而不是悄悄 erase 中间元素。
    TORCH_CHECK(it + 1 == active_stack_.end(),
                "RpuKernelGraph: abort() called on non-top graph; nested "
                "caller must abort inner graph first");
    TORCH_CHECK(!canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active,
                "RpuKernelGraph: abort is forbidden inside an active "
                "canonical FMB DMA callback/burst");
    if (canonical_dma_build_trace_active_) {
        canonical_dma_poisoned_ = true;
    }

    invalidate_impl(/*from_abort=*/true);
    tensor_refs_.clear();
    pending_kernel_id_.reset();
    pending_kernel_name_.reset();
    replay_child_skip_ = ReplayChildSkipState{};
    SPM_POOL.exit_defer_scope();
    active_stack_.pop_back();
    scope_open_ = false;
    execution_owner_thread_ = std::thread::id{};
    RpuExecutionCoordinator::exit_graph(
        *execution_claim_, "RpuKernelGraph::abort");
}

// =============================================================================
// invalidate
// =============================================================================

void RpuKernelGraph::invalidate() {
    const uint64_t expected_physical_token =
        bound_physical_execution_token_for_graph(graph_lifetime_id_);
    RpuExecutionGraphInvalidationGuard cleanup(
        "RpuKernelGraph::invalidate",
        expected_physical_token);
    invalidate_impl(/*from_abort=*/false);
}

void RpuKernelGraph::invalidate_impl(bool from_abort) {
    TORCH_CHECK(from_abort || !scope_open_,
                "RpuKernelGraph: invalidate is forbidden while a Graph scope "
                "is open; use abort for scope cleanup");
    TORCH_CHECK(!canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active,
                "RpuKernelGraph: invalidate is forbidden inside an active "
                "canonical FMB DMA callback/burst");
    kernels_.clear();
    nodes_.clear();
    segments_.clear();
    // invalidate 是硬重置:录制状态没了,keep_alive 的引用也就没有持有的理由。
    // 保持三条生命周期入口(begin_impl PASSTHROUGH / end / invalidate←abort)
    // 对称,免得下一个 keep_alive 生产方继承一个静默的滞留窗口。
    tensor_refs_.clear();
    boundary_flush_ptrs_.clear();
    // segments_ 被清,所有 seg idx 失效;private_queue_ 的 kd_buf 仍存
    // 但内容不再对应任何 segment (即将被下一轮 prepare_segment_queue 覆写)。
    // 重置 fingerprint 强制 fallback rebuild,防 stale data 被 sync-only 误读。
    private_queue_built_segment_idx_ = -1;
    // invalidate 是硬重置;两张 admission 表都清。persistent 资源由 owner
    // tree 管理；先遍历 owner tree mark_released 再 clear,跟 begin 收尾对齐。
    for (auto& r : host_callback_persistent_owners_) r.mark_released();
    host_callback_persistent_owners_.clear();
    for (auto& r : host_callback_zero_copy_holders_) r.mark_released();
    host_callback_zero_copy_holders_.clear();
    host_callback_live_tier1_out_ranges_.clear();
    // 同 begin(): 一并释放 pool 持有的 LocalSPM_t,清反查表。
    // abort() 也走这条路径,所以两条入口共用一份生命周期清理。
    local_spm_pool_.clear();
    global_spm_pool_.clear();
    // invalidate 也是硬重置,清 RECORDING dep 推断映射。
    recording_dev_addr_writer_.clear();
    // invalidate 硬清 Tier3 transient buffer owner tree(同 begin)。
    for (auto& r : tier3_oneshot_transient_owners_) r.mark_released();
    tier3_oneshot_transient_owners_.clear();
    tier3_oneshot_transient_tensors_.clear();
    last_stats_ = {};
    replayable_ = false;
    non_replayable_reason_.clear();   // A6 — 一并清掉(下次 begin 会重新设)
    built_signature_ = {};
    has_built_signature_ = false;
    pending_signature_.reset();
    semantic_dma_owner_generation_.reset();
    semantic_dma_expected_owner_generation_ = 0;
    has_semantic_dma_nodes_ = false;
    semantic_dma_replay_owner_armed_ = false;
    has_semantic_spm_peer_nodes_ = false;
    semantic_spm_peer_replay_armed_ = false;
    pending_semantic_spm_producer_yield_.reset();
    semantic_spm_producer_yield_owner_generation_.reset();
    semantic_spm_producer_yield_expected_owner_generation_ = 0;
    semantic_spm_producer_yield_id_digest_ = 0;
    semantic_spm_producer_yield_replay_arm_id_digest_ = 0;
    has_semantic_spm_producer_yield_nodes_ = false;
    semantic_spm_producer_yield_replay_armed_ = false;
    semantic_spm_producer_yield_poisoned_ = false;
    composite_fmb_invocation_ = {};
    cancel_canonical_dma_callback_for_fmb();
    canonical_dma_poisoned_ = false;
    canonical_dma_build_trace_active_ = false;
    canonical_dma_build_trace_complete_ = false;
    canonical_dma_build_trace_profile_hash_ = 0;
    canonical_dma_build_trace_node_begin_ = 0;
    canonical_dma_build_trace_node_end_ = 0;
    op_stream_fully_skipped_ = false;
    cursor_ = 0;
    build_topology_hash_ = 0;
    topology_hash_requested_ = false;
    // post_fn fast-replay: clear so a stale post_fn-start cursor can't
    // leak across rebuilds (next RECORDING re-records it via mark_post_fn_cursor).
    has_post_fn_cursor_ = false;
    replay_child_skip_ = ReplayChildSkipState{};
    clear_kernel_register_build_state();
    state_ = State::PASSTHROUGH;
    recording_physical_execution_token_ = 0;
    built_physical_execution_token_ = 0;
    retained_physical_build_requested_ = false;
    // Publish retirement eligibility only after every retained Graph field and
    // prepared resource above has been invalidated.  The external guard still
    // blocks physical release until its explicit retire() follows.
    mark_retained_physical_graph_invalidated(graph_lifetime_id_);
}

// =============================================================================
// mark_non_replayable(reason) — Tier 3 / unsafe-copy 等场景标 scope 不可缓存
// =============================================================================
//
// 不调 sync_point()(保留 nodes_ 顺序),不动 cursor_,只翻 replayable_=false +
// 写入 reason。end() 看到 !replayable_ 走 oneshot 路径,nodes_ 内的 HostCallback
// Tier 3 节点在 oneshot 内按录制顺序穿插真跑,RPU 段不退化为逐 op launch。
// 多次调用时只保留第一个 reason(避免后面 op 把更早的根因覆盖掉)。

void RpuKernelGraph::mark_non_replayable(std::string reason) {
    TORCH_CHECK(!pending_semantic_spm_producer_yield_.has_value() &&
                    !has_semantic_spm_producer_yield_nodes_ &&
                    !semantic_spm_producer_yield_poisoned_,
                "RpuKernelGraph: mark_non_replayable is forbidden by semantic "
                "SPM producer-yield authority");
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !has_kernel_register_census_nodes_ &&
                    !pending_kernel_register_census_.has_value(),
                "RpuKernelGraph: mark_non_replayable is forbidden by the "
                "typed DDR-register census");
    TORCH_CHECK(!canonical_dma_poisoned_ &&
                    !canonical_dma_build_trace_active_ &&
                    !canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active,
                "RpuKernelGraph: mark_non_replayable is forbidden inside a "
                "poisoned or active canonical FMB DMA scope");
    replayable_ = false;
    if (non_replayable_reason_.empty()) {
        non_replayable_reason_ = std::move(reason);
    }
}

// =============================================================================
// record_boundary_flush — 登记 Op 代码的 rpu_ddr_flush(ptr) 请求到 boundary
// =============================================================================
// RECORDING/REPLAYING 状态下由 HostCallback / Tier3 的 stable-buffer 落地点
// (graph_runtime_capture.cpp / graph_runtime_execute.cpp) 调用。不进 nodes_，
// 不影响 replay cursor，仅把 ptr 累积到 boundary_flush_ptrs_，由生命周期钩子
// （batch 前后、sync_point 前后）统一下发。
// 当前不做去重 —— 128 个调用点 × 单层 ~50 flush，重复 flush 单个 dc civac
// 成本可忽略；若 profiling 显示这里成为热点再在登记侧去重。

void RpuKernelGraph::record_boundary_flush(void* ptr) {
    if (ptr == nullptr) return;
    check_kernel_register_node_emission("boundary DDR flush registration");
    boundary_flush_ptrs_.push_back(ptr);
}

// =============================================================================
// flush_boundary_ptrs — 按顺序下发所有登记的 boundary ptr（in-place 去重）
// =============================================================================
// 遵循 g_rpu_ddr_flush_enabled：关闭时直接返回，不动集合，便于对照测试。
// 不 clear 集合 —— clear 由调用方的生命周期钩子在合适时机做，
// 通常是「kernel batch 后」+「sync_point 后」+「end/abort/invalidate 收尾」。
//
// 去重：同一 ptr 可能被多个 Op 重复登记（例如 QKV 共享的 mask ptr）。
// in-place sort + unique 让后续 flush 只处理唯一地址；第二次
// （post-batch）flush 复用已去重的集合。

void RpuKernelGraph::flush_boundary_ptrs() {
    if (!g_rpu_ddr_flush_enabled) {
        last_stats_.boundary_flush_count = 0;
        return;
    }
    if (boundary_flush_ptrs_.empty()) {
        last_stats_.boundary_flush_count = 0;
        return;
    }

    std::sort(boundary_flush_ptrs_.begin(), boundary_flush_ptrs_.end());
    boundary_flush_ptrs_.erase(
        std::unique(boundary_flush_ptrs_.begin(), boundary_flush_ptrs_.end()),
        boundary_flush_ptrs_.end());

    for (void* p : boundary_flush_ptrs_) {
        if (p != nullptr) {
            rhino_lkn::RpuDdrFlush(p);
        }
    }
    last_stats_.boundary_flush_count = boundary_flush_ptrs_.size();
}
// _to_copy admission: HostCallback stable-storage tracking
// =============================================================================
//
// rpu_to_copy 决定要不要把 RPU→RPU dtype cast 推进 host_callback Tier2 deferred
// 时的关键判据:src 数据来源是不是 HostCallback Tier1/2 stable buffer。命中 →
// defer(上游真值要等 executor 才刷,inline 会读 stale);否则 → inline。
//
// 第一版按 storage identity 记 [base, base+nbytes) 区间。这覆盖 view/slice 下
// 游路径(stable.view(...).to(fp16) / stable[i:j].to(fp16) 仍能命中),裸
// data_ptr() 精确比对会漏。lookup 是 O(N),N = 每图 host_callback Tier1/2 节
// 点数(典型 < 10),线性扫开销可忽略。

// 内部 helper:抽 (host_base, dev_base, nbytes)。dev_base 通过
// rhino_lkn::RpuGetDevAddr 查;查不到回退 0(命中规则会跳过 dev_base==0 那套)。
namespace {
struct StableRangeView {
    uintptr_t host_base;
    uintptr_t dev_base;
    size_t nbytes;
};
StableRangeView extract_range(const at::Tensor& t) {
    StableRangeView r{0, 0, 0};
    if (!t.defined()) return r;
    void* host_ptr = t.data_ptr();
    if (host_ptr == nullptr) return r;
    const size_t nbytes_t = t.numel() * t.element_size();
    if (nbytes_t == 0) return r;
    r.host_base = reinterpret_cast<uintptr_t>(host_ptr);
    r.nbytes    = nbytes_t;
    // RpuGetDevAddr 对非 RPU tensor 会返回 0/未定义;只 best-effort 拿。
    if (t.device().type() == c10::DeviceType::PrivateUse1) {
        const uint64_t dev = rhino_lkn::RpuGetDevAddr(host_ptr);
        r.dev_base = static_cast<uintptr_t>(dev);
    }
    return r;
}

bool to_copy_log_enabled() {
    static const bool v = []() {
        const char* e = std::getenv("RPU_GRAPH_TO_COPY_LOG");
        return e && *e && *e != '0';
    }();
    return v;
}
}  // namespace

// 内部 helper:同 (host_base, nbytes) 已存在则跳过。
namespace {
void push_unique_range(std::vector<RpuKernelGraph::HostCallbackStableRange>& vec,
                       const StableRangeView& v) {
    for (const auto& r : vec) {
        if (r.host_base == v.host_base && r.nbytes == v.nbytes) return;
    }
    vec.push_back({v.host_base, v.dev_base, v.nbytes});
}
bool range_hit(const StableRangeView& v,
               const std::vector<RpuKernelGraph::HostCallbackStableRange>& vec) {
    for (const auto& r : vec) {
        const bool host_hit =
            r.host_base != 0 &&
            v.host_base >= r.host_base &&
            v.host_base < r.host_base + r.nbytes;
        const bool dev_hit =
            r.dev_base != 0 && v.dev_base != 0 &&
            v.dev_base >= r.dev_base &&
            v.dev_base < r.dev_base + r.nbytes;
        if (host_hit || dev_hit) return true;
    }
    return false;
}

// G4.B-2 — owner tree 版本的 dedup / lookup。
// dedup 仍按 (host_base, nbytes) 二元组(沿用 push_unique_range 语义);命中
// 走 GraphOwnedResource::contains_host_addr / contains_dev_addr 点命中,跟
// 原 range_hit 1-byte 起始落在 owner 范围内的判定等价。
void push_unique_owner(std::vector<GraphOwnedResource>& vec,
                       const StableRangeView& v) {
    const uintptr_t v_host = v.host_base;
    for (const auto& r : vec) {
        if (reinterpret_cast<uintptr_t>(r.ptr) == v_host && r.nbytes == v.nbytes) {
            return;
        }
    }
    vec.emplace_back(ResourceKind::StableOutput,
                     kInvalidNodeId,
                     reinterpret_cast<void*>(v_host),
                     /*dev=*/static_cast<uint64_t>(v.dev_base),
                     /*nbytes=*/v.nbytes);
}
bool owner_range_hit(const StableRangeView& v,
                     const std::vector<GraphOwnedResource>& vec) {
    for (const auto& r : vec) {
        if (r.contains_host_addr(v.host_base)) return true;
        if (v.dev_base != 0 && r.contains_dev_addr(static_cast<uint64_t>(v.dev_base))) {
            return true;
        }
    }
    return false;
}
}  // namespace

void RpuKernelGraph::register_host_callback_persistent_stable(
    const at::Tensor& stable) {
    StableRangeView v = extract_range(stable);
    if (v.host_base == 0 || v.nbytes == 0) return;
    push_unique_owner(host_callback_persistent_owners_, v);
    if (to_copy_log_enabled()) {
        std::cerr << "[STABLE-REG persist] host=0x" << std::hex << v.host_base
                  << " dev=0x" << v.dev_base << std::dec
                  << " nbytes=" << v.nbytes
                  << " dtype=" << stable.scalar_type()
                  << " sizes=" << stable.sizes() << "\n";
    }
}

void RpuKernelGraph::register_host_callback_live_tier1_out(
    const at::Tensor& fresh_out_arg) {
    StableRangeView v = extract_range(fresh_out_arg);
    if (v.host_base == 0 || v.nbytes == 0) return;
    push_unique_range(host_callback_live_tier1_out_ranges_, v);
    if (to_copy_log_enabled()) {
        std::cerr << "[STABLE-REG live] host=0x" << std::hex << v.host_base
                  << " dev=0x" << v.dev_base << std::dec
                  << " nbytes=" << v.nbytes
                  << " dtype=" << fresh_out_arg.scalar_type()
                  << " sizes=" << fresh_out_arg.sizes() << "\n";
    }
}

bool RpuKernelGraph::is_host_callback_stable_input(const at::Tensor& src) const {
    if (host_callback_persistent_owners_.empty() &&
        host_callback_live_tier1_out_ranges_.empty()) return false;
    StableRangeView v = extract_range(src);
    if (v.host_base == 0) return false;
    // 命中规则:src 的 host_base 落在某 range 内,**或** src 的 dev_base
    // (若可查)落在该 range 内。先查 persistent(owner tree StableOutput)
    // 再查 live(Tier1 .out fresh args,本期未迁 owner tree)。两表语义不同
    // 但命中条件一致。
    return owner_range_hit(v, host_callback_persistent_owners_) ||
           range_hit(v, host_callback_live_tier1_out_ranges_);
}

bool RpuKernelGraph::should_defer_host_op_input(const at::Tensor& src) const {
    enum class Mode { Auto, ForceOff, ForceOn };
    static const Mode mode = []() {
        // 优先读升级名 RPU_GRAPH_HOST_OP_DEFER_GATE;fall back 到 legacy alias
        // RPU_GRAPH_DEFER_TO_COPY(早先 _to_copy 单点 hack 的 env 名,G2.B
        // unified gate 起跟新名同义,保留兼容诊断脚本不破)。
        const char* e = std::getenv("RPU_GRAPH_HOST_OP_DEFER_GATE");
        if (e == nullptr || *e == '\0')
            e = std::getenv("RPU_GRAPH_DEFER_TO_COPY");
        if (e == nullptr || *e == '\0') return Mode::Auto;
        std::string v(e);
        if (v == "auto") return Mode::Auto;
        if (v == "0" || v == "off" || v == "false") return Mode::ForceOff;
        return Mode::ForceOn;
    }();
    if (mode == Mode::ForceOff) return false;
    if (state_ != State::RECORDING && state_ != State::REPLAYING) return false;
    if (mode == Mode::ForceOn) return true;
    return is_host_callback_stable_input(src);
}

// =============================================================================
// G4.B-4 — ZeroCopy storage holder 镜像注册
// =============================================================================

void RpuKernelGraph::register_zero_copy_holder(const at::Tensor& view) {
    if (!view.defined()) return;
    void* host_ptr = nullptr;
    size_t bytes = 0;
    if (view.has_storage()) {
        // Storage::data() 返 const void*;owner tree 把 ptr 当 opaque key,
        // 不写不读内容,const_cast 跟 dev_addr/uintptr 化 lookup 路径一致。
        host_ptr = const_cast<void*>(view.storage().data());
        bytes = view.storage().nbytes();
    }
    host_callback_zero_copy_holders_.emplace_back(
        ResourceKind::ZeroCopyStorageHolder,
        kInvalidNodeId,
        host_ptr,
        /*dev=*/0,
        bytes);
}

size_t RpuKernelGraph::zero_copy_holder_count() const {
    return host_callback_zero_copy_holders_.size();
}

void RpuKernelGraph::truncate_zero_copy_holders(size_t to_size) {
    if (to_size <= host_callback_zero_copy_holders_.size()) {
        // resize 前先 mark_released,保持 owner tree 的状态迁移语义。
        for (size_t i = to_size; i < host_callback_zero_copy_holders_.size(); ++i) {
            host_callback_zero_copy_holders_[i].mark_released();
        }
        host_callback_zero_copy_holders_.resize(to_size);
    }
}

// =============================================================================
// Tier3 transient buffer owner tree 注册 / 反查
// =============================================================================

size_t RpuKernelGraph::register_tier3_transient_buffer(
    int owner_node_id, const at::Tensor& buffer, uint64_t dev_addr) {
    TORCH_CHECK(buffer.defined(),
                "register_tier3_transient_buffer: buffer not defined");
    TORCH_CHECK(buffer.device().type() == c10::DeviceType::PrivateUse1,
                "register_tier3_transient_buffer: buffer must be on RPU; got ",
                buffer.device());
    TORCH_CHECK(tier3_oneshot_transient_owners_.size() ==
                tier3_oneshot_transient_tensors_.size(),
                "register_tier3_transient_buffer: owner/tensor vectors out of "
                "sync (",
                tier3_oneshot_transient_owners_.size(), " vs ",
                tier3_oneshot_transient_tensors_.size(), ")");
    void* host_ptr = buffer.data_ptr();
    const size_t nbytes = static_cast<size_t>(buffer.numel()) *
                          static_cast<size_t>(buffer.element_size());
    const size_t idx = tier3_oneshot_transient_owners_.size();
    tier3_oneshot_transient_owners_.emplace_back(
        ResourceKind::TransientTier3Buffer,
        owner_node_id,
        host_ptr,
        dev_addr,
        nbytes);
    tier3_oneshot_transient_tensors_.push_back(buffer);
    return idx;
}

const at::Tensor& RpuKernelGraph::tier3_transient_tensor_at(size_t idx) const {
    TORCH_CHECK(idx < tier3_oneshot_transient_tensors_.size(),
                "tier3_transient_tensor_at: idx ", idx, " out of range (size=",
                tier3_oneshot_transient_tensors_.size(), ")");
    return tier3_oneshot_transient_tensors_[idx];
}

// =============================================================================
// keep_alive
// =============================================================================

void RpuKernelGraph::keep_alive(const at::Tensor& t) {
    if (state_ == State::RECORDING || state_ == State::REPLAYING) {
        tensor_refs_.push_back(t);
    }
}

// =============================================================================
// RECORDING 期 dev_addr → writer node 推断 helper
// =============================================================================

void RpuKernelGraph::track_recording_writer_dev_addr(uint64_t dev_addr,
                                                      int node_id) {
    if (state_ != State::RECORDING) return;
    if (dev_addr == 0) return;
    if (node_id < 0) return;
    // 同 dev_addr 后写覆盖前者:caching allocator 复用 ptr 给新节点 alloc 时,
    // 老 writer 已对当前 forward 失效。Tier3Oneshot 节点本身的 last_output_dev
    // 也走这条覆盖语义(每次 adopt_returns 都重新填)。
    recording_dev_addr_writer_[dev_addr] = node_id;
}

std::vector<int> RpuKernelGraph::resolve_recording_input_deps(
    int consumer_node_id, const std::vector<uint64_t>& input_dev_addrs) {
    std::vector<int> deps;
    if (state_ != State::RECORDING) return deps;
    if (consumer_node_id < 0) return deps;

    std::vector<int> seen;  // 去重小 vector(deps 个数预期 <= 4 量级)
    for (uint64_t dev : input_dev_addrs) {
        if (dev == 0) continue;
        auto it = recording_dev_addr_writer_.find(dev);
        if (it == recording_dev_addr_writer_.end()) continue;
        int writer_id = it->second;
        if (writer_id < 0 ||
            writer_id >= static_cast<int>(nodes_.size())) continue;
        // 自循环排除(同节点既写又读自己,理论上不会发生)
        if (writer_id == consumer_node_id) continue;
        if (std::find(seen.begin(), seen.end(), writer_id) != seen.end()) {
            continue;
        }
        seen.push_back(writer_id);
        deps.push_back(writer_id);
        // 反向回填:上游若是 Tier3Oneshot,push consumer_id 到它的 output_dep
        auto& upstream = nodes_[writer_id];
        if (upstream.kind == GraphNodeKind::Tier3Oneshot) {
            auto& outd = upstream.as_tier3_oneshot().output_dep_node_ids;
            if (std::find(outd.begin(), outd.end(), consumer_node_id) ==
                outd.end()) {
                outd.push_back(consumer_node_id);
            }
        }
    }
    std::sort(deps.begin(), deps.end());
    return deps;
}
