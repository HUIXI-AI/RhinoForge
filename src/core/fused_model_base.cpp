// fused_model_base.cpp — flat single-class implementation
//
// FusedModelBase uses a PImpl to hide internal state.
// Framework features:
//   - ModelStaticConfig.preload_fn / kv_first_fn / post_fn are dispatched
//     via std::invoke inside run_all_layers.
//   - BufferDecl.preload_callback fires on persistent-generation advance or
//     preload_callbacks_dirty_.
//   - Two independent dirty flags — weights_dirty_ (preload_fn
//     emission) and preload_callbacks_dirty_ (per-buffer callback loop) — each
//     cleared at its own dispatch point.
//   - Preload callbacks use the graph-aware RpuQueue
//     proxy 进 RpuKernelGraph::active() (有 scope → 进 RECORDING/REPLAYING,
//     无 scope → 立即发射)。

#include "fused_model_base.h"
#include "fused_model_base_impl.h"

#include "rpu_spm_allocator.h"   // SPM_ALLOC, SpmAllocator::AllocRequest
#include "rpu_spm_pipeline.h"
#include "rpu_lingbot2_ring_collective.h"
#include "rpu_kernel_decls.h"    // rpu_launch_*, PreparedMask, sdpa_prepare_mask, sdpa_dma_mask_to_spm
#include "rpu_profile.h"         // g_rpu_debug_level / log_at, rpu_ddr_flush
#include "rpu_helpers.h"         // ValuOpType
#include "rhino_launch_buffer.h" // RpuGetDevAddr (mutable-DMA live addr)
#include "graph/graph_runtime.h" // RpuKernelGraph (keep_alive, has_active)

#include <ATen/ATen.h>
#include <c10/util/ScopeExit.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace v3 {

namespace {

bool env_enabled(const char* name, bool reject_false_prefix) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0' || value[0] == '0') return false;
    return !reject_false_prefix ||
        (value[0] != 'f' && value[0] != 'F');
}

constexpr uint64_t kFmbFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFmbFnvPrime = 0x100000001b3ULL;
constexpr GraphNodeKindMask kSpmFmbOpaqueGraphNodeKinds =
    graph_node_kind_bit(GraphNodeKind::ChildGraph) |
    graph_node_kind_bit(GraphNodeKind::HostCallback) |
    graph_node_kind_bit(GraphNodeKind::Tier3Oneshot);

uint64_t fmb_hash_u64(uint64_t hash, uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
        hash = (hash ^ static_cast<uint8_t>(value >> shift)) * kFmbFnvPrime;
    }
    return hash;
}

uint64_t fmb_hash_string(uint64_t hash, const std::string& value) {
    hash = fmb_hash_u64(hash, value.size());
    for (unsigned char byte : value) {
        hash = (hash ^ byte) * kFmbFnvPrime;
    }
    return hash;
}

uint64_t fmb_nonzero_hash(uint64_t hash) {
    return hash == 0 ? 0x9e3779b97f4a7c15ULL : hash;
}

uint64_t fmb_composite_occurrence_profile_hash(
    uint64_t legacy_profile_hash,
    uint64_t layout_hash,
    uint64_t allocation_hash,
    uint64_t declaration_hash,
    uint64_t persistent_hash,
    SpmFmbPreloadCapability preload_capability,
    SpmFmbCompositeOccurrenceCapability composite_capability,
    uint64_t composite_policy_fingerprint) {
    if (composite_capability ==
        SpmFmbCompositeOccurrenceCapability::Unsealed) {
        return 0;
    }
    uint64_t hash = fmb_hash_u64(
        kFmbFnvOffset, UINT64_C(0x434f4d5052463031));
    hash = fmb_hash_u64(hash, legacy_profile_hash);
    hash = fmb_hash_u64(hash, layout_hash);
    hash = fmb_hash_u64(hash, allocation_hash);
    hash = fmb_hash_u64(hash, declaration_hash);
    hash = fmb_hash_u64(hash, persistent_hash);
    hash = fmb_hash_u64(
        hash, static_cast<uint8_t>(preload_capability));
    hash = fmb_hash_u64(
        hash, static_cast<uint8_t>(composite_capability));
    hash = fmb_hash_u64(hash, composite_policy_fingerprint);
    return fmb_nonzero_hash(hash);
}

struct DenseDmaEndpointDescriptor {
    uint64_t semantic_id = 0;
    SpmFmbDmaEndpointRole role = SpmFmbDmaEndpointRole::Ingress;
    SpmFmbDdrArenaRole arena = SpmFmbDdrArenaRole::LiveInput;
    SpmFmbDdrAccess access = SpmFmbDdrAccess::Read;
    SpmFmbDdrAddressMode address_mode =
        SpmFmbDdrAddressMode::Stable;
    uint8_t expected_dma_variant = 0;
    uint8_t expected_occurrence_count = 0;
    size_t byte_begin = 0;
    size_t byte_count = 0;
};

DenseDmaEndpointDescriptor make_dense_dma_endpoint_descriptor(
        uint64_t profile_hash,
        uint64_t allocation_hash,
        uint32_t callback_key,
        uint32_t body_id,
        uint32_t framework_group_id,
        int layer,
        size_t member_ordinal,
        size_t member_count,
        int64_t packed_row_begin,
        int64_t row_count,
        int64_t local_position_begin,
        int64_t local_kv_seq_len,
        SpmFmbDmaEndpointRole role,
        SpmFmbDdrArenaRole arena,
        SpmFmbDdrAccess access,
        SpmFmbDdrAddressMode address_mode,
        uint8_t expected_occurrence_count,
        size_t byte_begin,
        size_t byte_count) {
    TORCH_CHECK(profile_hash != 0 && allocation_hash != 0 &&
                    callback_key != 0 && layer >= 0 &&
                    member_count >= 2 && member_ordinal < member_count &&
                    packed_row_begin >= 0 && row_count > 0 &&
                    local_position_begin >= 0 && local_kv_seq_len > 0 &&
                    byte_count > 0 && byte_count % 16 == 0 &&
                    expected_occurrence_count >= 1 &&
                    expected_occurrence_count <= 8 &&
                    (role == SpmFmbDmaEndpointRole::Ingress ||
                     expected_occurrence_count == 1),
                "schema-v11 DMA endpoint: invalid canonical descriptor");
    uint8_t expected_dma_variant =
        static_cast<uint8_t>(DmaNodeData::Variant::Fixed);
    if (address_mode == SpmFmbDdrAddressMode::MutableBase) {
        expected_dma_variant = static_cast<uint8_t>(
            role == SpmFmbDmaEndpointRole::Ingress
                ? DmaNodeData::Variant::MutableSrc
                : DmaNodeData::Variant::MutableDst);
    }

    uint64_t hash = fmb_hash_u64(kFmbFnvOffset, 18);
    hash = fmb_hash_u64(hash, profile_hash);
    hash = fmb_hash_u64(hash, allocation_hash);
    hash = fmb_hash_u64(hash, callback_key);
    hash = fmb_hash_u64(hash, body_id);
    hash = fmb_hash_u64(hash, framework_group_id);
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(layer));
    hash = fmb_hash_u64(hash, member_ordinal);
    hash = fmb_hash_u64(hash, member_count);
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(packed_row_begin));
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(row_count));
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(local_position_begin));
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(local_kv_seq_len));
    hash = fmb_hash_u64(hash, static_cast<uint8_t>(role));
    hash = fmb_hash_u64(hash, static_cast<uint8_t>(arena));
    hash = fmb_hash_u64(hash, static_cast<uint8_t>(access));
    hash = fmb_hash_u64(hash, static_cast<uint8_t>(address_mode));
    hash = fmb_hash_u64(hash, expected_dma_variant);
    hash = fmb_hash_u64(hash, expected_occurrence_count);
    hash = fmb_hash_u64(hash, byte_begin);
    hash = fmb_hash_u64(hash, byte_count);
    return DenseDmaEndpointDescriptor{
        fmb_nonzero_hash(hash), role, arena, access, address_mode,
        expected_dma_variant, expected_occurrence_count,
        byte_begin, byte_count};
}

enum class BuildTraceWindowHashMode : uint8_t {
    Absolute = 1,
    Relative = 2,
};

struct HashedBuildTraceCallback {
    uint64_t hash = 0;
    size_t access_count = 0;
};

HashedBuildTraceCallback hash_build_trace_callback(
    uint64_t hash,
    const FusedModelBase::Impl::BuildTraceCallbackRecord& callback,
    BuildTraceWindowHashMode window_mode) {
    hash = fmb_hash_u64(hash, callback.key);
    hash = fmb_hash_u64(hash, static_cast<uint8_t>(callback.kind));
    hash = fmb_hash_u64(hash, callback.body_id);
    hash = fmb_hash_u64(hash, callback.group_id);
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(callback.layer));
    hash = fmb_hash_u64(hash, callback.traversal_ordinal);
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(callback.chunk_index));
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(callback.chunk_offset));
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(callback.chunk_len));
    hash = fmb_hash_u64(hash, static_cast<uint64_t>(callback.chunk_kv_seq_len));
    hash = fmb_hash_u64(hash, callback.scope_mask);
    if (window_mode == BuildTraceWindowHashMode::Absolute) {
        // Preserve the frozen schema-v8 callback byte stream exactly.  The
        // profile hash already binds IO actions and phase chronology.
        hash = fmb_hash_u64(hash, callback.graph_node_begin);
        hash = fmb_hash_u64(hash, callback.graph_node_end);
    } else {
        TORCH_INTERNAL_ASSERT(
            window_mode == BuildTraceWindowHashMode::Relative &&
            callback.graph_node_begin <= callback.graph_node_end);
        hash = fmb_hash_u64(hash, callback.updates_io_context);
        hash = fmb_hash_u64(hash, callback.input_in_spm);
        hash = fmb_hash_u64(hash, callback.output_to_spm);
        hash = fmb_hash_u64(
            hash, static_cast<uint64_t>(callback.local_phase_begin));
        hash = fmb_hash_u64(
            hash, static_cast<uint64_t>(callback.local_phase_end));
        hash = fmb_hash_u64(hash, callback.global_origin);
        hash = fmb_hash_u64(
            hash, callback.graph_node_end - callback.graph_node_begin);
    }
    hash = fmb_hash_u64(hash, callback.graph_node_kind_mask);
    hash = fmb_hash_u64(hash, callback.graph_topology_hash);
    hash = fmb_hash_u64(hash, callback.accesses.size());

    size_t access_count = 0;
    for (const auto& access : callback.accesses) {
        TORCH_CHECK(access.resolve_count <=
                        std::numeric_limits<size_t>::max() - access_count,
                    "schema-v8 BUILD trace seal: callback access count "
                    "overflow");
        access_count += access.resolve_count;
        hash = fmb_hash_u64(hash, access.allocation_ordinal);
        hash = fmb_hash_u64(hash, access.declaration_ordinal);
        hash = fmb_hash_u64(hash, access.alias_root_ordinal);
        hash = fmb_hash_u64(hash, static_cast<uint8_t>(access.storage));
        hash = fmb_hash_u64(hash, static_cast<uint8_t>(access.scope));
        hash = fmb_hash_u64(hash, static_cast<uint64_t>(access.layer));
        hash = fmb_hash_u64(hash, static_cast<uint8_t>(access.accessor));
        hash = fmb_hash_u64(hash, access.core_mask);
        hash = fmb_hash_u64(hash, access.root_offset);
        hash = fmb_hash_u64(hash, access.protected_bytes);
        hash = fmb_hash_u64(hash, access.resolve_count);
    }
    if (!callback.producer_yields.empty()) {
        hash = fmb_hash_u64(hash, 28);
        hash = fmb_hash_u64(hash, callback.producer_yields.size());
        for (const auto& yield : callback.producer_yields) {
            hash = fmb_hash_u64(hash, yield.ordinal);
            hash = fmb_hash_u64(hash, yield.semantic_id);
            hash = fmb_hash_u64(
                hash, static_cast<uint8_t>(yield.writer_kind));
            hash = fmb_hash_u64(
                hash, static_cast<uint8_t>(yield.spec.dtype));
            hash = fmb_hash_u64(
                hash, static_cast<uint64_t>(yield.spec.rows));
            hash = fmb_hash_u64(
                hash, static_cast<uint64_t>(yield.spec.cols));
            hash = fmb_hash_u64(
                hash, static_cast<uint8_t>(yield.spec.distribution));
            hash = fmb_hash_u64(hash, yield.allocation_ordinal);
            hash = fmb_hash_u64(hash, yield.declaration_ordinal);
            hash = fmb_hash_u64(hash, yield.alias_root_ordinal);
            hash = fmb_hash_u64(hash, yield.declaration_name_hash);
            hash = fmb_hash_u64(hash, yield.alias_root_name_hash);
            hash = fmb_hash_u64(hash, yield.root_offset);
            hash = fmb_hash_u64(
                hash, yield.allocation_offset_from_root);
            hash = fmb_hash_u64(hash, yield.logical_capacity);
            hash = fmb_hash_u64(hash, yield.aligned_capacity);
            if (window_mode == BuildTraceWindowHashMode::Absolute) {
                hash = fmb_hash_u64(hash, yield.graph_node_begin);
                hash = fmb_hash_u64(hash, yield.graph_node_end);
            } else {
                hash = fmb_hash_u64(
                    hash,
                    yield.graph_node_begin - callback.graph_node_begin);
                hash = fmb_hash_u64(
                    hash,
                    yield.graph_node_end - yield.graph_node_begin);
            }
            hash = fmb_hash_u64(hash, yield.graph_node_kind_mask);
            hash = fmb_hash_u64(hash, yield.graph_topology_hash);
        }
    }
    return {hash, access_count};
}

void advance_pipeline_manifest_generation(FusedModelBase::Impl& state) {
    TORCH_CHECK(state.pipeline_manifest_generation_ != nullptr &&
                    *state.pipeline_manifest_generation_ !=
                        std::numeric_limits<uint64_t>::max(),
                "FusedModelBase: pipeline manifest generation exhausted");
    ++*state.pipeline_manifest_generation_;
}

void retire_pipeline_manifest(FusedModelBase::Impl& state) {
    TORCH_CHECK(!state.pipeline_build_trace_.armed,
                "FusedModelBase: cancel the active BUILD access trace before "
                "retiring its allocation manifest");
    TORCH_CHECK(!state.pipeline_runtime_member_frame_.active,
                "FusedModelBase: cannot retire a manifest inside a schema-v12 "
                "runtime member callback");
    TORCH_CHECK(
        !state.composite_physical_execution_.armed,
        "FusedModelBase: finish or cancel the composite physical occurrence "
        "before retiring its allocation manifest");
    advance_pipeline_manifest_generation(state);
    state.pipeline_manifest_snapshot_ = {};
    state.pipeline_occurrence_schedule_present_ = false;
    state.pipeline_occurrence_schedule_ = {};
    state.pipeline_lease_epoch_ = 0;
    state.pipeline_plan_hash_ = 0;
    state.pipeline_layout_hash_ = 0;
    state.pipeline_temporary_bytes_ = 0;
    state.pipeline_scratch_base_ = 0;
    state.pipeline_temporary_names_.clear();
    state.pipeline_temporary_per_layer_names_.clear();
    state.pipeline_runtime_replay_authority_ = {};
    state.pipeline_post_fn_yield_replay_authority_ = {};
    state.pipeline_callback_yield_replay_authority_ = {};
    state.composite_physical_execution_ = {};
}

thread_local FusedModelBase::Impl* g_fmb_build_trace_owner = nullptr;
thread_local std::optional<GraphOpStreamStamp>
    g_fmb_build_trace_callback_begin;
thread_local std::optional<GraphOpStreamStamp>
    g_fmb_post_fn_yield_begin;
thread_local FusedModelBase::Impl* g_fmb_runtime_member_owner = nullptr;
thread_local FusedModelBase::Impl* g_fmb_cpu_yield_replay_owner = nullptr;

void clear_pipeline_build_trace_noexcept(
    FusedModelBase::Impl& state) noexcept {
    if (g_fmb_build_trace_owner == &state) {
        g_fmb_build_trace_owner = nullptr;
        g_fmb_build_trace_callback_begin.reset();
        g_fmb_post_fn_yield_begin.reset();
    }
    state.pipeline_build_trace_ = {};
}

const FusedModelBase::Impl::PipelineAllocationEntry*
find_trace_allocation(
    const FusedModelBase::Impl::CachedLayoutSnapshot& snapshot,
    const char* name,
    int layer,
    uint32_t* ordinal) {
    auto find_in = [&](const std::vector<
                           FusedModelBase::Impl::PipelineAllocationEntry>&
                           allocations)
        -> const FusedModelBase::Impl::PipelineAllocationEntry* {
        for (size_t index = 0; index < allocations.size(); ++index) {
            const auto& allocation = allocations[index];
            if (allocation.name == name && allocation.layer == layer) {
                TORCH_CHECK(index <= std::numeric_limits<uint32_t>::max(),
                            "schema-v8 BUILD trace: allocation ordinal "
                            "exceeds uint32");
                *ordinal = static_cast<uint32_t>(index);
                return &allocation;
            }
        }
        return nullptr;
    };
    if (const auto* allocation =
            find_in(snapshot.temporary_allocations)) {
        return allocation;
    }
    const auto* allocation = find_in(snapshot.persistent_allocations);
    if (allocation != nullptr) {
        TORCH_CHECK(snapshot.temporary_allocations.size() <=
                        std::numeric_limits<uint32_t>::max() - *ordinal,
                    "schema-v8 BUILD trace: combined allocation ordinal "
                    "overflows uint32");
        *ordinal += static_cast<uint32_t>(
            snapshot.temporary_allocations.size());
    }
    return allocation;
}

bool resolve_cpu_yield_replay_offset(
    FusedModelBase::Impl& state,
    const char* name,
    int layer,
    uint32_t* offset) {
    if (g_fmb_cpu_yield_replay_owner != &state) return false;
    TORCH_CHECK(
        state.pipeline_manifest_snapshot_.valid &&
            state.pipeline_manifest_snapshot_.cpu_dry &&
            (state.pipeline_post_fn_yield_replay_authority_.armed ||
             state.pipeline_callback_yield_replay_authority_.armed),
        "CPU producer-yield replay access has no live dry authority");
    uint32_t allocation_ordinal = 0;
    const auto* allocation = find_trace_allocation(
        state.pipeline_manifest_snapshot_, name, layer,
        &allocation_ordinal);
    TORCH_CHECK(
        allocation != nullptr,
        "CPU producer-yield replay access has no owned allocation '",
        name, "' layer ", layer);
    *offset = allocation->offset;
    return true;
}

const FusedModelBase::Impl::OwnedPipelineDecl* find_trace_declaration(
    const FusedModelBase::Impl::CachedLayoutSnapshot& snapshot,
    const char* name,
    uint32_t* ordinal) {
    for (size_t index = 0; index < snapshot.decls.size(); ++index) {
        if (snapshot.decls[index].name == name) {
            TORCH_CHECK(index <= std::numeric_limits<uint32_t>::max(),
                        "schema-v8 BUILD trace: declaration ordinal exceeds "
                        "uint32");
            *ordinal = static_cast<uint32_t>(index);
            return &snapshot.decls[index];
        }
    }
    return nullptr;
}

uint32_t find_trace_root_ordinal(
    const FusedModelBase::Impl::CachedLayoutSnapshot& snapshot,
    const FusedModelBase::Impl::PipelineAllocationEntry& allocation) {
    uint32_t root_ordinal = 0;
    const auto* root = find_trace_allocation(
        snapshot, allocation.alias_root.c_str(), allocation.layer,
        &root_ordinal);
    TORCH_CHECK(root != nullptr,
                "schema-v8 BUILD trace: allocation '", allocation.name,
                "' has no sealed alias root '", allocation.alias_root, "'");
    return root_ordinal;
}

bool record_pipeline_build_access(
    FusedModelBase::Impl& state,
    const char* name,
    int layer,
    SpmFmbBuildTraceAccessorKind accessor,
    int core,
    uint32_t* cpu_dry_offset) {
    TORCH_CHECK(g_fmb_build_trace_owner != nullptr,
                "schema-v8 BUILD trace STOP: armed trace accessed from the "
                "wrong owner thread");
    TORCH_CHECK(g_fmb_build_trace_owner == &state,
                "schema-v8 BUILD trace STOP: an armed model observed an SPM "
                "access through a different FusedModelBase owner");
    auto& candidate = state.pipeline_build_trace_;
    TORCH_CHECK(candidate.armed,
                "schema-v8 BUILD trace STOP: TLS owner is not armed");
    TORCH_CHECK(!(candidate.persistent_preload_callback_open &&
                  candidate.callback_open),
                "schema-v8 BUILD trace STOP: persistent preload and "
                "resolved callback scopes overlap");
    const bool resolved_callback_open =
        candidate.callback_open && !candidate.callbacks.empty();
    const bool persistent_preload_callback_open =
        candidate.persistent_preload_callback_open &&
        !candidate.callback_open;
    TORCH_CHECK(resolved_callback_open ||
                    persistent_preload_callback_open,
                "schema-v8 BUILD trace STOP: unscoped FusedModelBase SPM "
                "access '", name, "'");
    TORCH_CHECK(name != nullptr && name[0] != '\0',
                "schema-v8 BUILD trace STOP: empty SPM access name");
    const bool absolute_accessor =
        accessor == SpmFmbBuildTraceAccessorKind::Addr ||
        accessor == SpmFmbBuildTraceAccessorKind::LayerAddr;
    if (absolute_accessor) {
        TORCH_CHECK(core >= 0 && core < SpmAllocator::NUM_CORES,
                    "schema-v8 BUILD trace STOP: core ", core,
                    " is outside [0,", SpmAllocator::NUM_CORES, ")");
    }

    const auto& snapshot = state.pipeline_manifest_snapshot_;
    TORCH_CHECK(snapshot.valid &&
                    snapshot.allocation_hash == candidate.allocation_hash,
                "schema-v8 BUILD trace STOP: allocation snapshot drift");
    uint32_t allocation_ordinal = 0;
    const auto* allocation = find_trace_allocation(
        snapshot, name, layer, &allocation_ordinal);
    TORCH_CHECK(allocation != nullptr,
                "schema-v8 BUILD trace STOP: SPM access '", name,
                "' layer ", layer,
                " is not owned by the sealed allocation table");
    uint32_t declaration_ordinal = 0;
    const auto* declaration = find_trace_declaration(
        snapshot, name, &declaration_ordinal);
    TORCH_INTERNAL_ASSERT(declaration != nullptr);

    if (persistent_preload_callback_open) {
        const bool persistent_allocation =
            allocation->storage == StorageClass::Persistent ||
            allocation->storage == StorageClass::PersistentPerLayer;
        TORCH_CHECK(
            candidate.preload_capability ==
                    SpmFmbPreloadCapability::
                        PersistentOutsideResolvedWindow &&
                candidate.next_step == 0 && candidate.callbacks.empty() &&
                persistent_allocation &&
                declaration->storage == allocation->storage,
            "schema-v8 BUILD trace STOP: persistent preload callback "
            "accessed non-persistent FusedModelBase SPM '", name, "'");
        if (cpu_dry_offset != nullptr) {
            *cpu_dry_offset = allocation->offset;
        }
        return candidate.cpu_dry;
    }

    auto& callback = candidate.callbacks.back();
    TORCH_CHECK((callback.scope_mask &
                 spm_fmb_scope_bit(declaration->scope)) != 0,
                "schema-v8 BUILD trace STOP: scope drift for '", name,
                "' in callback ", callback.key);
    if (allocation->storage == StorageClass::TempPerLayer ||
        allocation->storage == StorageClass::PersistentPerLayer) {
        TORCH_CHECK(callback.layer == allocation->layer,
                    "schema-v8 BUILD trace STOP: per-layer access '", name,
                    "' layer ", allocation->layer,
                    " does not match callback layer ", callback.layer);
    }

    const uint32_t root_ordinal =
        find_trace_root_ordinal(snapshot, *allocation);
    uint32_t root_lookup_ordinal = 0;
    const auto* root = find_trace_allocation(
        snapshot, allocation->alias_root.c_str(), allocation->layer,
        &root_lookup_ordinal);
    TORCH_INTERNAL_ASSERT(root != nullptr &&
                          root_lookup_ordinal == root_ordinal);
    const bool whole_root_synonym =
        !declaration->alias_of.empty() && declaration->logical_bytes == 0;
    const size_t protected_bytes =
        declaration->alias_of.empty() || whole_root_synonym
        ? root->aligned_bytes : allocation->aligned_bytes;
    const uint8_t core_mask =
        absolute_accessor
        ? static_cast<uint8_t>(uint8_t{1} << core)
        : static_cast<uint8_t>(0xff);

    for (auto& access : callback.accesses) {
        if (access.allocation_ordinal == allocation_ordinal &&
            access.declaration_ordinal == declaration_ordinal &&
            access.alias_root_ordinal == root_ordinal &&
            access.storage == allocation->storage &&
            access.scope == declaration->scope &&
            access.layer == allocation->layer &&
            access.accessor == accessor &&
            access.root_offset == root->offset &&
            access.protected_bytes == protected_bytes) {
            TORCH_CHECK(access.resolve_count !=
                            std::numeric_limits<uint32_t>::max(),
                        "schema-v8 BUILD trace: access count overflow");
            ++access.resolve_count;
            access.core_mask |= core_mask;
            if (cpu_dry_offset != nullptr) {
                *cpu_dry_offset = allocation->offset;
            }
            return candidate.cpu_dry;
        }
    }

    callback.accesses.push_back(
        {allocation_ordinal, declaration_ordinal, root_ordinal,
         allocation->storage, declaration->scope, allocation->layer,
         accessor, core_mask, root->offset, protected_bytes, 1});
    if (cpu_dry_offset != nullptr) {
        *cpu_dry_offset = allocation->offset;
    }
    return candidate.cpu_dry;
}

}  // namespace

struct SpmFmbRuntimeDenseDdrMemberRef::State {
    const FusedModelBase::Impl* owner = nullptr;
    uint64_t callback_epoch = 0;
    const void* graph_identity = nullptr;
    uint64_t graph_build_generation = 0;
    uint64_t profile_hash = 0;
    uint64_t allocation_hash = 0;
    uint64_t expected_owner_generation = 0;
    size_t layer_group_ordinal = 0;
    size_t member_ordinal = 0;
    size_t member_count = 0;
    size_t callback_index = 0;
    uint32_t callback_key = 0;
    uint32_t body_id = 0;
    uint32_t framework_group_id = 0;
    int layer = -1;
    int64_t packed_row_begin = 0;
    int64_t row_count = 0;
    int64_t local_position_begin = 0;
    int64_t local_kv_seq_len = 0;
    SpmFmbDdrArenaRole ingress_arena = SpmFmbDdrArenaRole::LiveInput;
    SpmFmbDdrAddressMode ingress_address_mode =
        SpmFmbDdrAddressMode::Stable;
    size_t ingress_byte_begin = 0;
    size_t ingress_byte_count = 0;
    SpmFmbDdrArenaRole egress_arena = SpmFmbDdrArenaRole::FinalOutput;
    SpmFmbDdrAddressMode egress_address_mode =
        SpmFmbDdrAddressMode::Stable;
    size_t egress_byte_begin = 0;
    size_t egress_byte_count = 0;
    std::shared_ptr<const GraphDmaSemanticEndpoint> ingress_endpoint;
    std::shared_ptr<const GraphDmaSemanticEndpoint> egress_endpoint;
};

SpmFmbPostFnYieldScope::SpmFmbPostFnYieldScope(
    SpmFmbPostFnYieldScope&& other) noexcept
    : owner_(other.owner_),
      target_(std::move(other.target_)),
      active_(other.active_),
      contract_(other.contract_) {
    other.owner_ = nullptr;
    other.active_ = false;
}

SpmFmbPostFnYieldScope& SpmFmbPostFnYieldScope::operator=(
    SpmFmbPostFnYieldScope&& other) noexcept {
    if (this == &other) return *this;
    if (active_ && owner_ != nullptr) {
        owner_->cancel_spm_pipeline_post_fn_dense_yield(*this);
    }
    owner_ = other.owner_;
    target_ = std::move(other.target_);
    active_ = other.active_;
    contract_ = other.contract_;
    other.owner_ = nullptr;
    other.active_ = false;
    return *this;
}

SpmFmbPostFnYieldScope::~SpmFmbPostFnYieldScope() {
    if (active_ && owner_ != nullptr) {
        owner_->cancel_spm_pipeline_post_fn_dense_yield(*this);
    }
}

const SpmFmbPostFnYieldTarget& SpmFmbPostFnYieldScope::target() const {
    TORCH_CHECK(active_ && owner_ != nullptr && target_ != nullptr,
                "post_fn producer-yield scope is inactive");
    return *target_;
}

void SpmFmbPostFnYieldScope::finish() {
    TORCH_CHECK(active_ && owner_ != nullptr && target_ != nullptr,
                "post_fn producer-yield scope is inactive");
    owner_->finish_spm_pipeline_post_fn_dense_yield(*this);
}

struct SpmFmbCompletedBuildTrace::Payload {
    FusedModelBase* owner = nullptr;
    std::weak_ptr<uint64_t> owner_generation;
    uint64_t expected_owner_generation = 0;
    FusedModelBase::Impl::BuildTraceCandidate candidate;
};

SpmFmbCompletedBuildTrace::SpmFmbCompletedBuildTrace(
    std::unique_ptr<Payload> payload)
    : payload_(std::move(payload)) {}

SpmFmbCompletedBuildTrace::~SpmFmbCompletedBuildTrace() = default;
SpmFmbCompletedBuildTrace::SpmFmbCompletedBuildTrace(
    SpmFmbCompletedBuildTrace&&) noexcept = default;
SpmFmbCompletedBuildTrace& SpmFmbCompletedBuildTrace::operator=(
    SpmFmbCompletedBuildTrace&&) noexcept = default;

size_t SpmFmbCompletedBuildTrace::occurrence_ordinal() const {
    TORCH_CHECK(payload_ != nullptr && payload_->candidate.composite,
                "completed composite BUILD trace is empty");
    return payload_->candidate.composite_occurrence_ordinal;
}

uint64_t SpmFmbCompletedBuildTrace::profile_hash() const {
    TORCH_CHECK(payload_ != nullptr && payload_->candidate.composite,
                "completed composite BUILD trace is empty");
    return payload_->candidate.profile_hash;
}

struct SpmFmbSealedCompositeTrace::Payload {
    struct Occurrence {
        FusedModelBase* owner = nullptr;
        SpmFmbSealedBuildTrace trace;
        std::unique_ptr<SpmFmbSealedCallbackYields> yields;

        Occurrence(
            FusedModelBase* owner_arg,
            SpmFmbSealedBuildTrace trace_arg,
            std::unique_ptr<SpmFmbSealedCallbackYields> yields_arg)
            : owner(owner_arg),
              trace(std::move(trace_arg)),
              yields(std::move(yields_arg)) {}

        Occurrence(Occurrence&&) noexcept = default;
        Occurrence& operator=(Occurrence&&) noexcept = default;
        Occurrence(const Occurrence&) = delete;
        Occurrence& operator=(const Occurrence&) = delete;
    };

    size_t expected_occurrence_count = 0;
    size_t expected_producer_yield_count = 0;
    uint64_t policy_fingerprint = 0;
    uint64_t composite_identity = 0;
    uint64_t composite_hash = 0;
    const void* graph_identity = nullptr;
    uint64_t graph_build_generation = 0;
    uint64_t graph_signature_identity = 0;
    uint64_t graph_signature_segment_key = 0;
    uint64_t graph_topology_hash = 0;
    uint64_t semantic_yield_digest = 0;
    size_t producer_yield_count = 0;
    std::weak_ptr<uint64_t> producer_owner_generation;
    uint64_t expected_producer_owner_generation = 0;
    std::vector<Occurrence> occurrences;
};

SpmFmbSealedCompositeTrace::SpmFmbSealedCompositeTrace(
    std::shared_ptr<const Payload> payload)
    : payload_(std::move(payload)) {}

SpmFmbSealedCompositeTrace::~SpmFmbSealedCompositeTrace() = default;
SpmFmbSealedCompositeTrace::SpmFmbSealedCompositeTrace(
    const SpmFmbSealedCompositeTrace&) noexcept = default;
SpmFmbSealedCompositeTrace& SpmFmbSealedCompositeTrace::operator=(
    const SpmFmbSealedCompositeTrace&) noexcept = default;
SpmFmbSealedCompositeTrace::SpmFmbSealedCompositeTrace(
    SpmFmbSealedCompositeTrace&&) noexcept = default;
SpmFmbSealedCompositeTrace& SpmFmbSealedCompositeTrace::operator=(
    SpmFmbSealedCompositeTrace&&) noexcept = default;

uint64_t SpmFmbSealedCompositeTrace::composite_hash() const {
    TORCH_CHECK(payload_ != nullptr,
                "sealed composite BUILD trace is empty");
    return payload_->composite_hash;
}

size_t SpmFmbSealedCompositeTrace::occurrence_count() const {
    TORCH_CHECK(payload_ != nullptr,
                "sealed composite BUILD trace is empty");
    return payload_->occurrences.size();
}

size_t SpmFmbSealedCompositeTrace::producer_yield_count() const {
    TORCH_CHECK(payload_ != nullptr,
                "sealed composite BUILD trace is empty");
    return payload_->producer_yield_count;
}

SpmFmbCompositeYieldBinding SpmFmbCompositeYieldBinding::bind(
    const SpmFmbSealedCompositeTrace& composite,
    size_t producer_occurrence_ordinal,
    size_t producer_yield_ordinal,
    size_t consumer_occurrence_ordinal,
    const SpmFmbConsumerRowSliceEndpoint& consumer,
    uint64_t policy_version,
    Kind kind) {
    TORCH_CHECK(composite.payload_ != nullptr,
                "composite yield binding requires a sealed composite");
    TORCH_CHECK(policy_version != 0,
                "composite yield binding policy version zero is reserved");
    TORCH_CHECK(kind == Kind::DirectProduced || kind == Kind::RunAdd,
                "composite yield binding kind is invalid");

    const auto& payload = *composite.payload_;
    TORCH_CHECK(
        payload.composite_hash != 0 && payload.policy_fingerprint != 0 &&
            payload.expected_occurrence_count == payload.occurrences.size() &&
            payload.expected_producer_yield_count ==
                payload.producer_yield_count &&
            payload.producer_yield_count != 0,
        "composite yield binding received an incomplete composite seal");
    TORCH_CHECK(
        producer_occurrence_ordinal < payload.occurrences.size() &&
            consumer_occurrence_ordinal < payload.occurrences.size() &&
            producer_occurrence_ordinal != consumer_occurrence_ordinal,
        "composite yield binding occurrence ordinal is invalid");

    const auto& producer_occurrence =
        payload.occurrences[producer_occurrence_ordinal];
    const auto& consumer_occurrence =
        payload.occurrences[consumer_occurrence_ordinal];
    const auto& producer_trace = producer_occurrence.trace;
    const auto& consumer_trace = consumer_occurrence.trace;
    TORCH_CHECK(
        producer_occurrence.owner != nullptr &&
            producer_occurrence.yields != nullptr &&
            producer_trace.composite_ && producer_trace.composite_producer_ &&
            producer_trace.composite_occurrence_ordinal_ ==
                producer_occurrence_ordinal &&
            producer_trace.composite_policy_fingerprint_ != 0 &&
            producer_trace.composite_occurrence_profile_hash_ != 0 &&
            producer_trace.layer_producer_yield_capability_ ==
                SpmFmbLayerProducerYieldCapability::
                    CanonicalDenseReplicatedFp16 &&
            !producer_trace.cpu_dry_,
        "composite yield binding producer occurrence lacks live callback-yield "
        "authority");
    TORCH_CHECK(
        consumer_occurrence.owner != nullptr &&
            consumer_occurrence.yields == nullptr &&
            consumer_trace.composite_ && !consumer_trace.composite_producer_ &&
            consumer_trace.composite_occurrence_ordinal_ ==
                consumer_occurrence_ordinal &&
            consumer_trace.layer_producer_yield_capability_ ==
                SpmFmbLayerProducerYieldCapability::Unsealed &&
            !consumer_trace.cpu_dry_,
        "composite yield binding consumer occurrence is not one live, "
        "yield-free FMB trace");

    const auto& producer_yields = *producer_occurrence.yields;
    TORCH_CHECK(
        producer_yields.yield_hash_ != 0 &&
            producer_yields.capability_ ==
                SpmFmbLayerProducerYieldCapability::
                    CanonicalDenseReplicatedFp16 &&
            producer_yields.policy_fingerprint_ != 0 &&
            producer_yields.policy_fingerprint_ ==
                producer_trace.layer_producer_yield_policy_fingerprint_ &&
            producer_yields.source_trace_.trace_hash_ ==
                producer_trace.trace_hash_ &&
            producer_yield_ordinal < producer_yields.yields_.size(),
        "composite yield binding producer yield ordinal or policy is invalid");
    const auto& producer_yield =
        producer_yields.yields_[producer_yield_ordinal];
    TORCH_CHECK(
        producer_yield.ordinal == producer_yield_ordinal &&
            producer_yield.callback_index <
                producer_trace.callback_summaries_.size() &&
            producer_yield.semantic_id != 0 &&
            producer_yield.writer_kind ==
                SpmFmbTerminalWriterKind::AllReduceSumResidual &&
            producer_yield.logical_capacity ==
                producer_yield.spec.storage_bytes() &&
            producer_yield.aligned_capacity ==
                producer_yield.logical_capacity,
        "composite yield binding producer yield is malformed");
    producer_yield.spec.validate();
    const auto& producer_callback =
        producer_trace.callback_summaries_[producer_yield.callback_index];
    TORCH_CHECK(
        producer_callback.kind == producer_yield.callback_kind &&
            producer_callback.key == producer_yield.callback_key &&
            (producer_callback.kind == SpmFmbOccurrenceKind::LayerBody ||
             producer_callback.kind == SpmFmbOccurrenceKind::PostFn),
        "composite yield binding producer callback identity drifted");

    const std::shared_ptr<uint64_t> producer_owner =
        producer_trace.owner_generation_.lock();
    const std::shared_ptr<uint64_t> consumer_trace_owner =
        consumer_trace.owner_generation_.lock();
    const std::shared_ptr<const uint64_t> consumer_endpoint_owner =
        consumer.manifest_.owner_generation_.lock();
    TORCH_CHECK(
        producer_owner != nullptr && consumer_trace_owner != nullptr &&
            consumer_endpoint_owner != nullptr &&
            *producer_owner == producer_trace.expected_owner_generation_ &&
            *consumer_trace_owner ==
                consumer_trace.expected_owner_generation_ &&
            *consumer_endpoint_owner ==
                consumer.manifest_.expected_owner_generation_ &&
            consumer_trace_owner.get() == consumer_endpoint_owner.get(),
        "composite yield binding producer or consumer owner is stale");
    TORCH_CHECK(
        !consumer.manifest_.cpu_dry_ && consumer.identity_hash_ != 0 &&
            consumer.manifest_.profile_hash_ == consumer_trace.profile_hash_ &&
            consumer.manifest_.allocation_hash_ ==
                consumer_trace.allocation_hash_ &&
            consumer.manifest_.identity_hash_ ==
                consumer.manifest_.manifest_.layout_hash,
        "composite yield binding consumer manifest does not belong to the "
        "sealed consumer occurrence");

    size_t matching_consumer_callbacks = 0;
    for (const auto& callback : consumer_trace.callback_summaries_) {
        if (callback.kind == consumer.occurrence_kind_ &&
            callback.key == consumer.occurrence_key_ &&
            callback.body_id == consumer.body_id_ &&
            callback.group_id == consumer.group_id_ &&
            callback.layer == consumer.occurrence_layer_ &&
            callback.traversal_ordinal == consumer.traversal_ordinal_ &&
            callback.chunk_index == consumer.chunk_index_) {
            ++matching_consumer_callbacks;
        }
    }
    TORCH_CHECK(
        matching_consumer_callbacks == 1,
        "composite yield binding consumer endpoint matched ",
        matching_consumer_callbacks,
        " callbacks in its sealed consumer occurrence instead of exactly one");
    TORCH_CHECK(
        producer_yield.spec.dtype == consumer.consumer_.spec.dtype &&
            producer_yield.spec.distribution ==
                consumer.consumer_.spec.distribution &&
            producer_yield.spec.cols == consumer.consumer_.spec.cols &&
            producer_yield.spec.rows == consumer.row_count_,
        "composite yield binding producer and consumer row-slice geometry "
        "differ");

    // This identity deliberately omits composite/Graph identity, build
    // generation, Graph node windows, semantic marker IDs, callback/yield
    // topology hashes, and the full composite hash.  Those values remain in
    // the retained authority above and are revalidated separately.
    uint64_t stable_hash = fmb_hash_u64(
        kFmbFnvOffset, UINT64_C(0x434f4d5042494e44));
    const std::vector<uint64_t> stable_values = {
        static_cast<uint64_t>(kind),
        reinterpret_cast<uintptr_t>(producer_occurrence.owner),
        reinterpret_cast<uintptr_t>(consumer_occurrence.owner),
        payload.policy_fingerprint,
        payload.expected_occurrence_count,
        payload.expected_producer_yield_count,
        producer_occurrence_ordinal,
        producer_trace.composite_producer_local_ordinal_,
        producer_yield_ordinal,
        producer_yields.yields_.size(),
        consumer_occurrence_ordinal,
        producer_trace.profile_hash_,
        producer_trace.allocation_hash_,
        producer_trace.composite_occurrence_profile_hash_,
        producer_trace.expected_owner_generation_,
        producer_yields.policy_fingerprint_,
        producer_yield.callback_index,
        static_cast<uint8_t>(producer_callback.kind),
        producer_callback.key,
        producer_callback.body_id,
        producer_callback.group_id,
        static_cast<uint64_t>(producer_callback.layer),
        producer_callback.traversal_ordinal,
        static_cast<uint64_t>(producer_callback.chunk_index),
        static_cast<uint64_t>(producer_callback.chunk_offset),
        static_cast<uint64_t>(producer_callback.chunk_len),
        static_cast<uint64_t>(producer_callback.chunk_kv_seq_len),
        producer_callback.scope_mask,
        producer_callback.updates_io_context,
        producer_callback.input_in_spm,
        producer_callback.output_to_spm,
        static_cast<uint64_t>(producer_callback.local_phase_begin),
        static_cast<uint64_t>(producer_callback.local_phase_end),
        producer_callback.global_origin,
        static_cast<uint8_t>(producer_yield.writer_kind),
        static_cast<uint8_t>(producer_yield.spec.dtype),
        static_cast<uint64_t>(producer_yield.spec.rows),
        static_cast<uint64_t>(producer_yield.spec.cols),
        static_cast<uint8_t>(producer_yield.spec.distribution),
        producer_yield.allocation_ordinal,
        producer_yield.declaration_ordinal,
        producer_yield.alias_root_ordinal,
        producer_yield.declaration_name_hash,
        producer_yield.alias_root_name_hash,
        producer_yield.root_offset,
        producer_yield.allocation_offset_from_root,
        producer_yield.logical_capacity,
        producer_yield.aligned_capacity,
        consumer.manifest_.layout_hash_,
        consumer.manifest_.allocation_hash_,
        consumer.manifest_.profile_hash_,
        consumer.manifest_.expected_owner_generation_,
        consumer.identity_hash_,
        policy_version,
    };
    for (const uint64_t value : stable_values) {
        stable_hash = fmb_hash_u64(stable_hash, value);
    }
    stable_hash = fmb_nonzero_hash(stable_hash);

    return SpmFmbCompositeYieldBinding(
        composite, composite.payload_.get(), payload.occurrences.size(),
        payload.producer_yield_count, payload.policy_fingerprint,
        producer_occurrence_ordinal,
        producer_trace.composite_producer_local_ordinal_,
        producer_yield_ordinal, producer_yields.yields_.size(),
        consumer_occurrence_ordinal, producer_trace.profile_hash_,
        producer_trace.allocation_hash_,
        producer_trace.composite_occurrence_profile_hash_, producer_owner,
        producer_trace.expected_owner_generation_,
        producer_yields.policy_fingerprint_, producer_yield.callback_index,
        producer_callback.kind, producer_callback.key,
        producer_callback.body_id, producer_callback.group_id,
        producer_callback.layer, producer_callback.traversal_ordinal,
        producer_callback.chunk_index, producer_callback.chunk_offset,
        producer_callback.chunk_len, producer_callback.chunk_kv_seq_len,
        producer_callback.local_phase_begin,
        producer_callback.local_phase_end,
        producer_callback.global_origin,
        producer_yield.semantic_id, producer_yield.writer_kind,
        producer_yield.spec, producer_yield.allocation_ordinal,
        producer_yield.declaration_ordinal,
        producer_yield.alias_root_ordinal,
        producer_yield.declaration_name_hash,
        producer_yield.alias_root_name_hash, producer_yield.root_offset,
        producer_yield.allocation_offset_from_root,
        producer_yield.logical_capacity, producer_yield.aligned_capacity,
        consumer, policy_version, kind, stable_hash);
}

void SpmFmbCompositeYieldBinding::validate_live_authority(
    const void* expected_composite) const {
    TORCH_CHECK(composite_.payload_ != nullptr && expected_composite != nullptr &&
                    composite_authority_ == expected_composite &&
                    composite_.payload_.get() == expected_composite &&
                    stable_hash_ != 0,
                "composite yield binding authority is empty or foreign");
    const auto& payload = *composite_.payload_;
    TORCH_CHECK(
        payload.occurrences.size() == composite_occurrence_count_ &&
            payload.producer_yield_count ==
                composite_producer_yield_count_ &&
            payload.policy_fingerprint == composite_policy_fingerprint_ &&
            producer_occurrence_ordinal_ < payload.occurrences.size() &&
            consumer_occurrence_ordinal_ < payload.occurrences.size(),
        "composite yield binding retained seal drifted");
    const auto& producer =
        payload.occurrences[producer_occurrence_ordinal_];
    const auto& consumer_occurrence =
        payload.occurrences[consumer_occurrence_ordinal_];
    TORCH_CHECK(
        producer.yields != nullptr && consumer_occurrence.yields == nullptr &&
            producer.trace.composite_producer_ &&
            !consumer_occurrence.trace.composite_producer_ &&
            producer.trace.composite_producer_local_ordinal_ ==
                producer_local_ordinal_ &&
            producer.trace.profile_hash_ == producer_profile_hash_ &&
            producer.trace.allocation_hash_ == producer_allocation_hash_ &&
            producer.trace.composite_occurrence_profile_hash_ ==
                producer_occurrence_profile_hash_ &&
            producer_yield_ordinal_ < producer.yields->yields_.size() &&
            producer.yields->yields_.size() ==
                producer_occurrence_yield_count_,
        "composite yield binding retained producer or consumer drifted");
    const auto& yield = producer.yields->yields_[producer_yield_ordinal_];
    TORCH_CHECK(
        yield.ordinal == producer_yield_ordinal_ &&
            yield.callback_index == callback_index_ &&
            yield.semantic_id == semantic_yield_id_ &&
            yield.writer_kind == writer_kind_ && yield.spec == produced_spec_ &&
            yield.allocation_ordinal == allocation_ordinal_ &&
            yield.declaration_ordinal == declaration_ordinal_ &&
            yield.alias_root_ordinal == alias_root_ordinal_ &&
            yield.declaration_name_hash == declaration_name_hash_ &&
            yield.alias_root_name_hash == alias_root_name_hash_ &&
            yield.root_offset == root_offset_ &&
            yield.allocation_offset_from_root ==
                allocation_offset_from_root_ &&
            yield.logical_capacity == logical_capacity_ &&
            yield.aligned_capacity == aligned_capacity_,
        "composite yield binding retained yield drifted");
    const auto& callback = producer.trace.callback_summaries_[callback_index_];
    TORCH_CHECK(
        callback.kind == callback_kind_ && callback.key == callback_key_ &&
            callback.body_id == callback_body_id_ &&
            callback.group_id == callback_group_id_ &&
            callback.layer == callback_layer_ &&
            callback.traversal_ordinal == callback_traversal_ordinal_ &&
            callback.chunk_index == callback_chunk_index_ &&
            callback.chunk_offset == callback_chunk_offset_ &&
            callback.chunk_len == callback_chunk_len_ &&
            callback.chunk_kv_seq_len == callback_chunk_kv_seq_len_ &&
            callback.local_phase_begin == callback_local_phase_begin_ &&
            callback.local_phase_end == callback_local_phase_end_ &&
            callback.global_origin == callback_global_origin_,
        "composite yield binding retained callback drifted");
    const std::shared_ptr<uint64_t> producer_owner =
        producer_owner_generation_.lock();
    const std::shared_ptr<const uint64_t> consumer_owner =
        consumer_.manifest_.owner_generation_.lock();
    TORCH_CHECK(
        producer_owner != nullptr && consumer_owner != nullptr &&
            *producer_owner == expected_producer_owner_generation_ &&
            *consumer_owner ==
                consumer_.manifest_.expected_owner_generation_,
        "composite yield binding owner is stale");
}

SpmFmbCompositeDirectProducedBinding
SpmFmbCompositeDirectProducedBinding::bind(
    const SpmFmbSealedCompositeTrace& composite,
    size_t producer_occurrence_ordinal,
    size_t producer_yield_ordinal,
    size_t consumer_occurrence_ordinal,
    const SpmFmbConsumerRowSliceEndpoint& consumer,
    uint64_t policy_version) {
    return SpmFmbCompositeDirectProducedBinding(
        SpmFmbCompositeYieldBinding::bind(
            composite, producer_occurrence_ordinal,
            producer_yield_ordinal, consumer_occurrence_ordinal, consumer,
            policy_version,
            SpmFmbCompositeYieldBinding::Kind::DirectProduced));
}

SpmFmbCompositeRunAddBinding SpmFmbCompositeRunAddBinding::bind(
    const SpmFmbSealedCompositeTrace& composite,
    size_t producer_occurrence_ordinal,
    size_t producer_yield_ordinal,
    size_t consumer_occurrence_ordinal,
    const SpmFmbConsumerRowSliceEndpoint& consumer,
    uint64_t policy_version) {
    return SpmFmbCompositeRunAddBinding(
        SpmFmbCompositeYieldBinding::bind(
            composite, producer_occurrence_ordinal,
            producer_yield_ordinal, consumer_occurrence_ordinal, consumer,
            policy_version, SpmFmbCompositeYieldBinding::Kind::RunAdd));
}

struct SpmCompositeTraceCoordinator::Impl {
    enum class Mode : uint8_t {
        Idle,
        Building,
        Sealed,
        Replaying,
        ReplayComplete,
        Cancelled,
    };

    size_t expected_occurrence_count = 0;
    size_t expected_producer_yield_count = 0;
    uint64_t policy_fingerprint = 0;
    Mode mode = Mode::Idle;
    RpuKernelGraph* graph = nullptr;
    uint64_t graph_build_generation = 0;
    uint64_t graph_signature_identity = 0;
    uint64_t graph_signature_segment_key = 0;
    uint64_t composite_identity = 0;
    uint64_t next_scope_epoch = 1;
    uint64_t next_outer_fast_ticket_nonce = 0;
    uint64_t outer_fast_ticket_nonce = 0;
    std::thread::id owner_thread;
    uint64_t active_scope_epoch = 0;
    FusedModelBase* active_owner = nullptr;
    bool active_is_producer = false;
    size_t active_producer_local_ordinal = 0;
    FusedModelBase* producer_owner = nullptr;
    FusedModelBase* replay_consumer_owner = nullptr;
    std::weak_ptr<uint64_t> producer_owner_generation;
    uint64_t expected_producer_owner_generation = 0;
    uint64_t producer_layout_hash = 0;
    uint64_t producer_allocation_hash = 0;
    SpmFmbCompositeOccurrenceCapability producer_capability =
        SpmFmbCompositeOccurrenceCapability::Unsealed;
    uint64_t producer_policy_fingerprint = 0;
    uint64_t producer_composite_profile_hash = 0;
    size_t next_producer_occurrence = 0;
    size_t replay_consumer_occurrence_count = 0;
    bool replay_outer_fast_owner_order_exact = false;
    size_t next_replay_producer_occurrence = 0;
    size_t next_replay_occurrence = 0;
    size_t completed_replay_yields = 0;
    const SpmPipelineLease* physical_lease = nullptr;
    bool physical_require_committed = false;
    FusedModelBase* physical_producer_owner = nullptr;
    FusedModelBase* physical_consumer_owner = nullptr;
    std::shared_ptr<const uint8_t> lifetime =
        std::make_shared<const uint8_t>(0);
    std::vector<SpmFmbCompletedBuildTrace> completed;
    std::shared_ptr<const SpmFmbSealedCompositeTrace::Payload> replay;
};

SpmCompositeOuterFastReplayTicket::SpmCompositeOuterFastReplayTicket(
    SpmCompositeTraceCoordinator* coordinator,
    std::weak_ptr<const uint8_t> coordinator_lifetime,
    uint64_t nonce,
    size_t graph_extent,
    size_t occurrence_count,
    size_t producer_count,
    size_t producer_yield_count)
    : coordinator_(coordinator),
      coordinator_lifetime_(std::move(coordinator_lifetime)),
      nonce_(nonce),
      graph_extent_(graph_extent),
      occurrence_count_(occurrence_count),
      producer_count_(producer_count),
      producer_yield_count_(producer_yield_count) {}

SpmCompositeOuterFastReplayTicket::~SpmCompositeOuterFastReplayTicket() {
    if (coordinator_ != nullptr &&
        !coordinator_lifetime_.expired()) {
        coordinator_->cancel_outer_fast_replay_ticket(nonce_);
    }
}

SpmCompositeOuterFastReplayTicket::SpmCompositeOuterFastReplayTicket(
    SpmCompositeOuterFastReplayTicket&& other) noexcept
    : coordinator_(other.coordinator_),
      coordinator_lifetime_(std::move(other.coordinator_lifetime_)),
      nonce_(other.nonce_),
      graph_extent_(other.graph_extent_),
      occurrence_count_(other.occurrence_count_),
      producer_count_(other.producer_count_),
      producer_yield_count_(other.producer_yield_count_) {
    other.release();
}

void SpmCompositeOuterFastReplayTicket::release() noexcept {
    coordinator_ = nullptr;
    nonce_ = 0;
    graph_extent_ = 0;
    occurrence_count_ = 0;
    producer_count_ = 0;
    producer_yield_count_ = 0;
}

SpmCompositeOccurrenceScope::SpmCompositeOccurrenceScope(
    SpmCompositeTraceCoordinator* coordinator,
    std::weak_ptr<const uint8_t> coordinator_lifetime,
    uint64_t scope_epoch)
    : coordinator_(coordinator),
      coordinator_lifetime_(std::move(coordinator_lifetime)),
      scope_epoch_(scope_epoch) {}

SpmCompositeOccurrenceScope::~SpmCompositeOccurrenceScope() {
    if (active_ && coordinator_ != nullptr &&
        !coordinator_lifetime_.expired()) {
        coordinator_->cancel_occurrence(scope_epoch_);
    }
}

SpmCompositeOccurrenceScope::SpmCompositeOccurrenceScope(
    SpmCompositeOccurrenceScope&& other) noexcept
    : coordinator_(other.coordinator_),
      coordinator_lifetime_(std::move(other.coordinator_lifetime_)),
      scope_epoch_(other.scope_epoch_),
      active_(other.active_) {
    other.coordinator_ = nullptr;
    other.scope_epoch_ = 0;
    other.active_ = false;
}

SpmCompositeOccurrenceScope& SpmCompositeOccurrenceScope::operator=(
    SpmCompositeOccurrenceScope&& other) noexcept {
    if (this == &other) return *this;
    if (active_ && coordinator_ != nullptr &&
        !coordinator_lifetime_.expired()) {
        coordinator_->cancel_occurrence(scope_epoch_);
    }
    coordinator_ = other.coordinator_;
    coordinator_lifetime_ = std::move(other.coordinator_lifetime_);
    scope_epoch_ = other.scope_epoch_;
    active_ = other.active_;
    other.coordinator_ = nullptr;
    other.scope_epoch_ = 0;
    other.active_ = false;
    return *this;
}

void SpmCompositeOccurrenceScope::finish() {
    const std::shared_ptr<const uint8_t> keepalive =
        coordinator_lifetime_.lock();
    TORCH_CHECK(active_ && coordinator_ != nullptr && keepalive != nullptr,
                "composite occurrence scope is inactive");
    coordinator_->finish_occurrence(scope_epoch_);
    active_ = false;
}

SpmCompositeTraceCoordinator::SpmCompositeTraceCoordinator(
    size_t expected_occurrence_count,
    size_t expected_producer_yield_count,
    uint64_t policy_fingerprint)
    : pimpl_(std::make_unique<Impl>()) {
    TORCH_CHECK(expected_occurrence_count > 0 &&
                    expected_producer_yield_count > 0 &&
                    policy_fingerprint != 0,
                "composite trace coordinator requires nonempty occurrence "
                "and producer-yield counts plus a nonzero policy "
                "fingerprint");
    pimpl_->expected_occurrence_count = expected_occurrence_count;
    pimpl_->expected_producer_yield_count =
        expected_producer_yield_count;
    pimpl_->policy_fingerprint = policy_fingerprint;
    pimpl_->owner_thread = std::this_thread::get_id();
    pimpl_->completed.reserve(expected_occurrence_count);
}

SpmCompositeTraceCoordinator::~SpmCompositeTraceCoordinator() {
    cancel();
}

void SpmCompositeTraceCoordinator::adopt_physical_component(
    FusedModelBase& owner,
    const SpmPipelineLease& lease,
    bool producer) {
    TORCH_CHECK(
        pimpl_->owner_thread == std::this_thread::get_id() &&
            pimpl_->mode == Impl::Mode::Idle && pimpl_->graph == nullptr &&
            pimpl_->active_owner == nullptr &&
            (!pimpl_->physical_lease ||
             pimpl_->physical_lease == &lease) &&
            (producer ? pimpl_->physical_producer_owner == nullptr
                      : pimpl_->physical_consumer_owner == nullptr),
        "composite physical component adoption requires one idle role slot "
        "and lease");
    lease.validate_composite_execution_authority(
        lease.epoch(), lease.plan_hash(), /*require_committed=*/false);
    owner.adopt_spm_pipeline_composite_component(lease, producer);
    pimpl_->physical_lease = &lease;
    if (producer) {
        pimpl_->physical_producer_owner = &owner;
    } else {
        pimpl_->physical_consumer_owner = &owner;
    }
}

void SpmCompositeTraceCoordinator::release_physical_components(
    const SpmPipelineLease& lease) {
    TORCH_CHECK(
        pimpl_->owner_thread == std::this_thread::get_id() &&
            !RpuKernelGraph::has_active() &&
            pimpl_->active_scope_epoch == 0 &&
            pimpl_->active_owner == nullptr &&
            pimpl_->physical_lease == &lease &&
            (pimpl_->physical_producer_owner != nullptr ||
             pimpl_->physical_consumer_owner != nullptr) &&
            (pimpl_->mode == Impl::Mode::Idle ||
             pimpl_->mode == Impl::Mode::Sealed ||
             pimpl_->mode == Impl::Mode::ReplayComplete ||
             pimpl_->mode == Impl::Mode::Cancelled),
        "composite physical component release requires an inactive Graph "
        "transaction with at least one adopted owner");
    lease.validate_composite_execution_authority(
        lease.epoch(), lease.plan_hash(), lease.composite_committed());
    if (pimpl_->physical_producer_owner != nullptr) {
        pimpl_->physical_producer_owner
            ->validate_spm_pipeline_composite_component(
                lease, /*producer=*/true);
    }
    if (pimpl_->physical_consumer_owner != nullptr) {
        pimpl_->physical_consumer_owner
            ->validate_spm_pipeline_composite_component(
                lease, /*producer=*/false);
    }
    if (pimpl_->physical_consumer_owner != nullptr) {
        pimpl_->physical_consumer_owner->release_spm_pipeline_component(
            lease.epoch(), lease.plan_hash());
    }
    if (pimpl_->physical_producer_owner != nullptr) {
        pimpl_->physical_producer_owner->release_spm_pipeline_component(
            lease.epoch(), lease.plan_hash());
    }
    pimpl_->physical_producer_owner = nullptr;
    pimpl_->physical_consumer_owner = nullptr;
    pimpl_->physical_lease = nullptr;
    pimpl_->physical_require_committed = false;
}

void SpmCompositeTraceCoordinator::begin_build(RpuKernelGraph& graph) {
    TORCH_CHECK(pimpl_->owner_thread == std::this_thread::get_id() &&
                    pimpl_->mode == Impl::Mode::Idle &&
                    pimpl_->graph == nullptr &&
                    pimpl_->completed.empty() &&
                    !pimpl_->replay,
                "composite BUILD coordinator is not idle");
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    TORCH_CHECK(stamp.graph == &graph &&
                    stamp.state == RpuKernelGraph::State::RECORDING &&
                    stamp.position == 0 &&
                    stamp.build_generation != 0 &&
                    stamp.signature_identity != 0 &&
                    stamp.signature_segment_key != 0,
                "composite BUILD coordinator requires one signed outer "
                "Graph at RECORDING cursor zero");
    uint64_t identity = fmb_hash_u64(kFmbFnvOffset, 33);
    identity = fmb_hash_u64(identity, pimpl_->policy_fingerprint);
    identity = fmb_hash_u64(
        identity, pimpl_->expected_occurrence_count);
    identity = fmb_hash_u64(
        identity, pimpl_->expected_producer_yield_count);
    identity = fmb_hash_u64(identity, stamp.signature_identity);
    identity = fmb_hash_u64(identity, stamp.signature_segment_key);
    pimpl_->graph = &graph;
    pimpl_->graph_build_generation = stamp.build_generation;
    pimpl_->graph_signature_identity = stamp.signature_identity;
    pimpl_->graph_signature_segment_key =
        stamp.signature_segment_key;
    pimpl_->composite_identity = fmb_nonzero_hash(identity);
    auto& graph_state = graph.composite_fmb_invocation_;
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;
    TORCH_CHECK(graph_state.phase == GraphPhase::None,
                "composite BUILD Graph already has an invocation owner");
    graph_state.phase = GraphPhase::BuildOpen;
    graph_state.coordinator = this;
    graph_state.identity = pimpl_->composite_identity;
    graph_state.build_generation = stamp.build_generation;
    graph_state.signature_identity = stamp.signature_identity;
    graph_state.signature_segment_key = stamp.signature_segment_key;
    graph_state.expected_occurrences =
        pimpl_->expected_occurrence_count;
    graph_state.completed_occurrences = 0;
    graph_state.expected_producer_yields =
        pimpl_->expected_producer_yield_count;
    graph_state.completed_producer_yields = 0;
    graph_state.next_outer_begin = 0;
    pimpl_->mode = Impl::Mode::Building;
}

void SpmCompositeTraceCoordinator::begin_build(
    RpuKernelGraph& graph,
    const SpmPipelineLease& lease,
    bool require_committed) {
    TORCH_CHECK(
        pimpl_->physical_lease == &lease &&
            pimpl_->physical_producer_owner != nullptr &&
            pimpl_->physical_consumer_owner != nullptr &&
            lease.composite_committed() == require_committed,
        "lease-aware composite BUILD requires both adopted component roles "
        "and the requested reservation stage");
    lease.validate_composite_execution_authority(
        lease.epoch(), lease.plan_hash(), require_committed);
    begin_build(graph);
    auto cancel_failed_begin = c10::make_scope_exit([&] { cancel(); });
    pimpl_->physical_require_committed = require_committed;
    if (require_committed) {
        graph.request_retained_physical_arena_build();
    }
    cancel_failed_begin.release();
}

SpmCompositeOccurrenceScope
SpmCompositeTraceCoordinator::begin_build_occurrence(
    FusedModelBase& owner,
    const SpmFmbResolvedExecutionProfile& profile) {
    TORCH_CHECK(pimpl_->owner_thread == std::this_thread::get_id() &&
                    pimpl_->mode == Impl::Mode::Building &&
                    pimpl_->graph != nullptr &&
                    pimpl_->active_scope_epoch == 0 &&
                    pimpl_->active_owner == nullptr &&
                    pimpl_->completed.size() <
                        pimpl_->expected_occurrence_count,
                "composite BUILD occurrence is duplicate or out of range");
    const GraphOpStreamStamp stamp = pimpl_->graph->op_stream_stamp();
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;
    const auto& graph_state =
        pimpl_->graph->composite_fmb_invocation_;
    TORCH_CHECK(stamp.graph == pimpl_->graph &&
                    stamp.state == RpuKernelGraph::State::RECORDING &&
                    stamp.build_generation ==
                        pimpl_->graph_build_generation &&
                    stamp.signature_identity ==
                        pimpl_->graph_signature_identity &&
                    stamp.signature_segment_key ==
                        pimpl_->graph_signature_segment_key &&
                    graph_state.phase == GraphPhase::BuildOpen &&
                    graph_state.coordinator == this &&
                    graph_state.identity ==
                        pimpl_->composite_identity &&
                    graph_state.completed_occurrences ==
                        pimpl_->completed.size() &&
                    graph_state.next_outer_begin == stamp.position,
                "composite BUILD occurrence Graph identity drifted");
    TORCH_CHECK(pimpl_->next_scope_epoch != 0 &&
                    pimpl_->next_scope_epoch !=
                        std::numeric_limits<uint64_t>::max(),
                "composite occurrence scope epoch exhausted");
    const size_t ordinal = pimpl_->completed.size();
    auto cancel_failed_begin = c10::make_scope_exit([&] {
        if (owner.pimpl_->pipeline_build_trace_.armed) {
            clear_pipeline_build_trace_noexcept(*owner.pimpl_);
        }
        owner.cancel_spm_pipeline_composite_occurrence_runtime();
        owner.cancel_spm_pipeline_composite_physical_execution();
        if (pimpl_->graph != nullptr &&
            pimpl_->graph->composite_fmb_invocation_.coordinator == this) {
            pimpl_->graph->composite_fmb_invocation_.phase =
                GraphPhase::Poisoned;
            pimpl_->graph->composite_fmb_invocation_.coordinator = nullptr;
        }
        pimpl_->mode = Impl::Mode::Cancelled;
        pimpl_->active_scope_epoch = 0;
        pimpl_->active_owner = nullptr;
        pimpl_->active_is_producer = false;
        pimpl_->active_producer_local_ordinal = 0;
    });
    const bool producer =
        owner.spm_fmb_layer_producer_yield_capability() !=
        SpmFmbLayerProducerYieldCapability::Unsealed;
    if (pimpl_->physical_lease != nullptr) {
        TORCH_CHECK(
            (producer && pimpl_->physical_producer_owner == &owner) ||
                (!producer &&
                 pimpl_->physical_consumer_owner == &owner),
            "composite BUILD occurrence owner differs from its adopted "
            "physical role");
    }
    size_t producer_local_ordinal = 0;
    SpmFmbCompositeOccurrenceCapability composite_capability =
        SpmFmbCompositeOccurrenceCapability::Unsealed;
    uint64_t composite_policy_fingerprint = 0;
    bool preload_follower = false;
    if (producer) {
        producer_local_ordinal = pimpl_->next_producer_occurrence;
        composite_capability =
            owner.spm_fmb_composite_occurrence_capability();
        composite_policy_fingerprint =
            owner.spm_fmb_composite_occurrence_policy_fingerprint();
        const auto& snapshot = owner.pimpl_->pipeline_manifest_snapshot_;
        TORCH_CHECK(
            snapshot.valid &&
                ((composite_capability ==
                      SpmFmbCompositeOccurrenceCapability::Unsealed &&
                  composite_policy_fingerprint == 0) ||
                 (composite_capability ==
                      SpmFmbCompositeOccurrenceCapability::
                          StableThreeInputSlotsSameOwnerPreload &&
                  composite_policy_fingerprint != 0 &&
                  producer_local_ordinal < 3)),
            "composite BUILD producer has an invalid occurrence runtime "
            "capability or exceeds its stable input slots");
        if (pimpl_->producer_owner == nullptr) {
            pimpl_->producer_owner = &owner;
            pimpl_->producer_owner_generation =
                owner.pimpl_->pipeline_manifest_generation_;
            pimpl_->expected_producer_owner_generation =
                *owner.pimpl_->pipeline_manifest_generation_;
            pimpl_->producer_layout_hash = snapshot.layout_hash;
            pimpl_->producer_allocation_hash = snapshot.allocation_hash;
            pimpl_->producer_capability = composite_capability;
            pimpl_->producer_policy_fingerprint =
                composite_policy_fingerprint;
        } else {
            const std::shared_ptr<uint64_t> producer_generation =
                pimpl_->producer_owner_generation.lock();
            TORCH_CHECK(
                pimpl_->producer_owner == &owner &&
                    producer_generation != nullptr &&
                    producer_generation.get() ==
                        owner.pimpl_->pipeline_manifest_generation_.get() &&
                    *producer_generation ==
                        pimpl_->expected_producer_owner_generation &&
                    snapshot.layout_hash ==
                        pimpl_->producer_layout_hash &&
                    snapshot.allocation_hash ==
                        pimpl_->producer_allocation_hash &&
                    composite_capability ==
                        pimpl_->producer_capability &&
                    composite_policy_fingerprint ==
                        pimpl_->producer_policy_fingerprint,
                "composite BUILD v1 rejects a second or drifted producer "
                "owner/allocation policy");
        }
        preload_follower =
            composite_capability !=
                SpmFmbCompositeOccurrenceCapability::Unsealed &&
            producer_local_ordinal != 0;
    }
    owner.begin_spm_pipeline_build_trace_impl(
        profile, /*composite=*/true, pimpl_->composite_identity, ordinal,
        producer, producer_local_ordinal, composite_capability,
        composite_policy_fingerprint, preload_follower);
    if (producer) {
        const uint64_t occurrence_profile_hash =
            owner.pimpl_->pipeline_build_trace_
                .composite_occurrence_profile_hash;
        if (producer_local_ordinal == 0) {
            pimpl_->producer_composite_profile_hash =
                occurrence_profile_hash;
        } else {
            TORCH_CHECK(
                occurrence_profile_hash ==
                    pimpl_->producer_composite_profile_hash,
                "composite BUILD producer occurrence profile drifted");
        }
    }
    if (producer &&
        composite_capability !=
            SpmFmbCompositeOccurrenceCapability::Unsealed) {
        owner.arm_spm_pipeline_composite_occurrence_runtime(
            pimpl_->composite_identity, producer_local_ordinal,
            composite_capability, composite_policy_fingerprint,
            pimpl_->producer_layout_hash,
            pimpl_->producer_allocation_hash,
            owner.pimpl_->pipeline_manifest_snapshot_
                .persistent_generation,
            owner.pimpl_->pipeline_build_trace_
                .composite_occurrence_profile_hash,
            preload_follower);
    }
    if (pimpl_->physical_lease != nullptr) {
        owner.arm_spm_pipeline_composite_physical_execution(
            *pimpl_->physical_lease, pimpl_->composite_identity, producer,
            producer ? producer_local_ordinal : 0,
            pimpl_->physical_require_committed);
    }
    const uint64_t epoch = pimpl_->next_scope_epoch++;
    pimpl_->active_scope_epoch = epoch;
    pimpl_->active_owner = &owner;
    pimpl_->active_is_producer = producer;
    pimpl_->active_producer_local_ordinal = producer_local_ordinal;
    cancel_failed_begin.release();
    return SpmCompositeOccurrenceScope(
        this, pimpl_->lifetime, epoch);
}

SpmFmbSealedCompositeTrace
SpmCompositeTraceCoordinator::seal_build(RpuKernelGraph& graph) {
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;
    const auto& graph_state = graph.composite_fmb_invocation_;
    TORCH_CHECK(pimpl_->owner_thread == std::this_thread::get_id() &&
                    pimpl_->mode == Impl::Mode::Building &&
                    pimpl_->graph == &graph &&
                    pimpl_->active_scope_epoch == 0 &&
                    pimpl_->active_owner == nullptr &&
                    pimpl_->completed.size() ==
                        pimpl_->expected_occurrence_count &&
                    graph.state() == RpuKernelGraph::State::BUILT &&
                    graph_state.phase == GraphPhase::BuildComplete &&
                    graph_state.coordinator == this &&
                    graph_state.identity == pimpl_->composite_identity &&
                    graph_state.build_generation ==
                        pimpl_->graph_build_generation &&
                    graph_state.signature_identity ==
                        pimpl_->graph_signature_identity &&
                    graph_state.signature_segment_key ==
                        pimpl_->graph_signature_segment_key &&
                    graph_state.expected_occurrences ==
                        pimpl_->expected_occurrence_count &&
                    graph_state.completed_occurrences ==
                        pimpl_->expected_occurrence_count &&
                    graph_state.expected_producer_yields ==
                        pimpl_->expected_producer_yield_count &&
                    graph_state.completed_producer_yields ==
                        pimpl_->expected_producer_yield_count &&
                    graph_state.next_outer_begin == graph.graph_size(),
                "composite BUILD seal requires every occurrence complete "
                "and the owning outer Graph BUILT");
    auto invalidate_on_failure = c10::make_scope_exit([&] {
        graph.composite_fmb_invocation_.phase = GraphPhase::Poisoned;
        graph.composite_fmb_invocation_.coordinator = nullptr;
        try {
            graph.invalidate();
        } catch (...) {
            // A live physical lease may defer exact cache eviction.  Keep the
            // retained Graph poisoned; never throw a second exception while
            // unwinding the failed seal transaction.
        }
        pimpl_->mode = Impl::Mode::Cancelled;
        pimpl_->completed.clear();
    });

    auto payload =
        std::make_shared<SpmFmbSealedCompositeTrace::Payload>();
    payload->expected_occurrence_count =
        pimpl_->expected_occurrence_count;
    payload->expected_producer_yield_count =
        pimpl_->expected_producer_yield_count;
    payload->policy_fingerprint = pimpl_->policy_fingerprint;
    payload->composite_identity = pimpl_->composite_identity;
    payload->graph_identity = &graph;
    payload->graph_build_generation = graph.build_generation();
    payload->graph_signature_identity =
        graph.built_signature_identity();
    payload->graph_signature_segment_key =
        graph.built_signature().segment_key;
    payload->graph_topology_hash = graph.build_topology_hash();
    payload->occurrences.reserve(pimpl_->completed.size());

    size_t expected_outer_begin = 0;
    size_t expected_producer_local_ordinal = 0;
    std::shared_ptr<uint64_t> producer_owner;
    std::set<uint64_t> unique_semantic_yield_ids;
    std::vector<GraphSemanticSpmProducerYieldOccurrence>
        expected_graph_yields;
    for (size_t ordinal = 0; ordinal < pimpl_->completed.size(); ++ordinal) {
        SpmFmbCompletedBuildTrace& completed =
            pimpl_->completed[ordinal];
        TORCH_CHECK(completed.payload_ != nullptr,
                    "composite BUILD seal found an empty completed "
                    "occurrence");
        const auto& completed_candidate =
            completed.payload_->candidate;
        const bool completed_candidate_producer =
            completed_candidate.composite_producer;
        const size_t completed_candidate_producer_local_ordinal =
            completed_candidate.composite_producer_local_ordinal;
        const auto completed_candidate_capability =
            completed_candidate.composite_capability;
        const uint64_t completed_candidate_policy_fingerprint =
            completed_candidate.composite_policy_fingerprint;
        const uint64_t completed_candidate_profile_hash =
            completed_candidate.composite_occurrence_profile_hash;
        const bool completed_candidate_preload_follower =
            completed_candidate.composite_preload_follower;
        const bool completed_is_producer =
            completed_candidate.composite_producer;
        const bool completed_runtime_identity_ok =
            completed_is_producer
            ? (completed_candidate.composite_producer &&
               completed_candidate.composite_producer_local_ordinal ==
                   expected_producer_local_ordinal &&
               completed_candidate.composite_capability ==
                   pimpl_->producer_capability &&
               completed_candidate.composite_policy_fingerprint ==
                   pimpl_->producer_policy_fingerprint &&
               completed_candidate.composite_preload_follower ==
                   (completed_candidate.composite_capability !=
                        SpmFmbCompositeOccurrenceCapability::Unsealed &&
                    expected_producer_local_ordinal != 0) &&
               ((completed_candidate.composite_capability ==
                     SpmFmbCompositeOccurrenceCapability::Unsealed &&
                 completed_candidate
                         .composite_occurrence_profile_hash == 0) ||
                (completed_candidate.composite_capability ==
                     SpmFmbCompositeOccurrenceCapability::
                         StableThreeInputSlotsSameOwnerPreload &&
                 completed_candidate
                         .composite_occurrence_profile_hash != 0)) &&
               completed_candidate
                       .composite_occurrence_profile_hash ==
                   pimpl_->producer_composite_profile_hash)
            : (!completed_candidate.composite_producer &&
               completed_candidate
                       .composite_producer_local_ordinal == 0 &&
               completed_candidate.composite_capability ==
                   SpmFmbCompositeOccurrenceCapability::Unsealed &&
               completed_candidate.composite_policy_fingerprint == 0 &&
               completed_candidate.composite_occurrence_profile_hash ==
                   0 &&
               !completed_candidate.composite_preload_follower);
        TORCH_CHECK(completed.payload_ != nullptr &&
                        completed.payload_->owner != nullptr &&
                        completed_candidate.composite &&
                        completed_candidate.composite_identity ==
                            pimpl_->composite_identity &&
                        completed_candidate
                                .composite_occurrence_ordinal == ordinal &&
                        completed_candidate.composite_outer_begin ==
                            expected_outer_begin &&
                        completed_runtime_identity_ok,
                    "composite BUILD seal found a missing, reordered, or "
                    "foreign completed occurrence");
        const std::shared_ptr<uint64_t> completed_owner =
            completed.payload_->owner_generation.lock();
        TORCH_CHECK(
            completed_owner != nullptr &&
                *completed_owner ==
                    completed.payload_->expected_owner_generation,
            "composite BUILD seal owner is stale");
        FusedModelBase* owner = completed.payload_->owner;
        SpmFmbSealedBuildTrace trace =
            owner->seal_spm_pipeline_completed_build_trace(
                std::move(completed), graph);
        TORCH_CHECK(trace.composite_ &&
                        trace.composite_identity_ ==
                            pimpl_->composite_identity &&
                        trace.composite_occurrence_ordinal_ == ordinal &&
                        trace.composite_producer_ ==
                            completed_candidate_producer &&
                        trace.composite_producer_local_ordinal_ ==
                            completed_candidate_producer_local_ordinal &&
                        trace.composite_capability_ ==
                            completed_candidate_capability &&
                        trace.composite_policy_fingerprint_ ==
                            completed_candidate_policy_fingerprint &&
                        trace.composite_occurrence_profile_hash_ ==
                            completed_candidate_profile_hash &&
                        trace.composite_preload_follower_ ==
                            completed_candidate_preload_follower &&
                        trace.composite_outer_begin_ ==
                            expected_outer_begin &&
                        trace.graph_node_end_ <= graph.graph_size(),
                    "composite BUILD seal component identity drifted");
        TORCH_CHECK(
            graph.semantic_spm_producer_yield_id_digest_for_window(
                trace.composite_outer_begin_, trace.graph_node_begin_) == 0,
            "composite BUILD occurrence prefix contains an unclaimed "
            "producer-yield marker");

        std::unique_ptr<SpmFmbSealedCallbackYields> yields;
        const uint64_t component_marker_digest =
            graph.semantic_spm_producer_yield_id_digest_for_window(
                trace.graph_node_begin_, trace.graph_node_end_);
        if (trace.layer_producer_yield_capability_ ==
            SpmFmbLayerProducerYieldCapability::
                CanonicalDenseReplicatedFp16) {
            TORCH_CHECK(component_marker_digest != 0,
                        "composite producer occurrence has no marked "
                        "terminal writer");
            yields = std::make_unique<SpmFmbSealedCallbackYields>(
                owner->seal_spm_pipeline_callback_yields(trace, graph));
            TORCH_CHECK(
                payload->producer_yield_count <=
                    std::numeric_limits<size_t>::max() -
                        yields->yields_.size(),
                "composite producer-yield count overflow");
            payload->producer_yield_count += yields->yields_.size();
            for (const auto& yield : yields->yields_) {
                TORCH_CHECK(
                    yield.ordinal < yields->yields_.size() &&
                        yield.relative_node_begin <
                            yield.relative_node_end &&
                        yield.relative_node_end ==
                            yield.relative_node_begin + 1 &&
                        trace.graph_node_begin_ <=
                            std::numeric_limits<size_t>::max() -
                                yield.relative_node_end &&
                        trace.graph_node_begin_ +
                                yield.relative_node_end <=
                            trace.graph_node_end_ &&
                        unique_semantic_yield_ids
                            .insert(yield.semantic_id)
                            .second,
                    "composite BUILD contains a malformed or duplicate "
                    "producer-yield marker");
                GraphSemanticSpmProducerYieldOccurrence expected;
                expected.relative_node_index =
                    trace.graph_node_begin_ +
                    yield.relative_node_begin;
                expected.semantic_yield_id = yield.semantic_id;
                expected.writer_kind =
                    static_cast<uint8_t>(yield.writer_kind);
                expected_graph_yields.push_back(std::move(expected));
            }
            const std::shared_ptr<uint64_t> current_owner =
                trace.owner_generation_.lock();
            TORCH_CHECK(owner == pimpl_->producer_owner &&
                            current_owner != nullptr &&
                            current_owner.get() ==
                                pimpl_->producer_owner_generation.lock().get() &&
                            *current_owner ==
                                trace.expected_owner_generation_ &&
                            trace.allocation_hash_ ==
                                pimpl_->producer_allocation_hash &&
                            trace.composite_capability_ ==
                                pimpl_->producer_capability &&
                            trace.composite_policy_fingerprint_ ==
                                pimpl_->producer_policy_fingerprint &&
                            trace.composite_occurrence_profile_hash_ ==
                                pimpl_->producer_composite_profile_hash,
                        "composite producer owner is stale at seal");
            if (!producer_owner) {
                producer_owner = current_owner;
                payload->expected_producer_owner_generation =
                    trace.expected_owner_generation_;
            } else {
                TORCH_CHECK(
                    producer_owner.get() == current_owner.get() &&
                        payload->expected_producer_owner_generation ==
                            trace.expected_owner_generation_,
                    "composite BUILD v1 rejects a second producer owner");
            }
            ++expected_producer_local_ordinal;
        } else {
            TORCH_CHECK(
                trace.layer_producer_yield_capability_ ==
                        SpmFmbLayerProducerYieldCapability::Unsealed &&
                    component_marker_digest == 0,
                "composite consumer occurrence contains unsealed producer "
                "markers");
        }
        expected_outer_begin = trace.graph_node_end_;
        payload->occurrences.emplace_back(
            owner, std::move(trace), std::move(yields));
    }
    const auto observed_graph_yields =
        graph.semantic_spm_producer_yields_for_window(
            0, graph.graph_size());
    bool exact_yield_union =
        observed_graph_yields.size() == expected_graph_yields.size();
    if (exact_yield_union) {
        for (size_t index = 0;
             index < expected_graph_yields.size(); ++index) {
            const auto& expected = expected_graph_yields[index];
            const auto& observed = observed_graph_yields[index];
            if (expected.relative_node_index !=
                    observed.relative_node_index ||
                expected.semantic_yield_id !=
                    observed.semantic_yield_id ||
                expected.writer_kind != observed.writer_kind) {
                exact_yield_union = false;
                break;
            }
        }
    }
    TORCH_CHECK(expected_outer_begin == graph.graph_size() &&
                    expected_producer_local_ordinal ==
                        pimpl_->next_producer_occurrence &&
                    (pimpl_->producer_capability !=
                             SpmFmbCompositeOccurrenceCapability::
                                 StableThreeInputSlotsSameOwnerPreload ||
                     expected_producer_local_ordinal == 3) &&
                    payload->producer_yield_count ==
                        pimpl_->expected_producer_yield_count &&
                    exact_yield_union &&
                    producer_owner != nullptr,
                "composite BUILD occurrences do not exactly cover the outer "
                "Graph or its expected unique producer-yield marker union");
    payload->semantic_yield_digest =
        graph.semantic_spm_producer_yield_id_digest_for_fmb();
    payload->producer_owner_generation = producer_owner;

    uint64_t composite_hash = fmb_hash_u64(kFmbFnvOffset, 34);
    const std::vector<uint64_t> header_values = {
        payload->policy_fingerprint,
        payload->composite_identity,
        payload->graph_build_generation,
        payload->graph_signature_identity,
        payload->graph_signature_segment_key,
        payload->graph_topology_hash,
        payload->semantic_yield_digest,
        payload->expected_occurrence_count,
        payload->producer_yield_count,
    };
    for (const uint64_t value : header_values) {
        composite_hash = fmb_hash_u64(composite_hash, value);
    }
    for (const auto& occurrence : payload->occurrences) {
        composite_hash = fmb_hash_u64(
            composite_hash, occurrence.trace.trace_hash_);
        composite_hash = fmb_hash_u64(
            composite_hash,
            occurrence.yields ? occurrence.yields->yield_hash_ : 0);
    }
    payload->composite_hash = fmb_nonzero_hash(composite_hash);

    pimpl_->completed.clear();
    pimpl_->mode = Impl::Mode::Sealed;
    graph.composite_fmb_invocation_ = {};
    pimpl_->graph = nullptr;
    invalidate_on_failure.release();
    return SpmFmbSealedCompositeTrace(std::move(payload));
}

void SpmCompositeTraceCoordinator::begin_replay(
    const SpmFmbSealedCompositeTrace& sealed,
    RpuKernelGraph& graph) {
    TORCH_CHECK(pimpl_->owner_thread == std::this_thread::get_id() &&
                    pimpl_->mode == Impl::Mode::Idle &&
                    sealed.payload_ != nullptr &&
                    sealed.payload_->expected_occurrence_count ==
                        pimpl_->expected_occurrence_count &&
                    sealed.payload_->expected_producer_yield_count ==
                        pimpl_->expected_producer_yield_count &&
                    sealed.payload_->policy_fingerprint ==
                        pimpl_->policy_fingerprint,
                "composite REPLAY coordinator/token policy mismatch");
    const auto& payload = *sealed.payload_;
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    TORCH_CHECK(stamp.graph == &graph &&
                    stamp.state == RpuKernelGraph::State::REPLAYING &&
                    stamp.position == 0 &&
                    stamp.build_generation ==
                        payload.graph_build_generation &&
                    stamp.signature_identity ==
                        payload.graph_signature_identity &&
                    stamp.signature_segment_key ==
                        payload.graph_signature_segment_key &&
                    graph.build_topology_hash() ==
                        payload.graph_topology_hash &&
                    graph.semantic_spm_producer_yield_id_digest_for_fmb() ==
                        payload.semantic_yield_digest,
                "composite REPLAY Graph identity or producer digest is "
                "stale");
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;
    TORCH_CHECK(
        graph.composite_fmb_invocation_.phase == GraphPhase::None &&
            payload.graph_identity == &graph &&
            payload.composite_identity != 0 &&
            payload.composite_hash != 0 &&
            payload.occurrences.size() ==
                payload.expected_occurrence_count &&
            payload.producer_yield_count ==
                payload.expected_producer_yield_count,
        "composite REPLAY token or Graph transaction is stale");
    const std::shared_ptr<uint64_t> producer_owner =
        payload.producer_owner_generation.lock();
    TORCH_CHECK(producer_owner != nullptr &&
                    *producer_owner ==
                        payload.expected_producer_owner_generation,
                "composite REPLAY producer owner is stale");

    size_t expected_outer_begin = 0;
    size_t expected_producer_local_ordinal = 0;
    size_t expected_yield_count = 0;
    FusedModelBase* replay_producer_owner = nullptr;
    FusedModelBase* replay_consumer_owner = nullptr;
    size_t replay_consumer_occurrence_count = 0;
    bool replay_outer_fast_owner_order_exact =
        payload.occurrences.size() == 4;
    uint64_t replay_producer_allocation_hash = 0;
    uint64_t replay_producer_composite_profile_hash = 0;
    SpmFmbCompositeOccurrenceCapability replay_producer_capability =
        SpmFmbCompositeOccurrenceCapability::Unsealed;
    uint64_t replay_producer_policy_fingerprint = 0;
    std::set<uint64_t> unique_semantic_yield_ids;
    std::vector<GraphSemanticSpmProducerYieldOccurrence>
        expected_graph_yields;
    for (size_t ordinal = 0;
         ordinal < payload.occurrences.size(); ++ordinal) {
        const auto& occurrence = payload.occurrences[ordinal];
        const auto& trace = occurrence.trace;
        const std::shared_ptr<uint64_t> trace_owner =
            trace.owner_generation_.lock();
        const bool trace_is_producer = occurrence.yields != nullptr;
        replay_outer_fast_owner_order_exact =
            replay_outer_fast_owner_order_exact &&
            ((ordinal < 3 && trace_is_producer) ||
             (ordinal == 3 && !trace_is_producer));
        bool trace_occurrence_policy_ok = false;
        if (trace_is_producer && occurrence.owner != nullptr &&
            trace_owner != nullptr) {
            const bool unsealed_occurrence =
                trace.composite_capability_ ==
                        SpmFmbCompositeOccurrenceCapability::Unsealed &&
                trace.composite_policy_fingerprint_ == 0 &&
                trace.composite_occurrence_profile_hash_ == 0 &&
                !trace.composite_preload_follower_;
            const bool stable_three_slot_occurrence =
                trace.composite_capability_ ==
                    SpmFmbCompositeOccurrenceCapability::
                        StableThreeInputSlotsSameOwnerPreload &&
                trace.composite_policy_fingerprint_ != 0 &&
                trace.composite_producer_local_ordinal_ < 3 &&
                trace.composite_preload_follower_ ==
                    (trace.composite_producer_local_ordinal_ != 0) &&
                trace.composite_occurrence_profile_hash_ != 0;
            trace_occurrence_policy_ok =
                trace.composite_producer_ &&
                trace.composite_producer_local_ordinal_ ==
                    expected_producer_local_ordinal &&
                trace_owner.get() == producer_owner.get() &&
                trace.expected_owner_generation_ ==
                    payload.expected_producer_owner_generation &&
                (unsealed_occurrence || stable_three_slot_occurrence);
            if (trace_occurrence_policy_ok) {
                if (replay_producer_owner == nullptr) {
                    replay_producer_owner = occurrence.owner;
                    replay_producer_allocation_hash =
                        trace.allocation_hash_;
                    replay_producer_capability =
                        trace.composite_capability_;
                    replay_producer_policy_fingerprint =
                        trace.composite_policy_fingerprint_;
                    replay_producer_composite_profile_hash =
                        trace.composite_occurrence_profile_hash_;
                } else {
                    trace_occurrence_policy_ok =
                        replay_producer_owner == occurrence.owner &&
                        replay_producer_allocation_hash ==
                            trace.allocation_hash_ &&
                        replay_producer_capability ==
                            trace.composite_capability_ &&
                        replay_producer_policy_fingerprint ==
                            trace.composite_policy_fingerprint_ &&
                        replay_producer_composite_profile_hash ==
                            trace.composite_occurrence_profile_hash_;
                }
            }
        } else {
            trace_occurrence_policy_ok =
                !trace.composite_producer_ &&
                trace.composite_producer_local_ordinal_ == 0 &&
                trace.composite_capability_ ==
                    SpmFmbCompositeOccurrenceCapability::Unsealed &&
                trace.composite_policy_fingerprint_ == 0 &&
                trace.composite_occurrence_profile_hash_ == 0 &&
                !trace.composite_preload_follower_;
        }
        TORCH_CHECK(
            occurrence.owner != nullptr && trace_owner != nullptr &&
                *trace_owner == trace.expected_owner_generation_ &&
                trace.composite_ &&
                trace.composite_identity_ == payload.composite_identity &&
                trace.composite_occurrence_ordinal_ == ordinal &&
                trace.composite_outer_begin_ == expected_outer_begin &&
                trace.graph_identity_ == &graph &&
                trace.graph_build_generation_ ==
                    payload.graph_build_generation &&
                trace.graph_signature_identity_ ==
                    payload.graph_signature_identity &&
                trace.graph_signature_segment_key_ ==
                    payload.graph_signature_segment_key &&
                trace.graph_topology_hash_ ==
                    payload.graph_topology_hash &&
                trace.graph_node_kind_mask_ ==
                    graph.build_node_kind_mask() &&
                (trace.graph_node_kind_mask_ &
                 kSpmFmbOpaqueGraphNodeKinds) == 0 &&
                trace.graph_node_begin_ >=
                    trace.composite_outer_begin_ &&
                trace.graph_node_begin_ < trace.graph_node_end_ &&
                trace.graph_node_end_ <= graph.graph_size() &&
                trace.callback_count_ ==
                    trace.callback_summaries_.size() &&
                trace_occurrence_policy_ok,
            "composite REPLAY occurrence identity or Graph window is "
            "stale");
        if (occurrence.yields) {
            const auto& yields = *occurrence.yields;
            TORCH_CHECK(
                yields.source_trace_.trace_hash_ == trace.trace_hash_ &&
                    yields.yield_hash_ != 0 &&
                    yields.semantic_id_digest_ ==
                        payload.semantic_yield_digest &&
                    yields.capability_ ==
                        SpmFmbLayerProducerYieldCapability::
                            CanonicalDenseReplicatedFp16 &&
                    yields.policy_fingerprint_ ==
                        trace.layer_producer_yield_policy_fingerprint_ &&
                    !yields.yields_.empty() &&
                    expected_yield_count <=
                        std::numeric_limits<size_t>::max() -
                            yields.yields_.size(),
                "composite REPLAY producer-yield payload is stale");
            expected_yield_count += yields.yields_.size();
            for (size_t yield_ordinal = 0;
                 yield_ordinal < yields.yields_.size();
                 ++yield_ordinal) {
                const auto& yield = yields.yields_[yield_ordinal];
                TORCH_CHECK(
                    yield.ordinal == yield_ordinal &&
                        yield.relative_node_begin <
                            yield.relative_node_end &&
                        yield.relative_node_end ==
                            yield.relative_node_begin + 1 &&
                        trace.graph_node_begin_ <=
                            std::numeric_limits<size_t>::max() -
                                yield.relative_node_end &&
                        trace.graph_node_begin_ +
                                yield.relative_node_end <=
                            trace.graph_node_end_ &&
                        unique_semantic_yield_ids
                            .insert(yield.semantic_id)
                            .second,
                    "composite REPLAY producer-yield marker is malformed "
                    "or duplicated");
                GraphSemanticSpmProducerYieldOccurrence expected;
                expected.relative_node_index =
                    trace.graph_node_begin_ +
                    yield.relative_node_begin;
                expected.semantic_yield_id = yield.semantic_id;
                expected.writer_kind =
                    static_cast<uint8_t>(yield.writer_kind);
                expected_graph_yields.push_back(std::move(expected));
            }
            ++expected_producer_local_ordinal;
        } else {
            if (replay_consumer_owner == nullptr) {
                replay_consumer_owner = occurrence.owner;
            }
            ++replay_consumer_occurrence_count;
            TORCH_CHECK(
                trace.layer_producer_yield_capability_ ==
                        SpmFmbLayerProducerYieldCapability::Unsealed &&
                    graph.semantic_spm_producer_yield_id_digest_for_window(
                        trace.composite_outer_begin_,
                        trace.graph_node_end_) == 0,
                "composite REPLAY consumer occurrence contains an "
                "unsealed producer marker");
        }
        expected_outer_begin = trace.graph_node_end_;
    }
    const auto observed_graph_yields =
        graph.semantic_spm_producer_yields_for_window(
            0, graph.graph_size());
    bool exact_yield_union =
        observed_graph_yields.size() == expected_graph_yields.size();
    if (exact_yield_union) {
        for (size_t index = 0;
             index < expected_graph_yields.size(); ++index) {
            const auto& expected = expected_graph_yields[index];
            const auto& observed = observed_graph_yields[index];
            if (expected.relative_node_index !=
                    observed.relative_node_index ||
                expected.semantic_yield_id !=
                    observed.semantic_yield_id ||
                expected.writer_kind != observed.writer_kind) {
                exact_yield_union = false;
                break;
            }
        }
    }
    TORCH_CHECK(
        expected_outer_begin == graph.graph_size() &&
            replay_producer_owner != nullptr &&
            expected_producer_local_ordinal > 0 &&
            (replay_producer_capability !=
                     SpmFmbCompositeOccurrenceCapability::
                         StableThreeInputSlotsSameOwnerPreload ||
             expected_producer_local_ordinal == 3) &&
            expected_yield_count ==
                pimpl_->expected_producer_yield_count &&
            exact_yield_union,
        "composite REPLAY token does not exactly cover the retained Graph "
        "and producer-yield marker union");
    graph.arm_semantic_spm_producer_yield_replay_for_fmb(
        producer_owner, payload.expected_producer_owner_generation,
        payload.semantic_yield_digest);
    pimpl_->mode = Impl::Mode::Replaying;
    pimpl_->graph = &graph;
    pimpl_->graph_build_generation = stamp.build_generation;
    pimpl_->graph_signature_identity = stamp.signature_identity;
    pimpl_->graph_signature_segment_key =
        stamp.signature_segment_key;
    pimpl_->composite_identity = payload.composite_identity;
    pimpl_->replay = sealed.payload_;
    pimpl_->producer_owner = replay_producer_owner;
    pimpl_->replay_consumer_owner = replay_consumer_owner;
    pimpl_->producer_owner_generation =
        payload.producer_owner_generation;
    pimpl_->expected_producer_owner_generation =
        payload.expected_producer_owner_generation;
    pimpl_->producer_layout_hash = 0;
    pimpl_->producer_allocation_hash =
        replay_producer_allocation_hash;
    pimpl_->producer_capability = replay_producer_capability;
    pimpl_->producer_policy_fingerprint =
        replay_producer_policy_fingerprint;
    pimpl_->producer_composite_profile_hash =
        replay_producer_composite_profile_hash;
    pimpl_->next_producer_occurrence =
        expected_producer_local_ordinal;
    pimpl_->replay_consumer_occurrence_count =
        replay_consumer_occurrence_count;
    pimpl_->replay_outer_fast_owner_order_exact =
        replay_outer_fast_owner_order_exact;
    pimpl_->next_replay_producer_occurrence = 0;
    pimpl_->next_replay_occurrence = 0;
    pimpl_->completed_replay_yields = 0;
    auto& graph_state = graph.composite_fmb_invocation_;
    graph_state.phase = GraphPhase::ReplayOpen;
    graph_state.coordinator = this;
    graph_state.identity = payload.composite_identity;
    graph_state.build_generation = stamp.build_generation;
    graph_state.signature_identity = stamp.signature_identity;
    graph_state.signature_segment_key = stamp.signature_segment_key;
    graph_state.expected_occurrences =
        payload.expected_occurrence_count;
    graph_state.completed_occurrences = 0;
    graph_state.expected_producer_yields =
        payload.expected_producer_yield_count;
    graph_state.completed_producer_yields = 0;
    graph_state.next_outer_begin = 0;
}

void SpmCompositeTraceCoordinator::begin_replay(
    const SpmFmbSealedCompositeTrace& sealed,
    RpuKernelGraph& graph,
    const SpmPipelineLease& lease) {
    TORCH_CHECK(
        pimpl_->physical_lease == &lease &&
            pimpl_->physical_producer_owner != nullptr &&
            pimpl_->physical_consumer_owner != nullptr &&
            lease.composite_committed(),
        "lease-aware composite REPLAY requires both adopted roles and one "
        "committed reservation");
    lease.validate_composite_execution_authority(
        lease.epoch(), lease.plan_hash(), /*require_committed=*/true);
    begin_replay(sealed, graph);
    pimpl_->physical_require_committed = true;
}

SpmCompositeOuterFastReplayTicket
SpmCompositeTraceCoordinator::validate_outer_fast_replay() {
    constexpr size_t kOuterFastOccurrenceCount = 4;
    constexpr size_t kOuterFastProducerCount = 3;
    constexpr size_t kOuterFastProducerYieldCount = 12;
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;

    TORCH_CHECK(
        pimpl_->owner_thread == std::this_thread::get_id() &&
            pimpl_->mode == Impl::Mode::Replaying &&
            pimpl_->graph != nullptr && pimpl_->replay &&
            pimpl_->active_scope_epoch == 0 &&
            pimpl_->active_owner == nullptr &&
            pimpl_->outer_fast_ticket_nonce == 0 &&
            pimpl_->next_replay_occurrence == 0 &&
            pimpl_->next_replay_producer_occurrence == 0 &&
            pimpl_->completed_replay_yields == 0,
        "composite outer-fast validation requires one pristine REPLAY with "
        "no active occurrence or existing ticket");

    const auto& payload = *pimpl_->replay;
    TORCH_CHECK(
        pimpl_->expected_occurrence_count ==
                kOuterFastOccurrenceCount &&
            payload.expected_occurrence_count ==
                kOuterFastOccurrenceCount &&
            payload.occurrences.size() == kOuterFastOccurrenceCount &&
            pimpl_->producer_capability ==
                SpmFmbCompositeOccurrenceCapability::
                    StableThreeInputSlotsSameOwnerPreload &&
            pimpl_->next_producer_occurrence ==
                kOuterFastProducerCount &&
            pimpl_->replay_consumer_occurrence_count == 1 &&
            pimpl_->replay_consumer_owner != nullptr &&
            pimpl_->replay_outer_fast_owner_order_exact &&
            pimpl_->expected_producer_yield_count ==
                kOuterFastProducerYieldCount &&
            payload.expected_producer_yield_count ==
                kOuterFastProducerYieldCount &&
            payload.producer_yield_count ==
                kOuterFastProducerYieldCount,
        "composite outer-fast validation admits only the sealed "
        "4-occurrence/3-producer/1-consumer/12-yield owner order");

    TORCH_CHECK(
        pimpl_->physical_lease != nullptr &&
            pimpl_->physical_require_committed &&
            pimpl_->physical_producer_owner != nullptr &&
            pimpl_->physical_consumer_owner != nullptr &&
            pimpl_->physical_producer_owner == pimpl_->producer_owner &&
            pimpl_->physical_consumer_owner ==
                pimpl_->replay_consumer_owner &&
            pimpl_->physical_lease->composite_committed(),
        "composite outer-fast validation requires both adopted component "
        "owners and one committed physical lease");
    pimpl_->physical_lease->validate_composite_execution_authority(
        pimpl_->physical_lease->epoch(),
        pimpl_->physical_lease->plan_hash(),
        /*require_committed=*/true);
    pimpl_->physical_producer_owner
        ->validate_spm_pipeline_composite_component(
            *pimpl_->physical_lease, /*producer=*/true);
    pimpl_->physical_consumer_owner
        ->validate_spm_pipeline_composite_component(
            *pimpl_->physical_lease, /*producer=*/false);

    RpuKernelGraph& graph = *pimpl_->graph;
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    const auto& graph_state = graph.composite_fmb_invocation_;
    const std::shared_ptr<uint64_t> producer_owner =
        pimpl_->producer_owner_generation.lock();
    const std::shared_ptr<const uint64_t> graph_producer_owner =
        graph.semantic_spm_producer_yield_owner_generation_.lock();
    TORCH_CHECK(
        RpuKernelGraph::has_active() &&
            &RpuKernelGraph::active() == &graph &&
            stamp.graph == &graph &&
            stamp.state == RpuKernelGraph::State::REPLAYING &&
            stamp.position == 0 &&
            stamp.build_generation == pimpl_->graph_build_generation &&
            stamp.signature_identity ==
                pimpl_->graph_signature_identity &&
            stamp.signature_segment_key ==
                pimpl_->graph_signature_segment_key &&
            graph_state.phase == GraphPhase::ReplayOpen &&
            graph_state.coordinator == this &&
            graph_state.identity == pimpl_->composite_identity &&
            graph_state.expected_occurrences ==
                kOuterFastOccurrenceCount &&
            graph_state.completed_occurrences == 0 &&
            graph_state.expected_producer_yields ==
                kOuterFastProducerYieldCount &&
            graph_state.completed_producer_yields == 0 &&
            graph_state.next_outer_begin == 0,
        "composite outer-fast validation requires the exact owning Graph "
        "at REPLAY cursor zero");
    TORCH_CHECK(
        !graph.has_semantic_dma_nodes_ &&
            !graph.has_semantic_spm_peer_nodes_ &&
            !graph.canonical_dma_poisoned_ &&
            !graph.canonical_dma_build_trace_active_ &&
            !graph.canonical_dma_callback_active_ &&
            !graph.canonical_dma_burst_permit_.active &&
            graph.has_semantic_spm_producer_yield_nodes_ &&
            graph.semantic_spm_producer_yield_replay_armed_ &&
            !graph.semantic_spm_producer_yield_poisoned_ &&
            !graph.pending_semantic_spm_producer_yield_.has_value() &&
            graph.semantic_spm_producer_yield_replay_arm_id_digest_ ==
                payload.semantic_yield_digest &&
            producer_owner != nullptr &&
            graph_producer_owner != nullptr &&
            producer_owner.get() == graph_producer_owner.get() &&
            *producer_owner ==
                pimpl_->expected_producer_owner_generation &&
            graph.semantic_spm_producer_yield_expected_owner_generation_ ==
                pimpl_->expected_producer_owner_generation,
        "composite outer-fast validation supports only the exact armed "
        "producer-yield Graph without semantic DMA or typed SPM peers");
    TORCH_CHECK(
        pimpl_->next_outer_fast_ticket_nonce !=
            std::numeric_limits<uint64_t>::max(),
        "composite outer-fast replay ticket nonce exhausted");

    const uint64_t nonce = ++pimpl_->next_outer_fast_ticket_nonce;
    pimpl_->outer_fast_ticket_nonce = nonce;
    return SpmCompositeOuterFastReplayTicket(
        this, pimpl_->lifetime, nonce, graph.graph_size(),
        kOuterFastOccurrenceCount, kOuterFastProducerCount,
        kOuterFastProducerYieldCount);
}

void SpmCompositeTraceCoordinator::commit_outer_fast_replay(
    SpmCompositeOuterFastReplayTicket&& ticket) noexcept {
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;

    // The Graph census commit is the only operation allowed between ticket
    // validation and this publication.  A violated caller contract is
    // fail-stop: returning a partially published composite would make the
    // retained Graph appear safe after an uncommitted op-stream skip.
    if (ticket.coordinator_ != this ||
        ticket.coordinator_lifetime_.expired() || ticket.nonce_ == 0 ||
        pimpl_->owner_thread != std::this_thread::get_id() ||
        pimpl_->outer_fast_ticket_nonce != ticket.nonce_ ||
        pimpl_->mode != Impl::Mode::Replaying ||
        pimpl_->graph == nullptr || !pimpl_->replay ||
        pimpl_->physical_lease == nullptr ||
        !pimpl_->physical_require_committed ||
        pimpl_->physical_producer_owner == nullptr ||
        pimpl_->physical_consumer_owner == nullptr ||
        pimpl_->physical_producer_owner != pimpl_->producer_owner ||
        pimpl_->physical_consumer_owner !=
            pimpl_->replay_consumer_owner ||
        pimpl_->active_scope_epoch != 0 || pimpl_->active_owner != nullptr ||
        pimpl_->next_replay_occurrence != 0 ||
        pimpl_->next_replay_producer_occurrence != 0 ||
        pimpl_->completed_replay_yields != 0 ||
        pimpl_->next_producer_occurrence != 3 ||
        pimpl_->replay_consumer_occurrence_count != 1 ||
        !pimpl_->replay_outer_fast_owner_order_exact ||
        pimpl_->expected_occurrence_count != 4 ||
        pimpl_->expected_producer_yield_count != 12 ||
        ticket.occurrence_count_ != 4 || ticket.producer_count_ != 3 ||
        ticket.producer_yield_count_ != 12) {
        std::terminate();
    }

    RpuKernelGraph& graph = *pimpl_->graph;
    auto& graph_state = graph.composite_fmb_invocation_;
    if (graph.state_ != RpuKernelGraph::State::REPLAYING ||
        graph.cursor_ != ticket.graph_extent_ ||
        graph.nodes_.size() != ticket.graph_extent_ ||
        !graph.kernel_register_census_active_ ||
        !graph.kernel_register_census_complete_ ||
        graph.kernel_register_census_poisoned_ ||
        !graph.kernel_register_census_invocation_armed_ ||
        !graph.kernel_register_outer_fast_hit_ ||
        graph.kernel_register_outer_fast_validation_armed_ ||
        graph.kernel_register_outer_fast_prepared_.has_value() ||
        !graph.op_stream_fully_skipped_ ||
        !graph.semantic_spm_producer_yield_replay_armed_ ||
        graph.semantic_spm_producer_yield_poisoned_ ||
        graph.pending_semantic_spm_producer_yield_.has_value() ||
        graph.has_semantic_dma_nodes_ ||
        graph.has_semantic_spm_peer_nodes_ ||
        graph_state.phase != GraphPhase::ReplayOpen ||
        graph_state.coordinator != this ||
        graph_state.identity != pimpl_->composite_identity ||
        graph_state.expected_occurrences != ticket.occurrence_count_ ||
        graph_state.expected_producer_yields !=
            ticket.producer_yield_count_ ||
        graph_state.completed_occurrences != 0 ||
        graph_state.completed_producer_yields != 0 ||
        graph_state.next_outer_begin != 0) {
        std::terminate();
    }

    // No allocation, owner walk, occurrence walk, or throwing validation is
    // permitted below.  Publish counters first and completion phases last.
    pimpl_->next_replay_occurrence = ticket.occurrence_count_;
    pimpl_->next_replay_producer_occurrence = ticket.producer_count_;
    pimpl_->completed_replay_yields = ticket.producer_yield_count_;
    graph_state.completed_occurrences = ticket.occurrence_count_;
    graph_state.completed_producer_yields =
        ticket.producer_yield_count_;
    graph_state.next_outer_begin = ticket.graph_extent_;
    pimpl_->outer_fast_ticket_nonce = 0;
    ticket.release();
    graph_state.phase = GraphPhase::ReplayComplete;
    pimpl_->mode = Impl::Mode::ReplayComplete;
}

SpmCompositeOccurrenceScope
SpmCompositeTraceCoordinator::begin_replay_occurrence(
    FusedModelBase& owner) {
    TORCH_CHECK(pimpl_->owner_thread == std::this_thread::get_id() &&
                    pimpl_->mode == Impl::Mode::Replaying &&
                    pimpl_->graph != nullptr && pimpl_->replay &&
                    pimpl_->outer_fast_ticket_nonce == 0 &&
                    pimpl_->active_scope_epoch == 0 &&
                    pimpl_->active_owner == nullptr &&
                    pimpl_->next_replay_occurrence <
                        pimpl_->replay->occurrences.size(),
                "composite REPLAY occurrence is duplicate or out of range");
    TORCH_CHECK(pimpl_->next_scope_epoch != 0 &&
                    pimpl_->next_scope_epoch !=
                        std::numeric_limits<uint64_t>::max(),
                "composite occurrence scope epoch exhausted");
    const auto& occurrence = pimpl_->replay->occurrences[
        pimpl_->next_replay_occurrence];
    const auto& trace = occurrence.trace;
    const std::shared_ptr<uint64_t> trace_owner =
        trace.owner_generation_.lock();
    const GraphOpStreamStamp stamp = pimpl_->graph->op_stream_stamp();
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;
    const auto& graph_state =
        pimpl_->graph->composite_fmb_invocation_;
    auto cancel_failed_begin = c10::make_scope_exit([&] {
        owner.cancel_spm_pipeline_composite_occurrence_runtime();
        owner.cancel_spm_pipeline_composite_physical_execution();
        owner.cancel_spm_pipeline_runtime_member_callback();
        owner.cancel_spm_pipeline_callback_yield_replay();
        owner.cancel_spm_pipeline_post_fn_yield_replay();
        if (pimpl_->graph != nullptr &&
            pimpl_->graph->composite_fmb_invocation_.coordinator == this) {
            pimpl_->graph->composite_fmb_invocation_.phase =
                GraphPhase::Poisoned;
            pimpl_->graph->composite_fmb_invocation_.coordinator = nullptr;
        }
        pimpl_->mode = Impl::Mode::Cancelled;
        pimpl_->active_scope_epoch = 0;
        pimpl_->active_owner = nullptr;
        pimpl_->active_is_producer = false;
        pimpl_->active_producer_local_ordinal = 0;
    });
    const auto& snapshot =
        owner.pimpl_->pipeline_manifest_snapshot_;
    const bool trace_is_producer = occurrence.yields != nullptr;
    if (pimpl_->physical_lease != nullptr) {
        TORCH_CHECK(
            (trace_is_producer &&
             pimpl_->physical_producer_owner == &owner) ||
                (!trace_is_producer &&
                 pimpl_->physical_consumer_owner == &owner),
            "composite REPLAY occurrence owner differs from its adopted "
            "physical role");
    }
    const uint64_t current_composite_profile_hash =
        trace_is_producer
        ? fmb_composite_occurrence_profile_hash(
              trace.profile_hash_, snapshot.layout_hash,
              snapshot.allocation_hash, snapshot.declaration_hash,
              snapshot.persistent_hash,
              owner.spm_fmb_preload_capability(),
              trace.composite_capability_,
              trace.composite_policy_fingerprint_)
        : 0;
    const bool producer_occurrence_policy_ok =
        !trace_is_producer ||
        (trace.composite_producer_ &&
         trace.composite_producer_local_ordinal_ ==
             pimpl_->next_replay_producer_occurrence &&
         pimpl_->producer_owner == &owner &&
         trace_owner.get() ==
             pimpl_->producer_owner_generation.lock().get() &&
         trace.expected_owner_generation_ ==
             pimpl_->expected_producer_owner_generation &&
         trace.composite_capability_ ==
             pimpl_->producer_capability &&
         trace.composite_capability_ ==
             owner.spm_fmb_composite_occurrence_capability() &&
         trace.composite_policy_fingerprint_ ==
             pimpl_->producer_policy_fingerprint &&
         trace.composite_policy_fingerprint_ ==
             owner.spm_fmb_composite_occurrence_policy_fingerprint() &&
         trace.composite_occurrence_profile_hash_ ==
             pimpl_->producer_composite_profile_hash &&
         trace.composite_occurrence_profile_hash_ ==
             current_composite_profile_hash &&
         ((trace.composite_capability_ ==
               SpmFmbCompositeOccurrenceCapability::Unsealed &&
           trace.composite_policy_fingerprint_ == 0 &&
           trace.composite_occurrence_profile_hash_ == 0 &&
           !trace.composite_preload_follower_) ||
          (trace.composite_capability_ ==
               SpmFmbCompositeOccurrenceCapability::
                   StableThreeInputSlotsSameOwnerPreload &&
           trace.composite_producer_local_ordinal_ < 3 &&
           trace.composite_preload_follower_ ==
               (trace.composite_producer_local_ordinal_ != 0) &&
           owner.spm_fmb_preload_capability() ==
               SpmFmbPreloadCapability::
                   PersistentOutsideResolvedWindow)));
    const bool consumer_occurrence_policy_ok =
        trace_is_producer ||
        (!trace.composite_producer_ &&
         trace.composite_producer_local_ordinal_ == 0 &&
         trace.composite_capability_ ==
             SpmFmbCompositeOccurrenceCapability::Unsealed &&
         trace.composite_policy_fingerprint_ == 0 &&
         trace.composite_occurrence_profile_hash_ == 0 &&
         !trace.composite_preload_follower_);
    TORCH_CHECK(trace_owner != nullptr &&
                    occurrence.owner == &owner &&
                    trace_owner.get() ==
                        owner.pimpl_->pipeline_manifest_generation_.get() &&
                    *trace_owner == trace.expected_owner_generation_ &&
                    owner.pimpl_->pipeline_manifest_snapshot_.valid &&
                    owner.pimpl_->pipeline_manifest_snapshot_
                            .allocation_hash == trace.allocation_hash_ &&
                    trace.post_fn_yield_capability_ ==
                        owner.spm_fmb_post_fn_yield_capability() &&
                    trace.post_fn_yield_policy_fingerprint_ ==
                        owner.spm_fmb_post_fn_yield_policy_fingerprint() &&
                    trace.layer_producer_yield_capability_ ==
                        owner.spm_fmb_layer_producer_yield_capability() &&
                    trace.layer_producer_yield_policy_fingerprint_ ==
                        owner.spm_fmb_layer_producer_yield_policy_fingerprint() &&
                    trace.active_group_capability_ ==
                        owner.spm_fmb_active_group_capability() &&
                    trace.dense_ddr_member_capability_ ==
                        owner.spm_fmb_dense_ddr_member_capability() &&
                    trace.dma_endpoint_capability_ ==
                        owner.spm_fmb_dma_endpoint_capability() &&
                    trace.runtime_member_capability_ ==
                        owner.spm_fmb_runtime_member_capability() &&
                    trace.spm_peer_capability_ ==
                        owner.spm_fmb_spm_peer_capability() &&
                    trace.spm_peer_policy_fingerprint_ ==
                        owner.spm_fmb_spm_peer_policy_fingerprint() &&
                    (trace.dense_ddr_member_capability_ ==
                         SpmFmbDenseDdrMemberCapability::Unsealed ||
                     (trace.resolved_hidden_size_ == owner.pimpl_->hidden_size_ &&
                      trace.resolved_num_layers_ == owner.pimpl_->num_layers_)) &&
                    trace.composite_occurrence_ordinal_ ==
                        pimpl_->next_replay_occurrence &&
                    producer_occurrence_policy_ok &&
                    consumer_occurrence_policy_ok &&
                    graph_state.phase == GraphPhase::ReplayOpen &&
                    graph_state.coordinator == this &&
                    graph_state.identity ==
                        pimpl_->composite_identity &&
                    graph_state.completed_occurrences ==
                        pimpl_->next_replay_occurrence &&
                    graph_state.next_outer_begin ==
                        trace.composite_outer_begin_ &&
                    stamp.state == RpuKernelGraph::State::REPLAYING &&
                    stamp.position == trace.composite_outer_begin_ &&
                    stamp.build_generation ==
                        pimpl_->graph_build_generation &&
                    stamp.signature_identity ==
                        pimpl_->graph_signature_identity &&
                    stamp.signature_segment_key ==
                        pimpl_->graph_signature_segment_key,
                "composite REPLAY occurrence owner, model policy, order, "
                "or cursor drifted");
    FusedModelBase::Impl::RuntimeCallbackYieldReplayAuthority
        next_callback_authority;
    if (occurrence.yields) {
        TORCH_CHECK(
            occurrence.yields->capability_ ==
                    SpmFmbLayerProducerYieldCapability::
                        CanonicalDenseReplicatedFp16 &&
                occurrence.yields->policy_fingerprint_ ==
                    owner.spm_fmb_layer_producer_yield_policy_fingerprint() &&
                occurrence.yields->source_trace_.trace_hash_ ==
                    trace.trace_hash_,
            "composite REPLAY producer occurrence policy drifted");
        auto& authority =
            owner.pimpl_->pipeline_callback_yield_replay_authority_;
        TORCH_CHECK(!authority.armed && !authority.yield_open &&
                        authority.yields == nullptr &&
                        !owner.pimpl_
                             ->pipeline_post_fn_yield_replay_authority_.armed,
                    "composite REPLAY component producer authority "
                    "conflicts with an existing arm");
        next_callback_authority.armed = true;
        next_callback_authority.graph_identity = pimpl_->graph;
        next_callback_authority.graph_build_generation =
            stamp.build_generation;
        next_callback_authority.graph_signature_identity =
            stamp.signature_identity;
        next_callback_authority.profile_hash =
            stamp.signature_segment_key;
        next_callback_authority.yields =
            std::make_unique<SpmFmbSealedCallbackYields>(
            *occurrence.yields);
    }
    if (trace_is_producer &&
        trace.composite_capability_ !=
            SpmFmbCompositeOccurrenceCapability::Unsealed) {
        owner.arm_spm_pipeline_composite_occurrence_runtime(
            pimpl_->composite_identity,
            trace.composite_producer_local_ordinal_,
            trace.composite_capability_,
            trace.composite_policy_fingerprint_,
            snapshot.layout_hash, snapshot.allocation_hash,
            snapshot.persistent_generation,
            trace.composite_occurrence_profile_hash_,
            trace.composite_preload_follower_);
    }
    if (occurrence.yields) {
        owner.pimpl_->pipeline_callback_yield_replay_authority_ =
            std::move(next_callback_authority);
    }
    if (pimpl_->physical_lease != nullptr) {
        owner.arm_spm_pipeline_composite_physical_execution(
            *pimpl_->physical_lease, pimpl_->composite_identity,
            trace_is_producer,
            trace_is_producer
                ? trace.composite_producer_local_ordinal_
                : 0,
            /*require_committed=*/true);
    }
    if (trace_is_producer &&
        pimpl_->next_replay_producer_occurrence == 0) {
        pimpl_->producer_layout_hash = snapshot.layout_hash;
    } else if (trace_is_producer) {
        TORCH_CHECK(pimpl_->producer_layout_hash == snapshot.layout_hash,
                    "composite REPLAY producer layout drifted");
    }
    const uint64_t epoch = pimpl_->next_scope_epoch++;
    pimpl_->active_scope_epoch = epoch;
    pimpl_->active_owner = &owner;
    pimpl_->active_is_producer = trace_is_producer;
    pimpl_->active_producer_local_ordinal =
        trace_is_producer
        ? trace.composite_producer_local_ordinal_
        : 0;
    cancel_failed_begin.release();
    return SpmCompositeOccurrenceScope(
        this, pimpl_->lifetime, epoch);
}

void SpmCompositeTraceCoordinator::finish_occurrence(
    uint64_t scope_epoch) {
    TORCH_CHECK(pimpl_->owner_thread == std::this_thread::get_id() &&
                    pimpl_->active_scope_epoch == scope_epoch &&
                    scope_epoch != 0 && pimpl_->active_owner != nullptr &&
                    pimpl_->graph != nullptr,
                "composite occurrence finish has a stale scope token");
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;
    auto& graph_state = pimpl_->graph->composite_fmb_invocation_;
    if (pimpl_->mode == Impl::Mode::Building) {
        SpmFmbCompletedBuildTrace completed =
            pimpl_->active_owner
                ->complete_spm_pipeline_composite_build_trace();
        TORCH_INTERNAL_ASSERT(completed.payload_ != nullptr);
        const auto& candidate = completed.payload_->candidate;
        size_t producer_yield_count = 0;
        for (const auto& callback : candidate.callbacks) {
            TORCH_CHECK(
                producer_yield_count <=
                    std::numeric_limits<size_t>::max() -
                        callback.producer_yields.size(),
                "composite BUILD producer-yield count overflow");
            producer_yield_count += callback.producer_yields.size();
        }
        const GraphOpStreamStamp stamp =
            pimpl_->graph->op_stream_stamp();
        const auto& snapshot = pimpl_->active_owner->pimpl_
                                   ->pipeline_manifest_snapshot_;
        const std::shared_ptr<uint64_t> active_owner_generation =
            pimpl_->active_owner->pimpl_
                ->pipeline_manifest_generation_;
        const bool current_is_producer =
            pimpl_->active_owner
                    ->spm_fmb_layer_producer_yield_capability() !=
                SpmFmbLayerProducerYieldCapability::Unsealed;
        const auto current_composite_capability =
            pimpl_->active_owner
                ->spm_fmb_composite_occurrence_capability();
        const uint64_t current_composite_policy =
            pimpl_->active_owner
                ->spm_fmb_composite_occurrence_policy_fingerprint();
        const uint64_t current_composite_profile_hash =
            fmb_composite_occurrence_profile_hash(
                candidate.profile_hash, snapshot.layout_hash,
                snapshot.allocation_hash, snapshot.declaration_hash,
                snapshot.persistent_hash,
                pimpl_->active_owner->spm_fmb_preload_capability(),
                candidate.composite_capability,
                candidate.composite_policy_fingerprint);
        const bool stable_three_slot_occurrence =
            candidate.composite_capability ==
                SpmFmbCompositeOccurrenceCapability::
                    StableThreeInputSlotsSameOwnerPreload;
        TORCH_CHECK(
            completed.occurrence_ordinal() ==
                    pimpl_->completed.size() &&
                snapshot.valid && active_owner_generation != nullptr &&
                *active_owner_generation ==
                    candidate.expected_owner_generation &&
                snapshot.allocation_hash ==
                    candidate.allocation_hash &&
                current_is_producer ==
                    candidate.composite_producer &&
                current_composite_capability ==
                    candidate.composite_capability &&
                current_composite_policy ==
                    candidate.composite_policy_fingerprint &&
                current_composite_profile_hash ==
                    candidate.composite_occurrence_profile_hash &&
                (!stable_three_slot_occurrence ||
                 (candidate.composite_producer &&
                  candidate.composite_producer_local_ordinal < 3 &&
                  candidate.composite_preload_follower ==
                      (candidate.composite_producer_local_ordinal != 0) &&
                  (candidate.composite_preload_follower
                       ? candidate.composite_outer_begin ==
                             candidate.graph_node_begin
                       : (candidate.cpu_dry ||
                          candidate.composite_outer_begin <
                              candidate.graph_node_begin)))) &&
                candidate.composite_producer ==
                    pimpl_->active_is_producer &&
                (!candidate.composite_producer ||
                 candidate.composite_producer_local_ordinal ==
                     pimpl_->active_producer_local_ordinal) &&
                graph_state.phase == GraphPhase::BuildOpen &&
                graph_state.coordinator == this &&
                graph_state.identity == pimpl_->composite_identity &&
                graph_state.build_generation ==
                    pimpl_->graph_build_generation &&
                graph_state.signature_identity ==
                    pimpl_->graph_signature_identity &&
                graph_state.signature_segment_key ==
                    pimpl_->graph_signature_segment_key &&
                graph_state.completed_occurrences ==
                    pimpl_->completed.size() &&
                graph_state.next_outer_begin ==
                    candidate.composite_outer_begin &&
                graph_state.completed_producer_yields <=
                    graph_state.expected_producer_yields &&
                producer_yield_count <=
                    graph_state.expected_producer_yields -
                        graph_state.completed_producer_yields &&
                stamp.state == RpuKernelGraph::State::RECORDING &&
                stamp.position ==
                    candidate.callbacks.back().graph_node_end &&
                stamp.build_generation ==
                    pimpl_->graph_build_generation &&
                stamp.signature_identity ==
                    pimpl_->graph_signature_identity &&
                stamp.signature_segment_key ==
                    pimpl_->graph_signature_segment_key,
            "composite BUILD occurrence order, yield count, or outer Graph "
            "window drifted at finish");
        TORCH_CHECK(
            !pimpl_->active_is_producer ||
                pimpl_->next_producer_occurrence <
                    std::numeric_limits<size_t>::max(),
            "composite BUILD producer-local cursor overflow");
        const size_t producer_count_after =
            pimpl_->next_producer_occurrence +
            (pimpl_->active_is_producer ? 1 : 0);
        const size_t occurrence_count_after =
            graph_state.completed_occurrences + 1;
        TORCH_CHECK(
            occurrence_count_after < graph_state.expected_occurrences ||
                (occurrence_count_after ==
                     graph_state.expected_occurrences &&
                 (pimpl_->producer_capability !=
                      SpmFmbCompositeOccurrenceCapability::
                          StableThreeInputSlotsSameOwnerPreload ||
                  producer_count_after == 3)),
            "composite BUILD stable-three policy requires exactly three "
            "producer occurrences before Graph submission");
        if (pimpl_->physical_lease != nullptr) {
            pimpl_->active_owner
                ->finish_spm_pipeline_composite_physical_execution(
                    *pimpl_->physical_lease,
                    pimpl_->composite_identity,
                    pimpl_->active_is_producer,
                    pimpl_->active_is_producer
                        ? candidate.composite_producer_local_ordinal
                        : 0);
        }
        if (candidate.composite_producer &&
            candidate.composite_capability !=
                SpmFmbCompositeOccurrenceCapability::Unsealed) {
            pimpl_->active_owner
                ->finish_spm_pipeline_composite_occurrence_runtime(
                    pimpl_->composite_identity,
                    candidate.composite_producer_local_ordinal);
        }
        const size_t occurrence_end = stamp.position;
        pimpl_->completed.push_back(std::move(completed));
        if (pimpl_->active_is_producer) {
            TORCH_CHECK(
                pimpl_->active_producer_local_ordinal ==
                    pimpl_->next_producer_occurrence &&
                    pimpl_->next_producer_occurrence <
                        std::numeric_limits<size_t>::max(),
                "composite BUILD producer-local cursor drifted");
            ++pimpl_->next_producer_occurrence;
        }
        ++graph_state.completed_occurrences;
        graph_state.completed_producer_yields += producer_yield_count;
        graph_state.next_outer_begin = occurrence_end;
        if (graph_state.completed_occurrences ==
            graph_state.expected_occurrences) {
            graph_state.phase = GraphPhase::BuildComplete;
        }
    } else {
        TORCH_CHECK(pimpl_->mode == Impl::Mode::Replaying &&
                        pimpl_->replay &&
                        pimpl_->next_replay_occurrence <
                            pimpl_->replay->occurrences.size(),
                    "composite occurrence finish is not in BUILD/REPLAY");
        const auto& occurrence = pimpl_->replay->occurrences[
            pimpl_->next_replay_occurrence];
        const GraphOpStreamStamp stamp = pimpl_->graph->op_stream_stamp();
        const size_t producer_yield_count =
            occurrence.yields ? occurrence.yields->yields_.size() : 0;
        TORCH_CHECK(graph_state.phase == GraphPhase::ReplayOpen &&
                        graph_state.coordinator == this &&
                        graph_state.identity ==
                            pimpl_->composite_identity &&
                        graph_state.completed_occurrences ==
                            pimpl_->next_replay_occurrence &&
                        graph_state.next_outer_begin ==
                            occurrence.trace.composite_outer_begin_ &&
                        pimpl_->active_is_producer ==
                            occurrence.trace.composite_producer_ &&
                        (!pimpl_->active_is_producer ||
                         pimpl_->active_producer_local_ordinal ==
                             occurrence.trace
                                 .composite_producer_local_ordinal_) &&
                        graph_state.completed_producer_yields <=
                            graph_state.expected_producer_yields &&
                        producer_yield_count <=
                            graph_state.expected_producer_yields -
                                graph_state.completed_producer_yields &&
                        stamp.state == RpuKernelGraph::State::REPLAYING &&
                        stamp.position ==
                            occurrence.trace.graph_node_end_ &&
                        !pimpl_->active_owner->pimpl_
                             ->pipeline_runtime_member_frame_.active &&
                        g_fmb_runtime_member_owner !=
                            pimpl_->active_owner->pimpl_.get() &&
                        (!occurrence.yields ||
                         (!pimpl_->active_owner->pimpl_
                               ->pipeline_callback_yield_replay_authority_
                               .armed &&
                          !pimpl_->active_owner->pimpl_
                               ->pipeline_callback_yield_replay_authority_
                               .yield_open &&
                          pimpl_->active_owner->pimpl_
                                  ->pipeline_callback_yield_replay_authority_
                                  .yields == nullptr)),
                    "composite REPLAY occurrence did not consume its exact "
                    "window or leave producer authority closed");
        if (pimpl_->physical_lease != nullptr) {
            pimpl_->active_owner
                ->finish_spm_pipeline_composite_physical_execution(
                    *pimpl_->physical_lease,
                    pimpl_->composite_identity,
                    pimpl_->active_is_producer,
                    pimpl_->active_is_producer
                        ? occurrence.trace
                              .composite_producer_local_ordinal_
                        : 0);
        }
        if (occurrence.trace.composite_producer_ &&
            occurrence.trace.composite_capability_ !=
                SpmFmbCompositeOccurrenceCapability::Unsealed) {
            pimpl_->active_owner
                ->finish_spm_pipeline_composite_occurrence_runtime(
                    pimpl_->composite_identity,
                    occurrence.trace
                        .composite_producer_local_ordinal_);
        }
        if (pimpl_->active_is_producer) {
            TORCH_CHECK(
                pimpl_->active_producer_local_ordinal ==
                    pimpl_->next_replay_producer_occurrence &&
                    pimpl_->next_replay_producer_occurrence <
                        std::numeric_limits<size_t>::max(),
                "composite REPLAY producer-local cursor drifted");
            ++pimpl_->next_replay_producer_occurrence;
        }
        ++pimpl_->next_replay_occurrence;
        ++graph_state.completed_occurrences;
        graph_state.completed_producer_yields += producer_yield_count;
        graph_state.next_outer_begin = stamp.position;
        pimpl_->completed_replay_yields += producer_yield_count;
    }
    pimpl_->active_scope_epoch = 0;
    pimpl_->active_owner = nullptr;
    pimpl_->active_is_producer = false;
    pimpl_->active_producer_local_ordinal = 0;
}

void SpmCompositeTraceCoordinator::finish_replay() {
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;
    TORCH_CHECK(pimpl_->owner_thread == std::this_thread::get_id() &&
                    pimpl_->mode == Impl::Mode::Replaying &&
                    pimpl_->graph != nullptr && pimpl_->replay &&
                    pimpl_->outer_fast_ticket_nonce == 0 &&
                    pimpl_->active_scope_epoch == 0 &&
                    pimpl_->active_owner == nullptr &&
                    pimpl_->next_replay_occurrence ==
                        pimpl_->replay->occurrences.size(),
                "composite REPLAY finish requires every occurrence exactly "
                "once");
    const GraphOpStreamStamp stamp = pimpl_->graph->op_stream_stamp();
    auto& graph_state =
        pimpl_->graph->composite_fmb_invocation_;
    TORCH_CHECK(stamp.state == RpuKernelGraph::State::REPLAYING &&
                    stamp.position == pimpl_->graph->graph_size() &&
                    graph_state.phase == GraphPhase::ReplayOpen &&
                    graph_state.coordinator == this &&
                    graph_state.identity ==
                        pimpl_->composite_identity &&
                    graph_state.completed_occurrences ==
                        graph_state.expected_occurrences &&
                    graph_state.completed_producer_yields ==
                        graph_state.expected_producer_yields &&
                    pimpl_->completed_replay_yields ==
                        pimpl_->expected_producer_yield_count &&
                    pimpl_->next_replay_producer_occurrence ==
                        pimpl_->next_producer_occurrence &&
                    (pimpl_->producer_capability !=
                             SpmFmbCompositeOccurrenceCapability::
                                 StableThreeInputSlotsSameOwnerPreload ||
                     pimpl_->next_replay_producer_occurrence == 3) &&
                    graph_state.next_outer_begin ==
                        pimpl_->graph->graph_size(),
                "composite REPLAY left an unclaimed outer Graph suffix");
    graph_state.phase = GraphPhase::ReplayComplete;
    pimpl_->mode = Impl::Mode::ReplayComplete;
}

void SpmCompositeTraceCoordinator::cancel_outer_fast_replay_ticket(
    uint64_t nonce) noexcept {
    if (!pimpl_ ||
        pimpl_->owner_thread != std::this_thread::get_id() ||
        nonce == 0 || pimpl_->outer_fast_ticket_nonce != nonce) {
        return;
    }
    pimpl_->outer_fast_ticket_nonce = 0;
    cancel();
}

void SpmCompositeTraceCoordinator::cancel_occurrence(
    uint64_t scope_epoch) noexcept {
    if (pimpl_->owner_thread != std::this_thread::get_id()) return;
    if (scope_epoch == 0 ||
        pimpl_->active_scope_epoch != scope_epoch) {
        return;
    }
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;
    if (pimpl_->graph != nullptr &&
        pimpl_->graph->composite_fmb_invocation_.coordinator == this) {
        pimpl_->graph->composite_fmb_invocation_.phase =
            GraphPhase::Poisoned;
        pimpl_->graph->composite_fmb_invocation_.coordinator = nullptr;
    }
    if (pimpl_->graph != nullptr && RpuKernelGraph::has_active() &&
        &RpuKernelGraph::active() == pimpl_->graph) {
        pimpl_->graph->poison_semantic_spm_producer_yield_for_fmb();
    }
    if (pimpl_->active_owner != nullptr) {
        clear_pipeline_build_trace_noexcept(*pimpl_->active_owner->pimpl_);
        pimpl_->active_owner
            ->cancel_spm_pipeline_composite_occurrence_runtime();
        pimpl_->active_owner
            ->cancel_spm_pipeline_composite_physical_execution();
        pimpl_->active_owner
            ->cancel_spm_pipeline_runtime_member_callback();
        pimpl_->active_owner
            ->cancel_spm_pipeline_callback_yield_replay();
        pimpl_->active_owner
            ->cancel_spm_pipeline_post_fn_yield_replay();
    }
    pimpl_->active_scope_epoch = 0;
    pimpl_->active_owner = nullptr;
    pimpl_->active_is_producer = false;
    pimpl_->active_producer_local_ordinal = 0;
    pimpl_->mode = Impl::Mode::Cancelled;
}

void SpmCompositeTraceCoordinator::cancel() noexcept {
    if (!pimpl_) return;
    if (pimpl_->owner_thread != std::this_thread::get_id()) {
        const bool transaction_open =
            pimpl_->active_scope_epoch != 0 || pimpl_->graph != nullptr ||
            pimpl_->mode == Impl::Mode::Building ||
            pimpl_->mode == Impl::Mode::Replaying ||
            pimpl_->mode == Impl::Mode::ReplayComplete;
        // The active FMB authorities are thread_local on owner_thread.  A
        // foreign thread cannot clear them or safely mutate its live Graph;
        // continuing would leave dangling coordinator identity in both.
        if (transaction_open) std::terminate();
        pimpl_->mode = Impl::Mode::Cancelled;
        pimpl_->outer_fast_ticket_nonce = 0;
        return;
    }
    using GraphPhase =
        RpuKernelGraph::CompositeFmbInvocationState::Phase;
    if (pimpl_->active_scope_epoch != 0) {
        cancel_occurrence(pimpl_->active_scope_epoch);
    } else if (pimpl_->mode == Impl::Mode::Building ||
               pimpl_->mode == Impl::Mode::Replaying ||
               pimpl_->mode == Impl::Mode::ReplayComplete) {
        const bool graph_is_active =
            pimpl_->graph != nullptr && RpuKernelGraph::has_active() &&
            &RpuKernelGraph::active() == pimpl_->graph;
        if (pimpl_->graph != nullptr &&
            pimpl_->graph->composite_fmb_invocation_.coordinator == this) {
            if (pimpl_->mode == Impl::Mode::ReplayComplete &&
                !graph_is_active &&
                pimpl_->graph->composite_fmb_invocation_.phase ==
                    GraphPhase::ReplayComplete) {
                pimpl_->graph->composite_fmb_invocation_ = {};
            } else {
                pimpl_->graph->composite_fmb_invocation_.phase =
                    GraphPhase::Poisoned;
                pimpl_->graph->composite_fmb_invocation_.coordinator =
                    nullptr;
            }
        }
        if (graph_is_active) {
            pimpl_->graph->poison_semantic_spm_producer_yield_for_fmb();
        } else if (pimpl_->mode == Impl::Mode::Building &&
                   pimpl_->graph != nullptr) {
            try {
                pimpl_->graph->invalidate();
            } catch (...) {
                // Keep a retained Graph poisoned until its physical owner can
                // perform exact deferred eviction.
            }
        }
        pimpl_->mode = Impl::Mode::Cancelled;
    }
    pimpl_->outer_fast_ticket_nonce = 0;
    pimpl_->replay_consumer_owner = nullptr;
    pimpl_->replay_consumer_occurrence_count = 0;
    pimpl_->replay_outer_fast_owner_order_exact = false;
    pimpl_->completed.clear();
    pimpl_->replay.reset();
}

size_t SpmCompositeTraceCoordinator::expected_occurrence_count() const {
    return pimpl_->expected_occurrence_count;
}

size_t SpmCompositeTraceCoordinator::completed_occurrence_count() const {
    if (pimpl_->mode == Impl::Mode::Replaying ||
        pimpl_->mode == Impl::Mode::ReplayComplete) {
        return pimpl_->next_replay_occurrence;
    }
    return pimpl_->completed.size();
}

namespace {

void validate_dense_spm_peer_callback_complete(
    const FusedModelBase::Impl::RuntimeDenseMemberFrame& frame);

}  // namespace

// =============================================================================
// Constructor / destructor
// =============================================================================

FusedModelBase::FusedModelBase() : pimpl_(std::make_unique<Impl>()) {
    pimpl_->global_fast_replay_enabled_ =
        env_enabled("RPU_WALL_OSS_FAST_REPLAY", /*reject_false_prefix=*/true);
    pimpl_->deep_fast_replay_enabled_ =
        env_enabled("RPU_DEEP_FAST_REPLAY", /*reject_false_prefix=*/false);
    // Register with SpmAllocator so super_persistent is released
    // when the last live instance is destroyed (prevents cross-load SPM leak).
    SPM_ALLOC.register_instance();
}

FusedModelBase::~FusedModelBase() {
    if (g_fmb_build_trace_owner == pimpl_.get()) {
        g_fmb_build_trace_owner = nullptr;
        g_fmb_build_trace_callback_begin.reset();
        g_fmb_post_fn_yield_begin.reset();
    }
    if (g_fmb_runtime_member_owner == pimpl_.get()) {
        g_fmb_runtime_member_owner = nullptr;
    }
    // GraphCache ownership lives in the Python adapter; destroying this C++
    // model handle has no graph-cache eviction responsibility.
    // Unregister; when the count drops to 0, SpmAllocator releases the
    // super_persistent floor (sp_floor_ → SPM_USABLE).
    SPM_ALLOC.unregister_instance();
}

// =============================================================================
// Narrow facade — delegating one-liners
// =============================================================================

InferenceContext&       FusedModelBase::ctx()       { return pimpl_->ctx_; }
const InferenceContext& FusedModelBase::ctx() const { return pimpl_->ctx_; }
SdpaStableMaskCache& FusedModelBase::sdpa_stable_mask_cache() {
    return pimpl_->sdpa_stable_mask_cache_;
}

uint32_t FusedModelBase::addr(int core, const char* name) const {
    uint32_t cpu_dry_offset = 0;
    if (C10_UNLIKELY(resolve_cpu_yield_replay_offset(
            *pimpl_, name, /*layer=*/-1, &cpu_dry_offset))) {
        return cpu_dry_offset;
    }
    if (C10_UNLIKELY(pimpl_->pipeline_build_trace_.armed ||
                     g_fmb_build_trace_owner != nullptr) &&
        record_pipeline_build_access(
            *pimpl_, name, /*layer=*/-1,
            SpmFmbBuildTraceAccessorKind::Addr, core,
            &cpu_dry_offset)) {
        return cpu_dry_offset;
    }
    auto it = pimpl_->offsets_.find(name);
    TORCH_CHECK(it != pimpl_->offsets_.end(), "FusedModelBase::addr: buffer '", name, "' not found");
    const uint32_t base = pimpl_->pipeline_scratch_base_ != 0 &&
            pimpl_->pipeline_temporary_names_.count(name)
        ? pimpl_->pipeline_scratch_base_ : 0;
    return SPM_ALLOC.addr(core, base + it->second);
}

uint32_t FusedModelBase::layer_addr(int layer, int core, const char* name) const {
    uint32_t cpu_dry_offset = 0;
    if (C10_UNLIKELY(resolve_cpu_yield_replay_offset(
            *pimpl_, name, layer, &cpu_dry_offset))) {
        return cpu_dry_offset;
    }
    if (C10_UNLIKELY(pimpl_->pipeline_build_trace_.armed ||
                     g_fmb_build_trace_owner != nullptr) &&
        record_pipeline_build_access(
            *pimpl_, name, layer,
            SpmFmbBuildTraceAccessorKind::LayerAddr, core,
            &cpu_dry_offset)) {
        return cpu_dry_offset;
    }
    TORCH_CHECK(layer >= 0 && layer < (int)pimpl_->per_layer_offsets_.size(),
                "FusedModelBase::layer_addr: layer ", layer, " out of range [0,",
                pimpl_->per_layer_offsets_.size(), ")");
    auto it = pimpl_->per_layer_offsets_[layer].find(name);
    TORCH_CHECK(it != pimpl_->per_layer_offsets_[layer].end(),
                "FusedModelBase::layer_addr: buffer '", name, "' not found at layer ", layer);
    const uint32_t base = pimpl_->pipeline_scratch_base_ != 0 &&
            pimpl_->pipeline_temporary_per_layer_names_.count(name)
        ? pimpl_->pipeline_scratch_base_ : 0;
    return SPM_ALLOC.addr(core, base + it->second);
}

SpmOffset FusedModelBase::addr_offset(const char* name) const {
    uint32_t cpu_dry_offset = 0;
    if (C10_UNLIKELY(resolve_cpu_yield_replay_offset(
            *pimpl_, name, /*layer=*/-1, &cpu_dry_offset))) {
        return SpmOffset{cpu_dry_offset};
    }
    if (C10_UNLIKELY(pimpl_->pipeline_build_trace_.armed ||
                     g_fmb_build_trace_owner != nullptr) &&
        record_pipeline_build_access(
            *pimpl_, name, /*layer=*/-1,
            SpmFmbBuildTraceAccessorKind::AddrOffset, /*core=*/0,
            &cpu_dry_offset)) {
        return SpmOffset{cpu_dry_offset};
    }
    auto it = pimpl_->offsets_.find(name);
    TORCH_CHECK(it != pimpl_->offsets_.end(),
                "FusedModelBase::addr_offset: buffer '", name, "' not found");
    const uint32_t base = pimpl_->pipeline_scratch_base_ != 0 &&
            pimpl_->pipeline_temporary_names_.count(name)
        ? pimpl_->pipeline_scratch_base_ : 0;
    return SpmOffset{base + it->second};
}

SpmOffset FusedModelBase::layer_addr_offset(int layer, const char* name) const {
    uint32_t cpu_dry_offset = 0;
    if (C10_UNLIKELY(resolve_cpu_yield_replay_offset(
            *pimpl_, name, layer, &cpu_dry_offset))) {
        return SpmOffset{cpu_dry_offset};
    }
    if (C10_UNLIKELY(pimpl_->pipeline_build_trace_.armed ||
                     g_fmb_build_trace_owner != nullptr) &&
        record_pipeline_build_access(
            *pimpl_, name, layer,
            SpmFmbBuildTraceAccessorKind::LayerAddrOffset, /*core=*/0,
            &cpu_dry_offset)) {
        return SpmOffset{cpu_dry_offset};
    }
    TORCH_CHECK(layer >= 0 && layer < (int)pimpl_->per_layer_offsets_.size(),
                "FusedModelBase::layer_addr_offset: layer ", layer, " out of range");
    auto it = pimpl_->per_layer_offsets_[layer].find(name);
    TORCH_CHECK(it != pimpl_->per_layer_offsets_[layer].end(),
                "FusedModelBase::layer_addr_offset: buffer '", name, "' not found at layer ", layer);
    const uint32_t base = pimpl_->pipeline_scratch_base_ != 0 &&
            pimpl_->pipeline_temporary_per_layer_names_.count(name)
        ? pimpl_->pipeline_scratch_base_ : 0;
    return SpmOffset{base + it->second};
}

at::Tensor&       FusedModelBase::post_output_tensor()       { return pimpl_->post_output_tensor_; }
const at::Tensor& FusedModelBase::post_output_tensor() const { return pimpl_->post_output_tensor_; }

c10::Half* FusedModelBase::output_ptr() {
    return pimpl_->output_tensor_.data_ptr<c10::Half>();
}
const at::Tensor& FusedModelBase::output_tensor() const { return pimpl_->output_tensor_; }

void FusedModelBase::set_chunk_size_override(int64_t cs) {
    TORCH_CHECK(cs == 0 || (cs >= 16 && cs % 16 == 0),
                "set_chunk_size_override: chunk size must be 0 (auto) or a "
                "positive multiple of 16, got ", cs);
    pimpl_->chunk_size_override_ = cs;
}
int64_t FusedModelBase::get_chunk_size_override() const  { return pimpl_->chunk_size_override_; }
int64_t FusedModelBase::get_last_resolved_chunk_size() const { return pimpl_->last_resolved_chunk_size_; }

// Certified chunk envelope.
void FusedModelBase::set_chunk_envelope(int64_t max_kv_len, int64_t chunk) {
    TORCH_CHECK(max_kv_len > 0,
                "set_chunk_envelope: max_kv_len must be > 0 (an envelope with "
                "max_kv_len=0 is the UNDECLARED sentinel and cannot be set "
                "explicitly), got ", max_kv_len);
    TORCH_CHECK(chunk == 0 || (chunk >= 16 && chunk % 16 == 0),
                "set_chunk_envelope: chunk must be 0 (auto certified) or a "
                "positive multiple of 16, got ",
                chunk);
    // Same cold contract as set_configured_chunk_size_cap: the envelope keys the
    // SPM layout and the GraphCache entries built from it, so a live handle must
    // not be hot-switched.
    TORCH_CHECK(get_last_resolved_chunk_size() == 0,
                "set_chunk_envelope must be called before the first forward");
    pimpl_->chunk_envelope_ = ChunkEnvelope{max_kv_len, chunk};
}

const FusedModelBase::ChunkEnvelope& FusedModelBase::chunk_envelope() const {
    return pimpl_->chunk_envelope_;
}

int64_t FusedModelBase::enforce_chunk_envelope(int64_t seq_len, int64_t position,
                                               const char* model_name) const {
    const ChunkEnvelope& e = pimpl_->chunk_envelope_;

    // Decode is exempt BY MECHANISM, not by convenience: a single-token step
    // resolves to the 16-row minimum, so it cannot reach a chunk large enough to
    // bust SPM. Gating it would only add churn to every decode-only harness
    // without removing any wedge path.
    if (seq_len <= 1) return e.chunk;

    TORCH_CHECK(e.declared(),
        "REFUSING PREFILL: no certified chunk envelope declared for this ",
        model_name, " handle (seq_len=", seq_len, ", position=", position, "). "
        "This is deny-by-default: the auto chunk search can pick a chunk above "
        "the model's true SPM ceiling, which busts the 8191 KB SPM, hangs the "
        "oversized HW DMA and WEDGES the board (251 does not auto-recover). "
        "Declare a validated envelope on this handle before the first forward.");

    const int64_t kv_len = position + seq_len;
    TORCH_CHECK(kv_len <= e.max_kv_len,
        "REFUSING PREFILL: ", model_name, " kv_len=", kv_len, " (position=",
        position, " + seq_len=", seq_len, ") exceeds this handle's CERTIFIED "
        "envelope max_kv_len=", e.max_kv_len, " (chunk=", e.chunk,
        ", 0 means auto). "
        "This combination has not been validated on hardware. Validate it on a "
        "resettable board and update the model profile; do not widen the "
        "envelope merely to make this pass.");

    return e.chunk;
}

void FusedModelBase::set_model_params(int64_t num_q_heads, int64_t num_kv_heads,
                                      int64_t head_dim, int64_t hidden_size,
                                      int64_t intermediate_size) {
    pimpl_->num_q_heads_        = num_q_heads;
    pimpl_->num_kv_heads_       = num_kv_heads;
    pimpl_->head_dim_           = head_dim;
    pimpl_->hidden_size_        = hidden_size;
    pimpl_->intermediate_size_  = intermediate_size;
    pimpl_->attn_tp_            = std::min(8, (int)num_kv_heads);
}

void FusedModelBase::set_num_layers(int64_t n) { pimpl_->num_layers_ = n; }

int64_t FusedModelBase::num_q_heads()       const { return pimpl_->num_q_heads_; }
int64_t FusedModelBase::num_kv_heads()      const { return pimpl_->num_kv_heads_; }
int64_t FusedModelBase::head_dim()          const { return pimpl_->head_dim_; }
int64_t FusedModelBase::hidden_size()       const { return pimpl_->hidden_size_; }
int64_t FusedModelBase::intermediate_size() const { return pimpl_->intermediate_size_; }
int     FusedModelBase::attn_tp()           const { return pimpl_->attn_tp_; }
int64_t FusedModelBase::num_layers()        const { return pimpl_->num_layers_; }

// Fresh at::empty per forward plus a lifetime anchor makes the "tracked" in
// the name true.
//
// A caller that accumulates
// per-forward outputs (Pi0.5 embed_prefix's per-image list, GR00T's per-step
// action) must get distinct DDR each time, or torch.cat replays the last slot.
//
// Subclasses may bake
// RpuGetDevAddr(out.data_ptr()) into a DEFERRED DMA destination — see
// rpu_gr00t_dit_model.cpp x_out_dst_base_ in both step_forward and
// unroll_forward — and the graph does not execute until RpuKernelGraph::end(),
// long after this function and often after the Python frame that called it.
// A device address is not a reference, so without keep_alive the block can be
// freed and re-issued (SDK batch buffers share the same dmabuf pool) between
// recording and execution, and the graph then WRITES over whoever owns it now.
// keep_alive holds the ref until end() clears tensor_refs_ AFTER execution, so
// every allocate_tracked_output caller is correct by construction rather than
// by the accident of the Python caller keeping a named local.
// No-op outside RECORDING/REPLAYING; costs one refcount and does not flush.
at::Tensor FusedModelBase::allocate_tracked_output(at::IntArrayRef shape) {
    auto opts = at::TensorOptions()
                    .dtype(at::kHalf)
                    .device(at::kPrivateUse1);
    auto out = at::empty(shape, opts);
    if (RpuKernelGraph::has_active()) {
        RpuKernelGraph::active().keep_alive(out);
    }
    return out;
}

// =============================================================================
// Model state invalidation — subclass calls from set_weights.
// Sets both weights_dirty_ and preload_callbacks_dirty_.
// =============================================================================

void FusedModelBase::invalidate_model_state() {
    // The generation binds outer-fast and manifest ownership to the current
    // weights. It is deliberately excluded from compute_params_hash_impl: a
    // weight refresh does not change buffer sizes and must not force re-layout.
    // A subclass whose declarations change uses subclass_layout_hash().
    TORCH_CHECK(pimpl_->outer_fast_model_generation_ != nullptr &&
                    pimpl_->model_state_gen_ !=
                        std::numeric_limits<uint64_t>::max() &&
                    *pimpl_->outer_fast_model_generation_ !=
                        std::numeric_limits<uint64_t>::max(),
                "FusedModelBase: model-state generation exhausted");
    pimpl_->model_state_gen_++;
    ++*pimpl_->outer_fast_model_generation_;
    TORCH_INTERNAL_ASSERT(
        pimpl_->model_state_gen_ ==
        *pimpl_->outer_fast_model_generation_);
    pimpl_->weights_dirty_           = true;  // preload_fn path
    pimpl_->preload_callbacks_dirty_ = true;  // Per-buffer callback path.
    retire_pipeline_manifest(*pimpl_);
}

// =============================================================================
// params_hash / persistent_hash / layout_signature (verbatim from v2)
// =============================================================================

static int64_t compute_params_hash_impl(
    const LayoutContext& ctx,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t head_dim,
    int64_t hidden_size, int64_t intermediate_size,
    int64_t subclass_layout_hash)
{
    // chunk_size in the hash ensures re-allocation when the chunk changes.
    // max_kv_seq_len affects SPM sizing only when use_attn_mask is true.
    int64_t h = num_q_heads * 1000003 ^ num_kv_heads * 999983 ^
                head_dim * 999979 ^ hidden_size * 999961 ^
                intermediate_size * 999931 ^
                ctx.chunk_size * 999929 ^ ctx.num_layers * 999917 ^
                (ctx.use_attn_mask ? ctx.max_kv_seq_len * 999907 : 0) ^
                (ctx.use_attn_mask ? 999883 : 0);
    // The bidirectional-no-mask (KV_FIRST) layout differs
    // from the causal layout at the SAME (chunk_size, use_attn_mask=false) —
    // CausalDecoderModel::declare_buffers splits q_kv, sizes k/v at kv_cs, and
    // flips buffer scopes / sdpa_tmp mask on this bit. Fold the derived bit in so
    // a handle that switches causal↔bidirectional re-allocates. Causal and
    // explicit-mask paths leave the term 0 → byte-identical hash for every
    // existing allocation (zero perturbation).
    const bool kv_first_layout = !ctx.is_causal && !ctx.use_attn_mask;
    h ^= (kv_first_layout ? 999149 : 0);
    // kv_insert_chunk_size feeds declare_buffers'
    // effective_kv_cs() → res/q_kv/k/v sizing (CausalDecoderModel KV_FIRST and
    // HaloImageFlowModel). It was absent from the hash, masked only by the old
    // hardcoded planner constant; once plan_kv_first_chunks probes the real
    // footprint the chosen value varies, so it MUST key the allocation. Default 0
    // (single-chunk / causal) leaves the term 0 → zero perturbation.
    h ^= (ctx.kv_insert_chunk_size > 0 ? ctx.kv_insert_chunk_size * 999133 : 0);
    // The fields above are the ONLY layout inputs the framework can see. Every
    // other sizing input a subclass reads in declare_buffers (patch count, image
    // grid, window length, action horizon, deepstack depth, ...) arrives here
    // through FusedModelBase::subclass_layout_hash(). Default 0 → this term
    // vanishes and the hash is byte-identical to before the hook existed.
    h ^= subclass_layout_hash;
    // Batch decode: declare_buffers sizes row-parallel slots at
    // batch_size * chunk_size, so the layout differs per B at one chunk_size.
    // Default 1 leaves the term 0 → zero perturbation for existing allocations.
    h ^= (ctx.batch_size > 1 ? ctx.batch_size * 999119 : 0);
    return h;
}

static int64_t compute_persistent_hash_impl(const std::vector<BufferDecl>& decls) {
    int64_t h = 0;
    for (auto& d : decls) {
        if (d.alias_of) continue;
        if (d.storage != StorageClass::Persistent &&
            d.storage != StorageClass::PersistentPerLayer) continue;
        uint64_t name_h = 1469598103934665603ull;
        for (const char* p = d.name; *p; ++p) {
            name_h ^= (unsigned char)*p;
            name_h *= 1099511628211ull;
        }
        h ^= (int64_t)(name_h ^
                       ((uint64_t)d.size * 0x9e3779b97f4a7c15ull) ^
                       ((uint64_t)d.per_layer * 0xbf58476d1ce4e5b9ull) ^
                       ((uint64_t)d.storage * 0x94d049bb133111ebull) ^
                       ((uint64_t)d.reverse_layer_alloc * 0xd6e8feb86659fd93ull));
    }
    return h;
}

static size_t align_spm_bytes(size_t bytes) {
    TORCH_CHECK(bytes <= std::numeric_limits<size_t>::max() -
                             (SpmAllocator::ALIGN - 1),
                "FusedModelBase: SPM byte count overflows alignment");
    return (bytes + SpmAllocator::ALIGN - 1) &
           ~(SpmAllocator::ALIGN - 1);
}

static uint64_t pipeline_layout_hash_impl(
    const std::vector<BufferDecl>& decls,
    int64_t params_hash,
    size_t temporary_bytes,
    const std::unordered_map<std::string, uint32_t>& offsets,
    const std::vector<std::unordered_map<std::string, uint32_t>>&
        per_layer_offsets) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    auto mix = [&](uint64_t value) {
        for (int shift = 0; shift < 64; shift += 8) {
            hash = (hash ^ static_cast<uint8_t>(value >> shift)) *
                   0x100000001b3ULL;
        }
    };
    mix(static_cast<uint64_t>(params_hash));
    mix(temporary_bytes);
    for (const BufferDecl& decl : decls) {
        if (decl.storage != StorageClass::Temp &&
            decl.storage != StorageClass::TempPerLayer) {
            continue;
        }
        TORCH_CHECK(decl.name != nullptr,
                    "FusedModelBase: unnamed pipeline buffer");
        for (const unsigned char* p =
                 reinterpret_cast<const unsigned char*>(decl.name);
             *p; ++p) {
            hash = (hash ^ *p) * 0x100000001b3ULL;
        }
        mix(static_cast<uint64_t>(decl.size));
        mix(static_cast<uint64_t>(decl.phase_start));
        mix(static_cast<uint64_t>(decl.phase_end));
        mix(static_cast<uint64_t>(decl.storage));
        mix(static_cast<uint64_t>(decl.per_layer));
        mix(static_cast<uint64_t>(decl.scope));
        if (decl.alias_of != nullptr) {
            for (const unsigned char* p =
                     reinterpret_cast<const unsigned char*>(decl.alias_of);
                 *p; ++p) {
                hash = (hash ^ *p) * 0x100000001b3ULL;
            }
        }
        if (decl.storage == StorageClass::Temp) {
            auto it = offsets.find(decl.name);
            TORCH_CHECK(it != offsets.end(),
                        "FusedModelBase: missing Temp offset while hashing '",
                        decl.name, "'");
            mix(it->second);
        } else {
            TORCH_CHECK(decl.per_layer >= 0 &&
                            static_cast<size_t>(decl.per_layer) <=
                                per_layer_offsets.size(),
                        "FusedModelBase: missing TempPerLayer maps while hashing '",
                        decl.name, "'");
            mix(static_cast<uint64_t>(decl.per_layer));
            for (int layer = 0; layer < decl.per_layer; ++layer) {
                auto it = per_layer_offsets[layer].find(decl.name);
                TORCH_CHECK(it != per_layer_offsets[layer].end(),
                            "FusedModelBase: missing layer ", layer,
                            " TempPerLayer offset while hashing '", decl.name,
                            "'");
                mix(static_cast<uint64_t>(layer));
                mix(it->second);
            }
        }
    }
    return hash == 0 ? 0x9e3779b97f4a7c15ULL : hash;
}

using OwnedPipelineDecl = FusedModelBase::Impl::OwnedPipelineDecl;
using PipelineAllocationEntry =
    FusedModelBase::Impl::PipelineAllocationEntry;
using CachedLayoutSnapshot = FusedModelBase::Impl::CachedLayoutSnapshot;

static std::vector<OwnedPipelineDecl> own_pipeline_decls(
    const std::vector<BufferDecl>& decls,
    bool require_seal_eligibility = true) {
    std::vector<OwnedPipelineDecl> owned;
    owned.reserve(decls.size());
    std::unordered_map<std::string, size_t> by_name;
    for (const BufferDecl& decl : decls) {
        TORCH_CHECK(decl.name != nullptr && decl.name[0] != '\0',
                    "FusedModelBase manifest snapshot: empty buffer name");
        const std::string name(decl.name);
        TORCH_CHECK(by_name.emplace(name, owned.size()).second,
                    "FusedModelBase manifest snapshot: duplicate buffer '",
                    name, "'");
        if (require_seal_eligibility) {
            const int storage_value = static_cast<int>(decl.storage);
            TORCH_CHECK(
                storage_value >= static_cast<int>(StorageClass::Temp) &&
                    storage_value <=
                        static_cast<int>(StorageClass::PersistentPerLayer),
                "FusedModelBase manifest snapshot: invalid storage class for '",
                name, "'");
            TORCH_CHECK(
                static_cast<uint8_t>(decl.scope) <=
                    static_cast<uint8_t>(BufferScope::OutsideLayerLoop),
                "FusedModelBase manifest snapshot: invalid buffer scope for '",
                name, "'");
            TORCH_CHECK(decl.phase_start >= 0 &&
                            decl.phase_start <= decl.phase_end,
                        "FusedModelBase manifest snapshot: invalid phase for '",
                        name, "'");
            TORCH_CHECK(decl.size >= 0,
                        "FusedModelBase manifest snapshot: negative size for '",
                        name, "'");
            TORCH_CHECK(decl.alias_of != nullptr || decl.size > 0,
                        "FusedModelBase manifest snapshot: non-alias '", name,
                        "' must have positive size");
            if (decl.storage == StorageClass::TempPerLayer ||
                decl.storage == StorageClass::PersistentPerLayer) {
                TORCH_CHECK(
                    decl.per_layer > 0,
                    "FusedModelBase manifest snapshot: per-layer buffer '",
                    name, "' needs positive per_layer");
            }
            TORCH_CHECK(!decl.preload_callback ||
                            decl.storage == StorageClass::Persistent ||
                            decl.storage ==
                                StorageClass::PersistentPerLayer,
                        "FusedModelBase manifest snapshot: temporary buffer '",
                        name, "' cannot declare a preload callback");
            TORCH_CHECK(!decl.reverse_layer_alloc ||
                            decl.storage ==
                                StorageClass::PersistentPerLayer,
                        "FusedModelBase manifest snapshot: "
                        "reverse_layer_alloc is valid only for "
                        "PersistentPerLayer; got '", name, "'");
        }
        OwnedPipelineDecl item;
        item.name = name;
        item.logical_bytes = decl.size;
        item.aligned_bytes = decl.size > 0
            ? align_spm_bytes(static_cast<size_t>(decl.size)) : 0;
        item.phase_start = decl.phase_start;
        item.phase_end = decl.phase_end;
        item.storage = decl.storage;
        item.per_layer = decl.per_layer;
        item.alias_of = decl.alias_of == nullptr
            ? std::string() : std::string(decl.alias_of);
        item.scope = decl.scope;
        item.reverse_layer_alloc = decl.reverse_layer_alloc;
        item.preload_present = static_cast<bool>(decl.preload_callback);
        owned.push_back(std::move(item));
    }

    if (!require_seal_eligibility) return owned;

    for (OwnedPipelineDecl& decl : owned) {
        if (decl.alias_of.empty()) continue;
        TORCH_CHECK(decl.storage == StorageClass::Temp,
                    "FusedModelBase manifest snapshot: aliases are supported "
                    "only for flat Temp; got alias '", decl.name, "'");
        TORCH_CHECK(decl.per_layer == 0 &&
                        !decl.reverse_layer_alloc &&
                        !decl.preload_present,
                    "FusedModelBase manifest snapshot: flat Temp alias '",
                    decl.name,
                    "' cannot carry per-layer/reverse/preload metadata");
        auto target_it = by_name.find(decl.alias_of);
        TORCH_CHECK(target_it != by_name.end(),
                    "FusedModelBase manifest snapshot: alias target '",
                    decl.alias_of, "' for '", decl.name, "' is missing");
        const OwnedPipelineDecl& target = owned[target_it->second];
        TORCH_CHECK(target.alias_of.empty(),
                    "FusedModelBase manifest snapshot: alias chains/cycles are "
                    "not supported; '", decl.name, "' targets alias '",
                    decl.alias_of, "'");
        TORCH_CHECK(target.storage == StorageClass::Temp,
                    "FusedModelBase manifest snapshot: alias '", decl.name,
                    "' must target flat Temp '", decl.alias_of, "'");
        TORCH_CHECK(decl.scope == target.scope,
                    "FusedModelBase manifest snapshot: alias '", decl.name,
                    "' must have the same scope as root '", decl.alias_of,
                    "'");
        if (decl.logical_bytes == 0) {
            decl.aligned_bytes = target.aligned_bytes;
        } else {
            TORCH_CHECK(decl.logical_bytes <= target.logical_bytes &&
                            decl.aligned_bytes <= target.aligned_bytes,
                        "FusedModelBase manifest snapshot: alias '", decl.name,
                        "' span exceeds root '", decl.alias_of, "'");
        }
    }
    return owned;
}

static uint64_t owned_pipeline_decl_hash(
    const std::vector<OwnedPipelineDecl>& decls) {
    uint64_t hash = fmb_hash_u64(kFmbFnvOffset, 1);
    hash = fmb_hash_u64(hash, decls.size());
    for (const OwnedPipelineDecl& decl : decls) {
        hash = fmb_hash_string(hash, decl.name);
        hash = fmb_hash_u64(hash,
                            static_cast<uint64_t>(decl.logical_bytes));
        hash = fmb_hash_u64(hash,
                            static_cast<uint64_t>(decl.phase_start));
        hash = fmb_hash_u64(hash,
                            static_cast<uint64_t>(decl.phase_end));
        hash = fmb_hash_u64(hash, static_cast<uint8_t>(decl.storage));
        hash = fmb_hash_u64(hash, static_cast<uint64_t>(decl.per_layer));
        hash = fmb_hash_string(hash, decl.alias_of);
        hash = fmb_hash_u64(hash, static_cast<uint8_t>(decl.scope));
        hash = fmb_hash_u64(hash, decl.reverse_layer_alloc);
        hash = fmb_hash_u64(hash, decl.preload_present);
    }
    return fmb_nonzero_hash(hash);
}

static bool same_owned_pipeline_declarations(
    const std::vector<OwnedPipelineDecl>& lhs,
    const std::vector<OwnedPipelineDecl>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t index = 0; index < lhs.size(); ++index) {
        const OwnedPipelineDecl& left = lhs[index];
        const OwnedPipelineDecl& right = rhs[index];
        if (left.name != right.name ||
            left.logical_bytes != right.logical_bytes ||
            left.phase_start != right.phase_start ||
            left.phase_end != right.phase_end ||
            left.storage != right.storage ||
            left.per_layer != right.per_layer ||
            left.alias_of != right.alias_of ||
            left.scope != right.scope ||
            left.reverse_layer_alloc != right.reverse_layer_alloc ||
            left.preload_present != right.preload_present) {
            return false;
        }
    }
    return true;
}

static void sort_pipeline_allocations(
    std::vector<PipelineAllocationEntry>& entries) {
    std::sort(entries.begin(), entries.end(),
              [](const PipelineAllocationEntry& lhs,
                 const PipelineAllocationEntry& rhs) {
                  return std::tie(lhs.storage, lhs.name, lhs.layer,
                                  lhs.offset, lhs.aligned_bytes) <
                         std::tie(rhs.storage, rhs.name, rhs.layer,
                                  rhs.offset, rhs.aligned_bytes);
              });
}

static uint64_t pipeline_allocation_entries_hash(
    const std::vector<PipelineAllocationEntry>& entries,
    uint64_t seed) {
    uint64_t hash = fmb_hash_u64(seed, entries.size());
    for (const PipelineAllocationEntry& entry : entries) {
        hash = fmb_hash_string(hash, entry.name);
        hash = fmb_hash_u64(hash, static_cast<uint8_t>(entry.storage));
        hash = fmb_hash_u64(hash, static_cast<uint64_t>(entry.layer));
        hash = fmb_hash_u64(hash, entry.logical_bytes);
        hash = fmb_hash_u64(hash, entry.aligned_bytes);
        hash = fmb_hash_u64(hash, entry.offset);
        hash = fmb_hash_string(hash, entry.alias_root);
        hash = fmb_hash_u64(hash, entry.reverse_layer_alloc);
        hash = fmb_hash_u64(hash, entry.preload_present);
    }
    return hash;
}

static CachedLayoutSnapshot build_cached_layout_snapshot(
    const std::vector<BufferDecl>& raw_decls,
    const LayoutContext& layout_context,
    int64_t params_hash,
    size_t required_extent,
    uint64_t layout_hash,
    const std::unordered_map<std::string, uint32_t>& offsets,
    const std::vector<std::unordered_map<std::string, uint32_t>>&
        per_layer_offsets,
    const std::unordered_map<std::string, uint32_t>& persistent_offsets,
    const std::vector<std::unordered_map<std::string, uint32_t>>&
        persistent_per_layer_offsets,
    bool cpu_dry,
    uint64_t allocator_generation,
    uint64_t persistent_generation,
    uint64_t persistent_layout_version,
    size_t persistent_floor,
    size_t persistent_top) {
    CachedLayoutSnapshot snapshot;
    snapshot.valid = true;
    snapshot.cpu_dry = cpu_dry;
    snapshot.params_hash = params_hash;
    snapshot.layout_context = layout_context;
    snapshot.required_extent = required_extent;
    snapshot.layout_hash = layout_hash;
    snapshot.allocator_generation = allocator_generation;
    snapshot.persistent_generation = persistent_generation;
    snapshot.persistent_layout_version = persistent_layout_version;
    snapshot.persistent_floor = persistent_floor;
    snapshot.persistent_top = persistent_top;
    snapshot.decls = own_pipeline_decls(raw_decls);
    snapshot.declaration_hash = owned_pipeline_decl_hash(snapshot.decls);

    std::unordered_map<std::string, const OwnedPipelineDecl*> by_name;
    for (const OwnedPipelineDecl& decl : snapshot.decls) {
        by_name.emplace(decl.name, &decl);
    }
    size_t max_temporary_root_end = 0;
    bool saw_temporary_zero = false;
    for (const OwnedPipelineDecl& decl : snapshot.decls) {
        if (decl.storage == StorageClass::Temp) {
            auto it = offsets.find(decl.name);
            TORCH_CHECK(it != offsets.end(),
                        "FusedModelBase manifest snapshot: missing Temp offset '",
                        decl.name, "'");
            const OwnedPipelineDecl& root = decl.alias_of.empty()
                ? decl : *by_name.at(decl.alias_of);
            if (!decl.alias_of.empty()) {
                auto root_offset = offsets.find(root.name);
                TORCH_CHECK(root_offset != offsets.end() &&
                                root_offset->second == it->second,
                            "FusedModelBase manifest snapshot: alias '",
                            decl.name, "' offset differs from root '",
                            root.name, "'");
            }
            const size_t end = static_cast<size_t>(it->second) +
                               decl.aligned_bytes;
            TORCH_CHECK(end <= required_extent,
                        "FusedModelBase manifest snapshot: Temp '", decl.name,
                        "' exceeds required extent");
            if (decl.alias_of.empty()) {
                max_temporary_root_end =
                    std::max(max_temporary_root_end, end);
            }
            saw_temporary_zero = saw_temporary_zero || it->second == 0;
            snapshot.temporary_allocations.push_back(
                {decl.name, decl.storage, -1,
                 static_cast<size_t>(decl.logical_bytes), decl.aligned_bytes,
                 it->second, root.name, decl.reverse_layer_alloc,
                 decl.preload_present});
        } else if (decl.storage == StorageClass::TempPerLayer) {
            TORCH_CHECK(decl.alias_of.empty(),
                        "FusedModelBase manifest snapshot: TempPerLayer alias '",
                        decl.name, "' is unsupported");
            TORCH_CHECK(static_cast<size_t>(decl.per_layer) <=
                            per_layer_offsets.size(),
                        "FusedModelBase manifest snapshot: missing per-layer "
                        "maps for '", decl.name, "'");
            for (int layer = 0; layer < decl.per_layer; ++layer) {
                auto it = per_layer_offsets[layer].find(decl.name);
                TORCH_CHECK(it != per_layer_offsets[layer].end(),
                            "FusedModelBase manifest snapshot: missing layer ",
                            layer, " offset for '", decl.name, "'");
                const size_t end = static_cast<size_t>(it->second) +
                                   decl.aligned_bytes;
                TORCH_CHECK(end <= required_extent,
                            "FusedModelBase manifest snapshot: TempPerLayer '",
                            decl.name, "' exceeds required extent");
                max_temporary_root_end =
                    std::max(max_temporary_root_end, end);
                saw_temporary_zero = saw_temporary_zero || it->second == 0;
                snapshot.temporary_allocations.push_back(
                    {decl.name, decl.storage, layer,
                     static_cast<size_t>(decl.logical_bytes),
                     decl.aligned_bytes, it->second, decl.name,
                     decl.reverse_layer_alloc, decl.preload_present});
            }
        } else if (decl.storage == StorageClass::Persistent) {
            TORCH_CHECK(decl.alias_of.empty(),
                        "FusedModelBase manifest snapshot: Persistent alias '",
                        decl.name, "' is unsupported");
            auto it = persistent_offsets.find(decl.name);
            TORCH_CHECK(it != persistent_offsets.end(),
                        "FusedModelBase manifest snapshot: missing Persistent '",
                        decl.name, "'");
            snapshot.persistent_allocations.push_back(
                {decl.name, decl.storage, -1,
                 static_cast<size_t>(decl.logical_bytes), decl.aligned_bytes,
                 it->second, decl.name, decl.reverse_layer_alloc,
                 decl.preload_present});
        } else {
            TORCH_CHECK(decl.storage == StorageClass::PersistentPerLayer &&
                            decl.alias_of.empty(),
                        "FusedModelBase manifest snapshot: unsupported "
                        "persistent alias/storage for '", decl.name, "'");
            TORCH_CHECK(static_cast<size_t>(decl.per_layer) <=
                            persistent_per_layer_offsets.size(),
                        "FusedModelBase manifest snapshot: missing persistent "
                        "layer maps for '", decl.name, "'");
            for (int layer = 0; layer < decl.per_layer; ++layer) {
                auto it = persistent_per_layer_offsets[layer].find(decl.name);
                TORCH_CHECK(
                    it != persistent_per_layer_offsets[layer].end(),
                    "FusedModelBase manifest snapshot: missing persistent "
                    "layer ", layer, " offset for '", decl.name, "'");
                snapshot.persistent_allocations.push_back(
                    {decl.name, decl.storage, layer,
                     static_cast<size_t>(decl.logical_bytes),
                     decl.aligned_bytes, it->second, decl.name,
                     decl.reverse_layer_alloc, decl.preload_present});
            }
        }
    }
    TORCH_CHECK(!snapshot.temporary_allocations.empty() &&
                    saw_temporary_zero &&
                    max_temporary_root_end == required_extent,
                "FusedModelBase manifest snapshot: temporary allocation table "
                "root extent ", max_temporary_root_end,
                " does not exactly match cached extent ", required_extent);
    TORCH_CHECK(persistent_floor <= persistent_top &&
                    persistent_top <= SpmAllocator::SPM_USABLE,
                "FusedModelBase manifest snapshot: invalid persistent bounds");
    for (const PipelineAllocationEntry& entry :
         snapshot.persistent_allocations) {
        const size_t begin = entry.offset;
        TORCH_CHECK(begin >= persistent_floor &&
                        begin <= persistent_top &&
                        entry.aligned_bytes <= persistent_top - begin,
                    "FusedModelBase manifest snapshot: Persistent '",
                    entry.name, "' escapes sealed persistent bounds");
    }

    sort_pipeline_allocations(snapshot.temporary_allocations);
    sort_pipeline_allocations(snapshot.persistent_allocations);
    for (size_t lhs = 0; lhs < snapshot.persistent_allocations.size(); ++lhs) {
        const PipelineAllocationEntry& left =
            snapshot.persistent_allocations[lhs];
        const uint64_t left_end =
            static_cast<uint64_t>(left.offset) + left.aligned_bytes;
        for (size_t rhs = lhs + 1;
             rhs < snapshot.persistent_allocations.size(); ++rhs) {
            const PipelineAllocationEntry& right =
                snapshot.persistent_allocations[rhs];
            const uint64_t right_end =
                static_cast<uint64_t>(right.offset) + right.aligned_bytes;
            TORCH_CHECK(left_end <= right.offset || right_end <= left.offset,
                        "FusedModelBase manifest snapshot: Persistent entries '",
                        left.name, "' and '", right.name, "' overlap");
        }
    }
    for (const OwnedPipelineDecl& decl : snapshot.decls) {
        if (decl.storage != StorageClass::PersistentPerLayer) continue;
        std::vector<const PipelineAllocationEntry*> layers(decl.per_layer,
                                                            nullptr);
        for (const PipelineAllocationEntry& entry :
             snapshot.persistent_allocations) {
            if (entry.name == decl.name && entry.layer >= 0 &&
                entry.layer < decl.per_layer) {
                layers[entry.layer] = &entry;
            }
        }
        for (int layer = 0; layer + 1 < decl.per_layer; ++layer) {
            TORCH_INTERNAL_ASSERT(layers[layer] != nullptr &&
                                  layers[layer + 1] != nullptr);
            if (decl.reverse_layer_alloc) {
                TORCH_CHECK(
                    static_cast<uint64_t>(layers[layer]->offset) +
                            layers[layer]->aligned_bytes ==
                        layers[layer + 1]->offset,
                    "FusedModelBase manifest snapshot: reverse-layer "
                    "PersistentPerLayer direction mismatch for '",
                    decl.name, "'");
            } else {
                TORCH_CHECK(
                    static_cast<uint64_t>(layers[layer + 1]->offset) +
                            layers[layer + 1]->aligned_bytes ==
                        layers[layer]->offset,
                    "FusedModelBase manifest snapshot: ascending-declaration "
                    "PersistentPerLayer direction mismatch for '",
                    decl.name, "'");
            }
        }
    }
    uint64_t allocation_hash = fmb_hash_u64(kFmbFnvOffset, 2);
    allocation_hash = fmb_hash_u64(
        allocation_hash, static_cast<uint64_t>(params_hash));
    allocation_hash = fmb_hash_u64(allocation_hash, required_extent);
    allocation_hash = fmb_hash_u64(
        allocation_hash, snapshot.declaration_hash);
    allocation_hash = pipeline_allocation_entries_hash(
        snapshot.temporary_allocations, allocation_hash);
    snapshot.allocation_hash = fmb_nonzero_hash(allocation_hash);

    uint64_t persistent_hash = fmb_hash_u64(kFmbFnvOffset, 3);
    persistent_hash = fmb_hash_u64(persistent_hash,
                                   persistent_layout_version);
    persistent_hash = pipeline_allocation_entries_hash(
        snapshot.persistent_allocations, persistent_hash);
    snapshot.persistent_hash = fmb_nonzero_hash(persistent_hash);
    return snapshot;
}

static int64_t estimate_fixed_overhead(const std::vector<BufferDecl>& decls) {
    int64_t total = 0;
    for (auto& d : decls) {
        if (d.alias_of) continue;
        switch (d.storage) {
            case StorageClass::Persistent:
                total += d.size; break;
            case StorageClass::PersistentPerLayer:
            case StorageClass::TempPerLayer:
                total += d.size * d.per_layer; break;
            default:
                break;
        }
    }
    return total;
}

// Return the exact peak that alloc_temporary_aliased() will request. A
// max-live-bytes calculation is only a lower bound: first-fit placement can
// leave holes smaller than a later buffer and therefore consume more SPM than
// the simultaneous live set. The chunk planner must use the allocator's pure
// plan or it can select a chunk that passes budgeting and then OOMs at dispatch.
int64_t detail::estimate_temporary_total(const std::vector<BufferDecl>& decls) {
    std::vector<SpmAllocator::AllocRequest> requests;
    requests.reserve(decls.size());
    for (const BufferDecl& decl : decls) {
        if (decl.storage != StorageClass::Temp || decl.alias_of) continue;
        requests.push_back({decl.size, decl.phase_start, decl.phase_end,
                            static_cast<int>(decl.scope)});
    }
    return static_cast<int64_t>(
        SpmAllocator::plan_temporary_aliased(requests).peak_bytes);
}

bool detail::prefer_balanced_chunk(int64_t seq_len, int64_t candidate,
                                   int64_t current) {
    if (current <= 0) return true;
    const int64_t candidate_chunks = (seq_len + candidate - 1) / candidate;
    const int64_t current_chunks = (seq_len + current - 1) / current;
    if (candidate_chunks != current_chunks)
        return candidate_chunks < current_chunks;
    return candidate_chunks * candidate - seq_len
         < current_chunks * current - seq_len;
}

// =============================================================================
// allocate_temp_aliased / resolve_aliases (verbatim from v2)
// =============================================================================

static void allocate_temp_aliased_impl(
    const std::vector<BufferDecl>& decls,
    std::unordered_map<std::string, uint32_t>& offsets)
{
    std::vector<SpmAllocator::AllocRequest> reqs;
    std::vector<std::string> names;
    for (auto& d : decls) {
        if (d.storage == StorageClass::Temp && !d.alias_of) {
            reqs.push_back({d.size, d.phase_start, d.phase_end, static_cast<int>(d.scope)});
            names.push_back(d.name);
        }
    }
    if (reqs.empty()) return;
    auto offs = SPM_ALLOC.alloc_temporary_aliased(reqs);
    for (size_t i = 0; i < offs.size(); i++) offsets[names[i]] = offs[i];
}

static void resolve_aliases_impl(
    const std::vector<BufferDecl>& decls,
    std::unordered_map<std::string, uint32_t>& offsets)
{
    for (auto& d : decls) {
        if (d.alias_of) {
            auto it = offsets.find(d.alias_of);
            TORCH_CHECK(it != offsets.end(),
                        "BufferDecl alias target '", d.alias_of,
                        "' not found for '", d.name, "'");
            offsets[d.name] = it->second;
        }
    }
}

// =============================================================================
// ensure_allocated — three-path state machine
//
// Path 1: params_changed → full realloc of temp + persistent (first call only)
// Path 2: gen_changed → temp-only rebuild, persistent survives
// Path 3: nothing to do, full reuse
//
// preload_callbacks_dirty_ is set when persistent_generation advances, not
// just by invalidate_model_state.
// =============================================================================

// 多实例共存时不发假的"persistent 被抹掉"警报（RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN，
// 默认 OFF —— 不改任何既有模型的行为）。理由见下面 Step 2 的注释。
static bool coexist_keep_persistent_gen() {
    static const bool v = [] {
        const char* e = std::getenv("RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN");
        if (!e) return false;
        const std::string t(e);
        return t == "1" || t == "true" || t == "True" || t == "on";
    }();
    return v;
}

static void ensure_allocated_impl(
    FusedModelBase::Impl& s,
    const std::vector<BufferDecl>& decls,
    const LayoutContext& ctx,
    std::function<int64_t(const LayoutContext&)> estimate_temp_fn,
    int64_t subclass_layout_hash,
    bool enforce_allocation_identity)
{
    int64_t new_hash = compute_params_hash_impl(
        ctx, s.num_q_heads_, s.num_kv_heads_, s.head_dim_,
        s.hidden_size_, s.intermediate_size_, subclass_layout_hash);
    std::vector<OwnedPipelineDecl> fresh_owned_declarations;
    uint64_t fresh_declaration_hash = 0;
    bool fresh_declaration_captured = false;
    auto capture_fresh_declaration = [&] {
        if (fresh_declaration_captured) return;
        fresh_owned_declarations = own_pipeline_decls(
            decls, /*require_seal_eligibility=*/false);
        fresh_declaration_hash =
            owned_pipeline_decl_hash(fresh_owned_declarations);
        fresh_declaration_captured = true;
    };
    bool params_changed = !s.valid_ || (new_hash != s.cached_params_hash_);
    bool gen_changed    = (s.alloc_gen_ != SPM_ALLOC.generation());
    if (enforce_allocation_identity) {
        // Explicit pipeline prepare is cold control-plane work.  It pays the
        // owned-string/hash comparison so a fresh declaration B can never be
        // paired with allocation A.  Ordinary run/REPLAY passes false and does
        // not add this host walk to Path 3.
        capture_fresh_declaration();
        if (!s.allocation_declaration_valid_ ||
            fresh_declaration_hash != s.allocation_declaration_hash_ ||
            !same_owned_pipeline_declarations(
                fresh_owned_declarations,
                s.allocation_declarations_)) {
            params_changed = true;
        }
    }

    // persistent_generation-aware Path 1 promotion
    bool persistent_invalidated =
        (s.cached_persistent_gen_ != SPM_ALLOC.persistent_generation());
    if (!params_changed && gen_changed && s.persistent_usage_ > 0 &&
        persistent_invalidated) {
        params_changed = true;
    }

    // Persistent shape detection: once allocated, shape cannot change.
    int64_t new_persistent_hash = compute_persistent_hash_impl(decls);
    bool persistent_needs_alloc = !s.persistent_allocated_;
    if (s.persistent_allocated_ && new_persistent_hash != s.persistent_layout_hash_) {
        TORCH_CHECK(false,
            "FusedModelBase: Persistent buffer shape change detected after "
            "initial allocation (底层 SP 区单调下降, 不能回收). "
            "Recreate the model instance to use new shapes.");
    }

    (void)estimate_temp_fn;  // not needed here (compute_chunks uses it)

    const bool layout_rebuilt = params_changed || gen_changed;
    if (layout_rebuilt) capture_fresh_declaration();
    if (layout_rebuilt &&
        (s.pipeline_manifest_snapshot_.valid ||
         s.pipeline_occurrence_schedule_present_)) {
        retire_pipeline_manifest(s);
    }

    if (params_changed) {
        // Path 1: full realloc of temp. Persistent buffers allocated only on
        // first Path 1 and preserved across all subsequent invocations.

        // Step 1: first-Path-1 Persistent allocation
        if (persistent_needs_alloc) {
            SPM_ALLOC.reset_all();
            s.persistent_offsets_.clear();
            s.persistent_per_layer_offsets_.clear();
            s.persistent_usage_ = 0;
            if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

            for (auto& d : decls) {
                if (d.alias_of) continue;
                if (d.storage == StorageClass::PersistentPerLayer && d.per_layer > 0) {
                    if ((int)s.persistent_per_layer_offsets_.size() < d.per_layer)
                        s.persistent_per_layer_offsets_.resize(d.per_layer);
                    // B1: reverse_layer_alloc walks layers DESCENDING so the downward
                    // bump allocator lays layer 0 lowest — matching DDR [num_layers, h]
                    // order, which lets one DMA fill all layers (BufferDecl comment).
                    if (d.reverse_layer_alloc) {
                        for (int L = d.per_layer - 1; L >= 0; L--) {
                            s.persistent_per_layer_offsets_[L][d.name] =
                                SPM_ALLOC.alloc_super_persistent(d.size);
                        }
                    } else {
                        for (int L = 0; L < d.per_layer; L++) {
                            s.persistent_per_layer_offsets_[L][d.name] =
                                SPM_ALLOC.alloc_super_persistent(d.size);
                        }
                    }
                    s.persistent_usage_ += d.size * d.per_layer;
                } else if (d.storage == StorageClass::Persistent) {
                    s.persistent_offsets_[d.name] =
                        SPM_ALLOC.alloc_super_persistent(d.size);
                    s.persistent_usage_ += d.size;
                }
            }
            s.persistent_allocated_ = true;
            s.persistent_layout_hash_ = new_persistent_hash;
        }

        // Step 2: clear temp (persistent survives above sp_floor_)。
        //
        // ⚠️ `reset_all()` 还会推进 `persistent_generation_`，而上面的检查把它读作
        // "我的 persistent 被抹了" —— **多个实例共存时这是一个自持循环**：任一实例的
        // Path 1 会把其余实例在各自的下一次 forward 提升到 Path 1，后者的 Path 1 又再次
        // 推进代际，永不收敛。多个 handle 共存时会导致各实例在每帧重复
        // realloc、re-preload 和 re-BUILD。
        //
        // 而普通 persistent 区（`[p_start_, sp_floor_)`）在本仓库**无人使用** ——
        // `SpmAllocator::alloc_persistent()` 零调用点，每个 `StorageClass::Persistent`
        // 声明都走上面的 `alloc_super_persistent`。该区为空时 `reset_all()` 与
        // `reset_temporary()` 释放的字节**完全相同**，区别只剩那一个代际计数。
        //
        // 单实例模型下两支行为**完全一致**（`persistent_invalidated` 是在本函数入口
        // 算的，单实例时自己上一轮的 bump 已在函数末尾被吸收 ⇒ 恒为 false）。
        // 默认仍走 reset_all；只有显式 opt-in 才取无假警报的那支。
        if (coexist_keep_persistent_gen() && SPM_ALLOC.persistent_used() == 0)
            SPM_ALLOC.reset_temporary();
        else
            SPM_ALLOC.reset_all();
        s.offsets_.clear();
        s.per_layer_offsets_.clear();
        s.temp_per_layer_usage_ = 0;
        if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

        // Step 3: copy persistent offsets into unified offsets_ / per_layer_offsets_
        for (auto& kv : s.persistent_offsets_) s.offsets_[kv.first] = kv.second;
        if (s.persistent_per_layer_offsets_.size() > s.per_layer_offsets_.size())
            s.per_layer_offsets_.resize(s.persistent_per_layer_offsets_.size());
        for (size_t L = 0; L < s.persistent_per_layer_offsets_.size(); L++) {
            for (auto& kv : s.persistent_per_layer_offsets_[L])
                s.per_layer_offsets_[L][kv.first] = kv.second;
        }

        // Step 4: TempPerLayer (alloc_temporary, freed by reset_temporary)
        for (auto& d : decls) {
            if (d.alias_of || d.storage != StorageClass::TempPerLayer) continue;
            if ((int)s.per_layer_offsets_.size() < d.per_layer)
                s.per_layer_offsets_.resize(d.per_layer);
            for (int L = 0; L < d.per_layer; L++) {
                s.per_layer_offsets_[L][d.name] = SPM_ALLOC.alloc_temporary(d.size);
            }
            s.temp_per_layer_usage_ += d.size * d.per_layer;
        }

        // Step 5: Temp lifecycle aliasing
        allocate_temp_aliased_impl(decls, s.offsets_);
        resolve_aliases_impl(decls, s.offsets_);

        if (persistent_needs_alloc) {
            s.persistent_layout_version_++;
            s.weights_dirty_ = true;  // preload_fn path
        }

        // Persistent advance triggers preload_callbacks_dirty_ too —
        // a buffer's SPM address may have moved, so the callback must re-fire.
        if (persistent_invalidated) s.preload_callbacks_dirty_ = true;

        s.valid_ = true;
        s.cached_params_hash_ = new_hash;

    } else if (gen_changed) {
        // Path 2: only rebuild temporary (persistent survives)
        if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();

        s.temp_per_layer_usage_ = 0;
        for (auto& d : decls) {
            if (d.storage == StorageClass::TempPerLayer && !d.alias_of) {
                for (int L = 0; L < d.per_layer; L++) {
                    s.per_layer_offsets_[L][d.name] = SPM_ALLOC.alloc_temporary(d.size);
                }
                s.temp_per_layer_usage_ += d.size * d.per_layer;
            }
        }

        allocate_temp_aliased_impl(decls, s.offsets_);
        resolve_aliases_impl(decls, s.offsets_);
    }
    // Path 3: full reuse, no work

    s.alloc_gen_ = SPM_ALLOC.generation();
    s.cached_persistent_gen_ = SPM_ALLOC.persistent_generation();

    if (layout_rebuilt) {
        advance_pipeline_manifest_generation(s);
        s.allocation_declaration_valid_ = true;
        s.allocation_declaration_hash_ = fresh_declaration_hash;
        s.allocation_declarations_ = std::move(fresh_owned_declarations);
    }
}

// =============================================================================
// compute_chunks — linear scan (auto-path); honors subclass kernel-validity hook
//
// The combined SPM-budget and subclass kernel-validity predicate is not
// monotone in cs, so binary search is invalid. Scan every candidate satisfying
// both predicates. Among those candidates, prefer
// the fewest chunks and then the smallest tail deficit.
//
// Cost: bounded by seq_len/16 declare_buffers() probes (max 512 at seq=8192),
// each cheap relative to the actual forward.
// =============================================================================

static std::vector<ChunkInfo> compute_chunks_impl(
    FusedModelBase::Impl& s,
    std::function<std::vector<BufferDecl>(const LayoutContext&)> decl_fn,
    std::function<bool(int64_t)> valid_fn,
    int64_t chunk_size_cap,
    int64_t seq_len, int64_t position, const LayoutContext& base_ctx,
    int64_t* resolved_chunk_size)
{
    // Fixed declarations and the allocator capacity available after the next
    // temporary reset constrain both auto and explicit chunk selection.  Keep
    // this one budget so an override cannot pass the dry plan and OOM when the
    // real layout is materialised beside another subsystem's persistent SPM.
    LayoutContext probe_ctx = base_ctx;
    probe_ctx.chunk_size = 16;
    const int64_t fixed_overhead =
        estimate_fixed_overhead(decl_fn(probe_ctx));
    int64_t available =
        (int64_t)(SpmAllocator::SPM_USABLE * 0.99) - fixed_overhead;

    // `fixed_overhead` probes only THIS model's decls, so the budget above
    // silently assumes the rest of SPM is free. In a multi-subsystem model it
    // is not: pi05 runs SigLIP before the VLM prefill and its persistent
    // norms/biases are still resident. Count the capacity the new layout gets
    // after reset, not the previous phase's still-live temporaries.
    constexpr int64_t OOB_RESERVE = 64 * 1024;
    const int64_t after_reset_free = (int64_t)SPM_ALLOC.free_space()
                                   + (int64_t)SPM_ALLOC.temporary_used();
    const int64_t actual_free = after_reset_free - OOB_RESERVE;
    if (actual_free > 0) available = std::min(available, actual_free);
    available = std::max(available, (int64_t)0);
    if (std::getenv("HALO_CHUNK_DIAG") != nullptr) {
        fprintf(stderr,
                "[CHUNK_DIAG] cap seq_len=%ld free=%ld temp=%ld "
                "after_reset=%ld budget=%ld\n",
                (long)seq_len, (long)SPM_ALLOC.free_space(),
                (long)SPM_ALLOC.temporary_used(), (long)after_reset_free,
                (long)available);
    }

    auto temporary_total_for = [&](int64_t cs) {
        LayoutContext ctx = base_ctx;
        ctx.chunk_size = cs;
        return detail::estimate_temporary_total(decl_fn(ctx));
    };

    int64_t chunk_size = 0;
    if (s.chunk_size_override_ > 0) {
        // Clamp the override to the v16-rounded seq_len: a single override is
        // sticky across prefill + decode, but cs > seq_len has no meaning and
        // would fail the subclass validator (e.g. decode at seq_len=1 with a
        // prefill override of 512 → valid_fn(512) rejects against sQry=1).
        // The auto path below already implicitly bounds cs by hi=ceil16(seq_len).
        int64_t hi = ((seq_len + 15) / 16) * 16;
        chunk_size = std::min(s.chunk_size_override_, hi);
        // Diagnostic-only escape hatch. It bypasses only the subclass/kernel
        // validity gate; the hard SPM-fit check remains non-negotiable.
        static const bool force_unsafe = [] {
            const char* value = std::getenv("RPU_CHUNK_FORCE_UNSAFE");
            return value != nullptr &&
                (std::string(value) == "1" || std::string(value) == "true");
        }();
        if (!valid_fn(chunk_size)) {
            TORCH_CHECK(force_unsafe,
                "chunk_size_override(", s.chunk_size_override_, ", clamped to ",
                chunk_size, ") fails validity check (position=", position,
                ", seq_len=", seq_len, ")");
            if (log_at(2)) {
                std::cerr
                    << "[CHUNK] WARNING: RPU_CHUNK_FORCE_UNSAFE bypassed "
                       "the validity check for chunk_size="
                    << chunk_size << " (position=" << position
                    << ", seq_len=" << seq_len
                    << ") -- output may be numerically wrong\n";
            }
        }
        const int64_t total = temporary_total_for(chunk_size);
        TORCH_CHECK(total <= available,
            "chunk_size_override(", s.chunk_size_override_, ", clamped to ",
            chunk_size, ") exceeds SPM budget (position=", position,
            ", seq_len=", seq_len, ", required=", total,
            " bytes, available=", available, " bytes)");
    } else {
        int64_t hi = ((seq_len + 15) / 16) * 16;
        if (chunk_size_cap > 0) {
            const int64_t aligned_cap = (chunk_size_cap / 16) * 16;
            TORCH_CHECK(aligned_cap >= 16,
                        "chunk_size_cap must be 0 or >= 16, got ",
                        chunk_size_cap);
            hi = std::min(hi, aligned_cap);
        }

        // Top-down linear scan over [16, hi] (step 16). First minimize the
        // number of chunks, then minimize n*cs-seq_len: the latter is both the
        // tail deficit and the padding needed to make all chunks equal.
        const bool chunk_diag = (std::getenv("HALO_CHUNK_DIAG") != nullptr);
        chunk_size = 0;
        for (int64_t cs = hi; cs >= 16; cs -= 16) {
            if (!chunk_diag && chunk_size > 0 &&
                (seq_len + cs - 1) / cs > (seq_len + chunk_size - 1) / chunk_size)
                break;
            const int64_t total = temporary_total_for(cs);
            const bool fits = (total <= available);
            const bool valid = (!valid_fn || valid_fn(cs));
            const bool preferred = fits && valid &&
                detail::prefer_balanced_chunk(seq_len, cs, chunk_size);
            if (chunk_diag) {
                fprintf(stderr,
                    "[CHUNK_DIAG] cs=%ld total=%ld avail=%ld fits=%d valid=%d%s\n",
                    (long)cs, (long)total, (long)available, (int)fits, (int)valid,
                    preferred ? "  <-- BEST" : "");
            }
            if (preferred) chunk_size = cs;
        }
        TORCH_CHECK(chunk_size > 0,
                    "compute_chunks: no chunk_size in [16, ", hi,
                    "] (step 16) satisfies BOTH SPM budget AND subclass "
                    "kernel-validity. seq_len=", seq_len,
                    ", chunk_size_cap=", chunk_size_cap,
                    ", available=", available, " bytes. "
                    "Use the model's supported per-instance or per-handle chunk "
                    "control if an explicit size is required.");
    }

    // Ensure the minimum chunk fits.
    {
        LayoutContext min_ctx = base_ctx;
        min_ctx.chunk_size = chunk_size;
        auto min_decls = decl_fn(min_ctx);
        int64_t fixed = estimate_fixed_overhead(min_decls);
        int64_t temp_peak = detail::estimate_temporary_total(min_decls);
        TORCH_CHECK(fixed + temp_peak <=
                        (int64_t)SpmAllocator::SPM_PLANNING_BUDGET,
                    "Cannot fit even chunk_size=", chunk_size, " in SPM. "
                    "Fixed=", fixed, ", Temp peak=", temp_peak,
                    ", Available=",
                    (int64_t)SpmAllocator::SPM_PLANNING_BUDGET);
    }

    *resolved_chunk_size = chunk_size;

    std::vector<ChunkInfo> chunks;
    for (int64_t off = 0; off < seq_len; off += chunk_size) {
        int64_t len = std::min(chunk_size, seq_len - off);
        chunks.push_back({(int)(off / chunk_size), off, len, position + off + len});
    }
    return chunks;
}

int64_t FusedModelBase::resolve_chunk_size_for_shape(
    int64_t seq_len, int64_t position,
    const std::optional<at::Tensor>& attention_mask, bool is_causal) {
    TORCH_CHECK(seq_len > 0 && position >= 0,
                "resolve_chunk_size_for_shape: invalid seq_len/position");
    const auto static_cfg = static_config();
    TORCH_CHECK(static_cfg.num_layers > 0,
                "resolve_chunk_size_for_shape: model weights are not initialized");
    TORCH_CHECK(static_cfg.num_layers == pimpl_->num_layers_,
                "resolve_chunk_size_for_shape: num_layers mismatch");

    const InferenceContext saved_ctx = pimpl_->ctx_;
    auto restore_ctx = c10::make_scope_exit([&] { pimpl_->ctx_ = saved_ctx; });
    const bool use_explicit_mask = !is_causal && attention_mask.has_value();
    pimpl_->ctx_.attention_mask = use_explicit_mask
        ? attention_mask : std::nullopt;
    pimpl_->ctx_.position = position;
    pimpl_->ctx_.seq_len = seq_len;
    pimpl_->ctx_.batch_size = 1;
    pimpl_->ctx_.is_causal = is_causal;
    pimpl_->ctx_.input_in_spm = false;
    pimpl_->ctx_.output_to_spm = false;

    LayoutContext layout_ctx;
    layout_ctx.max_kv_seq_len = use_explicit_mask
        ? attention_mask->size(-1) : position + seq_len;
    layout_ctx.num_layers = static_cfg.num_layers;
    layout_ctx.use_attn_mask = use_explicit_mask;
    layout_ctx.is_causal = is_causal;
    auto decl_fn = [this](const LayoutContext& c) {
        return this->declare_buffers(c);
    };
    auto valid_fn = [this, seq_len, position](int64_t cs) {
        return this->subclass_chunk_size_valid(cs, seq_len, position);
    };
    const int64_t chunk_size_cap =
        subclass_chunk_size_cap(seq_len, position);
    int64_t resolved = 0;
    auto chunks = compute_chunks_impl(
        *pimpl_, decl_fn, valid_fn, chunk_size_cap,
        seq_len, position, layout_ctx, &resolved);
    TORCH_CHECK(!chunks.empty(),
                "resolve_chunk_size_for_shape: planner returned no chunks");
    return resolved;
}

SpmPipelineCausalPrefillShape
FusedModelBase::resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract(
    int64_t execution_len) {
    TORCH_CHECK(
        !RpuKernelGraph::has_active(),
        "resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract must run "
        "outside Graph capture");
    TORCH_CHECK(
        execution_len > 1,
        "resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract requires "
        "a multi-token execution length");
    const ModelStaticConfig static_cfg = static_config();
    TORCH_CHECK(
        static_cfg.num_layers > 0 &&
            static_cfg.num_layers == pimpl_->num_layers_,
        "resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract requires "
        "initialized, matching model layers");

    const InferenceContext saved_ctx = pimpl_->ctx_;
    auto restore_ctx = c10::make_scope_exit([&] { pimpl_->ctx_ = saved_ctx; });
    pimpl_->ctx_.attention_mask = std::nullopt;
    pimpl_->ctx_.position = 0;
    pimpl_->ctx_.seq_len = execution_len;
    pimpl_->ctx_.batch_size = 1;
    pimpl_->ctx_.is_causal = true;
    pimpl_->ctx_.input_in_spm = false;
    pimpl_->ctx_.output_to_spm = false;

    LayoutContext base_layout;
    base_layout.max_kv_seq_len = execution_len;
    base_layout.num_layers = static_cfg.num_layers;
    base_layout.use_attn_mask = false;
    base_layout.is_causal = true;
    auto decl_fn = [this](const LayoutContext& context) {
        return this->declare_buffers(context);
    };
    auto valid_fn = [this, execution_len](int64_t candidate) {
        return this->subclass_chunk_size_valid(
            candidate, execution_len, /*position=*/0);
    };
    const int64_t chunk_size_cap =
        subclass_chunk_size_cap(execution_len, /*position=*/0);
    int64_t resolved_chunk_size = 0;
    std::vector<ChunkInfo> chunks = compute_chunks_impl(
        *pimpl_, decl_fn, valid_fn, chunk_size_cap,
        execution_len, /*position=*/0, base_layout,
        &resolved_chunk_size);
    TORCH_CHECK(
        !chunks.empty(),
        "resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract: planner "
        "returned no chunks");

    ChunkPlan plan{
        chunks.front().len, static_cast<int64_t>(chunks.size())};
    ModelDynamicConfig dynamic_cfg = dynamic_config(plan);
    ChunkPlan kv_insert_plan = plan;
    std::vector<ChunkInfo> kv_insert_chunks = chunks;
    if (dynamic_cfg.chunk_mode == ChunkMode::KV_FIRST &&
        static_cfg.kv_first_chunk_plan_fn != nullptr) {
        kv_insert_plan = std::invoke(
            static_cfg.kv_first_chunk_plan_fn, *this, plan);
    }
    if (dynamic_cfg.chunk_mode == ChunkMode::KV_FIRST &&
        kv_insert_plan.chunk_size != plan.chunk_size) {
        TORCH_CHECK(
            kv_insert_plan.chunk_size > 0,
            "resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract: "
            "invalid KV_FIRST chunk size");
        kv_insert_chunks.clear();
        for (int64_t offset = 0; offset < execution_len;
             offset += kv_insert_plan.chunk_size) {
            const int64_t length = std::min(
                kv_insert_plan.chunk_size, execution_len - offset);
            kv_insert_chunks.push_back({
                static_cast<int>(offset / kv_insert_plan.chunk_size),
                offset, length, offset + length});
        }
    }

    LayoutContext allocation_layout = base_layout;
    allocation_layout.chunk_size = chunks.front().len;
    if (dynamic_cfg.chunk_mode == ChunkMode::KV_FIRST &&
        !kv_insert_chunks.empty() &&
        kv_insert_chunks.front().len != chunks.front().len) {
        allocation_layout.kv_insert_chunk_size =
            kv_insert_chunks.front().len;
    }
    InterLayerIO io = dynamic_cfg.inter_layer_io;
    if (io == InterLayerIO::AUTO) {
        io = chunks.size() == 1
            ? InterLayerIO::SPM_RESIDENT
            : InterLayerIO::DDR_PINGPONG;
    }

    SpmPipelineCausalPrefillShape result;
    result.resolved_chunk_size = resolved_chunk_size;
    result.chunks = std::move(chunks);
    result.kv_insert_chunks = std::move(kv_insert_chunks);
    result.allocation_layout = allocation_layout;
    result.chunk_mode = dynamic_cfg.chunk_mode;
    result.inter_layer_io = io;
    result.chunk_outer_within_group =
        dynamic_cfg.chunk_outer_within_group;
    return result;
}

SpmPipelineComponentLayout FusedModelBase::prepare_spm_pipeline_component(
    const LayoutContext& layout_ctx) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "prepare_spm_pipeline_component must run outside Graph capture");
    TORCH_CHECK(layout_ctx.chunk_size > 0,
                "prepare_spm_pipeline_component requires a positive chunk_size");
    TORCH_CHECK(
        !pimpl_->pipeline_manifest_snapshot_.valid ||
            !pimpl_->pipeline_manifest_snapshot_.cpu_dry,
        "prepare_spm_pipeline_component cannot replace a CPU-dry manifest; "
        "cancel the dry prepare first");

    auto decls = declare_buffers(layout_ctx);
    auto estimate_fn = [this](const LayoutContext& ctx) {
        return detail::estimate_temporary_total(declare_buffers(ctx));
    };
    ensure_allocated_impl(*pimpl_, decls, layout_ctx, estimate_fn,
                          subclass_layout_hash(),
                          /*enforce_allocation_identity=*/true);
    pimpl_->last_decls_ = decls;
    const std::vector<OwnedPipelineDecl> prepared_declarations =
        own_pipeline_decls(decls, /*require_seal_eligibility=*/true);
    const uint64_t prepared_declaration_hash =
        owned_pipeline_decl_hash(prepared_declarations);
    TORCH_CHECK(pimpl_->allocation_declaration_valid_ &&
                    pimpl_->allocation_declaration_hash_ ==
                        prepared_declaration_hash &&
                    same_owned_pipeline_declarations(
                        prepared_declarations,
                        pimpl_->allocation_declarations_),
                "prepare_spm_pipeline_component: fresh declarations do not "
                "match the allocation-owned identity");

    size_t per_layer_bytes = 0;
    for (const BufferDecl& decl : decls) {
        if (decl.storage != StorageClass::TempPerLayer || decl.alias_of) continue;
        TORCH_CHECK(decl.per_layer >= 0,
                    "prepare_spm_pipeline_component: negative per_layer for '",
                    decl.name, "'");
        const size_t one = align_spm_bytes(static_cast<size_t>(decl.size));
        TORCH_CHECK(static_cast<size_t>(decl.per_layer) <=
                        std::numeric_limits<size_t>::max() / one,
                    "prepare_spm_pipeline_component: TempPerLayer size overflow");
        per_layer_bytes += one * static_cast<size_t>(decl.per_layer);
    }
    const size_t temporary_bytes = per_layer_bytes +
        static_cast<size_t>(detail::estimate_temporary_total(decls));
    TORCH_CHECK(temporary_bytes > 0,
                "prepare_spm_pipeline_component: component has no temporary arena");

    bool saw_zero = false;
    auto check_range = [&](uint32_t offset, const BufferDecl& decl) {
        const size_t bytes = align_spm_bytes(static_cast<size_t>(decl.size));
        const uint64_t end = static_cast<uint64_t>(offset) + bytes;
        TORCH_CHECK(end <= temporary_bytes,
                    "prepare_spm_pipeline_component: cached buffer '",
                    decl.name, "' range [", offset, ", ", end,
                    ") is not zero-based within temporary peak ",
                    temporary_bytes);
        saw_zero = saw_zero || offset == 0;
    };
    for (const BufferDecl& decl : decls) {
        if (decl.storage == StorageClass::Temp) {
            auto it = pimpl_->offsets_.find(decl.name);
            TORCH_CHECK(it != pimpl_->offsets_.end(),
                        "prepare_spm_pipeline_component: missing Temp buffer '",
                        decl.name, "'");
            check_range(it->second, decl);
        } else if (decl.storage == StorageClass::TempPerLayer) {
            TORCH_CHECK(static_cast<int>(pimpl_->per_layer_offsets_.size()) >=
                            decl.per_layer,
                        "prepare_spm_pipeline_component: missing TempPerLayer '",
                        decl.name, "'");
            for (int layer = 0; layer < decl.per_layer; ++layer) {
                auto it = pimpl_->per_layer_offsets_[layer].find(decl.name);
                TORCH_CHECK(it != pimpl_->per_layer_offsets_[layer].end(),
                            "prepare_spm_pipeline_component: missing layer ",
                            layer, " buffer '", decl.name, "'");
                check_range(it->second, decl);
            }
        }
    }
    TORCH_CHECK(saw_zero,
                "prepare_spm_pipeline_component: cached layout is not zero-based");

    const uint64_t layout_hash = pipeline_layout_hash_impl(
        decls, pimpl_->cached_params_hash_, temporary_bytes,
        pimpl_->offsets_, pimpl_->per_layer_offsets_);
    const size_t persistent_floor = SpmAllocator::SPM_USABLE -
        SPM_ALLOC.super_persistent_used();
    CachedLayoutSnapshot snapshot = build_cached_layout_snapshot(
        decls, layout_ctx, pimpl_->cached_params_hash_, temporary_bytes,
        layout_hash,
        pimpl_->offsets_, pimpl_->per_layer_offsets_,
        pimpl_->persistent_offsets_,
        pimpl_->persistent_per_layer_offsets_,
        /*cpu_dry=*/false, SPM_ALLOC.generation(),
        SPM_ALLOC.persistent_generation(),
        pimpl_->persistent_layout_version_, persistent_floor,
        SpmAllocator::SPM_USABLE);
    retire_pipeline_manifest(*pimpl_);
    pimpl_->pipeline_manifest_snapshot_ = std::move(snapshot);
    pimpl_->allocation_declaration_valid_ = true;
    pimpl_->allocation_declaration_hash_ = prepared_declaration_hash;
    pimpl_->allocation_declarations_ = prepared_declarations;
    pimpl_->pipeline_layout_hash_ = layout_hash;
    pimpl_->pipeline_temporary_bytes_ = temporary_bytes;
    pimpl_->pipeline_lease_epoch_ = 0;
    pimpl_->pipeline_plan_hash_ = 0;
    pimpl_->pipeline_scratch_base_ = 0;
    pimpl_->pipeline_temporary_names_.clear();
    pimpl_->pipeline_temporary_per_layer_names_.clear();
    for (const BufferDecl& decl : decls) {
        if (decl.storage == StorageClass::Temp) {
            pimpl_->pipeline_temporary_names_.insert(decl.name);
        } else if (decl.storage == StorageClass::TempPerLayer) {
            pimpl_->pipeline_temporary_per_layer_names_.insert(decl.name);
        }
    }
    return {temporary_bytes, layout_hash};
}

SpmPipelineComponentLayout
FusedModelBase::prepare_spm_pipeline_component_for_cpu_contract(
    const LayoutContext& layout_ctx) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "prepare_spm_pipeline_component_for_cpu_contract must run "
                "outside Graph capture");
    TORCH_CHECK(layout_ctx.chunk_size > 0,
                "prepare_spm_pipeline_component_for_cpu_contract requires a "
                "positive chunk_size");
    TORCH_CHECK(
        pimpl_->pipeline_lease_epoch_ == 0 &&
            pimpl_->pipeline_plan_hash_ == 0 &&
            pimpl_->pipeline_scratch_base_ == 0,
        "prepare_spm_pipeline_component_for_cpu_contract rejects an active "
        "physical component");
    TORCH_CHECK(
        !pimpl_->pipeline_manifest_snapshot_.valid ||
            pimpl_->pipeline_manifest_snapshot_.cpu_dry,
        "prepare_spm_pipeline_component_for_cpu_contract cannot replace a "
        "live allocation manifest");

    const std::vector<BufferDecl> decls = declare_buffers(layout_ctx);
    // Own and validate all names before constructing any table.  The returned
    // copy is intentionally discarded here; build_cached_layout_snapshot owns
    // the final copy only after the pure plan is complete.
    (void)own_pipeline_decls(decls);

    std::unordered_map<std::string, uint32_t> offsets;
    std::vector<std::unordered_map<std::string, uint32_t>> per_layer_offsets;
    std::unordered_map<std::string, uint32_t> persistent_offsets;
    std::vector<std::unordered_map<std::string, uint32_t>>
        persistent_per_layer_offsets;

    size_t persistent_floor = SpmAllocator::SPM_USABLE;
    bool has_persistent = false;
    auto allocate_persistent = [&](const BufferDecl& decl) {
        const size_t bytes = align_spm_bytes(static_cast<size_t>(decl.size));
        TORCH_CHECK(persistent_floor >= bytes,
                    "FusedModelBase CPU manifest planner: Persistent '",
                    decl.name, "' exceeds SPM");
        persistent_floor -= bytes;
        has_persistent = true;
        return static_cast<uint32_t>(persistent_floor);
    };
    for (const BufferDecl& decl : decls) {
        if (decl.alias_of != nullptr) continue;
        if (decl.storage == StorageClass::Persistent) {
            persistent_offsets[decl.name] = allocate_persistent(decl);
        } else if (decl.storage == StorageClass::PersistentPerLayer) {
            if (static_cast<int>(persistent_per_layer_offsets.size()) <
                decl.per_layer) {
                persistent_per_layer_offsets.resize(decl.per_layer);
            }
            if (decl.reverse_layer_alloc) {
                for (int layer = decl.per_layer - 1; layer >= 0; --layer) {
                    persistent_per_layer_offsets[layer][decl.name] =
                        allocate_persistent(decl);
                }
            } else {
                for (int layer = 0; layer < decl.per_layer; ++layer) {
                    persistent_per_layer_offsets[layer][decl.name] =
                        allocate_persistent(decl);
                }
            }
        }
    }

    size_t temporary_cursor = 0;
    for (const BufferDecl& decl : decls) {
        if (decl.storage != StorageClass::TempPerLayer) continue;
        if (static_cast<int>(per_layer_offsets.size()) < decl.per_layer) {
            per_layer_offsets.resize(decl.per_layer);
        }
        const size_t bytes = align_spm_bytes(static_cast<size_t>(decl.size));
        for (int layer = 0; layer < decl.per_layer; ++layer) {
            TORCH_CHECK(temporary_cursor <=
                            std::numeric_limits<uint32_t>::max(),
                        "FusedModelBase CPU manifest planner: temporary "
                        "offset exceeds uint32");
            per_layer_offsets[layer][decl.name] =
                static_cast<uint32_t>(temporary_cursor);
            TORCH_CHECK(temporary_cursor <=
                            std::numeric_limits<size_t>::max() - bytes,
                        "FusedModelBase CPU manifest planner: temporary size "
                        "overflow");
            temporary_cursor += bytes;
        }
    }

    std::vector<SpmAllocator::AllocRequest> requests;
    std::vector<std::string> request_names;
    for (const BufferDecl& decl : decls) {
        if (decl.storage == StorageClass::Temp && decl.alias_of == nullptr) {
            requests.push_back({decl.size, decl.phase_start, decl.phase_end,
                                static_cast<int>(decl.scope)});
            request_names.emplace_back(decl.name);
        }
    }
    const SpmAllocator::AliasedPlan temp_plan =
        SpmAllocator::plan_temporary_aliased(requests);
    TORCH_CHECK(temp_plan.offsets.size() == request_names.size(),
                "FusedModelBase CPU manifest planner: first-fit result size "
                "mismatch");
    for (size_t index = 0; index < request_names.size(); ++index) {
        const uint64_t offset = temporary_cursor + temp_plan.offsets[index];
        TORCH_CHECK(offset <= std::numeric_limits<uint32_t>::max(),
                    "FusedModelBase CPU manifest planner: aliased Temp offset "
                    "exceeds uint32");
        offsets[request_names[index]] = static_cast<uint32_t>(offset);
    }
    for (const BufferDecl& decl : decls) {
        if (decl.alias_of == nullptr) continue;
        const auto root = offsets.find(decl.alias_of);
        TORCH_CHECK(root != offsets.end(),
                    "FusedModelBase CPU manifest planner: alias root '",
                    decl.alias_of, "' is missing");
        offsets[decl.name] = root->second;
    }

    TORCH_CHECK(temporary_cursor <=
                    std::numeric_limits<size_t>::max() - temp_plan.peak_bytes,
                "FusedModelBase CPU manifest planner: required extent "
                "overflow");
    const size_t required_extent = temporary_cursor + temp_plan.peak_bytes;
    TORCH_CHECK(required_extent > 0,
                "FusedModelBase CPU manifest planner: component has no "
                "temporary arena");
    TORCH_CHECK(required_extent <= persistent_floor,
                "FusedModelBase CPU manifest planner: temporary extent ",
                required_extent, " overlaps persistent floor ",
                persistent_floor);

    const int64_t params_hash = compute_params_hash_impl(
        layout_ctx, pimpl_->num_q_heads_, pimpl_->num_kv_heads_,
        pimpl_->head_dim_, pimpl_->hidden_size_,
        pimpl_->intermediate_size_, subclass_layout_hash());
    const uint64_t layout_hash = pipeline_layout_hash_impl(
        decls, params_hash, required_extent, offsets, per_layer_offsets);
    CachedLayoutSnapshot snapshot = build_cached_layout_snapshot(
        decls, layout_ctx, params_hash, required_extent, layout_hash, offsets,
        per_layer_offsets, persistent_offsets,
        persistent_per_layer_offsets, /*cpu_dry=*/true,
        /*allocator_generation=*/0, /*persistent_generation=*/0,
        has_persistent ? 1 : 0, persistent_floor,
        SpmAllocator::SPM_USABLE);

    const uint64_t declaration_hash = snapshot.declaration_hash;
    std::vector<OwnedPipelineDecl> allocation_declarations = snapshot.decls;
    std::unordered_set<std::string> temporary_names;
    std::unordered_set<std::string> temporary_per_layer_names;
    temporary_names.reserve(decls.size());
    temporary_per_layer_names.reserve(decls.size());
    for (const BufferDecl& decl : decls) {
        if (decl.storage == StorageClass::Temp) {
            temporary_names.insert(decl.name);
        } else if (decl.storage == StorageClass::TempPerLayer) {
            temporary_per_layer_names.insert(decl.name);
        }
    }

    // Everything that can allocate is complete.  The retirement plus moves
    // below form one no-allocation commit, so a failed dry plan cannot leave a
    // half-installed manifest behind.
    retire_pipeline_manifest(*pimpl_);
    pimpl_->pipeline_manifest_snapshot_ = std::move(snapshot);
    pimpl_->allocation_declaration_valid_ = true;
    pimpl_->allocation_declaration_hash_ = declaration_hash;
    pimpl_->allocation_declarations_ = std::move(allocation_declarations);
    pimpl_->pipeline_layout_hash_ = layout_hash;
    pimpl_->pipeline_temporary_bytes_ = required_extent;
    pimpl_->pipeline_temporary_names_.swap(temporary_names);
    pimpl_->pipeline_temporary_per_layer_names_.swap(
        temporary_per_layer_names);
    return {required_extent, layout_hash};
}

void FusedModelBase::cancel_spm_pipeline_component_for_cpu_contract() {
    TORCH_CHECK(
        !RpuKernelGraph::has_active(),
        "cancel_spm_pipeline_component_for_cpu_contract must run outside "
        "Graph capture");
    TORCH_CHECK(
        pimpl_->pipeline_manifest_snapshot_.valid &&
            pimpl_->pipeline_manifest_snapshot_.cpu_dry,
        "cancel_spm_pipeline_component_for_cpu_contract requires one "
        "prepared CPU-dry manifest");
    TORCH_CHECK(
        pimpl_->pipeline_lease_epoch_ == 0 &&
            pimpl_->pipeline_plan_hash_ == 0 &&
            pimpl_->pipeline_scratch_base_ == 0,
        "cancel_spm_pipeline_component_for_cpu_contract rejects an active "
        "physical component");

    retire_pipeline_manifest(*pimpl_);
    // CPU planning owns a declaration snapshot but never installs its local
    // offsets into the live allocator-backed layout.  Do not let a later
    // physical prepare mistake that dry declaration table for live allocation
    // authority; ordinary Path 3 continues to use its unchanged cached offsets.
    pimpl_->allocation_declaration_valid_ = false;
    pimpl_->allocation_declaration_hash_ = 0;
    pimpl_->allocation_declarations_.clear();
}

void FusedModelBase::set_spm_pipeline_occurrence_schedule(
    const SpmFmbOccurrenceSchedule& schedule) {
    TORCH_CHECK(pimpl_->pipeline_manifest_snapshot_.valid,
                "set_spm_pipeline_occurrence_schedule requires a prepared "
                "owned allocation snapshot");
    TORCH_CHECK(schedule.version != 0,
                "set_spm_pipeline_occurrence_schedule requires a nonzero "
                "schedule version");
    TORCH_CHECK(!schedule.occurrences.empty(),
                "set_spm_pipeline_occurrence_schedule requires at least one "
                "occurrence");
    std::set<uint32_t> occurrence_keys;
    std::set<std::tuple<uint8_t, uint32_t, uint32_t, uint32_t, uint32_t,
                        uint8_t, int>> occurrence_profiles;
    std::vector<std::pair<uint32_t, uint32_t>> global_windows;
    for (const SpmFmbOccurrence& occurrence : schedule.occurrences) {
        TORCH_CHECK(occurrence.key != 0,
                    "set_spm_pipeline_occurrence_schedule: occurrence key "
                    "zero is reserved");
        TORCH_CHECK(occurrence_keys.insert(occurrence.key).second,
                    "set_spm_pipeline_occurrence_schedule: duplicate "
                    "occurrence key ", occurrence.key);
        TORCH_CHECK(occurrence.local_phase_begin >= 0 &&
                        occurrence.local_phase_begin <=
                            occurrence.local_phase_end,
                    "set_spm_pipeline_occurrence_schedule: invalid local "
                    "phase window for occurrence ", occurrence.key);
        TORCH_CHECK(static_cast<uint8_t>(occurrence.kind) >=
                        static_cast<uint8_t>(
                            SpmFmbOccurrenceKind::LayerBody) &&
                        static_cast<uint8_t>(occurrence.kind) <=
                        static_cast<uint8_t>(
                            SpmFmbOccurrenceKind::ExternalFill),
                    "set_spm_pipeline_occurrence_schedule: invalid semantic "
                    "kind for occurrence ", occurrence.key);
        TORCH_CHECK(static_cast<uint8_t>(occurrence.scope) <=
                        static_cast<uint8_t>(BufferScope::OutsideLayerLoop),
                    "set_spm_pipeline_occurrence_schedule: invalid buffer "
                    "scope for occurrence ", occurrence.key);
        if (occurrence.kind == SpmFmbOccurrenceKind::LayerBody ||
            occurrence.kind == SpmFmbOccurrenceKind::KvInsertBody) {
            TORCH_CHECK(occurrence.layer >= 0,
                        "set_spm_pipeline_occurrence_schedule: layer-body "
                        "occurrence ", occurrence.key,
                        " requires a concrete layer");
        }
        TORCH_CHECK(
            occurrence_profiles.emplace(
                static_cast<uint8_t>(occurrence.kind), occurrence.body_id,
                occurrence.group_id, occurrence.chunk_ordinal,
                occurrence.repetition_ordinal,
                static_cast<uint8_t>(occurrence.scope),
                occurrence.layer).second,
            "set_spm_pipeline_occurrence_schedule: duplicate semantic "
            "execution profile for occurrence ", occurrence.key);
        const uint64_t width = static_cast<uint64_t>(
            occurrence.local_phase_end - occurrence.local_phase_begin);
        TORCH_CHECK(width <= std::numeric_limits<uint32_t>::max() -
                                 occurrence.global_origin.value(),
                    "set_spm_pipeline_occurrence_schedule: global event "
                    "window overflows for occurrence ", occurrence.key);
        const uint32_t global_end = occurrence.global_origin.value() +
            static_cast<uint32_t>(width);
        for (const auto& window : global_windows) {
            TORCH_CHECK(global_end < window.first ||
                            window.second <
                                occurrence.global_origin.value(),
                        "set_spm_pipeline_occurrence_schedule: occurrence ",
                        occurrence.key,
                        " overlaps another closed global event window");
        }
        global_windows.emplace_back(occurrence.global_origin.value(),
                                    global_end);
    }
    for (const SpmFmbNamedUseExtension& extension :
         schedule.use_extensions) {
        TORCH_CHECK(!extension.name.empty(),
                    "set_spm_pipeline_occurrence_schedule: extension name "
                    "must not be empty");
        TORCH_CHECK(extension.local_phase_begin >= 0 &&
                        extension.local_phase_begin <=
                            extension.local_phase_end,
                    "set_spm_pipeline_occurrence_schedule: invalid extension "
                    "phase window for '", extension.name, "'");
    }

    SpmFmbOccurrenceSchedule owned = schedule;
    pimpl_->pipeline_occurrence_schedule_ = std::move(owned);
    pimpl_->pipeline_occurrence_schedule_present_ = true;
}

SpmFmbSealedPhaseManifest FusedModelBase::seal_spm_pipeline_manifest(
    SpmScratchId arena,
    uint32_t arena_base) const {
    const CachedLayoutSnapshot& snapshot =
        pimpl_->pipeline_manifest_snapshot_;
    TORCH_CHECK(snapshot.valid,
                "seal_spm_pipeline_manifest requires a prepared owned "
                "allocation snapshot");
    TORCH_CHECK(pimpl_->pipeline_occurrence_schedule_present_,
                "seal_spm_pipeline_manifest: model occurrence schedule is "
                "missing; the default is intentionally fail-closed");
    TORCH_CHECK(arena.value() != 0,
                "seal_spm_pipeline_manifest: arena ID zero is reserved");
    TORCH_CHECK(arena_base % SpmAllocator::ALIGN == 0,
                "seal_spm_pipeline_manifest: arena base must be 256-byte "
                "aligned");
    TORCH_CHECK(snapshot.required_extent > 0 &&
                    snapshot.required_extent % SpmAllocator::ALIGN == 0,
                "seal_spm_pipeline_manifest: snapshot extent is invalid");
    TORCH_CHECK(snapshot.required_extent <=
                    std::numeric_limits<uint32_t>::max() - arena_base,
                "seal_spm_pipeline_manifest: placed arena exceeds uint32");
    TORCH_CHECK(pimpl_->pipeline_manifest_generation_ != nullptr,
                "seal_spm_pipeline_manifest: owner generation is missing");

    const SpmFmbOccurrenceSchedule& schedule =
        pimpl_->pipeline_occurrence_schedule_;
    std::vector<SpmFmbOccurrence> occurrences = schedule.occurrences;
    std::sort(occurrences.begin(), occurrences.end(),
              [](const SpmFmbOccurrence& lhs,
                 const SpmFmbOccurrence& rhs) {
                  return lhs.key < rhs.key;
              });
    std::map<uint32_t, const SpmFmbOccurrence*> occurrence_by_key;
    for (const SpmFmbOccurrence& occurrence : occurrences) {
        TORCH_CHECK(occurrence_by_key.emplace(occurrence.key, &occurrence)
                        .second,
                    "seal_spm_pipeline_manifest: duplicate occurrence key ",
                    occurrence.key);
    }

    auto map_local_event = [](const SpmFmbOccurrence& occurrence,
                              int local_phase) {
        TORCH_CHECK(local_phase >= occurrence.local_phase_begin &&
                        local_phase <= occurrence.local_phase_end,
                    "seal_spm_pipeline_manifest: local phase ", local_phase,
                    " escapes occurrence ", occurrence.key, " window [",
                    occurrence.local_phase_begin, ", ",
                    occurrence.local_phase_end, "]");
        const uint64_t delta = static_cast<uint64_t>(
            local_phase - occurrence.local_phase_begin);
        TORCH_CHECK(delta <= std::numeric_limits<uint32_t>::max() -
                                 occurrence.global_origin.value(),
                    "seal_spm_pipeline_manifest: global event overflow in "
                    "occurrence ", occurrence.key);
        return SpmPipelineEvent(
            occurrence.global_origin.value() + static_cast<uint32_t>(delta));
    };

    std::map<std::pair<std::string, int>, const OwnedPipelineDecl*>
        declaration_by_instance;
    std::map<std::pair<std::string, int>, const PipelineAllocationEntry*>
        allocation_by_instance;
    std::map<std::string, const OwnedPipelineDecl*> declaration_by_name;
    for (const OwnedPipelineDecl& decl : snapshot.decls) {
        declaration_by_name.emplace(decl.name, &decl);
    }
    for (const PipelineAllocationEntry& allocation :
         snapshot.temporary_allocations) {
        const auto key = std::make_pair(allocation.name, allocation.layer);
        TORCH_CHECK(allocation_by_instance.emplace(key, &allocation).second,
                    "seal_spm_pipeline_manifest: duplicate allocation entry '",
                    allocation.name, "' layer ", allocation.layer);
        const auto decl = declaration_by_name.find(allocation.name);
        TORCH_INTERNAL_ASSERT(decl != declaration_by_name.end());
        declaration_by_instance.emplace(key, decl->second);
    }

    struct RawUse {
        std::string declaration_name;
        std::string logical_identity;
        int layer = -1;
        uint32_t offset = 0;
        size_t root_bytes = 0;
        SpmPipelineEvent first;
        SpmPipelineEvent last;
        uint32_t occurrence_key = 0;
    };
    std::vector<RawUse> raw_uses;
    std::map<std::pair<std::string, int>, size_t> ordinary_use_count;
    std::map<uint32_t, size_t> occurrence_hit_count;

    auto append_use = [&](const PipelineAllocationEntry& allocation,
                          const OwnedPipelineDecl& decl,
                          const SpmFmbOccurrence& occurrence,
                          int local_begin,
                          int local_end,
                          bool ordinary) {
        const auto root_key =
            std::make_pair(allocation.alias_root, allocation.layer);
        auto root = allocation_by_instance.find(root_key);
        TORCH_CHECK(root != allocation_by_instance.end(),
                    "seal_spm_pipeline_manifest: allocation '",
                    allocation.name, "' has missing alias root '",
                    allocation.alias_root, "'");
        const bool whole_root_synonym =
            !decl.alias_of.empty() && decl.logical_bytes == 0;
        const std::string logical_identity =
            decl.alias_of.empty() || whole_root_synonym
            ? allocation.alias_root : allocation.name;
        const size_t protected_bytes =
            decl.alias_of.empty() || whole_root_synonym
            ? root->second->aligned_bytes : allocation.aligned_bytes;
        raw_uses.push_back(
            {allocation.name, logical_identity, allocation.layer,
             root->second->offset, protected_bytes,
             map_local_event(occurrence, local_begin),
             map_local_event(occurrence, local_end), occurrence.key});
        if (ordinary) ++ordinary_use_count[
            std::make_pair(allocation.name, allocation.layer)];
        ++occurrence_hit_count[occurrence.key];
    };

    for (const auto& item : allocation_by_instance) {
        const PipelineAllocationEntry& allocation = *item.second;
        const OwnedPipelineDecl& decl =
            *declaration_by_instance.at(item.first);
        for (const SpmFmbOccurrence& occurrence : occurrences) {
            if (occurrence.scope != decl.scope ||
                decl.phase_start < occurrence.local_phase_begin ||
                decl.phase_end > occurrence.local_phase_end) {
                continue;
            }
            if (decl.storage == StorageClass::TempPerLayer &&
                occurrence.layer != allocation.layer) {
                continue;
            }
            append_use(allocation, decl, occurrence, decl.phase_start,
                       decl.phase_end, /*ordinary=*/true);
        }
        TORCH_CHECK(ordinary_use_count[item.first] > 0,
                    "seal_spm_pipeline_manifest: no ordinary occurrence maps "
                    "temporary allocation '", allocation.name, "' layer ",
                    allocation.layer);
    }

    std::vector<SpmFmbNamedUseExtension> extensions =
        schedule.use_extensions;
    std::sort(extensions.begin(), extensions.end(),
              [](const SpmFmbNamedUseExtension& lhs,
                 const SpmFmbNamedUseExtension& rhs) {
                  return std::tie(lhs.name, lhs.layer, lhs.occurrence_key,
                                  lhs.local_phase_begin,
                                  lhs.local_phase_end) <
                         std::tie(rhs.name, rhs.layer, rhs.occurrence_key,
                                  rhs.local_phase_begin,
                                  rhs.local_phase_end);
              });
    std::set<std::tuple<std::string, int, uint32_t>> extension_keys;
    for (const SpmFmbNamedUseExtension& extension : extensions) {
        TORCH_CHECK(extension_keys.emplace(
                        extension.name, extension.layer,
                        extension.occurrence_key).second,
                    "seal_spm_pipeline_manifest: duplicate extension for '",
                    extension.name, "' layer ", extension.layer,
                    " occurrence ", extension.occurrence_key);
        auto occurrence = occurrence_by_key.find(extension.occurrence_key);
        TORCH_CHECK(occurrence != occurrence_by_key.end(),
                    "seal_spm_pipeline_manifest: extension for '",
                    extension.name, "' references unknown occurrence ",
                    extension.occurrence_key);
        const auto allocation = allocation_by_instance.find(
            std::make_pair(extension.name, extension.layer));
        TORCH_CHECK(allocation != allocation_by_instance.end(),
                    "seal_spm_pipeline_manifest: extension references unknown "
                    "temporary allocation '", extension.name, "' layer ",
                    extension.layer);
        const OwnedPipelineDecl& decl = *declaration_by_instance.at(
            allocation->first);
        append_use(*allocation->second, decl, *occurrence->second,
                   extension.local_phase_begin,
                   extension.local_phase_end, /*ordinary=*/false);
    }
    for (const SpmFmbOccurrence& occurrence : occurrences) {
        TORCH_CHECK(occurrence_hit_count[occurrence.key] > 0,
                    "seal_spm_pipeline_manifest: occurrence ",
                    occurrence.key, " maps no ordinary or appended use");
    }

    auto event_overlap = [](const RawUse& lhs, const RawUse& rhs) {
        return !(lhs.last.value() < rhs.first.value() ||
                 rhs.last.value() < lhs.first.value());
    };
    auto byte_overlap = [](const RawUse& lhs, const RawUse& rhs) {
        const uint64_t lhs_end =
            static_cast<uint64_t>(lhs.offset) + lhs.root_bytes;
        const uint64_t rhs_end =
            static_cast<uint64_t>(rhs.offset) + rhs.root_bytes;
        return lhs.offset < rhs_end && rhs.offset < lhs_end;
    };
    for (size_t lhs = 0; lhs < raw_uses.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < raw_uses.size(); ++rhs) {
            TORCH_CHECK(
                !byte_overlap(raw_uses[lhs], raw_uses[rhs]) ||
                    !event_overlap(raw_uses[lhs], raw_uses[rhs]) ||
                    (raw_uses[lhs].logical_identity ==
                         raw_uses[rhs].logical_identity &&
                     raw_uses[lhs].layer == raw_uses[rhs].layer),
                "seal_spm_pipeline_manifest: distinct logical allocations '",
                raw_uses[lhs].declaration_name, "' and '",
                raw_uses[rhs].declaration_name,
                "' overlap in byte x global-event space");
        }
    }

    std::sort(raw_uses.begin(), raw_uses.end(),
              [](const RawUse& lhs, const RawUse& rhs) {
                  return std::make_tuple(
                             lhs.logical_identity, lhs.layer, lhs.offset,
                             lhs.root_bytes, lhs.first.value(),
                             lhs.last.value()) <
                         std::make_tuple(
                             rhs.logical_identity, rhs.layer, rhs.offset,
                             rhs.root_bytes, rhs.first.value(),
                             rhs.last.value());
              });
    std::vector<RawUse> canonical_uses;
    for (const RawUse& use : raw_uses) {
        if (!canonical_uses.empty()) {
            RawUse& previous = canonical_uses.back();
            const bool same_allocation =
                previous.logical_identity == use.logical_identity &&
                previous.layer == use.layer &&
                previous.offset == use.offset &&
                previous.root_bytes == use.root_bytes;
            const bool adjacent_or_overlapping =
                use.first.value() <= previous.last.value() ||
                (previous.last.value() !=
                     std::numeric_limits<uint32_t>::max() &&
                 use.first.value() == previous.last.value() + 1);
            if (same_allocation && adjacent_or_overlapping) {
                if (previous.last.value() < use.last.value()) {
                    previous.last = use.last;
                }
                continue;
            }
        }
        canonical_uses.push_back(use);
    }

    std::vector<SpmPhaseSlice> ranges;
    ranges.reserve(canonical_uses.size());
    for (const RawUse& use : canonical_uses) {
        ranges.emplace_back(arena, use.offset, use.root_bytes,
                            use.first, use.last);
    }
    std::sort(ranges.begin(), ranges.end(),
              [](const SpmPhaseSlice& lhs, const SpmPhaseSlice& rhs) {
                  return std::make_tuple(
                             lhs.offset, lhs.bytes,
                             lhs.first_event.value(),
                             lhs.last_event.value()) <
                         std::make_tuple(
                             rhs.offset, rhs.bytes,
                             rhs.first_event.value(),
                             rhs.last_event.value());
              });
    TORCH_CHECK(!ranges.empty(),
                "seal_spm_pipeline_manifest: canonical range table is empty");

    uint64_t schedule_hash = fmb_hash_u64(kFmbFnvOffset, 4);
    schedule_hash = fmb_hash_u64(schedule_hash, schedule.version);
    schedule_hash = fmb_hash_u64(schedule_hash, occurrences.size());
    for (const SpmFmbOccurrence& occurrence : occurrences) {
        schedule_hash = fmb_hash_u64(schedule_hash, occurrence.key);
        schedule_hash = fmb_hash_u64(
            schedule_hash, static_cast<uint8_t>(occurrence.kind));
        schedule_hash = fmb_hash_u64(schedule_hash, occurrence.body_id);
        schedule_hash = fmb_hash_u64(schedule_hash, occurrence.group_id);
        schedule_hash = fmb_hash_u64(schedule_hash,
                                     occurrence.chunk_ordinal);
        schedule_hash = fmb_hash_u64(schedule_hash,
                                     occurrence.repetition_ordinal);
        schedule_hash = fmb_hash_u64(
            schedule_hash, static_cast<uint8_t>(occurrence.scope));
        schedule_hash = fmb_hash_u64(
            schedule_hash, static_cast<uint64_t>(occurrence.layer));
        schedule_hash = fmb_hash_u64(
            schedule_hash,
            static_cast<uint64_t>(occurrence.local_phase_begin));
        schedule_hash = fmb_hash_u64(
            schedule_hash,
            static_cast<uint64_t>(occurrence.local_phase_end));
        schedule_hash = fmb_hash_u64(
            schedule_hash, occurrence.global_origin.value());
    }
    schedule_hash = fmb_hash_u64(schedule_hash, extensions.size());
    for (const SpmFmbNamedUseExtension& extension : extensions) {
        schedule_hash = fmb_hash_string(schedule_hash, extension.name);
        schedule_hash = fmb_hash_u64(
            schedule_hash, static_cast<uint64_t>(extension.layer));
        schedule_hash = fmb_hash_u64(schedule_hash,
                                     extension.occurrence_key);
        schedule_hash = fmb_hash_u64(
            schedule_hash,
            static_cast<uint64_t>(extension.local_phase_begin));
        schedule_hash = fmb_hash_u64(
            schedule_hash,
            static_cast<uint64_t>(extension.local_phase_end));
    }
    schedule_hash = fmb_nonzero_hash(schedule_hash);

    uint64_t range_hash = fmb_hash_u64(kFmbFnvOffset, 5);
    range_hash = fmb_hash_u64(range_hash, ranges.size());
    for (const SpmPhaseSlice& range : ranges) {
        range_hash = fmb_hash_u64(range_hash, range.offset);
        range_hash = fmb_hash_u64(range_hash, range.bytes);
        range_hash = fmb_hash_u64(range_hash,
                                  range.first_event.value());
        range_hash = fmb_hash_u64(range_hash,
                                  range.last_event.value());
    }
    range_hash = fmb_nonzero_hash(range_hash);

    uint64_t identity_hash = fmb_hash_u64(kFmbFnvOffset, 6);
    identity_hash = fmb_hash_u64(identity_hash, arena.value());
    identity_hash = fmb_hash_u64(identity_hash, arena_base);
    identity_hash = fmb_hash_u64(identity_hash, snapshot.required_extent);
    identity_hash = fmb_hash_u64(identity_hash, snapshot.layout_hash);
    identity_hash = fmb_hash_u64(identity_hash,
                                 snapshot.declaration_hash);
    identity_hash = fmb_hash_u64(identity_hash,
                                 snapshot.persistent_hash);
    identity_hash = fmb_hash_u64(identity_hash,
                                 snapshot.allocation_hash);
    identity_hash = fmb_hash_u64(identity_hash, schedule_hash);
    identity_hash = fmb_hash_u64(identity_hash, range_hash);
    identity_hash = fmb_hash_u64(
        identity_hash, snapshot.temporary_allocations.size());
    identity_hash = fmb_hash_u64(identity_hash, ranges.size());
    identity_hash = fmb_hash_u64(identity_hash, snapshot.cpu_dry);
    identity_hash = fmb_nonzero_hash(identity_hash);

    SpmComponentPhaseManifest manifest{
        arena, arena_base, snapshot.required_extent, identity_hash,
        std::move(ranges)};
    return SpmFmbSealedPhaseManifest(
        manifest, snapshot.required_extent, snapshot.layout_hash,
        snapshot.declaration_hash, snapshot.persistent_hash,
        snapshot.allocation_hash, schedule_hash, range_hash,
        snapshot.temporary_allocations.size(), identity_hash, snapshot.cpu_dry,
        pimpl_->pipeline_manifest_generation_,
        *pimpl_->pipeline_manifest_generation_);
}

SpmFmbResolvedPhaseManifest
FusedModelBase::seal_spm_pipeline_resolved_manifest(
    const SpmFmbResolvedExecutionProfile& profile,
    SpmScratchId arena,
    uint32_t arena_base) const {
    const CachedLayoutSnapshot& snapshot =
        pimpl_->pipeline_manifest_snapshot_;
    TORCH_CHECK(snapshot.valid,
                "schema-v7 resolved seal requires a prepared owned "
                "allocation snapshot");
    TORCH_CHECK(snapshot.cpu_dry,
                "schema-v7 resolved seal accepts only a CPU-contract "
                "allocation snapshot");
    TORCH_CHECK(profile.cpu_dry_,
                "schema-v7 resolved seal accepts only a CPU-dry execution "
                "profile");
    return seal_spm_pipeline_resolved_manifest_impl(
        profile, arena, arena_base, /*require_cpu_dry=*/true);
}

SpmFmbResolvedPhaseManifest
FusedModelBase::seal_spm_pipeline_live_resolved_manifest(
    const SpmFmbResolvedExecutionProfile& profile,
    SpmScratchId arena,
    uint32_t arena_base) const {
    const CachedLayoutSnapshot& snapshot =
        pimpl_->pipeline_manifest_snapshot_;
    TORCH_CHECK(snapshot.valid,
                "live resolved seal requires a prepared owned allocation "
                "snapshot");
    TORCH_CHECK(!snapshot.cpu_dry,
                "live resolved seal accepts only a real allocation snapshot");
    TORCH_CHECK(!profile.cpu_dry_,
                "live resolved seal accepts only a live execution profile");
    return seal_spm_pipeline_resolved_manifest_impl(
        profile, arena, arena_base, /*require_cpu_dry=*/false);
}

SpmFmbResolvedPhaseManifest
FusedModelBase::seal_spm_pipeline_resolved_manifest_impl(
    const SpmFmbResolvedExecutionProfile& profile,
    SpmScratchId arena,
    uint32_t arena_base,
    bool require_cpu_dry) const {
    const CachedLayoutSnapshot& snapshot =
        pimpl_->pipeline_manifest_snapshot_;
    TORCH_INTERNAL_ASSERT(snapshot.valid &&
                          snapshot.cpu_dry == require_cpu_dry &&
                          profile.cpu_dry_ == require_cpu_dry);
    const std::shared_ptr<const uint64_t> profile_owner =
        profile.owner_generation_.lock();
    TORCH_CHECK(profile_owner != nullptr &&
                    profile_owner.get() ==
                        pimpl_->pipeline_manifest_generation_.get() &&
                    *profile_owner == profile.expected_owner_generation_,
                "schema-v7 resolved seal: execution profile owner is stale");
    TORCH_CHECK(profile.expected_layout_hash_ == snapshot.layout_hash &&
                    profile.expected_allocation_hash_ ==
                        snapshot.allocation_hash,
                "schema-v7 resolved seal: execution profile allocation "
                "identity mismatch");
    TORCH_CHECK(profile.profile_hash_ != 0 && !profile.steps_.empty(),
                "schema-v7 resolved seal: empty execution profile");
    TORCH_CHECK(arena.value() != 0,
                "schema-v7 resolved seal: arena ID zero is reserved");
    TORCH_CHECK(arena_base % SpmAllocator::ALIGN == 0 &&
                    snapshot.required_extent > 0 &&
                    snapshot.required_extent % SpmAllocator::ALIGN == 0,
                "schema-v7 resolved seal: arena base/extent must be "
                "256-byte aligned");
    TORCH_CHECK(snapshot.required_extent <=
                    std::numeric_limits<uint32_t>::max() - arena_base,
                "schema-v7 resolved seal: placed arena exceeds uint32");

    std::map<std::pair<std::string, int>, const OwnedPipelineDecl*>
        declaration_by_instance;
    std::map<std::pair<std::string, int>, const PipelineAllocationEntry*>
        allocation_by_instance;
    std::map<std::string, const OwnedPipelineDecl*> declaration_by_name;
    for (const OwnedPipelineDecl& decl : snapshot.decls) {
        declaration_by_name.emplace(decl.name, &decl);
    }
    for (const PipelineAllocationEntry& allocation :
         snapshot.temporary_allocations) {
        const auto key = std::make_pair(allocation.name, allocation.layer);
        TORCH_CHECK(allocation_by_instance.emplace(key, &allocation).second,
                    "schema-v7 resolved seal: duplicate allocation '",
                    allocation.name, "' layer ", allocation.layer);
        declaration_by_instance.emplace(
            key, declaration_by_name.at(allocation.name));
    }

    struct RawUse {
        std::string declaration_name;
        std::string logical_identity;
        int layer = -1;
        uint32_t offset = 0;
        size_t bytes = 0;
        SpmPipelineEvent first;
        SpmPipelineEvent last;
    };
    std::vector<RawUse> raw_uses;
    std::map<std::pair<std::string, int>, size_t> mapped_count;
    auto map_local_event = [](const SpmFmbResolvedExecutionStep& step,
                              int local_phase) {
        TORCH_CHECK(local_phase >= step.local_phase_begin &&
                        local_phase <= step.local_phase_end,
                    "schema-v7 resolved seal: declaration phase escapes "
                    "resolved step window");
        const uint64_t delta = static_cast<uint64_t>(
            local_phase - step.local_phase_begin);
        TORCH_CHECK(delta <= std::numeric_limits<uint32_t>::max() -
                                 step.global_origin.value(),
                    "schema-v7 resolved seal: global event overflow");
        return SpmPipelineEvent(
            step.global_origin.value() + static_cast<uint32_t>(delta));
    };

    for (const auto& item : allocation_by_instance) {
        const PipelineAllocationEntry& allocation = *item.second;
        const OwnedPipelineDecl& decl =
            *declaration_by_instance.at(item.first);
        for (const SpmFmbResolvedExecutionStep& step : profile.steps_) {
            if ((step.scope_mask & spm_fmb_scope_bit(decl.scope)) == 0) {
                continue;
            }
            if (decl.storage == StorageClass::TempPerLayer &&
                step.layer != allocation.layer) {
                continue;
            }
            auto root = allocation_by_instance.find(
                std::make_pair(allocation.alias_root, allocation.layer));
            TORCH_CHECK(root != allocation_by_instance.end(),
                        "schema-v7 resolved seal: missing alias root '",
                        allocation.alias_root, "'");
            const bool whole_root_synonym =
                !decl.alias_of.empty() && decl.logical_bytes == 0;
            const std::string logical_identity =
                decl.alias_of.empty() || whole_root_synonym
                ? allocation.alias_root : allocation.name;
            const size_t protected_bytes =
                decl.alias_of.empty() || whole_root_synonym
                ? root->second->aligned_bytes : allocation.aligned_bytes;
            raw_uses.push_back(
                {allocation.name, logical_identity, allocation.layer,
                 root->second->offset, protected_bytes,
                 map_local_event(step, decl.phase_start),
                 map_local_event(step, decl.phase_end)});
            ++mapped_count[item.first];
        }
        TORCH_CHECK(mapped_count[item.first] > 0,
                    "schema-v7 resolved seal: no framework step maps '",
                    allocation.name, "' layer ", allocation.layer);
    }

    auto event_overlap = [](const RawUse& lhs, const RawUse& rhs) {
        return !(lhs.last.value() < rhs.first.value() ||
                 rhs.last.value() < lhs.first.value());
    };
    auto byte_overlap = [](const RawUse& lhs, const RawUse& rhs) {
        const uint64_t lhs_end =
            static_cast<uint64_t>(lhs.offset) + lhs.bytes;
        const uint64_t rhs_end =
            static_cast<uint64_t>(rhs.offset) + rhs.bytes;
        return lhs.offset < rhs_end && rhs.offset < lhs_end;
    };
    for (size_t lhs = 0; lhs < raw_uses.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < raw_uses.size(); ++rhs) {
            TORCH_CHECK(
                !byte_overlap(raw_uses[lhs], raw_uses[rhs]) ||
                    !event_overlap(raw_uses[lhs], raw_uses[rhs]) ||
                    (raw_uses[lhs].logical_identity ==
                         raw_uses[rhs].logical_identity &&
                     raw_uses[lhs].layer == raw_uses[rhs].layer),
                "schema-v7 resolved seal: distinct logical allocations '",
                raw_uses[lhs].declaration_name, "' and '",
                raw_uses[rhs].declaration_name,
                "' overlap in byte x global-event space");
        }
    }

    std::sort(raw_uses.begin(), raw_uses.end(),
              [](const RawUse& lhs, const RawUse& rhs) {
                  return std::make_tuple(
                             lhs.logical_identity, lhs.layer, lhs.offset,
                             lhs.bytes, lhs.first.value(), lhs.last.value()) <
                         std::make_tuple(
                             rhs.logical_identity, rhs.layer, rhs.offset,
                             rhs.bytes, rhs.first.value(), rhs.last.value());
              });
    std::vector<RawUse> canonical_uses;
    for (const RawUse& use : raw_uses) {
        if (!canonical_uses.empty()) {
            RawUse& previous = canonical_uses.back();
            const bool same_allocation =
                previous.logical_identity == use.logical_identity &&
                previous.layer == use.layer &&
                previous.offset == use.offset &&
                previous.bytes == use.bytes;
            const bool adjacent_or_overlapping =
                use.first.value() <= previous.last.value() ||
                (previous.last.value() !=
                     std::numeric_limits<uint32_t>::max() &&
                 use.first.value() == previous.last.value() + 1);
            if (same_allocation && adjacent_or_overlapping) {
                if (previous.last.value() < use.last.value()) {
                    previous.last = use.last;
                }
                continue;
            }
        }
        canonical_uses.push_back(use);
    }

    std::vector<SpmPhaseSlice> ranges;
    ranges.reserve(canonical_uses.size());
    for (const RawUse& use : canonical_uses) {
        ranges.emplace_back(arena, use.offset, use.bytes,
                            use.first, use.last);
    }
    std::sort(ranges.begin(), ranges.end(),
              [](const SpmPhaseSlice& lhs, const SpmPhaseSlice& rhs) {
                  return std::make_tuple(
                             lhs.offset, lhs.bytes,
                             lhs.first_event.value(),
                             lhs.last_event.value()) <
                         std::make_tuple(
                             rhs.offset, rhs.bytes,
                             rhs.first_event.value(),
                             rhs.last_event.value());
              });
    TORCH_CHECK(!ranges.empty(),
                "schema-v7 resolved seal: canonical range table is empty");

    uint64_t range_hash = fmb_hash_u64(kFmbFnvOffset, 7);
    range_hash = fmb_hash_u64(range_hash, ranges.size());
    for (const SpmPhaseSlice& range : ranges) {
        range_hash = fmb_hash_u64(range_hash, range.offset);
        range_hash = fmb_hash_u64(range_hash, range.bytes);
        range_hash = fmb_hash_u64(range_hash,
                                  range.first_event.value());
        range_hash = fmb_hash_u64(range_hash,
                                  range.last_event.value());
    }
    range_hash = fmb_nonzero_hash(range_hash);

    uint64_t identity_hash = fmb_hash_u64(kFmbFnvOffset, 8);
    identity_hash = fmb_hash_u64(identity_hash, arena.value());
    identity_hash = fmb_hash_u64(identity_hash, arena_base);
    identity_hash = fmb_hash_u64(identity_hash, snapshot.required_extent);
    identity_hash = fmb_hash_u64(identity_hash, snapshot.layout_hash);
    identity_hash = fmb_hash_u64(identity_hash,
                                 snapshot.declaration_hash);
    identity_hash = fmb_hash_u64(identity_hash,
                                 snapshot.persistent_hash);
    identity_hash = fmb_hash_u64(identity_hash,
                                 snapshot.allocation_hash);
    identity_hash = fmb_hash_u64(identity_hash, profile.profile_hash_);
    identity_hash = fmb_hash_u64(identity_hash, range_hash);
    identity_hash = fmb_hash_u64(
        identity_hash, snapshot.temporary_allocations.size());
    identity_hash = fmb_hash_u64(identity_hash, ranges.size());
    identity_hash = fmb_nonzero_hash(identity_hash);

    SpmComponentPhaseManifest manifest{
        arena, arena_base, snapshot.required_extent, identity_hash,
        std::move(ranges)};
    return SpmFmbResolvedPhaseManifest(
        manifest, snapshot.required_extent, snapshot.layout_hash,
        snapshot.declaration_hash, snapshot.persistent_hash,
        snapshot.allocation_hash, profile.profile_hash_, range_hash,
        snapshot.temporary_allocations.size(), identity_hash,
        snapshot.cpu_dry,
        pimpl_->pipeline_manifest_generation_,
        *pimpl_->pipeline_manifest_generation_);
}

SpmFmbConsumerRowSliceEndpoint
FusedModelBase::seal_spm_pipeline_consumer_row_slice_endpoint(
    const SpmFmbResolvedExecutionProfile& profile,
    const SpmFmbResolvedPhaseManifest& live_manifest,
    const char* owned_buffer_name,
    int allocation_layer,
    SpmChunkKey consumer_chunk,
    const SpmDense2DSpec& storage_spec,
    int64_t row_begin,
    int64_t row_count,
    const SpmFmbConsumerOccurrenceSelector& occurrence) const {
    const CachedLayoutSnapshot& snapshot =
        pimpl_->pipeline_manifest_snapshot_;
    TORCH_CHECK(snapshot.valid && !snapshot.cpu_dry &&
                    !profile.cpu_dry_ && !live_manifest.cpu_dry_,
                "consumer row-slice endpoint requires live allocation, "
                "profile, and manifest authority");
    TORCH_CHECK(owned_buffer_name != nullptr && *owned_buffer_name != '\0',
                "consumer row-slice endpoint requires an owned buffer name");
    TORCH_CHECK(consumer_chunk.port().value() != 0,
                "consumer row-slice endpoint port ID zero is reserved");

    const std::shared_ptr<const uint64_t> profile_owner =
        profile.owner_generation_.lock();
    const std::shared_ptr<const uint64_t> manifest_owner =
        live_manifest.owner_generation_.lock();
    TORCH_CHECK(
        profile_owner != nullptr && manifest_owner != nullptr &&
            profile_owner.get() ==
                pimpl_->pipeline_manifest_generation_.get() &&
            manifest_owner.get() ==
                pimpl_->pipeline_manifest_generation_.get() &&
            profile.expected_owner_generation_ ==
                live_manifest.expected_owner_generation_ &&
            *profile_owner == profile.expected_owner_generation_ &&
            *manifest_owner == live_manifest.expected_owner_generation_,
        "consumer row-slice endpoint owner is stale or foreign");
    TORCH_CHECK(
        profile.expected_layout_hash_ == snapshot.layout_hash &&
            profile.expected_allocation_hash_ == snapshot.allocation_hash &&
            live_manifest.layout_hash_ == snapshot.layout_hash &&
            live_manifest.declaration_hash_ ==
                snapshot.declaration_hash &&
            live_manifest.persistent_hash_ == snapshot.persistent_hash &&
            live_manifest.allocation_hash_ == snapshot.allocation_hash &&
            live_manifest.profile_hash_ == profile.profile_hash_ &&
            live_manifest.identity_hash_ != 0 &&
            live_manifest.range_hash_ != 0,
        "consumer row-slice endpoint allocation/profile/manifest identity "
        "mismatch");

    uint32_t allocation_ordinal = 0;
    const PipelineAllocationEntry* allocation = find_trace_allocation(
        snapshot, owned_buffer_name, allocation_layer,
        &allocation_ordinal);
    TORCH_CHECK(allocation != nullptr,
                "consumer row-slice endpoint has no owned allocation '",
                owned_buffer_name, "' layer ", allocation_layer);
    uint32_t declaration_ordinal = 0;
    const OwnedPipelineDecl* declaration = find_trace_declaration(
        snapshot, owned_buffer_name, &declaration_ordinal);
    TORCH_INTERNAL_ASSERT(declaration != nullptr);
    const uint32_t alias_root_ordinal =
        find_trace_root_ordinal(snapshot, *allocation);
    uint32_t root_lookup_ordinal = 0;
    const PipelineAllocationEntry* root = find_trace_allocation(
        snapshot, allocation->alias_root.c_str(), allocation->layer,
        &root_lookup_ordinal);
    TORCH_INTERNAL_ASSERT(root != nullptr &&
                          root_lookup_ordinal == alias_root_ordinal);

    TORCH_CHECK(
        allocation->storage == StorageClass::Temp &&
            declaration->storage == StorageClass::Temp &&
            declaration->scope == BufferScope::LayerWide &&
            allocation->layer == -1 && allocation_layer == -1,
        "consumer row-slice endpoint initially admits only flat Temp + "
        "LayerWide allocations");
    TORCH_CHECK(
        declaration->alias_of.empty() &&
            allocation->alias_root == allocation->name &&
            root == allocation && !declaration->reverse_layer_alloc &&
            !allocation->reverse_layer_alloc &&
            !declaration->preload_present &&
            !allocation->preload_present,
        "consumer row-slice endpoint rejects aliases, reverse allocation, "
        "and preload callbacks");

    storage_spec.validate();
    TORCH_CHECK(storage_spec.dtype == SpmPortDType::Fp16 &&
                    storage_spec.distribution ==
                        SpmPortDistribution::Replicated &&
                    storage_spec.cols % 16 == 0,
                "consumer row-slice endpoint requires replicated FP16 with "
                "columns divisible by 16");
    const size_t storage_bytes = storage_spec.storage_bytes();
    TORCH_CHECK(storage_bytes == allocation->logical_bytes &&
                    storage_bytes == allocation->aligned_bytes &&
                    allocation->aligned_bytes == root->aligned_bytes,
                "consumer row-slice endpoint dense storage must exactly "
                "cover its owned root allocation");
    TORCH_CHECK(row_begin >= 0 && row_count > 0 &&
                    row_begin <= storage_spec.rows &&
                    row_count <= storage_spec.rows - row_begin,
                "consumer row-slice endpoint row range is empty or out of "
                "bounds");
    const size_t row_bytes =
        storage_bytes / static_cast<size_t>(storage_spec.rows);
    const size_t slice_byte_begin =
        static_cast<size_t>(row_begin) * row_bytes;
    const size_t slice_byte_count =
        static_cast<size_t>(row_count) * row_bytes;
    TORCH_INTERNAL_ASSERT(slice_byte_begin <= storage_bytes &&
                          slice_byte_count <=
                              storage_bytes - slice_byte_begin);

    const SpmFmbResolvedExecutionStep* selected_step = nullptr;
    size_t selected_count = 0;
    for (const SpmFmbResolvedExecutionStep& step : profile.steps_) {
        if (step.kind == occurrence.kind &&
            step.body_id == occurrence.body_id &&
            step.group_id == occurrence.group_id &&
            step.layer == occurrence.layer &&
            step.chunk_index == occurrence.chunk_index) {
            selected_step = &step;
            ++selected_count;
        }
    }
    TORCH_CHECK(selected_count == 1 && selected_step != nullptr,
                "consumer row-slice endpoint occurrence selector matched ",
                selected_count, " resolved steps instead of exactly one");
    TORCH_CHECK(
        (selected_step->scope_mask &
         spm_fmb_scope_bit(declaration->scope)) != 0 &&
            declaration->phase_start >=
                selected_step->local_phase_begin &&
            declaration->phase_end <= selected_step->local_phase_end,
        "consumer row-slice endpoint declaration is outside the selected "
        "resolved occurrence");
    const uint64_t first_delta = static_cast<uint64_t>(
        declaration->phase_start - selected_step->local_phase_begin);
    const uint64_t last_delta = static_cast<uint64_t>(
        declaration->phase_end - selected_step->local_phase_begin);
    TORCH_CHECK(
        first_delta <= std::numeric_limits<uint32_t>::max() -
                           selected_step->global_origin.value() &&
            last_delta <= std::numeric_limits<uint32_t>::max() -
                              selected_step->global_origin.value(),
        "consumer row-slice endpoint global event overflows uint32");
    const SpmPipelineEvent global_first_event(
        selected_step->global_origin.value() +
        static_cast<uint32_t>(first_delta));
    const SpmPipelineEvent global_last_event(
        selected_step->global_origin.value() +
        static_cast<uint32_t>(last_delta));

    const SpmPhaseSlice* selected_range = nullptr;
    size_t selected_range_count = 0;
    for (const SpmPhaseSlice& range :
         live_manifest.manifest_.live_ranges) {
        if (range.arena == live_manifest.manifest_.arena &&
            range.offset == root->offset &&
            range.bytes == root->aligned_bytes &&
            range.first_event.value() <= global_first_event.value() &&
            global_last_event.value() <= range.last_event.value()) {
            selected_range = &range;
            ++selected_range_count;
        }
    }
    TORCH_CHECK(
        selected_range_count == 1 && selected_range != nullptr,
        "consumer row-slice endpoint found ", selected_range_count,
        " canonical live ranges covering the selected allocation/occurrence "
        "instead of exactly one");

    const uint64_t declaration_name_hash = fmb_nonzero_hash(
        fmb_hash_string(kFmbFnvOffset, allocation->name));
    const uint64_t alias_root_name_hash = fmb_nonzero_hash(
        fmb_hash_string(kFmbFnvOffset, allocation->alias_root));
    uint64_t identity_hash = fmb_hash_u64(kFmbFnvOffset, 26);
    const std::vector<uint64_t> identity_values = {
             live_manifest.identity_hash_, live_manifest.layout_hash_,
             live_manifest.declaration_hash_,
             live_manifest.persistent_hash_,
             live_manifest.allocation_hash_, live_manifest.profile_hash_,
             live_manifest.range_hash_,
             static_cast<uint64_t>(consumer_chunk.port().value()),
             static_cast<uint64_t>(consumer_chunk.ordinal()),
             static_cast<uint64_t>(storage_spec.dtype),
             static_cast<uint64_t>(storage_spec.rows),
             static_cast<uint64_t>(storage_spec.cols),
             static_cast<uint64_t>(storage_spec.distribution),
             static_cast<uint64_t>(row_begin),
             static_cast<uint64_t>(row_count), allocation_ordinal,
             declaration_ordinal, alias_root_ordinal,
             declaration_name_hash, alias_root_name_hash,
             static_cast<uint64_t>(allocation->storage),
             static_cast<uint64_t>(declaration->scope),
             static_cast<uint64_t>(allocation->layer),
             static_cast<uint64_t>(declaration->phase_start),
             static_cast<uint64_t>(declaration->phase_end), root->offset,
             allocation->offset - root->offset, root->aligned_bytes,
             allocation->logical_bytes, allocation->aligned_bytes,
             row_bytes, slice_byte_begin, slice_byte_count,
             static_cast<uint64_t>(selected_step->kind),
             selected_step->key, selected_step->body_id,
             selected_step->group_id,
             static_cast<uint64_t>(selected_step->layer),
             selected_step->traversal_ordinal,
             static_cast<uint64_t>(selected_step->chunk_index),
             global_first_event.value(), global_last_event.value(),
             selected_range->first_event.value(),
             selected_range->last_event.value()};
    for (uint64_t value : identity_values) {
        identity_hash = fmb_hash_u64(identity_hash, value);
    }
    identity_hash = fmb_nonzero_hash(identity_hash);

    const SpmBorrowedConsumerChunkRef consumer{
        consumer_chunk, storage_spec, *selected_range};
    return SpmFmbConsumerRowSliceEndpoint(
        live_manifest, consumer, row_begin, row_count,
        allocation_ordinal, declaration_ordinal, alias_root_ordinal,
        declaration_name_hash, alias_root_name_hash,
        allocation->offset - root->offset, root->aligned_bytes,
        allocation->logical_bytes, allocation->aligned_bytes,
        selected_step->kind, selected_step->key,
        selected_step->body_id, selected_step->group_id,
        selected_step->layer, selected_step->traversal_ordinal,
        selected_step->chunk_index, global_first_event, global_last_event,
        identity_hash);
}

SpmFmbPostFnYieldScope
FusedModelBase::begin_spm_pipeline_post_fn_dense_yield(
    const char* owned_preflight_output_name,
    int allocation_layer,
    const SpmDense2DSpec& produced_spec) {
    return begin_spm_pipeline_dense_producer_yield_impl(
        owned_preflight_output_name, allocation_layer, produced_spec,
        /*callback_window_contract=*/false);
}

SpmFmbPostFnYieldScope
FusedModelBase::begin_spm_pipeline_dense_producer_yield(
    const char* owned_preflight_output_name,
    int allocation_layer,
    const SpmDense2DSpec& produced_spec) {
    return begin_spm_pipeline_dense_producer_yield_impl(
        owned_preflight_output_name, allocation_layer, produced_spec,
        /*callback_window_contract=*/true);
}

SpmFmbPostFnYieldScope
FusedModelBase::begin_spm_pipeline_dense_producer_yield_impl(
    const char* owned_preflight_output_name,
    int allocation_layer,
    const SpmDense2DSpec& produced_spec,
    bool callback_window_contract) {
    auto& candidate = pimpl_->pipeline_build_trace_;
    auto& replay = pimpl_->pipeline_post_fn_yield_replay_authority_;
    auto& callback_replay =
        pimpl_->pipeline_callback_yield_replay_authority_;
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "post_fn producer yield requires an active Graph");
    RpuKernelGraph& graph = RpuKernelGraph::active();
    const bool recording = graph.state() == RpuKernelGraph::State::RECORDING;
    const bool replaying = graph.state() == RpuKernelGraph::State::REPLAYING;
    TORCH_CHECK(recording || replaying,
                "post_fn producer yield requires Graph BUILD or REPLAY");
    if (recording) {
        const bool legacy_post_fn_contract =
            !callback_window_contract && candidate.armed &&
            candidate.profile != nullptr && candidate.callback_open &&
            !candidate.callbacks.empty() &&
            candidate.callbacks.back().kind ==
                SpmFmbOccurrenceKind::PostFn &&
            candidate.profile->post_fn_yield_capability_ ==
                SpmFmbPostFnYieldCapability::
                    CanonicalDenseReplicatedFp16;
        const bool generic_callback_contract =
            callback_window_contract && candidate.armed &&
            candidate.profile != nullptr && candidate.callback_open &&
            !candidate.callbacks.empty() &&
            (candidate.callbacks.back().kind ==
                 SpmFmbOccurrenceKind::LayerBody ||
             candidate.callbacks.back().kind ==
                 SpmFmbOccurrenceKind::PostFn) &&
            candidate.profile->layer_producer_yield_capability_ ==
                SpmFmbLayerProducerYieldCapability::
                    CanonicalDenseReplicatedFp16;
        TORCH_CHECK(
            legacy_post_fn_contract || generic_callback_contract,
            "producer yield requires an armed resolved PostFn or LayerBody "
            "BUILD callback under its matching capability");
        TORCH_CHECK(
            !candidate.producer_yield_open &&
                !candidate.pending_producer_yield.has_value() &&
                !g_fmb_post_fn_yield_begin.has_value(),
            "post_fn producer yield BUILD scopes cannot overlap");
    } else {
        const bool legacy_replay =
            !callback_window_contract && replay.armed &&
            !replay.yield_open && replay.yields != nullptr &&
            replay.next_yield < replay.yields->yields_.size() &&
            replay.graph_identity == &graph;
        const bool callback_replay_ready =
            callback_window_contract && callback_replay.armed &&
            !callback_replay.yield_open &&
            callback_replay.yields != nullptr &&
            callback_replay.next_yield <
                callback_replay.yields->yields_.size() &&
            callback_replay.graph_identity == &graph;
        TORCH_CHECK(
            (legacy_replay || callback_replay_ready) &&
                !g_fmb_post_fn_yield_begin.has_value(),
            "producer yield REPLAY authority is absent, stale, or already "
            "open");
    }
    TORCH_CHECK(
        owned_preflight_output_name != nullptr &&
            *owned_preflight_output_name != '\0',
        "post_fn producer yield requires an owned preflight output name");
    produced_spec.validate();
    TORCH_CHECK(
        produced_spec.dtype == SpmPortDType::Fp16 &&
            produced_spec.distribution ==
                SpmPortDistribution::Replicated &&
            produced_spec.cols % 16 == 0,
        "post_fn producer yield requires replicated FP16 with columns "
        "divisible by 16");

    const CachedLayoutSnapshot& snapshot =
        pimpl_->pipeline_manifest_snapshot_;
    uint32_t allocation_ordinal = 0;
    const PipelineAllocationEntry* allocation = find_trace_allocation(
        snapshot, owned_preflight_output_name, allocation_layer,
        &allocation_ordinal);
    TORCH_CHECK(allocation != nullptr,
                "post_fn producer yield has no owned allocation '",
                owned_preflight_output_name, "' layer ", allocation_layer);
    uint32_t declaration_ordinal = 0;
    const OwnedPipelineDecl* declaration = find_trace_declaration(
        snapshot, owned_preflight_output_name, &declaration_ordinal);
    TORCH_INTERNAL_ASSERT(declaration != nullptr);
    const uint32_t alias_root_ordinal =
        find_trace_root_ordinal(snapshot, *allocation);
    uint32_t root_lookup_ordinal = 0;
    const PipelineAllocationEntry* root = find_trace_allocation(
        snapshot, allocation->alias_root.c_str(), allocation->layer,
        &root_lookup_ordinal);
    TORCH_INTERNAL_ASSERT(root != nullptr &&
                          root_lookup_ordinal == alias_root_ordinal);
    TORCH_CHECK(
        allocation->storage == StorageClass::Temp &&
            declaration->storage == StorageClass::Temp &&
            allocation->layer == -1 && allocation_layer == -1 &&
            (declaration->scope == BufferScope::LayerWide ||
             declaration->scope == BufferScope::Compute) &&
            !declaration->reverse_layer_alloc &&
            !allocation->reverse_layer_alloc &&
            !declaration->preload_present &&
            !allocation->preload_present,
        "post_fn producer yield initially accepts only flat temporary "
        "LayerWide/Compute outputs without preload or reverse allocation");
    const size_t produced_bytes = produced_spec.storage_bytes();
    TORCH_CHECK(
        produced_bytes == allocation->logical_bytes &&
            produced_bytes == allocation->aligned_bytes &&
            allocation->offset >= root->offset &&
            allocation->offset - root->offset <= root->aligned_bytes &&
            allocation->aligned_bytes <=
                root->aligned_bytes - (allocation->offset - root->offset),
        "post_fn producer yield dense spec does not exactly cover its owned "
        "preflight allocation");

    const uint32_t output_address =
        addr(/*core=*/0, owned_preflight_output_name);
    const GraphOpStreamStamp begin = graph.op_stream_stamp();
    uint32_t producer_output_address = output_address;
    auto& physical = pimpl_->composite_physical_execution_;
    if (C10_UNLIKELY(physical.armed)) {
        TORCH_CHECK(
            physical.producer && physical.lease != nullptr &&
                physical.graph_identity == &graph &&
                static_cast<uint8_t>(begin.state) == physical.graph_state &&
                begin.build_generation == physical.graph_build_generation &&
                begin.signature_identity ==
                    physical.graph_signature_identity &&
                begin.signature_segment_key ==
                    physical.graph_signature_segment_key &&
                physical.next_target < physical.expected_target_count,
            "composite producer yield has stale or exhausted physical "
            "authority");
        physical.lease->validate_composite_execution_authority(
            physical.lease_epoch, physical.plan_hash,
            physical.require_committed);
        TORCH_CHECK(
            physical.expected_target_count ==
                physical.lease->composite_yields_per_producer(),
            "composite producer yield count drifted after arm");
        producer_output_address =
            physical.lease->resolve_composite_producer_target_addr(
                /*core=*/0, physical.producer_local_ordinal,
                physical.next_target);
        TORCH_CHECK(
            producer_output_address != 0,
            "composite producer yield resolved a zero physical target");
        ++physical.next_target;
    }
    const uint64_t declaration_name_hash = fmb_nonzero_hash(
        fmb_hash_string(kFmbFnvOffset, allocation->name));
    const uint64_t alias_root_name_hash = fmb_nonzero_hash(
        fmb_hash_string(kFmbFnvOffset, allocation->alias_root));
    if (recording) {
        TORCH_CHECK(&graph == candidate.graph_identity &&
                        begin.build_generation ==
                            candidate.graph_build_generation &&
                        begin.signature_identity ==
                            candidate.graph_signature_identity &&
                        begin.signature_segment_key ==
                            candidate.graph_signature_segment_key,
                    "post_fn producer yield BUILD Graph identity drifted at "
                    "begin");
        size_t ordinal =
            candidate.callbacks.back().producer_yields.size();
        if (callback_window_contract) {
            ordinal = 0;
            for (const auto& callback : candidate.callbacks) {
                TORCH_CHECK(
                    callback.producer_yields.size() <=
                        std::numeric_limits<size_t>::max() - ordinal,
                    "callback producer yield ordinal overflow");
                ordinal += callback.producer_yields.size();
            }
        }
        uint64_t semantic_id = fmb_hash_u64(
            kFmbFnvOffset, callback_window_contract ? 31 : 28);
        const std::vector<uint64_t> identity_values = {
            candidate.profile_hash,
            candidate.allocation_hash,
            candidate.callbacks.back().key,
            ordinal,
            static_cast<uint8_t>(
                SpmFmbTerminalWriterKind::AllReduceSumResidual),
            static_cast<uint8_t>(produced_spec.dtype),
            static_cast<uint64_t>(produced_spec.rows),
            static_cast<uint64_t>(produced_spec.cols),
            static_cast<uint8_t>(produced_spec.distribution),
            allocation_ordinal,
            declaration_ordinal,
            alias_root_ordinal,
            declaration_name_hash,
            alias_root_name_hash,
            root->offset,
            allocation->offset - root->offset,
            allocation->logical_bytes,
            allocation->aligned_bytes,
            callback_window_contract
                ? candidate.profile
                      ->layer_producer_yield_policy_fingerprint_
                : candidate.profile->post_fn_yield_policy_fingerprint_,
        };
        for (uint64_t value : identity_values) {
            semantic_id = fmb_hash_u64(semantic_id, value);
        }
        if (candidate.composite) {
            semantic_id = fmb_hash_u64(
                semantic_id, UINT64_C(0x434f4d50594c4431));
            semantic_id = fmb_hash_u64(
                semantic_id, candidate.composite_identity);
            semantic_id = fmb_hash_u64(
                semantic_id, candidate.composite_occurrence_ordinal);
            if (candidate.composite_capability !=
                SpmFmbCompositeOccurrenceCapability::Unsealed) {
                semantic_id = fmb_hash_u64(
                    semantic_id, UINT64_C(0x434f4d50594c4432));
                semantic_id = fmb_hash_u64(
                    semantic_id, candidate.composite_producer);
                semantic_id = fmb_hash_u64(
                    semantic_id,
                    candidate.composite_producer_local_ordinal);
                semantic_id = fmb_hash_u64(
                    semantic_id,
                    static_cast<uint8_t>(
                        candidate.composite_capability));
                semantic_id = fmb_hash_u64(
                    semantic_id,
                    candidate.composite_policy_fingerprint);
                semantic_id = fmb_hash_u64(
                    semantic_id,
                    candidate.composite_occurrence_profile_hash);
                semantic_id = fmb_hash_u64(
                    semantic_id,
                    candidate.composite_preload_follower);
            }
        }
        semantic_id = fmb_nonzero_hash(semantic_id);
        auto target = std::unique_ptr<SpmFmbPostFnYieldTarget>(
            new SpmFmbPostFnYieldTarget(
                producer_output_address, produced_spec, semantic_id,
                SpmFmbTerminalWriterKind::AllReduceSumResidual,
                pimpl_->pipeline_manifest_generation_,
                candidate.expected_owner_generation));

        Impl::BuildTraceProducerYieldRecord pending;
        pending.ordinal = ordinal;
        pending.semantic_id = semantic_id;
        pending.writer_kind =
            SpmFmbTerminalWriterKind::AllReduceSumResidual;
        pending.spec = produced_spec;
        pending.allocation_ordinal = allocation_ordinal;
        pending.declaration_ordinal = declaration_ordinal;
        pending.alias_root_ordinal = alias_root_ordinal;
        pending.declaration_name_hash = declaration_name_hash;
        pending.alias_root_name_hash = alias_root_name_hash;
        pending.root_offset = root->offset;
        pending.allocation_offset_from_root =
            allocation->offset - root->offset;
        pending.logical_capacity = allocation->logical_bytes;
        pending.aligned_capacity = allocation->aligned_bytes;
        pending.graph_node_begin = begin.position;
        candidate.pending_producer_yield = std::move(pending);
        candidate.producer_yield_open = true;
        g_fmb_post_fn_yield_begin = begin;
        return SpmFmbPostFnYieldScope(
            this, std::move(target),
            callback_window_contract
                ? SpmFmbPostFnYieldScope::Contract::CallbackWindow
                : SpmFmbPostFnYieldScope::Contract::PostFnExact);
    }

    if (callback_window_contract) {
        const auto& sealed = *callback_replay.yields;
        const auto& trace = sealed.source_trace_;
        const auto& expected =
            sealed.yields_[callback_replay.next_yield];
        const size_t expected_begin =
            trace.graph_node_begin_ + expected.relative_node_begin;
        TORCH_CHECK(
            begin.graph == &graph &&
                begin.build_generation ==
                    callback_replay.graph_build_generation &&
                begin.signature_identity ==
                    callback_replay.graph_signature_identity &&
                begin.signature_segment_key ==
                    callback_replay.profile_hash &&
                begin.position == expected_begin &&
                expected.ordinal == callback_replay.next_yield &&
                expected.writer_kind ==
                    SpmFmbTerminalWriterKind::AllReduceSumResidual &&
                expected.spec == produced_spec &&
                expected.allocation_ordinal == allocation_ordinal &&
                expected.declaration_ordinal == declaration_ordinal &&
                expected.alias_root_ordinal == alias_root_ordinal &&
                expected.declaration_name_hash == declaration_name_hash &&
                expected.alias_root_name_hash == alias_root_name_hash &&
                expected.root_offset == root->offset &&
                expected.allocation_offset_from_root ==
                    allocation->offset - root->offset &&
                expected.logical_capacity == allocation->logical_bytes &&
                expected.aligned_capacity == allocation->aligned_bytes,
            "callback producer yield REPLAY allocation, order, or Graph "
            "cursor drifted from the sealed BUILD");
        auto target = std::unique_ptr<SpmFmbPostFnYieldTarget>(
            new SpmFmbPostFnYieldTarget(
                producer_output_address, produced_spec, expected.semantic_id,
                expected.writer_kind,
                pimpl_->pipeline_manifest_generation_,
                trace.expected_owner_generation_));
        callback_replay.yield_open = true;
        g_fmb_post_fn_yield_begin = begin;
        return SpmFmbPostFnYieldScope(
            this, std::move(target),
            SpmFmbPostFnYieldScope::Contract::CallbackWindow);
    }

    const auto& sealed = *replay.yields;
    const auto& trace = sealed.source_trace_;
    const auto& expected = sealed.yields_[replay.next_yield];
    const size_t expected_begin =
        trace.graph_node_begin_ + expected.relative_node_begin;
    TORCH_CHECK(
        begin.graph == &graph &&
            begin.build_generation == replay.graph_build_generation &&
            begin.signature_identity == replay.graph_signature_identity &&
            begin.signature_segment_key == replay.profile_hash &&
            begin.position == expected_begin &&
            expected.ordinal == replay.next_yield &&
            expected.writer_kind ==
                SpmFmbTerminalWriterKind::AllReduceSumResidual &&
            expected.spec == produced_spec &&
            expected.allocation_ordinal == allocation_ordinal &&
            expected.declaration_ordinal == declaration_ordinal &&
            expected.alias_root_ordinal == alias_root_ordinal &&
            expected.declaration_name_hash == declaration_name_hash &&
            expected.alias_root_name_hash == alias_root_name_hash &&
            expected.root_offset == root->offset &&
            expected.allocation_offset_from_root ==
                allocation->offset - root->offset &&
            expected.logical_capacity == allocation->logical_bytes &&
            expected.aligned_capacity == allocation->aligned_bytes,
        "post_fn producer yield REPLAY allocation, order, or Graph cursor "
        "drifted from the sealed BUILD");
    auto target = std::unique_ptr<SpmFmbPostFnYieldTarget>(
        new SpmFmbPostFnYieldTarget(
            producer_output_address, produced_spec, expected.semantic_id,
            expected.writer_kind, pimpl_->pipeline_manifest_generation_,
            trace.expected_owner_generation_));
    replay.yield_open = true;
    g_fmb_post_fn_yield_begin = begin;
    return SpmFmbPostFnYieldScope(
        this, std::move(target),
        SpmFmbPostFnYieldScope::Contract::PostFnExact);
}

void FusedModelBase::finish_spm_pipeline_post_fn_dense_yield(
    SpmFmbPostFnYieldScope& scope) {
    auto& candidate = pimpl_->pipeline_build_trace_;
    auto& replay = pimpl_->pipeline_post_fn_yield_replay_authority_;
    auto& callback_replay =
        pimpl_->pipeline_callback_yield_replay_authority_;
    TORCH_CHECK(scope.owner_ == this && scope.active_ &&
                    scope.target_ != nullptr &&
                    g_fmb_post_fn_yield_begin.has_value(),
                "post_fn producer yield finish has no matching open scope");
    RpuKernelGraph& graph = RpuKernelGraph::active();
    if (graph.state() == RpuKernelGraph::State::REPLAYING) {
        if (scope.contract_ ==
            SpmFmbPostFnYieldScope::Contract::CallbackWindow) {
            TORCH_CHECK(
                callback_replay.armed && callback_replay.yield_open &&
                    callback_replay.yields != nullptr &&
                    callback_replay.next_yield <
                        callback_replay.yields->yields_.size() &&
                    callback_replay.graph_identity == &graph,
                "callback producer yield REPLAY finish authority drifted");
            const GraphOpStreamStamp end = graph.op_stream_stamp();
            const auto& trace = callback_replay.yields->source_trace_;
            const auto& expected =
                callback_replay.yields
                    ->yields_[callback_replay.next_yield];
            const size_t expected_begin =
                trace.graph_node_begin_ + expected.relative_node_begin;
            const size_t expected_end =
                trace.graph_node_begin_ + expected.relative_node_end;
            TORCH_CHECK(
                g_fmb_post_fn_yield_begin->state ==
                        RpuKernelGraph::State::REPLAYING &&
                    g_fmb_post_fn_yield_begin->position == expected_begin &&
                    end.graph == &graph &&
                    end.build_generation ==
                        callback_replay.graph_build_generation &&
                    end.signature_identity ==
                        callback_replay.graph_signature_identity &&
                    end.signature_segment_key ==
                        callback_replay.profile_hash &&
                    end.position == expected_end,
                "callback producer yield REPLAY did not consume its exact "
                "sealed terminal-writer window");
            callback_replay.yield_open = false;
            ++callback_replay.next_yield;
            g_fmb_post_fn_yield_begin.reset();
            scope.active_ = false;
            return;
        }
        TORCH_CHECK(
            replay.armed && replay.yield_open &&
                replay.yields != nullptr &&
                replay.next_yield < replay.yields->yields_.size() &&
                replay.graph_identity == &graph,
            "post_fn producer yield REPLAY finish authority drifted");
        const GraphOpStreamStamp end = graph.op_stream_stamp();
        const auto& trace = replay.yields->source_trace_;
        const auto& expected =
            replay.yields->yields_[replay.next_yield];
        const size_t expected_begin =
            trace.graph_node_begin_ + expected.relative_node_begin;
        const size_t expected_end =
            trace.graph_node_begin_ + expected.relative_node_end;
        TORCH_CHECK(
            g_fmb_post_fn_yield_begin->state ==
                    RpuKernelGraph::State::REPLAYING &&
                g_fmb_post_fn_yield_begin->position == expected_begin &&
                end.graph == &graph &&
                end.build_generation == replay.graph_build_generation &&
                end.signature_identity == replay.graph_signature_identity &&
                end.signature_segment_key == replay.profile_hash &&
                end.position == expected_end,
            "post_fn producer yield REPLAY did not consume its exact sealed "
            "terminal-writer window");
        replay.yield_open = false;
        ++replay.next_yield;
        g_fmb_post_fn_yield_begin.reset();
        scope.active_ = false;
        return;
    }

    TORCH_CHECK(candidate.armed && candidate.profile != nullptr &&
                    candidate.callback_open &&
                    candidate.producer_yield_open &&
                    candidate.pending_producer_yield.has_value() &&
                    &graph == candidate.graph_identity &&
                    graph.state() == RpuKernelGraph::State::RECORDING,
                "post_fn producer yield BUILD finish authority drifted");
    TORCH_CHECK(
        (scope.contract_ ==
             SpmFmbPostFnYieldScope::Contract::PostFnExact &&
         candidate.profile->post_fn_yield_capability_ ==
             SpmFmbPostFnYieldCapability::
                 CanonicalDenseReplicatedFp16 &&
         candidate.callbacks.back().kind ==
             SpmFmbOccurrenceKind::PostFn) ||
            (scope.contract_ ==
                 SpmFmbPostFnYieldScope::Contract::CallbackWindow &&
             candidate.profile->layer_producer_yield_capability_ ==
                 SpmFmbLayerProducerYieldCapability::
                     CanonicalDenseReplicatedFp16 &&
             (candidate.callbacks.back().kind ==
                  SpmFmbOccurrenceKind::LayerBody ||
              candidate.callbacks.back().kind ==
                  SpmFmbOccurrenceKind::PostFn)),
        "producer yield BUILD scope/callback capability drifted");
    const GraphOpStreamStamp end = graph.op_stream_stamp();
    const GraphOpStreamWindowDigest digest =
        graph.digest_op_stream_window(*g_fmb_post_fn_yield_begin, end);
    const auto& marker_occurrences =
        digest.semantic_spm_producer_yields;
    const auto& pending = *candidate.pending_producer_yield;
    TORCH_CHECK(
        digest.node_count > 0 &&
            !digest.contains(GraphNodeKind::ChildGraph) &&
            !digest.contains(GraphNodeKind::HostCallback) &&
            !digest.contains(GraphNodeKind::Tier3Oneshot) &&
            digest.kind_mask ==
                graph_node_kind_bit(GraphNodeKind::Kernel) &&
            marker_occurrences.size() == 1,
        "post_fn producer yield requires a kernel-only Graph window with "
        "exactly one semantic terminal writer");
    const auto& marker = marker_occurrences.front();
    TORCH_CHECK(
            marker.relative_node_index + 1 == digest.node_count &&
            marker.semantic_yield_id == pending.semantic_id &&
            marker.writer_kind ==
                static_cast<uint8_t>(pending.writer_kind) &&
            digest.semantic_spm_producer_yield_id_digest != 0,
        "post_fn producer yield requires its matching terminal writer as "
        "the final Graph node");

    Impl::BuildTraceProducerYieldRecord completed = pending;
    completed.graph_node_end = digest.end;
    completed.graph_node_kind_mask = digest.kind_mask;
    completed.graph_topology_hash = digest.topology_hash;
    candidate.callbacks.back().producer_yields.push_back(
        std::move(completed));
    candidate.pending_producer_yield.reset();
    candidate.producer_yield_open = false;
    g_fmb_post_fn_yield_begin.reset();
    scope.active_ = false;
}

void FusedModelBase::cancel_spm_pipeline_post_fn_dense_yield(
    SpmFmbPostFnYieldScope& scope) noexcept {
    if (RpuKernelGraph::has_active()) {
        RpuKernelGraph::active()
            .poison_semantic_spm_producer_yield_for_fmb();
    }
    pimpl_->pipeline_build_trace_.pending_producer_yield.reset();
    pimpl_->pipeline_build_trace_.producer_yield_open = false;
    if (scope.contract_ ==
        SpmFmbPostFnYieldScope::Contract::CallbackWindow) {
        pimpl_->pipeline_callback_yield_replay_authority_ = {};
    } else {
        pimpl_->pipeline_post_fn_yield_replay_authority_ = {};
    }
    g_fmb_post_fn_yield_begin.reset();
    scope.active_ = false;
}

void FusedModelBase::adopt_spm_pipeline_component(
    const SpmPipelineLease& lease,
    const SpmTensorView& scratch) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "adopt_spm_pipeline_component must run outside Graph capture");
    lease.validate_physical(lease.epoch(), lease.plan_hash(),
                            /*require_sealed=*/false);
    TORCH_CHECK(pimpl_->valid_ && pimpl_->persistent_allocated_,
                "adopt_spm_pipeline_component requires a prepared component");
    TORCH_CHECK(pimpl_->pipeline_layout_hash_ != 0 &&
                    pimpl_->pipeline_temporary_bytes_ > 0,
                "adopt_spm_pipeline_component requires a prepared cached layout");
    TORCH_CHECK(pimpl_->pipeline_layout_hash_ == pipeline_layout_hash_impl(
                    pimpl_->last_decls_, pimpl_->cached_params_hash_,
                    pimpl_->pipeline_temporary_bytes_, pimpl_->offsets_,
                    pimpl_->per_layer_offsets_),
                "adopt_spm_pipeline_component: cached SPM offsets drifted "
                "after prepare");

    const uint32_t scratch_begin = scratch.resolve_physical_offset(lease);
    TORCH_CHECK(scratch.size_bytes() >= pimpl_->pipeline_temporary_bytes_,
                "adopt_spm_pipeline_component: scratch view has ",
                scratch.size_bytes(), " bytes but cached layout needs ",
                pimpl_->pipeline_temporary_bytes_);
    const uint64_t scratch_end = static_cast<uint64_t>(scratch_begin) +
                                 scratch.size_bytes();

    auto check_range = [&](uint32_t offset, const BufferDecl& decl) {
        const size_t bytes = align_spm_bytes(static_cast<size_t>(decl.size));
        const uint64_t physical_offset =
            static_cast<uint64_t>(scratch_begin) + offset;
        const uint64_t end = physical_offset + bytes;
        TORCH_CHECK(physical_offset >= scratch_begin && end <= scratch_end,
                    "adopt_spm_pipeline_component: cached buffer '", decl.name,
                    "' escapes scratch view");
        TORCH_CHECK(physical_offset <= std::numeric_limits<uint32_t>::max(),
                    "adopt_spm_pipeline_component: rebased buffer '",
                    decl.name, "' exceeds uint32 SPM offset range");
        lease.validate_physical_range(
            static_cast<uint32_t>(physical_offset), bytes);
    };
    for (const BufferDecl& decl : pimpl_->last_decls_) {
        if (decl.storage == StorageClass::Temp) {
            auto it = pimpl_->offsets_.find(decl.name);
            TORCH_CHECK(it != pimpl_->offsets_.end(),
                        "adopt_spm_pipeline_component: missing Temp buffer '",
                        decl.name, "'");
            check_range(it->second, decl);
        } else if (decl.storage == StorageClass::TempPerLayer) {
            for (int layer = 0; layer < decl.per_layer; ++layer) {
                TORCH_CHECK(layer <
                                static_cast<int>(pimpl_->per_layer_offsets_.size()),
                            "adopt_spm_pipeline_component: missing layer map");
                auto it = pimpl_->per_layer_offsets_[layer].find(decl.name);
                TORCH_CHECK(it != pimpl_->per_layer_offsets_[layer].end(),
                            "adopt_spm_pipeline_component: missing TempPerLayer '",
                            decl.name, "'");
                check_range(it->second, decl);
            }
        }
    }

    // Every framework Persistent* slot is allocated from super-persistent SPM.
    // A sibling component's prepare may have advanced persistent_generation via
    // reset_all(), but it cannot erase those slots.  Acknowledge that transition
    // only while no ordinary persistent range exists; callbacks remain dirty and
    // are captured normally by the first Z2 BUILD.
    TORCH_CHECK(SPM_ALLOC.persistent_used() == 0,
                "adopt_spm_pipeline_component cannot acknowledge ordinary "
                "persistent SPM after reset_all");
    if (pimpl_->cached_persistent_gen_ !=
        SPM_ALLOC.persistent_generation()) {
        pimpl_->preload_callbacks_dirty_ = true;
    }
    pimpl_->cached_persistent_gen_ = SPM_ALLOC.persistent_generation();
    pimpl_->alloc_gen_ = lease.allocator_generation();
    pimpl_->pipeline_lease_epoch_ = lease.epoch();
    pimpl_->pipeline_plan_hash_ = lease.plan_hash();
    pimpl_->pipeline_scratch_base_ = scratch_begin;
}

void FusedModelBase::adopt_spm_pipeline_composite_component(
    const SpmPipelineLease& lease,
    bool producer) {
    TORCH_CHECK(
        !RpuKernelGraph::has_active(),
        "adopt_spm_pipeline_composite_component must run outside Graph "
        "capture");
    lease.validate_composite_execution_authority(
        lease.epoch(), lease.plan_hash(), /*require_committed=*/false);
    const auto placement =
        lease.resolve_composite_component_placement(producer);
    TORCH_CHECK(
        static_cast<uint64_t>(lease.physical_arena_base()) +
                placement.relative_offset <=
            std::numeric_limits<uint32_t>::max(),
        "composite component scratch base exceeds uint32");
    const uint32_t expected_scratch_begin = static_cast<uint32_t>(
        lease.physical_arena_base() + placement.relative_offset);
    const bool fresh_component =
        pimpl_->pipeline_lease_epoch_ == 0 &&
        pimpl_->pipeline_plan_hash_ == 0 &&
        pimpl_->pipeline_scratch_base_ == 0;
    const bool same_component =
        pimpl_->pipeline_lease_epoch_ == lease.epoch() &&
        pimpl_->pipeline_plan_hash_ == lease.plan_hash() &&
        pimpl_->pipeline_scratch_base_ == expected_scratch_begin &&
        pimpl_->alloc_gen_ == lease.allocator_generation();
    TORCH_CHECK(
        pimpl_->valid_ && pimpl_->persistent_allocated_ &&
            (fresh_component || same_component),
        "composite component adoption requires one fresh or same-lease "
        "prepared owner");
    const CachedLayoutSnapshot& snapshot =
        pimpl_->pipeline_manifest_snapshot_;
    TORCH_CHECK(
        snapshot.valid && !snapshot.cpu_dry &&
            placement.layout_hash == pimpl_->pipeline_layout_hash_ &&
            placement.layout_hash == snapshot.layout_hash &&
            placement.allocation_hash == snapshot.allocation_hash &&
            placement.bytes == pimpl_->pipeline_temporary_bytes_ &&
            pimpl_->pipeline_layout_hash_ == pipeline_layout_hash_impl(
                pimpl_->last_decls_, pimpl_->cached_params_hash_,
                pimpl_->pipeline_temporary_bytes_, pimpl_->offsets_,
                pimpl_->per_layer_offsets_),
        "composite component adoption allocation identity drifted after "
        "reservation planning");
    const uint64_t scratch_begin64 =
        static_cast<uint64_t>(lease.physical_arena_base()) +
        placement.relative_offset;
    TORCH_INTERNAL_ASSERT(
        scratch_begin64 <= std::numeric_limits<uint32_t>::max());
    const uint32_t scratch_begin =
        static_cast<uint32_t>(scratch_begin64);
    TORCH_INTERNAL_ASSERT(scratch_begin == expected_scratch_begin);
    const uint64_t scratch_end = scratch_begin64 + placement.bytes;
    TORCH_CHECK(
        scratch_end >= scratch_begin64 &&
            scratch_end <= lease.physical_arena_end(),
        "composite component scratch range escapes the reservation arena");

    auto check_range = [&](uint32_t offset, const BufferDecl& decl) {
        const size_t bytes = align_spm_bytes(static_cast<size_t>(decl.size));
        const uint64_t physical_offset = scratch_begin64 + offset;
        const uint64_t end = physical_offset + bytes;
        TORCH_CHECK(
            end >= physical_offset && physical_offset >= scratch_begin64 &&
                end <= scratch_end &&
                physical_offset <= std::numeric_limits<uint32_t>::max(),
            "composite component cached buffer '", decl.name,
            "' escapes its reserved component range");
        lease.validate_physical_range(
            static_cast<uint32_t>(physical_offset), bytes);
    };
    for (const BufferDecl& decl : pimpl_->last_decls_) {
        if (decl.storage == StorageClass::Temp) {
            auto it = pimpl_->offsets_.find(decl.name);
            TORCH_CHECK(
                it != pimpl_->offsets_.end(),
                "composite component is missing Temp buffer '", decl.name,
                "'");
            check_range(it->second, decl);
        } else if (decl.storage == StorageClass::TempPerLayer) {
            for (int layer = 0; layer < decl.per_layer; ++layer) {
                TORCH_CHECK(
                    layer < static_cast<int>(
                                pimpl_->per_layer_offsets_.size()),
                    "composite component is missing a TempPerLayer map");
                auto it = pimpl_->per_layer_offsets_[layer].find(decl.name);
                TORCH_CHECK(
                    it != pimpl_->per_layer_offsets_[layer].end(),
                    "composite component is missing TempPerLayer buffer '",
                    decl.name, "'");
                check_range(it->second, decl);
            }
        }
    }

    TORCH_CHECK(
        SPM_ALLOC.persistent_used() == 0,
        "composite component cannot acknowledge ordinary persistent SPM "
        "after reset_all");
    if (pimpl_->cached_persistent_gen_ !=
        SPM_ALLOC.persistent_generation()) {
        pimpl_->preload_callbacks_dirty_ = true;
    }
    pimpl_->pipeline_manifest_snapshot_.persistent_generation =
        SPM_ALLOC.persistent_generation();
    pimpl_->cached_persistent_gen_ = SPM_ALLOC.persistent_generation();
    pimpl_->alloc_gen_ = lease.allocator_generation();
    pimpl_->pipeline_lease_epoch_ = lease.epoch();
    pimpl_->pipeline_plan_hash_ = lease.plan_hash();
    pimpl_->pipeline_scratch_base_ = scratch_begin;
}

void FusedModelBase::validate_spm_pipeline_composite_component(
    const SpmPipelineLease& lease,
    bool producer) const {
    lease.validate_composite_execution_authority(
        lease.epoch(), lease.plan_hash(), lease.composite_committed());
    const auto placement =
        lease.resolve_composite_component_placement(producer);
    TORCH_CHECK(
        pimpl_->valid_ && pimpl_->persistent_allocated_ &&
            pimpl_->pipeline_lease_epoch_ == lease.epoch() &&
            pimpl_->pipeline_plan_hash_ == lease.plan_hash() &&
            pimpl_->alloc_gen_ == lease.allocator_generation() &&
            pimpl_->cached_persistent_gen_ ==
                SPM_ALLOC.persistent_generation() &&
            pimpl_->pipeline_manifest_snapshot_.valid &&
            !pimpl_->pipeline_manifest_snapshot_.cpu_dry &&
            placement.layout_hash == pimpl_->pipeline_layout_hash_ &&
            placement.layout_hash ==
                pimpl_->pipeline_manifest_snapshot_.layout_hash &&
            placement.allocation_hash ==
                pimpl_->pipeline_manifest_snapshot_.allocation_hash &&
            placement.bytes == pimpl_->pipeline_temporary_bytes_ &&
            pimpl_->pipeline_layout_hash_ == pipeline_layout_hash_impl(
                pimpl_->last_decls_, pimpl_->cached_params_hash_,
                pimpl_->pipeline_temporary_bytes_, pimpl_->offsets_,
                pimpl_->per_layer_offsets_),
        "composite component validation observed stale owner or allocation "
        "identity");
    TORCH_CHECK(
        static_cast<uint64_t>(lease.physical_arena_base()) +
                placement.relative_offset <=
            std::numeric_limits<uint32_t>::max(),
        "composite component validation scratch base exceeds uint32");
    const uint32_t scratch_begin = static_cast<uint32_t>(
        lease.physical_arena_base() + placement.relative_offset);
    TORCH_CHECK(
        pimpl_->pipeline_scratch_base_ == scratch_begin,
        "composite component validation scratch placement drifted");
    lease.validate_physical_range(scratch_begin, placement.bytes);
}

void FusedModelBase::validate_spm_pipeline_component(
    const SpmPipelineLease& lease) const {
    lease.validate_physical(lease.epoch(), lease.plan_hash(),
                            /*require_sealed=*/true);
    TORCH_CHECK(pimpl_->pipeline_lease_epoch_ == lease.epoch() &&
                    pimpl_->pipeline_plan_hash_ == lease.plan_hash(),
                "validate_spm_pipeline_component: component was not adopted "
                "by the active physical lease");
    TORCH_CHECK(pimpl_->alloc_gen_ == lease.allocator_generation() &&
                    pimpl_->cached_persistent_gen_ ==
                        SPM_ALLOC.persistent_generation(),
                "validate_spm_pipeline_component: allocator generation drift");
    TORCH_CHECK(pimpl_->pipeline_layout_hash_ == pipeline_layout_hash_impl(
                    pimpl_->last_decls_, pimpl_->cached_params_hash_,
                    pimpl_->pipeline_temporary_bytes_, pimpl_->offsets_,
                    pimpl_->per_layer_offsets_),
                "validate_spm_pipeline_component: cached layout hash drift");
    lease.validate_physical_range(
        pimpl_->pipeline_scratch_base_,
        pimpl_->pipeline_temporary_bytes_);
}

void FusedModelBase::release_spm_pipeline_component(
    uint64_t epoch,
    uint64_t plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "release_spm_pipeline_component must run outside Graph capture");
    TORCH_CHECK(pimpl_->pipeline_lease_epoch_ == epoch &&
                    pimpl_->pipeline_plan_hash_ == plan_hash,
                "release_spm_pipeline_component: stale epoch/plan hash");
    pimpl_->pipeline_lease_epoch_ = 0;
    pimpl_->pipeline_plan_hash_ = 0;
    pimpl_->pipeline_scratch_base_ = 0;
}

void FusedModelBase::stage_spm_outer_fast_component(
    ::GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches,
    std::vector<
        std::pair<std::shared_ptr<const uint64_t>, uint64_t>>
        extra_owner_stamps) const {
    TORCH_CHECK(pimpl_->pipeline_lease_epoch_ != 0 &&
                    pimpl_->pipeline_plan_hash_ != 0 &&
                    pimpl_->pipeline_manifest_snapshot_.valid &&
                    pimpl_->pipeline_manifest_generation_ != nullptr &&
                    pimpl_->outer_fast_model_generation_ != nullptr &&
                    pimpl_->model_state_gen_ ==
                        *pimpl_->outer_fast_model_generation_,
                "schema-5 outer-fast component requires one current physical "
                "layout and synchronized model/manifest generations");
    TORCH_CHECK(static_cast<int64_t>(k_caches.size()) == pimpl_->num_layers_ &&
                    static_cast<int64_t>(v_caches.size()) ==
                        pimpl_->num_layers_,
                "schema-5 outer-fast component KV-cache count mismatch");

    auto validate_cache = [](const at::Tensor& tensor, const char* kind,
                             size_t ordinal) {
        TORCH_CHECK(tensor.defined() &&
                        tensor.device().type() == at::kPrivateUse1 &&
                        tensor.scalar_type() == at::kHalf &&
                        tensor.layout() == c10::Layout::Strided &&
                        tensor.is_contiguous(),
                    "schema-5 outer-fast ", kind, " cache[", ordinal,
                    "] must be contiguous FP16 RPU storage");
    };
    std::vector<at::Tensor> keys;
    std::vector<at::Tensor> values;
    keys.reserve(k_caches.size());
    values.reserve(v_caches.size());
    for (size_t i = 0; i < k_caches.size(); ++i) {
        validate_cache(k_caches[i], "key", i);
        validate_cache(v_caches[i], "value", i);
        keys.push_back(k_caches[i]);
        values.push_back(v_caches[i]);
    }

    std::vector<std::pair<std::shared_ptr<const uint64_t>, uint64_t>>
        owner_stamps;
    owner_stamps.reserve(2 + extra_owner_stamps.size());
    owner_stamps.emplace_back(
        pimpl_->outer_fast_model_generation_,
        *pimpl_->outer_fast_model_generation_);
    owner_stamps.emplace_back(
        pimpl_->pipeline_manifest_generation_,
        *pimpl_->pipeline_manifest_generation_);
    for (auto& stamp : extra_owner_stamps) {
        TORCH_CHECK(stamp.first != nullptr &&
                        *stamp.first == stamp.second,
                    "schema-5 outer-fast extra owner stamp is stale");
        owner_stamps.push_back(std::move(stamp));
    }
    guard.stage_outer_fast_component(
        std::move(owner_stamps), std::move(keys), std::move(values));
}

void FusedModelBase::bind_spm_outer_fast_input(
    ::GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& hidden_states) {
    TORCH_CHECK(hidden_states.defined() && hidden_states.dim() == 3 &&
                    hidden_states.size(0) == 1 &&
                    hidden_states.size(2) == pimpl_->hidden_size_ &&
                    hidden_states.scalar_type() == at::kHalf &&
                    hidden_states.device().type() == at::kPrivateUse1 &&
                    hidden_states.layout() == c10::Layout::Strided &&
                    hidden_states.is_contiguous(),
                "schema-5 outer-fast input must be [1,S,H] contiguous FP16 "
                "RPU storage");
    guard.bind_fast_mutable_dma(
        current_hidden_in_src_base(), hidden_states,
        GraphOuterFastDmaSide::Source,
        /*byte_begin=*/0, hidden_states.nbytes());
}

void FusedModelBase::bind_spm_outer_fast_composite_input(
    ::GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& hidden_states,
    size_t producer_ordinal) {
    TORCH_CHECK(
        pimpl_->pipeline_lease_epoch_ != 0 &&
            pimpl_->pipeline_plan_hash_ != 0 &&
            pimpl_->pipeline_manifest_snapshot_.valid &&
            pimpl_->pipeline_manifest_generation_ != nullptr &&
            pimpl_->outer_fast_model_generation_ != nullptr &&
            pimpl_->model_state_gen_ ==
                *pimpl_->outer_fast_model_generation_ &&
            spm_fmb_composite_occurrence_capability() ==
                SpmFmbCompositeOccurrenceCapability::
                    StableThreeInputSlotsSameOwnerPreload &&
            producer_ordinal <
                pimpl_->composite_hidden_in_src_bases_.size(),
        "schema-5 multiview outer-fast input requires the exact current "
        "three-slot composite physical authority");
    TORCH_CHECK(
        hidden_states.defined() && hidden_states.dim() == 3 &&
            hidden_states.size(0) == 1 &&
            hidden_states.size(2) == pimpl_->hidden_size_ &&
            hidden_states.scalar_type() == at::kHalf &&
            hidden_states.device().type() == at::kPrivateUse1 &&
            hidden_states.layout() == c10::Layout::Strided &&
            hidden_states.is_contiguous(),
        "schema-5 multiview outer-fast input must be [1,S,H] contiguous "
        "FP16 RPU storage");
    uint64_t& slot =
        pimpl_->composite_hidden_in_src_bases_[producer_ordinal];
    TORCH_CHECK(slot != 0,
                "schema-5 multiview outer-fast input slot was never "
                "published by the retained BUILD");
    guard.bind_fast_mutable_dma(
        slot, hidden_states, GraphOuterFastDmaSide::Source,
        /*byte_begin=*/0, hidden_states.nbytes());
}

// =============================================================================
// run_preload_callbacks_ — persistent callback dispatch
// =============================================================================

void FusedModelBase::run_preload_callbacks_(
    const std::vector<BufferDecl>& decls, FusedModelBase& self)
{
    // graph-scope override: inside RECORDING/REPLAYING the callbacks MUST run
    // every forward — RECORDING captures preload DMA nodes into the graph;
    // REPLAYING advances the cursor through those same nodes via the
    // record_dma_* validators. Honoring the dirty/gen gate inside REPLAYING
    // skips the cursor advance, so the next op-emitted DMA lands on a preload
    // node and trips "DMA variant drift" (Fixed vs MutableSrc) at cursor N.
    const bool inside_graph = graph_dma::active();

    // Outside graph scope (PASSTHROUGH/BUILT), return early only when
    // persistent-gen UNCHANGED AND dirty flag clear.
    if (!inside_graph
        && pimpl_->cached_preload_gen_ == SPM_ALLOC.persistent_generation()
        && !pimpl_->preload_callbacks_dirty_) {
        return;
    }

    // graph-naive: callback body 直接 enqueu_kernel(走 RpuQueue proxy)。
    // 无 graph scope 时 wq->enqueu_kernel fallback PASSTHROUGH 立即发射;
    // Python 端用 with cache.capture(WEIGHTS_SIG) 包整个
    // run_preload_callbacks_ 调用,callback 自动进 graph RECORDING/REPLAYING。
    auto invoke_preload_callback = [this, &self](
        const BufferDecl& d, int layer, uint32_t core0_addr) {
        auto& candidate = pimpl_->pipeline_build_trace_;
        const bool trace_scope =
            candidate.armed || g_fmb_build_trace_owner != nullptr;
        if (trace_scope) {
            TORCH_CHECK(RpuKernelGraph::has_active(),
                        "schema-v8 BUILD trace STOP: persistent preload "
                        "callback lost its active Graph");
            const GraphOpStreamStamp stamp =
                RpuKernelGraph::active().op_stream_stamp();
            TORCH_CHECK(
                candidate.armed &&
                    g_fmb_build_trace_owner == pimpl_.get() &&
                    candidate.profile != nullptr &&
                    candidate.profile->preload_capability_ ==
                        SpmFmbPreloadCapability::
                            PersistentOutsideResolvedWindow &&
                    candidate.next_step == 0 &&
                    candidate.callbacks.empty() &&
                    !candidate.persistent_preload_callback_open &&
                    !candidate.callback_open &&
                    stamp.graph == candidate.graph_identity &&
                    stamp.state == RpuKernelGraph::State::RECORDING &&
                    stamp.build_generation ==
                        candidate.graph_build_generation &&
                    stamp.signature_identity ==
                        candidate.graph_signature_identity &&
                    stamp.signature_segment_key ==
                        candidate.graph_signature_segment_key,
                "schema-v8 BUILD trace STOP: persistent preload callback "
                "has no lexical prelude authority");
            const auto& snapshot = pimpl_->pipeline_manifest_snapshot_;
            uint32_t declaration_ordinal = 0;
            const auto* sealed = find_trace_declaration(
                snapshot, d.name, &declaration_ordinal);
            TORCH_CHECK(
                sealed != nullptr && sealed->preload_present &&
                    sealed->storage == d.storage &&
                    sealed->per_layer == d.per_layer &&
                    (d.storage == StorageClass::Persistent ||
                     (d.storage == StorageClass::PersistentPerLayer &&
                      d.per_layer > 0 && layer >= 0 &&
                      layer < d.per_layer)) &&
                    (d.storage != StorageClass::Persistent ||
                     (d.per_layer == 0 && layer == -1)),
                "schema-v8 BUILD trace STOP: persistent preload callback "
                "declaration drift");
            candidate.persistent_preload_callback_open = true;
        }
        auto close_trace_scope = c10::make_scope_exit([&] {
            if (trace_scope) {
                candidate.persistent_preload_callback_open = false;
            }
        });
        d.preload_callback(self, layer, core0_addr);
    };

    for (const auto& d : decls) {
        if (!d.preload_callback) continue;
        if (d.storage == StorageClass::Persistent) {
            auto it = pimpl_->persistent_offsets_.find(d.name);
            if (it == pimpl_->persistent_offsets_.end()) continue;
            uint32_t core0_addr = SPM_ALLOC.addr(0, it->second);
            invoke_preload_callback(
                d, /*layer_idx=*/-1, core0_addr);
        } else if (d.storage == StorageClass::PersistentPerLayer) {
            for (int L = 0; L < d.per_layer; ++L) {
                if ((int)pimpl_->persistent_per_layer_offsets_.size() <= L) break;
                auto it = pimpl_->persistent_per_layer_offsets_[L].find(d.name);
                if (it == pimpl_->persistent_per_layer_offsets_[L].end()) continue;
                uint32_t core0_addr = SPM_ALLOC.addr(0, it->second);
                invoke_preload_callback(d, /*layer_idx=*/L, core0_addr);
            }
        }
    }

    // Clear both bookkeeping fields only after successful dispatch.
    pimpl_->cached_preload_gen_       = SPM_ALLOC.persistent_generation();
    pimpl_->preload_callbacks_dirty_  = false;
}

// =============================================================================
// drive_preload_for_test — diagnostic preload helper
//
// Runs ensure_allocated + run_preload_callbacks_ once against a minimal
// LayoutContext. Does NOT require hidden_states / kv caches / position.
// Used by the diagnostic model to exercise the preload-callback path.
// =============================================================================

void FusedModelBase::drive_preload_for_test() {
    LayoutContext alloc_ctx;
    // Respect chunk_size_override_ so trigger_relayout_for_test(NEW_SIZE) can
    // change the LayoutContext hash → ensure_allocated Path 1 → declare_buffers
    // re-invoked → fresh lambda closure registered.
    alloc_ctx.chunk_size       = (pimpl_->chunk_size_override_ > 0)
                                   ? pimpl_->chunk_size_override_
                                   : 16;
    alloc_ctx.max_kv_seq_len   = 0;
    alloc_ctx.num_layers       = 1;
    alloc_ctx.use_attn_mask    = false;
    // is_causal is a hashed layout field. This smoke harness only
    // exercises the preload-callback path (never KV_FIRST), so it is always causal;
    // set it explicitly rather than relying on the struct default to keep the
    // allocation identity unambiguous here.
    alloc_ctx.is_causal        = true;

    auto decl_fn = [this](const LayoutContext& c) { return this->declare_buffers(c); };
    auto estimate_fn = [&](const LayoutContext& c) {
        (void)c;
        return (int64_t)0;
    };

    auto decls = decl_fn(alloc_ctx);
    ensure_allocated_impl(*pimpl_, decls, alloc_ctx, estimate_fn,
                          subclass_layout_hash(),
                          /*enforce_allocation_identity=*/false);
    // Cache last_decls_ so preload callbacks see the most-recent lambda closures
    // so callbacks remain bound to the current declarations.
    pimpl_->last_decls_ = std::move(decls);

    run_preload_callbacks_(pimpl_->last_decls_, *this);
}

void FusedModelBase::drive_preload_callbacks_for_test() {
    TORCH_CHECK(pimpl_->valid_ && !pimpl_->last_decls_.empty(),
                "drive_preload_callbacks_for_test requires a prepared layout");
    run_preload_callbacks_(pimpl_->last_decls_, *this);
}

// =============================================================================
// DMA helpers (from v2 FusedModelBase)
// =============================================================================

// Batch decode: hidden_states is [B, S, H] contiguous, and batch > 1 is only
// admitted at S == 1 (run_all_layers), so the B rows of a chunk are exactly
// B*H contiguous elements at chunk.offset == 0 — the same batch-major row order
// the SPM buffers use. batch_size == 1 leaves both helpers identical to before.
int64_t FusedModelBase::chunk_rows_(const ChunkInfo& chunk) const {
    return pimpl_->ctx_.batch_size * chunk.len;
}
int64_t FusedModelBase::chunk_row_offset_(const ChunkInfo& chunk) const {
    return pimpl_->ctx_.batch_size * chunk.offset;
}

void FusedModelBase::emit_layer_input_dma(int layer_idx, const ChunkInfo& chunk,
                                          const char* dst_buf,
                                          int64_t dst_row_offset)
{
    TORCH_CHECK(
        spm_fmb_runtime_member_capability() !=
            SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs,
        "schema-v12 canonical DMA STOP: legacy layer DMA helper is "
        "forbidden");
    TORCH_CHECK(dst_row_offset >= 0,
                "emit_layer_input_dma: dst_row_offset must be non-negative");
    const uint32_t dst = addr(0, dst_buf) +
        dst_row_offset * pimpl_->hidden_size_ * sizeof(c10::Half);
    if (layer_idx == 0) {
        // Layer 0 reads caller-owned `hidden_states` (data_ptr fresh per
        // forward). Use the mutable-DMA wrapper: the framework patches each
        // chunk DMA's src to `*hidden_in_src_base_ + offset` at
        // REPLAY end() via Queue_t::update_dma_kernel.
        const int64_t chunk_offset_bytes =
            chunk_row_offset_(chunk) * pimpl_->hidden_size_
            * static_cast<int64_t>(sizeof(c10::Half));
        TORCH_INTERNAL_ASSERT(pimpl_->ctx_.hidden_states != nullptr);
        rpu_launch_ddr_broadcast_spm_dma_mutable(
            current_hidden_in_src_base(), *pimpl_->ctx_.hidden_states,
            chunk_offset_bytes,
            chunk_rows_(chunk) * pimpl_->hidden_size_,
            dst,
            /*num_cores=*/8);
        return;
    }
    // Inner layers read from the ping-pong chain buffers populated by the
    // previous layer's output DMA — pointer is registry-stable, regular DMA.
    const at::Tensor& input_tensor =
        (layer_idx % 2 == 0) ? pimpl_->ddr_bufA_ : pimpl_->ddr_bufB_;
    rpu_launch_ddr_broadcast_spm_dma(
        input_tensor, chunk_row_offset_(chunk) * pimpl_->hidden_size_,
        chunk_rows_(chunk) * pimpl_->hidden_size_,
        dst,
        /*num_cores=*/8);
}

void FusedModelBase::emit_layer_input_row_run_dma(
    int layer_idx,
    const ChunkInfo& chunk,
    int64_t global_row_begin,
    int64_t row_count,
    const char* dst_buf) {
    emit_layer_input_row_run_dma(
        current_hidden_in_src_base(), layer_idx, chunk, global_row_begin,
        row_count, dst_buf);
}

void FusedModelBase::emit_layer_input_row_run_dma(
    uint64_t& live_src_base,
    int layer_idx,
    const ChunkInfo& chunk,
    int64_t global_row_begin,
    int64_t row_count,
    const char* dst_buf) {
    TORCH_CHECK(
        layer_idx == 0 && pimpl_->ctx_.hidden_states != nullptr,
        "layer-input row-run DMA is valid only for the caller-owned layer-0 "
        "input");
    TORCH_CHECK(
        global_row_begin >= chunk.offset && row_count > 0 &&
            global_row_begin <= chunk.offset + chunk.len &&
            row_count <= chunk.offset + chunk.len - global_row_begin,
        "layer-input row-run [", global_row_begin, ", ",
        global_row_begin + row_count, ") escapes chunk [", chunk.offset,
        ", ", chunk.offset + chunk.len, ")");
    TORCH_CHECK(
        global_row_begin <=
                std::numeric_limits<int64_t>::max() / pimpl_->hidden_size_ &&
            row_count <=
                std::numeric_limits<int64_t>::max() / pimpl_->hidden_size_,
        "layer-input row-run byte geometry overflows int64");

    const int64_t row_bytes =
        pimpl_->hidden_size_ * static_cast<int64_t>(sizeof(c10::Half));
    TORCH_CHECK(
        global_row_begin <= std::numeric_limits<int64_t>::max() / row_bytes,
        "layer-input row-run source byte offset overflows int64");
    const int64_t local_row_begin = global_row_begin - chunk.offset;
    TORCH_CHECK(
        local_row_begin <= std::numeric_limits<uint32_t>::max() / row_bytes,
        "layer-input row-run destination byte offset overflows uint32");
    const uint32_t dst_byte_offset =
        static_cast<uint32_t>(local_row_begin * row_bytes);
    const uint32_t dst_base = addr(0, dst_buf);
    TORCH_CHECK(
        dst_base <= std::numeric_limits<uint32_t>::max() - dst_byte_offset,
        "layer-input row-run destination address overflows uint32");
    TORCH_CHECK(current_hidden_in_src_base() != 0,
                "layer-input row-run source resolved a zero RPU address");
    live_src_base = current_hidden_in_src_base();

    rpu_launch_ddr_broadcast_spm_dma_mutable(
        live_src_base, *pimpl_->ctx_.hidden_states,
        /*src_offset_bytes=*/global_row_begin * row_bytes,
        /*num_elements=*/row_count * pimpl_->hidden_size_,
        dst_base + dst_byte_offset,
        /*num_cores=*/8);
}

void FusedModelBase::
require_spm_pipeline_composite_consumer_physical_authority() const {
    const auto& runtime = pimpl_->composite_physical_execution_;
    const bool graph_current =
        RpuKernelGraph::has_active() &&
        &RpuKernelGraph::active() == runtime.graph_identity;
    const GraphOpStreamStamp stamp = graph_current
        ? RpuKernelGraph::active().op_stream_stamp()
        : GraphOpStreamStamp{};
    TORCH_CHECK(
        runtime.armed && !runtime.producer && runtime.lease != nullptr &&
            runtime.composite_identity != 0 &&
            runtime.producer_local_ordinal == 0 &&
            runtime.next_target <= runtime.expected_target_count &&
            runtime.expected_target_count ==
                runtime.lease->composite_run_add_count() &&
            graph_current &&
            static_cast<uint8_t>(stamp.state) == runtime.graph_state &&
            stamp.build_generation == runtime.graph_build_generation &&
            stamp.signature_identity ==
                runtime.graph_signature_identity &&
            stamp.signature_segment_key ==
                runtime.graph_signature_segment_key,
        "composite consumer has no current lexical physical authority");
    runtime.lease->validate_composite_execution_authority(
        runtime.lease_epoch, runtime.plan_hash,
        runtime.require_committed);
}

void FusedModelBase::emit_spm_pipeline_composite_run_add(
    int layer_idx,
    const ChunkInfo& chunk,
    int64_t global_row_begin,
    int64_t row_count,
    const char* dst_buf) {
    require_spm_pipeline_composite_consumer_physical_authority();
    auto& runtime = pimpl_->composite_physical_execution_;
    TORCH_CHECK(
        dst_buf != nullptr && *dst_buf != '\0' &&
            global_row_begin >= chunk.offset && row_count > 0 &&
            global_row_begin <= chunk.offset + chunk.len &&
            row_count <= chunk.offset + chunk.len - global_row_begin &&
            runtime.next_target < runtime.expected_target_count,
        "composite RunAdd residual row run escapes the current chunk or "
        "sealed cursor");

    const size_t producer_count =
        runtime.lease->composite_producer_occurrence_count();
    const size_t yields_per_producer =
        runtime.lease->composite_yields_per_producer();
    TORCH_INTERNAL_ASSERT(producer_count > 0 && yields_per_producer > 1);
    const size_t producer_local_ordinal =
        runtime.next_target % producer_count;
    const size_t producer_yield_ordinal =
        runtime.next_target / producer_count;
    TORCH_CHECK(
        producer_yield_ordinal + 1 < yields_per_producer,
        "composite RunAdd cursor reached the direct final yield");
    const auto target = runtime.lease->resolve_composite_run_add_addrs(
        /*core=*/0, producer_local_ordinal, producer_yield_ordinal);

    const auto& snapshot = pimpl_->pipeline_manifest_snapshot_;
    uint32_t allocation_ordinal = 0;
    const auto* allocation = find_trace_allocation(
        snapshot, dst_buf, /*layer=*/-1, &allocation_ordinal);
    uint32_t declaration_ordinal = 0;
    const auto* declaration = find_trace_declaration(
        snapshot, dst_buf, &declaration_ordinal);
    TORCH_CHECK(
        snapshot.valid && !snapshot.cpu_dry && allocation != nullptr &&
            declaration != nullptr &&
            allocation->storage == StorageClass::Temp &&
            declaration->storage == StorageClass::Temp &&
            allocation->layer == -1 &&
            (declaration->scope == BufferScope::LayerWide ||
             declaration->scope == BufferScope::Compute) &&
            !declaration->reverse_layer_alloc &&
            !allocation->reverse_layer_alloc &&
            !declaration->preload_present &&
            !allocation->preload_present,
        "composite RunAdd destination is not one owned flat residual "
        "allocation");

    TORCH_CHECK(
        pimpl_->hidden_size_ > 0 &&
            pimpl_->hidden_size_ <=
                std::numeric_limits<int64_t>::max() /
                    static_cast<int64_t>(sizeof(c10::Half)),
        "composite RunAdd row width overflows int64");
    const int64_t row_bytes =
        pimpl_->hidden_size_ * static_cast<int64_t>(sizeof(c10::Half));
    TORCH_CHECK(
        chunk.len <= std::numeric_limits<int64_t>::max() / row_bytes &&
            row_count <= std::numeric_limits<int64_t>::max() / row_bytes &&
            allocation->logical_bytes ==
                static_cast<size_t>(chunk.len * row_bytes) &&
            allocation->aligned_bytes == allocation->logical_bytes &&
            target.consumer_layer == layer_idx &&
            target.consumer_row_begin == global_row_begin &&
            target.consumer_row_count == row_count &&
            target.spec.dtype == SpmPortDType::Fp16 &&
            target.spec.distribution ==
                SpmPortDistribution::Replicated &&
            target.spec.rows == chunk.len &&
            target.spec.cols == pimpl_->hidden_size_ &&
            target.bytes == static_cast<size_t>(row_count * row_bytes),
        "composite RunAdd destination layer or dense row geometry drifted "
        "from the sealed endpoint");

    const int64_t local_row_begin = global_row_begin - chunk.offset;
    TORCH_CHECK(
        local_row_begin <=
            std::numeric_limits<uint32_t>::max() / row_bytes,
        "composite RunAdd destination byte offset overflows uint32");
    const uint32_t dst_offset =
        static_cast<uint32_t>(local_row_begin * row_bytes);
    const uint32_t residual_base = addr(/*core=*/0, dst_buf);
    TORCH_CHECK(
        residual_base <=
            std::numeric_limits<uint32_t>::max() - dst_offset &&
            target.consumer_addr == residual_base + dst_offset &&
            target.payload_addr != 0 &&
            target.payload_addr != target.consumer_addr,
        "composite RunAdd resolved addresses do not match the owned residual "
        "row slice");
    TORCH_CHECK(
        row_count <=
            std::numeric_limits<int64_t>::max() / pimpl_->hidden_size_,
        "composite RunAdd element count overflows int64");
    rpu_launch_eltwise_binary_spm_kernel(
        target.consumer_addr, target.payload_addr, target.consumer_addr,
        row_count * pimpl_->hidden_size_, ValuOpType::ADD,
        c10::Half(1.0f), /*num_cores=*/8);
    ++runtime.next_target;
}

void FusedModelBase::emit_layer_output_dma(int layer_idx, const ChunkInfo& chunk,
                                           const char* src_buf,
                                           uint32_t src_offset_bytes)
{
    TORCH_CHECK(
        spm_fmb_runtime_member_capability() !=
            SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs,
        "schema-v12 canonical DMA STOP: legacy layer DMA helper is "
        "forbidden");
    const bool is_last_layer = (layer_idx == pimpl_->num_layers_ - 1);
    // Both branches are registry-stable: output_tensor_ is per-shape stable
    // (kept alive by pimpl_->registry_); the ping-pong bufs are equivalently
    // owned. Both are safe to DMA-bake.
    const at::Tensor& out_tensor = is_last_layer
        ? pimpl_->output_tensor_
        : ((layer_idx % 2 == 0) ? pimpl_->ddr_bufB_
                                : pimpl_->ddr_bufA_);
    const uint32_t src_addr = addr(0, src_buf);
    TORCH_CHECK(
        src_addr <= std::numeric_limits<uint32_t>::max() - src_offset_bytes,
        "emit_layer_output_dma source offset overflows uint32");
    rpu_launch_spm_copy_ddr_dma(
        src_addr + src_offset_bytes, out_tensor,
        chunk_row_offset_(chunk) * pimpl_->hidden_size_,
        chunk_rows_(chunk) * pimpl_->hidden_size_);
}

void FusedModelBase::emit_layer_chain_scratch_store(
    int layer_idx, const ChunkInfo& chunk, const char* src_buf)
{
    TORCH_CHECK(
        spm_fmb_runtime_member_capability() !=
            SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs,
        "schema-v12 canonical DMA STOP: legacy layer DMA helper is "
        "forbidden");
    TORCH_CHECK(pimpl_->ddr_bufA_ptr_ != nullptr && pimpl_->ddr_bufB_ptr_ != nullptr,
                "emit_layer_chain_scratch_store requires DDR ping-pong buffers");
    const at::Tensor& out_tensor = (layer_idx % 2 == 0)
        ? pimpl_->ddr_bufB_ : pimpl_->ddr_bufA_;
    rpu_launch_spm_copy_ddr_dma(
        addr(0, src_buf), out_tensor,
        chunk_row_offset_(chunk) * pimpl_->hidden_size_,
        chunk_rows_(chunk) * pimpl_->hidden_size_);
}

const c10::Half* FusedModelBase::stable_layer_input_ddr_ptr(
    int layer_idx, const ChunkInfo& chunk) const
{
    TORCH_CHECK(
        spm_fmb_runtime_member_capability() !=
            SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs,
        "schema-v12 canonical DMA STOP: legacy DDR pointer helper is "
        "forbidden");
    TORCH_CHECK(layer_idx > 0 && layer_idx < pimpl_->num_layers_,
                "stable_layer_input_ddr_ptr requires an inner layer, got ",
                layer_idx);
    TORCH_CHECK(pimpl_->ddr_bufA_ptr_ != nullptr && pimpl_->ddr_bufB_ptr_ != nullptr,
                "stable_layer_input_ddr_ptr requires DDR ping-pong buffers");
    const c10::Half* input_base = (layer_idx % 2 == 0)
        ? pimpl_->ddr_bufA_ptr_ : pimpl_->ddr_bufB_ptr_;
    return input_base + chunk.offset * pimpl_->hidden_size_;
}

const c10::Half* FusedModelBase::layer_chain_scratch_ddr_ptr(
    int layer_idx, const ChunkInfo& chunk) const
{
    TORCH_CHECK(
        spm_fmb_runtime_member_capability() !=
            SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs,
        "schema-v12 canonical DMA STOP: legacy DDR pointer helper is "
        "forbidden");
    TORCH_CHECK(layer_idx >= 0 && layer_idx < pimpl_->num_layers_,
                "layer_chain_scratch_ddr_ptr layer out of range: ", layer_idx);
    TORCH_CHECK(pimpl_->ddr_bufA_ptr_ != nullptr && pimpl_->ddr_bufB_ptr_ != nullptr,
                "layer_chain_scratch_ddr_ptr requires DDR ping-pong buffers");
    const c10::Half* output_base = (layer_idx % 2 == 0)
        ? pimpl_->ddr_bufB_ptr_ : pimpl_->ddr_bufA_ptr_;
    return output_base + chunk.offset * pimpl_->hidden_size_;
}

void FusedModelBase::begin_spm_pipeline_runtime_member_callback_for_build(
        const SpmFmbResolvedExecutionProfile& profile,
        const SpmFmbResolvedExecutionStep& step) {
    const SpmFmbRuntimeMemberCapability capability =
        spm_fmb_runtime_member_capability();
    if (capability == SpmFmbRuntimeMemberCapability::Unsealed) {
        TORCH_CHECK(profile.runtime_member_capability_ == capability,
                    "schema-v12 runtime member callback: profile capability "
                    "drift");
        return;
    }
    TORCH_CHECK(
        capability ==
                SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs &&
            profile.runtime_member_capability_ == capability &&
            profile.cpu_dry_ && profile.build_body_iterations_ == 1 &&
            profile.resolved_chunk_mode_ ==
                static_cast<uint8_t>(ChunkMode::SEQUENTIAL) &&
            profile.resolved_inter_layer_io_ ==
                static_cast<uint8_t>(InterLayerIO::DDR_PINGPONG) &&
            !profile.resolved_chunk_outer_within_group_ &&
            step.kind == SpmFmbOccurrenceKind::LayerBody &&
            step.body_id == 0 && step.layer >= 0 &&
            step.layer < profile.resolved_num_layers_,
        "schema-v12 runtime member callback STOP: initial contract accepts "
        "only CPU-dry canonical sequential DDR callbacks");
    TORCH_CHECK(
        profile.spm_peer_capability_ == spm_fmb_spm_peer_capability(),
        "schema-v13 typed SPM peer callback STOP: profile capability drift");
    TORCH_CHECK(!pimpl_->pipeline_runtime_member_frame_.active &&
                    g_fmb_runtime_member_owner == nullptr,
                "schema-v12 runtime member callback STOP: nested callback");
    pimpl_->pipeline_runtime_replay_authority_ = {};
    const std::shared_ptr<const uint64_t> owner =
        profile.owner_generation_.lock();
    TORCH_CHECK(owner != nullptr &&
                    owner.get() ==
                        pimpl_->pipeline_manifest_generation_.get() &&
                    *owner == profile.expected_owner_generation_,
                "schema-v12 runtime member callback STOP: profile owner is "
                "stale");
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "schema-v12 runtime member callback requires an active Graph");
    RpuKernelGraph& graph = RpuKernelGraph::active();
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    TORCH_CHECK(stamp.state == RpuKernelGraph::State::RECORDING &&
                    stamp.signature_segment_key == profile.profile_hash_,
                "schema-v12 runtime member callback STOP: BUILD Graph "
                "identity drift");

    size_t callback_index = profile.steps_.size();
    std::vector<const SpmFmbResolvedExecutionStep*> layer_members;
    for (size_t index = 0; index < profile.steps_.size(); ++index) {
        const auto& candidate = profile.steps_[index];
        if (candidate.key == step.key) callback_index = index;
        if (candidate.kind == SpmFmbOccurrenceKind::LayerBody &&
            candidate.body_id == step.body_id &&
            candidate.group_id == step.group_id &&
            candidate.layer == step.layer) {
            layer_members.push_back(&candidate);
        }
    }
    TORCH_CHECK(callback_index < profile.steps_.size() &&
                    layer_members.size() >= 2,
                "schema-v12 runtime member callback STOP: canonical member "
                "group is incomplete");
    const bool carrier = step.chunk_index == 0;
    std::vector<Impl::RuntimeDenseMemberDescriptor> descriptors;
    if (carrier) {
        descriptors.reserve(layer_members.size());
        TORCH_CHECK(
            static_cast<uint64_t>(profile.resolved_hidden_size_) <=
                std::numeric_limits<size_t>::max() / sizeof(c10::Half),
            "schema-v12 runtime member callback: row bytes overflow");
        const size_t row_bytes =
            static_cast<size_t>(profile.resolved_hidden_size_) *
            sizeof(c10::Half);
        size_t expected_byte_begin = 0;
        for (size_t ordinal = 0; ordinal < layer_members.size(); ++ordinal) {
            const auto& member = *layer_members[ordinal];
            TORCH_CHECK(member.chunk_index == static_cast<int>(ordinal) &&
                            member.chunk_offset >= 0 &&
                            member.chunk_len > 0 &&
                            static_cast<uint64_t>(member.chunk_offset) <=
                                std::numeric_limits<size_t>::max() /
                                    row_bytes &&
                            static_cast<uint64_t>(member.chunk_len) <=
                                std::numeric_limits<size_t>::max() /
                                    row_bytes,
                        "schema-v12 runtime member callback STOP: member "
                        "geometry drift");
            const size_t byte_begin =
                static_cast<size_t>(member.chunk_offset) * row_bytes;
            const size_t byte_count =
                static_cast<size_t>(member.chunk_len) * row_bytes;
            TORCH_CHECK(byte_begin == expected_byte_begin &&
                            byte_count > 0 && byte_count % 16 == 0 &&
                            byte_count <=
                                std::numeric_limits<size_t>::max() -
                                    byte_begin,
                        "schema-v12 runtime member callback STOP: member "
                        "ranges are not one contiguous exact cover");
            expected_byte_begin = byte_begin + byte_count;
            const SpmFmbDdrArenaRole ingress_arena = member.layer == 0
                ? SpmFmbDdrArenaRole::LiveInput
                : (member.layer % 2 == 0
                       ? SpmFmbDdrArenaRole::PingPongA
                       : SpmFmbDdrArenaRole::PingPongB);
            const SpmFmbDdrArenaRole egress_arena =
                member.layer == profile.resolved_num_layers_ - 1
                ? SpmFmbDdrArenaRole::FinalOutput
                : (member.layer % 2 == 0
                       ? SpmFmbDdrArenaRole::PingPongB
                       : SpmFmbDdrArenaRole::PingPongA);
            descriptors.push_back({
                static_cast<size_t>(member.layer), ordinal,
                layer_members.size(),
                static_cast<size_t>(member.key - 1), member.key,
                member.body_id, member.group_id, member.layer,
                member.chunk_offset, member.chunk_len,
                /*local_position_begin=*/0,
                /*local_kv_seq_len=*/member.chunk_len,
                ingress_arena,
                member.layer == 0
                    ? SpmFmbDdrAddressMode::MutableBase
                    : SpmFmbDdrAddressMode::Stable,
                byte_begin, byte_count,
                egress_arena, SpmFmbDdrAddressMode::Stable,
                byte_begin, byte_count,
            });
        }
    }

    TORCH_CHECK(pimpl_->pipeline_runtime_callback_epoch_ !=
                    std::numeric_limits<uint64_t>::max(),
                "schema-v12 runtime member callback epoch exhausted");
    const uint64_t epoch = ++pimpl_->pipeline_runtime_callback_epoch_;
    graph.begin_canonical_dma_callback_for_fmb(
        pimpl_->pipeline_manifest_generation_,
        profile.expected_owner_generation_, profile.profile_hash_,
        static_cast<size_t>(step.layer), descriptors.size());

    Impl::RuntimeDenseMemberFrame frame;
    frame.active = true;
    frame.cpu_dry = true;
    frame.callback_epoch = epoch;
    frame.graph_identity = &graph;
    frame.graph_build_generation = stamp.build_generation;
    frame.profile_hash = profile.profile_hash_;
    frame.allocation_hash = profile.expected_allocation_hash_;
    frame.expected_owner_generation = profile.expected_owner_generation_;
    frame.spm_peer_capability = profile.spm_peer_capability_;
    frame.callback_index = callback_index;
    frame.callback_key = step.key;
    frame.layer = step.layer;
    frame.members = std::move(descriptors);
    pimpl_->pipeline_runtime_member_frame_ = std::move(frame);
    g_fmb_runtime_member_owner = pimpl_.get();
}

void FusedModelBase::begin_spm_pipeline_runtime_member_callback_for_replay(
        const SpmFmbSealedDmaEndpointBindings& bindings,
        size_t callback_index) {
    const auto& members = bindings.source_members_;
    const auto& groups = members.source_groups_;
    const auto& trace = groups.source_trace_;
    const SpmFmbRuntimeMemberCapability capability =
        spm_fmb_runtime_member_capability();
    TORCH_CHECK(
        capability ==
                SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs &&
            trace.runtime_member_capability_ == capability &&
            trace.cpu_dry_ && callback_index < trace.callback_summaries_.size(),
        "schema-v12 runtime member REPLAY callback STOP: binding capability "
        "or callback identity drift");
    TORCH_CHECK(
        trace.spm_peer_capability_ == spm_fmb_spm_peer_capability(),
        "schema-v13 typed SPM peer REPLAY callback STOP: capability drift");
    auto& authority = pimpl_->pipeline_runtime_replay_authority_;
    const bool authority_binding_matches =
        (authority.bindings != nullptr &&
         authority.bindings->binding_hash_ == bindings.binding_hash_) ||
        (authority.spm_peer_bindings != nullptr &&
         authority.spm_peer_bindings->source_dma_bindings_.binding_hash_ ==
             bindings.binding_hash_);
    TORCH_CHECK(
        authority.armed && authority_binding_matches &&
            authority.next_callback == callback_index &&
            !pimpl_->pipeline_runtime_member_frame_.active &&
            g_fmb_runtime_member_owner == nullptr &&
            RpuKernelGraph::has_active(),
        "schema-v12 runtime member REPLAY callback STOP: replay authority is "
        "stale or out of order");
    RpuKernelGraph& graph = RpuKernelGraph::active();
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    TORCH_CHECK(&graph == authority.graph_identity &&
                    stamp.state == RpuKernelGraph::State::REPLAYING &&
                    stamp.build_generation ==
                        authority.graph_build_generation &&
                    stamp.signature_identity ==
                        authority.graph_signature_identity &&
                    stamp.signature_segment_key == authority.profile_hash,
                "schema-v12 runtime member REPLAY callback STOP: Graph "
                "identity drift");
    const auto& callback = trace.callback_summaries_[callback_index];

    size_t layer_group_ordinal = groups.group_summaries_.size();
    bool carrier = false;
    for (size_t group_index = 0;
         group_index < groups.group_summaries_.size(); ++group_index) {
        const auto& group = groups.group_summaries_[group_index];
        if (callback_index >= group.first_callback &&
            callback_index < group.first_callback + group.member_count) {
            layer_group_ordinal = group_index;
            carrier = callback_index == group.first_callback;
            break;
        }
    }
    TORCH_CHECK(layer_group_ordinal < groups.group_summaries_.size() &&
                    callback.layer ==
                        static_cast<int>(layer_group_ordinal),
                "schema-v12 runtime member REPLAY callback STOP: layer "
                "group drift");

    std::vector<Impl::RuntimeDenseMemberDescriptor> descriptors;
    if (carrier) {
        descriptors.reserve(members.members_per_group_);
        for (size_t ordinal = 0; ordinal < members.members_per_group_;
             ++ordinal) {
            const auto& member = members.member_summaries_[
                layer_group_ordinal * members.members_per_group_ + ordinal];
            descriptors.push_back({
                member.layer_group_ordinal, member.member_ordinal,
                member.member_count, member.callback_index,
                member.callback_key, member.body_id,
                member.framework_group_id, member.layer,
                member.packed_row_begin, member.row_count,
                member.local_position_begin, member.local_kv_seq_len,
                member.ingress.arena, member.ingress.address_mode,
                member.ingress.byte_begin, member.ingress.byte_count,
                member.egress.arena, member.egress.address_mode,
                member.egress.byte_begin, member.egress.byte_count,
            });
        }
    }

    TORCH_CHECK(pimpl_->pipeline_runtime_callback_epoch_ !=
                    std::numeric_limits<uint64_t>::max(),
                "schema-v12 runtime member callback epoch exhausted");
    const uint64_t epoch = ++pimpl_->pipeline_runtime_callback_epoch_;
    graph.begin_canonical_dma_callback_for_fmb(
        pimpl_->pipeline_manifest_generation_,
        trace.expected_owner_generation_, trace.profile_hash_,
        layer_group_ordinal, descriptors.size());

    Impl::RuntimeDenseMemberFrame frame;
    frame.active = true;
    frame.cpu_dry = true;
    frame.callback_epoch = epoch;
    frame.graph_identity = &graph;
    frame.graph_build_generation = stamp.build_generation;
    frame.profile_hash = trace.profile_hash_;
    frame.allocation_hash = trace.allocation_hash_;
    frame.expected_owner_generation = trace.expected_owner_generation_;
    frame.spm_peer_capability = trace.spm_peer_capability_;
    frame.callback_index = callback_index;
    frame.callback_key = callback.key;
    frame.layer = callback.layer;
    frame.members = std::move(descriptors);
    pimpl_->pipeline_runtime_member_frame_ = std::move(frame);
    g_fmb_runtime_member_owner = pimpl_.get();
}

void FusedModelBase::begin_spm_pipeline_runtime_member_callback_for_replay(
        const SpmFmbSealedSpmPeerBindings& bindings,
        size_t callback_index) {
    begin_spm_pipeline_runtime_member_callback_for_replay(
        bindings.source_dma_bindings_, callback_index);
}

void FusedModelBase::end_spm_pipeline_runtime_member_callback() {
    if (spm_fmb_runtime_member_capability() ==
        SpmFmbRuntimeMemberCapability::Unsealed) {
        return;
    }
    auto& frame = pimpl_->pipeline_runtime_member_frame_;
    TORCH_CHECK(frame.active && g_fmb_runtime_member_owner == pimpl_.get(),
                "schema-v12 runtime member callback STOP: end without begin");
    TORCH_CHECK(
        frame.next_member == frame.members.size() &&
            frame.next_role == SpmFmbDmaEndpointRole::Ingress,
        "schema-v12 runtime member callback STOP: incomplete member/role "
        "exact cover");
    validate_dense_spm_peer_callback_complete(frame);
    TORCH_CHECK(RpuKernelGraph::has_active() &&
                    &RpuKernelGraph::active() == frame.graph_identity,
                "schema-v12 runtime member callback STOP: Graph owner drift "
                "at end");
    auto& authority = pimpl_->pipeline_runtime_replay_authority_;
    if (authority.armed &&
        RpuKernelGraph::active().state() ==
            RpuKernelGraph::State::REPLAYING) {
        TORCH_CHECK(authority.next_callback == frame.callback_index,
                    "schema-v12 runtime member callback STOP: replay cursor "
                    "drift at end");
    }
    RpuKernelGraph::active().end_canonical_dma_callback_for_fmb();
    if (authority.armed &&
        RpuKernelGraph::active().state() ==
            RpuKernelGraph::State::REPLAYING) {
        ++authority.next_callback;
    }
    frame = {};
    g_fmb_runtime_member_owner = nullptr;
}

void FusedModelBase::cancel_spm_pipeline_runtime_member_callback() noexcept {
    if (pimpl_->pipeline_runtime_member_frame_.active &&
        RpuKernelGraph::has_active() &&
        &RpuKernelGraph::active() ==
            pimpl_->pipeline_runtime_member_frame_.graph_identity) {
        RpuKernelGraph::active().cancel_canonical_dma_callback_for_fmb();
    }
    pimpl_->pipeline_runtime_member_frame_ = {};
    pimpl_->pipeline_runtime_replay_authority_ = {};
    if (g_fmb_runtime_member_owner == pimpl_.get()) {
        g_fmb_runtime_member_owner = nullptr;
    }
}

const SpmFmbRuntimeDenseDdrMemberRef::State&
FusedModelBase::validate_spm_pipeline_runtime_member_ref(
        const SpmFmbRuntimeDenseDdrMemberRef& member) const {
    TORCH_CHECK(member.state_ != nullptr,
                "schema-v12 runtime member ref STOP: empty reference");
    const auto& state = *member.state_;
    const auto& frame = pimpl_->pipeline_runtime_member_frame_;
    TORCH_CHECK(frame.active && g_fmb_runtime_member_owner == pimpl_.get(),
                "schema-v12 runtime member ref STOP: no active current "
                "callback");
    TORCH_CHECK(
        state.owner == pimpl_.get() &&
            state.callback_epoch == frame.callback_epoch &&
            state.graph_identity == frame.graph_identity &&
            state.graph_build_generation == frame.graph_build_generation &&
            state.profile_hash == frame.profile_hash &&
            state.allocation_hash == frame.allocation_hash &&
            state.expected_owner_generation ==
                frame.expected_owner_generation &&
            state.member_ordinal < frame.members.size() &&
            RpuKernelGraph::has_active() &&
            &RpuKernelGraph::active() == frame.graph_identity,
        "schema-v12 runtime member ref STOP: reference escaped its current "
        "callback");
    const auto& canonical = frame.members[state.member_ordinal];
    TORCH_CHECK(
        state.layer_group_ordinal == canonical.layer_group_ordinal &&
            state.member_ordinal == canonical.member_ordinal &&
            state.member_count == canonical.member_count &&
            state.callback_index == canonical.callback_index &&
            state.callback_key == canonical.callback_key &&
            state.body_id == canonical.body_id &&
            state.framework_group_id == canonical.framework_group_id &&
            state.layer == canonical.layer &&
            state.packed_row_begin == canonical.packed_row_begin &&
            state.row_count == canonical.row_count &&
            state.local_position_begin == canonical.local_position_begin &&
            state.local_kv_seq_len == canonical.local_kv_seq_len &&
            state.ingress_arena == canonical.ingress_arena &&
            state.ingress_address_mode ==
                canonical.ingress_address_mode &&
            state.ingress_byte_begin == canonical.ingress_byte_begin &&
            state.ingress_byte_count == canonical.ingress_byte_count &&
            state.egress_arena == canonical.egress_arena &&
            state.egress_address_mode == canonical.egress_address_mode &&
            state.egress_byte_begin == canonical.egress_byte_begin &&
            state.egress_byte_count == canonical.egress_byte_count &&
            state.ingress_endpoint != nullptr &&
            state.egress_endpoint != nullptr,
        "schema-v12 runtime member ref STOP: reference payload drift");
    return state;
}

size_t FusedModelBase::current_spm_pipeline_dense_ddr_member_count() const {
    TORCH_CHECK(spm_fmb_runtime_member_capability() ==
                    SpmFmbRuntimeMemberCapability::
                        CurrentCallbackDenseDdrRefs &&
                    pimpl_->pipeline_runtime_member_frame_.active &&
                    g_fmb_runtime_member_owner == pimpl_.get(),
                "schema-v12 runtime member ref STOP: no active current "
                "callback");
    return pimpl_->pipeline_runtime_member_frame_.members.size();
}

SpmFmbRuntimeDenseDdrMemberRef
FusedModelBase::current_spm_pipeline_dense_ddr_member(
        size_t ordinal) const {
    const size_t member_count =
        current_spm_pipeline_dense_ddr_member_count();
    TORCH_CHECK(ordinal < member_count,
                "schema-v12 runtime member ref STOP: member ordinal is "
                "outside the current carrier callback");
    const auto& frame = pimpl_->pipeline_runtime_member_frame_;
    const auto& member = frame.members[ordinal];
    const auto make_endpoint = [&](SpmFmbDmaEndpointRole role) {
        const bool ingress = role == SpmFmbDmaEndpointRole::Ingress;
        const SpmFmbDdrArenaRole arena = ingress
            ? member.ingress_arena : member.egress_arena;
        const SpmFmbDdrAddressMode address_mode = ingress
            ? member.ingress_address_mode : member.egress_address_mode;
        const size_t byte_begin = ingress
            ? member.ingress_byte_begin : member.egress_byte_begin;
        const size_t byte_count = ingress
            ? member.ingress_byte_count : member.egress_byte_count;
        const uint8_t occurrence_count = ingress ? 8 : 1;
        const DenseDmaEndpointDescriptor descriptor =
            make_dense_dma_endpoint_descriptor(
                frame.profile_hash, frame.allocation_hash,
                member.callback_key, member.body_id,
                member.framework_group_id, member.layer,
                member.member_ordinal, member.member_count,
                member.packed_row_begin, member.row_count,
                member.local_position_begin, member.local_kv_seq_len,
                role, arena,
                ingress ? SpmFmbDdrAccess::Read
                        : SpmFmbDdrAccess::Write,
                address_mode, occurrence_count, byte_begin, byte_count);
        return std::shared_ptr<const GraphDmaSemanticEndpoint>(
            new GraphDmaSemanticEndpoint(
                descriptor.semantic_id, frame.profile_hash,
                frame.allocation_hash, member.layer_group_ordinal,
                member.member_ordinal,
                static_cast<uint8_t>(descriptor.role),
                static_cast<uint8_t>(descriptor.arena),
                static_cast<uint8_t>(descriptor.access),
                static_cast<uint8_t>(descriptor.address_mode),
                descriptor.expected_dma_variant,
                descriptor.expected_occurrence_count,
                descriptor.byte_begin, descriptor.byte_count,
                /*requires_canonical_burst=*/true,
                pimpl_->pipeline_manifest_generation_,
                frame.expected_owner_generation));
    };

    auto state = std::make_shared<SpmFmbRuntimeDenseDdrMemberRef::State>();
    state->owner = pimpl_.get();
    state->callback_epoch = frame.callback_epoch;
    state->graph_identity = frame.graph_identity;
    state->graph_build_generation = frame.graph_build_generation;
    state->profile_hash = frame.profile_hash;
    state->allocation_hash = frame.allocation_hash;
    state->expected_owner_generation = frame.expected_owner_generation;
    state->layer_group_ordinal = member.layer_group_ordinal;
    state->member_ordinal = member.member_ordinal;
    state->member_count = member.member_count;
    state->callback_index = member.callback_index;
    state->callback_key = member.callback_key;
    state->body_id = member.body_id;
    state->framework_group_id = member.framework_group_id;
    state->layer = member.layer;
    state->packed_row_begin = member.packed_row_begin;
    state->row_count = member.row_count;
    state->local_position_begin = member.local_position_begin;
    state->local_kv_seq_len = member.local_kv_seq_len;
    state->ingress_arena = member.ingress_arena;
    state->ingress_address_mode = member.ingress_address_mode;
    state->ingress_byte_begin = member.ingress_byte_begin;
    state->ingress_byte_count = member.ingress_byte_count;
    state->egress_arena = member.egress_arena;
    state->egress_address_mode = member.egress_address_mode;
    state->egress_byte_begin = member.egress_byte_begin;
    state->egress_byte_count = member.egress_byte_count;
    state->ingress_endpoint = make_endpoint(
        SpmFmbDmaEndpointRole::Ingress);
    state->egress_endpoint = make_endpoint(
        SpmFmbDmaEndpointRole::Egress);
    return SpmFmbRuntimeDenseDdrMemberRef(std::move(state));
}

SpmFmbIndependentLocalContext
FusedModelBase::spm_pipeline_member_local_context(
        const SpmFmbRuntimeDenseDdrMemberRef& member) const {
    const auto& state = validate_spm_pipeline_runtime_member_ref(member);
    return SpmFmbIndependentLocalContext(
        state.member_ordinal, state.row_count,
        state.local_position_begin, state.local_kv_seq_len);
}

namespace {

uint64_t cpu_contract_ddr_arena_base(SpmFmbDdrArenaRole arena) {
    switch (arena) {
    case SpmFmbDdrArenaRole::LiveInput:
        return UINT64_C(0x21000000);
    case SpmFmbDdrArenaRole::PingPongA:
        return UINT64_C(0x61000000);
    case SpmFmbDdrArenaRole::PingPongB:
        return UINT64_C(0x62000000);
    case SpmFmbDdrArenaRole::FinalOutput:
        return UINT64_C(0x71000000);
    }
    TORCH_INTERNAL_ASSERT(false);
    return 0;
}

enum class DenseSpmPeerKind : uint8_t {
    SharedStage = 1,
    PackedGlobalSlice = 2,
};

uint64_t spm_peer_name_hash(const std::string& name) {
    return fmb_nonzero_hash(fmb_hash_string(kFmbFnvOffset, name));
}

FusedModelBase::Impl::BuildTraceSpmPeerRecord derive_dense_spm_peer(
    const FusedModelBase::Impl::CachedLayoutSnapshot& snapshot,
    const FusedModelBase::Impl::RuntimeDenseMemberFrame& frame,
    const FusedModelBase::Impl::RuntimeDenseMemberDescriptor& member,
    SpmFmbDmaEndpointRole role,
    uint64_t endpoint_id,
    const char* buffer_name) {
    TORCH_CHECK(snapshot.valid && snapshot.cpu_dry &&
                    snapshot.allocation_hash == frame.allocation_hash &&
                    buffer_name != nullptr && buffer_name[0] != '\0' &&
                    endpoint_id != 0 && member.member_count >= 2 &&
                    member.member_ordinal < member.member_count &&
                    frame.members.size() == member.member_count,
                "schema-v13 typed SPM peer STOP: stale snapshot, endpoint, "
                "or member identity");

    uint32_t allocation_ordinal = 0;
    const auto* allocation = find_trace_allocation(
        snapshot, buffer_name, /*layer=*/-1, &allocation_ordinal);
    TORCH_CHECK(
        allocation != nullptr,
        "schema-v13 typed SPM peer STOP: selected buffer '", buffer_name,
        "' is not one flat allocation owned by the sealed snapshot");
    uint32_t declaration_ordinal = 0;
    const auto* declaration = find_trace_declaration(
        snapshot, buffer_name, &declaration_ordinal);
    TORCH_INTERNAL_ASSERT(declaration != nullptr);
    const uint32_t alias_root_ordinal =
        find_trace_root_ordinal(snapshot, *allocation);
    uint32_t root_lookup_ordinal = 0;
    const auto* root = find_trace_allocation(
        snapshot, allocation->alias_root.c_str(), /*layer=*/-1,
        &root_lookup_ordinal);
    TORCH_INTERNAL_ASSERT(root != nullptr &&
                          root_lookup_ordinal == alias_root_ordinal);

    TORCH_CHECK(
        declaration->storage == StorageClass::Temp &&
            allocation->storage == StorageClass::Temp &&
            root->storage == StorageClass::Temp &&
            declaration->scope == BufferScope::LayerWide &&
            allocation->layer == -1 && root->layer == -1 &&
            declaration->per_layer == 0 &&
            !declaration->reverse_layer_alloc &&
            !declaration->preload_present &&
            !allocation->preload_present && !root->preload_present,
        "schema-v13 typed SPM peer STOP: selected peer must be one flat "
        "temporary LayerWide allocation without preload metadata");
    TORCH_CHECK(
        declaration->alias_of.empty() || declaration->logical_bytes > 0,
        "schema-v13 typed SPM peer STOP: whole-root zero-size alias is not "
        "an initial typed peer");
    TORCH_CHECK(
        allocation->offset >= root->offset &&
            static_cast<size_t>(allocation->offset - root->offset) <=
                root->aligned_bytes &&
            allocation->aligned_bytes <=
                root->aligned_bytes -
                    static_cast<size_t>(allocation->offset - root->offset),
        "schema-v13 typed SPM peer STOP: selected allocation is outside its "
        "sealed alias root");

    size_t max_member_bytes = 0;
    size_t dense_payload_bytes = 0;
    for (const auto& candidate : frame.members) {
        const size_t bytes = role == SpmFmbDmaEndpointRole::Ingress
            ? candidate.ingress_byte_count : candidate.egress_byte_count;
        TORCH_CHECK(
            bytes > 0 && bytes % 16 == 0 &&
                bytes <= std::numeric_limits<size_t>::max() -
                    dense_payload_bytes,
            "schema-v13 typed SPM peer STOP: dense member byte geometry "
            "overflows or is unsupported");
        max_member_bytes = std::max(max_member_bytes, bytes);
        dense_payload_bytes += bytes;
    }
    const size_t shared_capacity = align_spm_bytes(max_member_bytes);
    const size_t packed_capacity = align_spm_bytes(dense_payload_bytes);
    const bool shared_match =
        allocation->logical_bytes == max_member_bytes;
    const bool packed_match =
        allocation->logical_bytes == dense_payload_bytes;
    TORCH_CHECK(
        shared_match != packed_match,
        "schema-v13 typed SPM peer STOP: logical capacity is undersized, "
        "unsupported, or ambiguous between shared and packed");
    const DenseSpmPeerKind kind = shared_match
        ? DenseSpmPeerKind::SharedStage
        : DenseSpmPeerKind::PackedGlobalSlice;
    TORCH_CHECK(
        allocation->aligned_bytes ==
            (shared_match ? shared_capacity : packed_capacity),
        "schema-v13 typed SPM peer STOP: aligned capacity differs from the "
        "exact logical dense peer");

    const size_t packed_begin = role == SpmFmbDmaEndpointRole::Ingress
        ? member.ingress_byte_begin : member.egress_byte_begin;
    const size_t peer_slice_count = role == SpmFmbDmaEndpointRole::Ingress
        ? member.ingress_byte_count : member.egress_byte_count;
    const size_t peer_slice_begin =
        kind == DenseSpmPeerKind::SharedStage ? 0 : packed_begin;
    TORCH_CHECK(
        peer_slice_begin <= allocation->logical_bytes &&
            peer_slice_count <=
                allocation->logical_bytes - peer_slice_begin &&
            peer_slice_begin <= allocation->aligned_bytes &&
            peer_slice_count <=
                allocation->aligned_bytes - peer_slice_begin,
        "schema-v13 typed SPM peer STOP: derived dense slice exceeds the "
        "selected allocation");

    const uint8_t core_mask = role == SpmFmbDmaEndpointRole::Ingress
        ? static_cast<uint8_t>(0xff) : static_cast<uint8_t>(0x01);
    const uint8_t occurrence_count =
        role == SpmFmbDmaEndpointRole::Ingress ? 8 : 1;
    const uint32_t allocation_offset_from_root =
        allocation->offset - root->offset;
    const uint64_t declaration_name_hash =
        spm_peer_name_hash(declaration->name);
    const uint64_t alias_root_name_hash =
        spm_peer_name_hash(allocation->alias_root);

    uint64_t peer_hash = fmb_hash_u64(kFmbFnvOffset, 23);
    peer_hash = fmb_hash_u64(peer_hash, frame.profile_hash);
    peer_hash = fmb_hash_u64(peer_hash, frame.allocation_hash);
    peer_hash = fmb_hash_u64(peer_hash, endpoint_id);
    peer_hash = fmb_hash_u64(peer_hash, member.layer_group_ordinal);
    peer_hash = fmb_hash_u64(peer_hash, member.member_ordinal);
    peer_hash = fmb_hash_u64(peer_hash, member.member_count);
    peer_hash = fmb_hash_u64(peer_hash, member.callback_key);
    peer_hash = fmb_hash_u64(peer_hash, member.body_id);
    peer_hash = fmb_hash_u64(peer_hash, member.framework_group_id);
    peer_hash = fmb_hash_u64(
        peer_hash, static_cast<uint64_t>(member.layer));
    peer_hash = fmb_hash_u64(peer_hash, static_cast<uint8_t>(role));
    peer_hash = fmb_hash_u64(peer_hash, static_cast<uint8_t>(kind));
    peer_hash = fmb_hash_u64(peer_hash, allocation_ordinal);
    peer_hash = fmb_hash_u64(peer_hash, declaration_ordinal);
    peer_hash = fmb_hash_u64(peer_hash, alias_root_ordinal);
    peer_hash = fmb_hash_u64(peer_hash, declaration_name_hash);
    peer_hash = fmb_hash_u64(peer_hash, alias_root_name_hash);
    peer_hash = fmb_hash_u64(
        peer_hash, static_cast<uint8_t>(allocation->storage));
    peer_hash = fmb_hash_u64(
        peer_hash, static_cast<uint8_t>(declaration->scope));
    peer_hash = fmb_hash_u64(
        peer_hash, static_cast<uint64_t>(allocation->layer));
    peer_hash = fmb_hash_u64(
        peer_hash, static_cast<uint64_t>(declaration->phase_start));
    peer_hash = fmb_hash_u64(
        peer_hash, static_cast<uint64_t>(declaration->phase_end));
    peer_hash = fmb_hash_u64(peer_hash, root->offset);
    peer_hash = fmb_hash_u64(peer_hash, allocation_offset_from_root);
    peer_hash = fmb_hash_u64(peer_hash, allocation->logical_bytes);
    peer_hash = fmb_hash_u64(peer_hash, allocation->aligned_bytes);
    peer_hash = fmb_hash_u64(peer_hash, dense_payload_bytes);
    peer_hash = fmb_hash_u64(peer_hash, peer_slice_begin);
    peer_hash = fmb_hash_u64(peer_hash, peer_slice_count);
    peer_hash = fmb_hash_u64(
        peer_hash, static_cast<uint8_t>(SpmPortDType::Fp16));
    peer_hash = fmb_hash_u64(
        peer_hash, static_cast<uint8_t>(SpmPortDistribution::Replicated));
    peer_hash = fmb_hash_u64(peer_hash, core_mask);
    peer_hash = fmb_hash_u64(peer_hash, occurrence_count);
    peer_hash = fmb_nonzero_hash(peer_hash);

    return {
        member.layer_group_ordinal,
        member.member_ordinal,
        role,
        static_cast<uint8_t>(kind),
        endpoint_id,
        peer_hash,
        allocation_ordinal,
        declaration_ordinal,
        alias_root_ordinal,
        declaration_name_hash,
        alias_root_name_hash,
        static_cast<uint8_t>(allocation->storage),
        static_cast<uint8_t>(declaration->scope),
        allocation->layer,
        declaration->phase_start,
        declaration->phase_end,
        root->offset,
        allocation_offset_from_root,
        allocation->logical_bytes,
        allocation->aligned_bytes,
        peer_slice_begin,
        peer_slice_count,
        core_mask,
        occurrence_count,
        peer_hash,
    };
}

bool same_dense_spm_peer(
    const FusedModelBase::Impl::BuildTraceSpmPeerRecord& lhs,
    const FusedModelBase::Impl::BuildTraceSpmPeerRecord& rhs) {
    return lhs.layer_group_ordinal == rhs.layer_group_ordinal &&
        lhs.member_ordinal == rhs.member_ordinal &&
        lhs.role == rhs.role && lhs.kind == rhs.kind &&
        lhs.endpoint_id == rhs.endpoint_id && lhs.peer_id == rhs.peer_id &&
        lhs.allocation_ordinal == rhs.allocation_ordinal &&
        lhs.declaration_ordinal == rhs.declaration_ordinal &&
        lhs.alias_root_ordinal == rhs.alias_root_ordinal &&
        lhs.declaration_name_hash == rhs.declaration_name_hash &&
        lhs.alias_root_name_hash == rhs.alias_root_name_hash &&
        lhs.storage == rhs.storage && lhs.scope == rhs.scope &&
        lhs.layer == rhs.layer && lhs.phase_start == rhs.phase_start &&
        lhs.phase_end == rhs.phase_end &&
        lhs.root_offset == rhs.root_offset &&
        lhs.allocation_offset_from_root == rhs.allocation_offset_from_root &&
        lhs.logical_capacity == rhs.logical_capacity &&
        lhs.aligned_capacity == rhs.aligned_capacity &&
        lhs.peer_slice_begin == rhs.peer_slice_begin &&
        lhs.peer_slice_count == rhs.peer_slice_count &&
        lhs.core_mask == rhs.core_mask &&
        lhs.expected_occurrence_count == rhs.expected_occurrence_count &&
        lhs.semantic_hash == rhs.semantic_hash;
}

template <typename Peer>
uint64_t expected_cpu_dense_spm_peer_address(
    const Peer& peer,
    uint8_t channel) {
    TORCH_CHECK(
        peer.layer_group_ordinal <=
                std::numeric_limits<uint64_t>::max() /
                    UINT64_C(0x01000000) &&
            peer.root_offset <=
                std::numeric_limits<uint32_t>::max() -
                    peer.allocation_offset_from_root &&
            peer.peer_slice_begin <=
                std::numeric_limits<uint32_t>::max() &&
            static_cast<uint64_t>(peer.root_offset) +
                    peer.allocation_offset_from_root <=
                std::numeric_limits<uint32_t>::max() -
                    peer.peer_slice_begin,
        "schema-v13 typed SPM peer STOP: expected CPU address overflows");
    const uint64_t local = static_cast<uint64_t>(peer.root_offset) +
        peer.allocation_offset_from_root + peer.peer_slice_begin;
    uint64_t address = UINT64_C(0x41000000) +
        static_cast<uint64_t>(peer.layer_group_ordinal) *
            UINT64_C(0x01000000) +
        local;
    if (peer.role == SpmFmbDmaEndpointRole::Ingress) {
        TORCH_CHECK(channel < 8,
                    "schema-v13 typed SPM peer STOP: ingress core channel "
                    "is outside [0,8)");
        address += static_cast<uint64_t>(channel) *
            UINT64_C(0x00100000);
    } else {
        TORCH_CHECK(peer.role == SpmFmbDmaEndpointRole::Egress &&
                        channel == 0,
                    "schema-v13 typed SPM peer STOP: egress must use core "
                    "zero/channel zero");
    }
    return address;
}

void validate_dense_spm_peer_before_dma(
    const FusedModelBase::Impl::RuntimeDenseMemberFrame& frame,
    const FusedModelBase::Impl::BuildTraceSpmPeerRecord& peer) {
    const auto& accumulator = peer.role == SpmFmbDmaEndpointRole::Ingress
        ? frame.ingress_peer : frame.egress_peer;
    TORCH_CHECK(
        peer.member_ordinal == accumulator.seen_member_count,
        "schema-v13 typed SPM peer STOP: member order drift");
    if (accumulator.initialized) {
        TORCH_CHECK(
            peer.kind == accumulator.kind &&
                peer.allocation_ordinal == accumulator.allocation_ordinal &&
                peer.declaration_ordinal ==
                    accumulator.declaration_ordinal &&
                peer.alias_root_ordinal ==
                    accumulator.alias_root_ordinal &&
                peer.declaration_name_hash ==
                    accumulator.declaration_name_hash &&
                peer.alias_root_name_hash ==
                    accumulator.alias_root_name_hash &&
                peer.storage == accumulator.storage &&
                peer.scope == accumulator.scope &&
                peer.layer == accumulator.layer &&
                peer.phase_start == accumulator.phase_start &&
                peer.phase_end == accumulator.phase_end &&
                peer.root_offset == accumulator.root_offset &&
                peer.allocation_offset_from_root ==
                    accumulator.allocation_offset_from_root &&
                peer.logical_capacity == accumulator.logical_capacity &&
                peer.aligned_capacity == accumulator.aligned_capacity,
            "schema-v13 typed SPM peer STOP: one layer/role changed its "
            "allocation, root, layout, or capacity");
    }
    if (peer.kind ==
        static_cast<uint8_t>(DenseSpmPeerKind::SharedStage)) {
        TORCH_CHECK(peer.peer_slice_begin == 0,
                    "schema-v13 typed SPM peer STOP: shared-stage slice "
                    "must start at zero");
    } else {
        TORCH_CHECK(
            peer.kind == static_cast<uint8_t>(
                             DenseSpmPeerKind::PackedGlobalSlice) &&
                peer.peer_slice_begin == accumulator.next_packed_begin,
            "schema-v13 typed SPM peer STOP: packed-global slices have a "
            "gap, overlap, or order drift");
    }
}

void commit_dense_spm_peer_after_dma(
    FusedModelBase::Impl::RuntimeDenseMemberFrame& frame,
    const FusedModelBase::Impl::BuildTraceSpmPeerRecord& peer) {
    auto& accumulator = peer.role == SpmFmbDmaEndpointRole::Ingress
        ? frame.ingress_peer : frame.egress_peer;
    if (!accumulator.initialized) {
        accumulator.initialized = true;
        accumulator.kind = peer.kind;
        accumulator.allocation_ordinal = peer.allocation_ordinal;
        accumulator.declaration_ordinal = peer.declaration_ordinal;
        accumulator.alias_root_ordinal = peer.alias_root_ordinal;
        accumulator.declaration_name_hash = peer.declaration_name_hash;
        accumulator.alias_root_name_hash = peer.alias_root_name_hash;
        accumulator.storage = peer.storage;
        accumulator.scope = peer.scope;
        accumulator.layer = peer.layer;
        accumulator.phase_start = peer.phase_start;
        accumulator.phase_end = peer.phase_end;
        accumulator.root_offset = peer.root_offset;
        accumulator.allocation_offset_from_root =
            peer.allocation_offset_from_root;
        accumulator.logical_capacity = peer.logical_capacity;
        accumulator.aligned_capacity = peer.aligned_capacity;
        for (const auto& member : frame.members) {
            const size_t bytes = peer.role == SpmFmbDmaEndpointRole::Ingress
                ? member.ingress_byte_count : member.egress_byte_count;
            TORCH_INTERNAL_ASSERT(
                bytes <= std::numeric_limits<size_t>::max() -
                    accumulator.dense_payload_bytes);
            accumulator.dense_payload_bytes += bytes;
        }
    }
    if (peer.kind == static_cast<uint8_t>(
                         DenseSpmPeerKind::PackedGlobalSlice)) {
        accumulator.next_packed_begin += peer.peer_slice_count;
    }
    ++accumulator.seen_member_count;
}

void validate_dense_spm_peer_callback_complete(
    const FusedModelBase::Impl::RuntimeDenseMemberFrame& frame) {
    if (frame.spm_peer_capability == SpmFmbSpmPeerCapability::Unsealed ||
        frame.members.empty()) {
        return;
    }
    const auto validate = [&](const auto& accumulator) {
        TORCH_CHECK(
            accumulator.initialized &&
                accumulator.seen_member_count == frame.members.size() &&
                (accumulator.kind == static_cast<uint8_t>(
                     DenseSpmPeerKind::SharedStage) ||
                 (accumulator.kind == static_cast<uint8_t>(
                     DenseSpmPeerKind::PackedGlobalSlice) &&
                  accumulator.next_packed_begin ==
                      accumulator.dense_payload_bytes)),
            "schema-v13 typed SPM peer STOP: callback did not exactly cover "
            "its shared/packed member universe");
    };
    validate(frame.ingress_peer);
    validate(frame.egress_peer);
}

}  // namespace

void FusedModelBase::emit_spm_pipeline_member_ingress_dma(
        const SpmFmbRuntimeDenseDdrMemberRef& member,
        const char* dst_buf) {
    const auto& state = validate_spm_pipeline_runtime_member_ref(member);
    auto& frame = pimpl_->pipeline_runtime_member_frame_;
    TORCH_CHECK(frame.next_member == state.member_ordinal &&
                    frame.next_role == SpmFmbDmaEndpointRole::Ingress,
                "schema-v12 runtime member callback STOP: ingress is not "
                "the exact next member/role");
    std::optional<Impl::BuildTraceSpmPeerRecord> spm_peer;
    if (frame.spm_peer_capability != SpmFmbSpmPeerCapability::Unsealed) {
        TORCH_CHECK(
            frame.spm_peer_capability ==
                SpmFmbSpmPeerCapability::TypedDensePackedOrShared,
            "schema-v13 typed SPM peer STOP: invalid runtime capability");
        spm_peer = derive_dense_spm_peer(
            pimpl_->pipeline_manifest_snapshot_, frame,
            frame.members.at(state.member_ordinal),
            SpmFmbDmaEndpointRole::Ingress,
            state.ingress_endpoint->semantic_id_, dst_buf);
        validate_dense_spm_peer_before_dma(frame, *spm_peer);
        if (RpuKernelGraph::active().state() ==
            RpuKernelGraph::State::REPLAYING) {
            const auto& authority =
                pimpl_->pipeline_runtime_replay_authority_;
            TORCH_CHECK(
                authority.armed &&
                    authority.spm_peer_capability ==
                        frame.spm_peer_capability &&
                    authority.next_peer <
                        authority.expected_spm_peers.size() &&
                    same_dense_spm_peer(
                        *spm_peer,
                        authority.expected_spm_peers[authority.next_peer]),
                "schema-v13 typed SPM peer REPLAY STOP: ingress peer "
                "differs from the sealed token");
        }
    }
    const size_t spm_slice_begin =
        spm_peer.has_value() ? spm_peer->peer_slice_begin : 0;
    TORCH_CHECK(spm_slice_begin <= std::numeric_limits<uint32_t>::max(),
                "schema-v13 typed SPM peer STOP: slice offset exceeds "
                "uint32 SPM addressing");
    std::array<uint64_t, 8> spm_dst_addrs{};
    uint32_t cpu_dry_offset = 0;
    if (frame.cpu_dry &&
        RpuKernelGraph::active().state() ==
            RpuKernelGraph::State::REPLAYING) {
        uint32_t allocation_ordinal = 0;
        const auto* allocation = find_trace_allocation(
            pimpl_->pipeline_manifest_snapshot_, dst_buf, /*layer=*/-1,
            &allocation_ordinal);
        if (allocation == nullptr) {
            allocation = find_trace_allocation(
                pimpl_->pipeline_manifest_snapshot_, dst_buf, state.layer,
                &allocation_ordinal);
        }
        TORCH_CHECK(allocation != nullptr,
                    "schema-v12 CPU REPLAY ingress SPM buffer is not owned "
                    "by the allocation snapshot");
        cpu_dry_offset = allocation->offset;
    }
    for (int core = 0; core < 8; ++core) {
        const uint32_t resolved = frame.cpu_dry &&
                RpuKernelGraph::active().state() ==
                    RpuKernelGraph::State::REPLAYING
            ? cpu_dry_offset : addr(core, dst_buf);
        TORCH_CHECK(
            resolved <= std::numeric_limits<uint32_t>::max() -
                static_cast<uint32_t>(spm_slice_begin),
            "schema-v13 typed SPM peer STOP: resolved ingress address "
            "overflows uint32");
        const uint32_t peer_resolved =
            resolved + static_cast<uint32_t>(spm_slice_begin);
        spm_dst_addrs[core] = frame.cpu_dry
            ? UINT64_C(0x41000000) +
                static_cast<uint64_t>(state.layer) * UINT64_C(0x01000000) +
                static_cast<uint64_t>(core) * UINT64_C(0x00100000) +
                peer_resolved
            : static_cast<uint64_t>(peer_resolved);
    }

    uint64_t fixed_ddr_src = 0;
    const uint64_t* live_src_base = nullptr;
    if (state.ingress_address_mode == SpmFmbDdrAddressMode::MutableBase) {
        live_src_base = frame.cpu_dry
            ? &pimpl_->pipeline_cpu_dry_live_input_base_
            : &current_hidden_in_src_base();
    } else if (frame.cpu_dry) {
        fixed_ddr_src = cpu_contract_ddr_arena_base(state.ingress_arena) +
            state.ingress_byte_begin;
    } else {
        c10::Half* base = state.ingress_arena == SpmFmbDdrArenaRole::PingPongA
            ? pimpl_->ddr_bufA_ptr_ : pimpl_->ddr_bufB_ptr_;
        TORCH_CHECK(base != nullptr,
                    "schema-v12 canonical ingress requires stable DDR "
                    "ping-pong storage");
        c10::Half* ptr = base + state.ingress_byte_begin /
            sizeof(c10::Half);
        rpu_ddr_flush(ptr);
        fixed_ddr_src = ::rhino_lkn::RpuGetDevAddr(ptr);
    }

    RpuKernelGraph& graph = RpuKernelGraph::active();
    graph.begin_canonical_dma_burst_for_fmb(
        *state.ingress_endpoint,
        spm_peer.has_value() ? spm_peer->peer_id : 0);
    bool burst_complete = false;
    auto cancel_burst = c10::make_scope_exit([&] {
        if (!burst_complete) graph.cancel_canonical_dma_burst_for_fmb();
    });
    if (live_src_base != nullptr) {
        rpu_launch_canonical_ddr_broadcast_spm_dma_mutable_src(
            *state.ingress_endpoint, live_src_base,
            static_cast<int64_t>(state.ingress_byte_begin),
            state.ingress_byte_count, spm_dst_addrs, /*num_cores=*/8);
    } else {
        rpu_launch_canonical_ddr_broadcast_spm_dma_fixed(
            *state.ingress_endpoint, fixed_ddr_src,
            state.ingress_byte_count, spm_dst_addrs, /*num_cores=*/8);
    }
    graph.end_canonical_dma_burst_for_fmb();
    burst_complete = true;
    if (spm_peer.has_value()) {
        commit_dense_spm_peer_after_dma(frame, *spm_peer);
        if (graph.state() == RpuKernelGraph::State::RECORDING) {
            auto& candidate = pimpl_->pipeline_build_trace_;
            TORCH_CHECK(candidate.armed && candidate.callback_open &&
                            !candidate.callbacks.empty(),
                        "schema-v13 typed SPM peer STOP: missing BUILD "
                        "callback authority");
            candidate.callbacks.back().spm_peers.push_back(*spm_peer);
        } else {
            auto& authority = pimpl_->pipeline_runtime_replay_authority_;
            TORCH_INTERNAL_ASSERT(
                authority.next_peer < authority.expected_spm_peers.size());
            ++authority.next_peer;
        }
    }
    frame.next_role = SpmFmbDmaEndpointRole::Egress;
}

void FusedModelBase::emit_spm_pipeline_member_egress_dma(
        const SpmFmbRuntimeDenseDdrMemberRef& member,
        const char* src_buf) {
    const auto& state = validate_spm_pipeline_runtime_member_ref(member);
    auto& frame = pimpl_->pipeline_runtime_member_frame_;
    TORCH_CHECK(frame.next_member == state.member_ordinal &&
                    frame.next_role == SpmFmbDmaEndpointRole::Egress,
                "schema-v12 runtime member callback STOP: egress is not the "
                "exact next member/role");
    std::optional<Impl::BuildTraceSpmPeerRecord> spm_peer;
    if (frame.spm_peer_capability != SpmFmbSpmPeerCapability::Unsealed) {
        TORCH_CHECK(
            frame.spm_peer_capability ==
                SpmFmbSpmPeerCapability::TypedDensePackedOrShared,
            "schema-v13 typed SPM peer STOP: invalid runtime capability");
        spm_peer = derive_dense_spm_peer(
            pimpl_->pipeline_manifest_snapshot_, frame,
            frame.members.at(state.member_ordinal),
            SpmFmbDmaEndpointRole::Egress,
            state.egress_endpoint->semantic_id_, src_buf);
        validate_dense_spm_peer_before_dma(frame, *spm_peer);
        if (RpuKernelGraph::active().state() ==
            RpuKernelGraph::State::REPLAYING) {
            const auto& authority =
                pimpl_->pipeline_runtime_replay_authority_;
            TORCH_CHECK(
                authority.armed &&
                    authority.spm_peer_capability ==
                        frame.spm_peer_capability &&
                    authority.next_peer <
                        authority.expected_spm_peers.size() &&
                    same_dense_spm_peer(
                        *spm_peer,
                        authority.expected_spm_peers[authority.next_peer]),
                "schema-v13 typed SPM peer REPLAY STOP: egress peer differs "
                "from the sealed token");
        }
    }
    const size_t spm_slice_begin =
        spm_peer.has_value() ? spm_peer->peer_slice_begin : 0;
    TORCH_CHECK(spm_slice_begin <= std::numeric_limits<uint32_t>::max(),
                "schema-v13 typed SPM peer STOP: slice offset exceeds "
                "uint32 SPM addressing");
    uint32_t resolved = 0;
    if (frame.cpu_dry &&
        RpuKernelGraph::active().state() ==
            RpuKernelGraph::State::REPLAYING) {
        uint32_t allocation_ordinal = 0;
        const auto* allocation = find_trace_allocation(
            pimpl_->pipeline_manifest_snapshot_, src_buf, /*layer=*/-1,
            &allocation_ordinal);
        if (allocation == nullptr) {
            allocation = find_trace_allocation(
                pimpl_->pipeline_manifest_snapshot_, src_buf, state.layer,
                &allocation_ordinal);
        }
        TORCH_CHECK(allocation != nullptr,
                    "schema-v12 CPU REPLAY egress SPM buffer is not owned by "
                    "the allocation snapshot");
        resolved = allocation->offset;
    } else {
        resolved = addr(/*core=*/0, src_buf);
    }
    TORCH_CHECK(
        resolved <= std::numeric_limits<uint32_t>::max() -
            static_cast<uint32_t>(spm_slice_begin),
        "schema-v13 typed SPM peer STOP: resolved egress address overflows "
        "uint32");
    const uint32_t peer_resolved =
        resolved + static_cast<uint32_t>(spm_slice_begin);
    const uint64_t spm_src = frame.cpu_dry
        ? UINT64_C(0x41000000) +
            static_cast<uint64_t>(state.layer) * UINT64_C(0x01000000) +
            peer_resolved
        : static_cast<uint64_t>(peer_resolved);
    uint64_t ddr_dst = 0;
    if (frame.cpu_dry) {
        ddr_dst = cpu_contract_ddr_arena_base(state.egress_arena) +
            state.egress_byte_begin;
    } else {
        c10::Half* base = nullptr;
        switch (state.egress_arena) {
        case SpmFmbDdrArenaRole::PingPongA:
            base = pimpl_->ddr_bufA_ptr_;
            break;
        case SpmFmbDdrArenaRole::PingPongB:
            base = pimpl_->ddr_bufB_ptr_;
            break;
        case SpmFmbDdrArenaRole::FinalOutput:
            base = output_ptr();
            break;
        case SpmFmbDdrArenaRole::LiveInput:
            TORCH_CHECK(false,
                        "schema-v12 canonical egress cannot target live "
                        "input storage");
        }
        TORCH_CHECK(base != nullptr,
                    "schema-v12 canonical egress requires stable DDR "
                    "storage");
        c10::Half* ptr = base + state.egress_byte_begin /
            sizeof(c10::Half);
        rpu_ddr_flush(ptr);
        ddr_dst = ::rhino_lkn::RpuGetDevAddr(ptr);
    }

    RpuKernelGraph& graph = RpuKernelGraph::active();
    graph.begin_canonical_dma_burst_for_fmb(
        *state.egress_endpoint,
        spm_peer.has_value() ? spm_peer->peer_id : 0);
    bool burst_complete = false;
    auto cancel_burst = c10::make_scope_exit([&] {
        if (!burst_complete) graph.cancel_canonical_dma_burst_for_fmb();
    });
    rpu_launch_canonical_spm_copy_ddr_dma_fixed(
        *state.egress_endpoint, spm_src, ddr_dst,
        state.egress_byte_count);
    graph.end_canonical_dma_burst_for_fmb();
    burst_complete = true;
    if (spm_peer.has_value()) {
        commit_dense_spm_peer_after_dma(frame, *spm_peer);
        if (graph.state() == RpuKernelGraph::State::RECORDING) {
            auto& candidate = pimpl_->pipeline_build_trace_;
            TORCH_CHECK(candidate.armed && candidate.callback_open &&
                            !candidate.callbacks.empty(),
                        "schema-v13 typed SPM peer STOP: missing BUILD "
                        "callback authority");
            candidate.callbacks.back().spm_peers.push_back(*spm_peer);
        } else {
            auto& authority = pimpl_->pipeline_runtime_replay_authority_;
            TORCH_INTERNAL_ASSERT(
                authority.next_peer < authority.expected_spm_peers.size());
            ++authority.next_peer;
        }
    }
    frame.next_role = SpmFmbDmaEndpointRole::Ingress;
    ++frame.next_member;
}

void FusedModelBase::emit_mlp_pipeline(const at::Tensor& gate_w,
                                       const at::Tensor& up_w,
                                       const at::Tensor& down_w,
                                       int64_t seq_len,
                                       ActivationKind act,
                                       const at::Tensor& gate_ws,
                                       const at::Tensor& up_ws,
                                       const at::Tensor& down_ws,
                                       uint32_t gate_nvfp4_ts_addr,
                                       uint32_t up_nvfp4_ts_addr,
                                       uint32_t down_nvfp4_ts_addr,
                                       uint16_t nvfp4_layer_id,
                                       bool force_twostage_chunk_v2,
                                       bool down_out_bf16,
                                       bool residual_is_bf16,
                                       bool acc32)
{
    constexpr int kNumCores = 8;
    int64_t elems = seq_len * (pimpl_->intermediate_size_ / kNumCores);

    TORCH_CHECK(!(down_out_bf16 && residual_is_bf16),
                "MLP mixed residual transition cannot start from BF16");

    auto emit_linear = [&](uint32_t input, const at::Tensor& weight,
                           uint32_t output, int64_t n, int64_t k,
                           int partition, const at::Tensor& scale,
                           uint32_t nvfp4_tensor_scale_spm_addr) {
        rpu_launch_linear_spm_to_spm_acc16_kernel(
            input, weight, output, seq_len, n, k, partition,
            /*num_cores=*/8, /*bias_spm_addr=*/0, /*force_gemm=*/false, scale,
            nvfp4_tensor_scale_spm_addr, nvfp4_layer_id, acc32);
    };

    emit_linear(
        addr(0, "residual1"), gate_w, addr(0, "gate"),
        pimpl_->intermediate_size_, pimpl_->hidden_size_, /*partition=*/1,
        gate_ws, gate_nvfp4_ts_addr);

    switch (act) {
        case ActivationKind::SILU:
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "gate"), addr(0, "gate"),
                elems, ValuOpType::SILU);
            break;
        case ActivationKind::GELU:
            rpu_launch_eltwise_unary_spm_kernel(
                addr(0, "gate"), addr(0, "gate"),
                elems, ValuOpType::ADD, GeluMode::TANH);
            break;
        case ActivationKind::NONE: break;
    }

    emit_linear(
        addr(0, "residual1"), up_w, addr(0, "up"),
        pimpl_->intermediate_size_, pimpl_->hidden_size_, /*partition=*/1,
        up_ws, up_nvfp4_ts_addr);

    rpu_launch_eltwise_binary_spm_kernel(
        addr(0, "gate"), addr(0, "up"), addr(0, "gate"),
        elems, ValuOpType::MUL, c10::Half(1.0));

    if (down_out_bf16) {
        rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(
            addr(0, "gate"), down_w, addr(0, "down"),
            seq_len, pimpl_->hidden_size_, pimpl_->intermediate_size_,
            /*partition=*/0, /*num_cores=*/8);
        rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(
            addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
            seq_len, pimpl_->hidden_size_);
    } else {
        emit_linear(
            addr(0, "gate"), down_w, addr(0, "down"),
            pimpl_->hidden_size_, pimpl_->intermediate_size_, /*partition=*/0,
            down_ws, down_nvfp4_ts_addr);
        if (residual_is_bf16) {
            rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(
                addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
                seq_len, pimpl_->hidden_size_);
        } else {
            rpu_launch_all_reduce_sum_residual_kernel(
                addr(0, "down"), addr(0, "residual2"), addr(0, "residual1"),
                seq_len, pimpl_->hidden_size_, /*input_num_cores=*/8,
                /*output_num_cores=*/8,
                /*force_twostage=*/force_twostage_chunk_v2,
                /*force_chunk_v2=*/force_twostage_chunk_v2);
        }
    }
}

namespace {

struct FmbExecutionTraversalSpec {
    int64_t num_layers = 0;
    int64_t cross_batch = 0;
    ChunkMode chunk_mode = ChunkMode::SEQUENTIAL;
    InterLayerIO inter_layer_io = InterLayerIO::SPM_RESIDENT;
    bool chunk_outer_within_group = false;
    bool has_kv_first = false;
    const std::vector<ChunkInfo>* chunks = nullptr;
    const std::vector<ChunkInfo>* kv_insert_chunks = nullptr;
};

struct FmbExecutionStepView {
    SpmFmbOccurrenceKind kind = SpmFmbOccurrenceKind::LayerBody;
    uint32_t body_id = 0;
    uint32_t group_id = 0;
    int layer = -1;
    uint32_t traversal_ordinal = 0;
    const ChunkInfo* chunk = nullptr;
    SpmFmbScopeMask scope_mask = 0;
    bool updates_io_context = false;
    bool input_in_spm = false;
    bool output_to_spm = false;
};

constexpr SpmFmbScopeMask kLayerComputeScopes =
    spm_fmb_scope_bit(BufferScope::LayerWide) |
    spm_fmb_scope_bit(BufferScope::Compute);
constexpr SpmFmbScopeMask kLayerKvInsertScopes =
    spm_fmb_scope_bit(BufferScope::LayerWide) |
    spm_fmb_scope_bit(BufferScope::KvInsert);
// Single source of truth for schema-v7 profile materialization and the actual
// run_all_layers body/group/layer/chunk/KV traversal.  pre/post callbacks are
// deliberately outside its contract and schema-v7 STOPs when any is present.
// The helper is templated and emits stack-only POD views: ordinary Path 3 and
// REPLAY never allocate a profile, hash strings, or build a step vector.
template <typename Visitor>
void for_each_fmb_execution_step(const FmbExecutionTraversalSpec& spec,
                                 uint32_t body_id,
                                 Visitor&& visitor) {
    TORCH_INTERNAL_ASSERT(spec.chunks != nullptr &&
                          spec.kv_insert_chunks != nullptr);
    uint32_t traversal_ordinal = 0;
    uint32_t group_id = 0;
    for (int64_t gs = 0; gs < spec.num_layers;
         gs += spec.cross_batch, ++group_id) {
            const int64_t ge =
                std::min(gs + spec.cross_batch, spec.num_layers) - 1;
            if (spec.chunk_outer_within_group) {
                TORCH_CHECK(spec.chunk_mode == ChunkMode::SEQUENTIAL,
                            "chunk_outer_within_group requires SEQUENTIAL "
                            "chunk mode");
                TORCH_CHECK(spec.inter_layer_io ==
                                InterLayerIO::SPM_RESIDENT,
                            "chunk_outer_within_group requires SPM_RESIDENT "
                            "inter-layer I/O");
                for (const ChunkInfo& chunk : *spec.chunks) {
                    for (int64_t layer = gs; layer <= ge; ++layer) {
                        visitor(FmbExecutionStepView{
                            SpmFmbOccurrenceKind::LayerBody,
                            body_id, group_id, static_cast<int>(layer),
                            traversal_ordinal++, &chunk, kLayerComputeScopes,
                            true, layer != gs, layer != ge});
                    }
                }
                continue;
            }

            for (int64_t layer = gs; layer <= ge; ++layer) {
                const bool input_in_spm =
                    spec.inter_layer_io == InterLayerIO::SPM_RESIDENT &&
                    layer != gs;
                const bool output_to_spm =
                    spec.inter_layer_io == InterLayerIO::SPM_RESIDENT &&
                    layer != ge;
                if (spec.chunk_mode == ChunkMode::SEQUENTIAL) {
                    bool update_io_context = true;
                    for (const ChunkInfo& chunk : *spec.chunks) {
                        visitor(FmbExecutionStepView{
                            SpmFmbOccurrenceKind::LayerBody,
                            body_id, group_id, static_cast<int>(layer),
                            traversal_ordinal++, &chunk, kLayerComputeScopes,
                            update_io_context,
                            input_in_spm, output_to_spm});
                        update_io_context = false;
                    }
                    continue;
                }

                bool update_io_context = true;
                if (spec.has_kv_first) {
                    for (const ChunkInfo& chunk : *spec.kv_insert_chunks) {
                        visitor(FmbExecutionStepView{
                            SpmFmbOccurrenceKind::KvInsertBody,
                            body_id, group_id, static_cast<int>(layer),
                            traversal_ordinal++, &chunk,
                            kLayerKvInsertScopes,
                            update_io_context,
                            input_in_spm, output_to_spm});
                        update_io_context = false;
                    }
                }
                for (const ChunkInfo& chunk : *spec.chunks) {
                    visitor(FmbExecutionStepView{
                        SpmFmbOccurrenceKind::LayerBody,
                        body_id, group_id, static_cast<int>(layer),
                        traversal_ordinal++, &chunk, kLayerComputeScopes,
                        update_io_context,
                        input_in_spm, output_to_spm});
                    update_io_context = false;
                }
            }
    }
}

int64_t validate_fmb_resolved_chunks(const std::vector<ChunkInfo>& chunks,
                                     const char* role) {
    TORCH_CHECK(!chunks.empty(), "schema-v7 resolved profile: ", role,
                " chunks must not be empty");
    TORCH_CHECK(chunks.front().len > 0 && chunks.front().offset == 0,
                "schema-v7 resolved profile: ", role,
                " chunks must start at offset zero with positive length");
    TORCH_CHECK(chunks.size() <=
                    static_cast<size_t>(std::numeric_limits<int>::max()),
                "schema-v7 resolved profile: ", role,
                " chunk count exceeds int indexing");
    const int64_t full_len = chunks.front().len;
    TORCH_CHECK(chunks.front().kv_seq_len >= full_len,
                "schema-v7 resolved profile: ", role,
                " first kv_seq_len is smaller than its chunk length");
    const int64_t position_base = chunks.front().kv_seq_len - full_len;
    int64_t expected_offset = 0;
    for (size_t ordinal = 0; ordinal < chunks.size(); ++ordinal) {
        const ChunkInfo& chunk = chunks[ordinal];
        TORCH_CHECK(chunk.idx == static_cast<int>(ordinal) &&
                        chunk.offset == expected_offset && chunk.len > 0,
                    "schema-v7 resolved profile: ", role,
                    " chunks require contiguous idx/offset and positive len");
        TORCH_CHECK(ordinal + 1 == chunks.size()
                        ? chunk.len <= full_len
                        : chunk.len == full_len,
                    "schema-v7 resolved profile: ", role,
                    " permits only one final tail chunk");
        TORCH_CHECK(position_base <=
                        std::numeric_limits<int64_t>::max() - chunk.offset &&
                        position_base + chunk.offset <=
                            std::numeric_limits<int64_t>::max() - chunk.len,
                    "schema-v7 resolved profile: ", role,
                    " kv_seq_len calculation overflows int64");
        const int64_t expected_kv_seq_len =
            position_base + chunk.offset + chunk.len;
        TORCH_CHECK(chunk.kv_seq_len == expected_kv_seq_len,
                    "schema-v7 resolved profile: ", role,
                    " kv_seq_len must advance monotonically with the same "
                    "position base");
        TORCH_CHECK(expected_offset <=
                        std::numeric_limits<int64_t>::max() - chunk.len,
                    "schema-v7 resolved profile: ", role,
                    " total chunk length overflows int64");
        expected_offset += chunk.len;
    }
    return expected_offset;
}

void validate_trace_step_exact(
    const SpmFmbResolvedExecutionStep& expected,
    const FmbExecutionStepView& actual) {
    const bool chunk_matches = expected.has_chunk
        ? actual.chunk != nullptr &&
            expected.chunk_index == actual.chunk->idx &&
            expected.chunk_offset == actual.chunk->offset &&
            expected.chunk_len == actual.chunk->len &&
            expected.chunk_kv_seq_len == actual.chunk->kv_seq_len
        : actual.chunk == nullptr && expected.chunk_index == -1 &&
            expected.chunk_offset == 0 && expected.chunk_len == 0 &&
            expected.chunk_kv_seq_len == 0;
    TORCH_CHECK(
        expected.kind == actual.kind &&
            expected.body_id == actual.body_id &&
            expected.group_id == actual.group_id &&
            expected.layer == actual.layer &&
            expected.traversal_ordinal == actual.traversal_ordinal &&
            chunk_matches &&
            expected.scope_mask == actual.scope_mask &&
            expected.updates_io_context == actual.updates_io_context &&
            expected.input_in_spm == actual.input_in_spm &&
            expected.output_to_spm == actual.output_to_spm,
        "schema-v8 BUILD trace STOP: runtime callback drift at resolved "
        "profile step ", expected.key);
}

void begin_trace_callback(
    FusedModelBase::Impl& state,
    const SpmFmbResolvedExecutionStep& expected,
    const FmbExecutionStepView& actual) {
    auto& candidate = state.pipeline_build_trace_;
    TORCH_CHECK(candidate.armed && candidate.profile != nullptr,
                "schema-v8 BUILD trace STOP: callback without an armed "
                "profile");
    TORCH_CHECK(!candidate.persistent_preload_callback_open &&
                    !candidate.callback_open &&
                    !g_fmb_build_trace_callback_begin.has_value(),
                "schema-v8 BUILD trace STOP: nested callback scope");
    validate_trace_step_exact(expected, actual);

    TORCH_CHECK(RpuKernelGraph::has_active() &&
                    &RpuKernelGraph::active() == candidate.graph_identity,
                "schema-v8 BUILD trace STOP: callback Graph owner drift");
    const GraphOpStreamStamp stamp =
        RpuKernelGraph::active().op_stream_stamp();
    TORCH_CHECK(stamp.state == RpuKernelGraph::State::RECORDING &&
                    stamp.graph == candidate.graph_identity &&
                    stamp.build_generation ==
                        candidate.graph_build_generation &&
                    stamp.signature_identity ==
                        candidate.graph_signature_identity &&
                    stamp.signature_segment_key ==
                        candidate.graph_signature_segment_key,
                "schema-v8 BUILD trace STOP: callback BUILD identity drift");
    const size_t expected_node_begin = candidate.callbacks.empty()
        ? candidate.graph_node_begin
        : candidate.callbacks.back().graph_node_end;
    TORCH_CHECK(stamp.position == expected_node_begin,
                "schema-v8 BUILD trace STOP: unowned Graph nodes appeared "
                "between callback windows");

    FusedModelBase::Impl::BuildTraceCallbackRecord callback;
    callback.kind = expected.kind;
    callback.key = expected.key;
    callback.body_id = expected.body_id;
    callback.group_id = expected.group_id;
    callback.layer = expected.layer;
    callback.traversal_ordinal = expected.traversal_ordinal;
    callback.chunk_index = expected.chunk_index;
    callback.chunk_offset = expected.chunk_offset;
    callback.chunk_len = expected.chunk_len;
    callback.chunk_kv_seq_len = expected.chunk_kv_seq_len;
    callback.scope_mask = expected.scope_mask;
    callback.updates_io_context = expected.updates_io_context;
    callback.input_in_spm = expected.input_in_spm;
    callback.output_to_spm = expected.output_to_spm;
    callback.local_phase_begin = expected.local_phase_begin;
    callback.local_phase_end = expected.local_phase_end;
    callback.global_origin = expected.global_origin.value();
    callback.graph_node_begin = stamp.position;
    candidate.callbacks.push_back(std::move(callback));
    candidate.callback_open = true;
    g_fmb_build_trace_callback_begin = stamp;
}

}  // namespace

void FusedModelBase::end_spm_pipeline_build_trace_callback() {
    auto& state = *pimpl_;
    auto& candidate = state.pipeline_build_trace_;
    TORCH_CHECK(candidate.armed && candidate.callback_open &&
                    !candidate.callbacks.empty() &&
                    g_fmb_build_trace_callback_begin.has_value(),
                "schema-v8 BUILD trace STOP: callback end without begin");
    TORCH_CHECK(RpuKernelGraph::has_active() &&
                    &RpuKernelGraph::active() == candidate.graph_identity,
                "schema-v8 BUILD trace STOP: callback Graph owner drift at "
                "end");
    const GraphOpStreamStamp end =
        RpuKernelGraph::active().op_stream_stamp();
    const GraphOpStreamWindowDigest digest =
        RpuKernelGraph::active().digest_op_stream_window(
            *g_fmb_build_trace_callback_begin, end);
    TORCH_CHECK(!digest.contains(GraphNodeKind::ChildGraph) &&
                    !digest.contains(GraphNodeKind::HostCallback) &&
                    !digest.contains(GraphNodeKind::Tier3Oneshot),
                "schema-v8 BUILD trace STOP: callback window contains an "
                "opaque ChildGraph/HostCallback/Tier3Oneshot node");

    auto& callback = candidate.callbacks.back();
    TORCH_CHECK(digest.begin == callback.graph_node_begin &&
                    digest.end == end.position &&
                    digest.node_count == digest.end - digest.begin,
                "schema-v8 BUILD trace STOP: callback Graph window drift");
    TORCH_CHECK(!(digest.node_count == 0 && !callback.accesses.empty()),
                "schema-v8 BUILD trace STOP: callback accessed owned SPM but "
                "recorded no Graph node");
    TORCH_CHECK(!(digest.node_count != 0 && callback.accesses.empty()),
                "schema-v8 BUILD trace STOP: callback recorded Graph nodes "
                "without an owned SPM access");
    const bool legacy_post_fn_yields =
        candidate.profile->post_fn_yield_capability_ !=
        SpmFmbPostFnYieldCapability::Unsealed;
    const bool callback_window_yields =
        candidate.profile->layer_producer_yield_capability_ !=
        SpmFmbLayerProducerYieldCapability::Unsealed;
    if (legacy_post_fn_yields &&
        callback.kind == SpmFmbOccurrenceKind::PostFn) {
        TORCH_CHECK(
            !candidate.producer_yield_open &&
                !callback.producer_yields.empty(),
            "post_fn producer-yield trace requires at least one completed "
            "yield scope");
        size_t expected_yield_begin = callback.graph_node_begin;
        for (size_t ordinal = 0;
             ordinal < callback.producer_yields.size(); ++ordinal) {
            const auto& yield = callback.producer_yields[ordinal];
            TORCH_CHECK(
                yield.ordinal == ordinal &&
                    yield.graph_node_begin == expected_yield_begin &&
                    yield.graph_node_begin < yield.graph_node_end,
                "post_fn producer-yield scopes must be nonempty, ordered, "
                "and exactly adjacent");
            expected_yield_begin = yield.graph_node_end;
        }
        TORCH_CHECK(
            expected_yield_begin == digest.end,
            "post_fn producer-yield scopes must exactly cover the callback "
            "Graph window");
    } else if (callback_window_yields) {
        TORCH_CHECK(
            !candidate.producer_yield_open &&
                !candidate.pending_producer_yield.has_value() &&
                (callback.producer_yields.empty() ||
                 callback.kind == SpmFmbOccurrenceKind::LayerBody ||
                 callback.kind == SpmFmbOccurrenceKind::PostFn),
            "callback producer-yield trace escaped a canonical LayerBody "
            "or PostFn callback");
        size_t expected_ordinal = 0;
        for (size_t callback_index = 0;
             callback_index + 1 < candidate.callbacks.size();
             ++callback_index) {
            TORCH_CHECK(
                candidate.callbacks[callback_index]
                        .producer_yields.size() <=
                    std::numeric_limits<size_t>::max() - expected_ordinal,
                "callback producer-yield trace ordinal overflow");
            expected_ordinal +=
                candidate.callbacks[callback_index]
                    .producer_yields.size();
        }
        size_t previous_yield_end = callback.graph_node_begin;
        for (const auto& yield : callback.producer_yields) {
            TORCH_CHECK(
                yield.ordinal == expected_ordinal++ &&
                    yield.graph_node_begin >= previous_yield_end &&
                    yield.graph_node_begin < yield.graph_node_end &&
                    yield.graph_node_end <= digest.end,
                "callback producer-yield scopes must be globally ordered, "
                "nonempty, non-overlapping, and contained by their owner "
                "callback");
            previous_yield_end = yield.graph_node_end;
        }
    } else {
        TORCH_CHECK(
            !candidate.producer_yield_open &&
                callback.producer_yields.empty(),
            "producer-yield scope has no matching resolved capability");
    }
    std::sort(
        callback.accesses.begin(), callback.accesses.end(),
        [](const FusedModelBase::Impl::BuildTraceAccessRecord& lhs,
           const FusedModelBase::Impl::BuildTraceAccessRecord& rhs) {
            return std::tie(
                       lhs.allocation_ordinal, lhs.declaration_ordinal,
                       lhs.alias_root_ordinal, lhs.storage, lhs.scope,
                       lhs.layer, lhs.accessor, lhs.root_offset,
                       lhs.protected_bytes) <
                   std::tie(
                       rhs.allocation_ordinal, rhs.declaration_ordinal,
                       rhs.alias_root_ordinal, rhs.storage, rhs.scope,
                       rhs.layer, rhs.accessor, rhs.root_offset,
                       rhs.protected_bytes);
        });
    callback.graph_node_end = digest.end;
    callback.graph_node_kind_mask = digest.kind_mask;
    callback.graph_topology_hash = digest.topology_hash;
    callback.legacy_dma_count = digest.legacy_dma_count;
    callback.semantic_dmas.reserve(digest.semantic_dmas.size());
    for (const auto& dma : digest.semantic_dmas) {
        callback.semantic_dmas.push_back({
            dma.relative_node_index,
            dma.endpoint_id,
            dma.variant,
            dma.src_addr,
            dma.dst_addr,
            dma.bytes,
            dma.channel,
            dma.live_offset,
            dma.live_base,
            dma.owner_generation,
            dma.expected_owner_generation,
            dma.canonical_member_emission,
            dma.spm_peer_id,
        });
    }
    if (candidate.profile->spm_peer_capability_ !=
        SpmFmbSpmPeerCapability::Unsealed) {
        size_t occurrence_cursor = 0;
        for (const auto& peer : callback.spm_peers) {
            TORCH_CHECK(
                peer.peer_id != 0 && peer.endpoint_id != 0 &&
                    peer.expected_occurrence_count > 0,
                "schema-v13 typed SPM peer STOP: callback contains an "
                "empty peer identity");
            for (uint8_t channel = 0;
                 channel < peer.expected_occurrence_count; ++channel) {
                TORCH_CHECK(
                    occurrence_cursor < callback.semantic_dmas.size(),
                    "schema-v13 typed SPM peer STOP: callback omitted a "
                    "peer DMA occurrence");
                const auto& dma =
                    callback.semantic_dmas[occurrence_cursor++];
                const uint64_t expected_spm =
                    expected_cpu_dense_spm_peer_address(peer, channel);
                TORCH_CHECK(
                    dma.endpoint_id == peer.endpoint_id &&
                        dma.spm_peer_id == peer.peer_id &&
                        dma.canonical_member_emission &&
                        dma.bytes == peer.peer_slice_count &&
                        dma.channel == channel &&
                        (peer.role == SpmFmbDmaEndpointRole::Ingress
                             ? dma.dst_addr == expected_spm
                             : dma.src_addr == expected_spm),
                    "schema-v13 typed SPM peer STOP: callback Graph "
                    "occurrence/address differs from its allocation-derived "
                    "peer");
            }
        }
        TORCH_CHECK(
            occurrence_cursor == callback.semantic_dmas.size() &&
                (callback.semantic_dmas.empty() ==
                 callback.spm_peers.empty()),
            "schema-v13 typed SPM peer STOP: callback peer occurrence "
            "universe is not covered exactly once");
    }
    candidate.callback_open = false;
    g_fmb_build_trace_callback_begin.reset();
    ++candidate.next_step;
}

namespace {

template <typename Fn, typename EndFn>
void invoke_trace_callback(
    FusedModelBase::Impl& state,
    const SpmFmbResolvedExecutionStep& expected,
    const FmbExecutionStepView& step,
    Fn&& fn,
    EndFn&& end_fn) {
    begin_trace_callback(state, expected, step);
    try {
        std::forward<Fn>(fn)();
        std::forward<EndFn>(end_fn)();
    } catch (...) {
        state.pipeline_build_trace_.callback_open = false;
        state.pipeline_build_trace_.producer_yield_open = false;
        g_fmb_build_trace_callback_begin.reset();
        g_fmb_post_fn_yield_begin.reset();
        throw;
    }
}

}  // namespace

SpmFmbResolvedExecutionProfile
FusedModelBase::resolve_spm_pipeline_execution_profile_for_cpu_contract(
    const SpmFmbResolvedProfileRequest& request) {
    return resolve_spm_pipeline_execution_profile_impl(
        request, /*require_cpu_dry=*/true);
}

SpmFmbResolvedExecutionProfile
FusedModelBase::resolve_spm_pipeline_execution_profile_for_build_trace(
    const SpmFmbResolvedProfileRequest& request) {
    return resolve_spm_pipeline_execution_profile_impl(
        request, /*require_cpu_dry=*/false);
}

SpmFmbResolvedExecutionProfile
FusedModelBase::resolve_spm_pipeline_execution_profile_impl(
    const SpmFmbResolvedProfileRequest& request,
    bool require_cpu_dry) {
    const CachedLayoutSnapshot& snapshot =
        pimpl_->pipeline_manifest_snapshot_;
    TORCH_CHECK(snapshot.valid,
                "schema-v7 resolved profile requires a prepared owned "
                "allocation snapshot");
    TORCH_CHECK(snapshot.cpu_dry == require_cpu_dry,
                require_cpu_dry
                    ? "schema-v7 resolved profile accepts only a CPU-contract "
                      "allocation snapshot"
                    : "schema-v8 live resolved profile requires a real "
                      "allocation snapshot");
    TORCH_CHECK(request.version != 0,
                "schema-v7 resolved profile requires a nonzero request "
                "version");
    TORCH_CHECK(spm_fmb_traversal_capability() ==
                    SpmFmbTraversalCapability::CanonicalTraversal,
                "schema-v7 resolved profile STOP: model has not opted into "
                "CanonicalTraversal; hidden layer traversal is unsealed");
    const SpmFmbPreloadCapability preload_capability =
        spm_fmb_preload_capability();
    TORCH_CHECK(
        preload_capability == SpmFmbPreloadCapability::Unsealed ||
            preload_capability ==
                SpmFmbPreloadCapability::PersistentOutsideResolvedWindow,
        "resolved preload profile STOP: model returned an invalid "
        "capability");
    const SpmFmbPostFnYieldCapability post_fn_yield_capability =
        spm_fmb_post_fn_yield_capability();
    TORCH_CHECK(
        post_fn_yield_capability ==
                SpmFmbPostFnYieldCapability::Unsealed ||
            post_fn_yield_capability ==
                SpmFmbPostFnYieldCapability::CanonicalDenseReplicatedFp16,
        "resolved post_fn yield profile STOP: model returned an invalid "
        "capability");
    const uint64_t post_fn_yield_policy_fingerprint =
        spm_fmb_post_fn_yield_policy_fingerprint();
    TORCH_CHECK(
        (post_fn_yield_capability ==
             SpmFmbPostFnYieldCapability::Unsealed &&
         post_fn_yield_policy_fingerprint == 0) ||
            (post_fn_yield_capability ==
                 SpmFmbPostFnYieldCapability::
                     CanonicalDenseReplicatedFp16 &&
             post_fn_yield_policy_fingerprint != 0),
        "resolved post_fn yield profile requires a nonzero fingerprint "
        "only for an opted-in capability");
    const SpmFmbLayerProducerYieldCapability
        layer_producer_yield_capability =
            spm_fmb_layer_producer_yield_capability();
    TORCH_CHECK(
        layer_producer_yield_capability ==
                SpmFmbLayerProducerYieldCapability::Unsealed ||
            layer_producer_yield_capability ==
                SpmFmbLayerProducerYieldCapability::
                    CanonicalDenseReplicatedFp16,
        "resolved callback-yield profile STOP: model returned an invalid "
        "layer producer capability");
    const uint64_t layer_producer_yield_policy_fingerprint =
        spm_fmb_layer_producer_yield_policy_fingerprint();
    TORCH_CHECK(
        (layer_producer_yield_capability ==
             SpmFmbLayerProducerYieldCapability::Unsealed &&
         layer_producer_yield_policy_fingerprint == 0) ||
            (layer_producer_yield_capability ==
                 SpmFmbLayerProducerYieldCapability::
                     CanonicalDenseReplicatedFp16 &&
             layer_producer_yield_policy_fingerprint != 0),
        "resolved callback-yield profile requires a nonzero fingerprint "
        "only for an opted-in layer producer capability");
    TORCH_CHECK(
        post_fn_yield_capability ==
                SpmFmbPostFnYieldCapability::Unsealed ||
            layer_producer_yield_capability ==
                SpmFmbLayerProducerYieldCapability::Unsealed,
        "resolved callback-yield profile cannot mix the legacy exact "
        "PostFn capability with sparse callback-window authority");
    const SpmFmbActiveGroupCapability active_group_capability =
        spm_fmb_active_group_capability();
    TORCH_CHECK(
        active_group_capability == SpmFmbActiveGroupCapability::Unsealed ||
            active_group_capability ==
                SpmFmbActiveGroupCapability::CanonicalActiveFirstChunkRun,
        "schema-v9 active-group profile STOP: model returned an invalid "
        "active-group capability");
    const SpmFmbDenseDdrMemberCapability dense_ddr_member_capability =
        spm_fmb_dense_ddr_member_capability();
    TORCH_CHECK(
        dense_ddr_member_capability ==
                SpmFmbDenseDdrMemberCapability::Unsealed ||
            dense_ddr_member_capability ==
                SpmFmbDenseDdrMemberCapability::
                    IndependentLocalSequentialDdrPingPong,
        "schema-v10 dense-DDR member profile STOP: model returned an "
        "invalid capability");
    TORCH_CHECK(
        dense_ddr_member_capability ==
                SpmFmbDenseDdrMemberCapability::Unsealed ||
            active_group_capability ==
                SpmFmbActiveGroupCapability::CanonicalActiveFirstChunkRun,
        "schema-v10 dense-DDR member profile requires the canonical "
        "active-first chunk-run capability");
    const SpmFmbDmaEndpointCapability dma_endpoint_capability =
        spm_fmb_dma_endpoint_capability();
    TORCH_CHECK(
        dma_endpoint_capability ==
                SpmFmbDmaEndpointCapability::Unsealed ||
            dma_endpoint_capability ==
                SpmFmbDmaEndpointCapability::DenseDdrMemberRoleBindings,
        "schema-v11 DMA endpoint profile STOP: model returned an invalid "
        "capability");
    TORCH_CHECK(
        dma_endpoint_capability ==
                SpmFmbDmaEndpointCapability::Unsealed ||
            dense_ddr_member_capability ==
                SpmFmbDenseDdrMemberCapability::
                    IndependentLocalSequentialDdrPingPong,
        "schema-v11 DMA endpoint profile requires the schema-v10 dense-DDR "
        "member capability");
    const SpmFmbRuntimeMemberCapability runtime_member_capability =
        spm_fmb_runtime_member_capability();
    TORCH_CHECK(
        runtime_member_capability ==
                SpmFmbRuntimeMemberCapability::Unsealed ||
            runtime_member_capability ==
                SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs,
        "schema-v12 runtime member profile STOP: model returned an invalid "
        "capability");
    TORCH_CHECK(
        runtime_member_capability ==
                SpmFmbRuntimeMemberCapability::Unsealed ||
            dma_endpoint_capability ==
                SpmFmbDmaEndpointCapability::DenseDdrMemberRoleBindings,
        "schema-v12 runtime member profile requires schema-v11 DMA endpoint "
        "bindings");
    const SpmFmbSpmPeerCapability spm_peer_capability =
        spm_fmb_spm_peer_capability();
    TORCH_CHECK(
        spm_peer_capability == SpmFmbSpmPeerCapability::Unsealed ||
            spm_peer_capability ==
                SpmFmbSpmPeerCapability::TypedDensePackedOrShared,
        "schema-v13 typed SPM peer profile STOP: model returned an invalid "
        "capability");
    TORCH_CHECK(
        spm_peer_capability == SpmFmbSpmPeerCapability::Unsealed ||
            runtime_member_capability ==
                SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs,
        "schema-v13 typed SPM peer profile requires the complete schema-v12 "
        "runtime member contract");
    TORCH_CHECK(
        spm_peer_capability == SpmFmbSpmPeerCapability::Unsealed ||
            require_cpu_dry,
        "schema-v13 typed SPM peer profile is CPU-contract only");
    const uint64_t spm_peer_policy_fingerprint =
        spm_fmb_spm_peer_policy_fingerprint();
    TORCH_CHECK(
        (spm_peer_capability == SpmFmbSpmPeerCapability::Unsealed &&
         spm_peer_policy_fingerprint == 0) ||
            (spm_peer_capability ==
                 SpmFmbSpmPeerCapability::TypedDensePackedOrShared &&
             spm_peer_policy_fingerprint != 0),
        "schema-v13 typed SPM peer profile requires one nonzero, "
        "capability-scoped buffer-selection policy fingerprint");
    TORCH_CHECK(
        (post_fn_yield_capability ==
             SpmFmbPostFnYieldCapability::Unsealed &&
         layer_producer_yield_capability ==
             SpmFmbLayerProducerYieldCapability::Unsealed) ||
            (active_group_capability ==
                 SpmFmbActiveGroupCapability::Unsealed &&
             dense_ddr_member_capability ==
                 SpmFmbDenseDdrMemberCapability::Unsealed &&
             dma_endpoint_capability ==
                 SpmFmbDmaEndpointCapability::Unsealed &&
             runtime_member_capability ==
                 SpmFmbRuntimeMemberCapability::Unsealed &&
             spm_peer_capability ==
                 SpmFmbSpmPeerCapability::Unsealed),
        "resolved post_fn yield profile initially cannot compose with "
        "schema-v9 through schema-v13 member/DMA capabilities");

    const ModelStaticConfig static_cfg = static_config();
    TORCH_CHECK(static_cfg.num_layers > 0 &&
                    static_cfg.num_layers == pimpl_->num_layers_,
                "schema-v7 resolved profile: static layer count mismatch");
    TORCH_CHECK(static_cfg.preload_fn == nullptr,
                "schema-v7 resolved profile STOP: preload_fn is an opaque "
                "model callback");
    TORCH_CHECK(static_cfg.pre_layers_fn == nullptr &&
                    static_cfg.post_layers_fn == nullptr,
                "schema-v7 resolved profile STOP: pre/post layer callbacks "
                "are not represented by the canonical layer traversal");
    if (post_fn_yield_capability ==
            SpmFmbPostFnYieldCapability::Unsealed &&
        layer_producer_yield_capability ==
            SpmFmbLayerProducerYieldCapability::Unsealed) {
        TORCH_CHECK(static_cfg.post_fn == nullptr,
                    "schema-v7 resolved profile STOP: post_fn scope and "
                    "hidden internal traversal are unsealed");
    } else if (post_fn_yield_capability !=
               SpmFmbPostFnYieldCapability::Unsealed) {
        TORCH_CHECK(
            static_cfg.post_fn != nullptr &&
                static_cfg.post_output_shape.empty() &&
                !static_cfg.fast_replay_bake_post_fn,
            "resolved post_fn yield profile requires exactly one framework "
            "post_fn with no DDR post_output_shape or baked fast-replay "
            "downgrade");
    } else {
        TORCH_CHECK(
            static_cfg.post_output_shape.empty() &&
                !static_cfg.fast_replay_skip_layer_loop &&
                !static_cfg.fast_replay_skip_full_body &&
                !static_cfg.fast_replay_bake_post_fn,
            "resolved callback-yield profile rejects layer/full fast "
            "replay skips, baked PostFn, and DDR post outputs");
    }
    for (const OwnedPipelineDecl& decl : snapshot.decls) {
        if (decl.preload_present) {
            TORCH_CHECK(
                preload_capability != SpmFmbPreloadCapability::Unsealed,
                "schema-v7 resolved profile STOP: BufferDecl preload "
                "callback is an opaque model callback for '",
                decl.name, "'");
            TORCH_CHECK(
                decl.storage == StorageClass::Persistent ||
                    decl.storage == StorageClass::PersistentPerLayer,
                "resolved preload profile STOP: only Persistent or "
                "PersistentPerLayer callbacks may execute outside the "
                "resolved window; got temporary '", decl.name, "'");
        }
        TORCH_CHECK(
            !((decl.storage == StorageClass::Temp ||
               decl.storage == StorageClass::TempPerLayer) &&
              decl.scope == BufferScope::OutsideLayerLoop),
            "schema-v7 resolved profile STOP: OutsideLayerLoop temporary '",
            decl.name, "' has no canonical callback ownership");
    }

    const int64_t compute_total =
        validate_fmb_resolved_chunks(request.chunks, "compute");
    const LayoutContext& prepared_layout = snapshot.layout_context;
    TORCH_CHECK(!prepared_layout.use_attn_mask,
                "schema-v7 resolved profile STOP: explicit attention-mask "
                "layout is not supported in the initial canonical profile");
    TORCH_CHECK(prepared_layout.num_layers == static_cfg.num_layers,
                "schema-v7 resolved profile: prepared LayoutContext layer "
                "count mismatch");
    TORCH_CHECK(request.chunks.front().len == prepared_layout.chunk_size,
                "schema-v7 resolved profile: compute chunks do not match the "
                "allocation-time chunk_size");
    const int64_t compute_position =
        request.chunks.front().kv_seq_len - request.chunks.front().len;
    TORCH_CHECK(compute_position <=
                    std::numeric_limits<int64_t>::max() - compute_total &&
                    compute_position + compute_total ==
                        prepared_layout.max_kv_seq_len,
                "schema-v7 resolved profile: chunks do not match "
                "allocation-time max_kv_seq_len");
    ChunkPlan chunk_plan;
    chunk_plan.chunk_size = request.chunks.front().len;
    chunk_plan.num_chunks = static_cast<int64_t>(request.chunks.size());
    ModelDynamicConfig dynamic_cfg;
    {
        const InferenceContext saved_context = pimpl_->ctx_;
        auto restore_context = c10::make_scope_exit(
            [&] { pimpl_->ctx_ = saved_context; });
        pimpl_->ctx_.seq_len = compute_total;
        pimpl_->ctx_.position = compute_position;
        pimpl_->ctx_.batch_size = prepared_layout.batch_size;
        pimpl_->ctx_.is_causal = prepared_layout.is_causal;
        pimpl_->ctx_.attention_mask = std::nullopt;
        pimpl_->ctx_.input_in_spm = false;
        pimpl_->ctx_.output_to_spm = false;
        dynamic_cfg = dynamic_config(chunk_plan);
    }
    InterLayerIO io = dynamic_cfg.inter_layer_io;
    if (io == InterLayerIO::AUTO) {
        io = request.chunks.size() == 1
            ? InterLayerIO::SPM_RESIDENT : InterLayerIO::DDR_PINGPONG;
    }
    TORCH_CHECK(io == InterLayerIO::SPM_RESIDENT ||
                    io == InterLayerIO::DDR_PINGPONG,
                "schema-v7 resolved profile: dynamic_config returned an "
                "invalid inter-layer I/O mode");

    std::vector<ChunkInfo> no_kv_chunks;
    const std::vector<ChunkInfo>* kv_chunks = &no_kv_chunks;
    if (dynamic_cfg.chunk_mode == ChunkMode::SEQUENTIAL) {
        TORCH_CHECK(request.kv_insert_chunks.empty(),
                    "schema-v7 resolved profile: SEQUENTIAL traversal must "
                    "not provide KV-insert chunks");
        TORCH_CHECK(prepared_layout.kv_insert_chunk_size == 0,
                    "schema-v7 resolved profile: SEQUENTIAL traversal does "
                    "not match the allocation-time KV chunk layout");
    } else {
        TORCH_CHECK(dynamic_cfg.chunk_mode == ChunkMode::KV_FIRST &&
                        static_cfg.kv_first_fn != nullptr,
                    "schema-v7 resolved profile: KV_FIRST requires the "
                    "framework-owned kv_first callback");
        const int64_t kv_total = validate_fmb_resolved_chunks(
            request.kv_insert_chunks, "KV-insert");
        TORCH_CHECK(kv_total == compute_total,
                    "schema-v7 resolved profile: compute and KV-insert chunk "
                    "vectors must cover the same sequence length");
        const int64_t kv_position = request.kv_insert_chunks.front().kv_seq_len -
            request.kv_insert_chunks.front().len;
        TORCH_CHECK(compute_position == kv_position,
                    "schema-v7 resolved profile: compute and KV-insert chunks "
                    "must use the same position base");
        TORCH_CHECK(request.kv_insert_chunks.front().len ==
                        prepared_layout.effective_kv_cs(),
                    "schema-v7 resolved profile: KV-insert chunks do not "
                    "match allocation-time effective_kv_cs");
        const int64_t canonical_kv_insert_chunk_size =
            request.kv_insert_chunks.front().len !=
                    request.chunks.front().len
            ? request.kv_insert_chunks.front().len : 0;
        TORCH_CHECK(prepared_layout.kv_insert_chunk_size ==
                        canonical_kv_insert_chunk_size,
                    "schema-v7 resolved profile: KV-insert chunks do not "
                    "match run_all_layers' allocation-time canonical "
                    "LayoutContext");
        kv_chunks = &request.kv_insert_chunks;
    }
    TORCH_CHECK(!dynamic_cfg.chunk_outer_within_group ||
                    (dynamic_cfg.chunk_mode == ChunkMode::SEQUENTIAL &&
                     io == InterLayerIO::SPM_RESIDENT),
                "schema-v7 resolved profile: invalid chunk-outer policy");

    int64_t cross_batch = static_cfg.cross_layer_batch_size > 0
        ? static_cfg.cross_layer_batch_size : static_cfg.num_layers;
    if (cross_batch <= 0) cross_batch = static_cfg.num_layers;
    const int64_t build_body_iterations = static_cfg.body_iterations > 0
        ? static_cfg.body_iterations : 1;
    TORCH_CHECK(build_body_iterations <=
                    std::numeric_limits<uint32_t>::max(),
                "schema-v7 resolved profile: body iteration count exceeds "
                "uint32");
    const FmbExecutionTraversalSpec traversal{
        static_cfg.num_layers,
        cross_batch,
        dynamic_cfg.chunk_mode,
        io,
        dynamic_cfg.chunk_outer_within_group,
        static_cfg.kv_first_fn != nullptr,
        &request.chunks,
        kv_chunks,
    };

    std::map<std::pair<std::string, int>, const OwnedPipelineDecl*>
        declaration_by_instance;
    std::map<std::string, const OwnedPipelineDecl*> declaration_by_name;
    for (const OwnedPipelineDecl& decl : snapshot.decls) {
        declaration_by_name.emplace(decl.name, &decl);
    }
    for (const PipelineAllocationEntry& allocation :
         snapshot.temporary_allocations) {
        declaration_by_instance.emplace(
            std::make_pair(allocation.name, allocation.layer),
            declaration_by_name.at(allocation.name));
    }

    std::vector<SpmFmbResolvedExecutionStep> steps;
    uint64_t global_cursor = 0;
    uint32_t key = 1;
    for (int64_t body = 0; body < build_body_iterations; ++body) {
        for_each_fmb_execution_step(
            traversal, static_cast<uint32_t>(body),
            [&](const FmbExecutionStepView& step) {
                int local_begin = std::numeric_limits<int>::max();
                int local_end = std::numeric_limits<int>::min();
                size_t matched_allocations = 0;
                for (const auto& item : declaration_by_instance) {
                    const OwnedPipelineDecl& decl = *item.second;
                    if ((step.scope_mask & spm_fmb_scope_bit(decl.scope)) == 0) {
                        continue;
                    }
                    if (decl.storage == StorageClass::TempPerLayer &&
                        item.first.second != step.layer) {
                        continue;
                    }
                    local_begin = std::min(local_begin, decl.phase_start);
                    local_end = std::max(local_end, decl.phase_end);
                    ++matched_allocations;
                }
                TORCH_CHECK(
                    matched_allocations > 0,
                    "schema-v7 resolved profile: framework execution step ",
                    step.traversal_ordinal,
                    " maps no temporary allocation");
                const uint64_t width = static_cast<uint64_t>(
                    local_end - local_begin);
                TORCH_CHECK(global_cursor <=
                                std::numeric_limits<uint32_t>::max() &&
                                width <=
                                std::numeric_limits<uint32_t>::max() -
                                    global_cursor,
                            "schema-v7 resolved profile: global event "
                            "namespace overflow");
                TORCH_CHECK(key != 0,
                            "schema-v7 resolved profile: step key overflow");
                const ChunkInfo& chunk = *step.chunk;
                steps.emplace_back(
                    key++, step.kind, step.body_id, step.group_id, step.layer,
                    step.traversal_ordinal, true, chunk.idx, chunk.offset,
                    chunk.len, chunk.kv_seq_len, step.scope_mask,
                    step.updates_io_context, step.input_in_spm,
                    step.output_to_spm, local_begin, local_end,
                    SpmPipelineEvent(static_cast<uint32_t>(global_cursor)));
                global_cursor += width + 1;
            });
    }
    if (static_cfg.post_fn != nullptr &&
        (post_fn_yield_capability !=
             SpmFmbPostFnYieldCapability::Unsealed ||
         layer_producer_yield_capability !=
             SpmFmbLayerProducerYieldCapability::Unsealed)) {
        int local_begin = std::numeric_limits<int>::max();
        int local_end = std::numeric_limits<int>::min();
        size_t matched_allocations = 0;
        constexpr SpmFmbScopeMask kPostFnScopes =
            spm_fmb_scope_bit(BufferScope::LayerWide) |
            spm_fmb_scope_bit(BufferScope::Compute);
        for (const auto& item : declaration_by_instance) {
            const OwnedPipelineDecl& decl = *item.second;
            if (item.first.second != -1 ||
                (kPostFnScopes & spm_fmb_scope_bit(decl.scope)) == 0) {
                continue;
            }
            local_begin = std::min(local_begin, decl.phase_start);
            local_end = std::max(local_end, decl.phase_end);
            ++matched_allocations;
        }
        TORCH_CHECK(
            matched_allocations > 0,
            "resolved post_fn yield profile maps no flat LayerWide/Compute "
            "temporary allocation");
        const uint64_t width = static_cast<uint64_t>(
            local_end - local_begin);
        TORCH_CHECK(
            global_cursor <= std::numeric_limits<uint32_t>::max() &&
                width <= std::numeric_limits<uint32_t>::max() -
                             global_cursor,
            "resolved post_fn yield profile global event namespace "
            "overflows uint32");
        TORCH_CHECK(key != 0,
                    "resolved post_fn yield profile step key overflow");
        TORCH_CHECK(
            steps.size() <= std::numeric_limits<uint32_t>::max(),
            "resolved post_fn yield profile traversal ordinal overflows "
            "uint32");
        steps.emplace_back(
            key++, SpmFmbOccurrenceKind::PostFn,
            static_cast<uint32_t>(build_body_iterations),
            /*group_id=*/0, /*layer=*/-1,
            static_cast<uint32_t>(steps.size()),
            /*has_chunk=*/false, /*chunk_index=*/-1,
            /*chunk_offset=*/0, /*chunk_len=*/0,
            /*chunk_kv_seq_len=*/0, kPostFnScopes,
            /*updates_io_context=*/false, /*input_in_spm=*/false,
            /*output_to_spm=*/false, local_begin, local_end,
            SpmPipelineEvent(static_cast<uint32_t>(global_cursor)));
        global_cursor += width + 1;
    }
    TORCH_CHECK(!steps.empty(),
                "schema-v7 resolved profile: canonical traversal produced no "
                "execution steps");

    uint64_t profile_hash = fmb_hash_u64(kFmbFnvOffset, 9);
    profile_hash = fmb_hash_u64(profile_hash, request.version);
    // Preserve the frozen schema-v7 Unsealed hash byte-for-byte.  Only an
    // explicit schema-v9 opt-in extends the signature/profile identity.
    if (active_group_capability != SpmFmbActiveGroupCapability::Unsealed) {
        profile_hash = fmb_hash_u64(
            profile_hash, static_cast<uint8_t>(active_group_capability));
    }
    profile_hash = fmb_hash_u64(profile_hash, build_body_iterations);
    profile_hash = fmb_hash_u64(profile_hash, static_cfg.num_layers);
    profile_hash = fmb_hash_u64(profile_hash, cross_batch);
    profile_hash = fmb_hash_u64(
        profile_hash, static_cast<uint8_t>(dynamic_cfg.chunk_mode));
    profile_hash = fmb_hash_u64(profile_hash, static_cast<uint8_t>(io));
    profile_hash = fmb_hash_u64(profile_hash,
                                dynamic_cfg.chunk_outer_within_group);
    profile_hash = fmb_hash_u64(profile_hash, steps.size());
    for (const SpmFmbResolvedExecutionStep& step : steps) {
        profile_hash = fmb_hash_u64(profile_hash, step.key);
        profile_hash = fmb_hash_u64(
            profile_hash, static_cast<uint8_t>(step.kind));
        profile_hash = fmb_hash_u64(profile_hash, step.body_id);
        profile_hash = fmb_hash_u64(profile_hash, step.group_id);
        profile_hash = fmb_hash_u64(
            profile_hash, static_cast<uint64_t>(step.layer));
        profile_hash = fmb_hash_u64(profile_hash,
                                    step.traversal_ordinal);
        profile_hash = fmb_hash_u64(profile_hash, step.has_chunk);
        profile_hash = fmb_hash_u64(
            profile_hash, static_cast<uint64_t>(step.chunk_index));
        profile_hash = fmb_hash_u64(
            profile_hash, static_cast<uint64_t>(step.chunk_offset));
        profile_hash = fmb_hash_u64(
            profile_hash, static_cast<uint64_t>(step.chunk_len));
        profile_hash = fmb_hash_u64(
            profile_hash, static_cast<uint64_t>(step.chunk_kv_seq_len));
        profile_hash = fmb_hash_u64(profile_hash, step.scope_mask);
        profile_hash = fmb_hash_u64(profile_hash,
                                    step.updates_io_context);
        profile_hash = fmb_hash_u64(profile_hash, step.input_in_spm);
        profile_hash = fmb_hash_u64(profile_hash, step.output_to_spm);
        profile_hash = fmb_hash_u64(
            profile_hash, static_cast<uint64_t>(step.local_phase_begin));
        profile_hash = fmb_hash_u64(
            profile_hash, static_cast<uint64_t>(step.local_phase_end));
        profile_hash = fmb_hash_u64(
            profile_hash, step.global_origin.value());
    }
    // Freeze schema-v7/v9 byte streams: only an explicit v10 opt-in appends a
    // separately domain-tagged policy identity after the complete old profile.
    if (dense_ddr_member_capability !=
        SpmFmbDenseDdrMemberCapability::Unsealed) {
        uint64_t v10_policy_hash = fmb_hash_u64(kFmbFnvOffset, 14);
        v10_policy_hash = fmb_hash_u64(
            v10_policy_hash,
            static_cast<uint8_t>(dense_ddr_member_capability));
        profile_hash = fmb_hash_u64(
            profile_hash, fmb_nonzero_hash(v10_policy_hash));
    }
    if (dma_endpoint_capability !=
        SpmFmbDmaEndpointCapability::Unsealed) {
        uint64_t v11_policy_hash = fmb_hash_u64(kFmbFnvOffset, 17);
        v11_policy_hash = fmb_hash_u64(
            v11_policy_hash,
            static_cast<uint8_t>(dma_endpoint_capability));
        profile_hash = fmb_hash_u64(
            profile_hash, fmb_nonzero_hash(v11_policy_hash));
    }
    // Append the stronger current-callback authority as an independent
    // domain.  Every v7-v11 byte stream remains frozen when it is Unsealed.
    if (runtime_member_capability !=
        SpmFmbRuntimeMemberCapability::Unsealed) {
        uint64_t v12_policy_hash = fmb_hash_u64(kFmbFnvOffset, 21);
        v12_policy_hash = fmb_hash_u64(
            v12_policy_hash,
            static_cast<uint8_t>(runtime_member_capability));
        profile_hash = fmb_hash_u64(
            profile_hash, fmb_nonzero_hash(v12_policy_hash));
    }
    // Append schema-v13 only after the complete v12 byte stream.  Domain 22
    // is independent, so every Unsealed v7-v12 profile remains byte-identical.
    if (spm_peer_capability != SpmFmbSpmPeerCapability::Unsealed) {
        uint64_t v13_policy_hash = fmb_hash_u64(kFmbFnvOffset, 22);
        v13_policy_hash = fmb_hash_u64(
            v13_policy_hash, static_cast<uint8_t>(spm_peer_capability));
        v13_policy_hash = fmb_hash_u64(
            v13_policy_hash, spm_peer_policy_fingerprint);
        profile_hash = fmb_hash_u64(
            profile_hash, fmb_nonzero_hash(v13_policy_hash));
    }
    if (preload_capability != SpmFmbPreloadCapability::Unsealed) {
        uint64_t preload_policy_hash = fmb_hash_u64(kFmbFnvOffset, 25);
        preload_policy_hash = fmb_hash_u64(
            preload_policy_hash,
            static_cast<uint8_t>(preload_capability));
        preload_policy_hash = fmb_hash_u64(
            preload_policy_hash, snapshot.declaration_hash);
        preload_policy_hash = fmb_hash_u64(
            preload_policy_hash, snapshot.persistent_hash);
        profile_hash = fmb_hash_u64(
            profile_hash, fmb_nonzero_hash(preload_policy_hash));
    }
    if (post_fn_yield_capability !=
        SpmFmbPostFnYieldCapability::Unsealed) {
        constexpr SpmFmbScopeMask kPostFnScopes =
            spm_fmb_scope_bit(BufferScope::LayerWide) |
            spm_fmb_scope_bit(BufferScope::Compute);
        uint64_t post_fn_policy_hash = fmb_hash_u64(kFmbFnvOffset, 27);
        post_fn_policy_hash = fmb_hash_u64(
            post_fn_policy_hash,
            static_cast<uint8_t>(post_fn_yield_capability));
        post_fn_policy_hash = fmb_hash_u64(
            post_fn_policy_hash, post_fn_yield_policy_fingerprint);
        post_fn_policy_hash = fmb_hash_u64(
            post_fn_policy_hash, static_cfg.fast_replay_skip_layer_loop);
        post_fn_policy_hash = fmb_hash_u64(
            post_fn_policy_hash, static_cfg.fast_replay_bake_post_fn);
        post_fn_policy_hash = fmb_hash_u64(
            post_fn_policy_hash, kPostFnScopes);
        profile_hash = fmb_hash_u64(
            profile_hash, fmb_nonzero_hash(post_fn_policy_hash));
    }
    if (layer_producer_yield_capability !=
        SpmFmbLayerProducerYieldCapability::Unsealed) {
        uint64_t callback_yield_policy_hash =
            fmb_hash_u64(kFmbFnvOffset, 30);
        callback_yield_policy_hash = fmb_hash_u64(
            callback_yield_policy_hash,
            static_cast<uint8_t>(layer_producer_yield_capability));
        callback_yield_policy_hash = fmb_hash_u64(
            callback_yield_policy_hash,
            layer_producer_yield_policy_fingerprint);
        callback_yield_policy_hash = fmb_hash_u64(
            callback_yield_policy_hash,
            static_cfg.post_fn != nullptr);
        profile_hash = fmb_hash_u64(
            profile_hash,
            fmb_nonzero_hash(callback_yield_policy_hash));
    }
    profile_hash = fmb_nonzero_hash(profile_hash);
    return SpmFmbResolvedExecutionProfile(
        request.version, static_cast<uint32_t>(build_body_iterations), steps,
        profile_hash, snapshot.layout_hash, snapshot.allocation_hash,
        preload_capability,
        post_fn_yield_capability,
        post_fn_yield_policy_fingerprint,
        layer_producer_yield_capability,
        layer_producer_yield_policy_fingerprint,
        active_group_capability,
        dense_ddr_member_capability,
        dma_endpoint_capability,
        runtime_member_capability,
        spm_peer_capability,
        spm_peer_policy_fingerprint,
        static_cast<uint8_t>(dynamic_cfg.chunk_mode),
        static_cast<uint8_t>(io),
        dynamic_cfg.chunk_outer_within_group,
        cross_batch,
        pimpl_->hidden_size_,
        static_cfg.num_layers,
        snapshot.cpu_dry,
        pimpl_->pipeline_manifest_generation_,
        *pimpl_->pipeline_manifest_generation_);
}

void FusedModelBase::begin_spm_pipeline_build_trace(
    const SpmFmbResolvedExecutionProfile& profile) {
    begin_spm_pipeline_build_trace_impl(
        profile, /*composite=*/false, /*composite_identity=*/0,
        /*composite_occurrence_ordinal=*/0,
        /*composite_producer=*/false,
        /*composite_producer_local_ordinal=*/0,
        SpmFmbCompositeOccurrenceCapability::Unsealed,
        /*composite_policy_fingerprint=*/0,
        /*composite_preload_follower=*/false);
}

void FusedModelBase::begin_spm_pipeline_build_trace_impl(
    const SpmFmbResolvedExecutionProfile& profile,
    bool composite,
    uint64_t composite_identity,
    size_t composite_occurrence_ordinal,
    bool composite_producer,
    size_t composite_producer_local_ordinal,
    SpmFmbCompositeOccurrenceCapability composite_capability,
    uint64_t composite_policy_fingerprint,
    bool composite_preload_follower) {
    TORCH_CHECK(!pimpl_->pipeline_build_trace_.armed &&
                    g_fmb_build_trace_owner == nullptr &&
                    !g_fmb_build_trace_callback_begin.has_value() &&
                    !g_fmb_post_fn_yield_begin.has_value(),
                "schema-v8 BUILD trace: another trace is already armed on "
                "this thread");
    const CachedLayoutSnapshot& snapshot =
        pimpl_->pipeline_manifest_snapshot_;
    TORCH_CHECK(snapshot.valid,
                "schema-v8 BUILD trace requires a prepared allocation "
                "snapshot");
    const std::shared_ptr<const uint64_t> profile_owner =
        profile.owner_generation_.lock();
    TORCH_CHECK(profile_owner != nullptr &&
                    profile_owner.get() ==
                        pimpl_->pipeline_manifest_generation_.get() &&
                    *profile_owner == profile.expected_owner_generation_,
                "schema-v8 BUILD trace: resolved profile owner is stale");
    TORCH_CHECK(profile.profile_hash_ != 0 && !profile.steps_.empty() &&
                    profile.expected_layout_hash_ == snapshot.layout_hash &&
                    profile.expected_allocation_hash_ ==
                        snapshot.allocation_hash &&
                    profile.preload_capability_ ==
                        spm_fmb_preload_capability() &&
                    profile.post_fn_yield_capability_ ==
                        spm_fmb_post_fn_yield_capability() &&
                    profile.post_fn_yield_policy_fingerprint_ ==
                        spm_fmb_post_fn_yield_policy_fingerprint() &&
                    profile.layer_producer_yield_capability_ ==
                        spm_fmb_layer_producer_yield_capability() &&
                    profile.layer_producer_yield_policy_fingerprint_ ==
                        spm_fmb_layer_producer_yield_policy_fingerprint() &&
                    profile.active_group_capability_ ==
                        spm_fmb_active_group_capability() &&
                    profile.dense_ddr_member_capability_ ==
                        spm_fmb_dense_ddr_member_capability() &&
                    profile.dma_endpoint_capability_ ==
                        spm_fmb_dma_endpoint_capability() &&
                    profile.runtime_member_capability_ ==
                        spm_fmb_runtime_member_capability() &&
                    profile.spm_peer_capability_ ==
                        spm_fmb_spm_peer_capability() &&
                    profile.spm_peer_policy_fingerprint_ ==
                        spm_fmb_spm_peer_policy_fingerprint() &&
                    (profile.dense_ddr_member_capability_ ==
                         SpmFmbDenseDdrMemberCapability::Unsealed ||
                     (profile.resolved_hidden_size_ ==
                          pimpl_->hidden_size_ &&
                      profile.resolved_num_layers_ ==
                          pimpl_->num_layers_)) &&
                    profile.cpu_dry_ == snapshot.cpu_dry,
                "schema-v8 BUILD trace: profile/allocation identity mismatch");
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "schema-v8 BUILD trace begin requires an active Graph");
    RpuKernelGraph& graph = RpuKernelGraph::active();
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    TORCH_CHECK(stamp.graph == &graph &&
                    stamp.state == RpuKernelGraph::State::RECORDING,
                "schema-v8 BUILD trace begin requires Graph RECORDING");
    if (composite) {
        TORCH_CHECK(
            composite_identity != 0 &&
                profile.post_fn_yield_capability_ ==
                    SpmFmbPostFnYieldCapability::Unsealed &&
                profile.dma_endpoint_capability_ ==
                    SpmFmbDmaEndpointCapability::Unsealed &&
                profile.runtime_member_capability_ ==
                    SpmFmbRuntimeMemberCapability::Unsealed &&
                profile.spm_peer_capability_ ==
                    SpmFmbSpmPeerCapability::Unsealed,
            "composite BUILD trace v1 accepts one callback-yield producer "
            "owner and rejects legacy PostFn/semantic-DMA/SPM-peer "
            "component authority");
        const auto current_composite_capability =
            spm_fmb_composite_occurrence_capability();
        const uint64_t current_composite_policy =
            spm_fmb_composite_occurrence_policy_fingerprint();
        const bool unsealed_composite_runtime =
            composite_capability ==
                    SpmFmbCompositeOccurrenceCapability::Unsealed &&
                composite_policy_fingerprint == 0 &&
                !composite_preload_follower;
        const bool stable_three_slot_runtime =
            composite_producer &&
                composite_capability ==
                    SpmFmbCompositeOccurrenceCapability::
                        StableThreeInputSlotsSameOwnerPreload &&
                composite_policy_fingerprint != 0 &&
                composite_producer_local_ordinal < 3 &&
                composite_preload_follower ==
                    (composite_producer_local_ordinal != 0) &&
                profile.preload_capability_ ==
                    SpmFmbPreloadCapability::
                        PersistentOutsideResolvedWindow;
        TORCH_CHECK(
            current_composite_capability == composite_capability &&
                current_composite_policy ==
                    composite_policy_fingerprint &&
                ((composite_producer &&
                  (unsealed_composite_runtime ||
                   stable_three_slot_runtime)) ||
                 (!composite_producer &&
                  composite_producer_local_ordinal == 0 &&
                  unsealed_composite_runtime)),
            "composite BUILD trace occurrence runtime capability or local "
            "producer ordinal is invalid");
    } else {
        TORCH_CHECK(stamp.signature_segment_key == profile.profile_hash_,
                    "schema-v8 BUILD trace: Graph signature segment_key must "
                    "equal the resolved profile hash");
    }
    if (profile.dma_endpoint_capability_ !=
        SpmFmbDmaEndpointCapability::Unsealed) {
        graph.bind_semantic_dma_owner_for_fmb(
            pimpl_->pipeline_manifest_generation_,
            profile.expected_owner_generation_);
    }
    if (profile.runtime_member_capability_ !=
        SpmFmbRuntimeMemberCapability::Unsealed) {
        graph.begin_canonical_dma_build_trace_for_fmb(
            profile.profile_hash_);
    }

    const uint64_t composite_occurrence_profile_hash =
        composite
        ? fmb_composite_occurrence_profile_hash(
              profile.profile_hash_, snapshot.layout_hash,
              snapshot.allocation_hash, snapshot.declaration_hash,
              snapshot.persistent_hash, profile.preload_capability_,
              composite_capability, composite_policy_fingerprint)
        : 0;

    Impl::BuildTraceCandidate candidate;
    candidate.armed = true;
    candidate.composite = composite;
    candidate.composite_identity = composite_identity;
    candidate.composite_occurrence_ordinal =
        composite_occurrence_ordinal;
    candidate.composite_producer = composite_producer;
    candidate.composite_producer_local_ordinal =
        composite_producer_local_ordinal;
    candidate.composite_capability = composite_capability;
    candidate.composite_policy_fingerprint =
        composite_policy_fingerprint;
    candidate.composite_occurrence_profile_hash =
        composite_occurrence_profile_hash;
    candidate.composite_preload_follower =
        composite_preload_follower;
    candidate.composite_outer_begin = stamp.position;
    candidate.graph_identity = &graph;
    candidate.graph_build_generation = stamp.build_generation;
    candidate.graph_signature_identity = stamp.signature_identity;
    candidate.graph_signature_segment_key =
        stamp.signature_segment_key;
    candidate.graph_node_begin = stamp.position;
    candidate.profile_hash = profile.profile_hash_;
    candidate.allocation_hash = snapshot.allocation_hash;
    candidate.preload_capability = profile.preload_capability_;
    candidate.expected_owner_generation =
        profile.expected_owner_generation_;
    candidate.cpu_dry = profile.cpu_dry_;
    candidate.profile =
        std::make_unique<SpmFmbResolvedExecutionProfile>(profile);
    candidate.callbacks.reserve(profile.steps_.size());
    pimpl_->pipeline_build_trace_ = std::move(candidate);
    g_fmb_build_trace_owner = pimpl_.get();
}

void FusedModelBase::cancel_spm_pipeline_build_trace() {
    if (!pimpl_->pipeline_build_trace_.armed) return;
    TORCH_CHECK(g_fmb_build_trace_owner == pimpl_.get(),
                "schema-v8 BUILD trace cancel requires its owner thread");
    clear_pipeline_build_trace_noexcept(*pimpl_);
}

void FusedModelBase::arm_spm_pipeline_composite_occurrence_runtime(
    uint64_t composite_identity,
    size_t producer_local_ordinal,
    SpmFmbCompositeOccurrenceCapability capability,
    uint64_t policy_fingerprint,
    uint64_t expected_layout_hash,
    uint64_t expected_allocation_hash,
    uint64_t expected_persistent_generation,
    uint64_t composite_occurrence_profile_hash,
    bool preload_follower) {
    const auto& snapshot = pimpl_->pipeline_manifest_snapshot_;
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "composite occurrence runtime requires an active Graph");
    RpuKernelGraph& graph = RpuKernelGraph::active();
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    TORCH_CHECK(
        !pimpl_->composite_occurrence_runtime_.armed &&
            composite_identity != 0 && producer_local_ordinal < 3 &&
            capability ==
                SpmFmbCompositeOccurrenceCapability::
                    StableThreeInputSlotsSameOwnerPreload &&
            capability == spm_fmb_composite_occurrence_capability() &&
            policy_fingerprint != 0 &&
            policy_fingerprint ==
                spm_fmb_composite_occurrence_policy_fingerprint() &&
            preload_follower == (producer_local_ordinal != 0) &&
            spm_fmb_preload_capability() ==
                SpmFmbPreloadCapability::
                    PersistentOutsideResolvedWindow &&
            snapshot.valid && snapshot.layout_hash == expected_layout_hash &&
            snapshot.allocation_hash == expected_allocation_hash &&
            snapshot.persistent_generation ==
                expected_persistent_generation &&
            (snapshot.cpu_dry ||
             snapshot.persistent_generation ==
                 SPM_ALLOC.persistent_generation()) &&
            composite_occurrence_profile_hash != 0 &&
            (stamp.state == RpuKernelGraph::State::RECORDING ||
             stamp.state == RpuKernelGraph::State::REPLAYING) &&
            stamp.build_generation != 0 &&
            stamp.signature_identity != 0 &&
            stamp.signature_segment_key != 0 &&
            &pimpl_->composite_hidden_in_src_bases_[0] !=
                &pimpl_->composite_hidden_in_src_bases_[1] &&
            &pimpl_->composite_hidden_in_src_bases_[1] !=
                &pimpl_->composite_hidden_in_src_bases_[2],
        "composite occurrence runtime requires one current stable three-slot "
        "producer policy and allocation identity");
    Impl::CompositeOccurrenceRuntime runtime;
    runtime.armed = true;
    runtime.composite_identity = composite_identity;
    runtime.producer_local_ordinal = producer_local_ordinal;
    runtime.capability = capability;
    runtime.policy_fingerprint = policy_fingerprint;
    runtime.expected_layout_hash = expected_layout_hash;
    runtime.expected_allocation_hash = expected_allocation_hash;
    runtime.expected_persistent_generation =
        expected_persistent_generation;
    runtime.composite_occurrence_profile_hash =
        composite_occurrence_profile_hash;
    runtime.graph_identity = &graph;
    runtime.graph_state = static_cast<uint8_t>(stamp.state);
    runtime.graph_build_generation = stamp.build_generation;
    runtime.graph_signature_identity = stamp.signature_identity;
    runtime.graph_signature_segment_key =
        stamp.signature_segment_key;
    runtime.preload_follower = preload_follower;
    pimpl_->composite_occurrence_runtime_ = runtime;
}

void FusedModelBase::finish_spm_pipeline_composite_occurrence_runtime(
    uint64_t composite_identity,
    size_t producer_local_ordinal) {
    const auto& runtime = pimpl_->composite_occurrence_runtime_;
    const bool graph_current =
        RpuKernelGraph::has_active() &&
        &RpuKernelGraph::active() == runtime.graph_identity;
    const GraphOpStreamStamp stamp = graph_current
        ? RpuKernelGraph::active().op_stream_stamp()
        : GraphOpStreamStamp{};
    TORCH_CHECK(
        runtime.armed &&
            runtime.composite_identity == composite_identity &&
            runtime.producer_local_ordinal == producer_local_ordinal &&
            producer_local_ordinal <
                pimpl_->composite_hidden_in_src_bases_.size() &&
            graph_current &&
            static_cast<uint8_t>(stamp.state) == runtime.graph_state &&
            stamp.build_generation == runtime.graph_build_generation &&
            stamp.signature_identity ==
                runtime.graph_signature_identity &&
            stamp.signature_segment_key ==
                runtime.graph_signature_segment_key &&
            runtime.input_slot_consumed &&
            runtime.preload_decision_consumed &&
            pimpl_->composite_hidden_in_src_bases_[
                producer_local_ordinal] != 0,
        "composite producer occurrence did not consume its stable input "
        "slot and preload role exactly once");
    pimpl_->composite_occurrence_runtime_ = {};
}

void FusedModelBase::cancel_spm_pipeline_composite_occurrence_runtime()
    noexcept {
    pimpl_->composite_occurrence_runtime_ = {};
}

void FusedModelBase::arm_spm_pipeline_composite_physical_execution(
    const SpmPipelineLease& lease,
    uint64_t composite_identity,
    bool producer,
    size_t producer_local_ordinal,
    bool require_committed) {
    auto& runtime = pimpl_->composite_physical_execution_;
    TORCH_CHECK(
        !runtime.armed && runtime.lease == nullptr &&
            composite_identity != 0 && RpuKernelGraph::has_active(),
        "composite physical execution requires one fresh lexical Graph "
        "occurrence");
    RpuKernelGraph& graph = RpuKernelGraph::active();
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    TORCH_CHECK(
        stamp.state == RpuKernelGraph::State::RECORDING ||
            stamp.state == RpuKernelGraph::State::REPLAYING,
        "composite physical execution requires Graph BUILD or REPLAY");
    TORCH_CHECK(
        require_committed ||
            stamp.state == RpuKernelGraph::State::RECORDING,
        "an uncommitted composite reservation is valid only for its "
        "bootstrap BUILD");
    lease.validate_composite_execution_authority(
        lease.epoch(), lease.plan_hash(), require_committed);
    TORCH_CHECK(
        pimpl_->pipeline_lease_epoch_ == lease.epoch() &&
            pimpl_->pipeline_plan_hash_ == lease.plan_hash() &&
            pimpl_->alloc_gen_ == lease.allocator_generation(),
        "composite physical execution owner was not adopted by this "
        "reservation lease");

    const size_t producer_count =
        lease.composite_producer_occurrence_count();
    const size_t yields_per_producer =
        lease.composite_yields_per_producer();
    const size_t run_add_count = lease.composite_run_add_count();
    TORCH_CHECK(
        producer_count > 0 && yields_per_producer > 1 &&
            producer_count <=
                std::numeric_limits<size_t>::max() /
                    (yields_per_producer - 1) &&
            run_add_count ==
                producer_count * (yields_per_producer - 1) &&
            (producer ? producer_local_ordinal < producer_count
                      : producer_local_ordinal == 0),
        "composite physical execution target matrix is malformed");

    size_t preflight_count = 0;
    if (producer) {
        for (size_t yield = 0; yield < yields_per_producer; ++yield) {
            const uint32_t target =
                lease.resolve_composite_producer_target_addr(
                    /*core=*/0, producer_local_ordinal, yield);
            TORCH_CHECK(
                target != 0,
                "composite producer preflight resolved a zero target");
            ++preflight_count;
        }
        TORCH_CHECK(
            preflight_count == yields_per_producer,
            "composite producer preflight did not cover every yield");
    } else {
        for (size_t yield = 0; yield + 1 < yields_per_producer; ++yield) {
            for (size_t producer_ordinal = 0;
                 producer_ordinal < producer_count;
                 ++producer_ordinal) {
                const auto target = lease.resolve_composite_run_add_addrs(
                    /*core=*/0, producer_ordinal, yield);
                TORCH_CHECK(
                    target.payload_addr != 0 &&
                        target.consumer_addr != 0 &&
                        target.payload_addr != target.consumer_addr &&
                        target.bytes > 0,
                    "composite consumer preflight resolved an invalid "
                    "RunAdd target");
                ++preflight_count;
            }
        }
        TORCH_CHECK(
            preflight_count == run_add_count,
            "composite consumer preflight did not cover every retained "
            "payload and destination");
    }

    Impl::CompositePhysicalExecution next;
    next.armed = true;
    next.producer = producer;
    next.require_committed = require_committed;
    next.lease = &lease;
    next.lease_epoch = lease.epoch();
    next.plan_hash = lease.plan_hash();
    next.composite_identity = composite_identity;
    next.producer_local_ordinal = producer_local_ordinal;
    next.expected_target_count =
        producer ? yields_per_producer : run_add_count;
    next.graph_identity = &graph;
    next.graph_state = static_cast<uint8_t>(stamp.state);
    next.graph_build_generation = stamp.build_generation;
    next.graph_signature_identity = stamp.signature_identity;
    next.graph_signature_segment_key = stamp.signature_segment_key;
    runtime = next;
}

void FusedModelBase::finish_spm_pipeline_composite_physical_execution(
    const SpmPipelineLease& lease,
    uint64_t composite_identity,
    bool producer,
    size_t producer_local_ordinal) {
    const auto& runtime = pimpl_->composite_physical_execution_;
    const bool graph_current =
        RpuKernelGraph::has_active() &&
        &RpuKernelGraph::active() == runtime.graph_identity;
    const GraphOpStreamStamp stamp = graph_current
        ? RpuKernelGraph::active().op_stream_stamp()
        : GraphOpStreamStamp{};
    TORCH_CHECK(
        runtime.armed && runtime.lease == &lease &&
            runtime.lease_epoch == lease.epoch() &&
            runtime.plan_hash == lease.plan_hash() &&
            runtime.composite_identity == composite_identity &&
            runtime.producer == producer &&
            runtime.producer_local_ordinal == producer_local_ordinal &&
            runtime.next_target == runtime.expected_target_count &&
            graph_current &&
            static_cast<uint8_t>(stamp.state) == runtime.graph_state &&
            stamp.build_generation == runtime.graph_build_generation &&
            stamp.signature_identity ==
                runtime.graph_signature_identity &&
            stamp.signature_segment_key ==
                runtime.graph_signature_segment_key,
        "composite physical occurrence did not consume its exact target "
        "matrix under the original Graph authority");
    lease.validate_composite_execution_authority(
        runtime.lease_epoch, runtime.plan_hash,
        runtime.require_committed);
    pimpl_->composite_physical_execution_ = {};
}

void FusedModelBase::cancel_spm_pipeline_composite_physical_execution()
    noexcept {
    pimpl_->composite_physical_execution_ = {};
}

uint64_t& FusedModelBase::current_hidden_in_src_base() {
    auto& runtime = pimpl_->composite_occurrence_runtime_;
    if (!runtime.armed) return pimpl_->hidden_in_src_base_;
    TORCH_CHECK(RpuKernelGraph::has_active() &&
                    &RpuKernelGraph::active() == runtime.graph_identity,
                "composite mutable input slot lost its Graph owner");
    const GraphOpStreamStamp stamp =
        RpuKernelGraph::active().op_stream_stamp();
    TORCH_CHECK(
        runtime.capability ==
                SpmFmbCompositeOccurrenceCapability::
                    StableThreeInputSlotsSameOwnerPreload &&
            runtime.producer_local_ordinal <
                pimpl_->composite_hidden_in_src_bases_.size() &&
            runtime.composite_occurrence_profile_hash != 0 &&
            static_cast<uint8_t>(stamp.state) == runtime.graph_state &&
            stamp.build_generation == runtime.graph_build_generation &&
            stamp.signature_identity ==
                runtime.graph_signature_identity &&
            stamp.signature_segment_key ==
                runtime.graph_signature_segment_key,
        "composite mutable input slot authority is stale");
    runtime.input_slot_consumed = true;
    return pimpl_->composite_hidden_in_src_bases_[
        runtime.producer_local_ordinal];
}

bool FusedModelBase::consume_spm_pipeline_composite_preload_follower() {
    auto& runtime = pimpl_->composite_occurrence_runtime_;
    if (!runtime.armed) return false;
    const auto& snapshot = pimpl_->pipeline_manifest_snapshot_;
    TORCH_CHECK(RpuKernelGraph::has_active() &&
                    &RpuKernelGraph::active() == runtime.graph_identity,
                "composite preload role lost its Graph owner");
    const GraphOpStreamStamp stamp =
        RpuKernelGraph::active().op_stream_stamp();
    TORCH_CHECK(
        !runtime.preload_decision_consumed &&
            runtime.capability ==
                SpmFmbCompositeOccurrenceCapability::
                    StableThreeInputSlotsSameOwnerPreload &&
            runtime.policy_fingerprint ==
                spm_fmb_composite_occurrence_policy_fingerprint() &&
            snapshot.valid &&
            snapshot.layout_hash == runtime.expected_layout_hash &&
            snapshot.allocation_hash ==
                runtime.expected_allocation_hash &&
            snapshot.persistent_generation ==
                runtime.expected_persistent_generation &&
            runtime.composite_occurrence_profile_hash != 0 &&
            static_cast<uint8_t>(stamp.state) == runtime.graph_state &&
            stamp.build_generation == runtime.graph_build_generation &&
            stamp.signature_identity ==
                runtime.graph_signature_identity &&
            stamp.signature_segment_key ==
                runtime.graph_signature_segment_key &&
            (!runtime.preload_follower ||
             snapshot.cpu_dry ||
             (snapshot.persistent_generation ==
                  SPM_ALLOC.persistent_generation() &&
              pimpl_->cached_preload_gen_ ==
                  snapshot.persistent_generation &&
              !pimpl_->preload_callbacks_dirty_)),
        "composite preload follower requires the leader's current clean "
        "persistent allocation");
    runtime.preload_decision_consumed = true;
    return runtime.preload_follower;
}

std::pair<uintptr_t, uint64_t>
FusedModelBase::
exchange_spm_pipeline_composite_input_slot_for_cpu_contract(
    uint64_t value) {
    TORCH_CHECK(pimpl_->pipeline_manifest_snapshot_.valid &&
                    pimpl_->pipeline_manifest_snapshot_.cpu_dry,
                "composite input-slot probe requires a CPU-dry snapshot");
    uint64_t& slot = current_hidden_in_src_base();
    const uint64_t previous = slot;
    slot = value;
    return {reinterpret_cast<uintptr_t>(&slot), previous};
}

bool FusedModelBase::
consume_spm_pipeline_composite_preload_role_for_cpu_contract() {
    TORCH_CHECK(pimpl_->pipeline_manifest_snapshot_.valid &&
                    pimpl_->pipeline_manifest_snapshot_.cpu_dry,
                "composite preload-role probe requires a CPU-dry snapshot");
    return consume_spm_pipeline_composite_preload_follower();
}

SpmFmbCompletedBuildTrace
FusedModelBase::complete_spm_pipeline_composite_build_trace() {
    auto& candidate = pimpl_->pipeline_build_trace_;
    TORCH_CHECK(g_fmb_build_trace_owner == pimpl_.get() &&
                    candidate.armed && candidate.composite &&
                    candidate.composite_identity != 0 &&
                    candidate.profile != nullptr &&
                    candidate.traversal_complete &&
                    candidate.next_step ==
                        candidate.profile->steps_.size() &&
                    !candidate.persistent_preload_callback_open &&
                    !candidate.callback_open &&
                    !candidate.producer_yield_open &&
                    !candidate.pending_producer_yield.has_value() &&
                    !pimpl_->pipeline_runtime_member_frame_.active &&
                    g_fmb_runtime_member_owner != pimpl_.get() &&
                    !g_fmb_build_trace_callback_begin.has_value() &&
                    !g_fmb_post_fn_yield_begin.has_value() &&
                    !candidate.callbacks.empty(),
                "composite BUILD occurrence completion requires one exact "
                "finished FMB traversal");
    TORCH_CHECK(RpuKernelGraph::has_active() &&
                    &RpuKernelGraph::active() ==
                        candidate.graph_identity,
                "composite BUILD occurrence completion lost its outer "
                "Graph");
    const GraphOpStreamStamp stamp =
        RpuKernelGraph::active().op_stream_stamp();
    TORCH_CHECK(stamp.state == RpuKernelGraph::State::RECORDING &&
                    stamp.build_generation ==
                        candidate.graph_build_generation &&
                    stamp.signature_identity ==
                        candidate.graph_signature_identity &&
                    stamp.signature_segment_key ==
                        candidate.graph_signature_segment_key &&
                    candidate.composite_outer_begin <=
                        candidate.graph_node_begin &&
                    candidate.callbacks.back().graph_node_end ==
                        stamp.position,
                "composite BUILD occurrence completion Graph window "
                "drifted");
    auto payload =
        std::make_unique<SpmFmbCompletedBuildTrace::Payload>();
    payload->owner = this;
    payload->owner_generation =
        pimpl_->pipeline_manifest_generation_;
    payload->expected_owner_generation =
        candidate.expected_owner_generation;
    payload->candidate = std::move(candidate);
    candidate = {};
    g_fmb_build_trace_owner = nullptr;
    g_fmb_build_trace_callback_begin.reset();
    g_fmb_post_fn_yield_begin.reset();
    return SpmFmbCompletedBuildTrace(std::move(payload));
}

void FusedModelBase::drive_spm_pipeline_build_trace_for_cpu_contract(
    const SpmFmbResolvedExecutionProfile& profile) {
    auto& candidate = pimpl_->pipeline_build_trace_;
    TORCH_CHECK(g_fmb_build_trace_owner == pimpl_.get(),
                "schema-v8 CPU trace driver requires its owner thread");
    auto clear_failed_candidate = c10::make_scope_exit([&] {
        bool marker_may_have_been_recorded =
            candidate.pending_producer_yield.has_value();
        for (const auto& callback : candidate.callbacks) {
            marker_may_have_been_recorded =
                marker_may_have_been_recorded ||
                !callback.producer_yields.empty();
        }
        if (marker_may_have_been_recorded &&
            RpuKernelGraph::has_active()) {
            RpuKernelGraph::active()
                .poison_semantic_spm_producer_yield_for_fmb();
        }
        clear_pipeline_build_trace_noexcept(*pimpl_);
    });
    TORCH_CHECK(candidate.armed && candidate.profile != nullptr &&
                    candidate.cpu_dry && profile.cpu_dry_ &&
                    candidate.profile_hash == profile.profile_hash_ &&
                    candidate.profile->expected_layout_hash_ ==
                        profile.expected_layout_hash_ &&
                    candidate.profile->expected_allocation_hash_ ==
                        profile.expected_allocation_hash_ &&
                    candidate.profile->expected_owner_generation_ ==
                        profile.expected_owner_generation_ &&
                    candidate.next_step == 0 &&
                    candidate.callbacks.empty(),
                "schema-v8 CPU trace driver requires one fresh armed "
                "CPU-dry profile");
    const ModelStaticConfig static_cfg = static_config();
    const bool profile_has_post_fn =
        !candidate.profile->steps_.empty() &&
        candidate.profile->steps_.back().kind ==
            SpmFmbOccurrenceKind::PostFn;
    TORCH_CHECK(static_cfg.preload_fn == nullptr &&
                    static_cfg.pre_layers_fn == nullptr &&
                    static_cfg.post_layers_fn == nullptr &&
                    ((profile.layer_producer_yield_capability_ !=
                          SpmFmbLayerProducerYieldCapability::Unsealed) ||
                     (profile.post_fn_yield_capability_ ==
                          SpmFmbPostFnYieldCapability::Unsealed &&
                      static_cfg.post_fn == nullptr) ||
                     (profile.post_fn_yield_capability_ ==
                          SpmFmbPostFnYieldCapability::
                              CanonicalDenseReplicatedFp16 &&
                      static_cfg.post_fn != nullptr &&
                      static_cfg.post_output_shape.empty())) &&
                    ((profile.layer_producer_yield_capability_ ==
                          SpmFmbLayerProducerYieldCapability::Unsealed) ||
                     (profile.layer_producer_yield_capability_ ==
                          SpmFmbLayerProducerYieldCapability::
                              CanonicalDenseReplicatedFp16 &&
                      (static_cfg.post_fn != nullptr) ==
                          profile_has_post_fn &&
                      static_cfg.post_output_shape.empty() &&
                      !static_cfg.fast_replay_skip_layer_loop &&
                      !static_cfg.fast_replay_skip_full_body &&
                      !static_cfg.fast_replay_bake_post_fn)),
                "schema-v8 CPU trace driver STOP: opaque callbacks remain "
                "unsealed");
    candidate.cpu_driver = true;
    for (const SpmFmbResolvedExecutionStep& expected :
         candidate.profile->steps_) {
        ChunkInfo chunk{
            expected.chunk_index, expected.chunk_offset,
            expected.chunk_len, expected.chunk_kv_seq_len};
        FmbExecutionStepView actual{
            expected.kind, expected.body_id, expected.group_id,
            expected.layer, expected.traversal_ordinal,
            expected.has_chunk ? &chunk : nullptr,
            expected.scope_mask, expected.updates_io_context,
            expected.input_in_spm, expected.output_to_spm};
        if (actual.updates_io_context) {
            pimpl_->ctx_.input_in_spm = actual.input_in_spm;
            pimpl_->ctx_.output_to_spm = actual.output_to_spm;
        }
        pimpl_->ctx_.body_iter = expected.body_id;
        invoke_trace_callback(
            *pimpl_, expected, actual, [&] {
                if (expected.kind == SpmFmbOccurrenceKind::PostFn) {
                    TORCH_CHECK(
                        !expected.has_chunk && static_cfg.post_fn != nullptr,
                        "schema-v8 CPU trace driver: PostFn callback is "
                        "missing or carries chunk geometry");
                    std::invoke(static_cfg.post_fn, *this);
                    return;
                }
                begin_spm_pipeline_runtime_member_callback_for_build(
                    profile, expected);
                bool runtime_callback_complete = false;
                auto cancel_runtime_callback = c10::make_scope_exit([&] {
                    if (!runtime_callback_complete) {
                        cancel_spm_pipeline_runtime_member_callback();
                    }
                });
                if (expected.kind ==
                    SpmFmbOccurrenceKind::KvInsertBody) {
                    TORCH_CHECK(static_cfg.kv_first_fn != nullptr,
                                "schema-v8 CPU trace driver: KV callback "
                                "is missing");
                    std::invoke(static_cfg.kv_first_fn, *this,
                                expected.layer, chunk);
                } else {
                    TORCH_CHECK(
                        expected.kind ==
                            SpmFmbOccurrenceKind::LayerBody,
                        "schema-v8 CPU trace driver: unsupported callback "
                        "kind");
                    build_layer_subgraph(expected.layer, chunk);
                }
                end_spm_pipeline_runtime_member_callback();
                runtime_callback_complete = true;
            }, [this] { end_spm_pipeline_build_trace_callback(); });
    }
    TORCH_CHECK(candidate.next_step == candidate.profile->steps_.size(),
                "schema-v8 CPU trace driver: incomplete traversal");
    candidate.traversal_complete = true;
    if (profile.runtime_member_capability_ !=
        SpmFmbRuntimeMemberCapability::Unsealed) {
        RpuKernelGraph::active().complete_canonical_dma_build_trace_for_fmb();
    }
    clear_failed_candidate.release();
}

SpmFmbSealedBuildTrace FusedModelBase::seal_spm_pipeline_build_trace(
    const SpmFmbResolvedExecutionProfile& profile,
    const RpuKernelGraph& graph) {
    TORCH_CHECK(g_fmb_build_trace_owner == pimpl_.get(),
                "schema-v8 BUILD trace seal requires its owner thread");
    auto fail_and_cancel = c10::make_scope_exit(
        [&] { clear_pipeline_build_trace_noexcept(*pimpl_); });
    const auto& candidate = pimpl_->pipeline_build_trace_;
    TORCH_CHECK(candidate.armed && candidate.profile != nullptr &&
                    candidate.traversal_complete &&
                    !candidate.persistent_preload_callback_open &&
                    !candidate.callback_open &&
                    !g_fmb_build_trace_callback_begin.has_value(),
                "schema-v8 BUILD trace seal requires one complete candidate");
    TORCH_CHECK(candidate.graph_identity == &graph &&
                    graph.state() == RpuKernelGraph::State::BUILT,
                "schema-v8 BUILD trace seal requires the owning Graph BUILT");
    const auto& snapshot = pimpl_->pipeline_manifest_snapshot_;
    const bool candidate_is_producer =
        candidate.profile->layer_producer_yield_capability_ !=
        SpmFmbLayerProducerYieldCapability::Unsealed;
    const uint64_t expected_composite_occurrence_profile_hash =
        candidate.composite
        ? fmb_composite_occurrence_profile_hash(
              candidate.profile_hash, snapshot.layout_hash,
              snapshot.allocation_hash, snapshot.declaration_hash,
              snapshot.persistent_hash,
              candidate.profile->preload_capability_,
              candidate.composite_capability,
              candidate.composite_policy_fingerprint)
        : 0;
    const bool unsealed_composite_occurrence =
        candidate.composite_capability ==
                SpmFmbCompositeOccurrenceCapability::Unsealed &&
        candidate.composite_policy_fingerprint == 0 &&
        candidate.composite_occurrence_profile_hash == 0 &&
        !candidate.composite_preload_follower;
    const bool stable_three_slot_occurrence =
        candidate.composite && candidate_is_producer &&
        candidate.composite_capability ==
            SpmFmbCompositeOccurrenceCapability::
                StableThreeInputSlotsSameOwnerPreload &&
        candidate.composite_policy_fingerprint != 0 &&
        candidate.composite_producer_local_ordinal < 3 &&
        candidate.composite_preload_follower ==
            (candidate.composite_producer_local_ordinal != 0) &&
        candidate.composite_occurrence_profile_hash != 0 &&
        candidate.composite_occurrence_profile_hash ==
            expected_composite_occurrence_profile_hash &&
        candidate.profile->preload_capability_ ==
            SpmFmbPreloadCapability::PersistentOutsideResolvedWindow;
    TORCH_CHECK(
        snapshot.valid &&
            (!candidate.composite ||
             candidate.composite_producer == candidate_is_producer) &&
            ((candidate.composite &&
              (unsealed_composite_occurrence ||
               stable_three_slot_occurrence)) ||
             (!candidate.composite && !candidate.composite_producer &&
              candidate.composite_producer_local_ordinal == 0 &&
              unsealed_composite_occurrence &&
              candidate.composite_identity == 0 &&
              candidate.composite_occurrence_ordinal == 0)),
        "schema-v8 BUILD trace seal: composite occurrence policy drift");
    TORCH_CHECK(profile.profile_hash_ == candidate.profile_hash &&
                    profile.expected_allocation_hash_ ==
                        candidate.allocation_hash &&
                    candidate.profile->expected_layout_hash_ ==
                        profile.expected_layout_hash_ &&
                    candidate.profile->expected_owner_generation_ ==
                        profile.expected_owner_generation_ &&
                    candidate.profile->preload_capability_ ==
                        profile.preload_capability_ &&
                    profile.preload_capability_ ==
                        spm_fmb_preload_capability() &&
                    candidate.profile->post_fn_yield_capability_ ==
                        profile.post_fn_yield_capability_ &&
                    profile.post_fn_yield_capability_ ==
                        spm_fmb_post_fn_yield_capability() &&
                    candidate.profile->post_fn_yield_policy_fingerprint_ ==
                        profile.post_fn_yield_policy_fingerprint_ &&
                    profile.post_fn_yield_policy_fingerprint_ ==
                        spm_fmb_post_fn_yield_policy_fingerprint() &&
                    candidate.profile->layer_producer_yield_capability_ ==
                        profile.layer_producer_yield_capability_ &&
                    profile.layer_producer_yield_capability_ ==
                        spm_fmb_layer_producer_yield_capability() &&
                    candidate.profile
                            ->layer_producer_yield_policy_fingerprint_ ==
                        profile.layer_producer_yield_policy_fingerprint_ &&
                    profile.layer_producer_yield_policy_fingerprint_ ==
                        spm_fmb_layer_producer_yield_policy_fingerprint() &&
                    candidate.profile->active_group_capability_ ==
                        profile.active_group_capability_ &&
                    profile.active_group_capability_ ==
                        spm_fmb_active_group_capability() &&
                    candidate.profile->dense_ddr_member_capability_ ==
                        profile.dense_ddr_member_capability_ &&
                    profile.dense_ddr_member_capability_ ==
                        spm_fmb_dense_ddr_member_capability() &&
                    candidate.profile->dma_endpoint_capability_ ==
                        profile.dma_endpoint_capability_ &&
                    profile.dma_endpoint_capability_ ==
                        spm_fmb_dma_endpoint_capability() &&
                    candidate.profile->runtime_member_capability_ ==
                        profile.runtime_member_capability_ &&
                    profile.runtime_member_capability_ ==
                        spm_fmb_runtime_member_capability() &&
                    candidate.profile->spm_peer_capability_ ==
                        profile.spm_peer_capability_ &&
                    profile.spm_peer_capability_ ==
                        spm_fmb_spm_peer_capability() &&
                    candidate.profile->spm_peer_policy_fingerprint_ ==
                        profile.spm_peer_policy_fingerprint_ &&
                    profile.spm_peer_policy_fingerprint_ ==
                        spm_fmb_spm_peer_policy_fingerprint() &&
                    candidate.profile->build_body_iterations_ ==
                        profile.build_body_iterations_ &&
                    candidate.profile->resolved_chunk_mode_ ==
                        profile.resolved_chunk_mode_ &&
                    candidate.profile->resolved_inter_layer_io_ ==
                        profile.resolved_inter_layer_io_ &&
                    candidate.profile->resolved_chunk_outer_within_group_ ==
                        profile.resolved_chunk_outer_within_group_ &&
                    candidate.profile->resolved_cross_batch_ ==
                        profile.resolved_cross_batch_ &&
                    candidate.profile->resolved_hidden_size_ ==
                        profile.resolved_hidden_size_ &&
                    candidate.profile->resolved_num_layers_ ==
                        profile.resolved_num_layers_ &&
                    (profile.dense_ddr_member_capability_ ==
                         SpmFmbDenseDdrMemberCapability::Unsealed ||
                     (profile.resolved_hidden_size_ ==
                          pimpl_->hidden_size_ &&
                      profile.resolved_num_layers_ ==
                          pimpl_->num_layers_)) &&
                    profile.cpu_dry_ == candidate.cpu_dry &&
                    profile.steps_.size() == candidate.callbacks.size(),
                "schema-v8 BUILD trace seal: profile candidate drift");
    const std::shared_ptr<const uint64_t> owner =
        profile.owner_generation_.lock();
    TORCH_CHECK(owner != nullptr &&
                    owner.get() ==
                        pimpl_->pipeline_manifest_generation_.get() &&
                    *owner == candidate.expected_owner_generation &&
                    *owner == profile.expected_owner_generation_,
                "schema-v8 BUILD trace seal: profile owner is stale");
    TORCH_CHECK(graph.build_generation() ==
                    candidate.graph_build_generation &&
                    graph.build_topology_hash() != 0 &&
                    graph.built_signature().segment_key ==
                        candidate.graph_signature_segment_key &&
                    graph.built_signature_identity() ==
                        candidate.graph_signature_identity,
                "schema-v8 BUILD trace seal: Graph build identity drift");
    const GraphNodeKindMask outer_kind_mask =
        graph.build_node_kind_mask();
    TORCH_CHECK(
        (outer_kind_mask & kSpmFmbOpaqueGraphNodeKinds) == 0,
        "schema-v8 BUILD trace STOP: outer Graph contains an opaque "
        "ChildGraph/HostCallback/Tier3Oneshot node");

    size_t active_callbacks = 0;
    size_t noop_callbacks = 0;
    size_t access_count = 0;
    size_t previous_end = 0;
    bool have_previous = false;
    std::vector<SpmFmbSealedBuildTrace::CallbackSummary>
        callback_summaries;
    callback_summaries.reserve(candidate.callbacks.size());
    uint64_t trace_hash = fmb_hash_u64(kFmbFnvOffset, 10);
    trace_hash = fmb_hash_u64(trace_hash, profile.profile_hash_);
    trace_hash = fmb_hash_u64(trace_hash, candidate.allocation_hash);
    trace_hash = fmb_hash_u64(
        trace_hash, candidate.graph_signature_identity);
    trace_hash = fmb_hash_u64(
        trace_hash, candidate.graph_signature_segment_key);
    trace_hash = fmb_hash_u64(
        trace_hash, graph.build_topology_hash());
    trace_hash = fmb_hash_u64(trace_hash, outer_kind_mask);
    trace_hash = fmb_hash_u64(trace_hash, candidate.cpu_dry);
    if (candidate.composite) {
        trace_hash = fmb_hash_u64(
            trace_hash, UINT64_C(0x434f4d5054524331));
        trace_hash = fmb_hash_u64(
            trace_hash, candidate.composite_identity);
        trace_hash = fmb_hash_u64(
            trace_hash, candidate.composite_occurrence_ordinal);
        trace_hash = fmb_hash_u64(
            trace_hash, candidate.composite_outer_begin);
        if (candidate.composite_capability !=
            SpmFmbCompositeOccurrenceCapability::Unsealed) {
            trace_hash = fmb_hash_u64(
                trace_hash, UINT64_C(0x434f4d5054524332));
            trace_hash = fmb_hash_u64(
                trace_hash, candidate.composite_producer);
            trace_hash = fmb_hash_u64(
                trace_hash,
                candidate.composite_producer_local_ordinal);
            trace_hash = fmb_hash_u64(
                trace_hash,
                static_cast<uint8_t>(
                    candidate.composite_capability));
            trace_hash = fmb_hash_u64(
                trace_hash,
                candidate.composite_policy_fingerprint);
            trace_hash = fmb_hash_u64(
                trace_hash,
                candidate.composite_occurrence_profile_hash);
            trace_hash = fmb_hash_u64(
                trace_hash,
                candidate.composite_preload_follower);
        }
    }
    if (profile.active_group_capability_ !=
        SpmFmbActiveGroupCapability::Unsealed) {
        trace_hash = fmb_hash_u64(
            trace_hash,
            static_cast<uint8_t>(profile.active_group_capability_));
    }
    trace_hash = fmb_hash_u64(trace_hash, candidate.callbacks.size());
    for (const auto& callback : candidate.callbacks) {
        TORCH_CHECK(
            have_previous
                ? previous_end == callback.graph_node_begin
                : candidate.graph_node_begin ==
                    callback.graph_node_begin,
            "schema-v8 BUILD trace seal: callback windows are not exactly "
            "adjacent");
        previous_end = callback.graph_node_end;
        have_previous = true;
        if (callback.graph_node_begin == callback.graph_node_end) {
            ++noop_callbacks;
        } else {
            ++active_callbacks;
        }
        const HashedBuildTraceCallback absolute =
            hash_build_trace_callback(
                trace_hash, callback, BuildTraceWindowHashMode::Absolute);
        trace_hash = absolute.hash;
        TORCH_CHECK(absolute.access_count <=
                        std::numeric_limits<size_t>::max() - access_count,
                    "schema-v8 BUILD trace seal: access count overflow");
        access_count += absolute.access_count;
        const HashedBuildTraceCallback semantic =
            hash_build_trace_callback(
                fmb_hash_u64(kFmbFnvOffset, 11), callback,
                BuildTraceWindowHashMode::Relative);
        TORCH_INTERNAL_ASSERT(
            semantic.access_count == absolute.access_count);
        callback_summaries.push_back({
            callback.kind,
            callback.key,
            callback.body_id,
            callback.group_id,
            callback.layer,
            callback.traversal_ordinal,
            callback.chunk_index,
            callback.chunk_offset,
            callback.chunk_len,
            callback.chunk_kv_seq_len,
            callback.scope_mask,
            callback.updates_io_context,
            callback.input_in_spm,
            callback.output_to_spm,
            callback.local_phase_begin,
            callback.local_phase_end,
            callback.global_origin,
            callback.graph_node_begin,
            callback.graph_node_end,
            callback.graph_node_kind_mask,
            callback.graph_topology_hash,
            semantic.access_count,
            fmb_nonzero_hash(semantic.hash),
        });
        auto& sealed_callback = callback_summaries.back();
        sealed_callback.legacy_dma_count = callback.legacy_dma_count;
        sealed_callback.accesses.reserve(callback.accesses.size());
        for (const auto& access : callback.accesses) {
            sealed_callback.accesses.push_back({
                access.allocation_ordinal,
                access.declaration_ordinal,
                access.alias_root_ordinal,
                static_cast<uint8_t>(access.storage),
                static_cast<uint8_t>(access.scope),
                access.layer,
                access.accessor,
                access.core_mask,
                access.root_offset,
                access.protected_bytes,
                access.resolve_count,
            });
        }
        sealed_callback.semantic_dmas.reserve(
            callback.semantic_dmas.size());
        for (const auto& dma : callback.semantic_dmas) {
            sealed_callback.semantic_dmas.push_back({
                dma.relative_node_index,
                dma.endpoint_id,
                dma.variant,
                dma.src_addr,
                dma.dst_addr,
                dma.bytes,
                dma.channel,
                dma.live_offset,
                dma.live_base,
                dma.owner_generation,
                dma.expected_owner_generation,
                dma.canonical_member_emission,
                dma.spm_peer_id,
            });
        }
        sealed_callback.spm_peers.reserve(callback.spm_peers.size());
        for (const auto& peer : callback.spm_peers) {
            sealed_callback.spm_peers.push_back({
                peer.layer_group_ordinal,
                peer.member_ordinal,
                peer.role,
                peer.kind,
                peer.endpoint_id,
                peer.peer_id,
                peer.allocation_ordinal,
                peer.declaration_ordinal,
                peer.alias_root_ordinal,
                peer.declaration_name_hash,
                peer.alias_root_name_hash,
                peer.storage,
                peer.scope,
                peer.layer,
                peer.phase_start,
                peer.phase_end,
                peer.root_offset,
                peer.allocation_offset_from_root,
                peer.logical_capacity,
                peer.aligned_capacity,
                peer.peer_slice_begin,
                peer.peer_slice_count,
                peer.core_mask,
                peer.expected_occurrence_count,
                peer.semantic_hash,
            });
        }
        sealed_callback.producer_yields.reserve(
            callback.producer_yields.size());
        for (const auto& yield : callback.producer_yields) {
            sealed_callback.producer_yields.push_back({
                yield.ordinal,
                yield.semantic_id,
                yield.writer_kind,
                yield.spec,
                yield.allocation_ordinal,
                yield.declaration_ordinal,
                yield.alias_root_ordinal,
                yield.declaration_name_hash,
                yield.alias_root_name_hash,
                yield.root_offset,
                yield.allocation_offset_from_root,
                yield.logical_capacity,
                yield.aligned_capacity,
                yield.graph_node_begin,
                yield.graph_node_end,
                yield.graph_node_kind_mask,
                yield.graph_topology_hash,
            });
        }
    }
    TORCH_INTERNAL_ASSERT(have_previous &&
                          callback_summaries.size() ==
                              candidate.callbacks.size() &&
                          candidate.graph_node_begin <= previous_end &&
                          previous_end <= graph.graph_size());
    if (stable_three_slot_occurrence) {
        TORCH_CHECK(
            candidate.composite_preload_follower
                ? candidate.composite_outer_begin ==
                      candidate.graph_node_begin
                : (candidate.cpu_dry ||
                   candidate.composite_outer_begin <
                       candidate.graph_node_begin),
            "composite preload role did not produce the required "
            "leader/follower Graph prefix");
    }
    trace_hash = fmb_nonzero_hash(trace_hash);
    SpmFmbSealedBuildTrace result(
        trace_hash, candidate.profile_hash, candidate.allocation_hash,
        candidate.graph_build_generation,
        candidate.graph_signature_identity,
        candidate.graph_signature_segment_key,
        graph.build_topology_hash(), outer_kind_mask,
        candidate.callbacks.size(),
        active_callbacks, noop_callbacks, access_count,
        candidate.graph_node_begin, previous_end,
        profile.post_fn_yield_capability_,
        profile.post_fn_yield_policy_fingerprint_,
        profile.layer_producer_yield_capability_,
        profile.layer_producer_yield_policy_fingerprint_,
        profile.active_group_capability_,
        profile.dense_ddr_member_capability_,
        profile.dma_endpoint_capability_,
        profile.runtime_member_capability_,
        profile.spm_peer_capability_,
        profile.spm_peer_policy_fingerprint_,
        profile.build_body_iterations_,
        profile.resolved_chunk_mode_,
        profile.resolved_inter_layer_io_,
        profile.resolved_chunk_outer_within_group_,
        profile.resolved_cross_batch_,
        profile.resolved_hidden_size_,
        profile.resolved_num_layers_,
        callback_summaries,
        candidate.cpu_dry, &graph,
        pimpl_->pipeline_manifest_generation_,
        candidate.expected_owner_generation,
        candidate.composite,
        candidate.composite_identity,
        candidate.composite_occurrence_ordinal,
        candidate.composite_producer,
        candidate.composite_producer_local_ordinal,
        candidate.composite_capability,
        candidate.composite_policy_fingerprint,
        candidate.composite_occurrence_profile_hash,
        candidate.composite_preload_follower,
        candidate.composite_outer_begin);
    return result;
}

SpmFmbSealedBuildTrace
FusedModelBase::seal_spm_pipeline_completed_build_trace(
    SpmFmbCompletedBuildTrace&& completed,
    const RpuKernelGraph& graph) {
    TORCH_CHECK(completed.payload_ != nullptr &&
                    completed.payload_->owner == this &&
                    completed.payload_->candidate.composite &&
                    completed.payload_->candidate.profile != nullptr,
                "composite BUILD seal received an empty or foreign "
                "completed occurrence");
    TORCH_CHECK(!pimpl_->pipeline_build_trace_.armed &&
                    g_fmb_build_trace_owner == nullptr &&
                    !g_fmb_build_trace_callback_begin.has_value() &&
                    !g_fmb_post_fn_yield_begin.has_value(),
                "composite BUILD seal requires the global trace slot free");
    std::unique_ptr<SpmFmbCompletedBuildTrace::Payload> payload =
        std::move(completed.payload_);
    pimpl_->pipeline_build_trace_ = std::move(payload->candidate);
    g_fmb_build_trace_owner = pimpl_.get();
    auto clear_on_early_failure = c10::make_scope_exit(
        [&] { clear_pipeline_build_trace_noexcept(*pimpl_); });
    const SpmFmbResolvedExecutionProfile profile =
        *pimpl_->pipeline_build_trace_.profile;
    clear_on_early_failure.release();
    return seal_spm_pipeline_build_trace(profile, graph);
}

void FusedModelBase::validate_spm_pipeline_build_trace(
    const SpmFmbSealedBuildTrace& trace,
    const RpuKernelGraph& graph) const {
    const std::shared_ptr<const uint64_t> owner =
        trace.owner_generation_.lock();
    TORCH_CHECK(owner != nullptr &&
                    owner.get() ==
                        pimpl_->pipeline_manifest_generation_.get() &&
                    *owner == trace.expected_owner_generation_,
                "schema-v8 BUILD trace validation: owner is stale");
    TORCH_CHECK(pimpl_->pipeline_manifest_snapshot_.valid &&
                    pimpl_->pipeline_manifest_snapshot_.allocation_hash ==
                        trace.allocation_hash_,
                "schema-v8 BUILD trace validation: allocation identity "
                "drift");
    if (trace.composite_ &&
        trace.composite_capability_ !=
            SpmFmbCompositeOccurrenceCapability::Unsealed) {
        const auto& snapshot = pimpl_->pipeline_manifest_snapshot_;
        TORCH_CHECK(
            trace.composite_producer_ &&
                trace.composite_producer_local_ordinal_ < 3 &&
                trace.composite_capability_ ==
                    SpmFmbCompositeOccurrenceCapability::
                        StableThreeInputSlotsSameOwnerPreload &&
                trace.composite_capability_ ==
                    spm_fmb_composite_occurrence_capability() &&
                trace.composite_policy_fingerprint_ != 0 &&
                trace.composite_policy_fingerprint_ ==
                    spm_fmb_composite_occurrence_policy_fingerprint() &&
                trace.composite_preload_follower_ ==
                    (trace.composite_producer_local_ordinal_ != 0) &&
                trace.composite_occurrence_profile_hash_ ==
                    fmb_composite_occurrence_profile_hash(
                        trace.profile_hash_, snapshot.layout_hash,
                        snapshot.allocation_hash,
                        snapshot.declaration_hash,
                        snapshot.persistent_hash,
                        spm_fmb_preload_capability(),
                        trace.composite_capability_,
                        trace.composite_policy_fingerprint_),
            "composite BUILD trace validation: occurrence policy drift");
    }
    TORCH_CHECK(
        trace.post_fn_yield_capability_ ==
                spm_fmb_post_fn_yield_capability() &&
            trace.post_fn_yield_policy_fingerprint_ ==
                spm_fmb_post_fn_yield_policy_fingerprint() &&
            trace.layer_producer_yield_capability_ ==
                spm_fmb_layer_producer_yield_capability() &&
            trace.layer_producer_yield_policy_fingerprint_ ==
                spm_fmb_layer_producer_yield_policy_fingerprint(),
        "producer-yield BUILD trace validation: capability or "
        "policy fingerprint drift");
    TORCH_CHECK(trace.active_group_capability_ ==
                    spm_fmb_active_group_capability(),
                "schema-v9 BUILD trace validation: active-group capability "
                "drift");
    TORCH_CHECK(trace.dense_ddr_member_capability_ ==
                    spm_fmb_dense_ddr_member_capability(),
                "schema-v10 BUILD trace validation: dense-DDR member "
                "capability drift");
    TORCH_CHECK(trace.dma_endpoint_capability_ ==
                    spm_fmb_dma_endpoint_capability(),
                "schema-v11 BUILD trace validation: DMA endpoint "
                "capability drift");
    TORCH_CHECK(trace.runtime_member_capability_ ==
                    spm_fmb_runtime_member_capability(),
                "schema-v12 BUILD trace validation: runtime member "
                "capability drift");
    TORCH_CHECK(trace.spm_peer_capability_ ==
                    spm_fmb_spm_peer_capability(),
                "schema-v13 BUILD trace validation: typed SPM peer "
                "capability drift");
    TORCH_CHECK(trace.spm_peer_policy_fingerprint_ ==
                    spm_fmb_spm_peer_policy_fingerprint(),
                "schema-v13 BUILD trace validation: typed SPM peer policy "
                "fingerprint drift");
    TORCH_CHECK(
        trace.dense_ddr_member_capability_ ==
                SpmFmbDenseDdrMemberCapability::Unsealed ||
            (trace.resolved_hidden_size_ == pimpl_->hidden_size_ &&
             trace.resolved_num_layers_ == pimpl_->num_layers_),
        "schema-v10 BUILD trace validation: resolved policy/shape drift");
    TORCH_CHECK(trace.graph_identity_ == &graph &&
                    graph.state() == RpuKernelGraph::State::BUILT &&
                    graph.build_generation() ==
                        trace.graph_build_generation_ &&
                    graph.build_topology_hash() ==
                        trace.graph_topology_hash_ &&
                    graph.build_node_kind_mask() ==
                        trace.graph_node_kind_mask_ &&
                    (trace.graph_node_kind_mask_ &
                     kSpmFmbOpaqueGraphNodeKinds) == 0 &&
                    graph.built_signature().segment_key ==
                        trace.graph_signature_segment_key_ &&
                    graph.built_signature_identity() ==
                        trace.graph_signature_identity_,
                "schema-v8 BUILD trace validation: Graph identity is stale");
}

SpmFmbSealedPostFnYields
FusedModelBase::seal_spm_pipeline_post_fn_yields(
    const SpmFmbSealedBuildTrace& trace,
    const RpuKernelGraph& graph) const {
    validate_spm_pipeline_build_trace(trace, graph);
    const SpmFmbPostFnYieldCapability capability =
        spm_fmb_post_fn_yield_capability();
    const uint64_t policy_fingerprint =
        spm_fmb_post_fn_yield_policy_fingerprint();
    TORCH_CHECK(
        capability ==
                SpmFmbPostFnYieldCapability::
                    CanonicalDenseReplicatedFp16 &&
            trace.post_fn_yield_capability_ == capability &&
            policy_fingerprint != 0 &&
            trace.post_fn_yield_policy_fingerprint_ ==
                policy_fingerprint,
        "post_fn producer-yield seal STOP: model has not opted into one "
        "stable canonical dense-FP16 policy");
    TORCH_CHECK(
        !trace.callback_summaries_.empty() &&
            trace.callback_summaries_.size() == trace.callback_count_ &&
            trace.graph_node_begin_ < trace.graph_node_end_,
        "post_fn producer-yield seal STOP: source callback payload is "
        "incomplete");

    const SpmFmbSealedBuildTrace::CallbackSummary* post_fn = nullptr;
    size_t post_fn_count = 0;
    for (size_t callback_index = 0;
         callback_index < trace.callback_summaries_.size();
         ++callback_index) {
        const auto& callback = trace.callback_summaries_[callback_index];
        if (callback.kind == SpmFmbOccurrenceKind::PostFn) {
            ++post_fn_count;
            post_fn = &callback;
            TORCH_CHECK(
                callback_index + 1 == trace.callback_summaries_.size(),
                "post_fn producer-yield seal STOP: PostFn must be the "
                "terminal resolved callback");
        } else {
            TORCH_CHECK(
                callback.producer_yields.empty(),
                "post_fn producer-yield seal STOP: a non-PostFn callback "
                "contains producer authority");
        }
    }
    TORCH_CHECK(
        post_fn_count == 1 && post_fn != nullptr &&
            !post_fn->producer_yields.empty() &&
            post_fn->graph_node_begin < post_fn->graph_node_end &&
            post_fn->graph_node_end == trace.graph_node_end_,
        "post_fn producer-yield seal STOP: expected one nonempty terminal "
        "PostFn callback");

    const uint64_t semantic_id_digest =
        graph.semantic_spm_producer_yield_id_digest_for_fmb();
    TORCH_CHECK(
        semantic_id_digest != 0,
        "post_fn producer-yield seal STOP: Graph marker digest is absent");

    std::vector<SpmFmbSealedPostFnYields::YieldSummary> yields;
    yields.reserve(post_fn->producer_yields.size());
    size_t expected_node_begin = post_fn->graph_node_begin;
    for (size_t ordinal = 0;
         ordinal < post_fn->producer_yields.size(); ++ordinal) {
        const auto& source = post_fn->producer_yields[ordinal];
        source.spec.validate();
        TORCH_CHECK(
            source.ordinal == ordinal && source.semantic_id != 0 &&
                source.writer_kind ==
                    SpmFmbTerminalWriterKind::AllReduceSumResidual &&
                source.spec.dtype == SpmPortDType::Fp16 &&
                source.spec.distribution ==
                    SpmPortDistribution::Replicated &&
                source.spec.cols % 16 == 0 &&
                source.logical_capacity == source.spec.storage_bytes() &&
                source.aligned_capacity == source.logical_capacity &&
                source.graph_node_begin == expected_node_begin &&
                source.graph_node_begin < source.graph_node_end &&
                source.graph_node_end <= post_fn->graph_node_end &&
                source.graph_node_kind_mask ==
                    graph_node_kind_bit(GraphNodeKind::Kernel) &&
                source.graph_topology_hash != 0,
            "post_fn producer-yield seal STOP: yield ", ordinal,
            " is not one ordered canonical terminal-kernel production");

        const size_t relative_node_begin =
            source.graph_node_begin - trace.graph_node_begin_;
        const size_t relative_node_end =
            source.graph_node_end - trace.graph_node_begin_;
        uint64_t semantic_hash = fmb_hash_u64(kFmbFnvOffset, 29);
        const std::vector<uint64_t> values = {
            ordinal,
            source.semantic_id,
            static_cast<uint8_t>(source.writer_kind),
            static_cast<uint8_t>(source.spec.dtype),
            static_cast<uint64_t>(source.spec.rows),
            static_cast<uint64_t>(source.spec.cols),
            static_cast<uint8_t>(source.spec.distribution),
            source.allocation_ordinal,
            source.declaration_ordinal,
            source.alias_root_ordinal,
            source.declaration_name_hash,
            source.alias_root_name_hash,
            source.root_offset,
            source.allocation_offset_from_root,
            source.logical_capacity,
            source.aligned_capacity,
            relative_node_begin,
            relative_node_end,
            source.graph_node_kind_mask,
            source.graph_topology_hash,
        };
        for (const uint64_t value : values) {
            semantic_hash = fmb_hash_u64(semantic_hash, value);
        }
        semantic_hash = fmb_nonzero_hash(semantic_hash);
        yields.push_back({
            ordinal,
            source.semantic_id,
            source.writer_kind,
            source.spec,
            source.allocation_ordinal,
            source.declaration_ordinal,
            source.alias_root_ordinal,
            source.declaration_name_hash,
            source.alias_root_name_hash,
            source.root_offset,
            source.allocation_offset_from_root,
            source.logical_capacity,
            source.aligned_capacity,
            relative_node_begin,
            relative_node_end,
            semantic_hash,
        });
        expected_node_begin = source.graph_node_end;
    }
    TORCH_CHECK(
        expected_node_begin == post_fn->graph_node_end,
        "post_fn producer-yield seal STOP: yield windows do not exactly "
        "cover the terminal PostFn callback");

    uint64_t yield_hash = fmb_hash_u64(kFmbFnvOffset, 29);
    const std::vector<uint64_t> identity_values = {
        trace.trace_hash_,
        trace.profile_hash_,
        trace.allocation_hash_,
        trace.graph_build_generation_,
        trace.graph_signature_identity_,
        trace.graph_signature_segment_key_,
        trace.graph_topology_hash_,
        static_cast<uint64_t>(capability),
        policy_fingerprint,
        semantic_id_digest,
        post_fn->semantic_hash,
        yields.size(),
    };
    for (const uint64_t value : identity_values) {
        yield_hash = fmb_hash_u64(yield_hash, value);
    }
    for (const auto& yield : yields) {
        yield_hash = fmb_hash_u64(yield_hash, yield.semantic_hash);
    }
    yield_hash = fmb_nonzero_hash(yield_hash);
    return SpmFmbSealedPostFnYields(
        trace, yield_hash, semantic_id_digest, capability,
        policy_fingerprint, yields);
}

void FusedModelBase::validate_spm_pipeline_post_fn_yields(
    const SpmFmbSealedPostFnYields& yields,
    const RpuKernelGraph& graph) const {
    const SpmFmbSealedPostFnYields expected =
        seal_spm_pipeline_post_fn_yields(yields.source_trace_, graph);
    bool summaries_equal =
        yields.yields_.size() == expected.yields_.size();
    if (summaries_equal) {
        for (size_t index = 0; index < yields.yields_.size(); ++index) {
            const auto& actual = yields.yields_[index];
            const auto& canonical = expected.yields_[index];
            if (actual.ordinal != canonical.ordinal ||
                actual.semantic_id != canonical.semantic_id ||
                actual.writer_kind != canonical.writer_kind ||
                actual.spec != canonical.spec ||
                actual.allocation_ordinal !=
                    canonical.allocation_ordinal ||
                actual.declaration_ordinal !=
                    canonical.declaration_ordinal ||
                actual.alias_root_ordinal !=
                    canonical.alias_root_ordinal ||
                actual.declaration_name_hash !=
                    canonical.declaration_name_hash ||
                actual.alias_root_name_hash !=
                    canonical.alias_root_name_hash ||
                actual.root_offset != canonical.root_offset ||
                actual.allocation_offset_from_root !=
                    canonical.allocation_offset_from_root ||
                actual.logical_capacity != canonical.logical_capacity ||
                actual.aligned_capacity != canonical.aligned_capacity ||
                actual.relative_node_begin !=
                    canonical.relative_node_begin ||
                actual.relative_node_end != canonical.relative_node_end ||
                actual.semantic_hash != canonical.semantic_hash) {
                summaries_equal = false;
                break;
            }
        }
    }
    TORCH_CHECK(
        yields.yield_hash_ == expected.yield_hash_ &&
            yields.semantic_id_digest_ ==
                expected.semantic_id_digest_ &&
            yields.capability_ == expected.capability_ &&
            yields.policy_fingerprint_ == expected.policy_fingerprint_ &&
            summaries_equal,
        "post_fn producer-yield validation: sealed payload drift");
}

void FusedModelBase::arm_spm_pipeline_post_fn_yield_replay(
    const SpmFmbSealedPostFnYields& yields,
    RpuKernelGraph& graph) const {
    const auto& trace = yields.source_trace_;
    const std::shared_ptr<uint64_t> owner =
        trace.owner_generation_.lock();
    TORCH_CHECK(
        graph.state() == RpuKernelGraph::State::REPLAYING &&
            trace.graph_identity_ == &graph && owner != nullptr &&
            owner.get() == pimpl_->pipeline_manifest_generation_.get() &&
            *owner == trace.expected_owner_generation_ &&
            pimpl_->pipeline_manifest_snapshot_.valid &&
            pimpl_->pipeline_manifest_snapshot_.allocation_hash ==
                trace.allocation_hash_ &&
            yields.capability_ ==
                SpmFmbPostFnYieldCapability::
                    CanonicalDenseReplicatedFp16 &&
            yields.capability_ == spm_fmb_post_fn_yield_capability() &&
            trace.post_fn_yield_capability_ == yields.capability_ &&
            yields.policy_fingerprint_ != 0 &&
            yields.policy_fingerprint_ ==
                spm_fmb_post_fn_yield_policy_fingerprint() &&
            trace.post_fn_yield_policy_fingerprint_ ==
                yields.policy_fingerprint_ &&
            yields.yield_hash_ != 0 && !yields.yields_.empty() &&
            yields.semantic_id_digest_ != 0 &&
            graph.build_generation() == trace.graph_build_generation_ &&
            graph.build_topology_hash() == trace.graph_topology_hash_ &&
            graph.built_signature_identity() ==
                trace.graph_signature_identity_ &&
            graph.built_signature().segment_key ==
                trace.graph_signature_segment_key_ &&
            graph.semantic_spm_producer_yield_id_digest_for_fmb() ==
                yields.semantic_id_digest_,
        "post_fn producer-yield replay arm: sealed model/owner/Graph "
        "identity is stale");
    TORCH_CHECK(
        !pimpl_->pipeline_post_fn_yield_replay_authority_.armed &&
            !pimpl_->pipeline_post_fn_yield_replay_authority_.yield_open &&
            pimpl_->pipeline_post_fn_yield_replay_authority_.yields ==
                nullptr,
        "post_fn producer-yield replay arm is duplicate");
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    TORCH_CHECK(stamp.position == 0 &&
                    stamp.build_generation ==
                        trace.graph_build_generation_ &&
                    stamp.signature_identity ==
                        trace.graph_signature_identity_ &&
                    stamp.signature_segment_key ==
                        trace.graph_signature_segment_key_,
                "post_fn producer-yield replay arm requires cursor zero");
    Impl::RuntimePostFnYieldReplayAuthority authority;
    authority.armed = true;
    authority.graph_identity = &graph;
    authority.graph_build_generation = stamp.build_generation;
    authority.graph_signature_identity = stamp.signature_identity;
    authority.profile_hash = stamp.signature_segment_key;
    authority.yields = std::unique_ptr<SpmFmbSealedPostFnYields>(
        new SpmFmbSealedPostFnYields(yields));
    graph.arm_semantic_spm_producer_yield_replay_for_fmb(
        owner, trace.expected_owner_generation_,
        yields.semantic_id_digest_);
    pimpl_->pipeline_post_fn_yield_replay_authority_ =
        std::move(authority);
}

void FusedModelBase::
drive_spm_pipeline_post_fn_yield_replay_for_cpu_contract(
    const SpmFmbSealedPostFnYields& yields) {
    auto& authority =
        pimpl_->pipeline_post_fn_yield_replay_authority_;
    const auto& trace = yields.source_trace_;
    TORCH_CHECK(
        trace.cpu_dry_ && authority.armed &&
            authority.yields != nullptr &&
            authority.yields->yield_hash_ == yields.yield_hash_ &&
            authority.next_yield == 0 && !authority.yield_open,
        "post_fn producer-yield CPU REPLAY driver requires one freshly "
        "armed CPU-dry token");
    TORCH_CHECK(
        g_fmb_cpu_yield_replay_owner == nullptr,
        "post_fn producer-yield CPU REPLAY driver is nested");
    g_fmb_cpu_yield_replay_owner = pimpl_.get();
    auto clear_cpu_replay_owner = c10::make_scope_exit(
        [&] { g_fmb_cpu_yield_replay_owner = nullptr; });
    auto clear_failed_authority = c10::make_scope_exit([&] {
        if (authority.armed) {
            if (RpuKernelGraph::has_active()) {
                RpuKernelGraph::active()
                    .poison_semantic_spm_producer_yield_for_fmb();
            }
            authority = {};
            g_fmb_post_fn_yield_begin.reset();
        }
    });
    const ModelStaticConfig static_cfg = static_config();
    TORCH_CHECK(
        static_cfg.post_fn != nullptr &&
            static_cfg.post_output_shape.empty() &&
            trace.callback_summaries_.size() == trace.callback_count_,
        "post_fn producer-yield CPU REPLAY driver callback contract "
        "drifted");
    for (const auto& callback : trace.callback_summaries_) {
        if (callback.kind == SpmFmbOccurrenceKind::PostFn) {
            TORCH_CHECK(
                &callback == &trace.callback_summaries_.back(),
                "post_fn producer-yield CPU REPLAY driver encountered a "
                "nonterminal PostFn");
            std::invoke(static_cfg.post_fn, *this);
            continue;
        }
        TORCH_CHECK(
            callback.kind == SpmFmbOccurrenceKind::LayerBody &&
                callback.chunk_index >= 0,
            "post_fn producer-yield CPU REPLAY driver encountered an "
            "unsupported resolved callback");
        ChunkInfo chunk{
            callback.chunk_index, callback.chunk_offset,
            callback.chunk_len, callback.chunk_kv_seq_len};
        pimpl_->ctx_.body_iter = callback.body_id;
        build_layer_subgraph(callback.layer, chunk);
    }
    TORCH_CHECK(
        authority.armed && !authority.yield_open &&
            authority.yields != nullptr &&
            authority.next_yield == authority.yields->yields_.size(),
        "post_fn producer-yield CPU REPLAY driver omitted a sealed yield");
    authority = {};
    clear_failed_authority.release();
}

void FusedModelBase::
cancel_spm_pipeline_post_fn_yield_replay() noexcept {
    auto& authority =
        pimpl_->pipeline_post_fn_yield_replay_authority_;
    if (authority.armed && RpuKernelGraph::has_active() &&
        &RpuKernelGraph::active() == authority.graph_identity) {
        RpuKernelGraph::active()
            .poison_semantic_spm_producer_yield_for_fmb();
    }
    authority = {};
    g_fmb_post_fn_yield_begin.reset();
}

SpmFmbSealedCallbackYields
FusedModelBase::seal_spm_pipeline_callback_yields(
    const SpmFmbSealedBuildTrace& trace,
    const RpuKernelGraph& graph) const {
    validate_spm_pipeline_build_trace(trace, graph);
    const SpmFmbLayerProducerYieldCapability capability =
        spm_fmb_layer_producer_yield_capability();
    const uint64_t policy_fingerprint =
        spm_fmb_layer_producer_yield_policy_fingerprint();
    TORCH_CHECK(
        capability ==
                SpmFmbLayerProducerYieldCapability::
                    CanonicalDenseReplicatedFp16 &&
            trace.layer_producer_yield_capability_ == capability &&
            trace.post_fn_yield_capability_ ==
                SpmFmbPostFnYieldCapability::Unsealed &&
            policy_fingerprint != 0 &&
            trace.layer_producer_yield_policy_fingerprint_ ==
                policy_fingerprint,
        "callback producer-yield seal STOP: model has not opted into one "
        "stable canonical dense-FP16 callback-window policy");
    TORCH_CHECK(
        !trace.callback_summaries_.empty() &&
            trace.callback_summaries_.size() == trace.callback_count_ &&
            trace.graph_node_begin_ < trace.graph_node_end_,
        "callback producer-yield seal STOP: source callback payload is "
        "incomplete");

    const uint64_t semantic_id_digest =
        graph.semantic_spm_producer_yield_id_digest_for_fmb();
    TORCH_CHECK(
        semantic_id_digest != 0,
        "callback producer-yield seal STOP: Graph marker digest is absent");

    std::vector<SpmFmbSealedCallbackYields::YieldSummary> yields;
    size_t layer_yield_count = 0;
    bool saw_post_fn = false;
    size_t expected_ordinal = 0;
    for (size_t callback_index = 0;
         callback_index < trace.callback_summaries_.size();
         ++callback_index) {
        const auto& callback = trace.callback_summaries_[callback_index];
        if (callback.kind == SpmFmbOccurrenceKind::PostFn) {
            TORCH_CHECK(
                !saw_post_fn &&
                    callback_index + 1 ==
                        trace.callback_summaries_.size(),
                "callback producer-yield seal STOP: PostFn must be unique "
                "and terminal");
            saw_post_fn = true;
        }
        TORCH_CHECK(
            callback.kind == SpmFmbOccurrenceKind::LayerBody ||
                callback.kind == SpmFmbOccurrenceKind::PostFn ||
                callback.producer_yields.empty(),
            "callback producer-yield seal STOP: only LayerBody/PostFn may "
            "carry producer authority");

        size_t previous_yield_end = callback.graph_node_begin;
        for (const auto& source : callback.producer_yields) {
            source.spec.validate();
            TORCH_CHECK(
                source.ordinal == expected_ordinal &&
                    source.semantic_id != 0 &&
                    source.writer_kind ==
                        SpmFmbTerminalWriterKind::
                            AllReduceSumResidual &&
                    source.spec.dtype == SpmPortDType::Fp16 &&
                    source.spec.distribution ==
                        SpmPortDistribution::Replicated &&
                    source.spec.cols % 16 == 0 &&
                    source.logical_capacity ==
                        source.spec.storage_bytes() &&
                    source.aligned_capacity ==
                        source.logical_capacity &&
                    source.graph_node_begin >= previous_yield_end &&
                    source.graph_node_begin < source.graph_node_end &&
                    source.graph_node_end <= callback.graph_node_end &&
                    source.graph_node_kind_mask ==
                        graph_node_kind_bit(GraphNodeKind::Kernel) &&
                    source.graph_topology_hash != 0,
                "callback producer-yield seal STOP: yield ",
                expected_ordinal,
                " is not one ordered canonical terminal-kernel window");

            const size_t relative_node_begin =
                source.graph_node_begin - trace.graph_node_begin_;
            const size_t relative_node_end =
                source.graph_node_end - trace.graph_node_begin_;
            uint64_t semantic_hash = fmb_hash_u64(kFmbFnvOffset, 32);
            const std::vector<uint64_t> values = {
                expected_ordinal,
                callback_index,
                static_cast<uint8_t>(callback.kind),
                callback.key,
                source.semantic_id,
                static_cast<uint8_t>(source.writer_kind),
                static_cast<uint8_t>(source.spec.dtype),
                static_cast<uint64_t>(source.spec.rows),
                static_cast<uint64_t>(source.spec.cols),
                static_cast<uint8_t>(source.spec.distribution),
                source.allocation_ordinal,
                source.declaration_ordinal,
                source.alias_root_ordinal,
                source.declaration_name_hash,
                source.alias_root_name_hash,
                source.root_offset,
                source.allocation_offset_from_root,
                source.logical_capacity,
                source.aligned_capacity,
                relative_node_begin,
                relative_node_end,
                source.graph_node_kind_mask,
                source.graph_topology_hash,
            };
            for (const uint64_t value : values) {
                semantic_hash = fmb_hash_u64(semantic_hash, value);
            }
            semantic_hash = fmb_nonzero_hash(semantic_hash);
            yields.push_back({
                expected_ordinal,
                callback_index,
                callback.kind,
                callback.key,
                source.semantic_id,
                source.writer_kind,
                source.spec,
                source.allocation_ordinal,
                source.declaration_ordinal,
                source.alias_root_ordinal,
                source.declaration_name_hash,
                source.alias_root_name_hash,
                source.root_offset,
                source.allocation_offset_from_root,
                source.logical_capacity,
                source.aligned_capacity,
                relative_node_begin,
                relative_node_end,
                semantic_hash,
            });
            if (callback.kind == SpmFmbOccurrenceKind::LayerBody) {
                ++layer_yield_count;
            }
            previous_yield_end = source.graph_node_end;
            ++expected_ordinal;
        }
    }
    TORCH_CHECK(
        !yields.empty() && layer_yield_count > 0,
        "callback producer-yield seal STOP: expected at least one canonical "
        "LayerBody terminal writer");

    uint64_t yield_hash = fmb_hash_u64(kFmbFnvOffset, 32);
    const std::vector<uint64_t> identity_values = {
        trace.trace_hash_,
        trace.profile_hash_,
        trace.allocation_hash_,
        trace.graph_build_generation_,
        trace.graph_signature_identity_,
        trace.graph_signature_segment_key_,
        trace.graph_topology_hash_,
        static_cast<uint64_t>(capability),
        policy_fingerprint,
        semantic_id_digest,
        layer_yield_count,
        yields.size(),
    };
    for (const uint64_t value : identity_values) {
        yield_hash = fmb_hash_u64(yield_hash, value);
    }
    for (const auto& yield : yields) {
        yield_hash = fmb_hash_u64(yield_hash, yield.semantic_hash);
    }
    yield_hash = fmb_nonzero_hash(yield_hash);
    return SpmFmbSealedCallbackYields(
        trace, yield_hash, semantic_id_digest, capability,
        policy_fingerprint, yields);
}

void FusedModelBase::validate_spm_pipeline_callback_yields(
    const SpmFmbSealedCallbackYields& yields,
    const RpuKernelGraph& graph) const {
    const SpmFmbSealedCallbackYields expected =
        seal_spm_pipeline_callback_yields(yields.source_trace_, graph);
    bool summaries_equal =
        yields.yields_.size() == expected.yields_.size();
    if (summaries_equal) {
        for (size_t index = 0; index < yields.yields_.size(); ++index) {
            const auto& actual = yields.yields_[index];
            const auto& canonical = expected.yields_[index];
            if (actual.ordinal != canonical.ordinal ||
                actual.callback_index != canonical.callback_index ||
                actual.callback_kind != canonical.callback_kind ||
                actual.callback_key != canonical.callback_key ||
                actual.semantic_id != canonical.semantic_id ||
                actual.writer_kind != canonical.writer_kind ||
                actual.spec != canonical.spec ||
                actual.allocation_ordinal !=
                    canonical.allocation_ordinal ||
                actual.declaration_ordinal !=
                    canonical.declaration_ordinal ||
                actual.alias_root_ordinal !=
                    canonical.alias_root_ordinal ||
                actual.declaration_name_hash !=
                    canonical.declaration_name_hash ||
                actual.alias_root_name_hash !=
                    canonical.alias_root_name_hash ||
                actual.root_offset != canonical.root_offset ||
                actual.allocation_offset_from_root !=
                    canonical.allocation_offset_from_root ||
                actual.logical_capacity != canonical.logical_capacity ||
                actual.aligned_capacity != canonical.aligned_capacity ||
                actual.relative_node_begin !=
                    canonical.relative_node_begin ||
                actual.relative_node_end != canonical.relative_node_end ||
                actual.semantic_hash != canonical.semantic_hash) {
                summaries_equal = false;
                break;
            }
        }
    }
    TORCH_CHECK(
        yields.yield_hash_ == expected.yield_hash_ &&
            yields.semantic_id_digest_ ==
                expected.semantic_id_digest_ &&
            yields.capability_ == expected.capability_ &&
            yields.policy_fingerprint_ ==
                expected.policy_fingerprint_ &&
            summaries_equal,
        "callback producer-yield validation: sealed payload drift");
}

void FusedModelBase::arm_spm_pipeline_callback_yield_replay(
    const SpmFmbSealedCallbackYields& yields,
    RpuKernelGraph& graph) const {
    const auto& trace = yields.source_trace_;
    const std::shared_ptr<uint64_t> owner =
        trace.owner_generation_.lock();
    TORCH_CHECK(
        graph.state() == RpuKernelGraph::State::REPLAYING &&
            trace.graph_identity_ == &graph && owner != nullptr &&
            owner.get() ==
                pimpl_->pipeline_manifest_generation_.get() &&
            *owner == trace.expected_owner_generation_ &&
            pimpl_->pipeline_manifest_snapshot_.valid &&
            pimpl_->pipeline_manifest_snapshot_.allocation_hash ==
                trace.allocation_hash_ &&
            yields.capability_ ==
                SpmFmbLayerProducerYieldCapability::
                    CanonicalDenseReplicatedFp16 &&
            yields.capability_ ==
                spm_fmb_layer_producer_yield_capability() &&
            trace.layer_producer_yield_capability_ ==
                yields.capability_ &&
            yields.policy_fingerprint_ != 0 &&
            yields.policy_fingerprint_ ==
                spm_fmb_layer_producer_yield_policy_fingerprint() &&
            trace.layer_producer_yield_policy_fingerprint_ ==
                yields.policy_fingerprint_ &&
            yields.yield_hash_ != 0 && !yields.yields_.empty() &&
            yields.semantic_id_digest_ != 0 &&
            graph.build_generation() == trace.graph_build_generation_ &&
            graph.build_topology_hash() == trace.graph_topology_hash_ &&
            graph.built_signature_identity() ==
                trace.graph_signature_identity_ &&
            graph.built_signature().segment_key ==
                trace.graph_signature_segment_key_ &&
            graph.semantic_spm_producer_yield_id_digest_for_fmb() ==
                yields.semantic_id_digest_,
        "callback producer-yield replay arm: sealed model/owner/Graph "
        "identity is stale");
    TORCH_CHECK(
        !pimpl_->pipeline_callback_yield_replay_authority_.armed &&
            !pimpl_->pipeline_callback_yield_replay_authority_.yield_open &&
            pimpl_->pipeline_callback_yield_replay_authority_.yields ==
                nullptr &&
            !pimpl_->pipeline_post_fn_yield_replay_authority_.armed,
        "callback producer-yield replay arm is duplicate or conflicts with "
        "legacy PostFn authority");
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    TORCH_CHECK(
        stamp.position == 0 &&
            stamp.build_generation == trace.graph_build_generation_ &&
            stamp.signature_identity == trace.graph_signature_identity_ &&
            stamp.signature_segment_key ==
                trace.graph_signature_segment_key_,
        "callback producer-yield replay arm requires cursor zero");
    Impl::RuntimeCallbackYieldReplayAuthority authority;
    authority.armed = true;
    authority.graph_identity = &graph;
    authority.graph_build_generation = stamp.build_generation;
    authority.graph_signature_identity = stamp.signature_identity;
    authority.profile_hash = stamp.signature_segment_key;
    authority.yields =
        std::unique_ptr<SpmFmbSealedCallbackYields>(
            new SpmFmbSealedCallbackYields(yields));
    graph.arm_semantic_spm_producer_yield_replay_for_fmb(
        owner, trace.expected_owner_generation_,
        yields.semantic_id_digest_);
    pimpl_->pipeline_callback_yield_replay_authority_ =
        std::move(authority);
}

void FusedModelBase::
drive_spm_pipeline_callback_yield_replay_for_cpu_contract(
    const SpmFmbSealedCallbackYields& yields) {
    auto& authority =
        pimpl_->pipeline_callback_yield_replay_authority_;
    const auto& trace = yields.source_trace_;
    TORCH_CHECK(
        trace.cpu_dry_ && authority.armed &&
            authority.yields != nullptr &&
            authority.yields->yield_hash_ == yields.yield_hash_ &&
            authority.next_yield == 0 && !authority.yield_open,
        "callback producer-yield CPU REPLAY driver requires one freshly "
        "armed CPU-dry token");
    TORCH_CHECK(
        g_fmb_cpu_yield_replay_owner == nullptr,
        "callback producer-yield CPU REPLAY driver is nested");
    g_fmb_cpu_yield_replay_owner = pimpl_.get();
    auto clear_cpu_replay_owner = c10::make_scope_exit(
        [&] { g_fmb_cpu_yield_replay_owner = nullptr; });
    auto clear_failed_authority = c10::make_scope_exit([&] {
        if (authority.armed) {
            if (RpuKernelGraph::has_active()) {
                RpuKernelGraph::active()
                    .poison_semantic_spm_producer_yield_for_fmb();
            }
            authority = {};
            g_fmb_post_fn_yield_begin.reset();
        }
    });
    const ModelStaticConfig static_cfg = static_config();
    TORCH_CHECK(
        static_cfg.pre_layers_fn == nullptr &&
            static_cfg.post_layers_fn == nullptr &&
            static_cfg.post_output_shape.empty() &&
            trace.callback_summaries_.size() == trace.callback_count_,
        "callback producer-yield CPU REPLAY driver callback contract "
        "drifted");
    for (const auto& callback : trace.callback_summaries_) {
        if (callback.kind == SpmFmbOccurrenceKind::PostFn) {
            TORCH_CHECK(
                &callback == &trace.callback_summaries_.back() &&
                    static_cfg.post_fn != nullptr,
                "callback producer-yield CPU REPLAY driver encountered an "
                "invalid PostFn");
            std::invoke(static_cfg.post_fn, *this);
            continue;
        }
        TORCH_CHECK(
            callback.chunk_index >= 0,
            "callback producer-yield CPU REPLAY driver callback lacks "
            "chunk geometry");
        ChunkInfo chunk{
            callback.chunk_index, callback.chunk_offset,
            callback.chunk_len, callback.chunk_kv_seq_len};
        pimpl_->ctx_.body_iter = callback.body_id;
        if (callback.kind == SpmFmbOccurrenceKind::KvInsertBody) {
            TORCH_CHECK(
                static_cfg.kv_first_fn != nullptr,
                "callback producer-yield CPU REPLAY driver is missing the "
                "KV callback");
            std::invoke(
                static_cfg.kv_first_fn, *this, callback.layer, chunk);
        } else {
            TORCH_CHECK(
                callback.kind == SpmFmbOccurrenceKind::LayerBody,
                "callback producer-yield CPU REPLAY driver encountered an "
                "unsupported callback");
            build_layer_subgraph(callback.layer, chunk);
        }
    }
    TORCH_CHECK(
        authority.armed && !authority.yield_open &&
            authority.yields != nullptr &&
            authority.next_yield == authority.yields->yields_.size(),
        "callback producer-yield CPU REPLAY driver omitted a sealed yield");
    authority = {};
    clear_failed_authority.release();
}

void FusedModelBase::
drive_spm_pipeline_composite_callback_yield_replay_for_cpu_contract() {
    const auto& authority =
        pimpl_->pipeline_callback_yield_replay_authority_;
    TORCH_CHECK(authority.armed && authority.yields != nullptr &&
                    authority.yields->source_trace_.composite_,
                "composite callback-yield CPU REPLAY driver has no "
                "installed occurrence authority");
    const SpmFmbSealedCallbackYields yields = *authority.yields;
    drive_spm_pipeline_callback_yield_replay_for_cpu_contract(yields);
}

void FusedModelBase::
cancel_spm_pipeline_callback_yield_replay() noexcept {
    auto& authority =
        pimpl_->pipeline_callback_yield_replay_authority_;
    if (authority.armed && RpuKernelGraph::has_active() &&
        &RpuKernelGraph::active() == authority.graph_identity) {
        RpuKernelGraph::active()
            .poison_semantic_spm_producer_yield_for_fmb();
    }
    authority = {};
    g_fmb_post_fn_yield_begin.reset();
}

SpmFmbSealedActiveGroups FusedModelBase::seal_spm_pipeline_active_groups(
    const SpmFmbSealedBuildTrace& trace,
    const RpuKernelGraph& graph) const {
    validate_spm_pipeline_build_trace(trace, graph);
    const SpmFmbActiveGroupCapability capability =
        spm_fmb_active_group_capability();
    TORCH_CHECK(
        capability ==
                SpmFmbActiveGroupCapability::CanonicalActiveFirstChunkRun &&
            trace.active_group_capability_ == capability,
        "schema-v9 active-group STOP: model has not opted into the canonical "
        "active-first chunk-run contract");
    TORCH_CHECK(trace.cpu_dry_,
                "schema-v9 active-group STOP: this foundation accepts only a "
                "CPU-dry BUILD trace");
    const auto& callbacks = trace.callback_summaries_;
    TORCH_CHECK(!callbacks.empty() &&
                    callbacks.size() == trace.callback_count_ &&
                    trace.graph_node_begin_ <= trace.graph_node_end_ &&
                    trace.graph_node_end_ <= graph.graph_size(),
                "schema-v9 active-group STOP: source callback payload is "
                "incomplete");

    auto same_group_key = [](
        const SpmFmbSealedBuildTrace::CallbackSummary& lhs,
        const SpmFmbSealedBuildTrace::CallbackSummary& rhs) {
        return lhs.kind == rhs.kind &&
            lhs.body_id == rhs.body_id &&
            lhs.group_id == rhs.group_id &&
            lhs.layer == rhs.layer &&
            lhs.scope_mask == rhs.scope_mask &&
            lhs.input_in_spm == rhs.input_in_spm &&
            lhs.output_to_spm == rhs.output_to_spm &&
            lhs.local_phase_begin == rhs.local_phase_begin &&
            lhs.local_phase_end == rhs.local_phase_end;
    };
    auto is_active = [](
        const SpmFmbSealedBuildTrace::CallbackSummary& callback) {
        return callback.graph_node_begin < callback.graph_node_end &&
            callback.graph_node_kind_mask != 0 &&
            callback.graph_topology_hash != 0 &&
            callback.access_count != 0;
    };
    auto is_noop = [](
        const SpmFmbSealedBuildTrace::CallbackSummary& callback) {
        return callback.graph_node_begin == callback.graph_node_end &&
            callback.graph_node_kind_mask == 0 &&
            callback.access_count == 0;
    };

    std::vector<SpmFmbSealedActiveGroups::GroupSummary> group_summaries;
    group_summaries.reserve(callbacks.size() / 2);
    size_t callback_index = 0;
    size_t expected_run_members = 0;
    size_t active_callbacks = 0;
    size_t noop_callbacks = 0;
    size_t previous_graph_end = trace.graph_node_begin_;
    while (callback_index < callbacks.size()) {
        const size_t first_index = callback_index;
        const auto& first = callbacks[first_index];
        size_t end_index = first_index + 1;
        while (end_index < callbacks.size() &&
               same_group_key(first, callbacks[end_index])) {
            ++end_index;
        }
        const size_t member_count = end_index - first_index;
        TORCH_CHECK(member_count >= 2,
                    "schema-v9 active-group STOP: every active-first chunk "
                    "run requires at least one no-op follower");
        if (expected_run_members == 0) {
            expected_run_members = member_count;
        } else {
            TORCH_CHECK(member_count == expected_run_members,
                        "schema-v9 active-group STOP: resolved chunk-run "
                        "length drift across groups");
        }
        TORCH_CHECK(first.kind == SpmFmbOccurrenceKind::LayerBody &&
                        first.scope_mask == kLayerComputeScopes &&
                        first.chunk_index == 0 && first.chunk_offset == 0 &&
                        first.chunk_len > 0 &&
                        first.chunk_kv_seq_len >= first.chunk_len &&
                        first.updates_io_context &&
                        first.local_phase_begin <= first.local_phase_end &&
                        first.graph_node_begin == previous_graph_end &&
                        is_active(first) && first.semantic_hash != 0,
                    "schema-v9 active-group STOP: the first callback is not "
                    "one canonical active chunk carrier");

        uint64_t one_group_hash = fmb_hash_u64(kFmbFnvOffset, 13);
        one_group_hash = fmb_hash_u64(
            one_group_hash, group_summaries.size());
        one_group_hash = fmb_hash_u64(one_group_hash, first_index);
        one_group_hash = fmb_hash_u64(one_group_hash, member_count);
        const int64_t position_base =
            first.chunk_kv_seq_len - first.chunk_len;
        for (size_t member = 0; member < member_count; ++member) {
            const auto& callback = callbacks[first_index + member];
            const int64_t expected_offset =
                static_cast<int64_t>(member) * first.chunk_len;
            TORCH_CHECK(
                callback.key == first.key + member &&
                    callback.traversal_ordinal ==
                        first.traversal_ordinal + member &&
                    callback.chunk_index == static_cast<int>(member) &&
                    callback.chunk_offset == expected_offset &&
                    callback.chunk_len > 0 &&
                    (member + 1 == member_count
                         ? callback.chunk_len <= first.chunk_len
                         : callback.chunk_len == first.chunk_len) &&
                    callback.chunk_kv_seq_len ==
                        position_base + callback.chunk_offset +
                            callback.chunk_len &&
                    callback.graph_node_begin == previous_graph_end &&
                    callback.semantic_hash != 0 &&
                    (member == 0
                         ? is_active(callback)
                         : (!callback.updates_io_context &&
                            is_noop(callback))),
                "schema-v9 active-group STOP: member order, ChunkInfo, Graph "
                "continuity, or active/no-op evidence drift");
            if (member != 0) {
                const auto& previous = callbacks[first_index + member - 1];
                const uint64_t phase_width = static_cast<uint64_t>(
                    previous.local_phase_end -
                    previous.local_phase_begin);
                TORCH_CHECK(
                    static_cast<uint64_t>(previous.global_origin) +
                            phase_width + 1 ==
                        callback.global_origin,
                    "schema-v9 active-group STOP: global callback chronology "
                    "is not contiguous");
                ++noop_callbacks;
            } else {
                ++active_callbacks;
            }
            previous_graph_end = callback.graph_node_end;
            one_group_hash = fmb_hash_u64(one_group_hash, member);
            one_group_hash = fmb_hash_u64(
                one_group_hash, callback.semantic_hash);
        }
        one_group_hash = fmb_nonzero_hash(one_group_hash);
        group_summaries.push_back(
            {first_index, member_count, one_group_hash});
        callback_index = end_index;
    }

    TORCH_CHECK(callback_index == callbacks.size() &&
                    previous_graph_end == trace.graph_node_end_ &&
                    active_callbacks == trace.active_callback_count_ &&
                    noop_callbacks == trace.noop_callback_count_ &&
                    active_callbacks == group_summaries.size() &&
                    active_callbacks + noop_callbacks == callbacks.size(),
                "schema-v9 active-group STOP: groups do not cover the sealed "
                "component callback interval exactly once");

    uint64_t group_hash = fmb_hash_u64(kFmbFnvOffset, 12);
    group_hash = fmb_hash_u64(
        group_hash, static_cast<uint8_t>(capability));
    group_hash = fmb_hash_u64(group_hash, trace.trace_hash_);
    group_hash = fmb_hash_u64(group_hash, trace.profile_hash_);
    group_hash = fmb_hash_u64(group_hash, trace.allocation_hash_);
    group_hash = fmb_hash_u64(group_hash, trace.graph_node_begin_);
    group_hash = fmb_hash_u64(group_hash, trace.graph_node_end_);
    group_hash = fmb_hash_u64(group_hash, group_summaries.size());
    group_hash = fmb_hash_u64(group_hash, callbacks.size());
    group_hash = fmb_hash_u64(group_hash, active_callbacks);
    group_hash = fmb_hash_u64(group_hash, noop_callbacks);
    for (const auto& group : group_summaries) {
        group_hash = fmb_hash_u64(group_hash, group.first_callback);
        group_hash = fmb_hash_u64(group_hash, group.member_count);
        group_hash = fmb_hash_u64(group_hash, group.semantic_hash);
    }
    group_hash = fmb_nonzero_hash(group_hash);
    return SpmFmbSealedActiveGroups(
        trace, group_hash, group_summaries.size(), callbacks.size(),
        active_callbacks, noop_callbacks,
        trace.graph_node_end_ - trace.graph_node_begin_, capability,
        group_summaries);
}

void FusedModelBase::validate_spm_pipeline_active_groups(
    const SpmFmbSealedActiveGroups& groups,
    const RpuKernelGraph& graph) const {
    const SpmFmbSealedActiveGroups expected =
        seal_spm_pipeline_active_groups(groups.source_trace_, graph);
    bool summaries_equal =
        groups.group_summaries_.size() ==
        expected.group_summaries_.size();
    if (summaries_equal) {
        for (size_t index = 0;
             index < groups.group_summaries_.size(); ++index) {
            const auto& actual = groups.group_summaries_[index];
            const auto& canonical = expected.group_summaries_[index];
            if (actual.first_callback != canonical.first_callback ||
                actual.member_count != canonical.member_count ||
                actual.semantic_hash != canonical.semantic_hash) {
                summaries_equal = false;
                break;
            }
        }
    }
    TORCH_CHECK(
        groups.capability_ == expected.capability_ &&
            groups.group_hash_ == expected.group_hash_ &&
            groups.group_count_ == expected.group_count_ &&
            groups.member_callback_count_ ==
                expected.member_callback_count_ &&
            groups.active_callback_count_ ==
                expected.active_callback_count_ &&
            groups.noop_callback_count_ ==
                expected.noop_callback_count_ &&
            groups.component_node_count_ ==
                expected.component_node_count_ &&
            summaries_equal,
        "schema-v9 active-group validation: sealed payload drift");
}

SpmFmbSealedDenseDdrMembers
FusedModelBase::seal_spm_pipeline_dense_ddr_members(
    const SpmFmbSealedActiveGroups& groups,
    const RpuKernelGraph& graph) const {
    validate_spm_pipeline_active_groups(groups, graph);
    const SpmFmbDenseDdrMemberCapability capability =
        spm_fmb_dense_ddr_member_capability();
    TORCH_CHECK(
        capability == SpmFmbDenseDdrMemberCapability::
                IndependentLocalSequentialDdrPingPong &&
            groups.source_trace_.dense_ddr_member_capability_ == capability,
        "schema-v10 dense-DDR members STOP: model has not opted into the "
        "independent-local sequential DDR-ping-pong contract");
    TORCH_CHECK(groups.cpu_dry(),
                "schema-v10 dense-DDR members STOP: this foundation accepts "
                "only CPU-dry groups");

    const auto& source_trace = groups.source_trace_;
    const auto& callbacks = source_trace.callback_summaries_;
    const auto& source_group_summaries = groups.group_summaries_;
    TORCH_CHECK(
        source_trace.resolved_build_body_iterations_ == 1 &&
            source_trace.resolved_chunk_mode_ ==
                static_cast<uint8_t>(ChunkMode::SEQUENTIAL) &&
            source_trace.resolved_inter_layer_io_ ==
                static_cast<uint8_t>(InterLayerIO::DDR_PINGPONG) &&
            !source_trace.resolved_chunk_outer_within_group_ &&
            source_trace.resolved_cross_batch_ > 0,
        "schema-v10 dense-DDR members STOP: resolved traversal is not exact "
        "single-body SEQUENTIAL DDR_PINGPONG");
    TORCH_CHECK(
        source_trace.resolved_hidden_size_ > 0 &&
            source_trace.resolved_num_layers_ > 0 &&
            source_trace.resolved_hidden_size_ == pimpl_->hidden_size_ &&
            source_trace.resolved_num_layers_ == pimpl_->num_layers_,
        "schema-v10 dense-DDR members STOP: resolved policy/shape drift");
    TORCH_CHECK(source_group_summaries.size() == groups.group_count_ &&
                    source_group_summaries.size() ==
                        static_cast<size_t>(
                            source_trace.resolved_num_layers_) &&
                    !source_group_summaries.empty(),
                "schema-v10 dense-DDR members STOP: initial contract requires "
                "exactly one canonical group per model layer");

    TORCH_CHECK(
        static_cast<uint64_t>(source_trace.resolved_hidden_size_) <=
            std::numeric_limits<size_t>::max() / sizeof(c10::Half),
        "schema-v10 dense-DDR members STOP: row-byte calculation overflows");
    const size_t row_bytes =
        static_cast<size_t>(source_trace.resolved_hidden_size_) *
        sizeof(c10::Half);
    TORCH_INTERNAL_ASSERT(row_bytes != 0);

    auto checked_row_bytes = [&](int64_t rows, const char* field) {
        TORCH_CHECK(rows >= 0 &&
                        static_cast<uint64_t>(rows) <=
                            std::numeric_limits<size_t>::max() / row_bytes,
                    "schema-v10 dense-DDR members STOP: ", field,
                    " byte calculation overflows");
        return static_cast<size_t>(rows) * row_bytes;
    };
    auto ingress_arena = [](int layer) {
        if (layer == 0) return SpmFmbDdrArenaRole::LiveInput;
        return layer % 2 == 0
            ? SpmFmbDdrArenaRole::PingPongA
            : SpmFmbDdrArenaRole::PingPongB;
    };
    auto egress_arena = [&](int layer) {
        if (layer == source_trace.resolved_num_layers_ - 1) {
            return SpmFmbDdrArenaRole::FinalOutput;
        }
        return layer % 2 == 0
            ? SpmFmbDdrArenaRole::PingPongB
            : SpmFmbDdrArenaRole::PingPongA;
    };
    auto hash_slice = [&](
        uint64_t hash,
        const SpmFmbSealedDenseDdrMembers::DdrSlice& slice) {
        hash = fmb_hash_u64(hash, static_cast<uint8_t>(slice.arena));
        hash = fmb_hash_u64(hash, static_cast<uint8_t>(slice.access));
        hash = fmb_hash_u64(
            hash, static_cast<uint8_t>(slice.address_mode));
        hash = fmb_hash_u64(hash, slice.byte_begin);
        return fmb_hash_u64(hash, slice.byte_count);
    };

    std::vector<SpmFmbSealedDenseDdrMembers::MemberSummary>
        member_summaries;
    const size_t members_per_group =
        source_group_summaries.front().member_count;
    TORCH_CHECK(members_per_group >= 2 &&
                    members_per_group <=
                        std::numeric_limits<size_t>::max() /
                            source_group_summaries.size(),
                "schema-v10 dense-DDR members STOP: invalid member count");
    member_summaries.reserve(
        members_per_group * source_group_summaries.size());
    size_t rows_per_group = 0;
    size_t dense_bytes_per_group = 0;
    for (size_t layer_ordinal = 0;
         layer_ordinal < source_group_summaries.size(); ++layer_ordinal) {
        const auto& source_group =
            source_group_summaries[layer_ordinal];
        TORCH_CHECK(source_group.member_count == members_per_group &&
                        source_group.first_callback <= callbacks.size() &&
                        source_group.member_count <=
                            callbacks.size() - source_group.first_callback,
                    "schema-v10 dense-DDR members STOP: source group payload "
                    "is incomplete");
        const auto& first = callbacks[source_group.first_callback];
        TORCH_CHECK(
            first.kind == SpmFmbOccurrenceKind::LayerBody &&
                first.body_id == 0 &&
                first.layer == static_cast<int>(layer_ordinal) &&
                !first.input_in_spm && !first.output_to_spm &&
                static_cast<int64_t>(first.group_id) ==
                    static_cast<int64_t>(layer_ordinal) /
                        source_trace.resolved_cross_batch_,
            "schema-v10 dense-DDR members STOP: initial contract requires "
            "one body with layer-ordered DDR_PINGPONG groups");

        size_t expected_byte_begin = 0;
        for (size_t member_ordinal = 0;
             member_ordinal < members_per_group; ++member_ordinal) {
            const size_t callback_index =
                source_group.first_callback + member_ordinal;
            const auto& callback = callbacks[callback_index];
            TORCH_CHECK(
                callback.kind == first.kind &&
                    callback.body_id == first.body_id &&
                    callback.group_id == first.group_id &&
                    callback.layer == first.layer &&
                    callback.chunk_index ==
                        static_cast<int>(member_ordinal) &&
                    callback.chunk_offset >= 0 && callback.chunk_len > 0,
                "schema-v10 dense-DDR members STOP: member identity drift");
            const size_t byte_begin = checked_row_bytes(
                callback.chunk_offset, "member begin");
            const size_t byte_count = checked_row_bytes(
                callback.chunk_len, "member extent");
            TORCH_CHECK(byte_begin == expected_byte_begin &&
                            byte_count <=
                                std::numeric_limits<size_t>::max() -
                                    byte_begin,
                        "schema-v10 dense-DDR members STOP: member ranges "
                        "must be disjoint and exactly contiguous");
            expected_byte_begin = byte_begin + byte_count;

            SpmFmbSealedDenseDdrMembers::DdrSlice ingress{
                ingress_arena(callback.layer), SpmFmbDdrAccess::Read,
                callback.layer == 0
                    ? SpmFmbDdrAddressMode::MutableBase
                    : SpmFmbDdrAddressMode::Stable,
                byte_begin, byte_count};
            SpmFmbSealedDenseDdrMembers::DdrSlice egress{
                egress_arena(callback.layer), SpmFmbDdrAccess::Write,
                SpmFmbDdrAddressMode::Stable, byte_begin, byte_count};
            TORCH_CHECK(ingress.arena != egress.arena,
                        "schema-v10 dense-DDR members STOP: ingress and "
                        "egress arenas must be distinct");

            uint64_t member_hash = fmb_hash_u64(kFmbFnvOffset, 15);
            member_hash = fmb_hash_u64(member_hash, layer_ordinal);
            member_hash = fmb_hash_u64(member_hash, member_ordinal);
            member_hash = fmb_hash_u64(member_hash, members_per_group);
            member_hash = fmb_hash_u64(member_hash, callback_index);
            member_hash = fmb_hash_u64(member_hash, callback.key);
            member_hash = fmb_hash_u64(member_hash, callback.body_id);
            member_hash = fmb_hash_u64(member_hash, callback.group_id);
            member_hash = fmb_hash_u64(
                member_hash, static_cast<uint64_t>(callback.layer));
            member_hash = fmb_hash_u64(
                member_hash, static_cast<uint64_t>(callback.chunk_offset));
            member_hash = fmb_hash_u64(
                member_hash, static_cast<uint64_t>(callback.chunk_len));
            member_hash = fmb_hash_u64(member_hash, 0);
            member_hash = fmb_hash_u64(
                member_hash, static_cast<uint64_t>(callback.chunk_len));
            member_hash = hash_slice(member_hash, ingress);
            member_hash = hash_slice(member_hash, egress);
            member_hash = fmb_hash_u64(
                member_hash, callback.semantic_hash);
            member_hash = fmb_nonzero_hash(member_hash);
            member_summaries.push_back({
                layer_ordinal,
                member_ordinal,
                members_per_group,
                callback_index,
                callback.key,
                callback.body_id,
                callback.group_id,
                callback.layer,
                callback.chunk_offset,
                callback.chunk_len,
                /*local_position_begin=*/0,
                /*local_kv_seq_len=*/callback.chunk_len,
                ingress,
                egress,
                member_hash,
            });
        }
        const int64_t final_rows =
            callbacks[source_group.first_callback + members_per_group - 1]
                .chunk_offset +
            callbacks[source_group.first_callback + members_per_group - 1]
                .chunk_len;
        const size_t this_rows = static_cast<size_t>(final_rows);
        const size_t this_bytes = checked_row_bytes(
            final_rows, "group exact-cover extent");
        TORCH_CHECK(expected_byte_begin == this_bytes &&
                        (layer_ordinal == 0 ||
                         (this_rows == rows_per_group &&
                          this_bytes == dense_bytes_per_group)),
                    "schema-v10 dense-DDR members STOP: layer groups must "
                    "share one exact member universe");
        rows_per_group = this_rows;
        dense_bytes_per_group = this_bytes;
    }

    TORCH_CHECK(member_summaries.size() ==
                    source_group_summaries.size() * members_per_group,
                "schema-v10 dense-DDR members STOP: member universe is not "
                "covered exactly once");
    uint64_t semantic_hash = fmb_hash_u64(kFmbFnvOffset, 16);
    semantic_hash = fmb_hash_u64(
        semantic_hash, static_cast<uint8_t>(capability));
    semantic_hash = fmb_hash_u64(semantic_hash, groups.group_hash_);
    semantic_hash = fmb_hash_u64(semantic_hash, source_trace.profile_hash_);
    semantic_hash = fmb_hash_u64(
        semantic_hash, source_trace.allocation_hash_);
    semantic_hash = fmb_hash_u64(
        semantic_hash, source_group_summaries.size());
    semantic_hash = fmb_hash_u64(semantic_hash, members_per_group);
    semantic_hash = fmb_hash_u64(semantic_hash, row_bytes);
    semantic_hash = fmb_hash_u64(semantic_hash, rows_per_group);
    semantic_hash = fmb_hash_u64(
        semantic_hash, dense_bytes_per_group);
    semantic_hash = fmb_hash_u64(
        semantic_hash, member_summaries.size());
    for (const auto& member : member_summaries) {
        semantic_hash = fmb_hash_u64(
            semantic_hash, member.semantic_hash);
    }
    semantic_hash = fmb_nonzero_hash(semantic_hash);
    return SpmFmbSealedDenseDdrMembers(
        groups, semantic_hash, source_group_summaries.size(),
        members_per_group, row_bytes, rows_per_group,
        dense_bytes_per_group, capability, member_summaries);
}

void FusedModelBase::validate_spm_pipeline_dense_ddr_members(
    const SpmFmbSealedDenseDdrMembers& members,
    const RpuKernelGraph& graph) const {
    const SpmFmbSealedDenseDdrMembers expected =
        seal_spm_pipeline_dense_ddr_members(
            members.source_groups_, graph);
    auto slices_equal = [](
        const SpmFmbSealedDenseDdrMembers::DdrSlice& lhs,
        const SpmFmbSealedDenseDdrMembers::DdrSlice& rhs) {
        return lhs.arena == rhs.arena && lhs.access == rhs.access &&
            lhs.address_mode == rhs.address_mode &&
            lhs.byte_begin == rhs.byte_begin &&
            lhs.byte_count == rhs.byte_count;
    };
    bool summaries_equal =
        members.member_summaries_.size() ==
        expected.member_summaries_.size();
    if (summaries_equal) {
        for (size_t index = 0;
             index < members.member_summaries_.size(); ++index) {
            const auto& actual = members.member_summaries_[index];
            const auto& canonical = expected.member_summaries_[index];
            if (actual.layer_group_ordinal !=
                    canonical.layer_group_ordinal ||
                actual.member_ordinal != canonical.member_ordinal ||
                actual.member_count != canonical.member_count ||
                actual.callback_index != canonical.callback_index ||
                actual.callback_key != canonical.callback_key ||
                actual.body_id != canonical.body_id ||
                actual.framework_group_id !=
                    canonical.framework_group_id ||
                actual.layer != canonical.layer ||
                actual.packed_row_begin != canonical.packed_row_begin ||
                actual.row_count != canonical.row_count ||
                actual.local_position_begin !=
                    canonical.local_position_begin ||
                actual.local_kv_seq_len !=
                    canonical.local_kv_seq_len ||
                !slices_equal(actual.ingress, canonical.ingress) ||
                !slices_equal(actual.egress, canonical.egress) ||
                actual.semantic_hash != canonical.semantic_hash) {
                summaries_equal = false;
                break;
            }
        }
    }
    TORCH_CHECK(
        members.capability_ == expected.capability_ &&
            members.semantic_hash_ == expected.semantic_hash_ &&
            members.layer_group_count_ ==
                expected.layer_group_count_ &&
            members.members_per_group_ == expected.members_per_group_ &&
            members.row_bytes_ == expected.row_bytes_ &&
            members.rows_per_group_ == expected.rows_per_group_ &&
            members.dense_bytes_per_group_ ==
                expected.dense_bytes_per_group_ &&
            summaries_equal,
        "schema-v10 dense-DDR member validation: sealed payload drift");
}

GraphDmaSemanticEndpoint FusedModelBase::mint_spm_pipeline_dma_endpoint(
    const SpmFmbResolvedExecutionProfile& profile,
    size_t layer_group_ordinal,
    size_t member_ordinal,
    SpmFmbDmaEndpointRole role,
    uint8_t expected_occurrence_count) const {
    const SpmFmbDmaEndpointCapability capability =
        spm_fmb_dma_endpoint_capability();
    TORCH_CHECK(
        spm_fmb_runtime_member_capability() ==
                SpmFmbRuntimeMemberCapability::Unsealed &&
            profile.runtime_member_capability_ ==
                SpmFmbRuntimeMemberCapability::Unsealed,
        "schema-v12 runtime member ref STOP: raw semantic endpoint mint is "
        "forbidden");
    TORCH_CHECK(
        capability ==
                SpmFmbDmaEndpointCapability::DenseDdrMemberRoleBindings &&
            profile.dma_endpoint_capability_ == capability,
        "schema-v11 DMA endpoint mint STOP: model/profile has not opted in");
    TORCH_CHECK(
        role == SpmFmbDmaEndpointRole::Ingress ||
            role == SpmFmbDmaEndpointRole::Egress,
        "schema-v11 DMA endpoint mint: invalid endpoint role");
    TORCH_CHECK(
        expected_occurrence_count >= 1 &&
            expected_occurrence_count <= 8 &&
            (role == SpmFmbDmaEndpointRole::Ingress ||
             expected_occurrence_count == 1),
        "schema-v11 DMA endpoint mint: invalid declared physical "
        "multiplicity");
    const std::shared_ptr<const uint64_t> owner =
        profile.owner_generation_.lock();
    TORCH_CHECK(
        owner != nullptr &&
            owner.get() == pimpl_->pipeline_manifest_generation_.get() &&
            *owner == profile.expected_owner_generation_ &&
            pimpl_->pipeline_manifest_snapshot_.valid &&
            pimpl_->pipeline_manifest_snapshot_.allocation_hash ==
                profile.expected_allocation_hash_,
        "schema-v11 DMA endpoint mint: profile owner/allocation is stale");
    TORCH_CHECK(
        profile.cpu_dry_ && profile.build_body_iterations_ == 1 &&
            profile.resolved_chunk_mode_ ==
                static_cast<uint8_t>(ChunkMode::SEQUENTIAL) &&
            profile.resolved_inter_layer_io_ ==
                static_cast<uint8_t>(InterLayerIO::DDR_PINGPONG) &&
            !profile.resolved_chunk_outer_within_group_ &&
            profile.resolved_cross_batch_ > 0 &&
            profile.resolved_hidden_size_ == pimpl_->hidden_size_ &&
            profile.resolved_num_layers_ == pimpl_->num_layers_ &&
            layer_group_ordinal <
                static_cast<size_t>(profile.resolved_num_layers_),
        "schema-v11 DMA endpoint mint: resolved policy/shape drift");

    std::vector<const SpmFmbResolvedExecutionStep*> layer_members;
    for (const auto& step : profile.steps_) {
        if (step.kind == SpmFmbOccurrenceKind::LayerBody &&
            step.body_id == 0 &&
            step.layer == static_cast<int>(layer_group_ordinal)) {
            layer_members.push_back(&step);
        }
    }
    TORCH_CHECK(layer_members.size() >= 2 &&
                    member_ordinal < layer_members.size(),
                "schema-v11 DMA endpoint mint: member ordinal is outside "
                "the canonical layer group");
    const auto& step = *layer_members[member_ordinal];
    TORCH_CHECK(step.chunk_index == static_cast<int>(member_ordinal) &&
                    step.chunk_offset >= 0 && step.chunk_len > 0,
                "schema-v11 DMA endpoint mint: canonical member drift");
    TORCH_CHECK(
        static_cast<uint64_t>(profile.resolved_hidden_size_) <=
            std::numeric_limits<size_t>::max() / sizeof(c10::Half),
        "schema-v11 DMA endpoint mint: row-byte calculation overflows");
    const size_t row_bytes =
        static_cast<size_t>(profile.resolved_hidden_size_) *
        sizeof(c10::Half);
    TORCH_CHECK(
        static_cast<uint64_t>(step.chunk_offset) <=
                std::numeric_limits<size_t>::max() / row_bytes &&
            static_cast<uint64_t>(step.chunk_len) <=
                std::numeric_limits<size_t>::max() / row_bytes,
        "schema-v11 DMA endpoint mint: member byte range overflows");
    const size_t byte_begin =
        static_cast<size_t>(step.chunk_offset) * row_bytes;
    const size_t byte_count =
        static_cast<size_t>(step.chunk_len) * row_bytes;

    SpmFmbDdrArenaRole arena;
    SpmFmbDdrAccess access;
    SpmFmbDdrAddressMode address_mode;
    if (role == SpmFmbDmaEndpointRole::Ingress) {
        arena = step.layer == 0
            ? SpmFmbDdrArenaRole::LiveInput
            : (step.layer % 2 == 0
                   ? SpmFmbDdrArenaRole::PingPongA
                   : SpmFmbDdrArenaRole::PingPongB);
        access = SpmFmbDdrAccess::Read;
        address_mode = step.layer == 0
            ? SpmFmbDdrAddressMode::MutableBase
            : SpmFmbDdrAddressMode::Stable;
    } else {
        arena = step.layer == profile.resolved_num_layers_ - 1
            ? SpmFmbDdrArenaRole::FinalOutput
            : (step.layer % 2 == 0
                   ? SpmFmbDdrArenaRole::PingPongB
                   : SpmFmbDdrArenaRole::PingPongA);
        access = SpmFmbDdrAccess::Write;
        address_mode = SpmFmbDdrAddressMode::Stable;
    }
    const DenseDmaEndpointDescriptor descriptor =
        make_dense_dma_endpoint_descriptor(
            profile.profile_hash_, profile.expected_allocation_hash_,
            step.key, step.body_id, step.group_id, step.layer,
            member_ordinal, layer_members.size(), step.chunk_offset,
            step.chunk_len, /*local_position_begin=*/0,
            /*local_kv_seq_len=*/step.chunk_len, role, arena, access,
            address_mode, expected_occurrence_count,
            byte_begin, byte_count);
    return GraphDmaSemanticEndpoint(
        descriptor.semantic_id,
        profile.profile_hash_,
        profile.expected_allocation_hash_,
        layer_group_ordinal,
        member_ordinal,
        static_cast<uint8_t>(descriptor.role),
        static_cast<uint8_t>(descriptor.arena),
        static_cast<uint8_t>(descriptor.access),
        static_cast<uint8_t>(descriptor.address_mode),
        descriptor.expected_dma_variant,
        descriptor.expected_occurrence_count,
        descriptor.byte_begin,
        descriptor.byte_count,
        /*requires_canonical_burst=*/false,
        pimpl_->pipeline_manifest_generation_,
        profile.expected_owner_generation_);
}

SpmFmbSealedDmaEndpointBindings
FusedModelBase::seal_spm_pipeline_dma_endpoint_bindings(
    const SpmFmbSealedDenseDdrMembers& members,
    const RpuKernelGraph& graph) const {
    validate_spm_pipeline_dense_ddr_members(members, graph);
    const SpmFmbDmaEndpointCapability capability =
        spm_fmb_dma_endpoint_capability();
    const auto& source_trace =
        members.source_groups_.source_trace_;
    const SpmFmbRuntimeMemberCapability runtime_member_capability =
        spm_fmb_runtime_member_capability();
    TORCH_CHECK(
        capability ==
                SpmFmbDmaEndpointCapability::DenseDdrMemberRoleBindings &&
            source_trace.dma_endpoint_capability_ == capability,
        "schema-v11 DMA endpoint bindings STOP: model/trace has not opted in");
    TORCH_CHECK(
        source_trace.runtime_member_capability_ ==
            runtime_member_capability,
        "schema-v12 runtime member bindings STOP: model/trace capability "
        "drift");
    const SpmFmbSpmPeerCapability spm_peer_capability =
        spm_fmb_spm_peer_capability();
    TORCH_CHECK(
        source_trace.spm_peer_capability_ == spm_peer_capability,
        "schema-v13 typed SPM peer bindings STOP: model/trace capability "
        "drift");
    TORCH_CHECK(members.cpu_dry(),
                "schema-v11 DMA endpoint bindings STOP: initial contract "
                "accepts only CPU-dry members");

    const auto& callbacks = source_trace.callback_summaries_;
    const auto& group_summaries =
        members.source_groups_.group_summaries_;
    const auto& member_summaries = members.member_summaries_;
    TORCH_CHECK(
        member_summaries.size() ==
                members.layer_group_count_ * members.members_per_group_ &&
            group_summaries.size() == members.layer_group_count_,
        "schema-v11 DMA endpoint bindings STOP: source member payload drift");
    TORCH_CHECK(
        !graph.has_semantic_dma_outside_window(
            source_trace.graph_node_begin_, source_trace.graph_node_end_),
        "schema-v11 DMA endpoint bindings STOP: outer Graph contains a "
        "semantic DMA outside the claimed FMB callback interval");

    std::vector<SpmFmbSealedDmaEndpointBindings::BindingSummary>
        binding_summaries;
    TORCH_CHECK(
        member_summaries.size() <=
            std::numeric_limits<size_t>::max() / 2,
        "schema-v11 DMA endpoint bindings STOP: logical endpoint count "
        "overflows");
    binding_summaries.reserve(member_summaries.size() * 2);
    size_t dma_node_count = 0;
    size_t fixed_dma_node_count = 0;
    size_t mutable_dma_node_count = 0;
    size_t legacy_dma_node_count = 0;
    size_t global_occurrence = 0;

    for (size_t layer = 0; layer < members.layer_group_count_; ++layer) {
        const auto& group = group_summaries[layer];
        TORCH_CHECK(group.first_callback < callbacks.size(),
                    "schema-v11 DMA endpoint bindings STOP: callback index "
                    "is outside the source trace");
        const auto& carrier = callbacks[group.first_callback];
        TORCH_CHECK(carrier.layer == static_cast<int>(layer),
                    "schema-v11 DMA endpoint bindings STOP: carrier layer "
                    "drift");
        TORCH_CHECK(
            carrier.legacy_dma_count == 0,
            "schema-v11 DMA endpoint bindings STOP: claimed carrier "
            "contains a legacy ID-0 DMA");
        for (size_t follower = 1; follower < group.member_count; ++follower) {
            const auto& callback =
                callbacks[group.first_callback + follower];
            TORCH_CHECK(callback.semantic_dmas.empty() &&
                            callback.legacy_dma_count == 0,
                        "schema-v11 DMA endpoint bindings STOP: a no-op "
                        "follower contains a semantic or legacy ID-0 DMA");
        }

        size_t occurrence_cursor = 0;
        for (size_t member_ordinal = 0;
             member_ordinal < members.members_per_group_;
             ++member_ordinal) {
            const auto& member = member_summaries[
                layer * members.members_per_group_ + member_ordinal];
            TORCH_CHECK(member.layer_group_ordinal == layer &&
                            member.member_ordinal == member_ordinal,
                        "schema-v11 DMA endpoint bindings STOP: member order "
                        "drift");
            for (const SpmFmbDmaEndpointRole role : {
                     SpmFmbDmaEndpointRole::Ingress,
                     SpmFmbDmaEndpointRole::Egress}) {
                const auto& slice = role == SpmFmbDmaEndpointRole::Ingress
                    ? member.ingress
                    : member.egress;
                TORCH_CHECK(
                    occurrence_cursor < carrier.semantic_dmas.size(),
                    "schema-v11 DMA endpoint bindings STOP: missing, "
                    "misordered, or wrong member/role endpoint");
                DenseDmaEndpointDescriptor descriptor;
                size_t descriptor_match_count = 0;
                const uint8_t max_declared_count =
                    role == SpmFmbDmaEndpointRole::Ingress ? 8 : 1;
                for (uint8_t declared_count = 1;
                     declared_count <= max_declared_count;
                     ++declared_count) {
                    const DenseDmaEndpointDescriptor candidate =
                        make_dense_dma_endpoint_descriptor(
                            source_trace.profile_hash_,
                            source_trace.allocation_hash_,
                            member.callback_key,
                            member.body_id,
                            member.framework_group_id,
                            member.layer,
                            member.member_ordinal,
                            member.member_count,
                            member.packed_row_begin,
                            member.row_count,
                            member.local_position_begin,
                            member.local_kv_seq_len,
                            role,
                            slice.arena,
                            slice.access,
                            slice.address_mode,
                            declared_count,
                            slice.byte_begin,
                            slice.byte_count);
                    if (carrier.semantic_dmas[occurrence_cursor].endpoint_id ==
                        candidate.semantic_id) {
                        descriptor = candidate;
                        ++descriptor_match_count;
                    }
                }
                TORCH_CHECK(
                    descriptor_match_count == 1,
                    "schema-v11 DMA endpoint bindings STOP: missing, "
                    "misordered, ambiguous, or wrong member/role endpoint");
                const size_t occurrence_begin = global_occurrence;
                size_t occurrence_count = 0;
                uint16_t channel_mask = 0;
                size_t previous_relative_node = 0;
                bool have_previous_relative_node = false;
                const uint64_t* shared_live_base = nullptr;
                bool have_shared_live_base = false;
                while (occurrence_cursor < carrier.semantic_dmas.size() &&
                       carrier.semantic_dmas[occurrence_cursor].endpoint_id ==
                           descriptor.semantic_id) {
                    const auto& dma =
                        carrier.semantic_dmas[occurrence_cursor];
                    const std::shared_ptr<const uint64_t> dma_owner =
                        dma.owner_generation.lock();
                    const bool fixed =
                        descriptor.expected_dma_variant ==
                        static_cast<uint8_t>(
                            DmaNodeData::Variant::Fixed);
                    TORCH_CHECK(
                        runtime_member_capability ==
                                SpmFmbRuntimeMemberCapability::Unsealed ||
                            dma.canonical_member_emission,
                        "schema-v12 runtime member binding STOP: "
                        "non-canonical DMA occurrence");
                    TORCH_CHECK(
                        (spm_peer_capability ==
                             SpmFmbSpmPeerCapability::Unsealed &&
                         dma.spm_peer_id == 0) ||
                            (spm_peer_capability ==
                                 SpmFmbSpmPeerCapability::
                                     TypedDensePackedOrShared &&
                             dma.spm_peer_id != 0),
                        "schema-v13 typed SPM peer binding STOP: DMA peer "
                        "provenance is missing or unexpectedly present");
                    TORCH_CHECK(
                        dma_owner != nullptr &&
                            dma_owner.get() ==
                                pimpl_->pipeline_manifest_generation_.get() &&
                            dma.expected_owner_generation ==
                                source_trace.expected_owner_generation_ &&
                            *dma_owner ==
                                source_trace.expected_owner_generation_ &&
                            dma.variant == descriptor.expected_dma_variant &&
                            dma.bytes == descriptor.byte_count &&
                            dma.channel <
                                descriptor.expected_occurrence_count &&
                            dma.channel == occurrence_count &&
                            (!have_previous_relative_node ||
                             previous_relative_node <
                                 dma.relative_node_index) &&
                            (fixed
                                 ? (dma.live_offset == 0 &&
                                    dma.live_base == nullptr)
                                 : (dma.live_offset >= 0 &&
                                    dma.live_base != nullptr &&
                                    static_cast<uint64_t>(dma.live_offset) ==
                                        descriptor.byte_begin)),
                        "schema-v11 DMA endpoint bindings STOP: DMA owner, "
                        "variant, shape, slot, offset, order, or channel "
                        "drift");
                    if (!fixed) {
                        TORCH_CHECK(
                            !have_shared_live_base ||
                                shared_live_base == dma.live_base,
                            "schema-v11 DMA endpoint bindings STOP: one "
                            "logical mutable endpoint uses multiple live "
                            "slots");
                        shared_live_base = dma.live_base;
                        have_shared_live_base = true;
                    }
                    channel_mask |= static_cast<uint16_t>(1u << dma.channel);
                    previous_relative_node = dma.relative_node_index;
                    have_previous_relative_node = true;
                    ++occurrence_count;
                    ++occurrence_cursor;
                    ++global_occurrence;
                }
                TORCH_CHECK(
                    occurrence_count ==
                        descriptor.expected_occurrence_count,
                    "schema-v11 DMA endpoint bindings STOP: observed "
                    "physical DMA multiplicity differs from the declared "
                    "endpoint");
                TORCH_CHECK(
                    runtime_member_capability ==
                            SpmFmbRuntimeMemberCapability::Unsealed ||
                        (role == SpmFmbDmaEndpointRole::Ingress
                             ? occurrence_count == 8
                             : occurrence_count == 1),
                    "schema-v12 runtime member binding STOP: ingress is not "
                    "the fixed 8-lane cover or egress is not single-lane");
                TORCH_CHECK(
                    occurrence_count <=
                        std::numeric_limits<size_t>::max() - dma_node_count,
                    "schema-v11 DMA endpoint bindings STOP: DMA count "
                    "overflow");
                dma_node_count += occurrence_count;
                if (descriptor.expected_dma_variant ==
                    static_cast<uint8_t>(DmaNodeData::Variant::Fixed)) {
                    fixed_dma_node_count += occurrence_count;
                } else {
                    mutable_dma_node_count += occurrence_count;
                }

                uint64_t binding_hash = fmb_hash_u64(kFmbFnvOffset, 19);
                binding_hash = fmb_hash_u64(binding_hash, layer);
                binding_hash = fmb_hash_u64(
                    binding_hash, member_ordinal);
                binding_hash = fmb_hash_u64(
                    binding_hash, static_cast<uint8_t>(role));
                binding_hash = fmb_hash_u64(
                    binding_hash, descriptor.semantic_id);
                binding_hash = fmb_hash_u64(
                    binding_hash, descriptor.expected_dma_variant);
                binding_hash = fmb_hash_u64(
                    binding_hash, descriptor.byte_begin);
                binding_hash = fmb_hash_u64(
                    binding_hash, descriptor.byte_count);
                binding_hash = fmb_hash_u64(
                    binding_hash, occurrence_begin);
                binding_hash = fmb_hash_u64(
                    binding_hash, occurrence_count);
                binding_hash = fmb_hash_u64(binding_hash, channel_mask);
                binding_hash = fmb_nonzero_hash(binding_hash);
                binding_summaries.push_back({
                    layer,
                    member_ordinal,
                    role,
                    descriptor.semantic_id,
                    descriptor.expected_dma_variant,
                    descriptor.byte_begin,
                    descriptor.byte_count,
                    occurrence_begin,
                    occurrence_count,
                    channel_mask,
                    binding_hash,
                });
            }
        }
        TORCH_CHECK(
            occurrence_cursor == carrier.semantic_dmas.size(),
            "schema-v11 DMA endpoint bindings STOP: carrier contains an "
            "unknown or duplicate semantic endpoint");
    }
    TORCH_CHECK(
        binding_summaries.size() == member_summaries.size() * 2 &&
            dma_node_count == global_occurrence,
        "schema-v11 DMA endpoint bindings STOP: endpoint universe is not "
        "covered exactly once");

    uint64_t binding_hash = fmb_hash_u64(kFmbFnvOffset, 20);
    binding_hash = fmb_hash_u64(
        binding_hash, static_cast<uint8_t>(capability));
    binding_hash = fmb_hash_u64(binding_hash, members.semantic_hash_);
    binding_hash = fmb_hash_u64(binding_hash, binding_summaries.size());
    binding_hash = fmb_hash_u64(binding_hash, dma_node_count);
    binding_hash = fmb_hash_u64(binding_hash, fixed_dma_node_count);
    binding_hash = fmb_hash_u64(binding_hash, mutable_dma_node_count);
    binding_hash = fmb_hash_u64(binding_hash, legacy_dma_node_count);
    for (const auto& binding : binding_summaries) {
        binding_hash = fmb_hash_u64(
            binding_hash, binding.semantic_hash);
    }
    binding_hash = fmb_nonzero_hash(binding_hash);
    return SpmFmbSealedDmaEndpointBindings(
        members, binding_hash, dma_node_count, fixed_dma_node_count,
        mutable_dma_node_count, legacy_dma_node_count, capability,
        binding_summaries);
}

void FusedModelBase::validate_spm_pipeline_dma_endpoint_bindings(
    const SpmFmbSealedDmaEndpointBindings& bindings,
    const RpuKernelGraph& graph) const {
    const SpmFmbSealedDmaEndpointBindings expected =
        seal_spm_pipeline_dma_endpoint_bindings(
            bindings.source_members_, graph);
    bool summaries_equal =
        bindings.binding_summaries_.size() ==
        expected.binding_summaries_.size();
    if (summaries_equal) {
        for (size_t index = 0;
             index < bindings.binding_summaries_.size(); ++index) {
            const auto& actual = bindings.binding_summaries_[index];
            const auto& canonical = expected.binding_summaries_[index];
            if (actual.layer_group_ordinal !=
                    canonical.layer_group_ordinal ||
                actual.member_ordinal != canonical.member_ordinal ||
                actual.role != canonical.role ||
                actual.endpoint_id != canonical.endpoint_id ||
                actual.variant != canonical.variant ||
                actual.byte_begin != canonical.byte_begin ||
                actual.byte_count != canonical.byte_count ||
                actual.occurrence_begin != canonical.occurrence_begin ||
                actual.occurrence_count != canonical.occurrence_count ||
                actual.channel_mask != canonical.channel_mask ||
                actual.semantic_hash != canonical.semantic_hash) {
                summaries_equal = false;
                break;
            }
        }
    }
    TORCH_CHECK(
        bindings.capability_ == expected.capability_ &&
            bindings.binding_hash_ == expected.binding_hash_ &&
            bindings.dma_node_count_ == expected.dma_node_count_ &&
            bindings.fixed_dma_node_count_ ==
                expected.fixed_dma_node_count_ &&
            bindings.mutable_dma_node_count_ ==
                expected.mutable_dma_node_count_ &&
            bindings.legacy_dma_node_count_ ==
                expected.legacy_dma_node_count_ &&
            summaries_equal,
        "schema-v11 DMA endpoint binding validation: sealed payload drift");
}

SpmFmbSealedSpmPeerBindings
FusedModelBase::seal_spm_pipeline_spm_peer_bindings(
    const SpmFmbSealedDmaEndpointBindings& bindings,
    const RpuKernelGraph& graph) const {
    validate_spm_pipeline_dma_endpoint_bindings(bindings, graph);
    const SpmFmbSpmPeerCapability capability =
        spm_fmb_spm_peer_capability();
    const auto& trace =
        bindings.source_members_.source_groups_.source_trace_;
    TORCH_CHECK(
        capability ==
                SpmFmbSpmPeerCapability::TypedDensePackedOrShared &&
            trace.spm_peer_capability_ == capability && trace.cpu_dry_ &&
            trace.runtime_member_capability_ ==
                SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs,
        "schema-v13 typed SPM peer seal STOP: model/trace is not one "
        "CPU-dry schema-v13 contract");

    std::vector<const SpmFmbSealedBuildTrace::SemanticDmaSummary*>
        occurrences;
    occurrences.reserve(bindings.dma_node_count_);
    std::vector<SpmFmbSealedSpmPeerBindings::PeerSummary> peers;
    peers.reserve(bindings.binding_summaries_.size());
    for (const auto& callback : trace.callback_summaries_) {
        for (const auto& dma : callback.semantic_dmas) {
            occurrences.push_back(&dma);
        }
        peers.insert(
            peers.end(), callback.spm_peers.begin(),
            callback.spm_peers.end());
    }
    TORCH_CHECK(
        occurrences.size() == bindings.dma_node_count_ &&
            peers.size() == bindings.binding_summaries_.size(),
        "schema-v13 typed SPM peer seal STOP: peer/DMA universe is not one "
        "exact endpoint cover");

    size_t shared_count = 0;
    size_t packed_count = 0;
    for (size_t index = 0; index < peers.size(); ++index) {
        const auto& peer = peers[index];
        const auto& binding = bindings.binding_summaries_[index];
        TORCH_CHECK(
            peer.layer_group_ordinal ==
                    binding.layer_group_ordinal &&
                peer.member_ordinal == binding.member_ordinal &&
                peer.role == binding.role &&
                peer.endpoint_id == binding.endpoint_id &&
                peer.peer_id != 0 && peer.semantic_hash == peer.peer_id &&
                peer.peer_slice_count == binding.byte_count &&
                peer.expected_occurrence_count ==
                    binding.occurrence_count &&
                peer.core_mask ==
                    (peer.role == SpmFmbDmaEndpointRole::Ingress
                         ? static_cast<uint8_t>(0xff)
                         : static_cast<uint8_t>(0x01)) &&
                binding.occurrence_begin <= occurrences.size() &&
                binding.occurrence_count <=
                    occurrences.size() - binding.occurrence_begin,
            "schema-v13 typed SPM peer seal STOP: logical peer differs "
            "from its schema-v11 endpoint");
        const uint16_t expected_channel_mask =
            static_cast<uint16_t>(
                (uint16_t{1} << peer.expected_occurrence_count) - 1);
        TORCH_CHECK(
            binding.channel_mask == expected_channel_mask,
            "schema-v13 typed SPM peer seal STOP: physical channel cover "
            "drift");
        for (size_t occurrence = 0;
             occurrence < binding.occurrence_count; ++occurrence) {
            const auto& dma = *occurrences[
                binding.occurrence_begin + occurrence];
            const uint64_t expected_spm =
                expected_cpu_dense_spm_peer_address(
                    peer, static_cast<uint8_t>(occurrence));
            TORCH_CHECK(
                dma.endpoint_id == peer.endpoint_id &&
                    dma.spm_peer_id == peer.peer_id &&
                    dma.canonical_member_emission &&
                    dma.bytes == peer.peer_slice_count &&
                    dma.channel == occurrence &&
                    (peer.role == SpmFmbDmaEndpointRole::Ingress
                         ? dma.dst_addr == expected_spm
                         : dma.src_addr == expected_spm),
                "schema-v13 typed SPM peer seal STOP: Graph occurrence "
                "does not carry the sealed peer identity/address");
        }
        if (peer.kind ==
            static_cast<uint8_t>(DenseSpmPeerKind::SharedStage)) {
            TORCH_CHECK(peer.peer_slice_begin == 0,
                        "schema-v13 typed SPM peer seal STOP: shared-stage "
                        "slice does not begin at zero");
            ++shared_count;
        } else {
            TORCH_CHECK(
                peer.kind == static_cast<uint8_t>(
                                 DenseSpmPeerKind::PackedGlobalSlice),
                "schema-v13 typed SPM peer seal STOP: unknown peer kind");
            ++packed_count;
        }
    }
    TORCH_CHECK(shared_count + packed_count == peers.size(),
                "schema-v13 typed SPM peer seal STOP: peer kind count "
                "overflow");

    uint64_t binding_hash = fmb_hash_u64(kFmbFnvOffset, 24);
    binding_hash = fmb_hash_u64(
        binding_hash, static_cast<uint8_t>(capability));
    binding_hash = fmb_hash_u64(binding_hash, bindings.binding_hash_);
    binding_hash = fmb_hash_u64(binding_hash, peers.size());
    binding_hash = fmb_hash_u64(binding_hash, shared_count);
    binding_hash = fmb_hash_u64(binding_hash, packed_count);
    for (const auto& peer : peers) {
        binding_hash = fmb_hash_u64(binding_hash, peer.semantic_hash);
    }
    binding_hash = fmb_nonzero_hash(binding_hash);
    return SpmFmbSealedSpmPeerBindings(
        bindings, binding_hash, shared_count, packed_count, capability,
        peers);
}

void FusedModelBase::validate_spm_pipeline_spm_peer_bindings(
    const SpmFmbSealedSpmPeerBindings& bindings,
    const RpuKernelGraph& graph) const {
    const SpmFmbSealedSpmPeerBindings expected =
        seal_spm_pipeline_spm_peer_bindings(
            bindings.source_dma_bindings_, graph);
    bool peers_equal =
        bindings.peer_summaries_.size() ==
        expected.peer_summaries_.size();
    if (peers_equal) {
        for (size_t index = 0;
             index < bindings.peer_summaries_.size(); ++index) {
            const auto& actual = bindings.peer_summaries_[index];
            const auto& canonical = expected.peer_summaries_[index];
            if (actual.layer_group_ordinal !=
                    canonical.layer_group_ordinal ||
                actual.member_ordinal != canonical.member_ordinal ||
                actual.role != canonical.role ||
                actual.kind != canonical.kind ||
                actual.endpoint_id != canonical.endpoint_id ||
                actual.peer_id != canonical.peer_id ||
                actual.allocation_ordinal !=
                    canonical.allocation_ordinal ||
                actual.declaration_ordinal !=
                    canonical.declaration_ordinal ||
                actual.alias_root_ordinal !=
                    canonical.alias_root_ordinal ||
                actual.declaration_name_hash !=
                    canonical.declaration_name_hash ||
                actual.alias_root_name_hash !=
                    canonical.alias_root_name_hash ||
                actual.storage != canonical.storage ||
                actual.scope != canonical.scope ||
                actual.layer != canonical.layer ||
                actual.phase_start != canonical.phase_start ||
                actual.phase_end != canonical.phase_end ||
                actual.root_offset != canonical.root_offset ||
                actual.allocation_offset_from_root !=
                    canonical.allocation_offset_from_root ||
                actual.logical_capacity != canonical.logical_capacity ||
                actual.aligned_capacity != canonical.aligned_capacity ||
                actual.peer_slice_begin != canonical.peer_slice_begin ||
                actual.peer_slice_count != canonical.peer_slice_count ||
                actual.core_mask != canonical.core_mask ||
                actual.expected_occurrence_count !=
                    canonical.expected_occurrence_count ||
                actual.semantic_hash != canonical.semantic_hash) {
                peers_equal = false;
                break;
            }
        }
    }
    TORCH_CHECK(
        bindings.capability_ == expected.capability_ &&
            bindings.binding_hash_ == expected.binding_hash_ &&
            bindings.shared_stage_peer_count_ ==
                expected.shared_stage_peer_count_ &&
            bindings.packed_global_peer_count_ ==
                expected.packed_global_peer_count_ &&
            peers_equal,
        "schema-v13 typed SPM peer validation: sealed payload drift");
}

void FusedModelBase::arm_spm_pipeline_spm_peer_replay(
    const SpmFmbSealedSpmPeerBindings& bindings,
    RpuKernelGraph& graph) const {
    const auto& dma_bindings = bindings.source_dma_bindings_;
    const auto& trace =
        dma_bindings.source_members_.source_groups_.source_trace_;
    const std::shared_ptr<const uint64_t> owner =
        trace.owner_generation_.lock();
    TORCH_CHECK(
        graph.state() == RpuKernelGraph::State::REPLAYING &&
            trace.graph_identity_ == &graph && owner != nullptr &&
            owner.get() ==
                pimpl_->pipeline_manifest_generation_.get() &&
            *owner == trace.expected_owner_generation_ &&
            pimpl_->pipeline_manifest_snapshot_.valid &&
            pimpl_->pipeline_manifest_snapshot_.cpu_dry &&
            pimpl_->pipeline_manifest_snapshot_.allocation_hash ==
                trace.allocation_hash_ &&
            bindings.capability_ ==
                SpmFmbSpmPeerCapability::TypedDensePackedOrShared &&
            bindings.capability_ == spm_fmb_spm_peer_capability() &&
            trace.spm_peer_capability_ == bindings.capability_ &&
            trace.spm_peer_policy_fingerprint_ ==
                spm_fmb_spm_peer_policy_fingerprint() &&
            trace.active_group_capability_ ==
                spm_fmb_active_group_capability() &&
            trace.dense_ddr_member_capability_ ==
                spm_fmb_dense_ddr_member_capability() &&
            trace.dma_endpoint_capability_ ==
                spm_fmb_dma_endpoint_capability() &&
            trace.runtime_member_capability_ ==
                SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs &&
            trace.runtime_member_capability_ ==
                spm_fmb_runtime_member_capability() &&
            dma_bindings.capability_ ==
                SpmFmbDmaEndpointCapability::DenseDdrMemberRoleBindings &&
            dma_bindings.capability_ ==
                spm_fmb_dma_endpoint_capability() &&
            trace.resolved_hidden_size_ == pimpl_->hidden_size_ &&
            trace.resolved_num_layers_ == pimpl_->num_layers_ &&
            bindings.binding_hash_ != 0 &&
            bindings.peer_summaries_.size() ==
                dma_bindings.binding_summaries_.size() &&
            bindings.shared_stage_peer_count_ +
                    bindings.packed_global_peer_count_ ==
                bindings.peer_summaries_.size() &&
            graph.build_generation() == trace.graph_build_generation_ &&
            graph.build_topology_hash() == trace.graph_topology_hash_ &&
            graph.built_signature_identity() ==
                trace.graph_signature_identity_ &&
            graph.built_signature().segment_key ==
                trace.graph_signature_segment_key_,
        "schema-v13 typed SPM peer replay arm: sealed model/owner/Graph "
        "identity is stale");

    TORCH_CHECK(!pimpl_->pipeline_runtime_member_frame_.active &&
                    g_fmb_runtime_member_owner == nullptr,
                "schema-v13 typed SPM peer replay arm STOP: a callback is "
                "already active");
    const GraphOpStreamStamp stamp = graph.op_stream_stamp();
    Impl::RuntimeDmaReplayAuthority authority;
    authority.armed = true;
    authority.graph_identity = &graph;
    authority.graph_build_generation = stamp.build_generation;
    authority.graph_signature_identity = stamp.signature_identity;
    authority.profile_hash = stamp.signature_segment_key;
    authority.spm_peer_capability = bindings.capability_;
    authority.spm_peer_bindings =
        std::unique_ptr<SpmFmbSealedSpmPeerBindings>(
            new SpmFmbSealedSpmPeerBindings(bindings));
    authority.expected_spm_peers.reserve(
        bindings.peer_summaries_.size());
    for (const auto& peer : bindings.peer_summaries_) {
        authority.expected_spm_peers.push_back({
            peer.layer_group_ordinal,
            peer.member_ordinal,
            peer.role,
            peer.kind,
            peer.endpoint_id,
            peer.peer_id,
            peer.allocation_ordinal,
            peer.declaration_ordinal,
            peer.alias_root_ordinal,
            peer.declaration_name_hash,
            peer.alias_root_name_hash,
            peer.storage,
            peer.scope,
            peer.layer,
            peer.phase_start,
            peer.phase_end,
            peer.root_offset,
            peer.allocation_offset_from_root,
            peer.logical_capacity,
            peer.aligned_capacity,
            peer.peer_slice_begin,
            peer.peer_slice_count,
            peer.core_mask,
            peer.expected_occurrence_count,
            peer.semantic_hash,
        });
    }
    graph.arm_semantic_spm_peer_replay_for_fmb(
        pimpl_->pipeline_manifest_generation_,
        trace.expected_owner_generation_);
    pimpl_->pipeline_runtime_replay_authority_ = std::move(authority);
}

void FusedModelBase::arm_spm_pipeline_dma_endpoint_replay(
    const SpmFmbSealedDmaEndpointBindings& bindings,
    RpuKernelGraph& graph) const {
    TORCH_CHECK(
        spm_fmb_spm_peer_capability() ==
            SpmFmbSpmPeerCapability::Unsealed,
        "schema-v13 typed SPM peer replay STOP: the legacy schema-v11/v12 "
        "arm cannot authorize a schema4 Graph");
    const auto& trace = bindings.source_members_.source_groups_.source_trace_;
    const std::shared_ptr<const uint64_t> owner =
        trace.owner_generation_.lock();
    TORCH_CHECK(
        graph.state() == RpuKernelGraph::State::REPLAYING &&
            trace.graph_identity_ == &graph &&
            owner != nullptr &&
            owner.get() ==
                pimpl_->pipeline_manifest_generation_.get() &&
            *owner == trace.expected_owner_generation_ &&
            pimpl_->pipeline_manifest_snapshot_.valid &&
            pimpl_->pipeline_manifest_snapshot_.allocation_hash ==
                trace.allocation_hash_ &&
            bindings.capability_ ==
                SpmFmbDmaEndpointCapability::
                    DenseDdrMemberRoleBindings &&
            bindings.capability_ ==
                spm_fmb_dma_endpoint_capability() &&
            trace.active_group_capability_ ==
                spm_fmb_active_group_capability() &&
            trace.dense_ddr_member_capability_ ==
                spm_fmb_dense_ddr_member_capability() &&
            trace.dma_endpoint_capability_ ==
                spm_fmb_dma_endpoint_capability() &&
            trace.runtime_member_capability_ ==
                spm_fmb_runtime_member_capability() &&
            trace.spm_peer_capability_ ==
                SpmFmbSpmPeerCapability::Unsealed &&
            spm_fmb_spm_peer_capability() ==
                SpmFmbSpmPeerCapability::Unsealed &&
            trace.resolved_hidden_size_ == pimpl_->hidden_size_ &&
            trace.resolved_num_layers_ == pimpl_->num_layers_ &&
            bindings.binding_hash_ != 0 &&
            bindings.binding_summaries_.size() ==
                bindings.source_members_.member_summaries_.size() * 2 &&
            graph.build_generation() ==
                trace.graph_build_generation_ &&
            graph.build_topology_hash() == trace.graph_topology_hash_ &&
            graph.built_signature_identity() ==
                trace.graph_signature_identity_ &&
            graph.built_signature().segment_key ==
                trace.graph_signature_segment_key_,
        "schema-v11 DMA endpoint replay arm: sealed model/owner/Graph "
        "identity is stale");
    graph.arm_semantic_dma_owner_replay_for_fmb(
        pimpl_->pipeline_manifest_generation_,
        trace.expected_owner_generation_);
    pimpl_->pipeline_runtime_replay_authority_ = {};
    if (trace.runtime_member_capability_ !=
        SpmFmbRuntimeMemberCapability::Unsealed) {
        TORCH_CHECK(!pimpl_->pipeline_runtime_member_frame_.active &&
                        g_fmb_runtime_member_owner == nullptr,
                    "schema-v12 runtime member replay arm STOP: a callback "
                    "is already active");
        const GraphOpStreamStamp stamp = graph.op_stream_stamp();
        Impl::RuntimeDmaReplayAuthority authority;
        authority.armed = true;
        authority.graph_identity = &graph;
        authority.graph_build_generation = stamp.build_generation;
        authority.graph_signature_identity = stamp.signature_identity;
        authority.profile_hash = stamp.signature_segment_key;
        authority.spm_peer_capability =
            SpmFmbSpmPeerCapability::Unsealed;
        authority.bindings =
            std::unique_ptr<SpmFmbSealedDmaEndpointBindings>(
                new SpmFmbSealedDmaEndpointBindings(bindings));
        pimpl_->pipeline_runtime_replay_authority_ =
            std::move(authority);
    }
}

void FusedModelBase::drive_spm_pipeline_dma_endpoint_replay_for_cpu_contract(
        const SpmFmbSealedDmaEndpointBindings& bindings) {
    TORCH_CHECK(
        spm_fmb_spm_peer_capability() ==
            SpmFmbSpmPeerCapability::Unsealed,
        "schema-v13 typed SPM peer replay STOP: the legacy schema-v12 "
        "driver cannot authorize a schema4 Graph");
    auto& authority = pimpl_->pipeline_runtime_replay_authority_;
    const auto& trace = bindings.source_members_.source_groups_.source_trace_;
    TORCH_CHECK(
        trace.cpu_dry_ &&
            trace.spm_peer_capability_ ==
                SpmFmbSpmPeerCapability::Unsealed &&
            spm_fmb_spm_peer_capability() ==
                SpmFmbSpmPeerCapability::Unsealed &&
            trace.runtime_member_capability_ ==
                SpmFmbRuntimeMemberCapability::CurrentCallbackDenseDdrRefs &&
            authority.armed && authority.bindings != nullptr &&
            authority.bindings->binding_hash_ == bindings.binding_hash_ &&
            authority.next_callback == 0,
        "schema-v12 CPU REPLAY driver requires one freshly armed CPU-dry "
        "binding");
    auto clear_failed_authority = c10::make_scope_exit([&] {
        if (authority.armed) {
            cancel_spm_pipeline_runtime_member_callback();
        }
    });
    for (size_t callback_index = 0;
         callback_index < trace.callback_summaries_.size();
         ++callback_index) {
        const auto& callback = trace.callback_summaries_[callback_index];
        TORCH_CHECK(callback.kind == SpmFmbOccurrenceKind::LayerBody &&
                        callback.body_id == 0,
                    "schema-v12 CPU REPLAY driver encountered an unsupported "
                    "callback");
        ChunkInfo chunk{
            callback.chunk_index, callback.chunk_offset,
            callback.chunk_len, callback.chunk_kv_seq_len};
        begin_spm_pipeline_runtime_member_callback_for_replay(
            bindings, callback_index);
        bool callback_complete = false;
        auto cancel_callback = c10::make_scope_exit([&] {
            if (!callback_complete) {
                cancel_spm_pipeline_runtime_member_callback();
            }
        });
        build_layer_subgraph(callback.layer, chunk);
        end_spm_pipeline_runtime_member_callback();
        callback_complete = true;
    }
    TORCH_CHECK(authority.next_callback ==
                    trace.callback_summaries_.size(),
                "schema-v12 CPU REPLAY driver omitted a sealed callback");
    authority = {};
    clear_failed_authority.release();
}

void FusedModelBase::drive_spm_pipeline_spm_peer_replay_for_cpu_contract(
        const SpmFmbSealedSpmPeerBindings& bindings) {
    auto& authority = pimpl_->pipeline_runtime_replay_authority_;
    const auto& trace = bindings.source_dma_bindings_
        .source_members_.source_groups_.source_trace_;
    TORCH_CHECK(
        trace.cpu_dry_ &&
            trace.spm_peer_capability_ ==
                SpmFmbSpmPeerCapability::TypedDensePackedOrShared &&
            authority.armed && authority.spm_peer_bindings != nullptr &&
            authority.spm_peer_bindings->binding_hash_ ==
                bindings.binding_hash_ &&
            authority.spm_peer_capability == trace.spm_peer_capability_ &&
            authority.next_callback == 0 && authority.next_peer == 0 &&
            authority.expected_spm_peers.size() ==
                bindings.peer_summaries_.size(),
        "schema-v13 CPU REPLAY driver requires one freshly armed CPU-dry "
        "typed SPM peer token");
    auto clear_failed_authority = c10::make_scope_exit([&] {
        if (authority.armed) {
            cancel_spm_pipeline_runtime_member_callback();
        }
    });
    for (size_t callback_index = 0;
         callback_index < trace.callback_summaries_.size();
         ++callback_index) {
        const auto& callback = trace.callback_summaries_[callback_index];
        TORCH_CHECK(callback.kind == SpmFmbOccurrenceKind::LayerBody &&
                        callback.body_id == 0,
                    "schema-v13 CPU REPLAY driver encountered an "
                    "unsupported callback");
        ChunkInfo chunk{
            callback.chunk_index, callback.chunk_offset,
            callback.chunk_len, callback.chunk_kv_seq_len};
        begin_spm_pipeline_runtime_member_callback_for_replay(
            bindings, callback_index);
        bool callback_complete = false;
        auto cancel_callback = c10::make_scope_exit([&] {
            if (!callback_complete) {
                cancel_spm_pipeline_runtime_member_callback();
            }
        });
        build_layer_subgraph(callback.layer, chunk);
        end_spm_pipeline_runtime_member_callback();
        callback_complete = true;
    }
    TORCH_CHECK(
        authority.next_callback == trace.callback_summaries_.size() &&
            authority.next_peer == authority.expected_spm_peers.size(),
        "schema-v13 CPU REPLAY driver omitted a sealed callback or typed "
        "SPM peer");
    authority = {};
    clear_failed_authority.release();
}

// =============================================================================
// run_all_layers — 14-step driver. Graph admission is owned by the Python
// GraphCache; C++ run_all_layers emits the operation stream.
// =============================================================================

at::Tensor FusedModelBase::run_all_layers(
    const at::Tensor& hidden_states,
    std::vector<at::Tensor>& k_caches,
    std::vector<at::Tensor>& v_caches,
    const std::optional<at::Tensor>& attention_mask,
    int64_t position,
    bool is_causal)
{
    const bool post_fn_yield_replay_run =
        pimpl_->pipeline_post_fn_yield_replay_authority_.armed;
    const bool callback_yield_replay_run =
        pimpl_->pipeline_callback_yield_replay_authority_.armed;
    TORCH_CHECK(
        !(post_fn_yield_replay_run && callback_yield_replay_run),
        "producer-yield live REPLAY has conflicting legacy and callback "
        "authorities");
    bool post_fn_yield_replay_run_complete = false;
    bool callback_yield_replay_run_complete = false;
    auto cancel_incomplete_post_fn_yield_replay =
        c10::make_scope_exit([&] {
            if (post_fn_yield_replay_run &&
                !post_fn_yield_replay_run_complete) {
                cancel_spm_pipeline_post_fn_yield_replay();
            }
            if (callback_yield_replay_run &&
                !callback_yield_replay_run_complete) {
                cancel_spm_pipeline_callback_yield_replay();
            }
        });
    if (post_fn_yield_replay_run) {
        const auto& authority =
            pimpl_->pipeline_post_fn_yield_replay_authority_;
        TORCH_CHECK(
            RpuKernelGraph::has_active() &&
                &RpuKernelGraph::active() == authority.graph_identity &&
                RpuKernelGraph::active().state() ==
                    RpuKernelGraph::State::REPLAYING,
            "post_fn producer-yield live REPLAY authority has no matching "
            "active Graph scope");
    }
    if (callback_yield_replay_run) {
        const auto& authority =
            pimpl_->pipeline_callback_yield_replay_authority_;
        TORCH_CHECK(
            RpuKernelGraph::has_active() &&
                &RpuKernelGraph::active() == authority.graph_identity &&
                RpuKernelGraph::active().state() ==
                    RpuKernelGraph::State::REPLAYING,
            "callback producer-yield live REPLAY authority has no matching "
            "active Graph scope");
    }

    TORCH_CHECK(
        spm_fmb_runtime_member_capability() ==
                SpmFmbRuntimeMemberCapability::Unsealed &&
            spm_fmb_spm_peer_capability() ==
                SpmFmbSpmPeerCapability::Unsealed,
        "schema-v12/v13 runtime member and typed SPM peer STOP: the initial "
        "capability set is CPU-dry only and does not authorize production "
        "run_all_layers");

    const bool build_trace_run = pimpl_->pipeline_build_trace_.armed;
    if (C10_UNLIKELY(build_trace_run)) {
        TORCH_CHECK(g_fmb_build_trace_owner == pimpl_.get(),
                    "schema-v8 BUILD trace run requires its owner thread");
    }
    bool build_trace_run_complete = false;
    auto cancel_incomplete_build_trace = c10::make_scope_exit([&] {
        if (C10_UNLIKELY(build_trace_run) &&
            !build_trace_run_complete) {
            const auto& failed = pimpl_->pipeline_build_trace_;
            bool marker_may_have_been_recorded =
                failed.pending_producer_yield.has_value();
            for (const auto& callback : failed.callbacks) {
                marker_may_have_been_recorded =
                    marker_may_have_been_recorded ||
                    !callback.producer_yields.empty();
            }
            if (marker_may_have_been_recorded &&
                RpuKernelGraph::has_active()) {
                RpuKernelGraph::active()
                    .poison_semantic_spm_producer_yield_for_fmb();
            }
            clear_pipeline_build_trace_noexcept(*pimpl_);
        }
    });
    if (C10_UNLIKELY(build_trace_run)) {
        const auto& candidate = pimpl_->pipeline_build_trace_;
        TORCH_CHECK(candidate.profile != nullptr && !candidate.cpu_dry &&
                        candidate.next_step == 0 &&
                        candidate.callbacks.empty(),
                    "schema-v8 BUILD trace run requires one fresh armed live "
                    "profile");
        TORCH_CHECK(RpuKernelGraph::has_active() &&
                        &RpuKernelGraph::active() ==
                            candidate.graph_identity,
                    "schema-v8 BUILD trace run: active Graph owner drift");
        const GraphOpStreamStamp stamp =
            RpuKernelGraph::active().op_stream_stamp();
        TORCH_CHECK(stamp.state == RpuKernelGraph::State::RECORDING &&
                        stamp.build_generation ==
                            candidate.graph_build_generation &&
                        stamp.signature_identity ==
                            candidate.graph_signature_identity &&
                        stamp.signature_segment_key ==
                            candidate.graph_signature_segment_key,
                    "schema-v8 BUILD trace run requires the armed Graph "
                    "RECORDING identity");
    }

    // 1. static config
    auto static_cfg = static_config();
    TORCH_CHECK(static_cfg.num_layers > 0,
                "FusedModelBase::run_all_layers: static_cfg.num_layers must be > 0");
    TORCH_CHECK(pimpl_->num_layers_ == static_cfg.num_layers,
                "FusedModelBase: num_layers_ (", pimpl_->num_layers_,
                ") != static_cfg.num_layers (", static_cfg.num_layers, ")");
    if (C10_UNLIKELY(build_trace_run)) {
        const auto post_fn_capability =
            pimpl_->pipeline_build_trace_.profile
                ->post_fn_yield_capability_;
        const auto layer_yield_capability =
            pimpl_->pipeline_build_trace_.profile
                ->layer_producer_yield_capability_;
        const bool profile_has_post_fn =
            !pimpl_->pipeline_build_trace_.profile->steps_.empty() &&
            pimpl_->pipeline_build_trace_.profile->steps_.back().kind ==
                SpmFmbOccurrenceKind::PostFn;
        TORCH_CHECK(
            static_cfg.preload_fn == nullptr &&
                static_cfg.pre_layers_fn == nullptr &&
                static_cfg.post_layers_fn == nullptr &&
                ((layer_yield_capability !=
                      SpmFmbLayerProducerYieldCapability::Unsealed) ||
                 (post_fn_capability ==
                      SpmFmbPostFnYieldCapability::Unsealed &&
                  static_cfg.post_fn == nullptr) ||
                 (post_fn_capability ==
                      SpmFmbPostFnYieldCapability::
                          CanonicalDenseReplicatedFp16 &&
                  static_cfg.post_fn != nullptr &&
                  static_cfg.post_output_shape.empty() &&
                  !static_cfg.fast_replay_bake_post_fn)) &&
                ((layer_yield_capability ==
                      SpmFmbLayerProducerYieldCapability::Unsealed) ||
                 (layer_yield_capability ==
                      SpmFmbLayerProducerYieldCapability::
                          CanonicalDenseReplicatedFp16 &&
                  (static_cfg.post_fn != nullptr) == profile_has_post_fn &&
                  static_cfg.post_output_shape.empty() &&
                  !static_cfg.fast_replay_skip_layer_loop &&
                  !static_cfg.fast_replay_skip_full_body &&
                  !static_cfg.fast_replay_bake_post_fn)),
                    "schema-v8 BUILD trace STOP: opaque callbacks remain "
                    "unsealed");
    }

    // 2. shape contract
    TORCH_CHECK(hidden_states.dim() == 3,
                "FusedModelBase::run_all_layers: hidden_states must be 3D");
    TORCH_CHECK(hidden_states.size(2) == pimpl_->hidden_size_,
                "FusedModelBase::run_all_layers: hidden_states.size(2)=",
                hidden_states.size(2), " != hidden_size_=", pimpl_->hidden_size_);
    TORCH_CHECK(hidden_states.is_contiguous(),
                "FusedModelBase::run_all_layers: hidden_states must be contiguous");
    TORCH_CHECK(hidden_states.scalar_type() == at::kHalf,
                "FusedModelBase::run_all_layers: hidden_states must be FP16");
    // Device belongs in the contract next to dim/dtype/contiguity: step 10 takes
    // RpuGetDevAddr(hidden_states.data_ptr()) for the layer-input mutable DMA.
    // A CPU tensor here would therefore provide an invalid device address. The DMA-side
    // guard catches that, but only far from the cause and phrased as a DMA
    // failure; naming it at the boundary points at the actual mistake.
    TORCH_CHECK(hidden_states.device().type() == at::kPrivateUse1,
                "FusedModelBase::run_all_layers: hidden_states must be on the "
                "RPU device, got ", hidden_states.device());

    pimpl_->batch_size_ = hidden_states.size(0);
    int64_t seq_len = hidden_states.size(1);
    TORCH_CHECK(pimpl_->batch_size_ > 0 && seq_len > 0,
                "FusedModelBase::run_all_layers: empty batch/seq");
    // Batch decode: B independent sequences are packed into the GEMM M dim so a
    // decode step reads each weight ONCE for all B rows. Only seq_len == 1
    // qualifies: at seq_len > 1 the rows of one sequence carry consecutive RoPE
    // positions, so a packed [B, S] block would need per-row positions and a
    // block-diagonal causal mask (no such GQA kernel exists — see
    // rpu_sdpa_minibatch.cpp, which is MHA-only and mask-free). Batched prefill
    // is therefore driven as B separate run_all_layers calls from Python.
    TORCH_CHECK(pimpl_->batch_size_ == 1 || seq_len == 1,
                "FusedModelBase: batch_size > 1 requires seq_len == 1 (decode); got "
                "batch_size=", pimpl_->batch_size_, ", seq_len=", seq_len,
                ". Run batched prefill as one call per sequence.");
    TORCH_CHECK(pimpl_->batch_size_ == 1 || is_causal,
                "FusedModelBase: batch_size > 1 requires the causal decode path; "
                "got is_causal=false");
    TORCH_CHECK(pimpl_->batch_size_ == 1 || static_cfg.batch_decode_active,
                "FusedModelBase: batch_size > 1 is not enabled for this model "
                "request; only the explicitly admitted Qwen3 decode path may "
                "use batched rows");

    // 3. mask normalization
    bool use_explicit_mask = !is_causal && attention_mask.has_value();
    std::optional<at::Tensor> effective_mask =
        use_explicit_mask ? attention_mask : std::nullopt;
    int64_t max_kv_seq_len = use_explicit_mask
        ? attention_mask->size(-1)
        : (position + seq_len);

    // 4. KV cache validation + inference context
    TORCH_CHECK(static_cast<int64_t>(k_caches.size()) == pimpl_->num_layers_,
                "run_all_layers: k_caches.size()=", k_caches.size(),
                " != num_layers_=", pimpl_->num_layers_);
    TORCH_CHECK(static_cast<int64_t>(v_caches.size()) == pimpl_->num_layers_,
                "run_all_layers: v_caches.size()=", v_caches.size(),
                " != num_layers_=", pimpl_->num_layers_);

    for (int64_t L = 0; L < pimpl_->num_layers_; L++) {
        const at::Tensor& kc = k_caches[L];
        const at::Tensor& vc = v_caches[L];
        TORCH_CHECK(kc.defined() && vc.defined(),
                    "run_all_layers: kv_caches[", L, "] undefined");
        TORCH_CHECK(kc.device().type() == at::kPrivateUse1 &&
                    vc.device().type() == at::kPrivateUse1,
                    "run_all_layers: kv_caches[", L, "] must be on RPU");
        TORCH_CHECK(kc.scalar_type() == at::kHalf &&
                    vc.scalar_type() == at::kHalf,
                    "run_all_layers: kv_caches[", L, "] must be FP16");
        TORCH_CHECK(kc.is_contiguous() && vc.is_contiguous(),
                    "run_all_layers: kv_caches[", L, "] must be contiguous");
    }

    pimpl_->ctx_.hidden_states = &hidden_states;
    pimpl_->ctx_.k_caches = &k_caches;
    pimpl_->ctx_.v_caches = &v_caches;
    pimpl_->ctx_.attention_mask = effective_mask;
    pimpl_->ctx_.position = position;
    pimpl_->ctx_.seq_len  = seq_len;
    pimpl_->ctx_.batch_size = pimpl_->batch_size_;
    pimpl_->ctx_.is_causal = is_causal;
    pimpl_->ctx_.input_in_spm = false;
    pimpl_->ctx_.output_to_spm = false;

    // Deep fast-replay (RPU_DEEP_FAST_REPLAY): on a warm REPLAY that will FULLY skip the
    // op-stream (no post_fn — prefill/denoise/action graphs), the per-forward setup is
    // redundant. The SPM layout (declare_buffers + ensure_allocated, step 7) and the
    // weight-DMA nodes (preload callbacks + preload_fn, step 11) are
    // baked into the kd_buf and replayed as-is, so re-running them just rebuilds identical
    // state nobody reads (like the body re-walk the existing fast-replay already skips).
    // Skip them. Vision uses skip_layer_body (its post_fn re-emits and READS the SPM
    // layout) → NOT full-skip → keeps the setup. The kept per-forward bits still run:
    // ctx_, compute_chunks/dynamic_config (cheap; mask #23), the registry output_tensor_,
    // hidden_in_src_base_ + flush. Default OFF in C++; wall_oss setdefaults it on.
    const bool deep_full_skip =
        pimpl_->deep_fast_replay_enabled_ &&
        pimpl_->global_fast_replay_enabled_
        && !static_cfg.batch_decode_active
        && pimpl_->pipeline_lease_epoch_ == 0
        && RpuKernelGraph::has_active()
        && RpuKernelGraph::active().state() == RpuKernelGraph::State::REPLAYING
        && !(static_cfg.post_fn != nullptr && RpuKernelGraph::active().has_post_fn_cursor());

    // 5. chunk computation
    LayoutContext layout_ctx;
    layout_ctx.chunk_size = 0;
    layout_ctx.max_kv_seq_len = max_kv_seq_len;
    layout_ctx.num_layers = static_cfg.num_layers;
    layout_ctx.use_attn_mask = use_explicit_mask;
    // Thread the attention mode into the hashed LayoutContext
    // so declare_buffers derives kv_first_layout from it (pure function) and the
    // allocation re-keys when a handle switches causal↔bidirectional-no-mask.
    // Propagates to compute_chunks probe_ctx/min_ctx and alloc_ctx by value-copy.
    layout_ctx.is_causal = is_causal;
    // Batch decode: chunk planning stays PER SEQUENCE (seq_len == 1 → one chunk
    // of len 1); declare_buffers multiplies the row-parallel slots by this.
    layout_ctx.batch_size = pimpl_->batch_size_;

    auto decl_fn = [this](const LayoutContext& c) { return this->declare_buffers(c); };
    auto estimate_fn = [&](const LayoutContext& c) {
        return detail::estimate_temporary_total(decl_fn(c));
    };
    auto valid_fn = [this, seq_len, position](int64_t cs) {
        return this->subclass_chunk_size_valid(cs, seq_len, position);
    };
    const int64_t chunk_size_cap = subclass_chunk_size_cap(seq_len, position);

    int64_t resolved_chunk_size = 0;
    auto chunks = compute_chunks_impl(
        *pimpl_, decl_fn, valid_fn, chunk_size_cap,
        seq_len, position, layout_ctx,
        &resolved_chunk_size);
    TORCH_CHECK(!chunks.empty(), "FusedModelBase: compute_chunks returned empty list");
    // Planning-only adapter queries must not alter this public observation.
    pimpl_->last_resolved_chunk_size_ = resolved_chunk_size;

    ChunkPlan plan;
    plan.chunk_size = chunks[0].len;
    plan.num_chunks = static_cast<int64_t>(chunks.size());

    // 6. dynamic config
    auto dyn_cfg = dynamic_config(plan);

    // 6.5 KV_FIRST dual-chunk plan (kv_first_chunk_plan_fn)
    ChunkPlan kv_insert_plan = plan;
    std::vector<ChunkInfo> kv_insert_chunks = chunks;
    if (dyn_cfg.chunk_mode == ChunkMode::KV_FIRST) {
        if (static_cfg.kv_first_chunk_plan_fn != nullptr) {
            kv_insert_plan = std::invoke(static_cfg.kv_first_chunk_plan_fn, *this, plan);
        }
        if (kv_insert_plan.chunk_size != plan.chunk_size) {
            int64_t cs = kv_insert_plan.chunk_size;
            kv_insert_chunks.clear();
            for (int64_t off = 0; off < seq_len; off += cs) {
                int64_t len = std::min(cs, seq_len - off);
                kv_insert_chunks.push_back(
                    {static_cast<int>(off / cs), off, len, position + off + len});
            }
        }
    }

    // 7. SPM allocation (dual-chunk aware) — skipped on deep full-skip replay (baked).
    if (!deep_full_skip) {
        LayoutContext alloc_ctx = layout_ctx;
        alloc_ctx.chunk_size = chunks[0].len;
        if (dyn_cfg.chunk_mode == ChunkMode::KV_FIRST &&
            !kv_insert_chunks.empty() &&
            kv_insert_chunks[0].len != chunks[0].len) {
            alloc_ctx.kv_insert_chunk_size = kv_insert_chunks[0].len;
        }
        auto decls = decl_fn(alloc_ctx);
        ensure_allocated_impl(*pimpl_, decls, alloc_ctx, estimate_fn,
                              subclass_layout_hash(),
                              /*enforce_allocation_identity=*/false);
        // Cache the freshly-invoked declare_buffers result so
        // run_preload_callbacks_ captures the same BufferDecl instances
        // (their preload_callback std::functions).
        pimpl_->last_decls_ = std::move(decls);
    }

    // Fire preload callbacks on persistent advance / dirty flag
    // (skipped on deep full-skip — the persistent SPM is baked + replayed).
    // RhinoVLA per-model preload skip (fast_replay_skip_preload): on a fast-replay
    // REPLAY the cursor is set absolutely by skip_op_stream/skip_layer_body (both land
    // >= the preload region), so re-walking the ~144 callbacks is pure host
    // overhead; the built preload DMA nodes replay via the segment launch to the SAME
    // persistent SPM. Gated on fast_replay_skip_layer_loop + REPLAYING; other
    // models leave the flag false.
    const bool skip_preload_callbacks_replay =
        static_cfg.fast_replay_skip_preload &&
        static_cfg.fast_replay_skip_layer_loop &&
        pimpl_->pipeline_lease_epoch_ == 0 &&
        RpuKernelGraph::has_active() &&
        RpuKernelGraph::active().state() == RpuKernelGraph::State::REPLAYING;
    const bool composite_preload_follower =
        consume_spm_pipeline_composite_preload_follower();
    const bool skip_clean_preload_callbacks_recording =
        static_cfg.clean_recording_skip_preload_callbacks &&
        RpuKernelGraph::has_active() &&
        RpuKernelGraph::active().state() == RpuKernelGraph::State::RECORDING &&
        pimpl_->cached_preload_gen_ == SPM_ALLOC.persistent_generation() &&
        !pimpl_->preload_callbacks_dirty_;
    if (!deep_full_skip && !skip_preload_callbacks_replay &&
        !composite_preload_follower &&
        !skip_clean_preload_callbacks_recording)
        run_preload_callbacks_(pimpl_->last_decls_, *this);

    // 8. inter-layer I/O strategy
    InterLayerIO io = dyn_cfg.inter_layer_io;
    if (io == InterLayerIO::AUTO) {
        io = (chunks.size() == 1) ? InterLayerIO::SPM_RESIDENT : InterLayerIO::DDR_PINGPONG;
    }
    dyn_cfg.inter_layer_io = io;

    // 9. group count
    int64_t cross_batch = (static_cfg.cross_layer_batch_size > 0)
                          ? static_cfg.cross_layer_batch_size
                          : static_cfg.num_layers;
    if (cross_batch <= 0) cross_batch = static_cfg.num_layers;

    // 10. DDR buffers — looked up from the shape-keyed registry on
    //     pimpl_->registry_ so DMA-baked pointers remain stable across
    //     multi-key GraphCache replay.
    auto opts = hidden_states.options();
    pimpl_->post_output_tensor_ = at::Tensor{};

    // opts comes from hidden_states.options() (already assigned just above).
    // ShapeKey carries device so PrivateUse1 vs CPU never alias in the registry.
    ShapeKey skey{
        pimpl_->batch_size_,
        seq_len,
        pimpl_->hidden_size_,
        opts.dtype().toScalarType(),
        opts.device().type(),
    };

    int64_t num_groups = 1;
    if (static_cfg.num_layers > 0)
        num_groups = (static_cfg.num_layers + cross_batch - 1) / cross_batch;

    const bool need_chain_bufs = (io == InterLayerIO::DDR_PINGPONG) || (num_groups > 1);
    const auto& entry = pimpl_->registry_.get(skey, opts,
                                               /*need_ab=*/need_chain_bufs,
                                               /*need_out=*/true);

    pimpl_->output_tensor_ = entry.out;
    if (need_chain_bufs) {
        pimpl_->ddr_bufA_     = entry.a;
        pimpl_->ddr_bufB_     = entry.b;
        pimpl_->ddr_bufA_ptr_ = entry.a.data_ptr<c10::Half>();
        pimpl_->ddr_bufB_ptr_ = entry.b.data_ptr<c10::Half>();
    } else {
        pimpl_->ddr_bufA_     = at::Tensor{};
        pimpl_->ddr_bufB_     = at::Tensor{};
        pimpl_->ddr_bufA_ptr_ = nullptr;
        pimpl_->ddr_bufB_ptr_ = nullptr;
    }

    // Layer-0 mutable DMA: stash the caller's live dev addr and flush its DDR
    // cache lines once per forward. The framework's REPLAY end() patches every
    // layer-0 chunk DMA in kd_buf to use this address — no staging copy needed.
    // Boundary-flush variant (force) — caller's hidden_states may have CPU-dirty
    // lines from an upstream embedding op that wrote via host memcpy.
    current_hidden_in_src_base() =
        ::rhino_lkn::RpuGetDevAddr(hidden_states.data_ptr());
    rpu_ddr_flush_force_sized(
        hidden_states.data_ptr<c10::Half>(), hidden_states.nbytes());
    // Keep the source tensor alive until graph execution completes.
    // The two lines above capture only hidden_states' DEVICE ADDRESS: layer 0's
    // chunk DMA reads `*hidden_in_src_base_ + offset`, and no graph node holds a
    // reference to the tensor. But recording is not execution — the graph runs in
    // RpuKernelGraph::end(), i.e. when the enclosing `with cache.capture(sig):`
    // exits, by which time the Python adapter that built `hidden_states` has
    // returned and dropped its only reference. The DDR block is then free and its
    // device VA can be re-issued before the deferred DMA reads it.
    // ⚠️ The relevant vector is RpuKernelGraph::tensor_refs_ (graph_runtime.h) —
    // NOT this file's pimpl_->tensor_refs_, which is cleared before end() runs.
    // RpuKernelGraph::tensor_refs_ is cleared in
    // end() only AFTER execution, so keep_alive() is exactly the right lifetime.
    // Costs one refcount per forward and nothing else (keep_alive does not flush).
    if (RpuKernelGraph::has_active()) {
        RpuKernelGraph::active().keep_alive(hidden_states);
    }

    // 11. weights preload (preload_fn)
    //
    // graph-naive: GraphCache 已经移交 Python 端,这里不再做 weights_key_changed
    // / set_weights_key admission。preload_fn 直接调,callback body 走
    // wq->enqueu_kernel（无 graph scope 时立即发射，Python 用
    // with cache.capture(WEIGHTS_SIG) 包整段 run_all_layers 让其进
    // graph RECORDING/REPLAYING)。
    //
    // weights_dirty_ 仍然在这里清：Python 侧 admission 不知道 weights
    // 是否真的换过(set_weights C++ 端置 dirty),所以即使 Python 命中 BUILT,
    // 也要让本次 forward 走一遍 preload_fn 才能把 DMA 节点录进图。
    // RhinoVLA per-model preload_fn skip (fast_replay_skip_preload): on a clean-weights
    // REPLAY the preload op-walk is pure host overhead — the built DMA/memset nodes
    // replay via the segment launch and the cursor is overwritten by the skip below.
    // Validated profiles preserve output identity. Other models leave the flag false.
    const bool skip_preload_replay =
        static_cfg.fast_replay_skip_preload &&
        pimpl_->pipeline_lease_epoch_ == 0 &&
        RpuKernelGraph::has_active() &&
        RpuKernelGraph::active().state() == RpuKernelGraph::State::REPLAYING &&
        !pimpl_->weights_dirty_;
    if (static_cfg.preload_fn != nullptr && !deep_full_skip && !skip_preload_replay) {
        std::invoke(static_cfg.preload_fn, *this);
        pimpl_->weights_dirty_ = false;
    }
    if (C10_UNLIKELY(build_trace_run)) {
        auto& candidate = pimpl_->pipeline_build_trace_;
        TORCH_INTERNAL_ASSERT(candidate.profile != nullptr &&
                              candidate.next_step == 0 &&
                              candidate.callbacks.empty() &&
                              !candidate.persistent_preload_callback_open &&
                              !candidate.callback_open);
        const GraphOpStreamStamp resolved_begin =
            RpuKernelGraph::active().op_stream_stamp();
        TORCH_CHECK(
            resolved_begin.graph == candidate.graph_identity &&
                resolved_begin.state ==
                    RpuKernelGraph::State::RECORDING &&
                resolved_begin.build_generation ==
                    candidate.graph_build_generation &&
                resolved_begin.signature_identity ==
                    candidate.graph_signature_identity &&
                resolved_begin.signature_segment_key ==
                    candidate.graph_signature_segment_key,
            "schema-v8 BUILD trace resolved-window prelude Graph identity "
            "drifted");
        if (resolved_begin.position != candidate.graph_node_begin) {
            TORCH_CHECK(
                candidate.profile->preload_capability_ ==
                    SpmFmbPreloadCapability::
                        PersistentOutsideResolvedWindow,
                "schema-v8 BUILD trace found unowned Graph nodes before the "
                "first resolved callback");
            candidate.graph_node_begin = resolved_begin.position;
        }
    }

    // Repeat the body (11.5 pre → 12 layers → 12.5 post) body_iterations
    // times in one graph. Default 1 keeps non-Wall models byte-identical.
    // Setup (1–11) and post_fn/flush (13–14) stay once, outside the loop.
    // ── Fast replay (RPU_WALL_OSS_FAST_REPLAY; default OFF, wall_oss opts in) ─────
    // On a warm REPLAY the body would otherwise re-emit the same op stream recorded at
    // BUILD. All per-call deltas have already been applied outside the body loop:
    //   • position_ids → copied into the stable keepalive in CausalDecoderModel::forward
    //     (the prologue, before run_all_layers);
    //   • the explicit 2D mask → copied to its stable DDR slot by dynamic_config (step 6);
    //   • the caller's fresh input hidden_states → hidden_in_src_base_ refreshed above
    //     (layer-0 reads it via `*_mutable(&hidden_in_src_base_)` → re-derefed every replay).
    //   • op-specific mutable inputs (e.g. wall_oss denoise x0/b_all/te_all/x_traj/v_traj)
    //     are `*_mutable(&member,…)` bases set in the op prologue → also re-derefed.
    // Every kernel-register dev_addr is stable across replays (weights baked, KV cache
    // persistent, SPM absolute), and the body emits NO per-call host side-effect a kernel
    // reads. So pushing cursor_ to nodes_.size() (skip_op_stream_for_fast_replay) and
    // letting end() drive execute_graph_for_replaying is BIT-IDENTICAL to re-walking —
    // the segment fast-path re-derefs the mutable bases + reads the freshly-updated stable
    // buffers. Default OFF (other models' bodies are not contract-audited); wall_oss
    // setdefaults it on for its denoise/prefill/vision graphs. No post_output_shape op
    // opts in, so the post-loop returns the stable output_tensor_.
    // (fused_fast_replay is the per-handle latch taken near the top — reused by
    // deep_full_skip.) RhinoVLA opts in per-model (static_cfg.fast_replay_skip_
    // layer_loop) rather than via the RPU_WALL_OSS_FAST_REPLAY env that wall_oss,
    // pi05, gr00t and lingbot use. Honor either — the skip path below re-derefs
    // the graph's mutable bases so it is byte-identical to re-walking (verified
    // by the e2e idempotent_cos + action_cos gates). Models that set neither are
    // unaffected.
    // The per-handle global_fast_replay_enabled_ value is also reused by
    // deep_full_skip above.
    const bool fast_skip =
        (pimpl_->global_fast_replay_enabled_ ||
         static_cfg.fast_replay_skip_layer_loop)
        && !static_cfg.batch_decode_active
        && !callback_yield_replay_run
        && pimpl_->pipeline_lease_epoch_ == 0
        && RpuKernelGraph::has_active()
        && RpuKernelGraph::active().state() == RpuKernelGraph::State::REPLAYING;
    if (fast_skip) {
        auto& g = RpuKernelGraph::active();
        if (static_cfg.post_fn != nullptr && g.has_post_fn_cursor()
            && !static_cfg.fast_replay_bake_post_fn) {
            // For a post_fn graph, skip only the layer body and leave post_fn's
            // nodes ahead for the normal replay path to re-emit (its mutable-DMA
            // op patches the fresh per-forward output addr).
            g.skip_layer_body_for_fast_replay();
        } else {
            // No post_fn (denoise / prefill / vision / action), OR RhinoVLA
            // fast_replay_bake_post_fn (fused merger baked into the BUILT graph) →
            // skip the WHOLE op-stream; step 13 skips the host post_fn re-emission.
            // Byte-identical to the original fast-replay path.
            g.skip_op_stream_for_fast_replay();
        }
    }
    int64_t body_iters = fast_skip
        ? 0
        : (static_cfg.body_iterations > 0 ? static_cfg.body_iterations : 1);
    const FmbExecutionTraversalSpec execution_spec{
        static_cfg.num_layers,
        cross_batch,
        dyn_cfg.chunk_mode,
        io,
        dyn_cfg.chunk_outer_within_group,
        static_cfg.kv_first_fn != nullptr,
        &chunks,
        &kv_insert_chunks,
    };
    for (int64_t body_it = 0; body_it < body_iters; body_it++) {
    pimpl_->ctx_.body_iter = body_it;
    pimpl_->ctx_.position  = position;

    // 11.5 pre_layers_fn — once per body iteration, before the
    // layer loop. It emits into the caller-owned graph scope when one is
    // active. Subclass bodies must not open a raw or nested graph scope.
    if (static_cfg.pre_layers_fn != nullptr) {
        std::invoke(static_cfg.pre_layers_fn, *this);
    }

    // 12. main compute — for-each-layer-group
    //
    // Graph admission is adapter-owned. Each group directly iterates layer +
    // chunk + build_layer_subgraph;
    // op 内部 wq->enqueu_kernel 走 RpuQueue proxy(无 graph scope 立即发射,
    // 有 scope 则进 RECORDING/REPLAYING)。Python 端 with cache.capture(sig)
    // 由适配器提供。
    for_each_fmb_execution_step(
        execution_spec, static_cast<uint32_t>(body_it),
        [&](const FmbExecutionStepView& step) {
            if (step.updates_io_context) {
                pimpl_->ctx_.input_in_spm = step.input_in_spm;
                pimpl_->ctx_.output_to_spm = step.output_to_spm;
            }
            TORCH_INTERNAL_ASSERT(step.chunk != nullptr);
            auto invoke_callback = [&] {
                if (step.kind == SpmFmbOccurrenceKind::KvInsertBody) {
                    std::invoke(static_cfg.kv_first_fn, *this, step.layer,
                                *step.chunk);
                } else {
                    TORCH_INTERNAL_ASSERT(
                        step.kind == SpmFmbOccurrenceKind::LayerBody);
                    build_layer_subgraph(step.layer, *step.chunk);
                }
            };
            if (C10_UNLIKELY(build_trace_run)) {
                auto& candidate = pimpl_->pipeline_build_trace_;
                TORCH_CHECK(candidate.next_step <
                                candidate.profile->steps_.size(),
                            "schema-v8 BUILD trace STOP: runtime emitted an "
                            "extra callback");
                const SpmFmbResolvedExecutionStep& expected =
                    candidate.profile->steps_[candidate.next_step];
                invoke_trace_callback(
                    *pimpl_, expected, step, [&] {
                        begin_spm_pipeline_runtime_member_callback_for_build(
                            *candidate.profile, expected);
                        bool runtime_callback_complete = false;
                        auto cancel_runtime_callback =
                            c10::make_scope_exit([&] {
                                if (!runtime_callback_complete) {
                                    cancel_spm_pipeline_runtime_member_callback();
                                }
                            });
                        invoke_callback();
                        end_spm_pipeline_runtime_member_callback();
                        runtime_callback_complete = true;
                    },
                    [this] {
                        end_spm_pipeline_build_trace_callback();
                    });
            } else {
                invoke_callback();
            }
        });

    // 12.5 post_layers_fn — once per body iteration, after the
    // layer loop, in the same caller-owned capture as the rest of the body.
    if (static_cfg.post_layers_fn != nullptr) {
        std::invoke(static_cfg.post_layers_fn, *this);
    }
    }  // End body_iterations loop.

    if (C10_UNLIKELY(build_trace_run)) {
        auto& candidate = pimpl_->pipeline_build_trace_;
        TORCH_CHECK(candidate.profile != nullptr,
                    "schema-v8 BUILD trace STOP: missing resolved profile");
        const bool expects_post_fn =
            !candidate.profile->steps_.empty() &&
            candidate.profile->steps_.back().kind ==
                SpmFmbOccurrenceKind::PostFn;
        TORCH_CHECK(
            expects_post_fn
                ? candidate.next_step + 1 ==
                      candidate.profile->steps_.size() &&
                      candidate.profile->steps_.back().kind ==
                          SpmFmbOccurrenceKind::PostFn
                : candidate.next_step ==
                      candidate.profile->steps_.size(),
            "schema-v8 BUILD trace STOP: runtime omitted or reordered a "
            "resolved callback before post_fn");
        candidate.traversal_complete = !expects_post_fn;
    }

    // 13. post (post_fn)
    //
    // graph-naive: 同 step 12,直接 invoke post_fn。Python with capture 包整段
    // run_all_layers 时 post 也跟着进 graph;无 scope 时立即发射。
    if (static_cfg.post_fn != nullptr) {
        // Record the post_fn-start cursor BEFORE emitting any post_fn node. No-op
        // unless an active graph is RECORDING (PASSTHROUGH has no active graph;
        // REPLAYING is gated inside mark_post_fn_cursor). A typed register
        // census owns the complete outer op stream and skips it atomically on
        // replay, so it must not publish this legacy component-fast cursor.
        // Other graphs use the cursor to skip only the layer body and re-emit
        // post_fn on the normal path.
        if (RpuKernelGraph::has_active() &&
            !RpuKernelGraph::active().kernel_register_census_active()) {
            RpuKernelGraph::active().mark_post_fn_cursor();
        }

        // RhinoVLA fast_replay_bake_post_fn (vision fused merger): the post_fn nodes are
        // baked in the BUILT graph and replay via the segment launch (skip_op_stream set
        // the cursor to nodes_.size()); skip the host re-emission. Bit-identical — re-emit
        // on REPLAY only re-validates the same built nodes. Other models leave the flag
        // false → post_fn re-emits as before.
        const bool skip_post_fn_replay =
            static_cfg.fast_replay_bake_post_fn &&
            static_cfg.fast_replay_skip_layer_loop &&
            pimpl_->pipeline_lease_epoch_ == 0 &&
            RpuKernelGraph::has_active() &&
            RpuKernelGraph::active().state() == RpuKernelGraph::State::REPLAYING;
        if (!skip_post_fn_replay) {
            // If a post_output_shape is declared, allocate a fresh post_output_tensor_
            // (Pitfall 4 structural mitigation — at::empty per forward).
            if (!static_cfg.post_output_shape.empty()) {
                pimpl_->post_output_tensor_ = at::empty(static_cfg.post_output_shape, opts);
            }

            auto invoke_post_fn = [&] {
                std::invoke(static_cfg.post_fn, *this);
            };
            if (C10_UNLIKELY(build_trace_run)) {
                auto& candidate = pimpl_->pipeline_build_trace_;
                TORCH_CHECK(
                    candidate.profile != nullptr &&
                        candidate.next_step <
                            candidate.profile->steps_.size(),
                    "schema-v8 BUILD trace STOP: post_fn has no resolved "
                    "step");
                const SpmFmbResolvedExecutionStep& expected =
                    candidate.profile->steps_[candidate.next_step];
                FmbExecutionStepView actual{
                    SpmFmbOccurrenceKind::PostFn, expected.body_id,
                    expected.group_id, /*layer=*/-1,
                    expected.traversal_ordinal, /*chunk=*/nullptr,
                    expected.scope_mask, /*updates_io_context=*/false,
                    /*input_in_spm=*/false, /*output_to_spm=*/false};
                invoke_trace_callback(
                    *pimpl_, expected, actual, invoke_post_fn,
                    [this] { end_spm_pipeline_build_trace_callback(); });
                TORCH_CHECK(
                    candidate.next_step ==
                        candidate.profile->steps_.size(),
                    "schema-v8 BUILD trace STOP: post_fn did not complete "
                    "the resolved profile");
                candidate.traversal_complete = true;
            } else {
                invoke_post_fn();
            }

            if (pimpl_->post_output_tensor_.defined()) {
                rpu_ddr_flush(pimpl_->post_output_tensor_.data_ptr<c10::Half>());
            }
        }
    }

    if (C10_UNLIKELY(build_trace_run)) {
        TORCH_CHECK(
            pimpl_->pipeline_build_trace_.traversal_complete,
            "schema-v8 BUILD trace STOP: resolved post_fn was not emitted");
    }

    // 14. flush output + cleanup
    if (!pimpl_->post_output_tensor_.defined()) {
        rpu_ddr_flush(pimpl_->output_tensor_.data_ptr<c10::Half>());
    }
    // ⚠️ Pre-existing orphan, NOT the graph's keepalive: nothing in the repo ever
    // pushes into pimpl_->tensor_refs_ (grep — only this clear and its declaration
    // in fused_model_base_impl.h). It is also cleared here, i.e. before the
    // graph executes, so it could never serve the C-1 lifetime anyway. The one
    // that matters is RpuKernelGraph::tensor_refs_ — see the guard at step 10.
    pimpl_->tensor_refs_.clear();

    build_trace_run_complete = true;
    if (post_fn_yield_replay_run) {
        const auto& authority =
            pimpl_->pipeline_post_fn_yield_replay_authority_;
        TORCH_CHECK(
            authority.armed && authority.yields != nullptr &&
                !authority.yield_open &&
                authority.next_yield == authority.yields->yields_.size() &&
                !g_fmb_post_fn_yield_begin.has_value(),
            "post_fn producer-yield live REPLAY omitted a sealed yield");
        pimpl_->pipeline_post_fn_yield_replay_authority_ = {};
        post_fn_yield_replay_run_complete = true;
    }
    if (callback_yield_replay_run) {
        const auto& authority =
            pimpl_->pipeline_callback_yield_replay_authority_;
        TORCH_CHECK(
            authority.armed && authority.yields != nullptr &&
                !authority.yield_open &&
                authority.next_yield == authority.yields->yields_.size() &&
                !g_fmb_post_fn_yield_begin.has_value(),
            "callback producer-yield live REPLAY omitted a sealed yield");
        pimpl_->pipeline_callback_yield_replay_authority_ = {};
        callback_yield_replay_run_complete = true;
    }

    // Output lifetime contract by graph state:
    //
    // PASSTHROUGH: clone output_tensor_ so Python caller can accumulate
    //   returns into a list without aliasing (Pi0.5 multi-image,
    //   SigLIP projector). Cost: one hidden_size × seq_len × fp16
    //   memcpy per forward (~512 KB for Qwen3-8B seq=64; amortized
    //   under PyTorch's caching allocator).
    //
    // RECORDING/REPLAYING: MUST NOT clone. The clone() above would run
    //   *inside* the graph scope while kernels are still in nodes_
    //   (not yet executed by end() → build_batch+enqueu_batch). It
    //   would memcpy from a stale/empty output_tensor_ into a fresh
    //   tensor disconnected from the graph's actual write target.
    //   Symptom: lm_head sees zero/garbage logits → next_id=0 and
    //   prompt-fragment tokens.
    //
    //   Returning output_tensor_ raw means graph-capture callers are
    //   responsible for not aliasing across forwards. That's the
    //   contract — if they wrap forward in `with cache.capture(sig)`,
    //   they must honor graph-stable buffer semantics and avoid aliasing
    //   retained outputs across forwards.
    if (pimpl_->post_output_tensor_.defined()) {
        return pimpl_->post_output_tensor_;
    }
    auto graph_state = RpuKernelGraph::active().state();
    if (graph_state == RpuKernelGraph::State::PASSTHROUGH) {
        return pimpl_->output_tensor_.clone();
    }
    return pimpl_->output_tensor_;
}

}  // namespace v3
