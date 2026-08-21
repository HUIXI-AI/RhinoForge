// fused_model_base_impl.h — private PImpl state container
//
// Defines FusedModelBase::Impl. Included ONLY by fused_model_base.cpp.
// Holds the internal FusedModelBase framework state.
// Adding a field here does NOT change the public header.
//
// Two independent dirty flags (weights_dirty_ for preload_fn,
// preload_callbacks_dirty_ for Gemma path). Conflating them would cause
// Gemma per-forward re-DMA of all norms.

#pragma once

#include "fused_model_base.h"
#include "rpu_ddr_buffer_registry.h"
#include "rpu_kernel_decls.h"   // PreparedMask
#include "rpu_spm_allocator.h"  // SpmAllocator::AllocRequest
#include "rpu_spm_pipeline.h"
#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace v3 {

class FusedModelBase::Impl {
public:
    struct OwnedPipelineDecl {
        std::string name;
        int64_t logical_bytes = 0;
        size_t aligned_bytes = 0;
        int phase_start = 0;
        int phase_end = 0;
        StorageClass storage = StorageClass::Temp;
        int per_layer = 0;
        std::string alias_of;
        BufferScope scope = BufferScope::LayerWide;
        bool reverse_layer_alloc = false;
        bool preload_present = false;
    };

    struct PipelineAllocationEntry {
        std::string name;
        StorageClass storage = StorageClass::Temp;
        int layer = -1;
        size_t logical_bytes = 0;
        size_t aligned_bytes = 0;
        uint32_t offset = 0;
        std::string alias_root;
        bool reverse_layer_alloc = false;
        bool preload_present = false;
    };

    struct CachedLayoutSnapshot {
        bool valid = false;
        bool cpu_dry = true;
        int64_t params_hash = 0;
        LayoutContext layout_context;
        size_t required_extent = 0;
        uint64_t layout_hash = 0;
        uint64_t declaration_hash = 0;
        uint64_t allocation_hash = 0;
        uint64_t persistent_hash = 0;
        uint64_t allocator_generation = 0;
        uint64_t persistent_generation = 0;
        uint64_t persistent_layout_version = 0;
        size_t persistent_floor = SpmAllocator::SPM_USABLE;
        size_t persistent_top = SpmAllocator::SPM_USABLE;
        std::vector<OwnedPipelineDecl> decls;
        std::vector<PipelineAllocationEntry> temporary_allocations;
        std::vector<PipelineAllocationEntry> persistent_allocations;
    };

    struct BuildTraceAccessRecord {
        uint32_t allocation_ordinal = 0;
        uint32_t declaration_ordinal = 0;
        uint32_t alias_root_ordinal = 0;
        StorageClass storage = StorageClass::Temp;
        BufferScope scope = BufferScope::LayerWide;
        int layer = -1;
        SpmFmbBuildTraceAccessorKind accessor =
            SpmFmbBuildTraceAccessorKind::Addr;
        uint8_t core_mask = 0;
        uint32_t root_offset = 0;
        size_t protected_bytes = 0;
        uint32_t resolve_count = 0;
    };

    struct BuildTraceSemanticDmaRecord {
        size_t relative_node_index = 0;
        uint64_t endpoint_id = 0;
        uint8_t variant = 0;
        uint64_t src_addr = 0;
        uint64_t dst_addr = 0;
        size_t bytes = 0;
        uint8_t channel = 0;
        int64_t live_offset = 0;
        const uint64_t* live_base = nullptr;
        std::weak_ptr<const uint64_t> owner_generation;
        uint64_t expected_owner_generation = 0;
        bool canonical_member_emission = false;
        uint64_t spm_peer_id = 0;
    };

    struct BuildTraceSpmPeerRecord {
        size_t layer_group_ordinal = 0;
        size_t member_ordinal = 0;
        SpmFmbDmaEndpointRole role =
            static_cast<SpmFmbDmaEndpointRole>(0);
        uint8_t kind = 0;
        uint64_t endpoint_id = 0;
        uint64_t peer_id = 0;
        uint32_t allocation_ordinal = 0;
        uint32_t declaration_ordinal = 0;
        uint32_t alias_root_ordinal = 0;
        uint64_t declaration_name_hash = 0;
        uint64_t alias_root_name_hash = 0;
        uint8_t storage = 0;
        uint8_t scope = 0;
        int layer = -1;
        int phase_start = 0;
        int phase_end = 0;
        uint32_t root_offset = 0;
        uint32_t allocation_offset_from_root = 0;
        size_t logical_capacity = 0;
        size_t aligned_capacity = 0;
        size_t peer_slice_begin = 0;
        size_t peer_slice_count = 0;
        uint8_t core_mask = 0;
        uint8_t expected_occurrence_count = 0;
        uint64_t semantic_hash = 0;
    };

    struct BuildTraceProducerYieldRecord {
        size_t ordinal = 0;
        uint64_t semantic_id = 0;
        SpmFmbTerminalWriterKind writer_kind =
            SpmFmbTerminalWriterKind::AllReduceSumResidual;
        SpmDense2DSpec spec;
        uint32_t allocation_ordinal = 0;
        uint32_t declaration_ordinal = 0;
        uint32_t alias_root_ordinal = 0;
        uint64_t declaration_name_hash = 0;
        uint64_t alias_root_name_hash = 0;
        uint32_t root_offset = 0;
        uint32_t allocation_offset_from_root = 0;
        size_t logical_capacity = 0;
        size_t aligned_capacity = 0;
        size_t graph_node_begin = 0;
        size_t graph_node_end = 0;
        uint16_t graph_node_kind_mask = 0;
        uint64_t graph_topology_hash = 0;
    };

    struct BuildTraceCallbackRecord {
        SpmFmbOccurrenceKind kind = SpmFmbOccurrenceKind::LayerBody;
        uint32_t key = 0;
        uint32_t body_id = 0;
        uint32_t group_id = 0;
        int layer = -1;
        uint32_t traversal_ordinal = 0;
        int chunk_index = -1;
        int64_t chunk_offset = 0;
        int64_t chunk_len = 0;
        int64_t chunk_kv_seq_len = 0;
        SpmFmbScopeMask scope_mask = 0;
        bool updates_io_context = false;
        bool input_in_spm = false;
        bool output_to_spm = false;
        int local_phase_begin = 0;
        int local_phase_end = 0;
        uint32_t global_origin = 0;
        size_t graph_node_begin = 0;
        size_t graph_node_end = 0;
        uint16_t graph_node_kind_mask = 0;
        uint64_t graph_topology_hash = 0;
        size_t legacy_dma_count = 0;
        std::vector<BuildTraceAccessRecord> accesses;
        std::vector<BuildTraceSemanticDmaRecord> semantic_dmas;
        std::vector<BuildTraceSpmPeerRecord> spm_peers;
        std::vector<BuildTraceProducerYieldRecord> producer_yields;
    };

    struct BuildTraceCandidate {
        bool armed = false;
        bool persistent_preload_callback_open = false;
        bool callback_open = false;
        bool producer_yield_open = false;
        bool traversal_complete = false;
        bool cpu_driver = false;
        SpmFmbPreloadCapability preload_capability =
            SpmFmbPreloadCapability::Unsealed;
        bool composite = false;
        uint64_t composite_identity = 0;
        size_t composite_occurrence_ordinal = 0;
        bool composite_producer = false;
        size_t composite_producer_local_ordinal = 0;
        SpmFmbCompositeOccurrenceCapability composite_capability =
            SpmFmbCompositeOccurrenceCapability::Unsealed;
        uint64_t composite_policy_fingerprint = 0;
        uint64_t composite_occurrence_profile_hash = 0;
        bool composite_preload_follower = false;
        size_t composite_outer_begin = 0;
        const void* graph_identity = nullptr;
        uint64_t graph_build_generation = 0;
        uint64_t graph_signature_identity = 0;
        uint64_t graph_signature_segment_key = 0;
        size_t graph_node_begin = 0;
        uint64_t profile_hash = 0;
        uint64_t allocation_hash = 0;
        uint64_t expected_owner_generation = 0;
        bool cpu_dry = true;
        size_t next_step = 0;
        std::unique_ptr<SpmFmbResolvedExecutionProfile> profile;
        std::vector<BuildTraceCallbackRecord> callbacks;
        std::optional<BuildTraceProducerYieldRecord>
            pending_producer_yield;
    };

    struct CompositeOccurrenceRuntime {
        bool armed = false;
        bool input_slot_consumed = false;
        bool preload_decision_consumed = false;
        uint64_t composite_identity = 0;
        size_t producer_local_ordinal = 0;
        SpmFmbCompositeOccurrenceCapability capability =
            SpmFmbCompositeOccurrenceCapability::Unsealed;
        uint64_t policy_fingerprint = 0;
        uint64_t expected_layout_hash = 0;
        uint64_t expected_allocation_hash = 0;
        uint64_t expected_persistent_generation = 0;
        uint64_t composite_occurrence_profile_hash = 0;
        const void* graph_identity = nullptr;
        uint8_t graph_state = 0;
        uint64_t graph_build_generation = 0;
        uint64_t graph_signature_identity = 0;
        uint64_t graph_signature_segment_key = 0;
        bool preload_follower = false;
    };

    // One lexical physical-composite occurrence.  The retained lease owns all
    // placement; FMB keeps only its opaque identity, role, and exact-consumption
    // cursor.  Raw direct/payload/consumer addresses are resolved on demand and
    // are never cached here.
    struct CompositePhysicalExecution {
        bool armed = false;
        bool producer = false;
        bool require_committed = false;
        const SpmPipelineLease* lease = nullptr;
        uint64_t lease_epoch = 0;
        uint64_t plan_hash = 0;
        uint64_t composite_identity = 0;
        size_t producer_local_ordinal = 0;
        size_t next_target = 0;
        size_t expected_target_count = 0;
        const void* graph_identity = nullptr;
        uint8_t graph_state = 0;
        uint64_t graph_build_generation = 0;
        uint64_t graph_signature_identity = 0;
        uint64_t graph_signature_segment_key = 0;
    };

    struct RuntimeDenseMemberDescriptor {
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
        SpmFmbDdrArenaRole ingress_arena =
            SpmFmbDdrArenaRole::LiveInput;
        SpmFmbDdrAddressMode ingress_address_mode =
            SpmFmbDdrAddressMode::Stable;
        size_t ingress_byte_begin = 0;
        size_t ingress_byte_count = 0;
        SpmFmbDdrArenaRole egress_arena =
            SpmFmbDdrArenaRole::FinalOutput;
        SpmFmbDdrAddressMode egress_address_mode =
            SpmFmbDdrAddressMode::Stable;
        size_t egress_byte_begin = 0;
        size_t egress_byte_count = 0;
    };

    struct RuntimeDenseMemberFrame {
        bool active = false;
        bool cpu_dry = true;
        uint64_t callback_epoch = 0;
        const void* graph_identity = nullptr;
        uint64_t graph_build_generation = 0;
        uint64_t profile_hash = 0;
        uint64_t allocation_hash = 0;
        uint64_t expected_owner_generation = 0;
        SpmFmbSpmPeerCapability spm_peer_capability =
            SpmFmbSpmPeerCapability::Unsealed;
        size_t callback_index = 0;
        uint32_t callback_key = 0;
        int layer = -1;
        size_t next_member = 0;
        SpmFmbDmaEndpointRole next_role =
            SpmFmbDmaEndpointRole::Ingress;
        std::vector<RuntimeDenseMemberDescriptor> members;

        struct PeerRoleAccumulator {
            bool initialized = false;
            uint8_t kind = 0;
            uint32_t allocation_ordinal = 0;
            uint32_t declaration_ordinal = 0;
            uint32_t alias_root_ordinal = 0;
            uint64_t declaration_name_hash = 0;
            uint64_t alias_root_name_hash = 0;
            uint8_t storage = 0;
            uint8_t scope = 0;
            int layer = -1;
            int phase_start = 0;
            int phase_end = 0;
            uint32_t root_offset = 0;
            uint32_t allocation_offset_from_root = 0;
            size_t logical_capacity = 0;
            size_t aligned_capacity = 0;
            size_t dense_payload_bytes = 0;
            size_t seen_member_count = 0;
            size_t next_packed_begin = 0;
        };
        PeerRoleAccumulator ingress_peer;
        PeerRoleAccumulator egress_peer;
    };

    struct RuntimeDmaReplayAuthority {
        bool armed = false;
        const void* graph_identity = nullptr;
        uint64_t graph_build_generation = 0;
        uint64_t graph_signature_identity = 0;
        uint64_t profile_hash = 0;
        size_t next_callback = 0;
        size_t next_peer = 0;
        SpmFmbSpmPeerCapability spm_peer_capability =
            SpmFmbSpmPeerCapability::Unsealed;
        std::unique_ptr<SpmFmbSealedDmaEndpointBindings> bindings;
        std::unique_ptr<SpmFmbSealedSpmPeerBindings> spm_peer_bindings;
        std::vector<BuildTraceSpmPeerRecord> expected_spm_peers;
    };

    struct RuntimePostFnYieldReplayAuthority {
        bool armed = false;
        bool yield_open = false;
        const void* graph_identity = nullptr;
        uint64_t graph_build_generation = 0;
        uint64_t graph_signature_identity = 0;
        uint64_t profile_hash = 0;
        size_t next_yield = 0;
        std::unique_ptr<SpmFmbSealedPostFnYields> yields;
    };

    struct RuntimeCallbackYieldReplayAuthority {
        bool armed = false;
        bool yield_open = false;
        const void* graph_identity = nullptr;
        uint64_t graph_build_generation = 0;
        uint64_t graph_signature_identity = 0;
        uint64_t profile_hash = 0;
        size_t next_yield = 0;
        std::unique_ptr<SpmFmbSealedCallbackYields> yields;
    };

    // ===== Layer execution state =====
    bool     valid_                 = false;
    uint64_t alloc_gen_             = 0;
    uint64_t cached_persistent_gen_ = 0;  // for ensure_allocated Path 1 detection
    uint64_t cached_preload_gen_    = 0;  // Tracks when preload_callbacks last fired.
    int64_t  chunk_size_override_   = 0;
    // Per-handle resolved chunk_size, populated by run_all_layers after
    // compute_chunks_impl resolves a real dispatch.
    // Default 0 = sentinel "no forward has run yet for this handle". Read via the
    // public FusedModelBase::get_last_resolved_chunk_size() accessor; surfaced to
    // Python through rpu_causal_decoder_get_resolved_chunk_size(handle) op.
    int64_t  last_resolved_chunk_size_ = 0;
    // Certified chunk envelope. Default-constructed == UNDECLARED, and the
    // participating subclasses (CausalDecoderModel, Qwen3_5Model) deny prefill on
    // an undeclared handle. Every other subclass ignores it and is unchanged.
    FusedModelBase::ChunkEnvelope chunk_envelope_{};

    std::unordered_map<std::string, uint32_t>                offsets_;
    std::vector<std::unordered_map<std::string, uint32_t>>   per_layer_offsets_;
    int64_t  temp_per_layer_usage_  = 0;

    bool     persistent_allocated_  = false;
    int64_t  persistent_usage_      = 0;
    int64_t  persistent_layout_hash_ = 0;
    std::unordered_map<std::string, uint32_t>              persistent_offsets_;
    std::vector<std::unordered_map<std::string, uint32_t>> persistent_per_layer_offsets_;

    std::vector<at::Tensor> tensor_refs_;
    PreparedMask            prepared_mask_;
    SdpaStableMaskCache     sdpa_stable_mask_cache_;

    c10::Half* ddr_bufA_ptr_ = nullptr;
    c10::Half* ddr_bufB_ptr_ = nullptr;
    // Layer-0 input live RPU dev addr — set per forward to
    // `RpuGetDevAddr(hidden_states.data_ptr())`. The framework's mutable DMA
    // registry (slot.mutable_dmas) holds `&hidden_in_src_base_`; REPLAY end()
    // dereferences this and calls Queue_t::update_dma_kernel to patch each
    // layer-0 broadcast DMA's src in kd_buf. The chunk-offset (in bytes) is
    // baked into each MutableDmaSlot at BUILD time.
    uint64_t hidden_in_src_base_ = 0;
    std::array<uint64_t, 3> composite_hidden_in_src_bases_{};
    CompositeOccurrenceRuntime composite_occurrence_runtime_;
    CompositePhysicalExecution composite_physical_execution_;

    int64_t  cached_params_hash_     = 0;

    // Model parameters (set by subclass via set_model_params)
    int64_t num_q_heads_         = 0;
    int64_t num_kv_heads_        = 0;
    int64_t head_dim_            = 0;
    int64_t hidden_size_         = 0;
    int64_t intermediate_size_   = 0;
    int     attn_tp_             = 8;

    // ===== Model execution state =====
    uint64_t model_state_gen_            = 0;
    std::shared_ptr<uint64_t> outer_fast_model_generation_ =
        std::make_shared<uint64_t>(0);
    uint64_t persistent_layout_version_  = 0;  // 仅 Path 1 递增

    // Dirty-flag separation — each gates a different dispatch path:
    bool weights_dirty_           = true;  // preload_fn emission; cleared after dispatch
    bool preload_callbacks_dirty_ = true;  // BufferDecl callback loop; cleared after dispatch

    // Cold per-handle fast-replay policy. The environment is sampled when the
    // native model handle is constructed, so one handle cannot process-cache
    // the decision for a model loaded later.
    bool global_fast_replay_enabled_ = false;
    bool deep_fast_replay_enabled_   = false;

    DdrBufferRegistry registry_;
    at::Tensor       ddr_bufA_;
    at::Tensor       ddr_bufB_;
    at::Tensor       output_tensor_;
    at::Tensor       post_output_tensor_;
    InferenceContext ctx_;
    int64_t          num_layers_   = 0;
    int64_t          batch_size_   = 0;
    // Cached last-declared buffers, so run_preload_callbacks_ can re-fire
    // without invoking declare_buffers() again (the fresh lambda captures
    // from the most-recent declare_buffers invocation).
    std::vector<BufferDecl> last_decls_;

    // Transitional physical-pipeline adoption.  Offsets remain the ordinary
    // cached FMB layout; these fields only prove the current invocation is
    // bound to the checked lease that reserved their physical range.
    uint64_t pipeline_lease_epoch_ = 0;
    uint64_t pipeline_plan_hash_ = 0;
    uint64_t pipeline_layout_hash_ = 0;
    size_t pipeline_temporary_bytes_ = 0;
    uint32_t pipeline_scratch_base_ = 0;
    std::unordered_set<std::string> pipeline_temporary_names_;
    std::unordered_set<std::string> pipeline_temporary_per_layer_names_;

    CachedLayoutSnapshot pipeline_manifest_snapshot_;
    bool allocation_declaration_valid_ = false;
    uint64_t allocation_declaration_hash_ = 0;
    std::vector<OwnedPipelineDecl> allocation_declarations_;
    bool pipeline_occurrence_schedule_present_ = false;
    SpmFmbOccurrenceSchedule pipeline_occurrence_schedule_;
    std::shared_ptr<uint64_t> pipeline_manifest_generation_ =
        std::make_shared<uint64_t>(0);
    BuildTraceCandidate pipeline_build_trace_;
    RuntimeDenseMemberFrame pipeline_runtime_member_frame_;
    RuntimeDmaReplayAuthority pipeline_runtime_replay_authority_;
    RuntimePostFnYieldReplayAuthority
        pipeline_post_fn_yield_replay_authority_;
    RuntimeCallbackYieldReplayAuthority
        pipeline_callback_yield_replay_authority_;
    uint64_t pipeline_runtime_callback_epoch_ = 0;
    uint64_t pipeline_cpu_dry_live_input_base_ = UINT64_C(0x21000000);
};

}  // namespace v3
