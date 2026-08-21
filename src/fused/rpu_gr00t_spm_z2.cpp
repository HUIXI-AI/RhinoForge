#include "rpu_gr00t_spm_z2.h"

#include "core/fused_model_base.h"
#include "core/rpu_kernel_decls.h"
#include "core/rpu_spm_allocator.h"
#include "graph/graph_runtime.h"

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace {

constexpr v3::SpmScratchId kScratchRegion{1};
constexpr v3::SpmPortId kPortRegion{1};
constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

struct Gr00tZ2State {
    std::mutex mutex;
    std::optional<v3::SpmPipelinePlan> plan;
    std::unique_ptr<v3::SpmPipelineLease> lease;
    int64_t vl_handle = 0;
    int64_t denoise_handle = 0;
    int64_t seq_len = 0;
    size_t persistent_top_bytes = 0;
    uint64_t plan_hash = 0;
};

Gr00tZ2State& state() {
    static Gr00tZ2State value;
    return value;
}

int64_t encode_u64(uint64_t value) {
    int64_t encoded = 0;
    static_assert(sizeof(encoded) == sizeof(value));
    std::memcpy(&encoded, &value, sizeof(value));
    return encoded;
}

uint64_t decode_u64(int64_t value) {
    uint64_t decoded = 0;
    static_assert(sizeof(decoded) == sizeof(value));
    std::memcpy(&decoded, &value, sizeof(value));
    return decoded;
}

size_t live_persistent_top_bytes() {
    return SPM_ALLOC.super_persistent_used() + SPM_ALLOC.persistent_used();
}

uint64_t hash_u64(uint64_t hash, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash = (hash ^ static_cast<uint8_t>(value >> shift)) * kFnvPrime;
    }
    return hash;
}

uint64_t physical_binding_digest(const Gr00tZ2State& s) {
    TORCH_CHECK(s.lease && s.lease->physical() &&
                    s.lease->physical_sealed(),
                "GR00T Z2 physical digest requires one sealed lease");
    const auto scratch = s.lease->scratch(kScratchRegion);
    const auto port = s.lease->port(kPortRegion);
    const auto& spec = port.spec();
    uint64_t hash = kFnvOffset;
    hash = hash_u64(hash, UINT64_C(0x47525a3250485931));  // GRZ2PHY1
    hash = hash_u64(hash, s.plan_hash);
    hash = hash_u64(hash, s.lease->physical_arena_base());
    hash = hash_u64(hash, s.lease->physical_arena_end());
    hash = hash_u64(hash, s.lease->physical_top_reservation());
    hash = hash_u64(hash, scratch.resolve_physical_offset(*s.lease));
    hash = hash_u64(hash, scratch.size_bytes());
    hash = hash_u64(hash, port.resolve_physical_offset(*s.lease));
    hash = hash_u64(hash, port.resolve_physical_addr(0, *s.lease));
    hash = hash_u64(hash, port.size_bytes());
    hash = hash_u64(hash, static_cast<uint8_t>(spec.dtype));
    hash = hash_u64(hash, static_cast<uint64_t>(spec.rows));
    hash = hash_u64(hash, static_cast<uint64_t>(spec.cols));
    hash = hash_u64(hash, static_cast<uint8_t>(spec.distribution));
    return hash == 0 ? UINT64_C(1) : hash;
}

void check_identity(const Gr00tZ2State& s,
                    int64_t vl_handle,
                    int64_t denoise_handle,
                    int64_t seq_len,
                    uint64_t plan_hash,
                    const char* op) {
    TORCH_CHECK(s.plan.has_value(), op, ": prepare must be called first");
    TORCH_CHECK(s.vl_handle == vl_handle &&
                    s.denoise_handle == denoise_handle &&
                    s.seq_len == seq_len,
                op, ": prepared identity mismatch; expected handles/seq=(",
                s.vl_handle, ",", s.denoise_handle, ",", s.seq_len,
                "), got (", vl_handle, ",", denoise_handle, ",", seq_len,
                ")");
    TORCH_CHECK(s.plan_hash == plan_hash,
                op, ": stale plan hash; expected ", s.plan_hash,
                ", got ", plan_hash);
}

void validate_endpoint_specs_locked(const Gr00tZ2State& s,
                                    const char* op) {
    TORCH_CHECK(s.plan.has_value(), op, ": plan is missing");
    const auto produced = v3::gr00t_z2_internal::vl_encoder_port_spec(
        s.vl_handle, s.seq_len);
    const auto required = v3::gr00t_z2_internal::denoise_port_spec(
        s.denoise_handle, s.seq_len);
    const auto route = v3::SpmIdentityRoute::bind(
        kPortRegion, produced, required,
        /*first_write=*/0, /*last_read=*/1);
    TORCH_CHECK(route.spec() == s.plan->port_spec(kPortRegion),
                op, ": endpoint spec drifted after typed plan compilation");
}

template <typename Fn>
void cleanup_step(std::exception_ptr& failure, Fn&& fn) noexcept {
    try {
        fn();
    } catch (...) {
        if (!failure) failure = std::current_exception();
    }
}

void reset_prepared_state_locked(Gr00tZ2State& s) {
    TORCH_INTERNAL_ASSERT(!s.lease);
    s.plan.reset();
    s.vl_handle = 0;
    s.denoise_handle = 0;
    s.seq_len = 0;
    s.persistent_top_bytes = 0;
    s.plan_hash = 0;
}

void validate_active_pipeline_locked(Gr00tZ2State& s,
                                     uint64_t epoch,
                                     uint64_t plan_hash,
                                     const char* op) {
    TORCH_CHECK(s.lease, op, ": no physical lease is active");
    TORCH_CHECK(s.plan_hash == plan_hash,
                op, ": stale plan hash; expected ", s.plan_hash,
                ", got ", plan_hash);
    s.lease->validate_physical(epoch, plan_hash, /*require_sealed=*/true);
    v3::gr00t_z2_internal::validate_vl_encoder(s.vl_handle, *s.lease);
    v3::gr00t_z2_internal::validate_denoise(s.denoise_handle, *s.lease);
}

GraphDdrRegisterAbi ddr_abi(GraphDdrRegisterRole role,
                            GraphDdrRegisterAccess access,
                            uint16_t reg_lo,
                            uint16_t reg_hi) {
    return GraphDdrRegisterAbi{
        role, access, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
        reg_lo, reg_hi};
}

GraphKernelRegisterRule typed_rule(
        KernelId id,
        size_t count,
        std::vector<GraphDdrRegisterAbi> operands) {
    GraphKernelRegisterRule rule;
    rule.kernel_id = id;
    rule.kind = GraphKernelRegisterRuleKind::TypedDdr;
    rule.operands = std::move(operands);
    rule.expected_count = count;
    return rule;
}

GraphKernelRegisterRule intrinsic_no_ddr_rule(KernelId id, size_t count) {
    GraphKernelRegisterRule rule;
    rule.kernel_id = id;
    rule.kind = GraphKernelRegisterRuleKind::IntrinsicNoDdr;
    rule.expected_count = count;
    return rule;
}

GraphKernelRegisterRule explicit_no_ddr_rule(
        KernelId id, size_t count, GraphKernelNoDdrProof proof) {
    GraphKernelRegisterRule rule;
    rule.kernel_id = id;
    rule.kind = GraphKernelRegisterRuleKind::ExplicitNoDdr;
    rule.no_ddr_proof = proof;
    rule.expected_count = count;
    return rule;
}

GraphKernelRegisterCensusStats phase_stats(
        size_t nodes, size_t dmas, size_t barriers,
        size_t kernels, size_t typed, size_t no_ddr,
        size_t operands, size_t weights, size_t keys, size_t values,
        uint64_t node_hash, uint64_t kernel_hash) {
    GraphKernelRegisterCensusStats stats;
    stats.node_count = nodes;
    stats.dma_count = dmas;
    stats.barrier_count = barriers;
    stats.kernel_count = kernels;
    stats.typed_kernel_count = typed;
    stats.no_ddr_kernel_count = no_ddr;
    stats.operand_count = operands;
    stats.weight_operand_count = weights;
    stats.key_cache_operand_count = keys;
    stats.value_cache_operand_count = values;
    stats.semantic_input_operand_count = 0;
    stats.ordered_node_kind_hash = node_hash;
    stats.ordered_kernel_hash = kernel_hash;
    return stats;
}

const GraphKernelRegisterPolicy& gr00t_z2_register_policy() {
    // Exact ordered register census for this profile. The first guarded BUILD
    // confirms it against the active operator asset. Hashes use Graph emission
    // order and GraphNodeKind values.
    static const GraphKernelRegisterPolicy policy = [] {
        GraphKernelRegisterPolicy value;
        value.fingerprint = UINT64_C(0x475230305a325634);  // GR00Z2V4
        value.name = "gr00t_z2_register_census_v4";
        const auto weight = ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 2, 3);
        const auto key_write = ddr_abi(
            GraphDdrRegisterRole::KeyCache,
            GraphDdrRegisterAccess::Write, 8, 9);
        const auto value_write = ddr_abi(
            GraphDdrRegisterRole::ValueCache,
            GraphDdrRegisterAccess::Write, 8, 9);
        const auto key_read = ddr_abi(
            GraphDdrRegisterRole::KeyCache,
            GraphDdrRegisterAccess::Read, 50, 51);
        const auto value_read = ddr_abi(
            GraphDdrRegisterRole::ValueCache,
            GraphDdrRegisterAccess::Read, 52, 53);
        value.rules = {
            typed_rule(KernelId::PL_AT_FP16_M384N96, 942, {weight}),
            typed_rule(KernelId::PL_AT_FP16_M320N128, 12, {weight}),
            typed_rule(KernelId::LLAMA_INSERT_KCACHE_V16, 128,
                       {key_write}),
            typed_rule(KernelId::LLAMA_INSERT_VCACHE_V16, 128,
                       {value_write}),
            typed_rule(KernelId::LLAMA_INSERT_KCACHE, 4, {key_write}),
            typed_rule(KernelId::LLAMA_INSERT_VCACHE, 4, {value_write}),
            typed_rule(KernelId::SDPA_FLASH_ATTN_SPM, 132,
                       {key_read, value_read}),
            explicit_no_ddr_rule(
                KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED, 264,
                GraphKernelNoDdrProof::RingAllReduceSpmResidual),
            explicit_no_ddr_rule(
                KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_NOPACE, 128,
                GraphKernelNoDdrProof::RingAllReduceSpmResidual),
            intrinsic_no_ddr_rule(KernelId::MEMSET_SPM_MULTI_CORE_V2, 328),
            intrinsic_no_ddr_rule(KernelId::LAYER_NORM, 269),
            intrinsic_no_ddr_rule(KernelId::BINARY_SCALAR, 137),
            intrinsic_no_ddr_rule(KernelId::UNARY_GELU_TANH_SPM, 132),
            // Projection bias is consumed by the auto-tile Linear ABI.
            intrinsic_no_ddr_rule(KernelId::BINARY_SAMESHAPE, 12),
            intrinsic_no_ddr_rule(KernelId::UNARY_SPM, 4),
        };
        GraphKernelRegisterRule dynamic_binary;
        dynamic_binary.kernel_name = "binary_1xC_NxC_tileC";
        dynamic_binary.kind = GraphKernelRegisterRuleKind::IntrinsicNoDdr;
        dynamic_binary.expected_count = 1;
        value.rules.push_back(std::move(dynamic_binary));

        value.expected_node_count = 10782;
        value.expected_dma_count = 3033;
        value.expected_barrier_count = 5124;
        value.expected_kernel_count = 2625;
        value.expected_typed_kernel_count = 1350;
        value.expected_no_ddr_kernel_count = 1275;
        value.expected_operand_count = 1482;
        value.expected_weight_operand_count = 954;
        value.expected_key_cache_operand_count = 264;
        value.expected_value_cache_operand_count = 264;
        value.expected_semantic_input_operand_count = 0;
        value.expected_ordered_node_kind_hash =
            UINT64_C(0x1cfe991b4c0a51a5);
        value.expected_ordered_kernel_hash =
            UINT64_C(0x02a89226f052e980);
        value.phases = {
            GraphKernelRegisterPhaseExpectation{
                "vl_encoder",
                phase_stats(
                    866, 296, 504, 66, 36, 30, 40, 24, 8, 8,
                    UINT64_C(0xb07b80c16deb5555),
                    UINT64_C(0x728c2112d7fcf9b9))},
            GraphKernelRegisterPhaseExpectation{
                "denoise",
                phase_stats(
                    9916, 2737, 4620, 2559, 1314, 1245,
                    1442, 930, 256, 256,
                    UINT64_C(0x72c4af3f08d45a75),
                    UINT64_C(0xee5248a5e490c4e4))},
        };
        return value;
    }();
    return policy;
}

}  // namespace

int64_t rpu_gr00t_spm_z2_prepare(int64_t vl_handle,
                                 int64_t denoise_handle,
                                 int64_t seq_len) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "gr00t_spm_z2_prepare must run outside Graph capture");
    TORCH_CHECK(vl_handle > 0 && denoise_handle > 0,
                "gr00t_spm_z2_prepare: handles must be positive");
    TORCH_CHECK(seq_len > 0,
                "gr00t_spm_z2_prepare: seq_len must be positive");
    Gr00tZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(!s.lease,
                "gr00t_spm_z2_prepare: physical lease is active");
    if (s.plan.has_value()) {
        check_identity(s, vl_handle, denoise_handle, seq_len,
                       s.plan_hash, "gr00t_spm_z2_prepare");
        validate_endpoint_specs_locked(s, "gr00t_spm_z2_prepare");
        TORCH_CHECK(live_persistent_top_bytes() == s.persistent_top_bytes,
                    "gr00t_spm_z2_prepare: persistent top changed after prepare");
        return encode_u64(s.plan_hash);
    }

    bool vl_prepare_attempted = false;
    bool denoise_prepare_attempted = false;
    try {
        vl_prepare_attempted = true;
        const auto vl_layout =
            v3::gr00t_z2_internal::prepare_vl_encoder(
                vl_handle, seq_len);
        denoise_prepare_attempted = true;
        const auto denoise_layout =
            v3::gr00t_z2_internal::prepare_denoise(
                denoise_handle, seq_len);
        TORCH_CHECK(vl_layout.temporary_bytes > 0 &&
                        denoise_layout.temporary_bytes > 0,
                    "gr00t_spm_z2_prepare: component scratch must be non-zero");

        const size_t scratch_bytes =
            std::max(vl_layout.temporary_bytes,
                     denoise_layout.temporary_bytes);
        const size_t persistent_top = live_persistent_top_bytes();

        const auto produced =
            v3::gr00t_z2_internal::vl_encoder_port_spec(
                vl_handle, seq_len);
        const auto required =
            v3::gr00t_z2_internal::denoise_port_spec(
                denoise_handle, seq_len);
        const auto route = v3::SpmIdentityRoute::bind(
            kPortRegion, produced, required,
            /*first_write=*/0, /*last_read=*/1);
        const v3::SpmScratchDecl scratch{
            kScratchRegion, static_cast<int64_t>(scratch_bytes),
            /*first_event=*/0, /*last_event=*/1};

        auto plan = v3::SpmPipelinePlan::compile(
            {scratch}, {route}, persistent_top,
            SpmAllocator::SPM_PLANNING_BUDGET);

        s.vl_handle = vl_handle;
        s.denoise_handle = denoise_handle;
        s.seq_len = seq_len;
        s.persistent_top_bytes = persistent_top;
        s.plan_hash = plan.hash();
        s.plan = std::move(plan);
        validate_endpoint_specs_locked(s, "gr00t_spm_z2_prepare");
        return encode_u64(s.plan_hash);
    } catch (...) {
        std::exception_ptr failure = std::current_exception();
        if (denoise_prepare_attempted) cleanup_step(failure, [&] {
            v3::gr00t_z2_internal::unprepare_denoise(
                denoise_handle, seq_len);
        });
        if (vl_prepare_attempted) cleanup_step(failure, [&] {
            v3::gr00t_z2_internal::unprepare_vl_encoder(
                vl_handle, seq_len);
        });
        reset_prepared_state_locked(s);
        std::rethrow_exception(failure);
    }
}

void rpu_gr00t_spm_z2_unprepare(
    int64_t vl_handle,
    int64_t denoise_handle,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "gr00t_spm_z2_unprepare must run outside Graph capture");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Gr00tZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(!s.lease,
                "gr00t_spm_z2_unprepare: physical lease is active");
    check_identity(s, vl_handle, denoise_handle, s.seq_len, plan_hash,
                   "gr00t_spm_z2_unprepare");
    validate_endpoint_specs_locked(s, "gr00t_spm_z2_unprepare");

    std::exception_ptr failure;
    cleanup_step(failure, [&] {
        v3::gr00t_z2_internal::unprepare_denoise(
            denoise_handle, s.seq_len);
    });
    cleanup_step(failure, [&] {
        v3::gr00t_z2_internal::unprepare_vl_encoder(
            vl_handle, s.seq_len);
    });
    if (failure) std::rethrow_exception(failure);
    reset_prepared_state_locked(s);
}

int64_t rpu_gr00t_spm_z2_begin(int64_t vl_handle,
                               int64_t denoise_handle,
                               int64_t seq_len,
                               int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "gr00t_spm_z2_begin must run outside Graph capture");
    Gr00tZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    check_identity(s, vl_handle, denoise_handle, seq_len, plan_hash,
                   "gr00t_spm_z2_begin");
    TORCH_CHECK(!s.lease,
                "gr00t_spm_z2_begin: a physical lease is already active");
    validate_endpoint_specs_locked(s, "gr00t_spm_z2_begin");
    TORCH_CHECK(live_persistent_top_bytes() == s.persistent_top_bytes,
                "gr00t_spm_z2_begin: persistent top changed after prepare");

    auto lease = v3::SpmPipelineLease::acquire_physical(*s.plan);
    const auto scratch = lease->scratch(kScratchRegion);
    const auto port = lease->port(kPortRegion);
    bool vl_adopted = false;
    bool denoise_adopted = false;
    try {
        v3::gr00t_z2_internal::adopt_vl_encoder(
            vl_handle, *lease, scratch);
        vl_adopted = true;
        v3::gr00t_z2_internal::adopt_denoise(
            denoise_handle, *lease, scratch);
        denoise_adopted = true;
        v3::gr00t_z2_internal::bind_vl_encoder(
            vl_handle, *lease, port);
        v3::gr00t_z2_internal::bind_denoise(
            denoise_handle, *lease, port);
        lease->seal_physical();
        v3::gr00t_z2_internal::validate_vl_encoder(vl_handle, *lease);
        v3::gr00t_z2_internal::validate_denoise(denoise_handle, *lease);
    } catch (...) {
        std::exception_ptr failure = std::current_exception();
        if (denoise_adopted) cleanup_step(failure, [&] {
            v3::gr00t_z2_internal::clear_denoise(
                denoise_handle, lease->epoch(), lease->plan_hash());
        });
        if (vl_adopted) cleanup_step(failure, [&] {
            v3::gr00t_z2_internal::clear_vl_encoder(
                vl_handle, lease->epoch(), lease->plan_hash());
        });
        if (!lease->physical_sealed()) cleanup_step(failure, [&] {
            lease->seal_physical();
        });
        cleanup_step(failure, [&] { lease->release(); });
        lease.reset();
        std::rethrow_exception(failure);
    }

    const uint64_t epoch = lease->epoch();
    s.lease = std::move(lease);
    return encode_u64(epoch);
}

void rpu_gr00t_spm_z2_end(int64_t encoded_epoch,
                          int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "gr00t_spm_z2_end must run outside Graph capture");
    Gr00tZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.lease,
                "gr00t_spm_z2_end: no physical lease is active");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    TORCH_CHECK(s.lease->epoch() == epoch,
                "gr00t_spm_z2_end: stale epoch; expected ",
                s.lease->epoch(), ", got ", epoch);
    TORCH_CHECK(s.lease->plan_hash() == plan_hash &&
                    s.plan_hash == plan_hash,
                "gr00t_spm_z2_end: stale plan hash");
    // Establish caller identity/ownership before any model or allocator state
    // is changed. Stale-token and wrong-thread callers must fail non-mutating.
    s.lease->validate_physical_identity(epoch, plan_hash);

    std::exception_ptr failure;
    cleanup_step(failure, [&] {
        v3::gr00t_z2_internal::validate_vl_encoder(
            s.vl_handle, *s.lease);
        v3::gr00t_z2_internal::validate_denoise(
            s.denoise_handle, *s.lease);
    });
    cleanup_step(failure, [&] {
        v3::gr00t_z2_internal::clear_denoise(
            s.denoise_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] {
        v3::gr00t_z2_internal::clear_vl_encoder(
            s.vl_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] { s.lease->release(); });
    s.lease.reset();
    if (failure) std::rethrow_exception(failure);
}

void rpu_gr00t_spm_z2_prepare_masks(
    int64_t denoise_handle,
    const at::Tensor& tmask,
    const at::Tensor& imask,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "gr00t_spm_z2_prepare_masks must run outside Graph capture");
    Gr00tZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    TORCH_CHECK(s.plan.has_value() && !s.lease,
                "gr00t_spm_z2_prepare_masks requires a prepared inactive plan");
    TORCH_CHECK(s.denoise_handle == denoise_handle &&
                    s.plan_hash == plan_hash,
                "gr00t_spm_z2_prepare_masks: handle or plan hash mismatch");
    v3::gr00t_z2_internal::prepare_denoise_masks(
        denoise_handle, tmask, imask, plan_hash);
}

at::Tensor rpu_gr00t_spm_z2_pipeline_forward(
    int64_t vl_handle,
    int64_t denoise_handle,
    const at::Tensor& raw,
    at::TensorList vl_k_caches,
    at::TensorList vl_v_caches,
    const at::Tensor& noise,
    const at::Tensor& state_tensor,
    at::TensorList denoise_k_caches,
    at::TensorList denoise_v_caches,
    int64_t epoch,
    int64_t plan_hash) {
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "gr00t_spm_z2_pipeline_forward requires one active outer Graph");
    const uint64_t live_epoch = decode_u64(epoch);
    const uint64_t physical_plan_hash = decode_u64(plan_hash);
    uint64_t physical_digest = 0;
    {
        Gr00tZ2State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        TORCH_CHECK(s.vl_handle == vl_handle &&
                        s.denoise_handle == denoise_handle &&
                        s.seq_len == 82,
                    "gr00t_spm_z2_pipeline_forward: register census V1 "
                    "requires the exact prepared GR00T seq=82 profile");
        validate_active_pipeline_locked(
            s, live_epoch, physical_plan_hash,
            "gr00t_spm_z2_pipeline_forward preflight");
        physical_digest = physical_binding_digest(s);
    }

    GraphKernelRegisterCensusGuard register_census(
        RpuKernelGraph::active(), gr00t_z2_register_policy(),
        physical_plan_hash, live_epoch);
    register_census.stage_outer_fast_physical_digest(physical_digest);
    v3::gr00t_z2_internal::stage_vl_outer_fast_component(
        vl_handle, register_census, vl_k_caches, vl_v_caches);
    v3::gr00t_z2_internal::stage_denoise_outer_fast_component(
        denoise_handle, register_census,
        denoise_k_caches, denoise_v_caches);
    v3::gr00t_z2_internal::stage_denoise_outer_fast_invocation_authority(
        denoise_handle, register_census, physical_plan_hash);

    if (RpuKernelGraph::active().state() ==
        RpuKernelGraph::State::REPLAYING) {
        v3::gr00t_z2_internal::bind_vl_outer_fast_input(
            vl_handle, register_census, raw);
        at::Tensor output =
            v3::gr00t_z2_internal::prepare_denoise_outer_fast_replay(
                denoise_handle, register_census, noise, state_tensor,
                live_epoch, physical_plan_hash);
        auto ticket = register_census.validate_outer_fast_replay();
        {
            Gr00tZ2State& s = state();
            std::lock_guard<std::mutex> lock(s.mutex);
            validate_active_pipeline_locked(
                s, live_epoch, physical_plan_hash,
                "gr00t_spm_z2_pipeline_forward fast postflight");
            TORCH_CHECK(physical_binding_digest(s) == physical_digest,
                        "gr00t_spm_z2_pipeline_forward fast physical "
                        "binding drifted");
        }
        register_census.commit_outer_fast_replay(std::move(ticket));
        return output;
    }

    rpu_gr00t_vl_encoder_forward_z2(
        vl_handle, raw, vl_k_caches, vl_v_caches, epoch, plan_hash);
    register_census.checkpoint("vl_encoder");
    at::Tensor output =
        v3::gr00t_z2_internal::forward_denoise_prepared_masks(
        denoise_handle, noise, state_tensor,
        denoise_k_caches, denoise_v_caches,
        live_epoch, physical_plan_hash);
    {
        Gr00tZ2State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        validate_active_pipeline_locked(
            s, live_epoch, physical_plan_hash,
            "gr00t_spm_z2_pipeline_forward postflight");
    }
    register_census.complete();
    return output;
}

namespace v3::gr00t_z2_internal {

void validate_vl_dispatch(int64_t handle,
                          uint64_t epoch,
                          uint64_t plan_hash) {
    Gr00tZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.vl_handle == handle,
                "gr00t VL Z2 dispatch: active handle mismatch");
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "gr00t VL Z2 dispatch");
}

void validate_denoise_dispatch(int64_t handle,
                               uint64_t epoch,
                               uint64_t plan_hash) {
    Gr00tZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.denoise_handle == handle,
                "gr00t denoise Z2 dispatch: active handle mismatch");
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "gr00t denoise Z2 dispatch");
}

void check_vl_destroy_allowed(int64_t handle) {
    Gr00tZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK((!s.lease && !s.plan.has_value()) ||
                    s.vl_handle != handle,
                "cannot destroy the GR00T VL handle while its Z2 plan is "
                "prepared or active; clear the outer GraphCache and "
                "unprepare Z2 first");
}

void check_denoise_destroy_allowed(int64_t handle) {
    Gr00tZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK((!s.lease && !s.plan.has_value()) ||
                    s.denoise_handle != handle,
                "cannot destroy the GR00T denoise handle while its Z2 plan "
                "is prepared or active; clear the outer GraphCache and "
                "unprepare Z2 first");
}

}  // namespace v3::gr00t_z2_internal
