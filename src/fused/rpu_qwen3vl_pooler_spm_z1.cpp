#include "rpu_qwen3vl_pooler_spm_z1.h"

#include "core/rpu_kernel_decls.h"
#include "core/rpu_ops.h"
#include "core/rpu_spm_allocator.h"
#include "graph/graph_runtime.h"

#include <c10/util/Exception.h>

#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kNumPatches = 256;
constexpr int64_t kMergedRows = 64;
constexpr int64_t kExecutionLen = 128;
constexpr int64_t kImageRowBegin = 4;
constexpr int64_t kRealLen = 72;
constexpr int64_t kGrootRealLen = 82;
constexpr int64_t kTextHidden = 2048;
constexpr int64_t kRouteHalfRows = 32;
constexpr int kNumCores = 8;
constexpr size_t kComponentTemporaryBytes = 1'966'080;
constexpr size_t kExpectedDynamicPeakBytes = 2'490'368;
constexpr size_t kExpectedDeepstack1DynamicPeakBytes = 4'194'304;
constexpr size_t kExpectedDeepstack3DynamicPeakBytes = 4'718'592;
constexpr v3::SpmScratchId kVisionScratch{1};
constexpr v3::SpmScratchId kTextScratch{2};
constexpr v3::SpmPortId kPoolerSourcePort{1};
constexpr v3::SpmPortId kTextDestinationPort{2};
constexpr std::array<v3::SpmPortId, 3> kDeepstackPorts = {
    v3::SpmPortId{3}, v3::SpmPortId{4}, v3::SpmPortId{5}};
constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

struct Qwen3VlPoolerZ1State {
    std::mutex mutex;
    std::optional<v3::SpmPipelinePlan> plan;
    std::unique_ptr<v3::SpmPipelineLease> lease;
    int64_t vision_handle = 0;
    int64_t text_handle = 0;
    int64_t num_patches = 0;
    int64_t execution_len = 0;
    int64_t image_row_begin = 0;
    int64_t real_len = 0;
    size_t vision_temporary_bytes = 0;
    size_t text_temporary_bytes = 0;
    size_t persistent_top_bytes = 0;
    uint64_t route_hash = 0;
    uint64_t plan_hash = 0;
    int64_t retained_deepstack_count = 0;
    uint32_t source_addr = 0;
    uint32_t destination_addr = 0;
    std::array<uint32_t, 3> deepstack_addrs{};
    uint64_t hidden_prefix_src_base = 0;
    uint64_t hidden_suffix_src_base = 0;
    at::Tensor hidden_ref;
    std::vector<v3::SpmContiguousRowRun> lowered_runs;
    std::shared_ptr<uint64_t> invocation_prepared_generation =
        std::make_shared<uint64_t>(0);
    std::shared_ptr<uint64_t> invocation_consumed_generation =
        std::make_shared<uint64_t>(0);
    uint64_t active_invocation_generation = 0;
    bool partial_mrope = false;
};

Qwen3VlPoolerZ1State& state() {
    static Qwen3VlPoolerZ1State value;
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

struct AllocatorSnapshot {
    bool initialized = false;
    uint64_t generation = 0;
    uint64_t persistent_generation = 0;
    size_t temporary_bytes = 0;
    size_t persistent_bytes = 0;
    size_t super_persistent_bytes = 0;
    int active_instance_count = 0;
};

AllocatorSnapshot allocator_snapshot() {
    return {
        SPM_ALLOC.is_initialized(),
        SPM_ALLOC.generation(),
        SPM_ALLOC.persistent_generation(),
        SPM_ALLOC.temporary_used(),
        SPM_ALLOC.persistent_used(),
        SPM_ALLOC.super_persistent_used(),
        SPM_ALLOC.active_instance_count(),
    };
}

void check_allocator_unchanged(const AllocatorSnapshot& before,
                               const char* stage) {
    const AllocatorSnapshot after = allocator_snapshot();
    TORCH_CHECK(
        after.initialized == before.initialized &&
            after.generation == before.generation &&
            after.persistent_generation == before.persistent_generation &&
            after.temporary_bytes == before.temporary_bytes &&
            after.persistent_bytes == before.persistent_bytes &&
            after.super_persistent_bytes == before.super_persistent_bytes &&
            after.active_instance_count == before.active_instance_count,
        "qwen3vl_multiview_spm_dry_probe: allocator changed ", stage);
}

uint64_t hash_u64(uint64_t hash, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash = (hash ^ static_cast<uint8_t>(value >> shift)) * kFnvPrime;
    }
    return hash;
}

uint64_t hash_port_spec(uint64_t hash, const v3::SpmDense2DSpec& spec) {
    hash = hash_u64(hash, static_cast<uint8_t>(spec.dtype));
    hash = hash_u64(hash, static_cast<uint64_t>(spec.rows));
    hash = hash_u64(hash, static_cast<uint64_t>(spec.cols));
    return hash_u64(hash, static_cast<uint8_t>(spec.distribution));
}

bool ranges_overlap(size_t lhs_begin,
                    size_t lhs_size,
                    size_t rhs_begin,
                    size_t rhs_size) {
    return lhs_begin < rhs_begin + rhs_size &&
           rhs_begin < lhs_begin + lhs_size;
}

const std::vector<int32_t>& identity_permutation() {
    static const std::vector<int32_t> value = [] {
        std::vector<int32_t> result(kMergedRows);
        std::iota(result.begin(), result.end(), int32_t{0});
        return result;
    }();
    return value;
}

void check_canary_shape(int64_t num_patches,
                        int64_t execution_len,
                        int64_t image_row_begin,
                        int64_t real_len,
                        int64_t retained_deepstack_count,
                        const char* op) {
    const bool qwen3vl_profile =
        num_patches == kNumPatches && execution_len == kExecutionLen &&
        image_row_begin == kImageRowBegin && real_len == kRealLen;
    const bool groot_profile =
        num_patches == kNumPatches && execution_len == kExecutionLen &&
        image_row_begin == kImageRowBegin && real_len == kGrootRealLen &&
        retained_deepstack_count == 3;
    TORCH_CHECK(
        qwen3vl_profile || groot_profile,
        op, ": accepts only Qwen3-VL patches/execution/image_begin/real="
        "(256,128,4,72), or GR00T (256,128,4,82) with retained DeepStack "
        "count 3; got (", num_patches, ",", execution_len, ",",
        image_row_begin, ",", real_len, ") with retained count ",
        retained_deepstack_count);
}

void check_retained_deepstack_count(int64_t count, const char* op) {
    TORCH_CHECK(count == 0 || count == 1 || count == 3,
                op, ": retained_deepstack_count must be exactly 0, 1, or 3; "
                "got ", count);
}

v3::SpmPermutationRowRunsRoute endpoint_route(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin) {
    const auto source =
        v3::qwen3vl_pooler_z1_internal::vision_source_spec(
            vision_handle, num_patches);
    const auto destination =
        v3::qwen3vl_pooler_z1_internal::text_destination_spec(
            text_handle, execution_len);
    TORCH_CHECK(source.rows == kMergedRows && source.cols == kTextHidden &&
                    destination.rows == kExecutionLen &&
                    destination.cols == kTextHidden,
                "qwen3vl_pooler_spm_z1: component endpoint spec drifted");
    return v3::SpmPermutationRowRunsRoute::bind(
        kPoolerSourcePort, source,
        kTextDestinationPort, destination,
        identity_permutation(),
        {v3::SpmDstRowRun{image_row_begin, kMergedRows}},
        {/*source_first_write=*/0,
         /*destination_first_write=*/1,
         /*transform=*/1,
         /*destination_last_read=*/2});
}

v3::SpmIdentityRoute deepstack_route(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t ordinal) {
    TORCH_CHECK(ordinal >= 0 && ordinal < 3,
                "qwen3vl_pooler_spm_z1: retained DeepStack ordinal must be "
                "in [0,3), got ", ordinal);
    const auto produced =
        v3::qwen3vl_pooler_z1_internal::vision_deepstack_source_spec(
            vision_handle, num_patches, ordinal);
    const auto required =
        v3::qwen3vl_pooler_z1_internal::text_deepstack_spec(
            text_handle, ordinal);
    TORCH_CHECK(produced.rows == kMergedRows &&
                    produced.cols == kTextHidden && produced == required,
                "qwen3vl_pooler_spm_z1: retained DeepStack endpoint spec "
                "drifted at ordinal ", ordinal);
    return v3::SpmIdentityRoute::bind(
        kDeepstackPorts.at(static_cast<size_t>(ordinal)), produced, required,
        /*first_write=*/0, /*last_read=*/2);
}

std::vector<v3::SpmIdentityRoute> deepstack_routes(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t count) {
    check_retained_deepstack_count(
        count, "qwen3vl_pooler_spm_z1 retained routes");
    std::vector<v3::SpmIdentityRoute> routes;
    routes.reserve(static_cast<size_t>(count));
    for (int64_t ordinal = 0; ordinal < count; ++ordinal) {
        routes.push_back(deepstack_route(
            vision_handle, text_handle, num_patches, ordinal));
    }
    return routes;
}

v3::SpmPermutationRowRunsRoute endpoint_route(
    const Qwen3VlPoolerZ1State& s) {
    return endpoint_route(
        s.vision_handle, s.text_handle, s.num_patches,
        s.execution_len, s.image_row_begin);
}

void validate_prepared_locked(const Qwen3VlPoolerZ1State& s,
                              const char* op) {
    TORCH_CHECK(s.plan.has_value(), op, ": no prepared plan");
    check_retained_deepstack_count(s.retained_deepstack_count, op);
    check_canary_shape(s.num_patches, s.execution_len,
                       s.image_row_begin, s.real_len,
                       s.retained_deepstack_count, op);
    TORCH_CHECK(s.vision_handle > 0 && s.text_handle > 0 &&
                    s.vision_temporary_bytes == kComponentTemporaryBytes &&
                    s.text_temporary_bytes == kComponentTemporaryBytes &&
                    s.plan_hash == s.plan->hash() && s.plan_hash != 0,
                op, ": prepared component identity drifted");
    const auto route = endpoint_route(s);
    const size_t expected_peak = s.retained_deepstack_count == 3
        ? kExpectedDeepstack3DynamicPeakBytes
        : (s.retained_deepstack_count == 1
               ? kExpectedDeepstack1DynamicPeakBytes
               : kExpectedDynamicPeakBytes);
    TORCH_CHECK(route.hash() == s.route_hash &&
                    route.hash() == s.plan->permutation_row_runs_hash(
                                        kPoolerSourcePort,
                                        kTextDestinationPort) &&
                    route.source_spec() ==
                        s.plan->port_spec(kPoolerSourcePort) &&
                    route.destination_spec() ==
                        s.plan->port_spec(kTextDestinationPort) &&
                    s.plan->peak_bytes() == expected_peak,
                op, ": prepared route or exact dry-plan peak drifted");
    const auto retained_routes = deepstack_routes(
        s.vision_handle, s.text_handle, s.num_patches,
        s.retained_deepstack_count);
    for (size_t ordinal = 0; ordinal < kDeepstackPorts.size(); ++ordinal) {
        const bool requested =
            ordinal < static_cast<size_t>(s.retained_deepstack_count);
        TORCH_CHECK(
            s.plan->contains(kDeepstackPorts[ordinal]) == requested,
            op, ": retained DeepStack port presence drifted at ordinal ",
            ordinal);
        if (requested) {
            TORCH_CHECK(
                s.plan->port_spec(kDeepstackPorts[ordinal]) ==
                    retained_routes[ordinal].spec(),
                op, ": retained DeepStack route drifted at ordinal ", ordinal);
        }
    }
}

void check_identity(const Qwen3VlPoolerZ1State& s,
                    int64_t vision_handle,
                    int64_t text_handle,
                    int64_t num_patches,
                    int64_t execution_len,
                    int64_t image_row_begin,
                    int64_t real_len,
                    uint64_t plan_hash,
                    const char* op) {
    validate_prepared_locked(s, op);
    TORCH_CHECK(s.vision_handle == vision_handle &&
                    s.text_handle == text_handle &&
                    s.num_patches == num_patches &&
                    s.execution_len == execution_len &&
                    s.image_row_begin == image_row_begin &&
                    s.real_len == real_len &&
                    s.plan_hash == plan_hash,
                op, ": stale handle, profile, or plan token");
}

template <typename Fn>
void cleanup_step(std::exception_ptr& failure, Fn&& fn) noexcept {
    try {
        fn();
    } catch (...) {
        if (!failure) failure = std::current_exception();
    }
}

void reset_begin_state_locked(Qwen3VlPoolerZ1State& s) {
    TORCH_INTERNAL_ASSERT(!s.lease);
    s.source_addr = 0;
    s.destination_addr = 0;
    s.deepstack_addrs.fill(0);
    s.hidden_prefix_src_base = 0;
    s.hidden_suffix_src_base = 0;
    s.hidden_ref = at::Tensor{};
    s.lowered_runs.clear();
    s.active_invocation_generation = 0;
    s.partial_mrope = false;
}

void reset_prepared_state_locked(Qwen3VlPoolerZ1State& s) {
    TORCH_INTERNAL_ASSERT(!s.lease);
    reset_begin_state_locked(s);
    s.plan.reset();
    s.vision_handle = 0;
    s.text_handle = 0;
    s.num_patches = 0;
    s.execution_len = 0;
    s.image_row_begin = 0;
    s.real_len = 0;
    s.vision_temporary_bytes = 0;
    s.text_temporary_bytes = 0;
    s.persistent_top_bytes = 0;
    s.route_hash = 0;
    s.plan_hash = 0;
    s.retained_deepstack_count = 0;
}

void validate_active_pipeline_locked(Qwen3VlPoolerZ1State& s,
                                     uint64_t epoch,
                                     uint64_t plan_hash,
                                     const char* op) {
    TORCH_CHECK(s.lease, op, ": no physical lease is active");
    TORCH_CHECK(s.plan_hash == plan_hash,
                op, ": stale plan hash; expected ", s.plan_hash,
                ", got ", plan_hash);
    s.lease->validate_physical(epoch, plan_hash, /*require_sealed=*/true);
    v3::qwen3vl_pooler_z1_internal::validate_vision(
        s.vision_handle, *s.lease);
    v3::qwen3vl_pooler_z1_internal::validate_text(
        s.text_handle, *s.lease);
}

uint64_t physical_binding_digest(const Qwen3VlPoolerZ1State& s) {
    TORCH_CHECK(s.lease && s.lease->physical() &&
                    s.lease->physical_sealed(),
                "Qwen3-VL pooler Z1 physical digest requires one sealed "
                "lease");
    const auto vision_scratch = s.lease->scratch(kVisionScratch);
    const auto text_scratch = s.lease->scratch(kTextScratch);
    const auto route = s.lease->permutation_row_runs(
        kPoolerSourcePort, kTextDestinationPort);
    const auto& source = route.source();
    const auto& destination = route.destination();
    const auto lowered_runs = route.lower_contiguous_runs();
    TORCH_CHECK(route.route_hash() == s.route_hash &&
                    source.resolve_physical_addr(0, *s.lease) ==
                        s.source_addr &&
                    destination.resolve_physical_addr(0, *s.lease) ==
                        s.destination_addr &&
                    lowered_runs.size() == 1 &&
                    s.lowered_runs.size() == lowered_runs.size() &&
                    lowered_runs.front().source_row == 0 &&
                    lowered_runs.front().destination_row == kImageRowBegin &&
                    lowered_runs.front().row_count == kMergedRows &&
                    s.lowered_runs.front().source_row ==
                        lowered_runs.front().source_row &&
                    s.lowered_runs.front().destination_row ==
                        lowered_runs.front().destination_row &&
                    s.lowered_runs.front().row_count ==
                        lowered_runs.front().row_count,
                "Qwen3-VL pooler Z1 physical digest route drifted");
    for (size_t ordinal = 0; ordinal < kDeepstackPorts.size(); ++ordinal) {
        const bool requested =
            ordinal < static_cast<size_t>(s.retained_deepstack_count);
        if (requested) {
            const auto retained = s.lease->port(kDeepstackPorts[ordinal]);
            TORCH_CHECK(
                retained.spec() ==
                        deepstack_route(
                            s.vision_handle, s.text_handle, s.num_patches,
                            static_cast<int64_t>(ordinal)).spec() &&
                    retained.resolve_physical_addr(0, *s.lease) ==
                        s.deepstack_addrs[ordinal],
                "Qwen3-VL pooler Z1 physical digest retained DeepStack "
                "drifted at ordinal ", ordinal);
        } else {
            TORCH_CHECK(
                s.deepstack_addrs[ordinal] == 0,
                "Qwen3-VL pooler Z1 physical digest retained an unrequested "
                "DeepStack address at ordinal ", ordinal);
        }
    }

    uint64_t hash = kFnvOffset;
    hash = hash_u64(hash, UINT64_C(0x5133564c5a315048));  // Q3VLZ1PH
    hash = hash_u64(hash, s.plan_hash);
    hash = hash_u64(hash, s.lease->physical_arena_base());
    hash = hash_u64(hash, s.lease->physical_arena_end());
    hash = hash_u64(hash, s.lease->physical_top_reservation());
    hash = hash_u64(
        hash, static_cast<uint64_t>(s.retained_deepstack_count));
    hash = hash_u64(
        hash, vision_scratch.resolve_physical_offset(*s.lease));
    hash = hash_u64(hash, vision_scratch.size_bytes());
    hash = hash_u64(
        hash, text_scratch.resolve_physical_offset(*s.lease));
    hash = hash_u64(hash, text_scratch.size_bytes());
    hash = hash_u64(hash, source.resolve_physical_offset(*s.lease));
    hash = hash_u64(hash, source.size_bytes());
    hash = hash_port_spec(hash, source.spec());
    hash = hash_u64(hash, destination.resolve_physical_offset(*s.lease));
    hash = hash_u64(hash, destination.size_bytes());
    hash = hash_port_spec(hash, destination.spec());
    for (int core = 0; core < kNumCores; ++core) {
        hash = hash_u64(
            hash, vision_scratch.resolve_physical_addr(core, *s.lease));
        hash = hash_u64(
            hash, text_scratch.resolve_physical_addr(core, *s.lease));
        hash = hash_u64(
            hash, source.resolve_physical_addr(core, *s.lease));
        hash = hash_u64(
            hash, destination.resolve_physical_addr(core, *s.lease));
        for (int64_t ordinal = 0;
             ordinal < s.retained_deepstack_count; ++ordinal) {
            const auto retained = s.lease->port(
                kDeepstackPorts[static_cast<size_t>(ordinal)]);
            hash = hash_u64(
                hash, retained.resolve_physical_addr(core, *s.lease));
        }
    }
    for (int64_t ordinal = 0;
         ordinal < s.retained_deepstack_count; ++ordinal) {
        const auto retained = s.lease->port(
            kDeepstackPorts[static_cast<size_t>(ordinal)]);
        hash = hash_u64(
            hash, retained.resolve_physical_offset(*s.lease));
        hash = hash_u64(hash, retained.size_bytes());
        hash = hash_port_spec(hash, retained.spec());
        hash = hash_u64(
            hash, static_cast<uint64_t>(s.deepstack_addrs[ordinal]));
    }
    hash = hash_u64(hash, route.route_hash());
    hash = hash_u64(hash, static_cast<uint64_t>(s.num_patches));
    hash = hash_u64(hash, static_cast<uint64_t>(kMergedRows));
    hash = hash_u64(hash, static_cast<uint64_t>(s.execution_len));
    hash = hash_u64(hash, static_cast<uint64_t>(s.image_row_begin));
    hash = hash_u64(hash, static_cast<uint64_t>(s.real_len));
    hash = hash_u64(hash, static_cast<uint64_t>(kTextHidden));
    hash = hash_u64(hash, lowered_runs.size());
    for (const auto& run : lowered_runs) {
        hash = hash_u64(hash, static_cast<uint64_t>(run.source_row));
        hash = hash_u64(hash, static_cast<uint64_t>(run.destination_row));
        hash = hash_u64(hash, static_cast<uint64_t>(run.row_count));
    }
    return hash == 0 ? UINT64_C(1) : hash;
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

GraphKernelRegisterRule explicit_dynamic_no_ddr_rule(
        const char* name, size_t count, GraphKernelNoDdrProof proof) {
    GraphKernelRegisterRule rule;
    rule.kernel_name = name;
    rule.kind = GraphKernelRegisterRuleKind::ExplicitNoDdr;
    rule.no_ddr_proof = proof;
    rule.expected_count = count;
    return rule;
}

GraphKernelRegisterCensusStats phase_stats(
        size_t nodes, size_t dmas, size_t barriers,
        size_t kernels, size_t typed, size_t no_ddr,
        size_t operands, size_t weights, size_t keys, size_t values,
        size_t semantic_inputs,
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
    stats.semantic_input_operand_count = semantic_inputs;
    stats.ordered_node_kind_hash = node_hash;
    stats.ordered_kernel_hash = kernel_hash;
    return stats;
}

// The formula-specific GELU v2 hashes below are deterministic substitutions
// over the frozen v1 streams.  The first guarded native BUILD remains the
// authoritative recensus for every profile.
const GraphKernelRegisterPolicy& qwen3vl_pooler_z1_register_policy() {
    // Frozen from an address-independent captured BUILD stream.
    // Typed roles come from owner-bound launchers and are revalidated on the
    // first guarded BUILD.
    static const GraphKernelRegisterPolicy policy = [] {
        GraphKernelRegisterPolicy value;
        value.fingerprint = UINT64_C(0x5133564c5a315633);  // Q3VLZ1V3
        value.name = "qwen3vl_pooler_z1_register_census_v3";

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
        const auto rope_position = ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 0, 1);
        const auto mrope_cos = ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 6, 7);
        const auto mrope_sin = ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 8, 9);
        const auto mrope_position = ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 12, 13);

        value.rules = {
            typed_rule(KernelId::PL_AT_FP16_M320N128, 52, {weight}),
            typed_rule(KernelId::PL_AT_FP16_M384N96, 162, {weight}),
            typed_rule(KernelId::PL_AT_FP16_M480N64, 128, {weight}),
            typed_rule(KernelId::ROPE_2D_SPM, 48, {rope_position}),
            typed_rule(KernelId::MROPE, 56,
                       {mrope_cos, mrope_sin, mrope_position}),
            typed_rule(KernelId::LLAMA_INSERT_KCACHE_V16, 52,
                       {key_write}),
            typed_rule(KernelId::LLAMA_INSERT_VCACHE_V16, 52,
                       {value_write}),
            typed_rule(KernelId::SDPA_FLASH_ATTN_SPM, 52,
                       {key_read, value_read}),
            explicit_no_ddr_rule(
                KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED, 105,
                GraphKernelNoDdrProof::RingAllReduceSpmResidual),
            intrinsic_no_ddr_rule(KernelId::MEMSET_SPM_MULTI_CORE_V2, 50),
            intrinsic_no_ddr_rule(KernelId::LAYER_NORM, 49),
            intrinsic_no_ddr_rule(KernelId::RMS_NORM_BF16_SPM, 113),
            intrinsic_no_ddr_rule(KernelId::BINARY_SAMESHAPE, 28),
            intrinsic_no_ddr_rule(KernelId::UNARY_SPM, 28),
            intrinsic_no_ddr_rule(KernelId::UNARY_GELU_TANH_SPM, 24),
            intrinsic_no_ddr_rule(KernelId::UNARY_GELU_ERF_SPM, 1),
            explicit_dynamic_no_ddr_rule(
                "fill", 1, GraphKernelNoDdrProof::FillSpm),
            explicit_dynamic_no_ddr_rule(
                "slice_single_axis", 2,
                GraphKernelNoDdrProof::SliceSpm),
        };

        value.expected_node_count = 10052;
        value.expected_dma_count = 3337;
        value.expected_barrier_count = 5712;
        value.expected_kernel_count = 1003;
        value.expected_typed_kernel_count = 602;
        value.expected_no_ddr_kernel_count = 401;
        value.expected_operand_count = 766;
        value.expected_weight_operand_count = 454;
        value.expected_key_cache_operand_count = 104;
        value.expected_value_cache_operand_count = 104;
        value.expected_semantic_input_operand_count = 104;
        value.expected_ordered_node_kind_hash =
            UINT64_C(0xa3d08d6dc871e859);
        value.expected_ordered_kernel_hash =
            UINT64_C(0xef7fcd5d46561585);
        value.phases = {
            GraphKernelRegisterPhaseExpectation{
                "vision",
                phase_stats(
                    6957, 2416, 4102, 439, 266, 173,
                    290, 146, 48, 48, 48,
                    UINT64_C(0xdcc64c17aa5e7f45),
                    UINT64_C(0xeb9c5bbe550fd821))},
            GraphKernelRegisterPhaseExpectation{
                "handoff",
                phase_stats(
                    47, 16, 28, 3, 0, 3,
                    0, 0, 0, 0, 0,
                    UINT64_C(0xa21f233d7817267b),
                    UINT64_C(0x88028aeb1847a09c))},
            GraphKernelRegisterPhaseExpectation{
                "text",
                phase_stats(
                    3048, 905, 1582, 561, 336, 225,
                    476, 308, 56, 56, 56,
                    UINT64_C(0xa3cf00264ef1eb03),
                    UINT64_C(0x40a4fb74c129a684))},
        };
        return value;
    }();
    return policy;
}

const GraphKernelRegisterPolicy& qwen3vl_pooler_ds1_register_policy() {
    // Frozen from the pooler policy plus the existing Vision merger emitted
    // immediately at tap 5 and one existing Text layer-0 SPM add.  The first
    // guarded BUILD validates the exact address-independent stream.
    static const GraphKernelRegisterPolicy policy = [] {
        GraphKernelRegisterPolicy value;
        value.fingerprint = UINT64_C(0x5133564c44533133);  // Q3VLDS13
        value.name = "qwen3vl_pooler_ds1_register_census_v3";

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
        const auto rope_position = ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 0, 1);
        const auto mrope_cos = ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 6, 7);
        const auto mrope_sin = ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 8, 9);
        const auto mrope_position = ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 12, 13);

        value.rules = {
            typed_rule(KernelId::PL_AT_FP16_M320N128, 52, {weight}),
            typed_rule(KernelId::PL_AT_FP16_M384N96, 164, {weight}),
            typed_rule(KernelId::PL_AT_FP16_M480N64, 128, {weight}),
            typed_rule(KernelId::ROPE_2D_SPM, 48, {rope_position}),
            typed_rule(KernelId::MROPE, 56,
                       {mrope_cos, mrope_sin, mrope_position}),
            typed_rule(KernelId::LLAMA_INSERT_KCACHE_V16, 52,
                       {key_write}),
            typed_rule(KernelId::LLAMA_INSERT_VCACHE_V16, 52,
                       {value_write}),
            typed_rule(KernelId::SDPA_FLASH_ATTN_SPM, 52,
                       {key_read, value_read}),
            explicit_no_ddr_rule(
                KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED, 106,
                GraphKernelNoDdrProof::RingAllReduceSpmResidual),
            intrinsic_no_ddr_rule(KernelId::MEMSET_SPM_MULTI_CORE_V2, 52),
            intrinsic_no_ddr_rule(KernelId::LAYER_NORM, 50),
            intrinsic_no_ddr_rule(KernelId::RMS_NORM_BF16_SPM, 113),
            intrinsic_no_ddr_rule(KernelId::BINARY_SAMESHAPE, 29),
            intrinsic_no_ddr_rule(KernelId::UNARY_SPM, 28),
            intrinsic_no_ddr_rule(KernelId::UNARY_GELU_TANH_SPM, 24),
            intrinsic_no_ddr_rule(KernelId::UNARY_GELU_ERF_SPM, 2),
            explicit_dynamic_no_ddr_rule(
                "fill", 1, GraphKernelNoDdrProof::FillSpm),
            explicit_dynamic_no_ddr_rule(
                "slice_single_axis", 2,
                GraphKernelNoDdrProof::SliceSpm),
        };

        value.expected_node_count = 10127;
        value.expected_dma_count = 3362;
        value.expected_barrier_count = 5754;
        value.expected_kernel_count = 1011;
        value.expected_typed_kernel_count = 604;
        value.expected_no_ddr_kernel_count = 407;
        value.expected_operand_count = 768;
        value.expected_weight_operand_count = 456;
        value.expected_key_cache_operand_count = 104;
        value.expected_value_cache_operand_count = 104;
        value.expected_semantic_input_operand_count = 104;
        value.expected_ordered_node_kind_hash =
            UINT64_C(0xbd88d2887877515d);
        value.expected_ordered_kernel_hash =
            UINT64_C(0xc0f97cf612a17608);
        value.phases = {
            GraphKernelRegisterPhaseExpectation{
                "vision",
                phase_stats(
                    7031, 2441, 4144, 446, 268, 178,
                    292, 148, 48, 48, 48,
                    UINT64_C(0x94c72c177a2770b7),
                    UINT64_C(0x564531093ab6d335))},
            GraphKernelRegisterPhaseExpectation{
                "handoff",
                phase_stats(
                    47, 16, 28, 3, 0, 3,
                    0, 0, 0, 0, 0,
                    UINT64_C(0xa21f233d7817267b),
                    UINT64_C(0x88028aeb1847a09c))},
            GraphKernelRegisterPhaseExpectation{
                "text",
                phase_stats(
                    3049, 905, 1582, 562, 336, 226,
                    476, 308, 56, 56, 56,
                    UINT64_C(0x4ac36c18252961c9),
                    UINT64_C(0x426d67efee2c461d))},
        };
        return value;
    }();
    return policy;
}

const GraphKernelRegisterPolicy& qwen3vl_pooler_ds3_register_policy() {
    static const GraphKernelRegisterPolicy policy = [] {
        GraphKernelRegisterPolicy value;
        value.fingerprint = UINT64_C(0x5133564c44533333);  // Q3VLDS33
        value.name = "qwen3vl_pooler_ds3_register_census_v3";

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
        const auto rope_position = ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 0, 1);
        const auto mrope_cos = ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 6, 7);
        const auto mrope_sin = ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 8, 9);
        const auto mrope_position = ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 12, 13);

        value.rules = {
            typed_rule(KernelId::PL_AT_FP16_M320N128, 52, {weight}),
            typed_rule(KernelId::PL_AT_FP16_M384N96, 168, {weight}),
            typed_rule(KernelId::PL_AT_FP16_M480N64, 128, {weight}),
            typed_rule(KernelId::ROPE_2D_SPM, 48, {rope_position}),
            typed_rule(KernelId::MROPE, 56,
                       {mrope_cos, mrope_sin, mrope_position}),
            typed_rule(KernelId::LLAMA_INSERT_KCACHE_V16, 52,
                       {key_write}),
            typed_rule(KernelId::LLAMA_INSERT_VCACHE_V16, 52,
                       {value_write}),
            typed_rule(KernelId::SDPA_FLASH_ATTN_SPM, 52,
                       {key_read, value_read}),
            explicit_no_ddr_rule(
                KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED, 108,
                GraphKernelNoDdrProof::RingAllReduceSpmResidual),
            intrinsic_no_ddr_rule(KernelId::MEMSET_SPM_MULTI_CORE_V2, 56),
            intrinsic_no_ddr_rule(KernelId::LAYER_NORM, 52),
            intrinsic_no_ddr_rule(KernelId::RMS_NORM_BF16_SPM, 113),
            intrinsic_no_ddr_rule(KernelId::BINARY_SAMESHAPE, 31),
            intrinsic_no_ddr_rule(KernelId::UNARY_SPM, 28),
            intrinsic_no_ddr_rule(KernelId::UNARY_GELU_TANH_SPM, 24),
            intrinsic_no_ddr_rule(KernelId::UNARY_GELU_ERF_SPM, 4),
            explicit_dynamic_no_ddr_rule(
                "fill", 1, GraphKernelNoDdrProof::FillSpm),
            explicit_dynamic_no_ddr_rule(
                "slice_single_axis", 2,
                GraphKernelNoDdrProof::SliceSpm),
        };

        value.expected_node_count = 10277;
        value.expected_dma_count = 3412;
        value.expected_barrier_count = 5838;
        value.expected_kernel_count = 1027;
        value.expected_typed_kernel_count = 608;
        value.expected_no_ddr_kernel_count = 419;
        value.expected_operand_count = 772;
        value.expected_weight_operand_count = 460;
        value.expected_key_cache_operand_count = 104;
        value.expected_value_cache_operand_count = 104;
        value.expected_semantic_input_operand_count = 104;
        value.expected_ordered_node_kind_hash =
            UINT64_C(0xacaca5524c817c49);
        value.expected_ordered_kernel_hash =
            UINT64_C(0x2432d00094747250);
        value.phases = {
            GraphKernelRegisterPhaseExpectation{
                "vision",
                phase_stats(
                    7179, 2491, 4228, 460, 272, 188,
                    296, 152, 48, 48, 48,
                    UINT64_C(0x9279f1545ffacb9b),
                    UINT64_C(0xcf5fef9e2fa1b385))},
            GraphKernelRegisterPhaseExpectation{
                "handoff",
                phase_stats(
                    47, 16, 28, 3, 0, 3,
                    0, 0, 0, 0, 0,
                    UINT64_C(0xa21f233d7817267b),
                    UINT64_C(0x88028aeb1847a09c))},
            GraphKernelRegisterPhaseExpectation{
                "text",
                phase_stats(
                    3051, 905, 1582, 564, 336, 228,
                    476, 308, 56, 56, 56,
                    UINT64_C(0x01f2f9249027e721),
                    UINT64_C(0xd030a24b4f411df5))},
        };
        return value;
    }();
    return policy;
}

const GraphKernelRegisterPolicy&
qwen3vl_pooler_ds3_partial_mrope_register_policy() {
    // This node stream derives from the DS3 policy, with the 56 Text Q/K RoPE
    // nodes using partial_mrope. The kernel-name hashes apply that deterministic
    // substitution; two model-owned cos/sin keepalives replace MROPE's three
    // DDR operands.
    static const GraphKernelRegisterPolicy policy = [] {
        GraphKernelRegisterPolicy value =
            qwen3vl_pooler_ds3_register_policy();
        value.fingerprint = UINT64_C(0x5133564c44535034);  // Q3VLDSP4
        value.name =
            "qwen3vl_pooler_ds3_partial_mrope_register_census_v4";

        bool replaced = false;
        for (auto& rule : value.rules) {
            if (rule.kernel_id == KernelId::MROPE) {
                rule.kernel_id = KernelId::PARTIAL_MROPE;
                rule.operands = {
                    ddr_abi(
                        GraphDdrRegisterRole::SemanticInput,
                        GraphDdrRegisterAccess::Read, 0, 1),
                    ddr_abi(
                        GraphDdrRegisterRole::SemanticInput,
                        GraphDdrRegisterAccess::Read, 2, 3),
                };
                replaced = true;
            }
        }
        TORCH_INTERNAL_ASSERT(replaced);

        value.expected_operand_count = 716;
        value.expected_weight_operand_count = 348;
        value.expected_semantic_input_operand_count = 160;
        value.expected_ordered_kernel_hash =
            UINT64_C(0x96f763aafa93dc70);
        TORCH_INTERNAL_ASSERT(value.phases.size() == 3 &&
                              value.phases[2].name == "text");
        value.phases[2].stats.operand_count = 420;
        value.phases[2].stats.weight_operand_count = 196;
        value.phases[2].stats.semantic_input_operand_count = 112;
        value.phases[2].stats.ordered_kernel_hash =
            UINT64_C(0x63af0dcc2969bed5);
        return value;
    }();
    return policy;
}

const GraphKernelRegisterPolicy& groot_n17_pooler_ds3_register_policy() {
    static const GraphKernelRegisterPolicy policy = [] {
        GraphKernelRegisterPolicy value;
        value.fingerprint = UINT64_C(0x4752544453335633);  // GRTDS3V3
        value.name = "groot_n17_pooler_ds3_register_census_v3";

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
        const auto rope_position = ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 0, 1);
        const auto mrope_cos = ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 6, 7);
        const auto mrope_sin = ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 8, 9);
        const auto mrope_position = ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 12, 13);

        value.rules = {
            typed_rule(KernelId::PL_AT_FP16_M320N128, 40, {weight}),
            typed_rule(KernelId::PL_AT_FP16_M384N96, 120, {weight}),
            typed_rule(KernelId::PL_AT_FP16_M480N64, 104, {weight}),
            typed_rule(KernelId::ROPE_2D_SPM, 48, {rope_position}),
            typed_rule(KernelId::MROPE, 32,
                       {mrope_cos, mrope_sin, mrope_position}),
            typed_rule(KernelId::LLAMA_INSERT_KCACHE_V16, 40,
                       {key_write}),
            typed_rule(KernelId::LLAMA_INSERT_VCACHE_V16, 40,
                       {value_write}),
            typed_rule(KernelId::SDPA_FLASH_ATTN_SPM, 40,
                       {key_read, value_read}),
            explicit_no_ddr_rule(
                KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED, 84,
                GraphKernelNoDdrProof::RingAllReduceSpmResidual),
            intrinsic_no_ddr_rule(KernelId::MEMSET_SPM_MULTI_CORE_V2, 56),
            intrinsic_no_ddr_rule(KernelId::LAYER_NORM, 52),
            intrinsic_no_ddr_rule(KernelId::RMS_NORM_BF16_SPM, 65),
            intrinsic_no_ddr_rule(KernelId::BINARY_SAMESHAPE, 19),
            intrinsic_no_ddr_rule(KernelId::UNARY_SPM, 16),
            intrinsic_no_ddr_rule(KernelId::UNARY_GELU_TANH_SPM, 24),
            intrinsic_no_ddr_rule(KernelId::UNARY_GELU_ERF_SPM, 4),
            explicit_dynamic_no_ddr_rule(
                "fill", 1, GraphKernelNoDdrProof::FillSpm),
            explicit_dynamic_no_ddr_rule(
                "slice_single_axis", 2,
                GraphKernelNoDdrProof::SliceSpm),
        };

        value.expected_node_count = 8981;
        value.expected_dma_count = 3028;
        value.expected_barrier_count = 5166;
        value.expected_kernel_count = 787;
        value.expected_typed_kernel_count = 464;
        value.expected_no_ddr_kernel_count = 323;
        value.expected_operand_count = 568;
        value.expected_weight_operand_count = 328;
        value.expected_key_cache_operand_count = 80;
        value.expected_value_cache_operand_count = 80;
        value.expected_semantic_input_operand_count = 80;
        value.expected_ordered_node_kind_hash =
            UINT64_C(0xf94c8ed7da98d029);
        value.expected_ordered_kernel_hash =
            UINT64_C(0x50ab7f6c7f63c5f0);
        value.phases = {
            GraphKernelRegisterPhaseExpectation{
                "vision",
                phase_stats(
                    7179, 2491, 4228, 460, 272, 188,
                    296, 152, 48, 48, 48,
                    UINT64_C(0x9279f1545ffacb9b),
                    UINT64_C(0xcf5fef9e2fa1b385))},
            GraphKernelRegisterPhaseExpectation{
                "handoff",
                phase_stats(
                    47, 16, 28, 3, 0, 3,
                    0, 0, 0, 0, 0,
                    UINT64_C(0xa21f233d7817267b),
                    UINT64_C(0x88028aeb1847a09c))},
            GraphKernelRegisterPhaseExpectation{
                "text",
                phase_stats(
                    1755, 521, 910, 324, 192, 132,
                    272, 176, 32, 32, 32,
                    UINT64_C(0x4323d1115effb901),
                    UINT64_C(0x0556eab27b7313c5))},
        };
        return value;
    }();
    return policy;
}

void validate_text_hidden(const at::Tensor& hidden, const char* op) {
    TORCH_CHECK(hidden.defined() &&
                    hidden.device().type() == at::kPrivateUse1 &&
                    hidden.scalar_type() == at::kHalf &&
                    hidden.layout() == c10::Layout::Strided &&
                    hidden.is_contiguous() && hidden.dim() == 3 &&
                    hidden.size(0) == 1 &&
                    hidden.size(1) == kExecutionLen &&
                    hidden.size(2) == kTextHidden,
                op, ": text_hidden must be contiguous FP16 RPU "
                "[1,128,2048]");
}

void stage_non_image_rows_locked(Qwen3VlPoolerZ1State& s,
                                 const at::Tensor& hidden,
                                 uint64_t epoch,
                                 uint64_t plan_hash) {
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "qwen3vl_pooler_spm_z1 stage text input");
    validate_text_hidden(hidden, "qwen3vl_pooler_spm_z1");
    TORCH_CHECK(s.destination_addr != 0,
                "qwen3vl_pooler_spm_z1: destination port is not bound");

    const int64_t image_end = s.image_row_begin + kMergedRows;
    const int64_t suffix_rows = s.real_len - image_end;
    const int64_t pad_rows = s.execution_len - s.real_len;
    const bool qwen3vl_geometry =
        s.real_len == kRealLen &&
        suffix_rows == kRealLen - (kImageRowBegin + kMergedRows) &&
        pad_rows == kExecutionLen - kRealLen;
    const bool groot_geometry =
        s.real_len == kGrootRealLen &&
        suffix_rows == kGrootRealLen - (kImageRowBegin + kMergedRows) &&
        pad_rows == kExecutionLen - kGrootRealLen;
    TORCH_CHECK(s.image_row_begin == kImageRowBegin &&
                    (qwen3vl_geometry || groot_geometry),
                "qwen3vl_pooler_spm_z1: fixed prefix/suffix/pad geometry "
                "drifted");
    const int64_t row_bytes =
        kTextHidden * static_cast<int64_t>(sizeof(c10::Half));

    s.hidden_ref = hidden;
    const uint64_t hidden_dev_addr =
        ::rhino_lkn::RpuGetDevAddr(hidden.data_ptr<c10::Half>());
    TORCH_CHECK(hidden_dev_addr != 0,
                "qwen3vl_pooler_spm_z1: text_hidden resolved a zero RPU "
                "address");
    s.hidden_prefix_src_base = hidden_dev_addr;
    s.hidden_suffix_src_base = hidden_dev_addr;
    rpu_ddr_flush_force_sized(hidden.data_ptr<c10::Half>(), hidden.nbytes());

    rpu_launch_ddr_broadcast_spm_dma_mutable(
        s.hidden_prefix_src_base,
        hidden,
        /*src_offset_bytes=*/0,
        s.image_row_begin * kTextHidden,
        s.destination_addr,
        kNumCores);
    rpu_launch_ddr_broadcast_spm_dma_mutable(
        s.hidden_suffix_src_base,
        hidden,
        image_end * row_bytes,
        suffix_rows * kTextHidden,
        s.destination_addr + static_cast<uint32_t>(image_end * row_bytes),
        kNumCores);
    rpu_launch_fill_spm_kernel(
        s.destination_addr + static_cast<uint32_t>(s.real_len * row_bytes),
        pad_rows * kTextHidden,
        c10::Half(0.0f),
        kNumCores);
}

void bind_non_image_rows_outer_fast_locked(
        Qwen3VlPoolerZ1State& s,
        GraphKernelRegisterCensusGuard& guard,
        const at::Tensor& hidden,
        uint64_t epoch,
        uint64_t plan_hash) {
    validate_active_pipeline_locked(
        s, epoch, plan_hash,
        "qwen3vl_pooler_spm_z1 bind fast text input");
    validate_text_hidden(hidden, "qwen3vl_pooler_spm_z1 outer fast");

    const int64_t image_end = s.image_row_begin + kMergedRows;
    const int64_t suffix_rows = s.real_len - image_end;
    const bool qwen3vl_geometry =
        s.real_len == kRealLen &&
        suffix_rows == kRealLen - (kImageRowBegin + kMergedRows);
    const bool groot_geometry =
        s.real_len == kGrootRealLen &&
        suffix_rows == kGrootRealLen - (kImageRowBegin + kMergedRows);
    TORCH_CHECK(s.image_row_begin == kImageRowBegin &&
                    image_end == kImageRowBegin + kMergedRows &&
                    (qwen3vl_geometry || groot_geometry),
                "qwen3vl_pooler_spm_z1 outer fast text geometry drifted");
    const size_t row_bytes =
        static_cast<size_t>(kTextHidden) * sizeof(c10::Half);
    const size_t prefix_bytes =
        static_cast<size_t>(s.image_row_begin) * row_bytes;
    const size_t suffix_begin =
        static_cast<size_t>(image_end) * row_bytes;
    const size_t suffix_bytes =
        static_cast<size_t>(suffix_rows) * row_bytes;

    const uint64_t hidden_dev_addr =
        ::rhino_lkn::RpuGetDevAddr(hidden.data_ptr<c10::Half>());
    TORCH_CHECK(hidden_dev_addr != 0,
                "qwen3vl_pooler_spm_z1 outer fast text_hidden resolved a "
                "zero RPU address");
    guard.bind_fast_mutable_dma(
        s.hidden_prefix_src_base, hidden, GraphOuterFastDmaSide::Source,
        /*byte_begin=*/0, prefix_bytes);
    guard.bind_fast_mutable_dma(
        s.hidden_suffix_src_base, hidden, GraphOuterFastDmaSide::Source,
        suffix_begin, suffix_bytes);
}

void emit_identity_route_locked(Qwen3VlPoolerZ1State& s,
                                uint64_t epoch,
                                uint64_t plan_hash) {
    validate_active_pipeline_locked(
        s, epoch, plan_hash, "qwen3vl_pooler_spm_z1 identity route");
    TORCH_CHECK(s.source_addr != 0 && s.destination_addr != 0 &&
                    s.lowered_runs.size() == 1,
                "qwen3vl_pooler_spm_z1: identity route is not bound");
    const auto& run = s.lowered_runs.front();
    TORCH_CHECK(run.source_row == 0 &&
                    run.destination_row == kImageRowBegin &&
                    run.row_count == kMergedRows,
                "qwen3vl_pooler_spm_z1: identity route must lower to "
                "{source=0,destination=4,count=64}");

    const std::vector<int64_t> input_shape = {kMergedRows, kTextHidden};
    const std::vector<int64_t> half_shape = {kRouteHalfRows, kTextHidden};
    const int64_t row_bytes =
        kTextHidden * static_cast<int64_t>(sizeof(c10::Half));
    rpu_launch_slice_spm_kernel(
        s.source_addr,
        s.destination_addr + static_cast<uint32_t>(
            run.destination_row * row_bytes),
        input_shape,
        half_shape,
        {run.source_row, 0},
        kNumCores);
    rpu_launch_slice_spm_kernel(
        s.source_addr,
        s.destination_addr + static_cast<uint32_t>(
            (run.destination_row + kRouteHalfRows) * row_bytes),
        input_shape,
        half_shape,
        {run.source_row + kRouteHalfRows, 0},
        kNumCores);
}

}  // namespace

std::vector<int64_t> rpu_qwen3vl_multiview_spm_dry_probe(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len) {
    TORCH_CHECK(
        !RpuKernelGraph::has_active(),
        "qwen3vl_multiview_spm_dry_probe must run outside Graph capture");
    RpuExecutionCleanupGuard execution_guard(
        "qwen3vl_multiview_spm_dry_probe");
    TORCH_CHECK(
        vision_handle > 0 && text_handle > 0,
        "qwen3vl_multiview_spm_dry_probe: handles must be positive");
    TORCH_CHECK(
        num_patches == 256 && execution_len == 225,
        "qwen3vl_multiview_spm_dry_probe accepts only LingBot2 N256/S225");

    Qwen3VlPoolerZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(
        !s.plan && !s.lease && s.active_invocation_generation == 0 &&
            s.invocation_prepared_generation != nullptr &&
            s.invocation_consumed_generation != nullptr &&
            *s.invocation_prepared_generation ==
                *s.invocation_consumed_generation,
        "qwen3vl_multiview_spm_dry_probe requires an idle coordinator");

    const AllocatorSnapshot before = allocator_snapshot();
    std::vector<int64_t> result;
    bool vision_prepared = false;
    bool text_prepared = false;
    std::exception_ptr failure;
    try {
        const auto vision =
            v3::qwen3vl_pooler_z1_internal::prepare_multiview_vision_dry(
                vision_handle, num_patches);
        vision_prepared = true;
        check_allocator_unchanged(before, "after Vision dry prepare");

        const auto text =
            v3::qwen3vl_pooler_z1_internal::prepare_multiview_text_dry(
                text_handle, execution_len);
        text_prepared = true;
        check_allocator_unchanged(before, "after Text dry prepare");

        const auto& shape = text.shape;
        TORCH_CHECK(
            shape.chunks.size() == 1 &&
                shape.chunks.front().idx == 0 &&
                shape.chunks.front().offset == 0 &&
                shape.chunks.front().len == execution_len &&
                shape.chunks.front().kv_seq_len == execution_len &&
                shape.kv_insert_chunks.size() == 1 &&
                shape.kv_insert_chunks.front().idx == 0 &&
                shape.kv_insert_chunks.front().offset == 0 &&
                shape.kv_insert_chunks.front().len == execution_len &&
                shape.kv_insert_chunks.front().kv_seq_len == execution_len &&
                shape.chunk_mode == v3::ChunkMode::SEQUENTIAL &&
                shape.inter_layer_io == v3::InterLayerIO::SPM_RESIDENT &&
                !shape.chunk_outer_within_group,
            "qwen3vl_multiview_spm_dry_probe: exact Text chunk geometry "
            "drifted");

        constexpr int64_t kProbeSchema = 1;
        constexpr size_t kEffectiveBudget =
            SpmAllocator::SPM_PLANNING_BUDGET &
            ~(SpmAllocator::ALIGN - 1);
        result = {
            kProbeSchema,
            num_patches,
            execution_len,
            static_cast<int64_t>(vision.temporary_bytes),
            encode_u64(vision.layout_hash),
            shape.resolved_chunk_size,
            shape.allocation_layout.chunk_size,
            static_cast<int64_t>(shape.chunks.size()),
            shape.chunks.front().offset,
            shape.chunks.front().len,
            shape.chunks.front().kv_seq_len,
            static_cast<int64_t>(shape.kv_insert_chunks.size()),
            static_cast<int64_t>(shape.chunk_mode),
            static_cast<int64_t>(shape.inter_layer_io),
            shape.chunk_outer_within_group ? 1 : 0,
            static_cast<int64_t>(text.component.temporary_bytes),
            encode_u64(text.component.layout_hash),
            static_cast<int64_t>(live_persistent_top_bytes()),
            static_cast<int64_t>(SpmAllocator::SPM_PLANNING_BUDGET),
            static_cast<int64_t>(kEffectiveBudget),
            before.initialized ? 1 : 0,
            encode_u64(before.generation),
            encode_u64(before.persistent_generation),
            static_cast<int64_t>(before.temporary_bytes),
            static_cast<int64_t>(before.persistent_bytes),
            static_cast<int64_t>(before.super_persistent_bytes),
            static_cast<int64_t>(before.active_instance_count),
        };
    } catch (...) {
        failure = std::current_exception();
    }

    if (text_prepared) {
        cleanup_step(failure, [&] {
            v3::qwen3vl_pooler_z1_internal::cancel_multiview_text_dry(
                text_handle);
        });
    }
    if (vision_prepared) {
        cleanup_step(failure, [&] {
            v3::qwen3vl_pooler_z1_internal::cancel_multiview_vision_dry(
                vision_handle);
        });
    }
    check_allocator_unchanged(before, "after reverse dry cancel");
    if (failure) std::rethrow_exception(failure);
    return result;
}

int64_t rpu_qwen3vl_pooler_spm_z1_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len,
    int64_t retained_deepstack_count) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl_pooler_spm_z1_prepare must run outside Graph capture");
    TORCH_CHECK(vision_handle > 0 && text_handle > 0,
                "qwen3vl_pooler_spm_z1_prepare: handles must be positive");
    check_retained_deepstack_count(
        retained_deepstack_count, "qwen3vl_pooler_spm_z1_prepare");
    check_canary_shape(num_patches, execution_len, image_row_begin,
                       real_len, retained_deepstack_count,
                       "qwen3vl_pooler_spm_z1_prepare");

    Qwen3VlPoolerZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(!s.lease,
                "qwen3vl_pooler_spm_z1_prepare: physical lease is active");
    TORCH_INTERNAL_ASSERT(s.invocation_prepared_generation != nullptr &&
                          s.invocation_consumed_generation != nullptr);
    if (s.plan.has_value()) {
        check_identity(s, vision_handle, text_handle, num_patches,
                       execution_len, image_row_begin, real_len,
                       s.plan_hash, "qwen3vl_pooler_spm_z1_prepare");
        TORCH_CHECK(
            s.retained_deepstack_count == retained_deepstack_count,
            "qwen3vl_pooler_spm_z1_prepare: prepared retained DeepStack "
            "cardinality drifted");
        TORCH_CHECK(live_persistent_top_bytes() == s.persistent_top_bytes,
                    "qwen3vl_pooler_spm_z1_prepare: persistent top changed");
        return encode_u64(s.plan_hash);
    }

    bool vision_prepare_attempted = false;
    bool text_prepare_attempted = false;
    try {
        vision_prepare_attempted = true;
        const auto vision_layout =
            v3::qwen3vl_pooler_z1_internal::prepare_vision(
                vision_handle, num_patches, retained_deepstack_count);
        text_prepare_attempted = true;
        const auto text_layout = retained_deepstack_count == 3
            ? v3::qwen3vl_pooler_z1_internal::prepare_text_deepstack3(
                  text_handle, execution_len, real_len)
            : (retained_deepstack_count == 1
                   ? v3::qwen3vl_pooler_z1_internal::prepare_text_deepstack1(
                         text_handle, execution_len, real_len)
                   : v3::qwen3vl_pooler_z1_internal::prepare_text(
                         text_handle, execution_len, real_len));
        TORCH_CHECK(vision_layout.temporary_bytes ==
                        kComponentTemporaryBytes &&
                        text_layout.temporary_bytes ==
                        kComponentTemporaryBytes,
                    "qwen3vl_pooler_spm_z1_prepare: exact component temporary "
                    "bytes drifted; expected ", kComponentTemporaryBytes,
                    ", got vision/text=", vision_layout.temporary_bytes, "/",
                    text_layout.temporary_bytes);

        const size_t persistent_top_bytes = live_persistent_top_bytes();
        const std::vector<v3::SpmScratchDecl> scratch = {
            {kVisionScratch,
             static_cast<int64_t>(vision_layout.temporary_bytes),
             /*first_event=*/0, /*last_event=*/0},
            {kTextScratch,
             static_cast<int64_t>(text_layout.temporary_bytes),
             /*first_event=*/2, /*last_event=*/2},
        };
        const auto route = endpoint_route(
            vision_handle, text_handle, num_patches,
            execution_len, image_row_begin);
        const auto retained = deepstack_routes(
            vision_handle, text_handle, num_patches,
            retained_deepstack_count);
        auto plan = [&] {
            if (retained.empty()) {
                return v3::SpmPipelinePlan::compile(
                    scratch,
                    std::vector<v3::SpmPermutationRowRunsRoute>{route},
                    persistent_top_bytes,
                    SpmAllocator::SPM_PLANNING_BUDGET);
            }
            return v3::SpmPipelinePlan::compile(
                scratch, retained,
                std::vector<v3::SpmPermutationRowRunsRoute>{route},
                persistent_top_bytes, SpmAllocator::SPM_PLANNING_BUDGET);
        }();
        const size_t expected_peak = retained_deepstack_count == 3
            ? kExpectedDeepstack3DynamicPeakBytes
            : (retained_deepstack_count == 1
                   ? kExpectedDeepstack1DynamicPeakBytes
                   : kExpectedDynamicPeakBytes);
        TORCH_CHECK(plan.peak_bytes() == expected_peak,
                    "qwen3vl_pooler_spm_z1_prepare: exact dry-plan peak "
                    "drifted; expected ", expected_peak,
                    ", got ", plan.peak_bytes());

        s.vision_handle = vision_handle;
        s.text_handle = text_handle;
        s.num_patches = num_patches;
        s.execution_len = execution_len;
        s.image_row_begin = image_row_begin;
        s.real_len = real_len;
        s.vision_temporary_bytes = vision_layout.temporary_bytes;
        s.text_temporary_bytes = text_layout.temporary_bytes;
        s.persistent_top_bytes = persistent_top_bytes;
        s.route_hash = route.hash();
        s.plan_hash = plan.hash();
        s.retained_deepstack_count = retained_deepstack_count;
        s.plan = std::move(plan);
        validate_prepared_locked(s, "qwen3vl_pooler_spm_z1_prepare");
        return encode_u64(s.plan_hash);
    } catch (...) {
        std::exception_ptr failure = std::current_exception();
        if (text_prepare_attempted) cleanup_step(failure, [&] {
            v3::qwen3vl_pooler_z1_internal::unprepare_text(text_handle);
        });
        if (vision_prepare_attempted) cleanup_step(failure, [&] {
            v3::qwen3vl_pooler_z1_internal::unprepare_vision(vision_handle);
        });
        reset_prepared_state_locked(s);
        std::rethrow_exception(failure);
    }
}

void rpu_qwen3vl_pooler_spm_z1_unprepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl_pooler_spm_z1_unprepare must run outside Graph capture");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlPoolerZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(!s.lease,
                "qwen3vl_pooler_spm_z1_unprepare: physical lease is active");
    TORCH_CHECK(s.active_invocation_generation == 0 &&
                    s.invocation_prepared_generation != nullptr &&
                    s.invocation_consumed_generation != nullptr &&
                    *s.invocation_prepared_generation ==
                        *s.invocation_consumed_generation,
                "qwen3vl_pooler_spm_z1_unprepare: an invocation was not "
                "successfully consumed");
    check_identity(s, vision_handle, text_handle,
                   s.num_patches, s.execution_len, s.image_row_begin,
                   s.real_len,
                   plan_hash, "qwen3vl_pooler_spm_z1_unprepare");

    std::exception_ptr failure;
    cleanup_step(failure, [&] {
        v3::qwen3vl_pooler_z1_internal::unprepare_text(text_handle);
    });
    cleanup_step(failure, [&] {
        v3::qwen3vl_pooler_z1_internal::unprepare_vision(vision_handle);
    });
    if (failure) std::rethrow_exception(failure);
    reset_prepared_state_locked(s);
}

std::vector<int64_t> rpu_qwen3vl_pooler_spm_z1_plan_stats(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl_pooler_spm_z1_plan_stats must run outside Graph capture");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlPoolerZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    check_identity(s, vision_handle, text_handle,
                   s.num_patches, s.execution_len, s.image_row_begin,
                   s.real_len,
                   plan_hash, "qwen3vl_pooler_spm_z1_plan_stats");
    return {
        static_cast<int64_t>(s.vision_temporary_bytes),
        static_cast<int64_t>(s.text_temporary_bytes),
        static_cast<int64_t>(s.plan->port_spec(
            kPoolerSourcePort).storage_bytes()),
        static_cast<int64_t>(s.plan->port_spec(
            kTextDestinationPort).storage_bytes()),
        static_cast<int64_t>(s.persistent_top_bytes),
        static_cast<int64_t>(s.plan->peak_bytes()),
        static_cast<int64_t>(s.plan->total_bytes()),
        static_cast<int64_t>(s.plan->effective_budget_bytes()),
        encode_u64(s.route_hash),
        encode_u64(s.plan_hash),
    };
}

int64_t rpu_qwen3vl_pooler_spm_z1_begin(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len,
    const at::Tensor& position_ids,
    int64_t encoded_plan_hash,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl_pooler_spm_z1_begin must run outside Graph capture");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlPoolerZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    check_identity(s, vision_handle, text_handle, num_patches,
                   execution_len, image_row_begin, real_len,
                   plan_hash, "qwen3vl_pooler_spm_z1_begin");
    TORCH_CHECK(!s.lease,
                "qwen3vl_pooler_spm_z1_begin: a physical lease is already "
                "active");
    TORCH_CHECK(live_persistent_top_bytes() == s.persistent_top_bytes,
                "qwen3vl_pooler_spm_z1_begin: persistent top changed after "
                "prepare");
    TORCH_CHECK(s.active_invocation_generation == 0 &&
                    s.invocation_prepared_generation != nullptr &&
                    s.invocation_consumed_generation != nullptr &&
                    *s.invocation_prepared_generation ==
                        *s.invocation_consumed_generation,
                "qwen3vl_pooler_spm_z1_begin: prior invocation was not "
                "successfully consumed");
    TORCH_CHECK(*s.invocation_prepared_generation !=
                    std::numeric_limits<uint64_t>::max(),
                "qwen3vl_pooler_spm_z1_begin: invocation generation "
                "exhausted");
    const uint64_t next_invocation_generation =
        *s.invocation_prepared_generation + 1;

    reset_begin_state_locked(s);
    bool vision_prime_attempted = false;
    bool text_prime_attempted = false;
    bool vision_adopted = false;
    bool text_adopted = false;
    std::unique_ptr<v3::SpmPipelineLease> lease;
    uint32_t source_addr = 0;
    uint32_t destination_addr = 0;
    std::array<uint32_t, 3> deepstack_addrs{};
    std::vector<v3::SpmContiguousRowRun> lowered_runs;
    try {
        vision_prime_attempted = true;
        v3::qwen3vl_pooler_z1_internal::prime_vision_inputs(
            vision_handle, num_patches);
        text_prime_attempted = true;
        v3::qwen3vl_pooler_z1_internal::prime_text_inputs(
            text_handle, position_ids, execution_len,
            rope_cos_il, rope_sin_il);

        lease = v3::SpmPipelineLease::acquire_physical(*s.plan);
        const auto vision_scratch = lease->scratch(kVisionScratch);
        const auto text_scratch = lease->scratch(kTextScratch);
        const auto route = lease->permutation_row_runs(
            kPoolerSourcePort, kTextDestinationPort);
        TORCH_CHECK(route.route_hash() == s.route_hash,
                    "qwen3vl_pooler_spm_z1_begin: checked route hash drifted");

        const size_t vision_scratch_offset =
            vision_scratch.resolve_physical_offset(*lease);
        const size_t text_scratch_offset =
            text_scratch.resolve_physical_offset(*lease);
        const size_t source_offset =
            route.source().resolve_physical_offset(*lease);
        const size_t destination_offset =
            route.destination().resolve_physical_offset(*lease);
        TORCH_CHECK(!ranges_overlap(
                        source_offset, route.source().size_bytes(),
                        destination_offset, route.destination().size_bytes()),
                    "qwen3vl_pooler_spm_z1_begin: transform source/destination "
                    "overlap");
        TORCH_CHECK(!ranges_overlap(
                        vision_scratch_offset, vision_scratch.size_bytes(),
                        source_offset, route.source().size_bytes()),
                    "qwen3vl_pooler_spm_z1_begin: Vision scratch/source overlap "
                    "at producer write");
        TORCH_CHECK(!ranges_overlap(
                        text_scratch_offset, text_scratch.size_bytes(),
                        destination_offset, route.destination().size_bytes()),
                    "qwen3vl_pooler_spm_z1_begin: Text scratch/destination "
                    "overlap at consume");
        std::array<size_t, 3> retained_offsets{};
        std::array<size_t, 3> retained_sizes{};
        for (int64_t ordinal = 0;
             ordinal < s.retained_deepstack_count; ++ordinal) {
            const size_t index = static_cast<size_t>(ordinal);
            const auto retained = lease->port(kDeepstackPorts[index]);
            const size_t retained_offset =
                retained.resolve_physical_offset(*lease);
            retained_offsets[index] = retained_offset;
            retained_sizes[index] = retained.size_bytes();
            TORCH_CHECK(!ranges_overlap(
                            retained_offset, retained.size_bytes(),
                            vision_scratch_offset,
                            vision_scratch.size_bytes()),
                        "qwen3vl_pooler_spm_z1_begin: retained DeepStack/Vision "
                        "scratch overlap at ordinal ", ordinal);
            TORCH_CHECK(!ranges_overlap(
                            retained_offset, retained.size_bytes(),
                            source_offset, route.source().size_bytes()),
                        "qwen3vl_pooler_spm_z1_begin: retained DeepStack/pooler "
                        "source overlap at ordinal ", ordinal);
            TORCH_CHECK(!ranges_overlap(
                            retained_offset, retained.size_bytes(),
                            destination_offset,
                            route.destination().size_bytes()),
                        "qwen3vl_pooler_spm_z1_begin: retained DeepStack/text "
                        "destination overlap at ordinal ", ordinal);
            TORCH_CHECK(!ranges_overlap(
                            retained_offset, retained.size_bytes(),
                            text_scratch_offset, text_scratch.size_bytes()),
                        "qwen3vl_pooler_spm_z1_begin: retained DeepStack/Text "
                        "scratch overlap at ordinal ", ordinal);
            for (size_t prior = 0; prior < index; ++prior) {
                TORCH_CHECK(
                    !ranges_overlap(
                        retained_offset, retained.size_bytes(),
                        retained_offsets[prior], retained_sizes[prior]),
                    "qwen3vl_pooler_spm_z1_begin: retained DeepStack ports "
                    "overlap at ordinals ", prior, " and ", ordinal);
            }
        }

        v3::qwen3vl_pooler_z1_internal::adopt_vision(
            vision_handle, *lease, vision_scratch);
        vision_adopted = true;
        v3::qwen3vl_pooler_z1_internal::adopt_text(
            text_handle, *lease, text_scratch);
        text_adopted = true;
        v3::qwen3vl_pooler_z1_internal::bind_vision_source(
            vision_handle, *lease, route.source());
        v3::qwen3vl_pooler_z1_internal::bind_text_destination(
            text_handle, *lease, route.destination());
        for (int64_t ordinal = 0;
             ordinal < s.retained_deepstack_count; ++ordinal) {
            const auto retained = lease->port(
                kDeepstackPorts[static_cast<size_t>(ordinal)]);
            v3::qwen3vl_pooler_z1_internal::bind_vision_deepstack_source(
                vision_handle, *lease, retained, ordinal);
            v3::qwen3vl_pooler_z1_internal::bind_text_deepstack(
                text_handle, *lease, retained, ordinal);
        }
        lease->seal_physical();
        v3::qwen3vl_pooler_z1_internal::validate_vision(
            vision_handle, *lease);
        v3::qwen3vl_pooler_z1_internal::validate_text(
            text_handle, *lease);

        for (int64_t ordinal = 0;
             ordinal < s.retained_deepstack_count; ++ordinal) {
            const size_t index = static_cast<size_t>(ordinal);
            const auto retained = lease->port(kDeepstackPorts[index]);
            deepstack_addrs[index] = retained.resolve_physical_addr(
                /*core=*/0, *lease);
            TORCH_CHECK(
                deepstack_addrs[index] != 0,
                "qwen3vl_pooler_spm_z1_begin: retained DeepStack port "
                "resolved a zero address at ordinal ", ordinal);
        }

        source_addr =
            route.source().resolve_physical_addr(/*core=*/0, *lease);
        destination_addr =
            route.destination().resolve_physical_addr(/*core=*/0, *lease);
        lowered_runs = route.lower_contiguous_runs();
        TORCH_CHECK(lowered_runs.size() == 1 &&
                        lowered_runs.front().source_row == 0 &&
                        lowered_runs.front().destination_row == kImageRowBegin &&
                        lowered_runs.front().row_count == kMergedRows,
                    "qwen3vl_pooler_spm_z1_begin: identity route must lower "
                    "to one {source=0,destination=4,count=64} run");
    } catch (...) {
        std::exception_ptr failure = std::current_exception();
        if (text_adopted) cleanup_step(failure, [&] {
            v3::qwen3vl_pooler_z1_internal::clear_text(
                text_handle, lease->epoch(), lease->plan_hash());
        }); else if (text_prime_attempted) cleanup_step(failure, [&] {
            v3::qwen3vl_pooler_z1_internal::rollback_text_inputs(text_handle);
        });
        if (vision_adopted) cleanup_step(failure, [&] {
            v3::qwen3vl_pooler_z1_internal::clear_vision(
                vision_handle, lease->epoch(), lease->plan_hash());
        }); else if (vision_prime_attempted) cleanup_step(failure, [&] {
            v3::qwen3vl_pooler_z1_internal::rollback_vision_inputs(
                vision_handle);
        });
        if (lease) {
            if (!lease->physical_sealed()) cleanup_step(failure, [&] {
                lease->seal_physical();
            });
            cleanup_step(failure, [&] { lease->release(); });
        }
        reset_begin_state_locked(s);
        std::rethrow_exception(failure);
    }

    s.source_addr = source_addr;
    s.destination_addr = destination_addr;
    s.deepstack_addrs = deepstack_addrs;
    s.lowered_runs.swap(lowered_runs);
    s.partial_mrope = rope_cos_il.has_value();
    const uint64_t epoch = lease->epoch();
    s.lease = std::move(lease);
    *s.invocation_prepared_generation = next_invocation_generation;
    s.active_invocation_generation = next_invocation_generation;
    return encode_u64(epoch);
}

void rpu_qwen3vl_pooler_spm_z1_end(int64_t encoded_epoch,
                                   int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl_pooler_spm_z1_end must run outside Graph capture");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlPoolerZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.lease,
                "qwen3vl_pooler_spm_z1_end: no physical lease is active");
    s.lease->validate_physical_identity(epoch, plan_hash);

    std::exception_ptr failure;
    cleanup_step(failure, [&] {
        TORCH_CHECK(s.active_invocation_generation != 0 &&
                        s.invocation_prepared_generation != nullptr &&
                        s.invocation_consumed_generation != nullptr &&
                        *s.invocation_prepared_generation ==
                            s.active_invocation_generation &&
                        *s.invocation_consumed_generation ==
                            s.active_invocation_generation,
                    "qwen3vl_pooler_spm_z1_end: active invocation was not "
                    "successfully consumed");
    });
    cleanup_step(failure, [&] {
        v3::qwen3vl_pooler_z1_internal::validate_vision(
            s.vision_handle, *s.lease);
        v3::qwen3vl_pooler_z1_internal::validate_text(
            s.text_handle, *s.lease);
    });
    cleanup_step(failure, [&] {
        v3::qwen3vl_pooler_z1_internal::clear_text(
            s.text_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] {
        v3::qwen3vl_pooler_z1_internal::clear_vision(
            s.vision_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] { s.lease->release(); });
    s.lease.reset();
    reset_begin_state_locked(s);
    if (failure) std::rethrow_exception(failure);
}

void rpu_qwen3vl_pooler_spm_z1_abort(int64_t encoded_epoch,
                                     int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl_pooler_spm_z1_abort must run outside Graph capture");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlPoolerZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.lease,
                "qwen3vl_pooler_spm_z1_abort: no physical lease is active");
    s.lease->validate_physical_identity(epoch, plan_hash);
    TORCH_CHECK(
        s.active_invocation_generation != 0 &&
            s.invocation_prepared_generation != nullptr &&
            s.invocation_consumed_generation != nullptr &&
            *s.invocation_prepared_generation ==
                s.active_invocation_generation &&
            *s.invocation_consumed_generation <
                s.active_invocation_generation,
        "qwen3vl_pooler_spm_z1_abort requires one begun but unconsumed "
        "invocation; successfully consumed invocations must use end");

    std::exception_ptr failure;
    cleanup_step(failure, [&] {
        v3::qwen3vl_pooler_z1_internal::clear_text(
            s.text_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] {
        v3::qwen3vl_pooler_z1_internal::clear_vision(
            s.vision_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] { s.lease->release(); });
    s.lease.reset();
    // Retire the aborted generation without pretending that the Graph body
    // completed.  begin()/unprepare() require prepared == consumed while
    // inactive; preserving monotonicity also prevents a stale failed-capture
    // ticket from authorizing a later invocation.
    *s.invocation_consumed_generation = s.active_invocation_generation;
    reset_begin_state_locked(s);
    if (failure) std::rethrow_exception(failure);
}

void rpu_qwen3vl_pooler_spm_z1_cancel(int64_t encoded_epoch,
                                      int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl_pooler_spm_z1_cancel must run outside Graph capture");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlPoolerZ1State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.lease,
                "qwen3vl_pooler_spm_z1_cancel: no physical lease is active");
    s.lease->validate_physical_identity(epoch, plan_hash);
    TORCH_CHECK(s.active_invocation_generation != 0 &&
                    s.invocation_prepared_generation != nullptr &&
                    s.invocation_consumed_generation != nullptr,
                "qwen3vl_pooler_spm_z1_cancel: invocation authority is "
                "missing");
    const uint64_t active_generation = s.active_invocation_generation;
    const uint64_t prepared_generation =
        *s.invocation_prepared_generation;
    const uint64_t consumed_generation =
        *s.invocation_consumed_generation;
    TORCH_CHECK(
        prepared_generation == active_generation &&
            consumed_generation <= active_generation,
        "qwen3vl_pooler_spm_z1_cancel: stale invocation authority; active/"
        "prepared/consumed=", active_generation, "/", prepared_generation,
        "/", consumed_generation);
    const bool retire_unconsumed_generation =
        consumed_generation < active_generation;

    std::exception_ptr failure;
    cleanup_step(failure, [&] {
        v3::qwen3vl_pooler_z1_internal::clear_text(
            s.text_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] {
        v3::qwen3vl_pooler_z1_internal::clear_vision(
            s.vision_handle, epoch, plan_hash);
    });
    cleanup_step(failure, [&] { s.lease->release(); });
    s.lease.reset();
    if (retire_unconsumed_generation) {
        *s.invocation_consumed_generation = active_generation;
    }
    reset_begin_state_locked(s);
    if (failure) std::rethrow_exception(failure);
}

at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& vision_input,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    int64_t encoded_epoch,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(RpuKernelGraph::has_active(),
                "qwen3vl_pooler_spm_z1_pipeline_forward requires one active "
                "outer Graph");
    const uint64_t epoch = decode_u64(encoded_epoch);
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    uint64_t physical_digest = 0;
    uint64_t invocation_generation = 0;
    int64_t retained_deepstack_count = 0;
    bool groot_profile = false;
    bool partial_mrope = false;
    std::shared_ptr<const uint64_t> invocation_prepared_generation;
    std::shared_ptr<uint64_t> invocation_consumed_generation;
    {
        Qwen3VlPoolerZ1State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        TORCH_CHECK(s.vision_handle == vision_handle &&
                        s.text_handle == text_handle,
                    "qwen3vl_pooler_spm_z1_pipeline_forward: active handle "
                    "mismatch");
        validate_active_pipeline_locked(
            s, epoch, plan_hash,
            "qwen3vl_pooler_spm_z1_pipeline_forward preflight");
        physical_digest = physical_binding_digest(s);
        retained_deepstack_count = s.retained_deepstack_count;
        groot_profile = s.real_len == kGrootRealLen;
        partial_mrope = s.partial_mrope;
        invocation_generation = s.active_invocation_generation;
        invocation_prepared_generation =
            s.invocation_prepared_generation;
        invocation_consumed_generation =
            s.invocation_consumed_generation;
        TORCH_CHECK(invocation_generation != 0 &&
                        invocation_prepared_generation != nullptr &&
                        invocation_consumed_generation != nullptr &&
                        *invocation_prepared_generation ==
                            invocation_generation &&
                        *invocation_consumed_generation <
                            invocation_generation,
                    "qwen3vl_pooler_spm_z1_pipeline_forward: invocation "
                    "authority is missing or stale");
    }

    const GraphKernelRegisterPolicy& register_policy =
        groot_profile
        ? groot_n17_pooler_ds3_register_policy()
        : (retained_deepstack_count == 3
               ? (partial_mrope
                      ? qwen3vl_pooler_ds3_partial_mrope_register_policy()
                      : qwen3vl_pooler_ds3_register_policy())
               : (retained_deepstack_count == 1
                      ? qwen3vl_pooler_ds1_register_policy()
                      : qwen3vl_pooler_z1_register_policy()));
    GraphKernelRegisterCensusGuard register_census(
        RpuKernelGraph::active(), register_policy, plan_hash, epoch);
    register_census.stage_outer_fast_physical_digest(physical_digest);
    v3::qwen3vl_pooler_z1_internal::stage_vision_outer_fast_component(
        vision_handle, register_census, vision_k_caches, vision_v_caches);
    v3::qwen3vl_pooler_z1_internal::stage_text_outer_fast_component(
        text_handle, register_census, text_k_caches, text_v_caches);
    register_census.stage_outer_fast_invocation_authority(
        std::move(invocation_prepared_generation),
        invocation_consumed_generation, invocation_generation);

    if (RpuKernelGraph::active().state() ==
        RpuKernelGraph::State::REPLAYING) {
        v3::qwen3vl_pooler_z1_internal::bind_vision_outer_fast_input(
            vision_handle, register_census, vision_input);
        {
            Qwen3VlPoolerZ1State& s = state();
            std::lock_guard<std::mutex> lock(s.mutex);
            bind_non_image_rows_outer_fast_locked(
                s, register_census, text_hidden, epoch, plan_hash);
        }
        at::Tensor result =
            v3::qwen3vl_pooler_z1_internal::text_outer_fast_output(
                text_handle);
        auto ticket = register_census.validate_outer_fast_replay();
        {
            Qwen3VlPoolerZ1State& s = state();
            std::lock_guard<std::mutex> lock(s.mutex);
            validate_active_pipeline_locked(
                s, epoch, plan_hash,
                "qwen3vl_pooler_spm_z1_pipeline_forward fast postflight");
            TORCH_CHECK(physical_binding_digest(s) == physical_digest &&
                            s.active_invocation_generation ==
                                invocation_generation &&
                            *s.invocation_prepared_generation ==
                                invocation_generation &&
                            *s.invocation_consumed_generation <
                                invocation_generation,
                        "qwen3vl_pooler_spm_z1_pipeline_forward fast "
                        "authority drifted");
            register_census.commit_outer_fast_replay(std::move(ticket));
        }
        return result;
    }

    v3::qwen3vl_pooler_z1_internal::forward_vision_z1(
        vision_handle, vision_input, vision_k_caches, vision_v_caches,
        kNumPatches, epoch, plan_hash);
    register_census.checkpoint("vision");

    {
        Qwen3VlPoolerZ1State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        stage_non_image_rows_locked(s, text_hidden, epoch, plan_hash);
        emit_identity_route_locked(s, epoch, plan_hash);
    }
    register_census.checkpoint("handoff");

    at::Tensor result =
        v3::qwen3vl_pooler_z1_internal::forward_text_z1(
            text_handle, text_hidden, text_k_caches, text_v_caches,
            position_ids, epoch, plan_hash);
    {
        Qwen3VlPoolerZ1State& s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        validate_active_pipeline_locked(
            s, epoch, plan_hash,
            "qwen3vl_pooler_spm_z1_pipeline_forward postflight");
    }
    register_census.complete();
    return result;
}
