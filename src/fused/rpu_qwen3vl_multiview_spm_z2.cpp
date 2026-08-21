#include "rpu_qwen3vl_pooler_spm_z1.h"
#include "rpu_lingbot_v2_moe_model.h"

#include "core/rpu_kernel_decls.h"
#include "core/rpu_spm_allocator.h"
#include "graph/graph_runtime.h"

#include <c10/util/Exception.h>
#include <c10/util/ScopeExit.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kImageCount = 3;
constexpr int64_t kVisionLayers = 24;
constexpr int64_t kTextLayers = 36;
constexpr int64_t kNumPatches = 256;
constexpr int64_t kVisionHidden = 1024;
constexpr int64_t kMergedRows = 64;
constexpr int64_t kExecutionLen = 225;
constexpr int64_t kTextHidden = 2560;
constexpr int64_t kMropeHalf = 64;
constexpr size_t kVisionTemporaryBytes = 1'966'080;
constexpr size_t kTextTemporaryBytes = 4'163'072;
constexpr size_t kConsumerRootBytes = 1'152'000;
constexpr size_t kPersistentTopBytes = 774'144;
constexpr size_t kExpectedDynamicBytes = 7'112'192;
constexpr size_t kExpectedTotalBytes = 7'886'336;
constexpr size_t kExpectedBudgetBytes = 8'303'616;
constexpr size_t kExpectedHeadroomBytes = 417'280;
constexpr size_t kLingbot2ExpertPersistentBytes = 318'976;
constexpr size_t kLingbot2PersistentTopBytes = 1'093'120;
constexpr size_t kLingbot2ExpectedTotalBytes = 8'205'312;
constexpr size_t kLingbot2ExpectedHeadroomBytes = 98'304;
constexpr size_t kOccurrenceCount = 4;
constexpr size_t kYieldsPerProducer = 4;
constexpr size_t kProducerYieldCount = 12;
constexpr size_t kDirectBindingCount = 3;
constexpr size_t kRunAddBindingCount = 9;
constexpr uint64_t kSignatureSchema = UINT64_C(0x51564d565a324332);
constexpr std::array<int64_t, kImageCount> kImageRowBegins = {1, 67, 133};
constexpr v3::SpmScratchId kVisionArena{1};
constexpr v3::SpmScratchId kTextArena{2};

static_assert(
    (SpmAllocator::SPM_PLANNING_BUDGET &
     ~(SpmAllocator::ALIGN - 1)) == kExpectedBudgetBytes);

enum class CanaryPhase : uint8_t {
    Idle = 0,
    Reserved = 1,
    Bootstrapping = 2,
    Retained = 3,
    Poisoned = 4,
};

struct Qwen3VlMultiviewZ2State {
    std::mutex mutex;
    CanaryPhase phase = CanaryPhase::Idle;
    std::thread::id owner_thread;
    int64_t vision_handle = 0;
    int64_t text_handle = 0;
    int64_t expert_handle = 0;
    std::optional<v3::lingbot_v2_moe_internal::ExactSuffixProfileDescriptor>
        expert_profile;
    bool outer_twostage = false;
    bool vision_prepared = false;
    bool text_prepared = false;
    bool vision_primed = false;
    bool text_primed = false;
    size_t persistent_top_bytes = 0;
    uint64_t policy_version = 0;
    uint64_t plan_hash = 0;
    uint64_t authority_digest = 0;
    uint64_t bootstrap_build_count = 0;
    uint64_t retained_build_count = 0;
    uint64_t replay_count = 0;
    uint64_t forward_count = 0;
    uint32_t position_ids_version = 0;
    uint32_t rope_cos_il_version = 0;
    uint32_t rope_sin_il_version = 0;
    at::Tensor position_ids;
    std::optional<at::Tensor> rope_cos_il;
    std::optional<at::Tensor> rope_sin_il;
    std::optional<v3::SpmFmbResolvedExecutionProfile> vision_profile;
    std::optional<v3::SpmFmbResolvedExecutionProfile> text_profile;
    std::optional<v3::SpmFmbResolvedPhaseManifest> vision_manifest;
    std::optional<v3::SpmFmbResolvedPhaseManifest> text_manifest;
    std::vector<v3::SpmFmbConsumerRowSliceEndpoint> direct_endpoints;
    std::vector<v3::SpmFmbConsumerRowSliceEndpoint> run_add_endpoints;
    std::optional<v3::SpmFmbCompositePhysicalReservation> reservation;
    std::optional<v3::SpmFmbCompositePhysicalPlan> committed_plan;
    std::unique_ptr<v3::SpmPipelineLease> lease;
    bool lease_committed = false;
    bool graph_abort_stranded = false;
    bool physical_components_stranded = false;
    std::unique_ptr<RpuKernelGraph> retained_graph;
    std::optional<v3::SpmFmbSealedCompositeTrace> retained_trace;
    std::optional<RpuRetainedPhysicalArenaGuard> retained_guard;
};

Qwen3VlMultiviewZ2State& state() {
    static Qwen3VlMultiviewZ2State value;
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

void require_owner_thread(const Qwen3VlMultiviewZ2State& s,
                          const char* operation) {
    TORCH_CHECK(s.owner_thread == std::this_thread::get_id(), operation,
                ": retained physical canary is thread-affine");
}

void validate_position_ids(const at::Tensor& position_ids,
                           const char* operation) {
    TORCH_CHECK(
        position_ids.defined() && position_ids.dim() == 2 &&
            position_ids.size(0) == kExecutionLen &&
            position_ids.size(1) == 3 &&
            position_ids.device().type() == at::kPrivateUse1 &&
            position_ids.scalar_type() == at::kInt &&
            position_ids.is_contiguous(),
        operation, ": position_ids must be contiguous RPU int32 [225,3]");
    TORCH_CHECK(
        position_ids.unsafeGetTensorImpl()->version_counter().enabled(),
        operation,
        ": inference-mode position_ids are unsupported because the retained "
        "canary requires an explicit content-version gate");
}

void validate_rope_il(const at::Tensor& rope_cos_il,
                      const at::Tensor& rope_sin_il,
                      const char* operation) {
    TORCH_CHECK(
        rope_cos_il.defined() && rope_sin_il.defined() &&
            rope_cos_il.dim() == 2 && rope_cos_il.size(0) == kExecutionLen &&
            rope_cos_il.size(1) == kMropeHalf &&
            rope_sin_il.sizes() == rope_cos_il.sizes() &&
            rope_cos_il.device().type() == at::kPrivateUse1 &&
            rope_sin_il.device().type() == at::kPrivateUse1 &&
            rope_cos_il.scalar_type() == at::kHalf &&
            rope_sin_il.scalar_type() == at::kHalf &&
            rope_cos_il.is_contiguous() && rope_sin_il.is_contiguous(),
        operation,
        ": rope_cos_il/rope_sin_il must be matching contiguous RPU FP16 "
        "[225,64]");
    TORCH_CHECK(
        rope_cos_il.unsafeGetTensorImpl()->version_counter().enabled() &&
            rope_sin_il.unsafeGetTensorImpl()->version_counter().enabled(),
        operation,
        ": inference-mode rope_cos_il/rope_sin_il are unsupported because "
        "the retained canary requires an explicit content-version gate");
}

void validate_invocation(
    const Qwen3VlMultiviewZ2State& s,
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il,
    uint64_t plan_hash,
    const char* operation) {
    TORCH_CHECK(
        (s.phase == CanaryPhase::Reserved ||
         s.phase == CanaryPhase::Retained) &&
            s.reservation.has_value() &&
            ((s.phase == CanaryPhase::Reserved && s.lease) ||
             (s.phase == CanaryPhase::Retained && !s.lease &&
              s.committed_plan.has_value() &&
              s.retained_guard.has_value() &&
              s.retained_guard->active() &&
              !s.retained_guard->bound())),
        operation, ": prepare must complete before forward");
    require_owner_thread(s, operation);
    TORCH_CHECK(s.vision_handle == vision_handle &&
                    s.text_handle == text_handle &&
                    s.expert_handle == expert_handle &&
                    s.plan_hash == plan_hash,
                operation, ": handle or plan identity is stale");
    TORCH_CHECK(
        (expert_handle == 0 && !s.expert_profile.has_value()) ||
            (expert_handle > 0 && s.expert_profile.has_value()),
        operation, ": expert typed-profile identity is stale");
    TORCH_CHECK(vision_inputs.size() == kImageCount,
                operation, ": exactly three Vision inputs are required");
    for (int64_t image = 0; image < kImageCount; ++image) {
        const at::Tensor& input = vision_inputs[image];
        TORCH_CHECK(
            input.defined() && input.dim() == 3 && input.size(0) == 1 &&
                input.size(1) == kNumPatches &&
                input.size(2) == kVisionHidden &&
                input.device().type() == at::kPrivateUse1 &&
                input.scalar_type() == at::kHalf && input.is_contiguous(),
            operation, ": Vision input ", image,
            " must be contiguous RPU FP16 [1,256,1024]");
    }
    TORCH_CHECK(
        vision_k_caches.size() == kVisionLayers &&
            vision_v_caches.size() == kVisionLayers &&
            text_k_caches.size() == kTextLayers &&
            text_v_caches.size() == kTextLayers,
        operation, ": expected Vision/Text K/V layer counts 24/36");
    TORCH_CHECK(
        text_hidden.defined() && text_hidden.dim() == 3 &&
            text_hidden.size(0) == 1 &&
            text_hidden.size(1) == kExecutionLen &&
            text_hidden.size(2) == kTextHidden &&
            text_hidden.device().type() == at::kPrivateUse1 &&
            text_hidden.scalar_type() == at::kHalf &&
            text_hidden.is_contiguous(),
        operation, ": Text carrier must be contiguous RPU FP16 [1,225,2560]");
    validate_position_ids(position_ids, operation);
    TORCH_CHECK(
        rope_cos_il.has_value() == rope_sin_il.has_value() &&
            s.rope_cos_il.has_value() == rope_cos_il.has_value() &&
            s.rope_sin_il.has_value() == rope_sin_il.has_value(),
        operation, ": partial M-RoPE ownership mode drifted after prepare");
    TORCH_CHECK(
        s.position_ids.defined() &&
            s.position_ids.unsafeGetTensorImpl() ==
                position_ids.unsafeGetTensorImpl() &&
            s.position_ids.data_ptr<int32_t>() ==
                position_ids.data_ptr<int32_t>() &&
            position_ids.unsafeGetTensorImpl()
                    ->version_counter()
                    .current_version() == s.position_ids_version,
        operation,
        ": position_ids owner/address/content version drifted since the last "
        "prime; inputs must remain unchanged until forward");
    if (rope_cos_il.has_value()) {
        validate_rope_il(*rope_cos_il, *rope_sin_il, operation);
        TORCH_CHECK(
            s.rope_cos_il->unsafeGetTensorImpl() ==
                    rope_cos_il->unsafeGetTensorImpl() &&
                s.rope_sin_il->unsafeGetTensorImpl() ==
                    rope_sin_il->unsafeGetTensorImpl() &&
                s.rope_cos_il->data_ptr<c10::Half>() ==
                    rope_cos_il->data_ptr<c10::Half>() &&
                s.rope_sin_il->data_ptr<c10::Half>() ==
                    rope_sin_il->data_ptr<c10::Half>() &&
                rope_cos_il->unsafeGetTensorImpl()
                        ->version_counter().current_version() ==
                    s.rope_cos_il_version &&
                rope_sin_il->unsafeGetTensorImpl()
                        ->version_counter().current_version() ==
                    s.rope_sin_il_version,
            operation,
            ": partial M-RoPE owner/address/content version drifted after "
            "the last prime");
    }
}

v3::SpmDense2DSpec produced_spec() {
    v3::SpmDense2DSpec spec;
    spec.dtype = v3::SpmPortDType::Fp16;
    spec.rows = kMergedRows;
    spec.cols = kTextHidden;
    spec.distribution = v3::SpmPortDistribution::Replicated;
    spec.validate();
    return spec;
}

GraphDdrRegisterAbi outer_ddr_abi(
    GraphDdrRegisterRole role,
    GraphDdrRegisterAccess access,
    uint16_t reg_lo,
    uint16_t reg_hi) {
    return GraphDdrRegisterAbi{
        role, access, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
        reg_lo, reg_hi};
}

GraphKernelRegisterRule outer_typed_rule(
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

GraphKernelRegisterRule outer_intrinsic_no_ddr_rule(
    KernelId id,
    size_t count) {
    GraphKernelRegisterRule rule;
    rule.kernel_id = id;
    rule.kind = GraphKernelRegisterRuleKind::IntrinsicNoDdr;
    rule.expected_count = count;
    return rule;
}

GraphKernelRegisterRule outer_explicit_no_ddr_rule(
    KernelId id,
    size_t count,
    GraphKernelNoDdrProof proof) {
    GraphKernelRegisterRule rule;
    rule.kernel_id = id;
    rule.kind = GraphKernelRegisterRuleKind::ExplicitNoDdr;
    rule.no_ddr_proof = proof;
    rule.expected_count = count;
    return rule;
}

GraphKernelRegisterCensusStats outer_phase_stats(
    size_t nodes,
    size_t dmas,
    size_t barriers,
    size_t kernels,
    size_t typed,
    size_t no_ddr,
    size_t operands,
    size_t weights,
    size_t keys,
    size_t values,
    size_t semantic_inputs,
    uint64_t node_hash,
    uint64_t kernel_hash) {
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

const GraphKernelRegisterPolicy& lingbot2_multiview_w8_register_policy() {
    // These constants define the bounded ordered node and kernel streams for
    // the combined linear/GELU/MRoPE/ring path. Any launch-order or
    // kernel-identity drift fails closed.
    constexpr uint64_t kOrderedNodeKindHash =
        UINT64_C(0x1bf03075ed732d0b);
    constexpr uint64_t kOrderedKernelHash =
        UINT64_C(0x515455cf8bfc3415);
    static const GraphKernelRegisterPolicy policy = [] {
        GraphKernelRegisterPolicy value;
        value.fingerprint = UINT64_C(0x4c423257385a3244);  // LB2W8Z2D
        value.name =
            "lingbot2_multiview_w8_outer_ring_register_census_v5";

        const auto int8_weight = outer_ddr_abi(
            GraphDdrRegisterRole::Int8ModelWeight,
            GraphDdrRegisterAccess::Read, 2, 3);
        const auto scale = outer_ddr_abi(
            GraphDdrRegisterRole::PerChannelScale,
            GraphDdrRegisterAccess::Read, 20, 21);
        const auto fp16_weight = outer_ddr_abi(
            GraphDdrRegisterRole::ModelWeight,
            GraphDdrRegisterAccess::Read, 2, 3);
        const auto key_write = outer_ddr_abi(
            GraphDdrRegisterRole::KeyCache,
            GraphDdrRegisterAccess::Write, 8, 9);
        const auto value_write = outer_ddr_abi(
            GraphDdrRegisterRole::ValueCache,
            GraphDdrRegisterAccess::Write, 8, 9);
        const auto key_read = outer_ddr_abi(
            GraphDdrRegisterRole::KeyCache,
            GraphDdrRegisterAccess::Read, 50, 51);
        const auto value_read = outer_ddr_abi(
            GraphDdrRegisterRole::ValueCache,
            GraphDdrRegisterAccess::Read, 52, 53);
        const auto rope_position = outer_ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 0, 1);
        const auto partial_mrope_cos = outer_ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 0, 1);
        const auto partial_mrope_sin = outer_ddr_abi(
            GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, 2, 3);

        value.rules = {
            outer_typed_rule(
                KernelId::PL_AT_W8A16_M320N112, 144,
                {int8_weight, scale}),
            outer_typed_rule(
                KernelId::PL_AT_W8A16_M352N96, 180,
                {int8_weight, scale}),
            outer_typed_rule(
                KernelId::PL_AT_W8A16_M400N80, 72,
                {int8_weight, scale}),
            outer_typed_rule(
                KernelId::PL_AT_W8A16_M512N48, 288,
                {int8_weight, scale}),
            outer_typed_rule(
                KernelId::PL_AT_FP16_M528N48, 12, {fp16_weight}),
            outer_typed_rule(
                KernelId::PL_AT_FP16_M608N32, 12, {fp16_weight}),
            outer_typed_rule(
                KernelId::ROPE_2D_SPM, 144, {rope_position}),
            outer_typed_rule(
                KernelId::PARTIAL_MROPE, 72,
                {partial_mrope_cos, partial_mrope_sin}),
            outer_typed_rule(
                KernelId::LLAMA_INSERT_KCACHE, 36, {key_write}),
            outer_typed_rule(
                KernelId::LLAMA_INSERT_VCACHE, 36, {value_write}),
            outer_typed_rule(
                KernelId::LLAMA_INSERT_KCACHE_V16, 72, {key_write}),
            outer_typed_rule(
                KernelId::LLAMA_INSERT_VCACHE_V16, 72, {value_write}),
            outer_typed_rule(
                KernelId::SDPA_FLASH_ATTN_SPM, 108,
                {key_read, value_read}),
            outer_explicit_no_ddr_rule(
                KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_PACED, 228,
                GraphKernelNoDdrProof::RingAllReduceSpmResidual),
            outer_intrinsic_no_ddr_rule(
                KernelId::MEMSET_SPM_MULTI_CORE_V2, 72),
            outer_intrinsic_no_ddr_rule(KernelId::LAYER_NORM, 156),
            outer_intrinsic_no_ddr_rule(KernelId::RMS_NORM_BF16_SPM, 145),
            outer_intrinsic_no_ddr_rule(KernelId::BINARY_SAMESHAPE, 45),
            outer_intrinsic_no_ddr_rule(KernelId::UNARY_SPM, 36),
            outer_intrinsic_no_ddr_rule(
                KernelId::UNARY_GELU_TANH_SPM, 72),
            outer_intrinsic_no_ddr_rule(
                KernelId::UNARY_GELU_ERF_SPM, 12),
        };

        value.expected_node_count = 16818;
        value.expected_dma_count = 5466;
        value.expected_barrier_count = 9338;
        value.expected_kernel_count = 2014;
        value.expected_typed_kernel_count = 1248;
        value.expected_no_ddr_kernel_count = 766;
        value.expected_operand_count = 2112;
        value.expected_weight_operand_count = 1392;
        value.expected_key_cache_operand_count = 216;
        value.expected_value_cache_operand_count = 216;
        value.expected_semantic_input_operand_count = 288;
        value.expected_ordered_node_kind_hash = kOrderedNodeKindHash;
        value.expected_ordered_kernel_hash = kOrderedKernelHash;
        value.phases = {GraphKernelRegisterPhaseExpectation{
            "outer",
            outer_phase_stats(
                16818, 5466, 9338, 2014, 1248, 766,
                2112, 1392, 216, 216, 288,
                kOrderedNodeKindHash, kOrderedKernelHash)}};
        return value;
    }();
    return policy;
}

bool uses_lingbot2_multiview_grouped_outer_fast(
    const Qwen3VlMultiviewZ2State& s) {
    if (!s.outer_twostage || s.expert_handle <= 0 ||
        !s.expert_profile.has_value()) {
        return false;
    }
    const auto kind = s.expert_profile->kind;
    return kind == v3::lingbot_v2_moe_internal::ExactSuffixProfileKind::
                       GroupedW8A16 ||
        kind == v3::lingbot_v2_moe_internal::ExactSuffixProfileKind::
                    GroupedW4A16;
}

GraphSignature graph_signature(const Qwen3VlMultiviewZ2State& s,
                               bool retained) {
    TORCH_INTERNAL_ASSERT(s.vision_profile.has_value() &&
                          s.text_profile.has_value() &&
                          s.plan_hash != 0 && s.policy_version != 0);
    GraphSignature signature;
    const bool production = s.expert_handle != 0;
    signature.op_id = retained
        ? (production ? INT64_C(0x4c42325a32505202)
                      : INT64_C(0x51564d565a324202))
        : (production ? INT64_C(0x4c42325a32504201)
                      : INT64_C(0x51564d565a324101));
    signature.op_id_str = retained
        ? (production ? "lingbot2_multiview_spm_z2_retained"
                      : "qwen3vl_multiview_spm_z2_retained")
        : (production ? "lingbot2_multiview_spm_z2_bootstrap"
                      : "qwen3vl_multiview_spm_z2_bootstrap");
    signature.shapes = {
        kImageCount, kNumPatches, kVisionHidden,
        kExecutionLen, kTextHidden, 3, kMropeHalf,
        s.expert_handle,
        encode_u64(s.vision_profile->profile_hash()),
        encode_u64(s.text_profile->profile_hash()),
        production ? static_cast<int64_t>(s.expert_profile->kind) : 0,
        production ? encode_u64(s.expert_profile->profile_hash) : 0,
        production ? encode_u64(s.expert_profile->layout_hash) : 0,
        production
            ? static_cast<int64_t>(s.expert_profile->temporary_bytes)
            : 0,
    };
    signature.dtypes = {
        static_cast<uint8_t>(at::kHalf),
        static_cast<uint8_t>(at::kInt),
    };
    signature.flags = kSignatureSchema;
    signature.branch_key = retained ? UINT64_C(2) : UINT64_C(1);
    uint64_t segment = s.plan_hash ^ s.policy_version ^
        (retained ? UINT64_C(0xb4b4b4b4b4b4b4b4)
                  : UINT64_C(0xa3a3a3a3a3a3a3a3));
    if (retained && s.authority_digest != 0) {
        segment ^= s.authority_digest;
    }
    if (production) {
        segment ^= s.expert_profile->profile_hash;
        segment ^= s.expert_profile->layout_hash;
        segment ^= static_cast<uint64_t>(s.expert_profile->temporary_bytes);
    }
    signature.segment_key = segment == 0 ? signature.branch_key : segment;
    return signature;
}

struct CompositeBindings {
    std::vector<v3::SpmFmbCompositeDirectProducedBinding> direct;
    std::vector<v3::SpmFmbCompositeRunAddBinding> run_add;
};

CompositeBindings bind_composite(
    const Qwen3VlMultiviewZ2State& s,
    const v3::SpmFmbSealedCompositeTrace& trace) {
    TORCH_INTERNAL_ASSERT(s.direct_endpoints.size() == kDirectBindingCount &&
                          s.run_add_endpoints.size() == kRunAddBindingCount);
    CompositeBindings result;
    result.direct.reserve(kDirectBindingCount);
    for (size_t image = 0; image < kDirectBindingCount; ++image) {
        result.direct.push_back(
            v3::SpmFmbCompositeDirectProducedBinding::bind(
                trace, image, /*producer_yield_ordinal=*/3,
                /*consumer_occurrence_ordinal=*/3,
                s.direct_endpoints[image], s.policy_version));
    }
    result.run_add.reserve(kRunAddBindingCount);
    for (size_t tap = 0; tap < 3; ++tap) {
        for (size_t image = 0; image < kImageCount; ++image) {
            const size_t endpoint = tap * kImageCount + image;
            result.run_add.push_back(
                v3::SpmFmbCompositeRunAddBinding::bind(
                    trace, image, tap,
                    /*consumer_occurrence_ordinal=*/3,
                    s.run_add_endpoints[endpoint], s.policy_version));
        }
    }
    return result;
}

at::Tensor drive_build_occurrences(
    Qwen3VlMultiviewZ2State& s,
    v3::SpmCompositeTraceCoordinator& coordinator,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids) {
    for (size_t image = 0; image < kImageCount; ++image) {
        auto occurrence = coordinator.begin_build_occurrence(
            v3::qwen3vl_pooler_z1_internal::multiview_vision_owner(
                s.vision_handle),
            *s.vision_profile);
        v3::qwen3vl_pooler_z1_internal::forward_multiview_vision_composite(
            s.vision_handle, vision_inputs[image],
            vision_k_caches, vision_v_caches);
        occurrence.finish();
    }
    auto occurrence = coordinator.begin_build_occurrence(
        v3::qwen3vl_pooler_z1_internal::multiview_text_owner(
            s.text_handle),
        *s.text_profile);
    at::Tensor result =
        v3::qwen3vl_pooler_z1_internal::forward_multiview_text_composite(
            s.text_handle, text_hidden, text_k_caches, text_v_caches,
            position_ids, s.rope_cos_il, s.rope_sin_il);
    occurrence.finish();
    return result;
}

at::Tensor drive_replay_occurrences(
    Qwen3VlMultiviewZ2State& s,
    v3::SpmCompositeTraceCoordinator& coordinator,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids) {
    for (size_t image = 0; image < kImageCount; ++image) {
        auto occurrence = coordinator.begin_replay_occurrence(
            v3::qwen3vl_pooler_z1_internal::multiview_vision_owner(
                s.vision_handle));
        v3::qwen3vl_pooler_z1_internal::forward_multiview_vision_composite(
            s.vision_handle, vision_inputs[image],
            vision_k_caches, vision_v_caches);
        occurrence.finish();
    }
    auto occurrence = coordinator.begin_replay_occurrence(
        v3::qwen3vl_pooler_z1_internal::multiview_text_owner(
            s.text_handle));
    at::Tensor result =
        v3::qwen3vl_pooler_z1_internal::forward_multiview_text_composite(
            s.text_handle, text_hidden, text_k_caches, text_v_caches,
            position_ids, s.rope_cos_il, s.rope_sin_il);
    occurrence.finish();
    return result;
}

struct BuildResult {
    v3::SpmFmbSealedCompositeTrace trace;
    at::Tensor output;
};

BuildResult run_build(
    Qwen3VlMultiviewZ2State& s,
    RpuKernelGraph& graph,
    bool committed,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids) {
    TORCH_INTERNAL_ASSERT(s.lease && s.vision_profile && s.text_profile);
    v3::SpmCompositeTraceCoordinator coordinator(
        kOccurrenceCount, kProducerYieldCount, s.policy_version);
    bool adoption_started = false;
    bool graph_open = false;
    try {
        coordinator.adopt_physical_component(
            v3::qwen3vl_pooler_z1_internal::multiview_vision_owner(
                s.vision_handle),
            *s.lease, /*producer=*/true);
        adoption_started = true;
        coordinator.adopt_physical_component(
            v3::qwen3vl_pooler_z1_internal::multiview_text_owner(
                s.text_handle),
            *s.lease, /*producer=*/false);

        graph.begin(graph_signature(s, committed));
        graph_open = true;
        TORCH_CHECK(graph.state() == RpuKernelGraph::State::RECORDING,
                    "qwen3vl multiview Z2 BUILD unexpectedly hit replay");
        coordinator.begin_build(graph, *s.lease, committed);
        std::unique_ptr<GraphKernelRegisterCensusGuard> register_census;
        if (committed && uses_lingbot2_multiview_grouped_outer_fast(s)) {
            TORCH_CHECK(
                s.authority_digest != 0 && s.committed_plan.has_value() &&
                    s.authority_digest ==
                        s.committed_plan->authority_digest(),
                "LingBot2 multiview grouped retained BUILD physical digest is "
                "missing or stale");
            register_census =
                std::make_unique<GraphKernelRegisterCensusGuard>(
                    graph, lingbot2_multiview_w8_register_policy(),
                    s.plan_hash, s.lease->epoch());
            register_census->stage_outer_fast_physical_digest(
                s.authority_digest);
            v3::qwen3vl_pooler_z1_internal::
                stage_multiview_vision_outer_fast_component(
                    s.vision_handle, *register_census,
                    vision_k_caches, vision_v_caches);
            v3::qwen3vl_pooler_z1_internal::
                stage_multiview_text_outer_fast_component(
                    s.text_handle, *register_census,
                    text_k_caches, text_v_caches);
        }
        at::Tensor output = drive_build_occurrences(
            s, coordinator, vision_inputs, vision_k_caches,
            vision_v_caches, text_hidden, text_k_caches, text_v_caches,
            position_ids);
        if (register_census) register_census->complete();
        graph.end();
        graph_open = false;
        auto trace = coordinator.seal_build(graph);
        coordinator.release_physical_components(*s.lease);
        adoption_started = false;
        return {std::move(trace), std::move(output)};
    } catch (...) {
        std::exception_ptr failure = std::current_exception();
        std::exception_ptr cleanup_failure;
        coordinator.cancel();
        if (graph_open) {
            try {
                graph.abort();
                graph_open = false;
            } catch (...) {
                if (!cleanup_failure) {
                    cleanup_failure = std::current_exception();
                }
            }
        }
        if (adoption_started && !RpuKernelGraph::has_active()) {
            std::exception_ptr release_failure;
            try {
                coordinator.release_physical_components(*s.lease);
                adoption_started = false;
            } catch (...) {
                release_failure = std::current_exception();
                // A validation failure is normally pre-mutation.  Retry once
                // while the only object that can release the adoption is live.
                try {
                    coordinator.release_physical_components(*s.lease);
                    adoption_started = false;
                } catch (...) {
                    release_failure = std::current_exception();
                }
            }
            if (release_failure && adoption_started && !cleanup_failure) {
                cleanup_failure = release_failure;
            }
        }
        if (graph_open && RpuKernelGraph::has_active()) {
            s.graph_abort_stranded = true;
        }
        if (adoption_started) {
            // The coordinator is stack-local, so no outer close can release
            // an adoption after this function unwinds.  Retain the arena and
            // require process restart instead of releasing it underneath FMB.
            s.physical_components_stranded = true;
        }
        // A cleanup failure may leave a Graph claim or an adopted component
        // live, so it is the terminal diagnostic for this poisoned process.
        if (cleanup_failure) std::rethrow_exception(cleanup_failure);
        std::rethrow_exception(failure);
    }
}

at::Tensor run_replay(
    Qwen3VlMultiviewZ2State& s,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids) {
    TORCH_INTERNAL_ASSERT(s.lease && s.retained_graph && s.retained_trace &&
                          s.retained_guard && s.retained_guard->active() &&
                          !s.retained_guard->bound());
    v3::SpmCompositeTraceCoordinator coordinator(
        kOccurrenceCount, kProducerYieldCount, s.policy_version);
    bool adoption_started = false;
    bool graph_open = false;
    try {
        coordinator.adopt_physical_component(
            v3::qwen3vl_pooler_z1_internal::multiview_vision_owner(
                s.vision_handle),
            *s.lease, /*producer=*/true);
        adoption_started = true;
        coordinator.adopt_physical_component(
            v3::qwen3vl_pooler_z1_internal::multiview_text_owner(
                s.text_handle),
            *s.lease, /*producer=*/false);

        // The retained Graph is parked between public forwards.  Re-establish
        // the exact committed physical placement before making it replayable.
        s.lease->seal_physical();
        s.retained_guard->rebind(
            s.lease->physical_graph_authority());

        s.retained_graph->begin(graph_signature(s, /*retained=*/true));
        graph_open = true;
        TORCH_CHECK(
            s.retained_graph->state() == RpuKernelGraph::State::REPLAYING,
            "qwen3vl multiview Z2 retained Graph attempted a second BUILD");
        coordinator.begin_replay(
            *s.retained_trace, *s.retained_graph, *s.lease);
        std::unique_ptr<GraphKernelRegisterCensusGuard> register_census;
        const bool outer_fast =
            uses_lingbot2_multiview_grouped_outer_fast(s);
        if (outer_fast) {
            TORCH_CHECK(
                s.authority_digest != 0 && s.committed_plan.has_value() &&
                    s.authority_digest ==
                        s.committed_plan->authority_digest(),
                "LingBot2 multiview grouped REPLAY physical digest is missing or "
                "stale");
            register_census =
                std::make_unique<GraphKernelRegisterCensusGuard>(
                    *s.retained_graph,
                    lingbot2_multiview_w8_register_policy(),
                    s.plan_hash, s.lease->epoch());
            register_census->stage_outer_fast_physical_digest(
                s.authority_digest);
            v3::qwen3vl_pooler_z1_internal::
                stage_multiview_vision_outer_fast_component(
                    s.vision_handle, *register_census,
                    vision_k_caches, vision_v_caches);
            v3::qwen3vl_pooler_z1_internal::
                stage_multiview_text_outer_fast_component(
                    s.text_handle, *register_census,
                    text_k_caches, text_v_caches);
        }
        at::Tensor output;
        if (outer_fast) {
            TORCH_INTERNAL_ASSERT(register_census != nullptr);
            for (size_t image = 0; image < kImageCount; ++image) {
                v3::qwen3vl_pooler_z1_internal::
                    bind_multiview_vision_outer_fast_input(
                        s.vision_handle, *register_census,
                        vision_inputs[image], image);
            }
            v3::qwen3vl_pooler_z1_internal::
                bind_multiview_text_outer_fast_input(
                    s.text_handle, *register_census, text_hidden);
            output = v3::qwen3vl_pooler_z1_internal::
                multiview_text_outer_fast_output(s.text_handle);

            s.lease->validate_physical(
                s.lease->epoch(), s.plan_hash,
                /*require_sealed=*/true);
            TORCH_CHECK(
                s.authority_digest ==
                    s.committed_plan->authority_digest(),
                "LingBot2 multiview grouped REPLAY physical authority drifted "
                "after fast binding");
            auto composite_ticket =
                coordinator.validate_outer_fast_replay();
            auto census_ticket =
                register_census->validate_outer_fast_replay();
            register_census->commit_outer_fast_replay(
                std::move(census_ticket));
            coordinator.commit_outer_fast_replay(
                std::move(composite_ticket));
        } else {
            output = drive_replay_occurrences(
                s, coordinator, vision_inputs, vision_k_caches,
                vision_v_caches, text_hidden, text_k_caches, text_v_caches,
                position_ids);
            coordinator.finish_replay();
        }
        s.retained_graph->end();
        graph_open = false;
        coordinator.release_physical_components(*s.lease);
        adoption_started = false;
        s.retained_guard->park(s.lease->physical_graph_authority());
        s.lease->release();
        s.lease.reset();
        s.lease_committed = false;
        return output;
    } catch (...) {
        std::exception_ptr failure = std::current_exception();
        std::exception_ptr cleanup_failure;
        coordinator.cancel();
        if (graph_open) {
            try {
                s.retained_graph->abort();
                graph_open = false;
            } catch (...) {
                if (!cleanup_failure) {
                    cleanup_failure = std::current_exception();
                }
            }
        }
        if (adoption_started && !RpuKernelGraph::has_active()) {
            std::exception_ptr release_failure;
            try {
                coordinator.release_physical_components(*s.lease);
                adoption_started = false;
            } catch (...) {
                release_failure = std::current_exception();
                try {
                    coordinator.release_physical_components(*s.lease);
                    adoption_started = false;
                } catch (...) {
                    release_failure = std::current_exception();
                }
            }
            if (release_failure && adoption_started && !cleanup_failure) {
                cleanup_failure = release_failure;
            }
        }
        if (graph_open && RpuKernelGraph::has_active()) {
            s.graph_abort_stranded = true;
        }
        if (adoption_started) {
            s.physical_components_stranded = true;
        }
        // A cleanup failure may leave a Graph claim or an adopted component
        // live, so it is the terminal diagnostic for this poisoned process.
        if (cleanup_failure) std::rethrow_exception(cleanup_failure);
        std::rethrow_exception(failure);
    }
}

void reset_identity(Qwen3VlMultiviewZ2State& s) {
    s.phase = CanaryPhase::Idle;
    s.owner_thread = std::thread::id{};
    s.vision_handle = 0;
    s.text_handle = 0;
    s.expert_handle = 0;
    s.expert_profile.reset();
    s.outer_twostage = false;
    s.persistent_top_bytes = 0;
    s.policy_version = 0;
    s.plan_hash = 0;
    s.authority_digest = 0;
    s.bootstrap_build_count = 0;
    s.retained_build_count = 0;
    s.replay_count = 0;
    s.forward_count = 0;
    s.position_ids_version = 0;
    s.rope_cos_il_version = 0;
    s.rope_sin_il_version = 0;
    s.position_ids = at::Tensor();
    s.rope_cos_il.reset();
    s.rope_sin_il.reset();
    s.graph_abort_stranded = false;
    s.physical_components_stranded = false;
}

void teardown_locked(Qwen3VlMultiviewZ2State& s) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl multiview Z2 teardown must run outside Graph");
    require_owner_thread(s, "qwen3vl multiview Z2 teardown");

    if (s.retained_graph &&
        (s.retained_guard.has_value() ||
         s.retained_graph->has_provisional_physical_build() ||
         s.retained_graph->state() == RpuKernelGraph::State::BUILT)) {
        s.retained_graph->invalidate();
    }
    if (s.retained_guard.has_value()) {
        s.retained_guard->retire();
        s.retained_guard.reset();
    }
    s.retained_trace.reset();
    s.retained_graph.reset();

    TORCH_CHECK(
        !s.graph_abort_stranded && !s.physical_components_stranded,
        "qwen3vl multiview Z2 cleanup stranded a Graph or physical component "
        "adoption; the arena remains locked and the process must restart");

    if (s.lease) {
        // commit_composite intentionally precedes seal.  If the original seal
        // failed, retry it only after every Graph/provisional identity has
        // been retired; committed physical leases otherwise reject release.
        if (s.lease_committed && !s.lease->physical_sealed()) {
            s.lease->seal_physical();
        }
        s.lease->release();
        s.lease.reset();
        s.lease_committed = false;
    }
    s.committed_plan.reset();
    s.reservation.reset();
    s.run_add_endpoints.clear();
    s.direct_endpoints.clear();
    s.text_manifest.reset();
    s.vision_manifest.reset();
    s.text_profile.reset();
    s.vision_profile.reset();

    if (s.vision_primed) {
        v3::qwen3vl_pooler_z1_internal::clear_multiview_vision_position(
            s.vision_handle);
        s.vision_primed = false;
    }
    if (s.text_primed) {
        v3::qwen3vl_pooler_z1_internal::rollback_multiview_text_inputs(
            s.text_handle);
        s.text_primed = false;
    }
    if (s.text_prepared) {
        v3::qwen3vl_pooler_z1_internal::unprepare_multiview_text_composite(
            s.text_handle);
        s.text_prepared = false;
    }
    if (s.vision_prepared) {
        v3::qwen3vl_pooler_z1_internal::unprepare_multiview_vision_composite(
            s.vision_handle);
        s.vision_prepared = false;
    }
    reset_identity(s);
}

[[noreturn]] void teardown_after_failure(Qwen3VlMultiviewZ2State& s,
                                         std::exception_ptr original) {
    s.phase = CanaryPhase::Poisoned;
    try {
        teardown_locked(s);
    } catch (...) {
        // Keep every still-live field in place so an explicit close can retry
        // the fail-closed teardown.  The original execution error remains the
        // most actionable diagnostic for this call.
    }
    std::rethrow_exception(original);
}

}  // namespace

namespace {

int64_t prepare_impl(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    const at::Tensor& position_ids,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il,
    size_t expected_persistent_top,
    size_t expected_total,
    size_t expected_headroom,
    const char* operation) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl multiview Z2 prepare must run outside Graph");
    TORCH_CHECK(vision_handle > 0 && text_handle > 0,
                "qwen3vl multiview Z2 prepare requires positive handles");
    TORCH_CHECK(expert_handle >= 0,
                "qwen3vl multiview Z2 prepare rejects a negative expert handle");
    validate_position_ids(position_ids, operation);
    TORCH_CHECK(
        rope_cos_il.has_value() == rope_sin_il.has_value() &&
            (expert_handle > 0) == rope_cos_il.has_value(),
        operation,
        ": LingBot2 production requires partial M-RoPE cos/sin while the "
        "generic canary preserves legacy M-RoPE");
    if (rope_cos_il.has_value()) {
        validate_rope_il(*rope_cos_il, *rope_sin_il, operation);
    }

    Qwen3VlMultiviewZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.phase == CanaryPhase::Idle,
                "qwen3vl multiview Z2 has an active or poisoned lifecycle");

    std::optional<v3::lingbot_v2_moe_internal::ExactSuffixProfileDescriptor>
        expert_profile;
    if (expert_handle != 0) {
        expert_profile.emplace(
            v3::lingbot_v2_moe_internal::
                describe_exact_suffix_spm_profile(
                    expert_handle, kExecutionLen));
        const size_t projected_live_persistent_top = expected_persistent_top;
        TORCH_CHECK(
            projected_live_persistent_top <= kExpectedBudgetBytes &&
                expert_profile->temporary_bytes <=
                    kExpectedBudgetBytes - projected_live_persistent_top,
            operation, ": expert temporary plus projected live persistent "
            "top overflows the aligned 7.919 MiB ceiling");
        const size_t sequential_live_bytes =
            expert_profile->temporary_bytes + projected_live_persistent_top;
        TORCH_CHECK(
            sequential_live_bytes <= kExpectedBudgetBytes,
            operation, ": expert sequential live bytes exceed aligned "
            "SPM ceiling ", kExpectedBudgetBytes, "; temporary/persistent/"
            "total=", expert_profile->temporary_bytes, "/",
            projected_live_persistent_top, "/", sequential_live_bytes);
    }

    // Product-scoped cold policy: sample the exact env once at outer prepare.
    // The generic Qwen3-VL canary deliberately ignores ambient LingBot2 state.
    const char* outer_twostage_env =
        std::getenv("RPU_LINGBOT2_OUTER_TWOSTAGE");
    bool outer_twostage = false;
    if (expert_profile.has_value()) {
        const bool exact_zero = outer_twostage_env != nullptr &&
            std::strcmp(outer_twostage_env, "0") == 0;
        const bool exact_one = outer_twostage_env != nullptr &&
            std::strcmp(outer_twostage_env, "1") == 0;
        if (expert_profile->kind ==
            v3::lingbot_v2_moe_internal::ExactSuffixProfileKind::DenseFp16) {
            TORCH_CHECK(
                exact_zero, operation,
                ": DenseFp16 requires exact "
                "RPU_LINGBOT2_OUTER_TWOSTAGE=0");
        } else {
            TORCH_INTERNAL_ASSERT(
                expert_profile->kind ==
                    v3::lingbot_v2_moe_internal::ExactSuffixProfileKind::
                        GroupedW8A16 ||
                expert_profile->kind ==
                    v3::lingbot_v2_moe_internal::ExactSuffixProfileKind::
                        GroupedW4A16);
            TORCH_CHECK(
                exact_one, operation,
                ": GroupedW8A16/GroupedW4A16 requires exact "
                "RPU_LINGBOT2_OUTER_TWOSTAGE=1");
            outer_twostage = true;
        }
    }

    s.owner_thread = std::this_thread::get_id();
    s.vision_handle = vision_handle;
    s.text_handle = text_handle;
    s.expert_handle = expert_handle;
    s.expert_profile = expert_profile;
    s.outer_twostage = outer_twostage;

    try {
        const auto vision_layout =
            v3::qwen3vl_pooler_z1_internal::
                prepare_multiview_vision_composite(
                    vision_handle, kNumPatches, s.outer_twostage);
        s.vision_prepared = true;
        const auto text_layout =
            v3::qwen3vl_pooler_z1_internal::
                prepare_multiview_text_composite(
                    text_handle, kExecutionLen, s.outer_twostage);
        s.text_prepared = true;
        TORCH_CHECK(
            vision_layout.temporary_bytes == kVisionTemporaryBytes &&
                text_layout.temporary_bytes == kTextTemporaryBytes,
            "qwen3vl multiview Z2 component scratch drifted; expected ",
            kVisionTemporaryBytes, "/", kTextTemporaryBytes,
            ", got ", vision_layout.temporary_bytes, "/",
            text_layout.temporary_bytes);

        // Product-only cold allocation.  It deliberately runs after both V/T
        // component prepares and before their profiles/manifests are resolved:
        // the expert's first Path-1 reset_all preserves V/T super-persistent
        // weights, while V/T adoption below explicitly acknowledges the sibling
        // persistent-generation advance.  The later expert forward therefore
        // needs only a temporary Path-2 rebuild and cannot invalidate a parked
        // retained Graph's persistent-generation identity.
        if (expert_handle != 0) {
            TORCH_CHECK(
                SPM_ALLOC.persistent_used() == 0 &&
                    live_persistent_top_bytes() == kPersistentTopBytes,
                "LingBot2 multiview Z2 base Vision/Text persistent top "
                "drifted before expert cold prime; expected ordinary/top=0/",
                kPersistentTopBytes, ", got ",
                SPM_ALLOC.persistent_used(), "/",
                live_persistent_top_bytes());
            const auto expert_layout =
                v3::lingbot_v2_moe_internal::
                    prime_exact_suffix_spm_layout(
                        expert_handle, kExecutionLen, *s.expert_profile);
            TORCH_CHECK(
                expert_layout.temporary_bytes ==
                        s.expert_profile->temporary_bytes &&
                    expert_layout.layout_hash == s.expert_profile->layout_hash,
                "LingBot2 expert cold prime changed its admitted identity");
        }

        s.vision_profile.emplace(
            v3::qwen3vl_pooler_z1_internal::
                resolve_multiview_vision_profile(vision_handle));
        s.text_profile.emplace(
            v3::qwen3vl_pooler_z1_internal::
                resolve_multiview_text_profile(text_handle));
        s.text_manifest.emplace(
            v3::qwen3vl_pooler_z1_internal::
                seal_multiview_text_manifest(
                    text_handle, *s.text_profile, kTextArena,
                    /*arena_base=*/0));
        s.vision_manifest.emplace(
            v3::qwen3vl_pooler_z1_internal::
                seal_multiview_vision_manifest(
                    vision_handle, *s.vision_profile, kVisionArena,
                    static_cast<uint32_t>(kConsumerRootBytes)));
        s.policy_version =
            v3::qwen3vl_pooler_z1_internal::
                multiview_vision_occurrence_policy(vision_handle);
        TORCH_CHECK(s.policy_version != 0,
                    "qwen3vl multiview Z2 occurrence policy is zero");

        s.direct_endpoints.reserve(kDirectBindingCount);
        for (size_t image = 0; image < kImageCount; ++image) {
            s.direct_endpoints.push_back(
                v3::qwen3vl_pooler_z1_internal::
                    seal_multiview_text_consumer(
                        text_handle, *s.text_profile, *s.text_manifest,
                        v3::SpmPortId(static_cast<uint32_t>(101 + image)),
                        kImageRowBegins[image], /*layer_idx=*/0));
        }
        s.run_add_endpoints.reserve(kRunAddBindingCount);
        for (size_t tap = 0; tap < 3; ++tap) {
            for (size_t image = 0; image < kImageCount; ++image) {
                const uint32_t port = static_cast<uint32_t>(
                    201 + tap * kImageCount + image);
                s.run_add_endpoints.push_back(
                    v3::qwen3vl_pooler_z1_internal::
                        seal_multiview_text_consumer(
                            text_handle, *s.text_profile, *s.text_manifest,
                            v3::SpmPortId(port), kImageRowBegins[image],
                            static_cast<int>(tap)));
            }
        }

        s.persistent_top_bytes = live_persistent_top_bytes();
        TORCH_CHECK(s.persistent_top_bytes == expected_persistent_top,
                    "qwen3vl multiview Z2 persistent top drifted; expected ",
                    expected_persistent_top, ", got ",
                    s.persistent_top_bytes);
        if (s.expert_profile.has_value()) {
            TORCH_CHECK(
                s.expert_profile->temporary_bytes <=
                    kExpectedBudgetBytes - s.persistent_top_bytes,
                "LingBot2 expert temporary overlaps the live persistent top");
            const size_t sequential_live_bytes =
                s.expert_profile->temporary_bytes + s.persistent_top_bytes;
            TORCH_CHECK(
                sequential_live_bytes <= kExpectedBudgetBytes,
                "LingBot2 expert temporary plus live persistent top exceeds "
                "the aligned 7.919 MiB ceiling");
        }
        s.reservation.emplace(
            v3::SpmFmbCompositePhysicalReservation::
                compile_repeated_dense_producer(
                    *s.vision_manifest, *s.text_manifest,
                    s.direct_endpoints, s.run_add_endpoints,
                    produced_spec(), kYieldsPerProducer, s.policy_version,
                    s.persistent_top_bytes,
                    SpmAllocator::SPM_PLANNING_BUDGET));
        TORCH_CHECK(
            s.reservation->dynamic_bytes() == kExpectedDynamicBytes &&
                s.reservation->total_bytes() == expected_total &&
                s.reservation->effective_budget_bytes() ==
                    kExpectedBudgetBytes &&
                s.reservation->headroom_bytes() == expected_headroom &&
                s.reservation->producer_occurrence_count() == kImageCount &&
                s.reservation->producer_yield_count() ==
                    kProducerYieldCount &&
                s.reservation->direct_binding_count() ==
                    kDirectBindingCount &&
                s.reservation->run_add_binding_count() ==
                    kRunAddBindingCount,
            "qwen3vl multiview Z2 exact 7.919 MiB reservation drifted");
        s.plan_hash = s.reservation->hash();
        s.lease = v3::SpmPipelineLease::acquire_physical(*s.reservation);

        v3::qwen3vl_pooler_z1_internal::prime_multiview_vision_position(
            vision_handle);
        s.vision_primed = true;
        v3::qwen3vl_pooler_z1_internal::prime_multiview_text_inputs(
            text_handle, position_ids, kExecutionLen,
            rope_cos_il, rope_sin_il);
        s.text_primed = true;
        s.position_ids = position_ids;
        s.position_ids_version = position_ids.unsafeGetTensorImpl()
                                     ->version_counter()
                                     .current_version();
        s.rope_cos_il = rope_cos_il;
        s.rope_sin_il = rope_sin_il;
        if (rope_cos_il.has_value()) {
            s.rope_cos_il_version = rope_cos_il->unsafeGetTensorImpl()
                                        ->version_counter()
                                        .current_version();
            s.rope_sin_il_version = rope_sin_il->unsafeGetTensorImpl()
                                        ->version_counter()
                                        .current_version();
        }
        s.phase = CanaryPhase::Reserved;
        return encode_u64(s.plan_hash);
    } catch (...) {
        teardown_after_failure(s, std::current_exception());
    }
}

at::Tensor forward_impl(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    const std::optional<at::Tensor>& rope_cos_il,
    const std::optional<at::Tensor>& rope_sin_il,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl multiview Z2 native outer forward owns its Graph");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlMultiviewZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    validate_invocation(
        s, vision_handle, text_handle, expert_handle, vision_inputs,
        vision_k_caches,
        vision_v_caches, text_hidden, text_k_caches, text_v_caches,
        position_ids, rope_cos_il, rope_sin_il, plan_hash,
        "qwen3vl multiview Z2 forward");

    try {
        if (s.phase == CanaryPhase::Reserved) {
            s.phase = CanaryPhase::Bootstrapping;
            {
                RpuKernelGraph bootstrap_graph;
                BuildResult bootstrap = run_build(
                    s, bootstrap_graph, /*committed=*/false,
                    vision_inputs, vision_k_caches, vision_v_caches,
                    text_hidden, text_k_caches, text_v_caches,
                    position_ids);
                CompositeBindings bindings = bind_composite(s, bootstrap.trace);
                auto committed_plan =
                    v3::SpmFmbCompositePhysicalPlan::commit(
                        *s.reservation, bootstrap.trace,
                        bindings.direct, bindings.run_add);
                s.lease->commit_composite(committed_plan);
                s.lease_committed = true;
                s.authority_digest = committed_plan.authority_digest();
                s.committed_plan.emplace(std::move(committed_plan));
                ++s.bootstrap_build_count;
            }

            // Sealing before B makes every post-commit failure releasable.  B
            // still performs the exact committed lexical address validation;
            // only its Graph is eligible for the retained-arena guard.
            s.lease->seal_physical();
            s.retained_graph = std::make_unique<RpuKernelGraph>();
            BuildResult retained = run_build(
                s, *s.retained_graph, /*committed=*/true,
                vision_inputs, vision_k_caches, vision_v_caches,
                text_hidden, text_k_caches, text_v_caches,
                position_ids);
            CompositeBindings retained_bindings =
                bind_composite(s, retained.trace);
            const auto retained_plan =
                v3::SpmFmbCompositePhysicalPlan::commit(
                    *s.reservation, retained.trace,
                    retained_bindings.direct, retained_bindings.run_add);
            TORCH_CHECK(
                retained_plan.hash() == s.committed_plan->hash() &&
                    retained_plan.dynamic_bytes() ==
                        s.committed_plan->dynamic_bytes() &&
                    retained_plan.total_bytes() ==
                        s.committed_plan->total_bytes() &&
                    retained_plan.direct_binding_count() ==
                        s.committed_plan->direct_binding_count() &&
                    retained_plan.run_add_binding_count() ==
                        s.committed_plan->run_add_binding_count(),
                "qwen3vl multiview Z2 retained BUILD placement drifted from "
                "the bootstrap commit");
            s.retained_trace.emplace(std::move(retained.trace));
            s.retained_guard.emplace(
                s.retained_graph->arm_retained_physical_arena(
                    s.lease->physical_graph_authority()));
            ++s.retained_build_count;
            ++s.forward_count;
            s.phase = CanaryPhase::Retained;
            s.retained_guard->park(
                s.lease->physical_graph_authority());
            s.lease->release();
            s.lease.reset();
            s.lease_committed = false;
            // Canary-only borrowed result: the next forward may overwrite the
            // same registry-stable Text output.  Callers must consume/copy it
            // before another invocation; this is not a production contract.
            return retained.output;
        }

        TORCH_INTERNAL_ASSERT(s.phase == CanaryPhase::Retained);
        s.lease = v3::SpmPipelineLease::acquire_physical(
            *s.committed_plan);
        s.lease_committed = true;
        at::Tensor output = run_replay(
            s, vision_inputs, vision_k_caches, vision_v_caches,
            text_hidden, text_k_caches, text_v_caches, position_ids);
        ++s.replay_count;
        ++s.forward_count;
        // Same borrowed-output contract as the retained BUILD return above.
        return output;
    } catch (...) {
        teardown_after_failure(s, std::current_exception());
    }
}

void refresh_text_inputs_impl(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(
        !RpuKernelGraph::has_active(),
        "LingBot2 multiview Z2 text-input refresh must run outside Graph");
    TORCH_CHECK(
        expert_handle > 0,
        "LingBot2 multiview Z2 text-input refresh requires a positive expert "
        "handle");
    validate_position_ids(
        position_ids, "LingBot2 multiview Z2 text-input refresh");
    validate_rope_il(
        rope_cos_il, rope_sin_il,
        "LingBot2 multiview Z2 text-input refresh");

    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlMultiviewZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(
        s.phase == CanaryPhase::Retained && s.retained_graph &&
            s.retained_graph->state() == RpuKernelGraph::State::BUILT &&
            s.retained_trace.has_value() && s.retained_guard.has_value() &&
            s.retained_guard->active() && !s.retained_guard->bound() &&
            !s.lease && s.committed_plan.has_value() &&
            s.reservation.has_value() && s.text_prepared && s.text_primed,
        "LingBot2 multiview Z2 text-input refresh requires one parked "
        "retained Graph");
    require_owner_thread(s, "LingBot2 multiview Z2 text-input refresh");
    TORCH_CHECK(
        plan_hash != 0 && s.vision_handle == vision_handle &&
            s.text_handle == text_handle && s.expert_handle == expert_handle &&
            s.expert_profile.has_value() && s.plan_hash == plan_hash,
        "LingBot2 multiview Z2 text-input refresh handle or plan identity is "
        "stale");
    TORCH_CHECK(
        live_persistent_top_bytes() == s.persistent_top_bytes,
        "LingBot2 multiview Z2 text-input refresh persistent top drifted");
    TORCH_CHECK(
        s.position_ids.defined() &&
            s.position_ids.unsafeGetTensorImpl() ==
                position_ids.unsafeGetTensorImpl() &&
            s.position_ids.data_ptr<int32_t>() ==
                position_ids.data_ptr<int32_t>() &&
            s.rope_cos_il.has_value() && s.rope_sin_il.has_value() &&
            s.rope_cos_il->unsafeGetTensorImpl() ==
                rope_cos_il.unsafeGetTensorImpl() &&
            s.rope_sin_il->unsafeGetTensorImpl() ==
                rope_sin_il.unsafeGetTensorImpl() &&
            s.rope_cos_il->data_ptr<c10::Half>() ==
                rope_cos_il.data_ptr<c10::Half>() &&
            s.rope_sin_il->data_ptr<c10::Half>() ==
                rope_sin_il.data_ptr<c10::Half>(),
        "LingBot2 multiview Z2 text-input refresh requires the exact stable "
        "position/M-RoPE owners and addresses from prepare");

    const uint32_t position_ids_version = position_ids.unsafeGetTensorImpl()
                                              ->version_counter()
                                              .current_version();
    const uint32_t rope_cos_il_version = rope_cos_il.unsafeGetTensorImpl()
                                             ->version_counter()
                                             .current_version();
    const uint32_t rope_sin_il_version = rope_sin_il.unsafeGetTensorImpl()
                                             ->version_counter()
                                             .current_version();
    try {
        v3::qwen3vl_pooler_z1_internal::prime_multiview_text_inputs(
            text_handle, position_ids, kExecutionLen,
            rope_cos_il, rope_sin_il);
        TORCH_CHECK(
            position_ids.unsafeGetTensorImpl()
                        ->version_counter()
                        .current_version() == position_ids_version &&
                rope_cos_il.unsafeGetTensorImpl()
                        ->version_counter()
                        .current_version() == rope_cos_il_version &&
                rope_sin_il.unsafeGetTensorImpl()
                        ->version_counter()
                        .current_version() == rope_sin_il_version,
            "LingBot2 multiview Z2 text-input sources changed during refresh");
        s.position_ids_version = position_ids_version;
        s.rope_cos_il_version = rope_cos_il_version;
        s.rope_sin_il_version = rope_sin_il_version;
    } catch (...) {
        teardown_after_failure(s, std::current_exception());
    }
}

std::vector<int64_t> stats_impl(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl multiview Z2 stats must run outside Graph");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlMultiviewZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    TORCH_CHECK(s.phase != CanaryPhase::Idle,
                "qwen3vl multiview Z2 stats require a prepared lifecycle");
    require_owner_thread(s, "qwen3vl multiview Z2 stats");
    TORCH_CHECK(s.vision_handle == vision_handle &&
                    s.text_handle == text_handle &&
                    s.expert_handle == expert_handle &&
                    s.plan_hash == plan_hash && s.reservation.has_value(),
                "qwen3vl multiview Z2 stats identity is stale");

    GraphStats graph_stats;
    uint64_t build_generation = 0;
    size_t graph_size = 0;
    if (s.retained_graph) {
        graph_stats = s.retained_graph->debug_stats();
        build_generation = s.retained_graph->build_generation();
        graph_size = s.retained_graph->graph_size();
    }
    return {
        /*schema=*/3,
        static_cast<int64_t>(s.phase),
        encode_u64(s.plan_hash),
        static_cast<int64_t>(s.reservation->dynamic_bytes()),
        static_cast<int64_t>(s.reservation->reserved_top_bytes()),
        static_cast<int64_t>(s.reservation->total_bytes()),
        static_cast<int64_t>(s.reservation->effective_budget_bytes()),
        static_cast<int64_t>(s.reservation->headroom_bytes()),
        static_cast<int64_t>(s.bootstrap_build_count),
        static_cast<int64_t>(s.retained_build_count),
        static_cast<int64_t>(s.replay_count),
        static_cast<int64_t>(s.forward_count),
        static_cast<int64_t>(build_generation),
        static_cast<int64_t>(graph_size),
        static_cast<int64_t>(graph_stats.kernel_count),
        static_cast<int64_t>(graph_stats.data_node_count),
        static_cast<int64_t>(graph_stats.segment_count),
        static_cast<int64_t>(graph_stats.prepared_segment_hit_total),
        static_cast<int64_t>(graph_stats.prepared_segment_miss_total),
        static_cast<int64_t>(graph_stats.hw_batch_submit_total),
        s.retained_guard.has_value() && s.retained_guard->active() ? 1 : 0,
        s.retained_graph &&
                s.retained_graph->has_retained_physical_arena_guard()
            ? 1
            : 0,
        s.retained_guard.has_value() && s.retained_guard->bound() ? 1 : 0,
        s.lease ? 1 : 0,
        graph_stats.register_census_present ? 1 : 0,
        graph_stats.register_census_build_committed ? 1 : 0,
        encode_u64(graph_stats.register_census_policy_fingerprint),
        encode_u64(graph_stats.register_census_policy_digest),
        encode_u64(graph_stats.register_census_plan_hash),
        encode_u64(graph_stats.register_census_live_epoch),
        static_cast<int64_t>(graph_stats.register_census.node_count),
        static_cast<int64_t>(graph_stats.register_census.dma_count),
        static_cast<int64_t>(graph_stats.register_census.barrier_count),
        static_cast<int64_t>(graph_stats.register_census.kernel_count),
        static_cast<int64_t>(
            graph_stats.register_census.typed_kernel_count),
        static_cast<int64_t>(
            graph_stats.register_census.no_ddr_kernel_count),
        static_cast<int64_t>(graph_stats.register_census.operand_count),
        static_cast<int64_t>(
            graph_stats.register_census.weight_operand_count),
        static_cast<int64_t>(
            graph_stats.register_census.key_cache_operand_count),
        static_cast<int64_t>(
            graph_stats.register_census.value_cache_operand_count),
        static_cast<int64_t>(
            graph_stats.register_census.semantic_input_operand_count),
        encode_u64(graph_stats.register_census.ordered_node_kind_hash),
        encode_u64(graph_stats.register_census.ordered_kernel_hash),
        graph_stats.register_census_outer_fast_hit ? 1 : 0,
        static_cast<int64_t>(
            graph_stats.register_census_outer_fast_retained_validated),
        static_cast<int64_t>(
            graph_stats.register_census_outer_fast_mutable_slot_count),
        static_cast<int64_t>(
            graph_stats.register_census_outer_fast_mutable_occurrence_count),
        static_cast<int64_t>(
            graph_stats.register_census_host_reemitted_node_count),
        static_cast<int64_t>(
            graph_stats.register_census_host_reemitted_kernel_count),
    };
}

void close_impl(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    int64_t encoded_plan_hash) {
    TORCH_CHECK(!RpuKernelGraph::has_active(),
                "qwen3vl multiview Z2 close must run outside Graph");
    const uint64_t plan_hash = decode_u64(encoded_plan_hash);
    Qwen3VlMultiviewZ2State& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.phase == CanaryPhase::Idle) return;
    require_owner_thread(s, "qwen3vl multiview Z2 close");
    const bool poison_recovery =
        s.phase == CanaryPhase::Poisoned && plan_hash == 0;
    TORCH_CHECK(s.vision_handle == vision_handle &&
                    s.text_handle == text_handle &&
                    s.expert_handle == expert_handle &&
                    (s.plan_hash == plan_hash || poison_recovery),
                "qwen3vl multiview Z2 close identity is stale");
    teardown_locked(s);
}

}  // namespace

int64_t rpu_qwen3vl_multiview_spm_z2_canary_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& position_ids) {
    return prepare_impl(
        vision_handle, text_handle, /*expert_handle=*/0, position_ids,
        /*rope_cos_il=*/std::nullopt, /*rope_sin_il=*/std::nullopt,
        kPersistentTopBytes, kExpectedTotalBytes, kExpectedHeadroomBytes,
        "qwen3vl multiview Z2 canary prepare");
}

at::Tensor rpu_qwen3vl_multiview_spm_z2_canary_forward(
    int64_t vision_handle,
    int64_t text_handle,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    int64_t encoded_plan_hash) {
    return forward_impl(
        vision_handle, text_handle, /*expert_handle=*/0, vision_inputs,
        vision_k_caches, vision_v_caches, text_hidden, text_k_caches,
        text_v_caches, position_ids,
        /*rope_cos_il=*/std::nullopt, /*rope_sin_il=*/std::nullopt,
        encoded_plan_hash);
}

std::vector<int64_t> rpu_qwen3vl_multiview_spm_z2_canary_stats(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t encoded_plan_hash) {
    return stats_impl(
        vision_handle, text_handle, /*expert_handle=*/0,
        encoded_plan_hash);
}

void rpu_qwen3vl_multiview_spm_z2_canary_close(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t encoded_plan_hash) {
    close_impl(
        vision_handle, text_handle, /*expert_handle=*/0,
        encoded_plan_hash);
}

int64_t rpu_lingbot2_multiview_spm_z2_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il) {
    TORCH_CHECK(expert_handle > 0,
                "LingBot2 multiview Z2 prepare requires a positive expert handle");
    static_assert(
        kPersistentTopBytes + kLingbot2ExpertPersistentBytes ==
        kLingbot2PersistentTopBytes);
    static_assert(
        kExpectedDynamicBytes + kLingbot2PersistentTopBytes ==
        kLingbot2ExpectedTotalBytes);
    static_assert(
        kExpectedBudgetBytes - kLingbot2ExpectedTotalBytes ==
        kLingbot2ExpectedHeadroomBytes);
    return prepare_impl(
        vision_handle, text_handle, expert_handle, position_ids,
        rope_cos_il, rope_sin_il,
        kLingbot2PersistentTopBytes, kLingbot2ExpectedTotalBytes,
        kLingbot2ExpectedHeadroomBytes,
        "LingBot2 multiview Z2 prepare");
}

void rpu_lingbot2_multiview_spm_z2_refresh_text_inputs(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t encoded_plan_hash) {
    refresh_text_inputs_impl(
        vision_handle, text_handle, expert_handle, position_ids,
        rope_cos_il, rope_sin_il, encoded_plan_hash);
}

void rpu_lingbot2_multiview_spm_z2_forward(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    at::TensorList vision_inputs,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t encoded_plan_hash) {
    (void)forward_impl(
        vision_handle, text_handle, expert_handle, vision_inputs,
        vision_k_caches, vision_v_caches, text_hidden, text_k_caches,
        text_v_caches, position_ids, rope_cos_il, rope_sin_il,
        encoded_plan_hash);
}

std::vector<int64_t> rpu_lingbot2_multiview_spm_z2_stats(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    int64_t encoded_plan_hash) {
    return stats_impl(
        vision_handle, text_handle, expert_handle, encoded_plan_hash);
}

void rpu_lingbot2_multiview_spm_z2_close(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t expert_handle,
    int64_t encoded_plan_hash) {
    close_impl(
        vision_handle, text_handle, expert_handle, encoded_plan_hash);
}
