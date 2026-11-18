// Board-free contract for typed DDR-register roles and explicit no-DDR proofs.
// This test validates pure metadata only: it constructs no Graph, Tensor, Queue,
// DMA, or Kernel and never touches /dev/rpu.

#include "graph/graph_runtime.h"

#include <cstdio>
#include <exception>
#include <functional>
#include <string>

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(cond, label)                                                    \
    do {                                                                       \
        if (cond) {                                                            \
            ++g_passed;                                                        \
            std::printf("  [PASS] %s\n", label);                              \
        } else {                                                               \
            ++g_failed;                                                        \
            std::fprintf(stderr, "  [FAIL] %s (%s:%d)\n",                    \
                         label, __FILE__, __LINE__);                            \
        }                                                                      \
    } while (0)

bool throws_with(const std::function<void()>& fn, const char* message) {
    try {
        fn();
    } catch (const std::exception& error) {
        return std::string(error.what()).find(message) != std::string::npos;
    }
    return false;
}

GraphKernelRegisterCensusStats one_kernel_stats(
        size_t typed, size_t no_ddr, size_t operands,
        size_t weights, size_t keys, size_t values, size_t semantic_inputs) {
    GraphKernelRegisterCensusStats stats;
    stats.node_count = 1;
    stats.kernel_count = 1;
    stats.typed_kernel_count = typed;
    stats.no_ddr_kernel_count = no_ddr;
    stats.operand_count = operands;
    stats.weight_operand_count = weights;
    stats.key_cache_operand_count = keys;
    stats.value_cache_operand_count = values;
    stats.semantic_input_operand_count = semantic_inputs;
    stats.ordered_node_kind_hash = UINT64_C(0x1001);
    stats.ordered_kernel_hash = UINT64_C(0x1002);
    return stats;
}

GraphKernelRegisterPolicy semantic_policy() {
    GraphKernelRegisterPolicy policy;
    policy.fingerprint = UINT64_C(0x53454d494e505554);  // SEMINPUT
    policy.name = "semantic_input_contract";

    GraphKernelRegisterRule rule;
    rule.kernel_name = "semantic_input_contract_kernel";
    rule.kind = GraphKernelRegisterRuleKind::TypedDdr;
    rule.operands = {GraphDdrRegisterAbi{
        GraphDdrRegisterRole::SemanticInput,
        GraphDdrRegisterAccess::Read,
        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
        /*reg_lo=*/10, /*reg_hi=*/11}};
    rule.expected_count = 1;
    policy.rules = {rule};

    policy.expected_node_count = 1;
    policy.expected_kernel_count = 1;
    policy.expected_typed_kernel_count = 1;
    policy.expected_operand_count = 1;
    policy.expected_semantic_input_operand_count = 1;
    policy.expected_ordered_node_kind_hash = UINT64_C(0x1001);
    policy.expected_ordered_kernel_hash = UINT64_C(0x1002);
    policy.phases = {GraphKernelRegisterPhaseExpectation{
        "semantic_input", one_kernel_stats(
            /*typed=*/1, /*no_ddr=*/0, /*operands=*/1,
            /*weights=*/0, /*keys=*/0, /*values=*/0,
            /*semantic_inputs=*/1)}};
    return policy;
}

GraphKernelRegisterPolicy w8_linear_policy() {
    GraphKernelRegisterPolicy policy;
    policy.fingerprint = UINT64_C(0x57384c494e454152);  // W8LINEAR
    policy.name = "w8_linear_contract";

    GraphKernelRegisterRule rule;
    rule.kernel_name = "w8_linear_contract_kernel";
    rule.kind = GraphKernelRegisterRuleKind::TypedDdr;
    rule.operands = {
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::Int8ModelWeight,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,
            /*reg_lo=*/2, /*reg_hi=*/3},
        GraphDdrRegisterAbi{
            GraphDdrRegisterRole::PerChannelScale,
            GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,
            /*reg_lo=*/16, /*reg_hi=*/17},
    };
    rule.expected_count = 1;
    policy.rules = {rule};

    policy.expected_node_count = 1;
    policy.expected_kernel_count = 1;
    policy.expected_typed_kernel_count = 1;
    policy.expected_operand_count = 2;
    // Both operands belong to the aggregate model-weight partition, while
    // their distinct ABI roles carry the exact dtype/semantic contract.
    policy.expected_weight_operand_count = 2;
    policy.expected_ordered_node_kind_hash = UINT64_C(0x3001);
    policy.expected_ordered_kernel_hash = UINT64_C(0x3002);
    auto stats = one_kernel_stats(
        /*typed=*/1, /*no_ddr=*/0, /*operands=*/2,
        /*weights=*/2, /*keys=*/0, /*values=*/0,
        /*semantic_inputs=*/0);
    stats.ordered_node_kind_hash = UINT64_C(0x3001);
    stats.ordered_kernel_hash = UINT64_C(0x3002);
    policy.phases = {GraphKernelRegisterPhaseExpectation{
        "w8_linear", stats}};
    return policy;
}

GraphKernelRegisterPolicy repeated_key_cache_policy() {
    GraphKernelRegisterPolicy policy;
    policy.fingerprint = UINT64_C(0x4b564d554c544932);  // KVMULTI2
    policy.name = "repeated_key_cache_contract";

    GraphKernelRegisterRule rule;
    rule.kernel_name = "repeated_key_cache_kernel";
    rule.kind = GraphKernelRegisterRuleKind::TypedDdr;
    rule.operands = {GraphDdrRegisterAbi{
        GraphDdrRegisterRole::KeyCache,
        GraphDdrRegisterAccess::Write,
        GraphDdrRegisterEncoding::DevAddrShift8LoHi,
        /*reg_lo=*/8, /*reg_hi=*/9}};
    rule.expected_count = 2;
    policy.rules = {rule};

    policy.expected_node_count = 2;
    policy.expected_kernel_count = 2;
    policy.expected_typed_kernel_count = 2;
    policy.expected_operand_count = 2;
    policy.expected_key_cache_operand_count = 2;
    policy.expected_ordered_node_kind_hash = UINT64_C(0x4001);
    policy.expected_ordered_kernel_hash = UINT64_C(0x4002);
    auto stats = one_kernel_stats(
        /*typed=*/2, /*no_ddr=*/0, /*operands=*/2,
        /*weights=*/0, /*keys=*/2, /*values=*/0,
        /*semantic_inputs=*/0);
    stats.node_count = 2;
    stats.kernel_count = 2;
    stats.ordered_node_kind_hash = UINT64_C(0x4001);
    stats.ordered_kernel_hash = UINT64_C(0x4002);
    policy.phases = {GraphKernelRegisterPhaseExpectation{
        "repeated_key_cache", stats}};
    return policy;
}

GraphKernelRegisterPolicy no_ddr_policy(
        GraphKernelRegisterRuleKind kind,
        GraphKernelNoDdrProof proof) {
    GraphKernelRegisterPolicy policy;
    policy.fingerprint = UINT64_C(0x52494e4741525055);  // RINGARPU
    policy.name = "ring_allreduce_contract";

    GraphKernelRegisterRule rule;
    // The name is diagnostic only. Admission is controlled by kind + proof.
    rule.kernel_name = "llm_all_reduce_residual";
    rule.kind = kind;
    rule.no_ddr_proof = proof;
    rule.expected_count = 1;
    policy.rules = {rule};

    policy.expected_node_count = 1;
    policy.expected_kernel_count = 1;
    policy.expected_no_ddr_kernel_count = 1;
    policy.expected_ordered_node_kind_hash = UINT64_C(0x2001);
    policy.expected_ordered_kernel_hash = UINT64_C(0x2002);
    auto stats = one_kernel_stats(
        /*typed=*/0, /*no_ddr=*/1, /*operands=*/0,
        /*weights=*/0, /*keys=*/0, /*values=*/0,
        /*semantic_inputs=*/0);
    stats.ordered_node_kind_hash = UINT64_C(0x2001);
    stats.ordered_kernel_hash = UINT64_C(0x2002);
    policy.phases = {GraphKernelRegisterPhaseExpectation{
        "ring_allreduce", stats}};
    return policy;
}

GraphKernelRegisterPolicy named_no_ddr_policy(
        const char* kernel_name,
        GraphKernelRegisterRuleKind kind,
        GraphKernelNoDdrProof proof) {
    auto policy = no_ddr_policy(kind, proof);
    policy.rules[0].kernel_name = kernel_name;
    return policy;
}

void validate_dtype(GraphDdrRegisterRole role, at::ScalarType dtype) {
    RpuKernelGraphContractTestAccess::validate_ddr_register_dtype(role, dtype);
}

void test_role_dtype_matrix() {
    for (const auto role : {
             GraphDdrRegisterRole::ModelWeight,
             GraphDdrRegisterRole::KeyCache,
             GraphDdrRegisterRole::ValueCache}) {
        bool half_ok = true;
        try {
            validate_dtype(role, at::kHalf);
        } catch (...) {
            half_ok = false;
        }
        EXPECT(half_ok, "legacy DDR roles retain strict FP16 admission");
        EXPECT(throws_with(
                   [&] { validate_dtype(role, at::kShort); },
                   "role/dtype mismatch"),
               "legacy DDR roles reject int16");
        EXPECT(throws_with(
                   [&] { validate_dtype(role, at::kInt); },
                   "role/dtype mismatch"),
               "legacy DDR roles reject int32");
    }

    bool semantic_ok = true;
    try {
        validate_dtype(GraphDdrRegisterRole::SemanticInput, at::kHalf);
        validate_dtype(GraphDdrRegisterRole::SemanticInput, at::kShort);
        validate_dtype(GraphDdrRegisterRole::SemanticInput, at::kInt);
    } catch (...) {
        semantic_ok = false;
    }
    EXPECT(semantic_ok, "SemanticInput admits FP16, int16, and int32");
    EXPECT(throws_with(
               [&] {
                   validate_dtype(
                       GraphDdrRegisterRole::SemanticInput, at::kFloat);
               },
               "role/dtype mismatch"),
           "SemanticInput rejects unrelated dtypes");

    bool w8_ok = true;
    try {
        validate_dtype(
            GraphDdrRegisterRole::Int8ModelWeight, at::kChar);
        validate_dtype(
            GraphDdrRegisterRole::PerChannelScale, at::kHalf);
    } catch (...) {
        w8_ok = false;
    }
    EXPECT(w8_ok,
           "W8 roles admit signed-int8 weight plus FP16 scale exactly");
    EXPECT(throws_with(
               [&] {
                   validate_dtype(
                       GraphDdrRegisterRole::Int8ModelWeight, at::kHalf);
               },
               "role/dtype mismatch"),
           "W8 weight role rejects an FP16 scale operand");
    EXPECT(throws_with(
               [&] {
                   validate_dtype(
                       GraphDdrRegisterRole::PerChannelScale, at::kChar);
               },
               "role/dtype mismatch"),
           "W8 scale role rejects a signed-int8 weight operand");
    EXPECT(throws_with(
               [&] {
                   validate_dtype(
                       GraphDdrRegisterRole::Int8ModelWeight, at::kByte);
               },
               "role/dtype mismatch"),
           "W8 weight role rejects unsigned-byte/W4 storage");

    bool fp8_ok = true;
    try {
        validate_dtype(GraphDdrRegisterRole::Fp8ModelWeight, at::kByte);
    } catch (...) {
        fp8_ok = false;
    }
    EXPECT(fp8_ok, "FP8 weight role admits raw uint8 storage");
    for (const auto role : {GraphDdrRegisterRole::PackedFp4ModelWeight,
                            GraphDdrRegisterRole::Fp8BlockScale}) {
        validate_dtype(role, at::kByte);
        EXPECT(throws_with([&] { validate_dtype(role, at::kHalf); }, "role/dtype mismatch"),
               "NVFP4 raw-byte roles reject FP16 operands");
        EXPECT(throws_with([&] { validate_dtype(role, at::kChar); }, "role/dtype mismatch"),
               "NVFP4 raw-byte roles reject signed INT8 operands");
    }
    EXPECT(throws_with(
               [&] {
                   validate_dtype(
                       GraphDdrRegisterRole::Fp8ModelWeight, at::kChar);
               },
               "role/dtype mismatch"),
           "FP8 weight role rejects signed-int8 storage");
    EXPECT(throws_with(
               [&] {
                   validate_dtype(
                       GraphDdrRegisterRole::Fp8ModelWeight, at::kHalf);
               },
               "role/dtype mismatch"),
           "FP8 weight role rejects FP16 storage");
}

void test_w8_role_and_abi_contract() {
    const auto policy = w8_linear_policy();
    bool exact_ok = true;
    try {
        RpuKernelGraphContractTestAccess::validate_kernel_register_policy(
            policy);
    } catch (...) {
        exact_ok = false;
    }
    EXPECT(exact_ok,
           "W8 weight 2/3 plus per-channel scale 16/17 validates exactly");

    auto fp32_scale = policy;
    fp32_scale.rules[0].operands[1].role = GraphDdrRegisterRole::Fp32PerChannelScale;
    validate_dtype(GraphDdrRegisterRole::Fp32PerChannelScale, at::kFloat);
    RpuKernelGraphContractTestAccess::validate_kernel_register_policy(fp32_scale);
    EXPECT(throws_with([&] { validate_dtype(GraphDdrRegisterRole::PerChannelScale, at::kFloat); },
                       "role/dtype mismatch"), "legacy scale role still rejects FP32");
    EXPECT(throws_with([&] { validate_dtype(GraphDdrRegisterRole::Fp32PerChannelScale, at::kHalf); },
                       "role/dtype mismatch"), "FP32 scale rejects FP16 owner substitution");
    fp32_scale.rules[0].operands[1].access = GraphDdrRegisterAccess::Write;
    EXPECT(throws_with([&] { RpuKernelGraphContractTestAccess::validate_kernel_register_policy(fp32_scale); },
                       "invalid/overlapping"), "FP32 scales are read-only weight operands");

    auto missing_scale = policy;
    missing_scale.rules[0].operands.pop_back();
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(missing_scale);
               },
               "per-rule totals mismatch"),
           "W8 policy rejects a missing scale operand");

    auto scale_write = policy;
    scale_write.rules[0].operands[1].access =
        GraphDdrRegisterAccess::Write;
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(scale_write);
               },
               "invalid/overlapping address register ABI"),
           "W8 scale ABI is read-only");

    auto overlapping = policy;
    overlapping.rules[0].operands[1].reg_lo = 2;
    overlapping.rules[0].operands[1].reg_hi = 3;
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(overlapping);
               },
               "invalid/overlapping address register ABI"),
           "W8 policy rejects overlapping weight/scale address registers");

    auto split_pair = policy;
    split_pair.rules[0].operands[1].reg_hi = 18;
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(split_pair);
               },
               "invalid/overlapping address register ABI"),
           "W8 policy rejects a non-adjacent scale address pair");
}

void test_grouped_uint4_roles_and_abi() {
    auto policy = w8_linear_policy();
    policy.name = "grouped_uint4_contract";
    policy.rules[0].kernel_name = "grouped_parallel_linear_w4a16_pgrp_m128n128k256";
    policy.rules[0].operands = {
        {GraphDdrRegisterRole::PackedUint4ModelWeight, GraphDdrRegisterAccess::Read,
         GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3},
        {GraphDdrRegisterRole::PerGroupScale, GraphDdrRegisterAccess::Read,
         GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21},
        {GraphDdrRegisterRole::NegativeZeroPoint, GraphDdrRegisterAccess::Read,
         GraphDdrRegisterEncoding::DevAddrShift8LoHi, 22, 23},
    };
    policy.expected_operand_count = policy.expected_weight_operand_count = 3;
    policy.phases[0].stats.operand_count = policy.phases[0].stats.weight_operand_count = 3;
    RpuKernelGraphContractTestAccess::validate_kernel_register_policy(policy);
    validate_dtype(GraphDdrRegisterRole::PackedUint4ModelWeight, at::kByte);
    for (const auto role : {GraphDdrRegisterRole::PerGroupScale,
                            GraphDdrRegisterRole::NegativeZeroPoint}) {
        validate_dtype(role, at::kHalf);
        EXPECT(throws_with([&] { validate_dtype(role, at::kByte); }, "role/dtype mismatch"),
               "grouped scale/negative-zero roles reject packed byte weights");
        EXPECT(throws_with([&] { validate_dtype(role, at::kFloat); }, "role/dtype mismatch"),
               "grouped scale/negative-zero roles require their exact FP16 ABI");
    }
    EXPECT(throws_with([&] { validate_dtype(GraphDdrRegisterRole::PackedUint4ModelWeight, at::kChar); },
                       "role/dtype mismatch"), "unsigned packed weights reject signed INT8");
    EXPECT(throws_with([&] { validate_dtype(GraphDdrRegisterRole::PackedUint4ModelWeight, at::kHalf); },
                       "role/dtype mismatch"), "unsigned packed weights reject a scale owner");
    for (size_t i = 0; i < 3; ++i) {
        auto writable = policy;
        writable.rules[0].operands[i].access = GraphDdrRegisterAccess::Write;
        EXPECT(throws_with([&] {
            RpuKernelGraphContractTestAccess::validate_kernel_register_policy(writable);
        }, "invalid/overlapping"), "every grouped UINT4 model operand is read-only");
    }
    auto missing_zero = policy;
    missing_zero.rules[0].operands.pop_back();
    EXPECT(throws_with([&] {
        RpuKernelGraphContractTestAccess::validate_kernel_register_policy(missing_zero);
    }, "per-rule totals mismatch"), "grouped UINT4 policy rejects a missing negative-zero owner");
    auto overlap = policy;
    overlap.rules[0].operands[2].reg_lo = 20;
    overlap.rules[0].operands[2].reg_hi = 21;
    EXPECT(throws_with([&] {
        RpuKernelGraphContractTestAccess::validate_kernel_register_policy(overlap);
    }, "invalid/overlapping"), "grouped UINT4 policy rejects aliased scale/negative-zero registers");
}

void test_semantic_count_partition() {
    const auto policy = semantic_policy();
    bool exact_ok = true;
    try {
        RpuKernelGraphContractTestAccess::validate_kernel_register_policy(
            policy);
    } catch (...) {
        exact_ok = false;
    }
    EXPECT(exact_ok, "semantic-input role count validates exactly");
    EXPECT(
        RpuKernelGraphContractTestAccess::kernel_register_policy_digest(
            policy) != 0,
        "semantic-input policy has a nonzero stable digest");

    auto semantic_write = policy;
    semantic_write.rules[0].operands[0].access =
        GraphDdrRegisterAccess::Write;
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(semantic_write);
               },
               "invalid/overlapping address register ABI"),
           "SemanticInput policy operands are read-only");

    auto bad_global = policy;
    bad_global.expected_semantic_input_operand_count = 0;
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(bad_global);
               },
               "role partition is inconsistent"),
           "global semantic-input count mismatch is rejected");

    auto bad_phase = policy;
    bad_phase.phases[0].stats.semantic_input_operand_count = 0;
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(bad_phase);
               },
               "internally inconsistent"),
           "phase semantic-input count mismatch is rejected");
}

void test_repeated_cache_operand_rule_count() {
    bool repeated_ok = true;
    try {
        RpuKernelGraphContractTestAccess::validate_kernel_register_policy(
            repeated_key_cache_policy());
    } catch (...) {
        repeated_ok = false;
    }
    EXPECT(repeated_ok,
           "one typed cache rule admits two ordered node occurrences");
}

void test_ring_allreduce_explicit_proof() {
    const auto explicit_policy = no_ddr_policy(
        GraphKernelRegisterRuleKind::ExplicitNoDdr,
        GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    bool explicit_ok = true;
    try {
        RpuKernelGraphContractTestAccess::validate_kernel_register_policy(
            explicit_policy);
    } catch (...) {
        explicit_ok = false;
    }
    EXPECT(explicit_ok,
           "RingAllReduceSpmResidual is an admitted explicit proof");

    const auto missing_proof = no_ddr_policy(
        GraphKernelRegisterRuleKind::ExplicitNoDdr,
        GraphKernelNoDdrProof::None);
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(missing_proof);
               },
               "explicit no-DDR rule needs a launcher proof"),
           "kernel name alone cannot replace the explicit launcher proof");

    const auto intrinsic_with_proof = no_ddr_policy(
        GraphKernelRegisterRuleKind::IntrinsicNoDdr,
        GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(intrinsic_with_proof);
               },
               "intrinsic no-DDR rule cannot carry a launcher proof"),
           "ring proof cannot be promoted to intrinsic name trust");
}

void test_dynamic_spm_explicit_proofs() {
    for (const auto& policy : {
             named_no_ddr_policy(
                 "fill", GraphKernelRegisterRuleKind::ExplicitNoDdr,
                 GraphKernelNoDdrProof::FillSpm),
             named_no_ddr_policy(
                 "slice_last_dim_axis",
                 GraphKernelRegisterRuleKind::ExplicitNoDdr,
                 GraphKernelNoDdrProof::SliceSpm),
             named_no_ddr_policy(
                 "slice_single_axis",
                 GraphKernelRegisterRuleKind::ExplicitNoDdr,
                 GraphKernelNoDdrProof::SliceSpm)}) {
        bool explicit_ok = true;
        try {
            RpuKernelGraphContractTestAccess::validate_kernel_register_policy(
                policy);
        } catch (...) {
            explicit_ok = false;
        }
        EXPECT(explicit_ok,
               "fill/slice SPM proof is admitted only as explicit no-DDR");
    }

    const auto intrinsic_fill = named_no_ddr_policy(
        "fill", GraphKernelRegisterRuleKind::IntrinsicNoDdr,
        GraphKernelNoDdrProof::FillSpm);
    EXPECT(throws_with(
               [&] {
                   RpuKernelGraphContractTestAccess::
                       validate_kernel_register_policy(intrinsic_fill);
               },
               "intrinsic no-DDR rule cannot carry a launcher proof"),
           "dynamic fill name cannot replace its explicit launcher proof");
}

}  // namespace

int main() {
    test_role_dtype_matrix();
    test_w8_role_and_abi_contract();
    test_grouped_uint4_roles_and_abi();
    test_semantic_count_partition();
    test_repeated_cache_operand_rule_count();
    test_ring_allreduce_explicit_proof();
    test_dynamic_spm_explicit_proofs();
    std::printf("\nPassed: %d\nFailed: %d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
