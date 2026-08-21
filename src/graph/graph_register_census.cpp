// graph_register_census.cpp — typed DDR address-register provenance gate.
#include "graph/graph_runtime.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

constexpr uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr uint64_t kFnvPrime = UINT64_C(1099511628211);

uint64_t fnv_byte(uint64_t hash, uint8_t value) {
    return (hash ^ value) * kFnvPrime;
}

uint64_t fnv_u64(uint64_t hash, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        hash = fnv_byte(hash, static_cast<uint8_t>(value >> shift));
    }
    return hash;
}

uint64_t fnv_string(uint64_t hash, const std::string& value) {
    hash = fnv_u64(hash, static_cast<uint64_t>(value.size()));
    for (unsigned char ch : value) hash = fnv_byte(hash, ch);
    return hash;
}

template <typename L, typename R>
bool shared_owner_aliases(const std::shared_ptr<L>& lhs,
                          const std::shared_ptr<R>& rhs) noexcept {
    return static_cast<const void*>(lhs.get()) ==
            static_cast<const void*>(rhs.get()) ||
        (!lhs.owner_before(rhs) && !rhs.owner_before(lhs));
}

template <typename L, typename R>
bool same_shared_owner_identity(const std::shared_ptr<L>& lhs,
                                const std::shared_ptr<R>& rhs) noexcept {
    return lhs != nullptr && rhs != nullptr &&
        static_cast<const void*>(lhs.get()) ==
            static_cast<const void*>(rhs.get()) &&
        !lhs.owner_before(rhs) && !rhs.owner_before(lhs);
}

bool ddr_register_role_accepts_dtype(
        GraphDdrRegisterRole role, at::ScalarType dtype) noexcept {
    switch (role) {
    case GraphDdrRegisterRole::ModelWeight:
    case GraphDdrRegisterRole::KeyCache:
    case GraphDdrRegisterRole::ValueCache:
        return dtype == at::kHalf;
    case GraphDdrRegisterRole::SemanticInput:
        return dtype == at::kHalf || dtype == at::kShort ||
            dtype == at::kInt;
    case GraphDdrRegisterRole::Int8ModelWeight:
        return dtype == at::kChar;
    case GraphDdrRegisterRole::PerChannelScale:
        return dtype == at::kHalf;
    }
    return false;
}

void validate_ddr_register_role_dtype(
        GraphDdrRegisterRole role, at::ScalarType dtype) {
    TORCH_CHECK(
        ddr_register_role_accepts_dtype(role, dtype),
        "typed DDR-register role/dtype mismatch: role=",
        static_cast<unsigned>(role), ", dtype=", dtype,
        "; ModelWeight/KeyCache/ValueCache/PerChannelScale require FP16, "
        "Int8ModelWeight requires signed int8, while SemanticInput "
        "additionally admits int16/int32");
}

uint64_t hash_stats(uint64_t hash,
                    const GraphKernelRegisterCensusStats& stats) {
    hash = fnv_u64(hash, stats.node_count);
    hash = fnv_u64(hash, stats.dma_count);
    hash = fnv_u64(hash, stats.barrier_count);
    hash = fnv_u64(hash, stats.kernel_count);
    hash = fnv_u64(hash, stats.typed_kernel_count);
    hash = fnv_u64(hash, stats.no_ddr_kernel_count);
    hash = fnv_u64(hash, stats.operand_count);
    hash = fnv_u64(hash, stats.weight_operand_count);
    hash = fnv_u64(hash, stats.key_cache_operand_count);
    hash = fnv_u64(hash, stats.value_cache_operand_count);
    hash = fnv_u64(hash, stats.ordered_node_kind_hash);
    hash = fnv_u64(hash, stats.ordered_kernel_hash);
    // Preserve the frozen schema-5 digest for policies whose semantic-input
    // partition is empty.  New policies enter a disjoint extension domain.
    if (stats.semantic_input_operand_count != 0) {
        hash = fnv_u64(hash, UINT64_C(0x53454d5354415453));  // SEMSTATS
        hash = fnv_u64(hash, stats.semantic_input_operand_count);
    }
    return hash;
}

bool same_stats(const GraphKernelRegisterCensusStats& lhs,
                const GraphKernelRegisterCensusStats& rhs) {
    return lhs.node_count == rhs.node_count &&
        lhs.dma_count == rhs.dma_count &&
        lhs.barrier_count == rhs.barrier_count &&
        lhs.kernel_count == rhs.kernel_count &&
        lhs.typed_kernel_count == rhs.typed_kernel_count &&
        lhs.no_ddr_kernel_count == rhs.no_ddr_kernel_count &&
        lhs.operand_count == rhs.operand_count &&
        lhs.weight_operand_count == rhs.weight_operand_count &&
        lhs.key_cache_operand_count == rhs.key_cache_operand_count &&
        lhs.value_cache_operand_count == rhs.value_cache_operand_count &&
        lhs.semantic_input_operand_count ==
            rhs.semantic_input_operand_count &&
        lhs.ordered_node_kind_hash == rhs.ordered_node_kind_hash &&
        lhs.ordered_kernel_hash == rhs.ordered_kernel_hash;
}

uint64_t stable_policy_digest(const GraphKernelRegisterPolicy& policy) {
    uint64_t hash = kFnvOffset;
    hash = fnv_u64(hash, UINT64_C(0x524547504f4c5631));  // REGPOLV1
    hash = fnv_u64(hash, policy.fingerprint);
    hash = fnv_string(hash, policy.name);
    hash = fnv_u64(hash, policy.expected_kernel_count);
    hash = fnv_u64(hash, policy.expected_typed_kernel_count);
    hash = fnv_u64(hash, policy.expected_no_ddr_kernel_count);
    hash = fnv_u64(hash, policy.expected_operand_count);
    hash = fnv_u64(hash, policy.expected_weight_operand_count);
    hash = fnv_u64(hash, policy.expected_key_cache_operand_count);
    hash = fnv_u64(hash, policy.expected_value_cache_operand_count);
    hash = fnv_u64(hash, policy.expected_node_count);
    hash = fnv_u64(hash, policy.expected_dma_count);
    hash = fnv_u64(hash, policy.expected_barrier_count);
    hash = fnv_u64(hash, policy.expected_ordered_node_kind_hash);
    hash = fnv_u64(hash, policy.expected_ordered_kernel_hash);
    hash = fnv_u64(hash, policy.rules.size());
    for (const auto& rule : policy.rules) {
        hash = fnv_byte(hash, rule.kernel_id.has_value() ? 1 : 0);
        hash = fnv_u64(hash, rule.kernel_id.has_value()
                               ? static_cast<uint64_t>(*rule.kernel_id)
                               : 0);
        hash = fnv_string(hash, rule.kernel_name);
        hash = fnv_byte(hash, static_cast<uint8_t>(rule.kind));
        hash = fnv_byte(hash, static_cast<uint8_t>(rule.no_ddr_proof));
        hash = fnv_u64(hash, rule.expected_count);
        hash = fnv_u64(hash, rule.operands.size());
        for (const auto& abi : rule.operands) {
            hash = fnv_byte(hash, static_cast<uint8_t>(abi.role));
            hash = fnv_byte(hash, static_cast<uint8_t>(abi.access));
            hash = fnv_byte(hash, static_cast<uint8_t>(abi.encoding));
            hash = fnv_u64(hash, abi.reg_lo);
            hash = fnv_u64(hash, abi.reg_hi);
        }
    }
    hash = fnv_u64(hash, policy.phases.size());
    for (const auto& phase : policy.phases) {
        hash = fnv_string(hash, phase.name);
        hash = hash_stats(hash, phase.stats);
    }
    // Zero keeps legacy policy digests byte-for-byte stable (notably GR00T
    // schema-5); a nonzero semantic partition is explicitly domain-separated.
    if (policy.expected_semantic_input_operand_count != 0) {
        hash = fnv_u64(hash, UINT64_C(0x53454d504f4c4943));  // SEMPOLIC
        hash = fnv_u64(hash,
                       policy.expected_semantic_input_operand_count);
    }
    return hash;
}

bool same_policy(const GraphKernelRegisterPolicy& lhs,
                 const GraphKernelRegisterPolicy& rhs) {
    if (lhs.fingerprint != rhs.fingerprint || lhs.name != rhs.name ||
        lhs.expected_kernel_count != rhs.expected_kernel_count ||
        lhs.expected_typed_kernel_count != rhs.expected_typed_kernel_count ||
        lhs.expected_no_ddr_kernel_count != rhs.expected_no_ddr_kernel_count ||
        lhs.expected_operand_count != rhs.expected_operand_count ||
        lhs.expected_weight_operand_count !=
            rhs.expected_weight_operand_count ||
        lhs.expected_key_cache_operand_count !=
            rhs.expected_key_cache_operand_count ||
        lhs.expected_value_cache_operand_count !=
            rhs.expected_value_cache_operand_count ||
        lhs.expected_semantic_input_operand_count !=
            rhs.expected_semantic_input_operand_count ||
        lhs.expected_node_count != rhs.expected_node_count ||
        lhs.expected_dma_count != rhs.expected_dma_count ||
        lhs.expected_barrier_count != rhs.expected_barrier_count ||
        lhs.expected_ordered_node_kind_hash !=
            rhs.expected_ordered_node_kind_hash ||
        lhs.expected_ordered_kernel_hash !=
            rhs.expected_ordered_kernel_hash ||
        lhs.rules.size() != rhs.rules.size() ||
        lhs.phases.size() != rhs.phases.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.rules.size(); ++i) {
        const auto& a = lhs.rules[i];
        const auto& b = rhs.rules[i];
        if (a.kernel_id != b.kernel_id || a.kernel_name != b.kernel_name ||
            a.kind != b.kind || a.no_ddr_proof != b.no_ddr_proof ||
            a.operands != b.operands ||
            a.expected_count != b.expected_count) {
            return false;
        }
    }
    for (size_t i = 0; i < lhs.phases.size(); ++i) {
        if (lhs.phases[i].name != rhs.phases[i].name ||
            !same_stats(lhs.phases[i].stats, rhs.phases[i].stats)) {
            return false;
        }
    }
    return true;
}

uint64_t tensor_topology_id(const at::Tensor& tensor,
                            size_t tensor_nbytes,
                            size_t storage_nbytes) {
    uint64_t hash = kFnvOffset;
    hash = fnv_byte(hash, static_cast<uint8_t>(tensor.scalar_type()));
    hash = fnv_byte(hash, static_cast<uint8_t>(tensor.layout()));
    hash = fnv_byte(hash, static_cast<uint8_t>(tensor.device().type()));
    hash = fnv_u64(hash, static_cast<uint64_t>(
        static_cast<int64_t>(tensor.device().index())));
    hash = fnv_u64(hash, static_cast<uint64_t>(tensor.storage_offset()));
    hash = fnv_u64(hash, static_cast<uint64_t>(tensor_nbytes));
    hash = fnv_u64(hash, static_cast<uint64_t>(storage_nbytes));
    hash = fnv_u64(hash, static_cast<uint64_t>(tensor.dim()));
    for (const int64_t size : tensor.sizes()) {
        hash = fnv_u64(hash, static_cast<uint64_t>(size));
    }
    for (const int64_t stride : tensor.strides()) {
        hash = fnv_u64(hash, static_cast<uint64_t>(stride));
    }
    return hash;
}

GraphDdrRegisterOccurrence snapshot_operand(
        const GraphDdrRegisterOperandSpec& spec) {
    TORCH_CHECK(spec.tensor.defined(),
                "typed DDR-register operand is undefined");
    TORCH_CHECK(spec.tensor.device().type() == c10::DeviceType::PrivateUse1,
                "typed DDR-register operand must be RPU-resident; got ",
                spec.tensor.device());
    validate_ddr_register_role_dtype(
        spec.abi.role, spec.tensor.scalar_type());
    TORCH_CHECK(spec.tensor.layout() == c10::Layout::Strided,
                "typed DDR-register operand must use strided layout");
    TORCH_CHECK(spec.abi.encoding ==
                    GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                "typed DDR-register operand has unsupported encoding");
    TORCH_CHECK(spec.abi.reg_hi == spec.abi.reg_lo + 1 &&
                    (spec.abi.reg_lo & 1U) == 0,
                "typed DDR-register address ABI must use an even low register "
                "and its adjacent high register; got ", spec.abi.reg_lo,
                "/", spec.abi.reg_hi);

    GraphDdrRegisterOccurrence occurrence;
    occurrence.abi = spec.abi;
    occurrence.tensor_owner = spec.tensor;
    occurrence.storage_owner = spec.tensor.storage();
    TORCH_CHECK(static_cast<bool>(occurrence.storage_owner),
                "typed DDR-register operand has no StorageImpl");
    occurrence.tensor_impl = spec.tensor.unsafeGetTensorImpl();
    occurrence.storage_impl =
        occurrence.storage_owner.unsafeGetStorageImpl();
    occurrence.data_ptr = spec.tensor.const_data_ptr();
    occurrence.storage_data_ptr = occurrence.storage_owner.data();
    occurrence.dtype = spec.tensor.scalar_type();
    occurrence.device = spec.tensor.device();
    occurrence.layout = spec.tensor.layout();
    occurrence.sizes.assign(spec.tensor.sizes().begin(),
                            spec.tensor.sizes().end());
    occurrence.strides.assign(spec.tensor.strides().begin(),
                              spec.tensor.strides().end());
    occurrence.storage_offset = spec.tensor.storage_offset();
    occurrence.tensor_nbytes = spec.tensor.nbytes();
    occurrence.storage_nbytes = occurrence.storage_owner.nbytes();
    occurrence.tensor_topology_id = tensor_topology_id(
        spec.tensor, occurrence.tensor_nbytes, occurrence.storage_nbytes);

    TORCH_CHECK(occurrence.tensor_impl != nullptr &&
                    occurrence.storage_impl != nullptr &&
                    occurrence.data_ptr != nullptr &&
                    occurrence.storage_data_ptr != nullptr,
                "typed DDR-register operand has incomplete storage provenance");
    occurrence.dev_addr = rhino_lkn::RpuGetDevAddr(
        const_cast<void*>(occurrence.data_ptr));
    TORCH_CHECK(occurrence.dev_addr != 0,
                "typed DDR-register operand has no RPU device address");
    TORCH_CHECK((occurrence.dev_addr & UINT64_C(0xff)) == 0,
                "typed DDR-register device address must be 256-byte aligned; "
                "got 0x", std::hex, occurrence.dev_addr, std::dec);
    TORCH_CHECK((occurrence.dev_addr >> 8) <=
                    std::numeric_limits<uint32_t>::max(),
                "typed DDR-register device address cannot be represented by "
                "DevAddrShift8LoHi; got 0x", std::hex, occurrence.dev_addr,
                std::dec);
    return occurrence;
}

void check_same_occurrence(const GraphDdrRegisterOccurrence& built,
                           const GraphDdrRegisterOccurrence& live,
                           const char* context) {
    TORCH_CHECK(
        built.abi == live.abi &&
            built.tensor_impl == live.tensor_impl &&
            built.storage_impl == live.storage_impl &&
            built.data_ptr == live.data_ptr &&
            built.storage_data_ptr == live.storage_data_ptr &&
            built.dtype == live.dtype && built.device == live.device &&
            built.layout == live.layout && built.sizes == live.sizes &&
            built.strides == live.strides &&
            built.storage_offset == live.storage_offset &&
            built.tensor_nbytes == live.tensor_nbytes &&
            built.storage_nbytes == live.storage_nbytes &&
            built.dev_addr == live.dev_addr &&
            built.tensor_topology_id == live.tensor_topology_id &&
            built.storage_owner.unsafeGetStorageImpl() ==
                built.storage_impl,
        context,
        ": TensorImpl/StorageImpl/data/layout/address provenance drifted");
}

void check_same_retained_owner_snapshot(
        const GraphDdrRegisterOccurrence& first,
        const GraphDdrRegisterOccurrence& occurrence) {
    TORCH_CHECK(
        first.tensor_impl == occurrence.tensor_impl &&
            first.storage_impl == occurrence.storage_impl &&
            first.data_ptr == occurrence.data_ptr &&
            first.storage_data_ptr == occurrence.storage_data_ptr &&
            first.dtype == occurrence.dtype &&
            first.device == occurrence.device &&
            first.layout == occurrence.layout &&
            first.sizes == occurrence.sizes &&
            first.strides == occurrence.strides &&
            first.storage_offset == occurrence.storage_offset &&
            first.tensor_nbytes == occurrence.tensor_nbytes &&
            first.storage_nbytes == occurrence.storage_nbytes &&
            first.dev_addr == occurrence.dev_addr &&
            first.tensor_topology_id == occurrence.tensor_topology_id &&
            first.tensor_owner.defined() &&
            occurrence.tensor_owner.defined() &&
            first.tensor_owner.unsafeGetTensorImpl() == first.tensor_impl &&
            occurrence.tensor_owner.unsafeGetTensorImpl() ==
                occurrence.tensor_impl &&
            static_cast<bool>(first.storage_owner) &&
            static_cast<bool>(occurrence.storage_owner) &&
            first.storage_owner.unsafeGetStorageImpl() == first.storage_impl &&
            occurrence.storage_owner.unsafeGetStorageImpl() ==
                occurrence.storage_impl,
        "typed DDR-register retained occurrences sharing an owner have "
        "different BUILD provenance");
}

size_t cfg_index_for_reg(uint16_t reg) {
    if (reg <= 67) return reg / 2;
    if (reg >= 128 && reg <= 255) return 34 + (reg - 128) / 2;
    if (reg >= 4096) return 98 + (reg - 4096) / 2;
    TORCH_CHECK(false, "typed DDR-register ABI uses reserved register ", reg);
}

uint16_t read_kernel_reg(::rhino_lkn::Kernel_t& kernel, uint16_t reg) {
    uint64_t* args = nullptr;
    size_t args_bytes = 0;
    kernel.get_kernel_args_instr(&args, &args_bytes, nullptr, nullptr);
    const size_t cfg_index = cfg_index_for_reg(reg);
    TORCH_CHECK(args != nullptr &&
                    args_bytes >= (cfg_index + 1) * sizeof(uint64_t),
                "typed DDR-register readback cannot access register ", reg,
                " (args_bytes=", args_bytes, ")");
    const uint32_t cfg_data = static_cast<uint32_t>(args[cfg_index] >> 32);
    return (reg & 1U) != 0
        ? static_cast<uint16_t>(cfg_data >> 16)
        : static_cast<uint16_t>(cfg_data);
}

void check_register_pairs(
        ::rhino_lkn::Kernel_t& kernel,
        const std::vector<GraphKernelRegisterWriter::RegisterPair>& pairs,
        const char* context) {
    for (const auto& pair : pairs) {
        const uint16_t expected_lo =
            static_cast<uint16_t>(pair.encoded_addr);
        const uint16_t expected_hi =
            static_cast<uint16_t>(pair.encoded_addr >> 16);
        TORCH_CHECK(read_kernel_reg(kernel, pair.reg_lo) == expected_lo &&
                        read_kernel_reg(kernel, pair.reg_hi) == expected_hi,
                    context, ": address register pair ", pair.reg_lo, "/",
                    pair.reg_hi, " no longer contains the staged address");
    }
}

GraphKernelRegisterCensusStats stats_delta(
        const GraphKernelRegisterCensusStats& end,
        const GraphKernelRegisterCensusStats& begin) {
    GraphKernelRegisterCensusStats delta;
    delta.node_count = end.node_count - begin.node_count;
    delta.dma_count = end.dma_count - begin.dma_count;
    delta.barrier_count = end.barrier_count - begin.barrier_count;
    delta.kernel_count = end.kernel_count - begin.kernel_count;
    delta.typed_kernel_count =
        end.typed_kernel_count - begin.typed_kernel_count;
    delta.no_ddr_kernel_count =
        end.no_ddr_kernel_count - begin.no_ddr_kernel_count;
    delta.operand_count = end.operand_count - begin.operand_count;
    delta.weight_operand_count =
        end.weight_operand_count - begin.weight_operand_count;
    delta.key_cache_operand_count =
        end.key_cache_operand_count - begin.key_cache_operand_count;
    delta.value_cache_operand_count =
        end.value_cache_operand_count - begin.value_cache_operand_count;
    delta.semantic_input_operand_count =
        end.semantic_input_operand_count -
        begin.semantic_input_operand_count;
    // Phase hashes are initialized independently and maintained by the runtime;
    // subtraction is intentionally not meaningful for FNV streams.
    return delta;
}

GraphKernelRegisterCensusStats policy_total_stats(
        const GraphKernelRegisterPolicy& policy) {
    GraphKernelRegisterCensusStats stats;
    stats.node_count = policy.expected_node_count;
    stats.dma_count = policy.expected_dma_count;
    stats.barrier_count = policy.expected_barrier_count;
    stats.kernel_count = policy.expected_kernel_count;
    stats.typed_kernel_count = policy.expected_typed_kernel_count;
    stats.no_ddr_kernel_count = policy.expected_no_ddr_kernel_count;
    stats.operand_count = policy.expected_operand_count;
    stats.weight_operand_count = policy.expected_weight_operand_count;
    stats.key_cache_operand_count = policy.expected_key_cache_operand_count;
    stats.value_cache_operand_count = policy.expected_value_cache_operand_count;
    stats.semantic_input_operand_count =
        policy.expected_semantic_input_operand_count;
    stats.ordered_node_kind_hash = policy.expected_ordered_node_kind_hash;
    stats.ordered_kernel_hash = policy.expected_ordered_kernel_hash;
    return stats;
}

void check_stats_counts(const GraphKernelRegisterCensusStats& actual,
                        const GraphKernelRegisterCensusStats& expected,
                        const std::string& label) {
    TORCH_CHECK(
        actual.node_count == expected.node_count &&
            actual.dma_count == expected.dma_count &&
            actual.barrier_count == expected.barrier_count &&
            actual.kernel_count == expected.kernel_count &&
            actual.typed_kernel_count == expected.typed_kernel_count &&
            actual.no_ddr_kernel_count == expected.no_ddr_kernel_count &&
            actual.operand_count == expected.operand_count &&
            actual.weight_operand_count == expected.weight_operand_count &&
            actual.key_cache_operand_count ==
                expected.key_cache_operand_count &&
            actual.value_cache_operand_count ==
                expected.value_cache_operand_count &&
            actual.semantic_input_operand_count ==
                expected.semantic_input_operand_count,
        "typed DDR-register census ", label,
        " count partition mismatch; actual(node/dma/barrier/kernel/typed/"
        "no_ddr/operands/weight/key/value/semantic_input)=",
        actual.node_count, "/", actual.dma_count, "/",
        actual.barrier_count, "/", actual.kernel_count, "/",
        actual.typed_kernel_count, "/", actual.no_ddr_kernel_count, "/",
        actual.operand_count, "/", actual.weight_operand_count, "/",
        actual.key_cache_operand_count, "/",
        actual.value_cache_operand_count, "/",
        actual.semantic_input_operand_count,
        " expected=", expected.node_count, "/", expected.dma_count, "/",
        expected.barrier_count, "/", expected.kernel_count, "/",
        expected.typed_kernel_count, "/", expected.no_ddr_kernel_count,
        "/", expected.operand_count, "/",
        expected.weight_operand_count, "/",
        expected.key_cache_operand_count, "/",
        expected.value_cache_operand_count, "/",
        expected.semantic_input_operand_count);
}

void validate_policy(const GraphKernelRegisterPolicy& policy) {
    TORCH_CHECK(policy.fingerprint != 0 && !policy.name.empty(),
                "typed DDR-register policy needs a stable identity");
    TORCH_CHECK(policy.expected_kernel_count > 0 &&
                    policy.expected_node_count ==
                        policy.expected_kernel_count +
                            policy.expected_dma_count +
                            policy.expected_barrier_count &&
                    policy.expected_typed_kernel_count +
                        policy.expected_no_ddr_kernel_count ==
                        policy.expected_kernel_count &&
                    policy.expected_ordered_node_kind_hash != 0 &&
                    policy.expected_ordered_kernel_hash != 0,
                "typed DDR-register policy has an incomplete exact census");
    TORCH_CHECK(policy.expected_weight_operand_count +
                    policy.expected_key_cache_operand_count +
                    policy.expected_value_cache_operand_count +
                    policy.expected_semantic_input_operand_count ==
                    policy.expected_operand_count,
                "typed DDR-register policy role partition is inconsistent");

    size_t rule_kernel_total = 0;
    size_t rule_typed_total = 0;
    size_t rule_operand_total = 0;
    size_t rule_weight_total = 0;
    size_t rule_key_total = 0;
    size_t rule_value_total = 0;
    size_t rule_semantic_input_total = 0;
    std::unordered_set<uint64_t> ids;
    std::unordered_set<std::string> names;
    for (const auto& rule : policy.rules) {
        TORCH_CHECK(
            rule.kind == GraphKernelRegisterRuleKind::TypedDdr ||
                rule.kind == GraphKernelRegisterRuleKind::IntrinsicNoDdr ||
                rule.kind == GraphKernelRegisterRuleKind::ExplicitNoDdr,
            "typed DDR-register policy contains an invalid rule kind");
        TORCH_CHECK(rule.expected_count > 0 &&
                        (rule.kernel_id.has_value() !=
                         !rule.kernel_name.empty()),
                    "typed DDR-register policy rule needs exactly one kernel "
                    "identity and a nonzero count");
        if (rule.kernel_id.has_value()) {
            TORCH_CHECK(ids.insert(static_cast<uint64_t>(*rule.kernel_id)).second,
                        "typed DDR-register policy duplicates KernelId ",
                        static_cast<int>(*rule.kernel_id));
        } else {
            TORCH_CHECK(names.insert(rule.kernel_name).second,
                        "typed DDR-register policy duplicates kernel name '",
                        rule.kernel_name, "'");
        }
        rule_kernel_total += rule.expected_count;
        if (rule.kind == GraphKernelRegisterRuleKind::TypedDdr) {
            TORCH_CHECK(!rule.operands.empty() &&
                            rule.no_ddr_proof == GraphKernelNoDdrProof::None,
                        "typed DDR rule has no ABI operands or a no-DDR proof");
            rule_typed_total += rule.expected_count;
            std::unordered_set<uint16_t> registers;
            for (const auto& abi : rule.operands) {
                TORCH_CHECK(
                                (abi.role ==
                                     GraphDdrRegisterRole::ModelWeight ||
                                 abi.role == GraphDdrRegisterRole::KeyCache ||
                                 abi.role ==
                                     GraphDdrRegisterRole::ValueCache ||
                                 abi.role ==
                                     GraphDdrRegisterRole::SemanticInput ||
                                 abi.role ==
                                     GraphDdrRegisterRole::Int8ModelWeight ||
                                 abi.role ==
                                     GraphDdrRegisterRole::PerChannelScale) &&
                                (abi.access ==
                                     GraphDdrRegisterAccess::Read ||
                                 abi.access ==
                                     GraphDdrRegisterAccess::Write) &&
                                ((abi.role !=
                                      GraphDdrRegisterRole::SemanticInput &&
                                  abi.role !=
                                      GraphDdrRegisterRole::Int8ModelWeight &&
                                  abi.role !=
                                      GraphDdrRegisterRole::PerChannelScale) ||
                                 abi.access ==
                                     GraphDdrRegisterAccess::Read) &&
                                abi.encoding ==
                                GraphDdrRegisterEncoding::DevAddrShift8LoHi &&
                                abi.reg_hi == abi.reg_lo + 1 &&
                                (abi.reg_lo & 1U) == 0 &&
                                registers.insert(abi.reg_lo).second &&
                                registers.insert(abi.reg_hi).second,
                            "typed DDR rule has an invalid/overlapping address "
                            "register ABI");
                rule_operand_total += rule.expected_count;
                switch (abi.role) {
                case GraphDdrRegisterRole::ModelWeight:
                case GraphDdrRegisterRole::Int8ModelWeight:
                case GraphDdrRegisterRole::PerChannelScale:
                    rule_weight_total += rule.expected_count;
                    break;
                case GraphDdrRegisterRole::KeyCache:
                    rule_key_total += rule.expected_count;
                    break;
                case GraphDdrRegisterRole::ValueCache:
                    rule_value_total += rule.expected_count;
                    break;
                case GraphDdrRegisterRole::SemanticInput:
                    rule_semantic_input_total += rule.expected_count;
                    break;
                }
            }
        } else {
            TORCH_CHECK(rule.operands.empty(),
                        "no-DDR rule unexpectedly declares DDR operands");
            if (rule.kind == GraphKernelRegisterRuleKind::IntrinsicNoDdr) {
                TORCH_CHECK(rule.no_ddr_proof == GraphKernelNoDdrProof::None,
                            "intrinsic no-DDR rule cannot carry a launcher proof");
            } else {
                TORCH_CHECK(
                    rule.no_ddr_proof ==
                            GraphKernelNoDdrProof::AllReduceSpmResidualV2 ||
                        rule.no_ddr_proof ==
                            GraphKernelNoDdrProof::AllGatherSpm ||
                        rule.no_ddr_proof ==
                            GraphKernelNoDdrProof::
                                LegacyAllReduceSpmResidual ||
                        rule.no_ddr_proof ==
                            GraphKernelNoDdrProof::FillSpm ||
                        rule.no_ddr_proof ==
                            GraphKernelNoDdrProof::SliceSpm ||
                        rule.no_ddr_proof ==
                            GraphKernelNoDdrProof::RingAllReduceSpmResidual,
                            "explicit no-DDR rule needs a launcher proof");
            }
        }
    }
    TORCH_CHECK(rule_kernel_total == policy.expected_kernel_count &&
                    rule_typed_total == policy.expected_typed_kernel_count &&
                    rule_operand_total == policy.expected_operand_count &&
                    rule_weight_total ==
                        policy.expected_weight_operand_count &&
                    rule_key_total ==
                        policy.expected_key_cache_operand_count &&
                    rule_value_total ==
                        policy.expected_value_cache_operand_count &&
                    rule_semantic_input_total ==
                        policy.expected_semantic_input_operand_count,
                "typed DDR-register policy per-rule totals mismatch");
    TORCH_CHECK(!policy.phases.empty(),
                "typed DDR-register policy needs explicit phase partitions");
    GraphKernelRegisterCensusStats phase_totals;
    std::unordered_set<std::string> phase_names;
    for (const auto& phase : policy.phases) {
        const auto& stats = phase.stats;
        TORCH_CHECK(!phase.name.empty() &&
                        phase_names.insert(phase.name).second,
                    "typed DDR-register policy phase names must be nonempty "
                    "and unique");
        TORCH_CHECK(stats.node_count == stats.kernel_count + stats.dma_count +
                                             stats.barrier_count &&
                        stats.typed_kernel_count +
                                stats.no_ddr_kernel_count ==
                            stats.kernel_count &&
                        stats.weight_operand_count +
                        stats.key_cache_operand_count +
                                stats.value_cache_operand_count +
                                stats.semantic_input_operand_count ==
                            stats.operand_count &&
                        stats.ordered_node_kind_hash != 0 &&
                        stats.ordered_kernel_hash != 0,
                    "typed DDR-register policy phase '", phase.name,
                    "' is internally inconsistent");
        phase_totals.node_count += stats.node_count;
        phase_totals.dma_count += stats.dma_count;
        phase_totals.barrier_count += stats.barrier_count;
        phase_totals.kernel_count += stats.kernel_count;
        phase_totals.typed_kernel_count += stats.typed_kernel_count;
        phase_totals.no_ddr_kernel_count += stats.no_ddr_kernel_count;
        phase_totals.operand_count += stats.operand_count;
        phase_totals.weight_operand_count += stats.weight_operand_count;
        phase_totals.key_cache_operand_count +=
            stats.key_cache_operand_count;
        phase_totals.value_cache_operand_count +=
            stats.value_cache_operand_count;
        phase_totals.semantic_input_operand_count +=
            stats.semantic_input_operand_count;
    }
    TORCH_CHECK(
        phase_totals.node_count == policy.expected_node_count &&
            phase_totals.dma_count == policy.expected_dma_count &&
            phase_totals.barrier_count == policy.expected_barrier_count &&
            phase_totals.kernel_count == policy.expected_kernel_count &&
            phase_totals.typed_kernel_count ==
                policy.expected_typed_kernel_count &&
            phase_totals.no_ddr_kernel_count ==
                policy.expected_no_ddr_kernel_count &&
            phase_totals.operand_count == policy.expected_operand_count &&
            phase_totals.weight_operand_count ==
                policy.expected_weight_operand_count &&
            phase_totals.key_cache_operand_count ==
                policy.expected_key_cache_operand_count &&
            phase_totals.value_cache_operand_count ==
                policy.expected_value_cache_operand_count &&
            phase_totals.semantic_input_operand_count ==
                policy.expected_semantic_input_operand_count,
        "typed DDR-register policy phase totals do not match global totals");
    TORCH_CHECK(stable_policy_digest(policy) != 0,
                "typed DDR-register policy digest must be nonzero");
}

}  // namespace

void RpuKernelGraphContractTestAccess::validate_ddr_register_dtype(
        GraphDdrRegisterRole role, at::ScalarType dtype) {
    validate_ddr_register_role_dtype(role, dtype);
}

void RpuKernelGraphContractTestAccess::validate_kernel_register_policy(
        const GraphKernelRegisterPolicy& policy) {
    validate_policy(policy);
}

uint64_t RpuKernelGraphContractTestAccess::kernel_register_policy_digest(
        const GraphKernelRegisterPolicy& policy) {
    return stable_policy_digest(policy);
}

RpuKernelGraph::OuterFastTensorSnapshot
RpuKernelGraph::snapshot_kernel_register_outer_fast_tensor(
        const at::Tensor& tensor) {
    TORCH_CHECK(tensor.defined(),
                "outer fast replay tensor owner is undefined");
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
                "outer fast replay tensor owner must be RPU-resident; got ",
                tensor.device());
    TORCH_CHECK(tensor.layout() == c10::Layout::Strided,
                "outer fast replay tensor owner must use strided layout");

    OuterFastTensorSnapshot snapshot;
    snapshot.tensor_owner = tensor;
    snapshot.storage_owner = tensor.storage();
    TORCH_CHECK(static_cast<bool>(snapshot.storage_owner),
                "outer fast replay tensor owner has no StorageImpl");
    snapshot.tensor_impl = tensor.unsafeGetTensorImpl();
    snapshot.storage_impl = snapshot.storage_owner.unsafeGetStorageImpl();
    snapshot.data_ptr = tensor.const_data_ptr();
    snapshot.storage_data_ptr = snapshot.storage_owner.data();
    snapshot.dtype = tensor.scalar_type();
    snapshot.device = tensor.device();
    snapshot.layout = tensor.layout();
    snapshot.sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
    snapshot.strides.assign(tensor.strides().begin(), tensor.strides().end());
    snapshot.storage_offset = tensor.storage_offset();
    snapshot.tensor_nbytes = tensor.nbytes();
    snapshot.storage_nbytes = snapshot.storage_owner.nbytes();
    snapshot.topology_id = tensor_topology_id(
        tensor, snapshot.tensor_nbytes, snapshot.storage_nbytes);
    TORCH_CHECK(snapshot.tensor_impl != nullptr &&
                    snapshot.storage_impl != nullptr &&
                    snapshot.data_ptr != nullptr &&
                    snapshot.storage_data_ptr != nullptr &&
                    snapshot.tensor_nbytes > 0,
                "outer fast replay tensor owner has incomplete provenance");
    snapshot.dev_addr = rhino_lkn::RpuGetDevAddr(
        const_cast<void*>(snapshot.data_ptr));
    TORCH_CHECK(snapshot.dev_addr != 0,
                "outer fast replay tensor owner has no RPU device address");
    return snapshot;
}

void RpuKernelGraph::check_kernel_register_outer_fast_tensor_live(
        const OuterFastTensorSnapshot& snapshot,
        const char* context) {
    const at::Tensor& tensor = snapshot.tensor_owner;
    TORCH_CHECK(tensor.defined() &&
                    tensor.unsafeGetTensorImpl() == snapshot.tensor_impl &&
                    snapshot.storage_owner.unsafeGetStorageImpl() ==
                        snapshot.storage_impl &&
                    tensor.storage().unsafeGetStorageImpl() ==
                        snapshot.storage_impl &&
                    tensor.const_data_ptr() == snapshot.data_ptr &&
                    snapshot.storage_owner.data() == snapshot.storage_data_ptr &&
                    tensor.scalar_type() == snapshot.dtype &&
                    tensor.device() == snapshot.device &&
                    tensor.layout() == snapshot.layout &&
                    tensor.storage_offset() == snapshot.storage_offset &&
                    tensor.nbytes() == snapshot.tensor_nbytes &&
                    snapshot.storage_owner.nbytes() == snapshot.storage_nbytes &&
                    tensor.dim() == static_cast<int64_t>(snapshot.sizes.size()) &&
                    std::equal(tensor.sizes().begin(), tensor.sizes().end(),
                               snapshot.sizes.begin()) &&
                    std::equal(tensor.strides().begin(), tensor.strides().end(),
                               snapshot.strides.begin()) &&
                    tensor_topology_id(tensor, snapshot.tensor_nbytes,
                                       snapshot.storage_nbytes) ==
                        snapshot.topology_id &&
                    rhino_lkn::RpuGetDevAddr(
                        const_cast<void*>(snapshot.data_ptr)) ==
                        snapshot.dev_addr,
                context,
                ": TensorImpl/StorageImpl/data/layout/address provenance "
                "drifted");
}

void RpuKernelGraph::check_kernel_register_outer_fast_tensor_exact(
        const OuterFastTensorSnapshot& built,
        const OuterFastTensorSnapshot& live,
        const char* context) {
    check_kernel_register_outer_fast_tensor_live(built, context);
    check_kernel_register_outer_fast_tensor_live(live, context);
    TORCH_CHECK(built.tensor_impl == live.tensor_impl &&
                    built.storage_impl == live.storage_impl &&
                    built.data_ptr == live.data_ptr &&
                    built.storage_data_ptr == live.storage_data_ptr &&
                    built.dtype == live.dtype && built.device == live.device &&
                    built.layout == live.layout && built.sizes == live.sizes &&
                    built.strides == live.strides &&
                    built.storage_offset == live.storage_offset &&
                    built.tensor_nbytes == live.tensor_nbytes &&
                    built.storage_nbytes == live.storage_nbytes &&
                    built.dev_addr == live.dev_addr &&
                    built.topology_id == live.topology_id,
                context, ": current tensor owner differs from BUILD");
}

GraphKernelRegisterOuterFastTicket::GraphKernelRegisterOuterFastTicket(
        GraphKernelRegisterOuterFastTicket&& other) noexcept
    : graph_(other.graph_), nonce_(other.nonce_) {
    other.release();
}

GraphKernelRegisterOuterFastTicket::~GraphKernelRegisterOuterFastTicket() {
    if (graph_ != nullptr) {
        graph_->cancel_kernel_register_outer_fast_ticket(nonce_);
    }
}

void GraphKernelRegisterCensusGuard::stage_outer_fast_physical_digest(
        uint64_t digest) {
    TORCH_CHECK(!completed_,
                "outer fast physical binding after census completion");
    try {
        graph_->stage_kernel_register_outer_fast_physical_digest(digest);
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

void GraphKernelRegisterCensusGuard::stage_outer_fast_component(
        std::vector<std::pair<std::shared_ptr<const uint64_t>, uint64_t>>
            owner_stamps,
        std::vector<at::Tensor> key_cache,
        std::vector<at::Tensor> value_cache) {
    TORCH_CHECK(!completed_,
                "outer fast component binding after census completion");
    try {
        graph_->stage_kernel_register_outer_fast_component(
            std::move(owner_stamps), std::move(key_cache),
            std::move(value_cache));
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

void GraphKernelRegisterCensusGuard::stage_outer_fast_invocation_authority(
        std::shared_ptr<const uint64_t> prepared_owner,
        std::shared_ptr<uint64_t> consumed_owner,
        uint64_t expected_generation) {
    TORCH_CHECK(!completed_,
                "outer fast invocation authority after census completion");
    try {
        graph_->stage_kernel_register_outer_fast_invocation_authority(
            std::move(prepared_owner), std::move(consumed_owner),
            expected_generation);
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

void GraphKernelRegisterCensusGuard::stage_outer_fast_invocation_visibility(
        const at::Tensor& tensor) {
    TORCH_CHECK(!completed_,
                "outer fast invocation visibility after census completion");
    try {
        graph_->stage_kernel_register_outer_fast_invocation_visibility(tensor);
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

void GraphKernelRegisterCensusGuard::stage_fixed_dma_burst(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count) {
    TORCH_CHECK(!completed_,
                "outer fast fixed DMA burst after census completion");
    try {
        graph_->stage_kernel_register_fixed_dma_burst(
            tensor, side, byte_begin, byte_count,
            expected_occurrence_count);
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

void GraphKernelRegisterCensusGuard::stage_mutable_dma_burst(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count) {
    TORCH_CHECK(!completed_,
                "outer fast mutable DMA burst after census completion");
    try {
        graph_->stage_kernel_register_mutable_dma_burst(
            live_base, tensor, side, byte_begin, byte_count,
            expected_occurrence_count);
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

void GraphKernelRegisterCensusGuard::bind_fast_fixed_dma(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count) {
    TORCH_CHECK(!completed_,
                "outer fast fixed DMA binding after census completion");
    try {
        graph_->bind_kernel_register_outer_fast_fixed_dma(
            tensor, side, byte_begin, byte_count);
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

void GraphKernelRegisterCensusGuard::bind_fast_mutable_dma(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count) {
    TORCH_CHECK(!completed_,
                "outer fast mutable DMA binding after census completion");
    try {
        graph_->bind_kernel_register_outer_fast_mutable_dma(
            live_base, tensor, side, byte_begin, byte_count);
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

GraphKernelRegisterOuterFastTicket
GraphKernelRegisterCensusGuard::validate_outer_fast_replay() {
    TORCH_CHECK(!completed_,
                "outer fast validation after census completion");
    try {
        return graph_->validate_kernel_register_outer_fast_replay();
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

void GraphKernelRegisterCensusGuard::commit_outer_fast_replay(
        GraphKernelRegisterOuterFastTicket&& ticket) {
    TORCH_CHECK(!completed_, "outer fast replay committed twice");
    try {
        graph_->commit_kernel_register_outer_fast_replay(std::move(ticket));
        completed_ = true;
    } catch (...) {
        graph_->cancel_kernel_register_census();
        throw;
    }
}

GraphKernelRegisterWriter::GraphKernelRegisterWriter(
        GraphKernelRegisterWriter&& other) noexcept
    : graph_(other.graph_),
      ticket_(other.ticket_),
      pairs_(std::move(other.pairs_)),
      written_(other.written_) {
    other.graph_ = nullptr;
    other.ticket_ = 0;
    other.written_ = true;
}

void GraphKernelRegisterWriter::write(::rhino_lkn::Kernel_t& kernel) {
    TORCH_CHECK(!written_,
                "typed DDR-register writer is move-only and single-use");
    if (graph_ != nullptr) {
        graph_->write_pending_kernel_ddr_registers(ticket_, kernel);
    } else {
        for (const auto& pair : pairs_) {
            kernel.set_regs(pair.reg_lo,
                            static_cast<uint16_t>(pair.encoded_addr));
            kernel.set_regs(pair.reg_hi,
                            static_cast<uint16_t>(pair.encoded_addr >> 16));
        }
    }
    written_ = true;
}

const GraphKernelRegisterRule& RpuKernelGraph::kernel_register_rule_for(
        KernelId id) const {
    for (const auto& rule : kernel_register_census_policy_.rules) {
        if (rule.kernel_id.has_value() && *rule.kernel_id == id) return rule;
    }
    TORCH_CHECK(false, "typed DDR-register policy rejects KernelId ",
                static_cast<int>(id));
}

const GraphKernelRegisterRule& RpuKernelGraph::kernel_register_rule_for(
        const std::string& name) const {
    for (const auto& rule : kernel_register_census_policy_.rules) {
        if (!rule.kernel_name.empty() && rule.kernel_name == name) return rule;
    }
    TORCH_CHECK(false, "typed DDR-register policy rejects dynamic kernel '",
                name, "'");
}

void RpuKernelGraph::begin_kernel_register_census(
        const GraphKernelRegisterPolicy& policy,
        uint64_t physical_plan_hash,
        uint64_t live_physical_epoch) {
    validate_policy(policy);
    TORCH_CHECK(scope_open_ && active_stack_.size() == 1 &&
                    active_stack_.back() == this &&
                    (state_ == State::RECORDING ||
                     state_ == State::REPLAYING),
                "typed DDR-register census must start in the active Graph "
                "RECORDING/REPLAYING scope");
    TORCH_CHECK(!kernel_register_census_active_ &&
                    !kernel_register_census_poisoned_ &&
                    !kernel_register_census_complete_ &&
                    !kernel_register_census_invocation_armed_ &&
                    !pending_kernel_register_census_.has_value() &&
                    cursor_ == 0 && nodes_.empty() ==
                        (state_ == State::RECORDING),
                "typed DDR-register census must own the full Graph window "
                "from node/cursor zero");
    TORCH_CHECK(physical_plan_hash != 0 && live_physical_epoch != 0,
                "typed DDR-register census needs nonzero plan and live epoch");
    TORCH_CHECK(replayable_,
                "typed DDR-register census cannot enter a downgraded Graph");
    TORCH_CHECK(state_ != State::REPLAYING ||
                    !force_oneshot_on_replay_enabled(),
                "typed DDR-register REPLAY rejects "
                "RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY before census mutation");
    const uint64_t policy_digest = stable_policy_digest(policy);

    if (state_ == State::RECORDING) {
        TORCH_CHECK(!has_kernel_register_census_nodes_ &&
                        !kernel_register_census_build_committed_,
                    "typed DDR-register BUILD census is already present");
        // Publish fail-closed candidate authority before any potentially
        // allocating policy/vector copy. A caught bad_alloc cannot silently
        // resume this outer Graph as an untyped capture.
        has_kernel_register_census_nodes_ = true;
        kernel_register_census_policy_ = policy;
        kernel_register_census_plan_hash_ = physical_plan_hash;
        kernel_register_census_policy_digest_ = policy_digest;
        kernel_register_census_rule_counts_.assign(policy.rules.size(), 0);
    } else {
        TORCH_CHECK(has_kernel_register_census_nodes_ &&
                        kernel_register_census_build_committed_ &&
                        kernel_register_census_policy_digest_ ==
                            policy_digest &&
                        same_policy(kernel_register_census_policy_, policy) &&
                        kernel_register_census_plan_hash_ == physical_plan_hash,
                    "typed DDR-register REPLAY policy/plan does not match the "
                    "committed BUILD");
        kernel_register_census_rule_counts_.assign(
            kernel_register_census_policy_.rules.size(), 0);
    }

    kernel_register_census_active_ = true;
    kernel_register_census_complete_ = false;
    kernel_register_census_invocation_armed_ = false;
    kernel_register_census_live_epoch_ = live_physical_epoch;
    kernel_register_census_node_begin_ = 0;
    kernel_register_census_next_phase_ = 0;
    next_kernel_register_writer_ticket_ = 0;
    pending_kernel_register_census_.reset();
    kernel_register_census_stats_ = {};
    kernel_register_census_stats_.ordered_node_kind_hash = kFnvOffset;
    kernel_register_census_stats_.ordered_kernel_hash = kFnvOffset;
    kernel_register_census_phase_begin_stats_ =
        kernel_register_census_stats_;
}

GraphKernelRegisterWriter RpuKernelGraph::stage_kernel_ddr_registers(
        KernelId id, std::vector<GraphDdrRegisterOperandSpec> operands) {
    if (!kernel_register_census_active_) {
        TORCH_CHECK(!has_kernel_register_census_nodes_,
                    "typed DDR-register launcher ran without the required "
                    "census guard");
        std::vector<GraphKernelRegisterWriter::RegisterPair> pairs;
        pairs.reserve(operands.size());
        for (const auto& operand : operands) {
            const auto occurrence = snapshot_operand(operand);
            pairs.push_back(GraphKernelRegisterWriter::RegisterPair{
                occurrence.abi.reg_lo,
                occurrence.abi.reg_hi,
                static_cast<uint32_t>(occurrence.dev_addr >> 8)});
        }
        return GraphKernelRegisterWriter(nullptr, 0, std::move(pairs));
    }

    // All authority, policy, ABI, and replay-node checks precede tensor data
    // access and RpuGetDevAddr. A rejected route therefore performs no address
    // resolution and leaves the invocation retryable at the same cursor.
    TORCH_CHECK(!kernel_register_census_poisoned_ &&
                    !kernel_register_census_complete_ &&
                    !pending_kernel_register_census_.has_value(),
                "typed DDR-register stage overlaps a pending/completed census "
                "kernel");
    const auto& rule = kernel_register_rule_for(id);
    TORCH_CHECK(rule.kind == GraphKernelRegisterRuleKind::TypedDdr &&
                    rule.operands.size() == operands.size(),
                "typed DDR-register stage does not match policy rule kind/arity");
    for (size_t i = 0; i < operands.size(); ++i) {
        TORCH_CHECK(rule.operands[i] == operands[i].abi,
                    "typed DDR-register operand ABI/order mismatches policy");
    }
    const size_t rule_index = static_cast<size_t>(
        &rule - kernel_register_census_policy_.rules.data());
    const GraphKernelRegisterCensus* built_census = nullptr;
    if (state_ == State::REPLAYING) {
        TORCH_CHECK(cursor_ < nodes_.size() &&
                        nodes_[cursor_].kind == GraphNodeKind::Kernel,
                    "typed DDR-register stage expected the next replay node "
                    "to be Kernel at cursor ", cursor_);
        const auto& node = nodes_[cursor_].as_kernel();
        TORCH_CHECK(node.kernel_id.has_value() && *node.kernel_id == id &&
                        node.register_census.has_value(),
                    "typed DDR-register stage replay KernelId/census mismatch");
        built_census = &*node.register_census;
        TORCH_CHECK(
            built_census->kind == GraphKernelRegisterCensus::Kind::TypedDdr &&
                built_census->policy_rule_index == rule_index &&
                built_census->operands.size() == operands.size(),
            "typed DDR-register stage replay census shape mismatch");
        for (size_t i = 0; i < operands.size(); ++i) {
            TORCH_CHECK(built_census->operands[i].abi == operands[i].abi,
                        "typed DDR-register replay ABI/order drift");
        }
    }

    std::vector<GraphDdrRegisterOccurrence> occurrences;
    std::vector<GraphKernelRegisterWriter::RegisterPair> pairs;
    occurrences.reserve(operands.size());
    pairs.reserve(operands.size());
    for (const auto& operand : operands) {
        auto occurrence = snapshot_operand(operand);
        pairs.push_back(GraphKernelRegisterWriter::RegisterPair{
            occurrence.abi.reg_lo,
            occurrence.abi.reg_hi,
            static_cast<uint32_t>(occurrence.dev_addr >> 8)});
        occurrences.push_back(std::move(occurrence));
    }
    if (built_census != nullptr) {
        for (size_t i = 0; i < built_census->operands.size(); ++i) {
            check_same_occurrence(built_census->operands[i], occurrences[i],
                                  "typed DDR-register REPLAY");
        }
    }
    PendingKernelRegisterCensus pending;
    pending.kernel_id = id;
    pending.census.kind = GraphKernelRegisterCensus::Kind::TypedDdr;
    pending.census.policy_rule_index = rule_index;
    pending.census.operands = std::move(occurrences);
    pending.writer_ticket = ++next_kernel_register_writer_ticket_;
    pending.writer_required = true;
    pending.expected_pairs = pairs;

    if (built_census != nullptr) {
        pending.replay_node_idx = cursor_;
    }

    const uint64_t ticket = pending.writer_ticket;
    pending_kernel_register_census_.emplace(std::move(pending));
    return GraphKernelRegisterWriter(this, ticket, std::move(pairs));
}

void RpuKernelGraph::stage_kernel_no_ddr(
        KernelId id, GraphKernelNoDdrProof proof) {
    if (!kernel_register_census_active_) {
        TORCH_CHECK(!has_kernel_register_census_nodes_,
                    "explicit no-DDR launcher ran without the required census "
                    "guard");
        return;
    }
    TORCH_CHECK(!kernel_register_census_poisoned_ &&
                    !kernel_register_census_complete_ &&
                    !pending_kernel_register_census_.has_value(),
                "explicit no-DDR stage overlaps a pending/completed census "
                "kernel");
    const auto& rule = kernel_register_rule_for(id);
    TORCH_CHECK(rule.kind == GraphKernelRegisterRuleKind::ExplicitNoDdr &&
                    proof != GraphKernelNoDdrProof::None &&
                    rule.no_ddr_proof == proof,
                "explicit no-DDR proof mismatches policy");
    PendingKernelRegisterCensus pending;
    pending.kernel_id = id;
    pending.census.kind = GraphKernelRegisterCensus::Kind::NoDdr;
    pending.census.policy_rule_index = static_cast<size_t>(
        &rule - kernel_register_census_policy_.rules.data());
    pending.writer_committed = true;
    if (state_ == State::REPLAYING) {
        TORCH_CHECK(cursor_ < nodes_.size() &&
                        nodes_[cursor_].kind == GraphNodeKind::Kernel,
                    "explicit no-DDR stage expected Kernel at replay cursor ",
                    cursor_);
        const auto& node = nodes_[cursor_].as_kernel();
        TORCH_CHECK(node.kernel_id.has_value() && *node.kernel_id == id &&
                        node.register_census.has_value() &&
                        node.register_census->kind ==
                            GraphKernelRegisterCensus::Kind::NoDdr &&
                        node.register_census->policy_rule_index ==
                            pending.census.policy_rule_index,
                    "explicit no-DDR REPLAY census mismatch");
        pending.replay_node_idx = cursor_;
    }
    pending_kernel_register_census_.emplace(std::move(pending));
}

void RpuKernelGraph::stage_kernel_no_ddr(
        const std::string& name, GraphKernelNoDdrProof proof) {
    if (!kernel_register_census_active_) {
        TORCH_CHECK(!has_kernel_register_census_nodes_,
                    "explicit dynamic no-DDR launcher ran without the "
                    "required census guard");
        return;
    }
    TORCH_CHECK(!kernel_register_census_poisoned_ &&
                    !kernel_register_census_complete_ &&
                    !pending_kernel_register_census_.has_value(),
                "explicit dynamic no-DDR stage overlaps a pending/completed "
                "census kernel");
    const auto& rule = kernel_register_rule_for(name);
    TORCH_CHECK(rule.kind == GraphKernelRegisterRuleKind::ExplicitNoDdr &&
                    proof != GraphKernelNoDdrProof::None &&
                    rule.no_ddr_proof == proof,
                "explicit dynamic no-DDR proof mismatches policy");
    PendingKernelRegisterCensus pending;
    pending.kernel_name = name;
    pending.census.kind = GraphKernelRegisterCensus::Kind::NoDdr;
    pending.census.policy_rule_index = static_cast<size_t>(
        &rule - kernel_register_census_policy_.rules.data());
    pending.writer_committed = true;
    if (state_ == State::REPLAYING) {
        TORCH_CHECK(cursor_ < nodes_.size() &&
                        nodes_[cursor_].kind == GraphNodeKind::Kernel,
                    "explicit dynamic no-DDR stage expected Kernel at replay "
                    "cursor ", cursor_);
        const auto& node = nodes_[cursor_].as_kernel();
        TORCH_CHECK(!node.kernel_id.has_value() &&
                        node.kernel_name == name &&
                        node.register_census.has_value() &&
                        node.register_census->kind ==
                            GraphKernelRegisterCensus::Kind::NoDdr &&
                        node.register_census->policy_rule_index ==
                            pending.census.policy_rule_index,
                    "explicit dynamic no-DDR REPLAY census mismatch");
        pending.replay_node_idx = cursor_;
    }
    pending_kernel_register_census_.emplace(std::move(pending));
}

void RpuKernelGraph::admit_kernel_register_lookup(KernelId id) {
    check_kernel_register_dma_burst_closed("typed kernel lookup");
    if (!kernel_register_census_active_) {
        TORCH_CHECK(!has_kernel_register_census_nodes_,
                    "Kernel lookup bypassed the committed typed DDR-register "
                    "census guard");
        return;
    }
    TORCH_CHECK(!kernel_register_census_complete_ &&
                    !kernel_register_census_poisoned_,
                "Kernel lookup is forbidden after typed census completion or "
                "poisoning");
    const auto& rule = kernel_register_rule_for(id);
    if (!pending_kernel_register_census_.has_value()) {
        TORCH_CHECK(rule.kind ==
                        GraphKernelRegisterRuleKind::IntrinsicNoDdr,
                    "typed/explicit-no-DDR Kernel lookup was not staged");
        PendingKernelRegisterCensus pending;
        pending.kernel_id = id;
        pending.census.kind = GraphKernelRegisterCensus::Kind::NoDdr;
        pending.census.policy_rule_index = static_cast<size_t>(
            &rule - kernel_register_census_policy_.rules.data());
        pending.writer_committed = true;
        if (state_ == State::REPLAYING) {
            TORCH_CHECK(cursor_ < nodes_.size() &&
                            nodes_[cursor_].kind == GraphNodeKind::Kernel,
                        "intrinsic no-DDR lookup expected Kernel at replay "
                        "cursor ", cursor_);
            const auto& node = nodes_[cursor_].as_kernel();
            TORCH_CHECK(node.kernel_id.has_value() && *node.kernel_id == id &&
                            node.register_census.has_value() &&
                            node.register_census->kind ==
                                GraphKernelRegisterCensus::Kind::NoDdr &&
                            node.register_census->policy_rule_index ==
                                pending.census.policy_rule_index,
                        "intrinsic no-DDR REPLAY census mismatch");
            pending.replay_node_idx = cursor_;
        }
        pending_kernel_register_census_.emplace(std::move(pending));
    }
    auto& pending = *pending_kernel_register_census_;
    TORCH_CHECK(pending.kernel_id.has_value() && *pending.kernel_id == id &&
                    !pending.kernel_name.has_value() &&
                    !pending.kernel_lookup_seen,
                "typed DDR-register KernelId lookup order/identity mismatch");
    pending.kernel_lookup_seen = true;
}

void RpuKernelGraph::admit_kernel_register_lookup(const std::string& name) {
    check_kernel_register_dma_burst_closed("typed dynamic kernel lookup");
    if (!kernel_register_census_active_) {
        TORCH_CHECK(!has_kernel_register_census_nodes_,
                    "dynamic Kernel lookup bypassed the committed typed "
                    "DDR-register census guard");
        return;
    }
    TORCH_CHECK(!kernel_register_census_complete_ &&
                    !kernel_register_census_poisoned_,
                "dynamic Kernel lookup is forbidden after typed census "
                "completion or poisoning");
    if (!pending_kernel_register_census_.has_value()) {
        // Keep the sole dynamic intrinsic admission exact.  Every other
        // dynamic no-DDR kernel requires launcher staging with a proof.
        TORCH_CHECK(name == "binary_1xC_NxC_tileC",
                    "typed DDR-register V1 rejects every unstaged dynamic "
                    "Kernel except the exact no-DDR "
                    "binary_1xC_NxC_tileC identity");
        const auto& rule = kernel_register_rule_for(name);
        TORCH_CHECK(rule.kind == GraphKernelRegisterRuleKind::IntrinsicNoDdr,
                    "unstaged dynamic typed/explicit-no-DDR kernels are not "
                    "admitted");
        PendingKernelRegisterCensus pending;
        pending.kernel_name = name;
        pending.census.kind = GraphKernelRegisterCensus::Kind::NoDdr;
        pending.census.policy_rule_index = static_cast<size_t>(
            &rule - kernel_register_census_policy_.rules.data());
        pending.writer_committed = true;
        if (state_ == State::REPLAYING) {
            TORCH_CHECK(cursor_ < nodes_.size() &&
                            nodes_[cursor_].kind == GraphNodeKind::Kernel,
                        "dynamic no-DDR lookup expected Kernel at replay "
                        "cursor ", cursor_);
            const auto& node = nodes_[cursor_].as_kernel();
            TORCH_CHECK(!node.kernel_id.has_value() &&
                            node.kernel_name == name &&
                            node.register_census.has_value() &&
                            node.register_census->kind ==
                                GraphKernelRegisterCensus::Kind::NoDdr &&
                            node.register_census->policy_rule_index ==
                                pending.census.policy_rule_index,
                        "dynamic no-DDR REPLAY census mismatch");
            pending.replay_node_idx = cursor_;
        }
        pending_kernel_register_census_.emplace(std::move(pending));
    }
    auto& pending = *pending_kernel_register_census_;
    TORCH_CHECK(!pending.kernel_id.has_value() &&
                    pending.kernel_name.has_value() &&
                    *pending.kernel_name == name &&
                    !pending.kernel_lookup_seen,
                "typed DDR-register dynamic Kernel lookup order/identity "
                "mismatch");
    pending.kernel_lookup_seen = true;
}

void RpuKernelGraph::bind_pending_kernel_register_lookup(
        ::rhino_lkn::Kernel_t& kernel) {
    if (!kernel_register_census_active_ &&
        !has_kernel_register_census_nodes_) return;
    TORCH_CHECK(kernel_register_census_active_ &&
                    pending_kernel_register_census_.has_value(),
                "typed DDR-register GET returned without pending admission");
    auto& pending = *pending_kernel_register_census_;
    TORCH_CHECK(pending.kernel_lookup_seen &&
                    pending.lookup_kernel == nullptr,
                "typed DDR-register GET binding is missing or duplicated");
    if (state_ == State::RECORDING) {
        TORCH_CHECK(pending_kernel_idx_ < kernels_.size() &&
                        kernels_[pending_kernel_idx_].get() == &kernel,
                    "typed DDR-register GET returned a foreign RECORDING "
                    "Kernel clone");
    } else {
        TORCH_CHECK(state_ == State::REPLAYING &&
                        pending.replay_node_idx < nodes_.size(),
                    "typed DDR-register GET binding has invalid REPLAY state");
        const auto& node = nodes_[pending.replay_node_idx].as_kernel();
        TORCH_CHECK(node.kernel_idx < kernels_.size() &&
                        kernels_[node.kernel_idx].get() == &kernel,
                    "typed DDR-register GET returned a foreign REPLAY Kernel");
    }
    pending.lookup_kernel = &kernel;
}

void RpuKernelGraph::write_pending_kernel_ddr_registers(
        uint64_t ticket, ::rhino_lkn::Kernel_t& kernel) {
    TORCH_CHECK(kernel_register_census_active_ &&
                    !kernel_register_census_complete_ &&
                    pending_kernel_register_census_.has_value(),
                "typed DDR-register writer has no active pending Kernel");
    auto& pending = *pending_kernel_register_census_;
    TORCH_CHECK(pending.writer_required && !pending.writer_committed &&
                    pending.kernel_lookup_seen &&
                    pending.lookup_kernel == &kernel &&
                    pending.writer_ticket == ticket,
                "typed DDR-register writer ticket/order is stale");
    for (const auto& pair : pending.expected_pairs) {
        kernel.set_regs(pair.reg_lo,
                        static_cast<uint16_t>(pair.encoded_addr));
        kernel.set_regs(pair.reg_hi,
                        static_cast<uint16_t>(pair.encoded_addr >> 16));
    }
    pending.written_kernel = &kernel;
    pending.writer_committed = true;
}

void RpuKernelGraph::consume_pending_kernel_register_census(
        ::rhino_lkn::Kernel_t& kernel,
        KernelNodeData* recording_node,
        size_t replay_node_idx) {
    TORCH_CHECK(kernel_register_census_active_ &&
                    pending_kernel_register_census_.has_value(),
                "Kernel enqueue bypassed typed DDR-register census staging");
    auto& pending = *pending_kernel_register_census_;
    TORCH_CHECK(pending.kernel_lookup_seen && pending.writer_committed &&
                    pending.lookup_kernel == &kernel &&
                    (!pending.writer_required ||
                     pending.written_kernel == &kernel),
                "Kernel enqueue precedes typed lookup/address writer");
    if (pending.writer_required) {
        check_register_pairs(kernel, pending.expected_pairs,
                             "typed DDR-register enqueue preflight");
    }
    if (state_ == State::RECORDING) {
        TORCH_CHECK(recording_node != nullptr,
                    "typed DDR-register RECORDING enqueue has no node");
        recording_node->register_census = pending.census;
    } else {
        TORCH_CHECK(recording_node == nullptr &&
                        replay_node_idx == pending.replay_node_idx &&
                        replay_node_idx < nodes_.size(),
                    "typed DDR-register REPLAY enqueue cursor mismatch");
    }

    const size_t rule_index = pending.census.policy_rule_index;
    TORCH_CHECK(rule_index < kernel_register_census_rule_counts_.size(),
                "typed DDR-register policy rule index is invalid");
    ++kernel_register_census_rule_counts_[rule_index];
    ++kernel_register_census_stats_.kernel_count;
    if (pending.census.kind == GraphKernelRegisterCensus::Kind::TypedDdr) {
        ++kernel_register_census_stats_.typed_kernel_count;
        kernel_register_census_stats_.operand_count +=
            pending.census.operands.size();
        for (const auto& operand : pending.census.operands) {
            switch (operand.abi.role) {
            case GraphDdrRegisterRole::ModelWeight:
            case GraphDdrRegisterRole::Int8ModelWeight:
            case GraphDdrRegisterRole::PerChannelScale:
                ++kernel_register_census_stats_.weight_operand_count;
                break;
            case GraphDdrRegisterRole::KeyCache:
                ++kernel_register_census_stats_.key_cache_operand_count;
                break;
            case GraphDdrRegisterRole::ValueCache:
                ++kernel_register_census_stats_.value_cache_operand_count;
                break;
            case GraphDdrRegisterRole::SemanticInput:
                ++kernel_register_census_stats_.semantic_input_operand_count;
                break;
            }
        }
    } else {
        ++kernel_register_census_stats_.no_ddr_kernel_count;
    }
    const auto& rule = kernel_register_census_policy_.rules[rule_index];
    const std::string& name = !rule.kernel_name.empty()
        ? rule.kernel_name
        : std::string(KERNEL_ID_NAMES[
              static_cast<uint8_t>(*rule.kernel_id)]);
    observe_kernel_register_node(GraphNodeKind::Kernel, &name);
    pending_kernel_register_census_.reset();
}

void RpuKernelGraph::observe_kernel_register_node(
        GraphNodeKind kind, const std::string* kernel_name) {
    if (!kernel_register_census_active_) return;
    TORCH_CHECK(!kernel_register_census_complete_,
                "Graph node emitted after typed DDR-register census complete");
    ++kernel_register_census_stats_.node_count;
    kernel_register_census_stats_.ordered_node_kind_hash = fnv_byte(
        kernel_register_census_stats_.ordered_node_kind_hash,
        static_cast<uint8_t>(kind));
    kernel_register_census_phase_begin_stats_.ordered_node_kind_hash = fnv_byte(
        kernel_register_census_phase_begin_stats_.ordered_node_kind_hash,
        static_cast<uint8_t>(kind));
    if (kind == GraphNodeKind::Dma) {
        ++kernel_register_census_stats_.dma_count;
    } else if (kind == GraphNodeKind::Barrier) {
        ++kernel_register_census_stats_.barrier_count;
    } else if (kind == GraphNodeKind::Kernel) {
        TORCH_INTERNAL_ASSERT(kernel_name != nullptr);
        kernel_register_census_stats_.ordered_kernel_hash = fnv_string(
            kernel_register_census_stats_.ordered_kernel_hash, *kernel_name);
        kernel_register_census_phase_begin_stats_.ordered_kernel_hash =
            fnv_string(
                kernel_register_census_phase_begin_stats_.ordered_kernel_hash,
                *kernel_name);
    } else {
        TORCH_CHECK(false, "typed DDR-register census rejects Graph node kind ",
                    static_cast<int>(kind));
    }
}

void RpuKernelGraph::checkpoint_kernel_register_census(
        const std::string& phase_name) {
    TORCH_CHECK(kernel_register_census_active_ &&
                    !kernel_register_census_complete_ &&
                    !pending_kernel_register_census_.has_value() &&
                    kernel_register_census_next_phase_ <
                        kernel_register_census_policy_.phases.size(),
                "typed DDR-register phase checkpoint is out of order or has "
                "a pending Kernel");
    const auto& expected = kernel_register_census_policy_.phases[
        kernel_register_census_next_phase_];
    TORCH_CHECK(expected.name == phase_name,
                "typed DDR-register phase name/order mismatch: expected '",
                expected.name, "' got '", phase_name, "'");
    const auto delta = stats_delta(kernel_register_census_stats_,
                                   kernel_register_census_phase_begin_stats_);
    check_stats_counts(delta, expected.stats, phase_name);
    TORCH_CHECK(
        kernel_register_census_phase_begin_stats_.ordered_node_kind_hash ==
                expected.stats.ordered_node_kind_hash &&
            kernel_register_census_phase_begin_stats_.ordered_kernel_hash ==
                expected.stats.ordered_kernel_hash,
        "typed DDR-register census ", phase_name,
        " ordered node/kernel prefix mismatch; actual=0x", std::hex,
        kernel_register_census_phase_begin_stats_.ordered_node_kind_hash,
        "/0x",
        kernel_register_census_phase_begin_stats_.ordered_kernel_hash,
        " expected=0x", expected.stats.ordered_node_kind_hash, "/0x",
        expected.stats.ordered_kernel_hash, std::dec);
    ++kernel_register_census_next_phase_;
    kernel_register_census_phase_begin_stats_ =
        kernel_register_census_stats_;
    kernel_register_census_phase_begin_stats_.ordered_node_kind_hash =
        kFnvOffset;
    kernel_register_census_phase_begin_stats_.ordered_kernel_hash = kFnvOffset;
}

void RpuKernelGraph::complete_kernel_register_outer_fast_ordinary() {
    auto& candidate = kernel_register_outer_fast_candidate_;
    const auto& sealed_build = kernel_register_outer_fast_build_state_;
    if (!candidate.engaged) {
        TORCH_CHECK(state_ != State::REPLAYING || !sealed_build.staged,
                    "owner-bound schema-5 BUILD requires ordinary REPLAY to "
                    "stage its current physical/component/DMA bindings");
        return;
    }
    TORCH_CHECK(candidate.mode == OuterFastCandidateMode::Ordinary &&
                    candidate.physical_digest.has_value() &&
                    !candidate.components.empty() &&
                    !candidate.open_ordinary_dma_burst.has_value() &&
                    !candidate.ordinary_dma_bursts.empty() &&
                    !kernel_register_outer_fast_validation_armed_ &&
                    candidate.build_generation == build_generation_ &&
                    candidate.policy_digest ==
                        kernel_register_census_policy_digest_ &&
                    candidate.plan_hash == kernel_register_census_plan_hash_ &&
                    candidate.live_epoch == kernel_register_census_live_epoch_,
                "ordinary schema-5 outer binding is incomplete or stale");
    for (const auto& burst : candidate.ordinary_dma_bursts) {
        TORCH_CHECK(burst.observed_occurrence_count ==
                        burst.expected_occurrence_count,
                    "ordinary schema-5 DMA burst was not exactly consumed");
    }

    OuterFastPreparedCommit prepared;
    prepared.valid = true;
    prepared.fast = false;
    prepared.canonical_census = kernel_register_census_stats_;
    const auto resolve_invocation_visibility =
        [&](const OuterFastBuildState& build) {
            std::vector<size_t> ordinals;
            ordinals.reserve(candidate.invocation_visibility.size());
            for (const auto& current : candidate.invocation_visibility) {
                size_t matched_ordinal = build.fixed_dma_owners.size();
                size_t match_count = 0;
                for (size_t i = 0; i < build.fixed_dma_owners.size(); ++i) {
                    const auto& fixed = build.fixed_dma_owners[i];
                    if (fixed.side == GraphOuterFastDmaSide::Source &&
                        fixed.owner.tensor_impl == current.tensor_impl &&
                        fixed.owner.storage_impl == current.storage_impl &&
                        fixed.owner.topology_id == current.topology_id) {
                        matched_ordinal = i;
                        ++match_count;
                    }
                }
                TORCH_CHECK(match_count == 1,
                            "outer fast invocation visibility must resolve to "
                            "exactly one BUILD-derived fixed Source owner");
                check_kernel_register_outer_fast_tensor_exact(
                    build.fixed_dma_owners[matched_ordinal].owner, current,
                    "outer fast invocation visibility");
                ordinals.push_back(matched_ordinal);
            }
            std::sort(ordinals.begin(), ordinals.end());
            TORCH_CHECK(std::adjacent_find(ordinals.begin(), ordinals.end()) ==
                            ordinals.end(),
                        "outer fast invocation visibility contains a duplicate "
                        "fixed owner");
            return ordinals;
        };

    if (state_ == State::RECORDING) {
        OuterFastBuildState build;
        build.staged = true;
        build.physical_digest = *candidate.physical_digest;
        build.invocation_authorities.reserve(
            candidate.invocation_authorities.size());
        for (const auto& authority : candidate.invocation_authorities) {
            build.invocation_authorities.push_back(
                OuterFastBuildInvocationAuthority{
                    authority.prepared_owner, authority.consumed_owner});
        }
        build.components.reserve(candidate.components.size());
        for (const auto& current : candidate.components) {
            OuterFastBuildComponent component;
            component.owners.reserve(current.owners.size());
            for (const auto& owner : current.owners) {
                component.owners.push_back(
                    OuterFastBuildOwnerStamp{owner.owner,
                                             owner.expected_generation});
                prepared.invocation_owner_stamps.push_back(owner);
            }
            component.key_cache = current.key_cache;
            component.value_cache = current.value_cache;
            prepared.invocation_owners.insert(
                prepared.invocation_owners.end(),
                current.key_cache.begin(), current.key_cache.end());
            prepared.invocation_owners.insert(
                prepared.invocation_owners.end(),
                current.value_cache.begin(), current.value_cache.end());
            build.components.push_back(std::move(component));
        }

        build.dma_bursts.reserve(candidate.ordinary_dma_bursts.size());
        for (const auto& current : candidate.ordinary_dma_bursts) {
            OuterFastBuildDmaBurst burst;
            burst.variant = current.variant;
            burst.side = current.side;
            burst.owner = current.owner;
            burst.live_base = current.live_base;
            burst.byte_begin = current.byte_begin;
            burst.byte_count = current.byte_count;
            burst.expected_occurrence_count =
                current.expected_occurrence_count;
            build.dma_bursts.push_back(std::move(burst));
            prepared.invocation_owners.push_back(current.owner);
        }

        size_t observed_dma_nodes = 0;
        for (size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
            if (nodes_[node_index].kind != GraphNodeKind::Dma) continue;
            ++observed_dma_nodes;
            auto& node = nodes_[node_index].as_dma();
            TORCH_CHECK(node.outer_fast_owner_bound &&
                            node.outer_fast_owner_ordinal <
                                build.dma_bursts.size(),
                        "schema-5 outer binding requires every retained DMA "
                        "node to consume one typed owner burst");
            const size_t burst_ordinal = node.outer_fast_owner_ordinal;
            auto& burst = build.dma_bursts[burst_ordinal];
            size_t owner_ordinal = 0;
            if (burst.variant == DmaNodeData::Variant::Fixed) {
                bool found = false;
                for (size_t i = 0; i < build.fixed_dma_owners.size(); ++i) {
                    const auto& owner = build.fixed_dma_owners[i];
                    if (owner.side == burst.side &&
                        owner.byte_begin == burst.byte_begin &&
                        owner.byte_count == burst.byte_count &&
                        owner.owner.tensor_impl == burst.owner.tensor_impl &&
                        owner.owner.storage_impl == burst.owner.storage_impl) {
                        owner_ordinal = i;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    owner_ordinal = build.fixed_dma_owners.size();
                    OuterFastFixedDmaOwner owner;
                    owner.side = burst.side;
                    owner.owner = burst.owner;
                    owner.byte_begin = burst.byte_begin;
                    owner.byte_count = burst.byte_count;
                    build.fixed_dma_owners.push_back(std::move(owner));
                }
                ++build.fixed_dma_owners[owner_ordinal].occurrence_count;
            } else {
                bool found = false;
                for (size_t i = 0; i < build.mutable_dma_slots.size(); ++i) {
                    const auto& slot = build.mutable_dma_slots[i];
                    if (slot.live_base == burst.live_base) {
                        TORCH_CHECK(slot.side == burst.side &&
                                        slot.owner_topology_id ==
                                            burst.owner.topology_id &&
                                        slot.byte_begin == burst.byte_begin &&
                                        slot.byte_count == burst.byte_count,
                                    "one mutable live slot was rebound to "
                                    "incompatible owner semantics");
                        owner_ordinal = i;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    owner_ordinal = build.mutable_dma_slots.size();
                    OuterFastMutableDmaSlot slot;
                    slot.side = burst.side;
                    slot.live_base = burst.live_base;
                    slot.owner_topology_id = burst.owner.topology_id;
                    slot.byte_begin = burst.byte_begin;
                    slot.byte_count = burst.byte_count;
                    build.mutable_dma_slots.push_back(std::move(slot));
                }
                ++build.mutable_dma_slots[owner_ordinal].occurrence_count;
                ++build.mutable_occurrence_count;
            }
            burst.owner_ordinal = owner_ordinal;
            candidate.ordinary_dma_bursts[burst_ordinal].owner_ordinal =
                owner_ordinal;
            prepared.node_annotations.push_back(OuterFastNodeAnnotation{
                &node, burst.side, owner_ordinal,
                burst.owner.topology_id,
                node.outer_fast_owner_byte_offset});
        }
        TORCH_CHECK(observed_dma_nodes ==
                        kernel_register_census_policy_.expected_dma_count,
                    "schema-5 outer binding DMA sidecar does not cover the "
                    "full retained DMA window");
        build.invocation_visibility_fixed_owner_ordinals =
            resolve_invocation_visibility(build);

        uint64_t topology = kFnvOffset;
        topology = fnv_u64(topology, UINT64_C(0x4f55544641535431));
        topology = fnv_u64(topology, build.components.size());
        topology = fnv_u64(topology, build.fixed_dma_owners.size());
        topology = fnv_u64(topology, build.mutable_dma_slots.size());
        topology = fnv_u64(topology, build.mutable_occurrence_count);
        for (const auto& owner : build.fixed_dma_owners) {
            topology = fnv_byte(topology,
                static_cast<uint8_t>(owner.side));
            topology = fnv_u64(topology, owner.owner.topology_id);
            topology = fnv_u64(topology, owner.byte_begin);
            topology = fnv_u64(topology, owner.byte_count);
            topology = fnv_u64(topology, owner.occurrence_count);
        }
        for (const auto& slot : build.mutable_dma_slots) {
            topology = fnv_byte(topology,
                static_cast<uint8_t>(slot.side));
            topology = fnv_u64(topology, slot.owner_topology_id);
            topology = fnv_u64(topology, slot.byte_begin);
            topology = fnv_u64(topology, slot.byte_count);
            topology = fnv_u64(topology, slot.occurrence_count);
        }
        topology = fnv_u64(
            topology,
            build.invocation_visibility_fixed_owner_ordinals.size());
        for (const size_t ordinal :
             build.invocation_visibility_fixed_owner_ordinals) {
            topology = fnv_u64(topology, ordinal);
        }
        topology = fnv_u64(topology, build.invocation_authorities.size());
        for (size_t ordinal = 0;
             ordinal < build.invocation_authorities.size(); ++ordinal) {
            topology = fnv_u64(topology, ordinal);
            topology = fnv_byte(topology, UINT8_C(0));
            topology = fnv_byte(topology, UINT8_C(1));
        }
        build.topology_digest = topology;
        prepared.mutable_slot_count = build.mutable_dma_slots.size();
        prepared.mutable_occurrence_count = build.mutable_occurrence_count;
        prepared.build_state.emplace(std::move(build));
    } else {
        TORCH_CHECK(state_ == State::REPLAYING && sealed_build.staged &&
                        sealed_build.committed &&
                        candidate.build_topology_hash == build_topology_hash_ &&
                        *candidate.physical_digest ==
                            sealed_build.physical_digest &&
                        candidate.components.size() ==
                            sealed_build.components.size() &&
                        candidate.invocation_authorities.size() ==
                            sealed_build.invocation_authorities.size() &&
                        candidate.ordinary_dma_bursts.size() ==
                            sealed_build.dma_bursts.size(),
                    "ordinary schema-5 outer binding differs from BUILD");
        for (size_t i = 0;
             i < sealed_build.invocation_authorities.size(); ++i) {
            const auto prepared_owner = sealed_build
                .invocation_authorities[i].prepared_owner.lock();
            const auto consumed_owner = sealed_build
                .invocation_authorities[i].consumed_owner.lock();
            const auto& current = candidate.invocation_authorities[i];
            TORCH_CHECK(
                same_shared_owner_identity(prepared_owner,
                                           current.prepared_owner) &&
                    same_shared_owner_identity(consumed_owner,
                                               current.consumed_owner) &&
                    *current.prepared_owner ==
                        current.expected_generation &&
                    *current.consumed_owner <
                        current.expected_generation,
                "ordinary schema-5 invocation authority owner/order drifted");
        }
        TORCH_CHECK(resolve_invocation_visibility(sealed_build) ==
                        sealed_build
                            .invocation_visibility_fixed_owner_ordinals,
                    "ordinary schema-5 invocation visibility differs from "
                    "BUILD");
        prepared.mutable_slot_count =
            sealed_build.mutable_dma_slots.size();
        prepared.mutable_occurrence_count =
            sealed_build.mutable_occurrence_count;
        for (const auto& component : candidate.components) {
            prepared.invocation_owner_stamps.insert(
                prepared.invocation_owner_stamps.end(),
                component.owners.begin(), component.owners.end());
            prepared.invocation_owners.insert(
                prepared.invocation_owners.end(),
                component.key_cache.begin(), component.key_cache.end());
            prepared.invocation_owners.insert(
                prepared.invocation_owners.end(),
                component.value_cache.begin(), component.value_cache.end());
        }
        for (const auto& burst : candidate.ordinary_dma_bursts) {
            prepared.invocation_owners.push_back(burst.owner);
        }
    }

    const auto& target_build = prepared.build_state.has_value()
        ? *prepared.build_state : sealed_build;
    std::vector<bool> mutable_seen(target_build.mutable_dma_slots.size(), false);
    for (const auto& burst : candidate.ordinary_dma_bursts) {
        if (burst.variant == DmaNodeData::Variant::Fixed) continue;
        TORCH_CHECK(burst.owner_ordinal < target_build.mutable_dma_slots.size(),
                    "ordinary mutable DMA owner ordinal is invalid");
        const auto& slot = target_build.mutable_dma_slots[burst.owner_ordinal];
        TORCH_CHECK(slot.live_base == burst.live_base &&
                        slot.side == burst.side &&
                        slot.owner_topology_id == burst.owner.topology_id,
                    "ordinary mutable DMA slot owner drifted");
        if (!mutable_seen[burst.owner_ordinal]) {
            mutable_seen[burst.owner_ordinal] = true;
            TORCH_CHECK(
                burst.live_base != nullptr &&
                    *burst.live_base == burst.owner.dev_addr,
                "ordinary outer-fast mutable DMA slot must already publish "
                "the current op-walk owner address; ordinal=",
                burst.owner_ordinal, " side=",
                static_cast<int>(burst.side), " range=[", burst.byte_begin,
                ",", burst.byte_begin + burst.byte_count,
                ") occurrences=", burst.expected_occurrence_count,
                " current=0x", std::hex,
                burst.live_base == nullptr ? UINT64_C(0) : *burst.live_base,
                " expected=0x", burst.owner.dev_addr, std::dec);
            prepared.slot_writes.emplace_back(
                burst.live_base, burst.owner.dev_addr);
            if (slot.side == GraphOuterFastDmaSide::Destination) {
                if (slot.last_successful_destination.has_value()) {
                    TORCH_CHECK(
                        slot.last_successful_destination->tensor_impl !=
                                burst.owner.tensor_impl &&
                            slot.last_successful_destination->storage_impl !=
                                burst.owner.storage_impl,
                        "outer fast mutable destination must use a fresh "
                        "TensorImpl and StorageImpl each invocation");
                }
                prepared.mutable_successful_owners.emplace_back(
                    burst.owner_ordinal, burst.owner);
            }
        } else {
            const auto write = std::find_if(
                prepared.slot_writes.begin(), prepared.slot_writes.end(),
                [&](const auto& item) { return item.first == burst.live_base; });
            TORCH_INTERNAL_ASSERT(write != prepared.slot_writes.end());
            TORCH_CHECK(write->second == burst.owner.dev_addr,
                        "one mutable live slot was staged with two tensors");
        }
    }
    TORCH_CHECK(std::all_of(mutable_seen.begin(), mutable_seen.end(),
                            [](bool value) { return value; }),
                "ordinary schema-5 mutable slot exact-cover is incomplete");

    // Ordinary BUILD/REPLAY must preserve every flush registered by the
    // re-emitted op walk.  Only the fast path may synthesize the narrowed set
    // from the sealed outer-fast contract.
    prepared.boundary_visibility = boundary_flush_ptrs_;
    prepared.normal_visibility_owners = candidate.invocation_visibility;
    for (const auto& owner : candidate.invocation_visibility) {
        prepared.boundary_visibility.push_back(
            const_cast<void*>(owner.data_ptr));
    }
    for (const auto& burst : candidate.ordinary_dma_bursts) {
        if (burst.variant == DmaNodeData::Variant::MutableSrc) {
            prepared.force_visibility_owners.push_back(burst.owner);
            prepared.force_visibility.push_back(
                const_cast<void*>(burst.owner.data_ptr));
        } else if (burst.variant == DmaNodeData::Variant::MutableDst) {
            prepared.normal_visibility_owners.push_back(burst.owner);
            prepared.boundary_visibility.push_back(
                const_cast<void*>(burst.owner.data_ptr));
        }
        // A Destination needs force visibility on both sides of execution:
        // PRE prevents CPU-dirty initialization from overwriting the device
        // result, and POST invalidates the CPU view after the device write.
        if (burst.side == GraphOuterFastDmaSide::Destination) {
            prepared.force_visibility_owners.push_back(burst.owner);
            prepared.force_visibility.push_back(
                const_cast<void*>(burst.owner.data_ptr));
            prepared.force_post_visibility_owners.push_back(burst.owner);
            prepared.force_post_visibility.push_back(
                const_cast<void*>(burst.owner.data_ptr));
        }
    }
    auto sort_unique_ptrs = [](std::vector<void*>& ptrs) {
        std::sort(ptrs.begin(), ptrs.end());
        ptrs.erase(std::unique(ptrs.begin(), ptrs.end()), ptrs.end());
    };
    sort_unique_ptrs(prepared.boundary_visibility);
    sort_unique_ptrs(prepared.force_visibility);
    sort_unique_ptrs(prepared.force_post_visibility);
    prepared.invocation_authorities = candidate.invocation_authorities;
    prepared.retained_validated =
        validate_kernel_register_retained_occurrences();
    kernel_register_outer_fast_prepared_.emplace(std::move(prepared));
    commit_kernel_register_outer_fast_prepared(false);
}

void RpuKernelGraph::commit_kernel_register_outer_fast_prepared(
        bool fast) noexcept {
    auto& prepared = *kernel_register_outer_fast_prepared_;

    if (prepared.build_state.has_value()) {
        using std::swap;
        swap(kernel_register_outer_fast_build_state_,
             *prepared.build_state);
    }
    for (const auto& annotation : prepared.node_annotations) {
        auto& dma = *annotation.node;
        dma.outer_fast_owner_bound = true;
        dma.outer_fast_ddr_side = annotation.side;
        dma.outer_fast_owner_ordinal = annotation.owner_ordinal;
        dma.outer_fast_owner_topology_id = annotation.owner_topology_id;
        dma.outer_fast_owner_byte_offset = annotation.owner_byte_offset;
    }

    boundary_flush_ptrs_.swap(prepared.boundary_visibility);
    kernel_register_outer_fast_invocation_force_visibility_.swap(
        prepared.force_visibility);
    kernel_register_outer_fast_invocation_force_post_visibility_.swap(
        prepared.force_post_visibility);
    kernel_register_outer_fast_invocation_owners_.swap(
        prepared.invocation_owners);
    kernel_register_outer_fast_invocation_owner_stamps_.swap(
        prepared.invocation_owner_stamps);
    kernel_register_outer_fast_invocation_authorities_.swap(
        prepared.invocation_authorities);
    kernel_register_outer_fast_normal_visibility_owners_.swap(
        prepared.normal_visibility_owners);
    kernel_register_outer_fast_force_visibility_owners_.swap(
        prepared.force_visibility_owners);
    kernel_register_outer_fast_force_post_visibility_owners_.swap(
        prepared.force_post_visibility_owners);
    kernel_register_outer_fast_mutable_successful_owners_.swap(
        prepared.mutable_successful_owners);
    kernel_register_outer_fast_invocation_slot_values_.swap(
        prepared.slot_writes);
    // The ordinary op walk already publishes every mutable live base before
    // it emits the corresponding DMA and complete_ordinary validates that
    // value exactly.  Rewriting it here would create a second ordinary-path
    // publication authority.  Only a fast replay skips the op walk and must
    // therefore publish its validated slot values during the atomic commit.
    if (fast) {
        for (const auto& [slot, value] :
             kernel_register_outer_fast_invocation_slot_values_) {
            *slot = value;
        }
    }

    kernel_register_outer_fast_invocation_physical_digest_ =
        *kernel_register_outer_fast_candidate_.physical_digest;
    kernel_register_outer_fast_mutable_slot_count_ =
        prepared.mutable_slot_count;
    kernel_register_outer_fast_mutable_occurrence_count_ =
        prepared.mutable_occurrence_count;
    kernel_register_outer_fast_retained_validated_ =
        prepared.retained_validated;
    kernel_register_outer_fast_hit_ = fast;

    if (fast) {
        kernel_register_census_stats_ = prepared.canonical_census;
        kernel_register_census_next_phase_ =
            kernel_register_census_policy_.phases.size();
        kernel_register_census_complete_ = true;
        kernel_register_census_invocation_armed_ = true;
        has_kernel_register_census_nodes_ = true;
        kernel_register_host_reemitted_node_count_ = 0;
        kernel_register_host_reemitted_kernel_count_ = 0;
        op_stream_fully_skipped_ = true;
    }
    kernel_register_outer_fast_validation_armed_ = false;
    kernel_register_outer_fast_ticket_nonce_ = 0;
    kernel_register_outer_fast_prepared_.reset();
    if (fast) cursor_ = nodes_.size();
}

void RpuKernelGraph::mark_kernel_register_outer_fast_execution_success()
        noexcept {
    if (!kernel_register_outer_fast_build_state_.staged) return;
    for (auto& [ordinal, owner] :
         kernel_register_outer_fast_mutable_successful_owners_) {
        if (ordinal >=
            kernel_register_outer_fast_build_state_.mutable_dma_slots.size()) {
            continue;
        }
        auto& destination = kernel_register_outer_fast_build_state_
            .mutable_dma_slots[ordinal].last_successful_destination;
        if (!destination.has_value()) {
            destination.emplace(std::move(owner));
        } else {
            using std::swap;
            swap(*destination, owner);
        }
    }
    for (const auto& authority :
         kernel_register_outer_fast_invocation_authorities_) {
        *authority.consumed_owner = authority.expected_generation;
    }
}

void RpuKernelGraph::stage_kernel_register_outer_fast_physical_digest(
        uint64_t digest) {
    TORCH_CHECK(kernel_register_census_active_ &&
                    !kernel_register_census_complete_ &&
                    !kernel_register_census_poisoned_ && digest != 0 &&
                    !kernel_register_outer_fast_candidate_.engaged,
                "outer fast physical digest must be the first, unique "
                "binding in an active census");
    auto& candidate = kernel_register_outer_fast_candidate_;
    candidate.engaged = true;
    candidate.physical_digest = digest;
    candidate.build_generation = build_generation_;
    candidate.build_topology_hash = build_topology_hash_;
    candidate.policy_digest = kernel_register_census_policy_digest_;
    candidate.plan_hash = kernel_register_census_plan_hash_;
    candidate.live_epoch = kernel_register_census_live_epoch_;
}

void RpuKernelGraph::stage_kernel_register_outer_fast_component(
        std::vector<std::pair<std::shared_ptr<const uint64_t>, uint64_t>>
            owner_stamps,
        std::vector<at::Tensor> key_cache,
        std::vector<at::Tensor> value_cache) {
    auto& candidate = kernel_register_outer_fast_candidate_;
    TORCH_CHECK(kernel_register_census_active_ && candidate.engaged &&
                    candidate.mode == OuterFastCandidateMode::Unset &&
                    !kernel_register_outer_fast_validation_armed_ &&
                    !owner_stamps.empty() && !key_cache.empty() &&
                    key_cache.size() == value_cache.size(),
                "outer fast component requires physical binding, one or more "
                "owner stamps, and paired nonempty K/V catalogs");

    OuterFastComponentCandidate component;
    component.owners.reserve(owner_stamps.size());
    for (auto& [owner, expected] : owner_stamps) {
        TORCH_CHECK(owner != nullptr && expected != 0 && *owner == expected,
                    "outer fast component owner stamp is stale");
        component.owners.push_back(
            OuterFastOwnerStamp{std::move(owner), expected});
    }
    component.key_cache.reserve(key_cache.size());
    component.value_cache.reserve(value_cache.size());
    for (const auto& tensor : key_cache) {
        component.key_cache.push_back(
            snapshot_kernel_register_outer_fast_tensor(tensor));
    }
    for (const auto& tensor : value_cache) {
        component.value_cache.push_back(
            snapshot_kernel_register_outer_fast_tensor(tensor));
    }

    if (state_ == State::REPLAYING) {
        const auto& build = kernel_register_outer_fast_build_state_;
        const size_t ordinal = candidate.components.size();
        TORCH_CHECK(build.staged && build.committed &&
                        ordinal < build.components.size(),
                    "outer fast component catalog is extra or BUILD is "
                    "unsealed");
        const auto& sealed = build.components[ordinal];
        TORCH_CHECK(sealed.owners.size() == component.owners.size() &&
                        sealed.key_cache.size() == component.key_cache.size() &&
                        sealed.value_cache.size() ==
                            component.value_cache.size(),
                    "outer fast component owner/K/V catalog size drifted");
        for (size_t i = 0; i < sealed.owners.size(); ++i) {
            const auto build_owner = sealed.owners[i].owner.lock();
            TORCH_CHECK(build_owner != nullptr &&
                            build_owner.get() ==
                                component.owners[i].owner.get() &&
                            sealed.owners[i].expected_generation ==
                                component.owners[i].expected_generation &&
                            *build_owner ==
                                sealed.owners[i].expected_generation,
                        "outer fast component owner identity/generation "
                        "drifted");
        }
        for (size_t i = 0; i < sealed.key_cache.size(); ++i) {
            check_kernel_register_outer_fast_tensor_exact(
                sealed.key_cache[i], component.key_cache[i],
                "outer fast current K-cache catalog");
            check_kernel_register_outer_fast_tensor_exact(
                sealed.value_cache[i], component.value_cache[i],
                "outer fast current V-cache catalog");
        }
    }
    candidate.components.push_back(std::move(component));
}

void RpuKernelGraph::stage_kernel_register_outer_fast_invocation_authority(
        std::shared_ptr<const uint64_t> prepared_owner,
        std::shared_ptr<uint64_t> consumed_owner,
        uint64_t expected_generation) {
    auto& candidate = kernel_register_outer_fast_candidate_;
    TORCH_CHECK(kernel_register_census_active_ && candidate.engaged &&
                    !kernel_register_outer_fast_validation_armed_ &&
                    prepared_owner != nullptr && consumed_owner != nullptr &&
                    expected_generation != 0 &&
                    *prepared_owner == expected_generation &&
                    *consumed_owner < expected_generation,
                "outer fast invocation authority is missing or stale");
    TORCH_CHECK(!shared_owner_aliases(prepared_owner, consumed_owner),
                "outer fast invocation prepared/consumed owners alias");
    for (const auto& current : candidate.invocation_authorities) {
        TORCH_CHECK(
            !shared_owner_aliases(prepared_owner, current.prepared_owner) &&
                !shared_owner_aliases(consumed_owner,
                                      current.consumed_owner) &&
                !shared_owner_aliases(prepared_owner,
                                      current.consumed_owner) &&
                !shared_owner_aliases(consumed_owner,
                                      current.prepared_owner),
            "outer fast invocation authority owner is duplicated or aliased");
    }
    candidate.invocation_authorities.push_back(
        OuterFastInvocationAuthority{
            std::move(prepared_owner), std::move(consumed_owner),
            expected_generation});
}

void RpuKernelGraph::stage_kernel_register_outer_fast_invocation_visibility(
        const at::Tensor& tensor) {
    auto& candidate = kernel_register_outer_fast_candidate_;
    TORCH_CHECK(kernel_register_census_active_ && candidate.engaged &&
                    !kernel_register_outer_fast_validation_armed_,
                "outer fast invocation visibility requires one active "
                "candidate");
    candidate.invocation_visibility.push_back(
        snapshot_kernel_register_outer_fast_tensor(tensor));
}

void RpuKernelGraph::check_kernel_register_outer_fast_dma_range(
        const RpuKernelGraph::OuterFastTensorSnapshot& owner,
        size_t byte_begin, size_t byte_count) {
    TORCH_CHECK(owner.tensor_owner.is_contiguous(),
                "outer fast DMA owner must be contiguous");
    TORCH_CHECK(byte_count > 0 && byte_begin <= owner.tensor_nbytes &&
                    byte_count <= owner.tensor_nbytes - byte_begin,
                "outer fast DMA owner byte range is empty or out of bounds");
}

void RpuKernelGraph::stage_kernel_register_fixed_dma_burst(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count) {
    auto& candidate = kernel_register_outer_fast_candidate_;
    TORCH_CHECK(kernel_register_census_active_ && candidate.engaged &&
                    !kernel_register_outer_fast_validation_armed_ &&
                    candidate.mode != OuterFastCandidateMode::Fast &&
                    !candidate.open_ordinary_dma_burst.has_value() &&
                    (side == GraphOuterFastDmaSide::Source ||
                     side == GraphOuterFastDmaSide::Destination) &&
                    expected_occurrence_count > 0,
                "outer fast fixed DMA burst is overlapping, unbound, or "
                "invalid");
    OuterFastDmaBurstCandidate burst;
    burst.variant = DmaNodeData::Variant::Fixed;
    burst.side = side;
    burst.owner = snapshot_kernel_register_outer_fast_tensor(tensor);
    burst.byte_begin = byte_begin;
    burst.byte_count = byte_count;
    burst.expected_occurrence_count = expected_occurrence_count;
    check_kernel_register_outer_fast_dma_range(
        burst.owner, byte_begin, byte_count);

    const size_t ordinal = candidate.ordinary_dma_bursts.size();
    if (state_ == State::REPLAYING) {
        const auto& build = kernel_register_outer_fast_build_state_;
        TORCH_CHECK(build.staged && build.committed &&
                        ordinal < build.dma_bursts.size(),
                    "outer fast ordinary fixed DMA burst is extra or BUILD "
                    "is unsealed");
        const auto& sealed = build.dma_bursts[ordinal];
        TORCH_CHECK(sealed.variant == burst.variant &&
                        sealed.side == burst.side &&
                        sealed.byte_begin == byte_begin &&
                        sealed.byte_count == byte_count &&
                        sealed.expected_occurrence_count ==
                            expected_occurrence_count,
                    "outer fast ordinary fixed DMA burst contract drifted");
        check_kernel_register_outer_fast_tensor_exact(
            sealed.owner, burst.owner,
            "outer fast ordinary fixed DMA owner");
        burst.owner_ordinal = sealed.owner_ordinal;
    }
    candidate.mode = OuterFastCandidateMode::Ordinary;
    candidate.ordinary_dma_bursts.push_back(std::move(burst));
    candidate.open_ordinary_dma_burst = ordinal;
}

void RpuKernelGraph::stage_kernel_register_mutable_dma_burst(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count) {
    auto& candidate = kernel_register_outer_fast_candidate_;
    TORCH_CHECK(kernel_register_census_active_ && candidate.engaged &&
                    !kernel_register_outer_fast_validation_armed_ &&
                    candidate.mode != OuterFastCandidateMode::Fast &&
                    !candidate.open_ordinary_dma_burst.has_value() &&
                    (side == GraphOuterFastDmaSide::Source ||
                     side == GraphOuterFastDmaSide::Destination) &&
                    expected_occurrence_count > 0,
                "outer fast mutable DMA burst is overlapping, unbound, or "
                "invalid");
    OuterFastDmaBurstCandidate burst;
    burst.variant = side == GraphOuterFastDmaSide::Source
        ? DmaNodeData::Variant::MutableSrc
        : DmaNodeData::Variant::MutableDst;
    burst.side = side;
    burst.owner = snapshot_kernel_register_outer_fast_tensor(tensor);
    burst.live_base = &live_base;
    burst.byte_begin = byte_begin;
    burst.byte_count = byte_count;
    burst.expected_occurrence_count = expected_occurrence_count;
    check_kernel_register_outer_fast_dma_range(
        burst.owner, byte_begin, byte_count);

    const size_t ordinal = candidate.ordinary_dma_bursts.size();
    if (state_ == State::REPLAYING) {
        const auto& build = kernel_register_outer_fast_build_state_;
        TORCH_CHECK(build.staged && build.committed &&
                        ordinal < build.dma_bursts.size(),
                    "outer fast ordinary mutable DMA burst is extra or BUILD "
                    "is unsealed");
        const auto& sealed = build.dma_bursts[ordinal];
        TORCH_CHECK(sealed.variant == burst.variant &&
                        sealed.side == burst.side &&
                        sealed.live_base == burst.live_base &&
                        sealed.byte_begin == byte_begin &&
                        sealed.byte_count == byte_count &&
                        sealed.expected_occurrence_count ==
                            expected_occurrence_count &&
                        sealed.owner.topology_id == burst.owner.topology_id,
                    "outer fast ordinary mutable DMA slot/topology/range "
                    "drifted");
        burst.owner_ordinal = sealed.owner_ordinal;
    }
    candidate.mode = OuterFastCandidateMode::Ordinary;
    candidate.ordinary_dma_bursts.push_back(std::move(burst));
    candidate.open_ordinary_dma_burst = ordinal;
}

void RpuKernelGraph::stage_census_fixed_dma_burst_if_active(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count) {
    if (!kernel_register_census_active_) return;
    try {
        stage_kernel_register_fixed_dma_burst(
            tensor, side, byte_begin, byte_count,
            expected_occurrence_count);
    } catch (...) {
        cancel_kernel_register_census();
        throw;
    }
}

void RpuKernelGraph::stage_census_mutable_dma_burst_if_active(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count,
        size_t expected_occurrence_count) {
    if (!kernel_register_census_active_) return;
    try {
        stage_kernel_register_mutable_dma_burst(
            live_base, tensor, side, byte_begin, byte_count,
            expected_occurrence_count);
    } catch (...) {
        cancel_kernel_register_census();
        throw;
    }
}

void RpuKernelGraph::bind_kernel_register_outer_fast_fixed_dma(
        const at::Tensor& tensor, GraphOuterFastDmaSide side,
        size_t byte_begin, size_t byte_count) {
    auto& candidate = kernel_register_outer_fast_candidate_;
    TORCH_CHECK(state_ == State::REPLAYING &&
                    kernel_register_census_active_ && candidate.engaged &&
                    !kernel_register_outer_fast_validation_armed_ &&
                    candidate.mode != OuterFastCandidateMode::Ordinary &&
                    (side == GraphOuterFastDmaSide::Source ||
                     side == GraphOuterFastDmaSide::Destination),
                "outer fast fixed DMA binding requires an active REPLAY "
                "candidate");
    OuterFastDmaBurstCandidate binding;
    binding.variant = DmaNodeData::Variant::Fixed;
    binding.side = side;
    binding.owner = snapshot_kernel_register_outer_fast_tensor(tensor);
    binding.byte_begin = byte_begin;
    binding.byte_count = byte_count;
    check_kernel_register_outer_fast_dma_range(
        binding.owner, byte_begin, byte_count);
    candidate.mode = OuterFastCandidateMode::Fast;
    candidate.fast_fixed_dma_bindings.push_back(std::move(binding));
}

void RpuKernelGraph::bind_kernel_register_outer_fast_mutable_dma(
        uint64_t& live_base, const at::Tensor& tensor,
        GraphOuterFastDmaSide side, size_t byte_begin, size_t byte_count) {
    auto& candidate = kernel_register_outer_fast_candidate_;
    TORCH_CHECK(state_ == State::REPLAYING &&
                    kernel_register_census_active_ && candidate.engaged &&
                    !kernel_register_outer_fast_validation_armed_ &&
                    candidate.mode != OuterFastCandidateMode::Ordinary &&
                    (side == GraphOuterFastDmaSide::Source ||
                     side == GraphOuterFastDmaSide::Destination),
                "outer fast mutable DMA binding requires an active REPLAY "
                "candidate");
    OuterFastDmaBurstCandidate binding;
    binding.variant = side == GraphOuterFastDmaSide::Source
        ? DmaNodeData::Variant::MutableSrc
        : DmaNodeData::Variant::MutableDst;
    binding.side = side;
    binding.owner = snapshot_kernel_register_outer_fast_tensor(tensor);
    binding.live_base = &live_base;
    binding.byte_begin = byte_begin;
    binding.byte_count = byte_count;
    check_kernel_register_outer_fast_dma_range(
        binding.owner, byte_begin, byte_count);
    candidate.mode = OuterFastCandidateMode::Fast;
    candidate.fast_mutable_dma_bindings.push_back(std::move(binding));
}

GraphKernelRegisterOuterFastTicket
RpuKernelGraph::validate_kernel_register_outer_fast_replay() {
    auto& candidate = kernel_register_outer_fast_candidate_;
    const auto& build = kernel_register_outer_fast_build_state_;
    TORCH_CHECK(
        state_ == State::REPLAYING && scope_open_ &&
            active_stack_.size() == 1 && active_stack_.back() == this &&
            cursor_ == 0 && kernel_register_census_active_ &&
            !kernel_register_census_complete_ &&
            !kernel_register_census_poisoned_ &&
            !kernel_register_census_invocation_armed_ &&
            !pending_kernel_register_census_.has_value() &&
            !kernel_register_outer_fast_validation_armed_ &&
            !kernel_register_outer_fast_prepared_.has_value() &&
            replayable_ && has_built_signature_ &&
            kernel_register_census_build_committed_ &&
            !force_oneshot_on_replay_enabled(),
        "outer fast replay validation requires a pristine committed schema-5 "
        "REPLAY at cursor zero");
    TORCH_CHECK(
        candidate.engaged &&
            candidate.mode == OuterFastCandidateMode::Fast &&
            candidate.physical_digest.has_value() &&
            candidate.ordinary_dma_bursts.empty() &&
            !candidate.open_ordinary_dma_burst.has_value() &&
            build.staged && build.committed &&
            candidate.build_generation == build_generation_ &&
            candidate.build_topology_hash == build_topology_hash_ &&
            candidate.policy_digest == kernel_register_census_policy_digest_ &&
            candidate.plan_hash == kernel_register_census_plan_hash_ &&
            candidate.live_epoch == kernel_register_census_live_epoch_ &&
            *candidate.physical_digest == build.physical_digest &&
            candidate.components.size() == build.components.size() &&
            candidate.invocation_authorities.size() ==
                build.invocation_authorities.size(),
        "outer fast replay physical/component/build authority differs from "
        "the committed BUILD");
    TORCH_CHECK(candidate.fast_fixed_dma_bindings.empty(),
                "outer fast schema-5 replay derives fixed DMA authority from "
                "the retained BUILD catalog; dynamic fixed bindings are not "
                "admitted");
    TORCH_CHECK(!canonical_dma_poisoned_ &&
                    !canonical_dma_callback_active_ &&
                    !canonical_dma_burst_permit_.active &&
                    (!has_semantic_spm_peer_nodes_ ||
                     semantic_spm_peer_replay_armed_) &&
                    (!has_semantic_dma_nodes_ ||
                     semantic_dma_replay_owner_armed_),
                "outer fast replay semantic/canonical DMA authority is "
                "unarmed or poisoned");
    for (size_t i = 0; i < build.invocation_authorities.size(); ++i) {
        const auto prepared_owner =
            build.invocation_authorities[i].prepared_owner.lock();
        const auto consumed_owner =
            build.invocation_authorities[i].consumed_owner.lock();
        const auto& current = candidate.invocation_authorities[i];
        TORCH_CHECK(
            same_shared_owner_identity(prepared_owner,
                                       current.prepared_owner) &&
                same_shared_owner_identity(consumed_owner,
                                           current.consumed_owner) &&
                *current.prepared_owner == current.expected_generation &&
                *current.consumed_owner < current.expected_generation,
            "outer fast invocation authority owner/order drifted from BUILD");
    }

    OuterFastPreparedCommit prepared;
    prepared.valid = true;
    prepared.fast = true;
    prepared.canonical_census =
        policy_total_stats(kernel_register_census_policy_);
    prepared.mutable_slot_count = build.mutable_dma_slots.size();
    prepared.mutable_occurrence_count = build.mutable_occurrence_count;

    for (size_t component_index = 0;
         component_index < build.components.size(); ++component_index) {
        const auto& sealed = build.components[component_index];
        const auto& current = candidate.components[component_index];
        TORCH_CHECK(sealed.owners.size() == current.owners.size() &&
                        sealed.key_cache.size() == current.key_cache.size() &&
                        sealed.value_cache.size() ==
                            current.value_cache.size(),
                    "outer fast component catalog shape drifted");
        for (size_t i = 0; i < sealed.owners.size(); ++i) {
            const auto retained_owner = sealed.owners[i].owner.lock();
            TORCH_CHECK(
                retained_owner != nullptr && current.owners[i].owner != nullptr &&
                    retained_owner.get() == current.owners[i].owner.get() &&
                    sealed.owners[i].expected_generation ==
                        current.owners[i].expected_generation &&
                    *retained_owner == sealed.owners[i].expected_generation,
                "outer fast component owner identity/generation drifted");
            prepared.invocation_owner_stamps.push_back(current.owners[i]);
        }
        for (size_t i = 0; i < sealed.key_cache.size(); ++i) {
            check_kernel_register_outer_fast_tensor_exact(
                sealed.key_cache[i], current.key_cache[i],
                "outer fast K-cache catalog");
            check_kernel_register_outer_fast_tensor_exact(
                sealed.value_cache[i], current.value_cache[i],
                "outer fast V-cache catalog");
            prepared.invocation_owners.push_back(current.key_cache[i]);
            prepared.invocation_owners.push_back(current.value_cache[i]);
        }
    }
    for (const auto& authority : candidate.invocation_authorities) {
        TORCH_CHECK(authority.prepared_owner != nullptr &&
                        authority.consumed_owner != nullptr &&
                        *authority.prepared_owner ==
                            authority.expected_generation &&
                        *authority.consumed_owner <
                            authority.expected_generation,
                    "outer fast invocation-only authority is stale");
        prepared.invocation_authorities.push_back(authority);
    }

    // Fixed DMA owners are graph-retained strong Tensor+Storage seals.  The
    // caller does not re-enumerate thousands of fixed occurrences on replay.
    for (const auto& owner : build.fixed_dma_owners) {
        check_kernel_register_outer_fast_tensor_live(
            owner.owner, "outer fast retained fixed DMA owner");
        check_kernel_register_outer_fast_dma_range(
            owner.owner, owner.byte_begin, owner.byte_count);
        prepared.invocation_owners.push_back(owner.owner);
        if (owner.side == GraphOuterFastDmaSide::Destination) {
            prepared.normal_visibility_owners.push_back(owner.owner);
            prepared.boundary_visibility.push_back(
                const_cast<void*>(owner.owner.data_ptr));
            prepared.force_visibility_owners.push_back(owner.owner);
            prepared.force_visibility.push_back(
                const_cast<void*>(owner.owner.data_ptr));
            prepared.force_post_visibility_owners.push_back(owner.owner);
            prepared.force_post_visibility.push_back(
                const_cast<void*>(owner.owner.data_ptr));
        }
    }

    std::vector<size_t> visibility_ordinals;
    visibility_ordinals.reserve(candidate.invocation_visibility.size());
    for (const auto& current : candidate.invocation_visibility) {
        size_t matched_ordinal = build.fixed_dma_owners.size();
        size_t match_count = 0;
        for (size_t i = 0; i < build.fixed_dma_owners.size(); ++i) {
            const auto& fixed = build.fixed_dma_owners[i];
            if (fixed.side == GraphOuterFastDmaSide::Source &&
                fixed.owner.tensor_impl == current.tensor_impl &&
                fixed.owner.storage_impl == current.storage_impl &&
                fixed.owner.topology_id == current.topology_id) {
                matched_ordinal = i;
                ++match_count;
            }
        }
        TORCH_CHECK(match_count == 1,
                    "outer fast invocation visibility must resolve to exactly "
                    "one BUILD-derived fixed Source owner");
        check_kernel_register_outer_fast_tensor_exact(
            build.fixed_dma_owners[matched_ordinal].owner, current,
            "outer fast invocation visibility");
        visibility_ordinals.push_back(matched_ordinal);
        prepared.normal_visibility_owners.push_back(current);
        prepared.boundary_visibility.push_back(
            const_cast<void*>(current.data_ptr));
    }
    std::sort(visibility_ordinals.begin(), visibility_ordinals.end());
    TORCH_CHECK(
        std::adjacent_find(visibility_ordinals.begin(),
                           visibility_ordinals.end()) ==
                visibility_ordinals.end() &&
            visibility_ordinals ==
                build.invocation_visibility_fixed_owner_ordinals,
        "outer fast invocation visibility differs from BUILD");

    TORCH_CHECK(candidate.fast_mutable_dma_bindings.size() ==
                    build.mutable_dma_slots.size(),
                "outer fast mutable DMA bindings do not exact-cover the BUILD "
                "slot universe");
    std::vector<bool> mutable_seen(build.mutable_dma_slots.size(), false);
    prepared.force_visibility.reserve(build.mutable_dma_slots.size());
    for (const auto& binding : candidate.fast_mutable_dma_bindings) {
        size_t ordinal = build.mutable_dma_slots.size();
        for (size_t i = 0; i < build.mutable_dma_slots.size(); ++i) {
            if (build.mutable_dma_slots[i].live_base == binding.live_base) {
                ordinal = i;
                break;
            }
        }
        TORCH_CHECK(ordinal < build.mutable_dma_slots.size() &&
                        !mutable_seen[ordinal],
                    "outer fast mutable DMA slot is extra or duplicated");
        const auto& slot = build.mutable_dma_slots[ordinal];
        TORCH_CHECK(slot.side == binding.side &&
                        slot.owner_topology_id == binding.owner.topology_id &&
                        slot.byte_begin == binding.byte_begin &&
                        slot.byte_count == binding.byte_count,
                    "outer fast mutable DMA owner topology/range drifted");
        check_kernel_register_outer_fast_tensor_live(
            binding.owner, "outer fast current mutable DMA owner");
        if (slot.side == GraphOuterFastDmaSide::Destination &&
            slot.last_successful_destination.has_value()) {
            TORCH_CHECK(
                slot.last_successful_destination->tensor_impl !=
                        binding.owner.tensor_impl &&
                    slot.last_successful_destination->storage_impl !=
                        binding.owner.storage_impl,
                "outer fast mutable destination must use a fresh TensorImpl "
                "and StorageImpl each invocation");
        }
        mutable_seen[ordinal] = true;
        prepared.slot_writes.emplace_back(
            binding.live_base, binding.owner.dev_addr);
        prepared.invocation_owners.push_back(binding.owner);
        if (slot.side == GraphOuterFastDmaSide::Source) {
            prepared.force_visibility_owners.push_back(binding.owner);
            prepared.force_visibility.push_back(
                const_cast<void*>(binding.owner.data_ptr));
        }
        if (slot.side == GraphOuterFastDmaSide::Destination) {
            prepared.normal_visibility_owners.push_back(binding.owner);
            prepared.boundary_visibility.push_back(
                const_cast<void*>(binding.owner.data_ptr));
            prepared.force_visibility_owners.push_back(binding.owner);
            prepared.force_visibility.push_back(
                const_cast<void*>(binding.owner.data_ptr));
            prepared.force_post_visibility_owners.push_back(binding.owner);
            prepared.force_post_visibility.push_back(
                const_cast<void*>(binding.owner.data_ptr));
            prepared.mutable_successful_owners.emplace_back(
                ordinal, binding.owner);
        }
    }
    TORCH_CHECK(std::all_of(mutable_seen.begin(), mutable_seen.end(),
                            [](bool value) { return value; }),
                "outer fast mutable DMA exact-cover is incomplete");

    TORCH_CHECK(boundary_flush_ptrs_.empty(),
                "outer fast replay must prepare boundary visibility before "
                "publishing any flush entries");
    auto sort_unique_ptrs = [](std::vector<void*>& ptrs) {
        std::sort(ptrs.begin(), ptrs.end());
        ptrs.erase(std::unique(ptrs.begin(), ptrs.end()), ptrs.end());
    };
    sort_unique_ptrs(prepared.boundary_visibility);
    sort_unique_ptrs(prepared.force_visibility);
    sort_unique_ptrs(prepared.force_post_visibility);
    prepared.retained_validated =
        validate_kernel_register_retained_occurrences();

    TORCH_CHECK(next_kernel_register_outer_fast_ticket_nonce_ !=
                    std::numeric_limits<uint64_t>::max(),
                "outer fast replay validation ticket nonce exhausted");
    const uint64_t nonce =
        ++next_kernel_register_outer_fast_ticket_nonce_;
    kernel_register_outer_fast_prepared_.emplace(std::move(prepared));
    kernel_register_outer_fast_ticket_nonce_ = nonce;
    kernel_register_outer_fast_validation_armed_ = true;
    return GraphKernelRegisterOuterFastTicket(this, nonce);
}

void RpuKernelGraph::commit_kernel_register_outer_fast_replay(
        GraphKernelRegisterOuterFastTicket&& ticket) {
    TORCH_CHECK(ticket.graph_ == this && ticket.nonce_ != 0 &&
                    ticket.nonce_ == kernel_register_outer_fast_ticket_nonce_ &&
                    kernel_register_outer_fast_validation_armed_ &&
                    kernel_register_outer_fast_prepared_.has_value() &&
                    kernel_register_outer_fast_prepared_->valid &&
                    kernel_register_outer_fast_prepared_->fast &&
                    state_ == State::REPLAYING && cursor_ == 0 &&
                    kernel_register_census_active_ &&
                    !kernel_register_census_poisoned_ &&
                    boundary_flush_ptrs_.empty(),
                "outer fast replay commit ticket is stale or not authoritative");
    const auto& candidate = kernel_register_outer_fast_candidate_;
    const auto& build = kernel_register_outer_fast_build_state_;
    TORCH_CHECK(candidate.engaged &&
                    candidate.mode == OuterFastCandidateMode::Fast &&
                    candidate.physical_digest.has_value() &&
                    *candidate.physical_digest == build.physical_digest &&
                    candidate.build_generation == build_generation_ &&
                    candidate.build_topology_hash == build_topology_hash_ &&
                    candidate.policy_digest ==
                        kernel_register_census_policy_digest_ &&
                    candidate.plan_hash == kernel_register_census_plan_hash_ &&
                    candidate.live_epoch == kernel_register_census_live_epoch_,
                "outer fast replay authority drifted after validation");

    const auto& prepared = *kernel_register_outer_fast_prepared_;
    for (const auto& stamp : prepared.invocation_owner_stamps) {
        TORCH_CHECK(stamp.owner != nullptr &&
                        *stamp.owner == stamp.expected_generation,
                    "outer fast owner generation drifted after validation");
    }
    for (const auto& authority : prepared.invocation_authorities) {
        TORCH_CHECK(authority.prepared_owner != nullptr &&
                        authority.consumed_owner != nullptr &&
                        *authority.prepared_owner ==
                            authority.expected_generation &&
                        *authority.consumed_owner <
                            authority.expected_generation,
                    "outer fast invocation authority drifted after validation");
    }
    for (const auto& owner : prepared.invocation_owners) {
        check_kernel_register_outer_fast_tensor_live(
            owner, "outer fast tensor owner after validation");
    }
    for (const auto& [slot, value] : prepared.slot_writes) {
        TORCH_CHECK(slot != nullptr && value != 0,
                    "outer fast mutable slot write became invalid");
    }
    // All allocations and all throwing validation end above.  The private
    // aggregate commit below consists only of swaps, scalar writes, flag
    // publication, and cursor publication (last).
    commit_kernel_register_outer_fast_prepared(true);
    ticket.release();
}

void RpuKernelGraph::cancel_kernel_register_outer_fast_ticket(
        uint64_t nonce) noexcept {
    if (!kernel_register_outer_fast_validation_armed_ || nonce == 0 ||
        nonce != kernel_register_outer_fast_ticket_nonce_) {
        return;
    }
    kernel_register_outer_fast_prepared_.reset();
    kernel_register_outer_fast_validation_armed_ = false;
    kernel_register_outer_fast_ticket_nonce_ = 0;
    cursor_ = 0;
    cancel_kernel_register_census();
}

void RpuKernelGraph::consume_kernel_register_dma_burst(DmaNodeData& data) {
    if (!kernel_register_census_active_) return;
    auto& candidate = kernel_register_outer_fast_candidate_;
    if (!candidate.engaged &&
        !kernel_register_outer_fast_build_state_.staged) {
        return;
    }
    TORCH_CHECK(candidate.engaged &&
                    candidate.mode == OuterFastCandidateMode::Ordinary &&
                    candidate.open_ordinary_dma_burst.has_value() &&
                    !kernel_register_outer_fast_validation_armed_,
                "schema-5 owner-bound DMA was emitted without one open typed "
                "burst");
    const size_t burst_ordinal =
        *candidate.open_ordinary_dma_burst;
    TORCH_INTERNAL_ASSERT(
        burst_ordinal < candidate.ordinary_dma_bursts.size());
    auto& burst = candidate.ordinary_dma_bursts[burst_ordinal];
    TORCH_CHECK(data.variant == burst.variant &&
                    burst.observed_occurrence_count <
                        burst.expected_occurrence_count,
                "schema-5 DMA variant/count differs from the staged burst");

    uint64_t ddr_addr = 0;
    size_t owner_offset = 0;
    if (data.variant == DmaNodeData::Variant::Fixed) {
        ddr_addr = burst.side == GraphOuterFastDmaSide::Source
            ? data.src_addr : data.dst_addr;
        TORCH_CHECK(ddr_addr >= burst.owner.dev_addr,
                    "schema-5 fixed DMA address precedes its Tensor owner");
        const uint64_t delta = ddr_addr - burst.owner.dev_addr;
        TORCH_CHECK(delta <= std::numeric_limits<size_t>::max(),
                    "schema-5 fixed DMA Tensor offset overflows size_t");
        owner_offset = static_cast<size_t>(delta);
    } else {
        TORCH_CHECK(data.live_base == burst.live_base &&
                        data.live_offset >= 0 &&
                        ((burst.side == GraphOuterFastDmaSide::Source &&
                          data.variant == DmaNodeData::Variant::MutableSrc) ||
                         (burst.side ==
                              GraphOuterFastDmaSide::Destination &&
                          data.variant == DmaNodeData::Variant::MutableDst)),
                    "schema-5 mutable DMA live slot/direction drifted");
        owner_offset = static_cast<size_t>(data.live_offset);
    }
    TORCH_CHECK(owner_offset >= burst.byte_begin &&
                    owner_offset - burst.byte_begin <= burst.byte_count &&
                    data.bytes <= burst.byte_count -
                        (owner_offset - burst.byte_begin),
                "schema-5 DMA occurrence exceeds its staged Tensor range");

    data.outer_fast_owner_bound = true;
    data.outer_fast_ddr_side = burst.side;
    data.outer_fast_owner_ordinal = state_ == State::REPLAYING
        ? burst.owner_ordinal : burst_ordinal;
    data.outer_fast_owner_topology_id = burst.owner.topology_id;
    data.outer_fast_owner_byte_offset = owner_offset;
    ++burst.observed_occurrence_count;
    if (burst.observed_occurrence_count ==
        burst.expected_occurrence_count) {
        candidate.open_ordinary_dma_burst.reset();
    }
}

void RpuKernelGraph::check_kernel_register_dma_burst_closed(
        const char* operation) const {
    if (!kernel_register_census_active_) return;
    TORCH_CHECK(
        !kernel_register_outer_fast_candidate_.open_ordinary_dma_burst
             .has_value(),
        operation,
        ": a schema-5 typed DMA burst cannot cross a Kernel boundary");
}

void RpuKernelGraph::complete_kernel_register_census() {
    TORCH_CHECK(kernel_register_census_active_ &&
                    !kernel_register_census_complete_ &&
                    !kernel_register_census_poisoned_ &&
                    !pending_kernel_register_census_.has_value(),
                "typed DDR-register census cannot complete with pending or "
                "poisoned state");
    TORCH_CHECK(kernel_register_census_next_phase_ + 1 ==
                    kernel_register_census_policy_.phases.size(),
                "typed DDR-register census requires every non-final phase "
                "checkpoint before completion");
    const auto& final_phase = kernel_register_census_policy_.phases.back();
    const auto final_delta = stats_delta(
        kernel_register_census_stats_,
        kernel_register_census_phase_begin_stats_);
    check_stats_counts(final_delta, final_phase.stats, final_phase.name);
    TORCH_CHECK(
        kernel_register_census_phase_begin_stats_.ordered_node_kind_hash ==
                final_phase.stats.ordered_node_kind_hash &&
            kernel_register_census_phase_begin_stats_.ordered_kernel_hash ==
                final_phase.stats.ordered_kernel_hash,
        "typed DDR-register census final phase ordered stream mismatch; "
        "actual=0x", std::hex,
        kernel_register_census_phase_begin_stats_.ordered_node_kind_hash,
        "/0x",
        kernel_register_census_phase_begin_stats_.ordered_kernel_hash,
        " expected=0x", final_phase.stats.ordered_node_kind_hash, "/0x",
        final_phase.stats.ordered_kernel_hash, std::dec);

    const auto& policy = kernel_register_census_policy_;
    GraphKernelRegisterCensusStats expected_total;
    expected_total.node_count = policy.expected_node_count;
    expected_total.dma_count = policy.expected_dma_count;
    expected_total.barrier_count = policy.expected_barrier_count;
    expected_total.kernel_count = policy.expected_kernel_count;
    expected_total.typed_kernel_count =
        policy.expected_typed_kernel_count;
    expected_total.no_ddr_kernel_count =
        policy.expected_no_ddr_kernel_count;
    expected_total.operand_count = policy.expected_operand_count;
    expected_total.weight_operand_count =
        policy.expected_weight_operand_count;
    expected_total.key_cache_operand_count =
        policy.expected_key_cache_operand_count;
    expected_total.value_cache_operand_count =
        policy.expected_value_cache_operand_count;
    expected_total.semantic_input_operand_count =
        policy.expected_semantic_input_operand_count;
    check_stats_counts(kernel_register_census_stats_, expected_total,
                       "full window");
    TORCH_CHECK(
        kernel_register_census_stats_.ordered_node_kind_hash ==
                policy.expected_ordered_node_kind_hash &&
            kernel_register_census_stats_.ordered_kernel_hash ==
                policy.expected_ordered_kernel_hash,
        "typed DDR-register full-window ordered stream mismatch; actual=0x",
        std::hex, kernel_register_census_stats_.ordered_node_kind_hash,
        "/0x", kernel_register_census_stats_.ordered_kernel_hash,
        " expected=0x", policy.expected_ordered_node_kind_hash, "/0x",
        policy.expected_ordered_kernel_hash, std::dec);
    for (size_t i = 0; i < policy.rules.size(); ++i) {
        TORCH_CHECK(kernel_register_census_rule_counts_[i] ==
                        policy.rules[i].expected_count,
                    "typed DDR-register per-rule count mismatch at rule ", i);
    }
    TORCH_CHECK(kernel_register_census_node_begin_ == 0 &&
                    kernel_register_census_stats_.node_count == nodes_.size() &&
                    cursor_ == (state_ == State::REPLAYING
                                    ? nodes_.size()
                                    : 0),
                "typed DDR-register census does not cover the exact full "
                "Graph window");
    build_kernel_register_retained_catalog();
    complete_kernel_register_outer_fast_ordinary();
    kernel_register_host_reemitted_node_count_ =
        kernel_register_census_stats_.node_count;
    kernel_register_host_reemitted_kernel_count_ =
        kernel_register_census_stats_.kernel_count;
    kernel_register_census_complete_ = true;
    kernel_register_census_invocation_armed_ = true;
    has_kernel_register_census_nodes_ = true;
}

void RpuKernelGraph::build_kernel_register_retained_catalog() {
    const auto expected =
        kernel_register_census_policy_.expected_operand_count;
    if (state_ == State::REPLAYING) {
        TORCH_CHECK(kernel_register_retained_catalog_.size() == expected,
                    "typed DDR-register retained catalog differs from BUILD");
        return;
    }

    TORCH_CHECK(state_ == State::RECORDING &&
                    kernel_register_retained_catalog_.empty(),
                "typed DDR-register retained catalog must be built once");
    std::vector<RetainedRegisterRef> catalog;
    catalog.reserve(expected);
    for (size_t node_index = 0; node_index < nodes_.size(); ++node_index) {
        const auto& node = nodes_[node_index];
        if (node.kind != GraphNodeKind::Kernel) continue;
        const auto& kernel_node = node.as_kernel();
        TORCH_CHECK(kernel_node.register_census.has_value() &&
                        kernel_node.kernel_idx < kernels_.size(),
                    "typed DDR-register catalog found an untyped Kernel");
        const auto& census = *kernel_node.register_census;
        if (census.kind != GraphKernelRegisterCensus::Kind::TypedDdr) {
            TORCH_CHECK(census.operands.empty(),
                        "typed DDR-register no-DDR proof retained operands");
            continue;
        }
        for (size_t operand_index = 0;
             operand_index < census.operands.size(); ++operand_index) {
            catalog.push_back(RetainedRegisterRef{
                node_index, operand_index, kernel_node.kernel_idx});
        }
    }
    TORCH_CHECK(catalog.size() == expected,
                "typed DDR-register retained catalog count drifted");
    kernel_register_retained_catalog_.swap(catalog);
}

size_t RpuKernelGraph::validate_kernel_register_retained_occurrences() const {
    TORCH_CHECK(kernel_register_retained_catalog_.size() ==
                    kernel_register_census_policy_.expected_operand_count,
                "typed DDR-register retained catalog count drifted");
    // Tensor copies share one mutation domain; distinct view TensorImpls do
    // not.  The duplicate snapshot check below rejects any set_ rebind.
    std::unordered_map<const c10::TensorImpl*,
                       const GraphDdrRegisterOccurrence*>
        validated_owners;
    validated_owners.reserve(kernel_register_retained_catalog_.size());
    for (const auto& ref : kernel_register_retained_catalog_) {
        TORCH_CHECK(ref.node_index < nodes_.size() &&
                        nodes_[ref.node_index].kind == GraphNodeKind::Kernel,
                    "typed DDR-register retained catalog node is stale");
        const auto& kernel_node = nodes_[ref.node_index].as_kernel();
        TORCH_CHECK(kernel_node.register_census.has_value() &&
                        kernel_node.kernel_idx == ref.kernel_index &&
                        ref.kernel_index < kernels_.size(),
                    "typed DDR-register retained catalog Kernel is stale");
        const auto& census = *kernel_node.register_census;
        TORCH_CHECK(census.kind ==
                            GraphKernelRegisterCensus::Kind::TypedDdr &&
                        ref.operand_index < census.operands.size(),
                    "typed DDR-register retained catalog operand is stale");
        const auto& operand = census.operands[ref.operand_index];
        auto& kernel = *kernels_[ref.kernel_index];
        const at::Tensor& tensor = operand.tensor_owner;
        validate_ddr_register_role_dtype(
            operand.abi.role, operand.dtype);
        validate_ddr_register_role_dtype(
            operand.abi.role, tensor.scalar_type());
        const auto [owner_it, inserted] =
            validated_owners.emplace(operand.tensor_impl, &operand);
        if (!inserted) {
            check_same_retained_owner_snapshot(*owner_it->second, operand);
        } else {
            TORCH_CHECK(tensor.defined() &&
                            tensor.unsafeGetTensorImpl() ==
                                operand.tensor_impl &&
                            static_cast<bool>(operand.storage_owner) &&
                            operand.storage_owner.unsafeGetStorageImpl() ==
                                operand.storage_impl &&
                            tensor.storage().unsafeGetStorageImpl() ==
                                operand.storage_impl &&
                            tensor.const_data_ptr() == operand.data_ptr &&
                            operand.storage_owner.data() ==
                                operand.storage_data_ptr &&
                            tensor.scalar_type() == operand.dtype &&
                            tensor.device() == operand.device &&
                            tensor.layout() == operand.layout &&
                            tensor.storage_offset() == operand.storage_offset &&
                            tensor.nbytes() == operand.tensor_nbytes &&
                            operand.storage_owner.nbytes() ==
                                operand.storage_nbytes &&
                            tensor.dim() ==
                                static_cast<int64_t>(operand.sizes.size()) &&
                            std::equal(tensor.sizes().begin(),
                                       tensor.sizes().end(),
                                       operand.sizes.begin()) &&
                            std::equal(tensor.strides().begin(),
                                       tensor.strides().end(),
                                       operand.strides.begin()) &&
                            tensor_topology_id(
                                tensor, operand.tensor_nbytes,
                                operand.storage_nbytes) ==
                                operand.tensor_topology_id &&
                            rhino_lkn::RpuGetDevAddr(
                                const_cast<void*>(operand.data_ptr)) ==
                                operand.dev_addr,
                        "typed DDR-register retained TensorImpl/StorageImpl/"
                        "layout/address provenance drifted");
        }
        const uint16_t expected_lo = static_cast<uint16_t>(
            operand.dev_addr >> 8);
        const uint16_t expected_hi = static_cast<uint16_t>(
            operand.dev_addr >> 24);
        const uint16_t actual_lo =
            read_kernel_reg(kernel, operand.abi.reg_lo);
        const uint16_t actual_hi =
            read_kernel_reg(kernel, operand.abi.reg_hi);
        TORCH_CHECK(actual_lo == expected_lo && actual_hi == expected_hi,
                    "typed DDR-register retained address pair ",
                    operand.abi.reg_lo, "/", operand.abi.reg_hi,
                    " no longer contains its BUILD address");
    }
    return kernel_register_retained_catalog_.size();
}

void RpuKernelGraph::verify_kernel_register_outer_fast_preflight() const {
    if (!kernel_register_outer_fast_build_state_.staged) return;
    const auto& build = kernel_register_outer_fast_build_state_;
    const auto& candidate = kernel_register_outer_fast_candidate_;
    TORCH_CHECK(candidate.engaged && candidate.physical_digest.has_value() &&
                    *candidate.physical_digest == build.physical_digest &&
                    kernel_register_outer_fast_invocation_physical_digest_ ==
                        build.physical_digest &&
                    candidate.build_generation == build_generation_ &&
                    candidate.build_topology_hash == build_topology_hash_ &&
                    candidate.policy_digest ==
                        kernel_register_census_policy_digest_ &&
                    candidate.plan_hash == kernel_register_census_plan_hash_ &&
                    candidate.live_epoch == kernel_register_census_live_epoch_ &&
                    (state_ != State::REPLAYING || build.committed),
                "outer fast end authority is stale or differs from BUILD");
    for (const auto& stamp :
         kernel_register_outer_fast_invocation_owner_stamps_) {
        TORCH_CHECK(stamp.owner != nullptr &&
                        *stamp.owner == stamp.expected_generation,
                    "outer fast end owner generation is stale");
    }
    for (const auto& authority :
         kernel_register_outer_fast_invocation_authorities_) {
        TORCH_CHECK(authority.prepared_owner != nullptr &&
                        authority.consumed_owner != nullptr &&
                        *authority.prepared_owner ==
                            authority.expected_generation &&
                        *authority.consumed_owner <
                            authority.expected_generation,
                    "outer fast end invocation authority is stale or already "
                    "consumed");
    }
    for (const auto& owner : kernel_register_outer_fast_invocation_owners_) {
        check_kernel_register_outer_fast_tensor_live(
            owner, "outer fast end tensor owner");
    }
    for (const auto& owner :
         kernel_register_outer_fast_normal_visibility_owners_) {
        check_kernel_register_outer_fast_tensor_live(
            owner, "outer fast end normal-visibility owner");
        TORCH_CHECK(std::binary_search(
                        boundary_flush_ptrs_.begin(),
                        boundary_flush_ptrs_.end(),
                        const_cast<void*>(owner.data_ptr)),
                    "outer fast end normal boundary visibility is incomplete");
    }
    for (const auto& owner :
         kernel_register_outer_fast_force_visibility_owners_) {
        check_kernel_register_outer_fast_tensor_live(
            owner, "outer fast end force-visibility owner");
        TORCH_CHECK(std::binary_search(
                        kernel_register_outer_fast_invocation_force_visibility_
                            .begin(),
                        kernel_register_outer_fast_invocation_force_visibility_
                            .end(),
                        const_cast<void*>(owner.data_ptr)),
                    "outer fast end force visibility is incomplete");
    }
    for (const auto& owner :
         kernel_register_outer_fast_force_post_visibility_owners_) {
        check_kernel_register_outer_fast_tensor_live(
            owner, "outer fast end force-post-visibility owner");
        TORCH_CHECK(std::binary_search(
                        kernel_register_outer_fast_invocation_force_post_visibility_
                            .begin(),
                        kernel_register_outer_fast_invocation_force_post_visibility_
                            .end(),
                        const_cast<void*>(owner.data_ptr)),
                    "outer fast end force-post visibility is incomplete");
    }
    for (const auto& owner : build.fixed_dma_owners) {
        check_kernel_register_outer_fast_tensor_live(
            owner.owner, "outer fast end retained fixed DMA owner");
        check_kernel_register_outer_fast_dma_range(
            owner.owner, owner.byte_begin, owner.byte_count);
    }
    for (const auto& [slot, value] :
         kernel_register_outer_fast_invocation_slot_values_) {
        TORCH_CHECK(slot != nullptr && value != 0 && *slot == value,
                    "outer fast end mutable DMA live slot drifted");
    }
}

void RpuKernelGraph::verify_kernel_register_outer_fast_launch_preflight()
        const {
    if (!kernel_register_outer_fast_build_state_.staged) return;
    verify_kernel_register_outer_fast_preflight();
    TORCH_CHECK(validate_kernel_register_retained_occurrences() ==
                    kernel_register_outer_fast_retained_validated_,
                "outer fast post-flush retained register count drifted");
}

void RpuKernelGraph::flush_kernel_register_outer_fast_force_visibility()
        const noexcept {
    if (!kernel_register_outer_fast_build_state_.staged ||
        !kernel_register_census_invocation_armed_ ||
        kernel_register_outer_fast_invocation_force_visibility_.empty() ||
        !g_rpu_ddr_flush_force_enabled) {
        return;
    }
    for (void* ptr :
         kernel_register_outer_fast_invocation_force_visibility_) {
        rhino_lkn::RpuDdrFlush(ptr);
    }
}

void RpuKernelGraph::flush_kernel_register_outer_fast_force_post_visibility()
        const noexcept {
    if (!kernel_register_outer_fast_build_state_.staged ||
        !kernel_register_census_invocation_armed_ ||
        kernel_register_outer_fast_invocation_force_post_visibility_.empty() ||
        !g_rpu_ddr_flush_force_enabled) {
        return;
    }
    for (void* ptr :
         kernel_register_outer_fast_invocation_force_post_visibility_) {
        rhino_lkn::RpuDdrFlush(ptr);
    }
}

void RpuKernelGraph::verify_kernel_register_end_preflight() const {
    if (!has_kernel_register_census_nodes_ &&
        !kernel_register_census_active_) return;
    TORCH_CHECK(kernel_register_census_active_ &&
                    kernel_register_census_complete_ &&
                    kernel_register_census_invocation_armed_ &&
                    !kernel_register_census_poisoned_ &&
                    !pending_kernel_register_census_.has_value() &&
                    kernel_register_census_live_epoch_ != 0 &&
                    kernel_register_census_plan_hash_ != 0 &&
                    kernel_register_census_stats_.node_count == nodes_.size(),
                "typed DDR-register end preflight is incomplete/unarmed");
    TORCH_CHECK(state_ != State::REPLAYING || cursor_ == nodes_.size(),
                "typed DDR-register REPLAY end preflight has an incomplete "
                "cursor");
    TORCH_CHECK(state_ != State::RECORDING ||
                    (replayable_ && pending_signature_.has_value()),
                "typed DDR-register BUILD requires a replayable signature");
    TORCH_CHECK(state_ != State::REPLAYING ||
                    !force_oneshot_on_replay_enabled(),
                "typed DDR-register REPLAY forbids "
                "RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY before any flush or SDK "
                "submission");
    if (!kernel_register_outer_fast_hit_) {
        TORCH_CHECK(validate_kernel_register_retained_occurrences() ==
                        kernel_register_census_policy_.expected_operand_count,
                    "typed DDR-register end retained operand count drifted");
    }
    verify_kernel_register_outer_fast_preflight();
}

void RpuKernelGraph::cancel_kernel_register_census() noexcept {
    if (!kernel_register_census_active_) return;
    pending_kernel_register_census_.reset();
    kernel_register_outer_fast_prepared_.reset();
    kernel_register_outer_fast_validation_armed_ = false;
    kernel_register_outer_fast_ticket_nonce_ = 0;
    kernel_register_census_active_ = false;
    kernel_register_census_complete_ = false;
    kernel_register_census_invocation_armed_ = false;
    kernel_register_census_poisoned_ = true;
}

void RpuKernelGraph::clear_kernel_register_invocation_state() noexcept {
    pending_kernel_register_census_.reset();
    kernel_register_census_active_ = false;
    kernel_register_census_complete_ = false;
    kernel_register_census_poisoned_ = false;
    kernel_register_census_invocation_armed_ = false;
    kernel_register_census_live_epoch_ = 0;
    kernel_register_census_node_begin_ = 0;
    kernel_register_census_next_phase_ = 0;
    next_kernel_register_writer_ticket_ = 0;
    kernel_register_census_stats_ = {};
    kernel_register_census_phase_begin_stats_ = {};
    kernel_register_census_rule_counts_.clear();
    kernel_register_outer_fast_candidate_ = {};
    kernel_register_outer_fast_prepared_.reset();
    kernel_register_outer_fast_validation_armed_ = false;
    kernel_register_outer_fast_hit_ = false;
    kernel_register_outer_fast_ticket_nonce_ = 0;
    kernel_register_outer_fast_invocation_physical_digest_ = 0;
    kernel_register_outer_fast_invocation_owners_.clear();
    kernel_register_outer_fast_invocation_owner_stamps_.clear();
    kernel_register_outer_fast_invocation_authorities_.clear();
    kernel_register_outer_fast_normal_visibility_owners_.clear();
    kernel_register_outer_fast_force_visibility_owners_.clear();
    kernel_register_outer_fast_force_post_visibility_owners_.clear();
    kernel_register_outer_fast_invocation_slot_values_.clear();
    kernel_register_outer_fast_invocation_force_visibility_.clear();
    kernel_register_outer_fast_invocation_force_post_visibility_.clear();
    kernel_register_outer_fast_mutable_successful_owners_.clear();
    kernel_register_outer_fast_retained_validated_ = 0;
    kernel_register_outer_fast_mutable_slot_count_ = 0;
    kernel_register_outer_fast_mutable_occurrence_count_ = 0;
    kernel_register_host_reemitted_node_count_ = 0;
    kernel_register_host_reemitted_kernel_count_ = 0;
}

void RpuKernelGraph::clear_kernel_register_build_state() noexcept {
    clear_kernel_register_invocation_state();
    has_kernel_register_census_nodes_ = false;
    kernel_register_census_build_committed_ = false;
    kernel_register_census_plan_hash_ = 0;
    kernel_register_census_policy_digest_ = 0;
    kernel_register_census_policy_ = {};
    kernel_register_retained_catalog_.clear();
    kernel_register_outer_fast_build_state_ = {};
}

void RpuKernelGraph::check_kernel_register_node_emission(
        const char* operation) const {
    if (!kernel_register_census_active_ &&
        !has_kernel_register_census_nodes_) return;
    TORCH_CHECK(kernel_register_census_active_ &&
                    !kernel_register_census_complete_ &&
                    !kernel_register_census_poisoned_,
                operation,
                ": forbidden outside the open typed DDR-register census "
                "window");
}
