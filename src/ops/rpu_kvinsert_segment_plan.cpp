#include "rpu_kvinsert_segment_plan.h"

#include <c10/util/Exception.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <tuple>

namespace {

constexpr int64_t kRowAlignment = 16;
constexpr int64_t kMaxKernelRows =
    static_cast<int64_t>(std::numeric_limits<uint16_t>::max());
constexpr int64_t kMaxKernelPosition =
    static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
constexpr uint32_t kKnownRouteMask =
    (1U << static_cast<uint8_t>(KvInsertRoute::V2)) |
    (1U << static_cast<uint8_t>(KvInsertRoute::ALIGNED_V16)) |
    (1U << static_cast<uint8_t>(KvInsertRoute::PAD16_V16)) |
    (1U << static_cast<uint8_t>(KvInsertRoute::HYBRID2)) |
    (1U << static_cast<uint8_t>(KvInsertRoute::HYBRID3));

uint64_t dependency_hash_append(uint64_t hash, const char* value) {
    while (*value != '\0') {
        hash ^= static_cast<uint8_t>(*value++);
        hash *= 1099511628211ULL;
    }
    hash ^= static_cast<uint8_t>('|');
    return hash * 1099511628211ULL;
}

int64_t current_dependency_identity() {
    uint64_t hash = 1469598103934665603ULL;
    hash = dependency_hash_append(hash, kKvInsertCostRuntimeRelease);
    hash = dependency_hash_append(hash, kKvInsertCostLaunchSha256);
    hash = dependency_hash_append(hash, kKvInsertCostOperatorSha256);
    return static_cast<int64_t>(hash &
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
}

// P7 certificates are external, immutable owner data, not compiled into the
// runtime payload whose digest they bind. Unmeasured callers remain
// CAPABILITY_RANKED rather than inheriting another owner's measurements.

const KvInsertCostCertificate& structural_cost_certificate() {
    static const KvInsertCostCertificate certificate{
        current_dependency_identity(), 0, false, {}, {}};
    return certificate;
}

bool has_capability(uint32_t capabilities, KvInsertCapability capability) {
    return (capabilities & static_cast<uint32_t>(capability)) != 0;
}

int64_t align16(int64_t rows) {
    return ((rows + kRowAlignment - 1) / kRowAlignment) * kRowAlignment;
}

uint32_t route_bit(KvInsertRoute route) {
    return 1U << static_cast<uint8_t>(route);
}

int64_t route_cost(const KvInsertSegmentPlan& plan,
                   const KvInsertCostCertificate& certificate) {
    if (certificate.sealed) {
        const auto route = static_cast<size_t>(plan.route());
        TORCH_CHECK(route < certificate.route_costs_ns.size() &&
                        certificate.route_costs_ns[route] > 0,
                    "KV-insert certificate has no measured route cost");
        return certificate.route_costs_ns[route];
    }
    int64_t score = 0;
    for (size_t i = 0; i < plan.segment_count(); ++i) {
        const KvInsertSegment& part = plan.segment(i);
        const int64_t grid = part.kernel == KvInsertKernel::V16
            ? part.rows / kRowAlignment : part.rows;
        TORCH_CHECK(grid <= std::numeric_limits<int64_t>::max() - score,
                    "KV-insert route cost overflows int64");
        score += grid;
    }
    return score;
}

void validate_partition(int num_cores, int64_t num_kv_heads,
                        int64_t head_dim) {
    TORCH_CHECK(num_cores > 0 && num_cores <= 8,
                "KV-insert segment plan: num_cores must be in [1, 8], got ",
                num_cores);
    TORCH_CHECK(num_kv_heads > 0 && head_dim > 0,
                "KV-insert segment plan: num_kv_heads and head_dim must be "
                "positive");
    // Complete-head V2 partitions (including NextDiT TP6) keep eight padded
    // cache lanes; split-head partitions retain the divisor-of-eight ABI.
    TORCH_CHECK(num_kv_heads % num_cores == 0 ||
                    (8 % num_cores == 0 && num_cores % num_kv_heads == 0),
                "KV-insert segment plan: num_kv_heads must be divisible by "
                "num_cores, or divide a num_cores that divides 8");
    TORCH_CHECK(num_kv_heads <=
                    std::numeric_limits<int64_t>::max() / head_dim,
                "KV-insert segment plan: KV partition size overflows int64");
    const int64_t total_kv_dim = num_kv_heads * head_dim;
    TORCH_CHECK(total_kv_dim % num_cores == 0,
                "KV-insert segment plan: KV dimension is not TP-partitionable");
    TORCH_CHECK(head_dim % kRowAlignment == 0,
                "KV-insert segment plan: head_dim must be 16-aligned");
    const int64_t kernel_heads = num_kv_heads >= num_cores
        ? num_kv_heads / num_cores
        : 1;
    const int64_t kernel_head_dim = num_kv_heads >= num_cores
        ? head_dim
        : total_kv_dim / num_cores;
    TORCH_CHECK(kernel_heads <= kMaxKernelRows &&
                    kernel_head_dim <= kMaxKernelRows,
                "KV-insert segment plan: kernel TP parameters exceed uint16");
}

bool v16_shape_ok(int64_t position, int64_t rows, int num_cores,
                  int64_t num_kv_heads, int64_t head_dim,
                  bool allow_non8_tp) {
    return position >= 0 && rows >= kRowAlignment &&
        position % kRowAlignment == 0 && rows % kRowAlignment == 0 &&
        // NON8_TP_V16 certifies only TP1/2/4, not every valid V2 partition.
        num_cores > 0 && num_cores <= 8 && 8 % num_cores == 0 &&
        (num_cores == 8 || allow_non8_tp) &&
        num_kv_heads > 0 && num_kv_heads % num_cores == 0 &&
        head_dim > 0 && head_dim % kRowAlignment == 0 &&
        // The 16-token byte stride must fit uint16 for this cache ABI.
        num_kv_heads / num_cores <= (kMaxKernelRows / 32) / head_dim;
}

void require_capability(uint32_t capabilities,
                        KvInsertCapability capability,
                        const char* name) {
    TORCH_CHECK(has_capability(capabilities, capability),
                "KV-insert segment plan: exact route requires certified ",
                name, " capability");
}

KvInsertSegment segment(KvInsertKernel kernel, int64_t base_position,
                        int64_t token_offset, int64_t rows) {
    return {kernel, base_position + token_offset, token_offset, rows};
}

void validate_padding(int64_t position, int64_t logical_rows,
                      int64_t physical_rows) {
    TORCH_CHECK(position % kRowAlignment == 0 &&
                    physical_rows == align16(logical_rows),
                "KV-insert segment plan: physical padding must be exact "
                "PAD16 at an aligned position");
}

}  // namespace

const KvInsertCostCertificate& rpu_kvinsert_cost_certificate() {
    return structural_cost_certificate();
}

KvInsertCostScope KvInsertCostCatalog::scope(
    int64_t site_id, int64_t graph_lifecycle) const {
    TORCH_CHECK(site_id > 0, "KV-insert cost scope requires a stable site ID");
    TORCH_CHECK(graph_lifecycle >= 1 && graph_lifecycle <= 4,
                "KV-insert cost scope requires an actual Graph lifecycle");
    if (!bound()) return {};
    KvInsertCostScope result = scope_;
    result.site_identity = site_id;
    result.graph_lifecycle = graph_lifecycle;
    return result;
}

KvInsertCostCatalog KvInsertCostCatalog::from_arguments(
    c10::ArrayRef<int64_t> identity,
    const std::string& artifact_sha256,
    c10::ArrayRef<int64_t> certificate_rows) {
    TORCH_CHECK(identity.size() == 3 && identity[0] > 0 &&
                    identity[1] > 0 && identity[2] > 0,
                "KV-insert catalog requires bound profile/dependency/runtime identities");
    TORCH_CHECK(certificate_rows.size() >= 2 && certificate_rows[0] == 2 &&
                    certificate_rows[1] >= 0 && certificate_rows[1] <= 4096,
                "KV-insert catalog has an unsupported schema/count");
    const size_t count = static_cast<size_t>(certificate_rows[1]);
    TORCH_CHECK(certificate_rows.size() ==
                    2 + count * kKvInsertCostCertificateWords,
                "KV-insert catalog has a truncated or trailing certificate row");
    TORCH_CHECK(count == 0 ? artifact_sha256.empty()
                          : artifact_sha256.size() == 64,
                "KV-insert catalog requires its full external artifact SHA256");
    int64_t artifact_identity = 0;
    for (size_t i = 0; i < artifact_sha256.size(); ++i) {
        const char ch = artifact_sha256[i];
        TORCH_CHECK((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'),
                    "KV-insert catalog SHA256 must be lowercase hexadecimal");
        if (i < 15) {
            artifact_identity = artifact_identity * 16 +
                (ch <= '9' ? ch - '0' : ch - 'a' + 10);
        }
    }
    if (artifact_identity == 0) artifact_identity = 1;

    KvInsertCostCatalog result;
    result.scope_ = {identity[0], 0, identity[1], identity[2]};
    result.artifact_sha256_ = artifact_sha256;
    result.certificates_.reserve(count);
    std::set<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t,
                        int64_t, int64_t, int64_t, int64_t>> domains;
    for (size_t i = 0; i < count; ++i) {
        const auto row = certificate_rows.slice(
            2 + i * kKvInsertCostCertificateWords,
            kKvInsertCostCertificateWords);
        TORCH_CHECK(row[0] == current_dependency_identity() &&
                        row[1] == artifact_identity && row[2] == identity[0] &&
                        row[3] > 0 && row[4] == identity[1] &&
                        row[5] == identity[2],
                    "KV-insert catalog certificate identity/domain mismatch");
        TORCH_CHECK(row[9] > 0 && row[9] <= 8 && row[12] > 0 &&
                        row[12] <= kKnownRouteMask &&
                        (static_cast<uint32_t>(row[12]) & ~kKnownRouteMask) == 0,
                    "KV-insert catalog has an invalid core count or route mask");
        TORCH_CHECK(row[19] >= 1 && row[19] <= 4,
                    "KV-insert catalog has an invalid Graph lifecycle");
        KvInsertCostCertificate certificate;
        certificate.dependency_identity = row[0];
        certificate.certificate_identity = row[1];
        certificate.sealed = true;
        certificate.domain = {
            {row[2], row[3], row[4], row[5], row[19]}, row[6], row[7], row[8],
            static_cast<int>(row[9]), row[10], row[11],
            static_cast<uint32_t>(row[12])};
        for (size_t route = 0; route < certificate.route_costs_ns.size(); ++route) {
            const int64_t cost = row[13 + route];
            TORCH_CHECK((row[12] & (1LL << route)) ? cost > 0 : cost == 0,
                        "KV-insert catalog must cover exactly its feasible route costs");
            certificate.route_costs_ns[route] = cost;
        }
        uint32_t capabilities = 0;
        const auto mask = certificate.domain.feasible_route_mask;
        if (mask & route_bit(KvInsertRoute::V2)) capabilities |= KV_INSERT_CAP_V2;
        if (mask & ~route_bit(KvInsertRoute::V2)) capabilities |= KV_INSERT_CAP_V16;
        if (row[8] != row[7]) capabilities |= KV_INSERT_CAP_PAD16;
        if (mask & route_bit(KvInsertRoute::HYBRID2)) capabilities |= KV_INSERT_CAP_HYBRID2;
        if (mask & route_bit(KvInsertRoute::HYBRID3)) capabilities |= KV_INSERT_CAP_HYBRID3;
        if (row[9] != 8) capabilities |= KV_INSERT_CAP_NON8_TP_V16;
        const auto structural = rpu_resolve_kvinsert_segment_plan_auto(
            row[6], row[7], row[8], static_cast<int>(row[9]), row[10], row[11],
            capabilities);
        TORCH_CHECK(structural.feasible_route_mask() == mask,
                    "KV-insert catalog contains a shape-infeasible route domain");
        TORCH_CHECK(domains.emplace(row[3], row[19], row[6], row[7], row[8], row[9],
                                   row[10], row[11], row[12]).second,
                    "KV-insert catalog contains duplicate cost domains");
        result.certificates_.push_back(std::move(certificate));
    }
    return result;
}

bool rpu_kvinsert_cost_certificate_matches(
    const KvInsertCostCertificate& certificate,
    const KvInsertCostDomain& domain) {
    const auto& measured = certificate.domain;
    return certificate.sealed && certificate.certificate_identity > 0 &&
        certificate.dependency_identity == current_dependency_identity() &&
        domain.scope.profile_identity > 0 && domain.scope.site_identity > 0 &&
        domain.scope.dependency_manifest_identity > 0 &&
        domain.scope.runtime_identity > 0 &&
        domain.scope.graph_lifecycle >= 1 && domain.scope.graph_lifecycle <= 4 &&
        measured.scope.profile_identity == domain.scope.profile_identity &&
        measured.scope.site_identity == domain.scope.site_identity &&
        measured.scope.dependency_manifest_identity ==
            domain.scope.dependency_manifest_identity &&
        measured.scope.runtime_identity == domain.scope.runtime_identity &&
        measured.scope.graph_lifecycle == domain.scope.graph_lifecycle &&
        measured.position == domain.position &&
        measured.logical_rows == domain.logical_rows &&
        measured.physical_rows == domain.physical_rows &&
        measured.num_cores == domain.num_cores &&
        measured.num_kv_heads == domain.num_kv_heads &&
        measured.head_dim == domain.head_dim &&
        measured.feasible_route_mask == domain.feasible_route_mask;
}

KvInsertCostCertificate rpu_kvinsert_cost_certificate(
    const KvInsertCostDomain& domain,
    c10::ArrayRef<KvInsertCostCertificate> cost_certificates) {
    TORCH_CHECK(domain.scope.profile_identity >= 0 &&
                    domain.scope.site_identity >= 0 &&
                    domain.scope.dependency_manifest_identity >= 0 &&
                    domain.scope.runtime_identity >= 0 &&
                    domain.scope.graph_lifecycle >= 0 &&
                    domain.scope.graph_lifecycle <= 4,
                "KV-insert cost scope identities/lifecycle are invalid");
    TORCH_CHECK(cost_certificates.empty() ||
                    (domain.scope.profile_identity > 0 &&
                     domain.scope.site_identity > 0 &&
                     domain.scope.dependency_manifest_identity > 0 &&
                     domain.scope.runtime_identity > 0 &&
                     domain.scope.graph_lifecycle > 0),
                "KV-insert cost certificates require a bound cost scope");
    const KvInsertCostCertificate* selected = nullptr;
    for (const auto& certificate : cost_certificates) {
        if (!rpu_kvinsert_cost_certificate_matches(certificate, domain)) continue;
        TORCH_CHECK(domain.feasible_route_mask != 0 &&
                        (domain.feasible_route_mask & ~kKnownRouteMask) == 0,
                    "KV-insert cost certificate has an invalid route domain");
        for (size_t route = 0; route < certificate.route_costs_ns.size(); ++route) {
            const bool feasible = (domain.feasible_route_mask & (1U << route)) != 0;
            TORCH_CHECK(feasible ? certificate.route_costs_ns[route] > 0
                                 : certificate.route_costs_ns[route] == 0,
                        "KV-insert cost certificate must cover the exact route domain");
        }
        TORCH_CHECK(selected == nullptr,
                    "KV-insert cost domain has duplicate sealed certificates");
        selected = &certificate;
    }
    return selected ? *selected : structural_cost_certificate();
}

void rpu_validate_kvinsert_segment_plan(
    const KvInsertSegmentPlan& plan,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim) {
    validate_partition(num_cores, num_kv_heads, head_dim);
    TORCH_CHECK(plan.logical_rows() > 0 &&
                    plan.logical_rows() <= kMaxKernelRows,
                "KV-insert segment plan: logical_rows must fit uint16");
    TORCH_CHECK(plan.physical_rows() >= plan.logical_rows() &&
                    plan.physical_rows() <= kMaxKernelRows,
                "KV-insert segment plan: physical_rows must cover logical_rows "
                "and fit uint16");
    TORCH_CHECK(plan.segment_count() > 0 && plan.segment_count() <= 3,
                "KV-insert segment plan: segment_count must be in [1, 3]");

    const int64_t base_position = plan.segment(0).position;
    TORCH_CHECK(base_position >= 0 && base_position <= kMaxKernelPosition,
                "KV-insert segment plan: position must fit uint32");

    const uint32_t capabilities = plan.certified_capabilities();
    int64_t covered = 0;
    bool uses_v16 = false;
    for (size_t i = 0; i < 3; ++i) {
        const KvInsertSegment& part = plan.segment(i);
        if (i >= plan.segment_count()) {
            TORCH_CHECK(part.kernel == KvInsertKernel::INVALID &&
                            part.position == 0 && part.token_offset == 0 &&
                            part.rows == 0,
                        "KV-insert segment plan: unused segment words must be "
                        "zero");
            continue;
        }
        TORCH_CHECK(part.kernel == KvInsertKernel::V2 ||
                        part.kernel == KvInsertKernel::V16,
                    "KV-insert segment plan: invalid segment kernel");
        TORCH_CHECK(part.token_offset == covered && part.rows > 0 &&
                        part.rows <= kMaxKernelRows,
                    "KV-insert segment plan: segments must be positive and "
                    "contiguous");
        TORCH_CHECK(base_position <= kMaxKernelPosition - part.token_offset &&
                        part.position == base_position + part.token_offset,
                    "KV-insert segment plan: segment position/token_offset "
                    "mismatch or overflow");
        if (part.kernel == KvInsertKernel::V16) {
            uses_v16 = true;
            TORCH_CHECK(v16_shape_ok(part.position, part.rows, num_cores,
                                    num_kv_heads, head_dim,
                                    /*allow_non8_tp=*/true),
                        "KV-insert segment plan: invalid V16 segment shape");
        }
        covered += part.rows;
    }
    TORCH_CHECK(covered == plan.physical_rows(),
                "KV-insert segment plan: segment coverage does not match "
                "physical_rows");
    if (uses_v16 && num_cores != 8) {
        require_capability(capabilities, KV_INSERT_CAP_NON8_TP_V16,
                           "NON8_TP_V16");
    }

    const auto& first = plan.segment(0);
    switch (plan.route()) {
        case KvInsertRoute::V2:
            require_capability(capabilities, KV_INSERT_CAP_V2, "V2");
            TORCH_CHECK(plan.segment_count() == 1 &&
                            first.kernel == KvInsertKernel::V2,
                        "KV-insert segment plan: V2 route must contain one V2 "
                        "segment");
            if (plan.physical_rows() != plan.logical_rows()) {
                require_capability(capabilities, KV_INSERT_CAP_PAD16,
                                   "PAD16");
                validate_padding(base_position, plan.logical_rows(),
                                 plan.physical_rows());
            }
            break;
        case KvInsertRoute::ALIGNED_V16:
            require_capability(capabilities, KV_INSERT_CAP_V16, "V16");
            TORCH_CHECK(plan.segment_count() == 1 &&
                            first.kernel == KvInsertKernel::V16 &&
                            plan.physical_rows() == plan.logical_rows(),
                        "KV-insert segment plan: aligned V16 route shape "
                        "mismatch");
            break;
        case KvInsertRoute::PAD16_V16:
            require_capability(capabilities, KV_INSERT_CAP_V16, "V16");
            require_capability(capabilities, KV_INSERT_CAP_PAD16, "PAD16");
            TORCH_CHECK(plan.segment_count() == 1 &&
                            first.kernel == KvInsertKernel::V16 &&
                            plan.physical_rows() > plan.logical_rows(),
                        "KV-insert segment plan: PAD16 V16 route shape "
                        "mismatch");
            validate_padding(base_position, plan.logical_rows(),
                             plan.physical_rows());
            break;
        case KvInsertRoute::HYBRID2: {
            require_capability(capabilities, KV_INSERT_CAP_V2, "V2");
            require_capability(capabilities, KV_INSERT_CAP_V16, "V16");
            require_capability(capabilities, KV_INSERT_CAP_HYBRID2,
                               "HYBRID2");
            const int64_t bulk = (plan.logical_rows() / 16) * 16;
            const int64_t tail = plan.logical_rows() - bulk;
            TORCH_CHECK(plan.physical_rows() == plan.logical_rows() &&
                            plan.segment_count() == 2 && tail > 0 && bulk >= 16 &&
                            plan.segment(0).kernel == KvInsertKernel::V16 &&
                            plan.segment(0).rows == bulk &&
                            plan.segment(1).kernel == KvInsertKernel::V2 &&
                            plan.segment(1).rows == tail,
                        "KV-insert segment plan: HYBRID2 route shape mismatch");
            break;
        }
        case KvInsertRoute::HYBRID3: {
            require_capability(capabilities, KV_INSERT_CAP_V2, "V2");
            require_capability(capabilities, KV_INSERT_CAP_V16, "V16");
            require_capability(capabilities, KV_INSERT_CAP_HYBRID3,
                               "HYBRID3");
            const int64_t head = (16 - base_position % 16) % 16;
            const int64_t rest = plan.logical_rows() - head;
            const int64_t bulk = rest > 0 ? (rest / 16) * 16 : 0;
            const int64_t tail = rest > 0 ? rest - bulk : 0;
            const uint8_t expected_count = tail > 0 ? 3 : 2;
            TORCH_CHECK(plan.physical_rows() == plan.logical_rows() && head > 0 &&
                            bulk >= 16 &&
                            plan.segment_count() == expected_count &&
                            plan.segment(0).kernel == KvInsertKernel::V2 &&
                            plan.segment(0).rows == head &&
                            plan.segment(1).kernel == KvInsertKernel::V16 &&
                            plan.segment(1).rows == bulk &&
                            (tail == 0 ||
                             (plan.segment(2).kernel == KvInsertKernel::V2 &&
                              plan.segment(2).rows == tail)),
                        "KV-insert segment plan: HYBRID3 route shape mismatch");
            break;
        }
        default:
            TORCH_CHECK(false, "KV-insert segment plan: invalid route");
    }

    const KvInsertCostCertificate& certificate = plan.cost_certificate();
    TORCH_CHECK(plan.dependency_identity() == certificate.dependency_identity,
                "KV-insert segment plan: stale dependency identity");
    TORCH_CHECK(plan.cost_score() == route_cost(plan, certificate),
                "KV-insert segment plan: stale route cost score");
    TORCH_CHECK((plan.feasible_route_mask() & ~kKnownRouteMask) == 0 &&
                    (plan.feasible_route_mask() &
                     route_bit(plan.route())) != 0,
                "KV-insert segment plan: invalid feasible route domain");
    if (plan.selection_state() == KvInsertSelectionState::EXACT) {
        TORCH_CHECK(plan.feasible_route_mask() == route_bit(plan.route()) &&
                        plan.cost_certificate_identity() == 0,
                    "KV-insert exact route has non-canonical selection metadata");
    } else if (plan.selection_state() ==
                   KvInsertSelectionState::CAPABILITY_RANKED) {
        TORCH_CHECK(!certificate.sealed &&
                        plan.cost_certificate_identity() == 0,
                    "KV-insert capability-ranked route cannot claim a hardware "
                    "cost certificate");
    } else if (plan.selection_state() ==
                   KvInsertSelectionState::HARDWARE_COST_CERTIFIED) {
        TORCH_CHECK(rpu_kvinsert_cost_certificate_matches(certificate, {
                        plan.cost_scope(), base_position, plan.logical_rows(),
                        plan.physical_rows(), num_cores, num_kv_heads, head_dim,
                        plan.feasible_route_mask()}) &&
                        plan.cost_certificate_identity() ==
                            certificate.certificate_identity,
                    "KV-insert route has a stale hardware cost certificate");
    } else {
        TORCH_CHECK(false,
                    "KV-insert segment plan: invalid selection state");
    }
}

KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan(
    int64_t position,
    int64_t logical_rows,
    int64_t physical_rows,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim,
    uint32_t certified_capabilities,
    KvInsertRoute exact_route) {
    validate_partition(num_cores, num_kv_heads, head_dim);
    TORCH_CHECK(position >= 0 && position <= kMaxKernelPosition,
                "KV-insert segment plan: position must fit uint32");
    TORCH_CHECK(logical_rows > 0 && logical_rows <= kMaxKernelRows &&
                    physical_rows >= logical_rows &&
                    physical_rows <= kMaxKernelRows,
                "KV-insert segment plan: invalid logical/physical rows");
    if (physical_rows != logical_rows) {
        require_capability(certified_capabilities, KV_INSERT_CAP_PAD16,
                           "PAD16");
        validate_padding(position, logical_rows, physical_rows);
    }

    const bool allow_non8 = has_capability(
        certified_capabilities, KV_INSERT_CAP_NON8_TP_V16);
    KvInsertSegmentPlan plan(
        exact_route, logical_rows, physical_rows, certified_capabilities);

    switch (exact_route) {
        case KvInsertRoute::V2:
            require_capability(certified_capabilities, KV_INSERT_CAP_V2,
                               "V2");
            plan.segment_count_ = 1;
            plan.segments_[0] = segment(KvInsertKernel::V2, position, 0,
                                       physical_rows);
            break;
        case KvInsertRoute::ALIGNED_V16:
            require_capability(certified_capabilities, KV_INSERT_CAP_V16,
                               "V16");
            TORCH_CHECK(physical_rows == logical_rows &&
                            v16_shape_ok(position, physical_rows, num_cores,
                                         num_kv_heads, head_dim, allow_non8),
                        "KV-insert segment plan: exact aligned V16 route is "
                        "not shape-feasible");
            plan.segment_count_ = 1;
            plan.segments_[0] = segment(KvInsertKernel::V16, position, 0,
                                       physical_rows);
            break;
        case KvInsertRoute::PAD16_V16:
            require_capability(certified_capabilities, KV_INSERT_CAP_V16,
                               "V16");
            require_capability(certified_capabilities, KV_INSERT_CAP_PAD16,
                               "PAD16");
            TORCH_CHECK(physical_rows > logical_rows &&
                            v16_shape_ok(position, physical_rows, num_cores,
                                         num_kv_heads, head_dim, allow_non8),
                        "KV-insert segment plan: exact PAD16 V16 route is not "
                        "shape-feasible");
            plan.segment_count_ = 1;
            plan.segments_[0] = segment(KvInsertKernel::V16, position, 0,
                                       physical_rows);
            break;
        case KvInsertRoute::HYBRID2: {
            require_capability(certified_capabilities, KV_INSERT_CAP_V2,
                               "V2");
            require_capability(certified_capabilities, KV_INSERT_CAP_V16,
                               "V16");
            require_capability(certified_capabilities, KV_INSERT_CAP_HYBRID2,
                               "HYBRID2");
            TORCH_CHECK(physical_rows == logical_rows,
                        "KV-insert segment plan: HYBRID2 cannot pad rows");
            const int64_t bulk = (logical_rows / 16) * 16;
            const int64_t tail = logical_rows - bulk;
            TORCH_CHECK(tail > 0 && bulk >= 16 &&
                            v16_shape_ok(position, bulk, num_cores,
                                         num_kv_heads, head_dim, allow_non8),
                        "KV-insert segment plan: exact HYBRID2 route is not "
                        "shape-feasible");
            plan.segment_count_ = 2;
            plan.segments_[0] = segment(KvInsertKernel::V16, position, 0, bulk);
            plan.segments_[1] = segment(KvInsertKernel::V2, position, bulk, tail);
            break;
        }
        case KvInsertRoute::HYBRID3: {
            require_capability(certified_capabilities, KV_INSERT_CAP_V2,
                               "V2");
            require_capability(certified_capabilities, KV_INSERT_CAP_V16,
                               "V16");
            require_capability(certified_capabilities, KV_INSERT_CAP_HYBRID3,
                               "HYBRID3");
            TORCH_CHECK(physical_rows == logical_rows,
                        "KV-insert segment plan: HYBRID3 cannot pad rows");
            const int64_t head = (16 - position % 16) % 16;
            const int64_t rest = logical_rows - head;
            const int64_t bulk = rest > 0 ? (rest / 16) * 16 : 0;
            const int64_t tail = rest > 0 ? rest - bulk : 0;
            TORCH_CHECK(head > 0 && bulk >= 16 &&
                            v16_shape_ok(position + head, bulk, num_cores,
                                         num_kv_heads, head_dim, allow_non8),
                        "KV-insert segment plan: exact HYBRID3 route is not "
                        "shape-feasible");
            plan.segment_count_ = tail > 0 ? 3 : 2;
            plan.segments_[0] = segment(KvInsertKernel::V2, position, 0, head);
            plan.segments_[1] = segment(KvInsertKernel::V16, position, head, bulk);
            if (tail > 0) {
                plan.segments_[2] = segment(KvInsertKernel::V2, position,
                                           head + bulk, tail);
            }
            break;
        }
        default:
            TORCH_CHECK(false, "KV-insert segment plan: invalid exact route");
    }

    const KvInsertCostCertificate& certificate = structural_cost_certificate();
    plan.selection_state_ = KvInsertSelectionState::EXACT;
    plan.feasible_route_mask_ = route_bit(exact_route);
    plan.cost_score_ = route_cost(plan, certificate);
    plan.dependency_identity_ = certificate.dependency_identity;
    plan.cost_certificate_identity_ = 0;
    plan.cost_certificate_ = certificate;
    rpu_validate_kvinsert_segment_plan(plan, num_cores, num_kv_heads, head_dim);
    return plan;
}

KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan_auto(
    int64_t position,
    int64_t logical_rows,
    int64_t physical_rows,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim,
    uint32_t certified_capabilities,
    KvInsertCostScope cost_scope,
    c10::ArrayRef<KvInsertCostCertificate> cost_certificates) {
    validate_partition(num_cores, num_kv_heads, head_dim);
    TORCH_CHECK(position >= 0 && position <= kMaxKernelPosition,
                "KV-insert segment plan: position must fit uint32");
    TORCH_CHECK(logical_rows > 0 && logical_rows <= kMaxKernelRows &&
                    physical_rows >= logical_rows &&
                    physical_rows <= kMaxKernelRows,
                "KV-insert segment plan: invalid logical/physical rows");
    if (physical_rows != logical_rows) {
        require_capability(certified_capabilities, KV_INSERT_CAP_PAD16,
                           "PAD16");
        validate_padding(position, logical_rows, physical_rows);
    }
    const bool allow_non8 = has_capability(
        certified_capabilities, KV_INSERT_CAP_NON8_TP_V16);
    const int64_t head = (16 - position % 16) % 16;
    const int64_t rest = logical_rows - head;
    const int64_t middle = rest > 0 ? (rest / 16) * 16 : 0;
    uint32_t feasible_routes = 0;
    if (has_capability(certified_capabilities, KV_INSERT_CAP_V2)) {
        feasible_routes |= route_bit(KvInsertRoute::V2);
    }
    if (physical_rows == logical_rows && head > 0 && middle >= 16 &&
        has_capability(certified_capabilities, KV_INSERT_CAP_V2) &&
        has_capability(certified_capabilities, KV_INSERT_CAP_V16) &&
        has_capability(certified_capabilities, KV_INSERT_CAP_HYBRID3) &&
        v16_shape_ok(position + head, middle, num_cores, num_kv_heads,
                     head_dim, allow_non8)) {
        feasible_routes |= route_bit(KvInsertRoute::HYBRID3);
    }
    if (has_capability(certified_capabilities, KV_INSERT_CAP_V16) &&
        v16_shape_ok(position, physical_rows, num_cores, num_kv_heads,
                     head_dim, allow_non8)) {
        const KvInsertRoute route = physical_rows == logical_rows
            ? KvInsertRoute::ALIGNED_V16
            : KvInsertRoute::PAD16_V16;
        feasible_routes |= route_bit(route);
    }
    const int64_t bulk = (logical_rows / 16) * 16;
    if (physical_rows == logical_rows && position % 16 == 0 &&
        bulk >= 16 && bulk < logical_rows &&
        has_capability(certified_capabilities, KV_INSERT_CAP_V2) &&
        has_capability(certified_capabilities, KV_INSERT_CAP_V16) &&
        has_capability(certified_capabilities, KV_INSERT_CAP_HYBRID2) &&
        v16_shape_ok(position, bulk, num_cores, num_kv_heads, head_dim,
                     allow_non8)) {
        feasible_routes |= route_bit(KvInsertRoute::HYBRID2);
    }
    TORCH_CHECK(feasible_routes != 0,
                "KV-insert AUTO found no certified feasible route");
    const KvInsertCostCertificate certificate =
        rpu_kvinsert_cost_certificate({
            cost_scope, position, logical_rows, physical_rows, num_cores,
            num_kv_heads, head_dim, feasible_routes}, cost_certificates);

    constexpr std::array<KvInsertRoute, 5> routes{
        KvInsertRoute::V2,
        KvInsertRoute::ALIGNED_V16,
        KvInsertRoute::PAD16_V16,
        KvInsertRoute::HYBRID2,
        KvInsertRoute::HYBRID3,
    };
    std::optional<KvInsertSegmentPlan> winner;
    int64_t winner_score = std::numeric_limits<int64_t>::max();
    for (const KvInsertRoute route : routes) {
        if ((feasible_routes & route_bit(route)) == 0) continue;
        KvInsertSegmentPlan candidate = rpu_resolve_kvinsert_segment_plan(
            position, logical_rows, physical_rows, num_cores, num_kv_heads,
            head_dim, certified_capabilities, route);
        const int64_t score = route_cost(candidate, certificate);
        if (!winner || score < winner_score ||
            (score == winner_score &&
             static_cast<uint8_t>(route) <
                 static_cast<uint8_t>(winner->route()))) {
            winner = std::move(candidate);
            winner_score = score;
        }
    }
    TORCH_CHECK(winner.has_value(),
                "KV-insert AUTO candidate enumeration is empty");
    winner->selection_state_ = certificate.sealed
        ? KvInsertSelectionState::HARDWARE_COST_CERTIFIED
        : KvInsertSelectionState::CAPABILITY_RANKED;
    winner->feasible_route_mask_ = feasible_routes;
    winner->cost_score_ = winner_score;
    winner->dependency_identity_ = certificate.dependency_identity;
    winner->cost_certificate_identity_ = certificate.certificate_identity;
    winner->cost_scope_ = cost_scope;
    winner->cost_certificate_ = certificate;
    rpu_validate_kvinsert_segment_plan(
        *winner, num_cores, num_kv_heads, head_dim);
    return *winner;
}

KvInsertRouteArguments rpu_kvinsert_route_arguments(
    const KvInsertSegmentPlan& plan,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim) {
    rpu_validate_kvinsert_segment_plan(plan, num_cores, num_kv_heads, head_dim);
    KvInsertRouteArguments arguments{};
    arguments[0] = 2;
    arguments[1] = static_cast<int64_t>(plan.route());
    arguments[2] = plan.logical_rows();
    arguments[3] = plan.physical_rows();
    arguments[4] = plan.segment_count();
    for (size_t i = 0; i < plan.segment_count(); ++i) {
        const size_t base = 5 + i * 4;
        arguments[base] = static_cast<int64_t>(plan.segment(i).kernel);
        arguments[base + 1] = plan.segment(i).position;
        arguments[base + 2] = plan.segment(i).token_offset;
        arguments[base + 3] = plan.segment(i).rows;
    }
    arguments[17] = static_cast<int64_t>(plan.selection_state());
    arguments[18] = plan.feasible_route_mask();
    arguments[19] = plan.cost_score();
    arguments[20] = plan.dependency_identity();
    arguments[21] = plan.cost_certificate_identity();
    return arguments;
}

KvInsertSegmentPlan rpu_kvinsert_segment_plan_from_route_arguments(
    c10::ArrayRef<int64_t> arguments,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim,
    KvInsertCostScope cost_scope,
    c10::ArrayRef<KvInsertCostCertificate> cost_certificates) {
    TORCH_CHECK(
        arguments.size() == kKvInsertRouteArgumentWords &&
            arguments[0] == 2,
        "KV-insert descriptor route arguments have an unsupported schema");
    const auto route = static_cast<KvInsertRoute>(arguments[1]);
    uint32_t capabilities = 0;
    switch (route) {
        case KvInsertRoute::V2:
            capabilities = KV_INSERT_CAP_V2;
            break;
        case KvInsertRoute::ALIGNED_V16:
            capabilities = KV_INSERT_CAP_V16;
            break;
        case KvInsertRoute::PAD16_V16:
            capabilities = KV_INSERT_CAP_V16 | KV_INSERT_CAP_PAD16;
            break;
        case KvInsertRoute::HYBRID2:
            capabilities = KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
                KV_INSERT_CAP_HYBRID2;
            break;
        case KvInsertRoute::HYBRID3:
            capabilities = KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
                KV_INSERT_CAP_HYBRID3;
            break;
        default:
            TORCH_CHECK(false,
                        "KV-insert descriptor contains an invalid route");
    }
    if (arguments[3] != arguments[2]) {
        capabilities |= KV_INSERT_CAP_PAD16;
    }
    if (num_cores != 8 && route != KvInsertRoute::V2) {
        capabilities |= KV_INSERT_CAP_NON8_TP_V16;
    }
    TORCH_CHECK(arguments[5] != 0 && arguments[7] == 0,
                "KV-insert descriptor has no canonical first segment");
    const auto selection_state =
        static_cast<KvInsertSelectionState>(arguments[17]);
    const uint32_t feasible_routes = static_cast<uint32_t>(arguments[18]);
    TORCH_CHECK(arguments[18] >= 0 &&
                    arguments[18] <= std::numeric_limits<uint32_t>::max(),
                "KV-insert descriptor contains an invalid route domain");
    if (feasible_routes & route_bit(KvInsertRoute::V2)) {
        capabilities |= KV_INSERT_CAP_V2;
    }
    if (feasible_routes &
        (route_bit(KvInsertRoute::ALIGNED_V16) |
         route_bit(KvInsertRoute::PAD16_V16) |
         route_bit(KvInsertRoute::HYBRID2) |
         route_bit(KvInsertRoute::HYBRID3))) {
        capabilities |= KV_INSERT_CAP_V16;
    }
    if (feasible_routes & route_bit(KvInsertRoute::PAD16_V16)) {
        capabilities |= KV_INSERT_CAP_PAD16;
    }
    if (feasible_routes & route_bit(KvInsertRoute::HYBRID2)) {
        capabilities |= KV_INSERT_CAP_HYBRID2;
    }
    if (feasible_routes & route_bit(KvInsertRoute::HYBRID3)) {
        capabilities |= KV_INSERT_CAP_HYBRID3;
    }
    if (num_cores != 8 &&
        (feasible_routes & ~route_bit(KvInsertRoute::V2)) != 0) {
        capabilities |= KV_INSERT_CAP_NON8_TP_V16;
    }
    KvInsertSegmentPlan plan = selection_state == KvInsertSelectionState::EXACT
        ? rpu_resolve_kvinsert_segment_plan(
              arguments[6], arguments[2], arguments[3], num_cores,
              num_kv_heads, head_dim, capabilities, route)
        : rpu_resolve_kvinsert_segment_plan_auto(
              arguments[6], arguments[2], arguments[3], num_cores,
              num_kv_heads, head_dim, capabilities, cost_scope, cost_certificates);
    TORCH_CHECK(plan.route() == route,
                "KV-insert descriptor route is not the ranked domain winner");
    const KvInsertRouteArguments canonical = rpu_kvinsert_route_arguments(
        plan, num_cores, num_kv_heads, head_dim);
    TORCH_CHECK(
        std::equal(canonical.begin(), canonical.end(), arguments.begin()),
        "KV-insert descriptor route arguments are not canonical");
    return plan;
}

KvInsertSegmentPlan rpu_rebase_kvinsert_segment_plan_position(
    const KvInsertSegmentPlan& template_plan,
    int64_t live_position,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim) {
    const KvInsertSegmentPlan rebased = rpu_resolve_kvinsert_segment_plan(
        live_position, template_plan.logical_rows(),
        template_plan.physical_rows(), num_cores, num_kv_heads, head_dim,
        template_plan.certified_capabilities(), template_plan.route());
    TORCH_CHECK(
        rebased.route() == template_plan.route() &&
            rebased.logical_rows() == template_plan.logical_rows() &&
            rebased.physical_rows() == template_plan.physical_rows() &&
            rebased.segment_count() == template_plan.segment_count(),
        "KV-insert dynamic position changed the frozen route topology");
    for (size_t i = 0; i < rebased.segment_count(); ++i) {
        TORCH_CHECK(
            rebased.segment(i).kernel == template_plan.segment(i).kernel &&
                rebased.segment(i).token_offset ==
                    template_plan.segment(i).token_offset &&
                rebased.segment(i).rows == template_plan.segment(i).rows,
            "KV-insert dynamic position changed the frozen segment topology");
    }
    return rebased;
}

KvInsertSegmentPlan rpu_resolve_legacy_kvinsert_segment_plan(
    int64_t position,
    int64_t logical_rows,
    int64_t spm_rows,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim,
    bool allow_non8_v16,
    bool allow_hybrid_v16,
    const KvInsertLegacyPolicy& policy) {
    TORCH_CHECK(logical_rows > 0 && logical_rows <= kMaxKernelRows,
                "KV-insert segment plan: legacy seq_len must fit uint16");
    TORCH_CHECK(spm_rows == 0 || spm_rows >= logical_rows,
                "KV-insert spm_rows must cover seq_len; got spm_rows=",
                spm_rows, " seq_len=", logical_rows);
    const int64_t padded_rows = logical_rows > 0 ? align16(logical_rows) : 0;
    const int64_t physical_rows =
        spm_rows >= padded_rows && padded_rows != logical_rows &&
                position % 16 == 0
        ? padded_rows
        : logical_rows;
    const bool any_tp = policy.any_tp_v16 && num_cores > 0 &&
        8 % num_cores == 0;
    const bool whole_allow_non8 = allow_non8_v16 || any_tp;

    const int64_t head3 = position >= 0 ? (16 - position % 16) % 16 : 0;
    const int64_t rest3 = logical_rows - head3;
    const int64_t bulk3 = rest3 > 0 ? (rest3 / 16) * 16 : 0;
    if (policy.hybrid3_v16 && policy.v16_enabled && head3 > 0 &&
        rest3 > 0 && bulk3 >= 16 &&
        v16_shape_ok(position + head3, bulk3, num_cores, num_kv_heads,
                     head_dim, whole_allow_non8)) {
        uint32_t caps = KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
            KV_INSERT_CAP_HYBRID3;
        if (whole_allow_non8) caps |= KV_INSERT_CAP_NON8_TP_V16;
        return rpu_resolve_kvinsert_segment_plan(
            position, logical_rows, logical_rows, num_cores, num_kv_heads,
            head_dim, caps, KvInsertRoute::HYBRID3);
    }

    const int64_t bulk2 = (physical_rows / 16) * 16;
    const int64_t tail2 = physical_rows - bulk2;
    if ((policy.hybrid2_v16 || allow_hybrid_v16) && policy.v16_enabled &&
        tail2 > 0 && bulk2 >= 16 &&
        v16_shape_ok(position, bulk2, num_cores, num_kv_heads, head_dim,
                     allow_non8_v16)) {
        uint32_t caps = KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 |
            KV_INSERT_CAP_HYBRID2;
        if (allow_non8_v16) caps |= KV_INSERT_CAP_NON8_TP_V16;
        return rpu_resolve_kvinsert_segment_plan(
            position, logical_rows, physical_rows, num_cores, num_kv_heads,
            head_dim, caps, KvInsertRoute::HYBRID2);
    }

    if (policy.v16_enabled &&
        v16_shape_ok(position, physical_rows, num_cores, num_kv_heads,
                     head_dim, whole_allow_non8)) {
        uint32_t caps = KV_INSERT_CAP_V16;
        if (whole_allow_non8) caps |= KV_INSERT_CAP_NON8_TP_V16;
        const KvInsertRoute route = physical_rows == logical_rows
            ? KvInsertRoute::ALIGNED_V16
            : KvInsertRoute::PAD16_V16;
        if (route == KvInsertRoute::PAD16_V16) {
            caps |= KV_INSERT_CAP_PAD16;
        }
        return rpu_resolve_kvinsert_segment_plan(
            position, logical_rows, physical_rows, num_cores, num_kv_heads,
            head_dim, caps, route);
    }

    uint32_t caps = KV_INSERT_CAP_V2;
    if (physical_rows != logical_rows) caps |= KV_INSERT_CAP_PAD16;
    return rpu_resolve_kvinsert_segment_plan(
        position, logical_rows, physical_rows, num_cores, num_kv_heads,
        head_dim, caps, KvInsertRoute::V2);
}

KvInsertShapePlan rpu_kvinsert_shape_plan(int64_t position, int64_t seq_len,
                                          int num_cores,
                                          int64_t num_kv_heads,
                                          int64_t head_dim) {
    KvInsertShapePlan plan{};
    plan.bulk_seq = (seq_len / 16) * 16;
    plan.tail_seq = seq_len - plan.bulk_seq;
    plan.v16_shape_ok = v16_shape_ok(position, seq_len, num_cores,
                                     num_kv_heads, head_dim,
                                     /*allow_non8_tp=*/false);
    plan.hybrid_shape_ok = plan.tail_seq > 0 && plan.bulk_seq >= 16 &&
        v16_shape_ok(position, plan.bulk_seq, num_cores, num_kv_heads,
                     head_dim, /*allow_non8_tp=*/false);
    return plan;
}
