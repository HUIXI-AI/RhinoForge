#pragma once

#include <tuple>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <c10/util/ArrayRef.h>

enum class KvInsertKernel : uint8_t {
    INVALID = 0,
    V2 = 1,
    V16 = 2,
};

enum class KvInsertRoute : uint8_t {
    INVALID = 0,
    V2 = 1,
    ALIGNED_V16 = 2,
    PAD16_V16 = 3,
    HYBRID2 = 4,
    HYBRID3 = 5,
};

enum class KvInsertSelectionState : uint8_t {
    INVALID = 0,
    EXACT = 1,
    CAPABILITY_RANKED = 2,
    HARDWARE_COST_CERTIFIED = 3,
};

enum KvInsertCapability : uint32_t {
    KV_INSERT_CAP_V2 = 1u << 0,
    KV_INSERT_CAP_V16 = 1u << 1,
    KV_INSERT_CAP_PAD16 = 1u << 2,
    KV_INSERT_CAP_HYBRID2 = 1u << 3,
    KV_INSERT_CAP_HYBRID3 = 1u << 4,
    KV_INSERT_CAP_NON8_TP_V16 = 1u << 5,
};

// The descriptor freezes one exact route/topology while a retained decode
// graph supplies the live cache position at dispatch time.  Keep this bit away
// from the small owner-local DDR-reason values already carried in route flags.
constexpr int64_t KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION = 1LL << 16;

struct KvInsertSegment {
    KvInsertKernel kernel = KvInsertKernel::INVALID;
    int64_t position = 0;
    int64_t token_offset = 0;
    int64_t rows = 0;
};

// Internal identity supplied by the owner, never inferred from a request label
// or a handle address. Profile binds geometry/precision/runtime capability;
// site binds the concrete physical insert use. Zero means not yet bound.
struct KvInsertCostScope {
    int64_t profile_identity = 0;
    int64_t site_identity = 0;
    // Projections of the existing verified dependency-manifest (includes SDK)
    // and runtime-payload-manifest digests. Do not replace either with a ref SHA.
    int64_t dependency_manifest_identity = 0;
    int64_t runtime_identity = 0;
    // Actual FMB lifecycle: retained=1, one-shot=2, native-composite=3,
    // composite-child=4. Zero is unbound, never cost-certified.
    int64_t graph_lifecycle = 0;
};

struct KvInsertCostDomain {
    KvInsertCostScope scope;
    int64_t position = 0;
    int64_t logical_rows = 0;
    int64_t physical_rows = 0;
    int num_cores = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    uint32_t feasible_route_mask = 0;
};

// Read-only diagnostic query, never an executor input or an execution wire.
// (native owner kind, admission reason, profile words, exact domain rows).
using KvInsertCostDomainQuery = std::tuple<
    std::string, std::string, std::vector<int64_t>,
    std::vector<std::vector<int64_t>>>;

struct KvInsertCostCertificate {
    int64_t dependency_identity = 0;
    int64_t certificate_identity = 0;
    bool sealed = false;
    KvInsertCostDomain domain;
    // Same-session median device ns per complete route, indexed by route enum.
    // Infeasible routes (including INVALID) must have zero cost. Structural
    // grid counts remain an unsealed fallback, never fitted "measured" weights.
    std::array<int64_t, 6> route_costs_ns{};
};

// An owner-local, immutable copy of already-verified external cost data. This
// native boundary checks wire/domain consistency; the CI artifact verifier owns
// SHA/build/job provenance. No user execution config accepts this representation.
class KvInsertCostCatalog final {
public:
    KvInsertCostCatalog() = default;
    static KvInsertCostCatalog from_arguments(
        c10::ArrayRef<int64_t> identity,
        const std::string& artifact_sha256,
        c10::ArrayRef<int64_t> certificate_rows);

    bool bound() const { return scope_.profile_identity > 0; }
    KvInsertCostScope scope(int64_t site_id, int64_t graph_lifecycle) const;
    const std::string& artifact_sha256() const { return artifact_sha256_; }
    c10::ArrayRef<KvInsertCostCertificate> certificates() const {
        return certificates_;
    }

private:
    KvInsertCostScope scope_;
    std::string artifact_sha256_;
    std::vector<KvInsertCostCertificate> certificates_;
};

// identity = [profile, dependency-manifest, runtime] SHA256 projections.
// Rows = [version=2, count, (dependency-id, certificate-id,
//         profile, site, dependency-manifest, runtime, position, logical,
//         physical, cores, heads, head-dim, feasible-mask, costs[6],
//         graph-lifecycle) x count].
constexpr size_t kKvInsertCostCertificateWords = 20;

// Resolver-only construction prevents a caller from hand-authoring a V16 plan
// that bypasses its certified capability.  K and V receive the same immutable
// value by const reference.
class KvInsertSegmentPlan final {
public:
    KvInsertRoute route() const { return route_; }
    int64_t logical_rows() const { return logical_rows_; }
    int64_t physical_rows() const { return physical_rows_; }
    size_t segment_count() const { return segment_count_; }
    const KvInsertSegment& segment(size_t index) const {
        return segments_.at(index);
    }
    uint32_t certified_capabilities() const {
        return certified_capabilities_;
    }
    KvInsertSelectionState selection_state() const {
        return selection_state_;
    }
    uint32_t feasible_route_mask() const { return feasible_route_mask_; }
    size_t candidate_count() const {
        size_t count = 0;
        for (uint32_t mask = feasible_route_mask_; mask != 0; mask >>= 1) {
            count += mask & 1U;
        }
        return count;
    }
    int64_t cost_score() const { return cost_score_; }
    int64_t dependency_identity() const { return dependency_identity_; }
    int64_t cost_certificate_identity() const {
        return cost_certificate_identity_;
    }
    KvInsertCostScope cost_scope() const { return cost_scope_; }
    const KvInsertCostCertificate& cost_certificate() const {
        return cost_certificate_;
    }

private:
    KvInsertSegmentPlan(KvInsertRoute route,
                        int64_t logical_rows,
                        int64_t physical_rows,
                        uint32_t certified_capabilities)
        : route_(route),
          logical_rows_(logical_rows),
          physical_rows_(physical_rows),
          certified_capabilities_(certified_capabilities) {}

    KvInsertRoute route_;
    int64_t logical_rows_;
    int64_t physical_rows_;
    uint32_t certified_capabilities_;
    uint8_t segment_count_ = 0;
    std::array<KvInsertSegment, 3> segments_{};
    KvInsertSelectionState selection_state_ =
        KvInsertSelectionState::INVALID;
    uint32_t feasible_route_mask_ = 0;
    int64_t cost_score_ = 0;
    int64_t dependency_identity_ = 0;
    int64_t cost_certificate_identity_ = 0;
    KvInsertCostScope cost_scope_;
    // Own the selected value: launch validation must not consult a replaced or
    // destroyed external catalog after the plan has been built.
    KvInsertCostCertificate cost_certificate_;

    friend KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan(
        int64_t position,
        int64_t logical_rows,
        int64_t physical_rows,
        int num_cores,
        int64_t num_kv_heads,
        int64_t head_dim,
        uint32_t certified_capabilities,
        KvInsertRoute exact_route);
    friend KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan_auto(
        int64_t position,
        int64_t logical_rows,
        int64_t physical_rows,
        int num_cores,
        int64_t num_kv_heads,
        int64_t head_dim,
        uint32_t certified_capabilities,
        KvInsertCostScope cost_scope,
        c10::ArrayRef<KvInsertCostCertificate> cost_certificates);
    friend KvInsertSegmentPlan rpu_kvinsert_segment_plan_from_route_arguments(
        c10::ArrayRef<int64_t> arguments,
        int num_cores,
        int64_t num_kv_heads,
        int64_t head_dim,
        KvInsertCostScope cost_scope,
        c10::ArrayRef<KvInsertCostCertificate> cost_certificates);
};

// Public provider release and opaque asset identities. A dependency bump
// invalidates sealed hardware cost certificates; it does not relabel
// historical calibration evidence. Hashes refer to the shared library and
// combined operator payload supplied with this release.
inline constexpr char kKvInsertCostRuntimeRelease[] = "runtime-v1.1.0";
inline constexpr char kKvInsertCostLaunchSha256[] =
    "30b98620d37645d72ecff4390cc95c0061ef9bf3a9c6b7fde69b359c2d00c423";
inline constexpr char kKvInsertCostOperatorSha256[] =
    "a582e62a6577a112afaef5896818d6673c297bba6deb527eef70bd388acec587";

// The no-argument overload is only the unsealed structural fallback.
const KvInsertCostCertificate& rpu_kvinsert_cost_certificate();
KvInsertCostCertificate rpu_kvinsert_cost_certificate(
    const KvInsertCostDomain& domain,
    c10::ArrayRef<KvInsertCostCertificate> cost_certificates = {});
bool rpu_kvinsert_cost_certificate_matches(
    const KvInsertCostCertificate& certificate,
    const KvInsertCostDomain& domain);

constexpr size_t kKvInsertRouteTopologyWords = 17;
constexpr size_t kKvInsertRouteArgumentWords = 22;
using KvInsertRouteArguments =
    std::array<int64_t, kKvInsertRouteArgumentWords>;

KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan(
    int64_t position,
    int64_t logical_rows,
    int64_t physical_rows,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim,
    uint32_t certified_capabilities,
    KvInsertRoute exact_route);

KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan_auto(
    int64_t position,
    int64_t logical_rows,
    int64_t physical_rows,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim,
    uint32_t certified_capabilities,
    KvInsertCostScope cost_scope = {},
    c10::ArrayRef<KvInsertCostCertificate> cost_certificates = {});

void rpu_validate_kvinsert_segment_plan(
    const KvInsertSegmentPlan& plan,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim);

// FMB KV_INSERT route arguments carry one fixed representation:
// [version=2, route, logical, physical, count,
//  (kernel, position, token_offset, rows) x 3,
//  selection_state, feasible_route_mask, selected_cost,
//  dependency_identity, cost_certificate_identity].
// Unused segment words are zero.  The final identity is zero until P7 seals a
// dependency-bound hardware cost certificate.
KvInsertRouteArguments rpu_kvinsert_route_arguments(
    const KvInsertSegmentPlan& plan,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim);

// Restore the exact typed plan carried by a COMPLETE FMB descriptor.  A
// non-canonical or tampered argument vector is rejected before launch. A
// sealed AUTO receipt additionally requires its owner-supplied cost scope;
// an unbound consumer cannot borrow profile authority from the encoded ID.
KvInsertSegmentPlan rpu_kvinsert_segment_plan_from_route_arguments(
    c10::ArrayRef<int64_t> arguments,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim,
    KvInsertCostScope cost_scope = {},
    c10::ArrayRef<KvInsertCostCertificate> cost_certificates = {});

// Rebase only the segment positions of a descriptor-restored plan.  Route,
// segment kernels, offsets, and row counts remain frozen; a live position that
// would require a different topology fails closed.
KvInsertSegmentPlan rpu_rebase_kvinsert_segment_plan_position(
    const KvInsertSegmentPlan& template_plan,
    int64_t live_position,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim);

// Compatibility-only input to the old launcher ABI.  It contains values after
// translating legacy environment switches; the resolver itself never reads
// process state.
struct KvInsertLegacyPolicy {
    bool v16_enabled = true;
    bool any_tp_v16 = false;
    bool hybrid2_v16 = false;
    bool hybrid3_v16 = false;
};

KvInsertSegmentPlan rpu_resolve_legacy_kvinsert_segment_plan(
    int64_t position,
    int64_t logical_rows,
    int64_t spm_rows,
    int num_cores,
    int64_t num_kv_heads,
    int64_t head_dim,
    bool allow_non8_v16,
    bool allow_hybrid_v16,
    const KvInsertLegacyPolicy& policy);

// Retained pure mechanism-test surface.
struct KvInsertShapePlan {
    int64_t bulk_seq;
    int64_t tail_seq;
    bool v16_shape_ok;
    bool hybrid_shape_ok;
};

KvInsertShapePlan rpu_kvinsert_shape_plan(int64_t position, int64_t seq_len,
                                          int num_cores,
                                          int64_t num_kv_heads,
                                          int64_t head_dim);
