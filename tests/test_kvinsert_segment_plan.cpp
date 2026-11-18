#include "ops/rpu_kvinsert_segment_plan.h"

#include <c10/util/Exception.h>

#include <iostream>
#include <type_traits>

namespace {

int failed = 0;

#define EXPECT(condition, message)                                            \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "FAIL: " << (message) << '\n';                     \
            ++failed;                                                        \
        }                                                                     \
    } while (0)

template <typename Fn>
bool throws_c10(Fn&& fn) {
    try {
        fn();
    } catch (const c10::Error&) {
        return true;
    }
    return false;
}

KvInsertSegmentPlan legacy(
    int64_t position, int64_t rows, int64_t spm_rows,
    int cores, bool allow_non8, bool allow_hybrid,
    KvInsertLegacyPolicy policy) {
    return rpu_resolve_legacy_kvinsert_segment_plan(
        position, rows, spm_rows, cores, /*nkv=*/8, /*head_dim=*/128,
        allow_non8, allow_hybrid, policy);
}

}  // namespace

int main() {
    static_assert(!std::is_default_constructible_v<KvInsertSegmentPlan>);
    static_assert(!std::is_aggregate_v<KvInsertSegmentPlan>);

    constexpr uint32_t all_caps =
        KV_INSERT_CAP_V2 | KV_INSERT_CAP_V16 | KV_INSERT_CAP_PAD16 |
        KV_INSERT_CAP_HYBRID2 | KV_INSERT_CAP_HYBRID3;

    const auto v2 = rpu_resolve_kvinsert_segment_plan(
        7, 15, 15, 8, 8, 128, KV_INSERT_CAP_V2, KvInsertRoute::V2);
    EXPECT(v2.segment_count() == 1 &&
               v2.segment(0).kernel == KvInsertKernel::V2 &&
               v2.segment(0).position == 7 && v2.segment(0).rows == 15,
           "V2 resolves one exact segment");

    const auto aligned = rpu_resolve_kvinsert_segment_plan(
        16, 32, 32, 8, 8, 128, all_caps,
        KvInsertRoute::ALIGNED_V16);
    EXPECT(aligned.segment_count() == 1 &&
               aligned.segment(0).kernel == KvInsertKernel::V16 &&
               aligned.segment(0).rows == 32,
           "aligned V16 resolves one segment");

    // NextDiT uses six complete heads on cores 0..5 with eight cache lanes.
    // This V2 geometry does not certify six-core V16 or hybrid kernels.
    for (const auto shape : {std::array<int64_t, 3>{0, 32, 32},
                             {0, 128, 128}, {0, 260, 260},
                             {1, 32, 32}, {0, 17, 32}}) {
        const auto [position, logical, physical] = shape;
        const auto caps = all_caps | KV_INSERT_CAP_NON8_TP_V16;
        for (const auto& plan : {
                 rpu_resolve_kvinsert_segment_plan(
                     position, logical, physical, 6, 6, 64, caps,
                     KvInsertRoute::V2),
                 rpu_resolve_kvinsert_segment_plan_auto(
                     position, logical, physical, 6, 6, 64, caps),
                 rpu_resolve_legacy_kvinsert_segment_plan(
                     position, logical, physical, 6, 6, 64, true, true,
                     {true, true, true, true})}) {
            EXPECT(plan.route() == KvInsertRoute::V2 &&
                       plan.segment_count() == 1 &&
                       plan.segment(0).kernel == KvInsertKernel::V2 &&
                       plan.segment(0).position == position &&
                       plan.segment(0).rows == physical &&
                       plan.feasible_route_mask() == 2 &&
                       plan.cost_score() == physical &&
                       plan.cost_certificate_identity() == 0,
                   "six-core exact/AUTO/legacy retain the sole V2 route");
            const auto wire = rpu_kvinsert_route_arguments(plan, 6, 6, 64);
            const auto restored = rpu_kvinsert_segment_plan_from_route_arguments(
                wire, 6, 6, 64);
            EXPECT(rpu_kvinsert_route_arguments(restored, 6, 6, 64) == wire,
                   "six-core V2 descriptor preserves its topology and cost domain");
        }
        for (const auto route : {KvInsertRoute::ALIGNED_V16,
                                KvInsertRoute::PAD16_V16,
                                KvInsertRoute::HYBRID2,
                                KvInsertRoute::HYBRID3}) {
            EXPECT(throws_c10([&] {
                (void)rpu_resolve_kvinsert_segment_plan(
                    position, logical, physical, 6, 6, 64, caps, route);
            }), "six-core V16 and hybrid routes remain uncertified");
        }
    }
    for (const auto geometry : {std::array<int64_t, 3>{0, 6, 64},
                               {9, 9, 64}, {6, 5, 64}, {6, 3, 64},
                               {3, 1, 192}}) {
        const auto [cores, heads, width] = geometry;
        EXPECT(throws_c10([&] {
            (void)rpu_resolve_kvinsert_segment_plan_auto(
                0, 32, 32, static_cast<int>(cores), heads, width, all_caps);
        }), "invalid or non-eight-divisor split-head partitions remain rejected");
    }

    // The shipped padded-eight-lane cache uses the frozen V16 kernels, whose
    // per-warp byte step (heads_per_core * head_dim * 32) is a uint16.
    for (int cores : {1, 2, 4, 8}) {
        const uint32_t caps = all_caps | KV_INSERT_CAP_NON8_TP_V16;
        EXPECT(rpu_resolve_kvinsert_segment_plan(
                   0, 32, 32, cores, cores, 2032, caps,
                   KvInsertRoute::ALIGNED_V16).route() ==
                   KvInsertRoute::ALIGNED_V16,
               "V16 accepts the last aligned non-overflowing per-core width");
        EXPECT(throws_c10([&] {
                   (void)rpu_resolve_kvinsert_segment_plan(
                       0, 32, 32, cores, cores * 16, 128, caps,
                       KvInsertRoute::ALIGNED_V16);
               }), "exact V16 rejects the uint16 per-warp step overflow");
        EXPECT(rpu_resolve_kvinsert_segment_plan_auto(
                   0, 33, 33, cores, cores * 16, 128, caps).route() ==
                   KvInsertRoute::V2,
               "AUTO excludes overflowing V16 and hybrid candidates");
        EXPECT(rpu_resolve_legacy_kvinsert_segment_plan(
                   0, 32, 32, cores, cores * 16, 128, true, true,
                   {true, true, true, true}).route() == KvInsertRoute::V2,
               "legacy compatibility uses the same V16 width guard");
    }

    const auto padded = rpu_resolve_kvinsert_segment_plan(
        0, 17, 32, 8, 8, 128, all_caps, KvInsertRoute::PAD16_V16);
    EXPECT(padded.logical_rows() == 17 && padded.physical_rows() == 32 &&
               padded.segment(0).kernel == KvInsertKernel::V16,
           "PAD16 keeps logical and physical rows distinct");

    const auto hybrid2 = rpu_resolve_kvinsert_segment_plan(
        0, 255, 255, 8, 8, 128, all_caps, KvInsertRoute::HYBRID2);
    EXPECT(hybrid2.segment_count() == 2 &&
               hybrid2.segment(0).kernel == KvInsertKernel::V16 &&
               hybrid2.segment(0).rows == 240 &&
               hybrid2.segment(1).kernel == KvInsertKernel::V2 &&
               hybrid2.segment(1).position == 240 &&
               hybrid2.segment(1).token_offset == 240 &&
               hybrid2.segment(1).rows == 15,
           "HYBRID2 resolves aligned bulk plus V2 tail");
    EXPECT(hybrid2.selection_state() == KvInsertSelectionState::EXACT &&
               hybrid2.candidate_count() == 1 &&
               hybrid2.cost_score() == 30 &&
               hybrid2.dependency_identity() > 0 &&
               hybrid2.cost_certificate_identity() == 0,
           "an exact route reports a one-point unsealed receipt");

    const auto hybrid3 = rpu_resolve_kvinsert_segment_plan(
        225, 51, 51, 8, 8, 128, all_caps, KvInsertRoute::HYBRID3);
    EXPECT(hybrid3.segment_count() == 3 &&
               hybrid3.segment(0).kernel == KvInsertKernel::V2 &&
               hybrid3.segment(0).rows == 15 &&
               hybrid3.segment(1).kernel == KvInsertKernel::V16 &&
               hybrid3.segment(1).position == 240 &&
               hybrid3.segment(1).rows == 32 &&
               hybrid3.segment(2).kernel == KvInsertKernel::V2 &&
               hybrid3.segment(2).position == 272 &&
               hybrid3.segment(2).rows == 4,
           "HYBRID3 resolves V2 head, V16 bulk, and V2 tail");

    const auto encoded = rpu_kvinsert_route_arguments(
        hybrid3, 8, 8, 128);
    EXPECT(encoded.size() == kKvInsertRouteArgumentWords &&
               encoded[0] == 2 &&
               encoded[1] == static_cast<int64_t>(KvInsertRoute::HYBRID3) &&
               encoded[2] == 51 && encoded[3] == 51 && encoded[4] == 3 &&
               encoded[5] == static_cast<int64_t>(KvInsertKernel::V2) &&
               encoded[6] == 225 && encoded[7] == 0 && encoded[8] == 15 &&
               encoded[9] == static_cast<int64_t>(KvInsertKernel::V16) &&
               encoded[10] == 240 && encoded[11] == 15 &&
               encoded[12] == 32 &&
               encoded[13] == static_cast<int64_t>(KvInsertKernel::V2) &&
               encoded[14] == 272 && encoded[15] == 47 && encoded[16] == 4 &&
               encoded[17] ==
                   static_cast<int64_t>(KvInsertSelectionState::EXACT) &&
               encoded[18] ==
                   (1U << static_cast<uint8_t>(KvInsertRoute::HYBRID3)) &&
               encoded[19] == 21 && encoded[20] > 0 && encoded[21] == 0,
           "FMB KV_INSERT arguments bind topology and selection receipt");
    const std::vector<int64_t> encoded_vector(encoded.begin(), encoded.end());
    const auto decoded = rpu_kvinsert_segment_plan_from_route_arguments(
        encoded_vector, 8, 8, 128);
    EXPECT(decoded.route() == KvInsertRoute::HYBRID3 &&
               decoded.segment_count() == hybrid3.segment_count(),
           "descriptor arguments round-trip to the exact typed plan");
    auto tampered = encoded_vector;
    tampered.back() = 1;
    EXPECT(throws_c10([&] {
               (void)rpu_kvinsert_segment_plan_from_route_arguments(
                   tampered, 8, 8, 128);
           }),
           "non-canonical descriptor arguments fail closed");
    const auto auto_hybrid3 = rpu_resolve_kvinsert_segment_plan_auto(
        225, 51, 51, 8, 8, 128, all_caps);
    EXPECT(auto_hybrid3.route() == KvInsertRoute::HYBRID3 &&
               auto_hybrid3.selection_state() ==
                   KvInsertSelectionState::CAPABILITY_RANKED &&
               auto_hybrid3.candidate_count() == 2 &&
               auto_hybrid3.cost_score() == 21 &&
               auto_hybrid3.dependency_identity() ==
                   rpu_kvinsert_cost_certificate().dependency_identity &&
               auto_hybrid3.cost_certificate_identity() == 0 &&
               !rpu_kvinsert_cost_certificate().sealed,
           "AUTO chooses the certified unaligned head/bulk/tail route");
    const auto auto_padded = rpu_resolve_kvinsert_segment_plan_auto(
        0, 17, 32, 8, 8, 128, all_caps);
    EXPECT(auto_padded.route() == KvInsertRoute::PAD16_V16 &&
               auto_padded.candidate_count() == 2 &&
               auto_padded.cost_score() == 2,
           "AUTO keeps logical and padded physical rows distinct");
    EXPECT(rpu_resolve_kvinsert_segment_plan_auto(
               0, 31, 31, 8, 8, 128,
               KV_INSERT_CAP_V2).route() == KvInsertRoute::V2 &&
               rpu_resolve_kvinsert_segment_plan_auto(
                   0, 31, 31, 8, 8, 128,
                   KV_INSERT_CAP_V2).candidate_count() == 1,
           "AUTO falls back to the only certified route");

    const KvInsertCostDomain measured_domain{
        {111, 222, 444, 555, 1}, 225, 51, 51, 8, 8, 128,
        auto_hybrid3.feasible_route_mask()};
    KvInsertCostCertificate fixture{
        auto_hybrid3.dependency_identity(), 333, true,
        measured_domain, {0, 11, 0, 0, 0, 26}};
    EXPECT(rpu_kvinsert_cost_certificate_matches(fixture, measured_domain),
           "a synthetic sealed certificate matches only its measured domain");
    for (int field = 0; field < 12; ++field) {
        auto changed = measured_domain;
        switch (field) {
            case 0: ++changed.scope.profile_identity; break;
            case 1: ++changed.scope.site_identity; break;
            case 2: ++changed.position; break;
            case 3: ++changed.logical_rows; break;
            case 4: ++changed.physical_rows; break;
            case 5: changed.num_cores = 4; break;
            case 6: changed.num_kv_heads = 16; break;
            case 7: changed.head_dim = 256; break;
            case 8: changed.feasible_route_mask = 2; break;
            case 9: ++changed.scope.dependency_manifest_identity; break;
            case 10: ++changed.scope.runtime_identity; break;
            case 11: ++changed.scope.graph_lifecycle; break;
        }
        EXPECT(!rpu_kvinsert_cost_certificate_matches(fixture, changed),
               "a certificate cannot cross profile/site/shape/TP/route domains");
    }
    fixture.sealed = false;
    EXPECT(!rpu_kvinsert_cost_certificate_matches(fixture, measured_domain),
           "unsealed data cannot certify a route");
    fixture.sealed = true;
    ++fixture.dependency_identity;
    EXPECT(!rpu_kvinsert_cost_certificate_matches(fixture, measured_domain),
           "a dependency bump invalidates the sealed cost certificate");
    const auto scoped_auto = rpu_resolve_kvinsert_segment_plan_auto(
        225, 51, 51, 8, 8, 128, all_caps, measured_domain.scope);
    EXPECT(scoped_auto.cost_scope().profile_identity == 111 &&
               scoped_auto.cost_scope().site_identity == 222 &&
               scoped_auto.selection_state() == KvInsertSelectionState::CAPABILITY_RANKED &&
               scoped_auto.cost_certificate_identity() == 0 &&
               !rpu_kvinsert_cost_certificate(measured_domain).sealed,
           "a bound but unmeasured profile still reports capability selection");

    --fixture.dependency_identity;
    // Historical main65d49a7b9 dependency tuple: preserve its identity and
    // costs verbatim. A new BASE must not promote this old sealed receipt.
    uint64_t old_hash = 1469598103934665603ULL;
    for (const char* sha : {"bdc0dae56a6176c2fed6ae94f7c8c38a42bc7f06",
                           "5f339220245a617056fb461b431f064449a6ad41",
                           "f8d6b94b67d1be9416e35e79b5f8f94b03e7fab2"}) {
        for (; *sha; ++sha) { old_hash ^= uint8_t(*sha); old_hash *= 1099511628211ULL; }
        old_hash ^= uint8_t('|'); old_hash *= 1099511628211ULL;
    }
    const int64_t old_dependency = static_cast<int64_t>(old_hash & ((1ULL << 63) - 1));
    auto historical = fixture;
    historical.dependency_identity = old_dependency;
    const std::array<KvInsertCostCertificate, 1> historical_catalog{historical};
    EXPECT(old_dependency != rpu_kvinsert_cost_certificate().dependency_identity &&
               !rpu_kvinsert_cost_certificate_matches(historical, measured_domain),
           "new BASE tuple rejects the unchanged historical sealed receipt");
    const auto unmeasured = rpu_resolve_kvinsert_segment_plan_auto(
        225, 51, 51, 8, 8, 128, all_caps, measured_domain.scope, historical_catalog);
    EXPECT(unmeasured.selection_state() == KvInsertSelectionState::CAPABILITY_RANKED &&
               unmeasured.cost_certificate_identity() == 0 &&
               !unmeasured.cost_certificate().sealed &&
               unmeasured.dependency_identity() == rpu_kvinsert_cost_certificate().dependency_identity,
           "new tuple without its own measurement stays unsealed and capability ranked");
    EXPECT(historical_catalog[0].dependency_identity == old_dependency &&
               historical_catalog[0].sealed && historical_catalog[0].route_costs_ns == fixture.route_costs_ns,
           "historical receipt is neither mutated nor relabeled");
    const auto measured_plan = [&] {
        const std::vector<KvInsertCostCertificate> owner_catalog{fixture};
        return rpu_resolve_kvinsert_segment_plan_auto(
            225, 51, 51, 8, 8, 128, all_caps,
            measured_domain.scope, owner_catalog);
    }();
    EXPECT(measured_plan.route() == KvInsertRoute::V2 &&
               measured_plan.cost_score() == 11 &&
               measured_plan.cost_certificate_identity() == 333 &&
               measured_plan.selection_state() ==
                   KvInsertSelectionState::HARDWARE_COST_CERTIFIED,
           "complete-route device medians can beat structural grid ranking");
    // The catalog is gone. Both K/V validation and encoding use the plan's
    // value snapshot, with no process-global lookup or dangling reference.
    rpu_validate_kvinsert_segment_plan(measured_plan, 8, 8, 128);
    const auto measured_wire = rpu_kvinsert_route_arguments(
        measured_plan, 8, 8, 128);
    const std::vector<int64_t> measured_words(
        measured_wire.begin(), measured_wire.end());
    const std::array<KvInsertCostCertificate, 1> measured_catalog{fixture};
    const std::string artifact_sha256(64, 'a');
    constexpr int64_t artifact_projection = 0xaaaaaaaaaaaaaaaLL;
    const std::array<int64_t, 3> owner_identity{111, 444, 555};
    const std::vector<int64_t> catalog_words{
        2, 1, fixture.dependency_identity, artifact_projection,
        111, 222, 444, 555, 225, 51, 51, 8, 8, 128,
        static_cast<int64_t>(auto_hybrid3.feasible_route_mask()),
        0, 11, 0, 0, 0, 26, 1};
    const auto owner_catalog = KvInsertCostCatalog::from_arguments(
        owner_identity, artifact_sha256, catalog_words);
    auto historical_words = catalog_words;
    historical_words[2] = old_dependency;
    EXPECT(throws_c10([&] {
        (void)KvInsertCostCatalog::from_arguments(owner_identity, artifact_sha256, historical_words);
    }), "cold catalog admission rejects the old dependency tuple before installation");
    const std::vector<int64_t> six_core_catalog_words{
        2, 1, fixture.dependency_identity, artifact_projection,
        111, 222, 444, 555, 0, 260, 260, 6, 6, 64, 2,
        0, 11, 0, 0, 0, 0, 1};
    const auto six_core_catalog = KvInsertCostCatalog::from_arguments(
        owner_identity, artifact_sha256, six_core_catalog_words);
    const auto six_core_plan = rpu_resolve_kvinsert_segment_plan_auto(
        0, 260, 260, 6, 6, 64, all_caps | KV_INSERT_CAP_NON8_TP_V16,
        six_core_catalog.scope(222, 1), six_core_catalog.certificates());
    const auto six_core_wire = rpu_kvinsert_route_arguments(six_core_plan, 6, 6, 64);
    EXPECT(six_core_plan.feasible_route_mask() == 2 &&
               six_core_plan.cost_score() == 11 &&
               rpu_kvinsert_route_arguments(
                   rpu_kvinsert_segment_plan_from_route_arguments(
                       six_core_wire, 6, 6, 64, six_core_catalog.scope(222, 1),
                       six_core_catalog.certificates()), 6, 6, 64) == six_core_wire,
           "six-core V2 catalog and descriptor restore bind the actual TP domain");
    auto six_core_hybrid_catalog = six_core_catalog_words;
    six_core_hybrid_catalog[14] |= 1 << static_cast<int>(KvInsertRoute::HYBRID2);
    six_core_hybrid_catalog[15 + static_cast<int>(KvInsertRoute::HYBRID2)] = 1;
    EXPECT(throws_c10([&] {
        (void)KvInsertCostCatalog::from_arguments(
            owner_identity, artifact_sha256, six_core_hybrid_catalog);
    }), "a cost catalog cannot certify six-core hybrid feasibility");
    EXPECT(owner_catalog.bound() &&
               owner_catalog.artifact_sha256() == artifact_sha256 &&
               owner_catalog.scope(222, 1).site_identity == 222 &&
               owner_catalog.scope(222, 1).graph_lifecycle == 1,
           "owner catalog retains full artifact identity and exact stable site");
    const auto catalog_plan = rpu_resolve_kvinsert_segment_plan_auto(
        225, 51, 51, 8, 8, 128, all_caps,
        owner_catalog.scope(222, 1), owner_catalog.certificates());
    const auto catalog_wire = rpu_kvinsert_route_arguments(catalog_plan, 8, 8, 128);
    const auto catalog_restored = rpu_kvinsert_segment_plan_from_route_arguments(
        catalog_wire, 8, 8, 128,
        owner_catalog.scope(222, 1), owner_catalog.certificates());
    EXPECT(catalog_restored.route() == KvInsertRoute::V2 &&
               catalog_restored.cost_certificate_identity() == artifact_projection,
           "AUTO and descriptor restore consume the same immutable owner catalog");
    EXPECT(throws_c10([&] {
        (void)rpu_kvinsert_segment_plan_from_route_arguments(
            catalog_wire, 8, 8, 128,
            owner_catalog.scope(223, 1), owner_catalog.certificates());
    }), "a different physical site cannot restore a measured descriptor");
    for (int64_t lifecycle : {2, 3, 4}) {
        const auto different = rpu_resolve_kvinsert_segment_plan_auto(
            225, 51, 51, 8, 8, 128, all_caps,
            owner_catalog.scope(222, lifecycle), owner_catalog.certificates());
        EXPECT(different.selection_state() == KvInsertSelectionState::CAPABILITY_RANKED &&
                   different.cost_certificate_identity() == 0,
               "another lifecycle cannot borrow the measured cost winner");
        EXPECT(throws_c10([&] {
            (void)rpu_kvinsert_segment_plan_from_route_arguments(
                catalog_wire, 8, 8, 128,
                owner_catalog.scope(222, lifecycle), owner_catalog.certificates());
        }), "certified descriptor restore rejects a different actual lifecycle");
    }
    auto multiple_lifecycles = catalog_words;
    multiple_lifecycles[1] = 2;
    multiple_lifecycles.insert(multiple_lifecycles.end(),
                              catalog_words.begin() + 2, catalog_words.end());
    multiple_lifecycles[2 + kKvInsertCostCertificateWords + 13 + 1] = 99;
    multiple_lifecycles[2 + kKvInsertCostCertificateWords + 13 + 5] = 3;
    multiple_lifecycles.back() = 2;
    const auto two_lifecycles = KvInsertCostCatalog::from_arguments(
        owner_identity, artifact_sha256, multiple_lifecycles);
    const auto one_shot = rpu_resolve_kvinsert_segment_plan_auto(
        225, 51, 51, 8, 8, 128, all_caps,
        two_lifecycles.scope(222, 2), two_lifecycles.certificates());
    const auto retained = rpu_resolve_kvinsert_segment_plan_auto(
        225, 51, 51, 8, 8, 128, all_caps,
        two_lifecycles.scope(222, 1), two_lifecycles.certificates());
    EXPECT(one_shot.route() == KvInsertRoute::HYBRID3 && one_shot.cost_score() == 3 &&
               retained.route() == KvInsertRoute::V2 && retained.cost_score() == 11,
           "same site/geometry lifecycle rows coexist and select only their own costs");
    EXPECT(throws_c10([&] {
        (void)rpu_kvinsert_segment_plan_from_route_arguments(
            catalog_wire, 8, 8, 128,
            two_lifecycles.scope(222, 2), two_lifecycles.certificates());
    }), "a second lifecycle row cannot restore the first lifecycle's winner");
    for (int64_t lifecycle : {-1, 0, 5}) {
        auto invalid_lifecycle = catalog_words;
        invalid_lifecycle.back() = lifecycle;
        EXPECT(throws_c10([&] {
            (void)KvInsertCostCatalog::from_arguments(
                owner_identity, artifact_sha256, invalid_lifecycle);
        }), "catalog rejects an unknown Graph lifecycle");
        EXPECT(throws_c10([&] {
            (void)owner_catalog.scope(222, lifecycle);
        }), "consumer requires its actual known lifecycle");
    }
    auto old_wire = catalog_words;
    old_wire[0] = 1;
    old_wire.pop_back();
    EXPECT(throws_c10([&] {
        (void)KvInsertCostCatalog::from_arguments(
            owner_identity, artifact_sha256, old_wire);
    }), "old v1 catalog cannot silently omit lifecycle authority");
    const auto unmeasured_catalog = KvInsertCostCatalog::from_arguments(
        owner_identity, "", {2, 0});
    EXPECT(unmeasured_catalog.bound() && unmeasured_catalog.certificates().empty() &&
               unmeasured_catalog.artifact_sha256().empty(),
           "bound unmeasured catalog stays distinct from an unbound owner");
    EXPECT(!KvInsertCostCatalog().bound() &&
               KvInsertCostCatalog().scope(222, 1).profile_identity == 0,
           "unbound owner retains the structural fallback");
    EXPECT(throws_c10([&] { (void)owner_catalog.scope(0, 1); }),
           "cost scope cannot infer a missing stable site ID");
    for (size_t word : {size_t(0), size_t(1), size_t(2), size_t(3),
                        size_t(4), size_t(6), size_t(7), size_t(11),
                        size_t(14), size_t(15), size_t(16), size_t(20)}) {
        auto bad = catalog_words;
        bad[word] = word == 14 ? 64 : word == 15 ? 1 : 0;
        EXPECT(throws_c10([&] {
            (void)KvInsertCostCatalog::from_arguments(
                owner_identity, artifact_sha256, bad);
        }), "malformed schema/identity/mask/cost cannot enter an owner catalog");
    }
    for (size_t word : {size_t(8), size_t(9), size_t(10), size_t(12), size_t(13)}) {
        auto bad = catalog_words;
        bad[word] = -1;
        EXPECT(throws_c10([&] {
            (void)KvInsertCostCatalog::from_arguments(
                owner_identity, artifact_sha256, bad);
        }), "invalid physical geometry cannot enter an owner catalog");
    }
    auto duplicate = catalog_words;
    duplicate[1] = 2;
    duplicate.insert(duplicate.end(), catalog_words.begin() + 2, catalog_words.end());
    EXPECT(throws_c10([&] {
        (void)KvInsertCostCatalog::from_arguments(
            owner_identity, artifact_sha256, duplicate);
    }), "duplicate owner cost domains fail before publication");
    auto infeasible = catalog_words;
    infeasible[8] = 224;
    EXPECT(throws_c10([&] {
        (void)KvInsertCostCatalog::from_arguments(
            owner_identity, artifact_sha256, infeasible);
    }), "hybrid3 cannot claim an aligned-position domain");
    for (const std::string& bad_sha : {std::string(), std::string(64, 'A'),
                                      std::string(63, 'a')}) {
        EXPECT(throws_c10([&] {
            (void)KvInsertCostCatalog::from_arguments(
                owner_identity, bad_sha, catalog_words);
        }), "nonempty catalog requires its exact lowercase full SHA256");
    }
    EXPECT(throws_c10([&] {
        (void)KvInsertCostCatalog::from_arguments({0, 444, 555}, artifact_sha256,
                                                 catalog_words);
    }), "catalog cannot be installed for an unbound profile");
    EXPECT(throws_c10([&] {
        (void)KvInsertCostCatalog::from_arguments(owner_identity, artifact_sha256,
                                                 {2, 0});
    }), "an empty catalog cannot claim an artifact");
    const auto restored_measured = rpu_kvinsert_segment_plan_from_route_arguments(
        measured_words, 8, 8, 128, measured_domain.scope, measured_catalog);
    EXPECT(restored_measured.cost_score() == 11 &&
               restored_measured.route() == KvInsertRoute::V2,
           "certified wire restores only with the same scope and catalog");
    EXPECT(throws_c10([&] {
               (void)rpu_kvinsert_segment_plan_from_route_arguments(
                   measured_words, 8, 8, 128, measured_domain.scope);
           }), "missing catalog cannot restore certified route words");
    EXPECT(rpu_resolve_kvinsert_segment_plan_auto(
               225, 51, 51, 8, 8, 128, all_caps, measured_domain.scope).route() ==
               KvInsertRoute::HYBRID3,
           "one owner's injected costs never leak into another owner");
    EXPECT(throws_c10([&] {
               (void)rpu_resolve_kvinsert_segment_plan_auto(
                   225, 51, 51, 8, 8, 128, all_caps, {}, measured_catalog);
           }), "a catalog requires an independently bound owner scope");
    const std::array<KvInsertCostCertificate, 2> duplicates{fixture, fixture};
    EXPECT(throws_c10([&] {
               (void)rpu_resolve_kvinsert_segment_plan_auto(
                   225, 51, 51, 8, 8, 128, all_caps,
                   measured_domain.scope, duplicates);
           }), "duplicate certificates cannot choose by catalog order");
    for (int bad_route : {0, 3, 5}) {
        auto invalid = measured_catalog;
        invalid[0].route_costs_ns[bad_route] = bad_route == 5 ? 0 : 1;
        EXPECT(throws_c10([&] {
                   (void)rpu_resolve_kvinsert_segment_plan_auto(
                       225, 51, 51, 8, 8, 128, all_caps,
                       measured_domain.scope, invalid);
               }), "missing or out-of-domain measured costs fail closed");
    }

    const auto encoded_auto = rpu_kvinsert_route_arguments(
        auto_hybrid3, 8, 8, 128);
    const std::vector<int64_t> encoded_auto_vector(
        encoded_auto.begin(), encoded_auto.end());
    const auto decoded_auto =
        rpu_kvinsert_segment_plan_from_route_arguments(
            encoded_auto_vector, 8, 8, 128);
    EXPECT(decoded_auto.selection_state() ==
               KvInsertSelectionState::CAPABILITY_RANKED &&
               decoded_auto.route() == KvInsertRoute::HYBRID3 &&
               decoded_auto.feasible_route_mask() ==
                   auto_hybrid3.feasible_route_mask(),
           "ranked AUTO receipt round-trips and replays its domain winner");
    auto tampered_cost = encoded_auto_vector;
    ++tampered_cost[19];
    EXPECT(throws_c10([&] {
               (void)rpu_kvinsert_segment_plan_from_route_arguments(
                   tampered_cost, 8, 8, 128);
           }),
           "tampered AUTO cost receipt fails closed");
    auto tampered_dependency = encoded_auto_vector;
    ++tampered_dependency[20];
    EXPECT(throws_c10([&] {
               (void)rpu_kvinsert_segment_plan_from_route_arguments(
                   tampered_dependency, 8, 8, 128);
           }),
           "stale dependency-bound receipt fails closed");

    const auto decode_template = rpu_resolve_kvinsert_segment_plan(
        0, 1, 1, 8, 8, 128, KV_INSERT_CAP_V2, KvInsertRoute::V2);
    const auto decode_rebased = rpu_rebase_kvinsert_segment_plan_position(
        decode_template, 225, 8, 8, 128);
    const auto decode_rebased_args = rpu_kvinsert_route_arguments(
        decode_rebased, 8, 8, 128);
    EXPECT(decode_rebased.route() == decode_template.route() &&
               decode_rebased.segment_count() ==
                   decode_template.segment_count() &&
               decode_rebased.segment(0).position == 225 &&
               decode_rebased_args[6] == 225,
           "dynamic decode position preserves topology and regenerates words");
    EXPECT(throws_c10([&] {
               (void)rpu_rebase_kvinsert_segment_plan_position(
                   aligned, 17, 8, 8, 128);
           }),
           "dynamic position cannot silently change an aligned V16 route");

    EXPECT(throws_c10([&] {
               (void)rpu_resolve_kvinsert_segment_plan(
                   0, 32, 32, 8, 8, 128, KV_INSERT_CAP_V2,
                   KvInsertRoute::ALIGNED_V16);
           }),
           "uncertified exact route fails");
    EXPECT(throws_c10([&] {
               (void)rpu_resolve_kvinsert_segment_plan(
                   0, 17, 31, 8, 8, 128, all_caps,
                   KvInsertRoute::PAD16_V16);
           }),
           "non-exact physical padding fails");
    EXPECT(throws_c10([&] {
               (void)rpu_resolve_kvinsert_segment_plan(
                   0, 32, 32, 4, 8, 128, all_caps,
                   KvInsertRoute::ALIGNED_V16);
           }),
           "non-8 TP V16 requires its certified capability");
    const auto certified_non8 = rpu_resolve_kvinsert_segment_plan(
        0, 32, 32, 4, 8, 128,
        all_caps | KV_INSERT_CAP_NON8_TP_V16,
        KvInsertRoute::ALIGNED_V16);
    EXPECT(certified_non8.certified_capabilities() &
               KV_INSERT_CAP_NON8_TP_V16,
           "resolved non-8 TP plan retains its capability identity");
    EXPECT(throws_c10([&] {
               (void)rpu_resolve_kvinsert_segment_plan(
                   225, 51, 52, 8, 8, 128, all_caps,
                   KvInsertRoute::HYBRID3);
           }),
           "HYBRID3 with non-exact physical coverage fails");

    // Legacy precedence/shape table.  These cases mirror the old K and V
    // launchers, including the padded-V2 kill-switch fallback.
    const KvInsertLegacyPolicy defaults{};
    EXPECT(legacy(0, 32, 0, 8, false, false, defaults).route() ==
               KvInsertRoute::ALIGNED_V16,
           "legacy default selects aligned V16");
    EXPECT(legacy(1, 32, 0, 8, false, false, defaults).route() ==
               KvInsertRoute::V2,
           "legacy unaligned position falls back to V2");
    auto pad_and_hybrid = defaults;
    pad_and_hybrid.hybrid2_v16 = true;
    EXPECT(legacy(0, 17, 32, 8, false, false, pad_and_hybrid).route() ==
               KvInsertRoute::PAD16_V16,
           "legacy spm_rows PAD16 precedes enabled hybrid2");

    auto v16_off = defaults;
    v16_off.v16_enabled = false;
    const auto padded_v2 = legacy(0, 17, 32, 8, false, true, v16_off);
    EXPECT(padded_v2.route() == KvInsertRoute::V2 &&
               padded_v2.physical_rows() == 32 &&
               padded_v2.segment(0).rows == 32,
           "legacy v16 kill-switch preserves padded V2 emission");
    const auto padded_col_partition =
        rpu_resolve_legacy_kvinsert_segment_plan(
            0, 17, 32, 8, /*nkv=*/4, /*head_dim=*/128,
            /*allow_non8_v16=*/true, /*allow_hybrid_v16=*/true,
            defaults);
    EXPECT(padded_col_partition.route() == KvInsertRoute::V2 &&
               padded_col_partition.physical_rows() == 32,
           "legacy PAD capacity preserves padded V2 for a col partition");

    auto hybrid2_on = defaults;
    hybrid2_on.hybrid2_v16 = true;
    EXPECT(legacy(0, 255, 0, 8, false, false, hybrid2_on).route() ==
               KvInsertRoute::HYBRID2,
           "legacy shared/Rhino hybrid2 switch selects HYBRID2");
    EXPECT(legacy(0, 17, 17, 8, false, false, hybrid2_on).route() ==
               KvInsertRoute::HYBRID2,
           "legacy hybrid2 remains selected when SPM cannot cover PAD16");
    EXPECT(legacy(0, 255, 0, 8, false, true, defaults).route() ==
               KvInsertRoute::HYBRID2,
           "legacy explicit hybrid2 opt-in selects HYBRID2");

    auto hybrid3_on = defaults;
    hybrid3_on.hybrid3_v16 = true;
    EXPECT(legacy(225, 51, 0, 8, false, false, hybrid3_on).route() ==
               KvInsertRoute::HYBRID3,
           "legacy hybrid3 switch selects HYBRID3 before fallback");

    auto any_tp = defaults;
    any_tp.any_tp_v16 = true;
    EXPECT(legacy(0, 32, 0, 4, false, false, any_tp).route() ==
               KvInsertRoute::ALIGNED_V16,
           "legacy ANY_TP applies to whole V16");
    EXPECT(legacy(0, 31, 0, 4, false, true, any_tp).route() ==
               KvInsertRoute::V2,
           "legacy ANY_TP does not relax explicitly enabled HYBRID2");
    EXPECT(legacy(0, 31, 0, 4, true, true, defaults).route() ==
               KvInsertRoute::HYBRID2,
           "legacy explicit non8 and hybrid opt-ins compose");
    EXPECT(legacy(225, 51, 0, 4, false, false,
                  KvInsertLegacyPolicy{true, true, false, true}).route() ==
               KvInsertRoute::HYBRID3,
           "legacy ANY_TP applies to HYBRID3");
    EXPECT(legacy(0, 32, 0, 8, false, true, v16_off).route() ==
               KvInsertRoute::V2,
           "legacy v16 kill-switch dominates every v16 route");
    EXPECT(throws_c10([&] {
               (void)legacy(0, 32, 31, 8, false, false, defaults);
           }),
           "legacy short SPM capacity fails before launch");

    if (failed == 0) {
        std::cout << "KV-insert segment plan tests passed\n";
    }
    return failed == 0 ? 0 : 1;
}
