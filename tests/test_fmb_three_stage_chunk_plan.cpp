#include "core/fused_model_base.h"
#include "core/execution_topology.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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

template <typename Fn>
bool throws_c10_containing(Fn&& fn, const std::string& needle) {
    try {
        fn();
    } catch (const c10::Error& error) {
        return std::string(error.what_without_backtrace()).find(needle) !=
            std::string::npos;
    }
    return false;
}

std::vector<v3::ChunkInfo> input_chunks() {
    return {{0, 0, 24, 24}, {1, 24, 16, 40}};
}

std::vector<v3::ChunkInfo> attention_chunks() {
    return {{0, 0, 16, 16}, {1, 16, 16, 32}, {2, 32, 8, 40}};
}

std::vector<v3::FmbExecutionSpan> semantic_spans() {
    return {{0, 16, 0}, {16, 24, 1}};
}

v3::FmbStageBoundaryPolicies allow_cross() {
    return {};
}

}  // namespace

int main() {
    const auto wall_dual448_single_image =
        v3::compose_fmb_three_stage_chunk_plan(
            {{0, 0, 1024, 1024}},
            {{0, 0, 768, 768}, {1, 768, 256, 1024}},
            {{0, 0, 768, 768}, {1, 768, 256, 1024}},
            {{0, 1024}}, v3::ChunkMode::KV_FIRST);
    EXPECT(
        wall_dual448_single_image.compute.plan.chunk_size == 768 &&
            wall_dual448_single_image.compute.plan.num_chunks == 2 &&
            wall_dual448_single_image.compute.chunks.back().len == 256 &&
            v3::fmb_chunk_topology(wall_dual448_single_image) ==
                v3::FmbChunkTopology::SIMC,
        "Wall dual-448 executes each semantic image as exact 768+256 SIMC");


    const auto six = v3::resolve_decoder_execution_topology(
        6, 16, 8, 128, 1024, 3072);
    EXPECT(six.num_cores == 6 && six.attention_tp == 4 && six.mlp_tp == 6 &&
               six.physical_kv_cores == 8,
           "six-core Qwen3 preserves the physical KV ABI and uses legal per-layer TP");
    const auto eight = v3::resolve_decoder_execution_topology(
        8, 16, 8, 128, 1024, 3072);
    EXPECT(eight.attention_tp == 8 && eight.mlp_tp == 8,
           "default topology preserves eight-core weight layouts");
    EXPECT(v3::resolve_column_projection_cores(6, 151936) == 4 &&
               v3::resolve_column_projection_cores(6, 3072) == 6,
           "one aligned projection rule resolves vocabulary and MLP core counts");
    const auto four = v3::resolve_decoder_execution_topology(
        4, 16, 8, 128, 1024, 3072);
    EXPECT(four.num_cores == 4 && four.attention_tp == 4 && four.mlp_tp == 4 &&
               four.physical_kv_cores == 8,
           "four-core template keeps eight physical KV controller stripes");
    for (const auto& shape : std::vector<std::vector<int64_t>>{
            {1024, 3072, 16, 28}, {2048, 6144, 16, 28},
            {2560, 9728, 32, 36}, {4096, 12288, 32, 36}}) {
        EXPECT(v3::qwen3_fp16_profile_num_layers(
                   shape[2], 8, 128, shape[0], shape[1]) == shape[3],
               "fixed Qwen3 geometry binds its actual layer count");
        for (int budget : {4, 6, 8}) {
            const auto topology = v3::resolve_decoder_execution_topology(
                budget, shape[2], 8, 128, shape[0], shape[1]);
            EXPECT(topology.num_cores == budget &&
                       topology.attention_tp == (budget == 8 ? 8 : 4) &&
                       topology.mlp_tp == budget &&
                       topology.physical_kv_cores == 8,
                   "dense Qwen3 family uses fixed legal core templates");
            EXPECT(v3::decoder_mlp_intermediate_size(shape[1], topology.mlp_tp) ==
                       (budget == 6 && shape[1] == 9728 ? 9792 : shape[1]),
                   "physical padding is confined to 4B six-core MLP");
        }
        EXPECT(v3::qwen3_fp16_profile_num_layers(
                   shape[2], 8, 128, shape[0], shape[1] + 16) == 0 &&
                   v3::qwen3_fp16_profile_num_layers(
                       shape[2], 4, 128, shape[0], shape[1]) == 0 &&
                   v3::qwen3_fp16_profile_num_layers(
                       shape[2], 8, 64, shape[0], shape[1]) == 0,
               "aligned nearby dimensions do not manufacture a Qwen3 profile");
    }
    EXPECT(v3::qwen3_vl_text_prefix_num_layers(16, 8, 128, 2048, 6144) == 16 &&
               v3::qwen3_vl_text_prefix_num_layers(16, 8, 128, 1024, 3072) == 0 &&
               v3::qwen3_vl_text_prefix_num_layers(32, 8, 128, 2560, 9728) == 0,
           "only the GR00T 2B text prefix admits a truncated layer count");
    for (const int budget : {4, 5, 6}) {
        bool rejected = false;
        try {
            (void)v3::resolve_decoder_execution_topology(
                budget, 16, 8, 128, 1024, 2048);
        } catch (const std::invalid_argument&) { rejected = true; }
        EXPECT(rejected, "unadmitted reduced-core geometry must not invent a template");
    }
    const auto native_eight = v3::resolve_decoder_execution_topology(
        8, 24, 6, 128, 3072, 8192);
    EXPECT(native_eight.attention_tp == 6 && native_eight.mlp_tp == 8,
           "default eight budget retains existing non-eight attention TP");
    bool invalid_topology = false;
    try {
        (void)v3::resolve_decoder_execution_topology(0, 16, 8, 128, 1024, 3072);
    } catch (const std::invalid_argument&) { invalid_topology = true; }
    EXPECT(invalid_topology, "zero-core topology fails before division or launch");
    EXPECT(v3::fmb_ring_all_reduce_route_selector(40, 1024, 6) ==
               static_cast<int64_t>(v3::FmbSharedAllReduceRouteSelector::RING_PACED) &&
               v3::fmb_ring_all_reduce_route_selector(40, 1024, 8) ==
               static_cast<int64_t>(v3::FmbSharedAllReduceRouteSelector::RING_NOPACE),
           "ring pacing follows active consumer count across the 11 KiB crossover");
    EXPECT(throws_c10([] { v3::fmb_ring_all_reduce_route_selector(40, 1024, 9); }),
           "ring planner rejects a core budget beyond physical capacity");
    const auto wall_exact = v3::compose_fmb_three_stage_chunk_plan(
        {{0, 0, 1728, 1728}}, {{0, 0, 1728, 1728}},
        {{0, 0, 576, 576}, {1, 576, 576, 1152},
         {2, 1152, 576, 1728}},
        {{0, 576, 0}, {576, 576, 1}, {1152, 576, 2}},
        v3::ChunkMode::KV_FIRST,
        {v3::FmbSpanBoundaryPolicy::ALLOW_CROSS,
         v3::FmbSpanBoundaryPolicy::ALLOW_CROSS,
         v3::FmbSpanBoundaryPolicy::KEEP_LOCAL});
    EXPECT(wall_exact.input.plan.num_chunks == 1 &&
               wall_exact.qkv.plan.num_chunks == 1 &&
               wall_exact.compute.plan.num_chunks == 3 &&
               v3::fmb_chunk_topology(wall_exact) ==
                   v3::FmbChunkTopology::MIMC,
           "Wall exact resolves packed patch/QKV plus three compute chunks");

    const v3::FmbPrefillStageCandidate wall_candidate{
        /*input_chunk_size=*/1728,
        /*qkv_chunk_size=*/1728,
        /*compute_chunk_size=*/576,
        wall_exact};
    const std::vector<int64_t> wall_descriptor =
        v3::encode_fmb_prefill_stage_candidate(wall_candidate);
    const auto decoded_wall =
        v3::decode_fmb_prefill_stage_candidate(wall_descriptor);
    EXPECT(wall_descriptor[0] == 3 &&
               decoded_wall.input_chunk_size == 1728 &&
               decoded_wall.qkv_chunk_size == 1728 &&
               decoded_wall.compute_chunk_size == 576 &&
               decoded_wall.stage_plan.input.chunks.size() == 1 &&
               decoded_wall.stage_plan.input.chunks.front().len == 1728 &&
               decoded_wall.stage_plan.qkv.chunks.size() == 1 &&
               decoded_wall.stage_plan.compute.chunks.size() == 3 &&
               decoded_wall.stage_plan.spans.size() == 3 &&
               decoded_wall.stage_plan.boundary_policies.compute ==
                   v3::FmbSpanBoundaryPolicy::KEEP_LOCAL &&
               decoded_wall.stage_plan.chunk_mode ==
                   v3::ChunkMode::KV_FIRST &&
               decoded_wall.physical_manifest.state ==
                   v3::FmbPhysicalManifestState::UNSPECIFIED,
           "candidate descriptor preserves true input/QKV/compute schedules, "
           "spans, policies, mode, capacities, and migration state");
    EXPECT(wall_descriptor[5] != 0 || wall_descriptor[6] != 0,
           "candidate descriptor publishes a non-zero native plan fingerprint");

    auto stale_fingerprint = wall_descriptor;
    stale_fingerprint[6] ^= 1;
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::decode_fmb_prefill_stage_candidate(
                       stale_fingerprint);
               },
               "RPU_PLANNER_REJECT:CAPABILITY:"),
           "a stale descriptor fingerprint is rejected with a stable code");

    const size_t wall_stage_words = 15 +
        4 * (wall_exact.input.chunks.size() +
             wall_exact.qkv.chunks.size() +
             wall_exact.compute.chunks.size()) +
        3 * wall_exact.spans.size();
    auto legacy_v1_descriptor = std::vector<int64_t>(
        wall_descriptor.begin(),
        wall_descriptor.begin() + wall_stage_words);
    legacy_v1_descriptor[0] = 1;
    legacy_v1_descriptor[1] =
        static_cast<int64_t>(legacy_v1_descriptor.size());
    const auto decoded_legacy_v1 =
        v3::decode_fmb_prefill_stage_candidate(legacy_v1_descriptor);
    EXPECT(decoded_legacy_v1.stage_plan.compute.chunks.size() == 3 &&
               decoded_legacy_v1.physical_manifest.state ==
                   v3::FmbPhysicalManifestState::UNSPECIFIED,
           "candidate descriptor v1 remains decode compatible without "
           "inventing route authority");

    auto contaminated_unspecified = wall_descriptor;
    contaminated_unspecified[wall_stage_words + 1] = 1728;
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::decode_fmb_prefill_stage_candidate(
                       contaminated_unspecified);
               },
               "UNSPECIFIED physical manifest carries route authority"),
           "migration-state descriptors cannot smuggle physical authority");

    auto truncated_v3 = wall_descriptor;
    truncated_v3.pop_back();
    truncated_v3[1] = static_cast<int64_t>(truncated_v3.size());
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::decode_fmb_prefill_stage_candidate(truncated_v3);
               },
               "physical payload is truncated"),
           "candidate descriptor v3 rejects a truncated physical header");

    auto routed_wall_candidate = wall_candidate;
    routed_wall_candidate.physical_manifest = {
        v3::FmbPhysicalManifestState::COMPLETE,
        /*logical_length=*/1720,
        /*physical_length=*/1728,
        /*execution_padding_rows=*/8,
        /*kv_logical_length=*/1720,
        /*kv_insert_physical_rows=*/1728,
        v3::FmbGraphLifecycle::BOUNDED_ONESHOT,
        v3::FmbLinearAccumulationPolicy::MIXED_BY_SITE,
        {
            {/*site_id=*/101, v3::FmbRouteFamily::ATTENTION,
             /*selector=*/2, /*flags=*/0, {/*arguments=*/}},
            {/*site_id=*/201, v3::FmbRouteFamily::LINEAR,
             /*selector=*/3, /*flags=*/0, {16, 128}},
            {/*site_id=*/301, v3::FmbRouteFamily::KV_INSERT,
             /*selector=*/4, /*flags=*/1, {0, 1712, 16}},
            {/*site_id=*/401, v3::FmbRouteFamily::NORMALIZATION,
             /*selector=*/1, /*flags=*/0, {1728}},
            {/*site_id=*/501, v3::FmbRouteFamily::ACTIVATION,
             /*selector=*/2, /*flags=*/0, {}},
            {/*site_id=*/601, v3::FmbRouteFamily::GRAPH_SCHEDULE,
             /*selector=*/1, /*flags=*/0, {1}},
        }};
    const auto routed_wall_descriptor =
        v3::encode_fmb_prefill_stage_candidate(routed_wall_candidate);
    const auto decoded_routed_wall =
        v3::decode_fmb_prefill_stage_candidate(routed_wall_descriptor);
    EXPECT(decoded_routed_wall.physical_manifest.state ==
               v3::FmbPhysicalManifestState::COMPLETE &&
               decoded_routed_wall.physical_manifest.logical_length == 1720 &&
               decoded_routed_wall.physical_manifest.physical_length == 1728 &&
               decoded_routed_wall.physical_manifest.execution_padding_rows ==
                   8 &&
               decoded_routed_wall.physical_manifest.kv_logical_length ==
                   1720 &&
               decoded_routed_wall.physical_manifest.kv_insert_physical_rows ==
                   1728 &&
               decoded_routed_wall.physical_manifest.graph_lifecycle ==
                   v3::FmbGraphLifecycle::BOUNDED_ONESHOT &&
               decoded_routed_wall.physical_manifest.linear_accumulation ==
                   v3::FmbLinearAccumulationPolicy::MIXED_BY_SITE &&
               decoded_routed_wall.physical_manifest.routes.size() == 6 &&
               decoded_routed_wall.physical_manifest.routes[2].arguments ==
                   std::vector<int64_t>({0, 1712, 16}) &&
               decoded_routed_wall.physical_manifest.routes.back().family ==
                   v3::FmbRouteFamily::GRAPH_SCHEDULE,
           "candidate descriptor v3 round-trips physical lengths, lifecycle, "
           "ACC policy, and the complete per-site route manifest");

    // Opt-in ranking is wire authority and cache identity, while ordinary
    // owners keep their byte-compatible V3 descriptors.
    auto ranked_candidate = routed_wall_candidate;
    ranked_candidate.ranking = {v3::FmbPrefillRankingPolicy::OUTER_THEN_MIXER_V1, 32};
    const auto ranked_descriptor = v3::encode_fmb_prefill_stage_candidate(ranked_candidate);
    const auto ranked_decoded = v3::decode_fmb_prefill_stage_candidate(ranked_descriptor);
    EXPECT(ranked_descriptor[0] == 4 && routed_wall_descriptor[0] == 3 &&
               ranked_descriptor.size() == routed_wall_descriptor.size() + 2 &&
               ranked_decoded.ranking.mixer_calls == 32 &&
               ranked_decoded.ranking.policy == v3::FmbPrefillRankingPolicy::OUTER_THEN_MIXER_V1,
           "opt-in V4 round-trips ranking without changing legacy V3");
    EXPECT(!v3::detail::fmb_prefill_stage_candidate_same_identity(
               ranked_candidate, routed_wall_candidate),
           "ranking policy participates in candidate identity");
    v3::FmbPreparedStageCandidateCache ranking_cache;
    auto prepared_ranked = ranking_cache.prepare(ranked_descriptor);
    auto changed_ranked = ranked_descriptor;
    changed_ranked.back() = 33;
    EXPECT(prepared_ranked != ranking_cache.prepare(changed_ranked),
           "same manifest with a different metric cannot reuse a prepared descriptor");
    for (const int64_t policy : {0, 3}) {
        auto bad = ranked_descriptor;
        bad[bad.size() - 2] = policy;
        EXPECT(throws_c10([&] { v3::decode_fmb_prefill_stage_candidate(bad); }),
               "V4 rejects default or unknown policy tags");
    }
    for (const int64_t count : {0, 1729}) {
        auto bad = ranked_descriptor;
        bad.back() = count;
        EXPECT(throws_c10([&] { v3::decode_fmb_prefill_stage_candidate(bad); }),
               "V4 rejects missing or impossible mixer counts");
    }
    const auto make_ranked = [](int64_t capacity, int64_t calls,
            int64_t execution = 4096, int64_t logical = 4096,
            v3::FmbPrefillRankingPolicy policy = v3::FmbPrefillRankingPolicy::OUTER_THEN_MIXER_V1) {
        v3::LayoutContext layout;
        layout.chunk_size = capacity;
        auto plan = v3::compose_fmb_default_three_stage_chunk_plan(
            layout, execution, 0, v3::ChunkMode::SEQUENTIAL);
        v3::FmbPrefillStageCandidate candidate{execution, capacity, capacity, plan, {}};
        candidate.physical_manifest = {
            v3::FmbPhysicalManifestState::COMPLETE, logical, execution,
            execution-logical, logical, execution,
            v3::FmbGraphLifecycle::BOUNDED_ONESHOT,
            v3::FmbLinearAccumulationPolicy::ACC16,
            {{1, v3::FmbRouteFamily::GRAPH_SCHEDULE, 1, 0, {}}}};
        candidate.ranking = {policy, calls};
        return candidate;
    };
    const auto c704 = make_ranked(704, 32), c768 = make_ranked(768, 11);
    EXPECT(v3::detail::fmb_prefill_stage_candidate_score(c768) <
               v3::detail::fmb_prefill_stage_candidate_score(c704),
           "native AUTO rank agrees with Python: six outer calls then 11 vs 32 mixers");
    const auto old640 = make_ranked(640, 5, 1600, 1539);
    auto new640 = old640, new768 = make_ranked(768, 5, 1600, 1539);
    EXPECT(v3::detail::fmb_prefill_stage_candidate_score(old640) <
               v3::detail::fmb_prefill_stage_candidate_score(new768),
           "V1 remains tail-deficit-first on equal outer and mixer counts");
    new640.ranking.policy = new768.ranking.policy =
        v3::FmbPrefillRankingPolicy::OUTER_THEN_MIXER_LARGER_WINDOW_V2;
    EXPECT(v3::detail::fmb_prefill_stage_candidate_score(new768) <
               v3::detail::fmb_prefill_stage_candidate_score(new640),
           "V2 uses larger capacity before tail deficit, matching Python");
    const auto v2_descriptor = v3::encode_fmb_prefill_stage_candidate(new640);
    EXPECT(v3::decode_fmb_prefill_stage_candidate(v2_descriptor).ranking.policy ==
               v3::FmbPrefillRankingPolicy::OUTER_THEN_MIXER_LARGER_WINDOW_V2 &&
               ranking_cache.prepare(v2_descriptor) != ranking_cache.prepare(
                   v3::encode_fmb_prefill_stage_candidate(old640)),
           "V4 supports V1/V2 and separates policy versions in prepared cache");
    const auto less_padding = make_ranked(640, 3, 800, 769, new640.ranking.policy);
    const auto more_padding = make_ranked(768, 3, 832, 769, new640.ranking.policy);
    EXPECT(v3::detail::fmb_prefill_stage_candidate_score(less_padding) <
               v3::detail::fmb_prefill_stage_candidate_score(more_padding),
           "V2 prefers less actual execution padding before larger C");

    auto collective_candidate = routed_wall_candidate;
    collective_candidate.physical_manifest.routes.back().family =
        v3::FmbRouteFamily::COLLECTIVE;
    const auto collective_descriptor =
        v3::encode_fmb_prefill_stage_candidate(collective_candidate);
    const auto decoded_collective =
        v3::decode_fmb_prefill_stage_candidate(collective_descriptor);
    EXPECT(decoded_collective.physical_manifest.routes.back().family ==
               v3::FmbRouteFamily::COLLECTIVE,
           "candidate descriptor codec accepts the distinct collective "
           "route family");

    // V2 used a five-word route header and implicitly addressed invocation 0.
    // Removing only the new V3 invocation words must retain both its stored
    // fingerprint and decoded route identity.
    auto legacy_v2_routed = routed_wall_descriptor;
    size_t legacy_route_cursor = wall_stage_words + 11;
    const int64_t legacy_route_count =
        legacy_v2_routed[wall_stage_words + 8];
    for (int64_t ordinal = 0; ordinal < legacy_route_count; ++ordinal) {
        const int64_t argument_count = legacy_v2_routed[legacy_route_cursor + 4];
        legacy_v2_routed.erase(
            legacy_v2_routed.begin() + legacy_route_cursor + 5);
        legacy_route_cursor += 5 + static_cast<size_t>(argument_count);
    }
    legacy_v2_routed[0] = 2;
    legacy_v2_routed[1] = static_cast<int64_t>(legacy_v2_routed.size());
    const auto decoded_legacy_v2 =
        v3::decode_fmb_prefill_stage_candidate(legacy_v2_routed);
    EXPECT(decoded_legacy_v2.physical_manifest.routes.size() == 6 &&
               decoded_legacy_v2.physical_manifest.routes.front().invocation ==
                   0 &&
               v3::fmb_physical_manifest_fingerprint(
                   decoded_legacy_v2.physical_manifest) ==
                   v3::fmb_physical_manifest_fingerprint(
                       routed_wall_candidate.physical_manifest),
           "candidate descriptor v2 remains compatible as invocation zero");

    const auto decode_plan = v3::compose_fmb_three_stage_chunk_plan(
        {{0, 0, 1, 1}}, {{0, 0, 1, 1}}, {{0, 0, 1, 1}},
        {{0, 1}}, v3::ChunkMode::SEQUENTIAL);
    v3::FmbPrefillStageCandidate decode_candidate{
        /*input_chunk_size=*/1,
        /*qkv_chunk_size=*/16,
        /*compute_chunk_size=*/16,
        decode_plan,
        {v3::FmbPhysicalManifestState::COMPLETE,
         /*logical_length=*/1,
         /*physical_length=*/1,
         /*execution_padding_rows=*/0,
         /*kv_logical_length=*/64,
         /*kv_insert_physical_rows=*/1,
         v3::FmbGraphLifecycle::RETAINED_CACHE,
         v3::FmbLinearAccumulationPolicy::ACC16,
         {{/*site_id=*/1, v3::FmbRouteFamily::LINEAR,
           static_cast<int64_t>(v3::FmbLinearRouteSelector::GEMV),
           /*flags=*/0, {}}}}};
    const uint64_t decode_stage_fingerprint =
        v3::fmb_three_stage_chunk_plan_fingerprint(
            decode_candidate.stage_plan);
    const uint64_t decode_manifest_fingerprint =
        v3::fmb_physical_manifest_fingerprint(
            decode_candidate.physical_manifest);
    EXPECT(!throws_c10([&] {
               v3::rebase_fmb_retained_decode_candidate(
                   decode_candidate, /*position=*/17);
           }) &&
               decode_candidate.stage_plan.input.chunks.front().kv_seq_len ==
                   18 &&
               decode_candidate.stage_plan.qkv.chunks.front().kv_seq_len ==
                   18 &&
               decode_candidate.stage_plan.compute.chunks.front().kv_seq_len ==
                   18 &&
               v3::fmb_three_stage_chunk_plan_fingerprint(
                   decode_candidate.stage_plan) == decode_stage_fingerprint &&
               v3::fmb_physical_manifest_fingerprint(
                   decode_candidate.physical_manifest) ==
                   decode_manifest_fingerprint,
           "retained single-token decode rebases only its working KV schedule "
           "without changing Graph identities");

    auto bounded_decode = decode_candidate;
    bounded_decode.physical_manifest.graph_lifecycle =
        v3::FmbGraphLifecycle::BOUNDED_ONESHOT;
    EXPECT(throws_c10_containing(
               [&] {
                   v3::rebase_fmb_retained_decode_candidate(
                       bounded_decode, /*position=*/0);
               },
               "COMPLETE single-token RETAINED_CACHE"),
           "BOUNDED_ONESHOT cannot enter retained decode rebasing");
    auto multi_token_decode = decode_candidate;
    multi_token_decode.physical_manifest.logical_length = 2;
    multi_token_decode.physical_manifest.physical_length = 2;
    EXPECT(throws_c10_containing(
               [&] {
                   v3::rebase_fmb_retained_decode_candidate(
                       multi_token_decode, /*position=*/0);
               },
               "COMPLETE single-token RETAINED_CACHE"),
           "multi-token descriptors cannot enter retained decode rebasing");
    auto unspecified_decode = decode_candidate;
    unspecified_decode.physical_manifest = {};
    EXPECT(throws_c10_containing(
               [&] {
                   v3::rebase_fmb_retained_decode_candidate(
                       unspecified_decode, /*position=*/0);
               },
               "COMPLETE single-token RETAINED_CACHE"),
           "UNSPECIFIED descriptors cannot enter retained decode rebasing");

    const v3::FmbPhysicalManifestForwardCapability deny_default{};
    EXPECT(!throws_c10([&] {
               v3::validate_fmb_physical_manifest_forward_capability(
                   decoded_wall.physical_manifest, deny_default);
           }),
           "UNSPECIFIED descriptors retain legacy forward compatibility");
    EXPECT(throws_c10_containing(
               [&] {
                   v3::validate_fmb_physical_manifest_forward_capability(
                       decoded_routed_wall.physical_manifest, deny_default);
               },
               "explicit native forward capability"),
           "COMPLETE descriptors are denied by the default native capability");
    const v3::FmbPhysicalManifestForwardCapability bounded_capability{
        true, v3::FmbGraphLifecycle::BOUNDED_ONESHOT};
    EXPECT(!throws_c10([&] {
               v3::validate_fmb_physical_manifest_forward_capability(
                   decoded_routed_wall.physical_manifest,
                   bounded_capability);
           }),
           "an explicit matching native capability admits COMPLETE authority");
    const v3::FmbPhysicalManifestForwardCapability stale_lifecycle{
        true, v3::FmbGraphLifecycle::RETAINED_CACHE};
    EXPECT(throws_c10_containing(
               [&] {
                   v3::validate_fmb_physical_manifest_forward_capability(
                       decoded_routed_wall.physical_manifest,
                       stale_lifecycle);
               },
               "Graph lifecycle does not match"),
           "the native call lifecycle must match the COMPLETE descriptor");

    v3::FmbPhysicalManifestConsumer unspecified_consumer(
        decoded_wall.physical_manifest);
    EXPECT(!unspecified_consumer.complete() &&
               unspecified_consumer.fingerprint() == 0 &&
               !throws_c10([&] {
                   unspecified_consumer.require_all_consumed();
               }),
           "UNSPECIFIED manifests add no route or identity requirement");

    v3::FmbPhysicalManifestConsumer route_consumer(
        decoded_routed_wall.physical_manifest);
    EXPECT(route_consumer.complete() && route_consumer.fingerprint() != 0 &&
               route_consumer.manifest().routes.size() == 6,
           "consumer exposes a read-only COMPLETE manifest and fingerprint");
    EXPECT(throws_c10_containing(
               [&] {
                   (void)route_consumer.find_route(
                       v3::FmbRouteFamily::LINEAR, 101);
               },
               "wrong family"),
           "route lookup rejects a site consumed under the wrong family");
    EXPECT(throws_c10_containing(
               [&] {
                   (void)route_consumer.find_route(
                       v3::FmbRouteFamily::ATTENTION, 999);
               },
               "has no route"),
           "route lookup rejects an unregistered semantic site");
    EXPECT(throws_c10_containing(
               [&] {
                   (void)route_consumer.consume_route(
                       v3::FmbRouteFamily::ATTENTION, 101,
                       /*resolved_selector=*/3, /*resolved_flags=*/0);
               },
               "actually dispatched"),
           "route consumption rejects a locally selected different policy");
    EXPECT(route_consumer.attention_policy_for_site(101) ==
               v3::AttentionExecutionPolicy::SPM_KV_BY_MHA &&
               route_consumer.attention_layout_policy() ==
                   v3::AttentionExecutionPolicy::SPM_KV_BY_MHA,
           "ATTENTION selector 2 maps to the existing resolved SPM policy");
    const v3::FmbRouteManifestEntry ddr_attention{
        /*site_id=*/102, v3::FmbRouteFamily::ATTENTION,
        /*selector=*/1, /*flags=*/0, {/*arguments=*/}};
    EXPECT(v3::fmb_attention_execution_policy(ddr_attention) ==
               v3::AttentionExecutionPolicy::DDR_KV,
           "ATTENTION selector 1 maps to the existing resolved DDR policy");
    auto unknown_attention = ddr_attention;
    unknown_attention.selector = 3;
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::fmb_attention_execution_policy(
                       unknown_attention);
               },
               "not a resolved AttentionExecutionPolicy"),
           "ATTENTION rejects an unmapped selector instead of inventing a "
           "policy");
    auto attention_with_arguments = ddr_attention;
    attention_with_arguments.flags = 7;
    attention_with_arguments.arguments = {16, 4};
    EXPECT(v3::fmb_attention_execution_policy(attention_with_arguments) ==
               v3::AttentionExecutionPolicy::DDR_KV,
           "ATTENTION retains the shared policy selector while owner-specific "
           "flags and arguments identify the exact launcher variant");
    auto attention_abi_manifest = decoded_routed_wall.physical_manifest;
    attention_abi_manifest.routes.front().flags = 7;
    attention_abi_manifest.routes.front().arguments = {16, 4};
    v3::FmbPhysicalManifestConsumer attention_abi_consumer(
        attention_abi_manifest);
    EXPECT(!throws_c10([&] {
               (void)attention_abi_consumer.consume_route(
                   v3::FmbRouteFamily::ATTENTION, 101,
                   static_cast<int64_t>(
                       v3::AttentionExecutionPolicy::SPM_KV_BY_MHA),
                   /*resolved_flags=*/7, {16, 4});
           }),
           "ATTENTION consumes owner-specific flags and arguments exactly");
    EXPECT(throws_c10_containing(
               [&] {
                   (void)attention_abi_consumer.consume_route(
                       v3::FmbRouteFamily::ATTENTION, 101,
                       static_cast<int64_t>(
                           v3::AttentionExecutionPolicy::SPM_KV_BY_MHA),
                       /*resolved_flags=*/6, {16, 4});
               },
               "actually dispatched"),
           "ATTENTION rejects a launcher variant that differs from its "
           "descriptor flags");

    auto mixed_attention_manifest =
        decoded_routed_wall.physical_manifest;
    mixed_attention_manifest.routes.insert(
        mixed_attention_manifest.routes.begin() + 1, ddr_attention);
    v3::FmbPhysicalManifestConsumer mixed_attention_consumer(
        mixed_attention_manifest);
    EXPECT(mixed_attention_consumer.attention_policy_for_site(102) ==
               v3::AttentionExecutionPolicy::DDR_KV &&
               mixed_attention_consumer.attention_layout_policy() ==
                   v3::AttentionExecutionPolicy::SPM_KV_BY_MHA,
           "mixed per-site routes keep exact DDR authority while their union "
           "layout reserves raw-SPM attention buffers");
    EXPECT(!throws_c10([&] {
               (void)route_consumer.consume_attention_route(
                   101, v3::AttentionExecutionPolicy::SPM_KV_BY_MHA);
               // One source site may execute once per layer/chunk; repeating
               // the same exact receipt remains valid.
               (void)route_consumer.consume_attention_route(
                   101, v3::AttentionExecutionPolicy::SPM_KV_BY_MHA);
           }),
           "one semantic attention site may repeatedly consume its exact "
           "resolved policy");
    EXPECT(throws_c10_containing(
               [&] {
                   (void)mixed_attention_consumer.consume_attention_route(
                       102,
                       v3::AttentionExecutionPolicy::SPM_KV_BY_MHA);
               },
               "not the resolved policy actually dispatched"),
           "per-site attention receipt rejects the aggregate layout policy "
           "when that site selected DDR");
    v3::FmbPhysicalManifestConsumer omitted_route_consumer(
        decoded_routed_wall.physical_manifest);
    (void)omitted_route_consumer.consume_route(
        v3::FmbRouteFamily::ATTENTION, 101, 2, 0);
    EXPECT(throws_c10_containing(
               [&] { omitted_route_consumer.require_all_consumed(); },
               "was not consumed"),
           "forward completion rejects any omitted manifest route");
    const std::vector<int64_t> linear_arguments{16, 128};
    const std::vector<int64_t> kv_arguments{0, 1712, 16};
    (void)route_consumer.consume_route(
        v3::FmbRouteFamily::LINEAR, 201, 3, 0, linear_arguments);
    (void)route_consumer.consume_route(
        v3::FmbRouteFamily::KV_INSERT, 301, 4, 1, kv_arguments);
    (void)route_consumer.consume_route(
        v3::FmbRouteFamily::NORMALIZATION, 401, 1, 0, {1728});
    (void)route_consumer.consume_route(
        v3::FmbRouteFamily::ACTIVATION, 501, 2, 0);
    (void)route_consumer.consume_route(
        v3::FmbRouteFamily::GRAPH_SCHEDULE, 601, 1, 0, {1});
    EXPECT(!throws_c10([&] { route_consumer.require_all_consumed(); }),
           "forward completion accepts an exactly consumed route manifest");

    v3::FmbPhysicalManifestConsumer replay_receipt_consumer(
        decoded_routed_wall.physical_manifest);
    EXPECT(throws_c10_containing(
               [&] {
                   replay_receipt_consumer
                       .inherit_matching_graph_replay_receipt(
                           replay_receipt_consumer.fingerprint() ^ 1);
               },
               "does not match the COMPLETE physical manifest identity"),
           "fast replay cannot inherit route receipts from a different "
           "physical identity");
    EXPECT(!throws_c10([&] {
               replay_receipt_consumer
                   .inherit_matching_graph_replay_receipt(
                       replay_receipt_consumer.fingerprint());
               replay_receipt_consumer.require_all_consumed();
           }),
           "an exact matching Graph replay identity inherits the completed "
           "BUILD route receipt");

    v3::FmbPhysicalManifestConsumer external_prologue_consumer(
        decoded_routed_wall.physical_manifest);
    (void)external_prologue_consumer.consume_attention_route(
        101, v3::AttentionExecutionPolicy::SPM_KV_BY_MHA);
    EXPECT(!throws_c10([&] {
               external_prologue_consumer.retain_consumed_if_matching(
                   decoded_routed_wall.physical_manifest);
           }),
           "run_all_layers accepts a same-fingerprint external prologue");
    (void)external_prologue_consumer.consume_route(
        v3::FmbRouteFamily::LINEAR, 201, 3, 0, {16, 128});
    (void)external_prologue_consumer.consume_route(
        v3::FmbRouteFamily::KV_INSERT, 301, 4, 1, {0, 1712, 16});
    (void)external_prologue_consumer.consume_route(
        v3::FmbRouteFamily::NORMALIZATION, 401, 1, 0, {1728});
    (void)external_prologue_consumer.consume_route(
        v3::FmbRouteFamily::ACTIVATION, 501, 2, 0);
    (void)external_prologue_consumer.consume_route(
        v3::FmbRouteFamily::GRAPH_SCHEDULE, 601, 1, 0, {1});
    EXPECT(!throws_c10(
               [&] { external_prologue_consumer.require_all_consumed(); }),
           "same-fingerprint transfer retains prologue receipts until final "
           "completion");

    auto different_external_manifest =
        decoded_routed_wall.physical_manifest;
    different_external_manifest.routes[1].selector += 1;
    EXPECT(throws_c10_containing(
               [&] {
                   external_prologue_consumer.retain_consumed_if_matching(
                       different_external_manifest);
               },
               "does not match the run_all_layers candidate"),
           "external prologue transfer rejects a different fingerprint");

    v3::FmbPhysicalManifestConsumer external_replay_consumer(
        decoded_routed_wall.physical_manifest);
    (void)external_replay_consumer.consume_attention_route(
        101, v3::AttentionExecutionPolicy::SPM_KV_BY_MHA);
    external_replay_consumer.retain_consumed_if_matching(
        decoded_routed_wall.physical_manifest);
    EXPECT(!throws_c10([&] {
               external_replay_consumer.inherit_matching_graph_replay_receipt(
                   external_replay_consumer.fingerprint());
               external_replay_consumer.require_all_consumed();
           }),
           "same-fingerprint external REPLAY may inherit completed BUILD "
           "receipts");

    auto changed_route_identity = decoded_routed_wall.physical_manifest;
    changed_route_identity.routes[1].selector += 1;
    EXPECT(v3::fmb_physical_manifest_fingerprint(changed_route_identity) !=
               route_consumer.fingerprint(),
           "a route policy change produces a different physical identity");

    auto short_kv_logical = routed_wall_candidate;
    short_kv_logical.physical_manifest.kv_logical_length = 1719;
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::encode_fmb_prefill_stage_candidate(
                       short_kv_logical);
               },
               "KV lengths do not cover logical execution rows"),
           "KV logical length cannot borrow execution padding semantics");
    auto short_kv_insert = routed_wall_candidate;
    short_kv_insert.physical_manifest.kv_insert_physical_rows = 1719;
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::encode_fmb_prefill_stage_candidate(
                       short_kv_insert);
               },
               "KV lengths do not cover logical execution rows"),
           "KV-insert physical rows must cover this invocation's logical rows");

    auto short_positioned_kv = routed_wall_candidate;
    for (auto* stage : {&short_positioned_kv.stage_plan.input,
                        &short_positioned_kv.stage_plan.qkv,
                        &short_positioned_kv.stage_plan.compute}) {
        for (auto& chunk : stage->chunks) chunk.kv_seq_len += 32;
    }
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::encode_fmb_prefill_stage_candidate(
                       short_positioned_kv);
               },
               "KV lengths do not cover logical execution rows"),
           "KV logical length must cover position plus logical rows");

    auto stale_route = routed_wall_descriptor;
    // Keep the mutated selector inside the shared ATTENTION ABI so this case
    // reaches the independent fingerprint gate (SPM=2 -> DDR=1).
    stale_route[wall_stage_words + 11 + 2] ^= 3;
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::decode_fmb_prefill_stage_candidate(stale_route);
               },
               "physical manifest fingerprint is stale"),
           "route mutation is rejected by the independent manifest fingerprint");

    auto noncanonical_routes = routed_wall_candidate;
    std::swap(noncanonical_routes.physical_manifest.routes[0],
              noncanonical_routes.physical_manifest.routes[1]);
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::encode_fmb_prefill_stage_candidate(
                       noncanonical_routes);
               },
               "routes are not canonical and unique"),
           "complete route manifests require stable canonical site order");

    auto duplicate_routes = routed_wall_candidate;
    duplicate_routes.physical_manifest.routes[1] =
        duplicate_routes.physical_manifest.routes[0];
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::encode_fmb_prefill_stage_candidate(
                       duplicate_routes);
               },
               "routes are not canonical and unique"),
           "complete route manifests reject duplicate route identity");

    auto invoked_routes = routed_wall_candidate;
    auto second_attention_invocation =
        invoked_routes.physical_manifest.routes.front();
    second_attention_invocation.invocation = 1;
    invoked_routes.physical_manifest.routes.insert(
        invoked_routes.physical_manifest.routes.begin() + 1,
        second_attention_invocation);
    const auto invoked_descriptor =
        v3::encode_fmb_prefill_stage_candidate(invoked_routes);
    const auto decoded_invoked =
        v3::decode_fmb_prefill_stage_candidate(invoked_descriptor);
    v3::FmbPhysicalManifestConsumer invoked_consumer(
        decoded_invoked.physical_manifest);
    EXPECT(decoded_invoked.physical_manifest.routes[1].invocation == 1 &&
               invoked_consumer.find_route(
                   v3::FmbRouteFamily::ATTENTION, 101, 1).invocation == 1,
           "candidate descriptor v3 distinguishes repeated site invocations");
    (void)invoked_consumer.consume_attention_route(
        101, v3::AttentionExecutionPolicy::SPM_KV_BY_MHA, 0);
    (void)invoked_consumer.consume_attention_route(
        101, v3::AttentionExecutionPolicy::SPM_KV_BY_MHA, 1);
    (void)invoked_consumer.consume_route(
        v3::FmbRouteFamily::LINEAR, 201, 3, 0, linear_arguments);
    (void)invoked_consumer.consume_route(
        v3::FmbRouteFamily::KV_INSERT, 301, 4, 1, kv_arguments);
    (void)invoked_consumer.consume_route(
        v3::FmbRouteFamily::NORMALIZATION, 401, 1, 0, {1728});
    (void)invoked_consumer.consume_route(
        v3::FmbRouteFamily::ACTIVATION, 501, 2, 0);
    (void)invoked_consumer.consume_route(
        v3::FmbRouteFamily::GRAPH_SCHEDULE, 601, 1, 0, {1});
    EXPECT(!throws_c10([&] { invoked_consumer.require_all_consumed(); }),
           "consumer requires and accepts each repeated-site invocation");

    const std::vector<v3::ChunkInfo> shared_chunks{
        {0, 0, 64, 64}, {1, 64, 16, 80}};
    const auto shared_plan = v3::compose_fmb_three_stage_chunk_plan(
        shared_chunks, shared_chunks, shared_chunks, {{0, 80}},
        v3::ChunkMode::SEQUENTIAL);
    v3::FmbPhysicalExecutionManifest shared_manifest{
        v3::FmbPhysicalManifestState::COMPLETE,
        /*logical_length=*/80,
        /*physical_length=*/80,
        /*execution_padding_rows=*/0,
        /*kv_logical_length=*/80,
        /*kv_insert_physical_rows=*/80,
        v3::FmbGraphLifecycle::RETAINED_CACHE,
        v3::FmbLinearAccumulationPolicy::MIXED_BY_SITE,
        {}};
    constexpr uint32_t shared_route_mask =
        v3::FMB_SHARED_LAYER_INPUT_DMA |
        v3::FMB_SHARED_LAYER_INPUT_ROW_RUN_DMA |
        v3::FMB_SHARED_LAYER_INPUT_SCATTER_DMA |
        v3::FMB_SHARED_PIPELINE_INGRESS_DMA |
        v3::FMB_SHARED_MLP_AUTO_TILE |
        v3::FMB_SHARED_MLP_ACC32_OUT_BF16 |
        v3::FMB_SHARED_MLP_BF16_FP16_REDUCE |
        v3::FMB_SHARED_MLP_FP16_BF16_REDUCE |
        v3::FMB_SHARED_MLP_RING_REDUCE;
    v3::append_fmb_shared_runtime_routes(
        shared_manifest, shared_plan, /*hidden_size=*/1024,
        shared_route_mask);
    EXPECT(shared_manifest.routes.size() == 10 &&
               v3::fmb_ring_all_reduce_route_selector(64, 1024) ==
                   static_cast<int64_t>(v3::FmbSharedAllReduceRouteSelector::
                       RING_PACED) &&
               v3::fmb_ring_all_reduce_route_selector(16, 1024) ==
                   static_cast<int64_t>(v3::FmbSharedAllReduceRouteSelector::
                       RING_NOPACE) &&
               !throws_c10([&] {
                   v3::validate_fmb_physical_manifest(
                       shared_manifest, /*stage_physical_length=*/80,
                       /*position_base=*/0);
               }),
           "one shared augmentation publishes canonical FMB routes and a "
           "per-chunk ring invocation across the 11 KiB crossover");
    for (const int budget : {4, 6}) {
        auto reduced_manifest = shared_manifest;
        reduced_manifest.routes.clear();
        v3::append_fmb_shared_runtime_routes(
            reduced_manifest, shared_plan, 1024,
            v3::FMB_SHARED_LAYER_INPUT_DMA | v3::FMB_SHARED_MLP_AUTO_TILE,
            1, budget, budget);
        v3::FmbPhysicalManifestConsumer reduced_consumer(reduced_manifest);
        EXPECT(throws_c10([&] {
                   reduced_consumer.consume_route(
                       v3::FmbRouteFamily::MUTABLE_DMA,
                       v3::FMB_SHARED_LAYER_INPUT_DMA_SITE, 1, 0, {2, 8, 8});
               }) && !throws_c10([&] {
                   reduced_consumer.consume_route(
                       v3::FmbRouteFamily::MUTABLE_DMA,
                       v3::FMB_SHARED_LAYER_INPUT_DMA_SITE, 1, 0,
                       {2, budget, budget});
               }),
               "shared physical route rejects an eight-core descriptor under a reduced budget");
    }
    v3::FmbPhysicalManifestConsumer shared_consumer(shared_manifest);
    EXPECT(throws_c10([&] {
               shared_consumer.consume_route(
                   v3::FmbRouteFamily::MUTABLE_DMA,
                   v3::FMB_SHARED_LAYER_INPUT_SCATTER_DMA_SITE,
                   static_cast<int64_t>(v3::FmbSharedMutableDmaRouteSelector::DDR_BROADCAST_TO_SPM), 0);
           }) && !throws_c10([&] {
               shared_consumer.consume_route(
                   v3::FmbRouteFamily::MUTABLE_DMA,
                   v3::FMB_SHARED_LAYER_INPUT_SCATTER_DMA_SITE,
                   static_cast<int64_t>(v3::FmbSharedMutableDmaRouteSelector::DDR_SCATTER_TO_SPM), 0);
           }), "residual shard mutable DMA has a distinct scatter authority");
    auto partial_mlp_manifest = shared_manifest;
    v3::append_fmb_shared_runtime_routes(
        partial_mlp_manifest, shared_plan, 1024,
        v3::FMB_SHARED_MLP_PREPARE_INPUT, 1, 6, 4);
    v3::FmbPhysicalManifestConsumer partial_mlp_consumer(partial_mlp_manifest);
    EXPECT(!throws_c10([&] {
               partial_mlp_consumer.consume_route(
                   v3::FmbRouteFamily::ALL_REDUCE,
                   v3::FMB_SHARED_MLP_PREPARE_INPUT_SITE, 3, 0, {4, 6, 1});
           }) && throws_c10([&] {
               partial_mlp_consumer.consume_route(
                   v3::FmbRouteFamily::ALL_REDUCE,
                   v3::FMB_SHARED_MLP_PREPARE_INPUT_SITE, 3, 0, {4, 8, 1});
           }) && throws_c10([&] {
               shared_consumer.consume_route(
                   v3::FmbRouteFamily::ALL_REDUCE,
                   v3::FMB_SHARED_MLP_PREPARE_INPUT_SITE, 3, 0, {4, 6, 1});
           }),
           "partial MLP preparation requires the exact producer/consumer/clear descriptor");
    (void)shared_consumer.consume_route(
        v3::FmbRouteFamily::MUTABLE_DMA,
        v3::FMB_SHARED_LAYER_INPUT_DMA_SITE, 1, 0);
    (void)shared_consumer.consume_route(
        v3::FmbRouteFamily::MUTABLE_DMA,
        v3::FMB_SHARED_LAYER_INPUT_ROW_RUN_DMA_SITE, 1, 0);
    (void)shared_consumer.consume_route(
        v3::FmbRouteFamily::MUTABLE_DMA,
        v3::FMB_SHARED_PIPELINE_INGRESS_DMA_SITE, 2, 0);
    (void)shared_consumer.consume_route(
        v3::FmbRouteFamily::LINEAR,
        v3::FMB_SHARED_MLP_AUTO_TILE_SITE, 1, 0);
    (void)shared_consumer.consume_route(
        v3::FmbRouteFamily::LINEAR,
        v3::FMB_SHARED_MLP_ACC32_OUT_BF16_SITE, 4, 0);
    (void)shared_consumer.consume_route(
        v3::FmbRouteFamily::ALL_REDUCE,
        v3::FMB_SHARED_MLP_BF16_FP16_REDUCE_SITE, 5, 0);
    (void)shared_consumer.consume_route(
        v3::FmbRouteFamily::ALL_REDUCE,
        v3::FMB_SHARED_MLP_FP16_BF16_REDUCE_SITE, 4, 0);
    (void)shared_consumer.consume_route(
        v3::FmbRouteFamily::ALL_REDUCE,
        v3::FMB_SHARED_MLP_RING_REDUCE_SITE, 2, 0, {}, 0);
    (void)shared_consumer.consume_route(
        v3::FmbRouteFamily::ALL_REDUCE,
        v3::FMB_SHARED_MLP_RING_REDUCE_SITE, 1, 0, {}, 1);
    EXPECT(!throws_c10([&] { shared_consumer.require_all_consumed(); }),
           "shared dispatch receipts consume every augmented route");
    auto no_mlp_manifest = shared_manifest;
    no_mlp_manifest.routes.clear();
    v3::append_fmb_shared_runtime_routes(
        no_mlp_manifest, shared_plan, /*hidden_size=*/1024,
        v3::FMB_SHARED_LAYER_INPUT_DMA);
    v3::FmbPhysicalManifestConsumer no_mlp_consumer(no_mlp_manifest);
    (void)no_mlp_consumer.consume_route(
        v3::FmbRouteFamily::MUTABLE_DMA,
        v3::FMB_SHARED_LAYER_INPUT_DMA_SITE, 1, 0);
    EXPECT(!throws_c10([&] { no_mlp_consumer.require_all_consumed(); }),
           "a skip-down owner does not claim an unissued shared MLP route");
    auto falsely_claimed_mlp = no_mlp_manifest;
    falsely_claimed_mlp.routes.clear();
    v3::append_fmb_shared_runtime_routes(
        falsely_claimed_mlp, shared_plan, /*hidden_size=*/1024,
        v3::FMB_SHARED_LAYER_INPUT_DMA | v3::FMB_SHARED_MLP_AUTO_TILE);
    v3::FmbPhysicalManifestConsumer false_claim_consumer(falsely_claimed_mlp);
    (void)false_claim_consumer.consume_route(
        v3::FmbRouteFamily::MUTABLE_DMA,
        v3::FMB_SHARED_LAYER_INPUT_DMA_SITE, 1, 0);
    EXPECT(throws_c10_containing(
               [&] { false_claim_consumer.require_all_consumed(); },
               "was not consumed"),
           "a falsely declared shared MLP route fails closed");
    const std::vector<v3::ChunkInfo> one_row{{0, 0, 1, 1}};
    const auto batch_plan = v3::compose_fmb_three_stage_chunk_plan(
        one_row, one_row, one_row, {{0, 1}}, v3::ChunkMode::SEQUENTIAL);
    auto gemv_manifest = shared_manifest;
    gemv_manifest.routes.clear();
    gemv_manifest.logical_length = gemv_manifest.physical_length = 1;
    gemv_manifest.linear_accumulation = v3::FmbLinearAccumulationPolicy::ACC16;
    v3::append_fmb_shared_runtime_routes(
        gemv_manifest, batch_plan, 2560, v3::FMB_SHARED_MLP_GEMV);
    v3::FmbPhysicalManifestConsumer gemv_consumer(gemv_manifest);
    EXPECT(throws_c10([&] {
        gemv_consumer.consume_route(v3::FmbRouteFamily::LINEAR,
            v3::FMB_SHARED_MLP_AUTO_TILE_SITE, 1, 0);
    }), "GEMV descriptor rejects an AUTO_TILE dispatch");
    (void)gemv_consumer.consume_route(v3::FmbRouteFamily::LINEAR,
        v3::FMB_SHARED_MLP_AUTO_TILE_SITE, 2, 0);
    EXPECT(!throws_c10([&] { gemv_consumer.require_all_consumed(); }),
           "GEMV descriptor consumes the matching physical implementation");
    EXPECT(throws_c10([&] {
        v3::append_fmb_shared_runtime_routes(gemv_manifest, batch_plan, 2560,
            v3::FMB_SHARED_MLP_AUTO_TILE | v3::FMB_SHARED_MLP_GEMV);
    }), "shared MLP rejects conflicting Linear selectors");
    gemv_manifest.physical_length = 16;
    EXPECT(throws_c10([&] {
        v3::append_fmb_shared_runtime_routes(
            gemv_manifest, shared_plan, 2560, v3::FMB_SHARED_MLP_GEMV);
    }), "shared MLP rejects prefill GEMV before dispatch");
    auto batch_manifest = shared_manifest;
    batch_manifest.routes.clear();
    batch_manifest.logical_length = 1;
    batch_manifest.physical_length = 1;
    batch_manifest.kv_logical_length = 1;
    batch_manifest.kv_insert_physical_rows = 1;
    v3::append_fmb_shared_runtime_routes(
        batch_manifest, batch_plan, /*hidden_size=*/4096,
        v3::FMB_SHARED_MLP_RING_REDUCE,
        /*compute_row_multiplier=*/16);
    EXPECT(batch_manifest.routes.front().selector ==
               static_cast<int64_t>(v3::FmbSharedAllReduceRouteSelector::
                   RING_PACED),
           "shared ring authority includes the batch-row multiplier");
    EXPECT(throws_c10_containing(
               [&] {
                   auto invalid = shared_manifest;
                   v3::append_fmb_shared_runtime_routes(
                       invalid, shared_plan, /*hidden_size=*/1024,
                       1U << 31);
               },
               "known route mask"),
           "shared augmentation rejects unknown owner route claims");

    const std::vector<int64_t> domain_descriptor =
        v3::encode_fmb_prefill_stage_domain(
            {wall_candidate, wall_candidate});
    EXPECT(domain_descriptor.size() ==
               static_cast<size_t>(4 + 2 * wall_descriptor.size()) &&
               domain_descriptor[0] == 1 && domain_descriptor[1] == 2 &&
               domain_descriptor[2] ==
                   static_cast<int64_t>(wall_descriptor.size()) &&
               domain_descriptor[3 + wall_descriptor.size()] ==
                   static_cast<int64_t>(wall_descriptor.size()),
           "domain descriptor frames every complete native candidate");

    auto route_variant = routed_wall_candidate;
    route_variant.physical_manifest.routes.front().selector = 1;
    std::vector<v3::FmbPrefillStageCandidate> physical_domain{
        routed_wall_candidate, route_variant};
    std::sort(
        physical_domain.begin(), physical_domain.end(),
        v3::detail::fmb_prefill_stage_candidate_less);
    physical_domain.erase(
        std::unique(
            physical_domain.begin(), physical_domain.end(),
            v3::detail::fmb_prefill_stage_candidate_same_identity),
        physical_domain.end());
    EXPECT(physical_domain.size() == 2,
           "same-chunk candidates with different physical manifests remain "
           "distinct in the native domain");

    auto raw_spm = routed_wall_candidate;
    raw_spm.physical_manifest.routes = {
        {/*site_id=*/101, v3::FmbRouteFamily::ATTENTION,
         static_cast<int64_t>(v3::AttentionExecutionPolicy::SPM_KV_BY_MHA),
         /*flags=*/0, {}},
        {/*site_id=*/401, v3::FmbRouteFamily::ROPE,
         /*selector=*/2, v3::FMB_ROUTE_FLAG_ROPE_TABLE_SPM, {}},
    };
    auto raw_ddr = raw_spm;
    raw_ddr.physical_manifest.routes.back().selector = 1;
    raw_ddr.physical_manifest.routes.back().flags =
        v3::FMB_ROUTE_FLAG_ROPE_TABLE_DDR;
    auto ddr_spm = raw_spm;
    ddr_spm.physical_manifest.routes.front().selector =
        static_cast<int64_t>(v3::AttentionExecutionPolicy::DDR_KV);
    EXPECT(
        v3::fmb_rope_table_residency(raw_spm.physical_manifest) ==
                v3::FmbRopeTableResidency::SPM &&
            v3::fmb_rope_table_residency(raw_ddr.physical_manifest) ==
                v3::FmbRopeTableResidency::DDR,
        "the COMPLETE manifest owns 2D RoPE table residency");
    std::vector<v3::FmbPrefillStageCandidate> ranked_physical_domain{
        ddr_spm, raw_ddr, raw_spm};
    std::sort(
        ranked_physical_domain.begin(), ranked_physical_domain.end(),
        v3::detail::fmb_prefill_stage_candidate_less);
    EXPECT(
        v3::fmb_rope_table_residency(
            ranked_physical_domain[0].physical_manifest) ==
                v3::FmbRopeTableResidency::SPM &&
            v3::fmb_rope_table_residency(
                ranked_physical_domain[1].physical_manifest) ==
                v3::FmbRopeTableResidency::DDR &&
            ranked_physical_domain[2].physical_manifest.routes.front()
                    .selector ==
                static_cast<int64_t>(
                    v3::AttentionExecutionPolicy::DDR_KV),
        "same-chunk candidates prefer fewer DDR attention sites, then SPM "
        "RoPE residency, before fingerprint total ordering");

    auto conflicting_rope = raw_spm.physical_manifest;
    conflicting_rope.routes.back().flags |=
        v3::FMB_ROUTE_FLAG_ROPE_TABLE_DDR;
    EXPECT(throws_c10_containing(
               [&] {
                   (void)v3::fmb_rope_table_residency(conflicting_rope);
               },
               "conflicting DDR/SPM"),
           "one ROPE route cannot claim both DDR and SPM residency");
    auto mixed_rope = raw_spm.physical_manifest;
    mixed_rope.routes.push_back(
        {/*site_id=*/402, v3::FmbRouteFamily::ROPE,
         /*selector=*/1, v3::FMB_ROUTE_FLAG_ROPE_TABLE_DDR, {}});
    EXPECT(throws_c10_containing(
               [&] { (void)v3::fmb_rope_table_residency(mixed_rope); },
               "mixes ROPE table residencies"),
           "one COMPLETE manifest cannot mix DDR and SPM RoPE residency");

    const auto one_tail_plan = v3::compose_fmb_three_stage_chunk_plan(
        {{0, 0, 225, 225}}, {{0, 0, 225, 225}},
        {{0, 0, 225, 225}}, {{0, 225}},
        v3::ChunkMode::SEQUENTIAL);
    const auto one_tail_descriptor =
        v3::encode_fmb_prefill_stage_candidate(
            {/*input_chunk_size=*/225,
             /*qkv_chunk_size=*/240,
             /*compute_chunk_size=*/240,
             one_tail_plan});
    const auto decoded_one_tail =
        v3::decode_fmb_prefill_stage_candidate(one_tail_descriptor);
    EXPECT(decoded_one_tail.compute_chunk_size == 240 &&
               decoded_one_tail.qkv_chunk_size == 240 &&
               decoded_one_tail.stage_plan.compute.chunks.front().len == 225,
           "descriptor keeps v16 capacity distinct from a short physical tail");
    for (const int64_t rows : {INT64_C(82), INT64_C(41)}) {
        const int64_t capacity = ((rows + 15) / 16) * 16;
        const auto plan = v3::compose_fmb_three_stage_chunk_plan(
            {{0, 0, rows, rows}}, {{0, 0, rows, rows}},
            {{0, 0, rows, rows}}, {{0, rows}}, v3::ChunkMode::SEQUENTIAL);
        const auto encoded = v3::encode_fmb_prefill_stage_candidate(
            {rows, capacity, capacity, plan});
        const auto decoded = v3::decode_fmb_prefill_stage_candidate(encoded);
        EXPECT(decoded.compute_chunk_size == capacity &&
                   decoded.stage_plan.compute.chunks.front().len == rows &&
                   decoded.stage_plan.qkv.chunks.front().len == rows,
               "GR00T Z2 keeps logical rows and aligned capacity separate");
        EXPECT(throws_c10_containing(
                   [&] { (void)v3::encode_fmb_prefill_stage_candidate(
                       {rows, rows, rows, plan}); },
                   "16-aligned"),
               "logical GR00T rows cannot be used as physical capacity");
    }

    v3::detail::validate_fmb_exact_chunk_size(
        /*requested_chunk_size=*/256, /*execution_len=*/256,
        "host exact contract");
    EXPECT(throws_c10_containing(
               [&] {
                   v3::detail::validate_fmb_exact_chunk_size(
                       /*requested_chunk_size=*/272,
                       /*execution_len=*/256,
                       "host exact contract");
               },
               "RPU_PLANNER_REJECT:EXACT_MISMATCH:"),
           "planned EXACT above ceil16(execution_len) rejects instead of clamps");

    EXPECT(throws_c10([&] {
               v3::compose_fmb_three_stage_chunk_plan(
                   input_chunks(), attention_chunks(), attention_chunks(),
                   semantic_spans(), v3::ChunkMode::SEQUENTIAL);
           }),
           "multi-span callers cannot inherit an implicit all-cross policy");
    EXPECT(throws_c10([&] {
               v3::compose_fmb_three_stage_chunk_plan(
                   input_chunks(), attention_chunks(), attention_chunks(),
                   {{0, 16}, {16, 24}}, v3::ChunkMode::SEQUENTIAL,
                   allow_cross());
           }),
           "multi-span callers must publish explicit group IDs");

    const std::vector<v3::FmbExecutionSpan> one_group{
        {0, 16, 7}, {16, 24, 7}};
    const auto group_local = v3::compose_fmb_three_stage_chunk_plan(
        input_chunks(), attention_chunks(), attention_chunks(), one_group,
        v3::ChunkMode::SEQUENTIAL,
        {v3::FmbSpanBoundaryPolicy::GROUP_LOCAL,
         v3::FmbSpanBoundaryPolicy::ALLOW_CROSS,
         v3::FmbSpanBoundaryPolicy::ALLOW_CROSS});
    EXPECT(group_local.input.plan.num_chunks == 2,
           "GROUP_LOCAL permits a chunk to cross spans in one group");
    EXPECT(throws_c10([&] {
               v3::compose_fmb_three_stage_chunk_plan(
                   input_chunks(), attention_chunks(), attention_chunks(),
                   semantic_spans(), v3::ChunkMode::SEQUENTIAL,
                   {v3::FmbSpanBoundaryPolicy::KEEP_LOCAL,
                    v3::FmbSpanBoundaryPolicy::ALLOW_CROSS,
                    v3::FmbSpanBoundaryPolicy::ALLOW_CROSS});
           }),
           "KEEP_LOCAL rejects a chunk that crosses a span boundary");

    const auto sequential = v3::compose_fmb_three_stage_chunk_plan(
        input_chunks(), attention_chunks(), attention_chunks(), semantic_spans(),
        v3::ChunkMode::SEQUENTIAL, allow_cross());
    EXPECT(sequential.compute.plan.chunk_size == 16 &&
               sequential.compute.plan.num_chunks == 3 &&
               sequential.compute.chunks.back().len == 8,
           "SEQUENTIAL accepts equal qkv/compute schedules with a final tail");

    const auto kv_first = v3::compose_fmb_three_stage_chunk_plan(
        input_chunks(), attention_chunks(), attention_chunks(), semantic_spans(),
        v3::ChunkMode::KV_FIRST, allow_cross());
    EXPECT(kv_first.chunk_mode == v3::ChunkMode::KV_FIRST,
           "KV_FIRST accepts an equal resolved physical schedule");

    const auto divergent_kv_first =
        v3::compose_fmb_three_stage_chunk_plan(
            input_chunks(), {{0, 0, 40, 40}}, attention_chunks(),
            semantic_spans(), v3::ChunkMode::KV_FIRST, allow_cross());
    EXPECT(divergent_kv_first.qkv.plan.chunk_size == 40 &&
               divergent_kv_first.compute.plan.chunk_size == 16 &&
               v3::fmb_chunk_topology(divergent_kv_first) ==
                   v3::FmbChunkTopology::MIMC,
           "KV_FIRST accepts one packed QKV chunk plus multi-chunk compute");

    EXPECT(throws_c10([&] {
               v3::compose_fmb_three_stage_chunk_plan(
                   input_chunks(), {{0, 0, 40, 40}}, attention_chunks(),
                   semantic_spans(), v3::ChunkMode::SEQUENTIAL,
                   allow_cross());
           }),
           "SEQUENTIAL rejects divergent qkv/compute schedules");

    const auto simc = v3::compose_fmb_three_stage_chunk_plan(
        {{0, 0, 40, 40}}, attention_chunks(), attention_chunks(),
        {{0, 40}}, v3::ChunkMode::SEQUENTIAL);
    EXPECT(v3::fmb_chunk_topology(simc) ==
               v3::FmbChunkTopology::SIMC,
           "one span plus multiple compute chunks classifies as SIMC");

    const auto sisc = v3::compose_fmb_three_stage_chunk_plan(
        {{0, 0, 40, 40}}, {{0, 0, 40, 40}}, {{0, 0, 40, 40}},
        {{0, 40}}, v3::ChunkMode::SEQUENTIAL);
    EXPECT(v3::fmb_chunk_topology(sisc) ==
               v3::FmbChunkTopology::SISC,
           "one span plus one compute chunk classifies as SISC");

    const auto misc = v3::compose_fmb_three_stage_chunk_plan(
        {{0, 0, 40, 40}}, {{0, 0, 40, 40}}, {{0, 0, 40, 40}},
        semantic_spans(), v3::ChunkMode::SEQUENTIAL, allow_cross());
    EXPECT(v3::fmb_chunk_topology(misc) ==
               v3::FmbChunkTopology::MISC,
           "multiple spans plus one compute chunk classifies as MISC");

    auto invalid_mode = sequential;
    invalid_mode.chunk_mode = static_cast<v3::ChunkMode>(2);
    EXPECT(throws_c10([&] {
               v3::validate_fmb_three_stage_chunk_plan(invalid_mode);
           }),
           "an unknown ChunkMode is rejected");

    auto oversized_single = v3::compose_fmb_three_stage_chunk_plan(
        {{0, 0, 40, 40}}, {{0, 0, 40, 40}}, {{0, 0, 40, 40}},
        semantic_spans(), v3::ChunkMode::SEQUENTIAL, allow_cross());
    oversized_single.compute.plan.chunk_size = 48;
    EXPECT(throws_c10([&] {
               v3::validate_fmb_three_stage_chunk_plan(oversized_single);
           }),
           "a one-chunk plan requires chunk_size to equal the resolved len");

    const std::vector<v3::ChunkInfo> invalid_tail = {
        {0, 0, 16, 16}, {1, 16, 24, 40}};
    EXPECT(throws_c10([&] {
               v3::compose_fmb_three_stage_chunk_plan(
                   input_chunks(), invalid_tail, invalid_tail, semantic_spans(),
                   v3::ChunkMode::SEQUENTIAL, allow_cross());
           }),
           "a final tail larger than chunk_size is rejected");

    EXPECT(throws_c10([&] {
               v3::compose_fmb_three_stage_chunk_plan(
                   input_chunks(), {{0, 0, 32, 32}}, {{0, 0, 32, 32}},
                   semantic_spans(), v3::ChunkMode::SEQUENTIAL,
                   allow_cross());
           }),
           "stage physical-length mismatch is rejected");

    auto bad_kv_position = sequential;
    bad_kv_position.compute.chunks[1].kv_seq_len = 31;
    EXPECT(throws_c10([&] {
               v3::validate_fmb_three_stage_chunk_plan(bad_kv_position);
           }),
           "a drifting kv_seq_len position base is rejected");

    auto cross_stage_position = sequential;
    for (auto& chunk : cross_stage_position.qkv.chunks) {
        chunk.kv_seq_len += 100;
    }
    for (auto& chunk : cross_stage_position.compute.chunks) {
        chunk.kv_seq_len += 100;
    }
    EXPECT(throws_c10([&] {
               v3::validate_fmb_three_stage_chunk_plan(
                   cross_stage_position);
           }),
           "a mismatched cross-stage position base is rejected");

    const auto nonzero_position = v3::compose_fmb_three_stage_chunk_plan(
        {{0, 0, 40, 140}}, {{0, 0, 16, 116}, {1, 16, 16, 132},
                            {2, 32, 8, 140}},
        {{0, 0, 16, 116}, {1, 16, 16, 132}, {2, 32, 8, 140}},
        {{0, 40}}, v3::ChunkMode::SEQUENTIAL);
    EXPECT(nonzero_position.compute.chunks.front().kv_seq_len == 116,
           "language/action plans accept one matching non-zero position base");

    v3::LayoutContext default_layout;
    default_layout.chunk_size = 16;
    default_layout.kv_insert_chunk_size = 40;
    const auto default_kv_first =
        v3::compose_fmb_default_three_stage_chunk_plan(
            default_layout, /*execution_len=*/40, /*position=*/100,
            v3::ChunkMode::KV_FIRST);
    EXPECT(default_kv_first.input.chunks.size() == 1 &&
               default_kv_first.qkv.chunks.size() == 1 &&
               default_kv_first.compute.chunks.size() == 3 &&
               default_kv_first.compute.chunks.front().kv_seq_len == 116,
           "default physical prepare plan matches KV_FIRST runtime chunks");

    const auto default_sequential =
        v3::compose_fmb_default_three_stage_chunk_plan(
            default_layout, /*execution_len=*/40, /*position=*/100,
            v3::ChunkMode::SEQUENTIAL);
    EXPECT(default_sequential.qkv.chunks.size() == 3 &&
               default_sequential.compute.chunks.size() == 3 &&
               default_sequential.qkv.chunks.front().len == 16,
           "SEQUENTIAL default plan ignores an inactive KV_FIRST chunk size");

    const uint64_t base_fingerprint =
        v3::fmb_three_stage_chunk_plan_fingerprint(sequential);
    auto different_spans = sequential;
    different_spans.spans = {{0, 8, 0}, {8, 32, 1}};
    EXPECT(base_fingerprint !=
               v3::fmb_three_stage_chunk_plan_fingerprint(different_spans),
           "semantic spans participate in the allocation identity");

    auto different_group = sequential;
    different_group.spans[1].group_id = 0;
    EXPECT(base_fingerprint !=
               v3::fmb_three_stage_chunk_plan_fingerprint(different_group),
           "span groups participate in the allocation identity");
    auto different_policy = sequential;
    different_policy.boundary_policies.qkv =
        v3::FmbSpanBoundaryPolicy::KEEP_LOCAL;
    EXPECT(base_fingerprint !=
               v3::fmb_three_stage_chunk_plan_fingerprint(different_policy),
           "boundary policies participate in the allocation identity");

    auto same_layout_later_position = sequential;
    for (auto* stage : {&same_layout_later_position.input,
                        &same_layout_later_position.qkv,
                        &same_layout_later_position.compute}) {
        for (auto& chunk : stage->chunks) chunk.kv_seq_len += 100;
    }
    EXPECT(base_fingerprint ==
               v3::fmb_three_stage_chunk_plan_fingerprint(
                   same_layout_later_position),
           "absolute decode position does not churn the allocation identity");

    EXPECT(v3::detail::resolve_attention_execution_policy(
               v3::AttentionExecutionPolicy::AUTO,
               /*eligible=*/true, /*fits=*/true) ==
               v3::AttentionExecutionPolicy::SPM_KV_BY_MHA,
           "AUTO selects raw-SPM KV only for an eligible fitting profile");
    EXPECT(v3::detail::resolve_attention_execution_policy(
               v3::AttentionExecutionPolicy::AUTO,
               /*eligible=*/true, /*fits=*/false) ==
               v3::AttentionExecutionPolicy::DDR_KV,
           "AUTO preserves DDR KV when the exact SPM layout is too large");
    EXPECT(v3::detail::resolve_attention_execution_policy(
               v3::AttentionExecutionPolicy::AUTO,
               /*eligible=*/false, /*fits=*/true) ==
               v3::AttentionExecutionPolicy::DDR_KV,
           "AUTO preserves DDR KV without a certified model/kernel profile");
    EXPECT(v3::detail::resolve_attention_execution_policy(
               v3::AttentionExecutionPolicy::DDR_KV,
               /*eligible=*/true, /*fits=*/true) ==
               v3::AttentionExecutionPolicy::DDR_KV,
           "an explicit DDR override is permanent even when SPM could fit");
    EXPECT(throws_c10([&] {
               v3::detail::resolve_attention_execution_policy(
                   v3::AttentionExecutionPolicy::SPM_KV_BY_MHA,
                   /*eligible=*/false, /*fits=*/true);
           }),
           "an unsupported explicit SPM request fails closed");

    auto span_gap = sequential;
    span_gap.spans[1].offset = 17;
    EXPECT(throws_c10([&] {
               v3::validate_fmb_three_stage_chunk_plan(span_gap);
           }),
           "a semantic-span gap is rejected");

    if (failed != 0) return 1;
    std::cout << "PASS: FMB three-stage chunk planning contract\n";
    return 0;
}
