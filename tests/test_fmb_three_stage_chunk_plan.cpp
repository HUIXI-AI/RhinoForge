#include "core/fused_model_base.h"

#include <iostream>
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

std::vector<v3::ChunkInfo> input_chunks() {
    return {{0, 0, 24, 24}, {1, 24, 16, 40}};
}

std::vector<v3::ChunkInfo> attention_chunks() {
    return {{0, 0, 16, 16}, {1, 16, 16, 32}, {2, 32, 8, 40}};
}

std::vector<v3::FmbExecutionSpan> semantic_spans() {
    return {{0, 16}, {16, 24}};
}

}  // namespace

int main() {
    const auto wall_exact = v3::compose_fmb_three_stage_chunk_plan(
        {{0, 0, 1728, 1728}}, {{0, 0, 1728, 1728}},
        {{0, 0, 576, 576}, {1, 576, 576, 1152},
         {2, 1152, 576, 1728}},
        {{0, 576}, {576, 576}, {1152, 576}},
        v3::ChunkMode::KV_FIRST);
    EXPECT(wall_exact.input.plan.num_chunks == 1 &&
               wall_exact.qkv.plan.num_chunks == 1 &&
               wall_exact.compute.plan.num_chunks == 3 &&
               v3::fmb_chunk_topology(wall_exact) ==
                   v3::FmbChunkTopology::MIMC,
           "Wall exact resolves packed patch/QKV plus three compute chunks");

    const auto sequential = v3::compose_fmb_three_stage_chunk_plan(
        input_chunks(), attention_chunks(), attention_chunks(), semantic_spans(),
        v3::ChunkMode::SEQUENTIAL);
    EXPECT(sequential.compute.plan.chunk_size == 16 &&
               sequential.compute.plan.num_chunks == 3 &&
               sequential.compute.chunks.back().len == 8,
           "SEQUENTIAL accepts equal qkv/compute schedules with a final tail");

    const auto kv_first = v3::compose_fmb_three_stage_chunk_plan(
        input_chunks(), attention_chunks(), attention_chunks(), semantic_spans(),
        v3::ChunkMode::KV_FIRST);
    EXPECT(kv_first.chunk_mode == v3::ChunkMode::KV_FIRST,
           "KV_FIRST accepts an equal resolved physical schedule");

    const auto divergent_kv_first =
        v3::compose_fmb_three_stage_chunk_plan(
            input_chunks(), {{0, 0, 40, 40}}, attention_chunks(),
            semantic_spans(), v3::ChunkMode::KV_FIRST);
    EXPECT(divergent_kv_first.qkv.plan.chunk_size == 40 &&
               divergent_kv_first.compute.plan.chunk_size == 16 &&
               v3::fmb_chunk_topology(divergent_kv_first) ==
                   v3::FmbChunkTopology::MIMC,
           "KV_FIRST accepts one packed QKV chunk plus multi-chunk compute");

    EXPECT(throws_c10([&] {
               v3::compose_fmb_three_stage_chunk_plan(
                   input_chunks(), {{0, 0, 40, 40}}, attention_chunks(),
                   semantic_spans(), v3::ChunkMode::SEQUENTIAL);
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
        semantic_spans(), v3::ChunkMode::SEQUENTIAL);
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
        semantic_spans(), v3::ChunkMode::SEQUENTIAL);
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
                   v3::ChunkMode::SEQUENTIAL);
           }),
           "a final tail larger than chunk_size is rejected");

    EXPECT(throws_c10([&] {
               v3::compose_fmb_three_stage_chunk_plan(
                   input_chunks(), {{0, 0, 32, 32}}, {{0, 0, 32, 32}},
                   semantic_spans(), v3::ChunkMode::SEQUENTIAL);
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
    different_spans.spans = {{0, 8}, {8, 32}};
    EXPECT(base_fingerprint !=
               v3::fmb_three_stage_chunk_plan_fingerprint(different_spans),
           "semantic spans participate in the allocation identity");

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
