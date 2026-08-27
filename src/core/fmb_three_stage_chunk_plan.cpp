#include "core/fused_model_base.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace v3 {

namespace {

ResolvedFmbStageChunks make_stage(std::vector<ChunkInfo> chunks) {
    return {{chunks.empty() ? 0 : chunks.front().len,
             static_cast<int64_t>(chunks.size())},
            std::move(chunks)};
}

struct ValidatedStage {
    int64_t total = 0;
    int64_t position_base = 0;
};

ValidatedStage validate_stage(const ResolvedFmbStageChunks& stage,
                              const char* role) {
    const int64_t total = detail::validate_resolved_chunk_coverage(
        stage.chunks, "FMB three-stage plan: ", role);
    TORCH_CHECK(
        stage.plan.chunk_size > 0 &&
            stage.plan.num_chunks ==
                static_cast<int64_t>(stage.chunks.size()),
        "FMB three-stage plan: ", role,
        " ChunkPlan does not match its resolved chunks");
    for (size_t i = 0; i < stage.chunks.size(); ++i) {
        const bool is_tail = i + 1 == stage.chunks.size();
        TORCH_CHECK(
            stage.chunks.size() == 1
                ? stage.chunks[i].len == stage.plan.chunk_size
                : (is_tail
                       ? stage.chunks[i].len <= stage.plan.chunk_size
                       : stage.chunks[i].len == stage.plan.chunk_size),
            "FMB three-stage plan: ", role,
            " resolved chunks do not match chunk_size");
    }
    return {total, stage.chunks.front().kv_seq_len -
                       stage.chunks.front().len};
}

bool same_schedule(const std::vector<ChunkInfo>& lhs,
                   const std::vector<ChunkInfo>& rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].idx != rhs[i].idx ||
            lhs[i].offset != rhs[i].offset ||
            lhs[i].len != rhs[i].len ||
            lhs[i].kv_seq_len != rhs[i].kv_seq_len) {
            return false;
        }
    }
    return true;
}

uint64_t mix_u64(uint64_t hash, uint64_t value) {
    constexpr uint64_t kPrime = 1099511628211ull;
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xffu;
        hash *= kPrime;
    }
    return hash;
}

uint64_t mix_chunks(uint64_t hash, const ResolvedFmbStageChunks& stage) {
    hash = mix_u64(hash, static_cast<uint64_t>(stage.chunks.size()));
    for (const ChunkInfo& chunk : stage.chunks) {
        hash = mix_u64(hash, static_cast<uint64_t>(chunk.idx));
        hash = mix_u64(hash, static_cast<uint64_t>(chunk.offset));
        hash = mix_u64(hash, static_cast<uint64_t>(chunk.len));
    }
    return hash;
}

}  // namespace

namespace detail {

int64_t validate_resolved_chunk_coverage(
    const std::vector<ChunkInfo>& chunks,
    const char* contract,
    const char* role) {
    TORCH_CHECK(!chunks.empty(), contract, role,
                " chunks must not be empty");
    TORCH_CHECK(chunks.front().len > 0 && chunks.front().offset == 0, contract,
                role,
                " chunks must start at offset zero with positive length");
    TORCH_CHECK(chunks.size() <=
                    static_cast<size_t>(std::numeric_limits<int>::max()),
                contract, role,
                " chunk count exceeds int indexing");
    const int64_t full_len = chunks.front().len;
    TORCH_CHECK(chunks.front().kv_seq_len >= full_len, contract, role,
                " first kv_seq_len is smaller than its chunk length");
    const int64_t position_base = chunks.front().kv_seq_len - full_len;
    int64_t expected_offset = 0;
    for (size_t ordinal = 0; ordinal < chunks.size(); ++ordinal) {
        const ChunkInfo& chunk = chunks[ordinal];
        TORCH_CHECK(chunk.idx == static_cast<int>(ordinal) &&
                        chunk.offset == expected_offset && chunk.len > 0,
                    contract, role,
                    " chunks require contiguous idx/offset and positive len");
        TORCH_CHECK(ordinal + 1 == chunks.size()
                        ? chunk.len <= full_len
                        : chunk.len == full_len,
                    contract, role,
                    " permits only one final tail chunk");
        TORCH_CHECK(expected_offset <=
                        std::numeric_limits<int64_t>::max() - chunk.len,
                    contract, role,
                    " total chunk length overflows int64");
        TORCH_CHECK(position_base <=
                        std::numeric_limits<int64_t>::max() - chunk.offset &&
                        position_base + chunk.offset <=
                            std::numeric_limits<int64_t>::max() - chunk.len,
                    contract, role,
                    " kv_seq_len calculation overflows int64");
        TORCH_CHECK(
            chunk.kv_seq_len == position_base + chunk.offset + chunk.len,
            contract, role,
            " kv_seq_len must advance with one position base");
        expected_offset += chunk.len;
    }
    return expected_offset;
}

AttentionExecutionPolicy resolve_attention_execution_policy(
    AttentionExecutionPolicy requested,
    bool model_kernel_eligible,
    bool exact_spm_layout_fits) {
    switch (requested) {
        case AttentionExecutionPolicy::AUTO:
            return model_kernel_eligible && exact_spm_layout_fits
                ? AttentionExecutionPolicy::SPM_KV_BY_MHA
                : AttentionExecutionPolicy::DDR_KV;
        case AttentionExecutionPolicy::DDR_KV:
            return AttentionExecutionPolicy::DDR_KV;
        case AttentionExecutionPolicy::SPM_KV_BY_MHA:
            TORCH_CHECK(
                model_kernel_eligible,
                "FMB attention policy: SPM_KV_BY_MHA was requested but the "
                "model/kernel profile is not eligible");
            TORCH_CHECK(
                exact_spm_layout_fits,
                "FMB attention policy: SPM_KV_BY_MHA was requested but the "
                "exact joint SPM layout does not fit");
            return AttentionExecutionPolicy::SPM_KV_BY_MHA;
        default:
            TORCH_CHECK(false,
                        "FMB attention policy: unsupported requested value");
    }
    return AttentionExecutionPolicy::DDR_KV;
}

}  // namespace detail

void validate_fmb_three_stage_chunk_plan(
    const FmbThreeStageChunkPlan& plan) {
    const ValidatedStage input = validate_stage(plan.input, "input");
    const ValidatedStage qkv = validate_stage(plan.qkv, "qkv");
    const ValidatedStage compute = validate_stage(plan.compute, "compute");
    TORCH_CHECK(
        input.total == qkv.total && qkv.total == compute.total,
        "FMB three-stage plan: input/qkv/compute must cover the same "
        "physical length");
    TORCH_CHECK(
        input.position_base == qkv.position_base &&
            qkv.position_base == compute.position_base,
        "FMB three-stage plan: input/qkv/compute require one matching "
        "position base");

    switch (plan.chunk_mode) {
        case ChunkMode::SEQUENTIAL:
            TORCH_CHECK(
                same_schedule(plan.qkv.chunks, plan.compute.chunks),
                "FMB three-stage plan: SEQUENTIAL requires matching qkv "
                "and compute schedules");
            break;
        case ChunkMode::KV_FIRST:
            break;
        default:
            TORCH_CHECK(false,
                        "FMB three-stage plan: unsupported ChunkMode");
    }

    TORCH_CHECK(!plan.spans.empty(),
                "FMB three-stage plan: spans must not be empty");
    int64_t span_total = 0;
    for (const FmbExecutionSpan& span : plan.spans) {
        TORCH_CHECK(
            span.offset == span_total && span.len > 0,
            "FMB three-stage plan: spans require contiguous offsets "
            "and positive lengths");
        TORCH_CHECK(
            span_total <= std::numeric_limits<int64_t>::max() - span.len,
            "FMB three-stage plan: span length overflows int64");
        span_total += span.len;
    }
    TORCH_CHECK(
        span_total == input.total,
        "FMB three-stage plan: spans must cover the physical length");
}

FmbChunkTopology fmb_chunk_topology(
    const FmbThreeStageChunkPlan& plan) {
    validate_fmb_three_stage_chunk_plan(plan);
    const bool multi_input = plan.spans.size() > 1;
    const bool multi_chunk = plan.compute.chunks.size() > 1;
    if (multi_input) {
        return multi_chunk
            ? FmbChunkTopology::MIMC
            : FmbChunkTopology::MISC;
    }
    return multi_chunk
        ? FmbChunkTopology::SIMC
        : FmbChunkTopology::SISC;
}

uint64_t fmb_three_stage_chunk_plan_fingerprint(
    const FmbThreeStageChunkPlan& plan) {
    validate_fmb_three_stage_chunk_plan(plan);
    uint64_t hash = 1469598103934665603ull;
    hash = mix_u64(hash, static_cast<uint64_t>(plan.chunk_mode));
    hash = mix_chunks(hash, plan.input);
    hash = mix_chunks(hash, plan.qkv);
    hash = mix_chunks(hash, plan.compute);
    hash = mix_u64(hash, static_cast<uint64_t>(plan.spans.size()));
    for (const FmbExecutionSpan& span : plan.spans) {
        hash = mix_u64(hash, static_cast<uint64_t>(span.offset));
        hash = mix_u64(hash, static_cast<uint64_t>(span.len));
    }
    return hash == 0 ? 1 : hash;
}

FmbThreeStageChunkPlan compose_fmb_default_three_stage_chunk_plan(
    const LayoutContext& layout,
    int64_t execution_len,
    int64_t position,
    ChunkMode chunk_mode) {
    TORCH_CHECK(execution_len > 0 && position >= 0,
                "FMB default three-stage plan requires positive execution "
                "length and non-negative position");
    TORCH_CHECK(layout.chunk_size > 0 && layout.effective_kv_cs() > 0,
                "FMB default three-stage plan requires positive compute/QKV "
                "chunk sizes");
    TORCH_CHECK(position <=
                    std::numeric_limits<int64_t>::max() - execution_len,
                "FMB default three-stage plan position overflows int64");

    auto split = [execution_len, position](int64_t chunk_size) {
        std::vector<ChunkInfo> chunks;
        for (int64_t offset = 0; offset < execution_len;) {
            const int64_t len =
                std::min(chunk_size, execution_len - offset);
            chunks.push_back({static_cast<int>(chunks.size()), offset, len,
                              position + offset + len});
            offset += len;
        }
        return chunks;
    };

    return compose_fmb_three_stage_chunk_plan(
        {{0, 0, execution_len, position + execution_len}},
        split(chunk_mode == ChunkMode::KV_FIRST
                  ? layout.effective_kv_cs()
                  : layout.chunk_size),
        split(layout.chunk_size),
        {{0, execution_len}}, chunk_mode);
}

FmbThreeStageChunkPlan compose_fmb_three_stage_chunk_plan(
    std::vector<ChunkInfo> input_chunks,
    std::vector<ChunkInfo> qkv_chunks,
    std::vector<ChunkInfo> compute_chunks,
    std::vector<FmbExecutionSpan> spans,
    ChunkMode chunk_mode) {
    FmbThreeStageChunkPlan result{
        make_stage(std::move(input_chunks)),
        make_stage(std::move(qkv_chunks)),
        make_stage(std::move(compute_chunks)),
        std::move(spans),
        chunk_mode,
    };
    validate_fmb_three_stage_chunk_plan(result);
    return result;
}

}  // namespace v3
