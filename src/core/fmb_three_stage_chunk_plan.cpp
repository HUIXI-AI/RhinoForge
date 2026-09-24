#include "core/fused_model_base.h"

#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
#include <utility>

namespace v3 {

namespace {

constexpr int64_t kPrefillStageCandidateDescriptorVersionV1 = 1;
constexpr int64_t kPrefillStageCandidateDescriptorVersionV2 = 2;
constexpr int64_t kPrefillStageCandidateDescriptorVersion = 3;
constexpr int64_t kPrefillStageDomainDescriptorVersion = 1;
constexpr size_t kPrefillStageCandidateHeaderWords = 15;
constexpr size_t kPhysicalManifestHeaderWords = 11;
constexpr size_t kMaxPrefillStageCandidateDescriptorWords = 65536;

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

uint64_t physical_manifest_fingerprint_impl(
    const FmbPhysicalExecutionManifest& manifest) {
    uint64_t hash = 1469598103934665603ull;
    // "FMB_PHY2" domain-separates this hash from the stage-plan fingerprint.
    hash = mix_u64(hash, 0x464d425f50485932ull);
    hash = mix_u64(hash, static_cast<uint64_t>(manifest.state));
    hash = mix_u64(hash, static_cast<uint64_t>(manifest.logical_length));
    hash = mix_u64(hash, static_cast<uint64_t>(manifest.physical_length));
    hash = mix_u64(
        hash, static_cast<uint64_t>(manifest.execution_padding_rows));
    hash = mix_u64(hash, static_cast<uint64_t>(manifest.kv_logical_length));
    hash = mix_u64(
        hash, static_cast<uint64_t>(manifest.kv_insert_physical_rows));
    hash = mix_u64(hash, static_cast<uint64_t>(manifest.graph_lifecycle));
    hash = mix_u64(
        hash, static_cast<uint64_t>(manifest.linear_accumulation));
    hash = mix_u64(hash, static_cast<uint64_t>(manifest.routes.size()));
    for (const FmbRouteManifestEntry& route : manifest.routes) {
        hash = mix_u64(hash, static_cast<uint64_t>(route.family));
        hash = mix_u64(hash, static_cast<uint64_t>(route.site_id));
        hash = mix_u64(hash, static_cast<uint64_t>(route.selector));
        hash = mix_u64(hash, static_cast<uint64_t>(route.flags));
        hash = mix_u64(hash, static_cast<uint64_t>(route.arguments.size()));
        for (int64_t argument : route.arguments) {
            hash = mix_u64(hash, static_cast<uint64_t>(argument));
        }
        // Invocation zero is the v1/v2 identity. Keeping it byte-identical
        // lets the v3 decoder accept persisted v2 descriptors.
        if (route.invocation != 0) {
            hash = mix_u64(hash, static_cast<uint64_t>(route.invocation));
        }
    }
    return hash == 0 ? 1 : hash;
}

void validate_physical_manifest_impl(
    const FmbPhysicalExecutionManifest& manifest,
    int64_t stage_physical_length,
    int64_t position_base) {
    const char* reject = "RPU_PLANNER_REJECT:CAPABILITY: ";
    const int64_t raw_state = static_cast<int64_t>(manifest.state);
    TORCH_CHECK(
        raw_state == static_cast<int64_t>(
                         FmbPhysicalManifestState::UNSPECIFIED) ||
            raw_state == static_cast<int64_t>(
                             FmbPhysicalManifestState::COMPLETE),
        reject, "physical manifest has an unknown state");
    if (manifest.state == FmbPhysicalManifestState::UNSPECIFIED) {
        TORCH_CHECK(
            manifest.logical_length == 0 &&
                manifest.physical_length == 0 &&
                manifest.execution_padding_rows == 0 &&
                manifest.kv_logical_length == 0 &&
                manifest.kv_insert_physical_rows == 0 &&
                manifest.graph_lifecycle == FmbGraphLifecycle::UNSPECIFIED &&
                manifest.linear_accumulation ==
                    FmbLinearAccumulationPolicy::UNSPECIFIED &&
                manifest.routes.empty(),
            reject, "UNSPECIFIED physical manifest carries route authority");
        return;
    }

    TORCH_CHECK(
        manifest.logical_length > 0 && manifest.physical_length > 0 &&
            manifest.execution_padding_rows >= 0 &&
            manifest.kv_logical_length > 0 &&
            manifest.kv_insert_physical_rows > 0,
        reject, "complete physical manifest has invalid lengths");
    TORCH_CHECK(
        manifest.logical_length <=
            std::numeric_limits<int64_t>::max() -
                manifest.execution_padding_rows &&
            manifest.logical_length + manifest.execution_padding_rows ==
                manifest.physical_length,
        reject, "complete physical manifest execution padding is inconsistent");
    TORCH_CHECK(
        manifest.physical_length == stage_physical_length,
        reject, "complete physical manifest length does not match stage plan");
    TORCH_CHECK(
        position_base <= std::numeric_limits<int64_t>::max() -
                manifest.logical_length &&
            manifest.kv_logical_length >=
                position_base + manifest.logical_length &&
            manifest.kv_insert_physical_rows >= manifest.logical_length,
        reject, "complete physical manifest KV lengths do not cover logical "
                "execution rows");
    const int64_t raw_lifecycle =
        static_cast<int64_t>(manifest.graph_lifecycle);
    TORCH_CHECK(
        raw_lifecycle >=
                static_cast<int64_t>(FmbGraphLifecycle::RETAINED_CACHE) &&
            raw_lifecycle <=
                static_cast<int64_t>(FmbGraphLifecycle::COMPOSITE_CHILD),
        reject, "complete physical manifest has no graph lifecycle");
    const int64_t raw_acc =
        static_cast<int64_t>(manifest.linear_accumulation);
    TORCH_CHECK(
        raw_acc >=
                static_cast<int64_t>(FmbLinearAccumulationPolicy::ACC16) &&
            raw_acc <= static_cast<int64_t>(
                           FmbLinearAccumulationPolicy::MIXED_BY_SITE),
        reject, "complete physical manifest has no linear ACC policy");
    TORCH_CHECK(
        !manifest.routes.empty(),
        reject, "complete physical manifest has no route entries");

    std::tuple<int64_t, int64_t, int64_t> previous_identity{-1, -1, -1};
    for (const FmbRouteManifestEntry& route : manifest.routes) {
        const int64_t family = static_cast<int64_t>(route.family);
        TORCH_CHECK(
            family >= static_cast<int64_t>(FmbRouteFamily::ATTENTION) &&
                family <= static_cast<int64_t>(
                              FmbRouteFamily::COLLECTIVE),
            reject, "physical manifest route has an unknown family");
        TORCH_CHECK(
            route.site_id > 0 && route.selector > 0 && route.flags >= 0 &&
                route.invocation >= 0,
            reject, "physical manifest route has invalid identity/policy");
        const auto identity =
            std::make_tuple(family, route.site_id, route.invocation);
        TORCH_CHECK(
            identity > previous_identity,
            reject, "physical manifest routes are not canonical and unique");
        for (int64_t argument : route.arguments) {
            TORCH_CHECK(
                argument >= 0,
                reject, "physical manifest route argument is negative");
        }
        if (route.family == FmbRouteFamily::ATTENTION) {
            (void)fmb_attention_execution_policy(route);
        }
        previous_identity = identity;
    }
    (void)fmb_rope_table_residency(manifest);
}

void validate_candidate_capacity(
    int64_t capacity,
    const ResolvedFmbStageChunks& stage,
    const char* role,
    bool require_v16) {
    TORCH_CHECK(
        capacity > 0 && (!require_v16 || capacity % 16 == 0),
        "RPU_PLANNER_REJECT:CAPABILITY: ", role,
        " capacity must be positive",
        require_v16 ? " and 16-aligned" : "", ", got ", capacity);
    TORCH_CHECK(
        capacity >= stage.plan.chunk_size &&
            (stage.chunks.size() == 1 || capacity == stage.plan.chunk_size),
        "RPU_PLANNER_REJECT:CAPABILITY: ", role,
        " capacity does not describe its resolved schedule");
}

void append_stage_descriptor(
    std::vector<int64_t>& descriptor,
    const ResolvedFmbStageChunks& stage) {
    for (const ChunkInfo& chunk : stage.chunks) {
        descriptor.push_back(chunk.idx);
        descriptor.push_back(chunk.offset);
        descriptor.push_back(chunk.len);
        descriptor.push_back(chunk.kv_seq_len);
    }
}

}  // namespace

uint64_t fmb_physical_manifest_fingerprint(
    const FmbPhysicalExecutionManifest& manifest) {
    return physical_manifest_fingerprint_impl(manifest);
}

namespace detail {

namespace {

std::tuple<int64_t, int64_t> physical_capability_rank(
    const FmbPhysicalExecutionManifest& manifest) {
    std::set<int64_t> ddr_attention_sites;
    for (const FmbRouteManifestEntry& route : manifest.routes) {
        if (route.family == FmbRouteFamily::ATTENTION &&
            route.selector == static_cast<int64_t>(
                                  AttentionExecutionPolicy::DDR_KV)) {
            ddr_attention_sites.insert(route.site_id);
        }
    }
    const int64_t rope_residency_rank =
        fmb_rope_table_residency(manifest) ==
                FmbRopeTableResidency::DDR
        ? 1
        : 0;
    return {static_cast<int64_t>(ddr_attention_sites.size()),
            rope_residency_rank};
}

}  // namespace

bool fmb_prefill_stage_candidate_less(
    const FmbPrefillStageCandidate& lhs,
    const FmbPrefillStageCandidate& rhs) {
    return std::make_tuple(
               lhs.input_chunk_size, lhs.qkv_chunk_size,
               lhs.compute_chunk_size,
               physical_capability_rank(lhs.physical_manifest),
               fmb_physical_manifest_fingerprint(lhs.physical_manifest)) <
        std::make_tuple(
               rhs.input_chunk_size, rhs.qkv_chunk_size,
               rhs.compute_chunk_size,
               physical_capability_rank(rhs.physical_manifest),
               fmb_physical_manifest_fingerprint(rhs.physical_manifest));
}

bool fmb_prefill_stage_candidate_same_identity(
    const FmbPrefillStageCandidate& lhs,
    const FmbPrefillStageCandidate& rhs) {
    return lhs.input_chunk_size == rhs.input_chunk_size &&
        lhs.qkv_chunk_size == rhs.qkv_chunk_size &&
        lhs.compute_chunk_size == rhs.compute_chunk_size &&
        fmb_physical_manifest_fingerprint(lhs.physical_manifest) ==
            fmb_physical_manifest_fingerprint(rhs.physical_manifest);
}

}  // namespace detail

int64_t fmb_ring_all_reduce_route_selector(int64_t rows, int64_t cols,
                                          int num_cores) {
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8,
                "RPU_PLANNER_REJECT:CAPABILITY: ring cores must be in [1,8]");
    TORCH_CHECK(
        rows > 0 && cols > 0 &&
            static_cast<uint64_t>(rows) <=
                std::numeric_limits<uint64_t>::max() /
                    static_cast<uint64_t>(cols),
        "RPU_PLANNER_REJECT:CAPABILITY: ring route requires positive "
        "non-overflowing geometry");
    const uint64_t kCoreCount = static_cast<uint64_t>(num_cores);
    constexpr uint64_t kNopaceLimitBytes = 11U * 1024U;
    const uint64_t total = static_cast<uint64_t>(rows) *
        static_cast<uint64_t>(cols);
    const uint64_t unaligned =
        total / kCoreCount + (total % kCoreCount != 0);
    TORCH_CHECK(
        unaligned <= std::numeric_limits<uint64_t>::max() - 15U,
        "RPU_PLANNER_REJECT:CAPABILITY: ring route alignment overflows");
    const uint64_t average =
        ((unaligned + 15U) / 16U) * 16U;
    return static_cast<int64_t>(
        average < kNopaceLimitBytes / sizeof(c10::Half)
            ? FmbSharedAllReduceRouteSelector::RING_NOPACE
            : FmbSharedAllReduceRouteSelector::RING_PACED);
}

int64_t fmb_ring_all_reduce_route_selector(int64_t rows, int64_t cols, bool pi05_xor3, int num_cores) {
    const bool exact = ((rows == 512 || rows == 768) && cols == 1152) ||
        ((rows == 272 || rows == 288 || rows == 304 || rows == 320 || rows == 400 || rows == 416 || rows == 432 || rows == 448) && cols == 2048) || (rows == 50 && cols == 1024);
    if (!pi05_xor3 || !exact || num_cores != 8) return fmb_ring_all_reduce_route_selector(rows, cols, num_cores);
    return static_cast<int64_t>(FmbSharedAllReduceRouteSelector::RING_PI05_XOR3);
}

void append_fmb_shared_runtime_routes(
    FmbPhysicalExecutionManifest& manifest, const FmbThreeStageChunkPlan& plan,
    int64_t hidden_size, uint32_t route_mask, int64_t compute_row_multiplier,
    int num_cores, int mlp_num_cores) {
    append_fmb_shared_runtime_routes(manifest, plan, hidden_size, route_mask,
                                    compute_row_multiplier, /*pi05_xor3=*/false, num_cores, mlp_num_cores);
}

void append_fmb_shared_runtime_routes(
    FmbPhysicalExecutionManifest& manifest,
    const FmbThreeStageChunkPlan& plan,
    int64_t hidden_size,
    uint32_t route_mask,
    int64_t compute_row_multiplier,
    bool pi05_xor3,
    int num_cores,
    int mlp_num_cores) {
    constexpr uint32_t kKnownRoutes =
        FMB_SHARED_LAYER_INPUT_DMA |
        FMB_SHARED_LAYER_INPUT_ROW_RUN_DMA |
        FMB_SHARED_PIPELINE_INGRESS_DMA |
        FMB_SHARED_MLP_AUTO_TILE |
        FMB_SHARED_MLP_ACC32_OUT_BF16 |
        FMB_SHARED_MLP_BF16_FP16_REDUCE |
        FMB_SHARED_MLP_FP16_BF16_REDUCE |
        FMB_SHARED_MLP_RING_REDUCE |
        FMB_SHARED_MLP_PREPARE_INPUT |
        FMB_SHARED_MLP_GEMV;
    TORCH_CHECK(
        manifest.state == FmbPhysicalManifestState::COMPLETE &&
            (route_mask & ~kKnownRoutes) == 0,
        "RPU_PLANNER_REJECT:CAPABILITY: shared runtime routes require a "
        "COMPLETE manifest and a known route mask");
    TORCH_CHECK(num_cores >= 1 && num_cores <= 8 &&
                    mlp_num_cores >= 1 && mlp_num_cores <= num_cores,
                "RPU_PLANNER_REJECT:CAPABILITY: shared route topology exceeds the core budget");

    const auto append = [&](int64_t site_id, FmbRouteFamily family,
                            int64_t selector, int64_t invocation = 0,
                            std::vector<int64_t> arguments = {}) {
        if (num_cores != 8 && arguments.empty()) {
            arguments = {2, num_cores, mlp_num_cores};
        }
        manifest.routes.push_back(
            {site_id, family, selector, /*flags=*/0,
             std::move(arguments), invocation});
    };
    if (route_mask & FMB_SHARED_MLP_PREPARE_INPUT) {
        TORCH_CHECK(mlp_num_cores >= 1 && mlp_num_cores < num_cores && num_cores <= 8,
                    "RPU_PLANNER_REJECT:CAPABILITY: MLP ring preparation requires "
                    "a proper prefix producer within the consumer budget");
        append(FMB_SHARED_MLP_PREPARE_INPUT_SITE, FmbRouteFamily::ALL_REDUCE,
               static_cast<int64_t>(FmbSharedAllReduceRouteSelector::PREPARE_RING_INPUT),
               0, {mlp_num_cores, num_cores, 1});  // clear all consumer shards
    }
    if (route_mask & FMB_SHARED_LAYER_INPUT_DMA) {
        append(
            FMB_SHARED_LAYER_INPUT_DMA_SITE, FmbRouteFamily::MUTABLE_DMA,
            static_cast<int64_t>(
                FmbSharedMutableDmaRouteSelector::DDR_BROADCAST_TO_SPM));
    }
    if (route_mask & FMB_SHARED_LAYER_INPUT_ROW_RUN_DMA) {
        append(
            FMB_SHARED_LAYER_INPUT_ROW_RUN_DMA_SITE,
            FmbRouteFamily::MUTABLE_DMA,
            static_cast<int64_t>(
                FmbSharedMutableDmaRouteSelector::DDR_BROADCAST_TO_SPM));
    }
    if (route_mask & FMB_SHARED_PIPELINE_INGRESS_DMA) {
        append(
            FMB_SHARED_PIPELINE_INGRESS_DMA_SITE,
            FmbRouteFamily::MUTABLE_DMA,
            static_cast<int64_t>(FmbSharedMutableDmaRouteSelector::
                CANONICAL_DDR_BROADCAST_TO_SPM_MUTABLE_SRC));
    }
    TORCH_CHECK(!((route_mask & FMB_SHARED_MLP_AUTO_TILE) &&
                  (route_mask & FMB_SHARED_MLP_GEMV)),
                "RPU_PLANNER_REJECT:CAPABILITY: shared MLP has conflicting Linear routes");
    TORCH_CHECK(!(route_mask & FMB_SHARED_MLP_GEMV) ||
                   (manifest.physical_length == 1 && compute_row_multiplier == 1 &&
                    num_cores == 8 && mlp_num_cores == 8 &&
                    manifest.linear_accumulation == FmbLinearAccumulationPolicy::ACC16),
               "RPU_PLANNER_REJECT:CAPABILITY: shared MLP GEMV requires TP8 ACC16 M1");
    if (route_mask & (FMB_SHARED_MLP_AUTO_TILE | FMB_SHARED_MLP_GEMV)) {
        // Keep the established source-site identity; the descriptor records
        // which physical Linear implementation that site must consume.
        append(
            FMB_SHARED_MLP_AUTO_TILE_SITE, FmbRouteFamily::LINEAR,
            static_cast<int64_t>((route_mask & FMB_SHARED_MLP_GEMV)
                ? FmbLinearRouteSelector::GEMV : FmbLinearRouteSelector::AUTO_TILE));
    }
    if (route_mask & FMB_SHARED_MLP_ACC32_OUT_BF16) {
        append(
            FMB_SHARED_MLP_ACC32_OUT_BF16_SITE, FmbRouteFamily::LINEAR,
            static_cast<int64_t>(
                FmbLinearRouteSelector::ACC32_OUT_BF16));
    }
    if (route_mask & FMB_SHARED_MLP_BF16_FP16_REDUCE) {
        append(
            FMB_SHARED_MLP_BF16_FP16_REDUCE_SITE,
            FmbRouteFamily::ALL_REDUCE,
            static_cast<int64_t>(FmbSharedAllReduceRouteSelector::
                BF16_PARTIAL_FP16_RESIDUAL));
    }
    if (route_mask & FMB_SHARED_MLP_FP16_BF16_REDUCE) {
        append(
            FMB_SHARED_MLP_FP16_BF16_REDUCE_SITE,
            FmbRouteFamily::ALL_REDUCE,
            static_cast<int64_t>(FmbSharedAllReduceRouteSelector::
                FP16_PARTIAL_BF16_RESIDUAL));
    }
    if (route_mask & FMB_SHARED_MLP_RING_REDUCE) {
        TORCH_CHECK(
            hidden_size > 0 && compute_row_multiplier > 0,
            "RPU_PLANNER_REJECT:CAPABILITY: shared MLP ring route requires "
            "a positive hidden size and row multiplier");
        for (const ChunkInfo& chunk : plan.compute.chunks) {
            TORCH_CHECK(
                chunk.len <= std::numeric_limits<int64_t>::max() /
                    compute_row_multiplier,
                "RPU_PLANNER_REJECT:CAPABILITY: shared MLP row geometry "
                "overflows int64");
            append(
                FMB_SHARED_MLP_RING_REDUCE_SITE,
                FmbRouteFamily::ALL_REDUCE,
                fmb_ring_all_reduce_route_selector(
                    chunk.len * compute_row_multiplier, hidden_size, pi05_xor3, num_cores),
                chunk.idx);
        }
    }

    std::sort(
        manifest.routes.begin(), manifest.routes.end(),
        [](const FmbRouteManifestEntry& lhs,
           const FmbRouteManifestEntry& rhs) {
            return std::make_tuple(
                       static_cast<int64_t>(lhs.family), lhs.site_id,
                       lhs.invocation) <
                std::make_tuple(
                       static_cast<int64_t>(rhs.family), rhs.site_id,
                       rhs.invocation);
        });
}

void validate_fmb_physical_manifest(
    const FmbPhysicalExecutionManifest& manifest,
    int64_t stage_physical_length,
    int64_t position_base) {
    validate_physical_manifest_impl(
        manifest, stage_physical_length, position_base);
}

void validate_fmb_physical_manifest_forward_capability(
    const FmbPhysicalExecutionManifest& manifest,
    const FmbPhysicalManifestForwardCapability& capability) {
    if (manifest.state == FmbPhysicalManifestState::UNSPECIFIED) return;
    TORCH_CHECK(
        manifest.state == FmbPhysicalManifestState::COMPLETE &&
            capability.complete_descriptor,
        "RPU_PLANNER_REJECT:CAPABILITY: COMPLETE physical manifest requires "
        "an explicit native forward capability");
    TORCH_CHECK(
        capability.graph_lifecycle != FmbGraphLifecycle::UNSPECIFIED &&
            capability.graph_lifecycle == manifest.graph_lifecycle,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: COMPLETE physical manifest Graph "
        "lifecycle does not match the current native forward");
}

AttentionExecutionPolicy fmb_attention_execution_policy(
    const FmbRouteManifestEntry& route) {
    TORCH_CHECK(
        route.family == FmbRouteFamily::ATTENTION,
        "RPU_PLANNER_REJECT:CAPABILITY: attention selector mapping requires "
        "an ATTENTION route");
    switch (route.selector) {
        case static_cast<int64_t>(AttentionExecutionPolicy::DDR_KV):
            return AttentionExecutionPolicy::DDR_KV;
        case static_cast<int64_t>(
            AttentionExecutionPolicy::SPM_KV_BY_MHA):
            return AttentionExecutionPolicy::SPM_KV_BY_MHA;
        default:
            TORCH_CHECK(
                false,
                "RPU_PLANNER_REJECT:CAPABILITY: ATTENTION route selector=",
                route.selector,
                " is not a resolved AttentionExecutionPolicy for site_id=",
                route.site_id);
    }
    return AttentionExecutionPolicy::DDR_KV;
}

FmbRopeTableResidency fmb_rope_table_residency(
    const FmbPhysicalExecutionManifest& manifest) {
    FmbRopeTableResidency residency =
        FmbRopeTableResidency::UNSPECIFIED;
    constexpr int64_t kResidencyFlags =
        FMB_ROUTE_FLAG_ROPE_TABLE_DDR |
        FMB_ROUTE_FLAG_ROPE_TABLE_SPM;
    for (const FmbRouteManifestEntry& route : manifest.routes) {
        if (route.family != FmbRouteFamily::ROPE) continue;
        const int64_t flags = route.flags & kResidencyFlags;
        if (flags == 0) continue;
        TORCH_CHECK(
            flags != kResidencyFlags,
            "RPU_PLANNER_REJECT:CAPABILITY: ROPE route has conflicting "
            "DDR/SPM table-residency flags");
        const FmbRopeTableResidency route_residency =
            flags == FMB_ROUTE_FLAG_ROPE_TABLE_SPM
            ? FmbRopeTableResidency::SPM
            : FmbRopeTableResidency::DDR;
        TORCH_CHECK(
            residency == FmbRopeTableResidency::UNSPECIFIED ||
                residency == route_residency,
            "RPU_PLANNER_REJECT:CAPABILITY: COMPLETE manifest mixes ROPE "
            "table residencies");
        residency = route_residency;
    }
    return residency;
}

void FmbPhysicalManifestConsumer::reset(
    const FmbPhysicalExecutionManifest& manifest) {
    manifest_ = manifest;
    prepared_.reset();
    fingerprint_ = manifest.state == FmbPhysicalManifestState::COMPLETE
        ? fmb_physical_manifest_fingerprint(manifest) : 0;
    consumed_.assign(manifest.routes.size(), 0);
}

void FmbPhysicalManifestConsumer::reset_prepared(
    std::shared_ptr<const FmbPreparedStageCandidate> prepared) {
    TORCH_INTERNAL_ASSERT(prepared != nullptr);
    prepared_ = std::move(prepared);
    fingerprint_ = prepared_->manifest_fingerprint();
    consumed_.assign(manifest().routes.size(), 0);
}

void FmbPhysicalManifestConsumer::retain_consumed_if_matching(
    const FmbPhysicalExecutionManifest& manifest) const {
    const uint64_t incoming_fingerprint =
        manifest.state == FmbPhysicalManifestState::COMPLETE
        ? fmb_physical_manifest_fingerprint(manifest) : 0;
    TORCH_CHECK(
        complete() && fingerprint_ != 0 &&
            incoming_fingerprint == fingerprint_,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: external physical-manifest "
        "prologue does not match the run_all_layers candidate");
}

bool FmbPhysicalManifestConsumer::complete() const {
    return manifest().state == FmbPhysicalManifestState::COMPLETE;
}

const FmbPhysicalExecutionManifest&
FmbPhysicalManifestConsumer::manifest() const {
    return prepared_ ? prepared_->candidate().physical_manifest : manifest_;
}

uint64_t FmbPhysicalManifestConsumer::fingerprint() const {
    return fingerprint_;
}

const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(
    FmbRouteFamily family, int64_t site_id, int64_t invocation) const {
    TORCH_CHECK(
        complete(),
        "RPU_PLANNER_REJECT:CAPABILITY: physical route lookup requires a "
        "COMPLETE manifest");
    const int64_t raw_family = static_cast<int64_t>(family);
    TORCH_CHECK(
        raw_family >= static_cast<int64_t>(FmbRouteFamily::ATTENTION) &&
            raw_family <= static_cast<int64_t>(
                              FmbRouteFamily::COLLECTIVE) &&
            site_id > 0 && invocation >= 0,
        "RPU_PLANNER_REJECT:CAPABILITY: physical route lookup has an invalid "
        "family/site/invocation identity");
    const auto target = std::make_tuple(raw_family, site_id, invocation);
    const auto found = std::lower_bound(
        manifest().routes.begin(), manifest().routes.end(), target,
        [](const FmbRouteManifestEntry& route,
           const std::tuple<int64_t, int64_t, int64_t>& identity) {
            return std::make_tuple(static_cast<int64_t>(route.family),
                                   route.site_id, route.invocation) < identity;
        });
    if (found != manifest().routes.end() && found->family == family &&
        found->site_id == site_id && found->invocation == invocation) {
        return *found;
    }
    const bool wrong_family = std::any_of(
        manifest().routes.begin(), manifest().routes.end(),
        [site_id, invocation](const FmbRouteManifestEntry& route) {
            return route.site_id == site_id &&
                route.invocation == invocation;
        });
    TORCH_CHECK(
        !wrong_family,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: physical route site_id=",
        site_id, ", invocation=", invocation,
        " was consumed with the wrong family");
    TORCH_CHECK(
        false,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: physical manifest has no route "
        "for family=", raw_family, ", site_id=", site_id,
        ", invocation=", invocation);
}

const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(
    FmbRouteFamily family,
    int64_t site_id,
    int64_t resolved_selector,
    int64_t resolved_flags,
    at::IntArrayRef resolved_arguments,
    int64_t invocation) const {
    const FmbRouteManifestEntry& route =
        find_route(family, site_id, invocation);
    const bool arguments_match =
        route.arguments.size() == resolved_arguments.size() &&
        std::equal(route.arguments.begin(), route.arguments.end(),
                   resolved_arguments.begin());
    TORCH_CHECK(
        route.selector == resolved_selector && route.flags == resolved_flags &&
            arguments_match,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: physical route policy was not "
        "the selector/flags/arguments actually dispatched for family=",
        static_cast<int64_t>(family), ", site_id=", site_id,
        ", invocation=", invocation,
        ", expected_selector=", route.selector,
        ", actual_selector=", resolved_selector,
        ", expected_flags=", route.flags,
        ", actual_flags=", resolved_flags,
        ", expected_argument_count=", route.arguments.size(),
        ", actual_argument_count=", resolved_arguments.size(),
        ", arguments_match=", arguments_match);
    const size_t index = static_cast<size_t>(
        &route - manifest().routes.data());
    TORCH_INTERNAL_ASSERT(index < consumed_.size());
    consumed_[index] = 1;
    return route;
}

AttentionExecutionPolicy
FmbPhysicalManifestConsumer::attention_policy_for_site(
    int64_t site_id, int64_t invocation) const {
    return fmb_attention_execution_policy(
        find_route(FmbRouteFamily::ATTENTION, site_id, invocation));
}

std::optional<AttentionExecutionPolicy>
FmbPhysicalManifestConsumer::attention_layout_policy() const {
    TORCH_CHECK(
        complete(),
        "RPU_PLANNER_REJECT:CAPABILITY: attention layout policy requires a "
        "COMPLETE manifest");
    if (prepared_) return prepared_->attention_policy();
    std::optional<AttentionExecutionPolicy> policy;
    for (const FmbRouteManifestEntry& route : manifest().routes) {
        if (route.family != FmbRouteFamily::ATTENTION) continue;
        const AttentionExecutionPolicy site_policy =
            fmb_attention_execution_policy(route);
        if (!policy.has_value() ||
            site_policy == AttentionExecutionPolicy::SPM_KV_BY_MHA) {
            // Layout is the union of per-site requirements. Mixed manifests
            // remain per-site authority: a fixed DDR site does not erase the
            // raw-SPM buffers needed by a different SPM site.
            policy = site_policy;
        }
    }
    return policy;
}

const FmbRouteManifestEntry&
FmbPhysicalManifestConsumer::consume_attention_route(
    int64_t site_id,
    AttentionExecutionPolicy dispatched_policy,
    int64_t invocation) const {
    const AttentionExecutionPolicy planned_policy =
        attention_policy_for_site(site_id, invocation);
    TORCH_CHECK(
        dispatched_policy != AttentionExecutionPolicy::AUTO &&
            dispatched_policy == planned_policy,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: ATTENTION route was not the "
        "resolved policy actually dispatched for site_id=",
        site_id, ", invocation=", invocation);
    return consume_route(
        FmbRouteFamily::ATTENTION, site_id,
        static_cast<int64_t>(dispatched_policy), /*resolved_flags=*/0,
        /*resolved_arguments=*/{}, invocation);
}

void FmbPhysicalManifestConsumer::inherit_matching_graph_replay_receipt(
    uint64_t replayed_manifest_fingerprint) const {
    TORCH_CHECK(
        complete() && fingerprint_ != 0 &&
            replayed_manifest_fingerprint == fingerprint_,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: Graph replay receipt does not "
        "match the COMPLETE physical manifest identity");
    std::fill(consumed_.begin(), consumed_.end(), 1);
}

std::vector<uint8_t>
FmbPhysicalManifestConsumer::consumed_route_receipt() const {
    return consumed_;
}

void FmbPhysicalManifestConsumer::inherit_consumed_route_subset(
    uint64_t manifest_fingerprint,
    const std::vector<uint8_t>& receipt) const {
    TORCH_CHECK(
        complete() && fingerprint_ != 0 &&
            manifest_fingerprint == fingerprint_ &&
            receipt.size() == consumed_.size(),
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: composite preload receipt "
        "does not match the COMPLETE physical manifest identity");
    // Validate the entire subset before mutating any receipt bit.
    TORCH_CHECK(std::all_of(receipt.begin(), receipt.end(),
                            [](uint8_t bit) { return bit <= 1; }),
                "RPU_PLANNER_REJECT:CAPABILITY: invalid preload receipt bit");
    for (size_t index = 0; index < receipt.size(); ++index) {
        consumed_[index] |= receipt[index];
    }
}

void FmbPhysicalManifestConsumer::require_all_consumed() const {
    if (!complete()) return;
    for (size_t index = 0; index < consumed_.size(); ++index) {
        const FmbRouteManifestEntry& route = manifest().routes[index];
        TORCH_CHECK(
            consumed_[index] != 0,
            "RPU_PLANNER_REJECT:CAPABILITY: COMPLETE physical manifest route "
            "was not consumed: family=",
            static_cast<int64_t>(route.family), ", site_id=", route.site_id,
            ", invocation=", route.invocation);
    }
}

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

void validate_fmb_execution_spans(
    const std::vector<FmbExecutionSpan>& spans,
    int64_t execution_len,
    const char* contract) {
    TORCH_CHECK(!spans.empty(), contract, ": spans must not be empty");
    int64_t span_total = 0;
    for (const FmbExecutionSpan& span : spans) {
        TORCH_CHECK(
            span.offset == span_total && span.len > 0,
            contract, ": spans require contiguous offsets and positive lengths");
        TORCH_CHECK(spans.size() == 1 || span.group_id >= 0,
                    contract, ": multi-span plans require explicit group IDs");
        TORCH_CHECK(
            span_total <= std::numeric_limits<int64_t>::max() - span.len,
            contract, ": span length overflows int64");
        span_total += span.len;
    }
    TORCH_CHECK(span_total == execution_len,
                contract, ": spans must cover the physical length");
}

bool fmb_chunks_respect_boundary_policy(
    const std::vector<ChunkInfo>& chunks,
    const std::vector<FmbExecutionSpan>& spans,
    FmbSpanBoundaryPolicy policy) {
    if (policy == FmbSpanBoundaryPolicy::ALLOW_CROSS) return true;
    if (policy != FmbSpanBoundaryPolicy::KEEP_LOCAL &&
        policy != FmbSpanBoundaryPolicy::GROUP_LOCAL) {
        return false;
    }

    size_t span_index = 0;
    for (const ChunkInfo& chunk : chunks) {
        if (chunk.len <= 0 ||
            chunk.offset > std::numeric_limits<int64_t>::max() - chunk.len) {
            return false;
        }
        const int64_t chunk_end = chunk.offset + chunk.len;
        while (span_index < spans.size() &&
               spans[span_index].offset + spans[span_index].len <=
                   chunk.offset) {
            ++span_index;
        }
        if (span_index == spans.size() ||
            spans[span_index].offset > chunk.offset) {
            return false;
        }

        const int64_t group_id = spans[span_index].group_id;
        if (policy == FmbSpanBoundaryPolicy::GROUP_LOCAL && group_id < 0) {
            return false;
        }
        int64_t span_end =
            spans[span_index].offset + spans[span_index].len;
        size_t last_span = span_index;
        while (chunk_end > span_end) {
            if (policy == FmbSpanBoundaryPolicy::KEEP_LOCAL ||
                ++last_span == spans.size() ||
                spans[last_span].offset != span_end ||
                spans[last_span].group_id != group_id) {
                return false;
            }
            span_end = spans[last_span].offset + spans[last_span].len;
        }
        span_index = last_span;
    }
    return true;
}

void validate_fmb_planning_shape(
    int64_t execution_len,
    int64_t position,
    const char* contract) {
    TORCH_CHECK(
        execution_len > 0 &&
            execution_len <= std::numeric_limits<int64_t>::max() - 15,
        "RPU_PLANNER_REJECT:CAPABILITY: ", contract,
        ": invalid execution length ", execution_len);
    TORCH_CHECK(
        position >= 0 &&
            position <= std::numeric_limits<int64_t>::max() - execution_len,
        "RPU_PLANNER_REJECT:CAPABILITY: ", contract,
        ": invalid position/execution length (position=", position,
        ", execution_len=", execution_len, ")");
}

void validate_fmb_exact_chunk_size(
    int64_t requested_chunk_size,
    int64_t execution_len,
    const char* contract) {
    validate_fmb_planning_shape(
        execution_len, /*position=*/0, contract);
    TORCH_CHECK(
        requested_chunk_size >= 16 && requested_chunk_size % 16 == 0,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: ", contract,
        ": exact chunk size must be a positive multiple of 16, got ",
        requested_chunk_size);
    const int64_t rounded_execution_len =
        ((execution_len + 15) / 16) * 16;
    TORCH_CHECK(
        requested_chunk_size <= rounded_execution_len,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: ", contract,
        ": exact chunk_size=", requested_chunk_size,
        " exceeds ceil16(execution_len)=", rounded_execution_len);
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
                "RPU_PLANNER_REJECT:CAPABILITY: FMB attention policy: "
                "SPM_KV_BY_MHA was requested but the "
                "model/kernel profile is not eligible");
            TORCH_CHECK(
                exact_spm_layout_fits,
                "RPU_PLANNER_REJECT:CAPABILITY: FMB attention policy: "
                "SPM_KV_BY_MHA was requested but the "
                "exact joint SPM layout does not fit");
            return AttentionExecutionPolicy::SPM_KV_BY_MHA;
        default:
            TORCH_CHECK(
                false,
                "RPU_PLANNER_REJECT:CAPABILITY: FMB attention policy: "
                "unsupported requested value");
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

    detail::validate_fmb_execution_spans(
        plan.spans, input.total, "FMB three-stage plan");
    TORCH_CHECK(detail::fmb_chunks_respect_boundary_policy(
                    plan.input.chunks, plan.spans,
                    plan.boundary_policies.input),
                "FMB three-stage plan: input chunks violate boundary policy");
    TORCH_CHECK(detail::fmb_chunks_respect_boundary_policy(
                    plan.qkv.chunks, plan.spans,
                    plan.boundary_policies.qkv),
                "FMB three-stage plan: qkv chunks violate boundary policy");
    TORCH_CHECK(detail::fmb_chunks_respect_boundary_policy(
                    plan.compute.chunks, plan.spans,
                    plan.boundary_policies.compute),
                "FMB three-stage plan: compute chunks violate boundary policy");
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
    hash = mix_u64(hash, static_cast<uint64_t>(
                             plan.boundary_policies.input));
    hash = mix_u64(hash, static_cast<uint64_t>(
                             plan.boundary_policies.qkv));
    hash = mix_u64(hash, static_cast<uint64_t>(
                             plan.boundary_policies.compute));
    hash = mix_u64(hash, static_cast<uint64_t>(plan.spans.size()));
    for (const FmbExecutionSpan& span : plan.spans) {
        hash = mix_u64(hash, static_cast<uint64_t>(span.offset));
        hash = mix_u64(hash, static_cast<uint64_t>(span.len));
        hash = mix_u64(hash, static_cast<uint64_t>(span.group_id));
    }
    return hash == 0 ? 1 : hash;
}

std::vector<int64_t> encode_fmb_prefill_stage_candidate(
    const FmbPrefillStageCandidate& candidate) {
    validate_fmb_three_stage_chunk_plan(candidate.stage_plan);
    validate_candidate_capacity(
        candidate.input_chunk_size, candidate.stage_plan.input,
        "input", /*require_v16=*/false);
    validate_candidate_capacity(
        candidate.qkv_chunk_size, candidate.stage_plan.qkv,
        "qkv", /*require_v16=*/true);
    validate_candidate_capacity(
        candidate.compute_chunk_size, candidate.stage_plan.compute,
        "compute", /*require_v16=*/true);
    const ValidatedStage stage =
        validate_stage(candidate.stage_plan.input, "input");
    validate_fmb_physical_manifest(
        candidate.physical_manifest, stage.total, stage.position_base);

    const uint64_t stage_fingerprint =
        fmb_three_stage_chunk_plan_fingerprint(candidate.stage_plan);
    const uint64_t manifest_fingerprint =
        fmb_physical_manifest_fingerprint(candidate.physical_manifest);
    size_t total_words = kPrefillStageCandidateHeaderWords +
        4 * (candidate.stage_plan.input.chunks.size() +
             candidate.stage_plan.qkv.chunks.size() +
             candidate.stage_plan.compute.chunks.size()) +
        3 * candidate.stage_plan.spans.size() +
        kPhysicalManifestHeaderWords;
    for (const FmbRouteManifestEntry& route :
         candidate.physical_manifest.routes) {
        TORCH_CHECK(
            route.arguments.size() <=
                kMaxPrefillStageCandidateDescriptorWords,
            "RPU_PLANNER_REJECT:CAPABILITY: physical route argument count "
            "exceeds descriptor limit");
        TORCH_CHECK(
            total_words <= kMaxPrefillStageCandidateDescriptorWords - 6 &&
                route.arguments.size() <=
                    kMaxPrefillStageCandidateDescriptorWords -
                        total_words - 6,
            "RPU_PLANNER_REJECT:CAPABILITY: prefill stage descriptor exceeds ",
            kMaxPrefillStageCandidateDescriptorWords, " words");
        total_words += 6 + route.arguments.size();
    }
    TORCH_CHECK(
        total_words <= kMaxPrefillStageCandidateDescriptorWords,
        "RPU_PLANNER_REJECT:CAPABILITY: prefill stage descriptor exceeds ",
        kMaxPrefillStageCandidateDescriptorWords, " words");

    std::vector<int64_t> descriptor;
    descriptor.reserve(total_words);
    descriptor.insert(descriptor.end(), {
        kPrefillStageCandidateDescriptorVersion,
        static_cast<int64_t>(total_words),
        candidate.input_chunk_size,
        candidate.qkv_chunk_size,
        candidate.compute_chunk_size,
        static_cast<int64_t>(stage_fingerprint >> 32),
        static_cast<int64_t>(stage_fingerprint & 0xffffffffULL),
        static_cast<int64_t>(candidate.stage_plan.chunk_mode),
        static_cast<int64_t>(candidate.stage_plan.boundary_policies.input),
        static_cast<int64_t>(candidate.stage_plan.boundary_policies.qkv),
        static_cast<int64_t>(candidate.stage_plan.boundary_policies.compute),
        static_cast<int64_t>(candidate.stage_plan.input.chunks.size()),
        static_cast<int64_t>(candidate.stage_plan.qkv.chunks.size()),
        static_cast<int64_t>(candidate.stage_plan.compute.chunks.size()),
        static_cast<int64_t>(candidate.stage_plan.spans.size()),
    });
    append_stage_descriptor(descriptor, candidate.stage_plan.input);
    append_stage_descriptor(descriptor, candidate.stage_plan.qkv);
    append_stage_descriptor(descriptor, candidate.stage_plan.compute);
    for (const FmbExecutionSpan& span : candidate.stage_plan.spans) {
        descriptor.push_back(span.offset);
        descriptor.push_back(span.len);
        descriptor.push_back(span.group_id);
    }
    descriptor.insert(descriptor.end(), {
        static_cast<int64_t>(candidate.physical_manifest.state),
        candidate.physical_manifest.logical_length,
        candidate.physical_manifest.physical_length,
        candidate.physical_manifest.execution_padding_rows,
        candidate.physical_manifest.kv_logical_length,
        candidate.physical_manifest.kv_insert_physical_rows,
        static_cast<int64_t>(candidate.physical_manifest.graph_lifecycle),
        static_cast<int64_t>(candidate.physical_manifest.linear_accumulation),
        static_cast<int64_t>(candidate.physical_manifest.routes.size()),
        static_cast<int64_t>(manifest_fingerprint >> 32),
        static_cast<int64_t>(manifest_fingerprint & 0xffffffffULL),
    });
    for (const FmbRouteManifestEntry& route :
         candidate.physical_manifest.routes) {
        descriptor.insert(descriptor.end(), {
            static_cast<int64_t>(route.family),
            route.site_id,
            route.selector,
            route.flags,
            static_cast<int64_t>(route.arguments.size()),
            route.invocation,
        });
        descriptor.insert(
            descriptor.end(), route.arguments.begin(), route.arguments.end());
    }
    TORCH_INTERNAL_ASSERT(descriptor.size() == total_words);
    return descriptor;
}

FmbPrefillStageCandidate decode_fmb_prefill_stage_candidate(
    at::IntArrayRef descriptor) {
    const char* reject = "RPU_PLANNER_REJECT:CAPABILITY: ";
    TORCH_CHECK(
        descriptor.size() >= kPrefillStageCandidateHeaderWords &&
            descriptor.size() <= kMaxPrefillStageCandidateDescriptorWords,
        reject, "invalid prefill stage descriptor length ", descriptor.size());
    const int64_t descriptor_version = descriptor[0];
    TORCH_CHECK(
        descriptor_version == kPrefillStageCandidateDescriptorVersionV1 ||
            descriptor_version == kPrefillStageCandidateDescriptorVersionV2 ||
            descriptor_version == kPrefillStageCandidateDescriptorVersion,
        reject, "unsupported prefill stage descriptor version ",
        descriptor_version);
    TORCH_CHECK(
        descriptor[1] == static_cast<int64_t>(descriptor.size()),
        reject, "prefill stage descriptor length field is stale");

    auto decode_count = [&](size_t index, const char* role) {
        const int64_t value = descriptor[index];
        TORCH_CHECK(value > 0, reject, role, " count must be positive");
        return static_cast<size_t>(value);
    };
    const size_t input_count = decode_count(11, "input chunk");
    const size_t qkv_count = decode_count(12, "qkv chunk");
    const size_t compute_count = decode_count(13, "compute chunk");
    const size_t span_count = decode_count(14, "span");
    TORCH_CHECK(
        input_count <= (descriptor.size() - kPrefillStageCandidateHeaderWords) / 4 &&
            qkv_count <= (descriptor.size() - kPrefillStageCandidateHeaderWords) / 4 &&
            compute_count <= (descriptor.size() - kPrefillStageCandidateHeaderWords) / 4 &&
            span_count <= (descriptor.size() - kPrefillStageCandidateHeaderWords) / 3,
        reject, "prefill stage descriptor count exceeds its payload");
    const size_t stage_words = kPrefillStageCandidateHeaderWords +
        4 * (input_count + qkv_count + compute_count) + 3 * span_count;
    if (descriptor_version == kPrefillStageCandidateDescriptorVersionV1) {
        TORCH_CHECK(stage_words == descriptor.size(), reject,
                    "prefill stage descriptor payload size is inconsistent");
    } else {
        TORCH_CHECK(
            stage_words <= descriptor.size() &&
                descriptor.size() - stage_words >=
                    kPhysicalManifestHeaderWords,
            reject, "prefill stage descriptor physical payload is truncated");
    }

    TORCH_CHECK(
        descriptor[7] == static_cast<int64_t>(ChunkMode::SEQUENTIAL) ||
            descriptor[7] == static_cast<int64_t>(ChunkMode::KV_FIRST),
        reject, "prefill stage descriptor has an unknown chunk mode");
    auto decode_policy = [&](size_t index) {
        const int64_t value = descriptor[index];
        TORCH_CHECK(
            value >= static_cast<int64_t>(FmbSpanBoundaryPolicy::ALLOW_CROSS) &&
                value <= static_cast<int64_t>(FmbSpanBoundaryPolicy::GROUP_LOCAL),
            reject, "prefill stage descriptor has an unknown boundary policy");
        return static_cast<FmbSpanBoundaryPolicy>(value);
    };

    size_t cursor = kPrefillStageCandidateHeaderWords;
    auto decode_stage = [&](size_t count) {
        std::vector<ChunkInfo> chunks;
        chunks.reserve(count);
        for (size_t ordinal = 0; ordinal < count; ++ordinal) {
            const int64_t raw_index = descriptor[cursor++];
            TORCH_CHECK(
                raw_index >= std::numeric_limits<int>::min() &&
                    raw_index <= std::numeric_limits<int>::max(),
                reject, "prefill stage chunk index exceeds int range");
            chunks.push_back({
                static_cast<int>(raw_index), descriptor[cursor++],
                descriptor[cursor++], descriptor[cursor++]});
        }
        return make_stage(std::move(chunks));
    };

    FmbThreeStageChunkPlan plan;
    plan.input = decode_stage(input_count);
    plan.qkv = decode_stage(qkv_count);
    plan.compute = decode_stage(compute_count);
    plan.spans.reserve(span_count);
    for (size_t ordinal = 0; ordinal < span_count; ++ordinal) {
        plan.spans.push_back({descriptor[cursor++], descriptor[cursor++],
                              descriptor[cursor++]});
    }
    plan.boundary_policies = {
        decode_policy(8), decode_policy(9), decode_policy(10)};
    plan.chunk_mode = static_cast<ChunkMode>(descriptor[7]);
    try {
        validate_fmb_three_stage_chunk_plan(plan);
    } catch (const c10::Error& error) {
        TORCH_CHECK(
            false, reject, "invalid prefill stage schedule: ",
            error.what_without_backtrace());
    }

    TORCH_CHECK(
        descriptor[5] >= 0 && descriptor[5] <= 0xffffffffLL &&
            descriptor[6] >= 0 && descriptor[6] <= 0xffffffffLL,
        reject, "prefill stage descriptor fingerprint words are invalid");
    const uint64_t encoded_fingerprint =
        (static_cast<uint64_t>(descriptor[5]) << 32) |
        static_cast<uint64_t>(descriptor[6]);
    TORCH_CHECK(
        encoded_fingerprint != 0 &&
            encoded_fingerprint == fmb_three_stage_chunk_plan_fingerprint(plan),
        reject, "prefill stage descriptor fingerprint is stale");

    FmbPrefillStageCandidate candidate{
        descriptor[2], descriptor[3], descriptor[4], std::move(plan)};
    if (descriptor_version != kPrefillStageCandidateDescriptorVersionV1) {
        TORCH_INTERNAL_ASSERT(cursor == stage_words);
        const int64_t raw_state = descriptor[cursor++];
        const int64_t logical_length = descriptor[cursor++];
        const int64_t physical_length = descriptor[cursor++];
        const int64_t execution_padding_rows = descriptor[cursor++];
        const int64_t kv_logical_length = descriptor[cursor++];
        const int64_t kv_insert_physical_rows = descriptor[cursor++];
        const int64_t raw_lifecycle = descriptor[cursor++];
        const int64_t raw_acc = descriptor[cursor++];
        const int64_t raw_route_count = descriptor[cursor++];
        const int64_t manifest_fingerprint_hi = descriptor[cursor++];
        const int64_t manifest_fingerprint_lo = descriptor[cursor++];
        TORCH_CHECK(
            raw_route_count >= 0 &&
                static_cast<uint64_t>(raw_route_count) <= descriptor.size(),
            reject, "physical manifest route count exceeds its payload");
        TORCH_CHECK(
            manifest_fingerprint_hi >= 0 &&
                manifest_fingerprint_hi <= 0xffffffffLL &&
                manifest_fingerprint_lo >= 0 &&
                manifest_fingerprint_lo <= 0xffffffffLL,
            reject, "physical manifest fingerprint words are invalid");

        FmbPhysicalExecutionManifest manifest;
        manifest.state = static_cast<FmbPhysicalManifestState>(raw_state);
        manifest.logical_length = logical_length;
        manifest.physical_length = physical_length;
        manifest.execution_padding_rows = execution_padding_rows;
        manifest.kv_logical_length = kv_logical_length;
        manifest.kv_insert_physical_rows = kv_insert_physical_rows;
        manifest.graph_lifecycle =
            static_cast<FmbGraphLifecycle>(raw_lifecycle);
        manifest.linear_accumulation =
            static_cast<FmbLinearAccumulationPolicy>(raw_acc);
        manifest.routes.reserve(static_cast<size_t>(raw_route_count));
        for (int64_t ordinal = 0; ordinal < raw_route_count; ++ordinal) {
            const size_t route_header_words =
                descriptor_version == kPrefillStageCandidateDescriptorVersion
                    ? 6 : 5;
            TORCH_CHECK(
                descriptor.size() - cursor >= route_header_words,
                reject, "physical manifest route is truncated");
            const int64_t raw_family = descriptor[cursor++];
            FmbRouteManifestEntry route;
            route.family = static_cast<FmbRouteFamily>(raw_family);
            route.site_id = descriptor[cursor++];
            route.selector = descriptor[cursor++];
            route.flags = descriptor[cursor++];
            const int64_t raw_argument_count = descriptor[cursor++];
            route.invocation =
                descriptor_version == kPrefillStageCandidateDescriptorVersion
                    ? descriptor[cursor++] : 0;
            TORCH_CHECK(
                raw_argument_count >= 0 &&
                    static_cast<uint64_t>(raw_argument_count) <=
                        descriptor.size() - cursor,
                reject, "physical manifest route argument count exceeds its "
                        "payload");
            route.arguments.assign(
                descriptor.begin() + cursor,
                descriptor.begin() + cursor + raw_argument_count);
            cursor += static_cast<size_t>(raw_argument_count);
            manifest.routes.push_back(std::move(route));
        }
        TORCH_CHECK(
            cursor == descriptor.size(), reject,
            "prefill stage descriptor physical payload has trailing words");
        const ValidatedStage stage =
            validate_stage(candidate.stage_plan.input, "input");
        validate_fmb_physical_manifest(
            manifest, stage.total, stage.position_base);
        const uint64_t encoded_manifest_fingerprint =
            (static_cast<uint64_t>(manifest_fingerprint_hi) << 32) |
            static_cast<uint64_t>(manifest_fingerprint_lo);
        TORCH_CHECK(
            encoded_manifest_fingerprint != 0 &&
                encoded_manifest_fingerprint ==
                    fmb_physical_manifest_fingerprint(manifest),
            reject, "physical manifest fingerprint is stale");
        candidate.physical_manifest = std::move(manifest);
    }
    validate_candidate_capacity(
        candidate.input_chunk_size, candidate.stage_plan.input,
        "input", /*require_v16=*/false);
    validate_candidate_capacity(
        candidate.qkv_chunk_size, candidate.stage_plan.qkv,
        "qkv", /*require_v16=*/true);
    validate_candidate_capacity(
        candidate.compute_chunk_size, candidate.stage_plan.compute,
        "compute", /*require_v16=*/true);
    return candidate;
}

FmbPreparedStageCandidate::FmbPreparedStageCandidate(
    FmbPrefillStageCandidate candidate) : candidate_(std::move(candidate)) {
    const auto stage = validate_stage(candidate_.stage_plan.input, "prepared input");
    physical_length_ = stage.total;
    position_ = stage.position_base;
    stage_fingerprint_ = fmb_three_stage_chunk_plan_fingerprint(candidate_.stage_plan);
    topology_ = fmb_chunk_topology(candidate_.stage_plan);
    const auto& manifest = candidate_.physical_manifest;
    if (manifest.state == FmbPhysicalManifestState::COMPLETE) {
        manifest_fingerprint_ = fmb_physical_manifest_fingerprint(manifest);
        rope_residency_ = fmb_rope_table_residency(manifest);
        attention_policy_ = FmbPhysicalManifestConsumer(manifest).attention_layout_policy();
    }
}

void FmbPreparedStageCandidate::validate_request(
    int64_t physical_length, int64_t position) const {
    // Internal route structure was validated from immutable wire bytes. These
    // request-local quantities are never inherited from a previous forward.
    TORCH_CHECK(
        physical_length == physical_length_ && position == position_,
        "RPU_PLANNER_REJECT:EXACT_MISMATCH: prepared descriptor does not "
        "match the current physical length/position");
}

std::shared_ptr<const FmbPreparedStageCandidate>
FmbPreparedStageCandidateCache::prepare(at::IntArrayRef descriptor) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->descriptor.size() == descriptor.size() &&
            std::equal(it->descriptor.begin(), it->descriptor.end(), descriptor.begin())) {
            auto result = it->prepared;
            entries_.splice(entries_.begin(), entries_, it);
            return result;
        }
    }
    // Do not publish a partial/invalid parse. A changed payload with an old
    // fingerprint must take the decoder's original fail-closed path.
    auto prepared = std::shared_ptr<const FmbPreparedStageCandidate>(
        new FmbPreparedStageCandidate(decode_fmb_prefill_stage_candidate(descriptor)));
    entries_.push_front({std::vector<int64_t>(descriptor.begin(), descriptor.end()), prepared});
    if (entries_.size() > kCapacity) entries_.pop_back();
    return prepared;
}

void rebase_fmb_retained_decode_candidate(
    FmbPrefillStageCandidate& candidate, int64_t position) {
    TORCH_CHECK(
        position >= 0 &&
            candidate.physical_manifest.state ==
                FmbPhysicalManifestState::COMPLETE &&
            candidate.physical_manifest.graph_lifecycle ==
                FmbGraphLifecycle::RETAINED_CACHE &&
            candidate.physical_manifest.logical_length == 1 &&
            candidate.physical_manifest.physical_length == 1 &&
            candidate.physical_manifest.execution_padding_rows == 0,
        "RPU_PLANNER_REJECT:CAPABILITY: retained decode rebase requires a "
        "COMPLETE single-token RETAINED_CACHE descriptor");
    for (ResolvedFmbStageChunks* stage : {
             &candidate.stage_plan.input,
             &candidate.stage_plan.qkv,
             &candidate.stage_plan.compute}) {
        const ValidatedStage validated =
            validate_stage(*stage, "retained decode");
        TORCH_CHECK(
            validated.total == 1 && validated.position_base == 0,
            "RPU_PLANNER_REJECT:EXACT_MISMATCH: retained decode descriptor "
            "must carry a canonical position-zero single-token schedule");
        for (ChunkInfo& chunk : stage->chunks) {
            chunk.kv_seq_len = position + chunk.offset + chunk.len;
        }
    }
    validate_fmb_three_stage_chunk_plan(candidate.stage_plan);
    validate_fmb_physical_manifest(
        candidate.physical_manifest, /*stage_physical_length=*/1, position);
}

std::vector<int64_t> encode_fmb_prefill_stage_domain(
    const std::vector<FmbPrefillStageCandidate>& candidates) {
    std::vector<int64_t> descriptor{
        kPrefillStageDomainDescriptorVersion,
        static_cast<int64_t>(candidates.size())};
    for (const FmbPrefillStageCandidate& candidate : candidates) {
        std::vector<int64_t> encoded =
            encode_fmb_prefill_stage_candidate(candidate);
        descriptor.push_back(static_cast<int64_t>(encoded.size()));
        descriptor.insert(descriptor.end(), encoded.begin(), encoded.end());
    }
    return descriptor;
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
    TORCH_CHECK(
        spans.size() == 1,
        "FMB three-stage plan: multi-span callers must declare boundary "
        "policies explicitly");
    return compose_fmb_three_stage_chunk_plan(
        std::move(input_chunks), std::move(qkv_chunks),
        std::move(compute_chunks), std::move(spans), chunk_mode, {});
}

FmbThreeStageChunkPlan compose_fmb_three_stage_chunk_plan(
    std::vector<ChunkInfo> input_chunks,
    std::vector<ChunkInfo> qkv_chunks,
    std::vector<ChunkInfo> compute_chunks,
    std::vector<FmbExecutionSpan> spans,
    ChunkMode chunk_mode,
    FmbStageBoundaryPolicies boundary_policies) {
    FmbThreeStageChunkPlan result{
        make_stage(std::move(input_chunks)),
        make_stage(std::move(qkv_chunks)),
        make_stage(std::move(compute_chunks)),
        std::move(spans),
        boundary_policies,
        chunk_mode,
    };
    validate_fmb_three_stage_chunk_plan(result);
    return result;
}

}  // namespace v3
