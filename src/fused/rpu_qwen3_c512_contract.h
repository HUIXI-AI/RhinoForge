// Exact plain Qwen3-32B P512 W8A8 schedule. Full projections; tiled FP16 residual I/O.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace v3::qwen3_c512 {
inline constexpr int64_t kPolicySite = 0x515743353132504cLL;
inline constexpr int64_t kDmaSite = 0x515743353132444dLL;
inline constexpr int64_t kFinalNormSite = 0x515743353132464eLL;
inline constexpr int64_t kRingSelector = 204;
inline constexpr int64_t kRmsSelector = 4;
inline constexpr int64_t kRows = 512;
inline constexpr int64_t kSegmentRows = 128;
inline constexpr int64_t kHeadReserve = 65536;
enum class DmaRole : int64_t { InputNorm, InputResidual, PostStore, PostNorm, PostResidual, Output };
// Two independently owned roots form the O output only while both are live.
// The prefix later owns compact MLP Q; the suffix retires before Gate is written.
inline constexpr int64_t kOprojHalfBytes = 2621440;
inline constexpr std::array<DmaRole, 5> kDmaRoles{{DmaRole::InputNorm,
    DmaRole::InputResidual, DmaRole::PostStore, DmaRole::PostResidual, DmaRole::Output}};
struct Buffer { const char* name; int64_t bytes; int first, last; const char* alias = nullptr; };
inline constexpr std::array<Buffer, 26> kBuffers{{
    // Reserve the prefix before Q/K scratch, and O input before SDPA retires.
    // This deterministic first-fit placement must still pass the runtime
    // adjacency check: declaration order alone is not an ownership proof.
    {"a8_o_sx", 1024, 3, 6}, {"oproj", kOprojHalfBytes, 3, 9},
    {"k_norm_scratch", 131072, 3, 3},
    {"q", 1048576, 2, 4}, {"k", 131072, 2, 4}, {"v", 131072, 2, 4},
    {"output", 1048576, 4, 5}, {"oproj_suffix", kOprojHalfBytes, 6, 7},
    {"sdpa_tmp", 262144, 4, 4}, {"gate", 3276800, 9, 10},
    {"down", 5242880, 11, 12}, {"lm_head_out", 50176, 13, 13},
    {"a8_qkv", 2621440, 1, 2}, {"a8_qkv_sx", 1024, 1, 2},
    {"a8_o", 524288, 4, 6},
    {"a8_mlp", kOprojHalfBytes, 7, 9, "oproj"}, {"a8_mlp_sx", 1024, 7, 9},
    {"a8_down", 1638400, 10, 11}, {"a8_down_sx", 1024, 10, 11},
    {"residual1", 1310720, 1, 1}, {"o_residual_tile", 1310720, 7, 7},
    {"down_residual_tile", 1310720, 12, 12},
    {"final_norm_tile", 1310720, 13, 13}, {"q_norm_scratch", 1048576, 3, 3},
    {"o_compact_residual", 163840, 7, 7},
    {"down_compact_residual", 163840, 12, 12},
}};
inline std::vector<int64_t> policy_arguments(int64_t physical) {
    return {3, physical, 512, 512, 8, 1, 128, 4, 262144, 6882304, kHeadReserve,
        kOprojHalfBytes, 3, 9, 6, 7, 7, 9, 16, 163840};
}
inline std::vector<int64_t> ring_arguments() {
    return {2, 512, 5120, 8, 128, 4, 2, 16, 163840, 1310720, 2, 63488};
}
inline bool residual_dma(DmaRole role) {
    return role == DmaRole::InputResidual || role == DmaRole::PostResidual;
}
inline int64_t dma_selector(DmaRole role) { return residual_dma(role) ? 2 : 1; }
inline std::vector<int64_t> dma_arguments(DmaRole role) {
    return {2, 512, 5120, 8, 128, 4, static_cast<int64_t>(role),
        residual_dma(role) ? 16 : 128, residual_dma(role) ? 163840 : 0};
}
inline std::vector<int64_t> final_norm_arguments() { return {1, 512, 5120, 8, 128, 4, 6, 0x0011}; }
inline std::vector<int64_t> up_mul_arguments() {
    return {1, 512, 25600, 5120, 1, 8, 3200, 5120, 512, 64, 128, 8, 8, 32, 16, 16, 1};
}
inline bool oproj_regions_valid(uint64_t prefix, uint64_t suffix, uint64_t quant,
                                 uint64_t spm_base, uint64_t spm_bytes) {
    constexpr uint64_t full = 2 * kOprojHalfBytes;
    return prefix % 256 == 0 && suffix % 256 == 0 && quant == prefix &&
        suffix == prefix + kOprojHalfBytes && prefix >= spm_base &&
        spm_bytes >= full && prefix - spm_base <= spm_bytes - full &&
        prefix <= UINT32_MAX && full <= uint64_t(UINT32_MAX) + 1 - prefix;
}
inline bool residual_regions_valid(uint64_t partial, uint64_t compact,
                                   uint64_t output, uint64_t spm_base,
                                   uint64_t spm_bytes) {
    const std::array<uint64_t, 3> starts{partial, compact, output};
    const std::array<uint64_t, 3> sizes{5242880, 163840, 1310720};
    for (size_t i = 0; i < starts.size(); ++i) {
        if (starts[i] % 256 != 0 || starts[i] < spm_base ||
            sizes[i] > spm_bytes || starts[i] - spm_base > spm_bytes - sizes[i] ||
            starts[i] > UINT32_MAX || sizes[i] > uint64_t(UINT32_MAX) + 1 - starts[i])
            return false;
        for (size_t j = 0; j < i; ++j)
            if (starts[i] < starts[j] + sizes[j] && starts[j] < starts[i] + sizes[i])
                return false;
    }
    return true;
}
inline std::vector<int64_t> identity() {
    auto out = policy_arguments(512);
    out.insert(out.begin(), kPolicySite);
    for (const auto& b : kBuffers) out.insert(out.end(), {b.bytes, b.first, b.last, b.alias ? 1 : 0});
    const auto up = up_mul_arguments();
    out.insert(out.end(), up.begin(), up.end());
    return out;
}
} // namespace v3::qwen3_c512
