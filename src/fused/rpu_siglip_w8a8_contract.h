#pragma once

#include "../ops/rpu_linear_w8a8_contract.h"
#include <cstdint>
#include <array>
#include <cstddef>
#include <utility>
#include <stdexcept>
#include <vector>

// Cold Pi Vision LN1 -> shared activation quantization -> three independent
// Q/K/V projections. These arguments are part of the COMPLETE physical plan,
// not a late replacement of the ordinary A16 AUTO_TILE route.
namespace siglip_qkv_w8a8 {
inline void validate_rows(int64_t rows, int64_t images) {
    if ((images != 2 && images != 3) || rows != images * 256)
        throw std::invalid_argument("Pi Vision QKV W8A8 requires 2x256 or 3x256 rows");
}

inline rpu_pl_w8a8::Candidate candidate(int64_t rows) {
    validate_rows(rows, rows / 256);
    const auto shape = rpu_pl_w8a8::physical_shape(
        rows, 1280, 1152, rpu_pl_w8a8::Partition::Col, 8, true);
    return rpu_pl_w8a8::installed_candidate(
        rpu_pl_w8a8::select_tile(shape), shape.local_k);
}

inline std::vector<int64_t> norm_arguments(int64_t rows) {
    validate_rows(rows, rows / 256);
    // ABI1, M/H/TP, exact epsilon numerator/denominator, FP16 LN retirement,
    // signed INT8 Q, FP16 SX, complete padded SX element extent.
    return {1, rows, 1152, 8, 1, 1000000, 16, 8, 16, ((rows + 15) / 16) * 16};
}

inline std::vector<int64_t> linear_arguments(int64_t rows) {
    const auto selected = candidate(rows);
    const auto& t = selected.tile;
    // ABI1, physical global M/N/K, TP, col, bias, 32-bit accumulator/FP16 Y,
    // exact installed tile and every compile-time scheduling flag.
    return {1, rows, 1280, 1152, 8, 1, 1, 32, 16,
        static_cast<int64_t>(selected.ordinal), selected.compiled_local_k,
        t.m, t.n, t.k, t.resident, t.single_k, t.keep_sw, t.keep_bias,
        t.sx_fused, t.hoist_valu, t.n_major};
}
}  // namespace siglip_qkv_w8a8

// MLP keeps the existing full LayerWide roots. Only their dead subregions are
// reused: O partial -> owner Q -> FC2 partial; compact raw remains live until
// the unchanged compact-residual consumer. SX is reused after FC1 has consumed
// it, and GELU's Q occupies the unused tail of input_norm.
namespace siglip_mlp_w8a8 {
struct Workspace {
    uint64_t full_bytes, raw_bytes, fc1_q_bytes, fc2_q_offset, fc2_q_bytes;
    uint64_t sx_offset, sx_bytes, fc1_bytes;
};

inline Workspace workspace(int64_t rows) {
    siglip_qkv_w8a8::validate_rows(rows, rows / 256);
    const uint64_t m = static_cast<uint64_t>(rows);
    Workspace w{m * 1152 * 2, m * 1152 * 2 / 8, m * 1152,
        m * 1152 * 2 / 8, m * 544, m * (1152 * 2 / 8 + 544),
        m * 2, m * 544 * 2};
    if (w.sx_offset + w.sx_bytes > w.full_bytes ||
        w.fc1_q_bytes > w.full_bytes || w.fc2_q_offset % 256 ||
        w.sx_offset % 256 || w.sx_bytes % 256)
        throw std::invalid_argument("Pi Vision MLP W8A8 invalid bounded workspace");
    return w;
}

inline void validate_root_extents(int64_t rows, uint64_t oproj_bytes,
                                 uint64_t input_norm_bytes, uint64_t fc1_bytes,
                                 uint64_t residual_bytes) {
    const auto w = workspace(rows);
    if (oproj_bytes < w.full_bytes || input_norm_bytes < w.full_bytes ||
        fc1_bytes < w.fc1_bytes || residual_bytes < w.full_bytes)
        throw std::invalid_argument("Pi Vision MLP W8A8 requires complete live SPM roots");
}

inline void validate_roots(int64_t rows, uint64_t spm_base, uint64_t spm_bytes,
                          uint64_t oproj, uint64_t input_norm,
                          uint64_t fc1, uint64_t residual) {
    const auto w = workspace(rows);
    const std::array<std::pair<uint64_t, uint64_t>, 4> roots{{
        {oproj, w.full_bytes}, {input_norm, w.full_bytes},
        {fc1, w.fc1_bytes}, {residual, w.full_bytes}}};
    for (size_t i = 0; i < roots.size(); ++i) {
        const auto [address, bytes] = roots[i];
        if (address < spm_base || address % 256 || bytes > spm_bytes ||
            address - spm_base > spm_bytes - bytes)
            throw std::invalid_argument("Pi Vision MLP W8A8 root exceeds live SPM extent");
        for (size_t j = 0; j < i; ++j)
            if (address < roots[j].first + roots[j].second && roots[j].first < address + bytes)
                throw std::invalid_argument("Pi Vision MLP W8A8 full roots overlap");
    }
}

inline rpu_pl_w8a8::Candidate candidate(int64_t rows, bool second) {
    (void)workspace(rows);
    const auto shape = rpu_pl_w8a8::physical_shape(rows,
        second ? 1152 : 4352, second ? 4352 : 1152,
        second ? rpu_pl_w8a8::Partition::Row : rpu_pl_w8a8::Partition::Col, 8, true);
    return rpu_pl_w8a8::installed_candidate(rpu_pl_w8a8::select_tile(shape), shape.local_k);
}

inline std::vector<int64_t> owner_arguments(int64_t rows) {
    const auto w = workspace(rows);
    // Original owner reduce/LN ABI + INT8 Q / FP16 SX, exact Q=partial alias,
    // shared SX offset in input_norm, and the unchanged 21 barrier epochs.
    return {1, rows, 1152, 8, rows / 8, 2, 63488, int64_t(w.raw_bytes),
        int64_t(w.full_bytes), 1000000, 1, 1, 0, 8, 16,
        int64_t(w.fc1_q_bytes), int64_t(w.sx_offset), int64_t(w.sx_bytes), 21};
}

inline std::vector<int64_t> gelu_arguments(int64_t rows) {
    const auto w = workspace(rows);
    // Main TANH GELU FP16 retirement -> INT8 Q / FP16 SX; Q and SX are
    // disjoint from the compact raw prefix and the full FP16 FC1 input.
    return {1, rows, 544, 8, 16, 8, 16, int64_t(w.fc2_q_offset),
        int64_t(w.fc2_q_bytes), int64_t(w.sx_offset), int64_t(w.sx_bytes),
        int64_t(w.raw_bytes), int64_t(w.full_bytes), int64_t(w.fc1_bytes)};
}

inline std::vector<int64_t> linear_arguments(int64_t rows, bool second) {
    const auto selected = candidate(rows, second);
    const auto& t = selected.tile;
    return {1, rows, second ? 1152 : 4352, second ? 4352 : 1152, 8,
        second ? 0 : 1, 1, 32, 16, static_cast<int64_t>(selected.ordinal),
        selected.compiled_local_k, t.m, t.n, t.k, t.resident, t.single_k,
        t.keep_sw, t.keep_bias, t.sx_fused, t.hoist_valu, t.n_major};
}
}  // namespace siglip_mlp_w8a8
