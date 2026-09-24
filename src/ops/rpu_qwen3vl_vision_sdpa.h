#pragma once

#include "rpu_helpers.h"

// Ordinary N1200/C608+592 only. Keep the original allocation and layout;
// the candidate consumes less workspace, but does not enlarge the chunk domain.
constexpr int64_t QWEN3VL_N1200_SDPA_TMP_BYTES = 327680;

inline std::optional<SdpaTiling> rpu_qwen3vl_n1200_tm160_tiling(
    int64_t query_rows, int64_t key_rows, int64_t declared_tmp_bytes) {
    if ((query_rows != 608 && query_rows != 592) || key_rows != 1200)
        return std::nullopt;
    const auto tk = choose_tile_k_conservative(10, 4, /*MASK_NONE=*/0, 4);
    if (!tk) return std::nullopt;
    const SdpaTiling t{10, 4, *tk, 160, 64, *tk * 16};
    const int64_t grid_x = CeilDiv(query_rows, t.tile_m);
    const int64_t required_tmp = t.tile_n_v16 * t.tile_k * 2 * grid_x * 32;
    if (grid_x != 4 || t.tile_k != 256 ||
        t.tile_m_v16 > 11 || t.tile_k_v16 > 16 ||
        t.tile_m_v16 * t.tile_n_v16 * t.tile_k_v16 > 1024 ||
        GetMhaAttnUnivVlmNeed(10, 4, *tk, 4, 4, 4, 0) > SDPA_MAX_VLM_SIZE ||
        required_tmp > declared_tmp_bytes)
        return std::nullopt;
    return t;
}
