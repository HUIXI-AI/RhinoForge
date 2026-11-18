#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include "rpu_linear_tiling.h"
#include "rpu_grouped_device_m_contract.h"
#include "rpu_grouped_w8_device_m_contract.h"
#include "rpu_grouped_extra_device_m_contract.h"

// Unified main GroupedLinear metadata and candidate selection.
namespace rpu_grouped_tiling {

enum class DispatchKind : int64_t {
    Static = 0,
    Fp16Acc16DeviceM = 1,
    W8Acc32DeviceM = 2,
    Fp16Acc32N80DeviceM = 3,
    Fp16Acc32N32DeviceM = 4,
    W8Acc16N48DeviceM = 5,
    W8Acc32N48DeviceM = 6,
};

struct Dispatch {
    rpu_pl_tiling::TilePick tile;
    DispatchKind kind;
};

// total_m is sum(m_sizes): the packed expert-assignment rows, not source
// token count. Expert counts remain on device and can change during REPLAY.
// num_experts is the declared group capacity; zero means unspecified.
inline Dispatch select_dispatch(int64_t total_m, int64_t n, int64_t k,
                                int weight_mode, bool acc32,
                                int x_dtype = 2, int forced_n = 0, int forced_k = 0,
                                int64_t num_experts = 0) {
    // MGrpBase advances with sadd.u16 in both generated grouped programs.
    // Bound packed assignments, not source tokens, before selecting a kernel.
    if (total_m <= 0 || total_m > UINT16_MAX || k <= 0 || k > 65535)
        throw std::invalid_argument("invalid grouped packed activation geometry");
    const auto weight = weight_mode == 2 ? rpu_pl_tiling::WeightType::Fp16
        : weight_mode == 1 ? rpu_pl_tiling::WeightType::Int8
        : weight_mode == 5 ? rpu_pl_tiling::WeightType::Fp8E4M3
        : throw std::invalid_argument("unsupported grouped weight mode");
    const bool forced = forced_n != 0 || forced_k != 0;
    const bool measured_fp8 = weight_mode == 5 && num_experts == 256 &&
        !acc32 && x_dtype == 2 && !forced && (n == 64 || n == 96) && k == 2048 &&
        (total_m == 8 || total_m == 512 || total_m == 1024);
    auto tile = rpu_pl_tiling::select_tile(
        1, n, static_cast<int>(k), weight, acc32, rpu_pl_tiling::Family::Grouped,
        0, x_dtype, measured_fp8 ? (n == 96 ? 48 : 64) : forced_n,
        measured_fp8 ? 256 : forced_k);
    // Outside the paired FP8 geometry, retain the existing INT8 device-M
    // bridge. The shared byte kernel still receives the original FP8 WDType.
    if (weight_mode == 5 && x_dtype == 2 && !forced && !measured_fp8) {
        const auto measured = rpu_pl_tiling::select_tile(
            1, n, static_cast<int>(k), rpu_pl_tiling::WeightType::Int8, acc32,
            rpu_pl_tiling::Family::Grouped);
        if (rpu_grouped_extra_device_m::select_kind(total_m, weight_mode, acc32,
                measured.n_tile, measured.k_tile) != 0)
            tile = measured;
    }
    const auto kind = rpu_grouped_device_m::eligible(total_m, weight_mode, acc32,
        tile.n_tile, tile.k_tile, x_dtype, forced) ? DispatchKind::Fp16Acc16DeviceM
        : rpu_grouped_w8_device_m::eligible(total_m, weight_mode, acc32,
            tile.n_tile, tile.k_tile, x_dtype, forced) ? DispatchKind::W8Acc32DeviceM
        : static_cast<DispatchKind>(rpu_grouped_extra_device_m::select_kind(
            total_m, weight_mode, acc32, tile.n_tile, tile.k_tile, x_dtype, forced));
    return {tile, kind};
}

inline std::array<int64_t, 5> dispatch_identity(
    int64_t total_m, int64_t n, int64_t k, int weight_mode, bool acc32,
    int64_t num_experts = 0) {
    const auto dispatch = select_dispatch(
        total_m, n, k, weight_mode, acc32, 2, 0, 0, num_experts);
    return {total_m, dispatch.tile.m_tile, dispatch.tile.n_tile,
            dispatch.tile.k_tile, static_cast<int64_t>(dispatch.kind)};
}

// rhino-ops 135b93fc grouped UINT4 metadata:
// [expert,Qblock4,Nblock64,core,qslot4,n64], FP16 S and negative zero.
// Weight uses N32K32; the existing controller-striped metadata is retained.
struct Uint4MetadataLayout {
    static constexpr int64_t version = 1;
    int64_t q_blocks, n_blocks, alignment_bytes;
    int64_t group_block_bytes, expert_bytes;
};

inline Uint4MetadataLayout uint4_metadata_layout(
    int64_t local_n, int64_t local_k, int64_t group_size, int64_t cores) {
    if (local_n <= 0 || local_k <= 0 || cores < 1 || cores > 8 ||
        (group_size != 32 && group_size != 64 && group_size != 128) ||
        local_k % group_size)
        throw std::invalid_argument("invalid grouped UINT4 metadata geometry");
    const int64_t q_blocks = (local_k / group_size + 3) / 4;
    const int64_t n_blocks = (local_n + 63) / 64;
    const int64_t alignment = cores * 512;
    const int64_t group_block_bytes = n_blocks * alignment;
    return {q_blocks, n_blocks, alignment, group_block_bytes,
            q_blocks * group_block_bytes};
}

// Weight packing is N32/K32; metadata keeps its independent Q4/N64 version.
inline constexpr int64_t uint4_weight_layout_version = 2;

// Grouped uses the main PL M1 ranking, filtered by grouped VLM capacity.
struct Tile { int64_t n, m, k; };

inline Tile select_uint4(int64_t m, int64_t experts, int64_t n, int64_t k,
                        int group_size = 32, bool acc32 = false) {
    if (m <= 0 || m > UINT16_MAX || experts <= 0 || experts > rpu_grouped_device_m::kMaxGroups)
        throw std::invalid_argument("GroupedParallelLinear requires M in [1,65535] and experts in [1,32768]");
    // Query callers use int64 shapes, before the launcher's typed-register
    // checks. Reject before padding/narrowing so large shapes cannot alias a
    // legal N/K tile. Logical N16 still pads to the packed N32 geometry.
    if (n <= 0 || n > UINT32_MAX / 2 || n % 16 || k <= 0 || k > UINT16_MAX)
        throw std::invalid_argument("Grouped UINT4 requires aligned u32 row strides and u16 K");
    int measured_n = 0, measured_k = 0;
    // Exact measured assignment counts and logical shard geometry. Do not
    // infer the live routing distribution from total M or padded local N.
    if (experts == 256 && !acc32) {
        if (n == 128 && k == 2048 && group_size == 128) {
            if (m == 8 || m == 512 || m == 1024) {
                measured_n = 32;
                measured_k = 256;
            }
        } else if (n == 2048 && k == 128 && group_size == 64 && m == 8) {
            measured_n = 128;
            measured_k = 128;
        } else if (n == 64 && k == 2048 && group_size == 128 &&
                   (m == 8 || m == 512 || m == 1024)) {
            measured_n = 32;
            measured_k = 768;
        } else if (n == 2048 && k == 64 && group_size == 64 &&
                   (m == 8 || m == 512 || m == 1024)) {
            measured_n = 128;
            measured_k = m == 1024 ? 128 : 256;
        }
        // N2048/K128 Down M512/M1024 keep the default after balanced-route losses.
    }
    const auto tile = rpu_pl_tiling::select_tile(
        1, (n + 31) / 32 * 32, static_cast<int>(k),
        rpu_pl_tiling::WeightType::Uint4, acc32,
        rpu_pl_tiling::Family::Grouped, group_size, 2, measured_n, measured_k);
    return {tile.n_tile, tile.m_tile, tile.k_tile};
}
}  // namespace rpu_grouped_tiling
