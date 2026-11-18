#pragma once

#include "execution_topology.h"

#include <cstdint>
#include <stdexcept>

namespace v3 {

// Gemma4 E4B owns these exact reduced profiles. They reuse existing launchers
// and payloads only; the generic Qwen decoder admission stays closed.
//
// Budget  attention  MLP  per-layer-embedding (PLE)
//   8         2       8      8   (unchanged eight-core layout)
//   6         2       6      4   (MLP zero-padded 10240 -> 10272 FP16,
//                                  10368 INT8 / packed W4)
//   4         2       4      4
//
// Attention keeps its two KV-head lanes at every budget. PLE (width 256 =
// 16 FP16 granules) splits evenly over four or eight lanes but not six; at
// budget 6 it keeps the budget-4 shapes and feeds the six-lane ring through the
// existing partial-producer preparation.
constexpr int64_t GEMMA4_E4B_HIDDEN = 2560;
constexpr int64_t GEMMA4_E4B_INTERMEDIATE = 10240;
constexpr int64_t GEMMA4_E4B_PLE_DIM = 256;
constexpr int GEMMA4_ATTENTION_CORES = 2;

inline void require_gemma4_core_budget(int cores) {
    if (cores != 4 && cores != 6 && cores != 8)
        throw std::invalid_argument("Gemma4 execution cores must be 4, 6 or 8");
}

inline int gemma4_ple_cores(int cores) {
    require_gemma4_core_budget(cores);
    return cores == 8 ? 8 : 4;
}

// Column/row partitions need whole storage granules per lane: 16 FP16, 32 INT8
// or 64 packed-W4 channels. Only six lanes pad; four/eight lanes and the
// checkpoint config keep the logical width.
inline int64_t gemma4_mlp_physical_size(int64_t logical, int mlp_tp,
                                        int64_t lane_granule = 16) {
    if (mlp_tp != 6) return logical;
    if (logical <= 0 || logical % 16)
        throw std::invalid_argument("Gemma4 MLP width must be a positive multiple of 16");
    if (lane_granule != 16 && lane_granule != 32 && lane_granule != 64)
        throw std::invalid_argument("Gemma4 MLP lane granule must be 16, 32 or 64");
    const int64_t alignment = 6 * lane_granule;
    return ((logical + alignment - 1) / alignment) * alignment;
}

inline DecoderExecutionTopology resolve_gemma4_reduced_topology(
        int cores, int64_t q_heads, int64_t kv_heads, int64_t head_dim_max,
        int64_t hidden, int64_t logical_intermediate) {
    if ((cores != 4 && cores != 6) || q_heads != 8 || kv_heads != 2 ||
        head_dim_max != 512 || hidden != GEMMA4_E4B_HIDDEN ||
        logical_intermediate != GEMMA4_E4B_INTERMEDIATE)
        throw std::invalid_argument("no admitted reduced-core Gemma4 E4B profile");
    return {cores, GEMMA4_ATTENTION_CORES, cores};
}

}  // namespace v3
