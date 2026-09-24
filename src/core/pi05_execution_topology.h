#pragma once

#include "execution_topology.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace v3 {

// Pi0.5 owns these exact profiles. Do not extend the generic Qwen decoder
// admission with dimensions that happen to be shared by other architectures.
inline int64_t pi05_mlp_physical_size(
        int64_t logical, int cores, bool w8a16) {
    if (cores != 4 && cores != 6 && cores != 8)
        throw std::invalid_argument("Pi0.5 execution cores must be 4, 6 or 8");
    if (logical != 16384 && logical != 4096)
        throw std::invalid_argument("Pi0.5 requires logical MLP width 16384 or 4096");
    if (cores != 6) return logical;
    const int64_t alignment = cores * (w8a16 ? 32 : 16);
    return ((logical + alignment - 1) / alignment) * alignment;
}

inline int64_t validate_pi05_reduced_geometry(
        int cores, int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t logical_mlp, int64_t layers,
        bool expert, bool w8a16) {
    if ((cores != 4 && cores != 6) || q_heads != 8 || kv_heads != 4 ||
        head_dim != 256 || layers != 18 ||
        hidden != (expert ? 1024 : 2048) ||
        logical_mlp != (expert ? 4096 : 16384))
        throw std::invalid_argument("no admitted reduced-core Pi0.5 decoder profile");
    return pi05_mlp_physical_size(logical_mlp, cores, w8a16);
}

inline DecoderExecutionTopology resolve_pi05_reduced_physical_topology(
        int cores, int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t physical_mlp, bool expert) {
    const int64_t logical = expert ? 4096 : 16384;
    validate_pi05_reduced_geometry(
        cores, q_heads, kv_heads, head_dim, hidden, logical, 18, expert, false);
    if (physical_mlp != pi05_mlp_physical_size(logical, cores, false) &&
        physical_mlp != pi05_mlp_physical_size(logical, cores, true))
        throw std::invalid_argument("invalid Pi0.5 physical MLP width");
    return {cores, 4, cores};
}

inline void validate_pi05_reduced_projection_shapes(
        const std::array<std::array<int64_t, 2>, 7>& shapes,
        int64_t hidden, int64_t physical_mlp) {
    const std::array<std::array<int64_t, 2>, 7> expected{{
        {2048, hidden}, {1024, hidden}, {1024, hidden}, {hidden, 2048},
        {physical_mlp, hidden}, {physical_mlp, hidden}, {hidden, physical_mlp},
    }};
    if (shapes != expected)
        throw std::invalid_argument("Pi0.5 reduced projection payload shape mismatch");
}

}  // namespace v3
