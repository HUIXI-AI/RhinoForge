#pragma once

#include "core/execution_topology.h"

namespace v3 {

// Exact dense Qwen3.5 reduced-core profiles. KV is the already replicated
// attention width, not the checkpoint's logical GQA count. Layer schedule,
// GDN geometry and actual FP16 / exact legacy W8 tensors are validated before
// installation; this geometry helper alone does not admit a weight policy.
inline bool qwen35_legacy27_geometry(
        int64_t q_heads, int64_t effective_kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    return q_heads == 24 && effective_kv_heads == 24 && head_dim == 256 &&
           hidden == 5120 && intermediate == 17408;
}

inline int64_t qwen35_gdn_scalar_slots(int64_t value_heads, int compute_cores) {
    if ((compute_cores != 4 && compute_cores != 8) || value_heads <= 0 ||
        value_heads % compute_cores != 0)
        throw std::invalid_argument("invalid Qwen3.5 GDN scalar geometry");
    const int64_t local_heads = value_heads / compute_cores;
    return ((local_heads + 7) / 8) * 8;
}

inline int qwen35_reduced_profile_num_layers(
        int64_t q_heads, int64_t effective_kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    if (qwen35_legacy27_geometry(q_heads, effective_kv_heads, head_dim,
                               hidden, intermediate)) return 64;
    if (effective_kv_heads != 4 || head_dim != 256) return 0;
    if (q_heads == 8 &&
        ((hidden == 1024 && intermediate == 3584) ||
         (hidden == 2048 && intermediate == 6144))) return 24;
    if (q_heads == 16 &&
        ((hidden == 2560 && intermediate == 9216) ||
         (hidden == 4096 && intermediate == 12288))) return 32;
    return 0;
}

inline DecoderExecutionTopology resolve_qwen35_reduced_execution_topology(
        int owner, int64_t q_heads, int64_t effective_kv_heads,
        int64_t head_dim, int64_t hidden, int64_t intermediate) {
    if ((owner != 4 && owner != 6) ||
        qwen35_reduced_profile_num_layers(q_heads, effective_kv_heads,
                                         head_dim, hidden, intermediate) == 0) {
        throw std::invalid_argument("no admitted reduced-core Qwen3.5 template");
    }
    return {owner, 4, owner};
}

}  // namespace v3
