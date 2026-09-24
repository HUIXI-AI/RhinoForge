#pragma once

#include <cstdint>
#include <stdexcept>

namespace v3 {

// Execution cores are a prefix of the physical device. DDR controller stripes
// and the KV-cache ABI keep eight lanes even when fewer cores execute kernels.
struct DecoderExecutionTopology {
    static constexpr int physical_kv_cores = 8;
    int num_cores = 8;
    int attention_tp = 8;
    int mlp_tp = 8;
};

inline int resolve_column_projection_cores(int num_cores, int64_t outputs) {
    if (num_cores < 1 || num_cores > 8 || outputs <= 0 || outputs % 16) {
        throw std::invalid_argument("invalid aligned column projection geometry");
    }
    while ((outputs / 16) % num_cores) --num_cores;
    return num_cores;
}

// Exact dense FP16 Qwen3 family geometry. A zero result is not an admitted
// profile; callers with weights must additionally match the returned layer
// count before committing model state. Keep unrelated decoder owners outside
// this reduced-core family even when individual dimensions happen to match.
inline int qwen3_fp16_profile_num_layers(
        int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    if (kv_heads != 8 || head_dim != 128) return 0;
    if (q_heads == 16 &&
        ((hidden == 1024 && intermediate == 3072) ||
         (hidden == 2048 && intermediate == 6144))) return 28;
    if (q_heads == 32 &&
        ((hidden == 2560 && intermediate == 9728) ||
         (hidden == 4096 && intermediate == 12288))) return 36;
    return 0;
}

// Only execution width is padded. Exact profile admission above always sees
// logical geometry, so checkpoints declaring padded I=9792/3648 are unsupported.
inline int64_t decoder_mlp_intermediate_size(int64_t logical_size, int mlp_tp) {
    if (mlp_tp == 6) {
        if (logical_size == 9728) return 9792;
        if (logical_size == 3584) return 3648;
        if (logical_size == 17408) return 17472;  // exact legacy Qwen3.8 W8 owner
    }
    return logical_size;
}

inline DecoderExecutionTopology resolve_decoder_execution_topology(
        int num_cores, int64_t q_heads, int64_t kv_heads,
        int64_t head_dim, int64_t hidden, int64_t intermediate) {
    if (num_cores < 1 || num_cores > 8 || q_heads <= 0 || kv_heads <= 0 ||
        head_dim <= 0 || head_dim % 16 || hidden <= 0 || hidden % 16 ||
        intermediate <= 0 || intermediate % 16 || q_heads % kv_heads) {
        throw std::invalid_argument("invalid decoder execution topology geometry");
    }
    // Reduced-core admission uses fixed dense FP16 Qwen3 family templates.
    // Geometry for other owners must not silently manufacture a new profile.
    if (num_cores != 8) {
        if ((num_cores != 4 && num_cores != 6) ||
            qwen3_fp16_profile_num_layers(
                q_heads, kv_heads, head_dim, hidden, intermediate) == 0) {
            throw std::invalid_argument("no admitted reduced-core decoder template");
        }
        return {num_cores, 4, num_cores};
    }
    int attention = num_cores;
    while (q_heads % attention || kv_heads % attention) --attention;
    const int mlp = resolve_column_projection_cores(num_cores, intermediate);
    return {num_cores, attention, mlp};
}

}  // namespace v3
