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
    if (q_heads == 40 && hidden == 5120 && intermediate == 17408) return 40;
    if (q_heads == 64 && hidden == 5120 && intermediate == 25600) return 64;
    return 0;
}

// GR00T N1.7 runs only the first 16 layers of exact Qwen3-VL-2B text
// (select_layer 16, headless). It reuses the 2B template above; only
// CausalDecoderModel's M-RoPE [24,20,20] + DeepStack admission accepts it.
inline int qwen3_vl_text_prefix_num_layers(
        int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    return q_heads == 16 && kv_heads == 8 && head_dim == 128 &&
           hidden == 2048 && intermediate == 6144 ? 16 : 0;
}

// Exact dense FP16 Llama-3.2-1B geometry (D64, no QK-norm). It reuses the
// reduced Qwen3 template; CausalDecoderModel additionally binds it to absent
// Q/K norms and the cold 128256 vocabulary before committing weights.
inline int llama_fp16_profile_num_layers(
        int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    return q_heads == 32 && kv_heads == 8 && head_dim == 64 &&
           hidden == 2048 && intermediate == 8192 ? 16 : 0;
}

// Exact InternVLA-N1 System-2 text decoder (Qwen2.5-VL-7B, W8A16, headless):
// Q28/KV4/D128, QKV bias, no QK-norm. Attention stays on four lanes as on eight
// cores; only the MLP follows the budget. CausalDecoderModel binds the bias,
// M-RoPE [16,24,24] and the zero vocabulary before committing weights.
inline int internvla_text_profile_num_layers(
        int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    return q_heads == 28 && kv_heads == 4 && head_dim == 128 &&
           hidden == 3584 && intermediate == 18944 ? 28 : 0;
}

// Exact Wall-OSS 0.5 decoders (Qwen2.5-VL-3B, headless): expert 0 is the text
// backbone, expert 1 the action expert sharing its KV cache. The adapter
// repeats the two checkpoint KV heads once per attention lane, so both owners
// arrive here as KV4 on the four-lane attention template. CausalDecoderModel
// binds bias, no QK-norm, M-RoPE [16,24,24] and precision before weights.
inline int wall_oss_text_profile_num_layers(
        int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    return q_heads == 16 && kv_heads == 4 && head_dim == 128 &&
           hidden == 2048 && intermediate == 11008 ? 36 : 0;
}

inline int wall_oss_action_profile_num_layers(
        int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    return q_heads == 16 && kv_heads == 4 && head_dim == 128 &&
           hidden == 1024 && intermediate == 2048 ? 36 : 0;
}

// Exact HyVLA decoders (MoT VLM and its action expert, headless): Q16/KV4/
// D128, no bias, QK-norm, eps 1e-5, no M-RoPE. Both are KV4 natively, so
// attention stays on four lanes; the VLM MLP follows the budget and the expert
// runs as a four-core child under either reduced root. CausalDecoderModel binds
// the semantics and precision before weights.
inline int hyvla_vlm_profile_num_layers(
        int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    return q_heads == 16 && kv_heads == 4 && head_dim == 128 &&
           hidden == 2048 && intermediate == 6144 ? 32 : 0;
}

inline int hyvla_expert_profile_num_layers(
        int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    return q_heads == 16 && kv_heads == 4 && head_dim == 128 &&
           hidden == 1024 && intermediate == 2048 ? 32 : 0;
}

// Exact LingBot v2 sparse-MoE action expert (headless): Q32/KV8/D128, H768,
// Qwen2.5 QKV bias, no QK-norm. The shared expert arrives zero-padded from
// I704 to 768; the routed experts (I512) cannot split over six lanes, so the
// expert runs as a four-core child under either reduced root and shares the
// four-lane attention cache with its Qwen3-VL text prefix.
inline int lingbot2_expert_profile_num_layers(
        int64_t q_heads, int64_t kv_heads, int64_t head_dim,
        int64_t hidden, int64_t intermediate) {
    return q_heads == 32 && kv_heads == 8 && head_dim == 128 &&
           hidden == 768 && intermediate == 768 ? 36 : 0;
}

// Only execution width is padded. Exact profile admission above always sees
// logical geometry, so checkpoints declaring padded I=9792/3648 are unsupported.
inline int64_t decoder_mlp_intermediate_size(int64_t logical_size, int mlp_tp) {
    if (mlp_tp == 6) {
        if (logical_size == 9728) return 9792;
        if (logical_size == 3584) return 3648;
        if (logical_size == 8192) return 8256;  // Llama-3.2-1B (8192 % 96 == 32)
        if (logical_size == 17408) return 17472;  // exact legacy Qwen3.8 W8 owner
        if (logical_size == 25600) return 25728;
        if (logical_size == 18944) return 19008;  // InternVLA text (18944 % 96 == 32)
        // Wall-OSS text (11008 % 96 == 64). 11136 rather than 11040 keeps the
        // W8 Down row stripe a multiple of 32 (K % 192), like every row above.
        if (logical_size == 11008) return 11136;
    }
    return logical_size;
}

// Signed W4 pgrp packing uses the unified N32K32 ABI, so a row-partitioned Down
// needs K/TP % 32. A six-lane packed MLP is zero-padded to a 192-aligned
// physical width; four/eight lanes and aligned widths keep the logical width.
// Python mirrors this in decoder_packed_w4_mlp_intermediate_size.
inline int64_t decoder_packed_w4_mlp_intermediate_size(int64_t logical_size, int mlp_tp) {
    if (mlp_tp != 6) return logical_size;
    constexpr int64_t alignment = 32 * 6;
    return ((logical_size + alignment - 1) / alignment) * alignment;
}

// Dense NVFP4 v2 linear kernels require local_n % 64 (col Gate/Up) and
// local_k % 64 (row Down, scale K/16 % 4 == 0). At six MLP lanes, physical
// width is zero-padded to a 384-aligned multiple; four/eight lanes keep the
// logical width. Python mirrors this in decoder_nvfp4_mlp_intermediate_size.
inline int64_t decoder_nvfp4_mlp_intermediate_size(int64_t logical_size, int mlp_tp) {
    if (mlp_tp != 6) return logical_size;
    constexpr int64_t alignment = 64 * 6;
    return ((logical_size + alignment - 1) / alignment) * alignment;
}

inline DecoderExecutionTopology resolve_decoder_execution_topology(
        int num_cores, int64_t q_heads, int64_t kv_heads,
        int64_t head_dim, int64_t hidden, int64_t intermediate) {
    if (num_cores < 1 || num_cores > 8 || q_heads <= 0 || kv_heads <= 0 ||
        head_dim <= 0 || head_dim % 16 || hidden <= 0 || hidden % 16 ||
        intermediate <= 0 || intermediate % 16 || q_heads % kv_heads) {
        throw std::invalid_argument("invalid decoder execution topology geometry");
    }
    // Reduced-core admission uses fixed dense FP16 Qwen3 family templates, the
    // exact Llama-3.2-1B row and the exact InternVLA, Wall-OSS, HyVLA and
    // LingBot v2 expert rows. Geometry
    // for other owners must not silently manufacture a new profile.
    if (num_cores != 8) {
        if ((num_cores != 4 && num_cores != 6) ||
            (qwen3_fp16_profile_num_layers(
                 q_heads, kv_heads, head_dim, hidden, intermediate) == 0 &&
             llama_fp16_profile_num_layers(
                 q_heads, kv_heads, head_dim, hidden, intermediate) == 0 &&
             internvla_text_profile_num_layers(
                 q_heads, kv_heads, head_dim, hidden, intermediate) == 0 &&
             wall_oss_text_profile_num_layers(
                 q_heads, kv_heads, head_dim, hidden, intermediate) == 0 &&
             wall_oss_action_profile_num_layers(
                 q_heads, kv_heads, head_dim, hidden, intermediate) == 0 &&
             hyvla_vlm_profile_num_layers(
                 q_heads, kv_heads, head_dim, hidden, intermediate) == 0 &&
             hyvla_expert_profile_num_layers(
                 q_heads, kv_heads, head_dim, hidden, intermediate) == 0 &&
             lingbot2_expert_profile_num_layers(
                 q_heads, kv_heads, head_dim, hidden, intermediate) == 0)) {
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
