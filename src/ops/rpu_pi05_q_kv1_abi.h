#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Current fused ABI only. DDR owners are installed by the retained Graph writer;
// absolute SPM addresses must never be encoded as shifted DDR operands.
namespace rpu_pi05_q_kv1 {
struct SpmArguments {
    uint32_t input, query_output, key_output, value_output;
    uint32_t query_block_scale, query_tensor_scale;
    uint16_t rows, layer;
};

inline std::array<uint16_t, 67> registers(const SpmArguments& a) {
    if (a.rows == 0 || a.rows > 64 || a.layer >= 18)
        throw std::invalid_argument("Pi Q+KV1 requires M1..64 and layer0..17");
    std::array<uint16_t, 67> r{};
    const auto u32 = [&](std::size_t slot, uint32_t value) {
        r[slot] = static_cast<uint16_t>(value);
        r[slot + 1] = static_cast<uint16_t>(value >> 16);
    };
    u32(0, a.input); u32(4, a.query_output);
    r[8] = a.rows; u32(10, 256); r[12] = 1024; r[13] = 16;
    r[15] = 8; u32(16, 2048); u32(18, 512);
    u32(20, a.query_block_scale); u32(22, 4096);
    u32(24, a.query_tensor_scale); r[26] = 1; r[27] = a.layer;
    r[28] = 4;
    // Explicit map from the current pair's live slots. Pair32..39 are V
    // operands, so the historical blanket +32 map is not accepted.
    u32(32, a.input); u32(36, a.key_output);
    r[40] = a.rows; u32(42, 32); r[44] = 64;
    r[46] = 8; r[47] = 1;  // TP8, W8A16 ACC16, no bias.
    u32(48, 2048); u32(50, 64); u32(58, a.value_output);
    r[64] = 6; r[65] = 1; r[66] = 1;
    return r;
}
}  // namespace rpu_pi05_q_kv1
