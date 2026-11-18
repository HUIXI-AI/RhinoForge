// Bounded plain Qwen3-32B prefill. Decode and all FP16 residual/math stay ordinary.
#pragma once
#include "../ops/rpu_linear_w8a8_contract.h"
#include "rpu_qwen3_c192_contract.h"
#include "rpu_qwen3_c512_contract.h"
#include <array>
#include <vector>

namespace v3::qwen3_w8a8 {
inline constexpr int64_t kLinearSelector = 201;
inline constexpr int64_t kQuantSelector = 1;
inline constexpr int64_t kRmsQuantSelector = 2;
inline constexpr int64_t kMulQuantSelector = 3;
inline constexpr int64_t kPolicyVersion = 4;
// COMPLETE route arguments are nonnegative. Preserve the exact GP10/11
// FP16 bit patterns instead of storing the signed numerical clamp minimum.
inline constexpr int64_t kQuantClampMaxBits = 0x57f0; // FP16(+127)
inline constexpr int64_t kQuantClampMinBits = 0xd7f0; // FP16(-127)
inline constexpr int64_t kRmsNormV32Grid8Selector = 6;
inline constexpr int64_t kRmsEpsilonBits = 0x0011; // FP16(1e-6)
inline constexpr int64_t kRmsReciprocalBits = 0x0a66; // FP16(1/5120)
inline constexpr int64_t kPolicySite = 0x5157454e4138504cLL;
inline constexpr std::array<int64_t, 4> kQuantSites{
    0x51574138514b5651LL, 0x515741384f505251LL,
    0x515741384d4c5051LL, 0x51574138444f5751LL};
inline constexpr int64_t kGateSite = 0x5157413847415445LL;
inline constexpr int64_t kUpSite = 0x5157413855505052LL;
inline constexpr int64_t kDownSite = 0x51574138444f574eLL;
inline constexpr int64_t kReduceSite = 0x5157413852454455LL;
enum class Projection : int { Q, K, V, O, Gate, Up, Down };
inline constexpr std::array<int64_t, 7> kGlobalN{8192, 1024, 1024, 5120, 25600, 25600, 5120};
inline constexpr std::array<int64_t, 7> kGlobalK{5120, 5120, 5120, 8192, 5120, 5120, 25600};
inline constexpr std::array<int, 7> kPartition{1, 1, 1, 0, 1, 1, 0};
inline constexpr std::array<int64_t, 4> kQuantK{5120, 1024, 5120, 3200};
inline constexpr bool valid_rows(int64_t rows) {
    return rows == 16 || rows == 32 || rows == 64 || rows == 128 || rows == 192 || rows == 256 || rows == 512;
}
inline void require_rows(int64_t rows) {
    if (!valid_rows(rows)) throw std::invalid_argument("Qwen3 prefill W8A8 requires C16/32/64/128/192/256/512");
}
inline constexpr bool uses_rms_quant(int64_t rows, size_t role) {
    return (rows == 128 || rows == 192 || rows == 256 || rows == 512) && (role == 0 || role == 2);
}
inline constexpr bool uses_mul_quant(int64_t rows, size_t role) {
    return (rows == 128 || rows == 192) && role == 3;
}
inline constexpr int64_t quant_selector(int64_t rows, size_t role) {
    return rows == 512 && uses_rms_quant(rows, role) ? qwen3_c512::kRmsSelector : uses_rms_quant(rows, role) ? kRmsQuantSelector :
        uses_mul_quant(rows, role) ? kMulQuantSelector : kQuantSelector;
}
inline rpu_pl_w8a8::Shape shape(int64_t rows, Projection projection) {
    require_rows(rows);
    const auto p = static_cast<size_t>(projection);
    if (p >= kGlobalN.size()) throw std::invalid_argument("Qwen3 W8A8 projection out of range");
    return rpu_pl_w8a8::physical_shape(rows, kGlobalN[p], kGlobalK[p],
        kPartition[p] ? rpu_pl_w8a8::Partition::Col : rpu_pl_w8a8::Partition::Row, 8, false);
}
inline std::vector<int64_t> linear_arguments(int64_t rows, Projection projection) {
    if (rows == 512 && projection == Projection::Up)
        return qwen3_c512::up_mul_arguments();
    if (rows == 256 && projection == Projection::Up)
        return qwen3_c192::up_mul_arguments(true);
    const auto s = shape(rows, projection);
    const auto t = rpu_pl_w8a8::select_tile(s);
    const auto c = rpu_pl_w8a8::installed_candidate(t, s.local_k);
    const auto p = static_cast<size_t>(projection);
    return {1, rows, kGlobalN[p], kGlobalK[p], kPartition[p], 8,
        s.local_n, s.local_k, static_cast<int64_t>(c.ordinal),
        t.m, t.n, t.k, t.resident, t.single_k, t.keep_sw, t.keep_bias,
        t.sx_fused, t.hoist_valu, t.n_major, t.vlm_rows,
        8, 8, 32, 16}; // signed X/W bits, accumulator bits, output bits
}
inline constexpr int64_t linear_selector(int64_t rows, Projection projection) {
    return (rows == 256 || rows == 512) && projection == Projection::Up
        ? qwen3_c192::kUpMulSelector : kLinearSelector;
}
inline std::vector<int64_t> quant_arguments(int64_t rows, size_t role) {
    require_rows(rows);
    if (role >= kQuantK.size()) throw std::invalid_argument("Qwen3 W8A8 quant role out of range");
    if (rows == 512 && uses_rms_quant(rows, role))
        return {role == 2 ? 6 : 5, rows, kQuantK[role], 8, 16, 8, 16, kQuantClampMaxBits, kQuantClampMinBits, 1,
            kRmsNormV32Grid8Selector, kRmsEpsilonBits, kRmsReciprocalBits, 128, 4, 8, 1, 1};
    if (uses_mul_quant(rows, role))
        return {4, rows, kQuantK[role], 8, 16, 8, 16,
            kQuantClampMaxBits, kQuantClampMinBits, 1, 16, 16, 8, 1, 1};
    if (uses_rms_quant(rows, role))
        // Original V32_GRID8 RMSNorm (including its FP16 retirement), then
        // compact INT8 quantization. Grid8, 16/24 rows per warp, four-row batches.
        return {3, rows, kQuantK[role], 8, 16, 8, 16, kQuantClampMaxBits, kQuantClampMinBits, 1,
            kRmsNormV32Grid8Selector, kRmsEpsilonBits, kRmsReciprocalBits,
            8, 1, 1, rows / 8, 4};
    return {2, rows, kQuantK[role], 8, 16, 8, 16, kQuantClampMaxBits, kQuantClampMinBits, 1};
}
struct Buffer {
    const char* name;
    int64_t bytes;
    int first, last;
};
inline std::array<Buffer, 8> buffers(int64_t rows) {
    require_rows(rows);
    const int64_t sx = ((rows * 2 + 255) / 256) * 256;
    // Each Q/SX pair is a real root. First-fit may reuse only dead storage.
    // The Down quantizer finishes reading Gate before Down's phase-8 writes.
    return {{{"a8_qkv", rows * 5120, 2, 2}, {"a8_qkv_sx", sx, 2, 2},
             {"a8_o", rows * 1024, 5, 5}, {"a8_o_sx", sx, 5, 5},
             {"a8_mlp", rows * 5120, 6, 7}, {"a8_mlp_sx", sx, 6, 7},
             {"a8_down", rows * 3200, 7, 8}, {"a8_down_sx", sx, 7, 8}}};
}
inline std::vector<int64_t> identity() {
    std::vector<int64_t> out{kPolicySite, kPolicyVersion, kLinearSelector,
        kQuantSelector, kRmsQuantSelector, kMulQuantSelector, 512, 128};
    for (int64_t rows : {16, 32, 64, 128}) {
        for (int p = 0; p != 7; ++p) {
            const auto args = linear_arguments(rows, static_cast<Projection>(p));
            out.insert(out.end(), args.begin(), args.end());
        }
        for (size_t role = 0; role != 4; ++role) {
            const auto args = quant_arguments(rows, role);
            out.insert(out.end(), args.begin(), args.end());
        }
        for (const auto& b : buffers(rows)) out.insert(out.end(), {b.bytes, b.first, b.last});
    }
    return out;
}
} // namespace v3::qwen3_w8a8
