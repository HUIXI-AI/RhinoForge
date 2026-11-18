// Exact plain Qwen3-32B P512 in-place schedules. Shared by A16 and A8.
#pragma once
#include <cstdint>
#include <vector>
namespace v3::qwen3_c192 {
inline constexpr int64_t kPolicySite = 0x515743313932504cLL;
inline constexpr int64_t kRingSelector = 201;
inline constexpr int64_t kDownReduceSite = 0x5157433139325244LL;
inline constexpr int64_t kMlpSite = 0x5157433139324d4cLL;
inline constexpr int64_t kUpMulSite = 0x51574355504d554cLL;
inline constexpr int64_t kUpMulSelector = 202;
inline constexpr int64_t kC256EagerHeadReserve = 64 * 1024;
inline std::vector<int64_t> up_mul_arguments(bool a8) {
    // Exact Up linear retires FP16 before multiplying the already-SiLU Gate,
    // then overwrites that Gate. No standalone Up output or additional SPM.
    return {1, 256, 25600, 5120, 1, 8, 3200, 5120,
        a8 ? 272 : 320, a8 ? 64 : 112, a8 ? 384 : 128,
        a8 ? 8 : 16, 8, a8 ? 32 : 16, 16, 16, 1};
}
inline constexpr bool valid_rows(int64_t rows, int64_t capacity = 192) {
    return capacity == 256 ? rows == 256 : rows == 192 || rows == 128;
}
inline std::vector<int64_t> ring_arguments(int64_t rows) {
    return {1, rows, 5120, 8, 1};
}
inline std::vector<int64_t> policy_arguments(int64_t physical, bool a8,
        int64_t capacity = 192) {
    if (capacity == 256)
        return {3, physical, 512, 256, 8, a8 ? 1 : 0, 256, 256, 131072,
            9, kC256EagerHeadReserve};
    return {1, physical, 512, 192, 8, a8 ? 1 : 0, 192, 192, 128, 65536};
}
inline std::vector<int64_t> identity(int64_t capacity = 192) {
    // Root lifetimes: residual[1,9], QKV norm[1,2], K scratch[3,3],
    // MLP norm[6,7]. Tail-max SDPA is 65536 bytes at phase4.
    if (capacity == 256) {
        // Full M256 projections, residual [1,9], and no full Up root.
        std::vector<int64_t> result{kPolicySite, 3, kRingSelector, 512, 256, 256, 131072,
            1, 9, 1, 2, 3, 3, 6, 7, 4, 4, 0, 9, kC256EagerHeadReserve};
        for (bool a8 : {false, true}) {
            const auto args = up_mul_arguments(a8);
            result.insert(result.end(), args.begin(), args.end());
        }
        return result;
    }
    return {kPolicySite, 1, kRingSelector, 512, 192, 128, 65536,
        1, 9, 1, 2, 3, 3, 6, 7, 4, 4};
}
} // namespace v3::qwen3_c192
