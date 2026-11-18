// Generated from rhino-ops linear_20261021/g05_contracts.json.
#pragma once
#include "rpu_kernel_cache.h"
#include <vector>
namespace rpu_g05_linear {
struct Contract { int m, n, k; uint16_t gx, gy, gz; };
inline Contract contract(KernelId id) {
  switch (id) {
    case KernelId::G05_KV_OWNER_FP16_ACC32_M32N256K1024: return {80, 80, 512, 4, 1, 1};
    case KernelId::G05_Q_GATE_FP16_ACC32_M32N256K1024: return {80, 80, 512, 8, 1, 1};
    case KernelId::G05_VISION_QKV_FP16_ACC32_M256N128K1024: return {368, 64, 128, 6, 1, 1};
    case KernelId::G05_KV_OWNER_W8A16_ACC16_M32N256K1024: return {48, 48, 768, 6, 1, 1};
    case KernelId::G05_Q_GATE_W8A16_ACC16_M32N256K1024: return {48, 48, 768, 12, 1, 1};
    case KernelId::G05_VISION_QKV_W8A16_ACC16_M256N128K1024: return {512, 48, 128, 9, 1, 1};
    case KernelId::G05_KV_OWNER_W8A16_ACC32_M32N256K1024: return {96, 48, 512, 6, 1, 1};
    case KernelId::G05_Q_GATE_W8A16_ACC32_M32N256K1024: return {96, 48, 512, 12, 1, 1};
    case KernelId::G05_VISION_QKV_W8A16_ACC32_M256N128K1024: return {400, 48, 128, 9, 1, 1};
    case KernelId::G05_GATE_UP_FP16_ACC16_M32N512K1024: return {208, 96, 256, 12, 1, 1};
    case KernelId::G05_TEXT_GATE_UP_FP16_ACC16_M256N768K2048: return {432, 80, 128, 20, 1, 1};
    case KernelId::G05_GATE_UP_FP16_ACC32_M32N512K1024: return {160, 96, 256, 12, 1, 1};
    case KernelId::G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048: return {368, 64, 128, 24, 1, 1};
    case KernelId::G05_GATE_UP_W8A16_ACC16_M32N512K1024: return {64, 80, 512, 14, 1, 1};
    case KernelId::G05_TEXT_GATE_UP_W8A16_ACC16_M256N768K2048: return {400, 80, 128, 20, 1, 1};
    case KernelId::G05_GATE_UP_W8A16_ACC32_M32N512K1024: return {48, 80, 512, 14, 1, 1};
    case KernelId::G05_TEXT_GATE_UP_W8A16_ACC32_M256N768K2048: return {288, 80, 128, 20, 1, 1};
    case KernelId::G05_QKV_GATE_OWNER_W8A16_ACC16_M32N256K1024: return {48, 48, 768, 18, 1, 1};
    case KernelId::G05_QKV_GATE_OWNER_W8A16_ACC32_M32N256K1024: return {96, 48, 512, 18, 1, 1};
    case KernelId::G05_GATE_KV_OWNER_W8A16_ACC16_M32N256K1024_N128: return {48, 48, 768, 12, 1, 1};
    case KernelId::G05_GATEUP_RESIDENT_FP16_ACC32_M304N80K128: return {160, 96, 256, 6, 1, 1};
    case KernelId::G05_GATEUP_RESIDENT_W8A16_ACC16_M352N96K128: return {64, 80, 512, 7, 1, 1};
    case KernelId::G05_GATEUP_RESIDENT_W8A16_ACC32_M240N96K128: return {48, 80, 512, 7, 1, 1};
    case KernelId::G05_O_RESIDENT_W8A16_ACC16_M32N1024K256: return {64, 80, 512, 13, 1, 1};
    case KernelId::G05_DOWN_PARTIAL672_W8A16_ACC16_M32N1024K512: return {48, 48, 768, 22, 1, 1};
    case KernelId::G05_VISION_FC2_WEIGHT_OUTER_W8A16_ACC16_M768N1024K512: return {64, 80, 512, 13, 1, 1};
    default: TORCH_CHECK(false, "Unknown current G05 Linear contract");
  }
}
inline std::vector<uint16_t> grid(KernelId id) {
  const auto c = contract(id); return {c.gx, c.gy, c.gz};
}
} // namespace rpu_g05_linear
