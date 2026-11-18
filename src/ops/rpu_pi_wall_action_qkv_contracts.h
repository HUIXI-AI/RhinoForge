// Generated from rhino-ops linear_20261021/wall_action_qkv_contracts.json.
#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
#include "rpu_kernel_cache.h"
namespace rpu_wall_action_qkv {
struct Contract { int m, n, k; uint16_t gx, gy, gz; };
inline Contract contract(KernelId id) {
  switch (id) {
    case KernelId::WALL_ACTION_QKV_FP16_PARTIAL_MROPE_M20N64K128: return {32, 64, 512, 8, 1, 1};
    case KernelId::WALL_ACTION_QKV_FP16_PARTIAL_MROPE_SPM_WEIGHT_M20N64K128: return {32, 64, 512, 8, 1, 1};
    case KernelId::WALL_ACTION_QKV_FP16_OWNER_PARTIAL_MROPE_M20N64K128: return {32, 64, 512, 8, 1, 1};
    case KernelId::WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64: return {32, 64, 512, 8, 1, 1};
    case KernelId::WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64_SPM_WEIGHT: return {32, 64, 512, 8, 1, 1};
    case KernelId::WALL_ACTION_QKV_OWNER_PARTIAL_MROPE_W8A16_M20N64: return {32, 64, 512, 8, 1, 1};
    default: throw std::invalid_argument("Unknown current Wall Action QKV contract");
  }
}
inline std::vector<uint16_t> grid(KernelId id) {
  const auto c = contract(id); return {c.gx, c.gy, c.gz};
}
}
