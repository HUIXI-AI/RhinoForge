// Generated from rhino-ops linear_20261021/qwen3vl_contracts.json.
#pragma once
#include <cstdint>
#include <vector>
#include <stdexcept>
#include "rpu_kernel_cache.h"
namespace rpu_qwen3vl_linear {
struct Contract { int m, n, k; uint16_t gx, gy, gz; };
inline Contract contract(KernelId id) {
  switch (id) {
    case KernelId::QWEN3VL_PREFILL_QKV_PLANAR_DIRECT: return {448, 64, 128, 8, 1, 1};
    case KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV: return {1, 160, 1024, 8, 1, 1};
    case KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8: return {16, 80, 512, 8, 1, 1};
    default: throw std::invalid_argument("Unknown current Qwen3-VL Linear contract");
  }
}
inline std::vector<uint16_t> grid(KernelId id) {
  const auto c = contract(id); return {c.gx, c.gy, c.gz};
}
}
