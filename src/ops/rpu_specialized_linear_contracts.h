// Generated from rhino-ops linear_20261021/specialization_contracts.json.
#pragma once
#include "rpu_kernel_cache.h"
namespace rpu_specialized_linear {
struct Contract { int m, n, k; uint16_t gx, gy, gz; };
inline Contract contract(KernelId id) {
  switch (id) {
    case KernelId::GR00T_FF1_RESIDENT_W8A16_M512N48K128: return {64, 80, 512, 10, 1, 1};
    case KernelId::GR00T_QKV_MULTIBASE_W8A16_M592N32K128: return {592, 32, 128, 18, 1, 1};
    case KernelId::HYVLA_EXPERT_GATEUP_SWIGLU_FP16_V2: return {96, 112, 384, 3, 1, 1};
    case KernelId::HYVLA_EXPERT_GATEUP_W8_S51_N48: return {192, 128, 128, 2, 1, 1};
    case KernelId::HYVLA_EXPERT_KV_FP16_S51: return {208, 96, 256, 4, 1, 1};
    case KernelId::HYVLA_EXPERT_KV_W8_S51: return {352, 48, 192, 6, 1, 1};
    case KernelId::HYVLA_EXPERT_O_RESIDENT_W8_S51: return {288, 128, 128, 8, 1, 1};
    case KernelId::HYVLA_EXPERT_O_RING_POSTNORM_S51: return {176, 128, 256, 8, 1, 1};
    case KernelId::HYVLA_EXPERT_O_RING_POSTNORM_W8_S51: return {288, 128, 128, 8, 1, 1};
    case KernelId::HYVLA_GATEUP_RESIDENT_PAIR_FP16_V2: return {112, 112, 384, 6, 1, 1};
    case KernelId::HYVLA_GATEUP_RESIDENT_PAIR_W8_V2: return {288, 128, 128, 4, 1, 1};
    case KernelId::HYVLA_RESIDENT_SWIGLU_FP16_V2: return {96, 112, 384, 3, 1, 1};
    case KernelId::HYVLA_W8_RESIDENT_SWIGLU_S51: return {192, 128, 128, 2, 1, 1};
    case KernelId::HYVLA_VLM_GATEUP_SWIGLU_FP16_V2: return {320, 80, 128, 10, 1, 1};
    case KernelId::HYVLA_VLM_GATEUP_SWIGLU_W8_V2: return {288, 80, 128, 10, 1, 1};
    case KernelId::LINGBOT2_PREFILL_GATEUP_SHARED_A_W8_M224: return {240, 96, 128, 13, 1, 1};
    case KernelId::LINGBOT2_PREFILL_QKV_MULTIBASE_FP16_M224: return {384, 96, 128, 10, 1, 1};
    case KernelId::LINGBOT2_PREFILL_QKV_MULTIBASE_W8_M224: return {320, 112, 128, 11, 1, 1};
    case KernelId::LINGBOT2_QKV_MULTIBASE_FP16_T51: return {208, 96, 256, 10, 1, 1};
    case KernelId::LINGBOT2_QKV_MULTIBASE_W8_T51: return {64, 80, 512, 11, 1, 1};
    case KernelId::LINGBOT2_SHARED_GATEUP_RESIDENT_SILU_FP16_T51: return {528, 48, 128, 2, 1, 1};
    case KernelId::LINGBOT2_SHARED_GATEUP_RESIDENT_SILU_W8_T51: return {64, 32, 128, 3, 1, 1};
    case KernelId::LINGBOT2_VISION_FC1_GELU_FP16: return {400, 96, 128, 6, 2, 1};
    case KernelId::LINGBOT2_VISION_FC1_GELU_W8: return {400, 80, 128, 7, 2, 1};
    case KernelId::LINGBOT2_VISION_QKV_JOINT_FP16_M768: return {480, 64, 128, 6, 2, 1};
    case KernelId::LINGBOT2_VISION_QKV_JOINT_W8_M768: return {464, 64, 128, 6, 2, 1};
    default: TORCH_CHECK(false, "Unknown current specialized Linear contract");
  }
}
inline std::vector<uint16_t> grid(KernelId id) {
  const auto c = contract(id); return {c.gx, c.gy, c.gz};
}
}
