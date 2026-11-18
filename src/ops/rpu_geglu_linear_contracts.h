// Generated from rhino-ops linear_20261021/contracts.json.
#pragma once
#include "rpu_kernel_cache.h"
namespace rpu_geglu_linear {
inline KernelId select(rpu_pl_tiling::WeightType dtype, bool acc32, int group_size = 0) {
  if (dtype == rpu_pl_tiling::WeightType::Fp16 && acc32 == false && (group_size == 0)) return KernelId::FUSED_GEGLU_W16A16_ACC16_M80N96K512;
  if (dtype == rpu_pl_tiling::WeightType::Fp16 && acc32 == true && (group_size == 0)) return KernelId::FUSED_GEGLU_W16A16_ACC32_M64N96K512;
  if (dtype == rpu_pl_tiling::WeightType::Int4 && acc32 == false && (group_size == 128)) return KernelId::FUSED_GEGLU_W4A16_PGRP_ACC16_M464N64K128;
  if (dtype == rpu_pl_tiling::WeightType::Int4 && acc32 == false && (group_size == 32 || group_size == 64)) return KernelId::FUSED_GEGLU_W4A16_PGRP_ACC16_M736N64K64;
  if (dtype == rpu_pl_tiling::WeightType::Int4 && acc32 == true && (group_size == 128)) return KernelId::FUSED_GEGLU_W4A16_PGRP_ACC32_M352N64K128;
  if (dtype == rpu_pl_tiling::WeightType::Int4 && acc32 == true && (group_size == 32 || group_size == 64)) return KernelId::FUSED_GEGLU_W4A16_PGRP_ACC32_M496N64K64;
  if (dtype == rpu_pl_tiling::WeightType::Int8 && acc32 == false && (group_size == 0)) return KernelId::FUSED_GEGLU_W8A16_ACC16_M64N80K512;
  if (dtype == rpu_pl_tiling::WeightType::Int8 && acc32 == true && (group_size == 0)) return KernelId::FUSED_GEGLU_W8A16_ACC32_M96N80K384;
  TORCH_CHECK(false, "Unsupported current fused GeGLU precision/group profile");
}
inline std::vector<uint16_t> grid(KernelId id) {
  switch (id) {
    case KernelId::FUSED_GEGLU_W16A16_ACC16_M80N96K512: return {6,1,1};
    case KernelId::FUSED_GEGLU_W16A16_ACC32_M64N96K512: return {6,1,1};
    case KernelId::FUSED_GEGLU_W4A16_PGRP_ACC16_M464N64K128: return {8,1,1};
    case KernelId::FUSED_GEGLU_W4A16_PGRP_ACC16_M736N64K64: return {8,1,1};
    case KernelId::FUSED_GEGLU_W4A16_PGRP_ACC32_M352N64K128: return {8,1,1};
    case KernelId::FUSED_GEGLU_W4A16_PGRP_ACC32_M496N64K64: return {8,1,1};
    case KernelId::FUSED_GEGLU_W8A16_ACC16_M64N80K512: return {7,1,1};
    case KernelId::FUSED_GEGLU_W8A16_ACC32_M96N80K384: return {7,1,1};
    default: TORCH_CHECK(false, "Unknown current fused GeGLU kernel");
  }
}
inline rpu_pl_tiling::TilePick tile(KernelId id) {
  switch (id) {
    case KernelId::FUSED_GEGLU_W16A16_ACC16_M80N96K512: return {96,80,512};
    case KernelId::FUSED_GEGLU_W16A16_ACC32_M64N96K512: return {96,64,512};
    case KernelId::FUSED_GEGLU_W4A16_PGRP_ACC16_M464N64K128: return {64,464,128};
    case KernelId::FUSED_GEGLU_W4A16_PGRP_ACC16_M736N64K64: return {64,736,64};
    case KernelId::FUSED_GEGLU_W4A16_PGRP_ACC32_M352N64K128: return {64,352,128};
    case KernelId::FUSED_GEGLU_W4A16_PGRP_ACC32_M496N64K64: return {64,496,64};
    case KernelId::FUSED_GEGLU_W8A16_ACC16_M64N80K512: return {80,64,512};
    case KernelId::FUSED_GEGLU_W8A16_ACC32_M96N80K384: return {80,96,384};
    default: TORCH_CHECK(false, "Unknown current fused GeGLU tile");
  }
}
}
