#pragma once
#include "rpu_kernel_cache.h"
#include "rpu_linear_tiling.h"
#include "rpu_pi_wall_tiling.h"

namespace rpu_pi_kv_linear {
inline rpu_pl_tiling::TilePick stripe_tile() {
    return rpu_pl_tiling::select_tile(50, 32, 1024,
        rpu_pl_tiling::WeightType::Int8, false);
}
inline KernelId stripe_id() {
    const auto c = stripe_tile();
    return parallel_linear_kernel_id(rpu_pl_tiling::WeightType::Int8, false, c);
}
inline rpu_pi_wall::Selection pair(bool fp16) {
    return rpu_pi_wall::select(rpu_pi_wall::Operation::KvOwner,
        fp16 ? rpu_pl_tiling::WeightType::Fp16 : rpu_pl_tiling::WeightType::Int8,
        false, 50, 32, 1024);
}
}
