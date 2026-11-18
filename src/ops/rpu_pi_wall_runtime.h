#pragma once
#include "rpu_kernel_cache.h"

// A cold capability snapshot covers every generated shape for the operation;
// a later chunk-size choice cannot silently pick an unavailable payload.
inline bool rpu_pi_wall_operation_loaded(rpu_pi_wall::Operation operation,
                                         rpu_pl_tiling::WeightType weight) {
    bool found = false;
    for (const auto& p : rpu_pi_wall::profiles) {
        if (p.operation != operation || p.weight != weight) continue;
        found = true;
        if (!KernelCache::instance().has_loaded(pi_wall_kernel_id(p.ordinal)))
            return false;
    }
    return found;
}
