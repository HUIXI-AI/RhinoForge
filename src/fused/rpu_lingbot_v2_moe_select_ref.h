// rpu_lingbot_v2_moe_select_ref.h — host reference for LingBot2 router selection.
// Header-only C++ with no Torch or accelerator dependencies.
//
// FP32 semantics: logits -> sigmoid -> add correction bias for selection only
// -> strict top-k -> gather unbiased scores -> normalize -> routed_scaling.
// Exactly k entries are selected. Strict > scanning ascending indices makes
// the lowest index win exact ties; references must use that same tie rule.
//
// The device path follows this selection contract after an FP16 storage boundary.
// Near-boundary values that collapse in FP16 can therefore differ from this
// FP32 reference. See rpu_lingbot_v2_moe_model.cpp for the device path.
#pragma once

#include <cstdint>
#include <cmath>

namespace lingbot_v2_moe_ref {

// Strict top-k router select, fp32 throughout.
//   logits      [E]  fp32 router logits (NOT sigmoided)
//   corr_bias   [E]  fp32 correction bias; affects SELECTION ONLY. May be null
//                    (treated as all-zero).
//   out_idx     [k]  selected expert indices, ordered by descending
//                    scores_for_choice, ties by ascending index
//   out_w       [k]  routing weights: gathered from UNBIASED sigmoid scores,
//                    normalised, then scaled by routed_scaling
// Returns the number selected — ALWAYS exactly k (this is the invariant the
// forbidden `>= threshold` mask violated: it returned >k on ties).
inline int router_select(const float* logits, const float* corr_bias,
                         int E, int k, float routed_scaling,
                         int32_t* out_idx, float* out_w) {
    constexpr int kMaxE = 256;
    if (E <= 0 || E > kMaxE || k <= 0 || k > E) return 0;

    float scores[kMaxE];   // UNBIASED sigmoid scores — the weight source
    float choice[kMaxE];   // scores + corr_bias — the SELECTION key
    bool  taken[kMaxE];
    for (int e = 0; e < E; ++e) {
        // sigmoid in fp32. Branch on sign for numerical stability: expf(+large)
        // overflows to inf, so fold the large-magnitude case onto expf(-|x|).
        const float x = logits[e];
        const float s = (x >= 0.0f) ? 1.0f / (1.0f + std::exp(-x))
                                    : std::exp(x) / (1.0f + std::exp(x));
        scores[e] = s;
        choice[e] = s + (corr_bias ? corr_bias[e] : 0.0f);
        taken[e]  = false;
    }

    // STRICT top-k: k passes of argmax over the not-yet-taken set. The scan runs
    // ascending index and uses a STRICT `>`, so on an exact tie the first (=
    // lowest) index is kept — the documented tie rule. Exactly k are selected by
    // construction, for any input, including all-equal.
    for (int j = 0; j < k; ++j) {
        int best = -1;
        for (int e = 0; e < E; ++e) {
            if (taken[e]) continue;
            if (best < 0 || choice[e] > choice[best]) best = e;
        }
        taken[best]  = true;
        out_idx[j]   = static_cast<int32_t>(best);
        out_w[j]     = scores[best];   // UNBIASED score, per the reference
    }

    // norm_topk_prob, then routed_scaling_factor.
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) sum += out_w[j];
    const float denom = sum + 1e-20f;
    for (int j = 0; j < k; ++j) out_w[j] = (out_w[j] / denom) * routed_scaling;
    return k;
}

}  // namespace lingbot_v2_moe_ref
