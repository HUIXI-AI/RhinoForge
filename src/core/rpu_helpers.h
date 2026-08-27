// rpu_helpers.h — SDPA tiling, memcpy utilities, zero-copy, hardware constants
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <ATen/ATen.h>
#include "rpu_profile.h"  // CeilDiv
#include "rhino_launch_kernel.h"

// =============================================================================
// SDPA tiling helper. It implements the admitted operator-asset constraints and
// the runtime tile-selection policy.
// =============================================================================
inline int64_t sdpa_tile_m_v16_initial(int64_t seq_q) {
    if (seq_q <= 176) return CeilDiv(seq_q, (int64_t)16);
    return 8;   // operator-asset hard cap (treg)
}

// Backwards-compat alias for the pybind export name (test surface).
inline int64_t sdpa_tile_m_v16(int64_t seq_q) {
    return sdpa_tile_m_v16_initial(seq_q);
}

// tile_k_v16 starts from tile_m_v16 and is reduced by sdpa_compute_tiling when
// required by the VLM budget.

// =============================================================================
// SDPA vctxlen tiling — VLM-based operator contract
// =============================================================================
#define SDPA_MAX_VLM_SIZE 400

struct SdpaTiling {
    int64_t tile_m_v16, tile_n_v16, tile_k_v16;
    int64_t tile_m, tile_n, tile_k;
};

enum class SdpaKernelType {
    FLASH_ATTN_SPM,
    FLASH_ATTN_SPM_VCTXLEN,
};

struct SdpaConfig {
    SdpaKernelType kernel;
    int64_t head_dim;
    int64_t num_q_heads;
    int64_t num_kv_heads;
    int num_cores;
    int attn_mask_type;  // 0=NONE, 1=LTM, 4=2D
};

// Compute the temporary vector-memory requirement for one attention tile.
inline uint32_t GetMhaAttnUnivVlmNeed(
    uint16_t tile_m_v16, uint16_t tile_n_v16, uint16_t tile_k_v16,
    uint16_t hdQryV16, uint16_t hdKeyV16, uint16_t hdValV16,
    uint16_t attn_mask_type)
{
    uint32_t vlm_output = tile_n_v16 * tile_m_v16;
    uint32_t vlm_prjQry = hdQryV16 * tile_m_v16;
    uint32_t vlm_prjKey = hdKeyV16 * tile_k_v16;
    uint32_t vlm_prjValT = hdValV16 * tile_k_v16;
    uint32_t vlm_attn = tile_m_v16 * std::max(tile_k_v16, tile_n_v16);
    uint32_t vlm_softmax_aux = tile_m_v16 * 4 + 4;

    uint32_t vlm_mask = 0;
    if (attn_mask_type == 2 || attn_mask_type == 3 || attn_mask_type == 4) {
        vlm_mask = tile_m_v16 * tile_k_v16;
    } else if (attn_mask_type == 5) {
        vlm_mask = tile_m_v16;
    }

    uint32_t res = std::max({vlm_output, vlm_prjQry, vlm_mask})
                 + std::max(vlm_prjKey, vlm_prjValT)
                 + vlm_attn + vlm_softmax_aux + 4;
    return res;
}

// =============================================================================
// choose_tile_k — pure function family
// Picks a tile_k_v16 satisfying constraints C1–C4, or std::nullopt.
//
//   C1: tile_k_v16 ∈ [1, 16]                         (treg limit)
//   C2: tile_m_v16 * tile_n_v16 * tile_k_v16 ≤ 1024  (total tile size)
//   C3: LTM (attn_mask_type==1) ⇒ tk % tm == 0
//   C4: GetMhaAttnUnivVlmNeed(tm, tn, tk, hd, hd, hd, mask) ≤ SDPA_MAX_VLM_SIZE
//
// Selection rule differs per mode:
//   AGGRESSIVE: largest tk_v16 ∈ [1, 16] satisfying C1–C4 (descending scan).
//   CONSERVATIVE: largest tk in Tier-1 {16,11,8,5,4,2,1} satisfying C1–C4;
//                 falls back to Tier-2 tk = tm_v16 if no Tier-1 candidate
//                 fits. Tier-2 outputs are NOT guaranteed to be in the
//                 admitted set; the fallback preserves tk=tm compatibility but
//                 requires qualification before release.
//                 Conservative deliberately does NOT return "largest"; e.g.
//                 at (tm=3, tn=8, LTM, hd=8) aggressive picks tk=15 while
//                 conservative picks tk=3 (Tier-2 fallback).
//
// nullopt iff no tk≥1 satisfies all 4 — caller sdpa_try_compute_tiling
// must then shrink tm/tn and retry.
//
// Default mode: CONSERVATIVE.
// Aggressive mode (opt-in via -DRPU_SDPA_TILE_K_AGGRESSIVE_MODE) allows any
// tk_v16 ∈ [1, 16]; reserved for explicit qualification sweeps.
// =============================================================================
#ifndef RPU_SDPA_TILE_K_AGGRESSIVE_MODE
#  define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
#endif

// Tile policy version = base * 2 + mode_bit.
//   base: bump on any algorithm change.
//   mode_bit: 0=aggressive, 1=conservative.
// GraphCacheKey embeds kSdpaTilePolicyVersion so cache invalidates on any
// algorithm change or mode flip.
constexpr int kSdpaTilePolicyBase = 2;
#ifdef RPU_SDPA_TILE_K_CONSERVATIVE_MODE
constexpr int kSdpaTilePolicyMode = 1;
#else
constexpr int kSdpaTilePolicyMode = 0;
#endif
constexpr int kSdpaTilePolicyVersion =
    kSdpaTilePolicyBase * 2 + kSdpaTilePolicyMode;

// Full descending scan: largest tk∈[1,16] satisfying C1–C4.
inline std::optional<int64_t> choose_tile_k_aggressive(
        int64_t tm_v16, int64_t tn_v16, int attn_mask_type,
        int64_t head_dim_v16) {
    TORCH_CHECK(tm_v16 >= 1 && tn_v16 >= 1 && head_dim_v16 >= 1,
                "choose_tile_k_aggressive: tm/tn/hd_v16 must be ≥ 1");
    const int64_t tn_tm = tm_v16 * tn_v16;
    if (tn_tm > 1024) return std::nullopt;
    const int64_t tk_cap = std::min<int64_t>(16, 1024 / tn_tm);
    if (tk_cap < 1) return std::nullopt;
    for (int64_t cand = tk_cap; cand >= 1; --cand) {
        if (attn_mask_type == 1 && cand % tm_v16 != 0) continue;
        const uint32_t vlm = GetMhaAttnUnivVlmNeed(
            (uint16_t)tm_v16, (uint16_t)tn_v16, (uint16_t)cand,
            (uint16_t)head_dim_v16, (uint16_t)head_dim_v16, (uint16_t)head_dim_v16,
            (uint16_t)attn_mask_type);
        if (vlm > SDPA_MAX_VLM_SIZE) continue;
        return cand;
    }
    return std::nullopt;
}

// Two-tier conservative scan:
//   Tier 1: {16,11,8,5,4,2,1} (admitted set)
//   Tier 2: tk = tm_v16       (compatibility fallback)
// "Conservative" avoids unqualified tile values; Tier-2 outputs require an
// explicit qualification sweep.
inline std::optional<int64_t> choose_tile_k_conservative(
        int64_t tm_v16, int64_t tn_v16, int attn_mask_type,
        int64_t head_dim_v16) {
    TORCH_CHECK(tm_v16 >= 1 && tn_v16 >= 1 && head_dim_v16 >= 1,
                "choose_tile_k_conservative: tm/tn/hd_v16 must be ≥ 1");
    constexpr std::array<int64_t, 7> kTier1 = {16, 11, 8, 5, 4, 2, 1};
    const int64_t tn_tm = tm_v16 * tn_v16;
    if (tn_tm > 1024) return std::nullopt;
    for (int64_t cand : kTier1) {
        if (tn_tm * cand > 1024) continue;                          // C2
        if (attn_mask_type == 1 && cand % tm_v16 != 0) continue;    // C3
        const uint32_t vlm = GetMhaAttnUnivVlmNeed(
            (uint16_t)tm_v16, (uint16_t)tn_v16, (uint16_t)cand,
            (uint16_t)head_dim_v16, (uint16_t)head_dim_v16, (uint16_t)head_dim_v16,
            (uint16_t)attn_mask_type);
        if (vlm > SDPA_MAX_VLM_SIZE) continue;                      // C4
        return cand;
    }
    // Tier 2 fallback: tk = tm. C3 trivially holds (tm % tm == 0).
    if (tm_v16 > 16) return std::nullopt;                           // C1
    if (tn_tm * tm_v16 > 1024) return std::nullopt;                 // C2
    const uint32_t vlm = GetMhaAttnUnivVlmNeed(
        (uint16_t)tm_v16, (uint16_t)tn_v16, (uint16_t)tm_v16,
        (uint16_t)head_dim_v16, (uint16_t)head_dim_v16, (uint16_t)head_dim_v16,
        (uint16_t)attn_mask_type);
    if (vlm > SDPA_MAX_VLM_SIZE) return std::nullopt;               // C4
    return tm_v16;
}

// Compile-time dispatcher. Default: conservative.
inline std::optional<int64_t> choose_tile_k(
        int64_t tm_v16, int64_t tn_v16, int attn_mask_type,
        int64_t head_dim_v16) {
#ifdef RPU_SDPA_TILE_K_CONSERVATIVE_MODE
    return choose_tile_k_conservative(tm_v16, tn_v16, attn_mask_type, head_dim_v16);
#else
    return choose_tile_k_aggressive(tm_v16, tn_v16, attn_mask_type, head_dim_v16);
#endif
}

// =============================================================================
// Shared utility: float32 → uint32 bit-cast
// =============================================================================
inline uint32_t float32_to_uint32(float f) {
    union { float f; uint32_t u; } conv;
    conv.f = f;
    return conv.u;
}

// SDPA SCM register base offset (shared across all SDPA variants)
#ifndef SCM_REG_OFFSET
#define SCM_REG_OFFSET 4096
#endif

// Launch v1 writes typed 32-bit fields directly. Legacy 16-bit wrapper slots
// use the compatibility mapping expected by the versioned operator asset.
inline void rpu_set_scm_u32_checked(
        ::rhino_lkn::Kernel_t* kernel, uint32_t reg_idx, uint32_t value,
        const char* context) {
    TORCH_CHECK(reg_idx >= static_cast<uint32_t>(SCM_REG_OFFSET) &&
                    ((reg_idx - static_cast<uint32_t>(SCM_REG_OFFSET)) & 1U) == 0,
                context, ": true uint32 SCM register must start at an even SCM "
                "index; got ", reg_idx);
    const uint32_t rc = kernel->set_regs(reg_idx, value);
    TORCH_CHECK(rc == ::rhino_lkn::kKernelRegOk,
                context, ": failed to set uint32 SCM register ", reg_idx,
                " (rc=", rc, ")");
}

inline void rpu_set_legacy_scm_u16_checked(
        ::rhino_lkn::Kernel_t* kernel, uint32_t logical_reg_idx,
        uint16_t value, const char* context) {
    TORCH_CHECK(logical_reg_idx >= static_cast<uint32_t>(SCM_REG_OFFSET),
                context, ": legacy uint16 SCM register is outside SCM; got ",
                logical_reg_idx);
    const uint32_t physical_reg_idx =
        static_cast<uint32_t>(SCM_REG_OFFSET) +
        ((logical_reg_idx - static_cast<uint32_t>(SCM_REG_OFFSET)) ^ 1U);
    const uint32_t rc = kernel->set_regs(physical_reg_idx, value);
    TORCH_CHECK(rc == ::rhino_lkn::kKernelRegOk,
                context, ": failed to set legacy uint16 SCM register ",
                logical_reg_idx, " via physical register ", physical_reg_idx,
                " (rc=", rc, ")");
}

// =============================================================================
// Unified SDPA tiling — outer shrink loop on tm/tn, tk from choose_tile_k.
//
// Replaces the legacy halve-tm→halve-tk→halve-tn loop (whose stale-tk state
// between iterations caused LTM tk%tm violations). choose_tile_k embeds the
// VLM check inline, so the outer loop only needs to halve tm/tn when no tk
// satisfies all 4 constraints at the current (tm, tn).
//
//   sdpa_try_compute_tiling: non-throwing; returns false on infeasibility.
//   sdpa_compute_tiling:     throwing wrapper for launcher-side calls.
//
// Predicate callers (sdpa_call_valid) MUST use the try variant — failing
// to find a tiling is a legitimate "this chunk_size is illegal" signal, not
// an error.
// =============================================================================

inline bool sdpa_try_compute_tiling(const SdpaConfig& cfg, int64_t seq_q,
                                     SdpaTiling* out) {
    if (seq_q <= 0) return false;
    if (cfg.head_dim <= 0 || cfg.head_dim % 16 != 0) return false;

    const int64_t head_dim_v16 = CeilDiv(cfg.head_dim, (int64_t)16);
    int64_t tile_n_v16 = head_dim_v16;
    int64_t tile_m_v16 = sdpa_tile_m_v16_initial(seq_q);
    bool aligned_ltm =
        cfg.attn_mask_type == 1 && seq_q > 176 && seq_q % 16 == 0;
    const int64_t seq_q_v16 = aligned_ltm ? seq_q / 16 : 0;
    if (aligned_ltm) {
        // A later LTM chunk starts at sQryAcc = N * seq_q. The kernel's
        // triangular-mask tile origin must divide that offset; otherwise the
        // first chunk is correct but every later chunk reads the wrong causal
        // region. Pick the largest admitted tile <= the normal 8-v16 cap that
        // divides seq_q.
        while (tile_m_v16 > 1 && seq_q_v16 % tile_m_v16 != 0) {
            --tile_m_v16;
        }
        const int64_t q_heads_per_core =
            cfg.num_cores > 0 ? cfg.num_q_heads / cfg.num_cores : 0;
        const int64_t kv_heads_per_core =
            CeilDiv(cfg.num_kv_heads, (int64_t)std::max(cfg.num_cores, 1));
        const int64_t gqa = q_heads_per_core
            / std::max(kv_heads_per_core, (int64_t)1);
        const int64_t grid = CeilDiv(seq_q, tile_m_v16 * 16);
        const int64_t sync_product = grid * gqa;
        if (sync_product > 8 && sync_product % 8 != 0) {
            // Preserve the established first/single-chunk tiling when the
            // aligned candidate itself violates the keeper's sync constraint
            // A later chunk is still rejected by the
            // explicit sQryAcc % tile_m check, so this fallback cannot silently
            // reintroduce the offset-corruption bug.
            tile_m_v16 = sdpa_tile_m_v16_initial(seq_q);
            aligned_ltm = false;
        }
    }

    constexpr int kMaxShrinkIter = 16;
    std::optional<int64_t> tk_opt;
    for (int it = 0; it < kMaxShrinkIter; ++it) {
        tk_opt = choose_tile_k(tile_m_v16, tile_n_v16,
                               cfg.attn_mask_type, head_dim_v16);
        if (tk_opt) break;
        // No valid tk at (tm, tn). Shrink tm first, then tn.
        if (tile_m_v16 > 1) {
            if (aligned_ltm) {
                do {
                    --tile_m_v16;
                } while (tile_m_v16 > 1
                         && seq_q_v16 % tile_m_v16 != 0);
            } else {
                tile_m_v16 >>= 1;
            }
        } else if (tile_n_v16 > 1) {
            tile_n_v16 >>= 1;
        } else {
            return false;  // tm=1, tn=1, still no tk — give up
        }
    }
    if (!tk_opt) return false;  // exhausted shrink iterations

    const int64_t tile_k_v16 = *tk_opt;

    // Defensive post-conditions — should hold given choose_tile_k passed.
    if (tile_k_v16 < 1 || tile_k_v16 > 16) return false;
    if (tile_m_v16 * tile_n_v16 * tile_k_v16 > 1024) return false;
    if (cfg.attn_mask_type == 1 && tile_k_v16 % tile_m_v16 != 0) return false;

    *out = SdpaTiling{
        tile_m_v16, tile_n_v16, tile_k_v16,
        tile_m_v16 * 16, tile_n_v16 * 16, tile_k_v16 * 16,
    };
    return true;
}

// Throwing wrapper. The stable "cannot shrink to fit VLM" substring is part of
// the validation error contract.
inline SdpaTiling sdpa_compute_tiling(const SdpaConfig& cfg, int64_t seq_q) {
    SdpaTiling t;
    TORCH_CHECK(sdpa_try_compute_tiling(cfg, seq_q, &t),
                "sdpa_compute_tiling: cannot shrink to fit VLM/treg/LTM "
                "constraints (seq_q=", seq_q, ", head_dim=", cfg.head_dim,
                ", attn_mask_type=", cfg.attn_mask_type, ")");
    return t;
}

// Unified tmp workspace size in v16 units.
inline int64_t sdpa_compute_tmp_v16_size(const SdpaConfig& cfg, int64_t seq_q) {
    SdpaTiling t = sdpa_compute_tiling(cfg, seq_q);
    int64_t nkv_head_per_core = CeilDiv(cfg.num_kv_heads, (int64_t)std::max(cfg.num_cores, 1));
    int64_t grid_dim_x = CeilDiv(seq_q, t.tile_m);
    return t.tile_n_v16 * t.tile_k * nkv_head_per_core * grid_dim_x;
}

// Per-call operator-asset constraint check — called by sdpa_is_valid_chunk_size
// for each kernel invocation in the chunked-prefill plan.
//
// Uses sdpa_try_compute_tiling (non-throwing). An infeasible tiling
// means "this chunk_size is illegal" — return false, do not throw.
static inline bool sdpa_call_valid(const SdpaConfig& cfg,
                                   int64_t sQry, int64_t sQryAcc) {
    if (sQry <= 0) return false;
    SdpaConfig call_cfg = cfg;
    if (sQry == 1 && call_cfg.attn_mask_type == 1) {
        // The fused launcher dispatches a single-token tail as MASK_NONE.
        call_cfg.attn_mask_type = 0;
    }
    SdpaTiling t;
    if (!sdpa_try_compute_tiling(call_cfg, sQry, &t)) return false;

    // Tile-level constraints.
    // (Old `tile_m_v16 > 8` and `tile_k_v16 != 8` checks removed —
    //  the admitted policy allows tile_m∈[1,11] for sQry∈[16,176] and dynamic
    //  tile_k. treg limit `tile_k_v16 <= 16` is enforced in
    //  sdpa_compute_tiling itself.)
    if (t.tile_m_v16 * t.tile_n_v16 * t.tile_k_v16 > 1024) return false;

    // VLM capacity
    uint32_t vlm = GetMhaAttnUnivVlmNeed(
        (uint16_t)t.tile_m_v16, (uint16_t)t.tile_n_v16, (uint16_t)t.tile_k_v16,
        (uint16_t)t.tile_n_v16, (uint16_t)t.tile_n_v16, (uint16_t)t.tile_n_v16,
        (uint16_t)call_cfg.attn_mask_type);
    if (vlm > SDPA_MAX_VLM_SIZE) return false;

    // Grid + sync: the admitted operator contract limits syncGroupSize to 8.
    // Larger products are not numerically qualified even if accepted by a
    // lower-level build check.
    int64_t grid_dim_x = CeilDiv(sQry, t.tile_m);
    int64_t num_heads_per_core = call_cfg.num_q_heads / call_cfg.num_cores;
    int64_t nkv_head_per_core = CeilDiv(call_cfg.num_kv_heads,
                                        (int64_t)std::max(call_cfg.num_cores, 1));
    int64_t gqa_group_size = num_heads_per_core / std::max(nkv_head_per_core, (int64_t)1);
    int64_t product = grid_dim_x * gqa_group_size;
    if (product > 8) return false;
    // Accumulated LTM with sync product 6 is not an admitted board profile.
    if (cfg.attn_mask_type == 1 && sQryAcc != 0 && product == 6) return false;

    // LTM per-call contract: tile_k_v16 % tile_m_v16 == 0.
    // sQryAcc % sQry: enforced (see sdpa_validate_kernel_call for runtime-divergence note)
    //
    // Single-token decode (sQry==1) is mask-agnostic: every causal LM fused
    // path launches that call with MASK_NONE (sdpa_causal = is_causal &&
    // seq_len > 1 in build_layer_subgraph), so the LTM-specific tile and
    // sQryAcc-alignment constraints don't apply to it. Skip them here so the
    // chunked-plan validator agrees with the keeper for arbitrary decode
    // positions (e.g. position=4097 → sQryAcc%16 != 0).
    if (call_cfg.attn_mask_type == 1 && sQry > 1) {
        if (t.tile_k_v16 % t.tile_m_v16 != 0) return false;
        if (sQryAcc != 0) {
            if (sQryAcc % 16 != 0) return false;
            if (sQryAcc % t.tile_m != 0) return false;
            if (sQryAcc % sQry != 0) return false;
        }
    }
    return true;
}

inline bool sdpa_is_valid_chunk_size(const SdpaConfig& cfg, int64_t cs,
                                     int64_t seq_len, int64_t position) {
    if (cs < 16 || cs % 16 != 0) return false;
    if (seq_len <= 0 || position < 0) return false;

    // Structural operator constraints.
    if (cfg.num_cores <= 0 || cfg.num_q_heads <= 0 || cfg.num_kv_heads <= 0)
        return false;
    if (cfg.num_q_heads % cfg.num_kv_heads != 0) return false;
    if (cfg.num_q_heads % cfg.num_cores != 0) return false;
    if ((cfg.num_kv_heads % cfg.num_cores != 0) &&
        (cfg.num_cores % cfg.num_kv_heads != 0)) return false;
    if (cfg.head_dim <= 0 || cfg.head_dim % 16 != 0) return false;

    // Simulate chunked-prefill call sequence:
    // [off=0..cs], ..., [off=(n-1)*cs..n*cs], [off=n*cs..seq_len]
    // full call i: sQry=cs, sQryAcc=position+i*cs
    // tail call:   sQry=rem, sQryAcc=position+seq_len-rem
    int64_t n_full = seq_len / cs;
    int64_t rem = seq_len % cs;

    // Two representative full calls — both are needed:
    //   - (cs, position): exercises tile/VLM/sync (always); LTM acc check only
    //     fires if sQryAcc=position != 0
    //   - (cs, position + cs): exercises LTM acc check at sQryAcc=cs (essential
    //     for the position=0 prefill case, where the i=0 sample skips LTM acc)
    // For all other i in [0, n_full), the constraints reduce to one of these
    // two cases mod cs and mod 16 (since cs % 16 == 0).
    if (n_full > 0) {
        if (!sdpa_call_valid(cfg, cs, position)) return false;
        if (n_full > 1 && !sdpa_call_valid(cfg, cs, position + cs)) return false;
    }

    if (rem != 0) {
        int64_t tail_sQryAcc = position + seq_len - rem;
        if (!sdpa_call_valid(cfg, rem, tail_sQryAcc)) return false;
        if (n_full > 0) {
            SdpaConfig tail_cfg = cfg;
            if (rem == 1 && tail_cfg.attn_mask_type == 1) {
                tail_cfg.attn_mask_type = 0;
            }
            const int64_t full_tmp = sdpa_compute_tmp_v16_size(cfg, cs);
            const int64_t tail_tmp = sdpa_compute_tmp_v16_size(tail_cfg, rem);
            if (tail_tmp > full_tmp) return false;
        }
    }

    return true;
}

// Keeper-side runtime validator. Called at each keeper entry before set_regs.
// Subset of operator-asset constraints derivable from keeper actuals;
// batch / Q/K/V head_dim split / sKey-sVal are caller invariants.
inline void sdpa_validate_kernel_call(int64_t seq_q, int64_t seq_k,
                                      int64_t tile_m_v16, int64_t tile_n_v16, int64_t tile_k_v16,
                                      int64_t tile_m,
                                      int64_t grid_dim_x, int64_t gqa_group_size,
                                      int64_t hd_q_v16, int64_t hd_k_v16, int64_t hd_v_v16,
                                      int attn_mask_type) {
    TORCH_CHECK(seq_q > 0, "SDPA seq_q must be > 0, got ", seq_q);
    TORCH_CHECK(seq_k >= seq_q, "SDPA seq_k(", seq_k, ") must be >= seq_q(", seq_q, ")");
    TORCH_CHECK(tile_m_v16 > 0 && tile_n_v16 > 0 && tile_m > 0,
        "SDPA tile_m_v16(", tile_m_v16, "), tile_n_v16(", tile_n_v16,
        "), tile_m(", tile_m, ") must be > 0");
    TORCH_CHECK(grid_dim_x > 0 && gqa_group_size > 0,
        "SDPA grid_dim_x(", grid_dim_x, "), gqa(", gqa_group_size, ") must be > 0");
    TORCH_CHECK(tile_k_v16 >= 1 && tile_k_v16 <= 16,
                "SDPA tile_k_v16(", tile_k_v16, ") must be in [1, 16] (treg limit)");
    TORCH_CHECK(tile_m_v16 >= 1 && tile_m_v16 <= 11,
                "SDPA tile_m_v16(", tile_m_v16, ") must be in [1, 11] "
                "(admitted boundary: ceil(176/16)=11)");

    int64_t total_tile = tile_m_v16 * tile_n_v16 * tile_k_v16;
    TORCH_CHECK(total_tile <= 1024,
        "SDPA total_tile_size(", total_tile, ") = tile_m_v16(", tile_m_v16,
        ") * tile_n_v16(", tile_n_v16, ") * tile_k_v16(", tile_k_v16, ") must be <= 1024");

    uint32_t vlm = GetMhaAttnUnivVlmNeed(
        (uint16_t)tile_m_v16, (uint16_t)tile_n_v16, (uint16_t)tile_k_v16,
        (uint16_t)hd_q_v16, (uint16_t)hd_k_v16, (uint16_t)hd_v_v16,
        (uint16_t)attn_mask_type);
    TORCH_CHECK(vlm <= SDPA_MAX_VLM_SIZE,
        "SDPA VLM need(", vlm, ") exceeds budget(", SDPA_MAX_VLM_SIZE, ")");

    // RPU_CHUNK_FORCE_UNSAFE=1 also lifts this launcher-side gate, so a chunk
    // size the planner rejected can still be TIMED end-to-end. Diagnostic only:
    // syncGroupSize > 8 is an empirical cmodel/HW limit whose violation yields
    // wrong numbers (and some products are not admitted by the operator asset),
    // so a forced run MUST be checked against a legal chunk
    // size before drawing any conclusion from it.
    static const bool sync_force_unsafe = [] {
        const char* e = std::getenv("RPU_CHUNK_FORCE_UNSAFE");
        return e && (std::string(e) == "1" || std::string(e) == "true");
    }();
    int64_t product = grid_dim_x * gqa_group_size;
    TORCH_CHECK(product <= 8 || sync_force_unsafe,
        "SDPA sync constraint: grid_dim_x(", grid_dim_x, ") * gqa(", gqa_group_size,
        ") = ", product, " must be <= 8 (seq_q=", seq_q, ").");

    if (attn_mask_type == 1) {
        TORCH_CHECK(tile_k_v16 % tile_m_v16 == 0,
            "LTM requires tile_k_v16(", tile_k_v16, ") % tile_m_v16(", tile_m_v16,
            ") == 0. seq_q=", seq_q);
        int64_t seq_q_acc = seq_k - seq_q;
        if (seq_q_acc != 0) {
            TORCH_CHECK(product != 6,
                "SDPA accumulated LTM sync product=6 is empirically broken on HW "
                "(seq_q=", seq_q, ", seq_q_acc=", seq_q_acc, ")");
            TORCH_CHECK(seq_q_acc % 16 == 0,
                "LTM sQryAcc(", seq_q_acc, ") must be 0 or v16-aligned");
            TORCH_CHECK(seq_q_acc % tile_m == 0,
                "LTM sQryAcc(", seq_q_acc, ") must be aligned to tile_m(",
                tile_m, ")");
            TORCH_CHECK(seq_q_acc % seq_q == 0,
                "LTM sQryAcc(", seq_q_acc, ") must be multiple of seq_q(", seq_q,
                "). NOTE: runtime _dp doesn't enforce this (TODO at line 1724-1726). "
                "We do — masks misalign without it.");
        }
    }
}

// 清零 kernel 的常用寄存器
inline void clear_kernel_regs(::rhino_lkn::Kernel_t* kernel) {
    for (uint32_t i = 0; i <= 66; i++) {
        kernel->set_regs(i, (uint16_t)0);
    }
}

// =============================================================================
// High-performance memcpy for SPM (uncached) <-> DDR (cached) data transfer
// =============================================================================

inline void spm_to_ddr(void* __restrict__ ddr_dst,
                       const void* __restrict__ spm_src,
                       size_t bytes) {
    memcpy(ddr_dst, spm_src, bytes);
    __sync_synchronize();
}

inline void ddr_to_spm(void* __restrict__ spm_dst,
                       const void* __restrict__ ddr_src,
                       size_t bytes) {
    memcpy(spm_dst, ddr_src, bytes);
    __sync_synchronize();
}

// =============================================================================
// Zero-Copy Device Conversion
// =============================================================================
at::Tensor rpu_to_cpu_zerocopy(const at::Tensor &rpu_tensor, bool flush = true);
// HostCallback already owns Graph ordering and boundary flushes.  Its CPU
// redispatch must not recursively sync the HostCallback node currently being
// recorded/executed.
at::Tensor rpu_to_cpu_zerocopy_no_graph_sync(
    const at::Tensor &rpu_tensor, bool flush = true);
at::Tensor cpu_to_rpu_zerocopy(const at::Tensor &cpu_tensor);

// =============================================================================
// Hardware Constants
// =============================================================================
#define NUM_THD_PER_WARP 16
#define NUM_WARP_PER_LAUNCH 8
#define NUM_THD_PER_LAUNCH 128
#define MAX_FP16_VLM_V16_PER_THD 24
#define V16_NUM 16

constexpr int UNIFIED_NUM_CORES = 8;
