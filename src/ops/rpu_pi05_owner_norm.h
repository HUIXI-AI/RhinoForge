#pragma once
#include <cstdint>

enum class Pi05OwnerNormResidual { Full, Compact };

// Exact G05 Text M256/H2048 TP8, original paced reduction and BASE RMSNorm.
// Four disjoint aligned full SPM roots (matrices 1 MiB, gamma 4096 B).
// Normalized output exactly replaces residual after its last global read.
// Raw retains ONLY this core's original global owner32 rows; the following
// same-shape ordinary MLP residual ring consumes those rows, not a full replica.
void rpu_launch_g05_text_owner_rms_m256h2048_kernel(
    uint32_t partial, uint32_t residual, uint32_t raw, uint32_t gamma);

// Exact M272/M288/M304/M320/M336/M400/M416/M432/M448/M464/M480, N2048, eight cores. Gamma has identical FP16 bytes on every
// core, as established by the model's broadcast preload and admission gate.
// All addresses are absolute core-0 SPM addresses and operands are disjoint.
void rpu_launch_pi05_owner_norm_spm_kernel(
    uint32_t partial, uint32_t residual, uint32_t compact_raw,
    uint32_t normalized, uint32_t gamma, Pi05OwnerNormResidual kind,
    double eps, int64_t rows = 400);

void rpu_launch_pi05_compact_residual_spm_kernel(
    uint32_t partial, uint32_t compact_residual, uint32_t full_raw, int64_t rows = 400);

// Experimental compact C272/C288/C304/C320/C336/C400/C416/C432/C448/C464 OwnerNorm: resident full-row quantization, then
// gather A8[rows,2080] including embedded scales and emit FP16 scale[rows].
void rpu_launch_pi05_owner_norm_a8_spm_kernel(
    uint32_t partial, uint32_t residual, uint32_t compact_raw,
    uint32_t a8, uint32_t gamma, uint32_t row_scale, double eps, int64_t rows = 400);

// Exact Vision M512 or M768/N1152, eight cores, owner64/96 rows. All addresses are
// absolute core-0 SPM addresses. Gamma/beta must have identical FP16 bytes
// across cores. The only permitted overlap is partial == normalized: the
// reduce-scatter completes on every core before LayerNorm writes that root.
void rpu_launch_pi05_vision_owner_layernorm_spm_kernel(
    uint32_t partial, uint32_t residual, uint32_t compact_raw,
    uint32_t normalized, uint32_t gamma, uint32_t beta, double eps, int64_t rows = 768);

// Consumes the owner64/96 raw residual retained by the preceding owner LayerNorm
// and returns full M512 or M768/N1152 raw values on all cores. All operands disjoint.
void rpu_launch_pi05_vision_compact_residual_spm_kernel(
    uint32_t partial, uint32_t compact_residual, uint32_t full_raw, int64_t rows = 768);

// Exact G05 K1 Vision M256/N1024, FP16, eight cores, epsilon 1e-6.
// All pointers are absolute core-0 SPM addresses. Full operands are 512 KiB,
// compact_raw is 64 KiB/core, and identical gamma/beta are 2 KiB/core each.
// Only partial == normalized may alias; compact_raw must survive the MLP.
void rpu_launch_g05_vision_owner_reduce_layernorm_m256n1024_kernel(
    uint64_t partial, uint64_t residual, uint64_t compact_raw,
    uint64_t normalized, uint64_t gamma, uint64_t beta);

// Consumes that compact raw residual and produces full FP16 raw output on all
// eight cores. All three intervals must be disjoint and 256-byte aligned.
void rpu_launch_g05_vision_compact_residual_m256n1024_kernel(
    uint64_t partial, uint64_t compact_raw, uint64_t full_output);
