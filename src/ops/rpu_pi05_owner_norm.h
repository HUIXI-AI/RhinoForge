#pragma once
#include <cstdint>

enum class Pi05OwnerNormResidual { Full, Compact };

// Exact M272/M288/M304/M320/M400/M416/M432/M448, N2048, eight cores. Gamma has identical FP16 bytes on every
// core, as established by the model's broadcast preload and admission gate.
// All addresses are absolute core-0 SPM addresses and operands are disjoint.
void rpu_launch_pi05_owner_norm_spm_kernel(
    uint32_t partial, uint32_t residual, uint32_t compact_raw,
    uint32_t normalized, uint32_t gamma, Pi05OwnerNormResidual kind,
    double eps, int64_t rows = 400);

void rpu_launch_pi05_compact_residual_spm_kernel(
    uint32_t partial, uint32_t compact_residual, uint32_t full_raw, int64_t rows = 400);

// Experimental compact C272/C288/C304/C320/C400/C416/C432/C448 OwnerNorm: resident full-row quantization, then
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
