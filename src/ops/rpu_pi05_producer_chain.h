#pragma once
#include <ATen/ATen.h>
#include <cstdint>

// Fixed M50 or M10 (H10) /N1024 continuation selected only by the Pi0.5
// denoise exact-profile attention route. All five operands are complete,
// mutually disjoint, 256-byte-aligned absolute core-0 SPM addresses. The
// zero/skip/gate inputs are read-only.
void rpu_launch_pi05_xor3_gated_residual_spm_kernel(
    uint32_t partial, uint32_t zero, uint32_t skip, uint32_t gate,
    uint32_t output, int64_t rows);

// Optional fusion of the continuation above and the Pi0.5 AdaRMS norm/shift
// into one launch, for exact M50/N1024. The shard is whole rows, so each row
// is normalized once by the core that produced it, before the one all-gather
// publishes the normalized output; the raw residual is published compactly,
// each core storing only its own shard rows. All eight operands are complete,
// mutually disjoint, 256-byte-aligned absolute core-0 SPM addresses, and the
// zero/skip/gate/scale/shift inputs are read-only.
void rpu_launch_pi05_xor3_gated_adarms_spm_kernel(
    uint32_t partial, uint32_t zero, uint32_t skip, uint32_t gate,
    uint32_t raw, uint32_t norm, uint32_t scale, uint32_t shift,
    int64_t rows, double eps);

// The continuation above without the norm, over the same whole-row shards, so
// a chain of fused launches can end in a complete raw residual that a
// standalone norm reads. Both whole-row kernels sum every element's partials
// in the flat-shard kernel's order, so this output is byte-identical to
// rpu_launch_pi05_xor3_gated_residual_spm_kernel's at M50.
void rpu_launch_pi05_xor3_gated_rows_spm_kernel(
    uint32_t partial, uint32_t zero, uint32_t skip, uint32_t gate,
    uint32_t raw, int64_t rows);
