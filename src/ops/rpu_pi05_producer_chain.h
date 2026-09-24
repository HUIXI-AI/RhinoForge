#pragma once
#include <ATen/ATen.h>
#include <cstdint>

// Experimental fixed Prefill O chain. No model selector calls this wrapper.
// All SPM addresses are absolute. Normalized may exactly alias residual or
// overlay dead input/partial in either fixed mask-v2 layout after producer
// completion. Raw and gamma must remain disjoint and live.
void rpu_launch_pi05_prefill_o_chain_spm_kernel(
    uint32_t input, const at::Tensor& weight, uint32_t partial,
    uint32_t residual, uint32_t raw, uint32_t normalized,
    uint32_t gamma, const at::Tensor& scale, double eps = 1e-6);

// Fixed M50/N1024 continuation selected only by the Pi0.5 denoise exact-profile
// attention route. All five operands are complete, mutually disjoint,
// 256-byte-aligned absolute core-0 SPM addresses. The zero/skip/gate inputs are
// read-only.
void rpu_launch_pi05_xor3_gated_residual_spm_kernel(
    uint32_t partial, uint32_t zero, uint32_t skip, uint32_t gate,
    uint32_t output);
