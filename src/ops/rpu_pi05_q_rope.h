#pragma once
#include <ATen/ATen.h>
#include <cstdint>

// Exact Pi action Q: W8 TP-column, M50/Nlocal256/K1024, full D256 RoPE.
// Both SPM operands are absolute core0 addresses with disjoint live extents.
void rpu_launch_pi05_q_rope_spm_kernel(
    uint32_t input,const at::Tensor& weight,uint32_t output,
    const at::Tensor& scale,const at::Tensor& cos,const at::Tensor& sin,
    int64_t logical_position);
