#pragma once
#include <ATen/ATen.h>
#include <cstdint>

// FP16/W8 ACC16 Down: two exact C272/C288/C304/C320/C400/C416/C432/C448 inputs share each N80/K128 weight
// tile. All four absolute SPM intervals must be disjoint, including inputs
// versus outputs: N blocks may retire before other blocks finish reading X.
void rpu_launch_pi05_down_pair_spm_kernel(
    uint32_t input0, uint32_t input1, const at::Tensor& weight,
    uint32_t output0, uint32_t output1, const at::Tensor& scale, int64_t rows = 400);
