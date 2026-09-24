#pragma once
#include <ATen/ATen.h>
#include <cstdint>

enum class Pi05PrefillPairedProjection { QkvColumn, AttentionOutputRow };

// Two exact C272/C288/C304/C320/C400/C416/C432/C448 FP16 or W8A16 projections sharing one weight pass. Q/K/V are separate calls
// with separate W tensors; this interface does not fuse their different W.
void rpu_launch_pi05_prefill_pair_spm_kernel(
    Pi05PrefillPairedProjection kind, uint32_t input0, uint32_t input1,
    const at::Tensor& weight, uint32_t output0, uint32_t output1,
    const at::Tensor& scale, int64_t rows = 400);
