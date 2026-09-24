#pragma once
#include <ATen/ATen.h>
#include <cstdint>

enum class Pi05VisionFcProjection { Fc1Column, Fc2Row };

// Exact M768, TP8, original FP16 bias already in SPM. Fc1 uses each core's
// 544-channel slice; Fc2 uses 1152 true bias values on core0 and zeros on 1..7.
// All SPM addresses are absolute core0 addresses, with disjoint X/Y/bias.
void rpu_launch_pi05_vision_fc_weight_outer_spm_kernel(
    Pi05VisionFcProjection role, uint32_t input_spm, const at::Tensor& weight,
    uint32_t output_spm, uint32_t bias_spm, const at::Tensor& scale);
