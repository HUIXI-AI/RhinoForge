#pragma once

#include <cstdint>

namespace v3 {
class SpmFmbPostFnYieldTarget;
}

// Compatibility aliases for LingBot2's sealed composite profiles. The
// implementation delegates to the same ring launcher used
// by the public residual all-reduce admission path.
void rpu_launch_lingbot2_ring_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    uint32_t output_spm,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores);

void rpu_launch_lingbot2_ring_all_reduce_sum_residual_kernel(
    uint32_t input_spm,
    uint32_t residual_spm,
    const v3::SpmFmbPostFnYieldTarget& output,
    int64_t M,
    int64_t N,
    int input_num_cores,
    int output_num_cores);
