#pragma once

#include "core/fused_model_base.h"
#include "core/rpu_spm_pipeline.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>

namespace v3::qwen3_5_z2_internal {

SpmPipelineComponentLayout prepare_vision(int64_t handle,
                                          int64_t num_patches);
SpmPipelineComponentLayout prepare_text(int64_t handle,
                                        int64_t execution_len,
                                        int64_t real_len);
SpmDense2DSpec vision_produced_spec(int64_t handle,
                                    int64_t num_patches);
SpmDense2DSpec text_storage_spec(int64_t handle,
                                 int64_t execution_len);

void adopt_vision(int64_t handle,
                  const SpmPipelineLease& lease,
                  const SpmTensorView& scratch);
void adopt_text(int64_t handle,
                const SpmPipelineLease& lease,
                const SpmTensorView& scratch);
void bind_vision_slice(int64_t handle,
                       const SpmPipelineLease& lease,
                       const SpmPortView& slice);
void bind_text(int64_t handle,
               const SpmPipelineLease& lease,
               const SpmPortView& storage);
void validate_vision(int64_t handle,
                     const SpmPipelineLease& lease);
void validate_text(int64_t handle,
                   const SpmPipelineLease& lease);
void clear_vision(int64_t handle, uint64_t epoch, uint64_t plan_hash);
void clear_text(int64_t handle, uint64_t epoch, uint64_t plan_hash);

void forward_vision_z2(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches,
    at::TensorList v_caches,
    int64_t num_patches,
    const std::optional<at::Tensor>& step0_pos,
    uint64_t epoch,
    uint64_t plan_hash);
at::Tensor forward_text_z2(
    int64_t handle,
    const at::Tensor& hidden,
    at::TensorList k_caches,
    at::TensorList v_caches,
    at::TensorList gdn_states,
    at::TensorList conv_states,
    uint64_t epoch,
    uint64_t plan_hash);

void validate_vision_dispatch(int64_t handle,
                              uint64_t epoch,
                              uint64_t plan_hash);
void validate_text_dispatch(int64_t handle,
                            uint64_t epoch,
                            uint64_t plan_hash);
void check_vision_destroy_allowed(int64_t handle);
void check_text_destroy_allowed(int64_t handle);

}  // namespace v3::qwen3_5_z2_internal
