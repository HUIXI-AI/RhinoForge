#pragma once

#include "core/fused_model_base.h"
#include "core/rpu_spm_pipeline.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>

namespace v3::wall_oss_z1_internal {

SpmPipelineComponentLayout prepare_vision(int64_t handle,
                                          int64_t num_patches);
void prime_vision_inputs(int64_t handle,
                         const at::Tensor& window_mask,
                         int64_t num_patches);
void rollback_vision_inputs(int64_t handle);
void unprepare_vision(int64_t handle);
SpmDense2DSpec vision_source_spec(int64_t handle,
                                  int64_t num_patches);
void adopt_vision(int64_t handle,
                  const SpmPipelineLease& lease,
                  const SpmTensorView& scratch);
void bind_vision_source(int64_t handle,
                        const SpmPipelineLease& lease,
                        const SpmPortView& source);
void validate_vision(int64_t handle,
                     const SpmPipelineLease& lease);
void clear_vision(int64_t handle, uint64_t epoch, uint64_t plan_hash);
void forward_vision_z1(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches,
    at::TensorList v_caches,
    int64_t num_patches,
    std::optional<at::Tensor> window_mask,
    uint64_t epoch,
    uint64_t plan_hash);

SpmPipelineComponentLayout prepare_text(int64_t handle,
                                        int64_t execution_len,
                                        int64_t real_len);
SpmDense2DSpec text_destination_spec(int64_t handle,
                                     int64_t execution_len);
void prime_text_inputs(int64_t handle,
                       const at::Tensor& position_ids,
                       const at::Tensor& rope_cos_il,
                       const at::Tensor& rope_sin_il,
                       int64_t execution_len);
void rollback_text_inputs(int64_t handle);
void unprepare_text(int64_t handle);
void adopt_text(int64_t handle,
                const SpmPipelineLease& lease,
                const SpmTensorView& scratch);
void bind_text_destination(int64_t handle,
                           const SpmPipelineLease& lease,
                           const SpmPortView& destination);
void validate_text(int64_t handle,
                   const SpmPipelineLease& lease);
void clear_text(int64_t handle, uint64_t epoch, uint64_t plan_hash);
at::Tensor forward_text_z1(
    int64_t handle,
    const at::Tensor& hidden_shape_carrier,
    at::TensorList k_caches,
    at::TensorList v_caches,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    uint64_t epoch,
    uint64_t plan_hash);

}  // namespace v3::wall_oss_z1_internal

int64_t rpu_wall_oss_spm_z1_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len);
void rpu_wall_oss_spm_z1_unprepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t plan_hash);
int64_t rpu_wall_oss_spm_z1_begin(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len,
    const at::Tensor& window_mask,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t plan_hash);
void rpu_wall_oss_spm_z1_end(int64_t epoch, int64_t plan_hash);
at::Tensor rpu_wall_oss_spm_z1_pipeline_forward(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& vision_input,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& window_mask,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    const at::Tensor& rope_cos_il,
    const at::Tensor& rope_sin_il,
    int64_t epoch,
    int64_t plan_hash);
at::Tensor rpu_wall_oss_spm_z1_export_destination(
    int64_t epoch, int64_t plan_hash);
