#pragma once

#include "core/fused_model_base.h"
#include "core/rpu_spm_pipeline.h"

#include <cstdint>

namespace v3::gr00t_z2_internal {

// The coordinator deliberately sees model handles only.  Concrete model types
// remain private to their translation units, and Python never receives an SPM
// address or offset.
SpmPipelineComponentLayout prepare_vl_encoder(int64_t handle, int64_t seq_len);
SpmPipelineComponentLayout prepare_denoise(int64_t handle, int64_t seq_len);
void unprepare_vl_encoder(int64_t handle, int64_t seq_len);
void unprepare_denoise(int64_t handle, int64_t seq_len);
SpmDense2DSpec vl_encoder_port_spec(int64_t handle, int64_t seq_len);
SpmDense2DSpec denoise_port_spec(int64_t handle, int64_t seq_len);
void stage_vl_outer_fast_component(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches);
void bind_vl_outer_fast_input(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& raw);
void stage_denoise_outer_fast_component(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches);
void stage_denoise_outer_fast_invocation_authority(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    uint64_t plan_hash);
at::Tensor prepare_denoise_outer_fast_replay(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& noise,
    const at::Tensor& state,
    uint64_t epoch,
    uint64_t plan_hash);
void prepare_denoise_masks(int64_t handle,
                           const at::Tensor& tmask,
                           const at::Tensor& imask,
                           uint64_t plan_hash);
at::Tensor forward_denoise_prepared_masks(
    int64_t handle,
    const at::Tensor& noise,
    const at::Tensor& state,
    at::TensorList k_caches,
    at::TensorList v_caches,
    uint64_t epoch,
    uint64_t plan_hash);

void adopt_vl_encoder(int64_t handle,
                      const SpmPipelineLease& lease,
                      const SpmTensorView& scratch);
void adopt_denoise(int64_t handle,
                   const SpmPipelineLease& lease,
                   const SpmTensorView& scratch);

void bind_vl_encoder(int64_t handle,
                     const SpmPipelineLease& lease,
                     const SpmPortView& port);
void bind_denoise(int64_t handle,
                  const SpmPipelineLease& lease,
                  const SpmPortView& port);

void validate_vl_encoder(int64_t handle,
                         const SpmPipelineLease& lease);
void validate_denoise(int64_t handle,
                      const SpmPipelineLease& lease);

void clear_vl_encoder(int64_t handle, uint64_t epoch, uint64_t plan_hash);
void clear_denoise(int64_t handle, uint64_t epoch, uint64_t plan_hash);

// Public Z2 dispatch and destroy wrappers must consult the live coordinator;
// model-local epoch fields alone cannot prove physical lease ownership.
void validate_vl_dispatch(int64_t handle,
                          uint64_t epoch,
                          uint64_t plan_hash);
void validate_denoise_dispatch(int64_t handle,
                               uint64_t epoch,
                               uint64_t plan_hash);
void check_vl_destroy_allowed(int64_t handle);
void check_denoise_destroy_allowed(int64_t handle);
void check_vl_component_destroy_allowed(int64_t handle);
void check_denoise_component_destroy_allowed(int64_t handle);

}  // namespace v3::gr00t_z2_internal

void rpu_gr00t_spm_z2_unprepare(
    int64_t vl_handle,
    int64_t denoise_handle,
    int64_t plan_hash);
