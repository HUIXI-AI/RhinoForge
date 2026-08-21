#pragma once

#include "core/fused_model_base.h"
#include "core/rpu_spm_pipeline.h"

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace v3::qwen3vl_pooler_z1_internal {

SpmPipelineComponentLayout prepare_vision(int64_t handle,
                                          int64_t num_patches,
                                          int64_t retained_deepstack_count = 0);
SpmPipelineComponentLayout prepare_multiview_vision_dry(
    int64_t handle,
    int64_t num_patches);
void cancel_multiview_vision_dry(int64_t handle);
SpmPipelineComponentLayout prepare_multiview_vision_composite(
    int64_t handle,
    int64_t num_patches);
SpmPipelineComponentLayout prepare_multiview_vision_composite(
    int64_t handle,
    int64_t num_patches,
    bool force_block_twostage);
SpmFmbResolvedExecutionProfile resolve_multiview_vision_profile(
    int64_t handle);
uint64_t multiview_vision_occurrence_policy(int64_t handle);
SpmFmbResolvedPhaseManifest seal_multiview_vision_manifest(
    int64_t handle,
    const SpmFmbResolvedExecutionProfile& profile,
    SpmScratchId arena,
    uint32_t arena_base);
FusedModelBase& multiview_vision_owner(int64_t handle);
void prime_multiview_vision_position(int64_t handle);
void clear_multiview_vision_position(int64_t handle);
void unprepare_multiview_vision_composite(int64_t handle);
void forward_multiview_vision_composite(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches,
    at::TensorList v_caches);
void stage_multiview_vision_outer_fast_component(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches);
void bind_multiview_vision_outer_fast_input(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& input,
    size_t ordinal);
void prime_vision_inputs(int64_t handle, int64_t num_patches);
void rollback_vision_inputs(int64_t handle);
void unprepare_vision(int64_t handle);
SpmDense2DSpec vision_source_spec(int64_t handle,
                                  int64_t num_patches);
SpmDense2DSpec vision_deepstack1_source_spec(int64_t handle,
                                             int64_t num_patches);
SpmDense2DSpec vision_deepstack_source_spec(int64_t handle,
                                            int64_t num_patches,
                                            int64_t ordinal);
void adopt_vision(int64_t handle,
                  const SpmPipelineLease& lease,
                  const SpmTensorView& scratch);
void bind_vision_source(int64_t handle,
                        const SpmPipelineLease& lease,
                        const SpmPortView& source);
void bind_vision_deepstack1_source(int64_t handle,
                                  const SpmPipelineLease& lease,
                                  const SpmPortView& source);
void bind_vision_deepstack_source(int64_t handle,
                                  const SpmPipelineLease& lease,
                                  const SpmPortView& source,
                                  int64_t ordinal);
void validate_vision(int64_t handle,
                     const SpmPipelineLease& lease);
void clear_vision(int64_t handle, uint64_t epoch, uint64_t plan_hash);
void stage_vision_outer_fast_component(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches);
void bind_vision_outer_fast_input(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& input);
void forward_vision_z1(
    int64_t handle,
    const at::Tensor& input,
    at::TensorList k_caches,
    at::TensorList v_caches,
    int64_t num_patches,
    uint64_t epoch,
    uint64_t plan_hash);
void check_vision_destroy_allowed(int64_t handle);

SpmPipelineComponentLayout prepare_text(int64_t handle,
                                        int64_t execution_len,
                                        int64_t real_len);
SpmPipelineComponentLayout prepare_text_deepstack1(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len);
SpmPipelineComponentLayout prepare_text_deepstack3(
    int64_t handle,
    int64_t execution_len,
    int64_t real_len);
SpmPipelineCausalPrefillDryLayout prepare_multiview_text_dry(
    int64_t handle,
    int64_t execution_len);
void cancel_multiview_text_dry(int64_t handle);
SpmPipelineComponentLayout prepare_multiview_text_composite(
    int64_t handle,
    int64_t execution_len);
SpmPipelineComponentLayout prepare_multiview_text_composite(
    int64_t handle,
    int64_t execution_len,
    bool force_block_twostage_chunk_v2);
SpmFmbResolvedExecutionProfile resolve_multiview_text_profile(
    int64_t handle);
FusedModelBase& multiview_text_owner(int64_t handle);
SpmFmbResolvedPhaseManifest seal_multiview_text_manifest(
    int64_t handle,
    const SpmFmbResolvedExecutionProfile& profile,
    SpmScratchId arena,
    uint32_t arena_base);
SpmFmbConsumerRowSliceEndpoint seal_multiview_text_consumer(
    int64_t handle,
    const SpmFmbResolvedExecutionProfile& profile,
    const SpmFmbResolvedPhaseManifest& manifest,
    SpmPortId destination_port,
    int64_t row_begin,
    int layer_idx);
void prime_multiview_text_inputs(
    int64_t handle,
    const at::Tensor& position_ids,
    int64_t execution_len,
    const std::optional<at::Tensor>& rope_cos_il = std::nullopt,
    const std::optional<at::Tensor>& rope_sin_il = std::nullopt);
void rollback_multiview_text_inputs(int64_t handle);
void unprepare_multiview_text_composite(int64_t handle);
at::Tensor forward_multiview_text_composite(
    int64_t handle,
    const at::Tensor& hidden_states,
    at::TensorList k_caches,
    at::TensorList v_caches,
    const at::Tensor& position_ids,
    const std::optional<at::Tensor>& rope_cos_il = std::nullopt,
    const std::optional<at::Tensor>& rope_sin_il = std::nullopt);
void stage_multiview_text_outer_fast_component(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches);
void bind_multiview_text_outer_fast_input(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    const at::Tensor& hidden_states);
at::Tensor multiview_text_outer_fast_output(int64_t handle);
void prime_text_inputs(int64_t handle,
                       const at::Tensor& position_ids,
                       int64_t execution_len,
                       const std::optional<at::Tensor>& rope_cos_il,
                       const std::optional<at::Tensor>& rope_sin_il);
void rollback_text_inputs(int64_t handle);
void unprepare_text(int64_t handle);
SpmDense2DSpec text_destination_spec(int64_t handle,
                                     int64_t execution_len);
SpmDense2DSpec text_deepstack1_spec(int64_t handle);
SpmDense2DSpec text_deepstack_spec(int64_t handle, int64_t ordinal);
void adopt_text(int64_t handle,
                const SpmPipelineLease& lease,
                const SpmTensorView& scratch);
void bind_text_destination(int64_t handle,
                           const SpmPipelineLease& lease,
                           const SpmPortView& destination);
void bind_text_deepstack1(int64_t handle,
                          const SpmPipelineLease& lease,
                          const SpmPortView& deepstack1);
void bind_text_deepstack(int64_t handle,
                         const SpmPipelineLease& lease,
                         const SpmPortView& deepstack,
                         int64_t ordinal);
void validate_text(int64_t handle,
                   const SpmPipelineLease& lease);
void clear_text(int64_t handle, uint64_t epoch, uint64_t plan_hash);
void stage_text_outer_fast_component(
    int64_t handle,
    ::GraphKernelRegisterCensusGuard& guard,
    at::TensorList k_caches,
    at::TensorList v_caches);
at::Tensor text_outer_fast_output(int64_t handle);
at::Tensor forward_text_z1(
    int64_t handle,
    const at::Tensor& hidden_shape_carrier,
    at::TensorList k_caches,
    at::TensorList v_caches,
    const at::Tensor& position_ids,
    uint64_t epoch,
    uint64_t plan_hash);
void check_text_destroy_allowed(int64_t handle);

}  // namespace v3::qwen3vl_pooler_z1_internal

// Transactional, planning-only LingBot2 multiview probe.  The returned vector
// contains the exact Vision N256 and Text S225/C240->C225 dry layouts followed
// by a process allocator snapshot.  Both dry manifests are retired before the
// call returns; the probe never acquires a physical lease or opens a Graph.
std::vector<int64_t> rpu_qwen3vl_multiview_spm_dry_probe(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len);

int64_t rpu_qwen3vl_pooler_spm_z1_prepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len,
    int64_t retained_deepstack_count = 0);
void rpu_qwen3vl_pooler_spm_z1_unprepare(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t plan_hash);
std::vector<int64_t> rpu_qwen3vl_pooler_spm_z1_plan_stats(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t plan_hash);
int64_t rpu_qwen3vl_pooler_spm_z1_begin(
    int64_t vision_handle,
    int64_t text_handle,
    int64_t num_patches,
    int64_t execution_len,
    int64_t image_row_begin,
    int64_t real_len,
    const at::Tensor& position_ids,
    int64_t plan_hash,
    const std::optional<at::Tensor>& rope_cos_il = std::nullopt,
    const std::optional<at::Tensor>& rope_sin_il = std::nullopt);
void rpu_qwen3vl_pooler_spm_z1_end(
    int64_t epoch,
    int64_t plan_hash);
void rpu_qwen3vl_pooler_spm_z1_abort(
    int64_t epoch,
    int64_t plan_hash);
void rpu_qwen3vl_pooler_spm_z1_cancel(
    int64_t epoch,
    int64_t plan_hash);
at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward(
    int64_t vision_handle,
    int64_t text_handle,
    const at::Tensor& vision_input,
    at::TensorList vision_k_caches,
    at::TensorList vision_v_caches,
    const at::Tensor& text_hidden,
    at::TensorList text_k_caches,
    at::TensorList text_v_caches,
    const at::Tensor& position_ids,
    int64_t epoch,
    int64_t plan_hash);
