cmake_minimum_required(VERSION 3.18)
project(rpu_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep production source membership explicit.
set(RPU_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/fused_model_base.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_caching_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_runtime_extras.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_runtime_state.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_spm_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_spm_pipeline.cpp"
)

set(RPU_FUSED_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3_5_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3_5_spm_z2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_wall_oss_spm_z1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3vl_pooler_spm_z1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3vl_multiview_spm_z2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gemma_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gemma4_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_adarms_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwenpi05_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_rhino_vla_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_pi05_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_pi05_denoise_step_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_wall_oss_action_step_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_lingbot_v2_moe_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_halo_action_expert_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_hyvla_vlm_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_hyvla_expert_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_halo_image_flow_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_siglip_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_hyvit2_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3vl_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3_5_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen25vl_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_dinov3_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gr00t_dit_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gr00t_vl_encoder_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gr00t_spm_z2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_internvla_nextdit_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_navdp_model.cpp"
)

set(RPU_GRAPH_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_infra.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_register_census.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_runtime.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_runtime_capture.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_runtime_child.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_runtime_execute.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_cpu_fallback_stub.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_pybind.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/custom_cpu_fallback_dispatch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_spm_reset_guard.cpp"
)

set(RPU_OPS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_argmax.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_bmm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_cast.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_concat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_conv2d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_eltwise_binary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_eltwise_unary.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_embedding.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_im2col.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_layernorm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_linear.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_linear_gemv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_llama_kvcache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_hyvla_merger.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_memcpy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_moe_select.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_mrope.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_partial_mrope.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_l2norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_cumsum.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_unit_tril_inv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_fla_conv1d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_tile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_slice.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_permute.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_gdn_recurrent.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_pad.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_reduce.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_rmsnorm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_rope.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_rope_2d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_all_gather.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_sdpa.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_sdpa_dispatch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_sdpa_vctxlen.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_sdpa_minibatch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_softmax.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_transpose.cpp"
)

add_library(rpu_backend SHARED
    ${RPU_CORE_SOURCES}
    ${RPU_FUSED_SOURCES}
    ${RPU_OPS_SOURCES}
    ${RPU_GRAPH_SOURCES}
)

target_link_libraries(rpu_backend PRIVATE ${TORCH_LIBRARIES})
target_include_directories(rpu_backend PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/fused
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/graph
)

target_link_libraries(rpu_backend
    PRIVATE
    rhino_launch::rhino_launch
    rpu_api
    Hxil
)

set_target_properties(rpu_backend PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
