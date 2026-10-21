cmake_minimum_required(VERSION 3.18)
project(rpu_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Keep production source membership explicit.
set(RPU_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/core/fused_model_base.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/fmb_three_stage_chunk_plan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_caching_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_runtime_extras.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_runtime_state.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_spm_allocator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core/rpu_spm_pipeline.cpp"
)

set(RPU_FUSED_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3_5_moe_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3_5_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gemma_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gemma4_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_adarms_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_rhino_vla_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_pi05_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_pi05_denoise_step_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_wall_oss_action_step_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_lingbot_v2_moe_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_hyvla_vlm_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_hyvla_expert_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_siglip_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_hyvit2_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3vl_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3vl_decode_replay.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3_5_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen25vl_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_dinov3_vision_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_whisper_decoder.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gr00t_dit_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gr00t_vl_encoder_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_navdp_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gemma2_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_gr00t_spm_z2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_halo_action_expert_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_halo_image_flow_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_internvla_nextdit_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_lingbot_denoise_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3_5_spm_z2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3vl_multiview_spm_z2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwen3vl_pooler_spm_z1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_qwenpi05_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fused/rpu_wall_oss_spm_z1.cpp"
)

set(RPU_GRAPH_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_infra.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_register_census.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_runtime.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_runtime_capture.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_runtime_child.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_runtime_execute.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_runtime_dump.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_cpu_fallback_stub.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_pybind.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/custom_cpu_fallback_dispatch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/graph/graph_spm_reset_guard.cpp"
)

set(RPU_OPS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_gather.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_grouped_parallel_linear.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_moe_gather_mul.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_moe_index_shuffle.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_moe_topk_prob_norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_scatternd_add.cpp"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_gr00t_adaln.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_pad.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_reduce.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_pi05_down_pair.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_pi05_owner_norm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_pi05_prefill_pair.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_pi05_producer_chain.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_pi05_q_rope.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_pi05_vision_fc.cpp"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_fast_pos_embed.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_kvinsert_segment_plan.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_linear_gemv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_qwen3_5_gdn_fused.cpp"
)

add_library(rpu_backend SHARED
    ${RPU_CORE_SOURCES}
    ${RPU_FUSED_SOURCES}
    ${RPU_OPS_SOURCES}
    ${RPU_GRAPH_SOURCES}
)

target_link_libraries(rpu_backend PRIVATE ${TORCH_LIBRARIES} ${CMAKE_DL_LIBS})
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

if(RHINOFORGE_BUILD_HOST_TESTS)
    set(RHINOFORGE_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tests")
    get_target_property(RHINOFORGE_LAUNCH_HEADERS
        rhino_launch::rhino_launch INTERFACE_INCLUDE_DIRECTORIES)

    add_executable(test_fmb_three_stage_chunk_plan
        ${RHINOFORGE_TEST_DIR}/test_fmb_three_stage_chunk_plan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/core/fmb_three_stage_chunk_plan.cpp)
    add_executable(test_kvinsert_segment_plan
        ${RHINOFORGE_TEST_DIR}/test_kvinsert_segment_plan.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ops/rpu_kvinsert_segment_plan.cpp)
    add_executable(test_rmsnorm_route
        ${RHINOFORGE_TEST_DIR}/test_rmsnorm_route.cpp)
    foreach(host_test IN ITEMS test_fmb_three_stage_chunk_plan
            test_kvinsert_segment_plan test_rmsnorm_route)
        target_include_directories(${host_test} PRIVATE
            ${TORCH_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR}/ops
            ${RHINOFORGE_LAUNCH_HEADERS})
        target_link_libraries(${host_test} PRIVATE ${TORCH_LIBRARIES})
    endforeach()

    add_executable(test_pi05_execution_topology
        ${RHINOFORGE_TEST_DIR}/test_pi05_execution_topology.cpp)
    target_include_directories(test_pi05_execution_topology PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core)
    add_executable(test_lingbot_v2_moe_select
        ${RHINOFORGE_TEST_DIR}/test_lingbot_v2_moe_select.cpp)
    target_include_directories(test_lingbot_v2_moe_select PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/fused)
    foreach(host_test IN ITEMS test_fmb_three_stage_chunk_plan
            test_kvinsert_segment_plan test_rmsnorm_route
            test_pi05_execution_topology test_lingbot_v2_moe_select)
        target_compile_options(${host_test} PRIVATE -UNDEBUG)
        set_target_properties(${host_test} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
        add_test(NAME ${host_test} COMMAND ${host_test})
    endforeach()

    add_library(test_lingbot_v2_moe_cos_sin_offset_static OBJECT
        ${RHINOFORGE_TEST_DIR}/test_lingbot_v2_moe_cos_sin_offset_static.cpp)
    target_compile_options(test_lingbot_v2_moe_cos_sin_offset_static PRIVATE -UNDEBUG)
    target_include_directories(test_lingbot_v2_moe_cos_sin_offset_static PRIVATE
        ${TORCH_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR}/fused
        ${CMAKE_CURRENT_SOURCE_DIR}/ops ${CMAKE_CURRENT_SOURCE_DIR}/graph
        ${RHINOFORGE_LAUNCH_HEADERS})
endif()

if(RHINOFORGE_BUILD_DEVICE_TESTS)
    # The installed extension hides native symbols. Link tests from the same
    # implementation sources without changing the published extension's ABI.
    add_library(rhinoforge_device_test_backend STATIC EXCLUDE_FROM_ALL
        ${RPU_CORE_SOURCES} ${RPU_FUSED_SOURCES}
        ${RPU_OPS_SOURCES} ${RPU_GRAPH_SOURCES})
    target_include_directories(rhinoforge_device_test_backend PRIVATE
        ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/core
        ${CMAKE_CURRENT_SOURCE_DIR}/fused ${CMAKE_CURRENT_SOURCE_DIR}/ops
        ${CMAKE_CURRENT_SOURCE_DIR}/graph ${CMAKE_BINARY_DIR}/generated)
    target_link_libraries(rhinoforge_device_test_backend PUBLIC
        ${TORCH_LIBRARIES} torch_python rhino_launch::rhino_launch
        rpu_api Hxil ${CMAKE_DL_LIBS} Python3::Python)
    set(RHINOFORGE_DEVICE_TESTS
        test_buffer_scope_contract
        test_graph_ddr_pingpong
        test_graph_multi_group
        test_graph_ops_canary
        test_graph_replay_dma_ordering
        test_graph_two_phase_capture
        test_spm_pipeline_contract)
    foreach(device_test IN LISTS RHINOFORGE_DEVICE_TESTS)
        add_executable(${device_test} EXCLUDE_FROM_ALL
            ${CMAKE_CURRENT_SOURCE_DIR}/../tests/${device_test}.cpp)
        target_compile_options(${device_test} PRIVATE -UNDEBUG)
        target_include_directories(${device_test} PRIVATE
            ${TORCH_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/core ${CMAKE_CURRENT_SOURCE_DIR}/fused
            ${CMAKE_CURRENT_SOURCE_DIR}/ops ${CMAKE_CURRENT_SOURCE_DIR}/graph
            ${CMAKE_BINARY_DIR}/generated)
        target_link_libraries(${device_test} PRIVATE
            "-Wl,--whole-archive" rhinoforge_device_test_backend
            "-Wl,--no-whole-archive")
        set_target_properties(${device_test} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    endforeach()
    add_custom_target(rhino_device_tests DEPENDS ${RHINOFORGE_DEVICE_TESTS})
endif()
