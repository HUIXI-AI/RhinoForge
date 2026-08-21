// CPython module bindings included by rpu_backend.cpp.
// Keeping the module body in an include fragment preserves the process-runtime
// initialization and teardown ordering of the composition translation unit.

#include "rhino_launch_buffer.h"

// ------------------------ Python binding to ensure PyInit symbol
// ------------------------
static std::string runtime_shared_object_path(
    const void* address, const char* label) {
  Dl_info info{};
  TORCH_CHECK(
      address != nullptr && dladdr(address, &info) != 0 &&
          info.dli_fname != nullptr && *info.dli_fname != '\0',
      "cannot resolve loaded ", label);
  return canonical_payload_path(info.dli_fname);
}

PYBIND11_MODULE(rpu_backend, m) {
  m.doc() = "RPU backend extension";

  // Explicit startup chain: PrivateUse1 allocator/guard, then kernel cache.
  initialize_rpu_backend_runtime();

  m.def("rpu_runtime_is_available", &rpu_runtime_is_available,
        "Return whether the initialized backend can access required device nodes");

  m.def("rpu_add", &rpu_add, "rpu_add");
  m.def("rpu_copy_", &rpu_copy_, "rpu_copy_");
  m.def("rpu_view", &rpu_view, "rpu_view");
  m.def("get_lkn_batch_config", &rpu_get_lkn_batch_config,
        "Get load-time and current SDK batch-buffer configuration");
  m.def("get_runtime_payload_paths", [] {
    void* sdk_symbol = dlsym(RTLD_DEFAULT, "RpuInit");
    TORCH_CHECK(sdk_symbol != nullptr,
                "cannot resolve loaded RPU SDK symbol RpuInit");
    const void* launch_symbol = reinterpret_cast<const void*>(
        reinterpret_cast<uintptr_t>(&::rhino_lkn::RpuDdrAlloc));
    py::dict result;
    result["extension"] = runtime_shared_object_path(
        static_cast<const void*>(&KERNEL_LIST), "rpu_backend extension");
    result["sdk"] = runtime_shared_object_path(sdk_symbol, "RPU SDK");
    result["launch"] = runtime_shared_object_path(
        launch_symbol, "Rhino launch library");
    result["oplibs"] = KernelCache::instance().loaded_oplib_paths();
    return result;
  }, "Return canonical paths for the loaded extension, SDK, and op libraries");

  // Debug and profile switches
  m.def("rpu_set_debug_level", &rpu_set_debug_level,
        "Set 6-level debug verbosity (0 SILENT, 1 ERROR, 2 WARN, 3 INFO, 4 DEBUG, 5 TRACE)");
  m.def("rpu_get_debug_level", &rpu_get_debug_level, "Get current debug level (0..5)");
  // BC: legacy bool toggle — true→DEBUG(4), false→ERROR(1).
  m.def("set_debug", &rpu_set_debug, "Enable/disable debug logging (CPU_FALLBACK, MANUAL_FALLBACK)");
  m.def("set_profile", &rpu_set_profile, "Enable/disable profile timing output");
  m.def("get_debug", &rpu_get_debug, "Get debug logging state");
  m.def("get_profile", &rpu_get_profile, "Get profile timing state");

  // Caching allocator switch
  m.def("set_caching_allocator", &rpu_set_caching_allocator, "Enable/disable caching allocator");
  m.def("get_caching_allocator", &rpu_get_caching_allocator, "Get caching allocator state");
  m.def("empty_cache", &rpu::empty_cache,
        "Release cached RPU allocator blocks");
  m.def("get_memory_stats", &rpu_get_memory_stats_py,
        "Get RPU caching allocator stats");
  m.def("reset_peak_memory_stats", &rpu::reset_peak_memory_stats,
        "Reset RPU allocator peak stats");
  m.def("reset_accumulated_memory_stats",
        &rpu::reset_accumulated_memory_stats,
        "Reset RPU allocator accumulated stats");

  // DDR flush switch
  m.def("set_ddr_flush", &rpu_set_ddr_flush, "Enable/disable DDR flush operations");
  m.def("get_ddr_flush", &rpu_get_ddr_flush, "Get DDR flush state");
  m.def("set_ddr_flush_force", &rpu_set_ddr_flush_force,
        "Enable/disable CPU<->RPU boundary force DDR flush");
  m.def("get_ddr_flush_force", &rpu_get_ddr_flush_force,
        "Get CPU<->RPU boundary force DDR flush state");

  // Profile reset (call before profiling runs to clear warmup data)
  m.def("reset_profile_accumulators", &rpu_reset_profile_accumulators, "Reset all profile accumulators");

  // HW perf trace — dump Chrome trace JSON for fused-decoder kernels.
  m.def("set_hw_perf_trace", &rpu_set_hw_perf_trace,
        "Enable HW cycle-precision kernel/DMA Chrome trace dump",
        py::arg("enabled"), py::arg("output_dir") = ".",
        py::arg("max_dumps") = 32);
  m.def("get_hw_perf_trace", &rpu_get_hw_perf_trace, "Get HW perf trace enabled state");

  // Shutdown - release all cached resources (SPM pool, caching allocator)
  // Must be called before program exit while the device runtime is still alive.
  m.def("shutdown", &rpu_shutdown, "Release all cached resources (call before exit)");

  // Debug export - store intermediate tensors for verification
  m.def("set_debug_export", &set_debug_export, "Enable/disable debug tensor export");
  m.def("get_debug_export", &get_debug_export, "Get debug tensor export state");
  m.def("get_debug_tensor", &get_debug_tensor, "Get debug tensor by name", py::arg("name"));
  m.def("clear_debug_tensors", &clear_debug_tensors, "Clear all debug tensors");
  m.def("list_debug_tensors", &list_debug_tensors, "List all debug tensor names");

  // SPM debug - control SPM/chunk size debug output
  m.def("set_spm_debug", &set_spm_debug, "Enable/disable SPM/chunk debug output");
  m.def("get_spm_debug", &get_spm_debug, "Get SPM debug state");
  m.def("get_allreduce_twostage", &rpu_get_allreduce_twostage_enabled,
        "Return the current process two-stage all-reduce selector");
  m.def("get_linear_acc32", &rpu_get_linear_acc32_enabled,
        "Return the process-cached fused Linear ACC32 selector");

  // set_chunk_size / get_chunk_size are not process-global. Chunk size keys the SPM
  // layout and the graph signature, so it is per-handle, not per-process; use
  // torch.ops.rpu.causal_decoder_set_chunk_size_override(handle, cs) (or
  // halo_image_flow_set_chunk_size_override). Rationale: rpu_runtime_state.h.

  // Cross-layer batch for multi-chunk prefill
  m.def("set_cross_layer_batch_prefill", &set_cross_layer_batch_prefill,
        "Enable cross-layer batch for multi-chunk prefill", py::arg("enabled"));
  m.def("get_cross_layer_batch_prefill", &get_cross_layer_batch_prefill,
        "Check if cross-layer batch prefill is enabled");
  m.def("set_cross_layer_batch_size", &set_cross_layer_batch_size,
        "Set number of layers per batch group", py::arg("size"));
  m.def("get_cross_layer_batch_size", &get_cross_layer_batch_size,
        "Get number of layers per batch group");

  // RMSNorm operation
  m.def("rms_norm", &rpu_rmsnorm, "RMSNorm operation (y = w * x * rsqrt(mean(x^2) + eps))",
        py::arg("input"), py::arg("normalized_shape"), py::arg("weight"), py::arg("eps") = 1e-5);

  // RoPE operation
  m.def("apply_rotary_pos_emb", &rpu_apply_rotary_pos_emb,
        "Apply rotary position embedding to query and key tensors",
        py::arg("query"), py::arg("key"), py::arg("cos"), py::arg("sin"), py::arg("position_ids"));

  // SDPA unified layout - use torch.ops.rpu.scaled_dot_product_attention_unified instead (faster aten dispatch)

  // LLaMA KV-Cache operations
  m.def("insert_vcache", &rpu_launch_llama_insert_vcache,
        "Insert V values into V-Cache (8-core kernel)",
        py::arg("v_input"), py::arg("v_cache"), py::arg("position"));

  m.def("insert_kcache", &rpu_launch_llama_insert_kcache,
        "Insert K values into K-Cache (8-core kernel)",
        py::arg("k_input"), py::arg("k_cache"), py::arg("position"));

  // Arena-staged inserts (RhinoVLA): stage through repo SPM_ALLOC (not SDK
  // LocalSPM) so a long denoise loop's external-prefix insert never collides
  // with cached graph SPM pools. Caller resets temporary after the per-layer loop.
  m.def("insert_vcache_arena", &rpu_launch_llama_insert_vcache_arena,
        "Insert V into V-Cache via repo SPM arena (8-core)",
        py::arg("v_input"), py::arg("v_cache"), py::arg("position"));

  m.def("insert_kcache_arena", &rpu_launch_llama_insert_kcache_arena,
        "Insert K into K-Cache via repo SPM arena (8-core)",
        py::arg("k_input"), py::arg("k_cache"), py::arg("position"));

  // Linear whose partition comes from the CALLER instead of being re-derived
  // from the shape. This is the escape hatch `aten::linear` does not have: its
  // schema carries no partition, so `rpu_linear` has to guess with
  // `rpu_get_linear_partition`, and that guess is wrong for every weight the
  // Python converter pinned to ROW by name. `runtime/weights.py`'s
  // `_guarded_linear_forward` routes those weights here with the partition the
  // converter recorded. Input must be 2-D and contiguous; the DDR kernel is
  // fixed at NUM_CORES, so a weight swizzled for a different core count cannot
  // use this entry point (the guard raises instead).
  m.def("linear_with_partition",
        [](const at::Tensor& input, const at::Tensor& weight,
           const c10::optional<at::Tensor>& bias, int partition) {
          TORCH_CHECK(input.dim() == 2,
                      "linear_with_partition: input must be 2-D [M, K], got ",
                      input.dim(), "-D");
          at::Tensor output = at::empty({input.size(0), weight.size(0)}, input.options());
          at::Tensor bias_t = bias.has_value() ? bias.value() : at::Tensor();
          bool has_bias = bias.has_value();
          rpu_launch_linear_ddr_kernel(input, weight, output, bias_t, has_bias, partition);
          return output;
        },
        "Linear with explicit partition (0=row, 1=col)",
        py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("partition"));

  // fused_qkv_attention removed — use fused_decoder_layer instead

  // Wall-OSS uses the same native validity predicate as automatic chunk
  // planning, avoiding a second Python implementation of the constraint.
  auto sdpa = m.def_submodule("sdpa_helpers", "SDPA runtime planning helpers");
  sdpa.def("is_valid_chunk_size",
      [](int64_t num_q_heads, int64_t num_kv_heads, int64_t num_cores,
         int64_t head_dim, int attn_mask_type,
         int64_t cs, int64_t seq_len, int64_t position) {
          SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM, head_dim,
                         num_q_heads, num_kv_heads, (int)num_cores, attn_mask_type};
          return sdpa_is_valid_chunk_size(cfg, cs, seq_len, position);
      },
      "Return whether chunk_size cs is valid for a chunked-prefill pass with "
      "the supplied sequence length and position.",
      py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("num_cores"),
      py::arg("head_dim"), py::arg("attn_mask_type"),
      py::arg("cs"), py::arg("seq_len"), py::arg("position"));
  // Bind Graph types after the device controls; Python re-exports them from
  // rpu_backend._cpp_ext.
  graph_pybind::add_bindings(m);
  m.def("reset_graph_cache", &invalidate_registered_rpu_kernel_graphs,
        "Invalidate all live top-level RPU graphs between forwards");

}  // end pybind module
