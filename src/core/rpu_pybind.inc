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
        "Get the immutable SDK-load batch-buffer configuration twice for ABI compatibility");
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
    py::dict oplib_roles;
    oplib_roles["combined"] = FROZEN_KERNEL_LIB_PATH();
    result["oplib_roles"] = std::move(oplib_roles);
    // Hash only the sealed snapshots used by Program_t, never the mutable
    // source paths. Each library is hashed once regardless of kernel count.
    py::dict oplib_payloads;
    const auto sha256 = py::module_::import("hashlib").attr("sha256");
    for (const auto& [path, bytes] : KernelCache::instance().loaded_oplib_bytes()) {
      py::dict payload;
      payload["sha256"] = sha256(py::bytes(bytes)).attr("hexdigest")();
      payload["size"] = bytes.size();
      oplib_payloads[py::str(path)] = std::move(payload);
    }
    result["oplib_payloads"] = std::move(oplib_payloads);
    return result;
  }, "Return loaded library paths and immutable loaded op-library identities");

  // Debug and profile switches
  m.def("rpu_set_debug_level", &rpu_set_debug_level,
        "Set 6-level debug verbosity (0 SILENT, 1 ERROR, 2 WARN, 3 INFO, 4 DEBUG, 5 TRACE)");
  m.def("rpu_get_debug_level", &rpu_get_debug_level, "Get current debug level (0..5)");
  // BC: legacy bool toggle — true→DEBUG(4), false→ERROR(1).
  m.def("set_debug", &rpu_set_debug, "Enable/disable debug logging (CPU_FALLBACK, MANUAL_FALLBACK)");
  m.def("set_profile", &rpu_set_profile, "Enable/disable profile timing output");
  m.def("get_debug", &rpu_get_debug, "Get debug logging state");
  m.def("get_profile", &rpu_get_profile, "Get profile timing state");

  // Cold process-wide tensor allocator policy.
  m.def("set_caching_allocator", &rpu_set_caching_allocator,
        "Select the caching allocator before the first RPU tensor allocation");
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
        py::arg("enabled"), py::arg("output_dir") = ".rpu-hw-perf",
        py::arg("max_dumps") = 32);
  m.def("get_hw_perf_trace", &rpu_get_hw_perf_trace, "Get HW perf trace enabled state");

  // Shutdown - release all cached resources (SPM pool, caching allocator)
  // Must be called before program exit while Rhino runtime is still alive
  m.def("shutdown", &rpu_shutdown, "Release all cached resources (call before exit)");

  // Debug export - store intermediate tensors for verification

  // SPM debug - control SPM/chunk size debug output
  m.def("set_spm_debug", &set_spm_debug, "Enable/disable SPM/chunk debug output");
  m.def("get_spm_debug", &get_spm_debug, "Get SPM debug state");
  // MR-D: set_chunk_size / get_chunk_size are GONE. Chunk size keys the SPM
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
        py::arg("v_input"), py::arg("v_cache"), py::arg("position"),
        py::arg("planned_route_arguments"));

  m.def("insert_kcache_arena", &rpu_launch_llama_insert_kcache_arena,
        "Insert K into K-Cache via repo SPM arena (8-core)",
        py::arg("k_input"), py::arg("k_cache"), py::arg("position"),
        py::arg("planned_route_arguments"));

  m.def("insert_kvcache_arena", &rpu_launch_llama_insert_kvcache_arena,
        "Insert paired K/V into caches via one repo SPM arena plan (8-core)",
        py::arg("k_input"), py::arg("v_input"), py::arg("k_cache"),
        py::arg("v_cache"), py::arg("position"),
        py::arg("planned_route_arguments"));

  m.def("resolve_kvcache_arena_plan",
        &rpu_resolve_llama_kvcache_arena_plan,
        "Resolve one immutable arena KV-insert route before launch",
        py::arg("position"), py::arg("rows"),
        py::arg("num_kv_heads"), py::arg("head_dim"));

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

  // SDPA helper exposure for tests.
  // Submodule name has NO leading underscore: __init__.py:73 filters
  // underscore-prefixed C-ext attrs out of the synthetic _cpp_ext module.
  // kv_helpers — retained pure KV-insert shape predicate test surface.  The
  // segment-plan resolver shares this shape contract; launchers consume its
  // immutable result and never read env policy themselves.
  auto kv = m.def_submodule("kv_helpers", "Internal KV-insert helpers (test only)");
  kv.def("shape_plan",
      [](int64_t position, int64_t seq_len, int num_cores,
         int64_t num_kv_heads, int64_t head_dim) {
          const KvInsertShapePlan p = rpu_kvinsert_shape_plan(
              position, seq_len, num_cores, num_kv_heads, head_dim);
          py::dict d;
          d["bulk_seq"] = p.bulk_seq;
          d["tail_seq"] = p.tail_seq;
          d["v16_shape_ok"] = p.v16_shape_ok;
          d["hybrid_shape_ok"] = p.hybrid_shape_ok;
          return d;
      },
      "Pure shape half of the KV-insert v16/hybrid decision: returns "
      "{bulk_seq, tail_seq, v16_shape_ok, hybrid_shape_ok}. No env gating.",
      py::arg("position"), py::arg("seq_len"), py::arg("num_cores"),
      py::arg("num_kv_heads"), py::arg("head_dim"));

  // chunk_helpers — the planner's SELECTION RULE, minus the SPM budget and the
  // per-model kernel predicate (those need weights and a layout). What is left
  // is the policy itself: among the splits that fit and are legal, take the one
  // with the fewest chunks, and among those the most even (n*cs - seq_len is
  // both the tail deficit and the padding that would make every chunk equal).
  // It had a C++ test in tests/test_buffer_scope_contract.cpp that CMake never
  // builds, so the rule has never actually been gated anywhere.
  auto chunk = m.def_submodule("chunk_helpers",
      "Internal chunk-plan selection rule (test only)");
  chunk.def("prefer_balanced",
      [](int64_t seq_len, int64_t candidate, int64_t current) {
          return v3::detail::prefer_balanced_chunk(seq_len, candidate, current);
      },
      "True when `candidate` is a better chunk size than `current` for "
      "seq_len: fewer chunks first, then the smaller tail deficit.",
      py::arg("seq_len"), py::arg("candidate"), py::arg("current"));
  chunk.def("best_chunk_size",
      [](int64_t seq_len, const std::vector<int64_t>& candidates) {
          int64_t best = 0;
          for (int64_t cs : candidates)
              if (v3::detail::prefer_balanced_chunk(seq_len, cs, best)) best = cs;
          return best;
      },
      "Apply the selection rule over an explicit candidate list (the caller "
      "supplies whatever passed the SPM budget and the kernel predicate).",
      py::arg("seq_len"), py::arg("candidates"));

  auto sdpa = m.def_submodule("sdpa_helpers", "Internal SDPA helpers (test only)");
  sdpa.def("tile_m_v16", &sdpa_tile_m_v16,
      "Initial tile_m_v16 rule (sQry<=176: ceil(sQry,16), else 8).",
      py::arg("seq_q"));
  sdpa.def("compute_tiling", [](int64_t head_dim, int64_t seq_q, int attn_mask_type) {
      SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM, head_dim, 0, 0, 0, attn_mask_type};
      SdpaTiling t = sdpa_compute_tiling(cfg, seq_q);
      py::dict d;
      d["tile_m_v16"] = t.tile_m_v16;
      d["tile_n_v16"] = t.tile_n_v16;
      d["tile_k_v16"] = t.tile_k_v16;
      d["tile_m"] = t.tile_m;
      return d;
  },
      "Compute (tile_m_v16, tile_n_v16, tile_k_v16, tile_m) for a given head_dim, seq_q, mask.",
      py::arg("head_dim"), py::arg("seq_q"), py::arg("attn_mask_type"));
  // R12: non-throwing tiling — returns None on infeasibility instead of throwing.
  sdpa.def("try_compute_tiling",
      [](int64_t head_dim, int64_t seq_q, int attn_mask_type) -> py::object {
          SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM, head_dim, 0, 0, 0, attn_mask_type};
          SdpaTiling t;
          if (!sdpa_try_compute_tiling(cfg, seq_q, &t)) return py::none();
          py::dict d;
          d["tile_m_v16"] = t.tile_m_v16;
          d["tile_n_v16"] = t.tile_n_v16;
          d["tile_k_v16"] = t.tile_k_v16;
          d["tile_m"] = t.tile_m;
          return d;
      },
      "Non-throwing variant of compute_tiling (R12). Returns None iff no "
      "(tm, tn, tk) tuple satisfies C1–C4 within kMaxShrinkIter=16 halvings.",
      py::arg("head_dim"), py::arg("seq_q"), py::arg("attn_mask_type"));
  sdpa.def("is_valid_chunk_size",
      [](int64_t num_q_heads, int64_t num_kv_heads, int64_t num_cores,
         int64_t head_dim, int attn_mask_type,
         int64_t cs, int64_t seq_len, int64_t position,
         int64_t max_sync_product) {
          SdpaConfig cfg{SdpaKernelType::FLASH_ATTN_SPM, head_dim,
                         num_q_heads, num_kv_heads, (int)num_cores,
                         attn_mask_type, max_sync_product};
          return sdpa_is_valid_chunk_size(cfg, cs, seq_len, position);
      },
      "Validity predicate: returns whether chunk_size cs is legal for "
      "a chunked-prefill pass with seq_len, position. Used by Qwen3 auto chunk picker. "
      "max_sync_product is an optional profile capability ceiling (0=structural rule only).",
      py::arg("num_q_heads"), py::arg("num_kv_heads"), py::arg("num_cores"),
      py::arg("head_dim"), py::arg("attn_mask_type"),
      py::arg("cs"), py::arg("seq_len"), py::arg("position"),
      py::arg("max_sync_product") = 0);
  sdpa.def("validate_kernel_call",
      &sdpa_validate_kernel_call,
      "Keeper-side runtime validator. Throws TORCH_CHECK if any call constraint "
      "fails. Called at every SDPA kernel launcher entry.",
      py::arg("seq_q"), py::arg("seq_k"),
      py::arg("tile_m_v16"), py::arg("tile_n_v16"), py::arg("tile_k_v16"),
      py::arg("tile_m"),
      py::arg("grid_dim_x"), py::arg("gqa_group_size"),
      py::arg("hd_q_v16"), py::arg("hd_k_v16"), py::arg("hd_v_v16"),
      py::arg("attn_mask_type"));

  // choose_tile_k family — pure helpers (R11/R13/R14). Test surface only.
  // Returns Python int on success, None on std::nullopt.
  using OptInt = std::optional<int64_t>;
  sdpa.def("choose_tile_k",
      [](int64_t tm_v16, int64_t tn_v16, int attn_mask_type, int64_t head_dim_v16) -> OptInt {
          return choose_tile_k(tm_v16, tn_v16, attn_mask_type, head_dim_v16);
      },
      "Compile-time-dispatched tile_k chooser (default conservative, "
      "aggressive iff -DRPU_SDPA_TILE_K_AGGRESSIVE_MODE).",
      py::arg("tm_v16"), py::arg("tn_v16"),
      py::arg("attn_mask_type"), py::arg("head_dim_v16"));
  sdpa.def("choose_tile_k_aggressive",
      [](int64_t tm_v16, int64_t tn_v16, int attn_mask_type, int64_t head_dim_v16) -> OptInt {
          return choose_tile_k_aggressive(tm_v16, tn_v16, attn_mask_type, head_dim_v16);
      },
      "Full descending [1,16] scan, returns nullopt iff no tk satisfies C1–C4.",
      py::arg("tm_v16"), py::arg("tn_v16"),
      py::arg("attn_mask_type"), py::arg("head_dim_v16"));
  sdpa.def("choose_tile_k_conservative",
      [](int64_t tm_v16, int64_t tn_v16, int attn_mask_type, int64_t head_dim_v16) -> OptInt {
          return choose_tile_k_conservative(tm_v16, tn_v16, attn_mask_type, head_dim_v16);
      },
      "Two-tier scan: Tier-1 {16,11,8,5,4,2,1}; Tier-2 tk=tm fallback.",
      py::arg("tm_v16"), py::arg("tn_v16"),
      py::arg("attn_mask_type"), py::arg("head_dim_v16"));
  sdpa.def("vlm_need",
      [](int64_t tm_v16, int64_t tn_v16, int64_t tk_v16,
         int64_t hd_q_v16, int64_t hd_k_v16, int64_t hd_v_v16, int attn_mask_type) {
          return (int64_t)GetMhaAttnUnivVlmNeed(
              (uint16_t)tm_v16, (uint16_t)tn_v16, (uint16_t)tk_v16,
              (uint16_t)hd_q_v16, (uint16_t)hd_k_v16, (uint16_t)hd_v_v16,
              (uint16_t)attn_mask_type);
      },
      "Direct exposure of GetMhaAttnUnivVlmNeed for golden-table tests.",
      py::arg("tm_v16"), py::arg("tn_v16"), py::arg("tk_v16"),
      py::arg("hd_q_v16"), py::arg("hd_k_v16"), py::arg("hd_v_v16"),
      py::arg("attn_mask_type"));
  sdpa.attr("MAX_VLM_SIZE") = py::int_((int64_t)SDPA_MAX_VLM_SIZE);
  sdpa.attr("kSdpaTilePolicyBase") = py::int_(kSdpaTilePolicyBase);
  sdpa.attr("kSdpaTilePolicyMode") = py::int_(kSdpaTilePolicyMode);
  sdpa.attr("kSdpaTilePolicyVersion") = py::int_(kSdpaTilePolicyVersion);

  // R4 GraphCacheKey policy-version inequality probe removed:legacy admission
  // helper depended on rpu_batch_context.h types which have been retired.
  // Python-side GraphSignature 现承担 sdpa_tile_policy_version 校验责任。

  // P5: bind Graph / GraphCache / GraphSignature 到当前 pybind module 末尾;
  // Python 端通过 rpu_backend._cpp_ext.{Graph,GraphCache,GraphSignature}
  // 拿到,__init__.py re-export 到顶层名字。
  graph_pybind::add_bindings(m);
  m.def("reset_graph_cache", &invalidate_registered_rpu_kernel_graphs,
        "Invalidate all live top-level RPU graphs between forwards");

}  // end pybind module
