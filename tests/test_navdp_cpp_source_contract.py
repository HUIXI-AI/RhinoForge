"""Board-free source contracts for the NavDP/Former fused subsystem."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/fused/rpu_navdp_model.cpp").read_text()
RUNTIME = (
    ROOT / "python/rpu_backend/adapters/navdp/runtime.py"
).read_text()
RGBD_RUNTIME = (
    ROOT / "python/rpu_backend/adapters/internvla_n1/rgbd_rpu.py"
).read_text()
REGISTRATIONS = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()


def _section(start: str, end: str) -> str:
    begin = SOURCE.index(start)
    return SOURCE[begin:SOURCE.index(end, begin)]


def test_only_the_two_certified_weight_profiles_are_admitted():
    setter = _section("void set_weights(", "// memory[")
    for token in (
        "N == kNavdpLayers",
        "ff_inter == kNavdpFf",
        "memory_len == kNavdpMemory",
        "N == kFormerLayers",
        "ff_inter == kFormerFf",
        "memory_len == kFormerMemory",
        "eps == kEps",
        "navdp_profile || former_profile",
    ):
        assert token in setter
    assert setter.index('check_length(n3_b, "n3_b")') < setter.index(
        'check_list(sq_w, "sq_w"'
    )
    assert setter.rstrip().endswith("invalidate_model_state();  // D-503\n    }")


def test_weight_lists_are_exact_fp16_contiguous_rpu_tensors():
    setter = _section("void set_weights(", "// memory[")
    assert setter.count("check_length(") == 26
    assert setter.count("check_list(") == 26
    for token in (
        "list[i].sizes() == sizes",
        "list[i].scalar_type() == at::kHalf",
        "list[i].device().type() == at::kPrivateUse1",
        "list[i].is_contiguous()",
    ):
        assert token in setter


def test_dispatch_mode_and_denoise_topology_are_startup_fixed():
    assert "enum class ExecutionMode : uint8_t { Unset, Forward, Denoise };" in SOURCE
    assert "handle is locked to denoise mode" in SOURCE
    assert "handle is locked to ordinary forward mode" in SOURCE
    assert "num_steps and batch are fixed after first dispatch" in SOURCE
    for token in (
        "{kDenoiseSteps, kNavdpMemory, kHidden}",
        "{kDenoiseSteps, B * kPredictSize, kActionDimPadded}",
        "CPU fp32 contiguous [20, 4]",
    ):
        assert token in SOURCE


def test_python_batch_unroll_latches_sample_num_before_native_dispatch():
    builder = RUNTIME[RUNTIME.index("def build_navdp("):]
    for env, field in (
        ("RPU_NAVDP_BATCH", "batch_unroll"),
        ("RPU_NAVDP_DENOISE_UNROLL", "denoise_unroll"),
    ):
        assert builder.index(f'{field} = rpu_env_bool("{env}"') < builder.index("load_file(str(checkpoint))")
        assert f"{field}={field}" in builder
        assert f"self._{field} = bool({field})" in RUNTIME
    predict = RUNTIME[RUNTIME.index("    def predict_action("):RUNTIME.index("    def destroy(self):")]
    assert "os.environ" not in predict and "rpu_env_bool(" not in predict
    assert "if self._denoise_unroll:" in predict
    method = RUNTIME[
        RUNTIME.index("    def _predict_action_unroll("):
        RUNTIME.index("    def destroy(self):")
    ]
    guard = "if batch_unroll and batch_n is not None and batch_n != sample_num:"
    assert guard in method
    assert method.index(guard) < method.index("from diffusers.schedulers")
    assert "RPU_NAVDP_BATCH=0" in method
    assert "os.environ" not in method
    assert "batch_unroll = self._batch_unroll" in method
    batch = method[method.index("if batch_unroll:"):method.index("outs = []")]
    for token in (
        "if batch_n is None:",
        "batch_size=sample_num",
        "self._batch_n = sample_num",
    ):
        assert token in batch
    assert batch.index("self._batch_n = sample_num") < batch.index(
        "navdp_denoise_loop_forward("
    )
    serial = method[method.index("outs = []"):]
    assert "_batch_n" not in serial


def test_layout_hash_covers_all_navdp_owned_layout_state():
    layout_hash = _section(
        "int64_t subclass_layout_hash() const override {",
        "std::vector<BufferDecl> declare_buffers",
    )
    for token in (
        "memory_len_",
        "predict_size_",
        "action_dim_",
        "former_mode_",
        "loop_mode_",
        "num_steps_",
        "batch_",
    ):
        assert token in layout_hash


def test_navdp_raw_spm_self_attention_is_exact_and_ddr_fallback_survives():
    eligibility = _section(
        "bool subclass_spm_kv_by_mha_eligible(",
        "std::vector<BufferDecl> declare_buffers",
    )
    for token in (
        "former_mode_",
        "position != 0",
        "plan.compute.chunks.size() != 1",
        "plan.spans.size()) != batch_",
        "num_layers() == kNavdpLayers",
        "memory_len_ == kNavdpMemory",
    ):
        assert token in eligibility
    assert "RPU_NAVDP_SPM_KV_BY_MHA" not in SOURCE
    assert (
        'rpu_env_bool("RPU_NAVDP_SPM_KV_BY_MHA", default=True)'
        in RUNTIME
    )
    assert "navdp_create(True, spm_kv_by_mha)" in RUNTIME
    assert "navdp_create(True, False)" in RGBD_RUNTIME

    layer = _section(
        "void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {",
        "\nprivate:\n",
    )
    self_attn = layer[layer.index("// ---- sub-block 1:"):layer.index(
        "// ---- sub-block 2:"
    )]
    cross_attn = layer[layer.index("// ---- sub-block 2:"):]
    for token in (
        "rpu_launch_insert_kvcache_spm_unified_with_plan",
        "AttentionExecutionPolicy::DDR_KV",
        "rpu_launch_v_transpose_spm(",
        "rpu_launch_sdpa_by_mha_spm(",
    ):
        assert token in self_attn
    assert "rpu_launch_sdpa_spm_unified_kernel_v2(" in cross_attn
    assert "rpu_launch_sdpa_by_mha_spm(" not in cross_attn


def test_navdp_kv_sites_consume_frozen_typed_segment_plans():
    for site_id in (
        5578031918666539451,
        5028686091549759246,
    ):
        assert str(site_id) in SOURCE
    assert "resolve_kvinsert_plan_auto(" in SOURCE
    assert "kNavdpSelfKvSite, manifest.graph_lifecycle," in SOURCE
    assert "kNavdpCrossKvSite, manifest.graph_lifecycle," in SOURCE
    assert "rpu_kvinsert_route_arguments(" in SOURCE
    assert "restore_kvinsert_plan(" in SOURCE
    assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in SOURCE
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in SOURCE
    assert "rpu_launch_insert_kvcache_spm_unified(" not in SOURCE
    assert "route.arguments" in SOURCE


def test_navdp_complete_descriptor_owns_schedule_and_physical_routes():
    manifest = _section(
        "FmbPhysicalExecutionManifest physical_manifest_for_candidate(",
        "FmbPhysicalManifestForwardCapability",
    )
    for token in (
        "kNavdpExecutionScheduleSite",
        "kNavdpFormerScheduleSite",
        "FmbRouteFamily::GRAPH_SCHEDULE",
        "kNavdpProjectionLinearSite",
        "kNavdpSelfAllReduceSite",
        "kNavdpLoopPreloadDmaSite",
        "kNavdpLoopInitialStateDmaSite",
        "kNavdpLoopMemoryDmaSite",
        "kNavdpLoopNoiseDmaSite",
        "kNavdpLoopOutputDmaSite",
        "kNavdpMemoryDmaSite",
        "append_fmb_shared_runtime_routes(",
    ):
        assert token in manifest

    layer = _section(
        "void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override {",
        "\nprivate:\n",
    )
    assert "consume_physical_attention_route(" not in layer
    assert "FmbRouteFamily::ATTENTION" in layer
    assert "{batch_, predict_size_, seq}" in layer
    assert "{batch_, predict_size_, memory_len_}" in layer

    resolver = _section(
        "KvInsertSegmentPlan resolve_kv_insert_plan(",
        "// affine LayerNorm",
    )
    for token in (
        "ctx().physical_route_invocation",
        "rpu_resolve_kvinsert_segment_plan(",
        "plan.route()",
        "const KvInsertSegmentPlan plan = restore_kvinsert_plan(",
        "const KvInsertSegmentPlan shape_plan =",
        "kKvInsertRouteTopologyWords",
        "return plan;",
        "route.arguments.size() == canonical.size()",
    ):
        assert token in resolver


def test_navdp_publishes_trajectory_spans_to_the_common_three_stage_plan():
    assert SOURCE.count("const std::vector<ChunkInfo> input_chunks") == 3
    assert SOURCE.count("const std::vector<FmbExecutionSpan> spans") == 1
    assert "std::vector<FmbExecutionSpan> spans;" in SOURCE
    assert "spans.push_back({b * predict_size_, predict_size_, b});" in SOURCE
    assert "stream * predict_size_, predict_size_, stream" in SOURCE
    assert "const FmbStageBoundaryPolicies boundary_policies{};" in SOURCE
    assert "/*is_causal=*/!former_mode_," in SOURCE
    assert "/*is_causal=*/true, input_chunks, spans" in SOURCE
    assert "encode_fmb_prefill_stage_domain(" in SOURCE
    assert "planned_stage_descriptor" in SOURCE


def test_navdp_rejects_partial_or_cross_trajectory_compute_chunks():
    planning = _section(
        "bool subclass_chunk_size_valid(",
        "int64_t subclass_layout_hash() const override {",
    )
    for token in (
        "seq_len % predict_size_ == 0",
        "chunk_size == seq_len",
        "plan.num_chunks == 1",
        "plan.chunk_size == bps()",
        "one complete, trajectory-aligned compute chunk",
    ):
        assert token in planning


def test_production_path_has_no_debug_behavior_switches():
    for env_name in (
        "RPU_NAVDP_TAP",
        "RPU_NAVDP_OTAP",
        "RPU_NAVDP_XKV_NORM",
    ):
        assert env_name not in SOURCE


def test_native_create_cannot_bypass_controlled_profile_policy():
    create = _section(
        "int64_t rpu_navdp_create(bool allow_numeric_blocked, bool spm_kv_by_mha) {",
        "void    rpu_navdp_destroy",
    )
    assert "TORCH_CHECK(\n        allow_numeric_blocked" in create
    assert "explicit controlled numeric-blocked admission" in create
    assert "std::getenv" not in create
    assert (
        'navdp_create(bool allow_numeric_blocked=False, '
        'bool spm_kv_by_mha=True) -> int'
        in REGISTRATIONS
    )
