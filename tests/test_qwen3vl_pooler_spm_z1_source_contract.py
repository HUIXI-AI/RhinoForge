"""Board-free structural gates for the Qwen3-VL pooler/retained-DS Z1 producer."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISION_MODEL = ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp"
VISION_ADAPTER = ROOT / "python/rpu_backend/adapters/qwen3_vl/vision.py"


def _source() -> str:
    return VISION_MODEL.read_text(encoding="utf-8")


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_exact_fp16_2b_profile_is_fail_closed():
    source = _source()
    profile = _between(
        source,
        "void validate_pooler_z1_profile() const",
        "void validate_pooler_z1_contract() const",
    )

    for declaration in (
        "QWEN3VL_POOLER_Z1_NUM_PATCHES = 256",
        "QWEN3VL_POOLER_Z1_VISION_HIDDEN = 1024",
        "QWEN3VL_POOLER_Z1_MERGE_HIDDEN = 4096",
        "QWEN3VL_POOLER_Z1_MERGED_ROWS = 64",
        "QWEN3VL_POOLER_Z1_TEXT_HIDDEN = 2048",
        "QWEN3VL_POOLER_Z1_DEEPSTACK1_TAP = 5",
        "QWEN3VL_POOLER_Z1_SCRATCH_BYTES = 1966080",
    ):
        assert declaration in source

    assert "num_layers() == 24" in profile
    assert "num_q_heads() == 16" in profile
    assert "head_dim() == 64" in profile
    assert "intermediate_size() == 4096" in profile
    assert "deepstack_visual_indexes_.empty()" not in profile
    assert "spm_rope_route_available()" in profile
    assert "merger_active()" in profile
    assert "!fast_replay_skip_layer_loop_" in profile
    assert "!preload_replay_skip_" in profile
    assert "!bake_merger_" in profile
    assert "!prefix_scatter_active_" in profile
    assert "!get_debug_export()" in profile
    assert "/*allow_w8a16=*/false" in profile
    assert "MultiviewLinearProfile::DenseFp16" in profile
    assert "standalone pooler Z1 rejects W8A16" in profile
    assert "weight.scalar_type() == at::kHalf" in profile
    assert "!scale.defined()" in profile
    assert "weight.scalar_type() == at::kChar" in profile
    assert "scale.scalar_type() == at::kHalf" in profile
    retained_profile = _between(
        source,
        "void validate_pooler_z1_retained_deepstack_profile(",
        "void validate_pooler_z1_contract() const",
    )
    assert "retained_count == 1 || retained_count == 3" in retained_profile
    assert "deepstack_visual_indexes_[0] ==" in retained_profile
    assert "QWEN3VL_POOLER_Z1_DEEPSTACK1_TAP" in retained_profile
    assert "deepstack_visual_indexes_[1] == 11" in retained_profile
    assert "deepstack_visual_indexes_[2] == 17" in retained_profile
    assert "ds_merger_m0_w_.size() == 3" in retained_profile
    assert "ordinal < retained_count" in retained_profile
    assert 'ds_merger_m0_w_[ordinal], "ds_m0_w", ordinal' in retained_profile
    assert 'ds_merger_m2_w_[ordinal], "ds_m2_w", ordinal' in retained_profile


def test_prepare_creates_fixed_q_slot_before_layout_and_prime_flushes_position():
    source = _source()
    prepare = _between(
        source,
        "SpmPipelineComponentLayout prepare_pooler_z1_layout(",
        "void prime_pooler_z1_inputs(",
    )
    prime = _between(
        source,
        "void prime_pooler_z1_inputs(",
        "void rollback_pooler_z1_inputs(",
    )
    unprepare = _between(
        source,
        "void unprepare_pooler_z1_layout()",
        "SpmDense2DSpec pooler_z1_source_spec(",
    )

    assert prepare.index("!RpuKernelGraph::has_active()") < prepare.index(
        "q_ddr_slots_.emplace("
    )
    assert prepare.index("q_ddr_slots_.emplace(") < prepare.index(
        "compose_fmb_default_three_stage_chunk_plan("
    )
    assert "int64_t retained_count" in prepare
    assert "retained_count == 0 || retained_count == 1 ||" in prepare
    assert "retained_count == 3" in prepare
    assert prepare.index("validate_pooler_z1_retained_deepstack_profile(retained_count);") < prepare.index(
        "compose_fmb_default_three_stage_chunk_plan("
    )
    assert "ChunkMode::KV_FIRST" in prepare
    assert prepare.index("prepare_spm_pipeline_component(") < prepare.index(
        "z1_retained_deepstack_count_ = retained_count"
    )
    assert "prepared.temporary_bytes" in prepare
    assert "QWEN3VL_POOLER_Z1_SCRATCH_BYTES" in prepare
    assert prime.index("!RpuKernelGraph::has_active()") < prime.index(
        "rpu_ddr_flush_force_sized("
    )
    assert prime.index("rpu_ddr_flush_force_sized(") < prime.index(
        "z1_inputs_primed_ = true"
    )
    assert unprepare.index("rollback_pooler_z1_inputs();") < unprepare.index(
        "z1_prepared_num_patches_ = 0"
    )


def test_typed_source_lifecycle_and_free_wrapper_surface_are_complete():
    source = _source()
    source_spec = _between(
        source,
        "SpmDense2DSpec pooler_z1_source_spec(",
        "void adopt_pooler_z1_layout(",
    )
    bind = _between(
        source,
        "void bind_pooler_z1_source(",
        "void validate_pooler_z1_layout(",
    )
    clear = _between(
        source,
        "void clear_pooler_z1_layout(",
        "void forward_pooler_z1(",
    )
    wrappers = source[source.index("namespace v3::qwen3vl_pooler_z1_internal {") :]

    assert "spec.dtype = SpmPortDType::Fp16" in source_spec
    assert "spec.rows = QWEN3VL_POOLER_Z1_MERGED_ROWS" in source_spec
    assert "spec.cols = QWEN3VL_POOLER_Z1_TEXT_HIDDEN" in source_spec
    assert "spec.distribution = SpmPortDistribution::Replicated" in source_spec
    assert "source.spec() == expected" in bind
    assert "source.size_bytes() == expected.storage_bytes()" in bind
    assert "source.resolve_physical_addr(/*core=*/0, lease)" in bind
    assert "SpmDense2DSpec pooler_z1_deepstack_source_spec(" in source_spec
    assert "ordinal < z1_retained_deepstack_count_" in source_spec
    assert "void bind_pooler_z1_deepstack_source(" in bind
    assert "ordinal == z1_retained_deepstack_bound_count_" in bind
    assert "z1_retained_deepstack_source_addrs_[ordinal] = resolved_addr" in bind
    assert "++z1_retained_deepstack_bound_count_" in bind
    assert "resolved_addr != z1_source_addr_" in bind
    assert clear.index("release_spm_pipeline_component(") < clear.index(
        "z1_adopted_ = false"
    )

    for wrapper in (
        "prepare_vision",
        "prime_vision_inputs",
        "rollback_vision_inputs",
        "unprepare_vision",
        "vision_source_spec",
        "vision_deepstack_source_spec",
        "vision_deepstack1_source_spec",
        "adopt_vision",
        "bind_vision_source",
        "bind_vision_deepstack_source",
        "bind_vision_deepstack1_source",
        "validate_vision",
        "clear_vision",
        "forward_vision_z1",
        "check_vision_destroy_allowed",
    ):
        assert f"{wrapper}(" in wrappers


def test_dispatch_is_outer_graph_only_and_resets_on_every_exit():
    source = _source()
    ordinary = _between(source, "at::Tensor forward(\n", "at::Tensor forward_impl(")
    static_config = _between(
        source, "ModelStaticConfig static_config()", "ModelDynamicConfig dynamic_config("
    )
    z1_forward = _between(
        source,
        "void forward_pooler_z1(",
        "void check_pooler_z1_destroy_allowed() const",
    )

    assert "z1_prepared_num_patches_ == 0" in ordinary
    assert "use the coordinator-owned " in ordinary
    assert '"Z1 dispatch or cancel the dry probe"' in ordinary
    assert z1_forward.index("RpuKernelGraph::has_active()") < z1_forward.index(
        "z1_dispatch_ = true"
    )
    assert z1_forward.index("z1_dispatch_ = true") < z1_forward.index("try {")
    assert "/*z1_dispatch=*/true" in z1_forward
    assert z1_forward.count("z1_dispatch_ = false") == 2
    assert "catch (...)" in z1_forward
    assert "z1_dispatch_" in static_config
    assert "&Qwen3VLVisionModel::pooler_z1_post_fn" in static_config
    assert "&Qwen3VLVisionModel::merger_post_fn" in static_config


def test_z1_merger_writes_port_directly_while_ordinary_fallback_stays_stable():
    source = _source()
    emit = _between(source, "void emit_one_merger(", "void merger_post_fn()")
    ordinary = _between(source, "void merger_post_fn()", "void pooler_z1_post_fn()")
    z1_post = _between(source, "void pooler_z1_post_fn()", "private:")
    declarations = _between(
        source,
        "std::vector<BufferDecl> declare_buffers(",
        "void emit_preload_weights()",
    )

    assert "uint32_t final_out_addr" in emit
    assert "final_out_addr, mrows, merger_out_hidden_" in emit
    assert 'addr(0, "merger_out"), mrows, oh' not in emit
    assert 'emit_one_merger(addr(0, "residual1"), addr(0, "merger_out")' in ordinary
    assert "pooler_merged_ddr_slots_.at(seq)" in ordinary
    assert "deepstack_ddr_slots_.at(key)" in ordinary
    assert z1_post.count("emit_one_merger(") == 1
    assert "z1_source_addr_" in z1_post
    assert "rpu_launch_spm_copy_ddr_dma(" not in z1_post
    assert "deepstack_ddr_slots_" not in z1_post
    # Ordinary/Z1 aliases follow the separately allocated chunked-merger
    # declarations. Select the actual alias rather than the first same name.
    start = declarations.index('decls.push_back({"merger_out",')
    merger_out_decl = declarations[start:declarations.index('});', start)]
    assert '"residual1"' in merger_out_decl
    assert "ALL" in merger_out_decl
    start = declarations.index('decls.push_back({"merger_normed",')
    merger_normed_decl = declarations[start:declarations.index('});', start)]
    assert '"oproj"' in merger_normed_decl
    assert "COMP" in merger_normed_decl


def test_z1_retained_ds_is_immediate_while_raw_snapshot_and_final_dma_stay_suppressed():
    source = _source()
    layer = _between(
        source,
        "void build_layer_subgraph(",
        "bool merger_active() const",
    )
    impl = _between(source, "at::Tensor forward_impl(", "SpmPipelineComponentLayout")

    assert "if (!pipeline_dispatch_active() &&" in layer
    immediate = _between(
        layer,
        "// Retained DeepStack physical handoff:",
        "// GAP-B7: DeepStack snapshot",
    )
    assert "z1_dispatch_ && z1_retained_deepstack_count_ != 0" in immediate
    assert "QWEN3VL_POOLER_Z1_DEEPSTACK_TAPS.begin()" in immediate
    assert "z1_retained_deepstack_count_" in immediate
    assert "static_cast<int64_t>(layer_idx)" in immediate
    assert "chunk.idx == 0" in immediate
    assert "chunk.offset == 0" in immediate
    assert "chunk.len == QWEN3VL_POOLER_Z1_NUM_PATCHES" in immediate
    assert immediate.count("emit_one_merger(") == 1
    assert "z1_retained_deepstack_source_addrs_[ordinal]" in immediate
    assert "ds_merger_ln_q_w_.at(ordinal)" in immediate
    assert "ds_merger_m2_w_.at(ordinal)" in immediate
    assert "rpu_launch_spm_copy_ddr_dma(" not in immediate
    assert "const bool pipeline_final =" in layer
    assert "((z1_dispatch_ && z1_bound_) ||" in layer
    assert "multiview_composite_dispatch_)" in layer
    assert "if (!ctx().output_to_spm && !pipeline_final)" in layer
    z1_post = _between(source, "void pooler_z1_post_fn()", "private:")
    assert "deepstack_visual_indexes_" not in z1_post
    assert "deepstack_ddr_slots_" not in z1_post
    assert "if (!pipeline_dispatch) {" in impl
    assert "pooler_merged_ddr_slots_" in impl


def test_retained_cardinality_and_binding_lifecycle_are_fail_closed():
    source = _source()
    prepare = _between(
        source,
        "SpmPipelineComponentLayout prepare_pooler_z1_layout(",
        "void prime_pooler_z1_inputs(",
    )
    validate = _between(
        source,
        "void validate_pooler_z1_layout(",
        "void clear_pooler_z1_layout(",
    )
    clear = _between(
        source,
        "void clear_pooler_z1_layout(",
        "void stage_pooler_z1_outer_fast_component(",
    )
    outer_stage = _between(
        source,
        "void stage_pooler_z1_outer_fast_component(",
        "void bind_pooler_z1_outer_fast_input(",
    )
    outer_bind = _between(
        source,
        "void bind_pooler_z1_outer_fast_input(",
        "void forward_pooler_z1(",
    )
    z1_forward = _between(
        source,
        "void forward_pooler_z1(",
        "void check_pooler_z1_destroy_allowed() const",
    )
    destroy = _between(
        source,
        "void check_pooler_z1_destroy_allowed() const",
        "// Retrieve deepstack snapshots",
    )
    wrappers = source[source.index("namespace v3::qwen3vl_pooler_z1_internal {") :]

    assert "retained_count == 0 || retained_count == 1 ||" in prepare
    assert "retained_count == 3" in prepare
    assert "retained_deepstack_bindings_complete()" in validate
    assert "z1_retained_deepstack_source_addrs_[prior] !=" in validate
    assert "z1_retained_deepstack_source_addrs_.fill(0)" in clear
    assert "z1_retained_deepstack_bound_count_ = 0" in clear
    assert "retained_deepstack_bindings_complete()" in outer_stage
    assert "retained_deepstack_bindings_complete()" in outer_bind
    assert "retained_deepstack_bindings_complete()" in z1_forward
    assert "z1_retained_deepstack_count_ == 0" in destroy
    assert "retained_deepstack_addresses_empty()" in destroy
    assert "int64_t retained_count" in _between(
        wrappers, "SpmPipelineComponentLayout prepare_vision(", "void prime_vision_inputs("
    )
    assert "/*ordinal=*/0" in _between(
        wrappers,
        "SpmDense2DSpec vision_deepstack1_source_spec(",
        "void adopt_vision(",
    )
    assert "/*ordinal=*/0" in _between(
        wrappers,
        "void bind_vision_deepstack1_source(",
        "void validate_vision(",
    )


def test_deepstack_merger_uses_hf_postshuffle_layernorm_geometry():
    source = _source()
    declarations = _between(
        source,
        "std::vector<BufferDecl> declare_buffers(",
        "void emit_preload_weights()",
    )
    prefix = _between(
        source, "void emit_one_merger_prefix(", "void emit_one_merger("
    )
    emit = _between(
        source, "void emit_one_merger(", "SpmDense2DSpec multiview_yield_spec()"
    )
    layer = _between(
        source,
        "void build_layer_subgraph(",
        "bool merger_active() const",
    )
    ordinary = _between(source, "void merger_post_fn()", "void pooler_z1_post_fn()")
    profile = _between(
        source,
        "void validate_pooler_z1_retained_deepstack_profile(",
        "void validate_pooler_z1_contract() const",
    )

    assert "deepstack_merger_active() ? merge_hidden_ : h" in declarations
    assert '"merger_ln_q_w",  merger_norm_w_sz' in declarations
    assert '"merger_ln_q_b",  merger_norm_w_sz' in declarations
    assert "bool postshuffle_norm = false" in prefix
    assert "norm_rows = postshuffle_norm ? mrows : seq" in prefix
    assert "norm_hidden = postshuffle_norm ? mh : h" in prefix
    assert prefix.count("norm_hidden") >= 6
    assert "norm_rows, norm_hidden" in prefix
    assert "emit_one_merger_prefix(" in emit
    assert "/*postshuffle_norm=*/true" in layer
    assert "/*postshuffle_norm=*/true" in ordinary
    assert 'ds_merger_ln_q_w_[ordinal], "ds_ln_q_w", ordinal, mh' in profile
    assert 'ds_merger_ln_q_b_[ordinal], "ds_ln_q_b", ordinal, mh' in profile


def test_destroy_guard_is_called_before_registry_destroy():
    source = _source()
    destroy = _between(
        source,
        "void rpu_qwen3vl_vision_destroy(",
        "void rpu_qwen3vl_vision_set_weights(",
    )
    assert destroy.index("check_pooler_z1_destroy_allowed()") < destroy.index(
        "Qwen3VLVisionRegistry::destroy("
    )
    assert "z1_prepared_num_patches_ == 0" in source
    assert "z1_retained_deepstack_count_ == 0" in source
    assert "z1_retained_deepstack_bound_count_ == 0" in source
    assert "retained_deepstack_addresses_empty()" in source


def test_python_ordinary_fused_merger_fallback_still_pops_and_clones():
    adapter = VISION_ADAPTER.read_text(encoding="utf-8")
    fallback = _between(
        adapter,
        "if _group_fused_merger:",
        "else:\n            if not _adaptive_merger:",
    )
    pop = fallback.index("qwen3vl_vision_pop_merged(handle)")
    pooler_clone = fallback.index("merged_all[0].clone()")
    deepstack_clone = fallback.index("merged_all[1 + layer_k].clone()")
    assert pop < pooler_clone < deepstack_clone
