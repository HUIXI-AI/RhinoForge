"""Board-free source contract for the S225 multiview Text consumer."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_CPP = ROOT / "src/fused/rpu_qwen3_model.cpp"
TEXT_HEADER = ROOT / "src/fused/rpu_qwen3_model.h"
FMB_CPP = ROOT / "src/core/fused_model_base.cpp"
FMB_IMPL = ROOT / "src/core/fused_model_base_impl.h"
REGISTRATIONS = ROOT / "src/core/rpu_dispatch_registrations.inc"


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_live_profile_is_exact_and_capabilities_are_composite_only():
    cpp = TEXT_CPP.read_text(encoding="utf-8")
    header = TEXT_HEADER.read_text(encoding="utf-8")
    prepare = _between(
        cpp,
        "CausalDecoderModel::prepare_qwen3vl_multiview_text_composite_layout(",
        "CausalDecoderModel::resolve_qwen3vl_multiview_text_composite_profile()",
    )
    resolve = _between(
        cpp,
        "CausalDecoderModel::resolve_qwen3vl_multiview_text_composite_profile()",
        "CausalDecoderModel::seal_qwen3vl_multiview_text_composite_manifest(",
    )
    for token in (
        "shape.resolved_chunk_size ==",
        "QWEN3VL_MULTIVIEW_TEXT_RESOLVED_CHUNK_SIZE",
        "shape.chunks.size() == 1",
        "shape.chunks.front().len == execution_len",
        "shape.kv_insert_chunks.size() == 1",
        "shape.allocation_layout.chunk_size ==",
        "QWEN3VL_MULTIVIEW_TEXT_ALLOCATION_CHUNK_SIZE",
        "shape.chunk_mode == ChunkMode::SEQUENTIAL",
        "shape.inter_layer_io == InterLayerIO::SPM_RESIDENT",
        "QWEN3VL_MULTIVIEW_TEXT_TEMPORARY_BYTES",
    ):
        assert token in prepare
    assert "set_chunk_size_override(225)" not in prepare
    assert "request.chunks =" in resolve
    assert "request.kv_insert_chunks" not in resolve
    assert "4163072" in header
    assert "UINT64_C(0x51564d5654533236)" in header
    capability = _between(
        header,
        "SpmFmbTraversalCapability spm_fmb_traversal_capability()",
        "ModelStaticConfig static_config() override",
    )
    assert capability.count("qwen3vl_multiview_text_composite_prepared_") == 2
    assert "CanonicalTraversal" in capability
    assert "PersistentOutsideResolvedWindow" in capability
    assert "spm_fmb_layer_producer_yield_capability" not in capability
    assert "spm_fmb_composite_occurrence_capability" not in capability


def test_composite_down_output_reuses_dead_up_slot_only_in_exact_lifecycle():
    header = TEXT_HEADER.read_text(encoding="utf-8")
    declarations = _between(
        header,
        "// MLP (phase 7-8, aliased with attention)",
        "// ★ Phase 2.5: decode-only fused lm_head output buffer.",
    )
    assert "qwen3vl_multiview_text_dry_prepared_ ||" in declarations
    assert "qwen3vl_multiview_text_composite_prepared_" in declarations
    assert ("const int down_phase_start = (exact_multiview_text_layout || "
            "lingbot_c400_full_layout) ? 8 : 7;") in " ".join(declarations.split())
    assert "lingbot_prefill_deepstack_reuses_down(ctx) && ctx.chunk_size == 400" in declarations
    assert "if (!wall_p640_staged_layout && !(qwen_inplace && qwen3_c256_))" in declarations
    assert '{"up",     mlp,      7, 7' in declarations
    assert '{"down",       down_sz, down_phase_start, 8' in declarations
    assert "aliases full `up` [7,7] with `down` [8,8]" in declarations


def test_mlp_operation_order_proves_up_to_down_handoff():
    fmb = FMB_CPP.read_text(encoding="utf-8")
    body = _between(
        fmb,
        "void FusedModelBase::emit_mlp_pipeline(",
        "namespace {\n\nstruct FmbExecutionTraversalSpec",
    )
    up_write = body.index(
        'addr(0, "residual1"), up_w, addr(0, "up")'
    )
    up_last_read = body.index(
        'addr(0, "gate"), addr(0, "up"), addr(0, "gate")'
    )
    down_first_write = body.index(
        'addr(0, "gate"), down_w, addr(0, "down")'
    )
    down_last_read = body.rindex(
        'addr(0, "down"), residual, addr(0, "residual1")'
    )
    assert up_write < up_last_read < down_first_write < down_last_read
    assert 'addr(0, "up")' not in body[down_first_write:]
    # Wall/G05 mixed precision adds mutually exclusive BF16-output,
    # BF16-residual, and FP16-residual consumers.  All variants must still
    # begin only after `up`'s final read, so the composite up->down alias is
    # safe regardless of the selected precision path.
    down_uses = [
        index for index in range(len(body))
        if body.startswith('addr(0, "down")', index)
    ]
    # Other core topologies also prepare the Down ring input after Mul.
    # Every read/write must remain after Up's last use, independent of the
    # number of mutually exclusive arithmetic and core-transfer branches.
    assert down_uses
    assert all(index > up_last_read for index in down_uses)


def test_layer0_ingress_is_exact_four_run_complement_before_rmsnorm():
    cpp = TEXT_CPP.read_text(encoding="utf-8")
    header = TEXT_HEADER.read_text(encoding="utf-8")
    fmb = FMB_CPP.read_text(encoding="utf-8")
    assert "std::array<int64_t, 2>{0, 1}" in header
    assert "std::array<int64_t, 2>{65, 2}" in header
    assert "std::array<int64_t, 2>{131, 2}" in header
    assert "std::array<int64_t, 2>{197, 28}" in header
    ingress = _between(
        cpp,
        "void CausalDecoderModel::emit_qwen3vl_multiview_text_complement_ingress(",
        "void CausalDecoderModel::emit_qwen3vl_multiview_text_retained_deepstack_add(",
    )
    assert "chunk.idx == 0 && chunk.offset == 0" in ingress
    assert "emit_layer_input_row_run_dma(" in ingress
    assert "emit_layer_input_dma(" not in ingress
    row_dma = _between(
        fmb,
        "void FusedModelBase::emit_layer_input_row_run_dma(",
        "void FusedModelBase::emit_layer_output_dma(",
    )
    assert "layer_idx == 0" in row_dma
    assert "current_hidden_in_src_base(), layer_idx" in row_dma
    assert "live_src_base = current_hidden_in_src_base();" in row_dma
    assert "live_src_base, *pimpl_->ctx_.hidden_states" in row_dma
    assert "rpu_launch_ddr_broadcast_spm_dma_mutable(" in row_dma
    sequential = _between(
        header,
        "// Input DMA (if input is not already in SPM)",
        "// Phase 1: Input RMSNorm",
    )
    assert sequential.index("emit_qwen3vl_multiview_text_complement_ingress(") < (
        sequential.index("emit_layer_input_dma(")
    )


def test_runadd_is_nine_existing_adds_at_post_mlp_injection_point():
    cpp = TEXT_CPP.read_text(encoding="utf-8")
    header = TEXT_HEADER.read_text(encoding="utf-8")
    fmb = FMB_CPP.read_text(encoding="utf-8")
    runadd = _between(
        cpp,
        "void CausalDecoderModel::emit_qwen3vl_multiview_text_retained_deepstack_add(",
        "SpmPipelineCausalPrefillDryLayout",
    )
    assert "layer_idx < 0 || layer_idx > 2" in runadd
    assert "for (size_t image = 0; image < 3; ++image)" in runadd
    assert "QWEN3VL_MULTIVIEW_TEXT_IMAGE_ROW_BEGINS[image]" in runadd
    assert "emit_spm_pipeline_composite_run_add(" in runadd
    assert "rpu_launch_eltwise_binary_spm_kernel(" not in runadd
    assert "runadd_addr" not in runadd.lower()
    assert "rpu_launch_ddr_" not in runadd
    assert "rpu_launch_linear" not in runadd

    generic_runadd = _between(
        fmb,
        "void FusedModelBase::emit_spm_pipeline_composite_run_add(",
        "void FusedModelBase::emit_layer_output_dma(",
    )
    for token in (
        "require_spm_pipeline_composite_consumer_physical_authority();",
        "resolve_composite_run_add_addrs(",
        "target.consumer_layer == layer_idx",
        "target.consumer_row_begin == global_row_begin",
        "target.consumer_row_count == row_count",
        "target.consumer_addr == residual_base + dst_offset",
        "rpu_launch_eltwise_binary_spm_kernel(",
        "ValuOpType::ADD",
        "c10::Half(1.0f)",
        "++runtime.next_target",
    ):
        assert token in generic_runadd
    assert "rpu_launch_ddr_" not in generic_runadd
    assert "rpu_launch_linear" not in generic_runadd
    # Both layer-body implementations route composite RunAdd ahead of legacy
    # retained-Z1 and dense-DDR DeepStack branches.
    assert header.count(
        "if (qwen3vl_multiview_text_composite_dispatch_) {\n"
        "                emit_qwen3vl_multiview_text_retained_deepstack_add("
    ) == 1
    assert header.count(
        "if (qwen3vl_multiview_text_composite_dispatch_) {\n"
        "            emit_qwen3vl_multiview_text_retained_deepstack_add("
    ) == 1


def test_missing_lexical_physical_authority_fails_before_side_effects():
    cpp = TEXT_CPP.read_text(encoding="utf-8")
    forward = _between(
        cpp,
        "at::Tensor CausalDecoderModel::forward_qwen3vl_multiview_text_composite(",
        "void CausalDecoderModel::emit_qwen3vl_multiview_text_complement_ingress(",
    )
    authority_gate = forward.index(
        "require_spm_pipeline_composite_consumer_physical_authority()"
    )
    graph_gate = forward.index("RpuKernelGraph::has_active()")
    dispatch = forward.index("qwen3vl_multiview_text_composite_dispatch_ = true")
    nested_forward = forward.index("at::Tensor result = forward(")
    assert authority_gate < graph_gate < dispatch < nested_forward


def test_physical_execution_keeps_only_lease_identity_and_exact_cursor():
    cpp = FMB_CPP.read_text(encoding="utf-8")
    impl = FMB_IMPL.read_text(encoding="utf-8")
    header = TEXT_HEADER.read_text(encoding="utf-8")

    state = _between(
        impl,
        "struct CompositePhysicalExecution {",
        "struct RuntimeDenseMemberDescriptor {",
    )
    for token in (
        "const SpmPipelineLease* lease",
        "uint64_t lease_epoch",
        "uint64_t plan_hash",
        "size_t next_target",
        "size_t expected_target_count",
        "const void* graph_identity",
    ):
        assert token in state
    assert "std::array" not in state
    assert "std::vector" not in state
    assert "payload_addr" not in state
    assert "consumer_addr" not in state

    for stale_surface in (
        "qwen3vl_multiview_text_runadd_addrs_",
        "qwen3vl_multiview_text_runadd_bound_count_",
        "qwen3vl_multiview_text_runadd_bindings_empty",
        "qwen3vl_multiview_text_runadd_bindings_complete",
    ):
        assert stale_surface not in cpp
        assert stale_surface not in header

    arm = _between(
        cpp,
        "void FusedModelBase::arm_spm_pipeline_composite_physical_execution(",
        "void FusedModelBase::finish_spm_pipeline_composite_physical_execution(",
    )
    assert "resolve_composite_producer_target_addr(" in arm
    assert "resolve_composite_run_add_addrs(" in arm
    assert "preflight_count == yields_per_producer" in arm
    assert "preflight_count == run_add_count" in arm


def test_committed_build_alone_requests_retained_physical_arena():
    cpp = FMB_CPP.read_text(encoding="utf-8")
    begin = _between(
        cpp,
        "void SpmCompositeTraceCoordinator::begin_build(\n"
        "    RpuKernelGraph& graph,\n"
        "    const SpmPipelineLease& lease,",
        "SpmCompositeOccurrenceScope\n"
        "SpmCompositeTraceCoordinator::begin_build_occurrence(",
    )
    assert "lease.composite_committed() == require_committed" in begin
    assert "if (require_committed)" in begin
    assert "graph.request_retained_physical_arena_build();" in begin

    release = _between(
        cpp,
        "void SpmCompositeTraceCoordinator::release_physical_components(",
        "void SpmCompositeTraceCoordinator::begin_build(RpuKernelGraph& graph)",
    )
    assert "lease.composite_committed()" in release
    assert release.count(
        "validate_spm_pipeline_composite_component("
    ) == 2
    assert "validate_spm_pipeline_component(" not in release

    validate_owner = _between(
        cpp,
        "void FusedModelBase::validate_spm_pipeline_composite_component(",
        "void FusedModelBase::validate_spm_pipeline_component(",
    )
    assert "validate_composite_execution_authority(" in validate_owner
    assert "resolve_composite_component_placement(producer)" in validate_owner
    assert "lease.validate_physical_range(" in validate_owner
    assert "require_sealed" not in validate_owner


def test_live_surface_is_internal_and_does_not_register_a_new_operator():
    registrations = REGISTRATIONS.read_text(encoding="utf-8")
    for name in (
        "prepare_multiview_text_composite",
        "resolve_multiview_text_profile",
        "seal_multiview_text_consumer",
        "forward_multiview_text_composite",
    ):
        assert name not in registrations
