"""Board-free source gates for the Qwen3-VL pooler-only Z1 lifecycle."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.cpp"
HEADER = ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.h"
CMAKE = ROOT / "src/CMakeLists.txt"
REGISTRATIONS = ROOT / "src/core/rpu_dispatch_registrations.inc"
KERNEL_DECLS = ROOT / "src/core/rpu_kernel_decls.h"
MODEL = ROOT / "src/fused/rpu_qwen3_model.cpp"


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]



def _assert_current_profile(name):
    policies = (ROOT / "src/fused/rpu_parallel_linear_register_policies.h").read_text()
    assert f"inline const GraphKernelRegisterPolicy& {name}()" in policies
    block = _between(policies, f"inline const GraphKernelRegisterPolicy& {name}()", "return value;")
    assert "p.fingerprint = UINT64_C(" in block and "rule(" in block
    source = COORDINATOR.read_text()
    assert f'return rpu_pl_census::policy("{name}");' in source


def test_exact_n256_geometry_and_budget_are_frozen():
    source = COORDINATOR.read_text(encoding="utf-8")
    for token in (
        "kNumPatches = 256",
        "kMergedRows = 64",
        "kExecutionLen = 128",
        "kImageRowBegin = 4",
        "kRealLen = 72",
        "kGrootRealLen = 82",
        "kTextHidden = 2048",
        "kComponentTemporaryBytes = 1'966'080",
        "kExpectedDynamicPeakBytes = 2'490'368",
        "kExpectedDeepstack1DynamicPeakBytes = 4'194'304",
        "kExpectedDeepstack3DynamicPeakBytes = 4'718'592",
        "kDeepstackPorts",
        "v3::SpmPortId{3}, v3::SpmPortId{4}, v3::SpmPortId{5}",
        "SpmAllocator::SPM_PLANNING_BUDGET",
    ):
        assert token in source
    assert "7.5" not in source


def test_route_is_identity_and_lifecycle_separates_three_events():
    source = COORDINATOR.read_text(encoding="utf-8")
    route = _between(
        source,
        "v3::SpmPermutationRowRunsRoute endpoint_route(",
        "v3::SpmPermutationRowRunsRoute endpoint_route(\n"
        "    const Qwen3VlPoolerZ1State& s)",
    )
    assert "identity_permutation()" in route
    assert "SpmDstRowRun{image_row_begin, kMergedRows}" in route
    assert route.index("source_first_write=*/0") < route.index(
        "destination_first_write=*/1"
    )
    assert route.index("destination_first_write=*/1") < route.index(
        "destination_last_read=*/2"
    )

    prepare = _between(
        source,
        "int64_t rpu_qwen3vl_pooler_spm_z1_prepare(",
        "void rpu_qwen3vl_pooler_spm_z1_unprepare(",
    )
    vision = prepare.index("prepare_vision(")
    text = prepare.index("prepare_text(")
    route_build = prepare.index("const auto route = endpoint_route(")
    compile_plan = prepare.index("SpmPipelinePlan::compile(")
    commit = prepare.index("s.vision_handle = vision_handle")
    validate = prepare.index("validate_prepared_locked(", commit)
    assert vision < text < route_build < compile_plan < commit < validate


def test_retained_deepstack_routes_and_prepare_cardinality_are_fail_closed():
    source = COORDINATOR.read_text(encoding="utf-8")
    retained = _between(
        source,
        "v3::SpmIdentityRoute deepstack_route(",
        "std::vector<v3::SpmIdentityRoute> deepstack_routes(",
    )
    assert "vision_deepstack_source_spec(" in retained
    assert "text_deepstack_spec(" in retained
    assert "produced == required" in retained
    assert "kDeepstackPorts.at(static_cast<size_t>(ordinal))" in retained
    assert "/*first_write=*/0, /*last_read=*/2" in retained

    prepare = _between(
        source,
        "int64_t rpu_qwen3vl_pooler_spm_z1_prepare(",
        "void rpu_qwen3vl_pooler_spm_z1_unprepare(",
    )
    assert "int64_t retained_deepstack_count" in prepare
    assert "s.retained_deepstack_count == retained_deepstack_count" in prepare
    assert "check_retained_deepstack_count(" in prepare
    assert "vision_handle, num_patches, retained_deepstack_count" in prepare
    assert "prepare_text_deepstack1(" in prepare
    assert "prepare_text_deepstack3(" in prepare
    assert "const auto retained = deepstack_routes(" in prepare
    assert "scratch, retained" in prepare
    assert "std::vector<v3::SpmPermutationRowRunsRoute>{route}" in prepare
    assert "kExpectedDeepstack1DynamicPeakBytes" in prepare
    assert "kExpectedDeepstack3DynamicPeakBytes" in prepare
    assert prepare.index(
        "s.retained_deepstack_count = retained_deepstack_count"
    ) < prepare.index("s.plan = std::move(plan)")

    header = HEADER.read_text(encoding="utf-8")
    assert "int64_t retained_deepstack_count = 0" in header
    registrations = REGISTRATIONS.read_text(encoding="utf-8")
    assert ("int retained_deepstack_count=0, int[] vision_stage_descriptor=[], "
            "int[] text_stage_descriptor=[], bool partial_mrope=False) -> int") in registrations


def test_prepare_failure_and_unprepare_are_reverse_component_transactions():
    source = COORDINATOR.read_text(encoding="utf-8")
    prepare = _between(
        source,
        "int64_t rpu_qwen3vl_pooler_spm_z1_prepare(",
        "void rpu_qwen3vl_pooler_spm_z1_unprepare(",
    )
    rollback_text = prepare.index("unprepare_text(")
    rollback_vision = prepare.index("unprepare_vision(")
    reset = prepare.index("reset_prepared_state_locked(s)", rollback_vision)
    assert rollback_text < rollback_vision < reset

    unprepare = _between(
        source,
        "void rpu_qwen3vl_pooler_spm_z1_unprepare(",
        "std::vector<int64_t> rpu_qwen3vl_pooler_spm_z1_plan_stats(",
    )
    assert unprepare.index("unprepare_text(") < unprepare.index(
        "unprepare_vision("
    )
    assert unprepare.index("unprepare_vision(") < unprepare.index(
        "reset_prepared_state_locked(s)"
    )


def test_physical_begin_is_one_prime_to_post_seal_transaction():
    source = COORDINATOR.read_text(encoding="utf-8")
    begin = _between(
        source,
        "int64_t rpu_qwen3vl_pooler_spm_z1_begin(",
        "void rpu_qwen3vl_pooler_spm_z1_end(",
    )
    transaction = begin.index("try {")
    vision_prime = begin.index("prime_vision_inputs(")
    text_prime = begin.index("prime_text_inputs(")
    acquire = begin.index("SpmPipelineLease::acquire_physical(")
    adopt_vision = begin.index("adopt_vision(")
    adopt_text = begin.index("adopt_text(")
    bind_vision = begin.index("bind_vision_source(")
    bind_text = begin.index("bind_text_destination(")
    seal = begin.index("seal_physical();")
    resolve = begin.index("resolve_physical_addr(")
    lower = begin.index("route.lower_contiguous_runs()")
    rollback_text = begin.index("rollback_text_inputs(")
    rollback_vision = begin.index("rollback_vision_inputs(")
    release = begin.index("lease->release()")
    commit = begin.index("s.source_addr = source_addr")

    assert "TORCH_CHECK(!RpuKernelGraph::has_active()" in begin
    assert transaction < vision_prime < text_prime < acquire
    assert acquire < adopt_vision < adopt_text < bind_vision < bind_text < seal
    assert seal < resolve < lower < rollback_text < rollback_vision < release < commit
    assert "lowered_runs.size() == 1" in begin
    assert "source_row == 0" in begin
    assert "destination_row == kImageRowBegin" in begin
    assert "row_count == kMergedRows" in begin


def test_retained_deepstack_ports_are_pairwise_disjoint_and_bound_by_ordinal():
    source = COORDINATOR.read_text(encoding="utf-8")
    begin = _between(
        source,
        "int64_t rpu_qwen3vl_pooler_spm_z1_begin(",
        "void rpu_qwen3vl_pooler_spm_z1_end(",
    )
    retained = begin.index("const auto retained = lease->port(kDeepstackPorts[index])")
    for peer in (
        "vision_scratch_offset",
        "source_offset",
        "destination_offset",
        "text_scratch_offset",
    ):
        assert peer in begin[retained:]
    assert "for (size_t prior = 0; prior < index; ++prior)" in begin[retained:]
    assert '"overlap at ordinals "' in begin[retained:]
    bind_vision = begin.index("bind_vision_deepstack_source(", retained)
    bind_text = begin.index("bind_text_deepstack(", bind_vision)
    seal = begin.index("lease->seal_physical();", bind_text)
    resolve = begin.index("deepstack_addrs[index] = retained.resolve_physical_addr(", seal)
    commit = begin.index("s.deepstack_addrs = deepstack_addrs", seal)
    assert retained < bind_vision < bind_text < seal < resolve < commit


def test_active_lease_guards_prepare_unprepare_and_end_is_reverse_cleanup():
    source = COORDINATOR.read_text(encoding="utf-8")
    prepare = _between(
        source,
        "int64_t rpu_qwen3vl_pooler_spm_z1_prepare(",
        "void rpu_qwen3vl_pooler_spm_z1_unprepare(",
    )
    unprepare = _between(
        source,
        "void rpu_qwen3vl_pooler_spm_z1_unprepare(",
        "std::vector<int64_t> rpu_qwen3vl_pooler_spm_z1_plan_stats(",
    )
    end = _between(
        source,
        "void rpu_qwen3vl_pooler_spm_z1_end(",
        "void rpu_qwen3vl_pooler_spm_z1_abort(",
    )
    assert 'TORCH_CHECK(!s.lease' in prepare
    assert 'TORCH_CHECK(!s.lease' in unprepare
    identity = end.index("validate_physical_identity(")
    clear_text = end.index("clear_text(")
    clear_vision = end.index("clear_vision(")
    release = end.index("s.lease->release()")
    reset = end.index("reset_begin_state_locked(s)")
    assert "TORCH_CHECK(!RpuKernelGraph::has_active()" in end
    assert identity < clear_text < clear_vision < release < reset


def test_end_and_abort_keep_consumed_and_unconsumed_cleanup_separate():
    source = COORDINATOR.read_text(encoding="utf-8")
    end = _between(
        source,
        "void rpu_qwen3vl_pooler_spm_z1_end(",
        "void rpu_qwen3vl_pooler_spm_z1_abort(",
    )
    abort = _between(
        source,
        "void rpu_qwen3vl_pooler_spm_z1_abort(",
        "void rpu_qwen3vl_pooler_spm_z1_cancel(",
    )

    assert "*s.invocation_consumed_generation ==" in end
    assert "s.active_invocation_generation" in end
    assert "*s.invocation_consumed_generation <" not in end
    assert "TORCH_CHECK(!RpuKernelGraph::has_active()" in abort
    identity = abort.index("validate_physical_identity(")
    unconsumed = abort.index("*s.invocation_consumed_generation <")
    clear_text = abort.index("clear_text(")
    clear_vision = abort.index("clear_vision(")
    release = abort.index("s.lease->release()")
    retire = abort.index(
        "*s.invocation_consumed_generation = "
        "s.active_invocation_generation"
    )
    reset = abort.index("reset_begin_state_locked(s)")
    assert (
        identity < unconsumed < clear_text < clear_vision < release
        < retire < reset
    )
    assert abort.count("cleanup_step(failure") == 3
    assert "successfully consumed invocations must use end" in abort


def test_cancel_reads_authority_and_cleans_consumed_or_unconsumed_invocation():
    source = COORDINATOR.read_text(encoding="utf-8")
    cancel = _between(
        source,
        "void rpu_qwen3vl_pooler_spm_z1_cancel(",
        "at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward(",
    )

    assert "TORCH_CHECK(!RpuKernelGraph::has_active()" in cancel
    identity = cancel.index("validate_physical_identity(")
    active = cancel.index(
        "const uint64_t active_generation = s.active_invocation_generation"
    )
    prepared = cancel.index(
        "const uint64_t prepared_generation =\n"
        "        *s.invocation_prepared_generation"
    )
    consumed = cancel.index(
        "const uint64_t consumed_generation =\n"
        "        *s.invocation_consumed_generation"
    )
    authority = cancel.index("consumed_generation <= active_generation")
    unconsumed = cancel.index("consumed_generation < active_generation")
    clear_text = cancel.index("clear_text(")
    clear_vision = cancel.index("clear_vision(")
    release = cancel.index("s.lease->release()")
    retirement_guard = cancel.index("if (retire_unconsumed_generation)")
    retire = cancel.index(
        "*s.invocation_consumed_generation = active_generation",
        retirement_guard,
    )
    reset = cancel.index("reset_begin_state_locked(s)")
    assert identity < active < prepared < consumed < authority < unconsumed
    assert unconsumed < clear_text < clear_vision < release
    assert release < retirement_guard < retire < reset
    assert cancel.count("cleanup_step(failure") == 3

    # The strict success/abort APIs remain disjoint; cancel is the only API
    # that inspects and accepts either consumed authority state.
    end = _between(
        source,
        "void rpu_qwen3vl_pooler_spm_z1_end(",
        "void rpu_qwen3vl_pooler_spm_z1_abort(",
    )
    abort = _between(
        source,
        "void rpu_qwen3vl_pooler_spm_z1_abort(",
        "void rpu_qwen3vl_pooler_spm_z1_cancel(",
    )
    assert "*s.invocation_consumed_generation ==" in end
    assert "*s.invocation_consumed_generation <" in abort
    assert "consumed_generation <= active_generation" not in end
    assert "consumed_generation <= active_generation" not in abort


def test_owned_hidden_ingress_stages_prefix_suffix_and_pad_after_vision():
    source = COORDINATOR.read_text(encoding="utf-8")
    stage = _between(
        source,
        "void stage_non_image_rows_locked(",
        "void emit_identity_route_locked(",
    )
    assert stage.count("rpu_launch_ddr_broadcast_spm_dma_mutable(") == 2
    assert "s.hidden_prefix_src_base,\n        hidden," in stage
    assert "s.hidden_suffix_src_base,\n        hidden," in stage
    assert "s.hidden_prefix_src_base = hidden_dev_addr" in stage
    assert "s.hidden_suffix_src_base = hidden_dev_addr" in stage
    assert "s.hidden_ref = hidden" in stage
    assert "image_end * row_bytes" in stage
    assert "suffix_rows * kTextHidden" in stage
    assert "rpu_launch_fill_spm_kernel(" in stage
    assert "pad_rows * kTextHidden" in stage

    pipeline = source[
        source.index("at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward("):
    ]
    vision = pipeline.index("forward_vision_z1(")
    stage_rows = pipeline.index("stage_non_image_rows_locked(")
    route = pipeline.index("emit_identity_route_locked(")
    text = pipeline.index("forward_text_z1(")
    assert vision < stage_rows < route < text
    assert "TORCH_CHECK(RpuKernelGraph::has_active()" in pipeline
    assert "FlushCaptureModeGuard" not in pipeline


def test_identity_run_emits_exactly_two_existing_32_row_slices():
    source = COORDINATOR.read_text(encoding="utf-8")
    emitter = _between(
        source,
        "void emit_identity_route_locked(",
        "\n}  // namespace",
    )
    assert "kRouteHalfRows = 32" in source
    assert "s.lowered_runs.size() == 1" in emitter
    assert emitter.count("rpu_launch_slice_spm_kernel(") == 2
    assert "half_shape" in emitter
    assert "run.source_row + kRouteHalfRows" in emitter
    assert "run.destination_row + kRouteHalfRows" in emitter


def test_physical_lifecycle_is_declared_and_registered_without_export():
    source = COORDINATOR.read_text(encoding="utf-8")
    assert "SpmPipelineLease::acquire_physical" in source
    assert "rpu_qwen3vl_pooler_spm_z1_begin" in source
    assert "rpu_qwen3vl_pooler_spm_z1_end" in source
    assert "rpu_qwen3vl_pooler_spm_z1_abort" in source
    assert "rpu_qwen3vl_pooler_spm_z1_cancel" in source
    assert "rpu_qwen3vl_pooler_spm_z1_pipeline_forward" in source
    assert "qwen3vl_pooler_spm_z1_export" not in source

    header = HEADER.read_text(encoding="utf-8")
    assert "namespace v3::qwen3vl_pooler_z1_internal" in header
    assert "rpu_qwen3vl_pooler_spm_z1_plan_stats" in header
    assert "rpu_qwen3vl_pooler_spm_z1_begin" in header
    assert "rpu_qwen3vl_pooler_spm_z1_end" in header
    assert "rpu_qwen3vl_pooler_spm_z1_abort" in header
    assert "rpu_qwen3vl_pooler_spm_z1_cancel" in header
    kernel_decls = KERNEL_DECLS.read_text(encoding="utf-8")
    assert "rpu_qwen3vl_pooler_spm_z1_abort" in kernel_decls
    assert "rpu_qwen3vl_pooler_spm_z1_cancel" in kernel_decls
    assert "rpu_qwen3vl_pooler_spm_z1_pipeline_forward" in header
    assert "rpu_qwen3vl_pooler_spm_z1.cpp" in CMAKE.read_text(encoding="utf-8")
    registrations = REGISTRATIONS.read_text(encoding="utf-8")
    for op in (
        "qwen3vl_pooler_spm_z1_prepare",
        "qwen3vl_pooler_spm_z1_unprepare",
        "qwen3vl_pooler_spm_z1_plan_stats",
        "qwen3vl_pooler_spm_z1_begin",
        "qwen3vl_pooler_spm_z1_end",
        "qwen3vl_pooler_spm_z1_abort",
        "qwen3vl_pooler_spm_z1_cancel",
        "qwen3vl_pooler_spm_z1_pipeline_forward",
    ):
        assert registrations.count(f'm.def("{op}') == 1
    assert registrations.count(
        'm.impl("qwen3vl_pooler_spm_z1_pipeline_forward"'
    ) == 2
    assert (
        "int plan_hash, Tensor? rope_cos_il=None, "
        "Tensor? rope_sin_il=None) -> int"
    ) in registrations


def test_exact_typed_register_policy_and_phase_boundaries_are_frozen():
    source = COORDINATOR.read_text(encoding="utf-8")
    _assert_current_profile("qwen3vl_pooler_z1_register_policy")

    pipeline = source[
        source.index("at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward("):
    ]
    guard = pipeline.index("std::make_unique<GraphKernelRegisterCensusGuard>(")
    vision = pipeline.index("forward_vision_z1(", guard)
    vision_checkpoint = pipeline.index(
        'register_census->checkpoint("vision")', vision
    )
    handoff = pipeline.index("stage_non_image_rows_locked(", vision_checkpoint)
    route = pipeline.index("emit_identity_route_locked(", handoff)
    handoff_checkpoint = pipeline.index(
        'register_census->checkpoint("handoff")', route
    )
    text = pipeline.index("forward_text_z1(", handoff_checkpoint)
    complete = pipeline.index("register_census->complete()", text)
    assert guard < vision < vision_checkpoint < handoff < route
    assert route < handoff_checkpoint < text < complete


def test_z1_selected_descriptors_are_cold_authority_not_a_new_executor():
    source = COORDINATOR.read_text(encoding="utf-8")
    prepare = _between(source, "int64_t rpu_qwen3vl_pooler_spm_z1_prepare(",
                       "void rpu_qwen3vl_pooler_spm_z1_unprepare(")
    assert "prepared winner changed; unprepare first" in prepare
    assert prepare.count('.kvinsert_cost_domain(') == 2
    assert "retained_deepstack_count, vision_stage_descriptor" in prepare
    assert prepare.count("execution_len, real_len, text_stage_descriptor") == 3
    assert "s.plan_hash = plan.hash()" in prepare  # remains the physical layout hash
    begin = _between(source, "int64_t rpu_qwen3vl_pooler_spm_z1_begin(",
                     "void rpu_qwen3vl_pooler_spm_z1_end(")
    assert begin.index('.kvinsert_cost_domain(') < begin.index("prime_vision_inputs(")
    pipeline = source[source.index("at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward("):]
    assert "KvInsertRoute::ALIGNED_V16" in pipeline
    assert "if (use_register_reuse)" in pipeline
    assert "if (register_census && RpuKernelGraph::active().state()" in pipeline
    assert "if (!register_census) *s.invocation_consumed_generation = invocation_generation" in pipeline
    assert "s.vision_stage_descriptor, &s.text_stage_descriptor" in source

    # The two namespaces expose identically named facades. Only Qwen Z1
    # gains the optional winner; Wall's independent three-argument ABI stays.
    models = MODEL.read_text(encoding="utf-8")
    wall = _between(models, "namespace v3::wall_oss_z1_internal {",
                    "}  // namespace v3::wall_oss_z1_internal")
    qwen = models[models.index("namespace v3::qwen3vl_pooler_z1_internal {"):]
    wall_prepare = _between(wall, "SpmPipelineComponentLayout prepare_text(",
                            "SpmDense2DSpec text_destination_spec(")
    qwen_prepare = _between(qwen, "SpmPipelineComponentLayout prepare_text(",
                            "SpmPipelineComponentLayout prepare_text_deepstack1(")
    assert "selected_descriptor" not in wall_prepare
    assert "at::IntArrayRef selected_descriptor," in qwen_prepare
    assert "bool partial_mrope)" in qwen_prepare
    assert "/*retained_deepstack_count=*/0, selected_descriptor" in qwen_prepare


def test_deepstack1_typed_register_policy_is_frozen_separately():
    source = COORDINATOR.read_text(encoding="utf-8")
    _assert_current_profile("qwen3vl_pooler_ds1_register_policy")

    pipeline = source[
        source.index("at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward("):
    ]
    assert "retained_deepstack_count = s.retained_deepstack_count" in pipeline
    assert "retained_deepstack_count == 1" in pipeline
    assert "? qwen3vl_pooler_ds1_register_policy()" in pipeline
    assert ": qwen3vl_pooler_z1_register_policy()" in pipeline


def test_deepstack3_typed_register_policy_is_frozen_separately():
    source = COORDINATOR.read_text(encoding="utf-8")
    _assert_current_profile("qwen3vl_pooler_ds3_register_policy")

    pipeline = source[
        source.index("at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward("):
    ]
    assert "retained_deepstack_count == 3" in pipeline
    assert "qwen3vl_pooler_ds3_register_policy()" in pipeline
    assert "? qwen3vl_pooler_ds1_register_policy()" in pipeline


def test_deepstack3_partial_mrope_policy_and_keepalive_owner_are_scoped():
    source = COORDINATOR.read_text(encoding="utf-8")
    _assert_current_profile("qwen3vl_pooler_ds3_partial_mrope_register_policy")

    model = MODEL.read_text(encoding="utf-8")
    prime = _between(
        model,
        "void CausalDecoderModel::prime_qwen3vl_pooler_z1_inputs(",
        "void CausalDecoderModel::rollback_qwen3vl_pooler_z1_inputs()",
    )
    assert "rope_cos_il.has_value() == rope_sin_il.has_value()" in prime
    assert "validate_qwen3vl_pooler_z1_rope_il" in prime
    assert "cos_il_keepalive_.narrow(" in prime
    assert "sin_il_keepalive_.narrow(" in prime
    assert "ka_last_cos_src_ = cos_dst.data_ptr()" in prime
    assert "ka_last_sin_src_ = sin_dst.data_ptr()" in prime

    pipeline = source[
        source.index("at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward("):
    ]
    assert "partial_mrope = s.partial_mrope" in pipeline
    assert "qwen3vl_pooler_ds3_partial_mrope_register_policy()" in pipeline


def test_groot_16_layer_s82_ds3_profile_and_register_policy_are_exact():
    source = COORDINATOR.read_text(encoding="utf-8")
    shape = _between(
        source,
        "void check_canary_shape(",
        "void check_retained_deepstack_count(",
    )
    for token in (
        "real_len == kRealLen",
        "real_len == kGrootRealLen",
        "retained_deepstack_count == 3",
        "qwen3vl_profile || groot_profile",
        "GR00T (256,128,4,82)",
    ):
        assert token in shape

    _assert_current_profile("groot_n17_pooler_ds3_register_policy")

    pipeline = source[
        source.index("at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward("):
    ]
    assert "groot_profile = s.real_len == kGrootRealLen" in pipeline
    assert "? groot_n17_pooler_ds3_register_policy()" in pipeline
    assert "qwen3vl_pooler_ds3_register_policy()" in pipeline


def test_profile_identity_and_ingress_geometry_hash_the_live_real_length():
    source = COORDINATOR.read_text(encoding="utf-8")
    from test_fmb_standalone_layout_rebuild import _definition
    digest = _definition(source, "uint64_t physical_binding_digest(")
    assert "static_cast<uint64_t>(s.real_len)" in digest
    assert "static_cast<uint64_t>(kRealLen)" not in digest

    stage = _between(
        source,
        "void stage_non_image_rows_locked(",
        "void emit_identity_route_locked(",
    )
    for token in (
        "qwen3vl_geometry",
        "groot_geometry",
        "s.real_len == kGrootRealLen",
        "pad_rows == kExecutionLen - kGrootRealLen",
    ):
        assert token in stage


def test_outer_fast_seals_physical_components_and_one_shot_authority():
    source = COORDINATOR.read_text(encoding="utf-8")
    from test_fmb_standalone_layout_rebuild import _definition
    digest = _definition(source, "uint64_t physical_binding_digest(")
    for token in (
        "physical_arena_base()",
        "physical_arena_end()",
        "physical_top_reservation()",
        "s.retained_deepstack_count",
        "kDeepstackPorts.size()",
        "retained.resolve_physical_offset",
        "retained.resolve_physical_addr(core",
        "s.deepstack_addrs[ordinal]",
        "vision_scratch.resolve_physical_offset",
        "text_scratch.resolve_physical_offset",
        "source.resolve_physical_addr(0",
        "destination.resolve_physical_addr(0",
        "for (int core = 0; core < kNumCores; ++core)",
        "vision_scratch.resolve_physical_addr(core",
        "text_scratch.resolve_physical_addr(core",
        "route.route_hash()",
        "run.source_row",
        "run.destination_row",
        "run.row_count",
    ):
        assert token in digest

    begin = _between(
        source,
        "int64_t rpu_qwen3vl_pooler_spm_z1_begin(",
        "void rpu_qwen3vl_pooler_spm_z1_end(",
    )
    equality = begin.index(
        "*s.invocation_prepared_generation ==\n"
        "                        *s.invocation_consumed_generation"
    )
    acquire = begin.index("SpmPipelineLease::acquire_physical(")
    lease_commit = begin.index("s.lease = std::move(lease)")
    publish = begin.index(
        "*s.invocation_prepared_generation = next_invocation_generation"
    )
    assert equality < acquire < lease_commit < publish

    end = _between(
        source,
        "void rpu_qwen3vl_pooler_spm_z1_end(",
        "void rpu_qwen3vl_pooler_spm_z1_abort(",
    )
    consumed = end.index(
        "*s.invocation_consumed_generation ==\n"
        "                            s.active_invocation_generation"
    )
    release = end.index("s.lease->release()")
    assert consumed < release


def test_outer_fast_replay_exactly_rebinds_three_mutable_source_slots():
    source = COORDINATOR.read_text(encoding="utf-8")
    text_bind = _between(
        source,
        "void bind_non_image_rows_outer_fast_locked(",
        "void emit_identity_route_locked(",
    )
    assert text_bind.count("guard.bind_fast_mutable_dma(") == 2
    assert "/*byte_begin=*/0, prefix_bytes" in text_bind
    assert "suffix_begin, suffix_bytes" in text_bind
    assert "s.hidden_prefix_src_base" in text_bind
    assert "s.hidden_suffix_src_base" in text_bind
    assert "s.hidden_prefix_src_base =" not in text_bind
    assert "s.hidden_suffix_src_base =" not in text_bind
    assert "s.hidden_ref =" not in text_bind

    pipeline = source[
        source.index("at::Tensor rpu_qwen3vl_pooler_spm_z1_pipeline_forward("):
    ]
    digest = pipeline.index("stage_outer_fast_physical_digest(")
    vision_component = pipeline.index(
        "stage_vision_outer_fast_component(", digest
    )
    text_component = pipeline.index(
        "stage_text_outer_fast_component(", vision_component
    )
    authority = pipeline.index(
        "stage_outer_fast_invocation_authority(", text_component
    )
    replay = pipeline.index("RpuKernelGraph::State::REPLAYING", authority)
    vision_bind = pipeline.index("bind_vision_outer_fast_input(", replay)
    text_bind_call = pipeline.index(
        "bind_non_image_rows_outer_fast_locked(", vision_bind
    )
    output = pipeline.index("text_outer_fast_output(", text_bind_call)
    validate = pipeline.index("validate_outer_fast_replay()", output)
    postflight = pipeline.index("physical_binding_digest(s) == physical_digest", validate)
    commit = pipeline.index("commit_outer_fast_replay(", postflight)
    ordinary_vision = pipeline.index("forward_vision_z1(", commit)
    assert digest < vision_component < text_component < authority < replay
    assert replay < vision_bind < text_bind_call < output < validate
    assert validate < postflight < commit < ordinary_vision
    locked_postflight = pipeline[validate:ordinary_vision]
    assert locked_postflight.index("std::lock_guard<std::mutex> lock(s.mutex)") < (
        locked_postflight.index("physical_binding_digest(s) == physical_digest")
    )
    assert locked_postflight.index(
        "physical_binding_digest(s) == physical_digest"
    ) < locked_postflight.index("commit_outer_fast_replay(")


def test_default_text_prepare_binds_complete_descriptor_before_allocation():
    model = MODEL.read_text(encoding="utf-8")
    prepare = _between(model, "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
                       "SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_destination_spec(")
    default = prepare.index("if (prepared_descriptor.empty())")
    resolve = prepare.index("encode_causal_physical_stage_descriptor(", default)
    bind = prepare.index("layout = bind_physical_layout_context(layout, manifest)", resolve)
    allocate = prepare.index("prepare_spm_pipeline_component(layout, stage_plan)", bind)
    assert default < resolve < bind < allocate
    assert "descriptor_partial_mrope == partial_mrope" in prepare
    assert "qwen3vl_pooler_z1_selected_stage_descriptor_.assign(" in prepare
    assert "set_chunk_size_override(" not in prepare
    header = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    dispatch = _between(header, "const int64_t saved_chunk_size_override =",
                        "// R-Phase 1 (Qwen3-VL):")
    # COMPLETE Z1 already supplies C128 and preserves the caller's policy.
    # Only the existing Wall/multiview paths acquire a scoped override.
    assert "wall_z1_dispatch_ || qwen3vl_multiview_text_composite_dispatch_" in dispatch
    assert "qwen3vl_pooler_z1_dispatch_" not in dispatch
    assert "set_chunk_size_override(saved_chunk_size_override)" in dispatch


def test_leased_layout_drift_fails_before_retiring_the_owner_manifest():
    source = (ROOT / "src/core/fused_model_base.cpp").read_text()
    ensure = _between(source, "static void ensure_allocated_impl(", "struct ChunkSizeProbe")
    guard = ensure.index("s.pipeline_lease_epoch_ == 0 ||")
    assert "!params_changed && !gen_changed && !persistent_invalidated" in ensure
    assert guard < ensure.index("retire_pipeline_manifest(s)")
    assert guard < ensure.index("SPM_ALLOC.reset_temporary()")
