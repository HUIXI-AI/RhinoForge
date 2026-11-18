from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.internvla_n1 import execution
from rpu_backend.adapters.internvla_n1.nextdit import NextDiTRPURuntime


ROOT = Path(__file__).resolve().parents[1]


class _GraphCache:
    def __init__(self):
        self.clears = 0
        self.warmups = 0

    def begin_warmup(self):
        self.warmups += 1

    def clear(self):
        self.clears += 1

    def cache_invariant_ok(self):
        return True


def _child(cache_name):
    child = SimpleNamespace()
    setattr(child, cache_name, _GraphCache())
    return child


def _native_domain(rows):
    schedule = []
    for _stage in range(3):
        schedule.extend((0, 0, rows, rows))
    descriptor = [
        1, 0, rows, rows, rows,
        0, 1, 0, 0, 0, 0, 1, 1, 1, 1,
        *schedule, 0, rows, -1,
    ]
    descriptor[1] = len(descriptor)
    return [1, 1, len(descriptor), *descriptor]


def test_internvla_has_stable_independent_component_capabilities():
    assert set(execution.INTERNVLA_EXECUTION_COMPONENTS) == {
        "system2_backbone",
        "system2_vision",
        "rgb_tower",
        "depth_tower",
        "rgbd_former",
        "navdp_action",
        "nextdit_action",
    }
    _root, children = execution.resolve_internvla_execution(
        {
            "components": {
                "system2_backbone": {
                    "prefill": {"chunk_size": 176, "padding_rows": 32}
                },
                "navdp_action": {"action": {"chunk_size": 32}},
            }
        },
        entry_point="test",
    )
    assert children["system2_backbone"]["prefill"] == {
        "chunk_size": 176,
        "padding_rows": 32,
    }
    assert children["navdp_action"]["action"]["chunk_size"] == 32
    assert children["nextdit_action"]["action"]["chunk_size"] == "auto"
    assert children["rgb_tower"] == {}
    assert children["depth_tower"] == {}

    with pytest.raises(ValueError, match="not supported"):
        execution.component_execution(
            {"vision": {"chunk_size": 16}},
            execution.RGB_COMPONENT,
            entry_point="test",
        )




@pytest.mark.parametrize("force_rebuild", [False, True])
def test_controller_uses_one_session_and_selective_child_invalidation(
    monkeypatch, force_rebuild,
):
    owner = SimpleNamespace()
    cache_names = {
        execution.BACKBONE_COMPONENT: "_graph_cache",
        execution.VISION_COMPONENT: "_graph_cache",
        execution.RGB_COMPONENT: "_gc",
        execution.DEPTH_COMPONENT: "_gc",
        execution.FORMER_COMPONENT: "gc",
        execution.NAVDP_COMPONENT: "_gc",
        execution.NEXTDIT_COMPONENT: "_gc",
    }
    children = {component: _child(name) for component, name in cache_names.items()}
    controller = execution.InternVLAExecutionController(owner, children)
    assert all(child._execution_session is controller.session for child in children.values())

    patch = {
        "components": {
            execution.NAVDP_COMPONENT: {
                "action": {"chunk_size": 32}
            }
        }
    }
    if force_rebuild:
        controller.session._reconfigure({}, force_rebuild=True)
    else:
        controller.reconfigure_rpu_execution(patch)
    for component, child in children.items():
        changed = int(force_rebuild or component == execution.NAVDP_COMPONENT)
        assert getattr(child, cache_names[component]).clears == changed
        assert getattr(child, cache_names[component]).warmups == changed
        assert child._fmb_execution_generation == changed
    assert controller.stats()["generation"] == 1

    with controller.execute():
        with pytest.raises(RuntimeError, match="inside a forward"):
            controller.reconfigure_rpu_execution({
                "components": {
                    execution.NAVDP_COMPONENT: {
                        "action": {"chunk_size": "auto"}
                    }
                }
            })


def test_controller_rejects_config_for_absent_children():
    with pytest.raises(ValueError, match="absent component"):
        execution.InternVLAExecutionController(
            SimpleNamespace(),
            {execution.BACKBONE_COMPONENT: _child("_graph_cache")},
            {
                "components": {
                    execution.NAVDP_COMPONENT: {
                        "action": {"chunk_size": 32}
                    }
                }
            },
        )
    with pytest.raises(ValueError, match="no present child"):
        execution.InternVLAExecutionController(
            SimpleNamespace(),
            {execution.BACKBONE_COMPONENT: _child("_graph_cache")},
            {"action": {"chunk_size": 32}},
        )


@pytest.mark.parametrize("fault", ["partial_clear", "publish", "rollback_clear"])
def test_shared_reconfigure_compensates_graph_and_publication_failures(monkeypatch, fault):
    children = {
        execution.NAVDP_COMPONENT: _child("_gc"),
        execution.FORMER_COMPONENT: _child("_gc"),
        execution.BACKBONE_COMPONENT: _child("_gc"),
    }
    controller = execution.InternVLAExecutionController(SimpleNamespace(), children)
    old_config = controller.session.config
    changed = {execution.NAVDP_COMPONENT, execution.FORMER_COMPONENT}
    for component, child in children.items():
        child._rpu_last_execution_plan = {"old": component}
        child._rpu_last_execution_plans = {component: "old", "foreign": "keep"}
        child._rpu_last_execution_receipts = {component: "old", "foreign": "keep"}
    count, enabled = [0], [True]
    for component in changed:
        cache = children[component]._gc
        original = cache.clear
        def clear(original=original):
            assert controller.session._lock._is_owned()
            original()
            count[0] += 1
            if enabled[0] and ((fault == "partial_clear" and count[0] == 2)
                               or fault == "rollback_clear"):
                raise RuntimeError("graph cleanup interrupted")
        monkeypatch.setattr(cache, "clear", clear)
    publish = controller._publish
    def interrupted_publish(*args, **kwargs):
        assert controller.session._lock._is_owned()
        publish(*args, **kwargs)
        if enabled[0] and fault == "publish" and kwargs["generation"] == 1:
            raise RuntimeError("configuration publication interrupted")
    monkeypatch.setattr(controller, "_publish", interrupted_publish)
    patch = {"components": {
        execution.NAVDP_COMPONENT: {"action": {"chunk_size": 32}},
        execution.FORMER_COMPONENT: {"vision": {"chunk_size": 32}},
    }}
    with pytest.raises(RuntimeError, match="interrupted") as caught:
        controller.reconfigure_rpu_execution(patch)
    assert controller.session.config == old_config
    assert controller.session.generation == 0
    assert all(child._fmb_execution_generation == 0 for child in children.values())
    untouched = children[execution.BACKBONE_COMPONENT]
    assert untouched._gc.clears == 0
    assert untouched._rpu_last_execution_plan == {"old": execution.BACKBONE_COMPONENT}
    if fault == "rollback_clear":
        assert controller.session.stats()["state"] == "POISONED"
        assert any("rollback failed" in note for note in caught.value.__notes__)
        with pytest.raises(RuntimeError, match="poisoned"):
            controller.reconfigure_rpu_execution(patch)
    else:
        assert controller.session.stats()["state"] == "QUIESCENT"
        assert controller._component_generations == dict.fromkeys(children, 0)
        for component in changed:
            child = children[component]
            assert child._gc.clears == 2
            assert not hasattr(child, "_rpu_last_execution_plan")
            assert child._rpu_last_execution_plans == {"foreign": "keep"}
            assert child._rpu_last_execution_receipts == {"foreign": "keep"}
        enabled[0] = False
        controller.reconfigure_rpu_execution(patch)
        assert controller.session.generation == 1
        assert all(children[component]._fmb_execution_generation == 1 for component in changed)
    enabled[0] = False


@pytest.mark.parametrize("legacy", ("auto", 16))
def test_shared_vision_reads_parent_canonical_chunk_cold_hot_and_restored(monkeypatch, legacy):
    from rpu_backend.adapters.wall_oss import vision

    child = vision.WallOssVision.__new__(vision.WallOssVision)
    child._handle, child._graph_cache = 17, _GraphCache()
    child._execution_chunk_size = legacy
    captured = []
    def plan(*args, **kwargs):
        captured.append(kwargs["exact_chunk_size"])
        kwargs["plan_result_sink"](SimpleNamespace(selected=SimpleNamespace(
            stage_tuple=SimpleNamespace(physical_descriptor=(1,)))))
        return 784, 784
    monkeypatch.setattr(vision, "plan_bounded_prefill_execution", plan)
    monkeypatch.setattr(vision, "_destroy_qwen25vl_vision_handle", lambda handle: None)
    parent = SimpleNamespace()
    controller = execution.InternVLAExecutionController(parent, {execution.VISION_COMPONENT: child},
        {"components": {execution.VISION_COMPONENT: {"vision": {"chunk_size": 784}}}})
    try:
        for chunk in (784, "auto", 784):
            controller.reconfigure_rpu_execution({"components": {execution.VISION_COMPONENT: {"vision": {"chunk_size": chunk}}}})
            child._validate_execution_chunks((784,))
            child._plan_native_execution(784, 1, None, None)
            assert captured[-1] == (None if chunk == "auto" else chunk)
        controller.reconfigure_rpu_execution({"components": {execution.VISION_COMPONENT: {"vision": {"chunk_size": 16}}}})
        with pytest.raises(ValueError, match="native Vision"):
            child._validate_execution_chunks((784,))
        assert captured == [784, None, 784]
        assert child._execution_chunk_size == legacy  # No parent-specific mirror was added.
    finally:
        controller.close()


@pytest.mark.parametrize("legacy", ("auto", 784))
def test_shared_vision_without_parent_preserves_legacy_cold_default(monkeypatch, legacy):
    from rpu_backend.adapters.wall_oss import vision

    child = vision.WallOssVision.__new__(vision.WallOssVision)
    child._execution_chunk_size, child._handle = legacy, 17
    child._graph_cache = _GraphCache()
    def plan(*args, **kwargs):
        assert kwargs["exact_chunk_size"] == (None if legacy == "auto" else legacy)
        kwargs["plan_result_sink"](SimpleNamespace(selected=SimpleNamespace(
            stage_tuple=SimpleNamespace(physical_descriptor=(1,)))))
        return 784, 784
    monkeypatch.setattr(vision, "plan_bounded_prefill_execution", plan)
    child._validate_execution_chunks((784,))
    child._plan_native_execution(784, 1, None, None)


def test_navdp_and_nextdit_search_the_native_domain(monkeypatch):
    from test_graph_cache_freeze import _cache

    calls = []
    for prefix in ("navdp", "internvla_nextdit"):
        monkeypatch.setattr(torch.ops.rpu, prefix + "_planner_cache_identity",
                            lambda handle: (int(handle), 0), raising=False)

    def navdp_domain(handle, rows, denoise_steps=0):
        calls.append(("navdp", handle, rows, denoise_steps))
        return _native_domain(rows)

    def nextdit_domain(
        handle, rows, cross_seq, use_action, populate_cross,
        action_batch, denoise_steps, prepare_modulation,
    ):
        calls.append((
            "nextdit", handle, rows, cross_seq, use_action,
            populate_cross, action_batch, denoise_steps,
            prepare_modulation,
        ))
        return _native_domain(rows)

    monkeypatch.setattr(
        torch.ops.rpu, "navdp_resolve_stage_domain", navdp_domain,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu, "internvla_nextdit_resolve_stage_domain",
        nextdit_domain, raising=False,
    )
    chunk, navdp_plan = execution.plan_navdp_component(
        7,
        128,
        component=execution.NAVDP_COMPONENT,
        generation=2,
        execution_owner=SimpleNamespace(),
        graph_cache=_cache(),
        execution={"action": {"chunk_size": "auto"}},
        denoise_steps=20,
    )
    assert chunk == 128
    assert navdp_plan.selected.stage_tuple.physical_descriptor
    assert dict(navdp_plan.selected.stage_tuple.physical_metadata)[
        "route_reason:MEMORY_CROSS_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI"
    ] == 1

    runtime = NextDiTRPURuntime.__new__(NextDiTRPURuntime)
    runtime._handle = 9
    runtime._graph_cache = _cache()
    runtime._rpu_execution = {"action": {"chunk_size": "auto"}}
    runtime._fmb_execution_generation = 3
    chunk, nextdit_plan = runtime._execution_plan(
        action_rows=64,
        cross_seq=34,
        use_action_encoder=True,
        populate_cross_cache=False,
        action_batch=2,
    )
    assert chunk == 64
    assert nextdit_plan.selected.stage_tuple.physical_descriptor
    repeated_chunk, repeated_plan = runtime._execution_plan(
        action_rows=64, cross_seq=34, use_action_encoder=True,
        populate_cross_cache=False, action_batch=2,
    )
    assert repeated_chunk == chunk and repeated_plan is nextdit_plan
    runtime._execution_plan(
        action_rows=64, cross_seq=35, use_action_encoder=True,
        populate_cross_cache=False, action_batch=2,
    )
    nextdit_metadata = dict(
        nextdit_plan.selected.stage_tuple.physical_metadata
    )
    assert nextdit_metadata["self_attention_joint_domain"] == 1
    assert nextdit_metadata[
        "route_reason:PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI"
    ] == 1
    owner = SimpleNamespace()
    receipt = execution.publish_execution_receipt(
        owner,
        nextdit_plan,
        component=execution.NEXTDIT_COMPONENT,
        stage="action",
        generation=3,
        logical_len=64,
        resolved_chunk_size=64,
    )
    assert receipt["self_attention_policy"] == "DDR_REQUIRED"
    assert receipt["cross_attention_policy"] == "DDR_REQUIRED"
    assert "self_attention" not in receipt["route_reasons"]
    assert calls == [
        ("navdp", 7, 128, 20),
        ("nextdit", 9, 64, 34, True, False, 2, 0, False),
        ("nextdit", 9, 64, 35, True, False, 2, 0, False),
    ]


def test_internvla_navdp_and_nextdit_native_descriptors_are_authoritative():
    navdp = (ROOT / "src/fused/rpu_navdp_model.cpp").read_text()
    nextdit = (
        ROOT / "src/fused/rpu_internvla_nextdit_model.cpp"
    ).read_text()
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    registrations = (
        ROOT / "src/core/rpu_dispatch_registrations.inc"
    ).read_text()

    for source in (navdp, nextdit):
        assert "physical_manifest_for_candidate(" in source
        assert "FmbPhysicalManifestState::COMPLETE" in source
        assert "FmbGraphLifecycle::COMPOSITE_CHILD" in source
        assert "physical_manifest_forward_capability(" in source
        assert "encode_fmb_prefill_stage_domain(" in source
        assert "planned_stage_descriptor" in source
        assert "FmbRouteFamily::ATTENTION" in source
        assert "consume_physical_route(" in source

    assert "AttentionExecutionPolicy::SPM_KV_BY_MHA" in navdp
    assert "Site 2 is prefix/memory cross attention and is DDR_REQUIRED" in navdp
    assert "physical_manifest_domain_for_candidate(" in nextdit
    assert "subclass_spm_kv_by_mha_eligible(" in nextdit
    assert "rpu_launch_sdpa_by_mha_spm(" in nextdit
    assert "retained cross-prefix K/V" in nextdit
    for source, site_ids in (
        (nextdit, (
            3168376967729637373,
            4135248324695006730,
            3580665462795039639,
            1322531321321232750,
            931457206627820470,
            7703637926943336897,
            3529610265379555678,
            2715595076421646185,
            5939385796971399306,
            3509028667928103821,
            7763914209787107799,
        )),
        (navdp, (
            5578031918666539451,
            5028686091549759246,
            5266334158747974487,
            7913923308750549684,
            4550112057379938325,
        )),
    ):
        for site_id in site_ids:
            assert str(site_id) in source
        assert "resolve_kvinsert_plan_auto(" in source
        assert "manifest.graph_lifecycle," in source
        assert "rpu_kvinsert_route_arguments(" in source
        assert "restore_kvinsert_plan(" in source
        assert "kKvInsertRouteTopologyWords" in source
        assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in source
        assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in source
        assert "rpu_launch_insert_kvcache_spm_unified(" not in source
        assert "static_cast<int64_t>(" in source
        assert "route.arguments" in source
    for op in (
        "navdp_resolve_stage_domain",
        "navdp_get_resolved_chunk_size",
        "internvla_nextdit_resolve_stage_domain",
        "internvla_nextdit_get_resolved_chunk_size",
    ):
        assert op in declarations
        assert op in registrations
    assert registrations.count("int[] planned_stage_descriptor=[]") >= 6


def test_internvla_ddr_fallbacks_publish_stable_reasons():
    shared = (
        ROOT / "python/rpu_backend/adapters/internvla_n1/execution.py"
    ).read_text()
    nextdit = (
        ROOT / "python/rpu_backend/adapters/internvla_n1/nextdit.py"
    ).read_text()
    for reason in (
        "PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI",
        "MEMORY_CROSS_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI",
    ):
        assert reason in shared or reason in nextdit
    assert "publish_component_execution_receipt(" in shared


def test_python_paths_key_graphs_and_publish_planner_receipts():
    files = {
        name: (ROOT / path).read_text()
        for name, path in {
            "backbone": "python/rpu_backend/adapters/internvla_n1/backbone.py",
            "nextdit": "python/rpu_backend/adapters/internvla_n1/nextdit.py",
            "rgbd": "python/rpu_backend/adapters/internvla_n1/rgbd_rpu.py",
            "navdp": "python/rpu_backend/adapters/navdp/runtime.py",
        }.items()
    }
    assert "*a6_plan.graph_key_words()" in files["backbone"]
    assert "tuple(execution_plan.graph_key_words())" in files["nextdit"]
    assert files["nextdit"].count("physical_descriptor") >= 4
    assert files["navdp"].count("physical_descriptor") >= 3
    assert "rpu_backend.adapters.internvla_n1.execution" not in files["navdp"]
    assert "NATIVE_FMB_DESCRIPTOR" in files["rgbd"]
    assert "dinov3_vision_resolve_stage_domain" in files["rgbd"]
    assert "planned_stage_descriptor" in files["rgbd"]
    assert "FIXED_NATIVE_SCALAR" not in files["rgbd"]
    assert files["rgbd"].count("publish_execution_receipt(") >= 2
    for source in files.values():
        assert "_fmb_execution_generation" in source
