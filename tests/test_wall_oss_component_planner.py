from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from rpu_backend.adapters.wall_oss import runtime
from rpu_backend.api._execution import bind_execution_session
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD


ROOT = Path(__file__).resolve().parents[1]


class _GraphCache:
    def __init__(self):
        self.resets = 0

    def begin_warmup(self):
        return self

    def clear(self):
        self.resets += 1

    def cache_invariant_ok(self):
        return True


def _owner(*, fused=True):
    owner = runtime.WallOssVLA.__new__(runtime.WallOssVLA)
    owner.llm = SimpleNamespace(
        _handle=11,
        _prefill_chunk_size_cap=384,
        _graph_cache=_GraphCache(),
        residual_scale=1,
    )
    owner.vision = SimpleNamespace(_handle=12, _graph_cache=_GraphCache())
    owner.action = SimpleNamespace(
        _handle=13,
        _fused=fused,
        _graph_cache=_GraphCache(),
    )
    owner._closed = False
    owner._prepared_graph_profile = {"phase": "READY"}
    owner._execution_generation = 0
    owner._component_generations = {
        component: 0 for component in runtime.WALL_OSS_EXECUTION_COMPONENTS
    }
    owner._execution_reconfigure_journal = None
    owner._rpu_last_execution_plan = {}
    resolved = runtime.resolve_wall_oss_execution_components(
        None, entry_point="test Wall"
    )
    owner._publish_execution_views(
        resolved,
        generation=0,
        component_generations=owner._component_generations,
    )
    owner._execution_session = bind_execution_session(
        owner,
        owner._rpu_execution,
        entry_point="test Wall",
        supported_components=runtime.WALL_OSS_EXECUTION_COMPONENTS,
        validate=owner.validate_execution_reconfigure,
        apply=owner.apply_execution_reconfigure,
        rollback=owner.rollback_execution_reconfigure,
        graph_mode=GRAPH_COMPOSITE_CHILD,
    )
    return owner


def test_wall_root_resolves_independent_child_views():
    root, text, vision, action = runtime.resolve_wall_oss_execution_components(
        {
            "prefill": {"chunk_size": 64},
            "vision": {"chunk_size": 128},
            "action": {"chunk_size": 32},
            "components": {
                runtime.WALL_OSS_TEXT_COMPONENT: {
                    "prefill": {"chunk_size": 96}
                }
            },
        },
        entry_point="test Wall",
    )

    assert root["components"][runtime.WALL_OSS_TEXT_COMPONENT]["prefill"] == {
        "chunk_size": 96
    }
    assert text == {"prefill": {"chunk_size": 96}}
    assert vision == {"vision": {"chunk_size": 128}}
    assert action == {"action": {"chunk_size": 32}}


def test_wall_accumulation_resolves_component_overrides_and_owned_text():
    root, text, vision, action = runtime.resolve_wall_oss_execution_components(
        {"prefill": {"linear_acc32": True},
         "vision": {"linear_acc32": False},
         "action": {"linear_acc32": True},
         "components": {runtime.WALL_OSS_TEXT_COMPONENT: {
             "prefill": {"linear_acc32": False}}}},
        entry_point="test Wall ACC",
    )
    assert root["prefill"]["linear_acc32"] is True
    assert text["prefill"]["linear_acc32"] is False
    assert vision["vision"]["linear_acc32"] is False
    assert action["action"]["linear_acc32"] is True
    owned = runtime._action_builder_execution(text, action)
    assert owned["prefill"]["linear_acc32"] is False
    assert owned["action"]["linear_acc32"] is True


@pytest.mark.parametrize("stage", ["prefill", "vision", "action"])
@pytest.mark.parametrize("previous", [False, True])
def test_wall_accumulation_hot_change_fails_before_native_or_graph(stage, previous):
    owner = _owner()
    old = {stage: {"linear_acc32": previous}}
    owner._publish_execution_views(
        runtime.resolve_wall_oss_execution_components(old, entry_point="test"),
        generation=0,
    )
    with pytest.raises(ValueError, match=f"{stage}.linear_acc32 is cold-only"):
        owner.validate_execution_reconfigure({stage: {"linear_acc32": not previous}})
    assert owner._execution_generation == 0
    assert owner._prepared_graph_profile == {"phase": "READY"}
    assert all(child._graph_cache.resets == 0
               for child in (owner.llm, owner.vision, owner.action))
    assert owner._rpu_execution == old


@pytest.mark.parametrize("enabled", [False, True])
def test_wall_same_effective_accumulation_allows_chunk_validation(monkeypatch, enabled):
    owner = _owner()
    old = {stage: {"linear_acc32": enabled} for stage in ("prefill", "vision", "action")}
    owner._publish_execution_views(
        runtime.resolve_wall_oss_execution_components(old, entry_point="test"),
        generation=0,
    )
    for name in ("execution_reconfigure_begin", "execution_reconfigure_commit",
                 "execution_reconfigure_abort", "execution_reconfigure_abort_attempt",
                 "causal_decoder_stage_chunk_size_override",
                 "qwen25vl_vision_stage_chunk_size", "wall_oss_action_stage_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *args: None, raising=False)
    owner.validate_execution_reconfigure(old if enabled else {})
    assert owner._execution_generation == 0


def test_wall_hot_change_is_one_native_transaction_and_child_generation(
    monkeypatch,
):
    events = []
    token = 73

    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_begin",
        lambda attempt: (events.append(("begin", attempt)), token)[1],
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_commit",
        lambda value: events.append(("commit", value)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_abort",
        lambda value: events.append(("abort", value)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "execution_reconfigure_abort_attempt",
        lambda attempt: False,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "causal_decoder_stage_chunk_size_override",
        lambda handle, value, chunk: events.append(
            ("text", handle, value, chunk)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen25vl_vision_stage_chunk_size",
        lambda *args: events.append(("vision", *args)),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rpu,
        "wall_oss_action_stage_chunk_size",
        lambda handle, value, chunk: events.append(
            ("action", handle, value, chunk)
        ),
        raising=False,
    )

    owner = _owner()
    result = owner._execution_session.reconfigure({
        "components": {
            runtime.WALL_OSS_TEXT_COMPONENT: {
                "prefill": {"chunk_size": 64}
            },
            runtime.WALL_OSS_ACTION_COMPONENT: {
                "action": {"chunk_size": 32}
            },
        }
    })

    assert events[0][0] == "begin"
    assert events[1:] == [
        ("text", 11, token, 64),
        ("action", 13, token, 32),
        ("commit", token),
    ]
    assert result is owner._rpu_execution
    assert owner._execution_session.generation == 1
    assert owner._component_generations == {
        runtime.WALL_OSS_TEXT_COMPONENT: 1,
        runtime.WALL_OSS_VISION_COMPONENT: 0,
        runtime.WALL_OSS_ACTION_COMPONENT: 1,
    }
    assert owner.llm._fmb_execution_generation == 1
    assert owner.vision._fmb_execution_generation == 0
    assert owner.action._fmb_execution_generation == 1
    assert owner.llm._graph_cache.resets == 1
    assert owner.action._graph_cache.resets == 1
    assert owner.vision._graph_cache.resets == 0
    assert owner._prepared_graph_profile is None


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_wall_commit_then_python_exception_compensates_or_poison(monkeypatch, rollback_fails):
    state = {"live": 0, "staged": 0, "commits": 0}
    def stage(handle, token, chunk):
        state["staged"] = chunk
    def commit(token):
        state["commits"] += 1
        if state["commits"] == 2 and rollback_fails:
            raise RuntimeError("rollback commit failed")
        state["live"] = state["staged"]
        if state["commits"] == 1:
            raise RuntimeError("committed then Python failed")
    for name, function in {
        "execution_reconfigure_begin": lambda attempt: attempt,
        "execution_reconfigure_commit": commit,
        "execution_reconfigure_abort": lambda token: None,
        "execution_reconfigure_abort_attempt": lambda attempt: False,
        "causal_decoder_stage_chunk_size_override": stage,
        "qwen25vl_vision_stage_chunk_size": lambda *args: None,
        "wall_oss_action_stage_chunk_size": lambda *args: None,
    }.items():
        monkeypatch.setattr(torch.ops.rpu, name, function, raising=False)
    owner = _owner()
    session = owner._execution_session
    old_config = session.config
    with pytest.raises(RuntimeError, match="committed then Python failed"):
        session.reconfigure({"prefill": {"chunk_size": 64}})
    assert state["commits"] == 2
    assert session.generation == 0 and session.config is old_config
    if rollback_fails:
        assert state["live"] == 64
        assert session.stats()["state"] == "POISONED"
        with pytest.raises(RuntimeError, match="poisoned"):
            session.reconfigure({"prefill": {"chunk_size": 96}})
    else:
        assert state["live"] == 0
        assert session.stats()["state"] == "QUIESCENT"
        assert owner.llm._graph_cache.resets == 1


def test_wall_same_config_force_rebuild_retires_every_real_child(monkeypatch):
    for name in ("execution_reconfigure_commit", "execution_reconfigure_abort",
                 "execution_reconfigure_abort_attempt", "causal_decoder_stage_chunk_size_override",
                 "qwen25vl_vision_stage_chunk_size", "wall_oss_action_stage_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *args: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "execution_reconfigure_begin", lambda attempt: attempt,
                        raising=False)
    owner = _owner()
    owner._execution_session._reconfigure({}, force_rebuild=True)
    assert owner._execution_session.config == {}
    assert owner._execution_session.generation == 1
    assert set(owner._component_generations.values()) == {1}
    for child in (owner.llm, owner.vision, owner.action):
        assert child._graph_cache.resets == 1
        assert child._fmb_execution_generation == 1
    assert owner._prepared_graph_profile is None


def test_shadowed_root_default_still_rolls_back_owner_publication(monkeypatch):
    for name in ("execution_reconfigure_commit", "execution_reconfigure_abort",
                 "execution_reconfigure_abort_attempt", "causal_decoder_stage_chunk_size_override",
                 "qwen25vl_vision_stage_chunk_size", "wall_oss_action_stage_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *args: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "execution_reconfigure_begin", lambda attempt: attempt,
                        raising=False)
    owner = _owner()
    session = owner._execution_session
    session.reconfigure({"components": {
        runtime.WALL_OSS_TEXT_COMPONENT: {"prefill": {"chunk_size": 96}}
    }})
    old = session.config
    generations = dict(owner._component_generations)
    publish = owner._publish_execution_views
    def interrupted(*args, **kwargs):
        publish(*args, **kwargs)
        if kwargs["generation"] == 2:
            raise KeyboardInterrupt
    owner._publish_execution_views = interrupted
    with pytest.raises(KeyboardInterrupt):
        session.reconfigure({"prefill": {"chunk_size": 64}})
    assert session.config is old and session.generation == 1
    assert owner._execution_generation == 1
    assert owner._component_generations == generations
    assert owner._text_execution["prefill"]["chunk_size"] == 96
    assert session.stats()["state"] == "QUIESCENT"


def test_wall_composite_collects_three_generation_checked_receipts():
    owner = _owner()
    children = (
        (runtime.WALL_OSS_TEXT_COMPONENT, owner.llm),
        (runtime.WALL_OSS_VISION_COMPONENT, owner.vision),
        (runtime.WALL_OSS_ACTION_COMPONENT, owner.action),
    )
    for component, child in children:
        child._rpu_last_execution_plan = {
            "component": component,
            "generation": 0,
            "dry_forward_agreement": True,
            "physical_plan_digest": f"digest-{component}",
        }

    owner._collect_component_plan_receipts()

    assert set(owner._rpu_last_execution_plan) == {
        runtime.WALL_OSS_TEXT_COMPONENT,
        runtime.WALL_OSS_VISION_COMPONENT,
        runtime.WALL_OSS_ACTION_COMPONENT,
    }


def test_wall_native_sources_use_descriptor_routes_and_typed_kv_plan():
    root = Path(__file__).resolve().parents[1]
    action = (root / "src/fused/rpu_wall_oss_action_step_model.cpp").read_text()
    vision = (root / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text()
    decoder = (root / "src/fused/rpu_qwen3_model.h").read_text()

    assert "CausalDecoderModel::physical_manifest_for_candidate(" in action
    assert "CAUSAL_DECODER_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED" in decoder
    assert "restore_kvinsert_plan(" in decoder
    assert "site_id, manifest.graph_lifecycle," in decoder
    assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in decoder
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan" in decoder
    assert "resolve_stage_domain(" in vision
    assert "planned_stage_descriptor" in vision
    assert "image_spans.push_back({offset, per_image, image})" in vision
    assert "restore_kvinsert_plan(" in vision
    assert "QWEN25_KV_INSERT_SITE, manifest.graph_lifecycle," in vision
    assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in vision
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in vision
    assert "rpu_launch_insert_kvcache_spm_unified(" not in vision
    for relative in ("action.py", "vision.py", "llm.py"):
        adapter = (
            ROOT / "python/rpu_backend/adapters/wall_oss" / relative
        ).read_text()
        assert '"physical_descriptor": tuple(' in adapter
