"""Reusable construction helpers for the RhinoVLA RPU action expert."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from rpu_backend.api._execution import (
    _freeze_rpu_execution,
    bind_execution_session,
    bind_rpu_execution,
    native_execution_reconfigure,
    resolve_component_rpu_execution,
)
from rpu_backend.runtime.execution_planner import GRAPH_COMPOSITE_CHILD


RHINOVLA_TEXT_COMPONENT = "language_model"
RHINOVLA_VISION_COMPONENT = "vision_encoder"
RHINOVLA_ACTION_COMPONENT = "action_expert"
_PIPELINE_COLD_ENV = {
    "prefill": {
        "prefix_prep_cache": "RPU_RHINOVLA_PREFIX_PREP_CACHE",
        "prefix_prep_rpu": "RPU_RHINOVLA_PREFIX_PREP_RPU",
        "prefix_no_clone": "RPU_RHINOVLA_PREFIX_NO_CLONE",
        "partial_mrope": "RPU_RHINOVLA_PARTIAL_MROPE",
        "merger_scatter": "RPU_RHINOVLA_MERGER_SCATTER",
        "fast_replay": "RPU_RHINOVLA_PREFILL_FAST_REPLAY",
        "preload_replay_skip": "RPU_RHINOVLA_PREFILL_PRELOAD_REPLAY_SKIP",
    },
    "vision": {
        "rpu_patch_embed": "RPU_RHINOVLA_VISION_RPU_PATCH_EMBED",
        "rpu_mergers": "RPU_RHINOVLA_VISION_RPU_MERGERS",
        "rpu_mergers_streaming": "RPU_RHINOVLA_VISION_RPU_MERGERS_STREAMING",
        "fast_replay": "RPU_RHINOVLA_VISION_FAST_REPLAY",
        "preload_replay_skip": "RPU_RHINOVLA_VISION_PRELOAD_REPLAY_SKIP",
        "bake_merger": "RPU_RHINOVLA_VISION_BAKE_MERGER",
        "fused_merger": "RPU_QWEN3VL_VISION_FUSED_MERGER",
    },
    "action": {
        "shared_cache": "RPU_RHINOVLA_SHARED_CACHE",
        "prefix_alias": "RPU_RHINOVLA_PREFIX_ALIAS",
        "denoise_unroll": "RPU_RHINOVLA_DENOISE_UNROLL",
        "fast_replay": "RPU_RHINOVLA_DENOISE_FAST_REPLAY",
    },
}
RHINOVLA_EXECUTION_COMPONENTS = {
    RHINOVLA_TEXT_COMPONENT: {
        "prefill": ("chunk_size", "padding_rows", "padding_budget", *_PIPELINE_COLD_ENV["prefill"]),
    },
    RHINOVLA_VISION_COMPONENT: {"vision": ("chunk_size", *_PIPELINE_COLD_ENV["vision"])},
    RHINOVLA_ACTION_COMPONENT: {"action": ("chunk_size", *_PIPELINE_COLD_ENV["action"])},
}


def resolve_rhinovla_execution(value, *, entry_point):
    from rpu_backend.api._execution import normalize_rpu_execution

    root = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=RHINOVLA_EXECUTION_COMPONENTS,
    )
    text = resolve_component_rpu_execution(
        root,
        RHINOVLA_TEXT_COMPONENT,
        entry_point=entry_point,
        supported_components=RHINOVLA_EXECUTION_COMPONENTS,
        profile_auto={"prefill": {"chunk_size": "auto"}},
    )
    vision = resolve_component_rpu_execution(
        root,
        RHINOVLA_VISION_COMPONENT,
        entry_point=entry_point,
        supported_components=RHINOVLA_EXECUTION_COMPONENTS,
        profile_auto={"vision": {"chunk_size": "auto"}},
    )
    action = resolve_component_rpu_execution(
        root,
        RHINOVLA_ACTION_COMPONENT,
        entry_point=entry_point,
        supported_components=RHINOVLA_EXECUTION_COMPONENTS,
        profile_auto={"action": {"chunk_size": "auto"}},
    )
    return root, text, vision, action


@dataclass(frozen=True)
class RhinoVLAPipelineColdConfig:
    """One immutable constructor snapshot; explicit config wins over legacy env."""

    stages: Mapping

    @classmethod
    def resolve(cls, value):
        from rpu_backend.runtime import rpu_env_bool

        root, *children = resolve_rhinovla_execution(value, entry_point="RhinoVLAOnRPU")
        stages = {}
        for (stage, fields), child in zip(_PIPELINE_COLD_ENV.items(), children):
            explicit = child.get(stage, {})
            stages[stage] = {
                field: explicit[field] if field in explicit else rpu_env_bool(env)
                for field, env in fields.items()
            }
        snapshot = cls(_freeze_rpu_execution(stages))
        snapshot._validate_dependencies()
        return snapshot, snapshot.bind_defaults(root)

    def bind_defaults(self, value):
        # Each Rhino stage has one physical owner. Publish defaults at the stage
        # level, retaining explicit child overrides and never injecting AUTO
        # chunks that would shadow later top-level hot requests.
        root = resolve_rhinovla_execution(value, entry_point="RhinoVLA cold configuration")[0]
        effective = dict(root)
        for stage, fields in self.stages.items():
            effective[stage] = {**fields, **root.get(stage, {})}
        return resolve_rhinovla_execution(effective, entry_point="RhinoVLA cold configuration")[0]

    def _validate_dependencies(self):
        p, v, a = (self.stages[stage] for stage in _PIPELINE_COLD_ENV)
        requirements = (
            (v["rpu_mergers_streaming"], v["rpu_mergers"], "vision.rpu_mergers_streaming requires rpu_mergers"),
            (p["prefix_prep_rpu"], p["prefix_prep_cache"], "prefill.prefix_prep_rpu requires prefix_prep_cache"),
            (p["partial_mrope"], p["prefix_prep_cache"], "prefill.partial_mrope requires prefix_prep_cache"),
            (p["merger_scatter"], p["prefix_prep_rpu"] and v["fused_merger"], "prefill.merger_scatter requires prefix_prep_rpu and vision.fused_merger"),
            (v["bake_merger"], v["fast_replay"] and v["fused_merger"], "vision.bake_merger requires fast_replay and fused_merger"),
            (a["prefix_alias"], a["shared_cache"], "action.prefix_alias requires shared_cache"),
        )
        for enabled, required, message in requirements:
            if enabled and not required:
                raise ValueError(f"RhinoVLA cold configuration: {message}")

    def validate_bound(self, value):
        _, *children = resolve_rhinovla_execution(value, entry_point="RhinoVLAPolicy.reconfigure")
        for (stage, fields), child in zip(self.stages.items(), children):
            for field, bound in fields.items():
                if child.get(stage, {}).get(field, bound) != bound:
                    raise ValueError(
                        f"RhinoVLA {stage}.{field} is cold-bound; rebuild the runtime to change it"
                    )


def _validate_pipeline_cold(runtime, value):
    snapshot = getattr(runtime, "_pipeline_cold_config", None)
    if snapshot is not None:
        snapshot.validate_bound(value)
        return
    _, *children = resolve_rhinovla_execution(value, entry_point="RhinoVLA runtime")
    if any(set(child.get(stage, {})).intersection(fields)
           for (stage, fields), child in zip(_PIPELINE_COLD_ENV.items(), children)):
        raise ValueError("RhinoVLA runtime must bind a pipeline cold snapshot before using physical bool controls")


def _native_chunk(config, stage):
    value = config.get(stage, {}).get("chunk_size", "auto")
    return 0 if value == "auto" else int(value)


def _reset_graph_cache(cache, *, label):
    if cache is None:
        return
    cache.begin_warmup()
    cache.clear()
    if not cache.cache_invariant_ok():
        raise RuntimeError(
            f"RhinoVLA {label} GraphCache invariant failed during reconfigure"
        )


_RETIREMENT_CHILDREN = {
    RHINOVLA_TEXT_COMPONENT: ("text_model", "_rpu_decoder_retirement_state",
                             ("_rpu_text_graph_cache", "_rpu_decoder_graph_cache")),
    RHINOVLA_VISION_COMPONENT: ("vision_model", "_rpu_vision_retirement_state",
                               ("_rpu_vision_graph_cache",)),
    RHINOVLA_ACTION_COMPONENT: ("action_expert", "_rpu_retirement_state", ("_rpu_graph_cache",)),
}
_FAILED_RETIREMENTS = []


def _retirement_session(owner):
    from rpu_backend.api._execution import ExecutionSession

    controller = vars(owner).get("_rhinovla_execution_controller")
    if isinstance(controller, _RhinoVLAExecutionController):
        session = vars(controller).get("session")
        if isinstance(session, ExecutionSession) and session._owner is owner:
            return session
    session = vars(owner).get("_execution_session")
    return session if isinstance(session, ExecutionSession) and session._owner is owner else None


def _retirement_failure(owner, error):
    from rpu_backend.api._execution import _mark_execution_process_unsafe

    if not any(value is owner for value in _FAILED_RETIREMENTS):
        _FAILED_RETIREMENTS.append(owner)
    vars(owner).setdefault("_rhinovla_retirement_failed", error)
    _mark_execution_process_unsafe("RhinoVLA retirement failed")
    session = _retirement_session(owner)
    if session is not None:
        session.poison()


def own_rhinovla_child(owner, child, component):
    """Adopt each installed child immediately, including before parent binding."""
    from threading import RLock
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    state = vars(owner)
    if any(state.get(name) for name in (
            "_rhinovla_closed", "_rhinovla_retiring", "_rhinovla_retirement_failed")):
        raise RuntimeError("RhinoVLA retirement owner is closed or failed")
    children = state.setdefault("_rhinovla_retirement_children", {})
    if component in children and children[component] is not child:
        raise RuntimeError("RhinoVLA retirement child identity changed")
    _name, attribute, _graphs = _RETIREMENT_CHILDREN[component]
    resource = vars(child).get(attribute)
    if (not isinstance(resource, _InstalledNativeResource) or resource.owner() is not child
            or resource.handle is None or resource.failed is not None):
        raise RuntimeError("RhinoVLA child lacks its actual installed retirement resource")
    parent = resource.parent() if resource.parent is not None else None
    if parent is not None and parent is not owner:
        raise RuntimeError("RhinoVLA child has a different retirement owner")
    if resource.parent is None and not resource.finalizer.alive:
        error = RuntimeError("RhinoVLA child lost its retirement finalizer before adoption")
        resource.retain_failure(error, owner)
        _retirement_failure(owner, error)
        raise error
    if (type(vars(child).get(resource.handle_name)) is not int
            or vars(child).get(resource.handle_name) != resource.handle
            or vars(child).get(resource.handle_name + "_finalizer") is not resource.finalizer):
        raise RuntimeError("RhinoVLA child retirement handle/finalizer identity changed")
    child_session = vars(child).get("_execution_session")
    if child_session is not None and child_session is not _retirement_session(owner):
        raise RuntimeError("RhinoVLA child has a different execution Session")
    state.setdefault("_rhinovla_retirement_lock", RLock())
    # Publish the journal before detaching the leaf callback. An interrupted
    # adoption must still leave the actual native resource owned by this bundle.
    children[component] = child
    resource.take_ownership(owner)
    return child


def close_rhinovla_runtime(owner, *, _gc=False, _release_claim=True):
    """One parent shutdown: all Graphs before any native handle, no retry."""
    from contextlib import nullcontext
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource

    state = vars(owner)
    if state.get("_rhinovla_retirement_failed") is not None:
        raise RuntimeError("RhinoVLA retirement already failed; restart the process")
    if state.get("_rhinovla_closed"):
        return
    session = _retirement_session(owner)
    lock = session._lock if session is not None else state.get("_rhinovla_retirement_lock")
    with lock if lock is not None else nullcontext():
        if state.get("_rhinovla_retiring") or state.get("_rhinovla_closed"):
            return
        if session is not None and session._active:
            raise RuntimeError("cannot close RhinoVLA during a forward")
        state["_rhinovla_retiring"] = True
        try:
            controller = state.get("_rhinovla_execution_controller")
            if (state.get("_execution_session") is not session
                    or (controller is not None and (
                        not isinstance(controller, _RhinoVLAExecutionController)
                        or getattr(controller, "runtime", None) is not owner
                        or getattr(controller, "session", None) is not session))
                    or (session is not None and session._owner is not owner)):
                raise RuntimeError("RhinoVLA retirement Session identity changed")
            children = state.get("_rhinovla_retirement_children", {})
            resources, graphs = [], []
            for component, child in children.items():
                name, attribute, graph_names = _RETIREMENT_CHILDREN[component]
                resource = vars(child).get(attribute)
                current = tuple(vars(child).get(key) for key in graph_names)
                if (controller is not None and getattr(controller, name) is not child
                        or vars(child).get("_execution_session") is not session
                        or not isinstance(resource, _InstalledNativeResource)
                        or (resource.owner() is not child and not (_gc and resource.owner() is None))
                        or resource.parent is None
                        or (resource.parent() is not owner and not (_gc and resource.parent() is None))
                        or resource.failed is not None or resource.raw_graphs
                        or resource.finalizer.alive
                        or resource.handle is None
                        or type(vars(child).get(resource.handle_name)) is not int
                        or vars(child).get(resource.handle_name) != resource.handle
                        or vars(child).get(resource.handle_name + "_finalizer") is not resource.finalizer
                        or any(graph is None for graph in current)
                        or {id(graph) for graph in current} != {id(graph) for graph in resource.graphs}):
                    raise RuntimeError("RhinoVLA retirement native/Graph/child identity changed")
                resources.append((child, resource))
                graphs.extend(current)
            if controller is not None and set(children) != set(_RETIREMENT_CHILDREN):
                raise RuntimeError("RhinoVLA retirement requires all three physical children")

            def retire():
                _execution._require_execution_process_safe()
                seen = set()
                for graph in graphs:
                    if id(graph) in seen:
                        continue
                    seen.add(id(graph))
                    graph.clear()
                    if not graph.cache_invariant_ok():
                        raise RuntimeError("RhinoVLA GraphCache invariant failed during close")
                for child, resource in resources:
                    handle = resource.handle
                    try:
                        resource.destroy(handle)
                    except BaseException as error:
                        resource.retain_failure(error, owner)
                        raise
                    resource._native_destroyed(handle)
                    vars(child)[resource.handle_name] = None  # Cyclic GC cleared weakrefs.
                state["_rhinovla_closed"] = True
                state.pop("_rhinovla_retirement_children", None)
                if _release_claim:
                    causal_lm._release_live_instance(owner)

            with causal_lm._LIVE_LOCK:
                live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                if live is not None and live is not owner:
                    raise RuntimeError("RhinoVLA retirement after live-owner handoff")
                if session is None:
                    retire()
                else:
                    session.shutdown(retire)
                if not state.get("_rhinovla_closed"):
                    raise RuntimeError("RhinoVLA closed Session still owns native resources")
        except BaseException as error:
            _retirement_failure(owner, error)
            raise
        finally:
            state.pop("_rhinovla_retiring", None)


def gc_close_rhinovla_runtime(owner):
    """Retire through the parent while cyclic GC still holds all children."""
    if (not vars(owner).get("_rhinovla_retirement_children")
            or vars(owner).get("_rhinovla_closed")
            or vars(owner).get("_rhinovla_retirement_failed") is not None):
        return
    try:
        close_rhinovla_runtime(owner, _gc=True)
    except BaseException as error:
        _retirement_failure(owner, error)


def _preflight_rhinovla_execution_session(runtime, root):
    """Reject an incompatible existing Session before adopting any child."""
    from rpu_backend.api._execution import (
        ExecutionSession,
        _component_capabilities,
    )

    session = vars(runtime).get("_execution_session")
    if session is None:
        return None
    if not isinstance(session, ExecutionSession):
        raise TypeError(
            "bind_rhinovla_execution_runtime: _execution_session has an invalid type"
        )
    with session._lock:
        expected_components, _ = _component_capabilities(
            RHINOVLA_EXECUTION_COMPONENTS,
            entry_point="bind_rhinovla_execution_runtime",
        )
        if session._owner is not runtime:
            raise ValueError("RhinoVLA execution Session owns a different runtime")
        if session._supported_components != expected_components:
            raise ValueError("RhinoVLA execution Session components changed")
        if session.config != root:
            raise ValueError("RhinoVLA execution Session configuration changed")
        if session.graph_mode != GRAPH_COMPOSITE_CHILD:
            raise ValueError("RhinoVLA execution Session graph mode changed")
        session.require_cold()
        occupied = [
            name
            for name in ("validate", "apply", "rollback")
            if getattr(session, f"_{name}") is not None
        ]
        if occupied:
            raise RuntimeError(
                "RhinoVLA execution Session callback(s) already bound: "
                f"{occupied}"
            )
    return session


class _RhinoVLAExecutionController:
    def __init__(self, runtime, text_model, vision_model, action_expert, value):
        import torch
        from rpu_backend.api._execution import _require_execution_process_safe

        _require_execution_process_safe()

        self.runtime = runtime
        self.text_model = text_model
        self.vision_model = vision_model
        self.action_expert = action_expert
        self._journal = None
        self._generation = 0
        self._component_generations = {
            component: 0 for component in RHINOVLA_EXECUTION_COMPONENTS
        }
        _validate_pipeline_cold(runtime, value)
        root = bind_rpu_execution(
            runtime,
            value,
            entry_point="bind_rhinovla_execution_runtime",
            supported_components=RHINOVLA_EXECUTION_COMPONENTS,
        )
        existing_session = _preflight_rhinovla_execution_session(runtime, root)
        resolved = resolve_rhinovla_execution(
            root, entry_point="bind_rhinovla_execution_runtime"
        )
        for component, child in ((RHINOVLA_TEXT_COMPONENT, text_model),
                                 (RHINOVLA_VISION_COMPONENT, vision_model),
                                 (RHINOVLA_ACTION_COMPONENT, action_expert)):
            own_rhinovla_child(runtime, child, component)
            if existing_session is not None:
                vars(child)["_execution_session"] = existing_session
        required = (
            "causal_decoder_set_chunk_size_override",
            "causal_decoder_get_resolved_chunk_size",
            "causal_decoder_enable_execution_reconfigure",
            "qwen3vl_vision_set_chunk_size",
            "qwen3vl_vision_get_resolved_chunk_size",
            "qwen3vl_vision_enable_execution_reconfigure",
            "rhino_vla_set_chunk_size",
            "rhino_vla_get_resolved_chunk_size",
            "rhino_vla_enable_execution_reconfigure",
        )
        missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
        if missing:
            raise RuntimeError(
                "RhinoVLA binary lacks execution-control op(s): "
                + ", ".join(missing)
            )
        dispatched = {
            RHINOVLA_TEXT_COMPONENT:
                torch.ops.rpu.causal_decoder_get_resolved_chunk_size(
                    text_model._rpu_decoder_handle
                ),
            RHINOVLA_VISION_COMPONENT:
                torch.ops.rpu.qwen3vl_vision_get_resolved_chunk_size(
                    vision_model._rpu_vision_handle
                ),
            RHINOVLA_ACTION_COMPONENT:
                torch.ops.rpu.rhino_vla_get_resolved_chunk_size(
                    action_expert._rpu_handle
                ),
        }
        started = {
            component: int(chunk) for component, chunk in dispatched.items()
            if int(chunk) != 0
        }
        if started:
            raise RuntimeError(
                "RhinoVLA execution children must be bound before their first "
                f"forward; already dispatched={started}"
            )
        torch.ops.rpu.causal_decoder_set_chunk_size_override(
            text_model._rpu_decoder_handle, _native_chunk(resolved[1], "prefill")
        )
        torch.ops.rpu.qwen3vl_vision_set_chunk_size(
            vision_model._rpu_vision_handle, _native_chunk(resolved[2], "vision")
        )
        torch.ops.rpu.rhino_vla_set_chunk_size(
            action_expert._rpu_handle, _native_chunk(resolved[3], "action")
        )
        torch.ops.rpu.causal_decoder_enable_execution_reconfigure(
            text_model._rpu_decoder_handle
        )
        torch.ops.rpu.qwen3vl_vision_enable_execution_reconfigure(
            vision_model._rpu_vision_handle
        )
        torch.ops.rpu.rhino_vla_enable_execution_reconfigure(
            action_expert._rpu_handle
        )
        self._publish(resolved, generation=0)
        self.session = bind_execution_session(
            runtime,
            root,
            entry_point="RhinoVLAPolicy",
            supported_components=RHINOVLA_EXECUTION_COMPONENTS,
            validate=self.validate,
            apply=self.apply,
            rollback=self.rollback,
            graph_mode=GRAPH_COMPOSITE_CHILD,
        )
        vars(runtime)["_rhinovla_execution_controller"] = self
        for child in (self.text_model, self.vision_model, self.action_expert):
            vars(child)["_execution_session"] = self.session
        for child, component in (
            (self.text_model, RHINOVLA_TEXT_COMPONENT),
            (self.vision_model, RHINOVLA_VISION_COMPONENT),
            (self.action_expert, RHINOVLA_ACTION_COMPONENT),
        ):
            self.session._bind_planner_owner(child, component)

    def close(self):
        if vars(self.runtime).get("_rhinovla_execution_controller") is not self:
            error = RuntimeError("RhinoVLA retirement controller identity changed")
            _retirement_failure(self.runtime, error)
            raise error
        close_rhinovla_runtime(self.runtime)

    def __del__(self):
        owner = getattr(self, "runtime", None)
        if owner is not None:
            gc_close_rhinovla_runtime(owner)

    def _publish(self, resolved, *, generation, component_generations=None):
        root, text, vision, action = resolved
        if component_generations is not None:
            self._component_generations = dict(component_generations)
        self._generation = int(generation)
        vars(self.runtime)["_rpu_execution"] = root
        vars(self.runtime)["_rhinovla_execution_components"] = {
            RHINOVLA_TEXT_COMPONENT: text,
            RHINOVLA_VISION_COMPONENT: vision,
            RHINOVLA_ACTION_COMPONENT: action,
        }
        vars(self.runtime)["_text_execution"] = text
        vars(self.runtime)["_vision_execution"] = vision
        vars(self.runtime)["_action_execution"] = action
        vars(self.runtime)["_rhinovla_execution_component_generations"] = dict(
            self._component_generations
        )
        for child, component, config in (
            (self.text_model, RHINOVLA_TEXT_COMPONENT, text),
            (self.vision_model, RHINOVLA_VISION_COMPONENT, vision),
            (self.action_expert, RHINOVLA_ACTION_COMPONENT, action),
        ):
            state = vars(child)
            state["_rpu_execution"] = config
            state["_fmb_execution_component_id"] = component
            state["_fmb_execution_generation"] = int(
                self._component_generations[component]
            )
        vars(self.vision_model)["_rpu_execution_graph_key_words"] = (
            int(self._component_generations[RHINOVLA_VISION_COMPONENT]),
        )

    def validate(self, value):
        import torch

        resolve_rhinovla_execution(value, entry_point="RhinoVLAPolicy.reconfigure")
        _validate_pipeline_cold(self.runtime, value)
        required = (
            "execution_reconfigure_begin",
            "execution_reconfigure_commit",
            "execution_reconfigure_abort",
            "execution_reconfigure_abort_attempt",
            "causal_decoder_stage_chunk_size_override",
            "qwen3vl_vision_stage_chunk_size",
            "rhino_vla_stage_chunk_size",
        )
        missing = [name for name in required if not hasattr(torch.ops.rpu, name)]
        if missing:
            raise RuntimeError(
                "RhinoVLA binary lacks hot-reconfigure op(s): "
                + ", ".join(missing)
            )

    def _reset(self, changed):
        if RHINOVLA_TEXT_COMPONENT in changed:
            _reset_graph_cache(
                getattr(self.text_model, "_rpu_text_graph_cache", None),
                label="text",
            )
            vars(self.text_model).pop("_rpu_last_execution_plan", None)
        if RHINOVLA_VISION_COMPONENT in changed:
            _reset_graph_cache(
                getattr(self.vision_model, "_rpu_vision_graph_cache", None),
                label="vision",
            )
            vars(self.vision_model).pop("_rpu_last_execution_plan", None)
        if RHINOVLA_ACTION_COMPONENT in changed:
            _reset_graph_cache(
                getattr(self.action_expert, "_rpu_graph_cache", None),
                label="action",
            )
            cache = getattr(self.action_expert, "_rpu_kv_cache", None)
            if cache is not None:
                cache.reset_to_position(0)
            vars(self.action_expert)["_rpu_prefix_source_snapshot"] = None
            vars(self.action_expert).pop("_rpu_last_execution_plan", None)

    def _apply_resolved(
        self, resolved, *, generation, component_generations=None,
        force_components=(),
    ):
        import torch

        old_components = getattr(
            self.runtime, "_rhinovla_execution_components", {}
        )
        new_components = dict(zip(
            RHINOVLA_EXECUTION_COMPONENTS, resolved[1:]
        ))
        changed = {
            component for component in RHINOVLA_EXECUTION_COMPONENTS
            if old_components.get(component) != new_components[component]
        }
        changed.update(force_components)
        if component_generations is None:
            component_generations = dict(self._component_generations)
            for component in changed:
                component_generations[component] += 1
        if changed:
            with native_execution_reconfigure(
                torch.ops.rpu, journal=self._journal,
            ) as token:
                if RHINOVLA_TEXT_COMPONENT in changed:
                    torch.ops.rpu.causal_decoder_stage_chunk_size_override(
                        self.text_model._rpu_decoder_handle,
                        token,
                        _native_chunk(new_components[RHINOVLA_TEXT_COMPONENT], "prefill"),
                    )
                if RHINOVLA_VISION_COMPONENT in changed:
                    torch.ops.rpu.qwen3vl_vision_stage_chunk_size(
                        self.vision_model._rpu_vision_handle,
                        token,
                        _native_chunk(new_components[RHINOVLA_VISION_COMPONENT], "vision"),
                    )
                if RHINOVLA_ACTION_COMPONENT in changed:
                    torch.ops.rpu.rhino_vla_stage_chunk_size(
                        self.action_expert._rpu_handle,
                        token,
                        _native_chunk(new_components[RHINOVLA_ACTION_COMPONENT], "action"),
                    )
            vars(self.vision_model)["_rpu_vision_execution_chunk_size"] = (
                new_components[RHINOVLA_VISION_COMPONENT]
                .get("vision", {}).get("chunk_size", "auto")
            )
            self._reset(changed)
        if self._journal is not None:
            self._journal["mutation_started"] = True
        self._publish(
            resolved,
            generation=generation,
            component_generations=component_generations,
        )

    def apply(self, old, new, generation, *, force_rebuild=False):
        self._journal = {"mutation_started": False}
        old_resolved = resolve_rhinovla_execution(
            old, entry_point="RhinoVLAPolicy.reconfigure rollback"
        )
        new_resolved = resolve_rhinovla_execution(
            new, entry_point="RhinoVLAPolicy.reconfigure"
        )
        old_components = dict(zip(
            RHINOVLA_EXECUTION_COMPONENTS, old_resolved[1:]
        ))
        new_components = dict(zip(
            RHINOVLA_EXECUTION_COMPONENTS, new_resolved[1:]
        ))
        self._journal = {
            "mutation_started": False,
            "resolved": old_resolved,
            "generation": self._generation,
            "component_generations": dict(self._component_generations),
            "changed": {
                component for component in RHINOVLA_EXECUTION_COMPONENTS
                if force_rebuild or old_components[component] != new_components[component]
            },
        }
        self._apply_resolved(
            new_resolved, generation=generation,
            force_components=self._journal["changed"] if force_rebuild else (),
        )
        # Retain undo state through the shared session's config publication.

    def rollback(self, old, _new, generation):
        journal = self._journal
        if journal is not None and not journal["mutation_started"]:
            self._journal = None
            return
        if journal is None:
            resolved = resolve_rhinovla_execution(
                old, entry_point="RhinoVLAPolicy.reconfigure rollback"
            )
            force = set(RHINOVLA_EXECUTION_COMPONENTS)
            component_generations = dict(self._component_generations)
        else:
            resolved = journal["resolved"]
            generation = journal["generation"]
            force = journal["changed"]
            component_generations = journal["component_generations"]
        self._apply_resolved(
            resolved,
            generation=generation,
            component_generations=component_generations,
            force_components=force,
        )
        self._journal = None

    def collect_receipts(self):
        receipts = {}
        for child, component, stage in (
            (self.text_model, RHINOVLA_TEXT_COMPONENT, "prefill"),
            (self.vision_model, RHINOVLA_VISION_COMPONENT, "vision"),
            (self.action_expert, RHINOVLA_ACTION_COMPONENT, "action"),
        ):
            receipt = getattr(child, "_rpu_last_execution_plan", None)
            if not isinstance(receipt, dict):
                raise RuntimeError(
                    f"RhinoVLA {component} did not publish a planner receipt"
                )
            if receipt.get("component") != component:
                raise RuntimeError(
                    f"RhinoVLA {component} receipt has wrong component identity"
                )
            if receipt.get("stage") != stage:
                raise RuntimeError(
                    f"RhinoVLA {component} receipt has wrong stage identity"
                )
            expected_generation = self._component_generations[component]
            if int(receipt.get("generation", -1)) != expected_generation:
                raise RuntimeError(
                    f"RhinoVLA {component} receipt generation is stale"
                )
            if (
                receipt.get("authority") != "NATIVE_A6_STAGE_DESCRIPTOR"
                or not receipt.get("dry_forward_agreement", False)
            ):
                raise RuntimeError(
                    f"RhinoVLA {component} receipt lacks native descriptor authority"
                )
            receipts[stage] = copy.deepcopy(receipt)
        return receipts

    def clear_receipts(self):
        for child in (self.text_model, self.vision_model, self.action_expert):
            vars(child).pop("_rpu_last_execution_plan", None)


def bind_rhinovla_execution_runtime(
    runtime,
    *,
    text_model,
    vision_model,
    action_expert,
    rpu_execution=None,
):
    """Bind the three explicit physical children of an external runtime."""
    from rpu_backend.api._execution import _require_execution_process_safe

    # Fail before adopting/detaching any child finalizer when another retirement
    # has already made native state unsafe.
    _require_execution_process_safe()
    existing = getattr(runtime, "_rhinovla_execution_controller", None)
    if existing is not None:
        if (not isinstance(existing, _RhinoVLAExecutionController)
            or existing.runtime is not runtime
            or
            existing.text_model is not text_model
            or existing.vision_model is not vision_model
            or existing.action_expert is not action_expert
        ):
            raise ValueError("RhinoVLA runtime physical children changed")
        return existing
    from contextlib import nullcontext
    from rpu_backend.api._execution import ExecutionSession

    session = vars(runtime).get("_execution_session")
    if session is not None and not isinstance(session, ExecutionSession):
        raise TypeError(
            "bind_rhinovla_execution_runtime: _execution_session has an invalid type"
        )
    try:
        with session._lock if session is not None else nullcontext():
            controller = _RhinoVLAExecutionController(
                runtime, text_model, vision_model, action_expert, rpu_execution
            )
    except BaseException as error:
        if vars(runtime).get("_rhinovla_retirement_children"):
            try:
                # Adoption means weights already materialized. A failed bind
                # must retain the live claim until the failed owner is collected.
                close_rhinovla_runtime(runtime, _release_claim=False)
            except BaseException as cleanup_error:
                error.add_note(
                    f"RhinoVLA execution bind cleanup failed: {cleanup_error!r}"
                )
        raise
    vars(runtime)["_rhinovla_execution_controller"] = controller
    return controller


def build_rpu_expert(
    expert: Any,
    *,
    prefix_len: int,
    suffix_len: int,
    max_seq_len: int,
    w8a16: bool = False,
    full_w8a16: bool = False,
) -> Any:
    """Build one expert whose runtime prefix may vary within ``max_seq_len``.

    ``prefix_len`` is only the initial position used while installing the
    handle. Every forward resolves physical KV rows and logical RoPE rows from
    its actual prefix/cache and right-padding mask.
    ``w8a16`` opts into the exact v3 ordinary-loop expert projection profile;
    its seven main projections are quantized before swizzling.
    """
    import torch

    from .convert import convert_expert_for_rpu
    from .fused import patch_rhino_vla_for_rpu
    from rpu_backend.api._execution import _require_execution_process_safe

    _require_execution_process_safe()
    torch.rpu.set_ddr_flush(True)
    converted = copy.deepcopy(expert)
    cpu_rotary = copy.deepcopy(converted.qwen_rotary_emb).cpu()
    converted = convert_expert_for_rpu(converted, w8a16=w8a16 or full_w8a16)
    patch_rhino_vla_for_rpu(
        converted,
        prefix_len=int(prefix_len),
        suffix_len=int(suffix_len),
        max_seq_len=int(max_seq_len),
        cpu_rotary_emb=cpu_rotary,
        **({"full_w8a16": True} if full_w8a16 else {}),
    )
    return converted


__all__ = [
    "RHINOVLA_ACTION_COMPONENT",
    "RHINOVLA_EXECUTION_COMPONENTS",
    "RHINOVLA_TEXT_COMPONENT",
    "RHINOVLA_VISION_COMPONENT",
    "bind_rhinovla_execution_runtime",
    "build_rpu_expert",
    "resolve_rhinovla_execution",
]
