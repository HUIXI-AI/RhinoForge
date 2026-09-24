"""Shared execution authority for the controlled InternVLA-N1 composite."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from threading import RLock
from typing import Any, Mapping

from rpu_backend.api._execution import (
    bind_execution_session,
    bind_rpu_execution,
    native_execution_reconfigure,
    normalize_rpu_execution,
    resolve_component_rpu_execution,
)
from rpu_backend.runtime.execution_planner import (
    GRAPH_COMPOSITE_CHILD,
    publish_component_execution_receipt,
)
from rpu_backend.runtime.decoder import plan_native_component_execution


BACKBONE_COMPONENT = "system2_backbone"
VISION_COMPONENT = "system2_vision"
RGB_COMPONENT = "rgb_tower"
DEPTH_COMPONENT = "depth_tower"
FORMER_COMPONENT = "rgbd_former"
NAVDP_COMPONENT = "navdp_action"
NEXTDIT_COMPONENT = "nextdit_action"

_NEXTDIT_DDR_ROUTE_REASONS = {
    "cross_attention": "PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI",
}
_NAVDP_DDR_ROUTE_REASONS = {
    "cross_attention": "MEMORY_CROSS_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI",
}

INTERNVLA_EXECUTION_COMPONENTS = {
    BACKBONE_COMPONENT: {
        "prefill": ("chunk_size", "padding_rows", "padding_budget"),
    },
    VISION_COMPONENT: {"vision": ("chunk_size",)},
    # The controlled DINO towers execute their fixed 257-token ABI.  257 is
    # intentionally not exposed through the public v16 exact-chunk grammar.
    RGB_COMPONENT: {"vision": ()},
    DEPTH_COMPONENT: {"vision": ()},
    FORMER_COMPONENT: {"vision": ("chunk_size",)},
    NAVDP_COMPONENT: {"action": ("chunk_size",)},
    NEXTDIT_COMPONENT: {"action": ("chunk_size",)},
}


def resolve_internvla_execution(value, *, entry_point: str):
    """Resolve one immutable parent mapping and all stable child views."""
    root = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=INTERNVLA_EXECUTION_COMPONENTS,
    )
    resolved = {}
    for component, capabilities in INTERNVLA_EXECUTION_COMPONENTS.items():
        stage = next(iter(capabilities))
        profile = (
            {stage: {"chunk_size": "auto"}}
            if "chunk_size" in capabilities[stage] else {}
        )
        resolved[component] = resolve_component_rpu_execution(
            root,
            component,
            entry_point=entry_point,
            supported_components=INTERNVLA_EXECUTION_COMPONENTS,
            profile_auto=profile,
        )
    return root, resolved


def component_execution(value, component: str, *, entry_point: str):
    """Resolve a component-local config without accepting sibling fields."""
    capabilities = INTERNVLA_EXECUTION_COMPONENTS[component]
    stage = next(iter(capabilities))
    local = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported=capabilities,
    )
    profile = (
        {stage: {"chunk_size": "auto"}}
        if "chunk_size" in capabilities[stage] else {}
    )
    return resolve_component_rpu_execution(
        local,
        component,
        entry_point=entry_point,
        supported_components={component: capabilities},
        profile_auto=profile,
    )


def plan_navdp_component(
    handle: int,
    packed_rows: int,
    *,
    component: str,
    generation: int,
    execution,
    denoise_steps: int = 0,
    execution_owner=None,
    graph_cache=None,
):
    """Resolve NavDP/Former through the native finite stage domain."""
    import torch

    stage_name = next(iter(INTERNVLA_EXECUTION_COMPONENTS[component]))
    return plan_native_component_execution(
        packed_rows,
        execution_owner=execution_owner,
        execution_component=component,
        execution_native=("navdp", int(handle)),
        component_id=component,
        stage=stage_name,
        generation=generation,
        execution=execution,
        resolve_stage_domain=lambda rows: (
            torch.ops.rpu.navdp_resolve_stage_domain(
                handle, int(rows), int(denoise_steps)
            )
        ),
        graph_mode=GRAPH_COMPOSITE_CHILD,
        request_id=f"internvla:{component}:{stage_name}",
        physical_metadata=(
            ("cross_attention_ddr_required", 1),
            (
                "route_reason:MEMORY_CROSS_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI",
                1,
            ),
            ("denoise_steps", int(denoise_steps)),
        ),
        queue_owner_id=int(handle),
        plan_signature=(int(denoise_steps),),
        graph_cache=graph_cache,
    )


def publish_execution_receipt(
    owner: Any,
    plan,
    *,
    component: str,
    stage: str,
    generation: int,
    logical_len: int,
    resolved_chunk_size: int,
):
    extra_fields = {}
    if component == NEXTDIT_COMPONENT:
        metadata = (
            {} if plan.selected is None
            else dict(plan.selected.stage_tuple.physical_metadata)
        )
        extra_fields["self_attention_policy"] = (
            "RAW_SPM"
            if metadata.get("raw_attention_site_count", 0) > 0
            else "DDR_REQUIRED"
        )
        extra_fields["cross_attention_policy"] = "DDR_REQUIRED"
        extra_fields["route_reasons"] = dict(_NEXTDIT_DDR_ROUTE_REASONS)
    elif component in (NAVDP_COMPONENT, FORMER_COMPONENT):
        extra_fields["cross_attention_policy"] = "DDR_REQUIRED"
        extra_fields["route_reasons"] = dict(_NAVDP_DDR_ROUTE_REASONS)
    return publish_component_execution_receipt(
        owner,
        plan,
        component=component,
        stage=stage,
        generation=generation,
        logical_len=logical_len,
        resolved_chunk_size=resolved_chunk_size,
        extra_fields=extra_fields,
    )


def _clear_graphs(child: Any) -> None:
    """Invalidate only Graph owners physically belonging to one child."""
    seen = set()
    for name in ("_graph_cache", "_gc", "gc"):
        cache = getattr(child, name, None)
        if cache is None or id(cache) in seen:
            continue
        seen.add(id(cache))
        cache.begin_warmup()
        cache.clear()
        if not cache.cache_invariant_ok():
            raise RuntimeError(
                f"InternVLA {type(child).__name__} GraphCache invariant failed"
            )
    state = vars(child)
    state.pop("_rpu_last_execution_plan", None)
    plans = state.get("_rpu_last_execution_plans")
    if isinstance(plans, dict):
        plans.pop(state.get("_fmb_execution_component_id"), None)
    receipts = state.get("_rpu_last_execution_receipts")
    if isinstance(receipts, dict):
        receipts.pop(state.get("_fmb_execution_component_id"), None)


_FAILED_RETIREMENTS = []


def _retirement_session(owner):
    controller = vars(owner).get("_internvla_execution_controller")
    return (controller.session if controller is not None
            else getattr(owner, "_execution_session", None))


def _retirement_failure(owner, error):
    from rpu_backend.api import _execution

    if not any(value is owner for value in _FAILED_RETIREMENTS):
        _FAILED_RETIREMENTS.append(owner)
    vars(owner)["_internvla_retirement_failed"] = error
    session = _retirement_session(owner)
    if session is not None:
        session.poison()
    _execution._mark_execution_process_unsafe(f"InternVLA retirement failed: {error}")


def own_internvla_child(owner, child):
    """Transfer an actual installed child, including during partial construction."""
    from rpu_backend.runtime._native_retirement import _InstalledNativeResource
    from rpu_backend.adapters.wall_oss.vision import WallOssVision

    state = vars(owner)
    if (state.get("_internvla_closed") or state.get("_internvla_retiring")
            or state.get("_internvla_retirement_failed")):
        raise RuntimeError("InternVLA owner is closed or failed")
    children = state.setdefault("_internvla_retirement_children", [])
    if any(item[0] is child for item in children):
        return child
    child_state = vars(child)
    resource = child_state.get("_native_resource")
    graphs = tuple(dict.fromkeys(
        child_state[name] for name in ("_graph_cache", "_gc", "gc")
        if child_state.get(name) is not None
    ))
    if "_internvla_retirement_lock" not in state:
        state["_internvla_retirement_lock"] = RLock()
    if isinstance(resource, _InstalledNativeResource):
        if (resource.owner() is not child or resource.handle is None
                or type(child_state.get(resource.handle_name)) is not int
                or child_state.get(resource.handle_name) != resource.handle
                or resource.failed is not None):
            raise RuntimeError("InternVLA child native resource identity changed")
        parent = resource.parent() if resource.parent is not None else None
        if parent is not None and parent is not owner:
            raise RuntimeError("InternVLA child has a different retirement owner")
    elif isinstance(child, WallOssVision):
        parent = child_state.get("_retirement_owner")
        if parent is not None and parent is not owner:
            raise RuntimeError("InternVLA Vision has a different retirement owner")
    elif any(child_state.get(name) is not None for name in ("handle", "_handle")):
        raise RuntimeError("InternVLA live child lacks its native retirement resource")
    else:
        return child  # Config-only planner consumers have no native lifetime.
    children.append((child, resource, graphs, child_state.get("_handle")))
    previous_parent = resource.parent if resource is not None else None
    try:
        if resource is not None:
            resource.take_ownership(owner)
        else:
            child_state["_retirement_owner"] = owner
            child_state["_gc_retirement_enabled"] = False
    except BaseException:
        if resource is not None and resource.finalizer.alive:
            resource.parent = previous_parent
            list.pop(children)
        elif resource is not None and resource.parent is not None and resource.parent() is owner:
            # detach may have completed before an interruption. Keep the actual
            # parent record; never leave a detached native resource ownerless.
            pass
        else:
            list.pop(children)
        raise
    return child


def close_internvla_runtime(owner, *, _gc=False):
    """Retire all parent graphs before any handle; keep the bundle on failure."""
    from rpu_backend.api import _execution

    state = vars(owner)
    if state.get("_internvla_retirement_failed") is not None:
        raise RuntimeError("InternVLA retirement already failed; restart the process")
    if state.get("_internvla_closed"):
        return
    children = state.get("_internvla_retirement_children", ())
    session = _retirement_session(owner)

    def retire():
        try:
            _execution._require_execution_process_safe()
            for child, resource, graphs, original_handle in children:
                child_state = vars(child)
                current = {id(child_state[name]) for name in ("_graph_cache", "_gc", "gc")
                           if child_state.get(name) is not None}
                if current != {id(graph) for graph in graphs}:
                    raise RuntimeError("InternVLA retirement GraphCache identity changed")
                if resource is not None:
                    actual_owner = resource.owner()
                    parent = resource.parent() if resource.parent is not None else None
                    if (child_state.get("_native_resource") is not resource
                            or (actual_owner is not child and not (_gc and actual_owner is None))
                            or (parent is not owner and not (_gc and parent is None))
                            or resource.failed is not None
                            or resource.raw_graphs
                            or child_state.get(resource.handle_name) != resource.handle
                            or type(child_state.get(resource.handle_name)) is not type(resource.handle)
                            or (resource.handle is not None
                                and {id(graph) for graph in resource.graphs} != current)):
                        raise RuntimeError("InternVLA native retirement identity changed")
                elif (child_state.get("_retirement_owner") is not owner
                      or type(child_state.get("_handle")) is not int
                      or child_state.get("_handle") != original_handle
                      or child_state.get("_closed", False)):
                    raise RuntimeError("InternVLA Vision retirement identity changed")
            seen = set()
            for _child, _resource, graphs, _original_handle in children:
                for graph in graphs:
                    if id(graph) in seen:
                        continue
                    seen.add(id(graph))
                    graph.clear()
                    if not graph.cache_invariant_ok():
                        raise RuntimeError("InternVLA GraphCache invariant failed during close")
            for child, resource, _graphs, _original_handle in children:
                if resource is None:
                    child._retire_native()
                    vars(child)["_closed"] = True
                elif resource.handle is not None:
                    handle = resource.handle
                    resource.destroy(handle)
                    resource._native_destroyed(handle)
                    vars(child)[resource.handle_name] = None  # Cyclic GC cleared weakrefs.
                    # NextDiT's CPU carrier has an explicit back-reference.
                    carrier = vars(child).get("_owner")
                    if getattr(carrier, "_rpu_nextdit_runtime", None) is child:
                        vars(carrier)["_rpu_nextdit_runtime"] = None
            state["_internvla_closed"] = True
            from rpu_backend.api.causal_lm import _release_live_instance
            _release_live_instance(owner)
            # The construction journal is not an additional lifetime for DDR
            # tensors after the caller drops its normal component fields.
            state.pop("_internvla_retirement_children", None)
        except BaseException as error:
            _retirement_failure(owner, error)
            raise

    lock = session._lock if session is not None else state.get("_internvla_retirement_lock")
    with lock if lock is not None else nullcontext():
        if state.get("_internvla_retiring") or state.get("_internvla_closed"):
            return
        controller = state.get("_internvla_execution_controller")
        if (state.get("_execution_session") is not session
                or (controller is not None and controller.owner is not owner)
                or any(vars(child).get("_execution_session") is not session
                       for child, _resource, _graphs, _handle in children)):
            error = RuntimeError("InternVLA retirement controller/child Session identity changed")
            _retirement_failure(owner, error)
            raise error
        if session is not None and session._active:
            raise RuntimeError("cannot close InternVLA during a forward")
        state["_internvla_retiring"] = True
        try:
            from rpu_backend.api import causal_lm
            with causal_lm._LIVE_LOCK:
                live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                if live is not None and live is not owner:
                    raise RuntimeError("InternVLA retirement after live-owner handoff")
                if session is None:
                    retire()
                elif session._owner is not owner:
                    raise RuntimeError("InternVLA retirement Session owner changed")
                else:
                    session.shutdown(retire)
                if not state.get("_internvla_closed"):
                    raise RuntimeError("InternVLA closed Session still owns native resources")
        except BaseException as error:
            _retirement_failure(owner, error)
            raise
        finally:
            state.pop("_internvla_retiring", None)


def gc_close_internvla_runtime(owner):
    """Use the original parent close while cyclic GC still holds its children."""
    if (not vars(owner).get("_internvla_retirement_children")
            or vars(owner).get("_internvla_closed")
            or vars(owner).get("_internvla_retirement_failed") is not None):
        return
    session = _retirement_session(owner)
    try:
        with session._lock if session is not None else nullcontext():
            from rpu_backend.api import causal_lm
            with causal_lm._LIVE_LOCK:
                live = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                if live is not None and live is not owner:
                    raise RuntimeError("InternVLA GC after live-owner handoff")
                close_internvla_runtime(owner, _gc=True)
    except BaseException as error:
        _retirement_failure(owner, error)


class InternVLAExecutionController:
    """One serialized session for independently planned RPU children.

    CPU-only goal/scheduler glue is intentionally absent: it has no native
    chunk, SPM, KV, or Graph authority and is therefore NOT_APPLICABLE.
    """

    def __init__(
        self,
        owner: Any,
        children: Mapping[str, Any],
        rpu_execution=None,
    ) -> None:
        unknown = set(children) - set(INTERNVLA_EXECUTION_COMPONENTS)
        if unknown:
            raise ValueError(
                f"InternVLA execution has unknown child IDs {sorted(unknown)}"
            )
        if not children:
            raise ValueError("InternVLA execution requires at least one RPU child")
        if vars(owner).get("_internvla_execution_controller") is not None:
            raise ValueError("InternVLA owner already has its execution controller")
        for component, child in children.items():
            if (getattr(child, "_execution_session", None) is not None
                    or getattr(child, "_planner_cost_session", None) is not None):
                raise ValueError(
                    f"InternVLA {component} already owns an ExecutionSession; "
                    "construct embedded children with deferred session binding"
                )
        self.owner = owner
        self.children = dict(children)
        self._journal = None
        self._generation = 0
        self._component_generations = {
            component: 0 for component in self.children
        }
        root = bind_rpu_execution(
            owner,
            rpu_execution,
            entry_point="bind_internvla_execution_runtime",
            supported_components=INTERNVLA_EXECUTION_COMPONENTS,
        )
        resolved = resolve_internvla_execution(
            root, entry_point="bind_internvla_execution_runtime"
        )
        self._reject_absent_overrides(root)
        self._publish(resolved, generation=0)
        self.session = bind_execution_session(
            owner,
            root,
            entry_point="InternVLAExecutionController",
            supported_components=INTERNVLA_EXECUTION_COMPONENTS,
            validate=self.validate,
            apply=self.apply,
            rollback=self.rollback,
            graph_mode=GRAPH_COMPOSITE_CHILD,
        )
        vars(owner)["_internvla_execution_controller"] = self
        # Child production methods may use execution_serialized directly; all
        # of them then share this parent lock/generation rather than creating
        # one session per subsystem.
        for component, child in self.children.items():
            vars(child)["_execution_session"] = self.session
            self.session._bind_planner_owner(child, component)
            own_internvla_child(owner, child)

    def close(self):
        if vars(self.owner).get("_internvla_execution_controller") is not self:
            error = RuntimeError("InternVLA retirement controller identity changed")
            _retirement_failure(self.owner, error)
            raise error
        close_internvla_runtime(self.owner)

    def __del__(self):
        owner = getattr(self, "owner", None)
        if owner is not None:
            gc_close_internvla_runtime(owner)

    def _reject_absent_overrides(self, root) -> None:
        requested = set(root.get("components", {}))
        absent = requested - set(self.children)
        if absent:
            raise ValueError(
                "InternVLA execution config names absent component(s): "
                f"{sorted(absent)}"
            )
        for stage, fields in root.items():
            if stage == "components":
                continue
            unused = [
                field for field in fields
                if not any(
                    field in INTERNVLA_EXECUTION_COMPONENTS[component].get(
                        stage, ()
                    )
                    for component in self.children
                )
            ]
            if unused:
                raise ValueError(
                    "InternVLA execution config has no present child for "
                    f"{stage} field(s) {sorted(unused)}"
                )

    def _publish(self, resolved, *, generation, component_generations=None):
        root, configs = resolved
        if component_generations is not None:
            self._component_generations = dict(component_generations)
        self._generation = int(generation)
        vars(self.owner)["_rpu_execution"] = root
        vars(self.owner)["_internvla_execution_components"] = {
            component: configs[component] for component in self.children
        }
        vars(self.owner)["_internvla_execution_component_generations"] = dict(
            self._component_generations
        )
        for component, child in self.children.items():
            state = vars(child)
            state["_rpu_execution"] = configs[component]
            state["_fmb_execution_component_id"] = component
            state["_fmb_execution_generation"] = int(
                self._component_generations[component]
            )

    def validate(self, value) -> None:
        root, _configs = resolve_internvla_execution(
            value, entry_point="InternVLAExecutionController.reconfigure"
        )
        self._reject_absent_overrides(root)

    def apply(self, old, new, next_generation, *, force_rebuild=False) -> None:
        self._journal = None
        old_resolved = resolve_internvla_execution(
            old, entry_point="InternVLAExecutionController.apply.old"
        )
        new_resolved = resolve_internvla_execution(
            new, entry_point="InternVLAExecutionController.apply.new"
        )
        changed = {
            component for component in self.children
            if old_resolved[1][component] != new_resolved[1][component]
        }
        if force_rebuild:
            changed.update(self.children)
        generations = dict(self._component_generations)
        for component in changed:
            generations[component] += 1
        self._journal = (old_resolved, dict(self._component_generations), changed)
        # Even descriptor-only changes participate in the native stop-the-world
        # transaction so publication cannot race a live RPU graph.
        with native_execution_reconfigure():
            for component in changed:
                _clear_graphs(self.children[component])
            self._publish(
                new_resolved,
                generation=next_generation,
                component_generations=generations,
            )
        # ponytail: one bounded undo snapshot survives shared facade publication;
        # the next apply replaces it after the previous transaction completes.

    def rollback(self, _old, _new, old_generation) -> None:
        if self._journal is None:
            return
        old_resolved, generations, changed = self._journal
        with native_execution_reconfigure():
            for component in changed:
                _clear_graphs(self.children[component])
            self._publish(
                old_resolved,
                generation=old_generation,
                component_generations=generations,
            )
        self._journal = None

    @contextmanager
    def execute(self):
        """Serialize one complete composite forward/build→replay transaction."""
        with self.session.execute() as generation:
            yield generation

    @property
    def rpu_execution(self):
        return self.session.config

    def reconfigure_rpu_execution(self, value):
        return self.session.reconfigure(value)

    def stats(self):
        return self.session.stats()


def bind_internvla_execution_runtime(
    owner: Any,
    *,
    backbone=None,
    vision=None,
    rgb_tower=None,
    depth_tower=None,
    rgbd_former=None,
    navdp=None,
    nextdit=None,
    rpu_execution=None,
) -> InternVLAExecutionController:
    children = {
        component: child
        for component, child in (
            (BACKBONE_COMPONENT, backbone),
            (VISION_COMPONENT, vision),
            (RGB_COMPONENT, rgb_tower),
            (DEPTH_COMPONENT, depth_tower),
            (FORMER_COMPONENT, rgbd_former),
            (NAVDP_COMPONENT, navdp),
            (NEXTDIT_COMPONENT, nextdit),
        )
        if child is not None
    }
    controller = InternVLAExecutionController(
        owner, children, rpu_execution=rpu_execution
    )
    vars(owner)["_internvla_execution_controller"] = controller
    return controller


__all__ = [
    "BACKBONE_COMPONENT",
    "VISION_COMPONENT",
    "RGB_COMPONENT",
    "DEPTH_COMPONENT",
    "FORMER_COMPONENT",
    "NAVDP_COMPONENT",
    "NEXTDIT_COMPONENT",
    "INTERNVLA_EXECUTION_COMPONENTS",
    "InternVLAExecutionController",
    "bind_internvla_execution_runtime",
    "component_execution",
    "plan_navdp_component",
    "publish_execution_receipt",
    "resolve_internvla_execution",
]
