"""HALO action expert 单步驱动（HaloStepRunner）：handle 装载 + prefix KV 灌入 + step。

执行模型（一次 act step）：
  host (fp32) ──组装 x_emb──> RPU fp16 ──halo_action_expert_step_forward──> hidden
  └ in_proj(x_t)+temb+sinpos → 行 1..16；embed_tokens(text_ids) → 行 0/17；
    fp32 → bf16 → fp16（对齐 HALO CUDA bf16 流水的舍入语义）。
  hidden 行 1..16 → host fp32 out_proj → v_t [16,14]。

prefix KV 灌入：CUDA NaiveCache 捕获的 prefix K/V [4706,2,128] bf16（已 RoPE）
直接 host 侧 swizzle 进 7-D RPUCache（slot=head：head 0→chunk 槽 0，head 1→槽 1，
其余 6 个 replica 槽留 0）。该 slot 映射来源于 attn_tp=2 时
``rpu_launch_insert_kcache_spm_unified`` "core c 写槽 c" 的内核行为，与 SDPA
读侧一致性由 halo_sdpa_spm_probe_cached（2026-06-09 C gate）间接验证；host 侧
重建无独立硬件证据（not-proven 单点），由本仓 @hardware 测试覆盖。

图缓存：per-model ``GraphCache``（pi05/wall_oss 先例），每步只在
``capture(GraphSignature)`` 内调一次 ``torch.ops.rpu.halo_action_expert_step_forward``
（step0 BUILD，后续 REPLAY）。GraphSignature 把 (chunk, hidden, layers,
prefix, rope_position) 全部编进 key，prefix/位置改变即重建。

所有权：runner 持有 text/act 权重、cos/sin、掩码与 RPUCache，与 handle 等寿命
（C++ 仅存引用计数副本）。handle 经 ``weakref.finalize`` 回收。
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import os
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import torch

import rpu_backend
from rpu_backend.api._execution import (
    bind_execution_session,
    execution_guard,
    normalize_rpu_execution,
    resolve_component_rpu_execution,
)
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.halo.weights import (
    validate_halo_core_layout,
    HaloBranchWeights,
    HaloConfig,
    HaloHostWeights,
    build_rope_tables,
    build_text_row_mask,
    load_halo_branches,
    sinusoidal_pos_embedding,
    timestep_embedding,
)
from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
from rpu_backend.runtime.execution_planner import (
    GRAPH_COMPOSITE_CHILD,
    plan_fixed_component_execution,
)

_S_CHUNK = 16          # 7-D cache seq / head_dim 块宽（api/cache.py 同源常量）
_KV_SLOTS = 8          # effective_kv_slots = NUM_CORES * nkv / attn_tp = 8*2/2
_PREFIX_HISTORY_DDR_REASON = (
    "PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI"
)

HALO_ACTION_COMPONENT = "action_expert"
HALO_IMAGE_FLOW_COMPONENT = "image_flow"
HALO_TEXT_CACHE_COMPONENT = "text_cache"
HALO_TEXT_DECODE_COMPONENT = "text_decode"
HALO_VISION_COMPONENT = "vision_encoder"
HALO_VIT_LM_COMPONENT = "vit_language_model"

# The component IDs name physical owners, not logical model families.  Decode
# has no public execution knob, but remains registered so its fixed receipt and
# Graph owner have an independent, stable identity.
HALO_EXECUTION_COMPONENTS = {
    HALO_ACTION_COMPONENT: {"action": ("chunk_size",)},
    HALO_IMAGE_FLOW_COMPONENT: {"action": ("chunk_size",)},
    HALO_TEXT_CACHE_COMPONENT: {"prefill": ("chunk_size",)},
    HALO_TEXT_DECODE_COMPONENT: {"decode": ()},
    HALO_VISION_COMPONENT: {"vision": ("chunk_size",)},
    HALO_VIT_LM_COMPONENT: {"prefill": ("chunk_size",)},
}

_HALO_COMPONENT_AUTO = {
    HALO_ACTION_COMPONENT: {"action": {"chunk_size": "auto"}},
    HALO_IMAGE_FLOW_COMPONENT: {"action": {"chunk_size": "auto"}},
    HALO_TEXT_CACHE_COMPONENT: {"prefill": {"chunk_size": "auto"}},
    HALO_TEXT_DECODE_COMPONENT: {},
    HALO_VISION_COMPONENT: {"vision": {"chunk_size": "auto"}},
    HALO_VIT_LM_COMPONENT: {"prefill": {"chunk_size": "auto"}},
}


def resolve_halo_execution(value, *, entry_point: str):
    """Resolve one canonical HALO mapping into six physical child views."""
    root = normalize_rpu_execution(
        value,
        entry_point=entry_point,
        supported_components=HALO_EXECUTION_COMPONENTS,
    )
    children = {
        component: resolve_component_rpu_execution(
            root,
            component,
            entry_point=entry_point,
            supported_components=HALO_EXECUTION_COMPONENTS,
            profile_auto=_HALO_COMPONENT_AUTO[component],
        )
        for component in HALO_EXECUTION_COMPONENTS
    }
    _validate_halo_execution(children, entry_point=entry_point)
    return root, children


def _validate_halo_execution(children, *, entry_point: str) -> None:
    action_chunk = children[HALO_ACTION_COMPONENT]["action"]["chunk_size"]
    if isinstance(action_chunk, int) and action_chunk != 32:
        raise ValueError(
            f"{entry_point}: action_expert action.chunk_size must be 'auto' "
            f"or exactly 32 for the fixed 18-row HALO action ABI; got "
            f"{action_chunk}"
        )


def _clear_halo_graph(cache, *, component: str) -> None:
    if cache is None:
        return
    begin_warmup = getattr(cache, "begin_warmup", None)
    if begin_warmup is not None:
        begin_warmup()
    cache.clear()
    invariant = getattr(cache, "cache_invariant_ok", None)
    if invariant is not None and not invariant():
        raise RuntimeError(
            f"HALO {component} GraphCache invariant failed during reconfigure"
        )


def _require_halo_process_safe() -> None:
    from rpu_backend.api import causal_lm

    with causal_lm._LIVE_LOCK:
        if causal_lm._LIVE_TERMINAL_REASON is not None:
            raise RuntimeError("HALO resources cannot run in an unsafe process; restart the process")


def _halo_execution_serialized(method):
    """Keep closed staged children out of an otherwise live shared Session."""
    @functools.wraps(method)
    def wrapped(owner, *args, **kwargs):
        with execution_guard(owner):
            _require_halo_process_safe()
            if getattr(owner, "_closed", False):
                raise RuntimeError("HALO component is closed")
            return method(owner, *args, **kwargs)
    return wrapped


# A failed last-owner GC cannot rely on a weak owner or a self-cycle to keep
# graph/native addresses alive. Retain only terminal failures until exit.
_FAILED_RETIREMENTS = []


class _HaloResources:
    """Pending/built resources shared by explicit close and weakref cleanup."""

    def __init__(self, controller=None, *, handles=None):
        # A strong Controller reference can reach its native planner owner and
        # keep the finalized runner alive through the finalizer registry.
        self.controller_ref = None if controller is None else weakref.ref(controller)
        self.handles = {} if handles is None else handles
        self.graphs = []
        self.finalizers = {}
        self.owner_ref = None
        self.failed = None
        self.closed = False

    @property
    def controller(self):
        return None if self.controller_ref is None else self.controller_ref()

    def retire(self, *, from_gc=False):
        session = getattr(self.controller, "_execution_session", None)
        with session._lock if session is not None else contextlib.nullcontext():
            if self.failed:
                raise RuntimeError(f"HALO retirement already failed: {self.failed}")
            if self.closed:
                return
            if session is not None and session._active and not from_gc:
                raise RuntimeError("cannot close HALO resources during an active forward")
            try:
                if session is not None and session._active:
                    raise RuntimeError("HALO GC retirement encountered an active forward")
                _require_halo_process_safe()
                if session is not None and (session._closed or session._poisoned):
                    raise RuntimeError("cannot retire HALO resources in a closed or poisoned Session")
                if not from_gc:
                    for name, finalizer in self.finalizers.items():
                        if name in self.handles and not finalizer.alive:
                            raise RuntimeError("HALO handle has a dead finalizer without confirmed retirement")
                # Every graph must retire before the first native owner does.
                seen = set()
                for graph in self.graphs:
                    if id(graph) in seen:
                        continue
                    seen.add(id(graph))
                    graph.clear()
                    invariant = getattr(graph, "cache_invariant_ok", None)
                    if invariant is not None and not invariant():
                        raise RuntimeError("HALO GraphCache invariant failed during close")
                owner = None if self.owner_ref is None else self.owner_ref()
                for name, (handle, destroy) in tuple(self.handles.items()):
                    destroy(handle)  # Raw destroy, never the best-effort finalizer.
                    del self.handles[name]
                    if owner is not None:
                        setattr(owner, name, None)
                    finalizer = self.finalizers.get(name)
                    if finalizer is not None:
                        finalizer.detach()
                if self.controller is not None:
                    self.controller._detach(owner)
                if owner is not None:
                    owner._closed = True
                self.closed = True
            except BaseException as cleanup_error:
                self.failed = repr(cleanup_error)
                _FAILED_RETIREMENTS.append(self)
                if session is not None:
                    session.poison()
                from rpu_backend.api import causal_lm

                with causal_lm._LIVE_LOCK:
                    if self.controller is not None:
                        self.controller._halo_failed_retirement = self
                    if causal_lm._LIVE_TERMINAL_REASON is None:
                        owner = causal_lm._LIVE_REF() if causal_lm._LIVE_REF is not None else None
                        if owner is None:
                            owner = self.controller or self
                            causal_lm._claim_live_instance(owner)
                        vars(owner)["_halo_failed_retirement"] = self
                        causal_lm._poison_live_instance(owner, f"HALO cleanup failed: {cleanup_error!r}", unsafe=True)
                    from rpu_backend.api._execution import _mark_execution_process_unsafe

                    _mark_execution_process_unsafe(f"HALO cleanup failed: {cleanup_error!r}")
                raise

    def bind(self, owner):
        self.owner_ref = weakref.ref(owner)
        owner._halo_resources = self
        for name in self.handles:
            finalizer = weakref.finalize(owner, _finalize_halo_resources, self)
            self.finalizers[name] = finalizer
            setattr(owner, f"{name}_finalizer", finalizer)


def _finalize_halo_resources(resources):
    if resources.failed:
        return  # Keep explicit failures terminal; GC must not retry a destroy.
    try:
        resources.retire(from_gc=True)
    except BaseException:
        pass  # retire() preserved the resources and latched Session/process failure.


def _close_halo_runner(owner, *, graphs, handles):
    resources = getattr(owner, "_halo_resources", None)
    if resources is None:
        resources = _HaloResources(getattr(owner, "_halo_execution_controller", None))
        resources.owner_ref = weakref.ref(owner)
        resources.graphs.extend(graphs)
        resources.handles.update(handles)
        resources.finalizers.update({
            name: finalizer for name in handles
            if (finalizer := getattr(owner, f"{name}_finalizer", None)) is not None
        })
        owner._halo_resources = resources
    resources.retire()


def _cleanup_halo_build(resources, build_error):
    try:
        resources.retire()
    except BaseException as cleanup_error:
        build_error.add_note(f"HALO builder cleanup also failed: {cleanup_error!r}")


class HaloExecutionController:
    """One stop-the-world session for HALO's independently cached children.

    Pass the same controller to staged builders that coexist in one process.
    Invalid fields fail before any builder loads weights. The fixed action
    expert accepts C32; other exact chunks must match a full descriptor in
    that request's native AUTO domain at forward time. Reconfiguration
    invalidates only the changed child's Graph owner, then builds/replays anew.
    """

    def __init__(self, rpu_execution: Mapping[str, Any] | None = None):
        root, children = resolve_halo_execution(
            rpu_execution, entry_point="HaloExecutionController"
        )
        self._rpu_execution = root
        self._halo_execution_components = children
        self._component_generations = {
            component: 0 for component in HALO_EXECUTION_COMPONENTS
        }
        self._owners: dict[str, tuple[weakref.ReferenceType, tuple[Any, ...]]] = {}
        self._execution_session = bind_execution_session(
            self,
            root,
            entry_point="HaloExecutionController",
            supported_components=HALO_EXECUTION_COMPONENTS,
            validate=self._validate_reconfigure,
            apply=self._apply_reconfigure,
            graph_mode=GRAPH_COMPOSITE_CHILD,
        )

    @property
    def rpu_execution(self):
        return self._rpu_execution

    def component_config(self, component: str):
        return self._halo_execution_components[component]

    def component_generation(self, component: str) -> int:
        return int(self._component_generations[component])

    def _require_attachable(self, components):
        for component in components:
            if component not in HALO_EXECUTION_COMPONENTS:
                raise ValueError(f"unknown HALO execution component {component!r}")
            registered = self._owners.get(component)
            previous = None if registered is None else registered[0]()
            if previous is not None and not getattr(previous, "_closed", False):
                raise RuntimeError(f"HALO execution component {component!r} already has a live owner")
            session = self._execution_session
            if (any(key[0] == component for key in session._planner_costs)
                    or any(key[0] == component for key in session._planner_native_catalogs)):
                raise RuntimeError(
                    f"HALO retired component {component!r} has installed costs; "
                    "a replacement requires a fresh Controller and cost verification"
                )

    def attach(self, owner, component_graphs: Mapping[str, Sequence[Any]]) -> None:
        with self._execution_session._lock:
            _require_halo_process_safe()
            if getattr(owner, "_closed", False):
                raise RuntimeError("cannot attach a closed HALO component")
            self._require_attachable(component_graphs)
            entries = {}
            for component, graphs in component_graphs.items():
                entries[component] = (weakref.ref(owner), tuple(graphs))
            previous_binding = getattr(owner, "_planner_cost_session", None)
            try:
                for component in entries:
                    self._execution_session._bind_planner_owner(owner, component)
            except BaseException:
                if previous_binding is None:
                    vars(owner).pop("_planner_cost_session", None)
                else:
                    vars(owner)["_planner_cost_session"] = previous_binding
                raise
            self._owners.update(entries)
            state = vars(owner)
            state["_halo_execution_controller"] = self
            state["_execution_session"] = self._execution_session
            self._publish_owner(owner)

    def _detach(self, owner):
        retired = set()
        for component, (owner_ref, _graphs) in tuple(self._owners.items()):
            if owner_ref() is owner:
                del self._owners[component]
                retired.add(component)
        session = self._execution_session
        for key, (native_owner, _handle) in tuple(session._planner_native.items()):
            if native_owner is owner:
                del session._planner_native[key]
        for key in tuple(session._planner_native_revalidators):
            if key[0] in retired:
                del session._planner_native_revalidators[key]

    def _publish_owner(self, owner) -> None:
        components = {
            component: self._halo_execution_components[component]
            for component, (owner_ref, _graphs) in self._owners.items()
            if owner_ref() is owner
        }
        state = vars(owner)
        state["_rpu_execution"] = self._rpu_execution
        state["_halo_execution_components"] = components
        state["_halo_execution_component_generations"] = {
            component: self._component_generations[component]
            for component in components
        }

    def _validate_reconfigure(self, value) -> None:
        _root, children = resolve_halo_execution(value, entry_point="HALO.reconfigure_rpu_execution")
        for component, stage in (
            (HALO_TEXT_CACHE_COMPONENT, "prefill"),
            (HALO_VISION_COMPONENT, "vision"),
            (HALO_VIT_LM_COMPONENT, "prefill"),
        ):
            requested = children[component][stage]["chunk_size"]
            if requested == "auto" or children[component] == self._halo_execution_components[component]:
                continue
            registered = self._owners.get(component)
            owner = None if registered is None else registered[0]()
            plan = getattr(owner, "_rpu_last_execution_plans", {}).get(component)
            if plan is not None and not any(
                candidate.stage_tuple.qkv_chunk == candidate.stage_tuple.compute_chunk == requested
                and (candidate.feasible or candidate.reject_reason == "exact_mismatch")
                for candidate in plan.candidates
            ):
                # Reject before retirement for the last actual request. A cold
                # or changed input still validates its own native domain before
                # forward mutates inputs/KV; this is not a new certificate.
                raise ValueError(f"HALO {component}: exact chunk {requested} is outside the last native request domain")

    def _apply_reconfigure(self, old_value, new_value, generation: int, *, force_rebuild=False) -> None:
        _old_root, old_children = resolve_halo_execution(
            old_value, entry_point="HALO.reconfigure_rpu_execution"
        )
        new_root, new_children = resolve_halo_execution(
            new_value, entry_point="HALO.reconfigure_rpu_execution"
        )
        changed = {
            component
            for component in HALO_EXECUTION_COMPONENTS
            if force_rebuild or old_children[component] != new_children[component]
        }
        for component in changed:
            registered = self._owners.get(component)
            if registered is not None:
                owner = registered[0]()
                if owner is not None:
                    for graph in registered[1]:
                        _clear_halo_graph(graph, component=component)
                    vars(owner).get("_rpu_last_execution_plans", {}).pop(
                        component, None
                    )
                    vars(owner).get("_rpu_last_execution_receipts", {}).pop(
                        component, None
                    )
            self._component_generations[component] += 1
        self._rpu_execution = new_root
        self._halo_execution_components = new_children
        for owner_ref, _graphs in self._owners.values():
            owner = owner_ref()
            if owner is not None:
                self._publish_owner(owner)

    def reconfigure_rpu_execution(self, value):
        return self._execution_session.reconfigure(value)


def _coerce_halo_execution_controller(
    rpu_execution,
    execution_controller: HaloExecutionController | None,
    *,
    entry_point: str,
    components: Sequence[str],
) -> HaloExecutionController:
    _require_halo_process_safe()
    if execution_controller is None:
        execution_controller = HaloExecutionController(rpu_execution)
    if not isinstance(execution_controller, HaloExecutionController):
        raise TypeError(
            f"{entry_point}: execution_controller must be a "
            "HaloExecutionController"
        )
    execution_controller._execution_session.require_cold()
    with execution_controller._execution_session._lock:
        execution_controller._require_attachable(components)
    if rpu_execution is not None:
        root, _children = resolve_halo_execution(
            rpu_execution, entry_point=entry_point
        )
        if root != execution_controller.rpu_execution:
            raise ValueError(
                f"{entry_point}: rpu_execution conflicts with the shared "
                "HaloExecutionController"
            )
    return execution_controller


def _halo_component_state(owner, component: str, stage: str):
    from rpu_backend.api._execution import ExecutionSession, _planner_owner_binding

    controller = getattr(owner, "_halo_execution_controller", None)
    session = getattr(owner, "_execution_session", None)
    if (not isinstance(controller, HaloExecutionController)
            or not isinstance(session, ExecutionSession)
            or vars(session) is not vars(controller._execution_session)):
        raise RuntimeError("HALO planning requires the actual Controller and execution Session")
    session = controller._execution_session
    registered = controller._owners.get(component)
    if (registered is None or registered[0]() is not owner
            or _planner_owner_binding(owner, stage, component)[0] is not session):
        raise RuntimeError("HALO planning requires the registered component owner and cost Session")
    return (
        controller.component_config(component).get(stage, {}),
        controller.component_generation(component),
    )


def _record_halo_plan(
    owner,
    component: str,
    stage: str,
    plan,
    *,
    authority: str,
    kv_route: str | None = None,
) -> None:
    vars(owner).setdefault("_rpu_last_execution_plans", {})[component] = plan
    receipt = plan.as_dict(include_candidates=False)
    receipt.update({
        "component": component,
        "stage": stage,
        "authority": authority,
        "semantic_spans": "INPUT_OWNED",
    })
    if kv_route is not None:
        receipt["kv_route"] = kv_route
        receipt["attention_policy"] = kv_route
        receipt["kv_route_reason"] = _PREFIX_HISTORY_DDR_REASON
        receipt["attention_reason"] = _PREFIX_HISTORY_DDR_REASON
    vars(owner).setdefault("_rpu_last_execution_receipts", {})[component] = receipt


def _plan_halo_fmb_execution(
    owner,
    *,
    component: str,
    stage: str,
    logical_len: int,
    position: int,
    handle: int,
    native_prefix: str,
    resolve_stage_domain,
    kv_route: str | None = None,
    graph_cache=None, plan_signature=(),
):
    """Query the live native domain; only a successful caller records execution."""
    stage_config, generation = _halo_component_state(owner, component, stage)
    requested_chunk = stage_config.get("chunk_size", "auto")
    box = {}
    metadata = [
        (f"component:{component}", 1),
        ("execution_generation", int(generation)),
    ]
    if kv_route == "DDR_REQUIRED":
        metadata.append(("kv_route:DDR_REQUIRED", 1))
        metadata.append((f"route_reason:{_PREFIX_HISTORY_DDR_REASON}", 1))
    plan_bounded_prefill_execution(
        logical_len,
        logical_len,
        0,
        execution_owner=owner,
        execution_stage=stage,
        execution_component=component,
        execution_native=(native_prefix, int(handle)),
        position=position,
        alignment=1,
        padding_rows=0,
        exact_chunk_size=(
            None if requested_chunk == "auto" else int(requested_chunk)
        ),
        resolve_stage_domain=resolve_stage_domain,
        plan_result_sink=lambda result: box.__setitem__("result", result),
        graph_mode=GRAPH_COMPOSITE_CHILD,
        queue_owner_id=int(handle),
        physical_metadata=metadata,
        plan_signature=plan_signature,
        graph_cache=graph_cache,
    )
    plan = box["result"]
    descriptor = plan.selected.stage_tuple.physical_descriptor
    if not descriptor:
        raise RuntimeError(f"HALO {component} A6 winner has no native descriptor")
    return plan, descriptor


def _plan_halo_fixed_execution(
    owner,
    *,
    component: str,
    stage: str,
    logical_len: int,
    execution_len: int,
    chunk_size: int,
    position: int,
    handle: int,
    kv_route: str | None = None,
):
    stage_config, generation = _halo_component_state(owner, component, stage)
    metadata = []
    if kv_route == "DDR_REQUIRED":
        metadata.append(("kv_route:DDR_REQUIRED", 1))
        metadata.append((f"route_reason:{_PREFIX_HISTORY_DDR_REASON}", 1))
    plan = plan_fixed_component_execution(
        logical_len,
        execution_len=execution_len,
        chunk_size=chunk_size,
        component_id=component,
        stage=stage,
        generation=generation,
        position=position,
        stage_config=stage_config,
        graph_mode=GRAPH_COMPOSITE_CHILD,
        physical_metadata=metadata,
        queue_owner_id=int(handle),
    )
    _record_halo_plan(
        owner,
        component,
        stage,
        plan,
        authority="FIXED_ABI_SINGLETON",
        kv_route=kv_route,
    )
    return plan


def _halo_profile_ctx():
    """``HALO_PROFILE=1`` 时开 ``torch.profiler``,否则 nullcontext(零开销)。

    RPU op 在 profiler 里 = 每高层调用一个 fused 事件(``test_v5_e2e_perf.py`` 同口径);
    ``GraphCache.capture(sig)`` 已把每次 forward 体裹进 ``record_function(sig.op_id)``,故
    ``halo_action_expert_step_cfg`` 等 op_id 自动成 profiler 事件,无需手动埋点。off 时
    ``with`` 的 ``as`` 目标为 ``None`` → 调用方据此跳过 dump。Profile-only 机制,不改算路。
    """
    if os.environ.get("HALO_PROFILE") != "1":
        return contextlib.nullcontext()
    import torch.profiler
    activities = [torch.profiler.ProfilerActivity.CPU]
    if hasattr(torch.profiler.ProfilerActivity, "PrivateUse1"):
        activities.append(torch.profiler.ProfilerActivity.PrivateUse1)
    return torch.profiler.profile(activities=activities)


def _dump_halo_profile(prof, tag: str) -> None:
    """导出 key_averages 表 + chrome trace 到 ``HALO_PROFILE_DIR``(默认 output/halo_profile)。"""
    if prof is None:
        return
    out_dir = Path(os.environ.get("HALO_PROFILE_DIR", "output/halo_profile"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    (out_dir / f"{tag}.optable.txt").write_text(table)
    prof.export_chrome_trace(str(out_dir / f"{tag}.trace.json"))
    print(f"[HALO_PROFILE] {tag} → {out_dir}/{tag}.{{optable.txt,trace.json}}", flush=True)
    print(table, flush=True)


def euler_action_schedule(num_timesteps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """HALO ``generate_action`` 去噪 schedule（源 `bagel.py:921-925`，逐位核对）。

    固定 Euler 环（无早停）：``timesteps = linspace(1, 0, num_timesteps)``；循环跑
    ``timesteps[:-1]`` 共 ``num_timesteps − 1`` 步；``dts = timesteps[:-1] − timesteps[1:]``
    （均匀 ``dt = 1/(num_timesteps − 1)``，nt=10 → 1/9）；更新式 ``x_t = x_t − v_t·dt[i]``
    （减号，t:1→0 反向 Euler）。返回 ``(timesteps[:-1], dts)``，长度均 ``num_timesteps − 1``。

    部署 ``num_timesteps`` 在 `halo_inferencer.py:155` **硬编码 10**（不可配）。
    **``timestep_shift`` 生成端 no-op**：`generate_action` 接收该形参但函数体不引用，
    schedule 是裸 linspace，无 shift 扭曲。
    """
    if num_timesteps < 2:
        raise ValueError(f"num_timesteps must be >= 2, got {num_timesteps}")
    timesteps = torch.linspace(1, 0, num_timesteps)
    return timesteps[:-1], timesteps[:-1] - timesteps[1:]


@dataclasses.dataclass(frozen=True)
class HaloRollout:
    """整轨去噪结果（:meth:`HaloStepRunner.rollout_cond` 等返回）。

    ``final_action`` [16,14] fp32 = `generate_action` 返回的最终 ``x_t``（验收单位）。
    ``per_step_v_t`` 逐步 v_t [16,14]（diagnostic：跨步漂移曲线）。
    ``timesteps`` / ``dts`` 实际 Euler schedule（长 ``num_timesteps − 1``），冻进结果便于核对。
    """

    final_action: torch.Tensor
    per_step_v_t: List[torch.Tensor]
    timesteps: torch.Tensor
    dts: torch.Tensor


@dataclasses.dataclass(frozen=True)
class HaloCfgBranch:
    """一个 CFG 分支:prefix-KV 银行(已 swizzle 进 cache)+ rope_position 表 + kv_len。

    三分支(cond / text-cfg / img-cfg)共用 :class:`HaloStepRunner` 的 handle / 权重 /
    x_emb,只换这里的 ``cache``(prefix-KV 银行)+ ``cos``/``sin``(rope_position)+
    ``prefix_len``(kv_len)。``branch_id`` 进 GraphSignature 确保各分支独立 BUILD —— 每个
    BUILD 捕获自己 cache / cos-sin 的 DMA 指针(三分支 cache 是不同张量),REPLAY 各自命中。
    (由 :meth:`HaloStepRunner.make_cfg_branch`(经 :func:`build_cfg_branch`)构造。)
    """

    name: str
    branch_id: int
    cache: RPUCache
    cos: torch.Tensor                 # rope_position 的 RoPE 表 [total_len,head_dim/2] fp16 RPU
    sin: torch.Tensor
    prefix_len: int                   # 该分支 kv_len(cond 4706 / text 4698 / img 8)
    rope_position: int
    attn_mask: torch.Tensor           # [1,1,cs,prefix_len+cs] fp16 全 0 —— SDPA mask 宽随 kv_len 变
    # 分配代数:每次新建分支(新 RPUCache/cos/sin = 新 DMA 地址)由
    # ``make_cfg_branch`` 单调递增并写入。它进 ``_forward_branch`` 的 GraphSignature → 同一
    # runner 跨 episode 复用、(branch_id,prefix_len,rope_position) 不变时,新分配仍产新 key →
    # 强制重 BUILD,杜绝旧 graph 硬编码的上轮 cache/cos DMA 指针被陈旧 replay(静默污染动作)。
    epoch: int


def _destroy_handle(handle: int) -> None:
    """Raw destroy; the common retirement boundary owns error handling."""
    torch.ops.rpu.halo_action_expert_destroy(handle)


def _swizzle_prefix_7d(
    prefix: torch.Tensor, max_seq_len: int, value_layout: bool
) -> torch.Tensor:
    """prefix [P,2,128] → 7-D cache CPU fp16（slot=head，replica 槽 0）。

    K 布局 [1,sVx,1,8, 8,16,16]（…,sChunk,dChunk），V 末两维转置。
    rows [P, max_seq) 留 0：[P, P+18) 由 step_forward 的 insert 覆盖。
    """
    p_len, nkv, hd = prefix.shape
    s_vx = max_seq_len // _S_CHUNK
    d_vx = hd // _S_CHUNK
    pad = torch.zeros(max_seq_len, nkv, hd, dtype=torch.float16)
    pad[:p_len] = prefix.to(torch.float16)
    out = torch.zeros(1, s_vx, 1, d_vx, _KV_SLOTS, _S_CHUNK, _S_CHUNK, dtype=torch.float16)
    for head in range(nkv):
        # [maxS,hd] → [sVx,16,dVx,16] → (sVx,dVx,16,16)；V 交换 (sChunk,dChunk)。
        blocks = pad[:, head, :].view(s_vx, _S_CHUNK, d_vx, _S_CHUNK)
        blocks = blocks.permute(0, 2, 3, 1) if value_layout else blocks.permute(0, 2, 1, 3)
        out[0, :, 0, :, head] = blocks
    return out.contiguous()


@dataclasses.dataclass
class HaloStepRunner:
    """HALO action expert 单步 runner（经 :func:`build_halo_action_expert` 构造）。

    字段所有权：全部 RPU 权重/表/cache 在 runner 存活期内有效；``_handle``
    由 finalize 释放。``rope_position`` 是 query 行共享的常数 position。
    """

    cfg: HaloConfig
    rope_position: int
    text: HaloBranchWeights
    act: HaloBranchWeights
    host: HaloHostWeights
    cache: RPUCache
    _handle: int
    _graph_cache: "rpu_backend.graph.GraphCache"
    _attn_mask: torch.Tensor                 # [1,1,cs,total_len] fp16 CPU，全 0
    # RoPE 表(rope_position 的 cos/sin,[total_len,head_dim/2] fp16 RPU)。改为
    # per-forward 传入 step_forward(对齐 qwenpi05 cos_ref_),CFG 三分支各传自己的表;
    # set_weights 仍收一份作基类契约+默认。runner 持有 cond 分支的表(与 handle 等寿命)。
    _cos: torch.Tensor
    _sin: torch.Tensor
    _x_emb: torch.Tensor = dataclasses.field(init=False)       # [1,cs,H] fp16 RPU
    _handle_finalizer: object | None = dataclasses.field(
        init=False, default=None, repr=False
    )
    _closed: bool = dataclasses.field(init=False, default=False, repr=False)
    # prefix 代数：每次 insert_prefix 换新 RPU 张量（DMA 指针变化），编进
    # GraphSignature 强制重 BUILD，禁止旧 graph 指针重放。（insert_prefix/step 路用。）
    _prefix_epoch: int = dataclasses.field(init=False, default=0)
    # CFG 分支分配代数：rollout_cfg/_forward_branch 路用。每次
    # make_cfg_branch 新建分支银行（新 DMA 地址）单调递增 → 进 _forward_branch 签名 →
    # 跨 episode 复用同 runner 时新分配仍重 BUILD（_prefix_epoch 只随 insert_prefix 动，
    # CFG 路从不调它，故须独立计数；镜像 image_flow._epoch_counter）。
    _cfg_branch_epoch: int = dataclasses.field(init=False, default=0)

    def __post_init__(self) -> None:
        cfg = self.cfg
        validate_halo_core_layout(cfg)
        # 稳定输入缓冲：BUILD 期烘焙 x_emb 的 DMA 地址，REPLAY 复用（先例 wall_oss
        # x buf）。输出走基类 output_tensor_（registry 跨 forward 稳定），无需独立
        # out buf —— 在 capture 退出后读 step_forward 的返回张量即可。
        self._x_emb = torch.empty(
            1, cfg.chunk_size, cfg.hidden_size, dtype=torch.float16, device="rpu")

    def close(self) -> None:
        """Clear graph ownership, then retire the native handle exactly once."""
        _close_halo_runner(self, graphs=(self._graph_cache,),
                           handles={"_handle": (self._handle, _destroy_handle)})

    # ── prefix ────────────────────────────────────────────────────────────
    @_halo_execution_serialized
    def insert_prefix(self, k_layers: List[torch.Tensor], v_layers: List[torch.Tensor]) -> None:
        """灌入 28 层 prefix K/V（NaiveCache 捕获，CPU bf16/fp32 [4706,2,128]）。

        K/V 已带原始 RoPE；host swizzle 后整体替换 cache 张量（与 C++ 按
        data_ptr DMA 的借用语义一致：step_forward 每次重新取列表）。
        """
        cfg = self.cfg
        expect = (cfg.prefix_len, cfg.num_kv_heads, cfg.head_dim)
        if len(k_layers) != cfg.num_layers or len(v_layers) != cfg.num_layers:
            raise ValueError(
                f"prefix layers mismatch: got {len(k_layers)}/{len(v_layers)}, "
                f"expect {cfg.num_layers}")
        for i, (k, v) in enumerate(zip(k_layers, v_layers)):
            if tuple(k.shape) != expect or tuple(v.shape) != expect:
                raise ValueError(
                    f"layer {i} prefix shape {tuple(k.shape)}/{tuple(v.shape)} != {expect}")

        # A prefix bank is one semantic input.  Prepare every allocation and
        # upload before publishing any layer so a late failure cannot leave a
        # mixture of old and new DMA owners under the old Graph epoch.
        next_k_caches = []
        next_v_caches = []
        for k, v in zip(k_layers, v_layers):
            next_k_caches.append(_swizzle_prefix_7d(
                k, self.cache.max_seq_len, value_layout=False).to("rpu"))
            next_v_caches.append(_swizzle_prefix_7d(
                v, self.cache.max_seq_len, value_layout=True).to("rpu"))

        # reset_to_position validates before mutating.  The execution guard
        # keeps this publication indivisible with respect to step/close, and
        # the new epoch prevents replay of Graphs captured with the old bank.
        self.cache.reset_to_position(cfg.prefix_len)
        self.cache.k_caches, self.cache.v_caches = next_k_caches, next_v_caches
        self._prefix_epoch += 1

    # ── host boundary ─────────────────────────────────────────────────────
    def _assemble_x_emb(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """host fp32 boundary → [1,cs,H] fp16 CPU（行 1..16 action，行 0/17 text）。

        与 HALO `_forward_action_flow` 逐项一致：
        action 行 = in_proj(x_t) + temb(timestep) + sinpos(action_position_ids)，
        text 行 = embed_tokens(text_ids)；fp32 → bf16 → fp16。
        """
        cfg, host = self.cfg, self.host
        x_act = x_t.float() @ host.in_proj_w.t() + host.in_proj_b          # [16,H]
        t_freq = timestep_embedding(timestep, cfg)                          # [16,256]
        temb = torch.nn.functional.silu(
            t_freq @ host.temb0_w.t() + host.temb0_b) @ host.temb2_w.t() + host.temb2_b
        sinpos = sinusoidal_pos_embedding(action_position_ids, cfg)         # [16,H]
        x_act = x_act + temb + sinpos

        x_emb = torch.zeros(cfg.chunk_size, cfg.hidden_size, dtype=torch.float32)
        x_emb[list(cfg.action_rows)] = x_act
        x_emb[list(cfg.text_rows)] = torch.nn.functional.embedding(
            text_ids, self.host.embed_tokens).float()
        return x_emb.to(torch.bfloat16).to(torch.float16).unsqueeze(0)

    # ── step ──────────────────────────────────────────────────────────────
    def _execution_plan(self, position: int):
        """Query the fixed action domain without preparing inputs or executing."""
        # The builder binds this same capacity to the native handle envelope.
        if (type(position) is not int or position < 0
                or position + self.cfg.chunk_size > self.cache.max_seq_len):
            raise ValueError("HALO action planning position exceeds the native owner cache capacity")
        return _plan_halo_fmb_execution(
            self,
            component=HALO_ACTION_COMPONENT,
            stage="action",
            logical_len=self.cfg.chunk_size,
            position=position,
            handle=self._handle,
            native_prefix="halo_action_expert",
            resolve_stage_domain=lambda _length: (
                torch.ops.rpu.halo_action_expert_resolve_stage_domain(
                    self._handle, position, 0
                )
            ),
            kv_route="DDR_REQUIRED",
            graph_cache=self._graph_cache,
            plan_signature=(),
        )

    @torch.no_grad()
    @_halo_execution_serialized
    def step(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_ids: torch.Tensor,
    ) -> torch.Tensor:
        """一次 act step。I/O 契约（全 CPU）：

          x_t [16,14] fp32；timestep [16] fp32（HALO 断言全行同值）；
          action_position_ids [16] int64；text_ids [2] int64 → v_t [16,14] fp32。

        每步先 reset_to_position(prefix)（K/V [4706,4724) 被本步 insert 覆盖）。
        proof 边界：数值正确性由 @hardware 测试对 CUDA golden gate，本函数
        自身不构成硬件证据。
        """
        cfg = self.cfg
        if self._prefix_epoch == 0:
            raise RuntimeError("HaloStepRunner.step: call insert_prefix() first")
        if tuple(x_t.shape) != (len(cfg.action_rows), cfg.action_dim):
            raise ValueError(f"x_t must be [16,{cfg.action_dim}], got {tuple(x_t.shape)}")
        plan, descriptor = self._execution_plan(cfg.prefix_len)
        self.cache.reset_to_position(cfg.prefix_len)
        self._x_emb.copy_(self._assemble_x_emb(x_t, timestep, action_position_ids, text_ids))
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_action_expert_step",
            shapes=[cfg.chunk_size, cfg.hidden_size],
            dyn_dims=[cfg.num_layers, cfg.prefix_len, self.rope_position,
                      self._prefix_epoch, *plan.graph_key_words()],
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            hidden_out = torch.ops.rpu.halo_action_expert_step_forward(
                self._handle, self._x_emb,
                self.cache.k_caches, self.cache.v_caches,
                self._cos, self._sin,                 # per-forward RoPE 表(cond 分支)
                self._attn_mask, cfg.prefix_len, descriptor)
        # 必须在 capture 作用域**退出后**读：返回的是基类 output_tensor_（graph 写
        # 目标），图在 with 块退出 end() 时才执行写入；作用域内读到的是尚未执行的
        # 全 0（已证模式 wall_oss/pi05；见 C++ step_forward 注释与基类契约）。
        hidden = hidden_out[0].float().cpu()                                # [cs,H]
        act = hidden[list(cfg.action_rows)]                                 # [16,H]
        result = act @ self.host.out_proj_w.t() + self.host.out_proj_b        # [16,14]
        _record_halo_plan(
            self, HALO_ACTION_COMPONENT, "action", plan,
            authority="NATIVE_FMB_DESCRIPTOR", kv_route="DDR_REQUIRED",
        )
        return result

    @torch.no_grad()
    @_halo_execution_serialized
    def rollout_cond(
        self,
        x_t_init: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_ids: torch.Tensor,
        *,
        num_timesteps: int = 10,
    ) -> HaloRollout:
        """cond-only 整轨去噪(无 CFG,单分支)。复刻 `generate_action` 固定 Euler 环。

        I/O 契约(全 CPU)：``x_t_init`` [16,14] fp32(= ``packed_init_noises``);
        ``action_position_ids`` [16] int64、``text_ids`` [2] int64(跨步常数)。每步:
        ``timestep = full(16, t_i)``(全行同值,满足 `_forward_action_flow` 断言),
        ``v_t = step(x_t, t_i, …)``,``x_t = x_t − v_t·dt[i]``;返回 final action +
        逐步 v_t(diagnostic)。

        **cond-only**:仅注入 cond prefix-KV(调用前须 :meth:`insert_prefix`);CFG
        三分支(text-cfg/img-cfg + global renorm)在后续含-CFG rollout 叠加。timestep 经
        mutable ``_x_emb`` 喂入、**不进 GraphSignature** → step0 BUILD、余步 REPLAY
        (build-once/replay-many,与单步 dispatch 测试同 replay 路径;跨步只换 x_emb 缓冲)。

        proof 边界:整轨数值正确性由 @hardware 测试对 CUDA golden gate;本函数自身
        不构成硬件证据(单步 :meth:`step` 同此口径)。
        """
        cfg = self.cfg
        if self._prefix_epoch == 0:
            raise RuntimeError("HaloStepRunner.rollout_cond: call insert_prefix() first")
        expect = (len(cfg.action_rows), cfg.action_dim)
        if tuple(x_t_init.shape) != expect:
            raise ValueError(f"x_t_init must be {list(expect)}, got {tuple(x_t_init.shape)}")
        timesteps, dts = euler_action_schedule(num_timesteps)
        x_t = x_t_init.to(torch.float32).clone()
        per_step_v_t: List[torch.Tensor] = []
        with _halo_profile_ctx() as _prof:                                 # HALO_PROFILE=1 才开
            for i in range(int(timesteps.shape[0])):
                timestep = torch.full(
                    (len(cfg.action_rows),), float(timesteps[i]), dtype=torch.float32)
                v_t = self.step(x_t, timestep, action_position_ids, text_ids)   # [16,14]
                per_step_v_t.append(v_t)
                x_t = x_t - v_t * float(dts[i])
        _dump_halo_profile(_prof, "halo_rollout_cond")
        return HaloRollout(
            final_action=x_t, per_step_v_t=per_step_v_t, timesteps=timesteps, dts=dts)

    # ── CFG 三分支 ─────────────────────────────────────────────────────────
    @_halo_execution_serialized
    def make_cfg_branch(
        self,
        name: str,
        branch_id: int,
        k_layers: List[torch.Tensor],
        v_layers: List[torch.Tensor],
        *,
        rope_position: int,
        prefix_len: int,
        max_seq_len: int = 4736,
    ) -> HaloCfgBranch:
        """构造带分配身份(epoch)的 CFG 分支 —— **生产路径唯一入口**。

        每次调用单调递增 ``self._cfg_branch_epoch`` 并注入新分支银行(新 RPUCache/cos/sin =
        新 DMA 地址),使 ``_forward_branch`` 的 GraphSignature 随每次新分配而变 → 同一 runner
        跨 episode 复用、(branch_id,prefix_len,rope_position) 不变时仍重 BUILD,杜绝旧 graph
        硬编码的上轮 DMA 指针被陈旧 replay。镜像 ``image_flow._prepare_branch``。
        """
        self._cfg_branch_epoch += 1
        return build_cfg_branch(
            name, branch_id, k_layers, v_layers,
            rope_position=rope_position, prefix_len=prefix_len,
            epoch=self._cfg_branch_epoch, cfg=self.cfg, max_seq_len=max_seq_len)

    def _forward_branch(self, branch: HaloCfgBranch) -> torch.Tensor:
        """一个 CFG 分支的 native forward → v_t [16,14]。

        ``self._x_emb`` 须已由调用方写入(三分支同一 x_emb;query 跨分支不变)。分支只换
        cache(prefix-KV 银行)/ cos-sin(rope_position)/ prefix_len(kv_len);共用 handle。
        GraphSignature 含 ``branch_id`` → 各分支独立 BUILD(捕获自己 cache/cos DMA 指针),
        step0 BUILD、后续步同分支 REPLAY。返回前必须在 capture 退出**后**读 hidden_out
        (基类 output_tensor_,图在 with 退出才写;三分支共用 output_tensor_,故每分支结果
        须在下个分支 forward 前读走 —— 本方法读后即返回,满足此序)。
        """
        cfg = self.cfg
        if (type(branch.prefix_len) is not int or branch.prefix_len < 0
                or branch.prefix_len + cfg.chunk_size > branch.cache.max_seq_len):
            raise ValueError("HALO action branch position exceeds its actual cache capacity")
        plan, descriptor = self._execution_plan(branch.prefix_len)
        branch.cache.reset_to_position(branch.prefix_len)
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_action_expert_step_cfg",
            shapes=[cfg.chunk_size, cfg.hidden_size],
            # 5 维拓扑契约 = _cfg_branch_dyn_dims（生产与 board-free 负测同源；含 branch.epoch
            # 分配身份,见该函数 docstring）。
            dyn_dims=[*_cfg_branch_dyn_dims(cfg, branch), *plan.graph_key_words()],
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            hidden_out = torch.ops.rpu.halo_action_expert_step_forward(
                self._handle, self._x_emb,
                branch.cache.k_caches, branch.cache.v_caches,
                branch.cos, branch.sin,
                branch.attn_mask, branch.prefix_len, descriptor)
        hidden = hidden_out[0].float().cpu()                                # [cs,H]
        act = hidden[list(cfg.action_rows)]                                 # [16,H]
        result = act @ self.host.out_proj_w.t() + self.host.out_proj_b        # [16,14]
        _record_halo_plan(
            self, HALO_ACTION_COMPONENT, "action", plan,
            authority="NATIVE_FMB_DESCRIPTOR", kv_route="DDR_REQUIRED",
        )
        return result

    @torch.no_grad()
    @_halo_execution_serialized
    def rollout_cfg(
        self,
        x_t_init: torch.Tensor,
        action_position_ids: torch.Tensor,
        text_ids: torch.Tensor,
        *,
        cond: HaloCfgBranch,
        text: HaloCfgBranch,
        img: HaloCfgBranch,
        cfg_text_scale: float,
        cfg_img_scale: float,
        cfg_interval: Sequence[float],
        cfg_renorm_min: float = 0.0,
        num_timesteps: int = 10,
    ) -> HaloRollout:
        """含-CFG 整轨去噪。复刻 `generate_action` + `_forward_action_flow` 三分支 + global renorm。

        每步对同一 x_t 跑 cond/text-cfg/img-cfg 三分支(各自 prefix-KV 银行 + rope_position +
        kv_len,共用 handle/权重/x_emb)→ host global renorm 合并 → ``x_t = x_t − v_t·dt``。
        CFG 判定 ``cfg_interval[0] < t ≤ cfg_interval[1]``(bagel.py:929-931);in-interval 步
        跑 3 分支组合,区间外仅 cond(scales=1)。组合(bagel.py:1087-1104,global):
        ``v_text_ = text + s_text·(cond − text)``;``v_ = img + s_img·(v_text_ − img)``;
        ``v_t = v_ · clamp(‖cond‖/‖v_‖, min, 1)``(整张量标量范数,host)。

        部署口径(用户 2026-06-15 拍板,golden output/halo_gen_action_golden/nt10/):
        ``cfg_text_scale=4.0 / cfg_img_scale=2.0 / cfg_interval=[0.0,1.0] / nt=10``。
        proof 边界:整轨数值正确性由 @hardware 测试对 CUDA golden gate。
        """
        cfg = self.cfg
        expect = (len(cfg.action_rows), cfg.action_dim)
        if tuple(x_t_init.shape) != expect:
            raise ValueError(f"x_t_init must be {list(expect)}, got {tuple(x_t_init.shape)}")
        timesteps, dts = euler_action_schedule(num_timesteps)
        x_t = x_t_init.to(torch.float32).clone()
        per_step_v_t: List[torch.Tensor] = []
        with _halo_profile_ctx() as _prof:                                 # HALO_PROFILE=1 才开
            for i in range(int(timesteps.shape[0])):
                t = float(timesteps[i])
                timestep = torch.full((len(cfg.action_rows),), t, dtype=torch.float32)
                # x_emb 三分支共用,每步写一次(query 跨分支不变,跨步随 x_t/timestep 变)。
                self._x_emb.copy_(self._assemble_x_emb(x_t, timestep, action_position_ids, text_ids))
                v_cond = self._forward_branch(cond)                         # cond 总跑
                # CFG 两个 scale 独立:区间内,text / img 各自 >1
                # 才跑对应分支。旧实现整块门控在 cfg_text_scale>1.0 → 会静默忽略「仅 image
                # CFG」(text=1.0, img>1.0)。部署 text=4/img=2 都 >1,下式与旧逐位一致。
                in_cfg = (cfg_interval[0] < t <= cfg_interval[1]
                          and (cfg_text_scale > 1.0 or cfg_img_scale > 1.0))
                if in_cfg:
                    v_text_ = v_cond
                    if cfg_text_scale > 1.0:
                        v_text = self._forward_branch(text)
                        v_text_ = v_text + cfg_text_scale * (v_cond - v_text)
                    if cfg_img_scale > 1.0:
                        v_img = self._forward_branch(img)
                        v_ = v_img + cfg_img_scale * (v_text_ - v_img)
                    else:
                        v_ = v_text_
                    scale = (v_cond.norm() / (v_.norm() + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t = v_ * scale                                       # global renorm
                else:
                    v_t = v_cond                                          # 区间外/无 CFG 仅 cond
                per_step_v_t.append(v_t)
                x_t = x_t - v_t * float(dts[i])
        _dump_halo_profile(_prof, "halo_rollout_cfg")
        return HaloRollout(
            final_action=x_t, per_step_v_t=per_step_v_t, timesteps=timesteps, dts=dts)

    def dispatch_submit_total(self) -> int:
        """图缓存累计的真实 /dev/rpu 批提交次数（GraphStats.hw_batch_submit_total 之和）。

        >0 证明 action 专家的 kernel 批确被同步发射到设备（Queue_t::enqueu_batch
        wait_finish=true），而非仅建图 / 走 replay 指针；用作 native dispatch 硬件
        证据（治理门 hardware_dispatch_proven 的 int>0 路径）。须在 ≥1 次 step() 后读。
        """
        return sum(int(s.hw_batch_submit_total) for s in self._graph_cache.snapshot())


def build_halo_action_expert(
    ckpt_path: "str | Path",
    *,
    rope_position: int,
    cfg: HaloConfig = HaloConfig(),
    max_seq_len: int = 4736,
    rpu_execution=None,
    execution_controller: HaloExecutionController | None = None,
) -> HaloStepRunner:
    """装载 HALO action expert：权重 → handle → set_weights/set_moe_weights。

    Args:
        ckpt_path: HALO ema.safetensors（bf16）。
        rope_position: query 行共享的常数 position（step0 冻结输入 = 11）。
        max_seq_len: KV cache 容量，须为 16 的倍数且 ≥ prefix+cs（4724→4736）。
    """
    validate_halo_core_layout(cfg)
    execution_controller = _coerce_halo_execution_controller(
        rpu_execution,
        execution_controller,
        entry_point="build_halo_action_expert",
        components=(HALO_ACTION_COMPONENT,),
    )
    if max_seq_len % _S_CHUNK != 0 or max_seq_len < cfg.total_len:
        raise ValueError(
            f"max_seq_len {max_seq_len} must be a multiple of {_S_CHUNK} and >= {cfg.total_len}")
    text, act, host = load_halo_branches(Path(ckpt_path), cfg)
    cos, sin = build_rope_tables(cfg, rope_position)
    handle = int(torch.ops.rpu.halo_action_expert_create())
    resources = _HaloResources(execution_controller, handles={"_handle": (handle, _destroy_handle)})
    try:
        torch.ops.rpu.halo_action_expert_set_weights(
            handle,
            text.q_w, text.k_w, text.v_w, text.o_w,
            text.q_norm, text.k_norm, text.input_norm, text.post_norm,
            text.gate_w, text.up_w, text.down_w,
            cos, sin, text.final_norm_w,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
            cfg.hidden_size, cfg.intermediate_size,
            cfg.rms_norm_eps, True,
            text.q_bias, text.k_bias, text.v_bias)
        torch.ops.rpu.halo_action_expert_set_moe_weights(
            handle,
            act.q_w, act.k_w, act.v_w, act.o_w,
            act.q_norm, act.k_norm, act.input_norm, act.post_norm,
            act.gate_w, act.up_w, act.down_w,
            act.q_bias, act.k_bias, act.v_bias,
            act.final_norm_w, build_text_row_mask(cfg), cfg.chunk_size)
        torch.ops.rpu.halo_action_expert_set_chunk_envelope(
            handle, int(max_seq_len), 32
        )
        # cache / attn_mask / runner 也必须在 try 内：weakref.finalize 只在
        # HaloStepRunner.__post_init__ 安装，若这几步失败而 handle 已建，孤儿 handle
        # 不会被回收（"一进程一个 live RPU model" 约束下重试会留下注册实例 + RPU 资源）。
        cache = RPUCache(
            num_layers=cfg.num_layers, batch_size=1, max_seq_len=max_seq_len,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)
        attn_mask = torch.zeros(
            1, 1, cfg.chunk_size, cfg.total_len, dtype=torch.float16)   # 非因果全注意，全 0
        graph = rpu_backend.graph.GraphCache()
        resources.graphs.append(graph)
        runner = HaloStepRunner(
            cfg=cfg, rope_position=rope_position, text=text, act=act, host=host,
            cache=cache, _handle=handle, _graph_cache=graph,
            _attn_mask=attn_mask, _cos=cos, _sin=sin)
        resources.bind(runner)
        execution_controller.attach(
            runner, {HALO_ACTION_COMPONENT: (runner._graph_cache,)}
        )
    except BaseException as build_error:
        # 构造任一步失败 → 孤儿 handle 立即回收（扩展自 pi05 orphan-handle cleanup；
        # 原仅裹 set_weights/set_moe_weights，遗漏 cache/runner）。
        _cleanup_halo_build(resources, build_error)
        raise
    return runner


def _cfg_branch_dyn_dims(cfg: HaloConfig, branch: HaloCfgBranch) -> List[int]:
    """``_forward_branch`` 的 GraphSignature ``dyn_dims`` —— 生产与 board-free 负测**单一真相源**。

    5 维:``num_layers`` / ``prefix_len``(=kv_len)/ ``rope_position`` / ``branch_id``
    (cache 银行身份)/ ``epoch``(分配代数,新分配 → 新 key → 重 BUILD)。
    ``timestep`` / cfg-scales 故意**不**进 key(不进 RPU 图:timestep 经 mutable ``_x_emb``
    缓冲、scales 在 host renorm;进 key 会破坏 build-once/replay-many + 让 replay 稳定门失效)。
    """
    return [cfg.num_layers, branch.prefix_len, branch.rope_position,
            branch.branch_id, branch.epoch]


def build_cfg_branch(
    name: str,
    branch_id: int,
    k_layers: List[torch.Tensor],
    v_layers: List[torch.Tensor],
    *,
    rope_position: int,
    prefix_len: int,
    epoch: int,
    cfg: HaloConfig = HaloConfig(),
    max_seq_len: int = 4736,
) -> HaloCfgBranch:
    """从一个 CFG 分支的 prefix-KV 银行构造 :class:`HaloCfgBranch`(供 :meth:`HaloStepRunner.rollout_cfg`)。

    ``k_layers``/``v_layers`` = 该分支 CUDA NaiveCache 捕获的 28 层 prefix K/V
    (``[P,2,128]`` bf16/fp32,已带原始 RoPE,P=该分支 kv_len)。swizzle 进新 RPUCache +
    建该 ``rope_position`` 的 cos/sin 表。三分支各调一次(cond/text-cfg/img-cfg)。
    不持 handle —— 共用 :class:`HaloStepRunner` 的句柄/权重(rollout_cfg 调度)。

    ``epoch`` = 分配代数(必填,):标识本次新建的 cache/cos/sin DMA 身份,
    进 ``_forward_branch`` 签名。**生产路径请用 :meth:`HaloStepRunner.make_cfg_branch`**(它单调
    递增 runner 的 ``_cfg_branch_epoch`` 后传入),勿手凑 ``epoch``(同值跨 episode 会撞 key →
    陈旧 replay)。``epoch`` 无默认值即为此 guard —— 漏传 → TypeError(逼调用方走 make_cfg_branch)。
    """
    validate_halo_core_layout(cfg)
    if len(k_layers) != cfg.num_layers or len(v_layers) != cfg.num_layers:
        raise ValueError(
            f"{name}: prefix layers {len(k_layers)}/{len(v_layers)} != {cfg.num_layers}")
    if not 0 <= prefix_len <= max_seq_len:
        raise ValueError(f"{name}: prefix_len {prefix_len} out of [0,{max_seq_len}]")
    cache = RPUCache(
        num_layers=cfg.num_layers, batch_size=1, max_seq_len=max_seq_len,
        num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)
    for i, (k, v) in enumerate(zip(k_layers, v_layers)):
        cache.k_caches[i] = _swizzle_prefix_7d(k, max_seq_len, value_layout=False).to("rpu")
        cache.v_caches[i] = _swizzle_prefix_7d(v, max_seq_len, value_layout=True).to("rpu")
    cache.reset_to_position(prefix_len)
    cos, sin = build_rope_tables(cfg, rope_position)
    # SDPA mask 宽 = 该分支 kv 跨度 prefix_len + cs(非因果全注意,全 0);三分支 kv_len 不同 → 宽不同。
    attn_mask = torch.zeros(1, 1, cfg.chunk_size, prefix_len + cfg.chunk_size, dtype=torch.float16)
    return HaloCfgBranch(
        name=name, branch_id=branch_id, cache=cache, cos=cos, sin=sin,
        prefix_len=prefix_len, rope_position=rope_position, attn_mask=attn_mask, epoch=epoch)
