"""Board-free complete-ready contracts for the Pi0.5 composite adapter."""
from __future__ import annotations

import inspect
import sys
import types

import pytest

from rpu_backend.adapters import pi05
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.api.policy import Pi05Policy


class _Node:
    pass


class _Finalizer:
    def __init__(self, *, alive: bool = True):
        self.alive = alive

    def detach(self):
        self.alive = False


def _bind_named_forward(owner, name: str) -> None:
    def component_forward(self):
        return self

    component_forward.__name__ = name
    owner.forward = types.MethodType(component_forward, owner)


def _component(
    *,
    handle: int,
    handle_attr: str,
    finalizer_attr: str,
    graph_cache_attr: str,
    forward_name: str,
) -> _Node:
    owner = _Node()
    setattr(owner, handle_attr, handle)
    setattr(owner, finalizer_attr, _Finalizer())
    setattr(owner, graph_cache_attr, object())
    owner._rpu_cache = object()
    _bind_named_forward(owner, forward_name)
    return owner


def _complete_policy(*, fused: bool = True, vlm_chunk_size: int = 160):
    vlm = _component(
        handle=11,
        handle_attr="_rpu_vlm_decoder_handle",
        finalizer_attr="_rpu_vlm_decoder_handle_finalizer",
        graph_cache_attr="_rpu_gemma_graph_cache",
        forward_name="rpu_gemma_model_forward",
    )
    vlm._rpu_chunk_size = vlm_chunk_size
    expert = _component(
        handle=13,
        handle_attr="_rpu_action_handle",
        finalizer_attr="_rpu_action_handle_finalizer",
        graph_cache_attr="_rpu_adarms_graph_cache",
        forward_name="rpu_adarms_model_forward",
    )
    vision = _component(
        handle=17,
        handle_attr="_rpu_vision_handle",
        finalizer_attr="_rpu_vision_handle_finalizer",
        graph_cache_attr="_rpu_siglip_graph_cache",
        forward_name="rpu_siglip_forward",
    )

    paligemma_model = _Node()
    paligemma_model.language_model = vlm
    paligemma_model.vision_tower = vision
    pawe = _Node()
    pawe.paligemma = _Node()
    pawe.paligemma.model = paligemma_model
    pawe.gemma_expert = _Node()
    pawe.gemma_expert.model = expert

    model = _Node()
    model.paligemma_with_expert = pawe
    model._rpu_runtime_device = types.SimpleNamespace(type="rpu")
    model._rpu_adarms_cond_cache = {}
    model._rpu_denoise_mask_cache = None
    model._pi05_language_prefix_cache = None
    model._pi05_prefix_assembly_ok = None
    model._pi05_prefix_assembly_enabled = True
    model._rpu_siglip_batch_n = 3
    model._rpu_legacy_prefix_pad16_request = True
    model._rpu_legacy_kvinsert_pad16_request = False
    if fused:
        model._rpu_fused_denoise_handle = 19
        model._rpu_fused_denoise_handle_finalizer = _Finalizer()
        model._rpu_fused_denoise_graph_cache = object()
    else:
        model._rpu_fused_denoise_handle = None
        model._rpu_fused_denoise_handle_finalizer = None
        model._rpu_fused_denoise_graph_cache = None

    policy = _Node()
    policy.model = model
    policy._rpu_swizzle_started = True
    policy._rpu_swizzled = True

    from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

    _PI05_HANDLES[model] = 1
    return policy, model, vlm, expert, vision


@pytest.mark.parametrize("fused", [True, False])
def test_complete_runtime_rehydrates_adapter_and_public_policy(fused):
    policy, model, _, _, _ = _complete_policy(fused=fused)
    try:
        adapter = pi05.Pi05Adapter(policy)
        assert adapter._rpu_is_ready is True
        assert adapter._pi05_handle == 1
        assert adapter._vlm_handle == 11
        assert adapter._expert_handle == 13
        assert adapter._siglip_handle == 17
        assert len(adapter._finalizers) == (1 if fused else 0)
        assert adapter.to_rpu() is policy

        public = Pi05Policy.from_lerobot_policy(policy)
        assert public._rpu_ready is True
        assert public.to("rpu") is public
    finally:
        from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

        _PI05_HANDLES.pop(model, None)


def test_complete_runtime_rehydrates_auto_chunk_policy():
    policy, model, _, _, _ = _complete_policy(vlm_chunk_size=0)
    try:
        adapter = pi05.Pi05Adapter(policy)
        assert adapter._rpu_is_ready is True
        assert adapter._vlm_chunk_size == 0
        assert adapter.to_rpu() is policy

        public = Pi05Policy.from_lerobot_policy(policy)
        assert public._rpu_ready is True
        assert public._vlm_chunk_size == 0
        assert public.to("rpu") is public
    finally:
        from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

        _PI05_HANDLES.pop(model, None)


def test_duplicate_wrapper_preserves_component_and_session_generations():
    policy, model, _, _, _ = _complete_policy()
    try:
        first = pi05.Pi05Adapter(policy)
        generations = {
            "vision_encoder": 1,
            "language_model": 4,
            "action_expert": 2,
        }
        first._execution_session._generation = 4
        first._publish_execution_views(
            first._resolve_execution(
                first._rpu_execution, entry_point="test"
            ),
            generation=4,
            component_generations=generations,
        )

        second = pi05.Pi05Adapter(policy)
        assert second._execution_session is first._execution_session
        assert second._execution_generation == 4
        assert second._component_generations == generations
        assert model._pi05_execution_component_generations == generations
    finally:
        from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

        _PI05_HANDLES.pop(model, None)


def test_text_hot_reconfigure_updates_one_child_and_exact_graph_owners(
    monkeypatch,
):
    from rpu_backend.api._execution import reconfigure_rpu_execution
    from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

    class Cache:
        def __init__(self):
            self.clears = 0

        def begin_warmup(self):
            pass

        def clear(self):
            self.clears += 1

        def cache_invariant_ok(self):
            return True

    policy, model, vlm, expert, vision = _complete_policy()
    model.config = types.SimpleNamespace(
        chunk_size=50,
        image_features=("cam0", "cam1", "cam2"),
    )
    gemma_cache, expert_cache, vision_cache, fused_cache = (
        Cache(), Cache(), Cache(), Cache()
    )
    vlm._rpu_gemma_graph_cache = gemma_cache
    expert._rpu_adarms_graph_cache = expert_cache
    vision._rpu_siglip_graph_cache = vision_cache
    model._rpu_fused_denoise_graph_cache = fused_cache
    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "1")
    monkeypatch.setenv("RPU_PI05_SIGLIP_BATCH", "1")
    for name, value in {
        "execution_reconfigure_begin": lambda _attempt: 7,
        "execution_reconfigure_commit": lambda _token: None,
        "execution_reconfigure_abort": lambda _token: None,
        "execution_reconfigure_abort_attempt": lambda _attempt: False,
    }.items():
        monkeypatch.setattr(pi05.torch.ops.rpu, name, value, raising=False)

    try:
        adapter = pi05.Pi05Adapter(policy)
        result = reconfigure_rpu_execution(
            model,
            {
                "components": {
                    "language_model": {
                        "prefill": {"chunk_size": 192},
                    },
                },
            },
        )

        assert result["components"]["language_model"]["prefill"] == {
            "chunk_size": 192
        }
        assert vlm._rpu_chunk_size == 192
        assert adapter._component_generations == {
            "vision_encoder": 0,
            "language_model": 1,
            "action_expert": 0,
        }
        assert gemma_cache.clears == 1
        assert expert_cache.clears == 1
        assert fused_cache.clears == 1
        assert vision_cache.clears == 0
    finally:
        _PI05_HANDLES.pop(model, None)


def test_text_hot_reconfigure_rolls_back_geometry_and_generation(monkeypatch):
    from rpu_backend.api._execution import (
        reconfigure_rpu_execution,
        rpu_execution_stats,
    )
    from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

    class Cache:
        def __init__(self, *, fail_once=False):
            self.fail_once = fail_once

        def begin_warmup(self):
            pass

        def clear(self):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("clear interrupted")

        def cache_invariant_ok(self):
            return True

    policy, model, vlm, expert, vision = _complete_policy()
    model.config = types.SimpleNamespace(
        chunk_size=50,
        image_features=("cam0", "cam1", "cam2"),
    )
    vlm._rpu_gemma_graph_cache = Cache(fail_once=True)
    expert._rpu_adarms_graph_cache = Cache()
    vision._rpu_siglip_graph_cache = Cache()
    model._rpu_fused_denoise_graph_cache = Cache()
    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "1")
    monkeypatch.setenv("RPU_PI05_SIGLIP_BATCH", "1")
    for name, value in {
        "execution_reconfigure_begin": lambda _attempt: 7,
        "execution_reconfigure_commit": lambda _token: None,
        "execution_reconfigure_abort": lambda _token: None,
        "execution_reconfigure_abort_attempt": lambda _attempt: False,
    }.items():
        monkeypatch.setattr(pi05.torch.ops.rpu, name, value, raising=False)

    try:
        adapter = pi05.Pi05Adapter(policy)
        old_config = adapter._execution_session.config
        with pytest.raises(RuntimeError, match="clear interrupted"):
            reconfigure_rpu_execution(
                model,
                {
                    "components": {
                        "language_model": {
                            "prefill": {"chunk_size": 192},
                        },
                    },
                },
            )

        assert adapter._execution_session.config is old_config
        assert adapter._execution_session.generation == 0
        assert vlm._rpu_chunk_size == 160
        assert adapter._component_generations == {
            "vision_encoder": 0,
            "language_model": 0,
            "action_expert": 0,
        }
        assert rpu_execution_stats(model)["state"] == "QUIESCENT"
    finally:
        _PI05_HANDLES.pop(model, None)


def test_native_begin_failure_does_not_rollback_or_poison(monkeypatch):
    from rpu_backend.api._execution import (
        reconfigure_rpu_execution,
        rpu_execution_stats,
    )
    from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

    policy, model, vlm, _expert, _vision = _complete_policy()
    model.config = types.SimpleNamespace(
        chunk_size=50,
        image_features=("cam0", "cam1", "cam2"),
    )
    attempts = []
    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "1")
    monkeypatch.setenv("RPU_PI05_SIGLIP_BATCH", "1")
    monkeypatch.setattr(
        pi05.torch.ops.rpu,
        "execution_reconfigure_begin",
        lambda _attempt: (_ for _ in ()).throw(RuntimeError("begin failed")),
        raising=False,
    )
    monkeypatch.setattr(
        pi05.torch.ops.rpu,
        "execution_reconfigure_abort_attempt",
        attempts.append,
        raising=False,
    )
    monkeypatch.setattr(
        pi05.torch.ops.rpu,
        "execution_reconfigure_commit",
        lambda _token: None,
        raising=False,
    )
    monkeypatch.setattr(
        pi05.torch.ops.rpu,
        "execution_reconfigure_abort",
        lambda _token: None,
        raising=False,
    )

    try:
        adapter = pi05.Pi05Adapter(policy)
        old_config = adapter._execution_session.config
        with pytest.raises(RuntimeError, match="begin failed"):
            reconfigure_rpu_execution(
                model,
                {
                    "components": {
                        "language_model": {
                            "prefill": {"chunk_size": 192},
                        },
                    },
                },
            )

        assert len(attempts) == 1
        assert adapter._execution_session.config is old_config
        assert adapter._execution_session.generation == 0
        assert vlm._rpu_chunk_size == 160
        assert rpu_execution_stats(model)["state"] == "QUIESCENT"
    finally:
        _PI05_HANDLES.pop(model, None)


def test_close_retires_children_through_execution_session(monkeypatch):
    from rpu_backend.api._execution import rpu_execution_stats
    from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

    policy, model, vlm, expert, vision = _complete_policy()
    destroyed = []
    for op in ("pi05_denoise_step_destroy", "siglip_destroy", "adarms_destroy", "gemma_destroy"):
        monkeypatch.setattr(pi05.torch.ops.rpu, op, destroyed.append, raising=False)
    for owner, name in ((model, "_rpu_fused_denoise_graph_cache"),
                        (vlm, "_rpu_gemma_graph_cache"),
                        (expert, "_rpu_adarms_graph_cache"),
                        (vision, "_rpu_siglip_graph_cache")):
        setattr(owner, name, types.SimpleNamespace(
            clear=lambda: None, cache_invariant_ok=lambda: True))
    adapter = pi05.Pi05Adapter(policy)
    adapter.close()

    assert destroyed == [19, 17, 13, 11]
    assert rpu_execution_stats(model)["state"] == "CLOSED"
    assert adapter._rpu_is_ready is False
    assert model not in _PI05_HANDLES
    assert not hasattr(vlm, "_rpu_vlm_decoder_handle")
    assert not hasattr(expert, "_rpu_action_handle")
    assert not hasattr(vision, "_rpu_vision_handle")
    adapter.close()  # idempotent through ExecutionSession.shutdown
    with pytest.raises(RPUBackendError, match="policy is closed"):
        adapter.to_rpu()


@pytest.mark.parametrize(
    "dead_component", [None, "vlm", "expert", "vision", "fused", "sentinel"]
)
def test_marker_without_complete_runtime_is_rejected_before_claim(
    monkeypatch, dead_component
):
    if dead_component is None:
        policy = _Node()
        policy.model = _Node()
        policy._rpu_swizzle_started = True
        policy._rpu_swizzled = True
        model = None
    else:
        policy, model, vlm, expert, vision = _complete_policy()
        if dead_component == "fused":
            model._rpu_fused_denoise_handle_finalizer.alive = False
        elif dead_component == "sentinel":
            from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

            _PI05_HANDLES.pop(model, None)
        else:
            owner = {
                "vlm": vlm, "expert": expert, "vision": vision
            }[dead_component]
            finalizer_name = {
                "vlm": "_rpu_vlm_decoder_handle_finalizer",
                "expert": "_rpu_action_handle_finalizer",
                "vision": "_rpu_vision_handle_finalizer",
            }[dead_component]
            getattr(owner, finalizer_name).alive = False
    claimed = []
    monkeypatch.setattr(
        pi05, "_claim_live_instance", lambda owner: claimed.append(owner)
    )
    try:
        adapter = pi05.Pi05Adapter(policy)
        assert adapter._rpu_is_ready is False

        with pytest.raises(RPUBackendError, match="ownership is incomplete"):
            adapter.to_rpu()

        assert claimed == []
        assert adapter._pi05_handle is None
        assert adapter._vlm_handle is None
        assert adapter._expert_handle is None
        assert adapter._siglip_handle is None
    finally:
        if model is not None:
            from rpu_backend.adapters.pi05.runtime import _PI05_HANDLES

            _PI05_HANDLES.pop(model, None)


def test_pre_swizzle_base_exception_releases_owner_and_poisons_retry(
    monkeypatch,
):
    class InstallInterrupted(BaseException):
        pass

    policy = _Node()
    policy.model = _Node()
    policy.model.config = types.SimpleNamespace(
        chunk_size=50, image_features=()
    )
    events = []
    monkeypatch.setattr(
        pi05, "_claim_live_instance", lambda owner: events.append("claim")
    )
    monkeypatch.setattr(
        pi05, "_release_live_instance", lambda owner: events.append("release")
    )
    monkeypatch.setattr(pi05, "_preflight_pi05_cold_model", lambda owner, *, attn_tp=8: None)
    monkeypatch.setattr(
        pi05.torch.rpu, "set_caching_allocator", lambda enabled: None
    )
    def interrupt(owner):
        events.append("patch")
        raise InstallInterrupted("synthetic install interruption")

    # The injected interruption happens before any upstream model code runs.
    patches = types.ModuleType("rpu_backend.adapters.pi05.patches")
    patches.install_class_patches = interrupt
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    adapter = pi05.Pi05Adapter(policy)

    with pytest.raises(InstallInterrupted, match="install interruption"):
        adapter.to_rpu()

    assert events == ["claim", "patch", "release"]
    assert policy._rpu_swizzle_started is True
    assert not hasattr(policy, "_rpu_swizzled")
    assert adapter._rpu_is_ready is False
    with pytest.raises(RPUBackendError, match="prior swizzle attempt"):
        adapter.to_rpu()


def test_partial_component_cleanup_bypasses_rejecting_attribute_hooks():
    class RejectingNode:
        def __setattr__(self, name, value):
            if self.__dict__.get("_reject_runtime_attrs", False):
                raise AssertionError("cleanup re-entered __setattr__")
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if self.__dict__.get("_reject_runtime_attrs", False):
                raise AssertionError("cleanup re-entered __delattr__")
            super().__delattr__(name)

    events = []

    class Cache:
        def clear(self):
            events.append("clear")

    class Finalizer:
        alive = True

        def __call__(self):
            self.alive = False
            events.append("finalize")
            return True

    owner = RejectingNode()
    vars(owner).update(
        {
            "_rpu_graph_cache": Cache(),
            "_rpu_handle": 23,
            "_rpu_handle_finalizer": Finalizer(),
            "_reject_runtime_attrs": True,
        }
    )

    pi05._clear_component_graph_cache(owner, "_rpu_graph_cache")
    pi05._cleanup_component_handle(
        owner,
        handle_attr="_rpu_handle",
        finalizer_attr="_rpu_handle_finalizer",
        destroy=lambda handle: pytest.fail(f"unexpected raw destroy: {handle}"),
    )

    assert events == ["clear", "finalize"]
    assert not hasattr(owner, "_rpu_graph_cache")
    assert not hasattr(owner, "_rpu_handle")
    assert not hasattr(owner, "_rpu_handle_finalizer")


def test_ready_marker_is_published_after_postinstall_and_validator():
    source = inspect.getsource(pi05.Pi05Adapter.to_rpu)
    ready_publish = source.index(
        "self._lerobot_policy._rpu_swizzled = True"
    )
    assert source.index("validate_postinstall(self._lerobot_policy)") < ready_publish
    assert source.index(
        "install_hw_attr_validator(self._lerobot_policy)"
    ) < ready_publish
    assert "except BaseException as error:" in source


def test_public_policy_rejects_cpu_after_install_started():
    moved = []
    lerobot_policy = _Node()
    lerobot_policy._rpu_swizzle_started = True
    lerobot_policy.to = lambda device: moved.append(device)
    policy = Pi05Policy.__new__(Pi05Policy)
    policy._lerobot_policy = lerobot_policy
    policy._adapter = types.SimpleNamespace(_rpu_is_ready=False)
    policy._rpu_ready = False

    with pytest.raises(RPUBackendError, match="irreversible"):
        policy.to("cpu")
    assert moved == []
