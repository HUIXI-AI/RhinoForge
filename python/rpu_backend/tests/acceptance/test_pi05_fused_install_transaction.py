"""Board-free lifecycle contracts for the Pi0.5 fused denoise handle."""
from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rpu_backend
from rpu_backend.adapters import pi05
from rpu_backend.adapters.pi05 import w8a16, weights


ROOT = Path(__file__).resolve().parents[4]


class _FakeTensor:
    dtype = torch.float16

    def new_empty(self, *shape):
        return _FakeTensor()


class _InstallInterrupted(BaseException):
    pass


class _Interruptible:
    def arm_assignment_failure(self, name):
        object.__setattr__(self, "_fail_assignment", name)

    def __setattr__(self, name, value):
        if self.__dict__.get("_fail_assignment") == name:
            object.__setattr__(self, "_fail_assignment", None)
            raise _InstallInterrupted(
                f"synthetic publication interruption for {name}")
        object.__setattr__(self, name, value)


class _PolicyModel(_Interruptible):
    pass


class _Adapter(_Interruptible, pi05.Pi05Adapter):
    pass


def _linear():
    return SimpleNamespace(weight=_FakeTensor())


def _adapter():
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            q_proj=_linear(),
            k_proj=_linear(),
            v_proj=_linear(),
            o_proj=_linear(),
        ),
        mlp=SimpleNamespace(
            gate_proj=_linear(), up_proj=_linear(), down_proj=_linear()),
        input_layernorm=SimpleNamespace(
            _rpu_dense_w_rp=_FakeTensor(),
            _rpu_dense_b_rp=_FakeTensor(),
        ),
        post_attention_layernorm=SimpleNamespace(
            _rpu_dense_w_rp=_FakeTensor(),
            _rpu_dense_b_rp=_FakeTensor(),
        ),
    )
    expert = SimpleNamespace(
        config=SimpleNamespace(
            num_key_value_heads=4,
            num_attention_heads=8,
            head_dim=16,
            intermediate_size=256,
            rms_norm_eps=1e-6,
        ),
        layers=[layer],
        norm=SimpleNamespace(
            _rpu_dense_w_rp=_FakeTensor(),
            _rpu_dense_b_rp=_FakeTensor(),
        ),
        _gemma_rope_cos=_FakeTensor(),
        _gemma_rope_sin=_FakeTensor(),
        _rpu_cache=SimpleNamespace(max_seq_len=512),
    )
    model = _PolicyModel()
    model.paligemma_with_expert = SimpleNamespace(
        gemma_expert=SimpleNamespace(model=expert),
        paligemma=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(
            config=SimpleNamespace(num_key_value_heads=1),
            _rpu_cache=SimpleNamespace(max_seq_len=3072)))))
    model.action_in_proj = SimpleNamespace(
        out_features=128,
        _rpu_w_num_cores_1=_FakeTensor(),
        _rpu_bias=_FakeTensor(),
    )
    model.action_out_proj = SimpleNamespace(
        _rpu_w_num_cores_1=_FakeTensor(),
        _rpu_bias=_FakeTensor(),
    )
    model.config = SimpleNamespace(max_action_dim=7, chunk_size=50)

    adapter = _Adapter.__new__(_Adapter)
    adapter._lerobot_policy = SimpleNamespace(model=model)
    adapter._rpu_execution = {}
    adapter._finalizers = []
    return adapter, model


@pytest.fixture
def native_install(monkeypatch):
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime import _native_retirement

    gc.collect()
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    state = {
        "next_handle": 1,
        "alive": set(),
        "events": [],
        "fail_set": set(),
        "fail_chunk": set(),
        "fail_envelope": set(),
        "envelopes": {},
        "fail_destroy": set(),
    }

    def create(linear_acc32=False):
        handle = state["next_handle"]
        state["next_handle"] += 1
        state["alive"].add(handle)
        state.setdefault("accumulation", {})[handle] = linear_acc32
        state["events"].append(("create", handle))
        return handle

    def set_kv_source_heads(handle, heads):
        assert ("set_weights", handle) not in state["events"]
        state.setdefault("kv_source_heads", {})[handle] = heads

    def set_weights(handle, *args):
        state["events"].append(("set_weights", handle))
        if handle in state["fail_set"]:
            raise RuntimeError(f"synthetic set_weights failure for {handle}")

    def set_chunk_size(handle, chunk_size):
        state["events"].append(("set_chunk_size", handle, chunk_size))
        if handle in state["fail_chunk"]:
            raise RuntimeError(
                f"synthetic set_chunk_size failure for {handle}"
            )

    def destroy(handle):
        state["events"].append(("destroy", handle))
        if handle in state["fail_destroy"]:
            raise RuntimeError(f"synthetic destroy failure for {handle}")
        if handle not in state["alive"]:
            raise RuntimeError(f"double destroy for {handle}")
        state["alive"].remove(handle)

    def set_envelope(handle, capacity, chunk):
        assert state["events"][-1] == ("set_weights", handle)
        if handle in state["fail_envelope"]:
            raise RuntimeError("synthetic envelope failure")
        state["envelopes"][handle] = (capacity, chunk)

    class GraphCache:
        def __init__(self, **kwargs):
            self.clear_calls = 0

        def clear(self):
            self.clear_calls += 1

        def cache_invariant_ok(self):
            return True

    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "1")
    monkeypatch.delenv("RPU_PI05_ADARMS_DENSE_W8A16", raising=False)
    def prepare_final_norm(expert, *, num_cores):
        # This fixture exercises the historical default eight-core install.
        assert num_cores == 8

    monkeypatch.setattr(weights, "_prepare_final_norm_weights", prepare_final_norm)
    monkeypatch.setattr(
        weights, "_swizzle_action_proj_weights", lambda *args: None)
    monkeypatch.setattr(
        w8a16,
        "pi05_expert_scale_lists",
        lambda *a, **k: ([], [], [], [], [], [], []),
    )
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", GraphCache)
    monkeypatch.setattr(
        pi05.torch.ops.rpu, "pi05_denoise_step_create", create, raising=False)
    monkeypatch.setattr(
        pi05.torch.ops.rpu,
        "pi05_denoise_step_set_weights",
        set_weights,
        raising=False,
    )
    monkeypatch.setattr(
        pi05.torch.ops.rpu,
        "pi05_denoise_step_set_kv_source_heads",
        set_kv_source_heads,
        raising=False,
    )
    monkeypatch.setattr(
        pi05.torch.ops.rpu,
        "pi05_denoise_step_set_chunk_size",
        set_chunk_size,
        raising=False,
    )
    monkeypatch.setattr(
        pi05.torch.ops.rpu, "pi05_denoise_step_destroy", destroy, raising=False)
    monkeypatch.setattr(pi05.torch.ops.rpu, "pi05_denoise_step_set_chunk_envelope", set_envelope, raising=False)
    return state


def _finalize(model):
    finalizer = getattr(model, "_rpu_fused_denoise_handle_finalizer", None)
    if finalizer is not None and finalizer.alive:
        finalizer()


def test_envelope_uses_prefix_vlm_cache_not_expert_cache(native_install):
    adapter, model = _adapter()
    try:
        adapter._install_fused_denoise_handle()
        assert native_install["envelopes"] == {1: (3072, 0)}
    finally:
        _finalize(model)


@pytest.mark.parametrize("replacement", [False, True])
def test_envelope_failure_preserves_old_denoise_and_retires_pending(native_install, replacement):
    adapter, model = _adapter()
    if replacement:
        adapter._install_fused_denoise_handle()
    before = dict(vars(model))
    finalizers = list(adapter._finalizers)
    pending = 2 if replacement else 1
    native_install["fail_envelope"].add(pending)
    try:
        with pytest.raises(RuntimeError, match="envelope failure"):
            adapter._install_fused_denoise_handle()
        assert vars(model) == before and adapter._finalizers == finalizers
        assert native_install["alive"] == ({1} if replacement else set())
        assert native_install["events"].count(("destroy", pending)) == 1
    finally:
        _finalize(model)


def test_native_set_weights_validates_parallel_lists_and_dense_mode():
    source = (
        ROOT / "src/fused/rpu_pi05_denoise_step_model.cpp"
    ).read_text(encoding="utf-8")
    setter = source.split(
        "void Pi05DenoiseStepModel::set_weights(", 1
    )[1].split("// step_forward", 1)[0]
    checks = setter[:setter.index("layer_weights_.clear();")]

    scale_check = checks.index("check_pi05_scale_lists(")
    for name in (
        "q_w", "k_w", "v_w", "o_w", "gate_w", "up_w", "down_w",
        "attn_dense_w_list", "attn_dense_b_list",
        "mlp_dense_w_list", "mlp_dense_b_list",
    ):
        marker = f"check_list({name},"
        assert marker in checks
        assert checks.index(marker) < scale_check

    assert "dense scales must be all " in checks
    assert "provided or all empty" in checks
    assert "has_attn_dense_scale == has_mlp_dense_scale" in checks
    assert "has_attn_dense_scale == has_final_dense_scale" in checks
    assert "has_dense_scale ? at::kChar : at::kHalf" in checks
    assert "weight.size(0) == 3 * hidden_size" in checks
    assert "weight.size(1) == hidden_size" in checks
    assert checks.count("numel() == 3 * hidden_size") >= 2
    assert checks.count("device().type() == at::kPrivateUse1") >= 3
    assert checks.count("is_contiguous()") >= 3
    assert checks.count("check_dense(") == 3


def test_cache_prepare_failure_does_not_create_or_publish(
    native_install, monkeypatch
):
    adapter, model = _adapter()

    class FailingCache:
        def __init__(self, **kwargs):
            raise RuntimeError("synthetic cache preparation failure")

    monkeypatch.setattr(rpu_backend.graph, "GraphCache", FailingCache)
    with pytest.raises(RuntimeError, match="cache preparation failure"):
        adapter._install_fused_denoise_handle()

    assert native_install["events"] == []
    assert not hasattr(model, "_rpu_fused_denoise_handle")
    assert adapter._finalizers == []


def test_set_weights_failure_cleans_pending_handle(native_install):
    adapter, model = _adapter()
    native_install["fail_set"].add(1)

    with pytest.raises(RuntimeError, match="set_weights failure"):
        adapter._install_fused_denoise_handle()

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_fused_denoise_handle")
    assert adapter._finalizers == []


def test_chunk_size_failure_cleans_pending_handle(native_install):
    adapter, model = _adapter()
    adapter._rpu_execution = {"action": {"chunk_size": 64}}
    native_install["fail_chunk"].add(1)

    with pytest.raises(RuntimeError, match="set_chunk_size failure"):
        adapter._install_fused_denoise_handle()

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("set_chunk_size", 1, 64),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_fused_denoise_handle")
    assert adapter._finalizers == []


def test_failed_replacement_preserves_published_install(native_install):
    adapter, model = _adapter()
    adapter._install_fused_denoise_handle()
    old_finalizer = model._rpu_fused_denoise_handle_finalizer
    old_cache = model._rpu_fused_denoise_graph_cache
    native_install["events"].clear()
    native_install["fail_set"].add(2)

    with pytest.raises(RuntimeError, match="set_weights failure"):
        adapter._install_fused_denoise_handle()

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("destroy", 2),
    ]
    assert native_install["alive"] == {1}
    assert model._rpu_fused_denoise_handle == 1
    assert model._rpu_fused_denoise_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_fused_denoise_graph_cache is old_cache
    assert adapter._finalizers == [old_finalizer]
    _finalize(model)


def test_old_destroy_failure_aborts_replacement(native_install):
    from rpu_backend.api import _execution
    from rpu_backend.runtime import _native_retirement

    adapter, model = _adapter()
    adapter._install_fused_denoise_handle()
    old_finalizer = model._rpu_fused_denoise_handle_finalizer
    native_install["events"].clear()
    native_install["fail_destroy"].add(1)

    with pytest.raises(RuntimeError, match="destroy failure"):
        adapter._install_fused_denoise_handle()

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("set_chunk_size", 2, 0),
        ("destroy", 1),
    ]
    assert native_install["alive"] == {1, 2}
    pending = model._rpu_fused_denoise_retirement_state
    assert model._rpu_fused_denoise_handle == pending.handle == 2
    assert pending.failed is not None and pending.keepalive and pending.graphs
    assert {resource.handle for resource in _native_retirement._FAILED_RETIREMENTS} == {1, 2}
    assert _execution._UNSAFE_PROCESS_REASON
    before = list(native_install["events"])
    native_install["fail_destroy"].clear()
    with pytest.raises(RuntimeError, match="unsafe|restart|Restart|retirement"):
        adapter._install_fused_denoise_handle()
    old_finalizer()
    _finalize(model)
    assert native_install["events"] == before


def test_first_install_publication_baseexception_rolls_back(native_install):
    adapter, model = _adapter()
    old_finalizers = adapter._finalizers
    model.arm_assignment_failure("_rpu_fused_denoise_handle_finalizer")

    with pytest.raises(_InstallInterrupted, match="publication interruption"):
        adapter._install_fused_denoise_handle()

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("set_chunk_size", 1, 0),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_fused_denoise_handle")
    assert not hasattr(model, "_rpu_fused_denoise_graph_cache")
    assert adapter._finalizers is old_finalizers


def test_replacement_adapter_publication_baseexception_preserves_old_install(
    native_install,
):
    adapter, model = _adapter()
    adapter._install_fused_denoise_handle()
    old_finalizers = adapter._finalizers
    old_finalizer = model._rpu_fused_denoise_handle_finalizer
    old_cache = model._rpu_fused_denoise_graph_cache
    native_install["events"].clear()
    adapter.arm_assignment_failure("_finalizers")

    with pytest.raises(_InstallInterrupted, match="publication interruption"):
        adapter._install_fused_denoise_handle()

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("set_chunk_size", 2, 0),
        ("destroy", 2),
    ]
    assert native_install["alive"] == {1}
    assert model._rpu_fused_denoise_handle == 1
    assert model._rpu_fused_denoise_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_fused_denoise_graph_cache is old_cache
    assert adapter._finalizers is old_finalizers
    _finalize(model)


def test_successful_replacement_retires_old_finalizer(native_install):
    adapter, model = _adapter()
    adapter._install_fused_denoise_handle()
    old_finalizer = model._rpu_fused_denoise_handle_finalizer
    old_cache = model._rpu_fused_denoise_graph_cache
    native_install["events"].clear()

    adapter._install_fused_denoise_handle()

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("set_chunk_size", 2, 0),
        ("destroy", 1),
    ]
    assert native_install["alive"] == {2}
    assert model._rpu_fused_denoise_handle == 2
    assert model._rpu_fused_denoise_handle_finalizer.alive
    assert not old_finalizer.alive
    assert model._rpu_fused_denoise_graph_cache is not old_cache
    assert adapter._finalizers == [model._rpu_fused_denoise_handle_finalizer]
    _finalize(model)


def test_replacement_does_not_double_destroy_dead_finalizer(native_install):
    adapter, model = _adapter()
    adapter._install_fused_denoise_handle()
    model._rpu_fused_denoise_handle_finalizer()
    native_install["events"].clear()

    adapter._install_fused_denoise_handle()

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("set_chunk_size", 2, 0),
    ]
    assert native_install["alive"] == {2}
    assert model._rpu_fused_denoise_handle == 2
    assert model._rpu_fused_denoise_handle_finalizer.alive
    assert adapter._finalizers == [model._rpu_fused_denoise_handle_finalizer]
    _finalize(model)


def test_optout_retires_existing_handle_without_replacement(
    native_install, monkeypatch
):
    adapter, model = _adapter()
    adapter._install_fused_denoise_handle()
    old_finalizer = model._rpu_fused_denoise_handle_finalizer
    native_install["events"].clear()
    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "0")

    adapter._install_fused_denoise_handle()

    assert native_install["events"] == [("destroy", 1)]
    assert native_install["alive"] == set()
    assert not old_finalizer.alive
    assert model._rpu_fused_denoise_handle is None
    assert model._rpu_fused_denoise_handle_finalizer is None
    assert model._rpu_fused_denoise_graph_cache is None
    assert adapter._finalizers == []


def test_optout_destroy_failure_retains_resources_without_finalizer_retry(
    native_install, monkeypatch
):
    adapter, model = _adapter()
    adapter._install_fused_denoise_handle()
    old_finalizer = model._rpu_fused_denoise_handle_finalizer
    old_cache = model._rpu_fused_denoise_graph_cache
    native_install["events"].clear()
    native_install["fail_destroy"].add(1)
    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "0")

    with pytest.raises(RuntimeError, match="destroy failure"):
        adapter._install_fused_denoise_handle()

    assert native_install["events"] == [("destroy", 1)]
    assert native_install["alive"] == {1}
    assert model._rpu_fused_denoise_handle == 1
    assert model._rpu_fused_denoise_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_fused_denoise_graph_cache is old_cache
    native_install["fail_destroy"].clear()
    _finalize(model)
    assert native_install["events"] == [("destroy", 1)]
    assert model._rpu_fused_denoise_retirement_state.failed is not None


def test_optout_publication_baseexception_does_not_touch_old_runtime(
    native_install, monkeypatch,
):
    adapter, model = _adapter()
    adapter._install_fused_denoise_handle()
    old_finalizers = adapter._finalizers
    old_finalizer = model._rpu_fused_denoise_handle_finalizer
    old_cache = model._rpu_fused_denoise_graph_cache
    native_install["events"].clear()
    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "0")
    model.arm_assignment_failure("_rpu_fused_denoise_handle_finalizer")

    with pytest.raises(_InstallInterrupted, match="publication interruption"):
        adapter._install_fused_denoise_handle()

    assert native_install["events"] == []
    assert native_install["alive"] == {1}
    assert model._rpu_fused_denoise_handle == 1
    assert model._rpu_fused_denoise_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_fused_denoise_graph_cache is old_cache
    assert old_cache.clear_calls == 0
    assert adapter._finalizers is old_finalizers
    _finalize(model)


def test_optout_does_not_double_destroy_dead_finalizer(
    native_install, monkeypatch
):
    adapter, model = _adapter()
    adapter._install_fused_denoise_handle()
    model._rpu_fused_denoise_handle_finalizer()
    native_install["events"].clear()
    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "0")

    adapter._install_fused_denoise_handle()

    assert native_install["events"] == []
    assert native_install["alive"] == set()
    assert model._rpu_fused_denoise_handle is None
    assert model._rpu_fused_denoise_handle_finalizer is None
    assert model._rpu_fused_denoise_graph_cache is None


@pytest.mark.parametrize("acc32", (False, True))
def test_fused_action_accumulation_is_bound_before_weights(native_install, acc32):
    adapter, model = _adapter()
    adapter._rpu_execution = {"action": {"linear_acc32": acc32}}
    try:
        adapter._install_fused_denoise_handle()
        handle = model._rpu_fused_denoise_handle
        assert native_install["accumulation"] == {handle: acc32}
        assert native_install["events"][:2] == [("create", handle), ("set_weights", handle)]
        # The larger checkpoint KV head count of the cache's two writers
        # (VLM 1, expert 4), bound before the weights.
        assert native_install["kv_source_heads"] == {handle: 4}
    finally:
        _finalize(model)
