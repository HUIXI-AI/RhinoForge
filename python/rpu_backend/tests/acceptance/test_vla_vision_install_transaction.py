"""Board-free install transaction contracts for Qwen3-VL/RhinoVLA vision."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import rpu_backend
from rpu_backend.adapters.qwen3_vl import vision as qwen3vl_vision
from rpu_backend.adapters.rhinovla import vision as rhinovla_vision


_ADAPTERS = (qwen3vl_vision, rhinovla_vision)


class _FakeTensor:
    dtype = torch.float16

    @property
    def data(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return torch.ones(1, dtype=self.dtype)

    def contiguous(self):
        return self

    def detach(self):
        return torch.ones(1, dtype=self.dtype)


class _Movable:
    def to(self, *args, **kwargs):
        return self


class _PublishInterrupted(BaseException):
    pass


class _Model:
    def arm_assignment_failure(self, name):
        object.__setattr__(self, "_fail_assignment", name)

    def __setattr__(self, name, value):
        if self.__dict__.get("_fail_assignment") == name:
            object.__setattr__(self, "_fail_assignment", None)
            raise _PublishInterrupted(
                f"synthetic publication interruption for {name}")
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs):
        return "original"


def _model():
    tensor = _FakeTensor()
    block = SimpleNamespace(
        _rpu_q_w=tensor,
        _rpu_k_w=tensor,
        _rpu_v_w=tensor,
        _rpu_q_b=tensor,
        _rpu_k_b=tensor,
        _rpu_v_b=tensor,
        _rpu_o_b=tensor,
        _rpu_fc1_b=tensor,
        _rpu_fc2_b=tensor,
        attn=SimpleNamespace(qkv=SimpleNamespace(weight=torch.ones(1)),
                             proj=SimpleNamespace(weight=tensor)),
        mlp=SimpleNamespace(
            linear_fc1=SimpleNamespace(weight=tensor),
            linear_fc2=SimpleNamespace(weight=tensor),
        ),
        norm1=SimpleNamespace(weight=tensor, bias=tensor),
        norm2=SimpleNamespace(weight=tensor, bias=tensor),
    )
    model = _Model()
    model.blocks = [block]
    model.config = SimpleNamespace(
        num_heads=8,
        hidden_size=128,
        intermediate_size=256,
        spatial_merge_size=2,
        deepstack_visual_indexes=[],
        hidden_act="gelu_pytorch_tanh",
    )
    model.patch_embed = SimpleNamespace(
        proj=SimpleNamespace(weight=tensor, bias=None))
    model.pos_embed = _Movable()
    model.merger = _Movable()
    model.deepstack_merger_list = []
    return model


@pytest.fixture(params=_ADAPTERS, ids=("qwen3_vl", "rhinovla"))
def native_install(request, monkeypatch):
    import gc
    gc.collect()
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime import _native_retirement
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    adapter = request.param
    state = {
        "adapter": adapter,
        "next_handle": 1,
        "alive": set(),
        "events": [],
        "fail_set": set(),
        "fail_chunk": set(),
        "fail_destroy": set(),
        "set_error": None,
        "envelopes": [],
        "fail_envelope": False,
    }

    def create():
        handle = state["next_handle"]
        state["next_handle"] += 1
        state["alive"].add(handle)
        state["events"].append(("create", handle))
        return handle

    def set_weights(handle, *args):
        state["events"].append(("set_weights", handle))
        if state["set_error"] is not None:
            raise state["set_error"]
        if handle in state["fail_set"]:
            raise RuntimeError(f"synthetic set_weights failure for {handle}")

    def set_rope(handle, *args):
        state["events"].append(("set_rope", handle))

    def set_envelope(handle, capacity, chunk):
        assert state["events"][-1] == ("set_weights", handle)
        state["envelopes"].append((handle, capacity, chunk))
        if state["fail_envelope"]:
            raise RuntimeError("synthetic vision envelope failure")

    def set_chunk_size(handle, chunk_size):
        state["events"].append(("set_chunk_size", handle, chunk_size))
        if handle in state["fail_chunk"]:
            raise RuntimeError(
                f"synthetic set_chunk_size failure for {handle}"
            )

    def position_idx_keepalive(handle):
        state["events"].append(("keepalive", handle))
        return _FakeTensor()

    def destroy(handle):
        state["events"].append(("destroy", handle))
        if handle in state["fail_destroy"]:
            raise RuntimeError(f"synthetic destroy failure for {handle}")
        if handle not in state["alive"]:
            raise RuntimeError(f"double destroy for {handle}")
        state["alive"].remove(handle)

    class GraphCache:
        def clear(self):
            pass

        def cache_invariant_ok(self):
            return True

    monkeypatch.setattr(
        adapter, "_convert_vision_block_weights_for_rpu", lambda *a, **k: None)
    monkeypatch.setattr(
        adapter, "build_vision_rope_tables",
        lambda *a, **k: (_FakeTensor(), _FakeTensor()))
    monkeypatch.setattr(
        adapter, "_fold_conv3d_to_linear_weight", lambda weight: _FakeTensor())
    monkeypatch.setattr(
        adapter, "_make_dummy_vision_kv_caches",
        lambda layers, capacity, *args: SimpleNamespace(max_seq_len=capacity))
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", GraphCache)

    from rpu_backend.graph import lazy_init_guard

    monkeypatch.setattr(lazy_init_guard, "_verify_lazy_init", lambda model: None)
    monkeypatch.setattr(
        adapter.torch.ops.rpu, "qwen3vl_vision_create", create, raising=False)
    monkeypatch.setattr(
        adapter.torch.ops.rpu, "qwen3vl_vision_set_weights", set_weights,
        raising=False)
    monkeypatch.setattr(adapter.torch.ops.rpu, "qwen3vl_vision_set_weights_w8a16", set_weights, raising=False)
    monkeypatch.setattr(adapter.torch.ops.rpu, "qwen3vl_vision_set_chunk_envelope", set_envelope, raising=False)
    monkeypatch.setattr(
        adapter.torch.ops.rpu, "qwen3vl_vision_set_rope", set_rope,
        raising=False)
    monkeypatch.setattr(
        adapter.torch.ops.rpu, "qwen3vl_vision_set_rope_route", set_rope,
        raising=False)
    monkeypatch.setattr(
        adapter.torch.ops.rpu, "qwen3vl_vision_set_chunk_size",
        set_chunk_size, raising=False)
    monkeypatch.setattr(
        adapter.torch.ops.rpu, "qwen3vl_vision_position_idx_keepalive",
        position_idx_keepalive, raising=False)
    monkeypatch.setattr(
        adapter.torch.ops.rpu, "qwen3vl_vision_destroy", destroy,
        raising=False)
    return state


def _install(state, model, **kwargs):
    return state["adapter"].install_qwen3_vl_vision_for_rpu(model, **kwargs)


def _post_weight_events(state, handle):
    events = [("set_rope", handle), ("set_chunk_size", handle, 0)]
    events.append(("keepalive", handle))
    return events


def _finalize(model):
    finalizer = getattr(model, "_rpu_vision_handle_finalizer", None)
    if finalizer is not None and finalizer.alive:
        finalizer()


@pytest.mark.parametrize("capacity", [640, 8192])
def test_cold_envelope_uses_actual_vision_cache(native_install, capacity):
    model = _model()
    try:
        _install(native_install, model, max_seq_len=capacity)
        assert native_install["envelopes"] == [(1, model._rpu_vision_kv_cache.max_seq_len, 0)]
        assert model._rpu_vision_kv_cache.max_seq_len == capacity
    finally:
        _finalize(model)


def test_cold_envelope_failure_preserves_old_vision_install(native_install):
    model = _model()
    _install(native_install, model)
    before = dict(vars(model))
    native_install["fail_envelope"] = True
    try:
        with pytest.raises(RuntimeError, match="vision envelope failure"):
            _install(native_install, model, max_seq_len=640)
        assert vars(model) == before
        assert native_install["alive"] == {1}
        assert native_install["events"].count(("destroy", 2)) == 1
    finally:
        _finalize(model)


@pytest.mark.parametrize("native_install", [qwen3vl_vision], indirect=True)
def test_w8_cold_envelope_shares_the_fp16_install_binding(native_install):
    model = _model()
    block = model.blocks[0]
    block.attn.qkv = SimpleNamespace(weight=_FakeTensor())
    for name in ("q", "k", "v", "o", "fc1", "fc2"):
        setattr(model.blocks[0], "_rpu_" + name + "_ws", _FakeTensor())
    try:
        _install(native_install, model, max_seq_len=768, w8a16=True)
        assert native_install["envelopes"] == [(1, 768, 0)]
    finally:
        _finalize(model)


def test_prepare_failure_does_not_create_or_publish_handle(
    native_install, monkeypatch
):
    model = _model()
    original_forward = model.forward
    adapter = native_install["adapter"]

    def fail_cache(*args, **kwargs):
        raise RuntimeError("synthetic cache preparation failure")

    monkeypatch.setattr(adapter, "_make_dummy_vision_kv_caches", fail_cache)
    with pytest.raises(RuntimeError, match="cache preparation failure"):
        _install(native_install, model)

    assert native_install["events"] == []
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_vision_handle")
    assert "forward" not in vars(model)
    assert model.forward.__func__ is original_forward.__func__


def test_install_flags_reject_truthy_strings_before_prepare(native_install):
    model = _model()
    adapter = native_install["adapter"]
    kwargs = (
        {"w8a16": "false"}
        if adapter is qwen3vl_vision
        else {"rpu_patch_embed": "false"}
    )

    with pytest.raises(TypeError, match="must be a bool"):
        adapter.install_qwen3_vl_vision_for_rpu(model, **kwargs)

    assert native_install["events"] == []
    assert native_install["alive"] == set()


def test_install_rejects_unsupported_gelu_before_prepare(
    native_install, monkeypatch
):
    model = _model()
    model.config.hidden_act = "gelu"
    prepare_calls = []
    monkeypatch.setattr(
        native_install["adapter"],
        "_convert_vision_block_weights_for_rpu",
        lambda *args, **kwargs: prepare_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="hidden_act must be 'gelu_pytorch_tanh'"):
        _install(native_install, model)

    assert prepare_calls == []
    assert native_install["events"] == []
    assert native_install["alive"] == set()


def test_set_weights_failure_cleans_pending_handle_immediately(native_install):
    model = _model()
    original_forward = model.forward
    native_install["fail_set"].add(1)

    with pytest.raises(RuntimeError, match="set_weights failure"):
        _install(native_install, model)

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_vision_handle")
    assert "forward" not in vars(model)
    assert model.forward.__func__ is original_forward.__func__


def test_chunk_size_setter_matches_adapter_contract(native_install):
    model = _model()
    native_install["fail_chunk"].add(1)
    with pytest.raises(RuntimeError, match="set_chunk_size failure"):
        _install(native_install, model, execution_chunk_size=64)

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("set_rope", 1),
        ("set_chunk_size", 1, 64),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_vision_handle")


def test_base_exception_cleans_pending_handle_immediately(native_install):
    class InstallInterrupted(BaseException):
        pass

    model = _model()
    native_install["set_error"] = InstallInterrupted(
        "synthetic install interruption"
    )

    with pytest.raises(InstallInterrupted, match="install interruption"):
        _install(native_install, model)

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_vision_handle")


def test_first_install_publication_baseexception_rolls_back(native_install):
    model = _model()
    original_forward = model.forward
    model.arm_assignment_failure("_rpu_vision_handle_finalizer")

    with pytest.raises(_PublishInterrupted, match="publication interruption"):
        _install(native_install, model)

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        *_post_weight_events(native_install, 1),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_vision_handle")
    assert "forward" not in vars(model)
    assert model.forward.__func__ is original_forward.__func__


def test_failed_replacement_preserves_published_install(native_install):
    model = _model()
    assert _install(native_install, model) == 1
    old_forward = model.forward
    old_cache = model._rpu_vision_graph_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    model._rpu_vision_fused_merger = True
    native_install["events"].clear()
    native_install["fail_set"].add(2)

    with pytest.raises(RuntimeError, match="set_weights failure"):
        _install(native_install, model)

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("destroy", 2),
    ]
    assert native_install["alive"] == {1}
    assert model._rpu_vision_handle == 1
    assert model._rpu_vision_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_vision_graph_cache is old_cache
    assert model._rpu_vision_fused_merger is True
    assert model.forward is old_forward
    _finalize(model)


def test_replacement_publication_baseexception_preserves_old_install(
    native_install,
):
    model = _model()
    assert _install(native_install, model) == 1
    old_forward = model.forward
    old_cache = model._rpu_vision_graph_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install["events"].clear()
    model.arm_assignment_failure("forward")

    with pytest.raises(_PublishInterrupted, match="publication interruption"):
        _install(native_install, model)

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        *_post_weight_events(native_install, 2),
        ("destroy", 2),
    ]
    assert native_install["alive"] == {1}
    assert model._rpu_vision_handle == 1
    assert model._rpu_vision_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_vision_graph_cache is old_cache
    assert model.forward is old_forward
    _finalize(model)


def test_old_destroy_failure_aborts_replacement(native_install):
    model = _model()
    assert _install(native_install, model) == 1
    old_forward = model.forward
    old_cache = model._rpu_vision_graph_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install["events"].clear()
    native_install["fail_destroy"].add(1)

    with pytest.raises(RuntimeError, match="destroy failure"):
        _install(native_install, model)

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        *_post_weight_events(native_install, 2),
        ("destroy", 1),
    ]
    assert native_install["alive"] == {1, 2}
    assert model._rpu_vision_handle == 1
    assert model._rpu_vision_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_vision_graph_cache is old_cache
    assert model.forward is old_forward
    assert model._rpu_vision_retirement_state.failed is not None


def test_successful_replacement_retires_old_install(native_install):
    model = _model()
    assert _install(native_install, model) == 1
    old_cache = model._rpu_vision_graph_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    model._rpu_vision_fused_merger = True
    native_install["events"].clear()

    assert _install(native_install, model) == 2

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        *_post_weight_events(native_install, 2),
        ("destroy", 1),
    ]
    assert native_install["alive"] == {2}
    assert model._rpu_vision_handle == 2
    assert model._rpu_vision_handle_finalizer.alive
    assert not old_finalizer.alive
    assert model._rpu_vision_graph_cache is not old_cache
    assert not hasattr(model, "_rpu_vision_fused_merger")
    _finalize(model)


def test_replacement_does_not_double_destroy_dead_finalizer(native_install):
    model = _model()
    assert _install(native_install, model) == 1
    model._rpu_vision_handle_finalizer()
    native_install["events"].clear()

    assert _install(native_install, model) == 2

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        *_post_weight_events(native_install, 2),
    ]
    assert native_install["alive"] == {2}
    assert model._rpu_vision_handle == 2
    assert model._rpu_vision_handle_finalizer.alive
    _finalize(model)


def _conversion_block(hidden_size=8, intermediate_size=16):
    return SimpleNamespace(
        attn=SimpleNamespace(
            qkv=torch.nn.Linear(hidden_size, 3 * hidden_size, bias=True),
            proj=torch.nn.Linear(hidden_size, hidden_size, bias=True),
        ),
        mlp=SimpleNamespace(
            linear_fc1=torch.nn.Linear(
                hidden_size, intermediate_size, bias=True
            ),
            linear_fc2=torch.nn.Linear(
                intermediate_size, hidden_size, bias=True
            ),
        ),
    )


def _convert_block(adapter, block):
    kwargs = {"w8a16": False} if adapter is qwen3vl_vision else {}
    adapter._convert_vision_block_weights_for_rpu(
        block, num_heads=8, hidden_size=8, **kwargs
    )


@pytest.mark.parametrize("adapter", _ADAPTERS, ids=("qwen3_vl", "rhinovla"))
def test_block_conversion_is_idempotent_in_the_same_mode(adapter, monkeypatch):
    block = _conversion_block()
    calls = []

    def col(weight, *args, **kwargs):
        calls.append("col")
        return weight.clone()

    def row(weight, *args, **kwargs):
        calls.append("row")
        return weight.clone()

    monkeypatch.setattr(adapter, "tp_col_swizzle_mc_weight", col)
    monkeypatch.setattr(adapter, "tp_row_swizzle_mc_weight", row)

    _convert_block(adapter, block)
    first_calls = list(calls)
    _convert_block(adapter, block)

    assert calls == first_calls
    expected_mode = False if adapter is qwen3vl_vision else True
    assert block._rpu_qwen3vl_vision_weights_converted is expected_mode


@pytest.mark.parametrize("adapter", _ADAPTERS, ids=("qwen3_vl", "rhinovla"))
def test_partial_block_conversion_poison_prevents_double_swizzle(
    adapter, monkeypatch
):
    block = _conversion_block()
    row_calls = 0

    monkeypatch.setattr(
        adapter,
        "tp_col_swizzle_mc_weight",
        lambda weight, *args, **kwargs: weight.clone(),
    )

    def fail_second_row(weight, *args, **kwargs):
        nonlocal row_calls
        row_calls += 1
        if row_calls == 2:
            raise RuntimeError("synthetic partial row-swizzle failure")
        return weight.clone()

    monkeypatch.setattr(adapter, "tp_row_swizzle_mc_weight", fail_second_row)
    with pytest.raises(RuntimeError, match="partial row-swizzle failure"):
        _convert_block(adapter, block)

    marker = block._rpu_qwen3vl_vision_weights_converted
    assert marker == adapter._VISION_CONVERSION_IN_PROGRESS
    with pytest.raises(RuntimeError, match="partial prior weight conversion"):
        _convert_block(adapter, block)
    assert row_calls == 2
