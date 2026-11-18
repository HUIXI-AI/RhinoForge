"""Board-free lifecycle contracts for the DINOv3 vision installer."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import rpu_backend
from rpu_backend.adapters import dinov3
from rpu_backend.api.errors import RPUBackendError


class _FakeTensor:
    def __getitem__(self, key):
        return self

    def contiguous(self):
        return self


class _InstallInterrupted(BaseException):
    pass


def test_forward_bounds_accept_profile_and_reject_larger_image():
    dinov3._validate_dinov3_input_bounds(
        14, 14, 201, max_hw=14, max_seq_len=256
    )
    with pytest.raises(RPUBackendError, match="max_hw=14"):
        dinov3._validate_dinov3_input_bounds(
            16, 16, 261, max_hw=14, max_seq_len=256
        )
    with pytest.raises(RPUBackendError, match="max_seq_len=256"):
        dinov3._validate_dinov3_input_bounds(
            14, 14, 257, max_hw=14, max_seq_len=256
        )


class _Model:
    def __setattr__(self, name, value):
        if self.__dict__.get("_reject_runtime_cleanup", False):
            raise AssertionError("rollback re-entered __setattr__")
        if name == self.__dict__.get("_fail_publish_attr"):
            self.__dict__["_fail_publish_attr"] = None
            self.__dict__["_reject_runtime_cleanup"] = True
            raise _InstallInterrupted(
                f"synthetic publication interruption for {name}"
            )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if self.__dict__.get("_reject_runtime_cleanup", False):
            raise AssertionError("rollback re-entered __delattr__")
        super().__delattr__(name)

    def to(self, *args, **kwargs):
        return self

    def forward(self, *args, **kwargs):
        return "original"


def _projection(*, bias=True):
    return SimpleNamespace(
        weight=_FakeTensor(),
        bias=_FakeTensor() if bias else None,
    )


def _model():
    layer = SimpleNamespace(
        attention=SimpleNamespace(
            q_proj=_projection(),
            k_proj=_projection(bias=False),
            v_proj=_projection(),
            o_proj=_projection(),
        ),
        mlp=SimpleNamespace(
            up_proj=_projection(),
            down_proj=_projection(),
        ),
        norm1=SimpleNamespace(weight=_FakeTensor(), bias=_FakeTensor()),
        norm2=SimpleNamespace(weight=_FakeTensor(), bias=_FakeTensor()),
    )
    model = _Model()
    model.config = SimpleNamespace(
        model_type="dinov3_vit",
        use_gated_mlp=False,
        hidden_size=768,
        intermediate_size=3072,
        num_attention_heads=12,
        num_hidden_layers=12,
        num_register_tokens=4,
        layer_norm_eps=1e-6,
        rope_theta=100.0,
        image_size=224,
        patch_size=16,
    )
    model.layer = [layer]
    model._rpu_dinov3_gamma_attn = _FakeTensor()
    model._rpu_dinov3_gamma_mlp = _FakeTensor()
    return model


@pytest.fixture
def native_install(monkeypatch):
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime import _native_retirement

    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    # Lifecycle doubles have no physical CPU alias; shared-position acquisition
    # is covered separately with real ATen storage and explicit flush modes.
    monkeypatch.setattr(torch.rpu, "get_ddr_flush", lambda: True, raising=False)
    monkeypatch.setattr(torch.rpu, "get_ddr_flush_force", lambda: False, raising=False)
    state = {
        "next_handle": 1,
        "alive": set(),
        "events": [],
        "fail_set": set(),
        "fail_destroy": set(),
        "graph_caches": [],
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
        if handle in state["fail_set"]:
            raise RuntimeError(f"synthetic set_weights failure for {handle}")

    def set_rope(handle, *args):
        state["events"].append(("set_rope", handle))

    def set_envelope(handle, capacity, chunk):
        assert state["events"][-1] == ("set_weights", handle)
        state["envelopes"].append((handle, capacity, chunk))
        if state["fail_envelope"]:
            raise RuntimeError("synthetic DINO envelope failure")

    def set_chunk_size(handle, chunk_size):
        state["events"].append(("set_chunk_size", handle, chunk_size))

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
        def __init__(self):
            self.cleared = False
            state["graph_caches"].append(self)

        def clear(self):
            self.cleared = True

        def cache_invariant_ok(self):
            return self.cleared

    monkeypatch.setattr(
        dinov3, "build_dinov3_rope_tables",
        lambda *a, **k: (_FakeTensor(), _FakeTensor()))
    monkeypatch.setattr(
        dinov3, "_make_dummy_dinov3_kv_caches",
        lambda layers, capacity, *args: SimpleNamespace(max_seq_len=capacity))
    monkeypatch.setattr(rpu_backend.graph, "GraphCache", GraphCache)
    monkeypatch.setattr(
        dinov3.torch.ops.rpu, "dinov3_vision_create", create, raising=False)
    monkeypatch.setattr(
        dinov3.torch.ops.rpu, "dinov3_vision_set_weights", set_weights,
        raising=False)
    monkeypatch.setattr(dinov3.torch.ops.rpu, "dinov3_vision_set_chunk_envelope", set_envelope, raising=False)
    monkeypatch.setattr(
        dinov3.torch.ops.rpu, "dinov3_vision_set_rope", set_rope,
        raising=False)
    monkeypatch.setattr(
        dinov3.torch.ops.rpu, "dinov3_vision_set_chunk_size", set_chunk_size,
        raising=False)
    monkeypatch.setattr(
        dinov3.torch.ops.rpu, "dinov3_vision_position_idx_keepalive",
        position_idx_keepalive, raising=False)
    monkeypatch.setattr(
        dinov3.torch.ops.rpu, "dinov3_vision_destroy", destroy,
        raising=False)
    return state


def _install(model):
    return dinov3.install_dinov3_vision_for_rpu(model)


def _finalize(model):
    finalizer = getattr(model, "_rpu_dinov3_handle_finalizer", None)
    if finalizer is not None and finalizer.alive:
        finalizer()


@pytest.mark.parametrize("capacity", [512, 8192])
def test_cold_envelope_uses_actual_dino_cache(native_install, capacity):
    model = _model()
    try:
        dinov3.install_dinov3_vision_for_rpu(model, max_seq_len=capacity)
        assert native_install["envelopes"] == [(1, model._rpu_dinov3_kv_cache.max_seq_len, 0)]
        assert model._rpu_dinov3_kv_cache.max_seq_len == capacity
    finally:
        _finalize(model)


def test_cold_envelope_failure_preserves_old_dino_install(native_install):
    model = _model()
    _install(model)
    before = dict(vars(model))
    native_install["fail_envelope"] = True
    try:
        with pytest.raises(RuntimeError, match="DINO envelope failure"):
            dinov3.install_dinov3_vision_for_rpu(model, max_seq_len=512)
        assert vars(model) == before
        assert native_install["alive"] == {1}
        assert native_install["events"].count(("destroy", 2)) == 1
    finally:
        _finalize(model)


def test_execution_config_accepts_auto_or_exact_single_chunk():
    auto_model = _model()
    auto = dinov3.DINOv3Adapter(
        auto_model,
        rpu_execution={"vision": {"chunk_size": "auto"}},
    )
    assert auto._rpu_execution == {"vision": {"chunk_size": "auto"}}
    assert auto_model._rpu_execution is auto._rpu_execution

    exact_model = _model()
    exact = dinov3.DINOv3Adapter(
        exact_model,
        rpu_execution={"vision": {"chunk_size": 208}},
    )
    assert exact._rpu_execution == {"vision": {"chunk_size": 208}}
    assert exact_model._rpu_execution is exact._rpu_execution


@pytest.mark.parametrize("chunk_size", [192, 224])
def test_execution_config_rejects_non_exact_single_chunk(chunk_size):
    with pytest.raises(ValueError, match="single-chunk capacity 208"):
        dinov3.DINOv3Adapter(
            _model(),
            rpu_execution={"vision": {"chunk_size": chunk_size}},
        )


@pytest.mark.parametrize("field,value", [
    ("padding_rows", "auto"),
    ("padding_budget", 16),
])
def test_execution_config_rejects_padding(field, value):
    with pytest.raises(ValueError, match="not supported"):
        dinov3.DINOv3Adapter(
            _model(),
            rpu_execution={"vision": {field: value}},
        )


def test_explicit_single_chunk_is_bound_to_pending_handle(native_install):
    model = _model()
    model._rpu_execution = {"vision": {"chunk_size": 208}}

    assert _install(model) == 1
    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("set_rope", 1),
        ("set_chunk_size", 1, 208),
        ("keepalive", 1),
    ]
    _finalize(model)


def test_explicit_chunk_rejects_a_different_runtime_grid_before_dispatch():
    runtime = SimpleNamespace(
        _rpu_dinov3_handle=1,
        _rpu_dinov3_num_special_tokens=5,
        _rpu_dinov3_patch_size=16,
        _rpu_dinov3_max_hw=14,
        _rpu_dinov3_max_seq_len=208,
        _rpu_execution={"vision": {"chunk_size": 208}},
    )

    with pytest.raises(RPUBackendError, match="capacity 160.*got 208"):
        dinov3._rpu_dinov3_forward(
            runtime,
            torch.empty(1, 3, 192, 192),
        )


@pytest.mark.parametrize("failure", [None, "final_norm", "output_transfer"])
def test_forward_publishes_receipt_only_after_final_outputs(monkeypatch, failure):
    from contextlib import nullcontext

    previous = {"previous": "successful forward"}
    model = SimpleNamespace(
        _rpu_dinov3_handle=17, _rpu_dinov3_num_special_tokens=1,
        _rpu_dinov3_patch_size=1, _rpu_dinov3_max_hw=1,
        _rpu_dinov3_max_seq_len=16, _rpu_dinov3_num_layers=1,
        _rpu_dinov3_hidden_size=16, _rpu_execution={},
        _rpu_dinov3_patch_embed_w=torch.ones(16, 3, dtype=torch.float16),
        _rpu_dinov3_patch_embed_b=None, _rpu_dinov3_cls_reg=torch.ones(1, 16, dtype=torch.float16),
        _rpu_dinov3_position_idx_keepalive=torch.zeros(1, 2, dtype=torch.int16),
        _rpu_dinov3_kv_cache=SimpleNamespace(reset_to_position=lambda _p: None,
                                            k_caches=[None], v_caches=[None]),
        _rpu_dinov3_graph_cache=SimpleNamespace(capture=lambda _sig: nullcontext()),
        config=SimpleNamespace(hidden_size=16, layer_norm_eps=1.e-6),
        norm=torch.nn.LayerNorm(16, dtype=torch.float16), _rpu_last_execution_plan=previous,
    )
    model._rpu_dinov3_retirement_state = SimpleNamespace(
        handle=17, failed=None,
        keepalive=((), model._rpu_dinov3_position_idx_keepalive, None),
    )
    plan = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(
        compute_chunk=16, physical_descriptor=(1,))), graph_key_words=lambda: (1,),
        graph_mode="RETAINED_CACHE", selection_scope="fixture", physical_plan_digest="physical",
        plan_digest="plan")
    monkeypatch.setattr(dinov3, "_plan_dinov3_vision_execution", lambda *a, **kw: plan)
    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", lambda **kw: SimpleNamespace(**kw))
    calls = []
    def forward(_handle, sequence, *args):
        calls.append(sequence)
        return sequence.clone()
    monkeypatch.setattr(torch.ops.rpu, "dinov3_vision_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "dinov3_vision_get_resolved_chunk_size", lambda _h: 16, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    original_to = torch.Tensor.to
    transfers = []
    def cpu_transfer(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            transfers.append(tensor)
            if failure == "output_transfer" and calls:
                raise RuntimeError("injected output_transfer")
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    if failure == "final_norm":
        def fail_norm(*args, **kwargs):
            raise RuntimeError("injected final_norm")
        monkeypatch.setattr(dinov3.F, "layer_norm", fail_norm)
    if failure:
        with pytest.raises(RuntimeError, match="injected " + failure):
            dinov3._rpu_dinov3_forward(model, torch.ones(1, 3, 1, 1))
        assert model._rpu_last_execution_plan is previous
    else:
        output = dinov3._rpu_dinov3_forward(model, torch.ones(1, 3, 1, 1))
        assert output.last_hidden_state.shape == (1, 2, 16)
        assert output.pooler_output.shape == (1, 16)
        assert model._rpu_last_execution_plan["vision"]["dry_forward_agreement"] is True
    assert len(calls) == 1  # Exercise the actual forward's post-native failure boundary.


def test_native_forward_preserves_cold_chunk_override():
    source = (
        Path(__file__).resolve().parents[4]
        / "src/fused/rpu_dinov3_vision_model.cpp"
    ).read_text(encoding="utf-8")
    assert 'model->set_control_chunk_size_override(chunk_size, "rpu_dinov3_vision_set_chunk_size");' in source
    assert "model->get_last_resolved_chunk_size() == 0" in source
    assert "rpu_dinov3_vision_resolve_chunk_size" in source
    assert "set_chunk_size_override(0);" not in source

    adapter = (
        Path(__file__).resolve().parents[2] / "adapters/dinov3.py"
    ).read_text(encoding="utf-8")
    assert "DINOv3 vision dry/forward chunk plan drift" in adapter


def test_prepare_failure_does_not_create_or_publish_handle(
    native_install, monkeypatch
):
    model = _model()

    def fail_cache(*args, **kwargs):
        raise RuntimeError("synthetic cache preparation failure")

    monkeypatch.setattr(dinov3, "_make_dummy_dinov3_kv_caches", fail_cache)
    with pytest.raises(RuntimeError, match="cache preparation failure"):
        _install(model)

    assert native_install["events"] == []
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_dinov3_handle")


def test_set_weights_failure_cleans_pending_handle(native_install):
    model = _model()
    native_install["fail_set"].add(1)

    with pytest.raises(RuntimeError, match="set_weights failure"):
        _install(model)

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_dinov3_handle")


def test_first_install_publication_base_exception_rolls_back(native_install):
    model = _model()
    model._fail_publish_attr = "_rpu_dinov3_handle_finalizer"

    with pytest.raises(_InstallInterrupted, match="publication interruption"):
        _install(model)

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("set_rope", 1),
        ("keepalive", 1),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    for attr in dinov3._DINOV3_HANDLE_INSTALL_ATTRS:
        assert not hasattr(model, attr)


def test_failed_replacement_preserves_published_install(native_install):
    model = _model()
    assert _install(model) == 1
    old_cache = model._rpu_dinov3_graph_cache
    old_finalizer = model._rpu_dinov3_handle_finalizer
    native_install["events"].clear()
    native_install["fail_set"].add(2)

    with pytest.raises(RuntimeError, match="set_weights failure"):
        _install(model)

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("destroy", 2),
    ]
    assert native_install["alive"] == {1}
    assert model._rpu_dinov3_handle == 1
    assert model._rpu_dinov3_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_dinov3_graph_cache is old_cache
    _finalize(model)


def test_replacement_publication_base_exception_preserves_old_install(
    native_install,
):
    model = _model()
    assert _install(model) == 1
    old_cache = model._rpu_dinov3_graph_cache
    old_finalizer = model._rpu_dinov3_handle_finalizer
    native_install["events"].clear()
    model._fail_publish_attr = "_rpu_dinov3_handle_finalizer"

    with pytest.raises(_InstallInterrupted, match="publication interruption"):
        _install(model)

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("set_rope", 2),
        ("keepalive", 2),
        ("destroy", 2),
    ]
    assert native_install["alive"] == {1}
    assert model._rpu_dinov3_handle == 1
    assert model._rpu_dinov3_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_dinov3_graph_cache is old_cache
    _finalize(model)


def test_old_destroy_failure_aborts_replacement(native_install):
    model = _model()
    assert _install(model) == 1
    old_cache = model._rpu_dinov3_graph_cache
    old_finalizer = model._rpu_dinov3_handle_finalizer
    native_install["events"].clear()
    native_install["fail_destroy"].add(1)

    with pytest.raises(RuntimeError, match="destroy failure"):
        _install(model)

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("set_rope", 2),
        ("keepalive", 2),
        ("destroy", 1),
    ]
    # Once the old destroy is uncertain, no sibling/native cleanup is safe.
    assert native_install["alive"] == {1, 2}
    assert model._rpu_dinov3_handle == 1
    assert model._rpu_dinov3_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_dinov3_graph_cache is old_cache
    before = list(native_install["events"])
    _finalize(model)
    assert native_install["events"] == before


def test_successful_replacement_retires_old_install(native_install):
    model = _model()
    assert _install(model) == 1
    old_cache = model._rpu_dinov3_graph_cache
    old_finalizer = model._rpu_dinov3_handle_finalizer
    native_install["events"].clear()

    assert _install(model) == 2

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("set_rope", 2),
        ("keepalive", 2),
        ("destroy", 1),
    ]
    assert native_install["alive"] == {2}
    assert model._rpu_dinov3_handle == 2
    assert model._rpu_dinov3_handle_finalizer.alive
    assert not old_finalizer.alive
    assert model._rpu_dinov3_graph_cache is not old_cache
    _finalize(model)


def test_replacement_does_not_double_destroy_dead_finalizer(native_install):
    model = _model()
    assert _install(model) == 1
    model._rpu_dinov3_handle_finalizer()
    native_install["events"].clear()

    assert _install(model) == 2

    assert native_install["events"] == [
        ("create", 2),
        ("set_weights", 2),
        ("set_rope", 2),
        ("keepalive", 2),
    ]
    assert native_install["alive"] == {2}
    assert model._rpu_dinov3_handle == 2
    assert model._rpu_dinov3_handle_finalizer.alive
    _finalize(model)


def test_outer_failure_clears_installed_runtime(native_install, monkeypatch):
    model = _model()
    adapter = dinov3.DINOv3Adapter(model)
    monkeypatch.setattr(dinov3, "_claim_live_instance", lambda owner: None)
    monkeypatch.setattr(
        dinov3, "_convert_dinov3_weights_for_rpu", lambda owner: None)
    monkeypatch.setattr(
        dinov3, "_install_dinov3_rpu_keepalives", lambda owner: None)

    def fail_runtime_attrs(owner):
        raise RuntimeError("synthetic runtime-attr failure")

    monkeypatch.setattr(dinov3, "_apply_runtime_attrs", fail_runtime_attrs)
    with pytest.raises(RuntimeError, match="runtime-attr failure"):
        adapter.to_rpu()

    assert native_install["events"] == [
        ("create", 1),
        ("set_weights", 1),
        ("set_rope", 1),
        ("keepalive", 1),
        ("destroy", 1),
    ]
    assert native_install["alive"] == set()
    assert native_install["graph_caches"][0].cleared
    assert not hasattr(model, "_rpu_dinov3_handle")
    assert not hasattr(model, "_rpu_swizzled")
    assert model._rpu_swizzle_started is True


def test_dead_finalizer_without_resource_state_is_not_retirement_proof(native_install):
    model = _model()
    model._rpu_dinov3_handle = 41
    model._rpu_dinov3_handle_finalizer = SimpleNamespace(alive=False)
    with pytest.raises(RuntimeError, match="actual resource state"):
        dinov3._close_dinov3_model(model)
    assert native_install["events"] == []
    assert model._rpu_dinov3_handle == 41
    assert not model._rpu_dinov3_handle_finalizer.alive


def test_outer_failure_after_forward_binding_restores_original(
    native_install, monkeypatch
):
    model = _model()
    original_forward = model.forward
    adapter = dinov3.DINOv3Adapter(model)
    monkeypatch.setattr(dinov3, "_claim_live_instance", lambda owner: None)
    monkeypatch.setattr(
        dinov3, "_convert_dinov3_weights_for_rpu", lambda owner: None)
    monkeypatch.setattr(
        dinov3, "_install_dinov3_rpu_keepalives", lambda owner: None)
    monkeypatch.setattr(dinov3, "_apply_runtime_attrs", lambda owner: None)

    def fail_profile(config):
        raise RuntimeError("synthetic post-forward failure")

    monkeypatch.setattr(dinov3, "_profile", fail_profile)
    with pytest.raises(RuntimeError, match="post-forward failure"):
        adapter.to_rpu()

    assert native_install["alive"] == set()
    assert not hasattr(model, "_rpu_dinov3_handle")
    assert "forward" not in vars(model)
    assert model.forward.__func__ is original_forward.__func__
    assert not hasattr(model, "_rpu_swizzled")


def _prepare_outer_install(monkeypatch, model):
    monkeypatch.setattr(dinov3, "_claim_live_instance", lambda owner: None)

    def convert(owner):
        owner._rpu_dinov3_patch_embed_w = _FakeTensor()
        owner._rpu_dinov3_patch_embed_b = _FakeTensor()

    def keepalives(owner):
        owner._rpu_dinov3_cls_reg = _FakeTensor()
        owner._rpu_dinov3_gamma_attn = _FakeTensor()
        owner._rpu_dinov3_gamma_mlp = _FakeTensor()

    monkeypatch.setattr(dinov3, "_convert_dinov3_weights_for_rpu", convert)
    monkeypatch.setattr(
        dinov3, "_install_dinov3_rpu_keepalives", keepalives
    )


def test_outer_ready_requires_complete_runtime_and_reentry_is_noop(
    native_install, monkeypatch
):
    model = _model()
    _prepare_outer_install(monkeypatch, model)
    adapter = dinov3.DINOv3Adapter(model)

    assert adapter.to_rpu() is model
    assert dinov3._dinov3_runtime_complete(model) is True
    first_events = list(native_install["events"])

    second_adapter = dinov3.DINOv3Adapter(model)
    assert second_adapter.to_rpu() is model
    assert native_install["events"] == first_events
    _finalize(model)


def test_outer_incomplete_ready_marker_is_rejected_before_mutation(
    native_install, monkeypatch
):
    model = _model()
    model._rpu_swizzled = True
    claims = []
    monkeypatch.setattr(
        dinov3, "_claim_live_instance", lambda owner: claims.append(owner)
    )
    adapter = dinov3.DINOv3Adapter(model)

    with pytest.raises(RPUBackendError, match="runtime is incomplete"):
        adapter.to_rpu()

    assert claims == []
    assert native_install["events"] == []


def test_outer_ready_publish_base_exception_cleans_and_poison_model(
    native_install, monkeypatch
):
    model = _model()
    original_forward = model.forward
    _prepare_outer_install(monkeypatch, model)
    adapter = dinov3.DINOv3Adapter(model)
    model._fail_publish_attr = "_rpu_swizzled"

    with pytest.raises(_InstallInterrupted, match="publication interruption"):
        adapter.to_rpu()

    assert adapter._rpu_is_ready is False
    assert model._rpu_swizzle_started is True
    assert not hasattr(model, "_rpu_swizzled")
    assert native_install["alive"] == set()
    assert native_install["graph_caches"][0].cleared
    assert not hasattr(model, "_rpu_dinov3_handle")
    assert "forward" not in vars(model)
    assert model.forward.__func__ is original_forward.__func__
    with pytest.raises(RPUBackendError, match="swizzled|RPU"):
        model.to("cpu")
