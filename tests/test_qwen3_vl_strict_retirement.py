"""Actual composite/Session/GC flow; native operations are CPU doubles."""
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from rpu_backend.adapters import qwen3_vl as qvl
from rpu_backend.api import _execution, causal_lm
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import install_runtime, _Model  # noqa: F401


@pytest.fixture
def actual(install_runtime, monkeypatch):
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setattr(qvl, "_claim_live_instance", causal_lm._claim_live_instance)
    for name, finalizer_name, destroy in (
        ("install_qwen3_vl_vision_for_rpu", "_rpu_vision_handle_finalizer", "qwen3vl_vision_destroy"),
        ("install_qwen3_vl_text_for_rpu", "_rpu_decoder_handle_finalizer", "causal_decoder_destroy"),
    ):
        original = getattr(qvl, name)
        def install(child, original=original, finalizer_name=finalizer_name, destroy=destroy, **kwargs):
            handle = original(child, **kwargs)
            setattr(child, finalizer_name, weakref.finalize(
                child, lambda handle: getattr(torch.ops.rpu, destroy)(handle), handle))
            return handle
        monkeypatch.setattr(qvl, name, install)
    return install_runtime


@pytest.mark.parametrize("fault", ["clear", "invariant", "destroy"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_close_retains_owner_and_never_retries(actual, monkeypatch, fault, error_type):
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    child = model.model.visual
    original = error_type("actual retirement failed")
    def fail(*args):
        raise original
    if fault == "clear":
        child._rpu_vision_graph_cache.clear = fail
    elif fault == "invariant":
        child._rpu_vision_graph_cache.cache_invariant_ok = lambda: False
    else:
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", fail, raising=False)
    actual["events"].clear()
    with pytest.raises(RuntimeError if fault == "invariant" else error_type) as caught:
        adapter.close()
    if fault != "invariant":
        assert caught.value is original
    assert child._rpu_vision_handle == 11
    assert not adapter._closed and adapter in qvl._FAILED_RETIREMENTS
    assert model in qvl._FAILED_RETIREMENTS and _execution._UNSAFE_PROCESS_REASON
    assert adapter._execution_session.stats()["state"] == "POISONED"
    if fault != "destroy":
        assert not any(event[0] == "destroy" for event in actual["events"])
        assert model.model.language_model._rpu_decoder_handle == 22
    before = list(actual["events"])
    with pytest.raises(RuntimeError):
        adapter.close()
    adapter.__del__()
    assert actual["events"] == before
    with pytest.raises(RuntimeError, match="unsafe"):
        causal_lm._claim_live_instance(_Model())


def test_failed_build_preserves_original_error_and_all_remaining_resources(actual, monkeypatch):
    original = KeyboardInterrupt("original lm head failed")
    actual["set_error"] = original
    install = qvl.install_qwen3_vl_vision_for_rpu
    def install_vision(child, **kwargs):
        handle = install(child, **kwargs)
        def failed_clear():
            raise RuntimeError("graph retirement failed")
        child._rpu_vision_graph_cache.clear = failed_clear
        return handle
    monkeypatch.setattr(qvl, "install_qwen3_vl_vision_for_rpu", install_vision)
    adapter = qvl.Qwen3VLAdapter(_Model())
    with pytest.raises(KeyboardInterrupt) as caught:
        adapter.to_rpu()
    assert caught.value is original and "graph retirement failed" in original.__notes__[0]
    assert not any(event[0] == "destroy" for event in actual["events"])
    assert adapter in qvl._FAILED_RETIREMENTS
    assert adapter.model.model.language_model._rpu_decoder_handle == 22
    assert adapter.model.model.visual._rpu_vision_handle == 11
    assert _execution._UNSAFE_PROCESS_REASON


@pytest.mark.parametrize("failed", [False, True])
def test_actual_cyclic_gc_clears_all_graphs_before_native(actual, monkeypatch, failed):
    if failed:
        def destroy(handle):
            actual["events"].append(("destroy", "vision"))
            raise RuntimeError("vision destroy failed")
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", destroy, raising=False)
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    assert not model.model.visual._rpu_vision_handle_finalizer.alive
    assert not model.model.language_model._rpu_decoder_handle_finalizer.alive
    reference = weakref.ref(model)
    actual["events"].clear()
    del adapter, model
    gc.collect()
    assert actual["events"] == [
        ("clear", "text"), ("clear", "decoder"), ("clear", "vision"),
        ("destroy", "text"), ("destroy", "vision")]
    before = list(actual["events"])
    gc.collect()
    assert actual["events"] == before
    if failed:
        assert qvl._FAILED_RETIREMENTS and _execution._UNSAFE_PROCESS_REASON
    else:
        assert reference() is None and _execution._UNSAFE_PROCESS_REASON is None


def test_secondary_adapter_cannot_gc_retire_the_primary_owner(actual):
    primary = qvl.Qwen3VLAdapter(_Model())
    model = primary.to_rpu()
    secondary = qvl.Qwen3VLAdapter(model)
    assert secondary.to_rpu() is model
    before = list(actual["events"])
    secondary.__del__()
    assert actual["events"] == before and not primary._closed
    secondary.close()
    assert primary._closed and secondary._closed
    assert primary._execution_session.stats()["state"] == "CLOSED"


@pytest.mark.parametrize("mirror", [False, True])
def test_closed_model_and_children_cannot_reenter_swizzled_hf_forward(actual, mirror):
    primary = qvl.Qwen3VLAdapter(_Model())
    model = primary.to_rpu()
    adapter = qvl.Qwen3VLAdapter(model) if mirror else primary
    adapter.close()
    for child in (model, model.model.language_model, model.model.visual):
        with pytest.raises(qvl.RPUBackendError, match="closed.*reload"):
            child.forward()


def _public_example():
    path = Path(__file__).resolve().parents[1] / "examples/qwen3_vl.py"
    spec = importlib.util.spec_from_file_location("qwen3_vl_public_close", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", ["adapter", "example"])
def test_fake_parent_cannot_acknowledge_retirement(actual, entry):
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    fake = SimpleNamespace(model=model, close=lambda: pytest.fail("fake close invoked"))
    vars(model)["_qwen3_vl_retirement_owner"] = fake
    before = list(actual["events"])
    try:
        if entry == "example":
            close = lambda: _public_example().close_model(model)
        else:
            close = adapter.close
        with pytest.raises(RuntimeError, match="actual composite retirement owner"):
            close()
        assert not adapter._closed and actual["events"] == before
    finally:
        vars(model)["_qwen3_vl_retirement_owner"] = adapter
        adapter.close()


def test_public_example_uses_actual_composite_graph_first_retirement(actual):
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    actual["events"].clear()
    _public_example().close_model(model)
    assert adapter._closed and adapter._execution_session.stats()["state"] == "CLOSED"
    assert actual["events"] == [
        ("clear", "text"), ("clear", "decoder"), ("clear", "vision"),
        ("destroy", "text"), ("destroy", "vision")]


def _attach_shared_resources(monkeypatch):
    from rpu_backend.runtime import _native_retirement

    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    resources = []
    for name, prefix, graph_names, op in (
        ("install_qwen3_vl_text_for_rpu", "_rpu_decoder_",
         ("_rpu_text_graph_cache", "_rpu_decoder_graph_cache"), "causal_decoder_destroy"),
        ("install_qwen3_vl_vision_for_rpu", "_rpu_vision_",
         ("_rpu_vision_graph_cache",), "qwen3vl_vision_destroy"),
    ):
        original = getattr(qvl, name)
        def install(child, original=original, prefix=prefix, graph_names=graph_names, op=op, **kwargs):
            handle = original(child, **kwargs)
            getattr(child, prefix + "handle_finalizer").detach()
            state = _native_retirement._InstalledNativeResource(
                child, handle, lambda handle: getattr(torch.ops.rpu, op)(handle),
                graphs=[getattr(child, attr) for attr in graph_names],
                keepalive=(torch.ones(1),), label=prefix, handle_name=prefix + "handle")
            setattr(child, prefix + "retirement_state", state)
            setattr(child, prefix + "handle_finalizer", state.finalizer)
            resources.append(state)
            return handle
        monkeypatch.setattr(qvl, name, install)
    return resources


@pytest.mark.parametrize("fault", [None, "clear", "text_destroy", "vision_destroy"])
def test_composite_retirement_updates_actual_shared_resource_state(actual, monkeypatch, fault):
    resources = _attach_shared_resources(monkeypatch)
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    assert all(state.parent() is adapter and not state.finalizer.alive for state in resources)
    def fail(*args):
        raise RuntimeError("shared parent retirement failed")
    if fault == "clear":
        model.model.visual._rpu_vision_graph_cache.clear = fail
    elif fault is not None:
        op = "causal_decoder_destroy" if fault == "text_destroy" else "qwen3vl_vision_destroy"
        monkeypatch.setattr(torch.ops.rpu, op, fail, raising=False)
    actual["events"].clear()
    if fault is None:
        adapter.close()
        assert all(state.handle is None and not state.graphs and not state.keepalive for state in resources)
        assert actual["events"] == [
            ("clear", "text"), ("clear", "decoder"), ("clear", "vision"),
            ("destroy", "text"), ("destroy", "vision")]
    else:
        with pytest.raises(RuntimeError, match="shared parent retirement failed"):
            adapter.close()
        live = [state for state in resources if state.handle is not None]
        assert len(live) == (1 if fault == "vision_destroy" else 2)
        assert all(state.failed and state.keepalive and state.graphs for state in live)
    before = list(actual["events"])
    for state in resources:
        if state.failed:
            with pytest.raises(RuntimeError, match="already failed"):
                state.retire()
        else:
            state.retire()
        state.finalizer()
    assert actual["events"] == before


@pytest.mark.parametrize("failure", [False, True])
def test_actual_shared_resource_cyclic_gc_uses_strong_composite_owner(actual, monkeypatch, failure):
    resources = _attach_shared_resources(monkeypatch)
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    if failure:
        def destroy(handle):
            actual["events"].append(("destroy", "vision"))
            raise RuntimeError("shared Vision GC destroy failed")
        monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", destroy, raising=False)
    reference = weakref.ref(model)
    actual["events"].clear()
    del adapter, model
    gc.collect()
    gc.collect()
    assert reference() is None
    assert actual["events"] == [
        ("clear", "text"), ("clear", "decoder"), ("clear", "vision"),
        ("destroy", "text"), ("destroy", "vision")]
    live = [state for state in resources if state.handle is not None]
    if failure:
        assert len(live) == 1 and live[0].failed and live[0].keepalive
        assert qvl._FAILED_RETIREMENTS and _execution._UNSAFE_PROCESS_REASON
    else:
        assert live == [] and _execution._UNSAFE_PROCESS_REASON is None


def test_active_close_rejects_without_poison(actual):
    adapter = qvl.Qwen3VLAdapter(_Model())
    adapter.to_rpu()
    before = list(actual["events"])
    with adapter._execution_session.execute():
        with pytest.raises(RuntimeError, match="during a forward"):
            adapter.close()
    assert actual["events"] == before and _execution._UNSAFE_PROCESS_REASON is None
    adapter.close()


def test_closed_session_cannot_skip_live_native_retirement(actual):
    adapter = qvl.Qwen3VLAdapter(_Model())
    adapter.to_rpu()
    adapter._execution_session.close()
    before = list(actual["events"])
    with pytest.raises(RuntimeError, match="still owns native resources"):
        adapter.close()
    assert actual["events"] == before and _execution._UNSAFE_PROCESS_REASON
    assert adapter in qvl._FAILED_RETIREMENTS and not adapter._closed


def test_gc_after_live_handoff_is_terminal_without_touching_old_native(actual):
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    causal_lm._release_live_instance(model)
    replacement = _Model()
    causal_lm._claim_live_instance(replacement)
    before = list(actual["events"])
    adapter.__del__()
    assert actual["events"] == before and _execution._UNSAFE_PROCESS_REASON
    assert adapter in qvl._FAILED_RETIREMENTS


def test_public_example_retains_failed_owner_and_does_not_retry(actual, monkeypatch):
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    attempts = []
    def failed_destroy(handle):
        attempts.append(handle)
        raise RuntimeError("vision retirement failed")
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_destroy", failed_destroy, raising=False)
    example = _public_example()
    with pytest.raises(RuntimeError, match="vision retirement failed"):
        example.close_model(model)
    assert model._qwen3_vl_retirement_owner is adapter and not adapter._closed
    assert model in qvl._FAILED_RETIREMENTS and adapter in qvl._FAILED_RETIREMENTS
    assert model.model.visual._rpu_vision_handle == 11
    assert adapter._execution_session.stats()["state"] == "POISONED"
    with pytest.raises(RuntimeError):
        example.close_model(model)
    assert attempts == [11]


@pytest.mark.parametrize("fault", ["fake_parent", "foreign_session", "gc_disabled", "poisoned_session"])
def test_complete_requires_actual_live_session_controller(actual, fault):
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    children = (model, model.model.language_model, model.model.visual)
    session = adapter._execution_session
    try:
        if fault == "fake_parent":
            fake = SimpleNamespace(model=model, _closed=False, _retirement_failed=None)
            for child in children:
                vars(child)["_qwen3_vl_retirement_owner"] = fake
        elif fault == "foreign_session":
            adapter._execution_session = SimpleNamespace(_owner=model)
        elif fault == "gc_disabled":
            adapter._gc_retirement_enabled = False
        else:
            session.poison()
        assert not qvl._qwen3_vl_runtime_complete(model)
    finally:
        for child in children:
            vars(child)["_qwen3_vl_retirement_owner"] = adapter
        adapter._execution_session = session
        adapter._gc_retirement_enabled = True
        adapter.close()


@pytest.mark.parametrize('failed', (False, True))
def test_original_uninstall_prefix_cleanup_preserves_failed_owner_fields(actual, monkeypatch, failed):
    adapter = qvl.Qwen3VLAdapter(_Model())
    model = adapter.to_rpu()
    text, vision = model.model.language_model, model.model.visual
    text._rpu_decoder_hidden_size = 128
    text._rpu_decoder_num_layers = 1
    text._rpu_deepstack_lang_layers = (1,)
    # Conditional cleanup of another registered vision key is a host role;
    # planting it here does not turn QVL into the Q35 execution profile.
    vision._rpu_vision_control_snapshot = sentinel = object()
    vision.unrelated_prefix_sentinel = sentinel
    if failed:
        def fail():
            raise RuntimeError('prefix cleanup must follow retirement')
        vision._rpu_vision_graph_cache.clear = fail
        with pytest.raises(RuntimeError, match='prefix cleanup must follow'):
            adapter.close()
        assert text._rpu_decoder_hidden_size == 128 and text._rpu_decoder_num_layers == 1
        assert vision._rpu_vision_control_snapshot is sentinel
    else:
        adapter.close()
        assert not any(name.startswith(('_rpu_decoder_', '_rpu_text_')) for name in vars(text))
        assert '_rpu_deepstack_lang_layers' not in vars(text)
        assert not any(name.startswith('_rpu_vision_') for name in vars(vision))
    assert vision.unrelated_prefix_sentinel is sentinel
