"""Actual System-2 constructors/GC, with only device operations replaced on CPU."""
import gc
import json
from types import SimpleNamespace
import weakref

import pytest
import torch

from rpu_backend.adapters.internvla_n1 import backbone
from rpu_backend.api import _execution, causal_lm
from rpu_backend.runtime import _native_retirement


class Cache:
    def __init__(self, events):
        self.events = events

    def clear(self):
        self.events.append("clear")

    def cache_invariant_ok(self):
        self.events.append("invariant")
        return True


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    state = {"events": [], "destroy_error": None, "resources": []}
    def destroy(handle):
        state["events"].append(("destroy", handle))
        if state["destroy_error"] is not None:
            raise state["destroy_error"]
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_destroy", destroy, raising=False)
    yield state
    # CPU doubles must not run after their per-test process latch is restored.
    for resource in state["resources"] + _native_retirement._FAILED_RETIREMENTS:
        resource.finalizer.detach()


def actual(native):
    model = backbone.Qwen25VLBackbone(
        handle=41, cache=SimpleNamespace(), graph_cache=Cache(native["events"]),
        hidden_size=16, num_layers=1, head_dim=16, rope_theta=10000,
        mrope_section=[2, 2, 4], partial_mrope=False, host_cache=True)
    native["resources"].append(model._native_resource)
    return model


@pytest.mark.parametrize("length,stage,expected", [
    (260, {"chunk_size": "auto"}, (272, None, "auto", 64)),
    (289, {"chunk_size": "auto"}, (304, None, "auto", 64)),
    (289, {"chunk_size": "auto", "padding_rows": 0}, (289, None, 0, 0)),
    (1904, {"chunk_size": "auto"}, (1936, 176, 32, 0)),
    (1904, {"chunk_size": 160, "padding_rows": 0}, (1904, 160, 0, 0)),
    (1904, {"chunk_size": 160, "padding_rows": "auto", "padding_budget": 16},
     (1920, 160, "auto", 16)),
    (260, {"chunk_size": "auto", "padding_rows": "auto", "padding_budget": 32},
     (272, None, "auto", 32)),
])
def test_cold_prefill_and_forward_share_original_planner(native, monkeypatch, length, stage, expected):
    model = actual(native)
    model.cache.max_seq_len = 4096
    model.cache.position = 19
    model._rpu_execution = {"prefill": stage}
    model._fmb_execution_generation = 7
    calls = []
    plan = SimpleNamespace(selected=SimpleNamespace(
        stage_tuple=SimpleNamespace(physical_descriptor=(2, 3, 4))))
    def resolve(*args):
        calls.append(args)
        return [1, 2, 3]
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_prefill_stage_domain", resolve, raising=False)
    def bounded(logical, limit, budget, **kwargs):
        execution, chunk, padding, expected_budget = expected
        assert (logical, limit, budget) == (length, model.cache.max_seq_len, expected_budget)
        assert kwargs["execution_owner"] is model
        assert kwargs["execution_native"] == ("causal_decoder", 41)
        assert (kwargs["execution_component"], kwargs["execution_stage"], kwargs["graph_mode"]) == (
            "system2_backbone", "prefill", "COMPOSITE_CHILD")
        assert (kwargs["exact_chunk_size"], kwargs["padding_rows"]) == (chunk, padding)
        assert kwargs["physical_metadata"] == (("execution_generation", 7),)
        assert kwargs["queue_owner_id"] == 41 and kwargs["alignment"] == 1
        assert kwargs["resolve_stage_domain"](execution) == [1, 2, 3]
        kwargs["plan_result_sink"](plan)
        return execution, chunk or 272
    monkeypatch.setattr(backbone, "plan_bounded_prefill_execution", bounded)
    assert model._prefill_execution_plan(length) == (expected[0], expected[1] or 272, plan)
    assert calls == [(41, expected[0], 0, True, 0, 1, 4, length)]
    assert model.cache.position == 19 and native["events"] == []

    # The production forward must consume that same helper before any transfer
    # or native forward; this check never creates an device tensor or graph.
    class ReachedPlanner(Exception):
        pass
    def stop(logical):
        assert logical == length
        raise ReachedPlanner
    monkeypatch.setattr(model, "_prefill_execution_plan", stop)
    with pytest.raises(ReachedPlanner):
        model.forward_embeds(torch.ones(1, length, 16), torch.zeros(length, 3, dtype=torch.long))
    assert model.cache.position == 19 and native["events"] == []
    model.destroy()


@pytest.mark.parametrize("fault", [None, "clear", "invariant", "destroy"])
def test_actual_destroy_is_graph_first_and_never_retries_failure(native, fault):
    model = actual(native)
    resource = model._native_resource
    if fault == "clear":
        model._graph_cache.clear = lambda: (_ for _ in ()).throw(RuntimeError("clear failed"))
    elif fault == "invariant":
        model._graph_cache.cache_invariant_ok = lambda: False
    elif fault == "destroy":
        native["destroy_error"] = KeyboardInterrupt("destroy failed")
    if fault is None:
        model.destroy()
        model.destroy()
        assert native["events"] == ["clear", "invariant", ("destroy", 41)]
        assert model._handle is None and resource.handle is None
    else:
        with pytest.raises((RuntimeError, KeyboardInterrupt)):
            model.destroy()
        assert resource.failed and resource.handle == 41 and resource.keepalive
        assert _execution._UNSAFE_PROCESS_REASON
        before = list(native["events"])
        with pytest.raises(RuntimeError):
            model.destroy()
        resource.finalizer()
        assert native["events"] == before


@pytest.mark.parametrize("closed", [False, True])
def test_actual_session_controls_destroy_before_any_native(native, closed):
    model = actual(native)
    session = _execution.bind_execution_session(
        model, {}, entry_point="System-2-test", graph_mode="COMPOSITE_CHILD")
    if closed:
        session.close()
        with pytest.raises(RuntimeError, match="still owns native resources"):
            model.destroy()
        assert model._handle == 41 and model._native_resource.failed
    else:
        with session.execute():
            with pytest.raises(RuntimeError, match="during a forward"):
                model.destroy()
        assert native["events"] == [] and _execution._UNSAFE_PROCESS_REASON is None
        model.destroy()
        assert session.stats()["state"] == "CLOSED"
    if closed:
        assert native["events"] == [] and _execution._UNSAFE_PROCESS_REASON


@pytest.mark.parametrize("failed", [False, True])
def test_actual_last_owner_gc_holds_graph_inputs_until_success(native, failed):
    model = actual(native)
    resource = model._native_resource
    tensor = torch.ones(1)
    model._prefill_input_cache[1] = tensor
    tensor_ref, model_ref = weakref.ref(tensor), weakref.ref(model)
    if failed:
        native["destroy_error"] = RuntimeError("GC destroy failed")
    del model, tensor
    gc.collect()
    gc.collect()
    assert model_ref() is None
    assert native["events"] == ["clear", "invariant", ("destroy", 41)]
    assert (tensor_ref() is not None) is failed
    assert (resource.handle == 41) is failed
    assert bool(_execution._UNSAFE_PROCESS_REASON) is failed


@pytest.mark.parametrize("failure", [None, "set_weights", "constructor"])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_real_builder_preserves_original_failure_and_pending_weights(
        native, monkeypatch, tmp_path, failure, cleanup_failure):
    config = dict(backbone._PROFILE, rope_scaling={"mrope_section": backbone._MROPE_SECTION})
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(backbone, "require_controlled_evaluation", lambda *a: None)
    monkeypatch.setattr(backbone, "verify_asset_manifest", lambda *a: None)
    monkeypatch.setattr(backbone, "_validate_checkpoint_profile", lambda *a: None)
    class Store:
        weight_map = {}
        def get_tensor(self, name):
            return torch.ones(1)
    monkeypatch.setattr(backbone, "_SafeTensorStore", lambda *a, **k: Store())
    for name in ("_to_rpu_half", "_to_rpu_int8", "tp_col_swizzle_mc_weight", "tp_row_swizzle_mc_weight"):
        monkeypatch.setattr(backbone, name, lambda tensor, *a, **k: tensor)
    monkeypatch.setattr(backbone, "_build_rope_tables", lambda *a: (torch.ones(1), torch.ones(1)))
    monkeypatch.setattr(backbone, "RPUCache", lambda **k: SimpleNamespace())
    monkeypatch.setattr(backbone.rpu_backend.graph, "GraphCache", lambda: Cache(native["events"]))
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_create", lambda: 41, raising=False)
    primary = KeyboardInterrupt("original System-2 build failure")
    weight_refs = []
    def set_weights(handle, *args):
        weight_refs.append(weakref.ref(args[0][0]))
        if failure == "set_weights":
            raise primary
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_w8a16", set_weights, raising=False)
    for op in ("causal_decoder_set_chunk_envelope", "causal_decoder_set_chunk_size_override"):
        monkeypatch.setattr(torch.ops.rpu, op, lambda *a: None, raising=False)
    if failure == "constructor":
        original = backbone.Qwen25VLBackbone.__init__
        def init(self, **kwargs):
            original(self, **kwargs)
            raise primary
        monkeypatch.setattr(backbone.Qwen25VLBackbone, "__init__", init)
    if cleanup_failure:
        native["destroy_error"] = RuntimeError("pending cleanup failed")
    if failure is None:
        model = backbone.build_qwen25vl_backbone(tmp_path, asset_manifest={})
        assert weight_refs[0]() is not None
        reference = weakref.ref(model)
        del model
        gc.collect()
        gc.collect()
        assert reference() is None
        assert native["events"] == ["clear", "invariant", ("destroy", 41)]
        assert (weight_refs[0]() is not None) is cleanup_failure
        assert bool(_execution._UNSAFE_PROCESS_REASON) is cleanup_failure
        return
    with pytest.raises(KeyboardInterrupt) as caught:
        backbone.build_qwen25vl_backbone(tmp_path, asset_manifest={})
    assert caught.value is primary
    assert native["events"] == ["clear", "invariant", ("destroy", 41)]
    if cleanup_failure:
        resource, = _native_retirement._FAILED_RETIREMENTS
        assert resource.handle == 41 and resource.failed and resource.owner() is not None
        assert weight_refs[0]() is not None and "cleanup failed" in primary.__notes__[0]
        before = list(native["events"])
        resource.finalizer()
        gc.collect()
        assert native["events"] == before
