"""Board-free failure cleanup contract for the Qwen3.5 text adapter."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import qwen3_5


def test_failed_text_cleanup_bypasses_rejecting_attribute_hooks(monkeypatch):
    class RejectingOwner:
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

        def cache_invariant_ok(self):
            return True

    class Finalizer:
        alive = True

        def detach(self):
            self.alive = False
            events.append("detach")
            return True

    import torch
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_destroy", lambda _h: events.append("destroy"), raising=False)
    runtime_state = SimpleNamespace(
        handle=31,
        graph_cache=Cache(),
        handle_finalizer=Finalizer(),
    )
    inner = RejectingOwner()
    vars(inner).update(
        {
            "_rpu_qwen3_5": runtime_state,
            "_reject_runtime_attrs": True,
        }
    )
    model = RejectingOwner()
    original_forward = lambda: "original-forward"
    vars(model).update(
        {
            "forward": lambda: "patched-forward",
            "_lm_head_w_rpu_keepalive": object(),
            "_reject_runtime_attrs": True,
        }
    )
    adapter = SimpleNamespace(
        _handle=31,
        _graph_cache=runtime_state.graph_cache,
        _rpu_is_ready=True,
    )

    qwen3_5._cleanup_failed_text_install(
        adapter,
        model,
        inner,
        True,
        original_forward,
    )

    assert events == ["clear", "destroy", "detach"]
    assert not hasattr(inner, "_rpu_qwen3_5")
    assert not hasattr(model, "_lm_head_w_rpu_keepalive")
    assert model.forward is original_forward
    assert adapter._handle is None
    assert adapter._graph_cache is None
    assert adapter._rpu_is_ready is False


@pytest.mark.parametrize("release_source", [False, True])
def test_untied_head_retires_source_before_device_allocation(monkeypatch, release_source):
    head = torch.nn.Linear(32, 128, bias=False, dtype=torch.float16)
    embedding = torch.nn.Embedding(128, 32, dtype=torch.float16)
    shared_parameter = head.weight
    expected = qwen3_5.transform_linear_weight(head.weight.detach(), 1, 4)
    events = []
    original_to = torch.Tensor.to

    def upload(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            assert head.weight is shared_parameter
            assert head.weight.numel() == (0 if release_source else 4096)
            assert events == (["release"] if release_source else [])
            assert torch.equal(tensor, expected)
            return tensor.clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", upload)
    monkeypatch.setattr("rpu_backend.runtime.weights._release_cpu_weight_pages",
                        lambda: events.append("release"))
    result = qwen3_5._upload_lm_head_weight(head, embedding, 4, release_source=release_source)
    assert torch.equal(result, expected)
    assert embedding.weight.numel() == 4096


@pytest.mark.parametrize("sharing", ["parameter", "storage", "fallback"])
def test_head_retirement_preserves_tied_embedding(monkeypatch, sharing):
    embedding = torch.nn.Embedding(128, 32, dtype=torch.float16)
    head = torch.nn.Linear(32, 128, bias=False, dtype=torch.float16)
    head.weight = (embedding.weight if sharing == "parameter" else
                   torch.nn.Parameter(embedding.weight.detach()))
    original = embedding.weight.detach().clone()
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda tensor, *a, **k:
                        tensor.clone() if a and a[0] == "rpu" else original_to(tensor, *a, **k))
    monkeypatch.setattr("rpu_backend.runtime.weights._release_cpu_weight_pages",
                        lambda: pytest.fail("tied embedding must remain live"))
    qwen3_5._upload_lm_head_weight(
        None if sharing == "fallback" else head, embedding, 4, release_source=True)
    assert torch.equal(embedding.weight, original)
    assert torch.equal(head.weight, original)


def test_head_upload_failure_is_propagated_after_source_retirement(monkeypatch):
    head = torch.nn.Linear(32, 128, bias=False, dtype=torch.float16)
    embedding = torch.nn.Embedding(128, 32, dtype=torch.float16)
    original_to = torch.Tensor.to

    def upload(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            assert head.weight.numel() == 0
            raise RuntimeError("device allocation failed")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", upload)
    monkeypatch.setattr("rpu_backend.runtime.weights._release_cpu_weight_pages", lambda: None)
    with pytest.raises(RuntimeError, match="device allocation failed"):
        qwen3_5._upload_lm_head_weight(head, embedding, 4, release_source=True)
