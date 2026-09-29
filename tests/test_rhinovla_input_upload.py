"""Packed request publication and content-based mask reuse without a board."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


def _helpers():
    path = Path(__file__).parents[1] / "python/rpu_backend/adapters/rhinovla/pipeline.py"
    names = {"_upload_denoise_inputs", "_reuse_denoise_attention_mask"}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


@pytest.mark.parametrize("first_inference", [False, True])
def test_one_publication_refreshes_every_field_and_reuses_tracked_storage(monkeypatch, first_inference):
    empty_like, copy = torch.empty_like, torch.Tensor.copy_
    devices, publications = [], []

    def allocate(value, **kw):
        assert kw.pop("device") == "rpu"
        device = empty_like(value, **kw)
        devices.append(device)
        return device

    def copy_(destination, source, *args, **kw):
        if any(destination is device for device in devices):
            publications.append(destination)
        return copy(destination, source, *args, **kw)

    monkeypatch.setattr(torch, "empty_like", allocate)
    monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    upload = _helpers()["_upload_denoise_inputs"]
    owner = NS()
    inputs = [torch.arange(21).reshape(3, 7).float(),
              torch.ones(1, 2, 5, dtype=torch.bool), torch.ones(1, 5),
              torch.arange(10).reshape(1, 5, 2).transpose(1, 2).float(),
              torch.arange(10).reshape(1, 10)[:, ::2].float()]
    with torch.inference_mode(first_inference):
        first = upload(owner, inputs)
    assert len(publications) == len(devices) == 1
    for result, source in zip(first, inputs):
        assert torch.equal(result, source.half())
        assert result.is_contiguous() and not result.is_inference()
        assert result.data_ptr() % 64 == 0
        assert result.untyped_storage().data_ptr() == devices[0].data_ptr()
    saved = [value.clone() for value in first]
    for value in inputs:
        if value.dtype == torch.bool:
            value.logical_not_()
        else:
            value.add_(0.5)
    second = upload(owner, inputs)
    assert second is first and len(publications) == 2 and len(devices) == 1
    assert all(torch.equal(result, source.half()) for result, source in zip(second, inputs))
    assert all(not torch.equal(a, b) for a, b in zip(saved, second))
    # Geometry replacement is bounded to one current buffer and cannot alias
    # an older shape's borrowed views still retained by a caller.
    changed = [value.reshape(-1) for value in inputs]
    third = upload(owner, changed)
    assert len(devices) == 2 and len(publications) == 3
    assert third[0].untyped_storage().data_ptr() != first[0].untyped_storage().data_ptr()
    assert owner._denoise_input_upload[4] is third


def test_resident_inputs_keep_the_existing_direct_conversion_path():
    calls = []

    class Input:
        def __init__(self, device):
            self.device = NS(type=device)

        def to(self, **kw):
            calls.append(kw)
            return self

        def contiguous(self):
            return self

    owner = NS()
    inputs = (Input("rpu"), Input("cpu"))
    assert _helpers()["_upload_denoise_inputs"](owner, inputs) == inputs
    assert calls == [{"device": "rpu", "dtype": torch.float16}] * 2
    assert not vars(owner)


def test_mask_reuse_compares_values_owns_storage_and_tracks_inference_inputs():
    reuse = _helpers()["_reuse_denoise_attention_mask"]
    owner = NS()
    with torch.inference_mode():
        source = torch.zeros(1, 1, 3, 16, dtype=torch.float16)
        first = reuse(owner, source)
    assert not first.is_inference() and first._version >= 0
    version = first._version
    assert first.data_ptr() != source.data_ptr()
    assert reuse(owner, torch.zeros_like(source)) is first
    assert first._version == version
    with torch.inference_mode():
        source[..., -1] = -torch.inf
    second = reuse(owner, source)
    assert second is not first and torch.equal(second, source)
    assert torch.count_nonzero(first) == 0
    assert reuse(owner, source.clone()) is second
    restored = reuse(owner, torch.zeros_like(source))
    assert restored is not second and torch.equal(restored, first)
    assert reuse(owner, restored.float()).dtype == torch.float32
    assert reuse(owner, restored[:, :, :1]).shape == (1, 1, 1, 16)
