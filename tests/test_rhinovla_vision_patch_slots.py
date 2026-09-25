"""Private patch slots refresh values and preserve eager accumulation choices."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


def _helper():
    path = Path(__file__).parents[1] / "python/rpu_backend/adapters/rhinovla/vision.py"
    wanted = {"_bounded_cache_put", "_cached_vision_patch_projection"}
    body = [node for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in wanted]
    scope = {"torch": torch, "Any": Any, "_RPU_VISION_INPUT_CACHE_MAX_ENTRIES": 4}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), scope)
    return scope["_cached_vision_patch_projection"]


@pytest.mark.parametrize("w8", [False, True])
def test_patch_slot_refreshes_after_inference_caller_without_position_accumulation(monkeypatch, w8):
    calls = []
    def linear(x, weight, bias, out, acc32):
        calls.append(acc32)
        out.copy_((x.float() @ weight.float().T + bias.float()).half())
    def quantized(x, weight, scale, bias, out, acc32):
        linear(x, weight.float() * scale.float()[:, None], bias, out, acc32)
    monkeypatch.setattr(torch.ops.rpu, "linear_into", linear, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "linear_w8a16_into", quantized, raising=False)
    model = SimpleNamespace(
        _rpu_vision_patch_embed_w_rpu=torch.tensor([[1, 2, 3], [3, 2, 1]],
                                                  dtype=torch.int8 if w8 else torch.float16),
        _rpu_vision_patch_embed_b_rpu=torch.ones(2, dtype=torch.float16),
        _rpu_vision_patch_embed_scale=torch.ones(2, dtype=torch.float16),
        _rpu_vision_w8a16=w8,
        _rpu_vision_patch_input_cache={},
    )
    project = _helper()
    with torch.inference_mode():
        first = project(model, torch.ones((2, 3), dtype=torch.float16))
        first.add_(2)  # The caller's position addition must not survive refresh.
        held = first.clone()
    assert not first.is_inference()
    second = project(model, torch.full((2, 3), 2, dtype=torch.float16))
    assert second is first
    assert torch.equal(second, torch.full((2, 2), 13, dtype=torch.float16))
    assert torch.equal(held, torch.full((2, 2), 9, dtype=torch.float16))
    assert calls == [not w8, not w8]
    for rows in range(3, 10):
        assert project(model, torch.ones((rows, 3), dtype=torch.float16)).shape == (rows, 2)
    assert len(model._rpu_vision_patch_input_cache) == 4
