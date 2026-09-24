"""Public greedy selection contracts without loading the native backend."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_generation_test", ROOT / "python/rpu_backend/api/generation.py")
generation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generation)
greedy_token_ids = generation.greedy_token_ids


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("sequence", [False, True])
def test_ties_nan_infinities_noncontiguous_and_independent_output(dtype, sequence):
    values = torch.tensor([
        [1, 3, 3, 2, 0, 0, 0, 0, 3, 0, 0, 0, 3],
        [1, float("nan"), 4, 0, 0, 0, 0, 0, float("nan"), 0, 0, 0, 0],
        [float("-inf")] * 13,
        [float("inf"), 0, 0, 0, 0, 0, 0, 0, float("inf"), 0, 0, 0, 0],
        [-0., 0., -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    ], dtype=dtype)
    # A strided vocabulary and a separate earlier sequence row must not change
    # which physical logits are selected, or cause an in-place normalization.
    storage = torch.zeros((*values.shape, 2), dtype=dtype)
    storage[..., 0] = values
    logits = storage[..., 0]
    if sequence:
        logits = torch.stack((torch.zeros_like(logits), logits), dim=1)[..., ::2]
    before = logits.contiguous().view(torch.uint8).clone()
    last = logits[:, -1] if sequence else logits
    expected = last.argmax(dim=-1, keepdim=True)
    first, second = greedy_token_ids(logits), greedy_token_ids(logits)
    assert first.device.type == "cpu" and first.dtype == torch.int64
    assert first.shape == (values.shape[0], 1) and torch.equal(first, expected)
    assert torch.equal(first, second) and first.data_ptr() != second.data_ptr()
    first.fill_(-1)
    assert torch.equal(second, expected)
    assert torch.equal(logits.contiguous().view(torch.uint8), before)


@pytest.mark.parametrize("dtype,values", [
    (torch.float32, [1., 1. + 2. ** -20]),
    (torch.float64, [1., 1. + 2. ** -40]),
    (torch.bfloat16, [65536., 131072.]),
])
def test_cpu_precision_is_not_narrowed(dtype, values):
    assert greedy_token_ids(torch.tensor([values], dtype=dtype)).item() == 1


@pytest.mark.parametrize("shape", [(0, 13), (0, 2, 13)])
def test_empty_batch(shape):
    result = greedy_token_ids(torch.empty(shape))
    assert result.shape == (0, 1) and result.dtype == torch.int64 and result.device.type == "cpu"


@pytest.mark.parametrize("value,error", [
    ([1., 2.], TypeError), (torch.tensor(1.), ValueError),
    (torch.ones(3), ValueError), (torch.ones(1, 1, 1, 3), ValueError),
    (torch.empty(1, 0), ValueError), (torch.empty(1, 0, 3), ValueError),
    (torch.ones(1, 3, dtype=torch.int64), TypeError),
    (torch.ones(1, 3, dtype=torch.bool), TypeError),
    (torch.ones(1, 3, dtype=torch.complex64), TypeError),
    (torch.ones(1, 3, device="meta"), ValueError),
    (torch.ones(1, 3).to_sparse(), TypeError),
])
def test_invalid_input_fails(value, error):
    with pytest.raises(error):
        greedy_token_ids(value)


class RpuStandIn(torch.Tensor):
    """Exercise Python routing only; no claim about native device numerics."""
    @property
    def device(self):
        return SimpleNamespace(type="rpu")


def test_rpu_route_uses_only_final_rows_and_rejects_non_fp16(monkeypatch):
    calls = []
    def native(rows):
        calls.append(rows)
        return rows.as_subclass(torch.Tensor).argmax(-1)
    monkeypatch.setattr(torch.ops.rpu, "argmax_lastdim_host", native, raising=False)
    values = torch.arange(2 * 3 * 13, dtype=torch.float16).view(2, 3, 13)
    got = greedy_token_ids(values.as_subclass(RpuStandIn))
    assert got.tolist() == [[12], [12]]
    assert len(calls) == 1 and calls[0].shape == (2, 13) and calls[0].is_contiguous()
    assert torch.equal(calls[0].as_subclass(torch.Tensor), values[:, -1])
    with pytest.raises(TypeError, match="FP16 RPU"):
        greedy_token_ids(values.float().as_subclass(RpuStandIn))
    assert len(calls) == 1
