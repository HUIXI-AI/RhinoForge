"""Public prefill requests retain explicit controls when using shared planning."""
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("logical_rows", [128, 480, 8200])
@pytest.mark.parametrize(
    "stage, expected_chunk, expected_padding, expected_budget",
    [({}, None, "auto", 64), ({"chunk_size": 64, "padding_rows": 16}, 64, 16, 0),
     ({"padding_rows": "auto", "padding_budget": 64}, None, "auto", 64)],
)
def test_prefill_planning_preserves_explicit_public_options(
    monkeypatch, logical_rows, stage, expected_chunk, expected_padding, expected_budget
):
    from rpu_backend.adapters.internvla_n1 import backbone
    calls = []
    descriptor = object()
    result = SimpleNamespace(selected=SimpleNamespace(
        stage_tuple=SimpleNamespace(physical_descriptor=descriptor)))

    def planner(logical, max_length, budget, **kwargs):
        calls.append((logical, max_length, budget, kwargs))
        kwargs["plan_result_sink"](result)
        padding = 16 if kwargs["padding_rows"] == "auto" else kwargs["padding_rows"]
        return logical + padding, kwargs["exact_chunk_size"] or 128

    monkeypatch.setattr(backbone, "plan_bounded_prefill_execution", planner)
    model = SimpleNamespace(
        _handle=7, _rpu_execution={"prefill": stage},
        cache=SimpleNamespace(max_seq_len=16384), _graph_cache=object(),
    )
    execution_rows, chunk, actual = backbone.Qwen25VLBackbone._prefill_execution_plan(model, logical_rows)
    logical, max_length, budget, kwargs = calls[0]
    assert logical == logical_rows
    assert max_length == model.cache.max_seq_len
    assert budget == expected_budget
    assert kwargs["exact_chunk_size"] == expected_chunk
    assert kwargs["padding_rows"] == expected_padding
    assert kwargs["execution_owner"] is model
    assert kwargs["execution_native"] == ("causal_decoder", 7)
    assert execution_rows == logical_rows + (16 if expected_padding == "auto" else expected_padding)
    assert chunk == (expected_chunk or 128)
    assert actual is result
