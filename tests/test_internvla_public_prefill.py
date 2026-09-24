"""Public prefill requests use the shared planner without implicit shape recipes."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("logical_rows", [128, 480])
@pytest.mark.parametrize(
    "stage, expected_chunk, expected_padding",
    [({}, None, 0), ({"chunk_size": 64, "padding_rows": 16}, 64, 16)],
)
def test_prefill_planning_preserves_explicit_public_options(
    logical_rows, stage, expected_chunk, expected_padding
):
    source = (
        Path(__file__).resolve().parents[1]
        / "python/rpu_backend/adapters/internvla_n1/backbone.py"
    ).read_text()
    tree = ast.parse(source)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "Qwen25VLBackbone")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_prefill_execution_plan")
    cap = next(node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name)
                       and target.id == "_CERTIFIED_MAX_KV_LEN"
                       for target in node.targets))
    calls = []
    descriptor = object()
    result = SimpleNamespace(selected=SimpleNamespace(
        stage_tuple=SimpleNamespace(physical_descriptor=descriptor)))

    def planner(logical, max_length, budget, **kwargs):
        calls.append((logical, max_length, budget, kwargs))
        kwargs["plan_result_sink"](result)
        return logical + kwargs["padding_rows"], kwargs["exact_chunk_size"] or 128

    namespace = {"plan_bounded_prefill_execution": planner}
    extracted = ast.Module(body=[cap, method], type_ignores=[])
    exec(compile(extracted, "<public-prefill-method>", "exec"), namespace)
    model = SimpleNamespace(
        _handle=7, _rpu_execution={"prefill": stage},
        cache=SimpleNamespace(max_seq_len=4096), _graph_cache=object(),
    )
    execution_rows, chunk, actual = namespace[method.name](model, logical_rows)
    logical, max_length, budget, kwargs = calls[0]
    assert logical == logical_rows
    assert max_length == namespace["_CERTIFIED_MAX_KV_LEN"]
    assert budget == 0
    assert kwargs["exact_chunk_size"] == expected_chunk
    assert kwargs["padding_rows"] == expected_padding
    assert kwargs["execution_owner"] is model
    assert kwargs["execution_native"] == ("causal_decoder", 7)
    assert execution_rows == logical_rows + expected_padding
    assert chunk == (expected_chunk or 128)
    assert actual is result
