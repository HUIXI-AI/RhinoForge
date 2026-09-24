"""Public forward planning must work without private diagnostic API bindings."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_text_prefill_keeps_retained_owner_without_debug_api(monkeypatch):
    from rpu_backend.adapters.qwen3_5 import text

    assert not hasattr(torch.rpu, "get_debug_export")
    calls = []
    retained = SimpleNamespace(
        is_frozen=lambda: False,
        capture=lambda sig: calls.append(("capture", sig.op_id)) or nullcontext(),
        evict=lambda sig: calls.append(("evict", sig)),
    )
    state = SimpleNamespace(
        handle=1, num_layers=24, hidden_size=2048, chunk_envelope=(8192, 640),
        exact_chunk_size=None, padding_rows="auto", padding_budget=0,
        prefill_graph_cache=retained, public_retained_prefill=True,
        rotary_dim=64, mrope_section=None,
    )
    owner = SimpleNamespace(_rpu_qwen3_5=state)
    cache = SimpleNamespace(position=0, max_seq_len=1024, allocated_max_seq_len=1024)
    plan = SimpleNamespace(as_dict=lambda **kwargs: {"logical_len": 32},
                           graph_key_words=lambda: (11, 12))

    def native_plan(length, limit, budget, **kwargs):
        assert (length, limit, budget) == (32, 1024, 0)
        assert kwargs["graph_cache"] is retained
        assert kwargs["graph_mode"] == "RETAINED_CACHE"
        kwargs["plan_result_sink"](plan)
        return 32, 32

    monkeypatch.setattr(text, "_plan_prefill_execution", native_plan)
    result = text._preflight_qwen3_5_text(owner, cache, 32)
    assert result == (32, 32, plan) and state.last_a6_plan == {"logical_len": 32}
    with text._prepare_qwen35_prefill_capture(
        state, seq_len=32, real_len=32, planned_chunk_size=32, plan=plan,
    ):
        pass
    assert calls == [("capture", "rpu_qwen3_5_prefill")]
    assert state.prefill_graph_cache is retained


@pytest.mark.parametrize("graph_disabled", [False, True])
def test_vision_planning_preserves_explicit_graph_mode_without_debug_api(monkeypatch, graph_disabled):
    from rpu_backend.adapters.qwen3_5 import vision
    from rpu_backend.runtime import decoder

    assert not hasattr(torch.rpu, "get_debug_export")
    retained, plan = object(), object()
    owner = SimpleNamespace(_rpu_vision_graph_disable=graph_disabled,
                            _rpu_vision_kv_cache=SimpleNamespace(max_seq_len=1024),
                            _rpu_vision_graph_cache=retained)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_resolve_prefill_domain", lambda *args: (), raising=False)

    def native_plan(length, limit, budget, **kwargs):
        assert (length, limit, budget) == (16, 1024, 0)
        assert kwargs["graph_cache"] is (None if graph_disabled else retained)
        assert kwargs["plan_signature"] == (1, 1, False, graph_disabled)
        kwargs["plan_result_sink"](plan)

    monkeypatch.setattr(decoder, "plan_bounded_prefill_execution", native_plan)
    assert vision._plan_qwen3_5_vision_group(owner, 1, 16, group_id=0) is plan


def test_nextdit_native_guard_needs_only_a_live_owner():
    from rpu_backend.adapters.internvla_n1.nextdit import NextDiTRPURuntime

    assert not hasattr(torch.rpu, "get_debug_export")
    owner = SimpleNamespace(_handle=1, _collect_layer_outputs=True, _debug_export=False)
    NextDiTRPURuntime._require_native(owner)
    owner._handle = None
    with pytest.raises(RuntimeError, match="destroyed"):
        NextDiTRPURuntime._require_native(owner)


def test_production_has_no_calls_to_unavailable_debug_bindings():
    root = Path(__file__).resolve().parents[1] / "python/rpu_backend"
    calls = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"get_debug_export", "internvla_nextdit_set_debug_export"}:
                    calls.append((str(path.relative_to(root)), node.lineno))
    assert calls == []
