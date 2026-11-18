"""Board-free contracts for the single shared M-RoPE installer mechanism."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
DECODER = ROOT / "python/rpu_backend/runtime/decoder.py"


@pytest.fixture
def shared(monkeypatch):
    from rpu_backend.api import _execution
    from rpu_backend.runtime import decoder
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    tree = ast.parse(DECODER.read_text())
    names = {"build_mrope_cos_sin_tables", "install_mrope_text_decoder_for_rpu", "pad_mrope_prefill_inputs"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"torch": torch, "Any": Any,
        "_new_decoder_graph_cache": lambda topology, *, max_entries=None: Cache(),
        "_validate_mrope_decoder_topology": decoder._validate_mrope_decoder_topology}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(DECODER), "exec"), namespace)
    events = []
    class Cache:
        def clear(self):
            events.append("clear-new")
    backend = ModuleType("rpu_backend")
    backend.graph = SimpleNamespace(GraphCache=Cache)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    namespace["build_mrope_cos_sin_tables"] = lambda *args, **kw: ("cos", "sin")
    return namespace, events


@pytest.mark.parametrize("interrupt", [False, True])
def test_shared_install_preserves_transaction_and_explicit_certificate_authority(shared, interrupt):
    namespace, events = shared
    old_cache = SimpleNamespace(clear=lambda: events.append("clear-old"))
    model = SimpleNamespace(config=SimpleNamespace(
        rope_scaling={"mrope_section": [24, 20, 20]}, num_hidden_layers=28, hidden_size=2048),
        _rpu_text_graph_cache=old_cache, _rpu_text_num_layers=7,
        _rpu_text_hidden_size=8, _rpu_text_deepstack_hash=9)
    original = dict(vars(model))
    envelope = object()
    def install(value, **kwargs):
        assert value is model and kwargs["arch"] == "bound-arch"
        assert kwargs["chunk_envelope_for"] is envelope
        assert kwargs["deepstack_lang_layers"] == [0, 1, 2]
        assert kwargs["cos_sin"] == ("cos", "sin")
        assert kwargs["logical_vocab_size"] == 151936
        if interrupt:
            raise KeyboardInterrupt("publication interrupted")
        return 17
    namespace["_install_causal_decoder_forward"] = install
    call = lambda: namespace["install_mrope_text_decoder_for_rpu"](
        model, arch="bound-arch", chunk_envelope_for=envelope,
        logical_vocab_size=151936,
        vision_config=SimpleNamespace(deepstack_visual_indexes=[5, 11, 17]))
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            call()
        # Native retirement (including both caches) belongs to the shared
        # installer. The wrapper must not retry graph clear after it fails.
        assert vars(model) == original and events == []
    else:
        assert call() == 17
        assert model._rpu_text_graph_cache is not old_cache
        assert model._rpu_text_num_layers == 28 and events == []


def test_shared_rollback_does_not_reenter_rejecting_attribute_hook(shared):
    namespace, events = shared
    class Model(SimpleNamespace):
        def __setattr__(self, name, value):
            raise RuntimeError("reject publication")
        def __delattr__(self, name):
            pytest.fail("rollback must not call model delete hook")
    model = Model(config=SimpleNamespace(rope_scaling={"mrope_section": [1, 1, 1]},
                                        num_hidden_layers=28, hidden_size=2048),
                  _rpu_text_num_layers=7)
    original = dict(vars(model))
    namespace["_install_causal_decoder_forward"] = lambda *a, **kw: pytest.fail("too late")
    with pytest.raises(RuntimeError, match="reject publication"):
        namespace["install_mrope_text_decoder_for_rpu"](model, arch="bound", chunk_envelope_for=object())
    assert vars(model) == original and events == []


def test_qwen_wrapper_keeps_its_existing_profile_guard(shared):
    namespace, _ = shared
    path = ROOT / "python/rpu_backend/adapters/qwen3_vl/text.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "install_qwen3_vl_text_for_rpu")
    seen = []
    namespace.update(QWEN3_VL_TEXT_ARCH="qwen3_vl_text",
        _chunk_envelope_for_execution=lambda *args: seen.append(args) or (336, 320),
        install_mrope_text_decoder_for_rpu=lambda model, **kwargs: kwargs)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    request = {"prefill": {"chunk_size": 64}}
    result = namespace[node.name](object(), execution_config=request)
    assert result["chunk_envelope_for"]("qwen3_vl_text", 28, 2048) == (336, 320)
    assert seen == [("qwen3_vl_text", 28, 2048, request)]
    assert result["execution_config"] is request
