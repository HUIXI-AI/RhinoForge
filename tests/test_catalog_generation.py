"""CPU checks for shared example measurement; never initialize the backend."""
import contextlib
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import subprocess
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]



def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def greedy_api(monkeypatch):
    generation = load("example_generation_test", "python/rpu_backend/api/generation.py")
    api = ModuleType("rpu_backend.api")
    api.greedy_token_ids = Mock(wraps=generation.greedy_token_ids)
    monkeypatch.setitem(sys.modules, "rpu_backend", ModuleType("rpu_backend"))
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)
    return api


def test_generation_reuses_cache_but_resets_each_request(greedy_api):
    runner = load("example_text_test", "examples/_text_vision_runners.py")
    cache = Mock()
    model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[[0., 2., 1.]]])))
    for _ in range(2):
        out = runner._generate(model, cache, torch.tensor([[0, 2]]), 3)
        assert out["token_ids"].tolist() == [[1, 1, 1, 1]]
    assert cache.reset.call_count == 2
    assert model.call_count == 8
    assert greedy_api.greedy_token_ids.call_count == 8
    assert all(call.kwargs["past_key_values"] is cache for call in model.call_args_list)
    for index, call in enumerate(model.call_args_list):
        if index % 4 == 0:
            assert call.kwargs["logits_to_keep"] == 1
            assert call.kwargs["input_ids"].shape == (1, 2)
        else:
            assert "logits_to_keep" not in call.kwargs
            assert call.kwargs["input_ids"].shape == (1, 1)


@pytest.mark.parametrize("sequence, generation_eos, config_eos, expected", [
    ([2], None, 2, [2]),
    ([1, 3], [2, 3], None, [1, 3]),
    ([2, 3], 3, 2, [2, 3]),
])
def test_generation_stops_on_checkpoint_eos_without_another_forward(
        sequence, generation_eos, config_eos, expected, greedy_api):
    runner = load("example_eos_test", "examples/_text_vision_runners.py")
    cache = Mock()
    outputs = [SimpleNamespace(logits=torch.nn.functional.one_hot(
        torch.tensor([[token]]), num_classes=4).float()) for token in sequence]
    model = Mock(side_effect=outputs)
    model.config = SimpleNamespace(eos_token_id=config_eos)
    model.generation_config = SimpleNamespace(eos_token_id=generation_eos)
    result = runner._generate(model, cache, torch.tensor([[0, 1]]), 31)
    assert result["token_ids"].tolist() == [expected]
    assert model.call_count == len(expected)
    assert greedy_api.greedy_token_ids.call_count == len(expected)
    assert torch.equal(result["logits"], outputs[-1].logits)
    cache.reset.assert_called_once_with()


def test_fixed_decode_ignores_eos_and_keeps_owned_tokens_and_phase_timings(monkeypatch, greedy_api):
    runner = load("example_fixed_decode_test", "examples/_text_vision_runners.py")
    # A reusable output buffer models the backend's shared logits storage.
    buffer = torch.zeros(1, 1, 4)
    sequence = iter([2, 1, 3])

    def forward(**_kwargs):
        buffer.zero_()
        buffer[0, 0, next(sequence)] = 5
        return SimpleNamespace(logits=buffer)

    model = Mock(side_effect=forward)
    model.config = SimpleNamespace(eos_token_id=2)
    model.generation_config = None
    ticks = iter([0., .5, 1., 1.1, 2., 2.2])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    out = runner._generate(model, Mock(), torch.tensor([[0, 1, 0]]), 2,
                           stop_on_eos=False)
    assert out["token_ids"].tolist() == [[2, 1, 3]]
    assert out["first_logits"].argmax().item() == 2
    assert model.call_count == 3
    stats = out["generation"]
    assert stats["prefill_tokens"] == 3 and stats["decode_calls"] == 2
    assert stats["prefill_ms"] == 500.
    assert stats["decode_ms"] == pytest.approx(300.)
    assert stats["prefill_tokens_per_second"] == 6.
    assert stats["decode_tokens_per_second"] == pytest.approx(2 / .3)


@pytest.mark.parametrize("fail", [None, "inference", "cleanup", "inference_cleanup"])
def test_example_times_actual_inference_and_records_failures(
        monkeypatch, tmp_path, fail):
    runner = load("public_runner_test", "examples/_runner.py")
    performance = load("performance_runner_test", "python/rpu_backend/runtime/performance.py")
    common = ModuleType("_common")
    common.config_metadata = lambda config: config["example"]
    monkeypatch.setitem(sys.modules, "_common", common)
    monkeypatch.setitem(sys.modules, "rpu_backend", ModuleType("rpu_backend"))
    monkeypatch.setitem(sys.modules, "rpu_backend.runtime.performance", performance)
    # Match the live backend: RPU calls are synchronous and this namespace
    # intentionally has no CUDA-style synchronize method.
    monkeypatch.setattr(torch, "rpu", SimpleNamespace(
        hw_perf_trace=lambda *_a, **_kw: contextlib.nullcontext(),
        _is_in_bad_fork=lambda: False, manual_seed_all=lambda seed: None), raising=False)
    output = torch.tensor([1., 2.])
    samples = [{"logits": output, "generation": {
        "prefill_tokens": 64, "decode_calls": 2,
        "prefill_ms": ms, "decode_ms": ms,
        "prefill_tokens_per_second": 64000 / ms,
        "decode_tokens_per_second": 2000 / ms,
    }} for ms in (10., 100., 300.)]
    infer = Mock(side_effect=samples)
    owner = SimpleNamespace(close=Mock())
    if fail in {"inference", "inference_cleanup"}:
        infer.side_effect = RuntimeError("forward failed")
    if fail in {"cleanup", "inference_cleanup"}:
        owner.close.side_effect = RuntimeError("close failed")
    monkeypatch.setattr(runner, "_prepare", lambda config: (infer, owner))
    config = {"example": {"profile_id": "test.profile", "target": "qwen3"}, "input": {},
              "run": {"output_dir": str(tmp_path), "warmup": 1, "warmup_decode_steps": 2,
                      "runs": 2, "hwperf": fail == "hwperf", "inference_mode": False}}
    if fail:
        with pytest.raises(RuntimeError):
            runner.run(config)
    else:
        assert runner.run(config) == 0
        assert infer.call_count == 3
        assert infer.call_args_list[0].kwargs == {"decode_steps": 2}
        assert all(not call.kwargs for call in infer.call_args_list[1:])
    report_path, = tmp_path.glob("*/report.json")
    report = json.loads(report_path.read_text())
    assert report["status"] == ("FAIL" if fail else "PASS")
    assert report["correctness"] == "not_run"
    assert report["formal_certification"] is False
    assert "support" not in report
    assert "evaluation_only" not in report["config"]["example"]
    assert report["cpu_runtime"]["inference_mode"] is False
    owner.close.assert_called_once()
    assert report_path.with_name("report.md").exists() is (fail is None)
    if fail == "inference_cleanup":
        assert report["error"] == "RuntimeError: forward failed"
        assert report["cleanup_error"] == "RuntimeError: close failed"
    if not fail:
        assert len(report["wall_ms_runs"]) == 2
        assert torch.equal(torch.load(report_path.with_name("output.pt"), weights_only=True)["logits"], output)
        assert len(report["generation_runs"]) == 2
        # Match the report's throughput from mean latency, not mean token/s.
        assert report["generation_summary"]["prefill_tokens_per_second"] == 320.
        assert report["generation_summary"]["decode_tokens_per_second"] == 10.


def test_cleanup_failure_never_prints_success(monkeypatch, tmp_path, capsys):
    test_example_times_actual_inference_and_records_failures(
        monkeypatch, tmp_path, "cleanup")
    assert "inference PASS:" not in capsys.readouterr().out


def test_returned_output_has_independent_storage_and_rejects_nonfinite():
    runner = load("public_runner_output_test", "examples/_runner.py")
    original = torch.tensor([1.])
    copied = runner._cpu_output({"action": original})
    original.fill_(2.)
    assert copied["action"].item() == 1.
    with pytest.raises(ValueError, match="NaN/Inf"):
        runner._cpu_output(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="NaN/Inf"):
        runner._cpu_output({"score": float("inf")})
