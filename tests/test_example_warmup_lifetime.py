"""Warm BUILD/REPLAY and retire outputs before the next measurement."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _runner(monkeypatch):
    spec = importlib.util.spec_from_file_location("warmup_runner", ROOT / "examples/_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    common = ModuleType("_common")
    common.config_metadata = lambda config: config["example"]
    monkeypatch.setitem(sys.modules, "_common", common)
    monkeypatch.setitem(sys.modules, "rpu_backend", ModuleType("rpu_backend"))
    performance = ModuleType("rpu_backend.runtime.performance")
    performance._profile_ctx = lambda _: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "rpu_backend.runtime.performance", performance)
    monkeypatch.setattr(runner, "_provenance", lambda _: {})
    monkeypatch.setattr(torch, "manual_seed", lambda _: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    return runner


@pytest.mark.parametrize("target", ["qwen3", "llama", "qwen3_5", "qwen3_vl", "dinov3",
                                   "gemma4", "wall_oss", "hy_vla", "qwen3_5_vision",
                                   "lingbot2", "halo", "rhinovla", "pi05"])
def test_default_warms_build_and_replay_and_releases_original_outputs(monkeypatch, tmp_path, target):
    _exercise(monkeypatch, tmp_path, target)


@pytest.mark.parametrize("warmup", [0, 1])
def test_explicit_cold_diagnostic_count_has_no_hidden_calls(monkeypatch, tmp_path, warmup):
    _exercise(monkeypatch, tmp_path, "rhinovla", warmup)


def _exercise(monkeypatch, tmp_path, target, warmup=None):
    runner = _runner(monkeypatch)
    clock = SimpleNamespace(seconds=0.0, active=False)
    refs, released, closed = [], [], []
    # Synthetic clock advances, unrelated to hardware measurements.
    durations = [0.500, 0.250, 0.030, 0.040, 0.020]

    def tick():
        clock.active = not clock.active
        return clock.seconds

    def infer():
        assert clock.active
        # A delayed release on assignment would keep the previous raw output
        # alive throughout this call, even though its CPU snapshot was saved.
        assert all(ref() is None for ref in refs)
        index = len(refs)
        clock.seconds += durations[index]
        value = torch.tensor([float(index)])
        refs.append(weakref.ref(value))
        weakref.finalize(value, lambda: released.append(clock.active))
        return value

    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=tick))
    monkeypatch.setattr(runner, "_prepare", lambda _: (
        infer, SimpleNamespace(close=lambda: closed.append(True))))
    options = {"runs": 3, "output_dir": str(tmp_path), "inference_mode": False}
    if warmup is not None:
        options["warmup"] = warmup
    config = {"example": {"profile_id": "test.warmup", "target": target},
              "input": {}, "run": options}
    assert runner.run(config) == 0
    count = 2 if warmup is None else warmup
    report_path, = tmp_path.glob("*/report.json")
    report = json.loads(report_path.read_text())
    assert report["warmup"] == count
    assert report["warmup_wall_ms_runs"] == pytest.approx([d * 1000 for d in durations[:count]])
    assert report["wall_ms_runs"] == pytest.approx([d * 1000 for d in durations[count:count+3]])
    assert len(refs) == count + 3
    assert released == [False] * (count + 3)
    assert closed == [True]
    assert torch.equal(torch.load(report_path.with_name("output.pt"), weights_only=True),
                       torch.tensor([float(count + 2)]))


def test_runnable_catalog_templates_include_replay_warmup():
    import tomllib

    for path in (ROOT / "examples/configs").rglob("*.toml"):
        config = tomllib.loads(path.read_text())
        if "example" in config or "pi05" in config:
            assert config["run"]["warmup"] >= 2, path
            assert config["run"].get("warmup_decode_steps", config["input"].get("decode_steps")) == config["input"].get("decode_steps"), path


@pytest.mark.parametrize("template", [
    "qwen3/text/0_6b/fp16.toml", "llama/3_2_1b/fp16.toml",
    "qwen3_5/text/2b/fp16.toml", "qwen3_vl/text/4b/fp16.toml",
    "qwen3_5/vl/2b/fp16.toml", "gemma4/e4b/fp16.toml",
])
@pytest.mark.parametrize("warmup", [None, 0, 1, 2, 3])
@pytest.mark.parametrize("warmup_decode_steps", [0, 3, 4])
def test_short_decode_warmup_is_only_an_explicit_cold_diagnostic(
        monkeypatch, tmp_path, template, warmup, warmup_decode_steps):
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    import _common

    source = (ROOT / "examples/configs" / template).read_text()
    source = source.replace("warmup = 2\n", "" if warmup is None else f"warmup = {warmup}\n")
    source = source.replace("[run]\n", f"[run]\nwarmup_decode_steps = {warmup_decode_steps}\n")
    path = tmp_path / "case.toml"
    path.write_text(source)
    if (warmup is None or warmup >= 2) and warmup_decode_steps != 4:
        with pytest.raises(_common.ConfigError, match="requires full decode warmup"):
            _common.load_config(path)
    else:
        config = _common.load_config(path)
        assert config["run"]["warmup_decode_steps"] == warmup_decode_steps
