"""Shared CPU settings: cold validation, aliases and effective runner state."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from _example_toml import load_example_toml


ROOT = Path(__file__).resolve().parents[1]
CPU = vars(importlib.import_module("examples.common.cpu"))


def load(name, relative):
    if relative.startswith("examples/"):
        return importlib.import_module(relative[:-3].replace("/", "."))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("options,expected", [
    (None, {"level": "ERROR", "summary": True}),
    ({}, {"level": "ERROR", "summary": True}),
    ({"summary": False}, {"level": "ERROR", "summary": False}),
    *[({"level": level}, {"level": level, "summary": True})
      for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")],
])
def test_logging_defaults_and_levels_do_not_mutate_config(options, expected):
    common = load("example_config_test", "examples/common/config.py")
    config = {} if options is None else {"logging": options}
    before = deepcopy(config)
    result = common.validate_logging_config(config)
    assert result == expected
    assert config == before
    result["summary"] = not result["summary"]
    assert config == before


@pytest.mark.parametrize("options,match", [
    (None, r"\[logging\]"), (False, r"\[logging\]"),
    ([], r"\[logging\]"), ("ERROR", r"\[logging\]"),
    ({"file": "diagnostics.log"}, r"\[logging\]"),
    ({"summaries": True}, r"\[logging\]"),
    *[({"level": value}, r"\[logging\].level")
      for value in (None, 40, True, [], "error", "WARN", "", " ERROR ")],
    *[({"summary": value}, r"\[logging\].summary")
      for value in (None, 0, 1, "true", [], {})],
])
def test_logging_rejects_unknown_fields_and_ambiguous_types(options, match):
    common = load("example_config_test", "examples/common/config.py")
    with pytest.raises(common.ConfigError, match=match):
        common.validate_logging_config({"logging": options})


def test_logging_is_validated_by_public_and_optimized_entries_without_runtime(tmp_path):
    # One fixture per strict top-level schema, plus Rhino's factory fragment.
    fixtures = [
        ("qwen3/text/0_6b/fp16.toml", None),
        ("pi05/h50/fp16.toml", "pi05.optimized"),
        ("lingbot/v2/fp16.toml", "lingbot2.optimized"),
        ("g05/fp16.toml", "g05.optimized"),
        ("hy_vla/fp16.toml", "hy_vla.optimized"),
        ("wall_oss/base/3cam/224/fp16.toml", "wall_oss.base"),
        ("wall_oss/sorting/fp16.toml", "wall_oss.sorting"),
        ("rhinovla/v3/fp16.toml", None),
    ]
    files = []
    for index, (relative, runner) in enumerate(fixtures):
        path = tmp_path / f"config-{index}.toml"
        path.write_text((ROOT / "examples/configs" / relative).read_text()
                        + '\n[logging]\nlevel = "CRITICAL"\nsummary = false\n')
        files.append((str(path), runner))
    code = '''
from copy import deepcopy
import importlib.abc, importlib, json, pathlib, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'rpu_backend'}:
            raise AssertionError('configuration check imported ' + fullname)
sys.meta_path.insert(0, NoRuntime())
sys.path.insert(0, sys.argv[1])
from examples.common.config import ConfigError, load_config
for filename, runner in json.loads(sys.argv[2]):
    path = pathlib.Path(filename)
    config = load_config(path)
    assert config['logging'] == {'level': 'CRITICAL', 'summary': False}
    if runner:
        validate = importlib.import_module('examples.runners.' + runner).validate_example_config
        before = deepcopy(config)
        validate(config)
        assert config == before
        config['logging']['summary'] = 1
        try:
            validate(config)
        except ValueError as exc:
            assert '[logging].summary' in str(exc), str(exc)
        else:
            raise AssertionError(runner + ' bypassed logging validation')
    path.write_text(path.read_text().replace('summary = false', 'summary = 1'))
    try:
        load_config(path)
    except ConfigError as exc:
        assert '[logging].summary' in str(exc), str(exc)
    else:
        raise AssertionError(filename + ' accepted a nonboolean summary')
'''
    checked = subprocess.run(
        [sys.executable, "-S", "-c", code, str(ROOT), json.dumps(files)],
        capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.parametrize("value", [0, -1, True, "8", 2.5, None])
@pytest.mark.parametrize("section,key", [("cpu", "threads"), ("run", "cpu_threads"),
                                          ("run", "torch_num_threads")])
def test_thread_values_are_positive_integers(value, section, key):
    with pytest.raises(ValueError, match="positive integer"):
        CPU["resolve_cpu_threads"]({section: {key: value}})


def test_aliases_preserve_defaults_and_reject_conflicts_without_mutation():
    assert CPU["resolve_cpu_threads"]({}) is None
    assert CPU["resolve_cpu_threads"]({}, default=8) == 8
    config = {"cpu": {"threads": 3}, "run": {"cpu_threads": 3, "torch_num_threads": 3}}
    before = deepcopy(config)
    assert CPU["resolve_cpu_threads"](config) == 3 and config == before
    config["run"]["cpu_threads"] = 8
    with pytest.raises(ValueError, match="conflicting CPU thread settings"):
        CPU["resolve_cpu_threads"](config)
    for cpu in ([], 4, {"cores": 8}, {"threads": 8, "affinity": [0]}):
        with pytest.raises(ValueError, match=r"\[cpu\]"):
            CPU["resolve_cpu_threads"]({"cpu": cpu})


@pytest.mark.parametrize("relative", [
        "qwen3/text/0_6b/fp16.toml", "pi05/h50/fp16.toml",
        "wall_oss/sorting/fp16.toml", "wall_oss/base/3cam/224/fp16.toml", "hy_vla/fp16.toml",
        "g05/fp16.toml", "g05/w8a16_action_nvfp4.toml",
])
def test_public_cpu_config_checks_without_torch_backend_or_assets(tmp_path, relative):
    source = (ROOT / "examples/configs" / relative).read_text()
    source = re.sub(r"(?m)^(?:cpu_threads|torch_num_threads)\s*=.*\n", "", source)
    source = re.sub(r"(?ms)^\[cpu\]\s*\n.*?(?=^\[|\Z)", "", source)
    path = tmp_path / "config.toml"
    path.write_text(source + "\n[cpu]\nthreads = 3\n")
    code = '''
import importlib.abc, pathlib, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'rpu_backend'}:
            raise AssertionError('configuration check imported ' + fullname)
sys.meta_path.insert(0, NoRuntime())
sys.path.insert(0, sys.argv[1])
from examples.common.config import load_config, cpu_runtime
config = load_config(pathlib.Path(sys.argv[2]))
assert cpu_runtime().resolve_cpu_threads(config) == 3
'''
    checked = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT), str(path)],
                             capture_output=True, text=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_effective_threads_and_environment_are_set_before_first_torch_import():
    code = '''
import importlib.abc, json, os, runpy, sys
class ColdImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch':
            assert os.environ['OMP_NUM_THREADS'] == os.environ['MKL_NUM_THREADS'] == '3'
        if fullname.split('.')[0] == 'rpu_backend':
            raise AssertionError('CPU helper imported backend')
sys.meta_path.insert(0, ColdImport())
cpu = runpy.run_path(sys.argv[1])
assert 'torch' not in sys.modules
report = cpu['configure_cpu_threads'](3)
import torch
assert torch.get_num_threads() == report['torch_num_threads'] == 3
assert report['requested_threads'] == 3
print(json.dumps(report))
'''
    env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run([sys.executable, "-c", code, str(ROOT / "examples/common/cpu.py")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["OMP_NUM_THREADS"] == "3"


def test_shared_runner_applies_threads_before_prepare_and_reports_effective_values(monkeypatch, tmp_path):
    import torch
    runner = load("cpu_public_runner", "examples/common/measurement.py")
    performance = load("cpu_performance", "python/rpu_backend/runtime/performance.py")
    common = importlib.import_module("examples.common.config")
    monkeypatch.setattr(common, "cpu_runtime", lambda: SimpleNamespace(**CPU))
    monkeypatch.setattr(common, "example_profiles", lambda: {"test": {"id": "test", "support": {"status": "SUPPORTED"}}})
    for name, module in (("rpu_backend", ModuleType("rpu_backend")),
                         ("rpu_backend.runtime.performance", performance)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    previous = torch.get_num_threads()
    def prepare(config):
        assert torch.get_num_threads() == 3
        assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "3"
        return lambda: torch.ones(1), SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(runner, "_prepare", prepare)
    monkeypatch.setattr(runner, "_provenance", lambda config: {})
    config = {"example": {"profile_id": "test", "target": "qwen3"}, "input": {},
              "cpu": {"threads": 3}, "run": {"warmup": 0, "runs": 1, "output_dir": str(tmp_path)}}
    try:
        assert runner.run(config) == 0
        report = json.loads(next(tmp_path.glob("*/report.json")).read_text())
        assert report["cpu_runtime"]["torch_num_threads"] == 3
        assert report["cpu_runtime"]["requested_threads"] == 3
        assert report["config"]["cpu"] == {"threads": 3}
    finally:
        torch.set_num_threads(previous)


def test_wall_base_passes_public_threads_and_rejects_conflicts_before_engine(monkeypatch):
    import tomllib
    runner = load("cpu_wall_base", "examples/runners/wall_oss/base.py")
    config = load_example_toml(ROOT / "examples/configs/wall_oss/base/3cam/224/fp16.toml")
    config["run"].pop("cpu_threads", None)
    config["cpu"] = {"threads": 3}
    argv = runner.example_argv(config)
    assert argv[argv.index("--cpu-threads") + 1] == "3"
    config["run"]["cpu_threads"] = 8
    monkeypatch.setattr(runner, "validation_runner", lambda: pytest.fail("engine imported"))
    with pytest.raises(ValueError, match="conflicting CPU"):
        runner.run_example_config(config)


def test_wall_engine_uses_shared_cpu_settings_before_backend(monkeypatch, tmp_path):
    import torch
    engine = load("cpu_wall_engine", "examples/runners/wall_oss/base_engine.py")
    args = engine.arguments(["--mode", "candidate", "--precision", "fp16", "--fp16", str(tmp_path),
                             "--cpu-threads", "3", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(engine, "cold_environment", lambda args: {})
    backend, api = ModuleType("rpu_backend"), ModuleType("rpu_backend.api")
    backend.__file__ = "/deliberately-wrong-checkout/__init__.py"
    api.WallOssPolicy = object()
    monkeypatch.setitem(sys.modules, "rpu_backend", backend)
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)
    monkeypatch.setitem(sys.modules, "transformers", ModuleType("transformers"))
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    previous = torch.get_num_threads()
    try:
        # Deliberately stops before model loading; CPU state is already applied
        # and preserved in the failure report by the real engine entry.
        assert engine.main(args) == 1
        report = json.loads((tmp_path / "report.json").read_text())
        assert "wrong source checkout" in report["error"]
        assert report["cpu_runtime"]["torch_num_threads"] == torch.get_num_threads() == 3
        assert report["cpu_runtime"]["OMP_NUM_THREADS"] == "3"
    finally:
        torch.set_num_threads(previous)


def test_g05_aliases_reject_conflicts_before_cold_library_binding(monkeypatch):
    import tomllib
    runner = load("cpu_g05_validation", "examples/runners/g05/optimized.py")
    config = load_example_toml(ROOT / "examples/configs/g05/fp16.toml")
    config["cpu"] = {"threads": 3}
    config["run"].update(cpu_threads=3, torch_num_threads=3)
    runner.validate_example_config(config)
    config["run"]["cpu_threads"] = 8
    monkeypatch.setattr(runner, "library_helpers", lambda: pytest.fail("cold library work started"))
    with pytest.raises(ValueError, match="conflicting CPU thread settings"):
        runner.run_example_config(config)
    for invalid in ([], {"cores": 8}, {"threads": True}):
        config["run"].pop("cpu_threads")
        config["cpu"] = invalid
        with pytest.raises(ValueError, match=r"\[cpu\]"):
            runner.validate_example_config(config)
        config["run"]["cpu_threads"] = 8


@pytest.mark.parametrize("multiple", [False, True])
def test_g05_cold_cpu_binding_precedes_torch_backend_and_survives_failure_report(tmp_path, multiple):
    code = r'''
import importlib.abc, importlib.util, json, os, pathlib, sys, tomllib
root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(root))
bootstrap = r"""
from examples.runners.g05 import optimized as runner
from examples.common import measurement as _runner
_runner._provenance = lambda config: {}
events = []
def record(event):
    events.append(event)
    (out / f'imports-{os.getpid()}.json').write_text(json.dumps(events))
class ColdImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch':
            assert events == ['cold']
            assert os.environ['OMP_NUM_THREADS'] == os.environ['MKL_NUM_THREADS'] == '3'
            record('torch')
        if fullname == 'rpu_backend':
            assert events == ['cold', 'torch']
            assert sys.modules['torch'].get_num_threads() == 3
            record('backend')
            raise RuntimeError('stop before model assets')
sys.meta_path.insert(0, ColdImport())
def bind(env):
    record('cold')
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
runner.library_helpers = lambda: dict(
    validate_libraries=lambda table: None, bind_libraries=lambda config: ({}, {}),
    bind_cold_environment=bind)
"""
exec(bootstrap)
from examples.common.config import normalize_config
config = normalize_config(tomllib.loads((root / 'examples/configs/g05/fp16.toml').read_text()))
config['cpu']['threads'] = 3
config['run']['output_dir'] = str(out)
multiple = sys.argv[3] == 'True'
if multiple:
    config['input'].pop('text_tokens')
    config['input']['text_lengths'] = [32, 64]
    # Install the same import observer in the real fresh child. Parent-local
    # monkeypatches do not survive exec; keep the production stdin/config,
    # process isolation and report handling intact.
    original_run = runner.subprocess.run
    def observed_child(command, **kwargs):
        # The report's chip_info probe runs where it is installed; only the row child is observed.
        if '_run_single_example_config' not in command[-1]:
            return original_run(command, **kwargs)
        assert command[1:3] == ['-B', '-c']
        assert '_run_single_example_config' in command[3]
        prefix = ('import importlib.abc, json, os, pathlib, sys; '
                  f'root = pathlib.Path({str(root)!r}); out = pathlib.Path({str(out)!r}); '
                  'sys.path.insert(0, str(root));\n')
        return original_run([*command[:3], prefix + bootstrap + '\n' + command[3]], **kwargs)
    runner.subprocess.run = observed_child
else:
    config['input']['text_tokens'] = 32
try:
    runner.run_example_config(config)
except RuntimeError as error:
    expected = 'G05 T32: RuntimeError: stop before model assets' if multiple else 'stop before model assets'
    assert str(error) == expected
else:
    raise AssertionError('backend import was not intercepted')
report = json.loads((out / 'report.json').read_text())
assert report['status'] == 'FAILED'
assert report['cpu_runtime']['requested_threads'] == report['cpu_runtime']['torch_num_threads'] == 3
assert report['cpu_runtime']['OMP_NUM_THREADS'] == report['cpu_runtime']['MKL_NUM_THREADS'] == '3'
if multiple:
    assert list(report['rows']) == ['32']
    assert report['rows']['32']['cpu_runtime'] == report['cpu_runtime']
    assert report['rows']['32']['process_exit_code'] != 0
    assert not (out / 'text_64').exists()
    assert events == ['cold', 'torch']
    children = list(out.glob('imports-*.json'))
    children = [path for path in children if path.name != f'imports-{os.getpid()}.json']
    assert len(children) == 1
    assert json.loads(children[0].read_text()) == ['cold', 'torch', 'backend']
else:
    assert events == ['cold', 'torch', 'backend']
print(json.dumps(report['cpu_runtime']))
'''
    env = dict(os.environ, OMP_NUM_THREADS="11", MKL_NUM_THREADS="11",
               TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    result = subprocess.run([sys.executable, "-c", code, str(ROOT), str(tmp_path), str(multiple)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["requested_threads"] == 3
