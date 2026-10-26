"""One Pi0.5 entry resolves physical choices before backend imports and load."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "examples/configs/pi05"


def load(relative):
    return importlib.import_module(relative.removeprefix("python/")[:-3].replace("/", "."))


@pytest.mark.parametrize("warmup", [0, 1, 2])
def test_direct_cli_requires_positive_warmup_and_keeps_measured_run_count(warmup, monkeypatch):
    runner = load("examples/runners/pi05/optimized.py")
    calls = []
    monkeypatch.setattr(runner, "run_profile", lambda args: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["run_pi05_optimized.py", "--precision", "fp16",
                                     "--checkpoint", "/checkpoint", "--batch", "/batch.pt",
                                     "--warmup", str(warmup), "--runs", "3"])
    if warmup == 0:
        with pytest.raises(SystemExit) as caught:
            runner.main()
        assert caught.value.code == 2
        assert calls == []
    else:
        assert runner.main() == 0
        assert len(calls) == 1
        assert calls[0].warmup == warmup and calls[0].runs == 3


def test_shipped_configs_resolve_physical_choices_and_registered_scope():
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    paths = set(CONFIGS.rglob("*.toml"))
    expected = {CONFIGS / f"{cameras}cam/{precision}.toml"
                for cameras in (2, 3)
                for precision in ("fp16", "w8a16", "w8a16_action_nvfp4",
                                  "w8a16_prefill_w8a8_action_nvfp4")}
    expected |= {CONFIGS / f"generic/3cam/{precision}_tp{cores}.toml"
                 for precision in ("fp16", "w8a16") for cores in (4, 6)}
    expected.add(CONFIGS / "3cam/w8a16_t160.toml")
    expected.add(CONFIGS / "3cam/w8a16_action_nvfp4_t160.toml")
    expected.add(CONFIGS / "3cam/w8a16_prefill_w8a8_action_nvfp4_t160.toml")
    assert paths == expected
    registered = set()
    for path in sorted(paths):
        config = common.load_config(path)
        args = runner.example_arguments(config)
        precision = config["pi05"]["precision"]
        assert args.precision == ("w4a16" if "nvfp4" in precision else precision)
        assert args.public_precision == precision
        assert args.prefill_w8a8 == (precision == "w8a16_prefill_w8a8_action_nvfp4")
        metadata = common.example_metadata(config)
        if metadata is not None:
            registered.add(metadata["profile_id"])
        execution, paired, chunk = runner.resolve_execution(args)
        if execution["model"]["num_cores"] == 8:
            assert paired and chunk == (256 * args.cameras + args.text_tokens) // 2
            for text_tokens in (32, 64, 96, 128, 160):
                candidate = deepcopy(config)
                candidate["input"]["text_tokens"] = text_tokens
                _, admitted, bucket = runner.resolve_execution(runner.example_arguments(candidate))
                assert admitted and bucket == (256 * args.cameras + text_tokens) // 2
        else:
            assert not paired
    required = {profile_id for profile_id, profile in common.planner_profiles().items()
                if profile["family"] == "pi05" and common.required_example(profile)}
    retired = "pi05-libero-finetuned-w4real-kvint8.w4a16-kvint8.vla"
    assert retired in required  # Its registered capability was not withdrawn.
    assert registered == required - {retired}


def test_source_distribution_entries_use_public_catalog_without_manifests(tmp_path):
    for relative in ("examples/common", "examples/runners/pi05", "examples/configs/pi05"):
        shutil.copytree(ROOT / relative, tmp_path / relative,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for filename in ("profiles.json", "config_names.json"):
        relative = Path("examples/configs") / filename
        shutil.copy2(ROOT / relative, tmp_path / relative)
    leaf = "python/rpu_backend/api/_execution.py"
    (tmp_path / leaf).parent.mkdir(parents=True)
    shutil.copy2(ROOT / leaf, tmp_path / leaf)
    script = """
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
class NoBackendImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'rpu_backend'):
            raise AssertionError('configuration metadata imported the backend')
sys.meta_path.insert(0, NoBackendImports())
from examples.common import config as common
from examples.runners.pi05 import optimized as runner

assert not Path('manifests').exists()
profiles = common.example_profiles()
seen = []
def capture(args, *, bind_libraries):
    assert bind_libraries is True
    seen.append(args)
runner.run_profile = capture
configs = [common.load_config(path) for path in sorted(Path('examples/configs/pi05').rglob('*.toml'))]
legacy = common.load_config(Path('examples/configs/pi05/3cam/fp16.toml'))
legacy['pi05']['precision'] = 'w4a16_kvint8'
legacy['input']['checkpoint'] = 'pi05-libero-finetuned-w4real-kvint8'
configs.append(legacy)
registered = set()
for config in configs:
    metadata = runner.registered_example(config)
    assert runner.run_example_config(config) == 0
    if metadata is None:
        assert seen[-1].registered_profile is None
    else:
        profile = profiles[metadata['profile_id']]
        assert seen[-1].registered_profile == {
            key: profile[key] for key in ('id', 'support', 'input_envelope')}
        registered.add(profile['id'])
assert registered == {key for key, profile in profiles.items()
                      if profile['family'] == 'pi05' and common.required_example(profile)}

# Missing public support metadata must fail before model execution.
path = Path('examples/configs/profiles.json')
catalog = json.loads(path.read_text())
profile_id = 'pi05-libero-finetuned.fp16.vla'
catalog['profiles'] = [row for row in catalog['profiles'] if row['id'] != profile_id]
path.write_text(json.dumps(catalog))
common.planner_profiles.cache_clear()
common.example_profiles.cache_clear()
before = len(seen)
try:
    runner.run_example_config(common.load_config(Path('examples/configs/pi05/3cam/fp16.toml')))
except KeyError as exc:
    assert exc.args == (profile_id,)
else:
    raise AssertionError('missing public profile reached execution')
assert len(seen) == before
assert 'torch' not in sys.modules and 'rpu_backend' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script], cwd=tmp_path,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("section,key,value", [
    ("pi05", "evaluation_only", False),
    ("pi05", "evaluation_only", True),
    ("pi05", "precision", "legacy_int4"),
    ("pi05", "precision", "w4a16"),
    ("pi05", "precision", "w8a16_w4a16"),
    ("pi05", "precision", "w8a16_w8a8_w4a16"),
    ("pi05", "num_cores", 4),
    ("input", "cameras", 1),
    ("input", "text_tokens", 80),
    ("input", "num_steps", 5),
    ("run", "warmup", -1),
    ("run", "runs", True),
    ("input", "batch", ""),
    ("input", "noise", 4),
])
def test_config_rejects_unmeasured_controls_and_ambiguous_precision(section, key, value):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = common.load_config(CONFIGS / "3cam/w8a16_prefill_w8a8_action_nvfp4.toml")
    config[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_example_config(config)


@pytest.mark.parametrize("change", [
    {"checkpoint": "/local/another-checkpoint"},
    {"cameras": 2},
    {"text_tokens": 64},
])
def test_custom_assets_and_geometries_do_not_inherit_registered_support(change):
    common = load("examples/common/config.py")
    config = common.load_config(CONFIGS / "3cam/fp16.toml")
    config["input"].update(change)
    assert common.example_metadata(config) is None


@pytest.mark.parametrize("execution", [
    {"prefill": {"chunk_size": 320}},
    {"prefill": {"padding_rows": 16}},
    {"components": {"language_model": {"prefill": {"padding_budget": 16}}}},
    {"vision": {"chunk_size": 256}},
])
def test_custom_tp8_plans_do_not_inherit_exact_c400_support(execution):
    common = load("examples/common/config.py")
    for name in ("fp16", "w8a16", "w4a16_kvint8"):
        config = common.load_config(CONFIGS / "3cam/fp16.toml")
        config["pi05"]["precision"] = name
        config["input"]["checkpoint"] = {
            "fp16": "pi05-libero-finetuned",
            "w8a16": "pi05-libero-finetuned-w8a16-vlm-expert",
            "w4a16_kvint8": "pi05-libero-finetuned-w4real-kvint8",
        }[name]
        assert common.example_metadata(config) is not None
        config["rpu_execution"] = execution
        assert common.example_metadata(config) is None


def test_public_entry_dispatches_one_engine_for_selected_config(monkeypatch, capsys):
    common = load("examples/common/config.py")
    path = CONFIGS / "2cam/w8a16_action_nvfp4.toml"
    default = CONFIGS / "3cam/w8a16.toml"
    monkeypatch.setattr(sys, "argv", ["pi05.py", "--config", str(path), "--check-config"])
    assert importlib.import_module("examples.common.cli").main() == 0
    assert "syntax/input envelope only" in capsys.readouterr().out
    runner = common._pi05_runner()
    seen = []
    monkeypatch.setattr(runner, "run_example_config", lambda config: seen.append(config) or 0)
    monkeypatch.setattr(importlib.import_module("examples.common.cli"), "_pi05_runner", lambda: runner)
    for selected in (path, default, CONFIGS / "generic/3cam/fp16_tp4.toml"):
        monkeypatch.setattr(sys, "argv", ["pi05.py", "--config", str(selected)])
        assert importlib.import_module("examples.common.cli").main() == 0
    assert [item["pi05"]["precision"] for item in seen] == ["w8a16_action_nvfp4", "w8a16", "fp16"]


def test_library_binding_follows_current_aliases_and_manifest(monkeypatch, tmp_path):
    runner = load("examples/runners/pi05/optimized.py")
    (tmp_path / "release").mkdir()
    artifacts, selected = {}, {}
    for key, role, alias, content in (
        ("RPU_KERNEL_LIB_PATH", "rhino_ops_main", "rhinoOpLib_current.ref", b"main payload"),
        ("RPU_KERNEL_LIB_PATH_EXPANSION", "rhino_ops_expansion", "rhinoExpansionOpLib_current.ref", b"expansion payload"),
    ):
        payload = tmp_path / alias
        payload.write_bytes(content)
        artifacts[role] = {"path": alias, "sha256": hashlib.sha256(content).hexdigest()}
        selected[key] = str(payload)
        monkeypatch.setenv(key, str(payload))
    (tmp_path / "release/runtime-dependencies.json").write_text(json.dumps({"artifacts": artifacts}))
    env, records = runner.profile_libraries(tmp_path)
    assert env == selected
    for key, role in (("RPU_KERNEL_LIB_PATH", "rhino_ops_main"), ("RPU_KERNEL_LIB_PATH_EXPANSION", "rhino_ops_expansion")):
        assert records[key] == {"path": env[key], "sha256": artifacts[role]["sha256"]}
    (tmp_path / "rhinoExpansionOpLib_current.ref").write_bytes(b"changed payload")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.profile_libraries(tmp_path)


def test_library_binding_is_explicit_and_rejects_inherited_payload(monkeypatch):
    runner = load("examples/runners/pi05/optimized.py")
    monkeypatch.delenv("RPU_KERNEL_LIB_PATH", raising=False)
    with pytest.raises(ValueError, match="set RPU_KERNEL_LIB_PATH"):
        runner.profile_libraries()
    env = {"RPU_KERNEL_LIB_PATH": "/path/to/main.ref", "RPU_KERNEL_LIB_PATH_EXPANSION": "/path/to/expansion.ref"}
    monkeypatch.delitem(sys.modules, "rpu_backend", raising=False)
    monkeypatch.setattr(runner.os, "environ", {key: value for key, value in runner.os.environ.items() if key not in env})
    monkeypatch.setenv("RPU_CHUNK_FORCE_UNSAFE", "0")
    runner.bind_cold_environment(env)
    monkeypatch.setenv("RPU_KERNEL_LIB_PATH", "/different.ref")
    with pytest.raises(ValueError, match="conflicting cold environment"):
        runner.bind_cold_environment(env)
    monkeypatch.setitem(sys.modules, "rpu_backend", ModuleType("rpu_backend"))
    with pytest.raises(ValueError, match="fresh process"):
        runner.bind_cold_environment({})


def test_stage_observers_restore_boundaries_when_inference_fails(monkeypatch):
    runner = load("examples/runners/pi05/optimized.py")
    pi05 = ModuleType("rpu_backend.adapters.pi05")
    adapters = ModuleType("rpu_backend.adapters")
    pi05.runtime = SimpleNamespace(_run_denoise_fused=lambda: 1)
    adapters.siglip = SimpleNamespace(_siglip_forward_images=lambda: 2)
    monkeypatch.setitem(sys.modules, "rpu_backend.adapters.pi05", pi05)
    monkeypatch.setitem(sys.modules, "rpu_backend.adapters", adapters)

    class Vlm:
        def _rpu_prefill_kv_only(self):
            return 3

    vlm = Vlm()
    policy = SimpleNamespace(_lerobot_policy=SimpleNamespace(model=SimpleNamespace(
        paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(model=SimpleNamespace(language_model=vlm))))))
    original = pi05.runtime._run_denoise_fused
    with pytest.raises(RuntimeError, match="forward failed"):
        with runner.stage_timings(policy) as phases:
            assert pi05.runtime._run_denoise_fused() == 1
            assert adapters.siglip._siglip_forward_images() == 2
            assert vlm._rpu_prefill_kv_only() == 3
            assert set(phases) == {"vision_ms", "prefill_ms", "denoise_ms"}
            raise RuntimeError("forward failed")
    assert pi05.runtime._run_denoise_fused is original
    assert "_rpu_prefill_kv_only" not in vars(vlm)


@pytest.mark.parametrize("key,value", [
    ("RPU_LINEAR_ACC32", "1"), ("RPU_PI05_PREFIX_MASK_CACHE", "0"),
    ("RPU_PI05_GEMMA_GRAPH", "0"), ("RPU_PI05_DENOISE_GRAPH", "0"),
    ("RPU_PI05_LOAD_PREFIX_EMBS", "/tmp/replacement.pt"),
])
def test_optimized_config_rejects_inherited_execution_and_input_overrides(monkeypatch, key, value):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    args = runner.example_arguments(common.load_config(CONFIGS / "3cam/fp16.toml"))
    monkeypatch.setattr(runner, "profile_libraries", lambda: ({}, {}))
    monkeypatch.delitem(sys.modules, "rpu_backend", raising=False)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        runner.run_profile(args, bind_libraries=True)


@pytest.mark.parametrize("mutation,error_role", [
    (None, None),
    ("missing_roles", "main"),
    ("missing_main", "main"),
    ("missing_expansion", "expansion"),
    ("combined_only", "main"),
    ("swapped_roles", "main"),
    ("main_hash", "main"),
    ("expansion_hash", "expansion"),
])
def test_actual_runtime_roles_must_match_selected_libraries(monkeypatch, mutation, error_role):
    runner = load("examples/runners/pi05/optimized.py")
    libraries = {"RPU_KERNEL_LIB_PATH": {"path": "/main.ref", "sha256": "main"},
                 "RPU_KERNEL_LIB_PATH_EXPANSION": {"path": "/pi05.ref", "sha256": "pi05"}}
    actual = {"extension": "/native.so", "oplib_roles": {"main": "/main.ref", "expansion": "/pi05.ref"},
              "oplib_payloads": {"/main.ref": {"sha256": "main"}, "/pi05.ref": {"sha256": "pi05"}}}
    backend = ModuleType("rpu_backend")
    backend._cpp_ext = SimpleNamespace(get_runtime_payload_paths=lambda: actual)
    monkeypatch.setitem(sys.modules, "rpu_backend", backend)
    if mutation is None:
        assert runner.runtime_payloads(libraries) is actual
        return
    if mutation == "missing_roles":
        del actual["oplib_roles"]
    elif mutation.startswith("missing_"):
        del actual["oplib_roles"][error_role]
    elif mutation == "combined_only":
        actual["oplib_roles"] = {"combined": "/main.ref"}
    elif mutation == "swapped_roles":
        actual["oplib_roles"] = {"main": "/pi05.ref", "expansion": "/main.ref"}
    else:
        path = actual["oplib_roles"][error_role]
        actual["oplib_payloads"][path]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match=f"loaded {error_role} library"):
        runner.runtime_payloads(libraries)


@pytest.mark.parametrize("precision,cameras,text_tokens,execution,enabled", [
    *[pytest.param(precision, cameras, 32, {}, True, id=f"{cameras}cam-{precision}")
      for cameras in (2, 3)
      for precision in ("w8a16_action_nvfp4", "w8a16_prefill_w8a8_action_nvfp4")],
    *[pytest.param(precision, cameras, 160, {}, "nvfp4" in precision,
                  id=f"T160-{cameras}cam-{precision}")
      for cameras in (2, 3)
      for precision in ("fp16", "w8a16", "w8a16_action_nvfp4", "w8a16_prefill_w8a8_action_nvfp4")],
    pytest.param("w8a16_action_nvfp4", 3, 160,
                 {"action": {"linear_acc32": True}}, False, id="T160-action-acc32"),
    pytest.param("w8a16_action_nvfp4", 3, 32,
                 {"action": {"linear_acc32": True}}, False, id="action-acc32"),
    pytest.param("w8a16_action_nvfp4", 3, 32,
                 {"action": {"linear_acc32": False}, "components": {
                     "action_expert": {"action": {"linear_acc32": True}}}},
                 False, id="component-selects-acc32"),
    pytest.param("w8a16_action_nvfp4", 3, 32,
                 {"action": {"linear_acc32": True}, "components": {
                     "action_expert": {"action": {"linear_acc32": False}}}},
                 True, id="component-selects-acc16"),
    pytest.param("w8a16_action_nvfp4", 3, 32,
                 {"vision": {"linear_acc32": True}, "prefill": {"linear_acc32": True}},
                 True, id="other-stages-acc32"),
    *[pytest.param("w8a16_action_nvfp4", 3, tokens, {}, False, id=f"T{tokens}")
      for tokens in (64, 96, 128)],
    *[pytest.param(precision, 3, 32, {}, False, id=precision)
      for precision in ("fp16", "w8a16", "w4a16_kvint8")],
    *[pytest.param("w8a16", 3, 32, {"model": {"num_cores": cores}}, False,
                  id=f"generic-tp{cores}") for cores in (4, 6)],
])
def test_nvfp4_acc16_fusion_resolves_at_cold_entry(
        monkeypatch, precision, cameras, text_tokens, execution, enabled):
    runner = load("examples/runners/pi05/optimized.py")
    config = {"pi05": {"precision": precision},
              "input": {"checkpoint": "/not-loaded/checkpoint", "batch": "/not-loaded/batch.pt",
                        "cameras": cameras, "text_tokens": text_tokens, "num_steps": 10},
              "run": {}, "rpu_execution": deepcopy(execution)}
    original = deepcopy(config)
    captured = []

    class StoppedBeforeLoad(Exception):
        pass

    def bind_environment(env):
        captured.append(dict(env))
        raise StoppedBeforeLoad

    monkeypatch.setattr(runner, "bind_cold_environment", bind_environment)
    with pytest.raises(StoppedBeforeLoad):
        runner.run_profile(runner.example_arguments(config))
    assert len(captured) == 1
    env = captured[0]
    assert env["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] == ("1" if enabled else "0")
    assert env["RPU_PI05_DENOISE_NVFP4_GEGLU_M50"] == "0"
    if enabled:
        for flag in ("DENOISE_W4_FASTPATH", "RING_XOR3", "DENOISE_XOR3_GATED_MLP",
                     "DENOISE_XOR3_GATED_ATTN"):
            assert env["RPU_PI05_" + flag] == "1"
    assert config == original


@pytest.mark.parametrize("precision,execution,paired,failure", [
    ("fp16", None, True, None),
    ("w8a16", None, True, None),
    ("w8a16_action_nvfp4", None, True, None),
    ("w4a16_kvint8", None, False, None),
    ("fp16", {stage: {"linear_acc32": True} for stage in ("vision", "prefill", "action")}, True, None),
    ("w8a16_action_nvfp4", {"action": {"linear_acc32": False}, "components": {
        "action_expert": {"action": {"linear_acc32": True}}}}, True, None),
    ("fp16", {"model": {"num_cores": 4}}, False, None),
    ("w8a16", {"model": {"num_cores": 6}}, False, None),
    ("fp16", {"prefill": {"chunk_size": 400, "linear_acc32": False}, "components": {
        "language_model": {"prefill": {"chunk_size": 128, "padding_rows": 16, "linear_acc32": True}}}}, False, None),
    ("fp16", {"components": {"vision_encoder": {"vision": {"chunk_size": 256}},
                            "action_expert": {"action": {"chunk_size": 64}}}}, False, None),
    ("fp16", {"vision": {"chunk_size": 768}, "action": {"chunk_size": 64},
              "prefill": {"padding_rows": "auto"}}, True, None),
    *[("fp16", None, True, failure) for failure in ("configure", "to", "prepare", "inference", "close")],
    ("fp16", None, True, "inference_and_close"),
])
def test_measurement_preserves_cold_execution_and_closes_before_reporting(
        monkeypatch, tmp_path, precision, execution, paired, failure, diagnostics=False, instruction_residency=None):
    runner = load("examples/runners/pi05/optimized.py")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    quant = {"fp16": None, "w8a16": {"method": "w8a16"},
             # Missing old scope/group metadata is valid for full-model INT4.
             "w4a16_kvint8": {"method": "w4a16", "mixed_int8_suffixes": ["k_proj.weight", "v_proj.weight"]},
             "w8a16_action_nvfp4": {"method": "nvfp4a16", "int4_components": ["expert"],
                 "mixed_int8_suffixes": ["k_proj.weight", "v_proj.weight"],
                 "nvfp4_abi": "striped_v2", "nvfp4_block_size": 16}}[precision]
    if quant is not None:
        (checkpoint / "rpu_quant_config.json").write_text(json.dumps(quant))
    batch_path = tmp_path / "batch.pt"
    torch.save({}, batch_path)
    config = {"pi05": {"precision": precision},
              "input": {"checkpoint": str(checkpoint), "batch": str(batch_path), "cameras": 3},
              "cpu": {"threads": 3},
              "run": {"warmup": 2, "runs": 3, "output_dir": str(tmp_path / "result")}}
    if execution is not None:
        config["rpu_execution"] = deepcopy(execution)
    if diagnostics:
        config["run"].update(profile=True, hwperf=True)
    if instruction_residency is not None:
        config["pi05"]["instruction_residency"] = instruction_residency
    original = deepcopy(config)
    expected_execution = deepcopy(execution) if execution is not None else {}
    expected_execution.setdefault("model", {"num_cores": 8})
    if paired:
        expected_execution.setdefault("prefill", {})["chunk_size"] = 400
    phases, calls = {}, []

    def event(name):
        calls.append(name)
        if failure == name or (failure == "inference_and_close" and name in {"inference", "close"}):
            raise RuntimeError(f"{name} failed")

    class Policy:
        _lerobot_policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, image_features={}))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert torch.get_num_threads() == 3
            assert calls == (["environment", "hwperf_on"] if diagnostics else ["environment"])
            event("load")
            assert args == (str(checkpoint),)
            expected_kwargs = {"dtype": torch.float16, "trust_remote_code": False,
                               "rpu_execution": expected_execution}
            if instruction_residency is not None:
                expected_kwargs["graph_runtime_policy"] = requested_policy
            assert kwargs == expected_kwargs
            return cls()

        def to(self, device):
            assert device == "rpu" and calls[-1] == "validate"
            event("to")

        def prepare_graphs(self, batch, *, num_steps, precompute_adarms):
            assert num_steps == 10 and precompute_adarms is paired
            assert calls[-1] == "to"
            event("prepare")
            return {"adarms_table": {"dense_w8a16": False} if paired else None, "phase": "READY",
                    "build_replay_bit_equal": True, "ready_replay_validated": True,
                    "ready_stats": {"denoise": {"recaptures": 0}}}

        def predict_action_chunk(self, batch, *, num_steps):
            assert num_steps == 10
            event("inference")
            phases.update(vision_ms=1., prefill_ms=2., denoise_ms=3.)
            return torch.zeros(1, 50, 7)

        def close(self):
            assert not (tmp_path / "result/report.json").exists()
            event("close")

    api, registry = ModuleType("rpu_backend.api"), ModuleType("rpu_backend.model_registry")
    api.Pi05Policy = Policy
    registry.model_path = lambda alias: checkpoint
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)
    monkeypatch.setitem(sys.modules, "rpu_backend.model_registry", registry)
    requested_policy = object()
    if instruction_residency is not None:
        graph = ModuleType("rpu_backend.graph")
        def from_environment(**kwargs):
            assert calls == ["environment"]
            assert kwargs == {"batch_snapshot_mode": instruction_residency}
            return requested_policy
        graph.GraphRuntimePolicy = SimpleNamespace(from_environment=from_environment)
        monkeypatch.setitem(sys.modules, "rpu_backend.graph", graph)
    bound_environment = {}

    def bind_environment(env):
        bound_environment.update(env)
        assert env["RPU_PI05_KVINSERT_PAD16"] == str(int(expected_execution["model"]["num_cores"] == 8))
        assert env["RPU_PI05_DENOISE_UNROLL"] == "1"
        assert env["RPU_PI05_SIGLIP_W8A16"] == str(int(precision != "fp16"))
        assert env["RPU_PI05_ADARMS_DENSE_W8A16"] == str(int(precision != "fp16"))
        serial = (execution or {}).get("components", {}).get("vision_encoder", {}).get("vision", {}).get("chunk_size") == 256
        packed = expected_execution["model"]["num_cores"] == 8 and not serial
        assert env["RPU_PI05_SIGLIP_BATCH"] == str(int(packed))
        assert env["RPU_PI05_RING_XOR3"] == str(int(paired))
        event("environment")

    monkeypatch.setattr(runner, "bind_cold_environment", bind_environment)
    monkeypatch.setattr(runner, "profile_libraries", lambda: ({}, {}))
    monkeypatch.setattr(runner, "runtime_payloads", lambda libraries: {"extension": "/native.so"})
    monkeypatch.setattr(runner, "configure_text_profile", lambda *args: event("configure") or {})
    monkeypatch.setattr(runner, "validate_camera_batch", lambda *args, **kwargs: event("validate"))
    threads = [8]
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    monkeypatch.setattr(torch, "set_num_threads", lambda count: threads.__setitem__(0, count))
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads[0])
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)

    @contextmanager
    def observe(policy):
        assert paired, "generic execution must not claim measurements for missing stages"
        yield phases

    monkeypatch.setattr(runner, "stage_timings", observe)
    if diagnostics:
        spec = importlib.util.spec_from_file_location(
            "pi05_example_performance", ROOT / "python/rpu_backend/runtime/performance.py")
        performance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(performance)

        @contextmanager
        def profile_context(enabled):
            assert enabled and calls.count("inference") == 5
            event("profile_on")
            yield SimpleNamespace(
                export_chrome_trace=lambda path: Path(path).write_text("{}"),
                key_averages=lambda: SimpleNamespace(table=lambda **kwargs: "profile summary"))
            event("profile_off")

        @contextmanager
        def hwperf_context(directory, *, max_dumps):
            assert calls == ["environment"] and max_dumps == 32
            event("hwperf_on")
            yield
            assert calls[-1] == "close"
            trace_dir = Path(directory)
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / "rpu_hwperf_mock.json").write_text(json.dumps({"traceEvents": [
                {"ph": "B", "ts": 1},
                {"ph": "E", "ts": 11, "cat": "Compute", "name": "linear_1", "args": {"duration_us": 10}},
            ]}))
            event("hwperf_off")

        performance._profile_ctx = profile_context
        monkeypatch.setitem(sys.modules, "rpu_backend.runtime.performance", performance)
        monkeypatch.setattr(torch, "rpu", SimpleNamespace(
            hw_perf_trace=hwperf_context, _is_in_bad_fork=lambda: False,
            manual_seed_all=lambda seed: None), raising=False)
    if failure:
        primary = "inference" if failure == "inference_and_close" else failure
        with pytest.raises(RuntimeError, match=f"{primary} failed"):
            runner.run_example_config(config)
        report = json.loads((tmp_path / "result/report.json").read_text())
        assert report["execution"]["status"] == "FAILED"
        assert report["validation"]["runtime"]["status"] == "NOT_COMPLETED"
        assert report["validation"]["reference"]["status"] == "NOT_RUN"
        assert report["error"] == f"RuntimeError: {primary} failed"
        if failure in {"close", "inference_and_close"}:
            assert report["cleanup_error"] == "RuntimeError: close failed"
    else:
        assert runner.run_example_config(config) == 0
        report = json.loads((tmp_path / "result/report.json").read_text())
        assert report["execution"]["status"] == "COMPLETED"
        assert report["validation"]["reference"]["status"] == "NOT_RUN"
        assert report["timing"][0]["samples"] == [row["total_ms"] for row in report["samples_ms"]]
        assert report["timing"][0]["unit"] == "ms" and report["timing"][0]["statistic"] == "mean"
        assert report["artifacts"]["report"] == str(tmp_path / "result/report.json")
        assert len(report["samples_ms"]) == 3 and report["warmup"] == 2
        assert report["cpu_runtime"]["torch_num_threads"] == 3
        assert report["cpu_runtime"]["OMP_NUM_THREADS"] == "3"
        assert report["execution_route"] == ("paired_tp8" if paired else "generic")
        assert report["precision"] == precision
        assert report["action_nvfp4_geglu_acc16"] is (
            bound_environment["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] == "1")
        if paired:
            assert report["mean_ms"]["denoise_ms"] == 3.
        else:
            assert report["timed_phases"] == ["total_ms"]
            assert set(report["mean_ms"]) == {"total_ms"}
            assert all(set(row) == {"total_ms"} for row in report["samples_ms"])
        assert report["formal_certification"] is False and report["correctness"] == "not_run"
        assert report["registered_profile"] is None
        assert report["rpu_execution"] == expected_execution
        assert report["instruction_residency"] == instruction_residency
        assert report["runtime_payloads"]["extension"] == "/native.so"
        assert report["graph_prepare"]["phase"] == "READY"
        assert report["graph_prepare"]["ready_replay_validated"] is True
        assert torch.load(tmp_path / "result/actions.pt", weights_only=True).shape == (1, 50, 7)
        assert calls.count("prepare") == 1 and calls.count("inference") == (6 if diagnostics else 5)
        assert report["profile"] is diagnostics and report["hwperf_enabled"] is diagnostics
        if diagnostics:
            assert str(tmp_path / "result/profile.trace.json") in report["artifacts"]["outputs"]
            assert str(tmp_path / "result/profile.txt") in report["artifacts"]["outputs"]
            assert str(tmp_path / "result/hwperf/rpu_hwperf_mock.json") in report["artifacts"]["outputs"]
            assert report["artifacts"]["hwperf_directory"] == str(tmp_path / "result/hwperf")
            assert (tmp_path / "result/profile.trace.json").is_file()
            assert (tmp_path / "result/profile.txt").read_text() == "profile summary"
            assert report["hwperf"][0]["rows"][0]["name"] == "linear"
            assert calls.index("profile_on") < calls.index("profile_off") < calls.index("close")
    assert calls[-1] == ("hwperf_off" if diagnostics else "close") and calls.count("close") == 1
    assert config == original


def test_diagnostic_profile_is_extra_and_hwperf_starts_before_model_install(monkeypatch, tmp_path):
    test_measurement_preserves_cold_execution_and_closes_before_reporting(
        monkeypatch, tmp_path, "fp16", None, True, None, diagnostics=True)


@pytest.mark.parametrize("execution", [
    {"action": {"linear_acc32": "true"}},
    {"vision": {"linear_acc32": 1}},
    {"prefill": {"linear_acc32": None}},
    {"action": {"unknown": True}},
    {"decode": {"linear_acc32": True}},
    {"components": {"unknown": {"action": {"linear_acc32": True}}}},
    {"components": {"vision_encoder": {"action": {"linear_acc32": True}}}},
    {"model": {"num_cores": 4}},
    {"prefill": {"chunk_size": 128}},
])
def test_accumulation_config_rejects_before_environment_or_loader(monkeypatch, execution):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = common.load_config(CONFIGS / "3cam/w8a16_action_nvfp4.toml")
    args = runner.example_arguments(config)
    config["rpu_execution"] = execution
    with pytest.raises((TypeError, ValueError)):
        runner.example_arguments(config)
    args.rpu_execution = execution
    monkeypatch.setattr(runner, "bind_cold_environment",
                        lambda env: pytest.fail("invalid execution config reached cold environment binding"))
    with pytest.raises((TypeError, ValueError)):
        runner.run_profile(args, bind_libraries=True)


def test_example_notes_do_not_publish_historical_model_results():
    runner = load("examples/runners/pi05/optimized.py")
    notes = runner.numerical_notes(cameras=3, text_tokens=128, valid_tokens=128,
                                  precision="w4a16", prefill_w8a8=False)
    assert notes == ["This example does not run a reference comparison or robot task."]
    notes = runner.numerical_notes(cameras=2, text_tokens=96, valid_tokens=65,
                                  precision="fp16", prefill_w8a8=False)
    assert notes == ["This example does not run a reference comparison or robot task."]


@pytest.mark.parametrize("filename,inputs,execution", [
    ("fp16", {"cameras": 2}, {"model": {"num_cores": 4}}),
    ("w8a16", {"text_tokens": 64}, {"model": {"num_cores": 6}}),
    ("w4a16_kvint8", {"cameras": 2}, {}),
    ("w4a16_kvint8", {}, {"model": {"num_cores": 6}}),
    ("fp16", {}, {"model": {"num_cores": 5}}),
    ("fp16", {}, {"model": {"num_cores": 4}, "vision": {"chunk_size": 768}}),
    ("fp16", {}, {"vision": {"chunk_size": 64}}),
    ("fp16", {}, {"action": {"chunk_size": 32}}),
])
def test_core_and_physical_chunk_mismatches_fail_before_runtime(filename, inputs, execution):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = common.load_config(CONFIGS / "3cam/fp16.toml")
    config["pi05"]["precision"] = filename
    config["input"].update(inputs)
    config["rpu_execution"] = execution
    with pytest.raises(ValueError):
        runner.example_arguments(config)


@pytest.mark.parametrize("precision,metadata", [
    ("w4a16_kvint8", {"method": "nvfp4a16"}),
    ("w4a16_kvint8", {"method": "w4a16", "int4_components": ["expert"]}),
    ("w4a16_kvint8", {"method": "w4a16", "mode": "group_symmetric"}),
    ("w4a16_kvint8", {"method": "w4a16", "mixed_int8_suffixes": ["q_proj.weight"]}),
    ("w8a16_action_nvfp4", {"method": "w4a16"}),
])
def test_wrong_checkpoint_recipe_rejects_before_policy_creation(monkeypatch, tmp_path, precision, metadata):
    runner = load("examples/runners/pi05/optimized.py")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    metadata = {"mixed_int8_suffixes": ["k_proj.weight", "v_proj.weight"], **metadata}
    (checkpoint / "rpu_quant_config.json").write_text(json.dumps(metadata))
    config = {"pi05": {"precision": precision},
              "input": {"checkpoint": str(checkpoint), "batch": str(tmp_path / "not-loaded.pt"), "cameras": 3},
              "run": {"warmup": 0, "runs": 1}}
    api, registry = ModuleType("rpu_backend.api"), ModuleType("rpu_backend.model_registry")
    api.Pi05Policy = SimpleNamespace(from_pretrained=lambda *a, **kw: pytest.fail("invalid checkpoint reached loader"))
    registry.model_path = lambda alias: checkpoint
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)
    monkeypatch.setitem(sys.modules, "rpu_backend.model_registry", registry)
    monkeypatch.setattr(runner, "bind_cold_environment", lambda env: None)
    monkeypatch.setattr(runner, "profile_libraries", lambda: ({}, {}))
    monkeypatch.setattr(runner, "runtime_payloads", lambda libraries: {})
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)
    with pytest.raises(ValueError):
        runner.run_example_config(config)


@pytest.mark.parametrize("mode", ("auto", "off"))
def test_optional_instruction_residency_reaches_cold_policy(monkeypatch, tmp_path, mode):
    test_measurement_preserves_cold_execution_and_closes_before_reporting(
        monkeypatch, tmp_path, "w8a16", None, True, None, instruction_residency=mode)


@pytest.mark.parametrize("value", (True, None, "force", 1))
def test_invalid_instruction_residency_is_rejected_before_loading(value):
    runner = load("examples/runners/pi05/optimized.py")
    config = load("examples/common/config.py").load_config(CONFIGS / "3cam/w8a16.toml")
    config["pi05"]["instruction_residency"] = value
    with pytest.raises(ValueError, match="instruction_residency"):
        runner.example_arguments(config)
