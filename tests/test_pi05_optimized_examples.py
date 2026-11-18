"""One Pi0.5 entry resolves physical choices before backend imports and load."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "examples/configs/pi05"


# The LIBERO checkpoint's config.json features (two 256 cameras, one 224 empty slot).
LIBERO_CONFIG = {"max_action_dim": 32, "image_resolution": [224, 224], "output_features": {
    "action": {"type": "ACTION", "shape": [7]}}, "input_features": {
    "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.images.image2": {"type": "VISUAL", "shape": [3, 256, 256]},
    "observation.images.empty_camera_0": {"type": "VISUAL", "shape": [3, 224, 224]},
    "observation.state": {"type": "STATE", "shape": [8]}}}


def load(relative):
    return importlib.import_module(relative[:-3].replace("/", "."))


def template(name, cameras=3, horizon="h50"):
    """A Pi0.5 template (3 cameras); cameras=2 applies preset 2cam_t32 with a 2-camera batch."""
    common = load("examples/common/config.py")
    path = CONFIGS / f"{horizon}/{name}.toml"
    if cameras == 3:
        return common.load_config(path)
    return common.load_config(path, input_preset="2cam_t32",
                              overrides={"input": {"batch": "/path/to/pi05_2cam_t32_batch.pt"}})


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
    h50_names = ("fp16", "w8a16", "w8a16_action_nvfp4", "w8a16_prefill_w8a8_action_nvfp4",
                 "w8a16_vision_w4a16_action_nvfp4")
    h10_names = ("fp16", "w8a16", "w8a16_action_nvfp4", "w8a16_prefill_w8a8_action_nvfp4")
    expected = ({CONFIGS / f"h50/{name}.toml" for name in h50_names}
                | {CONFIGS / f"h10/{name}.toml" for name in h10_names})
    # Real robot (848x480 cameras) and simulation (640x480) take the same model; only the frames differ.
    # The t64/ and t96/ templates carry the dataset's short task instruction at the
    # smallest width that holds it with any state of that layout.
    kuavo = {CONFIGS / f"kuavo/{site}{layout}/{precision}.toml": (dim, width, frame_size)
             for site, frame_size in (("", [848, 480]), ("sim/", [640, 480]))
             for layout, dim, width in (("single_arm", 8, 128), ("dual_arm", 16, 160), ("wholebody", 20, 192),
                                        ("single_arm/t64", 8, 64), ("dual_arm/t96", 16, 96))
             for precision in ("fp16", "w8a16")}
    assert paths == expected | set(kuavo)
    registered = set()
    entries = [(None, template(name, cameras, horizon))
               for horizon, names in (("h50", h50_names), ("h10", h10_names))
               for name in names for cameras in (2, 3)]
    entries.extend((path, common.load_config(path)) for path in sorted(kuavo))
    for path, config in entries:
        args = runner.example_arguments(config)
        precision = config["pi05"]["precision"]
        assert args.precision == ("w4a16" if "nvfp4" in precision else precision)
        assert args.public_precision == precision
        assert args.prefill_w8a8 == (precision == "w8a16_prefill_w8a8_action_nvfp4")
        # Vision W4 keeps SigLIP on four lanes under the paired budget 8 (no profile).
        assert args.vision_w4a16 == (precision == "w8a16_vision_w4a16_action_nvfp4")
        metadata = common.example_metadata(config)
        if metadata is not None:
            registered.add(metadata["profile_id"])
        if path in kuavo:
            # Raw openpi inputs; the checkpoint path is a placeholder, never a registered profile.
            assert metadata is None and args.batch is None and set(args.raw_inputs) >= set(runner._openpi.RAW_KEYS)
            dim, width, frame_size = kuavo[path]
            assert (args.action_dim, args.text_tokens, args.cameras) == (dim, width, 3)
            assert args.raw_inputs["frame_size"] == frame_size
            assert len(args.raw_inputs["delta_action_mask"]) == args.action_dim
        execution, paired, chunk = runner.resolve_execution(args)
        if execution["model"]["num_cores"] == 8:
            assert paired and chunk == (256 * args.cameras + args.text_tokens) // 2
            for text_tokens in (32, 64, 96, 128, 160):
                candidate = deepcopy(config)
                candidate["input"]["text_tokens"] = text_tokens
                _, admitted, bucket = runner.resolve_execution(runner.example_arguments(candidate))
                assert admitted and bucket == (256 * args.cameras + text_tokens) // 2
            candidate = deepcopy(config)
            candidate["input"]["text_tokens"] = 192
            if args.cameras == 3 and precision in ("fp16", "w8a16"):
                _, admitted, bucket = runner.resolve_execution(runner.example_arguments(candidate))
                assert admitted and bucket == 480
            else:
                with pytest.raises(ValueError, match="T192 requires"):
                    runner.resolve_execution(runner.example_arguments(candidate))
            if path in kuavo:
                for cores in (4, 6):
                    candidate = deepcopy(config)
                    candidate["rpu_execution"] = {"model": {"num_cores": cores}}
                    reduced, paired, chunk = runner.resolve_execution(runner.example_arguments(candidate))
                    assert reduced["model"]["num_cores"] == cores and not paired
                    assert chunk == "auto"
                    assert common.example_metadata(candidate) is None
            elif precision in ("fp16", "w8a16", "w8a16_action_nvfp4", "w8a16_vision_w4a16_action_nvfp4"):
                for cores in (4, 6):
                    candidate = deepcopy(config)
                    candidate["rpu_execution"] = {"model": {"num_cores": cores}}
                    reduced, paired, _ = runner.resolve_execution(runner.example_arguments(candidate))
                    assert reduced["model"]["num_cores"] == cores and not paired
                    metadata = common.example_metadata(candidate)
                    if "action_nvfp4" in precision or config["input"].get("action_horizon") == 10:
                        assert metadata is None  # NVFP4 and H10 templates are profile-less.
                    else:
                        registered.add(metadata["profile_id"])
        else:
            assert not paired
    required = {profile_id for profile_id, profile in common.planner_profiles().items()
                if profile["family"] == "pi05" and common.required_example(profile)}
    retired = "pi05-libero-finetuned-w4real-kvint8.w4a16-kvint8.vla"
    assert retired in required  # Its registered capability was not withdrawn.
    assert registered == required - {retired}


@pytest.mark.parametrize("cameras,horizon", [(2, 10), (2, 50), (3, 10), (3, 50)])
def test_full_prefix_toml_passes_the_real_public_cli(cameras, horizon, tmp_path, monkeypatch, capsys):
    full = cameras * 256 + 32
    text = (CONFIGS / f"h{horizon}/w8a16.toml").read_text()
    text = text.replace('precision = "w8a16"', 'precision = "w8a16"\nprefill_w8a8 = true')
    path = tmp_path / "full-prefix.toml"
    path.write_text(text + f"\n[rpu_execution.prefill]\nchunk_size = {full}\n")
    arguments = ["run.py", "--config", str(path), "--check-config"]
    if cameras == 2:
        arguments += ["--input", "2cam_t32", "--set", "input.batch=/path/to/pi05_2cam_t32_batch.pt"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert importlib.import_module("examples.common.cli").main() == 0
    assert "configuration OK" in capsys.readouterr().out
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = common.load_config(
        path, **({"input_preset": "2cam_t32", "overrides": {"input": {
            "batch": "/path/to/pi05_2cam_t32_batch.pt"}}} if cameras == 2 else {}))
    assert config["pi05"]["prefill_w8a8"] is True
    _, paired, actual = runner.resolve_execution(runner.example_arguments(config))
    assert paired and actual == full


@pytest.mark.parametrize("cameras,horizon", [(2, 10), (2, 50), (3, 10), (3, 50)])
@pytest.mark.parametrize("scope", ["root", "component", "root_component_auto"])
def test_public_full_prefix_w8a8_binds_the_actual_chunk_and_mature_routes(
        cameras, horizon, scope, monkeypatch):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = template("w8a16", cameras, f"h{horizon}")
    config["pi05"]["prefill_w8a8"] = True
    full = cameras * 256 + 32
    execution = config["rpu_execution"]
    if scope != "component":
        execution["prefill"] = {"chunk_size": full, "padding_budget": 0}
    if scope != "root":
        execution["components"] = {"language_model": {"prefill": {
            "chunk_size": full if scope == "component" else "auto"}}}
    original = deepcopy(config)
    args = runner.example_arguments(config)
    resolved, paired, chunk = runner.resolve_execution(args)
    assert paired and chunk == full and resolved["model"]["num_cores"] == 8
    assert resolved["prefill"] == {"chunk_size": full, "padding_rows": 0}
    child = resolved.get("components", {}).get("language_model", {}).get("prefill")
    if child is not None:
        assert child == {"chunk_size": full, "padding_rows": 0}
    assert common.example_metadata(config) is None  # No inherited A16 certification.
    captured = []

    class BeforeBackendImport(Exception):
        pass

    def capture(env):
        captured.append(dict(env))
        raise BeforeBackendImport

    monkeypatch.setattr(runner, "bind_cold_environment", capture)
    with pytest.raises(BeforeBackendImport):
        runner.run_profile(args)
    env, = captured
    assert env["RPU_PI05_PREFILL_FULL_CHUNK_W8A8"] == "1"
    assert env["RPU_PI05_PREFILL_GATEUP_W8A8"] == "1"
    assert env["RPU_PI05_PREFILL_GATEUP_W8A8_PANEL512"] == str(int(cameras == 2))
    for name in ("PREFILL_OWNER_NORM", "PREFILL_DOWN_PAIR", "PREFILL_GATEUP_PAIR",
                 "PREFILL_KV_ONLY", "PREFILL_KV1_PAIR_DIRECT", "DENOISE_GEGLU_M50"):
        assert env["RPU_PI05_" + name] == "1"
    assert env["RPU_PI05_DENOISE_W4_FASTPATH"] == "0"
    assert config == original


@pytest.mark.parametrize("change", [
    {"pi05": {"prefill_w8a8": False}},
    {"pi05": {"precision": "fp16", "prefill_w8a8": False}},
    {"pi05": {"precision": "w8a16_prefill_w8a8_action_nvfp4"}},
    {"input": {"text_tokens": 64}},
    {"input": {"action_horizon": 20}},
    {"rpu_execution": {"model": {"num_cores": 4}}},
    {"rpu_execution": {"model": {"num_cores": 6}}},
    {"rpu_execution": {"prefill": {"chunk_size": 560}}},
    {"rpu_execution": {"prefill": {"padding_rows": 16}}},
    {"rpu_execution": {"prefill": {"padding_budget": 16}}},
    {"rpu_execution": {"prefill": {"linear_acc32": True}}},
    {"rpu_execution": {"components": {"language_model": {"prefill": {"chunk_size": 272}}}}},
    {"rpu_execution": {"components": {"vision_encoder": {"vision": {"chunk_size": 256}}}}},
])
def test_full_prefix_rejects_unsupported_public_requests_before_load(change, monkeypatch):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = template("w8a16", 2, "h10")
    config["pi05"]["prefill_w8a8"] = True
    config["rpu_execution"]["prefill"] = {"chunk_size": 544}
    for section, fields in change.items():
        if section == "rpu_execution":
            for stage, values in fields.items():
                config[section].setdefault(stage, {}).update(deepcopy(values))
        else:
            config[section].update(fields)
    monkeypatch.setattr(runner, "bind_cold_environment",
                        lambda env: pytest.fail("unsupported request reached model load"))
    with pytest.raises((ValueError, TypeError)):
        runner.run_profile(runner.example_arguments(config))


@pytest.mark.parametrize("precision,value", [
    ("w8a16", "true"), ("w8a16", 1), ("w8a16", None),
    ("fp16", True), ("w8a16_action_nvfp4", True),
    ("w8a16_prefill_w8a8_action_nvfp4", False),
])
def test_public_prefill_precision_is_explicit_and_cannot_conflict_with_recipe(precision, value):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = template("w8a16", 2, "h50")
    config["pi05"].update(precision=precision, prefill_w8a8=value)
    with pytest.raises(ValueError, match="prefill_w8a8"):
        runner.example_arguments(config)


@pytest.mark.parametrize("cameras,horizon", [(2, 10), (2, 50), (3, 10), (3, 50)])
def test_default_half_chunk_binds_full_selector_off(cameras, horizon, monkeypatch):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = template("w8a16", cameras, f"h{horizon}")
    args = runner.example_arguments(config)
    _, paired, chunk = runner.resolve_execution(args)
    assert paired and chunk == (cameras * 256 + 32) // 2 and not args.prefill_w8a8

    class BeforeBackendImport(Exception):
        pass

    def capture(env):
        assert env["RPU_PI05_PREFILL_FULL_CHUNK_W8A8"] == "0"
        assert env["RPU_PI05_PREFILL_GATEUP_W8A8"] == "0"
        raise BeforeBackendImport

    monkeypatch.setattr(runner, "bind_cold_environment", capture)
    with pytest.raises(BeforeBackendImport):
        runner.run_profile(args)


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
    config = common.load_config(CONFIGS / "h50/w8a16_prefill_w8a8_action_nvfp4.toml")
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
    config = common.load_config(CONFIGS / "h50/fp16.toml")
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
        config = common.load_config(CONFIGS / "h50/fp16.toml")
        config["pi05"]["precision"] = name
        config["input"]["checkpoint"] = {
            "fp16": "pi05-libero-finetuned",
            "w8a16": "pi05-libero-finetuned-w8a16-vlm-expert",
            "w4a16_kvint8": "pi05-libero-finetuned-w4real-kvint8",
        }[name]
        assert common.example_metadata(config) is not None
        config["rpu_execution"] = execution
        assert common.example_metadata(config) is None


def test_public_entry_dispatches_one_engine_for_selected_config(monkeypatch, capsys, tmp_path):
    common = load("examples/common/config.py")
    path = CONFIGS / "h50/w8a16_action_nvfp4.toml"
    default = CONFIGS / "h50/w8a16.toml"
    two_cameras = ["--input", "2cam_t32", "--set", "input.batch=/path/to/pi05_2cam_t32_batch.pt"]
    monkeypatch.setattr(sys, "argv", ["pi05.py", "--config", str(path), *two_cameras, "--check-config"])
    assert importlib.import_module("examples.common.cli").main() == 0
    assert "syntax/input envelope only" in capsys.readouterr().out
    runner = common._pi05_runner()
    seen = []
    monkeypatch.setattr(runner, "run_example_config", lambda config: seen.append(config) or 0)
    monkeypatch.setattr(importlib.import_module("examples.common.cli"), "_pi05_runner", lambda: runner)
    reduced = tmp_path / "fp16.toml"
    reduced.write_text((CONFIGS / "h50/fp16.toml").read_text().replace("num_cores = 8", "num_cores = 4"))
    for selected, extra in ((path, two_cameras), (default, []), (reduced, [])):
        monkeypatch.setattr(sys, "argv", ["pi05.py", "--config", str(selected), *extra])
        assert importlib.import_module("examples.common.cli").main() == 0
    assert [item["pi05"]["precision"] for item in seen] == ["w8a16_action_nvfp4", "w8a16", "fp16"]
    assert [item["input"]["cameras"] for item in seen] == [2, 3, 3]


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
    pi05.runtime = SimpleNamespace(_run_denoise_fused=lambda: 1, _run_denoise_python_baseline=lambda: 4)
    adapters.siglip = SimpleNamespace(_siglip_forward_images=lambda: 2)
    monkeypatch.setitem(sys.modules, "rpu_backend.adapters.pi05", pi05)
    monkeypatch.setitem(sys.modules, "rpu_backend.adapters", adapters)

    class Vlm:
        def _rpu_prefill_kv_only(self):
            return 3

    class Expert:
        def embed_image(self):
            # A packed SigLIP call reached from serial vision is one Vision span.
            return adapters.siglip._siglip_forward_images()

    vlm, expert = Vlm(), Expert()
    expert.paligemma = SimpleNamespace(model=SimpleNamespace(language_model=vlm))
    policy = SimpleNamespace(_lerobot_policy=SimpleNamespace(model=SimpleNamespace(paligemma_with_expert=expert)))
    original = pi05.runtime._run_denoise_fused
    clock = iter(range(100))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    with pytest.raises(RuntimeError, match="forward failed"):
        with runner.stage_timings(policy) as phases:
            assert pi05.runtime._run_denoise_python_baseline() == 4
            assert expert.embed_image() == 2
            assert vlm._rpu_prefill_kv_only() == 3
            assert phases == {"denoise_ms": 1000., "vision_ms": 1000., "prefill_ms": 1000.}
            raise RuntimeError("forward failed")
    assert pi05.runtime._run_denoise_fused is original
    assert "_rpu_prefill_kv_only" not in vars(vlm) and "embed_image" not in vars(expert)


@pytest.mark.parametrize("key,value", [
    ("RPU_LINEAR_ACC32", "1"), ("RPU_PI05_PREFIX_MASK_CACHE", "0"),
    ("RPU_PI05_GEMMA_GRAPH", "0"), ("RPU_PI05_DENOISE_GRAPH", "0"),
    ("RPU_PI05_LOAD_PREFIX_EMBS", "/tmp/replacement.pt"),
])
def test_optimized_config_rejects_inherited_execution_and_input_overrides(monkeypatch, key, value):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    args = runner.example_arguments(common.load_config(CONFIGS / "h50/fp16.toml"))
    monkeypatch.setattr(runner, "profile_libraries", lambda: ({}, {}))
    monkeypatch.delitem(sys.modules, "rpu_backend", raising=False)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        runner.run_profile(args, bind_libraries=True)


def test_actual_runtime_roles_must_match_selected_libraries(monkeypatch):
    runner = load("examples/runners/pi05/optimized.py")
    libraries = {"RPU_KERNEL_LIB_PATH": {"path": "/main.ref", "sha256": "main"},
                 "RPU_KERNEL_LIB_PATH_EXPANSION": {"path": "/pi05.ref", "sha256": "pi05"}}
    actual = {"extension": "/native.so", "oplib_roles": {"main": "/main.ref", "expansion": "/pi05.ref"},
              "oplib_payloads": {"/main.ref": {"sha256": "main"}, "/pi05.ref": {"sha256": "pi05"}}}
    backend = ModuleType("rpu_backend")
    backend._cpp_ext = SimpleNamespace(get_runtime_payload_paths=lambda: actual)
    monkeypatch.setitem(sys.modules, "rpu_backend", backend)
    assert runner.runtime_payloads(libraries) is actual
    actual["oplib_payloads"]["/pi05.ref"]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="loaded expansion library"):
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


@pytest.mark.parametrize("precision", ["fp16", "w8a16"])
@pytest.mark.parametrize("execution,paired", [
    ({}, True),
    ({"vision": {"chunk_size": 768}, "action": {"chunk_size": 64}}, True),
    ({"vision": {"chunk_size": 256}}, False),
])
def test_t192_pairs_c480_and_streams_vision_layernorm_only_when_paired(monkeypatch, precision, execution, paired):
    runner = load("examples/runners/pi05/optimized.py")
    config = {"pi05": {"precision": precision},
              "input": {"checkpoint": "/not-loaded/checkpoint", "batch": "/not-loaded/batch.pt",
                        "cameras": 3, "text_tokens": 192, "num_steps": 10},
              "run": {}, "rpu_execution": deepcopy(execution)}
    captured = []

    class StoppedBeforeLoad(Exception):
        pass

    def bind_environment(env):
        captured.append(dict(env))
        raise StoppedBeforeLoad

    monkeypatch.setattr(runner, "bind_cold_environment", bind_environment)
    args = runner.example_arguments(config)
    _, admitted, chunk = runner.resolve_execution(args)
    assert admitted == paired and chunk == (480 if paired else 320)
    with pytest.raises(StoppedBeforeLoad):
        runner.run_profile(args)
    env = captured[0]
    # Paired: every TP8 fusion, plus the SigLIP LayerNorm streaming C480 needs
    # to fit; otherwise the plain generic kernels without streaming.
    generic = {"FUSED_DENOISE", "KVINSERT_PAD16", "DENOISE_UNROLL"}
    for name in runner._ON:
        assert env["RPU_PI05_" + name] == str(int(paired or name in generic)), name
    assert env["RPU_PI05_VISION_LN_STREAM"] == str(int(paired))
    assert env["RPU_PI05_PREFILL_GATEUP_W8A8"] == "0"


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
        monkeypatch, tmp_path, precision, execution, paired, failure, diagnostics=False,
        instruction_residency=None, full_chunk=False, cameras=3, horizon=50):
    runner = load("examples/runners/pi05/optimized.py")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps(LIBERO_CONFIG))
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
              "input": {"checkpoint": str(checkpoint), "batch": str(batch_path),
                        "cameras": cameras, "action_horizon": horizon},
              "cpu": {"threads": 3},
              "run": {"warmup": 2, "runs": 3, "output_dir": str(tmp_path / "result")}}
    if full_chunk:
        config["pi05"]["prefill_w8a8"] = True
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
        expected_execution.setdefault("prefill", {})["chunk_size"] = (
            cameras * 256 + 32 if full_chunk else (cameras * 256 + 32) // 2)
        if full_chunk:
            expected_execution["prefill"]["padding_rows"] = 0
    phases, calls = {}, []

    def event(name):
        calls.append(name)
        if failure == name or (failure == "inference_and_close" and name in {"inference", "close"}):
            raise RuntimeError(f"{name} failed")

    class Policy:
        _lerobot_policy = SimpleNamespace(config=SimpleNamespace(chunk_size=horizon, image_features={}))

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
            expected_precompute = paired or (
                expected_execution["model"]["num_cores"] in (4, 6)
                and precision in ("fp16", "w8a16", "w8a16_action_nvfp4")
            )
            assert num_steps == 10 and precompute_adarms is expected_precompute
            assert calls[-1] == "to"
            event("prepare")
            return {"adarms_table": {"dense_w8a16": False} if expected_precompute else None, "phase": "READY",
                    "build_replay_bit_equal": True, "ready_replay_validated": True,
                    "ready_stats": {"denoise": {"recaptures": 0}}}

        def predict_action_chunk(self, batch, *, num_steps):
            assert num_steps == 10
            event("inference")
            # Unpaired plans record what their boundaries observed, without the strict set.
            phases.update(vision_ms=1., prefill_ms=2., denoise_ms=3.) if paired else phases.update(denoise_ms=3.)
            return torch.zeros(1, horizon, 7)

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
        assert env["RPU_PI05_KVINSERT_PAD16"] == "1"
        assert env["RPU_PI05_DENOISE_UNROLL"] == "1"
        assert env["RPU_PI05_SIGLIP_W8A16"] == str(int(precision != "fp16"))
        assert env["RPU_PI05_ADARMS_DENSE_W8A16"] == str(int(precision != "fp16"))
        serial = (execution or {}).get("components", {}).get("vision_encoder", {}).get("vision", {}).get("chunk_size") == 256
        packed = not serial
        assert env["RPU_PI05_SIGLIP_BATCH"] == str(int(packed))
        assert env["RPU_PI05_RING_XOR3"] == str(int(paired))
        assert env["RPU_PI05_PREFILL_FULL_CHUNK_W8A8"] == str(int(full_chunk))
        event("environment")

    monkeypatch.setattr(runner, "bind_cold_environment", bind_environment)
    monkeypatch.setattr(runner, "profile_libraries", lambda: ({}, {}))
    monkeypatch.setattr(runner, "runtime_payloads", lambda libraries: {"extension": "/native.so"})
    monkeypatch.setattr(runner, "configure_text_profile", lambda *args: event("configure") or {})
    monkeypatch.setattr(runner, "configure_two_camera_policy", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "validate_camera_batch", lambda *args, **kwargs: event("validate"))
    threads = [8]
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    monkeypatch.setattr(torch, "set_num_threads", lambda count: threads.__setitem__(0, count))
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads[0])
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda *args: None)

    @contextmanager
    def observe(policy):
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
        assert report["numerical_notes"][0] == (
            "No CPU numerical comparison is recorded; robot/task acceptance is not run.")
        assert report["timing"][0]["samples"] == [row["total_ms"] for row in report["samples_ms"]]
        assert report["timing"][0]["unit"] == "ms" and report["timing"][0]["statistic"] == "mean"
        assert report["artifacts"]["report"] == str(tmp_path / "result/report.json")
        assert len(report["samples_ms"]) == 3 and report["warmup"] == 2
        assert report["cpu_runtime"]["torch_num_threads"] == 3
        assert report["cpu_runtime"]["OMP_NUM_THREADS"] == "3"
        assert report["execution_route"] == (
            "full_prefix_w8a8_tp8" if full_chunk else "paired_tp8" if paired else "generic")
        assert report["prefill_full_chunk_w8a8"] is full_chunk
        if full_chunk:
            assert report["prefill_chunk_size"] == cameras * 256 + 32
        assert report["precision"] == precision
        assert report["action_nvfp4_geglu_acc16"] is (
            bound_environment["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] == "1")
        if paired:
            assert report["mean_ms"]["denoise_ms"] == 3.
        else:
            assert report["timed_phases"] == ["total_ms", "denoise_ms"]
            assert set(report["mean_ms"]) == {"total_ms", "denoise_ms"}
            assert [item["metric"] for item in report["timing"]] == ["total_latency", "denoise_latency"]
        assert report["formal_certification"] is False and report["correctness"] == "not_run"
        assert report["registered_profile"] is None
        assert report["rpu_execution"] == expected_execution
        assert report["instruction_residency"] == instruction_residency
        assert report["runtime_payloads"]["extension"] == "/native.so"
        assert report["graph_prepare"]["phase"] == "READY"
        assert report["graph_prepare"]["ready_replay_validated"] is True
        assert torch.load(tmp_path / "result/actions.pt", weights_only=True).shape == (1, horizon, 7)
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


@pytest.mark.parametrize("cameras,horizon", [(2, 10), (2, 50), (3, 10), (3, 50)])
def test_full_prefix_cold_setup_reaches_public_measurement_and_report(monkeypatch, tmp_path, cameras, horizon):
    test_measurement_preserves_cold_execution_and_closes_before_reporting(
        monkeypatch, tmp_path, "w8a16", {"prefill": {"chunk_size": cameras * 256 + 32}},
        True, None, full_chunk=True, cameras=cameras, horizon=horizon)


@pytest.mark.parametrize("selector", ["prefill_w8a8", "qkv_w8a8", "mlp_w8a8"])
def test_t192_does_not_expand_w8a8_owner_domain(selector):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = template("w8a16")
    config["input"]["text_tokens"] = 192
    if selector == "prefill_w8a8":
        config["pi05"][selector] = True
    else:
        config["rpu_execution"]["components"] = {"vision_encoder": {"vision": {selector: True}}}
    with pytest.raises(ValueError, match="T192.*W8A8"):
        runner.example_arguments(config)


@pytest.mark.parametrize("execution", [
    {"action": {"linear_acc32": "true"}},
    {"vision": {"linear_acc32": 1}},
    {"prefill": {"linear_acc32": None}},
    {"action": {"unknown": True}},
    {"decode": {"linear_acc32": True}},
    {"components": {"unknown": {"action": {"linear_acc32": True}}}},
    {"components": {"vision_encoder": {"action": {"linear_acc32": True}}}},
    {"model": {"num_cores": 5}},
    {"prefill": {"chunk_size": 128}},
])
def test_accumulation_config_rejects_before_environment_or_loader(monkeypatch, execution):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = common.load_config(CONFIGS / "h50/w8a16_action_nvfp4.toml")
    args = runner.example_arguments(config)
    config["rpu_execution"] = execution
    with pytest.raises((TypeError, ValueError)):
        runner.example_arguments(config)
    args.rpu_execution = execution
    monkeypatch.setattr(runner, "bind_cold_environment",
                        lambda env: pytest.fail("invalid execution config reached cold environment binding"))
    with pytest.raises((TypeError, ValueError)):
        runner.run_profile(args, bind_libraries=True)


@pytest.mark.parametrize("cpu_reference_status", [None, "SKIPPED", "MEASURED"])
def test_known_numeric_failures_are_not_promoted_by_performance(cpu_reference_status):
    runner = load("examples/runners/pi05/optimized.py")
    notes = runner.numerical_notes(cameras=3, text_tokens=128, valid_tokens=128,
                                  precision="w4a16", prefill_w8a8=False,
                                  cpu_reference_status=cpu_reference_status)
    assert ("record-only" in notes[0]) == (cpu_reference_status == "MEASURED")
    assert ("No CPU numerical comparison" in notes[0]) == (cpu_reference_status != "MEASURED")
    assert "robot/task acceptance is not run" in notes[0]
    assert any("NO-GO" in note for note in notes)
    notes = runner.numerical_notes(cameras=2, text_tokens=96, valid_tokens=65,
                                  precision="fp16", prefill_w8a8=False,
                                  cpu_reference_status=cpu_reference_status)
    assert any("NO-GO" in note for note in notes)


@pytest.mark.parametrize("filename,inputs,execution", [
    ("w4a16_kvint8", {"cameras": 2}, {}),
    ("w4a16_kvint8", {}, {"model": {"num_cores": 6}}),
    ("fp16", {}, {"model": {"num_cores": 5}}),
    ("fp16", {}, {"model": {"num_cores": 4}, "vision": {"chunk_size": 512}}),
    ("fp16", {}, {"model": {"num_cores": 4}, "vision": {"linear_acc32": True, "chunk_size": 768}}),
    ("fp16", {}, {"vision": {"chunk_size": 64}}),
    ("fp16", {}, {"action": {"chunk_size": 32}}),
])
def test_core_and_physical_chunk_mismatches_fail_before_runtime(filename, inputs, execution):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = common.load_config(CONFIGS / "h50/fp16.toml")
    config["pi05"]["precision"] = filename
    config["input"].update(inputs)
    config["rpu_execution"] = execution
    with pytest.raises(ValueError):
        runner.example_arguments(config)


@pytest.mark.parametrize("filename,cores,inputs", [
    ("w8a16_action_nvfp4", 4, {"text_tokens": 64}),
    ("fp16", 4, {"cameras": 2, "text_tokens": 64}),
    ("w8a16", 6, {"text_tokens": 160}),
    ("w8a16_prefill_w8a8_action_nvfp4", 6, {"text_tokens": 64}),
])
def test_reduced_longer_text_resolves_the_generic_route_without_a_profile(filename, cores, inputs):
    """TP4/TP6 T64-T160 keep the unpaired generic plan; no registered profile applies."""
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = common.load_config(CONFIGS / "h50/fp16.toml")
    config["pi05"]["precision"] = filename
    config["input"].update(inputs)
    config["rpu_execution"] = {"model": {"num_cores": cores}}
    args = runner.example_arguments(config)
    assert runner.resolve_execution(args) == ({"model": {"num_cores": cores}}, False, "auto")
    assert runner.registered_example(config) is None


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("cameras", [2, 3])
def test_reduced_action_nvfp4_resolves_the_generic_route_without_a_profile(cores, cameras):
    """Action NVFP4 at TP4/TP6 is an unpaired generic plan and profile-less."""
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = template("w8a16_action_nvfp4", cameras=cameras)
    config["rpu_execution"]["model"]["num_cores"] = cores
    args = runner.example_arguments(config)
    assert args.precision == "w4a16" and not args.prefill_w8a8
    assert runner.resolve_execution(args) == ({"model": {"num_cores": cores}}, False, "auto")
    assert common.example_metadata(config) is None


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("precision", ["w8a16_prefill_w8a8_action_nvfp4", "w8a16"])
def test_reduced_prefill_w8a8_binds_the_cold_selector_without_a_profile(monkeypatch, cores, cameras, precision):
    """Prefill W8A8 at TP4/TP6 is the generic tile route, selected by one cold bit."""
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = template(precision, cameras=cameras)
    if precision == "w8a16":
        config["pi05"]["prefill_w8a8"] = True
    # This template leaves the budget at its documented default of eight.
    config["rpu_execution"] = {"model": {"num_cores": cores}}
    args = runner.example_arguments(config)
    assert args.precision == ("w8a16" if precision == "w8a16" else "w4a16") and args.prefill_w8a8
    assert runner.resolve_execution(args) == ({"model": {"num_cores": cores}}, False, "auto")
    assert common.example_metadata(config) is None

    class StoppedBeforeLoad(Exception):
        pass

    captured = []

    def bind_environment(env):
        captured.append(dict(env))
        raise StoppedBeforeLoad

    monkeypatch.setattr(runner, "bind_cold_environment", bind_environment)
    with pytest.raises(StoppedBeforeLoad):
        runner.run_profile(args)
    env = captured[0]
    assert env["RPU_PI05_PREFILL_GATEUP_W8A8"] == "1"
    # Reduced budgets admit raw-SPM packed Vision + PAD16 while keeping TP8-only XOR3/owner-LN off.
    enabled = {name for name in (*runner._ON, *runner._OFF) if env["RPU_PI05_" + name] == "1"}
    assert enabled == {
        "FUSED_DENOISE", "DENOISE_UNROLL", "KVINSERT_PAD16",
        "SIGLIP_BATCH", "VISION_MERGE_SEGMENT", "VISION_KV_SPM_ONLY",
        "VISION_Q_RESIDENT", "PREFILL_MASK_FP16",
    }


@pytest.mark.parametrize("precision", ["fp16", "w8a16"])
@pytest.mark.parametrize("cores", [4, 6])
def test_two_camera_reduced_cores_resolve_serial_generic_route(monkeypatch, precision, cores):
    """2cam TP4/TP6 admits packed 2x256 raw-SPM SigLIP4 and serial 256 override."""
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    config = template(precision, cameras=2)
    config["rpu_execution"]["model"]["num_cores"] = cores
    args = runner.example_arguments(config)
    assert runner.resolve_execution(args) == ({"model": {"num_cores": cores}}, False, "auto")
    assert common.example_metadata(config)["profile_id"].endswith(f".{precision}.vla-2cam-tp{cores}")
    serial = deepcopy(config)
    serial["rpu_execution"]["vision"] = {"chunk_size": 256}
    runner.example_arguments(serial)
    packed = deepcopy(config)
    packed["rpu_execution"]["vision"] = {"chunk_size": 512}
    runner.example_arguments(packed)
    wrong_packed = deepcopy(config)
    wrong_packed["rpu_execution"]["vision"] = {"chunk_size": 768}
    with pytest.raises(ValueError, match="Vision chunk must be one of"):
        runner.example_arguments(wrong_packed)

    class StoppedBeforeLoad(Exception):
        pass

    captured = []

    def bind_environment(env):
        captured.append(dict(env))
        raise StoppedBeforeLoad

    monkeypatch.setattr(runner, "bind_cold_environment", bind_environment)
    with pytest.raises(StoppedBeforeLoad):
        runner.run_profile(args)
    env = captured[0]
    enabled = {name for name in (*runner._ON, *runner._OFF) if env["RPU_PI05_" + name] == "1"}
    assert enabled == {
        "FUSED_DENOISE", "DENOISE_UNROLL", "KVINSERT_PAD16",
        "SIGLIP_BATCH", "VISION_MERGE_SEGMENT", "VISION_KV_SPM_ONLY",
        "VISION_Q_RESIDENT", "PREFILL_MASK_FP16",
    }
    assert env["RPU_PI05_SIGLIP_W8A16"] == env["RPU_PI05_ADARMS_DENSE_W8A16"] == str(int(precision == "w8a16"))
    assert env["RPU_PI05_PREFILL_GATEUP_W8A8"] == env["RPU_PI05_DENOISE_NVFP4_GEGLU_ACC16_M50"] == "0"


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
    config = load("examples/common/config.py").load_config(CONFIGS / "h50/w8a16.toml")
    config["pi05"]["instruction_residency"] = value
    with pytest.raises(ValueError, match="instruction_residency"):
        runner.example_arguments(config)


def test_each_registered_recipe_selects_the_route_that_quantizes_its_modules(monkeypatch, tmp_path):
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    family = common._backend_leaf("quant/precision_recipes.py").load(common.PRECISION_RECIPES)["pi05"]
    for name, item in family.recipes.items():
        precision, prefill_w8a8, vision_w4a16 = runner.recipe_route(item.selector)
        if precision == "w4a16_kvint8":
            # Whole-model per-channel INT4 with K/V INT8 runs the generic kernels, not the paired environment.
            assert (item.spec("prefill").key(), item.spec("prefill", "K").key()) == ("w4a16-channel", "w8a16"), name
            continue
        env = runner.profile_environment(precision, prefill_w8a8=prefill_w8a8, vision_w4a16=vision_w4a16)
        vision = item.spec("vision").key()
        assert env["RPU_PI05_SIGLIP_W4A16"] == str(int(vision == "w4a16-g32-rtn")), name
        # Vision W4 packs over the W8 install of all six projections.
        assert env["RPU_PI05_SIGLIP_W8A16"] == str(int(vision_w4a16 or item.spec("vision").format == "w8a16")), name
        assert env["RPU_PI05_ADARMS_DENSE_W8A16"] == str(int(item.spec("action", "AdaRMS").format == "w8a16")), name
        assert env["RPU_PI05_PREFILL_GATEUP_W8A8"] == str(int(item.spec("prefill", "Gate", 0).format == "w8a8")), name
    # A registered mix that no route implements fails before any backend import.
    data = json.loads(common.PRECISION_RECIPES.read_text(encoding="utf-8"))
    data["families"]["pi05"]["recipes"]["w8a16_action_nvfp4"]["rules"][0]["spec"] = "fp16"
    path = tmp_path / "precision_recipes.v1.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(common, "PRECISION_RECIPES", path)
    with pytest.raises(ValueError, match="Pi0.5 has no route for precision recipe pi05/w8a16_action_nvfp4"):
        runner.recipe_route("w8a16_action_nvfp4")
    with pytest.raises(ValueError, match="no route for precision recipe"):
        common.load_config(CONFIGS / "h50/w8a16_action_nvfp4.toml")


@pytest.mark.parametrize("precision", ["fp16", "w8a16"])
@pytest.mark.parametrize("cores", [4, 6])
def test_kuavo_t192_reduced_cores_keep_auto_chunk_while_tp8_fallback_uses_320(monkeypatch, precision, cores):
    """T192 at TP4/TP6 stays on 'auto' (avoiding 320-row Temp SPM overflow on TP4), while TP8 unpaired uses 320."""
    common = load("examples/common/config.py")
    runner = load("examples/runners/pi05/optimized.py")
    base = common.load_config(CONFIGS / f"kuavo/wholebody/{precision}.toml")
    tp8_args = runner.example_arguments(deepcopy(base))
    _, tp8_paired, tp8_chunk = runner.resolve_execution(tp8_args)
    assert tp8_paired is True and tp8_chunk == 480

    tp8_unpaired_cfg = deepcopy(base)
    tp8_unpaired_cfg["rpu_execution"] = {"vision": {"chunk_size": 256}}
    _, tp8_unpaired, tp8_unpaired_chunk = runner.resolve_execution(runner.example_arguments(tp8_unpaired_cfg))
    assert tp8_unpaired is False and tp8_unpaired_chunk == 320

    reduced_cfg = deepcopy(base)
    reduced_cfg["rpu_execution"] = {"model": {"num_cores": cores}}
    args = runner.example_arguments(reduced_cfg)
    assert runner.resolve_execution(args) == ({"model": {"num_cores": cores}}, False, "auto")

    class StoppedBeforeLoad(Exception):
        pass

    captured = []

    def bind_environment(env):
        captured.append(dict(env))
        raise StoppedBeforeLoad

    monkeypatch.setattr(runner, "bind_cold_environment", bind_environment)
    with pytest.raises(StoppedBeforeLoad):
        runner.run_profile(args)
    env = captured[0]
    assert env["RPU_PI05_VISION_LN_STREAM"] == "0"
    assert env["RPU_PI05_FUSED_DENOISE"] == "1"
    assert env["RPU_PI05_VISION_KV_SPM_ONLY"] == "1"
