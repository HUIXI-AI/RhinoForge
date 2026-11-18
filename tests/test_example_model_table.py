"""Unified [model] example layout and continued loading of the legacy layout."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "tests/fixtures/legacy_examples"


def _common():
    spec = importlib.util.spec_from_file_location(
        "examples_common_model_table", ROOT / "examples/common/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _common()


@pytest.mark.parametrize("legacy, unified", [
    pytest.param(
        '[example]\nprofile_id = "rhinovla.fp16.official-nonrtc"\nregistry_alias = "a"\ntarget = "rhinovla"\n'
        'dtype = "fp16"\n[example.opt_in]\n[input]\ncheckpoint = "/ckpt"\nnum_steps = 10\n',
        '[model]\nprofile_id = "rhinovla.fp16.official-nonrtc"\nregistry_alias = "a"\ntarget = "rhinovla"\n'
        'precision = "fp16"\ncheckpoint = "/ckpt"\n[model.opt_in]\n[input]\nnum_steps = 10\n',
        id="profile-bound"),
    pytest.param(
        '[example]\nprofile_id = "rhinovla.v3.w8a16.expert.trusted-factory"\ntarget = "rhinovla"\n'
        'dtype = "w8a16"\n[example.opt_in]\nRPU_RHINOVLA_FULL_W8A16 = "0"\nRPU_RHINOVLA_EXPERT_W8A16 = "0"\n'
        '[input]\nexpert_w8a16 = true\nfull_w8a16 = false\n',
        '[model]\nprofile_id = "rhinovla.v3.w8a16.expert.trusted-factory"\ntarget = "rhinovla"\n'
        'precision = "w8a16_action"\n',
        id="recipe-sets-switches"),
    pytest.param(
        '[pi05]\nprecision = "fp16"\n[input]\ncheckpoint = "pi05-libero-finetuned"\ncameras = 3\n',
        '[model]\ntarget = "pi05"\nprecision = "fp16"\ncheckpoint = "pi05-libero-finetuned"\n'
        '[input]\ncameras = 3\n',
        id="dedicated"),
    pytest.param(
        '[g05]\nprecision = "w8a16"\nlinear_accumulation = "acc16"\n[input]\nrelease_root = "/r"\n',
        '[model]\ntarget = "g05"\nprecision = "w8a16"\nlinear_acc32 = false\nrelease_root = "/r"\n',
        id="g05-accumulation"),
    pytest.param(
        '[g05]\nprecision = "w8a16_nvfp4"\n[input]\nrelease_root = "/r"\n',
        '[model]\ntarget = "g05"\nprecision = "w8a16_action_nvfp4"\nrelease_root = "/r"\n',
        id="recipe-name-to-selector"),
    pytest.param(
        '[wall_oss]\nprecision = "w8a16"\n[input]\nfp16_checkpoint = "/f"\nw8_checkpoint = "/w"\n',
        '[model]\ntarget = "wall_oss"\nprecision = "w8a16"\nfp16_checkpoint = "/f"\nw8_checkpoint = "/w"\n',
        id="wall-two-checkpoints"),
    pytest.param(
        '[qwen3_asr]\nprecision = "fp16"\ncheckpoint = "/asr"\n[input]\naudio = "a.wav"\n',
        '[model]\ntarget = "qwen3_asr"\nprecision = "fp16"\ncheckpoint = "/asr"\n[input]\naudio = "a.wav"\n',
        id="speech-keeps-model-table"),
])
def test_unified_layout_normalizes_to_the_legacy_runner_layout(legacy, unified):
    old = tomllib.loads(legacy)
    assert COMMON.normalize_config(old) is old
    assert COMMON.normalize_config(tomllib.loads(unified)) == old


@pytest.mark.parametrize("text, message", [
    ('[model]\ntarget = "pi05"\n[pi05]\nprecision = "fp16"\n', "cannot be combined"),
    ('[model]\nprecision = "fp16"\n', "target is required"),
    ('[model]\ntarget = "qwen3"\nprecision = "fp16"\n', "requires \\[model\\].profile_id"),
    ('[model]\ntarget = "pi05"\ncheckpoint = "a"\n[input]\ncheckpoint = "b"\n', "belongs in \\[model\\]"),
    ('[model]\nprofile_id = "p"\ntarget = "qwen3"\ndtype = "fp16"\n', "uses precision"),
    ('[model]\ntarget = "g05"\nprecision = "fp16"\nlinear_acc32 = true\nlinear_accumulation = "acc32"\n',
     "legacy G0.5 name"),
    ('[model]\ntarget = "g05"\nprecision = "fp16"\nlinear_acc32 = "acc32"\n', "must be boolean"),
    ('[model]\ntarget = "pi05"\n', "must name a registered precision recipe"),
    ('[model]\ntarget = "pi05"\nprecision = "w3a16"\n', "registered pi05 recipe"),
    # The runner selector is not a recipe name.
    ('[model]\ntarget = "g05"\nprecision = "w8a16_nvfp4"\n', "registered g05 recipe"),
    ('[model]\nprofile_id = "rhinovla.fp16.official-nonrtc"\ntarget = "rhinovla"\nprecision = "w8a16_pipeline"\n',
     "not registered for profile"),
    ('[model]\nprofile_id = "rhinovla.v3.fp16.trusted-factory"\ntarget = "rhinovla"\nprecision = "fp16"\n'
     '[model.opt_in]\nRPU_RHINOVLA_FULL_W8A16 = "0"\n', "set by precision recipe"),
    ('[model]\nprofile_id = "rhinovla.v3.w8a16.expert.trusted-factory"\ntarget = "rhinovla"\n'
     'precision = "w8a16_action"\n[input]\nexpert_w8a16 = true\n', "set by precision recipe"),
])
def test_unified_layout_rejects_ambiguous_tables(text, message):
    with pytest.raises(COMMON.ConfigError, match=message):
        COMMON.normalize_config(tomllib.loads(text))


@pytest.mark.parametrize("legacy", sorted(LEGACY.glob("*.toml")), ids=lambda path: path.stem)
def test_legacy_files_still_load_as_the_same_entry(legacy):
    # Each fixture is the pre-[model] copy of the template with the same flattened path;
    # a template merged into another TOML is that TOML with the preset of its old input.
    configs = ROOT / "examples/configs"
    templates = {path.relative_to(configs).as_posix(): (path, None) for path in configs.rglob("*.toml")}
    merged = json.loads((configs / "config_names.json").read_text(encoding="utf-8"))["merged"]
    templates.update({old.removeprefix("examples/configs/"): (ROOT / target["config"], target["input"])
                      for old, target in merged.items()})
    template, preset = next(value for name, value in templates.items() if name.replace("/", "_") == legacy.name)
    assert "model" in tomllib.loads(template.read_text())
    old = COMMON.load_config(legacy)
    # The legacy file names its own input file, of the preset's shape.
    files = {key: old["input"][key] for key in ("batch", "observation") if preset and key in old["input"]}
    new = COMMON.load_config(template, overrides={"input": files} if files else None, input_preset=preset)
    assert COMMON.config_target(old) == COMMON.config_target(new)
    old_meta, new_meta = COMMON.example_metadata(old), COMMON.example_metadata(new)
    assert (old_meta or {}).get("profile_id") == (new_meta or {}).get("profile_id")


def test_every_template_uses_the_unified_layout():
    for path in (ROOT / "examples/configs").rglob("*.toml"):
        raw = tomllib.loads(path.read_text())
        assert "model" in raw and not (COMMON.LEGACY_IDENTITY_TABLES | {"env"}) & set(raw), path


def test_rhinovla_v3_source_entry_runs_through_run_py_with_exact_switches(tmp_path):
    path = ROOT / "examples/configs/rhinovla/v3/controlled/w8a16_pipeline.toml"
    config = COMMON.load_config(path)
    profile = COMMON.example_profiles()[config["example"]["profile_id"]]
    assert profile["example_only"] and profile["variant"] == "v3-source"
    assert config["example"]["opt_in"] == {gate["name"]: gate["value"] for gate in profile["controlled_gates"]}
    assert {"source_dir", "checkpoint", "fixture"} <= set(config["input"])
    check = subprocess.run([sys.executable, str(ROOT / "examples/run.py"), "--config", str(path), "--check-config"],
                           cwd=ROOT, capture_output=True, text=True, check=False)
    assert check.returncode == 0 and "configuration OK" in check.stdout, check.stderr
    changed = tmp_path / "changed.toml"
    changed.write_text(path.read_text().replace('RPU_RHINOVLA_VISION_BATCH_VIEWS = "1"', 'RPU_RHINOVLA_VISION_BATCH_VIEWS = "0"'))
    with pytest.raises(COMMON.ConfigError, match="exact opt_in"):
        COMMON.load_config(changed)
    fp16 = tmp_path / "fp16.toml"
    fp16.write_text(path.read_text().replace('precision = "w8a16_pipeline"', 'precision = "fp16"'))
    with pytest.raises(COMMON.ConfigError):
        COMMON.load_config(fp16)


SOURCE_DIR = ROOT / "examples/configs/rhinovla/v3/controlled"


def test_rhinovla_v3_source_profiles_pin_every_trusted_factory_switch():
    # Unpinned switches would be taken from the caller's shell: an exported
    # EXPERT_W8A16=1 would build W8 weights under an fp16 profile.
    profiles = COMMON.example_profiles()
    names = [gate["name"] for gate in
             profiles["rhinovla.v3.w8a16.full-pipeline.trusted-factory"]["controlled_gates"]]
    for path in sorted(SOURCE_DIR.glob("*.toml")):
        config = COMMON.load_config(path)
        assert sorted(config["example"]["opt_in"]) == sorted(names), path


def test_rhinovla_v3_source_rejects_an_inherited_conflicting_switch(monkeypatch):
    from examples.common import measurement

    config = COMMON.load_config(SOURCE_DIR / "fp16.toml")
    # Register every switch with monkeypatch so the runner's os.environ writes are undone.
    for name, value in config["example"]["opt_in"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RPU_RHINOVLA_EXPERT_W8A16", "1")
    # Rejected before Torch or the backend is imported.
    with pytest.raises(ValueError, match="RPU_RHINOVLA_EXPERT_W8A16 conflicts"):
        measurement.run(config)


def _source_runner(monkeypatch, fixture, scales, full_inventory):
    import types

    module = importlib.import_module("examples.runners.rhinovla.runner")
    factory = types.SimpleNamespace(V3_SOURCE_SHA="src", V3_CHECKPOINT_SHA256="ckpt")
    built = {}

    class Policy:
        runtime = types.SimpleNamespace(er="expert")

        @classmethod
        def from_factory(cls, name, config, rpu_execution=None):
            built.update(name=name, config=config)
            return cls()

        def predict(self, **request):
            return request

    stubs = {
        "rpu_backend": types.ModuleType("rpu_backend"),
        "rpu_backend.api": types.SimpleNamespace(RhinoVLAPolicy=Policy),
        "rpu_backend.adapters.rhinovla.factory": factory,
        "rpu_backend.adapters.rhinovla.convert": types.SimpleNamespace(_gather_expert_scales=lambda er: scales),
        "rpu_backend.adapters.rhinovla.precision": types.SimpleNamespace(full_w8_inventory=full_inventory),
    }
    for name, stub in stubs.items():
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(module, "_load_mapping", lambda inputs, key: fixture)
    return module, built


def _fixture(**changes):
    metadata = {"source_commit": "src", "checkpoint_sha256": "ckpt", "dtype": "float32", "steps": 10,
                "rtc": False, "action_dim": 96, "action_horizon": 30, **changes}
    return {"metadata": metadata, "cases": [{"qwen_inputs": {}, "request": {"x0": 1}}]}


@pytest.mark.parametrize("stem, expert, full", [
    ("fp16", False, False), ("w8a16_action", True, False), ("w8a16_pipeline", True, True),
])
def test_rhinovla_v3_source_passes_and_confirms_the_profile_precision(monkeypatch, stem, expert, full):
    import types

    inventory = []
    module, built = _source_runner(monkeypatch, _fixture(), ["scale"] if expert else [],
                                   lambda runtime: inventory.append(runtime))
    runner = types.SimpleNamespace(source_config=COMMON.load_config(SOURCE_DIR / f"{stem}.toml"), owner=None)
    infer = module.prepare(runner)
    assert built["config"]["expert_w8a16"] is expert and built["config"]["full_w8a16"] is full
    assert inventory == ([runner.owner.runtime] if full else [])
    assert infer() == {"x0": 1}


def test_rhinovla_v3_source_rejects_wrong_fixture_or_installed_precision(monkeypatch):
    import types

    config = COMMON.load_config(SOURCE_DIR / "fp16.toml")
    module, _ = _source_runner(monkeypatch, _fixture(action_dim=72), [], lambda runtime: None)
    with pytest.raises(ValueError, match="fixture does not match"):
        module.prepare(types.SimpleNamespace(source_config=config, owner=None))
    module, _ = _source_runner(monkeypatch, _fixture(), ["scale"], lambda runtime: None)
    with pytest.raises(ValueError, match="expert precision does not match"):
        module.prepare(types.SimpleNamespace(source_config=config, owner=None))
