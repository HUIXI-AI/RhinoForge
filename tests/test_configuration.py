from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import pickle
import re
import runpy
import subprocess
import sys
import threading
import tomllib

import pytest

from rpu_backend.api._execution import normalize_rpu_execution


ROOT = Path(__file__).resolve().parents[1]


def _join(*parts: str) -> str:
    return "".join(parts)


RETIRED_RUNTIME_OPTIONS = {
    "RPU_ALLREDUCE_CHUNK_BALANCED",
    "RPU_ALLREDUCE_CHUNK_V2",
    "RPU_ALLREDUCE_FUSED",
    "RPU_ALLREDUCE_PARTIAL_TWOSTAGE",
    "RPU_ALLREDUCE_RING",
    "RPU_ALLREDUCE_TWOSTAGE",
    "RPU_ALLREDUCE_TWOSTAGE_TRACE",
    "RPU_ALLREDUCE_V2_CAP",
    "RPU_INTERNVLA_N1_EXACT_TWOSTAGE",
    "RPU_LINEAR_AUTOTILE",
    "RPU_LINEAR_AUTOTILE_LIB",
    "RPU_LINEAR_TILE_FORCE",
    "RPU_LINEAR_TILE_GY1",
    "RPU_LINEAR_TILE_M384N",
    "RPU_LINEAR_TILE_M384W",
    "RPU_LINEAR_TILE_M64",
    "RPU_LINEAR_TILE_VIS",
    "RPU_LINEAR_TILE_W8_FORCE",
    "RPU_LINEAR_TILE_W8_M384W",
    "RPU_LINEAR_TILE_W8_M64",
    "RPU_LINEAR_TILE_W8_M64N",
    "RPU_LINEAR_TILE_W8_VIS",
    "RPU_LINGBOT2_DENOISE_TWOSTAGE",
    "RPU_LINGBOT2_DOWN_ACC32_TILE",
    "RPU_LINGBOT2_OUTER_TWOSTAGE",
}


def test_execution_configuration_is_validated_and_frozen() -> None:
    config = normalize_rpu_execution(
        {
            "prefill": {
                "chunk_size": 128,
                "padding_budget": 64,
            }
        },
        entry_point="test",
        supported={"prefill": ("chunk_size", "padding_budget")},
    )
    assert dict(config["prefill"]) == {
        "chunk_size": 128,
        "padding_budget": 64,
    }
    with pytest.raises(TypeError, match="read-only"):
        config["prefill"].chunk_size = 64

    with pytest.raises(ValueError, match="positive multiple of 16"):
        normalize_rpu_execution(
            {"prefill": {"chunk_size": 17}},
            entry_point="test",
        )
    with pytest.raises(ValueError, match="conflicts"):
        normalize_rpu_execution(
            {"prefill": {"padding_rows": 16, "padding_budget": 16}},
            entry_point="test",
        )


def test_linear_acc32_requires_boolean_and_declared_owner_capability() -> None:
    supported = {
        "prefill": ("chunk_size", "linear_acc32"),
        "vision": ("chunk_size", "linear_acc32"),
    }
    config = normalize_rpu_execution(
        {"prefill": {"linear_acc32": True}, "vision": {"linear_acc32": False}},
        entry_point="test", supported=supported,
    )
    assert config["prefill"]["linear_acc32"] is True
    assert config["vision"]["linear_acc32"] is False

    with pytest.raises(TypeError, match="must be bool"):
        normalize_rpu_execution(
            {"prefill": {"linear_acc32": 1}},
            entry_point="test", supported=supported,
        )
    with pytest.raises(ValueError, match="explicit supported"):
        normalize_rpu_execution(
            {"vision": {"linear_acc32": True}}, entry_point="test",
        )
    with pytest.raises(ValueError, match="not supported"):
        normalize_rpu_execution(
            {"components": {"vision_encoder": {
                "vision": {"linear_acc32": True},
            }}},
            entry_point="test",
            supported_components={"vision_encoder": {"vision": ("chunk_size",)}},
        )


def test_example_toml_files_are_portable() -> None:
    config_dir = ROOT / "examples" / "configs"
    configs = sorted(config_dir.rglob("*.toml"))
    assert configs
    banned = (
        _join("/", "n", "fs"),
        _join("/", "da", "ta", "/"),
        _join("10", ".", "10", "."),
        _join("192", ".", "168", "."),
        _join("h", "f", "_"),
        _join("tok", "en", "="),
    )
    for path in configs:
        text = path.read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        assert parsed
        assert not any(value in text for value in banned), path


def test_example_configs_exclude_retired_runtime_options() -> None:
    for path in (ROOT / "examples" / "configs").rglob("*.toml"):
        text = path.read_text(encoding="utf-8")
        found = {name for name in RETIRED_RUNTIME_OPTIONS if name in text}
        assert not found, (path, found)


def test_example_catalog_uses_only_canonical_model_directories() -> None:
    import json

    root = ROOT / "examples/configs"
    assert not list(root.glob("*.toml"))
    directory_sets = {
        "qwen3": {"text"}, "qwen3_5": {"text", "vl", "legacy"},
        "qwen3_vl": {"text", "vision", "vl"}, "pi05": {"2cam", "3cam"},
        "lingbot": {"v2"}, "rhinovla": {"v1", "v3"}, "wall_oss": {"base"},
    }
    for family, expected in directory_sets.items():
        actual = {p.relative_to(root / family).parts[0]
                  for p in (root / family).rglob("*.toml")}
        assert actual == expected, (family, actual)
    referenced = set()
    for path in root.rglob("*.toml"):
        config = tomllib.loads(path.read_text())
        if "example" in config:
            referenced.add(config["example"]["profile_id"])
        if "pi05" in config:
            assert config.get("rpu_execution", {}).get("model", {}).get("num_cores", 8) == 8
    profiles = json.loads((root / "profiles.json").read_text())["profiles"]
    assert {p["id"] for p in profiles} == referenced


def test_every_runner_target_has_a_valid_toml() -> None:
    namespace = runpy.run_path(str(ROOT / "examples" / "run_model.py"))
    load_config = namespace["load_config"]
    run_target = namespace["_run_target"]
    targets = set(namespace["TARGETS"])
    covered = set()
    for path in sorted((ROOT / "examples" / "configs").rglob("*.toml")):
        config = load_config(path)
        if "example" in config or "pi05" in config:
            target = namespace["_catalog_module"]().config_metadata(config)["target"]
            target = {"qwen3_5": "qwen3_5_text"}.get(target, target)
        else:
            target = config["runner"]["target"]
        run_target(target, path, check_config=True)
        covered.add(target)
    assert covered == targets


@pytest.mark.parametrize("target", ["qwen3_5_text", "qwen3_5_vision"])
def test_runner_forwards_qwen35_cold_precision(target, tmp_path: Path) -> None:
    body = f'[runner]\ntarget = "{target}"\n[rpu_execution.prefill]\nlinear_acc32 = true\n'
    if target == "qwen3_5_vision":
        body += '[rpu_execution.vision]\nlinear_acc32 = false\nchunk_size = "auto"\n'
    path = tmp_path / "precision.toml"
    path.write_text(body)
    load_config = runpy.run_path(str(ROOT / "examples" / "run_model.py"))["load_config"]
    config = load_config(path)
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is True
    if target == "qwen3_5_vision":
        assert config["rpu_execution"]["vision"]["linear_acc32"] is False
    path.write_text(body.replace("linear_acc32 = true", "linear_acc32 = 1"))
    with pytest.raises(TypeError, match="must be bool"):
        load_config(path)


def test_runner_rejects_incompatible_pi_padding_before_loading(tmp_path: Path) -> None:
    path = ROOT / "examples" / "configs" / "pi05/2cam/w8a16_action_nvfp4.toml"
    load_config = runpy.run_path(str(ROOT / "examples" / "run_model.py"))["load_config"]
    load_config(path)
    invalid = tmp_path / "invalid-pi.toml"
    invalid.write_text(path.read_text() + "\n[rpu_execution.prefill]\npadding_budget = 64\n")
    with pytest.raises(ValueError, match="without extra padding"):
        load_config(invalid)


def test_qwen3_8b_profile_uses_auto_prefill_chunk() -> None:
    profile = tomllib.loads(
        (ROOT / "examples" / "configs" / "qwen3/text/8b/fp16.toml").read_text(
            encoding="utf-8"
        )
    )
    assert profile["rpu_execution"]["prefill"]["chunk_size"] == "auto"


@pytest.mark.parametrize("target,stages", [
    ("causal_lm", ("prefill",)),
    ("qwen3_vl", ("prefill", "vision")),
    ("qwen3_5_text", ("prefill",)),
    ("qwen3_5_vision", ("prefill", "vision")),
    ("pi05", ("prefill", "vision", "action")),
])
def test_runner_accepts_declared_cold_controls(target, stages, tmp_path):
    load = runpy.run_path(str(ROOT / "examples/run_model.py"))["load_config"]
    body = f'[runner]\ntarget = "{target}"\n[rpu_execution.model]\nnum_cores = 8\n'
    body += "".join(f"[rpu_execution.{stage}]\nlinear_acc32 = true\n" for stage in stages)
    path = tmp_path / "cold.toml"
    path.write_text(body)
    config = load(path)
    assert config["rpu_execution"]["model"]["num_cores"] == 8
    assert all(config["rpu_execution"][stage]["linear_acc32"] is True for stage in stages)
    path.write_text(body.replace("num_cores = 8", "num_cores = 5"))
    with pytest.raises(ValueError, match="num_cores"):
        load(path)
    path.write_text(body.replace("linear_acc32 = true", "linear_acc32 = 1"))
    with pytest.raises(TypeError, match="must be bool"):
        load(path)


def test_runner_qwen_vl_replay_is_prefill_only(tmp_path):
    load = runpy.run_path(str(ROOT / "examples/run_model.py"))["load_config"]
    path = tmp_path / "replay.toml"
    body = '[runner]\ntarget = "qwen3_vl"\n[rpu_execution.prefill]\nfast_replay = true\n'
    path.write_text(body)
    assert load(path)["rpu_execution"]["prefill"]["fast_replay"] is True
    path.write_text(body.replace(".prefill]", ".vision]"))
    with pytest.raises(ValueError, match="not supported"):
        load(path)


def test_pi_two_camera_a8_template_retains_optimized_admission(tmp_path):
    load = runpy.run_path(str(ROOT / "examples/run_model.py"))["load_config"]
    path = ROOT / "examples/configs/pi05/2cam/w8a16_prefill_w8a8_action_nvfp4.toml"
    config = load(path)
    helpers = runpy.run_path(str(ROOT / "examples/_policy_examples.py"))
    assert config["pi05"]["precision"] == "w8a16_prefill_w8a8_action_nvfp4"
    assert helpers["_execution"](config)["prefill"]["chunk_size"] == 272
    bad = tmp_path / "pi.toml"
    bad.write_text(path.read_text() + "\n[rpu_execution.model]\nnum_cores = 4\n")
    with pytest.raises(ValueError, match="TP4/TP6"):
        load(bad)


def test_qwen35_text_uses_checkpoint_aware_loader(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import torch
    import transformers
    from rpu_backend.api import RPUModelForConditionalGeneration

    class LoadedCheckpoint(Exception):
        pass

    seen = {}
    def exact_loader(checkpoint, **kwargs):
        seen.update(checkpoint=checkpoint, **kwargs)
        raise LoadedCheckpoint

    monkeypatch.setattr(RPUModelForConditionalGeneration, "from_pretrained", exact_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw:
                        lambda *a, **kw: SimpleNamespace(input_ids=torch.ones(1, 3, dtype=torch.int64)))
    # Caller-owned earlier-format files remain accepted without shipping
    # duplicate templates in the public catalog.
    path = tmp_path / "qwen35-text.toml"
    path.write_text('''[runner]
target = "qwen3_5_text"
[model]
checkpoint = "/path/to/qwen3.5-2b-w8a16"
local_files_only = true
[generation]
prompt = "Explain what a compiler does."
max_new_tokens = 8
[rpu_execution.model]
num_cores = 8
''')
    monkeypatch.setattr(sys, "argv", ["qwen3_5_text.py", "--config", str(path)])
    main = runpy.run_path(str(ROOT / "examples/qwen3_5_text.py"))["main"]
    with pytest.raises(LoadedCheckpoint):
        main()
    assert seen["checkpoint"] == "/path/to/qwen3.5-2b-w8a16"
    assert seen["device"] is None
    assert seen["trust_remote_code"] is False
    assert seen["rpu_execution"]["model"]["num_cores"] == 8


def test_qwen3_vl_examples_cover_single_and_multiple_images(tmp_path: Path) -> None:
    load_config = runpy.run_path(str(ROOT / "examples" / "qwen3_vl.py"))[
        "load_config"
    ]
    # Test the earlier public parser independently from the canonical catalog.
    base = '[model]\nalias = "qwen3-vl-2b"\n[request]\nprompt = "describe"\nmax_new_tokens = 1\n'
    single_path, multi_path = tmp_path / "single.toml", tmp_path / "multi.toml"
    single_path.write_text(base + 'image = "one.jpg"\n')
    multi_path.write_text(base + 'images = ["one.jpg", "two.jpg"]\n')
    single, multiple = load_config(single_path), load_config(multi_path)
    assert "image" in single["request"]
    assert len(multiple["request"]["images"]) == 2

    ambiguous = tmp_path / "ambiguous.toml"
    ambiguous.write_text(
        """
[model]
alias = "qwen3-vl-2b"
[request]
image = "one.jpg"
images = ["two.jpg"]
prompt = "describe"
max_new_tokens = 1
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_config(ambiguous)


def test_model_example_check_config_does_not_import_native(tmp_path: Path) -> None:
    guard = tmp_path / "guard"
    (guard / "rpu_backend").mkdir(parents=True)
    (guard / "rpu_backend" / "__init__.py").write_text(
        "raise AssertionError('rpu_backend imported')\n", encoding="utf-8"
    )
    (guard / "torch.py").write_text(
        "raise AssertionError('torch imported')\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(guard)
    env.pop("RPU_KERNEL_LIB_PATH", None)
    # Include every explicit template, not only each script's default. An
    # optimized profile must not bootstrap Torch during syntax validation.
    validate_all = subprocess.run(
        [sys.executable, "-S", "-c",
         "from pathlib import Path; import runpy; "
         "load = runpy.run_path('examples/run_model.py')['load_config']; "
         "[load(p) for p in Path('examples/configs').rglob('*.toml')]"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=30,
    )
    assert validate_all.returncode == 0, validate_all.stderr
    scripts = sorted(
        path
        for path in (ROOT / "examples").glob("*.py")
        if path.name != "verify_install.py" and not path.name.startswith("_")
    )
    assert scripts
    for script in scripts:
        result = subprocess.run(
            [sys.executable, "-S", str(script), "--check-config"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, (
            script.name,
            result.stdout,
            result.stderr,
        )
        assert "configuration OK" in result.stdout


@pytest.mark.parametrize(
    "body,expected",
    [
        ("[run]\nwarmup = 1\nruns = 3", "require the"),
        ("[input]\ndecode_steps = 32", "require the"),
        ("[runner.env]\nRPU_NOT_A_DOCUMENTED_OPTION = 1", "not documented"),
        ("[runner.env]\nRPU_KERNEL_LIB_PATH = 'x'", "not allowed"),
        ("[runner.env]\nRPU_API_TOKEN = 'x'", "not allowed"),
        ("[runner.env]\nLD_PRELOAD = 'x'", "not documented"),
        (
            "[runner.env]\nRPU_PI05_LOAD_NOISE = 'noise.pt'",
            "diagnostic-only",
        ),
        (
            "[rpu_execution.vision]\nchunk_size = 256",
            "not supported by this model entry point",
        ),
    ],
)
def test_full_runner_rejects_unsafe_configuration(
    tmp_path: Path, body: str, expected: str
) -> None:
    config = tmp_path / "invalid.toml"
    config.write_text(
        f"""
[runner]
target = "qwen3_5_text"

{body}

[model]
alias = "qwen3_5-0.8b"

[generation]
prompt = "hello"
max_new_tokens = 1
""".strip(),
        encoding="utf-8",
    )
    load_config = runpy.run_path(str(ROOT / "examples" / "run_model.py"))[
        "load_config"
    ]
    with pytest.raises(ValueError, match=expected):
        load_config(config)


def test_full_runner_applies_environment_only_for_execution(tmp_path: Path) -> None:
    script = ROOT / "examples" / "run_model.py"
    config = tmp_path / "runner-env.toml"
    config.write_text('[runner]\ntarget = "causal_lm"\n[runner.env]\nRPU_LOG_LEVEL = 3\n')
    code = f"""
import os, runpy, sys
ns = runpy.run_path({str(script)!r})
events = []
def target(name, path, *, check_config):
    events.append((check_config, os.environ.get('RPU_LOG_LEVEL')))
    assert 'torch' not in sys.modules
    assert 'rpu_backend' not in sys.modules
ns['main'].__globals__['_run_target'] = target
os.environ['RPU_LOG_LEVEL'] = 'sentinel'
sys.argv = [{str(script)!r}, '--config', {str(config)!r}, '--check-config']
assert ns['main']() == 0
assert os.environ['RPU_LOG_LEVEL'] == 'sentinel'
sys.argv = [{str(script)!r}, '--config', {str(config)!r}]
assert ns['main']() == 0
assert events == [(True, 'sentinel'), (False, '3')]
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_full_runner_accepts_documented_huggingface_cache_environment(
    tmp_path: Path,
) -> None:
    config = tmp_path / "hf-cache.toml"
    config.write_text(
        """
[runner]
target = "qwen3_5_text"

[runner.env]
HF_HOME = "/tmp/huggingface"

[model]
alias = "qwen3_5-0.8b"

[generation]
prompt = "hello"
max_new_tokens = 1
""".strip(),
        encoding="utf-8",
    )
    load_config = runpy.run_path(str(ROOT / "examples" / "run_model.py"))[
        "load_config"
    ]
    assert load_config(config)["runner"]["env"]["HF_HOME"] == "/tmp/huggingface"


def test_full_runner_profile_defaults_and_exception_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    import torch

    namespace = runpy.run_path(str(ROOT / "examples" / "run_model.py"))
    run_profiled = namespace["_run_profiled"]
    events = []
    kwargs = {}

    class Profiler:
        def __enter__(self):
            events.append("torch-enter")
            return self

        def __exit__(self, *_args):
            events.append("torch-exit")

        def export_chrome_trace(self, _path):
            events.append("torch-export")

    def profile(**values):
        kwargs.update(values)
        return Profiler()

    def target(*_args, **_kwargs):
        events.append("target")
        raise RuntimeError("target failed")

    monkeypatch.setattr(torch.profiler, "profile", profile)
    run_profiled.__globals__["_run_target"] = target
    config = {
        "runner": {
            "target": "causal_lm",
            "torch_profile": {
                "enabled": True,
                "output": str(tmp_path / "torch.json"),
            },
        }
    }
    with pytest.raises(RuntimeError, match="target failed"):
        run_profiled(config, tmp_path / "unused.toml")
    assert kwargs["record_shapes"] is False
    assert kwargs["profile_memory"] is False
    assert kwargs["with_stack"] is False
    assert events == [
        "torch-enter",
        "target",
        "torch-exit",
        "torch-export",
    ]


def _policy_with_runtime_env(name: str, runtime_env):
    from rpu_backend.api import (
        HyEmbodiedPolicy,
        Lingbot2Policy,
        WallOssPolicy,
    )

    if name == "lingbot2":
        return Lingbot2Policy.from_checkpoint(
            "unused/checkpoint", runtime_env=runtime_env
        )
    if name == "wall_oss":
        return WallOssPolicy.from_checkpoint(
            "unused/checkpoint", runtime_env=runtime_env
        )
    return HyEmbodiedPolicy.from_checkpoint(
        "unused/checkpoint", runtime_env=runtime_env
    )


@pytest.mark.parametrize(
    "loader",
    [
        pytest.param("causal", id="causal"),
        pytest.param("conditional", id="conditional"),
    ],
)
@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_generic_loaders_reject_custom_model_code(loader: str, value) -> None:
    if loader == "causal":
        from rpu_backend import RPUModelForCausalLM as model_loader
    else:
        from rpu_backend.api import (
            RPUModelForConditionalGeneration as model_loader,
        )

    with pytest.raises(ValueError, match="trust_remote_code=False"):
        model_loader.from_pretrained(
            "unused/checkpoint", trust_remote_code=value
        )
