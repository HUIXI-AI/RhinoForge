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
import tomllib

import pytest

from rpu_backend.api._execution import normalize_rpu_execution


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXAMPLE_CONFIGS = {
    "dinov3_vit_b.toml",
    "g05.toml",
    "gemma4.toml",
    "gr00t.toml",
    "hy_embodied.toml",
    "internvla_navdp.toml",
    "lingbot2.toml",
    "llama_3_2_1b.toml",
    "pi05_libero.toml",
    "pi05_libero_w4.toml",
    "pi05_libero_w8a16.toml",
    "qwen3_0_6b.toml",
    "qwen3_14b_w8a16.toml",
    "qwen3_1_7b.toml",
    "qwen3_4b.toml",
    "qwen3_5_0_8b.toml",
    "qwen3_5_2b.toml",
    "qwen3_5_4b.toml",
    "qwen3_5_9b.toml",
    "qwen3_5_vision_2b.toml",
    "qwen3_5_vision_4b.toml",
    "qwen3_8b.toml",
    "qwen3_vl_2b.toml",
    "qwen3_vl_32b_w8a16.toml",
    "qwen3_vl_4b.toml",
    "qwen3_vl_8b.toml",
    "rhinovla.toml",
    "siglip.toml",
    "wall_oss.toml",
    "wall_oss_w4a16.toml",
    "wall_oss_w8a16.toml",
}
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


def test_example_toml_files_are_portable() -> None:
    config_dir = ROOT / "examples" / "configs"
    configs = sorted(config_dir.glob("*.toml"))
    assert configs
    banned = (
        "/" + "nfs",
        "/" + "data/",
        "10." + "10.",
        "192." + "168.",
        "hf" + "_",
        "token" + "=",
    )
    for path in configs:
        text = path.read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        assert parsed
        assert not any(value in text for value in banned), path


def test_example_configs_are_the_single_public_profile_set() -> None:
    config_dir = ROOT / "examples" / "configs"
    actual = {path.name for path in config_dir.glob("*.toml")}
    assert actual == EXPECTED_EXAMPLE_CONFIGS
    assert not list(config_dir.glob("*_full.toml"))


def test_example_configs_exclude_retired_runtime_options() -> None:
    for path in (ROOT / "examples" / "configs").glob("*.toml"):
        text = path.read_text(encoding="utf-8")
        found = {name for name in RETIRED_RUNTIME_OPTIONS if name in text}
        assert not found, (path, found)


def test_every_runner_target_has_a_valid_toml() -> None:
    namespace = runpy.run_path(str(ROOT / "examples" / "run_model.py"))
    load_config = namespace["load_config"]
    run_target = namespace["_run_target"]
    targets = set(namespace["TARGETS"])
    covered = set()
    for path in sorted((ROOT / "examples" / "configs").glob("*.toml")):
        target = load_config(path)["runner"]["target"]
        run_target(target, path, check_config=True)
        covered.add(target)
    assert covered == targets


def test_qwen3_8b_profile_uses_auto_prefill_chunk() -> None:
    profile = tomllib.loads(
        (ROOT / "examples" / "configs" / "qwen3_8b.toml").read_text(
            encoding="utf-8"
        )
    )
    assert profile["rpu_execution"]["prefill"] == {
        "chunk_size": "auto",
        "padding_budget": 64,
    }


def test_internvla_example_uses_only_the_public_navdp_checkpoint() -> None:
    profile = tomllib.loads(
        (ROOT / "examples" / "configs" / "internvla_navdp.toml").read_text(
            encoding="utf-8"
        )
    )
    assert profile["model"]["alias"] == "internvla-n1-navdp"
    assert profile["model"]["source_only_acknowledged"] is True
    assert "navdp_checkpoint" not in profile["model"]
    assert profile["request"]["seed"] == 0

    assert not (
        ROOT / "python" / "rpu_backend" / "adapters" / "internvla_n1" / "nextdit.py"
    ).exists()
    assert not (ROOT / "src" / "fused" / "rpu_internvla_nextdit_model.cpp").exists()

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "python" / "rpu_backend" / "adapters" / "internvla_n1"
        ).glob("*.py")
    )
    assert "model.navdp." in source


def test_internvla_public_checkpoint_is_pinned_and_confined(tmp_path: Path) -> None:
    from rpu_backend.adapters.internvla_n1._checkpoint import (
        NAVDP_PREFIX,
        open_public_checkpoint,
    )

    config = tmp_path / "config.json"
    index = tmp_path / "model.safetensors.index.json"
    shard = tmp_path / "model-00001-of-00001.safetensors"
    config.write_text(
        json.dumps({"model_type": "internvla_n1", "system1": "navdp_async"}),
        encoding="utf-8",
    )
    index.write_text(
        json.dumps({"weight_map": {NAVDP_PREFIX + "weight": shard.name}}),
        encoding="utf-8",
    )
    shard.write_bytes(b"opaque-test-shard")

    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {path: digest(path) for path in (config, index, shard)}
    _, names, assets = open_public_checkpoint(
        tmp_path,
        manifest,
        prefixes=(NAVDP_PREFIX,),
        controlled_rpu=False,
    )
    assert names == (NAVDP_PREFIX + "weight",)
    assert assets == (config, index, shard)

    index.write_text(
        json.dumps({"weight_map": {NAVDP_PREFIX + "weight": "../escape.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes checkpoint_dir"):
        open_public_checkpoint(
            tmp_path,
            {config: digest(config), index: digest(index)},
            prefixes=(NAVDP_PREFIX,),
            controlled_rpu=False,
        )


def test_qwen3_vl_examples_cover_single_and_multiple_images(tmp_path: Path) -> None:
    load_config = runpy.run_path(str(ROOT / "examples" / "qwen3_vl.py"))[
        "load_config"
    ]
    single = load_config(ROOT / "examples" / "configs" / "qwen3_vl_2b.toml")
    multiple = load_config(ROOT / "examples" / "configs" / "qwen3_vl_4b.toml")
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


def test_wall_oss_w8a16_example_binds_fp16_source(tmp_path: Path) -> None:
    load_config = runpy.run_path(str(ROOT / "examples" / "wall_oss.py"))[
        "load_config"
    ]
    config = load_config(
        ROOT / "examples" / "configs" / "wall_oss_w8a16.toml"
    )
    assert config["model"]["precision"] == "w8a16"
    assert config["model"]["fp16_alias"] == "wall-oss-0.5"

    missing_source = tmp_path / "missing-source.toml"
    missing_source.write_text(
        """
[model]
alias = "wall-oss-0.5-w8a16"
precision = "w8a16"
[request]
images = ["frame.jpg"]
instruction = "move"
proprioception = [0.0]
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fp16_alias or fp16_checkpoint"):
        load_config(missing_source)


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
    scripts = sorted(
        path
        for path in (ROOT / "examples").glob("*.py")
        if path.name != "verify_install.py"
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


def test_rhinovla_placeholder_rejects_without_native_import(tmp_path: Path) -> None:
    guard = tmp_path / "guard"
    (guard / "rpu_backend").mkdir(parents=True)
    (guard / "rpu_backend" / "__init__.py").write_text(
        "raise AssertionError('rpu_backend imported')\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(guard)
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "examples" / "rhinovla.py")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Set [model].runtime_factory" in result.stderr
    assert "rpu_backend imported" not in result.stderr


@pytest.mark.parametrize(
    "body,expected",
    [
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
    config = ROOT / "examples" / "configs" / "qwen3_0_6b.toml"
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


def test_runtime_configuration_covers_source_environment_readers() -> None:
    pattern = re.compile(
        r'["\']((?:RPU|QWEN3|LKN|WALL_OSS|HF|HUGGINGFACE)_[A-Z0-9_]+)["\']'
    )
    source_names = set()
    for base in (ROOT / "python" / "rpu_backend", ROOT / "src"):
        for path in base.rglob("*"):
            if path.suffix in {
                ".py", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".inc"
            }:
                source_names.update(pattern.findall(path.read_text(encoding="utf-8")))

    text = (ROOT / "docs" / "runtime_config.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([A-Z][A-Z0-9_]+)` \|", text, re.MULTILINE)
    documented = set(rows)
    non_environment_constants = {
        "HF_ARCH",
        "QWEN3_VL_TEXT_ARCH",
        "QWEN3_VL_VISION_ARCH",
    }

    assert len(rows) == len(documented), "duplicate runtime-configuration row"
    assert source_names - non_environment_constants == documented


def test_runtime_configuration_chinese_mirror_keeps_variable_order() -> None:
    row_pattern = r"^\| `([A-Z][A-Z0-9_]+)` \|"
    english_text = (ROOT / "docs" / "runtime_config.md").read_text(
        encoding="utf-8"
    )
    chinese_text = (ROOT / "docs" / "runtime_config.zh.md").read_text(
        encoding="utf-8"
    )
    english = re.findall(
        row_pattern,
        english_text,
        re.MULTILINE,
    )
    chinese = re.findall(
        row_pattern,
        chinese_text,
        re.MULTILINE,
    )
    assert english
    assert chinese == english

    def table_code_tokens(text: str) -> list[list[str]]:
        return [
            re.findall(r"`([^`]+)`", line)
            for line in text.splitlines()
            if line.startswith("|") and "`" in line
        ]

    assert table_code_tokens(chinese_text) == table_code_tokens(english_text)
    assert len(re.findall(r"^#{2,3} ", chinese_text, re.MULTILINE)) == len(
        re.findall(r"^#{2,3} ", english_text, re.MULTILINE)
    )


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


@pytest.mark.parametrize("policy_name", ["lingbot2", "wall_oss", "hy_embodied"])
@pytest.mark.parametrize(
    "key",
    ["LD_PRELOAD", "PYTHONPATH", "RPU_BACKEND_SO", "HF_TOKEN", "RPU_TEST_ENV"],
)
def test_policy_runtime_env_rejects_unknown_keys_before_mutation(
    policy_name: str, key: str, monkeypatch
) -> None:
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="unsupported environment key"):
        _policy_with_runtime_env(policy_name, {key: "value"})
    assert key not in os.environ


@pytest.mark.parametrize(
    "policy_name,runtime_env,expected",
    [
        (
            "lingbot2",
            {
                "RPU_LINGBOT2_DENOISE_UNROLL": 0,
                "RPU_FASTREPLAY_SKIP_SYNC": True,
                "LKN_MAX_BATCH_ENTRIES": 131072,
            },
            {
                "RPU_LINGBOT2_DENOISE_UNROLL": "0",
                "RPU_FASTREPLAY_SKIP_SYNC": "True",
                "LKN_MAX_BATCH_ENTRIES": "131072",
            },
        ),
        (
            "wall_oss",
            {
                "RPU_WALL_OSS_FUSED_DENOISE": 1,
                "RPU_FASTREPLAY_SKIP_SYNC": False,
            },
            {
                "RPU_WALL_OSS_FUSED_DENOISE": "1",
                "RPU_FASTREPLAY_SKIP_SYNC": "False",
            },
        ),
        (
            "hy_embodied",
            {
                "RPU_HY_VLA_W8A16": "expert",
                "RPU_HY_VLA_DENOISE_UNROLL": 1,
            },
            {
                "RPU_HY_VLA_W8A16": "expert",
                "RPU_HY_VLA_DENOISE_UNROLL": "1",
            },
        ),
    ],
)
def test_policy_runtime_env_accepts_exact_keys_and_stringifies_values(
    policy_name: str, runtime_env, expected
) -> None:
    policy = _policy_with_runtime_env(policy_name, runtime_env)
    assert policy._runtime_env == expected


@pytest.mark.parametrize("policy_name", ["wall_oss", "hy_embodied"])
@pytest.mark.parametrize(
    "key",
    ["LKN_MAX_BATCH_ENTRIES", "LKN_KD_BUF_MB", "LKN_INSTR_BUF_MB"],
)
def test_policy_runtime_env_requires_lkn_capacity_before_import(
    policy_name: str, key: str
) -> None:
    with pytest.raises(ValueError, match="before importing rpu_backend"):
        _policy_with_runtime_env(policy_name, {key: "64"})


@pytest.mark.parametrize(
    "runtime_env",
    [
        [],
        {1: "value"},
        {"": "value"},
        {"A=B": "value"},
        {"GOOD": "value\0tail"},
    ],
)
def test_runtime_env_normalizer_rejects_unrepresentable_inputs(runtime_env) -> None:
    from rpu_backend.api._runtime_env import normalize_runtime_env

    with pytest.raises(ValueError, match="runtime_env"):
        normalize_runtime_env(
            runtime_env,
            owner="TestPolicy",
            allowed={"GOOD"},
        )


def test_hy_norm_stats_pickle_requires_opt_in_and_exact_hash(
    tmp_path: Path, monkeypatch
) -> None:
    import rpu_backend.api.hy_embodied as hy_embodied

    payload = pickle.dumps({"action_mean": [0], "action_std": [1]})
    path = tmp_path / "norm_stats.pkl"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    with pytest.raises(PermissionError, match="trust_norm_stats_pickle=True"):
        hy_embodied._load_trusted_norm_stats_pickle(
            path,
            trust_norm_stats_pickle=False,
            norm_stats_sha256=digest,
        )

    original_loads = pickle.loads
    called = False

    def unexpected_loads(_payload):
        nonlocal called
        called = True
        raise AssertionError("pickle.loads ran before hash verification")

    monkeypatch.setattr(hy_embodied.pickle, "loads", unexpected_loads)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        hy_embodied._load_trusted_norm_stats_pickle(
            path,
            trust_norm_stats_pickle=True,
            norm_stats_sha256="0" * 64,
        )
    assert not called

    monkeypatch.setattr(hy_embodied.pickle, "loads", original_loads)
    assert hy_embodied._load_trusted_norm_stats_pickle(
        path,
        trust_norm_stats_pickle=True,
        norm_stats_sha256=digest.upper(),
    ) == {"action_mean": [0], "action_std": [1]}


def test_hy_norm_stats_and_tokenizer_are_fail_closed(monkeypatch) -> None:
    from transformers import AutoTokenizer
    from rpu_backend.api import HyEmbodiedPolicy

    with pytest.raises(PermissionError, match="norm_stats_path requires"):
        HyEmbodiedPolicy.from_checkpoint(
            "unused/checkpoint", norm_stats_path="norm_stats.pkl"
        )

    policy = HyEmbodiedPolicy.from_checkpoint("unused/checkpoint")
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: kwargs,
    )
    assert policy._tokenizer()["trust_remote_code"] is False


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


def test_pi05_rejects_custom_model_code_before_optional_import() -> None:
    from rpu_backend.api import Pi05Policy

    with pytest.raises(ValueError, match="trust_remote_code=False"):
        Pi05Policy.from_pretrained(
            "unused/checkpoint", trust_remote_code=True
        )
