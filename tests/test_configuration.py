from __future__ import annotations

import contextlib
from pathlib import Path
import hashlib
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
    banned = ("/nfs", "/data/", "10.10.", "192.168.", "hf_", "token=")
    for path in configs:
        text = path.read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        assert parsed
        assert not any(value in text for value in banned), path


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
    config = ROOT / "examples" / "configs" / "qwen3_0_6b_full.toml"
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
        r'["\']((?:RPU|QWEN3|LKN|HALO|WALL_OSS|HF|HUGGINGFACE)_[A-Z0-9_]+)["\']'
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

    @contextlib.contextmanager
    def hw_profile(_output_dir, *, max_dumps):
        assert max_dumps == 2
        events.append("hw-enter")
        try:
            yield
        finally:
            events.append("hw-exit")

    def target(*_args, **_kwargs):
        events.append("target")
        raise RuntimeError("target failed")

    monkeypatch.setattr(torch.profiler, "profile", profile)
    monkeypatch.setattr(torch.rpu, "hw_perf_trace", hw_profile)
    run_profiled.__globals__["_run_target"] = target
    config = {
        "runner": {
            "target": "causal_lm",
            "torch_profile": {
                "enabled": True,
                "output": str(tmp_path / "torch.json"),
            },
            "hw_profile": {
                "enabled": True,
                "output_dir": str(tmp_path / "hw"),
                "max_dumps": 2,
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
        "hw-enter",
        "target",
        "hw-exit",
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
                "RPU_ALLREDUCE_RING": True,
                "LKN_MAX_BATCH_ENTRIES": 131072,
            },
            {
                "RPU_LINGBOT2_DENOISE_UNROLL": "0",
                "RPU_ALLREDUCE_RING": "True",
                "LKN_MAX_BATCH_ENTRIES": "131072",
            },
        ),
        (
            "wall_oss",
            {
                "RPU_WALL_OSS_FUSED_DENOISE": 1,
                "RPU_ALLREDUCE_TWOSTAGE": False,
            },
            {
                "RPU_WALL_OSS_FUSED_DENOISE": "1",
                "RPU_ALLREDUCE_TWOSTAGE": "False",
            },
        ),
        (
            "hy_embodied",
            {
                "RPU_HY_VLA_W8A16": "expert",
                "RPU_ALLREDUCE_TWOSTAGE": 1,
            },
            {
                "RPU_HY_VLA_W8A16": "expert",
                "RPU_ALLREDUCE_TWOSTAGE": "1",
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
