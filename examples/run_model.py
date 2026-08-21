#!/usr/bin/env python3
"""Run a public model example with TOML-owned runtime and profiling settings."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import re
import runpy
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "qwen3_0_6b_full.toml"
TARGETS = {
    "causal_lm": "causal_lm.py",
    "dinov3": "dinov3.py",
    "hy_embodied": "hy_embodied.py",
    "pi05": "pi05.py",
    "qwen3_5_text": "qwen3_5_text.py",
    "qwen3_vl": "qwen3_vl.py",
    "rhinovla": "rhinovla.py",
    "wall_oss": "wall_oss.py",
}
_EXECUTION_SUPPORT = {
    "causal_lm": {"prefill": ("chunk_size", "padding_rows", "padding_budget")},
    "dinov3": {"vision": ("chunk_size",)},
    "hy_embodied": {},
    "pi05": {
        "prefill": ("chunk_size", "padding_rows", "padding_budget"),
        "vision": ("chunk_size",),
        "action": ("chunk_size",),
    },
    "qwen3_5_text": {
        "prefill": ("chunk_size", "padding_rows", "padding_budget")
    },
    "qwen3_vl": {
        "prefill": ("chunk_size", "padding_rows", "padding_budget"),
        "vision": ("chunk_size",),
    },
    "rhinovla": None,
    "wall_oss": {
        "prefill": ("chunk_size", "padding_rows", "padding_budget"),
        "vision": ("chunk_size",),
        "action": ("chunk_size",),
    },
}
_RUNNER_KEYS = {"target", "env", "torch_profile", "hw_profile"}
_TORCH_PROFILE_KEYS = {
    "enabled",
    "output",
    "record_shapes",
    "profile_memory",
    "with_stack",
}
_HW_PROFILE_KEYS = {"enabled", "output_dir", "max_dumps"}
_DENIED_ENV = {"RPU_KERNEL_LIB_PATH"}
_CREDENTIAL_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")


def _require_bool(table: dict, key: str, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"[runner].{key} must be a boolean")
    return value


def _unknown_keys(table: dict, allowed: set[str], label: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"{label} has unsupported key(s): {', '.join(unknown)}")


def _runtime_environment_inventory() -> tuple[set[str], set[str]]:
    text = (ROOT / "docs" / "runtime_config.md").read_text(encoding="utf-8")
    diagnostic_heading = "## Diagnostic-only inventory"
    controls_heading = "## Non-environment runtime controls"
    if diagnostic_heading not in text or controls_heading not in text:
        raise RuntimeError("docs/runtime_config.md is missing its inventory headings")
    diagnostic_text = text.split(diagnostic_heading, 1)[1].split(
        controls_heading, 1
    )[0]
    row_pattern = r"^\| `([A-Z][A-Z0-9_]+)` \|"
    return (
        set(re.findall(row_pattern, text, re.MULTILINE)),
        set(re.findall(row_pattern, diagnostic_text, re.MULTILINE)),
    )


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    runner = config.get("runner")
    if not isinstance(runner, dict):
        raise ValueError("configuration needs a [runner] table")
    _unknown_keys(runner, _RUNNER_KEYS, "[runner]")
    target = runner.get("target")
    if target not in TARGETS:
        raise ValueError(
            f"[runner].target must be one of: {', '.join(sorted(TARGETS))}"
        )
    execution = config.get("rpu_execution")
    if execution is not None:
        normalize = runpy.run_path(
            str(ROOT / "python" / "rpu_backend" / "api" / "_execution.py")
        )["normalize_rpu_execution"]
        normalize(
            execution,
            entry_point="examples/run_model.py",
            supported=_EXECUTION_SUPPORT[target],
        )

    env = runner.get("env", {})
    if not isinstance(env, dict):
        raise ValueError("[runner.env] must be a table")
    documented, diagnostic_only = _runtime_environment_inventory()
    for name, value in env.items():
        if name in _DENIED_ENV or any(marker in name for marker in _CREDENTIAL_MARKERS):
            raise ValueError(f"[runner.env].{name} is not allowed")
        if name not in documented:
            raise ValueError(
                f"[runner.env].{name} is not documented in docs/runtime_config.md"
            )
        if name in diagnostic_only:
            raise ValueError(
                f"[runner.env].{name} is diagnostic-only and cannot be set in TOML"
            )
        if not isinstance(value, (str, int, bool)):
            raise ValueError(
                f"[runner.env].{name} must be a string, integer, or boolean"
            )

    torch_profile = runner.get("torch_profile", {})
    if not isinstance(torch_profile, dict):
        raise ValueError("[runner.torch_profile] must be a table")
    _unknown_keys(torch_profile, _TORCH_PROFILE_KEYS, "[runner.torch_profile]")
    _require_bool(torch_profile, "enabled", False)
    for key in ("record_shapes", "profile_memory", "with_stack"):
        _require_bool(torch_profile, key, False)
    output = torch_profile.get("output", "profiles/torch_trace.json")
    if not isinstance(output, str) or not output:
        raise ValueError("[runner.torch_profile].output must be a path string")

    hw_profile = runner.get("hw_profile", {})
    if not isinstance(hw_profile, dict):
        raise ValueError("[runner.hw_profile] must be a table")
    _unknown_keys(hw_profile, _HW_PROFILE_KEYS, "[runner.hw_profile]")
    _require_bool(hw_profile, "enabled", False)
    output_dir = hw_profile.get("output_dir", "profiles/hw")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("[runner.hw_profile].output_dir must be a path string")
    max_dumps = hw_profile.get("max_dumps", 32)
    if not isinstance(max_dumps, int) or isinstance(max_dumps, bool) or max_dumps < 1:
        raise ValueError("[runner.hw_profile].max_dumps must be a positive integer")
    return config


def _apply_environment(env: dict) -> None:
    for name, value in env.items():
        if isinstance(value, bool):
            value = "1" if value else "0"
        os.environ[name] = str(value)


def _run_target(target: str, config_path: Path, *, check_config: bool) -> None:
    script = Path(__file__).with_name(TARGETS[target])
    argv = [str(script), "--config", str(config_path)]
    if check_config:
        argv.append("--check-config")
    previous = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    finally:
        sys.argv = previous


def _run_profiled(config: dict, config_path: Path) -> None:
    runner = config["runner"]
    torch_cfg = runner.get("torch_profile", {})
    hw_cfg = runner.get("hw_profile", {})
    torch_enabled = torch_cfg.get("enabled", False)
    hw_enabled = hw_cfg.get("enabled", False)

    profiler = None
    if torch_enabled or hw_enabled:
        import torch
    if torch_enabled:
        activities = [torch.profiler.ProfilerActivity.CPU]
        private_use = getattr(torch.profiler.ProfilerActivity, "PrivateUse1", None)
        if private_use is not None:
            activities.append(private_use)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=torch_cfg.get("record_shapes", False),
            profile_memory=torch_cfg.get("profile_memory", False),
            with_stack=torch_cfg.get("with_stack", False),
        )

    try:
        with contextlib.ExitStack() as stack:
            if profiler is not None:
                stack.enter_context(profiler)
            if hw_enabled:
                import rpu_backend  # noqa: F401 - installs torch.rpu

                output_dir = Path(hw_cfg.get("output_dir", "profiles/hw")).expanduser()
                output_dir.mkdir(parents=True, exist_ok=True)
                stack.enter_context(
                    torch.rpu.hw_perf_trace(
                        str(output_dir), max_dumps=hw_cfg.get("max_dumps", 32)
                    )
                )
            _run_target(runner["target"], config_path, check_config=False)
    finally:
        if profiler is not None:
            output = Path(torch_cfg.get("output", "profiles/torch_trace.json")).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(output))
            print(f"torch profile: {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--list-targets", action="store_true")
    args = parser.parse_args()
    if args.list_targets:
        print("\n".join(sorted(TARGETS)))
        return 0

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    runner = config["runner"]
    if args.check_config:
        _run_target(runner["target"], config_path, check_config=True)
        print(f"runner configuration OK: {config_path}")
        return 0

    _apply_environment(runner.get("env", {}))
    _run_profiled(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
