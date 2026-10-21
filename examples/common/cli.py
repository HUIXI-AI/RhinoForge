"""Command-line entry for one validated example configuration."""
from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

from examples.common.config import (
    ConfigError, load_config, example_profiles, _g05_runner, _pi05_runner,
    _hyvla_runner, _lingbot2_runner, _wall_base_runner, _wall_sorting_runner,
)


def main(argv=None, *, keep_logging_until_exit=False) -> int:
    parser = argparse.ArgumentParser(description="Run one RhinoForge model configuration.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-config", action="store_true",
                        help="Check syntax/profile without importing Torch or the device backend.")
    parser.add_argument("--hwperf", "--hw-perf", dest="hwperf", action="store_true",
                        help="Collect hardware tracing in the result directory when supported.")
    parser.add_argument("--profile", action="store_true",
                        help="Collect an additional diagnostic profile when supported.")
    parser.add_argument("--model", type=Path,
                        help="Override a Qwen3.5 MoE checkpoint path; its profile remains validated.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config.expanduser().resolve())
        if args.model is not None:
            profile = example_profiles().get(config.get("example", {}).get("profile_id"), {})
            if (config.get("example", {}).get("target") not in {"qwen3_5", "qwen3_5_vision"}
                    or profile.get("family") != "qwen3-5-moe"):
                raise ConfigError("--model is currently supported only by Qwen3.5 MoE text/image profiles")
            config["_model_override"] = str(args.model.expanduser().resolve())
        if "env" in config:
            raise ConfigError(
                "Run this RhinoVLA v3 configuration with "
                "scripts/validate_rhinovla_v3.py run")
        if "wall_oss" not in config and "wall_oss_sorting" not in config:
            for flag in ("hwperf", "profile"):
                config["run"][flag] = bool(getattr(args, flag) or config["run"].get(flag, False))
        if args.check_config:
            return _dispatch(config, args)
        from examples.common.logging import logging_session
        with logging_session(config, keep_until_exit=keep_logging_until_exit):
            return _dispatch(config, args)
    except (ConfigError, OSError, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))


def _dispatch(config, args):
    from examples.common.logging import configuration_checked
    if any(name in config for name in ("qwen3_asr", "qwen3_tts", "whisper_asr")):
        from examples.runners.qwen3_audio.runner import validate_example_config, run_example_config
        try:
            validate_example_config(config)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if args.check_config:
            configuration_checked(f"configuration OK: {args.config} (speech; syntax/input envelope only)")
            return 0
        return run_example_config(config)
    if "wall_oss" in config or "wall_oss_sorting" in config:
        if args.hwperf or args.profile:
            kind = "base" if "wall_oss" in config else "sorting"
            raise ConfigError(f"{kind} latency examples do not enable tracing; omit --hwperf/--profile")
        engine = _wall_base_runner() if "wall_oss" in config else _wall_sorting_runner()
        if args.check_config:
            configuration_checked(f"configuration OK: {args.config} (Wall-OSS; syntax only)")
            return 0
        return engine.run_example_config(config)
    if any(name in config for name in ("g05", "pi05", "hy_vla", "lingbot2")):
        if "g05" in config:
            engine = _g05_runner()
        elif "pi05" in config:
            engine = _pi05_runner()
        elif "lingbot2" in config:
            engine = _lingbot2_runner()
        else:
            engine = _hyvla_runner()
        try:
            engine.validate_example_config(config)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        if args.check_config:
            configuration_checked(f"configuration OK: {args.config} (syntax/input envelope only)")
            return 0
        return engine.run_example_config(config)
    if args.check_config:
        configuration_checked(f"configuration OK: {args.config} ({config['example']['profile_id']}; syntax/profile only)")
        return 0
    from examples.common.measurement import run
    return run(config)
