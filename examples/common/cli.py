"""Command-line entry for one validated example configuration."""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import tomllib

from examples.common.config import (
    ConfigError, load_config, example_profiles, _g05_runner, _pi05_runner,
    _hyvla_runner, _lingbot2_runner, _wall_base_runner, _wall_sorting_runner,
)
from examples.common.config import _CORE_BUDGETS as CORE_BUDGETS
from examples.common.overrides import CALLER_TABLES, caller_overrides, recipe_path, toml_text


def check_cell(source, cores=None, input_preset=None, assignments=()):
    """The `--check-config` decision of one grid cell: None when accepted, else the reason it reports."""
    args = argparse.Namespace(config=source, check_config=True, hwperf=False, profile=False)
    try:
        config = load_config(source, overrides=caller_overrides(assignments, cores), input_preset=input_preset)
        with contextlib.redirect_stdout(io.StringIO()):
            _dispatch(config, args)
    except (ConfigError, OSError, tomllib.TOMLDecodeError) as exc:
        return str(exc)
    return None


def main(argv=None, *, keep_logging_until_exit=False) -> int:
    parser = argparse.ArgumentParser(description="Run one RhinoForge model configuration.")
    parser.add_argument("--config", type=Path, required=True,
                        help="A template, or with --recipe the model directory that holds it.")
    parser.add_argument("--recipe", metavar="NAME",
                        help="Select the template NAME.toml of the same model directory; "
                             "the error lists the directory's templates.")
    parser.add_argument("--check-config", action="store_true",
                        help="Check syntax/profile without importing Torch or the device backend.")
    parser.add_argument("--hwperf", "--hw-perf", dest="hwperf", action="store_true",
                        help="Collect hardware tracing in the result directory when supported.")
    parser.add_argument("--profile", action="store_true",
                        help="Collect an additional diagnostic profile when supported.")
    parser.add_argument("--model", type=Path,
                        help="Override a Qwen3.5 MoE checkpoint path; its profile remains validated.")
    parser.add_argument("--cores", type=int, choices=CORE_BUDGETS,
                        help="Override rpu_execution.model.num_cores; a refused budget reports its reason.")
    parser.add_argument("--input", dest="input_preset", metavar="PRESET",
                        help="Apply a named [input] preset of this model family from examples/inputs/; "
                             "'canonical' names the family's reference input. --set input.* fields win.")
    parser.add_argument("--set", dest="assignments", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help=f"Override one field of {', '.join(CALLER_TABLES)}; repeatable. "
                             "VALUE is a TOML literal; bare words are strings.")
    parser.add_argument("--emit-toml", type=Path, metavar="PATH",
                        help="Write the resolved configuration; running it reproduces this configuration.")
    args = parser.parse_args(argv)
    try:
        args.config = recipe_path(args.config.expanduser().resolve(), args.recipe)
        config = load_config(args.config,
                             overrides=caller_overrides(args.assignments, args.cores),
                             input_preset=args.input_preset)
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
        if args.emit_toml is not None:
            if "_model_override" in config:
                raise ConfigError("--emit-toml cannot record --model; write the checkpoint into the TOML")
            try:
                text = toml_text(config)
            except ValueError as exc:
                raise ConfigError(f"--emit-toml: {exc}") from exc
            args.emit_toml.expanduser().write_text(text, encoding="utf-8")
        if args.check_config:
            return _dispatch(config, args)
        from examples.common.logging import logging_session
        with logging_session(config, keep_until_exit=keep_logging_until_exit):
            return _dispatch(config, args)
    except (ConfigError, OSError, tomllib.TOMLDecodeError) as exc:
        parser.error(str(exc))


def _dispatch(config, args):
    from examples.common.logging import configuration_checked
    if "qwen3_asr" in config or "qwen3_tts" in config:
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
