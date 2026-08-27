#!/usr/bin/env python3
"""Run Source-only GR00T with caller-preprocessed tensor inputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "gr00t.toml"
ENV_KEYS = {
    "RPU_LOG_LEVEL",
    "RPU_WARMUP",
    "RPU_GR00T_KVPAD16",
    "RPU_GR00T_PARTIAL_MROPE",
}


def _runner_env(config: dict) -> dict:
    env = config.get("runner", {}).get("env", {})
    if not isinstance(env, dict):
        raise ValueError("[runner.env] must be a table")
    unknown = sorted(set(env) - ENV_KEYS)
    if unknown:
        raise ValueError(f"unsupported GR00T [runner.env] keys: {unknown}")
    return env


def _env_text(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (str, int)):
        return str(value)
    raise ValueError("[runner.env] values must be strings, integers, or booleans")


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    request = config["request"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if not model.get("qwen3vl_alias") and not model.get("qwen3vl_checkpoint"):
        raise ValueError("[model] needs qwen3vl_alias or qwen3vl_checkpoint")
    if model.get("source_only_acknowledged") is not True:
        raise ValueError("[model].source_only_acknowledged must be true")
    embodiment_id = model.get("embodiment_id", 20)
    if isinstance(embodiment_id, bool) or not isinstance(embodiment_id, int):
        raise ValueError("[model].embodiment_id must be an integer")
    if not isinstance(request.get("input_pt"), str) or not request["input_pt"]:
        raise ValueError("[request].input_pt must be a non-empty string")
    if request.get("num_steps", 4) != 4:
        raise ValueError("[request].num_steps must be 4 for this runtime")
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("[request].seed must be an integer")
    _runner_env(config)
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    model_config = config["model"]
    if args.check_config:
        location = model_config.get("checkpoint") or model_config["alias"]
        print(f"configuration OK: {args.config} ({location}; Source-only)")
        return 0

    for name, value in _runner_env(config).items():
        os.environ[name] = _env_text(value)

    import torch
    from rpu_backend.adapters.gr00t import build_gr00t_vla
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )
    qwen3vl = model_config.get("qwen3vl_checkpoint") or str(
        model_path(model_config["qwen3vl_alias"])
    )
    input_path = Path(config["request"]["input_pt"]).expanduser()
    if not input_path.is_file():
        raise SystemExit(f"GR00T preprocessed input not found: {input_path}")
    inputs = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(inputs, Mapping):
        raise SystemExit("GR00T input .pt must contain a tensor mapping")
    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state")
    missing = [name for name in required if not isinstance(inputs.get(name), torch.Tensor)]
    if missing:
        raise SystemExit(f"GR00T input .pt is missing tensor fields: {missing}")

    runtime = build_gr00t_vla(
        checkpoint,
        qwen3vl,
        embodiment_id=model_config.get("embodiment_id", 20),
        rpu_execution=config.get("rpu_execution"),
    )
    try:
        action = runtime.get_action(
            dict(inputs),
            seed=config["request"].get("seed", 0),
            num_steps=config["request"].get("num_steps", 4),
        )
        print(f"normalized_action_shape={tuple(action.shape)}")
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
