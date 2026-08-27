#!/usr/bin/env python3
"""Run the public source-only LingBot-VLA-V2 integration."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "lingbot2.toml"
ENV_KEYS = {
    "RPU_LOG_LEVEL",
    "RPU_LINGBOT2_ALLOW_UNVALIDATED",
}
POLICY_ENV_KEYS = {
    "RPU_LINGBOT2_ALLOW_UNVALIDATED",
}


def _runner_env(config: dict) -> dict:
    env = config.get("runner", {}).get("env", {})
    if not isinstance(env, dict):
        raise ValueError("[runner.env] must be a table")
    unknown = sorted(set(env) - ENV_KEYS)
    if unknown:
        raise ValueError(f"unsupported LingBot2 [runner.env] keys: {unknown}")
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
    if model.get("controlled_evaluation_acknowledged") is not True:
        raise ValueError("[model].controlled_evaluation_acknowledged must be true")
    if model.get("dtype") not in {"fp16", "w8a16", "w4a16"}:
        raise ValueError("[model].dtype must be fp16, w8a16, or w4a16")
    if (
        model.get("n_cameras", 3) != 3
        or model.get("image_size", 256) != 256
        or model.get("max_lang_tokens", 72) != 72
    ):
        raise ValueError(
            "this public example uses n_cameras=3, image_size=256, and "
            "max_lang_tokens=72"
        )
    images = request.get("images")
    if not isinstance(images, list) or len(images) != 3 or not all(
        isinstance(value, str) and value for value in images
    ):
        raise ValueError("[request].images must contain exactly three paths")
    if not isinstance(request.get("instruction"), str) or not request["instruction"]:
        raise ValueError("[request].instruction must be a non-empty string")
    proprioception = request.get("proprioception", [])
    if not isinstance(proprioception, list) or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in proprioception
    ):
        raise ValueError(
            "[request].proprioception must be a finite numeric array"
        )
    noise_seed = request.get("noise_seed", 0)
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, int):
        raise ValueError("[request].noise_seed must be an integer")
    env = _runner_env(config)
    if _env_text(env.get("RPU_LINGBOT2_ALLOW_UNVALIDATED", False)) != "1":
        raise ValueError(
            "source-only evaluation requires "
            "[runner.env].RPU_LINGBOT2_ALLOW_UNVALIDATED = true"
        )
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
        print(f"configuration OK: {args.config} ({location}; source-only)")
        return 0

    env = {name: _env_text(value) for name, value in _runner_env(config).items()}
    os.environ.update(env)

    from PIL import Image
    from rpu_backend.api import Lingbot2Policy
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )
    image_paths = [Path(value).expanduser() for value in config["request"]["images"]]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"LingBot2 image file(s) not found: {', '.join(missing)}")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    policy = Lingbot2Policy.from_checkpoint(
        checkpoint,
        dtype=str(model_config["dtype"]),
        n_cameras=3,
        image_size=256,
        max_lang_tokens=int(model_config.get("max_lang_tokens", 72)),
        runtime_env={name: env[name] for name in POLICY_ENV_KEYS},
    ).to("rpu")
    try:
        output = policy.infer(
            images=images,
            instruction=config["request"]["instruction"],
            proprioception=config["request"].get("proprioception"),
            noise_seed=int(config["request"].get("noise_seed", 0)),
        )
        print(
            f"normalized_actions={tuple(output.actions_normalized.shape)} "
            f"latency_ms={output.latency_ms:.3f} robot_certified=false"
        )
    finally:
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
