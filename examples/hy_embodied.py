#!/usr/bin/env python3
"""Run one normalized Hy-Embodied action-chunk inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "hy_embodied.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    request = config["request"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if not isinstance(request.get("images"), list) or len(request["images"]) != 3:
        raise ValueError("[request].images must contain the three UMI camera images")
    if not isinstance(request.get("instruction"), str):
        raise ValueError("[request].instruction must be a string")
    if config.get("rpu_execution"):
        raise ValueError("Hy-Embodied does not accept [rpu_execution]")
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
        print(f"configuration OK: {args.config} ({location})")
        return 0


    from PIL import Image
    from rpu_backend.api import HyEmbodiedPolicy
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )

    image_paths = [Path(value).expanduser() for value in config["request"]["images"]]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(
            f"Hy-Embodied image file(s) not found: {', '.join(missing)}"
        )
    images = [Image.open(path).convert("RGB") for path in image_paths]
    policy = HyEmbodiedPolicy.from_checkpoint(
        checkpoint,
        dtype=model_config.get("dtype", "fp16"),
        prefix_len=int(model_config.get("prefix_len", 240)),
    ).to("rpu")
    try:
        output = policy.infer(
            images=images,
            instruction=config["request"]["instruction"],
            state=config["request"].get("state"),
            noise_seed=int(config["request"].get("noise_seed", 0)),
        )
        print(
            f"actions_normalized={tuple(output.actions_normalized.shape)} "
            f"latency_ms={output.latency_ms:.3f}"
        )
    finally:
        policy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
