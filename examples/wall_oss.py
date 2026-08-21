#!/usr/bin/env python3
"""Run one Wall-OSS action-chunk inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "wall_oss.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    request = config["request"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if not isinstance(request.get("images"), list) or not request["images"]:
        raise ValueError("[request].images must be a non-empty array")
    if not isinstance(request.get("instruction"), str):
        raise ValueError("[request].instruction must be a string")
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
    from rpu_backend.api import WallOssPolicy
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )

    image_paths = [Path(value).expanduser() for value in config["request"]["images"]]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Wall-OSS image file(s) not found: {', '.join(missing)}")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    policy = WallOssPolicy.from_pretrained(
        checkpoint,
        dataset_key=model_config.get("dataset_key", "berkeley_autolab_ur5"),
        camera_names=tuple(model_config.get("camera_names", ["face_view"])),
        action_horizon=int(model_config.get("action_horizon", 32)),
        num_steps=int(model_config.get("num_steps", 10)),
        max_seq_len=int(model_config.get("max_seq_len", 2048)),
        rpu_execution=config.get("rpu_execution"),
    ).to("rpu")
    output = policy.infer(
        images=images,
        instruction=config["request"]["instruction"],
        proprioception=config["request"]["proprioception"],
        noise_seed=int(config["request"].get("noise_seed", 0)),
    )
    print(
        f"actions={tuple(output.actions.shape)} action_hz={output.action_hz:g} "
        f"latency_ms={output.latency_ms:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
