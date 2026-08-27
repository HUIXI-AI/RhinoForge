#!/usr/bin/env python3
"""Run the SigLIP vision component from a Pi0.5 model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "siglip.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    images = config["request"].get("images")
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if not isinstance(images, list) or not images or not all(
        isinstance(value, str) and value for value in images
    ):
        raise ValueError("[request].images must be a non-empty string array")
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

    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from rpu_backend.api import Pi05Policy
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )
    image_paths = [
        Path(value).expanduser() for value in config["request"]["images"]
    ]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"SigLIP image file(s) not found: {', '.join(missing)}")
    pixels = torch.stack([
        pil_to_tensor(
            Image.open(path).convert("RGB").resize(
                (224, 224), Image.Resampling.BILINEAR
            )
        ).float().div_(255)
        for path in image_paths
    ]).to(dtype=torch.float16, device="rpu")

    policy = Pi05Policy.from_pretrained(
        checkpoint,
        rpu_execution=config.get("rpu_execution"),
        trust_remote_code=False,
    ).to("rpu")
    vision_tower = (
        policy._lerobot_policy.model
        .paligemma_with_expert.paligemma.model.vision_tower
    )
    with torch.no_grad():
        output = vision_tower(pixels)
    features = (
        output.last_hidden_state
        if hasattr(output, "last_hidden_state")
        else output
    )
    print(f"last_hidden_state={tuple(features.shape)} dtype={features.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
