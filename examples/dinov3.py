#!/usr/bin/env python3
"""Run the supported DINOv3 ViT-B image encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "dinov3_vit_b.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if not isinstance(config["request"].get("image"), str):
        raise ValueError("[request].image must be a string")
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
    from transformers import AutoConfig, AutoImageProcessor, DINOv3ViTModel
    from rpu_backend.adapters.dinov3 import DINOv3Adapter
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )

    image_path = Path(config["request"]["image"]).expanduser()
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")
    local_only = bool(model_config.get("local_files_only", True))
    model_hf_config = AutoConfig.from_pretrained(
        checkpoint, local_files_only=local_only, trust_remote_code=False
    )
    DINOv3Adapter.preflight(model_hf_config)
    processor = AutoImageProcessor.from_pretrained(
        checkpoint, local_files_only=local_only, trust_remote_code=False
    )
    pixel_values = processor(
        images=Image.open(image_path).convert("RGB"), return_tensors="pt"
    ).pixel_values
    model = DINOv3ViTModel.from_pretrained(
        checkpoint,
        config=model_hf_config,
        dtype=torch.float16,
        local_files_only=local_only,
    ).eval()
    model = DINOv3Adapter(
        model, rpu_execution=config.get("rpu_execution")
    ).to_rpu()
    with torch.no_grad():
        output = model(pixel_values)
    print(
        f"last_hidden_state={tuple(output.last_hidden_state.shape)} "
        f"pooler_output={tuple(output.pooler_output.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
