#!/usr/bin/env python3
"""Run one Wall-OSS action-chunk inference."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "wall_oss.toml"
PRECISIONS = {"fp16", "w8a16", "w4a16", "nvfp4a16"}


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    request = config["request"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    precision = model.get("precision", "fp16")
    if precision not in PRECISIONS:
        raise ValueError(f"[model].precision must be one of {sorted(PRECISIONS)}")
    if precision != "fp16" and not (
        model.get("fp16_alias") or model.get("fp16_checkpoint")
    ):
        raise ValueError(
            "quantized Wall-OSS needs [model].fp16_alias or fp16_checkpoint"
        )
    images = request.get("images")
    if (
        not isinstance(images, list)
        or not images
        or not all(isinstance(value, str) and value for value in images)
    ):
        raise ValueError("[request].images must be a non-empty string array")
    if not isinstance(request.get("instruction"), str) or not request["instruction"]:
        raise ValueError("[request].instruction must be a non-empty string")
    cameras = model.get("camera_names", ["face_view"])
    if (
        not isinstance(cameras, list)
        or not cameras
        or not all(isinstance(value, str) and value for value in cameras)
        or len(cameras) != len(images)
    ):
        raise ValueError(
            "[model].camera_names must match the non-empty request image list"
        )
    proprioception = request.get("proprioception")
    if not isinstance(proprioception, list) or not proprioception or any(
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
    precision = model_config.get("precision", "fp16")
    fp16_source = model_config.get("fp16_checkpoint")
    if fp16_source is None and model_config.get("fp16_alias"):
        fp16_source = str(model_path(model_config["fp16_alias"]))

    image_paths = [Path(value).expanduser() for value in config["request"]["images"]]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Wall-OSS image file(s) not found: {', '.join(missing)}")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    policy = WallOssPolicy.from_pretrained(
        checkpoint,
        dataset_key=model_config.get("dataset_key", "berkeley_autolab_ur5"),
        camera_names=tuple(model_config.get("camera_names", ["face_view"])),
        delta_action=model_config.get("delta_action", False),
        state_bins=int(model_config.get("state_bins", 256)),
        action_hz=float(model_config.get("action_hz", 32.0)),
        action_horizon=int(model_config.get("action_horizon", 32)),
        num_steps=int(model_config.get("num_steps", 10)),
        max_seq_len=int(model_config.get("max_seq_len", 2048)),
        w8a16=precision == "w8a16",
        w4a16=precision == "w4a16",
        nvfp4a16=precision == "nvfp4a16",
        fp16_ckpt_dir=fp16_source,
        active_slots=model_config.get("active_slots"),
        runtime_env=model_config.get("runtime_env"),
        rpu_execution=config.get("rpu_execution"),
    ).to("rpu")
    request = config["request"]
    infer_options = {
        name: request[name]
        for name in (
            "state_mask",
            "noise",
            "dof_mask",
            "num_steps",
        )
        if name in request
    }
    output = policy.infer(
        images=images,
        instruction=request["instruction"],
        proprioception=request["proprioception"],
        noise_seed=int(request.get("noise_seed", 0)),
        **infer_options,
    )
    print(
        f"actions={tuple(output.actions.shape)} action_hz={output.action_hz:g} "
        f"latency_ms={output.latency_ms:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
