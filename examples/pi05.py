#!/usr/bin/env python3
"""Run one Pi0.5 action-chunk inference from a caller-provided batch."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "pi05_libero.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    request = config["request"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if not isinstance(request.get("batch_file"), str):
        raise ValueError("[request].batch_file must be a string")
    if not isinstance(request.get("num_steps"), int) or request["num_steps"] < 1:
        raise ValueError("[request].num_steps must be a positive integer")
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
    from rpu_backend.api import Pi05Policy
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )

    batch_path = Path(config["request"]["batch_file"]).expanduser()
    if not batch_path.is_file():
        raise SystemExit(f"Pi0.5 batch file not found: {batch_path}")
    batch = torch.load(batch_path, map_location="cpu", weights_only=True)
    if not isinstance(batch, dict):
        raise SystemExit("Pi0.5 batch file must contain a tensor dictionary")
    num_steps = config["request"]["num_steps"]
    policy = Pi05Policy.from_pretrained(
        checkpoint,
        rpu_execution=config.get("rpu_execution"),
        trust_remote_code=False,
    ).to("rpu")
    if bool(config["request"].get("prepare_graphs", False)):
        policy.prepare_graphs(batch, num_steps=num_steps)
    actions = policy.predict_action_chunk(batch, num_steps=num_steps)
    print(f"actions={tuple(actions.shape)} dtype={actions.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
