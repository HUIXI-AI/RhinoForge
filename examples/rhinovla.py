#!/usr/bin/env python3
"""Run RhinoVLA through a model-repository-owned runtime factory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "rhinovla.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    request = config["request"]
    factory = model.get("runtime_factory")
    if not isinstance(factory, str) or factory.count(":") != 1:
        raise ValueError("[model].runtime_factory must use module:callable syntax")
    if not isinstance(model.get("checkpoint"), str) or not model["checkpoint"]:
        raise ValueError("[model].checkpoint must be a non-empty string")
    if not isinstance(request.get("request_json"), str):
        raise ValueError("[request].request_json must be a string")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.check_config:
        print(f"configuration OK: {args.config}")
        return 0

    from rpu_backend.api import RhinoVLAPolicy

    model_config = config["model"]
    if model_config["runtime_factory"].startswith("your_package."):
        raise SystemExit(
            "Set [model].runtime_factory to the module:callable exported by "
            "your RhinoVLA model integration."
        )
    request_path = Path(config["request"]["request_json"]).expanduser()
    if not request_path.is_file():
        raise SystemExit(f"RhinoVLA request JSON not found: {request_path}")
    with request_path.open(encoding="utf-8") as stream:
        request = json.load(stream)
    if not isinstance(request, dict):
        raise SystemExit("RhinoVLA request JSON must contain an object")

    factory_kwargs = {}
    if config.get("runtime"):
        factory_kwargs["config"] = config["runtime"]
    policy = RhinoVLAPolicy.from_pretrained(
        model_config["checkpoint"],
        runtime_factory=model_config["runtime_factory"],
        rpu_execution=config.get("rpu_execution"),
        **factory_kwargs,
    )
    output = policy.predict(**request)
    print(f"output_type={type(output).__name__} shape={getattr(output, 'shape', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
