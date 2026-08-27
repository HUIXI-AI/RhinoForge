#!/usr/bin/env python3
"""Run the source-only public InternVLA-N1 NavDP component."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "internvla_navdp.toml"
ENV_KEYS = {
    "RPU_LOG_LEVEL",
    "RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED",
    "RPU_S2_SDPA_BF16",
    "RPU_NAVDP_DENOISE_UNROLL",
    "RPU_NAVDP_BATCH",
    "RPU_NAVDP_SPM_KV_BY_MHA",
}
EXACT_ON_KEYS = {
    "RPU_INTERNVLA_N1_ALLOW_NUMERIC_BLOCKED",
}
EXACT_OFF_KEYS = {
    "RPU_S2_SDPA_BF16",
}


def _runner_env(config: dict) -> dict:
    env = config.get("runner", {}).get("env", {})
    if not isinstance(env, dict):
        raise ValueError("[runner.env] must be a table")
    unknown = sorted(set(env) - ENV_KEYS)
    if unknown:
        raise ValueError(f"unsupported InternVLA/NavDP [runner.env] keys: {unknown}")
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
    if model.get("source_only_acknowledged") is not True:
        raise ValueError("[model].source_only_acknowledged must be true")
    for key in ("asset_manifest_json",):
        if not isinstance(model.get(key), str) or not model[key]:
            raise ValueError(f"[model].{key} must be a non-empty string")
    if not isinstance(request.get("input_pt"), str) or not request["input_pt"]:
        raise ValueError("[request].input_pt must be a non-empty string")
    sample_num = request.get("sample_num", 4)
    if isinstance(sample_num, bool) or not isinstance(sample_num, int) or sample_num < 1:
        raise ValueError("[request].sample_num must be a positive integer")
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("[request].seed must be an integer")
    env = _runner_env(config)
    missing_on = sorted(
        name for name in EXACT_ON_KEYS if _env_text(env.get(name, False)) != "1"
    )
    missing_off = sorted(
        name for name in EXACT_OFF_KEYS if _env_text(env.get(name, True)) != "0"
    )
    if missing_on or missing_off:
        raise ValueError(
            "controlled profile requires true selectors "
            f"{missing_on} and false selectors {missing_off}"
        )
    return config


def _load_manifest(path: Path) -> dict[Path, str]:
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise SystemExit("asset manifest JSON must contain a path-to-SHA256 object")
    result = {}
    for value, digest in raw.items():
        asset = Path(value).expanduser()
        if not asset.is_absolute():
            asset = path.parent / asset
        result[asset.resolve()] = digest
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    model_config = config["model"]
    if args.check_config:
        location = model_config.get("checkpoint") or model_config["alias"]
        print(
            f"configuration OK: {args.config} "
            f"({location}; source-only public NavDP component)"
        )
        return 0

    for name, value in _runner_env(config).items():
        os.environ[name] = _env_text(value)

    import torch
    from rpu_backend.adapters.navdp import build_navdp
    from rpu_backend.model_registry import model_path

    root = Path(model_config.get("checkpoint") or model_path(model_config["alias"]))
    manifest_path = Path(model_config["asset_manifest_json"]).expanduser()
    if not manifest_path.is_file():
        raise SystemExit(f"asset manifest JSON not found: {manifest_path}")
    input_path = Path(config["request"]["input_pt"]).expanduser()
    if not input_path.is_file():
        raise SystemExit(f"NavDP embedding input not found: {input_path}")
    values = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(values, Mapping):
        raise SystemExit("NavDP input .pt must contain a tensor mapping")
    for name, shape in (("goal_embed", (1, 1, 384)), ("rgbd_embed", (1, 32, 384))):
        value = values.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise SystemExit(f"NavDP {name} must be a tensor with shape {shape}")

    runtime = build_navdp(
        root,
        asset_manifest=_load_manifest(manifest_path),
    )
    try:
        action = runtime.predict_action(
            values["goal_embed"],
            values["rgbd_embed"],
            sample_num=config["request"].get("sample_num", 4),
            seed=config["request"].get("seed", 0),
        )
        print(f"normalized_action_shape={tuple(action.shape)} component_only=true")
    finally:
        runtime.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
