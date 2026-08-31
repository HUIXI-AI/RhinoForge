#!/usr/bin/env python3
"""Validate the public G0.5 source-integration configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "g05.toml"


def load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    if not model.get("alias") and not model.get("checkpoint"):
        raise ValueError("[model] needs alias or checkpoint")
    if model.get("source_only_acknowledged") is not True:
        raise ValueError("[model].source_only_acknowledged must be true")
    if model.get("source") != "OpenGalaxea/G05":
        raise ValueError("[model].source must be OpenGalaxea/G05")
    if model.get("revision") != "e312be81e90c56a55bcb26b57429bd39a335b449":
        raise ValueError("[model].revision must match the v1.0.0 public ledger")
    if model.get("subfolder") != "g05-base":
        raise ValueError("[model].subfolder must be g05-base")
    if model.get("implementation") != "OpenGalaxea/GalaxeaVLA":
        raise ValueError("[model].implementation must be OpenGalaxea/GalaxeaVLA")
    if (
        model.get("implementation_revision")
        != "89f2322b4ad016e192437adc1a2c253b05bab246"
    ):
        raise ValueError("[model].implementation_revision must match the public ledger")
    if (
        model.get("config_sha256")
        != "c98af37352c0f600341d2ecbdb49fcdfa87812198654448991615065fa1a4461"
    ):
        raise ValueError("[model].config_sha256 must match the resolved Hydra config")
    if model.get("license") != "G0.5 Community License (non-commercial)":
        raise ValueError("[model].license must retain the non-commercial terms")
    max_seq_len = model.get("max_seq_len", 2048)
    if max_seq_len != 2048:
        raise ValueError("[model].max_seq_len must be exactly 2048")
    if any(
        model.get(name) is not True
        for name in ("continuous_action", "discrete_action", "predict_cot")
    ):
        raise ValueError("[model] must keep continuous, discrete, and CoT enabled")
    env = config.get("runner", {}).get("env", {})
    if not isinstance(env, dict) or set(env) - {"RPU_LOG_LEVEL"}:
        raise ValueError("G0.5 [runner.env] supports only RPU_LOG_LEVEL")
    if any(not isinstance(value, (str, int, bool)) for value in env.values()):
        raise ValueError("G0.5 [runner.env] values must be scalar")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    model = config["model"]
    location = model.get("checkpoint") or model["alias"]
    if args.check_config:
        print(f"configuration OK: {args.config} ({location}; source only)")
        return 0

    raise SystemExit(
        "Construct the official CPU G05PolicyQwen35 from OpenGalaxea/GalaxeaVLA "
        "at revision 89f2322b4ad016e192437adc1a2c253b05bab246, "
        "call rpu_backend.adapters.g05.patch_g05_policy_for_rpu(policy, "
        f"max_seq_len={model.get('max_seq_len', 2048)}), then invoke the "
        "official simultaneous continuous/discrete/CoT request path. This "
        "non-commercial v1.0.0 integration is "
        "source-only until the public checkpoint profile passes the release gates."
    )


if __name__ == "__main__":
    raise SystemExit(main())
