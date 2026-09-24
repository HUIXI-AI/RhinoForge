#!/usr/bin/env python3
"""Run Source-only GR00T with caller-preprocessed tensor inputs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import tomllib


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "gr00t/n1_7_droid/fp16.toml"
ENV_KEYS = {
    "RPU_LOG_LEVEL",
    "RPU_WARMUP",
    "RPU_GR00T_KVPAD16",
    "RPU_GR00T_PARTIAL_MROPE",
}
GR00T_PROFILES = {
    "gr00t-n1.7-3b-droid-zero-shot": {
        "alias": "gr00t-n1d7-3b-droid-zero-shot",
        "checkpoint_repository": "nvidia/GR00T-N1.7-3B",
        "checkpoint_revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
        "processor_config_sha256": "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be",
        "statistics_sha256": "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9",
        "cosmos_repository": "nvidia/Cosmos-Reason2-2B",
        "cosmos_revision": "9ce19a195e423419c349abfc86fd07178b230561",
        "cosmos_manifest_sha256": "0aeff3308620feaa907990623b9db429a43cc671673d9f8ee34e4415144055a2",
        "embodiment_id": 24,
        "input_envelope": {
            "embodiment_tag": "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
            "camera_names": ["exterior_image_1_left", "wrist_image_left"],
            "video_delta_indices": [-15, 0],
            "images_per_request": 4,
            "image_grid_thw": [1, 16, 16],
            "image_target_size": [256, 256],
            "batch_size": 1,
            "state_dim": 17,
            "action_dim": 17,
            "action_horizon": 40,
            "processor_input_oracle_sequence_length": 277,
            "kvpad16_min_rows": 288,
            "rhinoforge_cache_capacity": 320,
        },
    },
    "gr00t-n1.7-droid-finetuned": {
        "alias": "gr00t-n1d7-droid-finetuned",
        "checkpoint_repository": "nvidia/GR00T-N1.7-DROID",
        "checkpoint_revision": "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
        "processor_config_sha256": "4b5c3bab3f148ff47ba903714c3247403c754f806ff2354a73acdfa2102a66fb",
        "statistics_sha256": "127832f7df25cda15da4ba6be81737f96b65673d0f892f9fc1bce1bc062fa858",
        "cosmos_repository": "nvidia/Cosmos-Reason2-2B",
        "cosmos_revision": "9ce19a195e423419c349abfc86fd07178b230561",
        "cosmos_manifest_sha256": "0aeff3308620feaa907990623b9db429a43cc671673d9f8ee34e4415144055a2",
        "embodiment_id": 24,
        "input_envelope": {
            "embodiment_tag": "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
            "camera_names": ["exterior_image_1_left", "wrist_image_left"],
            "video_delta_indices": [0],
            "images_per_request": 2,
            "image_grid_thw": [1, 16, 16],
            "image_target_size": [256, 256],
            "batch_size": 1,
            "state_dim": 17,
            "action_dim": 17,
            "action_horizon": 40,
            "processor_input_oracle_sequence_length": 145,
            "kvpad16_min_rows": 160,
            "rhinoforge_cache_capacity": 256,
        },
    },
}


def _runner_env(config: dict) -> dict:
    env = config.get("runner", {}).get("env", {})
    if not isinstance(env, dict):
        raise ValueError("[runner.env] must be a table")
    unknown = sorted(set(env) - ENV_KEYS)
    if unknown:
        raise ValueError(f"unsupported GR00T [runner.env] keys: {unknown}")
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
    profile = model.get("profile")
    identity = GR00T_PROFILES.get(profile)
    if identity is None:
        raise ValueError(f"[model].profile must be one of {sorted(GR00T_PROFILES)}")
    if model.get("alias") != identity["alias"] or model.get("checkpoint"):
        raise ValueError("[model].alias must match the selected immutable GR00T profile")
    if model.get("source_only_acknowledged") is not True:
        raise ValueError("[model].source_only_acknowledged must be true")
    for name in (
        "checkpoint_repository",
        "checkpoint_revision",
        "processor_config_sha256",
        "statistics_sha256",
        "cosmos_repository",
        "cosmos_revision",
        "cosmos_manifest_sha256",
        "embodiment_id",
    ):
        if model.get(name) != identity[name]:
            raise ValueError(f"[model].{name} does not match profile {profile!r}")
    if model.get("input_envelope") != identity["input_envelope"]:
        raise ValueError(f"[model].input_envelope does not match profile {profile!r}")
    if "qwen3vl_alias" in model or "qwen3vl_checkpoint" in model:
        raise ValueError("Qwen3-VL substitutes are not valid Cosmos profile identities")
    if not isinstance(request.get("input_pt"), str) or not request["input_pt"]:
        raise ValueError("[request].input_pt must be a non-empty string")
    if request.get("num_steps", 4) != 4:
        raise ValueError("[request].num_steps must be 4 for this runtime")
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("[request].seed must be an integer")
    _runner_env(config)
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
        print(
            f"configuration OK: {args.config.name} ({location}; Source-only, "
            "board validation pending)"
        )
        return 0

    for name, value in _runner_env(config).items():
        os.environ[name] = _env_text(value)

    import torch
    from rpu_backend.adapters.gr00t import build_gr00t_vla
    from rpu_backend.model_registry import model_path

    checkpoint = model_config.get("checkpoint") or str(
        model_path(model_config["alias"])
    )
    cosmos = model_config.get("cosmos_checkpoint")
    input_path = Path(config["request"]["input_pt"]).expanduser()
    if not input_path.is_file():
        raise SystemExit(f"GR00T preprocessed input not found: {input_path}")
    inputs = torch.load(input_path, map_location="cpu", weights_only=True)
    if not isinstance(inputs, Mapping):
        raise SystemExit("GR00T input .pt must contain a tensor mapping")
    required = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state")
    missing = [name for name in required if not isinstance(inputs.get(name), torch.Tensor)]
    if missing:
        raise SystemExit(f"GR00T input .pt is missing tensor fields: {missing}")

    runtime = build_gr00t_vla(
        checkpoint,
        cosmos,
        profile=model_config["profile"],
        checkpoint_repository=model_config["checkpoint_repository"],
        checkpoint_revision=model_config["checkpoint_revision"],
        processor_config_sha256=model_config["processor_config_sha256"],
        statistics_sha256=model_config.get("statistics_sha256"),
        cosmos_repository=model_config["cosmos_repository"],
        cosmos_revision=model_config.get("cosmos_revision"),
        cosmos_manifest_sha256=model_config.get("cosmos_manifest_sha256"),
        embodiment_id=model_config["embodiment_id"],
        input_envelope=model_config["input_envelope"],
        admission_inputs=inputs,
        rpu_execution=config.get("rpu_execution"),
    )
    try:
        action = runtime.get_action(
            dict(inputs),
            seed=config["request"].get("seed", 0),
            num_steps=config["request"].get("num_steps", 4),
        )
        print(f"normalized_action_shape={tuple(action.shape)}")
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
