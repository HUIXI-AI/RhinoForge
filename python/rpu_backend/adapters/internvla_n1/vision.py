"""Controlled InternVLA-N1 System-2 Qwen2.5-VL vision builder."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rpu_backend.adapters.wall_oss.vision import build_wall_oss_vision

from ._policy import require_controlled_evaluation, verify_asset_manifest


_VISION_PROFILE = {
    "depth": 32,
    "hidden_size": 1280,
    "num_heads": 16,
    "intermediate_size": 3420,
    "spatial_merge_size": 2,
    "out_hidden_size": 3584,
    "window_size": 112,
    "patch_size": 14,
    "fullatt_block_indexes": [7, 15, 23, 31],
}


def build_internvla_vision(
    ckpt_dir: str | Path,
    *,
    asset_manifest: Mapping[str | Path, str] | None = None,
):
    """Build the exact FP16 384px/784-patch controlled Vision profile."""
    require_controlled_evaluation(asset_manifest)
    root = Path(ckpt_dir).expanduser().resolve()
    config_path = root / "config.json"
    verify_asset_manifest((config_path,), asset_manifest)
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    vision_config = config.get("vision_config", {})
    errors = [
        f"{name}={vision_config.get(name)!r} (expected {expected!r})"
        for name, expected in _VISION_PROFILE.items()
        if vision_config.get(name) != expected
    ]
    if errors:
        raise ValueError(
            "InternVLA-N1 Vision checkpoint does not match the controlled "
            f"profile: {'; '.join(errors)}"
        )

    index_path = root / "model.safetensors.index.json"
    descriptor = index_path if index_path.is_file() else root / "model.safetensors"
    verify_asset_manifest((descriptor,), asset_manifest)
    if descriptor == index_path:
        with index_path.open(encoding="utf-8") as stream:
            weight_map = json.load(stream).get("weight_map", {})
        shards = tuple(
            (root / name).resolve() for name in sorted(set(weight_map.values()))
        )
        if not shards or any(root not in path.parents for path in shards):
            raise ValueError(
                "InternVLA-N1 Vision checkpoint index is empty or escapes ckpt_dir"
            )
        verify_asset_manifest(shards, asset_manifest)

    return build_wall_oss_vision(
        str(root),
        window=True,
        max_hw=32,
        max_seq_len=1024,
        w8a16=False,
        fp16_ckpt_dir=str(root),
        per_window_sdpa=True,
    )
