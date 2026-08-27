"""Controlled InternVLA-N1 System-2 Qwen2.5-VL vision builder."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from rpu_backend.adapters.wall_oss.vision import build_wall_oss_vision

from ._checkpoint import open_public_checkpoint


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
    root = Path(ckpt_dir).expanduser().resolve()
    config_path = root / "config.json"
    open_public_checkpoint(
        root,
        asset_manifest,
        prefixes=("visual.",),
        controlled_rpu=True,
    )
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

    return build_wall_oss_vision(
        str(root),
        window=True,
        max_hw=32,
        max_seq_len=1024,
        w8a16=False,
        fp16_ckpt_dir=str(root),
        per_window_sdpa=True,
    )
