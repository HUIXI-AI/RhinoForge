"""Verified access to the public InternVLA-N1-with-NavDP checkpoint."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from rpu_backend.adapters.wall_oss.llm import _SafeTensorStore
from rpu_backend.runtime import UnsupportedModelError

from ._policy import (
    require_controlled_evaluation,
    verify_asset_hashes,
    verify_asset_manifest,
)


NAVDP_PREFIX = "model.navdp."
_PUBLIC_PROFILE = {
    "model_type": "internvla_n1",
    "system1": "navdp_async",
}


def open_public_checkpoint(
    checkpoint_dir: str | Path,
    asset_manifest: Mapping[str | Path, str] | None,
    *,
    prefixes: Sequence[str],
    controlled_rpu: bool,
) -> tuple[_SafeTensorStore, tuple[str, ...], tuple[Path, ...]]:
    """Open only tensors backed by the public sharded checkpoint layout."""
    if controlled_rpu:
        require_controlled_evaluation(asset_manifest)
        verifier = verify_asset_manifest
    else:
        if not asset_manifest:
            raise UnsupportedModelError(
                "InternVLA-N1 source-only loading requires a caller-supplied "
                "asset_manifest."
            )
        verifier = verify_asset_hashes

    root = Path(checkpoint_dir).expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    descriptor = index_path if index_path.is_file() else single_path
    verifier((config_path, descriptor), asset_manifest)

    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    errors = [
        f"{key}={config.get(key)!r} (expected {expected!r})"
        for key, expected in _PUBLIC_PROFILE.items()
        if config.get(key) != expected
    ]
    if errors:
        raise ValueError(
            "checkpoint is not the public InternVLA-N1-with-NavDP profile: "
            + "; ".join(errors)
        )

    if descriptor == index_path:
        with index_path.open(encoding="utf-8") as stream:
            weight_map = json.load(stream).get("weight_map", {})
    else:
        from safetensors import safe_open

        with safe_open(single_path, framework="pt", device="cpu") as handle:
            weight_map = {name: single_path.name for name in handle.keys()}
    if not isinstance(weight_map, dict):
        raise ValueError("InternVLA-N1 checkpoint weight_map must be an object")

    names = tuple(
        sorted(
            name
            for name in weight_map
            if isinstance(name, str)
            and any(name.startswith(prefix) for prefix in prefixes)
        )
    )
    if not names:
        raise ValueError(
            "public InternVLA-N1 checkpoint has no tensors for prefixes "
            f"{tuple(prefixes)!r}"
        )
    shard_names = {weight_map[name] for name in names}
    if not all(isinstance(name, str) and name for name in shard_names):
        raise ValueError("InternVLA-N1 checkpoint index has invalid shard names")
    shards = tuple(sorted((root / name).resolve() for name in shard_names))
    if any(not path.is_relative_to(root) for path in shards):
        raise ValueError("InternVLA-N1 checkpoint index escapes checkpoint_dir")
    verifier(shards, asset_manifest)
    assets = (config_path, descriptor, *shards)
    return _SafeTensorStore(str(root), mmap=False), names, assets
