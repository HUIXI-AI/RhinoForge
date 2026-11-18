"""Fail-closed policy shared by controlled InternVLA-N1 RPU paths."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from rpu_backend.runtime import UnsupportedModelError


S2_SDPA_BF16_ENV = "RPU_S2_SDPA_BF16"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def require_numeric_blocked_opt_in() -> None:
    """Reject the retired BF16 SDPA selector.

    The System2/RGBD/NavDP chain is supported (2026-10-19 board run), so no
    numeric-blocked opt-in is required any more.
    """
    sdpa_bf16 = os.environ.get(S2_SDPA_BF16_ENV)
    if sdpa_bf16 not in (None, "", "0"):
        if sdpa_bf16 != "1":
            raise UnsupportedModelError(
                f"{S2_SDPA_BF16_ENV} accepts only literal '0' or '1'."
            )
        raise UnsupportedModelError(
            f"{S2_SDPA_BF16_ENV}=1 is unavailable with the current latest "
            "RPU op-library: its BF16 flash-attention kernel was removed. "
            "Do not side-load the retired MR33 op-library; leave the variable "
            "unset for the controlled FP16-SDPA evaluation."
        )


def require_controlled_environment() -> None:
    """Reject unsafe ambient selectors before any controlled-evaluation I/O."""
    require_numeric_blocked_opt_in()


def require_controlled_evaluation(
    asset_manifest: Mapping[str | Path, str] | None,
) -> None:
    require_controlled_environment()
    if not asset_manifest:
        raise UnsupportedModelError(
            "InternVLA-N1 controlled evaluation requires a caller-supplied "
            "asset_manifest mapping every consumed file path to its SHA-256."
        )


def verify_asset_hashes(
    required_assets: Sequence[str | Path],
    asset_manifest: Mapping[str | Path, str] | None,
) -> None:
    """Verify caller-supplied SHA-256 values without changing RPU policy."""
    if not asset_manifest:
        raise UnsupportedModelError(
            "InternVLA-N1 exact assets require a caller-supplied asset_manifest."
        )

    manifest: dict[Path, str] = {}
    for raw_path, digest in asset_manifest.items():
        path = Path(raw_path).expanduser().resolve()
        if path in manifest and manifest[path] != digest:
            raise UnsupportedModelError(f"conflicting SHA-256 values for {path}")
        manifest[path] = digest

    required = tuple(Path(path).expanduser().resolve() for path in required_assets)
    missing = [str(path) for path in required if path not in manifest]
    invalid = [
        str(path)
        for path in required
        if path in manifest
        and (
            not isinstance(manifest[path], str)
            or _SHA256_RE.fullmatch(manifest[path]) is None
        )
    ]
    if missing or invalid:
        raise UnsupportedModelError(
            "InternVLA-N1 asset manifest is incomplete or malformed: "
            f"missing={missing}, invalid_sha256={invalid}"
        )

    for path in required:
        if not path.is_file():
            raise UnsupportedModelError(f"InternVLA-N1 asset is not a file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if not hmac.compare_digest(digest.hexdigest(), manifest[path]):
            raise UnsupportedModelError(f"InternVLA-N1 asset SHA-256 mismatch: {path}")


def verify_asset_manifest(
    required_assets: Sequence[str | Path],
    asset_manifest: Mapping[str | Path, str] | None,
) -> None:
    """Verify controlled-RPU policy and caller-supplied asset SHA-256 values."""
    require_controlled_evaluation(asset_manifest)
    verify_asset_hashes(required_assets, asset_manifest)
