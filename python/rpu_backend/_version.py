"""Explicit distribution-version and API-level contract."""
from __future__ import annotations

import importlib.metadata as _metadata
import pathlib as _pathlib


# API compatibility evolves independently from distribution releases.
API_VERSION = "5.0.0"


def _source_distribution_version() -> str | None:
    """Read the same version file used by Python and CMake builds."""
    root = _pathlib.Path(__file__).resolve().parents[2]
    version_file = root / "VERSION"
    if not (root / "pyproject.toml").is_file() or not version_file.is_file():
        return None
    return version_file.read_text(encoding="utf-8").strip()


def _resolve_distribution_version() -> str:
    source_version = _source_distribution_version()
    if source_version is not None:
        return source_version
    for dist_name in ("rhinoforge", "rpu_backend"):
        try:
            return _metadata.version(dist_name)
        except _metadata.PackageNotFoundError:
            continue
    # An unpacked package without its project metadata is not a release
    # artifact. Keep imports usable while making the missing metadata loud.
    return "0+unknown"


DISTRIBUTION_VERSION = _resolve_distribution_version()


__all__ = ["API_VERSION", "DISTRIBUTION_VERSION"]
