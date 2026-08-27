"""Explicit distribution-version and API-level contract."""
from __future__ import annotations

import importlib.metadata as _metadata
import pathlib as _pathlib
import re as _re


# API compatibility evolves independently from distribution releases.
API_VERSION = "5.0.0"


def _source_distribution_version() -> str | None:
    """Read the canonical PEP 621 version when running from a source tree."""
    pyproject = _pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.is_file():
        return None
    text = pyproject.read_text(encoding="utf-8")
    project = _re.search(
        r"(?ms)^\[project\]\s*$\n(?P<body>.*?)(?=^\[|\Z)",
        text,
    )
    if project is None:
        return None
    match = _re.search(
        r"(?m)^version\s*=\s*['\"](?P<value>[^'\"]+)['\"]\s*$",
        project.group("body"),
    )
    return None if match is None else match.group("value")


def _resolve_distribution_version() -> str:
    source_version = _source_distribution_version()
    if source_version is not None:
        return source_version
    try:
        return _metadata.version("rhinoforge")
    except _metadata.PackageNotFoundError:
        # An unpacked package without its project metadata is not a release
        # artifact. Keep imports usable while making the missing metadata loud.
        return "0+unknown"


DISTRIBUTION_VERSION = _resolve_distribution_version()


__all__ = ["API_VERSION", "DISTRIBUTION_VERSION"]
