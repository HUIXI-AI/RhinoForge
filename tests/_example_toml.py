"""Read an example TOML in the runner-facing layout, as examples/run.py does.

Templates use the unified [model] table; runners and their validators receive
the legacy layout produced by normalize_config(). Loaded by path so tests do
not depend on the repository root being importable.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_example_config_layout", ROOT / "examples/common/config.py")
_CONFIG = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONFIG)
normalize_config = _CONFIG.normalize_config


def load_example_toml(path) -> dict:
    return normalize_config(tomllib.loads(Path(path).read_text()))
