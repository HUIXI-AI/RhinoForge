"""Optional Graph/FakeTensor package wiring for a loaded backend."""
from __future__ import annotations

import warnings
from typing import Any


def configure_graph_runtime(_cpp_ext: Any) -> None:
    """Install optional FakeTensor integration for a loaded backend.

    Returned a dict of `compile` / `dynamo_backend` exports until 2026-08-08.
    The Dynamo frontend is FROZEN (graph/dynamo_backend.py docstring), so it is
    no longer wired onto the package shim; the module stays importable directly.
    Importing `fake_dispatch` is the whole remaining job — it registers the
    FakeTensor rules as an import side effect.
    """
    try:
        from rpu_backend.graph import fake_dispatch as _fake_dispatch  # noqa: F401
    except Exception as exc:
        warnings.warn(
            "rpu_backend FakeTensor dispatch wire failed: "
            f"{type(exc).__name__}: {exc}",
            stacklevel=2,
        )


__all__ = ["configure_graph_runtime"]
