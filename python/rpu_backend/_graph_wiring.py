"""Optional Graph/FakeTensor package wiring for a loaded backend."""
from __future__ import annotations

import warnings
from typing import Any


def configure_graph_runtime(_cpp_ext: Any) -> None:
    """Install optional FakeTensor integration for a loaded backend.

    The unsupported Dynamo frontend is not wired onto the package shim; its
    module remains directly importable for compatibility.
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
