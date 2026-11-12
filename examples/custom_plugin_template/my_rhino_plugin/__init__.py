"""Example out-of-tree RhinoForge adapter plugin package."""
from __future__ import annotations

from pathlib import Path
from typing import Any

ARCH_NAME = "MyCustomForCausalLM"


class MyCustomAdapter:
    """Minimal external adapter demonstrating the RhinoForge plugin contract."""

    architecture = ARCH_NAME
    _native_loaded = False

    @classmethod
    def preflight(cls, config: Any) -> None:
        """Validate model configuration before weight loading or device allocation."""
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None and hidden_size <= 0:
            raise ValueError(f"{ARCH_NAME} requires a positive hidden_size, got {hidden_size!r}")

    @classmethod
    def load_native_ops(cls, library_path: str | Path | None = None) -> bool:
        """Optionally load an out-of-tree TORCH_LIBRARY_FRAGMENT shared object."""
        if cls._native_loaded:
            return True
        candidate = (
            Path(library_path)
            if library_path is not None
            else Path(__file__).resolve().with_name("libmy_rhino_custom_ops.so")
        )
        if not candidate.is_file():
            return False
        import torch

        torch.ops.load_library(str(candidate))
        cls._native_loaded = True
        return True


def register(api: Any) -> None:
    """Entry-point hook invoked by ``rpu_backend.runtime.registry.discover_plugins()``."""
    api.register_adapter(ARCH_NAME, MyCustomAdapter)


__all__ = ["ARCH_NAME", "MyCustomAdapter", "register"]
