"""Process-environment setup used by the package composition root.

This module intentionally imports only the standard library at module load so
the file-descriptor limit can be adjusted before PyTorch is imported.
"""
from __future__ import annotations

import resource


_TARGET_NOFILE = 1_048_576


def ensure_nofile_limit() -> None:
    """Best-effort raise of the soft fd limit without import-time output."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= _TARGET_NOFILE:
        return
    new_soft = min(_TARGET_NOFILE, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError):
        return


def configure_hf_cache() -> None:
    """Apply the project model-cache policy after the PyTorch import boundary."""
    from . import model_registry

    model_registry.set_hf_home_env()


__all__ = ["configure_hf_cache", "ensure_nofile_limit"]
