"""Controlled-evaluation InternVLA-N1 RPU adapters."""
from __future__ import annotations

__all__ = [
    "Qwen25VLBackbone",
    "build_qwen25vl_backbone",
    "build_internvla_vision",
    "NextDiTRPURuntime",
    "build_nextdit_runtime",
]

def __getattr__(name):
    # Keep the shared fail-closed policy import light: NavDP imports
    # ``internvla_n1._policy`` and must not pull in the System-2 backbone (and
    # its Wall-OSS dependency) merely by initializing this package.
    modules = {
        "Qwen25VLBackbone": ".backbone",
        "build_qwen25vl_backbone": ".backbone",
        "build_internvla_vision": ".vision",
        "NextDiTRPURuntime": ".nextdit",
        "build_nextdit_runtime": ".nextdit",
    }
    if name in modules:
        import importlib

        module = importlib.import_module(modules[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
