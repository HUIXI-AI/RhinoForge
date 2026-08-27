"""LingBot-VLA V2 (robbyant/lingbot-vla-v2-6b) RPU adapter — standalone runtime.

LingBot-VLA V2 = Qwen3-VL-4B-Instruct VLM + a Pi0.5-style AdaRMS flow-matching action expert on a
768-d / 36-layer decoder whose MLP is a 32-expert top_k-4 sparse MoE (routed inter 512 + one
always-on ungated shared expert, inter 704). It uses a standalone runtime because the model
package is not imported by RhinoForge.

Default OFF / opt-in: nothing imports this package automatically, there is no `register_adapter`
hook and no model_registry auto-dispatch.

⚠️ `convert` is deliberately NOT re-exported eagerly. It is the only CPU-safe module here
   (no rpu_backend import at module scope), so CPU-only callers can load it on an RPU-less
   host. Importing `runtime` pulls rpu_backend (and its .so) — so the lazy
   `__getattr__` below keeps `from ... import convert` cheap and device-free.
"""
from __future__ import annotations

__all__ = ["LingbotVlaV2Policy", "build_lingbot_vla_v2", "convert"]


def __getattr__(name):
    # PEP 562 lazy re-export: `runtime` (and therefore rpu_backend / the .so) is only imported
    # when a caller actually asks for the policy or the factory.
    #
    # Use importlib.import_module, NOT `from <this package> import <sub>`: the `from` form
    # re-enters this very __getattr__ for the submodule name (the submodule is not bound as an
    # attribute yet) and recurses until RecursionError.
    import importlib
    if name in ("LingbotVlaV2Policy", "build_lingbot_vla_v2"):
        runtime = importlib.import_module(".runtime", __name__)
        return getattr(runtime, name)
    if name == "convert":
        return importlib.import_module(".convert", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
