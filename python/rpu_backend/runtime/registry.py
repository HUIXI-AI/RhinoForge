"""Adapter registration and lazy discovery.

Public API (stable): register_adapter, get_adapter, list_adapters.
Discovery bootstrap: discover_all (called once from rpu_backend/__init__.py).
Built-in adapters are registered as lazy module targets; importing the backend
does not import model adapters or install their class-level patches.

Internal: ADAPTERS dict + _register_plugin_adapter_with_snapshot factory
(consumed by entry-point discovery).
"""
from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import threading
import warnings as _warnings
from types import SimpleNamespace
from typing import Any, Iterable


@dataclasses.dataclass(frozen=True)
class _LazyAdapter:
    module: str
    attr: str


ADAPTERS: dict[str, Any] = {}
_REGISTRY_LOCK = threading.RLock()
_PLUGINS_DISCOVERED = False


def register_adapter(arch_string: str, adapter_cls) -> None:
    """Register an HF architecture string -> adapter class binding.

    Idempotent on the same (arch, cls) pair (required for harness re-imports).
    Raises RuntimeError on duplicate-arch-with-different-class.
    """
    with _REGISTRY_LOCK:
        current = ADAPTERS.get(arch_string)
        if isinstance(current, _LazyAdapter):
            actual = (adapter_cls.__module__, adapter_cls.__name__)
            expected = (current.module, current.attr)
            if actual != expected:
                raise RuntimeError(
                    f"Adapter already registered for {arch_string!r}: "
                    f"lazy target {expected!r} != {actual!r}"
                )
            ADAPTERS[arch_string] = adapter_cls
            return
        if current is not None and current is not adapter_cls:
            raise RuntimeError(
                f"Adapter already registered for {arch_string!r}: "
                f"{current!r} != {adapter_cls!r}"
            )
        ADAPTERS[arch_string] = adapter_cls


def get_adapter(arch_string: str):
    """Return the adapter class or raise ``UnsupportedModelError``."""
    from rpu_backend.runtime import UnsupportedModelError
    with _REGISTRY_LOCK:
        adapter = ADAPTERS.get(arch_string)
    if adapter is None:
        _discover_plugins_once()
        with _REGISTRY_LOCK:
            adapter = ADAPTERS.get(arch_string)
    with _REGISTRY_LOCK:
        if adapter is None:
            raise UnsupportedModelError(
                f"No RPU adapter registered for HF architecture {arch_string!r}. "
                f"Registered archs: {sorted(list_adapters())}. "
                "See docs/api_reference.md#adapter-registry-and-plugins and "
                "docs/model_porting.md#6-register-the-adapter."
            )
        if not isinstance(adapter, _LazyAdapter):
            return adapter

        module = importlib.import_module(adapter.module)
        adapter_cls = getattr(module, adapter.attr)
        current = ADAPTERS.get(arch_string)
        if isinstance(current, _LazyAdapter):
            # Built-in modules normally replace the lazy target through their
            # existing register_adapter(...) call. Keep the manifest robust if
            # a future adapter omits that redundant module-level registration.
            ADAPTERS[arch_string] = adapter_cls
            current = adapter_cls
        if current is not adapter_cls:
            raise RuntimeError(
                f"Adapter target changed while loading {arch_string!r}: "
                f"{current!r} != {adapter_cls!r}"
            )
        return adapter_cls


def list_adapters() -> list[str]:
    """Return registered HF architecture names in insertion order."""
    with _REGISTRY_LOCK:
        return list(ADAPTERS.keys())


def _register_plugin_adapter_with_snapshot(builtin_snapshot):
    """Factory: returns a register callable that refuses any arch in `builtin_snapshot`.

    Discovery snapshots built-in names before loading entry points, then gives
    each plugin a namespace whose ``register_adapter`` uses this closure.
    """
    def register_plugin_adapter(arch: str, cls) -> None:
        if arch in builtin_snapshot:
            raise RuntimeError(
                f"{arch} is a built-in; plugins cannot override. "
                "See docs/api_reference.md#adapter-registry-and-plugins."
            )
        register_adapter(arch, cls)
    return register_plugin_adapter


def register_builtins(
    builtins: Iterable[tuple[str, str, str]],
) -> None:
    """Register injected ``(arch, module, class)`` targets without imports."""
    with _REGISTRY_LOCK:
        for arch, module, attr in builtins:
            target = _LazyAdapter(module, attr)
            current = ADAPTERS.get(arch)
            if current is None:
                ADAPTERS[arch] = target
                continue
            if isinstance(current, _LazyAdapter) and current == target:
                continue
            actual = (getattr(current, "__module__", None),
                      getattr(current, "__name__", None))
            if actual != (target.module, target.attr):
                raise RuntimeError(
                    f"Built-in adapter collision for {arch!r}: "
                    f"{current!r} != {(target.module, target.attr)!r}"
                )


def discover_in_tree(
    builtins: Iterable[tuple[str, str, str]] = (),
) -> None:
    """Compatibility name for lazy built-in registration.

    This intentionally does not scan or import ``rpu_backend.adapters``. A
    checked manifest makes package import deterministic and keeps model patch
    installation behind the first explicit architecture lookup.
    """
    register_builtins(builtins)


def discover_entry_points() -> None:
    """Explicitly load ``rpu_backend.plugins`` and reject shadow attempts.

    Registration collisions propagate as ``RuntimeError``. Other plugin
    failures become ``ImportWarning`` so one optional plugin cannot break the
    backend import.
    """
    global _PLUGINS_DISCOVERED
    with _REGISTRY_LOCK:
        builtin_snapshot = frozenset(ADAPTERS.keys())
        api = SimpleNamespace(
            register_adapter=_register_plugin_adapter_with_snapshot(builtin_snapshot)
        )
        for ep in importlib.metadata.entry_points(group="rpu_backend.plugins"):
            try:
                register_callable = ep.load()
                register_callable(api)
            except RuntimeError:
                # Propagate rejected overrides and duplicate registrations.
                raise
            except Exception as _e:
                # All other plugin-authoring errors: warn + continue so one
                # broken optional plugin does not break adapter lookup.
                _warnings.warn(
                    f"Entry-point plugin {ep.name!r} failed to register "
                    f"({type(_e).__name__}: {_e}). Skipping.",
                    ImportWarning,
                    stacklevel=2,
                )
        _PLUGINS_DISCOVERED = True


def _discover_plugins_once() -> None:
    """Load external plugins only after an explicit unknown-arch lookup."""
    with _REGISTRY_LOCK:
        if _PLUGINS_DISCOVERED:
            return
        discover_entry_points()


def discover_all(
    builtins: Iterable[tuple[str, str, str]] = (),
) -> None:
    """Cold-start bootstrap: register built-ins without executing plugins."""
    register_builtins(builtins)


discover_plugins = discover_entry_points


__all__ = [
    "discover_all",
    "discover_plugins",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
