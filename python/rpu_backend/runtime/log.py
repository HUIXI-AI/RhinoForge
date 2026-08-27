"""Unified debug verbosity bridge.

State syncing:
    - C++ side:   g_rpu_debug_level (std::atomic<int> 0..5) in libc-ext.
    - Python side: logging.getLogger("rpu_backend").setLevel(...).
    - set_debug_level(int) writes both atomically + a module-local mirror so
      `get_debug_level()` is coherent even in no-backend mode.

Level mapping:
    0 SILENT  → logging.CRITICAL + 1   (effectively disabled)
    1 ERROR   → logging.ERROR (40)
    2 WARN    → logging.WARNING (30)
    3 INFO    → logging.INFO (20)      ← startup default
    4 DEBUG   → logging.DEBUG (10)
    5 TRACE   → custom level 5         (addLevelName)

Startup-time C++ logs (KernelCache load banner, RpuInit, pybind11 ctor):
    these fire BEFORE this module imports, so only the `RPU_LOG_LEVEL` env
    var controls them. `set_debug_level()` does NOT retroactively suppress
    those.

fork/spawn semantics:
    - fork:  child inherits the in-memory level (C++ atomic + Python logger).
    - spawn: child only inherits `RPU_LOG_LEVEL` from os.environ; to propagate
      a runtime change, callers must set the env var explicitly before spawn.
"""
from __future__ import annotations

import logging
import sys as _sys


def _cpp_loaded() -> bool:
    """Mirror runtime/debug.py — true iff the .so was loaded at import time."""
    return ("rpu_backend._cpp_ext" in _sys.modules
            and bool(getattr(_sys.modules.get("rpu_backend"), "_cpp_loaded", False)))


def _cpp():
    return _sys.modules.get("rpu_backend._cpp_ext")


def _cpp_set(level: int) -> None:
    if _cpp_loaded() and hasattr(_cpp(), "rpu_set_debug_level"):
        _cpp().rpu_set_debug_level(level)


def _cpp_get() -> int:
    if _cpp_loaded() and hasattr(_cpp(), "rpu_get_debug_level"):
        return _cpp().rpu_get_debug_level()
    return _current_level


# Module-local mirror so `get_debug_level()` is coherent when the C++ backend
# is absent (otherwise _cpp_get would always return the INFO default,
# contradicting any `set_debug_level(4)` the user issued in Python).
_current_level: int = 3

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_LEVEL_TO_LOGGING = {
    0: logging.CRITICAL + 1,
    1: logging.ERROR,
    2: logging.WARNING,
    3: logging.INFO,
    4: logging.DEBUG,
    5: TRACE,
}

_LOG = logging.getLogger("rpu_backend")

# Handler policy: StreamHandler provides default visibility; propagate=False
# prevents duplicate output after a downstream `logging.basicConfig()`.
# Logging-capture integrations may set `_LOG.propagate = True` explicitly.
if not _LOG.handlers:
    _handler = logging.StreamHandler()  # stderr
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _LOG.addHandler(_handler)
_LOG.propagate = False


def set_debug_level(level: int) -> None:
    """Set RPU debug verbosity (0=SILENT .. 5=TRACE; default 3=INFO).

    Syncs the C++ `g_rpu_debug_level` atomic, the Python logger level, and
    the module-local mirror (used as truth source when the C++ backend is
    absent). Startup-time C++ logs are unaffected — only `RPU_LOG_LEVEL`
    env at .so load time can suppress those.
    """
    global _current_level
    if not (0 <= level <= 5):
        raise ValueError(f"level must be in [0, 5], got {level}")
    _current_level = level
    _cpp_set(level)
    _LOG.setLevel(_LEVEL_TO_LOGGING[level])


def get_debug_level() -> int:
    """Return the current debug level (0..5). Prefers the C++ atomic when
    loaded; falls back to the Python-side mirror in no-backend mode."""
    return _cpp_get()


def trace(msg, *args, **kw):
    """Convenience: log at TRACE level (5)."""
    _LOG.log(TRACE, msg, *args, **kw)


# Init-time sync: pick up whatever the C++ atomic was initialized to from
# the RPU_LOG_LEVEL env var so the Python logger starts at the matching level.
set_debug_level(_cpp_get())
