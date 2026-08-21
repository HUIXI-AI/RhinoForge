"""Locate and load the native backend without replacing the Python package."""
from __future__ import annotations

import importlib.util
import glob
import os
import sys
import types


_PACKAGE_METADATA = ("__file__", "__loader__", "__package__", "__path__", "__spec__")
_RPU_DEVICE_GLOB = "/dev/rpu*"
_MEM_DEVICE = "/dev/mem"


def device_nodes_accessible(
    *,
    rpu_paths: tuple[str, ...] | None = None,
    mem_path: str = _MEM_DEVICE,
    open_fn=None,
    close_fn=None,
) -> bool:
    """Return whether this process can open the device nodes required by Rhino Launch.

    The native SDK probes these nodes while its shared objects initialize.  A
    preflight here keeps a normal host-only ``import rpu_backend`` from entering
    that unsafe path.  The C++ allocator repeats the guard so direct PrivateUse1
    allocation also fails with an exception if access changes after import.
    """
    candidates = tuple(sorted(glob.glob(_RPU_DEVICE_GLOB))) if rpu_paths is None else rpu_paths
    if not candidates:
        return False
    open_fn = os.open if open_fn is None else open_fn
    close_fn = os.close if close_fn is None else close_fn
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    opened: list[int] = []
    try:
        rpu_fd = None
        for path in candidates:
            try:
                rpu_fd = open_fn(path, flags)
            except OSError:
                continue
            opened.append(rpu_fd)
            break
        if rpu_fd is None:
            return False
        opened.append(open_fn(mem_path, flags))
        return True
    except OSError:
        return False
    finally:
        for fd in reversed(opened):
            close_fn(fd)


def find_shared_object(
    *,
    package_dir: str | None = None,
) -> str | None:
    """Return the packaged backend shared object, if installed."""
    package_dir = package_dir or os.path.dirname(__file__)
    packaged = os.path.join(package_dir, "rpu_backend.so")
    return packaged if os.path.isfile(packaged) else None


def _load_extension(path: str, package_name: str) -> types.ModuleType:
    """Execute the extension and restore package metadata it may overwrite."""
    parent = sys.modules.get(package_name)
    parent_metadata = {}
    if parent is not None:
        parent_metadata = {
            name: getattr(parent, name)
            for name in _PACKAGE_METADATA
            if hasattr(parent, name)
        }

    spec = importlib.util.spec_from_file_location(package_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create an import spec for RPU backend: {path}")
    extension = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(extension)
    finally:
        if parent is not None:
            sys.modules[package_name] = parent
            for name, value in parent_metadata.items():
                setattr(parent, name, value)
    return extension


def load_cpp_extension(
    package_name: str = "rpu_backend",
) -> tuple[types.ModuleType, bool]:
    """Load the backend and install its public symbols in ``package._cpp_ext``."""
    path = find_shared_object()
    extension = None
    if path is not None and device_nodes_accessible():
        extension = _load_extension(path, package_name)

    cpp_ext = types.ModuleType(f"{package_name}._cpp_ext")
    if extension is not None:
        for name in dir(extension):
            if not name.startswith("_"):
                setattr(cpp_ext, name, getattr(extension, name))

    sys.modules[cpp_ext.__name__] = cpp_ext
    parent = sys.modules.get(package_name)
    loaded = extension is not None
    if parent is not None:
        parent._cpp_ext = cpp_ext
        parent._cpp_loaded = loaded
    return cpp_ext, loaded


__all__ = ["device_nodes_accessible", "find_shared_object", "load_cpp_extension"]
