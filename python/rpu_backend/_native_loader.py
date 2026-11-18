"""Locate and load the native backend without replacing the Python package."""
from __future__ import annotations

import importlib.util
import errno
import glob
import os
import sys
import types
from collections.abc import Mapping


_PACKAGE_METADATA = ("__file__", "__loader__", "__package__", "__path__", "__spec__")
_RPU_DEVICE_GLOB = "/dev/rpu*"
_MEM_DEVICE = "/dev/mem"


def _check_device_nodes(
    *,
    rpu_paths: tuple[str, ...] | None = None,
    mem_path: str = _MEM_DEVICE,
    open_fn=None,
    close_fn=None,
) -> None:
    """Open the required nodes, preserving the failing path and OS error.

    The native SDK probes these nodes while its shared objects initialize.  A
    preflight here keeps a normal host-only ``import rpu_backend`` from entering
    that unsafe path.  The C++ allocator repeats the guard so direct PrivateUse1
    allocation also fails with an exception if access changes after import.
    """
    candidates = tuple(sorted(glob.glob(_RPU_DEVICE_GLOB))) if rpu_paths is None else rpu_paths
    if not candidates:
        raise FileNotFoundError(errno.ENOENT, "No RPU device nodes found", _RPU_DEVICE_GLOB)
    open_fn = os.open if open_fn is None else open_fn
    close_fn = os.close if close_fn is None else close_fn
    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    opened: list[int] = []
    open_error = None
    try:
        rpu_fd = None
        failures: list[tuple[str, OSError]] = []
        for path in candidates:
            try:
                rpu_fd = open_fn(path, flags)
            except OSError as exc:
                failures.append((path, exc))
                continue
            opened.append(rpu_fd)
            break
        if rpu_fd is None:
            path, last_error = failures[-1]
            details = "; ".join(f"{path}: {exc}" for path, exc in failures)
            raise OSError(
                last_error.errno, "Cannot open any RPU device read/write; " + details, path,
            ) from last_error
        opened.append(open_fn(mem_path, flags))
    except BaseException as exc:
        open_error = exc
        raise
    finally:
        close_error = None
        for fd in reversed(opened):
            try:
                close_fn(fd)
            except OSError as exc:
                if open_error is not None:
                    # BaseException.add_note was introduced in Python 3.11.
                    add_note = getattr(open_error, "add_note", None)
                    if callable(add_note):
                        add_note(f"Device preflight cleanup also failed: {exc}")
                elif close_error is None:
                    close_error = exc
        if close_error is not None:
            raise close_error


def device_nodes_accessible(
    *,
    rpu_paths: tuple[str, ...] | None = None,
    mem_path: str = _MEM_DEVICE,
    open_fn=None,
    close_fn=None,
) -> bool:
    """Keep CPU-only imports and availability queries safe without device access."""
    try:
        _check_device_nodes(
            rpu_paths=rpu_paths, mem_path=mem_path, open_fn=open_fn, close_fn=close_fn,
        )
    except OSError:
        return False
    return True


def require_native_backend() -> None:
    """Fail before loading or transforming weights for an explicit RPU request.

    Access is checked on every request: permissions can change after import.
    Native registration itself is import-bound and is never retried in place.
    """
    from rpu_backend.runtime._errors import RPUBackendError

    try:
        _check_device_nodes()
    except OSError as exc:
        raise RPUBackendError(
            f"RPU device preflight failed: {exc}. Rhino Launch requires read/write "
            "access to an RPU device (/dev/rpu*) and /dev/mem. Ask the board "
            "administrator to check the driver and grant this account the required "
            "device permissions; then start a new Python process. "
            "See docs/getting_started.md."
        ) from exc

    if not getattr(sys.modules.get("rpu_backend"), "_cpp_loaded", False):
        raise RPUBackendError(
            "RPU native backend is not loaded in this process. Install/build "
            "rpu_backend.so with the matching runtime assets, check the import "
            "warning above, and start a new Python process. Restart is also "
            "required if device permissions were fixed after importing rpu_backend. "
            "See docs/getting_started.md and docs/runtime_assets.md."
        )


def find_shared_object(
    *,
    package_dir: str | None = None,
    executable: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve an explicit extension, an installed package, or a source build.

    ``RPU_BACKEND_SO`` overrides discovery. Editable installs retain their
    registered extension. Source lookup is limited to ``build_<environment>``
    and ``build``; unrelated build directories are never scanned.
    """
    explicit_directory = package_dir is not None
    package_dir = package_dir or os.path.dirname(__file__)
    executable = executable or sys.executable
    environ = os.environ if environ is None else environ
    repo_root = os.path.abspath(os.path.join(package_dir, "../.."))
    env_name = os.path.basename(os.path.dirname(os.path.dirname(executable)))

    explicit = environ.get("RPU_BACKEND_SO", "")
    if explicit:
        explicit = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"RPU_BACKEND_SO does not name a file: {explicit}")
        return explicit
    packaged = os.path.join(package_dir, "rpu_backend.so")
    if os.path.isfile(packaged):
        return packaged

    if not explicit_directory:
        # Editable builds register the extension outside the package directory.
        spec = importlib.util.find_spec(f"{__package__}.rpu_backend")
        if spec is not None and spec.origin is not None:
            if os.path.basename(spec.origin) == "rpu_backend.so" and os.path.isfile(spec.origin):
                return spec.origin

    search_roots = [
        os.path.join(repo_root, f"build_{env_name}"),
        os.path.join(repo_root, "build"),
    ]
    for root_dir in search_roots:
        if not os.path.isdir(root_dir):
            continue
        matches = []
        for subdir, _, files in os.walk(root_dir):
            if "rpu_backend.so" in files:
                matches.append(os.path.join(subdir, "rpu_backend.so"))
        if matches:
            return sorted(matches)[0]
    return None


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
        raise ImportError(f"cannot create an import spec for device backend: {path}")
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
    if path is None:
        print("[RhinoForge] warning: C++ backend (.so) not found. native backend not loaded.")
    else:
        try:
            _check_device_nodes()
        except OSError as exc:
            print(
                f"[RhinoForge] warning: device nodes are unavailable: {exc}; "
                "native backend not loaded. CPU-only use remains available."
            )
        else:
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


__all__ = ["device_nodes_accessible", "find_shared_object", "load_cpp_extension", "require_native_backend"]
