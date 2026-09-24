#!/usr/bin/env python3
"""Verify that RhinoForge can import and access the configured RPU."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="check package metadata and module locations without accessing the RPU",
    )
    args = parser.parse_args()

    installed_version = importlib.metadata.version("rhinoforge")
    owners = importlib.metadata.packages_distributions().get("rpu_backend") or []
    if "rhinoforge" not in owners:
        raise SystemExit(
            "the rpu_backend import is not owned by the rhinoforge distribution"
        )
    try:
        importlib.metadata.version("rpu_backend")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise SystemExit(
            "the legacy rpu_backend distribution is installed; use a fresh "
            "environment or uninstall it before installing RhinoForge"
        )

    if args.check_config:
        # Only discover top-level modules: importing either package can load
        # the native backend and initialize the board runtime.
        for name in ("torch", "rpu_backend"):
            spec = importlib.util.find_spec(name)
            if spec is None:
                raise SystemExit(f"Python module is not discoverable: {name}")
        torch_version = importlib.metadata.version("torch")
        print(
            f"configuration OK: rhinoforge={installed_version}, "
            f"torch={torch_version}"
        )
        print(f"rpu_backend location: {spec.origin}")
        return 0

    import torch
    import rpu_backend
    from rpu_backend.api import API_VERSION

    if installed_version != rpu_backend.__version__:
        raise SystemExit(
            "RhinoForge package metadata and import version differ: "
            f"{installed_version} != {rpu_backend.__version__}"
        )

    ref_path = os.environ.get("RPU_KERNEL_LIB_PATH")
    if ref_path and not Path(ref_path).expanduser().is_file():
        raise SystemExit(f"RPU_KERNEL_LIB_PATH does not name a file: {ref_path}")
    if not torch.rpu.is_available():
        raise SystemExit(
            "RPU is unavailable. Check board device permissions and the Rhino "
            "Launch installation."
        )

    expected = torch.arange(4, dtype=torch.float16)
    actual = expected.to("rpu").cpu()
    if not torch.equal(actual, expected):
        raise SystemExit("RPU host/device copy check failed")
    print(
        f"RhinoForge {rpu_backend.__version__} (API {API_VERSION}): RPU ready"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
