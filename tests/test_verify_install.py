"""Host-only checks for the installation verifier's metadata-only mode."""

from __future__ import annotations

import builtins
import importlib.machinery
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import runpy
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "verify_install.py"
BACKEND_ORIGIN = "/fake/site-packages/rpu_backend/__init__.py"


def _load_config_check(monkeypatch, *, missing_module=None, legacy=False):
    versions = {"rhinoforge": "1.0.0", "torch": "2.10.0"}
    if legacy:
        versions["rpu_backend"] = "0.9.0"

    def version(name):
        if name not in versions:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    original_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in ("torch", "rpu_backend"):
            if name == missing_module:
                return None
            origin = BACKEND_ORIGIN if name == "rpu_backend" else "/fake/torch/__init__.py"
            return importlib.machinery.ModuleSpec(name, loader=None, origin=origin)
        return original_find_spec(name, *args, **kwargs)

    original_import = builtins.__import__

    def host_only_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in ("torch", "rpu_backend"):
            raise AssertionError(f"--check-config imported runtime module {name}")
        return original_import(name, *args, **kwargs)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("--check-config called os.open")

    monkeypatch.setattr(importlib.metadata, "version", version)
    monkeypatch.setattr(
        importlib.metadata, "packages_distributions",
        lambda: {"rpu_backend": ["rhinoforge"]},
    )
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(builtins, "__import__", host_only_import)
    monkeypatch.setattr(os, "open", forbidden_open)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--check-config"])
    return runpy.run_path(str(SCRIPT))["main"]


def test_check_config_uses_metadata_without_importing_runtime_or_opening_device(
    monkeypatch, capsys,
):
    main = _load_config_check(monkeypatch)

    assert main() == 0

    output = capsys.readouterr().out
    assert "configuration OK" in output
    assert "1.0.0" in output
    assert "2.10.0" in output
    assert BACKEND_ORIGIN in output
    assert "api=" not in output
    assert "RPU ready" not in output


@pytest.mark.parametrize(
    ("missing_module", "legacy", "expected_error"),
    [
        ("torch", False, "torch"),
        ("rpu_backend", False, "rpu_backend"),
        (None, True, "legacy"),
    ],
)
def test_check_config_rejects_incomplete_or_conflicting_installation(
    monkeypatch, capsys, missing_module, legacy, expected_error,
):
    main = _load_config_check(monkeypatch, missing_module=missing_module, legacy=legacy)

    with pytest.raises(SystemExit, match=expected_error):
        main()

    assert "configuration OK" not in capsys.readouterr().out
