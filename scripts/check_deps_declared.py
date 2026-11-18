#!/usr/bin/env python3
"""Check that CI test imports have declared dependencies."""
from __future__ import annotations

import argparse
import ast
import pathlib
import sys
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_no_hardcoded_paths import (  # noqa: E402  single source of truth
    iter_ci_executed_files,
)

# import-name → PyPI dist-name, for the handful where they differ (this repo's
# deps + common ecosystem cases). Everything else: import name == dist name.
IMPORT_TO_DIST = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "skbuild": "scikit-build",
    "scikit_build_core": "scikit-build-core",
    "typing_extensions": "typing-extensions",
}

# First-party (never declared as a dependency of itself).
FIRST_PARTY = {"rpu_backend", "scripts", "examples"}

ARTIFACT_PROVIDED_IMPORTS = {}


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _stdlib() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", set()))
    # build/test tooling that ships with the interpreter or is always present
    names |= {"setuptools", "pkg_resources", "pytest", "_pytest"}
    return names


def _local_modules() -> set[str]:
    """Top-level names that resolve to an in-repo module — NOT third-party deps.
    Covers sys.path-hack imports like `import rhino_e2e` (a local test helper
    at tests/model/rhinovla/rhino_e2e.py), which must not be flagged."""
    names: set[str] = set()
    for rel in ("python", "tests", "scripts", "examples"):
        root = REPO_ROOT / rel
        for path in root.rglob("*.py"):
            names.add(path.stem)
        for init in root.rglob("__init__.py"):
            names.add(init.parent.name)
    names.update(path.stem for _, path in iter_ci_executed_files())
    names.update(path.stem for path in REPO_ROOT.glob("*.py"))
    return names


def _declared_dists() -> set[str]:
    """Package names declared in pyproject.toml and the host dependency lock (version/extras/markers stripped)."""
    out: set[str] = set()
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    project = metadata["project"]
    requirements = list(project["dependencies"]) + list(metadata["build-system"]["requires"])
    for group in project.get("optional-dependencies", {}).values():
        requirements.extend(group)
    requirements.extend((REPO_ROOT / "scripts/ci/host-image/requirements-host.txt").read_text().splitlines())
    for raw in requirements:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue  # blanks, comments, pip flags (--index-url / --no-binary …)
        name = line
        for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", ";", "[", " @ "):
            name = name.split(sep, 1)[0]
        out.add(_norm(name))
    return out


def _top_imports(path: pathlib.Path) -> set[str]:
    """Top-level module of every absolute import in the file (any nesting)."""
    mods: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"),
                     filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def _lint() -> int:
    stdlib = _stdlib()
    local = _local_modules()
    declared = _declared_dists()
    undeclared: dict[str, set[str]] = {}
    ci_files = tuple(iter_ci_executed_files())
    for rel, p in ci_files:
        if not p.exists():
            print(f"FAIL {rel}: CI_EXECUTED file missing", file=sys.stderr)
            return 1
        for mod in _top_imports(p):
            if (
                mod in stdlib
                or mod in FIRST_PARTY
                or mod in local
                or mod in ARTIFACT_PROVIDED_IMPORTS.get(rel, set())
            ):
                continue
            dist = _norm(IMPORT_TO_DIST.get(mod, mod))
            if dist not in declared:
                undeclared.setdefault(dist, set()).add(rel)
    if undeclared:
        for dist, files in sorted(undeclared.items()):
            print(f"FAIL undeclared dependency '{dist}' (imported by "
                  f"{', '.join(sorted(files))}) — add it to pyproject.toml and the host dependency lock",
                  file=sys.stderr)
        print(f"[check_deps_declared] {len(undeclared)} undeclared", file=sys.stderr)
        return 1
    print(f"[check_deps_declared] PASS ({len(ci_files)} CI files; all "
          f"third-party imports declared in pyproject.toml and the host dependency lock)", file=sys.stderr)
    return 0


def _self_test() -> int:
    bad = 0
    d = _declared_dists()
    for must in ("torch", "transformers", "numpy", "lerobot"):
        if must not in d:
            print(f"self-test BAD: '{must}' not parsed from pyproject.toml and the host dependency lock",
                  file=sys.stderr)
            bad += 1
    # import→dist normalization
    if _norm(IMPORT_TO_DIST["yaml"]) != "pyyaml":
        print("self-test BAD: yaml→pyyaml mapping", file=sys.stderr)
        bad += 1
    if bad:
        print(f"[check_deps_declared --self-test] {bad} FAIL", file=sys.stderr)
        return 1
    print("[check_deps_declared --self-test] PASS", file=sys.stderr)
    return 0


def main() -> int:
    # `_stdlib()` reads `sys.stdlib_module_names` of the RUNNING interpreter, and
    # the CI-executed files this lint scans import 3.11+ stdlib modules (e.g.
    # `tomllib` in test_v5_e2e_perf.py). Under <3.11 those read as undeclared
    # third-party deps → a false positive that would wrongly block the pipeline.
    # The CI runtime is 3.12; require >=3.11 here so an env drift fails with an
    # accurate message instead. (The shipped package stays 3.10-ok — only this
    # test-harness lint needs the newer stdlib view; do NOT lower this to match
    # pyproject `requires-python = ">=3.10"`, which governs the wheel, not tests.)
    if sys.version_info < (3, 11):
        print(f"check_deps_declared requires Python >= 3.11 (the scanned harness "
              f"uses 3.11+ stdlib such as tomllib); running under "
              f"{sys.version_info.major}.{sys.version_info.minor}", file=sys.stderr)
        return 1
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    return _self_test() if args.self_test else _lint()


if __name__ == "__main__":
    sys.exit(main())
