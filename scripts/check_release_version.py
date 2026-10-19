#!/usr/bin/env python3
"""Check distribution, build and runtime versions without importing the backend."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(
    r"(?P<base>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"
    r"(?P<stage>rc[1-9][0-9]*|\.dev(?:0|[1-9][0-9]*))?"
)


def check(root: Path, *, branch: str = "", tag: str = "") -> dict:
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError("VERSION must be a canonical release, rc or dev version")
    base, stage = match["base"], match["stage"] or ""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    if "version" in project["project"] or "version" not in project["project"].get("dynamic", []):
        raise ValueError("Python package version must be read dynamically from VERSION")
    metadata = project["tool"]["scikit-build"]["metadata"]["version"]
    if metadata["provider"] != "scikit_build_core.metadata.regex" or metadata["input"] != "VERSION":
        raise ValueError("Python metadata must use the VERSION regex provider")
    metadata_match = re.fullmatch(metadata["regex"], version)
    if metadata_match is None or metadata_match["value"] != version:
        raise ValueError("Python metadata does not preserve the complete version")
    expected_classifier = ("3 - Alpha" if stage.startswith(".dev") else
                           "4 - Beta" if stage.startswith("rc") else "5 - Production/Stable")
    if "Development Status :: " + expected_classifier not in project["project"]["classifiers"]:
        raise ValueError("Python development-status classifier must match the release stage")
    dependencies = json.loads((root / "release/runtime-dependencies.json").read_text())
    runtime_version = dependencies["runtime_bundle"]["version"]
    if VERSION_PATTERN.fullmatch(runtime_version) is None:
        raise ValueError("Runtime bundle must have a canonical version")
    launch = dependencies["launch_kernel"]
    cmake_version = launch["cmake_version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", cmake_version) or launch["version"] != cmake_version:
        raise ValueError("Launch version fields disagree")
    release_version = launch["release_version"]
    if release_version is not None:
        release_match = VERSION_PATTERN.fullmatch(release_version)
        if release_match is None or release_match["base"] != cmake_version:
            raise ValueError("Launch release and CMake versions disagree")
    files = launch["files"]
    if not files or not all(
            re.fullmatch(r"(?:include|lib)/[A-Za-z0-9_./-]+", name) and ".." not in Path(name).parts
            and re.fullmatch(r"[0-9a-f]{64}", digest) for name, digest in files.items()):
        raise ValueError("Launch package requires relative file paths and SHA256 identities")
    if not isinstance(launch["abi_version"], int) or launch["abi_version"] < 1:
        raise ValueError("Launch ABI version must be a positive integer")
    if not re.fullmatch(r"[0-9a-f]{40}", launch["commit"]):
        raise ValueError("Launch source must be pinned to a full commit")
    if branch == "main" and not stage.startswith(".dev"):
        raise ValueError("main must carry a development version")
    if branch.startswith("release/"):
        if branch != "release/" + ".".join(base.split(".")[:2]) or stage.startswith(".dev"):
            raise ValueError("release branches must match the version series and contain rc or stable versions")
    if tag and (tag != "v" + version or stage.startswith(".dev")):
        raise ValueError("release tag must match VERSION and cannot identify a dev build")
    return {"version": version, "cmake_version": base, "stage": stage or "stable",
            "runtime_bundle_version": runtime_version, "launch_commit": launch["commit"], "launch_abi": launch["abi_version"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--branch", default=os.environ.get("CI_COMMIT_BRANCH", ""))
    parser.add_argument("--tag", default=os.environ.get("CI_COMMIT_TAG", ""))
    args = parser.parse_args()
    try:
        result = check(args.root, branch=args.branch, tag=args.tag)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(1, f"Version check failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
