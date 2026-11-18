#!/usr/bin/env python3
"""Prepare LeRobot 0.5.1 with its upstream Transformers compatibility fix."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import sysconfig
import tempfile
import zipfile


WHEEL_SHA256 = "bbd11021023fde0947b6d1ff1c52fe91c86a28ab09a96359892f3ef7e8866862"
CONFIG_PATH = "lerobot/policies/groot/groot_n1.py"
ORIGINAL_SHA256 = "04aa7c9beb3f084046c82a9c0feda708ed0a4a4969c449621387fa1f5f27c3ad"
PREPARED_SHA256 = "f02d434c8e230d0a8c2b354d2ebb3bb2e7aec952433ac46c2458bb565d8fc121"
UPSTREAM_FIX = "https://github.com/huggingface/lerobot/pull/3414"
UPSTREAM_COMMIT = "12b88fce029cc3a8a94b061cd9e790018873c769"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(wheel: Path, output: Path) -> dict:
    """Copy the pinned wheel to a new directory and apply one config fix."""
    payload = wheel.read_bytes()
    if sha256(payload) != WHEEL_SHA256:
        raise ValueError("expected the official lerobot-0.5.1-py3-none-any.whl")
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    with zipfile.ZipFile(wheel) as archive:
        original = archive.read(CONFIG_PATH)
        if sha256(original) != ORIGINAL_SHA256:
            raise ValueError("LeRobot configuration source does not match version 0.5.1")
        text = original.decode("utf-8")
        replacements = {
            "from dataclasses import dataclass, field\n": "",
            "@dataclass\nclass GR00TN15Config": "class GR00TN15Config",
            '    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})': "    backbone_cfg: dict",
            '    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})': "    action_head_cfg: dict",
            '    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})': "    action_horizon: int",
            '    action_dim: int = field(init=False, metadata={"help": "Action dimension."})': "    action_dim: int",
            '    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})': '    compute_dtype: str = "float32"',
        }
        for before, after in replacements.items():
            if text.count(before) != 1:
                raise ValueError("LeRobot compatibility patch does not match its source")
            text = text.replace(before, after)
        prepared = text.encode("utf-8")
        if sha256(prepared) != PREPARED_SHA256:
            raise ValueError("prepared LeRobot configuration differs from the reviewed patch")
        for member in archive.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe wheel member")
        output.mkdir(parents=True)
        archive.extractall(output)
        (output / CONFIG_PATH).write_bytes(prepared)
    license_path = "lerobot-0.5.1.dist-info/licenses/LICENSE"
    receipt = {
        "package": "lerobot", "upstream_version": "0.5.1",
        "wheel_sha256": WHEEL_SHA256, "upstream_fix": UPSTREAM_FIX,
        "upstream_commit": UPSTREAM_COMMIT,
        "modified_file": CONFIG_PATH, "original_sha256": ORIGINAL_SHA256,
        "prepared_sha256": PREPARED_SHA256,
        "license": {"path": license_path, "sha256": sha256((output / license_path).read_bytes())},
        "metadata": "Original wheel metadata and license are retained; this receipt records the source patch.",
    }
    (output / "prepared-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def private_site(target: Path) -> Path:
    """Admit only the running private venv, never its inherited dependency base."""
    target = target.absolute()
    prefix = Path(sys.prefix).resolve()
    expected = Path(sysconfig.get_path("purelib")).resolve()
    if (sys.prefix == sys.base_prefix or target.is_symlink()
            or target.resolve() != expected or not expected.is_relative_to(prefix)
            or not expected.is_dir()):
        raise ValueError("--install-into must name this private virtual environment's purelib")
    if importlib.metadata.version("transformers") != "5.19.0":
        raise ValueError("private installation requires transformers==5.19.0")
    distribution = importlib.metadata.distribution("lerobot")
    if distribution.version != "0.5.1" or Path(distribution.locate_file("")).resolve() != expected:
        raise ValueError("install the official lerobot==0.5.1 wheel into this virtual environment first")
    return expected


def verify_package(package: Path, archive: zipfile.ZipFile) -> None:
    """Reject unknown package contents before replacing a private installation."""
    expected = {name.removeprefix("lerobot/"): archive.read(name)
                for name in archive.namelist()
                if name.startswith("lerobot/") and not name.endswith("/")}
    if package.is_symlink() or not package.is_dir():
        raise ValueError("installed LeRobot package must be a regular directory")
    files = {}
    for path in package.rglob("*"):
        if path.is_symlink():
            raise ValueError("installed LeRobot package contains a symlink")
        if path.is_file() and "__pycache__" not in path.relative_to(package).parts:
            files[path.relative_to(package).as_posix()] = path
    if files.keys() != expected.keys():
        raise ValueError("installed LeRobot package has unknown or missing files")
    for relative, original in expected.items():
        actual = sha256(files[relative].read_bytes())
        accepted = {sha256(original)}
        if "lerobot/" + relative == CONFIG_PATH:
            accepted.add(PREPARED_SHA256)
        if actual not in accepted:
            raise ValueError(f"installed LeRobot source differs from version 0.5.1: {relative}")


def install(wheel: Path, target: Path) -> dict:
    target = private_site(target)
    if any(name == "lerobot" or name.startswith("lerobot.") for name in sys.modules):
        raise ValueError("prepare LeRobot before importing it")
    with tempfile.TemporaryDirectory(prefix=".lerobot-prepare-", dir=target) as temporary:
        staging = Path(temporary)
        prepared = staging / "prepared"
        receipt = prepare(wheel, prepared)
        with zipfile.ZipFile(wheel) as archive:
            verify_package(target / "lerobot", archive)
        backup = staging / "original"
        (target / "lerobot").rename(backup)
        try:
            (prepared / "lerobot").rename(target / "lerobot")
        except BaseException:
            backup.rename(target / "lerobot")
            raise
        # Keep pip's original distribution metadata/license. The source change
        # is explicit and nested fresh environments inherit this plain path.
        receipt_path = target / "lerobot-0.5.1.dist-info/prepared-source.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--install-into", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.expanduser().resolve()
    if args.install_into is not None:
        install(wheel, args.install_into.expanduser())
        print(f"Prepared LeRobot in private environment: {args.install_into}")
    else:
        prepare(wheel, args.output.expanduser().resolve())
        print(f"Prepared LeRobot source: {args.output}")


if __name__ == "__main__":
    main()
