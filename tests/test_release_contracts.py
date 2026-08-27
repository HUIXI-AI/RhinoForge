from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8")
    return [
        ROOT / name
        for name in output.rstrip("\0").split("\0")
        if name and (ROOT / name).is_file()
    ]


def test_default_install_includes_model_loading_dependency() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "accelerate>=1.1.0,<2" in project["project"]["dependencies"]


def test_v1_kernel_manifest_is_generated_from_host_reachability(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "operator.ref"
    asset.write_bytes(b"opaque")
    manifest = Path(str(asset) + ".kernels")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "release/generate_kernel_manifest.py"),
            str(asset),
            str(manifest),
        ],
        check=True,
    )
    lines = manifest.read_text(encoding="ascii").splitlines()
    assert lines[:2] == ["rhinoforge-kernels-v1", "asset-size=6"]
    assert len(lines[2:]) == 209
    assert lines[2:] == sorted(set(lines[2:]))
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,254}", name) for name in lines[2:])


def test_public_model_ledger_has_complete_immutable_identities() -> None:
    path = ROOT / "release/public-models-v1.0.0.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema"] == "rhinoforge-public-models-v1"
    assert document["release"] == "1.0.0"

    profiles: set[str] = set()
    for model in document["models"]:
        assert model["profile"] not in profiles
        profiles.add(model["profile"])
        for asset in (model["checkpoint"], model.get("implementation")):
            if asset is None:
                continue
            assert asset["repository"]
            assert re.fullmatch(r"[0-9a-f]{40}", asset["revision"])
            assert asset.get("license")
        checkpoint = model["checkpoint"]
        assert checkpoint["access"] in {"public", "auto-gated", "manual-gated"}
        assert checkpoint["format"]

    serialized = json.dumps(document, sort_keys=True).lower()
    for forbidden in (
        "10." + "10.",
        "/" + "nfs",
        "customer checkpoint",
    ):
        assert forbidden not in serialized


def test_public_tree_excludes_internal_release_markers() -> None:
    files = [path for path in _tracked_files() if path != Path(__file__)]
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in files
    )
    forbidden = (
        (r"\b10(?:\.\d{1,3}){3}\b", 0),
        (r"\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b", 0),
        (r"\b192\.168(?:\.\d{1,3}){2}\b", 0),
        (r"/" + r"nfs\d*(?:[-_/]|$)", re.IGNORECASE),
        (r"(?<![A-Za-z0-9_.-])/(?:data2?|mnt/nvme)(?:/|$)", re.IGNORECASE),
        (r"\bCUSTOMER_[A-Z0-9_]+\b", 0),
        (r"\b(?:internal|private)[-_ ](?:codename|checkpoint)\b", re.IGNORECASE),
        (r"SHT_SYMTAB|\.symtab|\.strtab|EI_CLASS", 0),
        (r"rpu_(?:kernel_probe_by_name|rope_probe|rope_kernel_info)", 0),
        ("get_runtime_" + "payload_paths", 0),
        ("get_kernel_" + "args_instr", 0),
        ("hw_" + "perf_trace", re.IGNORECASE),
        ("dump_" + "hw_perf_chrome", re.IGNORECASE),
        (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", 0),
    )
    for pattern, flags in forbidden:
        assert re.search(pattern, text, flags) is None, pattern


def test_public_tree_contains_only_plain_bounded_files() -> None:
    entries = subprocess.check_output(
        ["git", "ls-files", "-s", "-z"], cwd=ROOT
    ).decode("utf-8").rstrip("\0").split("\0")
    for entry in entries:
        metadata, name = entry.split("\t", 1)
        if not (ROOT / name).is_file():
            continue
        mode, _object_id, stage = metadata.split()
        assert mode in {"100644", "100755"}, name
        assert stage == "0", name
        payload = (ROOT / name).read_bytes()
        assert len(payload) <= 5 * 1024 * 1024, name
        assert not payload.startswith(b"version https://git-lfs.github.com/spec/"), name


def test_runtime_diagnostics_do_not_expose_address_values() -> None:
    files = [
        path for path in _tracked_files()
        if path.parts[-1] != Path(__file__).name
        and any(part in {"python", "src"} for part in path.parts)
    ]
    forbidden = (
        r'<<\s*block->(?:ptr|prev|next)\b',
        r'"[^"\n]*(?:got|addr(?:ess)?|pointer|ptr)[^"\n]*",\s*'
        r'(?:[A-Za-z_]\w*->)?[A-Za-z_]\w*(?:addr|ptr)\w*\b',
        r'\bid=\{id\(',
        r'\bgm id=\{gm_id',
        r'gm_id history:\s*%s',
        r'gm_id\s*=\s*id\(gm\)',
        r'op_id_int\s*=\s*id\(callable_\)',
    )
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in forbidden:
            assert re.search(pattern, text) is None, f"{path}: {pattern}"
