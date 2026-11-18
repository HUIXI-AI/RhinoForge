#!/usr/bin/env python3
"""Export the validated examples catalog for consumers without importing torch."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import check_examples as examples

COMMON = examples.COMMON

ROOT = examples.ROOT
SCHEMA = "rpu-backend/example-catalog/v1"


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_config(config):
    result = copy.deepcopy(config)
    result.get("run", {}).pop("output_dir", None)
    return result


def export_catalog(*, commit=None):
    # Reuse the real validators, factory resolution and profile coverage checks.
    configs = examples.load_configs()
    catalog = examples.catalog(configs)
    registered = {(config_path, profile["id"]): (entry, profile)
                  for group in catalog.values() for entries in group.values()
                  for entry, config_path, _, profile in entries}
    entry_point = examples._runtime_entry()
    aliases = examples.COMMON.registry_aliases()
    rows = []
    for path in sorted(examples.CONFIGS.rglob("*.toml")):
        raw = examples.COMMON.normalize_config(tomllib.loads(path.read_text()))
        config = configs[path]
        metadata = examples.COMMON.example_metadata(config)
        profile_id = metadata.get("profile_id") if metadata else None
        entry, profile = registered.get((path, profile_id), (None, None))
        if metadata:
            target = examples.COMMON.config_target(config)
            alias = metadata.get("registry_alias")
            precision = metadata.get("dtype") or (profile or {}).get("precision")
        else:
            section = next((key for key in ("g05", "pi05", "hy_vla", "wall_oss", "wall_oss_sorting", "env")
                            if key in raw), None)
            target = examples.COMMON.config_target(config)
            inputs = raw.get("input", {})
            primary = ("fp16_checkpoint" if raw.get("wall_oss", {}).get("precision") == "fp16"
                       else "w8_checkpoint") if section == "wall_oss" else "checkpoint"
            alias = inputs.get(primary)
            alias = alias if alias in aliases else None
            precision = raw.get(section, {}).get("precision")
        if "env" in raw:
            entry = ROOT / "scripts/validate_rhinovla_v3.py"
        entry = entry or (entry_point if target is not None else None)
        if entry is None:
            raise ValueError(f"missing entry point for {path.relative_to(ROOT)}")
        execution = raw.get("rpu_execution", {})
        cores = execution.get("model", {}).get("num_cores", 8)
        rows.append({
            "id": path.relative_to(examples.CONFIGS).with_suffix("").as_posix(),
            "config_file": path.relative_to(ROOT).as_posix(),
            "file_sha256": file_sha(path),
            "config_sha256": identity(raw),
            "semantic_sha256": identity(semantic_config(raw)),
            "entry_point": entry.relative_to(ROOT).as_posix(),
            "entry_kind": "validation-fragment" if "env" in raw else "example",
            "profile_id": profile["id"] if profile else None,
            "asset_alias": alias,
            "precision": precision,
            "cores": cores,
            "component": profile.get("component") if profile else target,
            "support": copy.deepcopy(profile.get("support")) if profile else None,
            "config": raw,
        })
    # Bind code/config bytes, including the factory validators used above.
    files = set(ROOT.glob("examples/**/*.py")) | set(ROOT.glob("examples/configs/**/*.toml"))
    files.update(ROOT.glob("scripts/run_*optimized.py"))
    files.update(ROOT.glob("scripts/run_wall_oss*.py"))
    files.update(ROOT.glob("scripts/run_zoo_example.py"))
    files.update(ROOT.glob("examples/configs/**/*.json"))
    files.add(ROOT / "scripts/validation/wall_base.py")
    files.update({ROOT / "scripts/runtime_libraries.py", ROOT / "scripts/g05_libraries.py"})
    files.update(ROOT.glob("python/rpu_backend/api/_execution.py"))
    files.update(ROOT.glob("python/rpu_backend/model_registry.py"))
    # Factories and validators also import adapter/runtime helpers. Bind their
    # source bytes, rather than identifying a dirty checkout by HEAD alone.
    files.update(ROOT.glob("python/rpu_backend/**/*.py"))
    files.update(p for p in ROOT.glob('src/**/*') if p.is_file()
                 and (p.suffix in {'.cpp', '.cc', '.h', '.hpp', '.inc', '.def', '.cmake'} or p.name == 'CMakeLists.txt'))
    files.update({ROOT / 'CMakeLists.txt', ROOT / 'pyproject.toml', ROOT / 'requirements.txt'})
    files.update(ROOT.glob("python/rpu_backend/runtime/cpu.py"))
    files.update(ROOT.glob("manifests/*.json"))
    files.update(ROOT.glob("release/**/*.json"))
    files.update(ROOT.glob("release/**/*.cmake"))
    files.add(ROOT / "VERSION")
    files.update({Path(__file__).resolve(), Path(examples.__file__).resolve(), examples.COMMON.API_REFERENCE})
    source_files = {p.relative_to(ROOT).as_posix(): file_sha(p) for p in sorted(files) if p.is_file()}
    if commit is None:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("backend source commit provenance must be a full SHA1")
    source = {"commit": commit,
              "files": source_files, "content_sha256": identity(source_files)}
    groups = {}
    for row in rows:
        groups.setdefault(row["semantic_sha256"], []).append(row["id"])
    registry = examples.COMMON._load_file("_catalog_model_registry", ROOT / "python/rpu_backend/model_registry.py")
    return {"schema": SCHEMA, "source": source, "registry_paths": registry.MODELS, "entries": rows,
            "duplicates": [ids for ids in groups.values() if len(ids) > 1]}


def verify_source(snapshot):
    if snapshot.get("schema") != SCHEMA:
        raise ValueError("unknown examples catalog schema")
    source = snapshot["source"]
    if identity(source["files"]) != source["content_sha256"]:
        raise ValueError("examples source manifest identity differs")
    for name, digest in source["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or file_sha(path) != digest:
            raise ValueError(f"examples source changed: {name}")
    # Deployments may contain verified source bytes without Git metadata. The
    # exported commit is provenance; every source byte and catalog row still has
    # to match, including added/removed files and uncommitted edits.
    actual = export_catalog(commit=source["commit"])
    if actual != snapshot:
        raise ValueError("examples catalog changed; export and resolve again")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            verify_source(json.loads(args.verify.read_text()))
            print("PASS: examples catalog matches this checkout; no model or device loaded")
            return 0
        result = export_catalog()
        if "torch" in sys.modules:
            raise RuntimeError("catalog export imported torch")
        payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
            print(f"Exported {len(result['entries'])} configs; {len(result['duplicates'])} duplicate groups")
        else:
            print(payload, end="")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
