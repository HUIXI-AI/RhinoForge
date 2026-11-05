#!/usr/bin/env python3
"""Check docs/runtime_config.md against the code, in BOTH directions."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_ROOT = REPO_ROOT / "python" / "rpu_backend"
CPP_ROOT = REPO_ROOT / "src"
RUNTIME_CONFIG = REPO_ROOT / "docs" / "runtime_config.md"
# Internal integration controls retain their runtime readers without becoming
# part of the public configuration reference.
INTERNAL_ENV_NAMES = frozenset({
    "RPU_WALL_OSS_RTC_ALLOW_NUMERIC_BLOCKED",
    "RPU_WALL_OSS_RTC_EMBODIMENT",
})

ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
ENV_PREFIXES = (
    "RPU_",
    "QWEN3_",
    "HALO_",
    "WALL_OSS_",
    "HF_",
    "HUGGINGFACE_",
    "TRANSFORMERS_",
    "LKN_",
)
CPP_GETENV_RE = re.compile(
    r'(?:std::)?getenv\(\s*"([A-Z_][A-Z0-9_]*)"'
)
DOC_ENV_RE = re.compile(r"`([A-Z_][A-Z0-9_]*)`")
CPP_SUFFIXES = {".cc", ".cpp", ".h", ".hpp", ".inc"}

# Direction-2 surface: anywhere in the repo a documented name may legitimately
# be referenced from (production, tests, harnesses, CI, perf configs).
REFERENCE_ROOTS = (
    "python", "tests", "scripts", "src", "examples",
    ".gitlab-ci.yml",
)
REFERENCE_SUFFIXES = {
    ".py", ".pyi", ".sh", ".yml", ".yaml", ".toml", ".cfg", ".txt",
    ".cc", ".cpp", ".h", ".hpp", ".inc",
}


class CheckError(RuntimeError):
    """A source file or the runtime configuration contract is malformed."""


def _literal_string(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _module_string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value.value
    return constants


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_direct_env_call(func_name: str) -> bool:
    return (
        func_name.endswith(".getenv")
        or ("environ" in func_name and func_name.endswith((".get", ".setdefault")))
    )


def _looks_like_env_name(value: str) -> bool:
    return bool(ENV_NAME_RE.fullmatch(value))


def extract_python_source(source: str, filename: str = "<source>") -> dict[str, set[int]]:
    """Return literal environment reads and their line numbers from Python."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise CheckError(f"cannot parse {filename}: {exc}") from exc
    constants = _module_string_constants(tree)
    found: dict[str, set[int]] = defaultdict(set)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            value = _literal_string(node.args[0], constants)
            if value is None or not _looks_like_env_name(value):
                continue
            func_name = _call_name(node.func)
            # Direct os.environ/os.getenv reads accept any conventional env name.
            # Helper calls are included when their literal has a project/runtime
            # prefix; this catches _env_truthy("RPU_...") without knowing every
            # helper spelling in advance.
            if _is_direct_env_call(func_name) or value.startswith(ENV_PREFIXES):
                found[value].add(node.lineno)
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            owner = _call_name(node.value)
            if not owner.endswith(".environ"):
                continue
            value = _literal_string(node.slice, constants)
            if value is not None and _looks_like_env_name(value):
                found[value].add(node.lineno)
    return found


def extract_cpp_source(source: str) -> dict[str, set[int]]:
    """Return literal C/C++ getenv reads and their line numbers."""
    found: dict[str, set[int]] = defaultdict(set)
    for match in CPP_GETENV_RE.finditer(source):
        found[match.group(1)].add(source.count("\n", 0, match.start()) + 1)
    return found


def _merge_locations(
    target: dict[str, list[str]], path: Path, values: dict[str, set[int]]
) -> None:
    rel = path.relative_to(REPO_ROOT)
    for name, lines in values.items():
        target[name].extend(f"{rel}:{line}" for line in sorted(lines))


def source_env_reads() -> dict[str, list[str]]:
    """Collect environment reads from the production source roots."""
    found: dict[str, list[str]] = defaultdict(list)
    for path in sorted(PYTHON_ROOT.rglob("*.py")):
        rel_parts = path.relative_to(PYTHON_ROOT).parts
        if "tests" in rel_parts or "debug" in rel_parts or "__pycache__" in rel_parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CheckError(f"cannot read {path}: {exc}") from exc
        _merge_locations(found, path, extract_python_source(source, str(path)))

    for path in sorted(CPP_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in CPP_SUFFIXES:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CheckError(f"cannot read {path}: {exc}") from exc
        _merge_locations(found, path, extract_cpp_source(source))
    return dict(found)


def documented_envs() -> set[str]:
    try:
        source = RUNTIME_CONFIG.read_text(encoding="utf-8")
    except OSError as exc:
        raise CheckError(f"cannot read {RUNTIME_CONFIG}: {exc}") from exc
    return set(DOC_ENV_RE.findall(source))


def extract_declared_rows(source: str) -> dict[str, int]:
    """Return {env name: doc line} for names DECLARED in a table's first cell.

    A prose mention is not a declaration: only the leading cell of a markdown
    table row states "this file owns a row for this variable". Rows that name
    two variants in one cell (``\\`A\\` / \\`B\\```) declare both.
    """
    declared: dict[str, int] = {}
    for lineno, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        first_cell = stripped.strip("|").split("|")[0]
        for name in DOC_ENV_RE.findall(first_cell):
            if name.startswith(ENV_PREFIXES):
                declared.setdefault(name, lineno)
    return declared


def documented_env_rows() -> dict[str, int]:
    try:
        source = RUNTIME_CONFIG.read_text(encoding="utf-8")
    except OSError as exc:
        raise CheckError(f"cannot read {RUNTIME_CONFIG}: {exc}") from exc
    return extract_declared_rows(source)


def referenced_env_names(names: set[str]) -> set[str]:
    """Return the subset of `names` that appears anywhere under the code roots.

    Whole-word textual scan, not an AST one: the readers this has to see are
    spread over pytest modules, shell helpers, the CI YAML, perf-config TOMLs
    and C++ test sources, and the question here is only "does anything still
    mention this name", not "is it a well-formed read".
    """
    if not names:
        return set()
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b")
    found: set[str] = set()
    for root in REFERENCE_ROOTS:
        path = REPO_ROOT / root
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = [
                p for p in path.rglob("*")
                if p.is_file()
                and p.suffix in REFERENCE_SUFFIXES
                and "__pycache__" not in p.parts
            ]
        else:
            continue
        for candidate in candidates:
            try:
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                raise CheckError(f"cannot read {candidate}: {exc}") from exc
            found.update(pattern.findall(text))
            if found >= names:
                return found
    return found


def _self_test() -> int:
    py = '''
import os
NAME = "RPU_FROM_CONSTANT"
a = os.environ.get("ORDINARY_ENV")
b = os.getenv("HF_HOME")
c = _env_truthy("RPU_VIA_HELPER")
d = os.environ[NAME]
os.environ["WRITE_ONLY"] = "1"
'''
    expected_py = {"ORDINARY_ENV", "HF_HOME", "RPU_VIA_HELPER", "RPU_FROM_CONSTANT"}
    actual_py = set(extract_python_source(py))
    cpp = 'auto a = std::getenv("RPU_CPP"); auto b = getenv("HALO_CPP");'
    expected_cpp = {"RPU_CPP", "HALO_CPP"}
    actual_cpp = set(extract_cpp_source(cpp))
    if actual_py != expected_py or actual_cpp != expected_cpp:
        print(
            "FAIL: runtime-config checker self-test\n"
            f"  Python expected={sorted(expected_py)} actual={sorted(actual_py)}\n"
            f"  C++ expected={sorted(expected_cpp)} actual={sorted(actual_cpp)}",
            file=sys.stderr,
        )
        return 1

    # Direction 2: only a table's FIRST cell declares a row. A prose mention
    # (`RPU_PROSE_ONLY`) must not be treated as a declaration, or every
    # narrative aside in the file would become a ghost candidate.
    doc = (
        "Setting `RPU_PROSE_ONLY` also affects things.\n"
        "\n"
        "| Variable | Default |\n"
        "|---|---|\n"
        "| `RPU_ROW_ONE` | `0` |\n"
        "| `RPU_ROW_TWO` / `LKN_ROW_TWO_ALT` | `1` (mentions `RPU_IN_BODY`) |\n"
        "| `NOT_A_PREFIX_VAR` | `0` |\n"
    )
    expected_rows = {"RPU_ROW_ONE", "RPU_ROW_TWO", "LKN_ROW_TWO_ALT"}
    actual_rows = set(extract_declared_rows(doc))
    if actual_rows != expected_rows:
        print(
            "FAIL: runtime-config declared-row self-test\n"
            f"  expected={sorted(expected_rows)} actual={sorted(actual_rows)}",
            file=sys.stderr,
        )
        return 1

    # A name no file under the reference roots mentions must come back unfound
    # — otherwise the ghost check can never go red. The probe name is assembled
    # at runtime so this very file does not contain it as a literal (which is
    # exactly the trap: `scripts/` is one of the roots being scanned).
    ghost = "RPU_" + "GHOST" + "_NEVER_" + "REFERENCED"
    probe = referenced_env_names({"RPU_LOG_LEVEL", ghost})
    if probe != {"RPU_LOG_LEVEL"}:
        print(
            "FAIL: runtime-config reference-scan self-test\n"
            f"  expected={{'RPU_LOG_LEVEL'}} actual={sorted(probe)}",
            file=sys.stderr,
        )
        return 1

    print("PASS: runtime-config environment scanner self-test")
    return 0


def _check() -> int:
    try:
        reads = source_env_reads()
        documented = documented_envs()
        declared = documented_env_rows()
        referenced = referenced_env_names(set(declared))
    except CheckError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    failed = False

    public_reads = set(reads) - INTERNAL_ENV_NAMES
    missing = sorted(public_reads - documented)
    if missing:
        print(
            "FAIL: production environment reads missing from "
            "docs/runtime_config.md:",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  {name}: {', '.join(reads[name])}", file=sys.stderr)
        failed = True

    ghosts = sorted(set(declared) - referenced)
    if ghosts:
        print(
            "FAIL: docs/runtime_config.md declares variables nothing in the "
            "repo references (delete the row, or restore the reader):",
            file=sys.stderr,
        )
        for name in ghosts:
            print(
                f"  {name}: docs/runtime_config.md:{declared[name]}",
                file=sys.stderr,
            )
        failed = True

    if failed:
        return 1

    print(
        "PASS: docs/runtime_config.md covers all "
        f"{len(public_reads)} public production environment reads, and all "
        f"{len(declared)} documented variables are referenced"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    return _self_test() if args.self_test else _check()


if __name__ == "__main__":
    raise SystemExit(main())
