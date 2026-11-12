#!/usr/bin/env python3
"""Scan tracked text files to block internal network, host, path, and session leaks."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

_OCTET = r"(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])"
_LAB_SUBNET_PREFIX = r"\.".join(("10", "10"))
_WORKSTATION_PREFIX = "h" + "x"
_NAS_PREFIX = "-".join(("rpu", "nas"))
_CORP_DOMAIN = r"\.".join(("rhino", "auto"))
_NFS_MOUNT = "/" + "-".join(("nfs3", "sh"))
_PLANNING_DIR = r"\." + "planning" + "/"
_AGENT_THREAD = "codex" + r"[\s_-]+" + "thread"

FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "private_lab_ip",
        re.compile(r"(?<![0-9.])" + _LAB_SUBNET_PREFIX + r"\." + _OCTET + r"\." + _OCTET + r"(?!\.?[0-9])"),
        "private RFC1918 board or lab IPv4 address",
    ),
    (
        "internal_hostname",
        re.compile(
            r"\b(?:"
            + _WORKSTATION_PREFIX
            + r"-(?:desktop|ci(?:-[A-Za-z0-9_-]+)?|optiplex(?:-[A-Za-z0-9_-]+)?|[0-9]{2,})"
            + r"|"
            + _NAS_PREFIX
            + r"(?:-[A-Za-z0-9_-]+)?)\b",
            re.IGNORECASE,
        ),
        "internal workstation, CI runner, or NAS hostname",
    ),
    (
        "internal_email",
        re.compile(r"@[A-Za-z0-9.-]*\b" + _CORP_DOMAIN + r"\b", re.IGNORECASE),
        "internal corporate email domain",
    ),
    (
        "internal_nfs_path",
        re.compile(_NFS_MOUNT + r"(?=/|[\s\"'`:;,)\]]|$)"),
        "internal shared NFS mount path",
    ),
    (
        "planning_directory",
        re.compile(r"(?:^|[\s\"'`(/:=])" + _PLANNING_DIR),
        "internal planning directory reference",
    ),
    (
        "agent_thread_id",
        re.compile(r"\b" + _AGENT_THREAD + r"(?:\s+[0-9a-fA-F-]{8,})?\b", re.IGNORECASE),
        "internal agent thread reference",
    ),
)


def scan_text(text: str) -> list[tuple[int, str, str, str]]:
    """Return [(lineno, rule_id, reason, snippet)] for forbidden matches in `text`."""
    hits: list[tuple[int, str, str, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for rule_id, pattern, reason in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                hits.append((lineno, rule_id, reason, line.strip()[:120]))
                break
    return hits


def _iter_repo_files(root: Path, *, include_untracked: bool = False) -> list[Path]:
    cmd = ["git", "-C", str(root), "ls-files", "-z"]
    if include_untracked:
        cmd = ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    output = subprocess.check_output(cmd)
    paths: list[Path] = []
    for raw in sorted(set(output.decode("utf-8", errors="surrogateescape").split("\0"))):
        if not raw:
            continue
        candidate = root / raw
        if candidate.is_file() and not candidate.is_symlink():
            paths.append(candidate)
    return paths


def scan_repo(root: Path, *, include_untracked: bool = False) -> tuple[int, list[tuple[str, int, str, str, str]]]:
    """Scan repository text files and return (scanned_count, violations)."""
    files = _iter_repo_files(root, include_untracked=include_untracked)
    violations: list[tuple[str, int, str, str, str]] = []
    scanned = 0
    for path in files:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:8192]:
            continue
        scanned += 1
        text = raw.decode("utf-8", errors="replace")
        rel = path.relative_to(root).as_posix()
        for lineno, rule_id, reason, snippet in scan_text(text):
            violations.append((rel, lineno, rule_id, reason, snippet))
    return scanned, violations


def _self_test() -> int:
    should_hit = [
        ("ssh://git@" + ".".join(("10", "10", "88", "24")) + "/group/project.git", "private_lab_ip"),
        ("board_ip = '" + ".".join(("10", "10", "1", "153")) + "'", "private_lab_ip"),
        ("runner on " + "-".join(("hx", "desktop")), "internal_hostname"),
        ("node " + "-".join(("hx", "ci", "01")), "internal_hostname"),
        ("host " + "-".join(("hx", "OptiPlex", "7090")), "internal_hostname"),
        ("storage " + "-".join(("rpu", "nas", "02")), "internal_hostname"),
        ("Author: User <user@" + ".".join(("rhino", "auto")) + ">", "internal_email"),
        ("export PATH=/" + "-".join(("nfs3", "sh")) + "/miniconda3/bin:$PATH", "internal_nfs_path"),
        ("see ." + "planning/ROADMAP.md for notes", "planning_directory"),
        ("see " + " ".join(("codex", "thread")) + " 0195f2c4-1111-2222-3333-444455556666", "agent_thread_id"),
    ]
    should_miss = [
        "example_ip = '192.0.2.10'",
        "other_rfc1918 = '10.20.30.40'",
        "version = '1.10.10.2.0'",
        "offset = hx - 1",
        "Author: Contributor <contributor@example.com>",
        "cache_dir = '/var/cache/models'",
        "doc = 'docs/planning/overview.md'",
        "# codex round-2 review note",
    ]
    bad = 0
    for sample, expected_rule in should_hit:
        hits = scan_text(sample)
        if not hits or hits[0][1] != expected_rule:
            print(f"self-test FAIL (expected {expected_rule}): got {hits} for {sample!r}", file=sys.stderr)
            bad += 1
    for sample in should_miss:
        hits = scan_text(sample)
        if hits:
            print(f"self-test FAIL (expected clean): got {hits} for {sample!r}", file=sys.stderr)
            bad += 1
    if bad:
        print(f"[check_public_hygiene --self-test] {bad} FAIL", file=sys.stderr)
        return 1
    print("[check_public_hygiene --self-test] PASS", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--worktree", action="store_true",
                        help="include untracked non-ignored files in the hygiene scan")
    parser.add_argument("--self-test", action="store_true",
                        help="verify forbidden/allowed hygiene patterns and exit")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    scanned, violations = scan_repo(args.root, include_untracked=args.worktree)
    if violations:
        for rel, lineno, rule_id, reason, snippet in violations:
            print(f"FAIL {rel}:{lineno} [{rule_id}]: {reason}\n    {snippet}", file=sys.stderr)
        print(f"[check_public_hygiene] FAIL ({len(violations)} violation(s) across {scanned} text files)",
              file=sys.stderr)
        return 1
    print(f"[check_public_hygiene] PASS ({scanned} tracked text files clean)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
