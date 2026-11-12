#!/usr/bin/env python3
"""Verify Developer Certificate of Origin (Signed-off-by) trailers on commits."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]

_DCO_PREFIX = "Signed-off-by:"
_TRAILER_RE = re.compile(
    r"^Signed-off-by:\s+(?P<name>[^<>\r\n]+?)\s+<(?P<email>[^<>\s@]+@[^<>\s@]+\.[^<>\s@]+)>\s*$"
)
_INTERNAL_DOMAIN = ".".join(("rhino", "auto"))


def _is_internal_email(email: str) -> bool:
    domain = email.rsplit("@", 1)[-1].lower().strip(".")
    return domain == _INTERNAL_DOMAIN or domain.endswith("." + _INTERNAL_DOMAIN)


def validate_commit_message(message: str) -> tuple[bool, str]:
    """Validate DCO sign-off trailers in a single non-merge commit message."""
    valid_signoffs: list[tuple[str, str]] = []
    for raw_line in message.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith(_DCO_PREFIX.lower()):
            continue
        match = _TRAILER_RE.match(line)
        if match is None:
            return False, f"malformed DCO trailer: {line!r} (expected 'Signed-off-by: Name <email>')"
        name = match.group("name").strip()
        email = match.group("email").strip()
        if not name:
            return False, f"empty signer name in DCO trailer: {line!r}"
        if _is_internal_email(email):
            return False, f"internal domain email is not permitted in public DCO sign-off: <{email}>"
        valid_signoffs.append((name, email))
    if not valid_signoffs:
        return False, "missing 'Signed-off-by: Name <email>' trailer"
    return True, f"valid ({len(valid_signoffs)} sign-off(s))"


def _iter_non_merge_commits(repo: Path, rev_range: str = "") -> list[tuple[str, str]]:
    """Return [(sha, body)] for non-merge commits in `rev_range` or `HEAD`."""
    if rev_range:
        proc = subprocess.run(
            ["git", "-C", str(repo), "log", "--no-merges", "--format=%H%x00%B%x1e", rev_range],
            capture_output=True,
            text=True,
            check=True,
        )
        commits: list[tuple[str, str]] = []
        for chunk in proc.stdout.split("\x1e"):
            chunk = chunk.lstrip("\r\n")
            if not chunk.strip():
                continue
            sha, sep, body = chunk.partition("\x00")
            if sep and sha.strip():
                commits.append((sha.strip(), body))
        return commits

    proc = subprocess.run(
        ["git", "-C", str(repo), "show", "-s", "--format=%P%x00%H%x00%B", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    parents, _, rest = proc.stdout.partition("\x00")
    if len(parents.split()) > 1:
        return []
    sha, sep, body = rest.partition("\x00")
    return [(sha.strip(), body)] if sep and sha.strip() else []


def check_commits(repo: Path, rev_range: str = "") -> tuple[int, list[tuple[str, str]]]:
    """Return (checked_count, [(sha, error_reason)]) for `rev_range` or `HEAD`."""
    commits = _iter_non_merge_commits(repo, rev_range)
    failures: list[tuple[str, str]] = []
    for sha, body in commits:
        ok, reason = validate_commit_message(body)
        if not ok:
            failures.append((sha, reason))
    return len(commits), failures


def _self_test() -> int:
    corp_email = "dev@" + _INTERNAL_DOMAIN
    sub_corp_email = "dev@ci." + _INTERNAL_DOMAIN
    should_pass = [
        "feat: add widget\n\nSigned-off-by: Jane Doe <jane.doe@example.com>\n",
        "fix: handle edge case\n\nCo-authored-by: Peer <peer@example.org>\nSigned-off-by: xup16 <xuping908@gmail.com>\n",
    ]
    should_fail = [
        ("feat: missing trailer\n", "missing"),
        ("feat: malformed\n\nSigned-off-by: Jane Doe\n", "malformed"),
        ("feat: bad email\n\nSigned-off-by: Jane Doe <not-an-email>\n", "malformed"),
        (f"feat: internal email\n\nSigned-off-by: Dev <{corp_email}>\n", "internal domain"),
        (f"feat: internal subdomain\n\nSigned-off-by: Dev <{sub_corp_email}>\n", "internal domain"),
        (
            f"feat: mixed signoffs\n\nSigned-off-by: Jane Doe <jane@example.com>\nSigned-off-by: Dev <{corp_email}>\n",
            "internal domain",
        ),
    ]
    bad = 0
    for msg in should_pass:
        ok, reason = validate_commit_message(msg)
        if not ok:
            print(f"self-test FAIL (expected PASS): {reason} | {msg!r}", file=sys.stderr)
            bad += 1
    for msg, expected in should_fail:
        ok, reason = validate_commit_message(msg)
        if ok or expected not in reason:
            print(f"self-test FAIL (expected {expected!r}): ok={ok} reason={reason!r}", file=sys.stderr)
            bad += 1
    if bad:
        print(f"[check_dco --self-test] {bad} FAIL", file=sys.stderr)
        return 1
    print("[check_dco --self-test] PASS", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", "--root", dest="repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--range", dest="rev_range", default="",
                        help="git revision range such as base..head (defaults to HEAD)")
    parser.add_argument("--self-test", action="store_true",
                        help="run built-in DCO validator assertions and exit")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    try:
        checked, failures = check_commits(args.repo, args.rev_range)
    except subprocess.CalledProcessError as exc:
        print(f"[check_dco] FAIL: git command failed: {exc.stderr or exc}", file=sys.stderr)
        return 1
    if failures:
        for sha, reason in failures:
            print(f"FAIL {sha[:12]}: {reason}", file=sys.stderr)
        print(
            f"[check_dco] FAIL: {len(failures)}/{checked} non-merge commit(s) failed DCO check "
            "(amend with `git commit -s` using a public contributor email).",
            file=sys.stderr,
        )
        return 1
    target = args.rev_range or "HEAD"
    print(f"[check_dco] PASS ({checked} non-merge commit(s) verified in {target})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
