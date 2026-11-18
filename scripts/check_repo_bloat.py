#!/usr/bin/env python3
"""Fail if a tracked file is large enough to be an accident."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Per-file cap for tracked content. Goldens, checkpoints and reference tensors
# are all orders of magnitude above this, so the cap separates "source" from
# "artifact" without needing to enumerate artifact types.
MAX_TRACKED_BYTES = 1_000_000

# Runtime packages and generated reports are distributed outside Git.
ALLOWED: set[str] = {"src/ops/rpu_linear_tiling_data.h"}


def oversized(entries: "list[tuple[str, int]]") -> "list[tuple[str, int]]":
    """Tracked paths above the cap that are not explicitly allowed."""
    return [(path, size) for path, size in entries
            if size > MAX_TRACKED_BYTES and path not in ALLOWED]


def _tracked_sizes() -> "list[tuple[str, int]]":
    """(path, blob size) for every tracked file, via one batched git call."""
    listing = subprocess.run(
        ["git", "ls-files", "-s"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True).stdout.splitlines()
    pairs = []
    for line in listing:
        # <mode> <sha> <stage>\t<path>
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 2 or not path:
            continue
        # Skip gitlinks (mode 160000). A submodule entry is a COMMIT sha that
        # does not exist as an object in this repository, so cat-file answers
        # "<sha> missing" — a 3-token line that used to shift every subsequent
        # size onto the wrong path and then die on int('<sha>').
        if parts[0] == "160000":
            continue
        pairs.append((parts[1], path))
    if not pairs:
        return []
    proc = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectsize)"], cwd=REPO_ROOT,
        input="\n".join(sha for sha, _ in pairs),
        capture_output=True, text=True, check=True)
    return _pair_sizes(proc.stdout.splitlines(), pairs)


def _worktree_sizes(root=REPO_ROOT) -> "list[tuple[str, int]]":
    """Current source, including new non-ignored files and unstaged deletions.

    Local validation must inspect the same bytes as other source checks. The
    default index mode remains available for auditing what is staged to commit.
    """
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, capture_output=True, text=True, check=True).stdout
    entries = []
    for name in sorted(set(listing.split("\0")) - {""}):
        path = root / name
        if path.is_symlink():
            entries.append((name, len(path.readlink().as_posix().encode())))
        elif path.is_file():
            entries.append((name, path.stat().st_size))
    return entries


def _pair_sizes(lines, pairs):
    """Zip cat-file output back onto paths, LINE BY LINE.

    Splitting the whole output and zipping tokens breaks the moment one entry
    answers with something other than a bare number ("<sha> missing"): every
    later size lands on the wrong path and the run dies on int('<sha>').
    """
    out = []
    for line, (_sha, path) in zip(lines, pairs):
        tok = line.split()
        if len(tok) == 1 and tok[0].isdigit():
            out.append((path, int(tok[0])))
        # anything else ("<sha> missing", "<sha> ambiguous") is not a sizeable blob
    return out


def _self_test() -> int:
    cap = MAX_TRACKED_BYTES
    assert oversized([("a.py", 10)]) == []
    assert oversized([("golden.pt", cap + 1)]) == [("golden.pt", cap + 1)]
    assert oversized([("a.py", cap)]) == [], "cap itself is allowed"
    for allowed in ALLOWED:
        assert oversized([(allowed, 400_000_000)]) == [], f"{allowed} is exempt"
    assert oversized([("src/other.ref", cap + 1)]) == [("src/other.ref", cap + 1)], \
        "the exemption is per-path, not per-extension"
    assert oversized([("ci/hw_closure_receipt.v1.json", cap + 1)]) == [
        ("ci/hw_closure_receipt.v1.json", cap + 1)], \
        "the source contract exemption must not admit generated test receipts"
    # A submodule gitlink whose commit object is absent (the CI checkout: the
    # object lives in the submodule, not here) answers "<sha> missing". That
    # used to shift every later size onto the wrong path and crash on int().
    pairs = [("aaa", "a.py"), ("bbb", "third_party/sub"), ("ccc", "c.py")]
    lines = ["10", "bbb missing", "30"]
    assert _pair_sizes(lines, pairs) == [("a.py", 10), ("c.py", 30)], \
        "a missing object must be skipped without desynchronising the rest"
    assert _pair_sizes(["10"], pairs) == [("a.py", 10)], "short output is safe"
    with tempfile.TemporaryDirectory() as raw:
        root = pathlib.Path(raw)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        (root / "removed.py").write_text("old")
        subprocess.run(["git", "-C", str(root), "add", "removed.py"], check=True)
        before = (root / ".git/index").read_bytes()
        (root / "removed.py").unlink()
        (root / ".gitignore").write_text("ignored.bin\n")
        (root / "ignored.bin").write_bytes(b"x" * (cap + 1))
        (root / "new.bin").write_bytes(b"x" * (cap + 1))
        (root / "link").symlink_to("new.bin")
        current = dict(_worktree_sizes(root))
        assert "removed.py" not in current and "ignored.bin" not in current
        assert current["link"] == len("new.bin")
        assert oversized(list(current.items())) == [("new.bin", cap + 1)]
        assert (root / ".git/index").read_bytes() == before
    print("check_repo_bloat self-test: OK")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return _self_test()
    entries = _worktree_sizes() if "--worktree" in sys.argv else _tracked_sizes()
    bad = sorted(oversized(entries), key=lambda kv: -kv[1])
    total = sum(size for _, size in entries)
    if bad:
        print(f"FAIL: {len(bad)} tracked file(s) above "
              f"{MAX_TRACKED_BYTES / 1e6:.1f} MB")
        for path, size in bad:
            print(f"  - {size / 1e6:8.2f} MB  {path}")
        print("\nIf this is an artifact (golden, checkpoint, reference tensor):")
        print("  keep it outside the repository and use the runtime downloads.")
        print("If it belongs in the tree, add it to ALLOWED in this")
        print("script WITH the reason — every entry is permanent clone weight.")
        return 1
    print(f"PASS: {len(entries)} tracked files, {total / 1e6:.1f} MB, "
          f"none above {MAX_TRACKED_BYTES / 1e6:.1f} MB "
          f"({len(ALLOWED)} allowed exception(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
