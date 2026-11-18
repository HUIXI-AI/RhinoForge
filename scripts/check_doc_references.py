#!/usr/bin/env python3
"""Verify that intra-repo doc cross-references resolve to real files and anchors.

Scope = the operating-manual docs an agent/newcomer actually follows:
AGENTS.md, examples/, model performance, the top-level docs/*.md (+ .zh.md mirrors) and docs/guidelines/, and
the in-repo skill(s) under .agents/skills/. Files move often in this repo, and a
markdown link or an `@import` that points at a renamed/deleted path silently rots
— this gate fails when that happens.

What is checked (deliberately conservative to keep false-positives ~0):
  * markdown inline links `[text](target)` — target resolved relative to the
    containing file's directory (standard markdown), `#anchor`/`?query` stripped.
  * `@import` lines (`@path/to/file.md` at line start) — AGENTS.md's import
    mechanism, resolved relative to the repo root.
  * the `#anchor` of a link into a markdown file (including same-file `#anchor`
    links): it must be an explicit `<a id=…>`/`<a name=…>` or a heading slug.
    Both the GitLab slug (repeated hyphens squeezed) and the GitHub slug are
    accepted, each with the `-1`, `-2` suffixes of repeated headings.

What is intentionally NOT checked (would false-positive):
  * external URLs (http/https/mailto/tel/protocol-relative).
  * workspace/board artifact paths outside the repository.
  * templated/glob paths — anything containing < > * { } $ | backtick or a space
    (e.g. `adapters/<model>/`, `perf_results/<target>/<date>/`).
  * anchors into non-markdown files (e.g. `file.py#L10`), and anchors of links
    in docs/model_performance/history/ (unchanged 2026-10-21 archives whose
    targets were regenerated since).
  * knowledge/ `[[links]]` (dangling-to-unwritten is allowed by that layer's
    SCHEMA) and knowledge/ `sources:` (many legit external refs: memory/, repos).

Direct invocation:
    python scripts/check_doc_references.py
    python scripts/check_doc_references.py --self-test

Exit 0: every checked reference resolves. Exit 1: broken refs (file:line → target).
Static + stdlib-only + hardware-free (mirrors the other scripts/check_*.py gates).
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import urllib.parse


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Explicit scan set — the operating-manual surface, NOT the append-only history
# under docs/roadmap, docs/superpowers, docs/kernels (full of intentionally
# time-stamped/stale refs). Keep this list tight and load-bearing.
SCAN_GLOBS = (
    "README.md",
    "README_EN.md",
    "AGENTS.md",
    "docs/*.md",
    "docs/guidelines/*.md",
    "examples/**/*.md",
    ".agents/skills/**/*.md",
    "knowledge/INDEX.md",
)

ARCHIVED_PERFORMANCE_REPORTS = "docs/model_performance/history"

_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_AT_IMPORT = re.compile(r"^@(\S+\.md)\s*$")
_SKIP_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://|^(mailto|tel):")
# Template / glob / shell-var markers — a target carrying any of these is not a
# literal path.
_TEMPLATE_CHARS = set("<>*{}$|` ")
_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")
_EXPLICIT_ID = re.compile(r"""<a\s[^>]*?\b(?:id|name)\s*=\s*["']([^"']+)["']""")
_ANCHOR_CACHE: dict[pathlib.Path, set[str]] = {}


def _iter_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for glob in SCAN_GLOBS:
        files.extend(p for p in REPO_ROOT.glob(glob) if p.is_file())
    return sorted(set(files))


def _anchors(path: pathlib.Path) -> set[str]:
    """Lower-cased anchors a markdown file defines: explicit ids and heading slugs."""
    if path not in _ANCHOR_CACHE:
        anchors: set[str] = set()
        seen: dict[str, int] = {}
        fenced = False
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if _FENCE.match(line):
                fenced = not fenced
                continue
            if fenced:
                continue
            anchors.update(value.lower() for value in _EXPLICIT_ID.findall(line))
            m = _HEADING.match(line)
            if not m:
                continue
            text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(2))   # link -> its text
            text = re.sub(r"<[^>]+>", "", text).strip().lower()
            github = re.sub(r"[^\w\- ]", "", text).replace(" ", "-")
            for slug in {github, re.sub(r"-+", "-", github)}:
                count = seen.get(slug, 0)
                seen[slug] = count + 1
                anchors.add(slug if count == 0 else f"{slug}-{count}")
        _ANCHOR_CACHE[path] = anchors
    return _ANCHOR_CACHE[path]


def _anchor_of(raw: str) -> str | None:
    """The decoded `#anchor` of a link target, or None when it has none."""
    t = raw.strip()
    if "#" not in t or t.startswith("//") or _SKIP_SCHEME.match(t):
        return None
    anchor = urllib.parse.unquote(t.split("#", 1)[1].split("?", 1)[0].strip())
    return anchor or None


def _clean_target(raw: str) -> str | None:
    """Return a repo file path to verify, or None to skip this reference."""
    t = raw.strip()
    if not t or t.startswith("#"):            # empty or same-file anchor
        return None
    if t.startswith("//") or _SKIP_SCHEME.match(t):   # URL / protocol-relative
        return None
    t = t.split("#", 1)[0].split("?", 1)[0].strip()   # drop anchor / query
    if not t:
        return None
    if any(c in _TEMPLATE_CHARS for c in t):  # templated / glob / var
        return None
    # Vendored submodule (third_party/): checkout-state varies (populated in CI,
    # often not in a worktree) and its docs are the submodule's own concern, not
    # our doc surface — don't gate on it.
    if "third_party" in pathlib.PurePosixPath(t).parts:
        return None
    return t


def _broken_refs(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(lineno, target)] for references in `path` that do not resolve."""
    broken: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    base = path.parent
    # docs/model_performance/history/** are unchanged archives of the reports
    # replaced on 2026-10-21; their relative links were written for the same
    # path under docs/model_performance/ and are checked against it. The
    # archive's own README.md is a new index and resolves normally.
    archive = (REPO_ROOT / ARCHIVED_PERFORMANCE_REPORTS).resolve()
    archived = path.resolve().is_relative_to(archive) and path.resolve() != archive / "README.md"
    if archived:
        base = archive.parent / path.resolve().parent.relative_to(archive)
    for lineno, line in enumerate(text.splitlines(), 1):
        targets = list(_MD_LINK.findall(line))
        m = _AT_IMPORT.match(line.strip())
        # @import resolves relative to the repo root, not the containing file.
        at_targets = [(REPO_ROOT, m.group(1))] if m else []
        for raw in targets:
            anchor = None if archived else _anchor_of(raw)
            if raw.strip().startswith("#"):
                if anchor and anchor.lower() not in _anchors(path):
                    broken.append((lineno, raw.strip()))
                continue
            t = _clean_target(raw)
            if t is None:
                continue
            resolved = (base / t).resolve()
            if not resolved.is_relative_to(REPO_ROOT.resolve()):
                continue  # External workspace/board provenance, not an intra-repo link.
            if not resolved.exists():
                broken.append((lineno, raw.strip()))
            elif anchor and resolved.suffix == ".md" and anchor.lower() not in _anchors(resolved):
                broken.append((lineno, raw.strip()))
        for root, raw in at_targets:
            resolved = (root / raw).resolve()
            if not resolved.exists():
                broken.append((lineno, "@" + raw))
    return broken


def _run(files: list[pathlib.Path]) -> int:
    total = 0
    for path in files:
        for lineno, target in _broken_refs(path):
            rel = path.relative_to(REPO_ROOT)
            print(f"FAIL: {rel}:{lineno} -> broken reference {target!r}",
                  file=sys.stderr)
            total += 1
    if total:
        print(f"FAIL: {total} broken doc reference(s)", file=sys.stderr)
        return 1
    print(f"PASS: all doc references resolve ({len(files)} files scanned)")
    return 0


def _self_test() -> int:
    import tempfile
    global REPO_ROOT
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "real.md").write_text("# Sec\n## A / B\n## Sec\n<a id=\"explicit\"></a>\n```\n# fenced\n```\n")
    (d / "code.py").write_text("x = 1\n")
    (d / "sub").mkdir()
    (d / "sub" / "child.md").write_text("hi\n")
    doc = d / "doc.md"
    doc.write_text(
        "## Section\n"
        "[ok](real.md)\n"                       # resolves
        "[ok-anchor](real.md#sec)\n"            # resolves (heading slug)
        "[ok-repeat](real.md#sec-1)\n"          # resolves (second identical heading)
        "[ok-github](real.md#a--b)\n"           # resolves (GitHub slug)
        "[ok-gitlab](real.md#a-b)\n"            # resolves (GitLab slug, hyphens squeezed)
        "[ok-id](real.md#explicit)\n"           # resolves (explicit <a id>)
        "[ok-code](code.py#L1)\n"               # resolves (anchor into non-markdown not checked)
        "[ok-local](#section)\n"                # resolves (same-file heading)
        "[bad-anchor](real.md#fenced)\n"        # BROKEN (heading inside a code fence)
        "[bad-local](#nowhere)\n"               # BROKEN
        "[ok-rel](sub/child.md)\n"              # resolves
        "[url](https://example.com/x.md)\n"     # skipped (URL)
        "[tmpl](adapters/<model>/x.md)\n"       # skipped (template)
        "[glob](perf_results/*/r.md)\n"         # skipped (glob)
        "[submod](../third_party/foo/bar.md)\n" # skipped (vendored submodule)
        "[bad](does_not_exist.md)\n"            # BROKEN
        "[bad2](sub/missing.md)\n"              # BROKEN
        "[external](../external-workspace/missing.md)\n"  # outside repo scope
    )
    saved = REPO_ROOT
    REPO_ROOT = d
    try:
        broken = {t for _, t in _broken_refs(doc)}
        nested = d / "examples/configs/model/README.md"
        nested.parent.mkdir(parents=True)
        nested.write_text("[bad](missing.toml)\n")
        performance = d / "docs/example.md"
        performance.parent.mkdir(parents=True)
        performance.write_text("[bad](missing.json)\n")
        nested_ok = {nested, performance}.issubset(_iter_files())
        nested_ok = nested_ok and _broken_refs(nested) == [(1, "missing.toml")]
    finally:
        REPO_ROOT = saved
    expected = {"does_not_exist.md", "sub/missing.md", "real.md#fenced", "#nowhere"}
    ok = broken == expected
    print(("PASS" if ok else "FAIL")
          + f": self-test broken={sorted(broken)} expected={sorted(expected)}")
    # also exercise @import resolution relative to repo-root-like base
    imp = d / "imp.md"
    imp.write_text("@real.md\n@nope.md\n")
    # temporarily point resolution root at d by monkeypatching module global
    saved = REPO_ROOT
    REPO_ROOT = d
    try:
        imp_broken = {t for _, t in _broken_refs(imp)}
    finally:
        REPO_ROOT = saved
    ok2 = imp_broken == {"@nope.md"}
    print(("PASS" if ok2 else "FAIL")
          + f": self-test @import broken={sorted(imp_broken)} expected=['@nope.md']")
    print(("PASS" if nested_ok else "FAIL") + ": self-test nested example/performance discovery")
    return 0 if (ok and ok2 and nested_ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    return _run(_iter_files())


if __name__ == "__main__":
    sys.exit(main())
