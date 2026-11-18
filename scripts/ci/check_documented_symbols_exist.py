#!/usr/bin/env python3
"""Fail when a doc presents a symbol as LIVE and that symbol does not exist."""
from __future__ import annotations

import argparse
import os
import io
import pathlib
import re
import subprocess
import sys
import tokenize

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# The operating-manual surface — what a porter/newcomer actually follows. Same
# spirit as scripts/check_doc_references.py: NOT docs/roadmap, docs/kernels,
# historical reports, which are dated append-only records.
SCAN_GLOBS = ("README.md", "README_EN.md", "docs/*.md", "examples/configs/**/README*.md")

BUILT_SO = pathlib.Path("build/rpu_backend/rpu_backend.so")
SDK_DIRS = tuple(filter(None, (os.environ.get("RHINO_LAUNCH_INCLUDE_DIR"),)))

_CPP_EXTS = {".cpp", ".h", ".hpp", ".cc", ".c", ".cu", ".inc"}
_CPP_LINE_COMMENT = re.compile(r"//[^\n]*")
_CPP_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# ── the three rules ─────────────────────────────────────────────────────────
# R1: namespaces this repo / the pinned SDK owns outright.
_OWNED = re.compile(
    r"^(?:"
    r"rpu_launch_[A-Za-z0-9_]+"          # C kernel/DMA launcher wrappers
    r"|spm_alloc_[A-Za-z0-9_]+"          # SPM allocator free functions
    r"|SPM_ALLOC\.[A-Za-z0-9_]+"         # the allocator singleton's methods
    r"|RpuKernelGraph::[A-Za-z0-9_]+"    # graph-runtime type
    r"|Queue_t::[A-Za-z0-9_]+"           # the SDK's queue type
    r")$"
)
# R2: `<type> name(<args>);` — a C++ declaration, inside a ```cpp fence.
_PROTOTYPE = re.compile(
    r"^\s*(?:[A-Za-z_][\w:<>,*&\s]*?[\s*&])([A-Za-z_]\w*)\s*\([^;{]*\)\s*"
    r"(?:const\s*)?(?:noexcept\s*)?;\s*$"
)
_CPP_FENCES = {"cpp", "c++", "cc", "c"}
# R3: the line instructs the reader to call it.
_CALL_INSTRUCTION = re.compile(r"\b(?:[Cc]alls?|[Cc]alling|[Ii]nvokes?|[Ii]nvoking)\b|调用")

_CODE_SPAN = re.compile(r"`+([^`\n]+)`+")
_CALL_FORM = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*)\s*\(")
_QUALIFIED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\.)[A-Za-z_][A-Za-z0-9_]*)*$")

# Retraction vocabulary. A symbol named in a CLAIM carrying one of these is
# documented as DEAD — the opposite of the defect this gate exists for. Kept
# tight on purpose: every entry is unambiguous retraction language, not merely a
# negative word. ("removed" alone is NOT here; it is far too common.)
_TOMBSTONE = (
    "⛔",
    "no longer exist",
    "does not exist",
    "do not exist",
    "no such function",
    "no such symbol",
    "doc-only",
    "zero hits",
    "zero symbols",
    "zero matching symbols",
    "has been removed",
    "been removed from the sdk",
    "falsified",
    "已移除",
    "已删除",
    "已废弃",
    "不存在",
)

# <!-- doc-symbol: ignore <name> — <reason> -->
_SUPPRESS = re.compile(
    r"<!--\s*doc-symbol:\s*ignore\s+(?P<name>[A-Za-z_][\w:.]*)\s*(?P<reason>.*?)-->"
)


# ── resolution universe ─────────────────────────────────────────────────────
def _cpp_code_identifiers(root: pathlib.Path) -> set[str]:
    """Identifiers in C/C++ under `root`, with comments stripped.

    Comment-stripping is load-bearing: all five retired DMA wrappers and both
    retired SDK batch calls still appear in `src/` as historical COMMENTS. A
    naive grep would resolve them and the gate would be blind to case 2 and 3.
    """
    out: set[str] = set()
    if not root.exists():
        return out
    for path in root.rglob("*"):
        if path.suffix not in _CPP_EXTS or not path.is_file():
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        text = _CPP_BLOCK_COMMENT.sub(" ", text)
        text = _CPP_LINE_COMMENT.sub(" ", text)
        out.update(_IDENT.findall(text))
    return out


def _py_code_identifiers(root: pathlib.Path) -> set[str]:
    """NAME tokens in Python under `root` — comments and strings dropped.

    `tokenize` is what makes docstrings not count: `launch_prepared_batch` lives
    in a docstring in python/rpu_backend/graph/dynamo_backend.py and must not
    resolve from there.
    """
    out: set[str] = set()
    if not root.exists():
        return out
    for path in root.rglob("*.py"):
        try:
            src = path.read_text(errors="ignore")
        except OSError:
            continue
        try:
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type == tokenize.NAME:
                    out.add(tok.string)
        except (tokenize.TokenError, IndentationError, SyntaxError):
            # Unparseable file: fall back to a line-comment-stripped scan rather
            # than silently contributing nothing to the universe.
            out.update(_IDENT.findall(re.sub(r"#[^\n]*", " ", src)))
    return out


def _so_identifiers(so_path: pathlib.Path) -> set[str]:
    try:
        proc = subprocess.run(
            ["nm", "-C", "-D", str(so_path)], capture_output=True, text=True, check=False
        )
    except OSError:
        return set()
    if proc.returncode != 0:
        return set()
    return set(_IDENT.findall(proc.stdout))


class Universe:
    def __init__(self, repo_root: pathlib.Path):
        self.names: set[str] = set()
        self.names |= _cpp_code_identifiers(repo_root / "src")
        self.names |= _py_code_identifiers(repo_root / "python")
        self.names |= _py_code_identifiers(repo_root / "examples")
        self.sdk_names: set[str] = set()
        for rel in SDK_DIRS:
            self.sdk_names |= _cpp_code_identifiers(repo_root / rel)
        self.sdk_present = bool(self.sdk_names)
        self.names |= self.sdk_names
        so = repo_root / BUILT_SO
        self.so_present = so.is_file()
        self.so_names = _so_identifiers(so) if self.so_present else set()
        self.names |= self.so_names
        self._flat = tuple(self.names)

    def resolves(self, name: str) -> bool:
        """Exact hit, or shorthand: a substring of a real identifier.

        Shorthand tolerance is what lets docs write `all_reduce_sum_residual`
        for `rpu_launch_all_reduce_sum_residual_kernel` without a suppression.
        It costs nothing here: none of the three known defects is a substring of
        any surviving identifier.
        """
        for part in name.replace(".", "::").split("::"):
            if part in self.names:
                continue
            if any(part in known for known in self._flat):
                continue
            return False
        return True


# ── doc scanning ────────────────────────────────────────────────────────────
class Finding:
    __slots__ = ("name", "path", "line", "rule", "text")

    def __init__(self, name: str, path: str, line: int, rule: str, text: str):
        self.name, self.path, self.line, self.rule, self.text = name, path, line, rule, text

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] `{self.name}` — {self.text.strip()[:110]}"


_UNIT_START = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|#{1,6}\s|\|)")


def _claim_spans(lines: list[str]) -> list[int]:
    """Map line index → claim id: the scope a retraction marker covers.

    A claim is a prose paragraph (blank-line separated), a single list item
    (plus its indented continuation lines), a single table row, a heading, or a
    fenced block. Fine granularity is load-bearing in BOTH directions:

      * too coarse and a marker leaks. Measured: with plain blank-line
        paragraphs, re-adding `release_compute()` to README.md's "General
        checklist" was silently forgiven, because a **FALSIFIED** seven bullets
        further down put the whole run of bullets in one paragraph.
      * too fine and the marker is lost: architecture.md's ⚠️ retraction of the
        five DMA wrappers is one prose paragraph whose marker sits on line 1 and
        whose names sit on lines 3-5.
    """
    span = [0] * len(lines)
    cid = 0
    in_fence = False
    prev_blank = True
    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("```"):
            cid += 1
            in_fence = not in_fence
            span[i] = cid
            if not in_fence:
                cid += 1
                prev_blank = True
            continue
        if in_fence:
            span[i] = cid
            continue
        if not stripped:
            prev_blank = True
            span[i] = cid
            continue
        if prev_blank or _UNIT_START.match(raw):
            cid += 1
            prev_blank = False
        span[i] = cid
    return span


def _tombstone_claims(lines: list[str], span: list[int]) -> set[int]:
    flagged: set[int] = set()
    for i, raw in enumerate(lines):
        low = raw.lower()
        if any(marker in low for marker in _TOMBSTONE):
            flagged.add(span[i])
    return flagged


def _suppressions(lines: list[str]) -> tuple[dict[int, set[str]], list[str]]:
    """Line index → suppressed names, applying to that line and the next one."""
    by_line: dict[int, set[str]] = {}
    errors: list[str] = []
    for i, raw in enumerate(lines):
        for m in _SUPPRESS.finditer(raw):
            name = m.group("name")
            reason = m.group("reason").strip(" -—:")
            if not reason:
                errors.append(f"line {i + 1}: `doc-symbol: ignore {name}` carries no reason")
            for target in (i, i + 1):
                by_line.setdefault(target, set()).add(name)
    return by_line, errors


def scan_file(path: pathlib.Path, uni: Universe, rel: str) -> tuple[list[Finding], list[str]]:
    lines = path.read_text(errors="ignore").splitlines()
    span = _claim_spans(lines)
    dead = _tombstone_claims(lines, span)
    suppressed, errors = _suppressions(lines)
    errors = [f"{rel}:{e}" for e in errors]

    findings: list[Finding] = []
    in_fence = False
    fence_lang = ""

    def emit(name: str, idx: int, rule: str) -> None:
        if span[idx] in dead:
            return
        if name in suppressed.get(idx, ()):  # exact documented name
            return
        if name.split("::")[-1] in suppressed.get(idx, ()):
            return
        findings.append(Finding(name, rel, idx + 1, rule, lines[idx]))

    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence, fence_lang = False, ""
            else:
                in_fence, fence_lang = True, stripped[3:].strip().lower()
            continue

        if in_fence:
            # R2 — a C++ prototype written into a ```cpp block. Skipped entirely
            # when the SDK submodule is not checked out: the prototypes docs show
            # are mostly the SDK's own, so without it every one of them would
            # "not resolve" and the gate would fail for the wrong reason.
            if fence_lang in _CPP_FENCES and uni.sdk_present:
                m = _PROTOTYPE.match(raw)
                if m and not uni.resolves(m.group(1)):
                    emit(m.group(1), idx, "R2-cpp-prototype")
            continue

        spans = [m.group(1).strip() for m in _CODE_SPAN.finditer(raw)]

        # R1 — a name in a namespace the repo / pinned SDK owns.
        for body in spans:
            bare = body.split("(", 1)[0].strip().rstrip("()").strip()
            if not _QUALIFIED.match(bare):
                continue
            if bare.startswith("Queue_t::") and not uni.sdk_present:
                continue  # SDK-owned and the SDK is not checked out — unverifiable
            if _OWNED.match(bare) and not uni.resolves(bare):
                emit(bare, idx, "R1-owned-namespace")

        # R3 — the sentence tells the reader to call it.
        if _CALL_INSTRUCTION.search(raw):
            for body in spans:
                m = _CALL_FORM.match(body)
                if m and not uni.resolves(m.group(1)):
                    emit(m.group(1), idx, "R3-call-instruction")

    return findings, errors


def iter_docs(repo_root: pathlib.Path) -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for glob in SCAN_GLOBS:
        files.extend(p for p in repo_root.glob(glob) if p.is_file())
    return sorted(set(files))


def run(repo_root: pathlib.Path) -> int:
    uni = Universe(repo_root)
    docs = iter_docs(repo_root)

    findings: list[Finding] = []
    errors: list[str] = []
    for path in docs:
        f, e = scan_file(path, uni, str(path.relative_to(repo_root)))
        findings.extend(f)
        errors.extend(e)

    sources = "src+python+examples" + ("+sdk" if uni.sdk_present else "") + ("+so" if uni.so_present else "")
    print(f"scanned {len(docs)} doc files; universe = {len(uni.names)} identifiers ({sources})")

    if not uni.so_present:
        print(f"note: {BUILT_SO} absent; checking source declarations")
    if not uni.sdk_present:
        print("note: set RHINO_LAUNCH_INCLUDE_DIR to also check SDK declarations")

    if errors:
        print("\nMALFORMED SUPPRESSIONS (a suppression must carry a reason):")
        for e in errors:
            print(f"  {e}")
    if findings:
        print(f"\nFAIL: {len(findings)} symbol(s) documented as live that do not exist:\n")
        for f in sorted(findings, key=lambda x: (x.path, x.line)):
            print(f"  {f}")
        print("\nEach is either (a) a real doc defect — fix the doc, or (b) a symbol "
              "that moved — update the name, or (c) a deliberate mention of something "
              "dead — say so in the same paragraph (⛔ / \"no longer exists\" / "
              "\"does not exist\"), or suppress inline with\n"
              "      <!-- doc-symbol: ignore <name> — <reason> -->")
        return 1
    if errors:
        return 1
    print("OK: every symbol claimed live by the operating-manual docs resolves.")
    return 0


# ── self-test ───────────────────────────────────────────────────────────────
_SELF_TEST_DOCS = {
    # (filename, body, expected-names-flagged)
    "r1.md": (
        "# t\n\n"
        "Legacy `rpu_launch_totally_made_up_mover` is kept for back-compat.\n",
        {"rpu_launch_totally_made_up_mover"},
    ),
    "r1_table.md": (
        "# t\n\n"
        "| fn | mode |\n|---|---|\n"
        "| `rpu_launch_another_fake_one` | kernel |\n",
        {"rpu_launch_another_fake_one"},
    ),
    "r2.md": (
        "# t\n\n```cpp\n"
        "uint32_t totally_made_up_prepare(const std::vector<uint8_t> &ids);\n"
        "```\n",
        {"totally_made_up_prepare"},
    ),
    "r3.md": (
        "# t\n\n"
        "- Call `spm_alloc_reset_temporary()` + `made_up_release_thing()` at every "
        "subsystem boundary.\n",
        {"made_up_release_thing"},
    ),
    "tombstone.md": (
        "# t\n\n"
        "⛔ The mover `rpu_launch_totally_made_up_mover` **no longer exists**; call\n"
        "`rpu_launch_ddr_broadcast_spm_dma` instead.\n",
        set(),
    ),
    "tombstone_multiline.md": (
        "# t\n\n"
        "⚠️ **These NO LONGER EXIST** (verified): `rpu_launch_fake_a`,\n"
        "`rpu_launch_fake_b`, and `rpu_launch_fake_c`.\n",
        set(),
    ),
    "suppressed.md": (
        "# t\n\n"
        "<!-- doc-symbol: ignore rpu_launch_suppressed_example — self-test fixture -->\n"
        "Legacy `rpu_launch_suppressed_example` is discussed here.\n",
        set(),
    ),
    "suppressed_no_reason.md": (
        "# t\n\n"
        "<!-- doc-symbol: ignore rpu_launch_bad_suppression -->\n"
        "Legacy `rpu_launch_bad_suppression` is discussed here.\n",
        set(),  # flagged as a MALFORMED suppression, not as a finding
    ),
    "prose_not_flagged.md": (
        "# t\n\n"
        "Search 16-aligned lengths from `ceil16(logical_len)`; MLP is "
        "`SiLU(gate) * up -> down`; `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` "
        "registers it. Only use `buf(name)` / `layer_buf(L, name)` for SDPA.\n",
        set(),
    ),
    "real_names_ok.md": (
        "# t\n\n"
        "- Call `spm_alloc_reset_temporary()` at every subsystem boundary.\n"
        "`rpu_launch_ddr_broadcast_spm_dma_mutable` rewrites src via "
        "`Queue_t::update_dma_kernel`.\n"
        "```cpp\nuint32_t build_batch(void);\n```\n",
        set(),
    ),
}


def self_test(repo_root: pathlib.Path) -> int:
    import tempfile

    uni = Universe(repo_root)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        for fname, (body, expected) in _SELF_TEST_DOCS.items():
            path = tmp / fname
            path.write_text(body)
            findings, errors = scan_file(path, uni, fname)
            got = {f.name for f in findings}
            if got != expected:
                ok = False
                print(f"  FAIL {fname}: expected {sorted(expected)}, got {sorted(got)}")
            elif fname == "suppressed_no_reason.md" and not errors:
                ok = False
                print(f"  FAIL {fname}: reasonless suppression was not reported")
            elif fname != "suppressed_no_reason.md" and errors:
                ok = False
                print(f"  FAIL {fname}: unexpected suppression errors {errors}")
            else:
                print(f"  ok   {fname}")

    # The three historical defects must be unresolvable against the real tree —
    # if any of them starts resolving, the universe builder has regressed (most
    # likely: comment-stripping broke, since all of them survive in comments).
    for name in (
        "release_compute",
        "rpu_launch_ddr2spm_multicore",
        "rpu_launch_spm2ddr_scatter",
        "prepare_batch_begin",
        "launch_prepared_batch",
    ):
        if uni.resolves(name):
            ok = False
            print(f"  FAIL universe: retired symbol `{name}` resolves — comment/"
                  f"docstring stripping has regressed")
    # ... and a live one must resolve, or the universe is simply empty.
    for name in ("rpu_launch_memset_spm_multicore", "spm_alloc_reset_temporary"):
        if not uni.resolves(name):
            ok = False
            print(f"  FAIL universe: live symbol `{name}` does not resolve")

    print("SELF-TEST PASS" if ok else "SELF-TEST FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true", help="run the built-in fixtures")
    ap.add_argument("--root", default=str(REPO_ROOT), help="repo root (default: inferred)")
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    if args.self_test:
        return self_test(root)
    return run(root)


if __name__ == "__main__":
    sys.exit(main())
