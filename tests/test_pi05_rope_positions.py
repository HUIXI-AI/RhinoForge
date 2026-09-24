"""Pi0.5 RoPE positions: the prefix and the suffix must both be told where they are.

`position` in a fused pi05 subsystem is one number doing two jobs — the RoPE
table index and the KV cache write row. They are equal only while the LOGICAL
position equals the PHYSICAL row, which any padded or holed prefix breaks. Three
sites had it; all three are fixed (2026-08-12):

  1. VLM prefill  — adapters/pi05/gemma.py sends a position_ids-gathered
                    [seq, head_dim/2] RoPE table (the prefix's positions are not
                    contiguous when a modality is masked out, so it needs a table).
  2. action expert — adapters/pi05/adarms.py + runtime.py send a scalar logical
                    start (suffix_pad_masks is all ones, so its positions ARE
                    contiguous and a start offset is enough).
  3. per-chunk masks — rpu_sdpa.cpp keys its stable DDR slot on shape AND an
                    ordinal, so two equal-length chunks of one prefill stop
                    sharing a buffer.

This file gates (1) and (2), the two whose plumbing lives in Python: they must
be sent on EVERY forward, because which table/offset the kernel reads is decided
at graph BUILD and a handle that only sometimes sets it would replay one
forward's positions against another's input. (3) is C++-only and is covered by
the device legs.

Host-only: reads the adapters as source, loads no extension.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_PI05 = _ROOT / "python" / "rpu_backend" / "adapters" / "pi05"


def _fn(module: str, name: str):
    src = (_PI05 / module).read_text()
    tree = ast.parse(src, filename=module)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node, src
    raise AssertionError(f"{name} not found in {module}")


def _calls(fn_node, dotted: str):
    """Every call to `torch.ops.rpu.<dotted>` inside fn_node."""
    out = []
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == dotted:
            out.append(n)
    return out


def _guarding_ifs(fn_node, call):
    """The `if` statements that `call` sits inside, within fn_node."""
    found = []

    def walk(node, stack):
        for child in ast.iter_child_nodes(node):
            if child is call:
                found.extend(stack)
            walk(child, stack + [node] if isinstance(node, ast.If) else stack)

    walk(fn_node, [])
    return found


@pytest.mark.parametrize("module, fn, op", [
    ("gemma.py", "_run_gemma_prefill", "gemma_set_prefill_rope"),
    ("adarms.py", "rpu_adarms_model_forward", "adarms_set_rope_position"),
])
def test_positions_are_sent_unconditionally(module, fn, op):
    """Deleting the call, or hiding it behind `if position_ids is not None`,
    must turn this red — both would leave a graph BUILD keyed on one forward's
    positions being replayed against another's."""
    node, _ = _fn(module, fn)
    calls = _calls(node, op)
    assert len(calls) == 1, f"{fn} must call {op} exactly once, found {len(calls)}"
    assert not _guarding_ifs(node, calls[0]), (
        f"{op} is inside an `if` — it must run on every forward, because the "
        f"kernel's choice of table/offset is baked at graph BUILD")


def test_the_expert_start_is_in_the_graph_signature():
    """The suffix RoPE start is a scalar baked into the kernel at BUILD, so it
    has to be part of the signature or a replay applies the wrong one."""
    src = (_PI05 / "runtime.py").read_text()
    tree = ast.parse(src)
    helper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "_set_denoise_rope_position")
    assert any(isinstance(n, ast.Return) for n in ast.walk(helper)), \
        "_set_denoise_rope_position must RETURN the start so callers can key on it"
    # Both denoise paths must feed the returned value into their GraphSignature.
    for fn in ("_run_denoise_fused", "_run_denoise_loop"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == fn)
        text = ast.unparse(node)
        assert "_set_denoise_rope_position(" in text, f"{fn} never sets it"
        sig = text[text.index("GraphSignature("):]
        assert "rope_pos" in sig[:sig.index(")\n") if ")\n" in sig else len(sig)], \
            f"{fn}: rope_pos is not in the GraphSignature"


def test_logical_and_physical_only_agree_without_pad_rows():
    """The premise, in three lines: this is why one number cannot do both jobs."""
    pad = torch.tensor([[1] * 256 + [0] * 512 + [1] * 32])   # one camera of three
    physical = pad.shape[1]                                   # KV write row
    logical = int(pad.sum())                                  # RoPE position
    assert physical == 800 and logical == 288
    assert physical != logical, "a holed prefix must separate the two"
    # ...and with every row real they coincide, which is why production never saw it.
    full = torch.ones(1, 800, dtype=torch.long)
    assert full.shape[1] == int(full.sum())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
