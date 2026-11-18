"""An SDPA entry point's LAYOUT comes from its registration, never from a shape.

Two eager entry points, two different meanings for the same four dimensions:

    aten::scaled_dot_product_attention          [batch, heads, seq, head_dim]
    rpu::scaled_dot_product_attention_unified   [batch, seq, heads, head_dim]

They differ by a transpose of dims 1 and 2. So an implementation that reads the
wrong one returns a tensor of exactly the RIGHT SHAPE full of the wrong numbers:
no exception, no warning, nothing to notice. That is precisely what happened —
both entry points were registered to one C++ function, which therefore had to
recover the layout from the shapes with

    const bool is_pytorch_layout = (query.size(3) != 128);

head_dim is not a layout. Measured on the dev host (b=1, dims 8 and 64, is_causal,
fp32 CPU reference), `da2af70f`:

    aten F.sdpa, PyTorch layout, hd=128   cos 0.125696   <- the defect
    aten F.sdpa, PyTorch layout, hd= 64   cos 0.999997   <- why nobody saw it
    rpu unified, unified layout, hd=128   cos 0.999999

Fixed by splitting the functions so the layout comes from the registration
point. This file is the gate for that property.

WHAT THIS FILE ASSERTS, and why it is not the same as its sibling.
`test_sdpa_entry_point_layouts.py` drives each entry point at ITS OWN layout and
checks the answer is right. Necessary, but a shape sniff can satisfy it — the
old code passed 3 of those 4 cells. This file asserts the stronger property that
no sniff can satisfy:

  * one input tensor is pushed through BOTH entry points, and each must match
    the CPU reference for ITS contract and NOT match the other's;
  * therefore the two entry points must return DIFFERENT answers from IDENTICAL
    bytes at IDENTICAL shape — which any layout-from-shape implementation is
    incapable of, at any head_dim, by construction.

Both head_dims are parametrised because one alone cannot see this: 128 is the
number the old sniff keyed on, and at 64 the same call path was exact.

WHAT THIS FILE REPLACED (2026-08-09). It used to be four print-only functions
plus two assertions that a freshly-created contiguous tensor has contiguous
strides — `Q.stride() == (num_heads*seq*hd, seq*hd, hd, 1)`, with the right-hand
side computed from the same locals that built Q. True by construction, so the
file could not fail while carrying the name of the axis that produced the defect
above. Nothing verifiable was lost; for the record, what it did:
  - SwizzleLayout / PyTorchLayout: a Python re-derivation of the 7-D swizzled
    K/V address vs row-major strides, PRINTED side by side and never compared to
    the C++ addressing it models. The real gate is test_insert_kcache.py /
    test_vcache_comprehensive.py / test_vcache_padding.py (CPU reference,
    max_diff, already in CI).
  - test_attention_equivalence: F.sdpa vs a manual matmul+softmax, both on CPU
    in torch. It measured PyTorch against itself; no RPU tensor appeared.
  - test_tiling_constraints: printed a VLM/LTM feasibility table. The real gate
    is test_sdpa_constraints.py + test_keeper_total_tile.py, which drive the
    validator and assert it RAISES.

This is CPU work (the eager op computes on CPU since `d1e5d6b4`), so it needs no
device beyond tensor movement and costs nothing in the shared hw_mech process.
"""
import pytest
import torch
import torch.nn.functional as F
import rpu_backend
(BATCH, DIM1, DIM2) = (1, 8, 64)
SEED = 0
SAME = 0.999
DIFFERENT = 0.9

def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

def _qkv(head_dim):
    torch.manual_seed(SEED)
    return tuple((torch.randn(BATCH, DIM1, DIM2, head_dim, dtype=torch.float16) for _ in range(3)))

def _cpu_reference(q, k, v, *, unified):
    if unified:
        out = F.scaled_dot_product_attention(*(t.float().transpose(1, 2).contiguous() for t in (q, k, v)), is_causal=True)
        return out.transpose(1, 2).contiguous().half()
    return F.scaled_dot_product_attention(*(t.float() for t in (q, k, v)), is_causal=True).half()

def _via_aten(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def _via_unified(q, k, v):
    return torch.ops.rpu.scaled_dot_product_attention_unified(q, k, v, None, 0.0, True, None, False)
ENTRY_POINTS = {'aten': (_via_aten, False), 'rpu_unified': (_via_unified, True)}

@pytest.mark.parametrize('head_dim', [64, 128])
@pytest.mark.parametrize('entry', sorted(ENTRY_POINTS))
def test_entry_point_reads_its_registered_layout(entry, head_dim):
    (call, unified) = ENTRY_POINTS[entry]
    (q, k, v) = _qkv(head_dim)
    mine = _cpu_reference(q, k, v, unified=unified)
    theirs = _cpu_reference(q, k, v, unified=not unified)
    assert mine.shape == theirs.shape
    separation = _cos(mine, theirs)
    assert separation < DIFFERENT, f'the two layout readings agree at this shape (cos={separation:.6f}); pick dims where they differ or this test proves nothing'
    got = call(q.to('rpu'), k.to('rpu'), v.to('rpu')).cpu()
    assert got.shape == mine.shape
    (right, wrong) = (_cos(got, mine), _cos(got, theirs))
    own = 'unified' if unified else 'PyTorch'
    other = 'PyTorch' if unified else 'unified'
    assert right > SAME, f'{entry} head_dim={head_dim}: cos={right:.6f} against its own {own}-layout reference (cos={wrong:.6f} against the {other}-layout one) — the entry point is not honouring its registered layout'
    assert wrong < DIFFERENT, f'{entry} head_dim={head_dim}: output matches the {other}-layout reading (cos={wrong:.6f}) — layouts have been confused'

@pytest.mark.parametrize('head_dim', [64, 128])
def test_the_two_entry_points_disagree_on_identical_input(head_dim):
    (q, k, v) = _qkv(head_dim)
    args = (q.to('rpu'), k.to('rpu'), v.to('rpu'))
    from_aten = _via_aten(*args).cpu()
    from_unified = _via_unified(*args).cpu()
    assert from_aten.shape == from_unified.shape
    agreement = _cos(from_aten, from_unified)
    assert agreement < DIFFERENT, f'head_dim={head_dim}: aten and device-unified returned the SAME answer (cos={agreement:.6f}) for identical input. They declare layouts that differ by a dim1<->dim2 transpose, so one of them is now reading the wrong one — most likely the layout is being inferred from the shapes again instead of coming from the registration point'
