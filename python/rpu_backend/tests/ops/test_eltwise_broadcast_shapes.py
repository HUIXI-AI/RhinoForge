"""Eager `aten::{mul,add,sub,div}` broadcast parity vs CPU.

Found 2026-08-09 while judging the red script-style ops files: on the dev host EVERY
fp16 broadcast through `aten::mul` was wrong, and only equal shapes were right
(`[1,128,1024]*[1,1,1024]` cos 0.706753, `*[1024]` cos 0.705944,
`*[1,128,1]` cos 0.776297).

Root cause, fixed 2026-08-09 in `binary_op_impl` (`src/ops/rpu_eltwise_binary.cpp`):
the wrapper's broadcast fast path hands the DDR "batch" kernels a per-block
address stride in reg10/11. That register is in 256-byte (v128) units and the
KERNEL multiplies it by C, so the launcher must supply
`rows_per_block * 2 / 256` EXACTLY — an integer only when `rows_per_block` is a
multiple of 128. It was formed as `(rows_per_block * 2) >> 8`, which truncates,
and at `rows_per_block <= 127` truncates to ZERO: every block after the first
recomputed block 0's rows. The error therefore tracked stride exactness, not the
pattern or the op — `[1,512,256]` (rows/blk 256, exact) was already cos 0.999995
while `[1,704,256]` (rows/blk 352) was 0.928764. The same wrapper also stepped C
in whole v16 vectors, dropping the tail of a C that is not 16-aligned.

The fix demotes what the kernel cannot address back to the sameshape path, which
materialises the broadcast and is correct for any shape; the fast path is kept
where the stride is exact. All four ops route through `binary_op_impl`, so this
covers add/sub/div as well as the reported mul.

This file is the regression gate: the shapes below are the measured failures,
plus the exact-stride and equal-shape controls that bound the claim.

Second half (2026-08-10) makes the EAGER coverage complete rather than
anecdotal. `binary_op_impl` is reachable only from `aten::{add,sub,mul,div}.Tensor` and
their four in-place twins (`src/core/rpu_dispatch_registrations.inc:809, 810,
823, 824, 877, 878, 940, 942`) — no fused C++ subsystem reaches it, and the two
DDR broadcast launchers it guards have exactly one caller each, the wrapper
itself. So "the reachable space" is finite and enumerable:

  pattern       a is...   in-place?   demotion boundary reached by
  1xC_NxC       [1,C]     rejected    C=1024 n>64 / C=256 n>512
  NxC_1xC       [N,C]     allowed     same predicate, operands swapped
  Nx1_NxC       [N,1]     rejected    C=1024 n>65 / C=512 n>256
  NxC_Nx1       [N,C]     allowed     same predicate, operands swapped
  case 5        mid-axis broadcast, folded into one of the two above
  scalar        b.numel()==1 — never reaches pattern matching at all
  NONE          no expressible pattern (e.g. [B,1,N] op [B,N,1])

Every one of those, in both operand orders (the order decides `is_bopa`, which
is what keeps `sub`/`div` from computing `b - a`), on both sides of the
launchable boundary, for all four ops. Before this, EVERY case in the file put
the big tensor first, so `PATTERN_1xC_NxC` and `PATTERN_Nx1_NxC` — half the
matcher — had never run, and `is_bopa=true` had never reached the fast path at
all (the only non-commutative cases used shapes that get demoted).

What the file can and cannot prove about the demotion: a shape that IS demoted
is self-proving, because taking the fast path anyway is numerically wrong by
the measured amounts above, and because both launchers `TORCH_CHECK` the same
predicate — so a demoted pattern that reached one would raise, not quietly pass.
A shape that is NOT demoted has no such witness: over-demoting costs a copy, not
an answer, so "the fast path was taken" is not observable from Python. The
launchable rows below therefore gate the fast path's CORRECTNESS (including
operand order), not its selection.
"""
import pytest
import torch
import rpu_backend

def _check(shape_a, shape_b, op=torch.mul):
    torch.manual_seed(0)
    a = torch.randn(*shape_a, dtype=torch.float16)
    b = torch.randn(*shape_b, dtype=torch.float16).abs() + 0.5
    ref = op(a, b)
    got = op(a.to('rpu'), b.to('rpu')).cpu()
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert got.numel() and torch.isfinite(got).all() and torch.isfinite(ref).all()
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0).item()
    assert cos > 0.999, f'fp16 {tuple(shape_a)} {op.__name__} {tuple(shape_b)}: cos={cos:.6f} max_abs={(ref.float() - got.float()).abs().max().item():.6f}'
    torch.testing.assert_close(got, ref, rtol=0.001, atol=0.001)

def test_no_broadcast_is_correct():
    _check((1, 128, 1024), (1, 128, 1024))

def test_middle_axis_broadcast():
    _check((1, 128, 1024), (1, 1, 1024))

def test_scm_independent_u16_tail():
    _check((1, 33, 64), (1, 1, 64))

def test_rank_mismatch_broadcast():
    _check((1, 128, 1024), (1024,))

def test_trailing_axis_broadcast():
    _check((1, 128, 1024), (1, 128, 1))

@pytest.mark.parametrize('n', [256, 1024])
def test_many_blocks(n):
    _check((1, n, 1024), (1, 1, 1024))
    _check((1, n, 1024), (1, n, 1))

def test_partial_overlap_stride():
    _check((1, 704, 256), (1, 1, 256))

@pytest.mark.parametrize('n', [352, 512, 2048])
def test_shapes_that_were_already_right_stay_right(n):
    _check((1, n, 256), (1, 1, 256))
    _check((1, n, 16), (1, 1, 16))

@pytest.mark.parametrize('c', [24, 100])
def test_c_not_v16_aligned(c):
    _check((1, 8, c), (1, 1, c))
    _check((1, 8, c), (1, 8, 1))

@pytest.mark.parametrize('c', [3072, 4304, 6144])
def test_c_exhausts_the_vlm_budget(c):
    """A large C makes the geometry's own DIVISOR go to zero and then negative.

    VLM is a fixed 384 entries and each kernel reserves a share of them per v16
    column vector, so the divisor shrinks as C grows and hits zero at a finite
    C: 3072 for `1xC_NxC` ((384 - 2*ceil(C/16)) * 256), 6144 for `Nx1_NxC`
    (384 / (C/16 + 1)). Integer division by zero does not trap on this aarch64
    host — SDIV yields 0 — so the result was not a crash but a `blk_cnt` of 0
    or, past the zero point, a negative one that casts to a huge `size_t`.
    Either way `rows_per_block` truncated to 0, and `0 % 256 == 0` made
    `finish_geom` report the stride EXACT: the predicate whose entire job is to
    say "demote this" certified it instead. Measured on the dev host before the guard:

      1xC_NxC c=3072   blk_cnt 0                      rows/blk 0    launchable TRUE
      1xC_NxC c=4304   blk_cnt 18446744073709551607   rows/blk 0    launchable TRUE
      Nx1_NxC c=6144   blk_cnt 0                      rows/blk 0    launchable TRUE

    Both helpers are exercised per C, so each one crosses its own zero point.
    A pass here also proves the sameshape path was taken: the launchers
    TORCH_CHECK the same predicate, so a pattern that reached them would raise
    rather than quietly return the right answer.
    """
    _check((1, 100, c), (1, 1, c))    # 1xC_NxC
    _check((1, 100, c), (1, 100, 1))  # Nx1_NxC


def test_vision_intermediate_rank_mismatch():
    _check((1, 100, 4304), (4304,))

@pytest.mark.parametrize('op', [torch.mul, torch.add, torch.sub, torch.div])
def test_every_binary_op_shares_the_wrapper(op):
    _check((1, 128, 1024), (1, 1, 1024), op)
    _check((1, 128, 1024), (1, 128, 1), op)
PATTERN_MATRIX = [('1xC_NxC-launch', (1, 1, 256), (1, 512, 256), True), ('1xC_NxC-demote', (1, 1, 1024), (1, 128, 1024), False), ('NxC_1xC-launch', (1, 512, 256), (1, 1, 256), True), ('NxC_1xC-demote', (1, 128, 1024), (1, 1, 1024), False), ('Nx1_NxC-launch', (1, 64, 1), (1, 64, 1024), True), ('Nx1_NxC-demote', (1, 128, 1), (1, 128, 1024), False), ('NxC_Nx1-launch', (1, 64, 1024), (1, 64, 1), True), ('NxC_Nx1-demote', (1, 128, 1024), (1, 128, 1), False)]

@pytest.mark.parametrize('op', [torch.mul, torch.add, torch.sub, torch.div], ids=lambda o: o.__name__)
@pytest.mark.parametrize('label,sa,sb,launchable', PATTERN_MATRIX, ids=[r[0] for r in PATTERN_MATRIX])
def test_pattern_matrix(label, sa, sb, launchable, op):
    _check(sa, sb, op)

def test_mid_axis_broadcast_with_non_unit_prefix(record_property):
    for op in (torch.mul, torch.sub):
        _check((1, 1, 8, 64), (3, 1, 8, 64), op)
        _check((3, 1, 8, 64), (1, 1, 8, 64), op)
        _check((1, 1, 8, 1024), (128, 1, 8, 1024), op)

@pytest.mark.parametrize('op', [torch.mul, torch.div], ids=lambda o: o.__name__)
def test_rank_2_and_rank_4(op):
    _check((512, 256), (1, 256), op)
    _check((128, 1024), (1, 1024), op)
    _check((64, 1024), (64, 1), op)
    _check((1, 8, 64, 64), (1, 1, 1, 64), op)
    _check((2, 4, 16, 128), (2, 4, 16, 1), op)
    _check((2, 4, 16, 1024), (1024,), op)

def test_nx1_c256_special_branch(record_property):
    _check((1, 700, 256), (1, 700, 1))
    _check((1, 700, 1), (1, 700, 256))

def test_scalar_shaped_operand_never_reaches_pattern_matching(record_property):
    for op in (torch.mul, torch.add, torch.sub, torch.div):
        _check((1, 128, 1024), (1, 1, 1), op)

def test_unmatchable_broadcast_materialises():
    for op in (torch.mul, torch.sub):
        _check((1, 8, 1), (1, 1, 8), op)


#   The in-place twins reach the same matcher with `out_opt` set. Only the two
#   "small operand second" patterns are reachable in place; the other two would
#   have to grow `self` and the wrapper rejects them outright.
#
#   ⚠️ These rows caught a LIVE DEFECT on 2026-08-10, FIXED the same day in
#   src/ops/rpu_eltwise_binary.cpp. Keeping the account because the shape of the
#   bug is the point. `binary_op_inplace_wrapper` (rpu_eltwise_binary.cpp:525-532) calls
#   `binary_op_impl(..., /*is_commutative=*/true, &self)` — hard-coded, for every
#   op. `is_bopa` is `!is_commutative`, so it is always false in place, and the
#   two "small second" patterns then hand the kernel `small op large`, i.e.
#   `b - a` where `a - b` was asked for. Measured on the dev host, `[1,512,256]` with
#   `[1,1,256]`: `a.sub_(b)` gives cos +0.999992 against `b - a` and −0.999992
#   against `a - b`. Out-of-place `a - b` on the SAME shape is correct (the
#   Tensor entry points pass the real commutativity), and the demoted rows are
#   correct (the sameshape path never consults `is_bopa`) — so it is exactly the
#   in-place fast path. Fix: derive the flag, e.g.
#       const bool is_commutative = (op_type == ValuOpType::ADD ||
#                                    op_type == ValuOpType::MUL);
#   That fix landed; these are ordinary passing rows now, and they go red again
#   if the flag is ever re-hard-coded.
_INPLACE_ROWS = []
for _op in ('mul_', 'add_', 'sub_', 'div_'):
    for (_lbl, _sa, _sb, _launch) in (('NxC_1xC-launch', (1, 512, 256), (1, 1, 256), True), ('NxC_1xC-demote', (1, 128, 1024), (1, 1, 1024), False), ('NxC_Nx1-launch', (1, 64, 1024), (1, 64, 1), True), ('NxC_Nx1-demote', (1, 128, 1024), (1, 128, 1), False)):
        _INPLACE_ROWS.append(pytest.param(_op, _sa, _sb, _launch, id=f'{_op}-{_lbl}'))

@pytest.mark.parametrize('op,sa,sb,launchable', _INPLACE_ROWS)
def test_inplace_broadcast(op, sa, sb, launchable):
    torch.manual_seed(0)
    a = torch.randn(*sa, dtype=torch.float16).abs() + 0.5
    b = torch.randn(*sb, dtype=torch.float16).abs() + 0.5
    ref = getattr(a.clone(), op)(b)
    got = getattr(a.clone().to('rpu'), op)(b.to('rpu')).cpu()
    assert got.shape == ref.shape
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), got.float().flatten(), dim=0).item()
    assert cos > 0.999, (
        f"fp16 {tuple(sa)}.{op}({tuple(sb)}) launchable={launchable}: "
        f"cos={cos:.6f} "
        f"max_abs={(ref.float() - got.float()).abs().max().item():.6f}")


#   `alpha` multiplies the SECOND aten operand, and the DDR launchers bind it to
#   `scale_b` (rpu_eltwise_binary.cpp:1289 / 1164), which the kernel applies to
#   whichever operand it evaluates second — `input_b` normally, `input_a` under
#   BOPA. So alpha is correct exactly when the aten `b` lands in that slot.
#
#   ⚠️ Two rows are a LIVE DEFECT (same 2026-08-10 sweep, also NOT fixed here).
#   For the "small operand second" patterns the wrapper sets
#   `is_bopa = !is_commutative`, i.e. it swaps the operands for `add`/`mul`
#   because `a + b == b + a`. That identity does not survive alpha:
#   `a + α·b != b + α·a`. Measured on the dev host, `torch.add(a, b, alpha=2.5)` with
#   a=[1,512,256], b=[1,1,256] gives cos +0.999988 against `b + 2.5·a` and
#   +0.713017 against the correct `a + 2.5·b`; `[1,64,1024]` with `[1,64,1]`
#   behaves identically. `sub` is right on all four patterns (it never takes the
#   swap), and the sameshape/demoted path is right for both ops.
#   Fix: stop treating a commutative op as swappable when alpha != 1 —
#       const bool alpha_is_one = (alpha.to<double>() == 1.0);
#       ... is_bopa = !(is_commutative && alpha_is_one);
#   in the PATTERN_NxC_1xC and PATTERN_NxC_Nx1 arms. `sub`'s passing rows are the
#   evidence that BOPA puts alpha on the right operand.
_ALPHA_ROWS = []
for (_opname, _lbl, _sa, _sb, _bad) in (('add', '1xC_NxC-launch', (1, 1, 256), (1, 512, 256), False), ('add', 'NxC_1xC-launch', (1, 512, 256), (1, 1, 256), True), ('add', 'Nx1_NxC-launch', (1, 64, 1), (1, 64, 1024), False), ('add', 'NxC_Nx1-launch', (1, 64, 1024), (1, 64, 1), True), ('add', 'NxC_1xC-demote', (1, 128, 1024), (1, 1, 1024), False), ('sub', '1xC_NxC-launch', (1, 1, 256), (1, 512, 256), False), ('sub', 'NxC_1xC-launch', (1, 512, 256), (1, 1, 256), False), ('sub', 'Nx1_NxC-launch', (1, 64, 1), (1, 64, 1024), False), ('sub', 'NxC_Nx1-launch', (1, 64, 1024), (1, 64, 1), False), ('sub', 'NxC_1xC-demote', (1, 128, 1024), (1, 1, 1024), False)):
    _ALPHA_ROWS.append(pytest.param(_opname, _sa, _sb, id=f'{_opname}-{_lbl}'))

@pytest.mark.parametrize('opname,sa,sb', _ALPHA_ROWS)
def test_alpha_reaches_both_kernels(opname, sa, sb):
    op = torch.add if opname == 'add' else torch.sub
    torch.manual_seed(0)
    a = torch.randn(*sa, dtype=torch.float16)
    b = torch.randn(*sb, dtype=torch.float16).abs() + 0.5
    ref = op(a, b, alpha=2.5)
    got = op(a.to('rpu'), b.to('rpu'), alpha=2.5).cpu()
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0).item()
    assert cos > 0.999, f'{opname}(alpha=2.5) {tuple(sa)} {tuple(sb)}: cos={cos:.6f} max_abs={(ref.float() - got.float()).abs().max().item():.6f}'
