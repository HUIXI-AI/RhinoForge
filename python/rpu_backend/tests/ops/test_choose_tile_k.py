import pytest
from rpu_backend._cpp_ext import sdpa_helpers as h

def test_default_mode_is_conservative():
    assert h.kSdpaTilePolicyMode == 1, 'default must be conservative (mode=1); aggressive is opt-in'

def test_policy_version_encoding():
    # base * 2 + mode_bit; base=3 includes the sync-compatible LTM fallback.
    assert h.kSdpaTilePolicyBase == 3
    assert h.kSdpaTilePolicyVersion == h.kSdpaTilePolicyBase * 2 + h.kSdpaTilePolicyMode

def test_max_vlm_size_matches_header():
    assert h.MAX_VLM_SIZE == 400

def test_dispatcher_matches_conservative_in_default_mode():
    samples = [(1, 8, 0, 8), (8, 8, 1, 8), (4, 16, 1, 16), (1, 36, 0, 36), (8, 16, 1, 16), (3, 8, 1, 8)]
    for (tm, tn, mask, hd) in samples:
        assert h.choose_tile_k(tm, tn, mask, hd) == h.choose_tile_k_conservative(tm, tn, mask, hd), f'dispatcher disagrees with conservative at {(tm, tn, mask, hd)}'
R11_AGGRESSIVE = [((1, 8, 0, 8), 16), ((8, 8, 1, 8), 16), ((1, 16, 0, 16), 16), ((8, 16, 1, 16), None), ((4, 16, 1, 16), 12), ((1, 36, 0, 36), 8), ((8, 18, 1, 18), None), ((4, 18, 1, 18), 12), ((4, 36, 1, 36), None), ((2, 36, 1, 36), 6)]

@pytest.mark.parametrize('inp,expected', R11_AGGRESSIVE)
def test_choose_tile_k_aggressive_r11_table(inp, expected):
    got = h.choose_tile_k_aggressive(*inp)
    assert got == expected, f'aggressive({inp}) = {got}, want {expected}'
R11_CONSERVATIVE = [((1, 8, 0, 8), 16), ((8, 8, 1, 8), 16), ((1, 16, 0, 16), 16), ((8, 16, 1, 16), None), ((4, 16, 1, 16), 8), ((1, 36, 0, 36), 8), ((8, 18, 1, 18), None), ((4, 18, 1, 18), 8), ((4, 36, 1, 36), None), ((2, 36, 1, 36), 4)]

@pytest.mark.parametrize('inp,expected', R11_CONSERVATIVE)
def test_choose_tile_k_conservative_r11_inputs(inp, expected):
    got = h.choose_tile_k_conservative(*inp)
    assert got == expected, f'conservative({inp}) = {got}, want {expected}'
R13_DIVERGENCE = [((1, 8, 0, 8), 16, 16, 'T1 — match'), ((8, 8, 1, 8), 16, 16, 'T1 — match'), ((3, 8, 1, 8), 15, 3, 'T2 fallback (no Tier1 LTM divisor)'), ((6, 8, 1, 8), 12, 6, 'T2 fallback'), ((7, 8, 1, 8), 14, 7, 'T2 fallback'), ((9, 8, 1, 8), 9, 9, 'T2 fallback (numeric coincidence)'), ((10, 8, 1, 8), 10, 10, 'T2 fallback'), ((11, 8, 1, 8), 11, 11, 'T1 hit (11 in candidate set)')]

@pytest.mark.parametrize('inp,agg_expected,cons_expected,_note', R13_DIVERGENCE)
def test_r13_divergence(inp, agg_expected, cons_expected, _note):
    assert h.choose_tile_k_aggressive(*inp) == agg_expected, f'aggressive({inp}) != {agg_expected}'
    assert h.choose_tile_k_conservative(*inp) == cons_expected, f'conservative({inp}) != {cons_expected}'
VLM_GOLDENS = [((8, 16, 8, 16, 1), 424, False), ((4, 16, 12, 16, 1), 344, True), ((4, 16, 16, 16, 1), 408, False), ((1, 36, 8, 36, 0), 372, True), ((4, 18, 12, 18, 1), 384, True), ((2, 36, 6, 36, 1), 376, True)]

@pytest.mark.parametrize('args,expected_vlm,passes', VLM_GOLDENS)
def test_vlm_need_hand_computed(args, expected_vlm, passes):
    (tm, tn, tk, hd, mask) = args
    got = h.vlm_need(tm, tn, tk, hd, hd, hd, mask)
    assert got == expected_vlm, f'vlm_need(tm={tm},tn={tn},tk={tk},hd={hd},mask={mask}) = {got}, want {expected_vlm}'
    assert (got <= h.MAX_VLM_SIZE) == passes, f'VLM={got} vs MAX={h.MAX_VLM_SIZE} budget check failed for {args}'
SAMPLE_INPUTS = [(tm, tn, mask, hd) for tm in (1, 2, 4, 8) for tn in (8, 16, 18, 36) for mask in (0, 1) for hd in (8, 16, 18, 36)]

def _check_output_invariants(out, tm, tn, mask, hd):
    if out is None:
        return
    assert 1 <= out <= 16, f'tk={out} out of [1,16]'
    assert tm * tn * out <= 1024, f'tm*tn*tk = {tm * tn * out} > 1024'
    if mask == 1:
        assert out % tm == 0, f'LTM: tk={out} % tm={tm} != 0'
    vlm = h.vlm_need(tm, tn, out, hd, hd, hd, mask)
    assert vlm <= h.MAX_VLM_SIZE, f'VLM={vlm} > {h.MAX_VLM_SIZE}'

@pytest.mark.parametrize('inp', SAMPLE_INPUTS)
def test_aggressive_outputs_satisfy_invariants(inp):
    out = h.choose_tile_k_aggressive(*inp)
    _check_output_invariants(out, *inp)

@pytest.mark.parametrize('inp', SAMPLE_INPUTS)
def test_conservative_outputs_satisfy_invariants(inp):
    out = h.choose_tile_k_conservative(*inp)
    _check_output_invariants(out, *inp)

def test_conservative_tier1_or_tier2_only():
    tier1 = {16, 11, 8, 5, 4, 2, 1}
    for (tm, tn, mask, hd) in SAMPLE_INPUTS:
        out = h.choose_tile_k_conservative(tm, tn, mask, hd)
        if out is None:
            continue
        assert out in tier1 or out == tm, f'conservative({tm},{tn},{mask},{hd}) = {out} ∉ Tier1 ∪ {{tm}}'

@pytest.mark.parametrize('inp', SAMPLE_INPUTS)
def test_aggressive_ge_conservative_when_both_defined(inp):
    agg = h.choose_tile_k_aggressive(*inp)
    cons = h.choose_tile_k_conservative(*inp)
    if agg is None or cons is None:
        return
    assert agg >= cons, f'aggressive={agg} < conservative={cons} at {inp} (impossible)'

@pytest.mark.parametrize('bad_args', [(0, 8, 0, 8), (8, 0, 0, 8), (8, 8, 0, 0)])
def test_precondition_zero_inputs_throw(bad_args):
    with pytest.raises(RuntimeError, match='must be ≥ 1'):
        h.choose_tile_k_aggressive(*bad_args)
    with pytest.raises(RuntimeError, match='must be ≥ 1'):
        h.choose_tile_k_conservative(*bad_args)

@pytest.mark.parametrize('head_dim,seq_q,mask', [(128, 1, 0), (128, 16, 0), (128, 128, 1), (128, 512, 1), (128, 4096, 1), (256, 1, 0), (256, 512, 1), (288, 512, 1), (576, 1, 0), (576, 512, 1)])
def test_try_matches_throw_on_feasible(head_dim, seq_q, mask):
    t_throw = h.compute_tiling(head_dim=head_dim, seq_q=seq_q, attn_mask_type=mask)
    t_try = h.try_compute_tiling(head_dim=head_dim, seq_q=seq_q, attn_mask_type=mask)
    assert t_try is not None, f'try returned None on feasible {(head_dim, seq_q, mask)}'
    assert t_try == t_throw, f'try/throw diverge: {t_try} vs {t_throw}'

@pytest.mark.parametrize('head_dim,seq_q,mask', [(0, 128, 0), (-128, 128, 0), (100, 128, 0), (200, 128, 0), (128, 0, 0), (128, -1, 0), (8192, 64, 0), (4096, 64, 0)])
def test_try_returns_none_on_infeasible(head_dim, seq_q, mask):
    assert h.try_compute_tiling(head_dim=head_dim, seq_q=seq_q, attn_mask_type=mask) is None

@pytest.mark.parametrize('head_dim,seq_q,mask', [(0, 128, 0), (100, 128, 0), (128, 0, 0), (8192, 64, 0)])
def test_throw_wrapper_throws_on_infeasible(head_dim, seq_q, mask):
    with pytest.raises(RuntimeError, match='cannot shrink to fit VLM'):
        h.compute_tiling(head_dim=head_dim, seq_q=seq_q, attn_mask_type=mask)

def test_try_compute_tiling_invariants_over_sample():
    head_dims = [128, 256, 288, 576]
    seq_qs = [1, 16, 64, 128, 256, 512]
    masks = [0, 1]
    for hd in head_dims:
        hd_v16 = (hd + 15) // 16
        for sq in seq_qs:
            for mask in masks:
                t = h.try_compute_tiling(head_dim=hd, seq_q=sq, attn_mask_type=mask)
                if t is None:
                    continue
                (tm, tn, tk) = (t['tile_m_v16'], t['tile_n_v16'], t['tile_k_v16'])
                assert 1 <= tk <= 16, f'tk out of [1,16]: {t} @ hd={hd} sq={sq}'
                assert tm * tn * tk <= 1024, f'C2 violated: {t} @ hd={hd} sq={sq}'
                if mask == 1:
                    assert tk % tm == 0, f'C3 LTM violated: {t} @ hd={hd} sq={sq}'
                vlm = h.vlm_need(tm, tn, tk, hd_v16, hd_v16, hd_v16, mask)
                assert vlm <= h.MAX_VLM_SIZE, f'C4 VLM={vlm} > {h.MAX_VLM_SIZE}: {t} @ hd={hd} sq={sq}'
                assert tn <= hd_v16, f'tn={tn} > hd_v16={hd_v16}: {t}'

@pytest.mark.parametrize('head_dim,seq_q,mask,expected_tm,expected_tn,expected_tk', [(256, 512, 1, 4, 16, 8), (288, 512, 1, 4, 18, 8), (576, 512, 1, 2, 36, 4)])
def test_shrink_path_lands_on_expected_tile(head_dim, seq_q, mask, expected_tm, expected_tn, expected_tk):
    t = h.compute_tiling(head_dim=head_dim, seq_q=seq_q, attn_mask_type=mask)
    assert (t['tile_m_v16'], t['tile_n_v16'], t['tile_k_v16']) == (expected_tm, expected_tn, expected_tk), f'got {t}, want tm={expected_tm} tn={expected_tn} tk={expected_tk}'

def test_shrink_loop_terminates_on_pathological_hd():
    assert h.try_compute_tiling(head_dim=8192, seq_q=64, attn_mask_type=0) is None
    with pytest.raises(RuntimeError, match='cannot shrink to fit VLM'):
        h.compute_tiling(head_dim=8192, seq_q=64, attn_mask_type=0)
