import pytest
from rpu_backend._cpp_ext import sdpa_helpers as h

@pytest.mark.parametrize('seq_q,expected', [(1, 1), (16, 1), (17, 2), (32, 2), (64, 4), (128, 8), (144, 9), (160, 10), (176, 11), (177, 8), (256, 8), (320, 8), (512, 8), (1024, 8), (4096, 8)])
def test_tile_m_v16_matches_generator(seq_q, expected):
    assert h.tile_m_v16(seq_q) == expected, f'tile_m_v16({seq_q}) = {h.tile_m_v16(seq_q)}, expected {expected}'

@pytest.mark.parametrize('seq_q,expected_tile_m,expected_tile_k', [(16, 1, 16), (64, 4, 16), (128, 8, 16), (144, 9, 11), (176, 11, 11), (256, 8, 16), (4096, 8, 16)])
def test_compute_tiling_runtime_aligned(seq_q, expected_tile_m, expected_tile_k):
    t = h.compute_tiling(head_dim=128, seq_q=seq_q, attn_mask_type=0)
    assert t['tile_m_v16'] == expected_tile_m
    assert t['tile_k_v16'] == expected_tile_k
    assert t['tile_n_v16'] == 8

@pytest.mark.parametrize('seq_q,expected_tile_m,expected_tile_k', [(192, 6, 6), (288, 6, 6), (304, 1, 16), (320, 5, 5), (384, 8, 16)])
def test_compute_tiling_ltm_aligns_later_chunk_offset(seq_q, expected_tile_m, expected_tile_k):
    t = h.compute_tiling(head_dim=128, seq_q=seq_q, attn_mask_type=1)
    assert (t['tile_m_v16'], t['tile_k_v16']) == (expected_tile_m, expected_tile_k)
    assert seq_q // 16 % t['tile_m_v16'] == 0

def test_compute_tiling_vlm_auto_shrink():
    t = h.compute_tiling(head_dim=512, seq_q=176, attn_mask_type=0)
    assert t['tile_m_v16'] >= 1
    assert t['tile_n_v16'] >= 1
    assert t['tile_k_v16'] >= 1
    assert t['tile_m_v16'] * t['tile_n_v16'] * t['tile_k_v16'] <= 1024

def test_compute_tiling_vlm_unshrinkable_throws():
    with pytest.raises(RuntimeError, match='cannot shrink to fit VLM'):
        h.compute_tiling(head_dim=8192, seq_q=64, attn_mask_type=0)

def test_compute_tiling_rejects_zero_seq_q():
    with pytest.raises(RuntimeError, match='cannot shrink to fit VLM.*seq_q=0'):
        h.compute_tiling(head_dim=128, seq_q=0, attn_mask_type=0)

def _kc(grid_dim_x, gqa, **kw):
    h.validate_kernel_call(seq_q=kw.get('seq_q', 128), seq_k=kw.get('seq_k', 128), tile_m_v16=8, tile_n_v16=8, tile_k_v16=8, tile_m=128, grid_dim_x=grid_dim_x, gqa_group_size=gqa, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=0)

@pytest.mark.parametrize('product', [1, 4, 8, 16, 24, 32])
def test_validate_sync_accepts(product):
    _kc(grid_dim_x=product, gqa=1)

def test_halo_q12_kv2_single_block_uses_valid_product_six():
    t = h.compute_tiling(head_dim=128, seq_q=18, attn_mask_type=4)
    assert t == {'tile_m_v16': 2, 'tile_n_v16': 8, 'tile_k_v16': 16, 'tile_m': 32}
    assert h.is_valid_chunk_size(num_q_heads=12, num_kv_heads=2, num_cores=2, head_dim=128, attn_mask_type=4, cs=32, seq_len=18, position=4706)
    h.validate_kernel_call(seq_q=18, seq_k=4724, tile_m_v16=2, tile_n_v16=8, tile_k_v16=16, tile_m=32, grid_dim_x=1, gqa_group_size=6, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=4)

@pytest.mark.parametrize('product', [9, 12, 20, 28])
def test_validate_sync_rejects(product):
    with pytest.raises(RuntimeError, match='sync'):
        _kc(grid_dim_x=product, gqa=1)

def test_validate_total_tile_accepts():
    h.validate_kernel_call(seq_q=128, seq_k=128, tile_m_v16=8, tile_n_v16=16, tile_k_v16=8, tile_m=128, grid_dim_x=1, gqa_group_size=1, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=0)

def test_validate_total_tile_rejects():
    with pytest.raises(RuntimeError, match='total_tile'):
        h.validate_kernel_call(seq_q=128, seq_k=128, tile_m_v16=8, tile_n_v16=17, tile_k_v16=8, tile_m=128, grid_dim_x=1, gqa_group_size=1, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=0)
CFG = dict(num_q_heads=16, num_kv_heads=8, num_cores=8, head_dim=128, attn_mask_type=1)
CFG_QWEN3_8B = dict(num_q_heads=32, num_kv_heads=8, num_cores=8, head_dim=128, attn_mask_type=1)

def test_qwen3_8b_exact_chunk_geometry_is_not_every_aligned_value():
    admitted = {cs for cs in range(16, 257, 16) if h.is_valid_chunk_size(**CFG_QWEN3_8B, cs=cs, seq_len=1024, position=0)}
    assert admitted == {16, 32, 64, 128, 256}

def test_chunk_size_simple_accept():
    assert h.is_valid_chunk_size(**CFG, cs=512, seq_len=4096, position=0)

def test_chunk_size_rejects_broken_accumulated_product_6():
    assert h.is_valid_chunk_size(**CFG, cs=288, seq_len=288, position=0)
    assert not h.is_valid_chunk_size(**CFG, cs=288, seq_len=576, position=0)
    assert h.is_valid_chunk_size(**CFG, cs=384, seq_len=576, position=0)

@pytest.mark.parametrize("cs,tm,tk", [(640,80,80),(768,96,96)])
def test_chunk_size_sync_repair_retains_launch_constraints(cs,tm,tk):
    # The original 128-row tile has product 10/12 and remains illegal.
    # Policy3 selects a divisor with eight blocks and product16. The native
    # host probe checks the planner's actual selection for this exact CFG.
    with pytest.raises(RuntimeError, match="sync"):
        h.validate_kernel_call(seq_q=cs,seq_k=cs,tile_m_v16=8,tile_n_v16=8,
            tile_k_v16=16,tile_m=128,grid_dim_x=cs//128,gqa_group_size=2,
            hd_q_v16=8,hd_k_v16=8,hd_v_v16=8,attn_mask_type=1)
    assert h.choose_tile_k(tm//16,8,1,8)==tk//16
    for offset in range(0,4096,cs):
        rows=min(cs,4096-offset)
        m,k=(tm,tk) if rows==cs else (128,256)
        assert rows%m==0 and offset%m==0
        h.validate_kernel_call(seq_q=rows,seq_k=offset+rows,
            tile_m_v16=m//16,tile_n_v16=8,tile_k_v16=k//16,tile_m=m,
            grid_dim_x=rows//m,gqa_group_size=2,
            hd_q_v16=8,hd_k_v16=8,hd_v_v16=8,attn_mask_type=1)
    assert h.is_valid_chunk_size(**CFG,cs=cs,seq_len=4096,position=0)
    assert not h.is_valid_chunk_size(**CFG,cs=cs,seq_len=4096,position=0,max_sync_product=8)
    assert not h.is_valid_chunk_size(**CFG,cs=cs,seq_len=4095,position=0)
    assert not h.is_valid_chunk_size(**CFG,cs=cs,seq_len=4096,position=16)

def test_chunk_size_sync_accepts_multiple_of_8():
    assert h.is_valid_chunk_size(**CFG, cs=1024, seq_len=4096, position=0)

def test_profile_capability_can_reject_unmeasured_multiple_of_8_product():
    assert h.is_valid_chunk_size(**CFG, cs=1024, seq_len=4096, position=0, max_sync_product=0)
    assert not h.is_valid_chunk_size(**CFG, cs=1024, seq_len=4096, position=0, max_sync_product=8)
    assert not h.is_valid_chunk_size(**CFG, cs=512, seq_len=4096, position=0, max_sync_product=-1)

def test_chunk_size_ltm_tile_runtime_alignment_and_tail_workspace():
    assert h.is_valid_chunk_size(**CFG, cs=80, seq_len=4000, position=0)
    assert not h.is_valid_chunk_size(**CFG, cs=80, seq_len=4096, position=0)

def test_chunk_size_rejects_later_chunk_misaligned_to_tile():
    assert not h.is_valid_chunk_size(**CFG, cs=400, seq_len=800, position=0)

def test_chunk_size_tail_rejects_tile_misaligned_offset():
    assert not h.is_valid_chunk_size(**CFG, cs=480, seq_len=540, position=0)

def test_chunk_size_tail_NOW_VALID_under_runtime_alignment():
    assert h.is_valid_chunk_size(**CFG, cs=480, seq_len=560, position=0)

def test_chunk_size_tail_reject_acc():
    assert not h.is_valid_chunk_size(**CFG, cs=512, seq_len=900, position=0)

def test_chunk_size_continuation_accept():
    assert h.is_valid_chunk_size(**CFG, cs=512, seq_len=2048, position=2048)

def test_chunk_size_continuation_reject():
    assert not h.is_valid_chunk_size(**CFG, cs=512, seq_len=2048, position=128)

@pytest.mark.parametrize('cs,seq_len,expected', [(192, 576, True), (288, 576, False), (304, 608, False), (320, 640, True), (384, 768, False)])
def test_equal_two_ltm_chunk_alignment_matrix(cs, seq_len, expected):
    assert h.is_valid_chunk_size(**CFG, cs=cs, seq_len=seq_len, position=0) is expected

def test_validate_ltm_tile_check_runtime_aligned():
    h.validate_kernel_call(seq_q=144, seq_k=144, tile_m_v16=9, tile_n_v16=8, tile_k_v16=9, tile_m=144, grid_dim_x=1, gqa_group_size=1, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=1)

def test_validate_ltm_tile_check_rejects_when_tile_k_not_multiple():
    with pytest.raises(RuntimeError, match='LTM'):
        h.validate_kernel_call(seq_q=32, seq_k=32, tile_m_v16=2, tile_n_v16=8, tile_k_v16=9, tile_m=32, grid_dim_x=1, gqa_group_size=1, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=1)

def test_validate_ltm_rejects_acc_not_aligned_to_tile_m():
    with pytest.raises(RuntimeError, match='aligned to tile_m'):
        h.validate_kernel_call(seq_q=192, seq_k=384, tile_m_v16=8, tile_n_v16=8, tile_k_v16=16, tile_m=128, grid_dim_x=2, gqa_group_size=2, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=1)

def test_validate_ltm_rejects_broken_accumulated_product_6():
    with pytest.raises(RuntimeError, match='product=6'):
        h.validate_kernel_call(seq_q=288, seq_k=576, tile_m_v16=8, tile_n_v16=8, tile_k_v16=8, tile_m=128, grid_dim_x=3, gqa_group_size=2, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=1)
    h.validate_kernel_call(seq_q=288, seq_k=288, tile_m_v16=8, tile_n_v16=8, tile_k_v16=8, tile_m=128, grid_dim_x=3, gqa_group_size=2, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=1)

@pytest.mark.parametrize('seq_q,seq_k,tm_v16,tk_v16,grid', [(256, 512, 8, 16, 2), (320, 640, 5, 5, 4), (512, 1024, 8, 16, 4)])
def test_validate_ltm_accepts_aligned_equal_two(seq_q, seq_k, tm_v16, tk_v16, grid):
    h.validate_kernel_call(seq_q=seq_q, seq_k=seq_k, tile_m_v16=tm_v16, tile_n_v16=8, tile_k_v16=tk_v16, tile_m=tm_v16 * 16, grid_dim_x=grid, gqa_group_size=2, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=1)
