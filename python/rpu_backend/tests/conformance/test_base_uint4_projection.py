"""Dense BASE UINT4: production wrapper, independent exact sparse oracle."""
import os
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires locked RPU')


@pytest.mark.parametrize('group_size', [32, 64, 128])
@pytest.mark.parametrize('col', [False, True])
@pytest.mark.parametrize('acc32', [False, True])
def test_dense_base_uint4_group_and_n32_tails(group_size, col, acc32):
    from rpu_backend.quant.awq_gemm import swizzle_pack_uint4_pgrp
    from rpu_backend.quant.int4_pgrp_pack import swizzle_int4_pgrp_scale

    cores, m, ln, lk = 8, 17, 96, 384
    n, k = (cores * ln, lk) if col else (ln, cores * lk)
    partition = int(col)
    # N96 spans one N64 block and an N32 tail. G64/G128 have incomplete G4
    # metadata blocks. Nonuniform zero points and scales distinguish group,
    # channel, TP shard and nibble selection without floating rounding noise.
    q = ((torch.arange(n)[:, None] * 5 + torch.arange(k)[None, :] * 3) % 16).byte()
    g = k // group_size
    zero = ((torch.arange(g)[:, None] * 7 + torch.arange(n)[None, :]) % 16).byte()
    scale = torch.pow(2., -(torch.arange(g)[:, None] + torch.arange(n)[None, :]) % 3 - 3).half()
    x = torch.zeros((cores, m, lk), dtype=torch.float16)
    expected = torch.empty((cores, m, ln), dtype=torch.float16)
    for core in range(cores):
        ns = slice(core * ln, (core + 1) * ln) if col else slice(None)
        for row in range(m):
            kk = (row * 17 + core * 7) % lk
            global_k = kk if col else core * lk + kk
            x[core, row, kk] = -0.5 if row % 2 else 0.5
            expected[core, row] = ((q[ns, global_k].float() - zero[global_k // group_size, ns].float()) *
                                   scale[global_k // group_size, ns].float() * x[core, row, kk]).half()
    packed = swizzle_pack_uint4_pgrp(q, partition, cores).view(n, k // 2)
    s = swizzle_int4_pgrp_scale(scale, group_size, partition, cores)
    z = swizzle_int4_pgrp_scale(-zero.half(), group_size, partition, cores)
    actual = torch.ops.rpu.test_base_uint4_projection(
        x.to('rpu'), packed.to('rpu'), s.to('rpu'), z.to('rpu'), col, acc32).cpu()
    assert actual.shape == expected.shape and actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected), float((actual.float() - expected.float()).abs().max())


@pytest.mark.parametrize('col', [False, True])
@pytest.mark.parametrize('acc32', [False, True])
def test_dense_base_uint4_explicit_n16_to_n32_owner(col, acc32):
    """Codec padding consumed as an explicit physical N32 owner, not N16 admission."""
    from rpu_backend.quant.awq_gemm import swizzle_pack_uint4_pgrp
    from rpu_backend.quant.int4_pgrp_pack import swizzle_int4_pgrp_scale

    cores, m, logical_ln, physical_ln, lk, group_size = 8, 17, 16, 32, 384, 64
    logical_n, k = (cores * logical_ln, lk) if col else (logical_ln, cores * lk)
    physical_n = cores * physical_ln if col else physical_ln
    q = ((torch.arange(logical_n)[:, None] * 5 + torch.arange(k)[None, :] * 3) % 16).byte()
    groups = k // group_size
    zero = ((torch.arange(groups)[:, None] * 7 + torch.arange(logical_n)[None, :]) % 16).byte()
    scale = torch.pow(2., -(torch.arange(groups)[:, None] + torch.arange(logical_n)[None, :]) % 3 - 3).half()
    # Explicit ownership: each core's logical channels occupy its first16 slots.
    # Metadata must be extended per shard, not by appending to global col-N.
    physical_scale = torch.zeros((groups, physical_n), dtype=torch.float16)
    physical_zero = torch.zeros_like(physical_scale)
    for core in range(cores if col else 1):
        source = slice(core * logical_ln, (core + 1) * logical_ln) if col else slice(None)
        dest = slice(core * physical_ln, core * physical_ln + logical_ln) if col else slice(0, logical_ln)
        physical_scale[:, dest] = scale[:, source]
        physical_zero[:, dest] = -zero[:, source].half()
    x = torch.zeros((cores, m, lk), dtype=torch.float16)
    expected = torch.zeros((cores, m, physical_ln), dtype=torch.float16)
    for core in range(cores):
        ns = slice(core * logical_ln, (core + 1) * logical_ln) if col else slice(None)
        for row in range(m):
            kk = (row * 17 + core * 7) % lk
            global_k = kk if col else core * lk + kk
            x[core, row, kk] = -0.5 if row % 2 else 0.5
            expected[core, row, :logical_ln] = (
                (q[ns, global_k].float() - zero[global_k // group_size, ns].float()) *
                scale[global_k // group_size, ns].float() * x[core, row, kk]).half()
    # packer introduces zero code rows; bind the physical extent explicitly.
    packed = swizzle_pack_uint4_pgrp(q, int(col), cores).view(physical_n, k // 2)
    s = swizzle_int4_pgrp_scale(physical_scale, group_size, int(col), cores)
    z = swizzle_int4_pgrp_scale(physical_zero, group_size, int(col), cores)
    actual = torch.ops.rpu.test_base_uint4_projection(
        x.to('rpu'), packed.to('rpu'), s.to('rpu'), z.to('rpu'), col, acc32).cpu()
    assert actual.shape == expected.shape and actual.dtype == torch.float16
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected), float((actual.float() - expected.float()).abs().max())
    assert torch.count_nonzero(actual[..., logical_ln:]).item() == 0
