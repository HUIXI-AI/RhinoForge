"""Unit tests for int4 swizzle+pack helper (pure CPU, no rpu_backend C-ext).

Imported standalone so collection does not load the C extension (segfaults
outside the sudo/LD_PRELOAD hardware context).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_QUANT = Path(__file__).resolve().parents[2] / "quant"
sys.path.insert(0, str(_QUANT))
import _common as _qc  # noqa: E402
import int4_pgrp_pack as _pg  # noqa: E402

_RHINO = Path(os.environ.get("RHINO_OPS_GENERATOR", "/path/to/shared-assets"))


def _physical_addresses(n, k, partition, cores):
    """Independent scalar ABI addresses for every logical nibble."""
    local_n = n // cores if partition else n
    local_k = k if partition else k // cores
    nn, kk = np.indices((n, k), dtype=np.int64)
    core = nn // local_n if partition else kk // local_k
    ln, lk = nn % local_n, kk % local_k
    addresses = (((((ln // 32) * (local_k // 32) + lk // 32) * cores
                   + core) * 16 + ln % 16) * 2 + ln % 32 // 16) * 16 + lk % 16
    return addresses, (lk % 32 // 16) * 4


def _ref_n32k32_pack(vals, partition, cores):
    n, k = vals.shape
    local_n = n // cores if partition else n
    local_k = k if partition else k // cores
    result = np.zeros(cores * ((local_n + 31) // 32) * 32 * local_k // 2,
                      dtype=np.uint8)
    addresses, shifts = _physical_addresses(n, k, partition, cores)
    nibbles = vals.numpy().astype(np.uint8) & 15
    np.bitwise_or.at(result, addresses, (nibbles << shifts).astype(np.uint8))
    return result


def _unpack_n32k32(packed, n, k, partition, cores, signed):
    addresses, shifts = _physical_addresses(n, k, partition, cores)
    codes = ((packed.numpy()[addresses] >> shifts) & 15).astype(np.int8)
    return torch.from_numpy(np.where(codes >= 8, codes - 16, codes) if signed else codes.astype(np.uint8))


@pytest.mark.parametrize("partition", [0, 1])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_pgrp_swizzle_pack_matches_reference(partition, cores):
    torch.manual_seed(0)
    N, K = cores * 32, cores * 64
    vals = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    packed = _pg.swizzle_pack_int4_pgrp(vals, partition, cores)
    golden = _ref_n32k32_pack(vals, partition, cores)
    assert packed.dtype == torch.uint8
    assert packed.numel() == N * K // 2
    np.testing.assert_array_equal(packed.numpy(), golden)


@pytest.mark.parametrize("cores", [4, 6, 8])
def test_bounded_head_packing_matches_full_payload_across_block_boundary(cores):
    weight = torch.linspace(-1, 1, cores * 272 * 64).reshape(cores * 272, 64).half()
    values, logical_scale, _ = _pg.quantize_int4_group_wise(weight, 32)
    packed, scale = _pg.quantize_pack_int4_pgrp_col_bounded(weight, num_cores=cores)
    assert torch.equal(packed.flatten(), _pg.swizzle_pack_int4_pgrp(values, 1, cores))
    assert torch.equal(scale, _pg.swizzle_int4_pgrp_scale(logical_scale, 32, 1, cores))


@pytest.mark.parametrize("cores", [0, True, 2.5, 9])
def test_bounded_head_rejects_invalid_core_count(cores):
    with pytest.raises(ValueError, match="core count"):
        _pg.quantize_pack_int4_pgrp_col_bounded(torch.zeros(128, 64), num_cores=cores)


def test_quantize_int4_per_channel_ranges():
    torch.manual_seed(0)
    W = torch.randn(64, 128).to(torch.float16)
    vals, scale = _qc.quantize_linear_per_channel(W, bits=4)
    assert vals.dtype == torch.int8 and scale.dtype == torch.float16
    assert int(vals.max()) <= 7 and int(vals.min()) >= -8
    assert scale.numel() == 64


def test_per_channel_scale_is_repeated_into_pgrp_payload():
    torch.manual_seed(4)
    vals = torch.randint(-8, 8, (256, 512), dtype=torch.int8)
    scale = torch.arange(1, 257, dtype=torch.float16)
    packed, physical = _pg.pack_int4_per_channel_as_pgrp(
        vals, scale, partition=0, num_cores=8
    )
    logical = scale.reshape(1, -1).expand(16, -1).contiguous()
    expected_scale = _pg.swizzle_int4_pgrp_scale(
        logical, group_size=32, partition=0, num_cores=8
    )
    assert torch.equal(
        packed, _pg.swizzle_pack_int4_pgrp(vals, partition=0, num_cores=8)
    )
    assert torch.equal(physical, expected_scale)
    assert physical.size(0) == 32


@pytest.mark.parametrize(
    "partition,group_size,g,n", [(1, 32, 5, 130), (0, 64, 10, 65), (1, 128, 5, 130)]
)
def test_pgrp_scale_swizzle_physical_order_and_metadata(partition, group_size, g, n):
    cores = 2
    logical = torch.arange(1, g * n + 1, dtype=torch.float16).reshape(g, n)
    actual = _pg.swizzle_int4_pgrp_scale(logical, group_size, partition, cores)

    local_g = g if partition == 1 else g // cores
    local_n = n // cores if partition == 1 else n
    expected = torch.zeros(
        ((local_g + 3) // 4, (local_n + 63) // 64, cores, 4, 64),
        dtype=torch.float16,
    )
    for global_g in range(g):
        for global_n in range(n):
            core = global_n // local_n if partition == 1 else global_g // local_g
            lg = global_g if partition == 1 else global_g % local_g
            ln = global_n % local_n if partition == 1 else global_n
            expected[lg // 4, ln // 64, core, lg % 4, ln % 64] = logical[
                global_g, global_n
            ]

    assert actual.shape == (group_size, expected.numel() // group_size)
    assert actual.is_contiguous()
    assert torch.equal(actual.flatten(), expected.flatten())
    assert torch.count_nonzero(actual) == logical.numel()  # padded slots are zero


def test_pgrp_scale_swizzle_rejects_invalid_abi():
    scale = torch.ones((5, 130), dtype=torch.float16)
    with pytest.raises(ValueError, match="group_size"):
        _pg.swizzle_int4_pgrp_scale(scale, 16, 1, 2)
    with pytest.raises(ValueError, match="N%cores"):
        _pg.swizzle_int4_pgrp_scale(scale, 32, 1, 3)
    with pytest.raises(ValueError, match="G%cores"):
        _pg.swizzle_int4_pgrp_scale(scale, 32, 0, 2)
    with pytest.raises(ValueError, match="fp16"):
        _pg.swizzle_int4_pgrp_scale(scale.float(), 32, 1, 2)


@pytest.mark.parametrize("partition", [0, 1])
@pytest.mark.parametrize("cores", [1, 4, 6, 8])
@pytest.mark.parametrize("local_n,local_k", [(16, 32), (48, 96), (80, 128)])
@pytest.mark.parametrize("signed", [False, True])
def test_n32k32_roundtrip_and_zero_padding(partition, cores, local_n, local_k, signed):
    n, k = (local_n * cores, local_k) if partition else (local_n, local_k * cores)
    rng = torch.Generator().manual_seed(91)
    values = torch.randint(0, 16, (n, k), dtype=torch.uint8, generator=rng)
    if signed:
        values = values.to(torch.int8) - 8
    packed = _pg.swizzle_pack_w4_n32k32(values, partition, cores)
    expected = _ref_n32k32_pack(values, partition, cores)
    np.testing.assert_array_equal(packed.numpy(), expected)
    assert torch.equal(_unpack_n32k32(packed, n, k, partition, cores, signed), values)
    shape = _pg.w4_storage_shape(n, k, partition, cores)
    assert shape[0] * shape[1] == packed.numel()
    # Full decode verifies unused output rows are positive zero, not uninitialized.
    local = packed.view((local_n + 31) // 32, local_k // 32, cores, 16, 2, 16)
    if local_n % 32:
        assert torch.count_nonzero(local[-1, :, :, :, 1]) == 0


@pytest.mark.parametrize("fault", ["low", "high", "dtype", "rank", "local_n", "local_k", "partition"])
def test_n32k32_rejects_invalid_logical_payload(fault):
    values = torch.zeros(128, 64, dtype=torch.int8)
    partition = 1
    if fault == "low": values[0, 0] = -9
    elif fault == "high": values[0, 0] = 8
    elif fault == "dtype": values = values.float()
    elif fault == "rank": values = values.flatten()
    elif fault == "local_n": values = values[:120]
    elif fault == "local_k": values = values[:, :48]
    elif fault == "partition": partition = 2
    with pytest.raises(ValueError):
        _pg.swizzle_pack_int4_pgrp(values, partition, 8)


@pytest.mark.skipif(not (_RHINO / "ParallelLinearGenerator.py").is_file(),
                    reason="rhino-ops unified generator not available")
def test_pack_byte_identical_to_rhino_unified_generator():
    # Extract only the pure source packer; importing the generator would load
    # the hardware-generation stack just to compare its tensor transform.
    import ast
    source = ast.parse((_RHINO / "ParallelLinearGenerator.py").read_text())
    klass = next(node for node in source.body if isinstance(node, ast.ClassDef))
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef)
                  and node.name == "pack_w4_n32k32")
    method.decorator_list = []
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(_RHINO / "ParallelLinearGenerator.py"), "exec"), scope)
    for partition in (0, 1):
        vals = torch.randint(-8, 8, (384, 768), dtype=torch.int8)
        actual = _pg.swizzle_pack_int4_pgrp(vals, partition, 8)
        expected = scope["pack_w4_n32k32"](vals.numpy(), 8, "col" if partition else "row")
        np.testing.assert_array_equal(actual.numpy(), expected.reshape(-1))
