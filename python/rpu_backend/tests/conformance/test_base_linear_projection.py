"""Exact sparse fixtures exercise the actual formal BASE production launcher.

These integer/power-of-two fixtures prove register/layout/tail semantics; they
are not scalar error thresholds for general floating point or model outputs.
"""
from __future__ import annotations

import os
import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RPU_HW_AVAILABLE") != "1", reason="requires locked RPU hardware")


@pytest.mark.parametrize("family,m", [("fp16", 41), ("int8", 513), ("e4m3", 65)])
@pytest.mark.parametrize("partition", [0, 1])
@pytest.mark.parametrize("acc32", [False, True])
def test_base_linear_sparse_exact(family, m, partition, acc32):
    from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight, tp_row_swizzle_mc_weight

    cores, local_n, local_k = 8, 48, 96
    n, k = (local_n, cores * local_k) if partition == 0 else (cores * local_n, local_k)
    # One nonzero in each K shard. This remains exact under both accumulation
    # modes and exposes every core, output channel and tail row independently.
    x = torch.zeros((m, k), dtype=torch.float16)
    row = torch.arange(m)
    for core in range(cores if partition == 0 else 1):
        x[row, core * local_k + row % local_k] = (1 + row % 2).half()
    logical_w = ((torch.arange(n)[:, None] * 3 + torch.arange(k)[None, :]) % 5 - 2).half()
    scale = (2.0 ** ((torch.arange(n) % 3) - 1)).half() if family != "fp16" else None
    bias = ((torch.arange(n) % 7) - 3).half() / 2
    if family == "fp16":
        storage = logical_w
    elif family == "int8":
        storage = logical_w.to(torch.int8)
    else:
        storage = logical_w.to(torch.float8_e4m3fn).view(torch.uint8)
    pack = tp_row_swizzle_mc_weight if partition == 0 else tp_col_swizzle_mc_weight
    packed = pack(storage, cores, dwidth=storage.element_size())
    output = torch.ops.rpu.test_base_linear_projection(
        x.to("rpu"), packed.to("rpu"),
        scale.to("rpu") if scale is not None else None, bias.to("rpu"),
        partition, cores, acc32, 3)
    expected = x.float() @ logical_w.float().T
    if scale is not None:
        expected *= scale.float()
    expected += bias.float()
    assert output.shape == (m, n) and output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.equal(output, expected), (
        family, partition, acc32, float((output - expected).abs().max()))
