"""Exact copy/zero-fill and tail ownership for the production SPM pad wrapper."""
import os

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(os.environ.get("RPU_HW_AVAILABLE") != "1", reason="requires RPU hardware")
def test_pad_channel_short_rows_and_grid_tail():
    import rpu_backend  # noqa: F401
    import _load_conformance  # noqa: F401

    generator = torch.Generator().manual_seed(42)
    for rows, cols, front, tail in ((64, 3, 1, 4), (33, 129, 7, 8),
                                    (17, 65, 0, 7), (1, 3072, 0, 1024)):
        logical = rows * cols
        x = torch.full((((logical + 7) // 8) * 8,), 123, dtype=torch.float16)
        x[:logical] = torch.randn(logical, generator=generator).half()
        reference = F.pad(x[:logical].reshape(rows, cols), (front, tail)).flatten()
        seed = torch.full((((reference.numel() + 7) // 8) * 8 + 8,),
                          117, dtype=torch.float16)
        xr, yr = x.to("rpu"), seed.to("rpu")
        for cores in (1, 8):
            got = torch.ops.rpu.test_pad_channel(xr, yr, rows, cols, front, tail, cores)
            assert got.shape == (cores, seed.numel()) and got.dtype == torch.float16
            assert torch.equal(got[:, :reference.numel()].view(torch.int16),
                               reference.expand(cores, -1).view(torch.int16))
            assert torch.equal(got[:, reference.numel():].view(torch.int16),
                               seed[reference.numel():].expand(cores, -1).view(torch.int16))
        assert torch.equal(xr.cpu().view(torch.int16), x.view(torch.int16))
        assert torch.equal(yr.cpu().view(torch.int16), seed.view(torch.int16))
