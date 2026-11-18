"""Check the installed permutation against the kernel's byte-address formula."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


_path = Path(__file__).resolve().parents[1] / "python/rpu_backend/quant/nvfp4_pack.py"
_spec = importlib.util.spec_from_file_location("nvfp4_tile_pack_under_test", _path)
_pack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pack)


@pytest.mark.parametrize("n,k,partition", [
    (2048, 1024, 1), (4096, 1024, 1), (1024, 2048, 0),
    (1024, 4096, 0), (512, 1024, 1),
])
def test_scale_bytes_follow_kernel_addresses(n, k, partition):
    rng = np.random.default_rng(42)
    raw = rng.integers(0, 256, size=(k // 16, n), dtype=np.uint8)
    original = torch.from_numpy(raw.copy())
    packed = _pack.repack_nvfp4_tile_scales(original, partition)
    local_n = n // 8 if partition else n
    local_k = k if partition else k // 8
    assert tuple(packed.shape) == (local_n // 64, local_k // 128, 8, 512)
    assert packed.is_contiguous() and packed.storage_offset() == 0
    np.testing.assert_array_equal(original.numpy(), raw)
    c, g, j = np.indices((8, local_k // 16, local_n))
    source = ((g // 2) * 8 * local_n * 2 + c * local_n * 2
              + (j // 64) * 128 + ((j % 64) // 16) * 32
              + (g % 2) * 16 + j % 16)
    destination = (((j // 64) * (local_k // 128) + g // 8) * 8 + c) * 512
    destination += (4 * (g % 4) + (j % 64) // 16) * 32
    destination += 16 * ((g % 8) // 4) + j % 16
    assert np.unique(destination).size == raw.size
    np.testing.assert_array_equal(
        packed.numpy().reshape(-1)[destination], raw.reshape(-1)[source])
    # This must not silently accept the original striped byte order as tscale.
    assert np.count_nonzero(packed.numpy().reshape(-1) != raw.reshape(-1)) > raw.size // 2


@pytest.mark.parametrize("kind", ["dtype", "dim", "noncontiguous", "offset", "cores", "partition", "partial_n", "partial_k", "empty"])
def test_rejects_ambiguous_or_partial_owners(kind):
    value = torch.zeros((64, 2048), dtype=torch.uint8)
    partition, cores = 1, 8
    if kind == "dtype": value = value.float()
    if kind == "dim": value = value.flatten()
    if kind == "noncontiguous": value = value.t()
    if kind == "offset": value = torch.zeros((65, 2048), dtype=torch.uint8)[1:]
    if kind == "cores": cores = 4
    if kind == "partition": partition = 2
    if kind == "partial_n": value = torch.zeros((64, 1296), dtype=torch.uint8)
    if kind == "partial_k": value = torch.zeros((4, 2048), dtype=torch.uint8)
    if kind == "empty": value = torch.zeros((0, 2048), dtype=torch.uint8)
    with pytest.raises(ValueError):
        _pack.repack_nvfp4_tile_scales(value, partition, cores)
    with pytest.raises(ValueError):
        _pack.repack_nvfp4_spm_scales(value, partition, cores)


@pytest.mark.parametrize("n,k,partition", [(2048, 1024, 1), (1024, 2048, 0),
                                         (1024, 4096, 0), (4096, 1024, 1)])
def test_per_core_spm_scale_addresses_and_rotated_core_control(n, k, partition):
    raw = np.random.default_rng(71).integers(0, 256, size=(k // 16, n), dtype=np.uint8)
    packed = _pack.repack_nvfp4_spm_scales(torch.from_numpy(raw.copy()), partition)
    ln, lk = (n // 8, k) if partition else (n, k // 8)
    assert tuple(packed.shape) == (8, ln // 64, lk // 128, 512)
    assert packed.is_contiguous() and packed.storage_offset() == 0
    c, g, j = np.indices((8, lk // 16, ln))
    source = ((g // 2) * 8 * ln * 2 + c * ln * 2 + (j // 64) * 128
              + ((j % 64) // 16) * 32 + (g % 2) * 16 + j % 16)
    local = ((j // 64) * (lk // 128) + g // 8) * 512
    local += (4 * (g % 4) + (j % 64) // 16) * 32 + 16 * ((g % 8) // 4) + j % 16
    destination = c * (ln * lk // 16) + local
    actual = packed.numpy().reshape(-1)
    np.testing.assert_array_equal(actual[destination], raw.reshape(-1)[source])
    rotated = ((c + 1) % 8) * (ln * lk // 16) + local
    assert np.count_nonzero(actual[rotated] != raw.reshape(-1)[source]) > raw.size // 2
