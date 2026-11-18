"""Lossless QuantTrio AWQ GEMM codecs; direct zero, never GPTQ zero-plus-one.

Source words pack the N axis, not K. Logical codes are uint8 [N,K], scales and
integer zero points [K/128,N]. Runtime UINT4 uses the BASE N32K32 TP permutation;
scale and FP16 negative zero use the same controller-striped metadata layout.
"""
from __future__ import annotations

import numpy as np
import torch

from .int4_pgrp_pack import swizzle_int4_pgrp_scale
from .grouped_uint4_n32k32 import pack_grouped_uint4_n32k32


_BIT_TO_LOGICAL = (0, 2, 4, 6, 1, 3, 5, 7)


def unpack_awq_words(words: torch.Tensor) -> torch.Tensor:
    if (words.device.type != "cpu" or words.dtype != torch.int32
            or words.dim() != 2 or words.numel() == 0 or not words.is_contiguous()):
        raise ValueError("AWQ words must be nonempty contiguous CPU INT32 matrices")
    source = words.numpy().view(np.uint32)
    values = np.empty((source.shape[0], source.shape[1] * 8), dtype=np.uint8)
    for bit_lane, logical_lane in enumerate(_BIT_TO_LOGICAL):
        values[:, logical_lane::8] = (source >> (4 * bit_lane)) & 15
    return torch.from_numpy(values)


def pack_awq_words(values: torch.Tensor) -> torch.Tensor:
    """Reconstruct original words, including their signed INT32 bit patterns."""
    _check_uint4(values)
    if values.shape[1] % 8:
        raise ValueError("AWQ word width must be divisible by 8")
    source = values.numpy().astype(np.uint32)
    words = np.zeros((source.shape[0], source.shape[1] // 8), dtype=np.uint32)
    for bit_lane, logical_lane in enumerate(_BIT_TO_LOGICAL):
        words |= source[:, logical_lane::8] << (4 * bit_lane)
    return torch.from_numpy(words.view(np.int32))


def _check_uint4(value):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != torch.uint8 or value.dim() != 2
            or value.numel() == 0 or not value.is_contiguous()):
        raise ValueError("UINT4 requires a nonempty contiguous CPU uint8 matrix")
    if int(value.max()) > 15:
        raise ValueError("UINT4 codes must lie in [0,15]")


def unpack_awq_gemm(qweight, qzeros, scales, *, n: int, k: int):
    """Validate/unpack one original G128 triplet; retain the scale's FP16 bits."""
    if n <= 0 or k <= 0 or n % 8 or k % 128:
        raise ValueError("AWQ GEMM requires N%8==0 and K%128==0")
    for name, value, dtype, shape in (
        ("qweight", qweight, torch.int32, (k, n // 8)),
        ("qzeros", qzeros, torch.int32, (k // 128, n // 8)),
        ("scales", scales, torch.float16, (k // 128, n)),
    ):
        if (value.device.type != "cpu" or value.dtype != dtype
                or tuple(value.shape) != shape or not value.is_contiguous()):
            raise ValueError(f"AWQ {name} requires contiguous CPU {dtype} {shape}")
    if not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("AWQ scales must be finite and positive")
    q, zero = unpack_awq_words(qweight), unpack_awq_words(qzeros)
    if not torch.equal(pack_awq_words(q), qweight) or not torch.equal(pack_awq_words(zero), qzeros):
        raise ValueError("AWQ qweight/qzeros lossless roundtrip failed")
    return q.t().contiguous(), zero, scales


def swizzle_pack_uint4_pgrp(codes, partition, num_cores=8):
    """Dense shared-W4 payload, with a distinct unsigned range contract."""
    _check_uint4(codes)
    if type(num_cores) is not int or num_cores not in (4, 6, 8):
        raise ValueError("QuantTrio AWQ requires 4, 6 or 8 cores")
    if type(partition) is not int or partition not in (0, 1):
        raise ValueError("partition must be 0(row) or 1(col)")
    return pack_grouped_uint4_n32k32(
        codes.unsqueeze(0), partition="col" if partition else "row",
        num_cores=num_cores,
    ).reshape(-1)


def swizzle_awq_negative_zero(zero, partition, num_cores=8):
    _check_uint4(zero)
    if type(num_cores) is not int or num_cores not in (4, 6, 8):
        raise ValueError("QuantTrio AWQ requires 4, 6 or 8 cores")
    # Conversion precedes negation: negating uint8 would wrap modulo 256.
    return swizzle_int4_pgrp_scale(-zero.to(torch.float16), 128, partition, num_cores)
