"""INT4 per-channel weight pack for the wINT4a16_per_channel kernel ABI.

Quantize a Linear weight to per-channel signed int4 (values in int8 container),
swizzle with the kernel's num_ele_32B=64 (explicit dwidth=0.5, NOT auto from
element_size), then nibble-pack (subbyte_codec semantics). Output is a flat
uint8 DDR buffer of N*K/2 bytes plus the fp16 per-channel scale.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from ._common import quantize_linear_per_channel
except ImportError:  # standalone import (no rpu_backend package / C-ext)
    from _common import quantize_linear_per_channel

_NUM_ELE_32B_INT4 = 64  # 32 bytes / 0.5 byte-per-int4


def quantize_int4_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N,K] float -> (int4 values in int8 container [N,K], fp16 per-channel scale [N])."""
    return quantize_linear_per_channel(weight, bits=4)


def _swizzle_int4(vals: torch.Tensor, partition: int, num_cores: int) -> torch.Tensor:
    """Swizzle logical int4 values [N,K] -> flat C-order int tensor (num_ele_32B=64)."""
    N, K = vals.shape
    e = _NUM_ELE_32B_INT4
    if partition == 1:  # col
        if N % (16 * num_cores) != 0 or K % e != 0:
            raise ValueError(
                f"int4 col swizzle needs N%(16*cores)==0 and K%64==0, got N={N},K={K},cores={num_cores}")
        v = vals.view(num_cores, N // 16 // num_cores, 16, K // e, e).permute(1, 3, 0, 2, 4)
    elif partition == 0:  # row
        if N % 16 != 0 or K % (e * num_cores) != 0:
            raise ValueError(
                f"int4 row swizzle needs N%16==0 and K%(64*cores)==0, got N={N},K={K},cores={num_cores}")
        v = vals.view(N // 16, 16, num_cores, K // e // num_cores, e).permute(0, 3, 2, 1, 4)
    else:
        raise ValueError(f"partition must be 0(row) or 1(col), got {partition}")
    return v.contiguous().reshape(-1)


def _pack_int4_signed(flat: np.ndarray) -> np.ndarray:
    """Pack signed int4 (-8..7): byte=(odd<<4)|(even&0xF), two's complement nibbles."""
    a = np.asarray(flat, dtype=np.int8)
    if a.size % 2 != 0:
        a = np.append(a, np.int8(0))
    a = np.where(a < 0, a + 16, a).astype(np.uint8) & 0x0F
    return ((a[1::2] << 4) | a[0::2]).astype(np.uint8)


def swizzle_pack_int4(int4_vals: torch.Tensor, partition: int, num_cores: int) -> torch.Tensor:
    """int4 values [N,K] (int8 container) -> packed uint8 torch tensor [N*K/2]."""
    flat = _swizzle_int4(int4_vals.to(torch.int64), partition, num_cores)
    packed = _pack_int4_signed(flat.cpu().numpy())
    return torch.from_numpy(packed).contiguous()
