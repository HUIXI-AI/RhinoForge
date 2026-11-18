"""CPU codecs for the grouped UINT4 N32K32 weight layout.

Logical codes are uint8 [E,N,K]. Packed bytes are
[E,Nblock32,Kblock32,core,thread16,Nhalf2,byte16], flattened after E.
N/K arguments are global; each TP shard pads N independently to 32. The
caller supplies localN16/localK32 alignment and owns any logical K padding.
This is the weight layout of rhino-ops a2a3a2d8's unified ParallelLinear.

These functions preserve codes; they do not quantize or process scale/ZP.
With unchanged physical N/K and group size, metadata still uses the existing
q4_n64_controller_v1 codec in grouped_uint4. Do not swizzle it twice.
Dense and grouped UINT4 installations share this codec. Pre-migration N16K64
conversion is intentionally unsupported; callers must provide the new N32K32
layout explicitly, because dtype/shape alone cannot identify a legacy layout.
Packing alone does not qualify a kernel, model path or core topology.
"""
from __future__ import annotations

import torch


WEIGHT_LAYOUT = "grouped_uint4_n32k32_v1"


def _local_shape(n: int, k: int, partition: str, num_cores: int) -> tuple[int, int]:
    if type(num_cores) is not int or not 1 <= num_cores <= 8:
        raise ValueError("N32K32 requires integer num_cores in [1,8]")
    if type(n) is not int or type(k) is not int or n <= 0 or k <= 0:
        raise ValueError("N32K32 requires positive integer global n and k")
    if partition == "col":
        if n % num_cores:
            raise ValueError("N32K32 col requires N divisible by num_cores")
        nc, kc = n // num_cores, k
    elif partition == "row":
        if k % num_cores:
            raise ValueError("N32K32 row requires K divisible by num_cores")
        nc, kc = n, k // num_cores
    else:
        raise ValueError("partition must be col or row")
    if nc % 16 or kc % 32:
        raise ValueError("N32K32 requires localN16 and localK32 alignment")
    return nc, kc


def _check_cpu_u8(tensor: torch.Tensor, dim: int, name: str) -> None:
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
            or tensor.dtype != torch.uint8 or tensor.dim() != dim
            or tensor.numel() == 0):
        raise ValueError(f"{name} requires nonempty CPU uint8 rank-{dim} storage")


def pack_grouped_uint4_n32k32(
    codes: torch.Tensor, *, partition: str, num_cores: int = 8,
) -> torch.Tensor:
    """Pack CPU uint8 [E,N,K]; return contiguous [E,packed_expert_bytes].

    Codes must be in [0,15]. Newly introduced N padding has code zero; the
    caller must also provide zero metadata/output treatment for padded rows.
    Source storage is never modified, and the result owns fresh storage.
    """
    _check_cpu_u8(codes, 3, "UINT4 codes")
    e, n, k = map(int, codes.shape)
    nc, kc = _local_shape(n, k, partition, num_cores)
    if int(codes.max()) > 15:
        raise ValueError("UINT4 codes must lie in [0,15]")
    local = (codes.reshape(e, num_cores, nc, kc) if partition == "col"
             else codes.reshape(e, nc, num_cores, kc).permute(0, 2, 1, 3))
    nb = (nc + 31) // 32
    padded = torch.zeros((e, num_cores, nb * 32, kc), dtype=torch.uint8)
    padded[:, :, :nc].copy_(local)
    # The two K16 halves become low/high nibbles. The two N16 halves remain
    # adjacent byte16 vectors, so one thread fetches N=t and N=t+16 together.
    blocks = padded.reshape(e, num_cores, nb, 2, 16, kc // 32, 2, 16)
    packed = blocks[..., 0, :] | (blocks[..., 1, :] << 4)
    return packed.permute(0, 2, 5, 1, 4, 3, 6).contiguous().reshape(e, -1)


def unpack_grouped_uint4_n32k32(
    packed: torch.Tensor, *, n: int, k: int, partition: str, num_cores: int = 8,
) -> torch.Tensor:
    """Return original logical CPU uint8 [E,N,K] without N32 padding.

    n/k are the pre-N32-padding global dimensions. Require the exact packed
    byte count and zero N padding, so a wrong physical extent fails early.
    No dtype/shape check can distinguish all incorrectly labelled layouts;
    the caller must retain the weight-format identity alongside its owner.
    """
    _check_cpu_u8(packed, 2, "Packed N32K32 weight")
    nc, kc = _local_shape(n, k, partition, num_cores)
    e = int(packed.shape[0])
    nb = (nc + 31) // 32
    if packed.shape[1] != num_cores * nb * 32 * kc // 2:
        raise ValueError("Packed N32K32 weight has incorrect expert byte count")
    blocks = packed.reshape(e, nb, kc // 32, num_cores, 16, 2, 16)
    blocks = blocks.permute(0, 3, 1, 5, 4, 2, 6)
    local = torch.stack((blocks & 15, blocks >> 4), dim=-2).reshape(
        e, num_cores, nb * 32, kc)
    if nc != nb * 32 and bool(local[:, :, nc:].any()):
        raise ValueError("Packed N32K32 weight N padding must be zero")
    local = local[:, :, :nc]
    if partition == "row":
        local = local.permute(0, 2, 1, 3)
    return local.contiguous().reshape(e, n, k)




__all__ = [
    "WEIGHT_LAYOUT", "pack_grouped_uint4_n32k32",
    "unpack_grouped_uint4_n32k32",
]
