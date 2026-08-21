"""INT4 group-wise (pgrp / AWQ-style) weight pack for the
parallel_linear_wint4a16_pgrp_m{m}n{n}k128 kernel ABI.

Group-wise symmetric int4 along K (group_size, default 32), per output channel.
The weight transform follows the operator ABI:
  per-core 16N x 64K swizzle (tp_col/row_swizzle_mc_weight)
  -> *deinterleaved* int4 packing inside each 64-value K-group (_int4_pack_perm)
  -> two's-complement nibble pack (reuses int4_pack._pack_int4_signed).
Logical scale is (G=K/group_size, N) fp16 row-major.  The kernel consumes a
controller-striped [ceil(localG/4), ceil(localN/64), cores, 4, 64] payload;
swizzle_int4_pgrp_scale builds that payload and reshapes it to a 2-D tensor
whose first dimension carries group_size metadata for the C++ dispatcher.

The C++ dispatcher (rpu_launch_linear_spm_to_spm_acc16_kernel, kByte + 2D-scale
branch) detects pgrp by the 2-D scale and selects the tiled kernel via
select_tile_int4. Numerics are group-wise, so a pgrp checkpoint is NOT
interchangeable with a per-channel one.
"""
from __future__ import annotations

import numpy as np
import torch

try:
    from .int4_pack import _pack_int4_signed
except ImportError:  # standalone import (no rpu_backend package)
    from int4_pack import _pack_int4_signed

_NUM_ELE_32B_INT4 = 64  # 32 bytes / 0.5 byte-per-int4
_FP16_TINY = torch.finfo(torch.float16).tiny


def quantize_int4_group_wise(
    weight: torch.Tensor, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """[N,K] float -> (int4 vals int8-container [N,K], fp16 scale [G,N], group_size).

    Symmetric per (output-channel, K-group) block, qmax=7 / qmin=-8 (matching the
    per-channel scheme). Self-consistent: dequant(int4, scale) is exactly what the
    kernel computes (scale[g,n] * int4[n,k]), so the CPU reference uses the same.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D Linear weight, got {tuple(weight.shape)}")
    N, K = weight.shape
    if K % group_size != 0:
        raise ValueError(f"K={K} not divisible by group_size={group_size}")
    G = K // group_size
    qmax, qmin = 7, -8

    w = weight.detach().to(torch.float32).view(N, G, group_size)
    amax = w.abs().amax(dim=2)                                   # [N, G]
    scale = (amax / float(qmax)).to(torch.float16).clamp_min(_FP16_TINY)  # [N, G]
    q = (
        torch.round(w / scale.to(torch.float32).unsqueeze(2))
        .clamp_(qmin, qmax)
        .to(torch.int8)
        .view(N, K)
    )
    scale_gn = scale.t().contiguous()                            # [G, N], N contiguous
    return q, scale_gn, group_size


def dequantize_int4_group_wise(int4_vals: torch.Tensor, scale_gn: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_int4_group_wise -> fp16 [N, K] for CPU comparison."""
    G, N = scale_gn.shape
    K = int4_vals.shape[1]
    gs = K // G
    scale_ng = scale_gn.t().to(torch.float16)                    # [N, G]
    deq = int4_vals.view(N, G, gs).to(torch.float16) * scale_ng.unsqueeze(2)
    return deq.view(N, K)


def swizzle_int4_pgrp_scale(
    scale_gn: torch.Tensor, group_size: int, partition: int, num_cores: int
) -> torch.Tensor:
    """Pack logical fp16 ``[global_G, global_N]`` scales for the pgrp ABI.

    Physical storage is flattened from
    ``[ceil(localG/4), ceil(localN/64), cores, 4, 64]``.  The returned view is
    ``[group_size, physical_numel/group_size]``: its shape is ABI metadata;
    its contiguous bytes retain the controller-striped physical order.
    ``partition`` follows the Linear convention: 0=row, 1=col.
    """
    if scale_gn.dim() != 2:
        raise ValueError(f"expected logical scale [G,N], got {tuple(scale_gn.shape)}")
    if scale_gn.dtype != torch.float16:
        raise ValueError(f"pgrp scale must be fp16, got {scale_gn.dtype}")
    if group_size not in (32, 64, 128):
        raise ValueError(f"group_size must be 32, 64, or 128, got {group_size}")
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")

    global_g, global_n = scale_gn.shape
    if global_g <= 0 or global_n <= 0:
        raise ValueError(f"scale dimensions must be positive, got {tuple(scale_gn.shape)}")
    if partition == 1:
        if global_n % num_cores != 0:
            raise ValueError(
                f"pgrp col scale needs N%cores==0, got N={global_n},cores={num_cores}"
            )
        local_g, local_n = global_g, global_n // num_cores
        local_scale = scale_gn.reshape(global_g, num_cores, local_n).permute(1, 0, 2)
    elif partition == 0:
        if global_g % num_cores != 0:
            raise ValueError(
                f"pgrp row scale needs G%cores==0, got G={global_g},cores={num_cores}"
            )
        local_g, local_n = global_g // num_cores, global_n
        local_scale = scale_gn.reshape(num_cores, local_g, local_n)
    else:
        raise ValueError(f"partition must be 0(row) or 1(col), got {partition}")

    group_blocks = (local_g + 3) // 4
    n_blocks = (local_n + 63) // 64
    padded = scale_gn.new_zeros((num_cores, group_blocks * 4, n_blocks * 64))
    padded[:, :local_g, :local_n] = local_scale
    physical = (
        padded.reshape(num_cores, group_blocks, 4, n_blocks, 64)
        .permute(1, 3, 0, 2, 4)
        .contiguous()
    )
    return physical.reshape(group_size, -1)


def _int4_pack_perm() -> np.ndarray:
    """Deinterleave permutation required by each 64-value K group:
    [0,16, 1,17, ..., 15,31, 32,48, ..., 47,63]."""
    perm = np.empty(64, dtype=np.int64)
    for blk in range(2):
        for i in range(16):
            perm[blk * 32 + 2 * i] = blk * 32 + i
            perm[blk * 32 + 2 * i + 1] = blk * 32 + 16 + i
    return perm


def _pgrp_swizzle_perm(vals: torch.Tensor, partition: int, num_cores: int) -> torch.Tensor:
    """int4 vals [N,K] -> per-core 16Nx64K swizzle + deinterleave perm -> flat int tensor."""
    N, K = vals.shape
    e = _NUM_ELE_32B_INT4
    v = vals.to(torch.int64)
    if partition == 1:  # col
        if N % (16 * num_cores) != 0 or K % e != 0:
            raise ValueError(
                f"pgrp col swizzle needs N%(16*cores)==0 and K%64==0, got N={N},K={K},cores={num_cores}")
        v = v.view(num_cores, N // 16 // num_cores, 16, K // e, e).permute(1, 3, 0, 2, 4).contiguous()
    elif partition == 0:  # row
        if N % 16 != 0 or K % (e * num_cores) != 0:
            raise ValueError(
                f"pgrp row swizzle needs N%16==0 and K%(64*cores)==0, got N={N},K={K},cores={num_cores}")
        v = v.view(N // 16, 16, num_cores, K // e // num_cores, e).permute(0, 3, 2, 1, 4).contiguous()
    else:
        raise ValueError(f"partition must be 0(row) or 1(col), got {partition}")
    perm = torch.as_tensor(_int4_pack_perm())
    v = v[..., perm].contiguous()       # deinterleave the innermost 64-value K-group
    return v.reshape(-1)


def swizzle_pack_int4_pgrp(int4_vals: torch.Tensor, partition: int, num_cores: int) -> torch.Tensor:
    """int4 vals [N,K] (int8 container) -> packed uint8 torch tensor [N*K/2]."""
    flat = _pgrp_swizzle_perm(int4_vals, partition, num_cores)
    packed = _pack_int4_signed(flat.cpu().numpy())
    return torch.from_numpy(packed).contiguous()
