"""N32K32 weight packing for the unified parallel_linear W4 ABI.

Signed and unsigned codes share the physical order
``[Nblock32,Kblock32,core,thread16,Nhalf2,byte16]``. Each byte pairs K=j
and K=j+16; each aligned 32-byte thread load pairs N=t and N=t+16.
Output-channel tails are zero padded within each TP shard.
Logical scale is (G=K/group_size, N) fp16 row-major.  The kernel consumes a
controller-striped [ceil(localG/4), ceil(localN/64), cores, 4, 64] payload;
swizzle_int4_pgrp_scale builds that payload and reshapes it to a 2-D tensor
whose first dimension carries group_size metadata for the C++ dispatcher.

Per-channel quantization remains numerically representable: repeat its
one scale per output channel across all K groups, then use the same pgrp weight
and scale packing.  There is only one physical INT4 ABI.

The packed byte count can exceed N*K/2 when localN is not a multiple of 32.
Callers must retain the complete physical owner and keep logical dimensions
separate from its storage dimensions.
"""
from __future__ import annotations

import torch

_FP16_TINY = torch.finfo(torch.float16).tiny
W4_WEIGHT_LAYOUT = "n32k32"


def quantize_int4_group_wise(
    weight: torch.Tensor, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """[N,K] float -> (int4 vals int8-container [N,K], fp16 scale [G,N], group_size).

    Symmetric per (output-channel, K-group) block, qmax=7 / qmin=-8 (matching the
    per-channel scheme). Self-consistent: dequant(int4, scale) is exactly what the
    kernel computes (scale[g,n] * int4[n,k]), so the CPU golden uses the same.
    """
    if weight.dim() != 2:
        raise ValueError(f"expected 2-D Linear weight, got {tuple(weight.shape)}")
    if type(group_size) is not int or group_size not in (32, 64, 128):
        raise ValueError("group_size must be 32, 64, or 128")
    if not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("group INT4 requires finite floating-point weights")
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
    """Inverse of quantize_int4_group_wise -> fp16 [N, K] (for the CPU golden)."""
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


def pad_int4_mlp_stripe(
    int4_vals: torch.Tensor, scale_gn: torch.Tensor, *, partition: int,
    padded_size: int, group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad logical INT4 values along the MLP width before the pgrp pack.

    Col partition (Gate/Up) appends output rows N; row partition (Down)
    appends whole K groups. New values are exact zero and their logical
    scales are one, so padded Gate/Up rows produce zero and padded Down
    columns multiply those zero activations: the physical MLP equals the
    logical one. Existing values and scales are copied unchanged.
    """
    if int4_vals.dim() != 2 or scale_gn.dim() != 2:
        raise ValueError("MLP INT4 padding requires values [N,K] and scale [G,N]")
    n, k = (int(dim) for dim in int4_vals.shape)
    if partition not in (0, 1) or type(padded_size) is not int:
        raise ValueError("MLP INT4 padding requires partition 0/1 and an integer width")
    if group_size <= 0 or k % group_size or tuple(scale_gn.shape) != (k // group_size, n):
        raise ValueError(f"MLP INT4 padding requires scale [K/{group_size},N] for {(n, k)}")
    logical = n if partition == 1 else k
    if padded_size < logical or (partition == 0 and padded_size % group_size):
        raise ValueError(f"MLP INT4 padding cannot shrink or split groups: {logical}->{padded_size}")
    if padded_size == logical:
        return int4_vals, scale_gn
    shape = (padded_size, k) if partition == 1 else (n, padded_size)
    values = int4_vals.new_zeros(shape)
    values[:n, :k] = int4_vals
    groups = k // group_size if partition == 1 else padded_size // group_size
    scales = scale_gn.new_ones((groups, padded_size if partition == 1 else n))
    scales[:k // group_size, :n] = scale_gn
    return values, scales


def w4_storage_shape(n: int, k: int, partition: int, num_cores: int) -> tuple[int, int]:
    """Return physical ``[padded global N, global K/2]`` storage dimensions.

    Padding follows TP partitioning. K is never silently padded: a row shard's
    input and its scale groups must already describe the same K32 geometry.
    """
    if type(num_cores) is not int or not 1 <= num_cores <= 8:
        raise ValueError("W4 packing requires an integer core count in [1,8]")
    if any(type(v) is not int or v <= 0 for v in (n, k)):
        raise ValueError("W4 dimensions must be positive integers")
    if partition == 1:
        if n % num_cores:
            raise ValueError("W4 col packing requires N%cores==0")
        local_n, local_k = n // num_cores, k
    elif partition == 0:
        if k % num_cores:
            raise ValueError("W4 row packing requires K%cores==0")
        local_n, local_k = n, k // num_cores
    else:
        raise ValueError("partition must be 0(row) or 1(col)")
    if local_n % 16 or local_k % 32:
        raise ValueError("N32K32 packing requires localN%16==0 and localK%32==0")
    physical_n = ((local_n + 31) // 32) * 32
    return physical_n * (num_cores if partition == 1 else 1), k // 2


def swizzle_pack_w4_n32k32(
    values: torch.Tensor, partition: int, num_cores: int,
) -> torch.Tensor:
    """Pack signed INT8 or unsigned UINT8 logical codes into flat N32K32 bytes."""
    if (not isinstance(values, torch.Tensor) or values.dim() != 2
            or values.dtype not in (torch.int8, torch.uint8) or values.numel() == 0):
        raise ValueError("W4 codes require a nonempty INT8 or UINT8 [N,K] tensor")
    n, k = values.shape
    physical_n, _ = w4_storage_shape(n, k, partition, num_cores)
    low, high = (-8, 7) if values.dtype == torch.int8 else (0, 15)
    if int(values.min()) < low or int(values.max()) > high:
        raise ValueError(f"W4 {values.dtype} codes must lie in [{low},{high}]")
    local_n, local_k = (n // num_cores, k) if partition == 1 else (n, k // num_cores)
    padded_n = physical_n // num_cores if partition == 1 else physical_n
    codes = values.detach().to(device="cpu", dtype=torch.uint8) & 15
    local = (codes.reshape(num_cores, local_n, local_k) if partition == 1 else
             codes.reshape(local_n, num_cores, local_k).permute(1, 0, 2))
    if padded_n != local_n:
        padded = codes.new_zeros((num_cores, padded_n, local_k))
        padded[:, :local_n].copy_(local)
        local = padded
    groups = local.reshape(num_cores, padded_n // 32, 2, 16, local_k // 32, 2, 16)
    packed = groups[..., 0, :] | (groups[..., 1, :] << 4)
    return packed.permute(1, 4, 0, 3, 2, 5).contiguous().reshape(-1)


def swizzle_pack_int4_pgrp(int4_vals: torch.Tensor, partition: int, num_cores: int) -> torch.Tensor:
    """Signed logical INT4 [N,K] -> the complete flat N32K32 UINT8 owner."""
    if not isinstance(int4_vals, torch.Tensor) or int4_vals.dtype != torch.int8:
        raise ValueError("signed INT4 codes require an INT8 tensor")
    return swizzle_pack_w4_n32k32(int4_vals, partition, num_cores)


def pack_int4_per_channel_as_pgrp(
    int4_vals: torch.Tensor,
    scale_n: torch.Tensor,
    partition: int,
    num_cores: int,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack logical per-channel INT4 through the sole pgrp physical ABI.

    ``scale_n[n]`` is repeated for every K group, preserving the original
    per-output-channel dequantization exactly.  Returns a flat packed weight and
    the controller-striped 2-D scale payload.
    """
    if int4_vals.dim() != 2:
        raise ValueError(
            f"expected int4 values [N,K], got {tuple(int4_vals.shape)}"
        )
    n, k = int4_vals.shape
    if scale_n.dtype != torch.float16 or scale_n.dim() != 1 or scale_n.numel() != n:
        raise ValueError(
            f"per-channel scale must be fp16 [N]=[{n}], got "
            f"dtype={scale_n.dtype}, shape={tuple(scale_n.shape)}"
        )
    if type(group_size) is not int or group_size not in (32, 64, 128):
        raise ValueError("group_size must be 32, 64, or 128")
    if k % group_size != 0:
        raise ValueError(f"K={k} not divisible by group_size={group_size}")
    logical_scale = (
        scale_n.reshape(1, n).expand(k // group_size, n).contiguous()
    )
    return (
        swizzle_pack_int4_pgrp(int4_vals, partition, num_cores),
        swizzle_int4_pgrp_scale(
            logical_scale, group_size, partition, num_cores
        ),
    )


def quantize_pack_int4_pgrp_col_bounded(
    weight: torch.Tensor, group_size: int = 32, *, num_cores: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a col-partitioned lm_head for the current pgrp W4 kernel ABI."""
    if weight.dim() != 2:
        raise ValueError(
            f"W4 lm_head weight must be 2-D, got {tuple(weight.shape)}"
        )
    vocab, hidden = (int(dim) for dim in weight.shape)
    if group_size != 32:
        raise ValueError(
            "W4 col packing is validated only for group_size=32, "
            f"got {group_size}"
        )
    if type(num_cores) is not int or not 1 <= num_cores <= 8:
        raise ValueError("W4 col packing requires an integer core count in [1,8]")
    storage_shape = w4_storage_shape(vocab, hidden, 1, num_cores)
    # Bound the quantizer's fp32 and packer's uint8 temporaries. A block
    # contains the same local output rows from every core, so its packed
    # bytes occupy one contiguous interval of the original physical layout.
    local_rows = vocab // num_cores
    packed = torch.empty(storage_shape[0] * storage_shape[1], dtype=torch.uint8)
    logical_scale = torch.empty(hidden // group_size, vocab, dtype=torch.float16)
    core_weight = weight.detach().reshape(num_cores, local_rows, hidden)
    for start in range(0, local_rows, 256):
        stop = min(start + 256, local_rows)
        block = core_weight[:, start:stop].reshape(-1, hidden).to(torch.float16)
        values, block_scale, _ = quantize_int4_group_wise(block, group_size)
        physical_stop = (stop + 31) // 32 * 32
        packed[start * hidden * num_cores // 2:physical_stop * hidden * num_cores // 2] = swizzle_pack_int4_pgrp(
            values, partition=1, num_cores=num_cores,
        )
        logical_scale.reshape(hidden // group_size, num_cores, local_rows)[:, :, start:stop] = (
            block_scale.reshape(hidden // group_size, num_cores, stop - start)
        )
    packed = packed.reshape(storage_shape)
    scale = swizzle_int4_pgrp_scale(
        logical_scale, group_size, partition=1, num_cores=num_cores,
    )
    return packed.contiguous(), scale.contiguous()
