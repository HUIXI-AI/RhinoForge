"""Source-neutral UINT4 codecs for expert-major grouped linear kernels.

Logical codes are [N,K], packed checkpoints [K/8,N] INT32 and runtime
weights [E,physical_expert_bytes] UINT8 in N32K32 order. Canonical FP16 metadata is [E,Q,N]; the current
runtime ABI is q4_n64_controller_v1: [E,Q4,N64,core,4,64], zero padded.
Signed and unsigned codes use the same physical nibble layout.
"""
from __future__ import annotations

import numpy as np
import torch

from .fp8 import _validate_grouped_experts


def pack_uint4_qweight(codes: torch.Tensor) -> torch.Tensor:
    """Logical UINT4 [N,K] -> GPTQ-v1 INT32 [K/8,N]."""
    _check_codes(codes)
    n, k = codes.shape
    if k % 8:
        raise ValueError("UINT4 qweight requires K divisible by 8")
    source = codes.numpy().astype(np.uint32)
    result = np.zeros((n, k // 8), dtype=np.uint32)
    for offset in range(8):
        result |= source[:, offset::8] << (4 * offset)
    return torch.from_numpy(result.T.copy().view(np.int32))


def _unpack_group128_matrix(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
    g_idx: torch.Tensor, *, n: int, k: int, zero_word: int, source_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate a source-bound G128 encoding; never infer its zero convention."""
    if n <= 0 or k <= 0 or k % 128 or n % 8:
        raise ValueError(f"{source_name} matrix requires positive N%8==0 and K%128==0")
    for name, tensor, dtype, shape in (
        ("qweight", qweight, torch.int32, (k // 8, n)),
        ("scales", scales, torch.float16, (k // 128, n)),
        ("qzeros", qzeros, torch.int32, (k // 128, n // 8)),
        ("g_idx", g_idx, torch.int32, (k,)),
    ):
        if (tensor.device.type != "cpu" or tensor.dtype != dtype
                or tuple(tensor.shape) != shape or not tensor.is_contiguous()):
            raise ValueError(f"{source_name} {name} requires contiguous CPU {dtype} {shape}")
    if not bool((qzeros == zero_word).all()):
        raise ValueError(f"{source_name} qzeros must encode "
                         + ("zero-minus-one=7 (effective zero 8)" if zero_word == 0x77777777
                            else "direct zero 8 (0x88888888)"))
    if not torch.equal(g_idx, torch.arange(k, dtype=torch.int32) // 128):
        raise ValueError(f"{source_name} g_idx must be contiguous group128 without act-order")
    if not bool(torch.isfinite(scales).all()) or not bool((scales >= 0).all()):
        raise ValueError(f"{source_name} scales must be finite and nonnegative; zero is preserved")
    source = qweight.numpy().view(np.uint32)
    codes = np.empty((k, n), dtype=np.uint8)
    for offset in range(8):
        codes[offset::8] = (source >> (4 * offset)) & 15
    return torch.from_numpy(codes.T.copy()), scales


def unpack_gptq_sym8_matrix(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
    g_idx: torch.Tensor, *, n: int, k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the admitted official symmetric GPTQ direct-zero-8 snapshot.

    No RTN quantization or legacy GPTQ zero-plus-one conversion is performed.
    """
    return _unpack_group128_matrix(
        qweight, scales, qzeros, g_idx, n=n, k=k,
        zero_word=-2004318072, source_name="GPTQ")


def _check_codes(codes: torch.Tensor) -> None:
    if (codes.device.type != "cpu" or codes.dim() != 2
            or codes.dtype != torch.uint8 or codes.numel() == 0):
        raise ValueError("UINT4 codes must be a nonempty CPU uint8 [N,K] matrix")
    if int(codes.max()) > 15:
        raise ValueError("UINT4 codes must lie in [0,15]")


def pack_grouped_uint4(
    codes: torch.Tensor, *, partition: str, num_cores: int = 8,
) -> torch.Tensor:
    """[E,N,K] UINT4 -> current BASE N32K32 expert-major bytes.

    Each core's N extent is padded independently to 32 by the shared codec.
    Checkpoint words and controller-striped S/Z metadata remain unchanged.
    """
    from .grouped_uint4_n32k32 import pack_grouped_uint4_n32k32

    if (not isinstance(codes, torch.Tensor) or codes.dim() != 3):
        raise ValueError("grouped UINT4 requires nonempty CPU uint8 [E,N,K]")
    _validate_grouped_experts(codes.shape[0])
    return pack_grouped_uint4_n32k32(codes, partition=partition, num_cores=num_cores)


def runtime_group_scales(
    scales: torch.Tensor, *, k: int, partition: str, num_cores: int = 8,
) -> tuple[torch.Tensor, int]:
    """Refine logical G128 before runtime swizzle, preserving FP16 bits.

    Returns canonical [Qglobal,N] metadata, not the controller-striped ABI.
    In particular, TP8 Down K512 duplicates each G128 scale into two G64
    groups. pack_grouped_uint4_metadata subsequently pads local Q to four.
    """
    if (scales.device.type != "cpu" or scales.dtype != torch.float16
            or scales.dim() != 2 or k <= 0 or k % 128
            or scales.shape[0] != k // 128):
        raise ValueError("UINT4 scales must be CPU FP16 [K/128,N]")
    if type(num_cores) is not int or not 1 <= num_cores <= 8:
        raise ValueError("invalid num_cores")
    if partition not in ("col", "row") or (partition == "row" and k % num_cores):
        raise ValueError("invalid UINT4 partition or nondivisible K")
    local_k = k if partition == "col" else k // num_cores
    group_size = min(128, local_k)
    if group_size not in (32, 64, 128) or local_k % group_size:
        raise ValueError("UINT4 localK cannot represent supported runtime groups")
    return scales.repeat_interleave(128 // group_size, dim=0).contiguous(), group_size


def pack_grouped_uint4_metadata(
    metadata: torch.Tensor, *, partition: str, num_cores: int = 8,
) -> torch.Tensor:
    """Canonical FP16 [E,Qglobal,Nglobal] -> q4_n64_controller_v1.

    Match the unified GroupedParallelLinearGenerator metadata convention.
    TP col partitions N; TP row partitions Q after any logical group-size
    refinement. Each core contributes [4,64] FP16 values (512 bytes) per
    controller stripe. Local Q and N tails are padded with positive zero.

    Apply this same byte permutation to scales and negative zero points.
    No quantization, arithmetic, or source zero convention is inferred here.
    Native retention must align the actual DDR base to num_cores*512 bytes;
    a CPU tensor's data_ptr does not establish device address alignment.
    """
    if (metadata.device.type != "cpu" or metadata.dtype != torch.float16
            or metadata.dim() != 3 or metadata.numel() == 0):
        raise ValueError("grouped UINT4 metadata requires nonempty CPU FP16 [E,Q,N]")
    if type(num_cores) is not int or not 1 <= num_cores <= 8:
        raise ValueError("grouped UINT4 metadata requires integer num_cores in [1,8]")
    experts, global_q, global_n = metadata.shape
    _validate_grouped_experts(experts)
    if partition == "col":
        if global_n % (16 * num_cores):
            raise ValueError("UINT4 col metadata requires N%(16*cores)==0")
        local_q, local_n = global_q, global_n // num_cores
        local = metadata.reshape(experts, global_q, num_cores, local_n).permute(0, 2, 1, 3)
    elif partition == "row":
        if global_q % num_cores or global_n % 16:
            raise ValueError("UINT4 row metadata requires Q%cores==0 and N%16==0")
        local_q, local_n = global_q // num_cores, global_n
        local = metadata.reshape(experts, num_cores, local_q, local_n)
    else:
        raise ValueError("partition must be col or row")
    q_blocks, n_blocks = (local_q + 3) // 4, (local_n + 63) // 64
    padded = torch.zeros((experts, num_cores, q_blocks * 4, n_blocks * 64), dtype=torch.float16)
    padded[:, :, :local_q, :local_n].copy_(local)
    return padded.view(experts, num_cores, q_blocks, 4, n_blocks, 64).permute(
        0, 2, 4, 1, 3, 5).contiguous()
