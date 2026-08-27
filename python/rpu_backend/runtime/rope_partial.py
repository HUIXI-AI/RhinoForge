"""Host precompute of M-RoPE cos/sin for the `partial_mrope` kernel.

The `partial_mrope` kernel (vs `llama_mrope_interleave`) takes the per-token
cos/sin `[seq, rotary_dim/2]` directly, instead of a static 1D table +
position_ids + an in-kernel axis gather. The host must select the layout that
matches the model family:

* Qwen3-VL uses the THW-interleaved helper below.
* Qwen2.5-VL uses contiguous T/H/W frequency chunks.

Both helpers build in fp64 and cast the final tables to fp16.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch


def _validate_mrope_inputs(
    position_ids: torch.Tensor,
    *,
    head_dim: object,
    rope_theta: object,
    mrope_section: object,
    rotary_dim: object | None,
) -> tuple[int, int, float, tuple[int, int, int]]:
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError(
            "position_ids must be a torch.Tensor, got "
            f"{type(position_ids).__name__}"
        )
    if position_ids.device.type != "cpu":
        raise ValueError(
            f"position_ids must be on CPU, got {position_ids.device}"
        )
    if position_ids.dim() != 2 or position_ids.size(1) != 3:
        raise ValueError(
            "position_ids must have shape [seq, 3], got "
            f"{tuple(position_ids.shape)}"
        )
    if position_ids.size(0) == 0:
        raise ValueError("position_ids must contain at least one token")
    if (
        position_ids.dtype == torch.bool
        or position_ids.is_floating_point()
        or position_ids.is_complex()
    ):
        raise TypeError(
            "position_ids must use an integer dtype, got "
            f"{position_ids.dtype}"
        )
    if bool((position_ids < 0).any()):
        raise ValueError("position_ids must be non-negative")

    if isinstance(head_dim, bool) or not isinstance(head_dim, Integral):
        raise TypeError(f"head_dim must be an integer, got {head_dim!r}")
    head_dim = int(head_dim)
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(
            f"head_dim must be a positive even integer, got {head_dim}"
        )

    if rotary_dim is None:
        rotary_dim = head_dim
    elif isinstance(rotary_dim, bool) or not isinstance(rotary_dim, Integral):
        raise TypeError(f"rotary_dim must be an integer, got {rotary_dim!r}")
    rotary_dim = int(rotary_dim)
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be a positive even integer, got {rotary_dim}"
        )
    if rotary_dim > head_dim:
        raise ValueError(
            f"rotary_dim {rotary_dim} must be <= head_dim {head_dim}"
        )

    if isinstance(rope_theta, bool) or not isinstance(rope_theta, Real):
        raise TypeError(f"rope_theta must be a real number, got {rope_theta!r}")
    theta = float(rope_theta)
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError(
            f"rope_theta must be positive and finite, got {rope_theta!r}"
        )

    if not isinstance(mrope_section, (list, tuple)) or len(mrope_section) != 3:
        raise ValueError(
            "mrope_section must contain exactly three integer sections"
        )
    sections = []
    for index, value in enumerate(mrope_section):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"mrope_section[{index}] must be an integer, got {value!r}"
            )
        parsed = int(value)
        if parsed < 0:
            raise ValueError(
                f"mrope_section[{index}] must be non-negative, got {parsed}"
            )
        sections.append(parsed)
    half = rotary_dim // 2
    if sum(sections) != half:
        raise ValueError(
            f"mrope_section must sum to rotary_dim/2 ({half}), got "
            f"{tuple(sections)}"
        )
    return rotary_dim, half, theta, tuple(sections)


def _validate_tables(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    seq_len: int,
    half: int,
    builder: str,
) -> None:
    expected_shape = (seq_len, half)
    if tuple(cos.shape) != expected_shape or tuple(sin.shape) != expected_shape:
        raise RuntimeError(
            f"{builder} produced invalid table shapes: "
            f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, "
            f"expected={expected_shape}"
        )


def build_interleaved_mrope_cos_sin(
    position_ids: torch.Tensor,        # [seq, 3] int (T/H/W coordinates)
    *,
    head_dim: int,
    rope_theta: float,
    mrope_section: list[int],          # [T_pairs, H_pairs, W_pairs]
    rotary_dim: int | None = None,     # None -> head_dim (partial_rotary_factor = 1.0)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Qwen3-VL interleaved cos/sin, fp16, on CPU.

    This layout must match the public M-RoPE operator contract:

    * T is the baseline on every lane.
    * H overrides lanes ``1 + 3i``.
    * W overrides lanes ``2 + 3i``.
    """
    rotary_dim, half, rope_theta, sections = _validate_mrope_inputs(
        position_ids,
        head_dim=head_dim,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        rotary_dim=rotary_dim,
    )

    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )  # [half]
    pos = position_ids.to(torch.float64).transpose(0, 1)        # [3, seq]
    freqs = pos[:, :, None] * inv_freq[None, None, :]           # [3, seq, half]

    merged = freqs[0].clone()                                   # T baseline [seq, half]
    for dim_idx, offset in ((1, 1), (2, 2)):                    # H=1, W=2
        idx = slice(offset, sections[dim_idx] * 3, 3)           # clips at `half`
        merged[:, idx] = freqs[dim_idx][:, idx]

    cos = merged.cos().to(torch.float16).contiguous()
    sin = merged.sin().to(torch.float16).contiguous()
    _validate_tables(
        cos,
        sin,
        seq_len=position_ids.size(0),
        half=half,
        builder="build_interleaved_mrope_cos_sin",
    )
    return cos, sin


def build_chunked_mrope_cos_sin(
    position_ids: torch.Tensor,        # [seq, 3] int (T/H/W coordinates)
    *,
    head_dim: int,
    rope_theta: float,
    mrope_section: list[int],          # [T_pairs, H_pairs, W_pairs]
    rotary_dim: int | None = None,     # None -> head_dim (partial_rotary_factor = 1.0)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Qwen2.5-VL chunked M-RoPE (cos, sin), fp16, on CPU.

    Qwen2.5-VL assigns contiguous rotary-frequency ranges to T/H/W:
    ``[T * section[0], H * section[1], W * section[2]]``. This differs from
    Qwen3-VL's THW-interleaved layout implemented by
    :func:`build_interleaved_mrope_cos_sin`.
    """
    rotary_dim, half, rope_theta, sections = _validate_mrope_inputs(
        position_ids,
        head_dim=head_dim,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        rotary_dim=rotary_dim,
    )

    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float64) / rotary_dim)
    )
    pos = position_ids.to(torch.float64).transpose(0, 1)
    freqs = pos[:, :, None] * inv_freq[None, None, :]

    t_end = sections[0]
    h_end = t_end + sections[1]
    merged = torch.cat(
        [
            freqs[0, :, :t_end],
            freqs[1, :, t_end:h_end],
            freqs[2, :, h_end:half],
        ],
        dim=-1,
    )
    cos = merged.cos().to(torch.float16).contiguous()
    sin = merged.sin().to(torch.float16).contiguous()
    _validate_tables(
        cos,
        sin,
        seq_len=position_ids.size(0),
        half=half,
        builder="build_chunked_mrope_cos_sin",
    )
    return cos, sin
