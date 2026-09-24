"""Qwen3.5-MoE routed/shared weight preparation.

The grouped kernel consumes one TP-swizzled DDR tensor per projection.  Raw
Hugging Face experts are stored as ``gate_up_proj[E,2I,H]`` and
``down_proj[E,H,I]``; gate/up must be split before either quantization or
swizzle.  FP8 is the Rhino grouped ABI (raw E4M3 codes plus FP16 per-output
scale), not the official 128x128 block-scale checkpoint format.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rpu_backend.runtime.weights import transform_linear_weight
from rpu_backend.quant.fp8 import (
    dequantize_fp8_e4m3_per_output,
    pack_grouped_weight,
    quantize_fp8_e4m3_per_output,
)


MODE_UINT8 = 0
MODE_INT8 = 1
MODE_FP16 = 2
MODE_FP8_E2M5 = 4
MODE_FP8_E4M3 = 5
TP = 8
SHARED_SCALAR_LOCAL_N = 16


@dataclass(frozen=True)
class PackedMoeLayerWeights:
    router: torch.Tensor
    routed_gate: torch.Tensor
    routed_up: torch.Tensor
    routed_down: torch.Tensor
    routed_gate_scale: torch.Tensor
    routed_up_scale: torch.Tensor
    routed_down_scale: torch.Tensor
    shared_gate: torch.Tensor
    shared_up: torch.Tensor
    shared_down: torch.Tensor
    shared_scalar_gate: torch.Tensor
    routed_weight_mode: int


def _cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous()


def _source_scale(experts, name: str) -> torch.Tensor:
    scale = getattr(experts, name, None)
    if scale is None:
        raise ValueError(
            f"quantized Qwen3.5-MoE experts are missing buffer {name!r}"
        )
    if scale.dtype != torch.float16:
        raise TypeError(f"{name} must be FP16, got {scale.dtype}")
    scale = _cpu_contiguous(scale)
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise ValueError(f"{name} must contain only finite positive scales")
    return scale


def _prepare_routed(experts, num_cores: int):
    if type(num_cores) is not int or num_cores not in (4, 6, 8):
        raise ValueError(
            f"Qwen3.5-35B-A3B routed experts require TP4, TP6 or TP8, got {num_cores}"
        )
    gate_up = _cpu_contiguous(experts.gate_up_proj)
    down = _cpu_contiguous(experts.down_proj)
    if gate_up.dim() != 3 or down.dim() != 3:
        raise ValueError("Qwen3.5-MoE expert weights must be rank-3")
    e, two_i, h = map(int, gate_up.shape)
    if two_i % 2:
        raise ValueError(f"gate_up output width must be even, got {two_i}")
    i = two_i // 2
    if tuple(down.shape) != (e, h, i):
        raise ValueError(
            f"down_proj shape {tuple(down.shape)} must equal {(e, h, i)}"
        )
    gate, up = gate_up[:, :i, :], gate_up[:, i:, :]

    if gate_up.dtype != torch.uint8:
        raise TypeError(
            "Qwen3.5-35B-A3B routed experts require raw uint8 E4M3 mode-5 "
            f"storage, got {gate_up.dtype}"
        )
    markers = {
        "rpu_fp8_format": "fp8_e4m3",
        "rpu_weight_mode": MODE_FP8_E4M3,
        "rpu_fp8_scale_axis": -1,
    }
    mismatched = {
        name: getattr(experts, name, None)
        for name, expected in markers.items()
        if getattr(experts, name, None) != expected
    }
    if mismatched:
        raise ValueError(
            "routed raw uint8 tensors are missing the exact typed mode-5 "
            f"markers: {mismatched}"
        )
    mode = MODE_FP8_E4M3
    gate_up_scale = _source_scale(experts, "gate_up_proj_scale")
    down_scale = _source_scale(experts, "down_proj_scale")
    if tuple(gate_up_scale.shape) != (e, two_i):
        raise ValueError(
            f"gate_up_proj_scale shape {tuple(gate_up_scale.shape)} must be "
            f"{(e, two_i)}"
        )
    if tuple(down_scale.shape) != (e, h):
        raise ValueError(
            f"down_proj_scale shape {tuple(down_scale.shape)} must be {(e, h)}"
        )
    gate_scale = gate_up_scale[:, :i].contiguous()
    up_scale = gate_up_scale[:, i:].contiguous()
    if down.dtype != gate_up.dtype:
        raise TypeError(
            f"gate_up/down routed storage dtype mismatch: {gate_up.dtype} vs {down.dtype}"
        )
    if num_cores == 6:
        if i != 512:
            raise ValueError("Qwen3.5-35B-A3B TP6 requires logical expert width 512")
        # Split the logical gate/up halves first. All padding is cold and the
        # source codes remain unchanged; zero codes with unit scales describe
        # the 64 new channels exactly. Down pads its input axis only.
        gate = F.pad(gate, (0, 0, 0, 64))
        up = F.pad(up, (0, 0, 0, 64))
        down = F.pad(down, (0, 64))
        gate_scale = F.pad(gate_scale, (0, 64), value=1)
        up_scale = F.pad(up_scale, (0, 64), value=1)
    return (
        pack_grouped_weight(gate, partition="col", num_cores=num_cores),
        pack_grouped_weight(up, partition="col", num_cores=num_cores),
        pack_grouped_weight(down, partition="row", num_cores=num_cores),
        gate_scale, up_scale, down_scale, mode,
    )


def prepare_moe_layer_weights(
    mlp, *, num_cores: int = TP, device: str = "rpu"
) -> PackedMoeLayerWeights:
    """Prepare one official HF ``Qwen3_5MoeSparseMoeBlock`` for C++ binding."""
    if type(num_cores) is not int or num_cores not in (4, 6, 8):
        raise ValueError(f"Qwen3.5-35B-A3B requires TP4, TP6 or TP8, got {num_cores}")
    router_cores = 4 if num_cores == 6 else num_cores
    routed = _prepare_routed(mlp.experts, num_cores)
    routed_gate, routed_up, routed_down, gate_s, up_s, down_s, mode = routed

    fp16_weights = {
        "router": mlp.gate.weight,
        "shared_gate": mlp.shared_expert.gate_proj.weight,
        "shared_up": mlp.shared_expert.up_proj.weight,
        "shared_down": mlp.shared_expert.down_proj.weight,
        "shared_scalar": mlp.shared_expert_gate.weight,
    }
    wrong = {name: weight.dtype for name, weight in fp16_weights.items()
             if weight.dtype != torch.float16}
    if wrong:
        raise TypeError(
            "Qwen3.5-35B-A3B router/shared weights must remain FP16: "
            f"{wrong}"
        )
    router = transform_linear_weight(
        _cpu_contiguous(mlp.gate.weight), 1, router_cores
    )
    shared = mlp.shared_expert
    shared_gate_raw = _cpu_contiguous(shared.gate_proj.weight)
    shared_up_raw = _cpu_contiguous(shared.up_proj.weight)
    shared_down_raw = _cpu_contiguous(shared.down_proj.weight)
    if num_cores == 6:
        h = int(router.shape[1])
        if (tuple(shared_gate_raw.shape) != (512, h)
                or tuple(shared_up_raw.shape) != (512, h)
                or tuple(shared_down_raw.shape) != (h, 512)):
            raise ValueError("Qwen3.5-35B-A3B TP6 requires logical shared width 512")
        shared_gate_raw = F.pad(shared_gate_raw, (0, 0, 0, 64))
        shared_up_raw = F.pad(shared_up_raw, (0, 0, 0, 64))
        shared_down_raw = F.pad(shared_down_raw, (0, 64))
    shared_gate = transform_linear_weight(
        shared_gate_raw, 1, num_cores
    )
    shared_up = transform_linear_weight(
        shared_up_raw, 1, num_cores
    )
    shared_down = transform_linear_weight(
        shared_down_raw, 0, num_cores
    )
    scalar_raw = _cpu_contiguous(mlp.shared_expert_gate.weight)
    if tuple(scalar_raw.shape) != (1, router.shape[1]):
        raise ValueError(
            f"shared_expert_gate must be [1,H], got {tuple(scalar_raw.shape)}"
        )
    # Ordinary col-parallel Linear requires local_n to be 16-aligned.  Pad the
    # mathematical [1,H] row to [16*tp,H]; every row is identical, and the C++
    # graph slices lane 0 from each core's [T,16] output.
    shared_scalar = transform_linear_weight(
        scalar_raw.expand(
            SHARED_SCALAR_LOCAL_N * num_cores, -1
        ).contiguous(),
        1,
        num_cores,
    )

    def upload(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return tensor
        return tensor.contiguous().to(device)

    return PackedMoeLayerWeights(
        router=upload(router),
        routed_gate=upload(routed_gate),
        routed_up=upload(routed_up),
        routed_down=upload(routed_down),
        routed_gate_scale=upload(gate_s),
        routed_up_scale=upload(up_s),
        routed_down_scale=upload(down_s),
        shared_gate=upload(shared_gate),
        shared_up=upload(shared_up),
        shared_down=upload(shared_down),
        shared_scalar_gate=upload(shared_scalar),
        routed_weight_mode=mode,
    )


def build_route_tables(
    max_chunk: int, num_experts: int, top_k: int, *, device: str = "rpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the static tables consumed by the runtime-compatible shuffle."""
    if not (0 < max_chunk <= 512 and num_experts == 256 and top_k == 8):
        raise ValueError(
            "Qwen3.5-35B-A3B route tables require max_chunk in [1,512], "
            f"num_experts=256 and top_k=8; got {(max_chunk, num_experts, top_k)}"
        )
    expert_ids = torch.arange(num_experts, dtype=torch.float16).view(1, -1)
    expert_ids = expert_ids.expand(max_chunk, -1).contiguous()
    token_ids = torch.arange(max_chunk, dtype=torch.int16).view(-1, 1)
    token_ids = token_ids.expand(-1, top_k).contiguous()
    return expert_ids.to(device), token_ids.to(device)


__all__ = [
    "MODE_FP16", "MODE_FP8_E4M3", "PackedMoeLayerWeights",
    "build_route_tables", "dequantize_fp8_e4m3_per_output",
    "pack_grouped_weight", "prepare_moe_layer_weights",
    "quantize_fp8_e4m3_per_output",
]
