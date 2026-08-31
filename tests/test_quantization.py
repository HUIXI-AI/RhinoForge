from __future__ import annotations

from types import SimpleNamespace

import torch

from rpu_backend.quant._common import (
    dequantize_linear_per_channel,
    quantize_linear_per_channel,
    quantize_linear_per_channel_activation_aware,
)
from rpu_backend.quant.load import is_w8a16_config
from rpu_backend.quant.convert_pi05 import (
    _convert_tensor,
    _quant_config,
    _target_for_name,
)


def test_per_channel_quantization_metadata_and_values() -> None:
    weight = torch.tensor(
        [[-2.0, -0.5, 0.5, 2.0], [-0.25, 0.0, 0.25, 0.5]],
        dtype=torch.float32,
    )
    quantized, scale = quantize_linear_per_channel(weight)
    restored = dequantize_linear_per_channel(quantized, scale)

    assert quantized.dtype == torch.int8
    assert scale.dtype == torch.float16
    assert tuple(scale.shape) == (2,)
    assert int(quantized.min()) >= -128
    assert int(quantized.max()) <= 127
    assert torch.nn.functional.cosine_similarity(
        restored.float(), weight, dim=1
    ).min() > 0.999


def test_activation_aware_quantization_reduces_weighted_output_error() -> None:
    weight = torch.tensor([[10.0, 1.0, -1.0, 1.0]], dtype=torch.float16)
    second_moment = torch.tensor([0.0001, 1.0, 1.0, 1.0])

    plain_q, plain_scale = quantize_linear_per_channel(weight)
    aware_q, aware_scale = quantize_linear_per_channel_activation_aware(
        weight, second_moment
    )
    plain_error = (
        (weight.float() - plain_q.float() * plain_scale.float().unsqueeze(1)).square()
        * second_moment
    ).sum()
    aware_error = (
        (weight.float() - aware_q.float() * aware_scale.float().unsqueeze(1)).square()
        * second_moment
    ).sum()

    assert aware_scale.item() < plain_scale.item()
    assert aware_error < plain_error

    fallback_q, fallback_scale = quantize_linear_per_channel_activation_aware(
        weight, torch.zeros(weight.shape[1])
    )
    assert torch.equal(fallback_q, plain_q)
    assert torch.equal(fallback_scale, plain_scale)


def test_w8a16_config_marker() -> None:
    assert is_w8a16_config(
        SimpleNamespace(quant_config={"method": "w8a16"})
    )
    assert not is_w8a16_config(
        SimpleNamespace(quant_config={"method": "other"})
    )


def test_pi05_full_activation_aware_target_set() -> None:
    names = {
        "expert": (
            "model.paligemma_with_expert.gemma_expert.model.layers.0."
            "self_attn.q_proj.weight"
        ),
        "vlm": (
            "model.paligemma_with_expert.paligemma.model.language_model.layers.0."
            "mlp.down_proj.weight"
        ),
        "siglip": (
            "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model."
            "encoder.layers.0.mlp.fc1.weight"
        ),
        "adarms": (
            "model.paligemma_with_expert.gemma_expert.model.layers.0."
            "input_layernorm.dense.weight"
        ),
        "adarms_final": (
            "model.paligemma_with_expert.gemma_expert.model.norm.dense.weight"
        ),
    }

    assert _target_for_name(names["expert"]) == "expert"
    assert _target_for_name(names["vlm"]) == "vlm"
    assert _target_for_name(names["siglip"]) is None
    assert _target_for_name(
        names["siglip"], include_runtime_linears=True
    ) == "siglip"
    assert _target_for_name(
        names["adarms"], include_runtime_linears=True
    ) == "adarms"
    assert _target_for_name(
        names["adarms_final"], include_runtime_linears=True
    ) == "adarms"

    weight = torch.tensor([[10.0, 1.0, -1.0, 1.0]], dtype=torch.float16)
    converted, n_quant, _, _ = _convert_tensor(
        names["siglip"],
        weight,
        bits=8,
        input_second_moment=torch.tensor([0.0001, 1.0, 1.0, 1.0]),
        include_runtime_linears=True,
    )
    assert n_quant == 1
    assert converted[names["siglip"]].dtype == torch.int8
    assert converted[names["siglip"].replace(".weight", ".weight_scale")].dtype == torch.float16

    config = _quant_config(
        bits=8,
        activation_stats_sha256="0" * 64,
        include_runtime_linears=True,
    )
    assert config["target"] == "pi05_vlm_expert_siglip_adarms"
    assert config["runtime_quantized_groups"] == ["siglip", "adarms"]
