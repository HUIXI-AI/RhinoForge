from __future__ import annotations

from types import SimpleNamespace

import torch

from rpu_backend.quant._common import (
    dequantize_linear_per_channel,
    quantize_linear_per_channel,
)
from rpu_backend.quant.load import is_w8a16_config


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


def test_w8a16_config_marker() -> None:
    assert is_w8a16_config(
        SimpleNamespace(quant_config={"method": "w8a16"})
    )
    assert not is_w8a16_config(
        SimpleNamespace(quant_config={"method": "other"})
    )
