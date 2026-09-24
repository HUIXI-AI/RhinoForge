from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.quant._common import (
    dequantize_linear_per_channel,
    quantize_linear_per_channel,
)
from rpu_backend.quant.load import is_w8a16_config
from rpu_backend.quant.convert_pi05 import (
    PI05_W4_GROUP_SIZE,
    _convert_tensor as convert_pi05_tensor,
    _quant_config as pi05_quant_config,
)
from rpu_backend.quant.int4_pgrp_pack import (
    swizzle_int4_pgrp_scale,
    swizzle_pack_int4_pgrp,
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


def test_w8a16_config_marker() -> None:
    assert is_w8a16_config(
        SimpleNamespace(quant_config={"method": "w8a16"})
    )
    assert not is_w8a16_config(
        SimpleNamespace(quant_config={"method": "other"})
    )


def test_pi05_mixed_w4a16_g32_kv8_contract_and_runtime_pack(tmp_path) -> None:
    from rpu_backend.adapters.pi05.loader import (
        _install_pi05_w8a16_tensors,
        _load_pi05_rpu_quant_config,
        _pop_pi05_w8a16_tensors,
    )
    from rpu_backend.adapters.pi05.w4pack import pack_int4_projections_inplace

    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.paligemma_with_expert = torch.nn.Module()
    pawe = policy.model.paligemma_with_expert
    pawe.paligemma = torch.nn.Module()
    pawe.paligemma.model = torch.nn.Module()
    decoder = torch.nn.Module()
    pawe.paligemma.model.language_model = decoder
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.q_proj = torch.nn.Linear(64, 128, bias=False)
    layer.self_attn.k_proj = torch.nn.Linear(64, 128, bias=False)
    layer.mlp = torch.nn.Module()
    layer.mlp.down_proj = torch.nn.Linear(512, 16, bias=False)
    decoder.layers = torch.nn.ModuleList([layer])

    q_name = (
        "model.paligemma_with_expert.paligemma.model.language_model.layers.0."
        "self_attn.q_proj.weight"
    )
    k_name = q_name.replace("q_proj", "k_proj")
    down_name = q_name.replace("self_attn.q_proj", "mlp.down_proj")
    raw_q_name = q_name.removeprefix("model.")
    raw, n_quantized, _, _ = convert_pi05_tensor(
        raw_q_name, torch.ones(128, 64), bits=8
    )
    assert n_quantized == 1 and raw_q_name in raw

    config = pi05_quant_config(bits=4, real_w4=True)
    assert config["method"] == "w4a16"
    assert config["mode"] == "group_wise_symmetric"
    assert config["mixed_int8_suffixes"] == ["k_proj.weight", "v_proj.weight"]
    assert config["group_size"] == 32
    assert "mixed W4A16-G32-KV8" in config["note"]
    assert "Linear projection weights are quantized" in config["note"]
    assert "Activations and KV cache stay FP16" in config["note"]
    config_path = tmp_path / "rpu_quant_config.json"
    config_path.write_text(json.dumps(config))
    assert _load_pi05_rpu_quant_config(str(tmp_path))["group_size"] == 32
    config_path.write_text(json.dumps({**config, "mixed_int8_suffixes": ["k_proj"]}))
    from rpu_backend.api.errors import RPUBackendError
    with pytest.raises(RPUBackendError, match="k_proj and v_proj"):
        _load_pi05_rpu_quant_config(str(tmp_path))

    converted = {}
    for name, weight in (
        (q_name, torch.linspace(-2, 2, 128 * 64).reshape(128, 64)),
        (k_name, torch.linspace(-1, 1, 128 * 64).reshape(128, 64)),
        (down_name, torch.linspace(-3, 3, 16 * 512).reshape(16, 512)),
    ):
        tensors, _, _, _ = convert_pi05_tensor(
            name,
            weight,
            bits=4,
            int8_keep_suffixes=tuple(config["mixed_int8_suffixes"]),
            group_size=PI05_W4_GROUP_SIZE,
        )
        converted.update(tensors)

    q_scale_name = q_name.replace(".weight", ".weight_scale")
    k_scale_name = k_name.replace(".weight", ".weight_scale")
    down_scale_name = down_name.replace(".weight", ".weight_scale")
    q_scale = converted[q_scale_name].clone()
    q_values = converted[q_name].clone()
    down_scale = converted[down_scale_name].clone()
    down_values = converted[down_name].clone()
    assert q_scale.shape == (2, 128)
    assert converted[k_scale_name].shape == (128,)
    assert down_scale.shape == (16, 16)

    malformed = dict(converted)
    malformed[k_scale_name] = converted[k_scale_name].repeat(2, 1)
    with pytest.raises(RPUBackendError, match=r"weight_scale must be torch\.float16 \(128,\)"):
        _pop_pi05_w8a16_tensors(policy, malformed, config)
    malformed = dict(converted)
    malformed[q_scale_name] = q_scale[:1]
    with pytest.raises(RPUBackendError, match=r"weight_scale must be torch\.float16 \(2, 128\)"):
        _pop_pi05_w8a16_tensors(policy, malformed, config)

    consumed = _pop_pi05_w8a16_tensors(policy, converted, config)
    _install_pi05_w8a16_tensors(policy, consumed)
    assert layer.self_attn.q_proj.weight_scale.shape == (2, 128)
    assert layer.self_attn.k_proj.weight_scale.shape == (128,)

    assert pack_int4_projections_inplace(
        decoder, {"q_proj", "down_proj"}, attn_num_cores=8
    ) == 2
    assert torch.equal(
        layer.self_attn.q_proj.weight.cpu(),
        swizzle_pack_int4_pgrp(q_values, 1, 8).reshape(128, 32),
    )
    assert torch.equal(
        layer.self_attn.q_proj.weight_scale.cpu(),
        swizzle_int4_pgrp_scale(q_scale, 32, 1, 8),
    )
    assert torch.equal(
        layer.mlp.down_proj.weight.cpu(),
        swizzle_pack_int4_pgrp(down_values, 0, 8).reshape(16, 256),
    )
    assert torch.equal(
        layer.mlp.down_proj.weight_scale.cpu(),
        swizzle_int4_pgrp_scale(down_scale, 32, 0, 8),
    )
