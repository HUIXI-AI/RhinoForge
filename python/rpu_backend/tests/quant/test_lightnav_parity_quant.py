from types import SimpleNamespace

import torch

from rpu_backend.quant.convert_qwen3_vl import (
    LIGHTNAV_ACTIVATION_PROFILE,
    LIGHTNAV_ACTIVATION_VISION_PROFILE,
    LIGHTNAV_PARITY_PROFILE,
    is_lightnav_parity_weight,
    lightnav_activation_quant_config,
    lightnav_activation_stat_name,
    lightnav_parity_quant_config,
    quantize_weight,
)


def test_lightnav_parity_scope_is_exactly_first_four_attention_o_projections():
    selected = {
        layer for layer in range(36)
        if is_lightnav_parity_weight(
            f"model.language_model.layers.{layer}.self_attn.o_proj.weight"
        )
    }
    assert selected == {0, 1, 2, 3}
    assert not is_lightnav_parity_weight(
        "model.language_model.layers.0.self_attn.q_proj.weight"
    )


def test_lightnav_parity_config_keeps_vision_and_head_fp16():
    config = lightnav_parity_quant_config()
    assert config["profile"] == LIGHTNAV_PARITY_PROFILE
    assert config["vision_quantization"] == "fp16"
    assert config["quantized_lm_head"] is False


def test_lightnav_activation_stat_names_cover_text_projection_roles():
    assert lightnav_activation_stat_name(
        "model.language_model.layers.17.mlp.down_proj.weight"
    ) == "layers.17.down"
    assert lightnav_activation_stat_name(
        "model.language_model.layers.3.self_attn.q_proj.weight"
    ) == "layers.3.q"


def test_activation_aware_w8_reduces_weighted_output_error():
    weight = torch.tensor([[100.0, 1.2, -1.1, 0.9]], dtype=torch.float16)
    moment = torch.tensor([1e-6, 1.0, 1.0, 1.0], dtype=torch.float32)
    base_q, base_scale = quantize_weight(weight)
    aware_q, aware_scale = quantize_weight(weight, input_second_moment=moment)
    base_error = (((base_q.float() * base_scale.float().unsqueeze(1) - weight.float()) ** 2) * moment).sum()
    aware_error = (((aware_q.float() * aware_scale.float().unsqueeze(1) - weight.float()) ** 2) * moment).sum()
    assert aware_error < base_error


def test_lightnav_activation_profile_keeps_full_w8_fast_path_contract():
    config = lightnav_activation_quant_config()
    assert config["profile"] == LIGHTNAV_ACTIVATION_PROFILE
    assert config["vision_quantization"] == "fp16"
    assert config["activation_blend"] == 0.25
    assert "constrained_logit_bias" not in config

    vision_config = lightnav_activation_quant_config(vision=True)
    assert vision_config["profile"] == LIGHTNAV_ACTIVATION_VISION_PROFILE
    assert vision_config["vision_quantization"] == "per_channel_on_install"
    assert vision_config["vision_encoder_range_scale"] == 1.04
    assert "constrained_logit_bias" not in vision_config


def test_lightnav_activation_biases_are_position_aligned():
    from rpu_backend.adapters.qwen3_vl import _lightnav_position_logit_biases

    biases = _lightnav_position_logit_biases(SimpleNamespace(
        quant_config={"constrained_logit_bias": {
            "151873": 0.03125,
            "152162": 0.078125,
            "152346": -0.0390625,
        }},
    ))
    assert len(biases) == 5
    assert biases[2][151873 - 151675 + 2].item() == 0.03125
    assert biases[3][152162 - 151931 + 2].item() == 0.078125
    assert biases[4][152346 - 152187 + 2].item() == -0.0390625
