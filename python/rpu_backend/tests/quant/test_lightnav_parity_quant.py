from rpu_backend.quant.convert_qwen3_vl import (
    LIGHTNAV_PARITY_PROFILE,
    is_lightnav_parity_weight,
    lightnav_parity_quant_config,
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
