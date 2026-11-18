"""Real HF config/model boundaries for the exact MoE checkpoint loader."""
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForImageTextToText, Qwen3_5MoeConfig

from rpu_backend.adapters.qwen3_5 import vision
from rpu_backend.adapters.qwen3_5_moe import loader, quant_scope
from test_qwen35_moe_python_integration import _exact_config


@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen3_5_moe_vision"])
def test_legacy_and_current_vision_names_preserve_exact_admission(model_type):
    raw = _exact_config()
    raw["vision_config"]["model_type"] = model_type
    assert quant_scope.validate_exact_profile(raw)["profile"] == quant_scope.EXACT_PROFILE_ID
    config = Qwen3_5MoeConfig.from_dict(raw)
    for representation in (config, config.to_dict()):
        assert loader.parse_quant_config(representation)["weight_mode"] == 5
    visual = SimpleNamespace(config=config.vision_config, blocks=[object()] * 27)
    vision._validate_qwen3_5_moe_35b_vision_profile(visual, visual.config, 8)


@pytest.mark.parametrize("field,value", [
    ("model_type", "qwen3_5"), ("depth", 28), ("hidden_size", 1153),
    ("intermediate_size", 4352), ("num_heads", 18), ("out_hidden_size", 4096),
    ("in_channels", 4), ("patch_size", 14), ("spatial_merge_size", 1),
    ("temporal_patch_size", 1), ("num_position_embeddings", 2305),
    ("hidden_act", "gelu"), ("deepstack_visual_indexes", [1]),
    ("rope_parameters", {"rope_type": "default", "rope_theta": 10000.0}),
    ("rope_parameters", {"rope_type": "axial", "rope_theta": 20000.0}),
    ("rope_parameters", {"rope_type": "axial", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.5}),
])
def test_real_vision_config_still_rejects_semantic_neighbours(field, value):
    config = Qwen3_5MoeConfig.from_dict(_exact_config())
    # Bypass HF's strict assignment so the backend's own gate is exercised.
    object.__setattr__(config.vision_config, field, value)
    with pytest.raises(ValueError, match="vision_config"):
        quant_scope.validate_exact_profile(config)
    visual = SimpleNamespace(config=config.vision_config, blocks=[object()] * 27)
    with pytest.raises(ValueError, match="exact TP8"):
        vision._validate_qwen3_5_moe_35b_vision_profile(visual, visual.config, 8)


def test_real_meta_model_keeps_checkpoint_shapes_and_materializes_rotary_buffers():
    config = Qwen3_5MoeConfig.from_dict(_exact_config())
    loader.parse_quant_config(config)
    with torch.device("meta"):
        model = AutoModelForImageTextToText.from_config(config)
    parameters = dict(model.named_parameters())
    assert all(value.is_meta for value in parameters.values())
    assert set(quant_scope.expected_fp8_weight_names(config.text_config.layer_types)) <= parameters.keys()
    assert parameters["model.language_model.layers.0.mlp.experts.gate_up_proj"].shape == (256, 1024, 2048)
    assert parameters["model.language_model.layers.39.mlp.experts.down_proj"].shape == (256, 2048, 512)
    assert parameters["model.language_model.layers.0.mlp.gate.weight"].shape == (256, 2048)
    visual = model.model.visual
    vision._validate_qwen3_5_moe_35b_vision_profile(visual, config.vision_config, 8)
    assert visual.patch_embed.proj.weight.shape == (1152, 3, 2, 16, 16)
    assert visual.merger.linear_fc2.weight.shape == (2048, 4608)
    # A meta skeleton verifies names/shapes without pretending to be loaded weights.
    with pytest.raises(ValueError, match="CPU FP16"):
        vision._validate_qwen3_5_moe_35b_vision_weights(visual, config.vision_config)
    rotary = visual.rotary_pos_emb
    if hasattr(rotary, "compute_axial_rope_parameters"):
        reference = type(rotary)(config.vision_config)
    else:
        reference = type(rotary)(rotary.dim, rotary.theta)
    loader._materialize_meta_buffers(model)
    assert all(value.device.type == "cpu" and bool(torch.isfinite(value).all())
               for value in model.buffers())
    torch.testing.assert_close(rotary.inv_freq, reference.inv_freq, rtol=0, atol=0)
    legacy_inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 36, 2, dtype=torch.float32) / 36))
    torch.testing.assert_close(rotary.inv_freq, legacy_inv_freq, rtol=0, atol=0)
    if hasattr(reference, "original_inv_freq"):
        torch.testing.assert_close(rotary.original_inv_freq, reference.original_inv_freq, rtol=0, atol=0)
    assert all(value.is_meta for value in model.parameters())
