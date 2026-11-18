from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.dinov3 import _pad_attn_heads_to_multiple_of_cores


def _direct_model():
    attention = torch.nn.Module()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(attention, name, torch.nn.Linear(24, 24, bias=name != "k_proj"))
    return SimpleNamespace(config=SimpleNamespace(hidden_size=24, num_attention_heads=3),
                           layer=[SimpleNamespace(attention=attention)])


def test_dinov3_direct_encoder_padding_preserves_existing_weights():
    model = _direct_model()
    attention = model.layer[0].attention
    original = {name: layer.weight.detach().clone() for name, layer in attention.named_children()}
    _pad_attn_heads_to_multiple_of_cores(model)
    for name in ("q_proj", "k_proj", "v_proj"):
        weight = getattr(attention, name).weight
        assert tuple(weight.shape) == (64, 24)
        assert torch.equal(weight[:24], original[name])
        assert not weight[24:].count_nonzero()
    assert torch.equal(attention.o_proj.weight[:, :24], original["o_proj"])
    assert not attention.o_proj.weight[:, 24:].count_nonzero()


def test_dinov3_requires_the_direct_encoder_layer_layout():
    direct = _direct_model()
    wrapped = SimpleNamespace(config=direct.config, model=direct)
    _pad_attn_heads_to_multiple_of_cores(wrapped)
    assert tuple(direct.layer[0].attention.q_proj.weight.shape) == (64, 24)
    from rpu_backend.runtime import UnsupportedModelError
    with pytest.raises(UnsupportedModelError, match="encoder layer container"):
        _pad_attn_heads_to_multiple_of_cores(SimpleNamespace(config=direct.config))
