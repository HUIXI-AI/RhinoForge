"""Preserve Pi checkpoint tensors across HF's SigLIP container migration."""
import pytest
import torch
from transformers import SiglipVisionConfig, SiglipVisionModel, GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding

from rpu_backend.adapters.pi05.loader import (
    _align_siglip_checkpoint_keys, _materialize_gemma_rope,
)
from rpu_backend.api.errors import RPUBackendError

PREFIX = 'model.paligemma_with_expert.paligemma.model.vision_tower.'


def policy():
    tower = SiglipVisionModel(SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, image_size=8, patch_size=4,
    )).eval()
    root = current = torch.nn.Module()
    for name in PREFIX.rstrip('.').split('.')[:-1]:
        child = torch.nn.Module()
        current.add_module(name, child)
        current = child
    current.add_module('vision_tower', tower)
    return root, tower


def test_real_siglip_legacy_checkpoint_preserves_every_parameter_and_forward():
    original, tower = policy()
    state = original.state_dict()
    old = {key if key.startswith(PREFIX + 'vision_model.') else
           PREFIX + 'vision_model.' + key.removeprefix(PREFIX): value.clone()
           for key, value in state.items()}
    loaded, loaded_tower = policy()
    _align_siglip_checkpoint_keys(loaded, old)
    assert old.keys() == state.keys()
    loaded.load_state_dict(old, strict=True)
    assert all(torch.equal(value, old[key]) for key, value in state.items())
    pixels = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        assert torch.equal(tower(pixels).last_hidden_state,
                           loaded_tower(pixels).last_hidden_state)


@pytest.mark.parametrize('fault', ['collision', 'shape', 'unknown'])
def test_siglip_alias_rejects_ambiguous_or_invalid_payload_without_mutation(fault):
    class Target:
        def state_dict(self):
            return {PREFIX + 'embeddings.weight': torch.empty(2, 3)}
    legacy = PREFIX + 'vision_model.embeddings.weight'
    value = torch.ones(2, 3)
    state = {legacy: value}
    if fault == 'collision':
        state[PREFIX + 'embeddings.weight'] = torch.ones(2, 3)
    elif fault == 'shape':
        state[legacy] = torch.ones(3, 2)
    else:
        state[PREFIX + 'vision_model.unknown.weight'] = value
    before = dict(state)
    with pytest.raises(RPUBackendError, match='Pi05 SigLIP'):
        _align_siglip_checkpoint_keys(Target(), state)
    assert state.keys() == before.keys()
    assert all(state[key] is before[key] for key in before)


def test_gemma_meta_rotary_buffers_use_actual_config_and_remain_nonpersistent():
    config = GemmaConfig(hidden_size=32, intermediate_size=64,
                         num_attention_heads=2, num_key_value_heads=2,
                         head_dim=16, rope_parameters={'rope_type': 'default', 'rope_theta': 12345.0})
    with torch.device('meta'):
        module = GemmaRotaryEmbedding(config)
    reference = GemmaRotaryEmbedding(config)
    assert _materialize_gemma_rope(module)
    for name in ('inv_freq', 'original_inv_freq'):
        value = getattr(module, name)
        assert value.device.type == 'cpu'
        assert torch.equal(value, getattr(reference, name))
        assert torch.count_nonzero(value) == value.numel()
        assert name not in module.state_dict()
