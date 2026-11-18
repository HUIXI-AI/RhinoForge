"""One-time weight padding preserves MLP semantics and logical admission."""
from types import SimpleNamespace

import pytest
import torch
from rpu_backend.runtime import weights
from rpu_backend.runtime.topology import decoder_mlp_intermediate_size, resolve_decoder_topology


def _model(layers=2, logical=9728):
    blocks = []
    for _ in range(layers):
        mlp = torch.nn.Module()
        for name, n, k in (("gate_proj", logical, 64), ("up_proj", logical, 64),
                           ("down_proj", 64, logical)):
            mlp.add_module(name, torch.nn.Linear(k, n, bias=False, dtype=torch.float16))
        blocks.append(SimpleNamespace(mlp=mlp))
    return SimpleNamespace(config=SimpleNamespace(hidden_size=64, intermediate_size=logical),
                           model=SimpleNamespace(layers=blocks))


def _forward(mlp, x):
    gate = torch.nn.functional.linear(x, mlp.gate_proj.weight.float())
    up = torch.nn.functional.linear(x, mlp.up_proj.weight.float())
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up,
                                      mlp.down_proj.weight.float())


@pytest.mark.parametrize("logical, physical", [(3584, 3648), (8192, 8256), (9728, 9792), (17408, 17472), (25600, 25728)])
def test_static_padding_preserves_mlp_and_needs_no_activation_padding(monkeypatch, logical, physical):
    torch.manual_seed(91)
    model = _model(logical=logical)
    x = torch.randn(3, 64) * .2
    expected = [_forward(layer.mlp, x) for layer in model.model.layers]
    original = [{name: child.weight.detach().clone() for name, child in layer.mlp.named_children()}
                for layer in model.model.layers]
    weights.pad_decoder_mlp_weights(model, logical_size=logical, physical_size=physical)
    assert model.config.intermediate_size == logical
    for layer, before, want in zip(model.model.layers, original, expected):
        for name, child in layer.mlp.named_children():
            rows, cols = before[name].shape
            assert torch.equal(child.weight[:rows, :cols], before[name])
            tail = child.weight[:, cols:] if name == 'down_proj' else child.weight[rows:]
            assert torch.count_nonzero(tail) == 0
        torch.testing.assert_close(_forward(layer.mlp, x), want, atol=1e-7, rtol=2e-5)
    monkeypatch.setattr(torch.nn.functional, 'pad', lambda *_a, **_k: pytest.fail('steady pad'))
    for _ in range(3):
        torch.testing.assert_close(_forward(model.model.layers[0].mlp, x), expected[0], atol=1e-7, rtol=2e-5)
    with pytest.raises(ValueError, match='original contiguous'):
        weights.pad_decoder_mlp_weights(model, logical_size=logical, physical_size=physical)


@pytest.mark.parametrize("logical, physical", [(3584, 3648), (8192, 8256), (9728, 9792), (17408, 17472), (25600, 25728)])
def test_bad_last_projection_rejects_before_any_padding(logical, physical):
    model = _model(logical=logical)
    first = model.model.layers[0].mlp.gate_proj.weight
    model.model.layers[-1].mlp.down_proj = torch.nn.Linear(logical + 16, 64, bias=False, dtype=torch.float16)
    with pytest.raises(ValueError, match='original contiguous'):
        weights.pad_decoder_mlp_weights(model, logical_size=logical, physical_size=physical)
    assert model.model.layers[0].mlp.gate_proj.weight is first
    assert first.shape == (logical, 64)


def test_physical_width_does_not_expand_logical_model_admission():
    geometry = dict(hidden_size=2560, intermediate_size=9728, num_q_heads=32,
                    num_kv_heads=8, head_dim=128, vocab_size=151936)
    for cores in (4, 6, 8):
        t = resolve_decoder_topology(num_cores=cores, **geometry)
        physical = decoder_mlp_intermediate_size(9728, t.mlp_tp)
        assert physical == (9792 if cores == 6 else 9728)
        assert physical % (16 * t.mlp_tp) == 0
    geometry['intermediate_size'] = 9792
    with pytest.raises(ValueError, match='no admitted'):
        resolve_decoder_topology(num_cores=6, **geometry)


@pytest.mark.parametrize("logical, physical", [(9728, 9792), (17408, 17472), (25600, 25728)])
def test_int8_padding_preserves_dequantized_mlp_and_six_core_packing(logical, physical):
    torch.manual_seed(23)
    model = _model(logical=logical)
    for layer in model.model.layers:
        for projection in layer.mlp.children():
            projection.weight = torch.nn.Parameter(
                torch.randint(-8, 8, projection.weight.shape, dtype=torch.int8), requires_grad=False)
            projection.register_buffer("weight_scale", torch.rand(projection.out_features).half() * .01)

    def run(mlp, x):
        def linear(name, value):
            projection = getattr(mlp, name)
            weight = projection.weight.float() * projection.weight_scale.float()[:, None]
            return torch.nn.functional.linear(value, weight)
        return linear("down_proj", torch.nn.functional.silu(linear("gate_proj", x)) * linear("up_proj", x))

    x = torch.randn(2, 64) * .1
    expected = [run(layer.mlp, x) for layer in model.model.layers]
    weights.pad_decoder_mlp_weights(model, logical_size=logical, physical_size=physical)
    for layer, want in zip(model.model.layers, expected):
        torch.testing.assert_close(run(layer.mlp, x), want)
        for name, projection in layer.mlp.named_children():
            packed = weights.transform_linear_weight(projection.weight, 0 if name == "down_proj" else 1, 6)
            assert packed.shape == projection.weight.shape
            assert projection.weight_scale.shape == (projection.out_features,)


def test_bad_padding_storage_rejects_before_claim(monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from torch._subclasses.fake_tensor import FakeTensorMode
    from rpu_backend.adapters import qwen3
    with FakeTensorMode():
        model = Qwen3ForCausalLM._from_config(Qwen3Config(hidden_size=2560,
            intermediate_size=9728, num_hidden_layers=36, num_attention_heads=32,
            num_key_value_heads=8, head_dim=128, vocab_size=151936), dtype=torch.float16)
    model._rpu_execution = {'model': {'num_cores': 6}}
    adapter = qwen3.Qwen3Adapter(model)
    leaf = model.model.layers[-1].mlp.up_proj
    with leaf.weight.fake_mode:
        leaf.weight = torch.nn.Parameter(torch.empty(2560, 9728, dtype=torch.float16).t())
    monkeypatch.setattr(qwen3, '_claim_live_instance', lambda *_: pytest.fail('invalid storage claimed model'))
    with pytest.raises(ValueError, match='original contiguous'):
        adapter.to_rpu()
    assert not getattr(model, '_rpu_swizzle_started', False)
