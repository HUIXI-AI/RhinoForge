"""Exact existing model-zoo W8 assets keep their quantization at reduced cores."""
import copy

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import Qwen3Config, Qwen3ForCausalLM

from rpu_backend.adapters import qwen3
from rpu_backend.api import _execution
from rpu_backend.runtime.decoder import _validate_decoder_w8_scale_lists


SHAPES = ((1024, 3072, 28, 16), (2048, 6144, 28, 16),
          (2560, 9728, 36, 32), (4096, 12288, 36, 32),
          (5120, 17408, 40, 40), (5120, 25600, 64, 64))


def _config(shape=SHAPES[1]):
    hidden, intermediate, layers, heads = shape
    return Qwen3Config(
        architectures=["Qwen3ForCausalLM"], hidden_size=hidden, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=8,
        head_dim=128, vocab_size=151936, tie_word_embeddings=False, rms_norm_eps=1e-6,
        quant_config={"method": "w8a16", "mode": "per_channel_symmetric", "qaxis": 0,
            "skip_modules": [], "quantized_lm_head": True,
            "quantized_embed_tokens": False, "lm_head_untied": True,
            "embed_tokens_untied": False})


def _model(cores, shape=SHAPES[1]):
    with FakeTensorMode():
        model = Qwen3ForCausalLM._from_config(_config(shape), dtype=torch.float16)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight = torch.nn.Parameter(module.weight.to(torch.int8), requires_grad=False)
                module.register_buffer("weight_scale", torch.ones(module.out_features, dtype=torch.float16))
    model._rpu_execution = {"model": {"num_cores": cores}}
    return model


@pytest.fixture(autouse=True)
def safe_process(monkeypatch):
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)


@pytest.mark.parametrize("cores", (4, 6))
@pytest.mark.parametrize("shape", SHAPES)
def test_exact_w8_decoder_and_head_reach_claim_with_actual_cold_topology(monkeypatch, cores, shape):
    model = _model(cores, shape)
    adapter = qwen3.Qwen3Adapter(model)
    assert adapter._topology.identity() == (2, cores, 4, cores, cores, 8)
    assert model.lm_head.weight.dtype == torch.int8
    assert model.model.embed_tokens.weight.dtype == torch.float16
    class ReachedClaim(Exception):
        pass
    calls = []
    def claim(value):
        calls.append(value)
        raise ReachedClaim
    monkeypatch.setattr(qwen3, "_claim_live_instance", claim)
    monkeypatch.setattr(qwen3, "require_native_backend", lambda: None)
    with pytest.raises(ReachedClaim):
        adapter.to_rpu()
    assert calls == [model]
    assert not getattr(model, "_rpu_swizzle_started", False)


@pytest.mark.parametrize("fault", ("head_fp16", "head_scale", "last_scale", "embedding", "norm", "quant_config", "foreign_linear"))
def test_reduced_w8_rejects_changed_actual_inventory_before_claim(monkeypatch, fault):
    model = _model(6)
    adapter = qwen3.Qwen3Adapter(model)
    with model.lm_head.weight.fake_mode:
        if fault == "head_fp16":
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.half())
        elif fault == "head_scale":
            model.lm_head.weight_scale = model.lm_head.weight_scale[:-1]
        elif fault == "last_scale":
            model.model.layers[-1].mlp.down_proj.weight_scale = model.model.layers[-1].mlp.down_proj.weight_scale[:-1]
        elif fault == "embedding":
            model.model.embed_tokens.weight = torch.nn.Parameter(model.model.embed_tokens.weight.to(torch.int8), requires_grad=False)
        elif fault == "norm":
            model.model.layers[-1].self_attn.q_norm.weight = torch.nn.Parameter(
                model.model.layers[-1].self_attn.q_norm.weight.float())
        elif fault == "quant_config":
            model.config.quant_config = copy.deepcopy(model.config.quant_config)
            model.config.quant_config["qaxis"] = 1
        else:
            model.extra = torch.nn.Linear(16, 16, dtype=torch.float16)
    monkeypatch.setattr(qwen3, "_claim_live_instance", lambda *_: pytest.fail("invalid W8 tree claimed device"))
    with pytest.raises(qwen3.UnsupportedModelError):
        adapter.to_rpu()
    assert not getattr(model, "_rpu_swizzle_started", False)


@pytest.mark.parametrize("fault", ("other_size", "head_fp16_config", "quant_embedding", "awq", "wrong_rope"))
def test_reduced_w8_does_not_admit_nearby_profiles(fault):
    config = _config()
    if fault == "other_size":
        config.hidden_size, config.intermediate_size = 1024, 6144
    elif fault == "head_fp16_config":
        config.quant_config["quantized_lm_head"] = False
    elif fault == "quant_embedding":
        config.quant_config["quantized_embed_tokens"] = True
    elif fault == "awq":
        config.quantization_config = {"bits": 4}
    else:
        config.rope_parameters["rope_type"] = "linear"
    with pytest.raises(ValueError, match="exact plain FP16"):
        _execution.qwen3_core_profile(config, 6)


def test_native_install_boundary_keeps_all_real_scale_identities():
    model = _model(6)
    roles = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    scales = tuple([layer.get_submodule(role).weight_scale for layer in model.model.layers] for role in roles)
    _validate_decoder_w8_scale_lists(model.model, scales)
    with model.lm_head.weight.fake_mode:
        scales[-1][-1] = scales[-1][-1].clone()
    with pytest.raises(ValueError, match="actual INT8 weight / FP16 per-channel scale"):
        _validate_decoder_w8_scale_lists(model.model, scales)


def test_existing_eight_core_w8_route_keeps_its_default_topology():
    model = _model(8)
    adapter = qwen3.Qwen3Adapter(model)
    assert adapter._is_w8a16 and adapter._topology is None


def test_int8_head_cpu_handoff_swizzles_for_its_actual_four_core_partition(monkeypatch):
    model = _model(6)
    topology = qwen3._execution_topology(model.config, model._rpu_execution)
    model.model._rpu_decoder_topology = topology
    converted, published = [], []
    real_transform = qwen3.transform_linear_weight
    real_to = torch.Tensor.to
    def transform(weight, **kwargs):
        converted.append(kwargs)
        return real_transform(weight, **kwargs)
    def host_to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            return tensor
        return real_to(tensor, *args, **kwargs)
    monkeypatch.setattr(qwen3, "transform_linear_weight", transform)
    monkeypatch.setattr(torch.Tensor, "to", host_to)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_lm_head",
                        lambda *args: published.append(args), raising=False)
    from rpu_backend.runtime.weights import pad_decoder_lm_head_weight
    with model.lm_head.weight.fake_mode:
        pad_decoder_lm_head_weight(model, logical_size=151936, physical_size=151968)
        qwen3._push_lm_head_weight(model, 17)
    assert converted == [{"partition": 1, "num_cores": 6}]
    assert len(published) == 1 and published[0][0] == 17
    assert published[0][1].dtype == torch.int8 and published[0][1].shape == (151968, 2048)
    assert published[0][2].dtype == torch.float16 and published[0][2].shape == (151968,)
    assert model._rpu_lm_head_int8_cpu_ref.dtype == torch.int8
