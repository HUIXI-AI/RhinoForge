"""Host checks for the exact 9B TP8 W8 VL cold admission; no model execution."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter, cores, text, vision
from rpu_backend.adapters.qwen3_5.quant_scope import is_full_from_layer_types, resolve_quant_specs
from rpu_backend.api.errors import UnsupportedModelError


def _config():
    tc = NS(
        hidden_size=4096, intermediate_size=12288, num_hidden_layers=32,
        num_attention_heads=16, num_key_value_heads=4, head_dim=256,
        linear_num_key_heads=16, linear_num_value_heads=32,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, vocab_size=248320, model_type="qwen3_5_text",
        hidden_act="silu", rms_norm_eps=1e-6, attention_bias=False,
        attn_output_gate=True, use_cache=True, full_attention_interval=4,
        layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(32)],
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000,
                         "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                         "mrope_section": [11, 11, 10]},
    )
    vc = NS(depth=27, num_heads=16, hidden_size=1152, intermediate_size=4304,
            out_hidden_size=4096, hidden_act="gelu_pytorch_tanh", in_channels=3,
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
            num_position_embeddings=2304)
    return NS(text_config=tc, vision_config=vc, model_type="qwen3_5",
              tie_word_embeddings=False, image_token_id=248056,
              quant_config={"format_version": 1, "method": "w8a16"})


def _storage_model():
    """Tiny real CPU tensors cover every declared role, without allocating 9B weights.

    This fixture exercises storage validation only. Exact model/weight geometry
    is tested independently below, before the storage validator is reached.
    """
    model = torch.nn.Module()
    model.config = _config()
    model.model = torch.nn.Module()
    inner = model.model.language_model = torch.nn.Module()
    inner.layers = torch.nn.ModuleList(torch.nn.Module() for _ in range(32))
    specs = resolve_quant_specs(is_full_from_layer_types(model.config.text_config.layer_types),
                               model.config.quant_config)
    for name in specs:
        parent = inner
        parts = name.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                parent.add_module(part, torch.nn.Module())
            parent = getattr(parent, part)
        linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.float16)
        linear.weight = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.int8), requires_grad=False)
        linear.register_buffer("weight_scale", torch.ones(2, dtype=torch.float16))
        parent.add_module(parts[-1], linear)
    inner.embed_tokens = torch.nn.Embedding(2, 2, dtype=torch.float16)
    model.model.visual = torch.nn.Linear(2, 2, dtype=torch.float16)
    model.lm_head = torch.nn.Linear(2, 2, bias=False, dtype=torch.float16)
    return model, tuple(specs)


def test_uniform_w8_vl_preflight_preserves_nonretained_four_arena_scope(monkeypatch):
    config = _config()
    monkeypatch.setenv("QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED", "1")
    Qwen3_5Adapter.preflight_execution(config, {"model": {"num_cores": 8}})
    assert text._public_retained_prefill(config, public_owner=True, vision_admitted=False)
    assert not text._public_retained_prefill(config, public_owner=True, vision_admitted=True)
    policy = NS(graph_arena_count=4, execution_core_count=8, qwen35_legacy_27b_sdk_budget=False)
    state = NS(vision_arena_admitted=True, graph_cache=NS(runtime_policy=policy),
               execution_topology=cores.core_topology())
    options = vision._lazy_vision_graph_options(state)
    assert options["_graph_runtime_policy"] is policy and options["_cold_numeric_opt_in"] is True
    policy.graph_arena_count = 0
    with pytest.raises(RuntimeError, match="arena ownership"):
        vision._lazy_vision_graph_options(state)


@pytest.mark.parametrize("num_cores", [4, 6])
def test_quantized_9b_reduced_cores_still_rejected(num_cores):
    with pytest.raises(ValueError, match="exact 2B"):
        Qwen3_5Adapter.preflight_execution(_config(), {"model": {"num_cores": num_cores}})


@pytest.mark.parametrize("quant", [
    {"method": "w4a16_pgrp", "group_size": 32},
    {"format_version": 2, "default": {"method": "w8a16"},
     "projection_overrides": {"linear_attn.in_proj_b": {"method": "fp16"}}},
    {},
])
def test_9b_vl_does_not_admit_w4_mixed_or_missing_quant_method(quant):
    config = _config()
    config.quant_config = quant
    with pytest.raises(ValueError):
        cores.validate_dense_9b_vl_profile(config)


def test_9b_vl_geometry_remains_exact():
    config = _config()
    config.text_config.intermediate_size += 16
    with pytest.raises(ValueError, match="exact dense"):
        cores.validate_dense_9b_vl_profile(config)


def test_9b_vl_raw_geometry_is_checked_before_storage(monkeypatch):
    config = _config()
    shape = lambda *dims: NS(weight=NS(shape=dims))
    layer = NS(input_layernorm=shape(4096), post_attention_layernorm=shape(4096),
               mlp=NS(gate_proj=shape(12287, 4096)))
    inner = NS(config=config.text_config, layers=[layer] * 32,
               embed_tokens=shape(248320, 4096), norm=shape(4096))
    model = NS(config=config, model=NS(language_model=inner))
    monkeypatch.setattr(cores, "validate_cold_weight_dtype",
                        lambda *args, **kwargs: pytest.fail("invalid geometry reached storage admission"))
    with pytest.raises(ValueError, match="requires shape"):
        cores.validate_dense_9b_vl_weights(model)


def test_all_w8_roles_and_floating_nonquant_weights_survive_cold_half():
    model, names = _storage_model()
    assert len(names) == 248
    cores.validate_cold_weight_dtype(model)
    model.half()
    cores.validate_cold_weight_dtype(model, fp16=True)
    assert all(model.model.language_model.get_submodule(name).weight.dtype == torch.int8
               for name in names)


@pytest.mark.parametrize("mutation", [
    "float_weight", "missing_scale", "scale_shape", "scale_dtype",
    "scale_nan", "scale_zero", "scale_negative", "scale_noncontiguous",
    "meta_weight", "installed_partition",
])
@pytest.mark.parametrize("projection_index", [0, -1])
def test_cold_w8_storage_rejects_malformed_first_and_last_projection(mutation, projection_index):
    model, names = _storage_model()
    linear = model.model.language_model.get_submodule(names[projection_index])
    if mutation == "float_weight":
        linear.weight = torch.nn.Parameter(linear.weight.float(), requires_grad=False)
    elif mutation == "missing_scale":
        del linear.weight_scale
    elif mutation == "scale_shape":
        linear.weight_scale = torch.ones(3, dtype=torch.float16)
    elif mutation == "scale_dtype":
        linear.weight_scale = linear.weight_scale.float()
    elif mutation in ("scale_nan", "scale_zero", "scale_negative"):
        linear.weight_scale[0] = {"scale_nan": float("nan"), "scale_zero": 0, "scale_negative": -1}[mutation]
    elif mutation == "scale_noncontiguous":
        linear.weight_scale = torch.ones(4, dtype=torch.float16)[::2]
    elif mutation == "meta_weight":
        linear.weight = torch.nn.Parameter(torch.empty(2, 2, dtype=torch.int8, device="meta"), requires_grad=False)
    else:
        linear._rpu_linear_partition = "col"
    with pytest.raises(ValueError):
        cores.validate_cold_weight_dtype(model)


@pytest.mark.parametrize("target", ["model.visual", "model.language_model.embed_tokens", "lm_head"])
def test_nonquantized_vision_embedding_and_head_cannot_be_integer_or_fp32(target):
    model, _ = _storage_model()
    owner = model.get_submodule(target)
    for dtype in (torch.int8, torch.float32):
        owner.weight = torch.nn.Parameter(owner.weight.to(dtype), requires_grad=False)
        with pytest.raises(ValueError, match="cold weight"):
            cores.validate_cold_weight_dtype(model)


def test_integer_projection_alias_cannot_hide_a_quantized_vision_weight():
    model, names = _storage_model()
    model.model.visual.weight = model.model.language_model.get_submodule(names[0]).weight
    with pytest.raises(ValueError, match="model.visual.weight"):
        cores.validate_cold_weight_dtype(model)


def test_fp16_cold_weights_and_quant_gate_are_unchanged(monkeypatch):
    from rpu_backend.adapters.qwen3_5 import _check_quant_policy

    config = _config()
    config.quant_config = None
    cores.validate_dense_9b_vl_profile(config)
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    cores.validate_cold_weight_dtype(model)
    with pytest.raises(ValueError, match="FP16"):
        cores.validate_cold_weight_dtype(model, fp16=True)
    monkeypatch.delenv("QWEN3_5_QUANT_ALLOW_UNCERTIFIED", raising=False)
    with pytest.raises(UnsupportedModelError, match="QWEN3_5_QUANT_ALLOW_UNCERTIFIED=1"):
        _check_quant_policy(_config())
