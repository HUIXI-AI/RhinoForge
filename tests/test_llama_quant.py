"""Llama-3.2-1B on-install W8A16 / G32 W4A16 / NVFP4: profiles, preflight,
conversion and the actual installer payload at 4/6/8 cores. Host only; no board."""
from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import LlamaConfig, LlamaForCausalLM

from rpu_backend.adapters import llama, qwen3
from rpu_backend.api import causal_lm
from rpu_backend.api._execution import normalize_rpu_execution
from rpu_backend.quant import convert_qwen3_nvfp4 as nvfp4
from rpu_backend.quant import convert_qwen3_w4a16 as w4
from rpu_backend.quant._common import quantize_linear_per_channel
from rpu_backend.quant.convert_llama import quantize_llama_w8a16_
from rpu_backend.runtime import decoder, hw_attrs
from rpu_backend.tests.acceptance.test_causal_lm_adapter_install_lifecycle import (
    _Outer, _prepare_case,
)
from rpu_backend.tests.acceptance.test_decoder_install_transaction import (
    native_install, _FakeTensor, _finalize,
)

ROOT = Path(__file__).resolve().parents[1]
RECIPES = {  # TOML recipe -> (profile variant, on-install quantization)
    "w8a16_text": ("w8a16", "w8a16"),
    "w4a16_text_rtn_g32": ("w4a16-g32", "w4a16"),
    "nvfp4_text": ("nvfp4", "nvfp4"),
}
# Six MLP lanes: byte-width W8 and unified N32K32 W4 pad to 192 (8256); NVFP4 to 384 (8448).
MLP6 = {"w8a16": 8256, "w4a16": 8256, "nvfp4": 8448}


@pytest.fixture
def common():
    return importlib.import_module("examples.common.config")


def _config(**overrides):
    # Llama-3.2-1B config.json (the registered llama-3.2-1b asset).
    values = dict(
        architectures=["LlamaForCausalLM"],
        hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=32, num_key_value_heads=8, head_dim=64,
        vocab_size=128256, rms_norm_eps=1e-5, tie_word_embeddings=True,
        hidden_act="silu", attention_bias=False, mlp_bias=False,
        max_position_embeddings=131072, rope_theta=500000.0,
        rope_scaling={"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                      "original_max_position_embeddings": 8192, "rope_type": "llama3"})
    values.update(overrides)
    return LlamaConfig(**values)


@pytest.mark.parametrize("recipe", RECIPES)
def test_quant_profiles_bind_exact_recipe_geometry_and_cores(common, recipe):
    variant, quantization = RECIPES[recipe]
    config = common.load_config(ROOT / f"examples/configs/llama/3_2_1b/{recipe}.toml")
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["id"] == f"llama-3.2-1b.{variant}.text"
    assert profile["asset_alias"] == config["example"]["registry_alias"] == "llama-3.2-1b"
    assert common.install_quantization("llama", profile["id"]) == quantization
    assert limits["on_install_quantization"] == quantization
    assert limits.get("group_size") == (32 if quantization == "w4a16" else None)
    assert tuple(limits[key] for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "vocab_size")) == (2048, 8192, 16, 32, 8, 64, 128256)
    envelope = llama._CHUNK_ENVELOPE[("llama", 16, 2048)]
    assert limits["kv_capacity"] == "requested-at-install; allocation must succeed"
    assert (envelope.max_kv_len, limits["max_chunk_size"], envelope.chunk) == (0, 64, 64)
    assert limits["allowed_num_cores"] == [4, 6, 8]
    assert limits["batch_size"] == 1 and limits["formal_certification"] is False
    assert (config["input"]["decode_steps"], config["run"]["warmup"], config["run"]["runs"]) == (4, 2, 5)
    request = {"model": {"num_cores": 8}}
    for cores in (4, 6, 8):
        request["model"]["num_cores"] = cores
        common._validate_execution(request, profile=profile)
    for cores in (2, 5, 7):
        request["model"]["num_cores"] = cores
        with pytest.raises((ValueError, common.ConfigError)):
            common._validate_execution(request, profile=profile)
    # Llama has no accumulation knob: the recipe runs ACC16 and the loader refuses ACC32.
    assert "linear_acc32" not in config["rpu_execution"]["prefill"]
    with pytest.raises(ValueError, match="linear_acc32"):
        normalize_rpu_execution({"prefill": {"linear_acc32": True}}, entry_point="test",
                                supported=llama.LlamaAdapter.EXECUTION_SUPPORTED)


@pytest.mark.parametrize("quantization", ["w8a16", "w4a16", "nvfp4"])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_preflight_admits_every_recipe_and_core_budget(quantization, cores):
    profile = llama.LlamaAdapter.preflight_quantization(
        _config(), {"model": {"num_cores": cores}}, quantization)
    assert (profile.hidden_size, profile.num_layers, profile.qk_norm) == (2048, 16, False)


@pytest.mark.parametrize("fault", ["head_recipe", "tp5", "mlp_bias", "vocab", "awq", "rope"])
def test_public_llama_quant_rejects_before_weight_load(monkeypatch, fault):
    config = _config(**{"mlp_bias": {"mlp_bias": True}, "vocab": {"vocab_size": 128000},
                        "rope": {"rope_scaling": {"rope_type": "dynamic", "factor": 2.0}}}
                     .get(fault, {}))
    if fault == "awq":
        config.quantization_config = {"quant_method": "awq", "bits": 4}
    execution = {"model": {"num_cores": 5 if fault == "tp5" else 6}}
    monkeypatch.setattr(causal_lm.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: pytest.fail("invalid Llama recipe loaded weights"))
    with pytest.raises((llama.UnsupportedModelError, ValueError)):
        causal_lm.RPUModelForCausalLM.from_pretrained(
            "source", rpu_execution=execution,
            quantization="w4a16_lm_head" if fault == "head_recipe" else "w4a16")


def test_qwen3_does_not_inherit_the_llama_w8_recipe():
    from transformers import Qwen3Config
    config = Qwen3Config(hidden_size=1024, intermediate_size=3072, num_hidden_layers=28,
                         num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                         vocab_size=151936, tie_word_embeddings=True)
    with pytest.raises(qwen3.UnsupportedModelError):
        qwen3.Qwen3Adapter.preflight_quantization(config, {"model": {"num_cores": 8}}, "w8a16")


def _tiny_projections(layers=2):
    generator = torch.Generator().manual_seed(7)
    blocks = torch.nn.ModuleList()
    for index in range(layers):
        layer = torch.nn.Module()
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        for offset, (role, _) in enumerate(w4._PROJECTIONS):
            linear = torch.nn.Linear(64, 32, bias=False, dtype=torch.float16)
            weight = torch.randn(32, 64, generator=generator) * (index + 1) * (offset + 1) / 8
            linear.weight = torch.nn.Parameter(weight.half(), requires_grad=False)
            parent, name = role.split(".")
            setattr(getattr(layer, parent), name, linear)
        blocks.append(layer)
    return torch.nn.Module(), blocks


def test_w8a16_converter_quantizes_every_projection_per_channel():
    model, blocks = _tiny_projections()
    model.model = torch.nn.Module()
    model.model.layers = blocks
    originals = [[layer.get_submodule(role).weight.clone() for role, _ in w4._PROJECTIONS]
                 for layer in blocks]
    quantize_llama_w8a16_(model)
    for layer, sources in zip(blocks, originals, strict=True):
        for (role, _), source in zip(w4._PROJECTIONS, sources, strict=True):
            projection = layer.get_submodule(role)
            expected_weight, expected_scale = quantize_linear_per_channel(source, bits=8)
            assert projection.weight.dtype == torch.int8 and not projection.weight.requires_grad
            assert torch.equal(projection.weight, expected_weight)
            assert torch.equal(projection.weight_scale, expected_scale)
            assert projection.weight_scale.shape == (32,) and projection.weight_scale.dtype == torch.float16
            # A buffer moves with model.to('rpu'); it is not a checkpoint key.
            assert "weight_scale" in dict(projection.named_buffers())
            assert "weight_scale" not in projection.state_dict()
            # Per-channel dequantization stays within half a step.
            error = (projection.weight.float() * projection.weight_scale.float()[:, None] - source.float()).abs()
            assert torch.all(error <= projection.weight_scale.float()[:, None] / 2 + 1e-3)


@pytest.mark.parametrize("fault", ["dtype", "bias", "scaled"])
def test_w8a16_converter_checks_every_projection_before_replacing_any(fault):
    model, blocks = _tiny_projections()
    model.model = torch.nn.Module()
    model.model.layers = blocks
    last = blocks[-1].mlp.down_proj
    if fault == "dtype":
        last.weight = torch.nn.Parameter(last.weight.float(), requires_grad=False)
    elif fault == "bias":
        last.bias = torch.nn.Parameter(torch.zeros(32, dtype=torch.float16))
    else:
        last.register_buffer("weight_scale", torch.ones(32, dtype=torch.float16))
    with pytest.raises(ValueError, match="W8A16"):
        quantize_llama_w8a16_(model)
    assert all(layer.get_submodule(role).weight.dtype != torch.int8
               for layer in blocks for role, _ in w4._PROJECTIONS)


def _fake_model(cores):
    model = _Outer(_config())
    with FakeTensorMode():
        actual = LlamaForCausalLM._from_config(model.config, dtype=torch.float16)
    model.model, model.lm_head = actual.model, actual.lm_head
    model._rpu_execution = {"model": {"num_cores": cores}}
    return model


def _host_install(monkeypatch, model, quantization):
    # Real adapter, converters' orchestration, swizzle, installer validation and
    # A9; device/native execution and quantized tensor data are doubled.
    pre, post, stamp = (hw_attrs.validate_preinstall, hw_attrs.validate_postinstall,
                        hw_attrs.install_hw_attr_validator)
    _prepare_case(monkeypatch, llama, model)
    monkeypatch.setattr(hw_attrs, "validate_preinstall", pre)
    monkeypatch.setattr(hw_attrs, "validate_postinstall", post)
    monkeypatch.setattr(hw_attrs, "install_hw_attr_validator", stamp)
    from rpu_backend.runtime import weights
    # Every FakeTensor reports data_ptr() 0, so address-based tie detection
    # would mark each INT8 projection as tied to the embedding.
    monkeypatch.setattr(weights, "_collect_embedding_data_ptrs", lambda _module: set())
    swizzles = []
    def swizzle(owner, **kwargs):
        swizzles.append(kwargs)
        return weights.convert_linear_weights_inplace(owner, **kwargs)
    monkeypatch.setattr(llama, "swizzle_model_inplace", swizzle)
    packed = []
    if quantization == "w4a16":
        def quantize(weight, group_size):
            packed.append(tuple(weight.shape))
            return (torch.empty_like(weight, dtype=torch.int8),
                    torch.empty(weight.shape[1] // group_size, weight.shape[0], dtype=torch.float16),
                    group_size)
        monkeypatch.setattr(w4, "quantize_int4_group_wise", quantize)
        monkeypatch.setattr(w4, "swizzle_pack_int4_pgrp",
                            lambda value, *a: torch.empty(value.numel() // 2, dtype=torch.uint8))
    elif quantization == "nvfp4":
        def quantize(weight):
            packed.append(tuple(weight.shape))
            n, k = weight.shape
            return (torch.empty(n, k, dtype=torch.uint8), torch.empty(k // 16, n, dtype=torch.uint8),
                    torch.empty((), dtype=torch.float32))
        monkeypatch.setattr(nvfp4, "quantize_nvfp4", quantize)
        monkeypatch.setattr(nvfp4, "swizzle_pack_nvfp4",
                            lambda codes, *a: torch.empty(codes.numel() // 2, dtype=torch.uint8))
        monkeypatch.setattr(nvfp4, "swizzle_nvfp4_scales", lambda scales, *a: scales)
        monkeypatch.setattr(nvfp4, "pad_nvfp4_tensor_scales",
                            lambda values: torch.empty((len(values) + 7) // 8 * 8, dtype=torch.float32))
    weight_calls = []
    for op in ("causal_decoder_set_weights_w8a16", "causal_decoder_set_weights_nvfp4"):
        monkeypatch.setattr(torch.ops.rpu, op,
                            lambda *args, op=op: weight_calls.append((op, args)), raising=False)
    installs = []
    def install(inner, **kwargs):
        monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology",
                            lambda h: kwargs["topology"].identity()[1:], raising=False)
        installs.append(kwargs)
        table = _FakeTensor()
        return decoder._install_causal_decoder_forward(inner, **kwargs, cos_sin=(table, table))
    monkeypatch.setattr(llama, "_install_causal_decoder_forward", install)
    return swizzles, packed, weight_calls, installs


@pytest.mark.parametrize("quantization", ["w8a16", "w4a16", "nvfp4"])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_adapter_install_passes_the_actual_payload_to_the_installer(
        monkeypatch, native_install, quantization, cores):
    model = _fake_model(cores)
    swizzles, packed, weight_calls, installs = _host_install(monkeypatch, model, quantization)
    adapter = llama.LlamaAdapter(model, quantization=quantization)
    with model.lm_head.weight.fake_mode:
        assert adapter.to_rpu() is model
    topology = installs[0]["topology"]
    # Quantized installs bind the shared template at every budget, eight included.
    assert topology.identity()[1:4] == (cores, 8 if cores == 8 else 4, cores)
    assert installs[0]["arch"] == "llama"
    assert swizzles == [{"skip_names": set() if quantization == "w8a16" else set(w4.SKIP_LINEAR_NAMES),
                         "attn_num_cores": topology.attn_tp, "mlp_num_cores": cores,
                         "lm_head_num_cores": topology.lm_head_tp, "execution_core_count": cores}]
    physical = MLP6[quantization] if cores == 6 else 8192
    mlp = model.model.layers[-1].mlp
    assert (mlp.gate_proj.out_features, mlp.down_proj.in_features) == (physical, physical)
    assert model.config.intermediate_size == 8192
    nvfp4_install = quantization == "nvfp4"
    assert ("nvfp4_tensor_scales" in installs[0]) is nvfp4_install
    assert [op for op, _ in weight_calls] == [
        "causal_decoder_set_weights_nvfp4" if nvfp4_install else "causal_decoder_set_weights_w8a16"]
    if quantization != "w8a16":
        # NVFP4 quantizes the zero-padded stripe; G32 W4 quantizes the logical
        # weight and pads codes with zeros and unit scales.
        width = physical if quantization == "nvfp4" else 8192
        assert len(packed) == 16 * 7 and packed[4:7] == [(width, 2048)] * 2 + [(2048, width)]
    for (role, partition), scales in zip(w4._PROJECTIONS, installs[0]["scale_lists"], strict=True):
        for layer, scale in zip(model.model.layers, scales, strict=True):
            projection = layer.get_submodule(role)
            assert scale is projection.weight_scale
            assert projection._rpu_linear_partition == partition
            assert projection.weight.dtype == (torch.int8 if quantization == "w8a16" else torch.uint8)
    assert model.lm_head.weight.dtype == model.model.embed_tokens.weight.dtype == torch.float16
    assert decoder._causal_lm_runtime_complete(model)
    _finalize(model.model)


@pytest.mark.parametrize("quantization", ["w8a16", "w4a16", "nvfp4"])
@pytest.mark.parametrize("fault", ["missing_scales", "foreign_scale", "mlp_lanes"])
def test_installer_rejects_a_wrong_llama_payload_before_native(monkeypatch, native_install, quantization, fault):
    model = _fake_model(6)
    _, _, weight_calls, installs = _host_install(monkeypatch, model, quantization)
    real_install = llama._install_causal_decoder_forward
    def corrupt(inner, **kwargs):
        if fault == "missing_scales":
            kwargs["scale_lists"] = kwargs["scale_lists"][:6]
        elif fault == "foreign_scale":
            kwargs["scale_lists"][-1][-1] = kwargs["scale_lists"][-1][-1].clone()
        else:
            # The six-lane MLP payload under a topology that claims four lanes.
            kwargs["topology"] = dataclasses.replace(kwargs["topology"], mlp_tp=4)
        return real_install(inner, **kwargs)
    monkeypatch.setattr(llama, "_install_causal_decoder_forward", corrupt)
    adapter = llama.LlamaAdapter(model, quantization=quantization)
    with model.lm_head.weight.fake_mode:
        with pytest.raises(ValueError):
            adapter.to_rpu()
    assert native_install["events"] == [] and not weight_calls


def test_public_loader_keeps_the_cold_recipe_and_refuses_another(monkeypatch):
    model = _fake_model(6)
    del model._rpu_execution
    monkeypatch.setattr(causal_lm.AutoConfig, "from_pretrained", lambda *_a, **_k: model.config)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: model)
    returned = causal_lm.RPUModelForCausalLM.from_pretrained(
        "unused", quantization="w8a16", rpu_execution={"model": {"num_cores": 6}})
    assert returned is model and model.to.__rpu_quantization__ == "w8a16"
    assert not getattr(model, "_rpu_swizzle_started", False)
    assert all(layer.mlp.down_proj.weight.dtype == torch.float16 for layer in model.model.layers)
    for quantization in (None, "nvfp4"):
        with pytest.raises(llama.UnsupportedModelError, match="cold-only"):
            llama.LlamaAdapter(model, **({"quantization": quantization} if quantization else {}))
