"""DINOv3 ViT-B on-install W8A16 / G32 W4A16: profiles, preflight, the real
conversion of one ViT-B block, the installer payload and the native source
contract at 4/6/8 cores. Host only; none of this is a board run or numerical
evidence for the RPU kernels.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.common import config as COMMON  # noqa: E402
from rpu_backend.adapters import dinov3  # noqa: E402
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError  # noqa: E402
from rpu_backend.quant import convert_dinov3 as quant  # noqa: E402
from rpu_backend.quant._common import quantize_linear_per_channel  # noqa: E402
from rpu_backend.quant.int4_pgrp_pack import quantize_int4_group_wise  # noqa: E402
from rpu_backend.tests.acceptance.test_dinov3_install_transaction import (  # noqa: E402,F401
    _model, native_install,
)
from test_dinov3_reduced_cores import (  # noqa: E402,F401
    _attention, _block, _calls, _finalize, _Layer, _VitB, reduced_install,
)
from test_qwen3_vl_awq_packing import inverse_rpu_scale, inverse_rpu_weight  # noqa: E402

NATIVE = ROOT / "src/fused/rpu_dinov3_vision_model.cpp"
RECIPES = {  # TOML recipe -> (profile variant, on-install quantization)
    "w8a16_vision": ("w8a16", "w8a16"),
    "w4a16_vision_rtn_g32": ("w4a16-g32", "w4a16"),
}
STORAGE = {"w8a16": torch.int8, "w4a16": torch.uint8}


# ------------------------------------------------------------- profiles and preflight


@pytest.mark.parametrize("recipe", RECIPES)
def test_quant_profiles_bind_exact_recipe_and_cores(recipe):
    variant, quantization = RECIPES[recipe]
    config = COMMON.load_config(ROOT / f"examples/configs/dinov3/vit_b/{recipe}.toml")
    profile = COMMON.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["id"] == f"dinov3-vit-b.{variant}.vision"
    assert config["example"]["registry_alias"] == "dinov3-vit-b"
    assert COMMON.install_quantization("dinov3", profile["id"]) == quantization
    assert COMMON.install_quantization("dinov3", "dinov3-vit-b.fp16.vision") is None
    assert limits["on_install_quantization"] == quantization
    assert limits.get("group_size") == (32 if quantization == "w4a16" else None)
    assert limits["linear_accumulation"] == "ACC16"
    assert tuple(limits[key] for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "head_dim")) == (768, 3072, 12, 12, 64)
    assert limits["allowed_num_cores"] == [4, 6, 8] and limits["formal_certification"] is False
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    assert config["rpu_execution"]["model"] == {"num_cores": 8}
    for cores in (4, 6, 8):
        COMMON._validate_execution({"model": {"num_cores": cores}}, profile=profile)
    for cores in (2, 5, 7):
        with pytest.raises((ValueError, COMMON.ConfigError)):
            COMMON._validate_execution({"model": {"num_cores": cores}}, profile=profile)
    # A reduced budget needs the profile's own recipe binding.
    unbound = copy.deepcopy(profile)
    unbound["input_envelope"]["limits"].pop("on_install_quantization")
    with pytest.raises(COMMON.ConfigError, match="registered ViT-B vision profile"):
        COMMON._validate_execution({"model": {"num_cores": 6}}, profile=unbound)


@pytest.mark.parametrize("quantization", ["w8a16", "w4a16"])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_preflight_admits_both_recipes_at_every_budget(quantization, cores):
    config = _model().config
    assert dinov3.DINOv3Adapter.preflight_quantization(
        config, {"model": {"num_cores": cores}}, quantization) == cores
    assert dinov3.DINOv3Adapter.preflight_quantization(config, None, quantization) == 8


@pytest.mark.parametrize("fault", ["nvfp4", "w4a16_lm_head", "prequantized", "cores5",
                                   "mlp1536", "vit_l"])
def test_preflight_rejects_before_weights(fault):
    config = _model().config
    quantization, execution = "w4a16", {"model": {"num_cores": 6}}
    if fault in ("nvfp4", "w4a16_lm_head"):
        quantization = fault
    elif fault == "prequantized":
        config.quantization_config = {"quant_method": "awq", "bits": 4}
    elif fault == "cores5":
        execution = {"model": {"num_cores": 5}}
    elif fault == "mlp1536":
        config.intermediate_size, execution = 1536, {}
    else:
        config.hidden_size, config.num_attention_heads = 1024, 16
        config.intermediate_size, config.num_hidden_layers, execution = 4096, 24, {}
    with pytest.raises((UnsupportedModelError, ValueError)):
        dinov3.DINOv3Adapter.preflight_quantization(config, execution, quantization)


class _Stop(Exception):
    pass


@pytest.mark.parametrize("prequantized", [False, True])
def test_runner_refuses_the_recipe_before_reading_the_checkpoint(monkeypatch, prequantized):
    import transformers
    from examples.runners.dinov3 import runner
    hf_config = _model().config
    if prequantized:
        hf_config.quantization_config = {"quant_method": "awq", "bits": 4}
    example = {"target": "dinov3", "profile_id": "dinov3-vit-b.w4a16-g32.vision",
               "registry_alias": "dinov3-vit-b"}
    execution = {"model": {"num_cores": 6}}
    monkeypatch.setattr(runner, "text_context", lambda _config: (
        example, {"images": []}, "dinov3", "/cache/dinov3", execution, None))
    monkeypatch.setattr("examples.common.generation._image_inputs", lambda _inputs: [])
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: hf_config)

    def processor(*_a, **_k):
        raise _Stop

    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", processor)
    monkeypatch.setattr(transformers.DINOv3ViTModel, "from_pretrained",
                        lambda *_a, **_k: pytest.fail("weights were read"))
    # The admitted recipe reaches the processor; the rejected one stops first.
    with pytest.raises(UnsupportedModelError if prequantized else _Stop):
        runner.prepare({}, SimpleNamespace(owner=None))


# ------------------------------------------------------------------- adapter binding


def _vit_b_model(cores):
    model = _VitB(cores).half()
    del model._rpu_execution
    return model


def test_recipe_binds_cold_once_per_model():
    model = _vit_b_model(6)
    dinov3.DINOv3Adapter(model, rpu_execution={"model": {"num_cores": 6}}, quantization="w8a16")
    assert model.to.__rpu_quantization__ == "w8a16"
    dinov3.DINOv3Adapter(model, quantization="w8a16")
    for other in ("w4a16", None):
        with pytest.raises(UnsupportedModelError, match="cold-only"):
            dinov3.DINOv3Adapter(model, quantization=other)
    fp16 = _vit_b_model(8)
    dinov3.DINOv3Adapter(fp16)
    with pytest.raises(UnsupportedModelError, match="cold-only"):
        dinov3.DINOv3Adapter(fp16, quantization="w4a16")


def test_adapter_requires_the_original_cpu_fp16_projections():
    # Weight-less doubles and FP32 Linears are not the FP16 checkpoint.
    with pytest.raises(UnsupportedModelError, match="original unswizzled CPU FP16"):
        dinov3.DINOv3Adapter(_model(), quantization="w8a16")
    with pytest.raises(UnsupportedModelError, match="original unswizzled CPU FP16"):
        dinov3.DINOv3Adapter(_VitB(8).float(), quantization="w4a16")


def test_install_rechecks_the_source_and_binary_before_the_swizzle(monkeypatch):
    model = _vit_b_model(8)
    adapter = dinov3.DINOv3Adapter(model, quantization="w4a16")
    monkeypatch.setattr(dinov3, "torch", SimpleNamespace(ops=SimpleNamespace(rpu=SimpleNamespace())))
    with pytest.raises(RPUBackendError, match="dinov3_vision_set_weight_scales; rebuild"):
        adapter._to_rpu_locked()
    monkeypatch.undo()
    monkeypatch.setattr(torch.ops.rpu, "dinov3_vision_set_weight_scales", lambda *_a: None,
                        raising=False)
    down = model.layer[0].mlp.down_proj
    down.weight = nn.Parameter(down.weight.float(), requires_grad=False)
    with pytest.raises(UnsupportedModelError, match="layer0.mlp.down_proj"):
        adapter._to_rpu_locked()
    assert not {"_rpu_swizzle_started", "_rpu_dinov3_quantization"} & set(vars(model))


# ------------------------------------------------------------------ CPU conversion


def _natural(weight, partition, cores):
    """Undo the half/byte-width swizzle of transform_linear_weight: natural [N, K]."""
    n, k = weight.shape
    width = 32 // weight.element_size()
    if partition == 1:
        view = weight.view(n // 16 // cores, k // width, cores, 16, width)
        return view.permute(2, 0, 3, 1, 4).reshape(n, k)
    view = weight.view(n // 16, k // width // cores, cores, 16, width)
    return view.permute(0, 3, 2, 1, 4).reshape(n, k)


def _dequantized(linear, partition, cores, quantization):
    """Natural-order FP64 weight a converted leaf computes with."""
    if quantization == "w8a16":
        values = _natural(linear.weight, partition, cores)
        return values.double() * linear.weight_scale.double()[:, None]
    n, k = linear.out_features, linear.in_features
    values = inverse_rpu_weight(linear.weight, n, k, partition, cores)
    scales = inverse_rpu_scale(linear.weight_scale, n, k, partition, cores)
    return values.double() * scales.double().repeat_interleave(32, dim=1)


def _reference(weight, quantization):
    """The original FP16 weight through the recipe's own quantize/dequantize."""
    if quantization == "w8a16":
        codes, scale = quantize_linear_per_channel(weight, bits=8)
        return codes.double() * scale.double()[:, None]
    codes, scale, _ = quantize_int4_group_wise(weight, 32)
    return codes.double() * scale.double().t().repeat_interleave(32, dim=1)


def _block_outputs(x, weights, biases, heads):
    q = x @ weights["attention.q_proj"].T + biases["attention.q_proj"]
    k = x @ weights["attention.k_proj"].T
    v = x @ weights["attention.v_proj"].T + biases["attention.v_proj"]
    attn = _attention(q, k, v, heads) @ weights["attention.o_proj"].T + biases["attention.o_proj"]
    up = x @ weights["mlp.up_proj"].T + biases["mlp.up_proj"]
    return attn, F.gelu(up) @ weights["mlp.down_proj"].T + biases["mlp.down_proj"]


@pytest.mark.parametrize("quantization", ["w8a16", "w4a16"])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_converted_block_computes_the_recipe_weights(quantization, cores):
    torch.manual_seed(cores)
    model = _VitB(cores).half()
    block = model.layer[0]
    original = {name: param.detach().clone() for name, param in block.named_parameters()}
    model._rpu_dinov3_quantization = quantization
    dinov3._convert_dinov3_weights_for_rpu(model)
    assert model._rpu_vision_num_cores == cores

    heads = dinov3._padded_num_heads(12, cores)
    weights, biases, expected_weights, expected_biases = {}, {}, {}, {}
    for role, partition in quant.PROJECTIONS:
        linear = block.get_submodule(role)
        n, k = linear.out_features, linear.in_features
        scale = linear.weight_scale
        assert linear.weight.dtype == STORAGE[quantization] and not linear.weight.requires_grad
        assert (linear._rpu_linear_partition, linear._rpu_linear_num_cores) == (partition, cores)
        # A buffer moves with model.to('rpu'); it is not a checkpoint key.
        assert "weight_scale" in dict(linear.named_buffers())
        assert "weight_scale" not in linear.state_dict()
        assert scale.dtype == torch.float16 and scale.is_contiguous()
        if quantization == "w8a16":
            assert linear.weight.shape == (n, k) and scale.shape == (n,)
        else:
            # The packed shape and scale size the native set_weight_scales admits.
            local_n, local_k = (n // cores, k) if partition == 1 else (n, k // cores)
            assert linear.weight.shape == (n, k // 2) and scale.size(0) == 32
            assert scale.numel() == (math.ceil(local_k // 32 / 4) * math.ceil(local_n / 64)
                                     * cores * 256)
        weights[role] = _dequantized(linear, partition, cores, quantization)
        biases[role] = 0.0 if linear.bias is None else linear.bias.double()
        expected_weights[role] = _reference(original[f"{role}.weight"], quantization)
        source_bias = original.get(f"{role}.bias")
        expected_biases[role] = 0.0 if source_bias is None else source_bias.double()
    assert weights["attention.q_proj"].shape == (heads * 64, 768)
    # Eight cores pad 12 heads to 16 before quantization: four zero heads.
    zero_rows = int((weights["attention.q_proj"].abs().sum(1) == 0).sum())
    assert zero_rows == (256 if cores == 8 else 0)

    x = torch.randn(21, 768, dtype=torch.float64)
    actual = _block_outputs(x, weights, biases, heads)
    expected = _block_outputs(x, expected_weights, expected_biases, 12)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("quantization, fault", [
    ("w8a16", "dtype"), ("w8a16", "scaled"), ("w8a16", "swizzled"),
    ("w4a16", "dtype"), ("w4a16", "scaled"), ("w4a16", "swizzled"), ("w4a16", "shape"),
])
def test_converter_checks_every_projection_before_replacing_any(quantization, fault):
    model = _VitB(6).half()
    model.layer.append(_Layer().half())
    last = model.layer[-1].mlp
    if fault == "dtype":
        last.down_proj.weight = nn.Parameter(last.down_proj.weight.float(), requires_grad=False)
    elif fault == "scaled":
        last.down_proj.register_buffer("weight_scale", torch.ones(768, dtype=torch.float16))
    elif fault == "swizzled":
        last.down_proj._rpu_linear_partition = 0
    else:
        # K=3040 breaks the six-core int4 row rule (K % 384).
        last.down_proj = nn.Linear(3040, 768).half()
    with pytest.raises(ValueError, match="layer1.mlp.down_proj"):
        quant.quantize_dinov3_projections_(model.layer, quantization, num_cores=6)
    for layer in model.layer:
        for role, _ in quant.PROJECTIONS:
            linear = layer.get_submodule(role)
            assert linear.weight.is_floating_point()
            assert linear is last.down_proj or not hasattr(linear, "weight_scale")


# ------------------------------------------------------------------- native install


def _quantized_install_model(quantization, cores):
    model = _model()
    layer = model.layer[0]
    for role, _ in quant.PROJECTIONS:
        projection = quant._get(layer, role)
        if quantization == "w4a16":
            projection.weight = torch.zeros(64, 32, dtype=torch.uint8)
            projection.weight_scale = torch.ones(32, 8, dtype=torch.float16)
        else:
            projection.weight = torch.zeros(64, 64, dtype=torch.int8)
            projection.weight_scale = torch.ones(64, dtype=torch.float16)
    model._rpu_dinov3_quantization = quantization
    if cores != 8:
        model._rpu_vision_num_cores = cores
    return model


@pytest.fixture
def scale_calls(reduced_install, monkeypatch):
    calls = []

    def set_scales(handle, *scales):
        # After the weights and topology, before the envelope plans the chunk.
        assert reduced_install["events"][-1] == ("set_weights", handle)
        assert not reduced_install["envelopes"]
        calls.append((handle, scales))

    monkeypatch.setattr(dinov3.torch.ops.rpu, "dinov3_vision_set_weight_scales", set_scales,
                        raising=False)
    return calls


@pytest.mark.parametrize("quantization", ["w8a16", "w4a16", None])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_installer_hands_each_projection_scale_to_native(
        reduced_install, scale_calls, quantization, cores):
    if quantization is None:
        model = _model()
        if cores != 8:
            model._rpu_vision_num_cores = cores
    else:
        model = _quantized_install_model(quantization, cores)
    try:
        dinov3.install_dinov3_vision_for_rpu(model)
        keepalive = model._rpu_dinov3_retirement_state.keepalive[0]
        assert reduced_install["envelopes"] == [(1, 256, 0)]
        if quantization is None:
            assert scale_calls == [] and len(keepalive) == 4
            return
        (handle, scales), = scale_calls
        assert handle == 1 and len(scales) == 6
        layer = model.layer[0]
        for (role, _), role_scales in zip(quant.PROJECTIONS, scales):
            assert len(role_scales) == 1 and role_scales[0] is quant._get(layer, role).weight_scale
        assert len(keepalive) == 5 and all(a is b for a, b in zip(keepalive[4], scales))
    finally:
        _finalize(model)


@pytest.mark.parametrize("fault", ["missing", "shape", "dtype", "storage", "recipe"])
@pytest.mark.parametrize("quantization", ["w8a16", "w4a16"])
def test_installer_rejects_scale_payload_before_native(reduced_install, scale_calls,
                                                       quantization, fault):
    model = _quantized_install_model(quantization, 6)
    down = model.layer[0].mlp.down_proj
    if fault == "missing":
        down.weight_scale = None
    elif fault == "shape":
        down.weight_scale = (torch.ones(16, 8, dtype=torch.float16) if quantization == "w4a16"
                             else torch.ones(80, dtype=torch.float16))
    elif fault == "dtype":
        down.weight_scale = down.weight_scale.float()
    elif fault == "storage":
        down.weight = down.weight.view(torch.int8 if quantization == "w4a16" else torch.uint8)
    else:
        model._rpu_dinov3_quantization = "nvfp4"
    with pytest.raises(ValueError, match="own FP16 scale"):
        dinov3.install_dinov3_vision_for_rpu(model)
    assert reduced_install["events"] == [] and reduced_install["alive"] == set()
    assert scale_calls == [] and "_rpu_dinov3_handle" not in vars(model)


# ------------------------------------------------------------------ native contract


def test_native_scales_are_validated_whole_and_reach_the_acc16_launcher():
    source = NATIVE.read_text()
    scales = _block(source, "void set_weight_scales(")
    for condition in ("get_last_resolved_chunk_size() == 0", "n_layers == 12",
                      "projection_type() != at::kHalf", "!use_acc32_",
                      "!standalone_compatibility_", "!has_output_mlp_",
                      "!scale.is_same(weight)", "scale.size(0) == 32",
                      "scale.dim() == 1 && scale.size(0) == n"):
        assert condition in scales, condition
    # Every pair is checked before the first scale is published.
    assert scales.rindex("TORCH_CHECK(") < scales.index("w.q_s = retain(w.q_w, q[layer]);") < scales.index(
        "invalidate_model_state();")
    weights = _block(source, "void set_weights(")
    assert "N == 12 && hidden_size == 768 && intermediate_size == 3072" in weights
    assert "projection_type() == at::kHalf" in _block(source, "void set_acc32(bool enabled)")
    assert all(args[-1] == "scale"
               for args in _calls(source, "rpu_launch_linear_spm_to_spm_acc16_kernel"))
    pairs = sorted((args[1], args[2]) for args in _calls(source, "launch_linear"))
    roles = ("q", "k", "v", "o", "up", "down")
    assert [pair for pair in pairs if pair[0].startswith("lw.")] == sorted(
        (f"lw.{role}_w", f"lw.{role}_s") for role in roles)
    assert all(scale == "at::Tensor()" for weight, scale in pairs if not weight.startswith("lw."))
    for role in roles:
        assert f"&weights.{role}_s" in source  # graph identity covers the scales


def test_packed_w4_adds_the_full_row_bias_after_each_all_reduce():
    source = NATIVE.read_text()
    assert "const int row_bias_cores = projection_type() == at::kByte ? num_cores() : 1;" in source
    helper = _calls(_block(source, "void emit_reduced_row_bias("),
                    "rpu_launch_eltwise_binary_1xC_NxC_spm_kernel")
    assert helper == [["bias", "rows", "rows", "seq_len", "hidden_size()", "c10::Half(1.0f)",
                       "ValuOpType::ADD", "false", "num_cores()"]]
    build = _block(source, "void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk)")
    assert "if (weight.scalar_type() == at::kByte && partition == 0) bias = 0;" in build
    assert _calls(build, "emit_reduced_row_bias") == [
        ['layer_addr(layer_idx, 0, "o_bias")', 'addr(0, "input_norm")', "seq_len"],
        ['layer_addr(layer_idx, 0, "down_bias")', 'addr(0, "residual1")', "seq_len"]]
    for rows, bias, gamma in (("input_norm", "o_bias", "gamma_attn"),
                              ("residual1", "down_bias", "gamma_mlp")):
        reduce = build.index(f'addr(0, "zero_residual"), addr(0, "{rows}")')
        added = build.index(f'layer_addr(layer_idx, 0, "{bias}"), addr(0, "{rows}")')
        assert reduce < added < build.index(f'layer_addr(layer_idx, 0, "{gamma}")')


def test_scale_op_is_declared_and_registered():
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    assert "void rpu_dinov3_vision_set_weight_scales(" in decls
    assert 'm.def("dinov3_vision_set_weight_scales(int handle, Tensor[] q_scales, ' in registrations
    assert ('m.impl("dinov3_vision_set_weight_scales", '
            "TORCH_FN(rpu_dinov3_vision_set_weight_scales));") in registrations
    assert "->set_weight_scales(q_scales, k_scales, v_scales, o_scales, up_scales, down_scales);" \
        in NATIVE.read_text()
