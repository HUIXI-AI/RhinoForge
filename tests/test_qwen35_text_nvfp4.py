"""Dense Qwen3.5 text NVFP4 recipe: profiles, admission, widths and installer payload; no board."""
from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters import qwen3_5 as adapter
from rpu_backend.adapters.qwen3_5 import cores, quant_scope as qs, text, weights
from rpu_backend.quant.nvfp4_pack import quantize_nvfp4, swizzle_nvfp4_scales, swizzle_pack_nvfp4
from test_qwen35_core_profiles import _config as dense_config
from test_qwen35_dense27_profiles import runtime_config27
from test_qwen35_legacy_finalized_upload import _install, _StopAtNativeWeights, host_device

ROOT = Path(__file__).resolve().parents[1]
# hidden, intermediate, layers, query heads, KV heads, GDN value heads
SHAPES = {
    "0.8": (1024, 3584, 24, 8, 2, 16), "2": (2048, 6144, 24, 8, 2, 16),
    "4": (2560, 9216, 32, 16, 4, 32), "9": (4096, 12288, 32, 16, 4, 32),
    "27": (5120, 17408, 64, 24, 4, 48),
}
SIX_CORE_MLP = {"0.8": 3840, "2": 6144, "4": 9216, "9": 12288, "27": 17664}
ALIASES = {"0.8": "qwen3_5-0.8b", "2": "qwen3_5-2b", "4": "qwen3_5-4b",
           "9": "qwen3_5-9b", "27": "qwen3_5-27b-fp16"}


@pytest.fixture
def common():
    return importlib.import_module("examples.common.config")


def _source(size):
    return runtime_config27(qs.FP16) if size == "27" else dense_config(float(size) if size == "0.8" else int(size))


def _execution(cores_count, **extra):
    return {"model": {"num_cores": cores_count}, **extra}


@pytest.mark.parametrize("size", SHAPES)
def test_nvfp4_profiles_bind_exact_recipe_geometry_and_cores(common, size):
    path = ROOT / f"examples/configs/qwen3_5/text/{size.replace('.', '_')}b/nvfp4_text.toml"
    config = common.load_config(path)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["id"] == f"qwen3_5-{size}b.nvfp4.text-controlled"
    assert profile["asset_alias"] == config["example"]["registry_alias"] == ALIASES[size]
    assert profile["precision"] == config["example"]["dtype"] == limits["on_install_quantization"] == "nvfp4"
    assert common.install_quantization("qwen3_5", profile["id"]) == "nvfp4"
    assert tuple(limits[key] for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads")) == SHAPES[size][:5]
    assert (limits["head_dim"], limits["vocab_size"]) == (256, 248320)
    assert {limits["quantization"][key] for key in (
        "lm_head", "embedding", "norms", "gdn_in_proj_a_b")} == {"FP16"}
    assert limits["allowed_num_cores"] == [4, 6, 8]
    assert limits["batch_size"] == 1 and limits["formal_certification"] is False
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    request = {"model": {"num_cores": 8}, "prefill": {"linear_acc32": True}}
    common._validate_execution(request, profile=profile)
    for cores_count in (4, 6):
        request["model"]["num_cores"] = cores_count
        common._validate_execution(request, profile=profile)
    for cores_count in (2, 5, 7):
        request["model"]["num_cores"] = cores_count
        with pytest.raises((ValueError, common.ConfigError)):
            common._validate_execution(request, profile=profile)


def test_fp16_profiles_keep_converted_weight_loading(common):
    assert common.install_quantization("qwen3_5", "qwen3_5-2b.fp16.text") is None


def test_policy_quantizes_every_text_projection_except_gdn_b_a():
    layer_types = [qs.LINEAR_ATTENTION] * 3 + [qs.FULL_ATTENTION]
    specs = qs.resolve_quant_specs(qs.is_full_from_layer_types(layer_types), qs.nvfp4_text_quant_config())
    fp16 = {name for name, spec in specs.items() if spec.method == qs.FP16}
    assert fp16 == {f"layers.{i}.linear_attn.in_proj_{role}" for i in range(3) for role in "ab"}
    assert {spec.method for name, spec in specs.items() if name not in fp16} == {qs.NVFP4}
    assert qs.nvfp4_text_quant_config() is not qs.nvfp4_text_quant_config()


@pytest.mark.parametrize("fault", ["altered", "foreign_v2", "v1_default"])
def test_nvfp4_requires_the_exact_install_policy(fault):
    policy = qs.nvfp4_text_quant_config()
    if fault == "altered":
        del policy["projection_overrides"]["linear_attn.in_proj_a"]
    elif fault == "foreign_v2":
        policy = {"format_version": 2, "default": {"method": qs.NVFP4}}
    else:
        policy = {"format_version": 1, "method": qs.NVFP4}
    with pytest.raises(ValueError, match="NVFP4"):
        qs.resolve_quant_specs([0, 0, 0, 1], policy)


@pytest.mark.parametrize("size", SHAPES)
@pytest.mark.parametrize("cores_count", [4, 6, 8])
def test_preflight_admits_every_dense_size_and_core_budget(size, cores_count):
    config = _source(size)
    adapter.Qwen3_5Adapter.preflight_quantization(config, _execution(cores_count), "nvfp4")
    assert getattr(config, "quant_config", None) is None


@pytest.mark.parametrize("fault", [
    "recipe", "quantized", "text_quantized", "vision", "moe", "adaptive", "geometry", "cores5"])
def test_preflight_refuses_before_any_weight_load(fault):
    config = dense_config(2)
    tc = config.text_config
    execution, recipe = _execution(5 if fault == "cores5" else 8), "nvfp4"
    if fault == "recipe":
        recipe = "w4a16"
    elif fault == "quantized":
        config.quant_config = {"format_version": 1, "method": qs.W8A16}
    elif fault == "text_quantized":
        tc.quantization_config = {"quant_method": "awq", "bits": 4}
    elif fault == "vision":
        execution["vision"] = {}
    elif fault == "moe":
        tc.num_experts = 8
    elif fault == "adaptive":
        tc.adaptive_mode = "adaLN"
    elif fault == "geometry":
        tc.intermediate_size = 4096
    with pytest.raises(adapter.UnsupportedModelError):
        adapter.Qwen3_5Adapter.preflight_quantization(config, execution, recipe)


def test_binding_stamps_once_and_reuses_the_cold_recipe():
    model = NS(config=dense_config(2))
    adapter._bind_install_quantization(model, None, _execution(8))
    assert getattr(model.config, "quant_config", None) is None
    adapter._bind_install_quantization(model, "nvfp4", _execution(6))
    assert model.config.quant_config == qs.nvfp4_text_quant_config()
    assert model._qwen3_5_install_quantization == "nvfp4"
    # The runner wraps the loaded model again without a recipe.
    adapter._bind_install_quantization(model, None, None)
    adapter._bind_install_quantization(model, "nvfp4", None)
    with pytest.raises(adapter.UnsupportedModelError, match="cold-only"):
        adapter._bind_install_quantization(model, "w8a16", None)
    installed = NS(config=dense_config(2), _rpu_swizzled=True)
    with pytest.raises(adapter.UnsupportedModelError, match="before installation"):
        adapter._bind_install_quantization(installed, "nvfp4", _execution(8))
    assert getattr(installed.config, "quant_config", None) is None


@pytest.mark.parametrize("size", SHAPES)
def test_six_core_mlp_uses_the_packed_stripe_width(size):
    stamped = deepcopy(_source(size))
    stamped.quant_config = qs.nvfp4_text_quant_config()
    if size == "27":
        assert qs.validate_modern_27b_profile(stamped)
    for cores_count, expected in ((4, SHAPES[size][1]), (6, SIX_CORE_MLP[size]), (8, SHAPES[size][1])):
        topology = cores.validate_core_profile(stamped, _execution(cores_count))
        assert cores.text_mlp_intermediate_size(stamped, topology) == expected


@pytest.mark.parametrize("role,shape,padded", [
    ("gate_proj", (3584, 64), (3840, 64)), ("down_proj", (64, 17408), (64, 17664))])
def test_fp16_source_is_zero_padded_before_quantization(role, shape, padded):
    source = torch.randn(shape).half()
    logical, physical = (shape[0], padded[0]) if role == "gate_proj" else (shape[1], padded[1])
    out = weights.pad_quant_mlp_projection(
        source, projection_name=role, logical_size=logical, physical_size=physical)
    assert out.dtype == torch.float16 and tuple(out.shape) == padded
    assert torch.equal(out[:shape[0], :shape[1]], source)
    assert int(torch.count_nonzero(out)) == int(torch.count_nonzero(source))


def test_nine_b_nvfp4_keeps_the_one_shot_prefill_lifecycle():
    source = dense_config(9)
    stamped = deepcopy(source)
    stamped.quant_config = qs.nvfp4_text_quant_config()
    execution = _execution(8)
    assert text._public_retained_prefill(source, execution, public_owner=True)
    assert not text._public_retained_prefill(stamped, execution, public_owner=True)


def _tiny_fp16_model():
    """Smallest two-layer text satisfying every NVFP4 swizzle rule at 4 and 8 cores.

    Admission of this geometry is stubbed; installation, quantization,
    swizzling and table assembly are the production implementations.
    """
    generator = torch.Generator().manual_seed(29)
    config = NS(hidden_size=512, intermediate_size=512, num_hidden_layers=2,
                num_attention_heads=8, num_key_value_heads=8, head_dim=64,
                linear_num_key_heads=8, linear_num_value_heads=16,
                linear_key_head_dim=128, linear_value_head_dim=128,
                linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
                layer_types=["linear_attention", "full_attention"],
                quant_config=qs.nvfp4_text_quant_config())

    def linear(n, k, gain):
        module = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
        weight = torch.randn(n, k, generator=generator) * gain
        module.weight = torch.nn.Parameter(weight.half(), requires_grad=False)
        return module

    def norm(n):
        module = torch.nn.Module()
        module.weight = torch.nn.Parameter(torch.rand(n, generator=generator).half())
        return module

    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList()
    inner.embed_tokens = torch.nn.Embedding(16, 512, dtype=torch.float16)
    inner.norm = norm(512)
    for index, full in enumerate((False, True)):
        gain = 0.02 * (index + 1)
        layer = torch.nn.Module()
        layer.input_layernorm, layer.post_attention_layernorm = norm(512), norm(512)
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_proj = linear(512, 512, gain)
        layer.mlp.up_proj = linear(512, 512, 2 * gain)
        layer.mlp.down_proj = linear(512, 512, 3 * gain)
        if full:
            attn = layer.self_attn = torch.nn.Module()
            attn.q_proj = linear(1024, 512, gain)
            attn.k_proj, attn.v_proj = linear(512, 512, 2 * gain), linear(512, 512, 3 * gain)
            attn.o_proj = linear(512, 512, 4 * gain)
            attn.q_norm, attn.k_norm = norm(64), norm(64)
        else:
            attn = layer.linear_attn = torch.nn.Module()
            attn.in_proj_qkv = linear(4096, 512, gain)
            attn.in_proj_z = linear(2048, 512, 2 * gain)
            attn.in_proj_b, attn.in_proj_a = linear(16, 512, gain), linear(16, 512, gain)
            attn.out_proj = linear(512, 2048, 3 * gain)
            attn.conv1d = torch.nn.Conv1d(4096, 4096, 4, groups=4096,
                                          bias=False, dtype=torch.float16)
            attn.norm = norm(128)
            attn.A_log = torch.nn.Parameter(torch.rand(16, generator=generator).half())
            attn.dt_bias = torch.nn.Parameter(torch.rand(16, generator=generator).half())
        inner.layers.append(layer)
    return inner, config


# Table family -> set_weights weight/scale argument, partition, layer, source.
_OWNERS = {
    "q": (1, 43, 1, 1, lambda m: text.split_q_gate(m.layers[1].self_attn.q_proj.weight, 8, 64)[0]),
    "k": (2, 44, 1, 1, lambda m: m.layers[1].self_attn.k_proj.weight),
    "v": (3, 45, 1, 1, lambda m: m.layers[1].self_attn.v_proj.weight),
    "o": (4, 46, 0, 1, lambda m: m.layers[1].self_attn.o_proj.weight),
    "attn_gate": (7, 47, 1, 1, lambda m: text.split_q_gate(m.layers[1].self_attn.q_proj.weight, 8, 64)[1]),
    "gdn_q": (35, 53, 1, 0, lambda m: m.layers[0].linear_attn.in_proj_qkv.weight[:1024]),
    "gdn_k": (36, 54, 1, 0, lambda m: m.layers[0].linear_attn.in_proj_qkv.weight[1024:2048]),
    "gdn_v": (37, 55, 1, 0, lambda m: m.layers[0].linear_attn.in_proj_qkv.weight[2048:]),
    "gdn_z": (25, 51, 1, 0, lambda m: m.layers[0].linear_attn.in_proj_z.weight),
    "gdn_out": (26, 52, 0, 0, lambda m: m.layers[0].linear_attn.out_proj.weight),
}
for _layer in (0, 1):
    for _family, _w, _s, _partition, _role in (("gate", 10, 48, 1, "gate_proj"),
                                               ("up", 11, 49, 1, "up_proj"),
                                               ("down", 12, 50, 0, "down_proj")):
        _OWNERS[f"{_family}.{_layer}"] = (
            _w, _s, _partition, _layer,
            lambda m, layer=_layer, role=_role: m.layers[layer].mlp.get_submodule(role).weight)


@pytest.mark.parametrize("cores_count", [8, 4])
def test_installer_quantizes_each_native_owner_and_orders_tensor_scales(
    monkeypatch, host_device, cores_count,
):
    inner, config = _tiny_fp16_model()
    sources = {owner: entry[-1](inner).detach().clone() for owner, entry in _OWNERS.items()}
    topology = cores.core_topology(_execution(cores_count))
    monkeypatch.setattr(cores, "validate_core_profile", lambda *a: topology)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_execution_cores", lambda *a: None, raising=False)
    payload = {}

    def capture(*args, **kwargs):
        payload.update(args=args, kwargs=kwargs)
        raise _StopAtNativeWeights

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", capture, raising=False)
    with pytest.raises(_StopAtNativeWeights):
        _install(inner, config)
    args, tables = payload["args"], payload["kwargs"]["nvfp4_tensor_scales"]
    families = ("q", "k", "v", "o", "attn_gate", "gdn_q", "gdn_k", "gdn_v",
                "gdn_z", "gdn_out", "gate", "up", "down")
    assert len(tables) == len(families)
    assert all(t.dtype == torch.float32 and t.shape == (8,) and t.is_contiguous() for t in tables)
    present = set()
    for owner, (w_index, s_index, partition, layer, _) in _OWNERS.items():
        family = owner.split(".")[0]
        source = sources[owner]
        codes, block, tensor_scale = quantize_nvfp4(source)
        n, k = codes.shape
        lanes = topology.mlp_tp if family in ("gate", "up", "down") else topology.attn_tp
        weight, scale = args[w_index][layer], args[s_index][layer]
        assert weight.dtype == torch.uint8 and torch.equal(
            weight, swizzle_pack_nvfp4(codes, partition, lanes).reshape(n, k // 2)), owner
        assert scale.dtype == torch.uint8 and tuple(scale.shape) == (k // 16, n)
        assert torch.equal(scale, swizzle_nvfp4_scales(block, partition, lanes)), owner
        table = tables[families.index(family)]
        assert table[layer] == tensor_scale.to(torch.float32), owner
        present.add((family, layer))
    # Layers without a family and the padding to eight hold the identity scale.
    for family_index, family in enumerate(families):
        for layer in range(8):
            if (family, layer) not in present:
                assert tables[family_index][layer] == 1.0, (family, layer)
    # GDN b/a stay FP16 with no scale.
    for index, s_index in ((41, 56), (42, 57)):
        assert args[index][0].dtype == torch.float16 and args[s_index][0].numel() == 0


def test_installer_refuses_a_device_resident_nvfp4_source(monkeypatch, host_device):
    inner, config = _tiny_fp16_model()
    inner.layers[1].mlp.down_proj.weight = torch.nn.Parameter(
        torch.empty((512, 512), dtype=torch.float16, device="meta"), requires_grad=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights",
                        lambda *a, **k: pytest.fail("reached native weights"), raising=False)
    with pytest.raises(adapter.UnsupportedModelError, match="original CPU source"):
        _install(inner, config)
