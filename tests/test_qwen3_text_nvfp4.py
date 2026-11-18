"""Dense Qwen3 text NVFP4 recipe: profiles, preflight, packing and installer payload; no board."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from rpu_backend.adapters import qwen3
from rpu_backend.api import causal_lm
from rpu_backend.quant import convert_qwen3_nvfp4 as nvfp4
from rpu_backend.quant.convert_qwen3_w4a16 import _PROJECTIONS
from rpu_backend.quant.nvfp4_pack import quantize_nvfp4, swizzle_nvfp4_scales, swizzle_pack_nvfp4
from rpu_backend.quant.qwen3_profiles import qwen3_dense_quant_profile
from rpu_backend.runtime import decoder
from rpu_backend.runtime.topology import (
    decoder_nvfp4_mlp_intermediate_size, resolve_decoder_topology,
)
from rpu_backend.tests.acceptance.test_decoder_install_transaction import (
    native_install, _FakeTensor, _finalize,
)
from rpu_backend.tests.acceptance.test_qwen3_w4_public_install import (
    _dense_config, _host_install, _model,
)

ROOT = Path(__file__).resolve().parents[1]
SHAPES = {
    "0.6": (1024, 3072, 28, 16), "1.7": (2048, 6144, 28, 16),
    "4": (2560, 9728, 36, 32), "8": (4096, 12288, 36, 32),
    "14": (5120, 17408, 40, 40),
}


@pytest.fixture
def common():
    return importlib.import_module("examples.common.config")


def _config(size, **changes):
    h, i, layers, heads = SHAPES.get(size, (5120, 25600, 64, 64))
    values = dict(architectures=["Qwen3ForCausalLM"], hidden_size=h, intermediate_size=i,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=8,
        head_dim=128, vocab_size=151936, tie_word_embeddings=h < 4096)
    values.update(changes)
    return Qwen3Config(**values)


@pytest.mark.parametrize("size", SHAPES)
def test_nvfp4_profiles_bind_exact_recipe_geometry_and_cores(common, size):
    path = ROOT / f"examples/configs/qwen3/text/{size.replace('.', '_')}b/nvfp4_text.toml"
    config = common.load_config(path)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["id"] == f"qwen3-{size}b.nvfp4.text"
    assert profile["asset_alias"] == config["example"]["registry_alias"] == f"qwen3-{size}b"
    assert profile["precision"] == config["example"]["dtype"] == limits["on_install_quantization"] == "nvfp4"
    assert tuple(limits[key] for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"
    )) == SHAPES[size]
    assert (limits["num_key_value_heads"], limits["head_dim"], limits["vocab_size"]) == (8, 128, 151936)
    assert {limits["quantization"][key] for key in ("lm_head", "embedding", "norms")} == {"FP16"}
    assert limits["allowed_num_cores"] == [4, 6, 8]
    assert limits["batch_size"] == 1 and limits["formal_certification"] is False
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    assert (config["input"]["prefill_tokens"], config["input"]["decode_steps"]) == (128, 7)
    assert (config["run"]["warmup"], config["run"]["runs"]) == (2, 5)
    request = {"model": {"num_cores": 8}, "prefill": {"linear_acc32": True}}
    common._validate_execution(request, profile=profile)
    for cores in (4, 6):
        request["model"]["num_cores"] = cores
        common._validate_execution(request, profile=profile)
    for cores in (2, 5, 7):
        request["model"]["num_cores"] = cores
        with pytest.raises((ValueError, common.ConfigError)):
            common._validate_execution(request, profile=profile)


@pytest.mark.parametrize("size", SHAPES)
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_preflight_admits_native_nvfp4_sizes_and_core_budgets(size, cores):
    execution = {"model": {"num_cores": cores}}
    profile = qwen3.Qwen3Adapter.preflight_quantization(_config(size), execution, "nvfp4")
    assert (profile.hidden_size, profile.num_layers) == (SHAPES[size][0], SHAPES[size][2])


@pytest.mark.parametrize("fault", ["32b", "32b_tp6", "awq", "w8", "tp5", "untied"])
def test_public_nvfp4_rejects_before_weight_load(monkeypatch, fault):
    config = _config("32" if fault.startswith("32b") else "0.6")
    execution = {"model": {"num_cores": {"32b_tp6": 6, "tp5": 5}.get(fault, 8)}}
    if fault == "awq":
        config.quantization_config = {"quant_method": "awq", "bits": 4}
    elif fault == "w8":
        config.quant_config = {"method": "w8a16"}
    elif fault == "untied":
        config.tie_word_embeddings = False
    monkeypatch.setattr(causal_lm.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: pytest.fail("invalid NVFP4 source loaded weights"))
    with pytest.raises((qwen3.UnsupportedModelError, ValueError)):
        causal_lm.RPUModelForCausalLM.from_pretrained(
            "source", quantization="nvfp4", rpu_execution=execution)


def _tiny_model(layers=2):
    # Smallest projection widths satisfying both swizzle rules at attn 4 / MLP 8.
    shapes = {"self_attn.q_proj": (512, 512), "self_attn.k_proj": (128, 512),
              "self_attn.v_proj": (128, 512), "self_attn.o_proj": (512, 512),
              "mlp.gate_proj": (512, 512), "mlp.up_proj": (512, 512), "mlp.down_proj": (512, 1024)}
    generator = torch.Generator().manual_seed(5)
    blocks = torch.nn.ModuleList()
    for index in range(layers):
        layer = torch.nn.Module()
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        for offset, (role, (n, k)) in enumerate(shapes.items()):
            linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
            weight = torch.randn(n, k, generator=generator) * (index + 1) * (offset + 1) / 16
            linear.weight = torch.nn.Parameter(weight.half(), requires_grad=False)
            parent, name = role.split(".")
            setattr(getattr(layer, parent), name, linear)
        blocks.append(layer)
    return SimpleNamespace(model=SimpleNamespace(layers=blocks))


def test_converter_packs_every_projection_and_orders_tensor_scales():
    model = _tiny_model()
    topology = SimpleNamespace(attn_tp=4, mlp_tp=8)
    originals = [[layer.get_submodule(role).weight.clone() for role, _ in _PROJECTIONS]
                 for layer in model.model.layers]
    tables = nvfp4.convert_qwen3_to_nvfp4_(model, topology=topology)
    assert len(tables) == 7
    for index, ((role, partition), table) in enumerate(zip(_PROJECTIONS, tables, strict=True)):
        assert table.dtype == torch.float32 and table.shape == (8,) and table.is_contiguous()
        assert torch.all(table[2:] == 1)
        cores = 8 if role.startswith("mlp.") else 4
        for layer_index, layer in enumerate(model.model.layers):
            projection = layer.get_submodule(role)
            codes, scales, tensor_scale = quantize_nvfp4(originals[layer_index][index])
            n, k = codes.shape
            # Each projection uses its own partition and component width.
            assert torch.equal(projection.weight, swizzle_pack_nvfp4(codes, partition, cores).reshape(n, k // 2))
            assert not projection.weight.requires_grad
            assert torch.equal(projection.weight_scale, swizzle_nvfp4_scales(scales, partition, cores))
            assert projection.weight_scale.shape == (k // 16, n)
            assert "weight_scale" in dict(projection.named_buffers())
            assert "weight_scale" not in projection.state_dict()
            assert table[layer_index] == tensor_scale


@pytest.mark.parametrize("fault", ["row_k", "col_n", "dtype"])
def test_converter_checks_every_projection_before_packing(fault):
    model = _tiny_model()
    projection = model.model.layers[-1].get_submodule(
        "mlp.down_proj" if fault == "row_k" else "mlp.gate_proj")
    if fault == "row_k":
        projection.weight = torch.nn.Parameter(projection.weight[:, :960].clone(), requires_grad=False)
    elif fault == "col_n":
        projection.weight = torch.nn.Parameter(projection.weight[:448].clone(), requires_grad=False)
    else:
        projection.weight = torch.nn.Parameter(projection.weight.float(), requires_grad=False)
    with pytest.raises((ValueError, TypeError), match="NVFP4"):
        nvfp4.convert_qwen3_to_nvfp4_(model, topology=SimpleNamespace(attn_tp=4, mlp_tp=8))
    assert all(layer.get_submodule(role).weight.dtype != torch.uint8
               for layer in model.model.layers for role, _ in _PROJECTIONS)


def _padding_model():
    # Logical MLP 512 is not 384-aligned, so six lanes pad it to 768.
    shapes = {"self_attn.q_proj": (512, 256), "self_attn.k_proj": (128, 256),
              "self_attn.v_proj": (128, 256), "self_attn.o_proj": (256, 512),
              "mlp.gate_proj": (512, 256), "mlp.up_proj": (512, 256), "mlp.down_proj": (256, 512)}
    generator = torch.Generator().manual_seed(11)
    layer = torch.nn.Module()
    layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
    for role, (n, k) in shapes.items():
        linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
        linear.weight = torch.nn.Parameter(torch.randn(n, k, generator=generator).half(),
                                           requires_grad=False)
        parent, name = role.split(".")
        setattr(getattr(layer, parent), name, linear)
    return SimpleNamespace(model=SimpleNamespace(layers=torch.nn.ModuleList([layer])))


@pytest.mark.parametrize("mlp_tp,physical", [(6, 768), (4, 512)])
def test_converter_zero_pads_a_six_lane_mlp_before_quantization(mlp_tp, physical):
    model = _padding_model()
    layer = model.model.layers[0]
    originals = {role: layer.get_submodule(role).weight.clone() for role, _ in _PROJECTIONS}
    tables = nvfp4.convert_qwen3_to_nvfp4_(model, topology=SimpleNamespace(attn_tp=4, mlp_tp=mlp_tp))
    for index, (role, partition) in enumerate(_PROJECTIONS):
        projection = layer.get_submodule(role)
        source = originals[role]
        n, k = source.shape
        if role.startswith("mlp."):
            n, k = (n, physical) if partition == 0 else (physical, k)
        assert (projection.out_features, projection.in_features) == (n, k)
        assert tuple(projection.weight.shape) == (n, k // 2)
        assert tuple(projection.weight_scale.shape) == (k // 16, n)
        logical_codes, logical_scales, logical_tensor = quantize_nvfp4(source)
        codes, scales, tensor_scale = quantize_nvfp4(torch.nn.functional.pad(
            source, (0, k - source.shape[1], 0, n - source.shape[0])))
        # Zero blocks keep the logical quantization and the projection amax.
        assert tensor_scale == logical_tensor == tables[index][0]
        assert torch.equal(codes[:source.shape[0], :source.shape[1]], logical_codes)
        assert torch.equal(scales[:source.shape[1] // 16, :source.shape[0]], logical_scales)
        assert not codes[source.shape[0]:].any() and not codes[:, source.shape[1]:].any()
        assert not scales[source.shape[1] // 16:].any() and not scales[:, source.shape[0]:].any()
        cores = mlp_tp if role.startswith("mlp.") else 4
        assert torch.equal(projection.weight, swizzle_pack_nvfp4(codes, partition, cores).reshape(n, k // 2))
        assert torch.equal(projection.weight_scale, swizzle_nvfp4_scales(scales, partition, cores))


@pytest.mark.parametrize("layers,padded", [(28, 32), (36, 40), (40, 40)])
def test_tensor_scale_tables_pad_with_identity(layers, padded):
    values = [torch.tensor((i + 1) / 64, dtype=torch.float32) for i in range(layers)]
    table = nvfp4.pad_nvfp4_tensor_scales(values)
    assert table.shape == (padded,) and torch.equal(table[:layers], torch.stack(values))
    assert torch.all(table[layers:] == 1)
    for invalid in (torch.tensor(0.), torch.tensor(float("nan")), torch.tensor([1.]), torch.tensor(1.).half()):
        with pytest.raises(ValueError):
            nvfp4.pad_nvfp4_tensor_scales([invalid, *values[1:]])
    with pytest.raises(ValueError):
        nvfp4.pad_nvfp4_tensor_scales([])


def _payload(size, cores):
    config = _config(size)
    profile = qwen3_dense_quant_profile(config)
    topology = resolve_decoder_topology(num_cores=cores, **profile.geometry())
    h = profile.hidden_size
    i = decoder_nvfp4_mlp_intermediate_size(profile.intermediate_size, topology.mlp_tp)
    q = profile.num_q_heads * profile.head_dim
    roles = (("self_attn.q_proj", q, h), ("self_attn.k_proj", 1024, h), ("self_attn.v_proj", 1024, h),
             ("self_attn.o_proj", h, q), ("mlp.gate_proj", i, h), ("mlp.up_proj", i, h),
             ("mlp.down_proj", h, i))
    layer = torch.nn.Module()
    layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
    scales = []
    for role, n, k in roles:
        linear = torch.nn.Linear(k, n, bias=False, device="meta")
        linear.weight = torch.nn.Parameter(
            torch.empty(n, k // 2, device="meta", dtype=torch.uint8), requires_grad=False)
        linear.register_buffer("weight_scale", torch.empty(k // 16, n, device="meta", dtype=torch.uint8))
        parent, name = role.split(".")
        setattr(getattr(layer, parent), name, linear)
        scales.append([linear.weight_scale] * profile.num_layers)
    table = (profile.num_layers + 7) // 8 * 8
    tables = [torch.empty(table, dtype=torch.float32, device="meta") for _ in range(7)]
    return SimpleNamespace(config=config, layers=[layer] * profile.num_layers), scales, tables, topology


@pytest.mark.parametrize("size,cores", [("0.6", 4), ("0.6", 6), ("0.6", 8), ("1.7", 6),
                                        ("4", 4), ("4", 6), ("4", 8), ("8", 6), ("8", 8),
                                        ("14", 4), ("14", 6), ("14", 8)])
def test_installer_validator_accepts_actual_text_payload(size, cores):
    decoder._validate_nvfp4_payload(*_payload(size, cores))


@pytest.mark.parametrize("size,width", [("4", 9792), ("4", 9728), ("14", 17472)])
def test_installer_validator_requires_the_packed_six_lane_width(size, width):
    model, scales, tables, topology = _payload(size, 6)
    layer = model.layers[0]
    h = layer.mlp.down_proj.out_features
    # FP16/W8 pad 9728 to 9792; NVFP4 uses the packed 384-aligned width.
    for name, n, k in (("gate_proj", width, h), ("up_proj", width, h), ("down_proj", h, width)):
        linear = torch.nn.Linear(k, n, bias=False, device="meta")
        linear.weight = torch.nn.Parameter(
            torch.empty(n, k // 2, device="meta", dtype=torch.uint8), requires_grad=False)
        linear.register_buffer("weight_scale", torch.empty(k // 16, n, device="meta", dtype=torch.uint8))
        setattr(layer.mlp, name, linear)
    for index, name in enumerate(("gate_proj", "up_proj", "down_proj"), start=4):
        scales[index] = [getattr(layer.mlp, name).weight_scale] * len(model.layers)
    with pytest.raises(ValueError, match="packed weight and FP8 scales"):
        decoder._validate_nvfp4_payload(model, scales, tables, topology)


@pytest.mark.parametrize("fault", ["asr_table", "missing_table", "scale_shape", "w4_scale",
                                   "weight_dtype", "other_owner", "missing_layer", "nearby_profile"])
def test_installer_validator_rejects_wrong_text_payload(fault):
    model, scales, tables, topology = _payload("4", 8)
    projection = model.layers[0].mlp.down_proj
    if fault == "asr_table":
        tables[0] = torch.empty(32, dtype=torch.float32, device="meta")  # 36 layers pad to 40
    elif fault == "missing_table":
        tables.pop()
    elif fault in ("scale_shape", "w4_scale"):
        value = (projection.weight_scale[:-1] if fault == "scale_shape"
                 else torch.empty(32, 1216, dtype=torch.float16, device="meta"))
        projection.weight_scale = value
        scales[-1] = [value] * 36
    elif fault == "weight_dtype":
        projection.weight = torch.nn.Parameter(projection.weight.to(torch.int8), requires_grad=False)
    elif fault == "other_owner":
        scales[-1][-1] = scales[-1][-1].clone()
    elif fault == "missing_layer":
        scales[-1].pop()
    else:
        model.config.intermediate_size = 9216
    with pytest.raises(ValueError):
        decoder._validate_nvfp4_payload(model, scales, tables, topology)


@pytest.mark.parametrize("arch,bound", [("llama", False), ("qwen3", False)])
def test_explicit_nvfp4_payload_requires_a_bound_text_topology(monkeypatch, arch, bound):
    model, scales, tables, topology = _payload("0.6", 8)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_create",
        lambda *args, **kwargs: pytest.fail("unreviewed NVFP4 payload created a native decoder"), raising=False)
    with pytest.raises(ValueError, match="restricted to the reviewed ASR"):
        decoder._install_causal_decoder_forward(model, arch=arch, topology=topology if bound else None,
            scale_lists=scales, nvfp4_tensor_scales=tables,
            execution_config={"model": {"num_cores": 8}, "prefill": {"linear_acc32": False}},
            chunk_envelope_for=lambda *args: (512, 128))


def _fake_pack(monkeypatch):
    # Shapes and dtypes only; real packing is checked above without fake tensors.
    packed = []
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
    return packed


@pytest.mark.parametrize("cores", [4, 6, 8])
@pytest.mark.parametrize("fault", [None, "table_size", "missing_scales", "w4_route"])
def test_adapter_install_passes_nvfp4_payload_to_actual_installer(monkeypatch, native_install, cores, fault):
    model = _model(cores=cores)
    _host_install(monkeypatch, model)
    packed = _fake_pack(monkeypatch)
    weights_calls = []
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_nvfp4",
                        lambda *args: weights_calls.append(args), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_w8a16",
                        lambda *args: pytest.fail("NVFP4 used the packed W4 route"), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_linear_acc32", lambda h, mode: None, raising=False)
    installs = []
    def install(inner, **kwargs):
        monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology",
                            lambda h: kwargs["topology"].identity()[1:], raising=False)
        if fault == "table_size":
            kwargs["nvfp4_tensor_scales"] = (kwargs["nvfp4_tensor_scales"][0][:28],
                                             *kwargs["nvfp4_tensor_scales"][1:])
        elif fault == "missing_scales":
            kwargs["scale_lists"] = None
        elif fault == "w4_route":
            kwargs.pop("nvfp4_tensor_scales")
        installs.append(kwargs)
        table = _FakeTensor()
        return decoder._install_causal_decoder_forward(inner, **kwargs, cos_sin=(table, table))
    monkeypatch.setattr(qwen3, "_install_causal_decoder_forward", install)
    adapter = qwen3.Qwen3Adapter(model, quantization="nvfp4")
    with model.lm_head.weight.fake_mode:
        if fault:
            reason = {"table_size": "padded FP32", "missing_scales": "seven complete",
                      "w4_route": "packed Qwen3"}[fault]
            with pytest.raises(ValueError, match=reason):
                adapter.to_rpu()
            assert native_install["events"] == [] and not weights_calls
            return
        assert adapter.to_rpu() is model
    assert len(packed) == 28 * 7 and len(weights_calls) == 1
    tables = installs[0]["nvfp4_tensor_scales"]
    keepalive = model.model._rpu_decoder_retirement_state.keepalive
    assert len(tables) == 7 and all(any(t is kept for kept in keepalive) for t in tables)
    device = model.model.layers[0].self_attn.q_proj.weight.device
    assert all(t.dtype == torch.float32 and t.shape == (32,) and t.device == device for t in tables)
    assert installs[0]["topology"].identity()[1:4] == (cores, 8 if cores == 8 else 4, cores)
    args = weights_calls[0]
    # Handle, 23 shared arguments, seven block-scale families, then the seven tensor-scale tables.
    assert args[1][0] is model.model.layers[0].self_attn.q_proj.weight
    assert args[-8][-1] is model.model.layers[-1].mlp.down_proj.weight_scale
    assert all(actual is expected for actual, expected in zip(args[-7:], tables, strict=True))
    for role, _ in _PROJECTIONS:
        projection = model.model.layers[-1].get_submodule(role)
        assert projection.weight.dtype == torch.uint8 and projection.weight_scale.dtype == torch.uint8
        assert projection.weight_scale.shape == (projection.in_features // 16, projection.out_features)
    assert model.lm_head.weight.dtype == model.model.embed_tokens.weight.dtype == torch.float16
    assert decoder._causal_lm_runtime_complete(model)
    _finalize(model.model)


@pytest.mark.parametrize("size", ["4", "8", "14"])
@pytest.mark.parametrize("cores", [4, 6])
def test_reduced_dense_install_passes_the_physical_mlp(monkeypatch, native_install, size, cores):
    h, i, layers, heads = SHAPES[size]
    model = _model(cores=cores, **_dense_config((h, i, layers, heads)))
    _host_install(monkeypatch, model)
    packed = _fake_pack(monkeypatch)
    weights_calls = []
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_nvfp4",
                        lambda *args: weights_calls.append(args), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_linear_acc32", lambda h, mode: None, raising=False)
    installs = []
    def install(inner, **kwargs):
        monkeypatch.setattr(torch.ops.rpu, "causal_decoder_topology",
                            lambda h: kwargs["topology"].identity()[1:], raising=False)
        installs.append(kwargs)
        table = _FakeTensor()
        return decoder._install_causal_decoder_forward(inner, **kwargs, cos_sin=(table, table))
    monkeypatch.setattr(qwen3, "_install_causal_decoder_forward", install)
    adapter = qwen3.Qwen3Adapter(model, quantization="nvfp4")
    with model.lm_head.weight.fake_mode:
        assert adapter.to_rpu() is model
    physical = decoder_nvfp4_mlp_intermediate_size(i, 6) if cores == 6 else i
    assert physical == {"4": 9984 if cores == 6 else 9728, "8": 12288,
                        "14": 17664 if cores == 6 else 17408}[size]
    assert len(packed) == layers * 7 and len(weights_calls) == 1
    assert packed[4:7] == [(physical, h), (physical, h), (h, physical)]
    assert installs[0]["topology"].identity()[1:4] == (cores, 4, cores)
    mlp = model.model.layers[-1].mlp
    assert (mlp.gate_proj.out_features, mlp.down_proj.in_features) == (physical, physical)
    assert mlp.down_proj.weight_scale.shape == (physical // 16, h)
    assert model.config.intermediate_size == i
    _finalize(model.model)
