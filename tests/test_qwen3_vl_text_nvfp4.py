"""Qwen3-VL text NVFP4 recipe: profiles, preflight, installer payload and Vision scope; no board."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rpu_backend.adapters import qwen3_vl
from rpu_backend.adapters.qwen3_vl.text import install_qwen3_vl_text_for_rpu
from rpu_backend.api import conditional_generation as public
from rpu_backend.quant.nvfp4_pack import quantize_nvfp4, swizzle_nvfp4_scales, swizzle_pack_nvfp4
from rpu_backend.runtime import decoder
from rpu_backend.runtime.topology import decoder_nvfp4_mlp_intermediate_size
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import (
    _Model,
    install_runtime,  # noqa: F401
)
from test_qwen3_vl_runtime_quant_loading import config as vl_config
from test_qwen3vl_text_core46 import _SIZES, _model as text_model, native  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
ROLES = qwen3_vl._QWEN3_VL_TEXT_PROJECTIONS
PARTITIONS = {role: 0 if role.endswith(("o_proj", "down_proj")) else 1 for role in ROLES}


@pytest.fixture
def common():
    return importlib.import_module("examples.common.config")


def _source(size):
    return vl_config(int(size[:-1]), source=True)


@pytest.mark.parametrize("size", _SIZES)
def test_vl_nvfp4_profiles_bind_exact_recipe_geometry_and_cores(common, size):
    config = common.load_config(ROOT / f"examples/configs/qwen3_vl/text/{size}/nvfp4_text.toml")
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["id"] == f"qwen3-vl-{size}.nvfp4.text"
    assert profile["asset_alias"] == config["example"]["registry_alias"] == f"qwen3-vl-{size}"
    assert profile["family"] == "qwen3-vl" and profile["component"] == config["input"]["component"] == "text"
    assert profile["precision"] == config["example"]["dtype"] == limits["on_install_quantization"] == "nvfp4"
    assert tuple(limits[key] for key in (
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"
    )) == _SIZES[size]
    assert (limits["num_key_value_heads"], limits["head_dim"], limits["vocab_size"]) == (8, 128, 151936)
    assert {limits["quantization"][key] for key in ("lm_head", "embedding", "norms", "vision")} == {"FP16"}
    assert limits["allowed_num_cores"] == [4, 6, 8]
    assert limits["batch_size"] == 1 and limits["formal_certification"] is False
    # Same input as the W4 text rows, so the two precisions compare directly.
    assert (config["input"]["prefill_tokens"], config["input"]["decode_steps"]) == (512, 31)
    assert (config["run"]["warmup"], config["run"]["runs"]) == (2, 5)
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    request = {"model": {"num_cores": 8}}
    common._validate_execution(request, profile=profile)
    for cores in (4, 6):
        request["model"]["num_cores"] = cores
        common._validate_execution(request, profile=profile)
    for cores in (2, 5, 7):
        request["model"]["num_cores"] = cores
        with pytest.raises((ValueError, common.ConfigError)):
            common._validate_execution(request, profile=profile)


@pytest.mark.parametrize("size", _SIZES)
def test_checkpoint_check_reads_the_nested_text_geometry(common, monkeypatch, size):
    from transformers import AutoConfig

    config = common.load_config(ROOT / f"examples/configs/qwen3_vl/text/{size}/nvfp4_text.toml")
    source = _source(size)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **k: source)
    common.validate_checkpoint_profile(config, "source")
    source.text_config.intermediate_size += 64
    with pytest.raises(common.ConfigError, match="exact Qwen3 geometry"):
        common.validate_checkpoint_profile(config, "source")


@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_preflight_admits_native_sizes_and_core_budgets(size, cores):
    execution = {"model": {"num_cores": cores}}
    qwen3_vl.Qwen3VLAdapter.preflight_quantization(_source(size), execution, "nvfp4")


@pytest.mark.parametrize("fault", ["32b", "32b_tp6", "awq", "runtime_w4", "tp5", "rope", "recipe"])
def test_public_nvfp4_rejects_before_weight_load(monkeypatch, fault):
    size = 32 if fault.startswith("32b") else 2
    config = vl_config(size, source=fault != "runtime_w4", **({} if fault != "runtime_w4" else {"bits": 4}))
    cores = {"32b_tp6": 6, "tp5": 5}.get(fault, 8)
    if fault == "awq":
        config.quantization_config = {"quant_method": "compressed-tensors"}
    elif fault == "rope":
        config.text_config.rope_parameters["mrope_section"] = [24, 20, 16]
    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(public.AutoModelForImageTextToText, "from_pretrained",
                        lambda *a, **k: pytest.fail("invalid NVFP4 source loaded weights"))
    with pytest.raises((qwen3_vl.UnsupportedModelError, ValueError)):
        public.RPUModelForConditionalGeneration.from_pretrained(
            "source", quantization="w8a16" if fault == "recipe" else "nvfp4",
            rpu_execution={"model": {"num_cores": cores}})


def _nvfp4_text(size, cores, physical=None):
    """Replace a text fixture's projections with its exact stored NVFP4 payload."""
    model, topology = text_model(size, cores if cores != 8 else 4)
    # NVFP4 pads a six-lane MLP to the packed-W4 width, not the FP16/W8 one.
    physical = physical or decoder_nvfp4_mlp_intermediate_size(
        model.config.intermediate_size, topology.mlp_tp)
    scales = [[] for _ in ROLES]
    for layer in model.layers:
        # The fixture's layers are plain namespaces; real ones are modules.
        layer.get_submodule = lambda role, layer=layer: getattr(
            getattr(layer, role.split(".")[0]), role.split(".")[1])
        for index, role in enumerate(ROLES):
            projection = layer.get_submodule(role)
            if role.startswith("mlp."):
                if role.endswith("down_proj"):
                    projection.in_features = physical
                else:
                    projection.out_features = physical
            n, k = projection.out_features, projection.in_features
            projection.weight = torch.empty(n, k // 2, device="meta", dtype=torch.uint8)
            projection.weight_scale = torch.empty(k // 16, n, device="meta", dtype=torch.uint8)
            scales[index].append(projection.weight_scale)
    width = (len(model.layers) + 7) // 8 * 8
    tables = [torch.empty(width, device="meta", dtype=torch.float32) for _ in ROLES]
    return model, scales, tables, topology if cores != 8 else None


_PAYLOADS = [("2b", 4), ("2b", 6), ("2b", 8), ("4b", 4), ("4b", 6), ("4b", 8),
             ("8b", 4), ("8b", 6), ("8b", 8)]


@pytest.mark.parametrize("size,cores", _PAYLOADS)
def test_installer_validator_accepts_vl_payload_with_or_without_bound_topology(size, cores):
    model, scales, tables, topology = _nvfp4_text(size, cores)
    decoder._validate_nvfp4_payload(model, scales, tables, topology, arch="qwen3_vl_text")


@pytest.mark.parametrize("fault", ["table", "missing_table", "scale_shape", "w4_scale", "weight_dtype",
                                   "other_owner", "missing_layer", "nearby_profile", "plain_qwen3_config"])
def test_installer_validator_rejects_wrong_vl_payload(fault):
    model, scales, tables, topology = _nvfp4_text("4b", 8)
    projection = model.layers[0].mlp.down_proj
    if fault == "table":
        tables[0] = torch.empty(32, device="meta", dtype=torch.float32)  # 36 layers pad to 40
    elif fault == "missing_table":
        tables.pop()
    elif fault in ("scale_shape", "w4_scale"):
        value = (projection.weight_scale[:-1] if fault == "scale_shape"
                 else torch.empty(32, 1216, device="meta", dtype=torch.float16))
        projection.weight_scale = value
        scales[-1][0] = value
    elif fault == "weight_dtype":
        projection.weight = projection.weight.to(torch.int8)
    elif fault == "other_owner":
        scales[-1][-1] = scales[-1][-1].clone()
    elif fault == "missing_layer":
        scales[-1].pop()
    elif fault == "nearby_profile":
        model.config.intermediate_size = 9216
    else:
        model.config.model_type = "qwen3"
    with pytest.raises(ValueError):
        decoder._validate_nvfp4_payload(model, scales, tables, topology, arch="qwen3_vl_text")


@pytest.mark.parametrize("size,cores", _PAYLOADS)
def test_text_installer_routes_vl_nvfp4_to_its_native_setter(native, monkeypatch, size, cores):
    model, scales, tables, topology = _nvfp4_text(size, cores)
    calls = []
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_nvfp4",
                        lambda *args: calls.append(args), raising=False)
    for name in ("causal_decoder_set_weights_w8a16", "causal_decoder_set_weights"):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *a, name=name: pytest.fail(f"NVFP4 used {name}"), raising=False)
    execution = {"model": {"num_cores": cores}, "prefill": {"linear_acc32": False}}
    assert install_qwen3_vl_text_for_rpu(
        model, scale_lists=scales, nvfp4_tensor_scales=tables, execution_config=execution,
        deepstack_lang_layers=[0, 1, 2], max_seq_len=544,
        **({"topology": topology} if topology is not None else {})) == 1
    assert len(calls) == 1
    args = calls[0]
    assert args[1][0] is model.layers[0].self_attn.q_proj.weight
    assert args[-8][-1] is model.layers[-1].mlp.down_proj.weight_scale
    assert all(actual is expected for actual, expected in zip(args[-7:], tables, strict=True))
    integers = [list(a) for a in args if isinstance(a, (list, tuple)) and a and isinstance(a[0], int)]
    assert [24, 20, 20] in integers and [0, 1, 2] in integers
    assert getattr(model, "_rpu_decoder_topology", None) is topology
    keepalive = model._rpu_decoder_retirement_state.keepalive
    assert all(any(table is kept for kept in keepalive) for table in tables)
    model._rpu_decoder_retirement_state.retire()
    assert native["alive"] == set()


@pytest.mark.parametrize("fault", ["layers", "w8_width"])
def test_reduced_vl_nvfp4_refuses_a_payload_off_its_exact_profile(native, monkeypatch, fault):
    # The FP16/W8 six-lane width 9792 is not the NVFP4 packed width 9984.
    model, scales, tables, topology = _nvfp4_text("4b", 6, 9792 if fault == "w8_width" else None)
    if fault == "layers":
        # Same stored payload with one layer fewer: the exact profile refuses it first.
        model.layers = model.layers[:-1]
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_set_weights_nvfp4",
                        lambda *a: pytest.fail("off-profile payload reached native"), raising=False)
    with pytest.raises(ValueError):
        install_qwen3_vl_text_for_rpu(
            model, scale_lists=scales, nvfp4_tensor_scales=tables, topology=topology,
            execution_config={"model": {"num_cores": 6}}, deepstack_lang_layers=[0, 1, 2], max_seq_len=544)
    assert native["alive"] == set()


class _Layer(torch.nn.Module):
    """A small converter-valid layer; storage moves are device boundaries."""

    moves = None

    def to(self, *args, **kwargs):
        self.moves.append(args)
        return self


class _Head(torch.nn.Linear):
    def to(self, *args, **kwargs):
        return self


def _tiny_vl_model(size, cores, layers=2):
    # Smallest widths accepted by the swizzle rules at attention 4/8 and MLP 4/6/8.
    shapes = {"self_attn.q_proj": (512, 512), "self_attn.k_proj": (128, 512),
              "self_attn.v_proj": (128, 512), "self_attn.o_proj": (512, 512),
              "mlp.gate_proj": (1536, 512), "mlp.up_proj": (1536, 512), "mlp.down_proj": (512, 1536)}
    generator = torch.Generator().manual_seed(7)
    model = _Model()
    model.config = _source(size)
    model._rpu_execution = {"model": {"num_cores": cores}, "prefill": {"fast_replay": True}}
    blocks = []
    moves = []
    for index in range(layers):
        layer = _Layer()
        layer.moves = moves
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        for offset, (role, (n, k)) in enumerate(shapes.items()):
            linear = torch.nn.Linear(k, n, bias=False, dtype=torch.float16)
            weight = torch.randn(n, k, generator=generator) * (index + 1) * (offset + 1) / 16
            linear.weight = torch.nn.Parameter(weight.half(), requires_grad=False)
            owner, name = role.split(".")
            setattr(getattr(layer, owner), name, linear)
        blocks.append(layer)
    text = model.model.language_model
    text.config, text.layers = model.config.text_config, blocks
    model.model.visual.config = model.config.vision_config
    # The reduced install swizzles the head through a module view (384 is divisible by 16*{4,6,8}).
    model.lm_head = _Head(128, 384, bias=False, dtype=torch.float16)
    return model, moves


@pytest.fixture
def vl_install(monkeypatch, install_runtime):
    state = install_runtime
    monkeypatch.setattr(qwen3_vl, "enable_qwen3_vl_vision_merger_on_device", lambda model, **kwargs: (
        state["events"].append(("enable_vision_merger", kwargs)),
        setattr(model, "_rpu_vision_merger_on_device", True),
        setattr(model, "_rpu_patch_embed_on_device", True)))
    monkeypatch.setattr(qwen3_vl, "register_qwen3vl_vision_fused_merger",
                        lambda *a, **k: state["events"].append(("fused_vision_merger", k)))
    for name in ("causal_decoder_set_fast_replay", "causal_decoder_set_preload_replay_skip"):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda handle, enabled, name=name: state["events"].append((name, handle, enabled)),
                            raising=False)
    return state


@pytest.mark.parametrize("size,cores", [("2b", 8), ("4b", 8), ("8b", 8), ("2b", 6), ("2b", 4),
                                        ("4b", 6), ("8b", 4)])
def test_adapter_quantizes_text_once_and_keeps_head_and_vision_fp16(monkeypatch, vl_install, size, cores):
    state = vl_install
    from rpu_backend.quant import load as quant_load
    monkeypatch.setattr(quant_load, "_move_materialized_decoder_for_rpu",
                        lambda *a, **k: pytest.fail("NVFP4 took the FP16 storage-bank decoder move"))
    monkeypatch.setattr(qwen3_vl, "_validate_qwen3_vl_reduced_weights", lambda *a: None)
    model, moves = _tiny_vl_model(size, cores)
    text = model.model.language_model
    originals = [[layer.get_submodule(role).weight.clone() for role in ROLES] for layer in text.layers]
    owner = qwen3_vl.Qwen3VLAdapter(model, quantization="nvfp4")
    assert owner.to_rpu() is model
    args = state["text_kwargs"]
    tables = args["nvfp4_tensor_scales"]
    assert len(tables) == 7 and all(t.dtype == torch.float32 and t.shape == (8,) for t in tables)
    assert len(moves) == len(text.layers) and all(move == ("rpu",) for move in moves)
    attn, mlp = (8, 8) if cores == 8 else (4, cores)
    for index, role in enumerate(ROLES):
        lanes = mlp if role.startswith("mlp.") else attn
        partition = PARTITIONS[role]
        for number, layer in enumerate(text.layers):
            projection = layer.get_submodule(role)
            codes, block, tensor_scale = quantize_nvfp4(originals[number][index])
            n, k = codes.shape
            assert torch.equal(projection.weight, swizzle_pack_nvfp4(codes, partition, lanes).reshape(n, k // 2))
            assert torch.equal(projection.weight_scale, swizzle_nvfp4_scales(block, partition, lanes))
            assert tables[index][number] == tensor_scale
            assert args["scale_lists"][index][number] is projection.weight_scale
            assert projection._rpu_linear_partition == partition
            assert projection._rpu_linear_num_cores == lanes
    # The output head and Vision take their ordinary FP16 paths.
    assert model.lm_head.weight.dtype == torch.float16
    assert not args.get("runtime_align_w8a16", False)
    vision = state["vision_kwargs"]
    assert not vision.get("w8a16", False) and not vision.get("_allow_runtime_quantized_large", False)
    # Text was packed above: only the reduced install's separate head view is swizzled here.
    assert state["events"].count(("swizzle", "text")) == (0 if cores == 8 else 1)
    assert ("set_linear_acc32", 22, False) in state["events"]
    assert ("causal_decoder_set_fast_replay", 22, True) in state["events"]
    if cores == 8:
        assert ("enable_vision_merger", {}) in state["events"]
        assert not any(event[0] == "fused_vision_merger" for event in state["events"])
    owner.close()


def test_adapter_refuses_nvfp4_shape_before_replacing_a_projection(vl_install):
    model, _moves = _tiny_vl_model("2b", 8)
    last = model.model.language_model.layers[-1].mlp.down_proj
    last.weight = torch.nn.Parameter(last.weight[:, :960].clone(), requires_grad=False)
    with pytest.raises(ValueError, match="NVFP4"):
        qwen3_vl.Qwen3VLAdapter(model, quantization="nvfp4").to_rpu()
    assert all(layer.get_submodule(role).weight.dtype == torch.float16
               for layer in model.model.language_model.layers for role in ROLES)
    assert vl_install["text_kwargs"] is None


@pytest.mark.parametrize("stem,recipe", [("nvfp4_text", "nvfp4"), ("w4a16_text_awq_g32", None)])
def test_runner_requests_the_cold_recipe_only_for_nvfp4(common, monkeypatch, stem, recipe):
    import transformers
    from examples.common import generation
    from examples.runners.qwen3_vl import runner
    from rpu_backend import api, model_registry

    config = common.load_config(ROOT / f"examples/configs/qwen3_vl/text/2b/{stem}.toml")
    monkeypatch.setattr(common, "validate_checkpoint_profile", lambda *_: None)
    monkeypatch.setattr(model_registry, "model_path", lambda _: "/cache/qwen3-vl")
    monkeypatch.setattr(transformers, "AutoProcessor", SimpleNamespace(
        from_pretrained=lambda *_a, **_kw: SimpleNamespace(tokenizer=None)))
    ids = torch.ones((1, 512), dtype=torch.long)
    monkeypatch.setattr(generation, "_text_input_ids", lambda *_: ids)
    monkeypatch.setattr(generation, "_generate", Mock())
    loader = Mock(return_value=Mock())
    monkeypatch.setattr(api.RPUModelForConditionalGeneration, "from_pretrained", loader)
    monkeypatch.setattr(api.RPUCache, "from_model", Mock())
    runner.prepare(config, SimpleNamespace(owner=None))
    assert loader.call_args.kwargs.get("quantization") == recipe
