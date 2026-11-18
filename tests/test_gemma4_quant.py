"""Gemma4 E4B on-install W8A16 / G32 W4A16: profiles, preflight, the real
conversion of one decoder layer, the installer payload and the native source
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.common import config as COMMON  # noqa: E402
from rpu_backend.adapters import gemma4  # noqa: E402
from rpu_backend.adapters.gemma4 import runtime, topology as g4_topology  # noqa: E402
from rpu_backend.api.errors import RPUBackendError, UnsupportedModelError  # noqa: E402
from rpu_backend.quant import convert_gemma4 as quant  # noqa: E402
from rpu_backend.quant._common import quantize_linear_per_channel  # noqa: E402
from rpu_backend.quant.int4_pgrp_pack import quantize_int4_group_wise  # noqa: E402
from rpu_backend.runtime.weights import transform_linear_weight  # noqa: E402
from rpu_backend.tests.acceptance import test_gemma4_install_transaction as installed  # noqa: E402
from rpu_backend.tests.acceptance.test_gemma4_install_transaction import (  # noqa: E402,F401
    install_runtime,
)
from test_dinov3_quant import _natural  # noqa: E402
from test_gemma4_reduced_cores import (  # noqa: E402
    _body, _patch_topology_ops, _split_top, _strip_comments,
)
from test_qwen3_vl_awq_packing import inverse_rpu_scale, inverse_rpu_weight  # noqa: E402

NATIVE = ROOT / "src/fused/rpu_gemma4_model.cpp"
RECIPES = {  # TOML recipe -> (profile variant, on-install quantization)
    "w8a16_text": ("w8a16", "w8a16"),
    "w4a16_text_rtn_g32": ("w4a16-g32", "w4a16"),
}
STORAGE = {"w8a16": torch.int8, "w4a16": torch.uint8}
ROLES = gemma4._PROJECTION_ROLES
H, INTER = 2560, 10240


def _physical(cores, quantization):
    return 10240 if cores != 6 else (10272 if quantization is None else 10368)


def _config():
    return SimpleNamespace(text_config=installed._text_config())


# ------------------------------------------------------------- profiles and preflight


@pytest.mark.parametrize("recipe", RECIPES)
def test_quant_profiles_bind_exact_recipe_and_cores(recipe):
    variant, quantization = RECIPES[recipe]
    config = COMMON.load_config(ROOT / f"examples/configs/gemma4/e4b/{recipe}.toml")
    profile = COMMON.planner_profiles()[config["example"]["profile_id"]]
    limits = profile["input_envelope"]["limits"]
    assert profile["id"] == f"gemma4-e4b.{variant}.text"
    assert config["example"]["registry_alias"] == "gemma4-e4b"
    assert COMMON.install_quantization("gemma4", profile["id"]) == quantization
    assert COMMON.install_quantization("gemma4", "gemma4-e4b.fp16.text") is None
    assert limits["on_install_quantization"] == quantization
    assert limits.get("group_size") == (32 if quantization == "w4a16" else None)
    assert limits["linear_accumulation"] == "ACC16" and limits["profile"] == "e4b"
    assert (limits["prefill_tokens"], limits["decode_steps"]) == (256, 16)
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
    with pytest.raises(COMMON.ConfigError, match="registered E4B text profile"):
        COMMON._validate_execution({"model": {"num_cores": 6}}, profile=unbound)


@pytest.mark.parametrize("quantization", ["w8a16", "w4a16"])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_preflight_admits_both_recipes_at_every_budget(quantization, cores):
    assert gemma4.Gemma4Adapter.preflight_quantization(
        _config(), {"model": {"num_cores": cores}}, quantization) == cores
    assert gemma4.Gemma4Adapter.preflight_quantization(_config(), None, quantization) == 8


@pytest.mark.parametrize("fault", ["nvfp4", "w4a16_lm_head", "prequantized",
                                   "prequantized_text", "cores5", "not_e4b"])
def test_preflight_rejects_before_weights(fault):
    config = _config()
    quantization, execution = "w4a16", {"model": {"num_cores": 6}}
    if fault in ("nvfp4", "w4a16_lm_head"):
        quantization = fault
    elif fault == "prequantized":
        config.quantization_config = {"quant_method": "awq", "bits": 4}
    elif fault == "prequantized_text":
        config.text_config.quantization_config = {"quant_method": "gptq", "bits": 4}
    elif fault == "cores5":
        execution = {"model": {"num_cores": 5}}
    else:
        config.text_config.hidden_size, execution = 2048, {}
    with pytest.raises(UnsupportedModelError):
        gemma4.Gemma4Adapter.preflight_quantization(config, execution, quantization)


class _Stop(Exception):
    pass


@pytest.mark.parametrize("prequantized", [False, True])
def test_runner_refuses_the_recipe_before_reading_the_checkpoint(monkeypatch, prequantized):
    import transformers
    from examples.runners.gemma4 import runner
    hf_config = _config()
    if prequantized:
        hf_config.quantization_config = {"quant_method": "awq", "bits": 4}
    example = {"target": "gemma4", "profile_id": "gemma4-e4b.w4a16-g32.text",
               "registry_alias": "gemma4-e4b"}
    execution = {"model": {"num_cores": 6}}
    monkeypatch.setattr(runner, "text_context", lambda _config: (
        example, {}, "gemma4", "/cache/gemma4", execution, 4))
    monkeypatch.setattr(runner, "text_inputs", lambda *_a: (None, None, {}, 64))
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_a, **_k: hf_config)

    def weights(*_a, **_k):
        if prequantized:
            pytest.fail("weights were read")
        raise _Stop

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", weights)
    # The admitted recipe reaches the weight load; the rejected one stops first.
    with pytest.raises(UnsupportedModelError if prequantized else _Stop):
        runner.prepare({}, SimpleNamespace(owner=None))


# ------------------------------------------------------------------- adapter binding


def _projection_shapes(g):
    attention, kv = 8 * g.head_dim, g.num_kv_heads * g.head_dim
    return {"q": (attention, H), "k": (kv, H), "v": (kv, H), "o": (H, attention),
            "gate": (INTER, H), "up": (INTER, H), "down": (H, INTER)}


class _QuantSourceModel(installed._Model):
    """Install double whose state_dict carries every E4B projection (shape only).

    KV-shared layers have no k/v of their own, as in the checkpoint.
    """

    def __init__(self):
        super().__init__()
        self.overrides = {}

    def state_dict(self):
        values = super().state_dict()
        for layer, g in enumerate(gemma4.layer_geometry(self.config.text_config)):
            for role, shape in _projection_shapes(g).items():
                if role in ("k", "v") and g.is_kv_shared:
                    continue
                name = quant.projection_source(runtime.PFX, layer, g, role)
                values[name] = torch.zeros((), dtype=torch.bfloat16).expand(shape)
        for name, value in self.overrides.items():
            if value is None:
                values.pop(name)
            else:
                values[name] = value
        return values


def test_recipe_binds_cold_once_per_model():
    model = _QuantSourceModel()
    gemma4.Gemma4Adapter(model, rpu_execution={"model": {"num_cores": 6}}, quantization="w8a16")
    assert model.to.__rpu_quantization__ == "w8a16"
    gemma4.Gemma4Adapter(model, quantization="w8a16")
    for other in ("w4a16", None):
        with pytest.raises(UnsupportedModelError, match="cold-only"):
            gemma4.Gemma4Adapter(model, quantization=other)
    fp16 = installed._Model()
    gemma4.Gemma4Adapter(fp16)
    with pytest.raises(UnsupportedModelError, match="cold-only"):
        gemma4.Gemma4Adapter(fp16, quantization="w4a16")


DOWN41 = runtime.PFX + "layers.41.mlp.down_proj.weight"


@pytest.mark.parametrize("fault", ["dtype", "device", "shape", "missing"])
@pytest.mark.parametrize("quantization", ["w8a16", "w4a16"])
def test_adapter_requires_the_original_cpu_float_projections(quantization, fault):
    model = _QuantSourceModel()
    model.overrides[DOWN41] = {
        "dtype": torch.zeros((), dtype=torch.int8).expand(H, INTER),
        "device": torch.empty(H, INTER, dtype=torch.bfloat16, device="meta"),
        "shape": torch.zeros((), dtype=torch.bfloat16).expand(H, 10368),  # already padded
        "missing": None,
    }[fault]
    with pytest.raises(UnsupportedModelError,
                       match=f"layers.41.mlp.down_proj.weight: Gemma4 {quantization} requires "
                             "the original CPU float projection"):
        gemma4.Gemma4Adapter(model, quantization=quantization)


def test_install_rechecks_the_binary_and_source_before_the_swizzle(monkeypatch):
    model = _QuantSourceModel()
    adapter = gemma4.Gemma4Adapter(model, quantization="w4a16")
    monkeypatch.setattr(runtime, "swizzle_decoder_and_ple",
                        lambda *_a, **_k: pytest.fail("swizzled"))
    with monkeypatch.context() as patch:
        patch.setattr(torch.ops, "rpu", SimpleNamespace())
        with pytest.raises(RPUBackendError, match="gemma4_set_weight_scales; rebuild"):
            adapter.to_rpu()
    monkeypatch.setattr(torch.ops.rpu, "gemma4_set_weight_scales", lambda *_a: None,
                        raising=False)
    model.overrides[DOWN41] = torch.zeros((), dtype=torch.float16).expand(H, INTER + 1)
    with pytest.raises(UnsupportedModelError, match="layers.41.mlp.down_proj"):
        adapter.to_rpu()
    assert not any(hasattr(model, name) for name in gemma4._INSTALL_ATTRS)


# ------------------------------------------------------------------ CPU conversion


def _topology_and_storage_width(cores, quantization):
    topology = g4_topology.resolve_gemma4_topology(cores, quantization=quantization)
    return topology, topology.physical_intermediate


@pytest.mark.parametrize("quantization", [None, "w8a16", "w4a16"])
@pytest.mark.parametrize("cores", [8, 6, 4])
def test_six_lane_width_follows_the_projection_storage(quantization, cores):
    topology, physical = _topology_and_storage_width(cores, quantization)
    assert physical == _physical(cores, quantization)
    assert topology.native_readback()[-2:] == (INTER, physical)
    assert physical % (g4_topology.MLP_LANE_GRANULE[quantization] * topology.mlp_tp) == 0
    if quantization is not None:
        # Quantized storage is admitted only for the exact E4B geometry, even at eight.
        with pytest.raises(ValueError, match="E4B"):
            g4_topology.resolve_gemma4_topology(8, quantization=quantization,
                                                intermediate_size=9728)
    if cores == 6:
        other = g4_topology.resolve_gemma4_topology(6, quantization=(
            "w8a16" if quantization is None else None))
        with pytest.raises(ValueError, match="different projection storage"):
            runtime.swizzle_decoder_and_ple(lambda _name: pytest.fail("read"), [],
                                            topology=other, quantization=quantization)
    with pytest.raises(ValueError, match="quantization must be"):
        g4_topology.resolve_gemma4_topology(cores, quantization="nvfp4")


def _layer_source(seed):
    generator = torch.Generator().manual_seed(seed)
    prefix = runtime.PFX + "layers.0."
    names = {}
    for name, shape in {
        "self_attn.q_proj.weight": (8 * 256, H), "self_attn.k_proj.weight": (2 * 256, H),
        "self_attn.v_proj.weight": (2 * 256, H), "self_attn.o_proj.weight": (H, 8 * 256),
        "mlp.gate_proj.weight": (INTER, H), "mlp.up_proj.weight": (INTER, H),
        "mlp.down_proj.weight": (H, INTER),
        "per_layer_input_gate.weight": (256, H), "per_layer_projection.weight": (H, 256),
    }.items():
        names[prefix + name] = torch.randn(*shape, generator=generator) / shape[-1] ** 0.5
    for name, size in (("self_attn.q_norm.weight", 256), ("self_attn.k_norm.weight", 256),
                       ("input_layernorm.weight", H), ("post_attention_layernorm.weight", H),
                       ("pre_feedforward_layernorm.weight", H),
                       ("post_feedforward_layernorm.weight", H),
                       ("post_per_layer_input_norm.weight", H)):
        names[prefix + name] = torch.ones(size)
    names[prefix + "layer_scalar"] = torch.ones(1)
    return names


def _reference(weight, quantization):
    """The checkpoint FP16 weight through the recipe's own quantize/dequantize."""
    if quantization == "w8a16":
        codes, scale = quantize_linear_per_channel(weight, bits=8)
        return codes.double() * scale.double()[:, None]
    codes, scale, _ = quantize_int4_group_wise(weight, 32)
    return codes.double() * scale.double().t().repeat_interleave(32, dim=1)


@pytest.mark.parametrize("quantization", ["w8a16", "w4a16"])
@pytest.mark.parametrize("cores", [8, 6, 4])
def test_layer_conversion_computes_the_recipe_weights_on_every_lane(
        monkeypatch, quantization, cores):
    """Actual adapter preparation; inverting each payload for its own lanes."""
    names = _layer_source(cores)
    prefix = runtime.PFX + "layers.0."
    # Host stand-in for the RPU upload; the payloads are the real ones.
    monkeypatch.setattr(runtime, "_to_rpu", lambda t: t.contiguous())
    topology, physical = _topology_and_storage_width(cores, quantization)
    geom = gemma4.layer_geometry(installed._text_config())[:1]
    lists, ple = runtime.swizzle_decoder_and_ple(
        lambda name: names[name].float(), geom, topology=topology, quantization=quantization)
    assert {f"{role}_scale" for role in ROLES} <= set(lists)
    for role, (suffix, partition) in quant.PROJECTIONS.items():
        weight, scale = lists[role][0], lists[f"{role}_scale"][0]
        assert len(lists[role]) == len(lists[f"{role}_scale"]) == 1
        lanes = topology.mlp_tp if role in quant.MLP_ROLES else topology.attn_tp
        logical = names[prefix + suffix].half()
        n, k = logical.shape
        pn = physical if role in ("gate", "up") else n
        pk = physical if role == "down" else k
        assert weight.dtype == STORAGE[quantization] and weight.is_contiguous()
        assert scale.dtype == torch.float16 and scale.is_contiguous()
        if quantization == "w8a16":
            assert weight.shape == (pn, pk) and scale.shape == (pn,)
            natural_scale = scale.double()[:, None]
            values = _natural(weight, partition, lanes).double() * natural_scale
        else:
            # The packed shape and scale size the native set_weight_scales admits.
            local_n, local_k = (pn // lanes, pk) if partition == 1 else (pn, pk // lanes)
            assert weight.shape == (pn, pk // 2) and scale.size(0) == 32
            assert scale.numel() == (math.ceil(local_k // 32 / 4) * math.ceil(local_n / 64)
                                     * lanes * 256)
            natural_scale = inverse_rpu_scale(scale, pn, pk, partition, lanes).double()
            values = (inverse_rpu_weight(weight, pn, pk, partition, lanes).double()
                      * natural_scale.repeat_interleave(32, dim=1))
        # MLP6 pads after logical quantization: zero values, unit scales.
        if role in ("gate", "up"):
            assert not values[n:].any() and bool((natural_scale[n:] == 1).all())
            values = values[:n]
        elif role == "down":
            assert not values[:, k:].any()
            if quantization == "w4a16":
                assert bool((natural_scale[:, k // 32:] == 1).all())
            values = values[:, :k]
        assert torch.equal(values, _reference(logical, quantization)), role
    # PLE and norms keep the FP16 payload of the unquantized profile.
    assert torch.equal(ple["ple_gate"][0], transform_linear_weight(
        names[prefix + "per_layer_input_gate.weight"].half(), 1, num_cores=topology.ple_tp))
    assert lists["input_norm"][0].dtype == torch.float16


# ------------------------------------------------------------------- native install


def _fake_lists(quantization, layers=2):
    """A quantized swizzle result: projections plus their own scale lists."""
    lists = {}
    for role in ROLES:
        if quantization == "w8a16":
            lists[role] = [torch.zeros(64, 64, dtype=torch.int8) for _ in range(layers)]
            lists[f"{role}_scale"] = [torch.ones(64, dtype=torch.float16) for _ in range(layers)]
        else:
            lists[role] = [torch.zeros(64, 32, dtype=torch.uint8) for _ in range(layers)]
            lists[f"{role}_scale"] = [torch.ones(32, 8, dtype=torch.float16)
                                      for _ in range(layers)]
    return lists


@pytest.fixture
def scale_calls(install_runtime, monkeypatch):
    calls = []

    def set_scales(handle, *scales):
        calls.append((handle, scales))
        install_runtime["events"].append(("set_scales", handle))

    def envelope(handle, max_kv_len, chunk):
        install_runtime["events"].append(("envelope", handle))

    monkeypatch.setattr(torch.ops.rpu, "gemma4_set_weight_scales", set_scales, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "gemma4_set_chunk_envelope", envelope, raising=False)
    return calls


def _install(monkeypatch, state, payload, quantization, cores):
    seen, checked = {}, []
    monkeypatch.setattr(runtime, "swizzle_decoder_and_ple",
                        lambda raw, geom, **kwargs: seen.update(kwargs) or (dict(payload), {}))
    expected = g4_topology.resolve_gemma4_topology(cores, quantization=quantization)
    _patch_topology_ops(monkeypatch, state, lambda handle: expected.native_readback())
    monkeypatch.setattr(gemma4, "_validate_gemma4_quant_source",
                        lambda _model, recipe: checked.append(recipe))
    adapter = gemma4.Gemma4Adapter(installed._Model(), quantization=quantization,
                                   rpu_execution={"model": {"num_cores": cores}})
    return adapter, seen, checked, expected


@pytest.mark.parametrize("quantization", ["w8a16", "w4a16", None])
@pytest.mark.parametrize("cores", [4, 6, 8])
def test_installer_hands_each_projection_scale_to_native(
        install_runtime, scale_calls, monkeypatch, quantization, cores):
    payload = {"q": []} if quantization is None else _fake_lists(quantization)
    adapter, seen, checked, expected = _install(
        monkeypatch, install_runtime, payload, quantization, cores)
    model = adapter.to_rpu()
    try:
        assert seen["quantization"] == quantization and seen["topology"] == expected
        assert model._rpu_gemma4_quantization == quantization
        assert model._rpu_gemma4_topology == expected
        events = [event[0] for event in install_runtime["events"]]
        keepalive = model._rpu_gemma4_retirement_state.keepalive
        if quantization is None:
            assert scale_calls == [] and checked == [] and keepalive[9] == ()
            return
        assert checked == [quantization, quantization]  # constructor, then install
        (handle, scales), = scale_calls
        assert handle == 1 and len(scales) == 7
        for role, role_scales in zip(ROLES, scales):
            assert role_scales is payload[f"{role}_scale"]
        assert all(a is b for a, b in zip(keepalive[9], scales, strict=True))
        assert not any(key.endswith("_scale") for key in keepalive[0])
        # After the weights and the topology readback, before the chunk envelope.
        assert events.index("set_weights") < events.index("set_scales") < events.index("envelope")
        if cores != 8:
            assert events.index("get_topology") < events.index("set_scales")
    finally:
        installed._finalize(model)


@pytest.mark.parametrize("fault", ["missing", "count", "shape", "dtype", "device",
                                   "strided", "storage"])
@pytest.mark.parametrize("quantization", ["w8a16", "w4a16"])
def test_installer_rejects_scale_payload_before_native(
        install_runtime, scale_calls, monkeypatch, quantization, fault):
    payload = _fake_lists(quantization)
    scales = payload["down_scale"]
    w4 = quantization == "w4a16"
    if fault == "missing":
        scales[1] = None
    elif fault == "count":
        scales.pop()
    elif fault == "shape":
        scales[1] = (torch.ones(16, 8, dtype=torch.float16) if w4
                     else torch.ones(80, dtype=torch.float16))
    elif fault == "dtype":
        scales[1] = scales[1].float()
    elif fault == "device":
        scales[1] = scales[1].to("meta")
    elif fault == "strided":
        scales[1] = (torch.ones(32, 16, dtype=torch.float16)[:, ::2] if w4
                     else torch.ones(64, 2, dtype=torch.float16)[:, 0])
    else:
        payload["down"][1] = payload["down"][1].view(torch.int8 if w4 else torch.uint8)
    adapter, _seen, _checked, _expected = _install(
        monkeypatch, install_runtime, payload, quantization, 6)
    with pytest.raises(ValueError, match="own FP16 scale"):
        adapter.to_rpu()
    assert install_runtime["events"] == [] and install_runtime["alive"] == set()
    assert install_runtime["releases"] == [adapter.model]
    assert scale_calls == [] and "_rpu_gemma4_handle" not in vars(adapter.model)


# ------------------------------------------------------------------ native contract


def _launches(body):
    """Top-level argument lists of every ACC16 Linear launch in ``body``."""
    needle = "rpu_launch_linear_spm_to_spm_acc16_kernel("
    calls, start = [], 0
    while (begin := body.find(needle, start)) >= 0:
        position = begin + len(needle)
        depth = 1
        while depth:
            depth += (body[position] == "(") - (body[position] == ")")
            position += 1
        calls.append([arg.strip() for arg in _split_top(body[begin + len(needle):position - 1])])
        start = position
    return calls


def test_native_scales_are_validated_whole_and_reach_every_decoder_launch():
    source = NATIVE.read_text()
    scales = _body(source, "void set_weight_scales(")
    for condition in ("get_last_resolved_chunk_size() == 0", "n_layers == 42",
                      "projection_type() != at::kHalf",
                      "static_cast<int64_t>(scales.size()) == n_layers",
                      "g.is_kv_shared && (projection == 1 || projection == 2)",
                      "scale.device().type() == at::kPrivateUse1",
                      "!scale.is_same(weight)", "scale.size(0) == 32",
                      "scale.dim() == 1 && scale.size(0) == n"):
        assert condition in scales, condition
    # Every pair is checked before the first scale is published.
    assert scales.rindex("TORCH_CHECK(") < scales.index(
        "w.q_s = rpu_retain_linear_quant_scale(w.q_w, q[layer], attention_lanes);") < scales.index(
        "invalidate_model_state();")
    weights = " ".join(_body(source, "void set_weights(").split())
    assert ("N == 42 && num_q_heads == 8 && num_kv_heads_max == 2 && head_dim_max == 512 && "
            "hidden_size == GEMMA4_E4B_HIDDEN && intermediate_size == GEMMA4_E4B_INTERMEDIATE"
            ) in weights
    assert "gemma4_projection_lane_granule(projection_type)" in weights
    assert "projection_type() == at::kHalf || layer_weights_.front().q_s.defined()" in _body(
        source, "at::Tensor forward(")
    layer = _strip_comments(_body(
        source, "void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk)"))
    launches = {args[1]: args for args in _launches(layer)}
    assert sorted(launches) == sorted([f"lw.{role}_w" for role in ROLES]
                                      + ["lw.ple_gate_w", "lw.ple_proj_w"])
    for role in ROLES:
        assert launches[f"lw.{role}_w"][-2:] == ["0", f"lw.{role}_s"], role
        assert f"&weights.{role}_s" in source  # graph identity covers the scales
    # PLE keeps its FP16 launches.
    assert len(launches["lw.ple_gate_w"]) == len(launches["lw.ple_proj_w"]) == 8


def test_native_and_python_share_the_storage_granules():
    source = NATIVE.read_text()
    assert "return type == at::kByte ? 64 : type == at::kChar ? 32 : 16;" in source
    assert g4_topology.MLP_LANE_GRANULE == {None: 16, "w8a16": 32, "w4a16": 64}
    header = (ROOT / "src/core/gemma4_execution_topology.h").read_text()
    assert "const int64_t alignment = 6 * lane_granule;" in header
    assert "gemma4_projection_lane_granule(projection_type())" in _body(
        source, "int64_t resolve_model_mlp_intermediate_size(")


def test_scale_op_is_declared_and_registered():
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    assert "void rpu_gemma4_set_weight_scales(" in decls
    assert 'm.def("gemma4_set_weight_scales(int handle, Tensor[] q_scales, ' in registrations
    assert ('m.impl("gemma4_set_weight_scales", '
            "TORCH_FN(rpu_gemma4_set_weight_scales));") in registrations
    assert ('m.impl("gemma4_set_weight_scales", '
            "torch::CppFunction::makeFallthrough());") in registrations
    assert ("->set_weight_scales(q_scales, k_scales, v_scales, o_scales, gate_scales, "
            "up_scales, down_scales);") in " ".join(NATIVE.read_text().split())
