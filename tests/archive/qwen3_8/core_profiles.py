"""Exact legacy W8 caller/packing boundaries; host checks are not model acceptance."""
from copy import deepcopy

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from rpu_backend.adapters import qwen3_5
from rpu_backend.adapters.qwen3_5 import quant_scope as qs
from rpu_backend.adapters.qwen3_5.cores import validate_core_profile, topology_tuple
from rpu_backend.adapters.qwen3_5.weights import (
    pad_gdn_scalar, pad_legacy_mlp_projection, pad_legacy_mlp_scale,
)
from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
from test_qwen35_mlp_cold_padding import native_boundary, NativeBoundaryReached


def _config():
    return Qwen3_5TextConfig(
        hidden_size=5120, intermediate_size=17408, num_hidden_layers=64,
        num_attention_heads=24, num_key_value_heads=4, head_dim=256,
        vocab_size=248320, linear_num_key_heads=16, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        layer_types=([qs.LINEAR_ATTENTION] * 3 + [qs.FULL_ATTENTION]) * 16,
        hidden_act="silu", use_cache=True, attention_bias=False, attn_output_gate=True,
        output_gate_type="swish", tie_word_embeddings=False, rms_norm_eps=1e-6,
        rope_parameters=dict(rope_type="default", rope_theta=10_000_000,
            partial_rotary_factor=0.25, mrope_interleaved=True, mrope_section=[11,11,10]),
        quant_config=qs.legacy_w8a16_quant_config())


def _model():
    config = _config()
    with FakeTensorMode():
        model = Qwen3_5ForCausalLM._from_config(config, dtype=torch.float16)
        model.half().requires_grad_(False)
        specs = qs.resolve_quant_specs(qs.is_full_from_layer_types(config.layer_types), config.quant_config)
        for name, spec in specs.items():
            if spec.method == qs.W8A16:
                projection = model.model.get_submodule(name)
                projection.weight = torch.nn.Parameter(
                    torch.empty_like(projection.weight, dtype=torch.int8), requires_grad=False)
                projection.register_buffer("weight_scale", torch.ones(projection.out_features, dtype=torch.float16))
    model._qwen3_5_quant_config = deepcopy(config.quant_config)
    model._qwen3_5_legacy_text_only = True
    return model


@pytest.mark.parametrize("cores", [4, 6])
def test_legacy_public_install_binds_quant_padding_gdn_and_graph_before_native(native_boundary, monkeypatch, cores):
    from rpu_backend.graph import GraphRuntimePolicy

    b = native_boundary
    monkeypatch.setenv("QWEN3_5_QUANT_ALLOW_UNCERTIFIED", "1")
    monkeypatch.setattr(torch.rpu, "get_caching_allocator", lambda: False)
    policies = []

    def prepare(policy):
        policies.append(policy)
        return True

    monkeypatch.setattr(GraphRuntimePolicy, "prepare_arenas", prepare)
    model = _model()
    inner = model.model
    owner = qwen3_5.Qwen3_5Adapter(model, rpu_execution={"model": {"num_cores": cores}})
    with inner.norm.weight.fake_mode, pytest.raises(NativeBoundaryReached):
        owner.to_rpu(max_seq_len=64)
    assert len(policies) == 1
    assert policies[0].execution_core_count == cores
    assert policies[0].qwen35_legacy_27b_sdk_budget
    assert policies[0].graph_arena_count == 0  # legacy policy owns its separate three-arena budget
    args, = b.payloads
    physical = 17472 if cores == 6 else 17408
    assert args[17:22] == (24, 24, 256, 5120, 17408)
    assert args[30:35] == (48, 128, 128, 10240, 4)
    assert len(args[16]) == 64 and sum(args[16]) == 16
    for i, full in enumerate(args[16]):
        for ix, sx, shape in ((10, 48, (physical, 5120)), (11, 49, (physical, 5120)), (12, 50, (5120, physical))):
            assert tuple(args[ix][i].shape) == shape
            assert args[ix][i].dtype == torch.int8
            assert tuple(args[sx][i].shape) == (shape[0],)
            assert args[sx][i].dtype == torch.float16
        if full:
            assert args[1][i].shape == args[2][i].shape == args[3][i].shape == (6144, 5120)
            assert all(args[ix][i].dtype == torch.int8 for ix in (1, 2, 3, 4, 7))
        else:
            assert args[27][i].shape == args[28][i].shape == (64,)
            assert args[35][i].shape == args[36][i].shape == (2048, 5120)
            assert args[37][i].shape == (6144, 5120)
            assert all(args[ix][i].dtype == torch.float16 for ix in (25,26,35,36,37,41,42))
            assert args[38][i].shape == args[39][i].shape == (4,4,512)
            assert args[40][i].shape == (4,4,1536)
    assert inner.config.intermediate_size == 17408
    assert all(layer.mlp.gate_proj.weight.shape == (17408,5120) for layer in inner.layers)
    assert b.padded == []  # legacy int8 never enters the shared FP16-only padding helper
    assert b.events[-3:] == ["clear", "invalidate", ("destroy", 35)]
    assert owner._execution_session.stats()["state"] == "CLOSED"


@pytest.mark.parametrize("fault", ["missing_scale", "gdn_int8", "last_scale_shape", "last_weight_float", "already_swizzled"])
def test_legacy_last_cold_leaf_failure_precedes_claim_and_packing(native_boundary, monkeypatch, fault):
    b = native_boundary
    monkeypatch.setenv("QWEN3_5_QUANT_ALLOW_UNCERTIFIED", "1")
    model = _model()
    owner = qwen3_5.Qwen3_5Adapter(model, rpu_execution={"model": {"num_cores": 6}})
    target = model.model.layers[-1].mlp.down_proj
    with target.weight.fake_mode:
        if fault == "missing_scale":
            del target._buffers["weight_scale"]
        elif fault == "last_scale_shape":
            target.weight_scale = torch.empty(5119, dtype=torch.float16)
        elif fault == "last_weight_float":
            target.weight = torch.nn.Parameter(target.weight.half(), requires_grad=False)
        elif fault == "gdn_int8":
            gdn = model.model.layers[-2].linear_attn.in_proj_z
            gdn.weight = torch.nn.Parameter(gdn.weight.to(torch.int8), requires_grad=False)
        else:
            target._rpu_linear_partition = 0
        with pytest.raises(ValueError, match="legacy Qwen3.8-27B"):
            owner.to_rpu(max_seq_len=64)
    assert b.events == b.packed == b.payloads == []
    assert not getattr(model, "_rpu_swizzle_started", False)


@pytest.mark.parametrize("cores", [4, 6])
def test_exact_legacy_profile_and_cache_keep_twentyfour_kv_heads(cores):
    from types import SimpleNamespace

    config = _config()
    topology = validate_core_profile(config, {"model": {"num_cores": cores}})
    assert topology_tuple(topology) == (cores,4,cores,4,8)
    model = SimpleNamespace(config=config, _rpu_execution={"model": {"num_cores": cores}},
        model=SimpleNamespace(_rpu_qwen3_5=SimpleNamespace(execution_topology=topology)))
    cache = Qwen3_5Cache.from_model(model, 2, device="cpu")
    assert cache.num_kv_heads == 24 and cache.attn_tp == 4 and cache.physical_kv_cores == 8
    assert cache.recurrent_states[0].shape == (48,128,128)
    assert cache.conv_states[0].shape == (4,4,2560)
    pointers = cache.recurrent_states[0].data_ptr(), cache.conv_states[0].data_ptr()
    cache.recurrent_states[0].fill_(1); cache.conv_states[0].fill_(1)
    cache.reset()
    assert pointers == (cache.recurrent_states[0].data_ptr(), cache.conv_states[0].data_ptr())
    assert not cache.recurrent_states[0].count_nonzero() and not cache.conv_states[0].count_nonzero()


@pytest.mark.parametrize("field,value", [("vocab_size",151936), ("use_cache",False),
    ("linear_num_value_heads",32), ("intermediate_size",17472), ("num_key_value_heads",24),
    ("quantization_config",{"bits":8}), ("num_experts",8)])
def test_reduced_legacy_semantic_neighbours_remain_rejected(field, value):
    config = _config(); setattr(config, field, value)
    with pytest.raises(ValueError):
        validate_core_profile(config, {"model": {"num_cores": 6}})


@pytest.mark.parametrize("heads,cores", [(16,4),(32,4),(48,4),(48,8)])
def test_scalar_scatter_rows_preserve_all_head_values_and_padding(heads, cores):
    raw = torch.arange(heads, dtype=torch.float16) + 1
    packed = pad_gdn_scalar(raw, num_heads=heads, num_cores=cores).reshape(cores,-1)
    local = heads // cores
    assert packed.shape[1] * packed.element_size() % 16 == 0
    # This is the native scalar DMA's per-core row/stride interpretation.
    assert torch.equal(packed[:, :local].reshape(-1), raw)
    assert torch.count_nonzero(packed[:, local:]) == 0
    assert packed.numel() == (64 if heads == 48 else cores * 8)


def test_int8_padding_and_unit_scales_preserve_dequantized_mlp_projection():
    # Complete real byte payload, examined by rows/slices to avoid huge FP32 copies.
    for name in ("gate_proj", "up_proj", "down_proj"):
        down = name == "down_proj"
        shape = (5120,17408) if down else (17408,5120)
        weight = torch.ones(shape, dtype=torch.int8)
        weight[0,0] = -17; weight[-1,-1] = 31
        scale = torch.linspace(.001,.02,shape[0],dtype=torch.float16)
        padded = pad_legacy_mlp_projection(weight, projection_name=name, logical_size=17408, physical_size=17472)
        padded_scale = pad_legacy_mlp_scale(scale, projection_name=name)
        assert torch.equal(padded[:shape[0],:shape[1]], weight)
        assert not (padded[:,17408:] if down else padded[17408:]).count_nonzero()
        assert torch.equal(padded_scale[:shape[0]], scale)
        if not down:
            assert torch.equal(padded_scale[17408:], torch.ones(64,dtype=torch.float16))
        selected = [0,shape[0]-1]
        logical = weight[selected].float() * scale[selected,None].float()
        physical = padded[selected].float() * padded_scale[selected,None].float()
        x = torch.linspace(-.1,.1,shape[1])
        physical_x = torch.nn.functional.pad(x,(0,64)) if down else x
        torch.testing.assert_close(physical @ physical_x, logical @ x, rtol=0, atol=1e-6)
        assert weight.shape == shape and scale.shape == (shape[0],)


def test_legacy_int8_preparation_does_not_admit_fp16_or_padded_source():
    from rpu_backend.runtime.weights import pad_decoder_mlp_projection

    with FakeTensorMode():
        raw = torch.empty(17408,5120,dtype=torch.int8)
        padded = pad_decoder_mlp_projection(
            raw, projection_name="gate_proj", logical_size=17408, physical_size=17472)
        assert padded.shape == (17472, 5120) and padded.dtype == torch.int8
        for invalid in (raw.half(), torch.empty(17472,5120,dtype=torch.int8)):
            with pytest.raises(ValueError, match="original contiguous"):
                pad_legacy_mlp_projection(invalid,projection_name="gate_proj",logical_size=17408,physical_size=17472)


@pytest.mark.parametrize("cores", [4,6])
def test_legacy_public_toml_selects_canonical_profile_and_rejects_substitution(monkeypatch, cores, tmp_path):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    common = importlib.import_module("examples.common.config")
    canonical = root / "examples/configs/qwen3_5/text/27b/w8a16_legacy_text.toml"
    selected = tmp_path / f"legacy_{cores}.toml"
    selected.write_text(canonical.read_text().replace(
        "num_cores = 8", f"num_cores = {cores}", 1))
    config = common.load_config(selected)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    assert config["rpu_execution"]["model"]["num_cores"] == cores
    assert profile["precision"] == "w8a16" and profile["asset_alias"] == "qwen3_8-27b-w8a16"
    for field, bad in (("id", "qwen3_5-27b.w8a16.text-controlled"),
                       ("asset_alias", "qwen3_5-27b"), ("component", "multimodal"),
                       ("precision", "fp16")):
        altered = deepcopy(profile); altered[field] = bad
        with pytest.raises(common.ConfigError):
            common._validate_execution(config["rpu_execution"], profile=altered)
