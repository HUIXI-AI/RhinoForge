"""Exact 2B W8 core admission and real cold packing, without board tensors."""
import copy

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from transformers import Qwen3_5ForConditionalGeneration

from rpu_backend.adapters import qwen3_5
from rpu_backend.adapters.qwen3_5.cores import validate_core_profile
from rpu_backend.adapters.qwen3_5.quant_scope import W8A16, resolve_quant_specs
from rpu_backend.adapters.qwen3_5.text import _iter_text_projections
from test_qwen35_core_profiles import _config
from test_qwen35_mlp_cold_padding import NativeBoundaryReached, native_boundary


def _w8_config(size=2):
    config = _config(size)
    config.quant_config = {
        "format_version": 2, "architecture": "qwen3_5",
        "activation_dtype": "float16", "profile": "w8_main_w8_gdn",
        "default": {"method": "w8a16"},
        "projection_overrides": {
            "linear_attn.in_proj_b": {"method": "fp16"},
            "linear_attn.in_proj_a": {"method": "fp16"},
        },
    }
    return config


def _w8_model():
    config = _w8_config()
    with FakeTensorMode():
        model = Qwen3_5ForConditionalGeneration._from_config(config, dtype=torch.float16)
        is_full = [i % 4 == 3 for i in range(24)]
        specs = resolve_quant_specs(is_full, config.quant_config)
        for linear, name in _iter_text_projections(model.model.language_model, is_full):
            if specs[name].method == W8A16:
                linear.weight = torch.nn.Parameter(
                    torch.empty_like(linear.weight, dtype=torch.int8), requires_grad=False)
                linear.register_buffer("weight_scale", torch.empty(linear.weight.shape[0], dtype=torch.float16))
    model._qwen3_5_quant_config = config.quant_config
    return model


@pytest.mark.parametrize("cores", [4, 6])
def test_exact_w8_profile_uses_existing_role_topology_and_rejects_neighbours(cores):
    config = _w8_config()
    execution = {"model": {"num_cores": cores}}
    topology = validate_core_profile(config, execution)
    assert (topology.num_cores, topology.attn_tp, topology.mlp_tp) == (cores, 4, cores)
    for size in (0.8, 4, 9):
        assert validate_core_profile(_w8_config(size), execution).num_cores == cores
    for fault in ("w4_group64", "fp16_gdn", "width", "nested_quant"):
        bad = copy.deepcopy(config)
        if fault == "w4_group64":
            bad.quant_config["default"] = {"method": "w4a16_pgrp", "group_size": 64}
        elif fault == "fp16_gdn":
            bad.quant_config["projection_overrides"]["linear_attn.in_proj_z"] = {"method": "fp16"}
        elif fault == "width":
            bad.text_config.intermediate_size += 64
        else:
            bad.text_config.quant_config = bad.quant_config
        with pytest.raises(ValueError):
            validate_core_profile(bad, execution)


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("acc32", [False, True])
def test_public_w8_install_preserves_actual_dtype_scale_and_role_partition(native_boundary, monkeypatch, cores, acc32):
    model = _w8_model()
    inner = model.model.language_model
    adapter = qwen3_5.Qwen3_5Adapter(model, rpu_execution={
        "model": {"num_cores": cores}, "prefill": {"linear_acc32": acc32}})
    with inner.norm.weight.fake_mode, pytest.raises(NativeBoundaryReached):
        adapter.to_rpu(max_seq_len=64)
    b = native_boundary
    assert len(b.payloads) == 1 and b.padded == []
    args = b.payloads[0]
    assert args[17:22] == (8, 4, 256, 2048, 6144)
    assert ("cores", 35, cores) in b.events
    assert b.events.index(("acc32", 35, acc32)) < b.events.index("set_weights")
    for i in range(24):
        for wi, si in ((10, 48), (11, 49), (12, 50)):
            weight, scale = args[wi][i], args[si][i]
            assert weight.dtype == torch.int8 and weight.is_contiguous()
            assert scale.dtype == torch.float16 and scale.shape == (weight.shape[0],)
        if args[16][i]:
            for wi, si in ((1, 43), (2, 44), (3, 45), (4, 46), (7, 47)):
                assert args[wi][i].dtype == torch.int8
                assert args[si][i].shape == (args[wi][i].shape[0],)
            assert args[2][i].shape == args[3][i].shape == (1024, 2048)
        else:
            for wi, si in ((25, 51), (26, 52), (35, 53), (36, 54), (37, 55)):
                assert args[wi][i].dtype == torch.int8
                assert args[si][i].shape == (args[wi][i].shape[0],)
            for wi, si in ((41, 56), (42, 57)):
                assert args[wi][i].dtype == torch.float16 and args[wi][i].shape == (64, 2048)
                assert args[si][i].numel() == 0
            assert all(args[wi][i].shape == (4, 4, 512) for wi in (38, 39, 40))
    assert [p for p in b.packed if p[0] in ((6144, 2048), (2048, 6144))] == [
        p for _ in range(24) for p in (
            ((6144, 2048), 1, cores), ((6144, 2048), 1, cores), ((2048, 6144), 0, cores))]
    assert ((64, 2048), 1, 4) in b.packed
    assert b.events[-3:] == ["clear", "invalidate", ("destroy", 35)]


@pytest.mark.parametrize("fault", ["missing_scale", "short_scale", "fp16_weight", "int8_scalar", "noncontiguous", "already_packed"])
def test_bad_w8_storage_rejects_before_claim_and_irreversible_transform(native_boundary, monkeypatch, fault):
    model = _w8_model()
    inner = model.model.language_model
    with inner.norm.weight.fake_mode:
        last = inner.layers[-1].mlp.down_proj
        if fault == "missing_scale":
            del last.weight_scale
        elif fault == "short_scale":
            last.weight_scale = torch.empty(2047, dtype=torch.float16)
        elif fault == "fp16_weight":
            last.weight = torch.nn.Parameter(last.weight.half())
        elif fault == "int8_scalar":
            scalar = inner.layers[0].linear_attn.in_proj_b
            scalar.weight = torch.nn.Parameter(scalar.weight.to(torch.int8), requires_grad=False)
        elif fault == "already_packed":
            last._rpu_linear_partition = 0
        else:
            last.weight = torch.nn.Parameter(torch.empty(6144, 2048, dtype=torch.int8).t(), requires_grad=False)
        adapter = qwen3_5.Qwen3_5Adapter(model, rpu_execution={"model": {"num_cores": 6}})
        with pytest.raises(ValueError, match="raw quantized/FP16 storage|FP16 scale"):
            adapter.to_rpu(max_seq_len=64)
    assert native_boundary.events == native_boundary.packed == []
